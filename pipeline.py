"""
ChatEHR query classification pipeline.

Raw ChatEHR queries go in. Two labels come out for each one:

  MEDICAL layer (the what) - replicates Appendix C.1 of Shah et al.,
    "Adoption and Use of LLMs at an Academic Medical Center"
    (arXiv:2602.00074). Prompt is word for word from the paper.

  COGNITIVE layer (the why) - the PACT cognitive classifier, run in two
    passes: a deductive pass that maps each query to a defined task, and an
    optional inductive pass that lets the model name what it sees without
    ever being shown the task list.

Run `python pipeline.py --help` for options, or read README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Settings you may want to change
# --------------------------------------------------------------------------

# Which model does the labeling.
# The paper used "gpt-4.1". Change this to whatever model you are using.
MODEL = os.environ.get("CHATEHR_MODEL", "gpt-4.1")

# Temperature 0 so the same query always gets the same label.
# The paper does not state a temperature. Note this choice in your Methods.
TEMPERATURE = 0.0

# The embedding model named in the paper. Do not change this if you want
# your clustering to match theirs.
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Number of K-means clusters. The paper used 1000 across ~23,000 sessions.
# On a smaller set of queries 1000 is far too many, so the script scales it
# down automatically and tells you when it does.
DEFAULT_K = 1000

# Clusters whose centers are closer together than this get merged into one.
# The paper says they did this but never gives the number. ASK MIGUEL FUENTES
# FOR THE REAL VALUE. Until then this is a guess and you must report whatever
# you used.
DEFAULT_MERGE_THRESHOLD = 0.15

# How many labeling calls to run at once.
CONCURRENCY = 8

# Rows whose reply could not be parsed are reported under this label rather
# than dropped, so every distribution sums to the number of queries.
UNLABELED = "(unlabeled: no valid reply)"

HERE = Path(__file__).parent
PROMPT_MEDICAL = HERE / "prompt_c1_medical_task_normalization.txt"
PROMPT_COGNITIVE = HERE / "prompt_cognitive_deductive.txt"
PROMPT_INDUCTIVE = HERE / "prompt_cognitive_inductive.txt"


# --------------------------------------------------------------------------
# Step 1: get one query per row
# --------------------------------------------------------------------------

# Single letters are deliberately NOT markers here. A line starting "Q:" or
# "A:" inside a clinical answer ("Q: waves present in III") would otherwise be
# read as a speaker change, which both truncates the real provider turn and
# invents a provider query out of the model's own text.
USER_MARKERS = r"(?:user|provider|clinician|physician|doctor|human)"
ASSISTANT_MARKERS = r"(?:assistant|chatehr|ai|model|response)"
TURN_RE = re.compile(
    rf"^\s*({USER_MARKERS}|{ASSISTANT_MARKERS})\s*[:\-]\s*",
    re.IGNORECASE | re.MULTILINE,
)
IS_USER_RE = re.compile(rf"^{USER_MARKERS}$", re.IGNORECASE)

# Bare Q/A transcripts are still supported, but only when the whole cell is
# written that way: no named speakers anywhere, and at least two Q/A lines.
QA_TURN_RE = re.compile(r"^\s*(q|a)\s*[:\-]\s*", re.IGNORECASE | re.MULTILINE)


def split_conversation(text: str) -> list[str]:
    """Return just the provider's messages from one back-and-forth transcript.

    Only needed if your export gives you whole conversations. If it already
    has one query per row, use --input-format queries and this is skipped.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    def named_speaker_is_user(speaker: str) -> bool:
        return bool(IS_USER_RE.match(speaker))

    def qa_speaker_is_user(speaker: str) -> bool:
        return speaker.lower() == "q"

    matches = list(TURN_RE.finditer(text))
    is_user = named_speaker_is_user

    if not matches:
        qa = list(QA_TURN_RE.finditer(text))
        if len(qa) >= 2:
            matches, is_user = qa, qa_speaker_is_user
        else:
            # No speaker markers at all. Treat the whole cell as one query;
            # load_queries counts these and warns.
            return [text.strip()]

    turns = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body and is_user(m.group(1)):
            turns.append(body)
    return turns


def unique_queries(queries: list[str]) -> list[str]:
    """Distinct queries, first-seen order. Identical queries cost one call."""
    return list(dict.fromkeys(queries))


def distribution_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Counts and percentages for one label column, most common first.

    Rows whose reply could not be parsed are reported as UNLABELED rather than
    dropped: groupby discards null keys, which used to delete them while the
    percentage was still divided by the full row count, so the numbers
    under-reported and did not sum to 100.
    """
    return (
        df[col].fillna(UNLABELED)
        .value_counts(dropna=False)
        .rename("n_queries")
        .rename_axis(col)
        .reset_index()
        .assign(pct_of_queries=lambda x: 100 * x["n_queries"] / len(df))
    )


def prompt_fingerprint(path: Path) -> str:
    """Short hash of a prompt file, so a run can be tied to the exact wording."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def git_commit() -> str | None:
    """Short commit of this checkout, or None outside a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def load_queries(path: Path, input_format: str, text_column: str) -> pd.DataFrame:
    """Read the input file and return one row per provider query."""
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_json(path)

    if text_column not in df.columns:
        sys.exit(
            f"Column '{text_column}' not found. Columns in your file: "
            f"{list(df.columns)}"
        )

    # Keep any extra columns (department, role, date) so you can break the
    # distribution down by them later.
    passthrough = [c for c in df.columns if c != text_column]

    reserved = {"source_row", "turn", "query"} & set(passthrough)
    if reserved:
        print(
            f"NOTE: input column(s) {sorted(reserved)} clash with names this "
            f"script adds; yours are kept as orig_<name>."
        )

    rows = []
    unsplit = 0
    for idx, row in df.iterrows():
        if input_format == "queries":
            queries = [str(row[text_column]).strip()]
        else:
            raw = row[text_column]
            queries = split_conversation(raw)
            if (
                len(queries) == 1
                and isinstance(raw, str)
                and queries[0] == raw.strip()
            ):
                unsplit += 1

        for turn_no, q in enumerate(queries, start=1):
            if not q or q.lower() == "nan":
                continue
            record = {
                (f"orig_{c}" if c in reserved else c): row[c] for c in passthrough
            }
            record.update({"source_row": idx, "turn": turn_no, "query": q})
            rows.append(record)

    if unsplit:
        print(
            f"WARNING: {unsplit} of {len(df)} rows had no recognizable speaker "
            f"markers and were each treated as a single query. If your export "
            f"uses a different speaker label, add it to USER_MARKERS."
        )

    out = pd.DataFrame(rows)
    if out.empty:
        sys.exit("No queries found. Check --input-format and --text-column.")
    return out


# --------------------------------------------------------------------------
# Step 2: labeling (one AI call per query per layer)
# --------------------------------------------------------------------------


def load_catalog(prompt_text: str) -> list[str]:
    """Pull the 121 predefined task names out of the medical prompt file.

    Used afterwards to tell whether the model matched the catalog or wrote a
    new label. That split is the coverage number.
    """
    block = prompt_text.split("# Predefined Task Catalog", 1)[1]
    block = block.split("# Instructions", 1)[0]

    items, current = [], []
    for line in block.splitlines():
        if line.startswith("- "):
            if current:
                items.append(" ".join(current))
            current = [line[2:].strip()]
        elif current and line.strip() and not line.startswith("#"):
            current.append(line.strip())
    if current:
        items.append(" ".join(current))
    return items


def parse_json_reply(text: str) -> dict:
    """Read the model's JSON answer, tolerating code fences and stray prose."""
    if not text:
        return {}
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"_parse_error": text}


def call_one(client, prompt_template: str, query: str, retries: int = 5) -> str:
    prompt = prompt_template.replace("{USER_QUERY}", query)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # rate limits, transient network errors
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            print(f"  retrying in {wait}s after: {exc}", file=sys.stderr)
            time.sleep(wait)
    return ""


def run_layer(queries: list[str], prompt_path: Path, cache_path: Path, label: str
              ) -> dict[str, str]:
    """Send every unique query through one prompt, saving answers as it goes.

    Each layer is a separate, independent call. The inductive pass therefore
    never sees the cognitive task list, which is the point of running it.
    """
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Run: pip install -r requirements.txt")

    client = OpenAI()
    prompt_template = prompt_path.read_text()
    fingerprint = prompt_fingerprint(prompt_path)

    # Cached answers record the model and prompt version that produced them.
    # Anything labeled under a different model or an edited prompt is ignored
    # rather than silently reused.
    cache: dict[str, str] = {}
    stale = 0
    if cache_path.exists():
        with cache_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated last line from a hard kill
                if rec.get("model") != MODEL or rec.get("prompt_sha") != fingerprint:
                    stale += 1
                    continue
                cache[rec["query"]] = rec["reply"]
        print(f"[{label}] reusing {len(cache)} answers from a previous run.")
        if stale:
            print(
                f"[{label}] ignoring {stale} cached answers produced by a "
                f"different model or prompt version; they will be relabeled."
            )

    todo = [q for q in unique_queries(queries) if q not in cache]
    print(f"[{label}] labeling {len(todo)} new queries with {MODEL}...")

    failures: list[tuple[str, str]] = []
    if todo:
        with cache_path.open("a") as fh, ThreadPoolExecutor(CONCURRENCY) as pool:
            futures = {
                pool.submit(call_one, client, prompt_template, q): q for q in todo
            }
            for n, fut in enumerate(as_completed(futures), start=1):
                q = futures[fut]
                try:
                    reply = fut.result()
                except Exception as exc:
                    # One unrecoverable query must not abandon the whole layer.
                    # Failures are not cached, so a rerun retries just these.
                    failures.append((q, repr(exc)))
                    continue
                cache[q] = reply
                fh.write(
                    json.dumps({
                        "query": q,
                        "reply": reply,
                        "model": MODEL,
                        "prompt_sha": fingerprint,
                    })
                    + "\n"
                )
                fh.flush()
                if n % 50 == 0 or n == len(todo):
                    print(f"  [{label}] {n}/{len(todo)}")

    if failures:
        print(
            f"WARNING: [{label}] {len(failures)} of {len(todo)} queries could "
            f"not be labeled after retries. Those rows will be blank, and are "
            f"counted in run_summary.json. Rerun to retry only these.",
            file=sys.stderr,
        )
        for q, err in failures[:5]:
            print(f"  - {err}: {q[:80]}", file=sys.stderr)

    return cache


# --------------------------------------------------------------------------
# Step 3: embed and cluster
# --------------------------------------------------------------------------

_EMBEDDER = None


def embed(labels: list[str]) -> np.ndarray:
    """Turn each label into numbers that capture its meaning.

    Labels meaning similar things get similar numbers, which is what lets the
    next step group them.
    """
    global _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Run: pip install -r requirements.txt")

    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)

    unique = list(dict.fromkeys(labels))
    print(f"Embedding {len(unique)} unique labels...")
    vecs = _EMBEDDER.encode(unique, show_progress_bar=True, normalize_embeddings=True)
    lookup = {lab: vec for lab, vec in zip(unique, vecs)}
    return np.vstack([lookup[lab] for lab in labels])


def cluster(vectors: np.ndarray, labels: list[str], k: int, merge_threshold: float):
    """Group similar labels, name each group, merge near-duplicate groups."""
    from sklearn.cluster import KMeans

    n_unique = len(set(labels))
    k_used = min(k, max(2, n_unique))
    if k_used != k:
        print(
            f"NOTE: asked for {k} clusters but only {n_unique} distinct labels "
            f"exist, so using {k_used}. The paper's 1000 was for ~23,000 "
            f"sessions. Report the number you actually used."
        )

    print(f"Clustering {len(labels)} labels into {k_used} groups...")
    km = KMeans(n_clusters=k_used, random_state=0, n_init=10).fit(vectors)

    # Name each group with the label sitting closest to its center.
    names = {}
    for cid in range(k_used):
        members = np.where(km.labels_ == cid)[0]
        if len(members) == 0:
            continue
        center = km.cluster_centers_[cid]
        dists = np.linalg.norm(vectors[members] - center, axis=1)
        names[cid] = labels[members[int(np.argmin(dists))]]

    # Merge groups whose centers sit very close together.
    centers = km.cluster_centers_
    centers_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    distance = 1 - centers_norm @ centers_norm.T

    parent = list(range(k_used))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(k_used):
        for j in range(i + 1, k_used):
            if distance[i, j] < merge_threshold:
                union(i, j)

    sizes = Counter(km.labels_.tolist())
    grouped = {}
    for cid in range(k_used):
        grouped.setdefault(find(cid), []).append(cid)

    final_names = {}
    for root, members in grouped.items():
        biggest = max(members, key=lambda c: sizes.get(c, 0))
        for c in members:
            final_names[c] = names.get(biggest, names.get(c, "unlabeled"))

    print(f"  {k_used} groups before merging, {len(set(final_names.values()))} after.")
    return [final_names[c] for c in km.labels_], k_used


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="CSV or JSON file of queries")
    ap.add_argument("--outdir", type=Path, default=Path("output"))
    ap.add_argument(
        "--input-format",
        choices=["conversation", "queries"],
        default="queries",
        help="'queries' = one query per row (the usual case). "
        "'conversation' = raw transcripts to be split into provider turns.",
    )
    ap.add_argument("--text-column", default="query")
    ap.add_argument(
        "--layers",
        default="medical,cognitive",
        help="Comma-separated: medical, cognitive, inductive. "
        "Default runs medical and cognitive.",
    )
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD)
    ap.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Stop after labeling. Useful for a first look.",
    )
    args = ap.parse_args()

    layers = {s.strip() for s in args.layers.split(",") if s.strip()}
    unknown = layers - {"medical", "cognitive", "inductive"}
    if unknown:
        sys.exit(f"Unknown layer(s): {sorted(unknown)}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1
    df = load_queries(args.input, args.input_format, args.text_column)
    print(f"Found {len(df)} queries across {df['source_row'].nunique()} rows.\n")
    queries = df["query"].tolist()

    # ---- Step 2: medical layer
    if "medical" in layers:
        cache = run_layer(
            queries, PROMPT_MEDICAL,
            args.outdir / "cache_medical.jsonl", "medical",
        )
        df["medical_task_raw"] = df["query"].map(cache)

        catalog = set(load_catalog(PROMPT_MEDICAL.read_text()))
        df["medical_catalog_match"] = df["medical_task_raw"].isin(catalog)
        cov = df["medical_catalog_match"].mean()
        print(
            f"\nMedical catalog coverage: {cov:.1%} of queries matched one of "
            f"the {len(catalog)} predefined tasks.\n"
        )

    # ---- Step 2: cognitive layer (deductive)
    if "cognitive" in layers:
        cache = run_layer(
            queries, PROMPT_COGNITIVE,
            args.outdir / "cache_cognitive.jsonl", "cognitive",
        )
        parsed = {q: parse_json_reply(r) for q, r in cache.items()}
        # Keep this numeric: the model returns "3" as a string, which makes
        # every `df.cognitive_task_id == 3` filter silently False.
        df["cognitive_task_id"] = pd.to_numeric(
            df["query"].map(lambda q: parsed.get(q, {}).get("task_id")),
            errors="coerce",
        ).astype("Int64")
        df["cognitive_task"] = df["query"].map(
            lambda q: parsed.get(q, {}).get("cognitive_task")
        )
        df["cognitive_catalog_match"] = df["query"].map(
            lambda q: parsed.get(q, {}).get("catalog_match")
        )
        df["inference_depth"] = df["query"].map(
            lambda q: parsed.get(q, {}).get("inference_depth")
        )
        df["cognitive_rationale"] = df["query"].map(
            lambda q: parsed.get(q, {}).get("rationale")
        )
        df["cognitive_memo"] = df["query"].map(
            lambda q: parsed.get(q, {}).get("memo")
        )

        n_bad = sum(1 for v in parsed.values() if "_parse_error" in v)
        if n_bad:
            print(f"WARNING: {n_bad} cognitive answers were not valid JSON.")

        memo = df["cognitive_memo"].fillna("").astype(str).str.strip().str.lower()
        df["cognitive_forced_fit"] = memo.str.startswith("forced fit:")

        matched = df["cognitive_catalog_match"].fillna(False).mean()
        print(
            f"\nCognitive coverage: {matched:.1%} of queries had a driver that "
            f"is genuinely one of the 12.\n"
            f"Forced fits: {df['cognitive_forced_fit'].mean():.1%} were assigned "
            f"a task the prompt marked as the closest available rather than a "
            f"true match.\n"
        )

    # ---- Step 2: cognitive layer (inductive, blind to the task list)
    if "inductive" in layers:
        cache = run_layer(
            queries, PROMPT_INDUCTIVE,
            args.outdir / "cache_inductive.jsonl", "inductive",
        )
        parsed = {q: parse_json_reply(r) for q, r in cache.items()}
        df["cognitive_intent"] = df["query"].map(
            lambda q: parsed.get(q, {}).get("cognitive_intent")
        )
        df["secondary_intent"] = df["query"].map(
            lambda q: parsed.get(q, {}).get("secondary_intent")
        )
        df["inductive_clinical_task"] = df["query"].map(
            lambda q: parsed.get(q, {}).get("clinical_task")
        )
        df["failure_mode"] = df["query"].map(
            lambda q: parsed.get(q, {}).get("failure_mode")
        )
        df["non_cognitive"] = df["query"].map(
            lambda q: parsed.get(q, {}).get("non_cognitive")
        )

    # ---- Run settings. Written whether or not clustering runs, so a run is
    # always reproducible and two runs can be compared field by field.
    prompt_files = {
        "medical": PROMPT_MEDICAL,
        "cognitive": PROMPT_COGNITIVE,
        "inductive": PROMPT_INDUCTIVE,
    }
    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "model": MODEL,
        "temperature": TEMPERATURE,
        "embedding_model": EMBEDDING_MODEL,
        "k_requested": args.k,
        "merge_threshold": args.merge_threshold,
        "layers": sorted(layers),
        "input_format": args.input_format,
        "n_queries": int(len(df)),
        # The task list lives in the prompt file, so the prompt hash is the
        # only thing that identifies which taxonomy a run used.
        "prompt_versions": {
            name: {"file": path.name, "sha256_12": prompt_fingerprint(path)}
            for name, path in prompt_files.items()
            if name in layers
        },
    }
    if "medical" in layers:
        summary["medical_catalog_coverage_pct"] = round(
            100 * df["medical_catalog_match"].mean(), 1
        )
    if "cognitive" in layers:
        summary["cognitive_natural_fit_pct"] = round(
            100 * df["cognitive_catalog_match"].fillna(False).mean(), 1
        )
        summary["cognitive_forced_fit_pct"] = round(
            100 * df["cognitive_forced_fit"].mean(), 1
        )
        summary["cognitive_unlabeled"] = int(df["cognitive_task"].isna().sum())

    if args.skip_clustering:
        df.to_csv(args.outdir / "queries_labeled.csv", index=False)
        (args.outdir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Wrote {args.outdir / 'queries_labeled.csv'} and run_summary.json")
        return

    # ---- Step 3: grouping
    if "medical" in layers:
        vecs = embed(df["medical_task_raw"].tolist())
        df["medical_task_group"], k_used = cluster(
            vecs, df["medical_task_raw"].tolist(), args.k, args.merge_threshold
        )
        summary["medical_k_used"] = k_used
        summary["medical_groups_after_merge"] = int(df["medical_task_group"].nunique())

    if "inductive" in layers:
        intents = df["cognitive_intent"].fillna("(none)").astype(str).tolist()
        vecs = embed(intents)
        df["inductive_group"], k_used = cluster(
            vecs, intents, args.k, args.merge_threshold
        )
        summary["inductive_k_used"] = k_used
        summary["inductive_groups_after_merge"] = int(df["inductive_group"].nunique())

    # ---- Step 4: write everything out
    df.to_csv(args.outdir / "queries_labeled.csv", index=False)
    written = ["queries_labeled.csv"]

    def distribution(col: str, fname: str):
        d = distribution_table(df, col)
        d.to_csv(args.outdir / fname, index=False)
        written.append(fname)
        return d

    if "medical" in layers:
        med = distribution("medical_task_group", "distribution_medical.csv")
    if "cognitive" in layers:
        cog = distribution("cognitive_task", "distribution_cognitive.csv")
    if "inductive" in layers:
        distribution("inductive_group", "distribution_inductive.csv")

    # The payoff: what medical work, crossed with what cognitive work.
    if "medical" in layers and "cognitive" in layers:
        ct = pd.crosstab(
            df["medical_task_group"].fillna(UNLABELED),
            df["cognitive_task"].fillna(UNLABELED),
        )
        ct.to_csv(args.outdir / "crosstab_medical_x_cognitive.csv")
        written.append("crosstab_medical_x_cognitive.csv")

    # Does the blind pass rediscover the defined tasks?
    if "inductive" in layers and "cognitive" in layers:
        ct = pd.crosstab(
            df["inductive_group"].fillna(UNLABELED),
            df["cognitive_task"].fillna(UNLABELED),
        )
        ct.to_csv(args.outdir / "crosstab_inductive_x_cognitive.csv")
        written.append("crosstab_inductive_x_cognitive.csv")

    (args.outdir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    written.append("run_summary.json")

    print(f"\nWrote to {args.outdir}/")
    for f in written:
        print(f"  {f}")

    if "medical" in layers:
        print("\nTop 10 medical tasks:")
        print(med.head(10).to_string(index=False))
    if "cognitive" in layers:
        print("\nCognitive task distribution:")
        print(cog.to_string(index=False))


if __name__ == "__main__":
    main()
