"""
ChatEHR query classification pipeline.

Raw ChatEHR queries go in. Medical labels and cognitive annotations come out:

  MEDICAL layer (the what) - replicates Appendix C.1 of Shah et al.,
    "Adoption and Use of LLMs at an Academic Medical Center"
    (arXiv:2602.00074). Prompt follows the published appendix.

  COGNITIVE layer (the why) - the PACT cognitive classifier, run in two
    passes: a deductive pass that assesses fit to defined tasks, and an
    independent inductive pass that lets the model name what it sees without
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

# Temperature 0 reduces sampling variability; it does not guarantee determinism.
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
FIT_STATUSES = ("matched", "no_adequate_fit", "insufficient_context", "nonclinical")
STATUS_LABELS = {
    "no_adequate_fit": "(no adequate taxonomy fit)",
    "insufficient_context": "(insufficient context)",
    "nonclinical": "(nonclinical)",
    "invalid_response": UNLABELED,
}
VALIDATION_VERSION = 1

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


def validate_reply(reply: str, label: str, prompt_text: str) -> dict:
    """Reject invalid annotations before they enter the successful cache."""
    if not isinstance(reply, str):
        raise ValueError("Expected reply text")
    if label == "medical":
        if not reply.strip():
            raise ValueError("Empty medical label")
        return {}
    data = parse_json_reply(reply)
    if not isinstance(data, dict) or not data or "_parse_error" in data:
        raise ValueError("Expected a JSON object")
    if label == "cognitive":
        required = {"fit_status", "task_id", "cognitive_task", "inference_depth", "rationale", "memo"}
        if not required <= data.keys():
            raise ValueError("Missing cognitive fields")
        status = data["fit_status"]
        if status not in FIT_STATUSES:
            raise ValueError("Invalid fit status")
        catalog = {int(i): name.strip() for i, name in
                   re.findall(r"^## (\d+)\. (.+)$", prompt_text, re.M)}
        if status == "matched":
            task_id = data["task_id"]
            if type(task_id) is not int or task_id not in catalog:
                raise ValueError("Invalid task ID")
            if data["cognitive_task"] != catalog[task_id]:
                raise ValueError("Task ID/name mismatch")
        elif data["task_id"] is not None or data["cognitive_task"] is not None:
            raise ValueError("Unmatched requests cannot receive a task")
        if status in ("matched", "no_adequate_fit"):
            if data["inference_depth"] not in ("surface", "one_step"):
                raise ValueError("Invalid inference depth")
        elif data["inference_depth"] is not None:
            raise ValueError("Unknown/nonclinical cognition requires null depth")
        if not isinstance(data["rationale"], str) or not data["rationale"].strip():
            raise ValueError("Missing rationale")
        if data["memo"] is not None and not isinstance(data["memo"], str):
            raise ValueError("Invalid memo")
    elif label == "inductive":
        required = {"cognitive_intent", "secondary_intent", "clinical_task", "failure_mode",
                    "non_cognitive", "insufficient_context"}
        if not required <= data.keys():
            raise ValueError("Missing inductive fields")
        for field in ("non_cognitive", "insufficient_context"):
            if type(data[field]) is not bool:
                raise ValueError("Expected boolean flags")
        if data["non_cognitive"] and data["insufficient_context"]:
            raise ValueError("Conflicting inductive flags")
        for field in ("cognitive_intent", "secondary_intent", "failure_mode"):
            value = data[field]
            if value is not None and (not isinstance(value, str) or not value.strip()
                                      or value.strip().lower() == "null"):
                raise ValueError("Expected text or JSON null")
        if not isinstance(data["clinical_task"], str) or not data["clinical_task"].strip():
            raise ValueError("Missing clinical task description")
        excluded = data["non_cognitive"] or data["insufficient_context"]
        if excluded and (data["cognitive_intent"] is not None or data["secondary_intent"] is not None):
            raise ValueError("Excluded requests cannot receive intents")
        if not excluded and data["cognitive_intent"] is None:
            raise ValueError("Missing primary intent")
    else:
        raise ValueError("Unknown layer")
    return data


def cognitive_results(queries, cache, prompt_text):
    """Build nullable task columns and explicit statuses, including technical failures."""
    rows = []
    for query in queries:
        try:
            data = validate_reply(cache.get(query, ""), "cognitive", prompt_text)
        except ValueError:
            data = {"fit_status": "invalid_response"}
        status = data["fit_status"]
        rows.append({
            "cognitive_fit_status": status,
            "cognitive_task_id": data.get("task_id"),
            "cognitive_task": data.get("cognitive_task"),
            "cognitive_catalog_match": status == "matched" if status in ("matched", "no_adequate_fit") else None,
            "cognitive_outcome": data.get("cognitive_task") if status == "matched" else STATUS_LABELS[status],
            "inference_depth": data.get("inference_depth"),
            "cognitive_rationale": data.get("rationale"),
            "cognitive_memo": data.get("memo"),
        })
    result = pd.DataFrame(rows)
    result["cognitive_task_id"] = result["cognitive_task_id"].astype("Int64")
    result["cognitive_catalog_match"] = result["cognitive_catalog_match"].astype("boolean")
    return result


def cognitive_summary(statuses):
    counts = statuses.value_counts().to_dict()
    assessable = counts.get("matched", 0) + counts.get("no_adequate_fit", 0)
    return {
        "cognitive_status_counts": {key: int(counts.get(key, 0)) for key in (*FIT_STATUSES, "invalid_response")},
        "cognitive_assessable_clinical_queries": assessable,
        "cognitive_taxonomy_coverage_pct": round(100 * counts.get("matched", 0) / assessable, 1) if assessable else None,
        "cognitive_coverage_denominator": "matched + no_adequate_fit; excludes insufficient_context, nonclinical, invalid_response",
        "cognitive_unlabeled": int(counts.get("invalid_response", 0)),
    }


def call_one(client, prompt_template: str, query: str, retries: int = 5, validator=None) -> str:
    prompt = prompt_template.replace("{USER_QUERY}", query)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            reply = (resp.choices[0].message.content or "").strip()
            if validator is not None:
                validator(reply)
            return reply
        except Exception as exc:  # API errors and invalid annotations
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            print(f"  retrying in {wait}s after {type(exc).__name__}", file=sys.stderr)
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
                if not isinstance(rec, dict):
                    stale += 1
                    continue
                if (rec.get("model") != MODEL or rec.get("prompt_sha") != fingerprint
                        or rec.get("temperature") != TEMPERATURE
                        or rec.get("validation_version") != VALIDATION_VERSION):
                    stale += 1
                    continue
                try:
                    if not isinstance(rec.get("query"), str):
                        raise ValueError("Invalid cache query")
                    validate_reply(rec.get("reply", ""), label, prompt_template)
                except (ValueError, TypeError):
                    stale += 1
                    continue
                cache[rec["query"]] = rec["reply"]
        print(f"[{label}] reusing {len(cache)} answers from a previous run.")
        if stale:
            print(
                f"[{label}] ignoring {stale} cached answers produced by a "
                f"different settings/schema or invalid replies; they will be relabeled."
            )

    todo = [q for q in unique_queries(queries) if q not in cache]
    print(f"[{label}] labeling {len(todo)} new queries with {MODEL}...")

    failures: list[tuple[str, str]] = []
    if todo:
        with cache_path.open("a") as fh, ThreadPoolExecutor(CONCURRENCY) as pool:
            futures = {
                pool.submit(call_one, client, prompt_template, q,
                            validator=lambda r: validate_reply(r, label, prompt_template)): q
                for q in todo
            }
            for n, fut in enumerate(as_completed(futures), start=1):
                q = futures[fut]
                try:
                    reply = fut.result()
                except Exception as exc:
                    # One unrecoverable query must not abandon the whole layer.
                    # Failures are not cached, so a rerun retries just these.
                    failures.append((q, type(exc).__name__))
                    continue
                cache[q] = reply
                fh.write(
                    json.dumps({
                        "query": q,
                        "reply": reply,
                        "model": MODEL,
                        "prompt_sha": fingerprint,
                        "temperature": TEMPERATURE,
                        "validation_version": VALIDATION_VERSION,
                    })
                    + "\n"
                )
                fh.flush()
                if n % 50 == 0 or n == len(todo):
                    print(f"  [{label}] {n}/{len(todo)}")

    if failures:
        print(
            f"WARNING: [{label}] {len(failures)} of {len(todo)} queries could "
            f"not be labeled after retries. Those rows retain missing labels and are "
            f"counted in run_summary.json. Rerun to retry only these.",
            file=sys.stderr,
        )
        for q, err in failures[:5]:
            print(f"  - {err}: no valid annotation saved", file=sys.stderr)

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
    if not labels:
        return [], 0
    if len(set(labels)) == 1:
        return labels.copy(), 1
    from sklearn.cluster import KMeans

    n_unique = len(set(labels))
    k_used = min(k, n_unique, len(labels))
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
        default="medical,cognitive,inductive",
        help="Comma-separated: medical, cognitive, inductive. "
        "Default runs all three passes.",
    )
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD)
    ap.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip embeddings/clustering; write raw-label distributions and annotations.",
    )
    args = ap.parse_args()

    layers = {s.strip() for s in args.layers.split(",") if s.strip()}
    unknown = layers - {"medical", "cognitive", "inductive"}
    if unknown:
        sys.exit(f"Unknown layer(s): {sorted(unknown)}")

    if not layers:
        sys.exit("Select at least one layer")
    if args.k < 1 or not 0 <= args.merge_threshold <= 2:
        sys.exit("--k must be positive; --merge-threshold must be between 0 and 2")

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
        cognitive = cognitive_results(queries, cache, PROMPT_COGNITIVE.read_text())
        df = pd.concat([df, cognitive], axis=1)
        print("Cognitive fit statuses:")
        print(df["cognitive_fit_status"].value_counts().to_string())

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

        df["inductive_insufficient_context"] = df["query"].map(
            lambda q: parsed.get(q, {}).get("insufficient_context")
        )
        df["inductive_status"] = df["query"].map(
            lambda q: "invalid_response" if q not in parsed else
            "nonclinical" if parsed[q]["non_cognitive"] else
            "insufficient_context" if parsed[q]["insufficient_context"] else "identified"
        )

    # ---- Run settings. Written whether or not clustering runs, so a run is
    # traceable to its recorded settings.
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
        summary.update(cognitive_summary(df["cognitive_fit_status"]))
    if "inductive" in layers:
        summary["inductive_status_counts"] = {
            status: int((df["inductive_status"] == status).sum())
            for status in ("identified", "nonclinical", "insufficient_context", "invalid_response")
        }
    summary["skip_clustering"] = args.skip_clustering
    summary["validation_version"] = VALIDATION_VERSION

    # Preserve completed annotations even if optional grouping later fails.
    df.to_csv(args.outdir / "queries_labeled.csv", index=False)
    (args.outdir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    def group_valid(source, target, valid, prefix):
        df[target] = df[source].where(valid)
        if not args.skip_clustering and valid.any():
            labels = df.loc[valid, source].tolist()
            if len(set(labels)) == 1:
                groups, used = labels, 1
            else:
                groups, used = cluster(embed(labels), labels, args.k, args.merge_threshold)
            df.loc[valid, target] = groups
            summary[f"{prefix}_k_used"] = used
            summary[f"{prefix}_groups_after_merge"] = len(set(groups))

    if "medical" in layers:
        valid = df["medical_task_raw"].notna() & df["medical_task_raw"].ne("")
        group_valid("medical_task_raw", "medical_task_group", valid, "medical")
        summary["medical_unlabeled"] = int((~valid).sum())
    if "inductive" in layers:
        valid = df["inductive_status"].eq("identified")
        group_valid("cognitive_intent", "inductive_group", valid, "inductive")
        df.loc[~valid, "inductive_group"] = df.loc[~valid, "inductive_status"].map(STATUS_LABELS)

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
        cog = distribution("cognitive_outcome", "distribution_cognitive.csv")
        distribution("cognitive_fit_status", "distribution_cognitive_fit.csv")
    if "inductive" in layers:
        distribution("inductive_group", "distribution_inductive.csv")

    # The payoff: what medical work, crossed with what cognitive work.
    if "medical" in layers and "cognitive" in layers:
        ct = pd.crosstab(
            df["medical_task_group"].fillna(UNLABELED),
            df["cognitive_outcome"].fillna(UNLABELED),
        )
        ct.to_csv(args.outdir / "crosstab_medical_x_cognitive.csv")
        written.append("crosstab_medical_x_cognitive.csv")

    # Exploratory comparison; theme-to-taxonomy mapping requires human review.
    if "inductive" in layers and "cognitive" in layers:
        ct = pd.crosstab(
            df["inductive_group"].fillna(UNLABELED),
            df["cognitive_outcome"].fillna(UNLABELED),
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

