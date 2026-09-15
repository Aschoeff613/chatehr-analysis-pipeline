# ChatEHR query classification pipeline

ChatEHR queries go in. Each one comes out with two labels: what medical work was
asked for, and what thinking the request was serving.

**Medical layer (the what).** Copies Appendix C.1 of Shah et al., *Adoption and Use
of LLMs at an Academic Medical Center* (arXiv:2602.00074). The prompt is word for
word from the paper.

**Cognitive layer (the why).** The PACT cognitive classifier, run two ways. A
deductive pass that maps each query to one of the defined tasks, and an optional
inductive pass that asks the model to name what it sees without ever showing it the
task list, so you can check whether the defined tasks turn up on their own.

The two cognitive passes are separate calls. The inductive one never sees the task
list. That is the whole point of running it, and it is why they cannot be combined
into one call to save money.

---

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
export CHATEHR_MODEL="gpt-4.1"    # or whichever model you are using
```

---

## Running it

Check it works on the sample file first:

```bash
python pipeline.py sample_input.csv --input-format conversation \
    --text-column conversation --outdir output_test
```

Your real data, one query per row:

```bash
python pipeline.py your_file.csv --text-column query --outdir output
```

Pick which layers to run:

```bash
# both cognitive passes as well
python pipeline.py your_file.csv --layers medical,cognitive,inductive

# medical only
python pipeline.py your_file.csv --layers medical
```

Look at the labels before committing to the grouping step:

```bash
python pipeline.py your_file.csv --skip-clustering
```

Run the tests after any change to the splitter, the parsing, or the task list:

```bash
python -m pytest tests/ -q
```

---

## Your input file

A CSV with one query per row. Point `--text-column` at the column holding the
query. Any other columns you include are carried through, so ask the data team for
session ID, turn number, department, role and date, and you can break every
distribution down by them afterwards.

If you get conversation transcripts instead, add `--input-format conversation` and
the script pulls out the provider's turns. It recognizes `User:`, `Provider:`,
`Clinician:`, `Physician:`, `Doctor:` and `Human:` for the provider, and
`Assistant:`, `ChatEHR:`, `AI:`, `Model:` and `Response:` for the model, with `-`
accepted in place of `:`. Bare `Q:`/`A:` transcripts work too, but only when the
whole cell is written that way -- a stray `A:` line inside an answer will not split
a turn. Any row with no recognizable speaker label is kept whole as a single query
and counted in a warning, so you will see it rather than losing turns quietly.

---

## What you get

| File | What it is |
|---|---|
| `queries_labeled.csv` | Every query with all its labels. This is what you sample from for the clinician review. |
| `distribution_medical.csv` | Medical task breakdown. |
| `distribution_cognitive.csv` | Cognitive task breakdown. |
| `distribution_inductive.csv` | The tasks the blind pass came up with on its own. |
| `crosstab_medical_x_cognitive.csv` | Medical work against cognitive work. This is the two-layer result. |
| `crosstab_inductive_x_cognitive.csv` | Whether the blind pass rediscovered the defined tasks. |
| `run_summary.json` | Every setting used, including the prompt hash, the git commit, and the natural-fit and forced-fit rates. Copy into your Methods. |
| `cache_*.jsonl` | Saved as it goes. If a run dies, restart and it picks up rather than paying twice. |

---

## The cognitive task list

`prompt_cognitive_deductive.txt` holds the final 12 Delphi-selected tasks, numbered
1-12 by composite rank. Diagnostic reasoning is task 1; Rapid acuity appraisal is
task 12. The earlier 17-task version is in git history if you need it for the
Methods.

Every clinical query is assigned one of the 12. The five constructs the Delphi
dropped -- managing uncertainty, judging credibility and completeness, team and
distributed cognition, metacognitive self-regulation, and encounter scoping -- no
longer exist as labels. A query driven by one of them is assigned the closest of the
12, `catalog_match` is set to false, and the memo begins `forced fit: <driver>`. The
routing table is in the prompt under "Forced fits".

That gives you two numbers rather than one. `cognitive_natural_fit_pct` in
`run_summary.json` is the share whose driver is genuinely one of the 12;
`cognitive_forced_fit_pct` is the share the 12 do not natively cover. The forced-fit
rate is the finding, not a defect -- it is how much of real ChatEHR use sits outside
the final taxonomy. `queries_labeled.csv` carries a `cognitive_forced_fit` column so
you can pull those rows for the clinician review.

Document generation routes to task 3: what the request drives is deciding what
belongs in the record, and the medical layer already records which document was
asked for. When the provider specifies what to include and why, that selection is a
genuine task 3. When the request is bare ("generate the ED note"), it is task 3 as a
forced fit. Naming a recipient does not change the task. The 17-task version routed
documentation to team and distributed cognition on the argument that a note
transmits clinical state across providers and time; that does not hold against the
construct as defined, which requires trust, delegation or responsibility to be at
issue.

Billing lookups, software troubleshooting and work queue mechanics are the only
queries with no task at all. They get `task_id` 0 and
`"System operation, not clinical cognition"`.

If you edit the task list again, three things have to move together: the task
definitions, every `task N` cross-reference in the guardrails and boundary lines,
and the `1-12` range in the output format at the bottom. Nothing in `pipeline.py`
needs to change -- it reads whatever is in the prompt file, and `run_summary.json`
records the prompt hash so a run is always traceable to the wording it used.
`tests/test_pipeline.py` fails if a cross-reference points at a task that no longer
exists.

## Things to be aware of

**The cluster merge threshold is a guess.** The paper says they merged clusters that
sat close together but never says how close. The default here is 0.15. Ask Miguel
Fuentes for the real value. It changes how many final tasks you end up with, so
whatever you use has to go in your Methods.

**The number of groups scales down automatically.** The paper used 1000 across
roughly 23,000 sessions. On a smaller set that would put nearly every query in its
own group, so the script reduces it and says so. Report the number it prints, not
1000.

**You still need the clinician check.** The paper validated by taking 100 random
queries and having two clinicians say whether the assigned label was appropriate.
They got 73.5% using gpt-4.1. If you use a different model you cannot lean on that
number. Run the same check on both layers, with different reviewers for each, as
they did. `queries_labeled.csv` is what you sample from.

**Watch the 3 vs 7 confusion.** The cognitive prompt says so itself: gathering
information and interpreting it absorb most of the corpus and most of the
misclassification. In the 12-task numbering those are task 3 and task 7. Task 3 also
now receives four separate forced-fit drivers, so check it specifically during the
clinician review -- filter on `cognitive_forced_fit` and read the memos.

**Reruns no longer reuse labels across a prompt change.** Each line in
`cache_*.jsonl` records the model and the prompt hash that produced it. Cached
answers from a different model or an edited prompt are ignored and relabeled, and
the script says how many it skipped. You can safely rerun into an existing output
directory.

**A query that cannot be labeled no longer stops the run.** After five failed
attempts that query is skipped, the run finishes, and the count appears in
`run_summary.json` as `cognitive_unlabeled`. Failures are not cached, so rerunning
retries only those. Unlabeled rows are reported as
`(unlabeled: no valid reply)` in the distributions rather than dropped, so the
percentages still sum to 100.

## What this does not include

The paper's second classifier, which sorts queries by phrasing (summarization,
question answering, extraction, and so on). Your design uses the medical layer plus
the cognitive layer instead. The published version of that prompt also contains a
mislabeled example, which is a second reason to leave it out.
