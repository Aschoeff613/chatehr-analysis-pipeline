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

---

## Your input file

A CSV with one query per row. Point `--text-column` at the column holding the
query. Any other columns you include are carried through, so ask the data team for
session ID, turn number, department, role and date, and you can break every
distribution down by them afterwards.

If you get conversation transcripts instead, add `--input-format conversation` and
the script pulls out the provider's turns.

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
| `run_summary.json` | Every setting used. Copy into your Methods. |
| `cache_*.jsonl` | Saved as it goes. If a run dies, restart and it picks up rather than paying twice. |

---

## The cognitive task list

`prompt_cognitive_deductive.txt` holds the final 12 Delphi-selected tasks, numbered
1-12 by composite rank. Diagnostic reasoning is task 1; Rapid acuity appraisal is
task 12. The earlier 17-task version is in git history if you need it for the
Methods.

Five constructs were excluded: managing uncertainty, judging credibility and
completeness, team and distributed cognition, metacognitive self-regulation, and
encounter scoping. Queries driven by those are labeled `task_id` 0 with
`cognitive_task` set to an exact `"Outside the 12: <construct>"` string, so each one
counts as its own row in `distribution_cognitive.csv` rather than being absorbed
into a surviving task. Billing lookups and software troubleshooting still get
`"System operation, not clinical cognition"`.

Document generation is handled separately, and not as a cognitive task. The medical
layer already records which document was asked for, so on the cognitive layer a bare
"generate the ED note" is labeled `"Document production, not clinical cognition"`.
The 17-task version routed documentation to team and distributed cognition on the
argument that a note transmits clinical state across providers; that does not hold
against the construct as defined, which requires trust, delegation or responsibility
to be at issue. A document request that specifies what to include and why goes to
task 3, since the provider did the selection.

Expect the excluded and non-cognitive rows together to be a large share of the
corpus.

If you edit the task list again, three things have to move together: the task
definitions, every `task N` cross-reference in the guardrails and boundary lines,
and the `1-12` range in the output format at the bottom. Nothing in `pipeline.py`
needs to change -- it reads whatever is in the prompt file.

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
misclassification. In the 12-task numbering those are task 3 and task 7. The third
member of that old trio -- trusting the information -- is now outside the 12, so
also check that credibility queries are landing at 0 rather than being quietly
pulled into 3 or 7.

**Reruns reuse old labels, silently.** `cache_*.jsonl` is keyed on query text alone
and records nothing about which prompt or model produced the answer. Rerunning into
an output directory from before the 12-task switch will hand back the old 17-task
labels and print "labeling 0 new queries" as though all is well. Use a fresh
`--outdir` whenever the prompt or the model changes.

## What this does not include

The paper's second classifier, which sorts queries by phrasing (summarization,
question answering, extraction, and so on). Your design uses the medical layer plus
the cognitive layer instead. The published version of that prompt also contains a
mislabeled example, which is a second reason to leave it out.
