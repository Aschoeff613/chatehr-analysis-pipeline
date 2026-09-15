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

## Updating the cognitive task list to the final 12

The task definitions live in `prompt_cognitive_deductive.txt`, which currently holds
17. When the final 12 are ready, edit that file. Three things have to change
together, not just the list:

1. Delete the five dropped tasks and renumber the rest.
2. Fix every cross-reference. The guardrails and the boundary lines point at tasks
   by number ("this is task 3, not task 7") throughout. Those numbers all move.
3. Update the output format at the bottom, which says 1-17.

Nothing in `pipeline.py` needs to change. It reads whatever is in the prompt file.

---

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

**Watch the 3 / 7 / 8 confusion.** The cognitive prompt says so itself: getting
information, trusting it, and interpreting it absorb most of the corpus and most of
the misclassification. When you review the 100, check that confusion specifically
rather than just the overall agreement rate.

---

## What this does not include

The paper's second classifier, which sorts queries by phrasing (summarization,
question answering, extraction, and so on). Your design uses the medical layer plus
the cognitive layer instead. The published version of that prompt also contains a
mislabeled example, which is a second reason to leave it out.
