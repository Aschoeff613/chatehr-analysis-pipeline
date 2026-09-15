# ChatEHR query classification pipeline

Classify provider requests along two axes: medical work requested and cognitive
purpose inferred from the text. The cognitive axis uses independent deductive and
inductive calls. These annotations describe requests, not directly observed
clinician reasoning, model execution, or response quality.

## Setup and running

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
export CHATEHR_MODEL="gpt-4.1"
python pipeline.py your_file.csv --text-column query --outdir output
```

The default runs **medical, cognitive (deductive), and inductive** passes. Induction
runs on every request, including deductive matches, and never receives deductive
labels or task definitions. This adds an API call per unique request compared with
the earlier two-pass default.

```bash
# Inspect annotations without embedding or clustering; distributions still written
python pipeline.py your_file.csv --skip-clustering --outdir output_pilot

# Select specific passes
python pipeline.py your_file.csv --layers medical,cognitive

# Legacy transcript export
python pipeline.py sample_input.csv --input-format conversation \
    --text-column conversation --outdir output_test

python -m pytest tests/ -q
```

Input is CSV or JSON with one request per row. Extra columns are carried through:
include source platform/model, session ID, department, role, and date when available.
Conversation mode extracts user turns using speaker-label heuristics. It does not
retain preceding context for classification, so an ambiguous follow-up is judged on
its own text alone and will often not map. Audit transcript extraction before a study, since
clinical headings can resemble speaker labels. Prefer structured exports with one
request per row. Repeated identical request text reuses the same annotation.

## Deductive classification: assess the existing taxonomy

The prompt defines the final 12 Delphi-selected tasks, numbered 1-12. A request gets
a task only when one of the definitions adequately fits the primary cognitive need
its text actually shows. Nothing is forced to the closest task.

| cognitive_fit_status | Meaning | Task ID/name |
|---|---|---|
| matched | One definition adequately fits | Defined ID and name |
| does_not_map | It does not | Empty |
| invalid_response | No valid reply after retries | Empty |

Everything that is not a match lands in one bucket, whatever the reason: no
definition fits, the request is too thin to show which need is driving it, or it is
not clinical work at all. The rationale says which of those it was, in plain words,
so a reviewer can read any row and see why. That keeps the reason available without
turning it into a category to be counted, and keeps the headline number simple.

A bare clinical note request does not map, since "generate the ED note" does not by
itself show a cognitive need. Explicit selection of relevant clinical information
can support task 3.

`cognitive_outcome` is the column to group and cross-tabulate on: the task name for
matched rows, a readable bucket label otherwise. `cognitive_task` and
`cognitive_task_id` are empty for anything that did not map.

**Taxonomy coverage** is matched over every request that got a valid reply. Only
technical failures are outside the denominator, and the summary records the
denominator and all three counts. Coverage is null when nothing was classified.

Coverage is a classifier estimate until humans check it. Sample from
`queries_labeled.csv` and have reviewers judge both the assigned tasks and a sample
of the does-not-map bucket -- the second is the one that decides your coverage
figure, so it is worth checking that the model is not parking genuine matches there.

## Inductive classification: describe the observed operations

The separate prompt describes cognitive_intent without the task list. It also
records secondary_intent, clinical_task, failure_mode, non_cognitive, and
insufficient_context. Nonclinical requests, missing context, and technical failures
remain separate in the output and are excluded from embedding/clustering.

Only primary cognitive_intent enters clustering. Secondary intent is retained for
qualitative review, not counted in the primary-task distribution. Neither pass
estimates the prevalence of all operations within compound requests.

Inductive themes do **not automatically identify taxonomy gaps**. Reviewers should
examine themes without deductive labels, then map them to the frozen taxonomy.
A theme may be a synonym, a narrower instance of a task, or a genuinely uncovered
need. The inductive-by-deductive cross-tab is exploratory comparison, not independent
validation. Both calls use the same classifier model and may share biases.

## Outputs

| File | Contents |
|---|---|
| queries_labeled.csv | Requests, original metadata, annotations, explicit statuses |
| distribution_medical.csv | Medical label/group counts |
| distribution_cognitive.csv | Matched task counts plus separate unassigned/error outcomes |
| distribution_cognitive_fit.csv | All five deductive statuses, with percentages of all requests |
| distribution_inductive.csv | Primary intent/group counts plus separate nonclinical/context/error outcomes (the inductive pass keeps its own outcomes; it has no task list) |
| crosstab_medical_x_cognitive.csv | Medical work versus deductive outcomes |
| crosstab_inductive_x_cognitive.csv | Exploratory theme/intent versus deductive outcomes |
| run_summary.json | Settings, prompt hashes, counts, coverage denominator, grouping details |
| cache_*.jsonl | Validated successful replies, keyed by request and run settings |

Only files for selected layers are written. With --skip-clustering, group columns
contain raw labels/intents and all selected distributions and cross-tabs are still
written. Use a new output directory for each analysis configuration to avoid stale
files from prior layer selections. Row-level annotations and an initial summary
are saved before optional clustering so completed work survives grouping failures.

## Validation, caching, and reproducibility

Cognitive replies must have all required fields, valid statuses, integer task IDs,
matching canonical names, and consistent null/task fields. Inductive replies must
have boolean flags and consistent intent fields. Invalid replies are retried up to
five attempts and are not cached as successes. Technical failures retain rows and
are counted separately. Medical labels are checked for nonempty text, not semantic
correctness. Validate medical label quality separately.

Cache reuse requires matching model, prompt hash, temperature, and validation
version, plus a valid response under the current schema. Old caches are skipped;
the first run after this migration may relabel previously cached requests. Error
logs omit request snippets. Temperature zero reduces variability but does not
guarantee deterministic classifications.

The medical prompt follows Appendix C.1 of Shah et al., *Adoption and Use of LLMs
at an Academic Medical Center* ([paper](https://arxiv.org/abs/2602.00074)). Its
inherited dry-mouth/drug-interaction example remains a known limitation in this
change. The medical layer uses all-mpnet-base-v2 embeddings and K-means with a
maximum of 1,000 initial clusters. One distinct label needs no embedding or K-means.
Missing labels are excluded from clustering. The 0.15 cosine-distance merge
threshold is an assumption, not a parameter reported by the paper. Merges are
transitive and can combine distinct labels; review cluster membership and assess
sensitivity before interpreting grouped distributions.

## Migration from earlier versions

Earlier revisions assigned a closest task and recorded a `cognitive_forced_fit`
column, with `cognitive_natural_fit_pct` and `cognitive_forced_fit_pct` in the
summary. A later revision split non-matches into `no_adequate_fit`,
`insufficient_context` and `nonclinical`, and carried a derived
`cognitive_catalog_match` column. All of those are gone. There are now two
classification outcomes, `matched` and `does_not_map`, plus `invalid_response` for
technical failures.

Nonclinical requests no longer use task ID 0; anything unassigned has a null ID and
name. Scripts expecting complete task columns, the old statuses, or the old coverage
denominator need updating. Any prompt change invalidates cached annotations, which
the cache detects on its own.

## Before interpreting a study

- Obtain taxonomy-owner approval of the definitions and classification rules.
- Develop guidelines on a diverse sample; freeze prompts and test on a separate
  sample with two independent clinician reviewers and adjudication.
- Validate task identity and fit status, including rare tasks and ambiguous cases.
- Keep sessions together across development/validation splits; distinguish
  request-level from encounter-level prevalence and evaluate each source platform.
- Review inductive themes before exposing deductive labels, then map the themes.

Raw request text is sent to the configured API and retained in plaintext caches and
CSV outputs. Instructions to omit PHI from generated labels do not de-identify the
inputs. Use the institutionally approved endpoint and storage workflow for actual
encounters. The paper's medical-label agreement does not validate this cognitive
classifier or a new population.
