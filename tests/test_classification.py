"""Offline regression tests for fit status, cache safety and pipeline outputs."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as p


def cognitive(status="matched", **changes):
    data = dict(fit_status=status, task_id=3 if status == "matched" else None,
                cognitive_task="Directed information gathering & sufficiency" if status == "matched" else None,
                inference_depth="surface",
                rationale="Synthetic annotation for testing.", memo=None)
    data.update(changes)
    return json.dumps(data)


def inductive(status="identified"):
    return json.dumps(dict(cognitive_intent="Gather relevant information" if status == "identified" else None,
                           secondary_intent=None, clinical_task="chart review", failure_mode=None,
                           non_cognitive=status == "nonclinical", insufficient_context=status == "insufficient_context"))


@pytest.mark.parametrize("status", p.FIT_STATUSES)
def test_valid_statuses(status):
    result = p.validate_reply(cognitive(status), "cognitive", p.PROMPT_COGNITIVE.read_text())
    assert (result["task_id"] is not None) == (status == "matched")


@pytest.mark.parametrize("reply", ["[]", "null", "true", "1", "{}", "not json", "",
    cognitive(task_id=99), cognitive(task_id=True), cognitive(task_id="3"),
    cognitive(cognitive_task="Invented name"), cognitive("does_not_map", task_id=3),
    cognitive("does_not_map", cognitive_task="Directed information gathering & sufficiency"),
    cognitive("does_not_map", inference_depth="guessing"), cognitive(rationale=""),
    cognitive("forced_fit"), cognitive("no_adequate_fit"), cognitive("nonclinical")])
def test_invalid_cognitive_replies(reply):
    with pytest.raises(ValueError):
        p.validate_reply(reply, "cognitive", p.PROMPT_COGNITIVE.read_text())


@pytest.mark.parametrize("status", ["identified", "nonclinical", "insufficient_context"])
def test_inductive_statuses(status):
    p.validate_reply(inductive(status), "inductive", p.PROMPT_INDUCTIVE.read_text())


def test_inductive_rejects_string_booleans_and_conflicting_intents():
    for change in ({"non_cognitive": "false"}, {"insufficient_context": True},
                   {"cognitive_intent": "null"}, {"non_cognitive": True, "insufficient_context": True}):
        data=json.loads(inductive()); data.update(change)
        with pytest.raises(ValueError):
            p.validate_reply(json.dumps(data), "inductive", p.PROMPT_INDUCTIVE.read_text())


def test_only_matched_rows_carry_a_task():
    statuses = list(p.FIT_STATUSES)
    cache = {status: cognitive(status, memo="arbitrary review note") for status in statuses}
    df = p.cognitive_results(statuses + ["failed"], cache, p.PROMPT_COGNITIVE.read_text())
    assert df.cognitive_task_id.notna().sum() == 1
    assert df.cognitive_task.notna().sum() == 1
    # matched name, the does-not-map bucket, and the technical-failure bucket
    assert len(set(df.cognitive_outcome)) == 3


def test_coverage_counts_every_query_that_got_an_answer():
    statuses = ["matched", "matched", "matched", "does_not_map", "failed"]
    cache = {"matched": cognitive(), "does_not_map": cognitive("does_not_map")}
    df = p.cognitive_results(statuses, cache, p.PROMPT_COGNITIVE.read_text())
    summary = p.cognitive_summary(df.cognitive_fit_status)
    # 3 matched of 4 that were classified; the technical failure is excluded
    assert summary["cognitive_classified_queries"] == 4
    assert summary["cognitive_taxonomy_coverage_pct"] == 75.0
    assert summary["cognitive_unlabeled"] == 1


def test_coverage_is_null_when_nothing_was_classified():
    assert p.cognitive_summary(pd.Series(["invalid_response"]))[
        "cognitive_taxonomy_coverage_pct"
    ] is None


def test_a_query_that_does_not_map_may_still_report_depth():
    text = p.PROMPT_COGNITIVE.read_text()
    for depth in ("surface", "one_step", None):
        out = p.validate_reply(cognitive("does_not_map", inference_depth=depth), "cognitive", text)
        assert out["task_id"] is None


def fake_client(replies):
    queue=iter(replies)
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(queue)))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def test_invalid_output_retried(monkeypatch):
    client,calls=fake_client(["[]", cognitive()])
    monkeypatch.setattr(p.time,"sleep",lambda _: None)
    result=p.call_one(client,p.PROMPT_COGNITIVE.read_text(),"synthetic",validator=lambda r:
                      p.validate_reply(r,"cognitive",p.PROMPT_COGNITIVE.read_text()))
    assert result == cognitive()
    assert len(calls) == 2


def test_invalid_cache_and_failed_attempts_not_reused(tmp_path,monkeypatch):
    client,calls=fake_client(["[]"]*5 + [cognitive()])
    module=ModuleType("openai"); module.OpenAI=lambda: client
    monkeypatch.setitem(sys.modules,"openai",module)
    monkeypatch.setattr(p.time,"sleep",lambda _: None)
    cache=tmp_path/'cache.jsonl'
    cache.write_text(json.dumps(dict(query="synthetic",reply="[]",model=p.MODEL,
        prompt_sha=p.prompt_fingerprint(p.PROMPT_COGNITIVE),temperature=p.TEMPERATURE,
        validation_version=p.VALIDATION_VERSION))+"\n")
    assert p.run_layer(["synthetic"],p.PROMPT_COGNITIVE,cache,"cognitive") == {}
    assert len(cache.read_text().splitlines()) == 1
    assert p.run_layer(["synthetic"],p.PROMPT_COGNITIVE,cache,"cognitive")["synthetic"] == cognitive()
    assert p.run_layer(["synthetic"],p.PROMPT_COGNITIVE,cache,"cognitive")["synthetic"] == cognitive()
    assert len(calls) == 6


@pytest.mark.parametrize("skip", [True, False])
def test_full_pipeline_status_counts_and_independent_induction(tmp_path,monkeypatch,skip):
    statuses=list(p.FIT_STATUSES)+["failed"]
    source=tmp_path/'input.csv'; pd.DataFrame({"query":statuses}).to_csv(source,index=False)
    calls=[]
    def layer(queries,prompt,cache,label):
        calls.append((label,queries.copy()))
        if label == "cognitive":
            return {q:cognitive(q) for q in statuses[:-1]}
        if label == "medical":
            return {q:"Review history" for q in statuses[:-1]}
        return {"matched":inductive(),"does_not_map":inductive("nonclinical")}
    monkeypatch.setattr(p,"run_layer",layer)
    monkeypatch.setattr(p,"embed",lambda labels: pytest.fail("One unique valid label needs no embeddings"))
    dest=tmp_path/'output'
    monkeypatch.setattr(sys,"argv",["pipeline.py",str(source),"--outdir",str(dest)]+(["--skip-clustering"] if skip else []))
    p.main()
    assert calls == [(label,statuses) for label in ("medical","cognitive","inductive")]
    df=pd.read_csv(dest/'queries_labeled.csv')
    assert df.cognitive_task.notna().sum() == 1
    for name in ("medical","cognitive","cognitive_fit","inductive"):
        d=pd.read_csv(dest/f'distribution_{name}.csv')
        assert d.n_queries.sum() == len(statuses)
        assert d.pct_of_queries.sum() == pytest.approx(100)
    cross=pd.read_csv(dest/'crosstab_inductive_x_cognitive.csv',index_col=0)
    assert cross.to_numpy().sum() == len(statuses)
    summary=json.loads((dest/'run_summary.json').read_text())
    assert summary["cognitive_taxonomy_coverage_pct"] == 50
    assert summary["cognitive_status_counts"] == {"matched":1,"does_not_map":1,"invalid_response":1}
    # the inductive pass keeps its own outcomes, independent of the 12
    assert summary["inductive_status_counts"]["nonclinical"] == 1
    assert summary["inductive_status_counts"]["invalid_response"] == 1
    assert summary["medical_unlabeled"] == 1


def test_cluster_empty_and_single_without_sklearn():
    assert p.cluster(np.empty((0,2)),[],1000,.15) == ([],0)
    assert p.cluster(np.ones((1,2)),["one"],1000,.15) == (["one"],1)


def test_prompts_have_no_forced_routing_or_inductive_task_list():
    assert "forced fit" not in p.PROMPT_COGNITIVE.read_text().lower()
    assert "12-task" not in p.PROMPT_INDUCTIVE.read_text()


def test_all_failed_layers_complete_without_embedding(tmp_path, monkeypatch):
    source=tmp_path/'input.csv'
    source.write_text('query\nSynthetic request\n')
    dest=tmp_path/'output'
    monkeypatch.setattr(p,'run_layer',lambda *args: {})
    monkeypatch.setattr(p,'embed',lambda _: pytest.fail('No valid labels to embed'))
    monkeypatch.setattr(sys,'argv',['pipeline.py',str(source),'--outdir',str(dest)])
    p.main()
    for name in ('medical','cognitive','inductive'):
        table=pd.read_csv(dest/f'distribution_{name}.csv')
        assert table.n_queries.sum() == 1
        assert p.UNLABELED in table.iloc[:,0].tolist()
    summary=json.loads((dest/'run_summary.json').read_text())
    assert summary['cognitive_taxonomy_coverage_pct'] is None
