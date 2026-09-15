"""Tests for the parts that quietly corrupt data when they break:
transcript splitting, reply parsing, dedupe, and the distribution counts."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline as p  # noqa: E402


# ---------------------------------------------------------------- splitting

def test_splits_named_speakers():
    text = (
        "User: Summarize her cardiac history\n"
        "Assistant: She has...\n"
        "User: Any stress tests?\n"
        "Assistant: Yes..."
    )
    assert p.split_conversation(text) == [
        "Summarize her cardiac history",
        "Any stress tests?",
    ]


def test_lettered_options_inside_a_query_are_not_a_speaker_change():
    """"A:" on its own line used to end the turn and discard the rest."""
    text = "User: Which should I start,\nA: metoprolol or\nB: carvedilol?"
    turns = p.split_conversation(text)
    assert len(turns) == 1
    assert "metoprolol" in turns[0] and "carvedilol" in turns[0]


def test_q_wave_in_an_answer_does_not_become_a_provider_query():
    """"Q: waves present in III" used to be extracted as a provider turn."""
    text = (
        "User: Read this ECG\n"
        "Assistant: Findings:\n"
        "Q: waves present in III\n"
        "User: So is it old?"
    )
    assert p.split_conversation(text) == ["Read this ECG", "So is it old?"]


def test_bare_qa_transcripts_still_split():
    text = "Q: give me a differential\nA: consider...\nQ: and the next step?"
    assert p.split_conversation(text) == [
        "give me a differential",
        "and the next step?",
    ]


def test_dash_separator_and_case_insensitive():
    assert p.split_conversation("provider - differential please\nai - consider...") == [
        "differential please"
    ]


@pytest.mark.parametrize("text", ["", "   ", None, 3.5])
def test_empty_and_non_string_input(text):
    assert p.split_conversation(text) == []


def test_unmarked_text_is_kept_as_one_query():
    assert p.split_conversation("summarize the chart for back pain") == [
        "summarize the chart for back pain"
    ]


# ------------------------------------------------------------ reply parsing

def test_parses_fenced_json():
    assert p.parse_json_reply('```json\n{"task_id": "3"}\n```') == {"task_id": "3"}


def test_parses_json_wrapped_in_prose():
    assert p.parse_json_reply('Sure! {"task_id": "7"} hope that helps') == {
        "task_id": "7"
    }


def test_unparseable_reply_is_flagged_not_lost():
    out = p.parse_json_reply("Generate discharge summaries")
    assert out["_parse_error"] == "Generate discharge summaries"


def test_empty_reply():
    assert p.parse_json_reply("") == {}


# -------------------------------------------------------- medical catalog

def test_catalog_has_121_tasks():
    text = p.PROMPT_MEDICAL.read_text()
    assert len(p.load_catalog(text)) == 121


def test_catalog_rejoins_wrapped_entries():
    catalog = p.load_catalog(p.PROMPT_MEDICAL.read_text())
    assert "Interpret functional diagnostic tests (ECG, spirometry, stress tests)" in catalog


# -------------------------------------------------------------- dedupe

def test_dedupe_keeps_first_seen_order():
    assert p.unique_queries(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


# --------------------------------------------------------- distributions

def test_unlabeled_rows_are_counted_not_dropped():
    df = pd.DataFrame({"cognitive_task": ["Diagnostic reasoning"] * 2 + [None] * 2})
    d = p.distribution_table(df, "cognitive_task")
    assert d["n_queries"].sum() == len(df)
    assert d["pct_of_queries"].sum() == pytest.approx(100.0)
    assert p.UNLABELED in set(d["cognitive_task"])


def test_distribution_is_ordered_most_common_first():
    df = pd.DataFrame({"t": ["a", "b", "b", "c", "b"]})
    d = p.distribution_table(df, "t")
    assert list(d["t"])[0] == "b"


# ------------------------------------------------- cognitive task list

def test_cognitive_prompt_defines_exactly_12_tasks():
    import re
    text = p.PROMPT_COGNITIVE.read_text()
    assert len(re.findall(r"^## \d+\. ", text, re.M)) == 12


def test_no_cognitive_cross_reference_points_outside_the_12():
    import re
    text = p.PROMPT_COGNITIVE.read_text()
    defined = set(re.findall(r"^## (\d+)\. ", text, re.M))
    referenced = set(re.findall(r"task (\d+)", text))
    assert referenced <= defined, f"dangling: {sorted(referenced - defined)}"
