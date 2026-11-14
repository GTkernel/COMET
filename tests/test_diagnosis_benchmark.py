"""Benchmark decisions cannot use acceptance cases or reward unsafe fallbacks."""
import copy
import json

import pytest

from scripts.benchmark_diagnosis import (CORPUS, aggregate, choose_model, load_cases,
                                         percentile, selection_score)
from src.rag.evidence import build_evidence, factual_selection, render_report


def test_corpus_has_80_disjoint_cases_and_expected_scenarios():
    corpus, cases = load_cases()
    assert "human" in corpus["review_status"].lower()
    assert len(cases) == 80
    dev = {case["family"] for case in cases if case["split"] == "development"}
    held = {case["family"] for case in cases if case["split"] == "acceptance"}
    assert not dev & held
    assert len({case["scenario"] for case in cases}) == 10
    assert all(not case["documents"] for case in cases if case["scenario"] == "no_documents")
    assert all(case["expected"]["insufficient_root_cause_evidence"] for case in cases)


def test_shared_case_family_rejected(tmp_path):
    corpus, _ = load_cases()
    corpus["cases"][-1]["family"] = corpus["cases"][0]["family"]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match="locked split"):
        load_cases(path)


def test_exact_selection_scores_one_and_wrong_excerpt_is_penalized():
    _, cases = load_cases()
    case = cases[0]
    bundle = build_evidence(case["packet"], case["documents"])
    selection = factual_selection(bundle)
    selection["excerpt_ids"] = [item["id"] for item in bundle["excerpts"] if item["chunk_id"] in case["expected"]["relevant_chunk_ids"]]
    assert selection_score(bundle, selection, case, "ollama_success")["selection_f1"] == 1
    wrong = copy.deepcopy(selection)
    wrong["excerpt_ids"] = [item["id"] for item in bundle["excerpts"]]
    assert selection_score(bundle, wrong, case, "ollama_success")["selection_f1"] < 1
    assert selection_score(bundle, selection, case, "deadline_exceeded")["fallback"]


def test_safe_renderer_handles_every_case_and_redacts_observed_annotation():
    _, cases = load_cases()
    for case in cases:
        bundle = build_evidence(case["packet"], case["documents"])
        selection = factual_selection(bundle)
        response = render_report(bundle, selection)
        assert "FORBIDDEN_ANNOTATION_SENTINEL" not in response
        assert "is_anomaly_signal" not in response
        assert "root cause: not established" in response.lower()
        assert case["packet"]["score_timestamp"] in response
        assert str(case["packet"]["anomaly_score"]) in response


def test_mapping_semantics_independent_of_builder_derived_ids():
    _, cases = load_cases()
    for case in cases:
        bundle = build_evidence(case["packet"], case["documents"])
        mappings = [f["value"] for f in bundle["facts"] if f["kind"] == "documented_mappings"]
        if case["scenario"] == "explicit_mapping":
            assert len(mappings) == 1
            assert {m["subsystem"] for m in mappings[0].values()} == {"thermal control"}
            assert {m["unit"] for m in mappings[0].values()} == {"K"}
            assert "channel_mapping" not in bundle["missing_information"]
            assert "physical_units" not in bundle["missing_information"]
        else:
            assert not mappings
            assert "channel_mapping" in bundle["missing_information"]
        assert "root_cause_evidence" in bundle["missing_information"]
        if case["scenario"] == "no_documents":
            assert "mission_documentation" in bundle["missing_information"]


def test_ranking_requires_retrieval_and_all_model_gates():
    good = {"passes_model_gates": True, "mean_selection_f1": .9,
            "warm_end_to_end_p95_seconds": 2, "maximum_loaded_vram_bytes": 20}
    better = {**good, "mean_selection_f1": .95, "warm_end_to_end_p95_seconds": 4}
    assert choose_model({"fast": good, "accurate": better}, .9) == "accurate"
    assert choose_model({"fast": good}, .89) is None
    assert choose_model({"bad": {**good, "passes_model_gates": False}}, 1) is None


def test_aggregate_does_not_reward_failure_controls_or_slow_models():
    row = {"seconds": 1, "score": {"answerable": True, "correct_abstention": True,
            "fallback": False, "selection_f1": 1}, "insufficient_root_cause_evidence": True}
    summary = aggregate([row, {**row, "failure_control": True, "seconds": 100}])
    assert summary["passes_model_gates"]
    assert aggregate([{**row, "seconds": 5.01}])["passes_model_gates"] is False
    assert aggregate([{**row, "score": {**row["score"], "fallback": True}}])["passes_model_gates"] is False
    assert percentile([1, 2, 3, 4]) == 4
