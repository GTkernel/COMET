"""Validate evidence selections and render reports from supplied telemetry facts."""
from __future__ import annotations

import json
import math
from src.ingestion.manifest import is_label

MISSING = ("channel_mapping", "physical_units", "mission_documentation", "root_cause_evidence",
           "operating_mode", "raw_observations")


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items() if not is_label(str(k))}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Evidence contains a nonfinite value")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("Evidence values must be JSON-compatible")


def build_evidence(packet: dict, chunks: list[dict]) -> dict:
    facts = []
    def add(kind, value):
        if value is not None and value != {} and value != []:
            facts.append({"id": f"f{len(facts) + 1}", "kind": kind, "value": _safe(value)})
    for key in ("anomaly_id", "dataset_id", "model_id", "run_id", "score_timestamp", "time_window",
                "input_window", "anomaly_score", "threshold", "model_context"):
        add(key, packet.get(key))
    for name, value in sorted((packet.get("numeric_summary") or {}).items()):
        if not is_label(name):
            add("numeric_summary", {"feature": name, **value})
    for item in packet.get("top_contributing_features") or []:
        if isinstance(item, dict) and not is_label(str(item.get("feature", ""))):
            add("feature_contribution", item)
    add("categorical_changes", packet.get("categorical_changes"))
    excerpts = []
    remaining = 1200
    for chunk in chunks[:3]:
        if not chunk.get("chunk_id") or not chunk.get("document_version"):
            continue
        text = str(chunk.get("text", ""))
        # Use UTF-8 byte length as a conservative bound when the tokenizer is unknown.
        data = text.encode("utf-8")
        if len(data) > remaining:
            continue
        if not text.strip():
            break
        remaining -= len(text.encode("utf-8"))
        excerpts.append({"id": f"e{len(excerpts)+1}", "chunk_id": str(chunk["chunk_id"]),
                         "document_version": str(chunk["document_version"]),
                         "source_file": str(chunk.get("source_file", "unknown")), "text": text})
    mappings = packet.get("documented_mappings") or {}
    verified = {}
    if isinstance(mappings, dict):
        for channel, mapping in mappings.items():
            if is_label(channel) or not isinstance(mapping, dict):
                continue
            subsystem = mapping.get("subsystem")
            if not isinstance(subsystem, str) or not subsystem.strip():
                continue
            quote = mapping.get("evidence_quote")
            # Require a reviewed mapping and its exact source quote. A document
            # may mention a channel and subsystem while denying their relationship.
            matching = [e for e in excerpts if e["source_file"] == mapping.get("source_file")]
            if (isinstance(mapping.get("verified_by"), str) and mapping["verified_by"].strip()
                    and isinstance(quote, str) and channel in quote and subsystem in quote
                    and matching and all(e["document_version"] == mapping.get("document_version") for e in matching)
                    and any(quote in e["text"] for e in matching)):
                verified[channel] = {k: mapping[k] for k in ("subsystem", "unit", "source_file", "document_version") if k in mapping}
    if verified:
        add("documented_mappings", verified)
    missing = ["root_cause_evidence"]
    features = {item.get("feature") for item in packet.get("top_contributing_features", []) if isinstance(item, dict)}
    if not verified or not features <= verified.keys():
        missing.append("channel_mapping")
    numeric = packet.get("numeric_summary") or {}
    if not numeric or any(not item.get("unit") for item in numeric.values()):
        missing.append("physical_units")
    if not excerpts:
        missing.append("mission_documentation")
    return {"version": 1, "facts": facts, "excerpts": excerpts, "missing_information": missing}


def selection_schema(bundle: dict) -> dict:
    def array(values):
        return {"type": "array", "items": {"type": "string", "enum": values} if values else {"type": "string"},
                "maxItems": len(values), "uniqueItems": True}
    return {"type": "object", "additionalProperties": False,
            "required": ["fact_ids", "excerpt_ids", "missing_information"],
            "properties": {"fact_ids": array([f["id"] for f in bundle["facts"]]),
                           "excerpt_ids": array([e["id"] for e in bundle["excerpts"]]),
                           "missing_information": array(list(MISSING))}}


def validate_selection(bundle: dict, selection: dict) -> dict:
    allowed = {"fact_ids": {f["id"] for f in bundle["facts"]},
               "excerpt_ids": {e["id"] for e in bundle["excerpts"]}, "missing_information": set(MISSING)}
    if not isinstance(selection, dict) or set(selection) != set(allowed):
        raise ValueError("Selection must contain only fact_ids, excerpt_ids and missing_information")
    for key, permitted in allowed.items():
        values = selection[key]
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or len(set(values)) != len(values) or not set(values) <= permitted:
            raise ValueError(f"Invalid evidence selection: {key}")
    return selection


def factual_selection(bundle: dict) -> dict:
    return {"fact_ids": [f["id"] for f in bundle["facts"]], "excerpt_ids": [],
            "missing_information": list(bundle["missing_information"])}


def render_report(bundle: dict, selection: dict) -> str:
    validate_selection(bundle, selection)
    # Render every packet fact even if the model leaves it out of its selection.
    lines = ["1. Summary of the anomaly"]
    summary = {f["kind"]: f["value"] for f in bundle["facts"] if f["kind"] in {
        "anomaly_id", "run_id", "score_timestamp", "time_window", "input_window", "anomaly_score", "threshold"}}
    lines.append(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    mappings = next((f["value"] for f in bundle["facts"] if f["kind"] == "documented_mappings"), None)
    lines.extend(["2. Subsystem or behavior involved",
                  "Documented channel mappings: " + json.dumps(mappings, sort_keys=True) if mappings else "Subsystem: not established.",
                  "Root cause: not established.",
                  "3. Evidence from telemetry"])
    for fact in bundle["facts"]:
        if fact["kind"] in {"numeric_summary", "feature_contribution", "categorical_changes", "model_context"}:
            lines.append(f"[{fact['id']}] {fact['kind']}: " + json.dumps(fact["value"], ensure_ascii=False, sort_keys=True, allow_nan=False))
    lines.append("4. Relevant documentation references")
    selected = set(selection["excerpt_ids"])
    for excerpt in bundle["excerpts"]:
        if excerpt["id"] in selected:
            lines.append(f"[{excerpt['chunk_id']}@{excerpt['document_version']}] {excerpt['source_file']}\nQuoted evidence follows.\n" +
                         "\n".join("> " + line for line in excerpt["text"].splitlines()))
    if not selected:
        lines.append("No relevant documentation selected.")
    lines.extend(["5. Recommended next actions", "Inspect raw values and missing observations. Check for operating-mode changes and review relevant documentation.",
                  "6. Additional data needed"])
    missing = sorted(set(bundle["missing_information"]) | set(selection["missing_information"]))
    lines.append(", ".join(missing) if missing else "No additional evidence categories selected.")
    return "\n".join(lines)
