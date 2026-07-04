#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


INVALID_LITERAL_SET = {"high", "mid", "low"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recover usable candidates from invalid_samples in TNL2K Qwen candidate cache."
    )
    parser.add_argument("--input", required=True, help="Input candidates json path")
    parser.add_argument("--output", required=True, help="Output candidates json path")
    parser.add_argument("--max-concept-words", type=int, default=4, help="Keep recovered concept only if words <= this value")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_json_object(text):
    if not text:
        raise ValueError("Empty text")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
    if candidate is None:
        raise ValueError("No JSON object found")
    return json.loads(candidate)


def normalize_signature(concept, background):
    return (str(concept).strip().lower(), str(background).strip().lower())


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out_path}. Use --overwrite")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", {})
    if not isinstance(entries, dict):
        raise SystemExit("Invalid format: root.entries is not a dict")

    recovered_total = 0
    touched_entries = 0
    parse_fail_total = 0

    for _, item in entries.items():
        if not isinstance(item, dict):
            continue
        candidates = item.get("candidates", [])
        invalid_samples = item.get("invalid_samples", [])
        if not isinstance(candidates, list) or not isinstance(invalid_samples, list):
            continue

        seen = set()
        for c in candidates:
            if not isinstance(c, dict):
                continue
            seen.add(normalize_signature(c.get("concept", ""), c.get("background", "")))

        recovered_here = 0
        for bad in invalid_samples:
            if not isinstance(bad, dict):
                continue
            raw_response = str(bad.get("raw_response", ""))
            error_msg = str(bad.get("error", ""))
            try:
                obj = extract_json_object(raw_response)
            except Exception:
                parse_fail_total += 1
                continue

            concept = str(obj.get("concept", obj.get("concepts", ""))).strip()
            background = str(obj.get("background", "")).strip()
            if not concept:
                continue
            if concept.lower() in INVALID_LITERAL_SET:
                continue
            if len(concept.split()) > int(args.max_concept_words):
                continue

            sig = normalize_signature(concept, background)
            if sig in seen:
                continue

            candidates.append(
                {
                    "concept": concept,
                    "background": background,
                    "raw_response": raw_response,
                    "recovered_from_invalid": True,
                    "recover_error": error_msg,
                }
            )
            seen.add(sig)
            recovered_here += 1

        if recovered_here > 0:
            touched_entries += 1
            recovered_total += recovered_here
            item["num_recovered_candidates"] = int(item.get("num_recovered_candidates", 0)) + recovered_here
            item["num_valid_candidates"] = len(candidates)

    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        data["meta"] = meta
    meta["recovery_max_concept_words"] = int(args.max_concept_words)
    meta["recovered_total"] = recovered_total
    meta["recovery_touched_entries"] = touched_entries
    meta["recovery_parse_fail_total"] = parse_fail_total

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[DONE] recovered={recovered_total}, touched_entries={touched_entries}, parse_fail={parse_fail_total}")
    print(f"[DONE] output={out_path}")


if __name__ == "__main__":
    main()
