#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import time
import urllib.request
import urllib.error


def read_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\\d+)\\s+(.*)$", line)
            if m:
                key = int(m.group(1))
                text = m.group(2).strip()
            else:
                key = len(lines)
                text = line
            lines.append({"key": key, "text": text})
    return lines


def select_texts(lines, mode, max_lines):
    if not lines:
        return []
    if mode == "first":
        return [lines[0]]
    if mode == "random":
        k = max_lines if max_lines > 0 else 1
        k = min(k, len(lines))
        return random.sample(lines, k)
    if mode == "all":
        if max_lines > 0:
            return lines[:max_lines]
        return lines
    raise ValueError("Unknown mode: %s" % mode)


def build_prompt(texts):
    text_block = "\n".join(["- %s" % t["text"] for t in texts])
    return (
        "You are given natural language descriptions used for visual object tracking.\n"
        "Extract a compact semantic summary for tracking. Output JSON ONLY with keys:\n"
        "target (string), attributes (list of strings), negatives (list of strings), bg_concepts (list of strings).\n"
        "Purpose:\n"
        "- target: the main object category/identity to be tracked.\n"
        "- attributes: short discriminative properties of the target (color, size, material, parts).\n"
        "- negatives: confusing objects similar to target that should be suppressed.\n"
        "- bg_concepts: background scene elements (not the target).\n"
        "Rules: use short noun phrases, lowercase, no punctuation in items.\n"
        "Examples:\n"
        "1) \"a white boat in the water\"\n"
        "   target: \"boat\"\n"
        "   attributes: [\"white\"]\n"
        "   negatives: [\"person\"]\n"
        "   bg_concepts: [\"water\"]\n"
        "2) \"a red car on the road\"\n"
        "   target: \"car\"\n"
        "   attributes: [\"red\"]\n"
        "   negatives: [\"bus\", \"truck\"]\n"
        "   bg_concepts: [\"road\"]\n\n"
        "Descriptions:\n%s\n"
    ) % text_block


def openai_chat(api_base, api_key, model, prompt, temperature):
    url = api_base.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = resp.read().decode("utf-8")
    result = json.loads(body)
    return result["choices"][0]["message"]["content"]


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    snippet = text[start:end + 1]
    return json.loads(snippet)


def mock_extract(texts):
    if not texts:
        return {"target": "", "attributes": [], "negatives": [], "bg_concepts": []}
    words = re.findall(r"[a-zA-Z0-9]+", texts[0]["text"].lower())
    target = " ".join(words[:2]) if words else ""
    return {"target": target, "attributes": [], "negatives": [], "bg_concepts": []}


def load_output(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {}, "data": {}}


def save_output(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Build semantic vocab using an LLM.")
    parser.add_argument("--input_dir", type=str, default="", help="Directory of *.txt files.")
    parser.add_argument("--input", type=str, default="", help="Single input file.")
    parser.add_argument("--seq_name", type=str, default="", help="Sequence name for single input.")
    parser.add_argument("--output", type=str, default="", help="Output JSON path.")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Write one JSON per txt file into this directory.")
    parser.add_argument("--mode", type=str, choices=["first", "random", "all"], default="first")
    parser.add_argument("--max_lines", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files.")
    parser.add_argument("--resume", action="store_true", help="Skip existing sequences.")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--by_key", action="store_true",
                        help="Generate concept per line key (concept_by_key).")

    parser.add_argument("--backend", type=str, choices=["openai", "mock"], default="openai")
    parser.add_argument("--api_base", type=str, default=os.environ.get("LLM_API_BASE", "http://localhost:8000"))
    parser.add_argument("--api_key", type=str, default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.2)

    args = parser.parse_args()

    if not args.input_dir and not args.input:
        raise ValueError("Provide --input_dir or --input")
    if args.input and not args.seq_name:
        raise ValueError("Provide --seq_name when using --input")
    if not args.output and not args.output_dir:
        raise ValueError("Provide --output or --output_dir")
    if args.output and args.output_dir:
        raise ValueError("Use only one of --output or --output_dir")

    if args.backend == "openai" and not args.api_base:
        raise ValueError("LLM_API_BASE is required for openai backend")
    if args.backend == "openai" and not args.api_key:
        raise ValueError("LLM_API_KEY is required for openai backend")

    def init_payload():
        return {"meta": {}, "data": {}}

    def update_meta(payload):
        payload.setdefault("meta", {})
        payload.setdefault("data", {})
        payload["meta"]["backend"] = args.backend
        payload["meta"]["model"] = args.model
        payload["meta"]["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    payload = None
    if args.output:
        payload = load_output(args.output)
        update_meta(payload)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    files = []
    if args.input_dir:
        for name in sorted(os.listdir(args.input_dir)):
            if name.endswith(".txt"):
                files.append(os.path.join(args.input_dir, name))
    else:
        files = [args.input]

    if args.limit > 0:
        files = files[:args.limit]

    for path in files:
        seq_name = args.seq_name
        if not seq_name:
            seq_name = os.path.splitext(os.path.basename(path))[0]
        lines = read_lines(path)
        selected = select_texts(lines, args.mode, args.max_lines)

        output_path = args.output
        if args.output_dir:
            output_path = os.path.join(args.output_dir, seq_name + ".json")
            if args.resume and os.path.exists(output_path):
                payload = load_output(output_path)
            else:
                payload = init_payload()
            update_meta(payload)

        if not args.by_key:
            if args.resume and seq_name in payload["data"]:
                continue
            prompt = build_prompt(selected)
            if args.backend == "mock":
                concept = mock_extract(selected)
            else:
                try:
                    content = openai_chat(args.api_base, args.api_key, args.model, prompt, args.temperature)
                    concept = extract_json(content)
                except Exception as e:
                    concept = {"error": str(e), "raw": None}

            payload["data"][seq_name] = {
                "source": path,
                "texts_used": [t["text"] for t in selected],
                "concept": concept,
            }
            save_output(output_path, payload)
            if args.sleep > 0:
                time.sleep(args.sleep)
            continue

        entry = payload["data"].get(seq_name, {})
        entry["source"] = path
        entry["texts_used"] = [t["text"] for t in selected]
        entry.setdefault("concept_by_key", {})

        for item in selected:
            key_str = str(item["key"])
            if args.resume and key_str in entry["concept_by_key"]:
                continue
            prompt = build_prompt([item])
            if args.backend == "mock":
                concept = mock_extract([item])
            else:
                try:
                    content = openai_chat(args.api_base, args.api_key, args.model, prompt, args.temperature)
                    concept = extract_json(content)
                except Exception as e:
                    concept = {"error": str(e), "raw": None}
            entry["concept_by_key"][key_str] = {
                "text": item["text"],
                "concept": concept,
            }
            payload["data"][seq_name] = entry
            save_output(output_path, payload)
            if args.sleep > 0:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
