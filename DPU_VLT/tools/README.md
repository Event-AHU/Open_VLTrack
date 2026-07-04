# Tools

## build_llm_vocab.py

Generate structured semantic vocab (target/attributes/negatives/bg concepts) from dataset text files using an OpenAI-compatible API.

### Basic usage

Single file:
```bash
python tools/build_llm_vocab.py \
  --input /path/to/seq.txt \
  --seq_name seq_name \
  --output /path/to/vocab.json
```

Directory:
```bash
python tools/build_llm_vocab.py \
  --input_dir /path/to/lasot_train_concise \
  --output /path/to/lasot_llm_vocab.json \
  --mode first --max_lines 1 --resume
```

One JSON per txt file:
```bash
python tools/build_llm_vocab.py \
  --input_dir /path/to/lasot_train_concise \
  --output_dir /path/to/lasot_llm_vocab_by_seq \
  --mode all --max_lines 0 --by_key --resume
```

### Qwen3 (DashScope) example

```bash
export DASHSCOPE_API_KEY=your_key
export LLM_API_KEY=$DASHSCOPE_API_KEY
export LLM_API_BASE="https://dashscope.aliyuncs.com/compatible-mode"
export LLM_MODEL="qwen3-max"

python tools/build_llm_vocab.py \
  --input_dir /path/to/lasot_train_concise \
  --output /path/to/lasot_llm_vocab.json \
  --mode first --max_lines 1 --resume
```

### Arguments

- `--input_dir`: directory containing `.txt` files.
- `--input`: single `.txt` file.
- `--seq_name`: required when using `--input`.
- `--output`: output JSON path.
- `--output_dir`: write one JSON per txt file into this directory.
- `--mode`: `first|random|all` (select which lines to feed).
- `--max_lines`: limit number of lines per file (used with `random|all`).
- `--limit`: limit number of files.
- `--resume`: skip sequences already in output.
- `--sleep`: sleep seconds between requests.
- `--by_key`: generate `concept_by_key` by each line key (useful for frame-aligned training).
- `--backend`: `openai|mock` (use `mock` for dry-run).
- `--api_base`, `--api_key`, `--model`, `--temperature`.

### Output format

```json
{
  "meta": {
    "backend": "openai",
    "model": "qwen3-max",
    "generated_at": "YYYY-MM-DD HH:MM:SS"
  },
  "data": {
    "seq_name": {
      "source": "/path/to/seq.txt",
      "texts_used": ["..."],
      "concept": {
        "target": "person",
        "attributes": ["red shirt", "backpack"],
        "negatives": ["another person", "bicycle"],
        "bg_concepts": ["street", "tree"]
      },
      "concept_by_key": {
        "0": {
          "text": "a white boat in the water.",
          "concept": {
            "target": "boat",
            "attributes": ["white"],
            "negatives": ["person"],
            "bg_concepts": ["water"]
          }
        }
      }
    }
  }
}
```
