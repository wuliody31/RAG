# Training data format

`retrieval_pairs.jsonl` uses one JSON object per line:

```json
{"query":"question text","positive":"relevant chunk","negative":"hard negative chunk"}
```

You can build it from `data/eval/questions.jsonl` after documents are ingested:

```bash
python scripts/build_training_data.py --eval data/eval/questions.jsonl --output data/training/retrieval_pairs.jsonl
```
