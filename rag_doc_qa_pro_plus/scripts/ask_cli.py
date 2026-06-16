from __future__ import annotations

import argparse
import json

from rich import print

from app.core.logging import setup_logging
from app.services.rag_service import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the RAG index")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    args = parser.parse_args()

    setup_logging()
    bundle = RAGService().answer(args.question)
    payload = {
        "question": bundle.question,
        "answer": bundle.answer,
        "citations": bundle.citations,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("[bold]Answer[/bold]")
        print(bundle.answer)
        print("\n[bold]Citations[/bold]")
        for c in bundle.citations:
            print(f"[{c['source_number']}] {c['file_name']} page={c['page']} score={c['score']:.4f}")


if __name__ == "__main__":
    main()
