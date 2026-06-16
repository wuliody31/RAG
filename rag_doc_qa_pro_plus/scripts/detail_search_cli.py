from __future__ import annotations

import argparse

from rich import print

from app.core.config import get_settings
from app.retrieval.detail_retriever import DetailRetriever, result_to_detail_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-grained document detail search")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["hybrid", "semantic", "keyword", "exact"], default="hybrid")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--file-name", default=None)
    parser.add_argument("--page-from", type=int, default=None)
    parser.add_argument("--page-to", type=int, default=None)
    parser.add_argument("--section", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--neighbors", action="store_true")
    args = parser.parse_args()

    results = DetailRetriever(get_settings()).search(
        args.query,
        mode=args.mode,
        top_k=args.top_k,
        doc_id=args.doc_id,
        file_name=args.file_name,
        page_from=args.page_from,
        page_to=args.page_to,
        section=args.section,
        include_neighbors=args.neighbors,
    )
    for item in results:
        d = result_to_detail_dict(item, full_text=False)
        print(f"[bold]{d['chunk_id']}[/bold] file={d['file_name']} page={d['page']} score={d['score']:.4f}")
        print(d["text"][:800])
        print("-" * 80)


if __name__ == "__main__":
    main()
