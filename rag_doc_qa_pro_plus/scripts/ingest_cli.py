from __future__ import annotations

import argparse
import json

from rich import print

from app.core.logging import setup_logging
from app.ingestion.ingestor import Ingestor


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index")
    parser.add_argument("--path", required=True, help="File or folder path")
    args = parser.parse_args()

    setup_logging()
    result = Ingestor().ingest_path(args.path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
