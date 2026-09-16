"""Index a corpus directory into a Milvus collection for evaluation.

Usage:
  python -m ragas_eval.seed_corpus <corpus_dir> [--collection NAME]

Defaults to the collection configured in .env. Use a dedicated collection
(e.g. rag_eval_general_v1) so evaluation corpora never mix with the
working knowledge base.
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUPPORTED = (".md", ".txt", ".pdf", ".docx")


def main():
    parser = argparse.ArgumentParser(description="Seed an eval corpus into Milvus")
    parser.add_argument("corpus_dir")
    parser.add_argument("--collection", default=None)
    args = parser.parse_args()

    from app.config import get_settings
    from app.services.chunker import DocumentChunker
    from app.services.document_parser import DocumentParser
    from app.services.embedding_service import EmbeddingService
    from app.services.milvus_store import MilvusStore

    settings = get_settings()
    if args.collection:
        settings.milvus_collection = args.collection

    files = sorted(
        os.path.join(args.corpus_dir, name)
        for name in os.listdir(args.corpus_dir)
        if name.lower().endswith(SUPPORTED)
    )
    if not files:
        print(f"No supported files ({', '.join(SUPPORTED)}) in {args.corpus_dir}")
        return

    store = MilvusStore(settings)
    store.connect()
    asyncio.run(store.init_collection())

    embed = EmbeddingService(settings)
    parser = DocumentParser()
    chunker = DocumentChunker(
        parent_size=settings.parent_chunk_size,
        child_size=settings.child_chunk_size,
        overlap=settings.chunk_overlap,
    )

    print(f"Collection: {settings.milvus_collection}")
    try:
        for path in files:
            name = os.path.basename(path)
            existing = (
                store.delete_by_doc_title(name)[0]
                if name in store.list_documents()
                else None
            )
            if existing:
                print(f"  Replaced existing: {name}")
            text = parser.parse(path)
            chunks = chunker.split(text, name)
            for c in chunks:
                c.source_type = os.path.splitext(name)[1].lstrip(".")
                c.file_path = path
            embeddings = embed.encode_documents([c.text for c in chunks])
            store.insert(
                chunks,
                [e["dense"] for e in embeddings],
                [e["sparse"] for e in embeddings],
            )
            print(f"  Indexed: {name} ({len(chunks)} chunks)")
    finally:
        embed.cleanup()
        if store.client:
            store.client.close()

    print(f"\nDocuments in {settings.milvus_collection}:")
    store2 = MilvusStore(settings)
    store2.connect()
    asyncio.run(store2.init_collection())
    for title in store2.list_documents():
        print(f"  - {title}")
    if store2.client:
        store2.client.close()


if __name__ == "__main__":
    main()
