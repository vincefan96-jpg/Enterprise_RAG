"""Batch upload test documents to the RAG API.

Usage: python ragas_eval/upload_docs.py [docs_dir] [api_base_url]
  docs_dir  — defaults to ragas_eval/testsets/general_50/corpus/
  api_url   — defaults to http://localhost:8000

Uploads through the running backend, so documents land in the working
knowledge base. For an isolated evaluation collection use seed_corpus.py.
"""

import os
import sys
import requests
import glob


EXT_TO_MIME = {
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
}


def main():
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "testsets", "general_50", "corpus"
    )
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    api = f"{base_url}/api/documents"

    files = []
    for ext in EXT_TO_MIME:
        files.extend(glob.glob(os.path.join(docs_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(docs_dir, f"*{ext.upper()}")))
    files = sorted(set(files))

    if not files:
        print(f"No supported files found in {docs_dir}")
        print(f"Supported: {', '.join(EXT_TO_MIME.keys())}")
        return

    for path in files:
        name = os.path.basename(path)
        _, ext = os.path.splitext(name)
        mime = EXT_TO_MIME.get(ext.lower(), "application/octet-stream")
        print(f"Uploading: {name} ...", end=" ")
        with open(path, "rb") as f:
            r = requests.post(
                f"{api}/upload",
                files={"file": (name, f, mime)},
                timeout=60,
                proxies={"http": None, "https": None},
            )
        if r.status_code == 200:
            data = r.json()
            print(f"OK ({data.get('chunks', '?')} chunks)")
        else:
            detail = r.text[:120]
            print(f"FAIL ({r.status_code}): {detail}")

    print(f"\nDone. {len(files)} files uploaded.")
    print(f"Check: curl {base_url}/api/documents/list")


if __name__ == "__main__":
    main()
