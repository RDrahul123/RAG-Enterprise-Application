"""Build the index from everything in data/docs.

    python ingest.py
"""

from rag import DOCS_DIR, INDEX_DIR, RAG

if __name__ == "__main__":
    result = RAG().ingest()
    print(f"documents : {result['documents']}")
    print(f"pages     : {result['pages']}")
    print(f"chunks    : {result['chunks']}")
    print(f"index     : {INDEX_DIR}")
    print(f"docs dir  : {DOCS_DIR}")
