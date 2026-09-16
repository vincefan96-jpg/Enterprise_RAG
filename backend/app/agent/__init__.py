"""Agentic RAG P0 package.

Importing this package must stay lightweight: ``graph`` pulls in ``langgraph``
and is therefore imported lazily by the API only when agent mode is enabled.
"""
