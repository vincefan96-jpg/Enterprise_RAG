"""Ragas evaluation toolkit.

Exports are lazy: importing this package (or a light submodule such as
``ragas_eval.metrics``) must not pull in Ragas/Milvus, which are heavy and
on some Python versions interfere with the event loop.
"""

__all__ = ["TestQuery", "SyntheticTestSetGenerator", "RagasEvaluationRunner"]


def __getattr__(name):
    if name == "TestQuery":
        from .models import TestQuery

        return TestQuery
    if name == "SyntheticTestSetGenerator":
        from .generator import SyntheticTestSetGenerator

        return SyntheticTestSetGenerator
    if name == "RagasEvaluationRunner":
        from .runner import RagasEvaluationRunner

        return RagasEvaluationRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
