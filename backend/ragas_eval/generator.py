import json
from dataclasses import asdict
from .models import TestQuery


GENERATE_PROMPT = """You are generating test questions for a RAG evaluation dataset.

Based on the following document, generate {num_questions} diverse questions that the document can answer. The questions should:
- Be specific and answerable from the document content
- Cover different topics/sections of the document
- Vary in type: factual lookup, comparison, procedural, definition
- Be written in Chinese (matching the document language)

Document:
{document_body}

Output exactly {num_questions} questions, one per line, with no numbering or prefixes."""

ANSWER_PROMPT = """Based on the following document, write a concise answer (1-3 sentences) to the question. Extract information directly from the document, do not fabricate.

Document:
{document_body}

Question: {question}

Answer:"""


class SyntheticTestSetGenerator:
    def __init__(self, store, settings):
        self.store = store
        self.settings = settings

    def _get_llm(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.settings.deepseek_model,
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            temperature=0.7,
            max_tokens=1024,
            request_timeout=120,
            max_retries=2,
        )

    def _get_document_body(self, doc_title: str) -> str:
        from app.services.milvus_store import filter_by_doc_title

        try:
            results = self.store.client.query(
                collection_name=self.settings.milvus_collection,
                filter=filter_by_doc_title(doc_title),
                output_fields=["parent_text"],
                limit=10000,
            )
        except Exception:
            return ""

        seen = set()
        parts = []
        for r in results:
            pt = r.get("parent_text", "")
            if pt and pt not in seen:
                seen.add(pt)
                parts.append(pt)

        body = "\n\n".join(parts)
        return body[:8000]

    def generate(
        self,
        doc_titles: list[str] | None = None,
        num_questions: int = 3,
        generate_answers: bool = True,
    ) -> list[TestQuery]:
        if doc_titles is None:
            doc_titles = self.store.list_documents()

        llm = self._get_llm()
        dataset = []

        for title in doc_titles:
            body = self._get_document_body(title)
            if not body.strip():
                continue

            prompt = GENERATE_PROMPT.format(
                num_questions=num_questions,
                document_body=body,
            )
            try:
                result = llm.invoke(prompt).content.strip()
            except Exception:
                continue

            for line in result.split("\n"):
                q = line.strip()
                if q:
                    tq = TestQuery(question=q, relevant_doc_titles=[title])

                    if generate_answers:
                        try:
                            answer = llm.invoke(
                                ANSWER_PROMPT.format(
                                    document_body=body, question=q
                                )
                            ).content.strip()
                            tq.ground_truth_answer = answer
                        except Exception:
                            pass

                    dataset.append(tq)

        return dataset

    @staticmethod
    def save(dataset: list[TestQuery], path: str):
        data = {
            "version": 1,
            "queries": [asdict(q) for q in dataset],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> list[TestQuery]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            TestQuery(**q) for q in data.get("queries", [])
        ]
