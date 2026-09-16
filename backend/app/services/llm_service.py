import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import Settings


SYSTEM_PROMPT = """你是一个企业知识库助手。请严格基于以下提供的文档内容回答问题。

规则：
1. 只根据提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确回复"未找到相关内容"
3. 回答简洁准确，使用中文
4. 回答中的每处结论都要用文档编号标注出处，如 [1]；多处依据可标 [1][3]。不要在末尾另行罗列来源清单
5. 知识库可能同时包含公司统一制度与多条产品线的相似文档：若问题未指明产品线或部门，以公司统一制度/主文档为准，不得混用其他产品线的数值或条款
6. 只有当提供的文档都确实不含所需信息时才回复"未找到相关内容"；不要因为存在其他产品线的类似条款而拒答"""


DIRECT_SYSTEM_PROMPT = """你是一个企业知识库助手。用户当前的问题无需查询知识库，请直接、简洁地用中文回答。
如果问题涉及你无法确定的具体业务信息，请提示用户该问题可能需要查询知识库。"""


class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.deepseek_temperature,
            max_tokens=settings.deepseek_max_tokens,
            request_timeout=settings.deepseek_timeout,
            max_retries=settings.deepseek_max_retries,
        )
        self._answer_chain = self._build_answer_chain()
        self._direct_chain = self._build_direct_chain()

    def _build_answer_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "文档内容：\n{context}\n\n问题：{question}\n\n回答："),
        ])
        return prompt | self.llm | StrOutputParser()

    def _build_direct_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", DIRECT_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        return prompt | self.llm | StrOutputParser()

    def build_answer_chain(self):
        """Return a chain that takes {context, question} and returns the answer string."""
        return self._answer_chain

    def build_direct_chain(self):
        """Return a chain for questions that do not need knowledge retrieval."""
        return self._direct_chain

    def _format_context(self, documents) -> str:
        seen = set()
        parts = []
        for doc in documents:
            parent_id = doc.metadata.get("parent_doc_id", "")
            if parent_id in seen:
                continue
            seen.add(parent_id)
            parts.append(
                f"[来源: {doc.metadata.get('doc_title', '未知')}]\n"
                f"{doc.metadata.get('parent_text', doc.page_content)}"
            )
        return "\n\n---\n\n".join(parts)

    def build_numbered_context(self, documents) -> tuple[str, dict]:
        """Number each parent block so the model can cite it as [n].

        Returns (context_text, {n: doc_title}).
        """
        seen = set()
        parts = []
        mapping = {}
        for doc in documents:
            parent_id = doc.metadata.get("parent_doc_id", "")
            if parent_id in seen:
                continue
            seen.add(parent_id)
            number = len(mapping) + 1
            title = doc.metadata.get("doc_title", "未知")
            mapping[number] = title
            parts.append(
                f"[{number}] 来源: {title}\n"
                f"{doc.metadata.get('parent_text', doc.page_content)}"
            )
        return "\n\n---\n\n".join(parts), mapping

    @staticmethod
    def extract_cited_sources(answer: str, mapping: dict) -> list[str]:
        """Return only the documents the answer actually cited via [n]."""
        cited = []
        for n in re.findall(r"\[(\d+)\]", answer or ""):
            title = mapping.get(int(n))
            if title and title not in cited:
                cited.append(title)
        return cited

    def _extract_sources(self, documents) -> list[str]:
        return list(dict.fromkeys(
            d.metadata.get("doc_title", "未知") for d in documents
        ))[:5]
