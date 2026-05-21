"""
RAG：从 parsed_tesla.md 构建向量索引，使用 CitationQueryEngine 回答并附带出处。
默认 Gemini 嵌入（分批 + 限速）；可选本地 HuggingFace。索引持久化到 ./storage。
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

from llama_index.core import (
    Document,
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.bridge.pydantic import Field
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.schema import MetadataMode
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

# 默认 Markdown 与持久化目录（相对本脚本所在项目目录）
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MD = BASE_DIR / "parsed_tesla.md"
STORAGE_DIR = BASE_DIR / "storage"
EMBED_META_FILENAME = "embed_meta.json"

# BGE 非对称检索：查询侧官方推荐前缀（文档侧不加强制前缀）
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Gemini 嵌入限速：每批 10–20 条，批间休眠（降低 TPM）
GEMINI_EMBED_BATCH_MIN = 20
GEMINI_EMBED_BATCH_MAX = 40
GEMINI_EMBED_BATCH_SLEEP_SEC = 60


def _clamp_embed_batch_size(size: int) -> int:
    return max(GEMINI_EMBED_BATCH_MIN, min(GEMINI_EMBED_BATCH_MAX, size))


class RateLimitedGeminiEmbedding(GeminiEmbedding):
    """Gemini 嵌入：禁止整表一次请求，按小批调用并在批间 sleep。"""

    api_batch_size: int = Field(
        default=15,
        description="每次 embedContent / batch_embed 的切片数量（10–20）",
    )
    batch_sleep_seconds: float = Field(
        default=GEMINI_EMBED_BATCH_SLEEP_SEC,
        description="每批 API 调用结束后的强制休眠秒数",
    )

    def __init__(
        self,
        api_batch_size: int = 15,
        batch_sleep_seconds: float = GEMINI_EMBED_BATCH_SLEEP_SEC,
        **kwargs: Any,
    ) -> None:
        api_batch_size = _clamp_embed_batch_size(api_batch_size)
        # 与 LlamaIndex 外部分批对齐，避免一次把上百切片交给 _get_text_embeddings
        kwargs["embed_batch_size"] = api_batch_size
        super().__init__(
            api_batch_size=api_batch_size,
            batch_sleep_seconds=batch_sleep_seconds,
            **kwargs,
        )

    def _embed_chunk_sync(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if len(texts) == 1:
            return [
                self._model.embed_content(
                    model=self.model_name,
                    content=texts[0],
                    title=self.title,
                    task_type=self.task_type,
                    request_options=self._request_options,
                )["embedding"]
            ]
        result = self._model.embed_content(
            model=self.model_name,
            content=texts,
            title=self.title,
            task_type=self.task_type,
            request_options=self._request_options,
        )
        return result["embedding"]

    async def _embed_chunk_async(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if len(texts) == 1:
            result = await self._model.embed_content_async(
                model=self.model_name,
                content=texts[0],
                title=self.title,
                task_type=self.task_type,
                request_options=self._request_options,
            )
            return [result["embedding"]]
        result = await self._model.embed_content_async(
            model=self.model_name,
            content=texts,
            title=self.title,
            task_type=self.task_type,
            request_options=self._request_options,
        )
        return result["embedding"]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for start in range(0, len(texts), self.api_batch_size):
            chunk = texts[start : start + self.api_batch_size]
            all_embeddings.extend(self._embed_chunk_sync(chunk))
            time.sleep(self.batch_sleep_seconds)
        return all_embeddings

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for start in range(0, len(texts), self.api_batch_size):
            chunk = texts[start : start + self.api_batch_size]
            all_embeddings.extend(await self._embed_chunk_async(chunk))
            await asyncio.sleep(self.batch_sleep_seconds)
        return all_embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.embed_content(
            model=self.model_name,
            content=query,
            title=self.title,
            task_type="retrieval_query",
            request_options=self._request_options,
        )["embedding"]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        result = await self._model.embed_content_async(
            model=self.model_name,
            content=query,
            title=self.title,
            task_type="retrieval_query",
            request_options=self._request_options,
        )
        return result["embedding"]

# 强制在回答中标注资料编号（与 CitationQueryEngine 的 Source 1/2… 对应）
CITATION_QA_TEMPLATE_ZH = PromptTemplate(
    template=(
        "请仅根据下列编号资料作答。引用事实时必须在相关句子末尾标注出处编号，例如 [1]、[2]。\n"
        "每条回答都必须包含至少一处出处；只在你确实依据某条资料时再标注对应编号。\n"
        "若资料不足以回答或与问题无关，请明确说明，不要臆测。\n\n"
        "------\n"
        "{context_str}\n"
        "------\n"
        "问题：{query_str}\n"
        "回答："
    )
)

CITATION_REFINE_TEMPLATE_ZH = PromptTemplate(
    template=(
        "下面已有初答：{existing_answer}\n"
        "请仅根据下列编号资料进行补充或修正。引用时继续用 [1]、[2] 等形式标注出处；\n"
        "若资料无用则保持原回答。\n\n"
        "------\n"
        "{context_msg}\n"
        "------\n"
        "问题：{query_str}\n"
        "修订后的回答："
    )
)


def configure_global_models(google_api_key: str) -> tuple[Gemini, BaseEmbedding]:
    """
    在读取文档 / 建索引 / 创建 query_engine 之前绑定模型。
    若跳过此步，访问 Settings.llm 会懒加载为 OpenAI（见 llama_index.core.llms.utils.resolve_llm）。
    """
    use_local_embed = os.getenv("USE_LOCAL_EMBEDDING", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    if use_local_embed:
        print("嵌入：本地 HuggingFace（不消耗 Gemini 嵌入配额）")
        embed_model: BaseEmbedding = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            query_instruction=BGE_QUERY_INSTRUCTION,
            embed_batch_size=16,
            trust_remote_code=False,
        )
    else:
        api_batch = _clamp_embed_batch_size(
            int(os.getenv("GEMINI_EMBED_BATCH_SIZE", "15"))
        )
        batch_sleep = float(
            os.getenv("GEMINI_EMBED_BATCH_SLEEP", str(GEMINI_EMBED_BATCH_SLEEP_SEC))
        )
        print(
            f"嵌入：Gemini（每批 {api_batch} 切片，批间 sleep {batch_sleep}s，降低 TPM）"
        )
        embed_model = RateLimitedGeminiEmbedding(
            model_name="models/gemini-embedding-001",
            api_key=google_api_key,
            task_type="retrieval_document",
            api_batch_size=api_batch,
            batch_sleep_seconds=batch_sleep,
        )

    llm = Gemini(
        model="models/gemini-2.5-flash",
        api_key=google_api_key,
        temperature=0.1,
    )

    # 必须显式写入 Settings，避免任何环节触发 default -> OpenAI
    Settings.embed_model = embed_model
    Settings.llm = llm
    print(f"LLM：Gemini ({llm.model})")

    return llm, embed_model


def load_markdown_documents(md_path: Path) -> list[Document]:
    text = md_path.read_text(encoding="utf-8")
    return [
        Document(
            text=text,
            metadata={
                "file_name": md_path.name,
                "file_path": str(md_path.resolve()),
            },
        )
    ]


def _storage_looks_ready(persist_dir: Path) -> bool:
    """判断是否为已 persist 过的索引目录（避免空文件夹误走加载）。"""
    if not persist_dir.is_dir():
        return False
    return (persist_dir / "index_store.json").is_file() and (
        persist_dir / "docstore.json"
    ).is_file()


def _embedding_profile(embed_model: Any) -> dict[str, Any]:
    if isinstance(embed_model, HuggingFaceEmbedding):
        provider = "huggingface"
    elif isinstance(embed_model, (RateLimitedGeminiEmbedding, GeminiEmbedding)):
        provider = "gemini"
    else:
        provider = embed_model.class_name()
    return {
        "provider": provider,
        "model_name": getattr(embed_model, "model_name", "unknown"),
    }


def _probe_embedding_dim(embed_model: Any) -> int:
    vec = embed_model.get_query_embedding("dimension probe")
    return len(vec)


def _read_stored_embedding_dim(persist_dir: Path) -> int | None:
    meta_path = persist_dir / EMBED_META_FILENAME
    if meta_path.is_file():
        return int(json.loads(meta_path.read_text(encoding="utf-8"))["embedding_dim"])

    vs_path = persist_dir / "default__vector_store.json"
    if not vs_path.is_file():
        return None
    data = json.loads(vs_path.read_text(encoding="utf-8"))
    embedding_dict = data.get("embedding_dict") or {}
    if not embedding_dict:
        return None
    return len(next(iter(embedding_dict.values())))


def _write_embed_meta(persist_dir: Path, profile: dict[str, Any], dim: int) -> None:
    meta = {**profile, "embedding_dim": dim}
    (persist_dir / EMBED_META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _clear_storage(persist_dir: Path) -> None:
    if persist_dir.is_dir():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)


def _storage_matches_embed_model(persist_dir: Path, embed_model: Any) -> bool:
    profile = _embedding_profile(embed_model)
    meta_path = persist_dir / EMBED_META_FILENAME

    if meta_path.is_file():
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if stored.get("provider") != profile["provider"] or stored.get(
            "model_name"
        ) != profile["model_name"]:
            print(
                f"\n⚠️  嵌入模型已变更：索引为 {stored.get('provider')}/{stored.get('model_name')}，"
                f"当前为 {profile['provider']}/{profile['model_name']}。\n"
                f"   将自动删除 {persist_dir} 并重建索引…\n"
            )
            return False
        return True

    stored_dim = _read_stored_embedding_dim(persist_dir)
    if stored_dim is None:
        return True

    current_dim = _probe_embedding_dim(embed_model)
    if stored_dim == current_dim:
        return True

    print(
        f"\n⚠️  嵌入维度不一致：磁盘索引为 {stored_dim} 维，"
        f"当前模型 ({profile['provider']}/{profile['model_name']}) 为 {current_dim} 维。\n"
        f"   常见原因：曾用 HuggingFace(384) 建索引，现改用 Gemini(3072)，或相反。\n"
        f"   将自动删除 {persist_dir} 并重建索引…\n"
    )
    return False


def get_or_build_index(
    documents: list[Document],
    persist_dir: Path,
    embed_model: BaseEmbedding,
) -> VectorStoreIndex:
    if _storage_looks_ready(persist_dir) and _storage_matches_embed_model(
        persist_dir, embed_model
    ):
        print(f"从 {persist_dir} 加载已持久化的索引…")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        return load_index_from_storage(
            storage_context,
            embed_model=embed_model,
        )  # type: ignore[return-value]

    if _storage_looks_ready(persist_dir):
        _clear_storage(persist_dir)

    print("正在构建向量索引（首次会下载模型 / 调用嵌入 API，可能较慢）…")
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    dim = _probe_embedding_dim(embed_model)
    _write_embed_meta(persist_dir, _embedding_profile(embed_model), dim)
    print(f"索引已保存到 {persist_dir}（嵌入维度 {dim}）")
    return index


def main() -> None:
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("请在 .env 中设置 GOOGLE_API_KEY。", file=sys.stderr)
        sys.exit(1)

    # 最先绑定模型，再读文档 / 建索引 / 问答，避免 Settings 懒加载到 OpenAI default
    llm, embed_model = configure_global_models(google_api_key)

    md_path = Path(os.getenv("RAG_MARKDOWN_PATH", DEFAULT_MD)).expanduser().resolve()
    if not md_path.is_file():
        print(f"找不到 Markdown 文件: {md_path}", file=sys.stderr)
        sys.exit(1)

    persist_dir = Path(os.getenv("RAG_STORAGE_DIR", STORAGE_DIR)).expanduser().resolve()

    print(f"正在加载: {md_path}")
    documents = load_markdown_documents(md_path)

    index = get_or_build_index(documents, persist_dir, embed_model)

    # 显式传入 Gemini，禁止 query_engine / response_synthesizer 回退到 Settings default
    query_engine = CitationQueryEngine.from_args(
        index,
        llm=llm,
        embed_model=embed_model,
        similarity_top_k=2,
        citation_chunk_size=512,
        citation_chunk_overlap=64,
        citation_qa_template=CITATION_QA_TEMPLATE_ZH,
        citation_refine_template=CITATION_REFINE_TEMPLATE_ZH,
        metadata_mode=MetadataMode.LLM,
        streaming=True,
    )

    print("\n已就绪。输入问题查询 Tesla 10-K 解析内容；输入 quit / exit 退出。\n")

    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("再见。")
            break

        response = query_engine.query(question)
        print("\n助手:")
        if isinstance(response, StreamingResponse):
            response.print_response_stream()
            print()
            resolved = response.get_response()
        else:
            print(response)
            resolved = response

        # 显式列出引用节点，便于核对原文位置
        if getattr(resolved, "source_nodes", None):
            print("\n--- 出处（Source nodes）---")
            for i, node in enumerate(resolved.source_nodes, start=1):
                meta = node.node.metadata or {}
                name = meta.get("file_name", meta.get("file_path", "unknown"))
                score = getattr(node, "score", None)
                score_s = f", score={score:.4f}" if score is not None else ""
                preview = node.node.get_content().strip().replace("\n", " ")[:240]
                print(f"  [{i}] {name}{score_s}")
                if preview:
                    print(f"      摘录: {preview}…")
        print()


if __name__ == "__main__":
    main()
