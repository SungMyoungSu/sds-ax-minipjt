"""규정 하이브리드 검색: BM25 + 임베딩(Titan) + Chroma, RRF로 결합.

- MCP 서버에서 채팅 LLM을 부르지 않는다는 규칙의 유일한 예외가 이 모듈의 임베딩 호출이다.
- 임베딩이 없거나 실패하면 BM25만 쓰고 그 이유(마스킹)를 결과와 events에 남긴다(AC17).
- 근거 없음 판단은 BM25와 같은 핵심어 방식이다. 벡터는 순위에만 쓴다
  (벡터 유사도만으로는 "헌혈" 같은 규정 밖 질문도 봉사 조항과 가깝게 나오기 때문).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.embeddings import Embeddings

from volunteer_agent.config import Settings
from volunteer_agent.masking import mask_exception
from volunteer_agent.policy import (
    ClauseEntry,
    PolicySearchResult,
    bm25_ranking,
    has_evidence_terms,
    load_entries,
    make_hit,
)
from volunteer_agent.store import Store

logger = logging.getLogger("volunteer_agent.rag")

CHROMA_DIRNAME = "chroma"
COLLECTION = "policy_clauses"
RRF_K = 60
VECTOR_CANDIDATES = 10
NO_EMBEDDINGS_REASON = "임베딩 설정 없음(EMBEDDING_MODEL)"


def make_embeddings(settings: Settings) -> Embeddings | None:
    """설정에 임베딩 모델이 있으면 BedrockEmbeddings, 없으면 None(BM25만 사용)."""
    if not settings.embedding_model:
        return None
    from langchain_aws import BedrockEmbeddings

    kwargs: dict[str, Any] = {
        "model_id": settings.embedding_model,
        "region_name": settings.aws_region or "us-east-1",
    }
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        if settings.aws_session_token:
            kwargs["aws_session_token"] = settings.aws_session_token
    return BedrockEmbeddings(**kwargs)


def _clause_key(entry: ClauseEntry) -> str:
    return f"clause-{entry.clause.id}"


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. rankings: 순위 목록(entries 위치)의 목록."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class PolicyRetriever:
    """규정 검색기. embeddings가 None이면 BM25만 쓴다."""

    def __init__(self, store: Store, settings: Settings, embeddings: Embeddings | None):
        self.store = store
        self.settings = settings
        self.embeddings = embeddings
        self.events: list[dict[str, str]] = []  # 실행 추적(15단계)이 가져갈 이벤트
        self._vectorstore = None

    # -- 색인 --

    def vectorstore(self):
        if self._vectorstore is None:
            import chromadb
            from langchain_chroma import Chroma

            path = self.settings.ensure_data_dir() / CHROMA_DIRNAME
            client = chromadb.PersistentClient(
                path=str(path), settings=chromadb.config.Settings(anonymized_telemetry=False)
            )
            self._vectorstore = Chroma(client=client, collection_name=COLLECTION,
                                       embedding_function=self.embeddings)
        return self._vectorstore

    def sync_index(self, entries: list[ClauseEntry]) -> None:
        """DB 조항과 Chroma 색인을 맞춘다. 지워진 조항은 빼고, 없는 조항만 임베딩해 넣는다."""
        vs = self.vectorstore()
        wanted = {_clause_key(e): e for e in entries}
        existing = set(vs.get(include=[])["ids"])
        stale = sorted(existing - set(wanted))
        if stale:
            vs.delete(ids=stale)
        missing = [k for k in wanted if k not in existing]
        if missing:
            vs.add_texts(
                texts=[wanted[k].clause.text for k in missing],
                metadatas=[{"filename": wanted[k].filename, "clause_id": wanted[k].clause.clause_id}
                           for k in missing],
                ids=missing,
            )

    def index(self) -> str | None:
        """수집 직후 색인. 실패해도 예외를 내지 않고 이유를 돌려준다(검색 때 다시 시도)."""
        if self.embeddings is None:
            return NO_EMBEDDINGS_REASON
        try:
            self.sync_index(load_entries(self.store))
            return None
        except Exception as exc:  # noqa: BLE001 - 어떤 실패든 BM25로 계속
            return self._record_fallback(exc)

    # -- 검색 --

    def _record_fallback(self, exc: Exception) -> str:
        reason = mask_exception(exc, self.settings.secret_values())
        self.events.append({"event": "embedding_fallback", "reason": reason})
        logger.warning("embedding fallback to BM25: %s", reason)
        return reason

    def _vector_ranking(self, query: str, entries: list[ClauseEntry]) -> list[int]:
        self.sync_index(entries)
        position = {_clause_key(e): i for i, e in enumerate(entries)}
        k = min(VECTOR_CANDIDATES, len(entries))
        docs = self.vectorstore().similarity_search_with_score(query, k=k)
        ranking = []
        for doc, _distance in docs:
            idx = position.get(doc.id)
            if idx is not None and idx not in ranking:
                ranking.append(idx)
        return ranking

    def search(self, query: str, top_k: int = 3) -> PolicySearchResult:
        entries = load_entries(self.store)
        if not entries:
            return PolicySearchResult(status="no_documents")
        if not has_evidence_terms(query, entries):
            return PolicySearchResult(status="no_evidence")

        bm25 = [i for i, _ in bm25_ranking(query, entries)]
        vector: list[int] = []
        mode, reason = "hybrid", None
        if self.embeddings is None:
            mode, reason = "bm25", NO_EMBEDDINGS_REASON
        else:
            try:
                vector = self._vector_ranking(query, entries)
            except Exception as exc:  # noqa: BLE001
                mode, reason = "bm25", self._record_fallback(exc)

        fused = rrf([bm25, vector]) if mode == "hybrid" else rrf([bm25])
        hits = [make_hit(entries[i], score) for i, score in fused[:top_k]]
        if not hits:
            return PolicySearchResult(status="no_evidence", mode=mode, fallback_reason=reason)
        return PolicySearchResult(status="found", hits=hits, mode=mode, fallback_reason=reason)
