import math
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from config import load_settings
from policy import NO_EVIDENCE, ingest_policy
from retriever import CHROMA_DIRNAME, NO_EMBEDDINGS_REASON, PolicyRetriever, make_embeddings, rrf
from store import Store

ROOT = Path(__file__).resolve().parents[1]
POLICY_MD = ROOT / "eval" / "fixtures" / "volunteer_policy.md"
FAKE_KEY = "AKIAFAKEFAKEFAKEFAKE"

# 개념별 단어. 같은 개념 단어가 많을수록 그 축 값이 커진다.
CONCEPTS = [
    ["연간", "의무", "1년", "한 해", "매년", "얼마나", "분량", "채워"],
    ["확인서", "증빙"],
    ["분야", "환경", "복지"],
    ["목적", "장려"],
    ["문의", "담당"],
]


class ConceptEmbeddings(Embeddings):
    """규정 표현과 질문 표현이 달라도 같은 개념이면 가깝게 나오는 가짜 임베딩."""

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        v = [sum(text.count(w) for w in ws) + 0.01 for ws in CONCEPTS]
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.calls += 1
        return self._vec(text)


class FailingEmbeddings(Embeddings):
    def embed_documents(self, texts):
        raise RuntimeError(f"AccessDenied for key {FAKE_KEY}")

    def embed_query(self, text):
        raise RuntimeError(f"AccessDenied for key {FAKE_KEY}")


@pytest.fixture
def policy_store(store):
    ingest_policy(store, POLICY_MD)
    return store


def test_ac17_paraphrase_finds_annual_duty_clause(policy_store, settings):
    r = PolicyRetriever(policy_store, settings, ConceptEmbeddings())
    result = r.search("1년에 봉사 얼마나 해야 회사 기준 맞춰?")
    assert result.mode == "hybrid" and result.fallback_reason is None
    assert result.hits[0].clause_id == "제2조"
    assert result.hits[0].text in POLICY_MD.read_text(encoding="utf-8")  # 원문 그대로
    assert (settings.data_dir / CHROMA_DIRNAME).is_dir()


def test_vector_ranking_lifts_semantic_match(policy_store, settings):
    # 낱말이 겹치는 제1조·제3조가 BM25에서 앞서는 질문. 벡터 순위가 제2조를 끌어올린다(동사 어미 처리 뒤 질문 조정).
    from policy import search_policy

    query = "봉사시간 인정 기준과 분야, 1년 분량은?"
    bm25 = [h.clause_id for h in search_policy(policy_store, query).hits]
    hybrid = [h.clause_id for h in PolicyRetriever(policy_store, settings, ConceptEmbeddings()).search(query).hits]
    assert bm25.index("제2조") == 2
    assert hybrid.index("제2조") == 1


def test_ac17_embedding_failure_falls_back_to_bm25(policy_store, settings):
    r = PolicyRetriever(policy_store, settings, FailingEmbeddings())
    result = r.search("1년에 봉사 얼마나 해야 회사 기준 맞춰?")
    assert result.status == "found" and result.mode == "bm25"
    assert result.hits[0].clause_id == "제2조"
    assert "AccessDenied" in result.fallback_reason
    assert FAKE_KEY not in result.fallback_reason  # 마스킹
    assert r.events and r.events[0]["event"] == "embedding_fallback"
    assert FAKE_KEY not in str(r.events)


def test_index_failure_does_not_raise(policy_store, settings):
    r = PolicyRetriever(policy_store, settings, FailingEmbeddings())
    reason = r.index()
    assert reason and FAKE_KEY not in reason


def test_no_embeddings_uses_bm25(policy_store, settings):
    r = PolicyRetriever(policy_store, settings, None)
    result = r.search("확인서 없이 한 봉사도 인정돼?")
    assert result.mode == "bm25" and result.fallback_reason == NO_EMBEDDINGS_REASON
    assert result.hits[0].clause_id == "제4조"
    assert r.index() == NO_EMBEDDINGS_REASON


def test_no_evidence_and_no_documents_still_apply(store, settings):
    r = PolicyRetriever(store, settings, ConceptEmbeddings())
    assert r.search("몇 시간 해야 돼?").status == "no_documents"
    ingest_policy(store, POLICY_MD)
    result = r.search("헌혈도 봉사시간으로 인정돼?")
    assert result.status == "no_evidence" and result.message == NO_EVIDENCE


def test_index_syncs_with_reingest(store, settings, tmp_path):
    emb = ConceptEmbeddings()
    r = PolicyRetriever(store, settings, emb)
    f = tmp_path / "rules.md"
    f.write_text("## 제1조 (의무)\n연 4시간 이상 한다.\n\n## 제2조 (확인서)\n확인서를 낸다.", encoding="utf-8")
    ingest_policy(store, f)
    assert r.index() is None
    assert len(r.vectorstore().get(include=[])["ids"]) == 2
    calls = emb.calls
    r.search("의무 시간은?")
    assert emb.calls == calls + 1  # 이미 색인된 조항은 다시 임베딩하지 않음(질문만)

    f.write_text("## 제1조 (의무)\n연 6시간 이상 한다.", encoding="utf-8")
    ingest_policy(store, f)
    result = r.search("의무 시간은?")
    assert len(r.vectorstore().get(include=[])["ids"]) == 1  # 지난 조항은 색인에서 빠짐
    assert "6시간" in result.hits[0].text


def test_index_persists_across_retrievers(policy_store, settings):
    emb = ConceptEmbeddings()
    PolicyRetriever(policy_store, settings, emb).index()
    emb2 = ConceptEmbeddings()
    PolicyRetriever(policy_store, settings, emb2).search("확인서 없이 한 봉사도 인정돼?")
    assert emb2.calls == 1  # DATA_DIR/chroma에 남은 색인을 다시 씀


def test_rrf_orders_by_combined_rank():
    fused = rrf([[0, 1, 2], [1, 0]])
    assert [i for i, _ in fused] == [0, 1, 2] or [i for i, _ in fused] == [1, 0, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert dict(fused)[2] == pytest.approx(1 / 63)


def test_make_embeddings_none_without_model(settings):
    assert make_embeddings(settings) is None


# ---- 실연동 (키가 있을 때만) ----

def _live_settings():
    env = ROOT / ".env"
    s = load_settings(env_file=env) if env.is_file() else None
    if not s or not s.embedding_model or not (s.aws_access_key_id and s.aws_secret_access_key):
        pytest.skip("Bedrock 임베딩 설정이 없어 실연동 테스트를 건너뜁니다.")
    return s


@pytest.mark.live
def test_live_titan_paraphrase(tmp_path, clock):
    live = _live_settings()
    from dataclasses import replace

    s = replace(live, data_dir=tmp_path / "live_data")
    store = Store.open(s, now=clock)
    try:
        ingest_policy(store, POLICY_MD)
        r = PolicyRetriever(store, s, make_embeddings(s))
        result = r.search("1년에 봉사 얼마나 해야 회사 기준 맞춰?")
        assert result.mode == "hybrid", result.fallback_reason
        assert result.hits[0].clause_id == "제2조"
    finally:
        store.close()
