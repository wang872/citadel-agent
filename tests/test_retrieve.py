from citadel.bm25 import BM25Index
from citadel.chunk import chunk_document, load_kb
from citadel.dense import HashingEmbedder, cosine
from citadel.index import HybridIndex
from citadel.rrf import reciprocal_rank_fusion
from citadel.schema import Chunk


def test_bm25_prefers_the_document_that_mentions_the_query() -> None:
    chunks = [
        Chunk("pricing-0", "pricing", "专业版定价", "北辰云专业版价格为每月 1999 元。"),
        Chunk("sla-0", "sla", "可用性承诺", "北辰云对外可用性承诺为 99.9%。"),
    ]
    index = BM25Index()
    index.add(chunks)
    hits = index.search("北辰云专业版价格", k=2)
    assert [doc_id for doc_id, _ in hits] == ["pricing-0", "sla-0"]
    assert hits[0][1] > hits[1][1]
    assert index.search("可用性承诺", k=1)[0][0] == "sla-0"


def test_rrf_promotes_a_document_agreed_by_both_lists() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], top=3)
    assert fused[0][0] == "b"
    assert fused[0][1] > fused[1][1]


def test_hash_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder()
    first = embedder.embed("AES-256 静态加密")
    second = embedder.embed("AES-256 静态加密")
    assert first == second
    assert abs(cosine(first, first) - 1) < 1e-6
    assert cosine(first, embedder.embed("食堂菜单番茄鸡蛋")) < 0.8


def test_hybrid_search_returns_the_security_chunk_for_encryption() -> None:
    hits = HybridIndex(load_kb()).search("静态数据用什么加密", k=3)
    assert any(hit.chunk.doc_id == "security" for hit in hits)


def test_long_paragraph_is_split_with_overlap() -> None:
    paragraph = "加密" * 200
    chunks = chunk_document("security", "数据安全", paragraph, max_chars=100, overlap=20)
    assert len(chunks) > 1
    assert chunks[0].id == "security-0"
    assert chunks[1].text.startswith("加密")
    assert chunks[0].text[-20:] == chunks[1].text[:20]
