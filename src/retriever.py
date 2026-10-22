"""KCD-9 검색 파이프라인.

data/kcd9.csv 를 BM25(kiwipiepy 토크나이저)와 벡터 검색(Amazon Titan Text Embeddings V2, Chroma)을
EnsembleRetriever 로 합친 하이브리드 검색으로 찾는다. Chroma DB 는 `python -m src.retriever` 로
미리 만들어 두고, 검색할 때는 불러오기만 한다.
"""

import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from botocore.config import Config
from dotenv import load_dotenv
from kiwipiepy import Kiwi
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KCD_PATH = DATA_DIR / "kcd9.csv"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION = "kcd9"
REGION = "us-east-1"
TOP_K = 10
BUILD_BATCH = 500
BUILD_WORKERS = 8

_kiwi = Kiwi()
# Agent 가 search_kcd_codes 를 병렬로 부를 때 검색기를 한 번만 만들도록 첫 초기화를 잠근다
_init_lock = threading.Lock()


def _tokenize(text: str) -> list[str]:
    """kiwipiepy 로 문장을 형태소 단위로 나눠 BM25 토큰 목록을 만든다."""
    return [token.form for token in _kiwi.tokenize(text)]


def _load_documents() -> list[Document]:
    """kcd9.csv 의 각 행을 코드·한글 명칭·영문 명칭이 담긴 Document 로 만든다."""
    with KCD_PATH.open(encoding="utf-8") as f:
        return [
            Document(
                page_content=f"{row['name_ko']} {row['name_en']}",
                metadata={"code": row["code"], "name_ko": row["name_ko"]},
            )
            for row in csv.DictReader(f)
        ]


@lru_cache(maxsize=1)
def _bm25_retriever() -> BM25Retriever:
    """BM25 검색기를 한 번만 만들어 재사용한다."""
    return BM25Retriever.from_documents(
        _load_documents(), preprocess_func=_tokenize, k=TOP_K
    )


def _vector_store() -> Chroma:
    """data/chroma_db 의 KCD-9 Chroma 컬렉션을 Titan V2 임베딩과 함께 연다."""
    embeddings = BedrockEmbeddings(
        model_id=os.environ["BEDROCK_EMBEDDING_MODEL_ID"],
        region_name=REGION,
        config=Config(retries={"mode": "adaptive", "max_attempts": 10}),
    )
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def build_vector_db() -> None:
    """kcd9.csv 를 임베딩해 Chroma DB 를 미리 만든다.

    이미 들어 있는 코드는 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 만든다.
    """
    store = _vector_store()
    existing = set(store.get(include=[])["ids"])
    docs = [d for d in _load_documents() if d.metadata["code"] not in existing]
    print(f"기존 {len(existing)}건, 새로 임베딩할 문서 {len(docs)}건")
    batches = [docs[i : i + BUILD_BATCH] for i in range(0, len(docs), BUILD_BATCH)]

    def add(batch: list[Document]) -> int:
        store.add_documents(batch, ids=[d.metadata["code"] for d in batch])
        return len(batch)

    done = 0
    with ThreadPoolExecutor(BUILD_WORKERS) as pool:
        for n in pool.map(add, batches):
            done += n
            print(f"  {done}/{len(docs)}")
    print(f"완료: 컬렉션 {store._collection.count()}건")


@lru_cache(maxsize=1)
def _vector_retriever():
    """미리 만든 Chroma DB 를 불러오기만 해서 벡터 검색기를 만든다."""
    store = _vector_store()
    if store._collection.count() == 0:
        raise RuntimeError("Chroma DB 가 비어 있다. `python -m src.retriever` 로 먼저 만든다.")
    return store.as_retriever(search_kwargs={"k": TOP_K})


@lru_cache(maxsize=1)
def _ensemble_retriever() -> EnsembleRetriever:
    """BM25 와 벡터 검색을 같은 비중으로 합친 하이브리드 검색기를 만든다."""
    return EnsembleRetriever(
        retrievers=[_bm25_retriever(), _vector_retriever()],
        weights=[0.5, 0.5],
        id_key="code",
    )


def search(query: str) -> list[Document]:
    """검색어 하나로 KCD-9 문서를 하이브리드 검색해 상위 10건을 돌려준다."""
    with _init_lock:
        retriever = _ensemble_retriever()
    return retriever.invoke(query)[:TOP_K]


if __name__ == "__main__":
    build_vector_db()
