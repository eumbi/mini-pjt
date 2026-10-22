"""KCD-9 검색 파이프라인. data/kcd9.csv 를 BM25(kiwipiepy 토크나이저)로 검색한다."""

import csv
from functools import lru_cache
from pathlib import Path

from kiwipiepy import Kiwi
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

KCD_PATH = Path(__file__).resolve().parent.parent / "data" / "kcd9.csv"
TOP_K = 10

_kiwi = Kiwi()


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


def search(query: str) -> list[Document]:
    """검색어 하나로 KCD-9 문서를 검색해 상위 10건을 돌려준다."""
    return _bm25_retriever().invoke(query)
