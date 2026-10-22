"""API 서버. POST /query 하나로 질문을 받아 answer·contexts·trace 를 돌려준다."""

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import run

app = FastAPI()


class QueryRequest(BaseModel):
    """POST /query 요청 본문. question 에 환자 ID 와 작성자 메모를 담는다."""

    question: str


def _patient_contexts(output: dict) -> list[dict]:
    """get_patient_context 결과를 {환자ID}/{종류}/{항목} 문서들로 나눈다."""
    pid = output["patient_id"]
    note = output["start_note"]
    contexts = [
        {
            "doc_id": f"{pid}/profile/basic",
            "text": f"{output['age']}세 {output['sex']}, 입원 {output['admitted_at']}",
        },
        {
            "doc_id": f"{pid}/history/underlying_diseases",
            "text": "기저질환: " + ", ".join(output["underlying_diseases"]),
        },
        {
            "doc_id": f"{pid}/allergy/allergies",
            "text": "알레르기: " + ", ".join(output["allergies"]),
        },
    ]
    contexts += [
        {"doc_id": f"{pid}/diagnosis/{d['code']}", "text": f"{d['code']} {d['name']}"}
        for d in output["diagnoses"]
    ]
    contexts.append(
        {"doc_id": f"{pid}/note/{note['source']}", "text": f"{note['written_at']}\n{note['text']}"}
    )
    return contexts


def _order_text(order: dict) -> str:
    """처방 한 건을 '약품 용량 경로 빈도' 문장으로 만든다."""
    return f"{order['drug']} {order['dose']} {order['route']} {order['frequency']}"


def _changes_contexts(output: dict) -> list[dict]:
    """get_changes_since_last_note 결과를 Lab·판독·투약 문서들로 나눈다. '변화 없음' 항목은 건너뛴다."""
    pid = output["patient_id"]
    labs = output["labs"] if isinstance(output["labs"], list) else []
    readings = output["readings"] if isinstance(output["readings"], list) else []
    medications = output["medications"] if isinstance(output["medications"], dict) else {}
    contexts = []
    for lab in labs:
        text = (
            f"{lab['name']} {lab['value']} {lab['unit']} "
            f"(참고치 {lab['ref_low']}–{lab['ref_high']}, {lab['measured_at']})"
        )
        if lab["previous"]:
            text += f", 직전 {lab['previous']['value']} ({lab['previous']['measured_at']})"
        if lab["trend"]:
            text += ", 추세 " + "→".join(str(item["value"]) for item in lab["trend"])
        contexts.append({"doc_id": f"{pid}/lab/{lab['name']}", "text": text})
    for reading in readings:
        contexts.append(
            {
                "doc_id": f"{pid}/reading/{reading['exam']}",
                "text": f"{reading['exam']} ({reading['read_at']}): {reading['finding']}",
            }
        )
    for order in medications.get("new", []):
        contexts.append(
            {"doc_id": f"{pid}/medication/{order['drug']}", "text": "신규: " + _order_text(order)}
        )
    for order in medications.get("stopped", []):
        contexts.append(
            {"doc_id": f"{pid}/medication/{order['drug']}", "text": "중단: " + _order_text(order)}
        )
    for change in medications.get("changed", []):
        text = f"변경: {_order_text(change['before'])} → {_order_text(change['after'])}"
        contexts.append({"doc_id": f"{pid}/medication/{change['drug']}", "text": text})
    return contexts


def _kcd_contexts(output: dict) -> list[dict]:
    """search_kcd_codes 결과(상위 결과와 같은 묶음 코드)를 KCD/{코드} 문서들로 나눈다."""
    return [
        {"doc_id": r["source"], "text": f"{r['code']} {r['name']}"}
        for r in output["results"] + output.get("same_category", [])
    ]


def _build_contexts(trace: list[dict]) -> list[dict]:
    """trace 의 도구 기록에서 환자 데이터와 KCD 검색 결과를 contexts 로 모은다. 같은 doc_id 는 첫 것만 둔다."""
    builders = {
        "get_patient_context": _patient_contexts,
        "get_changes_since_last_note": _changes_contexts,
        "search_kcd_codes": _kcd_contexts,
    }
    contexts, seen = [], set()
    for record in trace:
        if record["step"] not in builders or not isinstance(record["output"], dict):
            continue
        for context in builders[record["step"]](record["output"]):
            if context["doc_id"] not in seen:
                seen.add(context["doc_id"])
                contexts.append(context)
    return contexts


@app.post("/query")
def query(request: QueryRequest) -> dict:
    """질문으로 에이전트를 돌려 answer·contexts·trace 를 돌려준다."""
    result = run(request.question)
    return {
        "answer": result["answer"],
        "contexts": _build_contexts(result["trace"]),
        "trace": result["trace"],
    }
