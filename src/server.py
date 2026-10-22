"""API 서버. POST /query 하나로 질문을 받아 answer·contexts·trace 를 돌려준다."""

import json

from fastapi import FastAPI
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from src.agent import run

app = FastAPI()


class QueryRequest(BaseModel):
    """POST /query 요청 본문. question 에 환자 ID 와 작성자 메모를 담는다."""

    question: str


def _tool_calls(messages: list) -> list[tuple[str, dict, dict | str]]:
    """대화 메시지에서 도구 호출을 순서대로 뽑아 (도구 이름, 입력, 출력) 목록으로 돌려준다."""
    outputs = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    calls = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            content = outputs.get(call["id"], "")
            try:
                output = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                output = content
            calls.append((call["name"], call["args"], output))
    return calls


def _build_trace(calls: list, answer: dict) -> list[dict]:
    """도구 호출 순서대로 trace 를 만들고, 마지막에 SOAP 정리 단계를 붙인다."""
    trace = [{"step": name, "input": args, "output": output} for name, args, output in calls]
    trace.append({"step": "soap_draft", "input": "도구 결과와 메모", "output": answer})
    return trace


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


def _changes_contexts(output: dict) -> list[dict]:
    """get_changes_since_last_note 결과를 Lab·판독 문서들로 나눈다."""
    pid = output["patient_id"]
    contexts = []
    for lab in output["labs"]:
        text = (
            f"{lab['name']} {lab['value']} {lab['unit']} "
            f"(참고치 {lab['ref_low']}–{lab['ref_high']}, {lab['measured_at']})"
        )
        if lab["previous"]:
            text += f", 직전 {lab['previous']['value']} ({lab['previous']['measured_at']})"
        contexts.append({"doc_id": f"{pid}/lab/{lab['name']}", "text": text})
    for reading in output["readings"]:
        contexts.append(
            {
                "doc_id": f"{pid}/reading/{reading['exam']}",
                "text": f"{reading['exam']} ({reading['read_at']}): {reading['finding']}",
            }
        )
    return contexts


def _kcd_contexts(output: dict) -> list[dict]:
    """search_kcd_codes 결과를 KCD/{코드} 문서들로 나눈다."""
    return [
        {"doc_id": r["source"], "text": f"{r['code']} {r['name']}"} for r in output["results"]
    ]


def _build_contexts(calls: list) -> list[dict]:
    """도구가 가져온 환자 데이터와 KCD 검색 결과를 contexts 로 모은다. 같은 doc_id 는 첫 것만 둔다."""
    builders = {
        "get_patient_context": _patient_contexts,
        "get_changes_since_last_note": _changes_contexts,
        "search_kcd_codes": _kcd_contexts,
    }
    contexts, seen = [], set()
    for name, _, output in calls:
        if name not in builders or not isinstance(output, dict):
            continue
        for context in builders[name](output):
            if context["doc_id"] not in seen:
                seen.add(context["doc_id"])
                contexts.append(context)
    return contexts


@app.post("/query")
def query(request: QueryRequest) -> dict:
    """질문으로 에이전트를 돌려 answer·contexts·trace 를 돌려준다."""
    result = run(request.question)
    calls = _tool_calls(result["messages"])
    return {
        "answer": result["answer"],
        "contexts": _build_contexts(calls),
        "trace": _build_trace(calls, result["answer"]),
    }
