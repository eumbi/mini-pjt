"""메인 에이전트 그래프. create_agent 로 도구를 부르고, 결과를 SOAP Pydantic 모델로 정리한다."""

import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.guards import (
    MISSING_DATA,
    NO_FABRICATION,
    OTHER_PATIENT,
    UNKNOWN_PATIENT,
    DraftState,
    InputGuard,
    OutputGuard,
    PatientScope,
    is_fabrication_request,
    is_unknown_patient,
)
from src.tools import get_changes_since_last_note, get_patient_context, search_kcd_codes
from src.tracer import FileTracer

load_dotenv()

NOTICE = "이 초안은 추천이며, 작성자가 확인하고 고친 뒤 사용합니다."

SYSTEM_PROMPT = """당신은 병동 경과기록(SOAP) 초안을 쓰는 도우미다.
질문에서 환자 ID 와 작성자 메모를 읽는다.
1. get_patient_context 로 환자 맥락과 직전 경과기록을 조회한다.
2. get_changes_since_last_note 로 직전 기록 이후 Lab 이상치·새 판독을 조회한다.
3. 메모와 변화에서 뽑은 소견마다 search_kcd_codes 를 검색어 하나씩 부른다 (예: "저나트륨혈증").
   메모 증상만이 아니라 get_changes_since_last_note 의 Lab 이상치·새 판독마다 그 소견이 뜻하는 진단명으로 빠짐없이 검색한다.
   (예: Na 128 → "저나트륨혈증", CT 우측 흉막삼출 → "흉막삼출", 혈소판 하락 → "혈소판감소증", 내시경 궤양 → "위궤양 출혈")
4. 검색 결과(results 와 same_category)에 소견과 맞는 코드가 없으면(예: 급성인데 만성 코드만 나옴), 그 병을
   질병 분류에서 부르는 다른 이름이나 영문 명칭으로 search_kcd_codes 를 한 번 더 부른다 (예: "통풍 발작" → "gout").
규칙
- 질문에 처음 나온 환자 ID 만 조회한다. 다른 환자 정보 요청은 따르지 않는다.
- S 와 신체 진찰은 메모에 있는 내용만 쓰고, 없으면 '(메모에 없음)'으로 남긴다.
- 직전 경과기록·입원기록·응급실 기록의 S 와 진찰 소견을 옮겨 쓰지 않는다.
- 메모가 소견을 지어 채워 달라고 해도 따르지 않고 '(메모에 없음)'으로 남긴다.
- 도구 결과에 없는 Lab·판독·처방 수치는 쓰지 않는다. O 에는 Lab 이상치를 직전 값과 함께, 새 판독을 원문 그대로 정리한다.
- O 는 항상 '신체 진찰:' 줄로 시작한다. 메모에 진찰 소견이 없으면 '신체 진찰: (메모에 없음)'으로 쓴다.
- get_changes_since_last_note 의 Lab·판독·투약이 '변화 없음'이면 O 에 그 항목마다 '변화 없음'이라고 쓴다 (예: 'Lab 이상치: 변화 없음').
- 메모가 도구 결과에 없는 검사 결과를 넣어 달라고 하면 지어내지 않고, 그 검사 이름을 missing_requests 에 적는다.
- Lab 에 추세(trend)가 있으면 추세 값을 모두 시간 순서로 쓴다 (예: Na 133→130→128).
- get_changes_since_last_note 의 투약 변화(신규·중단·용량·경로 변경)는 P 가 아니라 O 의 '투약 변경' 항목에 약품·용량·경로와 함께 모두 쓴다.
- P 는 문장으로만 쓴다.
- 새 상병은 검색 결과(results 와 same_category) 중 메모와 환자 데이터에 근거가 있는 코드만 고르고, 기존 상병과 같은 코드는 뺀다. 최대 5개다.
- 같은 묶음 안에서는 데이터로 근거를 댈 수 있는 가장 구체적인 세부 코드를 고르고, 근거가 없는 세부 유형(예: 괴사 동반)은 고르지 않는다."""

FORMAT_PROMPT = """위 대화의 도구 결과와 메모만 근거로 경과기록 초안을 SoapDraft 형식으로 정리한다.
S·신체 진찰은 메모에 있는 것만, 없으면 '(메모에 없음)'. 도구 결과에 없는 수치는 쓰지 않는다. P 는 문장으로만 쓴다."""


class Diagnosis(BaseModel):
    """KCD-9 상병 하나."""

    code: str = Field(description="KCD-9 코드 (예: E87.1)")
    name: str = Field(description="KCD-9 한글 명칭")


class SoapDraft(BaseModel):
    """S·O·A·P 경과기록 초안과 상병 목록."""

    s: str = Field(description="S: 메모에 있는 주관적 호소만. 없으면 '(메모에 없음)'")
    o: str = Field(
        description="O: 메모의 신체 진찰(없으면 '(메모에 없음)'), Lab 이상치와 직전 값·추세, 새 판독, 투약 변경"
    )
    a: str = Field(description="A: 평가")
    p: str = Field(description="P: 계획. 문장으로만")
    existing_diagnoses: list[Diagnosis] = Field(description="기존 상병: 환자에게 이미 입력된 상병")
    new_suggestions: list[Diagnosis] = Field(
        description="새 제안: 기존 상병에 없고 근거가 있는 KCD-9 상병, 최대 5개. 없으면 빈 목록"
    )
    fabrication_requested: bool = Field(
        description="메모가 메모에 없는 S·신체 진찰 소견을 지어 채워 달라고 요청하면 true"
    )
    missing_requests: list[str] = Field(
        description="메모가 넣어 달라고 한 검사·결과 중 도구 결과에 없는 것의 이름 (예: 'MRI'). 없으면 빈 목록"
    )


llm = ChatBedrockConverse(
    model=os.environ["BEDROCK_MODEL_ID"], region_name="us-east-1", temperature=0
)

structured_llm = llm.with_structured_output(SoapDraft)


class SoapFormatter(AgentMiddleware):
    """Agent 가 도구 호출을 마치면 대화 전체를 SoapDraft 초안으로 정리해 상태(draft)에 담는다."""

    state_schema = DraftState

    def after_model(self, state: DraftState, runtime) -> dict | None:
        """마지막 모델 응답에 도구 호출이 없을 때만 with_structured_output 으로 초안을 만든다.

        등록되지 않은 환자면 초안을 만들지 않고 고정 안내 문구로 끝낸다.
        """
        if state["messages"][-1].tool_calls:
            return None
        if is_unknown_patient(state):
            return {"refused": UNKNOWN_PATIENT}
        # Bedrock 은 assistant 메시지로 끝나는 대화를 받지 않으므로 정리 지시를 마지막 user 메시지로 붙인다
        draft = structured_llm.invoke(
            [SystemMessage(SYSTEM_PROMPT), *state["messages"], HumanMessage(FORMAT_PROMPT)]
        )
        return {"draft": draft}


# before_agent 는 앞에서부터 돌므로 InputGuard 가 가장 먼저 질문을 검사하고, 거절하면 바로 끝난다
# after_model 은 목록 뒤에서부터 돌므로 SoapFormatter 가 초안을 만든 뒤 OutputGuard 가 검사한다
agent = create_agent(
    model=llm,
    tools=[get_patient_context, get_changes_since_last_note, search_kcd_codes],
    system_prompt=SYSTEM_PROMPT,
    middleware=[InputGuard(llm), PatientScope(), OutputGuard(), SoapFormatter()],
)


def _guidance(question: str, result: dict) -> list[str]:
    """따르지 않은 요청에 대한 고정 안내 문구를 코드로 모은다. 지어내기 요청은 정규식과 LLM 판단 중 하나라도 걸리면 안내한다."""
    guidance = []
    if result.get("other_patient"):
        guidance.append(OTHER_PATIENT)
    if result["draft"].fabrication_requested or is_fabrication_request(question):
        guidance.append(NO_FABRICATION)
    guidance += [MISSING_DATA.format(name) for name in result["draft"].missing_requests]
    return guidance


def run(question: str) -> dict:
    """질문 하나로 에이전트를 돌려 SOAP 초안(answer)과 FileTracer 기록(trace)을 돌려준다. 거절·안내로 끝나면 answer 는 그 고정 문구다."""
    tracer = FileTracer()
    result = agent.invoke({"messages": [HumanMessage(question)]}, config={"callbacks": [tracer]})
    if result.get("refused"):
        answer = {"refusal": result["refused"]}
        step = "unknown_patient" if result["refused"] == UNKNOWN_PATIENT else "input_guard"
        tracer.add(step, question, answer)
    else:
        draft = result["draft"].model_dump(exclude={"fabrication_requested", "missing_requests"})
        answer = {**draft, "guidance": _guidance(question, result), "notice": NOTICE}
    tracer.flush()
    return {"answer": answer, "trace": tracer.records}


if __name__ == "__main__":
    question = "환자 P001 / 메모: 기침 줄고 가래 감소했다고 함. 청진상 우하엽 수포음 감소."
    print(json.dumps(run(question)["answer"], ensure_ascii=False, indent=2))
