"""메인 에이전트 그래프. create_agent 로 도구를 부르고, 결과를 SOAP Pydantic 모델로 정리한다."""

import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.guards import DraftState, OutputGuard
from src.tools import get_changes_since_last_note, get_patient_context, search_kcd_codes

load_dotenv()

NOTICE = "이 초안은 추천이며, 작성자가 확인하고 고친 뒤 사용합니다."

SYSTEM_PROMPT = """당신은 병동 경과기록(SOAP) 초안을 쓰는 도우미다.
질문에서 환자 ID 와 작성자 메모를 읽는다.
1. get_patient_context 로 환자 맥락과 직전 경과기록을 조회한다.
2. get_changes_since_last_note 로 직전 기록 이후 Lab 이상치·새 판독을 조회한다.
3. 메모와 변화에서 뽑은 소견마다 search_kcd_codes 를 검색어 하나씩 부른다 (예: "저나트륨혈증").
규칙
- S 와 신체 진찰은 메모에 있는 내용만 쓰고, 없으면 '(메모에 없음)'으로 남긴다.
- 도구 결과에 없는 Lab·판독·처방 수치는 쓰지 않는다. O 에는 Lab 이상치를 직전 값과 함께, 새 판독을 원문 그대로 정리한다.
- P 는 문장으로만 쓴다.
- 새 상병은 검색 결과 중 메모와 환자 데이터에 근거가 있는 코드만 고르고, 기존 상병과 같은 코드는 뺀다. 최대 5개다."""

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
        description="O: 메모의 신체 진찰(없으면 '(메모에 없음)'), Lab 이상치와 직전 값, 새 판독"
    )
    a: str = Field(description="A: 평가")
    p: str = Field(description="P: 계획. 문장으로만")
    existing_diagnoses: list[Diagnosis] = Field(description="기존 상병: 환자에게 이미 입력된 상병")
    new_suggestions: list[Diagnosis] = Field(
        description="새 제안: 기존 상병에 없고 근거가 있는 KCD-9 상병, 최대 5개. 없으면 빈 목록"
    )


llm = ChatBedrockConverse(
    model=os.environ["BEDROCK_MODEL_ID"], region_name="us-east-1", temperature=0
)

structured_llm = llm.with_structured_output(SoapDraft)


class SoapFormatter(AgentMiddleware):
    """Agent 가 도구 호출을 마치면 대화 전체를 SoapDraft 초안으로 정리해 상태(draft)에 담는다."""

    state_schema = DraftState

    def after_model(self, state: DraftState, runtime) -> dict | None:
        """마지막 모델 응답에 도구 호출이 없을 때만 with_structured_output 으로 초안을 만든다."""
        if state["messages"][-1].tool_calls:
            return None
        # Bedrock 은 assistant 메시지로 끝나는 대화를 받지 않으므로 정리 지시를 마지막 user 메시지로 붙인다
        draft = structured_llm.invoke(
            [SystemMessage(SYSTEM_PROMPT), *state["messages"], HumanMessage(FORMAT_PROMPT)]
        )
        return {"draft": draft}


# after_model 은 목록 뒤에서부터 돌므로 SoapFormatter 가 초안을 만든 뒤 OutputGuard 가 검사한다
agent = create_agent(
    model=llm,
    tools=[get_patient_context, get_changes_since_last_note, search_kcd_codes],
    system_prompt=SYSTEM_PROMPT,
    middleware=[OutputGuard(), SoapFormatter()],
)


def run(question: str) -> dict:
    """질문 하나로 에이전트를 돌려 SOAP 초안(answer)과 대화 메시지 전체를 돌려준다."""
    result = agent.invoke({"messages": [HumanMessage(question)]})
    draft = result["draft"]
    return {
        "answer": {**draft.model_dump(), "notice": NOTICE},
        "messages": result["messages"],
    }


if __name__ == "__main__":
    question = "환자 P001 / 메모: 기침 줄고 가래 감소했다고 함. 청진상 우하엽 수포음 감소."
    print(json.dumps(run(question)["answer"], ensure_ascii=False, indent=2))
