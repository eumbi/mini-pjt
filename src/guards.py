"""출력 가드레일. 새 제안이 KCD-9 에 있는 코드인지, 5개 이하인지 after_model 미들웨어에서 코드로 검사한다."""

import csv
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from typing_extensions import NotRequired

KCD_PATH = Path(__file__).resolve().parent.parent / "data" / "kcd9.csv"
MAX_SUGGESTIONS = 5

with KCD_PATH.open(encoding="utf-8") as f:
    KCD_CODES = {row["code"] for row in csv.DictReader(f)}


class DraftState(AgentState):
    """Agent 상태에 SOAP 초안(draft)을 더한 상태 스키마."""

    draft: NotRequired[Any]


def check_suggestions(draft: Any) -> Any:
    """새 제안에서 KCD-9 에 없는 코드와 기존 상병·중복 코드를 빼고, 앞에서부터 5개까지만 남긴다."""
    seen = {d.code for d in draft.existing_diagnoses}
    kept = []
    for suggestion in draft.new_suggestions:
        if suggestion.code in KCD_CODES and suggestion.code not in seen:
            seen.add(suggestion.code)
            kept.append(suggestion)
    return draft.model_copy(update={"new_suggestions": kept[:MAX_SUGGESTIONS]})


class OutputGuard(AgentMiddleware):
    """SOAP 초안이 만들어진 뒤 새 제안 목록을 검사해 걸러내는 출력 가드레일."""

    state_schema = DraftState

    def after_model(self, state: DraftState, runtime) -> dict | None:
        """초안이 있으면 새 제안을 걸러낸 초안으로 바꾼다. 도구 호출 중이면 아무것도 하지 않는다."""
        draft = state.get("draft")
        if draft is None:
            return None
        return {"draft": check_suggestions(draft)}
