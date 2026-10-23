"""트레이스 콜백. 도구 호출과 LLM 호출을 시작 순서대로 기록하고 JSONL 파일에 남긴다.

응답 trace 는 {step, input, output} 만 쓰고, 지연 시간(latency_s)과 LLM 토큰 수(usage)는 JSONL 에만 남긴다.
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent.parent / "trace.jsonl"


def _parse_output(content: Any) -> Any:
    """도구 결과 문자열을 JSON 으로 풀고, 풀리지 않으면 그대로 돌려준다."""
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return content


class FileTracer(BaseCallbackHandler):
    """질문 한 번 동안의 도구·LLM 호출을 {step, input, output} 목록으로 모으고 JSONL 로 쓰는 콜백."""

    def __init__(self, path: str | Path | None = None):
        """기록 경로(.env 의 TRACE_PATH, 없으면 프로젝트 루트 trace.jsonl)와 빈 기록을 준비한다."""
        self.path = Path(path or os.environ.get("TRACE_PATH") or DEFAULT_TRACE_PATH)
        self.trace_id = str(uuid.uuid4())
        self.records: list[dict] = []
        self.metrics: list[dict] = []
        self._index: dict[uuid.UUID, int] = {}
        self._started: dict[uuid.UUID, float] = {}

    def _start(self, run_id: uuid.UUID, step: str, input: Any) -> None:
        """호출이 시작될 때 기록에 자리를 잡아 시작 순서를 지키고, 시작 시각을 잰다."""
        self._index[run_id] = len(self.records)
        self._started[run_id] = time.perf_counter()
        self.records.append({"step": step, "input": input, "output": None})
        self.metrics.append({})

    def _end(self, run_id: uuid.UUID, output: Any, usage: dict | None = None) -> None:
        """호출이 끝나면 시작 때 잡은 자리에 출력을 채우고, 지연 시간과 토큰 수를 따로 적는다."""
        if run_id not in self._index:
            return
        index = self._index.pop(run_id)
        self.records[index]["output"] = output
        self.metrics[index]["latency_s"] = round(time.perf_counter() - self._started.pop(run_id), 2)
        if usage:
            self.metrics[index]["usage"] = usage

    def add(self, step: str, input: Any, output: Any) -> None:
        """콜백으로 잡히지 않는 단계(거절 등)를 같은 기록에 붙인다."""
        self.records.append({"step": step, "input": input, "output": output})
        self.metrics.append({})

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs) -> None:
        """도구 이름을 step 으로, 도구 인자를 input 으로 기록한다."""
        name = (serialized or {}).get("name") or kwargs.get("name", "tool")
        self._start(run_id, name, inputs if inputs is not None else input_str)

    def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        """도구 결과(ToolMessage 본문)를 JSON 으로 풀어 기록한다."""
        self._end(run_id, _parse_output(getattr(output, "content", output)))

    def on_tool_error(self, error, *, run_id, **kwargs) -> None:
        """도구 오류를 출력 자리에 기록한다."""
        self._end(run_id, {"error": str(error)})

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        """어느 그래프 노드의 LLM 호출인지 step 에 적고, 마지막 메시지만 input 으로 기록한다."""
        node = (metadata or {}).get("langgraph_node", "")
        last = messages[0][-1]
        self._start(run_id, f"llm:{node}", {"role": last.type, "content": last.content})

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        """LLM 응답의 본문과 도구 호출을 기록하고, 입력·출력 토큰 수(usage_metadata)를 따로 적는다."""
        message = response.generations[0][0].message
        self._end(
            run_id,
            {
                "text": message.text,
                "tool_calls": [{"name": c["name"], "args": c["args"]} for c in message.tool_calls],
            },
            usage=getattr(message, "usage_metadata", None),
        )

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        """LLM 오류를 출력 자리에 기록한다."""
        self._end(run_id, {"error": str(error)})

    def flush(self) -> None:
        """모은 기록을 trace_id·시각·지연 시간·토큰 수와 함께 JSONL 파일에 한 줄씩 덧붙인다."""
        now = datetime.now().isoformat(timespec="seconds")
        with self.path.open("a", encoding="utf-8") as f:
            for record, metric in zip(self.records, self.metrics):
                line = {"trace_id": self.trace_id, "time": now, **record, **metric}
                f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
