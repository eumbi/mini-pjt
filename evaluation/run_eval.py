"""인-아웃 세트 평가. 20건을 /query 경로로 돌리고, 도구 호출·상병 재현율·자동 가드레일 위반을 코드로 검사한다.
코드로 볼 수 없는 expected_traits·forbidden 항목, O 반영률, S·신체 진찰 지어냄은 LLM-as-Judge 로 판정한다.
RAGAS 4개 지표는 호출이 많아 라운드 리포트를 만들 때(--report)만 계산한다.

실행: 프로젝트 루트에서 python -m evaluation.run_eval [--ids 1,13] [--reuse]
"""

import argparse
import csv
import json
import os
import re
import sys
import types
from pathlib import Path

from langchain_aws import BedrockEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent import llm
from src.guards import KCD_CODES, MAX_SUGGESTIONS, find_patient_ids, memo_text
from src.server import QueryRequest, query

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "evaluation" / "test_queries.csv"
RESULTS_PATH = ROOT / "evaluation" / "results.json"
PATIENTS_DIR = ROOT / "data" / "patients"

LIST_FIELDS = ["expected_traits", "forbidden", "expected_tools"]
TOOL_NAMES = {"get_patient_context", "get_changes_since_last_note", "search_kcd_codes"}
EXPECTED_CODE = re.compile(r"상병 목록에 (\S+) 포함")
KCD_PATTERN = re.compile(r"\b[A-Z]\d{2}(?:\.\d+)?\b")
# 영문자 바로 뒤의 숫자(SpO2, mm3 등)는 수치가 아니므로 건너뛴다
NUMBER = re.compile(r"(?<![A-Za-z\d.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
# 날짜·시각(2026-10-20, 14:00)은 Lab·판독·처방 수치가 아니므로 숫자를 뽑기 전에 뺀다
DATE_TIME = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}:\d{2}")
SMALL_INT = 10
TARGET_RECALL = 0.8
TARGET_PASS = 16
TARGET_O_RATE = 0.7
RAGAS_METRICS = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
RAGAS_WORKERS = 4
# 일부러 넣은 Lab 변화·새 판독·투약 변경을 적은 trait. O 반영률의 분모다
O_TRAIT = re.compile(r"O에 .+ 반영")
# 코드로 검사하는 forbidden 항목과 check_guardrails 위반 문구의 앞부분
CODE_FORBIDDEN = {
    "새 상병 제안 6개 이상": "새 제안",
    "KCD-9에 없는 코드": "KCD-9 에 없는 코드",
    "데이터에 없는 Lab·판독·처방 수치": "데이터에 없는 수치",
}

JUDGE_PROMPT = """병동 경과기록(SOAP) 초안 도우미의 답변을 평가 항목별로 판정한다.
입력으로 질문, 작성자 메모, 도구가 가져온 환자 데이터, 답변(JSON)을 받는다.
- 답변에 refusal 이 있으면 초안 대신 그 안내 문구만 나간 것이다. guidance 는 답변과 함께 나간 안내 문구다.
- traits: 받은 기대 특성마다 항목 문구를 item 에 그대로 쓰고, 답변이 만족하면 ok=true.
  "O에 ... 반영"은 그 수치·변화·판독·투약 변경이 O 에 실제로 적혀 있어야 만족이다. 수치는 같아야 하고 표기(화살표, 단위)는 달라도 된다.
- forbidden: 받은 금지 항목마다 item 에 그대로 쓰고, 답변에 그 내용이 나오지 않았으면 ok=true, 나왔으면 ok=false.
- reason: 판정 근거를 답변에서 찾은 문구와 함께 한 문장으로 쓴다.
- fabricated: 답변의 S 와 O 의 신체 진찰 부분에 나온 소견 중 작성자 메모에 없는 것을 모두 적는다.
  메모를 표준 문체로 바꿔 쓴 것은 지어낸 것이 아니다. '(메모에 없음)'은 지어낸 것이 아니다.
  Lab·판독·처방과 활력징후처럼 환자 데이터에 있는 것은 신체 진찰이 아니므로 넣지 않는다. 없으면 빈 목록."""


class ItemVerdict(BaseModel):
    """평가 항목 하나의 판정."""

    item: str = Field(description="평가 항목 문구 그대로")
    ok: bool = Field(description="trait 는 만족하면 true, forbidden 은 나오지 않았으면 true")
    reason: str = Field(description="판정 근거 한 문장")


class CaseVerdict(BaseModel):
    """케이스 하나에 대한 LLM 판정 결과."""

    traits: list[ItemVerdict] = Field(description="받은 기대 특성마다 하나씩")
    forbidden: list[ItemVerdict] = Field(description="받은 금지 항목마다 하나씩")
    fabricated: list[str] = Field(description="S·신체 진찰에 나온 소견 중 메모에 없는 것")


judge_llm = llm.with_structured_output(CaseVerdict)


def load_cases() -> list[dict]:
    """test_queries.csv 를 읽어 세미콜론으로 구분된 필드를 목록으로 쪼갠 케이스 목록을 돌려준다."""
    with CASES_PATH.open(encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    for case in cases:
        for field in LIST_FIELDS:
            case[field] = [item.strip() for item in case[field].split(";") if item.strip()]
    return cases


def run_case(question: str) -> dict:
    """POST /query 와 같은 함수를 직접 불러 answer·contexts·trace 를 돌려준다."""
    return query(QueryRequest(question=question))


def called_tools(result: dict) -> list[str]:
    """trace 에서 도구 호출 단계 이름만 호출 순서대로 뽑는다."""
    return [record["step"] for record in result["trace"] if record["step"] in TOOL_NAMES]


def check_tools(case: dict, result: dict) -> bool:
    """기대 도구를 모두 불렀는지 본다. 기대 도구가 없으면 도구를 하나도 부르지 않아야 통과다."""
    called = set(called_tools(result))
    if not case["expected_tools"]:
        return not called
    return set(case["expected_tools"]) <= called


def expected_codes(case: dict) -> list[str]:
    """expected_traits 의 '상병 목록에 {코드} 포함' 항목에서 정답 코드를 뽑는다."""
    return [m.group(1) for trait in case["expected_traits"] if (m := EXPECTED_CODE.fullmatch(trait))]


def predicted_codes(answer: dict) -> list[str]:
    """답변의 기존 상병과 새 제안 코드를 모은다. 거절 답변이면 빈 목록이다."""
    if "refusal" in answer:
        return []
    return [d["code"] for d in answer["existing_diagnoses"] + answer["new_suggestions"]]


def load_patient(question: str) -> dict | None:
    """질문의 첫 환자 ID 로 더미 환자 JSON 을 읽는다. ID 가 없거나 파일이 없으면 None 이다."""
    patient_ids = find_patient_ids(question)
    if not patient_ids:
        return None
    path = PATIENTS_DIR / f"{patient_ids[0]}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_codes(question: str) -> list[str]:
    """기준선: 환자 데이터에 이미 입력된 기존 상병 코드만 돌려준다."""
    patient = load_patient(question)
    return [d["code"] for d in patient["diagnoses"]] if patient else []


def count_hits(expected: list[str], predicted: list[str]) -> tuple[int, int]:
    """정답 코드 중 완전 일치한 개수와 앞 세 자리가 일치한 개수를 센다."""
    exact = sum(code in predicted for code in expected)
    prefixes = {code[:3] for code in predicted}
    three = sum(code[:3] in prefixes for code in expected)
    return exact, three


def numbers_in(text: str) -> set[str]:
    """글에서 날짜·시각을 뺀 숫자를 뽑아 쉼표를 빼고 130.0 → 130 처럼 같은 꼴로 맞춘다."""
    found = set()
    for whole, decimal in NUMBER.findall(DATE_TIME.sub(" ", text)):
        value = float(whole.replace(",", "") + decimal)
        found.add(str(int(value)) if value.is_integer() else str(value))
    return found


def unsupported_numbers(question: str, answer: dict) -> list[str]:
    """O 에 나온 숫자 중 환자 데이터와 질문 어디에도 없는 것을 돌려준다. KCD 코드와 10 이하 정수는 빼고 본다.

    Lab·판독·처방 수치는 O 에 정리되므로 O 만 본다. A·P 의 계산 비율이나 치료 목표치는 검사하지 않는다.
    """
    patient = load_patient(question)
    source = json.dumps(patient, ensure_ascii=False) if patient else ""
    allowed = numbers_in(source) | numbers_in(question)
    text = KCD_PATTERN.sub(" ", answer["o"])
    used = numbers_in(text)
    small = {str(n) for n in range(SMALL_INT + 1)}
    return sorted(used - allowed - small, key=float)


def check_guardrails(case: dict, answer: dict) -> list[str]:
    """새 제안 개수, KCD-9 코드 여부, 데이터에 없는 수치를 검사해 위반 내용을 돌려준다."""
    if "refusal" in answer:
        return []
    violations = []
    if len(answer["new_suggestions"]) > MAX_SUGGESTIONS:
        violations.append(f"새 제안 {len(answer['new_suggestions'])}개")
    unknown = [code for code in predicted_codes(answer) if code not in KCD_CODES]
    if unknown:
        violations.append("KCD-9 에 없는 코드 " + ", ".join(unknown))
    numbers = unsupported_numbers(case["input"], answer)
    if numbers:
        violations.append("데이터에 없는 수치 " + ", ".join(numbers))
    return violations


def judge_items(case: dict) -> tuple[list[str], list[str]]:
    """LLM 으로 판정할 trait 와 forbidden 항목을 고른다. 상병 코드 포함과 자동 가드레일 항목은 코드로 보므로 뺀다."""
    traits = [t for t in case["expected_traits"] if not EXPECTED_CODE.fullmatch(t)]
    forbidden = [f for f in case["forbidden"] if f not in CODE_FORBIDDEN]
    return traits, forbidden


def judge_case(case: dict, result: dict) -> dict:
    """질문·메모·환자 데이터·답변을 LLM 판정기에 주고 항목별 판정과 지어낸 소견을 돌려준다."""
    traits, forbidden = judge_items(case)
    patient_data = "\n".join(
        f"[{c['doc_id']}] {c['text']}" for c in result["contexts"] if not c["doc_id"].startswith("KCD/")
    )
    content = (
        f"질문: {case['input']}\n"
        f"작성자 메모: {memo_text(case['input']) or '(없음)'}\n"
        f"환자 데이터:\n{patient_data or '(없음)'}\n"
        f"답변:\n{json.dumps(result['answer'], ensure_ascii=False, indent=2)}\n\n"
        "traits:\n" + "\n".join(f"- {t}" for t in traits) + "\n"
        "forbidden:\n" + "\n".join(f"- {f}" for f in forbidden)
    )
    verdict = judge_llm.invoke([SystemMessage(JUDGE_PROMPT), HumanMessage(content)])
    return verdict.model_dump()


def merge_verdict(case: dict, result: dict, violations: list[str]) -> list[str]:
    """코드 검사와 LLM 판정을 합쳐 실패 항목을 '[항목] 이유' 꼴로 돌려준다. 빈 목록이면 PASS 다."""
    failed = []
    predicted = predicted_codes(result["answer"])
    for code in expected_codes(case):
        if code not in predicted:
            failed.append(f"[상병 목록에 {code} 포함] 기존 상병·새 제안에 없음 (코드)")
    for item, prefix in CODE_FORBIDDEN.items():
        hits = [v for v in violations if v.startswith(prefix)]
        if item in case["forbidden"] and hits:
            failed.append(f"[금지: {item}] {'; '.join(hits)} (코드)")
    judge = result["judge"]
    traits, forbidden = judge_items(case)
    verdicts = {v["item"]: v for v in judge["traits"]}
    for trait in traits:
        v = verdicts.get(trait)
        if not v or not v["ok"]:
            failed.append(f"[{trait}] {v['reason'] if v else '판정 없음'}")
    verdicts = {v["item"]: v for v in judge["forbidden"]}
    for item in forbidden:
        v = verdicts.get(item)
        if not v or not v["ok"]:
            failed.append(f"[금지: {item}] {v['reason'] if v else '판정 없음'}")
    return failed


def o_reflection(case: dict, judge: dict) -> tuple[int, int]:
    """'O에 ... 반영' trait 중 LLM 이 만족으로 본 개수와 전체 개수를 센다."""
    targets = [t for t in case["expected_traits"] if O_TRAIT.fullmatch(t)]
    ok = {v["item"] for v in judge["traits"] if v["ok"]}
    return sum(t in ok for t in targets), len(targets)


def answer_text(answer: dict) -> str:
    """answer 를 S·O·A·P, 기존 상병, 새 제안, 고정 문구 순서의 글 하나로 편다. RAGAS 의 response 로 쓴다."""
    def codes(diagnoses: list[dict]) -> str:
        return ", ".join(f"{d['code']} {d['name']}" for d in diagnoses) or "없음"

    return "\n".join([
        f"S: {answer['s']}", f"O: {answer['o']}", f"A: {answer['a']}", f"P: {answer['p']}",
        f"기존 상병: {codes(answer['existing_diagnoses'])}",
        f"새 제안: {codes(answer['new_suggestions'])}",
        *answer["guidance"], answer["notice"],
    ])


def import_ragas():
    """ragas 를 불러온다. ragas 0.4.3 은 langchain-community 0.4 에서 빠진 VertexAI 채팅 모듈을
    isinstance 검사에만 쓰므로, 빈 클래스를 담은 대체 모듈을 먼저 등록한다."""
    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = module
    import ragas
    from ragas import metrics

    return ragas, metrics


def run_ragas(cases: list[dict], results: dict) -> dict[str, float]:
    """거절이 아니고 contexts 가 있는 케이스로 RAGAS 4개 지표를 계산해 케이스별 점수를 results 에 넣고 평균을 돌려준다.

    reference 는 정답 문서가 없으므로 expected_traits 를 이어 붙인 글로 쓴다.
    """
    ragas, metrics = import_ragas()
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    targets = [c for c in cases if "refusal" not in results[c["id"]]["answer"] and results[c["id"]]["contexts"]]
    dataset = ragas.EvaluationDataset.from_list([
        {
            "user_input": case["input"],
            "retrieved_contexts": [c["text"] for c in results[case["id"]]["contexts"]],
            "response": answer_text(results[case["id"]]["answer"]),
            "reference": "; ".join(case["expected_traits"]),
        }
        for case in targets
    ])
    embeddings = BedrockEmbeddings(model_id=os.environ["BEDROCK_EMBEDDING_MODEL_ID"], region_name="us-east-1")
    scores = ragas.evaluate(
        dataset,
        metrics=[getattr(metrics, name) for name in RAGAS_METRICS],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=ragas.RunConfig(max_workers=RAGAS_WORKERS),
        show_progress=True,
    ).to_pandas()
    for case, (_, row) in zip(targets, scores.iterrows()):
        results[case["id"]]["ragas"] = {name: float(row[name]) for name in RAGAS_METRICS}
    return {name: float(scores[name].mean()) for name in RAGAS_METRICS}


def load_results() -> dict:
    """저장된 실행 결과를 케이스 id 를 키로 읽는다. 파일이 없으면 빈 사전이다."""
    if not RESULTS_PATH.exists():
        return {}
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def save_results(results: dict) -> None:
    """실행 결과를 evaluation/results.json 에 id 순서로 저장한다."""
    ordered = dict(sorted(results.items(), key=lambda item: int(item[0])))
    RESULTS_PATH.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")


def percent(numerator: int, denominator: int) -> str:
    """'n/m = xx.x%' 문자열을 만든다."""
    rate = numerator / denominator if denominator else 0
    return f"{numerator}/{denominator} = {rate:.1%}"


def main() -> None:
    """케이스를 돌리거나(--reuse 면 저장된 결과를 읽어) LLM 판정을 붙이고, 케이스별 검사표와 요약을 출력한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", help="돌릴 케이스 id, 쉼표 구분 (예: 1,13)")
    parser.add_argument("--reuse", action="store_true", help="Agent 를 다시 돌리지 않고 results.json 으로 검사만 한다")
    parser.add_argument("--report", help="라운드 리포트 경로. 주면 RAGAS 4개 지표도 계산한다")
    args = parser.parse_args()

    cases = load_cases()
    if args.ids:
        ids = set(args.ids.split(","))
        cases = [case for case in cases if case["id"] in ids]

    results = load_results()
    for case in cases:
        if args.reuse:
            if case["id"] not in results:
                raise SystemExit(f"{case['id']}번 결과가 results.json 에 없습니다. --reuse 없이 먼저 실행하세요.")
            continue
        print(f"[{case['id']}] 실행 중...", flush=True)
        results[case["id"]] = {"question": case["input"], **run_case(case["input"])}
        save_results(results)

    for case in cases:
        print(f"[{case['id']}] 판정 중...", flush=True)
        results[case["id"]]["judge"] = judge_case(case, results[case["id"]])
    save_results(results)

    total = exact = three = base_exact = base_three = 0
    tools_ok = violation_count = passed = o_ok = o_total = 0
    fabrications = []
    print(f"\n{'id':>3} {'category':<10} {'판정':<4} {'도구':<4} {'완전':>5} {'세자리':>6}  위반")
    for case in cases:
        result = results[case["id"]]
        expected = expected_codes(case)
        hit_exact, hit_three = count_hits(expected, predicted_codes(result["answer"]))
        b_exact, b_three = count_hits(expected, baseline_codes(case["input"]))
        tool_pass = check_tools(case, result)
        violations = check_guardrails(case, result["answer"])

        total += len(expected)
        exact, three = exact + hit_exact, three + hit_three
        base_exact, base_three = base_exact + b_exact, base_three + b_three
        tools_ok += tool_pass
        violation_count += len(violations)
        failed = merge_verdict(case, result, violations)
        passed += not failed
        o_hit, o_count = o_reflection(case, result["judge"])
        o_ok, o_total = o_ok + o_hit, o_total + o_count
        if "refusal" not in result["answer"]:
            fabrications += [f"{case['id']}번: {item}" for item in result["judge"]["fabricated"]]

        verdict_mark = "FAIL" if failed else "PASS"
        tool_mark = "OK" if tool_pass else "MISS"
        codes = f"{hit_exact}/{len(expected)}", f"{hit_three}/{len(expected)}"
        print(
            f"{case['id']:>3} {case['category']:<10} {verdict_mark:<4} {tool_mark:<4} "
            f"{codes[0]:>5} {codes[1]:>6}  " + ("; ".join(violations) or "-")
        )
        if not tool_pass:
            print(f"    기대 도구 {case['expected_tools']} / 호출 {called_tools(result)}")
        for item in failed:
            print(f"    - {item}")

    recall = exact / total if total else 0
    o_rate = o_ok / o_total if o_total else 0
    print("\n== 요약 ==")
    print(f"케이스 통과: {passed}/{len(cases)} 통과  목표 {TARGET_PASS}건 {'달성' if passed >= TARGET_PASS else '미달'}")
    print(f"도구 호출 일치: {tools_ok}/{len(cases)}")
    print(
        f"상병 재현율(완전 일치):  {percent(exact, total)} (기준선 {percent(base_exact, total)})"
        f"  목표 80% {'달성' if recall >= TARGET_RECALL else '미달'}"
    )
    print(f"상병 재현율(앞 세 자리): {percent(three, total)} (기준선 {percent(base_three, total)})")
    print(f"자동 가드레일 위반: {violation_count}건  목표 0건 {'달성' if violation_count == 0 else '미달'}")
    print(f"O 반영률: {percent(o_ok, o_total)}  목표 70% {'달성' if o_rate >= TARGET_O_RATE else '미달'}")
    print(f"S·신체 진찰 지어냄: {len(fabrications)}건  목표 0건 {'달성' if not fabrications else '미달'}")
    for item in fabrications:
        print(f"    - {item}")

    if args.report:
        print("\nRAGAS 계산 중...", flush=True)
        ragas_means = run_ragas(cases, results)
        save_results(results)
        print("RAGAS: " + "  ".join(f"{name} {value:.3f}" for name, value in ragas_means.items()))


if __name__ == "__main__":
    main()
