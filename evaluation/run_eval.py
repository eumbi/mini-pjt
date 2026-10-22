"""인-아웃 세트 평가. 20건을 /query 경로로 돌리고, 도구 호출·상병 재현율·자동 가드레일 위반을 코드로 검사한다.

실행: 프로젝트 루트에서 python -m evaluation.run_eval [--ids 1,13] [--reuse]
"""

import argparse
import csv
import json
import re
from pathlib import Path

from src.guards import KCD_CODES, MAX_SUGGESTIONS, find_patient_ids
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
    """케이스를 돌리거나(--reuse 면 저장된 결과를 읽어) 케이스별 검사표와 요약을 출력한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", help="돌릴 케이스 id, 쉼표 구분 (예: 1,13)")
    parser.add_argument("--reuse", action="store_true", help="Agent 를 다시 돌리지 않고 results.json 으로 검사만 한다")
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

    total = exact = three = base_exact = base_three = 0
    tools_ok = violation_count = 0
    print(f"\n{'id':>3} {'category':<10} {'도구':<4} {'완전':>5} {'세자리':>6}  위반")
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

        tool_mark = "OK" if tool_pass else "MISS"
        codes = f"{hit_exact}/{len(expected)}", f"{hit_three}/{len(expected)}"
        print(
            f"{case['id']:>3} {case['category']:<10} {tool_mark:<4} {codes[0]:>5} {codes[1]:>6}  "
            + ("; ".join(violations) or "-")
        )
        if not tool_pass:
            print(f"    기대 도구 {case['expected_tools']} / 호출 {called_tools(result)}")

    recall = exact / total if total else 0
    print("\n== 요약 ==")
    print(f"도구 호출 일치: {tools_ok}/{len(cases)}")
    print(
        f"상병 재현율(완전 일치):  {percent(exact, total)} (기준선 {percent(base_exact, total)})"
        f"  목표 80% {'달성' if recall >= TARGET_RECALL else '미달'}"
    )
    print(f"상병 재현율(앞 세 자리): {percent(three, total)} (기준선 {percent(base_three, total)})")
    print(f"자동 가드레일 위반: {violation_count}건  목표 0건 {'달성' if violation_count == 0 else '미달'}")


if __name__ == "__main__":
    main()
