"""도메인 도구. data/patients/ 의 JSON 더미 데이터를 조회만 하고 LLM 을 부르지 않는다."""

import json
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from src.retriever import same_category, search

PATIENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "patients"


def _load_patient(patient_id: str) -> dict:
    """환자 ID 에 해당하는 JSON 파일을 읽어 dict 로 돌려준다."""
    path = PATIENTS_DIR / f"{patient_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@tool
def get_patient_context(patient_id: str) -> dict | str:
    """환자 ID 로 경과기록 작성의 출발점이 되는 환자 맥락을 조회한다.

    나이·성별, 입원 시각, 기저질환, 알레르기, 현재 입력된 상병, 출발 기록 본문을 돌려준다.
    출발 기록은 직전 경과기록이고, 없으면(입원 첫날) 입원기록, 입원기록도 없으면 응급실 기록이다.
    start_note.source 는 어느 기록에서 출발했는지를 나타낸다
    (progress_note: 직전 경과기록, admission_note: 입원기록, er_note: 응급실 기록).
    등록되지 않은 환자 ID 면 '데이터 없음: 등록되지 않은 환자'를 돌려준다.
    """
    if not (PATIENTS_DIR / f"{patient_id}.json").exists():
        return "데이터 없음: 등록되지 않은 환자"
    patient = _load_patient(patient_id)
    if patient["progress_notes"]:
        source = "progress_note"
        last_note = max(patient["progress_notes"], key=lambda note: note["written_at"])
    elif patient.get("admission_note"):
        source, last_note = "admission_note", patient["admission_note"]
    else:
        source, last_note = "er_note", patient["er_note"]
    return {
        "patient_id": patient["patient_id"],
        "age": patient["age"],
        "sex": patient["sex"],
        "admitted_at": patient["admitted_at"],
        "underlying_diseases": patient["underlying_diseases"],
        "allergies": patient["allergies"],
        "diagnoses": patient["diagnoses"],
        "start_note": {
            "source": source,
            "written_at": last_note["written_at"],
            "text": last_note["text"],
        },
    }


def _parse(timestamp: str) -> datetime:
    """ISO 형식 시각 문자열을 datetime 으로 바꾼다."""
    return datetime.fromisoformat(timestamp)


@tool
def get_changes_since_last_note(patient_id: str) -> dict | str:
    """환자 ID 로 직전 경과기록 이후의 변화를 조회한다.

    기준 시각(baseline_at)은 직전 경과기록의 작성 시각이고, 경과기록이 없으면 입원 시각이다.
    baseline_source 는 기준 시각의 출처다 (progress_note: 직전 경과기록, admitted_at: 입원 시각).
    labs: 기준 시각 이후 결과 중 참고치(하한·상한)를 벗어난 항목 전부. previous 에 같은 검사의 직전 결과를 붙인다.
      같은 검사가 이번 입원 중 3회 이상이면 trend 에 그 결과까지의 값을 시간 순서대로 나열하고, 아니면 trend 는 None 이다.
    readings: 기준 시각 이후 나온 판독의 검사명, 판독 시각, 판독 소견 원문.
    medications: 직전 처방(기준 시각에 유효한 처방)과 현재 처방(중단되지 않은 처방)을 약품별로 비교해
      new(신규), stopped(중단), changed(용량·경로·빈도 변경, before/after)로 나눈다.
    세 항목 중 변화가 없는 항목은 '변화 없음'으로 표시한다.
    등록되지 않은 환자 ID 면 '데이터 없음: 등록되지 않은 환자'를 돌려준다.
    """
    if not (PATIENTS_DIR / f"{patient_id}.json").exists():
        return "데이터 없음: 등록되지 않은 환자"
    patient = _load_patient(patient_id)
    if patient["progress_notes"]:
        baseline_source = "progress_note"
        baseline_at = max(note["written_at"] for note in patient["progress_notes"])
    else:
        baseline_source, baseline_at = "admitted_at", patient["admitted_at"]
    baseline = _parse(baseline_at)

    labs = sorted(patient["labs"], key=lambda lab: _parse(lab["measured_at"]))
    abnormal_labs = []
    for index, lab in enumerate(labs):
        if _parse(lab["measured_at"]) <= baseline:
            continue
        if lab["ref_low"] <= lab["value"] <= lab["ref_high"]:
            continue
        same = [prev for prev in labs[: index + 1] if prev["name"] == lab["name"]]
        earlier = same[-2] if len(same) >= 2 else None
        previous = (
            {"value": earlier["value"], "measured_at": earlier["measured_at"]} if earlier else None
        )
        trend = (
            [{"value": item["value"], "measured_at": item["measured_at"]} for item in same]
            if len(same) >= 3
            else None
        )
        abnormal_labs.append({**lab, "previous": previous, "trend": trend})

    new_readings = [
        {"exam": reading["exam"], "read_at": reading["read_at"], "finding": reading["finding"]}
        for reading in patient["readings"]
        if _parse(reading["read_at"]) > baseline
    ]

    def _order(rx: dict) -> dict:
        """처방 한 행에서 비교에 쓰는 필드만 남긴다."""
        return {key: rx[key] for key in ("drug", "dose", "route", "frequency")}

    before = {
        rx["drug"]: _order(rx)
        for rx in patient["prescriptions"]
        if _parse(rx["started_at"]) <= baseline
        and (rx["stopped_at"] is None or _parse(rx["stopped_at"]) > baseline)
    }
    current = {rx["drug"]: _order(rx) for rx in patient["prescriptions"] if rx["stopped_at"] is None}
    new = [current[drug] for drug in current if drug not in before]
    stopped = [before[drug] for drug in before if drug not in current]
    changed = [
        {"drug": drug, "before": before[drug], "after": current[drug]}
        for drug in current
        if drug in before and before[drug] != current[drug]
    ]
    medications = {"new": new, "stopped": stopped, "changed": changed}

    return {
        "patient_id": patient["patient_id"],
        "baseline_at": baseline_at,
        "baseline_source": baseline_source,
        "labs": abnormal_labs or "변화 없음",
        "readings": new_readings or "변화 없음",
        "medications": medications if new or stopped or changed else "변화 없음",
    }


@tool
def search_kcd_codes(query: str) -> dict | str:
    """소견 검색어 하나로 KCD-9 후보 코드를 검색한다.

    검색 결과 상위 10건(results)과, 상위 3건과 앞 세 자리가 같은 나머지 코드(same_category)를
    코드, 한글 명칭, 출처(KCD/{코드})로 돌려주기만 하고 고르지 않는다.
    어떤 코드를 쓸지는 Agent 가 판단한다. 결과가 없으면 '검색 결과 없음'을 돌려준다.
    """
    docs = search(query)
    if not docs:
        return "검색 결과 없음"

    def entry(doc) -> dict:
        return {
            "code": doc.metadata["code"],
            "name": doc.metadata["name_ko"],
            "source": f"KCD/{doc.metadata['code']}",
        }

    return {
        "query": query,
        "results": [entry(doc) for doc in docs],
        "same_category": [entry(doc) for doc in same_category(docs)],
    }
