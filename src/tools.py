"""도메인 도구. data/patients/ 의 JSON 더미 데이터를 조회만 하고 LLM 을 부르지 않는다."""

import json
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from src.retriever import search

PATIENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "patients"


def _load_patient(patient_id: str) -> dict:
    """환자 ID 에 해당하는 JSON 파일을 읽어 dict 로 돌려준다."""
    path = PATIENTS_DIR / f"{patient_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@tool
def get_patient_context(patient_id: str) -> dict:
    """환자 ID 로 경과기록 작성의 출발점이 되는 환자 맥락을 조회한다.

    나이·성별, 입원 시각, 기저질환, 알레르기, 현재 입력된 상병, 직전 경과기록 본문을 돌려준다.
    start_note.source 는 어느 기록에서 출발했는지를 나타낸다 (progress_note: 직전 경과기록).
    """
    patient = _load_patient(patient_id)
    last_note = max(patient["progress_notes"], key=lambda note: note["written_at"])
    return {
        "patient_id": patient["patient_id"],
        "age": patient["age"],
        "sex": patient["sex"],
        "admitted_at": patient["admitted_at"],
        "underlying_diseases": patient["underlying_diseases"],
        "allergies": patient["allergies"],
        "diagnoses": patient["diagnoses"],
        "start_note": {
            "source": "progress_note",
            "written_at": last_note["written_at"],
            "text": last_note["text"],
        },
    }


@tool
def get_changes_since_last_note(patient_id: str) -> dict:
    """환자 ID 로 직전 경과기록 이후의 변화를 조회한다.

    기준 시각(baseline_at)은 직전 경과기록의 작성 시각이다.
    labs: 기준 시각 이후 결과 중 참고치(하한·상한)를 벗어난 항목 전부. previous 에 같은 검사의 직전 결과를 붙인다.
    readings: 기준 시각 이후 나온 판독의 검사명, 판독 시각, 판독 소견 원문.
    """
    patient = _load_patient(patient_id)
    last_note = max(patient["progress_notes"], key=lambda note: note["written_at"])
    baseline = datetime.fromisoformat(last_note["written_at"])

    labs = sorted(patient["labs"], key=lambda lab: datetime.fromisoformat(lab["measured_at"]))
    abnormal_labs = []
    for index, lab in enumerate(labs):
        if datetime.fromisoformat(lab["measured_at"]) <= baseline:
            continue
        if lab["ref_low"] <= lab["value"] <= lab["ref_high"]:
            continue
        earlier = [prev for prev in labs[:index] if prev["name"] == lab["name"]]
        previous = (
            {"value": earlier[-1]["value"], "measured_at": earlier[-1]["measured_at"]}
            if earlier
            else None
        )
        abnormal_labs.append({**lab, "previous": previous})

    new_readings = [
        {"exam": reading["exam"], "read_at": reading["read_at"], "finding": reading["finding"]}
        for reading in patient["readings"]
        if datetime.fromisoformat(reading["read_at"]) > baseline
    ]

    return {
        "patient_id": patient["patient_id"],
        "baseline_at": last_note["written_at"],
        "labs": abnormal_labs,
        "readings": new_readings,
    }


@tool
def search_kcd_codes(query: str) -> dict | str:
    """소견 검색어 하나로 KCD-9 후보 코드를 검색한다.

    검색 결과 상위 10건을 코드, 한글 명칭, 출처(KCD/{코드})로 돌려주기만 하고 고르지 않는다.
    어떤 코드를 쓸지는 Agent 가 판단한다. 결과가 없으면 '검색 결과 없음'을 돌려준다.
    """
    docs = search(query)
    if not docs:
        return "검색 결과 없음"
    return {
        "query": query,
        "results": [
            {
                "code": doc.metadata["code"],
                "name": doc.metadata["name_ko"],
                "source": f"KCD/{doc.metadata['code']}",
            }
            for doc in docs
        ],
    }
