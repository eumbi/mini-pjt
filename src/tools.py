"""도메인 도구. data/patients/ 의 JSON 더미 데이터를 조회만 하고 LLM 을 부르지 않는다."""

import json
from pathlib import Path

from langchain_core.tools import tool

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
