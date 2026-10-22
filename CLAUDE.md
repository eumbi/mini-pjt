# 프로젝트 규칙

## 기술 스택
- Python, LangChain, LangGraph
- 모델은 Amazon Bedrock (ChatBedrockConverse)
- Agent 생성은 langchain.agents 의 create_agent 를 쓴다
- answer 는 Pydantic 모델로 정의한 구조화 출력으로 만든다 (필수 패턴 1)
- 트레이스는 LangSmith 로 남긴다 (필수 패턴 11)
- 평가는 RAGAS(context_recall, context_precision, faithfulness, answer_relevancy)와 LLM-as-Judge 를 쓴다 (필수 패턴 12)
- 무엇을 왜 만드는지는 SERVICE.md 를 본다

## 폴더 구조 (제출 규약. 바꾸지 않는다)
src/agent.py       메인 에이전트 그래프
src/tools.py       도메인 도구
src/retriever.py   RAG 파이프라인
data/              사용한 문서와 데이터
evaluation/        test_queries.csv 와 평가 리포트
  test_queries.csv   인-아웃 세트 (강의 §4-4 스키마)
  round1_report.md   1차 자체 평가 결과
  round2_report.md   2차 자체 평가 결과 (개선폭 명시)
SERVICE.md         서비스 스펙
README.md          강의 §4-5 템플릿 준수
requirements.txt

이 프로젝트에서 각 파일에 둘 것
- src/tools.py: 도구 3개. 동작 상세는 SERVICE.md 3절을 따른다
  - get_patient_context(patient_id): 환자 맥락 조회. JSON 조회만 하고 LLM 을 쓰지 않는다
  - get_changes_since_last_note(patient_id): 직전 기록 이후 변화 조회. JSON 조회와 비교만 하고 LLM 을 쓰지 않는다
  - recommend_kcd_codes(clinical_summary, existing_codes): 상병 코드 추천. retriever 검색 결과에서 LLM 이 고르고, KCD-9 목록에 있는지 코드로 다시 확인한다
- src/retriever.py: data/kcd9.csv 검색. BM25 와 벡터 검색(Amazon Titan Text Embeddings V2)을 합친 하이브리드 검색이며, 검색어마다 상위 10건을 돌려준다
- data/patients/{patient_id}.json: 더미 환자 데이터 (필드는 SERVICE.md 3절)
- data/kcd9.csv: 실제 KCD-9 전체 (코드, 한글 명칭, 영문 명칭)
- evaluation/: 20건의 test_queries.csv 와 라운드별 평가 리포트

## 주고받는 형식 (제출 규약)
- POST /query 로 받고 question 필드를 읽는다
- 답은 answer, contexts, trace 세 키로 돌려준다
- question 에는 환자 ID 와 작성자 메모를 담는다
- answer 는 S·O·A·P 순서의 초안, '기존 상병'과 '새 제안'(최대 5개) 두 목록, 고정 문구 "이 초안은 추천이며, 작성자가 확인하고 고친 뒤 사용합니다." 순서로 담는다
- contexts 는 [{"doc_id", "text"}] 목록이다. 도구가 가져온 환자 데이터와 KCD-9 검색 결과를 모두 담는다
  - doc_id 형식: 환자 데이터는 {환자ID}/{종류}/{항목} (예: P001/lab/Na), KCD 문서는 KCD/{코드}
- trace 는 [{"step", "input", "output"}] 목록이다. step 에는 도구 이름이나 단계 이름을 호출 순서대로 쓴다

## 도메인 규약
- 경과기록은 SOAP 양식으로 쓴다. 문체는 모든 의사가 이해할 수 있는 표준 문체로 한다
- 직전 경과기록을 입원 경과 요약으로 본다. 입원 첫날에는 입원기록에서, 입원기록도 없으면 응급실 기록에서 출발한다
- 과거 입원·외래 이력은 기저질환 확인에만 쓴다
- 비교 기준 시각은 직전 경과기록의 작성 시각이다. 직전 경과기록이 없으면 입원 시각을 쓴다
  - Lab: 참고치(하한·상한)를 벗어난 결과는 전부 포함하고 직전 결과를 함께 보여준다. 이번 입원 중 같은 검사가 3회 이상이면 시간 순서대로 추세를 보여준다
  - 판독: 기준 시각 이후에 나온 것만 쓴다
  - 투약: 직전 처방과 비교해 신규, 중단, 용량·경로 변경으로 나눈다
- 결과가 없을 때는 지어내지 않고 정해진 문구를 돌려준다: '데이터 없음: 등록되지 않은 환자', '변화 없음', '추천할 새 상병 없음'
- S와 신체 진찰은 메모에 있는 것만 쓰고, 메모에 없으면 '(메모에 없음)'으로 남긴다
- P는 문장으로만 쓴다
- 상병 코드는 KCD-9 기준이다. KCD-9에 없는 코드는 내지 않는다

## 평가 규약
- test_queries.csv 는 7개 열(id, category, input, expected_traits, forbidden, expected_tools, note)을 지킨다. 필드 안의 구분자는 세미콜론이다
  - category 는 positive / negative / edge / guardrail 중 하나다. 지금 구성은 8 / 4 / 5 / 3 이다
  - input 이 그대로 /query 의 question 이 된다
  - 정답 상병은 expected_traits 에 "상병 목록에 {코드} 포함" 형식으로 적는다. 상병 재현율은 이 형식을 읽어 계산한다
- 케이스 통과: expected_traits 를 모두 만족하고 forbidden 이 하나도 나오지 않으면 통과다. 목표 통과율은 80%(20건 중 16건) 이상이다
  - LLM-as-Judge 로 판정하고, 코드 일치와 제안 개수처럼 기계로 볼 수 있는 항목은 코드로 검사한다
  - 결과는 evaluation/round1_report.md, round2_report.md 에 라운드별로 남기고, RAGAS 4개 지표도 함께 적는다
- 상병 재현율은 인-아웃 세트의 정답 상병 중 "기존 상병 + 새 제안 5개"에 든 비율로 계산한다. 목표는 80% 이상이다
  - 정답 상병은 요청 시점까지의 데이터로 근거를 댈 수 있는 것만 넣는다. 수술 관련 코드는 넣지 않는다
  - KCD 코드가 완전히 같을 때만 맞은 것으로 센다. 상위 코드 일치는 인정하지 않는다
  - 분모는 인-아웃 세트의 정답 코드 전체다. 환자별 비율의 평균이 아니다
  - 기존 상병만으로 나오는 점수(기준선)를 리포트에 함께 적는다
- 더미 데이터는 정답 상병을 먼저 정하고 만든다. Lab, 판독, 처방에는 진단명을 직접 쓰지 않고 소견만 담는다
  - expected_traits 에 적은 수치와 변화가 그대로 나오도록 만든다
  - 응급실 기록에서 출발하는 케이스(P003)에는 입원기록을 넣지 않는다
- 정답 상병은 evaluation/test_queries.csv 에만 둔다. Agent 의 도구와 검색은 evaluation/ 을 읽지 않는다
- 정답 중 5건은 사람이 직접 확인한다
- O 반영률: 더미 데이터에 일부러 넣은 Lab 변화, 새 판독, 투약 변경 중 O에 반영된 비율. LLM 판정기로 재고, 목표는 70% 이상이다
- S·신체 진찰 지어냄: LLM 판정기가 메모와 대조한다. 목표는 0건이다. 메모가 비었거나 신체 진찰 내용이 없는 사례를 인-아웃 세트에 6건 넣는다
- 자동 가드레일 검사: 새 제안 5개 이하, KCD-9에 있는 코드인지, 데이터에 없는 Lab·판독·처방 수치를 썼는지 코드로 검사한다. 목표는 위반 0건이다

## 코드 규칙
- 파일 하나에 한 가지 역할만 둔다
- 함수와 도구에는 한국어 docstring 을 쓴다
- 비밀 값은 .env 에서 읽고 코드에 적지 않는다

## 하지 말 것
- 요청하지 않은 파일을 새로 만들지 않는다
- 기존 파일을 통째로 다시 쓰지 않는다. 바뀐 부분만 고친다