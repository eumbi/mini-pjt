# TASKS

- 한 번에 작업 하나만 한다 (30분~1시간). 끝나면 체크하고 커밋한다
- 무엇을 만드는지는 SERVICE.md, 어떻게 만드는지는 CLAUDE.md 를 따른다
- "확인"을 통과해야 끝난 것으로 본다

## M1. 최소 경로 — 1번 케이스(P001)가 /query 에서 답변까지 한 번 돈다
- [x] T1. 실행 환경: requirements.txt 작성, .env 에 Bedrock 자격 증명과 BEDROCK_MODEL_ID
  - 확인: Bedrock 모델을 한 번 호출해 응답 문자열이 출력된다
- [x] T2. P001 더미 데이터 1명: data/patients/P001.json
  - 확인: 1번 케이스 expected_traits 의 수치(Na 133→130→128, CRP 12.4→6.1, X선 판독)가 파일에 그대로 있다
- [x] T3. get_patient_context 최소판: 직전 경과기록 경로만
  - 확인: P001 로 부르면 기저질환·알레르기·입력 상병(J18.9)·직전 경과기록 본문이 나온다
- [x] T4. get_changes_since_last_note 최소판: Lab 이상치+직전값, 신규 판독만
  - 확인: P001 로 부르면 Na 128(직전 130)과 신규 흉부 X선 판독이 나온다
- [x] T5. 임시 KCD 검색: data/kcd9.csv 샘플 몇 줄 + BM25 검색 + search_kcd_codes
  - 확인: "저나트륨혈증"을 검색하면 E87.1 이 코드·명칭·출처(KCD/E87.1)로 나온다
- [x] T6. Agent 와 구조화 출력: src/agent.py (create_agent + 도구 3개 + 짧은 시스템 프롬프트 + SOAP Pydantic 모델)
  - 확인: P001 질문으로 실행하면 S·O·A·P, 기존 상병, 새 제안(E87.1 포함), 고정 문구가 나온다
- [x] T7. API 서버: src/server.py 의 POST /query (contexts 는 도구 결과, trace 는 도구 호출 순서)
  - 확인: curl 로 1번 케이스 input 을 보내면 answer·contexts·trace 세 키가 돌아오고 trace 에 도구 3개가 있다

## M2. 데이터와 도구 완성
- [x] T8. 더미 환자 P002~P008 (7명)
  - 확인: 각 케이스 expected_traits 의 수치와 변화가 해당 JSON 에 그대로 있다
- [x] T9. 더미 환자 P009~P013, P015 (6명). P003 은 입원기록 없이 응급실 기록만
  - 확인: 같은 대조가 통과하고 P003.json 에 입원기록이 없다
- [x] T10. get_patient_context 분기: 입원기록 → 응급실 기록 → 없는 환자
  - 확인: P010 은 입원기록, P003 은 응급실 기록에서 출발하고, P999 는 '데이터 없음: 등록되지 않은 환자'다
- [x] T11. get_changes_since_last_note 완성: Lab 추세(3회 이상), 투약 비교, '변화 없음'
  - 확인: P006 은 Hb 추세 3개 값, P002 는 ceftriaxone→ciprofloxacin 변경이 나오고, P012 는 세 항목 모두 '변화 없음'이다
- [x] T12. KCD-9 전체 교체: 원본(사람이 준비)을 data/kcd9.csv 형식으로 변환
  - 확인: test_queries.csv 의 정답 상병 코드가 모두 kcd9.csv 에 있다 (없으면 CSV 를 고친다)
- [x] T13. 하이브리드 검색: src/retriever.py 에 Chroma DB 미리 만들기 + BM25(kiwipiepy) + EnsembleRetriever
  - 확인: "저나트륨혈증" 상위 10건에 E87.1 이 있고, 두 번째 실행은 임베딩 없이 DB 를 불러오기만 한다

## M3. 안전과 관측
- [x] T14. 출력 가드레일: src/guards.py 의 after_model 미들웨어 (새 제안 5개 이하, KCD-9 코드 여부)
  - 확인: 새 제안이 6개거나 kcd9.csv 에 없는 코드가 섞인 답변을 넣으면 걸러진다
- [x] T15. 입력 가드레일과 고정 거절 문구: 정규식 + LLM 판별
  - 확인: 19번(프롬프트 인젝션)은 고정 문구로 거절되고, 1번 메모는 막히지 않는다
- [x] T16. 트레이스: src/tracer.py 의 FileTracer 로 JSONL 기록, 같은 기록으로 trace·contexts 채우기
  - 확인: 1번 실행 후 JSONL 에 도구 3개와 LLM 호출이 남고 응답 trace 와 순서가 같다
- [ ] T17. 예외 케이스 동작 다듬기 (환자 ID 없음, 메모 빈칸, 다른 환자 요구, 진찰 지어내기 유도)
  - 확인: 11·13·18·20번을 /query 로 보내면 각 note 에 적은 동작대로 나온다

## M4. 평가
- [ ] T18. run_eval.py ① 코드 검사: 20건 실행, 도구 호출, 상병 재현율(완전·앞 세 자리·기준선), 자동 가드레일 위반
  - 확인: 한 번 실행으로 재현율 두 점수와 기준선, 위반 건수가 출력된다
- [ ] T19. run_eval.py ② LLM-as-Judge: 케이스 pass/fail, O 반영률, S·신체 진찰 지어냄
  - 확인: 케이스별 PASS/FAIL 과 "XX/20 통과"가 출력된다
- [ ] T20. run_eval.py ③ RAGAS 4개 지표
  - 확인: context_recall·context_precision·faithfulness·answer_relevancy 값이 출력된다
- [ ] T21. 1차 평가: evaluation/round1_report.md
  - 확인: 통과 건수, 실패 케이스와 원인, 모든 지표가 리포트에 있다
- [ ] T22. 개선 1회: 1차 실패 원인 중 가장 많은 것 하나를 고친다 (필요하면 반복)
  - 확인: 고친 실패 케이스를 다시 돌려 PASS 로 바뀐다
- [ ] T23. 2차 평가: evaluation/round2_report.md
  - 확인: 1차 대비 개선폭(통과 건수와 지표)이 적혀 있다. 목표 16/20 이상

## M5. 제출
- [ ] T24. README.md (강의 §4-5 템플릿)
  - 확인: 활용 패턴·아키텍처·실행 방법·RAGAS·1·2차 통과율·회고·핵심 코드 위치가 모두 채워져 있다
- [ ] T25. 클린 환경 재현과 푸시
  - 확인: 새 가상환경에서 requirements.txt 설치 후 1번 케이스가 /query 로 동작하고 원격 main 에 올라가 있다
