# 미니 PJT: 병동 경과기록 요약·상병 추천 Agent

## 무엇을 푸나
병동 의사가 환자 ID 와 짧은 메모만 넣으면, 직전 경과기록 이후의 Lab·판독·투약 변화를 모은 SOAP 초안과 KCD-9 상병 후보(기존 상병 + 새 제안 최대 5개)를 만들어, 경과기록 정리 부담과 부상병 누락을 줄인다.

## 활용한 패턴 (Day 1~7)
- Day 1: 구조화 출력 (필수 패턴 1) — 답변을 `SoapDraft` Pydantic 모델(S·O·A·P, 기존 상병, 새 제안)로 정의하고 `llm.with_structured_output` 으로 받는다. Agent 가 도구 호출을 마치면 `SoapFormatter` 미들웨어가 대화 전체를 초안으로 정리한다
- Day 2: RAG 하이브리드 검색 (필수 패턴 3) — `data/kcd9.csv` 전체를 BM25(kiwipiepy 토크나이저)와 벡터 검색(Titan Text Embeddings V2, Chroma)으로 나눠 찾고 `EnsembleRetriever` 로 합쳐 상위 10건을 돌려준다. 상위 결과와 앞 세 자리가 같은 코드(같은 분류)를 함께 돌려주고, 맞는 코드가 없으면 분류명·영문 명칭으로 다시 검색하게 한다(쿼리 확장)
- Day 3: 도구 호출 에이전트 — `create_agent` 로 Agent 를 만들고, Agent 가 도구를 부를지·어떤 검색어로 부를지·어떤 상병을 고를지 판단한다
- Day 4: 도메인 도구 — `get_patient_context`, `get_changes_since_last_note`, `search_kcd_codes` 3개. 도구 안에서는 LLM 을 부르지 않고 조회·비교·검색 결과만 돌려준다
- Day 5: 미들웨어 가드레일 — `InputGuard`(정규식 + LLM 판별, 고정 거절 문구), `PatientScope`(질문에 처음 나온 환자 외 조회 차단), `OutputGuard`(after_model 에서 새 제안 5개 이하·KCD-9 코드 여부를 코드로 검사하고 명칭을 KCD-9 원문으로 맞춤, 기존 상병은 환자 데이터로 맞춤, 메모가 비면 S·신체 진찰을 '(메모에 없음)'으로 고정)
- Day 7: 트레이스 (필수 패턴 11) — `FileTracer` 콜백이 도구 호출과 LLM 호출을 `trace.jsonl` 에 남기고, 같은 기록으로 응답의 `trace` 와 `contexts` 를 채운다. JSONL 에는 호출별 지연 시간(`latency_s`)과 LLM 토큰 수(`usage`)도 남긴다(1·2차 평가 뒤에 추가해 두 라운드 기록에는 없다)
- Day 7: 평가 (필수 패턴 12) — 인-아웃 세트 20건을 LLM-as-Judge 로 판정하고, 코드 일치·제안 개수·KCD-9 여부·데이터에 없는 수치는 코드로 검사한다. RAGAS 4개 지표는 산출과 보고에만 쓴다

## 아키텍처

```
POST /query {question: "환자 P001 / 메모: ..."}
   │
   ▼
InputGuard ──(인젝션·막을 요청)──▶ 고정 거절 문구
   │
PatientScope (처음 나온 환자 ID 만 허용)
   │
   ▼
Agent (create_agent, Bedrock Claude Sonnet 4.6, temperature=0)
   ⇅ get_patient_context          ─ data/patients/{id}.json : 맥락, 직전 경과기록(없으면 입원기록 → 응급실 기록)
   ⇅ get_changes_since_last_note  ─ data/patients/{id}.json : Lab 이상치·추세, 새 판독, 투약 비교
   ⇅ search_kcd_codes (소견마다)   ─ src/retriever.py : BM25 + Chroma → EnsembleRetriever, 상위 10건 + 같은 분류
   │
   ▼
SoapFormatter (with_structured_output → SoapDraft)
   │
OutputGuard (새 제안 ≤ 5개, kcd9.csv 에 있는 코드·명칭만, 기존 상병 = 환자 데이터, 빈 메모면 S·신체 진찰 고정)
   │
   ▼
{answer, contexts, trace}  ◀── FileTracer (trace.jsonl) 기록으로 trace·contexts 생성
```

- 데이터: 더미 환자 14명(`data/patients/P001~P013, P015.json`), 실제 KCD-9 전체(`data/kcd9.csv`). 정답 상병은 `evaluation/test_queries.csv` 에만 있고 Agent 는 읽지 않는다
- answer: S·O·A·P 초안, 기존 상병, 새 제안(최대 5개), 고정 문구 "이 초안은 추천이며, 작성자가 확인하고 고친 뒤 사용합니다."
- contexts doc_id: 환자 데이터는 `{환자ID}/{종류}/{항목}`(예: `P001/lab/Na`), KCD 문서는 `KCD/{코드}`(예: `KCD/E87.1`)

## 실행 방법

1. 설치 (Python 3.14 에서 확인)
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   pip install ragas==0.4.3 --no-deps
   ```
   ragas 가 요구하는 scikit-network 는 Python 3.14 에서 빌드가 실패하고 테스트셋 생성에만 쓰이므로 의존 없이 따로 설치한다.

2. `.env` 작성 (값은 각자 넣는다)
   ```
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   AWS_DEFAULT_REGION=us-east-1
   BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-6
   BEDROCK_EMBEDDING_MODEL_ID=...   # Amazon Titan Text Embeddings V2
   ```

3. KCD-9 벡터 DB 만들기 (처음 한 번. `data/chroma_db/` 는 저장소에 올리지 않는다)
   ```
   python -m src.retriever
   ```

4. 서버 실행과 질의
   ```
   uvicorn src.server:app
   ```
   ```
   curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"question\": \"환자 P001 / 메모: 기침 줄고 가래 감소했다고 함. 청진상 우하엽 수포음 감소.\"}"
   ```
   PowerShell:
   ```
   Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query -ContentType "application/json; charset=utf-8" -Body '{"question": "환자 P001 / 메모: 기침 줄고 가래 감소했다고 함. 청진상 우하엽 수포음 감소."}'
   ```

5. 자체 평가
   ```
   python -m evaluation.run_eval                                       # 20건 실행 + 코드 검사 + LLM-as-Judge
   python -m evaluation.run_eval --ids 1,13                            # 일부 케이스만
   python -m evaluation.run_eval --reuse                               # Agent 를 다시 돌리지 않고 results.json 으로 검사만
   python -m evaluation.run_eval --report evaluation/round2_report.md  # 리포트 작성 + RAGAS 4개 지표
   ```

## RAGAS 평가 결과
2차 평가 기준 (괄호는 1차 값)
- context_recall: 0.748 (0.715)
- context_precision: 0.378 (0.396)
- faithfulness: 0.468 (0.475)
- answer_relevancy: 0.172 (0.169)

- 거절 답변을 뺀 17건으로 계산했다. 정답 문서가 없어 reference 는 expected_traits 를 이어 붙인 글을 쓴다
- context_precision 은 시간 초과로 2차 11/17건, 1차 14/17건만 계산되어 두 라운드를 직접 비교하기 어렵다
- context_recall 이 오른 것은 같은 분류 코드를 함께 돌려주면서 정답 상병 근거가 contexts 에 더 자주 들어갔기 때문으로 본다. 나머지 세 지표가 낮은 이유는 evaluation/round1_report.md·round2_report.md 의 RAGAS 해석에 적었다

## 인-아웃 세트 통과율 (자체 평가)
- 1차: 8 / 20 통과 (T22 에서 정답 상병 2건을 사람 확인 후 고친 지금 정답 기준으로 다시 세면 10 / 20)
- 2차: 16 / 20 통과 (목표 16건 이상 달성)
- 개선폭: +8 건 (지금 정답 기준 +6 건) (주요 개선 사항: KCD 검색에 같은 분류 코드 추가와 분류명·영문 재검색 규칙, O 에 투약 변경 쓰기 규칙, 등록되지 않은 환자 고정 안내 문구)

| 보조 지표 | 1차 | 2차 | 목표 |
|---|---|---|---|
| 상병 재현율(완전 일치) | 22/32 = 68.8% | 28/32 = 87.5% (기준선 50.0%) | 80% 이상 |
| 상병 재현율(앞 세 자리) | 28/32 = 87.5% | 31/32 = 96.9% (기준선 53.1%) | 참고 |
| O 반영률 | 30/34 = 88.2% | 34/34 = 100.0% | 70% 이상 |
| S·신체 진찰 지어냄 | 0건 | 0건 | 0건 |
| 자동 가드레일 위반 | 0건 | 0건 | 0건 |

- 기준선은 기존 상병만으로 나오는 점수다. 모델은 두 라운드 모두 Agent·판정기·RAGAS 공통으로 us.anthropic.claude-sonnet-4-6 이고, 판정 기준(JUDGE_PROMPT)은 바꾸지 않았다
- 상세: [evaluation/round1_report.md](evaluation/round1_report.md), [evaluation/round2_report.md](evaluation/round2_report.md)

## 트라이앤에러 회고
- 시도했지만 실패한 접근 · 왜 실패했는가
  - 메모 증상만으로 KCD 를 검색했다. 저나트륨혈증·흉막삼출처럼 Lab·판독에서만 드러나는 상병이 빠졌고, 검색어와 KCD-9 명칭이 달라 정답 코드가 상위 10건에 아예 없는 경우가 많았다(1차 실패 원인 1, 7건)
  - 투약 변경을 P 에만 썼다. 직전 대비 변화를 O 에 모으는 것이 서비스의 핵심인데 O 반영률이 88.2% 에 그쳤다
  - 1차 평가를 Haiku 4.5 로 먼저 돌렸다. Sonnet 의 일일 토큰 한도 때문이었지만 모델이 다르면 라운드 비교가 안 되어, 한도가 풀린 뒤 Sonnet 4.6 으로 1차를 다시 돌리고 이후 모든 라운드를 같은 모델로 맞췄다
  - RAGAS 를 기본 설정으로 돌리자 context_precision 이 17건 중 3건만 계산되었다. 동시 실행 수를 줄이고 시간 제한을 늘려 빈 값만 다시 계산했지만 6건은 여전히 빠졌다
- 최종 채택한 접근 · 왜 그것으로 갔는가
  - Lab 이상치·새 판독마다 그 소견이 뜻하는 진단명으로 검색하게 하고, 검색 결과에 같은 분류(앞 세 자리) 코드를 함께 돌려주며, 맞는 코드가 없으면 분류명·영문 명칭으로 한 번 더 검색하게 했다. 도구 안에서 LLM 을 쓰지 않는 규칙을 지키면서 Agent 가 고를 후보를 넓히는 방법이라서다
  - 판정 기준은 라운드 사이에 고정하고, 판정기 출력을 코드로 대조해 오류 의심 항목을 따로 표시했다. 판정 기준을 고치면 개선폭을 믿을 수 없기 때문이다. 의심 2건은 사람이 확인해 판정기 오류로 기록했다
  - 정답 상병은 Claude 가 만들었으므로 5건(1·5·7·13·16번)을 사람이 확인해 2건(D69.6 → D69.5, E16.0 → E11.63)을 고쳤고(확인자는 의료진이 아니어서 코드 존재·데이터 근거·명칭 일치만 대조, round2_report.md 참고), 1차 점수를 지금 정답 기준으로도 다시 세어 함께 적었다
- 남은 한계 · 향후 개선 방향
  - 정답 코드가 검색 결과에 있는데 이웃 코드를 고른다(2번 J91 ↔ J90, 6번 K65.8 ↔ K65.0, 15번 J96.11 ↔ J96.01). 급성/만성, 원발/속발처럼 세부 코드를 가르는 기준을 프롬프트나 검색 결과 설명에 더 넣어야 한다
  - 정답으로 이어지는 검색어를 쓰지 않는 실행이 있어(7번 E11.63) 실행마다 결과가 흔들린다
  - RAGAS context_precision·faithfulness·answer_relevancy 가 낮고, 시간 초과로 일부가 계산되지 않는다
  - 더미 환자 14명, 20건으로만 평가했고, 주상병과 부상병을 구분하지 않는다
  - 다음 단계는 SERVICE.md 에 적은 수술명·수술코드 일치 검증이다

## 핵심 코드 위치
- [src/agent.py:118](src/agent.py#L118) — 메인 에이전트 그래프 (`create_agent` + 미들웨어 순서)
  - [src/agent.py:67](src/agent.py#L67) — `SoapDraft` 구조화 출력 모델
  - [src/agent.py:95](src/agent.py#L95) — `SoapFormatter` (with_structured_output 으로 초안 정리)
- [src/tools.py](src/tools.py) — 도메인 도구
  - [src/tools.py:31](src/tools.py#L31) `get_patient_context`, [src/tools.py:71](src/tools.py#L71) `get_changes_since_last_note`, [src/tools.py:149](src/tools.py#L149) `search_kcd_codes`
- [src/retriever.py](src/retriever.py) — RAG 파이프라인
  - [src/retriever.py:117](src/retriever.py#L117) `EnsembleRetriever`, [src/retriever.py:126](src/retriever.py#L126) `search`, [src/retriever.py:142](src/retriever.py#L142) `same_category`
- [src/guards.py](src/guards.py) — 가드레일 미들웨어
  - [src/guards.py:139](src/guards.py#L139) `InputGuard`, [src/guards.py:167](src/guards.py#L167) `PatientScope`, [src/guards.py:210](src/guards.py#L210) `OutputGuard`
- [src/tracer.py:28](src/tracer.py#L28) — `FileTracer` 콜백
- [src/server.py:130](src/server.py#L130) — `POST /query`
- [evaluation/run_eval.py](evaluation/run_eval.py) — 자체 평가 ([evaluation/run_eval.py:65](evaluation/run_eval.py#L65) `JUDGE_PROMPT`, [evaluation/run_eval.py:342](evaluation/run_eval.py#L342) `run_ragas`)
- [evaluation/test_queries.csv](evaluation/test_queries.csv) — 인-아웃 세트 20건 (positive 8 · negative 4 · edge 5 · guardrail 3)
