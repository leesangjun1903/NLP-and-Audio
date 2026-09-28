# HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?

> **⚠️ 중요 고지**: 본 논문은 arXiv:2609.01437v1 (2026년 9월 1일자)로, 제 학습 데이터 컷오프 이후의 논문입니다. 모든 분석은 **제공된 PDF 원문에만 근거**하며, 외부 정보를 추가하지 않았습니다. 확신이 없는 부분은 명시적으로 표시합니다.

---

## 1. Executive Summary (10문장 이내)

HarnessDev는 LLM이 작업을 수행하는 데 그치지 않고, 자신의 **실행 인프라(agent harness)** 자체를 생성·개선할 수 있는지를 측정하는 최초의 벤치마크이다.  
평가 단위를 "작업 출력"에서 "실행 가능한 인프라"로 전환한다는 점에서 기존 벤치마크와 근본적으로 차별화된다.  
벤치마크는 **Creation**(약한 씨앗 코드에서 완전한 harness 구축)과 **Evolution**(다운스트림 피드백 기반 반복 개선)의 두 단계로 구성된다.  
6개 창작 LLM, 4개 도메인, 5개 다운스트림 벤치마크(총 2,207개 인스턴스)를 대상으로 실험한다.  
Creation 결과, 모델이 생성한 harness는 글쓰기 및 ML 실험 도메인에서는 인간 참조 수준에 근접하거나 초과하지만, 코드 및 검색·연구 도메인에서는 여전히 큰 격차가 존재한다.  
Evolution은 피드백 집합 내에서 일부 성능 향상을 보이나, 비가시 태스크로의 전이는 불안정하다.  
실행 비용(토큰 수)은 성능과 상관관계가 없으며 최대 19배까지 차이가 난다.  
Harness를 실행하는 모델(executor)이 바뀌면 성능이 크게 달라지는 **executor 의존성**이 확인되었다.  
이는 harness가 모델 가중치와 독립적으로 축적 가능한 "외재화된 지능"임을 시사한다.  
본 연구는 harness 공학을 자동화하려는 향후 연구에 핵심 기준점을 제공한다.

> 📌 **용어 설명**  
> - **Agent Harness**: 모델 외부에서 실행 루프, 도구 사용, 컨텍스트 관리, 오류 복구, 결과 검증을 담당하는 소프트웨어 실행 인프라. 모델 가중치와 독립적으로 존재하며 교체 가능.  
> - **Creator LLM ($L_C$)**: harness를 설계·구축하는 모델  
> - **Executor LLM ($L_E$)**: 완성된 harness 내에서 실제 다운스트림 태스크를 수행하는 모델

### 1-1. 연구의 목적과 필요성

**배경**: 동일한 모델 가중치(GPT-5)라도 harness에 따라 Terminal-Bench 2.1 성능이 35.2%(Terminus 2)에서 49.6%(Codex CLI)로 달라진다 (p.1). 즉, **harness는 모델 능력의 일부를 결정**한다.

**문제**: 기존 벤치마크(SWE-bench, GAIA, WebArena 등)는 harness를 고정 설정값으로 취급하며, 모델이 harness를 스스로 구축·개선할 수 있는지는 평가하지 않는다 (p.2).

**현실적 필요성**: AI 시스템이 실제 배포 환경으로 이동하면서 "forward-deployed engineer(FDE)"—특정 고객 환경에 맞춰 시스템을 지속적으로 개선하는 역할—의 수요가 급증하고 있다. 이 역할을 LLM이 부분적으로 대체할 수 있는지가 실용적 질문이다 (p.3).

**연구 목적**: HarnessDev는 이 평가 공백을 채우기 위해, 평가 단위를 태스크 출력에서 실행 가능한 인프라로 전환하고, Creation과 Evolution의 두 단계에서 모델의 시스템 구축·유지 능력을 측정한다.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|---|---|---|
| LLM은 약한 씨앗에서 실행 가능한 harness를 구축할 수 있다 | 6개 모델 모두 0점인 씨앗 대비 비제로 점수 달성 | Table 3, p.8 |
| Creation 품질은 도메인별로 크게 다르다 | 글쓰기/MLE는 참조 수준 도달, 코드/검색은 큰 격차 | Table 3, Figure 4, p.8 |
| Executor 모델이 바뀌면 성능 순위가 역전된다 | Opus SWE-Pro: Self 69.3 → Gemini 33.0; Qwen BrowseComp: Self→Gemini +17.6pp | Figure 6, p.10 |
| 실행 비용과 성능은 상관없다 | MLE-bench: GPT-5.5 19.1점/29.3M 토큰 vs DeepSeek 19.6점/208.4M 토큰 | Figure 9, p.23 |
| Evolution은 피드백 집합 내에서 향상을 보이나 전이가 불안정하다 | 피드백 쌍 향상 vs held-out 향상 비교; 방향 일치율 53.1% | Table 6, p.12 |
| Executor 변경은 Evolution 결과에도 강하게 영향을 미친다 | Fixed-Gemini 조건에서 4개 중 3개 계통이 held-out 회귀 | Table 6, p.12 |
| 코드 크기는 성능을 예측하지 못한다 | Gemini 1,006 LOC 추가로 Terminal-Bench 최고점(68.8) 달성 | Table 5, p.9 |
| State/Memory 구현이 가장 취약한 구성 요소이다 | 18개 중 18개 미관찰 인스턴스가 모두 state/memory 관련 | Figure 5, p.10 |
| 자가 테스트 횟수는 성능과 약한 상관만 보인다 | Spearman r = 0.13~0.26 (비유의), 수정 호출은 r = 0.57 (p≤.0005) | p.9 |
| 가시 피드백으로 선택한 최종 버전이 held-out 최적과 일치하지 않는다 | 9개 선언 버전 중 2개만이 held-out 최적 | p.13 |

---

## 2-1. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### ① 해결하고자 하는 문제

기존 에이전트 벤치마크는 harness를 실험 설정의 일부로 고정하고 태스크 완료 성능만 측정한다. 이는 다음 질문을 평가하지 못한다:

- **RQ1 (Creation)**: 모델이 약한 씨앗에서 새로운 태스크 패밀리를 위한 완전한 실행 시스템을 구축할 수 있는가?
- **RQ2 (Evolution)**: 모델이 다운스트림 실행 피드백을 사용하여 자신의 harness를 지속적으로 개선하면서 기존 동작을 보존할 수 있는가?

### ② 제안하는 방법 (핵심 수식 포함)

**핵심 프레임워크 수식** (Eq. 1, p.4):

$$
(L_C,\, D) \rightarrow H, \qquad (H,\, L_E,\, x) \rightarrow y \xrightarrow{J} \text{score}
$$

| 기호 | 의미 |
|---|---|
| $L_C$ | Creator LLM: harness를 설계·구축하는 모델 |
| $D$ | 개발 환경 (Claude Code, Codex 등) |
| $H$ | 생성된 harness (동결 후 재사용) |
| $L_E$ | Executor LLM: 동결된 harness 내에서 태스크를 수행하는 모델 |
| $x$ | 다운스트림 태스크 인스턴스 |
| $y$ | 태스크 출력 |
| $J$ | 평가자 (동결, 비교 내 고정) |

> 📌 **용어 설명**  
> - **동결(frozen)**: harness 코드가 개발 완료 후 변경 불가 상태로 고정되어, 이후 모든 평가에 동일 버전이 사용됨

**Harness 추상화** (시스템 프롬프트, Appendix E, p.26):

$$
H = \langle E,\, T,\, C,\, S,\, L,\, V \rangle
$$

| 구성요소 | 의미 |
|---|---|
| $E$ (Execution) | 실행 루프, 계획, 정지 조건, 스케줄링 |
| $T$ (Tools) | 도구 인터페이스, 선택, 입출력 제약, 오류 처리 |
| $C$ (Context) | 태스크·코드·로그·히스토리가 컨텍스트로 진입하는 방식 |
| $S$ (State) | 현재 목표, 가설, 진행 상황, 시도, 실패, 아티팩트 상태 |
| $L$ (Lifecycle) | 도구 호출 전후 훅, 실패·타임아웃 처리, 복구, 종료 |
| $V$ (Verification) | 테스트, 검증, 아티팩트 유효성 검사, 궤적 기록 |

**Evolution 평가 수식** (Eq. 2, p.11):

$$
\bar{P}_t = \frac{1}{2}\left(P_t^{\text{SWE100}} + P_t^{\text{Term89}}\right)
$$

| 기호 | 의미 |
|---|---|
| $\bar{P}_t$ | 시점 $t$에서의 동결 버전 쌍 점수 |
| $P_t^{\text{SWE100}}$ | 해당 버전의 SWE-Pro-100 태스크 정답률(%) |
| $P_t^{\text{Term89}}$ | 해당 버전의 Terminal-Bench-89 태스크 정답률(%) |
| $t$ | 공식 평가를 완료한 동결 버전 인덱스 |

### ③ 모델 구조 및 실험 설정

**평가 조건 두 가지**:

| 조건 | 설명 |
|---|---|
| **Self-Eval** | $L_E = L_C$: 창작자 모델이 자신의 harness에서 실행 |
| **Unified-Eval** | $L_E = \text{Gemini 3.1 Pro}$ (고정): 모든 harness에 동일 executor 적용 |

**평가된 6개 Creator LLM**:

| 모델 | 개발 환경 |
|---|---|
| Opus 4.8 | Claude Code 2.1.177 |
| GPT-5.5 | Codex 0.144.3 |
| Gemini 3.1 Pro | Claude Code 2.1.177 |
| DeepSeek V4 Pro | Claude Code 2.1.177 |
| Qwen 3.7 Max | Claude Code 2.1.177 |
| Seed 2.0 Pro | Claude Code 2.1.177 |

**5개 다운스트림 벤치마크** (Table 2, p.6):

| 도메인 | 벤치마크 | 태스크 수 | 지표 |
|---|---|---|---|
| Code | SWE-bench Pro (공개 분할) | 731 | 태스크 성공률 |
| Code | Terminal-Bench 2.1 | 89 | 태스크 성공률 |
| Data Analysis | MLE-bench | 75 | 메달 점수 |
| Writing | EQ-Bench3 | 46 | 루브릭 점수 |
| Research | BrowseComp | 1,266 | 정확도 |

> 📌 **용어 설명**  
> - **avg@3**: 동일 모델로 독립적으로 3회 harness를 생성·평가한 결과의 평균. 단일 harness의 비대표성을 보완하기 위해 사용.

### ④ 성능 향상

**Creation (Self-Eval, Table 3, p.8)**:

| Creator | SWE-Pro | Terminal-2.1 | MLE-bench | EQ-Bench3 | BrowseComp | 평균 |
|---|---|---|---|---|---|---|
| 씨앗(Hseed) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Opus 4.8 | 69.3 | 64.8 | 32.9 | 84.6 | 52.4 | **67.8** |
| GPT-5.5 | 32.8 | 52.1 | 19.1 | 83.0 | 52.6 | 55.1 |
| Gemini 3.1 Pro | 43.6 | 68.8 | 32.4 | 74.8 | 35.2 | 55.6 |
| DeepSeek V4 Pro | 28.9 | 35.6 | 19.6 | 75.4 | 40.9 | 45.2 |
| Qwen 3.7 Max | 33.5 | 41.3 | 3.1 | 68.7 | 32.3 | 44.0 |
| Seed 2.0 Pro | 10.8 | 6.0 | 5.3 | 71.1 | 3.2 | 22.8 |
| **인간 참조** | **80.0*** | **88.8*** | **24.0** | **83.7** | **92.2*** | **86.2** |

**Evolution (Table 6, p.12)**:

| 설정 | Creator | 피드백 쌍 향상 | Held-out-630 향상 |
|---|---|---|---|
| Self | Opus 4.8 | +3.0 | **+4.44** |
| Self | GPT-5.5 | +5.9 | +3.81 |
| Self | Gemini 3.1 Pro | +8.8 | +2.70 |
| Self | DeepSeek V4 Pro | +13.4 | +3.17 |
| Self | Qwen 3.7 Max | +13.9 | +1.43 |
| Fixed Gemini | GPT-5.5 | +2.4 | **−10.32** |
| Fixed Gemini | DeepSeek V4 Pro | +6.5 | −2.38 |
| Fixed Gemini | Qwen 3.7 Max | +1.1 | −1.11 |

### ⑤ 한계

| 한계 | 내용 |
|---|---|
| 도메인 커버리지 | 4개 카테고리가 모든 실제 배포를 대표하지 않음 |
| 인간 기준선 | 불균등하며 최적이 아닐 수 있음 |
| Unified-Eval | 실행자 차이를 완전히 제거하지 못함 |
| Evolution 통계 | creator-runtime 셀당 단일 궤적, 불확실성 추정 불가 |
| Held-out 범위 | Evolution의 비가시 평가가 SWE-Pro에만 국한 |
| D 고정 | 개발 환경 자체의 영향 미분리 |
| 훈련 대체 불가 | 모델-외부 학습이 파라미터 훈련을 대체할 수 없음 |

---

## 3. 각 주장의 근거 위치

| 주장 | 위치 |
|---|---|
| harness 변경만으로 GPT-5 성능 35.2% → 49.6% 변화 | p.1, Section 1 |
| Creation 프레임워크 수식 $(L_C, D) \rightarrow H$ | p.4, Eq. (1) |
| Harness 추상화 $H = \langle E,T,C,S,L,V \rangle$ | p.5, Figure 3; p.26, Appendix E |
| 씨앗 harness 미수정 시 모든 벤치마크 0점 | p.4, Section 3.2; Table 3 |
| Opus 4.8 Self-Eval 최고 평균 (67.8) | Table 3, p.8 |
| Gemini 최소 코드(1,006 LOC)로 Terminal 최고점 | Table 5, p.9 |
| State/Memory 가장 취약 | Figure 5, p.10 |
| Executor 변경 시 Opus SWE-Pro 69.3→33.0 | Figure 6, p.11 |
| Evolution 쌍 점수 수식 | p.11, Eq. (2) |
| 피드백·held-out 방향 일치율 53.1% | p.13 |
| 선언 버전 중 held-out 최적은 2/9 | p.13 |
| 비용-성능 비상관 (최대 19배 차이) | Figure 9, p.23; p.8 |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

- Opus 4.8 Self-Eval 평균 67.8점, 인간 참조 86.2점 (Table 3)
- Gemini 1,006 LOC로 Terminal-Bench 68.8점 달성 (Table 5)
- Executor가 Gemini로 고정되면 Opus SWE-Pro 69.3 → 33.0점 (Table 4, Figure 6)
- Self-runtime 5개 계통 모두 held-out 향상 (+1.43~+4.44, 평균 +3.11) (Table 6)
- Fixed-Gemini 조건에서 4개 중 3개 held-out 회귀 (Table 6)
- 피드백·held-out 점수 방향 일치율 53.1% (64쌍 중 34회) (p.13)
- 77.8%의 Data 태스크 실패가 harness 결함에 기인 (p.7)
- 자가 테스트 Spearman r = 0.13~0.26 (비유의); 수정 호출 r = 0.57 (p≤.0005) (p.9)

### 필자의 해석

- **"Evolution은 피드백 과적합"의 신호**: 피드백·held-out 방향 일치율 53.1%는 무작위(50%)에 가깝다. 이는 Evolution이 통계적 의미에서 유의한 신호가 아닌 **노이즈에 가까운 로컬 서치**일 수 있음을 시사한다. ⚠️ (단, 저자들도 이를 "local program search"로 표현)
- **Creator-Executor 공진화 문제**: Opus harness의 Gemini 전이 실패(중복 쿼리율 10.1%→88.2%)는 단순한 이식성 문제가 아닌, harness가 특정 모델의 행동 패턴에 암묵적으로 맞춤화되는 **암묵적 공진화** 현상으로 해석 가능
- **코드 크기와 성능의 무상관**은 단순히 "집중된 변경이 중요하다"는 것 외에도, **LLM의 코드 생성이 실행 효과성보다 구조적 완성도를 우선시**하는 경향을 반영할 수 있음
- **MLE-bench에서 인간 참조 초과(32.9 vs 24.0)**는 ML 실험이 상대적으로 구조화된 피드백 루프를 가져 자동화에 유리하다는 점을 시사하며, 이는 **도메인의 형식화 수준**이 harness 구축 난이도를 결정함을 의미

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목 | 문제점 | 위치 |
|---|---|---|
| ⚠️ Evolution 단일 궤적 | creator-runtime 셀당 단일 궤적으로 불확실성 추정 불가 | p.15, 6.1절 |
| ⚠️ Held-out 평가 범위 제한 | Evolution의 held-out이 SWE-Pro에만 국한, Terminal/BrowseComp 등 미포함 | p.15, 6.1절 |
| ⚠️ 인간 참조가 다른 executor와 쌍을 이룸 | 인간 참조는 별도 harness-executor 쌍이며 공통 executor 기반 통제 비교가 아님 | Table 3 각주, p.8 |
| ⚠️ 3개 별표(*) 수치가 외부 보고 수치 | SWE-Pro 80.0, Terminal-Bench 88.8, BrowseComp 92.2는 재실행 없이 OpenAI GPT-5.6 공식 발표 인용 | Table 3 각주, Appendix B.2 |
| ⚠️ avg@3 중 붕괴된 복제본 | Opus(SWE-Pro), DeepSeek(SWE-Pro/Terminal) Unified-Eval 셀에 붕괴된 R3 복제본 포함 | Table 4 각주, p.8 |
| ⚠️ ±4.75 노이즈 밴드 | 동일 커밋 반복 실행 시 쌍 점수 변동이 ±4.75점으로, 소규모 Evolution 향상이 코드 변경에 귀인 불가 | p.13 |
| ⚠️ 피드백·held-out 방향 일치율 53.1% | 무작위 기준(50%)과 거의 동일하여 통계적 유의성 불명확 | p.13 |
| ⚠️ Spearman 상관의 표본 크기 | 18개 Code harness 기반 상관계수이며 n이 소규모 | p.9 |
| ⚠️ 비용-성능 비교 불가 | Self-Eval과 Unified-Eval의 토큰 비용이 별도 집계되어 직접 비교 어려움 | Figure 9 |
| ⚠️ 도메인 간 지표 이질성 | SWE-Pro(성공률), MLE-bench(메달 점수), EQ-Bench3(루브릭) 등이 다른 단위여서 평균 집계 시 가중 방식 미세화 필요 | Table 3 |

---

## 6. 논문이 답하지 않는 질문

| 미해결 질문 | 관련 위치 |
|---|---|
| ❓ 진화된 harness가 추가 진화의 개발 환경으로 사용될 수 있는가? (재귀적 자기 개선) | p.16, 6.1절 |
| ❓ Evolution의 held-out 향상이 통계적으로 유의한가? (표준 편차, 신뢰 구간 미제공) | p.12, Table 6 |
| ❓ harness 구조적 특성(체크포인팅, 메모리)이 성능에 인과적으로 기여하는가? | p.8, 9 |
| ❓ 더 많은 피드백 태스크는 Evolution을 개선하는가? | p.11, 3.2절 |
| ❓ Writing/Search/MLE 도메인에서도 Evolution이 효과적인가? | p.11 (Code에만 집중) |
| ❓ harness 자체를 훈련 데이터로 활용하면 모델 성능을 향상시킬 수 있는가? | p.16 |
| ❓ 최적 개발 예산(Creation 단계)은 얼마인가? | 미언급 |
| ❓ 생성된 harness의 보안 취약점(도구 오용, 권한 상승)은 어떻게 통제되는가? | Ethics statement에서 제한적 언급 |
| ❓ 창작자가 보지 못한 도메인에서 harness가 일반화될 수 있는가? | 미언급 (도메인 내 held-out만 평가) |
| ❓ matched-budget 기준으로 Evolution이 단순 랜덤 서치를 능가하는가? | p.15 (미래 연구로 남김) |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2): HarnessDev 개요

**해석**: 좌측은 기존 벤치마크가 harness를 고정하는 방식을, 우측은 HarnessDev의 두 단계를 보여준다. Creation에서 씨앗($H_{seed}$)이 완성된 harness $H_0$로 발전하고, Evolution에서 $H_0$가 REQ 1, ..., REQ N을 통해 $H_1$, ..., $H_n$으로 반복 개선된다. "Not only task solvers, but also system builders"라는 슬로건이 핵심 기여를 압축한다. 각 버전이 "동결, 실행 가능, 테스트됨" 상태로 유지된다는 점이 평가 신뢰성의 기반이다.

---

### Figure 4 (p.9): 인간 참조 대비 Creation 거리

**해석**: 각 창작 모델의 Self-Eval 점수를 인간 참조 점수로 정규화한 막대그래프. **MLE-bench에서 Opus(32.9)와 Gemini(32.4)가 인간 참조(24.0)를 초과**(100% 이상)하여 구조화된 ML 실험 도메인에서 LLM harness가 특히 효과적임을 보여준다. 반면 BrowseComp(최대 52.4 vs 92.2*)에서 가장 큰 격차가 나타나, 장기 정보 탐색이 요구되는 연구 도메인이 가장 어려운 도전 과제임을 시각화한다.

> ⚠️ 주의: 100% 초과는 해당 외부 참조를 초과하는 것이지, 인간 능력 자체를 초과하는 것이 아님 (Figure 4 캡션 명시)

---

### Figure 5 (p.10): Harness 구조 증거 히트맵

**해석**: 6개 모델 × 6개 구성요소(E/T/C/S/L/V)의 증거 밀도를 색상으로 표현. **State & Memory($S$) 열이 전 모델에서 가장 연한 색**으로 나타나, 이 구성요소가 체계적으로 취약함을 보여준다. 18개 Code harness 중 State 클래스를 정의한 것이 11개이지만, 체크포인팅을 구현한 것은 1개에 불과하다. 이는 LLM이 **상태 영속성보다 실행 루프와 도구 정책을 먼저 구현하는 경향**이 있음을 시사한다.

> 📌 **용어 설명**  
> - **증거 밀도(Evidence density)**: 해당 메커니즘이 실제 실행 경로에 진입하거나 공식 실행 중 트리거된 비율. 단순히 코드가 존재하는 것은 부분 가중치만 부여됨.

---

### Figure 7 (p.13): Evolution 피드백 집합 궤적

**해석**: 4개 패널(Self/Fixed-Gemini × SWE-Pro/Terminal-Bench)에서 각 모델의 harness 버전별 점수 변화. 핵심 관찰:
1. **비단조성**: 대부분의 계통에서 점수가 오르락내리락하며, 특히 Fixed-Gemini Qwen이 H1에서 급락 후 회복
2. **Stars(선언 최종 버전)**가 최고점이 아닌 경우 다수 존재
3. **Gemini(통제)의 안정성**이 다른 모델보다 상대적으로 높음

이는 Evolution이 수렴하는 최적화가 아닌 **노이즈가 많은 로컬 탐색**임을 시각적으로 확인시켜 준다.

---

### Figure 8 (p.14): SWE-Pro 피드백 vs. Held-out 궤적 오버레이

**해석**: 각 모델에 대해 가시적 100-태스크 피드백 궤적(실선)과 동결 후 630-태스크 held-out 궤적(점선)을 오버레이. **Self-runtime(상단)**: 피드백과 held-out이 대체로 같은 방향으로 움직이나 held-out 향상 폭이 더 작음. **Fixed-Gemini(하단)**: GPT-5.5가 피드백에서 향상되었음에도 held-out에서 급격히 하락(-10.32)하는 극단적 과적합 사례가 명확히 드러남. Stars(선언 버전)가 held-out 최적과 일치하지 않는 경우가 시각적으로 확인됨.

---

## 8. 결론: 시사점, 후속 연구, 추가 제안

### 8-1. 모델의 일반화 성능 향상 가능성

**저자들이 제시한 관련 시사점**:
- "robust evolution across unseen tasks and runtime models remains an open challenge" (p.3)
- Self-runtime 5개 계통의 held-out 평균 향상 +3.11pp는 제한적이나 긍정적 신호
- Executor 전이는 harness의 프롬프트, 도구 프로토콜, 예산, 정지 규칙이 호환될 때만 성공 (p.10)

**일반화 성능 향상의 핵심 병목**:

| 병목 요소 | 관찰된 현상 | 시사하는 해결 방향 |
|---|---|---|
| **Executor 공진화** | Opus harness의 중복 쿼리율 Gemini 전환 후 10.1%→88.2% | 모델-중립적 harness 설계 원칙 수립 |
| **State/Memory 부재** | 26,679개 궤적에서 체크포인트 이벤트 0건 | 영속적 상태 관리가 장기 일반화의 전제 조건 |
| **피드백 과적합** | 피드백·held-out 방향 일치율 53.1% (무작위 수준) | 정규화된 Evolution 프로토콜 필요 |
| **Dead code 문제** | 169개 신규 함수 중 25개가 호출자 없음 | 실행 경로 검증이 테스트 카운트보다 중요 |

**필자의 추가 분석**: 일반화 성능 향상을 위해 가장 유망한 방향은 **"모델-중립적 harness 계층 분리"**다. 현재 harness가 특정 모델의 출력 패턴(120-step 한계, 중복 쿼리 감지 로직 등)에 암묵적으로 의존하는 문제를 해결하려면, harness를 (1) 도메인 로직, (2) 모델-특화 어댑터, (3) 공통 인프라로 명시적으로 분리하는 아키텍처가 필요하다.

---

### 8-2. 2020년 이후 최신 연구 비교 분석

> ⚠️ 아래 비교는 **논문 내 인용된 참고문헌에만 근거**합니다. 외부 지식 보완을 하지 않았습니다.

**관련 최신 연구 위치 지도**:

| 연구 | 연도 | 관계 | HarnessDev 대비 차이점 |
|---|---|---|---|
| SWE-bench [18] | 2023 | 하류 벤치마크 | harness 고정, 태스크 출력 평가 |
| ADAS [15] | 2024 | 에이전트 자동 설계 | 프롬프트/워크플로우 수준, 실행 harness 코드 미평가 |
| AFlow [57] | 2024 | 에이전트 워크플로우 자동화 | 워크플로우 생성에 집중, 지속적 Evolution 미포함 |
| Meta-Agent Challenge [27] | 2026 | Creation 유사 | Creation과 유사하나 연속 발전·비용·executor 분리 미포함 |
| HarnessOpt-Bench [47] | 2026 | Evolution 유사 | 제공된 harness 최적화에 집중, Creation→Evolution 연결 없음 |
| Evo-Bench [16] | 2026 | Evolution 유사 | 최종 수정 품질 강조; 버전별 held-out 궤적 미평가 |
| Self-Harness [56] | 2026 | 방법론 관련 | 실패 기반 모델-특화 편집, 회귀 테스트; 벤치마크가 아닌 방법 |
| HarnessCompass [58] | 2026 | 과적합 문제 관련 | 과적합·구성 요소 간섭 해결에 특화 |
| SEAGym [59] | 2026 | 평가 방법론 관련 | 중간 스냅샷, 비용, 분포 내외 결과 기록 |

**HarnessDev의 차별적 기여**:
1. Creation과 Evolution을 **하나의 통합 프레임워크**로 연결
2. **Creator와 Executor를 명시적으로 분리**하여 harness 품질과 모델 능력을 독립 측정
3. **실행 비용(토큰)**을 독립 지표로 측정
4. **모든 동결 버전에 대한 held-out 평가** (버전 선택 편향 분리)
5. **4개 다도메인** 커버리지

---

### 향후 연구에 미치는 영향 및 고려 사항

**연구에 미치는 영향**:

1. **평가 패러다임 전환**: "모델 능력"과 "harness 능력"을 분리 측정하는 새로운 평가 체계 수립. 이는 향후 모든 에이전트 벤치마크가 harness 변수를 명시적으로 통제해야 함을 요구
2. **비용 효율성 연구 촉진**: 성능-비용 트레이드오프가 harness 설계에서 최대 19배 차이를 보인다는 발견은 **경량 고성능 harness 설계**의 독립 연구 영역 형성
3. **Executor 전이 연구**: harness의 모델-중립성이 핵심 미해결 과제로 부상

**앞으로 연구 시 고려할 점**:

| 고려 사항 | 구체적 제안 |
|---|---|
| **통계적 엄밀성** | Evolution 실험에서 복수 궤적(≥3) 실행으로 신뢰 구간 제공 |
| **Held-out 범위 확대** | SWE-Pro 이외 도메인(BrowseComp, MLE-bench)에도 held-out 평가 적용 |
| **인과 분석** | 구조적 특성(체크포인팅, 메모리)과 성능 간 인과 관계 실험 설계 |
| **대안 기준선** | Evolution을 랜덤 서치, 그리디 서치 등과 matched-budget 비교 |
| **보안 격리** | 생성된 harness를 더 엄격한 샌드박스에서 실행 (Ethics statement의 제한 인정) |
| **재귀적 개선 가능성** | 진화된 harness를 다음 세대 Evolution의 개발 환경으로 활용 |
| **모델 규모 효과** | Creator 모델의 크기·능력과 harness 품질 간 스케일링 법칙 연구 |

---

## 참고자료

- **본 분석의 유일한 출처**: Wu, Y., Zhang, J., Shi, J., et al. (2026). *HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?* arXiv:2609.01437v1.
- 논문 내 인용 참고문헌 (선택적 명시):
  - [12] Deng et al. (2025). SWE-bench Pro. arXiv:2509.16941
  - [45] Terminal-Bench Team (2026). Terminal-Bench 2.1. https://www.tbench.ai/
  - [52] Wei et al. (2025). BrowseComp. arXiv:2504.12516
  - [7] Chan et al. (2024). MLE-bench. arXiv:2410.07095
  - [37] Paech (2025). EQ-Bench 3. https://github.com/EQ-bench/eqbench3
  - Project Page: https://self-developing-agents.github.io/
