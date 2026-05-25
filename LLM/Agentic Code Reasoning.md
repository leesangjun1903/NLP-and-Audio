# Agentic Code Reasoning

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

"Agentic Code Reasoning" (Ugare & Chandra, 2026, arXiv:2603.01896v2)은 **LLM 에이전트가 코드를 실제로 실행하지 않고도 코드베이스를 탐색하여 의미론적(semantic) 추론을 수행할 수 있는가**라는 질문에서 출발합니다.

이 논문은 **Semi-formal Reasoning(반형식 추론)**을 핵심 방법론으로 제안합니다. 이는 기존의 비구조적 Chain-of-Thought(CoT)와 완전 형식 검증(Lean, Coq 등) 사이의 중간 지점으로, 에이전트가 **명시적 전제(premises) → 실행 경로 추적(execution trace) → 형식적 결론(formal conclusion)**의 구조화된 템플릿을 따르도록 강제합니다.

### 1.2 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **방법론 제안** | Semi-formal Reasoning: 구조화된 프롬프트 템플릿을 통한 증명서(certificate) 기반 추론 |
| **패치 동등성 검증** | 정확도 78% → 88% (큐레이션 데이터셋), 실제 에이전트 생성 패치에서 93% 달성 |
| **코드 QA** | RubberDuckBench에서 87% 정확도 (+9pp over standard agentic) |
| **오류 위치 탐지** | Defects4J Top-5 정확도 최대 +12pp 향상 |
| **실용적 응용** | 실행 없는 RL 보상 신호(reward signal) 가능성 제시 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 2.1.1 핵심 문제 정의

기존 코드 에이전트 검증 방식에는 두 가지 극단이 존재합니다:

$$\underbrace{\text{비구조적 CoT}}_{\text{근거 없는 주장 허용}} \longleftrightarrow \underbrace{\text{완전 형식 검증 (Lean/Coq)}}_{\text{임의 코드에 적용 불가}}$$

- **비구조적 추론의 문제**: 에이전트가 근거 없이 주장을 생략하거나 추측 가능
- **완전 형식 검증의 문제**: 임의의 리포지토리 코드를 형식 언어로 변환하는 것이 비현실적
- **실행 기반 검증의 문제**: 샌드박스 환경 구성 비용, RL 학습 파이프라인에서의 연산 부담

#### 2.1.2 세 가지 평가 태스크

**태스크 1: 패치 동등성 검증 (Patch Equivalence Verification)**

$$\text{Definition 1: } P_1 \equiv P_2 \iff \forall t \in (F2P \cup P2P): \text{outcome}(P_1, t) = \text{outcome}(P_2, t)$$

두 패치 $P_1$, $P_2$가 테스트 스위트 $F2P \cup P2P$에서 동일한 pass/fail 결과를 낸다면 동등하다고 정의합니다.

- $F2P$: Fail-to-Pass 테스트 (버그 수정으로 새로 통과해야 하는 테스트)
- $P2P$: Pass-to-Pass 테스트 (기존에 통과하던 회귀 테스트)

**태스크 2: 오류 위치 탐지 (Fault Localization)**

Defects4J 벤치마크에서 실패 테스트만 주어졌을 때 버그 위치를 예측합니다.

$$\text{Top-N Accuracy} = \frac{|\{b \in \mathcal{B} : \text{GT}(b) \subseteq \text{Top-N Predictions}(b)\}|}{|\mathcal{B}|}$$

매칭 조건: $\text{pred start} \leq \text{region end}$ AND $\text{pred end} \geq \text{region start}$

**태스크 3: 코드 질의응답 (Code QA)**

RubberDuckBench: Python, Java, C++ 리포지토리에 대한 15개의 전문가 작성 질문으로 구성.

---

### 2.2 제안 방법: Semi-formal Reasoning

#### 2.2.1 방법론의 핵심 원리

Semi-formal Reasoning은 다음 원칙을 강제합니다:

$$\text{Evidence}(e_1, e_2, \ldots, e_n) \xrightarrow{\text{structured template}} \text{Conclusion}(c)$$

즉, 결론 $c$는 반드시 명시적으로 수집된 증거 $\{e_i\}$로부터 도출되어야 합니다.

#### 2.2.2 패치 동등성 검증 템플릿 구조

```
DEFINITIONS:
  D1: Two patches are EQUIVALENT MODULO TESTS iff test suite produces 
      identical pass/fail outcomes for both patches.

PREMISES:
  P1: Patch 1 modifies [file(s)] by [change]
  P2: Patch 2 modifies [file(s)] by [change]
  P3: F2P tests check [behavior]

ANALYSIS OF TEST BEHAVIOR:
  For each test:
    Claim 1.1: Patch 1 → test [name] PASS/FAIL because [execution trace]
    Claim 1.2: Patch 2 → test [name] PASS/FAIL because [execution trace]
    Comparison: SAME/DIFFERENT

COUNTEREXAMPLE (if NOT EQUIVALENT):
  Test [name] → different outcomes because [code trace evidence]

FORMAL CONCLUSION:
  By D1, since outcomes are [IDENTICAL/DIFFERENT],
  patches are [EQUIVALENT/NOT EQUIVALENT].
```

#### 2.2.3 오류 위치 탐지 템플릿 (4단계 체계)

```
Phase 1: PREMISE T1, T2, ... (테스트 의미론 분석)
Phase 2: METHOD → LOCATION → BEHAVIOR → RELEVANT (코드 경로 추적)
Phase 3: CLAIM D1: At [file:line], contradicts PREMISE T[N] because [reason]
Phase 4: Ranked Predictions citing CLAIM(s)
```

이 구조는 다음 불변식을 보장합니다:

$$\text{Prediction} \xleftarrow{\text{cites}} \text{CLAIM} \xleftarrow{\text{references}} \text{PREMISE}$$

#### 2.2.4 코드 QA 템플릿

```
FUNCTION TRACE TABLE: | Function | File:Line | Param Types | Return | Behavior(VERIFIED) |
DATA FLOW ANALYSIS: Created/Modified/Used at [file:line]
SEMANTIC PROPERTIES: Property + Evidence [file:line]
ALTERNATIVE HYPOTHESIS CHECK: Searched / Found / Conclusion
```

#### 2.2.5 Agentic vs. Single-Shot 설정

| 구분 | Single-Shot | Agentic |
|------|-------------|---------|
| 입력 | 정적 코드 스냅샷 | 리포지토리 탐색 (bash 도구) |
| 최대 스텝 수 | 1 | 100 |
| 코드 실행 | 불가 | 불가 (독립 Python 스크립트만 가능) |
| git 명령 | - | 비활성화 |

---

### 2.3 모델 구조

이 논문은 **새로운 모델 아키텍처를 학습시키는 것이 아닌**, 기존 LLM에 구조화된 프롬프트 템플릿을 적용하는 **프롬프팅 방법론**입니다.

$$\underbrace{\text{LLM}(\text{Claude Opus-4.5, Sonnet-4.5})}_{\text{기반 모델}} + \underbrace{\text{SWE-agent framework}}_{\text{에이전트 인터페이스}} + \underbrace{\text{Semi-formal Template}}_{\text{핵심 기여}} \rightarrow \text{Agentic Code Reasoner}$$

**에이전트 구성 요소:**
- **LLM 백본**: Claude Opus-4.5, Claude Sonnet-4.5
- **에이전트 프레임워크**: SWE-agent (Yang et al., 2024) 기반 미니멀 설정
- **도구**: bash 명령어 (파일 읽기, grep, 검색 등)
- **추론 형식**: Standard vs. Semi-formal (템플릿 프롬프팅)

---

### 2.4 성능 향상 결과

#### 2.4.1 패치 동등성 - 큐레이션 데이터셋 (170개)

| 추론 방식 | 전체 정확도 | 평균 스텝 | 비동등 정확도 | 동등 정확도 |
|-----------|------------|----------|--------------|------------|
| Standard | 78.2% | 10.08 | 78.6% | 78.0% |
| Semi-formal | **88.8%** | 28.17 | 82.9% | **93.0%** |

$$\Delta\text{Accuracy} = 88.8\% - 78.2\% = +10.6\text{pp}, \quad \text{Cost: } \frac{28.17}{10.08} \approx 2.8\times \text{ more steps}$$

#### 2.4.2 패치 동등성 - LLM 생성 패치 (200개, 균형 샘플)

| 모델 | 방식 | 정확도 | 평균 스텝 |
|------|------|--------|----------|
| difflib (No LLM) | threshold=0.4 | 73% | - |
| Sonnet-4.5 | Single Call | 80.0% | 1 |
| Sonnet-4.5 | Agentic (Semi-formal) | **91.5%** | 43.23 |
| Opus-4.5 | Single Call | 86.0% | 1 |
| Opus-4.5 | Agentic (Semi-formal) | **93.0%** | 37.82 |

#### 2.4.3 오류 위치 탐지 (Defects4J, 50개 소규모)

| 방식 | 탐색 | Top-1 (All) | Top-5 (All) | Top-5 (Any) |
|------|------|-------------|-------------|-------------|
| Standard | Single-shot | 36.1% | 55.6% | 69.4% |
| Semi-formal | Single-shot | 41.7% | **63.9%** | **77.8%** |
| Standard | Agentic | 46.5% | 60.5% | 81.4% |
| Semi-formal | Agentic | **53.5%** | **72.1%** | **88.4%** |

#### 2.4.4 오류 위치 탐지 (Defects4J, 100개 대규모, 90개 평가 가능)

| 모델 | 방식 | Top-1 (All) | Top-5 (All) | Top-5 (Any) |
|------|------|-------------|-------------|-------------|
| Opus-4.5 | Standard | 30.0% | 43.3% | 65.6% |
| Opus-4.5 | Semi-formal | **34.4%** | **47.8%** | **68.9%** |

$$\Delta\text{Top-5 (All)} = 47.8\% - 43.3\% = +4.5\text{pp (대규모)}, \quad +11.6\text{pp (소규모)}$$

#### 2.4.5 코드 질의응답 (RubberDuckBench, 15개)

| 모델 | 방식 | 정확도 | 평균 스텝 |
|------|------|--------|----------|
| Opus-4.5 | Single-shot | 76.2% | 1 |
| Opus-4.5 | Agentic (Standard) | 78.3% | 10.8 |
| Opus-4.5 | Agentic (Semi-formal) | **87.0%** | 19.7 |
| Sonnet-4.5 | Agentic (Standard) | 84.2% | 17.6 |
| Sonnet-4.5 | Agentic (Semi-formal) | 84.8% | 25.3 |

> **주목할 점**: Sonnet-4.5에서는 Semi-formal의 이점이 거의 없음 (+0.6pp). 이는 기반 모델의 역량이 충분히 강할 때 구조화 추론의 이점이 포화될 수 있음을 시사합니다.

### 2.5 한계점

| 한계 | 설명 |
|------|------|
| **계산 비용 증가** | Semi-formal은 Standard 대비 약 2~4배 더 많은 스텝 소요 |
| **불완전한 실행 추적** | 에이전트가 모든 코드 경로를 추적하지 못하는 경우 발생 |
| **제3자 라이브러리** | 소스 코드 없는 외부 라이브러리 의미론 추측 |
| **간접 버그** | 테스트가 직접 호출하지 않는 클래스의 버그 탐지 어려움 |
| **다파일 버그** | 5개 이상 파일에 걸친 버그는 체계적으로 어려움 |
| **데이터 오염 가능성** | SWE-bench 인스턴스가 LLM 학습 데이터에 포함 가능 |
| **소규모 QA 벤치마크** | RubberDuckBench는 15개 질문으로 통계적 신뢰성 제한 |
| **모델 의존성** | 강한 기반 모델(Sonnet-4.5)에서 Semi-formal 효과 포화 |
| **확신에 찬 오답** | 상세한 추적이 오히려 틀린 답에 대한 높은 확신 유도 가능 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 태스크 독립적 일반화

Semi-formal Reasoning의 핵심 일반화 원리는 **템플릿 구조의 보편성**입니다:

$$\underbrace{\text{Premises} \rightarrow \text{Claims} \rightarrow \text{Conclusion}}_{\text{공통 구조}} \begin{cases} \text{Patch Equiv: per-test traces} \\ \text{Fault Loc: divergence claims} \\ \text{Code QA: function trace table} \end{cases}$$

세 가지 서로 다른 태스크에서 일관된 성능 향상을 보여줌으로써, 이 방법론이 특정 태스크에 과적합된 것이 아님을 입증합니다.

### 3.2 언어 및 프레임워크 일반화

| 측면 | 근거 |
|------|------|
| **다중 언어** | Python, Java, C++ 리포지토리에서 모두 평가 (Table 7) |
| **다중 프레임워크** | Django, Mockito 등 다양한 실제 오픈소스 프로젝트 |
| **언어별 정확도** | C++(88.0%), Python(87.5%), Java(85.5%) - 편차 낮음 |

### 3.3 실세계 데이터 일반화

$$\underbrace{\text{Curated Dataset}(n=170)}_{\text{어려운 경계 케이스}} \rightarrow 88.8\%$$
$$\underbrace{\text{Real-world Agent Patches}(n=200)}_{\text{무작위 샘플링}} \rightarrow 93.0\%$$

실제 에이전트 생성 패치에서 오히려 더 높은 성능을 보이는 것은, **Semi-formal Reasoning이 실제 분포에서도 강건함**을 시사합니다.

### 3.4 일반화의 메커니즘

Semi-formal Reasoning이 일반화 성능을 향상시키는 근본적인 이유:

1. **인터프로시저럴 추론 강제**: 함수 호출 체인을 실제로 따라가도록 유도
   - Django 예시: `format()` 호출이 Python 내장 함수가 아닌 모듈 레벨 함수임을 발견
   
2. **확증 편향 감소**: 결론 전에 반드시 반례(counterexample) 검토 요구
   $$\text{bias}(\text{semi-formal}) < \text{bias}(\text{standard})$$

3. **지식 외 코드에서의 강건성**: 에이전트가 기억이 아닌 실시간 탐색으로 추론하므로 데이터 오염 효과 감소

### 3.5 일반화의 한계와 조건

```
일반화 성능이 높은 경우:
✓ 기반 모델이 적당히 강할 때 (Opus-4.5)
✓ 코드 경로가 비교적 명확한 단일 파일 버그
✓ 제3자 라이브러리 의존성이 낮을 때

일반화가 제한되는 경우:
✗ 기반 모델이 매우 강할 때 (Sonnet-4.5, 포화 효과)
✗ 도메인 특화 알고리즘 버그 (수치 해석 등)
✗ 5개 이상 파일에 걸친 복잡한 다중 위치 버그
```

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 실행 없는 코드 검증 관련 연구

| 연구 | 방법 | 특징 | Agentic Code Reasoning과의 차이 |
|------|------|------|----------------------------------|
| **SWE-RM** (Shum et al., 2025) | 보상 모델 학습 | 테스트 결과 근사 | 학습 기반 vs. 프롬프팅 기반 |
| **Agentic Rubrics** (Raghavendra et al., 2026) | LLM 생성 루브릭 | 검증 기준 분해 | 비구조적 추론 vs. 반형식 구조 |
| **CodeJudge** (Tong & Zhang, 2024) | LLM-as-judge | 코드 품질 평가 | 단일 판단 vs. 증명서 기반 |
| **SWE-RL** (Wei et al., 2025b) | RL + difflib 유사도 | 보상 신호 | 표면적 유사도(73%) vs. 의미론적 추론(93%) |

### 4.2 오류 위치 탐지 관련 연구

| 연구 | 방법 | 성능 특징 | 비교 포인트 |
|------|------|-----------|------------|
| **AgentFL** (Qin et al., 2025) | LLM 에이전트 기반 | 프로젝트 레벨 FL | 비구조적 추론 |
| **FlexFL** (Xu et al., 2025) | 오픈소스 LLM | 유연한 FL | 오픈소스 모델 활용 |
| **이 논문** | Semi-formal Agentic | Top-5: 72.1% (All) | 구조화 추론 + 에이전트 탐색 |

### 4.3 형식 검증 관련 연구

| 연구 | 형식 언어 | 적용 범위 | 한계 |
|------|-----------|-----------|------|
| **Baldur** (First et al., 2023) | Lean | 수학적 증명 | 임의 코드 불가 |
| **Cobblestone** (Kasibatla et al., 2025) | Coq | 분할정복 검증 | 형식화 비용 높음 |
| **VERINA** (Ye et al., 2025) | Lean | 검증 가능 코드 생성 | 제한된 언어 |
| **Verified Code Reasoning** (Sistla et al., 2025) | Datalog | 사후 검증 | 출력 검증 (입력 측 개선 아님) |
| **이 논문** | 자연어 템플릿 | 임의 리포지토리 | 자동 검증 불가 |

### 4.4 Chain-of-Thought 및 추론 개선 연구

| 연구 | 방법 | 도메인 | 이 논문과의 관계 |
|------|------|--------|----------------|
| **CoT Prompting** (Wei et al., 2022) | 중간 추론 단계 | 수학 문제 | 기반 아이디어, 코드로 확장 |
| **ReAct** (Yao et al., 2023) | 추론 + 행동 결합 | 에이전트 태스크 | 에이전트 설계 참고 |
| **CodeAct** (Wang et al., 2024) | 실행 가능 코드 행동 | 에이전트 | 코드 행동 개선 |
| **Aristotle** (Achim et al., 2025) | 반형식 수학 추론 | IMO 수학 | Semi-formal 수학 → 코드로 |
| **Olympiad RL** (Hubert et al., 2025) | RL + 형식 추론 | 올림피아드 수학 | 수학의 성공을 코드 영역으로 |
| **이 논문** | Semi-formal 코드 추론 | 소프트웨어 공학 | 수학 반형식 → 코드 적용 |

### 4.5 EquiBench와의 직접 비교

**EquiBench** (Wei et al., 2025a):
- 프로그램 동등성에 대한 LLM 추론 벤치마크
- **소규모 독립 코드 쌍** 집중
- 테스트 스위트 미사용

**이 논문**:
- **리포지토리 레벨 패치** (실제 맥락 포함)
- F2P/P2P 테스트 스위트 활용
- 에이전트 탐색 + 반형식 추론

---

## 5. 향후 연구에 미치는 영향 및 고려사항

### 5.1 향후 연구에 미치는 영향

#### 5.1.1 RL 학습 파이프라인에의 영향

```
기존: 생성 → 실행(샌드박스) → 보상
                    ↑ 비용 높음

제안: 생성 → Semi-formal 검증 → 보상
              ↑ 실행 없이 93% 정확도
```

$$\text{Cost Reduction} \approx \frac{\text{Sandbox Setup Cost}}{\text{LLM API Cost}} \gg 1$$

이는 SWE-Gym, R2E-Gym과 같은 RL 기반 SWE 에이전트 학습 파이프라인에서 **검증 비용을 크게 줄일 수 있는 가능성**을 제시합니다.

#### 5.1.2 정적 분석 도구 패러다임 전환

기존 정적 분석 도구는 특정 언어/프레임워크에 특화된 알고리즘이 필요했지만, 이 논문은 다음을 보여줍니다:

$$\underbrace{\text{Task-specific Algorithm}}_{\text{기존 정적 분석}} \rightarrow \underbrace{\text{Task-specific Template + LLM Agent}}_{\text{새로운 패러다임}}$$

이 패러다임은 **보안 취약점 탐지**, **코드 스멜 식별**, **API 오용 탐지** 등으로 확장 가능합니다.

#### 5.1.3 코드 리뷰 자동화

Semi-formal 증명서는 **인간이 검토하기 쉬운 형태의 코드 분석 결과**를 제공하므로, 자동화된 코드 리뷰 시스템의 신뢰성을 높이는 데 기여할 수 있습니다.

### 5.2 향후 연구 시 고려할 점

#### 5.2.1 포스트 트레이닝 적용

논문이 제안하는 미래 방향:

$$\text{Fine-tuning on Semi-formal Traces} \rightarrow \text{Internalized Structure}$$

- 현재: 프롬프트 오버헤드(2~4배 스텝 증가)
- 미래: 파인튜닝을 통해 구조를 내재화하면 프롬프트 없이도 구조적 추론 가능

**고려사항**: 파인튜닝 데이터 구성 방식, semi-formal trace의 품질 보장 메커니즘

#### 5.2.2 하이브리드 검증 아키텍처

$$\underbrace{\text{LLM Semi-formal Reasoning}}_{\text{유연성, 언어 무관}} + \underbrace{\text{Symbolic Execution / SMT Solver}}_{\text{형식적 보장}} \rightarrow \text{Hybrid Verifier}$$

- Semi-formal 추론으로 후보 경로 생성
- 경량 형식 도구로 핵심 클레임 검증
- Sistla et al. (2025)의 Datalog 기반 사후 검증과 결합 가능

#### 5.2.3 모델 역량과 템플릿 복잡성의 상관관계

**중요 발견**: Sonnet-4.5에서 Semi-formal의 이점이 포화됨

$$\text{Benefit}(\text{semi-formal}) = f(\text{Task Complexity}) - g(\text{Model Capability})$$

향후 연구에서는 **어떤 역량 수준의 모델에서 어느 정도 복잡도의 템플릿이 최적인가**를 체계적으로 연구할 필요가 있습니다.

#### 5.2.4 벤치마크 확장 필요성

| 현재 한계 | 개선 방향 |
|-----------|-----------|
| RubberDuckBench 15개 질문 | 수백 개 이상의 대규모 코드 QA 벤치마크 |
| Python/Java/C++ 한정 | Go, Rust, TypeScript 등으로 확장 |
| 단일 저자 생성 패치 | 다양한 에이전트 생성 패치 분포 |

#### 5.2.5 확신에 찬 오답(Confident Wrong Answers) 문제

Semi-formal Reasoning의 역설적 위험:

$$\text{Structured Reasoning} \rightarrow \text{Higher Confidence} \rightarrow \text{More Dangerous When Wrong}$$

py_5 예시에서 에이전트는 5개 함수를 추적했지만 다운스트림 처리를 놓쳐 **높은 확신의 오답**을 생성했습니다. 향후 연구에서는 다음을 고려해야 합니다:

- **불확실성 보정(Uncertainty Calibration)**: Semi-formal 추론의 자신감과 실제 정확도 간 관계 연구
- **완전성 검증(Completeness Check)**: 모든 관련 코드 경로가 추적되었는지 검증하는 메커니즘

#### 5.2.6 데이터 오염 문제의 체계적 해결

```
현재 접근: "상대적 비교이므로 모든 설정에 동등하게 영향"
개선 방향: 오염되지 않은 신규 벤치마크 구성
          + 에이전트 탐색 로그 분석으로 실시간 추론 vs. 기억 구분
```

#### 5.2.7 비용-성능 트레이드오프 최적화

$$\text{Efficiency} = \frac{\Delta\text{Accuracy}}{\Delta\text{Steps}} = \frac{10.6\text{pp}}{28.17 - 10.08} = \frac{10.6}{18.09} \approx 0.59\text{pp/step}$$

적응적 추론 깊이 결정: 쉬운 케이스에서는 Standard, 어려운 케이스에서만 Semi-formal을 선택적으로 적용하는 메타 전략 연구가 필요합니다.

---

## 참고 자료

### 주요 논문 (직접 인용)

1. **Ugare, S., & Chandra, S. (2026).** *Agentic Code Reasoning.* arXiv:2603.01896v2 [cs.SE]. https://arxiv.org/abs/2603.01896

2. **Just, R., Jalali, D., & Ernst, M. D. (2014).** *Defects4J: A Database of Existing Faults to Enable Controlled Testing Studies for Java Programs.* ISSTA 2014.

3. **Mohammad, F., Ayad, F., Maniatis, P., Chandra, S., & Dinella, E. (2026).** *RubberDuckBench: A Benchmark for AI Coding Assistants.*

4. **Yang, J., et al. (2024).** *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.* NeurIPS 2024.

5. **Wei, J., et al. (2022).** *Chain-of-thought prompting elicits reasoning in large language models.* NeurIPS 2022.

6. **Yao, S., et al. (2023).** *ReAct: Synergizing Reasoning and Acting in Language Models.* arXiv:2210.03629.

7. **Shum, K., et al. (2025).** *SWE-RM: Execution-free Feedback For Software Engineering Agents.* arXiv:2512.21919.

8. **Wei, Y., et al. (2025b).** *SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution.* NeurIPS 2025.

9. **Wei, A., et al. (2025a).** *EquiBench: Benchmarking Large Language Models' Reasoning about Program Semantics via Equivalence Checking.* EMNLP 2025.

10. **Sistla, M., et al. (2025).** *Towards Verified Code Reasoning by LLMs.* arXiv:2509.26546.

11. **Tong, W., & Zhang, T. (2024).** *CodeJudge: Evaluating Code Generation with Large Language Models.* arXiv:2410.02184.

12. **Raghavendra, M., et al. (2026).** *Agentic Rubrics as Contextual Verifiers for SWE Agents.* arXiv:2601.04171.

13. **Pan, J., et al. (2025).** *Training Software Engineering Agents and Verifiers with SWE-Gym.* arXiv:2412.21139.

14. **Jain, N., et al. (2025).** *R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents.* arXiv:2504.07164.

15. **Qin, Y., et al. (2025).** *AgentFL: Scaling LLM-based Fault Localization to Project-Level Context.* arXiv:2403.16362.

16. **Xu, C., et al. (2025).** *FlexFL: Flexible and Effective Fault Localization with Open-Source Large Language Models.* arXiv:2411.10714.

17. **Sultan, O., et al. (2026).** *LLMs versus the Halting Problem: Revisiting Program Termination Prediction.* arXiv:2601.18987.

18. **Wang, X., et al. (2024).** *Executable Code Actions Elicit Better LLM Agents.* arXiv:2402.01030.

19. **Xia, C. S., et al. (2024).** *Agentless: Demystifying LLM-based Software Engineering Agents.* arXiv:2407.01489.

20. **Hubert, T., et al. (2025).** *Olympiad-Level Formal Mathematical Reasoning with Reinforcement Learning.* Nature.
