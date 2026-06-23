# Agentic Framework for Deep Learning Workload Migration via In-Context Learning

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 PyTorch에서 JAX로의 딥러닝 코드 마이그레이션을 **완전 자동화**하기 위한 에이전틱(Agentic) 프레임워크를 제안합니다. LLM은 복잡한 텐서 연산에 대해 hallucination을 일으키기 쉬우므로, **In-Context Learning(ICL)** 과 **Oracle 기반 자기 디버깅(self-debugging)** 을 결합하여 이 문제를 극복할 수 있다고 주장합니다.

### 주요 기여 (3가지)

| 기여 항목 | 내용 |
|---|---|
| **① ICL 기반 구조적 앵커링** | 소수의 정제된 참조 번역 예제를 ICL 컨텍스트로 제공하여 hallucination 위험 감소 |
| **② Dynamic Execution Oracle** | PyTorch 모듈을 실제 실행하여 $state_{dict}$, 입력 텐서, 출력 활성화를 직렬화한 불변 Oracle 생성 |
| **③ 에이전틱 워크플로우를 위한 Ablation 가이드** | ICL, 명령어, 실행 기반 자기디버깅의 기여도를 개별 측정하여 실용적 설계 지침 제공 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**PyTorch → JAX 마이그레이션의 핵심 난점:**

- **패러다임 차이**: PyTorch의 객체지향·가변 상태(mutable state) vs. JAX의 함수형·무상태(stateless) 설계
- **텐서 레이아웃 불일치**: PyTorch는 `NCHW`, JAX는 `NHWC`
- **기본 정밀도 차이**: PyTorch는 `fp32`, JAX는 `bfloat16`
- **제어 흐름 차이**: Python 표준 제어 흐름 vs. `jax.lax.scan` 등 컴파일러 친화적 구조
- **LLM의 산술 한계**: LLM은 고차원 텐서 연산에서 hallucination 발생
- **RAG 파이프라인의 실패**: MaxText 레포지토리 기반 RAG 시도 시 "Lost in the Middle" 현상([4]) 및 분산 컴퓨팅 노이즈 문제 발생

---

### 2.2 제안하는 방법 (4단계 파이프라인)

#### Phase 1: ICL 기반 구조적 앵커링

RAG 대신 **소수·고밀도** ICL 컨텍스트를 사용:
- `gold_refs/torch` (PyTorch 예제)
- `gold_refs/jax` (JAX 번역 예제)
- `gold_refs/jax/test_templates` (테스트 템플릿)

이를 통해 LLM의 사전학습 패턴을 덮어쓰는 **태스크 벡터(task vector)**([3])를 형성하여 구문적 앵커 역할 수행.

#### Phase 2: Dynamic Execution Oracle 생성

LLM의 산술 추론에 의존하는 대신, PyTorch 모듈을 **직접 실행**하여 계산 상태를 추출:

$$\text{Oracle} = \{state_{dict}, X_{input}, Y_{output}\} \quad \text{serialized as} \quad \texttt{data/.pkl}$$

이 Oracle은 **불변(immutable) 수학적 ground truth** 로 기능.

#### Phase 3: 에이전틱 번역 및 Oracle 조건부 테스트 생성

ICL 앵커를 기반으로 `flax.linen` 추상화로 번역 후, Oracle 데이터에 기반한 테스트 하네스(test harness) 자동 생성. 평가는 3단계 계층적 프로세스:

1. **컴파일 성공 여부** (pass/fail)
2. **Shape 일치 여부** (NCHW → NHWC 등 레이아웃 변환 확인)
3. **수치 동등성 검사** (`numpy.allclose` 기반)

수치 동등성의 공식적 정의:

$$\max \left| f_{pt}(X, W_{pt}) - f_{jax}(X, W_{jax}) \right| < \epsilon$$

여기서 $\epsilon = 1 \times 10^{-7}$ 은 백엔드 컴파일러 간 표준 부동소수점 오차를 허용하는 임계값.

이를 **"Silver" 테스트 케이스**라 명명.

#### Phase 4: 반복적 자기 디버깅

생성된 모듈이 Oracle 조건부 테스트를 통과하지 못할 경우:

$$\text{Feedback} = \{\text{stderr}, \text{stack trace}, \text{numerical diff}\} \xrightarrow{\text{LLM context}} \text{Refined JAX Code}$$

이 루프를 반복하여 코드를 수정. 단, **reward hacking 방지**를 위해 최종 평가는 **별도의 수동 검증 스위트(manual evaluation suite)** 에서만 측정.

---

### 2.3 모델 구조

```
[Phase 1: Setup]
  gold_refs (PyTorch + JAX + 테스트 템플릿)
          ↓ ICL Context + Prompt Template
[Phase 2: Oracle 생성]
  torch_tests 실행 → data/.pkl (Oracle)
          ↓
[Phase 3: Translation]
  JAX Code Assist Agent → JAX Code + Generated Tests
          ↓ Load Ground Truth (Oracle)
[Phase 4: Self-Debugging Loop]
  Test Execution → Error Report → LLM Self-Correction → 반복
          ↓ (수렴 시)
  Final Manual Evaluation
```

---

### 2.4 성능 향상

**Ablation Study 결과 (Table 1):**

| 구성 | Level 1 수치 동등성 | Level 2 수치 동등성 |
|---|---|---|
| Baseline Only | 44% | 9% |
| Instruction Only | 44% | 18% |
| Instruction + Self-Debugging | 89% | 27% |
| **Full Pipeline (Ours)** | **100%** | **91%** |

**Level 3 (전체 레포지토리) 결과 (Table 2, 인간 평가):**

| 레포지토리 | 완성도 | 수치 동등성 | 가독성(1-5) |
|---|---|---|---|
| Code Whisper | 100% | 100% | 4.0 |
| T5 | 100% | 100% | 3.75 |
| Two-tower | 100% | 100% | 4.9 |
| SAM2 | 86% | 80% | 4.8 |
| DETR | 100% | 57% | 3.8 |

**번역 실패 유형 분류:**

- **Syntactic Hallucinations**: 필요한 wrapper 클래스 미생성, `NameError` 발생
- **API Signature Mismatches**: PyTorch 전용 kwargs를 JAX에 잘못 적용
- **State Mapping Discrepancies**: 가중치 전달 단계에서 `KeyError` 발생

---

### 2.5 한계점

| 한계 | 설명 |
|---|---|
| **레포지토리 수준 완전 자동화 미달** | Level 3에서 인간 전문가의 1-2회 수동 개입 필요 |
| **ICL 예제 민감성 미탐색** | ICL에 사용되는 구체적 예제가 결과에 미치는 영향 체계적 분석 부재 |
| **복잡한 의존성 관리 한계** | SAM2(80%), DETR(57%) 등 Facebook Research의 대규모 foundational 모델에서 성능 저하 |
| **평가 규모의 제한** | Level 1: 9개 연산, Level 2: 11개 모듈로 평가 집합이 상대적으로 소규모 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 핵심 설계 원칙

이 논문의 일반화 성능 향상은 다음 세 가지 메커니즘에 의해 지지됩니다:

**① ICL 기반 구조적 일반화:**

ICL은 LLM의 사전학습 패턴 위에 **태스크 벡터(task vector)**([3])를 형성합니다:

$$\vec{v}_{task} = \text{ICL}(\{(x_i^{pt}, x_i^{jax}, t_i)\}_{i=1}^{k})$$

여기서 $x_i^{pt}$는 PyTorch 예제, $x_i^{jax}$는 JAX 번역, $t_i$는 테스트 케이스. 소수의 정제된 예제만으로도 LLM이 새로운 모듈에 대해 올바른 JAX 패턴을 적용할 수 있음을 ablation으로 입증.

**② Oracle 기반 수학적 일반화:**

LLM의 파라메트릭 지식에 의존하지 않고, **실행 기반 검증(execution-based evaluation)**([1])을 통해 모든 새로운 모듈에 동일한 수학적 ground truth 기준을 적용:

$$\text{Pass} \iff \max|f_{pt}(X, W_{pt}) - f_{jax}(X, W_{jax})| < 10^{-7}$$

이는 특정 모델 아키텍처에 과적합되지 않는 **범용 검증 기준**.

**③ Self-Debugging 루프의 일반화:**

에이전틱 루프는 오류 유형에 관계없이 동일한 피드백 메커니즘을 적용:

$$\text{Code}_{t+1} = \text{LLM}(\text{Code}_t, \text{Feedback}_t, \text{ICL context})$$

이는 사전에 오류 패턴을 정의하지 않고도 새로운 API 불일치나 수치 오류를 자율적으로 수정.

### 3.2 오염 방지를 통한 일반화 측정 신뢰성

논문은 **ICL에 사용된 모듈이 평가 세트에 등장하지 않도록** 엄격히 분리:

$$\text{ICL Set} \cap \text{Evaluation Set} = \emptyset$$

이를 통해 보고된 91% 수치 동등성이 **진정한 일반화(uncontaminated generalization)** 를 반영함을 보장.

### 3.3 일반화의 현재 한계와 개선 가능성

- **현재 한계**: DETR(57%), SAM2(80%) 등 복잡한 의존성을 가진 대형 모델에서 일반화 저하
- **개선 방향**: 레포지토리 수준의 의존성 그래프(dependency graph) 분석을 ICL 컨텍스트에 통합 시 일반화 향상 기대

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 향후 연구에 미치는 영향

**① 코드 마이그레이션 자동화 패러다임 전환**

기존의 규칙 기반 또는 단순 LLM 프롬프팅 방식에서 **Oracle 기반 실행 검증 + 에이전틱 루프** 패러다임으로의 전환을 제시합니다. 이는 PyTorch↔JAX뿐 아니라 TensorFlow→JAX([6]), PyTorch→TensorFlow 등 다른 프레임워크 쌍으로의 확장 가능성을 시사합니다.

**② LLM의 코드 생성 신뢰성 연구에 기여**

LLM이 단독으로는 고차원 텐서 연산을 신뢰할 수 없다는 점을 실증적으로 보여주며([2]), **실행 기반 피드백(execution-grounded feedback)** 이 LLM 코드 생성 신뢰성을 높이는 핵심임을 확인. 이는 Codex([1]), Code Llama([7]) 등의 코드 생성 LLM 연구에 직접적인 영향을 미칩니다.

**③ 에이전틱 AI 시스템 설계 원칙 제시**

Reward hacking 방지를 위한 **디버깅 루프와 최종 평가의 분리**, ICL 앵커링과 실행 피드백의 결합 등은 향후 에이전틱 시스템 설계의 중요한 참조 사례가 됩니다.

**④ TPU/가속기 생태계 접근성 향상**

Google TPU의 JAX 생태계로의 진입 장벽을 낮춰, 더 많은 연구자들이 가속기 최적화 환경을 활용할 수 있는 기반을 마련합니다.

### 4.2 향후 연구 시 고려해야 할 점

**① ICL 예제 선택 전략의 체계적 탐구**

현재 논문은 ICL 예제의 구성이 성능에 미치는 영향을 체계적으로 분석하지 않았습니다. 향후 연구에서는 다음을 고려해야 합니다:
- 예제의 다양성(diversity) vs. 난이도(difficulty) 균형
- 자동화된 예제 선택 알고리즘 (예: 커버리지 기반 선택)
- 예제 수(k-shot)에 따른 성능 변화 곡선

**② 레포지토리 수준 의존성 관리**

Level 3에서 SAM2(80%), DETR(57%)의 낮은 성능은 **복잡한 모듈 간 의존성** 때문입니다. 향후 연구에서는:
- AST(Abstract Syntax Tree) 기반 의존성 그래프 자동 추출
- 모듈별 순차적 마이그레이션 전략
- 크로스 파일 참조 해결 메커니즘

**③ 정밀도 불일치 문제의 정교한 처리**

$\epsilon = 10^{-7}$ 임계값은 fp32 기준입니다. bfloat16 기반 JAX 환경에서는 더 큰 수치 오차가 발생할 수 있으므로, **정밀도 인식(precision-aware) 검증 기준** 이 필요합니다:

$$\epsilon_{adaptive} = f(\text{dtype}_{pt}, \text{dtype}_{jax}, \text{op complexity})$$

**④ Reward Hacking 방지 메커니즘 강화**

현재는 수동 평가 스위트로 reward hacking을 방지하지만, 이는 확장성이 낮습니다. 자동화된 **adversarial 테스트 생성** 또는 **형식 검증(formal verification)** 방법 연구가 필요합니다.

**⑤ 다양한 LLM 백본에 대한 일반화 검증**

현재 논문은 특정 LLM(Google 내부 모델로 추정)을 사용하며 LLM 종류를 명시하지 않습니다. 다양한 LLM(GPT-4, Claude, Code Llama 등)에서 동일한 파이프라인의 성능 검증이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 강점 | 본 논문과의 차이 |
|---|---|---|---|
| **Codex / HumanEval** (Chen et al., 2021)[1] | LLM 코드 생성, pass@k 평가 | 범용 코드 생성 능력 | 실행 검증 기반이나 framework 특화 마이그레이션 없음 |
| **Chain-of-Thought** (Wei et al., 2022)[9] | 단계적 추론 프롬프팅 | 추론 능력 향상 | 수학적 oracle 없이 LLM 내부 추론에 의존, hallucination 미해결 |
| **Self-Consistency** (Wang et al., 2022)[8] | 다수 추론 경로 집계 | 추론 신뢰성 향상 | 실행 검증 없음, 코드 마이그레이션 특화 아님 |
| **Lost in the Middle** (Liu et al., 2024)[4] | 긴 컨텍스트 처리 분석 | LLM 컨텍스트 활용 한계 규명 | 본 논문은 이 문제를 ICL 제한으로 우회 |
| **In-context learning creates task vectors** (Hendel et al., 2023)[3] | ICL의 내부 메커니즘 분석 | ICL이 task vector 형성 설명 | 본 논문의 ICL 설계의 이론적 근거 |
| **TF→JAX Multi-agent** (Nikolov et al., 2026)[6] | 다중 에이전트 TensorFlow→JAX | 다중 에이전트 협력 | 본 논문은 단일 에이전트 + Oracle로 더 단순하고 높은 성능 |
| **Code Llama** (Roziere et al., 2023)[7] | 코드 특화 LLM | 다양한 코드 태스크 | 프레임워크 특화 마이그레이션 및 실행 검증 미지원 |
| **Faith and Fate** (Dziri et al., 2023)[2] | Transformer의 compositionality 한계 분석 | LLM 산술 한계 규명 | 본 논문이 oracle로 해결하는 문제의 이론적 배경 |

### 비교 분석 요약

본 논문은 기존 연구들의 한계를 다음과 같이 극복합니다:

1. **Chain-of-Thought/Self-Consistency 대비**: LLM 내부 추론 대신 **외부 실행 oracle**로 수학적 정확성 보장
2. **RAG 기반 접근 대비**: 대규모 컨텍스트 검색 대신 **소규모 고품질 ICL**로 "Lost in the Middle" 문제 회피
3. **다중 에이전트([6]) 대비**: 단일 에이전트 + oracle 구조로 더 단순하면서도 높은 수치 동등성 달성

---

## 참고 자료

**본 논문 (주요 분석 대상):**
- Qiyue Liang, Steven Ingram, George Vanica, Andi Gavrilescu, Newfel Harrat, Hassan Sipra, Sethuraman Sankaran. "Agentic Framework for Deep Learning workload migration via In-Context Learning." *arXiv:2606.15994v1*, 2026.
- GitHub: https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxCode

**논문 내 인용 참고문헌:**
- [1] Chen et al., "Evaluating large language models trained on code (Codex)." *arXiv:2107.03374*, 2021.
- [2] Dziri et al., "Faith and fate: Limits of transformers on compositionality." *NeurIPS*, 2023.
- [3] Hendel, Geva, Globerson, "In-context learning creates task vectors." *EMNLP Findings*, 2023.
- [4] Liu et al., "Lost in the middle: How language models use long contexts." *TACL*, 2024.
- [5] Min et al., "Rethinking the role of demonstrations: What makes in-context learning work?" *EMNLP*, 2022.
- [6] Nikolov et al., "A multi-agent AI system for deep learning model migration from TensorFlow to JAX." *arXiv:2603.27296*, 2026.
- [7] Roziere et al., "Code Llama: Open foundation models for code." *arXiv:2308.12950*, 2023.
- [8] Wang et al., "Self-consistency improves chain of thought reasoning in language models." *arXiv:2203.11171*, 2022.
- [9] Wei et al., "Chain-of-thought prompting elicits reasoning in large language models." *NeurIPS*, 2022.
