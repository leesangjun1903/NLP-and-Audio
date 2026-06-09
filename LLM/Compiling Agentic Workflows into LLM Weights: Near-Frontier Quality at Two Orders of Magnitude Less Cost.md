# Compiling Agentic Workflows into LLM Weights

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 명제는 다음과 같이 압축됩니다:

> **"절차적 지식(procedural knowledge)은 프롬프트가 아닌 가중치(weights)에 속한다."**

현재 AI 개발 생태계에서 지배적인 **Surface Orchestration** 패러다임(LangGraph, CrewAI 등)은 런타임마다 외부 오케스트레이터가 LLM에게 지시를 주입하는 방식인데, 저자들은 이를 **절차적 작업에 있어 비효율적**이라고 주장합니다. 대신, 절차 자체를 소규모 fine-tuned 모델의 가중치로 "컴파일"하는 **Subterranean Agent** 방식이 품질·비용·유연성 세 측면 모두에서 경쟁력 있음을 실증합니다.

### 주요 기여

| 기여 영역 | 내용 |
|-----------|------|
| **아키텍처 제안** | Surface Orchestration vs. Subterranean Agent의 명확한 이분법적 프레임 제시 |
| **실증적 장벽 해소** | 품질/비용/유연성이라는 3가지 인식적 장벽을 정량적으로 반박 |
| **비용 분석** | 자가 호스팅 + 토큰 볼륨 감소의 복합 효과를 최초로 정량화 ( $128\times$ – $462\times$ ) |
| **재컴파일 사이클** | 30–50분 CI/CD 수준의 재훈련 가능성 실증 |
| **통제 비교 설계** | 동일 모델(3B) 기반 컴파일 효과 순수 분리 실험 |

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

현존 오케스트레이션 프레임워크들의 구조적 문제:

1. **추론 단편화**: 오케스트레이터가 로컬 노드 컨텍스트만으로 생성하여 전역적 일관성 저하
2. **라우팅 실패**: 의사결정 허브에서의 LLM 분류기 오류 (Travel 실패율 24%)
3. **비용 폭증**: 매 대화마다 프론티어 API 호출 + 절차 설명 프롬프트 포함
4. **보안 위험**: 독점 절차를 서드파티 제공자에게 노출
5. **컨텍스트 낭비**: 절차 직렬화가 컨텍스트 윈도우를 지속적으로 소모

### 2.2 제안 방법

#### 절차의 유향 그래프 표현

논문은 절차를 다음 유향 그래프로 형식화합니다:

$$F = (N, E, n_0, T)$$

- $N$: 역할(에이전트/사용자)과 프롬프트 템플릿을 가진 노드 집합
- $E \subseteq N \times N \times C$: 조건부 전이 엣지 집합 ($C$는 조건 집합)
- $n_0 \in N$: 시작 노드
- $T \subseteq N$: 단말 노드 집합 (성공, 이탈, 에스컬레이션)

#### 컴파일 파이프라인

```
[절차 정의] → [합성 대화 생성] → [전체 파라미터 Fine-tuning] → [오케스트레이터 없이 배포]
```

**Step 1: 절차 정의**
유향 그래프 $F$로 워크플로우를 정의합니다.

**Step 2: 합성 대화 생성**
플로우차트의 모든 유효한 경로를 순회하며 합성 대화를 생성합니다. 각 노드에서 Claude Sonnet 4.5가 노드의 프롬프트 템플릿과 전체 대화 이력을 받아 맥락에 적합한 응답을 생성합니다.

- Travel: 2,125개 대화, 86개 고유 비순환 경로 (4–17 턴)
- Zoom: 6,264개 대화 (8개 시드 × 783개), 60개 경로
- Insurance: 3,000개 대화, 2,381개 경로 (9–39 턴)

**Step 3: 전체 파라미터 Fine-tuning**

훈련 목적함수는 표준 언어 모델 최대 우도 추정(MLE)입니다:

```math
\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{ < t}, \mathcal{C})
```

여기서:
- $\theta$: 모델 파라미터 (전체 업데이트)
- $x_t$: 시간 $t$에서의 토큰
- $\mathcal{C}$: 합성 대화 컨텍스트 (절차적 구조가 암묵적으로 인코딩됨)
- $T$: 대화 총 토큰 수

> **LoRA 사용 이유 없음**: 별도 연구(Dennis et al., 2026b)에서 rank 16–128의 LoRA는 절차적 작업의 전체 파라미터 미세조정에 근접하지 못함이 확인되었습니다.

**훈련 설정:**

| 설정 | Travel (3B) | Zoom/Insurance (8B) |
|------|-------------|---------------------|
| 기반 모델 | Qwen 2.5 3B Instruct | Qwen3-8B |
| 정밀도 | bf16 | bf16 |
| 옵티마이저 | AdamW 8-bit | AdamW |
| 학습률 | $2 \times 10^{-5}$ (cosine decay) | $2 \times 10^{-5}$ |
| 배치 크기 | 16 (gradient accumulation) | 32 |
| 에폭 | 20 (최적 체크포인트: ~4에폭) | 10–20 |
| 하드웨어 | 단일 RTX 5090 | 8×A100 (DeepSpeed ZeRO-3) |

**Step 4: 최소 시스템 프롬프트로 배포**

런타임 시 모델은 다음과 같은 최소한의 시스템 프롬프트만 수신합니다:

> *"You are a helpful travel booking assistant."*

절차 지시문, 플로우차트 상태, 라우팅 로직은 일체 주입되지 않습니다.

### 2.3 모델 구조

#### 아키텍처 비교

```
[Surface Orchestration - 런타임]
User ↔ Orchestrator ↔ LLM
         (매 턴마다 프롬프트 주입 + 출력 파싱)

[Subterranean Agent - 런타임]
User ↔ LLM (절차가 가중치에 내재화)
         (Orchestrator는 훈련 데이터 생성에만 사용)
```

#### 평가 방법론

**LLM-as-Judge** 방식(Zheng et al., 2023)을 사용하며, Claude Sonnet 4.5와 GPT-4.1을 독립적으로 활용합니다. 5가지 기준(1–5점 척도):

| 평가 기준 | 설명 |
|-----------|------|
| **Task Success** | 절차를 올바르게 실행하고 적절한 단말 상태 도달 여부 |
| **Information Accuracy** | 사용자 제공 정보의 정확한 활용 및 보존 |
| **Consistency** | 대화 전반에 걸친 일관된 상태 유지 |
| **Graceful Handling** | 변화, 모호성, 엣지 케이스 처리 능력 |
| **Naturalness** | 숙련된 인간 에이전트와의 유사성 |

**통계 검증:**
- 쌍별 비교: Wilcoxon signed-rank test (paired) 또는 Mann–Whitney $U$ (unpaired)
- 효과 크기: Cohen's $d$ (pooled SD)
- 신뢰구간: Bootstrap 95% CI (10,000 resamples, percentile method)
- 다중비교 보정: Holm–Bonferroni correction ($\alpha = 0.05$)

### 2.4 성능 향상

#### 품질 성능 (Claude Sonnet 4.5 판정)

**Travel Booking (3B 모델)**

| 기준 | 3B Sub. | 3B Orch. | LG Orch. | In-Context |
|------|---------|----------|----------|------------|
| Task Success | 4.11 | 3.93 | 4.17 | **4.53** |
| Info. Accuracy | **4.75** | 4.69 | 4.21 | 4.64 |
| Consistency | 4.34 | 4.12 | 4.32 | **4.96** |
| Graceful Handling | 4.07 | 3.87 | 4.62 | **4.96** |
| Naturalness | 4.12 | 3.96 | 4.84 | **5.00** |

컴파일 효과 순수 분리 결과:
$$\Delta_{\text{Task Success}} = +0.18,\ p < 0.001,\ d = 0.22$$
$$\Delta_{\text{Consistency}} = +0.22,\ p < 0.001,\ d = 0.23$$
$$\Delta_{\text{Graceful}} = +0.20,\ p < 0.001,\ d = 0.23$$
$$\Delta_{\text{Naturalness}} = +0.17,\ p < 0.001,\ d = 0.21$$

**Insurance Claims (8B 모델, 55 노드)**

| 기준 | 8B Sub. | LG Orch. | In-Context | Sub./IC 비율 |
|------|---------|----------|------------|-------------|
| Task Success | 4.47 | 4.42 | **4.78** | 93.5% |
| Info. Accuracy | 4.40 | 4.45 | **4.78** | 92.1% |
| Consistency | 4.51 | 4.39 | **4.82** | 93.6% |
| Graceful Handling | **4.81** | 4.38 | 4.96 | 97.0% |
| Naturalness | **4.92** | 4.58 | 5.00 | 98.4% |

#### 비용 분석

비용 절감은 두 독립적 요소의 곱으로 계산됩니다:

$$\text{Cost Ratio} = \underbrace{\frac{C_{\text{API}}}{C_{\text{self-hosted}}}}_{\approx 65\times} \times \underbrace{\frac{V_{\text{in-context}}}{V_{\text{compiled}}}}_{\approx 2\text{–}7\times}$$

구체적으로:
- **Claude Sonnet 4.5**: \$3/M input tokens, \$15/M output tokens
- **자가 호스팅 (A100 80GB, vLLM)**: ~\$0.05/M input, ~\$0.23/M output

$$\text{Per-token reduction} \approx 65\times$$

토큰 볼륨 차이 (절차 복잡도에 비례 성장):

$$\text{Token Volume Ratio} \approx \begin{cases} 2\times & \text{(Travel, 14 nodes)} \\ 7\times & \text{(Insurance, 55 nodes)} \end{cases}$$

따라서 총 비용 절감:

$$\text{Total Reduction} = \begin{cases} 128\times & \text{(Travel)} \\ 296\times & \text{(Zoom)} \\ 462\times & \text{(Insurance)} \end{cases}$$

| 도메인 | In-Context | LG Orch. | Subterranean | IC/Sub 비율 |
|--------|-----------|----------|--------------|------------|
| Travel (14 nodes) | \$0.133 | \$0.077 | \$0.0010 | $128\times$ |
| Zoom (14 nodes) | \$0.103 | \$0.054 | \$0.0003 | $296\times$ |
| Insurance (55 nodes) | \$0.327 | \$0.174 | \$0.0007 | $462\times$ |

**일회성 컴파일 비용** = ~\$50–80 (데이터 생성 ~\$40 + fine-tuning ~\$10–40)

손익분기점:
$$N_{\text{break-even}} < 500 \text{ conversations (모든 도메인)}$$

#### 지연 시간 (Latency)

| 도메인 | Sub. (초) | LG Orch. (초) | 속도 향상 |
|--------|-----------|--------------|----------|
| Zoom | 29.5 | 52.1 | $1.77\times$ |
| Insurance | **43.2** | **120.8** | $\mathbf{2.8\times}$ |

#### 실패율

$$\text{Failure Rate} = P(\text{Task Success} \leq 3)$$

| 도메인 | Compiled | LG Orch. |
|--------|----------|----------|
| Travel | **5.5%** | 24.0% |
| Insurance | **9.0%** | 17.0% |
| Zoom | 11.0% | 9.0% |

### 2.5 한계

논문이 명시적·암묵적으로 인정하는 한계:

1. **In-Context 품질 갭**: 8B 모델은 여전히 In-Context 기준에서 2–13% 낮음 (특히 정보 정확도)
2. **도메인 제한**: 3개 도메인(여행, Zoom, 보험)만 평가; 오픈 엔디드 창의적 작업 미검증
3. **판정자 편향 가능성**: Claude가 데이터 생성·판정 모두 담당 (GPT-4.1로 교차 검증하여 부분 완화)
4. **단일 모델 패밀리**: Qwen 계열만 사용; 다른 아키텍처 일반화 미검증
5. **정적 워크플로우 가정**: 동적으로 변화하는 절차에 대한 연속 학습(continual learning) 미탐구
6. **하드웨어 의존성**: 8B 풀 파인튜닝에 8×A100 필요 (단일 GPU 환경에서는 3–4시간 소요)
7. **LoRA 대안 부재**: 파라미터 효율적 방법 연구(Dennis et al., 2026b)에서 실패했으나, 최신 LoRA 변형(DoRA 등)은 미평가
8. **합성 데이터 품질 의존성**: Claude Sonnet 4.5 생성 데이터 품질이 최종 모델 성능의 상한선

---

## 3. 모델의 일반화 성능 향상 가능성

이 섹션은 논문의 명시적·암묵적 내용과 관련 연구를 기반으로 분석합니다.

### 3.1 논문 내 일반화 관련 근거

#### (1) 절차 복잡도 스케일링 일반화

논문은 14노드(Travel/Zoom)에서 **55노드(Insurance, 2,381개 경로)**로 확장 시에도 성능이 유지됨을 보입니다:

$$\text{Quality}_{55\text{-node}} \approx 92\text{–}98\%\ \text{of In-Context}$$

이는 컴파일 접근법이 절차 복잡도에 대해 **비선형적으로 유리**함을 시사합니다. 프롬프트 크기가 상수인 컴파일 모델과 달리, 오케스트레이션 비용은 노드 수에 선형 증가합니다.

#### (2) 도메인 지식 내재화를 통한 일반화

Zoom 도메인 실험은 절차적 구조뿐 아니라 **도메인 특화 지식**(UI, 에러 코드, 설정 메뉴)도 가중치에 내재화됨을 보여줍니다. 이는 단순 절차 추종을 넘어 **지식 일반화**의 증거입니다.

#### (3) 시드 다양성 통한 경로 일반화

Zoom 훈련 데이터 생성 시 8개의 다른 랜덤 시드(42–49)를 사용하여 동일 경로라도 다양한 대화를 생성합니다:

$$\mathcal{D}_{\text{train}} = \bigcup_{s \in \{42,...,49\}} \mathcal{D}_s, \quad |\mathcal{D}_{\text{train}}| = 6,264$$

이는 **분포 커버리지 확장**을 통해 미관측 경로에 대한 일반화를 도모합니다.

#### (4) 사용자 스타일 다양성 내재화

시나리오 변수에 사용자 성격 유형(구체적 → 모호함, 열정적 → 회의적)을 포함시켜, 모델이 다양한 사용자 행동 패턴에 일반화하도록 설계됩니다.

#### (5) 암묵적 상태 추적을 통한 일반화

오케스트레이터 방식이 명시적 상태를 주입하는 반면, 컴파일 모델은 **암묵적 상태 추적**을 학습합니다. 이는 훈련 데이터에 없는 새로운 대화 경로에도 적용 가능한 더 강건한 내부 표현을 형성할 가능성이 있습니다.

### 3.2 일반화 성능 향상 가능성 분석

#### 긍정적 요인

| 요인 | 메커니즘 | 일반화 기대 효과 |
|------|----------|----------------|
| **전체 파라미터 업데이트** | 모델의 심층 상태 추적 능력 수정 | 표면적 스타일이 아닌 절차적 구조 자체를 내재화 |
| **다양한 시나리오 커버리지** | 경로 × 시나리오 변수 조합 | 분포 외 사례에 대한 강건성 |
| **자연스러운 대화 훈련** | 인위적 태그/어노테이션 없는 자연어 대화 | 실제 사용자 발화 패턴에 대한 일반화 |
| **전역적 대화 인식** | 전체 절차를 한 모델로 처리 | 국부적 오케스트레이터보다 더 일관된 장기 상태 추적 |

#### 잠재적 일반화 향상 전략 (논문에서 암시된 것)

**전략 1: 데이터 증강을 통한 경로 일반화**

$$|\mathcal{D}_{\text{augmented}}| = K \times |\mathcal{D}_{\text{base}}|, \quad K \in \{8, 16, ...\}$$

Zoom에서 8배 증강이 graceful handling을 $82\% \rightarrow 92\%$로 향상시킨 것처럼, 추가 증강이 일반화에 기여할 수 있습니다.

**전략 2: 다중 도메인 훈련 (논문에서 미탐구)**

$$\mathcal{D}_{\text{multi}} = \mathcal{D}_{\text{travel}} \cup \mathcal{D}_{\text{zoom}} \cup \mathcal{D}_{\text{insurance}}$$

다중 도메인 동시 훈련이 도메인 간 전이 학습(transfer learning) 효과를 낼 가능성이 있습니다.

**전략 3: 반사실적 경로 훈련 (논문에서 미탐구)**

훈련 데이터에 포함되지 않은 엣지 케이스 경로를 의도적으로 생성하여 분포 외 일반화를 강화할 수 있습니다.

### 3.3 일반화 한계

- **평가 데이터가 훈련 플로우차트 범위 내**: 완전히 새로운 도메인에 대한 제로샷 일반화는 미검증
- **사용자 시뮬레이터 한계**: Claude 기반 시뮬레이터가 실제 사용자의 다양성을 완전히 포착하지 못할 수 있음
- **정보 정확도에서의 병목**: "world knowledge 부족"이 명시적으로 언급됨 — 절차 일반화와 지식 일반화는 별개의 문제

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 패러다임 전환 가능성

이 논문은 **"오케스트레이션은 기본값이 아니다"**라는 명제를 실증적으로 제시합니다. 이는 향후 AI 에이전트 시스템 설계에서 다음과 같은 패러다임 변화를 촉진할 수 있습니다:

```
기존: 절차 → 오케스트레이터 → LLM (분리된 시스템)
제안: 절차 → [컴파일] → LLM 가중치 (통합된 시스템)
```

#### (2) "Procedural Knowledge Compiler"라는 새로운 연구 분야 개척

플로우차트 → 훈련 데이터 → 특화 모델의 파이프라인이 하나의 독립적 연구 분야로 발전할 가능성이 있습니다. 유사한 개념으로는:
- Program Synthesis (프로그램을 자동으로 생성)
- Knowledge Distillation (지식을 소형 모델로 증류)
- Neural Program Induction (절차를 신경망으로 내재화)

#### (3) LLM 배포 경제학에 대한 재고

$128\times$ – $462\times$의 비용 절감 실증은 **기업의 LLM 배포 의사결정 프레임워크**를 바꿀 수 있습니다:

$$\text{Total Cost} = C_{\text{compile}} + N_{\text{conv}} \cdot C_{\text{compiled inference}}$$

$$\ll N_{\text{conv}} \cdot C_{\text{API inference}} \quad \text{for } N_{\text{conv}} > 500$$

#### (4) 관련 연구 방향 자극

| 영향 받는 연구 분야 | 구체적 영향 |
|-----------------|------------|
| **Task-Oriented Dialogue** | SimpleTOD 계열의 산업적 확장 검증 |
| **Knowledge Distillation** | 절차적 지식 특화 증류 방법 연구 |
| **Continual Learning** | 절차 업데이트 시 재훈련 최소화 연구 |
| **PEFT 연구** | 절차적 작업에 적합한 새로운 파라미터 효율 방법 탐색 |
| **Synthetic Data Generation** | 플로우차트 기반 데이터 합성의 품질·다양성 향상 |

### 4.2 최신 관련 연구 비교 분석 (2020년 이후)

| 연구 | 연도 | 핵심 방법 | 논문과의 관계 |
|------|------|----------|-------------|
| **SimpleTOD** (Hosseini-Asl et al.) | 2020 | 태스크 지향 대화를 단일 시퀀스 예측으로 통합 | 개념적 선구자; 산업 규모 비용 분석 및 재컴파일 사이클 미검토 |
| **FireAct** (Chen et al.) | 2023 | GPT-4 ReAct 궤적으로 Llama2-7B 파인튜닝 | 프론티어 모델 증류 접근; 절차 컴파일 대신 추론 증류 |
| **AgentTuning** (Zeng et al.) | 2024 | 다양한 에이전트 궤적으로 Llama 2 instruction-tuning | 일반 에이전트 능력 향상; 절차 특화 아님 |
| **Agent Lumos** (Yin et al.) | 2024 | 계획(planning) + 기반화(grounding) 모듈 분리 훈련 | 모듈형 접근; 본 논문은 단일 통합 모델 추구 |
| **SynTOD** (Samarinas et al.) | 2024 | 상태 전이 그래프 기반 합성 데이터 생성 | 방법론적 유사성이 가장 높음; 비용·유연성 분석 부재 |
| **WorkflowLLM** (Fan et al.) | 2024 | 106K 워크플로우 샘플로 API 오케스트레이션 지식 컴파일 | 대규모 API 지식 컴파일; 절차적 대화 특화 아님 |
| **AutoTOD** (Xu et al.) | 2024 | 자율적 행동 시퀀싱 + 모듈형 시스템 실패 모드 분석 | 본 논문의 문제 진단과 유사; 해결책은 상이 |
| **Dennis et al. [2026a]** | 2026 | In-context prompting이 오케스트레이션 지배 | 본 논문의 직접적 전작; 품질 상한선 정의 |

**본 논문의 차별성:**
- 동일 모델 컨트롤 실험으로 **컴파일 효과 자체를 순수 분리**
- **비용 분석**을 정량적으로 최초 수행 ( $128\times$ – $462\times$ )
- **재컴파일 사이클**을 실측 (30–50분)
- **55노드 규모**의 복잡 절차 검증

### 4.3 향후 연구 시 고려할 점

#### 고려점 1: 평가 생태계의 다양화

현재 논문의 평가는 **합성 사용자 시뮬레이터** 기반입니다. 향후 연구에서는:

$$\text{Evaluation Diversity} = \alpha \cdot \text{Simulated} + \beta \cdot \text{Human} + \gamma \cdot \text{Adversarial}$$

- 실제 사용자 A/B 테스트 포함
- 적대적 사용자 행동(off-script 발화, 악의적 입력)에 대한 강건성 검증
- 도메인 전문가 평가 포함

#### 고려점 2: 절차 업데이트의 연속성 문제

현재 접근법은 절차 변경 시 **처음부터 재훈련**을 요구합니다. 이를 해결하기 위한 연구 방향:

$$\theta_{\text{new}} = \text{FineTune}(\theta_{\text{old}}, \mathcal{D}_{\Delta F})$$

여기서 $\mathcal{D}_{\Delta F}$는 변경된 절차 부분에 대한 데이터입니다. **Continual Learning** 기법(EWC, Progressive Neural Networks 등)의 적용 가능성을 탐구해야 합니다.

#### 고려점 3: 파라미터 효율적 방법 재탐구

논문은 LoRA(rank 16–128)의 실패를 보고하나, 최신 방법들에 대한 재검토가 필요합니다:

- **DoRA** (Weight-Decomposed Low-Rank Adaptation, Liu et al., 2024)
- **GaLore** (Gradient Low-Rank Projection, Zhao et al., 2024)
- **LoRA+** (Hayou et al., 2024)

절차적 지식 내재화에 특화된 새로운 PEFT 방법 설계도 유망한 방향입니다.

#### 고려점 4: 다중 절차 동시 내재화

단일 모델이 **여러 절차를 동시에** 처리할 수 있는지 탐구:

$$\theta^* = \arg\min_\theta \sum_{k=1}^{K} \mathcal{L}(\theta; F_k)$$

절차 간 간섭(interference) 및 파국적 망각(catastrophic forgetting) 문제가 핵심 과제가 될 것입니다.

#### 고려점 5: 훈련 데이터 품질의 상한선 문제

논문의 방법론에서 Claude Sonnet 4.5가 훈련 데이터를 생성하고 판정까지 담당합니다. 이는 다음 문제를 내포합니다:

$$\text{Quality}_{\text{compiled}} \leq \text{Quality}_{\text{data generator}} \leq \text{Quality}_{\text{frontier}}$$

향후 연구에서는:
- 다양한 프론티어 모델 앙상블로 데이터 생성
- 실제 전문가 대화 데이터 혼합
- 데이터 품질 필터링 파이프라인 개발

#### 고려점 6: 보안 및 적대적 견고성

가중치에 절차가 내재화될 경우, **모델 역공학(model inversion)** 공격을 통해 독점 절차가 추출될 위험이 있습니다:

$$\hat{F} = \text{Inversion}(\theta_{\text{compiled}})$$

이에 대한 방어 메커니즘 연구(차분 프라이버시 적용 훈련, 모델 워터마킹 등)가 필요합니다.

#### 고려점 7: 멀티모달 확장

현재 텍스트 기반 대화에만 집중되어 있으나, 실제 비즈니스 프로세스는 이미지, 문서, 음성 등 멀티모달 입력을 요구합니다. 비전-언어 모델(VLM)에의 적용 가능성 탐구가 필요합니다.

---

## 📚 참고 자료

**주요 참고 논문 (본 논문의 References 기반):**

1. **Dennis, S., Patil, R., Shabahang, K., & Guo, H.** (2026). *Compiling Agentic Workflows into LLM Weights: Near-Frontier Quality at Two Orders of Magnitude Less Cost.* arXiv:2605.22502v1

2. **Hosseini-Asl, E., McCann, B., Wu, C.-S., Yavuz, S., & Socher, R.** (2020). *A Simple Language Model for Task-Oriented Dialogue.* NeurIPS 33.

3. **Chen, B. et al.** (2023). *FireAct: Toward Language Agent Fine-Tuning.* arXiv:2310.05915

4. **Samarinas, C. et al.** (2024). *Simulating Task-Oriented Dialogues with State Transition Graphs and Large Language Models.* arXiv:2404.14772

5. **Fan, S. et al.** (2024). *WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models.* arXiv:2411.05451

6. **Yin, D. et al.** (2024). *Agent Lumos: Unified and Modular Training for Open-Source Language Agents.* ACL 2024.

7. **Zeng, A. et al.** (2024). *AgentTuning: Enabling Generalized Agent Abilities for LLMs.* Findings of ACL 2024.

8. **Xu, H.-D. et al.** (2024). *Rethinking Task-Oriented Dialogue Systems: From Complex Modularity to Zero-Shot Autonomous Agent.* ACL 2024.

9. **Zheng, L. et al.** (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 36.

10. **Kwon, W. et al.** (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.

11. **Cemri, M. et al.** (2026). *Why Do Multi-Agent LLM Systems Fail?* arXiv:2503.13657

12. **Dennis, S. et al.** (2026b). *Procedural Knowledge is Not Low-Rank: Why LoRA Fails to Internalize Multi-Step Procedures.* arXiv preprint.

13. **Panickssery, A., Bowman, S.R., & Feng, S.** (2024). *LLM Evaluators Recognize and Favor Their Own Generations.* arXiv:2404.13076

14. **Patel, K. et al.** (2024). *LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators.* arXiv:2411.00136

> **⚠️ 주의사항**: 이 논문은 arXiv:2605.22502v1 (2026년 5월 21일)에 공개된 프리프린트입니다. Dennis et al. [2026a], [2026b] 등 일부 참고문헌은 본 논문 저자들의 별도 프리프린트로, 공개 접근이 제한될 수 있습니다. 분석은 제공된 PDF 전문을 기반으로 하였으며, 해당 프리프린트 외 인용된 연구들의 내용은 논문 내 서술에 의거하였습니다.
