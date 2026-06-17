# AutoForge: Automated Environment Synthesis for Agentic Reinforcement Learning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

AutoForge는 **언어 기반 에이전트(Language-based Agents)를 위한 강화학습(RL) 훈련 환경을 자동으로 합성**하는 통합 프레임워크입니다. 기존 연구들이 반자동(semi-automated) 환경 합성이나 난이도가 낮은 태스크에 한정되었던 한계를 극복하고, 도구 설명 문서(tool description document)만으로 고난도·검증 가능한 태스크와 모의 환경을 자동 생성합니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| ① 자동화 합성 파이프라인 | 도구 설명 문서 입력만으로 모의 환경 및 고난도 태스크 자동 생성 |
| ② ERPO 알고리즘 | 시뮬레이션 사용자 불안정성 완화 + 환경 수준 어드밴티지 추정 |
| ③ 검증된 일반화 성능 | 인도메인/아웃오브도메인 벤치마크에서 강력한 성능 입증 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

논문은 기존 에이전틱 RL 연구의 세 가지 핵심 문제를 지적합니다:

**문제 ①: 환경 합성의 자동화 부족 및 낮은 난이도**
- 기존 연구(Fang et al., 2025; Qian et al., 2025b)는 반자동 합성이거나 충분히 복잡하지 않은 태스크에 한정
- 에이전트 훈련의 폭(breadth)과 깊이(depth) 모두 부족

**문제 ②: 시뮬레이션 사용자의 불안정성**
- LLM 기반 시뮬레이션 사용자는 환각(hallucination) 정보 제공 또는 정보 누락 가능
- 이로 인해 에이전트 훈련 시 잘못된 페널티 부과 → 어드밴티지 추정 편향

**문제 ③: 단일 환경 관점의 다중 환경 RL 훈련**
- 기존 접근(Qian et al., 2025b)은 다중 환경 RL을 단일 환경 관점에서 처리 → 비효율적이고 불안정

---

### 2.2 제안하는 방법

#### 2.2.1 환경 합성 파이프라인 (3단계)

**[단계 1] 환경 합성 (Environment Synthesis)**

합성 환경 $\mathcal{E}$를 상태(State)와 함수 집합(Function Set)의 튜플로 정의:

$$\mathcal{E} = (S, \mathcal{F})$$

상태 $S$는 키-값 쌍의 리스트로 표현:

$$S = [(K_1, V_1), (K_2, V_2), \ldots, (K_n, V_n)]$$

여기서 $K_i$는 속성 이름(예: `project_id`), $V_i$는 해당 속성의 값(예: `P0001`)입니다.

- **상태 구조 생성**: LLM이 도구 설명 문서로부터 속성명 $K_i$를 자동 생성
- **함수 집합 생성**: 상태 구조와 도구 설명을 기반으로 LLM이 Python 코드 자동 생성 (저비용, 고동시성, 높은 안정성 보장)

---

**[단계 2] 도구 시퀀스 생성 (Tool-Sequence Generation)**

태스크 난이도는 필요한 도구 시퀀스의 복잡도에 의해 결정된다고 가정합니다.

- **시퀀스 샘플링**: 도구들을 방향 그래프의 노드로 표현 → 랜덤 워크로 수천 개의 도구 시퀀스 생성
- **시퀀스 병합**: $i$번째 시퀀스 $\mathcal{T}\_i = [t_{i_1}, t_{i_2}, \ldots, t_{i_{|\mathcal{T}_i|}}]$와 $\mathcal{T}_j$를 병합하여 $\mathcal{T}'_k$ 생성 (LLM이 중복 도구 제거)
- **추론 노드 삽입**: 이전 도구 출력으로부터 고차원 정보를 추론하는 노드 $r_{k_j}$ 삽입:

$$\mathcal{T}''_k = [t_{k_1}, \ldots, r_{k_1}, \ldots, t_{k_i}, r_{k_j}, \ldots, t_{k_{|\mathcal{T}''_k|}}]$$

- **추론 엣지 통합**: 도구 간 의존 관계를 방향 엣지로 명시 → 최종적으로 **유향 비순환 그래프(DAG)** 구성:

$$G_k = (\mathcal{T}''_k, E_k)$$

---

**[단계 3] 태스크 생성 (Task Generation)**

$G_k$를 청사진으로 활용하여 RL 훈련 샘플 생성:
1. **환경 초기화**: 각 속성명 $K_i$에 대한 값 $V_i$ 생성, 초기 태스크 의도 $\tilde{Q}_k$ 생성
2. **도구 시퀀스 실행**: 위상적 순서에 따라 도구 실행 → 최종 환경 상태 $S^*_k$ 획득 (도구 시퀀스가 아닌 **최종 환경 상태** 기반 검증)
3. **태스크 정제**: 최소 필요 정보만 포함하도록 태스크 정제 → RL 훈련 샘플:

$$D_k = (Q_k, S_k, S^*_k, \mathcal{F})$$

---

#### 2.2.2 ERPO 알고리즘 (Environment-level Relative Policy Optimization)

GRPO(Shao et al., 2024)를 네 가지 측면으로 확장합니다.

**① 사용자 중심 롤아웃 (User-Centered Rollout)**

롤아웃 궤적:

$$\tau = (o_0, a_1, o_1, a_2, o_2, \ldots, a_n, o_n)$$

보상 함수 (이진 결과 기반):

$$R = \begin{cases} 1, & \text{if } S^* == \hat{S} \\ 0, & \text{if } S^* \neq \hat{S} \end{cases} \tag{1}$$

**② 인터리브드 씽킹 (Interleaved Thinking)**

Qwen3 기반 백본 모델의 기존 추론 콘텐츠 삭제 문제 해결 → 다중 턴 상호작용에서 이전 추론 내용(태스크 분석, 계획 등)을 보존하도록 훈련/추론 절차 수정

**③ MEU: 오류 사용자 행동 마스킹 (Masking out Erroneous User behaviors)**

LLM-as-judge를 사용하여 시뮬레이션 사용자 오류로 인한 궤적을 식별하고 마스킹:

$$J_{\text{ERPO}}(\theta) = \mathbb{E}_{D \sim \mathcal{D}, \{\tau_i\}_{i=1}^{M} \sim \pi_{\text{old}}(\cdot|D)} \left[ \frac{1}{\sum_{i=1}^{M} \mathbf{1}_{\text{MEU}}(\tau_i)} \sum_{i=1}^{M} \mathbf{1}_{\text{MEU}}(\tau_i) \times \min\left( \frac{\pi_\theta(\tau_i|D)}{\pi_{\text{old}}(\tau_i|D)} A_i, \, \text{clip}\left(\frac{\pi_\theta(\tau_i|D)}{\pi_{\text{old}}(\tau_i|D)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{old}}) \right] \tag{2}$$

여기서:
- $\epsilon, \beta$: 하이퍼파라미터
- $A_i$: 어드밴티지 함수
- $\mathbf{1}_{\text{MEU}}(\tau_i)$: 사용자 오류 없으면 1, 오류 있으면 0인 지시 함수

**④ 환경 수준 어드밴티지 추정 (Environment-Level Advantage Estimation)**

기존 GRPO의 그룹 수준 어드밴티지:

$$A_{ij}^{\text{group}} = \frac{R_{ij} - \text{mean}(\{R_{ij}\}_{j=1}^{M_i})}{\text{std}(\{R_{ij}\}_{j=1}^{M_i})} \tag{3}$$

→ 소규모 그룹 또는 헤비테일 분포에서 표준편차 추정 불안정

AutoForge의 **환경 수준 어드밴티지**:

$$A_{ij}^{\text{env}} = \frac{R_{ij} - \text{mean}(\{R_{ij}\}_{j=1}^{M_i})}{\text{std}(\{R_{ij}\}_{i=1,\ldots,P}^{j=1,\ldots,M_i})} \tag{4}$$

분자: 동일 질문 내 평균 차감 (질문 수준 상대 비교 유지)
분모: **환경 전체** 표준편차 사용 → 아웃라이어 영향 완화, 더 안정적인 추정

---

### 2.3 모델 구조

```
AutoForge 전체 구조

┌─────────────────────────────────────────────────────┐
│         합성 파이프라인 (Synthesis Pipeline)          │
│  도구 문서 → 도구 그래프 → 시퀀스 샘플링              │
│           → DAG 생성 → 태스크/환경 생성              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              ERPO 알고리즘                           │
│  롤아웃 → MEU 필터링 → 환경 수준 어드밴티지 계산      │
│       → 정책 업데이트 (클리핑 + KL 페널티)           │
└─────────────────────────────────────────────────────┘
```

- **백본 모델**: Qwen3-Thinking-30B-A3B (30B 파라미터, 3B 활성 파라미터)
- **환경 합성 모델**: Qwen3-Thinking-235B-A22B
- **사용자 시뮬레이터**: GPT-4.1 (훈련 및 평가)
- **훈련 설정**: 64 GPU, 배치 크기 32, 샘플당 8개 궤적, DAPO의 동적 샘플링 적용

---

### 2.4 성능 향상

**인도메인 성능** ($\tau$-bench, $\tau^2$-Bench, VitaBench):

| 모델 | Retail | Airline | Telecom | Delivery | In-store |
|------|--------|---------|---------|----------|----------|
| Qwen3-Thinking-30B-A3B (백본) | 67.8 | 48.0 | 26.3 | 35.0 | 40.0 |
| AgentScaler-30B-A3B | 70.4 | 54.0 | 55.3 | 25.0 | 33.0 |
| **AutoForge-30B-A3B** | **73.1** | **56.5** | **76.3** | **46.0** | **54.5** |
| GPT-o3 (closed-source) | 70.4 | 52.0 | 58.2 | 53.5 | 53.5 |

- 200B 이하 오픈소스 모델 중 최고 성능 달성
- 일부 벤치마크에서 GPT-o3 등 폐쇄형 대형 모델에 근접 또는 초과

**아웃오브도메인 성능** (ACEBench-zh): Figure 2 기준
- Qwen3-30B-A3B (백본) < AutoForge-SFT < AutoForge-RL 순으로 성능 향상
- RL 버전이 SFT 버전보다 더 큰 향상 → RL 훈련의 일반화 효과 입증

---

### 2.5 한계

1. **입력 제약**: 현재 파이프라인은 도구 설명 문서를 입력으로 요구 → 태스크 주제나 일반 텍스트로부터의 자동 환경 구축 불가
2. **제한된 환경 수**: 10개 가상 환경, 1,078개 태스크만 사용 → 환경 수 확장 시 RL 및 일반화에 미치는 영향 미검증
3. **결과 기반 보상만 사용**: ERPO는 이진 결과 보상에만 의존 → 단계별(turn-level) 가치 감독(value supervision) 미탐구
4. **사용자 시뮬레이터 의존성**: GPT-4.1 사용자 시뮬레이터의 성능이 평가 결과에 직접 영향 (Table 2: Telecom에서 OM 사용 시 76.3 → 90.4로 급증)
5. **언어 및 형식 제약**: 훈련 데이터는 영어, Hermes 형식 → 중국어/커스텀 형식 벤치마크에서 일부 성능 차이 존재 가능

---

## 3. 일반화 성능 향상 가능성 중점 분석

### 3.1 아웃오브도메인 일반화 실험 설계

AutoForge는 ACEBench-zh를 아웃오브도메인 벤치마크로 선택했으며, 이는 네 가지 측면에서 훈련 데이터와 완전히 다릅니다:

| 차이점 | 훈련 데이터 | ACEBench-zh |
|--------|-----------|-------------|
| 다중 턴 형식 | 표준 multi-turn format | 원시 프롬프트 문자열 형식 |
| 도구 호출 형식 | Hermes 형식 | 커스텀 형식 |
| 도구 자체 | 10개 가상 환경 도구 | 완전히 새로운 도구 (검증됨) |
| 언어 | 영어 | 중국어 |

### 3.2 일반화 성능의 원천 분석

**원천 ①: 다양하고 복잡한 합성 환경**

DAG 기반 태스크 생성은 단순 도구 호출을 넘어 멀티 요구사항, 고차 추론을 포함하는 구조적으로 다양한 태스크를 생성합니다. 이러한 구조적 다양성이 특정 도구나 형식에 과적합되지 않도록 합니다.

**원천 ②: MEU를 통한 편향 없는 학습**

$\mathbf{1}_{\text{MEU}}(\tau_i) = 0$으로 사용자 오류 궤적 마스킹 → 에이전트가 **진짜 도구 사용 능력**을 학습 (사용자의 특정 행동 패턴 암기 방지)

**원천 ③: 환경 수준 어드밴티지 추정**

$$A_{ij}^{\text{env}} = \frac{R_{ij} - \text{mean}(\{R_{ij}\}_{j=1}^{M_i})}{\text{std}(\{R_{ij}\}_{i=1,\ldots,P}^{j=1,\ldots,M_i})}$$

환경 내 상대적 성능을 비교함으로써 특정 환경의 난이도 차이를 정규화 → 더 보편적인 능력 학습 유도

**원천 ④: 인터리브드 씽킹**

이전 추론 내용 보존 → 새로운 형식이나 도구를 만나도 이전 분석/계획을 활용하는 메타 인지 능력 강화

### 3.3 일반화 성능의 정량적 근거

Figure 2 (ACEBench-zh 결과):
- SFT 버전도 백본 대비 향상 → 합성 환경 데이터 자체의 일반화 가치 입증
- RL 버전이 SFT 대비 추가 향상 → RL 훈련이 단순 패턴 학습을 넘어 에이전트 능력의 본질적 향상 유도
- 특히 Agent subset에서 전체 향상폭이 두드러짐

### 3.4 확장 가능성

논문이 명시적으로 언급한 미래 방향:
> "Investigating how scaling up the number of synthetic environments affects RL training and out-of-domain generalization would be beneficial."

현재 10개 환경, 1,078개 태스크만으로도 강력한 일반화를 보였으므로, 환경 수 확장 시 성능 향상 여지가 상당히 존재합니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 환경 에뮬레이션 연구 비교

| 연구 | 방식 | 자동화 수준 | 안정성 | 확장성 |
|------|------|-----------|--------|--------|
| $\tau$-bench (Yao et al., 2024) | 로컬 실행 환경 | 수동 주석 필요 | 높음 | 낮음 |
| ToolBench (Qin et al., 2023) | 모델 기반 시뮬레이터 | 높음 | 낮음(환각) | 중간 |
| ZeroSearch (Sun et al., 2025) | 모델 기반 시뮬레이터 | 높음 | 낮음(환각) | 높음 |
| AgentScaler (Fang et al., 2025) | 반자동 | 중간 | 중간 | 중간 |
| UserRL (Qian et al., 2025b) | 단일 환경 관점 | 중간 | 중간 | 낮음 |
| **AutoForge (본 논문)** | **자동 + 실행 가능 코드** | **높음** | **높음** | **높음** |

### 4.2 에이전틱 RL 알고리즘 비교

| 연구 | 어드밴티지 추정 | 사용자 불안정성 처리 | 다중 환경 지원 |
|------|---------------|------------------|--------------|
| GRPO (Shao et al., 2024) | 그룹 수준 | ✗ | ✗ |
| ToolRL (Qian et al., 2025a) | 그룹 수준 | ✗ | 부분적 |
| MUA-RL (Zhao et al., 2025) | 그룹 수준 | 부분적 | ✗ |
| DAPO (Yu et al., 2025) | 동적 샘플링 | ✗ | ✗ |
| **ERPO (AutoForge)** | **환경 수준** | **✓ (MEU)** | **✓** |

### 4.3 태스크 합성 연구 비교

- **TaskCraft (Shi et al., 2025)**: 에이전틱 태스크 자동 생성을 시도하나 복잡한 의존성 그래프 생성이 제한적
- **AgentGym-RL (Xi et al., 2025)**: 멀티 턴 RL 훈련이나 환경 합성 자동화 미흡
- **Toucan (Xu et al., 2025)**: MCP 환경에서 데이터 합성이나 실세계 API 의존성 존재
- **AutoForge**: DAG 기반 복잡 태스크 자동 생성 + 완전 실행 가능한 합성 환경의 결합으로 차별화

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5.1 향후 연구에 미치는 영향

**영향 ①: 에이전틱 RL 훈련 패러다임 전환**

AutoForge는 실세계 환경 의존성에서 벗어나 **완전 자동화된 합성 환경 기반 훈련**이 가능함을 입증했습니다. 이는 특정 도메인 데이터가 부족한 상황에서도 에이전트 훈련이 가능함을 시사하며, 향후 에이전틱 RL 연구의 표준 패러다임이 될 가능성이 있습니다.

**영향 ②: 환경 다양성과 일반화의 연관성 입증**

소수(10개)의 합성 환경으로도 강력한 아웃오브도메인 일반화를 달성한 결과는, **환경의 수보다 구조적 다양성**이 더 중요할 수 있음을 시사합니다. 이는 "환경 스케일링 법칙(Environment Scaling Law)" 연구로 이어질 수 있습니다.

**영향 ③: MEU 메커니즘의 광범위한 적용 가능성**

LLM-as-judge를 통한 사용자 오류 마스킹은 LLM 기반 환경 피드백의 신뢰성 문제를 다루는 보편적 해결책으로, 다른 에이전틱 RL 프레임워크에도 직접 적용 가능합니다.

**영향 ④: 환경 수준 어드밴티지 추정의 일반화**

ERPO의 환경 수준 어드밴티지 추정($A_{ij}^{\text{env}}$)은 다중 태스크/다중 도메인 RL 설정에서 표준편차 추정 불안정성을 해결하는 일반적 기법으로 발전할 수 있습니다.

**영향 ⑤: 인터리브드 씽킹의 다중 턴 에이전트 적용**

다중 턴 상호작용에서 이전 추론 보존의 중요성을 실증적으로 입증 → 멀티 턴 대화 에이전트 설계 시 필수 고려 요소로 자리잡을 것으로 예상

### 5.2 향후 연구 시 고려할 점

**고려 사항 ①: 환경 합성 품질 검증**

현재 합성 환경의 품질은 Qwen3-Thinking-235B-A22B에 의존합니다. 더 작은 모델로 환경을 합성했을 때의 품질 저하와 RL 성능의 관계를 체계적으로 분석해야 합니다.

**고려 사항 ②: 환경 수 확장 실험**

논문 자체가 한계로 인정한 바와 같이, 10개 → 100개 → 1000개 환경으로 확장 시 성능 향상 곡선(scaling curve)을 측정해야 합니다.

$$\text{Performance} = f(\text{num envs}, \text{task diversity}, \text{task difficulty})$$

**고려 사항 ③: 단계별 가치 감독 도입**

현재 이진 보상($R \in \{0, 1\}$)만 사용하여 희소 보상 문제가 존재합니다. 단계별 보상(process reward) 또는 잠재적 기반 보상 형성(potential-based reward shaping)을 결합한 연구가 필요합니다:

$$R_{\text{total}} = R_{\text{outcome}} + \lambda \sum_{t=1}^{T} r_t^{\text{step}}$$

**고려 사항 ④: 시뮬레이션-실세계 갭(Sim-to-Real Gap)**

합성 환경에서 학습한 에이전트가 실세계 API 및 도구와 상호작용 시 발생하는 분포 이동(distribution shift)을 정량적으로 측정하고 완화하는 연구가 필요합니다.

**고려 사항 ⑤: 다양한 보상 구조 실험**

현재 최종 상태 비교 기반 보상은 중간 과정의 올바름을 반영하지 못합니다. 부분 완료(partial completion)를 고려한 보상 함수 설계가 필요합니다:

$$R = \frac{|S^* \cap \hat{S}|}{|S^*|}$$

(일치하는 상태 속성 수 기반 연속 보상)

**고려 사항 ⑥: 다중 모달 에이전트로의 확장**

현재 AutoForge는 텍스트 기반 도구 호출에 집중합니다. 이미지, 코드 실행, 웹 브라우징 등 다중 모달 도구를 포함하는 환경으로 확장 시 파이프라인의 수정이 필요합니다.

**고려 사항 ⑦: MEU 정확도 평가**

LLM-as-judge의 판단 정확도(사용자 오류 감지 정밀도/재현율)가 훈련 안정성에 직접 영향을 미칩니다. MEU 판단의 오류율이 RL 성능에 미치는 민감도 분석이 필요합니다.

---

## 참고 자료 (출처)

본 답변은 다음 문서를 직접 분석하여 작성되었습니다:

1. **주 논문**: Cai, S., Fang, R., Wu, J., et al. (2025). *AutoForge: Automated Environment Synthesis for Agentic Reinforcement Learning*. arXiv:2512.22857v1 [cs.CL]

논문 내 인용된 관련 연구 (비교 분석에 활용):

2. Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300 (GRPO 알고리즘)
3. Yao, S., et al. (2024). *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*. arXiv:2406.12045
4. Barres, V., et al. (2025). *τ²-bench: Evaluating Conversational Agents in a Dual-Control Environment*. arXiv:2506.07982
5. He, W., et al. (2025). *VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-World Applications*. arXiv:2509.26490
6. Chen, C., et al. (2025a). *ACEBench: Who Wins the Match Point in Tool Usage?* arXiv:2501.12851
7. Fang, R., et al. (2025). *Towards General Agentic Intelligence via Environment Scaling*. arXiv:2509.13311 (AgentScaler)
8. Qian, C., et al. (2025a). *ToolRL: Reward is All Tool Learning Needs*. arXiv:2504.13958
9. Qian, C., et al. (2025b). *UserRL: Training Interactive User-Centric Agent via Reinforcement Learning*. arXiv:2509.19736
10. Yu, Q., et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. arXiv:2503.14476
11. Zhao, W., et al. (2025). *MUA-RL: Multi-Turn User-Interacting Agent Reinforcement Learning for Agentic Tool Use*. arXiv:2508.18669
12. Sun, H., et al. (2025). *ZeroSearch: Incentivize the Search Capability of LLMs without Searching*. arXiv:2505.04588
13. Qin, Y., et al. (2023). *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs*. arXiv:2307.16789
14. Xi, Z., et al. (2025). *AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning*. arXiv:2509.08755
15. Xu, Z., et al. (2025). *Toucan: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments*. arXiv:2510.01179
16. Shi, D., et al. (2025). *TaskCraft: Automated Generation of Agentic Tasks*. arXiv:2506.10055
17. Qwen Team. (2025c). *Qwen3 Technical Report*. arXiv:2505.09388
18. Zeng, A., et al. (2025). *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*. arXiv:2508.06471

> **정확도 고지**: 본 답변은 제공된 PDF 원문을 직접 분석하여 작성되었으며, 논문에 명시되지 않은 내용(예: 논문 발표 이후의 후속 연구 영향, 미발표 실험 결과 등)은 포함하지 않았습니다. 비교 분석 표의 일부 정성적 평가는 논문 본문 내용에 근거한 합리적 추론임을 밝힙니다.
