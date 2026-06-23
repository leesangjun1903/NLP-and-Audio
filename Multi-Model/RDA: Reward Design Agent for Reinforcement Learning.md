# RDA: Reward Design Agent for Reinforcement Learning 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 LLM 기반 보상 설계(Eureka 등)는 **수치적 피드백(성공률, 보상 통계)** 만을 사용하여 반영(reflection)을 수행하므로, 에이전트가 태스크를 "달성"하더라도 의도한 행동 방식과 **정렬(alignment)되지 않은 정책**을 학습하는 문제가 있다. RDA는 **VLM 기반 시각적 궤적 분석**을 도입하여 이 문제를 해결한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **VLM 기반 에이전틱 프레임워크** | 태스크 분해 → 시각적 궤적 평가 → 실패 진단 → 보상 수정의 폐루프 구현 |
| **서브태스크 분해** | 복잡한 태스크를 세분화하여 크레딧 할당 개선 |
| **시각적 정렬 평가** | 궤적 비디오를 VLM으로 분석하여 서브태스크 완료도 채점 |
| **이중 반영 메커니즘** | 서브태스크 반영 + 보상 반영을 분리하여 표적 수정(targeted correction) 가능 |
| **정량적 성능 향상** | HumanoidBench에서 Eureka 대비 정렬률 49% 상대적 향상 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**수동 보상 설계의 어려움**: 단일 스칼라 보상으로 다수의 경쟁적 목표를 인코딩해야 함.

**Eureka의 한계**: 수치 통계(mean, min, max)만으로 반영 수행 → 동일한 수치를 가지는 서로 다른 실패 모드를 구별 불가 → 정렬되지 않은 행동(예: 박스를 밀지 않고 던지는 행동) 발생.

### 2.2 공식적 정의

논문은 보상 설계를 **Reward Design Problem (RDP)** 로 형식화한다:

$$P = \langle \mathcal{I}, \mathcal{E}, \mathcal{R}, \mathcal{L}, \mathcal{S} \rangle$$

여기서:
- $\mathcal{I}$: 지시문 공간 (자연어 명령 등)
- $\mathcal{E} = \langle \mathcal{O}, \mathcal{A}, \mathcal{T} \rangle$: 환경 공간 (관측, 행동, 전이 동역학)
- $\mathcal{R}$: 보상 함수 공간
- $\mathcal{L}: \mathcal{E} \times \mathcal{R} \rightarrow \Pi$: 학습 알고리즘 (환경 + 보상 → 정책)
- $\mathcal{S}: \Pi \times \mathcal{I} \rightarrow \mathbb{R}$: 정렬 점수 함수

**RDP의 목표**는 정렬 점수를 최대화하는 최적 보상함수 $R^\star$를 찾는 것:

$$R^\star \in \arg\max_{R \in \mathcal{R}} S\!\left(\mathcal{L}(E, R),\, I\right)$$

### 2.3 제안 방법: RDA 알고리즘

#### 2.3.1 초기화 (Initialization)

**서브태스크 생성**: VLM이 지시문 $I$와 환경 $E$를 기반으로 $J$개의 서브태스크 분해:

$$T^1 = [t^1_1, t^1_2, \ldots, t^1_J]$$

**보상 후보 생성**: $(I, E, T^1)$ 조건 하에 $N$개의 초기 보상 후보 생성:

$$\{R^1_1, \ldots, R^1_N\} \subset \mathcal{R}$$

각 보상 함수는 서브태스크별 컴포넌트의 **가중 합**으로 구조화:

$$R = \sum_j w_j \cdot r_j^{\text{normalized}}, \quad r_j^{\text{normalized}} \in [0, 1]$$

#### 2.3.2 진화적 탐색 (Evolutionary Search)

반복 $i \in \{1, \ldots, M\}$에서 다음 단계를 수행:

**① 정책 훈련**:

$$\pi^i_n = \mathcal{L}(E, R^i_n)$$

**② 정책 롤아웃**: $K$개의 에피소드 궤적 수집:

$$\{\tau^i_{n,k}\}_{k=1}^{K}$$

**③ 시각적 분석**: 각 (궤적, 서브태스크) 쌍에 대해 VLM이 점수 및 설명 생성:

$$A^i_{n,k} = \{(s^i_{n,k,j},\, \rho^i_{n,k,j})\}_{j=1}^{J}$$

여기서 $s^i_{n,k,j} \in [0, 1]$은 서브태스크 $t^i_j$의 완료 점수.

**④ 요약 (Summarization)**: 모든 궤적·서브태스크에 걸쳐 평균 점수 계산:

$$\bar{s}^i_n = \mathbb{E}_{k,j}\!\left[s^i_{n,k,j}\right]$$

각 서브태스크 $j$의 실패 패턴을 분석하여 요약 $p^i_{n,j}$ 생성, 전체 요약:

$$G^i_n = (\bar{s}^i_n,\, \{p^i_{n,j}\}_{j=1}^{J})$$

**⑤ 보상 선택**:

$$n^\star_i = \arg\max_{n \in \{1,\ldots,N\}} \bar{s}^i_n$$

**⑥ 서브태스크 반영**: VLM이 $G^i_{n^\star_i}$를 기반으로 모호하거나 불충분한 서브태스크를 수정 (총 개수 $J$ 유지):

$$t^{i+1}_j = \text{revised}(t^i_j) \text{ 또는 } t^{i+1}_j = t^i_j$$

**⑦ 보상 반영**: $(I, E, T^{i+1}, R^i_{n^\star_i}, G^i_{n^\star_i})$를 기반으로 개선된 보상 후보 생성:

$$\{R^{i+1}_n\}_{n=1}^{N}$$

### 2.4 모델 구조

```
[자연어 지시문 I + 환경 E]
        ↓
[VLM (GPT-5)] → 서브태스크 T¹ 생성
        ↓
[VLM (GPT-5)] → N개 보상 후보 {R¹ₙ} 생성
        ↓
┌─────────────────────────────────┐
│     진화적 탐색 루프 (M=5회)       │
│  RL 훈련 (SAC + SimbaV2)         │
│     ↓                           │
│  K개 롤아웃 궤적 수집              │
│     ↓                           │
│  VLM 시각 분석 → Aⁱₙₖ            │
│     ↓                           │
│  요약 Gⁱₙ 생성                   │
│     ↓                           │
│  최고 후보 선택 (n*ᵢ)             │
│     ↓                           │
│  서브태스크 반영 → T^(i+1)        │
│     ↓                           │
│  보상 반영 → {R^(i+1)ₙ}          │
└─────────────────────────────────┘
        ↓
[최종 보상 R* 및 정책 π*]
```

**RL 알고리즘**: Soft Actor-Critic (SAC)  
**정책 아키텍처**: SimbaV2  
**Agent VLM**: GPT-5 (medium reasoning)  
**Evaluation VLM**: GPT-4.1 (편향 방지를 위해 생성기와 분리)

### 2.5 성능 향상

| 벤치마크 | 메트릭 | RDA | Eureka | Dense | Sparse |
|---|---|---|---|---|---|
| **ManiSkill** | 정렬률 | **0.95** | 0.87 | 0.68 | 0.28 |
| **ManiSkill** | 성공률 | **0.90** | 0.87 | 0.74 | 0.17 |
| **HumanoidBench** | 정렬률 | **0.70** | 0.47 | 0.16 | 0.07 |
| **HumanoidBench** | 성공률 | 0.42 | **0.52** | 0.05 | 0.00 |

**주요 관찰**:
- ManiSkill에서 RDA만이 PlugCharger(가장 어려운 태스크)를 해결 (성공률 0.67, Eureka 0.00)
- HumanoidBench에서 Eureka 대비 정렬률 **49% 상대적 향상** (0.47 → 0.70)
- Eureka는 HumanoidBench에서 성공률은 높지만(0.52), 정렬되지 않은 전략(박스를 던짐)으로 달성

**Ablation 결과**:

| 구성 | 정렬률 | 성공률 |
|---|---|---|
| Eureka (기준) | 0.31 | 0.50 |
| + Checkpoint 재활용 | 0.35 | 0.55 |
| + 시각적 궤적 분석 | 0.82 | 0.83 |
| + 서브태스크 분해 (Full RDA) | **0.84** | **1.00** |

→ **시각적 궤적 분석**이 정렬률 향상의 핵심 메커니즘  
→ **서브태스크 분해**는 PlugCharger 같은 정밀 조작 태스크에서 크레딧 할당 개선

**스케일링**: 후보 수 $N$ 증가 시 성능 단조 증가:

| $N$ | 정렬률 | 성공률 |
|---|---|---|
| 1 | 0.46 | 0.20 |
| 2 | 0.58 | 0.51 |
| 4 | 0.74 | 0.90 |
| 8 | **0.84** | **1.00** |

### 2.6 한계

1. **계산 비용**: 각 후보마다 전체 RL 훈련 필요 → HumanoidBench 1태스크 당 **240~360 GPU-hours** 소요
2. **VLM 제약**: 제한된 컨텍스트 길이, 높은 추론 비용, 세밀한 시각 추론의 부정확성
3. **서브태스크 충돌**: 초기 서브태스크 보상이 이후 서브태스크 탐색을 억제하는 경우 발생 (예: 농구 태스크에서 잡기 보상이 던지기 탐색을 억제)
4. **실세계 적용 미검증**: 시뮬레이터 환경에서만 평가
5. **평가의 순환성**: VLM으로 보상을 생성하고 VLM으로 평가 → 잠재적 편향 가능성 (GPT-5 생성, GPT-4.1 평가로 부분 완화)

---

## 3. 일반화 성능 향상 가능성

### 3.1 일반화 기제 분석

RDA의 구조는 여러 측면에서 **일반화 성능**에 기여한다:

#### (a) 서브태스크 기반 구조화된 보상

보상이 서브태스크 컴포넌트의 가중 합으로 구성되어:

$$R = \sum_{j=1}^{J} w_j \cdot r_j, \quad r_j \in [0,1]$$

각 컴포넌트가 의미론적으로 해석 가능하므로, **새로운 태스크**에서도 유사한 서브태스크 패턴을 재활용할 수 있다. 예를 들어 "이동 → 접촉 → 조작" 패턴은 다양한 조작 태스크에 공통적으로 적용 가능하다.

#### (b) 환경 무관 시각 평가

VLM이 환경 코드와 시각 정보를 **동시에** 활용하여 평가하므로, 특정 환경에 과적합되지 않는다. 논문에서도 12개의 다양한 ManiSkill 태스크(pick-and-place, 삽입, 비파지 조작)와 4개의 전신 조작 태스크에서 일관된 성능을 보임으로써 일반화 가능성을 시사한다.

#### (c) 태스크 분해의 적응성

서브태스크 수를 태스크 복잡도에 따라 조정 가능:

- 단순 태스크: ~5개 서브태스크
- 복잡/장기 태스크: ~10개 서브태스크

이는 태스크 복잡도에 따른 **적응적 표현**을 가능하게 한다.

#### (d) 진화적 탐색의 다양성

여러 후보 보상($N=4~8$)을 병렬로 탐색하여 지역 최적해에 빠질 위험을 줄인다. 이전 반복의 체크포인트를 활용한 **웜 스타팅(warm-starting)**은 탐색 효율을 높이면서도 다양성을 유지한다.

### 3.2 일반화의 한계 및 미해결 과제

그러나 다음과 같은 **일반화 제약**도 존재한다:

1. **시뮬레이터 의존성**: 모든 실험이 특권적 상태(privileged state) 접근 가능한 시뮬레이터에서 수행됨. 실세계(real world) 적용 시 상태 관측의 제한으로 보상 코드 생성이 어려울 수 있음

2. **도메인 이동 취약성**: 훈련 환경의 물리 파라미터(마찰, 질량 등)가 바뀌면 기존 보상 함수의 유효성이 저하될 수 있음

3. **지시문 의존성**: 자연어 지시문의 품질과 명확성에 크게 의존하므로, 모호하거나 불완전한 지시문에서는 성능 저하 가능

4. **VLM 편향 전파**: VLM의 시각적 이해 편향이 보상 설계에 전파될 수 있으며, 이는 특정 도메인(예: 드문 물체, 비표준 환경)에서 일반화를 저해할 수 있음

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (a) 자동화된 보상 설계 패러다임의 고도화

RDA는 Eureka가 열어놓은 **LLM 기반 보상 설계** 패러다임을 한 단계 발전시킨다. 이는 다음과 같은 연구 방향을 촉진할 것으로 예상된다:

- **멀티모달 피드백 통합**: 시각 외에 촉각, 음성, 언어 설명 등 다양한 피드백 모달리티를 활용한 보상 설계
- **계층적 에이전틱 RL**: 태스크 분해 에이전트, 보상 설계 에이전트, 정책 에이전트가 협력하는 다층 에이전틱 구조

#### (b) 정렬(Alignment)의 새로운 평가 기준

논문은 **성공률** 외에 **정렬률**을 독립적 평가 지표로 도입했다. 이는 향후 RL 연구에서 "태스크를 달성했는가" 뿐만 아니라 "의도한 방식으로 달성했는가"를 함께 평가하는 문화를 확산시킬 것이다.

#### (c) 시뮬레이터-실세계 전이에 대한 시사점

시각적 정렬을 강조하는 RDA의 방법론은 **Sim-to-Real 전이**에도 중요한 함의를 가진다. 실세계에서 발생하는 안전 제약(예: 물체를 던지지 않아야 함)을 시뮬레이터 단계에서 시각적으로 포착하고 수정할 수 있다면, 실세계 적용 시 위험성을 크게 줄일 수 있다.

#### (d) VLM-RL 공진화(Co-evolution)

RDA의 성능이 VLM 능력에 직접 의존함을 보였다. VLM이 발전함에 따라 RDA의 진단 정확도와 보상 설계 품질도 자동으로 향상되는 **공진화 구조**가 성립한다.

### 4.2 앞으로 연구 시 고려할 점

#### (a) 계산 효율성 개선

현재 각 후보마다 전체 RL 훈련을 요구하는 구조는 비효율적이다. 고려할 방향:

- **RL 훈련 전 내부 반영(Internal Reflection)**: 논문에서 Lu et al. (2024)의 방법을 언급하듯, RL 훈련 없이 보상 함수를 사전 진단하는 메커니즘 개발
- **메타 학습(Meta-Learning)**: 이전 태스크에서 학습한 보상 컴포넌트 패턴을 새로운 태스크에 빠르게 적용
- **분산 탐색**: 여러 후보를 병렬 GPU 클러스터에서 동시 평가

#### (b) 서브태스크 충돌 해결

논문에서 언급한 "농구 태스크에서 잡기 보상이 던지기 탐색을 억제"하는 문제에 대해:

- **명시적 서브태스크 상태 조건화**: 현재 서브태스크 상태에 따라 정책이 다른 보상 신호를 받도록 설계
- **계층적 강화학습(HRL)**: 상위 정책이 하위 정책의 서브태스크 전환을 관리
- **커리큘럼 학습**: 서브태스크를 순차적으로 학습하는 구조

#### (c) 실세계 적용성 검증

시뮬레이터에서 검증된 방법이 실세계에서도 유효한지 확인이 필요:

- **특권적 상태 없이 보상 설계**: 카메라 등 실세계에서 접근 가능한 센서만 사용
- **DreamEureka (Nahrendra et al., 2026)**과 같은 실세계 통합 실험 수행

#### (d) VLM 평가 편향 제어

생성 모델과 평가 모델의 분리(GPT-5 생성, GPT-4.1 평가)로 부분적으로 해결했지만, 다음을 추가로 고려:

- **인간 평가자 비교 연구**: VLM 기반 정렬률과 인간 평가의 상관관계 검증
- **다양한 VLM 앙상블 평가**: 단일 VLM 평가의 편향을 줄이기 위한 앙상블

#### (e) 일반화 벤치마크 확장

현재 실험은 ManiSkill(단기 지평)과 HumanoidBench(장기 지평)에 국한됨. 향후 고려:

- **로봇 유형 다양화**: 이족 보행 외에 사족 보행, 다관절 팔 등
- **태스크 도메인 확장**: 가정 환경, 제조 환경, 야외 환경 등

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법 | 피드백 유형 | 한계 |
|---|---|---|---|---|
| **Eureka** (Ma et al.) | 2023 | LLM 진화 탐색으로 보상 코드 생성 | 수치 통계 (mean/min/max) | 시각적 정렬 부재, 실패 모드 구별 불가 |
| **DrEureka** (Ma et al.) | 2024 | 안전 인식 프롬프팅 + 실세계 실험 | 수치 통계 + 안전 제약 | 여전히 시각적 피드백 없음 |
| **VIP** (Ma et al.) | 2022 | 비디오에서 암묵적 가치 함수 사전학습 | 데모 비디오 | 대규모 데모 필요 |
| **PEBBLE** (Lee et al.) | 2021 | 인간 선호도로 보상 학습 | 인간 비교 피드백 | 높은 인간 주석 비용 |
| **RL-VLM-F** (Wang et al.) | 2024 | VLM이 선호도 레이블 자동 생성 | VLM 비교 피드백 | 희소하고 거친 감독 |
| **RoboReward** (Lee et al.) | 2026 | 범용 VL 보상 모델 | VL 임베딩 유사도 | 병렬 시뮬레이터에서 비용 높음 |
| **Grove** (Cui et al.) | 2025 | CLIP 기반 정렬 보상 추가 | CLIP 유사도 | 실패 진단 정보 없음 |
| **VICTOR** (Hung et al.) | 2025 | 계층적 비전-지시 상관 보상 | 데모 + 지시문 | 데모 필요 |
| **Prompt-to-Policy** (Chung et al.) | 2026 | 역할 특화 에이전트 + VLM 궤적 판단 | VLM 궤적 판단 | 동시 연구, 상세 비교 미제공 |
| **RDA (본 논문)** | 2026 | VLM 기반 에이전틱 루프 + 서브태스크 분해 | 시각적 궤적 분석 + 언어 진단 | 계산 비용, 서브태스크 충돌 |

**핵심 차별점**: RDA는 단순히 정렬 불일치를 **감지**하는 것을 넘어, 서브태스크 수준의 세밀한 **진단과 표적 수정**을 제공하는 유일한 방법이다.

---

## 참고자료

- **본 논문**: Lee, H., Subramanian, A., et al. "RDA: Reward Design Agent for Reinforcement Learning." *Reinforcement Learning Journal*, 2026. arXiv:2606.01672v1
- Ma, Y. J., et al. "Eureka: Human-level reward design via coding large language models." arXiv:2310.12931, 2023.
- Ma, Y. J., et al. "DrEureka: Language model guided sim-to-real transfer." RSS, 2024.
- Christiano, P., et al. "Deep reinforcement learning from human preferences." NeurIPS 30, 2017.
- Wang, Y., et al. "RL-VLM-F: Reinforcement learning from vision language foundation model feedback." arXiv:2402.03681, 2024.
- Lee, T., et al. "RoboReward: General-purpose vision-language reward models for robotics." arXiv:2601.00675, 2026.
- Cui, J., et al. "Grove: A generalized reward for learning open-vocabulary physical skill." CVPR, 2025.
- Hung, K.-H., et al. "VICTOR: Learning hierarchical vision-instruction correlation rewards for long-horizon manipulation." ICLR, 2025.
- Chung, W., et al. "Prompt-to-Policy: Agentic engineering for reinforcement learning." 2026.
- Tao, S., et al. "ManiSkill3: GPU parallelized robotics simulation and rendering for generalizable embodied AI." arXiv:2410.00425, 2024.
- Sferrazza, C., et al. "HumanoidBench: Simulated humanoid benchmark for whole-body locomotion and manipulation." arXiv:2403.10506, 2024.
- Haarnoja, T., et al. "Soft actor-critic algorithms and applications." arXiv:1812.05905, 2018.
- Lee, H., et al. "Simba: Simplicity bias for scaling up parameters in deep reinforcement learning." arXiv:2410.09754, 2024.
- Muslimani, C., et al. "Towards improving reward design in RL: A reward alignment metric for RL practitioners." arXiv:2503.05996, 2025.
- Lu, C., et al. "The AI Scientist: Towards fully automated open-ended scientific discovery." arXiv:2408.06292, 2024.
- OpenAI. "GPT-4.1." https://openai.com/index/gpt-4-1/, 2025.
- OpenAI. "GPT-5." https://openai.com/index/introducing-gpt-5/, 2025.
