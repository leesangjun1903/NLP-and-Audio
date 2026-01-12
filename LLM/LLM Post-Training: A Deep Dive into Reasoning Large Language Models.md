# LLM Post-Training: A Deep Dive into Reasoning Large Language Models

### 1. 논문의 핵심 주장 및 주요 기여

이 논문은 대규모 언어 모델(LLM)의 포스트트레이닝 기법을 체계적으로 탐구하는 포괄적인 서베이로, 프리트레이닝 이후 모델의 능력을 정제하고 인간의 의도와 윤리적 고려사항에 정렬하는 데 중점을 두고 있습니다. 저자들은 세 가지 핵심 포스트트레이닝 전략을 강조합니다:[1]

**주요 기여**[1]
- 파인튜닝, 강화학습(RL), 테스트타임 스케일링을 포함한 포괄적이고 체계적인 포스트트레이닝 방법론 검토 제공
- 포스트트레이닝 기술의 역할과 상호 관계를 명확히 하는 구조화된 택소노미 제시
- 실제 응용 프로그램을 위한 벤치마크, 데이터셋, 평가 지표에 대한 실용적 지침 제공

### 2. 해결하고자 하는 문제, 제안 방법 및 모델 구조

#### 2.1 주요 해결 문제

논문이 다루는 핵심 문제들은 다음과 같습니다:[1]

1. **환각(Hallucinations)**: LLM이 사실에 기반하지 않은 내용을 생성하는 문제
2. **논리적 일관성 유지**: 장문의 담론에서 맥락 관련성 부재
3. **추론의 성질에 대한 논쟁**: LLM의 추론이 인간의 논리적 추론과 근본적으로 다름
4. **인간 의도 정렬**: 모델이 사용자 기대 및 윤리 기준과 일치하지 않을 수 있음

#### 2.2 제안하는 방법론

**최대우도추정(MLE) 기반의 기초**[1]

포스트트레이닝의 기초는 프리트레이닝에서 사용되는 다음과 같은 MLE 목표 함수입니다:

$$\mathcal{L}_{\text{MLE}} = -\sum_{t=1}^{T} \log P(y_t \mid y_{\lt t}, X)$$

여기서 $X$는 입력(프롬프트 또는 문맥)이고, $Y = \{y_1, y_2, \ldots, y_T\}$는 목표 출력 시퀀스입니다.

**마르코프 결정 프로세스(MDP) 기반 RL 프레임워크**[1]

LLM의 자동회귀 생성을 순차적 의사결정 문제로 모델링하면:

- 상태 $s_t$: 현재까지 생성된 토큰 시퀀스
- 행동 $a_t$: 다음 생성할 토큰
- 보상 $R(s_t, a_t)$: 생성된 텍스트의 품질 평가

**정책 그래디언트 및 어드밴티지 함수**[1]

RL 기반 시퀀셜 추론에서:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E} \left[ \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) A(s_t, a_t) \right]$$

여기서 어드밴티지 함수는:

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$

$V(s_t)$는 상태 $s_t$에서의 예상 반환값입니다.

#### 2.3 주요 포스트트레이닝 방법들

**1) 파인튜닝 (Fine-Tuning)**[1]

기본적인 감독 학습 기반 적응:
- **지시어 파인튜닝**: 다양한 작업에 대한 지시어-응답 쌍으로 학습
- **대화 파인튜닝**: 멀티턴 대화 형식으로 학습
- **CoT 파인튜닝**: 단계별 추론 과정을 학습

**2) 강화학습 최적화 방법들**[1]

**PPO (Proximal Policy Optimization)**

클리핑된 목적 함수를 사용하여 정책을 최적화:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_t \left[ \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t) \right]$$

여기서 $r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{ref}}}(a_t \mid s_t)}$는 확률 비율입니다.

**KL 패널티를 포함한 정책 최적화**

보상 최적화 중 기본 모델으로부터 과도한 편향을 방지:

$$J = \mathbb{E}_{x,y \sim D} \left[ r(x, y) - \beta \text{KL}(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)) \right]$$

여기서 $\beta$는 KL 패널티 강도를 조절합니다.

**RLHF (Reinforcement Learning from Human Feedback)**[1]

세 단계 프로세스:
1. SFT(감독 파인튜닝): 고품질 인간 생성 예제로 초기 학습
2. 보상 모델 학습: 인간 선호도 데이터를 사용하여 보상 함수 학습
3. RL 파인튜닝: PPO를 사용하여 보상 모델 점수 최대화

**Bradley-Terry 모델 (쌍 비교 기반)**[1]

두 응답 $y_j$와 $y_k$ 사이의 선호도:

$$P(y_j \succ y_k \mid x) = \frac{\exp(R(x, y_j))}{\exp(R(x, y_j)) + \exp(R(x, y_k))}$$

손실 함수:

$$\mathcal{L}_{\text{BT}} = -\sum_{(x, y_j \succ y_k) \in D} \log P(y_j \succ y_k \mid x)$$

**DPO (Direct Preference Optimization)**[1]

명시적 보상 모델 없이 직접 선호도 최적화:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{x, y_w, y_l \sim D_{\text{train}}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

**GRPO (Group Relative Policy Optimization)**[1]

각 질문 $q$에 대해 다중 출력 그룹을 샘플링하고 상대적 보상으로 평가:

그룹 내 정규화된 보상:

$$r_i' = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

GRPO 목적 함수:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G J(o_i, \theta, q) \right]$$

여기서:

$$J(o_i, \theta, q) = \mathbb{E}_t \left[ \min(r_t^{(i)} A_t^{(i)}, \text{clip}(r_t^{(i)}, 1-\epsilon, 1+\epsilon) A_t^{(i)}) - \beta \text{KL}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

**3) 테스트타임 스케일링 (Test-Time Scaling)**[1]

추론 시 계산 자원을 동적으로 조정:

- **Chain-of-Thought (CoT)**: 단계별 추론 유도
- **Tree-of-Thoughts (ToT)**: 여러 사고 경로를 나무 구조로 탐색
- **Best-of-N 검색**: N개 후보 중 최상의 선택
- **Self-Consistency**: 다중 추론 경로의 합의 활용

### 3. 모델 구조 및 성능 향상

#### 3.1 DeepSeek-R1의 멀티 스테이지 구조[2]

DeepSeek-R1은 다음과 같은 단계적 구조를 채택합니다:

**Step 1: 콜드스타트 RL 단계**
- 고품질 CoT 예제와 요약 데이터로 초기 모델 파인튜닝
- 구조화된 CoT 포맷으로 안정성 확보

**Step 2: 거부 샘플링 및 파인튜닝**
- RL 수렴 후 고품질 응답만 필터링
- 800k 샘플 규모의 데이터셋 확장

**Step 3: 추론 중심 RL**
- GRPO 알고리즘 적용
- 정확도 보상 + 언어 일관성 보상 결합

**Step 4: 인간 정렬을 위한 제2 RL 단계**
- 도움이 되고 해롭지 않은 특성 추가 정렬

**Step 5: 소규모 모델 증류**
- 대규모 모델의 추론 능력을 소규모 모델로 이전

#### 3.2 성능 향상 결과

**DeepSeek-R1-Zero의 성능 향상**[3][2]

AIME 2024 벤치마크에서의 개선:
- 초기: 15.6% Pass@1
- 최종: 71.0% Pass@1 (Majority Vote로 86.7%)
- 이는 OpenAI-o1-0912과 동등 수준

**DeepSeek-R1의 범용 성능**[3]

- MMLU: 90.8% (DeepSeek-V3 대비 향상)
- MMLU-Pro: 84.0%
- GPQA Diamond: 71.5%
- MATH-500: 97.3% (OpenAI-o1-1217과 동등)

#### 3.3 효율성 개선

**테스트타임 스케일링의 효율성**[4]

Compute-Optimal Scaling Strategy (COS)는:
- Best-of-N 대비 4배 효율 향상
- 정확도 94% 유지하면서 계산량 감소

**RL 포스트트레이닝의 스케일링 법칙**[5]

- 고정 데이터에서 큰 모델이 더 나은 샘플 효율 달성
- 데이터 제약 상황에서 고품질 데이터 재사용 효과적
- 최적 학습 효율: 분석적 학습 효율 항 $k(N)$ 적용

### 4. 한계와 미해결 과제

#### 4.1 Catastrophic Forgetting[1]

파인튜닝 과정에서 새로운 작업 학습 시 기존 능력 저하:
- **관찰된 저하**: 기본 모델 능력의 약 29% 저하
- **현황**: 심지어 LoRA 같은 매개변수 효율적 방법도 완전히 해결하지 못함
- **향후 방향**: 
  - 지속적 학습 전략 및 정규화 기법 개선
  - 개별 과제별 어댑터 격리
  - 상충하는 목표 간 균형 조정

#### 4.2 Reward Hacking[6][1]

모델이 진정한 추론 품질보다 프록시 지표를 최적화하려는 경향:

**자기강화학습의 한계**[6]

자기강화학습(Self-Rewarding Training, SRT)은 초기에 성능 개선을 보이지만:
- 확장 훈련 중 **완전한 성능 붕괴** 관찰
- 모든 테스트된 기본 모델에서 발생 (Llama-3.1-8B, 70B 등)
- 해결 방안: Ground-truth RL 훈련 신호의 필요성

**미해결 과제**[1]
- Reward misgeneralization: 모델이 표면적 프록시 지표 과최적화
- 실패 궤적 효과적 활용 부족
- Process-based 및 Outcome-based 보상 결합 전략 필요

#### 4.3 Inference-Time Trade-offs[1]

테스트타임 스케일링의 계산 오버헤드:
- **지연시간 증가**: 신뢰도 추정기 사용 시 약 18% 오버헤드
- **비효율성**: 과도한 계산량으로 환경 영향 증가
- **동적 할당 문제**: 쿼리 복잡도 기반 적응적 계산 배치의 어려움

#### 4.4 생성화 능력의 한계

**추론의 보편적 적용성 문제**[7]

최신 추론 모델(DeepSeek-R1, o1)이 보이는 패턴:
- 검증 가능한 도메인(코드, 수학)에서 뛰어난 성능
- 창의적 작문, 일반 상식 등 검증 불가능한 도메인에서는 혼재된 성능
- 추론 길이와 실제 성능 간 약한 상관관계 (r = -0.34, MMLU 포화)

**해결되지 않은 과제**:
- 장거리 추론의 신뢰성 검증 메커니즘 부족
- 멀티모달 추론 시 combinatorial state explosion (128k 토큰 이상)

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 테스트타임 컴퓨팅을 통한 일반화[7][1]

**핵심 발견**

작은 모델도 충분한 추론 시간 컴퓨팅으로 훨씬 큰 모델과 동등한 성능 달성:

- **성능 동등성**: 테스트타임 스케일링으로 14배 큰 모델과 동등 성능
- **비용 효율성**: 쉬운~중간 난이도 작업에서 4배 계산 절감
- **최적화 전략**: 쉬운 질문은 순차적 개선, 어려운 질문은 병렬 탐색

**제한사항**:
- 가장 어려운 작업에서는 프리트레이닝 규모가 여전히 우월
- 추론 토큰 제약이 있는 환경에서는 프리트레이닝이 필수

#### 5.2 CoT 기반 추론이 일반화를 돕는 메커니즘[7]

**구조적 정보 처리**

CoT는 정보를 작은 청크로 처리하여:
- 복잡성 관리 용이
- 중간 정보를 문맥 윈도우에 저장
- 매개변수 오버로딩 경감

**결과적 일반화 특성**:
- 숨겨진 상태 없이도 상태 공간에서 반복 처리 가능
- 다양한 도메인으로의 전이 학습 개선

#### 5.3 강화학습을 통한 자기진화 능력[2]

**DeepSeek-R1-Zero의 창발적 특성**

순수 RL 훈련을 통해 다음이 자동으로 나타남:
- **자기성찰(Reflection)**: 이전 단계 재검토 및 재평가
- **대안 탐색**: 문제 해결의 여러 접근법 시도
- **적응적 성능**: 복잡한 문제에 더 많은 계산 할당

**일반화 상황별 성능 패턴**:

| 모델 유형 | 검증 가능 도메인 | 검증 불가 도메인 | 비용 효율 |
|---------|------------|-------------|--------|
| 추론 모델 | 뛰어남 ↑ | 혼재 ↔ | 높음 |
| 표준 LLM | 양호 | 양호 | 낮음 |

### 6. 2020년 이후 관련 최신 연구 비교 분석

#### 6.1 연도별 연구 동향

**2020-2022: RLHF 시대**
- **InstructGPT (2022)**: 인간 피드백 기반 RL 도입으로 신기원 개척
- **주요 특징**: PPO 기반 정책 최적화, 명시적 보상 모델 훈련

**2023: 효율성과 직접 최적화 중심**
- **DPO (2023)**: 보상 모델 제거, 직접 선호도 최적화로 계산 비용 감소
- **CoT Distillation (2023)**: 교사 모델의 추론 능력 학생 모델로 전이
- **Reinforced Self-Training (ReST)**: 오프라인 RL을 통한 효율성 증대

**2024: GRPO와 순수 RL의 등장**
- **DeepSeek-Math + GRPO (2024.04)**: Value network 제거로 메모리 효율화
- **핵심 아이디어**: 절대적 가치 평가 대신 그룹 내 상대적 평가
- **영향**: 이후 대규모 추론 모델의 기초 알고리즘으로 채택

**2025: 다각화 및 적응적 방법론**
- **DeepSeek-R1 (2025.01)**: GRPO로 순수 RL 성공, 인간 정렬 단계 추가
- **Q-Sharp (2025.02)**: Distributional RL로 이론적 보증 제공
- **MAPoRL (2025.02)**: 멀티에이전트 협력적 포스트트레이닝
- **Light-R1 (2025.03)**: Curriculum 기반 SFT + DPO + RL 통합
- **Iterative DPO (2025.03)**: DPO 반복 적용으로 추론 개선

#### 6.2 주요 알고리즘 비교

| 알고리즘 | 연도 | 주요 혁신 | 계산 효율 | 보상 신호 | 적용 대상 |
|---------|------|----------|---------|---------|---------|
| PPO | 2017 | 클리핑된 목적함수 | 중간 | 명시적 V함수 | 초기 RLHF |
| RLHF | 2022 | 인간 피드백 | 낮음 | 학습된 보상 모델 | InstructGPT, ChatGPT |
| DPO | 2023 | 직접 선호도 | 높음 | 암시적 보상 | Zephyr, Mistral |
| GRPO | 2024 | 그룹 상대 평가 | 최고 | 그룹 내 비교 | DeepSeek-R1 |
| Q-Sharp | 2025 | Distributional RL | 높음 | 이론적 보증 | 장기 추론 |
| Iterative DPO | 2025 | 반복 최적화 | 높음 | 누적 선호도 | 확장된 CoT |

#### 6.3 성능 진전 추이

**수학 추론 벤치마크(AIME 2024) 성능**

```
2023: GPT-3.5-Turbo      10% 정도
2024 초: DeepSeekMath    47.8%
2024 말: o1             96.4%
2025: DeepSeek-R1       79.8%
2025: DeepSeek-R1-Zero  71.0%
```

**일반 능력 벤치마크(MMLU)**

- DeepSeek-R1: 90.8% (향상도: 72.0% → 90.8%)
- GPT-4o: ~88% 수준
- 추론 훈련이 일반 지식 성능도 향상

#### 6.4 현재의 주요 연구 방향

**1) 확장성과 효율성**
- 멀티모달 RL 통합 (이미지, 비디오 포함)
- 장문 문맥(128k+) 처리의 효율화
- Over-thinking 현상 제거 (추론 길이 최적화)

**2) 신뢰도와 안전성**
- Uncertainty-aware RL: 모델의 신뢰도 표현 학습
- Robust reward modeling: Reward hacking 방지
- Adversarial training: 적대적 입력에 대한 견고성

**3) 적응적 및 개인화된 포스트트레이닝**
- Privacy-preserving adaptation: 차등 개인정보보호 기반 파인튜닝
- Federated learning: 분산 환경에서의 포스트트레이닝
- Heterogeneous model collaboration: 이질적 모델 간 협력

**4) 자기강화학습의 한계 극복**
- Verifiable reward와 self-reward의 결합
- Failure trajectory의 효과적 활용
- Curriculum-based self-improvement

### 7. 향후 연구에 미치는 영향 및 고려사항

#### 7.1 이론적 및 개념적 기여

**파러다임 시프트**[7][1]

이 논문은 LLM 개선의 초점을 **프리트레이닝에서 포스트트레이닝으로 전환**:

- 프리트레이닝: 기초 능력 확보 (높은 초기 비용)
- 포스트트레이닝: 세밀한 조정 및 추론 최적화 (효율적)

**실무적 시사점**:
- 더 이상 단순히 모델 크기 확대만으로는 부족
- 포스트트레이닝 품질이 최종 성능의 주요 결정요소
- 계산 예산의 동적 분배가 중요

#### 7.2 추론 모델의 미래 방향

**일반화 능력의 확대**[7]

현재 추론 모델의 한계:
- 검증 불가능한 도메인(창의성, 소설 쓰기)에서 혼재된 성능
- 추론 길이와 정확도 간 약한 상관관계

**개선 전략**:
1. **하이브리드 보상 함수**: Process-based + Outcome-based 결합
2. **다중 도메인 RL**: 여러 도메인을 동시에 최적화
3. **적응적 계산 분배**: 복잡도에 따른 동적 리소스 할당

#### 7.3 구체적 연구 고려사항

**1) 보상 모델링의 개선**[1]

**현황**: 보상 misgeneralization, reward hacking 문제 지속

**제안 방향**:
- Contrastive step-wise evaluation으로 세밀한 신용 할당
- Dynamic credit assignment (Temporal difference 학습 적용)
- Failure-aware training (부정 예제 포함)

**핵심 수식**:

$$A(s_t, a_t) = \sum_{s \in S_{\text{future}}} \gamma^{s-t} r(s) - V(s_t)$$

Temporal difference를 변환기에 적응:
$$\Delta V_t = V(s_{t+1}) - V(s_t) + r(s_t, a_t)$$

**2) 효율적 RL 훈련**[1]

**현황**: 높은 계산 오버헤드, Over-thinking 현상 (추론의 22% 낭비)

**해결책**:
- Partial rollout strategies: 전체 시퀀스 대신 부분 탐색
- Adaptive length penalty: 학습된 압축 변환기 사용
- Hybrid MCTS + GRPO: 탐색-착취 트레이드오프 개선

```
전략 1: 순차적 개선 (Sequential)  → 쉬운 문제에 효율적
전략 2: 병렬 탐색 (Parallel)      → 어려운 문제에 효율적
동적 선택: 문제 난이도 기반
```

**3) 다중 목표 최적화**[1]

**상충하는 목표**:
- 정확도 vs 응답 길이
- 도움이 됨 vs 해롭지 않음
- 추론 성능 vs 언어 유창성

**권장 접근법**:
- Multi-attribute scoring: 여러 차원의 점수 결합
- Lexicographic preference: 우선순위 계층 설정
- Pareto frontier optimization: 상충 최소화

**4) 일반화 능력 강화를 위한 전략**[1]

**Catastrophic Forgetting 완화**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{new}} + \lambda \sum_i (\theta_i - \theta_i^*)^2$$

(Elastic Weight Consolidation)

**Out-of-domain 적응**:

$$\mathcal{L}_{\text{augmented}} = \mathcal{L}_{\text{in-domain}} + \alpha \mathcal{L}_{\text{ood}}$$

**5) 안전성과 신뢰도**[1]

**미해결 과제**:
- 신뢰도 추정의 18% 지연시간 오버헤드
- 포스트트레이닝 중 29% 기본 능력 저하

**향후 개선**:
- Efficient uncertainty quantification
- Robustness certification
- Adversarial training integration

#### 7.4 실무 배포 시 고려사항

**리소스 할당 결정**

| 상황 | 추천 전략 | 근거 |
|------|---------|------|
| 새로운 도메인 학습 필요 | 프리트레이닝 강화 | 기본 능력 필요 |
| 추론 정확도 개선 필요 | 테스트타임 스케일링 | 비용 효율성 |
| 인간 정렬 필요 | RLHF 또는 GRPO | 안정성 우수 |
| 빠른 반응 필요 | DPO 또는 경량 RL | 계산 효율 |

**프라이버시와 보안**[1]

- **감시 대상**: 민감 데이터 메모리화
- **완화 방법**: Differential privacy fine-tuning, Federated learning
- **검증**: Membership inference 공격 테스트

#### 7.5 벤치마킹 및 평가 표준화

논문이 강조하는 **포괄적 평가 체계**:[1]

**추론 성능**:
- MATH, GSM8K, AIME (수학)
- 코딩 벤치마크 (코드 생성)
- MultiHop QA (추론 연쇄)

**정렬 평가**:
- HelpSteer (다중 속성)
- UltraFeedback (선호도)
- HH-RLHF (안전성)

**신뢰도**:
- Calibration (신뢰도 표현)
- Uncertainty (불확실성 정량화)
- Robustness (적대적 예제)

***

## 결론

"LLM Post-Training: A Deep Dive into Reasoning Large Language Models"는 포스트트레이닝 기법의 포괄적 체계화를 제시하며, 특히 **강화학습 기반 추론 향상**이 향후 LLM 발전의 핵심이 될 것을 강조합니다. 

**주요 시사점**:

1. **패러다임 시프트**: 프리트레이닝 우월성에서 포스트트레이닝 중요성으로 전환
2. **효율성 개선**: GRPO 등의 새로운 알고리즘으로 계산 비용 획기적 감소
3. **미해결 과제**: Catastrophic forgetting, reward hacking, 일반화 한계는 여전히 해결 필요
4. **미래 방향**: 멀티모달 RL, 적응적 계산, 안전성 강화가 중요

2025년 현재 진행 중인 연구들은 이러한 과제들을 점진적으로 해결하고 있으며, 특히 **자기강화학습의 극복**, **다중 목표 최적화**, **도메인 일반화** 등이 차세대 포스트트레이닝의 핵심 분야가 될 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/270a5f39-5677-4422-be41-8a5b0c35d416/2502.21321v2.pdf)
[2](https://www.nature.com/articles/s41586-025-09422-z)
[3](https://arxiv.org/html/2501.12948v1)
[4](https://aclanthology.org/2025.acl-long.140.pdf)
[5](https://openreview.net/forum?id=KBut2YCZ4g)
[6](https://arxiv.org/pdf/2505.21444.pdf)
[7](https://www.interconnects.ai/p/why-reasoning-models-will-generalize)
[8](https://arxiv.org/pdf/2411.00062.pdf)
[9](http://arxiv.org/pdf/2502.18439.pdf)
[10](https://arxiv.org/pdf/2502.20548.pdf)
[11](http://arxiv.org/pdf/2403.08694.pdf)
[12](http://arxiv.org/pdf/2411.14457.pdf)
[13](https://arxiv.org/pdf/2308.08998.pdf)
[14](https://arxiv.org/html/2503.21807v1)
[15](https://arxiv.org/pdf/2503.12854.pdf)
[16](https://dev.datascienceassn.org/sites/default/files/pdf_files/LLM%20Post-Training%20-%20A%20Deep%20Dive%20into%20Reasoning%20Large%20Language%20Models.pdf)
[17](https://www.youtube.com/watch?v=eULIf02frIw)
[18](https://turingpost.co.kr/p/topic-40-grpo-flow-grpo)
[19](https://aclanthology.org/2025.coling-main.330/)
[20](https://arxiv.org/pdf/2501.12948.pdf)
[21](https://arxiv.org/html/2507.21931v1)
[22](https://arxiv.org/abs/2504.05518)
[23](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/deepseek-r1/)
[24](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
[25](https://www.sciencedirect.com/science/article/pii/S240595952500133X)
[26](https://smartest-suri.tistory.com/73)
[27](https://arxiv.org/pdf/2505.10543.pdf)
[28](https://arxiv.org/pdf/2510.00977.pdf)
[29](https://arxiv.org/html/2509.02547v1)
[30](https://arxiv.org/abs/2503.23487)
[31](https://arxiv.org/pdf/2504.20571.pdf)
[32](https://arxiv.org/html/2503.06072v3)
[33](https://arxiv.org/abs/2410.23123)
[34](https://arxiv.org/abs/2501.12948)
[35](https://arxiv.org/html/2504.20571v1)
[36](https://arxiv.org/abs/2503.10573)
[37](https://arxiv.org/html/2503.10460v3)
