# FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning

### 1. 핵심 주장 및 주요 기여

**FAPO(Flawed-Aware Policy Optimization)**는 LLM의 강화학습 기반 추론 능력 향상을 위해 설계된 혁신적 알고리즘입니다. 이 논문의 핵심 주장은 다음과 같습니다:[1]

#### 1.1 핵심 문제 인식

기존의 RLVR(Reinforcement Learning with Verifiable Rewards) 패러다임에서는 정답에 도달한 모든 rollout을 동일하게 보상합니다. 그러나 실제로는 모델이 **답변 추측(answer-guessing)** 이나 **논리 점프(jump-in-reasoning)** 같은 결함 있는 패턴을 통해 정답에 도달할 수 있으며, 이러한 "결함이 있는 긍정(flawed-positive)" rollout들이 정책에 불신뢰할 수 있는 추론 패턴을 내재화시킨다는 점을 파악했습니다.[1]

#### 1.2 주요 기여

1. **결함 긍정의 체계적 분석**: RLVR 과정에서 결함 긍정의 분포와 영향을 처음으로 체계적으로 분석했습니다. 연구 결과에 따르면:[1]
   - 초기 학습 단계에서는 결함 긍정이 정답에 도달하는 지름길로 작용하여 성능 향상을 가속화합니다.[1]
   - 모델이 성숙해질수록 이러한 결함 긍정은 신뢰할 수 없는 패턴을 강화하여 추론 능력을 제한합니다.[1]

2. **Parameter-Free Reward Penalty 메커니즘**: FAPO는 매개변수 조정 없이 결함 긍정 rollout에 동적 페널티를 적용하여 초기 따뜻하기 단계에서의 빠른 성능 향상과 후기 정제 단계에서의 신뢰할 수 있는 추론으로의 자연스러운 전환을 가능하게 합니다.[1]

3. **Generative Reward Model(GenRM)**: 결함 긍정을 정확하게 감지하기 위해 프로세스 레벨 보상을 갖춘 생성형 보상 모델을 도입했습니다. 이는 기존의 discriminative PRM과는 달리 해석 가능한 이유를 함께 제공합니다.[1]

***

### 2. 해결하고자 하는 문제와 제안된 방법

#### 2.1 문제의 정의

RLVR 기반 추론 최적화에서 근본적인 도전 과제는 **이진 결과 보상(binary outcome reward)**이 결함이 있는 추론 경로와 완전히 정확한 추론을 구분하지 못한다는 점입니다. 형식적으로:[1]

$$\text{Flawed Positive 조건: } \hat{a}_\pi = a^* \text{ and } \exists t \in \{1,2,\ldots,n\} \text{ s.t. step } x_t \text{ is logically invalid}$$

여기서 $\hat{a}_\pi$는 정책 $\pi$에 의한 예측 답변, $a^*$는 정답입니다.[1]

#### 2.2 GRPO 기반 정책 최적화

FAPO의 기초가 되는 Group Relative Policy Optimization(GRPO)는 다음과 같이 정의됩니다.[1]

**Advantage 추정:**

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{R_i\}^G_{i=1})}{\text{std}(\{R_i\}^G_{i=1})}$$

여기서 $r_i$는 각 단계의 보상, $R_i$는 전체 rollout의 보상입니다.[1]

**정책 목표 함수:**

$$J(θ) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \{o_i\}^G_{i=1} \sim \pi_{θ_{old}}(\cdot|q)} \frac{1}{\sum^G_{i=1}|o_i|} \sum^G_{i=1} |o_i| \sum^{|o_i|}_{t=1} \left[ \min\left( \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{θ_{old}}(o_t|q, o_{<t})} \hat{A}_{i,t}, \text{clip}\left(\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{θ_{old}}(o_t|q, o_{<t})}, 1-\epsilon_l, 1+\epsilon_h\right) \hat{A}_{i,t}\right)\right]$$

기본 보상 함수는 다음과 같습니다.[1]

$$R_{\text{RLVR}} = \begin{cases} 1 & \text{if } I(o, a^*) \\ -1 & \text{Otherwise} \end{cases}$$

#### 2.3 FAPO의 핵심 메커니즘

##### (1) Generative Reward Model (GenRM) 학습

결함 긍정을 정확하게 감지하기 위해, FAPO는 단계별 RL 최적화를 통해 GenRM을 훈련합니다.[1]

**결합 보상 함수:**

$$R_{\text{FAPO-GenRM}} = R_{\text{Outcome}} + R_{\text{Process}}$$

$$R_{\text{Outcome}} = \begin{cases} 1 & \text{if } \hat{y}_\theta = y^* \\ -1 & \text{Otherwise} \end{cases}$$

$$R_{\text{Process}} = \begin{cases} -\frac{|\hat{t}_\theta - t^*|}{n} & \text{if } \hat{y}_\theta = y^* = \text{FP} \\ 0 & \text{Otherwise} \end{cases}$$

여기서 $\hat{t}_\theta$와 $t^*$는 각각 예측된 오류 위치와 실제 오류 위치, $n$은 전체 단계 수입니다.[1]

이 설계의 핵심 아이디어는 두 가지입니다:[1]
- **추측 넘어서기**: 단순한 이진 판정 대신 정확한 오류 위치를 찾도록 유도
- **자연스러운 보상 이동**: 초기에는 정확성 개선이 주요 목표이지만, 정확도가 포화되면서 프로세스 최적화로 자연스럽게 전환

##### (2) 결함 긍정 페널티 메커니즘

최종 RL 최적화에서 FAPO는 적응형 보상 페널티를 도입합니다:[1]

```math
R_{\text{FAPO}}(o, a^*|\theta) = R_{\text{RLVR}}(o, a^*) + R_\Delta(o, a^*|\theta)
```

```math
R_\Delta(o, a^*|\theta) = \begin{cases} -\lambda & \text{if } I(o, a^*) \text{ and } \hat{y}_\theta(o, a^*) = \text{FP} \\ 0 & \text{Otherwise} \end{cases}
```

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{R_i\}^G_{i=1})}{\text{std}(\{R_i\}^G_{i=1})}$$

#### 2.4 이론적 분석: 최적화 동역학

FAPO는 학습의 자연스러운 전환을 가능하게 하는 이론적 기반을 제공합니다. $\rho = \alpha/\beta$를 긍정/부정 샘플의 비율로 정의하면:[1]

**최적화 방향 전환 조건:**

$$\hat{A}_{\text{Flawed}} = \frac{1-\lambda - \mu_{\text{FAPO}}}{\sigma_{\text{FAPO}}} < 0 \Rightarrow \frac{\alpha}{\beta} > \frac{2}{\lambda} - 1$$

따라서 $\rho_{\text{shift}} = 1$일 때 $\lambda = 1$로 설정합니다.[1]

**Scaling Factor 변화:**
$$\sigma^2_{\text{FAPO}} - \sigma^2_{\text{GRPO}} = \lambda\gamma(1-\gamma)\left(\lambda - \frac{4}{\alpha/\beta+1}\right)$$

$\alpha/\beta > 4/\lambda - 1$일 때, $\sigma_{\text{FAPO}} > \sigma_{\text{GRPO}}$이 되어 학습이 더욱 안정화됩니다.[1]

***

### 3. 모델 구조 및 인프라 설계

#### 3.1 GenRM 학습 구조

FAPO-GenRM-4B는 Qwen3-4B-Instruct를 기반 모델로 하여 학습됩니다:[1]

**데이터셋:**
- FAPO-Critic-85K: 여러 크기의 LLM(7B~70B)이 생성한 다중 응답
- DAPO-Math-17K: 기반 학습 데이터
- 최종 데이터: $\mathcal{D}\_{\text{FAPO-Critic}} = \{(q_i, r_i, t_i)\}^N_{i=1}$

**학습 전략:**
- Rollout 수: 16개
- Temperature: 1.2 (exploration 촉진)
- Global Batch Size: 512
- Learning Rate: $10^{-6}$

#### 3.2 추론 모델 최적화

FAPO-Reasoning은 GenRM을 활용하여 정책을 최적화합니다:[1]

**기본 모델:**
- AIME24/25: Qwen2.5-Math-7B, 32B
- GPQA-Diamond: 동일 기본 모델

**학습 설정:**
- Rollout 수: 8개 (효율성)
- Temperature: 1.0
- Clipping: $\epsilon_l = 0.2, \epsilon_h = 0.28$

#### 3.3 인프라 설계: 비동기 시스템

FAPO는 GenRM을 RL 파이프라인에 통합할 때 GPU 효율성을 극대화하기 위해 비동기 아키텍처를 채택합니다:[1]

**구조:**
- GenRM Inference를 Rollout Inference 및 Actor Training과 분리
- 여러 서버 워커를 통한 부하 분산
- GPU 대기 시간을 18% 이내로 제한

이는 특히 **Long-Tail 문제**를 완화하여 전체 훈련 시간 증가를 20% 이내로 제한합니다.[1]

***

### 4. 성능 향상 및 한계

#### 4.1 성능 향상 결과

**1) 결함 긍정 감지 성능**

FAPO-GenRM-4B의 성능 비교:[1]

| 평가 지표 | FAPO-GenRM-4B | Qwen3-32B (Teacher) | Qwen2.5-Math-PRM-72B (Discriminative SoTA) |
|---------|---------|---------|---------|
| FlawedPositiveBench F1 | 89.4 | 87.8 | 81.8 |
| ProcessBench Avg F1 | 83.3 | 82.0 | 76.8 |
| Token 사용량 | 1799 | 1868 | 1593 |

4B 모델이 32B 교사 모델과 72B discriminative 모델을 능가합니다.[1]

**2) 추론 성능 향상**

FAPO 적용 시 성능 개선:[1]

- **AIME24**: +4.7% (7B 모델)
- **AIME25**: +3.1% (32B 모델)
- **GPQA-Diamond**: +1.5% (32B 모델)

**3) 프로세스 신뢰성**

결함 긍정 비율 감소:[1]

| 벤치마크 | Baseline | FAPO | 개선 |
|---------|---------|---------|---------|
| AIME24 | 15.5% | 7.1% | -8.4%p |
| AIME25 | 10.9% | 1.7% | -9.2%p |
| GPQA-Diamond | 45.7% | 42.0% | -3.7%p |

**4) 학습 안정성**

- 기준선 대비 매끄러운 학습 곡선
- 후기 학습 단계에서 성능 저하 없음
- Token budget 증가 없이 성능 달성

#### 4.2 한계

**1) 알고리즘 수준 한계**

- **Reward Hacking 위험**: 세분화된 프로세스 보상이 보상 해킹에 취약할 수 있습니다. 예를 들어, 구간 비율 보상(step-ratio reward)은 모델이 높은 신뢰도를 보이는 단계만 출력하고 불확실한 단계를 건너뛰도록 유도하는 "jump-in-reasoning" 현상을 야기합니다.[1]

- **Lambda 값의 고정성**: $\lambda = 1$을 기본값으로 설정했으나, 이는 다양한 작업 도메인에서 최적이 아닐 수 있습니다.[1]

**2) 인프라 수준 한계**

- **완전 비동기 시스템의 미개발**: 현재의 비동기 설계는 여전히 두 단계 반복 구조를 유지하고 있으며, 완전 비동기 RL 시스템으로의 진화는 미래 연구 과제입니다.[1]

**3) 일반화 한계**

- **도메인 특이성**: 주로 수학적 추론(AIME24/25)과 다중선택(GPQA)에서 검증되었습니다.[1]
- **모델 아키텍처 제한**: Qwen/Llama 기반 모델에만 검증되었으며, MoE 또는 초대형 모델(100B+)에서의 효과는 미확인입니다.[1]

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 일반화 능력의 증거

**1) 자체 정정(Self-Correction) 메커니즘의 진화**

FAPO는 학습 초기에 자체 정정을 활용하지만, 점진적으로 완전히 정확한 rollout으로 전환합니다.[1]

$$\text{Rollout Length: Baseline } \approx 1200-1300 \text{ tokens} \rightarrow \text{ FAPO } \approx 1100 \text{ tokens}$$

이는 모델이 더 직접적이고 효율적인 추론으로 진화하고 있음을 시사합니다.[1]

**2) Cross-Domain Generalization**

- AIME (수학)에서 학습한 신뢰할 수 있는 추론 능력이 GPQA (일반 도메인)에 어느 정도 전이됩니다.[1]
- 다중선택 문제(GPQA)에서는 여전히 결함 긍정 비율이 높지만(42%), 기준선 대비 개선이 유지됩니다.[1]

#### 5.2 일반화 향상 가능 영역

**1) Process-Level 신호의 보편성**

GenRM의 구조(프로세스 레벨 페널티)는 다양한 추론 작업에 적용 가능합니다:[1]
- 코드 생성
- 논리 추론
- 다중 선택 문제 해결

**2) Majority-Guided 전략의 강건성**

$\lambda = 1$의 자동 결정 메커니즘은 작업 특성에 독립적으로 작동하도록 설계되었습니다. 이는 하이퍼파라미터 튜닝 없이 다양한 도메인에서의 적용성을 시사합니다.[1]

#### 5.3 일반화 개선의 잠재 경로

**논문 내 제시된 향후 연구 방향:**[1]

1. **다중선택 작업 확대**: GPQA-Diamond에서 결함 긍정 비율이 여전히 높은 점을 개선
2. **멀티턴 상호작용**: 대화형 시나리오에서의 신뢰성 평가
3. **에이전트 기반 RL**: 도구 사용 및 계획 작업으로의 확장

***

### 6. 관련 최신 연구 기반 영향 분석

#### 6.1 FAPO가 현재 연구 생태계에 미치는 영향

**1) Process Reward Model (PRM) 진화**

최근 연구들은 PRM의 효과성을 재검토하고 있습니다:[2][3]

- **Process Advantage Verifier (PAV)**: 기준 정책과 별개의 prover 정책에서 측정한 "진전(progress)"으로 프로세스 보상을 정의하는 새로운 패러다임이 제시되었습니다. FAPO의 "결함 긍정 감지"는 이와 유사하게 더 세밀한 신호를 제공합니다.[2]

- **Generative PRM의 확산**: GenRM 방식이 여러 후속 연구에서 채택되고 있으며, FAPO의 step-level reward 설계는 이 분야의 모범 사례가 되고 있습니다.[3]

**2) Reward Hacking 문제의 재조명**

FAPO는 reward hacking을 명시적으로 논의했습니다. 최근 연구들이 이를 추종하고 있습니다:[4][1]

- **Hierarchical Reward Model (HRM)**: PRM의 reward hacking 문제를 해결하기 위해 개별 및 연속 단계를 모두 평가하는 다층 구조를 제안합니다.[4]

**3) Self-Rewarding 패러다임의 재해석**

Process-based Self-Rewarding은 FAPO의 단계 레벨 평가 개념을 활용하여 자체 학습 능력을 강화했습니다.[5]

#### 6.2 향후 연구 시 고려할 점

**1) 도메인별 최적화 전략**

FAPO의 일반화를 위해서는:[6][1]

- **작업 특성 분석**: 결함 긍정의 분포가 작업 도메인에 따라 다르므로(수학 vs 다중선택), 작업별 맞춤형 $\lambda$ 값 또는 적응형 $\lambda$ 메커니즘 개발이 필요합니다.

- **Multi-Modal 확장**: 최근 video-SALMONN-o1 같은 연구에서 시각적 추론을 통한 PRM 개선을 시도 중이며, FAPO도 멀티모달 도메인으로의 확장을 고려할 필요가 있습니다.[7]

**2) 인프라 스케일링**

FAPO의 실용성을 위해:[8][1]

- **Fully Asynchronous RL**: 완전 비동기 시스템(예: AREAL)의 설계 원리를 FAPO에 통합하여 GPU 효율을 더욱 개선
- **Distributed GenRM**: 초대형 모델(100B+)을 위해 분산 GenRM 추론 최적화

**3) 신뢰성 보장 메커니즘**

향후 연구에서 추가할 사항:[9][4]

- **Interpretability Enhancement**: GenRM의 판정 근거를 더욱 상세히 제공하여 감시 가능성 향상
- **Out-of-Distribution Robustness**: 훈련 분포 밖의 문제에 대한 결함 긍정 감지 능력 강화

**4) 다양한 RL 알고리즘과의 호환성**

FAPO는 현재 GRPO 기반이지만, 최근 연구 동향을 고려하면:[10]

- DPO, IPO, KTO 등 다른 preference optimization 방법과의 통합
- Direct Preference Optimization 방식에서의 FAPO 적용 가능성

**5) Agent-Based RL로의 확장**

최근 Agent PRM 연구가 활발하므로:[3]

- 다단계 에이전트 행동에 대한 결함 긍정 감지
- 도구 사용 및 환경 상호작용 시 신뢰성 평가

***

### 7. 결론 및 종합 평가

FAPO는 **결함이 있는 긍정이 학습의 이중 역할을 한다**는 통찰력에서 출발하여, 이를 동적으로 조절하는 우아한 메커니즘을 제시합니다. 주요 성과는:[1]

1. **이론적 기여**: 학습의 동역학을 수학적으로 분석하고, parameter-free majority-guided 전략으로 자동 최적화 전환을 실현
2. **실용적 기여**: 40% 감소의 결함 긍정 비율과 3-5% 성능 향상을 token budget 증가 없이 달성
3. **방법론적 기여**: Generative Reward Model을 통한 해석 가능한 프로세스 레벨 평가

다만, **도메인 특이성**, **reward hacking 위험**, **완전 자동화 부재** 등의 한계를 인식하고, 향후 멀티모달 도메인, 에이전트 기반 RL, 더욱 견고한 비동기 인프라로의 확장이 필요합니다.[5][2][3][4][1]

특히 현재 LLM 기반 추론의 중심이 **지속적이고 신뢰할 수 있는 추론 능력** 구축으로 이동하는 가운데, FAPO의 결함 긍정 감지 및 적응형 페널티 메커니즘은 향후 RLVR 시스템의 핵심 모듈이 될 가능성이 높습니다.[11][1]

***

### 참고 문헌 (연도별 정렬)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b09cd57f-eb86-45a3-b7b6-f041a1a60305/2510.22543v1.pdf)
[2](https://arxiv.org/abs/2501.19324)
[3](https://arxiv.org/abs/2502.10325)
[4](https://arxiv.org/abs/2505.18761)
[5](https://arxiv.org/abs/2502.14768)
[6](https://arxiv.org/abs/2506.09014)
[7](https://arxiv.org/abs/2502.07191)
[8](https://arxiv.org/abs/2503.02390)
[9](https://arxiv.org/abs/2503.12123)
[10](https://arxiv.org/abs/2501.18858)
[11](https://toloka.ai/blog/reinforcement-learning-with-verifiable-rewards-unlocking-reliable-ai-reasoning/)
[12](https://arxiv.org/abs/2502.11775)
[13](https://arxiv.org/html/2503.21295v1)
[14](https://arxiv.org/html/2412.11006v1)
[15](https://arxiv.org/pdf/2410.08146.pdf)
[16](https://arxiv.org/pdf/2501.07861.pdf)
[17](https://arxiv.org/pdf/2503.13551.pdf)
[18](https://arxiv.org/pdf/2402.00658.pdf)
[19](https://arxiv.org/pdf/2501.18858.pdf)
[20](http://arxiv.org/pdf/2503.03746.pdf)
[21](https://proceedings.iclr.cc/paper_files/paper/2025/hash/98711dea460bdefe0e651ca23ec98ba2-Abstract-Conference.html)
[22](https://www.emergentmind.com/topics/generative-process-reward-model-llm-as-genprm)
[23](https://openreview.net/forum?id=A6Y7AqlzLW)
[24](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
[25](https://openreview.net/forum?id=MwU2SGLKpS)
[26](https://www.youtube.com/watch?v=S2qmzxwV8fg)
[27](https://www.promptfoo.dev/blog/rlvr-explained/)
[28](https://aclanthology.org/2025.acl-long.1297.pdf)
[29](https://arxiv.org/abs/2508.03556)
[30](https://aipapersacademy.com/generative-reward-models/)
[31](https://arxiv.org/abs/2501.07301)
[32](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)
