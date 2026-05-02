
# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
---

## 📌 참고 자료 및 출처

> - **[1] 원문 논문 (arXiv):** DeepSeek-AI, *"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"*, arXiv:2501.12948, 2025. https://arxiv.org/abs/2501.12948
> - **[2] Nature 게재본:** *"DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning"*, Nature, 2025. https://www.nature.com/articles/s41586-025-09422-z
> - **[3] Hugging Face 모델 페이지:** https://huggingface.co/deepseek-ai/DeepSeek-R1
> - **[4] GitHub 공식 저장소:** https://github.com/deepseek-ai/DeepSeek-R1
> - **[5] GRPO 관련 분석 (AI Papers Academy):** https://aipapersacademy.com/deepseekmath-grpo/
> - **[6] GRPO 수식 분석 (arXiv:2503.06639):** *"Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss, Dynamics, and Success Amplification"*
> - **[7] Dr. GRPO 분석 논문 (arXiv:2503.20783, COLM 2025):** *"GRPO Done Right"*
> - **[8] DataCamp 블로그:** https://www.datacamp.com/blog/deepseek-r1
> - **[9] Analytics Vidhya - DeepSeek R1 vs OpenAI o1:** https://www.analyticsvidhya.com/blog/2025/01/deepseek-r1-vs-openai-o1/
> - **[10] Premai 블로그:** *"Reasoning Models Explained: OpenAI o1/o3 vs DeepSeek R1 vs QwQ-32B"*: https://blog.premai.io/reasoning-models-explained-openai-o1-o3-vs-deepseek-r1-vs-qwq-32b/

---

## 1. 핵심 주장과 주요 기여 요약

일반 추론(General Reasoning)은 AI의 오랜 난제이며, LLM과 Chain-of-Thought(CoT) 프롬프팅은 기초적 추론 태스크에서 성과를 거뒀으나, 그 성공이 대규모 인간 어노테이션 데이터에 크게 의존하고, 더 복잡한 문제에서는 여전히 한계를 보였다.

### 핵심 주장

이 논문은 **순수 강화학습(Pure RL)만으로도 LLM의 추론 능력을 유도**할 수 있음을 증명하며, 인간이 레이블링한 추론 궤적(reasoning trajectory)이 불필요함을 보인다. 제안된 RL 프레임워크는 자기 반성(self-reflection), 검증(verification), 동적 전략 적응(dynamic strategy adaptation)과 같은 고급 추론 패턴의 창발적 발전을 촉진하며, 수학, 코딩 대회, STEM 분야에서 기존 지도학습 방식을 능가하는 성능을 달성한다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **DeepSeek-R1-Zero** | SFT 없이 순수 RL만으로 훈련된 최초의 오픈 추론 모델 |
| **DeepSeek-R1** | Cold-start 데이터 + 멀티스테이지 RL로 완성된 최종 모델 |
| **지식 증류 (Distillation)** | 대형 모델의 추론 능력을 소형 모델로 이전 |
| **오픈소스 공개** | 1.5B~70B까지 6개의 증류 모델 공개 |

DeepSeek-R1-Zero는 **자기 검증(self-verification), 반성(reflection), 긴 CoT 생성** 능력을 입증하며, 이는 SFT 없이 순수 RL만으로 LLM의 추론 능력을 유도할 수 있음을 검증한 최초의 오픈 연구이다.

---

## 2. 논문 세부 분석

### 2-1. 해결하고자 하는 문제

LLM과 CoT 프롬프팅은 기초 추론에 성공을 거뒀으나, 이 성공은 대규모 인간 어노테이션 데모에 크게 의존하며, 더 복잡한 문제에서는 모델 역량이 부족하다.

이를 해결하기 위해 **RL 프레임워크 내에서 자기 진화(self-evolution)**를 통해 인간 레이블링 의존도를 최소화하는 LLM의 추론 능력 개발 가능성을 탐색하며, DeepSeek-V3 Base 위에 **GRPO(Group Relative Policy Optimization)**를 RL 프레임워크로 사용한다.

---

### 2-2. 제안하는 방법 및 수식

#### (1) GRPO (Group Relative Policy Optimization)

훈련 비용 절감을 위해 **GRPO**를 채택하는데, 이는 보통 정책 모델과 같은 크기의 크리틱 모델을 사용하지 않고, **그룹 점수로부터 기준선(baseline)을 추정**하는 방식이다.

GRPO는 PPO의 최적화 프레임워크를 따르되, **어드밴티지(advantage) 계산 방식에서 차이**가 있다. 구체적으로 학습된 크리틱 대신 **몬테카를로 롤아웃(Monte Carlo rollouts)**으로 어드밴티지를 추정하며, 어드밴티지 함수에 화이트닝(whitening)을 적용하여 보상의 평균과 분산을 표준화한다.

**GRPO 목적 함수:**

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \min\left(r_i(\theta)\hat{A}_i,\ \text{clip}(r_i(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_i\right) - \beta \cdot \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]\right]
$$

여기서:
- $r_i(\theta) = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$ : 현재 정책과 이전 정책 간의 확률 비율
- $G$ : 그룹 내 샘플링 수 (논문에서는 16)
- $\hat{A}_i$ : 그룹 기반 정규화된 어드밴티지 (보상의 평균/표준편차로 정규화)
- $\varepsilon$ : 클리핑 파라미터 (논문에서 10으로 설정)
- $\beta$ : KL 패널티 계수 (논문에서 0.001로 설정)

**어드밴티지 추정:**

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}
$$

즉, 가치 모델(value model)이 평균값을 결정하는 대신, **샘플링된 응답들로부터 이를 추정**하는 것이다. 이것이 바로 GRPO 이름에서 "그룹 상대적(group relative)"의 의미이며, 어드밴티지는 샘플링된 응답 그룹에 상대적으로 계산된다.

#### (2) 보상 함수 설계

DeepSeek-R1-Zero의 경우 수학, 코딩, 논리 추론 데이터에 **규칙 기반 보상(rule-based reward)**을 사용하며, 크게 **정확도 보상(accuracy reward)**과 **포맷 보상(format reward)** 두 가지 유형으로 구성된다. 정확도 보상은 응답의 정확성을 평가하며, 예를 들어 수학 문제의 경우 특정 형식(예: 박스 안)의 최종 답변을 요구하여 신뢰할 수 있는 규칙 기반 검증을 가능하게 한다.

전체 보상은 다음과 같이 정의된다:

$$
R_{rule} = R_{acc} + R_{format}
$$

신경망 기반 보상 모델(neural reward model)은 의도적으로 사용하지 않았는데, 이는 **대규모 강화학습에서 신경망 보상 모델이 보상 해킹(reward hacking)에 취약**하기 때문이다.

---

### 2-3. 모델 구조

#### 기반 모델 (DeepSeek-V3 Base)

DeepSeek-V3는 **총 671B 파라미터**를 가지며, 토큰당 **37B 파라미터만 활성화**하는 MoE 구조로 효율성과 역량을 최적화한다. **14.8조 개의 고품질 토큰**으로 사전훈련되었으며, **MLA(Multi-head Latent Attention)**, 보조 손실 없는 로드밸런싱 전략, **MTP(Multi-Token Prediction)** 등 혁신적 기법을 포함한다.

#### DeepSeek-R1-Zero 훈련

SFT에 의존하지 않고 **기반 모델에 직접 RL을 적용**하여 복잡한 문제 해결을 위한 CoT 탐색을 허용하며, 이것이 DeepSeek-R1-Zero의 개발로 이어진다.

모델의 사고 과정은 명시적으로 `<think>`와 `</think>` 태그로 구분되며, 이를 통해 **해석 가능성(interpretability)을 향상**하고 후속 분석을 용이하게 한다.

#### DeepSeek-R1 멀티스테이지 훈련 파이프라인

DeepSeek-R1 파이프라인은 **개선된 추론 패턴을 발견하고 인간 선호와 정렬하는 두 개의 RL 단계**, 그리고 **모델의 추론 및 비추론 능력의 시드 역할을 하는 두 개의 SFT 단계**로 구성된다.

4단계 파이프라인:

| 단계 | 내용 |
|---|---|
| **Stage 1: Cold Start SFT** | 소량의 고품질 긴 CoT 데이터로 초기 파인튜닝 |
| **Stage 2: Reasoning-oriented RL** | 수학/코드/STEM에 집중한 RL 훈련 |
| **Stage 3: Rejection Sampling + SFT** | 거부 샘플링 + 비추론 데이터 혼합 파인튜닝 |
| **Stage 4: RL for All Scenarios** | 전체 시나리오에 대한 최종 RL 정렬 |

#### 지식 증류 모델

오픈소스로 공개된 증류 모델은 **1.5B, 7B, 8B, 14B, 32B, 70B** 6가지 규모이며 DeepSeek-R1로부터 증류되었다.

---

### 2-4. 성능 향상

AIME 2024에서 pass@1 점수가 **15.6%에서 71.0%로 향상**되었으며, 다수결 투표(majority voting)를 통해 **86.7%까지 개선**되어 OpenAI-o1-0912와 동등한 수준에 도달하였다.

최종 DeepSeek-R1은 AIME 2024에서 **79.8% Pass@1**로 OpenAI-o1-1217을 소폭 상회하며, MATH-500에서는 **97.3%**로 OpenAI-o1-1217과 동등한 수준이다.

창의적 글쓰기, 일반 QA 등 비추론 태스크에서도 DeepSeek-R1은 AlpacaEval 2.0에서 **87.6%의 length-controlled win-rate**, ArenaHard에서 **92.3%의 win-rate**를 달성하며, 긴 문맥 이해 태스크에서도 DeepSeek-V3를 크게 능가한다.

#### 증류 모델 성능

14B 증류 모델은 오픈소스 QwQ-32B-Preview를 크게 능가하며, 32B 및 70B 증류 모델은 **밀집 모델(dense model) 중 추론 벤치마크 신기록**을 세웠다.

DeepSeek-R1-Distill-Qwen-32B는 다양한 벤치마크에서 **OpenAI-o1-mini를 능가**하여 밀집 모델 중 최고 수준을 달성하였다.

---

### 2-5. 한계

DeepSeek-R1-Zero는 **무한 반복(endless repetition), 낮은 가독성(poor readability), 언어 혼재(language mixing)** 문제에 직면하였다.

DeepSeek-R1은 Chinese SimpleQA 벤치마크에서 DeepSeek-V3보다 낮은 성능을 보이는데, 이는 **Safety RL 이후 특정 쿼리에 대한 거부 경향** 때문이다.

엔지니어링 지향 코딩 태스크에서는 OpenAI-o1-1217이 Aider에서 DeepSeek-R1을 능가하며, 저자들은 **관련 RL 훈련 데이터가 현재 매우 제한적**이기 때문에 다음 버전에서 개선될 것으로 기대한다.

---

## 3. 일반화 성능 향상 가능성

### 3-1. 순수 RL을 통한 일반화

연구 결과는 RL이 어떠한 지도 파인튜닝 데이터 없이도 DeepSeek-R1-Zero가 **견고한 추론 능력을 달성하며 효과적으로 일반화**할 수 있음을 보여준다. 이는 모델이 RL만으로도 학습하고 일반화할 수 있음을 강조하는 주목할 만한 성과이다.

### 3-2. 창발적 추론 패턴

훈련 중 특히 흥미로운 현상으로 **"Aha Moment"**가 관찰되었다. 중간 버전의 모델에서 발생한 이 순간에 DeepSeek-R1-Zero는 초기 접근 방식을 재평가하며 문제에 더 많은 생각 시간을 할당하는 방법을 학습하였다. 이는 모델의 성장하는 추론 능력의 증거이자 강화학습이 예상치 못한 정교한 결과로 이어질 수 있음을 보여주는 매혹적인 사례이다.

### 3-3. 지식 증류를 통한 소형 모델 일반화

더 나아가, 대형 모델에서 나타나는 창발적 추론 패턴을 **소형 모델의 추론 능력 향상에 체계적으로 활용**할 수 있음이 확인되었다.

이 논문은 **대형 모델의 추론 패턴을 소형 모델에 증류하면, 소형 모델에서 직접 RL로 발견한 추론 패턴보다 더 나은 성능**이 달성됨을 입증한다.

Qwen2.5-32B를 기반 모델로 사용할 때, DeepSeek-R1에서 직접 증류하는 것이 RL을 적용하는 것보다 우수하였는데, 이는 **대형 기반 모델이 발견한 추론 패턴이 추론 능력 향상에 결정적**임을 증명한다.

### 3-4. 비추론 태스크로의 일반화

DeepSeek-R1은 STEM 관련 질문에서 대규모 강화학습을 통해 상당한 이득을 얻는 것 외에도, 긴 문맥 의존적 QA 태스크인 FRAMES에서도 탁월한 성능을 보이며 강력한 문서 분석 능력을 보여준다. 이는 AI 기반 검색 및 데이터 분석 태스크에서 추론 모델의 잠재력을 강조한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 모델/연구 | 출처 | 특징 | DeepSeek-R1과의 차이 |
|---|---|---|---|
| **GPT-4 / InstructGPT** | OpenAI, 2022~2023 | RLHF 기반 SFT 의존 | SFT 없는 순수 RL 불가 |
| **Chain-of-Thought Prompting** | Wei et al., 2022 | 프롬프팅으로 추론 유도 | 추론 능력 훈련 내재화 안 됨 |
| **OpenAI o1 / o3** | OpenAI, 2024~2025 | 추론 특화, 비공개 | 클로즈드 소스, 방법론 미공개 |
| **QwQ-32B** | Alibaba/Qwen, 2024 | 소형 추론 모델 | 증류 방식, 독자 RL 미적용 |
| **DeepSeekMath (GRPO)** | DeepSeek, 2024 | GRPO 최초 도입 | 수학 특화 도메인 모델 |
| **Dr. GRPO** | COLM 2025 | GRPO 최적화 편향 수정 | 더 나은 토큰 효율성 제안 |

추론 능력의 맥락에서 **OpenAI의 o1 시리즈 모델이 Chain-of-Thought 추론 과정의 길이를 늘림으로써 추론 시간 스케일링(inference-time scaling)을 최초로 도입**하였으며, 이 접근법은 수학, 코딩, 과학적 추론 등 다양한 추론 태스크에서 상당한 향상을 달성하였다.

2025년 1월 DeepSeek가 OpenAI o1과 동등한 성능의 오픈 웨이트 추론 모델 R1을 훨씬 낮은 비용으로 공개하였으며, 이후 Alibaba의 QwQ-32B는 32B 파라미터 모델이 20배 큰 모델과 경쟁할 수 있음을 보였다. OpenAI는 o3와 o4-mini로 대응하며 벤치마크를 더욱 향상시켰다.

후속 연구(Dr. GRPO)는 **GRPO의 최적화 편향(optimization bias)**이 점진적으로 더 긴 잘못된 응답을 생성할 수 있음을 밝히고, 이를 제거하는 간단한 수정을 제안하여 더 나은 토큰 효율성을 달성하였다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려사항

### 5-1. 연구에 미치는 영향

이 논문은 **SFT 없이 순수 RL만으로 LLM의 추론 능력이 유도될 수 있음을 검증한 최초의 오픈 연구**로서, 이 돌파구는 해당 분야의 미래 발전을 위한 길을 열어준다.

**① 인간 어노테이션 의존도 감소 패러다임 전환**

추론이 강화만으로 창발될 수 있다면, **레이블링 가능한 예제 수에 더 이상 제한되지 않으며, 오직 성공을 얼마나 잘 정의하느냐에만 제한**된다. 이는 AI 훈련 패러다임의 근본적인 전환을 의미한다.

**② 소형 모델 민주화**

더 적은 에너지 비용으로 강력한 AI에 대한 더 넓은 접근을 가능하게 하기 위해 여러 소형 모델을 증류하여 공개 공개하였으며, 이 증류 모델들은 강력한 추론 능력을 보여주고 기존 instruction-tuned 모델들의 성능을 능가한다.

**③ 비용 효율성**

DeepSeek는 입력 토큰 기준으로 OpenAI보다 **27배 저렴**하며, 캐시 시 58배 저렴하다. 백만 토큰당 $0.55 ($0.13 캐시)로, OpenAI의 $15.00($7.50 캐시)에 비해 훨씬 낮은 가격으로 다양한 새로운 사용 사례를 가능하게 한다.

---

### 5-2. 앞으로 연구 시 고려할 점

**① 보상 해킹(Reward Hacking) 방지**

검증 가능한 보상(verifiable reward)은 단순성과 편향 사이의 균형을 맞추며, 선호 데이터로 학습된 보상 모델보다 보상 해킹에 덜 취약한 것으로 여겨진다. 보상 해킹은 정책이 보상을 과최적화하여 모델 품질이 저하되는 강화학습의 일반적인 문제이다.

**② 비추론 태스크로의 확장**

사용자 친화적 모델 훈련, 즉 명확하고 일관된 CoT를 생성할 뿐만 아니라 **강력한 일반 능력도 갖춘 모델**을 어떻게 훈련할 것인지에 대한 질문은 여전히 중요한 연구 과제이다. 이를 해결하기 위해 4단계 파이프라인이 설계되었지만, 일반 능력의 추가 향상은 여전히 탐구 여지가 있다.

**③ 언어 일관성 및 가독성**

DeepSeek-R1-Zero가 직면한 **무한 반복, 낮은 가독성, 언어 혼재** 문제를 해결하기 위해 RL 전에 cold-start 데이터를 도입하는 방식이 사용되었으며, 이러한 출력 품질 문제는 향후 연구에서 지속적으로 다뤄져야 한다.

**④ 소형 모델에서의 직접 RL 효율화**

DeepSeek-R1에서 증류된 DeepSeek-R1-Distill-Qwen-32B는 모든 벤치마크에서 DeepSeek-R1-Zero-Qwen-32B를 크게 능가하였는데, 이는 **소형 모델에서의 RL 훈련 효율 개선이 중요한 연구 과제**임을 시사한다.

**⑤ 추론 과정의 평가 방법론 개선**

정확도만으로 평가하는 것은 이제 충분하지 않으며, **모델의 의사결정 과정을 포착하는 새로운 평가 방법**이 필요하다. 이는 단계별 추론 분석(step-by-step reasoning analysis)과 신뢰도 교정(confidence calibration) 등을 포함할 수 있다.

**⑥ 멀티모달 및 도메인 특화 확장**

DeepSeek-R1은 추론 집약적 태스크 스케일링의 병목을 해소하고 **교육 및 의료 분야**와 같이 자원 제약이 혁신을 방해하는 분야에서 새로운 기회를 열어준다.

---

## 요약 정리

```
DeepSeek-R1의 핵심 기여
├── 이론적 기여: 순수 RL만으로 추론 능력 창발 가능 입증
├── 방법론적 기여: GRPO + 멀티스테이지 파이프라인
├── 성능적 기여: OpenAI-o1-1217 동등 수준 달성 (오픈소스)
└── 실용적 기여: 1.5B~70B 증류 모델 MIT 라이선스 공개

향후 연구 방향
├── 더 효율적인 RL 알고리즘 개발 (Dr. GRPO 등)
├── 비검증 가능 태스크로의 보상 신호 확장
├── 언어 일관성 및 안전성 강화
├── 소형 모델에서의 RL 직접 훈련 효율화
└── 새로운 추론 평가 방법론 개발
```
