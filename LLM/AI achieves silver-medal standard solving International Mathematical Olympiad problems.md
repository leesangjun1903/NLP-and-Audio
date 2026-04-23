
# AI achieves silver-medal standard solving International Mathematical Olympiad problems

> **주요 출처 / 참고자료**
> - Google DeepMind 공식 블로그: *"AI achieves silver-medal standard solving International Mathematical Olympiad problems"* (2024.07.25) — https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/
> - Hubert, T., Mehta, R., Sartran, L. et al. *"Olympiad-level formal mathematical reasoning with reinforcement learning."* **Nature** (2025.11.12) — https://www.nature.com/articles/s41586-025-09833-y
> - Chervonyi, Y. et al. *"Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2."* arXiv:2502.03544 (2025) — https://arxiv.org/abs/2502.03544
> - Nature Portfolio Press Release: *"A mathematics medal-worthy AI system"* — https://www.natureasia.com/en/info/press-releases/detail/9147
> - Scientific American: *"AI Reaches Silver-Medal Level at This Year's Math Olympiad"* (2024.08) — https://www.scientificamerican.com/article/ai-reaches-silver-medal-level-at-this-years-math-olympiad/
> - LessWrong linkpost + Codeforces discussion (2024.07.25)
> - arxiv: *"Formal Mathematical Reasoning: A New Frontier in AI"* (arXiv:2412.16075)
> - Meta AI Blog: *"Teaching AI advanced mathematical reasoning"* (HyperTree Proof Search, NeurIPS 2022)
> - Google DeepMind 블로그: *"Advanced version of Gemini with Deep Think officially achieves gold-medal standard at the IMO"* (2025.07.21)

---

## 1. 🔑 핵심 주장 및 주요 기여 요약

Google DeepMind는 공식 수학 추론을 위한 강화학습 기반의 새로운 시스템인 **AlphaProof**와 개선된 기하학 문제 풀이 시스템인 **AlphaGeometry 2**를 공동으로 제안하였으며, 이 두 시스템이 함께 2024년 IMO(국제수학올림피아드) 6개 문제 중 4개를 풀어냄으로써 처음으로 은메달리스트 수준을 달성하였다.

시스템의 최종 점수는 **28점**(총 42점)으로, 각 풀이에 만점을 획득하여 은메달 구간의 상위 수준에 해당한다.

AlphaProof는 두 개의 대수 문제와 한 개의 정수론 문제를 풀었으며, 그 중에는 2024년 IMO에서 단 5명의 참가자만이 풀어낸 가장 어려운 문제도 포함되어 있다. AlphaGeometry 2는 기하학 문제를 증명하였으며, 두 개의 조합론 문제는 미해결로 남았다.

### 주요 기여 (Contributions) 요약표

| 항목 | 내용 |
|------|------|
| 시스템 | AlphaProof + AlphaGeometry 2 |
| 성과 | IMO 2024에서 28/42점 (은메달 상위권) |
| 핵심 방법론 | 형식 언어(Lean) 기반 RL + AlphaZero 알고리즘 |
| 의의 | AI 최초의 IMO 메달 수준 달성 |

---

## 2. 📐 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

현재 AI 시스템들은 추론 능력과 훈련 데이터의 한계로 인해 일반적인 수학 문제를 푸는 데 여전히 어려움을 겪고 있다.

일부 대형 언어 모델(LLM)은 유망한 능력을 보여주지만, 비공식적인 자연어 텍스트로 훈련되고 운영되기 때문에 추론의 정확성을 검증하기 어렵다는 문제가 있었다. Google DeepMind 연구진은 형식 수학 소프트웨어 환경(Lean)에서 강화학습을 적용하여, 추론을 자동으로 검증할 수 있는 증명을 생성하는 방법을 제시함으로써 이 문제를 극복하고자 하였다.

형식 언어는 수학적 추론을 포함하는 증명이 올바른지 공식적으로 검증될 수 있다는 결정적인 장점이 있다. 그러나 기계학습에서의 활용은 인간이 작성한 데이터의 양이 매우 제한적이라는 제약을 받아왔다. 반면, 자연어 기반 접근법은 훨씬 더 많은 데이터를 사용할 수 있지만, 그럴듯하지만 올바르지 않은 중간 추론 단계와 답을 환각(hallucination)할 수 있다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### 🔷 AlphaProof: RL 기반 형식 수학 증명 시스템

형식 언어인 Lean은 추론을 기반으로 하는 대화형 환경을 제공하며, 강화학습(RL)은 이러한 환경에서 학습할 수 있는 메커니즘을 제공한다. AlphaProof는 AlphaZero에서 영감을 받은 에이전트로, 수백만 개의 자동 형식화된 문제로 훈련하여 RL을 통해 형식 증명을 찾는 방법을 학습한다.

문제가 주어지면 AlphaProof는 해답 후보를 생성한 뒤, Lean 내의 가능한 증명 단계들을 탐색하여 이를 증명하거나 반증한다. 발견되고 검증된 각 증명은 AlphaProof의 언어 모델을 강화하는 데 사용되어, 그 이후 더 어려운 문제들을 풀어낼 능력을 향상시킨다.

AlphaProof의 핵심 훈련 루프를 수식으로 표현하면 다음과 같다:

**① 자동 형식화(Auto-Formalization):**

비공식 수학 문제 $x_{\text{informal}}$을 형식 언어 Lean 명제 $x_{\text{formal}}$로 변환하는 형식화 네트워크:

$$f_\theta: x_{\text{informal}} \rightarrow x_{\text{formal}} \in \mathcal{L}_{\text{Lean}}$$

AlphaProof는 약 8,000만 개의 명제를 자동 형식화하여 훈련 데이터로 활용한다.

**② AlphaZero 스타일의 강화학습:**

증명 탐색은 현재 증명 상태 $s_t$ (Lean tactic state)에서 다음 tactic(행동) $a_t$를 선택하는 정책 $\pi_\theta(a_t | s_t)$와 가치 함수 $V_\phi(s_t)$로 구성된다:

$$\pi_\theta, V_\phi = \arg\max_{\theta, \phi} \mathbb{E}\left[\sum_{t=0}^{T} r_t\right]$$

여기서 보상 $r_t$는 증명 완성 여부(이진 신호):

$$r_t = \begin{cases} +1 & \text{if proof is completed and verified by Lean} \\ 0 & \text{otherwise} \end{cases}$$

**③ Test-Time Reinforcement Learning (TTRL):**

가장 어려운 문제들에 대해 AlphaProof는 **Test-Time RL**을 사용하는데, 이는 추론 시간(inference time)에 수백만 개의 관련 문제 변형들을 생성하고 학습하여 깊은 문제별 적응을 가능하게 하는 방법이다.

AlphaProof는 대회 전 수 주 동안 다양한 난이도와 수학 주제 분야를 아우르는 수백만 개의 문제를 증명하거나 반증함으로써 IMO를 위해 훈련되었으며, 대회 중에도 훈련 루프가 적용되어 완전한 풀이를 찾을 때까지 대회 문제들의 자체 생성된 변형들의 증명을 강화하였다.

TTRL의 핵심 아이디어를 수식으로 표현하면:

$$\theta_{\text{TTRL}} = \theta_0 + \Delta\theta_{\text{self-play}}(P_{\text{contest}})$$

대회 문제 $P_{\text{contest}}$의 변형 문제들 $\{P_{\text{contest}}^{(i)}\}_{i=1}^{N}$을 자동 생성하고, 이를 풀면서 얻은 RL 신호로 모델 파라미터를 실시간 업데이트한다.

TTRL 단계에서 자동 생성된 Lean 변형 문제들은 단순화, 보조 정리 제안, 증명 단계, 재형식화, 유추 탐색 등 다양한 문제 해결 휴리스틱을 예시한다. 이러한 변형들과의 상호작용은 목표 명제에 대한 통찰력을 제공하고 증명 에이전트의 문제별 적응을 촉진한다.

---

#### 🔷 AlphaGeometry 2: 신경-기호 하이브리드 기하학 시스템

AlphaGeometry 2는 AlphaGeometry의 크게 향상된 버전으로, Gemini 기반의 언어 모델을 사용하는 **신경-기호 하이브리드 시스템(neuro-symbolic hybrid system)**이며 전작 대비 10배 이상의 합성 데이터로 처음부터 새로 훈련되었다.

AlphaGeometry 언어를 확장하여 객체의 이동을 포함하는 문제와 각도, 비율, 거리의 선형 방정식을 포함하는 문제를 다룰 수 있게 하였으며, 이를 통해 IMO 2000-2024 기하학 문제에 대한 AlphaGeometry 언어의 커버리지를 66%에서 88%로 크게 향상시켰다.

AG2의 탐색 과정도 더 나은 언어 모델링을 위한 Gemini 아키텍처의 사용과 탐색 트리들 간의 효과적인 통신을 가능하게 하는 새로운 **지식 공유 메커니즘(knowledge-sharing mechanism)**을 통해 크게 개선되었다.

---

### 2-3. 모델 구조

#### AlphaProof 구조:

AlphaProof는 Lean 정리 증명기 기반의 검증 가능한 환경과 상호작용하여 형식적 수학 증명을 발견하도록 설계된 RL 에이전트로, 아키텍처·훈련·추론이 여러 핵심적인 혁신들을 통합하고 있다.

```
[AlphaProof 전체 파이프라인]

비공식 수학 문제 (자연어)
         ↓
   형식화 네트워크 (Formalizer, Gemini 기반 LLM)
         ↓
   Lean 형식 명제 (formal statement)
         ↓
┌────────────────────────────────┐
│       AlphaZero RL 루프        │
│  정책 네트워크: π(a|s)        │
│  가치 네트워크: V(s)           │
│  탐색: MCTS / Best-first       │
│  환경: Lean tactic verifier    │
└────────────────────────────────┘
         ↓
   증명 성공 시 → 훈련 데이터로 재활용
   (Self-play, Expert Iteration)
         ↓
   Test-Time RL (경쟁 중 실시간 자기강화)
```

AlphaProof의 핵심 시스템은 딥러닝 모델이 각 증명 트리 노드에서 증명 단계(tactic)를 제안하는 **신경 정리 증명(Neural Theorem Proving)**을 구현하며, 이는 최선 우선 탐색(best-first)과 MCTS 같은 기호적 증명 탐색 알고리즘과 결합되어 후보 증명을 조립한다.

#### AlphaGeometry 2 구조:

AlphaGeometry 2의 탐색 과정은 더 나은 언어 모델링을 위한 Gemini 아키텍처 활용과, 여러 탐색 트리를 결합하는 새로운 지식 공유 메커니즘을 통해 크게 개선되었다.

```
[AlphaGeometry 2 구조]

자연어 기하학 문제
         ↓
   자동화된 다이어그램 생성 알고리즘
         ↓
   Gemini LLM (자연어 → AG 형식 언어)
         ↓
┌────────────────────────────────┐
│     신경-기호 하이브리드       │
│  LM: 보조 점 제안 (예: 점 E)  │
│  Symbolic Engine: 추론 규칙    │
│  Knowledge Sharing: 다중 탐색  │
└────────────────────────────────┘
         ↓
   형식 증명 (AG Domain Language)
```

이 프레임워크는 언어 모델이 문제 풀이에 도움이 될 수 있는 이른바 **보조 작도(auxiliary constructions)**를 때로 활용하는 것과 결합되어 있으며, 예컨대 삼각형에 점을 추가하여 사각형으로 만들 수 있다.

---

### 2-4. 성능 향상

| 지표 | AlphaGeometry (v1) | AlphaGeometry 2 |
|------|-------------------|-----------------|
| IMO 기하학 문제 풀이율 (2000-2024) | 53% | **83%** |
| AG 언어 커버리지 (2000-2024 IMO) | 66% | **88%** |
| 전체 풀이율 | - | **84%** |
| IMO 2024 P4 풀이 시간 | - | **19초** |

이러한 향상들이 집약되어 AlphaGeometry 2는 2000-2024년의 모든 IMO 기하학 문제에서 **84%의 풀이율**을 달성하였으며, 이는 AI의 도전적인 수학적 추론 과제 처리 능력에서의 중대한 도약을 나타내며 평균적인 IMO 금메달리스트를 능가한다.

또한 AlphaProof는 역대 수학 경시대회 문제에서 이전 최신 AI 시스템들의 결과를 뛰어넘음이 확인되었다.

---

### 2-5. 한계 (Limitations)

인간 참가자들에게 9시간의 제한이 주어지는 반면, DeepMind의 AI는 특히 어려운 문제 하나를 풀기 위해 3일이 걸렸다. 이는 인간 경쟁자들보다 훨씬 긴 시간이 필요하다는 중대한 한계이다.

현재 AlphaProof는 인간이 먼저 문제를 Lean 형식 표현으로 번역해야 하며, 자연어 문제를 스스로 이해하지 못한다. 이는 인간 수학자처럼 새 문제를 독자적으로 제안하거나, 어떤 문제가 연구할 가치가 있는지 판단할 수 없음을 의미한다.

두 개의 조합론(combinatorics) 문제는 여전히 미해결로 남아 있다.

이러한 유형의 문제는 종종 AlphaProof가 훈련 중 '본 적 없는' 범위 밖의 고도로 비구조화된 창의적 사고를 필요로 하며, 따라서 새롭고 미지의 어려운 문제를 다룰 수 있도록 AI를 더 일반적이고 적응력 있게 만드는 것이 다음 단계의 중요한 과제이다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

이 섹션은 본 연구에서 가장 중요한 미래 지향적 주제이다.

### 3-1. 현재의 일반화 구조: 합성 데이터 + 자기강화

수학적 공리는 원칙적으로 무한한 잠재적 데이터를 포함하고 있으며, 이는 도메인 내 모든 증명 가능한 사실을 함축하고 있기 때문이다. 만약 일반화될 수 있다면, 이 접근법의 중요한 이점은 비공식 데이터(예: 연구 논문)조차 희소한 완전히 새로운 수학 분야에도 적용 가능하다는 것이다. 합성 데이터를 생성함으로써, AI 시스템은 인간이 생성하는 훈련 데이터의 속도를 훨씬 능가하는 규모로 가능한 수학적 문제와 풀이의 방대한 공간을 탐색하고 학습할 수 있다.

이를 수식으로 표현하면:

합성 데이터 공간 $\mathcal{D}_{\text{synth}}$는 수학 공리 집합 $\mathcal{A}$로부터 도출 가능한 모든 정리들의 집합이므로:

$$|\mathcal{D}_{\text{synth}}| = |\{T : \mathcal{A} \vdash T\}| \approx \infty$$

이는 인간이 수기로 작성한 데이터 $\mathcal{D}_{\text{human}}$보다 사실상 무한히 크며, 일반화의 기반이 된다.

### 3-2. Test-Time RL을 통한 문제별 적응 (일반화의 핵심)

AlphaProof는 수백만 개의 자동 형식화된 문제로 RL 훈련을 통해 형식 증명을 찾는 방법을 학습하는 AlphaZero에서 영감을 받은 에이전트이며, 가장 어려운 문제들의 경우 추론 시간에 수백만 개의 관련 문제 변형을 생성하고 학습하는 **Test-Time RL**을 사용하여 깊은 문제별 적응(problem-specific adaptation)을 가능하게 한다.

이 방법의 수식적 의미:

기존 LLM은 훈련 파라미터 $\theta_{\text{pretrain}}$이 추론 시 고정되지만, AlphaProof의 TTRL은:

$$\theta_t \leftarrow \theta_{t-1} + \alpha \nabla_\theta \mathbb{E}_{P \sim \text{Variants}(P_{\text{test}})}\left[R(P, \theta_{t-1})\right]$$

즉, 테스트 시점의 문제 $P_{\text{test}}$로부터 관련 변형 문제들을 자동 생성하여 온라인 RL로 파라미터를 업데이트하는 방식이다. 이는 **out-of-distribution 문제에 대한 적응적 일반화**를 가능하게 한다.

### 3-3. AlphaGeometry 2의 언어 커버리지 확장 → 일반화 향상

AlphaGeometry 언어를 객체의 이동, 각도·비율·거리의 선형 방정식을 포함하는 문제까지 확장함으로써, IMO 2000-2024 기하학 문제에 대한 커버리지가 66%에서 88%로 향상되었다.

AlphaGeometry와 유사한 신경-기호 시스템들의 주요 약점 중 하나는 자연어 입력 문제를 도메인 특화 언어로 수동으로 변환해야 한다는 점이며, 이를 해결하기 위해 자연어 입력으로 기하학 문제를 직접 풀 수 있는 완전히 자동화된 시스템 구축에 진전을 이루었으며, Gemini를 활용하여 문제를 자연어에서 AlphaGeometry 언어로 번역하고 새로운 자동화된 다이어그램 생성 알고리즘을 구현하였다.

### 3-4. 자연어 추론 시스템과의 결합 가능성

IMO 작업의 일환으로 Gemini와 최신 연구를 기반으로 한 자연어 추론 시스템도 실험되었으며, 이 시스템은 문제를 형식 언어로 번역할 필요가 없어 다른 AI 시스템과 결합될 수 있다.

자연어 능숙함과 형식 언어에서의 검증된 추론을 포함한 엄밀한 추론을 결합한 에이전트가 수학자, 과학자, 엔지니어, 연구자들에게 더 복잡하고 고급 수학을 풀 수 있도록 도와주는 귀중한 도구가 될 것으로 기대된다.

---

## 4. 🔭 앞으로의 연구에 미치는 영향 및 고려사항

### 4-1. 앞으로의 연구에 미치는 영향

AI의 형식 수학적 추론에 대한 새로운 기회들은 연구 활동의 급증으로 이어지고 있으며, 최근 조사에 따르면 이 분야의 논문 수가 2023년에 거의 두 배로 증가하였으며 2024년에도 다시 두 배가 될 것으로 예상된다.

AlphaProof는 형식 수학적 추론을 AI의 새로운 프론티어로 확립하며, 기계학습·기호 추론·형식 검증의 강점을 결합한다. 이 설계는 데이터 부족, 검증 가능성, 확장 가능한 발견이라는 핵심 과제를 다루며, 미래의 자율적이고 신뢰할 수 있으며 재현 가능한 형식 수학의 기준을 설정한다.

이 분야의 발전은 공식 검증(formal verification)에도 즉각적인 응용을 가져오며, 이는 핵심적인 컴퓨터 과학 문제다. 공식 검증은 소프트웨어 및 하드웨어 시스템을 매우 견고하고 안전하게 만들 수 있지만, 역사적으로 가장 안전이 중요한 응용을 제외하고는 배치하기에 너무 비용이 많이 들었다.

AlphaProof가 더 나아가고 더 일반적으로 될 수 있는 아이디어들이 아직 많이 남아 있으며, 이제 LLM을 위해서도 탐색을 포함한 RL 원칙이 잘 작동함이 증명되었다.

### 4-2. 후속 연구에서 고려할 점

#### (1) 자동 형식화(Autoformalization)의 고도화
현재 AlphaProof는 인간이 문제를 Lean 형식 표현으로 먼저 번역해야 한다는 근본적 한계가 있다. 자연어에서 형식 언어로의 완전 자동 변환 파이프라인을 구축하는 것이 핵심 과제이다.

#### (2) 조합론·미지 분야로의 확장
AlphaProof와 유사한 시스템들이 직면한 열린 연구 과제로, 인간이 작성한 고급 형식 증명(예: 연구 수준 수학)의 데이터 부족 문제가 지속되고 있다. 특히 조합론 문제와 같이 탐색 공간이 폭발적으로 증가하는 문제 유형에 대한 해결 전략이 필요하다.

#### (3) 계산 효율성 (추론 속도) 개선
AlphaProof의 솔루션에 필요한 며칠에 걸친 계산 노력은 인간 참가자들이 직면하는 시간 제약을 훨씬 초과하며, 이를 인간 경쟁 수준의 시간으로 줄이는 것이 중요한 연구 방향이다.

#### (4) 형식-비형식 하이브리드 접근 강화
자연어 기반의 Gemini 접근법 외에도 형식 시스템인 AlphaGeometry와 AlphaProof에 대한 지속적인 발전도 이루어지고 있으며, 자연어 능숙함과 형식 언어에서의 검증된 추론을 결합한 에이전트가 인류 지식의 발전에 없어서는 안 될 도구가 될 것이라고 믿는다.

#### (5) 벤치마크 오염 및 평가 강건성
기존 벤치마크들은 최종 수치적 결과를 강조하는 경향이 있어, 중간 추론 단계의 엄밀성을 간과하고 벤치마크 포화 및 데이터 오염 문제가 발생한다. 이러한 제한된 범위는 복잡한 개념적 이해나 복잡한 논증 구성 능력을 충분히 평가하지 못할 수 있다.

---

## 5. 📊 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 기관 | 방법론 | IMO 성과 | 형식화 |
|------|------|------|--------|----------|--------|
| **GPT-f** | 2020 | OpenAI | LM + Expert Iteration, Metamath | IMO 문제 증명 불가 | Metamath |
| **HTPS (HyperTree Proof Search)** | 2022 | Meta AI | MCTS + Transformer, Expert Iteration | IMO 10문제 풀이 | Lean/Metamath |
| **Minerva** | 2022 | Google | 대규모 수학 데이터 사전훈련 + CoT | MATH 50.3% | 비형식 |
| **AlphaGeometry 1** | 2024.01 | DeepMind | Neuro-symbolic + 합성데이터 | IMO 기하 53% | AG 언어 |
| **AlphaProof + AlphaGeometry 2** | 2024.07 | DeepMind | AlphaZero RL + Lean + TTRL | **IMO 은메달 (28/42점)** | Lean |
| **DeepSeek-Prover** | 2024 | DeepSeek | LLM + Lean 형식 증명 | miniF2F 향상 | Lean |
| **AlphaGeometry 2 (독립 논문)** | 2025.02 | DeepMind | Gemini + 지식공유 탐색 | **IMO 기하 84% (금메달 초과)** | AG + NL |
| **Gemini Deep Think** | 2025.07 | DeepMind | 자연어 RL + 병렬 사고 | **IMO 금메달 (35/42점)** | 자연어 |

Meta AI의 HTPS(HyperTree Proof Search)는 성공적인 수학 증명 데이터셋에서 훈련되어 완전히 새롭고 매우 다른 종류의 문제로 일반화하는 방법을 학습하였으며, 일부 산술적 축약을 포함하는 IMO 문제에 대한 올바른 증명을 도출해낼 수 있었다.

HTPS는 이전 증명 탐색에서 온라인 훈련을 통해 학습하여 훈련 분포에서 먼 도메인까지 일반화할 수 있으며, HTPS만으로도 어노테이션된 증명에서 훈련된 모델이 Metamath 정리의 held-out 집합에서 65.4%를 증명하여 GPT-f의 최고 성능인 56.5%를 크게 뛰어넘었으며, 미증명 정리에 대한 온라인 훈련으로 정확도가 82.6%까지 향상되었다.

2025년 Gemini Deep Think 모델은 공식 문제 설명에서 직접 엄밀한 수학적 증명을 생성하여 4.5시간의 경쟁 제한 시간 내에 자연어로 end-to-end로 운영하였다. 이는 AlphaProof가 Lean 형식화와 3일의 계산이 필요했던 것과 비교하면 극적인 발전이다.

최신 성과로는 Gemini Deep Think의 고급 버전이 6개 IMO 문제 중 5개를 완벽하게 풀어 총 35점을 획득하며 금메달 수준의 성능을 달성하였다.

---

## 🎯 종합 결론

AlphaProof와 AlphaGeometry 2의 결합은 AI 수학 추론의 역사적 전환점이다. 핵심은 **형식 언어의 검증 가능성**과 **강화학습의 자기강화 능력**을 결합한 것이며, 특히 Test-Time RL을 통한 문제별 실시간 적응이 일반화 성능 향상의 핵심 열쇠이다. 향후 연구는 (1) 완전 자동화된 자연어-형식어 번역, (2) 조합론과 같은 미해결 분야 정복, (3) 추론 속도의 인간 수준 달성, (4) 형식-비형식 하이브리드 통합을 중심으로 전개될 것이며, 이는 단순한 경시대회를 넘어 실제 수학 연구와 소프트웨어 검증에 혁명을 가져올 가능성을 열고 있다.
