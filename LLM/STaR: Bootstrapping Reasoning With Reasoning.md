# STaR: Bootstrapping Reasoning With Reasoning

## 1. 핵심 주장과 주요 기여 요약

STaR(Self-Taught Reasoner)은 **소량의 few-shot rationale 예시만으로 대규모 추론 데이터셋을 부트스트랩(bootstrap)하여 언어 모델이 스스로 더 나은 추론 능력을 학습하도록 만드는** 자가 학습 프레임워크입니다. 핵심 통찰은 "모델이 생성한 추론 중 정답으로 이어진 것들만 선별하여 다시 학습 데이터로 활용한다"는 단순한 루프이며, 여기에 **rationalization(합리화)** 라는 보조 메커니즘을 추가하여 모델이 풀지 못한 문제도 정답을 힌트로 제공한 뒤 사후적으로 추론을 생성하도록 합니다.

저자들이 제시한 네 가지 주요 기여는 다음과 같습니다. 첫째, 새로 생성된 rationale의 정확성을 별도로 검증할 필요 없이 정답 일치 여부만으로 필터링하는 부트스트래핑 메커니즘을 제안했습니다. 둘째, rationalization을 도입하여 모델이 실패한 문제에서도 학습 신호를 얻도록 했습니다. 셋째, 산술·상식 추론 등 다양한 도메인에서 광범위한 ablation을 수행했습니다. 넷째, **사전학습된 LLM이 자신의 언어 모델링 능력을 반복적으로 활용해 스스로를 개선하는 최초의 기법** 중 하나를 제시했다고 주장합니다.

대표 성능: CommonsenseQA에서 GPT-J(6B)에 STaR을 적용한 결과 72.5%를 달성하여, few-shot 베이스라인 대비 +35.9%, 직접 정답을 예측하도록 fine-tune된 베이스라인 대비 +12.5% 향상되었으며, 30배 큰 GPT-3(73.0%)에 필적하는 성능을 보였습니다.

---

## 2. 문제·방법(수식 포함)·구조·성능·한계의 상세 설명

### 2.1 해결하고자 하는 문제

기존에 LLM에 추론 능력을 주입하는 방법은 두 가지였고, 두 가지 모두 심각한 단점이 있었습니다. 하나는 **수만 건의 rationale을 수작업/템플릿으로 구축**하는 방식인데, 이는 비용이 비싸고 일반화도 어렵습니다(Rajani et al., 2019; Nye et al., 2021). 다른 하나는 **few-shot in-context learning** 으로 chain-of-thought(CoT)을 유도하는 방식인데(Wei et al., 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"), few-shot CoT는 직접 정답을 예측하도록 fine-tune된 모델에 비해 성능이 크게 떨어집니다. STaR은 이 두 접근 방식의 중간 지점, 즉 **소수의 rationale 예시 + 대량의 정답만 있는 데이터셋** 으로부터 추론 능력을 자체 부트스트랩하는 것을 목표로 합니다.

### 2.2 방법: Rationale Generation Bootstrapping

학습 데이터셋을 $D = \{(x_i, y_i)\}\_{i=1}^{D}$, rationale을 포함한 few-shot 프롬프트 셋을 $P = \{(x_i^p, r_i^p, y_i^p)\}_{i=1}^{P}$ ( $P \ll D$, 예: $P=10$ )이라 하면, 각 예시에 대해 다음과 같이 프롬프트를 구성합니다.

$$\tilde{x}_i = (x_1^p, r_1^p, y_1^p, \dots, x_P^p, r_P^p, y_P^p, x_i)$$

모델은 이 프롬프트를 입력받아 $(\hat{r}_i, \hat{y}_i)$를 생성합니다. 그 후 **정답이 일치하는 경우에만 fine-tuning 데이터로 채택** 하는 필터링을 거칩니다.

$$D_n = \{(x_i, \hat{r}_i, y_i) \mid \hat{y}_i = y_i\}$$

이 데이터로 원래 사전학습 모델 $M$을 다시 fine-tune하여 $M_n$을 얻고, 이 과정을 반복합니다(과적합 방지를 위해 매 외부 루프마다 누적 학습이 아닌 원본 모델에서 다시 시작합니다).

### 2.3 RL 정책 그래디언트와의 등가성

이 절차는 RL 정책 그래디언트의 근사로 해석됩니다. 모델 $M$을 잠재변수 $r$을 갖는 모델 $p_M(y \mid x) = \sum_r p(r \mid x)\, p(y \mid x, r)$로 보면, indicator reward $\mathbb{1}(\hat{y}=y)$ 하에서 데이터셋 전체 기대 보상은

$$J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot \mid x_i)} \mathbb{1}(\hat{y}_i = y_i)$$

이고, 정책 그래디언트는 log-derivative trick으로

$$\nabla J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot \mid x_i)} \big[\mathbb{1}(\hat{y}_i = y_i) \cdot \nabla \log p_M(\hat{y}_i, \hat{r}_i \mid x_i)\big]$$

가 됩니다. indicator가 0인 샘플(정답 불일치)의 그래디언트가 자동으로 버려지는 것이 곧 STaR의 필터링 단계와 동치이고, STaR은 분산을 줄이기 위해 (1) greedy decoding으로 샘플링하고 (2) 동일 배치에 대해 여러 그래디언트 step을 밟는 정책 그래디언트류 알고리즘의 단순화된 형태로 볼 수 있습니다.

### 2.4 Rationalization: 실패 문제로부터의 학습 신호 확보

위 루프의 본질적 한계는 **모델이 실패하는 문제로부터는 학습 신호를 받지 못한다는 점** 입니다. 저자들은 이를 해결하기 위해 정답을 힌트로 제공하여 거꾸로 rationale을 생성하게 합니다. 이는 모델 분포 $p(r \mid x)$ 에서 샘플링하던 것을 $p(r \mid x, y)$ 라는 더 좋은 탐색 공간에서 샘플링하도록 바꾸는 것과 같으며, 본문 식 (1)의 off-policy 추정으로 해석됩니다.

$$D_n^{\text{rat}} = \{(x_i, \hat{r}_i^{\text{rat}}, y_i) \mid \hat{y}_i \neq y_i \land \hat{y}_i^{\text{rat}} = y_i\}$$

힌트는 학습 데이터에 포함시키지 않고 (마치 모델이 힌트 없이 추론한 것처럼) $D_n \cup D_n^{\text{rat}}$ 위에서 fine-tuning합니다. 전체 알고리즘은 논문의 Algorithm 1에 정리되어 있습니다.

### 2.5 모델 구조와 실험 설정

베이스 모델은 **GPT-J(6B 파라미터, 28-layer decoder-only Transformer, embedding 1024, 16 attention heads × dim 256, FFN hidden 16384, vocab 50.4K)** 입니다. 학습은 batch 8 × seq length 1024, packing 사용, weight decay 미사용, Adam optimizer, learning rate $10^{-6}$, 100 step warmup, 외부 루프마다 fine-tuning step을 20% 증가, 단일 TPU-v3 노드에서 수행되었습니다.

### 2.6 성능 결과

산술(2자리~5자리 덧셈, scratchpad 형식): 16번의 외부 루프 후 전체 정확도 89.5%. 직접 정답을 예측하도록 학습한 baseline은 76.3%. Few-shot CoT는 2자리 덧셈에서조차 1% 미만으로, rationalization이 추가되면 1 iteration 만에 32%로 급상승. Rationalization 없이는 ($n-1$)자리 정확도가 좋아져야 $n$자리가 풀리는 단계적(staged) 학습이 일어나지만, rationalization이 추가되면 여러 자릿수를 동시에 학습합니다.

CommonsenseQA(개발셋 정확도): few-shot direct GPT-J 20.9%, few-shot CoT GPT-J 36.6%, GPT-J direct fine-tuned 60.0%, **STaR(rationalization 없음) 68.8%, STaR(rationalization 포함) 72.5%**, 비교군인 GPT-3 direct fine-tuned 73.0%. STaR은 학습 데이터의 86.7%만 활용하면서 30배 큰 모델에 근접합니다. Prolific 크라우드워커 평가에서 STaR이 생성한 rationale은 few-shot CoT보다 30% 더 자주 선호되었고($p=.039$), 인간이 작성한 rationale보다 74%나 자주 선호되었습니다($p<.001$) — 다만 이는 인간 수준 추론을 의미하기보다 CQA 인간 rationale 데이터셋의 품질 문제를 시사합니다.

GSM8K(테스트셋): few-shot direct 3.0%, few-shot CoT 3.1%, GPT-J direct fine-tuned 5.8%, STaR(rationalization 없음) 10.1%, STaR(rationalization 포함) 10.7%. 학습 데이터의 25~28.7%만 사용했음을 고려하면 큰 향상이지만, 이 데이터셋에서 rationalization의 추가 이득은 미미했습니다.

### 2.7 한계

저자들 스스로 명시한 한계는 다음과 같습니다. 첫째, **첫 iteration에서 few-shot 성능이 random보다 유의미하게 높아야** 부트스트랩이 가능하기 때문에 충분히 큰 사전학습 모델이 필요합니다(GPT-2는 산술 도메인에서조차 부트스트랩 실패). 둘째, **chance 성능이 높은 환경**(예: 이진 결정)에서는 잘못된 rationale이 정답에 대량으로 통과되어 노이즈를 유발합니다. 셋째, 정답을 힌트로 변환하는 방식이 항상 자명하지 않습니다(예: 자유형 생성 과제). 넷째, 높은 temperature 샘플링이 rationalization을 대체하기 어려우며, 잘못된 추론으로 정답을 맞히는 사례가 늘어 일반화를 해칩니다. 다섯째, **충실성(faithfulness) 보장이 불가능** 하다는 점입니다 — 모델이 답을 먼저 결정하고 그 답을 정당화하는 rationale을 사후적으로 생성하더라도 이를 외부에서 식별할 방법이 없습니다(Jacovi & Goldberg, 2020).

---

## 3. 일반화 성능 향상 가능성에 대한 집중 분석

STaR이 일반화에 기여하는 메커니즘은 크게 세 가지로 정리할 수 있습니다.

**(a) 분포 외(OOD) 자릿수에 대한 일반화.** 산술 실험에서 5자리까지 학습한 모델에 20번째 iteration 이후 추가 자릿수를 도입했을 때, 학습 중 한 번도 본 적 없는 9자리·10자리 덧셈에서도 모델이 상당수를 풀어냈습니다(Figure 5). 이는 STaR이 단순히 표면 패턴을 외우는 것이 아니라 자릿수 합산이라는 **알고리즘적 절차** 를 학습하고 있다는 강한 증거입니다. Nye et al. (2021)의 scratchpad 연구가 같은 종류의 OOD 일반화를 보고했던 것과 일관됩니다.

**(b) 잠재변수 모델 관점에서의 분포 개선.** $p_M(y \mid x) = \sum_r p(r \mid x)\, p(y \mid x, r)$ 의 관점에서 STaR은 정답으로 이어지는 rationale에 가중치를 두는 importance-weighted MLE에 가깝고, rationalization은 $p(r \mid x, y)$ 라는 더 좋은 proposal로부터 off-policy 학습을 수행합니다. 이는 KL-정칙화된 정책 개선과 유사한 효과를 낳아, 모델이 "정답으로 이어지는 추론 분포" 쪽으로 단조 개선되는 경향을 만듭니다. 이 해석은 후속 연구에서 STaR을 **EM 알고리즘의 잠재 rationale 버전** 으로 정식화하는 길을 열었습니다.

**(c) Few-shot drift 억제.** 저자들이 보고한 흥미로운 현상은, 후속 iteration에서도 few-shot prompt를 유지하면 rationale 스타일이 초기 분포에서 멀어지는 "drift"가 줄어든다는 것입니다(60.9% → 68.8% rationalization 미사용, 69.9% → 72.5% rationalization 포함). 이는 일반화 측면에서 양면적입니다 — 스타일 일관성·해석 가능성에는 유리하지만, 초기 prompt가 갖는 한계 너머로 모델이 진화하는 것은 제약합니다.

다만 일반화에 관한 **명확한 위험 신호** 도 있습니다. (i) chance 성능이 높은 데이터셋에서 정답이 우연히 맞은 잘못된 rationale이 학습 데이터에 섞여 들어가면 모델이 잘못된 추론 패턴을 강화할 수 있습니다. (ii) Appendix G에서 저자들이 인정하듯 데이터셋 편향이 "유용한" 신호일 경우 STaR은 이를 **증폭** 시킵니다 — 특히 rationalization은 모델이 자연스럽게 도달하지 않을 편향된 답까지 끌어내므로 위험이 커집니다. (iii) 부록 A의 실패 사례(question begs the answer, world state assertions, red herrings, hint short-cutting)는 표면적으로 "추론처럼 보이는" 텍스트를 생성하면서도 실제 일반화 가능한 추론 능력으로 환원되지 않는 경우가 적지 않음을 보여줍니다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점, 2022년 이후 관련 연구 비교

### 4.1 STaR이 촉발한 연구 흐름

STaR은 2022년 발표 이후 **"verifiable answer + self-generated rationale + rejection sampling fine-tuning"** 이라는 패러다임의 표준 구성요소가 되었으며, 이후 등장한 거의 모든 대형 추론 모델의 학습 파이프라인에 직간접적으로 흔적이 남아 있습니다. 주요 후속 연구 흐름과의 비교는 아래와 같습니다.

**Self-Consistency (Wang et al., 2022, arXiv:2203.11171).** STaR과 거의 동시기에 등장. 여러 reasoning path를 샘플링한 뒤 정답을 다수결로 marginalize하여 GSM8K +17.9%, SVAMP +11.0%, AQuA +12.2% 등 큰 이득을 보고. STaR이 학습 시점에서 정답 일치 여부로 rationale을 필터링한다면, self-consistency는 추론 시점에서 다수결로 일관성을 강화합니다. 두 기법은 본질적으로 **상호 보완적** 이며, STaR 본문에서도 "answer가 없는 데이터셋에 majority vote를 ground-truth proxy로 활용"할 가능성을 미래 과제로 언급했습니다.

**V-STaR (Hosseini et al., 2024, arXiv:2402.06457).** STaR이 잘못된 해(incorrect solution)를 폐기하는 점을 한계로 지적하며, 정답·오답을 모두 활용해 DPO로 verifier를 학습하고 추론 시점에서 후보 중 최상의 해를 선택합니다. STaR의 가장 큰 한계 — "실패 사례에서 학습 신호 확보 불가" — 를 verifier 학습으로 우회한 사례입니다.

**Quiet-STaR (Zelikman et al., 2024, arXiv:2403.09629).** STaR 저자 본인이 일반화한 후속 연구. STaR이 QA라는 매우 제약된 환경에 묶여 있다는 점을 지적하며, 임의의 텍스트의 모든 토큰 위치에서 rationale을 생성하도록 일반화. 학습 가능한 시작/종료 토큰, 토큰별 병렬 샘플링, 확장된 teacher-forcing을 사용하여 인터넷 텍스트로 추가 사전학습한 후 GSM8K zero-shot 5.9%→10.9%, CommonsenseQA 36.3%→47.2% 향상. STaR이 "QA 데이터셋에서의 reasoning 학습"이었다면 Quiet-STaR은 "사전학습 자체에 reasoning을 통합"한 방향입니다.

**Self-Taught Optimizer / STOP (Zelikman et al., 2023).** 코드 생성 자기개선으로 STaR의 아이디어를 재귀적 자기개선으로 확장.

**OpenAI o1 / DeepSeek-R1 / Kimi k1.5 (2024–2025).** STaR 계열의 가장 큰 후속 충격. DeepSeek-R1은 인간 라벨링 추론 궤적 없이도 순수 강화학습만으로 LLM의 추론 능력을 유도할 수 있음을 보였고, self-reflection·verification·동적 전략 적응 같은 고급 추론 패턴이 emergent하게 발현되었습니다. DeepSeek-R1-Zero는 supervised fine-tuning 없이 base LLM에서 출발해 정답 보상과 형식 보상만으로 학습되었고, "aha moment"라 불리는 자기 수정 행동이 자발적으로 출현합니다. DeepSeek-R1은 GRPO(Group Relative Policy Optimization)를 채택하여 단순히 정답/오답이 아니라 과거 시도와의 상대적 비교로 정책을 갱신합니다. Kimi k1.5도 RL 이전에 long-CoT 예시로 fine-tune하는 cold start를 거쳐 자체 문제 해결 전략을 학습하며, 이후 추가 RL로 응답 길이를 줄여 AIME 2024에서 토큰 약 20% 감소를 달성했습니다. 이 흐름은 STaR의 두 가지 한계 — (a) 정답 일치 필터링이 단순 indicator reward에 머무른 점, (b) 실패 사례 활용 부족 — 를 "process reward, group-relative advantage, RL with verifiable rewards"로 적극적으로 메운 결과로 볼 수 있습니다.

또한 DeepSeek-R1과 같은 R1-style 모델들이 답을 생성할 때 과도하게 길고 중복적인 추론 체인을 만드는 "overthinking" 문제가 새로운 연구 주제로 부상했고, 이는 STaR이 다루지 않은 추론 효율성 차원의 과제입니다.

### 4.2 향후 연구에서 고려할 점

첫째, **chance 성능이 높은 과제에서의 잘못된 rationale 필터링** 이 여전히 미해결입니다. STaR 저자들도 결론에서 명시했듯, 이진 분류·5지선다 등에서는 정답이 우연히 일치하는 추론이 많이 섞입니다. process reward model(PRM)이나 verifier를 학습하는 V-STaR 계열, 또는 추론 단계별 검증을 수행하는 step-wise reward(예: OpenAI의 "Let's Verify Step by Step")가 한 답이 될 수 있습니다.

둘째, **편향 증폭과 충실성(faithfulness) 평가** 입니다. STaR은 정답 일치만으로 rationale의 "유용함"을 정의하는데, 이는 데이터셋 내 편향이 정답과 상관관계가 있다면 그 편향을 추론 절차에 새겨 넣을 위험을 갖습니다. rationale이 실제 모델 내부 결정 과정을 반영하는지 검증하는 mechanistic 해석 가능성 연구와의 결합이 필요합니다.

셋째, **rationalization을 일반 도메인으로 확장하는 어려움** 입니다. 객관식 또는 산술처럼 정답을 "힌트"로 자명하게 주입할 수 있는 도메인을 벗어나면 hint augmentation의 설계가 비자명합니다. 자유형 생성, open-domain QA, 코드 생성 등에서는 부분적 정답·테스트 케이스·실행 결과 등 다양한 검증 신호를 hint로 활용하는 새로운 설계가 필요합니다.

넷째, **scaling과 base 모델의 reasoning 능력 의존성** 입니다. STaR은 GPT-2 규모에서는 부트스트랩 자체가 불가능했습니다. DeepSeek-R1-Zero가 base 모델만으로 RL을 통해 "aha moment"를 만들어낸 것은 기반 모델 능력이 충분히 클 때 자기개선의 임계점이 낮아짐을 시사합니다. 향후 연구는 어느 규모·어떤 사전학습 분포에서 자기개선 임계점이 형성되는지 정량화할 필요가 있습니다.

다섯째, **explorative diversity와 exploitation의 균형** 입니다. STaR은 낮은 temperature greedy decoding으로 분산을 줄이지만, 이는 처음 푼 문제 위주로 데이터를 수집하게 만들어 어려운 문제로의 확장을 방해합니다. rationalization, 다양한 prompt augmentation, MCTS 기반 reasoning(예: AlphaProof, rStar) 같은 탐색 강화 기법과의 결합이 자연스러운 다음 단계입니다.

여섯째, **추론 비용 제어** 입니다. R1 계열에서 부각된 overthinking 문제는 STaR이 구조적으로 회피하기 어렵습니다 — 더 긴 rationale이 정답률을 높이면 학습 데이터로 더 자주 채택되어 모델은 점점 더 길게 생각하도록 진화할 수 있습니다. 길이에 대한 보상 조정(Kimi k1.5 방식) 또는 적응형 reasoning length가 결합되어야 합니다.

---

## 참고자료

- 본 논문(분석 대상): Zelikman, Wu, Mu, Goodman. *STaR: Bootstrapping Reasoning With Reasoning.* arXiv:2203.14465v2, 2022.
- Wei et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* arXiv:2201.11903 — https://arxiv.org/abs/2201.11903
- Wang et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* arXiv:2203.11171 — https://arxiv.org/abs/2203.11171
- Hosseini et al. *V-STaR: Training Verifiers for Self-Taught Reasoners.* arXiv:2402.06457 — https://arxiv.org/abs/2402.06457
- Zelikman et al. *Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking.* arXiv:2403.09629 — https://arxiv.org/html/2403.09629v1
- DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948 — https://arxiv.org/abs/2501.12948 ; Nature 게재본 — https://www.nature.com/articles/s41586-025-09422-z
- DeepLearning.AI The Batch, "How DeepSeek-R1 and Kimi k1.5 Use Reinforcement Learning to Improve Reasoning" — https://www.deeplearning.ai/the-batch/how-deepseek-r1-and-kimi-k1-5-use-reinforcement-learning-to-improve-reasoning/
- HuggingFace Blog, "From Zero to Reasoning Hero: How DeepSeek-R1 Leverages Reinforcement Learning" — https://huggingface.co/blog/NormalUhr/deepseek-r1-explained
- *Don't Overthink It: A Survey of Efficient R1-style Large Reasoning Models.* arXiv:2508.02120 — https://arxiv.org/pdf/2508.02120
- Emergent Mind, "Self-Taught Reasoning (STaR)" 토픽 페이지 — https://www.emergentmind.com/topics/self-taught-reasoning-star

위 답변에서 STaR 본문 내용(수식, 알고리즘, 표 1·2 수치, 부록의 한계 분석)은 업로드된 논문에서 직접 확인한 사실에 기반했고, 후속 연구의 비교는 위 출처들에서 검증했습니다. R1 계열의 정확한 기법별 기여도(예: GRPO 도입자, "aha moment"의 최초 명명 등)는 출처 간 표현이 다소 다를 수 있어 핵심 사실 위주로만 인용했습니다.
