# Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution

---

## 1. 핵심 주장과 주요 기여 요약

**Promptbreeder (PB)**는 Google DeepMind에서 제안한 **자기참조적 자기개선(self-referential self-improvement)** 메커니즘으로, 대규모 언어모델(LLM)의 프롬프트를 진화 알고리즘을 통해 자동으로 탐색·최적화하는 시스템이다.

### 핵심 주장
1. 수작업으로 설계된 프롬프트 전략(Chain-of-Thought, Plan-and-Solve 등)은 차선(sub-optimal)이며, **자동화된 진화적 탐색**을 통해 이를 능가할 수 있다.
2. LLM 자체를 **변이 연산자(mutation operator)**로 활용하여 태스크 프롬프트(task-prompt)를 진화시키는 동시에, **변이 프롬프트(mutation-prompt)** 자체도 함께 진화시키는 **자기참조적 구조**를 실현한다.
3. 파라미터 업데이트 없이 **자연어 수준**에서 자기개선이 가능하므로, 더 크고 능력 있는 LLM이 등장할수록 이 접근법의 이점이 확대된다.

### 주요 기여
- **(i)** 태스크 프롬프트 진화와 변이 프롬프트 진화를 동시에 수행하는 자기참조적 자기개선 방법론 제안
- **(ii)** GSM8K, SVAMP, MultiArith, AddSub, AQuA-RAT, SingleEq, CommonsenseQA, StrategyQA 등 **산술 및 상식 추론 벤치마크**에서 SOTA 대비 성능 향상 달성
- **(iii)** 다양한 자기참조적 구성 요소들의 기여도에 대한 체계적 분석 (ablation study)

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

LLM의 성능은 프롬프트의 구체적 표현(phrasing)에 극히 민감하다 (Madaan & Yazdanbakhsh, 2022). 기존의 수작업 프롬프트 설계는:
- **도메인 특화 적응이 어렵고**
- **탐색 공간이 방대하여** 최적 프롬프트를 찾기 힘들다

Automatic Prompt Engineer(APE, Zhou et al., 2023)는 자동 프롬프트 생성을 시도했으나, **3라운드 이후 수확 체감(diminishing returns)** 문제를 겪었다. Promptbreeder는 이 문제를 **다양성 유지 진화 알고리즘**으로 해결한다.

### 2.2 제안 방법 (수식 포함)

#### 기본 구조

Promptbreeder의 핵심은 LLM을 변이 연산자로 활용하는 것이다. 태스크 프롬프트 $P$를 변이 프롬프트 $M$으로 변이시키는 과정은 다음과 같다:

$$P' = \text{LLM}(M + P)$$

여기서 ' $+$ '는 문자열 연결(concatenation)을 의미한다.

자기참조적 메커니즘의 핵심인 **변이 프롬프트 자체의 진화(hyper-mutation)**는 하이퍼-변이 프롬프트 $H$를 사용하여 수행된다:

$$M' = \text{LLM}(H + M)$$

#### 진화 단위(Unit of Evolution)

각 진화 단위는 다음으로 구성된다:
- **태스크 프롬프트 집합** (통상 2개): $\{P_1, P_2\}$
- **변이 프롬프트**: $M$
- **(few-shot 경우)** 올바른 풀이 과정(workings out): $\{C_1, C_2, \ldots\}$

#### 적합도 평가

각 태스크 프롬프트의 적합도(fitness)는 훈련 데이터에서 무작위로 추출한 100개 Q&A 쌍에 대한 정확도로 측정된다:

$$\text{Fitness}(P) = \frac{1}{|B|}\sum_{(q,a) \in B} \mathbb{1}[\text{LLM}(P + q) = a]$$

여기서 $B$ 는 크기 100의 무작위 배치이다.

#### 진화 프레임워크

**이진 토너먼트 유전 알고리즘(Binary Tournament Genetic Algorithm)** (Harvey, 2011)을 사용한다:
1. 모집단에서 두 개체를 무작위 추출
2. 적합도가 높은 개체를 선택하여 변이 적용
3. 변이된 복제본으로 패자를 대체

#### 초기화

초기 태스크 프롬프트는 **사고 스타일(thinking-style)** $t \sim \mathcal{T}$, **변이 프롬프트** $m \sim \mathcal{M}$, **문제 설명(problem description)** $D$를 결합하여 생성된다:

$$P_{\text{init}} = \text{LLM}(m + t + \text{"INSTRUCTION:"} + D + \text{"INSTRUCTION MUTANT:"})$$

### 2.3 변이 연산자 (5개 클래스, 9개 연산자)

| 클래스 | 연산자 | 설명 |
|---|---|---|
| **Direct Mutation** | Zero-order Prompt Generation | 문제 설명으로부터 새 힌트를 생성 (부모 프롬프트 불사용) |
| | First-order Prompt Generation | $P' = \text{LLM}(M + P)$ — 표준 무성 변이 |
| **EDA Mutation** | EDA Mutation | 현재 모집단의 필터링된 프롬프트 목록으로부터 새 프롬프트 생성 |
| | EDA Rank and Index Mutation | 적합도 순서로 정렬된 목록에서 외삽 |
| | Lineage Based Mutation | 엘리트 계보(chronological elite history)로부터 생성 |
| **Hyper-Mutation** | Zero-order Hyper-Mutation | 문제 설명 + 사고 스타일로 새 변이 프롬프트 생성 |
| | First-order Hyper-Mutation | $M' = \text{LLM}(H + M)$ — 변이 프롬프트 자체를 개선 |
| **Lamarckian Mutation** | Working Out to Task-Prompt | 성공적 풀이 과정으로부터 태스크 프롬프트를 역공학 |
| **Crossover & Shuffling** | Prompt Crossover / Context Shuffling | 10% 확률로 다른 개체의 프롬프트로 교체; few-shot 컨텍스트 재구성 |

각 복제 이벤트에서 9개 연산자 중 하나가 **균일 확률**로 선택된다.

### 2.4 다양성 유지 메커니즘

1. **BERT 임베딩 코사인 유사도** 기반 필터링: 유사도 > 0.95인 개체는 EDA 리스트에서 제외
2. **적합도 공유(Fitness Sharing)**: BERT 유사도 기반 (Shir & Bäck, 2005)
3. **무작위 문자열 추가**: 지역 최적에 갇힐 때 프롬프트 앞에 임의 문자열 삽입
4. **Redescriber 온도 진화**: 1.0~2.0에서 초기화, 각 복제 시 $[-0.2, 0.2]$ 범위의 균일 난수로 변이

### 2.5 성능 향상

**Table 1** (논문 p.2)에서 주요 결과를 발췌하면:

| 벤치마크 | PS+ (text-davinci-003) | APE (PaLM 2-L) | OPRO (PaLM 2-L) | **PB Zero-shot** | **PB Few-shot** |
|---|---|---|---|---|---|
| MultiArith* | 91.8 | 95.8 | – | **99.7** | **100.0** |
| GSM8K | 59.3 | 77.9 | 80.2 | **83.9** | **83.5** |
| SVAMP* | 75.7 | 73.0 | – | **90.2** | **93.7** |
| CSQA | 71.9 | 67.3 | – | **85.4** | **85.9** |
| AQuA-RAT | 46.0 | 45.7 | – | **62.2** | **64.6** |
| SQA | 65.4 | 38.4 | – | **71.8** | **80.2** |

GSM8K에서 OPRO의 최적 프롬프트 "Take a deep breath and work on this problem step-by-step" (80.2%)를 능가하는 **"SOLUTION"**이라는 직관에 반하는 단순 프롬프트로 83.9%를 달성하였다.

ETHOS 혐오 발언 분류에서는 수작업 프롬프트(80%) 대비 **89%**를 달성하는 복잡한 도메인 특화 프롬프트를 진화시켰다.

### 2.6 한계

1. **프롬프팅 토폴로지 고정**: 프롬프트의 *내용*만 적응시키며, 프롬프팅 *알고리즘 자체*(조건부 적용, 분기 구조 등)는 진화시키지 못한다.
2. **평가 함수의 비자기참조성**: 변이 방식은 자기참조적으로 개선하지만, 적합도 평가 방식 자체는 외부에서 고정적으로 지정된다.
3. **계산 비용**: 모집단 크기 50, 20-30세대 진화에 수천 번의 LLM 호출이 필요하다.
4. **단일 모달리티**: 인간 사고의 멀티모달 특성(시각적 이미지, 억양 등)을 반영하지 못한다.
5. **LLM이 적합도 값을 이해하지 못하는 문제**: 예비 실험에서 LLM이 적합도 수치를 해석하지 못하여 EDA 변이에서 적합도 값을 제공하지 않는다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 도메인 적응을 통한 일반화

Promptbreeder는 **동일한 시스템**이 산술 추론, 상식 추론, 혐오 발언 분류, 명령 유도(instruction induction) 등 **다양한 도메인에 적용** 가능한 범용(general-purpose) 시스템이다. 사용자가 제공하는 것은 도메인별 문제 설명 $D$뿐이며, 나머지 과정은 자동화된다.

### 3.2 다양성 유지와 수확 체감 방지

APE가 겪었던 **수확 체감(diminishing returns)** 문제를 다음 메커니즘들로 해결한다:

- **Zero-order Prompt Generation**: 진화가 발산할 때 원래 문제 설명에서 새로운 프롬프트를 재생성하여, 자동 커리큘럼 학습의 균일 재표집(uniform re-sampling)과 유사한 효과를 낸다.
- **BERT 유사도 기반 다양성 필터링**: 코사인 유사도 임계값 0.95를 적용하여 모집단의 다양성을 유지한다.
- **다중 변이 연산자**: 9개의 다양한 연산자가 탐색 공간의 상이한 영역을 커버한다.

### 3.3 자기참조적 변이 프롬프트 진화의 일반화 기여

Ablation 분석 결과(Figure 4, Appendix L), **모든 자기참조적 연산자를 제거하면 거의 모든 데이터셋에서 성능이 하락**하였다:

- **사고 스타일 기반 초기 프롬프트 재기술(SR task-prompt)**: 가장 큰 긍정적 영향 — 제거 시 최대 -80% (StrategyQA)
- **Hyper-Mutation**: 제거 시 대부분 데이터셋에서 -13% ~ -62% 성능 하락
- **Lamarckian Mutation**: ETHOS에서 가장 큰 영향(제거 시 81.6% → 64.6%)

### 3.4 LLM 스케일링과의 시너지

논문의 핵심 가설 중 하나는 **더 강력한 LLM일수록 CoT의 이득이 커진다**(Wei et al., 2022)는 관찰에 기반한다. Promptbreeder는 파라미터 업데이트가 전혀 필요 없으므로:

$$\text{PB의 이점} \propto \text{기저 LLM의 능력}$$

이는 향후 더 큰 LLM에서 PB의 일반화 성능이 더욱 향상될 수 있음을 시사한다.

### 3.5 Few-shot 컨텍스트 진화를 통한 일반화

PB는 태스크 프롬프트뿐만 아니라 **few-shot 예시(context)**도 함께 진화시킨다. 올바른 풀이 과정만 컨텍스트에 축적하고, 무작위 재표집을 통해 다양한 예시를 유지함으로써 **과적합을 방지**하고 일반화를 촉진한다.

### 3.6 한계와 과적합 위험

- 일부 데이터셋(MultiArith*, SingleEq*, AddSub*, SVAMP*)에서는 **데이터의 절반을 훈련에 사용**하고 나머지 절반에서 테스트하였으므로, 훈련-테스트 분할에 의한 편향 가능성이 있다.
- 진화된 프롬프트가 **특정 LLM(PaLM 2-L)에 특화**되었을 가능성이 있으며, 다른 LLM으로의 전이 가능성은 검증되지 않았다.

---

## 4. 해당 논문이 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

1. **프롬프트 엔지니어링의 자동화 패러다임 확립**: PB는 "프롬프트 = LLM의 프로그램"이라는 관점(Zhou et al., 2023)을 실질적으로 구현하여, 수작업 프롬프트 설계에서 **자동 진화적 탐색**으로의 전환을 촉진한다.

2. **자기참조적 AI 시스템의 가능성 제시**: Schmidhuber (1993, 2003)의 자기참조적 자기개선 비전을 **자연어 수준**에서 실현함으로써, 파라미터 수정 없이도 자기개선이 가능함을 보여준다.

3. **진화 알고리즘과 LLM의 결합**: LLM을 변이 연산자로 활용하는 접근(Lehman et al., 2022; Meyerson et al., 2023)의 실용적 가치를 입증하였다.

4. **Open-ended self-improvement 방향 제시**: 프롬프트 내용뿐 아니라 프롬프팅 알고리즘 자체를 진화시키는, 더 개방적인 자기개선 시스템으로의 확장 방향을 명시적으로 제안한다.

### 4.2 향후 연구 시 고려할 점

1. **프롬프팅 토폴로지의 진화**: 현재 고정된 2단계 순차 적용 구조 대신, 조건부 적용·분기·루프 등 **프롬프팅 알고리즘 자체를 진화**시키는 연구가 필요하다.

2. **적합도 함수의 자기참조적 개선**: 현재 외부에서 고정된 적합도 함수를 LLM 자체가 개선하는 방향(auxiliary fitness measures의 자동 생성; cf. Jaderberg et al., 2017b)이 탐색되어야 한다.

3. **다중 LLM 간 전이 가능성**: 특정 LLM에서 진화된 프롬프트가 다른 LLM에서도 효과적인지에 대한 체계적 연구가 필요하다.

4. **계산 효율성**: 진화 과정의 LLM 호출 횟수를 줄이는 효율적 탐색 전략(surrogate model, early stopping 등)의 도입이 필요하다.

5. **멀티모달 확장**: 텍스트뿐 아니라 이미지, 코드 등 다양한 모달리티를 포함하는 프롬프트 전략의 진화가 가능할지 탐색해야 한다.

6. **LLM의 적합도 값 이해 능력 활용**: 논문에서는 LLM이 적합도 값을 이해하지 못한다고 보고했으나, 최근 연구(Mirchandani et al., 2023)에서는 가능성을 시사하므로 재검토가 필요하다.

7. **경쟁적 자기개선**: Self-play 방식으로 LLM 기반 정책들이 서로 경쟁하는 Socratic 대화 등의 **경쟁적 프롬프트 진화** 연구가 유망하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | PB와의 차이 |
|---|---|---|---|
| **Chain-of-Thought (CoT)** (Wei et al., 2022) | 2022 | 중간 추론 단계를 few-shot 예시로 제공 | 수작업 설계; PB가 Zero-shot/Few-shot 모두에서 능가 |
| **Zero-shot CoT** (Kojima et al., 2022) | 2022 | "Let's think step by step" 프롬프트 | 단일 고정 프롬프트; PB는 도메인 적응적 |
| **Self-Consistency (CoT-SC)** (Wang et al., 2022) | 2022 | 다양한 추론 경로 샘플링 후 다수결 | PB와 상호보완적; 진화된 프롬프트에 CoT-SC를 결합 가능 |
| **Automatic Prompt Engineer (APE)** (Zhou et al., 2023) | 2023 | LLM으로 프롬프트 후보 생성 및 변이 | 고정된 단일 변이 프롬프트 사용; **3라운드 후 수확 체감**; PB는 변이 프롬프트도 진화 |
| **Plan-and-Solve (PS/PS+)** (Wang et al., 2023b) | 2023 | 계획 수립 후 단계별 해결 유도 | 수작업 설계; PB가 모든 데이터셋에서 능가 |
| **OPRO** (Yang et al., 2023a) | 2023 | 단일 복잡 변이 프롬프트로 최적화; 고정 훈련셋에서 평가 | PB는 **다중 변이 프롬프트를 자율 진화**; 훈련셋의 무작위 부분집합에서 평가하여 다양성 유지; GSM8K에서 OPRO(80.2%) 대비 PB(83.9%) 우위 |
| **EvoPrompt** (Guo et al., 2023) | 2023 | 고정 변이/교차 프롬프트; 차분(difference) 기반 변이 | 전체 모집단을 수작업 초기화 필요; PB는 단일 문제 설명에서 시작; 변이 프롬프트의 자기참조적 개선 없음 |
| **Tree of Thoughts (ToT)** (Yao et al., 2023) | 2023 | CoT를 트리 구조로 확장; 백트래킹 가능 | PB와 상호보완적; ToT의 프롬프트를 PB로 최적화할 수 있음 |
| **Graph of Thoughts (GoT)** (Besta et al., 2023) | 2023 | 임의 그래프 구조로 일반화 | 구조 설계는 수작업; PB의 토폴로지 진화로 확장 가능 |
| **STaR** (Zelikman et al., 2022) | 2022 | CoT 생성 후 올바른 추론에 대해 파인튜닝 | 파라미터 업데이트 필요; PB는 파라미터 고정 |
| **Self-Refine** (Madaan et al., 2023) | 2023 | 응답 생성 → 피드백 → 개선의 반복 | 단일 추론 시 반복; PB는 모집단 수준에서 진화 |
| **APO** (Pryzant et al., 2023) | 2023 | "Gradient descent" 유사 텍스트 기반 프롬프트 최적화 | PB의 Lamarckian 연산자와 유사한 자기참조적 오류 수정; PB는 더 다양한 변이 연산자 활용 |
| **Evolution through Large Models (ELM)** (Lehman et al., 2022) | 2022 | LLM을 진화적 변이 연산자로 활용 | PB의 이론적 토대; PB는 프롬프트 도메인에 특화하여 자기참조적 확장 |
| **Self-referential weight matrix** (Irie et al., 2022) | 2022 | Fast weight programmer 기반 자기참조적 가중치 행렬 | 파라미터 수준 자기참조; PB는 **자연어 수준**에서 자기참조 — 더 확장 가능 |

### 비교 요약 수식

APE, OPRO, EvoPrompt, PB의 변이 구조를 비교하면:

**APE:**
$$P' = \text{LLM}(M_{\text{fixed}} + P)$$

**OPRO:**
$$P' = \text{LLM}(M_{\text{complex, fixed}} + \text{history})$$

**EvoPrompt:**
$$P' = \text{LLM}(M_{\text{fixed}} + P_1 + P_2) \quad \text{(differential evolution 스타일)}$$

**Promptbreeder:**
$$P' = \text{LLM}(M + P), \quad M' = \text{LLM}(H + M)$$

핵심 차이는 PB만이 $M$ 자체를 **진화적으로 개선**($M \to M'$)한다는 점이다.

---

## 참고자료

1. **Fernando, C., Banarse, D., Michalewski, H., Osindero, S., & Rocktäschel, T.** (2023). "Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution." *arXiv:2309.16797v1 [cs.CL]*. — 본 분석의 주 논문
2. **Wei, J. et al.** (2022). "Chain-of-thought prompting elicits reasoning in large language models." *NeurIPS 2022*.
3. **Zhou, Y. et al.** (2023). "Large language models are human-level prompt engineers." *ICLR 2023*.
4. **Yang, C. et al.** (2023a). "Large language models as optimizers." *arXiv:2309.03409*.
5. **Wang, L. et al.** (2023b). "Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models." *ACL 2023*.
6. **Guo, Q. et al.** (2023). "Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers." *arXiv*.
7. **Kojima, T. et al.** (2022). "Large language models are zero-shot reasoners." *NeurIPS 2022*.
8. **Lehman, J. et al.** (2022). "Evolution through large models." *arXiv:2206.08896*.
9. **Irie, K. et al.** (2022). "A modern self-referential weight matrix that learns to modify itself." *ICML 2022*.
10. **Schmidhuber, J.** (1993). "A 'Self-Referential' Weight Matrix." *ICANN '93*.
11. **Schmidhuber, J.** (2003). "Gödel machines: self-referential universal problem solvers making provably optimal self-improvements." *arXiv preprint cs/0309048*.
12. **Yao, S. et al.** (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." *arXiv*.
13. **Besta, M. et al.** (2023). "Graph of thoughts: Solving elaborate problems with large language models." *arXiv:2308.09687*.
14. **Madaan, A. et al.** (2023). "Self-refine: Iterative refinement with self-feedback." *arXiv:2303.17651*.
15. **Pryzant, R. et al.** (2023). "Automatic prompt optimization with 'gradient descent' and beam search." *arXiv:2305.03495*.
16. **Zelikman, E. et al.** (2022). "Star: Bootstrapping reasoning with reasoning." *NeurIPS 2022*.
17. **Harvey, I.** (2011). "The microbial genetic algorithm." *ECAL 2009, Springer*.
18. **Meyerson, E. et al.** (2023). "Language model crossover: Variation through few-shot prompting." *arXiv:2302.12170*.
19. **Anil, R. et al.** (2023). "PaLM 2 Technical Report." Google.
20. **Wang, X. et al.** (2022). "Self-consistency improves chain of thought reasoning in language models." *arXiv:2203.11171*.
