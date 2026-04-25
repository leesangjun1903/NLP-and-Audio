# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문의 핵심 주장은 **"Chain-of-Thought(CoT) 프롬프팅"**, 즉 중간 추론 단계(intermediate reasoning steps)의 연쇄를 few-shot 예시로 제공하면, 충분히 큰 LLM이 복잡한 추론 과제에서 현저한 성능 향상을 보인다는 것입니다.

수식으로 표현하면, 기존 few-shot 프롬프팅이 다음의 쌍을 사용하는 데 반해:

$$(\text{input}, \text{output})$$

CoT 프롬프팅은 아래의 트리플을 사용합니다:

$$(\text{input},\ \underbrace{z_1, z_2, \ldots, z_n}_{\text{chain of thought}},\ \text{output})$$

여기서 $z_i$는 최종 출력에 이르는 중간 자연어 추론 단계입니다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **방법론적 기여** | 별도의 파인튜닝 없이 few-shot 예시만으로 복잡 추론 유도 |
| **실증적 기여** | 산술·상식·기호 추론 벤치마크에서 SOTA 달성 |
| **이론적 기여** | CoT 추론이 모델 규모의 창발적(emergent) 능력임을 제시 |
| **해석 가능성** | 모델의 추론 과정을 자연어로 해석 가능하게 함 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

논문이 직면한 두 가지 기존 한계:

**① Rationale-augmented finetuning의 한계**
- 고품질 rationale 데이터셋 구축 비용이 매우 높음
- 특정 태스크에 종속적(task-specific)

**② 기존 few-shot 표준 프롬프팅의 한계**
- 산술·논리 등 복잡한 추론 태스크에서 성능이 낮음
- 모델 크기를 키워도 성능이 plateau를 보임 (flat scaling curve)

이를 수식으로 표현하면, 기존 표준 프롬프팅의 조건부 생성:

$$P(\text{output} \mid \text{input}, \text{exemplars})$$

CoT 프롬프팅의 조건부 생성:

$$P(\text{output} \mid \text{input},\ z_1, z_2, \ldots, z_n,\ \text{exemplars with CoT})$$

CoT는 중간 단계 $z_i$들을 명시적으로 생성함으로써 더 나은 $P(\text{output})$을 유도합니다.

---

### 2.2 제안하는 방법

**방법의 핵심**: 기존 few-shot 예시에서 답만 제공하던 것을, 자연어로 기술된 중간 추론 과정을 포함한 예시로 교체합니다.

```
[표준 프롬프팅]
Q: Roger has 5 tennis balls. He buys 2 more cans...
A: The answer is 11.

[CoT 프롬프팅]
Q: Roger has 5 tennis balls. He buys 2 more cans...
A: Roger started with 5 balls. 2 cans of 3 tennis balls 
   each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
```

**구현 세부사항**:
- 총 8개의 few-shot CoT 예시 사용 (대부분의 벤치마크)
- 별도 파인튜닝 없이 off-the-shelf 모델에 그대로 적용
- Greedy decoding 사용 (후속 연구에서 majority voting으로 개선)

외부 계산기(external calculator)를 추가한 경우:

$$\text{solve rate}_{\text{CoT+calc}} \geq \text{solve rate}_{\text{CoT}} \geq \text{solve rate}_{\text{standard}}$$

실험 결과 (GSM8K, PaLM 540B):
- 표준 프롬프팅: $17.9\%$
- CoT 프롬프팅: $56.9\%$
- CoT + 외부 계산기: $58.6\%$

---

### 2.3 모델 구조

논문은 새로운 모델 구조를 제안하지 않습니다. 대신 기존 대형 언어 모델들에 프롬프팅 방식을 적용합니다.

실험에 사용된 모델:

| 모델 | 파라미터 범위 |
|---|---|
| GPT-3 (InstructGPT 계열) | 350M ~ 175B |
| LaMDA | 422M ~ 137B |
| PaLM | 8B ~ 540B |
| UL2 | 20B |
| Codex | code-davinci-002 |

**핵심 발견: CoT는 창발적 능력(emergent ability)**

$$\text{CoT 효과} \approx 0 \quad \text{when} \quad |\theta| \lesssim 10^{11} \text{ parameters}$$

$$\text{CoT 효과} \gg 0 \quad \text{when} \quad |\theta| \gtrsim 10^{11} \text{ parameters}$$

즉, 약 $100B$ 파라미터 이상의 모델에서만 CoT가 효과적으로 작동합니다. 소규모 모델은 유창하지만 비논리적인 CoT를 생성하여 오히려 성능을 저하시킵니다.

---

### 2.4 성능 향상

**산술 추론 (Arithmetic Reasoning)**:

| 벤치마크 | 이전 최고 성능 | PaLM 540B 표준 | PaLM 540B CoT |
|---|---|---|---|
| GSM8K | 55% (finetuned GPT-3) | 17.9% | **56.9%** |
| SVAMP | 57.4% | 69.4% | **79.0%** |
| MAWPS | 88.4% | 79.2% | **93.3%** |
| AQuA | 37.9% | 25.2% | **35.8%** |
| ASDiv | 75.3% | 72.1% | **73.9%** |

**상식 추론 (Commonsense Reasoning)** (PaLM 540B):

| 벤치마크 | 표준 | CoT |
|---|---|---|
| StrategyQA | 68.6% | **77.8%** (이전 SOTA: 69.4%) |
| Sports Understanding | 80.5% | **95.4%** (인간 전문가: 84%) |
| Date Understanding | 49.0% | **65.3%** |

**기호 추론 (Symbolic Reasoning)** — OOD 일반화:

| 태스크 | 표준 (OOD) | CoT (OOD) |
|---|---|---|
| Last Letter Concat (4단어) | 0.0% | **63.0%** |
| Coin Flip (4회) | 54.8% | **90.2%** |

---

### 2.5 한계점 (Limitations)

논문이 명시한 한계:

1. **실제 추론 여부 불확실**: 모델이 진정한 의미의 "추론"을 하는지는 열린 질문
2. **규모 의존성**: $\sim 100B$ 이상의 대형 모델에서만 효과적 → 실용적 배포 비용 높음
3. **오류 가능성**: 올바른 추론 경로를 보장할 수 없으며, 잘못된 CoT로도 우연히 정답에 도달 가능
4. **파인튜닝 확장성**: few-shot 설정에서는 annotation 비용이 적지만, 파인튜닝을 위한 대규모 CoT 데이터 구축은 여전히 비용이 큼
5. **프롬프트 민감성**: 어노테이터, 예시 순서 등에 따른 분산 존재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Out-of-Distribution(OOD) 일반화

CoT 프롬프팅의 가장 두드러진 일반화 관련 발견은 **기호 추론 태스크에서의 길이 일반화(length generalization)**입니다.

실험 설계:
- **In-domain**: 2단어 이름으로 학습 → 2단어 이름 테스트
- **OOD**: 2단어로 학습 → 3~4단어 이름 테스트

$$\text{Generalization Gap} = \text{Accuracy}_{\text{in-domain}} - \text{Accuracy}_{\text{OOD}}$$

PaLM 540B 결과:

| 방법 | In-domain (2단어) | OOD (4단어) |
|---|---|---|
| 표준 프롬프팅 | 7.6% | 0.0% |
| CoT 프롬프팅 | **99.4%** | **63.0%** |

표준 프롬프팅은 OOD에서 완전히 실패하지만, CoT 프롬프팅은 upward scaling curve를 보이며 OOD에서도 의미 있는 성능을 유지합니다.

### 3.2 일반화를 가능하게 하는 메커니즘

논문은 다음의 절제 실험(ablation study)을 통해 CoT의 핵심이 **자연어로 표현된 순차적 추론** 자체임을 밝힙니다:

| 변형 | GSM8K 성능 | 해석 |
|---|---|---|
| 표준 프롬프팅 | 6.5% | 기준선 |
| 방정식만 출력 | 5.4% | CoT보다 낮음 |
| 가변 계산만 | 6.4% | 중간 토큰 수만으로는 불충분 |
| 답 이후 CoT | 6.1% | 순차적 생성이 핵심 |
| **CoT 프롬프팅** | **14.3%** | 최고 성능 |

이로부터 다음 결론을 도출할 수 있습니다:

$$\text{일반화 향상} \leftarrow \underbrace{\text{순차적 자연어 추론}}_{\text{CoT의 핵심}} \neq \underbrace{\text{단순 변수 계산량}}_{\text{variable compute}} \neq \underbrace{\text{수식 생성}}_{\text{equation only}}$$

### 3.3 태스크 간 일반화 (Cross-task Generalization)

하나의 8개 예시 세트가 여러 산술 벤치마크(GSM8K, SVAMP, ASDiv, MAWPS)에 동시에 적용되어 효과를 발휘합니다. 이는 CoT가 **특정 데이터 분포에 과적합하지 않고** 추론 패턴 자체를 전이한다는 점을 시사합니다.

수식화:

$$\text{Exemplars from } \mathcal{D}_{\text{train}}^{\text{GSM8K}} \xrightarrow{\text{CoT prompting}} \text{성능 향상 on } \mathcal{D}_{\text{test}}^{\text{SVAMP, ASDiv, MAWPS}}$$

### 3.4 모델 규모와 일반화 능력

오류 분석에서 발견된 규모 스케일링의 역할:

PaLM 62B의 오류 45개를 분석했을 때:
- 의미 이해 오류: 20개 → 540B에서 **6개 수정됨**
- 한 단계 누락 오류: 18개 → 540B에서 **12개 수정됨**
- 기타: 7개 → 540B에서 **4개 수정됨**

이는 모델 규모가 클수록:
$$P(\text{correct CoT} \mid \text{question}) \propto f(|\theta|)$$
가 증가함을 시사하며, 일반화 성능이 모델 파라미터 수에 따라 단조 증가함을 보여줍니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**① 프롬프팅 패러다임의 전환**

본 논문은 "입력 최적화(input-side prompting)"에 집중하던 기존 연구 흐름과 달리, "출력 측 중간 단계 생성(output-side augmentation)"이라는 새로운 방향을 제시했습니다. 이는 이후 수많은 파생 연구의 토대가 됩니다.

**② 창발적 능력 연구의 촉매**

Wei et al. (2022b) "Emergent Abilities of Large Language Models"와 함께, 모델 규모에 따른 비선형적 능력 출현 현상을 체계적으로 연구하는 방향을 개척했습니다.

**③ LLM 추론 능력 평가 기준 재설정**

표준 프롬프팅이 LLM 능력의 하한선(lower bound)에 불과하다는 주장은, 기존 벤치마크들의 평가 타당성에 대한 재고를 촉구했습니다.

### 4.2 2020년 이후 후속 연구 비교 분석

| 연구 | 핵심 아이디어 | CoT와의 관계 | 주요 성과 |
|---|---|---|---|
| **Self-Consistency** (Wang et al., 2022a) | 다수의 CoT 경로를 샘플링하여 다수결 투표 | CoT 확장 | GSM8K에서 CoT 단독 대비 추가 향상 |
| **Zero-shot CoT** (Kojima et al., 2022) | "Let's think step by step" 한 문장만으로 CoT 유발 | CoT 단순화 | 파인튜닝/예시 없이 CoT 효과 달성 |
| **STaR** (Zelikman et al., 2022) | CoT를 자기 생성하여 파인튜닝에 활용 (bootstrapping) | CoT + 파인튜닝 | 소규모 모델에서도 추론 능력 향상 |
| **Least-to-Most Prompting** (Zhou et al., 2022) | 문제를 하위 문제로 분해하여 순차 해결 | CoT 고도화 | 길이 일반화 추가 향상 |
| **Program of Thoughts** (Chen et al., 2022) | 자연어 대신 프로그램 코드로 추론 단계 표현 | CoT 변형 | 수치 계산 오류 감소 |
| **Tree of Thoughts** (Yao et al., 2023) | 선형 CoT를 트리 구조로 확장, backtracking 허용 | CoT 구조 확장 | 전략적 사고가 필요한 문제에서 향상 |
| **ReAct** (Yao et al., 2022) | 추론(Reasoning)과 행동(Acting)을 교차 수행 | CoT + 외부 도구 | 실제 환경과의 상호작용 가능 |

**비교 분석 요약**:

```
[CoT 논문] Few-shot + 자연어 중간 단계
    ↓
[Self-Consistency] 샘플 다양성으로 신뢰성 향상
    ↓
[Zero-shot CoT] 예시 없이도 CoT 유발 가능
    ↓
[Least-to-Most / ToT] 복잡한 분해 전략으로 확장
    ↓
[STaR / Self-Taught] 자기 개선(self-improvement) 루프
    ↓
[ReAct / PoT] 외부 도구·환경과 통합
```

---

### 4.3 앞으로 연구 시 고려할 점

**① 소규모 모델에서의 CoT 유발 방법 개발**

현재 CoT는 $\sim 100B$ 이상에서만 효과적입니다. Knowledge Distillation이나 STaR 방식의 bootstrapping을 통해, 소규모 모델에서도 유사한 추론 능력을 이끌어내는 연구가 필요합니다.

$$\text{목표: } P_{\text{small}}(\text{correct CoT}) \approx P_{\text{large}}(\text{correct CoT})$$

**② 추론 경로의 검증(Verification) 메커니즘**

CoT가 올바른 답을 생성하더라도 추론 경로가 틀릴 수 있습니다 (논문 내 "correct by chance" 사례). 이를 보완하는 verifier 모델이나 자기 검증(self-verification) 메커니즘 연구가 중요합니다.

**③ CoT의 사실성(Factuality) 보장**

생성된 CoT가 사실적으로 틀린 내용을 포함할 수 있습니다. Retrieval-Augmented Generation(RAG)과 CoT의 결합 등, 사실 기반 추론 보장 방법을 연구해야 합니다.

**④ 자동 CoT 생성 및 최적화**

현재 CoT 예시는 인간이 수동으로 작성합니다. LLM 자체를 사용한 자동 CoT 생성(Auto-CoT)이나 강화학습 기반 CoT 최적화 연구가 필요합니다.

**⑤ 다중 모달(Multi-modal) CoT 확장**

텍스트에 국한된 CoT를 이미지, 그래프, 표 등 다중 모달 데이터로 확장하는 연구가 필요합니다.

**⑥ CoT의 창발 조건 규명**

왜 특정 모델 크기($\sim 100B$)에서 CoT 능력이 창발하는지, 사전학습 데이터의 특성·모델 아키텍처·최적화 목적함수 중 어떤 요인이 결정적인지 연구가 필요합니다.

**⑦ 멀티-홉(Multi-hop) 추론의 충실도(Faithfulness) 평가**

생성된 CoT가 실제로 최종 답 도출에 인과적으로 기여하는지, 아니면 단순히 그럴듯하게 보이는 사후 설명인지를 구분하는 평가 방법론 개발이 필요합니다.

---

## 참고 자료

**1차 자료 (논문 본문)**
- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E. H., Le, Q. V., & Zhou, D. (2023). **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**. *NeurIPS 2022*. arXiv:2201.11903v6.

**논문 내 인용된 주요 후속/관련 연구**
- Wang, X., Wei, J., et al. (2022a). **Self-consistency improves chain of thought reasoning in language models**. arXiv:2203.11171.
- Wei, J., Tay, Y., et al. (2022b). **Emergent abilities of large language models**. *Transactions on Machine Learning Research*.
- Zelikman, E., Wu, Y., & Goodman, N. D. (2022). **STaR: Bootstrapping reasoning with reasoning**. arXiv:2203.14465.
- Cobbe, K., et al. (2021). **Training verifiers to solve math word problems**. arXiv:2110.14168.
- Brown, T., et al. (2020). **Language models are few-shot learners**. *NeurIPS*.
- Nye, M., et al. (2021). **Show your work: Scratchpads for intermediate computation with language models**. arXiv:2112.00114.

> **정확도 관련 고지**: 후속 연구 비교 분석 표(Tree of Thoughts, ReAct, Program of Thoughts 등)는 논문 원문에 직접 수록된 내용이 아니라, 해당 논문들의 공개 arXiv 논문 및 제목 기반으로 작성되었습니다. 구체적 수치는 각 원본 논문에서 직접 확인하시기 바랍니다.
