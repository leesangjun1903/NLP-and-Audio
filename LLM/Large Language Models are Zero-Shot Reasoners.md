
# Large Language Models are Zero-Shot Reasoners

> **논문 정보**
> - **저자**: Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa
> - **발표**: NeurIPS 2022 (Advances in Neural Information Processing Systems, Vol. 35, pp. 22199–22213)
> - **arXiv**: [2205.11916](https://arxiv.org/abs/2205.11916)
> - **공식 구현**: [GitHub - kojima-takeshi188/zero_shot_cot](https://github.com/kojima-takeshi188/zero_shot_cot)

---

## 1. 🎯 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

기존의 성공이 LLM의 few-shot 학습 능력 덕분이라는 통념과 달리, 이 논문은 각 답변 앞에 **"Let's think step by step"** 이라는 단순한 문장만 추가함으로써 LLM이 훌륭한 zero-shot 추론자가 될 수 있음을 보인다.

### 1.2 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **Zero-shot-CoT 제안** | 별도의 예시 없이 단일 프롬프트만으로 연쇄 추론 유도 |
| **Task-agnostic 설계** | 하나의 프롬프트로 다양한 추론 태스크에 적용 가능 |
| **강력한 Zero-shot 베이스라인 확립** | 추론 벤치마크의 최소 강력 기준선 제공 |
| **LLM 내재 역량 발견** | 파인튜닝 없이도 숨겨진 추론 능력이 존재함을 실증 |

이 단일 프롬프트가 매우 다양한 추론 태스크에서 범용적으로 작동한다는 사실은 LLM의 미탐구된 근본적인 zero-shot 능력을 시사한다. 저자들은 이 연구가 파인튜닝 데이터셋이나 few-shot 예시를 구성하기 **전에** LLM 내부에 내재된 방대한 zero-shot 지식을 먼저 탐구하는 것의 중요성을 강조한다고 밝힌다.

---

## 2. 🔬 해결하고자 하는 문제, 방법론, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

사전학습된 LLM은 태스크별 예시를 활용하는 excellent few-shot learner로 잘 알려져 있다. 특히 Chain-of-Thought(CoT) 프롬프팅은 단계적 답변 예시를 통해 복잡한 다단계 추론을 유도하며, 일반적인 스케일링 법칙을 따르지 않는 어려운 system-2 태스크인 산술 및 기호 추론에서 SOTA 성능을 달성하였다.

그러나 기존 CoT 방식은 다음 두 가지 핵심 문제를 가진다:

1. **Few-shot 의존성**: 각 태스크마다 수작업으로 만든 단계별 예시(exemplar)가 필수적
2. **태스크 특화성**: 각 태스크에 맞는 별도 프롬프트 설계가 필요하여 범용성 결여

이 논문은 **Zero-shot-CoT**, 즉 연쇄 추론을 위한 zero-shot 템플릿 기반 프롬프팅을 제안한다. 이는 단계별 few-shot 예시를 필요로 하지 않으며, 단일 템플릿으로 다양한 태스크에서 다중 홉 추론을 유도하는 본질적으로 task-agnostic한 방법이다. 핵심 아이디어는 단순히 **"Let's think step by step"** 을 추가하여 단계적 추론을 이끌어내는 것이다.

---

### 2.2 제안 방법 (Two-Stage Prompting Pipeline)

Zero-shot-CoT는 두 단계의 프롬프팅 과정을 사용한다. 입력 질문은 **"Q: [Question]. A: Let's think step by step"** 형태로 포맷팅된다.

#### **Stage 1: Reasoning Extraction (추론 생성)**

$$
\text{Input}_1 = Q \oplus \text{"A: Let's think step by step"}
$$

$$
\hat{R} = \text{LLM}(\text{Input}_1)
$$

- $Q$: 입력 질문
- $\hat{R}$: 생성된 Chain-of-Thought 추론 과정
- $\oplus$: 텍스트 연결(concatenation)

#### **Stage 2: Answer Extraction (답변 추출)**

$$
\text{Input}_2 = Q \oplus \text{"A: Let's think step by step"} \oplus \hat{R} \oplus \text{"Therefore, the answer is"}
$$

$$
\hat{A} = \text{LLM}(\text{Input}_2)
$$

- $\hat{A}$: 최종 답변

즉, 전체 파이프라인은 다음과 같이 요약된다:

$$
\hat{A} = f_{\text{LLM}}\Big( f_{\text{LLM}}(Q \oplus p_{\text{trigger}}) \Big)
$$

여기서 $p_{\text{trigger}} = \text{"Let's think step by step"}$은 모든 태스크에 동일하게 적용되는 **태스크 비의존적(task-agnostic) 단일 트리거 프롬프트**이다.

이 방식은 전통적인 few-shot 방법이 요구하는 세밀하게 설계된 예시들이 필요 없다. 연구자들은 단순성을 위해 **greedy decoding**을 사용하였다. 이 방법의 유연성은 동일한 기본 추론 구조를 유지하면서 다양한 답변 형식에 적응할 수 있다는 점에서 나오며, 이를 진정한 task-agnostic 방법으로 만들어 준다.

---

### 2.3 모델 구조

본 논문은 새로운 모델 아키텍처를 제안하지 않으며, **기존의 사전학습된 LLM을 그대로 활용**하는 프롬프팅 방법을 제안한다. 실험에 사용된 주요 모델은 다음과 같다:

| 모델 | 파라미터 수 | 비고 |
|---|---|---|
| InstructGPT (text-davinci-002) | ~175B | 주요 실험 모델 |
| PaLM | 540B | 검증 모델 |
| GPT-2, GPT-Neo, GPT-J, T0, OPT | 소~중형 | 모델 스케일 분석용 |

부록에서는 GPT-2, GPT-Neo, GPT-J, T0, OPT를 포함한 더 광범위한 언어 모델을 대상으로 광범위한 실험 결과를 제공한다.

---

### 2.4 성능 향상

실험 결과, Zero-shot-CoT는 산술 추론(MultiArith, GSM8K, AQUA-RAT, SVAMP), 기호 추론(Last Letter, Coin Flip), 기타 논리 추론(Date Understanding, Tracking Shuffled Objects) 등 다양한 벤치마크에서 수작업 few-shot 예시 없이도 유의미하게 성능을 향상시켰다. 특히 대형 InstructGPT 모델(text-davinci-002)로 MultiArith에서 17.7% → 78.7%, GSM8K에서 10.4% → 40.7%로 향상되었으며, 540B 파라미터의 PaLM에서도 유사한 수준의 개선이 확인되었다.

**주요 성능 비교표 (InstructGPT text-davinci-002 기준)**:

| 데이터셋 | Zero-shot | Zero-shot-CoT | 향상 |
|---|---|---|---|
| MultiArith | 17.7% | 78.7% | **+61.0%p** |
| GSM8K | 10.4% | 40.7% | **+30.3%p** |
| Last Letter | - | 대폭 향상 | ✅ |
| Coin Flip | - | 대폭 향상 | ✅ |

단일 고정 트리거 프롬프트가 복잡한 다중 홉 사고가 필요한 다양한 태스크에 걸쳐 LLM의 zero-shot 추론 능력을 **특히 모델 규모가 클수록** 더욱 크게 향상시킨다. 또한 최종 예측이 틀린 경우에도 합리적이고 이해 가능한 Chain-of-Thought를 생성한다.

---

### 2.5 한계점

**상식 추론(Commonsense Reasoning) 태스크에서는 Zero-shot-CoT가 성능 향상을 제공하지 못한다.** 많은 생성된 Chain-of-Thought 자체는 놀랍도록 논리적으로 정확하거나 인간이 이해할 수 있는 실수만을 포함하지만, 태스크 메트릭에는 직접적으로 반영되지 않는다.

Zero-shot-CoT는 추론 태스크가 더 복잡할수록 Few-shot CoT 프롬프팅만큼 효과적이지 않다. 또한 **답변 추출 단계(answer extraction step)는 종종 태스크 특화적이며**, 처음 보이는 것만큼 범용적이지 않다는 한계를 가진다.

**모델 크기 및 사전학습 의존성**: 소규모 모델이나 instruction-tuned되지 않은 모델은 CoT 트리거에 반응하지 않을 수 있으며, 특히 구조화된 문서에 대한 사전학습 부족으로 인해 table 형식 프로토콜에 취약하다.

추가적으로 다음과 같은 한계도 고려해야 한다:

- **환각(Hallucination) 위험**: 그럴듯하지만 오류가 있는 추론 체인 생성 가능
- **프롬프트 민감성**: 트리거 문구의 표현 방식에 따라 성능 변동 존재
- **복잡한 수식 계산 취약**: 심층적 수학 계산에서 정확성 한계 존재

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

### 3.1 Task-Agnostic 프롬프트의 범용성

이 단일 프롬프트가 매우 다양한 추론 태스크에서 범용적으로 작동한다는 사실은, 단순한 프롬프팅으로도 **고수준의 멀티태스크 광범위 인지 능력**이 추출될 수 있음을 시사한다.

Zero-shot-CoT의 일반화 성능 향상은 다음의 수식으로 표현할 수 있다. 기존의 Few-shot-CoT에서는 태스크별 학습 분포 $\mathcal{D}_{\text{task}}$에 의존하지만:

$$
P_{\text{Few-shot-CoT}}(\hat{A} \mid Q, \mathcal{E}_{\text{task}}) \quad \text{태스크 특화 예시} \mathcal{E}_{\text{task}} \text{ 필요}
$$

Zero-shot-CoT는 모든 태스크 $\mathcal{T}$에 동일한 단일 프롬프트 $p$를 사용:

$$
P_{\text{Zero-shot-CoT}}(\hat{A} \mid Q, p) \approx P_{\text{LLM}}(\hat{A} \mid Q \oplus p_{\text{trigger}} \oplus \hat{R}), \quad \forall \mathcal{T}
$$

이는 LLM이 사전학습을 통해 내재화한 **도메인 불변(domain-invariant)** 추론 패턴을 활성화함을 의미한다.

### 3.2 모델 스케일과 일반화

논문은 Zero-shot-CoT의 효과가 **모델 크기에 따라 극적으로 증가**함을 보여주었으며, 이 능력이 더 큰 모델에서 더 강하게 나타남을 시사한다. 즉, 더 작은 모델은 Zero-shot-CoT에서 더 낮은 성능을 보이고, 더 큰 모델은 유의미하게 더 나은 성능을 보인다.

이는 **창발적 능력(emergent capability)** 의 관점에서 이해할 수 있다:

$$
\text{Performance}(\text{Zero-shot-CoT}) \propto f(\text{Model Scale}), \quad f \text{: 비선형 증가 함수}
$$

### 3.3 Instruction Tuning과의 상호 보완성

Zero-shot-CoT는 instruction tuning에 대해 직교적(orthogonal)이며, instruction-tuned 모델에서도 zero-shot 성능을 추가로 높인다.

이는 Zero-shot-CoT가 다양한 LLM 학습 패러다임과 결합 가능하여 **범용적 일반화 도구**로서의 잠재력을 갖고 있음을 의미한다.

---

## 4. 🔮 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

#### **(1) 프롬프트 엔지니어링 연구의 새로운 방향 제시**
이 연구는 복잡한 few-shot 예시나 파인튜닝 데이터셋 구축에 투자하기 **전에**, 단순한 프롬프팅 전략을 통해 이미 모델에 존재하는 능력을 탐색해야 함을 시사한다.

#### **(2) Zero-shot 추론 후속 연구의 촉발**
Chain-of-Thought(CoT) 및 Zero-shot CoT는 수학, 기호 추론, 상식 태스크에서 성능을 크게 향상시킬 수 있음을 보였다. Self-Consistency는 다수의 CoT 샘플을 투표를 통해 집계하여 분산을 줄이고, Least-to-Most는 문제를 순차적으로 해결되는 하위 질문으로 분해하며, Plan-and-Solve는 모델에게 실행 전 계획을 먼저 스케치하도록 요청하고, ReAct는 짧은 추론 흔적과 도구 사용 행동을 교차시킨다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법 | Zero-shot 여부 | 핵심 아이디어 |
|---|---|---|---|---|
| **Chain-of-Thought Prompting** (Wei et al.) | 2022 | Few-shot-CoT | ❌ | 단계별 예시로 추론 유도 |
| **Zero-shot-CoT** (Kojima et al.) | 2022 | Zero-shot-CoT | ✅ | "Let's think step by step" 단일 트리거 |
| **Self-Consistency** (Wang et al.) | 2022 | Multi-path voting | 혼합 | 다수 추론 경로 → 다수결 |
| **Tree of Thoughts** (Yao et al.) | 2023 | 트리 탐색 | ❌/✅ | 추론 경로를 트리 구조로 확장 |
| **Plan-and-Solve** (Wang et al.) | 2023 | Zero-shot | ✅ | 계획 후 단계적 해결 |
| **Agent Instructs LLMs** (Crispino et al.) | 2024 | Agent 기반 | ✅ | 자율 에이전트가 추론 과정 지시 |

2023년 Tree of Thoughts(ToT) 프레임워크는 탐색이나 전략적 계획이 필요한 복잡한 태스크에서 LLM이 직면하는 한계를 해결하면서 CoT 접근법을 확장하였다. 이 프레임워크는 LLM이 여러 해결 경로를 생성, 평가 및 확장하도록 유도한다. 각 노드가 중간 단계를 나타내는 트리를 구성함으로써 lookahead와 backtracking 기능을 갖춘 체계적인 추론 경로 탐색을 가능하게 한다.

Wang et al.(2022)이 제안한 Self-Consistency는 chain-of-thought 프롬프팅에서 사용되는 단순한 greedy decoding을 대체하는 것을 목표로 하며, few-shot CoT를 통해 다양한 추론 경로를 샘플링하고, 가장 일관성 있는 답변을 선택하기 위해 생성된 결과를 활용한다.

Self-Consistency는 zero-shot chain-of-thought(Kojima et al., 2022)와 함께 적용되었을 때도 효과적이며, zero-shot CoT의 결과를 **+26.2%**까지 유의미하게 향상시킨다.

Crispino et al.(2024)은 일반 언어 이해 태스크에서 LLM의 zero-shot 추론 능력을 향상시키는 방법을 제안하였다. 구체적으로, 자율 에이전트(autonomous agent)를 구축하여 LLM의 추론 과정을 지시한다. 이 방법은 생성, 분류, 추론에 걸친 광범위한 데이터셋에서 성능을 연구하였으며, 평가한 29개 데이터셋 중 20개에서 SOTA zero-shot 성능을 달성하였다.

---

### 4.3 미래 연구 시 고려해야 할 점

#### ✅ **1. 소규모 모델에서의 일반화 연구**
현재 Zero-shot-CoT는 대형 모델에서 주로 효과적이다. 소규모·경량 모델에서도 유사한 추론 능력을 이끌어내기 위한 방법론(예: distillation, instruction tuning 결합)이 필요하다.

#### ✅ **2. 상식 추론(Commonsense Reasoning) 개선**
상식 추론 태스크(예: CommonsenseQA)에서는 zero-shot 환경에서 검증 및 self-consistency 점수가 선택을 개선하지 못할 수 있다.

이 영역에서의 Zero-shot-CoT 성능 향상은 향후 연구의 주요 과제로 남아 있다.

#### ✅ **3. 추론 정확성 검증 메커니즘**
생성된 Chain-of-Thought가 그럴듯하지만 잘못된 경우(hallucination)를 탐지하고 수정하는 **자동 검증 체계** 연구가 필요하다.

#### ✅ **4. 다국어 및 멀티모달 환경으로의 확장**
교차 언어(cross-lingual) 및 멀티모달 환경에서도, 가중 언어 앙상블(AutoCAP) 등을 통한 강인한 다국어 CoT 추론과 시각·혼합 도메인에서의 zero-shot CoT 프로토콜이 상징적, 절차적, 멀티모달 추론 환경으로 잘 일반화된다는 것이 확인되었다. 이 방향의 체계적 연구가 더욱 필요하다.

#### ✅ **5. 적응형 프롬프트 최적화**
Plan-and-Solve(PS+)는 zero-shot 계획에서 세부 변수 추출 및 중간 계산 지시를 통해 few-shot과의 격차를 줄이고 있다. 또한 인스턴스 적응형 프롬프팅(IAP)과 진화 알고리즘을 통한 질문별 프롬프트 선택이 추가적인 성능 향상을 가져온다.

#### ✅ **6. 추론 신뢰도 및 불확실성 정량화**
단일 추론 경로에만 의존하는 Zero-shot-CoT의 특성상, 신뢰도(confidence calibration) 및 불확실성 추정 연구가 실제 안전·신뢰성 요구 시스템에 중요하다.

---

## 📚 참고 자료 목록

| # | 제목/출처 | 유형 |
|---|---|---|
| 1 | Kojima, T. et al., **"Large Language Models are Zero-Shot Reasoners"**, NeurIPS 2022, Vol. 35, pp.22199–22213. [arXiv:2205.11916](https://arxiv.org/abs/2205.11916) | 주요 논문 |
| 2 | ACM Digital Library, **NeurIPS 2022 proceedings**: [dl.acm.org](https://dl.acm.org/doi/10.5555/3600270.3601883) | 학술지 |
| 3 | Semantic Scholar, [Large Language Models are Zero-Shot Reasoners](https://www.semanticscholar.org/paper/Large-Language-Models-are-Zero-Shot-Reasoners-Kojima-Gu/e7ad08848d5d7c5c47673ffe0da06af443643bda) | 논문 데이터베이스 |
| 4 | OpenReview.net, [Zero-Shot Reasoners PDF (NeurIPS version)](https://openreview.net/pdf?id=e2TBb5y0yFf) | 심사 버전 |
| 5 | GitHub Official Implementation, [kojima-takeshi188/zero_shot_cot](https://github.com/kojima-takeshi188/zero_shot_cot) | 공식 코드 |
| 6 | Notes by Lex, ["Large Language Models are Zero-Shot Reasoners"](https://notesbylex.com/large-language-models-are-zero-shot-reasoners-may-2022) | 해설 |
| 7 | Takara TLDR, [arXiv 2205.11916 Summary](https://tldr.takara.ai/p/2205.11916v4) | 요약 |
| 8 | Wang et al., **"Self-Consistency Improves Chain of Thought Reasoning in Language Models"**, ICLR 2023. [PDF](http://webdocs.cs.ualberta.ca/~dale/papers/iclr23b.pdf) | 관련 논문 |
| 9 | Crispino et al., **"Agent Instructs Large Language Models to be General Zero-Shot Reasoners"**, ICML 2024. [PMLR](https://proceedings.mlr.press/v235/crispino24a.html) | 관련 논문 |
| 10 | Yao et al., **"Tree of Thoughts: Deliberate Problem Solving with LLMs"** (2023) | 관련 논문 |
| 11 | Learn Prompting, ["Zero-Shot CoT Prompting"](https://learnprompting.org/docs/intermediate/zero_shot_cot) | 기술 해설 |
| 12 | Prompt Engineering Guide, ["Self-Consistency"](https://www.promptingguide.ai/techniques/consistency) | 기술 가이드 |
| 13 | Emergent Mind, [Zero-shot Chain-of-Thought Topic](https://www.emergentmind.com/topics/zero-shot-chain-of-thought-prompting) | 연구 동향 |
| 14 | Chowdhury et al., **"Zero-Shot Verification-guided Chain of Thoughts"**, arXiv:2501.13122 (2025) | 최신 연구 |
| 15 | ResearchGate, [Zero-Shot Reasoners Discussion](https://www.researchgate.net/publication/360834082_Large_Language_Models_are_Zero-Shot_Reasoners) | 학술 커뮤니티 |
