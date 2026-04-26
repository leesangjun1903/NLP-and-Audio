# Chain-of-Thought Reasoning without Prompting

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Wang & Zhou (2024)의 이 논문은 **"LLM은 프롬프트 없이도 추론할 수 있는가?"** 라는 근본적인 질문을 탐구합니다. 핵심 주장은 다음과 같습니다:

> **사전 학습된 LLM은 이미 내재적으로 Chain-of-Thought(CoT) 추론 경로를 보유하고 있으며, 이는 단순히 디코딩 과정을 변경함으로써 유도할 수 있다.**

종래의 연구들은 LLM이 직접 질의(QA) 방식에서는 추론 능력이 부족하다고 주장했으나 (Kojima et al., 2022; Wei et al., 2022), 이 논문은 그 믿음이 **그리디 디코딩(greedy decoding)만을 고려한 결과물**임을 증명합니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **발견 1** | 프롬프트 없이 디코딩 변경만으로 CoT 추론 유도 가능 |
| **발견 2** | CoT 경로의 존재가 최종 답변 신뢰도(confidence)와 상관관계를 가짐 |
| **제안 방법** | CoT-decoding: 답변 신뢰도 기반으로 CoT 경로를 선택하는 비지도 방법 |
| **분석 프레임워크** | 프롬프트의 혼입(confounding) 없이 LLM의 내재적 추론 능력 평가 가능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 CoT 연구의 한계:

- **Few-shot CoT prompting** (Wei et al., 2022): 수작업 프롬프트 엔지니어링 필요, 태스크 특화적
- **Zero-shot CoT prompting** (Kojima et al., 2022): "Let's think step by step"과 같은 명시적 지시 필요
- **Instruction Tuning** (Chung et al., 2022): 대규모 CoT 주석 데이터 필요, 비용이 큼

이러한 방법들의 공통적 한계:
1. **인간의 사전 지식(human priors) 주입** → LLM 고유의 추론 능력 평가 불가
2. **태스크별 튜닝 필요** → 일반화 어려움
3. **프롬프트가 실제로 무엇을 하는지 불분명** (Min et al., 2022; Webson & Pavlick, 2022)

### 2.2 제안 방법: CoT-Decoding

#### 기본 아이디어

표준 그리디 디코딩($k=0$)은 항상 최고 확률의 첫 번째 토큰만 선택합니다. CoT-decoding은 첫 번째 디코딩 스텝에서 **상위 $k$개의 대안 토큰**을 탐색한 후, 각 경로에서 그리디 디코딩을 계속 진행합니다.

#### 핵심 수식: 답변 신뢰도(Answer Confidence)

각 $k$번째 디코딩 경로의 신뢰도를 다음과 같이 정의합니다:

$$\Delta_{k,\text{answer}} = \frac{1}{|\text{answer}|} \sum_{x_t \in \text{answer}} p(x_t^1 \mid x_{ < t}) - p(x_t^2 \mid x_{ < t})$$

여기서:
- $x_t^1$: $t$번째 디코딩 스텝에서 **1위 토큰** (최고 소프트맥스 확률)
- $x_t^2$: $t$번째 디코딩 스텝에서 **2위 토큰**
- $x_{<t}$: $t$ 이전까지 생성된 토큰 시퀀스
- 합산은 최종 **답변 토큰(answer span)**에 대해서만 수행

이 메트릭은 Jiang & Gupta (2019)의 **minimum-margin** 접근법과 유사하며, 값이 클수록 모델이 해당 답변에 더 확신을 가지고 있음을 나타냅니다.

#### 경로 집계(Path Aggregation)

단일 최고 $\Delta$ 경로 선택 대신, **가중 집계(weighted aggregation)**를 사용합니다:

$$\tilde{\Delta}_a = \sum_{k} \Delta_{k,a}$$

여기서 $\Delta_{k,a}$는 $k$번째 경로의 답변이 $a$일 때의 신뢰도입니다. 이 집계 방법은 단일 경로 선택보다 결과의 안정성을 향상시킵니다.

#### 알고리즘 요약

```
Input: 질문 Q, 탐색할 상위 k개 토큰 수
1. "Q: [question]\nA:" 형식으로 입력 구성
2. 첫 번째 디코딩 스텝에서 상위 k개 토큰 선택
3. 각 k에 대해 그리디 디코딩으로 전체 경로 생성
4. 각 경로의 답변 스팬 식별
5. Δ_{k,answer} 계산
6. 가장 높은 Δ̃_a를 가진 답변 선택
```

### 2.3 모델 구조

이 논문은 **새로운 모델 구조를 제안하지 않습니다.** 대신 기존 사전 학습된 LLM의 **디코딩 절차**를 변경합니다:

실험에 사용된 모델:
- **PaLM-2** (Anil et al., 2023): X-Small, Small, Medium, Large 4가지 스케일
- **Mistral-7B** (Jiang et al., 2023): 사전 학습 및 명령어 튜닝 버전
- **Gemma-7B** (Team et al., 2024): 사전 학습 버전

입력 형식:
```
Q: [question]
A:
```

### 2.4 성능 향상

#### 다양한 디코딩 전략과의 비교 (Mistral-7B, GSM8K)

| 방법 | GSM8K 정확도 |
|------|-------------|
| Top- $k$ 샘플링 ($k=10$) | 4.9% |
| Nucleus 샘플링 ($p=0.9$) | 6.4% |
| 빔 서치 ($b=10$) | 6.7% |
| Temperature 샘플링 ($T=0.7$) | 7.5% |
| 그리디 디코딩 | 9.9% |
| Self-consistency (CoT 프롬프트 없음, 10경로) | 12.9% |
| **CoT-decoding** ($k=10$) | **25.1%** |

#### 모델 패밀리별 성능 향상

| 모델 | 태스크 | 그리디 | CoT-decoding | 향상 |
|------|--------|--------|--------------|------|
| Mistral-7B | GSM8K | 9.9% | 25.1% | +15.2%p |
| Mistral-7B | MultiArith | 14.3% | 45.7% | +31.4%p |
| Gemma-7B | GSM8K | 15.2% | 28.2% | +13.0%p |
| PaLM-2 L | GSM8K | 34.8% | 63.2% | +28.4%p |
| PaLM-2 L | Year Parity | 57.0% | 95.0% | +38.0%p |

#### Zero-shot CoT 프롬프트와 결합 시 성능 (GSM8K 전체 테스트셋)

| 방법 | Mistral-7B | PaLM-2 L | 연산 비용 |
|------|-----------|---------|---------|
| 그리디 디코딩 | 9.9% | 34.8% | $O(1)$ |
| Zero-shot CoT 프롬프트 | 17.5% | 75.1% | $O(1)$ |
| Self-consistency + Zero-shot CoT | 39.4% | 85.3% | $O(k)$ |
| CoT-decoding(agg) + Zero-shot CoT | **48.4%** | **87.0%** | $O(k)$ |

### 2.5 한계

1. **계산 비용 증가**: $O(k)$배의 디코딩 비용 발생
2. **개방형 답변 처리의 어려움**: 최종 두 토큰의 확률 차이($\Delta$)가 개방형 답변에서 덜 정밀할 수 있음
3. **첫 번째 토큰 분기에 제한**: 현재 방법은 첫 번째 디코딩 스텝에서만 분기를 탐색
4. **태스크 복잡도에 따른 한계**: 3단계 이상의 논리 조작이 필요한 고도로 합성적인 태스크에서는 CoT 경로를 찾기 어려움
5. **상태 추적(state tracking) 취약성**: Coin Flip, Web-of-Lies 등 상태 추적 과제에서 복잡성이 높아지면 실패

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 태스크-비가지(Task-Agnostic) 방법

CoT-decoding은 특정 태스크에 특화된 프롬프트를 사용하지 않아 본질적으로 **태스크-불가지론적(task-agnostic)**입니다:

$$\text{입력} = \text{"Q: [question]}\backslash\text{nA:"}$$

이 단일 형식이 수학 추론, 상식 추론, 기호적 추론 등 **다양한 태스크에 동일하게 적용**됩니다.

### 3.2 모델 스케일에 따른 일반화

PaLM-2 실험 결과:

| 모델 크기 | GSM8K 그리디 | GSM8K CoT-decoding | Year Parity 그리디 | Year Parity CoT-decoding |
|---------|-------------|---------------------|-------------------|------------------------|
| XS | 9.0% | 17.7% | - | - |
| Small | 14.3% | 35.1% | 61.0% | 65.0% |
| Medium | 21.0% | 39.7% | 55.0% | 89.0% |
| Large | 34.8% | 63.2% | 57.0% | 95.0% |

Year Parity 태스크에서 그리디 디코딩은 모델 크기를 키워도 성능이 개선되지 않는 반면(**scaling law 실패 지점**), CoT-decoding은 모델 크기 증가에 따라 일관된 향상을 보여줍니다.

### 3.3 사전 학습 분포와 일반화의 관계

논문은 CoT 경로의 존재 여부가 **사전 학습 분포에서의 태스크 노출 빈도**와 상관관계가 있음을 발견했습니다:

- **수학, 상식 추론**: 사전 학습 데이터에 풍부 → CoT 경로 자연스럽게 존재
- **복잡한 합성 태스크** (Big-Bench-Hard): 사전 학습 데이터에 희귀 → CoT 경로 찾기 어려움

이는 McCoy et al. (2023)의 "언어 모델은 훈련된 분포에 강하게 영향받는다"는 발견과 일치합니다.

### 3.4 명령어 튜닝 모델과의 격차 해소

$$\text{PaLM-2 Large (사전학습) + CoT-decoding: 63.2\%} \approx \text{PaLM-2 Large (명령어 튜닝): 67.8\%}$$

이 결과는 **대규모 CoT 주석 데이터 없이도** 디코딩 수정만으로 명령어 튜닝 수준의 성능에 근접할 수 있음을 보여줍니다. 나아가 명령어 튜닝 모델에 CoT-decoding을 적용하면 추가적인 개선이 가능합니다:

| 방법 | Mistral-7B 사전학습 | Mistral-7B 명령어 튜닝 |
|------|------------------|----------------------|
| 그리디 | 9.9% | 31.2% |
| CoT-decoding | 25.1% (+15.2) | **38.2% (+7.0)** |

### 3.5 태스크 내재적 취약점 발견

CoT-decoding을 통해 일반화를 방해하는 **모델의 근본적 취약점**이 드러납니다:

- **상태 추적 실패**: Coin Flip에서 중간 상태를 잃어버리는 경향
- **연산 순서 오류**: Multi-step Arithmetic에서 수학적 연산 순서 대신 좌→우 순서로 계산
- 이러한 발견은 향후 **특정 취약점을 보완하는 학습 방향**을 제시합니다

---

## 4. 미래 연구에 미치는 영향과 고려사항

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 디코딩 중심의 추론 연구 패러다임 전환

CoT-decoding은 "더 좋은 프롬프트 → 더 좋은 추론"이라는 기존 패러다임에서 벗어나 **"더 스마트한 디코딩 → 더 좋은 추론"**이라는 새로운 방향을 열었습니다. 이는 다음 연구들에 영향을 줄 것입니다:

- 디코딩 전략과 추론 능력의 상관관계 연구
- 프롬프트 엔지니어링 vs. 디코딩 최적화의 비교 연구
- 사전 학습 목표(pre-training objectives)가 내재적 추론 능력에 미치는 영향 연구

#### (2) LLM 평가 방법론의 재정립

기존 벤치마크 결과가 그리디 디코딩 기반이었다면, CoT-decoding 기반의 **더 공정한 내재적 능력 평가**가 가능해집니다. 이는 모델 비교 연구의 기준점을 재설정할 필요성을 제기합니다.

#### (3) Instruction Tuning 연구에의 시사점

CoT-decoding이 명령어 튜닝 효과를 부분적으로 재현할 수 있다는 발견은:
- 명령어 튜닝의 실제 역할에 대한 이론적 탐구를 자극
- **데이터 효율적 학습(data-efficient learning)** 연구로 이어질 가능성

#### (4) 합성 디코딩 전략 연구

CoT-decoding + Self-consistency + CoT 프롬프트의 조합이 최고 성능을 보여, **여러 디코딩 전략을 조합하는 메타 디코딩(meta-decoding) 연구**가 활발해질 전망입니다.

### 4.2 앞으로 연구 시 고려할 점

#### (1) 계산 효율성

현재 CoT-decoding은 $O(k)$의 비용이 발생합니다. 실용화를 위해 다음을 고려해야 합니다:
- **투기적 디코딩(speculative decoding)**과의 결합 (Chen et al., 2023a; Leviathan et al., 2022)
- 어떤 문제에서 CoT-decoding이 필요한지 사전 판별하는 **선택적 적용 전략**

#### (2) 분기점 최적화

현재는 첫 번째 토큰에서만 분기를 탐색하지만, 논문 자체도 이 한계를 인정합니다:
- 태스크에 따라 **최적 분기 위치**가 다를 수 있음
- Year Parity 태스크에서는 중간 분기가 더 효과적임이 관찰됨
- **동적 분기점 탐색(dynamic branching point search)** 연구 필요

#### (3) 개방형 생성 태스크로의 확장

현재 방법은 답변 스팬을 명확히 식별할 수 있는 태스크에 최적화되어 있습니다:
- 자유형식(free-form) 생성에서는 $\Delta$ 메트릭의 정밀도 저하
- Burns et al. (2023)의 latent knowledge 접근과 결합하여 더 넓은 답변 공간 처리 방법 필요

#### (4) 사전 학습 분포 의존성 극복

CoT 경로의 존재가 사전 학습 분포에 강하게 의존하는 점을 극복하기 위해:
- 희귀하거나 새로운 태스크에서의 CoT 경로 생성 연구
- **합성 데이터(synthetic data)** 기반 사전 학습을 통한 다양한 CoT 패턴 주입 연구

#### (5) 이론적 기반 확립

왜 CoT 경로가 높은 답변 신뢰도와 상관관계를 가지는가에 대한 이론적 설명이 부족합니다:
- Feng et al. (2023)의 이론적 관점에서의 추가 분석
- Prystawski et al. (2023)의 "경험의 지역성(locality of experience)"과의 연관성 탐구

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 Chain-of-Thought 관련 주요 연구 연대기

```
2021 ─── Nye et al.: Scratchpad (중간 계산 활용)
2022 ─── Wei et al.: Chain-of-Thought Prompting (NeurIPS 2022)
2022 ─── Kojima et al.: Zero-shot CoT ("Let's think step by step")
2022 ─── Wang et al.: Self-Consistency (ICLR 2023)
2023 ─── Yao et al.: Tree of Thoughts (NeurIPS 2023)
2023 ─── Lightman et al.: Let's Verify Step by Step
2023 ─── Xie et al.: Self-Evaluation Guided Beam Search
2024 ─── Wang & Zhou: CoT-Decoding (본 논문)
```

### 5.2 주요 연구와의 비교

| 논문 | 방법 | 프롬프트 필요 | 파인튜닝 필요 | 추가 모델 필요 | 핵심 아이디어 |
|------|------|:---:|:---:|:---:|------|
| Wei et al. (2022) NeurIPS | Few-shot CoT | ✅ | ❌ | ❌ | Few-shot 예시로 CoT 유도 |
| Kojima et al. (2022) NeurIPS | Zero-shot CoT | ✅ | ❌ | ❌ | "Let's think step by step" |
| Wang et al. (2022) ICLR 2023 | Self-Consistency | ✅ | ❌ | ❌ | 다수결 집계 |
| Yao et al. (2023) NeurIPS | Tree of Thoughts | ✅ | ❌ | ❌ | 트리 구조 탐색 |
| Lightman et al. (2023) | Process Reward | ✅ | ✅ | ✅ | 단계별 검증 |
| Xie et al. (2023) NeurIPS | Self-Eval Beam | ✅ | ❌ | ❌ | 빔 서치 + 자체 평가 |
| **Wang & Zhou (2024)** | **CoT-Decoding** | **❌** | **❌** | **❌** | **디코딩 변경만** |

### 5.3 Self-Consistency (Wang et al., 2022)와의 비교

Self-Consistency는 동일 프롬프트에서 여러 경로를 샘플링하여 다수결로 답변을 선택합니다:

$$\text{Self-Consistency}: p(\text{answer}) = \frac{\sum_i \mathbb{1}[\text{answer}_i = \text{answer}]}{k}$$

**CoT-decoding의 차별점**:
- Self-Consistency는 **CoT 프롬프트가 있어야** 각 경로에서 CoT가 생성됨
- 프롬프트 없이 샘플링하면 모델이 직접 답변을 출력하는 경향이 강해 CoT 유도 실패
- CoT-decoding은 **첫 토큰 분기**를 통해 명시적으로 다양성을 유도

### 5.4 Tree of Thoughts (Yao et al., 2023)와의 비교

ToT는 트리 구조로 다중 추론 경로를 탐색하고 백트래킹을 허용합니다:

$$\text{ToT}: \text{탐색 비용} = O(\text{depth} \times \text{branching factor})$$

**비교**:
- ToT는 훨씬 높은 계산 비용 ( $O(k^d)$ )
- ToT는 여전히 명시적 프롬프트와 외부 평가자(evaluator) 필요
- CoT-decoding은 $O(k)$로 훨씬 효율적이며 단일 모델만 필요

### 5.5 Contrastive Decoding (Li et al., 2023a; O'Brien & Lewis, 2023)과의 비교

Contrastive Decoding은 큰 모델과 작은 모델의 로짓 차이를 활용합니다:

$$\text{CD score}(x_t) = \log p_{\text{large}}(x_t) - \log p_{\text{small}}(x_t)$$

**비교**:
- 추가적인 소형 모델 필요 vs. CoT-decoding은 단일 모델
- O'Brien & Lewis (2023)는 Contrastive Decoding이 추론 향상에 기여함을 보였으나, CoT-decoding보다 제약이 많음

### 5.6 "Let's Verify Step by Step" (Lightman et al., 2023)과의 비교

프로세스 보상 모델(PRM)을 훈련하여 각 추론 단계를 검증합니다.

**비교**:
- PRM 훈련을 위한 **대규모 단계별 주석 데이터** 필요
- 추가 보상 모델 필요
- CoT-decoding은 비지도적이며 추가 모델 불필요

---

## 참고자료 (출처)

### 주 논문
- **Wang, X. & Zhou, D. (2024).** "Chain-of-Thought Reasoning without Prompting." *arXiv:2402.10200v2*. Google DeepMind.

### 논문 내 인용 참고문헌
- Wei, J. et al. (2022). "Chain of Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022*. https://openreview.net/forum?id=_VjQlMeSB_J
- Kojima, T. et al. (2022). "Large Language Models are Zero-Shot Reasoners." *NeurIPS 2022*, vol. 35, pp. 22199-22213.
- Wang, X. et al. (2023a). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." *ICLR 2023*. https://openreview.net/forum?id=1PL1NIMMrw
- Yao, S. et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." *NeurIPS 2023*. https://openreview.net/forum?id=5Xc1ecxO1h
- Lightman, H. et al. (2023). "Let's Verify Step by Step." *arXiv:2305.20050*.
- Xie, Y. et al. (2023). "Self-Evaluation Guided Beam Search for Reasoning." *NeurIPS 2023*. https://openreview.net/forum?id=Bw82hwg5Q3
- Anil, R. et al. (2023). "PaLM 2 Technical Report." *arXiv:2305.10403*.
- Jiang, A. Q. et al. (2023). "Mistral 7B." *arXiv:2310.06825*.
- Team, G. et al. (2024). "Gemma: Open Models Based on Gemini Research and Technology." *arXiv:2403.08295*.
- Cobbe, K. et al. (2021). "Training Verifiers to Solve Math Word Problems." *arXiv:2110.14168*.
- Chung, H. W. et al. (2022). "Scaling Instruction-Finetuned Language Models." *arXiv:2210.11610*.
- McCoy, R. T. et al. (2023). "Embers of Autoregression: Understanding Large Language Models through the Problem They Are Trained to Solve." *arXiv:2309.13638*.
- Feng, G. et al. (2023). "Towards Revealing the Mystery Behind Chain of Thought: A Theoretical Perspective." *NeurIPS 2023*. https://openreview.net/forum?id=qHrADgAdYu
- Prystawski, B. et al. (2023). "Why Think Step by Step? Reasoning Emerges from the Locality of Experience." *NeurIPS 2023*. https://openreview.net/forum?id=rcXXNFVlEn
- Li, X. L. et al. (2023a). "Contrastive Decoding: Open-Ended Text Generation as Optimization." *ACL 2023*, pp. 12286-12312.
- O'Brien, S. & Lewis, M. (2023). "Contrastive Decoding Improves Reasoning in Large Language Models." *arXiv:2309.09117*.
- Jiang, H. & Gupta, M. (2019). "Minimum-Margin Active Learning." *arXiv:1906.01583*.
- Burns, C. et al. (2023). "Discovering Latent Knowledge in Language Models without Supervision." *ICLR 2023*. https://openreview.net/forum?id=ETKGuby0hcs
- Suzgun, M. et al. (2022). "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them." *arXiv:2210.09261*.
- Allen-Zhu, Z. & Li, Y. (2023). "Physics of Language Models: Part 3.2, Knowledge Manipulation." *arXiv:2309.14402*.
- Zhou, Y. et al. (2024). "DistillSpec: Improving Speculative Decoding via Knowledge Distillation." *ICLR 2024*.
- Min, S. et al. (2022). "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?" *EMNLP 2022*.
- Webson, A. & Pavlick, E. (2022). "Do Prompt-Based Models Really Understand the Meaning of Their Prompts?" *NAACL 2022*, pp. 2300-2344.
