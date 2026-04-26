# Self-Consistency Improves Chain of Thought Reasoning in Language Models

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

Wang et al. (2023, ICLR)은 기존 Chain-of-Thought(CoT) 프롬프팅에서 사용되던 **탐욕적(Greedy) 디코딩 전략**이 지닌 한계를 지적하고, 이를 대체하는 새로운 디코딩 전략인 **Self-Consistency**를 제안합니다.

핵심 직관은 다음과 같습니다:

> *"복잡한 추론 문제는 올바른 정답에 도달하는 다양한 추론 경로를 허용하며, 올바른 추론 과정들은 다양하더라도 최종 답에서 더 높은 일치도를 보인다."*

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **새로운 디코딩 전략 제안** | Greedy decoding → Sample-and-Marginalize 방식으로 대체 |
| **비지도 학습 방식** | 추가 학습, fine-tuning, 인간 주석 불필요 |
| **광범위한 실험적 검증** | 4개 LLM, 다수 벤치마크에서 일관된 성능 향상 |
| **불확실성 추정 가능성** | Consistency와 정확도의 상관관계를 통한 모델 신뢰도 추정 |
| **기존 방법 대비 우수성** | Sample-and-rank, Beam search, Ensemble 방법 모두 능가 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

CoT 프롬프팅(Wei et al., 2022)은 LLM의 추론 능력을 크게 향상시켰으나, **단일 Greedy Decoding**에 의존하는 구조적 한계를 가집니다:

- **반복성(Repetitiveness)**: 항상 동일한 추론 경로만 생성
- **국소 최적(Local Optimality)**: 탐욕적 선택이 전역 최적을 보장하지 않음
- **단일 경로의 취약성**: 추론 중 한 단계에서 오류 발생 시 전체 답 오류로 이어짐

### 2.2 제안하는 방법

Self-Consistency는 **3단계 "Sample-and-Marginalize" 절차**로 구성됩니다:

**Step 1**: CoT 프롬프팅으로 LLM에 입력  
**Step 2**: 디코더에서 다양한 추론 경로 $m$개 샘플링  
**Step 3**: 추론 경로를 주변화(marginalize)하여 가장 일관된 답 선택

#### 수식 표현

$m$개의 후보 출력이 샘플링될 때, $i$번째 출력에 대한 추론 경로 $r_i$와 최종 답 $a_i$를 생성합니다. Self-consistency의 최종 답 선택은 **다수결(Majority Vote)**:

$$\hat{a} = \arg\max_{a} \sum_{i=1}^{m} \mathbb{1}(a_i = a)$$

또한 각 $(r_i, a_i)$에 대해 **정규화된 조건부 확률**로 가중치를 부여하는 방식도 제안합니다:

$$P(\mathbf{r}_i, \mathbf{a}_i \mid \text{prompt, question}) = \exp\left(\frac{1}{K} \sum_{k=1}^{K} \log P(t_k \mid \text{prompt, question}, t_1, \ldots, t_{k-1})\right)$$

여기서:
- $t_k$: $(r_i, a_i)$의 $k$번째 토큰
- $K$: $(r_i, a_i)$의 전체 토큰 수
- 이 수식은 출력 길이에 대한 **길이 정규화(length normalization)** 적용

#### 다양한 집계 전략 비교 (PaLM-540B 기준, 논문 Table 1)

| 집계 방법 | GSM8K | MultiArith | AQuA |
|-----------|-------|-----------|------|
| Greedy decode | 56.5 | 94.7 | 35.8 |
| Weighted avg (unnormalized) | 56.3 | 90.5 | 35.8 |
| Weighted sum (normalized) | 74.1 | 99.3 | 48.0 |
| **Unweighted sum (majority vote)** | **74.4** | **99.3** | **48.3** |

→ **단순 다수결(majority vote)이 정규화된 가중합과 거의 동등**한 성능을 보이며, 실용성이 높습니다.

### 2.3 모델 구조

Self-Consistency 자체는 별도의 모델 구조가 없으며, 기존 LLM 위에 적용되는 **디코딩 전략**입니다. 실험에 사용된 언어 모델:

| 모델 | 파라미터 수 | 유형 |
|------|-----------|------|
| UL2 (Tay et al., 2022) | 20B | Encoder-Decoder |
| GPT-3 (Brown et al., 2020) | 175B | Decoder-only |
| LaMDA (Thoppilan et al., 2022) | 137B | Decoder-only |
| PaLM (Chowdhery et al., 2022) | 540B | Decoder-only |

**샘플링 방식**: Temperature sampling ($T=0.5\sim0.7$), Top-k sampling ($k=40$), Nucleus sampling 등과 호환됩니다.

### 2.4 성능 향상

#### 산술 추론 (논문 Table 2 기반)

| 모델 | 벤치마크 | CoT (Greedy) | Self-Consistency | 향상폭 |
|------|---------|-------------|-----------------|--------|
| PaLM-540B | GSM8K | 56.5 | 74.4 | **+17.9%** |
| GPT-3 (code-davinci-002) | AQuA | 39.8 | 52.0 | **+12.2%** |
| GPT-3 (code-davinci-002) | SVAMP | 75.8 | 86.8 | **+11.0%** |
| LaMDA-137B | MultiArith | 51.8 | 75.7 | **+23.9%** |

#### 상식 추론 (논문 Table 3 기반)

| 모델 | 벤치마크 | CoT (Greedy) | Self-Consistency | 향상폭 |
|------|---------|-------------|-----------------|--------|
| GPT-3 (code-davinci-002) | StrategyQA | 73.4 | 79.8 | **+6.4%** |
| GPT-3 (code-davinci-002) | ARC-challenge | 83.6 | 87.5 | **+3.9%** |

### 2.5 한계점

논문에서 명시된 한계:

1. **추가 계산 비용**: 단일 추론 대비 $m$배의 디코딩이 필요 (단, 5~10개 경로로도 대부분의 성능 이득 달성 가능)
2. **고정 답변 공간 의존**: 현재 방식은 답이 고정된 집합(fixed answer set) 문제에만 직접 적용 가능 (개방형 생성 문제에는 일관성 측도 정의 필요)
3. **사실적 근거 부족**: LLM이 잘못되거나 비사실적인 추론 경로를 생성할 수 있음 (예: StrategyQA의 인구 수치 오류)
4. **소규모 모델에서 제한적 효과**: 특정 능력이 충분한 규모에서만 발현되므로, 소규모 모델에서 효과가 상대적으로 제한

---

## 3. 모델의 일반화 성능 향상 가능성

이 섹션은 논문의 핵심 관심 사항 중 하나입니다.

### 3.1 Out-of-Distribution(OOD) 일반화

논문은 **상징적 추론(Symbolic Reasoning)** 과제에서 명시적으로 OOD 설정을 테스트합니다:

- **설정**: 프롬프트 예시는 2-letter/2-flip으로 구성, 테스트는 4-letter/4-flip으로 평가
- **결과**: PaLM-540B + Self-consistency가 이 어려운 OOD 설정에서도 유의미한 성능 향상을 보임

$$\text{Letter (4-letter OOD)}: 65.8 \xrightarrow{\text{Self-Consistency}} 70.8 \quad (+5.0\%)$$

이는 **분포 외 입력에 대한 강건성**을 간접적으로 시사합니다.

### 3.2 다양한 태스크 유형으로의 일반화

Self-consistency는 단순 수학 문제를 넘어 다음 영역에서 모두 성능을 향상시켰습니다:

- 산술 추론 (AddSub, MultiArith, ASDiv, AQuA, SVAMP, GSM8K)
- 상식 추론 (CommonsenseQA, StrategyQA, ARC)
- 상징적 추론 (Last letter concatenation, Coinflip)
- 자연어 추론 (ANLI, e-SNLI, RTE)
- 질의응답 (BoolQ, HotpotQA)

이 광범위한 태스크 커버리지는 **태스크-독립적(task-agnostic) 일반화 능력**을 시사합니다.

### 3.3 CoT가 성능을 저하시키는 경우에도 일반화

Ye & Durrett (2022)이 지적한 "CoT가 오히려 성능을 저하시키는" 태스크(ANLI-R1, e-SNLI, RTE)에서도 Self-consistency는 **Standard prompting보다 우수한 성능**을 달성합니다:

| 방법 | ANLI R1 | e-SNLI | RTE |
|------|---------|--------|-----|
| Standard-prompting | 69.1 | 85.8 | 84.8 |
| CoT-prompting | 68.8 | 81.0 | 79.1 |
| **Self-consistency** | **78.5** | **88.4** | **86.3** |

→ CoT의 부정적 효과를 **능동적으로 극복**함으로써 더 넓은 범위의 태스크에서 사용 가능성을 보여줍니다.

### 3.4 불완전한 프롬프트에 대한 강건성

임의의 숫자로 추론 경로를 교란한 불완전한 프롬프트(imperfect prompts) 실험:

$$\text{Imperfect CoT (LaMDA-137B)}: 14.9 \xrightarrow{+\text{Self-Consistency (40 paths)}} 23.4$$

이는 프롬프트 품질에 대한 **강건성(Robustness)**을 의미하며, 일반화 성능의 중요한 지표입니다.

### 3.5 모델 크기에 따른 일반화 스케일링

$$\text{Self-consistency 효과} \propto \text{모델 크기}$$

소규모 모델(UL2-20B)에서는 +3 ~ 6%, 대규모 모델(LaMDA-137B, GPT-3)에서는 +9 ~ 23%의 향상을 보입니다. 이는 **모델 능력이 충분할 때 Self-consistency의 다양성 활용 능력이 극대화**됨을 시사합니다.

### 3.6 불확실성 추정을 통한 일반화 신뢰성

논문 Figure 5에 따르면 **Consistency(%)와 정확도 간의 강한 양의 상관관계**가 관찰됩니다:

$$\text{Consistency} = \frac{\max_a \sum_{i=1}^{m} \mathbb{1}(a_i = a)}{m} \times 100\%$$

이를 통해 모델이 "모를 때를 아는" 능력, 즉 **인식론적 불확실성(epistemic uncertainty) 추정**이 가능해지며, 이는 분포 외 입력에 대한 신뢰할 수 있는 감지 메커니즘으로 활용될 수 있습니다.

### 3.7 Zero-shot CoT로의 일반화

Kojima et al. (2022)의 Zero-shot CoT와 결합 시:

$$\text{Zero-shot CoT (PaLM-540B, GSM8K)}: 43.0 \xrightarrow{+\text{Self-Consistency}} 69.2 \quad (+26.2\%)$$

이는 Few-shot 설정 없이도 Self-consistency가 강력한 성능 향상을 이끌어냄을 보여줍니다.

---

## 4. 해당 논문이 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 미래 연구에 미치는 영향

#### (1) 추론 강화 방법론의 패러다임 전환
Self-consistency는 LLM의 추론 능력 강화를 위해 **모델 자체를 수정하지 않고 디코딩 전략을 개선**하는 방향성을 확립했습니다. 이는 이후 다음과 같은 연구 흐름을 촉발했습니다:

- **Tree of Thoughts** (Yao et al., 2023): 선형 추론 경로를 트리 구조로 확장하여 더 체계적인 탐색 수행
- **Graph of Thoughts** (Besta et al., 2023): 추론 단위를 그래프 구조로 연결
- **Least-to-Most Prompting** (Zhou et al., 2022): 복잡한 문제를 하위 문제로 분해

#### (2) 앙상블 방법론의 재해석
단일 모델에서 다양성을 추출하는 "자기 앙상블(self-ensemble)" 개념은, 추론 다양성이 정확도 향상에 기여한다는 실증적 근거를 제공했습니다. 이는 **데이터 증강(data augmentation)** 및 **테스트 타임 컴퓨트(test-time compute)** 연구에 영향을 미쳤습니다.

#### (3) LLM 불확실성 추정 연구 촉진
Consistency와 정확도의 상관관계는 LLM의 **교정(calibration)** 및 **불확실성 정량화** 연구에 새로운 방향을 제시했습니다.

#### (4) 테스트 타임 스케일링(Test-Time Compute Scaling) 연구
OpenAI의 o1, o3 시리즈 등 **테스트 타임 연산을 증가시켜 성능을 향상**시키는 연구 방향과 직접 연결됩니다. Self-consistency는 이 패러다임의 초기 실증적 근거를 제공했습니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 논문 내 참고 문헌 및 제가 알고 있는 관련 연구들을 기반으로 작성되었습니다. 단, 2023년 이후 연구의 구체적 수치는 논문 원문에 없으므로 개략적 비교로 제시합니다.

| 연구 | 방법 | 핵심 아이디어 | Self-consistency 대비 |
|------|------|-------------|----------------------|
| **Wei et al. (2022)** Chain-of-Thought | CoT Prompting | 단계별 추론 경로 생성 | Self-consistency의 기반, Greedy decoding 사용으로 한계 존재 |
| **Kojima et al. (2022)** Zero-shot CoT | "Let's think step by step" | 예시 없이 CoT 유도 | Self-consistency와 결합 시 +26.2% (GSM8K) |
| **Cobbe et al. (2021)** Training Verifiers | 별도 검증기 훈련 | 솔루션 랭킹을 위한 verifier 학습 | 추가 학습 데이터 필요 vs. Self-consistency는 비지도 방식 |
| **Yao et al. (2023)** Tree of Thoughts | 트리 탐색 | BFS/DFS로 추론 경로 탐색 | 더 체계적 탐색이지만 훨씬 높은 계산 비용 |
| **Lightman et al. (2023)** Process Reward Models | 과정 보상 모델 | 추론 각 단계에 보상 부여 | 인간 주석 필요 vs. Self-consistency는 주석 불필요 |
| **Ye & Durrett (2022)** | CoT 신뢰성 분석 | CoT가 성능 저하시킬 수 있음을 지적 | Self-consistency로 해당 문제 해결 가능함을 본 논문에서 실증 |

---

### 4.3 앞으로 연구 시 고려할 점

#### (1) 계산 효율성 문제
$m$개의 추론 경로 샘플링은 계산 비용을 $m$배 증가시킵니다. 향후 연구에서 고려할 사항:

$$\text{비용} = m \times \text{단일 추론 비용}$$

- **적응형 샘플링(Adaptive Sampling)**: 신뢰도가 높을 때 일찍 종료하는 전략 개발
- **경량화 모델과의 결합**: 소규모 모델에서 더 많은 경로를 샘플링하여 대규모 모델 단일 추론과 경쟁

#### (2) 개방형 생성 태스크로의 확장
현재 방법은 **고정된 답변 공간**을 전제합니다. 자유 텍스트 생성 문제에서의 일관성 측도 정의 연구가 필요합니다:
- 두 답변의 의미적 동등성 판단
- 생성된 텍스트 간 모순 감지

#### (3) 환각(Hallucination) 및 사실적 근거 문제
LLM이 생성하는 추론 경로 자체가 비사실적일 수 있습니다. 다수결이 **"잘못된 합의"** 를 형성할 위험:

$$P(\text{잘못된 합의}) > 0 \text{ when } P(\text{동일한 오류}) \text{ 가 높을 때}$$

- 외부 지식 베이스 통합
- 사실 검증(fact-checking) 모듈 결합

#### (4) 편향 증폭 가능성
다수결 방식은 모델의 기존 편향(bias)을 **증폭**시킬 수 있습니다. 공정성과 안전성 측면에서 추가 연구 필요합니다.

#### (5) 소규모 모델에서의 적용 가능성
본 논문의 결과는 주로 100B+ 규모 모델에서 두드러집니다. **소규모 모델(≤7B)에서의 효과성**을 높이는 방법 연구가 필요합니다:
- Knowledge distillation과 결합
- Fine-tuning을 통해 Self-consistency의 이점을 단일 패스로 내재화

#### (6) Self-consistency를 통한 데이터 생성 활용
논문 저자들이 미래 연구로 제안한 방향: Self-consistency로 생성된 올바른 추론 경로를 학습 데이터로 활용하여 **모델 자체를 fine-tuning**함으로써 단일 추론 패스에서도 높은 정확도 달성.

#### (7) 다중 모달(Multimodal) 확장
이미지-텍스트, 코드 생성 등 다양한 모달리티에서 추론 경로의 다양성과 일관성을 활용하는 연구 방향.

---

## 참고 자료

**주요 참고 논문 (본문 내 직접 인용 논문들):**

1. **Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E. H., Narang, S., Chowdhery, A., & Zhou, D. (2023).** Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*. arXiv:2203.11171v4

2. **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022).** Chain of Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*. arXiv:2201.11903

3. **Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022).** Large Language Models are Zero-Shot Reasoners. *NeurIPS 2022*.

4. **Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., ... & Schulman, J. (2021).** Training Verifiers to Solve Math Word Problems. arXiv:2110.14168

5. **Chowdhery, A., Narang, S., Devlin, J., et al. (2022).** PaLM: Scaling Language Modeling with Pathways. arXiv:2204.02311

6. **Brown, T., Mann, B., Ryder, N., et al. (2020).** Language Models are Few-Shot Learners. *NeurIPS 2020*.

7. **Ye, X., & Durrett, G. (2022).** The Unreliability of Explanations in Few-Shot Prompting for Textual Reasoning. *NeurIPS 2022*.

8. **Tay, Y., Dehghani, M., Tran, V. Q., et al. (2022).** Unifying Language Learning Paradigms. arXiv:2205.05131

9. **Thoppilan, R., De Freitas, D., Hall, J., et al. (2022).** LaMDA: Language Models for Dialog Applications. arXiv:2201.08239

10. **Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020).** The Curious Case of Neural Text Degeneration. *ICLR 2020*.
