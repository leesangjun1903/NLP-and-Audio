# Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

RLHF로 파인튜닝된 언어모델(RLHF-LM)은 조건부 확률(log probability)이 **심각하게 과신(overconfident)** 되어 있지만, 모델에게 직접 신뢰도를 언어로 표현(verbalize)하도록 요청하면 훨씬 더 잘 보정(calibrated)된 신뢰도 점수를 얻을 수 있다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| 최초 체계적 평가 | ChatGPT, GPT-4, Claude 등 주요 RLHF-LM의 캘리브레이션을 포괄적으로 비교 |
| Verbalized Confidence | 모델이 토큰 공간에서 수치/언어적 신뢰도를 직접 표현하는 방법 제안 |
| 다중 가설 생성 | 여러 후보 답변 생성 후 신뢰도 부여 시 캘리브레이션 향상 확인 |
| 심리학적 영감 | "Considering the Opposite" (Lord et al., 1985) 전략을 LLM에 적용 |
| ECE 50% 감소 | 최적 조합(Verbalized + Temperature Scaling) 시 ECE를 상대적으로 약 50% 감소 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

RLHF 파인튜닝은 인간이 선호하는 답변에 확률 질량(probability mass)을 집중시키는 방향으로 모델을 학습시킨다. 이 과정에서:

- 조건부 확률 $p(y|x)$가 정답 가능성을 제대로 반영하지 못함
- 모델이 **체계적으로 과신**하게 됨 (그림 2: RLHF 후 ECE 악화)
- 실제 사용되는 RLHF-LM(ChatGPT 등)의 log probability 기반 신뢰도가 신뢰 불가능

**핵심 문제**: *RLHF-LM에서 어떻게 잘 보정된 신뢰도 점수를 추출할 것인가?*

### 2-2. 평가 지표 (수식 포함)

**① Expected Calibration Error (ECE)**

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|^2$$

여기서:
- $B_m$: $m$번째 신뢰도 구간(bin)에 속하는 예측 집합
- $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}[\hat{y}_i = y_i]$: 구간 내 정확도
- $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$: 구간 내 평균 신뢰도

**② Temperature Scaling**

온도 매개변수 $\beta$를 사용하여 NLL을 최소화하는 방향으로 스케일링:

$$\tilde{p}_i \propto p_i^{\beta}$$

$$\beta^* = \arg\min_{\beta} \left[ -\sum_{i} \log \tilde{p}_i^{(y_i)} \right]$$

**③ Brier Score (BS)**

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2$$

여기서 $y_i \in \{0, 1\}$은 정답 여부 레이블.

**④ AUC (Selective Accuracy-Coverage)**

신뢰도 임계값 $\tau$에 따른 선택적 분류 성능:

$$\text{AUC} = \int_0^1 \text{Acc}(\tau) \, d\text{Cov}(\tau)$$

### 2-3. 제안하는 방법들

논문에서 비교한 방법들은 크게 두 범주로 나뉜다:

#### (A) 조건부 확률 기반 (Baseline)

**Label prob.**

$$c = p(y|x) \approx \frac{1}{n}\sum_{j=1}^{n} \mathbf{1}[\hat{y}^{(j)} = y^*], \quad n=10 \text{ samples}$$

**'Is True' prob.**

$$c = p(\text{True} | x, \hat{y})$$

샘플링으로 추정: $\hat{y} \sim p(\cdot|x)$를 먼저 샘플링 후, "이 답이 맞는가?"를 확률로 추정.

#### (B) Verbalized Confidence (제안 방법)

| 방법 | 설명 |
|------|------|
| **Verb. 1S top-k** | 1단계: k개 추측과 각 확률을 동시에 생성 |
| **Verb. 2S top-k** | 2단계: 1단계에서 답변 생성 → 2단계에서 확률 부여 |
| **Verb. 2S CoT** | Chain-of-Thought 후 확률 부여 |
| **Ling. 1S-human** | 언어 표현("Highly Likely" 등) → 설문 기반 확률 매핑 |
| **Ling. 1S-opt.** | 언어 표현 → hold-out 데이터로 최적화된 확률 매핑 |

**Ling. 1S-opt. 최적화**:

$$p^*(\ell) = \frac{1}{|D_\ell|} \sum_{i: \ell_i = \ell} \mathbf{1}[\hat{y}_i = y_i]$$

여기서 $\ell$은 특정 언어 표현(예: "Likely"), $D_\ell$은 해당 표현을 사용한 데이터 집합.

### 2-4. 모델 구조

별도의 새로운 모델 아키텍처를 제안하지 않고, **기존 RLHF-LM에 프롬프트 엔지니어링**을 적용:

```
실험 대상 모델:
- gpt-3.5-turbo (ChatGPT)
- gpt-4 (GPT-4)
- claude-1, claude-2
- Llama-2-70b-chat
```

**핵심 구조적 아이디어**: 언어모델이 이미 학습 중 신뢰도 표현 능력을 갖추었으며, 이를 적절한 프롬프트로 유도(elicit)할 수 있음.

### 2-5. 성능 향상

**TriviaQA에서 gpt-3.5-turbo 결과 (Table 1)**:

| 방법 | ECE | ECE-t |
|------|-----|-------|
| Label prob. (baseline) | 0.140 | 0.097 |
| Verb. 1S top-2 | **0.050** | **0.053** |
| Ling. 1S-opt. | 0.058 | 0.066 |

→ ECE 기준 약 **64% 상대적 감소** (0.140 → 0.050)

**SciQ에서 (Table 1)**:

| 방법 | ECE |
|------|-----|
| Label prob. | 0.256 |
| Verb. 1S top-4 | **0.065** |

→ ECE 기준 약 **75% 감소**

**TruthfulQA에서**:

| 방법 | ECE |
|------|-----|
| Label prob. | 0.451 |
| Ling. 1S-opt. | **0.125** |

→ ECE 기준 약 **72% 감소**

### 2-6. 한계

1. **도메인 제한**: 사실 회상(factual recall) 중심의 QA 태스크에만 집중; 추론(reasoning), 수학 등에의 적용 가능성 미검증
2. **단답형 한계**: 장문 생성(long-form generation)에의 확장 미검증
3. **모델 투명성 부재**: 클로즈드 소스 모델의 내부 구조 불명으로 분석 한계
4. **모델 간 일관성 부족**: Llama-2-70B-Chat은 GPT/Claude 대비 verbalized calibration 개선 효과가 불일치
5. **프롬프트 민감성**: 1단계/2단계 방식 간 캘리브레이션 결과가 크게 다를 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 캘리브레이션과 일반화의 관계

잘 보정된 신뢰도는 일반화 성능과 밀접하게 연결된다:

$$\text{잘 보정된 모델}: \quad \mathbb{P}(\hat{y} = y \,|\, \hat{p} = p) = p, \quad \forall p \in [0,1]$$

이 조건이 만족될 때:
- **선택적 예측(selective prediction)**: 낮은 신뢰도 예측을 기각하여 실제 사용 환경에서 오류율 감소
- **도메인 이동(domain shift) 탐지**: 신뢰도 분포 변화로 분포 외(OOD) 입력 탐지 가능
- **앙상블 효과**: 다중 가설 생성(top-k) 자체가 일종의 암묵적 앙상블 역할

### 3-2. 다중 가설 생성의 일반화 효과

**Verb. 1S top-k**에서 $k$가 증가할수록 캘리브레이션이 개선되는 현상:

$$k=1: \text{ECE}=0.234 \rightarrow k=4: \text{ECE}=0.065 \quad (\text{SciQ, gpt-3.5-turbo})$$

이는 심리학의 "Considering the Opposite" 전략(Lord et al., 1985)과 일치하며, 모델이:

$$p(\text{correct}) = \frac{p_{\text{top-1}}}{\sum_{j=1}^{k} p_j} \cdot \mathbf{1}[\hat{y}_1 = y]$$

형태로 상대적 신뢰도를 재조정하는 효과를 낳는다. 이는 다양한 도메인에서 더 robust한 신뢰도 추정으로 이어질 가능성이 있다.

### 3-3. TruthfulQA에서의 일반화 시사점

TruthfulQA는 인간의 일반적 오해를 테스트하는 **적대적(adversarial) 데이터셋**으로, 이 환경에서 verbalized calibration의 개선 효과가 가장 두드러졌다:

- GPT-4: Label prob. ECE=0.445 → Ling. 1S-opt. ECE=**0.082** (82% 감소)

이는 verbalized confidence가 단순 사실 회상을 넘어 **개념적으로 어려운 질문**에서도 일반화 가능성이 있음을 시사한다.

### 3-4. 오픈소스 모델에서의 일반화 제한

Llama-2-70B-Chat(오픈소스)에서는 개선 효과가 불일치:

```
SciQ - Label prob. ECE: 0.266
SciQ - Verb. 1S top-4 ECE: 0.105 (개선)
SciQ - Ling. 1S-human ECE: 0.071 (개선)

TriviaQA - Label prob. ECE: 0.151
TriviaQA - Ling. 1S-human ECE: 0.179 (오히려 악화)
```

→ **일반화 성능은 RLHF 훈련 품질 및 모델 규모에 크게 의존**함을 보여줌.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4-1. 향후 연구에 미치는 영향

#### ① 신뢰도 추출 패러다임 전환
기존: log probability 중심 → **제안: verbalized confidence 중심**

이는 클로즈드 API 모델에서도 캘리브레이션 연구가 가능하다는 것을 입증하여, 향후 연구의 실험 설계 방식에 직접적인 영향을 미친다.

#### ② RLHF 훈련 목표 재설계
$$\mathcal{L}_{\text{RLHF}} = \mathbb{E}[r(x, y)] - \beta \cdot \text{KL}[\pi_\theta || \pi_{\text{ref}}]$$

현재 RLHF 목표는 캘리브레이션을 명시적으로 고려하지 않는다. 향후 연구에서:

$$\mathcal{L}_{\text{RLHF+Cal}} = \mathbb{E}[r(x, y)] - \beta \cdot \text{KL}[\pi_\theta || \pi_{\text{ref}}] - \lambda \cdot \text{ECE}(\pi_\theta)$$

형태의 캘리브레이션 정규화 항 추가를 고려할 수 있다.

#### ③ 할루시네이션 감소 연구와의 연계
잘 보정된 신뢰도는 모델이 "모른다"고 적절히 표현할 수 있게 하여, 할루시네이션 감소 연구(Factuality, Honesty)와 직접 연결된다.

#### ④ 에이전트 시스템에서의 활용
LLM 기반 에이전트가 자신의 불확실성을 정확히 표현한다면:
- 사람 전문가에게 적절히 위임(deferral)
- 멀티 에이전트 시스템에서 신뢰도 기반 협업

### 4-2. 앞으로 연구 시 고려할 점

| 연구 방향 | 세부 고려사항 |
|-----------|---------------|
| **장문 생성** | 단답형 QA를 넘어 요약, 코드 생성 등에서의 신뢰도 측정 방법 |
| **추론 태스크** | 수학, 논리 추론에서 verbalized calibration 적용 가능성 |
| **프롬프트 강건성** | 프롬프트 변형에 따른 신뢰도 변동성 최소화 방법 |
| **다국어 일반화** | 영어 외 언어에서의 verbalized calibration 효과 검증 |
| **파인튜닝 방향** | verbalized calibration을 목표로 한 RLHF/SFT 훈련 방법 설계 |
| **OOD 탐지** | 신뢰도 기반 분포 외 데이터 탐지 시스템 구축 |
| **공정성** | 특정 인구통계/도메인에서의 신뢰도 편향 분석 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 내용 | 본 논문과의 관계 |
|------|------|-----------|-----------------|
| **Kadavath et al., "Language Models (Mostly) Know What They Know"** | 2022 | Claude 기반 모델의 자기 평가 능력 분석; "Is True" prob. 방법 제안 | 본 논문의 baseline 방법 출처; RLHF-LM으로 확장 |
| **Lin et al., "Teaching Models to Express Their Uncertainty in Words"** | 2022 | 모델이 언어적 불확실성 표현을 학습하도록 파인튜닝 | 본 논문은 파인튜닝 없이 프롬프팅만으로 달성 |
| **Kuhn et al., "Semantic Uncertainty"** | 2023 | 의미론적 불변성을 활용한 불확실성 추정 (ICLR 2023) | 자연어 생성에서의 불확실성; 본 논문은 QA 신뢰도에 집중 |
| **Xiao et al., "Uncertainty Quantification with Pre-trained LMs"** | 2022 | 대규모 사전학습 LM + temperature scaling의 우수한 캘리브레이션 확인 | 순수 사전학습 모델 중심; 본 논문은 RLHF-LM으로 확장 |
| **Mielke et al., "Reducing Conversational Agents' Overconfidence through Linguistic Calibration"** | 2022 | 대화 모델의 과신을 언어적 캘리브레이션으로 감소 | 유사한 방향이나 대화 모델에 한정; 본 논문은 RLHF-LM 전반 |
| **Zhou et al., "Navigating the Grey Area: Expressions of Overconfidence and Uncertainty in LMs"** | 2023 | LM의 과신/불확실성 표현 패턴 분석 | 본 논문의 related work; 언어 표현 분석 심화 |
| **Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT)** | 2022 | RLHF 파인튜닝 방법론 (NeurIPS 2022) | RLHF가 캘리브레이션에 미치는 악영향의 원인 |
| **Guo et al., "On Calibration of Modern Neural Networks"** | 2017 | ECE 지표 및 temperature scaling 제안 | 본 논문의 평가 지표와 스케일링 방법의 원출처 |

### 비교 분석 종합

```
[패러다임 변화]
2017-2020: 분류 모델 캘리브레이션 (Guo et al.)
     ↓
2020-2022: 사전학습 LM 캘리브레이션 (Kadavath, Xiao et al.)
     ↓
2022-2023: RLHF-LM 캘리브레이션 ← 본 논문의 위치
     ↓
2023-현재: 에이전트/장문생성 신뢰도, 멀티모달 캘리브레이션 (미래 방향)
```

**본 논문의 차별점**: 기존 연구들이 사전학습 모델 또는 파인튜닝을 통한 캘리브레이션 개선에 집중한 반면, 본 논문은 **파인튜닝 없이 프롬프팅만으로** 실용적인 RLHF-LM의 캘리브레이션을 개선할 수 있음을 최초로 체계적으로 입증하였다.

---

## 참고자료

- **주 논문**: Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., & Manning, C. D. (2023). "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback." arXiv:2305.14975v2.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks." ICML 2017.
- Kadavath, S., et al. (2022). "Language Models (Mostly) Know What They Know." arXiv:2207.05221.
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation." ICLR 2023.
- Lin, S., Hilton, J., & Evans, O. (2022a). "Teaching Models to Express Their Uncertainty in Words." TMLR.
- Lin, S., Hilton, J., & Evans, O. (2022b). "TruthfulQA: Measuring How Models Mimic Human Falsehoods." ACL 2022.
- Mielke, S. J., Szlam, A., Dinan, E., & Boureau, Y. (2022). "Reducing Conversational Agents' Overconfidence through Linguistic Calibration." TACL.
- Ouyang, L., et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." NeurIPS 2022.
- Xiao, Y., et al. (2022). "Uncertainty Quantification with Pre-trained Language Models: A Large-Scale Empirical Analysis." EMNLP 2022 Findings.
- Zhou, K., Jurafsky, D., & Hashimoto, T. (2023). "Navigating the Grey Area: Expressions of Overconfidence and Uncertainty in Language Models." arXiv.
- Lord, C., Lepper, M., & Preston, E. (1985). "Considering the Opposite: A Corrective Strategy for Social Judgment." Journal of Personality and Social Psychology.
- OpenAI. (2023). "GPT-4 Technical Report."
