# Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **단 하나의 트랜스포머 레이어만 학습해도 전체 파라미터 RL 학습(full-parameter RL training)의 성능 향상 대부분을 회복할 수 있으며, 경우에 따라서는 이를 능가할 수 있다.**

기존 RL 포스트 트레이닝 방법론은 모든 레이어를 균일하게 업데이트한다는 암묵적 가정을 채택하고 있었습니다. 이 논문은 이 가정에 정면으로 도전하며, RL 적응(adaptation)이 트랜스포머 레이어 간에 **극도로 불균일하게 분포**한다는 사실을 실증적으로 증명합니다.

### 주요 기여

| 기여 항목 | 내용 |
|:---|:---|
| **RL 적응 집중 현상 발견** | 단일 레이어 학습으로 전체 RL 이득의 최대 114%까지 회복 가능 |
| **Layer Contribution 메트릭 도입** | 각 레이어의 RL 기여도를 정량화하는 새로운 척도 제안 |
| **중간 레이어 집중 구조 발견** | 고기여도 레이어가 항상 트랜스포머 중간 부분에 집중됨을 확인 |
| **레이어 인식 학습 전략 제안** | 발견된 구조를 활용하여 표준 전체 파라미터 RL을 능가하는 실용적 전략 개발 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

RL 포스트 트레이닝은 LLM의 수학적 추론, 코드 생성, 에이전트 의사결정 등에서 핵심적인 역할을 수행하고 있습니다. 그러나 기존 연구들은 다음과 같은 근본적인 질문을 외면해왔습니다:

> **"RL로 인한 성능 향상은 네트워크 내 어디서 실제로 발생하는가?"**

구체적으로, 다음의 문제들이 미해결 상태였습니다:

- RL 적응이 모든 레이어에 균일하게 분포하는가, 아니면 특정 레이어에 집중되는가?
- 레이어별 기여도 패턴이 모델 패밀리, 알고리즘, 태스크 간에 일관성이 있는가?
- 이러한 구조적 이해를 바탕으로 RL 학습을 개선할 수 있는가?

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 기반 알고리즘: GRPO (Group Relative Policy Optimization)

RLVR은 언어 모델 정책 $\pi_\theta$를 최적화하여 검증 가능한 보상을 최대화합니다. 각 프롬프트 $x$에 대해 $G$개의 응답을 샘플링하고, 그룹 정규화된 어드밴티지를 계산합니다:

$$\hat{A}_i = \frac{r(x, y_i) - \text{mean}\left(\{r(x, y_j)\}_{j=1}^G\right)}{\text{std}\left(\{r(x, y_j)\}_{j=1}^G\right)} \tag{1}$$

이후 클리핑된 서로게이트 목적함수를 최대화하여 정책을 업데이트합니다:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{x, \{y_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min\left(\rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\right) - \beta\, D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \right) \right] \tag{2}$$

여기서:
- $\rho_i = \pi_\theta(y_i|x) / \pi_{\theta_{\text{old}}}(y_i|x)$: 중요도 샘플링 비율
- $\beta$: KL 패널티 계수
- $\pi_{\text{ref}}$: 참조 정책 (RL 이전 초기 모델)

#### 2.2.2 단일 레이어 학습 프레임워크

$L$개의 트랜스포머 레이어 $\{\theta_0, \theta_1, \ldots, \theta_{L-1}\}$을 가진 LLM에서, 레이어 $k$만을 학습하는 방식은 다음과 같습니다:

$$\theta_k \leftarrow \theta_k - \alpha\, \nabla_{\theta_k} \mathcal{L}_{\text{GRPO}}(\theta), \quad \text{all other parameters frozen.} \tag{3}$$

- 그래디언트 $\nabla_{\theta_k} \mathcal{L}_{\text{GRPO}}(\theta)$는 전체 네트워크를 통한 역전파로 계산되지만, **파라미터 업데이트는 오직 레이어 $k$에만 제한**됩니다.
- PyTorch에서는 타겟 레이어를 제외한 모든 파라미터에 `requires_grad=False`를 설정하여 구현합니다.

#### 2.2.3 Layer Contribution (레이어 기여도) 메트릭

레이어 $k$의 기여도를 정량화하는 핵심 메트릭:

$$\mathcal{C}(k) = \frac{S_k - S_{\text{base}}}{S_{\text{full}} - S_{\text{base}}} \tag{4}$$

- $S_k$: 레이어 $k$만 학습한 모델의 인도메인(in-domain) 성능
- $S_{\text{base}}$: RL 학습 이전 기본 모델의 성능
- $S_{\text{full}}$: 전체 파라미터 GRPO 학습 후 성능

| $\mathcal{C}(k)$ 값 | 해석 |
|:---|:---|
| $= 1.0$ | 단일 레이어 학습이 전체 파라미터 RL 이득과 동일 |
| $> 1.0$ | 단일 레이어 학습이 전체 파라미터 RL 능가 |
| $\approx 0$ | 해당 레이어가 RL 신호를 거의 흡수하지 못함 |
| $< 0$ | 해당 레이어 단독 학습 시 기본 모델보다 성능 저하 |

---

### 2.3 모델 구조 및 실험 설정

#### 실험 모델 요약

| 모델 | 패밀리 | 파라미터 | 레이어 수 | RL 알고리즘 | 태스크/데이터셋 |
|:---|:---|:---|:---|:---|:---|
| Qwen3-1.7B-Base | Qwen3 | 1.7B | 28 | GRPO | Math / NuminaMath-CoT |
| Qwen3-4B-Base | Qwen3 | 4B | 36 | GRPO | Math / NuminaMath-CoT |
| Qwen3-8B-Base | Qwen3 | 8B | 36 | GRPO | Math / NuminaMath-CoT |
| Qwen2.5-Math-1.5B | Qwen2.5 | 1.5B | 28 | Dr. GRPO | Math / MATH |
| Qwen2.5-1.5B-Instruct | Qwen2.5 | 1.5B | 28 | GiGPO | Agentic / ALFWorld |
| Qwen2.5-3B-Instruct | Qwen2.5 | 3B | 36 | GiGPO | Agentic / ALFWorld |
| DeepSeek-Distilled-Qwen-7B | Qwen2.5 | 7B | 28 | GRPO | Math / Skywork |

#### 공정 비교 프로토콜

- 전체 파라미터 기준선의 학습률은 $\{1 \times 10^{-6}, 3 \times 10^{-6}, 5 \times 10^{-6}, 1 \times 10^{-5}\}$ 중 최적값 선택
- 단일 레이어 학습에도 동일한 학습률($5 \times 10^{-6}$) 적용
- 배치 크기, KL 계수, 클립 범위, 에폭 수는 모두 동일하게 유지

---

### 2.4 성능 향상

#### 주요 실험 결과

**Qwen3 모델군 수학 평균 성능 비교:**

| 모델 | 기본 모델 | 전체 RL | 최고 단일 레이어 | Best Guided | 추가 이득 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Qwen3-1.7B | 44.1 | 50.8 | 51.8 (+1.0) | 53.7 | +43% |
| Qwen3-4B | 52.2 | 63.0 | 64.3 (+1.3) | 65.9 | +27% |
| Qwen3-8B | 58.0 | 66.4 | 67.1 (+0.7) | 69.1 | +32% |

**레이어 기여도 통계 요약 (7개 모델 전체):**

| 모델 | 최고 $\mathcal{C}$ | 최저 $\mathcal{C}$ | $\mathcal{C} \geq 1.0$ 레이어 수 | 중간 레이어 집중 |
|:---|:---:|:---:|:---:|:---:|
| Qwen3-1.7B-Base | 1.14 | 0.28 | 5/28 | ✓ |
| Qwen3-4B-Base | 1.06 | 0.66 | 4/36 | ✓ |
| Qwen3-8B-Base | 1.07 | −0.51 | 4/36 | ✓ |
| Qwen2.5-Math-1.5B | 1.01 | 0.42 | 2/28 | ✓ |
| Qwen2.5-1.5B-Instruct | 1.02 | 0.25 | 1/8† | ✓ |
| Qwen2.5-3B-Instruct | 1.01 | 0.17 | 1/11† | ✓ |
| DeepSeek-Distilled-Qwen-7B | 1.05 | 0.33 | 2/8† | ✓ |

#### 레이어 인식 학습 전략 성능 (§4)

1. **레이어 적응적 학습률**: 고기여도 상위 $k$개 레이어의 학습률을 $1 \times 10^{-5}$로 증가
   - Qwen3-8B-Base: $66.43 \to 67.42$ (+0.99)

2. **선택적 레이어 학습**: 상위 $k$개 레이어만 학습
   - Qwen3-8B-Base (Only B10): $66.43 \to 69.11$ (+2.68, 전체 RL 이득의 32% 추가 향상)

3. **휴리스틱 중간 레이어 선택**: 프로파일링 없이 위치 기반 중간 $k$개 레이어 선택
   - 모델 $L$개 레이어 중 $\lfloor L/2 - k/2 \rfloor$부터 $\lfloor L/2 + k/2 \rfloor$까지 선택
   - 전체 파라미터 기준선 대비 일관적으로 향상

4. **앙상블 (다수결 투표)**: 상위 7개 레이어 학습 모델의 예측을 다수결 투표로 결합
   - OlympiadBench: 기본 모델 18.8% → 전체 RL 26.9% → 단일 최고 레이어 28.3% → **레이어×7 투표 33.6%**

---

### 2.5 한계점

논문이 명시적으로 인정한 한계와 추가적으로 관찰되는 한계:

| 한계 유형 | 내용 |
|:---|:---|
| **검증 범위 제한** | 레이어 인식 학습 전략이 수학적 추론에만 검증됨; 코딩·에이전트 태스크 확장 미완 |
| **이론적 설명 부재** | 왜 중간 레이어가 RL 적응에 불균형적으로 중요한지에 대한 이론적 이해 부재 |
| **모델 스케일 제한** | 실험이 최대 8B 파라미터로 제한; 더 큰 모델(70B+)에서의 검증 필요 |
| **프로파일링 비용** | 최적 레이어 선택을 위해 모든 레이어를 개별 학습하는 초기 프로파일링 비용 발생 |
| **메트릭 의존성** | $\mathcal{C}(k)$가 특정 학습 설정에 상대적으로 정의되어 절대적 이식성 한계 가능 |
| **앙상블의 실용성** | 레이어별 모델 앙상블은 분석 도구로는 유효하나, 실제 배포 환경에서 다수 모델 유지 비용 높음 |

---

## 3. 일반화 성능 향상 가능성 (핵심 중점 분석)

이 논문에서 일반화(generalization)와 관련된 발견은 특히 주목할 만합니다.

### 3.1 In-Domain → Out-of-Distribution 일반화

논문은 $\mathcal{C}\_{\text{math}}(k)$ (인도메인 수학 기여도)와 $\mathcal{C}_{\text{all}}(k)$ (코드, 추론, 언어를 포함한 전체 기여도)를 비교합니다:

$$\mathcal{C}_{\text{all}}(k) = \frac{S_k^{\text{overall}} - S_{\text{base}}^{\text{overall}}}{S_{\text{full}}^{\text{overall}} - S_{\text{base}}^{\text{overall}}}$$

**핵심 발견**: 세 가지 Qwen3 모델 모두에서 $\mathcal{C}\_{\text{math}}$와 $\mathcal{C}_{\text{all}}$ 간의 피어슨 상관계수가 $r > 0.6$으로, 수학을 효과적으로 학습한 레이어가 **코딩, 추론, 언어 이해 등 OOD 태스크에서도 동시에 향상**됩니다.

구체적 예시 (Qwen3-1.7B-Base, Table 2):

| 레이어 | $\mathcal{C}_{\text{math}}$ | $\mathcal{C}_{\text{all}}$ | Code | Reasoning | Language |
|:---|:---:|:---:|:---:|:---:|:---:|
| Layer 10 (최고) | **1.14** | **1.03** | 34.6 | 21.9 | 47.2 |
| Layer 1 | 0.87 | **1.32** | **40.0** | 22.7 | 47.1 |
| Layer 7 | 0.80 | 1.09 | 38.1 | 22.4 | 46.5 |
| Layer 24 (최저) | 0.28 | 0.06 | 30.6 | 21.6 | 44.2 |

이는 단일 레이어 학습이 **훈련 목표에 과적합(overfitting)하지 않고 진정한 광범위 능력 향상**을 달성함을 시사합니다.

### 3.2 Cross-Dataset 일반화

동일 모델(Qwen3-1.7B-Base)에 대해 서로 다른 수학 데이터셋(NuminaMath-CoT vs. DeepScaleR)에서 레이어 기여도 순위의 일관성을 Spearman 순위 상관계수로 측정:

$$\rho_{\text{same task (math)}} = 0.76, \quad p < 0.001$$

이는 레이어 기여도 순위가 **특정 데이터셋의 내용이 아닌 모델의 내부 구조**에 의해 결정됨을 의미합니다.

### 3.3 Cross-Task 일반화

수학(NuminaMath-CoT) vs. 코딩(DeepCoder) 간의 레이어 기여도 순위 상관:

$$\rho_{\text{cross task (math vs. code)}} = 0.59, \quad p < 0.001$$

완전히 다른 능력(수학적 추론 vs. 코드 생성)을 대상으로 할 때도 동일한 레이어가 높은 기여도를 보입니다. 이는 레이어 기여도가 **태스크-특이적 속성이 아닌, 사전 학습된 모델의 내재적 속성**임을 강하게 시사합니다.

### 3.4 모델 패밀리·알고리즘 간 일반화

| 검증 축 | 결과 |
|:---|:---|
| Qwen2.5-Math-1.5B + Dr. GRPO | 중간 레이어 집중 패턴 동일하게 재현 |
| Qwen2.5-Instruct + GiGPO + ALFWorld (에이전트) | 동일한 구조적 패턴 유지; 최고 레이어 $\mathcal{C} > 1.0$ |
| DeepSeek-Distilled-Qwen-7B + GRPO | 증류 모델에서도 동일한 패턴 확인 |

### 3.5 일반화 향상 메커니즘: 레이어 다양성과 앙상블

고기여도 레이어들 간의 **상호 보완적 문제 해결 행동**이 일반화에 기여합니다:

- Qwen3-1.7B-Base에서 상위 7개 레이어 훈련 모델들의 쌍별 Jaccard 유사도: **평균 34.1%**
  - 즉, 유사한 정확도임에도 서로 다른 문제 집합을 해결
- 7개 레이어 훈련 모델의 다수결 투표 (OlympiadBench): $33.6\%$
  - 전체 RL×7 샘플 자기 일관성(self-consistency): $31.3\%$ 대비 우세
  - 이는 **구조적 다양성(structural diversity)**이 샘플링 다양성보다 효과적임을 시사

### 3.6 가중치 변화 vs. 기여도: 일반화 관련 핵심 인사이트

레이어 기여도는 **파라미터 변화 크기(weight change magnitude)**로 설명되지 않습니다:

- 전체 파라미터 학습 시: 레이어별 $\|\Delta\theta_k\|_2$가 $0.5$ ~ $0.8$ 범위로 **균일하게 분포**
- 그러나 레이어 기여도는 0.28~1.14로 **극도로 불균일**
- 단일 레이어 학습 시: 고기여도·저기여도 레이어 모두 유사한 가중치 변화를 보이지만, 성능 결과는 크게 다름

결론: **레이어 기여도는 RL 개선을 포착하는 파라미터 부분 공간의 효율성**을 반영하며, 이 속성이 태스크 전반에 걸친 일반화 성능을 결정합니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려사항

### 4.1 앞으로의 연구에 미치는 영향

#### (1) RL 포스트 트레이닝 패러다임 재고
기존의 "모든 레이어를 균일하게 업데이트" 패러다임에 근본적 의문을 제기합니다. 향후 RL 학습 방법론 설계 시 레이어 불균일성을 명시적으로 고려해야 할 것입니다.

#### (2) 효율적 RLVR 방향 제시
- **파라미터 효율적 RL (PERL)**: 고기여도 레이어만 업데이트하는 PEFT 유사 접근이 RL 영역에서 표준이 될 가능성
- **메모리 및 계산 효율성**: 전체 파라미터 RL 대비 훈련해야 할 레이어 수 감소로 GPU 메모리 및 계산 비용 절감 가능

#### (3) 트랜스포머 내부 구조 이해
- 왜 중간 레이어가 RL 적응에 최적인지에 대한 기계론적 해석 가능성(mechanistic interpretability) 연구 촉진
- 사전 학습 단계에서 어떤 메커니즘이 이 구조를 형성하는지에 대한 연구 동기 부여

#### (4) 모델 편집 및 지식 삽입
단일 레이어로 대부분의 RL 이득을 달성할 수 있다는 발견은 모델 편집, 지속 학습(continual learning), 도메인 적응 등에서 **최소 침습적(minimally invasive) 업데이트 전략**의 가능성을 열어줍니다.

#### (5) 앙상블 및 모델 다양성 연구
레이어별 훈련 모델들의 상호 보완성은 **저비용 고다양성 앙상블** 구성의 새로운 방법론을 제시합니다. 이는 Mixture-of-Experts, 모델 병합(model merging) 등과의 결합 연구를 자극할 수 있습니다.

---

### 4.2 앞으로 연구 시 고려해야 할 점

#### (1) 더 큰 모델 스케일로의 확장 검증
현재 실험은 최대 8B 파라미터까지만 검증되었습니다. 70B, 405B 등 대형 모델에서도 동일한 패턴이 유지되는지, 혹은 다른 분포를 보이는지 반드시 확인해야 합니다.

#### (2) 이론적 근거 마련
왜 40%~60% 깊이의 중간 레이어가 RL 신호를 더 잘 흡수하는지에 대한 이론적 설명이 필요합니다. 이는 다음과 같은 관점에서 접근 가능합니다:
- **표현 이론**: 중간 레이어의 특징 표현이 갖는 특성
- **그래디언트 흐름**: 역전파 시 중간 레이어에서의 그래디언트 신호 강도
- **정보 병목(Information Bottleneck)** 이론과의 연관성

#### (3) 사전 학습과 RL 적응의 관계 규명
Nepal et al. (2025)의 연구에 따르면 수학적 추론에 중요한 레이어가 사전 학습 단계에서 결정됩니다. 이와의 연관성을 분석하고, RL 적응에 중요한 레이어가 사전 학습 단계에서 어떻게 형성되는지 탐구해야 합니다.

#### (4) 다른 포스트 트레이닝 방법과의 비교
- **SFT(Supervised Fine-Tuning)**: 이미 LISA (Pan et al., 2024) 등에서 레이어 이질성이 관찰되었으나, RL에서의 패턴과 어떻게 다른지 비교 분석 필요
- **DPO (Direct Preference Optimization)**: RL의 대안인 DPO에서도 유사한 레이어 집중 현상이 나타나는지 확인

#### (5) 레이어 기여도의 동적 변화 연구
현재 연구는 학습 완료 후의 성능을 기준으로 기여도를 측정합니다. 학습 과정 중 레이어 기여도가 어떻게 변화하는지(동적 분석)를 추적하면 더 정교한 적응형 학습 전략 개발이 가능합니다.

#### (6) 레이어 기여도와 내부 표현의 연관성 분석
레이어 기여도가 높은 레이어에서 실제로 어떤 내부 표현(attention 패턴, 활성화 분포 등)의 변화가 일어나는지 분석함으로써, 메커니즘 수준의 이해를 높여야 합니다.

#### (7) 다중 레이어 조합 최적화
현재는 개별 레이어를 독립적으로 학습하여 기여도를 측정합니다. 레이어 간의 **시너지 효과(synergy)** 또는 **간섭 효과(interference)**를 고려한 조합 최적화 방법론(예: subset selection, 그리디 알고리즘) 개발이 필요합니다.

#### (8) 실제 배포 환경에서의 비용-효과 분석
레이어 기여도 프로파일링을 위한 초기 비용(모든 레이어 개별 학습)이 최종 성능 향상으로 얻는 이득을 초과하는지 여부를 실용적 관점에서 분석해야 합니다.

---

## 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 주요 내용 | 본 논문과의 관계 |
|:---|:---:|:---|:---|
| **LISA** (Pan et al.) | 2024 | SFT에서 레이어를 랜덤 샘플링하여 메모리 효율적 학습 | RL 설정이 아닌 SFT에서 레이어 이질성 활용; 본 논문은 RL로 확장 |
| **Cornerstone Layers** (Zhang et al.) | 2024 | 특정 레이어 제거 시 성능이 랜덤 수준으로 저하 ("코너스톤 레이어") | 레이어 중요성의 이질성 확인; 본 논문은 RL 기여도로 구체화 |
| **DeepSeekMath/GRPO** (Shao et al.) | 2024 | GRPO 알고리즘 제안; LLM 수학적 추론 강화 | 본 논문의 기반 RL 알고리즘 |
| **DeepSeek-R1** (Guo et al.) | 2025 | RL을 통한 LLM 추론 능력 강화 | RLVR의 중요성 부각; 본 논문의 연구 동기 |
| **Dr. GRPO** (Liu et al.) | 2025 | GRPO의 편향 수정 버전 제안 | 본 논문에서 검증 RL 알고리즘 중 하나로 채택 |
| **GiGPO** (Feng et al.) | 2025 | 에이전트 학습을 위한 그룹-인-그룹 정책 최적화 | 본 논문의 에이전트 실험에 활용 |
| **Qwen3** (Yang et al.) | 2025 | Qwen3 모델 패밀리 출시 | 본 논문의 주요 실험 모델 |
| **Layer Importance (Nepal et al.)** | 2025 | 수학적 추론에 중요한 레이어가 사전 학습 시 결정되고 포스트 트레이닝 이후에도 불변 | 본 논문 발견과 직접 연관; RL 적응의 중간 레이어 집중이 사전 학습에서 기원함을 시사 |
| **Layer Significance in Alignment** (Shi et al.) | 2025 | LLM 정렬에서 레이어 중요성 분석 | SFT 정렬 관점에서 레이어 이질성 연구; 본 논문은 RL로 확장 |
| **MISA** (Liu et al.) | 2026 | 중요도 인식 레이어 샘플링으로 SFT 개선 | 레이어 인식 최적화의 SFT 적용; 본 논문은 RL로 동일 아이디어 확장 |
| **Neural Thickets** (Gan & Isola) | 2026 | 사전 학습 가중치 주변에 다양한 태스크 전문가들이 밀집 분포 | 본 논문의 레이어별 훈련 모델 다양성 발견과 상호 보완적 관점 제공 |
| **AdaGradSelect** (Kumar et al.) | 2025 | 그래디언트 통계를 이용한 동적 레이어 선택으로 SFT 효율화 | SFT에서의 적응적 레이어 선택; 본 논문은 RL에서 레이어 기여도 기반 선택 |

---

## 참고 자료

본 답변은 다음 자료를 직접 참고하였습니다:

**주요 논문 (직접 분석):**
- Zhang, Z., Hu, R., Glentis, A., Li, D., Yau, C., Lin, H., & Hong, M. (2026). *"Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training."* arXiv:2607.01232v2.

**논문 내 인용 참고문헌 (논문 원문에 명시됨):**
- Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300.
- Pan, R. et al. (2024). *LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning.* arXiv:2403.17919.
- Zhang, Y., Dong, Y., & Kawaguchi, K. (2024). *Investigating Layer Importance in Large Language Models.* arXiv:2409.14381.
- Nepal, A. et al. (2025). *Layer Importance for Mathematical Reasoning is Forged in Pre-training and Invariant After Post-training.* arXiv:2506.22638.
- Shi, G. et al. (2025). *Understanding Layer Significance in LLM Alignment.* arXiv:2410.17875.
- Liu, Z. et al. (2025). *Understanding R1-Zero-like Training: A Critical Perspective.* Second Conference on Language Modeling.
- Feng, L. et al. (2025). *Group-in-Group Policy Optimization for LLM Agent Training.* arXiv:2505.10978.
- Yang, A. et al. (2025). *Qwen3 Technical Report.* arXiv:2505.09388.
- Guo, D. et al. (2025). *DeepSeek-R1: Incentivizes Reasoning in LLMs through Reinforcement Learning.* Nature, 645(8081):633-638.
- Liu, Y. et al. (2026). *MISA: Memory-Efficient LLMs Optimization with Module-wise Importance Sampling.* arXiv:2511.00056.
- Kumar, A. et al. (2025). *AdaGradSelect: An Adaptive Gradient-Guided Layer Selection Method for Efficient Fine-Tuning of SLMs.* arXiv:2512.15764.
- Gan, Y. & Isola, P. (2026). *Neural Thickets: Diverse Task Experts are Dense Around Pretrained Weights.* arXiv:2603.12228.
- Song, X. et al. (2026). *Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning.* arXiv:2510.02091.
- Wang, X. et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* arXiv:2203.11171.
- Shridhar, M. et al. (2021). *ALFWorld: Aligning Text and Embodied Environments for Interactive Learning.* arXiv:2010.03768.
- LI, J. et al. (2024). *NuminaMath.* GitHub/HuggingFace.
- Luo, M. et al. (2025a). *DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level.* Notion Blog.
- Luo, M. et al. (2025b). *DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL.* Notion Blog.
- He, J. et al. (2025). *Skywork Open Reasoner 1 Technical Report.* arXiv:2505.22312.
- Yu, Q. et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* arXiv:2503.14476.
