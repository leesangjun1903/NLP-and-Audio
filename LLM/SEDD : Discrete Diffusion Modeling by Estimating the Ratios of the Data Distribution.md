# SEDD : Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution

## 1. 핵심 주장과 주요 기여

**SEDD (Score Entropy Discrete Diffusion) 모델**은 기존 이산 확산 모델의 한계를 극복하여 자연어 생성에서 자기회귀 모델과 경쟁할 수 있는 수준의 성능을 달성했습니다.[1]

### 주요 기여점
- **Score Entropy 손실 함수 도입**: 연속 확산 모델의 score matching을 이산 공간으로 자연스럽게 확장[1]
- **데이터 분포 비율 추정**: 이산 확산 과정을 데이터 분포의 비율을 추정하여 매개변수화[1]
- **실질적 성능 향상**: 기존 언어 확산 모델 대비 25-75% perplexity 감소, GPT-2를 능가하는 성능[1]

## 2. 해결하고자 하는 문제

### 문제 정의
이산 데이터(자연어 등)에 대한 확산 모델은 연속 데이터에서의 성공에도 불구하고 자기회귀 모델에 비해 성능이 크게 뒤떨어지는 문제가 있었습니다. 기존 접근법들은:[1]

- **Mean Prediction**: 역밀도 $$p_{0|t}$$를 학습하지만 연속 시간에서 근사가 필요하고 성능이 낮음[1]
- **Ratio Matching**: 특수한 네트워크 구조가 필요하고 계산 비용이 높음[1]
- **Concrete Score Matching**: $$ℓ_2$$ 손실이 양수 제약과 호환되지 않아 발산 문제 발생[1]

## 3. 제안 방법 및 모델 구조

### Score Entropy 손실 함수

핵심 아이디어는 **Bregman divergence**를 기반으로 한 새로운 손실 함수입니다:

$$L_{SE} = \mathbb{E}_{x \sim p} \left[ \sum_{y \neq x} w_{xy} \left( s_\theta(x)_y - \frac{p(y)}{p(x)} \log s_\theta(x)_y + K\left(\frac{p(y)}{p(x)}\right) \right) \right]$$

여기서 $$K(a) = a(\log a - 1)$$는 정규화 상수 함수입니다.[1]

### Denoising Score Entropy

실제 구현에서는 다음의 tractable한 형태를 사용합니다:

$$L_{DSE} = \mathbb{E}_{x_0 \sim p_0, x \sim p(\cdot|x_0)} \left[ \sum_{y \neq x} w_{xy} \left( s_\theta(x)_y - \frac{p(y|x_0)}{p(x|x_0)} \log s_\theta(x)_y \right) \right]$$

### 모델 구조

**Diffusion Transformer 기반** 아키텍처를 사용하며:
- 시간 조건화가 포함된 인코더 전용 트랜스포머[1]
- Rotary positional encoding 적용[1]
- 토큰 레벨에서 독립적으로 perturbation 적용:

$$Q_t(x_1 \ldots x_i \ldots x_d, x_1 \ldots \tilde{x}_i \ldots x_d) = Q_{t}^{tok}(x_i, \tilde{x}_i)$$

**두 가지 전이 행렬** 사용:
- **Uniform Matrix**: 모든 토큰으로의 균등한 전이[1]
- **Absorbing Matrix**: MASK 토큰으로의 흡수 상태[1]

### 샘플링 전략

**Tweedie τ-leaping** 방법으로 최적 디노이징 수행:

$$\text{Transition probabilities} \propto \left[ \exp(-\sigma_{\Delta t}^t Q) s_\theta(x_t, t)_i \right]_y \exp(\sigma_{\Delta t}^t Q)(x_t^i, y)$$

## 4. 성능 향상 및 한계

### 성능 향상
- **Language Modeling**: GPT-2 대비 다수 zero-shot 태스크에서 우수한 성능[1]
- **Generation Quality**: 6-8배 향상된 generative perplexity (annealing 없이)[1]
- **Computational Efficiency**: 32배 적은 네트워크 평가로 유사한 품질 달성[1]
- **Controllability**: 임의 위치에서의 프롬프팅 및 infilling 가능[1]

### 한계
- **현대 대형 언어 모델과의 격차**: GPT-2 수준에는 도달했으나 최신 LLM과는 여전히 차이[1]
- **KV-cache 부재**: 자기회귀 모델의 KV-cache 최적화 부재로 인한 추론 속도 이슈[1]
- **제한된 하이퍼파라미터 탐색**: 체계적인 하이퍼파라미터 최적화 부족[1]

## 5. 일반화 성능 향상 가능성

### 핵심 개선 요소
**Score Entropy의 이론적 보장**: 충분한 샘플과 모델 용량에서 최적해 $$s_{\theta^*}(x)_y = \frac{p(y)}{p(x)}$$ 달성 보장[1]

**Likelihood 기반 훈련**: ELBO(Evidence Lower BOund)를 통한 직접적인 likelihood 최적화:

$$-\log p_\theta^0(x_0) \leq L_{DWDSE}(x_0) + D_{KL}(p_{T|0}(\cdot|x_0) \| p_{base})$$

### 일반화 개선 메커니즘
- **안정적 그래디언트**: $$\frac{1}{s_\theta(x)_y}$$ 정규화로 발산 방지[1]
- **조건부 생성**: Bayes 규칙을 통한 임의 조건부 생성 가능[1]
- **분포 호환성**: 실제 확산 전이 확률과의 이론적 일치성[1]

## 6. 미래 연구에 미치는 영향과 고려사항

### 연구 영향
**패러다임 전환**: 자기회귀 모델 중심에서 확산 기반 언어 모델로의 전환 가능성 제시[1]

**이론적 기여**: 이산 확산을 위한 견고한 수학적 프레임워크 제공[1]

**실용적 응용**: 제어 가능한 텍스트 생성과 효율적 추론의 새로운 가능성[1]

### 향후 연구 고려사항

**확장성 개선**: 
- 더 큰 모델 크기로의 확장성 검증 필요
- 효율적인 샘플링 알고리즘 개발[1]

**경험적 최적화**:
- 노이즈 스케줄 체계적 탐색
- 연속 확산 모델의 경험적 기법 적용[1]

**응용 분야 확장**:
- 다양한 이산 도메인으로의 적용
- 멀티모달 생성 모델로의 확장 가능성

이 연구는 이산 확산 모델이 자연어 생성에서 실용적으로 사용될 수 있는 가능성을 최초로 입증했으며, 향후 확산 기반 언어 모델 발전의 중요한 토대를 제공합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/94717c18-b5d9-4f3c-b20f-0910950997d4/2310.16834v3.pdf)
