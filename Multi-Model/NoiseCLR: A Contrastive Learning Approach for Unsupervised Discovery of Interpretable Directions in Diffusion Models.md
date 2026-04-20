
# NoiseCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions in Diffusion Models

> **논문 정보**
> - **저자:** Yusuf Dalva, Pinar Yanardag (Virginia Tech)
> - **arXiv:** [2312.05390](https://arxiv.org/abs/2312.05390) (2023년 12월 8일)
> - **발표:** CVPR 2024 (Oral) — pp. 24209–24218
> - **공식 프로젝트 페이지:** https://noiseclr.github.io
> - **공식 GitHub:** https://github.com/gemlab-vt/NoiseCLR

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Diffusion 모델은 고품질 이미지를 생성하는 강력한 도구로 부상했지만, 그 잠재 공간(latent space)은 GAN에 비해 충분히 탐구되거나 이해되지 않은 상태이다.

기존에 diffusion 모델의 잠재 공간을 탐색하는 방법들은 특정 의미론을 찾기 위해 텍스트 프롬프트에 의존한다. 그러나 이 접근 방식은 예술, 패션, 또는 적절한 텍스트 프롬프트를 구성하기 어렵거나 불가능한 의학과 같은 전문 분야에서 제한적일 수 있다.

이를 해결하기 위해 NoiseCLR은 다음과 같은 핵심 주장을 제시한다:

이 논문에서는 텍스트 프롬프트에 의존하지 않고 text-to-image diffusion 모델에서 잠재적 의미(latent semantics)를 발견하기 위한 비지도(unsupervised) 방법을 제안한다.

### 주요 기여

저자들이 아는 한, 이 접근 방식은 다양한 도메인 내 및 도메인 간에 방향들을 결합하는 수준까지, Stable Diffusion의 잠재 공간에서 방향들을 분리(disentangled) 방식으로 성공적으로 발견한 **최초의 비지도 방법**이다. 주요 기여는 다음과 같다: Stable Diffusion과 같은 사전 학습된 text-to-image diffusion 모델에서 의미론적 방향을 발견하기 위한 대조 학습 기반 프레임워크인 NoiseCLR을 제안한다.

이 접근 방식은 텍스트 프롬프트, 레이블 데이터, 사용자 안내 없이, 대상 도메인과 관련된 비교적 소수의 이미지(약 100장)에만 의존한다. 방법은 얼굴, 자동차, 고양이, 예술 작품 등 다양한 범주에서 다양하고 세밀한 방향을 발견하는 능력을 보여준다. 발견된 방향들은 고도로 분리(disentangled)되어 있으며 단일 도메인 내에서 또는 다양한 도메인에 걸쳐 여러 방향을 동시에 적용할 수 있다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**① Diffusion 모델 잠재 공간의 미탐구 문제**

Diffusion 모델은 고품질 이미지 생성에 강력한 도구로 부상했지만, 잠재 공간은 충분히 탐구되거나 이해되지 않고 있다.

반면, GAN 기반 모델은 잘 분리된 잠재 공간(disentangled latent space)으로 잘 알려져 있으며, 이것이 제어된 이미지 편집에서의 성공을 견인하는 핵심 특징이다.

**② 텍스트 프롬프트 의존성 문제**

Diffusion 기반 모델에서 생성 과정에 대한 세밀한 제어를 제공하는 대부분의 선행 연구는 잠재 벡터 혼합, 모델 파인튜닝, 임베딩 최적화와 같은 단순한 해결책에 초점을 맞추고 있다. 그러나 이런 방법들은 특정 의미론을 정확히 지정하기 위해 사용자 제공 텍스트 프롬프트에 의존한다(예: '안경을 쓴 여성의 사진'). 이 접근 방식은 적절한 텍스트 프롬프트를 만들기 어렵거나 방대한 도메인 지식이 요구되는 예술, 패션, 의학과 같은 분야에서 제한적일 수 있다.

이러한 한계는 비지도 방식으로 잠재 공간에서 방향을 발견하는 것의 중요성을 부각시킨다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) 핵심 아이디어: Noise Space에서의 대조 학습

NoiseCLR은 비지도 방식으로 잠재 방향을 학습하기 위해 대조 목적 함수(contrastive objective)를 사용한다. 이 방법의 핵심 통찰은 **노이즈 공간에서 동일한 편집은 서로 끌어당겨야 하고, 서로 다른 방향에 의한 편집은 서로 밀어내야 한다**는 것이다.

#### (B) 학습 절차

도메인별(예: 얼굴 이미지) $N$개의 레이블되지 않은 이미지가 주어지면, 먼저 $t$ 타임스텝에 대한 순방향 확산 과정(forward diffusion process)을 적용한다. 그런 다음 노이즈된 변수 $\{x_1, \ldots, x_N\}$을 사용하여, 학습된 잠재 방향을 조건으로 한 역노이즈 제거 단계를 적용한다. 방법은 Stable Diffusion과 같은 사전 학습된 노이즈 제거 네트워크에 대해 $K$개의 잠재 방향 $\{d_1, \ldots, d_K\}$를 발견하며, 이 방향들은 립스틱 추가와 같은 의미론적으로 의미 있는 편집에 대응한다.

#### (C) 수식 설명

Diffusion 모델의 순방향 과정(Forward Process)은 다음과 같다:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$

이를 $t$ 스텝에 걸쳐 합치면 다음과 같다:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t}(1 - \beta_s)$이다.

NoiseCLR에서 방향 $d_k$를 적용한 편집된 노이즈 예측은 다음과 같이 표현된다:

$$\hat{\epsilon}_\theta(x_t, t, d_k) = \epsilon_\theta(x_t + \alpha \cdot d_k, t)$$

여기서 $\alpha$는 편집 강도(scale)이고, $\epsilon_\theta$는 사전 학습된 UNet 노이즈 예측기이다.

**대조 손실 함수 (Contrastive Loss)**

NoiseCLR의 핵심은 노이즈 예측의 변화량(delta)을 특징으로 사용하는 대조 학습이다. 이미지 $x_i$에 방향 $d_k$를 적용할 때의 노이즈 변화량은:

$$\delta_{i,k} = \hat{\epsilon}_\theta(x_t^{(i)}, t, d_k) - \epsilon_\theta(x_t^{(i)}, t)$$

이를 특징 벡터로 삼아 InfoNCE 스타일의 대조 손실을 정의한다:

$$\mathcal{L}_{\text{NoiseCLR}} = -\sum_{k=1}^{K} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\delta_{i,k}, \delta_{j,k}) / \tau)}{\sum_{k' \neq k} \exp(\text{sim}(\delta_{i,k}, \delta_{i,k'}) / \tau)}$$

여기서:
- $\text{sim}(\cdot, \cdot)$: 코사인 유사도(cosine similarity)
- $\tau$: 온도 파라미터(temperature parameter)
- **Positive 쌍**: 동일한 방향 $d_k$를 서로 다른 이미지에 적용한 결과 $(\delta_{i,k}, \delta_{j,k})$
- **Negative 쌍**: 동일한 이미지에 서로 다른 방향을 적용한 결과 $(\delta_{i,k}, \delta_{i,k'})$

이 손실은 다음 두 가지를 동시에 달성한다:
- **같은 방향(d_k)으로 편집된 서로 다른 이미지들의 특징은 유사해야 한다(attract)**
- **다른 방향으로 편집된 특징들은 달라야 한다(repel)**

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        NoiseCLR 프레임워크                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [입력] N개 비레이블 이미지 (도메인별 ~100장)                        │
│         예: 얼굴 이미지, 고양이 이미지, 예술 작품 등                  │
│                         ↓                                        │
│  [Step 1] 순방향 확산 (Forward Diffusion, t 스텝)                  │
│           x₀ → x_t (노이즈 추가)                                  │
│                         ↓                                        │
│  [Step 2] 학습 가능한 방향 벡터 d_k 적용                            │
│           K개의 방향 {d₁, d₂, ..., d_K} (학습 파라미터)             │
│                         ↓                                        │
│  [Step 3] 사전학습 UNet (Stable Diffusion)으로                    │
│           노이즈 예측 차이(delta) 계산                              │
│           δᵢₖ = ε_θ(x_t + α·d_k) - ε_θ(x_t)                   │
│                         ↓                                        │
│  [Step 4] 대조 손실 (Contrastive Loss) 계산                       │
│           • Positive: 같은 방향 d_k, 다른 이미지                   │
│           • Negative: 같은 이미지, 다른 방향                       │
│                         ↓                                        │
│  [출력] K개의 분리된 방향 {d₁, ..., d_K}                           │
│         → 편집 적용 시 이미지에 의미론적 변환 수행                    │
│                                                                  │
│  ※ Stable Diffusion UNet 가중치는 고정 (No fine-tuning)           │
└─────────────────────────────────────────────────────────────────┘
```

**주요 구조적 특징:**
- 이 방법은 diffusion 모델의 파인튜닝이나 재학습이 필요 없으며, 방향 학습을 위해 어떤 레이블 데이터도 필요하지 않는다.
- 학습은 2개의 NVIDIA L40 GPU에서 수행되었으며, 데이터셋 디렉토리를 config 파일에 지정하는 방식으로 학습을 진행한다.

---

### 2.4 성능 향상

학습된 방향들은 동일 도메인 내(예: 다양한 유형의 얼굴 편집)에서든 도메인 간(예: 동일 이미지에 고양이와 얼굴 편집 동시 적용)에서든 서로 간섭 없이 동시에 적용될 수 있다. 광범위한 실험 결과, 이 방법이 diffusion 기반 및 GAN 기반 잠재 공간 편집 방법 모두에서 기존 접근법을 능가하며 고도로 분리된 편집을 달성함을 보여준다.

구체적인 성능 우위 포인트:

| 비교 기준 | NoiseCLR | 기존 방법 |
|---|---|---|
| 텍스트 프롬프트 필요 여부 | ❌ 불필요 | ✅ 필요 |
| 레이블 데이터 필요 여부 | ❌ 불필요 | ✅ 필요 (일부) |
| 모델 파인튜닝 여부 | ❌ 불필요 | ✅ 필요 (일부) |
| 도메인 간 편집 | ✅ 지원 | ❌ 미지원 |
| 편집 분리도(Disentanglement) | **State-of-the-art** | 낮음 |
| 필요 이미지 수 | ~100장 | 많은 데이터 필요 |

방법으로 학습된 방향들이 고도로 분리되어 있기 때문에, 서로 다른 도메인의 편집이 서로 영향을 미치는 것을 방지하기 위한 의미론적 마스크나 사용자 안내가 필요하지 않다.

---

### 2.5 한계점

NoiseCLR은 대조 학습을 통해 소규모 이미지 데이터셋에서 의미론적으로 의미 있는 방향을 식별하는 방법을 도입한다. 그러나 식별된 각 방향은 그것이 무엇을 나타내는지를 설명하기 위해 수동적 해석이 필요하다.

이런 방법들에는 몇 가지 제한이 있다: NoiseCLR과 같은 접근 방식은 잠재 공간 내에서 의미론적으로 중요할 수 있는 방향들의 집합을 식별하지만, 이러한 방향들은 그 중요성을 이해하기 위해 광범위한 수동적 해석이 요구되는 경우가 많다.

추가적인 한계:

1. **자동 레이블링 부재:** 발견된 방향에 자동으로 이름을 붙이는 메커니즘이 없어, 사람이 직접 각 방향을 해석해야 한다.
2. **도메인 특이성:** 특정 도메인의 이미지가 필요하여, 도메인 간 일반화가 완전히 자동화되지 않는다.
3. **방향 수 K 사전 설정:** 발견할 방향의 수 $K$를 사전에 지정해야 한다.
4. **Stable Diffusion 의존성:** 실험이 주로 Stable Diffusion 기반으로 이루어져, 다른 diffusion 아키텍처로의 범용성은 추가 검증이 필요하다.

---

## 3. 일반화 성능 향상 가능성

### 3.1 출-오브-도메인(Out-of-Domain) 일반화

학습된 편집은 도메인 내 예시(예: 사람 얼굴)뿐만 아니라 도메인 외(out-of-domain) 이미지(예: 회화)에도 효과적이다. 방법의 다양한 도메인에 걸친 일반화 가능성을 시연하기 위해, 예술 회화, 고양이, 자동차에 대한 편집 결과를 제공한다.

### 3.2 인트라/인터 도메인 일반화

NoiseCLR은 Stable Diffusion의 공유 잠재 공간 내에서 서로 다른 도메인의 잠재 방향을 학습할 수 있으므로, 도메인 내(intra-domain) 및 도메인 간(inter-domain) 편집을 모두 수행할 수 있다. 이 방법은 a) 같은 도메인의 편집을 동시에 적용하는 도메인 내 편집, b) 서로 다른 도메인의 편집을 결합해 동시에 적용하는 교차 도메인 편집을 모두 찾을 수 있다.

### 3.3 일반화 가능성의 핵심 메커니즘

편집 결과에서 보여지듯이, 이 방법은 단일 diffusion 모델을 사용해 다양한 도메인에서 잠재 방향을 학습하고 적용할 수 있다.

이러한 일반화는 다음 요소들에 의해 가능하다:

- **공유 잠재 공간 활용:** Stable Diffusion의 공유 latent space가 다양한 도메인 개념을 동시에 표현
- **도메인 무관 대조 학습:** 대조 학습 목적 함수 자체가 도메인에 특화되지 않음
- **소규모 데이터로 새 도메인 적응:** 약 100장의 이미지로 새 도메인의 방향을 학습 가능

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 GAN 기반 비지도 방향 발견 연구

| 논문 | 방법 | 특징 | 한계 |
|---|---|---|---|
| **GANSpace (ICCV 2021)** | PCA on GAN latent | 해석 가능한 GAN 편집 | GAN에만 적용 |
| **LatentCLR (ICCV 2021)** | 대조 학습 on GAN latent | 비지도 GAN 방향 발견 | GAN에만 적용 |
| **InterfaceGAN (TPAMI 2020)** | SVM on GAN latent | 감독 학습 기반 | 레이블 필요 |

예를 들어, LatentCLR(Yüksel et al., 2021)은 GAN의 잠재 공간에 대조 학습을 적용하여 의미 있는 변환을 식별하고, NoiseCLR(Dalva & Yanardag, 2024)은 Stable Diffusion과 같은 사전 학습된 text-to-image diffusion 모델에서 의미론적 방향을 발견한다.

### 4.2 Diffusion 모델 잠재 공간 탐색 연구

**Asyrp / h-space (ICLR 2023)**

Asyrp는 동결된 사전 학습 diffusion 모델에서 의미론적 잠재 공간을 발견하는 비대칭 역과정(asymmetric reverse process)을 제안한다. h-space라 명명된 의미론적 잠재 공간은 의미론적 이미지 조작을 위한 좋은 특성(동질성, 선형성, 강건성, 타임스텝 간 일관성)을 갖는다. 또한, 편집 강도 구간과 타임스텝별 품질 결핍이라는 정량화 가능한 측정 기준으로 다목적 편집 및 품질 향상을 위한 원칙적인 생성 과정 설계를 도입한다.

**Discovering Interpretable Directions in h-space (2023)**

이 논문에서는 h-space의 속성을 탐구하고 그 안에서 의미 있는 의미론적 방향을 찾기 위한 여러 새로운 방법을 제안한다. 시작은 사전 학습된 DDM에서 해석 가능한 의미론적 방향을 드러내는 비지도 방법을 연구하는 것이다.

**Self-Discovering Interpretable Diffusion Latent Directions (CVPR 2024)**

CVPR 2024 논문 "Self-Discovering Interpretable Diffusion Latent Directions for Responsible Text-to-Image Generation"은 책임감 있는 텍스트-이미지 생성을 위한 해석 가능한 diffusion 잠재 방향을 자기발견하는 내용을 다룬다.

### 4.3 종합 비교표

| 방법 | 연도 | 기반 모델 | 비지도 | 텍스트 불필요 | 다중 방향 동시 적용 | 도메인 간 편집 |
|---|---|---|---|---|---|---|
| InterfaceGAN | 2020 | GAN | ❌ | ✅ | ❌ | ❌ |
| GANSpace | 2021 | GAN | ✅ | ✅ | △ | ❌ |
| LatentCLR | 2021 | GAN | ✅ | ✅ | △ | ❌ |
| Asyrp (h-space) | 2022 | Diffusion | ❌ | ❌ | ❌ | ❌ |
| DiffusionCLIP | 2022 | Diffusion | ❌ | ❌ | ❌ | ❌ |
| **NoiseCLR** | **2023** | **Diffusion (SD)** | **✅** | **✅** | **✅** | **✅** |
| Self-Discovering | 2024 | Diffusion (SD) | △ | ❌ | △ | △ |

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5.1 연구 영향

**① Diffusion 모델 해석 가능성 연구의 새 방향**

NoiseCLR은 diffusion 모델의 노이즈 공간(noise space)이 대조 학습을 적용하기 적합한 구조를 갖고 있음을 최초로 실증하였다. 이는 diffusion 모델의 latent space 해석에 관한 후속 연구의 핵심 기반을 제공한다.

생성 모델에서 해석 가능한 방향을 발견하는 것은 다양한 이미지 생성 및 편집 작업의 근본이며, 이 맥락에서 대조 학습이 매우 효과적임이 입증되었다.

**② 비지도 시각적 표현 학습**

텍스트 프롬프트 없이 소수의 이미지만으로 의미론적 방향을 발견할 수 있다는 점은, 데이터가 희소하거나 텍스트 레이블을 붙이기 어려운 특수 도메인(의학 영상, 위성 이미지 등)에의 응용 가능성을 시사한다.

**③ 멀티모달 편집 방향의 선구자적 역할**

NoiseCLR은 Stable Diffusion의 공유 잠재 공간 내에서 서로 다른 도메인의 잠재 방향을 학습할 수 있으므로, 도메인 내 및 도메인 간 편집 모두를 수행할 수 있다. 이러한 인터-도메인 편집 가능성은 이후 멀티도메인 이미지 편집 연구의 토대를 제공한다.

**④ 후속 연구로의 영향 (FluxSpace, 등)**

FluxSpace는 Flux와 같은 정류 흐름 트랜스포머(rectified flow transformers)로 생성된 이미지의 의미론을 제어하는 능력을 가진 표현 공간을 활용하는 도메인 무관 이미지 편집 방법이다. 이는 다양한 이미지 편집 작업을 가능케 하며, NoiseCLR의 아이디어를 더 새로운 아키텍처로 확장한다.

---

### 5.2 향후 연구 시 고려할 점

**① 방향의 자동 해석 및 레이블링**

발견된 방향을 수동으로 해석해야 한다는 한계를 극복하기 위해, CLIP이나 LLM을 활용한 자동 의미론적 레이블링 방법을 연구할 필요가 있다.

**② 더 많은 도메인과 아키텍처로의 확장**

현재 NoiseCLR은 Stable Diffusion을 주 기반으로 사용한다. SDXL, DALL-E 3, Flux 등 다양한 최신 diffusion 아키텍처에서의 적용 가능성을 검증하는 연구가 필요하다.

**③ 방향 수(K) 자동 결정**

발견할 방향의 수 $K$를 사전에 지정해야 한다는 한계가 있다. 적응적으로 $K$를 결정하거나, 더 많은 $K$에서도 안정적으로 분리된 방향을 학습하는 방법이 필요하다.

**④ 의료, 과학 도메인 적용**

텍스트 프롬프트 접근 방식은 예술, 패션과 같은 분야에서나 적절한 텍스트 프롬프트를 구성하기 어려운 의학과 같은 전문 분야에서 제한적일 수 있다. NoiseCLR은 이러한 특수 도메인에서의 적용 가능성을 갖고 있으며, 의료 영상 분석, 세포 이미지 편집 등으로 확장하는 연구가 중요한 방향이다.

**⑤ 편집의 인과적 해석**

발견된 방향이 인과적으로 어떤 특성을 제어하는지, 그리고 바이어스(편향) 없이 공정한 방향을 발견하는 방법을 연구하는 것이 책임 있는 AI 관점에서 중요하다.

**⑥ 비디오·3D 생성 모델로의 확장**

이미지 도메인에서 검증된 NoiseCLR의 원리를 Video Diffusion, 3D Gaussian Splatting 기반 생성 모델 등으로 확장하는 연구가 활발히 진행될 것으로 예상된다.

---

## 참고 자료 및 출처

| 번호 | 제목 | 출처/링크 |
|---|---|---|
| 1 | **NoiseCLR (메인 논문)** | arXiv: [2312.05390](https://arxiv.org/abs/2312.05390), CVPR 2024 |
| 2 | **NoiseCLR 공식 프로젝트 페이지** | https://noiseclr.github.io |
| 3 | **NoiseCLR 공식 GitHub** | https://github.com/gemlab-vt/NoiseCLR |
| 4 | **NoiseCLR CVPR 2024 Oral 발표** | https://cvpr.thecvf.com/virtual/2024/oral/32002 |
| 5 | **NoiseCLR IEEE Xplore** | https://ieeexplore.ieee.org/document/10657503 |
| 6 | **Asyrp: Diffusion Models Already Have A Semantic Latent Space (ICLR 2023)** | arXiv: [2210.10960](https://arxiv.org/abs/2210.10960) |
| 7 | **Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models (2023)** | arXiv: [2303.11073](https://arxiv.org/abs/2303.11073) |
| 8 | **Self-Discovering Interpretable Diffusion Latent Directions for Responsible T2I Generation (CVPR 2024)** | GitHub: [hangligit/InterpretDiffusion](https://github.com/hangligit/InterpretDiffusion) |
| 9 | **LatentCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions (ICCV 2021)** | Proceedings of ICCV 2021, pp. 14263–14272 |
| 10 | **ResearchGate - NoiseCLR 관련 최신 후속 연구 참조** | https://www.researchgate.net/publication/384212033 |
| 11 | **NoiseCLR ADS Abstract** | https://ui.adsabs.harvard.edu/abs/2023arXiv231205390D |

> ⚠️ **주의:** 본 답변에서 제시된 수식 중 일부(특히 $\delta_{i,k}$를 이용한 구체적 대조 손실 수식)는 논문의 핵심 아이디어를 기반으로 재구성한 것입니다. 논문 원문 PDF의 전체 수식 표현에 대한 완전한 확인이 필요한 경우, [CVPR 2024 논문 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Dalva_NoiseCLR_A_Contrastive_Learning_Approach_for_Unsupervised_Discovery_of_Interpretable_CVPR_2024_paper.pdf)를 직접 참조하시기 바랍니다.
