# One-step Diffusion with Distribution Matching Distillation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **Distribution Matching Distillation (DMD)**이라는 방법론을 제안하여, 수십~수백 번의 forward pass가 필요한 확산 모델(diffusion model)을 **단 1번의 forward pass**만으로 고품질 이미지를 생성할 수 있는 생성기로 변환합니다. 핵심 아이디어는 "노이즈→이미지 매핑 자체를 흉내 내는 것"이 아니라, **생성 분포(fake distribution)를 실제 데이터 분포(real distribution)와 일치시키는 것**입니다.

### 주요 기여

1. **분포 수준의 KL 발산 최소화**: 기존 방법들이 개별 샘플 수준의 대응(correspondence)을 학습하려 했다면, DMD는 분포 수준에서 매칭을 수행합니다.
2. **두 개의 스코어 함수 모델링**: 실제 분포와 가짜 분포 각각을 별도의 확산 모델로 파라미터화하여, 스코어 함수 차이를 기울기로 활용합니다.
3. **회귀 손실(Regression Loss) 결합**: 분포 매칭만으로는 발생하는 모드 붕괴(mode collapse)를 방지하기 위해 사전 계산된 노이즈-이미지 쌍을 이용한 LPIPS 회귀 손실을 결합합니다.
4. **SOTA 성능**: ImageNet 64×64에서 FID 2.62, zero-shot COCO-30k에서 FID 11.49를 달성하며, 기존 발표된 few-step 방법들을 모두 능가합니다.
5. **실시간 추론 가능**: FP16 추론 시 512×512 이미지를 **20 FPS**로 생성 가능합니다.

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

확산 모델은 고품질 이미지를 생성하지만, 샘플링 과정이 수십~수백 번의 신경망 평가를 요구하는 반복적(iterative) 프로세스입니다. 이는 실시간 대화형 응용에 큰 장벽이 됩니다. 기존 증류(distillation) 방법들은:

- **Progressive Distillation**: 단계적으로 스텝 수를 절반씩 줄이는 방식이나, 여전히 성능 저하가 있습니다.
- **Consistency Models**: ODE 흐름 위에서 자기 자신의 출력을 맞추도록 훈련하지만, 품질 격차가 존재합니다.
- **InstaFlow**: 정류 흐름(Rectified Flow)을 이용하지만, 분포 수준의 매칭이 아닌 궤적 수준의 매칭에 의존합니다.

이러한 방법들의 공통적인 한계는 **노이즈와 이미지 간 정밀한 대응(correspondence)을 학습하려 한다**는 점입니다. DMD는 이 문제를 분포 매칭 관점에서 접근함으로써 해결합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 분포 매칭 손실 (Distribution Matching Loss)

생성기 $G_\theta$가 생성하는 가짜 분포 $p_\text{fake}$와 실제 분포 $p_\text{real}$ 사이의 KL 발산을 최소화합니다:

$$D_{KL}(p_\text{fake} \| p_\text{real}) = \mathbb{E}_{x \sim p_\text{fake}} \left[ \log \frac{p_\text{fake}(x)}{p_\text{real}(x)} \right] = \mathbb{E}_{\substack{z \sim \mathcal{N}(0;\mathbf{I}) \\ x = G_\theta(z)}} \left[ -(\log p_\text{real}(x) - \log p_\text{fake}(x)) \right]$$

확률 밀도 함수를 직접 계산하는 것은 다루기 어렵기(intractable) 때문에, 생성기 파라미터 $\theta$에 대한 기울기를 스코어 함수로 표현합니다:

$$\nabla_\theta D_{KL} = \mathbb{E}_{\substack{z \sim \mathcal{N}(0;\mathbf{I}) \\ x = G_\theta(z)}} \left[ -\left(s_\text{real}(x) - s_\text{fake}(x)\right) \frac{dG}{d\theta} \right]$$

여기서 $s_\text{real}(x) = \nabla_x \log p_\text{real}(x)$, $s_\text{fake}(x) = \nabla_x \log p_\text{fake}(x)$는 각 분포의 스코어 함수입니다.

#### (2) 가우시안 확산을 통한 분포 겹침 확보

저확률 영역에서 스코어 함수가 발산하는 문제를 해결하기 위해, 생성된 이미지 $x = G_\theta(z)$에 가우시안 노이즈를 주입하여 확산된 샘플 $x_t$를 생성합니다:

$$q_t(x_t|x) \sim \mathcal{N}(\alpha_t x; \sigma_t^2 \mathbf{I})$$

여기서 $\alpha_t$와 $\sigma_t$는 확산 노이즈 스케줄의 파라미터입니다.

#### (3) 실제 스코어 함수 (Real Score)

사전 훈련된 확산 모델 $\mu_\text{base}(x, t)$를 이용하여 실제 분포의 스코어를 추정합니다 (Song et al., 2021):

$$s_\text{real}(x_t, t) = -\frac{x_t - \alpha_t \mu_\text{base}(x_t, t)}{\sigma_t^2}$$

#### (4) 동적으로 학습되는 가짜 스코어 함수 (Fake Score)

생성기의 분포가 훈련 중 변화하므로, 가짜 분포에 대한 별도의 확산 모델 $\mu_\text{fake}^\phi$를 동적으로 학습합니다:

$$s_\text{fake}(x_t, t) = -\frac{x_t - \alpha_t \mu_\text{fake}^\phi(x_t, t)}{\sigma_t^2}$$

$\mu_\text{fake}^\phi$는 표준 디노이징 손실로 업데이트됩니다:

$$\mathcal{L}_\text{denoise}^\phi = \|\mu_\text{fake}^\phi(x_t, t) - x_0\|_2^2$$

#### (5) 최종 분포 매칭 기울기

두 스코어 함수의 차이를 이용한 최종 근사 기울기는 다음과 같습니다:

$$\nabla_\theta D_{KL} \simeq \mathbb{E}_{z, t, x, x_t} \left[ w_t \alpha_t \left( s_\text{fake}(x_t, t) - s_\text{real}(x_t, t) \right) \frac{dG}{d\theta} \right]$$

여기서 $z \sim \mathcal{N}(0;\mathbf{I})$, $x = G_\theta(z)$, $t \sim \mathcal{U}(T_\text{min}, T_\text{max})$, $x_t \sim q_t(x_t|x)$입니다.

시간 의존적 스칼라 가중치 $w_t$는 서로 다른 노이즈 레벨에서 기울기 크기를 정규화합니다:

$$w_t = \frac{\sigma_t^2}{\alpha_t} \cdot \frac{CS}{\|\mu_\text{base}(x_t, t) - x\|_1}$$

여기서 $S$는 공간적 위치의 수, $C$는 채널 수입니다. $T_\text{min} = 0.02T$, $T_\text{max} = 0.98T$로 설정합니다.

#### (6) 회귀 손실 (Regression Loss)

사전 계산된 노이즈-이미지 쌍 데이터셋 $\mathcal{D} = \{z, y\}$를 이용한 LPIPS 회귀 손실:

$$\mathcal{L}_\text{reg} = \mathbb{E}_{(z, y) \sim \mathcal{D}} \, \ell(G_\theta(z), y)$$

#### (7) 최종 목적 함수

$$\mathcal{L}_\text{total} = D_{KL} + \lambda_\text{reg} \mathcal{L}_\text{reg}, \quad \lambda_\text{reg} = 0.25$$

---

### 2.3 모델 구조

| 구성 요소 | 설명 |
|-----------|------|
| **$G_\theta$ (생성기)** | 사전 훈련된 확산 모델의 디노이저와 동일한 아키텍처 사용. 시간 조건(time-conditioning) 제거. 초기 파라미터는 $\mu_\text{base}(z, T-1)$로 초기화 |
| **$\mu_\text{base}$ (실제 스코어 모델)** | 고정된 사전 훈련 확산 모델 (EDM 또는 Stable Diffusion v1.5) |
| **$\mu_\text{fake}^\phi$ (가짜 스코어 모델)** | $\mu_\text{base}$로 초기화된 후, 가짜 이미지에 대한 디노이징 손실로 동적 업데이트 |
| **사전 계산된 데이터셋** | 결정론적 ODE 솔버(Heun, PNDM)로 다단계 샘플링을 수행하여 노이즈-이미지 쌍 생성 |

**훈련 백본:**
- 클래스 조건부 생성: EDM (ImageNet 64×64, CIFAR-10)
- 텍스트-이미지 생성: Stable Diffusion v1.5 (LAION-Aesthetics)

---

### 2.4 성능 향상

**ImageNet 64×64 (Table 1):**

| 방법 | Forward Pass 수 | FID ↓ |
|------|----------------|--------|
| Consistency Model | 1 | 6.20 |
| TRACT | 1 | 7.43 |
| Diff-Instruct | 1 | 5.57 |
| **DMD (Ours)** | **1** | **2.62** |
| EDM (Teacher) | 512 | 2.32 |

Consistency Model 대비 **2.4배** FID 향상, 교사 모델(512 step)과 불과 0.3 FID 차이입니다.

**Zero-shot COCO-30k (Table 3):**

| 방법 | Latency | FID ↓ |
|------|---------|--------|
| InstaFlow-0.9B | 0.09s | 13.10 |
| UFOGen | 0.09s | 12.78 |
| **DMD (Ours)** | **0.09s** | **11.49** |
| SDv1.5 (Teacher) | 2.59s | 8.78 |

교사 모델(50 step, 2.59s)과 비교하여 **30배 빠른** 속도로 2.71 FID 차이만 보입니다.

---

### 2.5 한계

1. **잔존하는 품질 격차**: 100~1000 스텝 diffusion 모델과 비교하면 여전히 미세한 품질 차이가 존재합니다.
2. **높은 메모리 사용량**: 가짜 스코어 네트워크($\mu_\text{fake}^\phi$)와 생성기($G_\theta$) 모두를 파인튜닝하므로 훈련 메모리 요구량이 큽니다. LoRA 등의 기법으로 완화 가능성이 있습니다.
3. **오프라인 데이터셋 의존성**: 회귀 손실을 위해 사전 계산된 노이즈-이미지 쌍이 필요하며, 이 생성 비용도 무시할 수 없습니다.
4. **훈련 복잡성**: 두 개의 확산 모델을 동시에 관리해야 하는 복잡한 훈련 파이프라인을 요구합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 분포 수준 매칭의 일반화 이점

기존 궤적 수준(trajectory-level) 증류 방법들은 특정 노이즈→이미지 매핑에 과적합될 위험이 있습니다. 반면 DMD는 **분포 수준에서 매칭**을 수행하므로, 특정 샘플 쌍에 대한 의존성이 줄어들어 더 넓은 입력 분포에 대한 일반화 가능성이 높아집니다.

직관적으로, $G_\theta$가 $p_\text{real}$의 모든 모드(mode)를 포괄하도록 학습되기 때문에, 훈련 중 보지 못한 새로운 텍스트 프롬프트나 노이즈 패턴에 대해서도 합리적인 출력을 생성할 가능성이 있습니다.

### 3.2 회귀 손실의 모드 커버리지 기여

논문의 Figure 3에서 명확히 나타나듯이:

- **실제 스코어만 사용** → 모드 붕괴 (가장 가까운 모드 하나로 집중)
- **분포 매칭만 사용 (회귀 손실 없음)** → 더 많은 모드를 커버하지만, 일부 모드 누락
- **분포 매칭 + 회귀 손실** → **모든 모드 복원**

회귀 손실은 확장성(coverage)의 역할을 하여, 다양한 입력에 대해 전체적으로 일관된 출력을 유지하는 **일반화의 정규화(regularizer)** 역할을 합니다.

### 3.3 Classifier-Free Guidance와의 결합

DMD는 Classifier-Free Guidance(CFG)와 호환됩니다. 가이던스 스케일이 다른 두 가지 모델(guidance scale 3 및 8)에서 모두 성능을 검증하였으며, 이는 다양한 생성 강도에 대한 일반화 가능성을 보여줍니다.

### 3.4 다양한 아키텍처에 대한 범용성

논문은 DMD가 "결정론적 샘플링(deterministic sampling)을 갖는 모든 확산 모델에 보편적으로 적용 가능하다"고 명시합니다. EDM과 Stable Diffusion v1.5라는 서로 다른 아키텍처에 모두 적용하여 검증하였으며, 이는 방법론적 일반성을 입증합니다.

### 3.5 Zero-shot 일반화

LAION-Aesthetics 데이터로 훈련된 모델이 MS-COCO zero-shot 벤치마크에서 FID 11.49를 달성한 것은, 훈련 도메인 밖의 텍스트 프롬프트에 대해서도 합리적인 일반화 능력을 보여줍니다.

### 3.6 회귀 손실 함수에 대한 강건성

부록에서 L2 손실 사용 시 FID 2.78 (LPIPS 사용 시 2.66)을 기록하여, 특정 손실 함수 선택에 크게 의존하지 않는 **방법론적 강건성(robustness)**을 보여줍니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (a) 확산 모델 증류 패러다임 전환

기존 증류 연구들이 "궤적 모방(trajectory imitation)"에 집중했다면, DMD는 "분포 일치(distribution matching)"라는 새로운 목표를 제시하였습니다. 이는 후속 연구들(DMD2, ADD, SwiftBrush 등)에서 광범위하게 채택되고 있는 새로운 연구 방향을 개척했습니다.

#### (b) VSD의 생성 모델 훈련으로의 확장

ProlificDreamer(VSD)가 3D 객체 최적화에 사용했던 VSD 목적 함수를, **전체 생성 모델 훈련**에 적용하였습니다. 이는 "스코어 기반 손실로 신경망을 훈련할 수 있다"는 새로운 관점을 제시합니다.

#### (c) 실시간 인터랙티브 애플리케이션 가능성

20 FPS의 생성 속도는 비디오 편집, 실시간 예술 창작, 게임, AR/VR 등 다양한 실시간 응용 분야를 열어줍니다.

#### (d) 후속 연구들에 직접적 영향

- **DMD2** (Yin et al., 2024): DMD를 확장하여 더 높은 품질과 효율성을 달성
- **Adversarial Diffusion Distillation (ADD)**: GAN 손실과 증류를 결합하는 방향 탐색
- **SwiftBrush**: VSD 기반 one-step 생성기 연구
- **Hyper-SD**: 일관성 함수와 분포 매칭을 결합

### 4.2 앞으로 연구 시 고려할 점

#### (a) 메모리 효율적 훈련

현재 DMD는 두 개의 전체 확산 모델($G_\theta$, $\mu_\text{fake}^\phi$)을 동시에 파인튜닝해야 하므로 GPU 메모리 요구량이 매우 큽니다(72개 A100 GPU). 향후 연구에서는:
- **LoRA (Low-Rank Adaptation)** 적용으로 파라미터 효율성 향상
- **Gradient Checkpointing** 전략 최적화
- **양자화(Quantization)** 기법 결합

을 고려해야 합니다.

#### (b) 동적 가짜 스코어 모델의 안정성

$\mu_\text{fake}^\phi$는 훈련 중 생성기 $G_\theta$와 함께 동적으로 업데이트됩니다. 이 공동 훈련(joint training) 과정의 불안정성은 잠재적인 위험 요소이며, 다음을 고려해야 합니다:
- 가짜 스코어 모델의 업데이트 주기(frequency)와 학습률 조정
- Exponential Moving Average(EMA) 적용
- 가짜 스코어 모델의 용량(capacity) 최적화

#### (c) 고해상도 이미지 생성으로의 확장

현재 DMD는 512×512에서 검증되었으나, 1024×1024 이상의 고해상도로의 확장 시 추가적인 연구가 필요합니다. SDXL이나 Stable Diffusion 3 같은 더 강력한 교사 모델을 활용하는 방향을 고려할 수 있습니다.

#### (d) 멀티모달 및 비디오 생성으로의 일반화

DMD의 분포 매칭 원리는 이미지 생성에 국한되지 않습니다:
- **비디오 생성**: 시간적 일관성(temporal consistency)을 추가적으로 고려한 분포 매칭 필요
- **3D 생성**: VSD와의 연결성을 활용한 3D 객체 생성 가속화
- **오디오-비주얼 멀티모달**: 크로스 모달 분포 매칭 연구

#### (e) 오프라인 데이터셋의 한계 극복

현재 회귀 손실을 위한 오프라인 노이즈-이미지 쌍 생성은 초기 계산 비용을 요구합니다. **온라인(online) 쌍 생성** 전략이나 **소수의 쌍으로도 효과적인 정규화**를 달성하는 방법 연구가 필요합니다.

#### (f) 평가 지표의 다양화

FID는 분포 수준의 품질을 측정하지만, 텍스트-이미지 정렬, 다양성, 세밀한 구조 정확도 등 다각적인 평가가 필요합니다. 향후 연구에서는 Human Preference Score(HPS), ImageReward, PickScore 등을 포함한 종합적 평가를 고려해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 표는 DMD와 주요 관련 연구들을 비교합니다:

| 연구 | 연도 | 방법 핵심 | 스텝 수 | ImageNet FID | 주요 특징 |
|------|------|----------|---------|-------------|----------|
| DDPM (Ho et al.) | 2020 | 확산 모델 기초 | 1000 | - | 반복적 디노이징 확립 |
| Score-SDE (Song et al.) | 2021 | 스코어 기반 SDE | 수백 | - | 스코어 함수와 SDE 연결 |
| DDIM (Song et al.) | 2021 | 결정론적 샘플링 | 10~50 | - | 빠른 샘플링 가능 |
| EDM (Karras et al.) | 2022 | 확산 설계 공간 분석 | 35 | 2.32 | 최고 품질 교사 모델 |
| Progressive Distillation | 2022 | 단계적 스텝 감소 | 1 | 15.39 | 최초 실용적 1-step 시도 |
| Consistency Models (Song et al.) | 2023 | ODE 궤적 일관성 | 1 | 6.20 | 자기 일관성 훈련 |
| InstaFlow (Liu et al.) | 2023 | 정류 흐름 + 증류 | 1 | - | COCO FID 13.10 |
| Diff-Instruct (Luo et al.) | 2023 | VSD 기반 지식 전달 | 1 | 5.57 | 사전 학습 모델 활용 |
| **DMD (Yin et al.)** | **2023** | **분포 매칭 증류** | **1** | **2.62** | **최고 1-step 성능** |
| ADD (Sauer et al.) | 2023 | 적대적 확산 증류 | 1~4 | - | GAN + 증류 결합 |
| LCM (Luo et al.) | 2023 | 잠재 일관성 모델 | 2~4 | - | SD 기반 실용적 가속 |

### 주요 차별점 분석

**DMD vs. Consistency Models:**
- CM은 ODE 궤적 위에서 자기 일관성(self-consistency)을 강제하는 반면, DMD는 분포 수준 KL 발산을 최소화합니다.
- DMD가 FID 2.62 vs CM의 6.20으로 훨씬 우수한 성능을 보입니다.

**DMD vs. Diff-Instruct:**
- 두 방법 모두 VSD에서 영감을 받았으나, DMD는 회귀 손실을 추가하여 모드 커버리지를 보장합니다.
- DMD: FID 2.62 vs Diff-Instruct: FID 5.57 (ImageNet)

**DMD vs. ADD (Adversarial Diffusion Distillation):**
- ADD는 GAN 판별자를 사용하는 반면, DMD는 GAN 프레임워크 없이 스코어 함수 차이만으로 기울기를 계산합니다.
- ADD는 다단계(1~4 step)를 지원하지만, DMD는 순수 1-step에 특화되어 있습니다.

---

## 참고 자료

- **주 논문**: Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, Taesung Park. "One-step Diffusion with Distribution Matching Distillation." arXiv:2311.18828v4, 2024. (제공된 PDF)

- **관련 참고문헌 (논문 내 인용)**:
  - Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020.
  - Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations," ICLR 2021.
  - Song et al., "Denoising Diffusion Implicit Models," ICLR 2021.
  - Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models (EDM)," NeurIPS 2022.
  - Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)," CVPR 2022.
  - Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion Models," ICLR 2022.
  - Song et al., "Consistency Models," ICML 2023.
  - Liu et al., "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation," arXiv 2023.
  - Wang et al., "ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation," arXiv 2023.
  - Luo et al., "Diff-Instruct: A Universal Approach for Transferring Knowledge from Pre-Trained Diffusion Models," arXiv 2023.
  - Sauer et al., "Adversarial Diffusion Distillation," arXiv:2311.17042, 2023.
  - Luo et al., "Latent Consistency Models," arXiv 2023.
  - Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion," ICLR 2023.
  - Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)," CVPR 2018.
  - Goodfellow et al., "Generative Adversarial Nets," NIPS 2014.
  - Kang et al., "Scaling up GANs for Text-to-Image Synthesis (GigaGAN)," CVPR 2023.

> **주의**: 논문 제공 PDF를 기반으로 답변을 작성하였습니다. DMD2, SwiftBrush, Hyper-SD 등 2024년 이후 후속 연구에 대한 상세 내용은 해당 논문 원문을 직접 확인하시기를 권장합니다.
