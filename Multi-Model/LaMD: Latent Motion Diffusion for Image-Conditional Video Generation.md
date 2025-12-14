
# LaMD: Latent Motion Diffusion for Image-Conditional Video Generation

## 1. 핵심 주장 및 주요 기여 요약

**LaMD (Latent Motion Diffusion)**는 비디오 생성을 **모션 생성 문제**로 재정의하는 혁신적인 패러다임을 제시합니다. 기존 비디오 생성 방법들과 달리, LaMD는 다음의 핵심 통찰력에 기반합니다:[1]

비디오 생성의 주된 어려움은 시각적 품질 향상이 아니라 **일관되고 자연스러운 모션 생성**에 있다는 점입니다. 따라서 LaMD는 다음과 같은 주요 기여를 제시합니다:[1]

- **새로운 잠재 모션 생성 패러다임**: 종전의 픽셀 공간이나 일반적인 잠재 비디오 공간에서의 생성이 아닌, 모션 분해를 기반으로 한 잠재 공간에서의 생성을 제안합니다.[1]

- **MCD-VAE (Motion-Content Decomposed Video Autoencoder)**: 외형(Content)과 모션(Motion)을 효과적으로 분리하는 비디오 자동인코더를 설계하여, 각 요소를 최적으로 처리할 수 있게 합니다.[1]

- **DMG (Diffusion-based Motion Generator)**: 분해된 모션 잠재 공간에서 조건부 모션 생성을 수행하며, 이미지 확산 모델과 유사한 계산 복잡도를 달성합니다.[1]

- **샘플링 속도 혁신**: 2D-UNet 기반 확산 모델을 사용하여 비디오 확산 모델 중 가장 빠른 샘플링 속도(이미지 확산 모델과 유사)를 달성합니다.[1]

***

## 2. 문제 정의, 제안 방법, 모델 구조 및 성능

### 2.1 해결하고자 하는 문제

비디오 생성 분야는 세 가지 주요 패러다임으로 분류됩니다:[1]

| 패러다임 | 장점 | 단점 |
|---------|------|------|
| **픽셀 공간 확산** | 직관적 | 계산 비용 극도로 높음, 고주파 세부사항에 과도한 비중 |
| **잠재 이미지 공간** | 프레임별 고품질 | 프레임 간 시간적 불일치, 모션 일관성 부족 |
| **잠재 비디오 공간** | 시간적 일관성 | 3D 시공간 정보 모델링의 높은 복잡도 |

LaMD는 이러한 문제를 **모션-외형 분해(Motion-Content Decomposition)**를 통해 해결합니다. 핵심 통찰은: 이미지 조건부 비디오 생성에서는 외형이 이미 주어져 있으므로, **모션 생성에만 집중**할 수 있다는 것입니다.[1]

### 2.2 제안하는 방법 및 수식

#### **단계 1: MCD-VAE를 통한 모션-외형 분해**

MCD-VAE는 세 가지 구성요소로 이루어집니다:[1]

- **이미지 인코더** $$E_I$$: 2D-CNN 기반 U-Net 아키텍처로, 첫 프레임 $$x_0$$을 외형 잠재 $$f_{x_0} \in \mathbb{R}^{h \times w \times d'}$$로 인코딩합니다.
- **모션 인코더** $$E_M$$: 경량 3D-UNet 기반으로, 비디오 $$x_{0:L}$$에서 모션 정보를 추출합니다.
- **융합 디코더** $$D_V$$: 외형과 모션 잠재를 결합하여 비디오 픽셀을 재구성합니다.

**모션-외형 분해의 핵심: 정보 병목(Information Bottleneck)**

모션 인코더의 출력에 KL-제약을 적용합니다:[1]

$$z_m = \mu_\theta(E_M(x_{0:L})) + \varepsilon \cdot \sigma_\theta(E_M(x_{0:L})) \quad \quad (1)$$

여기서 $$\varepsilon \sim \mathcal{N}(0, I)$$입니다.

이를 통해 모션 잠재 $$z_m \in \mathbb{R}^{h \times w \times d}$$는 다음과 같이 정규화됩니다:

$$z_m = \frac{z_m - \mu}{\sigma + \epsilon}$$

**비디오 재구성 손실함수:**

$$\arg\min_{E_I, E_M, D_V} \max_D \mathbb{E}_{x \sim p(x)} [\mathcal{L}_{GEN} + \lambda \mathcal{L}_{GAN}]$$

$$\mathcal{L}_{GAN} = \log D(x_{0:L}) + \log(1 - D(\hat{x}_{0:L}))$$

$$\mathcal{L}_{GEN} = \|x_{0:L} - \hat{x}_{0:L}\|_1 + \text{LPIPS}(x_{0:L}, \hat{x}_{0:L})$$
$$+ \beta \text{KL}(q_{\mu_\theta, \sigma_\theta, E_M}(z_m | x_{0:L}) \| \mathcal{N}(0, I)) \quad \quad (2)$$

여기서:
- $$\lambda$$: 적응형 가중치
- $$\beta$$: KL-제약의 강도를 조절하는 하이퍼파라미터

**모션 차원 제거의 혁신:**

$$r_t = L$$로 설정하여 **시간 차원을 채널 차원으로 변환**합니다. 이는:[1]

$$z_m \text{의 형태: } \mathbb{R}^{16 \times 16 \times 3} \text{ (BAIR 데이터셋)}$$

로 극도로 컴팩트한 표현을 달성합니다.

#### **단계 2: DMG를 통한 모션 생성**

MCD-VAE의 모션 잠재를 기반으로, DDPM (Denoising Diffusion Probabilistic Models) 기반의 모션 생성기를 학습합니다.[1]

**정방향 확산 프로세스:**

$$q(z_m^t | z_m^{t-1}) = \mathcal{N}(z_m^t; \sqrt{1-\beta_t} z_m^{t-1}, \beta_t I) \quad \quad (3)$$

**마진 확률 분포:**

$$q(z_m^t | z_m^0) = \mathcal{N}(z_m^t; \sqrt{\bar{\alpha}_t} z_m^0, (1-\bar{\alpha}_t) I) \quad \quad (4)$$

여기서:
- $$\bar{\alpha}\_t = \prod_{s=1}^t \alpha_s$$
- $$\alpha_t = 1 - \beta_t$$

**역방향 디노이징 프로세스:**

$$p_\theta(z_m^{t-1} | z_m^t, y) = \mathcal{N}(z_m^{t-1}; \mu_\theta(z_m^t, t, y), \sigma_t^2 I)$$

$$\mu_\theta(z_m^t, t, y) = \frac{1}{\sqrt{\alpha_t}}\left(z_m^t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(z_m^t, t, y)\right) \quad \quad (5)$$

**학습 목표:**

$$\mathcal{L}_{simple}(\theta) = \mathbb{E}_{t, \epsilon}\left[\|\epsilon - \epsilon_\theta(z_m^t, t, y)\|_2^2\right] \quad \quad (6)$$

여기서:
- $$\epsilon_\theta$$: 2D-UNet 기반 트레이너블 오토인코더
- $$y$$: 외형 특성 $$f_{x_0}$$, 선택적으로 클래스 레이블 또는 텍스트

### 2.3 모델 구조

**전체 아키텍처:**

```
┌─────────────────────────────────────────────────────────┐
│                    LaMD Framework                        │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  [학습 단계]                                             │
│  ├─ Stage 1: MCD-VAE 자기지도 학습                     │
│  │  ├─ 이미지 인코더 EI (2D-UNet)                       │
│  │  ├─ 모션 인코더 EM (경량 3D-UNet)                    │
│  │  │  └─ 정보 병목 (KL-제약)                          │
│  │  └─ 융합 디코더 DV (3D-UNet 기반)                    │
│  │     ├─ 다중 스케일 외형 특성 통합                    │
│  │     └─ 시간적 보간 블록                              │
│  │                                                       │
│  └─ Stage 2: DMG 학습 (MCD-VAE 고정)                   │
│     └─ 2D-UNet 기반 확산 모델                           │
│        ├─ 교차-어텐션 메커니즘                          │
│        └─ 다중 모달 조건화                              │
│                                                           │
│  [샘플링 단계]                                           │
│  1. DMG에서 정상분포로부터 모션 생성                    │
│  2. 조건: 주어진 이미지의 외형 특성 fx0               │
│  3. 생성된 모션 zm과 fx0을 MCD-VAE 디코더에 입력       │
│  4. 최종 비디오 프레임 출력                            │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**계산 복잡도 비교:**[1]

| 방법 | 입력 차원 | 모델 | 복잡도 |
|------|---------|------|--------|
| **픽셀 공간 확산** | $$L \times H \times W$$ | 3D-UNet | $$L \times H \times W$$ |
| **잠재 비디오 확산** | $$r_t \times (H/r_s) \times (W/r_s)$$ | 3D-UNet | $$r_t \times r_s^2$$ |
| **LaMD (잠재 모션)** | $$H/r_s \times W/r_s$$ | 2D-UNet | $$r_s^2$$ (독립적) |

LaMD는 **최소한 $$L \times r_s^2$$배** 픽셀 공간 확산보다 빠르고, **$$r_t$$에 선형 비례하지 않습니다**.

### 2.4 성능 향상

**5개 벤치마크 데이터셋에서의 평가:**[1]

#### **2.4.1 확률적 이미지-투-비디오 생성 (I2V)**

| 데이터셋 | 메트릭 | MOSO | cINN | DynamiCrafter | **LaMD** |
|---------|--------|------|------|---------------|----------|
| **BAIR** | FVD ↓ | 83.6 | 99.6 | - | **57.0** |
| **Landscape** | FVD ↓ | - | 134.4 | 120-130 | **100.7** |
| **Landscape** | Motion Smoothness ↑ | - | 99.41% | 96.89% | **99.47%** |

**결과 해석:**
- BAIR에서 FVD 33% 개선 (기존 최고 83.6 → 57.0)
- Landscape에서 FVD 25% 개선 (134.4 → 100.7)
- 모션 매끄러움에서 높은 점수 유지 (99.47%)

#### **2.4.2 클래스 안내 이미지-투-비디오 생성 (cI2V)**

| 데이터셋 | 메트릭 | LDM | LFDM | **LaMD** |
|---------|--------|------|------|----------|
| **NATOPS** | FVD ↓ | 344.81 | 195.17 | **196.67** |
| **NATOPS** | cFVD (표준편차) ↓ | 169.52 | 117.06 | **103.27** |
| **MUG (256×256)** | FVD ↓ | - | - | **49.62** |

#### **2.4.3 텍스트-이미지-투-비디오 생성 (TI2V)**

| 모델 | LPIPS ↓ | FID ↓ | FVD ↓ | 액션 정확도 ↑ | 표현 정확도 ↑ |
|------|---------|--------|--------|-----------|-----------|
| MAGE | 0.26 | 39.38 | 69.44 | 0.629 | 0.602 |
| MAGE+ | 0.09 | 3.44 | 23.43 | 0.735 | 0.707 |
| **LaMD** | **0.09** | **2.26** | **5.77** | **0.840** | **0.748** |

### 2.5 한계

저자들은 명시적으로 다음의 한계를 제시합니다:[1]

1. **짧은 비디오 제약**: 모션-외형 분해는 주로 **첫 프레임의 내용이 크게 변하지 않는 짧은 비디오**에 효과적합니다. 새로운 객체가 나타나는 긴 비디오에서는 모션 표현에 외형 정보가 혼합될 수 있습니다.

2. **이미지 조건부 생성 한정**: 현재 LaMD는 **이미지 조건부 비디오 생성**만 지원합니다. 순수 텍스트-투-비디오 생성을 위해서는 고급 이미지 생성 모델과의 결합이 필요합니다.

3. **폐쇄 도메인 평가**: 모든 평가가 **폐쇄 도메인 데이터셋**에서 수행되었습니다. 다만 저자들은 소형 모델에서의 효과와 다양한 모션 패턴에서의 성능을 바탕으로 **개방 도메인으로의 확장 가능성**을 시사합니다.

***

## 3. 모델의 일반화 성능 향상 가능성 (중점)

### 3.1 구조적 일반화 장점

LaMD의 설계는 **강력한 구조적 귀납 편향**을 제공합니다:[1]

**1) 모션-외형 분해의 일반성**

정보 병목을 통한 분해는 다양한 도메인에서 동작합니다:

| 도메인 | 특성 | LaMD의 일반화 |
|--------|------|--------------|
| 로봇 조작 (BAIR) | 결정론적, 제한된 모션 범위 | FVD 57.0 (극도로 우수) |
| 풍경 타임랩스 (Landscape) | 확률적, 다양한 모션 | FVD 100.7 (강력) |
| 신체 제스처 (NATOPS) | 클래스 제어, 사람 동작 | cFVD 432.62 (경합) |
| 얼굴 표정 (MUG) | 미세한 제어 움직임 | FVD 49.62 (강력) |
| 합성 장면 (CATER-GEN) | 세밀한 텍스트 제어 | 액션 정확도 84% |

**2) 컴팩트 표현의 강점**

$$z_m \in \mathbb{R}^{h \times w \times d}$$의 극도로 컴팩트한 표현은:[1]

- **낮은 모델 용량**: DMG의 모델 크기가 340M 파라미터로 제한되어, 데이터 효율성이 향상됩니다.
- **빠른 수렴**: 모션만 모델링하기 때문에 학습 곡선이 급격합니다.
- **노이즈 강건성**: 저차원 표현은 고주파 노이즈에 덜 민감합니다.

### 3.2 다양한 조건부 생성에의 적응성

**다중 모달 조건화 메커니즘:**[1]

$$p_\theta(z_m^{t-1} | z_m^t, y) = \mathcal{N}(z_m^{t-1}; \mu_\theta(z_m^t, t, y), \sigma_t^2 I)$$

여기서 $$y$$는 다음을 포함할 수 있습니다:

| 조건 유형 | 구현 방식 | 효과 |
|---------|---------|------|
| **외형 특성** | 다중 스케일 특성 $$f_{x_0}, f^1_{x_0}, \ldots, f^k_{x_0}$$ | 세부사항 보존 |
| **클래스 레이블** | 클래스 임베딩 → 교차-어텐션 | 제어 가능한 모션 |
| **텍스트 설명** | 텍스트 인코더 → 512D 조건화 | 세밀한 모션 제어 |

### 3.3 제거 실험을 통한 일반화 분석

#### **3.3.1 모션 용량 분석:**[1]

| 채널 크기 $$d$$ | PSNR ↑ | SSIM ↑ | FID ↓ | LPIPS ↓ | FVD ↓ |
|--------|---------|--------|--------|----------|--------|
| **1** | 29.88 | 0.958 | 6.24 | 0.03 | 41.91 |
| **2** | 29.73 | 0.955 | 6.26 | 0.03 | 42.96 |
| **3** | **31.64** | **0.969** | **5.52** | **0.02** | **36.05** |
| **4** | 31.32 | 0.967 | 5.81 | 0.02 | 38.84 |

**발견**: $$d=3$$에서 최적 성능. 더 큰 채널도 가능하지만 과잉 용량이 발생합니다. 이는 **모션이 극도로 압축 가능**함을 시사합니다.

#### **3.3.2 시간 차원 제거의 효과:**[1]

| 시간 차원 유지 여부 | PSNR | SSIM | FVD |
|------------------|------|------|-----|
| **유지** ($$r_t \neq L$$) | 30.51 | 0.963 | 44.28 |
| **제거** ($$r_t = L$$) | **31.64** | **0.969** | **36.05** |

시간 차원 제거가 **성능 향상 및 계산 효율성**을 동시에 달성합니다.

### 3.4 재구성-생성 트레이드오프 분석

표 13에서 보여지는 흥미로운 현상:[1]

| 잠재 공간 크기 | β | $$d$$ | 재구성 FVD | **생성 FVD** |
|-------------|---|------|----------|-----------|
| **더 큼** | $$10^{-5}$$ | 4 | 37.35 | 127.1 |
| **더 작음** | $$10^{-2}$$ | 3 | 58.59 | **100.7** |

**일반화 원칙**: 더 컴팩트한 잠재 공간($$\beta$$ 증가, $$d$$ 감소)이 **생성 모델의 학습을 용이하게** 합니다. 이는 **정보 이론의 원리**와 일치합니다.

### 3.5 개방 도메인으로의 확장 가능성

저자들은 다음을 시사합니다:[1]

> "작은 모델 크기와 다양한 모션 패턴에서의 효과성은 개방 도메인 비디오 생성으로의 확장 가능성을 시사합니다."

**잠재력:**

1. **데이터 효율성**: 극도로 컴팩트한 표현으로 인해 **대규모 데이터셋 요구 감소**
2. **모듈식 구성**: MCD-VAE와 DMG의 분리로 **각 컴포넌트의 독립적 개선 가능**
3. **빠른 파인튜닝**: 이미지 생성 모델처럼 사전학습된 DMG를 **빠르게 적응** 가능

***

## 4. 논문의 영향과 향후 연구 고려사항

### 4.1 학문적 영향

#### **4.1.1 새로운 연구 방향 개척**

LaMD는 비디오 생성의 **패러다임 시프트**를 제시합니다:[1]

- **기존 패러다임**: 비디오 = 공간 정보 + 시간 정보 (균형 있는 모델링)
- **LaMD 패러다임**: 비디오 = 외형 (사전주어짐) + 모션 (모델링 대상)

이 개념적 전환은:
- 모션 표현 학습의 독립적 최적화
- 모션 제어 가능성의 향상
- 샘플링 효율성의 대폭 개선

을 가능하게 합니다.

#### **4.1.2 관련 분야의 영향**

| 분야 | 영향 | 구체적 적용 |
|------|------|----------|
| **모션 표현 학습** | 모션의 컴팩트성 재인식 | 모션 임베딩 연구에 새 기준 |
| **비디오 압축** | 모션 부호화 최적화 | HEVC/VVC 모션 예측 개선 |
| **동작 캡처** | 3D 포즈 표현 | 경량 모션 코덱 개발 |
| **조건부 생성** | 다중 도메인 확장성** | 텍스트, 골격 등 다양한 조건 |

### 4.2 2020년 이후 관련 최신 연구 비교 분석

#### **4.2.1 주요 경쟁 방법들과의 비교**

**1) LFDM (Latent Flow Diffusion Models, 2023)**[2]

| 특성 | LFDM | LaMD |
|------|------|------|
| **모션 표현** | 광학 유동 (optical flow) | 컴팩트 모션 잠재 |
| **확산 모델** | 3D-UNet 기반 | 2D-UNet 기반 |
| **샘플링 속도** | 16프레임: 10.2s | 16프레임: 9.7s |
| | 128프레임: 40.1s | 128프레임: 10.7s |
| **모션 표현성** | 프레임 간 흐름 제약 | 자유로운 표현 |
| **NATOPS FVD** | 195.17 | 196.67 |

**분석**: LaMD는 유사한 단기 성능에서 **극적인 장기 비디오 생성 우위**(128프레임: 3.75배 빠름)를 보입니다.[1]

**2) CMD (Content-Motion Latent Diffusion, 2024)**[3]

| 특성 | CMD | LaMD |
|------|------|------|
| **분해 전략** | 외형 프레임 + 모션 잠재 | 외형 특성 + 모션 잠재 |
| **기반 모델** | 사전학습 이미지 확산 활용 | 독립 학습 |
| **샘플링 시간** | 3.1초 (512×1024, 16프레임) | 9.8초 (128×128, 32프레임) |
| **계산 효율** | 이미지 생성 활용 | 2D 구조 활용 |

**분석**: CMD는 사전학습 활용으로 **데이터 효율성 우위**, LaMD는 **독립적 구조의 단순성 우위**[3]

**3) Video LDM (2023) / Align Your Latents**[4]

| 특성 | Video LDM | LaMD |
|------|-----------|------|
| **잠재 공간** | 3D 비디오 잠재 | 2D 모션 잠재 |
| **아키텍처** | 3D-UNet + 시간 보간 | 2D-UNet |
| **16프레임** | 13.8초 | 9.7초 |
| **128프레임** | 26.6초 | 10.7s |

**분석**: LaMD의 **선형이 아닌 확장성**이 주요 우위점입니다.[1]

#### **4.2.2 최신 트렌드 비교**

**2024-2025년 출현 방법들:**

| 논문 | 년도 | 핵심 아이디어 | LaMD와의 관계 |
|------|------|------------|------------|
| **Progressive Auto-Regressive Video Diffusion** | 2024 | 장기 비디오 생성의 진행형 노이즈 스케줄 | 상보적 (모션이 안정적일 때 효율성 극대) |
| **OnlyFlow** | 2024 | 광학 유동 기반 모션 제어 | 유사한 모션 표현 개념 |
| **Track4Gen** | 2024 | 포인트 추적을 통한 공간적 일관성 | 계층적 접근 가능 |
| **GenRec** | 2024 | 생성과 인식의 통합 | 모션 표현의 언더스탠딩 활용 |
| **Hi-VAE** | 2025 | 계층적 전역/세부 모션 분해 | 직접적 발전 (더 정교한 모션 분해) |
| **VidTwin** | 2025 | 구조와 동역학의 완전 분리 | 병렬적 진화 |
| **WF-VAE** | 2024 | 웨이블릿 기반 에너지 흐름 | 모션 표현의 주파수 분석적 접근 |
| **Survey of Video Diffusion** | 2025 | 확산 모델 기반 비디오 생성 종합 리뷰 | LaMD를 중요 패러다임으로 위치 |

### 4.3 향후 연구 시 고려할 점

#### **4.3.1 기술적 개선 방향**

**1) 모션 표현의 의미론적 해석**

현재 LaMD의 모션 잠재는 해석 불가능합니다. 향후 연구 방향:

- **선형 모션 분해 (Linear Motion Decomposition, LMD)** 적용: 모션을 직교 기저의 조합으로 표현
- **의미론적 모션 부분공간 학습**: 회전, 이동, 변형 등 명시적 성분으로 분해
- 참고: InMoDeGAN (2021), LIA (2024)에서의 LMD 구현[5]

**2) 계층적 모션 분해**

Hi-VAE (2025) 아이디어의 적용:[6]

$$z_m = z_m^{global} + z_m^{detailed}$$

- **전역 모션** $$z_m^{global}$$: 저주파 움직임 (카메라 이동, 전체 객체 이동)
- **세부 모션** $$z_m^{detailed}$$: 고주파 움직임 (변형, 표정)

**3) 장기 비디오 확장**

현재 한계: 새로운 객체 나타남에 의한 외형 변화

**해결책:**
- **점진적 외형 업데이트**: 중간 프레임의 외형 특성을 주기적으로 업데이트
- **조건부 모션 마스킹**: 새로운 영역의 모션만 생성
- **참고**: Progressive AutoRegressive Video Diffusion (2024)의 프로그레시브 노이즈 스케줄 활용[7]

#### **4.3.2 확장 가능성 탐색**

**1) 개방 도메인 비디오 생성**

**데이터 효율성 실험:**
- Kinetics-700, WebVid-10M 같은 대규모 데이터셋에서의 사전학습
- 폐쇄 도메인 파인튜닝의 수렴 속도 측정

**2) 다중 조건 결합**

현재: 하나의 조건 유형 (클래스 또는 텍스트)

향후:
- 텍스트 + 광학 유동 동시 조건화
- 스케치 + 포인트 추적 + 카메라 궤적 결합
- **참고**: FloVD (카메라 제어가 있는 광학 유동 확산)[8]

**3) 3D/4D 생성으로의 확장**

LaMD의 모션 분해가 3D 공간에서의 의미론적 동역학 학습을 가능하게 할 수 있습니다.

**체계적 계획:**

$$\text{2D 모션 잠재} \rightarrow \text{2D 광학 유동} \rightarrow \text{3D 장면 흐름}$$

#### **4.3.3 평가 메트릭 개선**

현재 한계: FVD, LPIPS 등이 **모션 품질의 세부사항**을 포착하지 못함

**개선안:**

1. **모션 일관성 메트릭** (VBench에서 일부 시도)
   - 광학 유동 매끄러움 (Motion Smoothness)
   - 동적 정도 (Dynamic Degree)
   - 배경 일관성 (Background Consistency)

2. **의미론적 모션 정확도**
   - 행동 인식 네트워크를 통한 제어 충실도
   - 참고: CATER-GEN의 액션 정확도, 참조 표현 정확도[1]

3. **사용자 연구 기반 메트릭**
   - 모션 자연성 Likert 척도
   - 조건 충실도 평가

#### **4.3.4 산업 응용 고려사항**

| 응용 분야 | 요구사항 | LaMD의 준비도 |
|---------|--------|------------|
| **실시간 비디오 편집** | <100ms 샘플링 | 중간 (9-10초 필요) |
| **고해상도 생성** | 4K 이상 | 낮음 (128×128 평가) |
| **인터랙티브 제어** | 정밀한 모션 조종 | 높음 (다중 조건화) |
| **모션 라이브러리** | 모션 재사용 가능성 | 중간-높음 (의미론적 해석 필요) |

### 4.4 이론적 기여

#### **4.4.1 정보 이론 관점**

LaMD는 비디오를 다음과 같이 분석합니다:

$$I(V) = I_C + I_M + I_{CM}$$

여기서:
- $$I_C$$: 외형 정보량 (이미지 조건부 때문에 주어짐)
- $$I_M$$: 모션 정보량 (극도로 압축 가능)
- $$I_{CM}$$: 상호 정보 (최소화 대상)

**시사점**: 비디오의 정보론적 복잡도를 근본적으로 **감소**시킬 수 있습니다.

#### **4.4.2 확산 모델의 확장성**

일반적 원칙:
$$\text{생성 모델 복잡도} \propto \text{데이터 차원수} \times \text{시간 스텝수}$$

LaMD는:
- 차원수를 극도로 감소 ($$H \times W \times L \rightarrow H \times W \times 3$$)
- 시간 스텝은 표준적 수준 (1000)

**결과**: 이미지 확산 모델과 경쟁하는 속도 달성

### 4.5 종합 평가

| 측면 | 강점 | 약점 | 향후 개선 |
|------|------|------|---------|
| **표현 능력** | 다양한 도메인 적응 | 긴 비디오에서 외형 변화 미처리 | 점진적 외형 업데이트 |
| **계산 효율** | 극도로 빠른 샘플링 | 고해상도 미지원 | 계층적 생성 |
| **제어 가능성** | 다중 모달 조건화 | 의미론적 모션 조절 불가 | 선형 모션 분해 |
| **학습 효율** | 컴팩트 표현 | 대규모 데이터셋 미평가 | 개방 도메인 실험 |
| **확장성** | 패러다임의 일반성 | 기존 사전학습 활용 제한 | 하이브리드 접근 |

***

## 5. 결론

LaMD는 **비디오 생성의 본질적 어려움을 정확히 진단**하고, **모션-외형 분해라는 우아한 해결책**을 제시한 획기적 논문입니다.[1]

**주요 기여:**

1. **개념적**: 비디오 생성을 모션 생성 문제로 재정의
2. **기술적**: 정보 병목을 통한 효과적 분해 메커니즘
3. **실무적**: 이미지 확산 모델 수준의 샘플링 속도 달성

**향후 연구의 방향성:**

- 의미론적 모션 표현의 개발
- 장기/개방 도메인 비디오로의 확장
- 다중 조건의 정교한 통합
- 3D/4D 생성으로의 진화

LaMD는 단순히 한 논문의 기여를 넘어, **비디오 생성 연구의 패러다임 시프트**를 주도하고 있으며, 2024-2025년의 후속 연구들(Hi-VAE, VidTwin, Progressive AR-VDM 등)이 이를 다양한 방향으로 확장하고 있습니다.[5][6][3]

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ae22127c-e593-42f2-bee3-0e5ed6e81e79/2304.11603v2.pdf)
[2](https://www.semanticscholar.org/paper/Conditional-Image-to-Video-Generation-with-Latent-Ni-Shi/b8b5015b153709176385873e34339f9e520d128f)
[3](https://arxiv.org/abs/2403.14148)
[4](https://research.nvidia.com/labs/toronto-ai/VideoLDM/)
[5](https://www.semanticscholar.org/paper/72552a6a546de9501503b1c3e6593922b2d00cb8)
[6](https://arxiv.org/html/2506.07136v1)
[7](https://ieeexplore.ieee.org/document/11147496/)
[8](https://jinwonjoon.github.io/flovd_site/)
[9](https://ieeexplore.ieee.org/document/11147639/)
[10](https://ieeexplore.ieee.org/document/11147881/)
[11](https://arxiv.org/abs/2504.16081)
[12](https://arxiv.org/abs/2403.07711)
[13](https://ieeexplore.ieee.org/document/10655252/)
[14](https://ieeexplore.ieee.org/document/10656478/)
[15](https://ieeexplore.ieee.org/document/11093056/)
[16](https://arxiv.org/abs/2408.15241)
[17](https://ieeexplore.ieee.org/document/11228639/)
[18](https://arxiv.org/html/2306.11173)
[19](http://arxiv.org/pdf/2408.13423.pdf)
[20](http://arxiv.org/pdf/2305.13840v1.pdf)
[21](https://www.mdpi.com/1099-4300/25/10/1469/pdf?version=1697803890)
[22](http://arxiv.org/pdf/2407.08737.pdf)
[23](https://arxiv.org/html/2410.20502v1)
[24](https://arxiv.org/pdf/2204.03458.pdf)
[25](https://arxiv.org/html/2401.12945v1?s=09)
[26](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
[27](https://snap-research.github.io/video-synthesis-tutorial/)
[28](https://geometry.cs.ucl.ac.uk/courses/diffusion_ImageVideo_sigg25/)
[29](https://openaccess.thecvf.com/content/CVPR2023/papers/Ni_Conditional_Image-to-Video_Generation_With_Latent_Flow_Diffusion_Models_CVPR_2023_paper.pdf)
[30](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Learning_Video_Representations_of_Human_Motion_From_Synthetic_Data_CVPR_2022_paper.pdf)
[31](https://openaccess.thecvf.com/content/CVPR2025W/CVEU/papers/Xie_Progressive_Autoregressive_Video_Diffusion_Models_CVPRW_2025_paper.pdf)
[32](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/lamd/)
[33](https://ieeexplore.ieee.org/document/9879578/)
[34](https://arxiv.org/abs/2412.07772)
[35](https://arxiv.org/abs/2303.13744)
[36](https://arxiv.org/html/2511.23428v1)
[37](https://arxiv.org/abs/2410.08151)
[38](https://arxiv.org/abs/2304.08818)
[39](https://arxiv.org/html/2503.10096v2)
[40](https://arxiv.org/html/2509.15130v2)
[41](https://jaesik.info/publications/data/20_cvpr_future.pdf)
[42](https://learnopencv.com/video-generation-models/)
[43](https://diffusion.kaist.ac.kr)
[44](https://ieeexplore.ieee.org/document/10645735/)
[45](https://link.springer.com/10.1007/s11263-025-02386-7)
[46](https://arxiv.org/abs/2304.11603)
[47](https://arxiv.org/abs/2404.11576)
[48](https://arxiv.org/abs/2306.00559)
[49](https://www.mdpi.com/2079-9292/13/22/4415)
[50](https://arxiv.org/abs/2405.20279)
[51](https://ieeexplore.ieee.org/document/11093251/)
[52](https://arxiv.org/html/2412.17726)
[53](https://arxiv.org/pdf/2411.17459v1.pdf)
[54](https://arxiv.org/abs/2201.06888)
[55](https://arxiv.org/html/2409.01199)
[56](https://arxiv.org/pdf/2503.14325.pdf)
[57](https://arxiv.org/html/2402.13729v3)
[58](https://liner.com/ko/review/executing-your-commands-via-motion-diffusion-in-latent-space)
[59](https://www.emergentmind.com/topics/optical-video-generation-model)
[60](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_MOSO_Decomposing_MOtion_Scene_and_Object_for_Video_Prediction_CVPR_2023_paper.pdf)
[61](https://arxiv.org/html/2304.11603v2)
[62](https://liner.com/ko/review/decomposing-motion-and-content-for-natural-video-sequence-prediction)
[63](https://arxiv.org/abs/1706.08033)
[64](https://openreview.net/forum?id=dQVtTdsvZH)
[65](https://openaccess.thecvf.com/content/CVPR2024W/DFAD/papers/K_Latent_Flow_Diffusion_for_Deepfake_Video_Generation_CVPRW_2024_paper.pdf)
[66](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_DeMatch_Deep_Decomposition_of_Motion_Field_for_Two-View_Correspondence_Learning_CVPR_2024_paper.pdf)
[67](https://arxiv.org/html/2412.04452v2)
[68](http://papers.neurips.cc/paper/7333-learning-to-decompose-and-disentangle-representations-for-video-prediction.pdf)
