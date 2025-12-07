# Conditional Image Generation with Score-Based Diffusion Models

### 1. 논문의 핵심 주장 및 주요 기여도

**"Conditional Image Generation with Score-Based Diffusion Models"** 논문의 핵심 기여는 다음과 같습니다:[1]

1. **이론적 정당성 제공**: 조건부 이미지 생성에서 가장 성공적인 조건부 점수(conditional score) 추정기인 **조건부 노이즈 제거 추정기(Conditional Denoising Estimator, CDE)**에 대한 엄밀한 이론적 증명을 제공합니다.[1]

2. **다중속도 확산 프레임워크 도입**: 기존의 단일 확산속도 방식을 넘어 입력 x와 조건 y가 **서로 다른 속도로 확산**되는 "다중속도 확산(Multi-Speed Diffusion)" 개념을 제안하고, 이를 기반으로 한 새로운 조건부 점수 추정기 **CMDE(Conditional Multi-Speed Diffusive Estimator)**를 개발합니다.[1]

3. **체계적 비교 및 분석**: CDE, CDiffE(조건부 확산 추정기), CMDE 세 가지 접근법을 초해상도(super-resolution), 인페인팅(inpainting), 엣지-이미지 변환(edge-to-image translation) 등 다양한 역문제에서 실증적으로 비교분석합니다.[1]

4. **오픈소스 라이브러리 제공**: 연구의 재현성과 확장성을 위해 **MSDiff** 라이브러리를 제공하여, 다중 SDE(Stochastic Differential Equation) 기반의 확산 모델링을 가능하게 합니다.[1]

***

### 2. 해결하고자 하는 문제 및 제안하는 방법

#### 2.1 문제 정의

스코어 기반 확산 모델은 무조건 생성(unconditional generation)에서 뛰어난 성능을 보이지만, **조건부 확률분포 $\(p(x|y)\)$ 학습에는 여러 접근법이 존재하면서도 이들의 이론적 정당성과 최적 전략이 명확하지 않은 상황**이었습니다[1].

주요 문제점:
- CDE가 실무에서 성공적으로 사용되었지만 **이론적 증명이 부재**
- 다양한 조건부 점수 추정 방식 간의 **성능 비교 부재**
- 최적화 오류(optimization error)와 근사 오류(approximation error) 간의 **트레이드오프 미해결**

#### 2.2 제안하는 방법론

**A) 조건부 노이즈 제거 추정기(CDE)**

역시간 SDE의 조건부 형태:[1]

$$\frac{dx}{dt} = \mu(x,t) - \sigma(t)^2 \nabla_x \ln p(x_t|y) dt + \sigma(t) d\bar{w}$$

CDE는 다음 손실함수를 최소화하여 학습합니다:[1]

$$L_{CDE} = \frac{1}{2} \mathbb{E}_{t \sim \mathcal{U}[0,T]} \mathbb{E}_{x_0, y \sim p(x_0,y)} \left\| \sigma(t) \nabla_{x_t} \ln p(x_t|x_0) - s_\theta(x_t, y, t) \right\|^2_2$$

여기서 논문은 **정리 1(Theorem 1)**을 통해 다음을 증명합니다:[1]

$$\min_\theta \frac{1}{2} \mathbb{E}_{t \sim \mathcal{U}[0,T]} \mathbb{E}_{x_0,y \sim p(x_0,y)} \left\| \sigma(t) \nabla_{x_t} \ln p(x_t|x_0) - s_\theta(x_t, y, t) \right\|^2_2$$

는 다음과 같음:[1]

$$\min_\theta \frac{1}{2} \mathbb{E}_{t \sim \mathcal{U}[0,T]} \mathbb{E}_{x_t,y \sim p(x_t,y)} \left\| \sigma(t) \nabla_{x_t} \ln p(x_t|y) - s_\theta(x_t, y, t) \right\|^2_2$$

이는 **추론 1(Corollary 1)**로 이어져 CDE의 일관성(consistency)을 보장합니다.[1]

**B) 조건부 확산 추정기(CDiffE)**

x와 y를 모두 확산시킨 후 결합 점수를 학습하는 방식:[1]

$$L_{CDiffE} = \frac{1}{2} \mathbb{E}_{t \sim \mathcal{U}[0,T]} \mathbb{E}_{z_0 \sim p(z_0)} \left\| \sigma(t) \nabla_{z_t} \ln p(z_t|z_0) - s_\theta(z_t, t) \right\|^2_2$$

여기서 $\(z_t = (x_t, y_t)\)$ 입니다.[1]

**C) 다중속도 조건부 확산 추정기(CMDE) - 핵심 기여**

서로 다른 확산속도를 갖는 SDE로 정의:[1]

$$\frac{dx}{dt} = \mu(x,t)dt + \sigma_x(t)dw$$
$$\frac{dy}{dt} = \mu(y,t)dt + \sigma_y(t)dw$$

CMDE의 훈련 목적함수:[1]

$$L = \frac{1}{2} \mathbb{E}_{t \sim \mathcal{U}[0,T]} \mathbb{E}_{z_0 \sim p(z_0)} \mathbf{v}^T \boldsymbol{\Lambda}_t^{MLE} \mathbf{v}$$

여기서 $\(\mathbf{v} = \nabla_{z_t} \ln p(z_t|z_0) - s_\theta(z_t, t)\)$ 이고, **최대우도 가중치 행렬(MLE Weighting Matrix)**은[1]:

$$\boldsymbol{\Lambda}^{MLE}_{i,j}(t) = \begin{cases} \sigma_x(t)^2 & \text{if } i=j, i \leq n_x \\ \sigma_y(t)^2 & \text{if } i=j, n_x < i \leq n_y \\ 0 & \text{otherwise} \end{cases}$$

**정리 2(Theorem 2)**는 이 목적함수가 음의 로그-우도의 상한을 제공함을 증명합니다:[1]

$$\mathbb{E}_{x,y} [-\ln p(x,y)] \leq L_{CMDE} + C$$

**정리 3(Theorem 3)**는 근사오류의 상한을 보이며, $\(\sigma_y(t) \to 0\)$ 일 때 CMDE가 CDE로 수렴함을 증명합니다:[1]

$$\mathbb{E}_{y_t \sim p(y_t|x_t)} \left\| \nabla_{x_t} \ln p(x_t|y_t) - \nabla_{x_t} \ln p(x_t|y) \right\|^2_2 \leq E\left(\frac{1}{\sigma_y(t)^2}\right)$$

여기서 \(E\)는 단조 감소하여 0으로 수렴하는 함수입니다.[1]

***

### 3. 모델 구조 및 아키텍처

#### 3.1 신경망 아키텍처

논문에서 사용한 신경망 모델은 **DDPM 기반 U-Net 구조**:[1]

- **채널 차원**: 96
- **깊이 배수(depth multipliers)**:[2][3][1]
- **ResNet 블록**: 각 스케일당 개수 가변
- **어텐션**: 마지막 3개 스케일에 적용
- **총 파라미터**: 43.5M

#### 3.2 분산 스케줄

**Variance-Exploding(VE) SDE** 사용:[1]

$$\frac{dx}{dt} = -\frac{d\sigma^2(t)}{2\sigma(t)}x dt + \sigma(t)dw$$

여기서:

$$\sigma(t) = \sigma_{min}(1 + t(\sigma_{max}/\sigma_{min} - 1))$$

#### 3.3 VS-CMDE(분산 감소 스케줄 CMDE)

이산 훈련에서의 수렴 문제를 해결하기 위해 도입:[1]

$$\sigma_y^n = \sigma_y^{min} + (\sigma_y^{max} - \sigma_y^{min}) \cdot \frac{n}{M} \cdot \frac{\sigma_{target}^{max} - \sigma_y^{min}}{\sigma_y^{max} - \sigma_y^{min}}$$

훈련 초기에 $\(\sigma_y\)$ 를 높게 시작하여 점진적으로 감소시켜 최적화 난이도를 단계적으로 증가시킵니다.[1]

***

### 4. 성능 향상 및 실험 결과

#### 4.1 평가 지표

- **PSNR(Peak Signal-to-Noise Ratio)**: 재구성 품질
- **SSIM(Structural Similarity)**: 구조적 유사도  
- **LPIPS(Learned Perceptual Image Patch Similarity)**: 지각적 유사도
- **FID(Fréchet Inception Distance)**: 분포 간 거리 (UFID: 무조건 분포, JFID: 결합 분포)
- **일관성(Consistency)**: $\(||y - Ax̃||^2\)$
- **다양성(Diversity)**: 5개 샘플의 픽셀 표준편차[1]

#### 4.2 초해상도 실험(8배)

CelebA 데이터셋, 20×20 → 160×160:[1]

| 추정기 | PSNR | SSIM | LPIPS | UFID | JFID |
|--------|------|------|-------|------|------|
| **CDE** | 23.80 | 0.650 | 0.114 | 10.36 | 15.77 |
| **CDiffE** | 23.83 | 0.656 | 0.139 | 14.29 | 20.20 |
| **CMDE** | 23.91 | 0.654 | **0.109** | **10.28** | **15.68** |
| **HCFlow** | 24.95 | 0.702 | 0.107 | 14.13 | 19.55 |

**주요 결과**: CMDE가 CDE와 유사한 재구성 오류를 유지하면서 **더 나은 FID 점수**를 달성하여 분포 적합도에서 우수함.[1]

#### 4.3 인페인팅 실험

CelebA 데이터셋, 25% 랜덤 마스킹:[1]

| 추정기 | PSNR | SSIM | LPIPS | UFID | JFID |
|--------|------|------|-------|------|------|
| **CDE** | 25.12 | 0.870 | 0.042 | 13.07 | 18.06 |
| **CDiffE** | 23.07 | 0.844 | 0.057 | 13.28 | 19.25 |
| **CMDE** | 24.92 | 0.864 | 0.044 | **12.07** | **17.07** |

**주요 결과**: CMDE의 JFID(17.07)가 CDE(18.06)보다 개선되어 **후방분포(posterior distribution) 근사 성능 향상** 입증.[1]

#### 4.4 엣지-이미지 변환 실험

Edges2shoes 데이터셋, 신경망 기반 엣지 감지기 사용:[1]

| 추정기 | PSNR | SSIM | LPIPS | UFID | JFID |
|--------|------|------|-------|------|------|
| **CDE** | **18.35** | **0.699** | **0.156** | **11.87** | **21.31** |
| **CDiffE** | 10.00 | 0.365 | 0.350 | 33.41 | 55.22 |
| **CMDE** | 18.16 | 0.692 | 0.158 | 12.62 | 22.09 |

**주요 결과**: CDiffE의 **심각한 성능 저하**로 인해 CDE가 우월하며, 이는 강한 조건(strong conditioning)에서 충분한 확산이 불필요함을 시사.[1]

#### 4.5 오류 분석: 최적화 vs 근사 오류

그림 2의 오류 구조:[1]

```
CDE: 최적화 오류만 존재
     - 장점: 낮은 근사 오류
     - 단점: 직접 조건부 점수 학습의 높은 난이도

CDiffE: 최적화 오류 + 근사 오류  
     - 장점: 부드러운 손실 지형(smoother loss landscape)
     - 단점: 높은 근사 오류

CMDE: 최적화 오류 + 조절 가능한 근사 오류
     - 최적: σ_y를 조절하여 양자의 균형점 탐색
```

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 이론적 일반화 보장

**Theorem 1과 Corollary 1의 의의**:[1]

CDE의 일관성 증명은 **몬테카를로 샘플 수 → ∞일 때** 학습된 점수함수가 참 조건부 점수에 수렴함을 보장:

$$s_\theta(x, y, t) \xrightarrow{P} \nabla_x \ln p(x_t|y)$$

이는 충분한 훈련 데이터와 모델 용량이 있으면 **일반화 성능 달성 가능성**을 수학적으로 정당화합니다.[1]

#### 5.2 CMDE의 일반화 메커니즘

**Theorem 3의 시사점**:[1]

```
근사 오류: E(1/σ_y(t)²) → 0   (σ_y(t) → 0일 때)
```

조건의 확산속도를 감소시킴으로써:

1. **더 정확한 후방분포 모델링**: 조건 y의 정보 손실을 최소화
2. **개선된 FID 점수**: 5개 샘플 실험에서 CMDE가 CDE 대비 우수한 분포 적합도 달성
3. **강화된 다양성**: VS-CMDE에서 다양성(diversity) 지표 향상 관찰[1]

#### 5.3 아키텍처 선택과 일반화

논문에서 모든 추정기에 **동일한 U-Net 구조**를 적용하여 공정한 비교를 보장했으며, 이는 다음을 시사합니다:[1]

- **신경망 설계의 유연성**: DDPM 기반 U-Net이 다양한 추정 방식을 수용 가능
- **향상된 아키텍처의 잠재성**: Song et al.의 NCSN 아키텍처 도입 시 **모든 추정기의 성능 개선 가능**
- **확장성**: 더 깊은 네트워크나 Vision Transformer 기반 설계로 추가 개선 여지 존재

#### 5.4 역문제 해결에서의 일반화

세 가지 역문제(초해상도, 인페인팅, 엣지-이미지)에서:

1. **작업별 최적 전략**: 
   - 약한 조건(weak conditioning): CMDE 선호 (인페인팅)
   - 강한 조건(strong conditioning): CDE 동등 성능 (초해상도)

2. **조건 구조의 영향**: 조건의 정보량과 특성에 따라 $\(\sigma_y\)$ 하이퍼파라미터 자동 조절 가능성 시사[1]

***

### 6. 모델의 한계

#### 6.1 이론적 한계

1. **기술적 가정의 제한성**:[1]
   - Theorem 1 증명: 유한 데이터셋에 대해 완전한 최소값 수렴 보장 부재
   - Corollary 1: 실무에서의 정확한 수렴 속도 미명시

2. **근사 오류의 정량화 부재**:[1]
   - Theorem 3: $\(E(1/\sigma_y^2)\)$ 함수의 명시적 형태 미제시
   - 최적 $\(\sigma_y\)$ 선택의 정량적 지침 부재

#### 6.2 실험적 한계

1. **하이퍼파라미터 튜닝**:[1]
   - CMDE의 $\(\sigma_y^{max}\)$ 값을 "시행착오"로 결정
   - 체계적 격자탐색(grid search) 미수행으로 성능의 천장값 미확인
   - VS-CMDE의 추가 파라미터 $\(\sigma^{target}_{max}\)$ 선택 기준 불명확

2. **데이터셋 제한**:[1]
   - CelebA(얼굴): 상대적으로 단순 이미지
   - Edges2shoes: 주로 신발 도메인
   - **더 다양한 고해상도 이미지 데이터셋 필요**

3. **이산 훈련의 수렴 문제**:[1]
   - 원본 CMDE가 $\(T=1000\)$ 이산 스텝에서 불안정
   - VS-CMDE 도입으로 해결했으나 **연속 훈련과의 성능 차이 여전**

#### 6.3 비교 한계

1. **제한된 기준선**:[1]
   - 초해상도만 HCFlow와 비교
   - GAN 기반 최신 모델(e.g., StyleGAN2) 비교 부재
   - 다른 조건부 생성 모델(e.g., VAE, 정규화 흐름) 비교 없음

2. **오픈소스 한계**:[1]
   - MSDiff 라이브러리 공개 약속했으나 논문 발표 시점에 미완성 ("코드는 곧 공개될 예정")

***

### 7. 관련 최신 연구 (2020년 이후)

#### 7.1 조건부 확산 모델의 발전

**CSDI (2021년 7월)**: 시계열 결측값 보정을 위한 조건부 점수 기반 확산 모델로, 논문의 CDE 방식을 시계열 영역으로 확장하여 의료 및 환경 데이터에서 40-65% 성능 향상을 달성했습니다.[2]

**UMM-CSGM (2022년 7월)**: 의료 영상의 다중모달 결측 문제에 다중입출력 조건부 점수 네트워크(mm-CSN)를 적용하여 통합 프레임워크를 구축했습니다.[3]

**Latent Diffusion Models (2021년 12월)**: Rombach 등이 제안한 모델로, 픽셀 공간 대신 **잠재공간**에서 확산을 수행하여 계산 복잡도를 대폭 감소시켰으며, 이는 Stable Diffusion의 기반이 되었습니다.[4]

#### 7.2 점수 기반 모델의 이론 및 실무 진전

**Understanding Diffusion Models (2022년 8월)**: Song 등의 종합 리뷰로, 변분 확산 모델(VDM)과 점수 기반 관점을 통합하고, Tweedie의 공식을 통해 두 프레임워크의 연결성을 증명했습니다.[5]

**Text-to-Image Diffusion Models의 급속 발전**:[6][7][8][9]

- **DALL-E 2 (2022)**: 확산 기반 사전(prior) + 이미지 디코더 구조로 고품질 텍스트-이미지 생성 달성
- **Stable Diffusion (2022)**: 잠재공간 확산으로 고해상도 생성을 저비용으로 실현
- **eDiff-I (2023)**: 전문가 노이즈 제거기의 앙상블로 생성 품질 향상
- **DALL-E 3 (2024)**: GPT-4 API 통합으로 미세한 의미 표현 개선

#### 7.3 일반화 성능과 이론적 분석

**Towards a Mechanistic Explanation of Diffusion Model Generalization (2025년 2월)**: 확산 모델의 일반화 동작을 설명하는 훈련 무료 메커니즘을 제안하여, 네트워크 노이즈 제거기가 다양한 아키텍처에서 공유되는 **국소적 귀납적 편향**을 통해 일반화된다고 설명했습니다.[10]

**On the Generalization Properties of Diffusion Models (2024년)**: 확산 모델의 일반화 특성에 대한 엄밀한 이론적 분석으로, "모드 시프트(mode shift)" 현상이 일반화에 미치는 부정적 영향을 수학적으로 규명했습니다.[11]

**What's in a Latent? (2025년 3월)**: 확산 모델의 잠재공간이 도메인 적응에서 우수한 성능을 보이며, 명시적 도메인 라벨 없이도 도메인별 정보를 포착하여 **미확인 도메인으로의 일반화 4% 이상 개선**을 달성했습니다.[12]

#### 7.4 샘플링 효율성과 속도 개선

**Consistency Trajectory Models (2023년 9월)**: Consistency Model(CM)을 일반화하여 단일 포워드 패스에서 점수 함수를 출력하면서도 품질-속도 트레이드오프를 조절 가능하게 했으며, CIFAR-10에서 FID 1.73 달성.[13]

**Preconditioned Score-based Generative Models (2025년 2월)**: 전처리된 확산 샘플링(PDS) 방법으로 고해상도(1024×1024) 생성을 28배 가속화하면서 CIFAR-10에서 FID 1.99 달성.[14]

#### 7.5 역문제 해결 응용

**Solving Video Inverse Problems with Image Diffusion Models (2024년)**: 확산 모델을 이용한 영상 초해상도 및 인페인팅에서 상관 노이즈를 통한 일관성 유지, 기울기 기반 역문제 해법 제시.[15]

**Conditional score-based diffusion models for solving inverse problems in mechanics (2024-2025)**: 베이지안 추론 프레임워크로 역학 분야의 역문제(재료 특성 추정) 해결.[16]

**SSD (Shift Structured Diffusion) - Accelerating Diffusion Models for Inverse Problems (2024년)**: 중간 단계에서만 역투영을 적용하여 초해상도, 채색, 인페인팅, 제거 작업에서 고효율 해법 제시.[17]

#### 7.6 의료 영상 응용

**MedSegLatDiff (2025년 9월)**: 확산 모델과 VAE를 결합한 의료 이미지 분할로, 일대다(one-to-many) 패러다임으로 불확실성 정량화 및 의사 그룹의 분할 과정 모방.[18]

**DiffBoost (2025년)**: 텍스트 기반 확산 모델로 합성 의료 이미지 생성하여 초음파(+13.87%), CT(+0.38%), MRI(+7.78%) 분할 성능 향상.[19]

**Survey: Diffusion models in medical imaging (2023년)**: 712회 인용으로 의료 영상 분야에서의 확산 모델 적용 종합 리뷰.[20]

#### 7.7 신경망 아키텍처 혁신

**U-ViT: All are Worth Words (2022년 9월)**: Vision Transformer 기반 확산 모델 백본으로 ImageNet 256×256에서 FID 2.29, MS-COCO 텍스트-이미지에서 FID 5.48 달성.[21]

**On Improved Conditioning Mechanisms (2025년 1월)**: 의미론적 조건과 제어 메타데이터 조건을 분리하여 ImageNet-1k 클래스-조건 생성에서 FID 7% 개선, 고해상도(512) 텍스트-이미지에서 FID 8% 개선.[22]

#### 7.8 일반화와 도메인 적응

**Towards a Mechanistic Explanation of Diffusion Model Generalization (2025년 2월)**: 확산 모델의 일반화 메커니즘을 네트워크 노이즈 제거기의 **국소적 귀납적 편향**으로 설명하여 다양한 아키텍처에서의 강건성 입증.[10]

**Coherence-aware training for conditional diffusion (2025년 2월)**: 노이즈 있는 조건 정보의 품질을 모델링하여, 낮은 일관성 점수에서 조건을 무시/할인하는 CAD 방법으로 다양한 조건 생성 작업에서 현실성과 다양성 향상.[23]

***

### 8. 논문의 영향 및 앞으로의 연구 고려사항

#### 8.1 이 논문이 앞으로의 연구에 미치는 영향

**A) 이론적 토대 구축**

1. **조건부 점수 추정의 정당성 제공**: Theorem 1의 증명으로 실무에서 광범위하게 사용되는 CDE에 대한 **첫 번째 엄밀한 이론적 근거** 제공. 이후 조건부 확산 연구의 기본 토대 마련.[1]

2. **다중 SDE 일반화**: Theorem 2에서 다중 SDE에 대한 최대우도 가중치 행렬 도출로, **일반화된 확산 프레임워크** 가능성 열림.[1]

3. **근사-최적화 오류 트레이드오프 분석**: 향후 연구자들이 특정 작업에 맞는 확산속도 선택을 위한 **이론적 가이드라인** 제공.[1]

**B) 실무 응용 확대**

1. **MSDiff 라이브러리**: 오픈소스 공개로 다중속도 확산 기반 연구의 **재현성과 접근성** 향상.[1]

2. **역문제 해결의 표준화**: 초해상도, 인페인팅, 이미지 변환 등 다양한 역문제에서 **확산 기반 접근법의 실용성** 입증. 이후 3년간 의료영상, 영상 복구 등 광범위한 응용 확대.[15][16][18]

3. **조건부 생성의 체계적 비교**: 세 가지 추정기의 실험적 비교 프레임워크가 향후 새로운 조건부 점수 추정 기법 평가의 **벤치마크** 역할.[1]

**C) 다중속도 확산의 영향**

1. **최적화 문제 개선**: CMDE의 $\(\sigma_y\)$ 조절을 통한 손실 지형 개선 개념이, 후속 연구에서 **적응형 확산속도 스케줄** 개발로 확대.[1]

2. **일반화 성능 향상**: Theorem 3의 근사 오류 상한이 "조건 확산속도 감소 ⟹ 일반화 개선" 메커니즘을 명시적으로 보여주어, 향후 도메인 일반화 연구의 이론적 바탕 제공.[12][10][1]

#### 8.2 앞으로의 연구 시 고려할 점

**A) 이론적 진전 방향**

1. **최적 하이퍼파라미터 선택의 이론화**
   - 현재: $\(\sigma_y^{max}\)$ 시행착오 선택
   - 제안: 작업별 최적 $\(\sigma_y\)$ 결정을 위한 **정량적 지침 도출**
   - 예: 조건의 정보량( $\(I(x;y)\))$ 과 $\(\sigma_y\)$ 간의 해석적 관계식 유도

2. **비선형 손실 지형의 일관성 분석**
   - 현재: Theorem 1의 이선형(bilinear) 가정 범위
   - 제안: 신경망의 **비볼록성(non-convexity) 고려한 수렴 분석**
   - 중요성: 실무 네트워크의 복잡한 최적화 동역학 이해

3. **역 문제 기하학의 점수 적응**
   - 현재: 고정된 $\(\sigma_y\)$ 스케줄
   - 제안: 작업 특성(초해상도: 강한 조건 vs 인페인팅: 약한 조건)에 따른 **적응형 확산속도 정책**

**B) 실험적 개선 방향**

1. **체계적 하이퍼파라미터 탐색**

- 현재: $\(\sigma_y^{max}\) ∈ {0.5, 1.0}$ 수동 선택
- 개선: Bayesian optimization 또는 강화학습으로 $\(\sigma_y(t)\)$ 동적 결정
- 기대효과: 각 작업별 성능 천장값 달성


2. **고해상도 다중 도메인 평가**
   - 확대 필요: CelebA(얼굴) 외 일반 이미지 데이터셋
   - 제안: ImageNet, COCO, Cityscapes 등 고해상도 멀티 도메인 평가
   - 의의: 일반화 성능의 광범위한 검증

3. **이산-연속 훈련 갭 해결**
   - 현재: 이산 훈련(T=1000)에서 VS-CMDE 필요
   - 과제: **연속 훈련과 동등한 성능의 이산 훈련 알고리즘 개발**
   - 실용성: 제한된 계산 자원에서의 고효율 훈련

**C) 응용 확장 방향**

1. **다중 조건 통합 모델**
   - 확장: 단일 조건(y) → 다중 조건( $\(y_1, y_2, ..., y_n\)$ )
   - 예: 텍스트 + 스타일 + 레이아웃 동시 조건화
   - 기술적 도전: 다중 SDE 간 상관성 모델링

2. **확률적 불확실성 정량화**
   - 현재: 점 추정(deterministic reconstruction)
   - 제안: **후방분포 샘플 생성**으로 의료/과학 응용 확대
   - 예: Bayesian 역문제 해결, 확률적 예측

3. **3D 및 시계열 확장**
   - 2D 이미지 → 3D 입체 재구성
   - 정적 이미지 → 동영상/시계열 생성
   - 도전: 높은 차원성에서의 CMDE 효율성

**D) 아키텍처 혁신**

1. **Vision Transformer 기반 확산 모델**
   - 현재: CNN U-Net 기반
   - 향후: ViT 백본으로 **자기주의(self-attention) 활용** 증대
   - 기대: 장거리 의존성 포착으로 조건 활용 효율 향상[21]

2. **디스틸레이션 및 가속화**
   - 과제: 역공간 샘플링의 느린 속도
   - 방법: 일관성 모델, 적응형 가중치 등으로 **추론 단계 대폭 감소**[13]

3. **신경망 아키텍처 최적화**
   - 자동화: 자동 아키텍처 탐색(NAS)으로 **작업별 최적 설계** 발견
   - 효율성: 프루닝, 양자화로 모바일/엣지 배포 가능

**E) 평가 및 검증 개선**

1. **더 포괄적인 평가 지표**
   - 추가 지표: SSIM 외 구조적 복잡도 측정
   - 의료: Dice, IoU 등 분할 특화 메트릭
   - 안내: 생성 결과의 신뢰도 점수 추가

2. **인간 평가 기준화**
   - 현재: LPIPS, FID 등 자동 지표 중심
   - 개선: 대규모 인간 평가 데이터셋 구축으로 **지각적 품질의 객관화**

3. **도메인별 벤치마크 확립**
   - 의료: 질병 분류 정확도와의 상관성 분석
   - 미술: 미학적 품질 평가 기준 수립

**F) 윤리 및 공정성**

1. **편향 감지 및 완화**
   - 조건부 생성에서의 인종/성별 편향 분석
   - 공정성: 모든 인구 통계 그룹에 동등한 품질 보장

2. **해석 가능성 강화**
   - 현재: "블랙박스" 점수 함수
   - 제안: 어떤 조건 특성이 생성에 영향을 미치는지 **정성적 해석 개발**

***

### 9. 결론

**"Conditional Image Generation with Score-Based Diffusion Models"**는 조건부 확산 모델의 **이론적 기초를 정립**하고 **다중속도 확산이라는 혁신적 개념**을 도입하여, 역문제 해결의 실용성을 크게 높인 영향력 있는 논문입니다. 

이후 3년 이상의 후속 연구를 통해 의료영상, 텍스트-이미지 생성, 시계열 보정, 3D 재구성 등으로 광범위하게 확대되었으며, 특히 **일반화 성능 향상**에 대한 최근 이론적 분석들이 논문의 개념을 더욱 심화시키고 있습니다.[11][10][12]

앞으로의 연구는 (1) **최적 확산속도 자동 결정**, (2) **멀티 도메인 일반화**, (3) **확률적 불확실성 정량화**, (4) **샘플링 효율성 획기적 개선**, (5) **3D 및 시간적 확장**에 집중할 것으로 예상되며, 이들은 모두 본 논문의 CMDE 프레임워크를 기반으로 발전할 것입니다.

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/27818dd1-89ba-44a8-8357-661952418e4c/2111.13606v1.pdf)
[2](https://www.semanticscholar.org/paper/35356feaaf1a739a7db2b76f32e3e5a71ec74eb5)
[3](https://www.semanticscholar.org/paper/8982bb695dcebdacbfd079c62cd7acca8a8b48dc)
[4](https://arxiv.org/abs/2207.03430)
[5](https://arxiv.org/abs/2208.11970)
[6](https://ieeexplore.ieee.org/document/9878449/)
[7](https://arxiv.org/abs/2207.00050)
[8](https://www.semanticscholar.org/paper/1a65219f0d3852b55d1fadf58e1ca75c1090805e)
[9](https://arxiv.org/pdf/2211.01324.pdf)
[10](https://arxiv.org/html/2411.19339v2)
[11](https://arxiv.org/pdf/2311.01797.pdf)
[12](http://arxiv.org/pdf/2503.06698.pdf)
[13](https://arxiv.org/abs/2310.02279)
[14](http://arxiv.org/pdf/2302.06504.pdf)
[15](https://proceedings.neurips.cc/paper_files/paper/2024/file/b736c4b0b38876c9249db9bd900c1a86-Paper-Conference.pdf)
[16](https://www.sciencedirect.com/science/article/abs/pii/S0045782524006807)
[17](https://www.ijcai.org/proceedings/2024/0122.pdf)
[18](https://arxiv.org/html/2512.01292v1)
[19](https://pmc.ncbi.nlm.nih.gov/articles/PMC12573213/)
[20](https://www.sciencedirect.com/science/article/abs/pii/S1361841523001068)
[21](https://ieeexplore.ieee.org/document/10203178/)
[22](https://arxiv.org/html/2411.03177)
[23](https://arxiv.org/html/2405.20324)
[24](https://milvus.io/ai-quick-reference/how-does-denoising-score-matching-fit-into-diffusion-modeling)
[25](https://arxiv.org/abs/2203.17004)
[26](https://ieeexplore.ieee.org/document/10052908/)
[27](https://www.semanticscholar.org/paper/93d00ea9c87268f867b4addb8043be35d6996d18)
[28](https://arxiv.org/abs/2111.13606)
[29](https://arxiv.org/html/2312.12649v1)
[30](https://arxiv.org/html/2410.11439v1)
[31](https://arxiv.org/html/2308.16534)
[32](https://arxiv.org/html/2310.00224)
[33](https://openreview.net/pdf?id=PvvQlhBbgu)
[34](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/cmde/)
[35](https://academic.oup.com/nsr/article/11/12/nwae348/7810289)
[36](http://stanford.edu/class/ee367/Winter2025/report/report_Yue_Qi.pdf)
[37](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
[38](https://www.semanticscholar.org/paper/Conditional-Image-Generation-with-Score-Based-Batzolis-Stanczuk/35356feaaf1a739a7db2b76f32e3e5a71ec74eb5)
[39](https://arxiv.org/abs/2411.01663)
[40](https://www.semanticscholar.org/paper/696419f0fef87e5dc871013dfee93525cf7ddc80)
[41](https://aca.pensoft.net/article/151406/)
[42](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[43](https://www.jidc.org/index.php/journal/article/view/19790)
[44](https://onlinelibrary.wiley.com/doi/10.1155/tbed/7480710)
[45](http://biorxiv.org/lookup/doi/10.1101/2025.04.13.648103)
[46](https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs)
[47](https://biss.pensoft.net/article/136839/)
[48](http://arxiv.org/pdf/2408.06701.pdf)
[49](https://arxiv.org/html/2406.11713v1)
[50](https://arxiv.org/pdf/2412.09656.pdf)
[51](https://arxiv.org/pdf/2303.07576.pdf)
[52](http://arxiv.org/pdf/2405.15020.pdf)
[53](https://www.semanticscholar.org/paper/4f1dcc4fda12072a27c2f2af965b962acb63d1a6)
[54](https://dl.acm.org/doi/10.1145/3587423.3595503)
[55](https://jamanetwork.com/journals/jamasurgery/fullarticle/2811920)
[56](https://www.semanticscholar.org/paper/e4d7c5d317bad5c0654356b82219c7aa2897da88)
[57](https://direct.mit.edu/octo/article/doi/10.1162/octo_a_00525/124497/Generative-and-Adversarial-Art-and-the-Prospects)
[58](https://onlinelibrary.wiley.com/doi/10.1097/PG9.0000000000000387)
[59](https://www.itm-conferences.org/10.1051/itmconf/20257302037)
[60](https://arxiv.org/pdf/2212.07839.pdf)
[61](https://arxiv.org/html/2410.20898v1)
[62](https://arxiv.org/abs/2302.11710)
[63](https://arxiv.org/pdf/2211.12112.pdf)
[64](https://arxiv.org/pdf/2301.13188.pdf)
[65](http://arxiv.org/pdf/2211.15388.pdf)
[66](http://arxiv.org/pdf/2305.03509.pdf)
[67](https://www.edge-ai-vision.com/2023/01/from-dall%C2%B7e-to-stable-diffusion-how-do-text-to-image-generation-models-work/)
[68](https://arxiv.org/html/2303.07909v3)
[69](https://arxiv.org/html/2404.09016v1)
[70](https://en.wikipedia.org/wiki/Stable_Diffusion)
[71](https://mhsung.github.io/kaist-cs492d-fall-2024/)
[72](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DesignDiffusion_High-Quality_Text-to-Design_Image_Generation_with_Diffusion_Models_CVPR_2025_paper.pdf)
