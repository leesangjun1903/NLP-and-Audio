
# All are Worth Words: A ViT Backbone for Diffusion Models
## 1. 핵심 주장 및 주요 기여
### 1.1 핵심 주장
"All are Worth Words: A ViT Backbone for Diffusion Models"는 Vision Transformer (ViT)가 이미지 생성용 확산 모델(Diffusion Models)의 주요 백본 아키텍처로 사용될 수 있음을 입증한 논문이다. 저자들은 세 가지 핵심 주장을 제시한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

첫째, Vision Transformer는 CNN 기반 U-Net 대신 확산 모델에 효과적으로 사용될 수 있다. 둘째, 장거리 스킵 연결(long skip connections)이 확산 기반 이미지 모델링에 필수적이다. 셋째, CNN 기반 U-Net의 특징인 다운샘플링 및 업샘플링 연산자는 확산 모델에서 반드시 필요하지 않다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

### 1.2 주요 기여
이 논문의 주요 기여는 **U-ViT(U-shaped Vision Transformer)** 아키텍처의 제안이다. U-ViT는 다음과 같은 특징을 가진다:

- **통일된 토큰 처리**: 시간(time), 조건(condition), 노이즈 이미지 패치(noisy image patches)를 모두 동등하게 토큰으로 처리
- **장거리 스킵 연결**: 얕은 층(shallow layers)과 깊은 층(deep layers) 사이에 (# Blocks - 1)/2개의 스킵 연결 도입
- **선택적 합성곱 블록**: 출력 전 시각 품질 개선을 위한 3×3 합성곱 블록 추가

이러한 설계를 통해 U-ViT는 기존 CNN 기반 U-Net과 비교하여 경쟁력 있거나 우수한 성능을 달성했다. 특히, 클래스 조건부 이미지 생성(ImageNet 256×256)에서 **FID 2.29**의 기록을 세웠고, 텍스트-이미지 생성(MS-COCO)에서 **FID 5.48**을 달성했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

***

## 2. 해결하고자 하는 문제 및 제안 방법
### 2.1 문제 정의
확산 모델의 발전에도 불구하고, 이미지 생성 분야에서는 여전히 CNN 기반 U-Net이 지배적이다. 이는 다음과 같은 문제를 야기한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

1. **아키텍처 혁신의 정체**: Vision Transformer가 분류(classification), 객체 탐지(object detection) 등 다양한 시각 작업에서 우수한 성능을 보이고 있음에도, 확산 모델에서는 활용되지 못하고 있었다.

2. **구조적 필연성의 의문**: CNN의 다운샘플링/업샘플링이 확산 모델에 필수적인지에 대한 의문. 이러한 연산의 실제 필요성이 검증되지 않았다.

3. **일반화 능력의 한계**: CNN의 지역적 귀납 편향(local inductive bias)이 픽셀 레벨 예측 작업에 최적인지 불명확하다.

### 2.2 제안 방법론
#### 2.2.1 U-ViT 아키텍처 설계

U-ViT는 다음과 같이 설계된다:

**입력 처리:**
이미지 $x_t$를 패치로 분할하고, 시간 $t$와 조건 $c$를 임베딩한 후 모두를 토큰으로 변환한다.

$$\text{tokens} = \{\text{embed}(t), \text{embed}(c), \text{patch embed}(x_t)\}$$

**장거리 스킵 연결:**
$n$개의 Transformer 블록으로 구성된 네트워크에서, 다음과 같이 스킵 연결을 적용한다:

$$h_i^{\text{combine}} = \text{Linear}(\text{Concat}(h_i^{\text{main}}, h_j^{\text{skip}}))$$

여기서 $i$는 현재 레이어, $j$는 스킵된 초기 레이어이고, 

```math
(\text{\# Blocks} - 1) / 2
```

개의 이러한 연결이 네트워크 전체에 분포한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

**시간 조건화:**
Transformer 블록 내에서 시간 정보는 토큰으로 처리되며, 다음 블록으로 전달된다. 이는 Adaptive Layer Normalization (AdaLN)보다 우수한 성능을 보인다:

$$h_{\text{new}} = \text{TransformerBlock}(h, \text{time token})$$

**출력 생성:**
최종 토큰을 선형 투영(linear projection)으로 원래 이미지 공간으로 변환한 후, 선택적으로 3×3 합성곱을 적용한다:

$$\epsilon_\theta(x_t, t, c) = \text{Conv3×3}(\text{Rearrange}(\text{Linear}(h_L)))$$

#### 2.2.2 손실 함수

확산 모델의 표준적인 노이즈 예측 손실을 사용한다:

$$\mathcal{L} = \mathbb{E}_{t \sim \mathcal{U}[1,T], x_0, c, \epsilon} ||\epsilon - \epsilon_\theta(x_t, t, c)||_2^2$$

여기서 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$는 노이즈가 추가된 잠재 표현이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

### 2.3 모델 구조
**주요 구성 요소:**

| 구성 요소 | 설명 | 역할 |
|---------|------|------|
| **Patch Embedding** | 선형 투영 기반 패치 토큰화 | 이미지를 시퀀스로 변환 |
| **Position Embedding** | 1D 학습 가능한 위치 임베딩 | 공간 정보 인코딩 |
| **Time Embedding** | 정현파 + MLP 변환 | 확산 단계 정보 제공 |
| **Transformer Blocks** | Self-attention + FFN | 주요 계산 엔진 |
| **Long Skip Connections** | Concat + Linear | 저수준 특징 보존 |
| **Output Projection** | Linear + Conv3×3 | 노이즈 맵 생성 |

**U-ViT 구성 사양:**

| 모델 | 레이어 수 | 숨김 크기(D) | MLP 크기 | 주의 헤드 | 매개변수 |
|-----|---------|-----------|---------|---------|---------|
| U-ViT-Small | 13 | 512 | 2048 | 8 | 44M |
| U-ViT-Small (Deep) | 17 | 512 | 2048 | 8 | 58M |
| U-ViT-Mid | 17 | 768 | 3072 | 12 | 131M |
| U-ViT-Large | 21 | 1024 | 4096 | 16 | 287M |
| U-ViT-Huge | 29 | 1152 | 4608 | 16 | 501M |

***

## 3. 성능 향상 및 실험 결과
### 3.1 벤치마크 성능
#### 무조건부 및 클래스 조건부 이미지 생성

U-ViT는 다양한 데이터셋에서 경쟁력 있는 성능을 달성했다:

**CIFAR10:**
- U-ViT-S/2: FID 3.11 (44M 매개변수)
- DDPM++ cont.: FID 2.55 (62M 매개변수)
- EDM: FID 1.97 (56M 매개변수)

**CelebA 64×64:**
- U-ViT-S/4: FID 2.87 (44M 매개변수)
- Soft Truncation: FID 1.90 (62M 매개변수)

**ImageNet 64×64:**
- U-ViT-L/4: FID 4.26 (287M 매개변수)
- ADM: FID 2.07 (296M 매개변수)
- EDM: FID 1.36 (296M 매개변수)

**ImageNet 256×256 (잠재 확산 모델):**
- U-ViT-H/2: **FID 2.29** (501M + 84M AE) ← **기록적 성능** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)
- LDM: FID 3.60 (400M + 55M AE)
- ADM-G, ADM-U: FID 3.94

**ImageNet 512×512:**
- U-ViT-H/4: FID 4.05 (501M + 84M AE)
- ADM-G, ADM-U: FID 3.85

#### 텍스트-이미지 생성 (MS-COCO)

- U-ViT-S/2 (Deep): **FID 5.48** (58M + 123M TE + 84M AE) ← **기록적 성능** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)
- U-ViT-S/2: FID 5.95 (45M + 123M TE + 84M AE)
- U-Net*: FID 7.32 (53M + 123M TE + 84M AE)



### 3.2 일반화 성능 향상 메커니즘
#### 3.2.1 패치 크기의 효과

U-ViT의 일반화 성능에서 패치 크기는 결정적인 역할을 한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

$$\text{FID} = f(\text{patch size}, \text{resolution})$$

실험 결과에 따르면:

- **패치 크기 8**: FID ≈ 43 (CIFAR10)
- **패치 크기 4**: FID ≈ 31
- **패치 크기 2**: FID ≈ 7.1 (최적)
- **패치 크기 1**: 추가 개선 없음

**이유**: 확산 모델의 노이즈 예측 목표는 저수준(pixel-level) 예측 작업이므로 작은 패치 크기가 필요하다. 이는 분류 같은 고수준 작업과 구별된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

#### 3.2.2 깊이(Depth) 효과

네트워크 깊이의 증가는 초기에 성능을 개선하지만 수렴점이 있다:

- **깊이 9**: FID ≈ 13
- **깊이 13**: FID ≈ 7.1 (최적, 50K 반복)
- **깊이 17**: 50K 반복 내에서 추가 개선 없음

#### 3.2.3 너비(Width) 효과

숨김 차원의 증가는 비슷한 수렴 특성을 보인다:

- **D=256**: FID ≈ 11
- **D=512**: FID ≈ 7.1 (선택됨)
- **D=768**: 추가 개선 없음

#### 3.2.4 다양한 해상도에서의 우수성

특히 잠재 확산 모델(Latent Diffusion Model, LDM) 설정에서 U-ViT의 우수성이 두드러진다. 이는 다음 이유 때문이다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

1. **Latent space의 의미론적 구조**: VAE의 잠재 공간은 의미론적으로 구조화되어 있으며, ViT의 전역 주의(global attention)가 이를 효과적으로 모델링한다.

2. **토큰 기반 처리의 일관성**: 모든 입력을 토큰으로 처리함으로써, 조건과 이미지 특징 간의 상호작용이 모든 계층에서 일어난다. 반면 U-Net은 특정 계층(cross-attention)에서만 상호작용한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

3. **스킵 연결의 효과성**: 장거리 스킵 연결이 저수준 특징을 보존하면서도 ViT의 전역 모델링 능력을 활용한다.

### 3.3 수렴 속도 비교
Table 3는 다양한 샘플링 단계에서 U-ViT과 LDM의 성능을 비교한다:

| 샘플링 단계 | LDM (178K step) | U-ViT-H/2 (200K) | U-ViT-H/2 (500K) |
|-----------|-----------------|-----------------|-----------------|
| 4 | 34.48 | 16.48 | 15.44 |
| 5 | 12.73 | 4.94 | 4.64 |
| 10 | 4.51 | 3.87 | 3.18 |
| 15 | 3.87 | 3.54 | 2.92 |
| 20 | 3.68 | 2.91 | 2.53 |

U-ViT는 빠른 수렴으로 인해, 동일 학습 단계에서 더 나은 성능을 보인다.

***

## 4. 한계 및 제약사항
### 4.1 계산 복잡도
작은 패치 크기 (2×2)를 사용하는 것이 성능에 필수적이지만, 이는 계산 비용을 증가시킨다:

$$\text{토큰 수} = \frac{H \times W}{P^2}$$

패치 크기가 2일 때, 256×256 이미지는 16,384개의 토큰이 생성되어 상당한 메모리 요구량과 계산 시간을 야기한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

### 4.2 고해상도에서의 필수 조건
고해상도 이미지 (256×256, 512×512) 생성 시 잠재 확산 모델(LDM) 설정이 필수적이다. 픽셀 공간에서의 직접 확산은 불가능하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

### 4.3 추론 시간
- U-ViT-S: ~19초 (500개 샘플, A100 GPU)
- U-ViT-M: ~34초
- U-ViT-L: ~59초
- U-ViT-H: ~89초

Classifier-free guidance 사용 시 시간이 약 2배 증가한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

### 4.4 수렴 특성
패치 크기와 모델 크기의 상호작용으로 인해, 50K 반복 내에서 깊이 17 또는 너비 768의 추가 증가가 개선을 가져오지 못할 수 있다. 더 긴 학습이 필요할 수 있음을 시사한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

***

## 5. 모델의 일반화 성능 향상 가능성
### 5.1 토큰 기반 아키텍처의 이점
U-ViT의 가장 큰 강점은 **모든 입력을 토큰으로 통일 처리**하는 것이다. 이는 다음과 같은 일반화 이점을 제공한다:

1. **조건-이미지 상호작용의 균등성**: 텍스트, 클래스, 또는 다른 조건과 이미지 패치 간의 상호작용이 모든 Transformer 블록에서 발생한다. 이는 조건 정보가 각 계층의 특징 형성에 더 깊숙이 영향을 미친다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

2. **크로스 모달리티 확장성**: 동일한 토큰 처리 메커니즘으로 여러 모달리티(텍스트, 오디오, 구조)를 처리할 수 있어, 대규모 교차 모달리티 데이터셋에서의 확장에 유리하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

3. **일반화된 위치 정보**: 1D 학습 가능한 위치 임베딩은 2D 이미지 특화 위치 편향을 제거하여, 다양한 해상도와 패치 구성에 더 잘 적응할 수 있다.

### 5.2 장거리 의존성 모델링
전역 주의(global attention)는 픽셀 간의 장거리 의존성을 모델링한다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

이는 U-Net의 지역적 합성곱과 달리, 이미지의 어느 부분에서든 정보가 전파될 수 있음을 의미한다. 따라서 구조가 불규칙하거나 장거리 관계가 중요한 도메인에서 더 우수한 일반화를 기대할 수 있다.

### 5.3 장거리 스킵 연결의 역할
$$h_{\text{deep}} = h_{\text{intermediate}} + h_{\text{shallow}}$$

장거리 스킵 연결은 저수준 특징(edges, textures)이 깊은 계층까지 직접 전달되도록 보장한다. 확산 모델의 목표가 픽셀 레벨 예측이므로, 이는 다양한 데이터셋과 도메인에서 중요한 특징을 보존한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

### 5.4 실증적 증거
- **CIFAR10 → ImageNet 전이**: CIFAR10에서 학습한 설계 선택이 ImageNet에서도 일관되게 우수 (Figure 2-5 참고)
- **다해상도 강건성**: 64×64에서 512×512까지 동일 아키텍처로 경쟁력 있는 성능
- **텍스트-이미지 생성 우수성**: MS-COCO에서 U-Net보다 더 나은 텍스트 의미론 포착 (Figure 6) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

***

## 6. 2020년 이후 관련 최신 연구 비교 분석
### 6.1 Transformer 기반 확산 모델의 진화
#### 6.1.1 **DiT (Diffusion Transformers, 2022년)**

**저자**: William Peebles, Saining Xie (OpenAI/Meta)

**주요 특징**:
- Pure isotropic Transformer 구조 (다운샘플링 없음)
- Adaptive Layer Normalization (AdaLN-Zero) 기반 조건화
- 스케일 법칙 분석: GFLOPs와 FID의 명확한 관계 [arxiv](https://arxiv.org/abs/2212.09748)

**성능**:
- ImageNet 256×256: FID 2.27 (U-ViT와 유사)
- ImageNet 512×512: SOTA

**U-ViT와의 차이**:
- DiT: 스킵 연결 없음, AdaLN 사용
- U-ViT: 장거리 스킵 연결, 시간 토큰 방식 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

실험에서 U-ViT의 시간 토큰 처리가 AdaLN보다 우수함을 보였다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

#### 6.1.2 **DiffiT (2023년)**

**저자**: Ali Hatamizadeh, Jiaming Song, Guilin Liu (NVIDIA)

**주요 혁신**:
- Time-dependent Multihead Self Attention (TMSA) 도입
- 미세한 노이징 과정 제어
- 매개변수 효율성 향상 (DiT보다 16-20% 적은 매개변수) [arxiv](https://arxiv.org/abs/2312.02139)

**성능**:
- ImageNet 256×256: **FID 1.73** (U-ViT 2.29 대비 우수)
- 매개변수 효율성: DiT-XL/2 대비 19.85% 감소

**기여**:
- 시간 의존성 주의 메커니즘의 효과성 입증
- Transformer 블록 내 세밀한 시간 조건화

#### 6.1.3 **MDT (Masked Diffusion Transformer, 2023년)**

**주요 특징**:
- 이산 확산 위에 masked autoregressive 모델링
- VQ-GAN 기반 이산 토큰화
- 더 빠른 샘플링 가능

#### 6.1.4 **LaVin-DiT (2025년)**

**주요 특징**:
- 스케일: 0.1B에서 3.4B 매개변수까지
- 다양한 시각 작업에 대한 기초 모델 역할
- 전이 학습 능력 강화

**의의**:
- 대규모 Transformer 기반 확산 모델의 실용성 입증

### 6.2 최신 아키텍처 개선 (2024-2025)
#### **Skip-DiT (2024년)**

**핵심 기여**:
- DiT에 장거리 스킵 연결 추가
- **4.4배 학습 가속**, **1.5-2배 추론 가속**
- U-ViT의 스킵 연결 아이디어를 DiT에 적용 [semanticscholar](https://www.semanticscholar.org/paper/60dd6e831a6ce11f1fa69f859171a4ec65bf0420)

**의의**:
- U-ViT의 스킵 연결 설계가 널리 인정됨
- 최신 DiT 연구에서도 스킵 연결의 필수성 재확인

#### **FiTv2 (2024년)**

**특징**:
- 해상도에 자유로운(resolution-free) Flexible Vision Transformer
- Rotary Positional Embeddings (RoPE) 사용
- 다양한 입력 해상도에서 우수한 일반화

#### **LightningDiT (2025년)**

**혁신**:
- Vision 기초 모델 정렬 VAE (VA-VAE) 사용
- 고차원 잠재 공간에서 효율적 학습
- **ImageNet 256×256: FID 1.35** (최첨단)

**기술**: 의미론적 구조가 풍부한 잠재 공간 활용

#### **SVG Diffusion (2025년)**

**특징**:
- VAE 없이 DINO 특징 사용
- 의미론적 분산(semantic dispersion) 강조
- 빠른 수렴, 효율적 샘플링

**관련성**:
- U-ViT의 일반화 아이디어 확장
- Token 기반 처리의 중요성 재확인

### 6.3 성능 향상 궤적
| 연도 | 모델 | ImageNet 256×256 FID | 주요 혁신 |
|-----|------|-------------------|----------|
| 2021 | LDM | 3.60 | 잠재 공간 확산 |
| 2022 | U-ViT | 2.29 | 장거리 스킵 + 토큰화 |
| 2022 | DiT | 2.27 | Pure Transformer 스케일링 |
| 2023 | DiffiT | 1.73 | 시간 의존성 주의 |
| 2024 | Skip-DiT | ~2.1 | 스킵 연결 안정성 |
| 2024 | FiTv2 | ~1.8-2.0 | 해상도 자유성 |
| 2025 | LightningDiT | 1.35 | 의미론적 잠재 공간 |

### 6.4 연구 방향의 진화
1. **아키텍처에서 최적화로**: U-ViT/DiT 이후 연구는 기본 Transformer 아키텍처를 고정하고, 잠재 공간, 노이징 스케줄, 스케일링 법칙 최적화에 집중

2. **스킵 연결의 재평가**: U-ViT의 장거리 스킵 연결 설계가 최신 연구 (Skip-DiT)에서 다시 주목받음

3. **기초 모델로의 확장**: LaVin-DiT 등 대규모 스케일링을 통한 멀티태스크 학습

4. **의미론적 구조 활용**: SVG, VA-VAE 등 고도의 의미론적 잠재 공간 활용

***

## 7. 앞으로의 연구에 미치는 영향 및 연구 시 고려사항
### 7.1 학술적 영향
#### **패러다임 전환 촉발**

U-ViT는 확산 모델 백본 아키텍처 연구에 **근본적인 전환**을 가져왔다. CNN 기반 U-Net이 필연적이 아니라는 것을 보임으로써, 이후 연구자들이 다양한 Transformer 기반 설계를 자유롭게 탐색하도록 했다. 특히 DiT, DiffiT, Skip-DiT 등 이후의 주요 성과들은 모두 U-ViT가 제시한 아이디어에서 영감을 받았다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7db6d10d-7e8d-4cd9-93f3-9c1519807425/2209.12152v4.pdf)

#### **긴 연결의 재발견**

U-ViT는 오래된 설계 원리인 "장거리 스킵 연결"을 확산 모델 맥락에서 재조명했다. 이는 2024년 Skip-DiT에서 "스킵 연결이 Transformer 기반 확산 모델에 필수적"이라고 재확인되었고, 현재 표준 설계가 되었다. [semanticscholar](https://www.semanticscholar.org/paper/60dd6e831a6ce11f1fa69f859171a4ec65bf0420)

#### **토큰화의 통일화**

"모든 것을 토큰으로 처리한다"는 설계 철학은 이후 여러 멀티모달 생성 모델로 확장되었다. 이는 텍스트, 이미지, 비디오를 동등하게 취급하는 현재의 기초 모델 아키텍처의 토대가 되었다.

### 7.2 실용적 적용 방향
#### **도메인 특화 확산 모델**

U-ViT의 아키텍처는 다음 분야에서 실질적 기여를 할 수 있다:

1. **의료 이미징**: 저수준 특징이 중요한 의료 이미지 생성에 스킵 연결과 세밀한 토큰화가 특히 유용
2. **비디오 생성**: 장거리 시간적 의존성 모델링에 전역 주의가 필수적
3. **3D 생성**: 공간적 구조 보존에 스킵 연결의 효과성
4. **초해상도**: 저수준 특징 보존의 중요성

#### **효율성 개선**

최신 연구 (Skip-DiT, LightningDiT)는 U-ViT 기초 위에서:
- 4배 이상의 학습 가속
- 실시간 생성 가능성
- 엣지 디바이스 배포 가능성

을 달성하고 있다.

### 7.3 연구 시 고려사항
#### **7.3.1 패치 크기 최적화**

U-ViT 연구에서 확인된 바와 같이, 패치 크기는 매우 중요한 하이퍼파라미터이다:

$$\text{작은 패치} \Rightarrow \text{더 많은 토큰} \Rightarrow \text{더 높은 FID (성능)} \Rightarrow \text{더 큰 계산 비용}$$

**권장사항**:
- 저수준 작업 (노이즈 예측): 패치 크기 2-4
- 고수준 작업: 패치 크기 8-16
- 고해상도 이미지: 잠재 공간 사용 필수

#### **7.3.2 스킵 연결 설계**

합성(concatenation) + 선형 투영이 최적임이 검증되었다:

$$h_{\text{combined}} = \text{Linear}(\text{Concat}(h_{\text{main}}, h_{\text{skip}}))$$

**이유**: 단순 덧셈은 이미 Transformer 블록 내 skip connection이 있어 효과 미미

#### **7.3.3 시간 조건화 전략**

- **시간 토큰 (선택됨)**: 모든 계층에서 시간 정보 활용
- **AdaLN**: 일부 계층에서만 효과적

현재 표준은 혼합 방식 (예: AdaLN-Zero in DiT)

#### **7.3.4 잠재 공간의 중요성**

고해상도에서는 잠재 확산이 필수이며, **잠재 공간의 의미론적 구조**가 핵심이다:

- **VA-VAE (LightningDiT)**: 의미론적 구조 강화
- **SVG (2025)**: DINO 특징의 의미론적 분산 활용

**제안**: 자신의 도메인에 최적화된 VAE 또는 토큰라이저 개발

#### **7.3.5 일반화를 위한 전략**

U-ViT의 성공에 기반하여:

1. **데이터 증강**: 토큰 기반 처리로 인한 추상화 수준이 높으므로, 다양한 증강이 효과적
2. **전이 학습**: 큰 데이터셋으로 사전학습된 후, 소규모 도메인에 미세조정
3. **멀티태스크 학습**: 같은 아키텍처로 여러 생성 작업 동시 학습 (LaVin-DiT 스타일)
4. **기초 모델 활용**: 대규모 모델 (3B+ 매개변수)에서의 강한 전이 능력

#### **7.3.6 계산 효율성**

- **Gradient checkpointing**: 메모리 53GB → 10GB (A100에서 U-ViT-L/2)
- **Mixed precision training**: 실무 필수
- **DPM-Solver** 같은 고속 샘플러: 추론 가속

#### **7.3.7 평가 지표**

- **FID**: 전체 분포 품질
- **sFID**: 공간 세부사항
- **Inception Score**: 다양성
- **Precision/Recall**: 정밀도-재현율 트레이드오프
- **CLIPScore**: 텍스트-이미지 정렬 (조건 생성 시)

### 7.4 미해결 과제 및 미래 연구 방향
#### **해결되지 않은 문제**

1. **샘플 다양성 vs 정확도**: 높은 guidance scale에서 다양성 감소
2. **고해상도 효율성**: 512×512 이상에서 여전히 높은 계산 비용
3. **도메인 일반화**: 학습 데이터와 큰 차이가 있는 도메인에서의 성능 저하
4. **조건화 정보의 활용도**: 약한 조건 입력에서의 성능

#### **유망한 연구 방향**

1. **뉴럴 아키텍처 탐색 (NAS)**: 최적의 패치 크기, 스킵 연결 위치 자동 결정
2. **적응형 토큰화**: 이미지 복잡도에 따른 동적 패치 크기
3. **뮤체트 아키텍처**: 생성과 인식을 동시에 수행하는 통합 모델
4. **양자화 기술**: 16비트 이하의 저정밀도 확산 모델
5. **Flow Matching**: 확산 과정의 개선 대안

***

## 결론
"All are Worth Words: A ViT Backbone for Diffusion Models"는 확산 모델 연구에서 **패러다임적 변화**를 가져온 중요한 논문이다. U-ViT는 단순하면서도 효과적인 설계를 통해:

1. **CNN 기반 U-Net의 필연성 도전**: 다운샘플링/업샘플링이 필수가 아님을 입증
2. **Transformer의 적용 확대**: 확산 모델 분야에서 Transformer 기반 아키텍처 개척
3. **강력한 경험적 성과**: ImageNet 256×256에서 FID 2.29의 기록 달성

이 연구의 영향은 즉각적이고 광범위했다. DiT, DiffiT, Skip-DiT, LaVin-DiT, LightningDiT 등 이후의 주요 성과들이 모두 U-ViT의 기초 위에서 이루어졌고, 2024-2025년 현재 **FID 1.35** 수준의 성능도 U-ViT의 통찰 위에 구축되어 있다.

특히 **일반화 성능 향상** 측면에서 U-ViT의 기여는:
- 전역 주의를 통한 장거리 의존성 모델링
- 장거리 스킵 연결을 통한 저수준 특징 보존
- 통일된 토큰 처리를 통한 모달리티 자유도 증가

로 요약되며, 이는 현대의 대규모 멀티모달 생성 모델로의 진화에 근본적인 역할을 했다.

향후 연구자들은 U-ViT의 핵심 인사이트 (스킵 연결, 토큰화, 스케일링 법칙)를 바탕으로, 자신의 도메인에 특화된 아키텍처를 설계할 수 있을 것이다.

***

## 참고 문헌

<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_5][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_6][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_7][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_8][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2209.12152v4.pdf

[^1_2]: https://arxiv.org/abs/2212.09748

[^1_3]: https://arxiv.org/abs/2312.02139

[^1_4]: https://www.semanticscholar.org/paper/60dd6e831a6ce11f1fa69f859171a4ec65bf0420

[^1_5]: https://link.springer.com/10.1007/s00521-023-09021-x

[^1_6]: https://www.nature.com/articles/s41598-025-31623-9

[^1_7]: https://arxiv.org/abs/2503.19757

[^1_8]: https://ieeexplore.ieee.org/document/11094598/

[^1_9]: https://ieeexplore.ieee.org/document/10966907/

[^1_10]: https://ieeexplore.ieee.org/document/11044676/

[^1_11]: https://link.springer.com/10.1007/s10586-025-05618-0

[^1_12]: https://www.ewadirect.com/proceedings/ace/article/view/20804

[^1_13]: https://arxiv.org/abs/2511.06282

[^1_14]: https://link.springer.com/10.1007/s00371-025-04106-1

[^1_15]: https://arxiv.org/html/2411.11505

[^1_16]: http://arxiv.org/pdf/2212.09748v2.pdf

[^1_17]: http://arxiv.org/pdf/2405.14854.pdf

[^1_18]: https://arxiv.org/pdf/2209.12152.pdf

[^1_19]: https://arxiv.org/abs/2410.13925v1

[^1_20]: http://arxiv.org/pdf/2312.04557.pdf

[^1_21]: https://arxiv.org/html/2406.17173v2

[^1_22]: https://arxiv.org/html/2312.02139v3

[^1_23]: https://www.emergentmind.com/topics/diffusion-transformer-dit-architecture

[^1_24]: https://arxiv.org/abs/2503.06698v1

[^1_25]: https://openreview.net/pdf/7415fc810d502243f9eae150b2aa40ebbb6faa5e.pdf

[^1_26]: https://www.lightly.ai/blog/diffusion-transformers-dit

[^1_27]: https://ommer-lab.com/research/latent-diffusion-models/

[^1_28]: https://arxiv.org/pdf/2510.09586.pdf

[^1_29]: https://apxml.com/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/diffusion-transformers-dit

[^1_30]: https://icml.cc/virtual/2025/poster/44206

[^1_31]: https://en.wikipedia.org/wiki/Vision_transformer

[^1_32]: https://encord.com/blog/diffusion-models-with-transformers/

[^1_33]: https://liner.com/review/whats-in-latent-leveraging-diffusion-latent-space-for-domain-generalization

[^1_34]: https://cislab.hkust-gz.edu.cn/media/documents/_AAAI_2025__DiT4Edit_Camera_Ready.pdf

[^1_35]: https://arxiv.org/abs/2312.05387

[^1_36]: https://arxiv.org/html/2510.15301v1

[^1_37]: https://arxiv.org/html/2508.10711v1

[^1_38]: https://arxiv.org/abs/2411.17616

[^1_39]: https://arxiv.org/abs/2503.06698

[^1_40]: https://arxiv.org/html/2507.11540v1

[^1_41]: https://arxiv.org/html/2510.04797v1

[^1_42]: https://arxiv.org/abs/2504.16580

[^1_43]: https://arxiv.org/html/2505.04769v1

[^1_44]: https://arxiv.org/html/2505.13219v1

[^1_45]: https://openaccess.thecvf.com/content/ICCV2025/papers/Thomas_Whats_in_a_Latent_Leveraging_Diffusion_Latent_Space_for_Domain_ICCV_2025_paper.pdf

[^1_46]: https://arxiv.org/html/2411.15397v1

[^1_47]: https://arxiv.org/html/2410.03456v2

[^1_48]: https://arxiv.org/html/2506.00849v1

[^1_49]: https://www.youtube.com/watch?v=aSLDXdc2hkk

[^1_50]: https://leeyngdo.github.io/blog/generative-model/2024-07-01-diffusion-transformer/

[^1_51]: https://ieeexplore.ieee.org/document/11210878/

[^1_52]: https://ieeexplore.ieee.org/document/11085899/

[^1_53]: https://ieeexplore.ieee.org/document/10203178/

[^1_54]: https://www.mdpi.com/2227-7390/12/7/1028

[^1_55]: https://ieeexplore.ieee.org/document/10657395/

[^1_56]: https://ieeexplore.ieee.org/document/10667527/

[^1_57]: https://arxiv.org/abs/2209.12152

[^1_58]: https://ieeexplore.ieee.org/document/11220188/

[^1_59]: https://dl.acm.org/doi/10.1145/3746027.3755076

[^1_60]: https://arxiv.org/pdf/2310.13545.pdf

[^1_61]: https://arxiv.org/html/2411.17616

[^1_62]: http://arxiv.org/pdf/2503.18414.pdf

[^1_63]: https://arxiv.org/html/2312.11392v1

[^1_64]: https://arxiv.org/html/2402.15170v1

[^1_65]: https://arxiv.org/abs/2501.14524

[^1_66]: https://arxiv.org/abs/2212.13771

[^1_67]: https://www.emergentmind.com/papers/2209.12152

[^1_68]: https://apxml.com/courses/advanced-diffusion-architectures/chapter-2-advanced-unet-architectures/unet-time-embeddings

[^1_69]: https://openreview.net/pdf?id=DlRsoxjyPm

[^1_70]: https://aclanthology.org/2025.conll-1.26.pdf

[^1_71]: https://jeonghwarr.github.io/posts/Synthetic-Data-from-Diffusion-Models-Improves-ImageNet-Classification/

[^1_72]: https://apxml.com/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/dit-conditioning

[^1_73]: https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf

[^1_74]: https://openreview.net/pdf/6b72782694a08b0cde5dac666544509e5298c730.pdf

[^1_75]: https://eugeneyan.com/writing/text-to-image/

[^1_76]: https://github.com/openai/guided-diffusion/issues/14

[^1_77]: https://openaccess.thecvf.com/content/CVPR2023/papers/Bao_All_Are_Worth_Words_A_ViT_Backbone_for_Diffusion_Models_CVPR_2023_paper.pdf

[^1_78]: https://arxiv.org/abs/2211.04236

[^1_79]: https://arxiv.org/html/2410.19324v1

[^1_80]: https://arxiv.org/html/2510.01047v1

[^1_81]: https://arxiv.org/html/2503.18414v3

[^1_82]: https://arxiv.org/pdf/2311.04938.pdf

[^1_83]: https://arxiv.org/html/2506.10036v1

[^1_84]: https://arxiv.org/pdf/2411.10433.pdf

[^1_85]: https://arxiv.org/html/2509.06068v1

[^1_86]: https://arxiv.org/html/2504.12007v2

[^1_87]: https://arxiv.org/html/2410.01912v1

[^1_88]: https://arxiv.org/html/2505.18853v1

[^1_89]: https://arxiv.org/html/2404.10445v1
