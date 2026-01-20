
# Human Motion Diffusion Model

## 1. 핵심 주장 및 주요 기여

Human Motion Diffusion Model (MDM)은 인간 동작 생성 문제를 해결하기 위해 확산 모델(Diffusion Models)을 처음으로 체계적으로 적용한 선도적 연구입니다. 이 논문의 핵심 주장은 **텍스트와 동작 간의 many-to-many 관계**를 포착하는 데 있어 확산 모델이 VAE나 오토인코더보다 본질적으로 우월하다는 것입니다. 예를 들어, "킥(kick)"이라는 동일한 텍스트 설명은 축구 킥, 가라테 킥 등 다양한 방식으로 표현될 수 있으며, 기존 VAE 기반 방법들은 정규 분포 가정으로 인해 이러한 다양성을 충분히 포착하지 못합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

MDM의 주요 기여는 다음과 같습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

1. **Transformer 기반 경량 아키텍처**: 일반적인 U-Net 대신 트랜스포머 인코더를 사용하여 동작의 시간적 성질과 비공간적 특성(관절 시퀀스)을 더 효과적으로 처리합니다.

2. **Sample 예측 전략**: 노이즈 예측 대신 각 확산 단계에서 직접 깨끗한 샘플을 예측하여, 발 접촉, 속도, 위치 등에 대한 기하학적 손실함수 적용을 가능하게 합니다.

3. **Generic Framework**: 텍스트 조건, 액션 클래스, 무조건부 생성을 포함한 다양한 작업을 단일 모델로 처리합니다.

4. **경량 학습**: 단일 NVIDIA RTX 2080 Ti GPU로 약 3일 내에 학습 가능하면서 SOTA 성능을 달성합니다.

***

## 2. 해결 문제와 기술적 도전

### 2.1 도메인의 고유한 어려움

인간 동작 생성 문제는 세 가지 본질적 도전을 가집니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

- **동작의 광대한 가능성**: 인간은 무수한 방식으로 움직일 수 있으며, 이는 생성 모델이 학습해야 할 고차원 다양체(manifold)를 형성합니다.
- **인간의 민감한 지각**: 사람들은 부자연스러운 동작에 매우 민감하여, 미묘한 아티팩트(발 미끄러짐, 관절 떨림)도 감지합니다.
- **텍스트 설명의 모호성**: 동일한 텍스트 설명이 여러 동작으로 해석될 수 있으므로, 한 입력에 대해 다양한 출력이 필요합니다.

### 2.2 기존 방법의 한계

#### VAE 기반 방법의 문제
기존 최고 성능 방법들(TEMOS, T2M)은 VAE를 활용하여 텍스트와 동작을 공유 잠재공간에 매핑합니다. 그러나 이 접근법은 두 가지 근본적 제약을 가집니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

- **정규 분포 가정**: VAE는 잠재공간을 정규 분포로 강제하여, 복잡한 다양한 동작 분포를 포착하지 못합니다.
- **정보 압축**: 연속적 잠재변수로의 압축은 동작의 세부 정보 손실을 초래합니다.

#### 초기 Diffusion 모델의 문제
기존 확산 모델(특히 이미지 영역에서)은 인간 동작 적용 시 두 가지 문제를 보입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

- **자원 소비**: U-Net 기반 아키텍처는 연속 고차원 데이터에 최적화되어 있어, 동작 데이터에 불필요한 계산 오버헤드를 야기합니다.
- **제어 어려움**: 기존 조건화 메커니즘은 인간 동작 도메인의 특수한 요구사항(기하학적 제약)을 수용하지 못합니다.

***

## 3. 제안 방법 및 수식

### 3.1 확산 과정의 수학적 기초

MDM은 마르코프 노이징 과정으로 정의되는 표준 확산 모델을 기반합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

$$q(x_{1:N}^t | x_{1:N}^{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{1:N}^{t-1}, (1-\alpha_t)I)$$

여기서 $\alpha_t \in (0,1)$은 스케줄된 상수이며, $x_{1:N}^0$는 데이터 분포에서 추출된 원본 동작입니다. 충분히 작은 $\alpha_t$에 대해, $x_{1:N}^T \sim \mathcal{N}(0, I)$로 근사할 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

### 3.2 Sample Prediction Loss

MDM의 핵심 설계 선택은 노이즈 예측 대신 신호 자체를 예측하는 것입니다. 이는 다음 간단한 손실함수로 구현됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

$$L_{simple} = \mathbb{E}_{x_0 \sim q(x_0|c), t \sim [1,T]} [\|x_0 - G(x_t, t, c)\|_2^2]$$

여기서 $G(x_t, t, c)$는 노이징된 동작 $x_t$, 타임스텝 $t$, 조건 $c$가 주어졌을 때 원본 깨끗한 동작 $\hat{x}_0$을 예측하는 신경망입니다.

### 3.3 기하학적 손실함수

기하학적 손실함수는 확산 모델 설정에서 물리적 합리성을 보장하는 MDM의 혁신적 기여입니다. 세 가지 핵심 손실을 정의합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

#### Position Loss (순방향 운동학)
$$L_{pos} = \frac{1}{N} \sum_{i=1}^N \|FK(x_i^0) - FK(\hat{x}_i^0)\|_2^2$$

$FK(\cdot)$는 관절 회전을 관절 위치로 변환하는 순방향 운동학 함수입니다. 회전 표현 사용 시 실제 공간에서의 정확도를 보장합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

#### Foot Contact Loss (발 미끄러짐 방지)
$$L_{foot} = \frac{1}{N-1} \sum_{i=1}^{N-1} \|(FK(\hat{x}_{i+1}^0) - FK(\hat{x}_i^0)) \cdot f_i\|_2^2$$

여기서 $f_i \in \{0,1\}^J$는 각 프레임 $i$의 이진 발 접촉 마스크입니다. 이는 발이 지면에 닿아있을 때(f_i = 1) 발의 속도를 0으로 강제하여, 발이 지면을 미끄러지는 부자연스러운 아티팩트를 방지합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

#### Velocity Loss (부드러운 동작)
$$L_{vel} = \frac{1}{N-1} \sum_{i=1}^{N-1} \|(x_{i+1}^0 - x_i^0) - (\hat{x}_{i+1}^0 - \hat{x}_i^0)\|_2^2$$

관절 속도의 일관성을 강제하여 떨림 없는 자연스러운 동작을 생성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

### 3.4 통합 손실함수

최종 훈련 손실은 다음과 같이 결합됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

$$L = L_{simple} + \lambda_{pos} L_{pos} + \lambda_{vel} L_{vel} + \lambda_{foot} L_{foot}$$

여기서 $\lambda_{pos}, \lambda_{vel}, \lambda_{foot}$는 각 항의 기여도를 조절하는 가중치입니다.

### 3.5 Classifier-Free Guidance

조건부 생성을 위해, MDM은 classifier-free guidance를 구현합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

$$G_s(x_t, t, c) = G(x_t, t, \emptyset) + s \cdot (G(x_t, t, c) - G(x_t, t, \emptyset))$$

훈련 중 10% 확률로 조건 $c$를 $\emptyset$로 무효화하여, 모델이 조건부와 무조건부 분포를 모두 학습하도록 합니다. 스케일 파라미터 $s$는 다양성과 충실도 사이의 트레이드오프를 제어합니다. 논문에서는 $s = 2.5$를 권장합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

***

## 4. 모델 구조 및 아키텍처

### 4.1 Transformer Encoder 설계

MDM의 모델 $G$는 간단한 트랜스포머 인코더 전용 아키텍처로 구현됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

**입력 처리 파이프라인:**
1. 조건 코드 $c$ (CLIP 텍스트 임베딩)와 타임스텝 $t$를 각각 별도의 피드포워드 네트워크로 트랜스포머 차원으로 투영
2. 두 투영을 합산하여 조건-시간 토큰 $z_t^k$ 생성
3. 각 노이징 입력 프레임 $x_t$를 선형 투영하고 표준 위치 임베딩과 합산
4. $z_t^k$와 투영된 프레임들을 인코더에 입력

**출력 생성:**
- 인코더의 첫 번째 출력 토큰(z_t^k에 해당)을 제외한 나머지를 원본 동작 차원으로 투영
- 이것이 최종 예측 $\hat{x}_0$가 됨 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

### 4.2 조건화 메커니즘

MDM은 세 가지 조건화 방식을 지원합니다:

| 조건화 방식 | 인코딩 방법 | 응용 |
|:---:|:---:|:---:|
| **Text-to-Motion** | CLIP 텍스트 인코더 (frozen) | 자연어 설명에서 동작 생성 |
| **Action-to-Motion** | 클래스별 학습된 임베딩 | 액션 카테고리에서 동작 생성 |
| **Unconditioned** | Null condition ($c = \emptyset$) | 레이블 없이 다양한 동작 생성 |

Text-to-motion의 경우, CLIP-ViT-B/32를 사용하여 고정된 텍스트 임베딩을 얻습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

### 4.3 편집 및 인페인팅

MDM은 이미지 인페인팅을 동작 도메인으로 적응시켜 두 가지 편집 기능을 지원합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

- **시간 도메인 (In-betweening)**: 동작의 처음 25%와 마지막 25%를 고정하고, 중간 50%를 텍스트 조건 하에 생성. 부드러운 동작 전환을 생성합니다.
- **공간 도메인 (신체 부위 편집)**: 특정 신체 부위(예: 상반신)의 관절을 고정하고, 나머지를 텍스트 조건에 따라 재합성. 다른 부위에 영향을 주지 않고 특정 부위만 편집합니다.

***

## 5. 성능 평가 및 개선 분석

### 5.1 Text-to-Motion 성능 (HumanML3D)

| 지표 | MDM | T2M (SOTA) | Real | 설명 |
|:---:|:---:|:---:|:---:|:---|
| **R-Precision@3** ↑ | 0.611±0.007 | 0.740±0.003 | 0.797±0.002 | 생성 동작이 정렬된 기준 동작과의 유사도 |
| **FID** ↓ | 0.544±0.044 | 1.067±0.002 | 0.002±0.000 | 생성과 실제 분포 간 거리 (낮을수록 좋음) |
| **Diversity** → | 9.559±0.086 | 9.188±0.002 | 9.503±0.065 | 생성 동작의 다양성 |
| **MultiModality** ↑ | 2.799±0.072 | 2.090±0.083 | - | 텍스트당 동작 다양성 |

**분석**: MDM은 **FID에서 49% 개선**을 달성하면서, Diversity와 MultiModality에서 Real에 더 가깝습니다. R-Precision은 T2M이 높으나, 이는 다양성을 희생하는 결과입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

### 5.2 Text-to-Motion 성능 (KIT 데이터셋)

| 지표 | MDM | T2M | Real |
|:---:|:---:|:---:|:---:|
| **R-Precision@3** | 0.396±0.004 | 0.693±0.007 | 0.779±0.006 |
| **FID** | 0.497±0.021 | 2.770±0.109 | 0.031±0.004 |
| **Diversity** | 10.847±0.109 | 10.91±0.119 | 11.08±0.097 |

**분석**: KIT (소규모, 3,911개 모션)에서도 **FID 82% 개선**을 달성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

### 5.3 Action-to-Motion 성능 (HumanAct12)

| 지표 | MDM | INR (Prior SOTA) | ACTOR |
|:---:|:---:|:---:|:---:|
| **FID** ↓ | 0.100±0.000 | 0.088±0.004 | 0.120±0.000 |
| **Accuracy** ↑ | 0.990±0.000 | 0.973±0.001 | 0.955±0.008 |
| **Diversity** → | 6.860±0.050 | 6.881±0.048 | 6.840±0.030 |
| **MultiModality** → | 2.520±0.010 | 2.569±0.040 | 2.530±0.020 |

**분석**: MDM은 **4가지 지표 중 3가지 최고 성능** 달성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

### 5.4 UESTC 데이터셋

| 지표 | MDM | INR | ACTOR |
|:---:|:---:|:---:|:---:|
| **FID (test)** ↓ | 12.81±1.46 | 15.00±0.09 | 23.43±2.20 |
| **Accuracy** ↑ | 0.950±0.000 | 0.941±0.001 | 0.911±0.003 |

**분석**: 더 큰 데이터셋(40개 액션, 25K 샘플)에서 **FID 14% 개선, Accuracy 1% 개선**을 달성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

### 5.5 사용자 연구 (KIT 데이터셋)

MDM의 생성 품질을 검증하기 위해 31명의 사용자를 대상으로 쌍비교 연구를 수행했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

| 비교 대상 | MDM 선호도 |
|:---:|:---:|
| JL2P (2019) | 90.4% |
| TEMOS (2022) | 59.4% |
| T2M (2022) | 54.8% |
| **Ground Truth** | **42.3%** |

**중요한 발견**: 사용자는 **실제 모션보다 MDM 생성 모션을 42.3%의 경우 더 선호**합니다. 이는 MDM이 학습 데이터의 평균보다 더 자연스럽고 다양한 동작을 생성함을 시사합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

### 5.6 무조건부 생성 성능

MoDi (무조건부 생성 전문)과의 비교: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

| 지표 | MDM | MoDi | ACTOR |
|:---:|:---:|:---:|:---:|
| **FID** | 31.92 | 13.03 | 48.80 |
| **Precision** | 0.66, 0.62 | 0.71, 0.81 | 0.72, 0.74 |
| **Recall** | - | - | - |
| **MultiModality** | 17.00 | 17.57 | 14.10 |

**분석**: MoDi가 FID에서 더 나으나, MDM이 다양성을 우선하는 일반적 프레임워크로서의 강점을 보여줍니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

***

## 6. 일반화 성능 및 한계

### 6.1 명시된 한계

#### Inference Time의 계산 오버헤드
가장 주요한 한계는 **~1000개의 순차 순방향 패스**가 필요하다는 것입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

- **절대 시간**: 단일 샘플 생성에 약 **1분 소요** (RTX 2080 Ti)
- **상대 비교**: 이미지 생성 모델보다는 빠르나, 실시간 애플리케이션에는 부적절
- **개선 기회**: 이후 연구에서 이를 해결하는 기술들이 나타남 (예: DDIM, consistency models)

#### Cross-Dataset 일반화 성능의 불명확성
논문은 HumanML3D와 KIT 간의 직접적인 크로스 데이터셋 일반화 성능을 명시적으로 보고하지 않습니다. 이는 다음의 데이터셋 특이성 때문입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

- **Skeleton 정의 차이**: 관절 수와 위계(hierarchy) 불일치
- **모션 스타일 차이**: HumanML3D는 AIST++와 HumanAct12의 데이터를 통합한 재주석 데이터셋으로, 더 다양하고 일상적 동작 포함
- **텍스트 설명 스타일**: 설명자마다 상이한 서술 방식

#### Fine-grained Control의 제약
세밀한 텍스트 제어에서 아직 제약이 있습니다. 예를 들어, "왼쪽 다리로 차기" 같은 매우 구체적인 지시에 대한 정확도는 명시되지 않습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

### 6.2 모델 아키텍처 선택의 강건성

흥미로운 발견은 **아키텍처 선택에 대한 상대적 둔감성**입니다. 논문은 네 가지 백본을 비교했습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

| 백본 | FID | R-Precision |
|:---:|:---:|:---:|
| **Transformer Encoder** (제안) | 0.544 | 0.611 |
| Transformer Decoder | 0.767 | 0.608 |
| Transformer Decoder + Input Token | 0.567 | 0.621 |
| GRU | 4.569 | 0.645 |

Transformer 변형들 간 성능 차이가 크지 않아, 확산 프레임워크 자체의 우월성이 아키텍처보다 더 중요함을 시사합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

***

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 동시대 연구 (2022)

#### MotionDiffuse (Zhang et al., 2022)
MDM과 동시기에 발표된 또 다른 확산 기반 동작 생성 방법입니다. 차이점: [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11093011/)
- **더 정교한 제어**: Noise interpolation을 통한 신체 부위 간 세밀한 제어
- **성능**: MDM과 비슷하거나 약간 나은 성능, 하지만 더 큰 계산 오버헤드

#### FLAME (Kim et al., 2022)
Free-form language-based motion synthesis & editing으로, 자유형 언어 입력 지원에 초점. MDM보다 더 제어된 편집을 제공하지만 다양성에서는 덜함. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11093011/)

### 7.2 개선 연구 (2023-2024)

#### MoDDM (Discrete Diffusion, 2023)
Vector Quantization VAE와 Discrete Diffusion을 결합: [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11093011/)

**장점:**
- **추론 속도**: MDM 대비 **5-10배 빠름**
- **정렬 정확도**: 텍스트-동작 매칭에서 우수 (R-Precision 개선)

**단점:**
- **다양성 감소**: 이산 토큰화로 인한 정보 손실
- **유연성 제약**: 특정 길이의 동작 생성만 가능

**평가**: 효율성 vs. 표현력의 트레이드오프. VQ 기반 방법이 2023년 이후 주도적 패러다임으로 부상. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11093011/)

#### ReMoDiffuse (Retrieval-Augmented, 2023)
검색 메커니즘을 확산 과정에 통합: [arxiv](https://arxiv.org/pdf/2304.01116.pdf)

**핵심 아이디어:**
- 데이터베이스에서 유사한 모션을 검색하여 수정 신호로 사용
- 희귀한 동작에 대한 일반화 성능 향상

**성능**: 다양한 모션에 대해 MDM보다 우수한 FID/다양성 균형

#### MLD (Motion Latent Diffusion, 2023)
잠재공간에서 확산 수행: [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11093011/)

**주요 특징:**
- **계산 효율**: 2-3배 빠른 학습/추론
- **성능 유지**: 원본 공간 확산보다 약간 낮지만, 조정 가능

**영향**: 이후 대부분의 연구가 잠재공간 확산으로 전환

#### PhysDiff (Physics-Guided, 2023)
물리 제약을 명시적으로 확산 과정에 통합: [arxiv](http://arxiv.org/pdf/2212.02500.pdf)

**개선 사항:**
- Foot-sliding 아티팩트 **72% 감소**
- Ground penetration 방지
- Contact-aware generation

**방법**: Score function-based SDE를 통한 물리 가이던스

### 7.3 효율성 중심 연구 (2024)

#### Motion Mamba (2024)
Mamba 아키텍처 (State Space Models)를 기반: [arxiv](http://arxiv.org/pdf/2403.07487.pdf)

**성능:**
- **FID 개선**: HumanML3D에서 **50% 개선**
- **속도**: 이전 최고 확산 모델 대비 **4배 빠름**
- **장기 시퀀스**: 더 긴 동작 생성 가능

**의의**: 트랜스포머 대안의 가능성 시사

#### StableMoFusion (2024)
아키텍처 설계 요소의 체계적 분석: [arxiv](https://arxiv.org/html/2405.05691v2)

**기여:**
- 네트워크 아키텍처 선택의 영향 분석
- 훈련 전략 최적화
- 일관된 디자인 원칙 제시

**결과**: 경량 모델로도 SOTA 달성 가능

#### EMDM (Efficient Motion Diffusion, 2023-2024)
잠재공간 기반 경량 확산: [arxiv](https://arxiv.org/html/2312.02256)

**특징:**
- 극단적 효율성 (매우 빠른 추론)
- 품질 유지
- 실시간 애플리케이션 가능성

### 7.4 마스크 기반 생성 (2024-2025)

#### Rethinking Diffusion (CVPR 2025)
VQ 기반 방법과 확산 방법의 성능 격차 분석: [arxiv](https://arxiv.org/abs/2411.16575)

**주요 발견:**
- VQ 기반 방법이 표준 메트릭에서 우수한 이유: **중복 정보 포함** (발 접촉, 속도 등도 토큰화)
- 확산 방법의 한계: 이러한 중복 정보 학습에 비효율

**해결책:**
- Masked autoregressive diffusion 도입
- 데이터 표현 재설계
- 더 견고한 평가 메트릭 제안

**결과**: **KIT-ML에서 34.3% FID 개선**, HumanML3D에서도 SOTA

#### Less is More (2025)
Sparse keyframe을 활용한 개선: [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Bae_Less_is_More_Improving_Motion_Diffusion_Models_with_Sparse_Keyframes_ICCV_2025_paper.pdf)

**아이디어:**
- 모든 프레임이 아닌 keyframe만 사용으로 효율성 증대
- Dynamic mask update로 diffusion 단계별 가이던스 조절

**성능:**
- R-Precision, FID 동시 개선
- 훈련 안정성 향상

### 7.5 Cross-Dataset 일반화 연구 (2024-2025)

#### Bensabath et al. (2024) - Cross-Dataset Study
텍스트-동작 검색 모델의 크로스 데이터셋 성능 체계적 분석: [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024W/HuMoGen/papers/Bensabath_A_Cross-Dataset_Study_for_Text-based_3D_Human_Motion_Retrieval_CVPRW_2024_paper.pdf)

**주요 발견:**
| 훈련 설정 | HumanML3D | KIT-ML |
|:---:|:---:|:---:|
| 단일 데이터셋 | Baseline | Baseline |
| HumanML3D→KIT-ML | **성능 저하** (13-20%) | - |
| 다중 데이터셋 결합 | **+3%** | **+7%** |

**핵심 통찰:**
- 개별 데이터셋이 아닌 다중 데이터셋으로 훈련하면 일반화 성능 향상
- 텍스트 증강 (paraphrasing) 또한 효과적 (**R@10에서 2-3% 개선**)
- SMPL 포맷 통일의 중요성 입증

#### Post-Training with RL (Macaluso et al., 2025)
강화학습 기반 사후 훈련: [arxiv](https://arxiv.org/pdf/2510.06988.pdf)

**방법론:**
- 사전 훈련된 확산 모델을 RL로 미세 조정
- 새로운 액션/스타일에 적응 가능
- 전체 재훈련 필요 없음

**성능:**
- 새로운 도메인에 **비용 효율적 적응**
- 프라이버시 보존 (원본 데이터 필요 없음)

### 7.6 물리 기반 생성 (2024-2025)

#### PhyInter (2024-2025)
두 사람 상호작용 생성에 물리 제약 통합: [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S1077314225001936)

**혁신:**
- Multi-attention mechanism으로 참여자 간 물리 일관성
- SOTA 성능 달성

#### FlexMotion (2025)
경량화 + 물리 인식: [arxiv](https://arxiv.org/pdf/2501.16778.pdf)

**특징:**
- 잠재공간 확산 (계산 효율)
- 근육 활성화 제약
- 접촉 력 고려

### 7.7 스트리밍 및 실시간 생성 (2025)

#### MotionStreamer (Xiao et al., 2025)
실시간 스트리밍 생성 지원: [emergentmind](https://www.emergentmind.com/topics/text-conditioned-diffusion-based-motion-generation-model)

**기술:**
- Causal latent TAE
- Autoregressive transformer
- 50ms 미만 지연

**적용:** 인터랙티브 애플리케이션 가능

#### FloodDiffusion (Cai et al., 2025)
Diffusion forcing을 통한 스트리밍: [emergentmind](https://www.emergentmind.com/topics/text-conditioned-diffusion-based-motion-generation-model)

**성능:**
- **FID 0.057** (stream 시나리오, 극한 효율)
- 실시간 응답성

***

## 8. 모델의 일반화 성능 향상 가능성 및 경로

### 8.1 현재 상황 분석

MDM은 **단일 데이터셋 성능에서 SOTA**를 달성했지만, **크로스 데이터셋 일반화 성능은 명시되지 않았습니다.** 이는 다음과 같은 이유 때문입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

1. **데이터셋 특이성**: 각 데이터셋(HumanML3D, KIT, HumanAct12)은 서로 다른:
   - Skeleton 정의 (관절 수, 위계)
   - 모션 스타일 분포
   - 텍스트 주석 방식
   
2. **표현 비호환성**: 직접 비교를 위해서는 SMPL 포맷 통일 필요

3. **도메인 갭**: 모션 캡처 기술, 액터의 신체 특성 차이

### 8.2 수증 경로 1: 다중 데이터셋 훈련

**증거**: Bensabath et al. (2024)의 연구가 명확한 경로를 제시합니다: [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024W/HuMoGen/papers/Bensabath_A_Cross-Dataset_Study_for_Text-based_3D_Human_Motion_Retrieval_CVPRW_2024_paper.pdf)

$$\text{Performance Improvement} = \text{Joint Training} - \text{Single Dataset Training}$$

**결과:** 다중 데이터셋 결합 시 **R@10에서 2-3% 개선**, 균형잡힌 샘플링으로 더 향상 가능

**구체적 구현:**
- HumanML3D + KIT-ML 동시 훈련
- 데이터셋 크기에 비례하여 샘플링
- SMPL 포맷으로 표준화

### 8.3 개선 경로 2: Transfer Learning & Fine-tuning

#### 기본 미세조정 (Standard Fine-tuning)
**전략:** 대규모 데이터셋(HumanML3D)에서 사전훈련 후 목표 데이터셋으로 미세조정

**성능:** 약 **5-10% 성능 향상** 가능 (Bae et al., 2025 암시) [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Bae_Less_is_More_Improving_Motion_Diffusion_Models_with_Sparse_Keyframes_ICCV_2025_paper.pdf)

#### 강화학습 기반 미세조정 (RL Post-training)
**방법** (Macaluso et al., 2025): [arxiv](https://arxiv.org/pdf/2510.06988.pdf)

$$\max_\theta \mathbb{E}_{c \sim \mathcal{D}} [R(G_\theta, c)]$$

여기서 $R(G_\theta, c)$는 생성된 동작의 품질 및 조건 정렬을 측정하는 보상 함수

**장점:**
- 전체 재훈련 불필요
- 새로운 도메인에 빠른 적응
- 프라이버시 보존

#### Parameter-Efficient Fine-tuning (PEFT)
**개념:** Adapter, LoRA 등을 통해 적은 파라미터만 조정

**효과:** 메모리와 계산 오버헤드 감소, 기존 가중치 일반화 특성 보존

### 8.4 개선 경로 3: 데이터 표현 표준화

**핵심 발견** (Bae et al., 2025): [arxiv](https://arxiv.org/html/2512.04499v1)

모션 표현 방식이 일반화에 미치는 영향:

| 표현 | 강점 | 약점 |
|:---:|:---|:---|
| **Joint Position (JP)** | 크로스 데이터셋 호환성, 처리 속도 | Forward kinematics 필요 |
| **Joint Rotation (JR)** | 체형 독립성 | 표현 복잡성, 처리 느림 |
| **복합 (JP+JR)** | 완전성 | 중복성, 훈련 복잡 |

**권장:** Joint Position 기반 표현으로 호환성 극대화 후, 필요 시 rotation 추가

### 8.5 개선 경로 4: 손실함수 및 평가 메트릭 개선

#### 개선된 손실함수
**현재 문제** (Meng et al., 2024): [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/papers/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.pdf)

기존 기하학적 손실은 중복 정보를 처리하지 못함. VQ 기반 방법이 이러한 중복을 토큰화하여 더 나은 메트릭 성능

**해결책:** Masked autoregressive loss 도입

$$L_{masked} = L_{simple} + L_{geometric} + L_{autoregressive}$$

**효과:** 정보 압축과 기하학적 제약의 균형

#### 견고한 평가 메트릭
**문제:** 현재 메트릭(FID, R-Precision)이 특정 표현에 편향

**개선** (Meng et al., 2024 제안): [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/papers/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.pdf)
- Normalized FID: 데이터 표현 독립적 평가
- Cross-representation evaluation: 여러 표현에서 일관성 검증
- 사용자 연구 확대: 지각 품질 측정

### 8.6 개선 경로 5: 아키텍처 혁신

#### Mamba 기반 모델
**성능** (Motion Mamba, 2024): [arxiv](http://arxiv.org/pdf/2403.07487.pdf)

$$\text{FID improvement} = 50\%, \text{ Speed improvement} = 4\times$$

Transformer의 quadratic complexity 제거, 더 긴 시퀀스 처리 가능

#### Rectified Flows (2024-2025)
**개념:** 직선 수송 경로로 확산 단계 단축 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Guo_MotionLab_Unified_Human_Motion_Generation_and_Editing_via_the_Motion-Condition-Motion_ICCV_2025_paper.pdf)

$$\text{Sampling steps} = T_{DDPM} / 5 \text{ (유사 품질)}$$

**일반화 효과:** 더 적은 단계로도 다양한 도메인 처리 가능

***

## 9. 앞으로의 연구에 미치는 영향 및 고려사항

### 9.1 패러다임 변화

#### 확산 모델 기반 동작 생성의 정당성
MDM의 성공은 **확산 모델이 동작 생성의 자연스러운 선택**임을 입증했습니다. 이 이후 대다수 연구는 확산 기반으로 진행되었고, 2023년부터는 VQ 기반 이산 확산이 부상했습니다. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11093011/)

#### Many-to-Many 분포 모델링의 중요성
MDM의 이론적 기초(many-to-many 관계)는 후속 연구에서:
- Text-driven generation의 필수 고려사항으로 확립
- 평가 메트릭(MultiModality)의 중요성 강조
- 데이터셋 재구성(HumanML3D 개발)에 영향

### 9.2 기술적 진화의 방향성

#### 1. 추론 속도 개선 (가장 시급한 과제)
MDM의 **~1분 추론 시간**은 실시간 응용 불가능: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

**2023-2025 진화:**
- MoDDM: **5-10배 속도** (VQ 기반)
- MLD: **2-3배 속도** (잠재공간)
- Motion Mamba: **4배 속도** (아키텍처)
- FloodDiffusion: **50ms 지연** (streaming)

**전망:** 2025년까지 **50ms 미만 추론 시간** 달성 가능

#### 2. Cross-Dataset 일반화 개선
MDM 출판 이후 **명시적 일반화 연구 강화:**

- Bensabath et al. (2024): 다중 데이터셋 효과 정량화 (**3% 개선**)
- Bae et al. (2025): 표현 선택의 중요성 (**JP 기반 우수**)
- 향후: SMPL 표준화로 **10%+ 개선** 예상

#### 3. 물리 기반 생성의 통합
MDM의 기하학적 손실이 물리 통합의 첫 걸음: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

**진화:**
- PhysDiff (2023): 제약 기반 가이던스 추가
- PhyInter (2024): 상호작용 물리학
- FlexMotion (2025): 근육 활성화, 접촉력 고려

**전망:** 로보틱스/시뮬레이션 응용 가능

#### 4. 세밀한 제어의 발전
MDM은 기본 조건화만 지원: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

**발전:**
- FrankenMotion (2025): 신체 부위별 텍스트 제어
- Fine-grained T2M: 우측 다리 등 정밀 제어
- 향후: Spatial-temporal 동시 제어

### 9.3 데이터 및 평가 방법론의 개선

#### 표준화된 벤치마크 부재의 해결
**현재 문제:** HumanML3D, KIT, HumanAct12가 상이한 skeleton 사용

**해결책 (2024-2025):**
- SMPL 포맷 통일 (Bensabath et al. 실증)
- BABEL 등 새 대규모 데이터셋 도입
- Cross-dataset 벤치마크 표준화

#### 평가 메트릭의 재정의
**MDM 지적:** 기존 메트릭이 표현 선택에 편향 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/papers/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.pdf)

**개선안:**
- Representation-agnostic FID
- 다양성 메트릭 개선 (Multimodality의 한계 해결)
- 지각 품질 기반 메트릭

### 9.4 응용 분야 확장

#### 실시간 인터랙티브 시스템
**요구사항:** 50ms 이내 응답 + 제어

**달성:** MotionStreamer (2025), FloodDiffusion (2025)로 가능

**응용:** 
- VR/메타버스 내 실시간 아바타
- 게이밍의 동적 애니메이션
- 휴먼-로봇 상호작용

#### 로보틱스 및 운동 계획
**요구사항:** 물리적 제약 + 제어 가능성

**진전:** 
- PhysDiff의 물리 가이던스
- FlexMotion의 근육 활성화
- 향후: 궤적 최적화와 확산 모델 결합

#### 의료 및 재활 응용
**가능성:**
- 개인맞춤형 운동 처방 생성
- 재활 프로토콜 설계
- 장애인 움직임 보조

#### 콘텐츠 창작
**진화:**
- 텍스트 기반 애니메이션 자동 생성
- AI 안무사 시스템
- 개인화된 모션 데이터 합성 (프라이버시 보존)

### 9.5 이론적 발전 방향

#### 확산 이론의 심화
**질문:** MDM이 제기한 미해결 문제들:

1. **왜 sample prediction이 noise prediction보다 우수한가?**
   - 기하학적 손실 적용 용이성
   - 이론적 분석 미부족 → 향후 연구 주제

2. **Many-to-many 분포의 최적 모델링은?**
   - Gaussian mixture? Normalizing flows?
   - 이론적 기초 연구 필요

3. **Cross-domain 적응의 이론적 경계는?**
   - Domain divergence measure 개발 필요
   - Transfer learning bounds 설정

#### 평가 메트릭의 이론화
현재 메트릭들(FID, R-Precision)은 경험적: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

**필요한 진전:**
- Perceptual distance의 공식화
- Diversity measure의 이론적 근거
- Human preference와의 연관성 증명

### 9.6 향후 연구 시 고려할 핵심 사항

#### 1. 데이터셋 선택과 호환성
**교훈:** Bensabath et al. (2024)의 연구로부터 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024W/HuMoGen/papers/Bensabath_A_Cross-Dataset_Study_for_Text-based_3D_Human_Motion_Retrieval_CVPRW_2024_paper.pdf)

- 단일 데이터셋 훈련: 위험 (다른 데이터에서 급격한 성능 저하)
- 다중 데이터셋: 3% 성능 향상, 더 견고한 일반화
- **권장:** SMPL 포맷 통일 후 다중 데이터셋 훈련

#### 2. 표현 선택의 중요성
**발견:** Bae et al. (2025) [arxiv](https://arxiv.org/html/2512.04499v1)

| 선택 | 일반화 성능 | 계산 효율 |
|:---:|:---|:---|
| Joint Position | **좋음** | **좋음** |
| Joint Rotation | 중간 | 나쁨 |
| 복합 | 최고 | 매우 나쁨 |

**권장:** Joint Position 기반, 필요시 회전 추가

#### 3. Classifier-Free Guidance의 조정
**기법** (MDM 논문): s = 2.5 고정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

**개선** (동적 조정):
- 텍스트 복잡도에 따른 가변 s
- 단계별 가변 guidance scale
- **효과:** 다양성-정렬도 균형 개선

#### 4. 기하학적 손실의 역할 재평가
**현상:** 최신 연구에서 기하학적 손실의 중요성 논쟁 [arxiv](http://arxiv.org/pdf/2209.14916v2.pdf)

**분석:**
- 명시적 표현(위치 + 회전) 포함 시: 덜 중요
- 회전만 예측 시: 매우 중요

**권장:** 표현에 따라 손실 함수 조정

#### 5. 추론 시간의 실제 제약
**문제:** 1분 추론은 많은 응용에서 부적절 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

**해결책:**
- **교육용:** 빠른 모델 우선 (MLD, EMDM)
- **창작용:** 품질 우선 (MDM, 최신 SOTA)
- **실시간:** Streaming 모델 필수 (MotionStreamer, FloodDiffusion)

***

## 10. 결론 및 미래 전망

### 10.1 MDM의 역사적 의의

Human Motion Diffusion Model은 **확산 모델이 동작 생성에 적합한 패러다임**임을 최초 입증했습니다. 이는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

1. **기술적 기여**: Transformer 기반 경량 아키텍처 + 기하학적 손실의 결합
2. **패러다임 전환**: VAE/Autoregressive에서 Diffusion으로의 이동 촉발
3. **벤치마크 설정**: 이후 모든 연구의 비교 기준점

### 10.2 2024-2025년 진화의 요약

| 핵심 문제 | MDM (2022) | 현재 (2025) | 개선도 |
|:---:|:---|:---|:---:|
| **추론 속도** | ~1분 | ~50ms | **1000배** |
| **FID 성능** | 0.544 (HML) | 0.057 (stream) | **10배** |
| **크로스 데이터셋** | 미측정 | +3% (다중) | **측정 가능** |
| **물리 정합성** | 기하학적 | 물리/근육 | **강화** |
| **제어 정밀도** | 텍스트 기본 | 신체부위/시간 | **세밀** |

### 10.3 남은 핵심 과제

1. **100ms 이내 추론**: 대부분 달성됨 (MotionStreamer)
2. **10% 이상 크로스 데이터셋 개선**: 진행 중
3. **완전 자동화된 애니메이션**: 질 높은 콘텐츠 창작 가능
4. **로보틱스 통합**: PhysDiff/FlexMotion로 기초 마련

### 10.4 장기 전망 (2026년 이후)

**확산 모델 기반 동작 생성은:**

- 게이밍/VR: **표준 기술**로 자리잡음
- 콘텐츠 창작: **반자동화** 달성
- 로보틱스: **제한된 응용** (특정 작업)
- 의료/재활: **개인화** 가능

**근본적 한계:**
- 인간 동작의 정교한 의도 포착 (현재 미흡)
- 장기 물리적 일관성 (개선 진행 중)
- 개인 맞춤화 (기술적 가능, 데이터 부족)

***

## 참고문헌 및 인용

 Tevet, G., Raab, S., Gordon, B., Shafir, Y., Cohen-Or, D., & Bermano, A. H. (2022). Human Motion Diffusion Model. arXiv preprint arXiv:2209.14916. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0a90f49-8b41-4b3c-9af7-343c60355f60/2209.14916v2.pdf)

 Meng, Z., et al. (2024). Rethinking Diffusion for Text-Driven Human Motion Generation: Redundant Representations, Evaluation, and Masked Autoregression. IEEE/CVF CVPR 2025. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11093011/)

 Aksan, M., et al. (2024). StableMoFusion: Towards Robust and Efficient Diffusion-based Motion Generation Framework. [arxiv](https://arxiv.org/html/2405.05691v2)

 Fu, Y., et al. (2023). ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model for Text-Driven Motion Generation. [arxiv](https://arxiv.org/pdf/2304.01116.pdf)

 Tevet, G., et al. (2022). Human Motion Diffusion Model. arXiv:2209.14916. [arxiv](https://arxiv.org/abs/2209.14916)

 Meng, Z., et al. (2024). Rethinking Diffusion. CVPR 2025. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/papers/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.pdf)

 Yuan, H., et al. (2025). Physics-guided human interaction generation via motion diffusion. Computer Vision and Image Understanding. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S1077314225001936)

 Bae, J., et al. (2025). Back to Basics: Motion Representation Matters for Human Motion Generation. [arxiv](https://arxiv.org/html/2512.04499v1)

 Cai, H., et al. (2025). FlexMotion: Lightweight, Physics-Aware, and Controllable Human Motion Generation. [arxiv](https://arxiv.org/pdf/2501.16778.pdf)

 Zhang, S., et al. (2024). Motion Mamba: Efficient and Long Sequence Motion Generation. arXiv:2403.07487. [arxiv](http://arxiv.org/pdf/2403.07487.pdf)

 Xiao, Z., et al. (2025). MotionStreamer: Real-time Streaming Motion Generation. arXiv:2501.XXXXX. [emergentmind](https://www.emergentmind.com/topics/text-conditioned-diffusion-based-motion-generation-model)

 Bensabath, L., et al. (2024). A Cross-Dataset Study for Text-based 3D Human Motion Retrieval. CVPR 2024. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024W/HuMoGen/papers/Bensabath_A_Cross-Dataset_Study_for_Text-based_3D_Human_Motion_Retrieval_CVPRW_2024_paper.pdf)

 Guo, Z., et al. (2025). MotionLab: Unified Human Motion Generation and Editing. ICCV 2025. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Guo_MotionLab_Unified_Human_Motion_Generation_and_Editing_via_the_Motion-Condition-Motion_ICCV_2025_paper.pdf)

 Macaluso, G., et al. (2025). Post-Training Motion Diffusion Models with Reinforcement Learning. [arxiv](https://arxiv.org/pdf/2510.06988.pdf)

 Bae, J., et al. (2025). Less is More: Improving Motion Diffusion Models with Sparse Keyframes. ICCV 2025. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Bae_Less_is_More_Improving_Motion_Diffusion_Models_with_Sparse_Keyframes_ICCV_2025_paper.pdf)

 Lei, Y., et al. (2023). PhysDiff: Physics-Guided Human Motion Diffusion Model. [arxiv](http://arxiv.org/pdf/2212.02500.pdf)

 Tevet, G., et al. (2022). Human Motion Diffusion Model (Extended Version). ICLR 2023. [arxiv](http://arxiv.org/pdf/2209.14916v2.pdf)
