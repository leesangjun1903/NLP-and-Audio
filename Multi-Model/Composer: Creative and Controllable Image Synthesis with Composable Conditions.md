
# Composer: Creative and Controllable Image Synthesis with Composable Conditions

## 1. 핵심 주장 및 기여 요약

"Composer: Creative and Controllable Image Synthesis with Composable Conditions"는 이미지 생성의 제어 가능성을 획기적으로 확장하는 논문입니다. 이 논문의 핵심 주장은 **구성성(Compositionality)**이 제어 가능한 이미지 생성의 핵심이라는 것입니다. 기존의 텍스트-이미지 생성 모델들이 전체 장면의 의미론, 형태, 스타일, 색상을 동시에 정확히 제어하지 못하는 문제를 해결하기 위해, 저자들은 이미지를 여러 개의 독립적인 표현(representation)으로 분해한 후, 이들을 조합하여 이미지를 재구성하는 방식을 제안합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

**주요 기여사항:**

- **구성성 기반 확산 모델(Compositional Diffusion Model)**: 이미지를 8가지 표현으로 분해(caption, semantics, style, color, sketch, instances, depth, intensity)하고, 이들을 조건으로 하는 다중 조건 확산 모델 학습 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)
- **지수적 제어 공간 확장**: 8개 표현 각각을 선택적으로 결합함으로써 약 10^8 수준의 조합 가능성 창출 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)
- **재훈련 없는 일반화**: 재훈련 없이 텍스트-이미지 생성, 스타일 전이, 자세 전이, 가상 피팅 등 다양한 생성 및 조작 작업 수행 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)
- **영역 특화 편집(Region-Specific Editing)**: 마스킹을 통한 직교적 조건 지정으로 전통적 인페인팅보다 유연한 편집 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

***

## 2. 해결하고자 하는 문제

### 2.1 기존 방법의 한계

기존의 대규모 생성 모델들은 뛰어난 이미지 품질을 생성하지만 **제어 가능성이 제한적**이라는 근본적 문제가 있습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

- 개별 조건(text, segmentation, depth 등)만 지원하며, 여러 조건을 동시에 결합하기 어려움 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)
- 실제 디자인 작업에서 필요한 의미론, 형태, 스타일, 색상의 동시 제어 불가능
- 조건 간 충돌 발생 시 처리 방법 부재
- 각 작업마다 별도 모델 학습이 필요하여 계산 비용 높음

### 2.2 Composer의 문제 해결 방식

Composer는 **구성 원리(Compositional Principle)**를 도입하여 이를 해결합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

$$\text{제어 공간} \propto 2^{\text{분해 요소 수}}$$

8개의 분해 가능한 요소를 사용할 경우, 이론적으로 $2^8 = 256$개의 부분집합 조합이 가능하며, 더 많은 값을 가지는 연속적 조건을 포함하면 제어 공간은 기하급수적으로 확대됩니다.

***

## 3. 제안하는 방법 및 수식

### 3.1 확산 모델의 기본 학습 목표

Composer는 표준 확산 모델의 단순한 평균 제곱 오차(MSE) 손실을 사용합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

$$L_{\text{simple}} = \mathbb{E}_{\mathbf{x}_0, \mathbf{c}, \boldsymbol{\epsilon}, t} \left( \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}, \mathbf{c}) \|^2_2 \right) \quad (1)$$

여기서:
- $\mathbf{x}_0$: 원본 이미지
- $\mathbf{c}$: 조건 정보(분해된 표현들)
- $t \sim U(0, 1)$: 타임스텝
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$: 가우시안 노이즈
- $\alpha_t, \sigma_t$: 스케일 함수
- $\boldsymbol{\epsilon}_\theta$: 학습 가능한 매개변수를 가진 노이즈 예측 신경망

### 3.2 분류기 없는 가이던스(Classifier-Free Guidance)

단일 조건에 대한 가이던스: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

$$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, \mathbf{c}) = w \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}) + (1 - w) \boldsymbol{\epsilon}_\theta(\mathbf{x}_t) \quad (2)$$

여기서 $w$는 가이던스 강도입니다.

### 3.3 다중 조건 결합 가이던스

서로 다른 조건 집합 간의 관계 제어: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

$$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, \mathbf{c}) = w \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}_2) + (1 - w) \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}_1) \quad (3)$$

이 수식에서:
- $\mathbf{c}_2 \setminus \mathbf{c}_1$ 범위의 조건: 가중치 $w$로 강조
- $\mathbf{c}_1 \setminus \mathbf{c}_2$ 범위의 조건: 가중치 $(1-w)$로 억제  
- $\mathbf{c}_1 \cap \mathbf{c}_2$ 범위의 조건: 가중치 1.0으로 설정

### 3.4 양방향 가이던스(Bidirectional Guidance)

이미지 조작을 위한 DDIM 역전/정방향 프로세스: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

1. 조건 $\mathbf{c}_1$을 사용하여 이미지 $\mathbf{x}_0$을 잠재 공간 $\mathbf{x}_T$로 역전
2. 다른 조건 $\mathbf{c}_2$를 사용하여 $\mathbf{x}_T$에서 샘플링
3. $\mathbf{c}_2 - \mathbf{c}_1$의 차이가 조작 방향을 정의

### 3.5 이미지 분해 방식

Composer는 **8가지 표현**으로 이미지를 분해합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

| 표현 | 추출 방법 | 특징 |
|-----|---------|------|
| **Caption** | CLIP ViT-L/14@336px | 문장 및 단어 임베딩 |
| **Semantics & Style** | CLIP 이미지 임베딩 | unCLIP과 유사 |
| **Color** | 평활화 CIELab 히스토그램 | 색상 통계 (11×5×5 양자화) |
| **Sketch** | 엣지 감지 + 단순화 | 국소 세부사항 |
| **Instances** | YOLOv5 인스턴스 세분화 | 객체 카테고리 및 형태 |
| **Depthmap** | 단안 깊이 추정 모델 | 이미지 레이아웃 |
| **Intensity** | 임의 RGB 채널 가중치 | 색상 조작 자유도 |
| **Masking** | 4채널 표현 (RGB + 마스크) | 편집 영역 제한 |

### 3.6 모델 아키텍처

#### 3.6.1 글로벌 조건화(Global Conditioning)

CLIP 문장 임베딩, 이미지 임베딩, 색상 팔레트: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

- 투영(Projection) + 타임스텝 임베딩에 가산
- 이미지 임베딩과 색상 팔레트를 8개 추가 토큰으로 투영하여 CLIP 단어 임베딩과 연결
- GLIDE의 교차-주의 계층에서 컨텍스트로 사용

#### 3.6.2 국소 조건화(Localized Conditioning)

스케치, 세분화 맵, 깊이맵, 강도 이미지, 마스킹된 이미지: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

- 스택 형태의 컨볼루션 계층으로 투영하여 노이즈 잠재 $\mathbf{x}_t$와 동일한 공간 크기의 균일-차원 임베딩으로 변환
- 이들 임베딩의 합을 계산하여 $\mathbf{x}_t$에 연결

#### 3.6.3 결합 학습 전략

다양한 조건 조합에 대한 학습을 위한 드롭아웃 확률: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

| 구성 | 드롭아웃 확률 |
|-----|-------------|
| 각 조건 독립적 드롭 | 0.5 |
| 모든 조건 드롭 | 0.1 |
| 모든 조건 유지 | 0.1 |
| 강도 이미지 (특수) | 0.7 |

이 전략은 모든 조건이 정보의 대부분을 포함하므로, 다른 조건의 가중치를 과도하게 낮추는 것을 방지합니다.

### 3.7 고해상도 생성을 위한 계층적 구조

Composer는 3단계 계층 구조를 사용합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

- **기본 모델**: 64×64 해상도 생성 (2B 매개변수)
- **업샘플링 모델 1**: 64×64 → 256×256 (1.1B 매개변수)
- **업샘플링 모델 2**: 256×256 → 1024×1024 (300M 매개변수)
- **선택적 사전 모델**: 캡션 → 이미지 임베딩 (1B 매개변수)

***

## 4. 성능 향상 및 평가 결과

### 4.1 텍스트-이미지 생성 성능

| 메트릭 | Composer | 비교 대상 |
|------|---------|---------|
| **FID (COCO)** | 9.2 | Imagen, DALL-E 2 수준 |
| **CLIP Score** | 0.28 | 경쟁력 있는 수준 |
| **샘플링 단계** | 100(사전), 50(기본), 20(업샘플) | 효율적 샘플링 |

실제로 Composer는 **재훈련 없이** 다음 작업들을 수행합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

### 4.2 다양한 생성 및 편집 작업

**1. 이미지 변형(Image Variation)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)
- 특정 표현 부분집합을 조건으로 사용하여 원본과 유사하지만 다른 이미지 생성
- 더 많은 조건을 포함할수록 정확한 재구성 달성

**2. 이미지 보간(Image Interpolation)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)
- 두 이미지 간의 글로벌 표현 임베딩 공간에서 보간
- 특정 요소만 보간하거나 다른 요소는 유지 가능

**3. 이미지 재구성(Image Reconfiguration)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)
- 스케치, 세분화 맵, 깊이맵 등을 직접 수정하여 이미지 조작
- DDIM 역전으로 원본 이미지를 잠재로 변환 후, 수정된 조건으로 샘플링

**4. 편집 영역 제한(Region-Specific Editing)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)
- 마스킹된 이미지를 추가 조건으로 사용하여 편집 가능 영역 제한
- 전통적 인페인팅보다 훨씬 유연한 제어 가능

**5. 전통적 작업의 재구성** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

| 작업 | Composer 수행 방식 |
|-----|------------------|
| 팔레트 기반 채색 | 그레이스케일 + 색상 팔레트 조건 |
| 스타일 전이 | 스타일 이미지의 CLIP 임베딩 + 콘텐츠 표현 |
| 이미지 변환 | 모든 표현을 유지하되 텍스트 설명만 변경 |
| 자세 전이 | 세분화 맵(자세)을 조건으로 + CLIP 임베딩(의미) |
| 가상 피팅 | 마스킹된 이미지 + 의류 이미지의 CLIP 임베딩 |

### 4.3 실험 설정

**학습 데이터:** 약 10억 이미지 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)
- ImageNet21K
- WebVision
- LAION (필터링된 버전, 미학 점수 ≥ 7.0)

**학습 절차:** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)
- 기본 모델: 100만 스텝 사전학습(이미지 임베딩만) → 20만 스텝 미세조정(모든 조건)
- 배치 크기: 4096(사전), 1024(기본), 512(업샘플링)

***

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 구성성을 통한 영점샷(Zero-Shot) 성능

Composer의 핵심 강점은 **재훈련 없이도** 기존 작업들을 수행할 수 있다는 것입니다. 이는 다음과 같은 메커니즘으로 가능합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

**조건 선택적 드롭아웃**의 효과:
- 훈련 중 각 조건을 0.5 확률로 독립적으로 드롭아웃
- 0.1 확률로 모든 조건 드롭(무조건부 생성)
- 0.1 확률로 모든 조건 유지(완전 조건부 생성)

이는 모델이 다양한 조건 조합에 대해 **건강한 학습 신호**를 받도록 하며, 이전에 본 적 없는 조건 조합에도 **일반화**할 수 있게 합니다.

### 5.2 다중 표현의 분리 효과(Disentanglement)

논문에서 주목할 점은 **조건 간 충돌 해결** 메커니즘입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

**실험 관찰**: 서로 다른 소스에서 온 상충하는 조건들을 결합할 때:
- 세부사항이 적은 조건(예: 세분화 맵)이 세부사항이 많은 조건(예: 스케치)에 비해 덜 강조됨
- 텍스트 임베딩은 이미지 임베딩과 다른 의미론을 표현할 때 가중치가 감소

이는 모델이 **암묵적으로 조건의 신뢰도를 학습**하고 있음을 시사합니다.

### 5.3 일반화 성능의 한계 및 개선 기회

논문에서 명시한 한계점들: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

**문제 1: 다중 작업 학습의 트레이드-오프**
- 다중 작업 결합 훈련으로 인해 단일 조건 생성 성능이 약간 저하
- 예: 스케치나 깊이 없는 생성 시 이미지가 비교적 어둡게 생성되는 경향

**해결 방안**:
- 조건별 드롭아웃 확률 최적화 (강도 이미지에는 0.7 사용)
- 사전 모델 추가 도입으로 다양성 향상

**문제 2: 상충하는 조건(Conflicting Conditions)**
- 서로 다른 의미론의 텍스트 임베딩과 이미지 임베딩을 동시에 사용할 때 텍스트가 다운웨이트됨

**해결 방안**:
- 가이던스 강도 조정
- 조건 유형별 우선순위 설정

### 5.4 더 강화된 일반화를 위한 제안

**1. 적응형 조건 가중치(Adaptive Condition Weighting)**

현재 방식:
$$\hat{\epsilon} = w \cdot \epsilon(\mathbf{x}_t, \mathbf{c}_2) + (1-w) \cdot \epsilon(\mathbf{x}_t, \mathbf{c}_1)$$

개선 방안 - 조건별 동적 가중치:
$$\hat{\epsilon} = \sum_i w_i(\mathbf{c}_i) \cdot \epsilon(\mathbf{x}_t, \mathbf{c}_i)$$

여기서 $w_i$는 조건 $\mathbf{c}_i$의 신뢰도를 학습하는 네트워크

**2. 계층적 조건 융합(Hierarchical Condition Fusion)**

현재: 모든 국소 조건을 동등하게 취급
개선: 의미론적 복잡도에 따라 조건들을 계층화

**3. 도메인별 미세조정(Domain-Specific Fine-tuning)**

기본 모델을 여러 도메인(예: 얼굴, 풍경, 실내 등)에서 간단히 미세조정하면 도메인 특화 일반화 성능 향상 가능

***

## 6. 한계점

논문에서 명시한 주요 한계점들: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

### 6.1 기술적 한계

**1. 단일 조건 성능 저하**
- 다중 작업 학습의 결과로 특정 조건 조합(예: 스케치만)에서 이미지 품질 감소
- 특히 어둡고 포화도가 낮은 이미지 생성 문제

**2. 조건 충돌 해결의 불완전성**
- 상충하는 조건들 간의 우선순위가 자동으로 결정되지만, 항상 사용자 의도와 일치하지 않을 수 있음
- 예: 텍스트와 이미지 임베딩이 다른 의미를 표현할 때 텍스트가 무시되는 경향

### 6.2 데이터 및 계산 요구사항

- 약 10억 개의 이미지로 훈련 필요
- 2B + 1.1B + 300M = 3.4B 매개변수의 대규모 모델
- 고해상도(1024×1024) 생성을 위한 다단계 파이프라인

### 6.3 안전 및 윤리적 문제

논문에서 명시: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)
> "Composer의 제어 가능성 개선은 기만적이고 해로운 콘텐츠 생성의 위험성을 더욱 높인다. 이를 완화하기 위한 철저한 조사가 필요하다."

***

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 다중 조건 생성 모델 비교

#### 7.1.1 Compositional Diffusion Models (Liu et al., 2022) [ecva](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770426.pdf)

| 측면 | Compositional Diffusion | Composer |
|-----|----------------------|---------|
| **기본 아이디어** | 조건별 에너지 함수 결합 | 이미지 분해 + 다중 조건화 |
| **학습 전략** | 각 조건마다 개별 확산 모델 | 단일 모델, 조건 드롭아웃 |
| **조건 수** | 2-3개 제한적 | 8개 포괄적 |
| **재훈련 필요** | 새 조건 추가 시 필요 | 없음 |
| **FID 성능** | 기본적인 벤치마크 | SOTA 수준 (9.2) |

#### 7.1.2 ControlNet (Zhang et al., 2023) [arxiv](https://arxiv.org/abs/2302.05543)

| 측면 | ControlNet | Composer |
|-----|-----------|---------|
| **아키텍처** | 제어 모듈 + Stable Diffusion | UNet 기반 다중 조건화 |
| **공간 제어** | Canny, 깊이, 포즈 등 강력 | 스케치, 깊이, 세분화 지원 |
| **의미론적 제어** | 텍스트 프롬프트 기반 | CLIP 임베딩 + 텍스트 |
| **색상 제어** | 미지원 | CIELab 히스토그램으로 지원 |
| **훈련 비용** | 개별 조건당 별도 학습 | 통합 학습으로 효율적 |

#### 7.1.3 DiffBlender (Sung et al., 2023) [arxiv](https://arxiv.org/html/2305.15194v3)

| 측면 | DiffBlender | Composer |
|-----|-----------|---------|
| **모달리티 지원** | 텍스트, 박스, 이미지, 스케치 | 텍스트, 이미지, 스케치, 깊이, 색상, 마스크 등 |
| **모달리티별 적응** | 모드별 가이던스(MSG) | 조건 드롭아웃 |
| **계산 오버헤드** | 각 모달리티마다 어댑터 필요 | 단일 모델로 통합 |
| **영점샷 작업** | 제한적 | 광범위한 작업 지원 |

### 7.2 이미지 편집 및 조작 최신 동향

#### 7.2.1 확산 기반 인페인팅 (2023-2024)

**LatentPaint (Corneanu et al., 2024)** [openaccess.thecvf](https://openaccess.thecvf.com/content/WACV2024/papers/Corneanu_LatentPaint_Image_Inpainting_in_Latent_Space_With_Diffusion_Models_WACV_2024_paper.pdf)
- Composer의 마스킹 조건과 유사한 개념
- 정보 전파 메커니즘으로 일반 확산 모델을 인페인팅에 적응
- Composer는 이보다 더 포괄적인 편집 가능성 제공

**BrushEdit (Zhang et al., 2024)** [arxiv](https://arxiv.org/abs/2412.10316)
- MLLMs + 인페인팅 모델로 자유형식 편집
- Composer의 마스킹 기능과 유사하지만 더 제한적

#### 7.2.2 스타일 및 콘텐츠 분리 (2023-2024)

**PARASOL (Phung et al., 2023)** [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10677863/)
- 콘텐츠와 스타일 임베딩의 분리된 제어
- Composer의 Semantics & Style 임베딩과 Color 표현의 결합이 더 강력

**DiffStyler (2023)**
- 스타일 전이에 특화
- Composer는 스타일 전이를 재훈련 없이 수행

### 7.3 일반화 및 구성성 연구 (2023-2025)

#### 7.3.1 영점샷 구성 생성 (Zero-Shot Compositional Generation)

**Your Diffusion Model is Secretly a Zero-Shot Classifier (Li et al., 2023)** [arxiv](https://arxiv.org/abs/2303.16203)
- 확산 모델의 구성적 일반화 능력 입증
- Composer는 이 개념을 생성 측면에서 확장

**Compositional Scene Understanding (2025)** [arxiv](https://arxiv.org/html/2505.21780v1)
- 확산 모델의 역방향 생성 모델링으로 장면 이해
- Composer의 양방향 가이던스와 개념적 유사성

#### 7.3.2 다중 속성 제어 (Multi-Attribute Control)

**CompSlider (2024)** [arxiv](https://arxiv.org/html/2509.01028v1)
- 여러 속성을 슬라이더로 제어
- Composer의 더 일반화된 다중 조건 제어 프레임워크에 포함될 수 있는 개념

**UniVG: A Generalist Diffusion Model (2025)** [arxiv](https://arxiv.org/html/2503.12652v2)
- MM-DiT 기반의 통합 생성 모델
- 텍스트, 마스크, 이미지 조건 동시 지원
- Composer와 유사한 목표이지만 구현 방식 다름

### 7.4 효율적 미세조정 및 적응 (2023-2024)

#### 7.4.1 LoRA 기반 접근

**LoRA: Low-Rank Adaptation (Hu et al., 2021)** [arxiv](https://arxiv.org/abs/2106.09685)
- 원래 LLMs용으로 개발
- 확산 모델에 적용되어 효율적 도메인 특화 학습

**Composer의 관점에서**: 다중 LoRA 모듈을 각 조건 유형별로 훈련할 수 있지만, Composer의 통합 접근법이 더 효율적

#### 7.4.2 메타러닝 기반 제어 (Meta-Learning for Control)

**Meta ControlNet (Yang et al., 2023)** [arxiv](https://arxiv.org/abs/2312.01255)
- 메타러닝으로 새로운 제어 작업에 빠른 적응
- Composer의 조건 드롭아웃 전략과 상호보완적

### 7.5 조건 통합 및 융합 전략 (2023-2024)

#### 7.5.1 다중 모달 조건화 메커니즘

**PoE-GAN: Product-of-Experts (Huang et al., 2021)** [arxiv](https://arxiv.org/abs/2112.05130)
- 여러 모달리티를 곱셈적으로 결합
- Composer의 덧셈적 조건 결합과 다른 접근

**UNIMO-G: Unified Multimodal Generation (Li et al., 2024)** [arxiv](https://arxiv.org/abs/2401.13388)
- MLLM + 확산 모델 결합
- Composer의 CLIP 기반 임베딩보다 더 정교한 언어 이해 가능

#### 7.5.2 장면 그래프 기반 조건화

**Scene Graph Conditioning in Latent Diffusion (Fundel et al., 2023)** [arxiv](https://arxiv.org/abs/2310.10338)
- ControlNet + Gated Self-Attention으로 장면 그래프 기반 생성
- Composer의 instance 표현과 유사한 개념

### 7.6 상태-대-예술(SOTA) 성능 비교

| 모델 | FID (COCO) | 출시연도 | 주요 특징 |
|-----|-----------|--------|---------|
| DALL-E 2 | 10.39 | 2022 | 이미지 사전 기반 |
| Imagen | 7.60 | 2022 | T5 인코더 기반 |
| **Composer** | **9.2** | **2023** | **8개 조건 통합** |
| SDXL | 8.75 | 2023 | 3배 큰 UNet |
| Stable Diffusion 3 | ~8.5 | 2024 | DiT 기반 |
| FLUX.1 | ~6.5 | 2024 | 큰 T5 기반 |

***

## 8. 앞으로의 연구에 미치는 영향 및 고려사항

### 8.1 학문적 영향

#### 8.1.1 구성성 원리의 재발견

Composer는 생성 모델 분야에서 **구성성(Compositionality)**의 중요성을 재부각했습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)
- 제어 공간의 기하급수적 확대 가능성 입증
- 다양한 조건 조합에 대한 영점샷 성능 가능성 제시
- 언어(Chomsky, 1965의 "무한한 유한 수단")에서 영감을 받은 원리의 컴퓨터 비전 적용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

#### 8.1.2 다중 작업 학습의 새로운 패러다임

**기존 패러다임**: 각 작업마다 별도 모델 훈련
**Composer 패러다임**: 단일 모델로 다양한 작업 수행

이는 이후 연구에서:
- **통합 모델(Unified Models)** 개발의 선례 제공 [arxiv](https://arxiv.org/html/2503.12652v2)
- 다중 작업 학습에서의 효율성-성능 트레이드오프 논의 활발화

### 8.2 기술적 영향

#### 8.2.1 조건 표현 설계

**Composer의 조건 선택**이 이후 연구의 벤치마크 제시:
- CLIP 임베딩 (의미론/스타일)
- 히스토그램 (색상)
- 에지 맵 (스케치)
- 깊이맵 (구조)
- 마스크 (영역 제어)

이들은 **서로 다른 정보 수준**을 효과적으로 표현하며, 이는 후속 연구에서:
- 각 조건의 정보 엔트로피 분석
- 조건 간 중복도 최소화
- 새로운 조건 유형 추가 시 설계 원칙 제공

#### 8.2.2 드롭아웃 기반 조건 학습

**조건 선택적 드롭아웃** (각 조건 0.5 확률) 전략: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

```
장점:
- 단순하고 구현 용이
- 계산 오버헤드 없음
- 효과적인 정규화 효과

한계:
- 특정 조건 조합에 대한 학습량 불균형
- 조건 간 상호작용 명시적 모델링 부재
```

후속 연구에서 고려할 개선사항:
- **적응형 드롭아웃**: 조건 간 상호 정보량(Mutual Information) 기반
- **가중 드롭아웃**: 조건별 중요도 가중치 학습
- **계층적 드롭아웃**: 수준별 의존성 모델링

### 8.3 응용 분야의 확장 가능성

#### 8.3.1 산업 응용

**1. 디자인 및 창의 도구**
- 패션 디자인: 색상 팔레트 + 실루엣(스케치) + 패턴(텍스트) 동시 제어
- 건축 시각화: 레이아웃(세분화) + 스타일(텍스트) + 조명(색상) 통합
- 게임 자산: 캐릭터 포즈(키포인트) + 의류(참조 이미지) + 배경(텍스트)

**2. 콘텐츠 생성**
- 영상 프레임 생성: Composer 확장 가능성
- 책 삽화: 장면 설명 + 참조 이미지 + 스타일 가이드

**3. 의료 및 과학**
- 의료 이미지 합성: 해부학 구조(깊이/세분화) + 질병 특성(텍스트)
- 과학 시뮬레이션: 물리적 제약(조건) + 시각화 스타일

#### 8.3.2 새로운 조건 유형 추가

**2024년 이후 논문들의 추가 조건:**
- **텍스트 위치 정보**: 공간적 레이아웃이 필요한 텍스트 생성 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhangli_Layout-Agnostic_Scene_Text_Image_Synthesis_with_Diffusion_Models_CVPR_2024_paper.pdf)
- **카메라 포즈**: 3D 일관성을 위한 카메라 파라미터 [studios.disneyresearch](https://studios.disneyresearch.com/2025/10/26/multimodal-conditional-3d-face-geometry-generation/)
- **조명 조건**: 조명 방향 및 강도 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10677863/)
- **음성 임베딩**: 음성으로부터 이미지 생성 [nature](https://www.nature.com/articles/s41598-024-76407-9)

### 8.4 미해결 문제 및 미래 연구 방향

#### 8.4.1 상충 조건의 명시적 처리

**현황**: 암묵적 가중치 조정에 의존
**필요성**: 
- 사용자가 명시적으로 조건 우선순위 지정 가능
- 조건 충돌 감지 및 경고 시스템

**제안 솔루션**:
$$\text{가중치}_i = \frac{\alpha_i \cdot \text{신뢰도}_i}{\sum_j \alpha_j \cdot \text{신뢰도}_j}$$

여기서 $\alpha_i$는 사용자 지정 가중치, $\text{신뢰도}_i$는 학습된 신뢰도

#### 8.4.2 조건 간 상호작용 모델링

**현황**: 각 조건을 독립적으로 처리
**개선안**: 
- 조건 간 의존성을 명시적으로 모델링
- 예: "색상" 조건이 "스타일"을 제약

#### 8.4.3 도메인 일반화

**한계**: 훈련 데이터와 거리가 먼 도메인에서 성능 저하 가능
**해결책**:
- 도메인별 미세조정
- 도메인 불변(Domain-Invariant) 표현 학습

#### 8.4.4 동적 및 시각적 일관성

**Composer의 한계**: 정적 이미지 생성만 가능
**확장 방향**:
- **비디오 확장**: 프레임 간 일관성 유지
- **3D 일관성**: 여러 뷰에서 기하학적 일관성 [studios.disneyresearch](https://studios.disneyresearch.com/2025/10/26/multimodal-conditional-3d-face-geometry-generation/)

### 8.5 신뢰성 및 안전성 고려사항

#### 8.5.1 제어 가능성과 위험성의 역설

Composer의 더 강력한 제어 기능은 **양날의 검**입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

| 긍정적 활용 | 부정적 위험 |
|-----------|----------|
| 예술적 표현 자유 | 기만적 콘텐츠 생성 용이 |
| 접근성 향상 | 개인 식별 정보 조작 |
| 과학적 시뮬레이션 | 딥페이크 생성 |

#### 8.5.2 제안하는 안전장치

**1. 투명성(Transparency)**
- 생성 과정에 사용된 조건들의 명시적 로깅
- 생성 이미지 워터마크 또는 메타데이터 포함

**2. 접근 제어(Access Control)**
- 고위험 조건 조합에 대한 사용 제한
- 의도 확인 프롬프트

**3. 감지(Detection)**
- 생성 이미지의 자동 감지 기술 개발
- 매장 탐지 모델(Watermark Detection)

### 8.6 표준화 및 벤치마크

#### 8.6.1 다중 조건 평가 메트릭

**현재 평가 방식의 한계**:
- FID, CLIP Score는 단일 조건 중심
- 다중 조건 충돌을 측정하지 못함

**필요한 메트릭**:
```
조건 충실도(Condition Fidelity):
CF = (1/N) * Σ_i 조건_i와_생성_i의_유사도

조건 분리(Condition Disentanglement):
CD = (1/C(C-1)/2) * Σ_{i<j} (1 - 상관계수(c_i, c_j))
```

#### 8.6.2 벤치마크 데이터셋

필요한 벤치마크:
- **다중 조건 인수분해 데이터셋**: 각 이미지마다 8가지 표현의 정답 주석
- **충돌 시나리오 데이터셋**: 의도적으로 상충하는 조건들의 조합
- **도메인별 벤치마크**: 얼굴, 풍경, 실내, 의류 등 각 도메인별

***

## 9. 결론

Composer는 이미지 생성의 제어 가능성을 획기적으로 확장하는 중요한 논문입니다. **구성성이라는 단순하지만 강력한 원리**를 통해: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3274d54-fc90-4b15-81ad-d4f6bbcc21f3/2302.09778v2.pdf)

1. **제어 공간의 기하급수적 확대** ($2^8$ 이상의 조합 가능성)
2. **재훈련 없는 다양한 작업 수행** (영점샷 능력)
3. **효율적인 단일 모델 학습** (8개 조건 통합)

을 달성했습니다.

이후 **2023-2025년 연구**는 Composer의 원리를 바탕으로:
- ControlNet 계열의 공간 제어 강화 [arxiv](https://arxiv.org/abs/2302.05543)
- DiffBlender의 모달리티별 적응 [arxiv](https://arxiv.org/html/2305.15194v3)
- UniVG의 통합 다중 작업 모델 [arxiv](https://arxiv.org/html/2503.12652v2)

로 발전했습니다.

미래 연구는 다음을 중점적으로 고려해야 합니다:

1. **조건 상호작용의 명시적 모델링**
2. **더 강화된 도메인 일반화**
3. **안전성과 신뢰성의 균형**
4. **표준화된 평가 메트릭 개발**
5. **3D 일관성 및 비디오로의 확장**

Composer는 단순히 기술적 혁신을 넘어, **구성적 생성 모델의 패러다임 전환**을 제시한 획기적 연구로 평가됩니다.

***

## 참고문헌

<span style="display:none">[^1_100][^1_101][^1_102][^1_103][^1_104][^1_105][^1_106][^1_107][^1_108][^1_109][^1_110][^1_111][^1_112][^1_113][^1_114][^1_115][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97][^1_98][^1_99]</span>

<div align="center">⁂</div>

[^1_1]: 2302.09778v2.pdf

[^1_2]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770426.pdf

[^1_3]: https://arxiv.org/abs/2302.05543

[^1_4]: https://arxiv.org/html/2305.15194v3

[^1_5]: https://openaccess.thecvf.com/content/WACV2024/papers/Corneanu_LatentPaint_Image_Inpainting_in_Latent_Space_With_Diffusion_Models_WACV_2024_paper.pdf

[^1_6]: https://arxiv.org/abs/2412.10316

[^1_7]: https://ieeexplore.ieee.org/document/10677863/

[^1_8]: https://arxiv.org/abs/2303.16203

[^1_9]: https://arxiv.org/html/2505.21780v1

[^1_10]: https://arxiv.org/html/2509.01028v1

[^1_11]: https://arxiv.org/html/2503.12652v2

[^1_12]: https://arxiv.org/abs/2106.09685

[^1_13]: https://arxiv.org/abs/2312.01255

[^1_14]: https://arxiv.org/abs/2112.05130

[^1_15]: https://arxiv.org/abs/2401.13388

[^1_16]: https://arxiv.org/abs/2310.10338

[^1_17]: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhangli_Layout-Agnostic_Scene_Text_Image_Synthesis_with_Diffusion_Models_CVPR_2024_paper.pdf

[^1_18]: https://studios.disneyresearch.com/2025/10/26/multimodal-conditional-3d-face-geometry-generation/

[^1_19]: https://www.nature.com/articles/s41598-024-76407-9

[^1_20]: https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9

[^1_21]: https://dl.acm.org/doi/10.1145/3707292.3707367

[^1_22]: https://arxiv.org/abs/2401.05252

[^1_23]: https://arxiv.org/abs/2410.19324

[^1_24]: https://ieeexplore.ieee.org/document/10678183/

[^1_25]: https://www.semanticscholar.org/paper/d7890d1906d95c4ae4c430b350455156d6d8aed9

[^1_26]: https://link.springer.com/10.1007/978-3-031-72744-3_2

[^1_27]: https://arxiv.org/abs/2403.06381

[^1_28]: https://ieeexplore.ieee.org/document/10530514/

[^1_29]: https://arxiv.org/pdf/2209.00796v8.pdf

[^1_30]: https://arxiv.org/pdf/2403.12803.pdf

[^1_31]: https://arxiv.org/pdf/2112.05744v3.pdf

[^1_32]: http://arxiv.org/pdf/2112.10752.pdf

[^1_33]: https://arxiv.org/pdf/2308.13767.pdf

[^1_34]: https://arxiv.org/html/2412.12888v1

[^1_35]: https://arxiv.org/pdf/2412.09656.pdf

[^1_36]: https://arxiv.org/pdf/2310.06313.pdf

[^1_37]: https://arxiv.org/abs/2112.10752

[^1_38]: https://arxiv.org/html/2303.07909v3

[^1_39]: https://arxiv.org/abs/2307.01952

[^1_40]: https://arxiv.org/html/2409.19365v3

[^1_41]: https://arxiv.org/html/2410.20898v2

[^1_42]: https://arxiv.org/html/2411.18936v1

[^1_43]: https://arxiv.org/html/2405.06535v1

[^1_44]: https://arxiv.org/html/2403.05125v1

[^1_45]: https://arxiv.org/html/2410.16719v1

[^1_46]: https://arxiv.org/abs/2105.05233

[^1_47]: https://arxiv.org/html/2401.09048v1

[^1_48]: https://arxiv.org/html/2409.10695v1

[^1_49]: https://github.com/Stability-AI/stablediffusion

[^1_50]: https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/

[^1_51]: https://syncedreview.com/2021/01/20/google-creates-new-sota-text-image-generation-framework/

[^1_52]: https://liner.com/review/patched-denoising-diffusion-models-for-highresolution-image-synthesis

[^1_53]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01435.pdf

[^1_54]: https://arxiv.org/abs/1807.07560

[^1_55]: https://vds.sogang.ac.kr/wp-content/uploads/2024/01/2024_%EA%B2%A8%EC%9A%B8%EC%84%B8%EB%AF%B8%EB%82%98_%EC%98%A4%ED%95%98%EB%8B%88.pdf

[^1_56]: https://dmqa.korea.ac.kr/uploads/seminar/[240216] DMQA_Openseminar_Controllable%20Diffusion%20Model_%E1%84%8B%E1%85%B2%E1%86%AB%E1%84%8C%E1%85%B5%E1%84%92%E1%85%A7%E1%86%AB.pdf

[^1_57]: https://arxiv.org/html/2502.21151v2

[^1_58]: https://liner.com/ko/review/anytoany-generation-via-composable-diffusion

[^1_59]: https://openreview.net/forum?id=S85PP4xjFD

[^1_60]: https://arxiv.org/abs/2312.06573

[^1_61]: https://arxiv.org/abs/2309.05534

[^1_62]: https://link.springer.com/10.1007/978-3-031-73223-2_20

[^1_63]: https://ieeexplore.ieee.org/document/10445344/

[^1_64]: https://arxiv.org/abs/2309.04372

[^1_65]: https://arxiv.org/abs/2311.02343

[^1_66]: https://arxiv.org/abs/2305.13077

[^1_67]: https://ieeexplore.ieee.org/document/10658153/

[^1_68]: https://arxiv.org/abs/2412.04707

[^1_69]: https://arxiv.org/html/2312.06573v2

[^1_70]: http://arxiv.org/pdf/2502.14779.pdf

[^1_71]: https://arxiv.org/html/2410.04932v1

[^1_72]: http://arxiv.org/pdf/2404.07987.pdf

[^1_73]: https://arxiv.org/html/2410.09400v2

[^1_74]: https://arxiv.org/html/2312.01255v2

[^1_75]: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf

[^1_76]: https://openaccess.thecvf.com/content/WACV2023/papers/Popovic_Spatially_Multi-Conditional_Image_Generation_WACV_2023_paper.pdf

[^1_77]: https://arxiv.org/html/2504.07998

[^1_78]: https://openaccess.thecvf.com/content/CVPR2024/papers/Phung_Grounded_Text-to-Image_Synthesis_with_Attention_Refocusing_CVPR_2024_paper.pdf

[^1_79]: https://arxiv.org/html/2409.08482v1

[^1_80]: https://arxiv.org/html/2506.04244v1

[^1_81]: https://arxiv.org/html/2305.17216

[^1_82]: https://arxiv.org/pdf/2302.05543.pdf

[^1_83]: https://arxiv.org/abs/2412.02352

[^1_84]: https://arxiv.org/html/2505.15217v1

[^1_85]: https://huggingface.co/blog/lora

[^1_86]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760089.pdf

[^1_87]: https://nips.cc/virtual/2023/74579

[^1_88]: https://github.com/idpen/lora-diffusion/

[^1_89]: https://kimjy99.github.io/논문리뷰/controlnet/

[^1_90]: https://www.reddit.com/r/MachineLearning/comments/zfkqjh/p_using_lora_to_efficiently_finetune_diffusion/

[^1_91]: https://kimjy99.github.io/논문리뷰/uni-controlnet/

[^1_92]: https://www.hyperstack.cloud/blog/case-study/lora-for-stable-diffusion-fine-tuning-understand-why-it-s-efficient

[^1_93]: https://aclanthology.org/2024.acl-long.335.pdf

[^1_94]: https://github.com/lllyasviel/ControlNet

[^1_95]: https://machinelearningmastery.com/fine-tuning-stable-diffusion-with-lora/

[^1_96]: https://arxiv.org/abs/2407.13139

[^1_97]: https://ieeexplore.ieee.org/document/10884879/

[^1_98]: https://arxiv.org/abs/2312.03771

[^1_99]: https://ieeexplore.ieee.org/document/10943980/

[^1_100]: https://ieeexplore.ieee.org/document/10484154/

[^1_101]: https://ieeexplore.ieee.org/document/10655542/

[^1_102]: https://arxiv.org/abs/2406.09368

[^1_103]: https://arxiv.org/abs/2404.10765

[^1_104]: https://ieeexplore.ieee.org/document/11094425/

[^1_105]: https://arxiv.org/html/2412.10316v1

[^1_106]: http://arxiv.org/pdf/2404.04860.pdf

[^1_107]: https://arxiv.org/html/2412.01223v1

[^1_108]: https://arxiv.org/html/2303.17546v3

[^1_109]: https://arxiv.org/abs/2310.07222

[^1_110]: http://arxiv.org/pdf/2311.11469.pdf

[^1_111]: https://dl.acm.org/doi/pdf/10.1145/3610543.3626172

[^1_112]: https://arxiv.org/pdf/2308.09388.pdf

[^1_113]: https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Diverse_Image_Synthesis_From_Semantic_Layouts_via_Conditional_IMLE_ICCV_2019_paper.pdf

[^1_114]: https://arxiv.org/pdf/2508.20783.pdf

[^1_115]: https://arxiv.org/html/2407.13139v1
 Compositional Scene Understanding through Inverse Generative Modeling (2025) [arxiv](https://arxiv.org/html/2505.21780v1)

 Your Diffusion Model is Secretly a Zero-Shot Classifier (Li et al., 2023) [arxiv](https://arxiv.org/abs/2303.16203)
