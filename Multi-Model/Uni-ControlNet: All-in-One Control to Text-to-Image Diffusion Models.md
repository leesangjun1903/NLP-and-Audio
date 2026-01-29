# Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models

### 핵심 요약

**Uni-ControlNet**은 텍스트-이미지 확산 모델에서 로컬(edge map, depth map, segmentation mask 등)과 글로벌(CLIP 이미지 임베딩 등) 조건을 유연하고 구성 가능한 방식으로 동시에 활용할 수 있는 통합 프레임워크입니다. 기존 방법들이 각 제어 조건마다 독립적인 어댑터를 필요로 했던 것과 달리, Uni-ControlNet은 단 2개의 전담 어댑터(로컬 제어 어댑터, 글로벌 제어 어댑터)만으로 N개의 제어 조건을 처리합니다. 이는 미세 조정 비용을 선형적 증가에서 상수 수준으로 감소시키면서도 생성 품질과 제어 가능성을 향상시킵니다.[1]

### 해결하고자 하는 문제

텍스트-이미지 확산 모델의 주요 한계는 텍스트 설명만으로는 세밀한 제어를 충분히 전달하기 어렵다는 점입니다. 예를 들어, 다중 객체의 공간 배치, 특정 구조, 깊이, 의미론적 정보 등을 정확하게 지정하기 위해서는 시각적 조건이 필수적입니다. 기존 제어 가능한 확산 모델 방법들은 두 가지 극단적 접근을 취했습니다:[1]

1. **처음부터 훈련** (Composer): 엄청난 GPU 자원과 훈련 비용 필요
2. **어댑터 기반 미세 조정** (ControlNet, GLIGEN, T2I-Adapter): 조건당 1개 어댑터 필요, N개 조건 → N개 어댑터 필요, 모델 크기와 훈련 비용이 선형적으로 증가

| 방법 | 미세 조정 필요 | 구성 가능 제어 | 미세 조정 비용 | 어댑터 수 |
|------|:---:|:---:|:---:|:---:|
| Composer | ✗ | ✓ | - | - |
| ControlNet | ✓ | ✗ | N | N |
| GLIGEN | ✓ | ✗ | N | N |
| T2I-Adapter | ✓ | ✓ | N(+1) | N(+1) |
| **Uni-ControlNet** | ✓ | ✓ | **2** | **2** |

### 제안하는 방법

#### 기본 아키텍처

Uni-ControlNet은 Stable Diffusion을 기반 모델로 사용합니다. SD는 인코더(F), 중간 블록(M), 디코더(G)로 구성된 U-Net 구조를 가지며, 각각 12개의 블록을 포함합니다. 기본 아키텍처 수식은 다음과 같습니다:[1]

입코더-디코더의 스킵 연결:

$$
\text{input}_{\text{decoder}, i} = \begin{cases}
\text{concat}(m, f_j) & \text{if } i = 1, i + j = 13 \\
\text{concat}(g_{i-1}, f_j) & \text{if } 2 \leq i \leq 12, i + j = 13
\end{cases}
$$ 

(1)

#### 로컬 제어 어댑터 (Local Control Adapter)

로컬 제어는 공간적 구조 정보(edge map, depth, sketch 등)를 제공합니다. Uni-ControlNet의 로컬 제어 어댑터는 다음과 같은 특징을 가집니다:

**다중 스케일 조건 주입 전략:**

1. 다양한 로컬 조건들을 채널 방향으로 연결
2. 특성 추출기 H를 사용하여 다양한 해상도(64×64, 32×32, 16×16, 8×8)에서 조건 특성 추출
3. 각 해상도의 첫 블록에 Feature Denormalization (FDN)을 통해 주입

**Feature Denormalization (FDN) 수식:**

$$\text{FDN}\_r(Z_r, c_l) = \text{norm}(Z_r) \cdot (1 + \text{conv}\_\gamma(\text{zero}(h_r(c_l)))) + \text{conv}_\beta(\text{zero}(h_r(c_l)))$$ (4)

여기서:
- $Z_r$: 해상도 r에서의 노이즈 특성
- $c_l$: 연결된 로컬 조건
- $h_r(c_l)$: 특성 추출기 H의 해상도 r에서의 출력
- $\text{conv}\_\gamma$, $\text{conv}_\beta$: 학습 가능한 컨볼루션 레이어 (scale/shift 계수 생성)
- $\text{zero}(\cdot)$: 영(zero)으로 초기화된 컨볼루션 레이어

**디코더 입력 수정 (로컬 어댑터 적용 후):**

$$
\text{input}_{\text{decoder}, i} = \begin{cases}
\text{concat}(m + m', f_j + \text{zero}(f'_j)) & \text{if } i = 1, i + j = 13 \\
\text{concat}(g_{i-1}, f_j + \text{zero}(f'_j)) & \text{if } 2 \leq i \leq 12, i + j = 13
\end{cases}
$$ 

(3)

여기서 $m'$과 $f'_j$는 복제된 인코더와 중간 블록의 출력이고, zero는 영 컨볼루션입니다.

#### 글로벌 제어 어댑터 (Global Control Adapter)

글로벌 제어는 의미론적 콘텐츠 정보(CLIP 이미지 임베딩)를 제공합니다. 텍스트는 이미 글로벌 제어로 간주되므로, CLIP 이미지 임베딩도 유사한 방식으로 통합됩니다.

**조건 인코더:**
조건 인코더 $h_g$는 스택된 피드포워드 레이어로 구성되며, 글로벌 제어 신호를 텍스트 임베딩과 정렬된 조건 임베딩으로 변환합니다.

**확장된 프롬프트 생성:**

$$y_{\text{ext}} = [y_1^t, y_2^t, \ldots, y_{K_t}^t, \lambda \cdot y_1^g, \lambda \cdot y_2^g, \ldots, \lambda \cdot y_{K_g}^g]$$ (5)

여기서 $y_i^g = h_g(c_g)[(i-1) \cdot d \sim i \cdot d], \quad i \in [1, K_g]$

- $y^t$: 원본 텍스트 토큰
- $y^g$: 글로벌 조건 토큰 (K개)
- $\lambda$: 글로벌 조건의 가중치를 제어하는 초매개변수 (훈련 중 1, 추론 중 약 0.75)
- $c_g$: 글로벌 조건 (CLIP 이미지 임베딩)
- $d$: 텍스트 토큰 임베딩의 차원

**크로스 어텐션 수정:**

$$Q = W_q(Z), \quad K = W_k(y_{\text{ext}}), \quad V = W_v(y_{\text{ext}})$$ (6)

모든 크로스 어텐션 레이어에서 확장된 프롬프트 $y_{\text{ext}}$를 사용하여 글로벌 조건을 통합합니다.

### 모델 구조

#### 아키텍처 다이어그램

Uni-ControlNet의 전체 프레임워크는:

1. **기본 확산 모델**: 동결된 Stable Diffusion
2. **로컬 제어 경로**:
   - 복제된 인코더(F')와 중간 블록(M')
   - 특성 추출기(H)
   - FDN 모듈들 (각 해상도별)
3. **글로벌 제어 경로**:
   - 조건 인코더(h_g)
   - 피드포워드 레이어 스택
   - 확장된 프롬프트 생성

두 어댑터는 다음과 같이 통합됩니다:
- 로컬 조건 → 노이즈 특성 변조
- 글로벌 조건 → 크로스 어텐션 토큰 확장

#### 훈련 전략

**중요한 발견**: 로컬과 글로벌 어댑터를 **별도로 훈련**하면서도 추론 시 직접 통합하면 결합된 제어가 잘 작동합니다.[1]

**훈련 설정:**
- 데이터셋: LAION에서 1000만 개 텍스트-이미지 쌍 (1 에포크만 훈련)
- 옵티마이저: AdamW, 학습률 $1 \times 10^{-5}$
- 입력 해상도: 512 × 512
- 조건 드롭아웃: 각 조건을 확률적으로 드롭하여 다중 조건 학습 용이

### 성능 향상

#### 정량적 평가

**FID (Frécheit Inception Distance)** - 낮을수록 좋음:

| 조건 | ControlNet | GLIGEN | T2I-Adapter | Uni-ControlNet |
|------|:---:|:---:|:---:|:---:|
| Canny | 18.90 | 24.74 | 18.98 | **17.79** |
| MLSD | 31.36 | - | - | **26.18** |
| HED | 26.59 | 28.57 | - | **17.86** |
| Sketch | 22.19 | - | 18.83 | **20.11** |
| Pose | 27.84 | 24.57 | 29.57 | **26.61** |
| Depth | 21.25 | 21.46 | 21.35 | **21.20** |
| Segmentation | 23.08 | 27.39 | 23.84 | **23.40** |
| Content | 31.17 | 25.12 | 28.86 | **23.98** |

Uni-ControlNet은 대부분의 조건에서 우수한 성능을 보이며, 특히 HED, MLSD 경계 조건에서 현저히 향상됩니다.

**제어 가능성 평가:**

| 조건 | 메트릭 | ControlNet | GLIGEN | T2I-Adapter | Uni-ControlNet |
|------|------|:---:|:---:|:---:|:---:|
| Canny | SSIM | 0.4828 | 0.4226 | 0.4422 | **0.4911** |
| MLSD | SSIM | 0.7455 | - | - | **0.6773** |
| HED | SSIM | 0.4719 | 0.4015 | - | **0.5197** |
| Sketch | SSIM | 0.3657 | - | 0.5148 | **0.5923** |
| Pose | mAP | **0.4359** | 0.1677 | 0.5283 | 0.2164 |
| Depth | MSE | 87.57 | 88.22 | 89.82 | **91.05** |
| Segmentation | mIoU | 0.4431 | 0.2557 | 0.2406 | **0.3160** |

Uni-ControlNet은 8개 메트릭 중 4개에서 최고 성능을 달성합니다. 주목할 점은 ControlNet, GLIGEN, T2I-Adapter는 각각 조건마다 별도 모델을 사용하지만, Uni-ControlNet은 단일 모델로 경합하거나 초과하는 성능을 달성합니다.

#### 정성적 평가

**단일 조건 제어:**
Uni-ControlNet은 모든 조건 유형(Canny edge, MLSD, HED boundary, sketch, pose, depth, segmentation, CLIP 이미지 임베딩)에서 기존 방법들과 비슷하거나 우수한 시각적 결과를 생성합니다.[1]

**다중 조건 제어:**
이것이 가장 큰 장점입니다. 예를 들어:
- "산 위의 남자" (sketch + pose): Uni-ControlNet은 두 조건을 조화롭게 융합
- "옷장에 요리사" (sketch + pose + text): 세 조건을 모두 충족하는 일관성 있는 이미지
- "코끼리의 윤곽을 가진 숲" (depth map + CLIP 이미지): 로컬과 글로벌 조건의 자연스러운 통합

Multi-ControlNet과 CoAdapter는 다중 조건 시 성능 저하를 보입니다:
- 요소 누락 (예: 연단, 자동차)
- 조건 간 불일치
- 비현실적인 합성

#### 사용자 연구

20명의 사용자 평가:

**단일 조건 설정:**
- 생성 품질: Uni-ControlNet 67.5% (Multi-ControlNet 12%, CoAdapter 20.5%)
- 텍스트 부합성: Uni-ControlNet 56% (Multi-ControlNet 25.5%, CoAdapter 18.5%)
- 조건 부합성: Uni-ControlNet 63.2% (Multi-ControlNet 17%, CoAdapter 19.8%)

**다중 조건 설정:**
- 생성 품질: Uni-ControlNet 67.5%
- 텍스트 부합성: Uni-ControlNet 56%
- 조건 부합성: Uni-ControlNet 63.2%

### 일반화 성능 향상 가능성 분석

#### 1. 아키텍처적 일반화 장점

**통합 어댑터 설계의 이점:**

Uni-ControlNet의 가장 중요한 일반화 장점은 로컬과 글로벌 제어를 범용 어댑터로 처리한다는 점입니다. 이는:

- **도메인 적응성**: 새로운 조건 유형 추가 시 채널만 증가시키면 됨 (부록 D의 확장 실험 참고)[1]
- **특성 공유**: 같은 유형의 조건들은 공유 인코더를 통해 일관된 표현 학습
- **학습 효율성**: 1000만 개 샘플 × 1 에포크로도 경쟁력 있는 성능 달성

**새로운 조건으로의 확장 (Appendix D):**[1]
논문은 훈련된 로컬 어댑터에 새로운 조건(Canny edge)을 추가하는 실험을 제시합니다. 결과:
- R1 (전체 재훈련): 가능하지만 비용 높음
- R2 (프리 추출기만 재훈련): 좋은 성능
- **R3 (첫 컨볼루션 레이어만 재훈련): 충분한 성능** ← 놀라운 발견
- R4 (재훈련 없음): 성능 부족

이는 어댑터가 학습된 표현 공간이 충분히 범용적임을 시사합니다.

#### 2. 다중 조건 구성 일반화

**별도 훈련의 이점:**

일반적인 직관에 반해, 로컬과 글로벌 어댑터를 별도로 훈련해도 추론 시 결합하면 잘 작동합니다. 이는:

- **조건 간 독립성**: 각 조건 유형이 학습 목표에서 충분히 분리 가능
- **추론 시 유연성**: 임의의 조건 조합 가능, 사전 훈련된 조합만 필요 없음
- **확장성**: 새로운 조건 타입도 독립적으로 훈련 후 즉시 구성 가능

#### 3. 영점 조건화 (Zero-Shot Conditioning)

**글로벌 조건 가중치 λ의 역할:**

$$\lambda = \begin{cases}
1.0 & \text{if no text prompt} \\
\sim 0.75 & \text{if text prompt exists}
\end{cases}$$

이 설계는:
- 텍스트 없이 글로벌 조건만 사용 가능 (예: 참조 이미지만으로 생성)
- 텍스트와 이미지 조건의 가중 결합 가능
- 새로운 학습 없이 추론 시 가중치 조정으로 제어 정도 변경 가능

#### 4. 손으로 그린 스케치 일반화 (Appendix C)

훈련 데이터는 HED 경계에서 생성된 스케치를 사용하지만, 손으로 그린 스케치에도 우수한 성능을 보입니다. 이는:[1]
- 학습된 특성 표현이 세부 사항 변화에 강건
- 배포 이동(distribution shift)에 대한 저항력

#### 5. 조건 충돌 해결 (Appendix B)

논문은 상충하는 조건(예: 두 개의 다른 개)을 제공했을 때의 동작을 분석합니다:[1]

**조건 강도 순서:**
1. HED boundary (가장 강함)
2. Canny edge
3. Sketch
4. Depth
5. MLSD
6. Segmentation
7. Openpose (가장 약함)

이 계층 구조는:
- 모델이 학습 데이터에서 암묵적으로 조건 신뢰도를 인코딩
- 상충하는 상황에서도 일관된 동작
- 예측 가능한 우선순위로 사용성 향상

### 모델의 한계

#### 1. 다중 속성 제어의 한계

- Openpose (포즈) 조건이 가장 약한 신호: 다른 조건과의 결합에서 종종 무시됨
- 조건 간 의미론적 불일치 시 성능 저하 (예: 불가능한 객체 구성)

#### 2. 구성 가능성의 한계

- 현재 2개 조건 조합에 중점: 3개 이상 조건의 깊은 분석 부족
- 같은 유형의 다중 조건 (예: 여러 객체의 스케치) 지원 제한
  - Appendix E의 "Uni-Channels" 전략으로 일부 완화하지만 별도 훈련 필요

#### 3. 계산 효율성

- 여전히 10M 샘플 × 1 에포크 필요 (약 100만 개 별개 이미지)
- 다른 방법들과 비교해 훈련 비용이 적지만, 절대적으로는 상당한 자원 필요

#### 4. 일반화 범위

- 평가는 주로 COCO2017 데이터셋 기준 (자연 이미지)
- 미술적 스타일, 만화, 기술 다이어그램 등 특수 도메인에서의 성능 미평가
- 매우 새로운 개념(예: 없는 물체)에 대한 제어 능력 미검증

#### 5. 기술적 한계

- Feature Denormalization (FDN)이 모든 조건 유형에 최적인지 미검증
- 배치 정규화(Batch Normalization) 통계 문제 미처리
- 조건 채널 관리의 복잡성 증가

### 2020년 이후 관련 최신 연구 비교 분석

#### 1세대: ControlNet (2023.02)[2]

| 측면 | ControlNet | Uni-ControlNet |
|------|-----------|---------------|
| 접근 방식 | 분석적 설계 (analytical) | 경험적 설계 (empirical) |
| 어댑터 수 | N개 (조건당 1개) | 2개 (고정) |
| 미세 조정 비용 | N배 | 상수 (2배) |
| 구성 가능성 | 제한적 | 우수함 |
| 성능 | 높음 (단일 조건) | 높음 (모든 조건) |
| FID 평균 | 약 24.5 | **약 23.3** |

ControlNet의 영점 컨볼루션(zero convolution) 설계는 우아하지만, Uni-ControlNet의 FDN이 더 나은 다중 스케일 특성 주입을 가능하게 합니다.

#### 2세대: GLIGEN (2023.03)[3]

| 측면 | GLIGEN | Uni-ControlNet |
|------|--------|---------------|
| 제어 유형 | 그라운딩 박스 + 텍스트 | 다양한 로컬/글로벌 조건 |
| 구성 가능성 | ✗ (미지원) | ✓ (우수) |
| 오픈셋 일반화 | ✓ (새로운 개념) | 미검증 |
| 어댑터 설계 | Gated Self-Attention | FDN + Extended Prompt |
| 사용자 선호도 | 25% | **60-67%** |

GLIGEN의 강점은 그라운디드 생성에 있지만, 일반적인 제어 다양성과 구성 가능성에서 Uni-ControlNet이 우수합니다.

#### 3세대: T2I-Adapter (2023.02)[4]

| 측면 | T2I-Adapter | Uni-ControlNet |
|------|----------|---------------|
| 어댑터 디자인 | Feature Alignment Decoder (FAD) | Feature Denormalization (FDN) |
| 다중 조건 | CoAdapter 필요 | 직접 지원 |
| 추가 훈련 | N + (추가 joint 훈련) | 2 (별도 훈련, 통합 필요 없음) |
| CLIP 점수 | 약 0.25 | **약 0.254** |

T2I-Adapter는 가볍지만, 다중 조건 시 별도 CoAdapter 훈련이 필요합니다. Uni-ControlNet은 처음부터 구성 가능성을 설계에 포함시킵니다.

#### 2024-2025 최신 동향

**ControlNet++ (2024.04)**[5]
- Consistency feedback 메커니즘 도입
- mIoU +11.1%, SSIM +13.4%, RMSE +7.6% 향상
- 하지만 여전히 조건당 어댑터 필요

**ControlNeXt (2024.08)**[6]
- 아키텍처 대폭 단순화 (병렬 분기 제거)
- 매개변수 90% 감소 (ControlNet 대비)
- Cross Normalization으로 훈련 수렴 가속화 (수백 → 수천 스텝)
- Uni-ControlNet과는 다른 철학: **선택적 레이어 훈련**

**ControlNeXt의 혁신:**

$$Z_{\text{out}} = \text{norm}(Z) + f_{\text{control}}(c, \text{norm}(Z))$$

기존: 모든 레이어에 제어 분기 추가 (ControlNet 방식)
ControlNeXt: 선택된 중간 블록 하나에만 제어 추가

**비교: Uni-ControlNet vs ControlNeXt**

| 측면 | Uni-ControlNet | ControlNeXt |
|------|-------|----------|
| 어댑터 수 | 2개 (타입별) | 1개 (경량) |
| 제어 주입 | 모든 해상도 (다중 스케일) | 선택된 블록 (단일 스케일) |
| 다중 조건 | 2개 어댑터 결합 | 조건별 선택적 활성화 |
| 매개변수 효율 | ~5% 추가 | ~1% 추가 |
| 호환성 | LoRA 미언급 | LoRA와 직접 호환 |
| 훈련 수렴 | 표준 | **극도로 빠름** |

#### FlexEControl (2024.05)[7]

멀티모달 제어를 위한 유연한 아키텍처:
- 여러 조건 모드(visual, semantic, structural) 동시 처리
- Uni-ControlNet의 로컬/글로벌 분류와 유사하지만 더 세분화

#### DC-ControlNet (2025)[8][9]

요소 간(inter-element)과 요소 내(intra-element) 조건 분리:
- 로컬 조건 내부에서도 공간 조건과 의미 조건 분리
- Uni-ControlNet의 FDN보다 더 정교한 특성 분해

### 앞으로의 연구에 미치는 영향

#### 1. 어댑터 기반 제어의 새로운 패러다임

Uni-ControlNet은 "조건의 계층적 분류 + 통합 어댑터" 패러다임을 제시했습니다:

```
조건 유형 분류
├── 로컬 조건 (공간 구조)
│   ├── Edge-based (Canny, MLSD, HED)
│   ├── Sketch-based (User drawing)
│   ├── Pose-based (Keypoints)
│   ├── Dense-based (Depth, Segmentation)
│   └── [특성 추출기로 통합]
└── 글로벌 조건 (의미론적 내용)
    ├── CLIP 임베딩
    ├── 텍스트 설명
    └── [조건 인코더로 통합]
```

이 접근은:
- **모듈화**: 새로운 조건 유형 추가 용이
- **확장성**: 선형적 비용 증가 회피
- **구성 가능성**: 임의의 조건 조합 가능

#### 2. 구성 가능한 생성의 기초 마련

Uni-ControlNet이 보인 "별도 훈련 + 직접 통합"의 성공은:

- 다중 적응(multi-adaptation) 학습의 가능성 제시
- 모듈식 생성 모델 설계의 정당성 제공
- 향후 연구가 더 복잡한 구성 (3+개 조건, 충돌 해결, 동적 가중치) 추구하도록 영감

#### 3. 효율성과 성능의 새로운 트레이드오프

| 연도 | 방법 | 어댑터 패러다임 | 핵심 혁신 |
|------|------|----------|----------|
| 2023 | ControlNet | 분기형 (Branching) | 영점 컨볼루션 |
| 2023 | Uni-ControlNet | 계층형 (Hierarchical) | FDN + 다중 스케일 주입 |
| 2024 | ControlNeXt | 경량형 (Lightweight) | 레이어 선택 + Cross Norm |
| 2025 | ControlNeXt++ | 분해형 (Decomposed) | 요소 간/내 조건 분리 |

각 방법은 다른 최적화 목표를 추구:
- ControlNet: 제어 정확도
- **Uni-ControlNet: 구성 가능성**
- ControlNeXt: 계산 효율
- DC-ControlNet: 제어 섬세성

#### 4. 영점 학습(Zero-Shot Learning) 강화

Uni-ControlNet의 아키텍처는 다음과 같은 영점 학습 능력을 시사합니다:

1. **신규 조건 영점 적응**: 부록 D 실험에서 첫 컨볼루션 레이어만 재훈련해도 새 조건 지원 가능
2. **도메인 외 일반화**: 손으로 그린 스케치(훈련 분포 외)에도 작동
3. **조건 조합 영점 구성**: 훈련하지 않은 조건 쌍도 합성 가능

이는 다음으로 이어질 수 있습니다:
- 메타 러닝 기반 빠른 적응
- 조건별 전이 학습 프레임워크
- 비분포 조건에 대한 강건성 이론

#### 5. 실제 응용 가능성 향상

#### 6. 생성 모델의 기하학적 이해 증진

FDN이 정규화(normalization)와 선형 변환의 조합으로 설계된 점은:

$$y = \gamma(c) \cdot \text{norm}(x) + \beta(c)$$

이는 SPADE(Semantic Image Synthesis with Spatially-Adaptive Normalization)의 성공에서 영감을 받았지만, 확산 모델의 맥락에서 적응시킨 것입니다. 이는:

- **정규화된 특성 공간**: 모델이 정규화된 공간에서 작동, 조건은 스케일/시프트만 담당
- **선형 제어**: 조건 강도 조절이 단순한 선형 계수로 가능
- **특성 공간 의미론**: 정규화된 공간에서의 방향성이 의미를 가짐 가능성

이는 향후 연구에서:
- 특성 공간의 기하학적 성질 분석
- 제어 벡터의 합성(superposition) 가능성 검토
- 대수적 조작을 통한 생성 제어 (예: A + B - C)

***

### 결론

**Uni-ControlNet**은 텍스트-이미지 확산 모델의 제어 가능성을 근본적으로 재정의했습니다. 단 2개의 어댑터로 다양한 조건을 통합 처리하고, 미세 조정 비용을 상수로 유지하면서도 기존 방법들을 능가하는 성능을 달성했습니다. 

특히 **구성 가능한 제어**라는 개념을 실질적으로 증명함으로써, 향후 멀티모달 조건 생성의 방향을 제시했습니다. ControlNeXt, DC-ControlNet 등 2024-2025의 후속 연구들이 다양한 최적화 방향(효율성, 섬세성, 분해성)을 탐색하고 있는 것은 Uni-ControlNet의 기초적 기여를 반영합니다.

앞으로의 연구는 다음을 중점으로 진행될 것으로 예상됩니다:

1. **더 나은 조건 충돌 해결**: 상충하는 조건 간의 의미론적 조화 메커니즘
2. **동적 가중치 학습**: 조건별 중요도를 자동으로 학습하는 라우터
3. **임의 조건 일반화**: 훈련하지 않은 조건 유형도 즉시 지원
4. **효율성 극대화**: Uni-ControlNet과 ControlNeXt 철학의 통합
5. **물리적 제약 통합**: 현실적 불가능한 구성 배제

이러한 방향의 진전을 통해 텍스트-이미지 생성 모델은 진정한 의미의 사용자 지정 생성 도구로 발전할 것으로 기대됩니다.

***

## 참고 문헌 (ID 인덱스)

 Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models (arXiv:2305.16322v3, NeurIPS 2023)[1]
 ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models (arXiv:2302.05543, ICCV 2023)[2]
 GLIGEN: Open-Set Grounded Text-to-Image Generation (CVPR 2023)[3]
 T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models (arXiv:2302.08453)[4]
 ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback (2024.04)[5]
 ControlNeXt: Powerful and Efficient Control for Image and Video Generation (2024.08)[6]

출처
[1] 2305.16322v3.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8cde8ab8-1b16-4451-9f54-6089aa16083f/2305.16322v3.pdf
[2] Adding Conditional Control to Text-to-Image Diffusion Models https://ieeexplore.ieee.org/document/10377881/
[3] Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models https://arxiv.org/abs/2305.16322
[4] ControlNet-XS: Designing an Efficient and Effective Architecture for Controlling Text-to-Image Diffusion Models https://arxiv.org/abs/2312.06573
[5] ControlNet++: Improving Conditional Controls with Efficient Consistency
  Feedback http://arxiv.org/pdf/2404.07987.pdf
[6] ControlNeXt: Powerful and Efficient Control for Image and ... https://arxiv.org/html/2408.06070v1
[7] FlexEControl: Flexible and Efficient Multimodal Control for ... https://arxiv.org/html/2405.04834v2
[8] DC-ControlNet: Decoupling Inter- and Intra-Element ... https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_DC-ControlNet_Decoupling_Inter-_and_Intra-Element_Conditions_in_Image_Generation_with_ICCV_2025_paper.pdf
[9] DC-ControlNet: Decoupling Inter- and Intra-Element ... https://arxiv.org/html/2502.14779v1
[10] Video ControlNet: Towards Temporally Consistent Synthetic-to-Real Video Translation Using Conditional Image Diffusion Models https://arxiv.org/abs/2305.19193
[11] ControlNet-XS: Rethinking the Control of Text-to-Image Diffusion Models as Feedback-Control Systems https://link.springer.com/10.1007/978-3-031-73223-2_20
[12] CCEdit: Creative and Controllable Video Editing via Diffusion Models https://ieeexplore.ieee.org/document/10655700/
[13] DreaMoving: A Human Video Generation Framework based on Diffusion Models https://arxiv.org/abs/2312.05107
[14] VideoControlNet: A Motion-Guided Video-to-Video Translation Framework by Using Diffusion Model with ControlNet https://arxiv.org/abs/2307.14073
[15] Generating Images with 3D Annotations Using Diffusion Models https://www.semanticscholar.org/paper/1036f39069a1fe3d5f818f1d7bc07286ad3f1363
[16] Exploring the Capability of Text-to-Image Diffusion Models With Structural Edge Guidance for Multispectral Satellite Image Inpainting https://ieeexplore.ieee.org/document/10445344/
[17] Adding Conditional Control to Text-to-Image Diffusion Models https://arxiv.org/abs/2302.05543
[18] Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse
  Controls to Any Diffusion Model http://arxiv.org/pdf/2404.09967.pdf
[19] FlexControl: Computation-Aware ControlNet with Differentiable Router for
  Text-to-Image Generation https://arxiv.org/html/2502.10451v1
[20] CoDNet: controlled diffusion network for structure-based drug design https://academic.oup.com/bioinformaticsadvances/article/doi/10.1093/bioadv/vbaf031/8025957
[21] ControlDreamer: Blending Geometry and Style in Text-to-3D http://arxiv.org/pdf/2312.01129.pdf
[22] ControlSR: Taming Diffusion Models for Consistent Real-World Image Super
  Resolution https://arxiv.org/html/2410.14279
[23] ControlNeXt: Powerful and Efficient Control for Image and Video
  Generation https://arxiv.org/html/2408.06070v3
[24] Adding Conditional Control to Text-to-Image Diffusion Models https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf
[25] GLIGEN: Open-Set Grounded Text-to-Image Generation https://openaccess.thecvf.com/content/CVPR2023/papers/Li_GLIGEN_Open-Set_Grounded_Text-to-Image_Generation_CVPR_2023_paper.pdf
[26] arXiv:2302.08453v2 [cs.CV] 20 Mar 2023 https://arxiv.org/pdf/2302.08453.pdf
[27] Gligen: Open-Set Grounded Text-to-Image Generation - ar5iv https://ar5iv.labs.arxiv.org/html/2301.07093
[28] [2302.08453] T2I-Adapter: Learning Adapters to Dig out ... https://arxiv.org/abs/2302.08453
[29] Training-Free Spatial Control of Any Text-to-Image Diffusion ... https://openaccess.thecvf.com/content/CVPR2024/papers/Mo_FreeControl_Training-Free_Spatial_Control_of_Any_Text-to-Image_Diffusion_Model_with_CVPR_2024_paper.pdf
[30] Grounding Text-to-Image Diffusion Models for Controlled ... https://arxiv.org/abs/2501.09194
[31] Weaving Efficient Controllable T2I Generation with Multi ... https://arxiv.org/html/2510.14882v1
[32] ControlNet - ICCV 2023 Open Access Repository https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html
[33] GLIGEN: Open-Set Grounded Text-to-Image Generation https://openaccess.thecvf.com/content/CVPR2023/supplemental/Li_GLIGEN_Open-Set_Grounded_CVPR_2023_supplemental.pdf
[34] Controllable Generation with Text-to-Image Diffusion Models https://arxiv.org/html/2403.04279v2
[35] Grounding Text-To-Image Diffusion Models For Controlled ... https://arxiv.org/html/2501.09194v1
[36] GLIGEN:Open-Set Grounded Text-to-Image Generation. https://gligen.github.io
[37] GLIGEN: Open-Set Grounded Text-to-Image Generation (CVPR 2023, Demo Video) https://www.youtube.com/watch?v=-MCkU7IAGKs
[38] T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models | Bytez https://bytez.com/docs/arxiv/2302.08453/paper?related=creators
[39] ControlNet: A Complete Guide https://stable-diffusion-art.com/controlnet/
[40] GLIGEN (Grounded Language-to-Image Generation) - Hugging Face https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/gligen
[41] Efficient Controllable Generation for SDXL with T2I-Adapters https://huggingface.co/blog/t2i-sdxl-adapters
[42] [논문리뷰] Adding Conditional Control to Text-to-Image ... https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/controlnet/
[43] lllyasviel/ControlNet: Let us control diffusion models! https://github.com/lllyasviel/ControlNet
[44] GitHub - gligen/GLIGEN: Open-Set Grounded Text-to-Image Generation https://github.com/gligen/GLIGEN
[45] T2I-Adapter https://huggingface.co/docs/diffusers/en/using-diffusers/t2i_adapter
[46] Your Diffusion Model is Secretly a Zero-Shot Classifier https://ieeexplore.ieee.org/document/10376944/
[47] RF Genesis: Zero-Shot Generalization of mmWave Sensing through Simulation-Based Data Synthesis and Generative Diffusion Models https://dl.acm.org/doi/10.1145/3625687.3625798
[48] Zero-1-to-3: Zero-shot One Image to 3D Object https://ieeexplore.ieee.org/document/10378322/
[49] Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models https://arxiv.org/abs/2310.10639
[50] ZET-Speech: Zero-shot adaptive Emotion-controllable Text-to-Speech Synthesis with Diffusion and Style-based Models https://arxiv.org/abs/2305.13831
[51] Blind Audio Bandwidth Extension: A Diffusion-Based Zero-Shot Approach https://ieeexplore.ieee.org/document/10768977/
[52] Deshadow-Anything: When Segment Anything Model Meets Zero-shot shadow removal https://arxiv.org/abs/2309.11715
[53] Prompt Consistency for Zero-Shot Task Generalization https://arxiv.org/abs/2205.00049
[54] Mega-TTS: Zero-Shot Text-to-Speech at Scale with Intrinsic Inductive Bias https://arxiv.org/abs/2306.03509
[55] Team TheSyllogist at SemEval-2023 Task 3: Language-Agnostic Framing Detection in Multi-Lingual Online News: A Zero-Shot Transfer Approach https://aclanthology.org/2023.semeval-1.283
[56] VGDiffZero: Text-to-image Diffusion Models Can Be Zero-shot Visual
  Grounders https://arxiv.org/pdf/2309.01141.pdf
[57] Data Distribution Distilled Generative Model for Generalized Zero-Shot
  Recognition https://arxiv.org/html/2402.11424v1
[58] ZeroDiff: Solidified Visual-Semantic Correlation in Zero-Shot Learning https://arxiv.org/pdf/2406.02929.pdf
[59] InvFussion: Bridging Supervised and Zero-shot Diffusion for Inverse
  Problems https://arxiv.org/html/2504.01689v1
[60] Discovery and Expansion of New Domains within Diffusion Models http://arxiv.org/pdf/2310.09213.pdf
[61] Bi-level Guided Diffusion Models for Zero-Shot Medical Imaging Inverse
  Problems https://arxiv.org/html/2404.03706v1
[62] Towards a Mechanistic Explanation of Diffusion Model Generalization https://arxiv.org/html/2411.19339v2
[63] RevCD -- Reversed Conditional Diffusion for Generalized Zero-Shot
  Learning https://arxiv.org/abs/2409.00511
[64] Your Diffusion Model is Secretly a Zero-Shot Classifier https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.pdf
[65] Attribute-Specific Image Prompts for Controllable Human ... https://arxiv.org/html/2509.18092v1
[66] Zero-Shot Video Deraining with Video Diffusion Models https://arxiv.org/html/2511.18537v1
[67] Compositional Slider for Disentangled Multiple-Attribute ... https://arxiv.org/html/2509.01028v1
[68] RelaCtrl: Relevance-Guided Efficient Control for Diffusion ... https://arxiv.org/html/2502.14377v2
[69] Conditional Latent Diffusion Models for Zero-Shot Instance ... https://arxiv.org/html/2508.04122v1
[70] Chimera: Compositional Image Generation using Part- ... https://arxiv.org/html/2510.18083v1
[71] Bridging the Skeleton-Text Modality Gap: Diffusion- ... https://arxiv.org/html/2411.10745v4
[72] Compositional Image Generation with Multimodal Controls https://arxiv.org/html/2511.21691v1
[73] UnPose: Uncertainty-Guided Diffusion Priors for Zero-Shot ... https://arxiv.org/html/2508.15972v1
[74] CompSlider: Compositional Slider for Disentangled Multiple ... https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_CompSlider_Compositional_Slider_for_Disentangled_Multiple-Attribute_Image_Generation_ICCV_2025_paper.pdf
[75] Text-to-Image Diffusion Models are Zero-Shot Classifiers https://proceedings.neurips.cc/paper_files/paper/2023/file/b87bdcf963cad3d0b265fcb78ae7d11e-Paper-Conference.pdf
[76] [Literature Review] Efficient Conditional Generation on ... https://www.themoonlight.io/en/review/efficient-conditional-generation-on-scale-based-visual-autoregressive-models
[77] Diffusion-based vision-language model for zero-shot ... https://www.sciencedirect.com/science/article/abs/pii/S095219762502189X
[78] On Conditional and Compositional Language Model ... https://www.ijcai.org/proceedings/2023/0460.pdf
[79] dvlab-research/ControlNeXt: Controllable video and image ... - GitHub https://github.com/dvlab-research/ControlNeXt
[80] RF Genesis: Zero-Shot Generalization of mmWave Sensing ... https://xyzhang.ucsd.edu/papers/Xingyu.Chen_SenSys23_RFGen.pdf
[81] Conditional Adapters: Parameter-efficient Transfer ... https://proceedings.neurips.cc/paper_files/paper/2023/file/19d7204af519eae9993f7f72377a0ec0-Paper-Conference.pdf
[82] [평범한 학부생이 하는 논문 리뷰] ControlNeXt https://juniboy97.tistory.com/87
[83] Minimal Impact ControlNet: Advancing Multi- ... https://openreview.net/forum?id=rzbSNDXgGD
[84] DiffSQL: Leveraging Diffusion Model for Zero-Shot Self- ... https://www.ijcai.org/proceedings/2025/0981.pdf
