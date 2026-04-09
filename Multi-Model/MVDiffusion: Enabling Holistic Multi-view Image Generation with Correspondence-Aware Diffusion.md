# MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

MVDiffusion은 **픽셀 간 대응 관계(pixel-to-pixel correspondences)** 가 주어졌을 때, 텍스트 프롬프트로부터 일관된 멀티뷰 이미지를 **동시에(holistically)** 생성할 수 있다는 것을 주장합니다.

기존의 자기회귀(autoregressive) 방식(이미지 워핑 + 인페인팅 반복)이 가진 **오차 누적(error accumulation)** 문제와 **루프 클로저(loop closure)** 문제를 근본적으로 해결합니다.

### 주요 기여 (Contributions)

| 기여 | 설명 |
|------|------|
| **Correspondence-Aware Attention (CAA)** | 멀티뷰 간 픽셀 대응을 활용한 새로운 크로스뷰 어텐션 메커니즘 |
| **동시 병렬 생성** | 모든 뷰를 동시에 생성하여 오차 누적 방지 |
| **최소한의 수정** | 사전학습된 Stable Diffusion 가중치를 동결(freeze)하여 일반화 능력 보존 |
| **두 가지 태스크 SOTA** | 파노라마 생성 및 멀티뷰 깊이→이미지 생성 모두 최고 성능 달성 |
| **데이터 효율성** | 단 10k 파노라마 데이터로 훈련하여 강력한 일반화 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 멀티뷰 생성 방법들의 세 가지 핵심 문제:

**① 오차 누적 (Error Accumulation)**
- SceneScape, Text2Room 등은 $n$번째 이미지 생성 시 $(n-1)$번째 이미지에만 조건화
- 초기 오류가 점점 증폭되어 후반부 이미지 품질이 급격히 저하

**② 루프 클로저 실패 (Loop Closure Failure)**
- Text2Light 등 자기회귀 모델은 가장 왼쪽과 오른쪽 경계가 연결되지 않음
- 파노라마의 경우 $0°$와 $360°$가 자연스럽게 이어져야 하지만 불가능

**③ 전역적 일관성 부재 (Lack of Global Awareness)**
- 각 이미지가 국소적 컨텍스트에만 의존하여 전체적인 스타일/조명/구조 불일치 발생

---

### 2.2 제안하는 방법

#### 2.2.1 Latent Diffusion Model (LDM) 기반

LDM의 훈련 목적함수:

$$L_{LDM} := \mathbb{E}_{\mathcal{E}(\mathbf{x}), \mathbf{y}, \boldsymbol{\epsilon} \sim \mathcal{N}(0,1), t} \left[ \| \boldsymbol{\epsilon} - \epsilon_\theta(\mathbf{Z}_t, t, \tau_\theta(\mathbf{y})) \|_2^2 \right] $$

- $\mathbf{Z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{h \times w \times c}$: VAE 인코더로 얻은 잠재 표현
- $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$: 원본 이미지 (다운샘플링 팩터 $f = H/h = 8$)
- $\epsilon_\theta$: 시간 조건부 UNet 디노이징 네트워크
- $\tau_\theta(\mathbf{y})$: 텍스트/이미지 조건 인코더

#### 2.2.2 MVDiffusion 훈련 목적함수

$$L_{\text{MVDiffusion}} := \mathbb{E}_{\{\mathbf{Z}_t^i = \mathcal{E}(\mathbf{x}^i)\}_{i=1}^N, \{\boldsymbol{\epsilon}^i \sim \mathcal{N}(0,I)\}_{i=1}^N, \mathbf{y}, t} \left[ \sum_{i=1}^N \| \boldsymbol{\epsilon}^i - \epsilon_\theta^i\left(\{\mathbf{Z}_t^i\}, t, \tau_\theta(\mathbf{y})\right) \|_2^2 \right] $$

- $N$개 뷰를 동시에 디노이징
- 사전학습된 SD 가중치는 동결, **CAA 블록만 학습**

#### 2.2.3 Correspondence-Aware Attention (CAA) 핵심 수식

소스 특징맵의 위치 $\mathbf{s}$에서의 메시지 계산:

```math
\mathbf{M} = \sum_l \sum_{t_*^l \in \mathcal{N}(\mathbf{t}^l)} \text{SoftMax}\left( \left[\mathbf{W}_Q \bar{F}(\mathbf{s})\right] \cdot \left[\mathbf{W}_K \bar{F}^l(t_*^l)\right] \right) \mathbf{W}_V \bar{F}^l(t_*^l)
```

위치 인코딩이 적용된 특징:

```math
\bar{F}(\mathbf{s}) = F(\mathbf{s}) + \gamma(\mathbf{0}), \quad \bar{F}^l(t_*^l) = F^l(t_*^l) + \gamma(\mathbf{s}_*^l - \mathbf{s})
```

- $\gamma(\cdot)$: 2D 변위(displacement)에 대한 주파수 인코딩 (Fourier positional encoding)
- $\mathcal{N}(\mathbf{t}^l)$: 타겟 픽셀 $\mathbf{t}^l$ 주변 $K \times K$ ($K=3$, 9포인트) 이웃
- $\mathbf{s}_*^l - \mathbf{s}$: 소스-타겟 픽셀 간 2D 변위벡터
- $F^l(t_*^l)$: 정수 좌표가 아닐 경우 쌍선형 보간(bilinear interpolation) 적용
- **ControlNet 방식의 Zero-initialization**: 최종 선형 레이어와 잔차 블록의 합성곱 레이어를 0으로 초기화하여 원래 SD 기능 보존

---

### 2.3 모델 구조

```
입력: N개의 노이즈 잠재코드 {Z_t^i} + 텍스트 프롬프트 y
         ↓
┌─────────────────────────────────────────────────┐
│           Multi-branch UNet (가중치 공유)          │
│  Branch 1    Branch 2    ...    Branch N         │
│  [SD UNet]  [SD UNet]          [SD UNet]         │
│     ↓           ↓                  ↓             │
│  ┌──────────────────────────────────────┐        │
│  │  Correspondence-Aware Attention (CAA)│        │
│  │  - 크로스뷰 어텐션                    │        │
│  │  - 위치 인코딩 (변위 기반)             │        │
│  │  - Feed-Forward Network (FFN)        │        │
│  │  - Zero Convolution                  │        │
│  └──────────────────────────────────────┘        │
│     ↓           ↓                  ↓             │
│  (반복: 각 UNet 블록 후 CAA 삽입)                  │
└─────────────────────────────────────────────────┘
         ↓ (VAE 디코더)
출력: N개의 일관된 이미지 {x̃^i}
```

#### 두 가지 적용 태스크

**① 파노라마 생성**
- 8개의 512×512 퍼스펙티브 뷰 (FOV=90°, 45° 오버랩) 동시 생성
- 대응관계: 평면 단응성(planar homography)으로 계산
- 기반 모델: Stable Diffusion v2 / SD Inpainting

**② 멀티뷰 깊이→이미지 생성**
- **Generation Module**: 192×256 크기의 키프레임 생성 (SD depth-conditioned 기반)
- **Interpolation Module**: VideoLDM에서 영감받아 키프레임 사이 중간 프레임 생성
  - 키프레임 브랜치: 마스크=1, 생성된 이미지 조건화
  - 중간프레임 브랜치: 마스크=0, 새로운 이미지 생성
- 대응관계: 깊이 기반 역투영(unprojection) + 투영(projection)

#### 훈련 전략 (깊이→이미지)
- **1단계**: ScanNet 전체 데이터로 단일뷰 SD UNet 파인튜닝 (CAA 없음)
- **2단계**: CAA 블록 + 이미지 조건 블록 추가, 해당 파라미터만 학습

---

### 2.4 성능 향상

#### 파노라마 생성 (Matterport3D, 1092개 테스트)

| 방법 | FID↓ | IS↑ | CS↑ |
|------|------|-----|-----|
| Inpainting [11,18] | 42.13 | 7.08 | 29.05 |
| Text2Light [6] | 48.71 | 5.41 | 25.98 |
| SD (Pano) [36] | 23.02 | 6.58 | 28.63 |
| SD (Perspective) [36] | 25.59 | 7.29 | 30.25 |
| **MVDiffusion (Ours)** | **21.44** | **7.32** | **30.04** |

#### 멀티뷰 깊이→이미지 (ScanNet, 590 시퀀스)

| 방법 | FID↓ | IS↑ | CS↑ |
|------|------|-----|-----|
| RePaint [27] | 70.05 | 7.15 | 26.98 |
| ControlNet [52] | 43.67 | 7.23 | 28.14 |
| **Ours** | **23.10** | **7.27** | **29.03** |

#### 멀티뷰 일관성 (PSNR 기반)

| 방법 | 파노라마 PSNR↑ | 파노라마 Ratio↑ | Depth PSNR↑ | Depth Ratio↑ |
|------|--------------|----------------|-------------|--------------|
| G.T. | 37.7 | 1.00 | 21.41 | 1.00 |
| SD (Perspective) | 10.6 | 0.28 | 11.20 | 0.44 |
| **MVDiffusion** | **25.4** | **0.67** | **17.41** | **0.76** |

---

### 2.5 한계 (Limitations)

**① 추론 속도**: DDIM 50스텝 필요 → 느린 생성 속도 (모든 DM 기반 방법의 공통 한계)

**② 메모리 요구량**: $N$개 뷰의 병렬 디노이징으로 메모리 집약적 → 뷰 수 확장 어려움

**③ 확장성 제한**: 긴 가상 투어(long virtual tour)처럼 많은 이미지가 필요한 경우 적용 곤란

**④ 대응관계 전제**: 픽셀 간 대응관계가 사전에 알려져 있어야 적용 가능 (제약 조건)

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화의 핵심 메커니즘: 가중치 동결 전략

MVDiffusion의 일반화 성능의 핵심은 **사전학습된 Stable Diffusion 가중치를 완전히 동결(freeze)** 하는 전략입니다.

```
훈련 가능 파라미터: CAA 블록만 (새로 추가된 부분)
동결된 파라미터:   원래 SD UNet 전체 (기존 지식 보존)
```

이 설계의 핵심 이점:

- **SD의 광범위한 사전지식 보존**: 수억 장의 인터넷 이미지로 학습된 SD의 도메인 지식이 그대로 유지
- **도메인 이동(domain shift) 저항성**: 훈련 데이터와 다른 도메인에서도 SD의 표현력으로 대응 가능
- **적은 데이터로 충분**: CAA만 학습하므로 10k 파노라마만으로도 효과적

### 3.2 실험적 증거: "Wild" 일반화

논문에서 가장 강력한 일반화 증거는 **실내 데이터(Matterport3D)만으로 훈련했음에도 다양한 도메인에서 성공적 생성**을 보인 것입니다:

| 생성 도메인 | 훈련 데이터 포함 여부 |
|-----------|-------------------|
| 실내 씬 | ✅ 포함 |
| **야외 산/설경** | ❌ 미포함 → 성공 |
| **만화 스타일** | ❌ 미포함 → 성공 |
| **성곽/중세 건물** | ❌ 미포함 → 성공 |
| **정원/자연** | ❌ 미포함 → 성공 |

이는 CAA 블록이 **도메인 독립적인 기하학적 일관성 학습**만 담당하고, 시각적 다양성은 SD가 담당하기 때문입니다.

### 3.3 SD (Pano)와의 비교: 일반화의 핵심 차이

> *"Another shortcoming of this model [SD Pano] is its requirement for substantial data to reach robust generalization. In contrast, our model, leveraging a frozen pre-trained stable diffusion, demonstrates a robust generalization ability with a small amount of training data."* — 논문 원문

SD (Pano)는 전체 UNet을 파인튜닝하므로 훈련 분포(실내)에 과적합되어 야외 생성에 실패하는 반면, MVDiffusion은 동결된 SD 덕분에 훈련 분포 밖에서도 강건합니다.

### 3.4 프레임 수 일반화

> *"Notably, even though our model has been trained using a frame length of 12, it has the capability to be generalized to accommodate any number of frames."* — 논문 원문

훈련 시 12프레임 고정으로 학습했음에도, **임의 개수의 프레임**으로 일반화 가능합니다. 이는 CAA가 프레임 수에 독립적인 페어와이즈(pair-wise) 어텐션 구조를 가지기 때문입니다.

### 3.5 일반화 향상을 위한 잠재적 방향

논문의 구조에서 도출할 수 있는 일반화 향상 가능성:

**① 더 강력한 베이스 모델로 교체**
$$\text{SD} \rightarrow \text{SDXL, SD3, FLUX 등}$$
베이스 모델의 표현력 향상 → 즉각적인 일반화 향상 기대

**② CAA의 어텐션 범위 확장**
- 현재 $K=3$ (9포인트 이웃) → $K$ 값 증가로 더 넓은 컨텍스트 활용
- 계층적(hierarchical) CAA 설계로 다중 스케일 대응 학습

**③ 대응관계 없는 시나리오로 확장**
- 현재는 픽셀 대응이 필요 → Correspondence-free 방식으로 일반화
- NeRF/3DGS 기반 암묵적 대응 학습 가능성

**④ 다중 모달 조건화**
- 텍스트 + 이미지 + 깊이 + 법선 + 시맨틱 등 다양한 조건 통합
- 더 풍부한 조건으로 제어 가능성 향상

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 관련 연구 계보

```
[2020] NeRF (Mildenhall et al.) - 암묵적 신경 표현
[2020] DDPM (Ho et al.) - 확산 모델 기초
[2021] LDM/Stable Diffusion (Rombach et al.) - 잠재 공간 확산
[2022] DreamFusion (Poole et al.) - 텍스트→3D (SDS loss)
[2022] Text2Light (Chen et al.) - 자기회귀 HDR 파노라마
[2022] ControlNet (Zhang & Agrawala) - 조건부 확산 제어
[2023] Text2Room (Höllein et al.) - 자기회귀 3D 씬
[2023] SceneScape (Fridman et al.) - 자기회귀 씬 생성
[2023] MVDiffusion (Tang et al.) - 동시 멀티뷰 생성 ★
[2023] MultiDiffusion (Bar-Tal et al.) - 병렬 확산 경로 융합
[2023] VideoLDM (Blattmann et al.) - 비디오 잠재 확산
```

### 4.2 세부 비교 분석

| 방법 | 생성 방식 | 일관성 메커니즘 | 루프 클로저 | 일반화 | 데이터 요구량 |
|------|---------|--------------|-----------|--------|------------|
| **Text2Light** [6] | 자기회귀 | 없음 | ❌ | 낮음 | 대량 |
| **Text2Room** [18] | 자기회귀(워핑+인페인팅) | 지역적 | ❌ | 중간 | 대량 |
| **SceneScape** [11] | 자기회귀(워핑+인페인팅) | 지역적 | ❌ | 중간 | 대량 |
| **MultiDiffusion** [1] | 병렬(겹침 영역 평균) | 경계 공유 | △ | 높음 | 없음 |
| **ControlNet** [52] | 단일뷰 | 없음 | ❌ | 높음 | 중간 |
| **MVDiffusion** | **전역 병렬** | **CAA (전역)** | **✅** | **높음** | **소량** |

### 4.3 MVDiffusion 이후 후속 연구들과의 비교 (2023~2024)

> **⚠️ 주의**: 아래 내용은 제가 학습 데이터에서 파악한 내용이며, 논문에 명시적으로 언급되지 않은 후속 연구에 대해서는 정확도에 한계가 있을 수 있습니다.

MVDiffusion이 제시한 패러다임(동시 생성 + 대응 어텐션)은 이후 연구들에 영향을 주었으나, 논문 자체에서 비교한 방법들 외의 후속 연구에 대한 구체적 수치 비교는 제가 확신하는 수준에서만 제시합니다:

- **Zero-1-to-3** (Liu et al., 2023): 단일 이미지에서 임의 시점 생성에 집중, 대규모 3D 데이터 필요
- **SyncDreamer** (Liu et al., 2023): 멀티뷰 확산 모델로 3D 일관성 강화, MVDiffusion과 유사한 병렬 접근
- **MVDream** (Shi et al., 2023): 3D-aware 멀티뷰 확산으로 3D 객체에 집중

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### ① 패러다임 전환: 순차→동시 생성

MVDiffusion은 멀티뷰 생성 분야에서 **"자기회귀 순차 생성" → "전역 병렬 생성"** 이라는 패러다임 전환을 제시했습니다. 이는 다음 연구 방향에 직접적 영향을 미칩니다:

- 비디오 생성의 시간적 일관성 연구
- 3D 재구성과 생성의 융합 연구
- 장거리(long-range) 씬 생성 연구

#### ② 가중치 동결 파인튜닝 전략의 확산

CAA만 학습하고 베이스 모델을 동결하는 전략은 **파라미터 효율적 파인튜닝(PEFT)** 패러다임과 맥을 같이 하며, 이후 다양한 분야에서 유사한 접근법이 채택되었습니다.

#### ③ 픽셀 대응 기반 어텐션의 일반화

CAA의 설계 원리(대응 픽셀 간 로컬 이웃 어텐션 + 변위 위치 인코딩)는 다음 분야로 확장 가능성:

$$\text{멀티뷰 이미지} \rightarrow \text{비디오 프레임} \rightarrow \text{3D 포인트 클라우드} \rightarrow \text{동적 씬}$$

#### ④ 새로운 평가 지표 제안

"Overlapping PSNR Ratio" 지표는 멀티뷰 일관성 평가의 표준으로 활용될 수 있습니다:

$$\text{Ratio} = \frac{\text{PSNR}_{\text{generated, overlapping}}}{\text{PSNR}_{\text{GT, overlapping}}}$$

### 5.2 앞으로 연구 시 고려할 점

#### ① 계산 효율성 개선 (Computational Efficiency)

현재 병렬 디노이징은 GPU 메모리를 선형적으로 소모합니다:

$$\text{Memory} \propto N \times (\text{UNet memory}) + \text{CAA memory}$$

**고려 방향:**
- 일관성(consistency) 품질을 유지하면서 샘플링 스텝 수를 줄이는 연구 (e.g., consistency models, flow matching)
- 메모리 효율적인 어텐션 구현 (e.g., FlashAttention 적용)
- 점진적(progressive) 해상도 생성으로 메모리 부담 분산

#### ② 대응관계 없는 시나리오 (Correspondence-Free) 확장

현재 방법은 **픽셀 대응관계 사전 계산을 전제**로 합니다. 이를 극복하기 위한 연구:

- 깊이/포즈 추정 오차에 강건한 소프트 대응(soft correspondence) 학습
- 대응관계 없이 의미론적 유사성(semantic similarity)으로 크로스뷰 어텐션
- 잡음이 있는 대응에서의 강건성(robustness) 향상

#### ③ 동적 씬 및 시간적 일관성

정적 씬에서 **동적 씬**으로의 확장:

$$\text{공간적 다중뷰 일관성} \xrightarrow{\text{확장}} \text{시공간 일관성 (spatio-temporal)}$$

- 비디오 멀티뷰 생성: 공간(뷰) × 시간(프레임) 2D 어텐션 그리드
- 움직이는 물체의 일관된 추적 및 생성

#### ④ 더 큰 스케일의 씬 생성

현재 파노라마(8뷰)와 ScanNet 시퀀스(12프레임)의 한계를 넘어:

- **Long virtual tour**: 수백 프레임의 연속 씬 생성
- **메모리 효율적 청크(chunking) 전략**으로 임의 길이 시퀀스 처리
- 전역 씬 표현(NeRF/3DGS)과 통합하여 일관성 강화

#### ⑤ 평가 지표의 정교화

현재 PSNR 기반 일관성 지표의 한계:

- PSNR은 픽셀 수준 유사성만 측정, **지각적(perceptual) 일관성** 미반영
- 생성 이미지 간 **의미론적(semantic) 일관성** 평가 필요
- **인간 평가(human evaluation)** 기준의 자동화 지표 개발 필요

```python
# 제안하는 향상된 일관성 평가 파이프라인 (개념적)
consistency_score = weighted_average([
    psnr_ratio,           # 픽셀 수준 (현재)
    lpips_consistency,    # 지각적 유사성
    clip_semantic_sim,    # 의미론적 일관성
    depth_consistency     # 3D 기하학적 일관성
])
```

#### ⑥ 생성-재구성 공동 최적화

생성된 멀티뷰 이미지를 3D 재구성(NeRF, 3DGS)에 직접 활용하는 **end-to-end 파이프라인**:

$$\text{텍스트} \rightarrow \text{MVDiffusion} \rightarrow \text{멀티뷰 이미지} \rightarrow \text{3D 재구성}$$

각 단계의 손실을 통합하여 **3D 일관성을 직접 목적함수에 반영**하는 연구가 필요합니다.

---

## 참고 자료

**주요 참고 문헌 (논문 원문 기준):**

1. **Tang et al. (2023)** - "MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion," *NeurIPS 2023*. arXiv:2307.01097v7
2. **Rombach et al. (2022)** - "High-Resolution Image Synthesis with Latent Diffusion Models," *CVPR 2022*. [36]
3. **Zhang & Agrawala (2023)** - "Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)," arXiv:2302.05543 [52]
4. **Höllein et al. (2023)** - "Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models," arXiv:2303.11989 [18]
5. **Fridman et al. (2023)** - "SceneScape: Text-Driven Consistent Scene Generation," arXiv:2302.01133 [11]
6. **Chen et al. (2022)** - "Text2Light: Zero-Shot Text-Driven HDR Panorama Generation," *ACM TOG* [6]
7. **Bar-Tal et al. (2023)** - "MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation," arXiv:2302.08113 [1]
8. **Blattmann et al. (2023)** - "Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models (VideoLDM)," *CVPR 2023* [2]
9. **Ho et al. (2020)** - "Denoising Diffusion Probabilistic Models," *NeurIPS 2020* [16]
10. **Chang et al. (2017)** - "Matterport3D: Learning from RGB-D Data in Indoor Environments," arXiv:1709.06158 [4]
11. **MVDiffusion 프로젝트 페이지**: https://mvdiffusion.github.io/
12. **Vaswani et al. (2017)** - "Attention Is All You Need," *NeurIPS 2017* [50]
13. **Lugmayr et al. (2022)** - "RePaint: Inpainting using Denoising Diffusion Probabilistic Models," *CVPR 2022* [27]
14. **StabilityAI** - Stable Diffusion v2, Stable Diffusion Inpainting, Stable Diffusion 2 Depth [46, 47, 48]
