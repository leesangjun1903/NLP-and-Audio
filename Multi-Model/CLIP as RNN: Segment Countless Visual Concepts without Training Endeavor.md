# CLIP as RNN: Segment Countless Visual Concepts without Training Endeavor

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

"CLIP as RNN (CaR)"의 핵심 주장은 다음과 같습니다:

> **파인튜닝 없이도, 사전학습된 CLIP 모델의 방대한 어휘 공간(vocabulary space)을 완전히 보존하면서, 재귀적(recurrent) 정제 과정을 통해 고품질의 오픈-어휘 이미지 분할(open-vocabulary image segmentation)을 수행할 수 있다.**

기존 방법들은 마스크 어노테이션이나 추가적인 이미지-텍스트 데이터로 파인튜닝을 수행하여 CLIP의 광대한 어휘 공간을 제한시켰습니다. CaR은 이러한 파인튜닝 과정을 완전히 제거함으로써 CLIP의 어휘 공간을 보존하고, RNN과 유사한 반복 정제 구조를 통해 마스크 품질을 향상시킵니다.

### 주요 기여

1. **훈련 불필요(Training-Free) 재귀 프레임워크**: 고정된 CLIP 가중치를 공유하는 두 단계 분할기(two-stage segmenter)를 재귀 유닛으로 사용하는 새로운 아키텍처 제안
2. **광대한 어휘 공간 보존**: 파인튜닝 없이 CLIP의 전체 어휘 공간(브랜드, 랜드마크, 애니메이션 캐릭터 등)을 활용 가능
3. **State-of-the-Art 달성**: 제로샷 시맨틱 분할 및 참조 분할(referring segmentation) 벤치마크에서 기존 최고 성능 대폭 초과

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제 1: 파인튜닝으로 인한 어휘 공간 축소

마스크 어노테이션을 활용한 파인튜닝 방법들(OVSeg, Grounded SAM 등)은 PASCAL VOC, COCO 등 제한된 카테고리 목록을 가진 데이터셋을 사용합니다. 이는 CLIP이 본래 보유한 수십억 개의 어휘를 수십~수백 개로 제한시킵니다.

#### 문제 2: 파인튜닝 없이는 마스크 품질이 낮음

파인튜닝 없이 CLIP을 직접 사용하는 방법들(MaskCLIP, ReCo 등)은 약지도(weak supervision) 기반 사전학습 특성 때문에 밀집 예측(dense prediction)에 최적화되지 않아 마스크 품질이 열악합니다. 특히, **이미지에 존재하지 않는 객체에 대한 텍스트 쿼리가 입력될 때 오류가 심각합니다.**

### 2.2 제안하는 방법 (수식 포함)

#### RNN과의 유사성

표준 RNN의 수식은 다음과 같습니다:

$$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b_h) \tag{1}$$

$$y_t = W_{hy}h_t + b_y \tag{2}$$

CaR에서 텍스트 쿼리 $h_t$는 RNN의 숨김 상태(hidden state)에 대응하며, 각 타임스텝마다 동일한 가중치를 공유하는 분할기가 재귀적으로 동작합니다.

#### CaR의 핵심 수식

**Stage 1 - 마스크 제안 생성(Mask Proposal Generation):**

$$y_t = f(x_t, h_{t-1}; W_f) \tag{3}$$

- $f(\cdot, \cdot)$: 마스크 제안 생성기 (CLIP + gradCAM 기반)
- $x_t \in \mathbb{R}^{3 \times H \times W}$: 입력 이미지
- $h_{t-1}$: 이전 스텝의 텍스트 쿼리 집합
- $W_f$: 사전학습된 CLIP 가중치 (고정)

**시각적 프롬프팅(Visual Prompting):**

$$x'_t = v(x_t, y_t) \tag{4}$$

- $v(\cdot, \cdot)$: 시각적 프롬프트 함수 (빨간 원, 배경 블러 등 적용)
- $x'\_t$: $N_{t-1}$개의 프롬프트된 이미지들

**Stage 2 - 마스크 분류(Mask Classification):**

$$P_t = g(x'_t, h_{t-1}; W_g) \tag{5}$$

- $g(\cdot, \cdot)$: 마스크 분류기 (더 큰 CLIP 모델 사용)
- $P_t \in \mathbb{R}^{N_{t-1} \times N_{t-1}}$: 유사도 행렬

**임계값 기반 텍스트 쿼리 필터링:**

$$h^i_t = \sigma(P^{ii}_t) = \begin{cases} h^i_{t-1}, & \text{if } P^{ii}_t \geq \theta \\ \text{NULL}, & \text{if } P^{ii}_t < \theta \end{cases} \tag{6}$$

- $P^{ii}_t$: 정규화된 유사도 행렬의 $i$번째 대각 원소
- $\theta$: 수동 설정 임계값
- **의미**: 마스크와 텍스트 쿼리 간의 정렬(alignment) 점수가 낮은 쿼리를 반복적으로 제거

**CLIP-ES의 어피니티 행렬(Affinity Matrix) 계산:**

$$A = \frac{D + D^T}{2}, \quad \text{where } D = \text{Sinkhorn}(W_{attn}) \tag{7}$$

**CAA(Class-Aware Affinity) 정제:**

$$M^{aff}_c = B_c \odot A^t \cdot \text{vec}(M_c) \tag{8}$$

- $B_c \in \mathbb{R}^{1 \times hw}$: 클래스 $c$의 CAM으로부터 얻은 박스 마스크
- $\odot$: 하다마드 곱(Hadamard product)
- $t$: 정제 반복 횟수

#### 종료 조건

$$h_t == h_{t-1} \Rightarrow \text{Stop}$$

텍스트 쿼리 집합이 연속된 두 스텝 간에 동일해지면 재귀 과정을 종료하고 포스트 프로세싱을 적용합니다.

### 2.3 모델 구조

```
입력: 이미지 x₀ + 초기 텍스트 쿼리 집합 h₀
         ↓
┌─────────────────────────────────────────────┐
│           재귀 루프 (t = 1, 2, ...)          │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │      마스크 제안 생성기 f(·,·)        │    │
│  │  (CLIP ViT-B/16 + CLIP-ES/gradCAM) │    │
│  │  → 배경 쿼리 추가하여 false positive │    │
│  │    억제                             │    │
│  └──────────────┬──────────────────────┘    │
│                 ↓ y_t (마스크 제안)          │
│  ┌─────────────────────────────────────┐    │
│  │      시각적 프롬프팅 v(·,·)           │    │
│  │  (Red Circle, Background Blur 등)   │    │
│  └──────────────┬──────────────────────┘    │
│                 ↓ x'_t (프롬프트된 이미지)   │
│  ┌─────────────────────────────────────┐    │
│  │      마스크 분류기 g(·,·)             │    │
│  │      (CLIP ViT-L/14)               │    │
│  │  → 유사도 행렬 P_t 계산              │    │
│  └──────────────┬──────────────────────┘    │
│                 ↓                           │
│  ┌─────────────────────────────────────┐    │
│  │      임계값 함수 σ(·)                │    │
│  │  → 낮은 점수 쿼리 제거 → h_t         │    │
│  └──────────────┬──────────────────────┘    │
│                 ↓                           │
│         h_t == h_{t-1}? → YES: Stop        │
└─────────────────────────────────────────────┘
         ↓
포스트 프로세싱: denseCRF (+ 선택적 SAM 앙상블)
         ↓
최종 마스크 출력
```

**핵심 구성 요소:**

| 구성 요소 | 상세 내용 |
|---|---|
| 마스크 제안 생성기 | CLIP ViT-B/16 + CLIP-ES (gradCAM + CAA) |
| 마스크 분류기 | CLIP ViT-L/14 (더 큰 모델로 정밀도 향상) |
| 시각적 프롬프트 | Red Circle + Background Blur 조합이 최고 성능 |
| 포스트 프로세싱 | denseCRF (필수) + HQ-SAM (선택) |
| 하이퍼파라미터 | $\eta$ (마스크 이진화), $\theta$ (쿼리 필터링), $\lambda$ (CLIP-ES) |

### 2.4 성능 향상

#### 제로샷 시맨틱 분할 (Zero-Shot Semantic Segmentation)

| 데이터셋 | 기존 최고 (w/o 추가 데이터) | CaR | 향상 |
|---|---|---|---|
| Pascal VOC-21 | 38.8 (MaskCLIP) | **67.6** | **+28.8 mIoU** |
| COCO Object | 20.6 (MaskCLIP) | **36.6** | **+16.0 mIoU** |
| Pascal Context-60 | 23.6 (MaskCLIP) | **30.5** | **+6.9 mIoU** |
| ADE-150 | 11.2 (ReCo) | **17.7** | **+6.5 mIoU** |

**파인튜닝 방법과의 비교:**

| 데이터셋 | 기존 최고 (w/ 추가 데이터) | CaR | 향상 |
|---|---|---|---|
| Pascal VOC-21 | 55.0 (TCL, 15M 데이터) | **67.6** | **+12.6 mIoU** |
| COCO Object | 32.0 (CLIPpy, 134M 데이터) | **36.6** | **+4.6 mIoU** |
| Pascal Context-60 | 30.4 (TCL, 15M 데이터) | **30.5** | **+0.1 mIoU** |

#### 참조 이미지 분할 (Referring Image Segmentation)

| 데이터셋 | GL CLIP (기존 SOTA) | CaR |
|---|---|---|
| RefCOCO val | 26.20 | **33.57** |
| RefCOCO testA | 24.94 | **35.36** (+10.42) |
| RefCOCO+ testA | 25.64 | **36.03** (+10.39) |
| RefCOCOg val(G) | 33.61 | **36.63** |

#### 재귀 구조의 효과 (Ablation)

| 재귀 사용 | CAM 방법 | mIoU |
|---|---|---|
| ✗ | CLIP-ES | 15.2 |
| ✓ | CLIP-ES | **67.6 (+52.4)** |
| ✓ | gradCAM | 41.1 |

재귀 구조 도입만으로 **52.4 mIoU** 향상이라는 획기적인 결과를 보여줍니다.

### 2.5 한계점

1. **VLM 성능에 종속**: CaR의 성능 상한선은 사전학습된 VLM(CLIP)의 능력에 의해 결정됩니다. 예를 들어, CLIP이 수평 뒤집기(horizontal flipping) 증강으로 훈련되어 "left"와 "right"를 구별하기 어렵습니다.
2. **소형 객체 처리의 한계**: gradCAM 기반 마스크 제안이 소형 객체에 대해 부정확한 경우가 있습니다. FPN(Feature Pyramid Network) 등의 모듈 추가가 필요합니다.
3. **Stuff 카테고리 민감도 부족**: CLIP의 사전학습 데이터에서 "stuff" 카테고리가 덜 등장하므로 Pascal Context의 stuff 클래스에 대한 성능이 상대적으로 낮습니다.
4. **하이퍼파라미터 의존성**: $\eta$, $\theta$, $\lambda$ 등의 임계값이 데이터셋마다 다르게 설정되어야 하며, 이는 실용성을 다소 제한합니다.
5. **SAM과의 불완전한 통합**: SAM이 경계를 정밀하게 만들어주지만, 매칭 알고리즘의 불일치로 인해 false negative/positive 예측이 발생할 수 있습니다.
6. **추론 속도**: CPU 기반 CRF 포함 시 $500 \times 500$ 이미지에 대해 950ms 소요 (GPU CRF는 약 5배 빠름).

---

## 3. 일반화 성능 향상 가능성

### 3.1 CLIP의 광대한 어휘 공간 상속

CaR의 가장 강력한 일반화 원천은 **파인튜닝 없이 CLIP의 전체 어휘 공간을 보존**한다는 점입니다. 파인튜닝 기반 방법들은 수십~수백 개의 카테고리로 어휘를 제한하지만, CaR은 CLIP이 학습한 수십억 개의 인터넷 이미지-텍스트 쌍에서 파생된 어휘를 그대로 활용합니다. 이를 통해:

- **픽션 캐릭터** (배트맨, 피카추, 이브이 등)
- **랜드마크** (Space Needle Seattle, 에펠탑 등)  
- **브랜드** (Pepsi, Coca Cola, Manchester United 등)
- **참조 표현** (the man holding a baby, the women chatting 등)

위와 같은 다양한 개념들에 대해 추가 학습 없이 분할이 가능합니다.

### 3.2 재귀적 자기-정제 메커니즘

재귀 프레임워크는 일반화에 직접적으로 기여합니다:

$$h_t = \sigma(P_t) \subseteq h_{t-1}$$

각 타임스텝에서 낮은 신뢰도의 텍스트 쿼리를 제거하면서, **이미지에 실제로 존재하는 객체에 대한 쿼리만 남기는 자동 필터링**이 이루어집니다. 이는 다양한 도메인의 이미지에서 존재하지 않는 개념에 대한 오예측을 줄여 일반화를 향상시킵니다.

### 3.3 확장 가능한 CLIP 백본

Ablation 연구(Table 3)에서 마스크 분류기의 백본을 ViT-B/16에서 ViT-L/14로 교체했을 때 Pascal VOC에서 약 **+13.5 mIoU** 향상이 관찰되었습니다. 이는:

$$\text{성능} \propto \text{백본 크기}$$

라는 확장 가능성(scalability)을 시사하며, EVA-CLIP, OpenCLIP 등 더 강력한 VLM으로 교체하면 일반화 성능이 추가로 향상될 수 있음을 의미합니다.

### 3.4 배경 쿼리(Background Query)를 통한 도메인 적응

배경 쿼리 전략은 false positive를 억제하여 다양한 장면 유형에서의 일반화를 강화합니다:

- **Terrestrial**: ground, land, grass, tree 등
- **Aquatic-Atmospheric**: sea, ocean, sky, cloud 등
- **Man-Made**: building, road, bridge 등

모든 배경 쿼리를 조합했을 때 최고 성능(67.6 mIoU)을 달성했으며, 이는 배경 정보의 다양성이 일반화에 중요함을 시사합니다.

Pascal Context에서는 thing 카테고리에 대해 stuff 카테고리를 배경 쿼리로, 반대로 stuff에 대해 thing을 배경 쿼리로 사용하는 **상호 배경 전략(Mutual Background Strategy)**을 도입하여 도메인 특화 일반화를 달성했습니다.

### 3.5 다중 도메인 확장성

- **이미지 분할**: Pascal VOC, COCO, Pascal Context, ADE (8개 벤치마크)
- **참조 이미지 분할**: RefCOCO, RefCOCO+, RefCOCOg, GRES
- **비디오 참조 분할**: Ref-DAVIS 2017 ( $\mathcal{J}, \mathcal{F}$ = 30.34)

단일 프레임워크로 이미지와 비디오 도메인 모두를 처리하는 범용성을 보여줍니다.

### 3.6 다양한 VLM과의 호환성

논문 저자들이 명시적으로 언급한 바와 같이:

> *"since our method is fundamentally compatible with various Vision-Language Models (VLMs), it presents an intriguing opportunity to investigate integration with other VLMs."*

CaR 프레임워크는 CLIP 외에도 EVA-CLIP, SigLIP, InternVL 등 다른 VLM과도 호환 가능하며, 이는 미래 VLM 발전과 함께 CaR의 일반화 성능도 자동으로 향상될 수 있는 구조임을 의미합니다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 주요 연구 흐름 비교

| 방법 | 연도 | 파인튜닝 | 추가 데이터 | VOC-21 mIoU | 특징 |
|---|---|---|---|---|---|
| GroupViT | 2022 | ✓ | 26M | 52.3 | 계층적 그룹화, 텍스트 약지도 |
| MaskCLIP | 2022 | ✗ | - | 38.8 | CLIP 인코더 수정 |
| CLIP-ES | 2023 | ✗ | - | ~15 (w/o recurrence) | gradCAM + CAA |
| SegCLIP | 2023 | ✓ | 3.4M | 52.6 | 패치 집계 |
| TCL | 2023 | ✓ | 15M | 55.0 | 텍스트-그라운드 마스크 생성 |
| OVSeg | 2023 | ✓ | (mask ann.) | - | 마스크 적응 CLIP |
| ReCo | 2022 | ✗ | - | 25.1 | 검색-공동분할 |
| **CaR** | **2023** | **✗** | **-** | **67.6** | **재귀적 정제, VLM 보존** |

### 4.2 패러다임별 비교 분석

#### (1) 마스크 어노테이션 파인튜닝 방법 vs. CaR

**OVSeg** (Liang et al., CVPR 2023)와 **Grounded SAM** (Liu et al., 2023)은 강력한 마스크 분할 성능을 보이지만, 파인튜닝에 사용된 카테고리 외의 개념(예: Pepsi, Coca Cola, 애니메이션 캐릭터)에 대해 실패합니다. CaR은 이러한 개념들을 성공적으로 분할합니다.

#### (2) 이미지-텍스트 파인튜닝 방법 vs. CaR

**TCL** (Cha et al., CVPR 2023)은 15M 이미지-텍스트 쌍으로 파인튜닝하여 Pascal VOC에서 55.0 mIoU를 달성하지만, CaR은 **추가 데이터 없이** 67.6 mIoU로 초과 달성합니다. **CLIPpy** (Ranasinghe et al., 2022)는 134M 데이터를 활용하지만 CaR이 COCO Object에서 4.6 mIoU 앞섭니다.

#### (3) 훈련 불필요 방법 vs. CaR

**MaskCLIP** (Zhou et al., ECCV 2022)은 CLIP 이미지 인코더를 수정하여 밀집 예측을 수행하지만 Pascal VOC에서 38.8 mIoU에 그칩니다. **ReCo** (Shin et al., NeurIPS 2022)는 검색-공동분할 방법으로 25.1 mIoU를 달성합니다. CaR은 같은 훈련 불필요 설정에서 **67.6 mIoU로 +28.8 mIoU** 향상을 달성합니다.

#### (4) SAM 통합 방법 vs. CaR

**SAMCLIP** (Wang et al., 2023)은 CC15M+YFCC+IN21k (41M)으로 파인튜닝하고 SAM을 통합하여 Pascal VOC 60.6, Pascal Context 29.2 mIoU를 달성합니다. **CaR+SAM**은 추가 훈련 없이 Pascal VOC **70.2**, Pascal Context **31.1** mIoU를 달성하여 각각 **+9.6, +1.9** mIoU 우세합니다.

### 4.3 기술적 혁신 비교

| 측면 | 기존 방법들 | CaR의 혁신 |
|---|---|---|
| 반복적 정제 | 지도학습 기반 (Cascade RCNN, DETR) | **훈련 없는 재귀 정제** |
| 텍스트-비전 정렬 | 1회성 정렬 | **타임스텝별 점진적 정렬** |
| 어휘 공간 | 파인튜닝으로 제한 | **CLIP 전체 어휘 보존** |
| 시각적 프롬프팅 | 마스크 블랙아웃 (일반적) | **Circle + Blur 조합 최적화** |

---

## 5. 앞으로의 연구에 미치는 영향과 고려 사항

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 훈련 없는 오픈-어휘 분할 패러다임의 확립

CaR은 "파인튜닝 없이도 파인튜닝 방법을 능가할 수 있다"는 것을 실증적으로 보여줌으로써, **사전학습 VLM의 지식을 최대한 활용하는 추론 시 정제(inference-time refinement)** 패러다임의 가능성을 열었습니다. 이는 향후 오픈-어휘 분할 연구의 새로운 기준점이 될 것입니다.

#### (2) VLM-기반 반복 추론의 가능성 입증

RNN 아이디어를 VLM 추론에 적용하는 개념은 다른 비전-언어 과제에도 확장될 수 있습니다. 예를 들어, 오픈-어휘 객체 감지, 비디오 이해, 3D 장면 이해 등에서 유사한 반복 정제 접근법을 적용할 수 있습니다.

#### (3) 의사 레이블(Pseudo-label) 생성 도구로서의 활용

논문에서 명시한 바와 같이, CaR은 다른 오픈-어휘 분할기를 위한 **의사 레이블 생성 도구**로 활용될 수 있습니다. 이는 새로운 도메인에 대한 데이터 효율적 학습 파이프라인을 가능하게 합니다.

#### (4) 더 강력한 VLM과의 시너지

GPT-4V, LLaVA, InternVL 등 멀티모달 대형 언어 모델(MLLM)의 급속한 발전과 함께, CaR 프레임워크는 이러한 더 강력한 VLM을 백본으로 활용하면 자동적으로 성능이 향상될 수 있는 미래 지향적 구조입니다.

#### (5) 비디오 이해 및 로보틱스 분야 확장

CaR의 제로샷 비디오 참조 분할 (Ref-DAVIS 2017, J, F = 30.34) 결과는 로보틱스, 자율주행, 의료 영상 분석 등 다양한 실용적 응용 분야에서의 활용 가능성을 시사합니다.

### 5.2 앞으로 연구 시 고려할 점

#### (1) 더 강력한 마스크 제안 생성기 탐색

현재 CaR은 gradCAM/CLIP-ES 기반의 마스크 제안을 사용하며, 소형 객체 처리에 한계가 있습니다. 향후 연구에서는:
- **FPN(Feature Pyramid Network)** 통합으로 다중 스케일 객체 처리
- **SAM2** 등 최신 세그멘터와의 긴밀한 통합
- **Point-based prompting** 전략 개발이 필요합니다.

#### (2) 하이퍼파라미터 자동화

$\eta$, $\theta$, $\lambda$ 등의 임계값이 데이터셋마다 다르게 설정되어야 하는 한계를 극복하기 위해:
- **적응적 임계값 메커니즘** (예: 이미지 복잡도에 따른 동적 조정)
- **메타-학습(meta-learning)** 기반 하이퍼파라미터 최적화 연구가 필요합니다.

#### (3) 계산 효율성 개선

추론 속도 문제 (950ms/이미지)를 해결하기 위해:
- GPU 기반 CRF 활용 (약 5배 속도 향상)
- **지식 증류(knowledge distillation)**를 통한 경량화 모델 개발
- 재귀 종료 조건의 조기 예측 메커니즘 도입이 필요합니다.

#### (4) 공간적 관계 이해 개선

CLIP의 수평 뒤집기 증강으로 인한 "left/right" 구별 문제를 해결하기 위해:
- 공간 인식 사전학습(spatially-aware pretraining) 전략 탐구
- 위치 인코딩을 명시적으로 활용하는 VLM(예: Grounding DINO) 통합이 필요합니다.

#### (5) 공정한 평가 프로토콜 수립

오픈-어휘 분할 평가에서 "zero-shot"의 정의가 방법마다 다르게 사용됩니다 (예: seen/unseen 분리, 어노테이션 유형). 향후 연구는:
- **통일된 벤치마크 프로토콜** 수립
- **어휘 다양성 지표** 개발 (단순 mIoU를 넘어)
- **장미래 객체(unseen concept)** 에 대한 체계적 평가가 필요합니다.

#### (6) 다양한 VLM 백본 통합 실험

CaR 프레임워크의 범용성을 활용하여:
- EVA-CLIP, SigLIP, OpenCLIP 등 다양한 CLIP 변종
- 멀티모달 LLM (LLaVA, InstructBLIP 등)
- 도메인 특화 VLM (의료, 위성 이미지 등)과의 통합 연구가 필요합니다.

#### (7) Stuff 카테고리 처리 개선

현재 CaR은 thing 카테고리에 강하지만 stuff 카테고리(sky, ground 등)에 상대적으로 약합니다. 이를 위해:
- 상호 배경 전략(Mutual Background Strategy)의 일반화
- Panoptic segmentation으로의 확장
- Stuff 카테고리에 대한 특화된 CAM 방법 개발이 필요합니다.

---

## 참고 자료

**주요 논문 (직접 제공된 PDF):**
- **Sun, S., Li, R., Torr, P., Gu, X., & Li, S. (2024). "CLIP as RNN: Segment Countless Visual Concepts without Training Endeavor." arXiv:2312.07661v3. University of Oxford & Google Research.**

**논문 내 인용 주요 참고문헌:**
- Radford, A., et al. (2021). "Learning transferable visual models from natural language supervision." *ICML 2021.* (CLIP)
- Zhou, C., et al. (2022). "Extract free dense labels from clip." *ECCV 2022.* (MaskCLIP)
- Lin, Y., et al. (2023). "CLIP is also an efficient segmenter." *CVPR 2023.* (CLIP-ES)
- Xu, J., et al. (2022). "GroupViT: Semantic segmentation emerges from text supervision." *CVPR 2022.*
- Cha, J., et al. (2023). "Learning to generate text-grounded mask for open-world semantic segmentation." *CVPR 2023.* (TCL)
- Liang, F., et al. (2023). "Open-vocabulary semantic segmentation with mask-adapted CLIP." *CVPR 2023.* (OVSeg)
- Kirillov, A., et al. (2023). "Segment anything." *arXiv:2304.02643.* (SAM)
- Ke, L., et al. (2023). "Segment anything in high quality." *arXiv:2306.01567.* (HQ-SAM)
- Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual explanations from deep networks via gradient-based localization." *ICCV 2017.*
- Shin, G., et al. (2022). "ReCo: Retrieve and co-segment for zero-shot transfer." *NeurIPS 2022.*
- Krähenbühl, P., & Koltun, V. (2011). "Efficient inference in fully connected CRFs with Gaussian edge potentials." *NeurIPS 2011.* (denseCRF)
- Ranasinghe, K., et al. (2022). "Perceptual grouping in vision-language models." *arXiv:2210.09996.* (CLIPpy)
- Yu, S., et al. (2023). "Zero-shot referring image segmentation with global-local context features." *CVPR 2023.* (GL CLIP)
- Wang, H., et al. (2023). "SAM-CLIP: Merging vision foundation models towards semantic and spatial understanding." *arXiv:2310.15308.* (SAMCLIP)
- Liu, S., et al. (2023). "Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection." *arXiv:2303.05499.* (Grounded SAM)
