# BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing

---

## 1. 핵심 주장 및 주요 기여 요약

Subject-driven text-to-image generation 모델은 텍스트 프롬프트에 기반하여 입력 주체(subject)의 새로운 표현(rendition)을 생성하는데, 기존 모델은 긴 파인튜닝 시간과 주체 충실도(subject fidelity) 유지의 어려움을 겪고 있었다. 이를 극복하기 위해 BLIP-Diffusion은 주체 이미지와 텍스트 프롬프트를 동시에 입력으로 받는 멀티모달 제어가 가능한 새로운 subject-driven 이미지 생성 모델을 제안한다.

### 주요 기여:
1. **사전학습된 주체 표현(Pre-trained Subject Representation)**: 기존 subject-driven 생성 모델과 달리, BLIP-Diffusion은 주체 표현을 제공하기 위해 사전학습된 새로운 멀티모달 인코더를 도입하고, BLIP-2를 따라 텍스트와 정렬된 시각적 표현을 생성하도록 사전학습한 뒤, 확산 모델이 이러한 시각적 표현을 활용하여 새로운 주체 표현을 생성하도록 하는 subject representation learning task를 설계하였다.

2. **Zero-shot 생성 및 효율적 파인튜닝**: DreamBooth 등 이전 방법과 비교하여, zero-shot subject-driven 생성을 가능하게 하며, 맞춤형 주체에 대해 최대 20배 빠른 파인튜닝 효율성을 달성한다.

3. **기존 기술과의 유연한 통합**: ControlNet, prompt-to-prompt 등 기존 기술과 유연하게 결합하여 새로운 subject-driven 생성 및 편집 응용을 가능하게 한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

대부분의 사전학습된 text-to-image 모델은 멀티모달 제어(이미지와 텍스트를 동시에 제어 입력으로 사용)를 기본적으로 지원하지 않아, 텍스트 공간과 정렬되면서도 높은 충실도로 주체 시각 정보를 캡처하는 주체 표현을 학습하기 어렵다.

Textual Inversion은 placeholder 텍스트 임베딩을 사용하여 시각 개념을 표현하고 이를 최적화하는 방식이며, DreamBooth는 유사한 방법론에 추가적으로 확산 모델 자체를 파인튜닝하여 더 나은 표현력과 주체 충실도를 달성하지만, 두 방법 모두 새로운 주체마다 긴 파인튜닝 시간이 필요하여 확장이 어렵다는 단점이 있다.

### 2.2 제안하는 방법 및 모델 구조

#### 전체 구조

모델은 비전-언어 인코더(BLIP-2)와 잠재 확산 모델(Stable Diffusion)을 기반으로 구축된다. BLIP-2 인코더는 주체 이미지와 해당 카테고리 텍스트를 입력으로 받아, 텍스트와 정렬된 주체 표현을 출력한다. 이후 주체 표현을 프롬프트 임베딩에 삽입(infix)하여 잠재 확산 모델의 subject-driven 이미지 생성 및 편집을 안내한다.

구체적으로, BLIP-2 멀티모달 인코더의 출력을 확산 모델의 텍스트 인코더 입력에 연결한다.

#### 2단계 사전학습 전략 (Two-Stage Pre-training)

**Stage 1: 멀티모달 표현 학습 (Multimodal Representation Learning)**

첫 번째 사전학습 단계에서는 BLIP-2를 따라 입력 이미지로부터 텍스트와 정렬된 시각 특징을 생성하도록 멀티모달 표현 학습을 수행한다. 이 과정에서 ITC(Image-Text Contrastive learning) 손실, ITG(Image-grounded Text Generation) 손실, ITM(Image-Text Matching) 손실의 세 가지 비전-언어 사전학습 목표를 공동으로 학습한다. 이를 통해 모델은 다양한 시각적·텍스트 개념을 학습할 수 있다.

멀티모달 표현 학습에는 BLIP-2와 동일한 사전학습 데이터를 사용하며, 총 1.29억 개의 이미지(COCO, Visual Genome, CC3M, CC12M, SBU, LAION400M 115M 이미지)를 활용한다.

**Stage 2: 주체 표현 학습 (Subject Representation Learning) — Prompted Context Generation**

두 번째 사전학습 단계에서는 "prompted context generation"이라는 주체 표현 학습 과제를 설계하여, 확산 모델이 입력 시각 특징을 기반으로 새로운 주체 표현을 생성하도록 학습한다. 이를 위해 동일한 주체가 다른 맥락에서 나타나는 입력-목표 이미지 쌍을 구축하고, 주체를 랜덤 배경과 합성하여 입력 이미지를 생성한다. 사전학습 중에 합성 입력 이미지와 주체 클래스 레이블을 BLIP-2에 통과시켜 멀티모달 임베딩을 주체 표현으로 획득하며, 이 주체 표현은 텍스트 프롬프트와 결합되어 목표 이미지의 생성을 안내한다.

#### 핵심 수식

**Latent Diffusion Model의 기본 학습 목표:**

잠재 확산 모델은 $z$에 $t$ 스텝 동안 점진적으로 노이즈를 추가하여 다음 목표를 최적화한다:

$$\mathcal{L} = \mathbb{E}_{z, \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right]$$

여기서:
- $\epsilon$: 추가된 노이즈
- $\epsilon_\theta(z_t, t, c)$: 시간 스텝 $t$에서 신경망 $\epsilon_\theta$가 예측하는 노이즈
- $c$: 텍스트 프롬프트 조건
- $z_t$: 시간 스텝 $t$에서의 잠재 변수

**BLIP-Diffusion의 확장된 조건부 생성:**

텍스트 프롬프트 외에 주체 표현도 조건으로 사용하여, 멀티모달 조건을 갖춘 이미지 생성 아키텍처를 구현한다. 즉:

$$\mathcal{L}_{BLIP\text{-}Diff} = \mathbb{E}_{z, \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c_{text}, c_{subj}) \|^2 \right]$$

여기서 $c_{subj}$는 BLIP-2 멀티모달 인코더가 생성한 **subject prompt embedding**이다.

**BLIP-2의 3가지 사전학습 목표 (Stage 1):**

BLIP-2 사전학습의 세 가지 비전-언어 사전학습 목표:

1. **ITC (Image-Text Contrastive Learning)**:

$$\mathcal{L}_{ITC} = -\mathbb{E} \left[ \log \frac{\exp(\text{sim}(v, t^+)/\tau)}{\sum_{t'}\exp(\text{sim}(v, t')/\tau)} \right]$$

2. **ITG (Image-grounded Text Generation)**:

$$\mathcal{L}_{ITG} = -\sum_{i} \log P(w_i | w_{<i}, v)$$

3. **ITM (Image-Text Matching)**:

$$\mathcal{L}_{ITM} = -\mathbb{E} \left[ y \log p + (1-y) \log(1-p) \right]$$

여기서 $v$는 시각 표현, $t$는 텍스트 표현, $\tau$는 온도 파라미터, $w_i$는 텍스트 토큰이다.

#### 모델 아키텍처 상세

| 구성 요소 | 역할 |
|-----------|------|
| **Image Encoder (Frozen)** | 사전학습 중 이미지 인코더는 고정(frozen) 상태를 유지한다. |
| **BLIP-2 Q-Former** | 학습 가능한 쿼리 벡터를 통해 텍스트와 정렬된 시각 특징을 추출 |
| **Text Encoder (Diffusion)** | Subject embedding과 text embedding의 상호작용을 위해 학습 |
| **U-Net (Latent Diffusion)** | 노이즈 예측 네트워크, subject + text 조건부 생성 수행 |
| **Subject Prompt Embedding** | BLIP-2가 생성한 subject prompt embedding을 잠재 확산 모델이 사용하여 출력 주체 이미지를 생성한다. |

#### ControlNet 통합 및 이미지 편집

ControlNet 통합 시, 사전학습된 ControlNet의 U-Net을 BLIP-Diffusion의 U-Net에 잔차(residual) 연결로 부착한다. 이를 통해 엣지 맵, 깊이 맵 등의 구조적 조건과 주체 단서를 동시에 고려할 수 있으며, 원래 잠재 확산 모델의 아키텍처를 계승하므로 추가 학습 없이 오프더셸프(off-the-shelf) ControlNet과 통합할 수 있다.

Subject-driven 이미지 편집에서는 어텐션 맵을 사용하여 편집 영역을 자동 추출하고, 각 디노이징 스텝에서 추출된 편집 마스크에 기반하여 디노이징 잠재를 혼합한다. 비편집 영역의 잠재는 원래 생성에서, 편집 영역의 잠재는 subject-driven 생성에서 가져와 주체 특정 시각 정보를 포함하면서 비편집 영역을 보존한 편집 이미지를 얻는다.

### 2.3 성능 향상

#### 정량적 평가 (DreamBench)

DreamBooth 데이터셋(30개 주체, 각 4-7개 이미지)에서 DINO, CLIP-I(주체 정렬), CLIP-T(이미지-텍스트 정렬)을 사용하여 평가하며, 각 텍스트 프롬프트마다 4개의 이미지를 생성하여 총 3,000개의 이미지를 사용한다.

전체적인 결과는 정성적 결과와 일관되며, BLIP-Diffusion이 Textual Inversion과 Re-Imagen보다 우수하고, DreamBooth와 비슷한 성능을 보이면서 파인튜닝 노력은 훨씬 적게 필요하다.

| 메트릭 | 의미 | BLIP-Diffusion 성과 |
|--------|------|---------------------|
| **DINO** | Subject 시각적 유사도 | Textual Inversion, Re-Imagen 대비 우수 |
| **CLIP-I** | Subject-이미지 정렬 | DreamBooth와 유사 수준 |
| **CLIP-T** | 텍스트-이미지 정렬 | 경쟁력 있는 성능 |

**Ablation 결과 (250K steps):**

BLIP-Diffusion (250K steps): DINO=0.566, CLIP-I=0.773, CLIP-T=0.299이며, 멀티모달 사전학습 없이(w/o multimodal pre-training)는 DINO=0.521로 하락하였다.

**파인튜닝 효율성:**

BLIP-Diffusion은 보통 40-120 스텝만으로 파인튜닝이 가능하며, 이는 이전 연구(Textual Inversion, DreamBooth)보다 최대 20배 더 효율적이다.

DreamBench에서 평균 76 스텝으로 파인튜닝이 완료되며, 이는 단일 A100에서 약 40초 정도 소요된다.

#### 학습 설정

기반 확산 모델로 Stable Diffusion v1-5를 사용하며, 총 배치 크기 16, 상수 학습률 2e-6, AdamW 옵티마이저를 사용하여 500K 스텝 동안 학습하고, 16개의 A100 40GB GPU에서 6일이 소요된다.

### 2.4 한계

모델의 한계로는 잘못된 맥락 합성, 학습 세트에 대한 과적합 등이 있으며, 기본 확산 모델의 약점을 상속받아 텍스트 프롬프트 이해 및 세밀한 구성 관계 파악에 실패할 수 있다. 그러나 이 기법은 향후 확산 모델의 발전을 그대로 수용할 수 있는 범용적 특성을 가진다.

특히 제공된 주체 이미지의 시각적 다양성이 제한적일 때, 모델이 텍스트 프롬프트와 무관하게 목표 입력에 과적합(overfitting)되는 문제가 발생할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

BLIP-Diffusion의 일반화 성능과 관련된 핵심 설계 결정 및 향상 가능성은 다음과 같다:

### 3.1 일반화를 가능하게 하는 핵심 요소

**① 대규모 멀티모달 사전학습:**
범용 이미지-텍스트 쌍 데이터에 대한 멀티모달 표현 학습을 수행하여 모델이 다양한 시각적·텍스트 개념을 학습할 수 있게 한다. 이는 129M 이미지에 달하는 대규모 데이터셋을 활용한다.

**② Subject-Generic Pre-training:**
두 단계 사전학습 전략을 통해 범용적 주체 표현을 학습하며, 첫 번째 단계에서 텍스트와 정렬된 시각 특징을, 두 번째 단계에서 확산 모델이 입력 시각 특징을 기반으로 새로운 주체 표현을 생성하도록 학습한다.

**③ Ablation을 통해 확인된 중요성:**
멀티모달 표현 학습(Stage 1)을 수행하는 것이 핵심적이며, 이는 주체 임베딩과 텍스트 프롬프트 임베딩 간의 표현 갭(representation gap)을 연결해준다.

### 3.2 일반화 향상 가능성

**① 더 강력한 기반 모델 활용:**
제안된 기법은 향후 확산 모델의 발전을 수용할 수 있는 범용적(generic) 특성을 가진다. 따라서 SDXL, Stable Diffusion 3, FLUX 등 더 강력한 확산 모델과의 교체를 통해 일반화 성능의 추가 향상이 가능하다.

**② Zero-shot 능력 확장:**
사전학습된 주체 표현을 활용한 zero-shot 이미지 조작, 주체 기반 스타일 전이(style transfer) 및 주체 보간(subject interpolation) 등의 응용이 가능하다.

**③ 학습 데이터 확장:**
현재 주체 표현 학습에는 OpenImage-V6의 292K 이미지 서브셋을 사용하며, 인간 관련 주체는 제외되어 있다. 학습 데이터의 규모와 다양성을 확장하면 일반화 성능이 향상될 수 있다.

**④ 기존 도구와의 플러그인 호환성:**
기본 확산 모델에 대한 최소한의 아키텍처 변경으로 주체 표현을 주입하면서, 동시에 기본 모델링 역량을 대부분 계승하는 효과적인 솔루션을 제공한다. 이 모듈화된 설계 덕분에 다양한 다운스트림 응용으로의 일반화가 용이하다.

### 3.3 일반화 제한 요인

| 제한 요인 | 설명 |
|-----------|------|
| 학습 도메인 편향 | OpenImage 데이터셋 기반이므로 특정 도메인(의료, 위성 등)으로의 일반화에 한계 |
| 인간 주체 제외 | 사전학습에서 인간 주체 제거로 인물 관련 생성 성능 제한 |
| 기반 모델 의존성 | Stable Diffusion v1-5의 한계를 상속 |
| 복잡한 구성 관계 | 텍스트에 포함된 세밀한 공간적/구성적 관계 이해 부족 |

---

## 4. 논문의 영향 및 향후 연구 고려사항

### 4.1 연구에 미치는 영향

BLIP-Diffusion은 멀티모달 제어가 가능한 기초(foundation) text-to-image 생성 모델을 구축하기 위한 중요한 단계로 간주된다.

**① 사전학습 패러다임의 전환**: Subject-driven 생성에 사전학습된 주체 표현을 도입함으로써, "주체당 최적화(per-subject optimization)" 패러다임에서 "사전학습 + 경량 파인튜닝" 패러다임으로의 전환을 촉진하였다.

**② 멀티모달 제어의 표준화**: 이미지와 텍스트를 동시 조건으로 사용하는 접근법이 이후 IP-Adapter, PhotoMaker, SSR-Encoder 등 다양한 후속 연구에 영감을 주었다.

**③ 모듈형 아키텍처의 실용성 입증**: 이러한 응용들은 BLIP-Diffusion을 멀티모달 제어가 가능한 기초 text-to-image 생성 모델로 사용하는 잠재력을 보여준다.

### 4.2 향후 연구 시 고려할 점

1. **더 강력한 기반 모델과의 통합**: SDXL, SD3, FLUX 등 최신 확산 모델로의 업그레이드
2. **다중 주체(Multi-subject) 지원 확장**: 현재 단일 주체 중심이므로 복수 주체의 동시 제어 메커니즘 연구 필요
3. **인간 주체로의 확장**: 현재 사전학습에서 제외된 인간 도메인으로의 확장
4. **더 정밀한 주체-배경 분리**: 배경 정보 최소화와 주체 정보 극대화의 균형
5. **평가 체계 개선**: DINO, CLIP-I, CLIP-T 메트릭을 공동으로 고려해야 하며, 단순히 학습 세트를 복사하는 모델은 높은 DINO/CLIP-I와 낮은 CLIP-T를, 주체 지식 없는 모델은 높은 CLIP-T와 낮은 주체 정렬 점수를 보이므로, 두 경우 모두 바람직하지 않다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근법 | BLIP-Diffusion과의 차이 |
|------|------|------------|------------------------|
| **Textual Inversion** (Gal et al.) | 2022 | placeholder 텍스트 임베딩을 최적화하여 시각 개념을 표현 | 주체당 ~3,000 스텝 필요, zero-shot 불가 |
| **DreamBooth** (Ruiz et al.) | 2022 | Textual Inversion과 유사하나 확산 모델 자체도 추가 파인튜닝 | ~800 스텝 필요, 더 높은 충실도이나 느림 |
| **SuTI** (Chen et al.) | 2023 | 주체별 파인튜닝을 in-context learning으로 대체하며, 대규모 주체별 전문가 모델에서 도제 학습(apprenticeship learning)을 통해 학습 | 대규모 전문가 모델 학습 비용 높음, zero-shot 가능 |
| **IP-Adapter** (Ye et al.) | 2023 | 사전학습된 text-to-image 확산 모델에 이미지 프롬프트 기능을 부여하는 경량 어댑터로, 텍스트 특징과 이미지 특징에 대한 분리된 cross-attention 메커니즘을 핵심 설계로 사용 | 더 경량화되고 범용적, decoupled attention 사용 |
| **ELITE** (Wei et al.) | 2023 | 글로벌+로컬 매핑을 통한 인코더 기반 주체 표현 | BLIP-2 기반이 아닌 독자적 인코더 구조 |
| **PhotoMaker** (Li et al.) | 2023 | Stacked ID Embedding으로 현실적 인물 사진 커스터마이징 | 인물 특화, ID 임베딩 스태킹 방식 |
| **SSR-Encoder** | 2024 | 단일 또는 다중 참조 이미지에서 임의의 주체를 선택적으로 캡처하는 새로운 아키텍처로, 텍스트와 마스크를 포함한 다양한 쿼리 모달리티에 응답하며, 테스트 시간 파인튜닝이 불필요 | 다중 참조/다중 모달리티 쿼리 지원 |
| **ZipLoRA** (Shah et al.) | 2023 | LoRA 병합으로 임의의 주체+임의의 스타일 결합 | 주체와 스타일의 독립적 제어 가능 |
| **DreamBench++** (Peng et al.) | 2024 | 고급 멀티모달 GPT 모델로 자동화된 인간 정렬 벤치마크로, 프롬프트를 체계적으로 설계하여 GPT가 인간 정렬되고 자체 정렬되도록 함 | 평가 프레임워크, 생성 모델이 아님 |
| **BLIP3-o** (Salesforce) | 2025 | 자기회귀+확산 하이브리드 아키텍처의 체계적 탐색으로, CLIP 임베딩과 flow matching 손실이 더 빠른 학습 효율과 높은 품질 출력을 제공함을 입증 | BLIP 계보의 최신 통합 모델, 이해+생성 통합 |

### 패러다임별 비교 요약

```
┌─────────────────────────────────────────────────────────────────┐
│          Subject-Driven Generation 접근법 분류                    │
├─────────────────┬───────────────────┬───────────────────────────┤
│  최적화 기반      │  인코더 기반        │ 하이브리드/통합             │
│  (Per-subject)  │  (Feed-forward)   │                           │
├─────────────────┼───────────────────┼───────────────────────────┤
│ Textual Inv.    │ BLIP-Diffusion ★  │ BLIP3-o (2025)           │
│ DreamBooth      │ IP-Adapter        │ Diffusion Self-Distill.  │
│ Custom Diff.    │ SuTI              │                           │
│ SVDiff          │ ELITE             │                           │
│                 │ SSR-Encoder       │                           │
│                 │ PhotoMaker        │                           │
└─────────────────┴───────────────────┴───────────────────────────┘
         느림/고품질        빠름/유연          최신 트렌드
```

BLIP-Diffusion은 인코더 기반(feed-forward) 접근법의 선구적 모델로서, **사전학습된 주체 표현**이라는 개념을 통해 zero-shot 및 효율적 파인튜닝의 가능성을 열었으며, 이는 이후 IP-Adapter, SSR-Encoder 등 다양한 후속 연구의 토대가 되었다.

---

## 참고 자료 (References)

1. **[논문 원문]** Li, D., Li, J., & Hoi, S. C. H. (2023). "BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing." *NeurIPS 2023*. arXiv:2305.14720 — https://arxiv.org/abs/2305.14720
2. **[NeurIPS Proceedings]** https://proceedings.neurips.cc/paper_files/paper/2023/file/602e1a5de9c47df34cae39353a7f5bb1-Paper-Conference.pdf
3. **[프로젝트 페이지]** https://dxli94.github.io/BLIP-Diffusion-website/
4. **[Semantic Scholar]** https://www.semanticscholar.org/paper/BLIP-Diffusion-Li-Li/dc0c132b273456b288a785414db2fa72edf87b1a
5. **[OpenReview]** https://openreview.net/forum?id=g6We1SwaY9
6. **[HuggingFace Diffusers 문서]** https://huggingface.co/docs/diffusers/en/api/pipelines/blip_diffusion
7. **[HuggingFace 모델 카드]** https://huggingface.co/salesforce/blipdiffusion
8. **[ar5iv HTML 버전]** https://ar5iv.labs.arxiv.org/html/2305.14720
9. **[ResearchGate]** https://www.researchgate.net/publication/371009656
10. **[IP-Adapter]** Ye, H. et al. (2023). arXiv:2308.06721 — https://ip-adapter.github.io/
11. **[SuTI]** Chen, W. et al. (2023). arXiv:2304.00186 — https://open-vision-language.github.io/suti/
12. **[DreamBench++]** Peng, Y. et al. (2024). ICLR 2025 — https://dreambenchplus.github.io/
13. **[BLIP3-o]** Salesforce (2025) — https://www.salesforce.com/blog/blip3/
