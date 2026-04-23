
# ViewDiff: 3D-Consistent Image Generation with Text-to-Image Models

> **논문 정보:**
> - **제목:** ViewDiff: 3D-Consistent Image Generation with Text-to-Image Models
> - **저자:** Lukas Höllein, Aljaž Božič, Norman Müller, David Novotny, Hung-Yu Tseng, Christian Richardt, Michael Zollhöfer, Matthias Nießner
> - **소속:** Technical University of Munich & Meta
> - **학회:** CVPR 2024 (pp. 5043–5052)
> - **arXiv:** [2403.01807](https://arxiv.org/abs/2403.01807)
> - **GitHub:** [facebookresearch/ViewDiff](https://github.com/facebookresearch/ViewDiff)
> - **Project Page:** [lukashoel.github.io/ViewDiff](https://lukashoel.github.io/ViewDiff/)

---

## 1. 핵심 주장 및 주요 기여 요약

3D 에셋 생성은 텍스트 기반 2D 콘텐츠 생성의 최근 성공에 힘입어 폭발적인 관심을 받고 있다. 그러나 기존의 text-to-3D 방법들은 사전 학습된 text-to-image 확산 모델을 최적화 문제에 사용하거나 합성 데이터로 파인튜닝하는 방식을 택했으며, 이는 종종 배경 없이 비사실적인 3D 오브젝트를 생성하는 문제를 야기했다.

이에 대응하여, ViewDiff는 사전 학습된 text-to-image 모델을 prior로 활용하고, 실세계 데이터로부터 단일 denoising 과정 내에서 multi-view 이미지를 생성하는 방법을 제안한다. 구체적으로, 기존 text-to-image 모델의 U-Net 네트워크의 각 블록에 3D 볼륨 렌더링과 cross-frame-attention 레이어를 통합하며, 임의의 시점에서 더 3D-일관성 있는 이미지를 렌더링하는 오토리그레시브 생성을 설계한다. 이 모델은 실세계 오브젝트 데이터셋으로 훈련되어 진짜 배경(authentic surroundings) 속에서 다양한 고품질 형상과 텍스처의 인스턴스를 생성한다.

### 📌 핵심 기여 3가지

| 기여 항목 | 설명 |
|---|---|
| **① Cross-Frame-Attention** | 카메라 포즈(RT), 내부 파라미터(K), 강도(I)로 조건화된 뷰 간 어텐션 |
| **② Projection Layer** | 다중 뷰 피처로부터 3D 복셀 그리드 생성 후 NeRF 방식으로 렌더링 |
| **③ Autoregressive Generation** | 임의 시점에서 슬라이딩 윈도우 방식의 자기회귀적 뷰 확장 |

---

## 2. 상세 설명: 문제 → 방법 → 구조 → 성능 → 한계

---

### 2-1. 해결하고자 하는 문제

기존 text-to-3D 방법들은 사전 학습된 text-to-image 확산 모델을 최적화 문제에 사용하거나 합성 데이터로 파인튜닝하여 비사실적인 3D 오브젝트나 배경 없는 결과물을 생성하는 문제가 있었다. ViewDiff는 사전 학습된 text-to-image 모델을 prior로 활용하여 실세계 데이터로부터 단일 denoising 과정 내에서 multi-view 이미지를 생성하는 방법을 제시한다.

핵심 해결 과제는 두 가지이다:

1. **3D 일관성 부재:** 기존 2D T2I 모델은 각 뷰를 독립적으로 생성하여 뷰 간 기하학적·외관적 일관성을 보장하지 못함.
2. **합성 데이터 의존성:** 기존 방법들은 Objaverse 같은 합성 데이터셋에만 의존하여 실세계 배경과 조명을 재현하기 어려움.

---

### 2-2. 제안 방법 및 수식

#### (A) 기반 Diffusion 학습 수식

ViewDiff는 **Latent Diffusion Model (LDM)** 기반으로, 표준 DDPM 노이즈 스케줄을 따른다.

**포워드 프로세스 (Forward Process):**

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t}(1 - \beta_s)$는 누적 노이즈 스케일이다.

**디노이징 학습 목표 (Denoising Objective):**

$$\mathcal{L}_d = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]$$

여기서 $\epsilon_\theta$는 U-Net 기반 노이즈 예측 네트워크, $c$는 텍스트 또는 이미지 조건이다.

**Prior Preservation Loss:**

파인튜닝 시 2D prior를 보존하기 위해 DreamBooth 방식을 차용하여 다음 조합 손실을 사용한다:

$$\mathcal{L} = \mathcal{L}_d + \lambda \mathcal{L}_p$$

여기서 $\mathcal{L}_p$는 사전 학습된 T2I 모델의 prior를 보존하는 정규화 손실이고, $\lambda$는 가중치이다.

#### (B) Cross-Frame-Attention (포즈 조건화)

Self-attention을 카메라 포즈 $(RT)$, 내부 파라미터 $(K)$, 강도 $(I)$ 각각에 조건화된 cross-frame-attention으로 대체한다.

뷰포인트 정보를 통합하기 위해 conditioning 벡터 $z$가 모든 어텐션 레이어에 추가된다. $z$는 카메라 포즈 ($RT$ 행렬), 내부 파라미터 ($K$), 이미지 강도 ($I$)를 인코딩한다. 포즈와 내부 파라미터는 4D 벡터 $(z_1, z_2)$로 임베딩되고, 강도는 2D 벡터 (RGB 값의 평균과 분산, $z_3$)로 인코딩된다. Query(Q), Key(K), Value(V) 프로젝션은 LoRA(Low-Rank Adaptation)를 통해 conditioning 벡터를 포함하도록 수정된다.

수식으로 표현하면:

$$Q = W_Q h_i + s \cdot W'_Q [h_i; z]$$

여기서 $W'_Q$는 LoRA 선형 레이어, $s$는 스케일링 팩터(= 1로 설정), $h_i$는 i번째 뷰의 피처이다.

#### (C) Projection Layer (3D 볼륨 렌더링)

Projection Layer는 다중 뷰 피처로부터 3D 표현을 생성하고 이를 3D-일관성 있는 피처로 렌더링한다. 먼저 압축된 이미지 피처를 3D로 역투영하여 MLP로 공동 복셀 그리드에 집계한다. 그런 다음 3D CNN으로 복셀 그리드를 정제한다.

NeRF와 유사한 볼륨 렌더러가 그리드로부터 3D-일관성 있는 피처를 렌더링한다. 마지막으로 학습된 스케일 함수를 적용하여 피처 차원을 확장한다.

상세 파이프라인:

$$h_{1:N}^{in} \in \mathbb{R}^{C \times H \times W} \xrightarrow{\text{1×1 Conv}} \mathbb{R}^{C' \times H \times W} \xrightarrow{\text{Unproject}} \mathcal{V}^{(i)} \xrightarrow{\text{MLP Aggregate}} \mathcal{V} \xrightarrow{\text{3D CNN}} \tilde{\mathcal{V}} \xrightarrow{\text{NeRF Render}} h_{1:N}^{out}$$

Aggregator MLP는 per-view 가중치를 예측하여 가중 평균으로 각 뷰의 그리드를 단일 그리드로 병합한다. 이 MLP는 또한 타임스텝 임베딩과 함께 각 뷰의 복셀 그리드를 결합하기 위한 ray 방향 및 depth 인코딩을 포함한다.

볼륨 렌더링에서 그리드의 절반은 전경에, 나머지 절반은 배경에 할당되어 ray-marching 중 MERF의 배경 모델을 사용한다.

NeRF 스타일의 볼륨 렌더링 적분:

$$\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt, \quad T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds\right)$$

#### (D) Autoregressive Generation

전체 배치 $N = n_c + n_g$ 이미지를 조건부($n_c$)와 생성부($n_g$)로 분리한다. $n_c$개의 샘플은 입력으로 제공되는 이미지 및 카메라에 해당하며, 나머지 $n_g$개는 조건 이미지와 유사한 새로운 뷰를 생성해야 한다. 생성 파트는 가우시안 노이즈에서 시작하고, 다른 샘플들에는 노이즈가 없는 이미지를 제공한다.

---

### 2-3. 모델 구조

사전 학습된 text-to-image 모델을 3D 일관성 이미지 생성기로 변환하기 위해 multi-view supervision으로 파인튜닝한다. U-Net 아키텍처의 모든 블록에 새로운 레이어를 추가하여, 배치 내 multi-view 이미지 간 통신을 가능하게 하고 결합된 denoising 과정에서 3D-일관성 있는 이미지를 생성한다.

```
[U-Net Block 구조]
─────────────────────────────────────────────────
Input Feature h_i  (각 뷰마다)
     ↓
  ResNet Block (기존 유지)
     ↓
  Cross-Frame-Attention  ← 추가됨 (Pose RT, K, I 조건)
     ↓
  Projection Layer  ← 추가됨 (볼륨 렌더링 + 3D CNN)
     ↓
Output Feature h_i'
─────────────────────────────────────────────────
```

**훈련 세부 사항:**
CO3Dv2 데이터셋으로 훈련하며, 카테고리는 Teddybear, Hydrant, Apple, Donut를 선택한다. 카테고리당 500~1,000개의 오브젝트를 사용하며, 각 오브젝트는 256×256 해상도의 200장 이미지로 구성된다.

Pretrained latent diffusion text-to-image 모델을 기반으로 하며, U-Net만 파인튜닝하고 VAE 인코더·디코더는 고정(frozen)한다.

훈련은 2x A100 GPU, 60K 이터레이션, 배치 크기 64로 진행된다. 볼륨 렌더러 학습률은 0.005, 기타 레이어는 5e-5이며 AdamW 옵티마이저를 사용한다. 추론 시에는 RTX 3090 GPU에서 배치당 최대 30장 이미지를 생성하며, UniPC 샘플러와 10 denoising 스텝을 사용한다.

---

### 2-4. 성능 향상

기존 방법들과 비교했을 때, ViewDiff로 생성된 결과물은 일관성이 뛰어나며 우수한 시각적 품질을 보인다 (FID -30%, KID -37%).

정량적 지표로 PSNR, SSIM, LPIPS가 측정된다.

| 지표 | ViewDiff 개선 |
|---|---|
| **FID** | -30% 감소 (기존 대비) |
| **KID** | -37% 감소 (기존 대비) |
| **정성 평가** | 3D 일관성 + 실사 배경 유지 |

---

### 2-5. 한계점

몇 가지 한계가 존재한다. 첫째, 모델이 뷰 의존적 효과(예: 노출 변화)가 포함된 실세계 데이터셋으로 파인튜닝되었기 때문에, 서로 다른 시점에서 이러한 변형을 생성하는 것을 학습한다. 잠재적 해결책은 ControlNet을 통해 조명 조건을 추가하는 것이다.

둘째, 현재 연구는 오브젝트에 집중하고 있으나, 마찬가지로 대규모 장면 생성도 탐색될 수 있다.

추가적인 한계로는: 학습에 사용되는 현재 multi-view 데이터셋이 상대적으로 소규모이며 특정 장면 유형에 국한된 점(Dataset Constraints), 추론 시 여러 일관된 뷰를 생성하는 계산 비용이 높아 실시간 애플리케이션에서의 실용적 배포를 제한할 수 있는 점(Computational Complexity), 그리고 복잡하거나 미지의 3D 장면, 특히 복잡한 기하학적 구조나 새로운 구성에 얼마나 잘 일반화될지 불분명한 점(Generalization Ability)이 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화 성능과 직결된 핵심 요소 및 잠재적 향상 방향을 정리하면 다음과 같다.

### 3-1. 현재 일반화 능력의 원천

ViewDiff는 대형 2D text-to-image 모델의 표현력을 활용하고, 이 2D prior를 실세계 3D 데이터셋에서 파인튜닝하여 결합된 denoising 과정에서 다양한 multi-view 이미지를 생성한다.

즉, 수십억 개의 텍스트-이미지 쌍으로 학습된 **Stable Diffusion의 거대한 2D 시각 prior**를 그대로 승계하는 것이 강력한 일반화의 원천이다.

Lukas(저자)에 따르면, 사전 학습된 text-to-image 모델은 수십억 개의 텍스트-이미지 쌍으로 학습되었기 때문에 강력하다.

2D prior를 유지하는 핵심 과제는 텍스트 프롬프트 제어를 원하면서도 더 작은 3D 데이터셋에서 파인튜닝한다는 점이며, 이를 위해 DreamBooth 논문의 prior preservation 데이터셋 파인튜닝 트릭을 사용한다.

### 3-2. 일반화 성능 향상의 주요 방향

| 방향 | 설명 | 현재 상태 |
|---|---|---|
| **① 더 많은 카테고리로 확장** | CO3D의 4개 카테고리에서 전체 카테고리/대규모 장면으로 확장 | 미래 연구 과제 |
| **② 합성+실세계 혼합 훈련** | Objaverse(합성) + CO3D(실세계) 병행 훈련 | 미개척 |
| **③ 더 강력한 T2I 기반 모델** | SDXL, FLUX 등 최신 T2I 모델로의 이전 | 미래 연구 과제 |
| **④ 장면 스케일 확장** | 오브젝트 중심 → 실내/외 전체 장면 | 논문에서 한계로 인정 |
| **⑤ 조명 조건 제어** | ControlNet 연동으로 뷰 의존 조명 문제 해결 | ControlNet 제안됨 |

이 접근법은 CO3D와 같은 실세계 3D 데이터셋에서 T2I 모델을 파인튜닝하는 길을 열어주면서도, 사전 학습된 가중치에 인코딩된 대형 2D prior의 혜택을 받는다.

### 3-3. 일반화를 위한 훈련 전략

Projection layer에서 마지막 이미지를 복셀 그리드 구성에서 제외하는 방식으로, 새로운 뷰에서 렌더링 가능한 3D 표현을 학습하도록 강제한다. 조건 없는 생성(unconditional)과 이미지 조건부 생성(image-conditional)을 번갈아 훈련한다.

이 전략은 모델이 훈련 중 본 적 없는 새로운 시점(Novel View)에서도 일관된 출력을 생성할 수 있도록 유도한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 연구적 영향

**(A) 2D Prior 재활용 패러다임의 확립**

ViewDiff는 거대한 2D T2I 모델의 가중치를 그대로 상속하면서 최소한의 3D-aware 레이어만 추가함으로써, **"2D prior → 3D 생성"** 의 효율적 경로를 제시했다. 이는 이후 CAT3D, MultiDiff, SplatDiff 등의 연구에 직접적인 영향을 미쳤다.

ViewDiff는 이 분야에서 중요한 발전을 나타내며, text-to-3D 생성 분야의 추가 연구와 개발의 길을 열어준다.

**(B) 실세계 3D 데이터셋 활용의 중요성 부각**

ViewDiff 접근법은 2D text-to-image 모델로부터 3D-일관성 있는 콘텐츠를 생성하는 도전을 해결하는 중요한 단계를 나타낸다. 훈련 중 3D 장면의 다중 뷰를 활용함으로써, 모델은 설명된 콘텐츠의 내재적 3D 구조를 학습하여 더 사실적이고 몰입감 있는 이미지 생성을 가능하게 한다.

**(C) NeRF/3DGS 파이프라인과의 연결**

생성된 이미지로부터 NeRF를 훈련하는 쉬운 방법을 제공하며, 부드러운 렌더링 생성 시 표준 NeRF 규약의 transforms.json 파일을 저장하고 이를 Instant-NGP나 NeRFStudio 같은 표준 NeRF 프레임워크에 활용할 수 있다.

---

### 4-2. 연구 시 고려할 점

#### ① 데이터 다양성 확보

현재 훈련에 사용된 multi-view 데이터셋은 상대적으로 소규모이며 특정 장면 유형에 국한된다. 따라서 후속 연구에서는 더 광범위한 카테고리와 대규모의 실세계 데이터셋 구성이 필요하다.

#### ② 오토리그레시브 드리프트 문제

오토리그레시브 이미지 생성에 의존하는 접근법은 드리프트와 오류 누적에 취약하다. ViewDiff의 슬라이딩 윈도우 방식 오토리그레시브 생성 역시 장기 시퀀스에서 점진적 품질 저하 문제를 가질 수 있으므로, 이를 보완하는 **공동 생성(joint generation)** 전략 연구가 필요하다.

#### ③ 계산 효율성

추론 시 여러 일관된 뷰를 생성하는 것은 계산 비용이 높아 실시간 애플리케이션에서의 실용적 배포를 제한할 수 있다. 경량화 및 inference 최적화 (예: diffusion 스텝 수 감소, 모델 distillation)가 중요 과제다.

#### ④ 뷰 의존 조명 처리

Flickering(깜빡임)은 조명 차이로 인해 발생하며, 더 나은 데이터로 줄일 수 있다. 향후에는 조명 조건을 명시적으로 분리하거나 ControlNet을 통해 제어하는 연구가 유망하다.

#### ⑤ 장면 스케일로의 확장

현재는 오브젝트에 집중하지만, 대규모 데이터셋에서의 장면 스케일 생성도 탐색할 수 있다. 실내·외 장면으로 확장 시 카메라 궤적의 다양성과 폐색(occlusion) 처리 등 새로운 도전이 등장한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

Denoising diffusion 모델은 데이터 분포 모델링과 고품질 샘플 합성에서 큰 성공을 거두었다. 2D 이미지 도메인에서 최첨단 상태를 달성한 이후, 연구자들은 3D 데이터 생성에 diffusion 모델을 활용하는 방법을 탐색하기 시작했다.

| 논문 | 연도 | 핵심 방법 | 강점 | 약점 | ViewDiff와의 차이 |
|---|---|---|---|---|---|
| **DreamFusion** | 2022 | SDS Loss + NeRF | 텍스트→3D 최초 개념 | 느린 최적화, Janus 문제 | ViewDiff는 피드포워드 방식 |
| **Zero-1-to-3** | 2023 | 카메라 포즈 조건부 T2I | Zero-shot NVS | 뷰 간 일관성 부족 | ViewDiff는 결합 denoising |
| **SyncDreamer** | 2023 | 3D-aware feature attention | 다중 뷰 동기화 | 합성 데이터 의존 | ViewDiff는 실세계 데이터 사용 |
| **MVDream** | 2023 | Multi-view SDS | Janus 문제 해결 | SDS 최적화 필요 | ViewDiff는 feed-forward |
| **Zero123++** | 2023 | Multi-view base model | 고품질 일관성 | 합성 데이터 기반 | ViewDiff는 실배경 생성 |
| **ViewDiff** | 2024 | Projection + Cross-frame-attn | 실세계 배경, 단일 패스 | 소수 카테고리, 비용 | — |
| **MultiDiff** | 2024 | Video diffusion prior + depth | 장기 일관성 우수 | 장면 중심, 오브젝트 약 | 오토리그레시브 드리프트 해결 |
| **CAT3D** | 2024 | Multi-view diffusion (NeurIPS) | 높은 확장성 | — | 더 대규모 데이터 활용 |

SyncDreamer는 서로 다른 뷰의 해당 피처를 상관시키기 위해 3D-aware attention 메커니즘을 채택하며, MVDiffusion은 공유 가중치와 correspondence-aware attention이 있는 다중 브랜치 UNet을 통해 병렬로 multi-view 이미지를 생성한다.

MultiDiff는 단일 RGB 이미지로부터 장면의 일관된 Novel View Synthesis를 위한 새로운 접근법으로, 단일 참조 이미지에서 새로운 뷰를 합성하는 태스크는 본질적으로 관찰되지 않은 영역에 대한 여러 그럴듯한 설명이 존재하기 때문에 highly ill-posed 문제다. 이를 해결하기 위해 단안 깊이 예측기와 비디오-확산 모델 형태의 강력한 prior를 통합한다.

최신 픽셀 공간 diffusion 모델 아키텍처를 적용한 연구는 단일 뷰 데이터셋을 활용하는 새로운 NVS 훈련 방식을 도입하여, multi-view 데이터셋에 비해 상대적으로 풍부한 단일 뷰 데이터를 활용함으로써 도메인 외 콘텐츠를 가진 장면에 대한 향상된 일반화 능력을 이끌어냈다.

---

## 📚 참고 자료 및 출처

| # | 자료명 | 출처 |
|---|---|---|
| 1 | ViewDiff 공식 프로젝트 페이지 | https://lukashoel.github.io/ViewDiff/ |
| 2 | ViewDiff arXiv 논문 (2403.01807) | https://arxiv.org/abs/2403.01807 |
| 3 | ViewDiff HTML 풀텍스트 (v2) | https://arxiv.org/html/2403.01807v2 |
| 4 | GitHub 공식 저장소 (facebookresearch) | https://github.com/facebookresearch/ViewDiff |
| 5 | CVPR 2024 공식 포스터 페이지 | https://cvpr.thecvf.com/virtual/2024/poster/31616 |
| 6 | TUM 공식 논문 포털 | https://portal.fis.tum.de/en/publications/viewdiff-3d-consistent-image-generation-with-text-to-image-models/ |
| 7 | Hugging Face Papers | https://huggingface.co/papers/2403.01807 |
| 8 | Semantic Scholar | https://www.semanticscholar.org/paper/ViewDiff |
| 9 | AI Models FYI 분석 | https://www.aimodels.fyi/papers/arxiv/viewdiff-3d-consistent-image-generation-text-to |
| 10 | The Moonlight 문헌 리뷰 | https://www.themoonlight.io/en/review/viewdiff-3d-consistent-image-generation-with-text-to-image-models |
| 11 | Voxel51 저자 인터뷰 (Lukas Höllein) | https://medium.com/voxel51/lukas-höllein-on-the-challenges-and-opportunities-of-text-to-3d-with-viewdiff |
| 12 | MultiDiff arXiv (2406.18524) | https://arxiv.org/abs/2406.18524 |
| 13 | SyncDreamer arXiv (2309.03453) | https://arxiv.org/html/2309.03453v2 |
| 14 | Zero-1-to-3 GitHub | https://github.com/cvlab-columbia/zero123 |
| 15 | Zero123++ arXiv (2310.15110) | https://arxiv.org/pdf/2310.15110 |
| 16 | MVDream arXiv (2308.16512) | https://arxiv.org/html/2308.16512v3 |
| 17 | Diffusion Models for 3D Generation Survey | https://www.sciopen.com/article/10.26599/CVM.2025.9450452 |
| 18 | Novel View Synthesis with Pixel-Space Diffusion (2411.07765) | https://arxiv.org/abs/2411.07765 |
| 19 | AR-1-to-3 arXiv (2503.12929) | https://arxiv.org/html/2503.12929v1 |
| 20 | 3D-free meets 3D priors arXiv (2408.06157) | https://arxiv.org/html/2408.06157v3 |

> ⚠️ **정확도 주의:** 본 답변에서 수식 중 일부(특히 LoRA QKV 변형, Voxel 집계 MLP 세부 구조)는 공개된 프로젝트 페이지, arXiv HTML 풀텍스트, 및 the Moonlight 리뷰를 기반으로 재구성된 것이며, 논문 원문의 일부 수식 표기는 arXiv HTML 렌더링 제약으로 정확한 형식이 아닐 수 있습니다. 최고 정확도를 위해 [원문 PDF](https://arxiv.org/pdf/2403.01807)를 직접 확인하시길 권장합니다.
