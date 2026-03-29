# Video Generation Models as World Simulators
# Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models

---

## 1. 핵심 주장과 주요 기여 요약

### 논문 ①: "Video Generation Models as World Simulators" (OpenAI Technical Report, 2024.02)

이 기술 보고서는 (1) 모든 유형의 시각 데이터를 대규모 생성 모델 훈련이 가능한 통합 표현으로 변환하는 방법과 (2) Sora의 능력 및 한계에 대한 정성적 평가에 초점을 맞추고 있습니다.

**핵심 주장:**
- 비디오 생성 모델의 스케일링이 물리 세계의 범용 시뮬레이터를 구축하는 유망한 경로라고 제안합니다.
- 비디오 모델이 대규모 훈련 시 흥미로운 창발적(emergent) 능력을 보이며, 이러한 능력은 3D, 객체 등에 대한 명시적 귀납적 편향 없이 순전히 스케일의 현상으로 나타난다고 합니다.

**주요 기여:**
- LLM이 텍스트 토큰을 갖는 것처럼 Sora는 **visual patches**를 가지며, 패치가 다양한 유형의 비디오와 이미지에 대한 생성 모델 훈련에 매우 확장 가능하고 효과적인 표현임을 발견했습니다.
- Sora는 시각 데이터의 범용 모델로서, 다양한 지속 시간·종횡비·해상도의 비디오와 이미지를 생성할 수 있으며 최대 1분 길이의 고화질 비디오까지 가능합니다.

### 논문 ②: "Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models" (Liu et al., arXiv 2402.17177, 2024.02)

공개된 기술 보고서와 리버스 엔지니어링을 기반으로, Sora 모델의 배경, 관련 기술, 응용, 남은 과제 및 텍스트-투-비디오 AI 모델의 향후 방향에 대한 포괄적 리뷰를 제시합니다.

**핵심 주장:**
- Sora는 대규모 비전 모델(LVM)로서 스케일링 원칙에 부합하며, 텍스트-투-비디오 생성에서 여러 창발적 능력을 보여줍니다. 이는 LVM이 LLM에서 보이는 발전과 유사한 진보를 달성할 가능성을 보여줍니다.
- 다수의 LLM이 창발적 능력을 보이지만 비전 모델에서 동등한 능력을 보인 경우는 Sora가 처음이며, 컴퓨터 비전 분야에서 중요한 이정표입니다.

**주요 기여:**
- Sora의 개발 과정을 추적하고 핵심 기술을 조사하며, 영화 제작·교육·마케팅 등 다양한 산업에서의 응용과 잠재적 영향을 상세히 기술합니다. 또한 안전하고 편향 없는 비디오 생성 보장 등 주요 과제를 논의합니다.

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

기존의 비디오 생성 연구는 RNN, GAN, autoregressive transformer, diffusion model 등 다양한 방법을 사용했지만, 좁은 범주의 시각 데이터, 짧은 비디오, 또는 고정된 크기의 비디오에만 집중해왔습니다.

핵심 문제:
1. **다양한 해상도·종횡비·길이의 비디오를 통합적으로 처리**할 수 있는 범용 시각 생성 모델의 부재
2. **물리 세계 시뮬레이션** 능력의 부족
3. 긴 비디오에서의 **시간적 일관성(temporal consistency)** 유지

### 2.2 제안하는 방법 (수식 포함)

#### (A) 시각 데이터의 통합 표현: Spacetime Latent Patches

고수준에서, 비디오를 먼저 저차원 잠재 공간으로 압축한 후 시공간 패치(spacetime patches)로 분해하여 패치로 변환합니다.

**Step 1: Video Compression Network (VAE 기반 인코더)**

차원을 축소하는 네트워크를 훈련하며, 이 네트워크는 원시 비디오를 입력으로 받아 시간적·공간적으로 압축된 잠재 표현을 출력합니다. 대응하는 디코더 모델은 생성된 잠재 표현을 다시 픽셀 공간으로 매핑합니다.

원시 비디오 $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times 3}$에 대해 인코더 $\mathcal{E}$는:

$$\mathbf{z} = \mathcal{E}(\mathbf{V}) \in \mathbb{R}^{T' \times H' \times W' \times C}$$

여기서 $T' < T$, $H' < H$, $W' < W$이며, 시간적·공간적으로 모두 압축됩니다. 디코더 $\mathcal{D}$는:

$$\hat{\mathbf{V}} = \mathcal{D}(\mathbf{z}) \approx \mathbf{V}$$

압축 네트워크는 VAE 또는 VQ-VAE에 기반하여 구축됩니다.

**Step 2: Spacetime Patchification**

압축된 입력 비디오에서 시공간 패치의 시퀀스를 추출하며 이들은 트랜스포머 토큰으로 작동합니다. 이 방식은 이미지에도 적용됩니다(단일 프레임 비디오).

잠재 표현 $\mathbf{z}$를 고정 크기의 3D 패치로 분할:

$$\mathbf{z} \rightarrow \{p_1, p_2, \ldots, p_N\}, \quad p_i \in \mathbb{R}^{t \times h \times w \times C}$$

각 패치를 선형 임베딩하여 토큰화:

$$\mathbf{x}_i = \text{Linear}(\text{Flatten}(p_i)) + \mathbf{e}_{\text{pos}}^{(i)}$$

여기서 $\mathbf{e}_{\text{pos}}^{(i)}$는 시공간 위치 인코딩입니다.

#### (B) Diffusion Transformer (DiT) 아키텍처

Sora는 확산 모델(diffusion model)로서, 입력 노이즈 패치(및 텍스트 프롬프트 등 조건 정보)가 주어지면 원래의 "깨끗한" 패치를 예측하도록 훈련됩니다. 중요한 점은 Sora가 확산 트랜스포머(diffusion transformer)라는 것입니다.

**Forward diffusion process:**

$$q(\mathbf{z}_t | \mathbf{z}_0) = \mathcal{N}(\mathbf{z}_t; \sqrt{\bar{\alpha}_t}\mathbf{z}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t} \alpha_s$, $\alpha_t = 1 - \beta_t$이며, $\beta_t$는 노이즈 스케줄입니다.

**Training objective (denoising):**

$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_0, \boldsymbol{\epsilon}, t, \mathbf{c}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c}) \|^2 \right]$$

여기서:
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$: 샘플링된 노이즈
- $\boldsymbol{\epsilon}_\theta$: DiT 네트워크(denoiser)
- $\mathbf{c}$: 텍스트 조건 임베딩
- $t$: 확산 타임스텝

#### (C) Conditioning Mechanism

CLIP과 유사한 조건화 메커니즘이 LLM으로 보강된 사용자 지시문과 잠재적 시각적 프롬프트를 수신하여 확산 모델이 스타일화되거나 테마가 있는 비디오를 생성하도록 안내합니다.

**Adaptive Layer Normalization (adaLN)을 통한 조건 주입:**

$$\text{adaLN}(\mathbf{h}, \mathbf{c}) = \gamma(\mathbf{c}) \cdot \frac{\mathbf{h} - \mu(\mathbf{h})}{\sigma(\mathbf{h})} + \beta(\mathbf{c})$$

여기서 $\gamma(\mathbf{c})$와 $\beta(\mathbf{c})$는 텍스트 조건 $\mathbf{c}$에서 추정된 스케일·시프트 파라미터입니다.

### 2.3 모델 구조 개요

본질적으로 Sora는 유연한 샘플링 차원을 가진 확산 트랜스포머입니다. 세 부분으로 구성됩니다: (1) 시공간 압축기가 원본 비디오를 잠재 공간에 매핑하고, (2) ViT가 토큰화된 잠재 표현을 처리하여 노이즈 제거된 잠재 표현을 출력하며, (3) CLIP과 유사한 조건화 메커니즘이 확산 모델을 안내합니다. 여러 노이즈 제거 단계 후 디코더로 픽셀 공간에 매핑합니다.

```
┌─────────────────────────────────────────────────────┐
│              Sora Architecture Overview               │
├─────────────────────────────────────────────────────┤
│                                                       │
│  Raw Video V ∈ ℝ^{T×H×W×3}                          │
│       │                                               │
│       ▼                                               │
│  ┌──────────────────────┐                            │
│  │  Video Compression   │  (VAE Encoder)             │
│  │  Network (Encoder)   │  z = E(V)                  │
│  └──────────┬───────────┘                            │
│             │                                         │
│             ▼                                         │
│  ┌──────────────────────┐                            │
│  │  Spacetime Patch     │  {p₁, p₂, ..., pₙ}       │
│  │  Extraction          │  → Transformer Tokens      │
│  └──────────┬───────────┘                            │
│             │                                         │
│             ▼                                         │
│  ┌──────────────────────┐   ┌──────────────┐        │
│  │  Diffusion           │◄──│  Text/Visual  │        │
│  │  Transformer (DiT)   │   │  Conditioning │        │
│  │  (Denoising ViT)     │   │  (CLIP-like)  │        │
│  └──────────┬───────────┘   └──────────────┘        │
│             │                                         │
│             ▼                                         │
│  ┌──────────────────────┐                            │
│  │  Video Decoder       │  V̂ = D(ẑ)                 │
│  └──────────┬───────────┘                            │
│             │                                         │
│             ▼                                         │
│  Generated Video V̂ ∈ ℝ^{T×H×W×3}                   │
│                                                       │
└─────────────────────────────────────────────────────┘
```

생성 모델은 DiT에 기반하여 구축되며, 원래 DiT는 클래스-이미지 생성용이므로 두 가지 수정이 필요합니다: (1) self-attention 블록을 시간 모델링을 위해 확장하고, (2) 조건을 클래스에서 텍스트로 변경하고 텍스트 정보 주입 블록을 추가합니다.

**DiT Block의 수학적 표현:**

각 DiT 블록에서의 연산은 다음과 같이 표현됩니다:

$$\mathbf{h}' = \mathbf{h} + \text{ST-Attn}(\text{adaLN}(\mathbf{h}, \mathbf{c}))$$

$$\mathbf{h}'' = \mathbf{h}' + \text{Cross-Attn}(\text{adaLN}(\mathbf{h}', \mathbf{c}), \mathbf{c}_{\text{text}})$$

$$\mathbf{h}_{\text{out}} = \mathbf{h}'' + \text{FFN}(\text{adaLN}(\mathbf{h}'', \mathbf{c}))$$

여기서:
- $\text{ST-Attn}$: Spatial-Temporal Self-Attention
- $\text{Cross-Attn}$: 텍스트-비디오 교차 어텐션
- $\text{FFN}$: Pointwise Feed-Forward Network

### 2.4 성능 향상 요인

| 요인 | 설명 |
|------|------|
| **스케일링** | Sora는 확산 트랜스포머가 비디오 모델로서도 효과적으로 스케일링됨을 입증했습니다. 훈련 컴퓨트가 증가하면 샘플 품질이 현저히 개선됩니다. |
| **네이티브 종횡비 훈련** | 네이티브 종횡비로 비디오를 훈련하면 구성과 프레이밍이 개선됩니다. 정사각형 크롭으로 훈련된 모델과 비교하면 Sora가 더 우수한 프레이밍을 보여줍니다. |
| **Re-captioning** | DALL·E 3에서 도입된 re-captioning 기법을 비디오에 적용합니다. 매우 상세한 캡션 모델을 훈련한 뒤 훈련 세트의 모든 비디오에 텍스트 캡션을 생성합니다. |
| **유연한 입력 처리** | Sora는 임의 해상도(최대 1920×1080), 모든 종횡비, 모든 길이의 비디오에 대해 훈련하고 실행할 수 있습니다. 훈련 데이터를 고정된 크기로 리사이징하거나 크롭할 필요가 없습니다. |

### 2.5 한계

출시 시 OpenAI는 복잡한 물리 시뮬레이션, 인과 관계 이해, 좌우 구분의 제한된 능력 등 단점을 인정했습니다.

| 한계 | 세부 내용 |
|------|----------|
| **물리 시뮬레이션 부족** | Sora는 현재 많은 기본적 상호작용의 물리를 정확하게 모델링하지 못합니다(예: 유리 깨짐). |
| **인과관계 혼동** | 복잡한 장면의 물리를 시뮬레이션하는 데 어려움을 겪으며, 특정 인과 관계를 이해하지 못할 수 있습니다. |
| **공간 혼동** | 프롬프트의 공간적 세부사항(좌우 구분)을 혼동하거나 시간에 따른 이벤트의 정확한 기술에 어려움을 겪을 수 있습니다. |
| **자발적 객체 출현** | 특히 많은 엔티티가 포함된 장면에서 동물이나 사람이 자발적으로 나타날 수 있습니다. |
| **안전성·편향** | 안전하고 편향 없는 비디오 생성 보장 등 주요 과제가 남아있습니다. |
| **계산 비용** | 대규모·다양한 훈련 데이터의 필요성, 공정성과 강건성 보장의 어려움, 훈련 및 배포에 필요한 상당한 계산 리소스가 문제입니다. |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 스케일링 법칙과 일반화

비전 모델에 대한 스케일링 법칙에서, ViT 모델의 성능-컴퓨트 프론티어가 충분한 훈련 데이터와 함께 대략적으로 (포화하는) 멱법칙을 따름이 입증되었습니다.

비전 모델의 스케일링 법칙은 다음과 같이 표현될 수 있습니다:

$$L(C) \approx \alpha \cdot C^{-\beta} + L_\infty$$

여기서 $C$는 컴퓨트, $L$은 손실, $\alpha, \beta$는 스케일링 계수, $L_\infty$는 이론적 최소 손실입니다.

### 3.2 창발적 능력과 일반화

LLM에서의 창발적 능력은 특정 스케일에서 나타나는 정교한 행동이나 기능으로, 모델의 포괄적 훈련과 광범위한 파라미터 수의 결합에서 비롯됩니다. 이러한 능력의 출현은 소규모 모델의 성능으로부터 직접 예측할 수 없습니다.

Sora에서 관찰된 **일반화 성능 향상**의 핵심 요소:

**① 통합 표현을 통한 범용성:**
패치 기반 표현으로 다양한 해상도·지속시간·종횡비의 비디오와 이미지를 훈련할 수 있으며, 추론 시 랜덤 초기화된 패치를 적절한 크기의 그리드에 배열하여 생성 비디오 크기를 제어합니다.

**② 3D 일관성 (Zero-shot 3D understanding):**
Sora는 동적 카메라 모션이 있는 비디오를 생성할 수 있으며, 카메라가 이동하고 회전할 때 사람과 장면 요소가 3차원 공간에서 일관되게 움직입니다.

**③ 장거리 일관성과 객체 영속성:**
Sora는 항상은 아니지만 단거리 및 장거리 종속성을 모두 효과적으로 모델링할 수 있으며, 사람·동물·객체가 가려지거나 프레임을 벗어나도 지속됩니다.

**④ 세계 상호작용 시뮬레이션:**
Sora는 세계 상태에 영향을 미치는 행동을 간단한 방식으로 시뮬레이션할 수 있습니다. 예를 들어, 화가가 캔버스에 새 붓자국을 남기거나, 사람이 버거를 먹으면 물린 자국이 남습니다.

**⑤ 디지털 세계 시뮬레이션:**
Sora는 비디오 게임 같은 인공 프로세스도 시뮬레이션할 수 있습니다. Minecraft에서 기본 정책으로 플레이어를 제어하면서 세계와 역학을 고충실도로 렌더링할 수 있으며, "Minecraft"를 언급하는 캡션으로 zero-shot 유도가 가능합니다.

### 3.3 일반화 향상의 기술적 기반

**Diffusion Transformer의 스케일링 이점:**

많은 딥러닝 작업에서 트랜스포머 아키텍처가 파라미터 수에 따라 더 효과적으로 스케일링됩니다. DiT는 U-Net 노이즈 제거기를 트랜스포머 기반 노이즈 제거기로 대체합니다.

시공간 위치 인코딩을 사용하여 임의의 입력 형태를 수용하는 DiT 설계가 특히 영향력 있으며, 3D 시공간 좌표에서 위치 인코딩을 생성함으로써 고정된 입력 그리드를 강제하는 1차원 위치 인코딩의 한계를 회피합니다.

일반화 성능을 결정하는 핵심 수식 — Self-Attention의 시공간 확장:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

이를 시공간으로 확장:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

여기서 $\mathbf{X} \in \mathbb{R}^{(T' \cdot H' \cdot W') \times d}$는 시공간 패치의 평탄화된 시퀀스입니다.

OpenAI는 매우 긴 컨텍스트 윈도우를 사용하여 비디오의 모든 토큰을 패킹할 가능성이 높으며, 이는 multi-head attention 연산자가 시퀀스 길이에 대해 이차(quadratic) 비용을 나타내지만 계산 비용이 매우 높더라도 수행될 것입니다.

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 연구 패러다임 변화

이러한 능력들은 비디오 모델의 지속적 스케일링이 물리적·디지털 세계의 고능력 시뮬레이터 개발을 위한 유망한 경로임을 시사합니다.

**① "World Simulator"로서의 비디오 모델 패러다임:**
- Sora는 실세계를 이해하고 시뮬레이션할 수 있는 모델의 기반으로, AGI 달성을 위한 중요한 이정표가 될 수 있습니다.
- 이러한 시스템이 물리 세계를 깊이 이해하는 AI 모델 훈련에 중요할 것이라 믿어집니다.

**② 산업적 영향:**
영화 제작·교육·마케팅 분야에서의 잠재적 영향이 크며, 비디오 생성에서의 생산성과 창의성을 증진하는 새로운 인간-AI 상호작용 방식을 가능케 할 수 있습니다.

**③ 안전성·윤리적 고려:**
Sora와 유사 모델이 책임감 있게 활용되도록 강화된 보안 조치와 오남용 완화 방법론의 개발이 필요하며, 법적·심리적·기술적 전문성을 아우르는 학제간 협력의 필요성이 강조됩니다.

### 4.2 향후 연구 시 고려할 핵심 사항

| 연구 방향 | 세부 사항 |
|-----------|----------|
| **물리 엔진 통합** | 물리 엔진의 통합이 세계 모델 구축의 잠재적 경로가 될 수 있습니다. |
| **효율적 훈련·추론** | 훈련 및 추론 전략이 대규모 생성 모델의 성능과 효율성을 개선하며, 네이티브 종횡비 학습이 생성 비디오의 구성을 개선하지만 효율적 훈련을 위한 기술·엔지니어링 최적화가 필요합니다. |
| **데이터 품질·다양성** | 좋은 생성 성능을 위한 훈련 데이터의 필수성이 재차 강조되며, 광범위한 게임 비디오가 풍부한 물리 정보를 포함하여 물리 세계 이해를 도울 수 있습니다. |
| **비전-언어 통합** | 비전과 언어 이해의 통합, 더 효율적이고 에너지 효율적인 모델 개발, 이러한 강력한 AI 시스템의 배포를 둘러싼 윤리적 고려사항이 향후 방향입니다. |
| **장기 비디오 생성** | Sora의 중요한 돌파구는 매우 긴 비디오 생성 능력이며, 2초 비디오와 1분 비디오 생성의 차이는 엄청납니다. 오류 누적 및 시간에 따른 품질/일관성 유지가 주요 과제입니다. |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 텍스트-투-비디오 모델의 발전 타임라인

2020년대 동안 고품질 텍스트 조건부 비디오 생성의 발전은 주로 비디오 확산 모델의 개발에 의해 주도되었습니다.

| 연도 | 모델/시스템 | 핵심 특징 |
|------|------------|-----------|
| 2022 | CogVideo | 최초의 대규모 T2V 모델 (9.4B 파라미터) |
| 2022 | Make-A-Video (Meta), Imagen Video (Google) | Meta의 초기 T2V 진출; Google의 3D U-Net 기반 모델 |
| 2023 | Runway Gen-1/Gen-2 | 최초의 상용 T2V/V2V 모델 |
| 2024.01 | Google Lumiere | 고급 시간적 일관성을 가진 확산 기반 비디오 생성기 |
| 2024.02 | **Sora (OpenAI)** | 최대 1분의 초현실적 비디오 생성 — 주요 이정표 |
| 2024.06 | Runway Gen-3 Alpha | 더 많은 스타일 제어 |
| 2024.08 | CogVideoX | 오픈소스 후속작, 6초 클립 생성 |
| 2024.10 | Meta Movie Gen | 편집, 얼굴 통합, T2V |
| 2024.12 | Google Veo 2 | 더 강한 인과 관계 및 프롬프트 준수 |
| 2025.02 | Wan 2.1 (Alibaba) | 오픈소스, LoRA 파인튜닝으로 높은 커스터마이즈 가능 |
| 2025.05 | Google Veo 3 | 비디오+사운드/음성을 하나의 파이프라인에서 네이티브 생성하는 최초의 주요 모델 |
| 2025.09 | **Sora 2 (OpenAI)** | 물리적 정확성, 현실감, 제어 가능성이 향상된 최신 비디오 생성 모델 |

### 5.2 아키텍처 비교

| 모델 | 아키텍처 | Denoiser | 오픈소스 |
|------|----------|----------|---------|
| **Sora** | Latent Diffusion + DiT | Transformer | ✗ |
| **CogVideoX** | 3D VAE + DiT | Transformer | ✓ |
| **Stable Video Diffusion** | Latent Diffusion | U-Net | ✓ |
| **Runway Gen-3** | 향상된 트랜스포머 기반 아키텍처 | Transformer | ✗ |
| **Open-Sora Plan** | Wavelet-Flow VAE + Skiparse Denoiser | 3D Full Attention | ✓ |
| **Veo 2/3** | Diffusion-based | Unknown | ✗ |

### 5.3 오픈소스 생태계의 성장

Sora 데모의 영향은 즉각적이었으며, Google Veo2, Minimax, Runway Gen3 Alpha, Kling, Pika, Luma Dream Machine 등 주요 플레이어와 스타트업이 고능력 모델을 생산했습니다. 오픈소스에서도 CogVideoX, Mochi-1, Hunyuan, Allegro, LTX Video 등의 급증이 있었습니다.

### 5.4 핵심 기술 트렌드 비교

```
2020-2022: GAN/VAE 기반 → 짧고 저해상도 비디오
    │
2022-2023: U-Net 기반 Latent Diffusion → 품질 향상, 짧은 클립
    │
2024 (Sora): DiT (Transformer 기반 Diffusion) → 장기 비디오, 
             │   Spacetime Patches, 멀티 해상도
             │
2024-2025: DiT 패러다임 확산 →
             ├── CogVideoX: 3D Causal VAE + DiT
             ├── Open-Sora Plan: Skiparse Attention
             ├── Veo 3: 네이티브 오디오 통합
             └── Sora 2: 향상된 물리 시뮬레이션
```

2025년에 AI 비디오 모델은 큰 도약을 이루었으며, 네이티브 오디오 생성이 소비자 도구에 도입되고, 물리 및 모션 일관성이 개선되며, 카메라 제어가 더 시네마틱해졌습니다.

---

## 참고자료 (References)

1. **OpenAI**, "Video generation models as world simulators" (Technical Report, February 2024). [https://openai.com/index/video-generation-models-as-world-simulators/](https://openai.com/index/video-generation-models-as-world-simulators/)

2. **Yixin Liu, Kai Zhang, Yuan Li, et al.**, "Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models," arXiv:2402.17177, February 2024. [https://arxiv.org/abs/2402.17177](https://arxiv.org/abs/2402.17177)

3. **OpenAI**, "Sora 2 is here," September 2025. [https://openai.com/index/sora-2/](https://openai.com/index/sora-2/)

4. **Wikipedia**, "Sora (text-to-video model)." [https://en.wikipedia.org/wiki/Sora_(text-to-video_model)](https://en.wikipedia.org/wiki/Sora_(text-to-video_model))

5. **Peebles, W. & Xie, S.**, "Scalable Diffusion Models with Transformers (DiT)," ICCV 2023.

6. **Hugging Face Blog**, "State of Open Video Generation Models in Diffusers," January 2025. [https://huggingface.co/blog/video_gen](https://huggingface.co/blog/video_gen)

7. **Open-Sora Plan Team**, "Open-Sora Plan: Open-Source Large Video Generation Model," arXiv:2412.00131, November 2024. [https://arxiv.org/html/2412.00131v1](https://arxiv.org/html/2412.00131v1)

8. **"Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond,"** arXiv:2405.03520, May 2024. [https://arxiv.org/html/2405.03520v1](https://arxiv.org/html/2405.03520v1)

9. **Towards Data Science**, "Deep Dive into Sora's Diffusion Transformer (DiT) by Hand," January 2025. [https://towardsdatascience.com/deep-dive-into-soras-diffusion-transformer-dit-by-hand/](https://towardsdatascience.com/deep-dive-into-soras-diffusion-transformer-dit-by-hand/)

10. **allpcb.com**, "Sora: OpenAI's Video Model Architecture and Use Cases." [https://www.allpcb.com/allelectrohub/sora-openais-video-model-architecture-and-use-cases](https://www.allpcb.com/allelectrohub/sora-openais-video-model-architecture-and-use-cases)

11. **Artificial Cognition**, "Are Video Generation Models World Simulators?" March 2024. [https://artificialcognition.net/posts/video-generation-world-simulators/](https://artificialcognition.net/posts/video-generation-world-simulators/)

12. **gaga.art**, "The History & Future of AI Video Generation Models 2025." [https://gaga.art/blog/ai-video-generation-model/](https://gaga.art/blog/ai-video-generation-model/)

13. **GitHub - lichao-sun/SoraReview**, Official GitHub for the review paper. [https://github.com/lichao-sun/SoraReview](https://github.com/lichao-sun/SoraReview)

14. **Wikipedia**, "Text-to-video model." [https://en.wikipedia.org/wiki/Text-to-video_model](https://en.wikipedia.org/wiki/Text-to-video_model)

---

> **참고**: OpenAI의 기술 보고서는 모델 및 구현 세부사항이 포함되지 않았다고 명시하고 있어, 위의 수식 중 일부(특히 DiT 블록의 구체적 구현, adaLN 등)는 공개된 DiT 논문(Peebles & Xie, ICCV 2023)과 리뷰 논문의 리버스 엔지니어링에 기반한 것으로, OpenAI가 공식 확인한 세부사항이 아닐 수 있습니다. 정확한 내부 구현은 비공개 상태입니다.
