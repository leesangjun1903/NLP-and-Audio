# TRIP: Temporal Residual Learning with Image Noise Prior for Image-to-Video Diffusion Models

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

TRIP은 **정적 이미지로부터 유도된 이미지 노이즈 프라이어(Image Noise Prior)** 를 기반으로, Image-to-Video(I2V) 생성에서 발생하는 두 가지 핵심 문제를 동시에 해결한다:

1. **첫 번째 프레임과 후속 프레임 간의 정렬(alignment) 부족**
2. **인접 프레임 간 시간적 일관성(temporal coherence) 부족**

기존의 독립적 노이즈 예측(independent noise prediction) 방식은 주어진 이미지와 각 후속 프레임 간의 내재적 관계를 충분히 활용하지 못한다는 한계를 지적하며, 이를 **시간적 잔차 학습(Temporal Residual Learning)** 으로 재정의한다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **Image Noise Prior** | 첫 번째 프레임 잠재 코드와 노이즈가 추가된 비디오 잠재 코드로부터 1-step 역방향 확산을 통해 각 프레임의 참조 노이즈를 유도 |
| **Residual-like Dual-Path** | Shortcut Path(이미지 노이즈 프라이어) + Residual Path(3D-UNet)의 이중 경로 노이즈 예측 구조 제안 |
| **TNF Module** | Transformer 기반의 Temporal Noise Fusion 모듈로 두 경로의 노이즈를 동적으로 융합 |
| **Zero-shot 일반화** | WebVid-10M 학습 후 DTDB, MSR-VTT에서 제로샷 일반화 성능 검증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 I2V 접근법(예: VideoComposer)은 첫 번째 프레임의 이미지 잠재 코드를 단순히 노이즈 비디오 잠재 코드와 채널/시간 차원으로 연결(concatenation)한 후, 3D-UNet으로 독립적으로 각 프레임의 노이즈를 예측한다. 이 방식의 근본적 문제는:

- 각 프레임의 노이즈가 **독립적으로 예측**되어 첫 번째 프레임과의 내재적 관계가 미활용됨
- **시간적 모듈(temporal convolution, self-attention)에만 의존**하여 시간적 일관성 모델링이 취약함
- 결과적으로 합성된 프레임의 전경/배경 콘텐츠가 입력 이미지와 불일치하는 현상 발생

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 기본 비디오 확산 모델 (Preliminaries)

$N$개의 프레임으로 구성된 비디오 클립이 주어졌을 때, 사전 학습된 2D VAE 인코더 $\mathcal{E}(\cdot)$가 각 프레임의 잠재 코드 $\{z_0^i\}_{i=1}^N$를 추출한다. 순방향 확산 과정에서 시간 스텝 $t$의 노이즈 잠재 코드는:

```math
z_t = \sqrt{\bar{\alpha}_t}z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
```

여기서 $\bar{\alpha}\_t = \prod_{i=1}^{t}\alpha_t$, $\alpha_t = 1 - \beta_t$이다.

3D-UNet의 기본 학습 목표(MSE loss)는:

$$\mathcal{L} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I), t, c}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right] $$

#### 2.2.2 Image Noise Prior 도출

$i$번째 프레임의 잠재 코드 $z_0^i$를 1-step 역방향 확산으로 재구성:

$$z_0^i = \frac{z_t^i - \sqrt{1-\bar{\alpha}_t}\epsilon_t^i}{\sqrt{\bar{\alpha}_t}} $$

I2V의 핵심 가정: **짧은 비디오 클립에서 모든 프레임은 첫 번째 프레임과 내재적으로 연관됨**. 따라서 $i$번째 프레임 잠재 코드를 첫 번째 프레임 잠재 코드 $z_0^1$과 잔차 항 $\Delta z^i$의 합으로 표현:

$$z_0^i = z_0^1 + \Delta z^i $$

스케일 비율을 적용한 보조 변수 $C_t^i$ 도입:

$$C_t^i = \frac{\sqrt{\bar{\alpha}_t}\Delta z^i}{\sqrt{1-\bar{\alpha}_t}} $$

$$\Delta z^i = \frac{\sqrt{1-\bar{\alpha}_t}C_t^i}{\sqrt{\bar{\alpha}_t}} $$

식 (3), (6)을 식 (4)에 통합하면, 첫 번째 프레임 잠재 코드 $z_0^1$은:

$$z_0^1 = z_0^i - \Delta z^i = \frac{z_t^i - \sqrt{1-\bar{\alpha}_t}\epsilon_t^i}{\sqrt{\bar{\alpha}_t}} - \frac{\sqrt{1-\bar{\alpha}_t}C_t^i}{\sqrt{\bar{\alpha}_t}} = \frac{z_t^i - \sqrt{1-\bar{\alpha}_t}\epsilon_t^{i \to 1}}{\sqrt{\bar{\alpha}_t}} $$

**Image Noise Prior** $\epsilon_t^{i \to 1}$의 정의:

$$\boxed{\epsilon_t^{i \to 1} = \frac{z_t^i - \sqrt{\bar{\alpha}_t}z_0^1}{\sqrt{1-\bar{\alpha}_t}}} $$

이 $\epsilon_t^{i \to 1}$은 첫 번째 프레임 잠재 코드 $z_0^1$과 $i$번째 노이즈 프레임 잠재 코드 $z_t^i$ 사이의 관계를 인코딩하는 **참조 노이즈(reference noise)** 로 작용한다.

#### 2.2.3 Temporal Residual Learning

$i$번째 프레임의 추정 노이즈 $\tilde{\epsilon}_t^i$를 참조 노이즈(shortcut path)와 잔차 노이즈(residual path)의 선형 결합으로 표현:

$$\tilde{\epsilon}_t^i = \lambda^i \epsilon_t^{i \to 1} + (1 - \lambda^i)\Delta\tilde{\epsilon}_t^i $$

- $\lambda^i$: 프레임 인덱스 $i$에 따른 **선형 감쇠(linear decay)** 파라미터 (프레임이 멀어질수록 첫 번째 프레임과의 시간적 상관관계 감소를 반영)
- $\Delta\tilde{\epsilon}_t^i$: 3D-UNet이 학습하는 잔차 노이즈

I2V 확산 학습 목표:

$$\tilde{\mathcal{L}} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I), t, c, i}\left[\|\epsilon_t^i - \tilde{\epsilon}_t^i\|^2\right] $$

#### 2.2.4 Temporal Noise Fusion (TNF) Module

단순 선형 융합의 하이퍼파라미터 민감성 문제를 해결하기 위해, Transformer 기반 TNF 모듈을 설계:

$$\tilde{\epsilon}_\theta(z_t, t, c, i) = \tilde{\epsilon}_t^i = \varphi(\Delta\tilde{\epsilon}_t^i, \epsilon^{i \to 1}, t) $$

TNF 모듈 구조:
1. **Adaptive LayerNorm** (시간 스텝 $t$로 조정): 잔차 노이즈 $\Delta\tilde{\epsilon}_t^i$ 정규화
2. **Self-Attention**: 특징 강화
3. **Cross-Attention**: Query = 중간 특징, Key/Value = $[\epsilon_t^{i \to 1}, \Delta\tilde{\epsilon}_t^i]$ 연결 특징

### 2.3 모델 구조

```
입력 비디오
    ↓
[2D VAE 인코더 ε]
    ↓
{z_0^i}_{i=1}^N (프레임별 잠재 코드)
    ↓
순방향 확산 → {z_t^i}_{i=1}^N (노이즈 잠재 코드)
    ↓
[z_0^1 (첫 번째 프레임 잠재 코드) 시간 차원 연결]
    ↓
    ├─ [Shortcut Path] Image Noise Prior Estimation
    │       ε_t^{i→1} = (z_t^i - √ᾱ_t · z_0^1) / √(1-ᾱ_t)
    │
    └─ [Residual Path] 3D-UNet (Text Prompt Feature c)
            → Δε̃_t^i (잔차 노이즈)
                ↓
        [TNF Module]
        (Adaptive LayerNorm → Self-Attn → Cross-Attn)
                ↓
        ε̃_θ(z_t, t, c, i) (최종 노이즈)
                ↓
        반복적 디노이징 (DDIM, 50 steps)
                ↓
[2D VAE 디코더 D] → 생성 비디오
```

**백본**: Stable-Diffusion v2.1 기반 3D-UNet (ModelScopeT2V)
**학습 설정**: AdamW (3D-UNet: lr= $2\times10^{-6}$ , TNF: lr= $2\times10^{-4}$ ), 8×NVIDIA A800 GPU
**입력**: 16프레임, 256×256 해상도, 4fps

### 2.4 성능 향상

#### WebVid-10M 결과 (Table 1)

| 모델 | F-Consistency₄ (↑) | F-Consistencyall (↑) | FVD (↓) |
|---|---|---|---|
| T2V-Zero | 91.59 | 92.15 | 279 |
| VideoComposer | 88.78 | 92.52 | 231 |
| **TRIP** | **95.36** | **96.41** | **38.9** |

- T2V-Zero 대비 F-Consistency₄ **+3.77%p** 향상
- FVD에서 압도적 개선 (279 → 38.9, VideoComposer 231 → 38.9)

#### DTDB 결과 (Zero-shot, Table 2)

| 모델 | Zero-shot | FID (↓) | FVD (↓) |
|---|---|---|---|
| AL | No | 65.1 | 934.2 |
| cINN | No | 31.9 | 451.6 |
| **TRIP** | **Yes** | **24.8** | **433.9** |

#### MSR-VTT 결과 (Zero-shot, Table 3)

| 모델 | 유형 | FID (↓) | FVD (↓) |
|---|---|---|---|
| CogVideo | T2V | 23.59 | 1294 |
| Make-A-Video | T2V | 13.17 | - |
| ModelScopeT2V | T2V | 11.09 | 550 |
| VideoComposer | I2V | 31.29 | 208 |
| **TRIP** | **I2V** | **9.68** | **91.3** |

#### 인간 평가 (WebVid-10M, Table 4)

| 평가 항목 | vs. T2V-Zero | vs. VideoComposer |
|---|---|---|
| 시간적 일관성 | **96.9%** vs. 3.1% | **84.4%** vs. 15.6% |
| 동작 충실도 | **93.8%** vs. 6.2% | **81.3%** vs. 18.7% |
| 시각적 품질 | **90.6%** vs. 9.4% | **87.5%** vs. 12.5% |

### 2.5 한계

논문에서 명시적으로 언급된 한계는 제한적이나, 다음과 같은 한계를 유추할 수 있다:

1. **해상도 제한**: 256×256 고정 해상도로 실험. 고해상도(512×512, 1024×1024) 적용 시 성능/비용 문제 미검증
2. **프레임 수 제한**: 16프레임 클립으로 제한. 장시간 비디오 생성에 대한 검증 부재
3. **$\lambda^i$ 선형 감쇠 가정**: 프레임 인덱스에 따른 선형 감쇠는 단순화된 가정으로, 복잡한 모션 패턴에서는 적합하지 않을 수 있음
4. **단일 참조 이미지**: 첫 번째 프레임만을 참조로 사용. 중간 키프레임 조건화 불가
5. **계산 비용**: TNF 모듈과 Image Noise Prior 추정을 위한 추가 연산 오버헤드 미정량화
6. **샘플링 속도**: 4fps의 낮은 샘플링 레이트. 고속 모션에 대한 일반화 불확실

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 제로샷 일반화 실험 결과

TRIP의 가장 주목할 만한 일반화 성능 근거는 **WebVid-10M에서만 훈련한 후 DTDB와 MSR-VTT에서 제로샷으로 평가**했을 때의 결과다. 특히:

- **DTDB**: 사전 학습 없이 AL, cINN 대비 FID 24.8(최고), FVD 433.9 달성
- **MSR-VTT**: I2V 모델 중 FID 9.68, FVD 91.3으로 T2V 모델 포함 전체 최고 성능

이는 Image Noise Prior 기반의 잔차 학습이 특정 도메인에 과적합되지 않고, **도메인-불문 시간적 모델링 능력**을 학습함을 시사한다.

### 3.2 일반화 향상의 구조적 원인 분석

#### 3.2.1 물리적으로 해석 가능한 프라이어

Image Noise Prior $\epsilon_t^{i \to 1} = \frac{z_t^i - \sqrt{\bar{\alpha}_t}z_0^1}{\sqrt{1-\bar{\alpha}_t}}$는 단순히 학습된 파라미터가 아니라, **첫 번째 프레임과 i번째 노이즈 프레임 간의 수학적 관계에서 직접 도출**되는 닫힌 형태(closed-form)의 표현이다. 이는 데이터 분포에 무관하게 적용 가능한 귀납적 편향(inductive bias)으로 작용한다.

#### 3.2.2 잔차 학습의 일반화 이점

잔차 학습 패러다임은 3D-UNet이 **전체 노이즈가 아닌 참조 노이즈와의 차이(잔차)만을 학습**하게 한다. 이는:

$$\Delta\tilde{\epsilon}_t^i = \tilde{\epsilon}_t^i - \lambda^i\epsilon_t^{i \to 1}$$

로 분해되어, 3D-UNet이 학습해야 할 문제의 복잡도를 낮춰 **더 적은 데이터로도 일반화**가 용이해진다.

#### 3.2.3 튜닝 프리(Tuning-free) 확장성

논문 Section 4.5에서 보여주듯, TRIP은 별도의 파인튜닝 없이:
- **Stable Diffusion XL**로 생성한 이미지를 입력으로 T2V 파이프라인 구성 가능
- **InstructPix2Pix**, **ControlNet**으로 편집된 이미지를 조건으로 활용 가능

이러한 모듈식 확장성은 다양한 도메인 이미지에 대한 일반화 능력을 실질적으로 검증한다.

#### 3.2.4 Ablation 연구에서의 일반화 근거 (Table 5, 6)

| 비교 | 핵심 인사이트 |
|---|---|
| TRIP vs. TRIP $^-$ | Shortcut path 제거 시 F-Consistency₄ 94.66→95.36 하락: Image Noise Prior가 일관성의 핵심 |
| TRIP vs. TRIP $^W$ | TNF 모듈이 단순 선형 융합보다 FVD 43.0→38.9 개선: 동적 융합이 다양한 시나리오에 더 강건 |
| TRIP vs. TRIP $_C$ | 시간 차원 연결이 채널 연결보다 우월: 시간적 순서 정보 보존이 일반화에 기여 |

### 3.3 일반화 성능 향상 가능성 — 향후 방향

1. **더 다양한 도메인 프라이어**: 현재 Image Noise Prior는 첫 번째 프레임 기반이나, 이를 **중간 키프레임이나 다중 참조 이미지**로 확장 시 도메인 커버리지 증가 가능

2. **적응형 $\lambda^i$ 학습**: 현재의 선형 감쇠 $\lambda^i$를 도메인/모션 복잡도에 따라 적응적으로 학습하는 메타러닝 접근으로 일반화 가능

3. **사전 학습 데이터 다양화**: WebVid-10M만이 아닌 보다 다양한 비디오 데이터셋(HD-VILA-100M, InternVid 등)으로 사전 학습 시 일반화 능력 향상 기대

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 비디오 생성 확산 모델의 발전 계보

```
DDPM (Ho et al., NeurIPS 2020)
    └─ VDM (Ho et al., NeurIPS 2022) — T2V로 확장
        └─ Align Your Latents (Blattmann et al., CVPR 2023) — 고해상도 잠재 확산 T2V
        └─ VideoFusion (Luo et al., CVPR 2023) — 기본/잔차 노이즈 분해
        └─ ModelScopeT2V (Wang et al., arXiv 2023) — 3D-UNet 기반 T2V
        └─ VideoComposer (Wang et al., NeurIPS 2023) — I2V 조건부 합성
        └─ **TRIP** (Zhang et al., arXiv 2024) — 시간적 잔차 학습 I2V
```

### 4.2 주요 관련 연구와의 상세 비교

| 모델 | 연도 | 주요 방법 | I2V 지원 | 시간적 일관성 메커니즘 | 한계 |
|---|---|---|---|---|---|
| **DDPM** [Ho et al., 2020] | 2020 | 순수 확산 모델 | ✗ | - | 비디오 미지원 |
| **VDM** [Ho et al., 2022] | 2022 | 2D-UNet + Temporal Attention | △ | Temporal Attention | 조건화 약함 |
| **VideoFusion** [Luo et al., 2023] | 2023 | 기본노이즈 + 잔차노이즈 (이미지 확산 2개) | △ | 노이즈 분해 | 이미지 조건 미활용 |
| **VideoComposer** [Wang et al., 2023] | 2023 | 다중 조건 합성 (텍스트+광학흐름+깊이) | ✓ | 독립 노이즈 예측 | 프레임 간 관계 미활용 |
| **AnimateDiff** [Guo et al., 2023] | 2023 | Motion Module 삽입 | △ | Motion LoRA | 세밀한 I2V 정렬 부족 |
| **T2V-Zero** [Khachatryan et al., 2023] | 2023 | 제로샷 T2I→T2V | △ | Cross-frame Attention | I2V 특화 설계 부재 |
| **Preserve Your Own Correlation** [Ge et al., ICCV 2023] | 2023 | 공통노이즈 + 독립노이즈 가정 | △ | 노이즈 모델링 | 이미지 조건 미특화 |
| **DragNUWA** [Yin et al., 2023] | 2023 | 텍스트+이미지+궤적 조건 | ✓ | 궤적 기반 | 궤적 레이블 필요 |
| **TRIP (본 논문)** | 2024 | Image Noise Prior + 잔차 학습 | ✓✓ | 이중 경로 + TNF | 고해상도/장시간 미검증 |

### 4.3 VideoFusion vs. TRIP — 잔차 학습 관점 비교

VideoFusion도 노이즈를 분해하지만, 근본적 차이가 있다:

**VideoFusion**:
$$\epsilon_{\text{video}}^t = \epsilon_{\text{base}}^t + \epsilon_{\text{residual},i}^t$$
- 두 개의 독립적인 이미지 확산 모델로 기본/잔차 노이즈를 별도 추정
- 이미지 조건(첫 번째 프레임)을 직접 활용하지 않음

**TRIP**:
$$\tilde{\epsilon}_t^i = \lambda^i\epsilon_t^{i \to 1} + (1-\lambda^i)\Delta\tilde{\epsilon}_t^i$$
- $\epsilon_t^{i \to 1}$은 첫 번째 프레임에서 **수학적으로 도출**된 참조 노이즈
- 단일 3D-UNet으로 잔차를 학습하면서 시간적 관계를 명시적으로 모델링

### 4.4 Stable Video Diffusion (SVD) 및 최근 연구와의 관계

논문 작성 시점(2024년 3월) 기준으로 Stable Video Diffusion(Blattmann et al., 2023)은 직접 비교되지 않았으나, TRIP의 접근법은 SVD에서도 유사하게 적용 가능한 프레임워크다. SVD는 이미지 인코더를 통한 조건화에 집중하는 반면, TRIP은 **노이즈 공간에서의 수학적 관계**를 직접 활용한다는 차별점이 있다.

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5.1 향후 연구에 미치는 영향

#### 5.1.1 노이즈 공간에서의 조건화 패러다임 전환

TRIP은 I2V 생성에서 조건 이미지를 단순히 입력으로 연결하는 것을 넘어, **노이즈 공간에서 직접 수학적 프라이어를 유도**하는 새로운 패러다임을 제시한다. 이 아이디어는 다음 영역으로 확장 가능:

- **비디오 편집**: 키프레임을 기반으로 편집된 영역의 노이즈 프라이어 유도
- **3D 생성**: 다시점 이미지에서 공간적 노이즈 프라이어 구성
- **오디오-비디오 생성**: 오디오 신호로부터 유도된 프라이어와 결합

#### 5.1.2 잔차 학습의 확산 모델 적용 확산

HeNet의 Skip Connection이 딥러닝 전반에 영향을 준 것처럼, TRIP의 잔차적 노이즈 예측 프레임워크는 확산 모델 기반의 다양한 조건부 생성 태스크에 일반화될 수 있다:

$$\tilde{\epsilon}_\theta = f_{\text{prior}}(\text{condition}) + g_\theta(\text{residual})$$

이 분해 원칙은 **조건 정보의 활용 효율을 높이는 범용 설계 원칙**으로 자리잡을 가능성이 있다.

#### 5.1.3 제로샷 일반화 평가 프로토콜 확립

TRIP의 DTDB, MSR-VTT 제로샷 평가는 I2V 모델의 일반화 능력을 체계적으로 측정하는 벤치마킹 방법론으로 후속 연구에 영향을 미칠 것이다.

### 5.2 향후 연구 시 고려할 점

#### 5.2.1 기술적 고려 사항

1. **고해상도 확장성**:
   - 현재 256×256 제한. 512×512 이상에서 Image Noise Prior의 잠재 공간 특성 변화 고려 필요
   - 고해상도에서 TNF 모듈의 Cross-Attention 계산 복잡도 $O(N^2)$ 문제 해결 방안 연구 필요

2. **장시간 비디오 생성**:
   - 16프레임 이상의 장시간 비디오에서 선형 감쇠 $\lambda^i$의 적절한 설계 방안
   - 메모리 효율적인 시간적 확장 전략 (예: 슬라이딩 윈도우, 계층적 생성)

3. **$\lambda^i$ 설계 최적화**:
   - 현재의 단순 선형 감쇠 대신, 모션 복잡도, 장면 변화율, 카메라 움직임 등을 고려한 **적응형 $\lambda^i$** 학습
   - 강화학습 또는 메타러닝 기반의 $\lambda^i$ 최적화

4. **다중 참조 프레임 확장**:
   $$\epsilon_t^{i \to \{1, k\}} = \sum_{j \in \{1, k\}} w_j \cdot \frac{z_t^i - \sqrt{\bar{\alpha}_t}z_0^j}{\sqrt{1-\bar{\alpha}_t}}$$
   형태로 여러 키프레임을 동시에 참조하는 Multi-Reference Image Noise Prior 설계

5. **모션 다양성 vs. 정렬 트레이드오프**:
   - Image Noise Prior가 너무 강하면 첫 번째 프레임에 과도하게 고착되어 모션 다양성 감소 위험
   - 이를 제어하는 Classifier-Free Guidance 수준의 메커니즘 연구 필요:

```math
\tilde{\epsilon} = (1+s)\tilde{\epsilon}_\theta(z_t, t, c, \epsilon^{i\to1}) - s\cdot\tilde{\epsilon}_\theta(z_t, t, \emptyset, \emptyset)
```

#### 5.2.2 평가 방법론 관련 고려 사항

1. **표준화된 I2V 벤치마크 필요**: 현재 I2V 평가는 데이터셋마다 다른 메트릭을 사용. TRIP처럼 F-Consistency, FVD, FID를 함께 사용하는 통일된 프로토콜이 필요

2. **인과적 평가**: 시간적 일관성이 높다고 반드시 좋은 비디오가 아닐 수 있음 (정적인 비디오가 F-Consistency 최고). 모션 다양성과 품질의 균형 평가 메트릭 개발 필요

3. **장기 의존성 평가**: F-Consistency는 CLIP 임베딩 기반으로 의미적 일관성을 측정하지만, 픽셀 수준의 시간적 연속성이나 물리적 법칙 준수 여부를 별도로 평가하는 지표 필요

#### 5.2.3 응용 확장 관련 고려 사항

1. **실시간 생성**: DDIM 50스텝의 계산 비용을 줄이기 위한 Consistency Model 또는 Flow Matching 기반의 가속 기법 적용 가능성

2. **개인화된 I2V**: DreamBooth, LoRA 등 개인화 기법과 TRIP의 결합으로 특정 피사체나 스타일에 특화된 I2V 생성

3. **윤리적 고려**: 고품질 I2V 생성의 딥페이크 악용 가능성. 생성 비디오 감지(detection) 연구와 병행 필요

---

## 참고 자료

### 논문 원문
- **Zhang, Z., Long, F., Pan, Y., Qiu, Z., Yao, T., Cao, Y., & Mei, T. (2024). TRIP: Temporal Residual Learning with Image Noise Prior for Image-to-Video Diffusion Models.** *arXiv:2403.17005v1* [cs.CV]. https://arxiv.org/abs/2403.17005

### 논문 내 주요 인용 문헌 (본문에서 직접 참조됨)
- Ho, J., Jain, A., & Abbeel, P. (2020). **Denoising Diffusion Probabilistic Models.** NeurIPS.
- Ho, J., et al. (2022). **Video Diffusion Models.** NeurIPS.
- Blattmann, A., et al. (2023). **Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models.** CVPR.
- Wang, X., et al. (2023). **VideoComposer: Compositional Video Synthesis with Motion Controllability.** NeurIPS.
- Luo, Z., et al. (2023). **VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation.** CVPR.
- Khachatryan, L., et al. (2023). **Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators.** ICCV.
- Rombach, R., et al. (2022). **High-Resolution Image Synthesis with Latent Diffusion Models.** CVPR.
- Wang, J., et al. (2023). **ModelScope Text-to-Video Technical Report.** arXiv:2308.06571.
- Ge, S., et al. (2023). **Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models.** ICCV.
- He, K., et al. (2016). **Deep Residual Learning for Image Recognition.** CVPR.
- Dorkenwald, M., et al. (2021). **Stochastic Image-to-Video Synthesis using cINNs.** CVPR.
- Song, J., Meng, C., & Ermon, S. (2021). **Denoising Diffusion Implicit Models.** ICLR.

### 프로젝트 페이지
- TRIP 공식 프로젝트 페이지: https://trip-i2v.github.io/TRIP/

> **⚠️ 정확도 관련 주의사항**: 본 답변은 제공된 논문 PDF(arXiv:2403.17005v1)를 직접 분석한 결과에 기반합니다. Stable Video Diffusion(SVD), DiT 기반 최신 모델(Sora, CogVideoX 등)과의 비교는 해당 논문에서 직접 다루지 않으므로, 간접적 추론으로 서술했습니다. 2024년 이후 후속 연구 동향은 제 학습 데이터 한계로 인해 일부 불완전할 수 있습니다.
