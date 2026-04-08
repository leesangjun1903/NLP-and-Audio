
# LEDITS++: Limitless Image Editing using Text-to-Image Models

> **📌 논문 정보**
> - **저자:** Manuel Brack, Felix Friedrich, Katharina Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, Apolinário Passos
> - **발표:** CVPR 2024 (arXiv: 2311.16711)
> - **소속:** TU Darmstadt (AIML Lab), Hugging Face

---

## 1. 핵심 주장 및 주요 기여 요약

텍스트-이미지 확산 모델(Text-to-Image Diffusion Models)은 텍스트만으로 고품질 이미지를 생성하는 능력으로 큰 관심을 받고 있으며, 이를 실제 이미지 편집에 활용하려는 연구가 이어지고 있다. 그러나 기존의 이미지-이미지 편집 방법들은 비효율적이고 부정확하며 다목적성이 부족하다는 한계가 있다. 구체적으로, 기존 방법들은 시간이 많이 소요되는 파인튜닝을 요구하거나, 입력 이미지에서 불필요하게 크게 벗어나거나, 여러 편집을 동시에 지원하지 못하는 문제를 가진다.

이를 해결하기 위해, LEDITS++는 효율적이면서도 다목적인 이미지 편집 기법으로, 튜닝 및 최적화가 불필요하고, 소수의 확산 단계에서 실행되며, 여러 동시 편집을 기본 지원하고, 관련 이미지 영역에만 변경을 제한하며, 아키텍처에 독립적(architecture-agnostic)이다.

### 📌 주요 기여 (5가지)

LEDITS++는 다음의 기여를 제공한다: (i) LEDITS++의 공식적 정의 수립, (ii) 더 효율적인 확산 샘플링 방법을 위한 완전 역변환(perfect inversion) 유도, (iii) 효율성, 다목적성, 정밀도의 정성적·정량적 검증, (iv) 자동 및 사용자 인간 메트릭을 사용한 동시대 연구와의 철저한 비교, (v) 텍스트 기반 이미지 편집 평가를 위한 새로운 벤치마크 **TEdBench++** 도입.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

| 기존 방법의 문제점 | 설명 |
|---|---|
| **파인튜닝 필요** | Imagic 등은 각 이미지마다 모델 튜닝 필요 → 느림 |
| **이미지 왜곡** | 편집 시 원본 이미지에서 과도하게 이탈 |
| **단일 편집 한계** | 동시에 여러 편집 불가 |
| **DDIM 역변환 오류 누적** | 각 타임스텝에서 발생하는 작은 오류가 누적됨 |

기존 연구들은 결정론적 DDIM 샘플링 프로세스를 역변환하는 방식에 크게 의존했는데, DDIM 역변환은 입력 이미지로 다시 노이즈 제거했을 때 해당 이미지를 생성하는 초기 노이즈 벡터를 찾는다. 그러나 충실한 재구성은 아주 작은 스텝 수에서만 달성 가능하여 많은 역변환 스텝이 필요하고, 각 타임스텝에서 작은 오류가 축적되어 입력 이미지로부터 의미 있는 편차가 발생하며, 비용이 많이 드는 최적화 기반 오류 수정이 필요하다.

---

### 2-2. 제안하는 방법 및 수식

LEDITS++의 방법론은 세 가지 핵심 구성요소로 나뉜다: **(1) 효율적인 이미지 역변환(Efficient Image Inversion)**, **(2) 다목적 텍스트 편집(Versatile Textual Editing)**, **(3) 의미론적 기반(Semantic Grounding)**.

---

#### 🔷 Component 1: Perfect Inversion (DPM-Solver++ 역변환)

LEDITS++는 기존 DDPM 샘플링 방식에 대해 제안된 '편집 친화적 노이즈 공간(edit-friendly noise space)'과 완전한 입력 재구성 특성을, 훨씬 빠른 다단계 확률 미분방정식(SDE) 솔버에 적용하여 새롭게 유도한다. 이러한 DPM-Solver++의 새로운 역변환 가능성은 역변환과 추론을 합해 단 20단계의 확산 스텝으로 편집을 가능하게 한다.

확산 모델의 역방향 프로세스 기반:

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})
$$

여기서 LEDITS++는 DDPM 기반 편집 친화적 역변환의 특성을 DPM-Solver++ (고차 SDE 솔버)로 확장한다:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0(x_t) + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_t
$$

DDPM은 역방향 확산 과정을 SDE로 정식화할 때 1차 SDE 솔버로 간주될 수 있으며, 이 SDE는 더 고차의 미분 방정식 솔버를 사용하여 더 적은 스텝으로 더 효율적으로 풀 수 있다. 이에 따라 새롭고 빠른 기법인 **DPM-Solver++ Inversion**을 유도한다.

LEDITS++는 Imagic, Pix2Pix-Zero 등의 방법보다 수 배 빠르며, 추론 시 사용하는 것과 동일한 수의 확산 스텝만으로 역변환이 가능해 표준 DDIM 역변환 대비 **21배의 런타임 개선**을 달성한다. 또한 최근의 빠른 스케줄러를 사용해 전체 성능을 더욱 향상시키며, 최근의 DDPM 역변환보다 6배 빠르다.

---

#### 🔷 Component 2: Versatile Textual Editing (Guidance 수식)

각 개념에 대해 조건부 추정치와 비조건부 추정치를 기반으로 전용 가이던스 항을 설계한다. LEDITS++ 가이던스는 편집 방향(해당 개념에서 멀어지거나 가까워지는 방향)을 반영함과 동시에, 원하는 편집 효과에 대한 세밀한 제어를 극대화하도록 정의된다.

단일 편집 개념 $e$에 대한 LEDITS++ 가이던스 항:

$$
\hat{\epsilon}_\theta^{(e)}(x_t, t) = \epsilon_\theta(x_t, t, \emptyset) + s_e \cdot \phi_e(x_t, t) \cdot \left(\epsilon_\theta(x_t, t, e) - \epsilon_\theta(x_t, t, \emptyset)\right)
$$

여기서:
- $\epsilon_\theta(x_t, t, \emptyset)$: 비조건부(unconditional) 노이즈 추정치
- $\epsilon_\theta(x_t, t, e)$: 편집 프롬프트 $e$로 조건화된 노이즈 추정치
- $s_e$: 편집 강도(edit scale) — 클수록 편집 효과 강화
- $\phi_e$: 암묵적 마스크(implicit mask) 함수 — 편집이 적용될 픽셀의 비율 $\lambda \in (0,1)$을 결정

특히, 단일 개념 $e$에 대해 균일한 $\phi = s_e$인 경우, 이 수식은 Classifier-Free Guidance(CFG) 항으로 일반화된다.

여러 편집 개념 $I = \{e_i\}_{i \in I}$에 대한 다중 편집 가이던스:

$$
\hat{\epsilon}_\theta(x_t, t) = \epsilon_\theta(x_t, t, \emptyset) + \sum_{i \in I} s_{e_i} \cdot \phi_{e_i}(x_t, t) \cdot \left(\epsilon_\theta(x_t, t, e_i) - \epsilon_\theta(x_t, t, \emptyset)\right)
$$

---

#### 🔷 Component 3: Semantic Grounding (암묵적 마스킹)

LEDITS++ 가이던스에서의 마스킹 항은 **U-Net의 크로스-어텐션(cross-attention) 레이어에서 생성된 마스크** $M^1$과 **노이즈 추정치에서 유도된 마스크** $M^2$의 교집합으로 구성되어, 관련 이미지 영역에 초점을 맞추면서도 세밀한 입도를 갖는 마스크를 생성한다. 이러한 맵은 이미 존재하지 않는 편집 개념과 관련된 이미지 영역도 포착할 수 있으며, 여러 편집에서 각 편집 프롬프트에 대한 전용 마스크를 계산함으로써 해당 가이던스 항들이 서로 독립적으로 유지되어 간섭을 최소화한다.

$$
\phi_e(x_t, t) = M^1_e(x_t, t) \cap M^2_e(x_t, t)
$$

$M^1$은 크로스-어텐션 마스크로 LEDITS++ 논문의 Equation 12로 정의되며, $M^2$는 노이즈 추정치에서 유도된 마스크로 역시 Equation 12에 정의된다.

---

### 2-3. 모델 구조

LEDITS++는 `diffusers` 라이브러리에 `LEditsPPPipelineStableDiffusion` 및 `LEditsPPPipelineStableDiffusionXL`로 통합되어 있다. 세 가지 백본 아키텍처를 지원하며:

구현은 `StableDiffusionPipeline_LEDITS`, `StableDiffusionPipelineXL_LEDITS`, `IFDiffusion_LEDITS` 등 각각의 확산 파이프라인을 확장한다.

```
[Input Image x_0]
        ↓
[DPM-Solver++ Inversion → {z_t, x_1,...,x_T}]  ← Perfect Inversion (No Error)
        ↓
[Edit Guidance with Implicit Masking φ_e]        ← Cross-Attn Mask ∩ Noise Mask
        ↓
[Multi-Edit Guidance Summation]                  ← Isolated per concept e_i
        ↓
[DPM-Solver++ Denoising]
        ↓
[Edited Image x̂_0]
```

LEDITS++의 역변환 방식은 튜닝이나 최적화 없이 소수의 확산 스텝으로 고충실도 결과를 산출하며, 여러 동시 편집을 지원하고 아키텍처에 독립적이다.

---

### 2-4. 성능 향상

LEDITS++는 TEdBench++에서 SD1.5 기준 0.79, SD-XL 기준 0.87의 성공률을 달성하여, Imagic의 SD1.5 0.58, Imagen 0.83을 크게 상회한다. 또한 경쟁 방법과 비교해 현저히 낮은 LPIPS 점수를 보이며 이미지 품질을 유지한다. 예를 들어, TEdBench++에서 LEDITS++ (SD1.5)의 LPIPS는 0.30으로, Imagic의 0.57보다 훨씬 우수하다.

모든 방법에서 다목적성과 정밀도 사이의 자연스러운 트레이드오프가 존재하지만, LEDITS++는 이상적인 영역에 가장 근접하여 다른 모든 방법보다 명확히 우수한 성능을 보인다.

| 평가 지표 | LEDITS++ (SD1.5) | LEDITS++ (SDXL) | Imagic (SD1.5) |
|---|---|---|---|
| **TEdBench++ 성공률** | 0.79 | **0.87** | 0.58 |
| **LPIPS ↓** | **0.30** | - | 0.57 |
| **런타임** | DDPM 역변환 대비 **6배 빠름** | - | 매우 느림 |

---

### 2-5. 한계

LEDITS++도 일부 한계를 가진다. 중요한 구조적 변화를 포함하는 복잡한 편집에서는 여전히 아티팩트가 발생할 수 있으며, 최적의 결과를 얻기 위해서는 신중한 프롬프트 엔지니어링이 필요하다.

또한, 기반 확산 모델의 능력이 LEDITS++로 수행된 편집의 성공 여부와 품질에 직접적인 영향을 미친다는 모델 의존성(Model Dependency)이 존재한다.

---

## 3. 모델의 일반화 성능 향상 가능성

LEDITS++의 일반화 성능 향상 가능성은 다음 세 측면에서 특히 두드러진다.

#### 🔹 아키텍처 독립성 (Architecture-Agnosticism)

LEDITS++의 방법론은 여러 동시 편집을 지원하며 **아키텍처에 독립적**이다. 이는 Stable Diffusion 1.5, SDXL, DeepFloyd IF 등 다양한 확산 모델 백본에 곧바로 적용 가능함을 의미한다. 특정 모델 아키텍처에 종속되지 않으므로 새롭고 더 강력한 T2I 모델이 등장할수록 편집 품질이 자동으로 향상된다.

#### 🔹 파인튜닝 불필요 (Parameter-Free)

LEDITS++는 텍스트-이미지 확산 모델을 사용한 효율적이고 다목적인 이미지 편집을 위한 새로운 방법으로, 파인튜닝이나 최적화를 전혀 요구하지 않는 파라미터-프리(parameter-free) 솔루션이다. 이는 임의의 도메인(예술 이미지, 의료 이미지, 위성 이미지 등)에서도 추가 학습 없이 바로 적용이 가능함을 의미한다.

#### 🔹 의미론적 마스킹의 도메인 확장성

암묵적 마스킹 기법은 이미 이미지에 존재하지 않는 편집 개념과 관련된 영역도 포착할 수 있음을 실험적으로 입증하였다. 이는 단순히 존재하는 객체의 변환뿐만 아니라 새로운 개념의 추가/삽입에도 일반화됨을 보여준다.

#### 🔹 다중 편집 간 독립성

여러 편집에서 각 편집 프롬프트에 대한 전용 마스크를 계산함으로써 해당 가이던스 항들이 서로 독립적으로 유지되어 상호 간섭을 최소화한다. 편집 수가 증가해도 성능이 쉽게 저하되지 않아 복잡한 시나리오에서의 일반화 가능성이 높다.

또한 이 방법은 AI 생성 이미지와 실제 사진 모두에서 동일하게 잘 작동한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 📊 관련 연구 계보

```
[2020] DDPM (Ho et al.)
   → [2021] DDIM (Song et al.) — 결정론적 역변환 도입
   → [2022] Prompt-to-Prompt (Hertz et al.) — Cross-Attention 조작
   → [2022] Imagic — 파인튜닝 기반 편집
   → [2023] InstructPix2Pix (Brooks et al.) — 대규모 학습 기반
   → [2023] SEGA (Brack et al.) — 의미론적 가이던스
   → [2023] LEDITS (Tsaban et al.) — DDPM 역변환 + SEGA
   → [2023] Pix2Pix-Zero — DDIM 역변환 + 최적화
   → [2023] Edit Friendly DDPM (Huberman-Spiegelglas et al.)
   → [2023/2024] LEDITS++ (Brack et al.) ← 현재 논문 (CVPR 2024)
   → [2024] MGIE, MagicBrush — LLM/MLLM 통합
```

InstructPix2Pix는 이미지 편집 능력을 부여하기 위해 대규모 확산 모델을 계속 학습시키는 방식이며, 개별 입력 이미지에 대한 파인튜닝으로 생성을 제약하는 방식은 유용하지만 계산 비용이 매우 높다.

초기 IIE(Instruction-based Image Editing)의 이정표로는 GPT-3와 Prompt-to-Prompt로 합성 데이터를 학습한 InstructPix2Pix가 있으며, MagicBrush, UltraEdit, HumanEdit, AnyEdit 등 고품질 큐레이션 데이터셋을 활용한 후속 연구들이 성능을 향상시켰다.

최근에는 LLM이 통합되어 명령 이해도를 향상시키는 연구가 증가하고 있으며, MGIE와 SmartEdit은 다중모달 대형 언어 모델(MLLM)을 활용하여 정밀한 가이던스를 제공한다.

| 방법 | 파인튜닝 | 동시 다중 편집 | 의미론적 마스킹 | 속도 |
|---|---|---|---|---|
| **Imagic** | ✅ 필요 | ❌ | ❌ | 느림 |
| **InstructPix2Pix** | ✅ 대규모 학습 | ❌ | ❌ | 빠름 |
| **Pix2Pix-Zero** | ❌ | ❌ | ❌ | 느림 (최적화) |
| **Prompt-to-Prompt** | ❌ | 제한적 | Cross-Attn | 보통 |
| **LEDITS** | ❌ | 제한적 | ✅ | 느림 (DDPM) |
| **LEDITS++** | ❌ | ✅ 완전 지원 | ✅ 교집합 마스크 | **매우 빠름** |
| **MGIE / SmartEdit** | ✅ LLM 통합 | 제한적 | ✅ | 보통 |

---

## 5. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 🔭 앞으로의 연구에 미치는 영향

**① 새로운 벤치마크 기준 제시**

LEDITS++는 새로운 벤치마크인 TEdBench++를 제안함으로써, 텍스트 기반 이미지 편집 평가를 위한 더 총체적이고 일관성 있는 테스트베드를 제공하며, 다중-조건부 편집, 객체 제거, 스타일 전환, 복잡한 교체 등 다양한 편집 과제를 포함하는 평가 기준을 확립하였다.

**② 파인튜닝-프리 패러다임의 확산**

편집 친화적 노이즈 공간 특성과 완전한 입력 재구성을 훨씬 빠른 SDE 솔버에 유도함으로써, DDPM 샘플링 방식의 특성을 더 빠른 솔버로 확장하는 원리를 제시하였다. 이를 통해 역변환과 추론을 합해 단 20단계의 확산 스텝으로 편집이 가능하다.

**③ 다중 편집 격리의 표준화**

지식의 범위에서 LEDITS++는 복수의 편집을 본질적으로 독립적으로 지원하는 유일한 확산 기반 이미지 편집 방법으로, 더 복잡한 이미지 조작을 가능하게 한다.

**④ 실용적 통합 및 접근성**

LEDITS++는 `diffusers` 라이브러리에 완전히 통합되어 있으며, Colab과 HuggingFace에서 인터랙티브 데모도 제공되어 연구자와 실무자가 쉽게 접근할 수 있다.

---

### ⚠️ 앞으로 연구 시 고려할 점

**1. 더 강력한 기반 모델 활용**
기반 확산 모델의 능력이 LEDITS++의 편집 성공 여부와 품질에 직접적인 영향을 미치므로, DiT(Diffusion Transformer), FLUX 등 최신 T2I 모델 백본에 LEDITS++의 역변환 메커니즘을 적용하는 연구가 필요하다.

**2. 구조적 편집의 한계 극복**
중요한 구조적 변화를 포함하는 복잡한 편집에서 여전히 아티팩트가 발생할 수 있으며, 최적의 결과를 위해 신중한 프롬프트 엔지니어링이 필요하다는 한계를 극복하기 위한 더 강력한 구조적 제어 기법 연구가 필요하다.

**3. LLM/MLLM 통합**
최근 LLM을 통합하여 명령 이해도를 향상시키는 연구 트렌드를 반영하여, LEDITS++의 프롬프트 엔지니어링 의존성을 줄이고 더 자연스러운 명령어 처리를 위한 MLLM 통합 방향을 고려해야 한다.

**4. 실시간 편집 및 동영상으로의 확장**
LEDITS++의 목표는 사용자가 반복적으로 모델과 상호작용하며 다양한 편집을 탐색할 수 있는 빠른 탐색적 워크플로우를 가능하게 하는 것이다. 이를 더욱 발전시켜 비디오 편집 또는 실시간 스트리밍 환경에서의 적용 가능성을 탐구할 필요가 있다.

**5. 평가 메트릭의 다양화**
CLIP 점수(높을수록 좋음)와 LPIPS 유사도(낮을수록 좋음)를 주요 평가 지표로 사용하지만, 사람 지각과의 정렬, 의미론적 일관성, 시각적 아티팩트 측정 등 더 종합적인 평가 체계의 도입이 필요하다.

---

## 📚 참고 자료 및 출처

| # | 자료 | 링크/출처 |
|---|---|---|
| 1 | **LEDITS++ 공식 arXiv 논문** (v2) | https://arxiv.org/abs/2311.16711 |
| 2 | **CVPR 2024 공식 논문 PDF** | https://openaccess.thecvf.com/content/CVPR2024/papers/Brack_LEDITS_Limitless_Image_Editing_using_Text-to-Image_Models_CVPR_2024_paper.pdf |
| 3 | **IEEE Xplore 게재 버전** | https://ieeexplore.ieee.org/iel8/10654794/10654797/10656542.pdf |
| 4 | **NeurIPS Creativity Workshop 2023 버전** | https://neuripscreativityworkshop.github.io/2023/papers/ml4cd2023_paper18.pdf |
| 5 | **공식 GitHub 구현** | https://github.com/ml-research/ledits_pp |
| 6 | **프로젝트 페이지 (HuggingFace)** | https://leditsplusplus-project.static.hf.space/index.html |
| 7 | **HuggingFace Paper 페이지** | https://huggingface.co/papers/2311.16711 |
| 8 | **HuggingFace Diffusers API 문서** | https://huggingface.co/docs/diffusers/en/api/pipelines/ledits_pp |
| 9 | **arXiv HTML 버전** | https://arxiv.org/html/2311.16711v2 |
| 10 | **NeurIPS 2023 발표 페이지** | https://neurips.cc/virtual/2023/81377 |
| 11 | **TEdBench++ 데이터셋 (HuggingFace)** | https://huggingface.co/datasets/AIML-TUDA/TEdBench_plusplus |
| 12 | **LEDITS (선행 연구)**: Tsaban et al. "LEDITS: Real Image Editing with DDPM Inversion and Semantic Guidance" | https://arxiv.org/abs/2307.00522 |
| 13 | **관련 비교 연구** InstructPix2Pix: Brooks et al. (CVPR 2023) | arXiv:2211.09800 |
| 14 | **Semantic Scholar 논문 페이지** | https://www.semanticscholar.org/paper/LEDITS%2B%2B |
| 15 | **I2EBench 비교 벤치마크 연구** (NeurIPS 2024) | https://papers.nips.cc/paper_files/paper/2024/file/48fecef47b19fe501d27d338b6d52582 |
