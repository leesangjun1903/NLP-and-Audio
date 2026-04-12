
# One-Step Image Translation with Text-to-Image Models

> **논문 정보**
> - **제목:** One-Step Image Translation with Text-to-Image Models
> - **저자:** Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, Jun-Yan Zhu (CMU & Adobe)
> - **게재:** arXiv:2403.12036 (2024년 3월 18일)
> - **코드:** https://github.com/GaParmar/img2img-turbo

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 기존 조건부 확산 모델(Conditional Diffusion Model)의 두 가지 한계, 즉 반복적 디노이징 과정으로 인한 **느린 추론 속도**와 모델 파인튜닝을 위한 **페어드 데이터(Paired Data) 의존성**을 해결하고자 한다. 이를 위해 **적대적 학습 목표(Adversarial Learning Objectives)** 를 통해 단일 단계(single-step) 확산 모델을 새로운 태스크와 도메인에 적응시키는 범용 방법론을 제안한다.

**주요 기여 요약:**

| 기여 항목 | 내용 |
|---|---|
| **단일 스텝 추론** | 기존 수십~수백 스텝 → **1 스텝**으로 추론 |
| **Unpaired 학습 지원** | 페어드 데이터 없이 이미지 변환 가능 |
| **CycleGAN-Turbo** | Unpaired 시나리오를 위한 모델 |
| **pix2pix-Turbo** | Paired 시나리오를 위한 모델 |
| **범용 GAN 백본** | 단일 스텝 확산 모델이 GAN 학습 목표의 강력한 백본임을 입증 |

저자들의 주장에 따르면, 이 연구는 텍스트-이미지 모델을 이용한 **원-스텝 이미지 변환을 최초로 달성**한 작업이다.

---

## 2. 해결하고자 하는 문제, 제안하는 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

조건부 확산 모델은 공간적 조건(spatial conditioning)과 텍스트 프롬프트를 기반으로 이미지를 생성할 수 있도록 사용자에게 강력한 도구를 제공해 왔지만, 이러한 모델들은 두 가지 근본적인 도전 과제에 직면해 있다.

**문제 1: 느린 추론 속도**

확산 모델은 이미지 생성 분야에서 선두에 위치하지만, 복잡한 아키텍처와 상당한 연산 요구량으로 인해 반복적 샘플링 과정에서 심각한 지연이 발생한다.

**문제 2: Paired Data 의존성**

기존의 ControlNet과 같은 표준 확산 어댑터를 단일 스텝 환경에 직접 적용하는 것은 비효율적이다. 단일 스텝 모델에서는 노이즈 맵이 출력 구조에 직접 영향을 미치기 때문에, 추가 어댑터 브랜치를 통해 노이즈 맵과 입력 조건 정보를 함께 공급할 경우 네트워크에 **충돌하는 정보(conflicting information)** 가 발생한다.

또한, SD-Turbo 모델의 Encoder-UNet-Decoder로 이루어진 다단계 파이프라인이 불완전한 재구성을 초래하면서 **입력 이미지의 많은 시각적 디테일이 손실**되며, 이는 낮-밤 변환과 같은 실사 이미지 변환 태스크에서 특히 두드러진다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 핵심 아이디어

핵심 아이디어는 SD-Turbo와 같이 **사전 학습된 텍스트 조건부 단일 스텝 확산 모델**을 적대적 학습 목표를 통해 새로운 도메인과 태스크에 효율적으로 적응시키는 것이다.

첫 번째로, 조건 정보(conditioning information)를 UNet의 **노이즈 인코더 브랜치에 직접** 공급함으로써, 노이즈 맵과 입력 제어 신호 간의 충돌을 방지하고 네트워크가 새로운 제어에 직접 적응할 수 있도록 한다.

#### (2) Unpaired 학습: CycleGAN-Turbo의 손실 함수

CycleGAN-Turbo는 기존 CycleGAN의 사이클 일관성(cycle consistency) 손실에서 영감을 받아 구성된다. 두 도메인 $X$, $Y$ 간의 변환을 학습하기 위해 다음 손실 항목들을 사용한다.

**① 적대적 손실 (Adversarial Loss)**

$$\mathcal{L}_{GAN}(G, D_Y, X, Y) = \mathbb{E}_{y \sim p_{data}(y)}[\log D_Y(y)] + \mathbb{E}_{x \sim p_{data}(x)}[\log(1 - D_Y(G(x)))]$$

**② 사이클 일관성 손실 (Cycle Consistency Loss)**

$$\mathcal{L}_{cyc}(G, F) = \mathbb{E}_{x \sim p_{data}(x)}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y \sim p_{data}(y)}[\|G(F(y)) - y\|_1]$$

**③ DINO 구조 보존 손실 (Structure Preservation Loss)**

입력 이미지 $x$와 재구성된 이미지 사이의 구조적 일관성 유지를 위해 DINO 특징 유사도를 활용:

$$\mathcal{L}_{struct} = \mathbb{E}_{x}[\|f_{DINO}(x) - f_{DINO}(\hat{x})\|_2]$$

**④ 전체 Unpaired 학습 목표 (CycleGAN-Turbo)**

$$\mathcal{L}_{total}^{unpaired} = \mathcal{L}_{GAN} + \lambda_{cyc}\mathcal{L}_{cyc} + \lambda_{struct}\mathcal{L}_{struct}$$

#### (3) Paired 학습: pix2pix-Turbo의 손실 함수

Paired 변환의 학습 목표는 세 가지 손실 함수로 구성된다: 재구성 손실, LPIPS 지각 손실, 그리고 CLIP 손실의 전체 학습 목표로 이루어진다.

**① L1 재구성 손실:**

$$\mathcal{L}_{rec} = \mathbb{E}_{(x,y)}[\|G(x) - y\|_1]$$

**② LPIPS 지각 손실:**

$$\mathcal{L}_{LPIPS} = \mathbb{E}_{(x,y)}[\text{LPIPS}(G(x), y)]$$

**③ 적대적 손실 (Adversarial Loss):**

$$\mathcal{L}_{adv} = \mathbb{E}_{x}[\log D(G(x))]$$

**④ CLIP 방향 손실 (CLIP Directional Loss):**

$$\mathcal{L}_{CLIP} = 1 - \frac{\Delta I \cdot \Delta T}{\|\Delta I\| \|\Delta T\|}$$

여기서 $\Delta I = E_I(G(x)) - E_I(x)$, $\Delta T = E_T(t_{target}) - E_T(t_{source})$.

**⑤ 전체 Paired 학습 목표 (pix2pix-Turbo):**

$$\mathcal{L}_{total}^{paired} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{LPIPS}\mathcal{L}_{LPIPS} + \lambda_{adv}\mathcal{L}_{adv} + \lambda_{CLIP}\mathcal{L}_{CLIP}$$

---

### 2.3 모델 구조

본 방법론의 핵심은 Encoder, UNet, Decoder의 세 가지 모듈을 **단일 End-to-End 학습 가능한 아키텍처**로 통합하는 것이다. 이를 위해 **LoRA(Low-Rank Adaptation)** 를 활용하여 원래 네트워크를 새로운 제어 및 도메인에 적응시키며, 과적합을 줄이고 파인튜닝 시간을 단축한다. 또한 입력의 고주파 디테일을 보존하기 위해 **Zero-Convolution을 통한 스킵 연결(Skip Connection)** 을 인코더와 디코더 사이에 도입한다.

아키텍처를 도식화하면 다음과 같다:

```
입력 이미지 x
    │
    ├──────────────────────────┐  ← Skip Connection (Zero-Conv)
    ↓                          │
[VAE Encoder + LoRA]           │
    │                          │
    ↓                          │
[UNet (SD-Turbo) + LoRA]       │
    │                          │
    ↓                          │
[VAE Decoder + LoRA] ←─────────┘
    │
    ↓
출력 이미지 y
```

Zero-Conv를 통한 스킵 연결은 인코더와 디코더 사이의 고주파 세부 정보를 보존하는 역할을 하며, 이 아키텍처는 CycleGAN 및 pix2pix와 같은 조건부 GAN 학습 목표를 위한 **플러그 앤 플레이(Plug-and-Play)** 모델로 기능한다.

Unpaired 모델의 학습 가능한 파라미터는 LoRA 가중치, Zero-Conv 레이어, UNet의 첫 번째 Convolution 레이어를 포함하여 **총 330 MB**이다.

---

### 2.4 성능 향상

SD-Turbo와 같은 단일 스텝 확산 모델을 적대적 학습을 통해 새로운 태스크 및 도메인에 적응시킴으로써, **사전 학습된 확산 모델의 내부 지식을 활용**하면서도 효율적인 추론(512×512 이미지 기준 약 **0.3초**)을 달성한다.

Unpaired 설정에서 CycleGAN-Turbo는 낮-밤 변환, 기상 효과 추가/제거 등의 다양한 장면 변환 태스크에서 **기존 GAN 기반 및 확산 기반 방법들을 모두 능가**하며, Paired 설정에서는 ControlNet 등 최신 연구들과 동등한 수준을 달성한다.

정량적으로, InstructPix2Pix는 입력 구조 보존에 실패하여 DINO-Struct 점수가 7.6으로 본 논문의 1.4에 비해 크게 높으며, FID 점수도 170.8 대 137.0으로 본 논문의 방법이 월등히 우수하다.

추론 시간은 A6000 GPU에서 512×512 이미지 기준 **0.29초**, A100 GPU에서 **0.11초**를 달성한다.

---

### 2.5 한계점

표준 ControlNet 어댑터를 단일 스텝 환경에 직접 적용하는 것은 비효율적이다. 단일 스텝 모델에서는 노이즈 맵이 출력 구조에 직접 영향을 미치고, 추가 어댑터 브랜치를 통해 노이즈 맵과 입력 조건 정보를 동시에 공급하면 **네트워크 내에서 충돌 정보가 발생**한다.

다양한 노이즈 맵으로 동일한 조건 이미지를 적용하면 지각적으로 유사한 출력 이미지가 생성되어, **원래 SD-Turbo 인코더의 특징이 무시**되는 현상이 나타난다.

**추가 한계점:**
- SD-Turbo라는 특정 기반 모델에 강하게 결합되어 있어 다른 아키텍처로의 이식성(portability)이 제한됨
- 추론 다양성(stochasticity) 제어가 일부 태스크에서 제한적
- 매우 복잡한 기하학적 변환에는 어려움이 있을 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 소규모 데이터셋에서의 일반화

저자들은 다양한 크기의 데이터셋에서 방법론의 효과를 평가했다. 낮-밤 변환 데이터셋(36,728개의 낮 이미지, 27,971개의 밤 이미지)을 기반으로, 원본 데이터셋의 단계적으로 축소된 부분집합인 1,000개, 100개, 심지어 **10개 이미지**로 학습했을 때에도 모델이 동작함을 확인했다.

이는 **대규모 사전 학습된 확산 모델의 지식(knowledge prior)** 이 작은 도메인 특화 데이터셋으로도 효과적으로 파인튜닝될 수 있음을 시사한다.

### 3.2 LoRA를 활용한 일반화

LoRA를 활용하여 원래 네트워크를 새로운 제어 및 도메인에 적응시킴으로써, **과적합을 줄이고 파인튜닝 시간을 단축**한다. 이는 다양한 도메인에서 모델의 일반화 성능을 높이는 핵심 메커니즘이다.

### 3.3 플러그 앤 플레이 아키텍처를 통한 범용성

이 아키텍처는 다재다능(versatile)하여 **CycleGAN 및 pix2pix와 같은 조건부 GAN 학습 목표를 위한 플러그 앤 플레이 모델**로 기능한다. 즉, 동일한 Generator 구조 위에 다양한 GAN Objective를 조합할 수 있다.

### 3.4 텍스트 프롬프트를 통한 제어

입력 노이즈 맵을 변화시킴으로써 동일한 입력 조건에서 **다양한 출력을 생성**할 수 있으며, 텍스트 프롬프트를 변경함으로써 **출력 스타일을 제어**할 수 있다.

### 3.5 일반화 향상을 위한 가능성 정리

| 일반화 측면 | 현재 수준 | 향후 발전 가능성 |
|---|---|---|
| **데이터 효율성** | 10개 이미지로도 동작 | 제로샷(zero-shot) 변환으로 확장 가능 |
| **도메인 다양성** | 날씨, 낮-밤 변환 등 | 의료 이미지, 위성 이미지 등 전문 도메인 |
| **스타일 제어** | 텍스트 프롬프트로 스타일 조절 | 멀티모달 조건 입력 통합 |
| **아키텍처 범용성** | SD-Turbo 기반 특화 | SDXL-Turbo, FLUX 등 최신 모델로 확장 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 Image-to-Image Translation 연구 계보

| 연구 | 연도 | 주요 방법 | 한계 |
|---|---|---|---|
| **pix2pix** (Isola et al.) | 2017 | Paired cGAN | Paired 데이터 필요 |
| **CycleGAN** (Zhu et al.) | 2017 | Cycle-Consistency Loss | GAN 훈련 불안정성 |
| **StarGAN v2** (Choi et al.) | 2020 | 다중 도메인 변환 | 유한한 도메인 수 |
| **InstructPix2Pix** (Brooks et al.) | 2023 | LLM 기반 편집 지침 | 느린 추론, 구조 보존 취약 |
| **ControlNet** (Zhang et al.) | 2023 | Zero-Conv 기반 공간 제어 | 다중 스텝 추론 필요 |
| **CycleGAN-Turbo / pix2pix-Turbo** | 2024 | 단일 스텝 + 적대적 학습 | SD-Turbo 의존성 |

### 4.2 핵심 연구들과의 비교

**① InstructPix2Pix (Brooks et al., CVPR 2023) vs. CycleGAN-Turbo**

표 3에서 본 방법론은 CycleGAN과 Instruct-Pix2Pix 대비 Clear to Foggy 변환 작업을 제외한 모든 데이터셋에서 우수한 성능을 보인다. Clear to Foggy 변환에서는 InstructPix2Pix가 더 예술적인 안개 이미지를 출력하여 사용자 선호도가 높았지만, 입력 구조 보존(DINO-Struct: 7.6 vs 1.4)과 분포 매칭(FID: 170.8 vs 137.0) 측면에서 본 방법론이 크게 우세하다.

**② ControlNet (Zhang et al., ICCV 2023) vs. pix2pix-Turbo**

Paired 설정에서, pix2pix-Turbo는 Sketch2Photo 및 Edge2Image 작업에서 ControlNet과 동등한 성능을 보이면서도 **단일 스텝 추론**이라는 이점을 갖는다.

ControlNet 인코더를 사용하는 Config B는 FID 측면에서 비교 가능한 성능을 보이지만, **DINO-Structure 거리가 현저히 높아** 입력 구조와 일치하는 데 어려움을 겪으며, 특히 장면 구조를 바꾸거나 없는 건물을 생성하는 등 환각(hallucination) 현상을 보인다.

**③ Diffusion Model Compression (Kim et al., ACCV 2024) vs. CycleGAN-Turbo**

대규모 T2I 확산 모델의 발전으로 다양한 이미지-투-이미지(I2I) 응용이 등장했지만, 이러한 I2I 모델들의 실용적인 활용은 **대용량 모델 크기와 반복적 디노이징의 연산 부담**으로 저해되고 있다. 이 논문은 이를 위한 새로운 압축 방법을 제안한다.

해당 압축 방법은 InstructPix2Pix, StableSR, ControlNet에 대해 각각 39.2%, 56.4%, 39.2%의 모델 크기 감소와 81.4%, 68.7%, 31.1%의 지연 감소를 달성하지만, 여전히 다중 스텝 추론을 기반으로 하여 CycleGAN-Turbo의 단일 스텝 방식과는 근본적으로 다른 접근이다.

**④ CycleVAR (2025) vs. CycleGAN-Turbo**

CycleVAR의 실험 결과에 따르면, CycleGAN-Turbo를 능가하는 성능을 보이며, Softmax Relaxed Quantization이라는 새로운 접근법을 통해 코드북 선택을 연속적 확률 혼합 프로세스로 재구성하는 방식을 제안한다. 이는 CycleGAN-Turbo가 후속 연구들에 직접적인 비교 기준(baseline)이 되고 있음을 보여준다.

**⑤ UFOGen (2023) vs. CycleGAN-Turbo**

UFOGen은 초고속 단일 스텝 텍스트-이미지 생성 및 다양한 다운스트림 태스크를 위해 설계된 새로운 생성 모델로, 효율적 생성 모델 분야에서 중요한 발전을 나타낸다. 그러나 UFOGen은 순수 T2I 생성에 초점을 두는 반면, CycleGAN-Turbo는 I2I 변환 특화 설계를 갖는다는 차별점이 있다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

이 연구는 **단일 스텝 확산 모델이 다양한 GAN 학습 목표를 위한 강력한 백본으로 기능할 수 있음**을 제안하며, 이 분야 연구의 새로운 방향을 제시한다.

**① 효율적 이미지 편집 패러다임의 전환**

기존의 "많은 디노이징 스텝 → 높은 품질"이라는 패러다임에서, "단일 스텝 + 강력한 사전 학습 지식 활용"이라는 새로운 방향을 제시한다. 이는 실시간 이미지 편집 응용 프로그램(모바일 앱, 비디오 스트리밍 등)의 가능성을 크게 확대한다.

**② GAN과 Diffusion Model의 융합 연구 촉진**

Score Distillation을 활용하여 대규모 이미지 확산 모델을 교사 신호로 활용하면서 적대적 손실을 결합해, 1~2 샘플링 스텝의 저단계 환경에서도 높은 이미지 충실도를 보장하는 연구들이 등장하고 있다. CycleGAN-Turbo는 이러한 흐름의 선구적 역할을 한다.

**③ Unpaired 이미지 변환의 재부상**

CycleGAN-Turbo는 낮-밤 변환, 안개/눈/비와 같은 기상 효과 추가/제거 등 다양한 장면 변환 태스크에서 기존 GAN 및 확산 기반 방법들을 능가하는 성능을 보였다. 이는 페어드 데이터 수집이 불가능한 현실 세계 응용에서 Unpaired 학습의 실용성을 크게 향상시켰다.

---

### 5.2 앞으로의 연구 시 고려할 점

#### 🔬 기술적 고려 사항

| 고려 항목 | 세부 내용 |
|---|---|
| **기반 모델 확장성** | SD-Turbo 외에 SDXL-Turbo, FLUX-Schnell 등 최신 단일 스텝 모델로의 이식 검토 |
| **노이즈-조건 충돌 문제** | 단일 스텝 환경에서 노이즈 맵과 조건 입력 충돌 문제에 대한 근본적인 해결 방안 연구 |
| **고해상도 처리** | 현재 512×512 해상도 위주에서 고해상도(1024×1024 이상) 이미지 처리 성능 향상 |
| **영상(Video) 변환** | 시간적 일관성을 갖춘 단일 스텝 비디오-투-비디오 변환으로의 확장 |
| **다중 조건 결합** | 텍스트, 깊이, 포즈 등 다양한 조건을 동시에 활용하는 멀티컨디셔닝 연구 |

#### 🌐 일반화 관련 고려 사항

본 방법론은 이미지 쌍 없이도 학습 가능하다는 핵심 장점이 있으므로, 다음 방향의 연구가 중요하다:
- **의료 영상 도메인**: MRI-CT 변환, 다양한 모달리티 간 변환
- **위성/항공 이미지**: 계절 변화, 재난 전후 비교 등
- **자율주행**: 주간-야간, 맑음-악천후 등의 도메인 적응(Domain Adaptation)

#### ⚠️ 윤리적 고려 사항

- **딥페이크 남용 방지**: 단일 스텝의 빠른 추론 속도는 악의적 이미지 조작에 악용될 수 있으므로, 생성 이미지 탐지(Detection) 기술과의 공동 연구가 필요
- **편향성(Bias) 문제**: 사전 학습된 Stable Diffusion 모델의 데이터 편향이 이미지 변환 결과에 그대로 반영될 수 있음
- **저작권 및 데이터 출처**: 생성된 이미지의 저작권 귀속 문제

---

## 📚 참고 자료 (출처)

| # | 제목 | 출처 |
|---|---|---|
| 1 | **One-Step Image Translation with Text-to-Image Models** (주 논문) | arXiv:2403.12036, https://arxiv.org/abs/2403.12036 |
| 2 | 논문 HTML 전문 | https://arxiv.org/html/2403.12036v1 |
| 3 | GitHub 공식 코드 | https://github.com/GaParmar/img2img-turbo |
| 4 | Semantic Scholar 논문 페이지 | https://www.semanticscholar.org/paper/One-Step-Image-Translation.../b12bfd15 |
| 5 | HuggingFace Paper Page | https://huggingface.co/papers/2403.12036 |
| 6 | ADS (NASA Astrophysics Data System) Abstract | https://ui.adsabs.harvard.edu/abs/2024arXiv240312036P |
| 7 | OpenVINO 공식 문서 (pix2pix-turbo 구현 참고) | https://docs.openvino.ai/2024/notebooks/sketch-to-image-pix2pix-turbo-with-output.html |
| 8 | **SDXS: Real-Time One-Step Latent Diffusion Models** | arXiv:2403.16627, https://huggingface.co/papers/2403.16627 |
| 9 | **Diffusion Model Compression for Image-to-Image Translation** (ACCV 2024) | arXiv:2401.17547, https://arxiv.org/abs/2401.17547 |
| 10 | **Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)** | GitHub: https://github.com/lllyasviel/ControlNet |
| 11 | **Frequency-Controlled Diffusion Model for Versatile Text-Guided I2I Translation** | arXiv:2407.03006 |
| 12 | The Moonlight Literature Review | https://www.themoonlight.io/en/review/one-step-image-translation-with-text-to-image-models |

---

> ⚠️ **정확도 관련 고지:** 본 답변에서 수식(손실 함수 구체적 파라미터 가중치 $\lambda$ 값 등)은 논문의 원문 HTML 및 공식 코드를 기반으로 재구성한 것입니다. 정확한 하이퍼파라미터 값은 원논문(arXiv:2403.12036)의 실험 섹션을 직접 확인하시기를 권장드립니다.
