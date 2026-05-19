
# AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction

> **📌 논문 정보**
> - **제목**: AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction
> - **저자**: Zhen Xing, Qi Dai, Zejia Weng, Zuxuan Wu, Yu-Gang Jiang
> - **소속**: Fudan University, Microsoft Research Asia
> - **arXiv**: [2406.06465](https://arxiv.org/abs/2406.06465) (2024.06.10)
> - **학회**: ICCV 2025 수록 확인

---

## 1. 핵심 주장 및 주요 기여 요약

### ✅ 핵심 주장

Text-guided video prediction(TVP)은 초기 프레임으로부터 텍스트 지시(instruction)에 따라 미래 프레임의 움직임을 예측하는 태스크로, 가상현실(VR), 로보틱스, 콘텐츠 생성 등 다양한 분야에 폭넓게 활용될 수 있다.

기존 TVP 방법들은 주로 제한적인 규모의 비디오 데이터셋으로 인해 프레임 일관성(frame consistency)과 시간적 안정성(temporal stability)에서 어려움을 겪는다. 사전 학습된 Image2Video 확산 모델(Image2Video diffusion model)은 비디오 동역학(video dynamics)에 대한 우수한 사전 지식(prior)을 보유하고 있으나, 텍스트 제어 능력이 부재하다. 따라서 Image2Video 모델을 전이(transfer)하여 비디오 동역학 사전 지식을 활용하면서 동시에 지시(instruction) 제어를 주입해 제어 가능한 비디오를 생성하는 것이 핵심 과제이다.

### ✅ 주요 기여 3가지

이를 해결하기 위해 ① MLLM(Multi-Modal Large Language Model)을 도입하여 초기 프레임과 텍스트 지시를 기반으로 미래 비디오 상태를 예측하고, ② 지시(instruction)와 프레임을 조건부 임베딩으로 통합하는 **DQFormer(Dual Query Transformer)** 아키텍처를 설계하였으며, ③ 최소한의 학습 비용으로 범용 비디오 확산 모델을 특정 시나리오에 빠르게 전이할 수 있는 **Long-Short Term Temporal Adapter** 및 **Spatial Adapter**를 개발하였다.

저자들은 잘 사전 학습된 비디오 확산 모델을 도메인 특화 비디오 생성 태스크로 전이하는 것을 최초로 탐구하였으며, 실험을 통해 이 방법의 실현 가능성과 광범위한 잠재력을 입증하여 미래 연구의 길을 열었다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 🔴 2-1. 해결하고자 하는 문제

Image2Video 확산 모델을 TVP 태스크에 적응시키는 데 있어 두 가지 주요 과제가 존재한다. 첫째, 텍스트 조건(textual condition)을 어떻게 설계하고 확산 모델에 주입하여 비디오 생성을 안내할 것인가, 둘째, 어떻게 낮은 학습 비용으로 현재 모델을 목표 데이터셋에 적응시켜 생성 비디오를 실제 시나리오에 더 근접하게 만들 것인가이다.

단일 텍스트 지시만으로는 비디오의 시간적 동역학을 완전히 포착하기 어렵기 때문에, 초기 프레임과 텍스트 지시로부터 미래 비디오의 발전 상태를 예측하기 위해 MLLM을 도입한다.

범용 오픈소스 모델 중 가장 발전된 방법인 Stable Video Diffusion(SVD)은 대규모 비디오 데이터셋 사전학습의 스케일링 법칙을 검증하여 비디오 생성에서 최첨단 성능을 보인다. 본 논문은 SVD를 기반 모델(base model)로 채택하였으나, SVD의 공개 버전은 이미지-비디오 생성(image-to-video generation)만 지원하며 본질적으로 제어 불가능하다.

---

### 🔵 2-2. 제안하는 방법 (수식 포함)

#### (1) 확산 모델 기본 목적 함수 (Diffusion Model Training Objective)

Diffusion 모델의 표준 학습 손실은 다음과 같다:

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, t}\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t, \mathbf{c}\right)\right\|^2\right]$$

여기서:
- $\mathbf{x}_0$: 원본 비디오 잠재 벡터 (latent)
- $\mathbf{x}_t$: 타임스텝 $t$에서 노이즈가 추가된 잠재 벡터
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$: 추가된 가우시안 노이즈
- $\boldsymbol{\epsilon}_\theta$: 노이즈 예측 네트워크 (UNet 기반)
- $\mathbf{c}$: 조건(condition) — 텍스트 임베딩 + 비주얼 임베딩 포함

#### (2) 순방향 확산 과정 (Forward Diffusion)

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\,(1 - \bar{\alpha}_t)\,\mathbf{I}\right)$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t}(1-\beta_s)$이며, $\beta_s$는 노이즈 스케줄.

#### (3) 조건부 임베딩: MCondition 생성

DQFormer는 두 개의 브랜치로 구성되어 있다: 하나는 텍스트와 비주얼로부터 멀티모달 제어 정보를 학습하는 브랜치이고, 다른 하나는 텍스트 상태 조건을 프레임 단위 제어로 분해(decompose)하는 브랜치이다.

수식적으로 DQFormer의 두 브랜치를 표현하면:

$$\mathbf{e}_{\text{multi}} = \text{CrossAttn}\left(Q_{\text{visual}},\, K_{\text{text}},\, V_{\text{text}}\right)$$

$$\mathbf{e}_{\text{frame}} = \text{CrossAttn}\left(Q_{\text{frame}},\, K_{\text{state}},\, V_{\text{state}}\right)$$

$$\mathbf{c}_{\text{MCondition}} = \text{Concat}\left[\mathbf{e}_{\text{multi}},\, \mathbf{e}_{\text{frame}}\right]$$

최종적으로 멀티 조건(MCondition)은 Cross-Attention 메커니즘을 통해 UNet에 주입된다.

#### (4) Long-Short Term Temporal Adapter

Long-Short Term Temporal Adapter와 Spatial Adapter는 최소한의 학습 비용(minimal training costs)으로 범용 비디오 확산 모델을 특정 시나리오로 빠르게 전이시킬 수 있도록 설계되었다.

Temporal Adapter의 일반적 수식 형태:

- **Short-Term Temporal Adapter** (인접 프레임 간 지역적 시간 의존성 포착):

$$\mathbf{h}^{\text{short}} = \text{Conv3D}_{1\times 1 \times k}\left(\mathbf{h}_{\text{in}}\right) + \mathbf{h}_{\text{in}}$$

- **Long-Term Temporal Adapter** (전체 시퀀스에 걸친 전역적 시간 의존성 포착):

$$\mathbf{h}^{\text{long}} = \text{TemporalAttn}\left(\mathbf{h}_{\text{in}}\right) + \mathbf{h}_{\text{in}}$$

- **Spatial Adapter** (공간 구조 적응):

$$\mathbf{h}^{\text{spatial}} = W_{\text{down}}\,\sigma\!\left(W_{\text{up}}\,\mathbf{h}_{\text{in}}\right) + \mathbf{h}_{\text{in}}$$

> ⚠️ 주의: 위 Temporal/Spatial Adapter의 세부 수식은 논문 PDF의 전체 접근 없이 정확히 확인되지 않았으므로, 일반적인 adapter 설계 관례를 참조하여 기술하였습니다. 정확한 수식은 논문 원문을 직접 참조하시기 바랍니다.

---

### 🟢 2-3. 모델 구조 (Architecture)

AID의 전체 파이프라인은 다음 단계로 구성된다:

```
[초기 프레임 I₀] + [텍스트 지시 T]
         ↓
   [MLLM: 미래 상태 예측]
         ↓ (State Descriptions S)
   [DQFormer (Dual Query Transformer)]
   ├── Branch 1: Visual + Text → e_multi
   └── Branch 2: State → e_frame (frame-level control)
         ↓ MCondition
   [SVD 기반 Video UNet]
   ├── Spatial Adapter
   ├── Short-Term Temporal Adapter
   └── Long-Term Temporal Adapter
         ↓ Cross-Attention (MCondition)
   [생성된 미래 프레임 비디오]
```

- **기반 모델**: Stable Video Diffusion (SVD)
- 초기 $K$ 프레임과 이후 프레임의 마스크를 조건으로 사용하며, 텍스트 조건 또한 미래 프레임 예측을 안내하기 위해 모델에 주입된다.
- 미래 상태 예측을 위해 MLLM을 활용하고, 다양한 모달리티의 제어 조건을 통합하기 위한 Dual-Branch DQFormer 모듈을 설계하며, 적은 파라미터와 학습 비용으로 모델 전이를 가능하게 하는 Spatial/Temporal Adapter를 활용한다.

---

### 🟡 2-4. 성능 향상

실험 결과, AID는 Something Something V2(SSv2), Epic Kitchen-100, Bridge Data, UCF-101의 4개 데이터셋에서 최첨단(SoTA) 기법들을 크게 능가하였다. 특히 Bridge 데이터셋에서 FVD(Fréchet Video Distance) 기준 **91.2%**, SSv2에서 **55.5%** 개선을 달성하였다.

| 데이터셋 | FVD 개선율 |
|:---:|:---:|
| Bridge Data | **91.2%** ↑ |
| SSv2 | **55.5%** ↑ |
| Epic Kitchen-100 | SoTA 능가 |
| UCF-101 | SoTA 능가 |

---

### 🔴 2-5. 한계

모델은 현재 짧은 비디오 클립에 제한적이며, 더 길고 복잡한 비디오 시퀀스 생성에는 어려움이 있을 수 있다.

추가적으로 논문에서 언급된 한계:
- Sora, SVD와 같이 비공개 대규모 데이터셋으로 학습된 모델과의 격차가 여전히 존재한다.
- UCF-101과 같이 클래스 조건부 비디오 예측 벤치마크는 TVP 태스크에 완전히 적합하지 않아 평가 설정에 제약이 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

AID의 일반화 성능에 관한 논의는 다음 세 가지 측면에서 이루어진다:

### 📌 3-1. 도메인 간 일반화 (Cross-Domain Generalization)

AID는 Something Something V2, Epic Kitchen-100, Bridge Data, UCF-101이라는 **이질적인 4개 도메인**에서 모두 SoTA를 능가하는 성능을 달성함으로써, 특정 도메인에 과적합되지 않는 높은 일반화 능력을 입증하였다.

### 📌 3-2. 전이 학습 효율성 (Parameter-Efficient Transfer)

Long-Short Term Temporal Adapter와 Spatial Adapter는 최소한의 학습 비용으로 범용 비디오 확산 모델을 특정 시나리오로 빠르게 전이시킬 수 있어, 새로운 도메인에 적응할 때 전체 모델을 재학습하지 않아도 된다. 이는 **소수 데이터 환경**이나 **새로운 도메인**에서의 일반화를 크게 촉진한다.

Adapter 기반 PEFT(Parameter-Efficient Fine-Tuning) 전략의 파라미터 효율성:

$$\theta_{\text{train}} = \theta_{\text{adapter}} \ll \theta_{\text{total}}$$

즉, 전체 파라미터 $\theta_{\text{total}}$ 중 소수의 어댑터 파라미터 $\theta_{\text{adapter}}$만 학습함으로써 새로운 도메인 적응이 가능하다.

### 📌 3-3. 사전 학습 Prior의 활용

사전 학습된 Image2Video 모델은 비디오 동역학에 대한 우수한 사전 지식(prior)을 보유하고 있으며, 이를 전이함으로써 텍스트 제어가 가능한 비디오 생성을 달성하는 것이 핵심 아이디어이다. 이러한 대규모 사전 학습 Prior의 활용은 **소규모 도메인별 데이터셋에서도 높은 일반화 성능**을 가능하게 한다.

### 📌 3-4. MLLM 활용의 일반화 기여

단일 텍스트 지시만으로 비디오의 시간적 동역학을 완전히 포착하기 어렵기 때문에, MLLM을 통해 초기 프레임과 텍스트 지시로부터 미래 비디오의 발전 상태를 예측한다. MLLM은 다양한 도메인의 시각-언어 이해를 갖추고 있어, **도메인 변화에 강건한 시맨틱 표현**을 제공한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 모델/논문 | 연도 | 접근법 | 핵심 특징 | AID와의 비교 |
|---|---|---|---|---|
| **DDPM** (Ho et al.) | 2020 | 확산 모델 | 기초 확산 프레임워크 제시 | AID의 이론적 기반 |
| **InstructPix2Pix** | 2022 | 이미지 편집 | 텍스트 지시 기반 이미지 편집 | 이미지에 한정, 비디오 시간적 일관성 부재 |
| **Stable Diffusion** | 2022 | 잠재 확산 모델 | 대규모 이미지 생성 | 기존 TVP의 기반, 비디오 동역학 Prior 부족 |
| **SVD** (Blattmann et al.) | 2023 | Image2Video | 대규모 비디오 Prior | 텍스트 제어 불가, AID의 기반 모델 |
| **VideoCrafter** (Chen et al.) | 2023~2024 | T2V | 고품질 비디오 생성 | 일반 T2V, 특정 도메인 TVP에 특화 안됨 |
| **AVID** | 2024 | 확산 모델 적응 | 액션 조건부 비디오 예측 | 사전 학습된 image-to-video 모델이 주어진 액션 시퀀스에 정확한 비디오를 생성할 수 없는 문제를 해결하고자, 사전 학습 모델의 출력을 조정하여 정확한 액션 조건부 비디오 예측을 달성한다. AID와 유사하나 경량 어댑터 방식이 다름 |
| **AID (본 논문)** | 2024 | I2V 전이 학습 | MLLM + DQFormer + Temporal/Spatial Adapter | 4개 이질 도메인 SoTA |

### 핵심 차별점 정리

기존 TVP 방법들이 Stable Diffusion을 적응시켜 의미 있는 발전을 이루었으나, 제한된 규모의 비디오 데이터셋으로 인해 프레임 일관성과 시간적 안정성에서 한계를 보인 반면, AID는 이를 대규모 사전 학습된 I2V 모델의 Prior를 전이하는 방식으로 해결하였다는 점에서 차별화된다.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 🌟 5-1. 연구에 미치는 영향

**① I2V → TVP 전이 패러다임 확립**
저자들은 잘 사전 학습된 비디오 확산 모델을 도메인 특화 비디오 생성 태스크로 전이하는 것을 최초로 탐구하였으며, 이 방법의 실현 가능성과 광범위한 잠재력을 입증하여 미래 연구의 길을 열었다. 이는 **"대규모 기반 모델 → 경량 어댑터 → 도메인 특화"** 라는 연구 패러다임을 비디오 생성 분야에 본격적으로 도입한 선구적 연구로 평가된다.

**② 로보틱스 및 세계 모델(World Model) 연구로의 확장**
TVP의 응용 분야로서 가상현실, 로보틱스, 콘텐츠 생성 등이 거론되며, 특히 로보틱스 분야에서 AID와 같이 텍스트 지시 기반으로 미래 비디오 상태를 예측하는 모델은 **정책 학습(policy learning)**이나 **세계 모델(world model)** 연구에 직접적으로 활용될 수 있다.

**③ MLLM + Diffusion Model 결합 연구 촉진**
MLLM을 도입하여 초기 프레임과 텍스트 지시로부터 미래 비디오 상태를 예측하는 접근법은 MLLM과 생성 모델의 결합 연구를 가속화하는 방향성을 제시한다.

**④ Parameter-Efficient 비디오 적응 연구**
Long-Short Term Temporal Adapter 설계는 비디오 생성 모델의 PEFT(Parameter-Efficient Fine-Tuning) 연구에 새로운 방향을 제공하며, 다양한 비디오 이해 및 생성 태스크에 적용 가능하다.

---

### ⚠️ 5-2. 앞으로 연구 시 고려할 점

| 고려 사항 | 내용 |
|---|---|
| **장시간 비디오 생성** | 현재 모델이 짧은 클립에 제한되어 있어, 오토레그레시브(autoregressive) 방식의 확장 또는 계층적 예측 전략이 필요하다. |
| **데이터 편향 문제** | 이미지 생성의 성공은 수십억 규모의 공개 이미지-텍스트 데이터셋 덕분이며, 비디오 분야의 Sora, SVD 같은 모델은 비공개 대규모 데이터셋에 의존한다. 공개 가능한 대규모 비디오-텍스트 데이터 구축이 시급하다. |
| **텍스트 지시의 모호성** | 단일 텍스트 지시는 비디오의 시간적 동역학을 완전히 포착하지 못할 수 있어, 더 구체적이고 구조화된 지시 체계(예: 계층적 지시, 다중 턴 지시)가 연구되어야 한다. |
| **평가 메트릭 다양화** | FVD 중심 평가를 넘어, 텍스트-비디오 정렬도(text-video alignment), 물리적 타당성(physical plausibility), 사용자 연구(user study) 등 다각적인 평가 체계가 필요하다. |
| **실시간 추론 효율화** | MLLM + Diffusion Model의 조합은 추론 비용이 높으므로, 추론 가속화(distillation, 캐싱 등) 연구가 병행되어야 한다. |
| **도메인 일반화 검증 강화** | 현재 4개 데이터셋 외 의료, 자율주행, 스포츠 등 더 다양한 도메인에서의 검증과 도메인 편향(domain shift) 분석이 필요하다. |
| **다중 모달 조건 확장** | 텍스트+이미지 조건을 넘어, 오디오, 깊이 맵(depth map), 포즈(pose) 등 다양한 모달리티를 조건으로 활용하는 연구로 확장 가능하다. |

---

## 📚 참고 자료 및 출처

| # | 출처 |
|---|---|
| 1 | Zhen Xing et al., **"AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction"**, arXiv:2406.06465, 2024. [https://arxiv.org/abs/2406.06465](https://arxiv.org/abs/2406.06465) |
| 2 | AID 프로젝트 페이지: [https://chenhsing.github.io/AID](https://chenhsing.github.io/AID) |
| 3 | ICCV 2025 보충 자료: [https://openaccess.thecvf.com/content/ICCV2025/supplemental/Xing_AID_Adapting_Image2Video_ICCV_2025_supplemental.pdf](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Xing_AID_Adapting_Image2Video_ICCV_2025_supplemental.pdf) |
| 4 | AI Models FYI 논문 요약: [https://www.aimodels.fyi/papers/arxiv/aid-adapting-image2video-diffusion-models-instruction-guided](https://www.aimodels.fyi/papers/arxiv/aid-adapting-image2video-diffusion-models-instruction-guided) |
| 5 | Blattmann et al., **"Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets"**, arXiv:2311.15127, 2023. |
| 6 | Brooks et al., **"InstructPix2Pix: Learning to Follow Image Editing Instructions"**, CVPR 2023. |
| 7 | Russell Mendonca et al., **"AVID: Adapting Video Diffusion Models to World Models"**, arXiv:2410.12822, 2024. [https://arxiv.org/abs/2410.12822](https://arxiv.org/abs/2410.12822) |

---

> ⚠️ **정확도 안내**: 본 답변의 핵심 내용(문제 정의, 주요 기여, 성능 수치, 구조)은 논문의 arXiv 공개 자료 및 HTML 버전을 기반으로 확인된 정보입니다. **수식의 일부 세부 구성 요소**(특히 Temporal/Spatial Adapter의 내부 수식)는 논문 전체 PDF의 세부 내용을 직접 확인하지 못한 부분이 있어 일반적인 관례를 참조하였음을 명시합니다. 완전한 수식 확인을 위해서는 논문 원문 PDF([https://arxiv.org/pdf/2406.06465](https://arxiv.org/pdf/2406.06465)) 를 직접 참조하시기 바랍니다.
