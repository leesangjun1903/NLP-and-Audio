
# CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer

> **논문 정보**
> - **제목:** CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer
> - **저자:** Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang 외 13인 (THUDM / Zhipu AI)
> - **arXiv:** [arXiv:2408.06072](https://arxiv.org/abs/2408.06072) (2024년 8월 최초 공개, 2025년 3월 v3 업데이트)
> - **학술대회:** ICLR 2025 (Conference Paper)
> - **공개 코드/모델:** https://github.com/THUDM/CogVideo

---

## 1. 핵심 주장과 주요 기여 (Executive Summary)

CogVideoX는 Diffusion Transformer 기반의 대규모 텍스트-투-비디오 생성 모델로, 초당 16프레임, 768×1360 해상도로 텍스트 프롬프트에 정렬된 **10초짜리 연속 비디오**를 생성할 수 있습니다.

저자들은 텍스트-투-비디오 생성에서 효율적인 영상 데이터 모델링, 우수한 텍스트-비디오 정렬, 그리고 고품질 텍스트-비디오 데이터셋 구축이라는 세 가지 핵심 과제를 다루며, 이를 해결하기 위한 세 가지 핵심 기술 개발을 제시합니다: **3D Variational Autoencoder (VAE)**, **Expert Adaptive LayerNorm을 갖춘 Expert Transformer**, 그리고 **종합적인 텍스트-비디오 데이터 처리 파이프라인**입니다.

결과적으로, CogVideoX는 다수의 자동화된 기계 지표와 인간 평가 모두에서 **State-of-the-Art(SOTA) 성능**을 달성했습니다.

---

## 2. 해결하고자 하는 문제

이전 비디오 생성 모델들은 **제한적인 움직임과 짧은 지속 시간**의 문제를 겪었으며, 특히 텍스트를 기반으로 일관된 서사를 가진 비디오를 생성하는 것이 매우 어려웠습니다.

구체적으로 해결하고자 한 문제는 다음 세 가지입니다:

| 문제 | 기존 한계 |
|---|---|
| 영상 압축 효율 | 2D VAE는 프레임 개별 처리 → 시간적 일관성 부족, 깜박임(flicker) 발생 |
| 텍스트-비디오 정렬 | 단순 크로스-어텐션 방식의 한계로 언어-영상 간 심층 융합 불가 |
| 긴 영상 일관성 | 단일 해상도·단일 길이 학습으로 다양한 길이의 영상 생성 어려움 |

---

## 3. 제안하는 방법 (수식 포함)

### 3-1. 3D Causal VAE

모델의 핵심에는 **3D Causal VAE**가 있으며, 이는 공간적·시간적 차원 모두에서 비디오 데이터를 압축·복원합니다. 이 VAE는 3차원 합성곱(3D convolutions)을 사용하여 더 높은 압축 비율과 프레임 연속성을 달성하고, **시간적 인과 합성곱(temporally causal convolutions)**을 사용하여 미래 정보가 현재 프레임 예측에 영향을 미치지 않도록 합니다.

이 VAE는 **4×8×8의 압축 비율**을 달성합니다 — 시간 차원을 4배, 공간 각 차원을 8배 압축하여, 픽셀 공간에서 잠재 공간까지 총 **256배의 압축 인수**를 갖게 됩니다. 이는 트랜스포머가 처리해야 하는 시퀀스 길이를 대폭 줄입니다.

비디오 $\mathbf{X} \in \mathbb{R}^{T \times H \times W \times C}$가 3D Causal VAE 인코더 $\mathcal{E}$를 통해 잠재 벡터 $\mathbf{z}_v$로 압축됩니다:

$$\mathbf{z}_v = \mathcal{E}(\mathbf{X}), \quad \mathbf{z}_v \in \mathbb{R}^{\frac{T}{4} \times \frac{H}{8} \times \frac{W}{8} \times d}$$

VAE 학습 손실은 재구성 손실, 지각 손실(LPIPS), KL 정규화 손실, 그리고 GAN 손실의 합으로 구성됩니다:

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{L1}} + \lambda_{\text{LPIPS}} \mathcal{L}_{\text{LPIPS}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}} + \lambda_{\text{GAN}} \mathcal{L}_{\text{GAN}}$$

CogVideoX를 위해 설계된 3D VAE는 비디오 데이터를 공간적뿐 아니라 **시간적으로도 압축**하여 시퀀스 길이와 훈련 연산량을 크게 줄입니다. 이 접근법은 2D VAE를 사용하는 모델에서 자주 발생하는 **깜박임(flicker) 현상도 완화**합니다. 시간적 인과 합성곱과 컨텍스트 병렬성을 활용하여 대규모 비디오 데이터셋을 효율적으로 처리하면서 시간적 인과관계를 유지합니다.

---

### 3-2. Expert Transformer with Expert Adaptive LayerNorm

CogVideoX의 디노이징 백본은 **Expert Transformer**로, 텍스트와 비디오 두 모달리티의 융합을 담당합니다.

Expert Transformer 내부의 **Expert Adaptive LayerNorm**은 텍스트와 시각 모달리티를 더 잘 정렬하기 위해 설계되었으며, 융합 전에 각 모달리티를 독립적으로 처리함으로써 강건한 교차 모달 정렬을 보장하고 추가적인 파라미터 수를 최소화합니다.

Expert Adaptive LayerNorm 수식은 다음과 같습니다:

$$\text{AdaLN}^{\text{expert}}(\mathbf{h}_{\text{text}}, \mathbf{h}_{\text{video}}) = \gamma_m \cdot \text{LayerNorm}(\mathbf{h}_m) + \beta_m$$

여기서 $m \in \{\text{text}, \text{video}\}$이며, $\gamma_m$과 $\beta_m$은 각 모달리티에 특화된 스케일 및 시프트 파라미터입니다. 기존의 MMDiT 방식과 달리, Expert AdaLN은 각 모달리티를 독립적으로 정규화한 후 Joint Attention으로 융합합니다.

CogVideoX에서 비디오와 텍스트는 **3D Full Attention**을 통해 상호작용합니다. 비디오 입력 $X \in \mathbb{R}^{T \times H \times W \times C}$는 3D Causal VAE를 통해 잠재 공간 $z_v$로 압축되고, 텍스트 입력 $z_t$는 각각의 시각 및 텍스트 Expert Transformer로 인코딩됩니다.

Joint 3D Full Attention의 핵심 수식:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

여기서 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$는 비디오 토큰과 텍스트 토큰을 시퀀스 차원에서 연결(concatenate)하여 형성됩니다:

$$[\mathbf{z}_v \| \mathbf{z}_t] \rightarrow \text{Joint Attention} \rightarrow \text{분리하여 각 모달리티 업데이트}$$

---

### 3-3. 3D Rotary Position Embedding (3D-RoPE)

위치 인코딩으로 CogVideoX-2B는 3D 사인-코사인 절대 위치 임베딩을, **5B 및 1.5 변형에서는 3D Rotary Position Embeddings (3D-RoPE)**를 사용합니다. 3D-RoPE 방식은 공간 좌표(높이, 너비)와 시간 좌표(프레임 인덱스)를 독립적으로 임베딩하여, 모델이 **다양한 해상도와 영상 길이에 일반화**할 수 있도록 합니다.

$$\mathbf{q}' = \mathbf{q} \cdot \left(\cos(m\theta_{x,y,t}), \sin(m\theta_{x,y,t})\right)$$

여기서 $m$은 위치 인덱스이고, $\theta_{x}, \theta_{y}, \theta_{t}$는 각각 공간(x, y)과 시간(t) 차원의 주파수를 나타냅니다.

3D RoPE를 Sinusoidal 절대 위치 임베딩과 비교한 결과, RoPE의 손실 곡선이 절대 위치 임베딩보다 **훨씬 빠르게 수렴**하는 것으로 나타났습니다. 이는 LLM에서의 일반적인 선택과 일치합니다.

---

### 3-4. Progressive Training & Frame Pack (Multi-Resolution Training)

CogVideoX는 **Progressive(점진적) 학습**과 **혼합 길이 훈련(Frame Pack)**을 포함한 훈련 기법을 활용하여 시간적 일관성을 향상시키고 깜박임을 줄입니다.

학습 전략의 단계:
1. **해상도 점진적 학습:** 저해상도 → 고해상도 순으로 학습
2. **Frame Pack (혼합 길이 훈련):** 다양한 길이의 영상을 하나의 배치로 묶어 동시 학습
3. **Multi-resolution training:** 다양한 종횡비(aspect ratio) 영상 학습

Diffusion 학습의 기본 목적함수(Denoising Score Matching):

$$\mathcal{L}_{\text{DM}} = \mathbb{E}_{\mathbf{z}, t, \boldsymbol{\epsilon}}\left[\left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c})\right\|^2\right]$$

여기서 $\mathbf{z}_t$는 $t$번째 타임스텝에서 노이즈가 추가된 잠재 벡터, $\boldsymbol{\epsilon}$은 가우시안 노이즈, $\mathbf{c}$는 텍스트 조건입니다.

---

### 3-5. 데이터 처리 파이프라인

데이터 파이프라인에서 Panda70M 모델로 짧은 비디오 캡션을 생성하고, 프레임을 추출하여 밀집 이미지 캡션을 만든 뒤, GPT-4를 활용하여 최종 비디오 캡션으로 요약합니다. 이 과정을 가속화하기 위해 GPT-4 요약 결과로 Llama 2 모델을 파인튜닝했습니다.

학습 데이터셋은 약 **3,500만 개의 고품질 비디오 클립**으로 구성되며, 동적 콘텐츠와 설명적 텍스트 정보를 보장하기 위해 비디오 필터와 캡션 모델로 선별됩니다. 학습 세트는 LAION-5B 및 COYO-700M과 같은 대규모 데이터셋에서 추출한 **20억 개의 고득점 이미지**로 추가 증강됩니다.

---

## 4. 모델 구조 전체 요약

```
텍스트 프롬프트 (T5 인코더)
       ↓
   텍스트 임베딩 z_t
       ↓
┌──────────────────────────────────────────────┐
│         Expert Transformer Block              │
│  ┌────────────────┐   ┌───────────────────┐  │
│  │ Text Expert    │   │ Video Expert      │  │
│  │ AdaLN + FFN   │   │ AdaLN + FFN       │  │
│  └──────┬─────────┘   └────────┬──────────┘  │
│         └──── 3D Full Attention ─────────────┘│
│                 ↓                              │
│          3D-RoPE 위치 인코딩                   │
└──────────────────────────────────────────────┘
       ↓
   디노이징 잠재 z_v
       ↓
   3D Causal VAE 디코더
       ↓
   생성 비디오 (768×1360, 16fps, 10초)
```

시각적 정보와 텍스트 정보 간의 융합은 **패치화된 이미지/비디오 임베딩**과 **텍스트 임베딩**을 시퀀스 차원에서 연결(concatenate)함으로써 이루어지며, 두 임베딩 모두 모달리티별 백본(텍스트의 경우 T5 인코더)에 의해 인코딩됩니다.

---

## 5. 성능 향상

CogVideoX-2B의 성능은 VBench 벤치마크의 자동화 지표를 통해 평가되며, Human Action (96.6), Scene (55.35), Dynamic Degree (66.39), Multiple Objects (57.68), Appearance Style (24.37), Dynamic Quality (57.7), GPT4o-MT Score (3.09) 등에서 측정됩니다.

3D VAE는 표준 벤치마크에서 **PSNR 29.1**과 **프레임 깜박임 점수 85.5**를 달성합니다.

폐쇄형 모델 중 최고 수준인 Kling(2024.7)과 CogVideoX-5B를 비교했을 때, CogVideoX-5B가 **인간 평가에서 더 높은 점수**를 기록했습니다.

인간 평가에서 CogVideoX는 **감각적 품질(sensory quality), 지시 따르기(instruction following), 물리 시뮬레이션(physics simulation), 커버 품질(cover quality)** 등에서 다른 모델보다 높은 점수를 달성했습니다.

---

## 6. 한계점

논문에는 잠재적 한계나 주의사항에 대한 논의가 부족합니다. 예를 들어, 모델이 **희귀하거나 비일상적인 텍스트 프롬프트**를 어떻게 처리하는지, 또는 **고도로 복잡하거나 추상적인 콘텐츠**의 비디오 생성 시 성능에 대한 논의가 없습니다. 또한 3D VAE 및 Expert Transformer 구성 요소의 **연산 및 메모리 요구 사항**도 충분히 다루어지지 않아 실제 배포 시 고려가 필요합니다.

CogVideoX-2B는 주로 **영어 프롬프트**로 학습 및 평가되었으며, 다국어 지원은 외부 번역 모델에 의존합니다.

더 큰 압축 비율을 가진 VAE 탐색은 향후 과제로 남아 있으며, 공격적인 VAE 압축 비율은 모델 수렴에 어려움을 줄 수 있습니다.

---

## 7. 모델의 일반화 성능 향상 가능성

### 7-1. 3D-RoPE를 통한 해상도/길이 일반화

3D-RoPE는 공간 좌표(높이, 너비)와 시간 좌표(프레임 인덱스)를 독립적으로 임베딩하여 모델이 **다양한 해상도와 영상 길이에 일반화**할 수 있도록 합니다. 이는 특정 해상도에 고정되지 않고 훈련되지 않은 길이의 비디오에 대해서도 일반화할 수 있는 핵심 메커니즘입니다.

### 7-2. Frame Pack을 통한 다양한 길이 일반화

점진적 학습(progressive training)과 다중 해상도 프레임 패킹(multi-resolution frame packing)을 통해 CogVideoX는 **다양한 형태(diverse shapes)와 동적 움직임**을 가진 일관적이고 장시간의 비디오를 생성하는 데 탁월합니다.

### 7-3. 이미지-비디오 공동 학습을 통한 일반화

학습 세트는 LAION-5B 및 COYO-700M과 같은 대규모 데이터셋에서 추출한 **20억 개의 고득점 이미지**로 추가 증강됩니다. 이미지와 비디오를 함께 학습함으로써 모델은 정적 장면과 동적 장면 모두에서의 일반적인 시각 표현을 학습할 수 있습니다.

비디오 피처는 **Patchify 접근법**으로 인코딩되며, 이는 이미지와 비디오를 프레임 기반 잠재 코드 시퀀스로 재형식화하여 **이미지와 비디오 공동 학습**을 가능하게 합니다.

### 7-4. Expert AdaLN의 모달리티 일반화 기여

3D VAE와 Expert Transformer의 통합은 비디오 Diffusion 모델의 효율성과 효과성에 새로운 기준을 세웁니다. 향후 연구는 이러한 기반을 바탕으로 모델 아키텍처와 훈련 방식을 더욱 개선하고, 잠재적으로 **더 높은 해상도와 더 긴 영상 지속 시간**으로 이러한 기술을 확장할 수 있습니다.

### 7-5. 다운스트림 Task 일반화 (파인튜닝 측면)

cogvideox-factory는 CogVideoX를 위한 비용 효율적인 파인튜닝 프레임워크로, **단일 4090 GPU로 CogVideoX-5B의 파인튜닝**이 가능합니다.

CogVideoX 시리즈 모델은 현재 **텍스트-투-비디오 생성, 비디오 연속(continuation), 이미지-투-비디오 생성**이라는 세 가지 Task를 지원합니다.

---

## 8. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 출시 연도 | 아키텍처 | 최대 해상도 | 최대 길이 | 오픈소스 | 주요 특징 |
|---|---|---|---|---|---|---|
| **Make-A-Video** (Meta) | 2022 | Diffusion + Pseudo-3D | 768px | 2.5초 | ❌ | 이미지 사전학습 활용 |
| **VideoLDM** (Stability AI) | 2023 | Latent Diffusion | 512px | ~2초 | ✅ | 잠재 공간 확산 |
| **CogVideo** (THUDM) | 2022~2023 (ICLR'23) | Transformer | 480px | 4초 | ✅ | 최초 오픈소스 대형 T2V |
| **Sora** (OpenAI) | 2024.02 | DiT + Spacetime Patch | 1080p | 60초 | ❌ | 세계 시뮬레이션 수준 |
| **Lumiere** (Google) | 2024.01 | Space-Time U-Net | HD | ~5초 | ❌ | 시공간 확산 |
| **Kling** (Kuaishou) | 2024.06 | 비공개 | 1080p | 120초 | ❌ | 긴 비디오, 높은 품질 |
| **CogVideoX** (THUDM) | **2024.08** | **3D VAE + Expert DiT** | **768×1360** | **10초** | **✅** | **Expert AdaLN, 3D-RoPE** |
| **Hunyuan Video** (Tencent) | 2024.12 | Diffusion Transformer | HD | ~5초 | ✅ | 오픈소스 고품질 |
| **Wan 2.1** (Alibaba) | 2025.02 | Diffusion Transformer | HD | ~10초 | ✅ | LoRA 파인튜닝 지원 |

CogVideoX는 ICLR 2025 컨퍼런스 논문으로 채택되었으며, AI 커뮤니티에서 **가장 널리 채택된 오픈소스 비디오 생성 프레임워크 중 하나**가 되었습니다.

DiT(Diffusion Transformer, Peebles & Xie, 2023)의 도입으로 Sora와 CogVideoX와 같은 T2V 생성 모델이 DiT 아키텍처를 기반으로 개발되었으며, 특히 CogVideoX는 **3D-VAE를 활용하여 복잡한 시공간적 관계를 포착하는 모델의 능력을 향상**시킵니다.

---

## 9. 앞으로의 연구에 미치는 영향 및 고려 사항

### 9-1. 연구에 미치는 영향

**① DiT 기반 Video Generation의 표준화**
- CogVideoX는 비디오 생성에서 **3D VAE + DiT** 조합을 사실상의 표준(de facto standard)으로 정립했습니다. 이후 Hunyuan Video, Wan 등 주요 오픈소스 모델이 유사한 설계 철학을 따릅니다.

**② Expert Transformer 패러다임의 확산**
3D VAE와 Expert Transformer의 통합은 비디오 Diffusion 모델의 효율성과 효과성에 **새로운 기준**을 세웁니다. 이 패러다임은 멀티모달 생성 연구 전반에 영향을 줄 것으로 예상됩니다.

**③ 오픈소스 생태계 기여**
3D Causal VAE와 CogVideoX의 모델 가중치가 공개적으로 제공되어, 연구 커뮤니티의 후속 연구(파인튜닝, ControlNet, Interpolation 등)를 대폭 촉진했습니다.

**④ 영상 캡셔닝 모델의 중요성 부각**
다양한 텍스트 및 비디오 데이터 전처리 전략을 포함한 효과적인 파이프라인과 혁신적인 비디오 캡셔닝 모델이 생성 품질과 의미적 정렬을 크게 향상시킴을 보여줍니다.

---

### 9-2. 앞으로 연구 시 고려할 점

**① 더 긴 영상 및 더 높은 해상도로의 확장**
스케일링 법칙(scaling laws) 탐구와 더 긴 고해상도 비디오 생성에 집중함으로써 텍스트-투-비디오 생성 능력의 경계를 확장할 수 있습니다.

**② 물리 시뮬레이션 및 복잡한 동작 이해**
CogVideoX를 더 강건하고 효율적으로 만들기 위한 추가 연구가 필요하며, 잠재적으로 **대안적인 아키텍처나 훈련 기법**을 탐색하는 것이 도움이 될 수 있습니다. 특히 복잡한 물리 상호작용 및 다중 객체 추론에서의 일반화 성능 향상이 주요 과제입니다.

**③ Frame Pack의 타 도메인 적용 가능성 탐구**
혼합 길이 학습 방식(Frame Pack)이 오디오, 언어 생성 등 다른 시퀀스 모델링 과제에도 일반화될 수 있는지 탐구가 필요합니다.

**④ 다국어 및 다문화 일반화**
현재 모델이 주로 영어 프롬프트 기반으로 학습된 한계를 극복하기 위해, 다국어 학습 및 문화적 다양성을 반영한 데이터셋 구축이 필요합니다.

**⑤ 메모리 효율화와 경량화**
PytorchAO와 Optimum-quanto를 이용해 텍스트 인코더, Transformer, VAE 모듈을 양자화하여 CogVideoX의 메모리 요구를 줄일 수 있으며, 무료 T4 Colab이나 더 작은 VRAM의 GPU에서도 실행이 가능하게 되었습니다. 그러나 실시간 응용을 위한 추론 속도 최적화 연구는 여전히 중요한 과제입니다.

**⑥ 제어 가능성(Controllability) 연구**
ControlNet, LoRA 등을 활용한 미세 제어 연구가 활발히 이루어지고 있으며, 이를 통한 캐릭터 일관성, 카메라 무브먼트 제어 등의 일반화 성능 향상이 중요한 방향입니다.

---

## 📚 참고 자료 및 출처

1. **arXiv 논문 (v1~v3):** Yang, Z. et al., "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer," arXiv:2408.06072. https://arxiv.org/abs/2408.06072
2. **ICLR 2025 Conference Paper:** https://proceedings.iclr.cc/paper_files/paper/2025/hash/ce31378e9f41d8907e97dab172b6c559-Abstract-Conference.html
3. **OpenReview (ICLR 2025):** https://openreview.net/forum?id=LQzN6TRFg9
4. **Hugging Face Paper Page:** https://huggingface.co/papers/2408.06072
5. **GitHub (공식 코드):** https://github.com/THUDM/CogVideo
6. **Hugging Face Model (CogVideoX-5B):** https://huggingface.co/zai-org/CogVideoX-5b
7. **Semantic Scholar:** https://www.semanticscholar.org/paper/CogVideoX:-Text-to-Video-Diffusion-Models-with-An-Yang-Teng/7b248d78573ccf0dca6aa2cec2743d3eccaa9d1a
8. **EmergentMind 분석:** https://www.emergentmind.com/papers/2408.06072
9. **Open Laboratory (CogVideoX-2B):** https://openlaboratory.ai/models/cog-video-x-2b
10. **AI Wiki (CogVideoX):** https://aiwiki.ai/wiki/cogvideo
11. **CustomVideoX (관련 후속 연구):** arXiv:2502.06527, https://arxiv.org/html/2502.06527v1
12. **From Sora What We Can See: A Survey (T2V 서베이):** arXiv:2405.10674, https://arxiv.org/html/2405.10674v1
13. **AI Video Generation Model Evolution (관련 타임라인):** https://gaga.art/blog/ai-video-generation-model/
