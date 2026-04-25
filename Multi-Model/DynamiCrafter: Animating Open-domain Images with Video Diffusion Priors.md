
# DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors

> **논문 정보**
> - **제목:** DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors
> - **저자:** Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Xintao Wang, Tien-Tsin Wong, Ying Shan
> - **소속:** The Chinese University of Hong Kong, Tencent AI Lab, Peking University
> - **발표:** ECCV 2024 (Oral), arXiv:2310.12190
> - **GitHub:** https://github.com/Doubiiu/DynamiCrafter

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

전통적인 이미지 애니메이션 기법들은 주로 자연 장면의 확률적 동역학(예: 구름, 유체)이나 도메인 특화 모션(예: 인간 머리카락, 신체 모션)에 집중하여, 보다 일반적인 시각적 콘텐츠에 대한 적용 가능성이 제한되어 있다. 이를 극복하기 위해 논문은 오픈 도메인 이미지에 대한 동적 콘텐츠 합성을 탐구하여 이를 애니메이션 비디오로 변환하는 방법을 제안한다.

핵심 아이디어는 이미지를 생성 프로세스에 가이던스로 통합함으로써 텍스트-투-비디오(T2V) 확산 모델의 **모션 사전(motion prior)**을 활용하는 것이다.

### 1.2 주요 기여

| 기여 | 설명 |
|------|------|
| **이중 스트림 이미지 주입** | 텍스트 정렬 컨텍스트 표현 + 시각적 세부 가이던스의 상보적 결합 |
| **쿼리 트랜스포머 기반 이미지 인코딩** | CLIP 이미지 특징을 T2V 모델과 호환 가능한 공간으로 변환 |
| **오픈 도메인 일반화** | 도메인 제한 없이 다양한 스타일/콘텐츠의 정지 이미지 애니메이션화 |
| **다양한 응용** | 스토리텔링 비디오 생성, 루핑 비디오 생성, 생성적 프레임 보간 지원 |

광범위한 실험은 기존 경쟁 방법들에 비해 현저한 우월성을 입증하며, Gen-2, PikaLabs와 같은 최신 상업용 데모와도 비교 가능한 성능을 보인다. 더불어, 서로 다른 시각적 주입 스트림의 역할, 텍스트 프롬프트의 유용성 및 동역학 제어 가능성에 대한 논의와 분석을 제공하여 후속 연구를 이끈다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 접근법들은 이미지 애니메이션에 적합하지 않은데, 이는 덜 포괄적인 이미지 주입 메커니즘으로 인해 급격한 시간적 변화 또는 입력 이미지와의 낮은 시각적 일치도를 초래하기 때문이다. 이에 DynamiCrafter는 시각적 세부 가이던스와 텍스트 정렬 컨텍스트 표현으로 구성된 **이중 스트림 주입 방식(dual-stream injection approach)**을 제안한다.

구체적으로 해결하려는 핵심 문제:
1. **도메인 특화 한계:** 기존 방법들은 특정 도메인(인체, 풍경 등)에 한정
2. **시각적 세부 정보 손실:** 생성된 비디오에서 입력 이미지의 세부 정보가 유실
3. **부자연스러운 모션:** 논리적이지 않거나 비현실적인 동작 생성

### 2.2 제안 방법 및 수식

#### 📌 문제 수식화

정지 이미지가 주어졌을 때, 이미지로부터 모든 시각적 콘텐츠를 계승하고 암묵적으로 제안되는 자연스러운 동역학을 보이는 짧은 비디오 클립을 생성하는 것이 목표이다. 정지 이미지는 결과 프레임 시퀀스의 임의의 위치에 나타날 수 있다. 기술적으로 이 과제는 높은 수준의 시각적 일치도를 요구하는 **이미지 조건부 비디오 생성**의 특수한 형태로 공식화된다.

#### 📌 Diffusion Model 기반 학습 목적 함수

일반적인 Latent Diffusion 기반 비디오 생성 목적 함수:

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, \mathbf{c}_{\text{txt}}, \mathbf{c}_{\text{img}}, t, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta\left(\mathbf{x}_t, t, \mathbf{c}_{\text{txt}}, \mathbf{c}_{\text{img}}\right) \right\|^2 \right]$$

여기서:
- $\mathbf{x}_0$: 원본 비디오 잠재 표현(latent)
- $\mathbf{x}_t$: 시각 $t$에서 노이즈가 추가된 잠재 표현
- $\mathbf{c}_{\text{txt}}$: 텍스트 조건
- $\mathbf{c}_{\text{img}}$: 이미지 조건 (DynamiCrafter의 핵심 추가 조건)
- $\epsilon_\theta$: 학습 가능한 디노이징 U-Net

#### 📌 이중 스트림 이미지 주입 (Dual-Stream Image Injection)

이 문제를 해결하기 위해 텍스트 정렬 컨텍스트 표현(text-aligned context representation)과 시각적 세부 가이던스(visual detail guidance)로 구성된 **이중 스트림 이미지 주입 패러다임**을 제안한다. 이를 DynamiCrafter라 부른다. 주어진 이미지에 대해 먼저 특별히 설계된 컨텍스트 학습 네트워크를 통해 텍스트 정렬 리치 컨텍스트 표현 공간으로 투영한다. 구체적으로, 텍스트 정렬 이미지 특징을 추출하기 위한 사전 학습된 **CLIP 이미지 인코더**와 확산 모델에 대한 적응을 촉진하기 위한 **학습 가능한 쿼리 트랜스포머**로 구성된다.

**스트림 1: 텍스트 정렬 컨텍스트 표현**

$$\mathbf{f}_{\text{ctx}} = \text{QueryTransformer}\left(\text{CLIP}_{\text{img}}(\mathbf{I})\right)$$

- CLIP 이미지 인코더로 이미지 특징 추출
- 쿼리 트랜스포머(Q-Former)로 T2V 모델의 텍스트 임베딩 공간과 정렬
- 프레임별 컨텍스트 표현을 디노이징 U-Net의 **공간적 이중-어텐션 트랜스포머(spatial dual-attn transformer)**에 적용

수식화 (Dual Cross-Attention):

$$\mathbf{h}' = \text{Attention}\left(\mathbf{Q}(\mathbf{h}),\ \mathbf{K}(\mathbf{c}_{\text{txt}}),\ \mathbf{V}(\mathbf{c}_{\text{txt}})\right) + \text{Attention}\left(\mathbf{Q}(\mathbf{h}),\ \mathbf{K}(\mathbf{f}_{\text{ctx}}),\ \mathbf{V}(\mathbf{f}_{\text{ctx}})\right)$$

여기서 $\mathbf{h}$는 U-Net 중간 특징이고, 텍스트와 이미지 컨텍스트를 동시에 참조한다.

**스트림 2: 시각적 세부 가이던스**

더 정밀한 이미지 정보를 보완하기 위해, 전체 이미지를 초기 노이즈와 연결(concatenate)하여 확산 모델에 추가로 입력한다.

$$\tilde{\mathbf{x}}_t = \text{Concat}\left[\mathbf{x}_t,\ \mathbf{z}_{\text{img}}\right] \in \mathbb{R}^{T \times 2C \times H \times W}$$

- $\mathbf{z}_{\text{img}}$: VAE로 인코딩된 이미지 잠재 표현 (각 프레임에 동일하게 복제)
- 조건 이미지 잠재를 채널 차원으로 노이즈 잠재와 연결하여 디노이징 U-Net의 입력 채널을 추가하며, 새로 추가된 입력 채널에 해당하는 가중치는 0으로 초기화된다.

**Classifier-Free Guidance (CFG) 조건부 생성:**

$$\tilde{\epsilon}_\theta = \epsilon_\theta(\mathbf{x}_t, \varnothing, \varnothing) + s_{\text{txt}}\left[\epsilon_\theta(\mathbf{x}_t, \mathbf{c}_{\text{txt}}, \varnothing) - \epsilon_\theta(\mathbf{x}_t, \varnothing, \varnothing)\right] + s_{\text{img}}\left[\epsilon_\theta(\mathbf{x}_t, \mathbf{c}_{\text{txt}}, \mathbf{c}_{\text{img}}) - \epsilon_\theta(\mathbf{x}_t, \mathbf{c}_{\text{txt}}, \varnothing)\right]$$

여기서 $s_{\text{txt}}$, $s_{\text{img}}$는 각각 텍스트와 이미지 조건의 영향도를 제어하는 가이던스 스케일이다.

### 2.3 모델 구조

DynamiCrafter 프레임워크는 세부 정보 가이던스와 컨텍스트 제어에 중요한 역할을 하는 두 가지 상보적 스트림을 통해 조건 이미지를 통합한다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DynamiCrafter Architecture                    │
│                                                                  │
│  Input Image ─┬─► CLIP Image Encoder ─► Query Transformer ─┐   │
│               │        (Stream 1: Context Representation)   │   │
│               │                                             ▼   │
│               │                               Spatial Dual-Attn │
│               │                               Transformer       │
│               │                                  in U-Net    ▲  │
│               │                                              │  │
│               └─► VAE Encoder ─► Concat with Noise ─────────┘  │
│                       (Stream 2: Visual Detail Guidance)        │
│                                                                  │
│  Text Prompt ──► CLIP Text Encoder ─► Cross-Attention in U-Net  │
│                                                                  │
│  T2V Pre-trained Denoising U-Net (3D U-Net with Temporal Attn) │
│                                                                  │
│  VAE Decoder ──► Generated Video Frames                         │
└─────────────────────────────────────────────────────────────────┘
```

**핵심 구성 요소:**

| 컴포넌트 | 역할 |
|--------|------|
| **CLIP Image Encoder** (Frozen) | 텍스트 정렬 이미지 특징 추출 |
| **Query Transformer** (Learnable) | 이미지 특징을 확산 모델 호환 공간으로 변환 |
| **3D Denoising U-Net** (Fine-tuned) | 시공간 비디오 생성 |
| **Temporal Attention** | 프레임 간 시간적 일관성 유지 |
| **Spatial Dual-Attn Transformer** | 텍스트 + 이미지 컨텍스트를 동시에 참조 |
| **FPS Embedding Layer** | 프레임 속도 제어 |

모델은 상대적으로 자원 친화적인 방식으로 8개의 NVIDIA V100 GPU만을 사용하여 T2V 모델을 파인튜닝한다. 주로 실제 세계 비디오로 구성된 WebVid-10M을 사용하여 파인튜닝이 진행된다.

**2단계 학습 전략:**

이 방법은 이미지 조건부 컨텍스트를 T2V 모델에 정렬하고 모션 합성 중 미세한 입력 세부 정보를 보존하기 위한 **3단계 학습 파이프라인**을 사용한다.

- **Stage 1:** 쿼리 트랜스포머 + Spatial Attn만 학습 (Temporal Attn 고정)
- **Stage 2:** 전체 모델 파인튜닝 (시각적 세부 가이던스 스트림 포함)

### 2.4 성능 향상

정량적 및 정성적 평가에서 오픈 도메인 베이스라인에 비해 모션 품질, 시간적 일관성, 입력 이미지와의 시각적 일치도에서 **상당한 개선**을 보여주며, 상업용 데모와도 경쟁적인 성능을 발휘한다.

DynamiCrafter의 1024×576 버전은 VBench의 I2V 벤치마크 리스트에서 **1위**를 기록하였다.

현재 DynamiCrafter는 576×1024 해상도에서 최대 16 프레임의 비디오 생성을 지원한다.

### 2.5 한계점

CLIP 인코더가 언어와 시각적 특징을 정렬하도록 설계되어 있어 입력 정보를 완전히 보존하는 데 한계가 있으며, 생성된 콘텐츠에서 일부 불일치가 나타날 수 있다.

생성되는 비디오는 비교적 짧으며(2초, FPS=8), 모델은 판독 가능한 텍스트를 렌더링할 수 없고, 얼굴과 사람이 제대로 생성되지 않을 수 있으며, 모델의 오토인코딩 부분은 손실이 있어 약간의 플리커링 아티팩트가 발생한다.

이미지 콘텐츠 이해 측면에서의 도전적인 케이스가 존재하며, 데이터셋에 정밀한 모션 설명이 부족하여 특정 모션을 생성하는 데 한계가 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 오픈 도메인 일반화의 핵심 요소

이 방법은 풍경, 인간, 동물, 차량, 조각상 등 다양한 콘텐츠와 실사, AI 생성, 회화, 점토, 애니메이션, 등각 일러스트레이션 등 다양한 스타일을 가진 정지 이미지에 대해 평가된다.

이는 DynamiCrafter가 도메인-특화 모션 학습에서 벗어나 **오픈 도메인 일반화**를 달성하는 핵심 근거이다.

### 3.2 일반화 성능 향상 메커니즘

| 요소 | 일반화 기여 |
|------|-----------|
| **T2V 사전 지식 활용** | 대규모 비디오 데이터로 사전 학습된 T2V 모델의 광범위한 모션 패턴 내재화 |
| **CLIP 이미지-텍스트 정렬** | 이미지를 언어와 호환 가능한 공간으로 투영하여 다양한 시각 도메인 처리 |
| **Dual-stream 주입** | 고수준 의미 이해와 저수준 시각 세부 정보를 동시에 처리 |
| **텍스트 프롬프트 제어** | 텍스트를 통한 추가적인 동적 제어로 다양한 시나리오 대응 |

학습 중에 비디오 프레임을 무작위로 선택하여 이중 스트림 이미지 주입 메커니즘을 통해 디노이징 프로세스의 이미지 조건으로 사용하므로, 추론 시에는 입력 정지 이미지에서 조건을 받아 노이즈로부터 애니메이션 클립을 생성할 수 있다.

### 3.3 WebVid-10M-motion 데이터셋 기여

WebVid-10M-motion 어노테이션에는 원래 어노테이션 외에 `dynamic_confidence`, `dynamic_wording`, `dynamic_source_category`의 세 가지 모션 관련 어노테이션이 추가된다.

이러한 **모션 메타데이터**는 모델이 다양한 종류의 모션 패턴을 학습하여 일반화 능력을 강화하는 데 직접적으로 기여한다.

### 3.4 일반화 한계 및 개선 방향

현재의 일반화 한계:
- CLIP 인코더가 입력 정보를 완전히 보존하는 능력에 한계가 있어, 생성된 콘텐츠가 일부 불일치를 보일 수 있다.
- 일부 시각적 세부 정보가 결과 비디오에서 여전히 보존되는 데 어려움을 겪는다.

**일반화 향상 가능성:**

1. **더 강력한 이미지 인코더 사용** (CLIP → DINOv2, SigLIP 등)
2. **대규모 고품질 데이터셋으로 파인튜닝** (WebVid-10M 외 다양한 소스)
3. **Diffusion Transformer(DiT) 기반 아키텍처 전환** (Sora 계열)
4. **Zero-shot 도메인 전이를 위한 어댑터 기법** 도입

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 Image-to-Video (I2V) 계열 방법 비교

| 방법 | 발표 연도 | 핵심 접근 | 한계 | DynamiCrafter 대비 |
|------|---------|---------|------|------------------|
| **LVDM** (He et al.) | 2022 | 잠재 비디오 확산 모델, 임의 길이 | 이미지 조건 부족 | DC: 이미지 가이던스 명시적 통합 |
| **ModelScope** | 2022 | T2V, 3D U-Net 기반 | 텍스트만 입력 가능 | DC: 이미지+텍스트 이중 조건 |
| **AnimateDiff** | 2023 | SD에 모션 모듈 플러그인 추가 | 도메인 범용 I2V 어려움 | DC: T2V prior 직접 활용 |
| **SEINE** | 2023 (ICLR 2024) | 숏-투-롱 비디오, 장면 전환 | 장면 전환 특화 | DC: 단일 이미지 애니메이션에 특화 |
| **Stable Video Diffusion (SVD)** | 2023 | 대규모 데이터 기반 I2V | 텍스트 제어 약함 | DC: 텍스트 모션 제어 가능 |
| **Gen-2 (Runway)** | 2023 | 상업용 I2V | 비공개 | DC: 오픈소스로 비교 가능 성능 |
| **Emu Video** | 2023 | 명시적 이미지 조건 T2V | 메타(Meta) 내부 모델 | DC: 공개 재현 가능 방법 제시 |
| **DynamiCrafter** | 2023 (ECCV 2024) | Dual-stream, T2V prior 활용 | 짧은 비디오, 얼굴 생성 약함 | — |

T2V 모델들은 비디오 프레임 간의 관계를 포착하기 위해 시간적 컨볼루션 또는 시간적 자기-어텐션과 같은 시간적 모듈을 T2I 확산 모델에 통합하며, AnimateDiff와 같은 대표적인 접근법은 공간 모델링과 시간 모델링을 분리하여 플러그-앤-플레이 방식으로 비디오 데이터에서 모션 사전 지식을 학습한다.

사전 학습된 이미지-텍스트 확산 모델에 시간적 레이어를 삽입하여 이를 "팽창(inflate)"하는 방법은 텍스트-이미지 쌍의 사전 지식이 새 모델에 의해 계승되어 텍스트-비디오 쌍 데이터 요구 사항을 완화하는 데 도움을 준다.

### 4.2 DynamiCrafter vs. Stable Video Diffusion (SVD) 심층 비교

SVD 모델은 3단계 학습을 거쳤으며, 이미지 모델 학습, 이미지 모델을 비디오 모델로 확장하여 대규모 비디오 데이터셋으로 사전 학습, 그리고 고품질 소규모 데이터셋으로 파인튜닝하는 단계로 구성된다.

- **SVD**: 대규모 데이터 기반 범용 I2V, 텍스트 제어 약함
- **DynamiCrafter**: 텍스트+이미지 이중 조건, 명시적 모션 제어, 오픈 도메인 다양성 강조

### 4.3 DynamiCrafter의 후속 연구에 대한 영향

이미지 애니메이션 외에도 DynamiCrafter는 스토리텔링 비디오 생성, 루핑 비디오 생성, 생성적 프레임 보간과 같은 응용을 쉽게 지원할 수 있도록 적응될 수 있다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구적 영향

**① 오픈 도메인 이미지 애니메이션의 패러다임 전환**

DynamiCrafter는 특정 도메인에 제한된 기존 접근법에서 벗어나 **T2V 사전 지식을 활용한 범용 I2V 생성**이라는 새로운 연구 방향을 제시하였다. 이는 이후 SVD, I2V-Adapter, AnimateAnyone 등 다수의 후속 연구에 직접적인 영향을 미쳤다.

**② 이중 스트림 조건 주입 프레임워크**

DynamiCrafter는 사전 학습된 비디오 확산 사전 지식을 활용하고 텍스트 정렬 컨텍스트 표현과 직접적인 시각적 세부 가이던스를 결합하는 이중 스트림 이미지 주입 메커니즘을 도입하여 모션 일관성과 세부 정보 보존 애니메이션을 생성한다. 이 아이디어는 멀티모달 조건부 생성 연구 전반에 영향을 준다.

**③ 실용적 응용 확장성**

텍스트 기반 모션 제어 탐구와 스토리텔링, 루핑, 프레임 보간과 같은 실용적 응용을 시연함으로써 프레임워크의 다양성과 잠재적 한계를 강조한다.

### 5.2 앞으로 연구 시 고려할 점

#### ✅ 기술적 고려 사항

1. **더 강력한 시각적 인코더 필요성**
   - CLIP의 언어-비전 정렬 공간이 갖는 한계(세부 시각 정보 손실)를 극복하기 위해 DINOv2, SigLIP, EVA-CLIP 등의 강력한 인코더 활용 연구 필요

2. **장기 비디오 일관성 문제**
   - 비디오 확산 트랜스포머는 학습 길이를 넘어서 일반화하는 데 어려움을 겪으며, 이를 비디오 길이 외삽(video length extrapolation) 문제라 하고, 주기적 콘텐츠 반복 및 보편적 품질 저하의 두 가지 실패 모드가 존재한다.

3. **얼굴 및 사람 생성 품질**
   - 모델은 판독 가능한 텍스트를 렌더링할 수 없으며, 얼굴과 사람이 제대로 생성되지 않을 수 있다. 이 문제를 해결하기 위한 도메인-특화 보정 메커니즘 연구가 필요하다.

4. **모션 제어 정밀도 향상**
   - 현재 텍스트로 모션을 제어하지만, 광학 흐름(optical flow), 포즈 시퀀스, 깊이 맵 등을 추가 조건으로 활용하는 연구 방향

5. **DiT 기반 아키텍처 전환**
   - Sora는 비디오 및 이미지 잠재 코드의 시공간 패치에서 동작하는 DiT(Diffusion Transformer) 아키텍처를 활용하며, 시각적 입력이 트랜스포머 입력 토큰으로 역할하는 시공간 패치 시퀀스로 표현된다. 이 방향으로의 발전이 일반화 성능 향상에 기여할 것이다.

#### ✅ 데이터 및 학습 고려 사항

6. **고품질 모션-어노테이션 데이터 확보**
   - 정밀한 모션 설명이 포함된 대규모 데이터셋 구축이 일반화 향상의 핵심
   - WebVid-10M-motion의 모션 메타데이터 방향을 더 발전시키는 연구

7. **효율적 파인튜닝 기법**
   - LoRA, Adapter 등 파라미터 효율적 방법을 통해 다양한 도메인에의 빠른 적응 연구

#### ✅ 평가 및 사회적 고려 사항

8. **더 포괄적인 평가 메트릭**
   - FVD, SSIM 등 기존 메트릭 외에 모션 자연스러움, 시각적 일관성 등을 종합적으로 측정하는 새로운 벤치마크 필요

9. **윤리적 사용 및 오용 방지**
   - 딥페이크, 허위 정보 생성 등에 악용 가능성 → 워터마킹, 생성 이미지 탐지 기술 병행 연구 필요

---

## 📚 참고 자료 (출처)

| # | 자료 | URL / 출처 |
|---|------|------------|
| 1 | **DynamiCrafter 논문 (arXiv)** | https://arxiv.org/abs/2310.12190 |
| 2 | **DynamiCrafter ECCV 2024 (Springer)** | https://doi.org/10.1007/978-3-031-72952-2_23 |
| 3 | **DynamiCrafter GitHub** | https://github.com/Doubiiu/DynamiCrafter |
| 4 | **DynamiCrafter 프로젝트 페이지** | https://doubiiu.github.io/projects/DynamiCrafter/ |
| 5 | **ar5iv 논문 전문 (HTML)** | https://ar5iv.labs.arxiv.org/html/2310.12190 |
| 6 | **Hugging Face 논문 페이지** | https://huggingface.co/papers/2310.12190 |
| 7 | **DynamiCrafter HuggingFace 모델 (1024)** | https://huggingface.co/Doubiiu/DynamiCrafter_1024 |
| 8 | **Unite.AI 심층 분석** | https://www.unite.ai/dynamicrafter-animating-open-domain-images-with-video-diffusion-priors/ |
| 9 | **ScienceStack 분석** | https://www.sciencestack.ai/paper/2310.12190 |
| 10 | **OpenVINO 구현 문서** | https://docs.openvino.ai/2024/notebooks/dynamicrafter-animating-images-with-output.html |
| 11 | **ECCV 2024 Oral 발표 페이지** | https://eccv.ecva.net/virtual/2024/oral/1246 |
| 12 | **I2V-Adapter 논문 (arXiv)** | https://arxiv.org/html/2312.16693v4 |
| 13 | **Lilian Weng - Diffusion Models for Video Generation** | https://lilianweng.github.io/posts/2024-04-12-diffusion-video/ |
| 14 | **Stable Video Diffusion (SVD) 분석** | https://stable-diffusion-art.com/stable-video-diffusion-img2vid/ |
| 15 | **Open-Source AI Video Models (2026 비교)** | https://aifreeforever.com/blog/open-source-ai-video-models-free-tools-to-make-videos |
| 16 | **ResearchGate DynamiCrafter 페이지** | https://www.researchgate.net/publication/386217794 |
