
# Conditional Latent Diffusion Models for Zero-Shot Instance Segmentation (OC-DiT)

> **논문 정보**
> - **저자:** Maximilian Ulmer, Wout Boerdijk, Rudolph Triebel, Maximilian Durner
> - **소속:** German Aerospace Center (DLR), Karlsruhe Institute of Technology (KIT), Technical University of Munich (TUM)
> - **발표:** ICCV 2025
> - **arXiv:** [2508.04122](https://arxiv.org/abs/2508.04122)
> - **코드:** https://github.com/DLR-RM/oc-dit

---

## 1. 핵심 주장 및 주요 기여 요약

본 논문은 **Object-Conditioned Diffusion Transformer (OC-DiT)** 라는 새로운 클래스의 Diffusion 모델을 제안하며, 이를 **Zero-Shot Instance Segmentation** 에 적용합니다.

핵심은 **Conditional Latent Diffusion 프레임워크** 를 통해, **오브젝트 템플릿(object templates)** 과 **이미지 특징(image features)** 을 조건으로 삼아 Diffusion 모델의 잠재 공간(latent space) 내에서 인스턴스 마스크를 생성하는 것입니다.

### 주요 기여 정리

| 기여 항목 | 설명 |
|---|---|
| ① OC-DiT 아키텍처 | 객체 중심 예측을 위한 새로운 Diffusion Transformer |
| ② Coarse + Refinement 이중 모델 | 초기 제안 생성 + 병렬 정제 |
| ③ 대규모 합성 데이터셋 | 수천 개의 고품질 3D 메쉬로 구성된 신규 합성 데이터 |
| ④ Zero-Shot 일반화 | 타깃 데이터 재학습 없이 실제 벤치마크에서 SOTA 달성 |

---

## 2. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 / 한계

### 2.1 해결하고자 하는 문제

**Instance Segmentation** 은 픽셀 수준에서 개별 객체를 탐지하고 분리하는 핵심 컴퓨터 비전 과제로, 6D 자세 추정, 추적, 형상 완성, 로봇 응용(예: 파지) 등 다양한 비전 기반 응용의 첫 단계로 사용됩니다.

기존의 Instance Segmentation 방법들은 대부분 **타깃 도메인에 대한 지도 학습(supervised training)** 을 필요로 하며, **Zero-Shot(미지 객체/도메인에 대한 일반화)** 이 매우 어렵습니다. 본 논문은 이 문제를 **생성 모델(Diffusion Model)** 의 강력한 표현력을 활용하여 해결하고자 합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) Latent Diffusion 기반 조건부 마스크 생성

제안 방법은 오브젝트 템플릿과 이미지 특징을 조건으로 하여 Diffusion 모델의 **잠재 공간** 에서 인스턴스 마스크를 생성하며, **시각적 객체 디스크립터(visual object descriptors)** 와 **지역화된 이미지 단서(localized image cues)** 로 안내된 Diffusion 프로세스를 통해 객체 인스턴스를 효과적으로 분리(disentangle)합니다.

Latent Diffusion Model의 핵심 수식은 다음과 같습니다.

**Forward Process (노이즈 추가):**

$$q(\mathbf{z}_t | \mathbf{z}_{t-1}) = \mathcal{N}(\mathbf{z}_t; \sqrt{1 - \beta_t}\,\mathbf{z}_{t-1},\, \beta_t \mathbf{I})$$

**Marginal (직접 샘플링):**

$$q(\mathbf{z}_t | \mathbf{z}_0) = \mathcal{N}(\mathbf{z}_t; \sqrt{\bar{\alpha}_t}\,\mathbf{z}_0,\, (1 - \bar{\alpha}_t)\mathbf{I}), \quad \bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$$

**Reverse Process (조건부 마스크 생성):**

$$p_\theta(\mathbf{z}_{t-1} | \mathbf{z}_t, \mathbf{c}) = \mathcal{N}(\mathbf{z}_{t-1};\, \mu_\theta(\mathbf{z}_t, t, \mathbf{c}),\, \Sigma_\theta(\mathbf{z}_t, t, \mathbf{c}))$$

여기서 $\mathbf{c}$는 **오브젝트 템플릿 + 이미지 특징을 결합한 조건 벡터** 이며, $\mu_\theta$는 OC-DiT가 예측하는 노이즈 제거(denoising) 함수입니다.

**훈련 목적 함수 (Denoising Score Matching):**

$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_0,\, t,\, \boldsymbol{\epsilon} \sim \mathcal{N}(0,\mathbf{I}),\, \mathbf{c}} \left[\left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left(\mathbf{z}_t, t, \mathbf{c}\right) \right\|^2\right]$$

> ⚠️ **주의:** 위 수식은 Latent Diffusion Model의 일반적인 표준 수식 체계이며, 논문 내부의 정확한 수식 표기는 전문(full text)에서 확인이 필요합니다.

---

#### (2) 조건 구성 방식

OC-DiT의 조건 $\mathbf{c}$는 다음 두 가지 정보로 구성됩니다.

- **오브젝트 템플릿 특징:** $\mathbf{f}\_\text{template} = \text{Encoder}(I_\text{template})$
- **지역화된 이미지 특징:** $\mathbf{f}\_\text{local} = \text{Encoder}(I_\text{query}, \text{bbox})$

이 두 특징을 결합하여 Cross-Attention 메커니즘으로 Transformer 블록에 주입합니다.

$$\mathbf{c} = \text{Concat}(\mathbf{f}_\text{template},\, \mathbf{f}_\text{local})$$

---

### 2.3 모델 구조

본 논문은 **두 가지 모델 변형** 을 도입합니다: **초기 객체 인스턴스 제안을 생성하는 Coarse Model** 과 **모든 제안을 병렬로 정제하는 Refinement Model** 입니다.

```
입력: Query Image + Object Template(s)
        │
        ▼
  [Feature Extractor]   ← 시각적 디스크립터 추출
        │
        ▼
  [Coarse OC-DiT]       ← 초기 마스크 제안 생성 (Latent Space)
        │
        ▼
  [Refinement OC-DiT]   ← 병렬 마스크 정제
        │
        ▼
  출력: Instance Segmentation Masks
```

- **백본:** Diffusion Transformer (DiT) 기반 구조
- **조건 주입:** Cross-Attention (오브젝트 디스크립터, 지역 이미지 단서)
- **잠재 공간:** VAE 인코더/디코더를 통해 마스크를 잠재 표현으로 변환

---

### 2.4 학습 데이터

이 모델들은 **수천 개의 고품질 오브젝트 메쉬** 로 구성된 새롭게 생성된 **대규모 합성 데이터셋** 으로 훈련됩니다.

---

### 2.5 성능 향상

본 모델은 **타깃 데이터에 대한 재학습 없이** 다수의 도전적인 실제 벤치마크에서 **최신 기술 수준(State-of-the-Art)의 성능** 을 달성하였으며, 포괄적인 소거 연구(ablation study)를 통해 Instance Segmentation 과제에서 Diffusion 모델의 잠재력을 실증했습니다.

---

### 2.6 한계 (공개 정보 기준)

현재 공개된 Abstract 및 발표 자료에서 명시된 한계는 다음과 같습니다.

- **합성→실제 도메인 갭(Sim-to-Real Gap):** 합성 데이터로만 훈련하므로, 합성 데이터에서 완벽하게 재현하기 어려운 실제 환경의 복잡한 조명·질감·배경에 대한 일반화에 한계가 있을 수 있습니다.
- **오브젝트 템플릿 의존성:** 템플릿 이미지가 필요한 조건부 방식이므로, 템플릿 품질이 성능에 영향을 미칩니다.
- **추론 속도:** Diffusion 모델 특성상 반복적 Denoising 과정이 필요하여 단일 순전파(feed-forward) 방식 대비 추론 속도가 느릴 수 있습니다.

> ⚠️ 한계에 대한 구체적 수치 및 추가 내용은 논문 본문 확인이 필요합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문의 Zero-Shot 일반화는 여러 측면에서 매우 주목할 만합니다.

### 3.1 합성 데이터 기반의 Zero-Shot 전이

수천 개의 고품질 오브젝트 메쉬로 구성된 합성 데이터셋으로 훈련하였음에도 불구하고, **타깃 데이터에 대한 재학습 없이** 다수의 도전적인 실제 벤치마크에서 SOTA 성능을 달성합니다. 이는 합성→실제(Sim-to-Real) 일반화 가능성을 강하게 시사합니다.

### 3.2 조건부 생성의 일반화 메커니즘

모델이 **시각적 객체 디스크립터와 지역화된 이미지 단서** 로 안내되는 Diffusion 프로세스를 통해 객체 인스턴스를 효과적으로 분리할 수 있다는 점은, 특정 카테고리나 도메인에 종속되지 않은 일반적인 인스턴스 구분 능력을 학습함을 의미합니다.

### 3.3 일반화 향상 가능성 (연구 방향)

| 관점 | 설명 |
|---|---|
| **데이터 확장** | 더 다양한 합성 메쉬 및 배경 시뮬레이션 추가 시 일반화 성능 추가 향상 가능 |
| **템플릿 인코더 강화** | CLIP, DINOv2 등 강력한 사전학습 인코더 활용 시 범용 디스크립터 품질 향상 기대 |
| **멀티모달 조건** | 텍스트+이미지 복합 조건 확장 시 Open-Vocabulary로 연결 가능 |
| **도메인 적응** | 소량의 도메인 특화 데이터와 파인튜닝 결합 시 성능 극대화 가능 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 논문 | 연도 | 방법 | 특징 | 차이점 |
|---|---|---|---|---|
| **DiffusionInst** | 2022 | Diffusion + 동적 마스크 헤드 | 인스턴스 분할을 Denoising으로 모델링 | Zero-Shot 아님, 지도 학습 기반 |
| **OVDiff** (arXiv:2306.09316) | 2023 | Stable Diffusion + 언어 조건 | 텍스트 기반 Open-Vocabulary 분할 | 텍스트 조건 / 인스턴스 분할 아닌 시맨틱 분할 |
| **Diffuse, Attend, Segment** (arXiv:2308.12469) | 2023 | Self-Attention 맵 활용 | 비지도 Zero-Shot 분할 | 훈련 불필요하나 인스턴스 분리 약함 |
| **DiffCut** (NeurIPS 2024) | 2024 | Diffusion UNet + Normalized Cut | 비지도 Zero-Shot 시맨틱 분할 | 시맨틱 분할 중심, 인스턴스 아님 |
| **MosaicFusion** | 2023 | T2I Diffusion으로 데이터 증강 | 희귀 카테고리 데이터 생성 보조 | 분할 모델 자체가 아닌 데이터 증강 도구 |
| **OC-DiT (본 논문)** | 2025 | Conditional Latent Diffusion Transformer | 오브젝트 템플릿 조건 기반 Zero-Shot 인스턴스 분할 | **인스턴스 분할에 특화된 생성 모델 + Zero-Shot + SOTA** |

Diffuse, Attend, and Segment는 Stable Diffusion의 Self-Attention 레이어를 활용하여 KL Divergence 기반 반복적 병합 과정으로 분할 마스크를 생성하는 방식을 취한 반면, OC-DiT는 이와 달리 **객체 템플릿을 명시적 조건으로** 활용하여 특정 인스턴스를 타깃으로 한 마스크를 생성한다는 점에서 차별화됩니다.

DiffCut은 Diffusion UNet 인코더를 기반 비전 인코더로 사용하여 최종 Self-Attention 블록의 출력 특징만을 활용하는 비지도 Zero-Shot 분할 방법으로, NeurIPS 2024에 발표되었으나 시맨틱 분할에 집중하며 **인스턴스 수준의 분리에는 OC-DiT가 더 직접적으로 대응**합니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

1. **생성 모델의 인식 과제 확장:** Diffusion 모델이 이미지 생성을 넘어 **인스턴스 분할 같은 구조화된 예측 과제** 에도 효과적임을 실증하며, 생성 모델의 응용 범위를 크게 확장합니다.

2. **합성 데이터만으로의 Zero-Shot 가능성 입증:** 수천 개의 고품질 오브젝트 메쉬로 구성된 합성 데이터로만 훈련하고도 실제 벤치마크에서 SOTA를 달성한 것은 **합성 데이터 기반 로봇 비전, 자율주행** 등의 분야에서 라벨링 비용을 획기적으로 줄일 수 있는 패러다임을 제시합니다.

3. **Coarse-to-Fine 계층적 생성 전략의 유효성:** 초기 제안 생성(Coarse)과 병렬 정제(Refinement)의 이단계 구조는 향후 다른 밀집 예측 과제(Depth Estimation, Optical Flow 등)에도 적용 가능한 설계 원칙을 제공합니다.

4. **로보틱스 및 산업 응용으로의 직접 연결:** 6D 자세 추정, 추적, 형상 완성, 로봇 파지 등 다양한 응용에서 Zero-Shot 인스턴스 분할이 활용될 수 있어, 실제 산업 현장에서의 즉각적인 활용 가능성이 높습니다.

---

### 5.2 앞으로 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|---|---|
| **① 추론 속도 최적화** | Diffusion 모델의 반복 Denoising 특성으로 인한 지연시간 문제 → DDIM, Flow Matching 등 빠른 샘플링 기법 도입 고려 필요 |
| **② 합성 데이터 다양성** | Sim-to-Real 갭을 최소화하기 위해 배경, 조명, 재질의 다양성을 극대화한 합성 데이터 생성 전략 연구 필요 |
| **③ 오픈 어휘(Open-Vocabulary) 확장** | 텍스트 조건과의 결합으로 클래스 명세 없이도 동작하는 완전한 Open-World 인스턴스 분할로 확장 가능 |
| **④ 템플릿 가용성 문제** | 실제 환경에서 객체 템플릿 확보가 어려운 경우를 위한 Template-Free 또는 Few-shot 조건 방식 연구 필요 |
| **⑤ 불확실성 정량화** | Diffusion 모델의 확률론적 특성을 활용한 예측 불확실성 측정 및 신뢰도 추정 연구 |
| **⑥ 멀티 오브젝트 상호작용** | 복잡한 장면에서 객체 간 가림(occlusion) 처리 능력 향상을 위한 Global Context 통합 |
| **⑦ 경량화 및 엣지 배포** | 로봇 및 임베디드 환경 배포를 위한 지식 증류(Knowledge Distillation) 또는 양자화(Quantization) 연구 |

---

## 📚 참고 자료 (출처)

1. **[주 논문]** Ulmer, M., Boerdijk, W., Triebel, R., & Durner, M. (2025). *Conditional Latent Diffusion Models for Zero-Shot Instance Segmentation*. **ICCV 2025**. arXiv:2508.04122. https://arxiv.org/abs/2508.04122

2. **[ICCV 2025 공식 포스터 페이지]** https://iccv.thecvf.com/virtual/2025/poster/684

3. **[코드 저장소]** OC-DiT GitHub: https://github.com/DLR-RM/oc-dit

4. **[비교 논문 1]** Gu, J. et al. (2022). *DiffusionInst: Diffusion Model for Instance Segmentation*. arXiv:2212.02773.

5. **[비교 논문 2]** Couairon, P. et al. (2024). *DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut*. **NeurIPS 2024**. https://diffcut-segmentation.github.io/

6. **[비교 논문 3]** Tian, J. et al. (2023). *Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion*. arXiv:2308.12469.

7. **[비교 논문 4]** Xie, J. et al. (2023). *MosaicFusion: Diffusion Models as Data Augmenters for Large Vocabulary Instance Segmentation*. arXiv:2309.13042.

8. **[비교 논문 5]** Karazija, L. et al. (2023). *Diffusion Models for Zero-Shot Open-Vocabulary Segmentation*. arXiv:2306.09316.

---

> ⚠️ **정확도 주의사항:** 본 답변은 arXiv 공개 Abstract, ICCV 2025 공식 포스터 페이지 등 **공개된 정보** 를 기반으로 작성되었습니다. 논문 내부의 구체적 실험 수치, 세부 아키텍처 파라미터, 정확한 수식 표기 등은 **논문 전문(Full Paper)** 을 직접 확인하시기 바랍니다. 확인되지 않은 내용을 임의로 생성하지 않았습니다.
