
# Diffusion Curriculum (DisCL): Synthetic-to-Real Data Curriculum via Image-Guided Diffusion

> **논문 정보**
> - **저자:** Yijun Liang\*, Shweta Bhardwaj\*, Tianyi Zhou (University of Maryland, College Park)
> - **arXiv ID:** [2410.13674](https://arxiv.org/abs/2410.13674) (2024년 10월 발표, ICCV 2025 게재 확정)
> - **프로젝트:** https://github.com/tianyi-lab/DisCL
> - **발표 버전:** v1~v4 (최신 v4: 2025년 9월)

---

## 1. 📌 핵심 주장 및 주요 기여 (요약)

저품질 또는 희소한 데이터는 실제 딥러닝 모델 훈련에 심각한 도전 과제를 제시한다. 이 논문은 이 문제를 해결하기 위해 **Diffusion Curriculum (DisCL)** 을 제안합니다.

### 🔑 핵심 주장 3가지

| 주장 | 내용 |
|------|------|
| **한계 지적** | 텍스트 기반 생성만으로는 합성 이미지가 원본 분포에서 벗어나 성능을 저하시킴 |
| **핵심 제안** | Image Guidance를 통해 Synthetic ↔ Real의 연속 스펙트럼 데이터를 생성 |
| **커리큘럼** | 학습 단계별로 guidance level을 조정하여 "어려운 샘플"을 점진적으로 학습 |

### 🏆 주요 기여

저자들은 이미지 가이드 확산 모델을 이용해 커리큘럼 학습을 위한 Synthetic-to-Real 데이터 스펙트럼을 생성하는 프레임워크인 DisCL을 제안하며, 이미지 가이던스를 점진적으로 조정함으로써 어렵고 희소한 샘플에 대한 학습을 개선한다.

1. **Image-Guided Diffusion 기반 Syn-to-Real 스펙트럼 데이터 생성**
2. **Non-Adaptive & Adaptive Curriculum 전략 설계**
3. **Long-tail 분류 및 저품질 데이터 학습이라는 두 어려운 과제에 적용**

---

## 2. 🔍 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

---

### 2-1. 해결하고자 하는 문제

기존 머신러닝 접근법은 데이터의 품질과 양에 크게 의존하지만, 실제 환경에서 수집된 데이터는 품질과 수량이 보장되지 않는다. 예를 들어 동물 카메라, 교통 카메라, 로봇 카메라가 촬영한 장면의 조명, 날씨, 모션 블러, 객체 위치 등을 통제하기 어려우며, 클래스 간 데이터 불균형으로 인해 tail 클래스에서 모델 성능이 크게 저하된다.

**구체적 문제 두 가지:**
- 🐘 **Long-tail 분류:** tail 클래스의 데이터가 극히 적어 일반화 불가
- 📷 **저품질 데이터 학습:** 카메라 트랩 등 환경에서 획득한 이미지의 품질이 낮음

저품질/희소 데이터는 모델이 학습 분포와 테스트 분포 간의 격차에 더 취약하게 만들어 OOD 도전 과제를 제기하며, 이러한 "어려운" 훈련 데이터는 효과적인 학습을 방해하고 편향이나 이상치를 유발하며 다른 데이터의 학습에도 영향을 줄 수 있다.

텍스트만을 사용한 가이던스는 합성 이미지와 원본 이미지 간의 근접성을 제어할 수 없어, 모델 성능에 해로운 분포 외 데이터가 생성된다.

---

### 2-2. 제안 방법 (수식 포함)

#### 📐 핵심 개념: Image Guidance 강도 $\lambda$

DisCL의 핵심은 **이미지 가이던스 강도** $\lambda \in [0, 1]$를 통해 합성 이미지의 위치를 Synthetic ↔ Real 스펙트럼 위에서 제어하는 것입니다.

각 보간 레벨의 합성 데이터는 텍스트 프롬프트(예: 클래스 이름)와 실제 이미지 양쪽의 가중 가이던스 하에 생성된다. 강한 이미지 가이던스는 원본 이미지와의 시각적 유사성을 보존하는 반면, 약한 이미지 가이던스는 저품질/희소 데이터에 대해 고품질이고 다양하며 잠재적으로 더 쉬운(예: 전형적인 특징을 가진) 데이터를 생성할 수 있다.

**생성 이미지의 조건부 분포:**

$$
x_{\text{syn}}^{(\lambda)} \sim p_\theta(x \mid c_{\text{text}}, c_{\text{image}}; \lambda)
$$

여기서:
- $c_{\text{text}}$: 텍스트 프롬프트 (예: 클래스 이름)
- $c_{\text{image}}$: 실제 이미지 가이던스
- $\lambda$: 이미지 가이던스 강도 ($\lambda \uparrow$ → 실제 이미지에 더 가까워짐)

**가이던스 방향의 트레이드오프:**

$$
\text{이미지 유사도} \propto \lambda, \quad \text{다양성(Diversity)} \propto (1 - \lambda)
$$

즉,
- **강한 가이던스** ($\lambda \to 1$): 실제 이미지와 유사 → 학습하기 어렵지만 분포 격차 작음
- **약한 가이던스** ($\lambda \to 0$): 다양하고 쉬운 이미지 → 분포 격차 큼

강한 이미지 가이던스로 생성된 이미지는 훈련 데이터와 유사하지만 학습하기 어렵고, 약한 이미지 가이던스의 합성 이미지는 모델이 학습하기 더 쉽지만 원본 데이터와의 분포 격차가 더 커진다.

#### 📐 CLIPScore 기반 품질 필터링

생성된 이미지 중 품질이 낮은 것을 제거하기 위해 CLIPScore 기반 필터링을 적용:

$$
\text{CLIPScore}(x_{\text{syn}}, c_{\text{text}}) = \cos\left(\text{CLIP}_v(x_{\text{syn}}),\ \text{CLIP}_t(c_{\text{text}})\right)
$$

CLIPScore 임계값의 선택은 해당 작업에 내재된 생성 품질과 신중하게 조율되어야 함을 절제 연구(Ablation Study) 결과가 보여준다.

#### 📐 Adaptive Curriculum 전략: Progress 기반 $\lambda$ 선택

각 에폭(epoch)마다 각 $\lambda$에 해당하는 검증 서브셋에서 ground-truth 클래스 신뢰도의 향상으로 정의된 "진전도(progress)"를 기반으로 이미지 가이던스 레벨 $\lambda$를 선택하며, 가장 높은 진전도를 보이는 가이던스 레벨이 다음 에폭의 훈련에 선택된다.

수식으로 표현하면:

$$
\lambda_t^* = \arg\max_{\lambda \in \Lambda} \Delta_{\text{progress}}(\lambda, t)
$$

$$
\Delta_{\text{progress}}(\lambda, t) = \mathbb{E}_{(x,y) \in \mathcal{V}_\lambda}\left[p_t(y \mid x) - p_{t-1}(y \mid x)\right]
$$

여기서 $\mathcal{V}_\lambda$는 가이던스 레벨 $\lambda$에 대응하는 검증 서브셋.

---

### 2-3. 모델 구조 (Two-Phase Pipeline)

DisCL의 개요는 두 단계로 구성된다: **(Phase 1) Syn-to-Real 데이터 생성**과 **(Phase 2) Generative Curriculum 학습**이다.

```
┌─────────────────────────────────────────────────────────────┐
│                    DisCL Pipeline                           │
│                                                             │
│  Phase 1: Syn-to-Real Data Generation                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Pretrained Model로 "Hard Samples" 식별            │  │
│  │ 2. Hard Samples → Image Guidance로 사용              │  │
│  │ 3. λ ∈ {λ₁, λ₂, ..., λₙ} 로 다양한 강도의         │  │
│  │    합성 이미지 생성 (Stable Diffusion 기반)          │  │
│  │ 4. CLIPScore 필터링으로 저품질 제거                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  Phase 2: Generative Curriculum Learning                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Non-Adaptive: 사전 정의된 λ 스케줄 (쉬운→어려운)   │  │
│  │ Adaptive: 검증 Progress 기반으로 λ 동적 선택        │  │
│  │ → Hard sample용 합성 데이터 + 일반 실제 데이터 혼합 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

Phase 1에서는 사전 학습된 모델을 사용해 원본 이미지에서 "어려운(hard)" 샘플을 식별하고, 이를 가이던스로 사용하여 이미지 가이던스 강도 $\lambda$를 변화시켜 전체 합성-실제 스펙트럼 이미지를 생성한다. Phase 2에서는 커리큘럼 전략(Non-Adaptive 또는 Adaptive)이 최대 진전도를 위해 $\lambda_i$를 결정하여 전체 스펙트럼에서 훈련 데이터를 선택한다.

**Long-tail 분류에서의 커리큘럼 방향:**

먼저 tail 클래스의 다양한 합성 이미지에 모델을 노출시키고, 점진적으로 원본 이미지와 유사한 task-specific 분포로 전환한다.

**저품질 데이터 학습에서의 커리큘럼 방향:**

낮은 가이던스의 고품질 이미지에 집중하여 전형적인 특징(Prototypical Feature)을 학습하는 것을 워밍업으로 삼고, 이후 다양성이나 품질이 약할 수 있는 높은 가이던스 이미지를 학습한다.

---

### 2-4. 성능 향상

iWildCam 데이터셋에 DisCL을 적용했을 때 OOD 및 ID 매크로 정확도에서 각각 2.7%와 2.1%의 향상을 보였다. ImageNet-LT에서는 DisCL이 base 모델의 tail-class 정확도를 4.4%에서 23.64%로 향상시키고, 전체 클래스 정확도에서 4.02% 향상을 이끌었다.

| 데이터셋 | 지표 | 향상폭 |
|----------|------|--------|
| iWildCam | OOD Macro-Accuracy | **+2.7%** |
| iWildCam | ID Macro-Accuracy | **+2.1%** |
| ImageNet-LT | Tail-class Accuracy | **4.4% → 23.64%** |
| ImageNet-LT | All-class Accuracy | **+4.02%** |

Long-tail 벤치마크(ImageNet-LT, iNaturalist, CIFAR-100) 및 저품질 데이터(iWildCam) 전반에 걸쳐 DisCL은 다양한 base 모델 설정에서 tail-class 및 전체 정확도를 일관되게 향상시켰다.

비교 베이스라인:
LP-FT (Kumar et al., 2022), FLYP (Goyal et al., 2023), ALIA (Dunlap et al., 2024)의 세 가지 벤치마크 알고리즘과 비교 실험을 수행하였다.

---

### 2-5. 한계

1. **합성 데이터 스케일링 한계:**
DisCL은 3~4배 스케일까지는 tail 클래스 정확도를 지속적으로 향상시키지만, 이 지점을 넘어서면 many 및 medium 클래스에서 약간의 성능 저하와 함께 이득이 줄어든다. 따라서 모든 DisCL 학습 실험에서는 합성 tail 데이터를 3배 스케일로 사용하였다.

2. **도메인 격차 문제:**
지나치게 높거나 낮은 가이던스 레벨은 학습 데이터와 테스트 분포 간의 도메인 격차를 확대할 수 있다.

3. **CLIPScore 임계값 민감도:**
절제 연구 결과는 CLIPScore 임계값의 선택이 해당 작업에 내재된 생성 품질과 신중하게 조율되어야 함을 보여준다.

4. **오프라인 생성 비용:** 각 $\lambda$ 수준별 합성 데이터를 사전에 생성해야 하므로 상당한 컴퓨팅 비용이 발생합니다 (논문 본문에서도 언급).

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

이 논문의 가장 중요한 기여 중 하나는 **OOD(Out-of-Distribution) 일반화 성능 향상**입니다.

### 3-1. 분포 격차 해소를 통한 일반화

DisCL은 저품질 또는 희소 데이터를 다룰 때 모델 성능을 향상시키기 위해 설계된 새로운 패러다임을 제시하며, 합성 데이터 스펙트럼을 이용하여 특히 어려운 샘플에 대해 원본 데이터와 목표 데이터 분포 간의 격차를 효과적으로 줄인다.

### 3-2. Hard Sample 학습을 통한 일반화

Syn-to-Real 보간은 사전 정의된 스케줄 또는 학습 동역학에 따라 가이던스 레벨을 선택함으로써 서로 다른 훈련 단계에 맞게 데이터의 품질, 다양성 및/또는 난이도를 조정할 수 있는 새로운 생성 커리큘럼(generative curriculum) 공간을 만들어낸다.

### 3-3. Tail Class 일반화

많은 클래스 정확도가 스케일링에 걸쳐 가장 낮은 저하를 보이는 것은 hard-sample 합성 데이터가 many-class 표현을 방해하지 않고 tail 클래스 일반화를 향상시킬 수 있다는 long-tail 학습의 연구 결과를 확인시켜 준다.

### 3-4. OOD 일반화 수치

iWildCam은 카메라 트랩으로 촬영된 182가지 동물 종을 분류하는 작업이며, 모델 성능은 표준 OOD 및 ID 테스트 셋에서 macro F1 점수로 평가된다. 이 실험은 전형적인 실제 세계 OOD 시나리오를 다루며, DisCL이 이를 **+2.7% OOD 향상**으로 개선했다는 것은 모델의 실제 환경 적용 가능성을 크게 높입니다.

### 3-5. 일반화 향상 메커니즘 요약

```
약한 가이던스 (λ↓)     →  다양하고 전형적인 특징 학습 (Prototypical Features)
        ↓ 커리큘럼
강한 가이던스 (λ↑)     →  실제 데이터와 유사한 분포에서 정밀 학습
        ↓
OOD 일반화 ↑ + ID 정확도 ↑
```

---

## 4. 🔮 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 연구에 미치는 영향

#### ① 생성 모델 + 커리큘럼 학습의 새로운 패러다임

Syn-to-Real 보간은 사전 정의된 스케줄 또는 학습 동역학에 따라 가이던스 레벨을 선택함으로써 서로 다른 훈련 단계에 맞게 데이터의 품질, 다양성, 난이도를 조정하는 생성 커리큘럼을 설계할 수 있는 새로운 합성 데이터 공간을 만들어낸다.

이는 기존의 고정된 데이터셋 기반 커리큘럼 학습(Bengio et al., 2009)에서 **동적 생성 기반 커리큘럼**으로의 패러다임 전환을 촉진합니다.

#### ② Self-Evolving AI의 가능성

확산 모델은 텍스트 가이드 프롬프트를 통해 고품질의 다양한 합성 데이터를 생성함으로써 자기 진화(self-evolving) AI를 구축하는 새로운 문을 열어준다.

#### ③ 의료, 로보틱스, 자율주행 등 다양한 도메인 적용

확산 기반 합성 증강과 커리큘럼 학습의 통합이 데이터 불균형과 제한된 어노테이션으로 인해 기존 AI 모델에 도전적인 어려운 샘플의 탐지를 향상시킬 수 있는지를 평가하는 연구가 이미 의료 분야(흉부 X선 폐결절 탐지)에서 진행되고 있다.

### 4-2. 앞으로의 연구에서 고려할 점

| 고려 사항 | 설명 |
|-----------|------|
| **온라인 생성** | 현재 오프라인으로 사전 생성하는 방식 → 학습 중 실시간 동적 생성으로 발전 필요 |
| **$\lambda$ 자동화** | 현재 수동으로 조정하는 CLIPScore 임계값을 자동화하는 방법 연구 필요 |
| **도메인 일반화** | 의료, 위성, 로보틱스 등 다양한 도메인에서의 검증 필요 |
| **다중 모달 확장** | 텍스트-이미지 외 비디오, 3D 등 다른 모달리티로의 확장 가능성 |
| **합성 데이터 한계** | 3~4배 이상 합성 데이터 사용 시 성능 저하 현상의 이론적 규명 |
| **비전-언어 모델 통합** | CLIP, BLIP 등과의 더 긴밀한 통합으로 guidance quality 향상 |
| **공정성 및 편향** | 합성 데이터가 특정 집단에 대한 편향을 증폭시킬 가능성 검토 |

---

## 5. 📊 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 주요 특징 | DisCL과 비교 |
|------|------|-----------|-------------|
| **ALIA** (Dunlap et al., 2024, NeurIPS) | 자동 확산 기반 증강 | 텍스트 가이드로 비전 데이터셋 다양화 | 텍스트만 사용, 분포 제어 부족 |
| **DreamDA** (Fu et al., 2024) | Diffusion 기반 생성 증강 | 다양한 태스크에서 합성 데이터 활용 | 커리큘럼 전략 없음 |
| **LDMLR** (2024) | 확산 모델 + Long-tail | Latent 기반 long-tail 인식 | 정적 데이터 생성, 커리큘럼 없음 |
| **LP-FT** (Kumar et al., 2022) | Fine-tuning 전략 | Zero-shot 모델 파인튜닝 개선 | 데이터 생성 없음 |
| **FLYP** (Goyal et al., 2023, CVPR) | CLIP 파인튜닝 | Pretrain 방식과 동일하게 파인튜닝 | 데이터 생성 없음 |
| **Feedback-guided Synthesis** (Hemmat et al., 2023) | 피드백 기반 합성 | 불균형 분류를 위한 데이터 합성 | 단순 합성, 커리큘럼 없음 |
| **DisCL (본 논문)** (2024, ICCV 2025) | Image-Guided Diffusion + Curriculum | **Syn-to-Real 스펙트럼 + 적응형 커리큘럼** | **Image guidance로 분포 격차 정밀 제어** |

데이터 증강과 합성은 어려운 실제 데이터의 도전에 대응하기 위해 연구되어 왔다. 희소 클래스에 사전 정의된 변환을 적용하거나 배경을 수정하는 방식이 사용되었으나, 증강된 데이터는 원본과의 충분한 다양성이 부족하여, 최근 GAN이나 Stable Diffusion과 같은 텍스트-이미지 생성 모델이 더 정교한 데이터 합성을 가능하게 했다.

DisCL의 차별점은 **단순 합성**이 아닌 **연속 스펙트럼** 위에서 **적응적 커리큘럼**을 설계했다는 점이며, 이는 이전 연구들에서 시도되지 않은 접근법입니다.

---

## 📚 참고 자료 및 출처

| 번호 | 제목 및 출처 |
|------|-------------|
| 1 | **[Primary Paper]** Yijun Liang, Shweta Bhardwaj, Tianyi Zhou. "Diffusion Curriculum: Synthetic-to-Real Generative Curriculum Learning via Image-Guided Diffusion." arXiv:2410.13674, ICCV 2025. https://arxiv.org/abs/2410.13674 |
| 2 | **[Project Page]** DisCL Official Project Page. https://joliang17.github.io/DisCL/ |
| 3 | **[PDF]** arXiv PDF: https://arxiv.org/pdf/2410.13674 |
| 4 | **[HTML Full Paper]** arXiv HTML v4: https://arxiv.org/html/2410.13674v4 |
| 5 | **[ICCV 2025 Official]** OpenAccess CVF: https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Diffusion_Curriculum_Synthetic-to-Real_Data_Curriculum_via_Image-Guided_Diffusion_ICCV_2025_paper.pdf |
| 6 | **[OpenReview]** https://openreview.net/forum?id=0RgLIMh94b |
| 7 | **[HuggingFace]** https://huggingface.co/papers/2410.13674 |
| 8 | **[ResearchGate]** https://www.researchgate.net/publication/385010285 |
| 9 | **[GitHub Code]** https://github.com/tianyi-lab/DisCL |
| 10 | **[관련 연구]** Bengio et al. "Curriculum Learning." ICML 2009. |
| 11 | **[관련 연구]** Dunlap et al. "Diversify Your Vision Datasets with Automatic Diffusion-Based Augmentation." NeurIPS 2024. |
| 12 | **[관련 연구]** Ho & Salimans. "Classifier-Free Diffusion Guidance." arXiv:2207.12598, 2022. |
| 13 | **[관련 연구]** Hessel et al. "CLIPScore: A Reference-Free Evaluation Metric for Image Captioning." 2022. |
| 14 | **[관련 연구]** Goyal et al. "Finetune Like You Pretrain." CVPR 2023. |

> ⚠️ **주의:** 논문 내부의 상세 수식 일부(특히 Stable Diffusion의 noise schedule 등)는 공개된 HTML/PDF 버전에서 직접 확인 가능하며, 본 답변은 공개된 arXiv 및 ICCV 2025 official 버전을 기반으로 작성하였습니다. 논문의 세부 구현 수식은 원문을 직접 참조하시기를 권장합니다.
