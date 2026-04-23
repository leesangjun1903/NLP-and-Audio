
# TokenCompose: Text-to-Image Diffusion with Token-level Supervision

> **논문 정보:**
> - **저자:** Zirui Wang, Zhizhou Sha, Zheng Ding, Yilin Wang, Zhuowen Tu
> - **학회:** CVPR 2024, pp. 8553–8564
> - **arXiv:** [2312.03626](https://arxiv.org/abs/2312.03626)
> - **프로젝트 페이지:** [mlpc-ucsd.github.io/TokenCompose](https://mlpc-ucsd.github.io/TokenCompose/)
> - **GitHub:** [github.com/mlpc-ucsd/TokenCompose](https://github.com/mlpc-ucsd/TokenCompose)

---

## 1. 핵심 주장과 주요 기여 요약

### 🎯 핵심 주장

TokenCompose는 사용자가 지정한 텍스트 프롬프트와 모델이 생성한 이미지 사이의 일관성을 향상시키는 Latent Diffusion Model(LDM) 기반 텍스트-이미지 생성 모델입니다. 기존 LDM의 표준 노이즈 제거 과정은 텍스트 프롬프트를 조건으로만 사용하며, 텍스트 프롬프트와 이미지 내용 간의 일관성에 대한 **명시적 제약이 없어** 복수의 객체 카테고리를 구성하는 데 불만족스러운 결과를 낳습니다.

### 🏆 주요 기여 (Contributions)

| 기여 항목 | 설명 |
|-----------|------|
| **Token-level 일관성 손실** | 토큰별 일관성 항을 도입해 multi-category 구성 강화 |
| **MultiGen 벤치마크** | 복수 카테고리 인스턴스 생성을 평가하는 새로운 벤치마크 |
| **추가 모듈 없는 파인튜닝** | 기존 파이프라인에 그대로 적용 가능 |
| **Plug-and-play 모델** | HuggingFace Diffusers 등 표준 라이브러리와 호환 |

TokenCompose는 파인튜닝 단계에서 이미지 내용과 객체 분할 맵 간의 **토큰 단위 일관성 항**을 도입하여 다중 카테고리 인스턴스 구성을 개선하고자 합니다. 추가적인 인간 레이블링 없이 기존 텍스트 조건부 확산 모델의 학습 파이프라인에 직접 적용할 수 있으며, Stable Diffusion을 파인튜닝함으로써 다중 카테고리 구성 및 포토리얼리즘 모두에서 상당한 성능 향상을 보입니다.

---

## 2. 상세 분석: 문제 → 방법(수식) → 구조 → 성능 → 한계

### 2.1 해결하고자 하는 문제

텍스트-이미지 확산 모델들이 품질·해상도·다양성 면에서 놀라운 발전을 이뤘지만, 텍스트 프롬프트와 생성 이미지 내용 사이에는 **주요 일관성 문제**가 여전히 존재합니다. 특히 현실 세계에서 동시에 잘 등장하지 않는 여러 객체 카테고리가 프롬프트에 포함될 경우, 모델은 구성 능력을 잃어 버려—객체가 이미지에 나타나지 않거나, 생성 결과가 보기 좋지 않을 수 있습니다.

**핵심 문제 정리:**
- ❌ Stable Diffusion은 텍스트 프롬프트를 조건으로만 사용 → 텍스트-이미지 일관성 보장 없음
- ❌ 복수 객체 카테고리(예: "고양이와 와인잔")를 동시에 정확히 생성하지 못함
- ❌ Cross-attention map이 객체를 구분하지 못하고 혼용(confusion) 발생

---

### 2.2 제안하는 방법 및 학습 파이프라인

#### Step 1: 데이터 전처리

학습 프롬프트가 주어지면, **POS 태거**(품사 태거)와 **Grounded SAM**을 활용하여 프롬프트의 명사 토큰에 해당하는 이미지의 **이진 분할 맵(binary segmentation maps)** 을 모두 추출합니다.

그 후, 확산 모델의 denoising U-Net을 원래의 **노이즈 제거 목적함수**와 **토큰 단위 목적함수**를 동시에 최적화합니다.

#### Step 2: 학습 목적함수 (수식)

표준 LDM의 기본 목적함수는 다음과 같습니다:

$$
\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}, t}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \tau_\theta(y))\|^2\right]
$$

여기서:
- $\mathbf{z}_t$: 시각 타임스텝 $t$에서의 노이즈가 추가된 잠재 변수(latent)
- $\boldsymbol{\epsilon}$: 실제 노이즈
- $\boldsymbol{\epsilon}_\theta$: U-Net 기반 노이즈 예측 함수
- $\tau_\theta(y)$: 텍스트 인코더(CLIP)에 의해 인코딩된 텍스트 임베딩

**TokenCompose는 여기에 두 가지 추가 손실 항을 도입합니다:**

**① Token Loss ($\mathcal{L}_\text{token}$)**

Cross-attention map $\mathcal{A}_i^{(m)}$ 과 세그멘테이션 맵 $\mathcal{M}_i^{(m)}$ 사이의 일관성을 최대화합니다:

$$
\mathcal{L}_{\text{token}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{m} \text{IoU}\!\left(\mathcal{A}_i^{(m)}, \mathcal{M}_i^{(m)}\right)
$$

또는 더 일반적으로, 각 명사 토큰 $i$에 대해 attention 활성화 영역이 분할 맵과 일치하도록 강제합니다.

이 문제를 완화하기 위해, cross-attention map의 활성화 영역을 감독하는 학습 제약을 추가합니다. 각 텍스트 토큰 $i$에 대해, 이미지 이해를 위해 학습된 파운데이션 모델을 활용하여 해당 이미지로부터 분할 마스크 $\mathcal{M}_i$를 추출합니다.

**② Pixel Loss ($\mathcal{L}_\text{pixel}$)**

표준 LDM의 노이즈 예측 손실을 픽셀 레벨에서 유지:

$$
\mathcal{L}_{\text{pixel}} = \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}, t}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \tau_\theta(y))\|^2\right]
$$

**③ 최종 통합 학습 목적함수**

$\mathcal{L}\_\text{pixel}$이 대략 일정하게 유지되면서 $\mathcal{L}\_\text{token}$에 의해 모델이 공동 최적화됩니다. 토큰 수준 및 픽셀 수준 최적화에 대한 시각화를 통해 $\mathcal{L}_\text{pixel}$의 세분성 차이를 보여주며, 최종 학습 목적함수는 다음과 같습니다:

$$
\boxed{\mathcal{L}_\text{total} = \mathcal{L}_\text{pixel} + \lambda \cdot \mathcal{L}_\text{token}}
$$

여기서 $\lambda$는 두 손실 항의 균형을 조절하는 하이퍼파라미터입니다.

---

### 2.3 모델 구조

파인튜닝된 모델은 **추가 모듈을 포함하지 않으며**, HuggingFace Diffusers와 같은 표준 확산 모델 라이브러리에서 사전학습된 U-Net을 파인튜닝된 U-Net으로 교체하는 방식으로 **플러그앤플레이(plug-and-play)**로 사용할 수 있습니다.

```
[TokenCompose 학습 파이프라인]

  텍스트 프롬프트
       │
       ▼
  POS 태거 (명사 추출)
       │
       ▼
  Grounded SAM → Binary Segmentation Map {M_i}
       │
       ▼
  ┌─────────────────────────────────────┐
  │   Denoising U-Net (Stable Diffusion) │
  │                                     │
  │  Cross-Attention Map {A_i}          │
  │  ↓                                  │
  │  L_token: A_i ↔ M_i 일관성 강제     │
  │  L_pixel: 기존 노이즈 예측 손실     │
  └─────────────────────────────────────┘
       │
       ▼
  L_total = L_pixel + λ·L_token
```

**핵심 구조적 특징:**
- 백본: **Stable Diffusion 1.4** U-Net (기존 아키텍처 그대로 유지)
- 추가 파라미터: **없음** (추가 모듈 없이 기존 U-Net 파인튜닝만)
- Cross-attention map 해상도 정합: **Bilinear interpolation + Binarization**으로 크기 통일

Stable Diffusion 1.4는 cross-attention map에서 객체를 구별하는 데 어려움을 겪는 반면, TokenCompose 모델은 객체를 효과적으로 그라운딩하는 데 탁월합니다. 타임스텝별 cross-attention map 시각화를 비교한 결과, 파인튜닝된 모델의 cross-attention map이 **훨씬 강력한 그라운딩 능력**을 보여줍니다.

---

### 2.4 MultiGen 벤치마크

기존의 다중 카테고리 인스턴스 구성 벤치마크들은 주로 **두 가지 카테고리**의 생성에 초점을 맞추고 있습니다. 반면 선도적인 이미지 생성 모델들은 다중 카테고리 인스턴스 구성에서 크게 개선되었습니다. 이 연구 격차를 채우기 위해 **MultiGen 벤치마크**를 제안하며, 이는 임의의 복수 카테고리 조합에서 객체를 생성하는 것을 요구하는 텍스트 프롬프트를 포함합니다.

- **MG2**: 2개 카테고리 동시 생성
- **MG3**: 3개 카테고리 동시 생성
- **MG4**: 4개 카테고리 동시 생성
- **MG5**: 5개 카테고리 동시 생성

---

### 2.5 성능 향상

성능 평가는 다중 카테고리 인스턴스 구성(VISOR 벤치마크의 **Object Accuracy(OA)**, MultiGen 벤치마크의 **MG2~5**), 포토리얼리즘(**COCO 및 Flickr30K Entities 검증 세트의 FID**), 추론 효율성을 기준으로 합니다. 모든 비교는 **Stable Diffusion 1.4**를 기반으로 합니다.

TokenCompose의 효과는 Frozen SD 모델, Composable Diffusion, Structured Diffusion, Layout Guidance Diffusion, Attend-and-Excite와의 비교를 통해 검증되었습니다.

**주요 비교 결과 요약 (SD 1.4 vs TokenCompose):**

| 지표 | SD 1.4 (기준) | TokenCompose | 비고 |
|------|--------------|--------------|------|
| Object Accuracy (OA) | 29.86 | **향상** | VISOR Benchmark |
| MG2 (COCO) | 90.72 | **향상** | 2-category |
| MG3 (COCO) | 50.74 | **향상** | 3-category |
| MG4 (COCO) | 11.68 | **향상** | 4-category |
| MG5 (COCO) | 0.88 | **향상** | 5-category |
| FID (COCO) | 20.88 | **향상** | Photorealism |

처음 세 열은 현실 세계에서 공동 등장 확률이 낮거나 인스턴스 크기 차이가 큰 두 카테고리의 구성을 보여주며, 마지막 세 열은 각 텍스트 토큰의 시각적 표현 이해가 필요한 세 가지 카테고리의 구성을 보여줍니다.

---

### 2.6 한계점 (Limitations)

TokenCompose는 이미지 이해 모델을 사용하여 텍스트 조건부 생성 모델에 이미지-토큰 일관성을 적용하려는 **선구적인 연구** 중 하나로서, 현재는 텍스트 프롬프트의 **명사 토큰에만** 감독 항을 추가합니다. 이 접근법이 다중 카테고리 인스턴스 구성을 크게 향상시키지만, **형용사, 동사, 한정사** 등 프롬프트의 더 많은 요소들을 세밀한 토큰 수준 학습 목표로 활용하는 것이 향후 과제로 남아 있습니다.

또한 논문은 **다양하고 복잡한 텍스트 프롬프트에 대한 모델의 견고성**을 탐구하지 않고 있으며, 이것이 향후 연구의 중요한 영역이 될 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

TokenCompose의 일반화 능력은 여러 측면에서 분석될 수 있습니다.

### 3.1 기존 파이프라인과의 호환성

TokenCompose는 **추가적인 인간 레이블링 없이** 텍스트 조건부 확산 모델의 기존 학습 파이프라인에 직접 적용될 수 있어, 다양한 기반 모델로의 확장이 용이합니다.

### 3.2 Downstream Task에서의 일반화

다중 카테고리 인스턴스 구성이 성공적인 하위 텍스트 조건부 구성 생성의 전제 조건으로 작용하므로, 프롬프트에 언급된 모든 인스턴스를 생성할 확률이 높아지면서 **하위 메트릭의 개선**이 이루어진다고 추론할 수 있습니다.

### 3.3 후속 연구에서 확인된 일반화

TokenCompose는 **RealCompo(NeurIPS 2024)**에서 향상된 구성력을 위한 **기반 모델**로 활용되었으며, token-level consistency terms으로 파인튜닝된 Stable Diffusion 모델로서 multi-category 인스턴스 구성 및 포토리얼리즘에서 개선을 보입니다.

RealCompo의 다른 모델로의 일반화 비교에서, Stable Diffusion v1.5, TokenCompose 등 T2I 모델과 GLIGEN, Layout Guidance 등 L2I 모델을 쌍으로 결합하여 RealCompo의 네 가지 버전을 구성하는 실험이 수행되었습니다.

또한 TokenCompose의 학습 방법론은 **CoMat(NeurIPS 2024)**에 통합되어 텍스트-이미지 속성 할당에서 향상된 성능을 보였습니다.

### 3.4 일반화 한계 요인

현재 TokenCompose는 **명사 토큰에만** 감독 항을 추가하는 것으로 제한됩니다. 형용사, 동사, 한정사와 같은 더 많은 텍스트 요소들을 세밀한 토큰 수준 학습 목표로 활용하면 일반화 성능이 더 향상될 수 있습니다.

**일반화 향상 가능 방향 요약:**

```
현재 TokenCompose
  ├── 명사 토큰 → 이진 세그멘테이션 맵 감독
  │
  └── [미래 확장 가능성]
        ├── 형용사 → 속성 바인딩(attribute binding)
        ├── 동사   → 행동/관계 표현
        ├── 공간 관계 → 레이아웃 일관성
        └── 더 큰 기반 모델(SDXL, SD3 등)로의 전이
```

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 방법론 분류 및 비교표

| 방법 | 연도/학회 | 핵심 접근법 | 훈련 필요 | 추가 모듈 | 강점 |
|------|-----------|-------------|-----------|-----------|------|
| **Composable Diffusion** | 2022 | Score decomposition | ❌ | ❌ | 간단한 구성 |
| **Structured Diffusion** | 2022 | Constituency tree + attention 조작 | ❌ | ❌ | 구문 구조 활용 |
| **Attend-and-Excite** | 2023 | Gaussian kernel attention + 추론 시 최적화 | ❌ | ❌ | 학습 불필요 |
| **Layout Guidance Diffusion** | 2023 | Bounding box + gradient guidance | ❌ | 사용자 레이아웃 필요 | 공간 제어 |
| **TokenCompose** | **CVPR 2024** | **Token-level 세그멘테이션 감독 파인튜닝** | ✅ | ❌ | **다중 카테고리 구성 + 포토리얼리즘** |
| **RealCompo** | **NeurIPS 2024** | T2I + L2I 동적 밸런싱 | ❌ | 필요 | 현실감 + 구성력 균형 |
| **CoMat** | **NeurIPS 2024** | Image-to-Text 개념 매칭 파인튜닝 | ✅ | ❌ | 속성 바인딩 강화 |

### 4.2 RealCompo와의 비교

RealCompo는 T2I 모델과 공간 인식 이미지 확산 모델(레이아웃, 키포인트, 세그멘테이션 맵 등)의 장점을 활용하여 생성 이미지의 현실감과 구성력을 향상시키는 **학습 없는(training-free)** 프레임워크입니다. 두 모델의 강점을 노이즈 제거 과정에서 동적으로 균형잡는 새로운 **밸런서**가 제안되어, 추가 학습 없이 플러그앤플레이로 사용 가능합니다.

> **핵심 차이:** TokenCompose는 **파인튜닝 기반**으로 모델 가중치를 업데이트하는 반면, RealCompo는 **추론 시 두 모델을 동적으로 결합**합니다.

### 4.3 CoMat와의 비교

CoMat는 텍스트-이미지 불일치를 **개념 무시(concept ignorance)**와 **개념 오매핑(concept mismapping)**으로 분해하여, 이미지-텍스트 개념 매칭 메커니즘을 갖춘 엔드투엔드 확산 모델 파인튜닝 전략을 제안합니다. 이미지-텍스트 개념 활성화 모듈로 무시된 개념을 재방문하도록 하고, 속성 집중 모듈로 각 개체의 텍스트 조건을 해당 이미지 영역에 올바르게 매핑합니다.

CoMat의 학습 방법론은 **TokenCompose의 학습 방법론을 통합**하여 텍스트-이미지 속성 할당에서 향상된 성능을 보였습니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려점

### 5.1 연구적 영향

TokenCompose는 토큰 수준 감독을 통합하여 생성 이미지의 구성력과 현실감을 향상시키는 텍스트-이미지 생성의 중요한 발전을 나타냅니다. 이 접근법은 생성 과정에서 세밀한 지침을 제공하는 것의 잠재적 이점을 보여주며, **구성적 이해, 그라운딩 언어 모델링, 멀티모달 생성** 분야의 새로운 연구 방향을 열어줍니다.

근본적인 구성 생성 문제로서, TokenCompose의 학습 프레임워크와 벤치마크가 이해와 생성 간의 시너지를 효과적으로 활용하여 **어느 방향이든 개선**하는 미래 연구에 영감을 줄 것으로 기대됩니다.

### 5.2 구체적 연구 영향 및 고려점

#### ✅ 긍정적 영향

1. **이해 모델 ↔ 생성 모델 시너지 패러다임 확립**
   TokenCompose는 파운데이션 이미지 이해 모델을 활용하여 텍스트 조건부 생성 모델의 그라운딩 능력을 향상시키는 가능성을 탐구하며, 다중 카테고리 인스턴스 구성과 개선된 이미지 품질 모두에서 탁월한 성능을 보입니다. 또한 모델이 하나의 이미지에서 여러 카테고리의 인스턴스를 생성해야 하는 도전적인 **MultiGen 벤치마크**를 제안합니다.

2. **파인튜닝 기반 구성 향상의 기준 방법 제공**
   - RealCompo, CoMat 등 후속 연구의 베이스라인/베이스 모델로 광범위하게 사용됨

3. **레이블 없는 학습 가능성 입증**
   TokenCompose는 **추가 인간 레이블링 없이** 기존 텍스트 조건부 확산 모델의 학습 파이프라인에 직접 적용될 수 있어, 대규모 자동화 학습의 가능성을 보여줍니다.

#### ⚠️ 향후 연구 시 고려점

1. **명사 이외의 토큰 확장**
   현재는 명사 토큰에만 감독 항을 추가하고 있으나, **형용사(색상, 크기), 동사(행동), 한정사(수량)**에 대한 세밀한 토큰 수준 학습 목표로 확장하면 더욱 풍부한 텍스트-이미지 정렬을 달성할 수 있습니다.

2. **더 강력한 기반 모델로의 전이**
   - SDXL, Stable Diffusion 3, FLUX 등 최신 모델에 Token-level supervision 적용 실험 필요
   - 더 큰 모델에서도 동일한 파이프라인이 효과적인지 검증 필요

3. **Grounded SAM 의존성 관리**
   - 세그멘테이션 품질이 학습 성능에 직접 영향 → SAM 업그레이드(SAM2 등) 연계 연구 가능

4. **복잡한 공간 관계 표현**
   - "고양이 위에 있는 모자"와 같은 상대적 공간 관계는 현재 모델이 명시적으로 처리하지 않음
   - 레이아웃/공간 관계 정보를 token-level 목표에 통합하는 연구 필요

5. **학습 없는(training-free) 방법과의 통합**
   - TokenCompose(학습 기반)와 Attend-and-Excite, RealCompo 등 추론 시 제어 방법을 결합하는 하이브리드 접근 탐색

6. **다국어 및 다양한 도메인 일반화**
   - 영어 중심 CLIP 텍스트 인코더 의존 → 비영어 프롬프트 또는 전문 도메인(의료, 위성 이미지 등)에서의 성능 평가 필요

---

## 📚 참고 자료 (출처)

| 번호 | 제목 | 출처 |
|------|------|------|
| [1] | **TokenCompose: Text-to-Image Diffusion with Token-level Supervision** | arXiv:2312.03626 / CVPR 2024, pp. 8553-8564 |
| [2] | **CVPR 2024 Open Access (TokenCompose)** | https://openaccess.thecvf.com/content/CVPR2024/html/Wang_TokenCompose_... |
| [3] | **TokenCompose Project Page** | https://mlpc-ucsd.github.io/TokenCompose/ |
| [4] | **TokenCompose GitHub** | https://github.com/mlpc-ucsd/TokenCompose |
| [5] | **TokenCompose - IEEE Xplore** | https://ieeexplore.ieee.org/document/10657918/ |
| [6] | **TokenCompose - Semantic Scholar** | https://www.semanticscholar.org/paper/TokenCompose... |
| [7] | **RealCompo (NeurIPS 2024)** | arXiv:2402.12908 / OpenReview: https://openreview.net/forum?id=R8mfn3rHd5 |
| [8] | **CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching (NeurIPS 2024)** | arXiv:2404.03653 / OpenReview: https://openreview.net/forum?id=OW1ldvMNJ6 |
| [9] | **TokenCompose - AI Models FYI** | https://www.aimodels.fyi/papers/arxiv/tokencompose-text-to-image-diffusion-token-level |

---

> **⚠️ 정확도 관련 안내:**
> 논문 내 구체적인 수식의 세부 표기(예: 손실 함수의 정확한 IoU 형태, 가중치 $\lambda$ 값 등)는 HTML 변환 오류 등으로 인해 원문 PDF에서 완전한 형태로 확인하지 못한 부분이 있습니다. 수식의 개념적 구조는 논문의 방법론에 따라 정확하게 재구성하였으나, **최종 정확한 수식 확인은 원문 PDF**(arXiv:2312.03626)를 직접 참고하시기 바랍니다.
