
# Progressive Compositionality in Text-to-Image Generative Models

> **논문 정보**
> - **제목**: Progressive Compositionality in Text-to-Image Generative Models
> - **저자**: Evans Xu Han 외 3인
> - **arXiv**: [2410.16719](https://arxiv.org/abs/2410.16719) (2024년 10월 22일 제출, 2025년 4월 업데이트)
> - **학회**: ICLR 2025 (Poster)
> - **프로젝트 페이지**: https://evansh666.github.io/EvoGen_Page/
> - **코드**: https://github.com/evansh666/EvoGen

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

기존 접근법들은 고정된 사전 정의된 구조(compositional architecture)에 의존하거나 어려운 부정 캡션(negative captions)을 생성하는 방식으로, 새로운 분포로의 일반화를 제한한다. 이 논문은 **커리큘럼 훈련(curriculum training)** 이 생성 모델에 조합성(compositionality)에 대한 근본적인 이해를 부여하는 데 핵심적이라고 주장한다.

### 🏆 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **ConPair 데이터셋** | LLM + VQA 기반 고품질 대조 이미지 쌍 15k 자동 구축 |
| **EvoGen 프레임워크** | 3단계 커리큘럼 대조 학습 파이프라인 |
| **SOTA 성능** | T2I-CompBench에서 DALL-E 3를 포함한 모든 기준 모델 초과 |

ConPair는 최소한의 시각적 표현 차이를 가진 고품질 대조 이미지로 구성된 구성적 데이터셋이며, LLM을 활용하여 구성 프롬프트의 복잡성을 확장하는 동시에 자연스러운 맥락 설계를 유지한다. 데이터셋의 이미지들은 텍스트 프롬프트와의 정확한 정렬을 보장하기 위해 VQA 체커의 도움으로 생성된다. 또한 구성적 이해를 향상시키기 위해 커리큘럼 대조 학습을 diffusion model에 통합한다.

---

## 2. 해결하고자 하는 문제, 방법론, 모델 구조, 성능 및 한계

### 2.1 🚧 해결하고자 하는 문제

Diffusion model의 인상적인 T2I 합성 능력에도 불구하고, 특히 복잡한 상황에서 객체와 속성 사이의 구성적 관계(compositional relationships)를 이해하는 데 어려움을 겪는다.

기존 모델들은 "Two cats are playing"이라는 프롬프트에서 고양이 세 마리를 생성하는 것과 같은 기본적인 작업에서도 어려움을 겪는다. 단계적 훈련 전략은 모델이 복잡한 사례를 다루기 전에 견고한 기반을 구축하도록 돕는다.

비록 구성적 생성을 위한 많은 데이터셋이 존재하지만, 자연스럽고 합리적인 맥락 내에서 단순한 샘플부터 복잡한 샘플까지의 명확한 진행 과정을 제공하는 데이터셋은 여전히 크게 부족하다.

---

### 2.2 📐 제안하는 방법 (수식 포함)

#### (A) ConPair 데이터셋 구축 파이프라인

ConPair의 각 샘플은 긍정 캡션(positive caption)과 연관된 이미지 쌍으로 구성된다. 캡션은 GPT-4로 구성되며, **색상(color), 형태(shape), 질감(texture), 개수(counting), 공간 관계(spatial), 비공간 관계(non-spatial), 장면(scene), 복잡(complex)** 의 8가지 구성성 카테고리를 포함한다.

각 이미지 쌍은 주어진 프롬프트로부터 생성된 긍정 이미지(positive image)와, 프롬프트와 의미적으로 일치하지 않는 부정 이미지(negative image)로 구성되며, 두 이미지의 차이는 최소화된다.

데이터셋 자동 큐레이션의 핵심 수식은 VQA 점수 기반 필터링이다. 이미지 $I$와 프롬프트 $T$에 대한 VQA 정렬 점수 $\theta$를 기준으로:

$$\theta(I, T) = \frac{1}{|Q|}\sum_{q \in Q} \mathbb{1}[\text{VQA}(I, q) = \text{GT}(q, T)]$$

$Q$는 프롬프트 $T$로부터 생성된 질문 집합, $\text{GT}$는 정답을 나타내며, 임계값 $\theta \geq \theta_{\min}$ 을 만족하는 이미지만 ConPair에 포함된다.

#### (B) EvoGen: 커리큘럼 대조 학습 목적 함수

EvoGen은 긍정 이미지-텍스트 쌍 간의 유사성을 최대화하고 부정 쌍에 대해서는 최소화하는 대조 손실 함수를 통합하여, 구성적 불일치를 구별하도록 모델을 효과적으로 안내한다.

전체 학습 목적 함수는 **Diffusion Score Matching Loss** 와 **Contrastive Loss** 의 결합이다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}} + \lambda \cdot \mathcal{L}_{\text{contrastive}}$$

**Diffusion Loss (표준 denoising score matching):**

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) \right\|^2 \right]$$

여기서 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$, $c$는 텍스트 조건이다.

**Contrastive Loss:**

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(\mathbf{f}^+, \mathbf{c}) / \tau)}{\exp(\text{sim}(\mathbf{f}^+, \mathbf{c}) / \tau) + \exp(\text{sim}(\mathbf{f}^-, \mathbf{c}) / \tau)}$$

여기서 $\mathbf{f}^+$는 긍정 이미지의 latent representation, $\mathbf{f}^-$는 부정 이미지의 latent representation, $\mathbf{c}$는 텍스트 임베딩, $\tau$는 온도(temperature) 파라미터이며, $\lambda$는 두 손실 사이의 균형을 맞추는 계수이다 (실험에서 `--contrastive_loss_coeff 0.1` 사용).

---

### 2.3 🏗️ 모델 구조 (EvoGen 3단계 커리큘럼)

커리큘럼은 세 가지 서브 태스크로 설계된다: (1) 단일 객체-속성 구성 학습, (2) 두 객체 간 속성 결합(attribute binding) 마스터링, (3) 다중 객체가 있는 복잡한 장면 처리.

데이터셋을 세 단계로 나누어 모델이 단순한 구성 작업에서 복잡한 작업으로 점진적으로 발전할 수 있도록 하는 간단하지만 효과적인 다단계 파인튜닝 패러다임을 도입한다. 첫 번째 단계에서 샘플은 특정 속성(예: 형태, 색상, 개수, 질감)이나 특정 동작, 또는 간단한 정적 장면을 가진 단일 객체로 구성된다. 대응하는 부정 이미지와 긍정 이미지 사이의 차이는 명확하고 눈에 띄도록 설계된다.

```
┌─────────────────────────────────────────────┐
│              EvoGen Pipeline                │
├─────────────────────────────────────────────┤
│ Stage I  : 단일 객체 + 단일 속성             │
│            (색상, 형태, 개수, 질감)          │
│            → 명확한 positive/negative 쌍     │
├─────────────────────────────────────────────┤
│ Stage II : 두 객체 간 속성 결합              │
│            (attribute binding)              │
│            → 미묘한 시각적 차이              │
├─────────────────────────────────────────────┤
│ Stage III: 다중 객체 복잡 장면               │
│            (spatial/non-spatial/complex)    │
│            → 최소한의 시각적 불일치          │
└─────────────────────────────────────────────┘
         ↓ (각 단계: Contrastive Fine-tuning)
   Stable Diffusion v2 / SD3-Medium 기반 모델
```

---

### 2.4 📊 성능 향상

EvoGen은 T2I-CompBench의 모든 카테고리에서 최고 성능을 보이며, 특히 'Non-spatial' 및 'Complex' 구성 카테고리에서 현저한 향상을 보인다. EVOGEN-SD3-MEDIUM은 Non-Spatial 카테고리에서 36.95%, Complex 카테고리에서 49.07%를 달성하여, 다음으로 좋은 모델인 DALL-E 3의 28.65%, 37.73%를 각각 상회한다.

DALL-E 2, SD v3, SDXL, PixArt-α와 비교한 사용자 연구에서 참가자들은 텍스트 프롬프트와의 우수한 정렬 능력으로 EVOGEN을 일관되게 선호하였다. 미적 품질은 일부 경우에서 약간 낮게 인식될 수 있지만, 구성적 정확도라는 주요 목표는 EVOGEN에 의해 더 효과적으로 달성된다.

Complex 성능의 현저한 향상은 주로 복잡한 구성 요소를 가진 고품질 대조 샘플을 활용하는 Stage-III 훈련 덕분이다.

#### 성능 비교표 (T2I-CompBench, Non-Spatial / Complex)

| 모델 | Non-Spatial | Complex |
|------|------------|---------|
| DALL-E 3 | 28.65% | 37.73% |
| **EvoGen-SD3-Medium** | **36.95%** | **49.07%** |

---

### 2.5 ⚠️ 한계

대조 손실의 효과를 파인튜닝과 비교 검증한 결과, 대조 손실은 속성 결합(attribute binding) 카테고리에서 성능을 향상시키지만, 객체 관계(object relationships) 및 복잡한 장면에서는 영향이 더 적다. 이는 속성 불일치가 모델에서 감지하기 더 쉬운 반면, 관계 차이는 더 복잡하기 때문으로 추측된다.

미적 품질(aesthetic quality)은 일부 경우에서 다른 모델 대비 약간 낮게 인식될 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 **일반화(generalization)** 는 가장 핵심적인 동기 중 하나이다.

### 3.1 기존 방법의 일반화 한계 지적

기존 접근법들은 구성적 아키텍처를 구축하거나 어려운 부정 캡션을 생성하는 방식이 주로 고정된 사전 정의된 구성 구조를 가정하며, 이는 새로운 분포로의 일반화를 제한한다.

다른 연구들은 각 구성 모델이 독립적인 도메인의 분포를 포착하여 구성적 성능을 향상시키는 구성 생성 모델을 개발한다. 그러나 이러한 접근법들은 모델을 구성하기 위한 고정된 사전 정의된 구조를 가정하므로 새로운 분포로의 일반화가 제한된다.

### 3.2 EvoGen이 일반화를 향상시키는 메커니즘

1. **LLM 기반 자동 데이터 다양성 확장**:
   LLM을 활용하여 구성 프롬프트의 복잡성을 확장하면서 자연스러운 맥락 설계를 유지한다. 이는 특정 도메인에 국한되지 않는 다양하고 자연스러운 시나리오를 학습 데이터로 활용 가능하게 한다.

2. **커리큘럼 학습을 통한 단계적 복잡성 일반화**:
   커리큘럼 훈련은 diffusion model에 구성성에 대한 근본적인 이해를 부여하는 데 핵심적이며, 파인튜닝 중 점진적으로 더 복잡한 구성 시나리오를 도입한다. 이 단계적 훈련 전략은 모델이 복잡한 사례를 다루기 전에 견고한 기반을 구축하도록 돕는다.

3. **VQA 기반 품질 보증**:
   VQA 지원을 통한 이미지 생성 방법이 모든 카테고리에 걸쳐 이미지 충실도를 일관되게 향상시키며, ConPair가 고품질 이미지-텍스트 쌍을 포함한다는 것을 보여준다.

4. **8가지 구성성 카테고리의 포괄적 커버리지**:
   이미지 쌍들은 최소한의 시각적 불일치를 특징으로 하며, 특히 복잡하고 자연스러운 시나리오를 포함한 광범위한 속성 카테고리를 다룬다.

5. **모델 아키텍처 독립적 파인튜닝**:
   EvoGen은 SD v2, SDXL, SD3-Medium 등 다양한 백본에 적용 가능하여, 특정 아키텍처에 종속되지 않는 범용적인 일반화 방법론임을 실증한다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 📈 향후 연구에 미치는 영향

#### (1) 커리큘럼 학습의 생성 모델 표준화 가능성
커리큘럼 훈련이 생성 모델에 구성성에 대한 근본적인 이해를 부여하는 데 핵심적이라는 주장은 T2I 모델뿐 아니라 **텍스트-투-비디오(T2V), 3D 생성, 멀티모달 모델** 전반에 커리큘럼 설계 원리가 적용될 수 있음을 시사한다.

#### (2) 자동화된 데이터 큐레이션 패러다임 확립
LLM을 활용해 현실적이고 복잡한 시나리오를 구성하고, VQA 시스템 및 diffusion model과 함께 활용하여 15k 쌍의 고품질 대조 이미지로 구성된 ConPair 데이터셋을 자동으로 큐레이션하는 방법은 수동 어노테이션 비용 없이 확장 가능한 데이터 구축 방법론의 새로운 기준을 제시한다.

#### (3) 대조 학습과 생성 모델의 통합 연구 촉진
기존 대조 학습은 주로 판별 모델(discriminative model)에 집중되었으나, EvoGen은 이를 생성 모델의 파인튜닝에 효과적으로 결합하는 방법을 보여준다. 이는 **RLHF, DPO 등 인간 피드백 기반 파인튜닝**과의 결합 가능성도 열어준다.

#### (4) 구성성 벤치마크 연구의 가속화
EvoGen이 모델의 구성적 이해를 크게 향상시키고 대부분의 기준 생성 방법을 능가함을 보여주는 광범위한 실험은 향후 연구자들이 T2I-CompBench 등의 벤치마크에서 보다 공정한 비교를 수행할 기준점을 제공한다.

---

### 4.2 🔬 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|----------|----------|
| **미적 품질 vs. 구성적 정확도의 균형** | 미적 품질이 일부 경우에서 다소 낮게 인식될 수 있어, 두 목표를 동시에 최적화하는 multi-objective 학습 전략 연구가 필요하다. |
| **복잡한 관계에 대한 대조 손실 한계 극복** | 대조 손실은 속성 결합에서는 성능을 향상시키지만 객체 관계 및 복잡한 장면에서는 효과가 제한적이며, 이는 관계 차이가 더 복잡하기 때문이므로, 관계 인식 특화 손실 함수 설계가 요구된다. |
| **동적 데이터 난이도 조정 메커니즘** | 혼합 난이도 데이터로 모델을 훈련할 때 모델을 압도하고 최적화에 미치지 못하는 학습을 야기할 수 있다는 점에서, 난이도를 동적으로 조정하는 적응형 커리큘럼 방법론 연구가 중요하다. |
| **비디오 및 3D 도메인 확장** | T2I에서 검증된 커리큘럼 접근법을 T2V, Text-to-3D로 확장 시 시간적/공간적 구성성 문제를 어떻게 정의하고 학습할지 연구가 필요하다. |
| **GPT-4 의존성 문제** | ConPair의 캡션은 GPT-4로 구성되므로, 오픈소스 LLM으로 대체 가능한 파이프라인 설계가 재현성과 접근성 향상을 위해 요구된다. |
| **평가 지표의 다양화** | VQA 점수와 T2I-CompBench 외에 인간 인지 연구 기반의 구성성 평가 지표 개발이 필요하다. |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 방법론 | 핵심 특징 | 일반화 수준 |
|------|--------|----------|------------|
| **Attend-and-Excite** (SIGGRAPH 2023) | Cross-attention 최적화 | 추론 시점 attention map 강화 | 낮음 (inference-time only) |
| **Structured Diffusion Guidance** (ICLR 2023) | 언어 구조 기반 유도 | 구문 트리를 guidance에 활용 | 중간 (구조 고정) |
| **WiCLP** (2024) | CLIP 표현 공간에 선형 투영을 파인튜닝하는 간단하고 강력한 기준선 WiCLP 제안 | 파라미터 효율적, CLIP 개선 | 중간 |
| **MCCD** (2024) | 다중 에이전트 협업 diffusion | 복잡한 T2I 생성에 특화 | 중간 |
| **EvoGen (본 논문)** (ICLR 2025) | 커리큘럼 훈련 + LLM/VQA 기반 자동 데이터셋 ConPair | 3단계 커리큘럼 + 대조 학습 | **높음** (다양한 분포 일반화) |

기존 접근법들이 새로운 분포로의 일반화를 제한하는 고정된 사전 정의된 구성 구조를 가정하는 것과 달리, EvoGen은 커리큘럼 기반 점진적 학습으로 이 한계를 극복하는 차별화된 위치를 점한다.

---

## 📚 참고 자료 및 출처

1. **arXiv 논문 원문**: [arXiv:2410.16719](https://arxiv.org/abs/2410.16719) — *Progressive Compositionality in Text-to-Image Generative Models*, Evans Xu Han et al.
2. **ICLR 2025 OpenReview**: [openreview.net/forum?id=S85PP4xjFD](https://openreview.net/forum?id=S85PP4xjFD)
3. **ICLR 2025 Poster 페이지**: [iclr.cc/virtual/2025/poster/29626](https://iclr.cc/virtual/2025/poster/29626)
4. **HTML 논문 전문**: [arxiv.org/html/2410.16719](https://arxiv.org/html/2410.16719)
5. **GitHub 코드 저장소**: [github.com/evansh666/EvoGen](https://github.com/evansh666/EvoGen)
6. **Liner Quick Review**: [liner.com/review/progressive-compositionality-in-texttoimage-generative-models](https://liner.com/review/progressive-compositionality-in-texttoimage-generative-models)
7. **Paper Reading Club 요약**: [paperreading.club/page?id=261380](http://paperreading.club/page?id=261380)
8. **비교 연구 - WiCLP**: [arXiv:2406.07844](https://arxiv.org/html/2406.07844) — *Improving Compositional Attribute Binding in T2I Generative Models via Enhanced Text Embeddings*

> ⚠️ **정확도 주의사항**: 본 답변에서 수식의 구체적 구현 세부사항(예: 온도 파라미터 값, contrastive loss의 정확한 형태)은 GitHub 코드 공개 정보 및 논문 HTML에서 일부 추론한 부분이 있습니다. 정확한 수식 전체는 논문 원문 PDF를 직접 참조하시기를 권장합니다.
