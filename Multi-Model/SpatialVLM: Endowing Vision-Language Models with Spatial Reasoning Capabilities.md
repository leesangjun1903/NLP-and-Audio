
# SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities

> **논문 정보:**
> Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh, Leonidas Guibas, Fei Xia
> **발표:** CVPR 2024, pp. 14455–14465
> **arXiv:** [2401.12168](https://arxiv.org/abs/2401.12168) | **프로젝트 사이트:** [spatial-vlm.github.io](https://spatial-vlm.github.io/)

---

## 1. 핵심 주장 및 주요 기여 (Executive Summary)

### 🔑 핵심 가설

VLM의 제한된 공간 추론 능력은 모델 아키텍처의 근본적인 한계 때문이 아니라, 학습 데이터에 공간 추론 데이터가 부족하기 때문이라는 것이 핵심 가설입니다.

### 📌 주요 기여 요약

| 기여 | 설명 |
|------|------|
| 데이터 생성 파이프라인 | 인터넷 스케일 3D Spatial VQA 데이터 자동 생성 |
| 최초 메트릭 공간 데이터셋 | 인터넷 규모의 최초 3D 공간 추론 데이터셋 |
| 학습 레시피 연구 | 데이터 품질, 파이프라인, 아키텍처 요소 분석 |
| 하위 응용 | Chain-of-Thought 공간 추론 및 로보틱스 보상 주석 |

먼저 1,000만 장의 실세계 이미지에서 20억 개의 VQA 예시로 확장되는 자동 3D 공간 VQA 데이터 생성 프레임워크를 개발하고, 데이터 품질, 학습 파이프라인, VLM 아키텍처를 포함한 다양한 학습 레시피 요소를 탐구했습니다.

---

## 2. 해결하고자 하는 문제

### 🚨 기존 VLM의 한계

VLM은 특정 VQA 벤치마크에서 뛰어난 성능을 보이지만, 물리적 객체 간의 거리나 크기 차이 같은 정량적 관계 인식과 같은 3D 공간 추론 능력이 부족합니다.

파운데이션 모델의 탐구는 종종 인간 능력에서 영감을 받습니다. 인간은 체화된 경험과 진화적 발달을 통해 타고난 공간 추론 능력을 가지고 있습니다. 우리는 복잡한 사고 과정 없이도 물체 간의 위치 관계나 거리·크기 추정 같은 공간적 관계를 자연스럽게 파악합니다. 이러한 직접적 공간 추론에서의 자연스러운 능력과 현재 VLM의 한계 사이의 대조는, VLM이 여러 단계의 공간 추론을 필요로 하는 실세계 과제를 수행하는 것을 방해합니다.

---

## 3. 제안 방법 (데이터 파이프라인 + 수식)

### 3.1 데이터 합성 파이프라인

구체적으로, 1) 개방 어휘 탐지(open-vocabulary detection), 2) 메트릭 깊이 추정(metric depth estimation), 3) 의미론적 분할(semantic segmentation), 4) 객체 중심 캡셔닝 모델을 결합하여 실세계 데이터를 대규모로 밀집하게 주석 처리합니다.

파이프라인은 다음 5단계로 구성됩니다:

**① 시맨틱 필터링 (Semantic Filtering)**

CLIP을 사용하여 잡음이 많은 인터넷 이미지를 필터링하고 장면 수준의 사진만 유지합니다.

**② 전문가 모델 적용 (Expert Model Annotation)**

인터넷 규모의 이미지에 사전 학습된 전문가 모델을 적용하여 객체 중심 분할(segmentation), 깊이(depth) 및 캡션을 획득합니다.

**③ 2D → 3D 포인트 클라우드 변환**

2D 이미지를 3D 포인트 클라우드로 변환하여 3D 바운딩 박스 같은 유용한 속성을 추출하는 형상 분석 규칙으로 파싱합니다.

이 과정에서 깊이 추정 모델을 이용한 **메트릭 좌표계 변환**이 핵심입니다:

$$
\mathbf{P}_{3D} = \mathbf{D}(u, v) \cdot \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

여기서:
- $\mathbf{P}_{3D}$: 3D 포인트 클라우드 좌표
- $\mathbf{D}(u, v)$: 픽셀 $(u, v)$에서의 메트릭 깊이 값
- $\mathbf{K}$: 카메라 내부 파라미터 행렬 (Intrinsic Matrix)

**카메라 좌표계 → 측지 좌표계 변환:**

단안(monocular) 2D 픽셀을 메트릭 스케일의 3D 포인트 클라우드로 변환하기 위해 깊이 추정을 수행합니다. 그 후 포인트 클라우드의 카메라 좌표계를 측지 좌표계(geodetic coordinate system)로 정규화하는데, 이는 수평 표면(예: "바닥", "테이블 상단") 분할 및 프레임 변환을 통해 이루어집니다.

회전 변환 행렬:

$$
\mathbf{P}_{geo} = \mathbf{R} \cdot \mathbf{P}_{cam}
$$

**④ 중의성 해소 (Ambiguity Resolution)**

CLIP 유사도 점수를 사용하여 객체 캡션을 클러스터링함으로써 모호한 질문을 피합니다.

**두 객체 간 3D 거리 계산:**

$$
d = \sqrt{(\Delta x)^2 + (\Delta y)^2 + (\Delta z)^2}
$$

여기서 $\Delta x, \Delta y, \Delta z$는 측지 좌표계로 변환된 두 객체 중심 간의 좌표 차이입니다.

**⑤ VQA 데이터 생성**

질문 유형별 템플릿 기반 QA 생성:

| 질문 유형 | 예시 |
|---|---|
| 이진 술어 (Binary Predicate) | "Is A to the left of B?" |
| 정량적 추정 (Quantitative) | "How far is A from B?" → "It's [Distance]." |

---

### 3.2 모델 학습 전략

저자들은 PaLM-E 학습 세트와 공간 VQA 데이터셋을 혼합하여 모델을 학습시킵니다.

**혼합 학습 손실 (Mixed Training Loss):**

$$
\mathcal{L}_{total} = \mathcal{L}_{LM} + \lambda \cdot \mathcal{L}_{spatial}
$$

여기서:
- $\mathcal{L}_{LM}$: 일반 언어 모델 손실 (기존 VQA 태스크)
- $\mathcal{L}_{spatial}$: 공간 추론 VQA에 대한 언어 모델 손실
- $\lambda$: 공간 데이터 비중 조절 하이퍼파라미터

모델은 질문에 대해 자연어로 답변을 생성하며, 정량적 거리는 다음과 같은 형태로 출력됩니다:

$$
\hat{d} = f_{\theta}(\mathbf{I}, Q)
$$

여기서 $\mathbf{I}$는 입력 이미지, $Q$는 공간 관련 질문, $f_\theta$는 파인튜닝된 VLM입니다.

---

## 4. 모델 구조

### 4.1 베이스 모델

저자들은 PaLI-X 55B 등 대형 VLM을 베이스로 사용합니다.

### 4.2 학습 설정 비교 (Ablation)

| 설정 | 질적 VQA | 정량적 VQA |
|---|---|---|
| ViT 고정 (Frozen) | 향상 | 제한적 |
| ViT 비고정 (Unfrozen) | 최고 성능 | 가능 |

이미지 인코더를 비고정(unfrozen)으로 파인튜닝하면, SpatialVLM은 객체 간의 수평 거리와 같은 정량적 공간 추정 질문에 답변할 수 있습니다.

### 4.3 Chain-of-Thought 공간 추론 구조

SpatialVLM은 기반 개념을 쿼리할 수 있는 자연어 인터페이스를 제공하며, 강력한 LLM과 결합하면 복잡한 공간 추론을 수행할 수 있습니다. 이 방법을 "Chain-of-Thought Spatial Reasoning"이라 부릅니다. 합성 데이터에는 직접적인 공간 추론 질문만 포함되어 있지만, VLM이 이를 조합하여 다단계 연쇄 추론이 필요한 복잡한 질문을 해결하기 쉽습니다.

```
복잡한 질문 (예: "세 물체가 정삼각형을 형성하는가?")
        ↓
   [LLM 조율자]
        ↓
단순 질문 분해 → SpatialVLM 쿼리 → 결과 통합 → 최종 답변
```

---

## 5. 성능 향상

### 5.1 정성적(Qualitative) VQA

SpatialVLM은 합성 공간 VQA 데이터로 학습되지 않은 모든 기준 모델들에 비해 현저히 높은 정확도를 달성하며, GPT-4V를 포함한 다른 비전-언어 모델들을 능가합니다.

### 5.2 정량적(Quantitative) 거리 추정

SpatialVLM은 정량적 거리 추정에서 인간이 주석 처리한 정답의 0.5배~2배 범위 내에 답변이 포함되는 비율이 37.2%로, 기준 모델들보다 더 자주 정답에 근접한 거리 추정값을 출력합니다.

정량적 추정 태스크에서, 본 방법은 기준 모델들보다 더 빈번하게(99.0%) 올바른 형식으로 출력합니다.

### 5.3 일반 VQA 성능 유지

공간 VQA 데이터와의 공동 학습으로 인해 다른 태스크 성능이 저하되는지 확인하기 위해, 공간 VQA 데이터 없이 학습된 바닐라 PaLM 2-E와 비교한 결과, 제한적인 공간 추론 질문이 포함된 OKVQA 벤치마크에서 유사한 성능을, 공간 추론 질문이 포함된 VQA-v2 test-dev 벤치마크에서는 약간 더 나은 성능을 보였습니다.

### 5.4 로보틱스 응용

실세계 단위로 공간을 직관적으로 정량적으로 추론하는 능력 덕분에, SpatialVLM은 로보틱스 태스크의 세밀한 보상 주석자(reward annotator)로 활용할 수 있습니다. SpatialVLM은 로봇 팔이 콜라 캔에 접근하는 과정에서 단조롭게 감소하는 거리 추정을 올바르게 제공하며, 이를 강화학습의 보상 신호로 사용할 수 있습니다. SpatialVLM은 이진 성공/실패 레이블만 주석 처리할 수 있는 기존 방법들과 달리, 오픈 어휘 로보틱스 태스크에 대한 밀집 보상(dense reward)을 주석 처리할 수 있습니다.

---

## 6. 모델의 한계

SpatialVLM은 중간 거리 범위에서는 잘 작동하지만, 모델은 전문가 비전 모델로부터 데이터 합성 파이프라인의 편향과 한계를 물려받습니다.

데이터 구성 파이프라인이 탐지, 분할, 메트릭 깊이 추정, 카메라 보정 등 특정 모델에 의존하기 때문에, 이러한 모델 주도적 속성은 데이터셋 내 정량적 레이블에 체계적 오류를 도입할 수 있습니다.

추가적인 한계점:

- **템플릿 기반 한계:** 직접 공간 쿼리는 유한한 템플릿 집합에 기반하지만, SpatialVLM은 공간 추론 구성 요소를 필요로 하는 더 복잡한 연쇄 사고 추론으로 확장될 수 있습니다.
- **중의성 문제:** SpatialVLM에서 발견된 중의성 문제로, 이미지 내의 유사한 여러 객체들이 캡션 레이블에 혼동을 줄 수 있습니다.

---

## 7. 일반화 성능 향상 가능성

### 7.1 파인튜닝을 통한 일반성 계승

이 연구에서 생성된 VQA 데이터셋에 VLM을 파인튜닝하는 방식으로 공간 추론 문제에 접근함으로써, 기반 VLM의 일반성과 추론 능력을 계승합니다.

### 7.2 Chain-of-Thought로의 확장

직접 공간 추론 능력을 갖춘 SpatialVLM이 LLM과 대화하도록 하여 Chain-of-Thought 공간 추론을 수행할 수 있습니다. 직접 추론 능력과 연쇄 사고 추론을 결합하면 여러 단계의 질문에 답할 수 있습니다.

### 7.3 일반 VQA 태스크 언더피팅 시사점

이 결과는 VLM이 일반적으로 공간 추론에 가까운 태스크 분포에서 언더피팅(underfitting)되어 있으며, 이 영역에서 추가 학습을 통해 혜택을 받을 수 있음을 시사합니다.

### 7.4 인터넷 이미지 기반 대규모 일반화

이는 더 복잡한 공간 추론 질문과 답변을 생성하여 통합된 멀티모달 대형 언어 모델을 학습시킬 미래의 기회를 열어줍니다. 2D 이미지를 메트릭 스케일 3D 포인트 클라우드로 변환하는 자동 3D 공간 VQA 데이터 생성 프레임워크를 개발하고, 1,000만 장의 실세계 이미지에 대해 20억 개의 VQA 예시로 데이터 파이프라인을 확장하였습니다.

---

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 연구 흐름 타임라인

```
2022 → Semantic Abstraction (Ha & Song) : 오픈 월드 3D 장면 이해 기초
2023 → 3D-LLM (Hong et al., NeurIPS) : LLM에 3D 세계 주입
2024.01 → SpatialVLM (Chen et al., CVPR) : 인터넷 스케일 공간 VQA
2024.06 → SpatialBot (Cai et al.) : RGB-D 통합 공간 추론
2024.06 → SpatialRGPT (Cheng et al., NeurIPS) : 3D 장면 그래프 + 깊이 통합
2025 → RoboSpatial (Song et al., CVPR) : 로보틱스 특화 공간 이해
2025 → DepthLM : VLM에서의 메트릭 깊이 추정
```

### 8.2 주요 논문 상세 비교

#### 🔹 SpatialRGPT (NeurIPS 2024)

SpatialRGPT는 3D 장면 그래프에서 지역 표현의 효과적인 학습을 가능하게 하는 데이터 큐레이션 파이프라인과 기존 VLM의 시각 인코더에 깊이 정보를 통합하는 유연한 "플러그인" 모듈이라는 두 가지 핵심 혁신을 통해 VLM의 공간 이해를 발전시킵니다. 추론 시 사용자가 지정한 지역 제안(region proposals)이 주어지면, SpatialRGPT는 상대적인 방향과 거리를 정확하게 인식할 수 있습니다.

이 결과들은 VLM이 일반적으로 공간 추론 태스크에서 성능이 낮지만, 일반 VQA 성능을 저해하지 않으면서 특정 공간 VQA 학습을 통해 향상될 수 있다는 SpatialVLM의 발견과 일치합니다.

#### 🔹 SpatialBot (ICRA 2025)

SpatialBot은 RGB와 깊이 이미지를 모두 통합하여 공간 이해를 크게 향상시키는 Vision Language Model입니다. SpatialBot은 새로운 SpatialQA 데이터셋을 사용하여 학습되며, 이 데이터셋은 깊이 데이터로 모델의 추론 능력을 향상시키기 위한 다단계 깊이 관련 질문을 포함합니다.

#### 🔹 DepthLM (2025)

텍스트 기반 지도 파인튜닝(SFT)과 희소 레이블이 VLM의 강력한 3D 이해를 열어주기에 충분하며, 밀집 예측 헤드나 복잡한 회귀/정규화 손실 없이도 가능하다는 것을 보여줍니다.

### 8.3 방법론 비교표

| 논문 | 연도 | 데이터 | 아키텍처 특징 | 핵심 차이점 |
|------|------|--------|------------|------------|
| **SpatialVLM** | CVPR 2024 | 2B VQA (10M 이미지) | PaLI-X 파인튜닝 | 최초 인터넷 스케일 메트릭 3D VQA |
| **SpatialRGPT** | NeurIPS 2024 | 3D 장면 그래프 | 깊이 플러그인 모듈 + 지역 표현 | Region-level 쿼리 지원 |
| **SpatialBot** | ICRA 2025 | RGB-D 쌍 | RGB+Depth 인코더 | RGB-D 직접 입력 |
| **DepthLM** | 2025 | 다양한 깊이 데이터셋 | 표준 VLM + 프롬프트 설계 | 카메라 내부 파라미터 모호성 해결 |

---

## 9. 향후 연구에 미치는 영향 및 고려 사항

### 9.1 향후 연구에 미치는 영향

**① 데이터 중심(Data-Centric) 패러다임 확립**

오늘날 VLM의 공간 추론 능력 부재의 이유는 아키텍처가 아니라 공간 추론 학습 데이터의 부족이라는 가설을 제시하며, 이 인사이트를 따라 공간 추론 질문을 포함하는 VQA 데이터를 생성하는 파이프라인을 설계합니다. 이는 이후 연구들이 공간 데이터 생성에 집중하는 계기가 되었습니다.

**② 로보틱스와 VLM의 브릿지**

인간처럼 직접 공간 추론을 수행하는 능력을 활용하여, SpatialVLM이 LLM과 대화하도록 하여 Chain-of-Thought 공간 추론을 수행할 수 있으며, 직접 추론 능력과 연쇄 사고 추론을 결합하면 여러 단계의 질문에 답할 수 있습니다.

**③ 새로운 벤치마크 및 평가 체계의 촉발**

SpatialVLM, SpatialRGPT-Bench, Spatial-MM 등은 관계적 이해를 위한 인간 주석 QA를 도입하면서 다양한 벤치마크가 파생되었습니다.

**④ 오픈소스 생태계 기여**

논문 공개 후, VLM 연구 커뮤니티로부터 열정적인 반응을 얻었으며, remyxai라는 사용자가 데이터 합성 파이프라인의 오픈소스 구현을 제공하였습니다.

### 9.2 향후 연구 시 고려할 점

#### ⚠️ 1. 메트릭 깊이 추정의 카메라 모호성 해결

메트릭 깊이 혼합 데이터 학습에서 중요한 문제는 카메라 모호성(camera ambiguity)이며, 서로 다른 카메라로 촬영된 유사한 이미지들은 극단적으로 다른 스케일을 가질 수 있습니다. 따라서 카메라 내부 파라미터를 학습에 통합하는 방법론이 필요합니다.

#### ⚠️ 2. 파이프라인 노이즈로 인한 레이블 오류 최소화

데이터 구성 파이프라인이 탐지, 분할, 메트릭 깊이 추정, 카메라 보정 등의 특정 모델에 의존하며, 이러한 모델 주도적 속성은 데이터셋 내 정량적 레이블에 체계적 오류를 도입할 수 있습니다.

#### ⚠️ 3. Region-Level 공간 이해 강화

대부분의 VLM은 전체 이미지 파서로 기능하며, 사용자가 공간적 관계를 이해하고자 하는 특정 영역을 지정하는 것을 지원하지 않습니다. 또한 방향이나 거리 같은 공간 관계를 정확하게 인식하는 것은 RGB 픽셀 데이터만으로는 불가능합니다.

#### ⚠️ 4. 다양한 환경(실내/실외/에고센트릭)으로의 확장

소형 모델(3B, 8B)은 다중 뷰 3D 추론에 대한 제한된 용량을 보여 우연 수준에 가까운 성능을 나타냅니다. 반면 대형 모델들은 우연 수준 대비 실질적인 향상을 보이지만, 여전히 인간 수준과 비교했을 때 눈에 띄는 격차가 있습니다.

#### ⚠️ 5. 강화학습(RL) 활용 가능성

강화학습(RL)도 3D 이해 학습에 활용할 수 있으며, SFT가 더 효율적이지만 두 방법 모두 3D 이해를 학습할 수 있습니다.

#### ⚠️ 6. 기하학적 프리미티브 확장

추가적인 더 미묘한 기하학적 프리미티브(geometric primitives)에 대한 연구가 필요합니다. 즉, 현재는 거리/방향 위주의 질문 템플릿에 집중되어 있으나, 더 복잡한 기하학적 관계(각도, 부피, 상대적 자세 등)로의 확장이 요구됩니다.

---

## 📚 참고 자료 (References)

| # | 제목 | 출처 |
|---|------|------|
| 1 | **SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities** | arXiv:2401.12168 / CVPR 2024, pp.14455–14465 |
| 2 | SpatialVLM 프로젝트 웹사이트 | https://spatial-vlm.github.io/ |
| 3 | SpatialVLM CVPR Open Access PDF | https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_SpatialVLM... |
| 4 | SpatialVLM IEEE Xplore | https://ieeexplore.ieee.org/document/10658310/ |
| 5 | SpatialVLM HuggingFace Papers | https://huggingface.co/papers/2401.12168 |
| 6 | SpatialVLM Semantic Scholar | https://www.semanticscholar.org/paper/SpatialVLM... |
| 7 | **SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models** | NeurIPS 2024 / OpenReview / NeurIPS Proceedings PDF |
| 8 | **SpatialBot: Precise Spatial Understanding with Vision Language Models** | ICRA 2025 / arXiv:2406.13642 |
| 9 | **DepthLM: Metric Depth From Vision Language Models** | arXiv:2509.25413 |
| 10 | **SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models** | arXiv:2509.17664 |
| 11 | **EarthSpatialBench: Benchmarking Spatial Reasoning Capabilities of Multimodal LLMs on Earth Imagery** | arXiv:2602.15918 |
| 12 | Awesome-Spatial-Intelligence-in-VLM (Paper List) | https://github.com/mll-lab-nu/Awesome-Spatial-Intelligence-in-VLM |
| 13 | **RoboSpatial: Teaching spatial understanding to 2D and 3D VLMs for robotics** | CVPR 2025 |
