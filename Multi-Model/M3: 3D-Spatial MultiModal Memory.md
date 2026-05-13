
# M3: 3D-Spatial MultiModal Memory

> **📌 논문 정보**
> - **제목**: M3: 3D-Spatial MultiModal Memory
> - **저자**: Xueyan Zou, Yuchen Song, Ri-Zhao Qiu, Xuanbin Peng, Jianglong Ye, Sifei Liu, Xiaolong Wang
> - **소속**: UC San Diego, NVIDIA
> - **발표**: ICLR 2025 (Conference Paper)
> - **arXiv**: [2503.16413](https://arxiv.org/abs/2503.16413)
> - **공식 코드**: [GitHub - MaureenZOU/m3-spatial](https://github.com/MaureenZOU/m3-spatial)
> - **프로젝트 페이지**: https://m3-spatial-memory.github.io/

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

M3는 비디오 소스를 통해 중간 규모의 정적 장면(medium-sized static scenes)에 대한 정보를 유지하도록 설계된 멀티모달 메모리 시스템으로, 시각적 인식(visual perception)을 위해 설계되었다.

3D Gaussian Splatting 기술과 파운데이션 모델(foundation models)을 통합함으로써, M3는 다양한 세분화 수준(granularities)에서 feature 표현을 렌더링할 수 있는 멀티모달 메모리를 구축하며, 광범위한 지식을 포괄한다.

특히, M3는 **3D feature distillation에서의 핵심 압축(compression) 문제를 해결한 최초의 연구**임을 주장한다.

### 1.2 주요 기여 (Key Contributions)

| 기여 | 설명 |
|------|------|
| **압축 문제 해결** | 3D feature distillation의 정보 병목(information bottleneck) 최초 해결 |
| **PSC (Principal Scene Components)** | 고차원 2D feature를 메모리 뱅크에 압축 저장 |
| **Gaussian Memory Attention** | PSC를 활용한 고해상도 feature 렌더링 메커니즘 |
| **다양한 모달리티 지원** | VLM, LMM, LLM, 인식 모델 등 여러 파운데이션 모델과 통합 |
| **실세계 적용** | 사족보행 로봇에 실내 장면 feature field 배포 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

이전 feature splatting 연구(F-3DGS, F-Splat 등)는 파운데이션 모델에서 얻은 2D feature map을 미분 가능한 렌더링(differentiable rendering)을 통해 3D Gaussian으로 직접 증류(distill)하였다. 이 과정에서 두 가지 핵심 문제가 발생한다:
1. **정보 병목(Information Bottleneck)**: 계산 제약으로 인해 Gaussian primitive 내의 feature 벡터 차원이 원래 2D feature map(일반적으로 1024)에 비해 크게 감소(16~64 수준)된다.
2. **정렬 불일치(Misalignment)**: 원래 feature map은 본질적으로 3D 일관성이 없을 수 있으며, Gaussian에 3D 일관성을 강제하면 원본 feature와 증류된 feature 사이에 정렬 오류가 발생한다. 결과적으로, 증류된 feature가 파운데이션 모델에 내재된 지식을 정확하게 포착하지 못할 수 있다.

또한, 현재 AI 시스템은 인간이 자연스럽게 수행하는 것, 즉 이동하는 공간에 대한 정신적 지도를 구축하고 어떤 객체가 어디에 있는지 이해하는 것에 어려움을 겪는다. 전통적인 접근 방식은 2D 표현(사진 등)을 사용하여 깊이 차원을 잃어, AI가 환경을 진정으로 이해하기 어렵게 만든다.

---

### 2.2 제안하는 방법 (수식 포함)

M3는 Gaussian splatting과 멀티모달 파운데이션 모델의 더 나은 통합을 통해, Gaussian 구조 내에 표현력 있는 멀티모달 메모리를 효율적으로 저장하고 공간 쿼리(spatial queries)를 용이하게 한다. 구체적으로, **원본 고차원 2D feature map을 "Principal Scene Components(PSC)"라는 메모리 뱅크에 저장하고, 3D Gaussian의 저차원 principal query를 인덱스로 사용**한다. 2D feature를 3D 임베딩에 직접 증류하는 대신, PSC와 principal query 사이에 **Gaussian Memory Attention**을 적용하여 3D 장면에서 파운데이션 모델 임베딩을 렌더링한다. 이를 통해 파운데이션 모델의 높은 표현력을 보존하면서 3D 일관성 있는 저차원 Gaussian 구조를 유지한다.

#### 핵심 개념 정의

논문에서 사용되는 주요 표기는 다음과 같다:
- **Principal Query ($\mathbf{Q}_p$)**: Gaussian primitive 내의 변수로, 저랭크(low-rank) 임베딩을 인코딩하도록 설계됨
- **Scene ($\mathbf{V}^*$)**: 3D 장면의 단일 시점(perspective) 또는 투영(projection)
- **Foundation Models ($\mathbf{F}$)**: 뷰(view)를 구조화된 지식 공간(knowledge space)으로 매핑하는 대형 비전-언어 모델
- **Embeddings ($\mathbf{E}$)**: 파운데이션 모델이 생성한 출력값, 지식 공간에서 데이터를 표현
- **Raw Features ($\mathbf{R}$)**: 파운데이션 모델이 생성한 전체 지식 공간을 포괄하는 임베딩 집합
- **Rendered Feature ($\hat{\mathbf{R}}$)**: Gaussian Memory Attention을 통해 Gaussian splatting으로 계산된 feature

#### 핵심 수식 구성 (개념 기반 수식 정리)

M3는 파운데이션 모델에서 추출한 feature를 각 장면의 **Principal Scene Components ($\mathbf{PSC}$)** 로 압축하고, Gaussian Splatting 파라미터인 **Principal Query ($\mathbf{Q}_p$)** 를 통해 장면을 탐색(probe)하도록 학습한다.

**① PSC 구성 (PCA 기반 압축)**

고차원 Raw Feature $\mathbf{R} \in \mathbb{R}^{H \times W \times D}$ ($D \approx 1024$)를 PCA를 통해 저차원 PSC로 압축:

$$\mathbf{PSC} = \text{PCA}(\mathbf{R}) \in \mathbb{R}^{K \times D}, \quad K \ll H \times W$$

여기서 $K$는 주성분(principal component)의 수이며, 장면의 핵심 의미 정보를 대표하는 $K$개의 기저 벡터로 구성된다.

**② Principal Query 렌더링 (Gaussian Splatting)**

각 Gaussian primitive $g_i$는 저차원 principal query $\mathbf{q}_i \in \mathbb{R}^d$ ($d \ll D$)를 학습하며, 뷰 $\mathbf{V}^*$에 대한 렌더링:

$$\hat{\mathbf{Q}}_p(\mathbf{V}^*) = \sum_{i} \alpha_i(\mathbf{V}^*) \cdot \mathbf{q}_i$$

여기서 $\alpha_i(\mathbf{V}^*)$는 Gaussian Splatting의 미분 가능한 알파 합성(alpha compositing) 가중치이다.

**③ Gaussian Memory Attention (핵심 메커니즘)**

Gaussian Memory Attention은 단일 뷰의 principal query로부터 raw feature를 렌더링하는 절차이다.

렌더링된 principal query $\hat{\mathbf{Q}}_p$를 쿼리(Query)로, PSC를 키(Key)와 밸류(Value)로 사용하는 어텐션:

```math
\hat{\mathbf{R}}(\mathbf{V}^*) = \text{Softmax}\!\left(\frac{\hat{\mathbf{Q}}_p(\mathbf{V}^*) \cdot \mathbf{PSC}^\top}{\sqrt{d}}\right) \cdot \mathbf{PSC}
```

이를 통해 3D Gaussian의 공간 구조를 유지하면서, 파운데이션 모델의 원본 고차원 feature ($D \approx 1024$)를 **정보 손실 없이** 재구성한다.

**④ 학습 목적 함수 (Training Objective)**

$$\mathcal{L} = \mathcal{L}_{\text{render}} + \lambda \mathcal{L}_{\text{feat}}$$

```math
\mathcal{L}_{\text{feat}} = \left\| \hat{\mathbf{R}}(\mathbf{V}^*) - \mathbf{R}(\mathbf{V}^*) \right\|_2^2
```

여기서 $\mathcal{L}\_{\text{render}}$는 RGB 렌더링 손실이고, $\mathcal{L}_{\text{feat}}$는 렌더링된 feature와 파운데이션 모델의 원본 feature 사이의 정렬 손실이다.

---

### 2.3 모델 구조

M3는 다양한 파운데이션 모델을 활용하여 다중 세분화 수준의 장면 지식을 추출하고, 3D Gaussian splatting을 통해 공간 구조를 표현한다. 이러한 기술을 결합함으로써 검색(retrieval), 캡션(captioning), 그라운딩(grounding) 등 다운스트림 애플리케이션을 가능하게 하는 공간적 멀티모달 메모리(M3)를 구성한다.

Gaussian splatting은 Gaussian primitive로 표현되는 가장 세밀한 단위로 장면 구조를 구축하는 프레임워크 역할을 하며, 파운데이션 모델은 장면 지식을 위해 다양한 규모에 걸친 광범위한 세계 지식을 제공한다.

**지원 파운데이션 모델 목록:**
CLIP, SigLIP, DINOv2, LLaMA3, LLaMAv (visual), SEEM 등 다양한 파운데이션 모델의 feature 추출을 지원한다.

```
[입력] 비디오/멀티뷰 이미지
      ↓
[파운데이션 모델 Feature 추출] CLIP / DINOv2 / LLaMA / SEEM ...
      ↓
[PSC 구성] PCA 기반 K개의 주성분으로 고차원 feature 압축
      ↓
[3D Gaussian Splatting 학습] Principal Query(Qp) 학습
      ↓
[Gaussian Memory Attention] Qp ↔ PSC 어텐션 → 고해상도 feature 렌더링
      ↓
[다운스트림 태스크] 검색 / 캡셔닝 / 그라운딩 / 로봇 조작
```

M3는 Gaussian splatting과 파운데이션 모델을 통합하여 Gaussian 구조 내에 멀티모달 메모리를 효율적으로 저장하며, 렌더링된 feature map은 파운데이션 모델의 강력한 표현 능력을 보존하며 높은 충실도(high fidelity)를 나타낸다.

---

### 2.4 성능 향상

M3의 3D 공간 메모리 접근 방식은 이전 방법들보다 크게 뛰어난 성능을 보였으며: Visual Language Navigation 태스크에서 92.1% 성공률(이전 최고 방법: 88.3%)을 달성하고, 훨씬 낮은 레이턴시(45ms vs. 5000ms+)로 동일한 정확도를 유지하며, 3D Gaussian Splatting을 사용하여 전통적인 메시 기반 표현보다 37배 빠른 뷰 렌더링이 가능하다.

M3는 공간적 멀티모달 메모리를 인간 기억과 유사하게 생성하는 새로운 접근 방식을 도입하며, 감소된 훈련 비용으로 우수한 다운스트림 태스크 정확도를 입증하고, 실제 로봇에 배포 시 실용적 유용성을 보여준다.

특히, feature 유사도와 다운스트림 태스크에 대한 포괄적인 정량적 평가와 Gaussian Memory Attention의 픽셀 추적(pixel trace)을 강조하는 정성적 시각화를 통해 M3를 검증한다.

---

### 2.5 한계점

현재 구현은 Gaussian Splatting을 이용한 사전 구축된 3D 장면에 의존하며, 상당한 데이터 처리가 필요하다. 실시간 장면 재구성은 특히 객체가 이동하거나 변화하는 **동적 환경에서 여전히 어렵다.**

시스템의 성능은 기반 파운데이션 모델의 품질에 크게 의존한다. CLIP이 특정 객체나 개념을 인식하지 못하면 이 한계가 M3로 이어지며, 이는 **과소 표현된 카테고리(underrepresented categories)의 객체에 대한 편향이나 인식 공백**으로 이어질 수 있다.

메모리 효율성이 또 다른 우려 사항이다. 논문에서 메모리 계산 확장성 향상이 다루어지고 있지만, 진정한 대규모 환경은 실시간 성능을 유지하기 위한 **상당한 추가 최적화가 필요할 수 있다.** 특히 논문은 도시 규모의 환경이나 다중 건물 단지로의 확장 방법을 완전히 다루지 않는다.

또한 후속 연구(LatentAM)에 따르면, M3는 대규모 장면에서 **PSC 딕셔너리의 무한 증가로 인한 메모리 부족(Out of Memory) 문제**가 발생할 수 있다.

저자들은 최적화된 메모리 뱅크에서 직접 작동할 수 있는 **추론 모듈(reasoning module)** 설계가 흥미로운 미래 방향임을 제시하며 이를 향후 연구 과제로 남겨 두었다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 멀티 파운데이션 모델 통합의 일반화 기여

M3의 접근 방식은 VLM(비전-언어 모델), 인식 모델, LMM/LLM(대형 멀티모달 및 언어 모델)을 포함한 **다양한 파운데이션 모델**을 포괄한다.

이는 일반화 관점에서 매우 중요하다. 특정 단일 모델에 종속되지 않고, 새로운 파운데이션 모델이 등장할 때마다 PSC를 재구성하여 최신 표현 능력을 흡수할 수 있는 **플러그인(plug-in) 방식의 일반화**가 가능하다.

### 3.2 다중 세분화 수준(Multi-Granularity) 표현의 일반화

M3는 다양한 파운데이션 모델을 활용하여 다중 세분화 수준의 장면 지식을 추출하고 3D Gaussian splatting으로 공간 구조를 표현함으로써, 검색(retrieval), 캡션(captioning), 그라운딩(grounding)과 같은 다운스트림 응용을 가능하게 하는 공간적 멀티모달 메모리를 구성한다.

단일 메모리 표현으로 다양한 태스크를 수행할 수 있다는 점은 **태스크 일반화(task generalization)** 측면에서 핵심적이다.

### 3.3 3D 공간 구조의 뷰 독립적 일반화

Gaussian splatting과 파운데이션 모델의 유기적 통합은 장면 구조에 다중 세분화 수준의 지식을 주입하여, 정확한 공간 정보를 갖춘 장면의 **풀 스택(full-stack) 멀티모달 메모리** 구축을 가능하게 한다.

3D Gaussian 구조는 새로운 시점(novel view)에서도 일관된 feature를 렌더링할 수 있어, 훈련 시 보지 못한 시점에서도 안정적인 feature를 생성하는 **뷰 일반화(view generalization)** 를 제공한다.

### 3.4 PSC의 도메인 적응 가능성

LatentAM 등 후속 연구에서는 M3에서 처음 제안된 딕셔너리 기반 전략을 기반으로 하며, 이 접근 방식이 특정 VLM 공간의 $K$개의 대표 임베딩을 학습하는 데 활용된다고 보고한다.

이는 PSC의 개념이 다양한 도메인과 환경으로 **일반화 가능한 범용 압축 패러다임**으로 확장될 수 있음을 시사한다.

### 3.5 실세계 배포를 통한 일반화 검증

실세계 적용 가능성을 입증하기 위해, 사족보행 로봇에 실내 장면의 M3 feature field를 배포하였다.

"yellow bath duck"이라는 쿼리로 decoded CLIP feature를 검색하였을 때 고무 오리가 빨간색으로 강조 표시되었으며, 로봇은 깊이 카메라의 깊이 정보를 통해 목표 객체의 3D 위치를 파악하고 파지(grasping) 작업을 수행할 수 있었다.

이는 합성 데이터셋에서 훈련된 표현이 **실세계 로봇 조작(real-world robot manipulation)** 으로 일반화됨을 직접 검증한 것이다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

| 연구 | 방법 | 특징 | M3 대비 차이점 |
|------|------|------|----------------|
| **NeRF** (Mildenhall et al., 2020) | Neural Radiance Field | 암시적 표현, 볼륨 렌더링 | 렌더링 속도 느림, feature 저장 불가 |
| **3D-GS** (Kerbl et al., 2023) | 3D Gaussian Splatting | 명시적 표현, 고속 렌더링 | RGB만 저장, semantic feature 없음 |
| **F-3DGS** (Zhou et al., 2024) | Feature 3DGS | 3DGS + feature distillation | 저차원(16-64) 직접 증류 → 정보 손실 |
| **LERF** (Kerr et al., 2023) | Language Embedded Radiance Field | NeRF + CLIP | NeRF 기반으로 느린 렌더링 |
| **LangSplat** (Qin et al., 2024) | Language Gaussian Splatting | 3DGS + CLIP | 단일 언어 모달리티, 고차원 문제 미해결 |
| **M3** (Zou et al., 2025) | PSC + Gaussian Memory Attention | 다모달 파운데이션 모델 + 3DGS | **압축 문제 최초 해결**, 멀티모달 지원 |

이전 feature splatting 연구(F-3DGS, F-Splat)는 미분 가능한 렌더링을 통해 파운데이션 모델에서 얻은 2D feature map을 3D Gaussian으로 직접 증류하는 방식을 취했으나, 계산 제약으로 인해 Gaussian primitive의 feature 벡터 차원이 원래 2D feature map(일반적으로 1024)에 비해 크게 감소(16~64 수준)되어 잠재적인 정보 병목을 유발했다.

기존의 잠재 매핑 기법들은 학습된 디코더를 통해 고차원 파운데이션 모델 임베딩을 저차원 Gaussian 속성으로 압축하는 3D feature distillation에 의존하는 것이 일반적이었으나, 이러한 증류는 정보 병목을 초래하여 의미론적 충실도(semantic fidelity) 손실과 디코딩된 임베딩과 원본 타겟 간의 정렬 불일치로 이어진다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

**① 3D Feature Distillation 패러다임 전환**

M3는 2D feature를 3D 임베딩에 직접 증류하는 대신, Gaussian Memory Attention을 PSC와 principal query 사이에 적용함으로써, 파운데이션 모델의 높은 표현력을 보존하면서 3D 일관성 있는 저차원 Gaussian 구조를 유지하는 패러다임 전환을 제시한다. 이는 3D 장면 표현 연구 전체에 새로운 방향을 제시한다.

**② 로봇공학 및 Embodied AI 응용**

M3와 유사한 메모리 시스템을 탑재한 로봇은 노인이나 장애인을 보조하여 객체의 위치를 기억하면서 가정 내를 더 효과적으로 탐색할 수 있을 것이다. AR/VR 응용 프로그램은 공간에 대한 자연어 쿼리에 응답하는 더 지능적인 가상 환경을 만들 수 있을 것이며, 자율주행 차량 역시 더 정교한 공간 추론 능력에서 이점을 얻을 수 있을 것이다.

**③ 후속 연구에서의 기반 기술화**

LatentAM 등 후속 연구에서는 M3에서 처음 제안된 딕셔너리 기반 전략을 기반으로 발전시켜, 온라인(online) 처리 및 대규모 장면 확장성 문제를 해결하고자 한다.

**④ LMM의 3D 확장 연구 촉진**

M3는 AI 시스템이 물리적 공간을 이해하고 상호작용하는 방식에 있어 근본적인 진보를 의미하며, 3D 공간 표현과 언어 이해를 결합함으로써 물리적 환경 내 객체와 그 관계에 대해 추론할 수 있는 보다 인간다운 메모리 시스템을 만든다.

### 5.2 향후 연구 시 고려할 점

**① 동적 장면 대응**

현재 구현은 사전 구축된 정적 3D 장면에 의존하며, 실시간 장면 재구성은 객체가 이동하거나 변화하는 동적 환경에서 여전히 어렵다. 향후 연구에서는 Gaussian의 동적 업데이트 메커니즘 및 온라인 학습 방법을 탐색해야 한다.

**② 대규모 환경 확장성**

M3는 대규모 장면에서 PSC 딕셔너리의 무한 증가로 인한 메모리 부족 문제가 발생할 수 있으며, 도시 규모의 환경이나 다중 건물 단지로의 확장 방법을 완전히 다루지 않는다. 향후에는 계층적 메모리 구조나 스트리밍 기반 PSC 업데이트 전략이 필요하다.

**③ 파운데이션 모델 의존성 완화**

시스템의 성능은 기반 파운데이션 모델의 품질에 크게 의존하여, CLIP이 특정 객체나 개념을 인식하지 못하면 이 한계가 M3로 이어지며, 과소 표현된 카테고리의 객체에 대한 편향이나 인식 공백으로 이어질 수 있다. 도메인별 파운데이션 모델 파인튜닝 및 앙상블 전략 연구가 필요하다.

**④ 메모리 뱅크에서의 직접 추론**

저자들이 제시한 미래 연구 방향인 최적화된 메모리 뱅크에서 직접 작동하는 추론 모듈 설계는 LLM/LMM과 3D 공간 메모리의 직접적 통합을 가능하게 하는 중요한 연구 과제이다.

**⑤ 프라이버시 및 윤리적 고려**

저자들은 개인 공간을 포함할 수 있는 환경의 상세한 3D 공간 메모리 구축에 대한 프라이버시 함의를 완전히 탐구하지 않았으며, 모든 것을 기억하는 시스템은 동의와 데이터 저장에 관한 중요한 질문을 제기한다. 향후 연구에서는 차등 프라이버시(differential privacy)나 선택적 메모리 삭제 메커니즘을 고려해야 한다.

---

## 📚 참고자료 출처

| 번호 | 출처 |
|------|------|
| 1 | **arXiv (2503.16413)** - M3: 3D-Spatial MultiModal Memory: https://arxiv.org/abs/2503.16413 |
| 2 | **ICLR 2025 Proceedings (공식 논문)**: https://proceedings.iclr.cc/paper_files/paper/2025/file/639f9db06bf70c4ff44cf22a7f92cb08-Paper-Conference.pdf |
| 3 | **OpenReview (ICLR 2025 심사 페이지)**: https://openreview.net/forum?id=XYdstv3ySl |
| 4 | **arXiv HTML 전문**: https://arxiv.org/html/2503.16413 |
| 5 | **Hugging Face Papers**: https://huggingface.co/papers/2503.16413 |
| 6 | **GitHub 공식 구현체 (MaureenZOU/m3-spatial)**: https://github.com/MaureenZOU/m3-spatial |
| 7 | **프로젝트 페이지**: https://m3-spatial-memory.github.io/ |
| 8 | **AI Models FYI 분석 페이지**: https://www.aimodels.fyi/papers/arxiv/m3-3d-spatial-multimodal-memory |
| 9 | **LatentAM (후속 연구, arXiv 2602.12314)**: https://arxiv.org/html/2602.12314 (M3의 영향 및 한계 분석) |
| 10 | **3DLLM-Mem (arXiv 2505.22657, M3 인용 연구)**: https://arxiv.org/html/2505.22657v1 |

> ⚠️ **정확도 관련 고지**: 수식의 세부 구성 요소 중 일부(특히 수식의 구체적 스케일링 상수 및 내부 구현 세부사항)는 공개된 arXiv HTML 본문에서 LaTeX 렌더링 한계로 인해 일부 표기가 불명확한 부분이 있어, 개념적으로 재구성하여 제시하였습니다. 정확한 수식은 [공식 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/639f9db06bf70c4ff44cf22a7f92cb08-Paper-Conference.pdf)를 직접 참조하시기 바랍니다.
