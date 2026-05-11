
# COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting

> **논문 정보**: Jiaxin Zhang, Junjun Jiang, Youyu Chen, Kui Jiang, Xianming Liu  
> **발표**: CVPR 2025 | arXiv: 2503.19443 (2025년 3월 25일)  
> **공식 코드**: https://github.com/ZestfulJX/COB-GS

---

## 1. 핵심 주장과 주요 기여 요약

3D 비전 도메인에서 고품질 장면 이해를 위해 정확한 객체 분할이 매우 중요하다. 그러나 3DGS 기반의 3D 분할은 객체 경계를 정확하게 구분하는 데 어려움을 겪는데, 이는 Gaussian primitives가 고유한 부피와 학습 중 의미론적 안내의 부재로 인해 객체 경계를 가로질러 확장되기 때문이다.

이에 대한 COB-GS의 핵심 주장은 다음과 같다:

COB-GS는 장면 내에서 얽혀 있는 Gaussian primitives의 흐릿한 경계를 명확하게 구분함으로써 분할 정확도를 개선하는 것을 목표로 한다. 기존 접근법들이 모호한 Gaussian을 제거하고 시각적 품질을 희생하는 것과 달리, COB-GS는 3DGS 정제 방법으로서 의미론적 정보와 시각적 정보를 공동으로 최적화하여 두 수준이 효과적으로 협력하도록 한다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| **① Boundary-Adaptive Gaussian Splitting (BAGS)** | 마스크 그래디언트 통계를 활용하여 경계 Gaussian 식별 및 분할 |
| **② Boundary-Guided Texture Restoration (BGTR)** | 분할 후 저하된 경계 영역의 텍스처 복원 |
| **③ Robustness Against Erroneous Masks (RAEM)** | 사전 학습 모델의 부정확한 마스크에 대한 강건성 확보 |

논문의 ablation study에서 BAGS는 경계 적응형 Gaussian 분할을, BGTR은 경계 안내 텍스처 복원을, RAEM은 오류 마스크에 대한 강건성 향상을 각각 의미한다.

---

## 2. 논문 상세 설명

### 2-1. 해결하고자 하는 문제

#### 문제 1: 경계 Gaussian의 모호성

기존의 장면 재구성은 의미론적 정보를 무시하고 주로 시각적 최적화에 집중하며 Gaussian primitives의 체적 특성을 간과한다. 이러한 간과는 장면 분할 시 경계 Gaussian의 레이블이 흐릿해져, 흐릿한 객체 경계로 특징지어지는 부정확한 분할 결과를 초래한다.

#### 문제 2: 기존 방법의 한계

일부 기존 방법들은 모호한 경계 Gaussian을 직접 삭제하지만, Gaussian primitives를 경계에서 단순 제거하면 시각적 품질을 저하시킨다.

SAGD는 잘못된 GT 마스크에 대한 강건성이 부족하여 분할된 경계를 따라 작은 Gaussian 아티팩트를 생성한다. FlashSplat은 경계 흐림을 줄이지만 객체의 구조적 무결성을 훼손한다.

#### 문제 3: 두 가지 분할 패러다임의 한계

현재 3DGS 분할을 실행하는 두 가지 주요 방법론이 존재한다: feature 기반 방법과 mask 기반 방법. Feature 기반 방법은 3D 장면 재구성과 함께 각 Gaussian primitive에 대한 구별적 특징 속성을 학습하도록 작동한다. Mask 기반 방법은 SAM의 의미론적 마스크를 활용하여 재구성된 3DGS 장면에서 각 3D Gaussian의 카테고리 레이블을 학습한다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### 3DGS 기본 원리

3D Gaussian Splatting에서 각 Gaussian primitive $G_i$는 다음과 같이 정의된다:

$$G_i = \{\mu_i, \Sigma_i, \alpha_i, c_i\}$$

여기서 $\mu_i$는 3D 위치(mean), $\Sigma_i$는 공분산 행렬(covariance), $\alpha_i$는 불투명도(opacity), $c_i$는 색상(color)이다.

2D 렌더링은 alpha blending으로 수행된다:

$$\hat{I}(p) = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j < i}(1 - \alpha_j)$$

각 Gaussian에 mask label $l_i \in [0, 1]$을 추가하여 mask rendering을 수행한다:

$$\hat{M}(p) = \sum_{i \in \mathcal{N}} l_i \alpha_i \prod_{j < i}(1 - \alpha_j)$$

#### ① Boundary-Adaptive Gaussian Splitting (BAGS)

경계 Gaussian을 식별하기 위해 **마스크 그래디언트 통계**를 활용한다. 각 Gaussian의 누적 마스크 그래디언트 $\nabla l_i$를 추적하여 경계 여부를 판단한다:

경계 Gaussian 집합 $\mathcal{B}$를 다음 조건으로 정의한다:

```math
\mathcal{B} = \left\{ G_i \;\middle|\; \frac{\|\nabla l_i\|}{\max_j \|\nabla l_j\|} > \delta \right\}
```

$\delta$가 증가할수록 모호한 경계 Gaussian의 수는 감소하지만 전체 Gaussian의 수는 빠르게 증가한다. $\delta$가 증가할수록 의미론적으로 모호한 Gaussian에 대한 판별이 강화되어 더 많은 분할이 발생한다.

분할 과정에서 소규모 Gaussian은 $\mathcal{B}$에서 제외된다. 나머지 대형 Gaussian에 대해 각각을 두 개의 작은 Gaussian으로 교체하며 원본 크기에서 축소한다.

분할된 두 Gaussian의 위치는 원본 Gaussian의 주축(principal axis) 방향으로 배치:

$$\mu_{i,1} = \mu_i + \frac{\sigma_{\max}}{2} \cdot \mathbf{v}_{\max}, \quad \mu_{i,2} = \mu_i - \frac{\sigma_{\max}}{2} \cdot \mathbf{v}_{\max}$$

여기서 $\sigma_{\max}$와 $\mathbf{v}_{\max}$는 $\Sigma_i$의 최대 고유값과 그에 대응하는 고유벡터이다.

#### ② 전체 최적화 손실 함수

COB-GS의 공동 최적화는 마스크 손실과 텍스처 손실로 구성된다:

$$\mathcal{L}_{\text{mask}} = \lambda_1 \mathcal{L}_{\text{BCE}}(\hat{M}, M) + \lambda_2 \mathcal{L}_{\text{reg}}$$

$$\mathcal{L}_{\text{texture}} = (1 - \lambda_{\text{ssim}})\mathcal{L}_1(\hat{I}, I) + \lambda_{\text{ssim}} \mathcal{L}_{\text{SSIM}}(\hat{I}, I)$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mask}} + \mathcal{L}_{\text{texture}}$$

#### ③ 파이프라인 구조

COB-GS의 파이프라인은 다음과 같다: 왼쪽에는 두 단계의 마스크 생성 방법이 제시되며, 텍스트 프롬프트를 기반으로 이미지 시퀀스에서 SAM2를 활용하여 마스크 예측을 수행해 관심 영역의 마스크를 얻는다. 이미지와 마스크는 3DGS 정제의 감독으로 사용된다. 오른쪽에서는 재구성된 3DGS 장면에 대해 마스크와 텍스처를 공동으로 교대 최적화한다. 마스크 최적화에서는 경계 구조를 정제하기 위해 boundary-adaptive Gaussian splitting이 수행된다.

#### ④ 두 단계 마스크 생성 방법

Grounded-SAM-2를 기반으로 한 안정적인 시퀀스 마스크 추출 방법을 제공한다.

두 단계 마스크 생성 방법에서 SAM2 hiera large 모델과 Grounding DINO swinb 모델을 활용한다.

#### ⑤ 학습 전략

각 단계는 $2 \times V$ 이터레이션으로 학습되며, $V$는 입력 이미지의 수이다. 서로 다른 장면 유형에 두 가지 하이퍼파라미터 세트가 사용된다: forward 장면의 경우 $\delta = 0.5$로 설정하고 총 $22 \times V$ 이터레이션의 교대 최적화를 수행하며, surrounding 장면의 경우 $\delta = 0.8$로 설정하고 $14 \times V$ 이터레이션의 교대 최적화를 수행한다.

---

### 2-3. 모델 구조

```
[입력: 이미지 시퀀스 + 텍스트 프롬프트]
         ↓
[두 단계 마스크 생성 (SAM2 + Grounding DINO)]
         ↓
[기존 3DGS 장면 재구성 (30,000 iters)]
         ↓
┌─────────────────────────────────────┐
│         COB-GS 공동 최적화 루프      │
│                                     │
│  ┌──────────────┐ ┌──────────────┐  │
│  │  마스크 최적화│→│ 텍스처 최적화│  │
│  │    (BAGS)    │ │   (BGTR)    │  │
│  └──────────────┘ └──────────────┘  │
│            교대 반복                │
└─────────────────────────────────────┘
         ↓
[강건성 처리: 소규모 모호 Gaussian 정제 (RAEM)]
         ↓
[출력: 명확한 경계 + 고품질 텍스처를 가진 분할 결과]
```

COB-GS는 두 가지 구성 요소로 이루어진다: 최적화 프로세스와 강건성 프로세스. 최적화 프로세스는 마스크 최적화와 텍스처 최적화를 교대로 수행한다. 마스크 최적화 단계에서 마스크 레이블을 최적화하고 Gaussian 분할을 수행하며, 마스크 레이블의 학습률은 0.1로 설정된다. 텍스처 최적화 단계에서는 기하학과 텍스처를 최적화하며, 외관의 학습률은 원래 3DGS 설정을 따른다.

---

### 2-4. 성능 향상

#### 정량적 평가 (NVOS 데이터셋)

COB-GS의 결과는 객체의 경계를 더 명확하게 분할하며, 흐릿한 Gaussian 없이 객체 제거 후 배경이 더 깔끔하다.

Ablation 결과:

Gaussian 분할 없이는 대형 모호 Gaussian이 가장 낮은 메트릭을 산출한다. Gaussian 분할만 적용하면 분할 메트릭은 향상되지만 텍스처 품질이 저하된다. 공동 최적화는 텍스처 품질을 개선하지만 분할 정확도가 약간 저하된다. 마지막으로 소형 Gaussian에 대한 강건 처리를 포함하면 최적 성능을 달성한다.

#### 시각적 품질 평가

참조 이미지 부재로 인해 CLIP-IQA(no-reference IQA)를 활용하여 텍스트 프롬프트와 이미지의 일치도를 평가한다. 경계 품질에 집중한 세 가지 프롬프트를 설정하여 분할 결과를 종합적으로 평가한다.

SAGD와 FlashSplat은 Gaussian을 거칠게 처리하여 분할 정확도를 위해 외관을 파괴하고 시각적 품질을 희생한다. SA3D는 명확한 모호 경계 Gaussian을 나타낸다. COB-GS는 전반적으로 높은 시각적 품질을 달성한다.

---

### 2-5. 한계점

논문에서 명시적으로 인정되는 한계점 및 검색된 결과 기반 분석:

1. **장면별 최적화 의존성**: COB-GS는 원래 3D Gaussian Splatting 기반의 후처리 방법으로, 각 장면에 대해 원래 3DGS 파라미터에 따라 30,000 이터레이션의 학습을 수행하여 원래 3DGS 장면을 먼저 얻어야 한다. 이는 장면별 재학습이 필요함을 의미한다.

2. **계산 비용**: SAGA와 같은 feature 기반 방법은 mask 기반 방법보다 더 많은 시간과 메모리를 소비한다. 이는 feature 기반 방법이 전체 장면에 대해 고차원 특징을 최적화하는 반면, COB-GS는 레이블 최적화에만 집중하기 때문이다.

3. **단일 분할 한계**: SAGA는 단일 학습 세션으로 다중 분할을 허용하여 빈번한 분할이 필요한 오프라인 고정 장면에 적합하지만, mask 기반 방법은 재구성된 3DGS에 레이블을 직접 할당하여 더 빠른 단일 분할을 제공한다.

4. **동적 장면 미지원**: COB-GS는 정적 3DGS 장면에 특화되어 있어 동적 장면(Dynamic Gaussian)으로의 직접 확장이 어렵다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 텍스트 프롬프트 기반 일반화

COB-GS는 이미지와 텍스트 프롬프트만으로 3DGS 분할을 완료하는 방법을 추가로 소개한다.

open-vocabulary 분할 능력을 평가한다. 이는 특정 카테고리에 국한되지 않는 개방형 어휘 분할이 가능함을 의미한다.

### 3-2. 사전 학습 모델의 부정확한 마스크에 대한 강건성

COB-GS는 분할 정확도와 사전 학습 모델에서 나온 부정확한 마스크에 대한 강건성을 실질적으로 향상시키며, 높은 시각적 품질을 유지하면서 명확한 경계를 만들어낸다.

이는 SAM2, Grounded-SAM 등 다양한 2D foundation model을 교체해도 안정적인 성능을 낼 수 있는 가능성을 시사한다.

### 3-3. 다중 객체 분할로의 확장

단일 객체 및 다중 객체 분할을 포함한 분할된 객체와 객체 제거 후 배경을 시각화하여 현재 SOTA 3DGS 분할 방법들과 다중 장면에서 비교한다.

다중 객체 분할의 효율적 업데이트가 지원된다.

### 3-4. 다양한 데이터셋에서의 일반화

COB-GS는 다음의 다양한 데이터셋에서 평가를 진행한다:
- **NVOS dataset**: 기준 비교 평가
- **LERF-mask dataset**: open-vocabulary 분할 능력 평가
- **Tanks & Temples (T&T) dataset**: 단일 객체 분할
- **MIP-360 dataset**: 복잡한 surrounding 장면

단일 객체 분할의 경우, T&T 데이터셋의 Truck 장면과 MIP-360 데이터셋의 Kitchen 장면을 선택하였다.

### 3-5. 일반화 성능의 핵심 요인

일반화 성능 향상의 핵심은 다음 요소들에 있다:

1. **Foundation Model 활용**: 관련 연구인 SA3D는 SAM을 사용하여 최적화 효율성과 마스크 뷰 일관성을 개선하고, SAM2와 같은 기반 모델의 등장으로 비디오 시퀀스에 걸친 마스크 예측이 가능해졌다.

2. **역방향 렌더링 기반 레이블 추출**: 역방향 렌더링을 통한 추출 프로세스는 텍스처 최적화가 동시에 장면 레이블을 최적화하는 것을 보장한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. 3DGS 분할 방법 계보

```
NeRF 기반 (2020~2022)
    ↓
Semantic-NeRF, NVOS, SA3D
    ↓
3DGS 기반 등장 (2023~)
    ↓
Feature 기반: SAGA, GaussianGrouping, Feature3DGS, SAGD
Mask 기반: FlashSplat, SA3D (3DGS 버전), COB-GS
    ↓
최신: GaussianTrimmer (plug-and-play), COB-GS (CVPR 2025)
```

### 4-2. 주요 방법 비교표

| 방법 | 학회/연도 | 핵심 접근법 | 경계 처리 | 시각적 품질 |
|------|-----------|-------------|-----------|------------|
| **Semantic-NeRF** | ICCV 2021 | NeRF + 의미론적 레이블 | ❌ 낮음 | NeRF 한계 |
| **NVOS** | CVPR 2022 | NeRF + MLP 기반 선택 | ❌ 낮음 | NeRF 한계 |
| **SAGA** | AAAI 2025 | Scale-gated affinity feature | △ 중간 | △ 중간 |
| **GaussianGrouping** | ECCV 2024 | tracker + joint learning | △ 중간 | △ 중간 |
| **SAGD** | 2024 | Gaussian decomposition | △ 중간 | ❌ 아티팩트 |
| **FlashSplat** | ECCV 2024 | Linear programming | △ 중간 | ❌ 구조 파괴 |
| **COB-GS** | **CVPR 2025** | BAGS + BGTR + RAEM | ✅ 높음 | ✅ 높음 |
| **GaussianTrimmer** | 2025 | Plug-and-play trimming | ✅ 높음 | △ 중간 |

### 4-3. 각 방법의 특징 분석

**SAGA (Segment Any 3D Gaussians, AAAI 2025)**:
SAGA는 3D-GS 기반의 고효율 3D 프롬프트 분할 방법이다. 2D 시각적 프롬프트가 주어지면 SAGA는 4ms 이내에 3D Gaussian으로 표현된 해당 3D 타겟을 분할할 수 있다. 이는 각 3D Gaussian에 scale-gated affinity feature를 부착하여 다중 세분성 분할을 가능하게 함으로써 달성된다.

**FlashSplat (ECCV 2024)**:
기존 방법들은 각 Gaussian에 고유한 레이블을 할당하기 위해 반복적 경사 하강에 의존하여 긴 최적화와 차선의 솔루션을 초래한다. FlashSplat은 3D-GS 분할을 위한 전역 최적 솔버를 제안하며, 재구성된 3D-GS 장면에서 2D 마스크 렌더링이 각 Gaussian의 레이블에 대해 본질적으로 선형 함수라는 점을 핵심 통찰로 활용한다.

**GaussianTrimmer (2025)**:
GaussianTrimmer는 효율적이고 플러그-앤-플레이 방식의 후처리 방법으로 기존 3D Gaussian 분할 방법의 거친 경계를 다듬을 수 있다. 이 방법은 두 가지 핵심 단계로 구성된다: 균일하고 잘 커버된 가상 카메라 생성, 가상 카메라의 2D 분할 결과를 기반으로 primitive 수준에서 Gaussian 트리밍.

**COB-GS vs 비교 대상 핵심 차별점**:
기존 접근법들이 모호한 Gaussian을 제거하고 시각적 품질을 희생하는 것과 달리, COB-GS는 3DGS 정제 방법으로서 의미론적 정보와 시각적 정보를 공동으로 최적화하여 두 수준이 효과적으로 협력하도록 한다.

COB-GS는 의미론과 텍스처를 동시에 학습하는 공동 최적화 프레임워크를 제안하여 시각적 충실도를 유지하면서 더 선명한 객체 경계를 달성한다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

#### ① 의미-시각 공동 최적화 패러다임의 확립

COB-GS는 "경계 Gaussian을 제거"하는 대신 "정제하여 재배치"하는 새로운 패러다임을 확립했다. 경계 적응형 Gaussian 분할 기법을 도입하여 의미론적 그래디언트 통계를 활용하여 모호한 Gaussian을 식별하고 분할함으로써 객체 경계에 밀접하게 정렬하며, 시각적 최적화를 위해 정제된 경계 구조 특히 경계 영역의 저하된 차선 텍스처를 복원한다.

이는 향후 3DGS 편집, 조작, 합성 분야 연구에서도 **경계 인식 최적화**가 핵심 고려사항이 되어야 함을 시사한다.

#### ② 3DGS와 Foundation Model의 융합 연구 촉진

SAM2와 같은 기반 모델의 등장으로 비디오 시퀀스에 걸친 마스크 예측이 가능해졌다. SAM2는 SAM의 인코더-디코더 구조를 유지하며, 인코더가 이미지를 입력으로 받는다.

COB-GS의 성공은 SAM2, Grounded-DINO 등의 foundation model을 3DGS 파이프라인에 효과적으로 통합하는 연구를 더욱 가속화할 것으로 예상된다.

#### ③ 3D 장면 편집·상호작용의 품질 기준 향상

COB-GS의 목표는 foreground와 background를 포함한 3DGS에서의 고품질 3D 분할을 달성하는 것이다. 이는 기존 방법의 불분명한 분할 결과, 흐릿한 경계 Gaussian, 공동 최적화의 분할 결과, 사전 학습 모델의 잘못된 마스크로 인한 작은 모호 Gaussian 문제를 해결한다.

로봇공학, 자율주행, AR/VR 등 downstream task에서 3DGS 세그멘테이션의 품질 기준이 높아지게 될 것이다.

#### ④ 플러그-앤-플레이 후처리 연구의 방향 제시

COB-GS는 경계 문제를 다루지만 Gaussian 표현의 다른 측면에 집중한다. 반면 GaussianTrimmer는 최소 지연으로 기존 분할 결과를 향상시키는 후처리 단계로서 효율적이고 효과적인 경계 트리밍을 강조한다.

---

### 5-2. 앞으로의 연구에서 고려할 점

#### 고려사항 1: 동적 장면으로의 확장

현재 COB-GS는 정적 장면에 특화되어 있다. 3DGS는 최근 NeRF의 효율적이고 경쟁력 있는 대안으로 등장하여 실시간으로 고충실도의 사실적인 렌더링을 가능하게 하고 있다. 새로운 뷰 합성을 넘어 3DGS의 명시적이고 컴팩트한 특성은 기하학적, 의미론적 이해가 필요한 다양한 downstream 애플리케이션을 가능하게 한다. 따라서 동적 3DGS(4D Gaussian Splatting) 환경에서의 경계 분할 연구가 필요하다.

#### 고려사항 2: Generalizable 3DGS와의 결합

일부 최근 방법들은 정확도가 높지만 대규모 사전 학습 기반이라 장면당 수 분에서 수십 분이 소요된다. 장면 독립적(scene-independent) 일반화 가능한 3DGS와 COB-GS 방식을 결합하여 학습 없이(training-free) 고품질 경계 분할을 수행하는 연구가 중요하다.

#### 고려사항 3: 하이퍼파라미터 $\delta$의 자동 조정

$\delta$가 증가함에 따라 모호한 경계 Gaussian의 수는 감소하지만 전체 Gaussian 수는 빠르게 증가하며, 의미론적으로 모호한 Gaussian에 대한 판별이 강화되어 더 많은 분할이 발생한다. 이 $\delta$ 임계값을 장면 특성에 따라 자동으로 추정하는 adaptive 메커니즘 연구가 필요하다.

#### 고려사항 4: 실시간 응용을 위한 경량화

feature 기반 방법들은 mask 기반 방법보다 더 많은 시간과 메모리를 소비하며, 이는 feature 기반 방법이 전체 장면에 대해 고차원 특징을 최적화하는 반면 COB-GS는 레이블 최적화에만 집중하기 때문이다. 이 차이는 최적화 시간에서 명확하게 나타난다. 로봇, 자율주행 등 실시간 응용을 위한 경량화 및 온라인 처리 방법 연구가 요구된다.

#### 고려사항 5: 다중 모달리티 프롬프트 지원 확대

현재 텍스트 및 포인트 기반 프롬프트를 지원하지만, 향후에는 이미지 참조(reference image), 오디오, 자연어 추론 등 다양한 모달리티를 지원하는 방향이 요구된다.

#### 고려사항 6: 3DGS 경계와 표면 재구성의 통합

3D Gaussian의 스케일 변동 범위가 크기 때문에 전경과 배경에 걸쳐 있는 대형 Gaussian이 분할된 객체의 들쭉날쭉한 경계를 초래한다. 2D Gaussian Splatting(2DGS)이나 SuGaR와 같이 표면 정렬된 Gaussian 표현과의 통합을 통해 경계 분할의 근본적 해결 가능성을 탐구할 필요가 있다.

---

## 참고문헌 및 출처

| 번호 | 제목 | 출처 |
|------|------|------|
| 1 | **COB-GS 논문 (arXiv)** | https://arxiv.org/abs/2503.19443 |
| 2 | **COB-GS 논문 (arXiv HTML 상세)** | https://arxiv.org/html/2503.19443 |
| 3 | **COB-GS CVPR 2025 논문 (PDF)** | https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_COB-GS_Clear_Object_Boundaries_in_3DGS_Segmentation_Based_on_Boundary-Adaptive_CVPR_2025_paper.pdf |
| 4 | **COB-GS CVPR 2025 Supplemental (PDF)** | https://openaccess.thecvf.com/content/CVPR2025/supplemental/Zhang_COB-GS_Clear_Object_CVPR_2025_supplemental.pdf |
| 5 | **COB-GS 공식 GitHub** | https://github.com/ZestfulJX/COB-GS |
| 6 | **CVPR 2025 포스터 페이지** | https://cvpr.thecvf.com/virtual/2025/poster/32678 |
| 7 | **FlashSplat (ECCV 2024)** | https://arxiv.org/abs/2409.08270 |
| 8 | **SAGA: Segment Any 3D Gaussians (AAAI 2025)** | https://arxiv.org/abs/2312.00860 |
| 9 | **GaussianTrimmer (2025)** | https://arxiv.org/html/2601.12683 |
| 10 | **SAGD: Semantic Anything in 3D Gaussians (2024)** | https://github.com/XuHu0529/SAGS |
| 11 | **A Survey on 3DGS Applications (2025)** | https://arxiv.org/abs/2508.09977 |
| 12 | **Awesome-3DGS-Applications (GitHub)** | https://github.com/heshuting555/Awesome-3DGS-Applications |
| 13 | **Segment Anything in 3D with Radiance Fields (SA3D)** | https://arxiv.org/abs/2304.12308 |
| 14 | **ResearchGate (COB-GS)** | https://www.researchgate.net/publication/390175882 |

> ⚠️ **정확도 관련 고지**: 본 답변의 일부 수식(특히 손실 함수의 상세 계수 및 경계 Gaussian 분할의 정확한 수식 표현)은 공개된 arXiv HTML 및 GitHub 기반으로 재구성되었으며, CVPR 2025 최종 논문 PDF의 세부 수식과 미세하게 다를 수 있습니다. 정확한 수식은 공식 논문 PDF(https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_COB-GS_Clear_Object_Boundaries_in_3DGS_Segmentation_Based_on_Boundary-Adaptive_CVPR_2025_paper.pdf)를 직접 참조하시기 바랍니다.
