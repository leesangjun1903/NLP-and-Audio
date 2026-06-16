# Learning Cross-Modal Retrieval with Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Hu et al., CVPR 2021)은 **멀티모달 데이터(이미지-텍스트 등)에서 노이즈 레이블(noisy labels) 환경 하에서도 강건한 크로스 모달 검색(cross-modal retrieval)** 이 가능한 프레임워크를 제안합니다. 핵심 주장은 다음과 같습니다:

> "기존 크로스 모달 학습 방법들은 노이즈 레이블에 매우 취약하며, 노이즈 레이블 문제와 크로스 모달 이질성(heterogeneous gap) 문제를 **동시에** 해결하는 새로운 프레임워크가 필요하다."

### 주요 기여

| 기여 | 설명 |
|------|------|
| **MRL 프레임워크** | 노이즈 레이블 환경에서 멀티모달 학습을 위한 일반적 프레임워크 제안 |
| **Robust Clustering Loss (RC, $\mathcal{L}_r$)** | 노이즈 샘플 영향을 억제하고 크로스 모달 간격을 좁히는 손실 함수 |
| **Multimodal Contrastive Loss (MC, $\mathcal{L}_c$)** | 서로 다른 모달리티 간 상호 정보를 최대화하는 비지도적 대조 손실 함수 |
| **실험적 검증** | 4개 벤치마크, 14개 SOTA 비교, 다양한 노이즈 비율(0.2~0.8)에서 성능 우위 입증 |

---

## 2. 세부 분석: 문제 정의, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

**세 가지 핵심 문제:**

1. **노이즈 레이블 과적합 문제**: 딥 뉴럴 네트워크(DNN)는 노이즈 레이블에 쉽게 과적합되어 일반화 성능이 저하됩니다.
2. **크로스 모달 이질성(heterogeneity)**: 이미지와 텍스트는 완전히 다른 특징 공간에 존재하여, 노이즈 환경에서의 공통 표현 학습이 어렵습니다.
3. **모달리티별 레이블 불일치**: 서로 다른 모달리티에 대한 레이블이 서로 다른 주석자(annotator)에 의해 달릴 수 있어, **페어링(pairing)이 보장되지 않습니다.**

논문의 Figure 2에서 보여주듯이, 표준 Cross-Entropy(CE) 손실로 학습할 경우 노이즈 비율 0.6 환경에서 훈련 세트에는 과적합되지만, 검증 세트의 MAP(Mean Average Precision)는 급격히 하락합니다.

### 2.2 제안 방법 (수식 포함)

#### 기본 표기법

$K$ -범주 멀티모달 데이터셋 $\mathcal{D} = \{\mathcal{M}\_i\}_{i=1}^{m}$에서, $\mathcal{M}_i = \{(\mathbf{x}_j^i, y_j^i)\}\_{j=1}^{N}$은 $i$번째 모달리티를 나타냅니다. 각 모달리티별 네트워크 $f_i$를 통해 공통 정규화 표현을 얻습니다:

$$\mathbf{z}_j^i = f_i(\mathbf{x}_j^i) \subset \mathbb{R}^L \tag{1}$$

---

#### 2.2.1 Robust Clustering Loss (RC, $\mathcal{L}_r$)

**표준 Cross-Entropy의 문제점:**

클러스터링 중심 $\mathbf{C} = \{\mathbf{c}_1, \cdots, \mathbf{c}_K\}$에 대해, 샘플 $\mathbf{x}_j^i$가 $k$번째 클래스에 속할 확률은:

$$p(k|\mathbf{x}_j^i) = \frac{\exp\left(\frac{1}{\tau_1} \mathbf{c}_k^T \mathbf{z}_j^i\right)}{\sum_{t=1}^{K} \exp\left(\frac{1}{\tau_1} \mathbf{c}_t^T \mathbf{z}_j^i\right)} \tag{2}$$

여기서 $\tau_1$은 온도(temperature) 파라미터입니다. 이를 이용한 표준 CE 손실:

$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{j=1}^{N} \sum_{k=1}^{K} q(k|\mathbf{x}_j^i) \log p(k|\mathbf{x}_j^i) \tag{3}$$

$$q(k|\mathbf{x}_j^i) = \begin{cases} 1 & \text{if } k = y_j^i \\ 0 & \text{otherwise} \end{cases} \tag{4}$$

**CE 손실의 핵심 문제**: $CE(p) = -\log(p)$는 확률 $p$가 낮을수록(어려운 샘플일수록) 큰 손실값을 부여합니다. 노이즈 레이블 환경에서 "어려운 샘플"은 종종 "노이즈 샘플"이므로, 네트워크가 노이즈 샘플을 더 집중적으로 학습하게 되는 문제가 발생합니다.

**제안하는 RC 손실:**

저자들은 **양성 샘플의 음의 로그 가능도를 최소화하는 대신, 음성 샘플의 로그 가능도를 최소화**하는 방식으로 손실 함수를 재설계합니다:

$$\mathcal{L}_r = \frac{1}{N} \sum_{i=1}^{m} \sum_{j=1}^{N} \log(1 - p(y_j^i|\mathbf{x}_j^i)) \tag{5}$$

> **핵심 아이디어**: $RC(p) = \log(1-p)$는 $p$가 클수록(쉬운/clean 샘플일수록) 큰 손실값을 부여하고, $p$가 작을수록(어려운/noisy 샘플일수록) 작은 손실값을 부여합니다. 이로써 clean 샘플이 그래디언트를 지배하게 됩니다.

CE와 RC 손실 함수의 비교:
- $CE(p) = -\log(p)$: $p \to 0$일 때 손실 $\to \infty$ (어려운 샘플에 집중)
- $RC(p) = \log(1-p)$: $p \to 1$일 때 손실 $\to -\infty$의 절댓값 큼 (쉬운 샘플에 집중)

---

#### 2.2.2 Multimodal Contrastive Loss (MC, $\mathcal{L}_c$)

샘플 $\mathbf{x}_j^i$가 $m$개의 모달리티에서 $j$번째 인스턴스에 속할 확률:

$$P(j|\mathbf{x}_j^i) = \frac{\sum_{l=1}^{m} \exp\left(\frac{1}{\tau_2}\left(\mathbf{z}_j^l\right)^T \mathbf{z}_j^i\right)}{\sum_{l=1}^{m} \sum_{t=1}^{N} \exp\left(\frac{1}{\tau_2}\left(\mathbf{z}_t^l\right)^T \mathbf{z}_j^i\right)} \tag{6}$$

여기서 $\tau_2$는 온도 파라미터입니다.

MC 손실은 동일 인스턴스의 서로 다른 모달리티 샘플들 $\{x_j^k\}\_{k=1}^{m}$을 컴팩트하게, 다른 인스턴스 샘플들 $\{x_l^k\}_{l \neq j}$을 산란시키는 목표로 다음과 같이 정의됩니다:

$$\mathcal{L}_c = -\frac{1}{N} \sum_{i=1}^{m} \sum_{j=1}^{N} \log\left(P(j|\mathbf{x}_j^i)\right) \tag{7}$$

> **핵심 아이디어**: 멀티모달 데이터는 본질적으로 동일 객체를 여러 모달리티로 기술하므로, **증강(augmentation) 없이 자연스럽게 양성 쌍(positive pair)을 정의**할 수 있습니다. 이는 단일 모달 대조 학습(SimCLR, MoCo 등)을 멀티모달로 확장한 것입니다.

---

#### 2.2.3 최종 손실 함수

$$\mathcal{L} = \beta \mathcal{L}_r + (1 - \beta) \mathcal{L}_c \tag{8}$$

여기서 $\beta \in [0, 1]$는 두 손실 함수의 균형을 조절하는 하이퍼파라미터입니다.

### 2.3 모델 구조

```
입력 (멀티모달 데이터)
├── 이미지 X₁ (노이즈 레이블 Y₁)
│   └── CNN Backbone (VGG-19 / AlexNet, frozen)
│       └── FC(4096) → ReLU → FC(4096) → ReLU → FC(L)
│
└── 텍스트 Xₘ (노이즈 레이블 Yₘ)
    └── Text Backbone (Doc2Vec / LDA, frozen)
        └── FC(4096) → ReLU → FC(4096) → ReLU → FC(L)
                              ↓
               공통 단위 구 (Common Unit Sphere) 상의 표현 Z
                              ↓
        ┌─────────────────────────────┐
        │  Robust Clustering Loss Lᵣ  │  ← 클러스터 C = {c₁,...,cₖ}
        │  Multimodal Contrastive Lc  │  ← 인스턴스/페어 수준 대조
        └─────────────────────────────┘
                    ↓
             L = βLᵣ + (1-β)Lc
```

**세부 구현 사항:**
- 옵티마이저: Adam, 학습률 $\alpha = 0.0001$
- 온도 파라미터: $\tau_1 = \tau_2 = 1$
- 배치 크기: Wikipedia(50), INRIA-Websearch(200), NUS-WIDE/XMediaNet(500)
- 최대 에포크: 100
- 백본은 훈련 중 **동결(frozen)** 상태 유지

### 2.4 성능 향상

**4개 데이터셋에서의 MAP 비교 (노이즈 비율 0.8 기준):**

| 데이터셋 | 방향 | 최고 베이스라인 | MRL (Ours) | 향상폭 |
|----------|------|----------------|------------|--------|
| Wikipedia | Image→Text | 0.251 (SMLN) | **0.435** | +18.4%p |
| Wikipedia | Text→Image | 0.237 (SMLN) | **0.400** | +16.3%p |
| INRIA-Websearch | Image→Text | 0.275 (MCCA) | **0.417** | +14.2%p |
| NUS-WIDE | Image→Text | 0.628 (deep-SM) | **0.669** | +4.1%p |
| XMediaNet | Image→Text | 0.070 (SMLN) | **0.334** | +26.4%p |

논문에서는 **80% 노이즈 비율에서 최고 베이스라인 대비 6.5% 이상의 성능 향상**을 보고합니다.

**어블레이션 스터디 (Wikipedia, Image→Text):**

| 방법 | 0.2 | 0.4 | 0.6 | 0.8 |
|------|-----|-----|-----|-----|
| CE (베이스라인) | 0.441 | 0.387 | 0.293 | 0.178 |
| MRL ($\mathcal{L}_r$ only) | 0.482 | 0.434 | 0.363 | 0.239 |
| MRL ($\mathcal{L}_c$ only) | 0.412 | 0.412 | 0.412 | 0.412 |
| **Full MRL** | **0.514** | **0.491** | **0.464** | **0.435** |

### 2.5 한계점

논문에서 명시적으로 언급하거나 분석에서 도출되는 한계점들:

1. **비대칭 노이즈(asymmetric noise) 미검토**: 실험은 **대칭 노이즈(symmetric noise)** 만을 다루며, 실제 환경에서 더 흔한 비대칭 노이즈에 대한 검증이 부재합니다.
2. **하이퍼파라미터 민감성**: $\beta$ 파라미터의 최적 범위가 데이터셋에 따라 크게 달라집니다 (Wikipedia: $0.5 \sim 0.8$, INRIA-Websearch: $0.85 \sim 0.95$). 이는 실제 적용 시 데이터셋별 튜닝이 필요함을 의미합니다.
3. **이진 모달리티(bimodal)에 한정된 실험**: 이론적으로 $m$개 모달리티로 확장 가능하다고 주장하지만, 실제 실험은 이미지-텍스트 2개 모달리티에만 집중되어 있습니다.
4. **백본 동결(frozen backbone)**: 공정한 비교를 위해 백본을 동결했지만, 엔드-투-엔드 파인튜닝 시 성능 변화에 대한 분석이 없습니다.
5. **단일 레이블(single-label) 가정**: 멀티레이블 환경에 대한 확장이 검토되지 않았습니다.
6. **노이즈 유형 탐지 부재**: 노이즈 샘플을 명시적으로 식별/분리하지 않고 손실 함수 재설계로 암묵적으로 처리하므로, 노이즈 구조에 대한 해석 가능성이 낮습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 RC 손실의 일반화 메커니즘

RC 손실 $\mathcal{L}\_r = \frac{1}{N}\sum_{i=1}^{m}\sum_{j=1}^{N}\log(1-p(y_j^i|\mathbf{x}_j^i))$은 다음과 같은 방식으로 일반화 성능을 향상시킵니다:

**수학적 관점**: $p(y_j^i|\mathbf{x}_j^i)$가 높을수록(clean 샘플일 가능성 높음) $\log(1-p)$의 절댓값이 커져 그래디언트가 더 크게 반영됩니다. 노이즈 샘플은 일반적으로 낮은 확률값을 가지므로 $\log(1-p) \approx 0$에 가까워져 학습에 미치는 영향이 최소화됩니다. 이는 **암묵적 샘플 가중치(implicit sample weighting)** 효과로, 별도의 노이즈 탐지 모듈 없이도 일반화 성능을 향상시킵니다.

**비교 관점**:
- Focal Loss: $\mathcal{L}_{FL} = -(1-p)^\gamma \log(p)$ → 어려운 샘플(낮은 $p$)에 집중 (clean 데이터 가정)
- RC Loss: $\mathcal{L}_{RC} = \log(1-p)$ → 쉬운 샘플(높은 $p$)에 집중 (noisy 데이터 가정)

### 3.2 MC 손실의 일반화 메커니즘

MC 손실은 **레이블 정보 없이 인스턴스 레벨의 판별력을 학습**합니다. 이는:

- **노이즈 레이블로부터 독립적인 표현 학습**: 레이블에 의존하지 않고 멀티모달 쌍의 자연적 동시 발생(co-occurrence) 특성을 활용합니다.
- **공통 표현 공간 정규화**: 서로 다른 모달리티를 단위 구(unit sphere) 위에 강제로 매핑함으로써 표현의 균등 분포를 유도하고 과적합을 방지합니다.
- **배치 내 음성 샘플 활용**: 배치 내의 모든 다른 인스턴스를 음성 샘플로 활용하므로, 추가적인 메모리 뱅크나 모멘텀 인코더 없이도 풍부한 학습 신호를 제공합니다.

### 3.3 일반화 성능 관련 핵심 관찰

1. **노이즈 비율 증가에 따른 상대적 성능 우위 증가**: 다른 방법들의 성능이 노이즈 비율 증가에 따라 급격히 하락하는 반면, MRL은 완만한 하락을 보입니다. 예를 들어 Wikipedia Image→Text에서 deep-SM은 노이즈 0.2→0.8 시 0.441→0.178 (-26.3%p) 하락하지만, MRL은 0.514→0.435 (-7.9%p)에 그칩니다.

2. **클래스 수와 노이즈 강건성의 관계**: 클래스 수가 많을수록(더 어려운 태스크) 딥 네트워크의 노이즈 과적합이 심해지며, 이 경우 MRL의 상대적 이점이 더 커집니다. XMediaNet(200 클래스)에서 가장 큰 향상폭을 보이는 것이 이를 뒷받침합니다.

3. **$\mathcal{L}_c$의 노이즈 불변성**: 어블레이션 스터디에서 $\mathcal{L}_c$만 사용하는 경우 모든 노이즈 비율에서 거의 동일한 성능(~0.412)을 보입니다. 이는 $\mathcal{L}_c$가 레이블에 무관하게 **노이즈 비율에 강건한 표현**을 학습함을 의미합니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려 사항

### 4.1 앞으로의 연구에 미치는 영향

**① 멀티모달 노이즈 레이블 연구의 선구자적 역할**
이 논문은 크로스 모달 검색 + 노이즈 레이블이라는 조합을 본격적으로 다룬 초기 연구 중 하나로, 이후 연구들의 벤치마크 실험 설계(대칭 노이즈, 4개 데이터셋, MAP 지표)에 영향을 미칩니다.

**② RC 손실의 확장 가능성**
$\log(1-p)$ 형태의 손실은 이론적으로 어떤 클래스 확률 기반 분류기에도 적용 가능하므로, 단일 모달 학습, 멀티레이블 분류, 세그멘테이션 등 다양한 태스크로 확장될 수 있습니다.

**③ 자기 지도 학습과 노이즈 강건성의 결합**
MC 손실은 자기 지도 대조 학습(self-supervised contrastive learning)을 멀티모달 노이즈 환경에 적용한 선례로, 이후 CLIP, ALIGN 등 대규모 비전-언어 모델의 노이즈 강건성 연구에 시사점을 제공합니다.

**④ 실용적 데이터 수집 패러다임의 변화**
크라우드소싱 등 저비용 레이블링으로 수집된 멀티모달 데이터도 효과적으로 활용할 수 있음을 보여줌으로써, 대규모 멀티모달 데이터 수집의 실용적 가이드라인을 제시합니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 본 논문(MRL, CVPR 2021)과 관련된 주요 최신 연구들과의 비교입니다. **단, 직접 제공된 논문 PDF 외의 내용은 일반적 연구 동향에 기반하며, 세부 수치는 해당 논문을 직접 확인하시기 바랍니다.**

#### 노이즈 레이블 학습 관련 주요 연구

| 연구 | 학회/연도 | 핵심 방법 | MRL과의 차이점 |
|------|-----------|-----------|----------------|
| **DivideMix** (Li et al.) | ICLR 2020 | GMM 기반 clean/noisy 분리 + 반지도학습 | 단일 모달, 복잡한 2단계 훈련 |
| **SELF** (Nguyen et al.) | ICLR 2020 | 자기 앙상블로 노이즈 필터링 | 단일 모달, 추가 모델 필요 |
| **Normalized Loss** (Ma et al.) | ICML 2020 | 손실 함수 정규화 | 단일 모달, 크로스 모달 불가 |

DivideMix의 손실 함수 (참고):

$$\mathcal{L} = \mathcal{L}_X + \lambda_u \mathcal{L}_U + \lambda_r \mathcal{L}_{reg}$$

여기서 $\mathcal{L}_X$는 레이블된 데이터, $\mathcal{L}_U$는 비레이블 데이터에 대한 손실입니다. MRL과 달리 명시적인 clean/noisy 분리 단계가 필요하여 구현이 복잡합니다.

#### 크로스 모달 검색 관련 주요 연구 (2020년 이후)

| 연구 | 학회/연도 | 핵심 방법 | MRL과의 관계 |
|------|-----------|-----------|--------------|
| **CLIP** (Radford et al.) | 2021 | 대규모 이미지-텍스트 대조 학습 | MRL의 MC 손실과 유사한 대조 학습 원리, 훨씬 대규모 |
| **NCR** (Huang et al.) | NeurIPS 2021 | 그래프 기반 노이즈 강건 대응 학습 | 유사 문제 설정, 그래프 구조 활용이 차별점 |
| **DECL** (Qin et al.) | MM 2022 | 엔티티 수준 대조 학습 | 더 세밀한 엔티티 정렬 강조 |

**CLIP의 손실 함수** (개념적으로 MC와 유사):

$$\mathcal{L}_{CLIP} = \frac{1}{2}\left(\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}\right)$$

$$\mathcal{L}_{I \to T} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(\mathbf{z}_i^I, \mathbf{z}_i^T)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(\mathbf{z}_i^I, \mathbf{z}_j^T)/\tau)}$$

MRL의 $\mathcal{L}_c$는 이보다 일반화된 $m$개 모달리티 확장 형태이며, 노이즈 강건성을 명시적으로 목표로 합니다.

### 4.3 향후 연구 시 고려해야 할 점

**① 노이즈 유형의 다양화**
- 대칭 노이즈(symmetric noise)에서 **비대칭 노이즈(asymmetric noise)**, **인스턴스 의존 노이즈(instance-dependent noise)** 로 연구 범위 확장이 필요합니다.
- 실제 크라우드소싱 데이터에서의 노이즈는 종종 **클래스 조건부 비대칭 노이즈**에 해당합니다.

**② 크로스 모달 노이즈 페어링 불일치 처리**
- 논문이 언급한 대로 서로 다른 모달리티의 레이블이 독립적으로 노이즈를 가질 수 있습니다.
- 향후 연구는 **모달리티 간 레이블 일관성(consistency)을 명시적으로 모델링**할 필요가 있습니다.

예를 들어, 이미지-텍스트 페어 $(x^I_j, x^T_j)$에 대해 모달리티 간 레이블 일치 여부를 판단하는 별도 모듈:

$$s_j = \text{sim}(\mathbf{z}_j^I, \mathbf{z}_j^T), \quad \tilde{y}_j = \arg\max_k p(k|x_j^I) \cdot p(k|x_j^T)$$

**③ 대규모 사전 학습 모델(CLIP, ALIGN 등)과의 결합**
- 현재 MRL은 VGG-19, Doc2Vec 등 비교적 오래된 백본을 사용합니다.
- ViT 기반의 멀티모달 사전 학습 모델과 결합하여 파인튜닝하는 방식에서의 노이즈 강건성 연구가 필요합니다.

**④ 동적 $\beta$ 스케줄링**
- 현재 $\beta$는 고정 하이퍼파라미터입니다.
- 훈련 초기에는 $\mathcal{L}_c$ (unsupervised)에 집중하고 점차 $\mathcal{L}_r$ (supervised)의 비중을 높이는 **커리큘럼 학습(curriculum learning)** 방식이 효과적일 수 있습니다:

$$\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min}) \cdot \frac{t}{T}$$

**⑤ 설명 가능성(Explainability)과 노이즈 진단**
- RC 손실이 어떤 샘플을 noisy로 판단하는지에 대한 **명시적인 해석 도구**가 필요합니다.
- 노이즈 샘플 식별 결과를 시각화하거나, 실제 노이즈 레이블과의 일치율을 측정하는 분석이 후속 연구에서 보강되어야 합니다.

**⑥ 3개 이상 모달리티로의 확장 검증**
- 이미지, 텍스트, 오디오, 비디오 등 **3개 이상의 모달리티**가 혼재하는 실제 멀티미디어 데이터에서의 MRL 프레임워크 검증이 필요합니다.
- 이 경우 $\mathcal{L}_c$의 계산 복잡도가 $O(m^2 N^2)$로 증가할 수 있어 효율적인 근사 방법이 요구됩니다.

**⑦ 연속형 노이즈 수준에 대한 적응형 학습**
- 현재 실험은 0.2, 0.4, 0.6, 0.8의 고정 노이즈 비율을 가정하지만, 실제 데이터는 클래스/샘플별로 노이즈 수준이 다를 수 있습니다.
- 노이즈 비율을 데이터 기반으로 추정하는 **적응형 프레임워크** 개발이 필요합니다.

---

## 참고 자료

**주요 논문 (PDF 기반):**
- Hu, P., Peng, X., Zhu, H., Zhen, L., & Lin, J. (2021). **Learning Cross-Modal Retrieval with Noisy Labels.** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2021)*, pp. 5403–5413.

**논문 내 인용된 핵심 참고문헌 (논문 Reference 섹션 기반):**
- Han, B., et al. (2018). Co-teaching: Robust training of deep neural networks with extremely noisy labels. *NeurIPS 2018.* [10]
- Li, J., Socher, R., & Hoi, S. C. H. (2020). DivideMix: Learning with noisy labels as semi-supervised learning. *ICLR 2020.* [28]
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML 2020.* [4]
- Wu, Z., Xiong, Y., Yu, S. X., & Lin, D. (2018). Unsupervised feature learning via non-parametric instance discrimination. *CVPR 2018.* [54]
- Ma, X., et al. (2020). Normalized loss functions for deep learning with noisy labels. *ICML 2020.* [32]
- Mandal, D., & Biswas, S. (2020). Cross-modal retrieval with noisy labels. *IEEE ICIP 2020.* [33]
- Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. *ICLR 2017.* [59]
- Jiang, L., et al. (2018). MentorNet: Learning data-driven curriculum for very deep neural networks on corrupted labels. *ICML 2018.* [19]
