# Audio Tagging by Cross Filtering Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
본 논문은 오디오 태깅(Audio Tagging) 태스크에서 **노이즈 레이블(Noisy Labels) 문제**를 효과적으로 해결하기 위한 새로운 프레임워크인 **CrossFilter**를 제안합니다. 딥러닝 모델은 뛰어난 기억 능력(memorization ability) 때문에 잘못된 레이블에 과적합되기 쉬우며, 이를 극복하기 위해 **다중 오디오 표현(Cross-Representation)**, **노이즈 필터링(Noise Filtering)**, **멀티태스크 학습(Multi-Task Learning)** 을 결합한 통합 프레임워크를 제시합니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **① Cross-Representation** | 서로 다른 오디오 표현(Logmel, CQT 등)을 두 네트워크에 입력하여 피어 네트워크 간 다양성(diversity) 극대화 |
| **② Noise Filtering (NF)** | 두 네트워크가 협력하여 점진적으로 올바르게 레이블된 데이터를 선별 (노이즈 비율 사전 지식 불필요) |
| **③ Multi-Task Learning (MTL)** | 정제 서브셋과 노이즈 서브셋에 서로 다른 손실 함수를 적용하여 전체 데이터셋을 활용하고 일반화 성능 향상 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

대규모 오디오 데이터셋 구축 시 발생하는 **노이즈 레이블 문제**입니다. 예를 들어, AudioSet은 527개 클래스 중 약 18%에서 레이블 오류율이 50% 이상으로 추정됩니다. 실제 환경에서는 **bi-quality 데이터셋** (소량의 검증된 Curated 데이터 + 대량의 노이즈 데이터)이 일반적입니다.

기존 방법의 한계:
- **노이즈 전이 행렬(Noise Transition Matrix)** 추정: 클래스 수가 많을수록 정확한 추정이 어려움
- **Co-teaching**: 피어 네트워크 간 다양성이 랜덤 초기화에만 의존하여 훈련 중 합의(consensus)에 수렴
- **WeblyNet**: 동일 입력 표현의 두 가지 병목 특징(bottleneck features) 사용으로 다양성이 불충분

---

### 2-2. 제안 방법 및 수식

#### (A) 데이터셋 정의

- 정제 데이터셋: $\mathcal{C} = \{(x_c, y_c) \mid c = 1, 2, \ldots, C\}$
- 노이즈 데이터셋: $\mathcal{N} = \{(x_n, y_n) \mid n = 1, 2, \ldots, N\}$
- 목표: $\mathcal{C} \cup \mathcal{N}$으로부터 학습하여 미지의 인스턴스를 분류

---

#### (B) Cross-Representation

두 네트워크 $M^1$, $M^2$는 각각 서로 다른 오디오 표현을 입력으로 받습니다:
- $M^1$: **Logmel** 표현 사용 ($Rep^1$)
- $M^2$: **CQT** 표현 사용 ($Rep^2$)

실험 결과, Logmel+CQT 조합이 가장 우수한 성능을 보였습니다 (mAP@3: 92.43%).

---

#### (C) Noise Filtering 알고리즘

핵심 직관: *네트워크의 예측이 주어진 레이블과 일치한다면, 해당 레이블은 올바를 가능성이 높다.*

**Algorithm 1 (Noise Filtering)**:

```
초기화: C¹=C²=C, N¹=N²=N, k=0
for j = 1 to EPOCH:
    M¹, M² 각각 미니배치로 훈련
    N = Shuffle(N)
    for (x_n, y_n) ∈ N:
        ŷ¹_n = M¹(Rep¹(x_n))
        if Agree(y_n, ŷ¹_n) and δ(y_n, C¹) < k:
            N² ← N² - {(x_n, y_n)}  (크로스 방향!)
            C² ← C² ∪ {(x_n, y_n)}
        ŷ²_n = M²(Rep²(x_n))
        if Agree(y_n, ŷ²_n) and δ(y_n, C²) < k:
            N¹ ← N¹ - {(x_n, y_n)}
            C¹ ← C¹ ∪ {(x_n, y_n)}
    k = Step(j)
```

**핵심 설계 원리**:
- $M^1$이 선별한 데이터 → $\mathcal{C}^2$에 추가 (자기 자신이 아닌 피어 네트워크용)
- 클래스 불균형 방지: 클래스 $y_n$에 대해 최대 $k$개만 선택
  $$\delta(y_n, \mathcal{C}^r) < k$$
- 새로 추가되는 pseudo curated 데이터의 상한:
  $$\min(N, J \times k), \quad J = \text{클래스 수}$$
- 커리큘럼 학습 원리에 따라 $k$를 에폭에 따라 선형 증가

---

#### (D) Multi-Task Learning 손실 함수

**정제 데이터셋 $\mathcal{C}^r$에 대한 손실 ($loss_c$)**:

단일 레이블 분류 (CCE):

$$loss_{c(\text{CCE})} = -\sum_{j=1}^{J} y_{ij} \log(\phi_j(\boldsymbol{f}(\boldsymbol{x}_i; \boldsymbol{\theta_c}))) \tag{1}$$

다중 레이블 분류 (BCE):

$$loss_{c(\text{BCE})} = -\sum_{j=1}^{J} y_{ij} \log(\sigma_j(\boldsymbol{f}(\boldsymbol{x}_i; \boldsymbol{\theta_c}))) - \sum_{j=1}^{J} (1 - y_{ij}) \log(\sigma_j(1 - \boldsymbol{f}(\boldsymbol{x}_i; \boldsymbol{\theta_c}))) \tag{2}$$

여기서 $\phi_j$는 Softmax, $\sigma_j$는 Sigmoid 함수의 $j$번째 원소.

**노이즈 데이터셋 $\mathcal{N}^r$에 대한 손실 ($loss_n$) — 노이즈 강건 손실**:

$$loss_n = \frac{1 - \left(\sum_{j=1}^{J} y_{ij} \phi_j(\boldsymbol{f}(\boldsymbol{x}_i; \boldsymbol{\theta_n}))\right)^q}{q}, \quad q \in (0, 1] \tag{3}$$

- $q = 1$: 평균 절대 오차(MAE)와 동일 → 노이즈에 강건하나 최적화 어려움
- $q \to 0$: 교차 엔트로피(CCE)와 동일 → 최적화는 쉬우나 노이즈에 취약
- $q \in (0,1)$: 두 장점을 절충 (실험에서 $q=0.5$ 사용)

**전체 위험 함수**:

$$\mathcal{R}_{\mathcal{C} \cup \mathcal{N}} = \mathbb{E}_{\mathcal{C}}[loss_c] + \lambda \mathbb{E}_{\mathcal{N}}[loss_n] \tag{4}$$

여기서 $\lambda$는 검증 세트에서 그리드 서치로 결정되는 하이퍼파라미터.

---

### 2-3. 모델 구조

```
[Bi-quality Data]
        │
        ├──────────────────────┬──────────────────────┐
        │                     │                      │
   {C¹, N¹}             {C², N²}               
   [Logmel Rep¹]        [CQT Rep²]             
        │                     │                
   [MobileNetV2]         [MobileNetV2]          
   [Max-Mean Pooling]    [Max-Mean Pooling]     
        │                     │                
        └──────┬───────────────┘               
               │
        [Noise Filtering]
        (크로스 방향 레이블 선별)
               │
        ┌──────┴──────┐
   Curated(C^r)   Noisy(N^r)
        │              │
   FC(θ_c)         FC(θ_n)
        │              │
    loss_c          loss_n
    (CCE/BCE)       (L_q)
```

**세부 구현**:
- 백본: MobileNetV2 (width multiplier=1)
- 풀링: frequency 축 → Max Pooling, time 축 → Mean Pooling
- 입력 크기: $64 \times 800$
- 데이터 증강: SpecAugment + MixUp ($\gamma \sim \text{Beta}(\alpha, \alpha), \alpha=1$)
- 추론 시: 4초 세그먼트 5개 평균 + 두 모델 출력 확률 합산
- 최적화: Adam, Cosine Annealing LR ($5\times10^{-5} \to 5\times10^{-4} \to 5\times10^{-6}$, 300 에폭)

---

### 2-4. 성능 향상

#### FSDKaggle2018 (mAP@3, %)

| 방법 | mAP@3 |
|------|-------|
| Baseline | 69.43 |
| Surrogate Loss | 90.87 |
| Co-teaching | 92.50 |
| Iterative Training (ensemble) | 94.96 |
| Loss Masking (ensemble) | 95.38 |
| **CrossFilter (단일 모델)** | **95.59** |

#### FSDKaggle2019 (lwlrap)

| 방법 | lwlrap |
|------|-------|
| Baseline | 0.5460 |
| Co-teaching | 0.7071 |
| **CrossFilter** | **0.7195** |

#### 각 컴포넌트별 기여 (FSDKaggle2018)

| 설정 | M¹(Logmel) | M²(CQT) | M¹+M² |
|------|-----------|---------|-------|
| NF 없음 | 93.04 | 92.87 | 94.08 |
| **NF 적용** | **94.68** | **94.50** | **95.59** |

| MTL 설정 | Logmel mAP@3 |
|---------|-------------|
| MTL 없음 (CCE) | 92.32 |
| MTL (lossₙ=CCE) | 92.71 |
| MTL (lossₙ=Lq) | **93.04** |

---

### 2-5. 한계

1. **단기 음향 이벤트 인식 어려움**: Tap, Chirp-tweet, Fireworks 등 짧고 순간적인 소리는 여전히 낮은 성능 (클래스 불균형 + 시간 불균형 문제)
2. **세밀한 클래스 혼동**: Female-Speech의 5.3%가 Male-Speech로, Child-Speech의 12.6%가 Crowd로 오분류
3. **노이즈 비율 추정 불필요하나 $\lambda$, $q$ 하이퍼파라미터 튜닝 필요**
4. **도메인 불일치**: FSDKaggle2019에서 Curated(Freesound)와 Noisy(Flickr videos) 간 도메인 차이 존재
5. **확장성**: 현재 두 가지 표현(Logmel+CQT)만 사용; 세 가지 이상 표현으로의 확장 연구 미흡

---

## 3. 모델의 일반화 성능 향상과 관련된 내용

### 3-1. MTL을 통한 일반화 향상

논문에서 MTL은 일반화 성능 향상의 핵심 동력입니다. 두 가지 근거:

**근거 ①**: 정제 데이터와 노이즈 데이터의 **분포 불일치** 문제
- 단일 분류기로 훈련하면 노이즈 데이터의 영향으로 정제 데이터에 잘 맞지 않음
- 별도 분류기 $f(\cdot; \boldsymbol{\theta_c})$와 $f(\cdot; \boldsymbol{\theta_n})$를 사용하여 이 문제 해소

**근거 ②**: **데이터 활용 극대화**
- Co-teaching 등 기존 선택적 샘플링 방법은 노이즈 데이터를 폐기 → 과적합 위험
- MTL은 전체 $\mathcal{C} \cup \mathcal{N}$을 활용하면서 손실 함수로 영향을 차별화

### 3-2. Cross-Representation을 통한 일반화

서로 다른 오디오 표현은 **상보적(complementary) 정보**를 제공합니다:

| 표현 | 주요 특성 |
|------|---------|
| Logmel | 저주파 강조, 연속 음향에 강함 |
| CQT | 저주파 해상도 우수, 음악적 특성 포착 |
| MFCC | 컴팩트한 표현, 음성 특징에 효과적 |
| Spec | 중·고주파 강조 |

Logmel과 CQT의 조합이 최고 성능을 보인 이유는 두 표현이 오디오 신호의 **서로 다른 주파수 특성**을 강조하기 때문입니다.

### 3-3. Curriculum Learning 방식의 일반화

$k$를 점진적으로 증가시키는 Noise Filtering은 **커리큘럼 학습** 원리를 따릅니다:
$$k = \text{Step}(j) \propto j \text{ (에폭에 선형 비례)}$$

신뢰할 수 있는 데이터부터 학습 → 점진적으로 불확실한 데이터 포함 → **나쁜 지역 최솟값(bad local optimum) 회피**

실험에서 처음부터 전체 $\mathcal{C}^r$로 학습 시(Dataset Filtering) 성능이 저하됨을 확인:
- FSD-2018: 95.07 vs **95.59** (CrossFilter)
- FSD-2019: 0.7064 vs **0.7195** (CrossFilter)

### 3-4. 클래스 균형 유지를 통한 일반화

$\delta(y_n, \mathcal{C}^r) < k$ 조건으로 클래스별 최대 $k$개만 선택함으로써 **클래스 불균형** 문제를 완화하고 모델이 특정 클래스에 편향되지 않도록 합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 앞으로의 연구에 미치는 영향

#### (A) 오디오 AI 분야

1. **Multi-Representation Learning의 표준화**: CrossFilter는 단순 앙상블이 아닌, 노이즈 강건성을 위한 다중 표현 활용의 새로운 패러다임을 제시합니다. 향후 오디오 분야에서 표현 조합 선택이 주요 설계 요소로 부각될 것입니다.

2. **DCASE 챌린지 방법론 영향**: FSDKaggle2018에서 앙상블 모델을 뛰어넘는 단일 프레임워크로서, DCASE 2021 이후 챌린지에서 Cross-representation 기반 노이즈 필터링 전략이 기준선(baseline)으로 인용될 가능성이 높습니다.

3. **오디오 데이터셋 구축 전략**: NF를 **데이터셋 필터링 도구**로 활용 가능 → 대규모 웹 크롤링 데이터의 품질 개선에 응용 가능

#### (B) 노이즈 레이블 학습(LNL) 분야

1. **도메인 다양성(Domain Diversity)의 중요성 부각**: 기존 Co-teaching이 랜덤 초기화로 다양성을 확보하려 했던 것과 달리, CrossFilter는 **입력 표현 수준의 다양성**이 더 효과적임을 실증합니다. 이는 컴퓨터 비전, NLP 등 다른 도메인의 LNL 연구에도 영향을 줄 수 있습니다.

2. **노이즈 비율 불필요 프레임워크**: 실용적 환경에서 노이즈 비율을 알 수 없는 경우가 많으므로, 이 접근법은 실용성 높은 연구 방향을 제시합니다.

#### (C) 멀티태스크 학습 분야

$\mathcal{R}\_{\mathcal{C} \cup \mathcal{N}} = \mathbb{E}\_{\mathcal{C}}[loss_c] + \lambda \mathbb{E}_{\mathcal{N}}[loss_n]$ 형태의 **이중 손실 전략**은 노이즈 데이터를 버리지 않고 활용하는 방법론으로, 데이터 효율성(data efficiency) 연구에 기여합니다.

---

### 4-2. 앞으로 연구 시 고려할 점

#### (A) 기술적 개선 방향

1. **세 가지 이상의 표현으로 확장**:
   - 현재는 Logmel + CQT 두 가지만 사용
   - Spec, MFCC, raw waveform(end-to-end) 등 추가 표현의 체계적 조합 탐색
   - 표현 조합의 **자동 탐색(AutoML/NAS)** 적용 가능성

2. **동적 하이퍼파라미터 조정**:
   - $\lambda$, $q$를 훈련 중 동적으로 조정하는 메커니즘 연구 필요
   - 예: 에폭에 따른 $q$의 점진적 변화 ($q: 0 \to 1$)

3. **Noise Filtering의 신뢰도 정량화**:
   - 현재 Agree 함수는 단순 최고 신뢰도 클래스와 레이블 일치 여부만 확인
   - 예측 확률의 **불확실성(uncertainty)**을 활용한 더 정교한 선별 기준 도입
   - 예: Bayesian Deep Learning 기반 신뢰 구간 활용

4. **단기·순간 음향 이벤트 처리**:
   - 시간 불균형 문제 해결을 위한 **시간 가중 손실 함수** 도입
   - Attention 메커니즘으로 단기 음향 이벤트에 집중하는 구조 탐색

5. **도메인 불일치 문제**:
   - FSDKaggle2019에서 Curated(Freesound)와 Noisy(Flickr) 간 도메인 차이
   - **Domain Adaptation** 또는 **Domain Adversarial Training** 결합 연구

6. **더 강력한 백본 탐색**:
   - 현재 MobileNetV2 사용 → EfficientNet, Transformer 기반 모델(AST, HTS-AT) 적용 시 추가 성능 향상 예상

#### (B) 방법론적 확장

7. **Semi-supervised Learning과의 결합**:
   - 노이즈 필터링으로 선별된 pseudo curated 데이터를 semi-supervised 프레임워크에 통합

8. **Self-supervised Pre-training 활용**:
   - BYOL, SimCLR 등 자기지도학습으로 사전학습된 표현을 CrossFilter에 결합하여 초기 필터링 품질 향상

9. **Fine-grained 분류 강화**:
   - 계층적 레이블 구조(AudioSet ontology)를 활용한 **계층적 손실 함수** 도입

10. **확장성 연구**:
    - AudioSet 전체(527 클래스, 5000시간)와 같은 대규모 데이터셋에서의 검증
    - 분산 학습(distributed training) 환경에서의 적용 가능성

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 내용은 제공된 논문(2020년 7월)에 직접 언급되지 않은 2020년 이후 연구입니다. 본 논문에서 인용된 문헌 범위를 벗어나므로, **일반적으로 알려진 연구 동향**을 기반으로 작성합니다. 구체적인 수치나 세부 내용은 원 논문을 직접 확인하시기 바랍니다.

### 5-1. Transformer 기반 오디오 태깅 연구

**Audio Spectrogram Transformer (AST)** (Gong et al., INTERSPEECH 2021):
- Vision Transformer(ViT)를 오디오에 적용
- AudioSet에서 기존 CNN 기반 방법 대비 성능 향상
- CrossFilter와의 비교 관점: AST는 노이즈 레이블 문제를 직접 다루지 않으나, 강력한 표현 학습 능력으로 노이즈에 어느 정도 강건할 수 있음
- **결합 가능성**: CrossFilter의 백본을 AST로 교체하면 추가 성능 향상 기대

**HTS-AT (Kong et al., ICASSP 2022)**:
- Swin Transformer를 오디오에 적용
- 시간-주파수 패치 기반 계층적 표현 학습

### 5-2. 노이즈 레이블 학습의 최신 발전

**DivideMix (Li et al., ICLR 2020)**:
- GMM(Gaussian Mixture Model)을 사용하여 clean/noisy 샘플을 확률적으로 분리
- MixMatch를 활용한 semi-supervised 학습 결합
- CrossFilter와 비교: DivideMix는 컴퓨터 비전에 초점; CrossFilter는 오디오 특화 + Cross-Representation이 강점

**ELR (Early-Learning Regularization)** (Liu et al., NeurIPS 2020):
- 딥러닝 모델의 초기 학습(early learning) 현상을 활용한 정규화
- CrossFilter의 커리큘럼 학습 원리와 개념적으로 유사

**UNICON (Karim et al., CVPR 2022)**:
- 대조 학습(Contrastive Learning)과 semi-supervised 학습을 결합한 통합 프레임워크
- **CrossFilter 대비**: UNICON은 레이블 분포 추정에 더 정교하나, 오디오 도메인 특화 표현 다양성 활용 없음

### 5-3. 오디오 도메인 자기지도 학습

**BYOL-A (Niizumi et al., 2021)**:
- 오디오 도메인의 자기지도 대조 학습
- 노이즈 레이블 환경에서 pre-training으로 CrossFilter 초기화에 활용 가능

**AudioMAE (He et al., 2022)**:
- Masked Autoencoder를 오디오에 적용
- 대규모 비지도 학습으로 강건한 표현 획득

### 5-4. 종합 비교 테이블

| 연구 | 연도 | 노이즈 처리 | 오디오 특화 | 표현 다양성 | 데이터 효율 |
|------|------|-----------|-----------|-----------|-----------|
| CrossFilter | 2020 | ✅ (NF+MTL) | ✅ | ✅ (Multi-Rep) | ✅ (MTL) |
| DivideMix | 2020 | ✅ (GMM) | ❌ (CV) | ❌ | ✅ (Semi-SL) |
| AST | 2021 | ❌ | ✅ | ❌ | ❌ |
| UNICON | 2022 | ✅ | ❌ (CV) | ❌ | ✅ |
| AudioMAE | 2022 | ❌ (간접) | ✅ | ❌ | ✅ |

**CrossFilter의 지속적 강점**: 오디오 도메인 특화 표현(Logmel, CQT)을 이용한 피어 네트워크 다양성 확보는 Transformer 기반 방법들에도 적용 가능한 아이디어이며, 노이즈 레이블 오디오 태깅에서 여전히 유효한 접근법입니다.

---

## 참고 자료

**주 논문:**
- Zhu, B., Xu, K., Kong, Q., Wang, H., & Peng, Y. (2020). "Audio Tagging by Cross Filtering Noisy Labels." *arXiv:2007.08165v1*

**논문 내 인용 문헌 (주요):**
- Han, B. et al. (2018). "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS 2018*
- Kumar, A. et al. (2019). "Learning sound events from webly labeled data." *IJCAI 2019*
- Fonseca, E. et al. (2019). "Learning sound event classifiers from web audio with noisy labels." *ICASSP 2019*
- Zhang, Z. & Sabuncu, M. (2018). "Generalized cross entropy loss for training deep neural networks with noisy labels." *NeurIPS 2018*
- Sandler, M. et al. (2018). "MobileNetV2: Inverted residuals and linear bottlenecks." *CVPR 2018*
- Park, D.S. et al. (2019). "SpecAugment: A simple data augmentation method for automatic speech recognition." *Interspeech 2019*
- Bengio, Y. et al. (2009). "Curriculum learning." *ICML 2009*
- Fonseca, E. et al. (2018/2019). FSDKaggle2018/2019 Dataset papers. *DCASE Workshop*

**2020년 이후 비교 참고 연구 (논문 원본에 없는 내용, 일반 지식 기반):**
- Gong, Y. et al. (2021). "AST: Audio Spectrogram Transformer." *INTERSPEECH 2021*
- Li, J. et al. (2020). "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." *ICLR 2020*
- Kong, Q. et al. (2022). "HTS-AT: A Hierarchical Token-Semantic Audio Transformer." *ICASSP 2022*
