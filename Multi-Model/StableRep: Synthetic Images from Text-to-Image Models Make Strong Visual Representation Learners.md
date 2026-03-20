# StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
StableRep은 **텍스트-이미지 생성 모델(Stable Diffusion)로 생성한 합성 이미지만으로도 실제 이미지에 필적하거나 이를 능가하는 시각적 표현(visual representation)을 학습할 수 있다**는 것을 실증적으로 보여준다. 생성 모델을 **무한한 데이터 샘플링이 가능한 "데이터 엔진"**으로 재해석하며, 동일한 텍스트 프롬프트에서 생성된 다수의 이미지를 양성 쌍(positive pairs)으로 활용하는 **다중 양성 대조 학습(multi-positive contrastive learning)** 방법을 제안한다.

### 주요 기여 4가지
1. **합성 이미지의 효과 발견**: Stable Diffusion에서 적절한 classifier-free guidance scale을 설정하면, 합성 이미지로 훈련한 자기지도 학습 모델이 동일 규모의 실제 이미지로 훈련한 모델과 동등하거나 더 우수한 성능을 달성한다.
2. **StableRep 방법론 제안**: 동일 프롬프트에서 생성된 다수의 이미지를 양성으로 활용하는 multi-positive contrastive loss를 제안한다.
3. **높은 정확도 달성**: 합성 이미지만으로 ViT-B/16에서 ImageNet 선형 평가 76.7% 정확도를 달성한다.
4. **데이터 효율성**: 언어 감독(language supervision)을 추가한 StableRep+(20M 합성 이미지)이 CLIP(50M 실제 이미지)보다 더 높은 정확도를 달성, **5배의 캡션 효율성**을 보인다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

현대 비전 모델은 대규모 실제 이미지 데이터셋에 크게 의존한다. 그러나 실제 데이터 수집은 비용이 높고, 노이즈가 많으며, 사회적 편향을 반영할 수 있고, 도메인 갭 문제가 존재한다. 본 논문은 다음 질문에 답하고자 한��:

> **"텍스트-이미지 생성 모델의 합성 이미지를 실제 이미지 대신 사용하여 강력한 시각적 표현을 학습할 수 있는가?"**

### 2.2 제안하는 방법

#### (A) Stable Diffusion을 활용한 합성 이미지 생성

Stable Diffusion은 latent space에서 확산 과정을 수행하는 denoising diffusion probabilistic model이다. **Classifier-free guidance**를 통해 조건부 및 비조건부 스코어 추정치를 선형 결합한다:

$$\tilde{\epsilon}(\mathbf{t}, \mathbf{z}_\lambda) = w \cdot \epsilon(\mathbf{t}, \mathbf{z}_\lambda) + (1 - w) \cdot \epsilon(\mathbf{z}_\lambda) \tag{1}$$

여기서 $w$는 guidance scale로, 생성 이미지의 **다양성(diversity)과 품질(quality) 사이의 트레이드오프**를 제어한다. 기존 image-text 데이터셋(CC3M, CC12M, RedCaps)의 텍스트 캡션 $\{\mathbf{t}\_i\}_{i=1}^{N}$을 사용해 합성 이미지를 생성한다.

**핵심 발견**: 최적의 guidance scale $w$는 목적에 따라 다르다:
- SimCLR의 경우: $w \approx 8$ – $10$이 최적
- MAE의 경우: $w \approx 6$이 최적  
- StableRep의 경우: 작은 $w \in \{2, 3\}$이 최적 (더 큰 intra-caption 변이를 유도)
- CLIP의 경우: $w = 2$가 최적 (하지만 실제 이미지 대비 성능 하락)
- FID 기준 최적: $w = 2$ (표현 학습 최적값과 상이)

#### (B) Multi-Positive Contrastive Loss (핵심 방법론)

동일 텍스트 프롬프트에서 서로 다른 latent noise $\mathbf{z}$를 사용하여 생성된 다수의 이미지들은 **유사한 시각적 의미론(visual semantics)**을 공유하므로, 이들을 서로에 대한 다중 양성 샘플로 활용한다.

**Step 1: Contrastive categorical distribution** — 인코딩된 앵커 샘플 $\mathbf{a}$와 후보 집합 $\{\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_K\}$에 대해:

$$\mathbf{q}_i = \frac{\exp(\mathbf{a} \cdot \mathbf{b}_i / \tau)}{\sum_{j=1}^{K} \exp(\mathbf{a} \cdot \mathbf{b}_j / \tau)} \tag{2}$$

여기서 $\tau \in \mathbb{R}_+$는 온도 하이퍼파라미터, $\mathbf{a}$와 모든 $\mathbf{b}$는 $\ell_2$ 정규화된 벡터이다.

**Step 2: Ground-truth distribution** — 동일 캡션에서 생성된 이미지끼리 match되는 것으로 정의:

$$\mathbf{p}_i = \frac{\mathbb{1}_{\text{match}}(\mathbf{a}, \mathbf{b}_i)}{\sum_{j=1}^{K} \mathbb{1}_{\text{match}}(\mathbf{a}, \mathbf{b}_j)} \tag{3}$$

여기서 $\mathbb{1}_{\text{match}}(\cdot, \cdot)$는 앵커와 후보가 동일 캡션에서 생성되었는지를 나타내는 indicator function이다.

**Step 3: Multi-positive contrastive loss** — ground-truth 분포 $\mathbf{p}$와 contrastive 분포 $\mathbf{q}$ 간의 크로스엔트로피:

$$\mathcal{L} = H(\mathbf{p}, \mathbf{q}) = -\sum_{i=1}^{K} \mathbf{p}_i \log \mathbf{q}_i \tag{4}$$

이 손실 함수는 기존 SimCLR의 single-positive contrastive loss의 **일반화된 형태**로, 단일 양성 쌍에서는 $\mathbf{p}$가 one-hot 벡터로 환원된다. Supervised contrastive learning (Khosla et al., 2020)과 유사하나, **클래스 레이블 없이 동일 캡션에서 생성된 이미지만을 매칭 기준**으로 사용한다는 핵심 차이가 있다.

#### (C) StableRep+ (언어 감독 추가)

StableRep에 CLIP 스타일의 언어 감독을 추가:

$$\mathcal{L}_{\text{StableRep+}} = \mathcal{L}_{\text{StableRep}} + 0.5 \cdot (\mathcal{L}_{i2t} + \mathcal{L}_{t2i})$$

여기서 $\mathcal{L}\_{i2t}$, $\mathcal{L}_{t2i}$는 각각 image-to-text, text-to-image contrastive loss로, 식 (4)로 기술된다.

### 2.3 모델 구조

| 구성 요소 | 세부 사항 |
|---------|--------|
| **이미지 인코더** | ViT-B/16 또는 ViT-L/16 (Vision Transformer) |
| **Projection Head** | 3-layer MLP (hidden dim: 4096, output dim: 256), BatchNorm 적용 |
| **텍스트 인코더** (CLIP/StableRep+) | 12-layer Transformer (width: 512, heads: 8) |
| **생성 모델** | Stable Diffusion v1.5 (50 DDIM steps) |
| **배치 구성** | $n \times m = 8192$ ($n$: 캡션 수, $m = 6$: 캡션당 이미지 수) |
| **사전 생성** | 각 텍스트 프롬프트당 10개 이미지 생성, 매 iteration에서 6개 랜덤 샘플링 |

### 2.4 성능 향상

#### ImageNet 선형 평가 (ViT-B/16, CC12M, 35 epochs)

| 방법 | 데이터 | Top-1 Accuracy |
|-----|------|---------------|
| SimCLR | Real | 61.5% |
| CLIP | Real | 70.3% |
| SimCLR | Synthetic | 63.7% (+2.2%) |
| CLIP | Synthetic | 67.8% (−2.6%) |
| **StableRep** | **Synthetic** | **73.5% (+3.2% vs Real CLIP)** |

#### 장기 훈련 + 스케일링 결과

| 설정 | ImageNet Linear Acc. |
|-----|---------------------|
| StableRep, CC12M, 35ep | 72.8% |
| StableRep, CC12M, 105ep | 75.7% |
| StableRep, RedCaps, 105ep | **76.7%** |
| ViT-L/16, CC12M, 35ep | 74.7% |

#### 데이터 효율성 (LAION 하위 집합)

StableRep+(10M 캡션, 20M 합성 이미지) > CLIP(50M 캡션, 50M 실제 이미지): **5배 캡션 효율, 2.5배 이미지 효율**

#### 다양한 다운스트림 태스크
- **11개 세분화 분류 데이터셋**: StableRep이 11개 전체에서 최고 성능
- **Few-shot (5-way 5-shot)**: 10개 중 9개 데이터셋에서 최고 성능 (평균 89.8%)
- **ADE20k 의미 분할**: StableRep(CC12M, 105ep) 49.4 mIoU > MAE(IN1k Real) 48.1 mIoU
- **공정성(FairFace)**: 합성 데이터 훈련 시 worst-class accuracy 개선 (0.3% → 10.0% on CC12M)

### 2.5 한계

1. **이미지 생성 속도**: A100 GPU에서 약 0.8초/이미지로 느림 → 온라인 합성 훈련 불가
2. **의미 불일치(Semantic mismatch)**: 텍스트 프롬프트와 생성 이미지 간 불일치 존재 (세분화 클래스에서 특히 문제)
3. **CLIP 성능 하락**: 합성 이미지로 CLIP 훈련 시 실제 이미지 대비 성능 하락 (zero-shot: 34.9% vs 40.2%)
4. **편향 문제**: mode collapse로 인한 "프로토타입" 이미지 편향, 지리적 편향 잔존
5. **이론적 이해 부족**: 합성 이미지가 실제 이미지보다 우수한 표현을 학습하는 이유에 대한 완전한 이해 부재
6. **ViT-L 훈련 불안정성**: BatchNorm 사용으로 인해 loss가 NaN으로 폭발하는 문제
7. **이미지 귀속(attribution)**: 합성 데이터 사용 시 저작권/귀속 문제

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능이 향상되는 메커니즘

#### (a) 생성 모델의 제어 가능성(Controllability)

합성 데이터는 guidance scale $w$, 텍스트 프롬프트, latent noise $\mathbf{z}$ 등을 통해 **샘플링의 정밀한 제어**가 가능하다. 이는 실제 데이터에서는 불가능한 수준의 **데이터 구성 유연성**을 제공한다:

- **Intra-caption invariance**: 동일 캡션에서 다양한 이미지를 생성함으로써, 기존 self-supervised learning의 intra-image invariance(augmentation 기반)를 넘어서는 **더 의미론적 수준의 불변성**을 학습
- **Guidance scale 조절**: StableRep에서 작은 $w$(2~3)는 생성 이미지 간 더 큰 변이를 유발하여, 모델이 **더 강한 의미론적 불변성**을 학습하도록 강제

#### (b) 생성 모델의 일반화 능력 활용

생성 모델 자체가 훈련 데이터 너머로 일반화할 수 있으므로, 실제 데이터보다 **더 풍부한(richer) 합성 훈련 세트**를 제공할 수 있다.

#### (c) 다중 양성 쌍의 효과

- 동일 캡션에서 생성된 이미지들은 **시각적 스타일, 구성, 색상 등은 다르지만 의미적으로 일관**됨
- 이는 전통적 augmentation(crop, color jitter 등)보다 **더 풍부하고 의미론적인 양성 쌍**을 구성
- 표 5a에서 캡션당 이미지 수 $l$을 늘릴수록 성능 향상 (1개: 61.2% → 8개: 66.2%, +4.8%)

### 3.2 일반화 성능의 실증적 증거

1. **다양한 SSL 방법에 대한 범용성**: SimCLR, MAE, DINO, BYOL 등 5가지 방법 중 대부분에서 합성 이미지가 실제 이미지를 능가 (Figure 3)
2. **11개 다양한 도메인 데이터셋**: CIFAR-10/100, Aircraft, Cars, DTD, Flowers, Pets, SUN397, Caltech-101, Food-101, VOC2007 — StableRep이 전체에서 최고 성능 달성
3. **Few-shot 일반화**: 5-way 5-shot에서 10개 중 9개 데이터셋에서 우수, 평균 89.8%
4. **Dense prediction으로의 전이**: ADE20k 의미 분할에서도 MAE(ImageNet Real) 대비 우수
5. **스케일링 행동**: 더 긴 훈련(105ep), 더 큰 모델(ViT-L), 더 많은 캡션 모두에서 성능 향상을 보임 (Table 6)
6. **Random Downsample augmentation**: 저해상도 도메인(CIFAR-10/100)으로의 전이 개선을 위한 기법으로 추가 일반화 성능 향상 (+1.5% average for StableRep)

### 3.3 일반화 성능 향상의 한계와 과제

- CLIP 훈련에서 합성 이미지 사용 시 **zero-shot 성능 하락** → 텍스트-이미지 정렬의 세밀도 부족
- **구성적 이해(Compositionality)**: ARO 벤치마크에서 일관되지 않은 결과
- 모든 개선이 **특정 평가 프로토콜(linear probing, few-shot)에 국한**될 가능성

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

1. **데이터 수집 패러다임의 전환**: 실제 이미지 수집의 필요성을 근본적으로 재고하게 함. 생성 모델을 "무한한 데이터 엔진"으로 활용하는 새로운 패러다임 제시
2. **합성 데이터 품질 연구 촉진**: guidance scale, 프롬프트 설계, 생성 다양성 등이 표현 학습에 미치는 영향에 대한 체계적 연구 필요성 제기
3. **대조 학습의 새로운 방향**: Multi-positive contrastive learning은 생성 모델 외에도 여러 정보 소스에서 양성 쌍을 구성하는 데 적용 가능
4. **비용 효율적 AI 학습**: 데이터 수집·주석 비용 절감, 특히 프라이버시에 민감한 의료·금융 분야에서의 활용 가능성
5. **Foundation model 학습 방법론 변화**: 실제 데이터 의존도를 줄이면서도 높은 표현 품질을 유지할 수 있는 가능성 입증

### 4.2 향후 연구 시 고려할 점

1. **이미지 생성 속도 개선**: 현재 0.8초/이미지(A100)는 온라인 학습에 병목 → 빠른 생성 모델(consistency models, 1-step diffusion 등) 활용 필요
2. **의미적 정렬(Semantic Alignment) 향상**: 텍스트-이미지 불일치 문제 해결을 위한 필터링 또는 검증 메커니즘 필요
3. **편향 및 공정성**: 생성 모델의 편향이 학습된 표현에 전파되는 경로를 이해하고 완화하는 연구 필요
4. **스케일링 법칙 탐구**: 더 큰 생성 모델(SDXL, DALL-E 3 등), 더 많은 캡션, 더 큰 인코더에서의 스케일링 행동 연구
5. **이론적 이해**: 합성 이미지가 실제 이미지보다 우수한 표현을 학습시키는 근본적 원인 규명
6. **하이브리드 접근법**: 실제+합성 데이터 혼합 학습의 최적 비율과 전략 연구
7. **다양한 생성 모델 탐구**: Stable Diffusion 외 다른 생성 모델(DALL-E 3, Imagen, Muse 등)과의 비교 및 조합

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 내용 | StableRep과의 관계 |
|-----|------|---------|-----------------|
| **SimCLR** (Chen et al.) | 2020 | 단일 이미지의 두 augmented view를 대조하는 자기지도 학습 | StableRep의 baseline. StableRep은 SimCLR의 intra-image invariance를 intra-caption invariance로 확장 |
| **BYOL** (Grill et al.) | 2020 | Negative sample 없이 자기지도 학습 | 합성 이미지에서 실제 이미지와 동등한 성능 (Figure 3) |
| **CLIP** (Radford et al.) | 2021 | 대규모 이미지-텍스트 쌍에 대한 대조 학습 | StableRep이 동일 텍스트 셋 + 실제 이미지 사용 CLIP을 능가 |
| **MAE** (He et al.) | 2022 | Masked image modeling으로 자기지도 표현 학습 | 합성 이미지($w=6$)에서 실제 이미지 대비 +4.2% 선형 평가 성능 |
| **DINO** (Caron et al.) | 2021 | Self-distillation 기반 자기지도 학습 | 합성 이미지에서 실제 이미지 대비 +2.4%(CC3M) 향상 |
| **Supervised Contrastive Learning** (Khosla et al.) | 2020 | 클래스 레이블 기반 multi-positive contrastive loss | StableRep의 loss 형태와 유사하나, StableRep은 클래스 레이블 대신 캡션 기반 매칭 사용 |
| **SLIP** (Mu et al.) | 2022 | CLIP + Self-supervised learning 결합 | StableRep+와 유사한 하이브리드 접근이나, StableRep은 합성 데이터 전용 |
| **Fake it till you make it** (Sariyildiz et al.) | 2023 | 합성 ImageNet 클론에서 전이 학습 가능한 표현 학습 | 지도 학습 기반인 반면, StableRep은 자기지도/대조 학습 기반 |
| **Azizi et al.** | 2023 | Diffusion 모델의 합성 데이터로 ImageNet 분류 성능 향상 | 지도 학습 데이터 증강인 반면, StableRep은 표현 사전훈련에 초점 |
| **He et al. (Is Synthetic Data Ready?)** | 2022 | 생성 모델 합성 데이터의 인식 과제 준비도 평가 | 지도 학습 관점이며, StableRep은 자기지도 표현 학습으로 더 긍정적 결과 도출 |
| **Jahanian et al.** | 2021 | 생성 모델의 latent variable 조작을 통한 멀티뷰 표현 학습 | StableRep의 직접적 선행 연구. StableRep은 텍스트 조건 생성으로 확장 |
| **Divide and Contrast** (Tian et al.) | 2021 | Uncurated 데이터에서의 자기지도 학습 | Uncurated 데이터의 도메인 갭 문제를 StableRep이 합성 데이터로 해결 |

### 주요 차별점 정리

- **기존 합성 데이터 연구들** (Azizi et al., Sariyildiz et al., He et al.)은 주로 **지도 학습(supervised learning)** 맥락에서 합성 데이터를 활용한 반면, StableRep은 **자기지도 표현 학습(self-supervised representation learning)** 에서 합성 데이터의 가능성을 최초로 대규모로 입증
- **기존 대조 학습** (SimCLR, MoCo, BYOL)의 양성 쌍이 **동일 이미지의 augmentation**에 기반한 반면, StableRep은 **동일 캡션에서 생성된 서로 다른 이미지**를 양성으로 활용하여 **의미론적 수준의 불변성** 학습
- **CLIP 대비** 합성 이미지만으로도 더 우수한 표현 학습이 가능함을 보여, **실제 이미지-텍스트 쌍 수집의 필요성을 근본적으로 재고**하게 함

---

## 참고자료

1. **Tian, Y., Fan, L., Isola, P., Chang, H., & Krishnan, D.** (2023). "StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners." *NeurIPS 2023*. arXiv:2306.00984v2.
2. **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G.** (2020). "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)." *ICML 2020*.
3. **Radford, A., et al.** (2021). "Learning Transferable Visual Models from Natural Language Supervision (CLIP)." *ICML 2021*.
4. **He, K., et al.** (2022). "Masked Autoencoders Are Scalable Vision Learners (MAE)." *CVPR 2022*.
5. **Khosla, P., et al.** (2020). "Supervised Contrastive Learning." arXiv:2004.11362.
6. **Rombach, R., et al.** (2022). "High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)." *CVPR 2022*.
7. **Caron, M., et al.** (2021). "Emerging Properties in Self-Supervised Vision Transformers (DINO)." *ICCV 2021*.
8. **Grill, J.-B., et al.** (2020). "Bootstrap Your Own Latent (BYOL)." *NeurIPS 2020*.
9. **Azizi, S., et al.** (2023). "Synthetic Data from Diffusion Models Improves ImageNet Classification." arXiv:2304.08466.
10. **Sariyildiz, M. B., et al.** (2023). "Fake It Till You Make It: Learning Transferable Representations from Synthetic ImageNet Clones." *CVPR 2023*.
11. **Jahanian, A., Puig, X., Tian, Y., & Isola, P.** (2021). "Generative Models as a Data Source for Multiview Representation Learning." arXiv:2106.05258.
12. **Mu, N., et al.** (2022). "SLIP: Self-supervision Meets Language-Image Pre-training." *ECCV 2022*.
13. **Ho, J., & Salimans, T.** (2022). "Classifier-Free Diffusion Guidance." arXiv:2207.12598.
14. **Chen, X., Xie, S., & He, K.** (2021). "An Empirical Study of Training Self-Supervised Vision Transformers (MoCo v3)." *ICCV 2021*.
15. **He, R., et al.** (2022). "Is Synthetic Data from Generative Models Ready for Image Recognition?" arXiv:2210.07574.
