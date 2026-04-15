# Language-driven Semantic Segmentation

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
LSeg는 **언어(텍스트) 임베딩을 활용하여 고정된 라벨 세트 없이도 제로샷(zero-shot) 의미론적 분할을 수행**할 수 있다는 것을 보여줍니다. 텍스트 인코더와 이미지 인코더를 공통 임베딩 공간에서 정렬함으로써, 학습 시 본 적 없는 클래스도 추론 시점에 유연하게 처리할 수 있습니다.

### 주요 기여
| 기여 항목 | 내용 |
|---|---|
| **제로샷 세그멘테이션** | 추가 학습 샘플 없이 미학습 클래스 분할 가능 |
| **유연한 라벨 체계** | 테스트 시 라벨 집합을 임의로 추가·제거·재정렬 가능 |
| **CLIP 기반 텍스트 인코더 활용** | 의미적으로 유사한 레이블이 임베딩 공간에서 인접하게 배치됨 |
| **공간 정규화 블록 도입** | 예측 결과의 공간적 일관성을 유지하면서 해상도 복원 |
| **고정 라벨 세트에서도 경쟁력 있는 성능** | 기존 표준 세그멘테이션 모델 대비 무시할 수 있는 수준의 성능 손실만 발생 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

기존 의미론적 분할 모델의 핵심 한계는 다음과 같습니다:

- **고정 라벨 세트**: 학습 데이터셋에 정의된 수십~수백 개의 클래스만 인식 가능
- **높은 어노테이션 비용**: 픽셀 단위 레이블링은 극도로 노동 집약적
- **미학습 클래스 처리 불가**: 새로운 클래스에 대응하려면 재학습 필요
- **Few-shot 방법의 한계**: 적어도 하나의 레이블된 샘플이 필요

영어에는 수십만 개의 명사가 존재하는데, 기존 모델이 수십~수백 개의 고정 클래스만 다루는 것은 근본적인 제약이라고 논문은 주장합니다.

---

### 2-2. 제안하는 방법 및 수식

#### 전체 프레임워크 구조

$$\text{LSeg: } \underbrace{\text{Text Encoder}}_{\text{CLIP}} \quad + \quad \underbrace{\text{Image Encoder}}_{\text{DPT (ViT)}} \quad \xrightarrow{\text{내적(dot product)}} \quad \text{Segmentation Map}$$

#### (1) 텍스트 인코더

$N$개의 입력 레이블을 공통 임베딩 공간 $\mathbb{R}^C$으로 변환합니다:

$$T_1, T_2, \ldots, T_N \in \mathbb{R}^C$$

- CLIP(Contrastive Language-Image Pre-training)의 사전학습된 텍스트 인코더 사용
- 학습 중 **텍스트 인코더는 동결(freeze)**되며, 이미지 인코더만 업데이트

#### (2) 이미지 인코더

입력 이미지 크기가 $H \times W$이고, 다운샘플링 인수가 $s$일 때:

$$\tilde{H} = \frac{H}{s}, \quad \tilde{W} = \frac{W}{s}$$

픽셀별 밀집 임베딩:

$$I \in \mathbb{R}^{\tilde{H} \times \tilde{W} \times C}$$

픽셀 $(i,j)$의 임베딩을 $I_{ij} \in \mathbb{R}^C$로 표기합니다.

#### (3) Word-Pixel 상관관계 텐서

이미지 픽셀 임베딩과 텍스트 임베딩의 내적으로 상관관계 텐서를 정의합니다:

$$f_{ijk} = I_{ij} \cdot T_k $$

이를 통해 $\tilde{H} \times \tilde{W} \times N$ 크기의 텐서를 생성하며, 픽셀 $(i,j)$에 대한 $N$차원 벡터:

$$F_{ij} = (f_{ij1}, f_{ij2}, \ldots, f_{ijN})^T \in \mathbb{R}^N$$

#### (4) 학습 목적 함수 (Training Objective)

픽셀별 소프트맥스(softmax) 목적 함수를 정의합니다. 픽셀 $(i,j)$의 정답 레이블이 $y_{ij}$일 때, 온도 스케일링된 소프트맥스를 최대화:

$$\sum_{i,j=1}^{H,W} \text{softmax}_{y_{ij}}\left(\frac{F_{ij}}{t}\right) $$

실제 구현에서는 **온도 스케일링된 픽셀별 크로스엔트로피 손실**을 최소화:

$$\mathcal{L} = -\sum_{i,j} \log \frac{\exp(f_{ij,y_{ij}}/t)}{\sum_{k=1}^{N} \exp(f_{ijk}/t)}$$

여기서 온도 파라미터 $t = 0.07$ (Wu et al., 2018; Radford et al., 2021 설정 준수).

#### (5) 공간 정규화 블록 (Spatial Regularization)

메모리 제약으로 낮은 해상도로 예측된 결과를 원본 해상도로 복원할 때, **라벨 순서에 불변(equivariant)**한 연산만 허용:

- **DepthwiseBlock**: Depthwise 컨볼루션 + 비선형 활성화의 연속
- **BottleneckBlock**: Depthwise 컨볼루션 + 레이블 집합에 대한 Max-Pooling 결과 추가

최종적으로 **Bilinear Interpolation**으로 원본 해상도 복원.

> 핵심 제약: 채널 간 상호작용 없이 각 채널(=각 라벨)을 독립적으로 처리해야 함 → 라벨 순서가 바뀌어도 결과에 영향 없음

---

### 2-3. 모델 구조

```
입력 이미지 (H × W)
        ↓
[Image Encoder: DPT (Dense Prediction Transformer)]
  - Backbone: ViT-L/16 또는 ResNet101
  - 출력: I ∈ R^{H̃ × W̃ × C}
        ↓
[Word-Pixel Correlation]
  - f_ijk = I_ij · T_k  →  F ∈ R^{H̃ × W̃ × N}
        ↑
[Text Encoder: CLIP (Frozen)]
  - 입력: N개의 텍스트 레이블
  - 출력: T_1,...,T_N ∈ R^C
        ↓
[Spatial Regularization Block]
  - DepthwiseBlock 또는 BottleneckBlock
  - Bilinear Interpolation (H × W 복원)
        ↓
출력 세그멘테이션 맵 (H × W)
```

#### 학습 설정
- **옵티마이저**: SGD (momentum=0.9)
- **학습률 스케줄러**: Polynomial decay (rate=0.9)
- **배치 크기**: 6 (Quadro RTX 6000 × 6)
- **이미지 인코더 초기화**: ImageNet 사전학습 ViT 또는 ResNet 가중치
- **텍스트 인코더**: 학습 중 동결

---

### 2-4. 성능 향상

#### PASCAL-5 $^i$ (mIoU %)

| 방법 | 백본 | 유형 | 평균 mIoU | FB-IoU |
|---|---|---|---|---|
| ZS3Net | ResNet101 | zero-shot | 38.3 | 57.7 |
| **LSeg** | **ResNet101** | **zero-shot** | **47.4** | **64.1** |
| **LSeg** | **ViT-L/16** | **zero-shot** | **52.3** | **67.0** |
| HSNet | ResNet | 1-shot | 66.2 | 77.6 |

#### COCO-20 $^i$ (mIoU %)

| 방법 | 백본 | 유형 | 평균 mIoU | FB-IoU |
|---|---|---|---|---|
| ZS3Net | ResNet101 | zero-shot | 21.1 | 55.1 |
| **LSeg** | **ResNet101** | **zero-shot** | **23.4** | **57.9** |
| **LSeg** | **ViT-L/16** | **zero-shot** | **27.2** | **59.9** |

#### FSS-1000 (mIoU %)

| 방법 | 백본 | 유형 | mIoU |
|---|---|---|---|
| HSNet | ResNet101 | 1-shot | 86.5 |
| **LSeg** | **ResNet101** | **zero-shot** | **84.7** |
| **LSeg** | **ViT-L/16** | **zero-shot** | **87.8** |

> ⭐ FSS-1000에서 LSeg(ViT-L/16)는 최신 1-shot 방법인 HSNet을 **제로샷으로 능가**

#### ADE20K 고정 라벨 세트 (표준 세그멘테이션)

| 방법 | 백본 | Text Encoder | mIoU (%) |
|---|---|---|---|
| DPT | ViT-L/16 | - | 47.63 |
| **LSeg** | **ViT-L/16** | **RN50×16** | **47.25** |

> DPT 대비 단 **0.38% 손실**만 발생 → 언어 기반 방식이 고정 라벨에서도 경쟁력 있음

---

### 2-5. 한계

1. **오탐(False Assignment)**: 테스트 라벨이 실제 클래스를 포함하지 않을 경우, 임베딩 공간에서 가장 가까운 잘못된 레이블 할당
   - 예: 강아지 이미지에 "toy", "grass"만 제공 시 → 강아지를 "toy"로 분류
2. **다중 레이블 모호성**: 여러 설명이 가능한 픽셀에서 가장 확률 높은 하나만 선택
   - 예: "window"가 있음에도 "house"로 잘못 분류
3. **긍정 샘플만 학습**: 부정 샘플(negative sample) 학습이 없어 분포 외 클래스에 취약
4. **Zero-shot 표준 벤치마크 부재**: 정규화된 zero-shot 평가 프로토콜이 없어 few-shot 벤치마크에서 비교해야 하는 한계

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화의 핵심 메커니즘

LSeg의 일반화 성능은 다음 세 가지 핵심 메커니즘에서 비롯됩니다:

#### (A) 텍스트 임베딩의 의미론적 연속성 활용

CLIP 텍스트 인코더는 의미적으로 유사한 개념을 임베딩 공간에서 인접하게 배치합니다:

$$d(\text{emb}(\text{"dog"}), \text{emb}(\text{"pet"})) \ll d(\text{emb}(\text{"dog"}), \text{emb}(\text{"vehicle"}))$$

이 성질로 인해 학습 중 "dog"를 보았다면, 테스트 시 "pet"이라는 레이블도 유사한 픽셀 영역에 적용 가능합니다.

#### (B) 계층적 레이블 구조 암묵적 지원

논문의 실험에서 확인된 예시:
- "cat" → "furry"로 레이블 교체 시 유사한 영역 분할 (상위 개념으로 일반화)
- "sofa" → "furniture"로 교체 시 더 넓은 영역(소파 + 선반)을 포함하여 분할

$$\text{emb}(\text{"cat"}) \approx \text{emb}(\text{"furry"}) \quad \text{(in semantic direction)}$$

#### (C) 7개 데이터셋 혼합 학습을 통한 도메인 일반화

ADE20K, BDD, Cityscapes, COCO-Panoptic, IDD, Mapillary Vistas, SUN RGBD의 **7개 다양한 데이터셋**을 텍스트 인코더를 통해 통합 학습함으로써:
- 전처리나 재레이블링 없이 각 데이터셋의 원본 레이블 그대로 사용
- 도메인 간 일반화 가능성 증대

#### (D) 미학습 레이블 처리 실험 결과

| 미학습 레이블 | 결과 | 근거 |
|---|---|---|
| "greenery" | 성공적 분할 | "plant"와 임베딩 유사도 높음 |
| "dessert" | "cake" 영역에 정확히 할당 | 상위 카테고리 관계 반영 |
| "furniture" | 소파+선반 모두 포함 | 하위 개념들의 상위 개념으로 일반화 |

### 3-2. 일반화를 위한 Ablation 결과

텍스트 인코더 크기가 클수록 일반화 성능 향상:

| Text Encoder | Embedding Dim | mIoU (ADE20K) |
|---|---|---|
| ViT-B/32 | 512 | 37.83% |
| ViT-B/16 | 512 | 38.69% |
| RN50×4 | 640 | 38.93% |
| **RN50×16** | **768** | **40.36%** |

임베딩 차원이 클수록 더 세밀한 의미 구분이 가능하여 일반화 성능이 향상됩니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4-1. 연구에 미치는 영향

#### (A) Vision-Language 모델의 밀집 예측 작업으로의 확장
LSeg는 CLIP과 같은 이미지-텍스트 정렬 모델이 **이미지 수준**이 아닌 **픽셀 수준** 태스크에도 효과적으로 활용될 수 있음을 최초로 체계적으로 증명했습니다. 이는 후속 연구들(OpenSeg, OVSeg, ODISE 등)의 직접적인 방향을 제시했습니다.

#### (B) 제로샷 세그멘테이션의 새로운 패러다임 정립
기존 생성 모델 기반(ZS3Net 등)의 제로샷 접근법에서 **대규모 사전학습 언어 모델 활용 패러다임**으로의 전환을 이끌었습니다.

#### (C) 다중 데이터셋 통합 학습의 실용적 가능성 제시
텍스트 인코더를 중간 매개로 사용함으로써, 서로 다른 레이블 체계를 가진 여러 데이터셋을 자연스럽게 통합 학습할 수 있는 방법론을 제시했습니다.

### 4-2. 관련 후속 연구 비교 분석 (2020년 이후)

**⚠️ 주의**: 아래의 후속 연구들은 제가 학습한 지식 기반으로 기술하며, 일부 세부 수치는 부정확할 수 있습니다. 반드시 원문 확인을 권장합니다.

| 연구 | 방법 | LSeg 대비 차이점 |
|---|---|---|
| **OpenSeg** (Ghiasi et al., 2022, ECCV) | 그룹화된 세그먼트를 텍스트와 정렬 | 픽셀 대신 세그먼트 단위로 정렬하여 더 정교한 경계 처리 |
| **OVSeg** (Liang et al., 2023, CVPR) | 마스크된 이미지로 CLIP 파인튜닝 | 마스크 영역에 특화된 CLIP 적응으로 성능 향상 |
| **FC-CLIP** (Yu et al., 2023) | 단일 CLIP 모델로 분할 수행 | 별도 이미지 인코더 없이 CLIP 하나로 통합 |
| **ODISE** (Xu et al., 2023, CVPR) | 확산 모델(Diffusion Model) 내부 표현 활용 | 생성 모델의 내부 특성 맵을 세그멘테이션에 활용 |
| **CAT-Seg** (Cho et al., 2023) | 비용 집계(cost aggregation) 방식 | 이미지-텍스트 상관 비용을 공간적으로 집계하여 처리 |
| **SAN** (Xu et al., 2023) | 사이드 어댑터 네트워크 | CLIP 동결 후 경량 어댑터만 학습 |

### 4-3. 앞으로 연구 시 고려할 점

#### (A) 부정 샘플(Negative Sample) 학습의 필요성
현재 LSeg는 긍정 샘플만으로 학습되어, 제공된 레이블 집합에 없는 클래스가 이미지에 존재할 경우 오분류가 발생합니다. 향후 연구에서는:

$$\mathcal{L}_{\text{improved}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{neg}}$$

부정 샘플 페널티 항을 추가하는 방향을 고려해야 합니다.

#### (B) 더 세밀한 Prompt Engineering
단순 레이블 단어 대신 풍부한 텍스트 설명(예: "a photo of a fluffy cat")을 활용하거나, 학습 가능한 프롬프트(Learnable Prompt)를 도입하면 일반화 성능을 더욱 향상시킬 수 있습니다.

#### (C) 표준화된 제로샷 세그멘테이션 벤치마크 구축
논문 자체가 인정하듯, 표준화된 zero-shot 평가 프로토콜이 부재합니다. 공정한 비교를 위한 새로운 벤치마크 구축이 필요합니다.

#### (D) 다중 레이블 할당 문제
하나의 픽셀이 여러 레이블에 해당할 수 있는 경우(예: 창문 = 건물의 일부)를 처리하기 위한 **Soft/Multi-label 분류** 접근법이 필요합니다.

#### (E) 계산 효율성
CLIP + DPT의 조합은 추론 비용이 높습니다. 경량화된 비전-언어 모델(예: 지식 증류, 양자화)과의 결합을 통한 실용적 배포 방안이 필요합니다.

#### (F) 언어 모델의 편향(Bias) 상속 문제
논문의 Ethics Statement에서도 언급하듯, CLIP과 같은 대규모 언어-비전 모델은 학습 데이터의 편향을 내재하고 있으며, 이것이 세그멘테이션 모델에 그대로 상속될 수 있습니다. 편향 완화(Bias Mitigation) 전략 연구가 필요합니다.

---

## 참고 자료

### 주요 참고 논문 (본 PDF에서 직접 인용된 문헌)
1. **Li et al. (2022)** - "Language-driven Semantic Segmentation (LSeg)" - *ICLR 2022* (본 논문, arXiv:2201.03546)
2. **Radford et al. (2021)** - "Learning Transferable Visual Models From Natural Language Supervision (CLIP)" - *ICML 2021*
3. **Ranftl et al. (2021)** - "Vision Transformers for Dense Prediction (DPT)" - *ICCV 2021*
4. **Dosovitskiy et al. (2021)** - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)" - *ICLR 2021*
5. **Bucher et al. (2019)** - "Zero-Shot Semantic Segmentation (ZS3Net)" - *NeurIPS 2019*
6. **Min et al. (2021)** - "Hypercorrelation Squeeze for Few-Shot Segmentation (HSNet)" - *ICCV 2021*
7. **Wu et al. (2018)** - "Unsupervised Feature Learning via Non-Parametric Instance Discrimination" - *CVPR 2018*
8. **Zhou et al. (2019)** - "Semantic Understanding of Scenes through the ADE20K Dataset" - *IJCV 2019*

### 후속 연구 (추가 참고)
9. **Ghiasi et al. (2022)** - "Scaling Open-Vocabulary Image Segmentation with Image-Level Labels (OpenSeg)" - *ECCV 2022*
10. **Liang et al. (2023)** - "Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP (OVSeg)" - *CVPR 2023*

> ⚠️ **정확도 주의**: 후속 연구 비교 분석 섹션(4-2)의 구체적 내용(논문명, 학회, 방법론 세부사항)은 제 학습 데이터 기반이며, 100% 정확성을 보장하기 어렵습니다. 관련 연구를 참고할 경우 반드시 Google Scholar, arXiv, Semantic Scholar 등을 통해 원문을 직접 확인하시길 권장합니다.
