# Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
기존 2-stage 개방형 어휘(open-vocabulary) 시맨틱 세그멘테이션에서 **사전학습된 CLIP이 마스킹된 이미지(masked image)에 대해 제대로 작동하지 못하는 것이 성능의 병목(bottleneck)**임을 실증적으로 밝히고, 이를 해결하기 위해 **CLIP을 마스크 이미지에 적응(adapt)시키는 방법론(OVSeg)**을 제안한다.

### 주요 기여 (4가지)
1. **병목 진단**: Oracle mask + 일반 CLIP → 20.1% mIoU vs. 일반 mask + Oracle classification → 66.5% mIoU 실험을 통해 CLIP의 masked image 분류 능력이 핵심 병목임을 증명
2. **다양한 mask-category 쌍 수집**: 이미지-캡션 데이터셋에서 명사를 추출하고 CLIP을 이용해 mask proposal과 매칭함으로써, 고정된 클래스셋(171개)이 아닌 **27K개의 다양한 명사**를 활용하여 CLIP의 일반화 능력을 보존
3. **Mask Prompt Tuning (MPT)**: CLIP의 가중치를 변경하지 않으면서 마스크 영역의 zero token을 학습 가능한 프롬프트로 대체하는 기법 제안
4. **SOTA 달성**: COCO에서 학습 후 ADE20K-150에서 29.6% mIoU (이전 SOTA 대비 +8.5%), 최초로 open-vocabulary generalist 모델이 2017년 supervised specialist 모델 성능에 도달

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**Open-vocabulary semantic segmentation**은 학습 시 보지 못한 임의의 텍스트 카테고리로 이미지를 세그멘테이션하는 과제이다. 기존 2-stage 접근법은:

- **1단계**: Class-agnostic mask proposal 생성 (MaskFormer 등)
- **2단계**: 사전학습된 CLIP으로 masked region 분류

이 파이프라인의 근본적 문제는 **CLIP이 자연 이미지(natural image)로 학습되었기 때문에, 배경이 제거된 masked image에 대해 큰 도메인 갭(domain gap)**이 존재한다는 것이다. 구체적으로:

- CLIP 학습 데이터: 최소한의 augmentation이 적용된 자연 이미지
- 실제 입력: crop + resize + 배경 마스킹으로 인한 대량의 빈(blank) 영역

### 2.2 제안하는 방법

#### (A) 2-Stage 모델 구조

MaskFormer를 세그멘테이션 모델로 사용하되, 클래스 예측 헤드를 CLIP 임베딩 차원의 proposal embedding으로 교체한다.

$K$개 카테고리에 대해 CLIP 텍스트 인코더로 텍스트 임베딩 $\{t_k \mid t_k \in \mathbb{R}^C\}_{k=1,\cdots,K}$를 생성하고, mask embedding $v_i$와의 유사도를 기반으로 분류한다:

$$p_{i,k} = \frac{\exp(\sigma(v_i, t_k) / \tau)}{\sum_{k'} \exp(\sigma(v_i, t_{k'}) / \tau)}$$

여기서 $\sigma(\cdot, \cdot)$는 코사인 유사도, $\tau$는 temperature coefficient이다.

CLIP 분기에서는 masked crop을 CLIP에 입력하여 $\hat{p}_{i,k}$를 계산하고, 최종 예측은 두 분기의 앙상블이다:

$$p_{i,k}^{\text{final}} = p_{i,k}^{(1-\lambda)} \cdot \hat{p}_{i,k}^{\lambda}, \quad \lambda \in [0,1]$$

#### (B) 다양한 Mask-Category 쌍 수집 (Section 3.2)

**문제**: COCO-Stuff의 GT 라벨(171 클래스)로 CLIP을 미세조정하면, 제한된 어휘에 과적합되어 일반화 능력 상실 (A-847에서 -2.0% 하락)

**해결**: 이미지-캡션 데이터셋(COCO Captions)에서 self-labeling 전략을 통해 데이터 수집:

1. 사전학습된 MaskFormer로 class-agnostic mask proposal 생성
2. 캡션에서 off-the-shelf language parser로 명사 추출
3. CLIP으로 각 명사에 가장 잘 매칭되는 mask proposal 할당

결과: 1 caption/image 기준 **440K 쌍, 12K 고유 명사** 수집 (5 captions 기준 1.3M 쌍, 27K 명사)

#### (C) Mask Prompt Tuning (Section 3.3)

Masked image를 CLIP에 입력하면, 배경이 제거된 패치들이 **zero token**이 되어 도메인 분포 이동(distribution shift)을 야기한다.

마스크 이미지가 토큰화되면 $T \in \mathbb{R}^{N_p \times E}$ ($N_p$: 패치 수, $E$: 토큰 차원)가 되고, 축약된 바이너리 마스크 $M_p \in \{0,1\}^{N_p}$가 주어진다. 학습 가능한 프롬프트 텐서 $P \in \mathbb{R}^{N_p \times E}$를 할당하여:

$$T_{\text{input}} = T \otimes M_p + P \otimes (1 - M_p)$$

여기서 $\otimes$는 element-wise multiplication이다. "Deep prompts" 방식으로 트랜스포머의 깊은 레이어에도 적용 가능하다.

**핵심 특성**: 
- CLIP 가중치를 **변경하지 않으므로** 멀티태스크 가중치 공유 가능
- Full fine-tuning과 결합 시 추가적 성능 향상 (Table 3: case (c)에서 A-150 +0.8%)
- 학습 파라미터 수가 full fine-tuning 대비 수 자릿수(orders of magnitude) 적음

### 2.3 모델 구조

전체 파이프라인 (Figure 2):

```
[입력 이미지] → [MaskFormer (Swin-B/R-101c)] → N개 mask proposals + N개 proposal embeddings
                                                    ↓
                                              [Masked Crop] → [CLIP Image Encoder (ViT-L/14)] → 시각 임베딩
                                                                                                    ↓
                                              [텍스트 쿼리] → [CLIP Text Encoder] → 텍스트 임베딩 → 코사인 유사도 → 분류
                                                    ↓
                                              [앙상블: MaskFormer 예측 × CLIP 예측] → 최종 세그멘테이션
```

- **MaskFormer**: COCO-Stuff 171 클래스로 학습, "no object" 임베딩 $\emptyset$ 추가
- **CLIP 적응**: 캡션 기반 mask-category 쌍으로 이미지 인코더만 미세조정 (텍스트 인코더 동결)
- **3가지 적응 방식**: (a) MPT only, (b) Full FT only, (c) MPT + FT (순서: FT → MPT)

### 2.4 성능 향상

**Table 1 주요 결과 (COCO 학습 → 제로샷 평가):**

| 모델 | Backbone | A-847 | PC-459 | A-150 | PC-59 | PAS-20 |
|------|----------|-------|--------|-------|-------|--------|
| OpenSeg | R-101 | 4.0 | 6.5 | 15.3 | 36.9 | 60.0 |
| OVSeg (R-101c) | R-101c | **7.1** | **11.0** | **24.8** | **53.3** | **92.6** |
| OpenSeg | Eff-B7 | 6.3 | 9.0 | 21.1 | 42.1 | - |
| OVSeg (Swin-B) | Swin-B | **9.0** | **12.4** | **29.6** | **55.7** | **94.5** |

- A-150에서 이전 SOTA 대비 **+8.5%** (21.1% → 29.6%)
- A-847에서 **+2.7%**, PC-459에서 **+3.4%**
- PAS-20에서 94.5%로 supervised specialist (SelfTrain, 90.0%) 초과

**Ablation 결과 (Table 3):**

| 방법 | A-847 | A-150 | PC-59 |
|------|-------|-------|-------|
| Baseline (원본 CLIP) | 7.3 | 21.8 | 51.4 |
| MPT only | 8.4 (+1.1) | 26.5 (+4.7) | 55.4 (+4.0) |
| Full FT only | 8.8 (+1.5) | 28.8 (+7.0) | 55.7 (+4.3) |
| MPT + FT | **9.0 (+1.7)** | **29.6 (+7.8)** | **55.7 (+4.3)** |

### 2.5 한계

1. **추론 속도**: CLIP으로 수백 개 region을 분류하는 것이 시간 소모적 (MaskFormer ~0.2s, CLIP region classification ~0.6s per image)
2. **평가 지표의 모호성**: 언어 기반 카테고리는 본질적으로 모호함 (예: "building" vs. "skyscraper", "rail" vs. "road") → 합리적 예측이 오답 처리
3. **Mask proposal의 한계**: Class-agnostic proposal이 완전히 class-agnostic하지 않음 (학습 데이터의 클래스 정의에 의존)
4. **Seen vs. Unseen 성능 격차**: ADE20K-150에서 seen 카테고리 평균 IoU 37.6% vs. unseen 21.9%
5. **GT 라벨과 pseudo 라벨의 결합이 오히려 성능 저하** (Table 8: 28.8% → 26.7%)

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화 성능의 핵심 메커니즘

OVSeg의 일반화 성능 향상 전략은 크게 세 가지 축으로 구성된다:

#### (1) 텍스트 다양성을 통한 일반화 보존

이 논문의 가장 중요한 발견 중 하나는 **학습 데이터의 텍스트 다양성(vocabulary diversity)이 일반화 능력과 직결**된다는 것이다.

- **GT 라벨 사용 (171 클래스)**: A-847에서 5.3% (baseline 7.3% 대비 -2.0% 하락) → **과적합으로 일반화 능력 상실**
- **캡션 기반 (12K 명사)**: A-847에서 8.8% (+1.5% 향상) → **다양한 어휘가 open-vocabulary 능력 보존**

이는 CLIP의 원래 학습 데이터가 수억 개의 이미지-텍스트 쌍으로 구성되어 방대한 어휘를 포함하고 있기 때문이다. 제한된 클래스셋으로 미세조정하면 이 풍부한 표현 공간이 축소된다.

#### (2) Mask Prompt Tuning의 일반화 친화적 특성

MPT는 CLIP의 가중치를 전혀 변경하지 않으므로:
- CLIP이 사전학습에서 획득한 **범용적 시각-언어 정렬(vision-language alignment)을 완전히 보존**
- 입력 레벨에서만 도메인 갭을 해소하는 "어댑터" 역할
- Full FT 모델에 추가 적용 시에도 +0.8% 추가 향상 (Table 3), 이는 FT로 인한 일반화 손실을 부분적으로 보상함을 시사

VPT(Visual Prompt Tuning)와의 비교 (Table 7):
- MPT: 26.5% mIoU (A-150) — 추가 연산 없음
- VPT: 25.5% mIoU (A-150) — 40% 추가 연산 필요
- MPT가 우수한 이유: **zero masked token을 직접 대체**하여 도메인 분포 이동을 직접적으로 완화

#### (3) Proposal 기반 매칭의 장점

GT 마스크 대신 생성된 proposal을 사용하면 (Table 2):
- GT 마스크 + 1 caption: A-150 24.2%
- Proposal + 1 caption: A-150 **28.8%** (+4.6%)

이유: GT 마스크에는 라벨이 없는 영역이 많아 무시되지만, proposal (보통 100개)은 이미지의 대부분 관심 영역을 커버하여 더 풍부한 학습 신호를 제공한다.

### 3.2 일반화 성능의 정량적 분석

**Seen vs. Unseen 카테고리 (Figure 8)**:
- Seen 카테고리 (COCO-Stuff와 유사): 평균 IoU **37.6%**
- Unseen 카테고리 (ADE20K 고유): 평균 IoU **21.9%**
- 격차가 존재하지만, 기존 방법론 대비 unseen 카테고리에서도 유의미한 성능

**데이터셋 간 전이 (Cross-dataset generalization)**:
- COCO에서 학습 → ADE20K, Pascal Context, Pascal VOC에서 제로샷 평가
- 5개 벤치마크 모두에서 일관된 SOTA 달성

### 3.3 일반화 성능 향상의 추가 가능성

논문에서 시사하는 향후 개선 방향:

1. **더 대규모 캡션 데이터**: CC3M, CC12M, LAION 등 웹 스케일 이미지-캡션 데이터 활용 → 어휘 다양성 극대화
2. **균형 잡힌 데이터 혼합**: GT 라벨과 pseudo 라벨의 비율 조정 (예: 10% GT + 90% pseudo) → 정확성과 다양성의 균형
3. **더 강력한 mask proposal 생성기**: Class-agnostic proposal의 품질 향상 → 더 정확한 mask-category 매칭
4. **CLIP 모델 스케일링**: 더 큰 CLIP 모델(ViT-G 등) 사용 → 본래의 일반화 능력 강화

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (1) 2-Stage 패러다임의 재정립
OVSeg는 "mask proposal + CLIP 분류"라는 2-stage 패러다임에서 **CLIP 적응(adaptation)의 중요성**을 처음으로 체계적으로 입증하였다. 이후 많은 연구들이 이 통찰을 기반으로 발전하였다.

#### (2) Foundation Model 적응 방법론의 확장
Mask Prompt Tuning은 **도메인 특화 프롬프팅의 새로운 패러다임**을 제시하였다. 가중치를 변경하지 않으면서 입력 레벨에서 도메인 갭을 해소하는 아이디어는 의료 영상, 위성 영상 등 다양한 도메인에 적용 가능하다.

#### (3) 약한 감독(Weak Supervision)의 효과성 입증
정교한 수동 라벨보다 **노이지하지만 다양한** 자동 라벨이 일반화에 더 유리하다는 발견은, 데이터 수집 전략에 대한 패러다임 전환을 촉발하였다.

#### (4) Open-vocabulary와 Supervised 모델 간 격차 축소
최초로 generalist 모델이 2017년 specialist 모델 성능에 도달하여, **범용 세그멘테이션 모델의 실용성**을 입증하였다.

### 4.2 향후 연구 시 고려사항

1. **추론 효율성**: 수백 개 region을 개별적으로 CLIP에 입력하는 것은 비효율적 → 배치 처리, 경량화, 또는 1-stage 접근법 연구 필요
2. **평가 프로토콜 개선**: 언어 기반 카테고리의 모호성 문제 → 의미적 유사성을 고려한 평가 지표 설계 필요
3. **더 강력한 mask proposal**: SAM(Segment Anything Model) 등 범용 세그멘테이션 모델과의 결합 가능성
4. **멀티모달 학습 데이터 확장**: 웹 스케일 데이터 활용 시 노이즈 제어 및 데이터 품질 관리
5. **Panoptic/Instance 세그멘테이션으로의 확장**: 시맨틱 세그멘테이션을 넘어 더 세밀한 인스턴스 구분

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연도 | 연구 | 핵심 접근법 | OVSeg 대비 차이점 | A-150 mIoU |
|------|------|-----------|----------------|-----------|
| 2021 | **CLIP** (Radford et al.) | 대규모 이미지-텍스트 대비 학습 | OVSeg의 기반 모델 | - |
| 2021 | **LSeg** (Li et al.) | 픽셀 임베딩을 CLIP 텍스트 임베딩에 정렬 | 픽셀 레벨 접근, 마스크 기반이 아님 | - |
| 2021 | **OpenSeg** (Ghiasi et al.) | 세그먼트-레벨 시각 특징과 텍스트의 region-word grounding | CLIP 적응 없이 원본 사용 | 21.1 (Eff-B7) |
| 2021 | **SimBaseline/ZSSeg** (Xu et al.) | 2-stage: MaskFormer + 원본 CLIP | CLIP 적응 없음 | 15.3 |
| 2022 | **ZegFormer** (Ding et al.) | Decoupled zero-shot 세그멘테이션 | CLIP 적응 없이 원본 사용 | 16.4 |
| 2022 | **GroupViT** (Xu et al.) | 텍스트 감독만으로 세그멘테이션 마스크 학습 | Mask proposal 불필요, 순수 텍스트 감독 | - |
| 2022 | **RegionCLIP** (Zhong et al.) | Region proposal로 CLIP 적응 (객체 탐지) | 마스킹 없는 region crop 사용, MPT 없음 | - |
| 2022 | **OVSeg (본 논문)** | Mask-adapted CLIP + MPT | - | **29.6** (Swin-B) |
| 2023 | **SAN** (Xu et al.) | Side Adapter Network | CLIP에 사이드 어댑터 부착 | - |
| 2023 | **SAM** (Kirillov et al.) | Segment Anything Model | Class-agnostic 범용 세그멘테이션, 분류 없음 | - |
| 2023 | **ODISE** (Xu et al.) | Diffusion 모델의 내부 표현 활용 | 확산 모델의 풍부한 시각 특징 활용 | 29.9 |
| 2023 | **SED** (Xie et al.) | 1-stage open-vocab 세그멘테이션 | 2-stage 비효율성 해결, 단일 패스 추론 | 31.6 |
| 2023 | **FC-CLIP** (Yu et al.) | Frozen CLIP backbone + 경량 디코더 | CLIP 가중치 완전 동결, 효율적 | 34.1 |
| 2024 | **CAT-Seg** (Cho et al.) | CLIP과 세그멘테이션의 cost aggregation | 비용 볼륨 기반 밀집 정렬 | 31.5 |

### 주요 트렌드 분석

1. **CLIP 적응 → CLIP 동결**: OVSeg 이후 연구들은 CLIP을 미세조정하기보다 **완전 동결 상태에서 활용**하는 방향으로 발전 (FC-CLIP 등)
2. **2-Stage → 1-Stage**: 추론 효율성 문제로 인해 **단일 패스 방법론**이 등장 (SED 등)
3. **SAM과의 결합**: SAM의 강력한 class-agnostic 세그멘테이션 능력을 CLIP의 분류 능력과 결합하는 연구 활발
4. **확산 모델(Diffusion Model) 활용**: ODISE 등에서 확산 모델의 풍부한 내부 표현을 활용한 세그멘테이션 연구 등장

### OVSeg의 역사적 위치
OVSeg는 **2-stage open-vocabulary segmentation에서 CLIP 적응의 필요성을 최초로 체계적으로 입증**한 논문으로, 이후 연구들의 방향성을 설정하는 데 큰 영향을 미쳤다. 특히 (1) masked image 도메인 갭 문제 정의, (2) 텍스트 다양성의 중요성 발견, (3) mask prompt tuning이라는 새로운 적응 기법은 후속 연구의 중요한 참고점이 되고 있다.

---

## 참고자료

1. **Liang, F. et al.** "Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP." *arXiv:2210.04150v3*, 2023. (본 논문)
2. **Radford, A. et al.** "Learning Transferable Visual Models From Natural Language Supervision." *ICML*, 2021. (CLIP)
3. **Ghiasi, G. et al.** "Open-Vocabulary Image Segmentation." *arXiv:2112.12143*, 2021. (OpenSeg)
4. **Xu, M. et al.** "A Simple Baseline for Zero-Shot Semantic Segmentation with Pre-trained Vision-Language Model." *arXiv:2112.14757*, 2021. (SimBaseline/ZSSeg)
5. **Ding, J. et al.** "Decoupling Zero-Shot Semantic Segmentation." *CVPR*, 2022. (ZegFormer)
6. **Li, B. et al.** "Language-driven Semantic Segmentation." *arXiv:2201.03546*, 2022. (LSeg)
7. **Zhong, Y. et al.** "RegionCLIP: Region-based Language-Image Pretraining." *CVPR*, 2022. (RegionCLIP)
8. **Jia, M. et al.** "Visual Prompt Tuning." *arXiv:2203.12119*, 2022. (VPT)
9. **Cheng, B. et al.** "Per-Pixel Classification is Not All You Need for Semantic Segmentation." *NeurIPS*, 2021. (MaskFormer)
10. **Xu, M. et al.** "Side Adapter Network for Open-Vocabulary Semantic Segmentation." *arXiv:2302.12242*, 2023. (SAN)
11. **Xu, J. et al.** "GroupViT: Semantic Segmentation Emerges from Text Supervision." *CVPR*, 2022. (GroupViT)
12. **Yu, Q. et al.** "Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP." *NeurIPS*, 2023. (FC-CLIP)
13. **Xu, J. et al.** "ODISE: Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models." *CVPR*, 2023. (ODISE)
14. **Cho, S. et al.** "CAT-Seg: Cost Aggregation for Open-Vocabulary Panoptic Segmentation." *CVPR*, 2024. (CAT-Seg)
15. **Kirillov, A. et al.** "Segment Anything." *ICCV*, 2023. (SAM)

> **주의**: FC-CLIP, ODISE, CAT-Seg, SED 등 후속 연구의 정확한 mIoU 수치는 각 논문의 실험 설정(backbone, 학습 데이터 등)에 따라 다를 수 있으며, 위 표의 수치는 해당 논문들에서 보고된 대표적 결과를 참고한 것입니다. 직접적인 공정 비교를 위해서는 동일 조건하의 실험이 필요합니다.
