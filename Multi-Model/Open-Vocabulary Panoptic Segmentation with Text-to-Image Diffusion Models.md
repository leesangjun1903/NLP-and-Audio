# Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models

---

# 1. 핵심 주장 및 주요 기여 (요약)

ODISE(Open-vocabulary DIffusion-based panoptic SEgmentation)는 사전 학습된 텍스트-이미지 확산(diffusion) 모델과 판별(discriminative) 모델을 통합하여 개방형 어휘(open-vocabulary) 파놉틱 세그멘테이션을 수행하는 프레임워크입니다.

**핵심 주장:**
- 텍스트-이미지 확산 모델은 다양한 개방형 어휘의 언어 설명으로 고품질 이미지를 생성할 수 있는 뛰어난 능력을 가지고 있으며, 이는 해당 모델의 내부 표현 공간이 실세계의 개방적 개념들과 높은 상관관계를 가짐을 보여줍니다.
- 한편 CLIP과 같은 텍스트-이미지 판별 모델은 이미지를 개방형 어휘 레이블로 분류하는 데 뛰어나므로, 두 모델의 동결된(frozen) 내부 표현을 결합하여 야생의 모든 카테고리에 대한 파놉틱 세그멘테이션을 수행합니다.

**주요 기여:**
1. 대규모 텍스트-이미지 확산 모델을 개방형 어휘 세그멘테이션에 활용한 최초의 연구입니다.
2. 확산 모델 내부 표현의 K-Means 클러스터링이 의미적으로 차별화되고 공간적으로 잘 국소화된 정보를 보여줌을 입증합니다.
3. 개방형 어휘 파놉틱 및 시맨틱 세그멘테이션 모두에서 이전 최고 성능(state-of-the-art)을 크게 상회하며, COCO 훈련만으로 ADE20K 데이터셋에서 23.4 PQ, 30.0 mIoU를 달성하여 기존 대비 각각 8.3 PQ, 7.9 mIoU의 절대적 향상을 이루었습니다.
4. 학습 가능한 파라미터가 28.1M에 불과하여 효율적입니다.

---

# 2. 상세 분석

## 2.1 해결하고자 하는 문제

개방형 어휘 인식(open-vocabulary recognition)은 많은 관심을 받고 있지만, 모든 객체 인스턴스와 장면의 시맨틱을 동시에 파싱하는 통합 프레임워크(즉, 파놉틱 세그멘테이션)가 부족합니다. 기존 개방형 어휘 인식 방법들은 주로 텍스트-이미지 판별 모델에 의존하여 공간적·관계적 이해에 어려움을 겪으며, 이는 파놉틱 세그멘테이션의 병목(bottleneck)이 됩니다.

기존 두 단계(two-stage) 접근법의 한계:
- OVSeg 연구에서 밝혀졌듯, 클래스 비구분(class-agnostic) 마스크는 객체 위치 파악에는 성공하지만, 정확한 클래스 레이블 할당에는 종종 실패합니다.
- 사전 학습된 CLIP은 마스크된 이미지를 만족스럽게 분류할 수 없어, 두 단계 개방형 어휘 세그멘테이션 모델의 성능 병목으로 작용합니다.

## 2.2 제안하는 방법 (ODISE Architecture)

### 전체 파이프라인

ODISE는 먼저 입력 이미지를 implicit captioner(이미지 인코더 $V$와 MLP)를 통해 암묵적 텍스트 임베딩으로 변환합니다. 이미지와 해당 암묵적 텍스트 임베딩을 입력으로 하여, 동결된 텍스트-이미지 확산 UNet에서 확산 특징(diffusion features)을 추출합니다. UNet 특징을 사용하여, 마스크 생성기(mask generator)가 클래스 비구분 이진 마스크와 연관된 마스크 임베딩 특징을 예측합니다. 마스크 임베딩과 훈련 카테고리 이름의 텍스트 임베딩 간 내적(dot product)을 수행하여 분류합니다.

### (A) Implicit Captioner

별도의 캡셔닝 네트워크를 사용하는 대신, 입력 이미지로부터 텍스트 임베딩을 직접 생성하는 네트워크를 훈련합니다. 이 텍스트 임베딩은 확산 모델에 직접 입력되며, 이를 "implicit captioner"라고 명명합니다.

Implicit captioner의 수식은 다음과 같습니다:

$$c_I = \text{MLP}(V(x))$$

여기서 $V(\cdot)$는 이미지 인코더(예: CLIP 이미지 인코더), $x$는 입력 이미지, $c_I$는 생성된 암묵적 텍스트 임베딩입니다.

### (B) Diffusion Feature Extraction

확산 기반 텍스트-이미지 생성 모델은 보통 UNet 아키텍처를 사용하여 디노이징 과정을 학습합니다. UNet은 합성곱 블록, 업샘플링/다운샘플링 블록, 스킵 연결, 그리고 텍스트 임베딩과 UNet 특징 간의 교차 어텐션(cross-attention)을 수행하는 어텐션 블록으로 구성됩니다. 디노이징의 매 단계에서, 텍스트 입력을 활용하여 노이즈가 포함된 입력 이미지의 디노이징 방향을 추론합니다.

확산 모델의 디노이징 과정에서 특징을 추출하는 수식:

$$\hat{\epsilon}_\theta(x_t, t, c_I) = \text{UNet}_\theta(x_t, t, c_I)$$

여기서:
- $x_t$: 타임스텝 $t$에서의 노이즈 이미지 (latent)
- $t$: 디노이징 타임스텝
- $c_I$: implicit captioner에서 생성된 텍스트 임베딩
- $\theta$: 동결된 UNet 파라미터

UNet 내부의 다양한 해상도 특징맵을 결합하여 다중 스케일 확산 특징 $F_{\text{diff}}$를 구성합니다:

$$F_{\text{diff}} = \{f_l^{\text{UNet}} \mid l = 1, 2, \ldots, L\}$$

### (C) Mask Generator (Mask2Former 기반)

UNet 특징을 사용하여, 마스크 생성기는 클래스 비구분 이진 마스크와 해당 마스크 임베딩 특징을 예측합니다. 마스크 임베딩과 훈련 카테고리 이름의 텍스트 임베딩 간 내적을 수행하여 분류합니다.

마스크 생성기의 출력:

$$\{(m_i, e_i)\}_{i=1}^{N} = \text{MaskGenerator}(F_{\text{diff}})$$

여기서 $m_i \in \{0, 1\}^{H \times W}$는 이진 마스크, $e_i \in \mathbb{R}^d$는 마스크 임베딩, $N$은 마스크 제안(proposal) 수입니다.

### (D) Open-Vocabulary Classification

분류 시, 마스크 임베딩과 카테고리 텍스트 임베딩 간의 유사도 계산:

$$p(c \mid m_i) = \frac{\exp(e_i \cdot t_c / \tau)}{\sum_{c' \in \mathcal{C}} \exp(e_i \cdot t_{c'} / \tau)}$$

여기서:
- $t_c$: 카테고리 $c$의 텍스트 임베딩 (CLIP 텍스트 인코더를 통해 생성)
- $\tau$: 온도 파라미터 (temperature)
- $\mathcal{C}$: 테스트 카테고리 집합

### (E) 학습 손실 함수

유사도 행렬에 대한 분류는 정답 카테고리 레이블에 대한 교차 엔트로피 손실(cross-entropy loss, 레이블 감독 경로) 또는 페어링된 이미지 캡션에 대한 그라운딩 손실(grounding loss, 캡션 감독 경로)로 학습됩니다.

전체 손실 함수는 Mask2Former의 구조를 따릅니다:

$$\mathcal{L} = \lambda_{\text{ce}} \mathcal{L}_{\text{CE}} + \lambda_{\text{mask}} \mathcal{L}_{\text{mask}} + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}$$

여기서:
- $\mathcal{L}_{\text{CE}}$: 마스크 분류를 위한 교차 엔트로피 손실
- $\mathcal{L}_{\text{mask}}$: 이진 마스크 예측을 위한 바이너리 교차 엔트로피 손실
- $\mathcal{L}_{\text{dice}}$: Dice 손실
- $\lambda_{\text{ce}}, \lambda_{\text{mask}}, \lambda_{\text{dice}}$: 각 손실의 가중치

캡션 기반 그라운딩 손실 (caption-based grounding loss)은 다음과 같이 정의됩니다:

$$\mathcal{L}_{\text{ground}} = -\sum_{i} \log \frac{\exp(e_i \cdot t_{w_i} / \tau)}{\sum_{w \in \mathcal{W}} \exp(e_i \cdot t_w / \tau)}$$

여기서 $\mathcal{W}$는 캡션 내 명사 집합, $w_i$는 마스크 $m_i$에 매칭된 명사입니다.

## 2.3 모델 구조 요약

| 구성 요소 | 세부 사항 |
|---|---|
| **Diffusion Backbone** | Stable Diffusion v1.3 (Frozen UNet) |
| **Discriminative Model** | CLIP (Frozen) |
| **Implicit Captioner** | CLIP Image Encoder + MLP |
| **Mask Generator** | Mask2Former 기반 |
| **학습 가능 파라미터** | 28.1M |
| **훈련 데이터** | COCO (panoptic annotations) |

## 2.4 성능 향상

ODISE는 개방형 어휘 파놉틱 및 시맨틱 세그멘테이션 모두에서 이전 최고 성능 대비 큰 폭으로 향상되었습니다. COCO 훈련만으로 ADE20K에서 23.4 PQ, 30.0 mIoU를 달성하여 기존 대비 각각 8.3 PQ, 7.9 mIoU 절대 향상을 이루었습니다.

MaskCLIP 대비 ADE20K에서 8.3 PQ의 절대 향상을 달성하였습니다.

개방형 어휘 인스턴스 세그멘테이션에서도 ADE20K에서 MaskCLIP 대비 8.4 mAP의 향상을 보였습니다.

Cityscapes와 Mapillary Vistas에서도 CLIP 기반 변형 대비 큰 폭으로 성능이 우수하며(PQ 23.9 vs 18.5, 14.2 vs 11.7), 확산 특징이 다양한 도시 및 거리 장면 환경에서 더 견고하고 일반화 가능함을 시사합니다.

## 2.5 한계

1. **추론 속도**: 확산 모델의 특징 추출에 디노이징 과정이 필요하여 추론 시간이 상대적으로 깁니다.
2. **확산 모델 의존성**: Stable Diffusion의 사전 학습 데이터 분포에 영향을 받으므로, 해당 분포에서 벗어난 도메인(의료, 위성 영상 등)에서는 성능 저하 가능성이 있습니다.
3. **해상도 제한**: Stable Diffusion의 latent space 해상도(통상 64×64)에 의해 세밀한 경계 예측이 제한될 수 있습니다.
4. M-ODISE와 같은 후속 연구에서 확인되었듯, 특정 도메인(예: 유방촬영술)에 적용 시 전문 아키텍처 대비 한계가 있으며, 현재 파놉틱 세그멘테이션 기법을 특정 분석에 맞추기 위한 추가 기술 발전이 필요합니다.

---

# 3. 모델의 일반화 성능 향상 가능성

ODISE의 일반화 성능이 뛰어난 핵심 이유와 추가 향상 가능성:

### 3.1 확산 모델의 내부 표현이 갖는 일반화 능력

텍스트-이미지 확산 모델은 인터넷 규모의 이미지-텍스트 쌍으로 학습되었기 때문에, 내부 표현이 실세계의 개방적 개념들과 높은 상관관계를 가집니다. 이는 훈련 시 본 적 없는 카테고리도 의미적으로 풍부한 특징으로 표현 가능함을 의미합니다.

### 3.2 생성 모델 + 판별 모델의 시너지

확산 모델과 판별 모델 양쪽의 예측을 융합(fusion)하면, 각 모델을 개별적으로 사용할 때보다 PQ, mAP, mIoU 모두 더 높은 성능을 달성합니다.

- **확산 모델 특징**: 공간적 localization과 의미적 분화에 강점
- **CLIP 특징**: 카테고리 분류와 개방형 어휘 매칭에 강점

최종 분류 점수 융합:

$$s_c^{\text{final}}(m_i) = \alpha \cdot s_c^{\text{diff}}(m_i) + (1 - \alpha) \cdot s_c^{\text{CLIP}}(m_i)$$

### 3.3 Implicit Captioner의 역할

Implicit Captioner는 입력 이미지에서 텍스트 임베딩을 생성하여, 확산 모델의 시각적 표현 추출 시 명시적으로 페어링된 캡션이 필요 없도록 합니다. 이를 통해 추론 시 텍스트 입력 없이도 확산 특징을 효과적으로 추출할 수 있어 일반화를 강화합니다.

### 3.4 향후 일반화 성능 향상 방향

| 방향 | 설명 |
|---|---|
| **더 강력한 확산 모델** | SDXL, Stable Diffusion 3 등 더 큰 모델로 대체 시 표현력 증대 |
| **다중 타임스텝 특징 앙상블** | 여러 디노이징 타임스텝의 특징을 결합하여 다양한 의미 수준 포착 |
| **도메인 적응** | LoRA 등으로 특정 도메인에 확산 모델을 미세조정하여 전이 성능 향상 |
| **더 나은 VLM** | EVA-CLIP, SigLIP 등 더 강력한 판별 모델 활용 |

LVIS, COCO, ADE20K의 카테고리 이름을 병합하여 약 1,500개의 클래스에 대해 직접 개방형 어휘 추론을 수행할 수 있음을 시연하였습니다.

---

# 4. 향후 연구에 미치는 영향 및 고려사항

## 4.1 연구에 미치는 영향

1. **생성 모델의 표현을 판별 과제에 활용하는 패러다임 개척**: ODISE는 생성 모델(확산 모델)의 풍부한 내부 표현을 세그멘테이션이라는 판별 과제에 활용할 수 있음을 최초로 체계적으로 입증하여, 이후 수많은 후속 연구의 기반이 되었습니다.

2. **"Frozen Foundation Model + Lightweight Head" 패러다임 강화**: 거대 기반 모델을 동결한 채 소규모 학습 가능 모듈만 추가하는 효율적 접근법의 유효성을 입증했습니다.

3. **개방형 어휘 파놉틱 세그멘테이션의 벤치마크 설정**: ADE20K에서 파놉틱, 인스턴스, 시맨틱 세그멘테이션을 모두 평가하고 Pascal 데이터셋에서도 시맨틱 세그멘테이션을 평가하며, 단일 모델로 모든 태스크와 데이터셋에서 평가합니다.

## 4.2 앞으로 연구 시 고려할 점

1. **추론 효율성**: 확산 모델 기반 특징 추출은 추론 비용이 높으므로, 특징 증류(feature distillation), 또는 단일 스텝 추론 기법 연구가 필요합니다.
2. **확산 모델 vs. 판별 모델의 최적 융합 비율**: 도메인과 태스크에 따른 최적 $\alpha$ 설정 연구가 필요합니다.
3. **멀티모달 대형 모델과의 결합**: GPT-4V, Qwen-VL 등 MLLM과 결합한 세그멘테이션 접근법 탐구가 필요합니다.
4. **비디오 및 3D 확장**: 시공간적 일관성을 고려한 확산 특징 활용 연구가 유망합니다.
5. **경계 정확도 향상**: Latent space의 해상도 한계를 극복하기 위한 고해상도 특징 추출 기법이 필요합니다.

---

# 5. 2020년 이후 관련 최신 연구 비교 분석

2023년 이후 개방형 어휘 세그멘테이션 분야에서는 OVSeg, X-Decoder, SAN, ODISE, FreeSeg, OpenSeeD, FC-CLIP 등 다수의 중요 연구들이 발표되었습니다.

| 연구 | 발표 | 핵심 접근 | 특징 |
|---|---|---|---|
| **GroupViT** (Xu et al.) | CVPR 2022 | 텍스트 감독 기반 시맨틱 세그멘테이션 | 그룹화를 통한 제로샷 세그멘테이션 |
| **OVSeg** (Liang et al.) | CVPR 2023 | 마스크된 이미지와 텍스트 어노테이션으로 CLIP을 미세조정하여 분류 정확도 향상 | CLIP 도메인 갭 해소 |
| **SAN** (Side Adapter Network) | CVPR 2023 | CLIP에 아키텍처적 개선(side adapter)을 추가한 직접 정렬 방식 | 효율적 구조 |
| **ODISE** (Xu et al.) | CVPR 2023 | 확산 모델로 마스크 생성, CLIP 기반 특징으로 인식 수행 | 최초 확산 모델 기반 |
| **FC-CLIP** (Yu et al.) | NeurIPS 2023 | CLIP 프레임워크 내에 학습 가능한 마스크 예측기를 통합하여 세그멘테이션 정확도 향상 | 단일 단계, 효율적 |
| **CAT-Seg** (Cho et al.) | 2024 | cost map을 공간적/스케일 간 집계하여 성능 개선 | 비용 맵 정제 |
| **SED** (Xie et al.) | CVPR 2024 | 텍스트-이미지 확산 모델로 마스크 제안을 생성하고 분류를 수행하는 ODISE와 달리, 경량 인코더-디코더 아키텍처로 다양한 장면 카테고리에서 견고한 성능을 제공합니다. | 속도-정확도 최적 균형 |
| **OpenSeg-R** | 2025 | 시각적/의미적으로 유사한 카테고리를 구분하기 어려운 기존 한계를 계층적 시각적 추론으로 해결 | VLM 추론 결합 |

### 접근법 분류 비교

두 단계(two-stage) 방법인 OVSeg, MaskCLIP, ODISE 등은 먼저 클래스 비구분 마스크 제안을 생성한 후 텍스트 임베딩으로 분류합니다.

반면 단일 단계(single-stage) 방법인 LSeg, FC-CLIP, CAT-Seg, SED 등은 픽셀 수준의 특징을 CLIP 텍스트 임베딩과 직접 정렬하며, SAN과 FC-CLIP은 아키텍처적 개선을, CAT-Seg와 SED는 cost map의 공간적/스케일 간 집계를 통해 성능을 높입니다.

### ODISE의 차별적 위치

ODISE는 **생성 모델(확산 모델)의 표현력**을 세그멘테이션에 최초로 활용했다는 점에서 고유한 위치를 가지며, 이후 연구들이 효율성(FC-CLIP, SED), 추론 능력(OpenSeg-R), 도메인 적응(M-ODISE) 등 다양한 방향으로 확장하는 기반이 되었습니다.

---

## 참고 자료 및 출처

1. **Xu, J., Liu, S., Vahdat, A., Byeon, W., Wang, X., & De Mello, S.** (2023). "Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models." *CVPR 2023*, pp. 2955–2966. [arXiv:2303.04803](https://arxiv.org/abs/2303.04803)
2. **ODISE Official GitHub**: [https://github.com/NVlabs/ODISE](https://github.com/NVlabs/ODISE)
3. **ODISE Project Page (Jiarui Xu)**: [https://jerryxu.net/ODISE/](https://jerryxu.net/ODISE/)
4. **NVIDIA Research Publication Page**: [https://research.nvidia.com/publication/2023-06_open-vocabulary-panoptic-segmentation-text-image-diffusion-models](https://research.nvidia.com/publication/2023-06_open-vocabulary-panoptic-segmentation-text-image-diffusion-models)
5. **CVPR 2023 Open Access**: [https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Open-Vocabulary_Panoptic_Segmentation_With_Text-to-Image_Diffusion_Models_CVPR_2023_paper.html](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Open-Vocabulary_Panoptic_Segmentation_With_Text-to-Image_Diffusion_Models_CVPR_2023_paper.html)
6. **Liner Quick Review**: [https://liner.com/review/openvocabulary-panoptic-segmentation-with-texttoimage-diffusion-models](https://liner.com/review/openvocabulary-panoptic-segmentation-with-texttoimage-diffusion-models)
7. **Awesome Open-Vocabulary Semantic Segmentation (GitHub list)**: [https://github.com/Qinying-Liu/Awesome-Open-Vocabulary-Semantic-Segmentation](https://github.com/Qinying-Liu/Awesome-Open-Vocabulary-Semantic-Segmentation)
8. **SED (Xie et al., CVPR 2024)**: [https://arxiv.org/html/2311.15537](https://arxiv.org/html/2311.15537)
9. **OpenSeg-R (2025)**: [https://arxiv.org/html/2505.16974](https://arxiv.org/html/2505.16974)
10. **M-ODISE (Panoptic Segmentation of Mammograms)**: [https://arxiv.org/html/2407.14326](https://arxiv.org/html/2407.14326)
11. **ResearchGate - ODISE Overview Diagram**: [https://www.researchgate.net/figure/ODISE-Overview-and-Training-Pipeline](https://www.researchgate.net/figure/ODISE-Overview-and-Training-Pipeline-We-first-encode-the-input-image-into-an-implicit_fig2_369090393)

> **참고**: 위 수식 중 일부(예: 최종 융합 점수, implicit captioner 수식)는 논문의 방법론을 정확히 기술하기 위해 논문 본문의 기술적 설명을 기반으로 재구성한 것입니다. 논문 원문의 정확한 수식 기호는 원 PDF를 직접 참조하시기를 권장합니다.
