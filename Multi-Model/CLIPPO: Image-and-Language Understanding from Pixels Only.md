# CLIPPO: Image-and-Language Understanding from Pixels Only

---

## 1. 핵심 주장과 주요 기여 요약

**핵심 주장:** 텍스트를 이미지로 렌더링하여 **단일 Vision Transformer(ViT)**만으로 이미지, 텍스트, 멀티모달 태스크를 모두 처리할 수 있으며, 별도의 텍스트 인코더, 토크나이저, 워드 임베딩 없이도 CLIP 수준의 성능에 근접할 수 있다.

**주요 기여:**
1. **완전한 모달리티 통합:** 텍스트를 RGB 이미지로 렌더링하여 단일 ViT 인코더로 이미지·텍스트·멀티모달 입력을 모두 처리하는 CLIPPO 모델 제안
2. **파라미터 효율성:** CLIP 대비 약 절반의 파라미터(93M vs. 203M)로 유사한 성능 달성
3. **토크나이저 불필요:** 어휘/토크나이저 설계를 우회하여 다국어 검색에서 강점 발휘
4. **언어 이해 능력:** 워드 수준 손실(LM, MLM) 없이 대조학습만으로 GLUE 벤치마크에서 PIXEL을 상회하고 BERT에 근접
5. **VQA 가능성:** 질문과 이미지를 함께 렌더링하는 것만으로 VQA 수행 가능

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 멀티모달 모델(CLIP, ALIGN 등)은 다음과 같은 문제를 가진다:
- **모달리티별 별도 인코더:** 이미지 인코더와 텍스트 인코더가 분리되어 파라미터 수가 증가하고 엔지니어링 복잡도가 높아짐
- **토크나이저 의존성:** 텍스트 처리 시 특정 언어/도메인에 최적화된 토크나이저가 필요하며, 다국어 환경에서 성능 편차 발생
- **모달리티별 전처리 파이프라인:** 학습과 전이 과정에서 모달리티/태스크별 전처리가 상이

CLIPPO는 "**텍스트를 이미지로 렌더링**"하여 **단일 비전 인코더**로 모든 모달리티를 통합함으로써 이러한 문제를 해결하고자 한다.

### 2.2 제안하는 방법

#### 모델 구조

CLIPPO는 단일 **ViT-B/16 또는 ViT-L/16** 인코더를 사용한다. 입력은 모두 RGB 이미지 형태로 통일된다:

- **일반 이미지:** 원본 이미지를 그대로 사용
- **텍스트:** GNU Unifont 비트맵 폰트를 사용하여 텍스트를 $224 \times 224$ (또는 $384 \times 384$) RGB 이미지로 렌더링
- **VQA:** 질문 텍스트를 이미지 상단에 렌더링하여 하나의 이미지로 결합

핵심적으로, **패치 임베딩(convolutional patch embedding)이 이미지와 렌더링된 텍스트에 대해 완전히 공유**되며, 별도의 토큰 임베딩 테이블이 존재하지 않는다. MAP(Multi-head Attention Pooling) 헤드를 사용하여 최종 임베딩을 생성한다.

#### 대조 학습 손실 함수

CLIPPO는 CLIP과 동일한 대조 손실(contrastive loss)을 사용한다. 배치 내 $N$개의 이미지-텍스트 쌍 $(I_i, T_i)$에 대해, 이미지 임베딩 $\mathbf{z}_i^I = f(I_i)$와 텍스트 이미지 임베딩 $\mathbf{z}_i^T = f(\text{render}(T_i))$를 동일한 인코더 $f$로 계산한다.

**Image-to-Text 방향 손실:**

$$\mathcal{L}_{I \to T} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\left(\text{sim}(\mathbf{z}_i^I, \mathbf{z}_i^T) / \tau\right)}{\sum_{j=1}^{N} \exp\left(\text{sim}(\mathbf{z}_i^I, \mathbf{z}_j^T) / \tau\right)}$$

**Text-to-Image 방향 손실:**

$$\mathcal{L}_{T \to I} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\left(\text{sim}(\mathbf{z}_i^T, \mathbf{z}_i^I) / \tau\right)}{\sum_{j=1}^{N} \exp\left(\text{sim}(\mathbf{z}_i^T, \mathbf{z}_j^I) / \tau\right)}$$

여기서 $\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$는 코사인 유사도이며, $\tau$는 학습 가능한 온도 파라미터(초기값 10)이다.

**총 손실:**

$$\mathcal{L} = \frac{1}{2}\left(\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}\right)$$

#### 텍스트/텍스트 대조학습 (Next Sentence Prediction, NSP)

언어 이해 능력 향상을 위해, C4 코퍼스에서 연속 문장 쌍 $(S_1, S_2)$을 샘플링하여 렌더링된 이미지 쌍으로 대조학습을 수행한다. 이 NSP 손실은 이미지-텍스트 대조 손실과 동일한 형태로, 배치 내 텍스트/텍스트 쌍의 비율(25%, 50%)을 조절하여 co-training한다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{img-text}} + \mathcal{L}_{\text{text-text (NSP)}}$$

### 2.3 학습 세부사항

| 항목 | 설정 |
|---|---|
| 기본 아키텍처 | ViT-B/16 (93M 파라미터) |
| 대규모 아키텍처 | ViT-L/16 (316M 파라미터) |
| 표현 차원 | 768 |
| 배치 크기 | 10,240 |
| 학습 스텝 | 250k (기본) |
| 옵티마이저 | Adafactor ($\text{lr}=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$) |
| 가중치 감쇠 | $10^{-4}$ (decoupled) |
| 학습 데이터 | WebLI (10B 이미지, 12B alt-text, 109개 언어) |
| 텍스트 렌더링 | GNU Unifont 비트맵 폰트 |
| 해상도 | $224 \times 224$ (기본), $384 \times 384$ (고해상도) |

### 2.4 성능 결과

#### 이미지 분류 및 검색 (Table 1)

| 모델 | 파라미터 | I1k 0-shot | COCO I→T | Flickr I→T |
|---|---|---|---|---|
| CLIP* | 203M | 65.1 | 48.5 | 79.2 |
| 1T-CLIP | 118M | 62.3 | 48.0 | 77.5 |
| **CLIPPO** | **93M** | **61.4** | **47.3** | **76.4** |
| CLIPPO L/16 | 316M | 67.4 | 50.6 | 79.2 |
| 1T-CLIP L/16 | 349M | 67.8 | 50.7 | 81.0 |

→ CLIPPO는 CLIP* 대비 약 2–3%p 하락이나, **파라미터는 절반 이하**

#### GLUE 벤치마크 (Table 2)

| 모델 | 학습 데이터 | GLUE avg |
|---|---|---|
| BERT-Base | Wiki+BC | 83.1 |
| PIXEL | Wiki+BC | 76.3 |
| BiLSTM+Attn, ELMo | — | 70.0 |
| CLIPPO | WebLI | 68.6 |
| CLIPPO | WebLI + 25%C4 | 74.4 |
| CLIPPO | WebLI + 50%C4 | 76.6 |
| CLIPPO | C4 only | 78.4 |
| CLIPPO L/16 | WebLI + 50%C4 | **80.0** |

→ C4 co-training으로 PIXEL 수준에 도달, L/16에서 80.0으로 BERT에 근접

#### VQA (VQAv2 test-dev, Figure 2 & Table 7)

| 모델 | 해상도 | Overall |
|---|---|---|
| CLIP* | 224 | 63.31 |
| CLIPPO | 224 | 66.29 |
| CLIPPO 25%C4 | 224 | 66.74 |
| CLIPPO L/16 25%C4 | 384 | **71.78** |
| ViLT B/32 | 384 | 70.33 |
| METER CLIP B/32+BERT | 224 | 69.56 |

→ CLIPPO는 VQA 전용 학습 없이도 태스크-특화 모델(ViLT, METER)과 경쟁적 성능

### 2.5 한계

1. **Co-training 트레이드오프:** C4 비율이 높아질수록 이미지/검색 성능 저하 (e.g., 50%C4에서 ImageNet 0-shot 53.1%로 하락)
2. **깨끗한 렌더링 텍스트 의존:** 문서/웹페이지의 다양한 레이아웃 처리에 한계
3. **생성 모델 부재:** 인코더-전용 구조로 텍스트 생성(캡셔닝 등) 불가
4. **다국어 렌더링 한계:** 아랍어 등 RTL 스크립트의 왼-오 렌더링 등 ad-hoc 설계
5. **CoLA 태스크 실패:** alt-text가 문법적 문장이 아니므로 문법 판단 능력 부재 (CoLA ≈ 0.0~1.8)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 모달리티 간 일반화

CLIPPO의 핵심 일반화 메커니즘은 **모달리티-불가지론적(modality-agnostic) 설계**이다:

- **공유 패치 임베딩:** 이미지와 렌더링된 텍스트가 동일한 $16 \times 16$ 패치 임베딩을 공유함으로써, 모델이 시각적 패턴과 텍스트 패턴을 통합적으로 학습
- **별도 패치 임베딩 실험(Ablation, Table 9):** 이미지/텍스트에 별도 패치 임베딩이나 별도 헤드를 사용해도 성능 향상이 없음 → 통합 임베딩이 일반화에 유리함을 시사
- **VQA 일반화:** 사전학습 시 이미지+텍스트를 결합한 입력을 본 적이 없음에도, 질문을 이미지에 렌더링하기만 하면 VQA에서 66.3의 점수를 달성 → 중간 레이어에서 이미지-텍스트 융합이 자연스럽게 발생

### 3.2 다국어 일반화

CLIPPO의 **토크나이저-프리** 설계는 다국어 일반화에 근본적 이점을 제공한다:

- **CrossModal3600 (36개 언어):** CLIPPO는 32k 어휘의 SentencePiece 토크나이저를 사용하는 1T-CLIP과 동등하거나 우수한 검색 성능 (Figure 4)
- **토크나이제이션 효율성(Figure 3):** C4의 104개 언어 중 대다수에서 CLIPPO가 기존 토크나이저 대비 더 짧은 시퀀스를 생성 — 특히 비라틴 문자 언어에서 유리
- **언어별 균등한 처리:** 토크나이저가 특정 데이터셋에 맞춰 학습되지 않으므로, 모든 언어에 동일한 렌더링 스키마 적용

### 3.3 모달리티 갭 감소를 통한 일반화

논문은 Liang et al. (2022)의 modality gap 분석을 적용한다 (Figure 5):

| 모델 | Modality Gap |
|---|---|
| CLIP* | 0.731 |
| CLIPPO | 0.600 |
| CLIPPO 25%C4 | **0.099** |

- C4 co-training이 모달리티 갭을 **0.099**까지 극적으로 감소시키며, 이미지-텍스트 임베딩이 겹치게 됨
- 이는 **다중 태스크 일반화**의 기반이 될 수 있으나, 검색 성능 저하와의 트레이드오프가 존재

### 3.4 스케일링에 따른 일반화

- **B/16 → L/16 스케일링:** CLIPPO L/16은 GLUE 80.0, VQA 71.78을 달성하며, 1T-CLIP L/16과의 격차가 사라지거나 역전
- **해상도 증가 (224→384px):** VQA에서 모든 모델이 2-3점 향상, CLIPPO L/16 25%C4가 71.78로 최고 성능
- **학습 스텝 증가 (100k→250k):** 영어 전용 CLIPPO와 1T-CLIP 간 격차가 줄어듦

### 3.5 일반화 제약 및 개선 방향

- **시각적 텍스트 표현 한계:** 현재 16px 라인 높이의 고정 렌더링으로, 더 긴 텍스트나 복잡한 레이아웃 처리에 한계 → **동적 해상도**, **다양한 폰트/노이즈 증강**으로 개선 가능
- **Co-training 밸런싱:** 이미지와 텍스트 태스크 간 성능 트레이드오프를 해소하기 위한 **동적 배치 비율 조절**, **그래디언트 밸런싱** 등 필요
- **스펙트로그램 등 추가 모달리티:** 논문은 CLIPPO가 오디오(스펙트로그램) 등으로 확장 가능함을 시사

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구 영향

1. **멀티모달 통합 패러다임 확장:** "모든 것을 이미지로" 접근법이 텍스트뿐 아니라 오디오(스펙트로그램), 구조화된 데이터(테이블) 등으로 확장될 수 있음을 보임
2. **토크나이저-프리 NLP:** 기존 PIXEL(MAE 기반)에 대비, **대조학습만으로** 토크나이저 없이 강력한 언어 이해 가능함을 입증
3. **효율적 멀티모달 모델:** 단일 인코더로 다중 태스크를 수행하는 파라미터-효율적 모델의 가능성 제시
4. **다국어 접근 민주화:** 토크나이저 편향 없이 소수 언어를 포함한 다국어 처리 간소화

### 4.2 향후 연구 시 고려할 점

1. **Co-training 최적화:** 이미지-텍스트 성능과 언어 이해 성능 간 트레이드오프를 해결하기 위한 **커리큘럼 학습**, **동적 배치 비율**, **그래디언트 조정** 전략 연구 필요
2. **생성 모델과의 결합:** 인코더-전용 한계를 극복하기 위해 **픽셀 기반 디코더**(렌더링된 텍스트 생성)나 **토크나이저-프리 생성 모델** 연구
3. **더 다양한 시각적 텍스트 표현:** 폰트 변화, 노이즈 주입, 다양한 레이아웃, 문서 이미지 등을 포함한 학습으로 실제 환경 일반화 향상
4. **대규모 스케일링:** 현재 ViT-L/16까지 실험; ViT-H, ViT-G 등 더 큰 모델에서의 행동 연구
5. **평가 벤치마크 확장:** 현재 GLUE, VQAv2, ImageNet 등에 국한; OCR, 문서 이해, 비디오 등 더 넓은 평가

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근 | CLIPPO와의 비교 |
|---|---|---|---|
| **CLIP** (Radford et al.) | 2021 | 두 개의 분리된 인코더(이미지+텍스트)로 대조학습 | CLIPPO는 단일 인코더로 통합, 절반 파라미터로 유사 성능 |
| **ALIGN** (Jia et al.) | 2021 | 노이즈 웹 데이터로 대규모 대조학습 | CLIPPO는 토크나이저 없이 유사 패러다임 구현 |
| **PIXEL** (Rust et al.) | 2023 (ICLR) | 렌더링된 텍스트에 MAE로 언어 모델링 | CLIPPO는 대조학습만으로 PIXEL을 GLUE에서 상회 (78.4 vs. 76.3); PIXEL은 단일 모달, CLIPPO는 멀티모달 |
| **LIMoE** (Mustafa et al.) | 2022 (NeurIPS) | Mixture of Experts로 단일 타워 대조학습 | CLIPPO는 MoE 없이 완전 공유 파라미터로 유사 성능 달성 |
| **PaLI** (Chen et al.) | 2023 (ICLR) | 대규모 다국어 언어-이미지 모델, 생성형 | CLIPPO는 훨씬 작은 규모지만 인코더-전용으로 유사한 다국어 이점; PaLI는 생성 가능 |
| **CoCa** (Yu et al.) | 2022 | 대조+캡셔닝 결합 학습 | CoCa는 생성 능력 보유; CLIPPO는 대조학습만 사용하는 더 간결한 접근 |
| **ViLT** (Kim et al.) | 2021 (ICML) | 단일 트랜스포머로 이미지 패치+텍스트 토큰 결합 | CLIPPO는 토크나이저 없이 유사 VQA 성능 (70.12 vs. 70.33 @ 384px); ViLT는 다중 목적함수 사용 |
| **BEiT-3** (Wang et al.) | 2022 | "Image as a Foreign Language", 통합 사전학습 | BEiT-3는 토큰화된 텍스트 사용; CLIPPO는 완전 픽셀 기반으로 더 극단적 통합 |
| **Pix2Struct** (Lee et al.) | 2022 | 스크린샷 파싱 기반 시각-언어 이해 | 문서/UI 이해에 특화; CLIPPO는 범용 멀티모달 모델이나 깨끗한 렌더링에 의존 |
| **SimVLM** (Wang et al.) | 2022 (ICLR) | 약한 지도학습으로 간단한 시각 언어 모델 | SimVLM은 생성형+prefix LM; CLIPPO는 인코더-전용+대조학습으로 더 단순 |
| **LiT** (Zhai et al.) | 2022 (CVPR) | 동결된 이미지 인코더 + 텍스트 튜닝 | LiT는 사전학습된 이미지 인코더 활용; CLIPPO는 from-scratch 통합 학습 |
| **MS-CLIP** (You et al.) | 2022 (ECCV) | 모달리티 공유 대조학습, 선택적 모듈 공유 | CLIPPO는 **완전** 공유 (선택적 공유 아님)로 더 극단적 통합 |

### 핵심 비교 통찰

1. **통합의 스펙트럼:** CLIP(완전 분리) → LIMoE/MS-CLIP(부분 공유) → 1T-CLIP(인코더 공유, 임베딩 분리) → **CLIPPO(완전 통합, 토크나이저 제거)** — CLIPPO가 가장 극단적인 통합을 달성
2. **성능-효율성 트레이드오프:** CLIPPO는 파라미터 효율에서 최고이나, CLIP* 대비 1-3%p 성능 하락이 일관되게 나타남
3. **생성 능력 부재:** CoCa, PaLI, SimVLM 등 최신 연구 대비 CLIPPO의 가장 큰 한계; 픽셀 기반 텍스트 생성은 미해결 과제
4. **다국어 이점:** PIXEL과 함께 토크나이저-프리 접근의 다국어 이점을 입증한 중요한 연구

---

## 참고자료

1. Tschannen, M., Mustafa, B., & Houlsby, N. (2023). "CLIPPO: Image-and-Language Understanding from Pixels Only." *arXiv:2212.08045v2* [cs.CV]. (본 논문)
2. Radford, A. et al. (2021). "Learning Transferable Visual Models from Natural Language Supervision." *ICML 2021.*
3. Jia, C. et al. (2021). "Scaling Up Visual and Vision-Language Representation Learning with Noisy Text Supervision (ALIGN)." *ICML 2021.*
4. Rust, P. et al. (2023). "Language Modelling with Pixels (PIXEL)." *ICLR 2023.*
5. Dosovitskiy, A. et al. (2021). "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." *ICLR 2021.*
6. Mustafa, B. et al. (2022). "Multimodal Contrastive Learning with LIMoE." *NeurIPS 2022.*
7. Kim, W. et al. (2021). "ViLT: Vision-and-Language Transformer without Convolution or Region Supervision." *ICML 2021.*
8. Dou, Z.-Y. et al. (2022). "An Empirical Study of Training End-to-End Vision-and-Language Transformers (METER)." *CVPR 2022.*
9. Chen, X. et al. (2023). "PaLI: A Jointly-Scaled Multilingual Language-Image Model." *ICLR 2023.*
10. Wang, W. et al. (2022). "Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks (BEiT-3)." *arXiv:2208.10442.*
11. Yu, J. et al. (2022). "CoCa: Contrastive Captioners are Image-Text Foundation Models." *TMLR 2022.*
12. Zhai, X. et al. (2022). "LiT: Zero-Shot Transfer with Locked-Image Text Tuning." *CVPR 2022.*
13. You, H. et al. (2022). "Learning Visual Representation from Modality-Shared Contrastive Language-Image Pre-training (MS-CLIP)." *ECCV 2022.*
14. Lee, K. et al. (2022). "Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding." *arXiv:2210.03347.*
15. Wang, A. et al. (2019). "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding." *ICLR 2019.*
16. Liang, W. et al. (2022). "Mind the Gap: Understanding the Modality Gap in Multi-Modal Contrastive Representation Learning." *NeurIPS 2022.*
17. Wang, Z. et al. (2022). "SimVLM: Simple Visual Language Model Pretraining with Weak Supervision." *ICLR 2022.*
