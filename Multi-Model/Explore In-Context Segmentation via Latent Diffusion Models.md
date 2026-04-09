# Explore In-Context Segmentation via Latent Diffusion Models

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(LDIS, **L**atent **D**iffusion-based **I**n-context **S**egmentation)은 **Latent Diffusion Model(LDM)을 In-Context Segmentation에 최초로 적용**할 수 있음을 주장합니다. 기존의 판별적(discriminative) 방법이나 Masked Image Modeling(MIM) 기반 방법과 달리, **텍스트 프롬프트나 추가 신경망 없이** 시각적 프롬프트만으로 세그멘테이션을 수행할 수 있음을 보입니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **LDM의 In-Context 세그멘테이션 잠재력 발굴** | 생성 모델을 세그멘테이션 태스크로 최초 전환 |
| **3가지 핵심 요소 분석** | Instruction Extraction, Output Alignment, Meta-architectures |
| **Two-stage Masking 전략** | 배경 정보 누출 방지를 위한 Pre/Post Masking |
| **Augmented Pseudo-masking Target** | 원본 이미지 정보를 보존하는 학습 타겟 설계 |
| **새로운 벤치마크 제안** | 이미지/비디오 세그멘테이션을 통합한 공정한 평가 환경 구성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 방법들의 한계:

1. **판별적 모델**: 카테고리 수/데이터량 제약, 새로운 클래스 일반화 어려움
2. **MIM 기반 방법(Painter, SegGPT)**: 대규모 데이터 필요, 생성 모델의 잠재력 미활용
3. **기존 LDM 기반 방법**: 텍스트 프롬프트 의존, 추가 신경망 필요
4. **Few-shot Segmentation**: 소규모 데이터 오버피팅, 일반화 평가 어려움

```
핵심 질문:
1. LDM이 In-Context Segmentation을 수행할 수 있는가?
2. 성능에 중요한 요소는 무엇이며, 어떤 영향을 미치는가?
```

### 2.2 제안 방법 (수식 포함)

#### (A) 사전 지식: 확산 모델

**순방향 과정(Forward Process)**:

$$q(z_t | z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t} z_0, (1 - \bar{\alpha}_t) I)$$

여기서 $\bar{\alpha}\_t = \prod_{i=0}^{T}(1 - \beta_s)$이고, $\beta$는 노이즈 스케줄러에 의해 결정됩니다.

**학습 손실 (L2 loss)**:

$$\mathcal{L} = \frac{1}{2} ||\epsilon_\theta(z_t, t) - \epsilon(t)||^2$$

**In-Context Segmentation 정의**:

쿼리 이미지 $I_q$와 컨텍스트 집합 $S = (I_i, M_i)_{i=1}^K$가 주어졌을 때:

$$g(I_q, S) \mapsto M_q$$

#### (B) Instruction Extraction: Two-Stage Masking

배경 정보 누출을 막기 위해 **Pre-masking + Post-masking** 2단계 전략을 사용합니다.

$$\tau = F(E_\tau(I_s, M_s)) \tag{1}$$

- $E_\tau$: 프롬프트 인코더 (Alpha-CLIP ViT-L)
- $M_s$: 프롬프트 마스크
- $F$: 선형 투영(linear projection)
- Pre-masking: 입력 단계에서 마스크 적용
- Post-masking: Cross-Attention 레이어에서 어텐션 맵으로 마스크 활용

#### (C) Output Alignment: Pseudo Masking

이진 마스크(1채널)와 이미지(3채널) 간의 불일치를 해소하기 위한 **Pseudo Mask** 설계:

$$\mathcal{M}_{vi} = \begin{cases} (b, a, (a+b)/2), & M_i \in bg \\ (a, b, (a+b)/2), & M_i \in fg \end{cases} \tag{2}$$

여기서 $a < b$, $bg$는 배경, $fg$는 전경을 의미합니다.

**추론 시 마스크 복원**:

$$\tilde{M} = \tilde{\mathcal{M}}_v[1] > \tilde{\mathcal{M}}_v[0] \tag{3}$$

**Augmented Pseudo Mask** (쿼리 이미지 정보 융합):

$$\mathcal{M}_a = (1 - \gamma)\mathcal{M}_v + \gamma I_q \tag{4}$$

$\gamma$는 이미지 정보 강도를 조절하는 파라미터입니다.

#### (D) Meta-Architectures

##### LDIS-1 (단일 스텝 샘플링)

입력 노이즈 latent:

$$z_t = \sqrt{\bar{\alpha}_t} z_q + \sqrt{1 - \bar{\alpha}_t} \epsilon_t$$

**픽셀 공간 최적화**:

$$\mathcal{L}_{fp} = \mathbb{E}_{z_t, \tau} \left[ ||\mathcal{M} - \tilde{\mathcal{M}}_t||_2^2 \right] \tag{5}$$

**잠재 공간 최적화**:

$$\mathcal{L}_{fl} = \mathbb{E}_{z_t, \tau} \left[ ||z_p - \tilde{z}_t||_2^2 \right] \tag{6}$$

여기서 $\tilde{\mathcal{M}}_t = \mathcal{D}(\tilde{z}_t)$, $z_p = \mathcal{E}(\mathcal{M})$.

**비디오 다중 카테고리 처리** (카테고리별 전경 확률 계산):

$$\tilde{p}_c = \frac{\exp(\mathcal{M}[1])}{\exp(\mathcal{M}[0])}, \quad p_c = \frac{\tilde{p}_c}{1 + \sum_{i=1}^{C} \tilde{p}_i} \tag{7}$$

##### LDIS-n (다중 스텝 샘플링)

입력 차원을 4 → 8로 확장하여, 노이즈 추가된 pseudo mask latent와 쿼리 latent를 채널 방향으로 연결:

$$z_t = \text{CONCAT}((\sqrt{\bar{\alpha}_t} z_p + \sqrt{1 - \bar{\alpha}_t} \epsilon_t); z_q) \tag{8}$$

여기서 $z_q \in \mathbb{R}^{4 \times H \times W}$, $z_p \in \mathbb{R}^{4 \times H \times W}$, $z_t \in \mathbb{R}^{8 \times H \times W}$.

**노이즈 예측 손실**:

$$\mathcal{L}_n = \mathbb{E}_{z_t, t, \tau} \left[ ||\epsilon_t - \tilde{z}_t||_2^2 \right] \tag{9}$$

**Classifier-Free Guidance(CFG)**:

$$\tilde{z}_t(z_q, \tau) = \tilde{z}_t(\varnothing, \varnothing) + \gamma_q \cdot (\tilde{z}_t(z_q, \varnothing) - \tilde{z}_t(\varnothing, \varnothing)) + \gamma_\tau \cdot (\tilde{z}_t(z_q, \tau) - \tilde{z}_t(z_q, \varnothing)) \tag{10}$$

### 2.3 모델 구조

```
전체 파이프라인:
[프롬프트 이미지 Is + 프롬프트 마스크 Ms]
        ↓ (Two-stage Masking)
[Alpha-CLIP 프롬프트 인코더 Eτ]
        ↓
[In-context Instructions τ] ──────────────────┐
                                               ↓
[쿼리 이미지 Iq] → [VAE 인코더 E] → [노이즈 추가] → [U-Net (SD 1.5 기반)]
                                                           ↓ Cross-Attention
                                               [예측 z̃t]
                                                    ↓
                                          [VAE 디코더 D]
                                                    ↓
                                          [Pseudo Mask → 이진 마스크]
```

**구현 세부사항**:
- 기반 모델: Stable Diffusion 1.5
- 해상도: $256 \times 256$
- 학습: 160K iterations, batch size 64, AdamW optimizer
- 프롬프트 인코더: Alpha-CLIP ViT-L
- CFG 계수: $\gamma_q = 1.5$, $\gamma_\tau = 7$
- 훈련 데이터: PASCAL + COCO + DAVIS-16 + VSPW

### 2.4 성능 향상

| 방법 | PASCAL mIoU | COCO mIoU | DAVIS-16 J&F |
|------|-------------|-----------|--------------|
| PFENet (판별적) | 76.4 | 49.4 | - |
| SegGPT (MIM) | 75.6 | 50.7 | **80.5** |
| PerSAM (SAM) | 47.6 | 25.5 | 68.7 |
| Prompt Diffusion (LDM) | 9.0 | 5.9 | - |
| **LDIS-n (ours)** | 76.7 | 52.6 | 64.7 |
| **LDIS-1 (ours)** | **85.3** | **62.6** | 67.8 |

**주요 ablation 결과**:

| 설계 요소 | 변형 | mIoU |
|-----------|------|------|
| Instruction (Two-stage) | Pre-mask만 | 43.8 |
| | Post-mask만 | 49.7 |
| | **Pre+Post (full)** | **52.6** |
| Output Alignment | Vanilla $\mathcal{M}_v$ | 39.4 |
| | $+\epsilon$ (perturbation) | 48.7 |
| | **$+I$ ($\mathcal{M}_a$)** | **49.7** |
| Meta-arch (LDIS-1) | LoRA rank 1 | 53.0 |
| | Pixel Space Opt. | 55.6 |
| | **Full train** | **61.3** |

### 2.5 한계점

1. **비디오 세그멘테이션 성능**: 공간적 prior(픽셀 레벨 프롬프트) 미활용으로 SegGPT 대비 비디오 태스크 성능 열위
2. **해상도 제한**: $256 \times 256$ 해상도로 제한되어 세밀한 경계 표현 어려움
3. **데이터 효율성**: 완전 파인튜닝 필요 (LoRA 적용 시 성능 저하 큼)
4. **VAE 변환 오류**: 잠재 공간 최적화 시 VAE 인코더/디코더의 변환 오류 누적
5. **추론 속도**: 특히 LDIS-n의 다중 스텝 샘플링은 실시간 적용 어려움
6. **잘못된 지시(instruction)**: 충돌하는 지시 제공 시 성능 저하

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능하게 하는 핵심 설계

#### (a) In-Context Learning 패러다임

기존 few-shot 방법이 폐쇄 집합(closed-set) 카테고리에 제한되는 것과 달리, **LDIS는 에피소드 학습(episodic learning) 방식**으로 새로운 시각 개념에 동적으로 적응합니다:

$$g(I_q, S) \mapsto M_q, \quad S = \{(I_i, M_i)\}_{i=1}^K$$

학습 시 보지 못한 카테고리도 비주얼 프롬프트만으로 세그멘테이션 가능합니다.

#### (b) Two-Stage Masking의 일반화 기여

배경 정보 누출을 차단함으로써, 모델이 **개념의 본질적 특징**에 집중하도록 유도합니다. 이는 도메인 전이(domain transfer) 시 불필요한 배경 편향을 줄입니다.

```
Pre-masking: 입력 단계에서 배경 제거 → 전경 특징 강조
Post-masking: Attention 레이어에서 관련 토큰만 활성화
결과: 개념 순수성(concept purity) 향상 → 일반화 능력 강화
```

#### (c) Augmented Pseudo-mask의 일반화 기여

$$\mathcal{M}_a = (1 - \gamma)\mathcal{M}_v + \gamma I_q$$

이 설계는 모델이 **단순히 마스크 패턴을 기억하는 것을 방지**하고, 쿼리 이미지의 실제 의미론적 내용을 함께 학습하도록 강제합니다. 이는 미지의 도메인에서도 의미론적 연결을 유지하는 데 기여합니다.

#### (d) Classifier-Free Guidance를 통한 일반화

학습 시 $p=0.05$ 확률로 쿼리와 지시를 null embedding으로 설정:

$$\tilde{z}_t(z_q, \tau) = \tilde{z}_t(\varnothing, \varnothing) + \gamma_q(\tilde{z}_t(z_q, \varnothing) - \tilde{z}_t(\varnothing, \varnothing)) + \gamma_\tau(\tilde{z}_t(z_q, \tau) - \tilde{z}_t(z_q, \varnothing))$$

이를 통해 **프롬프트 의존도를 조절**하여 프롬프트 품질에 robust한 예측이 가능합니다.

#### (e) 복수 프롬프트의 일반화 효과

| 프롬프트 수 | mIoU |
|------------|------|
| 1 | 49.7 |
| 3 | 55.3 |
| 5 | 56.2 |
| 10 | **56.7** |
| 20 | 55.7 |

여러 시각적 관점이 결합되면 **개념의 다양한 변형(intra-class variation)**을 포괄하여 일반화 성능이 향상됩니다.

#### (f) 이미지/비디오 통합 학습

PASCAL, COCO (이미지), DAVIS-16 (VOS), VSPW (VSS)를 **하나의 모델로 공동 학습**함으로써 도메인 간 지식 전이(cross-domain transfer)가 이루어집니다.

### 3.2 일반화의 한계와 극복 방향

| 한계 | 원인 | 극복 방향 |
|------|------|-----------|
| 비디오 spatial prior 미활용 | 임베딩만 사용, 픽셀 레벨 공간 정보 없음 | 공간 어텐션 모듈 추가 |
| LoRA 적용 시 성능 저하 | SD의 생성 태스크 편향 | 세그멘테이션 특화 사전학습 |
| 저해상도 제한 | SD 1.5 아키텍처 한계 | 고해상도 LDM 활용 |

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (a) 생성-인식 통합 패러다임 전환

LDIS는 **"생성 모델 = 인식 불가"라는 통념을 깨는 선구적 연구**입니다. 향후 연구에서:
- 생성 모델을 세그멘테이션/검출/깊이 추정 등 다양한 인식 태스크에 적용하는 연구 활성화
- 텍스트/이미지를 동시에 생성하며 인식하는 통합 아키텍처 연구

#### (b) In-Context Learning의 비전 태스크 확장

LLM의 in-context learning 패러다임을 비전 태스크에 본격 도입하는 방향을 제시:
- 추가 파인튜닝 없는 제로샷 비전 태스크 수행
- 멀티모달 in-context learning으로의 확장

#### (c) 공정한 벤치마크 제공

이미지+비디오 세그멘테이션을 통합한 벤치마크는 향후 in-context segmentation 연구의 **표준 평가 프로토콜**로 활용 가능성이 높습니다.

### 4.2 향후 연구 시 고려사항

#### 기술적 고려사항

1. **고해상도 처리**: 현재 $256 \times 256$ 제한을 극복하기 위한 Hierarchical LDM 또는 패치 기반 처리 전략 필요

2. **효율적 추론**: LDIS-n의 다중 스텝 샘플링 비용 감소를 위해 **일관성 모델(Consistency Models)** 또는 **DDIM** 가속 적용 연구 필요

3. **공간 정보 통합**:

$$z_{\text{spatial}} = \text{CONCAT}(z_\text{semantic}, z_\text{spatial prior})$$

비디오 태스크에서 픽셀 레벨 공간 prior를 latent 공간에서 통합하는 방법 연구

4. **Loss 함수 다양화**: L2 loss 외에 **IoU loss, Dice loss** 등 세그멘테이션 특화 손실 함수 도입으로 경계 정확도 향상 가능

5. **더 강력한 프롬프트 인코더**: CLIP 기반 인코더의 한계를 넘는 **DINOv2, SigLIP** 등 최신 비전 인코더 활용

6. **비디오 시간적 일관성**: 프레임 간 시간적 일관성을 위한 **temporal attention** 또는 **optical flow 기반 가이던스** 통합

#### 연구 방향성 고려사항

7. **데이터 효율성 연구**: Foundation Model의 사전 지식을 더 효과적으로 활용하는 파라미터 효율적 전이학습 방법 (현재 LoRA 적용 시 큰 성능 저하 문제 해결 필요)

8. **오픈 어휘(Open-vocabulary) 확장**: 텍스트+이미지 복합 프롬프트를 활용한 더 유연한 in-context segmentation

9. **3D/포인트 클라우드 확장**: LDM의 3D 생성 능력을 활용한 3D 세그멘테이션으로의 확장

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 In-Context Segmentation 계보

```
GPT-3 (2020) → Visual Prompting/Bar et al. (NeurIPS 2022)
                        ↓
              Painter (CVPR 2023) → SegGPT (ICCV 2023)
                                           ↓
                              LDIS (본 논문, AAAI 2025)
```

### 5.2 주요 방법론 비교 분석

| 방법 | 연도 | 접근법 | 프롬프트 유형 | 데이터 규모 | 이미지 mIoU (COCO) | 특징 |
|------|------|--------|--------------|------------|-------------------|------|
| **Visual Prompting** (Bar et al.) | NeurIPS 2022 | MAE 기반 MIM | 이미지 | 중간 | 낮음 | 최초 비전 in-context |
| **Painter** (Wang et al.) | CVPR 2023 | ViT 기반 MIM | 이미지 | 대규모 | 30.8 | 다중 태스크 통합 |
| **SegGPT** (Wang et al.) | ICCV 2023 | Painter 기반 MIM | 이미지(픽셀) | 대규모 | 50.7 | 공간 prior 활용 |
| **PerSAM** (Zhang et al.) | ICLR 2024 | SAM 파인튜닝 | 이미지 | 소규모 | 25.5 | SAM 기반 1-shot |
| **Prompt Diffusion** (Wang et al.) | NeurIPS 2023 | ControlNet 유사 LDM | 이미지+텍스트 | 중간 | 5.9 | LDM 최초 시도 |
| **LDIS-1 (ours)** | AAAI 2025 | 순수 LDM 기반 | 이미지만 | 중간 | **62.6** | 추가 NN 불필요 |

### 5.3 LDM 기반 세그멘테이션 연구 비교

| 방법 | 연도 | LDM 활용 방식 | 텍스트 의존 | 추가 NN | 특징 |
|------|------|--------------|------------|---------|------|
| **DDPM-Seg** (Baranchuk et al.) | ICLR 2022 | 피처 추출기 | 불필요 | 필요 (디코더) | 레이블 효율적 학습 |
| **VPD** (Zhao et al.) | ICCV 2023 | 피처 추출기 | **필요** | 필요 | SD 피처 활용 |
| **OV-Seg** (Xu et al.) | CVPR 2023 | 피처 추출기 | **필요** | 필요 | 오픈 어휘 파노픽 |
| **InstructDiffusion** (Geng et al.) | arXiv 2023 | 생성 | **필요** | 필요 | 범용 인터페이스 |
| **Prompt Diffusion** | NeurIPS 2023 | ControlNet | **필요** | 필요 | In-context 최초 LDM |
| **UniGS** (Qi et al.) | CVPR 2024 | 생성 | 불필요 | 부분 | 클래스 무관 생성 |
| **LDIS (ours)** | AAAI 2025 | 순수 생성 | **불필요** | **불필요** | 최소한의 수정 |

### 5.4 Few-shot Segmentation과의 비교

| 방법 | 연도 | COCO-20i 1-shot mIoU | 접근법 |
|------|------|---------------------|--------|
| RePRI (Boudiaf et al.) | CVPR 2021 | 34.1 | Transductive 추론 |
| BAM (Lang et al.) | CVPR 2022 | 46.2 | 비분할 정보 학습 |
| FPTrans (Zhang et al.) | NeurIPS 2022 | 47.0 | 피처-프록시 트랜스포머 |
| PerSAM (Zhang et al.) | ICLR 2024 | 22.3 | SAM 기반 |
| **LDIS-1 (ours)** | AAAI 2025 | **60.3** | LDM 기반 |

### 5.5 분석 및 시사점

**LDIS의 차별성**:
- Prompt Diffusion 대비 COCO mIoU **56.7p 향상** (5.9 → 62.6): 텍스트 프롬프트 의존 제거 및 추가 신경망 제거의 효과
- SegGPT 대비 이미지 세그멘테이션 성능 우위 달성: **더 적은 데이터**로 더 높은 성능
- 비디오 세그멘테이션에서는 SegGPT 대비 열위: 공간 정보 활용의 중요성 확인

**연구 트렌드**:
1. 판별적 모델 → 생성 모델로의 패러다임 전환
2. 텍스트 프롬프트 의존 → 순수 시각 프롬프트
3. 태스크 특화 모델 → 통합 일반 모델
4. 대규모 데이터 의존 → 효율적 학습

---

## 참고 자료

1. **Wang, C., et al.** (2025). "Explore In-Context Segmentation via Latent Diffusion Models." *arXiv:2403.09616v2* [AAAI 2025]
2. **Rombach, R., et al.** (2022). "High-resolution image synthesis with latent diffusion models." *CVPR 2022*
3. **Wang, X., et al.** (2023). "SegGPT: Segmenting Everything In Context." *ICCV 2023*
4. **Wang, X., et al.** (2023). "Images Speak in Images: A Generalist Painter for In-Context Visual Learning." *CVPR 2023*
5. **Zhang, R., et al.** (2024). "Personalize Segment Anything Model with One Shot." *ICLR 2024*
6. **Wang, Z., et al.** (2023). "In-context learning unlocked for diffusion models." *NeurIPS 2023*
7. **Zhao, W., et al.** (2023). "Unleashing Text-to-Image Diffusion Models for Visual Perception." *ICCV 2023*
8. **Ho, J., and Salimans, T.** (2021). "Classifier-Free Diffusion Guidance." *NeurIPS Workshops 2021*
9. **Sun, Z., et al.** (2024). "Alpha-CLIP: A CLIP Model Focusing on Wherever You Want." *CVPR 2024*
10. **Brown, T., et al.** (2020). "Language Models are Few-Shot Learners." *NeurIPS 2020*
11. **Kirillov, A., et al.** (2023). "Segment Anything." *ICCV 2023*
12. **Bar, A., et al.** (2022). "Visual Prompting via Image Inpainting." *NeurIPS 2022*
13. **Hu, E. J., et al.** (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*
14. **Project page**: https://wang-chaoyang.github.io/project/refldmseg

> **정확도 주의사항**: 본 답변은 제공된 논문 PDF(arXiv:2403.09616v2)를 직접 참조하여 작성되었으며, 비교 분석 테이블의 일부 수치는 논문 내 Table 3, 8의 데이터를 기반으로 합니다. 논문에 명시되지 않은 타 방법론의 세부 내용은 해당 논문의 원문을 추가 확인하시기 바랍니다.
