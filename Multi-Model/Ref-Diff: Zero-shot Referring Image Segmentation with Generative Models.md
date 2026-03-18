# Ref-Diff: Zero-shot Referring Image Segmentation with Generative Models

---

## 1. 핵심 주장 및 주요 기여 요약

**Ref-Diff**는 Minheng Ni 등(Harbin Institute of Technology, 2023)이 제안한 **Zero-shot Referring Image Segmentation(RIS)** 프레임워크로, 기존에 판별 모델(Discriminative Model, 예: CLIP)에만 의존하던 접근법과 달리 **생성 모델(Generative Model, 예: Stable Diffusion)**의 세밀한 다중 모달 정보를 활용하여 참조 텍스트에 해당하는 이미지 영역을 분할하는 방법을 제안한다.

### 주요 기여 (3가지)
1. **생성 모델의 활용 가능성 입증**: 생성 모델이 시각 요소와 텍스트 설명 사이의 암묵적 관계를 학습하고 있으며, 이를 zero-shot RIS에 효과적으로 활용할 수 있음을 최초로 체계적으로 보여줌.
2. **프로포절 생성기 불필요**: 생성 모델의 cross-attention 맵에서 직접 instance proposal을 추출할 수 있어 **별도의 외부 proposal generator 없이도** 기존 약지도(weakly-supervised) 방법에 필적하는 성능을 달성함.
3. **생성-판별 모델 결합 프레임워크**: 생성 모델과 판별 모델을 상호 보완적으로 결합하여 기존 경쟁 방법들을 **약 10 mIoU 이상** 큰 폭으로 능가하는 성능을 달성.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**Referring Image Segmentation (RIS)**은 주어진 자연어 텍스트 설명에 해당하는 이미지 내 특정 인스턴스 영역을 픽셀 수준에서 분할하는 작업이다. 기존 fully-supervised 방법은 (이미지, 텍스트, 마스크) 트리플릿 어노테이션이 필요하여 비용이 높다.

**Zero-shot RIS**에서는:
- 어떠한 RIS 학습 데이터(이미지, 참조 텍스트, 마스크)도 사용하지 않음
- 사전학습된 모델의 지식만으로 추론 수행

기존 zero-shot 방법들은 주로 CLIP 같은 **판별 모델**에 의존하였으나, CLIP은 이미지-텍스트의 **전역적(global) 유사도**만 학습하므로 **세밀한 시각 요소 지역화(localization)**에 한계가 있다. 이 논문은 **생성 모델이 텍스트와 시각 요소 간의 세밀한 관계를 암묵적으로 학습하고 있음**에 주목하여, 이를 zero-shot RIS에 최초로 체계적으로 활용한다.

### 2.2 제안 방법 (수식 포함)

#### 전체 프레임워크

REF-DIFF는 **Generative Process**와 **Discriminative Process**로 구성되며, 최종 유사도 점수는 두 과정의 가중합으로 결정된다:

$$
\mathbf{s}_i = \alpha \mathbf{s}_i^{\mathrm{G}} + (1 - \alpha) \mathbf{s}_i^{\mathrm{D}}
$$

여기서 $\alpha$는 하이퍼파라미터, $\mathbf{s}_i^{\mathrm{G}}$와 $\mathbf{s}_i^{\mathrm{D}}$는 각각 생성 모델과 판별 모델이 $i$번째 proposal에 대해 계산한 유사도 점수이다. 최종 분할 마스크는:

$$
\hat{\mathbf{m}} = \arg\max_{\mathcal{M}_i} \mathbf{s}_i
$$

#### (A) Generative Process (Stable Diffusion 기반)

**Step 1: 노이즈 추가 (DDIM Inversion)**

실제 이미지 $\mathbf{x}$에 특정 단계 $t$까지 가우시안 노이즈를 추가하여 잠재 표현을 얻는다:

$$
\mathbf{x_t} = \sigma_t(\mathbf{x})
$$

**Step 2: Cross-Attention 계산**

역확산(inverse diffusion) 과정에서, 텍스트 인코더 $\Psi_{\text{lan}}$과 이미지 인코더 $\Psi_{\text{vis}}$를 사용하여:
- 텍스트 특징: $\mathbf{K} = \Psi_{\text{lan}}(T) \in \mathbb{R}^{l \times d}$ (토큰 수 $l$, 차원 $d$)
- 이미지 특징: $\mathbf{Q} = \Psi_{\text{vis}}(\mathbf{x}_i) \in \mathbb{R}^{w \times h \times d}$

Cross-attention:

$$
\mathbf{a} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)
$$

여기서 $\mathbf{a} \in \mathbb{R}^{w \times h \times l \times N}$이고, $N$은 attention head 수이다. 모든 head를 평균하여 $\bar{\mathbf{a}} \in \mathbb{R}^{w \times h \times l}$을 얻는다.

**Step 3: Root Token 선택 및 정규화**

구문 분석(syntax analysis)으로 참조 텍스트의 **ROOT 토큰**을 추출한다. ROOT 토큰 $k$에 대한 cross-attention 맵 $\bar{\mathbf{a}}_k \in \mathbb{R}^{w \times h}$를 정규화 및 리사이즈:

$$
\mathbf{c} = \phi_{w \times h \rightarrow W \times H}\left(\frac{\bar{\mathbf{a}}_k - \min(\bar{\mathbf{a}}_k)}{\max(\bar{\mathbf{a}}_k) - \min(\bar{\mathbf{a}}_k) + \epsilon}\right)
$$

여기서 $\phi_{w \times h \rightarrow W \times H}$는 쌍선형 보간(bi-linear interpolation) 함수이다.

**핵심 통찰**: ROOT 토큰은 문장의 맥락적 상관관계를 포착하며(예: "black horse jumping"에서 "horse" 토큰이 "black"과 "jumping"의 문맥 정보를 함께 인코딩), 해당 attention 영역이 참조 영역과 높은 확률로 일치한다.

**Step 4: Weight-free Proposal Filter**

생성 모델의 cross-attention 맵 $\mathbf{c}$에서 직접 마스크 proposal을 추출:

```math
\mathcal{M} = \{\psi(\mathbf{c} \geq \mu) \mid \mu \in \{5\%, 10\%, \ldots, 95\%\}\}
```

여기서 $\psi$는 이진화 함수이다. 이를 통해 **외부 proposal generator 없이** proposal 생성이 가능하다.

**Step 5: Generative Matching**

각 proposal $\mathcal{M}_i$와 cross-attention 맵 $\mathbf{c}$ 간의 유사도:

$$
\mathbf{s}_i^{\mathrm{G}} = \frac{|\mathbf{c} \odot \mathcal{M}_i|}{|\mathcal{M}_i|} - \frac{|\mathbf{c} \odot (1 - \mathcal{M}_i)|}{|1 - \mathcal{M}_i|}
$$

이 수식은 proposal 내부의 평균 attention 강도에서 외부의 평균 attention 강도를 뺀 값으로, **대비(contrast) 기반 매칭**이다.

#### (B) Discriminative Process (CLIP 기반)

**Positional Bias**: 참조 텍스트에 방향 단서(left, right, top, bottom)가 포함된 경우 이를 명시적으로 인코딩:

$$
\mathbf{x}' = \mathbf{x} \odot \mathbf{P}
$$

$\mathbf{P} \in \mathbb{R}^{W \times H \times C}$는 방향 축을 따라 1에서 0으로 감소하는 소프트 마스크이다.

**Proposal 표현**:

$$
\mathbf{v}_i = \beta f_{\mathcal{M}_i}(\mathbf{x} \odot \mathbf{P}) + (1 - \beta) f(\mathbf{x} \odot \mathcal{M}_i)
$$

여기서 $f$는 바닐라 CLIP 이미지 인코더, $f_{\mathcal{M}_i}$는 mask proposal $\mathcal{M}_i$ 기반으로 self-attention이 수정된 CLIP 인코더이다. `[CLS]` 토큰과 proposal 외부 패치 토큰 간의 attention 가중치를 0으로 설정하여 해당 영역에만 집중한다.

**Discriminative Matching**:

$$
\mathbf{s}_i^{\mathrm{D}} = \mathbf{v}_i \mathbf{r}^\top
$$

여기서 $\mathbf{r} \in \mathbb{R}^d$는 CLIP 텍스트 인코더로 얻은 참조 텍스트의 평균 표현이다.

### 2.3 모델 구조

| 구성 요소 | 역할 | 모델 |
|---------|------|------|
| Generative Process | Cross-attention 기반 상관 맵 생성, proposal 추출, 생성적 매칭 | Stable Diffusion V1.5 |
| Discriminative Process | 위치 편향 적용, proposal별 시각 표현 인코딩, 판별적 매칭 | CLIP (ViT-B/16) |
| Proposal Generator | (선택적) 세밀한 인스턴스 분할 | SAM 또는 Weight-free Filter |
| 구문 분석기 | 참조 텍스트에서 ROOT 토큰 및 방향 단서 추출 | Syntax Parser |

**주요 하이퍼파라미터**: $t = 2$, $\alpha = 0.1$, $\beta = 0.3$

### 2.4 성능 향상

#### 정량적 결과 (mIoU 기준)

| 방법 | RefCOCO val | RefCOCO+ val | RefCOCOg val(U) |
|------|------------|-------------|----------------|
| TSEG (약지도) | 25.95 | 22.62 | 23.41 |
| Global-Local CLIP | 26.20 | 27.80 | 33.52 |
| SAM-CLIP | 26.33 | 25.70 | 38.75 |
| **REF-DIFF** | **37.21** | **37.29** | **44.02** |

- 기존 최고 zero-shot 방법 대비 **약 10 mIoU 이상 향상**
- 약지도 학습 방법(TSEG)보다도 크게 우수
- PhraseCut 데이터셋에서도 oIoU 기준 23.64 → **29.42**로 향상 (일반화 성능 입증)

#### Ablation Study 결과 (mIoU, RefCOCO val)

| 변형 | Segmentor | Generative | Discriminative | mIoU |
|------|-----------|------------|----------------|------|
| REF-DIFF/G | ✗ | ✓ | ✗ | 21.53 |
| REF-DIFF/GS | ✓ | ✓ | ✗ | 29.82 |
| REF-DIFF/DS | ✓ | ✗ | ✓ | 35.27 |
| **REF-DIFF** | ✓ | ✓ | ✓ | **37.21** |

**REF-DIFF/G (생성 모델 단독)**만으로도 약지도 방법 TSEG와 비교 가능한 성능을 달성하였으며, 특히 RefCOCOg에서는 TSEG를 능가 (27.03 vs 23.41).

### 2.5 한계점

1. **높은 추론 비용**: Stable Diffusion + CLIP + SAM의 사전학습 모듈들을 모두 사용하므로 추론 시 **높은 계산 오버헤드**가 발생
2. **모호한 참조 표현에 대한 취약성**: 참조 텍스트에 모호성이 있을 경우(예: "second arm from left") 잘못된 분할이 발생 (Figure 6 실패 사례)
3. **ROOT 토큰 의존성**: 구문 분석에 의한 ROOT 토큰 추출이 항상 최적의 참조 토큰을 선택하지는 않을 수 있음
4. **생성 모델의 편향**: 생성 모델이 시각적으로 두드러진 특징(salient feature)에 과도하게 집중하여 불완전한 분할을 생성할 수 있음 (예: 노트북의 화면만 분할)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능의 근거

REF-DIFF의 일반화 성능은 다음 요인들에 기인한다:

**(1) Zero-shot 설정의 본질적 일반화**
- 어떠한 RIS 학습 데이터도 사용하지 않으므로, 특정 데이터셋에 대한 과적합(overfitting)이 원천적으로 방지됨
- Stable Diffusion과 CLIP이 **대규모 다중 모달 데이터**로 사전학습되어 광범위한 시각-언어 개념을 내재화

**(2) 도메인 간 일반화 입증**
- RefCOCO, RefCOCO+, RefCOCOg, PhraseCut 등 **4개의 서로 다른 벤치마크**에서 일관된 성능 향상을 보임
- 특히 PhraseCut(Table 3)에서의 성능 향상은 학습에 사용되지 않은 데이터셋에서도 **강건한 일반화**가 가능함을 입증

**(3) 생성-판별 모델의 상호 보완성**
- 생성 모델: **세밀한 지역화(localization)** 능력이 우수하나 시각적 두드러짐(saliency)에 편향
- 판별 모델: **범주 분류(categorization)** 능력이 우수하나 공간적 정밀도가 부족
- 두 모델의 결합이 각각의 약점을 보완하여 **다양한 시나리오에서의 일반화** 성능을 향상

### 3.2 일반화 성능 향상을 위한 향후 방향

**(1) 더 강력한 생성 모델 활용**
- SDXL, Stable Diffusion 3, DALL-E 3 등 최신 생성 모델로의 교체 시 cross-attention의 품질 향상이 예상됨
- 더 높은 해상도의 attention 맵을 제공하는 모델을 통해 세밀한 분할 정확도 개선 가능

**(2) 다중 토큰 활용**
- 현재 ROOT 토큰 하나에만 의존하는 대신, **여러 토큰의 attention 맵을 가중 결합**하여 복합적인 참조 표현에 대한 일반화 가능

**(3) 도메인 적응 없는 확장성**
- 의료 영상, 위성 영상, 산업 검사 등 **다양한 도메인**으로의 확장 가능성
- Stable Diffusion이 학습한 시각적 개념이 해당 도메인을 충분히 포함하는지가 핵심 제한 요인

**(4) 프롬프트 엔지니어링 및 구문 분석 강화**
- LLM 기반의 더 정교한 참조 텍스트 해석을 통해 모호한 표현에 대한 강건성 향상
- 복잡한 관계 표현(spatial reasoning)의 이해 개선

---

## 4. 연구 영향 및 향후 고려사항

### 4.1 학술적 영향

1. **패러다임 전환**: 생성 모델을 순수 생성 작업이 아닌 **비전-언어 이해 태스크의 핵심 도구**로 활용할 수 있음을 입증하여, 판별 모델 중심의 기존 패러다임에 새로운 방향을 제시
2. **생성-판별 모델 융합**: 두 유형의 모델이 상호 보완적임을 정량적으로 보여주어, **멀티 모달 모델 앙상블**의 새로운 연구 방향을 개척
3. **Training-free 접근법의 실용성**: 학습 없이도 높은 성능을 달성할 수 있음을 보여, **데이터 부족 환경**에서의 AI 배포 가능성을 확대

### 4.2 실용적 영향

- **이미지 편집**: 자연어로 지정한 영역의 정밀 분할을 통한 인터랙티브 편집
- **로봇 제어**: 자연어 명령 기반의 객체 인식 및 조작
- **인간-기계 상호작용**: 배포 비용 절감으로 실제 애플리케이션 적용 가속화

### 4.3 향후 연구 시 고려사항

| 고려 사항 | 세부 내용 |
|----------|---------|
| **계산 효율성** | Stable Diffusion + CLIP + SAM의 동시 사용에 따른 높은 계산 비용을 경량화하는 방법 연구 필요 |
| **모호성 처리** | 참조 표현의 모호성에 강건한 해석 메커니즘 개발 (LLM 기반 disambiguation 등) |
| **단일 통합 모델** | 생성-판별 기능을 하나의 모델에 통합하여 효율성 향상 (예: 최신 멀티모달 LLM 활용) |
| **시간적 확장** | 비디오 참조 분할(video referring segmentation)로의 확장 |
| **평가 다양성** | 더 다양한 도메인 및 언어에서의 평가를 통한 일반화 검증 |
| **Attention 해석** | 생성 모델 내부 attention의 이론적 분석 및 최적 활용 방법 연구 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 비교 연구 개요

| 논문 | 연도 | 접근 방식 | 핵심 특징 | Zero-shot 여부 |
|------|------|---------|---------|-------------|
| **CLIP** (Radford et al.) | 2021 | 판별 (대조학습) | 이미지-텍스트 전역 유사도 학습 | 가능 (간접적) |
| **Stable Diffusion** (Rombach et al.) | 2022 | 생성 (Latent Diffusion) | 텍스트→이미지 생성, 세밀한 cross-attention | 원래 생성 목적 |
| **TSEG** (Strudel et al.) | 2022 | 약지도 학습 | 이미지-텍스트 쌍으로 약지도 학습 | ✗ |
| **Global-Local CLIP** (Yu et al.) | 2023 | Zero-shot 판별 | CLIP의 전역/지역 특징 결합, FreeSOLO proposal | ✓ |
| **SAM** (Kirillov et al.) | 2023 | Promptable 분할 | 범용 세그멘테이션 모델 | ✓ (분할만) |
| **DiffuMask** (Wu et al.) | 2023 | 합성 데이터 생성 | Diffusion으로 세그멘테이션 학습 데이터 합성 | ✗ |
| **VPD/DDP** (Ji et al.) | 2023 | 전이 학습 | Diffusion 특징을 dense prediction에 전이 | ✗ |
| **Ref-Diff (본 논문)** | 2023 | **Zero-shot 생성+판별** | **Diffusion cross-attention + CLIP 결합** | **✓** |

### 5.2 핵심 비교 분석

#### CLIP 기반 방법 vs. Ref-Diff

**CLIP** (Radford et al., ICML 2021)은 대조 학습을 통해 이미지-텍스트의 **전역 유사도**를 학습한다. 이로 인해:
- **장점**: 강력한 zero-shot 분류 능력
- **한계**: 픽셀 수준의 세밀한 지역화 불가, 같은 클래스의 다른 인스턴스 구분 어려움

**Global-Local CLIP** (Yu et al., CVPR 2023)은 CLIP의 전역-지역 특징을 결합하여 이 한계를 부분적으로 극복하였으나, 여전히 판별 모델의 본질적 한계(공간 정보 손실)를 완전히 해결하지 못했다.

Ref-Diff는 생성 모델의 cross-attention을 통해 **각 토큰별 공간적 attention 분포**를 직접 활용하므로, CLIP만으로는 불가능한 **세밀한 공간 지역화**를 달성한다.

#### Diffusion 모델 활용 연구 비교

| 연구 | Diffusion 활용 방식 | RIS 적용 | 학습 필요 여부 |
|------|-----------------|---------|-----------|
| DiffuMask (Wu et al., 2023) | 합성 학습 데이터 생성 | 간접적 | 합성 데이터로 학습 필요 |
| Label-efficient Seg (Baranchuk et al., ICLR 2022) | Diffusion 특징 추출 후 세그멘테이션 | 의미론적 분할만 | 소량 학습 필요 |
| DDP (Ji et al., 2023) | Dense prediction을 위한 전이 학습 | ✗ | 학습 필요 |
| **Ref-Diff** | **Cross-attention 직접 활용 (training-free)** | **✓ (RIS)** | **✗ (zero-shot)** |

Ref-Diff의 차별점은 Diffusion 모델의 cross-attention을 **학습 없이 직접** RIS에 활용한다는 것이다. 기존 연구들은 주로 transfer learning이나 합성 데이터 생성에 초점을 맞추었다.

#### SAM과의 관계

**SAM** (Kirillov et al., 2023)은 범용 세그멘테이션 모델로, 어떤 프롬프트에 대해서도 마스크를 생성할 수 있다. 그러나:
- SAM 자체는 **의미적 이해 능력이 없음** (어떤 객체인지 판단 불가)
- SAM + CLIP (SAM-CLIP 베이스라인)만으로는 개선 폭이 미미 (Table 2: 26.33 mIoU)
- Ref-Diff는 SAM의 proposal 위에 **생성 모델의 세밀한 의미적 매칭**을 추가하여 큰 폭의 성능 향상을 달성 (37.21 mIoU)

### 5.3 최근 후속 연구 동향 (2023~)

Ref-Diff 이후, 생성 모델을 비전-언어 이해에 활용하는 연구가 활발히 진행되고 있다:

1. **멀티모달 LLM 기반 접근**: GPT-4V, LLaVA 등의 멀티모달 LLM을 활용한 참조 분할 연구가 증가
2. **Grounded SAM 계열**: Grounding DINO + SAM의 결합으로 텍스트 기반 분할을 수행하는 방법들이 등장
3. **Diffusion 기반 dense prediction**: VPD(Visual Perception with Diffusion)와 같이 diffusion 특징을 다양한 dense prediction 태스크에 활용하는 연구가 확대

---

## 참고 자료

1. Ni, M., Zhang, Y., Feng, K., Li, X., Guo, Y., & Zuo, W. (2023). "Ref-Diff: Zero-shot Referring Image Segmentation with Generative Models." *arXiv preprint arXiv:2308.16777v2.* — 본 논문
2. Radford, A. et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021.* — CLIP
3. Rombach, R. et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022.* — Stable Diffusion
4. Yu, S., Seo, P.H., & Son, J. (2023). "Zero-shot Referring Image Segmentation with Global-Local Context Features." *CVPR 2023.* — Global-Local CLIP
5. Kirillov, A. et al. (2023). "Segment Anything." *arXiv:2304.02643.* — SAM
6. Strudel, R., Laptev, I., & Schmid, C. (2022). "Weakly-supervised Segmentation of Referring Expressions." *arXiv:2205.04725.* — TSEG
7. Wu, W. et al. (2023). "DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models." *arXiv:2303.11681.* — DiffuMask
8. Baranchuk, D. et al. (2022). "Label-Efficient Semantic Segmentation with Diffusion Models." *ICLR 2022.*
9. Ji, Y. et al. (2023). "DDP: Diffusion Model for Dense Visual Prediction." *arXiv:2303.17559.*
10. Li, A.C. et al. (2023). "Your Diffusion Model is Secretly a Zero-Shot Classifier." *arXiv:2303.16203.*
11. Clark, K. & Jaini, P. (2023). "Text-to-Image Diffusion Models are Zero-Shot Classifiers." *arXiv:2303.15233.*
12. GitHub Repository: [https://github.com/kodenii/Ref-Diff](https://github.com/kodenii/Ref-Diff)
