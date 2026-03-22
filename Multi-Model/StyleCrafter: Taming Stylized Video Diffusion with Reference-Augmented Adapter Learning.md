# StyleCrafter: Taming Stylized Video Diffusion with Reference-Augmented Adapter Learning

---

## 1. 핵심 주장 및 주요 기여 (Summary)

Text-to-Video(T2V) 모델은 다양한 비디오 생성에서 뛰어난 능력을 보여주었으나, (i) 텍스트가 특정 스타일을 표현하는 데 본질적으로 부적합하고, (ii) 스타일 충실도(style fidelity)가 일반적으로 저하되는 문제로 인해 사용자가 원하는 예술적 비디오 생성에 어려움을 겪고 있다.

이를 해결하기 위해 StyleCrafter는 사전 학습된 T2V 모델에 **스타일 제어 어댑터(style control adapter)**를 추가하여, 참조 이미지를 제공하면 어떤 스타일로든 비디오를 생성할 수 있는 범용 방법을 제안한다.

### 주요 기여 (Contributions)

본 논문의 기여는 다음 세 가지로 요약된다:
1. 사전 학습된 T2V 모델에 스타일 어댑터를 추가하여 스타일화 생성을 개선하는 개념 제안
2. 텍스트와 이미지 입력으로부터 콘텐츠-스타일 분리 생성을 촉진하는 효율적 네트워크 탐색
3. 기존 베이스라인 대비 현저한 성능 우위 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

StyleCrafter가 해결하고자 하는 핵심 문제는 두 가지이다:
(i) 스타일 전이/보존의 고전적 문제로서, 스타일 제어 어댑터가 참조 이미지로부터 **콘텐츠-스타일이 분리된 방식**으로 정확한 스타일 개념을 추출해야 한다.
(ii) 오픈소스 스타일화 비디오의 **희소성**이 T2V 모델의 적응 학습을 어렵게 한다.

기존 비디오는 대부분 포토리얼리즘으로 촬영되어 스타일 데이터의 다양성이 부족하며, T2I 모델의 가중치를 초기화하거나 이미지-비디오 데이터셋을 공동으로 학습하는 전략이 있지만, 생성된 스타일화 비디오는 일반적으로 스타일 충실도가 저하된다.

### 2.2 제안하는 방법

#### (A) 2단계 학습 전략 (Two-Stage Training)

스타일화 비디오 데이터셋의 희소성을 고려하여, 먼저 **스타일이 풍부한 이미지 데이터셋**으로 스타일 제어 어댑터를 학습한 후, 맞춤형 미세 조정 패러다임을 통해 학습된 스타일화 능력을 비디오 생성에 전이하는 방식을 제안한다.

이 접근의 장점은 이중적이다: 한편으로는 스타일화 이미지로 학습된 어댑터가 입력 이미지로부터 스타일 개념을 효과적으로 추출할 수 있어 희소한 스타일화 비디오의 필요성을 제거하고, 다른 한편으로는 미세 조정 패러다임이 T2V 모델의 비디오 시간적 품질 저하를 방지하면서 스타일 개념에 더 잘 적응하도록 한다.

#### (B) 모델 구조 (Architecture)

스타일 어댑터는 세 가지 핵심 컴포넌트로 구성된다: **스타일 특징 추출기(Style Feature Extractor)**, **이중 교차-어텐션 모듈(Dual Cross-Attention Module)**, **문맥 인식 스케일 팩터 예측기(Context-Aware Scale Factor Predictor)**.

**① 스타일 특징 추출기 (Style Feature Extractor)**

콘텐츠-스타일 분리를 효과적으로 포착하고 촉진하기 위해, 널리 사용되는 **Query Transformer**를 채택하여 단일 이미지로부터 스타일 개념을 추출한다.

CLIP 이미지 인코더를 통해 참조 이미지의 특징을 추출한 후, 학습 가능한 쿼리 토큰 $Q_s$를 사용하는 Query Transformer로 스타일 임베딩을 획득한다:

$$f_{\text{style}} = \text{QueryTransformer}(Q_s, \; \text{CLIP}_{\text{image}}(I_{\text{ref}}))$$

여기서 $I_{\text{ref}}$는 참조 스타일 이미지이고, $Q_s \in \mathbb{R}^{N \times d}$는 $N$개의 학습 가능한 쿼리 토큰이다.

**② 이중 교차-어텐션 모듈 (Dual Cross-Attention)**

기존 T2V 모델의 각 cross-attention 레이어 옆에 **스타일 전용 cross-attention 레이어**를 병렬로 추가한다. U-Net 잠재 특징 $z$에 대해:

$$\text{Attn}_{\text{text}}(z) = \text{Softmax}\!\left(\frac{Q_z \cdot K_{\text{text}}^T}{\sqrt{d_k}}\right) V_{\text{text}}$$

$$\text{Attn}_{\text{style}}(z) = \text{Softmax}\!\left(\frac{Q_z \cdot K_{\text{style}}^T}{\sqrt{d_k}}\right) V_{\text{style}}$$

여기서 $Q_z = W_Q \cdot z$이고, $K_{\text{text}}, V_{\text{text}}$는 텍스트 임베딩으로부터, $K_{\text{style}}, V_{\text{style}}$는 스타일 임베딩으로부터 각각 프로젝션된다.

**③ Scale-Adaptive Fusion Module**

텍스트 기반 콘텐츠 특징과 이미지 기반 스타일 특징의 영향을 균형 있게 조절하는 **스케일 적응형 융합 모듈**을 설계하여, 다양한 텍스트-스타일 조합에 대한 일반화를 돕는다.

문맥 인식 스케일 팩터 $\alpha$를 동적으로 예측하여 두 어텐션 출력을 융합한다:

$$\alpha = \sigma\!\left(\text{MLP}([f_{\text{text}}; f_{\text{style}}])\right)$$

$$h = (1 - \alpha) \cdot \text{Attn}_{\text{text}}(z) + \alpha \cdot \text{Attn}_{\text{style}}(z)$$

여기서 $\sigma$는 시그모이드 함수이고, $[\cdot;\cdot]$는 연결(concatenation)이다.

#### (C) 콘텐츠-스타일 분리 전략

콘텐츠-스타일 분리를 촉진하기 위해, **텍스트 프롬프트에서 스타일 설명을 제거**하고, 분리 학습 전략을 통해 참조 이미지에서만 스타일 정보를 추출한다.

또한 **세심하게 설계된 데이터 증강 전략**을 사용하여 분리 학습을 강화한다. 구체적으로, 학습 중 참조 이미지와 대상 이미지에 서로 다른 랜덤 크롭 및 색상 변환을 적용하여 콘텐츠 누출(content leakage)을 방지한다.

#### (D) 학습 목적 함수

Diffusion 모델의 표준 노이즈 예측 손실을 기반으로 학습한다:

$$\mathcal{L} = \mathbb{E}_{z_0, \epsilon \sim \mathcal{N}(0,I), t, c_{\text{text}}, c_{\text{style}}} \left[\| \epsilon - \epsilon_\theta(z_t, t, c_{\text{text}}, c_{\text{style}}) \|_2^2 \right]$$

여기서 $z_t$는 타임스텝 $t$에서의 노이즈가 추가된 잠재 변수, $c_{\text{text}}$는 텍스트 조건, $c_{\text{style}}$는 스타일 참조 조건이다.

### 2.3 성능 향상

포괄적 실험을 통해 StyleCrafter가 스타일화 이미지 생성과 스타일화 비디오 생성 모두에서 기존 경쟁자들을 현저히 능가함을 입증하였다.

광범위한 실험이 제안된 설계의 효과를 증명하였고, 기존 경쟁자들과의 비교에서 시각적 품질과 효율성 면에서 본 방법의 우월성을 보여주었다.

비교 대상으로는 DreamBooth, InST(Inversion-based Style Transfer), IP-Adapter 등이 포함되며, StyleCrafter는 별도의 스타일별 파인튜닝 없이 단일 참조 이미지만으로 다양한 스타일의 비디오를 생성할 수 있어 **유연성과 효율성**에서 우위를 보인다.

### 2.4 한계

본 방법은 일정한 한계를 가지고 있는데, 예를 들어 참조 이미지가 대상 스타일을 충분히 나타내지 못하거나, 제시된 스타일이 극도로 보지 못한 스타일(extremely unseen)인 경우 바람직한 결과를 생성하지 못한다.

또한 StyleStudio(2024) 논문에서 지적된 바와 같이, StyleCrafter의 주목할 만한 문제점은 **콘텐츠 누출(content leakage)**로, 스타일 이미지의 관련 없는 콘텐츠 요소가 생성 결과에 나타나 최종 출력에 영향을 미치는 현상이 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

StyleCrafter의 일반화 성능은 아래 세 가지 핵심 설계에서 비롯된다:

### 3.1 Scale-Adaptive Fusion의 역할

텍스트 기반 콘텐츠 특징과 이미지 기반 스타일 특징의 영향을 균형 있게 조절하는 스케일 적응형 융합 모듈을 설계하여, 다양한 텍스트-스타일 조합에 대한 **일반화를 돕는다**.

동적으로 $\alpha$를 예측하는 것이 핵심으로, 이는 특정 스타일-콘텐츠 조합에 고정된 비율을 사용하는 것 대비 **보지 못한 조합에 대한 강건성**을 높인다.

### 3.2 이미지→비디오 전이 패러다임

스타일화 비디오의 희소성을 고려하여, 먼저 이미지 데이터셋으로 스타일 어댑터를 학습하고, 그 후 **공유된 공간적 가중치(shared spatial weights)**를 통해 학습된 스타일화 능력을 T2V 모델로 전이하는 맞춤형 미세 조정 패러다임을 제안한다.

이 전략 덕분에 풍부한 이미지 스타일 데이터를 활용할 수 있어, 비디오 도메인에서의 데이터 부족 문제를 우회하고 **다양한 스타일에 대한 일반화**를 달성한다.

### 3.3 데이터 증강 기반 분리 학습

학습 과정에서 세심하게 설계된 데이터 증강 전략을 적용하여 분리 학습을 강화한다. 이 전략은 스타일 어댑터가 콘텐츠에 과적합(overfit)되지 않도록 하여, 추론 시 학습에 포함되지 않았던 새로운 스타일-콘텐츠 조합에 대해서도 안정적인 생성을 가능하게 한다.

### 3.4 향후 일반화 성능 개선 방향

- **더 큰 규모의 스타일 데이터셋** 활용으로 어댑터의 스타일 표현 범위 확장
- **콘텐츠 누출 문제 완화**: StyleStudio에서 제안한 Teacher Model 도입이 StyleCrafter에 적용될 때 콘텐츠 누출 발생을 효과적으로 줄여준다.
- DiT(Diffusion Transformer) 기반 최신 T2V 모델로의 확장을 통한 기반 모델 역량 강화

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

이 연구는 텍스트-비디오 생성 분야에서 중요한 진전을 나타내며, 콘텐츠-스타일 분리와 효과적인 스타일 전이에 대한 강조는 이 분야의 향후 발전에 대한 가능성을 보여준다.

1. **어댑터 기반 스타일 제어 패러다임 확립**: 모델 전체를 재학습하지 않고 플러그인 방식의 어댑터만 추가하여 새로운 제어 능력을 부여하는 패러다임을 스타일 비디오 생성 분야에서 선도적으로 제시
2. **이미지→비디오 지식 전이 프레임워크**: 데이터 부족 문제를 이미지 도메인의 풍부한 데이터로 우회하는 2단계 학습 전략은 다른 비디오 생성 과제에도 적용 가능
3. **멀티모달 조건부 생성의 발전**: 텍스트(콘텐츠)와 이미지(스타일)라는 이종 조건을 효과적으로 결합하는 융합 메커니즘을 제시

### 4.2 향후 연구 시 고려 사항

| 고려 사항 | 설명 |
|---|---|
| **콘텐츠 누출 완화** | 스타일 참조에서 비의도적 콘텐츠가 생성물에 전이되는 문제의 근본적 해결 필요 |
| **극단적 스타일 일반화** | 학습 분포에서 크게 벗어난 스타일에 대한 대응력 강화 |
| **시간적 일관성 강화** | 긴 비디오 생성 시 스타일의 시간적 일관성 유지 |
| **DiT 아키텍처 호환** | U-Net 기반에서 Transformer 기반(DiT)으로 진화하는 최신 T2V 모델과의 통합 |
| **효율성** | 어댑터 크기 및 추론 비용의 최적화 |
| **다중 스타일 조합** | 여러 참조 이미지에서 스타일 요소를 선택적으로 조합하는 능력 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근법 | StyleCrafter 대비 특징 |
|---|---|---|---|
| **AnimateDiff** (Guo et al.) | 2023 | 개인화된 T2I 확산 모델을 특정 튜닝 없이 애니메이션화 | 모션 모듈 삽입 방식; 스타일 제어는 T2I 모델(LoRA 등)에 의존 |
| **Text2Video-Zero** (Khachatryan et al.) | 2023 | T2I 확산 모델이 제로샷 비디오 생성기 역할 | 학습 없이 비디오 생성하지만 스타일 제어 메커니즘 부재 |
| **VideocrAfter2** (Chen et al.) | 2024 | 데이터 한계를 극복한 고품질 비디오 확산 모델 | 기반 T2V 모델 개선에 집중; 별도 스타일 어댑터 없음 |
| **Lumiere** (Bar-Tal et al.) | 2024 | 전체 프레임률의 비디오 클립을 직접 생성하는 Space-Time U-Net 아키텍처 도입 | 시간적 일관성 극대화; 스타일화는 가중치 보간 방식으로 제한적 |
| **Sora** (Brooks et al.) | 2024 | DiT 아키텍처를 활용하여 시공간 패치 위에서 동작; 시각적 입력을 시공간 패치 시퀀스로 표현 | 대규모 Transformer 기반; 범용 비디오 생성에 탁월하지만 참조 기반 스타일 제어는 명시적이지 않음 |
| **StyleStudio** (2024) | 2024 | 이미지 스타일을 효과적으로 보존하면서 텍스트 프롬프트에 정확하게 따르는 생성 | Teacher Model과 cross-modal AdaIN을 통해 StyleCrafter의 콘텐츠 누출 문제를 개선 |
| **ArtCrafter** (2025) | 2025 | 텍스트-이미지 스타일 전이를 위한 새로운 프레임워크로, 어텐션 기반 스타일 추출 모듈을 도입하여 이미지 내의 미묘한 스타일 요소를 포착 | 다층 perceiver attention으로 미세한 스타일 캡처; StyleCrafter가 텍스트와 더 일관된 출력을 생성하지만 스타일이 참조 이미지와 상당히 다른 경우가 있음을 지적 |
| **CSGO** (2024) | 2024 | 스타일 전이에서 가장 효과적이고 최첨단 방법 중 하나로 인정 | 대규모 아트 데이터셋으로 콘텐츠-스타일 분리 성능 극대화 |

### 기술 동향 요약

확산 모델의 최근 발전은 비디오 생성을 혁신하여 기존 GAN 기반 접근법 대비 우수한 시간적 일관성과 시각적 품질을 제공하지만, 모션 일관성, 계산 효율성, 윤리적 고려에서 상당한 도전이 남아 있다.

특히 Peebles & Xie(2023)가 Transformer를 확산 모델에 도입한 이후, 이러한 발전이 최근 고성능 비디오 생성 모델 Sora 등으로 이어지고 있다. StyleCrafter의 U-Net 기반 설계를 DiT 기반으로 확장하는 것이 향후 핵심 과제가 될 것이다.

---

## 참고자료

1. **Liu, G. et al.** "StyleCrafter: Enhancing Stylized Text-to-Video Generation with Style Adapter." *arXiv:2312.00330* (2023) / *ACM Transactions on Graphics (TOG)* 2024 — [arxiv.org/abs/2312.00330](https://arxiv.org/abs/2312.00330)
2. **StyleCrafter Project Page** — [gongyeliu.github.io/StyleCrafter.github.io](https://gongyeliu.github.io/StyleCrafter.github.io/)
3. **StyleCrafter GitHub Repository** — [github.com/GongyeLiu/StyleCrafter](https://github.com/GongyeLiu/StyleCrafter)
4. **StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements** — [arxiv.org/abs/2412.08503](https://arxiv.org/html/2412.08503v1)
5. **ArtCrafter: Text-Image Aligning Style Transfer via Embedding Reframing** — [arxiv.org/abs/2501.02064](https://arxiv.org/html/2501.02064v1)
6. **Lilian Weng, "Diffusion Models for Video Generation"** — [lilianweng.github.io](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
7. **Wang et al.** "Survey of Video Diffusion Models: Foundations, Implementations, and Applications." *TMLR* 2025 — [arxiv.org/abs/2504.16081](https://arxiv.org/abs/2504.16081)
8. **aimodels.fyi** StyleCrafter 분석 페이지 — [aimodels.fyi](https://www.aimodels.fyi/papers/arxiv/stylecrafter-enhancing-stylized-text-to-video-generation)
9. **OpenReview** StyleCrafter — [openreview.net](https://openreview.net/forum?id=xJdBZJcquK)
10. **Hugging Face Papers** — [huggingface.co/papers/2312.00330](https://huggingface.co/papers/2312.00330)

> **참고**: 위의 수식은 논문의 구조적 설계 설명을 기반으로 재구성한 것입니다. 논문 원문의 정확한 수식 표기와 약간의 차이가 있을 수 있으며, 보다 정확한 수식은 원본 PDF를 직접 참조하시기 바랍니다.
