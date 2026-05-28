
# LongCat-Video Technical Report 

> **참고 자료 (출처)**
> 1. Meituan LongCat Team, *"LongCat-Video Technical Report"*, arXiv:2510.22200 (2025.10.25) — https://arxiv.org/abs/2510.22200
> 2. arXiv PDF — https://arxiv.org/pdf/2510.22200
> 3. Hugging Face Paper Page — https://huggingface.co/papers/2510.22200
> 4. GitHub (meituan-longcat/LongCat-Video) — https://github.com/meituan-longcat/LongCat-Video
> 5. AlphaXiv Overview — https://www.alphaxiv.org/overview/2510.22200v2
> 6. HyperAI Papers — https://hyper.ai/en/papers/2510.22200
> 7. DeepWiki Evaluation & Benchmarks — https://deepwiki.com/meituan-longcat/LongCat-Video/11-evaluation-and-benchmarks
> 8. LongCat AI Official Site — https://www.longcatai.net / https://www.longcatai.org
> 9. Efficient Coder Blog — https://www.xugj520.cn/en/archives/longcat-video-longform-generation.html

---

## 1. 핵심 주장 및 주요 기여 요약

LongCat-Video는 13.6B 파라미터 규모의 기초(foundational) 비디오 생성 모델로, 다양한 비디오 생성 태스크에서 강력한 성능을 제공하며, 특히 효율적이고 고품질인 장시간(long) 비디오 생성에 특화되어 있으며, 이를 월드 모델(world model)을 향한 첫 번째 단계로 제시합니다.

주요 기여를 표로 정리하면 다음과 같습니다.

| 기여 항목 | 내용 |
|---|---|
| 통합 아키텍처 | DiT 기반 단일 모델로 T2V, I2V, Video-Continuation 지원 |
| 장시간 비디오 생성 | Video-Continuation 프리트레이닝으로 수 분 분량의 영상 생성 |
| 효율적 추론 | Coarse-to-Fine 전략 + Block Sparse Attention |
| 정렬 학습 | Multi-Reward RLHF (GRPO 기반) |
| 오픈소스 공개 | 코드 및 모델 가중치 MIT 라이선스로 공개 |

LongCat-Video의 핵심 기술 기여는 ① 단일 프레임워크에서 T2V, I2V, Video-Continuation을 모두 지원하는 통합 아키텍처, ② Block Sparse Attention·Coarse-to-Fine·고급 캐싱 메커니즘의 결합으로 실용적인 장형 비디오 생성 구현, ③ 멀티 리워드 GRPO의 비디오 생성 적용을 선도한 점, ④ 코드·모델 가중치·Block Sparse Attention 등의 핵심 구성 요소 공개로 커뮤니티 연구를 촉진한 점입니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 접근법들은 오류 누적과 Dense Attention의 높은 연산 비용으로 인해 장시간 비디오에서의 시간적 일관성(temporal coherence) 유지에 어려움을 겪었으며, 대부분의 모델은 태스크 특화(task-specific)이거나 장시간 비디오 생성을 위해 사후(post-hoc) 파인튜닝이 필요했습니다.

구체적으로는 다음 3가지 문제를 해결합니다:
1. **다중 태스크 분리 문제**: T2V, I2V, Video-Continuation을 별도 모델로 학습
2. **장시간 영상의 품질 저하**: 색상 드리프트(color drifting), 시각적 일관성 붕괴
3. **연산 비효율**: Attention 복잡도가 토큰 수에 따라 $O(n^2)$으로 증가

---

### 2.2 제안하는 방법 및 수식

#### (A) Diffusion Transformer (DiT) 기반 확산 과정

LongCat-Video의 기반은 **확산 모델(Diffusion Model)**입니다. 표준 확산 과정은 다음과 같이 정의됩니다.

**순방향 과정 (Forward Process):**
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t \mathbf{I})$$

**역방향 과정 (Reverse Process, 모델이 학습하는 부분):**
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1};\; \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t))$$

**학습 목적함수 (Score Matching / Noise Prediction):**

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t}\left[\left\|\epsilon - \epsilon_\theta(x_t, t, c)\right\|^2\right]$$

여기서 $c$는 텍스트 또는 이미지 조건(conditioning), $\epsilon$는 가우시안 노이즈, $\epsilon_\theta$는 DiT 기반 노이즈 예측 네트워크입니다.

#### (B) Block Sparse Attention (BSA)

비디오 생성의 연산 비용은 해상도와 프레임 수가 높아질수록 Attention 복잡도가 토큰 수에 대해 이차적으로(quadratically) 증가하는 문제가 있으며, LongCat-Video는 Block Sparse Attention 메커니즘을 구현하여 Attention 연산량을 표준 Dense Attention의 10% 미만으로 줄였습니다.

Dense Attention의 복잡도:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V, \quad \text{복잡도: } O(n^2 d)$$

Block Sparse Attention에서는 전체 Attention 행렬을 블록 단위로 나누어 중요한 블록만 선택적으로 계산합니다:
$$\text{BSA}(Q, K, V) = \text{softmax}\!\left(\frac{Q_b K_b^\top}{\sqrt{d_k}} \odot \mathbf{M}_b\right)V_b$$

여기서 $\mathbf{M}_b \in \{0,1\}^{B \times B}$는 블록 단위 마스크 행렬이며, 약 **10% 이하의 블록만 활성화**합니다.

#### (C) Coarse-to-Fine (C2F) 생성 전략

C2F 전략은 먼저 480p/15fps의 거친(rough) 영상을 생성한 후, Expert LoRA 레이어를 사용하여 720p/30fps로 정제(refine)하며, 이를 통해 **약 10배의 추론 속도 향상**을 달성합니다. Block Sparse Attention(BSA)은 관련 Attention 가중치의 10%만 연산하면서도 품질을 거의 유지합니다.

시간·공간 축 모두에 적용되는 C2F를 수식으로 표현하면:

$$\hat{V}^{\text{fine}} = \text{LoRA Refine}\!\left(f_\theta^{\text{coarse}}(z_T, c)\right)$$

여기서 $f_\theta^{\text{coarse}}$는 저해상도 디노이징 함수, $\text{LoRA Refine}$은 Expert LoRA 모듈입니다.

고해상도 생성을 위해 Expert LoRA 모듈을 훈련하여 베이스 모델의 지식을 효과적으로 활용합니다.

#### (D) Multi-Reward RLHF (GRPO)

Multi-reward GRPO(Group Relative Policy Optimization) 기반의 RLHF는 다양한 품질 지표에 맞춘 정렬(alignment)을 강화하며, 주요 상업용 및 오픈소스 모델에 준하는 성능을 달성합니다.

GRPO의 목적 함수는 다음과 같이 일반화됩니다:

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}_{x \sim \pi_\theta}\left[\sum_{k=1}^{K} \lambda_k r_k(x) - \beta \cdot D_{\text{KL}}\!\left(\pi_\theta \| \pi_{\text{ref}}\right)\right]$$

여기서:
- $r_k(x)$: $k$번째 리워드 모델의 점수 (시각 품질, 동작 품질, 텍스트 정렬 등)
- $\lambda_k$: 각 리워드의 가중치
- $\beta$: KL 패널티 계수 (reward hacking 방지)
- $\pi_{\text{ref}}$: 참조 정책(SFT 모델)

---

### 2.3 모델 구조

LongCat-Video는 13.6B 파라미터의 Diffusion Transformer 모델로, 텍스트-투-비디오, 이미지-투-비디오, 비디오-컨티뉴에이션 태스크를 통합하여 Coarse-to-Fine 샘플링과 Block Sparse Attention을 통해 수 분 분량의 고품질 비디오를 효율적으로 생성합니다.

모델의 학습 파이프라인은 크게 4단계로 구성됩니다:

| 단계 | 내용 |
|---|---|
| **Base Model Training** | DiT 기반, Video-Continuation 프리트레이닝 포함 |
| **RLHF Training** | Multi-Reward GRPO 적용 |
| **Acceleration Training** | 모델 증류(Distillation) + Expert LoRA 훈련 |
| **Data Pipeline** | 전처리 → 어노테이션 2단계 데이터 큐레이션 |

데이터셋은 다양한 원본 비디오 소스로부터 구축되며, 데이터 전처리(Preprocessing)와 데이터 어노테이션(Annotation)의 2단계 파이프라인을 통해 큐레이션됩니다.

LongCat-Video는 Wan 2.2 대비 총 파라미터 수가 48.5% 적으면서도 유사하거나 더 나은 성능을 달성하며, Dense 아키텍처이므로 MoE 시스템에 비해 배포와 최적화가 더 단순하고, 단일 Dense 모델이 T2V, I2V, Video-Continuation 태스크 전체를 처리하여 아키텍처 다용도성을 증명합니다.

---

### 2.4 성능 향상

LongCat-Video는 시각적 품질(Visual Quality) 평가에서 3.27점으로 최고 점수를 달성하여 미적으로 우수한 프레임 생성 능력을 입증했으나, 이미지 정렬(Image-Alignment, 4.04점)과 동작 품질(Motion Quality, 3.59점)에서는 다른 모델들에 비해 낮은 점수를 기록했습니다.

LongCat-Video는 Video-Continuation 태스크에 네이티브로 프리트레이닝되어, 색상 드리프트나 품질 저하 없이 수 분 분량의 비디오를 생성할 수 있습니다.

---

### 2.5 한계

전체적인 품질(Overall Quality) 평가에서 LongCat-Video(3.17점)는 경쟁력 있는 수준이나 다른 모델들보다 낮으며, Seedance 1.0이 3.35점으로 가장 높은 전체 점수를 기록하였습니다. 이는 시각적 충실도(visual fidelity)에서는 뛰어나지만, 시간적 일관성(temporal consistency)과 소스 이미지와의 정렬에는 개선의 여지가 있음을 시사합니다.

비교 대상 모델(Veo3, PixVerse-V5, Seedance 1.0, Hailuo-02)은 아키텍처와 파라미터 수가 공개되지 않아 직접적인 아키텍처 비교에 한계가 있으며, 평가는 T2V와 I2V 태스크에 집중되어 있고 Video-Continuation 성능은 독립적으로 벤치마킹되지 않았습니다. 또한 모든 평가가 720p, 30fps, 93프레임(약 3초) 조건에서 수행되어, 다른 해상도나 더 긴 영상에서의 성능은 달라질 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

일반화 성능과 관련하여 LongCat-Video는 여러 방면에서 가능성을 보입니다.

### 3.1 통합 아키텍처에 의한 일반화

저자들의 주요 기여는 입력 프레임 수에 대한 조건화(conditioning)를 통해 T2V, I2V, Video-Continuation 태스크를 네이티브로 지원하는 통합 아키텍처로, 원활한 태스크 전환이 가능합니다. 특히 Video-Continuation 프리트레이닝을 통해 성능 저하 없이 수 분 분량의 비디오를 생성할 수 있습니다.

이 설계는 **단일 모델로 다양한 도메인과 시나리오에 대응**할 수 있는 일반화 기반을 제공합니다.

### 3.2 Multi-Reward RLHF에 의한 일반화

멀티 리워드 GRPO의 비디오 생성에 대한 선도적 적용은 인간 선호도에 다양하게 정렬하면서도 리워드 해킹(reward hacking)을 방지하는 청사진을 제공합니다.

이는 특정 지표 과적합 없이 다양한 품질 기준에서 균형 잡힌 성능을 이끌어내어 일반화를 향상시킵니다.

### 3.3 Avatar 1.5에서의 도메인 일반화 실증

LongCat-Video-Avatar 1.5는 스타일화된 도메인(애니메이션, 동물, 복잡한 실세계 조건)으로의 일반화를 지원하며, 스텝 증류(step distillation)를 통해 8스텝 추론 가속을 달성합니다.

이는 LongCat-Video의 기반 아키텍처가 **새로운 도메인으로의 전이(transfer)** 가능성이 높음을 실증적으로 보여줍니다.

### 3.4 일반화를 위한 수식적 관점

Multi-Reward RLHF에서 여러 리워드 신호의 균형은 다음과 같은 Pareto-optimal 방식으로 이해할 수 있습니다:

$$\pi^* = \arg\max_{\pi_\theta} \sum_{k=1}^{K} \lambda_k \mathbb{E}_{x \sim \pi_\theta}[r_k(x)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

$\lambda_k$의 적절한 조정을 통해 특정 도메인/태스크에 편향되지 않는 범용적 정책(policy)을 학습하는 것이 일반화 성능 향상의 핵심입니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

| 영향 영역 | 구체적 내용 |
|---|---|
| **월드 모델 연구** | 장시간 비디오 생성 → 물리 시뮬레이션, 자율주행, 구현 AI 기반 제공 |
| **효율적 Attention 연구** | BSA의 오픈소스 공개로 후속 희소 어텐션 연구 촉진 |
| **RLHF in Generation** | GRPO의 비디오 생성 적용 청사진 제공 |
| **통합 멀티태스크 모델** | 단일 모델 다중 태스크 패러다임의 비디오 생성 표준화 |
| **오픈소스 생태계** | 코드·가중치 공개로 학계·산업계 접근성 민주화 |

비디오 생성은 자율주행과 구현 AI 등 월드 모델 응용을 위한 강건한 기반을 구축하고 있으며, 물리적 시뮬레이션과 장시간 비디오 생성의 지속적인 발전과 함께 지능형 시스템의 복잡한 실세계 배포와 진화를 가속하고 있습니다.

### 4.2 향후 연구 시 고려할 점

**① 시간적 일관성(Temporal Consistency) 개선**
모델이 시각적 충실도에서는 뛰어나지만, 시간적 일관성 유지와 소스 이미지와의 정렬에 개선의 여지가 있음을 인식하고 이를 집중 연구해야 합니다.

**② 더 긴 영상 및 다양한 해상도 벤치마킹**
Video-Continuation 성능이 독립적으로 벤치마킹되지 않고, 모든 평가가 720p/93프레임 조건에서 수행된 점은 더 긴 영상이나 다른 해상도에서의 성능 파악이 필요함을 시사합니다.

**③ 물리적 정확성(Physical Plausibility)**
Avatar 1.5에서 물리적 합리성(physical rationality)과 시간적 안정성의 중요성이 강조된 만큼, 기반 모델에서도 물리 법칙을 준수하는 영상 생성 메커니즘 연구가 필요합니다.

**④ 공정한 모델 비교 방법론 수립**
비교 대상 독점 모델들(Veo3 등)의 아키텍처와 파라미터 수가 공개되지 않아 직접적인 아키텍처 비교에 한계가 있으므로, 표준화된 공개 벤치마크와 평가 프레임워크 구축이 시급합니다.

**⑤ 데이터 다양성 및 도메인 커버리지**
일반화 성능 향상을 위해서는 훈련 데이터의 도메인 다양성 확보가 필수적이며, 특히 의료, 과학, 산업 등 특수 도메인으로의 전이 학습 가능성을 탐구해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

주요 관련 연구로는 상업용 모델인 Veo(Google, 2024), Sora(OpenAI, 2024), Seedance(2025), Kling(Kuaishou, 2024), Hailuo(MiniMax, 2024), PixVerse(2024)가 있으며, 오픈소스 모델로는 Wan(2025), HunyuanVideo(Kong et al., 2024), Step-Video(Ma et al., 2025), CogVideoX(Yang et al., 2024) 등이 뛰어난 성능을 보이고 있습니다.

| 모델 | 연도 | 특징 | LongCat-Video와의 비교 |
|---|---|---|---|
| **Sora** (OpenAI) | 2024 | 최초의 대규모 T2V 모델, 긴 영상 생성 | 비공개 아키텍처, 추론 효율 미공개 |
| **HunyuanVideo** | 2024 | 오픈소스 고품질 T2V | 단일 태스크 중심, 장시간 영상 미특화 |
| **CogVideoX** | 2024 | DiT 기반, 오픈소스 | 단일 태스크 중심 |
| **Wan 2.2** | 2025 | MoE 구조, 28B 파라미터 | LongCat-Video가 48.5% 적은 파라미터로 유사 성능 |
| **Seedance 1.0** | 2025 | 전체 품질 1위(3.35점) | 동작 품질·전체 품질에서 LongCat-Video 상회 |
| **LongCat-Video** | 2025 | 통합 아키텍처, BSA, GRPO | 시각 품질 1위, 장시간 생성 특화, 오픈소스 |

LongCat-Video의 차별점은 **단일 모델로 다중 태스크를 네이티브 지원**하고, **BSA를 통한 Dense Attention 대비 10배 연산 절감**, 그리고 **Video-Continuation 프리트레이닝**을 통한 장시간 영상 품질 유지라는 세 축의 혁신에 있습니다.

Multi-reward RLHF 훈련을 통해 LongCat-Video는 최신 클로즈드소스 및 선도적 오픈소스 모델과 동등한 수준의 성능을 달성하며, 코드와 모델 가중치를 공개하여 해당 분야의 발전을 가속합니다.

---

> ⚠️ **정확도 관련 고지**: 본 보고서는 공개된 arXiv 논문(2510.22200) 및 공식 GitHub, Hugging Face 페이지를 기반으로 작성되었습니다. 논문 내부의 세부 수식(예: 정확한 BSA 마스크 설계, GRPO 리워드 세부 구성 등)은 원문을 직접 참조하시기 바랍니다. 제시된 수식 중 일반 Diffusion/Attention/GRPO 수식은 해당 분야의 표준 수식이며, 논문 특유의 변형 부분은 원문 PDF에서 확인을 권장합니다.
