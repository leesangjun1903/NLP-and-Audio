
# VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation

> **논문 정보**
> - **제목:** VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation
> - **arXiv ID:** 2409.04429 (2024)
> - **게재:** ICLR 2025 (accepted)
> - **저자:** Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, Song Han, Yao Lu
> - **소속:** MIT HAN Lab, NVIDIA

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

VILA-U는 Video, Image, Language 이해와 생성을 통합하는 **Unified Foundation Model**입니다. 기존의 시각-언어 모델(VLM)들은 시각 콘텐츠를 이해하고 생성하는 데 **별도의 모듈**을 사용함으로써 정렬 불일치(misalignment)와 복잡성 증가를 초래했습니다. 반면, VILA-U는 두 작업 모두에 **단일 자기회귀(next-token prediction) 프레임워크**를 사용하여 diffusion 모델과 같은 외부 컴포넌트를 제거합니다.

### 1.2 주요 기여 (3가지)

VILA-U 성공의 핵심 요인은 두 가지입니다: **(1) 사전학습 단계에서 이산 시각 토큰(discrete visual tokens)을 텍스트 입력과 정렬하는 통합 비전 타워(unified vision tower)**가 시각 인식을 향상시키고, **(2) 고품질 데이터셋으로 학습된 자기회귀 이미지 생성**이 diffusion 모델에 필적하는 품질을 달성합니다. 이를 통해 VILA-U는 완전히 토큰 기반의 자기회귀 프레임워크로 더 복잡한 모델들과 비슷한 성능을 냅니다.

이 논문은 **ICLR 2025**에 채택되었습니다.

---

## 2. 해결하고자 하는 문제, 제안하는 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

기존의 통합 방식에는 두 가지 주류 접근법이 존재했습니다:
1. **VQGAN 기반 토크나이저** 접근법: 시각 입력을 이산 토큰으로 변환하고 자기회귀 모델을 사용하지만, VQGAN 인코더의 시각 토큰은 **의미 정보(semantic information)가 부족**하여 시각 이해 작업에서 심각한 성능 저하가 발생합니다.
2. **CLIP 기반 코드북 양자화** 접근법: CLIP 특징이 풍부한 의미 정보를 포함하여 이해 작업에서 우수한 성능을 보이지만, **디코딩 능력이 없어** 시각적 출력을 생성하기 위해 diffusion 모델 같은 외부 시각 생성 모델이 필요하고 인프라 복잡성이 증가합니다.

기존의 대규모 기반 모델 학습 파이프라인은 이미 언어 모델링의 next-token prediction에 최적화되어 있는데, diffusion 모델을 지원하는 별도의 스택을 설계하고 유지하는 것은 **상당한 엔지니어링 비용**을 초래합니다.

### 2.2 제안하는 방법 및 핵심 수식

VILA-U는 두 가지 핵심 원칙을 중심으로 설계되었습니다.

**(1)** 기존의 통합 end-to-end 자기회귀 VLM이 경쟁력 있는 시각 이해 성능을 달성하지 못하는 이유는, VQGAN 토큰이 이미지 재구성 손실만으로 학습되어 **텍스트 입력과 정렬되지 않기 때문**입니다. 따라서 VQ 비전 타워 사전 학습 시 텍스트 정렬을 도입하는 것이 핵심입니다. **(2)** 자기회귀 이미지 생성은 **충분한 크기의 고품질 데이터로 학습하면 diffusion 모델과 유사한 품질**을 달성할 수 있습니다.

#### 2.2.1 통합 비전 타워 학습 (손실 함수)

VILA-U는 비전 타워 학습에 **텍스트-이미지 대조 손실(text-image contrastive loss)**과 **VQ 기반 이미지 재구성 손실(VQ-based image reconstruction loss)**을 포함시킵니다. 이미지에서 추출된 특징은 주로 잔여 양자화(residual quantization)를 통해 이산화됩니다. 한 경로에서는 이산 시각 특징이 디코더를 통해 이미지를 재구성하고 재구성 손실을 계산하며, 다른 경로에서는 텍스트 인코더가 제공하는 텍스트 특징과의 이미지-텍스트 대조 손실을 계산합니다. 이 학습 절차를 통해 비전 타워는 이해와 생성 모두에 적합한 이산 특징을 추출하는 법을 학습합니다.

총 손실 함수는 가중합(weighted sum)으로 결합됩니다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda \cdot \mathcal{L}_{\text{contrastive}}$$

여기서 $\mathcal{L}_{\text{recon}}$은 VQ 기반 이미지 재구성 손실, $\mathcal{L}_{\text{contrastive}}$는 텍스트-이미지 대조 손실, $\lambda$는 균형 가중치입니다.

#### 2.2.2 잔여 벡터 양자화 (RVQ: Residual Vector Quantization)

잔여 벡터 양자화(RVQ)는 시각 특징을 이산화하는 핵심 메커니즘입니다. 시각 특징의 표현 능력은 양자화기에서 사용하는 코드 크기에 크게 의존하며, 고수준 및 저수준 특징을 모두 포함하기 위해 벡터 특징 공간에 더 많은 용량이 필요합니다.

RVQ의 수학적 표현:

$$\hat{z} = \sum_{d=1}^{D} e_{q_d}$$

- $D$: 잔여 깊이 (residual depth)
- $e_{q_d}$: $d$번째 코드북에서 선택된 코드워드
- $\hat{z}$: 최종 양자화 표현 (각 단계 코드워드의 합)

각 단계의 잔여:
$$r_d = z - \sum_{k=1}^{d-1} e_{q_k}, \quad q_d = \arg\min_{j} \| r_d - e_j \|_2$$

코드북 크기는 16,384이며, 모든 이미지와 비디오는 $256 \times 256$ 또는 $384 \times 384$ 해상도로 조정되고, 각 이미지 또는 비디오 프레임은 잔여 깊이 $D=4$ (256 해상도) 또는 $D=16$ (384 해상도)의 코드로 변환됩니다.

#### 2.2.3 통합 자기회귀 학습 목적함수

시각 입력은 이산 토큰으로 변환되어 텍스트 토큰과 연결되어 다중 모달 토큰 시퀀스를 형성합니다. 모든 토큰은 next-token prediction 과정에 참여하며 **통합 학습 목적함수**를 구성합니다. 추론 시에는 출력 토큰이 텍스트 디토크나이저 또는 비전 타워 디코더에 의해 디코딩되어 다중 모달 콘텐츠를 생성합니다.

$$\mathcal{L}_{\text{NTP}} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta)$$

여기서 $x_t$는 시각 또는 텍스트 토큰 시퀀스의 $t$번째 토큰이며, $\theta$는 모델 파라미터입니다.

시각 생성 시 **Classifier-Free Guidance (CFG)**가 적용됩니다:

$$\tilde{p}(x_t \mid c) = p(x_t \mid \emptyset)^{1-\alpha} \cdot p(x_t \mid c)^{\alpha}$$

혹은 로짓(logit) 형태로:

$$\tilde{\ell}(x_t \mid c) = \ell(x_t \mid \emptyset) + w \cdot \left(\ell(x_t \mid c) - \ell(x_t \mid \emptyset)\right)$$

여기서 $w$는 CFG 가중치, $c$는 텍스트 조건입니다.

### 2.3 모델 구조

VILA-U는 CLIP처럼 풍부한 의미 정보를 추출하면서도 VQGAN처럼 이미지 재구성 능력을 지원하는 **통합 비전 타워**를 설계합니다. 이는 오토인코더 학습 과정에 재구성 손실과 대조 손실을 모두 통합하고, 시각 특징의 표현 능력을 강화하기 위해 잔여 양자화(residual quantization)를 활용함으로써 달성됩니다. 이 기반 위에 단일 end-to-end 자기회귀 프레임워크를 개발합니다.

**모델 구조 요약:**

```
[입력 이미지/비디오]
        ↓
[비전 인코더 (ViT 계열)]
        ↓
[잔여 벡터 양자화 (RVQ)]
  ├─→ [재구성 경로] → [비전 디코더] → ℒ_recon
  └─→ [텍스트 정렬 경로] → [텍스트 인코더] → ℒ_contrastive
        ↓
[이산 시각 토큰 + 텍스트 토큰 연결]
        ↓
[대형 언어 모델 (LLM) — next-token prediction]
        ↓
[시각 토큰 출력] → [비전 타워 디코더] → 이미지/비디오 생성
[텍스트 토큰 출력] → 텍스트 응답
```

비전 타워는 COYO-700M 데이터셋에서 학습되고 ImageNet에서 zero-shot 분류 및 재구성 성능을 평가합니다.

### 2.4 성능 향상

이 접근법은 모델을 단순화할 뿐만 아니라 시각-언어 이해 및 생성에서 **거의 최첨단(near state-of-the-art) 성능**을 달성합니다.

**이미지 생성:**

MJHQ-30K 벤치마크에서 VILA-U는 CFG 3.0 기준 **FID 12.8**을 달성하며, 이는 LWM의 17.77과 Show-o의 15.18보다 우수한 결과이며, SD-XL(9.55), PixArt(6.14) 등 일부 diffusion 모델에 근접합니다.

VILA-U는 수십억 쌍의 이미지-텍스트 데이터로 학습된 diffusion 기반 방법에는 미치지 못하지만, **훨씬 적은 데이터**로도 고급 프롬프트에서 SD v2.1 및 SD-XL에 필적하는 성능을 보입니다.

**이해 성능:**

VILA-U는 end-to-end 자기회귀 모델과 연속 토큰 VLM 사이의 시각 이해 성능 격차를 **획기적으로 좁히면서** 경쟁력 있는 native 시각 생성 능력을 도입합니다.

### 2.5 한계

통합 비전 타워에서 **재구성 품질과 텍스트 정렬 사이의 트레이드오프**가 생성 성능에 미묘한 영향을 줍니다. 대조 손실의 도입은 시각 이해를 향상시키지만, 재구성 FID를 RQ-VAE의 1.30에서 VILA-U의 1.80으로 약간 저하시키고, 이는 생성 FID를 RQ-VAE 기반 생성의 12.0 대비 13.2로 소폭 악화시킵니다. 이는 **통합 프레임워크 내에서 이해와 생성을 동시에 최적화하는 것의 어려움**을 보여줍니다.

또한 대규모 기반 모델로서 **확장성(scalability)** 문제가 남아 있으며, 표준 벤치마크 성능 평가에 집중되어 있어 다양한 도메인에서의 **강건성(robustness)**에 대한 추가 연구가 필요합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능하게 하는 핵심 메커니즘

VILA-U의 통합 비전 타워는 벡터 양자화를 통해 시각 입력을 이산 토큰으로 변환하고 대조 학습으로 텍스트 입력과 정렬합니다. VILA-U의 다중 모달 학습은 소규모 고품질 이미지-텍스트 코퍼스에서 시각 및 텍스트 토큰 모두에 대한 **통합 next-token prediction 목적함수**를 활용합니다.

이는 모달리티 간 공유 표현 공간(shared representation space)을 형성하여 **zero-shot 및 cross-modal 일반화**를 촉진합니다.

### 3.2 단일 프레임워크에 의한 태스크 간 일반화

이러한 "통합" 또는 "다중 모달" AI 시스템은 단일 목적 모델과 비교하여 **더 유연하고 효율적이며 적응력 있는 잠재력**을 가집니다. 또한 다양한 시각 및 언어 능력 간의 시너지를 활용할 수 있게 합니다.

VILA-U는 통합 학습 프레임워크를 통해 시각 및 텍스트 모달리티 간의 상관관계를 효과적이고 효율적으로 학습하며, 특히 더 나은 텍스트 추종 능력이 필요한 고급 프롬프트에서 훨씬 적은 학습 데이터와 시간으로도 diffusion 기반 방법과 비교적 작은 성능 차이를 유지합니다.

### 3.3 고품질 소규모 데이터의 일반화 효과

자기회귀 이미지 생성은 **충분한 크기의 고품질 데이터**로 학습하면 diffusion 모델과 유사한 품질을 달성할 수 있습니다.

이는 대규모 데이터 의존성 없이도 높은 일반화를 달성할 수 있는 **데이터 효율적 일반화** 가능성을 시사합니다.

### 3.4 이해-생성 간 상호 강화

VILA-U와 관련된 후속 연구는 통합 VLM에서 이해와 생성 작업 간의 **일반화를 체계적으로 조사**하여, 실제 세계 시나리오에 긴밀하게 정렬된 데이터셋을 설계하고 광범위한 실험 및 정량적 평가를 수행합니다.

Liquid 연구는 시각 이해와 생성이 단일 자기회귀 목적함수 및 공유 시각 토큰 표현 하에 통합될 때 **상호 이익을 줄 수 있다**는 새로운 통찰을 제시합니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

통합 멀티모달 모델의 타임라인을 보면, 2023~2025년에 걸쳐 Anole, Show-o, Transfusion, Emu3, VILA-U, Chameleon, Janus, Janus-flow, Janus-Pro, UniTok, Harmon 등 매우 다양한 모델들이 급격히 발전해왔습니다.

| 모델 | 연도 | 접근법 | 특징 | VILA-U와의 차이 |
|------|------|--------|------|-----------------|
| **LWM** | 2024 | VQGAN 기반 | 백만 길이 비디오+언어 | 의미 정렬 없는 재구성 토큰 |
| **Chameleon** | 2024 | VQ-VAE | 혼합 모달 early-fusion | 이해에 집중, 생성 미공개 |
| **Emu3** | 2024 | VQGAN 기반 | next-token prediction | 재구성 기반 토큰화 |
| **Janus** | 2024 | 이중 인코더 분리 | 이해/생성 경로 분리 | VILA-U는 단일 타워로 통합 |
| **Show-o** | 2024 | 혼합 확산+자기회귀 | 이산+연속 토큰 혼합 | VILA-U는 순수 자기회귀 |
| **VILA-U** | 2024 | RVQ + 대조학습 + 자기회귀 | 단일 엔드-투-엔드 통합 | 본 논문 |

LWM은 VQGAN 토크나이저를 사용하여 이미지를 의미적 감독 없이 이산 잠재 코드로 인코딩하며, 시각 및 텍스트 토큰을 함께 직렬화하여 통합 자기회귀 모델링을 제안합니다. LWM은 순수 재구성 기반 시각 토큰과 텍스트 설명만으로도 대규모 다중 모달 생성이 가능함을 보여주지만, **전문화된 의미 토크나이제이션 없이**는 이해 성능이 제한됩니다.

Janus-Pro-7B는 DeepSeek AI가 도입한 통합 다중 모달 모델로, **이해와 생성 양쪽에서 뛰어난** 성능을 발휘합니다. 그러나 Janus는 이해와 생성에 **별도의 시각 인코더**를 사용하는 반면, VILA-U는 단일 통합 비전 타워를 유지합니다.

Harmon은 공유 MAR 인코더로 이해와 생성 작업을 조화시키는 통합 자기회귀 프레임워크로, GenEval, MJHQ30K, WISE 벤치마크에서 최첨단 이미지 생성 결과를 달성하면서 전용 의미 인코더를 가진 방법들과 이미지 이해 벤치마크에서 동등한 성능을 발휘합니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

**① 단일 프레임워크 패러다임의 확립:**
언어 이해와 생성 작업을 하나의 자기회귀 next-token prediction 프레임워크로 통합하는 방법은 diffusion 모델과 같은 추가 컴포넌트를 활용하는 대부분의 VLM보다 간결하고, 자기회귀 방법도 최첨단 VLM과 비교 가능한 성능을 달성할 수 있음을 입증합니다.

**② 비전 토크나이저 설계 방향성 제시:**
UniTok은 새로운 멀티 코드북 양자화 메커니즘을 특징으로 하는 통합 토크나이저를 도입하여 어휘 크기와 병목 차원을 효과적으로 확장하며, **재구성과 의미적 감독이 본질적으로 상충하지 않는다**는 것을 보여줍니다.

**③ 데이터 효율적 학습의 새 기준:**
통합 아키텍처와 학습 프로세스를 활용하여 다양한 응용 프로그램에 효율적으로 파인튜닝 가능한 범용 시각-언어 표현을 학습할 수 있으며, 이는 기존의 단일 목적 모델과 비교하여 더 유연하고 효율적이며 적응력 있는 AI 시스템으로 이어질 수 있습니다.

### 5.2 앞으로의 연구 시 고려할 점

**① 이해-생성 트레이드오프 해결:**
통합 비전 타워에서 재구성 품질과 텍스트 정렬 사이의 트레이드오프가 생성 성능에 영향을 주는 문제가 있습니다. 대조 손실의 도입이 재구성 FID를 1.30에서 1.80으로 저하시키는 현상은, 통합 프레임워크 내에서 이해와 생성 최적화 간의 **섬세한 균형**이 중요한 연구 과제임을 시사합니다.

**② 고해상도 및 장시간 비디오 확장:**
현재 VILA-U는 $256 \times 256$ 및 $384 \times 384$ 해상도에서만 평가되었으며, 더 높은 해상도와 더 복잡한 비디오 시퀀스로의 확장은 **양자화 코드 크기와 잔여 깊이를 증가**시켜야 하는 도전 과제로 남아 있습니다.

**③ 강건성 및 윤리적 고려:**
스케일러빌리티, 강건성, 해석 가능성, 윤리와 같은 중요한 도전 과제들은 이 기술이 계속 발전함에 따라 반드시 다루어져야 합니다.

**④ 이해-생성 상호 일반화 탐구:**
Semantic Drift Protocol(SDP)은 I2T와 T2I를 여러 세대에 걸쳐 교대로 반복하여 **의미적 드리프트를 정량화**하는 순환 평가 프로토콜로, 통합 VLM의 일반화 성능을 측정하는 새로운 지표(MCD, MGG)를 제안합니다. 이러한 순환적 일관성 평가는 앞으로의 통합 모델 연구에서 중요한 지표로 활용될 것입니다.

**⑤ 확장 가능한 데이터 커리큘럼 설계:**
VILA-U는 수십억 쌍의 이미지-텍스트 데이터로 학습된 diffusion 기반 방법보다 성능이 낮지만, 훨씬 적은 데이터와 시간으로 비교 가능한 결과를 달성합니다. 이는 고품질의 규모 있는 데이터로 더 나은 일반화가 가능함을 시사합니다. 따라서 **데이터 품질 기반의 커리큘럼 학습 전략**이 중요한 연구 방향입니다.

---

## 📚 참고문헌 및 출처

| # | 제목 / 출처 | 링크 |
|---|-------------|------|
| 1 | **VILA-U (arXiv 원문)** | https://arxiv.org/abs/2409.04429 |
| 2 | **VILA-U (ICLR 2025 논문)** | https://proceedings.iclr.cc/paper_files/paper/2025/file/e9e140df6de01afb672cb859d203c307-Paper-Conference.pdf |
| 3 | **VILA-U (HTML 전문)** | https://arxiv.org/html/2409.04429v3 |
| 4 | **VILA-U GitHub (MIT HAN Lab)** | https://github.com/mit-han-lab/vila-u |
| 5 | **VILA-U 프로젝트 페이지 (HAN Lab)** | https://hanlab.mit.edu/projects/vila-u |
| 6 | **VILA-U (OpenReview ICLR 2025)** | https://openreview.net/forum?id=02haSpO453 |
| 7 | **VILA-U (ResearchGate PDF)** | https://www.researchgate.net/publication/383864222 |
| 8 | **VILA-U (Semantic Scholar)** | https://www.semanticscholar.org/paper/VILA-U:-a-Unified-Foundation-Model-Integrating-and-Wu-Zhang/c6db7e8f83ae54a3634d4dcb17db03c0c55ec2b5 |
| 9 | **VILA-U (NVIDIA Research)** | https://research.nvidia.com/labs/eai/publication/vila-u/ |
| 10 | **VILA-U Quick Review (Liner)** | https://liner.com/review/vilau-a-unified-foundation-model-integrating-visual-understanding-and-generation |
| 11 | **Janus: Decoupling Visual Encoding (arXiv)** | https://arxiv.org/html/2410.13848v1 |
| 12 | **Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities (arXiv 2025)** | https://arxiv.org/html/2505.02567v3 |
| 13 | **Awesome Unified Multimodal Models (GitHub)** | https://github.com/showlab/Awesome-Unified-Multimodal-Models |
| 14 | **Vision Language Models – Better, Faster, Stronger (HuggingFace Blog, 2025)** | https://huggingface.co/blog/vlms-2025 |
| 15 | **HuggingFace Paper Page (VILA-U)** | https://huggingface.co/papers/2409.04429 |
