
# Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction

> **논문 정보**
> - **제목:** Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction
> - **저자:** Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang (Peking University / ByteDance)
> - **발표:** NeurIPS 2024 (Oral, **Best Paper Award** 🏆)
> - **arXiv:** [2404.02905](https://arxiv.org/abs/2404.02905)

---

## 1. 핵심 주장 및 주요 기여 요약

VAR은 이미지에 대한 오토회귀(Autoregressive) 학습을 기존 래스터 스캔 방식의 "next-token prediction"에서 벗어나, 거친 해상도에서 정밀한 해상도로 진행하는 **"next-scale prediction"** (또는 "next-resolution prediction")으로 재정의하는 새로운 생성 패러다임입니다.

이 직관적인 방법론은 AR 트랜스포머가 시각적 분포를 빠르게 학습하고 일반화를 잘 하도록 하며, VAR은 최초로 GPT-style AR 모델이 이미지 생성에서 확산 트랜스포머(Diffusion Transformer)를 능가하도록 만들었습니다.

VAR은 **NeurIPS 2024 Best Paper Award**를 수상하였습니다.

### 주요 기여 4가지


1. **새로운 시각적 생성 프레임워크:** 멀티스케일 오토회귀 패러다임과 next-scale prediction을 결합하여 컴퓨터 비전의 오토회귀 알고리즘 설계에 새로운 통찰을 제공합니다.
2. **스케일링 법칙 및 제로샷 일반화 검증:** VAR 모델의 스케일링 법칙과 제로샷 일반화 가능성을 실험적으로 검증하여, LLM의 특성을 시각 도메인에서 재현하였습니다.
3. **성능 돌파구:** GPT-style 오토회귀 방법이 사상 최초로 강력한 확산 모델을 이미지 합성에서 능가하게 하였습니다.
4. **오픈소스 공개:** VQ 토크나이저와 오토회귀 모델 훈련 파이프라인을 포함한 포괄적인 코드를 공개하였습니다.


---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존 이미지 오토회귀 모델(VQGAN, DALL-E 등)은 이미지를 2D 토큰 그리드로 이산화한 뒤 1D 시퀀스로 펼쳐 AR 학습을 수행합니다.

이 접근법은 두 가지 핵심 문제를 유발합니다. 첫째, **수학적 전제 위반(Mathematical Premise Violation)**: VQVAE의 인코더는 상호 의존적인 feature vector $f^{(i,j)}$를 가진 이미지 feature map을 생성하는데, 양자화(quantization)와 평탄화(flattening) 이후에도 토큰 시퀀스 $(x_1, x_2, \ldots, x_{h \times w})$는 양방향 상관관계를 유지합니다. 이는 각 토큰 $x_t$가 오직 이전 접두사 $(x_1, x_2, \ldots, x_{t-1})$에만 의존해야 한다는 오토회귀 모델의 단방향 의존성 가정과 모순됩니다.

둘째, **제로샷 일반화 불가**: 이미지 오토회귀 모델링의 단방향 특성은 양방향 추론이 필요한 작업에서의 일반화 가능성을 제한합니다.

또한 기존 AR 모델들의 스케일링 법칙은 충분히 탐구되지 않았으며, 성능이 확산 모델에 크게 뒤처졌습니다. LLM의 놀라운 성과에 비해 컴퓨터 비전에서 AR 모델의 잠재력은 잠겨 있는 상태였습니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① 기존 Next-Token Prediction (표준 AR)

이산 토큰 시퀀스 $x = (x_1, x_2, \ldots, x_T)$에서 각 토큰 $x_t \in [V]$의 확률은:

$$p(x) = \prod_{t=1}^{T} p_\theta(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

이를 학습하는 것이 "next-token prediction"이며, 이미지에서는 2D 구조를 1D로 flatten하는 과정이 필수적입니다.

#### ② VAR: Next-Scale Prediction

VAR의 접근법은 이미지를 멀티스케일 토큰 맵으로 인코딩하는 것으로 시작합니다. 오토회귀 프로세스는 $1 \times 1$ 토큰 맵에서 시작하여 해상도를 점진적으로 확장하며: 각 단계에서 트랜스포머는 이전 모든 맵을 조건으로 다음 고해상도 토큰 맵을 예측합니다.

이미지를 $K$개의 스케일 토큰 맵 $r_k \in [V]^{h_k \times w_k}$ $(k=1,\ldots,K)$로 양자화한다고 할 때, VAR의 결합 확률은:

$$p(r_1, r_2, \ldots, r_K) = \prod_{k=1}^{K} p_\theta(r_k \mid r_1, r_2, \ldots, r_{k-1})$$

각 스케일 $k$에서의 토큰 맵 확률은:

```math
p_\theta(r_k \mid r_{ < k}) = \prod_{(i,j) \in r_k} p_\theta\!\left(r_k^{(i,j)} \mid r_1, r_2, \ldots, r_{k-1}\right)
```

여기서 **같은 스케일 내의 토큰들은 병렬로 생성**되어 추론 속도를 대폭 향상시킵니다.

이는 각 스케일이 오직 이전 스케일에만 의존하도록 보장함으로써, 인간의 시각적 지각 및 예술적 드로잉의 자연스러운 과정과 일치하며 수학적 일관성을 유지합니다.

#### ③ 학습 목적 함수

VAR의 트랜스포머 학습은 **교차 엔트로피(Cross-Entropy) 손실 최소화**로 이루어집니다:

$$\mathcal{L}_\text{VAR} = -\sum_{k=1}^{K} \sum_{(i,j)} \log p_\theta\!\left(r_k^{(i,j)} \mid r_1, \ldots, r_{k-1}\right)$$

1단계에서는 멀티스케일 양자화 오토인코더(VQVAE)가 이미지를 토큰 맵으로 인코딩하며 복합 재구성 손실로 학습됩니다. 2단계에서는 VAR 트랜스포머가 next-scale prediction 방식으로 교차 엔트로피 손실을 최소화하거나 우도를 최대화하여 학습됩니다.

---

### 2-3. 모델 구조

#### 멀티스케일 VQVAE (1단계)

VAR은 VQGAN과 유사한 아키텍처를 가지되 멀티스케일 양자화를 위해 수정된 멀티스케일 양자화 오토인코더를 사용하여, 각 스케일별로 이미지를 이산 토큰 맵으로 인코딩합니다. 이 과정은 스케일 간 일관성을 보장하기 위해 공유 코드북(shared codebook)을 사용하며, 업스케일링 시 정보 손실을 관리하기 위해 추가 컨볼루션 레이어를 적용합니다.

VAR 프레임워크는 멀티스케일 양자화 계층이 수정된 VQVAE 아키텍처를 사용하며, $K$개의 추가 컨볼루션 계층과 모든 스케일에 걸친 공유 코드북, 잠재 차원 32를 사용합니다.

#### VAR 트랜스포머 (2단계)

VAR 트랜스포머는 GPT-2와 유사한 decoder-only 트랜스포머 아키텍처를 기반으로 하되, 시각 도메인의 적응성을 위해 **Adaptive Layer Normalization(AdaLN)**으로 수정되어 있으며, 각 스케일 내에서 병렬 토큰 생성을 가능하게 합니다.

VAR은 시각적 오토회귀 학습을 위해 GPT-2 계열의 트랜스포머 아키텍처를 직접 활용합니다.

#### 계산 복잡도 비교

기존 AR 모델은 이미지 토큰 수가 해상도의 제곱에 비례하여 총 $\mathcal{O}(n^6)$의 계산량이 필요합니다. 반면 VAR은 $\mathcal{O}(n^4)$의 계산만 필요합니다.

VAR은 각 해상도 내에서 병렬 토큰 생성과 재귀적 스케일 확장을 사용하여 계산 복잡도를 $\mathcal{O}(n^4)$으로 줄입니다.

정리하면:
| 모델 유형 | 시간 복잡도 |
|---|---|
| 기존 Next-Token AR | $\mathcal{O}(n^6)$ |
| VAR (Next-Scale) | $\mathcal{O}(n^4)$ |

---

### 2-4. 성능 향상

ImageNet 256×256 벤치마크에서 VAR은 AR 기준선의 FID를 18.65에서 1.73으로, IS를 80.4에서 350.2로 대폭 향상시키며 약 20배 빠른 추론 속도를 달성하였습니다. 또한 이미지 품질, 추론 속도, 데이터 효율성, 확장성 등 다양한 차원에서 DiT를 능가함이 경험적으로 검증되었습니다.

VAR은 데이터 효율성이 높아 더 적은 훈련 에포크를 필요로 하며, 모델 파라미터가 증가함에 따라 FID와 IS 지표가 일관되게 향상되는 더 나은 확장성을 보입니다.

2B 파라미터까지 VAR을 확장하면서, 테스트 성능과 모델 파라미터 또는 훈련 컴퓨팅 사이의 명확한 거듭제곱 법칙 관계가 관찰되었으며, 피어슨 계수(Pearson coefficient)가 $-0.998$에 근접하여 성능 예측을 위한 강건한 프레임워크임을 나타냅니다.

2B 파라미터의 VAR은 FID 1.80을 달성하여, 3B 또는 7B 파라미터를 가진 L-DiT를 능가합니다.

성능 비교표:

| 모델 | FID↓ | IS↑ | 추론속도 |
|---|---|---|---|
| VQGAN (AR baseline) | 18.65 | 80.4 | 기준 |
| DiT-XL/2 (Diffusion) | ~2.27 | ~278 | 느림 |
| **VAR (2B params)** | **1.73** | **350.2** | **~20× 빠름** |

---

### 2-5. 한계점

VAR은 작동을 위해 멀티스케일 VQVAE가 필요합니다. 즉, 추가적인 1단계 토크나이저 훈련이 필수적이며 이는 구현 복잡도를 높입니다.

- **도메인 제약:** 논문에서 주요 실험은 ImageNet class-conditional 생성에 집중되어 있어, 텍스트-이미지 생성 등 다른 도메인에서의 성능은 후속 연구(Infinity 등)에서 별도로 검증되었습니다.
- **코드북 설계:** 공유 코드북 방식이 스케일 간 표현력을 제한할 수 있습니다.
- **고해상도 확장성:** 매우 고해상도(예: 2048×2048 이상)에서의 직접적 확장에 대한 검증은 제한적입니다.

---

## 3. 모델 일반화 성능 향상 가능성

VAR의 일반화 성능은 특히 주목할 만한 특성으로, LLM에서 관찰되는 두 가지 핵심 특성을 시각 도메인에서 재현합니다.

### 3-1. 제로샷 일반화 (Zero-Shot Generalization)

VAR은 이미지 인페인팅(in-painting), 아웃페인팅(out-painting), 편집 등 다운스트림 태스크에서 제로샷 일반화 능력을 선보였습니다. 이 결과들은 VAR이 LLM의 두 가지 중요한 특성인 스케일링 법칙과 제로샷 태스크 일반화를 초기적으로 모방하고 있음을 시사합니다.

VAR은 클래스 레이블 정보 없이 지정된 마스크 내의 토큰을 성공적으로 생성하는 이미지 인페인팅 및 아웃페인팅 태스크에서 테스트되었습니다. 이는 아키텍처 수정이나 파라미터 조정 없이 다운스트림 태스크에서도 잘 수행하는 VAR의 일반화 능력을 보여줍니다.

### 3-2. 수학적 일관성이 일반화를 가능하게 하는 이유

각 스케일이 이전 스케일에만 의존하도록 보장함으로써, VAR은 인간의 시각적 지각 및 예술적 드로잉의 자연스러운 과정과 일치하는 수학적 일관성을 유지합니다. VAR은 토큰 맵을 flatten하지 않음으로써 공간적 지역성(spatial locality)의 훼손 문제를 피하고, 멀티스케일 접근법으로 각 스케일 내 완전한 상관관계를 보장합니다.

VAR은 전반적인 이미지 구조를 학습하므로, 인페인팅, 아웃페인팅 같은 제로샷 태스크를 포함한 다양한 입력 조건을 잘 처리합니다. 2D 이미지 구조를 토큰 맵으로 유지함으로써 공간적 지역성과 구조를 보존하며, 멀티스케일 설정이 이러한 공간적 관계를 잘 학습하는 데 도움을 줍니다.

### 3-3. 스케일링 법칙과 일반화의 연관성

거듭제곱 스케일링 법칙은 오토회귀 모델의 크기 증가가 테스트 손실의 예측 가능한 감소로 이어짐을 보여주며, 이는 모델 파라미터 수, 훈련 토큰, 최적 훈련 컴퓨팅과 거듭제곱 관계를 따릅니다. 이 예측 가능성은 LLM의 확장성을 확인하는 것뿐만 아니라 더 큰 모델의 성능을 예측하는 도구로서, 자원 배분 최적화에도 기여합니다.

이러한 스케일링 법칙과 제로샷 태스크 일반화 가능성은 LLM의 특징으로서, VAR 트랜스포머 모델에서 초기적으로 검증되었습니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4-1. 연구에 미치는 영향

#### (1) AR 모델 패러다임의 재정립
VAR은 1) 표준 이미지 AR 모델에 내재된 일부 문제를 이론적으로 해결하고, 2) LLM 기반 AR 모델이 이미지 품질, 다양성, 데이터 효율성, 추론 속도 면에서 처음으로 강력한 확산 모델을 능가하도록 하는 새로운 시각적 생성 프레임워크를 소개하였습니다.

#### (2) 멀티모달 확장 가능성
VAR의 공식 GitHub에 따르면 VAR을 기반으로 텍스트-이미지 생성 모델 Infinity가 CVPR 2025 Oral로, 텍스트-비디오 생성 모델 InfinityStar가 NeurIPS 2025 Oral로 채택되었습니다. 이는 VAR 패러다임이 이미지 생성을 넘어 비디오와 멀티모달 방향으로 확장되고 있음을 보여줍니다.

#### (3) 후속 연구 파생
VAR을 활용한 후속 연구로는 VARGPT (시각적 오토회귀 멀티모달 대형 언어 모델), VARGPT-v1.1 (반복적 instruction tuning과 강화학습을 통한 개선), 그리고 [ICML 2025] Direct Discriminative Optimization 등이 등장하였습니다.

#### (4) 시각 생성에서의 스케일링 법칙 확립
VAR은 LLM에서 관찰되는 것과 유사한 거듭제곱 스케일링 법칙을 보여주고, 거의 완벽한 선형 상관계수를 가지며, 이미지 품질, 추론 속도, 데이터 효율성, 확장성 면에서 Diffusion Transformer를 능가합니다. 이는 시각 생성 분야에서 모델 크기와 성능의 관계를 체계적으로 연구할 수 있는 기반을 마련합니다.

---

### 4-2. 향후 연구 시 고려사항

| 항목 | 고려 내용 |
|---|---|
| **텍스트 조건부 생성** | ImageNet 조건부에서 나아가 텍스트-이미지(T2I) 생성으로 확장 필요 (Infinity 등에서 초기 검증됨) |
| **비디오 생성** | 시간 축을 추가한 next-scale prediction 설계 필요 |
| **코드북 최적화** | 공유 코드북의 한계를 극복하기 위한 동적 코드북 또는 계층적 코드북 탐색 |
| **고해상도 확장** | 512×512 이상 해상도에서의 품질 유지 및 복잡도 관리 |
| **멀티모달 통합** | 이해(understanding)와 생성(generation)의 통합 (VARGPT 계열 연구) |
| **윤리적 고려** | 강력한 이미지 생성 기술의 딥페이크 및 허위 정보 악용 방지 설계 |

향후 연구에서는 VAR과 텍스트 프롬프트 생성 태스크의 통합 및 비디오 생성으로의 확장이 예상되며, 이는 VAR의 확장성과 효율성을 활용하는 방향입니다.

저자들은 자신들의 연구 결과와 오픈소스가 자연어 처리 분야의 실질적인 성과를 컴퓨터 비전으로 보다 원활하게 통합하는 데 기여하여, 궁극적으로 강력한 멀티모달 AI 발전에 기여하기를 희망합니다.

---

## 5. 2020년 이후 관련 연구 비교 분석

VQGAN, DALL-E 및 그 후속 모델들은 AR 모델의 이미지 생성 잠재력을 보여주었으며, 이 모델들은 시각적 토크나이저를 사용하여 연속적인 이미지를 2D 토큰 그리드로 이산화한 뒤 AR 학습을 위해 1D 시퀀스로 flatten하는 방식입니다.

| 모델 | 연도 | 방법 | 한계 |
|---|---|---|---|
| **VQVAE-2** | 2019 | 다단계 VQVAE + 래스터스캔 AR | 품질 낮음 |
| **VQGAN** | 2021 | Adversarial Loss + AR Transformer | 1D flatten, 구조 손실 |
| **DALL-E** | 2021 | Large-scale AR (dVAE) | 해상도 제한 |
| **MaskGIT** | 2022 | BERT-style 마스크 예측 (병렬) | 단일 해상도 |
| **DiT** | 2022 | Diffusion + Transformer | 느린 추론 |
| **Parti** | 2022 | ViT-VQGAN + 20B AR | 확장성 비용 높음 |
| **MagViT-2** | 2023 | 영상용 개선 VQVAE | AR 성능 한계 |
| **VAR** | **2024** | **Next-Scale Prediction** | **멀티스케일 VQVAE 필요** |

MaskGIT은 BERT와 유사한 마스크 예측 트랜스포머와 VQ 오토인코더를 사용하여 greedy 알고리즘으로 VQ 토큰을 생성합니다. MagViT는 이 접근법을 비디오로 확장하고, MUSE는 MaskGIT을 3B 파라미터로 확장합니다.

확산 모델의 발전은 향상된 학습 또는 샘플링, 가이던스, 잠재 학습, 아키텍처를 중심으로 이루어졌으며, DiT와 U-ViT는 U-Net을 트랜스포머로 교체하여 Stable Diffusion 3.0, SORA, Vidu 등을 포함한 최근 이미지/비디오 합성 시스템에 영감을 주었습니다.

VAR의 핵심적 차별점은 **수학적 올바름 + 병렬 처리 + 멀티스케일 구조 보존**의 조합으로, 이전 AR 접근법들의 세 가지 근본 문제(수학적 전제 위반, 공간 구조 손실, 일반화 한계)를 동시에 해결한다는 점입니다.

---

## 📚 참고 자료 / 출처

1. **arXiv 논문 원문:** Tian, K. et al. (2024). *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction.* arXiv:2404.02905. https://arxiv.org/abs/2404.02905
2. **NeurIPS 2024 공식 페이지 (Oral):** https://neurips.cc/virtual/2024/oral/97960
3. **NeurIPS 2024 Proceedings:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/9a24e284b187f662681440ba15c416fb-Abstract-Conference.html
4. **OpenReview (NeurIPS 2024):** https://openreview.net/forum?id=gojL67CfS8
5. **GitHub 공식 구현체 (FoundationVision/VAR):** https://github.com/FoundationVision/VAR
6. **Medium 논문 리뷰 (Lukyanenko, A.):** https://artgor.medium.com/paper-review-visual-autoregressive-modeling-scalable-image-generation-via-next-scale-prediction-059c759139aa
7. **Unite.AI 분석 기사:** https://www.unite.ai/visual-autoregressive-modeling-scalable-image-generation-via-next-scale-prediction/
8. **Yifan Zhou's Blog (In-Depth Analysis):** https://zhouyifan.net/blog-en/2024/12/21/20241218-VAR/
9. **EmergentMind 논문 분석:** https://www.emergentmind.com/papers/2404.02905
10. **후속 연구 - Infinity (CVPR 2025 Oral):** arXiv:2412.04431
