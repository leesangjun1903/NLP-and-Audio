
# Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis

> **논문 정보:**
> - **저자:** Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, Xiaobing Liu (ByteDance)
> - **arXiv:** [2412.04431](https://arxiv.org/abs/2412.04431) (2024.12.05 제출)
> - **학회:** CVPR 2025 Oral
> - **GitHub:** [FoundationVision/Infinity](https://github.com/FoundationVision/Infinity)

---

## 1. 핵심 주장 및 주요 기여 요약

Infinity는 언어 지시를 따르며 고해상도의 사실적인 이미지를 생성할 수 있는 **Bitwise Visual AutoRegressive Modeling** 모델로, 무한 어휘 토크나이저 & 분류기(Infinite-Vocabulary Classifier, IVC) 및 비트와이즈 자기교정 메커니즘(Bitwise Self-Correction, BSC)을 갖춘 비트와이즈 토큰 예측 프레임워크 하에 시각적 자기회귀 모델을 재정의하며, 생성 능력과 세부 표현을 획기적으로 향상시킵니다.

### 핵심 기여 요약

| 기여 | 설명 |
|---|---|
| **Bitwise Visual Tokenizer** | 무한 어휘를 지원하는 비트와이즈 멀티스케일 잔차 양자화기 |
| **Infinite-Vocabulary Classifier (IVC)** | 지수적 파라미터 증가 문제를 선형으로 해결 |
| **Bitwise Self-Correction (BSC)** | teacher-forcing의 train-test 불일치 해소 |
| **스케일링** | 토크나이저 + 트랜스포머 동시 스케일링으로 강력한 스케일링 법칙 달성 |

Infinity는 자기회귀 텍스트-이미지 모델의 새로운 기록을 세우며, SD3-Medium과 SDXL 같은 최고급 확산(diffusion) 모델을 능가하며, SD3-Medium 대비 GenEval 벤치마크 점수를 0.62에서 0.73으로, ImageReward 벤치마크 점수를 0.87에서 0.96으로 향상시키고 66%의 win rate를 달성했습니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

자기회귀(AR) 또는 VAR 모델에서 사용되는 **인덱스 기반 이산 토크나이저**는 어휘 크기가 제한될 경우 양자화 오류가 상당하여 고해상도 이미지의 세밀한 세부 재구성에 어려움을 겪습니다. 생성 단계에서는 인덱스 기반 토큰이 흐릿한 지도(supervision)로 인해 시각적 세부 손실 및 지역 왜곡을 유발하며, LLM에서 상속된 **teacher-forcing 학습의 훈련-테스트 불일치**가 시각적 세부의 누적 오류를 증폭시킵니다. 이러한 도전들이 AR 모델의 중요한 병목이 됩니다.

구체적으로는 세 가지 문제:

1. **양자화 오류:** 어휘 크기의 제한으로 인해 이산 VQ-VAE는 항상 연속적 토크나이저보다 뒤처져 AR 기반 T2I 모델의 성능을 저해해왔습니다.
2. **Fuzzy Supervision:** 인덱스 레이블 기반 예측은 연속 특징 공간에서의 미세한 변동에도 레이블이 급격하게 변하는 불안정한 감독 문제를 가집니다.
3. **Train-test 불일치(Teacher Forcing):** VAR은 LLM에서 teacher-forcing 학습을 상속받는데, 이 방식은 시각 생성에 심각한 훈련-테스트 불일치를 초래합니다. 트랜스포머가 각 스케일에서 특징을 정제하기만 하고 실수를 인식·교정하는 능력이 없어, 이전 스케일에서 발생한 실수가 이후 스케일로 전파·증폭되어 최종 이미지를 망가뜨립니다.

---

### 2-2. 제안하는 방법 (수식 포함)

논문은 인덱스 기반 토큰을 비트와이즈 토큰으로 전면 대체하는 **비트와이즈 모델링(bitwise modeling)** 이라는 새로운 접근법을 제안하며, 이 프레임워크는 ①비트와이즈 시각 토크나이저, ②비트와이즈 무한 어휘 분류기(IVC), ③비트와이즈 자기교정(BSC)의 세 가지 핵심 모듈로 구성됩니다.

---

#### ① 비트와이즈 시각 토크나이저 (Bitwise Visual Tokenizer)

어휘 크기를 늘리는 것은 재구성 및 생성 품질 향상에 중요한 잠재력이 있지만, 기존 토크나이저에서 어휘를 직접 확장하면 메모리 소비와 계산 부담이 크게 증가합니다. 이 문제를 해결하기 위해 새로운 **비트와이즈 멀티스케일 잔차 양자화기**를 제안하며, 이는 메모리 사용량을 대폭 감소시켜 $V_d = 2^{32}$ 또는 $V_d = 2^{64}$와 같은 극대 어휘의 학습을 가능하게 합니다.

이미지 특징 맵 $\mathbf{F} \in \mathbb{R}^{h \times w \times d}$에 대해, 각 토큰을 $d$-비트 이진 코드로 표현합니다:

$$\mathbf{z} = \text{BinaryQuantize}(\mathbf{F}) \in \{0, 1\}^d$$

이때 어휘 크기는 $V_d = 2^d$이며, $d$가 증가함에 따라 어휘는 이론적으로 무한히 확장 가능합니다.

멀티스케일 잔차 양자화는 다음과 같이 계층적으로 수행됩니다:

$$\mathbf{R}_k = \text{BinaryQuantize}(\mathbf{F} - \sum_{i=1}^{k-1} \hat{\mathbf{F}}_i), \quad k = 1, 2, \ldots, K$$

여기서 $\mathbf{R}_k$는 $k$번째 스케일의 잔차 비트 표현이며, $\hat{\mathbf{F}}_i$는 $i$번째 스케일까지의 누적 재구성입니다.

---

#### ② 무한 어휘 분류기 (Infinite-Vocabulary Classifier, IVC)

시각 토크나이저는 연속 특징을 양자화하여 인덱스 레이블을 얻습니다. 기존 분류기는 $2^d$개의 인덱스를 예측하지만, IVC는 $d$개의 비트를 예측합니다. 연속 특징에서 0에 가까운 값의 미세한 변화는 인덱스 레이블의 완전한 변화를 초래하는 반면, 비트 레이블(즉, 양자화된 특징)은 미세하게 변하여 안정적인 지도를 제공합니다. 또한 기존 분류기의 파라미터는 $d$가 증가함에 따라 지수적으로 증가하지만, IVC는 선형으로 증가합니다. $d=32$, $h=2048$일 때 기존 분류기는 **8.8조(8.8T) 파라미터**가 필요한 반면, IVC는 단 **0.13M 파라미터**만 필요합니다.

수식으로 표현하면:

- **기존 분류기:** $W \in \mathbb{R}^{h \times 2^d}$ → 파라미터 수 $= h \cdot 2^d$
- **IVC:** $W_{\text{IVC}} \in \mathbb{R}^{h \times d}$ → 파라미터 수 $= h \cdot d$

각 비트 $b_i$에 대한 독립적인 이진 크로스엔트로피 손실:

$$\mathcal{L}_{\text{IVC}} = -\sum_{i=1}^{d} \left[ b_i \log \hat{b}_i + (1 - b_i) \log(1 - \hat{b}_i) \right]$$

실험 결과 IVC는 기존 분류기 ($V_d = 2^{16}$) 대비 파라미터를 **99.95% 절감**하면서도 더 좋은 성능을 달성했습니다.

---

#### ③ 비트와이즈 자기교정 (Bitwise Self-Correction, BSC)

VAR은 LLM으로부터 teacher-forcing 학습을 상속받으며, 이는 시각 생성에 심각한 훈련-테스트 불일치를 초래합니다. 특히 teacher-forcing 학습은 트랜스포머가 각 스케일에서 특징을 정제하기만 할 뿐, 실수를 인식·교정할 능력이 없게 만듭니다. 이전 스케일에서 발생한 실수가 이후 스케일로 전파·증폭되어 최종 이미지를 망가뜨리게 됩니다. 이 문제를 해결하기 위해 BSC를 제안하며, 구체적으로 $k$번째 스케일의 비트 $\mathbf{R}\_k$에 대해 $[0, p]$에서 균일하게 샘플링된 확률로 비트를 랜덤 플립하여 예측 오류를 모방하고, 플립된 값으로 트랜스포머 입력을 재계산하며, 재양자화를 통해 새로운 타깃 $\mathbf{R}_{k+1}$을 얻습니다.

BSC 과정을 수식으로 표현하면:

$$\tilde{\mathbf{R}}_k = \text{BitFlip}(\mathbf{R}_k, p_k), \quad p_k \sim \mathcal{U}(0, p_{\max})$$

$$\mathbf{R}_{k+1}^{\text{target}} = \text{BinaryQuantize}\left(\mathbf{F} - \sum_{i=1}^{k} \text{Decode}(\tilde{\mathbf{R}}_i)\right)$$

BSC는 트랜스포머의 입력과 레이블을 수정함으로써 구현되며, **추가적인 계산 비용을 발생시키지 않고 기존의 병렬 학습 특성을 유지**합니다.

---

#### ④ 전체 학습 목적 함수

Infinity의 전체 학습 목적은 VAR의 next-scale prediction 패러다임을 비트와이즈로 확장한 것으로, 텍스트 조건 $\mathbf{c}$가 주어졌을 때 각 스케일 $k$의 비트를 예측하는 확률을 최대화합니다:

$$\mathcal{L}_{\text{Infinity}} = -\sum_{k=1}^{K} \log p_\theta\left(\mathbf{R}_k \mid \mathbf{R}_1, \mathbf{R}_2, \ldots, \mathbf{R}_{k-1}, \mathbf{c}\right)$$

여기서:
- $K$: 총 스케일 수 (coarse → fine)
- $\mathbf{R}_k$: $k$번째 스케일의 비트와이즈 잔차 토큰
- $\mathbf{c}$: 텍스트 조건 (cross-attention을 통해 주입)
- $p_\theta$: 트랜스포머 기반 생성 모델

$\mathbf{R}\_k$를 예측할 때, $(\mathbf{R}\_1, \mathbf{R}\_2, \ldots, \mathbf{R}_{k-1})$이 접두 컨텍스트(prefixed context)로 작용하며, 텍스트 조건은 cross-attention을 통해 예측을 안내합니다. VAR과 달리 Infinity는 비트 레이블로 next-scale 예측을 수행합니다.

---

### 2-3. 모델 구조

Infinity는 이미지 합성을 위해 **시각 토크나이저(Visual Tokenizer)**와 **트랜스포머(Transformer)**를 통합합니다.

Infinity 아키텍처는 세 가지 핵심 구성 요소로 이루어집니다: ① 이미지 특징을 이진 토큰으로 변환하여 계산 오버헤드를 줄이는 비트와이즈 멀티스케일 양자화 토크나이저, ② 텍스트 프롬프트와 이전 스케일 정보를 조건으로 잔차를 예측하는 트랜스포머 기반 자기회귀 모델, ③ IVC를 통한 비트 레이블 예측.

모델 스케일은 다음과 같이 제공됩니다:
- **Infinity-2B**: 20억 파라미터
- **Infinity-8B**: 80억 파라미터

이 연구에서는 이산 VQ-VAE를 어휘 크기 확장만으로 연속 토크나이저에 필적하는 수준으로 학습시키는 데 성공했습니다.

---

### 2-4. 성능 향상

| 벤치마크 | SD3-Medium | Infinity (최고) |
|---|---|---|
| GenEval | 0.62 | **0.73** |
| ImageReward | 0.87 | **0.96** |
| 생성 속도 (1024×1024) | 2.08s | **0.8s** |
| Win Rate vs SD3 | — | **66%** |

추가 최적화 없이 Infinity는 $1024 \times 1024$ 이미지를 0.8초 만에 생성하며, SD3-Medium보다 **2.6배 빠른** 가장 빠른 텍스트-이미지 모델입니다.

이산 토크나이저가 연속 VAE에 필적하는 성능을 달성하여, ImageNet-256 벤치마크에서 rFID 점수가 0.87에서 0.33으로 향상되었습니다.

---

### 2-5. 한계

논문은 전통적인 자기회귀 모델의 한계로 인덱스 기반 토크나이저에 의한 어휘 크기 제약과 양자화 오류를 지적하며, 특히 고해상도 이미지에서 더욱 문제가 되고, 기존 모델들은 확산 모델이 만들어내는 수준의 디테일을 복제하기 어렵다고 설명합니다.

추가적으로 논문에서 암시하는 주요 한계:
- **도메인 외 일반화 미검증:** 논문은 주로 텍스트-이미지 생성에 집중하며, 비디오나 3D 생성으로의 확장은 후속 연구 과제로 남아 있습니다.
- **훈련 데이터 의존성:** 고품질의 대규모 텍스트-이미지 페어 데이터가 여전히 필요합니다.
- **BSC의 하이퍼파라미터 민감도:** 비트 플립 확률 $p_{\max}$의 설정이 성능에 영향을 미칠 수 있습니다.
- **이산 토큰의 근본적 한계:** Infinity는 이미지 토크나이저로 이산 토크나이저가 근연속(near-continuous) 성능에 근접할 수 있음을 시연했지만, 이론적으로 연속 표현과의 간극이 완전히 제거되지는 않습니다.

---

## 3. 일반화 성능 향상 가능성

### 3-1. 스케일링 법칙을 통한 일반화

자기회귀 언어 모델의 스케일링 법칙은 모델 크기, 데이터셋 크기, 계산량과 테스트 셋 크로스엔트로피 손실 사이의 **거듭제곱 법칙(power-law) 관계**를 밝혀냈으며, 이는 더 큰 모델의 성능 예측을 가능하게 하여 포화 없이 지속적인 개선을 이루는 효율적 자원 배분으로 이어집니다.

토크나이저 어휘 크기를 이론적으로 무한대로 확장하고 트랜스포머 크기를 동시에 확장함으로써, vanilla VAR 대비 강력한 스케일링 능력을 대폭 해방시킵니다.

### 3-2. Bitwise 표현의 일반화 이점

인덱스 기반 토크나이즈를 비트와이즈 토큰으로 대체함으로써 더욱 세밀한 표현이 가능해져 양자화 오류가 감소하고 출력에서 더 높은 충실도를 달성합니다. 이는 특정 해상도나 도메인에 과적합되지 않는 표현 방식으로, 다양한 스타일 및 종횡비(aspect ratio)에 대한 일반화를 지원합니다.

Infinity-2B 모델은 정확한 프롬프트 추종, 공간적 추론, 텍스트 렌더링, 그리고 다양한 스타일 및 종횡비에 걸친 미적 표현 능력을 showcasing합니다.

### 3-3. BSC를 통한 분포 이동 대응 (Robustness)

BSC는 훈련 중 예측 부정확성을 에뮬레이션하고 특징을 재양자화함으로써 누적 오류를 처리하여 **모델의 견고성(resilience)을 향상**시킵니다. 이는 추론 시 발생하는 예측 오류 분포와 훈련 분포 간의 간극을 줄이는 **데이터 증강(augmentation)적 관점**에서의 일반화 전략입니다.

BSC의 일반화 메커니즘 수식:

$$\tilde{\mathbf{R}}_k^{(n)} = \mathbf{R}_k^{(n)} \oplus \mathbf{e}_k, \quad \mathbf{e}_k \sim \text{Bernoulli}(p_k)^d, \quad p_k \sim \mathcal{U}(0, p_{\max})$$

여기서 $\oplus$는 XOR 연산(비트 플립), $\mathbf{e}_k$는 노이즈 마스크입니다. 이 확률적 교란은 모델이 다양한 오류 강도에 견고하도록 훈련시킵니다.

### 3-4. 무한 어휘의 일반화 의미

어휘 확장은 재구성에 도움이 되며, 어휘 크기에 제한된 이산 VQ-VAE는 항상 연속적 토크나이저보다 뒤처졌는데, 이 연구에서는 어휘 크기 확장만으로 이산 VQ-VAE가 연속 대응물에 필적하는 수준의 학습에 성공했습니다. 이는 모델이 더 풍부한 특징 공간에서 패턴을 학습함으로써 미학습 이미지 유형에 대한 일반화 능력이 증대됨을 의미합니다.

---

## 4. 후속 연구에 미치는 영향 및 고려할 점

### 4-1. 연구에 미치는 영향

#### (A) 이산 생성 모델의 패러다임 전환

Infinity는 이산 생성 모델의 스케일링 및 시각적 세부 표현 능력을 획기적으로 향상시키며, 이 프레임워크는 이산 생성 커뮤니티에 '무한(infinity)'이라는 새로운 가능성을 열어줄 것으로 기대됩니다.

#### (B) AR 모델이 확산 모델을 능가하는 가능성 제시

Infinity는 AR 텍스트-이미지 생성 모델의 상한선을 크게 높이고, 선도적인 확산 모델에 필적하거나 능가함을 광범위한 실험으로 입증하여, 이 프레임워크가 자기회귀 시각 모델링 발전을 실질적으로 촉진하고, 더 빠르고 현실적인 생성 모델을 위한 커뮤니티에 영감을 줄 것으로 기대됩니다.

#### (C) 비디오/멀티모달 생성으로의 확장

이미 Infinity를 VAR에 기반한 **텍스트-비디오 생성(Text-to-Video generation)** 모델(InfinityStar)로 확장한 연구가 발표되었습니다. 이는 비트와이즈 모델링이 이미지를 넘어 시간축을 포함하는 시각 생성 전반에 적용 가능함을 시사합니다.

#### (D) Tokenizer 연구의 새 방향

Infinity는 토크나이저와 트랜스포머의 동시 스케일링을 통해 이미지 토크나이저로 근연속(near-continuous) 성능을 달성하고 확산 모델을 능가하는 고품질 텍스트-이미지 생성의 가능성을 시연합니다.

#### (E) 관련 후속 연구 흐름 (2024년 이후)

관련 후속 연구로는 **HART(Hybrid Autoregressive Transformer)**, **DART(Denoising Autoregressive Transformer)**, **M-VAR(Decoupled Scale-wise Autoregressive Modeling)**, **FlowAR(Scale-wise Autoregressive + Flow Matching)** 등이 있습니다.

이들 연구와의 비교 분석:

| 모델 | 패러다임 | 핵심 특징 |
|---|---|---|
| **VAR** (NeurIPS 2024 Best) | Next-Scale AR | 인덱스 기반 코스-투-파인 예측 |
| **Infinity** (CVPR 2025 Oral) | Bitwise AR | 비트와이즈 토큰, 무한 어휘, BSC |
| **HART** (2024) | Hybrid AR+Diffusion | AR 토큰 + 잔차 확산 결합 |
| **FlowAR** (2024) | AR + Flow Matching | Next-Scale + Flow Matching 결합 |
| **DART** (2024) | Denoising AR | 확산 과정을 AR 프레임워크에 통합 |

VAR은 이미지의 자기회귀 학습을 코스-투-파인 "next-scale prediction"으로 재정의하며, 확산 트랜스포머 대비 더 우수한 **일반화 및 스케일링 능력**을 더 적은 스텝으로 달성하고, LLM의 강력한 스케일링 특성을 활용하면서 확산 모델의 장점도 동시에 취할 수 있습니다. Infinity는 이 VAR의 기반 위에 비트와이즈 접근을 더해 표현력의 한계를 제거했습니다.

---

### 4-2. 앞으로 연구 시 고려할 점

#### (A) 어휘 크기의 최적점 탐색
- $V_d = 2^d$에서 $d$의 증가가 항상 성능 향상을 보장하는지에 대한 **포화점(saturation point)** 분석이 필요합니다.
- 비트 수 $d$와 재구성 품질(rFID), 생성 품질(FID, GenEval) 간의 관계에 대한 체계적 스케일링 법칙 확립이 요구됩니다.

#### (B) Train-Test 불일치의 완전한 해소
- BSC는 확률적 비트 플립으로 오류를 모방하지만, 실제 추론 시 오류의 패턴이 훈련 중 시뮬레이션과 다를 수 있습니다.
- **Scheduled Sampling**, **Reinforcement Learning 기반 자기교정** 등의 보완적 접근이 필요할 수 있습니다.

#### (C) 다운스트림 태스크 일반화 검증
- 저자들은 텍스트-이미지 합성뿐만 아니라 시각 생성 모델 전반에 걸친 발견의 함의를 지적하며, 기존 방법론의 재평가를 촉구합니다. 따라서 이미지 편집, 인페인팅, 제어 가능한 생성 등의 다운스트림 태스크로의 전이 가능성을 검증하는 연구가 필요합니다.

#### (D) 비디오 및 3D 생성으로의 확장
- 시간적 일관성(temporal consistency) 유지를 위한 비트와이즈 멀티스케일 표현의 확장이 핵심 과제입니다.
- FoundationVision 그룹은 이미 GRN이라는 차세대 시각 합성 프레임워크를 발표했는데, 이는 확산도 자기회귀도 아닌 제3의 방식입니다.

#### (E) 효율적 추론 연구
- 현재 Infinity가 달성한 0.8초 추론 속도는 고무적이나, 실시간 응용을 위해서는 추가적인 **지식 증류(Knowledge Distillation)**, **투기적 디코딩(Speculative Decoding)** 등의 추론 가속 기법과의 결합이 필요합니다.

#### (F) 공정성 및 안전성
- 무한 어휘와 고해상도 생성 능력의 향상은 딥페이크, 허위 정보 생성 등의 악용 가능성도 동반합니다. 워터마킹 및 생성 이미지 탐지 기술과의 병행 연구가 필수적입니다.

---

## 📚 참고자료 및 출처

| 번호 | 참고자료 | URL |
|---|---|---|
| 1 | **Infinity 논문 (arXiv)** | https://arxiv.org/abs/2412.04431 |
| 2 | **Infinity 공식 GitHub** | https://github.com/FoundationVision/Infinity |
| 3 | **Infinity 프로젝트 페이지** | https://foundationvision.github.io/infinity.project/ |
| 4 | **CVPR 2025 공식 논문 페이지** | https://openaccess.thecvf.com/content/CVPR2025/html/Han_Infinity_Scaling_Bitwise_AutoRegressive_Modeling_for_High-Resolution_Image_Synthesis_CVPR_2025_paper.html |
| 5 | **IEEE Xplore** | https://ieeexplore.ieee.org/document/11092840/ |
| 6 | **Hugging Face Paper Page** | https://huggingface.co/papers/2412.04431 |
| 7 | **MarkTechPost 분석 기사** | https://www.marktechpost.com/2024/12/10/bytedance-introduces-infinity-... |
| 8 | **VAR 공식 GitHub** (베이스라인) | https://github.com/FoundationVision/VAR |
| 9 | **Moonlight Literature Review** | https://www.themoonlight.io/en/review/infinity-scaling-bitwise-autoregressive-modeling-... |
| 10 | **CVPR 2025 Poster 페이지** | https://cvpr.thecvf.com/virtual/2025/poster/34414 |
