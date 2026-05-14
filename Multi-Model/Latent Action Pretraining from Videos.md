
# LAPA: Latent Action Pretraining from Videos

---

## 📌 1. 핵심 주장과 주요 기여 요약

LAPA(Latent Action Pretraining for general Action models)는 **정답 로봇 액션 레이블 없이** Vision-Language-Action(VLA) 모델을 사전학습하는 **최초의 비지도 학습 방법**입니다.

### 주요 기여 (Key Contributions)

| 기여 | 설명 |
|------|------|
| **① 비지도 VLA 사전학습** | 액션 레이블 없는 인터넷 영상으로 사전학습 가능 |
| **② 잠재 액션 양자화** | VQ-VAE 기반으로 이산적 잠재 액션 학습 |
| **③ SOTA 달성** | 기존 VLA 모델을 능가하는 성능 |
| **④ 사전학습 효율** | 기존 대비 30배 이상의 사전학습 효율 |

LAPA는 **CoRL 2024 LangRob Workshop Best Paper Award**를 수상했으며(75개 논문 중), **ICLR 2025**에 게재되었습니다.

---

## 📌 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존 VLA 모델은 사전학습 시 인간 원격 조작자가 수집한 액션 레이블을 필요로 하며, 이는 가능한 데이터 소스와 규모를 크게 제한합니다.

인터넷 영상으로부터 학습하는 데는 두 가지 주요 도전이 있습니다: 첫째, 웹의 원시 데이터 대부분에 명시적 액션 레이블이 없고, 둘째, 웹의 데이터 분포가 일반 로봇 시스템의 구현체 및 환경과 근본적으로 다릅니다.

---

### 2-2. 제안하는 방법 (3단계 파이프라인)

LAPA는 두 개의 사전학습 단계와 하나의 파인튜닝 단계로 구성됩니다. 첫 번째 단계에서는 VQ-VAE 기반 목적함수를 사용하여 원시 이미지 프레임 간의 양자화된 잠재 액션을 학습합니다. 이는 언어 모델링에서 사용하는 BPE(Byte Pair Encoding)와 유사하게, 사전 정의된 액션 사전 없이 원자적 액션을 토크나이징하는 것으로 볼 수 있습니다. 두 번째 단계에서는 Vision-Language Model을 사전학습하여 비디오 관찰과 태스크 설명을 기반으로 첫 번째 단계에서 파생된 잠재 액션을 예측하는 행동 복제(behavior cloning)를 수행합니다. 마지막으로, 소규모 로봇 조작 데이터셋에서 파인튜닝하여 잠재 액션을 실제 로봇 액션으로 매핑하는 방법을 학습합니다.

---

### 2-3. 수식 (LaTeX)

#### **[Stage 1] Latent Action Quantization (LAQ)**

잠재 액션 양자화 모델은 인코더-디코더 구조로, 현재 프레임 $o_t$와 다음 프레임 $o_{t+1}$을 입력으로 받아 잠재 액션을 출력합니다.

**인코더 (Inverse Dynamics Model):**

$$z_t = \text{Enc}(o_t, o_{t+1})$$

**VQ-VAE 양자화:**

$$\hat{z}_t = \arg\min_{c_k \in \mathcal{C}} \| z_t - c_k \|_2$$

코드북 $\mathcal{C} = \{c_1, c_2, \ldots, c_{|C|}\}$에서 가장 가까운 벡터로 양자화됩니다.

**디코더 (미래 프레임 재구성):**

$$\hat{o}_{t+1} = \text{Dec}(o_t, \hat{z}_t)$$

**VQ-VAE 전체 손실 함수:**

$$\mathcal{L}_{\text{LAQ}} = \underbrace{\| o_{t+1} - \hat{o}_{t+1} \|_2^2}_{\text{Reconstruction Loss}} + \underbrace{\| \text{sg}[z_t] - \hat{z}_t \|_2^2}_{\text{Codebook Loss}} + \beta \underbrace{\| z_t - \text{sg}[\hat{z}_t] \|_2^2}_{\text{Commitment Loss}}$$

여기서 $\text{sg}[\cdot]$은 stop-gradient 연산자이고, $\beta$는 commitment loss 가중치입니다.

> 그래디언트 붕괴(gradient collapse)를 방지하기 위해 NSVQ(Vali & Bäckström, 2022)를 활용하며, 이는 벡터 양자화 오류를 원래 오류와 정규화된 노이즈 벡터의 곱으로 대체합니다. 또한 초기 학습 단계에서 코드북 활용도를 극대화하기 위해 NSVQ의 코드북 교체 기법을 적용합니다.

---

#### **[Stage 2] Latent Pretraining (VLA Behavior Cloning)**

사전학습된 VLM이 현재 이미지 $o_t$와 태스크 설명 $l$을 조건으로 잠재 액션 $\hat{z}_t$를 예측:

$$\pi_\theta(\hat{z}_t \mid o_t, l) = \text{VLM}_\theta(o_t, l)$$

**행동 복제 손실:**

$$\mathcal{L}_{\text{LP}} = -\mathbb{E}_{(o_t, o_{t+1}, l) \sim \mathcal{D}} \left[ \log \pi_\theta(\hat{z}_t \mid o_t, l) \right]$$

---

#### **[Stage 3] Action Finetuning**

소규모 로봇 데이터($\mathcal{D}_{\text{robot}}$)로 잠재 액션 공간에서 실제 액션 공간 $a_t \in \mathbb{R}^d$으로의 매핑 학습:

$$\hat{a}_t = f_\phi(\hat{z}_t), \quad \mathcal{L}_{\text{FT}} = \mathbb{E}_{(o_t, a_t, l) \sim \mathcal{D}_{\text{robot}}} \left[ \| a_t - \hat{a}_t \|_2^2 \right]$$

---

### 2-4. 모델 구조

LAPA는 크게 **Latent Action Quantization**과 **Latent Pretraining** 두 단계로 나뉩니다. 첫 단계에서는 VQ-VAE 기반 목적함수를 통해 비디오의 연속 프레임 사이에서 이산화된 잠재 델타 정보를 캡처합니다. 다음으로, 사전학습된 VLM이 현재 이미지와 언어 명령이 주어졌을 때 LAQ 모델의 인코더가 지정한 잠재 액션을 예측하도록 훈련됩니다. 잠재 사전학습 후 VLA 모델을 소수의 정답 액션 레이블이 달린 궤적으로 파인튜닝하여 잠재 공간을 실제 액션 공간으로 매핑합니다.

LAQ 모델의 **인코더**는 이후 LAPA의 다음 단계에서 역동역학 모델(inverse dynamics model)로 활용되고, **디코더**는 신경 기반 폐루프 롤아웃 생성에 사용됩니다.

```
┌─────────────────────────────────────────────────────────┐
│                  LAPA 전체 파이프라인                     │
├──────────────┬──────────────────┬───────────────────────┤
│   Stage 1    │    Stage 2       │       Stage 3         │
│  LAQ 학습    │ Latent VLA 학습  │  Action Finetuning    │
│              │                  │                       │
│ (o_t,o_t+1) │  (o_t, lang) →  │  latent z → robot a   │
│  → VQ-VAE  │   latent z 예측  │  (소규모 robot 데이터)  │
│  → codebook  │   (behavior      │                       │
│  quantize   │    cloning)      │                       │
└──────────────┴──────────────────┴───────────────────────┘
```

---

### 2-5. 성능 향상

LAPA는 인간 조작 비디오만으로 사전학습했음에도 더 큰 로봇 데이터셋인 Bridgev2보다 우수한 성능을 보이며, 사전학습 효율에서 30~40배 더 효율적입니다(272 H100 시간 vs. OpenVLA의 21,500 A100 시간).

OpenVLA 대비 **6.22%** 성능 향상을 달성하였으며, 사전학습 효율은 30배 이상입니다.

다중 구현체 환경(Open-X 사전학습)에서, LAPA는 3개 태스크 중 2개에서 OpenVLA를 크게 능가하며, 평균 성공률 **50.1%** vs. OpenVLA **43.9%**를 달성합니다.

---

### 2-6. 한계

강력한 전반적 일반화에도 불구하고, LAPA는 파지(grasping)와 같은 세밀한 동작 생성 태스크에서 한계를 보이며 OpenVLA보다 성능이 낮을 때도 있습니다. 예를 들어, pick-and-place 태스크에서 도달(reaching) 성능은 우수(83.33% vs. 66.67%)하지만 초기 파지(grasping) 단계에서 어려움을 보이며, 이는 현재의 잠재 액션 공간이나 제한된 파인튜닝 데이터(150개 궤적)가 다양한 물체에 대한 파지 액션 예측에 충분하지 않을 수 있음을 시사합니다.

LAPA는 사전학습 시 정답 액션 레이블을 사용하지 않기 때문에, 세밀한 모션 플래닝이 필요한 복잡한 태스크에서는 성능이 부족할 수 있습니다.

---

## 📌 3. 모델의 일반화 성능 향상 가능성

### 3-1. Cross-Embodiment 일반화

LAPA는 공유된 잠재 액션 공간을 사전학습에 활용함으로써 다중 구현체 환경에서의 효과성을 보여주는데, 이는 언어 및 이미지 표현이 비지도 방식으로 학습되는 것과 유사합니다. 반면, 기존의 액션 사전학습 방법들은 서로 다른 구현체와 데이터셋 간의 액션 표현 공간 차이로 인해 데이터셋 간 긍정적 전이(positive transfer)가 감소할 수 있습니다.

다중 구현체 환경에서, 각 잠재 액션이 구현체가 다르더라도 유사한 의미적 액션으로 매핑될 수 있음을 관찰하였습니다. 이는 잠재 액션이 구현체나 데이터셋에 무관하게 공유 표현 공간에서 학습된다는 주장을 뒷받침하며, 다양한 데이터셋 간의 더 강한 긍정적 전이를 가능하게 합니다.

### 3-2. Human-to-Robot Transfer

인간 조작 비디오에서 LAPA를 확장하기 위해, Something-Something V2 데이터셋(220K 영상)으로 LAPA를 사전학습하고 로봇 구현체에 파인튜닝한 결과, 인간 비디오로 학습된 LAPA가 OpenVLA(Bridge)보다 평균적으로 우수한 성능을 보였습니다. 인간-로봇 간 더 큰 구현체 격차(Human→Robot vs. Robot→Robot)에도 불구하고 로봇 조작에 더 나은 사전 지식을 학습한 것입니다. 이 결과는 시간 집약적 원격 조작이 필요한 값비싼 로봇 조작 데이터와 비교하여, 웹의 원시 인간 조작 비디오의 잠재력을 부각합니다.

### 3-3. 미래 스케일링 가능성

YouTube 영상 등 대규모 인터넷 비디오에 본 방법을 적용하면, NLP나 컴퓨터 비전의 파운데이션 모델과 유사하게 제너럴리스트 액션 파운데이션 모델의 대규모 사전학습의 잠재력을 열 수 있을 것으로 기대됩니다.

LAPA의 잠재 액션 표현은 사전학습 시 정답 액션에 의존하지 않음으로써, 특정 로봇 구현체와 액션 공간에 과적합되는 것을 피하고, 파인튜닝 시 액션 분포 변화에 대한 적응력을 향상시킵니다.

---

## 📌 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 연구명 | 연도 | 핵심 방법 | LAPA와의 비교 |
|--------|------|-----------|--------------|
| **VPT** (Baker et al., NeurIPS 2022) | 2022 | 온라인 비디오 시청으로 행동 학습 (IDM + BC) | 게임 도메인 중심; LAPA는 28.14% vs VPT 18.0% (cross-env) |
| **Genie** (Bruce et al., ICML 2024) | 2024 | 2D 게임 환경의 비지도 잠재 액션 학습 | 게임 특화; LAPA는 실제 로봇 조작으로 확장 |
| **OpenVLA** (Kim et al., 2024) | 2024 | 대규모 정답 액션 레이블로 VLA 학습 | LAPA는 정답 레이블 없이 OpenVLA를 능가 |
| **ConLA** (Dai et al., 2025) | 2025 | 대조 학습 기반 잠재 액션 분리 | LAPA의 shortcut learning 문제를 해결 |
| **LAWM** (2025) | 2025 | World Modeling 기반 잠재 액션 사전학습 | LAPA의 경량화 및 모델-무관 방식으로 확장 |

후속 연구 ConLA에서는 기존 VQ-VAE 기반 잠재 액션 학습 방법(LAPA 포함)이 **shortcut learning** 문제를 겪음을 지적하는데, 이는 모델이 실제 모션 역학보다 시각적 외관 단서에 과도하게 의존하는 현상입니다. 이를 해결하기 위해 대조 학습(contrastive learning)을 도입하여 시각적 표현과 액션 표현을 분리하여 진정한 모션 의미를 더 충실히 포착하는 잠재 액션을 가능하게 합니다.

ConLA는 인간 비디오만으로 사전학습 시 SimplerEnv 벤치마크에서 LAPA 대비 **12.5%** 성능 향상을 달성하며, 실제 로봇 궤적으로 사전학습한 모델을 **1.1%** 초과하는 성능을 보입니다.

LAPA와 villa-X 등 최근 방법들은 잠재 액션 표현을 도입하여 레이블 없는 데이터셋의 비지도 사전학습을 가능하게 하여 강력한 결과를 보였지만, 대형 모델 크기로 인해 실제 환경 배포가 어렵다는 한계가 있습니다.

---

## 📌 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

#### ① 데이터 병목 해결의 새로운 패러다임
다양한 실제 로봇 데이터셋은 대부분 인간 원격 조작을 필요로 하여 스케일링이 어렵습니다. 반면, 인터넷 영상 데이터는 인간 행동과 물리적 상호작용의 풍부한 사례를 대규모로 제공하며, 소규모 전문 로봇 데이터셋의 한계를 극복하는 유망한 접근법을 제시합니다.

#### ② Robotics Foundation Model로의 길 개척
인간 조작 비디오만으로 학습해도 긍정적 전이(positive transfer)가 나타남을 보여, 로보틱스 파운데이션 모델을 위한 웹 규모 데이터 활용 가능성을 열었습니다.

#### ③ 비지도 표현 학습의 로보틱스 적용 촉진
LAPA의 접근 방식은, 언어와 이미지 표현이 비지도 방식으로 학습되는 것과 유사하게, 공유 잠재 액션 공간을 사전학습에 활용하는 다중 구현체 환경에서의 효과성을 부각합니다.

---

### 5-2. 앞으로 연구 시 고려할 점

#### ① Shortcut Learning 문제 해결
VQ-VAE 기반 잠재 액션 학습에서 시각적 노이즈에서 순수한 잠재 액션을 추출하는 문제가 핵심 과제입니다. 비지도 환경에서 모션을 혼합된 시각적·모션 단서로부터 분리하는 것은 어렵기 때문에 의미있는 잠재 액션 추출을 안내하는 명시적 사전(prior)이 필요합니다.

#### ② 세밀한 동작 정밀도 향상
LAPA는 사전학습 시 정답 액션 레이블을 사용하지 않기 때문에, 세밀한 모션 플래닝이 필요한 복잡한 태스크에서 부족할 수 있습니다. 따라서 파인튜닝 단계에서의 데이터 효율성 향상 및 액션 정밀도 개선 연구가 필요합니다.

#### ③ 대규모 인터넷 영상 활용
YouTube 영상 등 대규모 인터넷 비디오에 접근 방식을 적용하면, NLP나 컴퓨터 비전의 파운데이션 모델처럼 제너럴리스트 액션 파운데이션 모델의 대규모 사전학습 가능성을 열 수 있을 것으로 기대됩니다.

#### ④ 모델 경량화와 실제 배포
현재 대형 모델 크기는 실제 환경 배포를 어렵게 만드는 공통적 한계이므로, 모델 압축, 증류(distillation), 경량 아키텍처 탐색이 중요한 연구 과제입니다.

#### ⑤ 도메인 갭(Domain Gap) 해소
임의의 구현체와 시점에서 레이블 없는 데이터를 활용할 수 있는 방향으로 연구가 발전해야 하며, 대규모 비디오 사전학습 후 최소한의 비용으로 다양한 로봇에 배포할 수 있는 크로스-구현체 제너럴리스트 정책 개발이 필요합니다.

---

## 📚 참고 자료 (References)

| # | 출처 | 링크 |
|---|------|-------|
| 1 | **[Main Paper] Ye et al. (2024). "Latent Action Pretraining from Videos." arXiv:2410.11758. ICLR 2025.** | https://arxiv.org/abs/2410.11758 |
| 2 | **[Project Page] LAPA Official Website** | https://latentactionpretraining.github.io/ |
| 3 | **[Code] GitHub - LatentActionPretraining/LAPA [ICLR 2025]** | https://github.com/LatentActionPretraining/LAPA |
| 4 | **[Review] OpenReview - Latent Action Pretraining from Videos** | https://openreview.net/forum?id=VYOe2eBQeh |
| 5 | **[Model] HuggingFace - LAPA-7B-openx** | https://huggingface.co/latent-action-pretraining/LAPA-7B-openx |
| 6 | **[HTML Paper] arXiv HTML - Latent Action Pretraining From Videos v1** | https://arxiv.org/html/2410.11758v1 |
| 7 | **[Summary] MarkTechPost - LAPA 논문 요약** | https://www.marktechpost.com/2024/10/20/... |
| 8 | **[Review] Liner.com - LAPA Quick Review** | https://liner.com/review/latent-action-pretraining-from-videos |
| 9 | **[Follow-up] Dai et al. (2025). "ConLA: Contrastive Latent Action Learning from Human Videos." arXiv:2602.00557** | https://arxiv.org/html/2602.00557v1 |
| 10 | **[Follow-up] "Latent Action Pretraining Through World Modeling (LAWM)." arXiv:2509.18428** | https://arxiv.org/html/2509.18428v1 |
| 11 | **[Follow-up] "Learning to Act Anywhere with Task-centric Latent Actions (UniVLA)." arXiv:2505.06111** | https://arxiv.org/pdf/2505.06111 |
| 12 | **[Citation] "Unifying Latent Action and Latent State Pre-training." SIGGRAPH Asia 2025.** | https://dl.acm.org/doi/10.1145/3757377.3763966 |
| 13 | **[NVIDIA Research Page] Latent Action Pretraining from Videos** | https://research.nvidia.com/publication/2025-04_latent-action-pretraining-videos |
