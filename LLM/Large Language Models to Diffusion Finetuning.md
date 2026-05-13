# Large Language Models to Diffusion Finetuning (L2D)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 사전학습된 대규모 언어 모델(LM)에 **Diffusion 프레임워크의 테스트 시간 연산 확장(test-time compute scaling) 능력**을 부여하는 새로운 파인튜닝 방법인 **L2D (LM to Diffusion)**를 제안합니다.

오토회귀(autoregressive) LM은 강력하지만, 추론 난이도에 따라 연산량을 동적으로 조절하는 능력이 부족합니다. 반면 Diffusion 모델은 반복적 생성 단계를 통해 연산 확장이 가능하지만, 언어 도메인에서의 성능은 오토회귀 LM에 비해 크게 뒤처집니다. L2D는 이 두 프레임워크의 장점을 결합합니다.

### 주요 기여

1. **L2D 프레임워크 제안**: 사전학습된 LM을 단일 스텝 Diffusion으로 재해석하고, 소량의 새 파라미터를 추가하여 멀티스텝 추론 능력 부여
2. **일관된 성능 향상**: Llama 3 및 Qwen 2.5 계열 4개 모델에서 수학, 코딩, 일반 지식 태스크 전반에 걸쳐 향상 입증
3. **직교적(orthogonal) 방향 제시**: 기존 파인튜닝 및 서치(search) 접근법과 완전히 호환되며 상호 보완적임을 실험적으로 입증
4. **Diffusion 고급 기법 적용**: Adaptive ODE solver, Classifier-free guidance 등 비전 도메인의 Diffusion 기법을 언어 모델에 적용

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

| 문제 | 설명 |
|------|------|
| LM의 연산 확장 불가 | 오토회귀 LM은 생성 토큰 수에 종속되어, 문제 난이도에 따른 동적 연산 배분이 불가 |
| Diffusion LM의 낮은 성능 | 언어 도메인에서 처음부터 Diffusion 학습 시 오토회귀 LM 대비 성능 열세 (Lou et al., 2024; Gulrajani & Hashimoto, 2024) |
| 두 패러다임의 분리 | AR의 강력한 표현력과 Diffusion의 반복적 정제(iterative refinement) 특성이 통합되지 않음 |

### 2.2 제안하는 방법 및 수식

#### Gaussian Diffusion 기초

corruption process를 통해 타깃 데이터 $x_1 \sim p^*$와 노이즈를 혼합하여 $x_t$를 생성합니다:

$$x_t = \alpha_t x_1 + \beta_t x_0, \quad x_0 \sim \mathcal{N}(0, I) \tag{1}$$

여기서 $\alpha_t = t$, $\beta_t = (1-t)$이며, Rectified Flow 스케줄을 따릅니다. 베이스 분포는 $p_0 := \mathcal{N}(0, \sigma^2 I)$로 정의됩니다 ($\sigma = 64$).

#### L2D 훈련 손실 (핵심 수식)

연속 도메인의 MSE 손실 대신, 언어 모델링과의 직접적 연결을 위해 **Cross-Entropy 손실**을 사용합니다:

$$L^{CE}(\theta) = -\mathbb{E}_{x_0, x_1, t}\left[\log\left(f_\theta(x_t, t, c)_y\right)\right] \tag{3}$$

where:
$$x_0 \sim \mathcal{N}(0, \sigma^2 I), \quad x_1 = V_y \sim p_1$$
$$t \sim \mathcal{U}[0,1], \quad x_t = tx_1 + (1-t)x_0$$

$f_\theta$는 어휘 토큰에 대한 $|V|$개의 로짓을 출력하는 Diffusion 네트워크이며, $c$는 선행 문맥 토큰, $y$는 레이블입니다.

**참고**: 기존 DDPM의 MSE 손실:

$$L^{L2}(\theta) = \mathbb{E}_{t, x_0, x_1}\left[\|x_1 - f_\theta(x_t, t)\|_2^2\right] \tag{2}$$

#### 추론 ODE (Constant Velocity Formulation)

Liu et al. (2022)의 Rectified Flow 속도 공식을 채택합니다:

$$dx_t = \frac{\hat{x} - x_t}{1 - t} \tag{4}$$

Euler 적분: $x_{t+\Delta t} = x_t + \Delta t \cdot dx_t$

#### 어휘 임베딩 정규화

Diffusion 경로의 어휘 임베딩 $V_y$를 구성할 때, 임베딩 크기의 무한 증가를 막기 위해 L2 정규화를 수행합니다:

$$V_y = \sqrt{\bar{d}} \cdot \frac{W_v V^l_y}{\|W_v V^l_y\|_2}, \quad \text{for all } y = 1, \ldots |V| \tag{5}$$

여기서 $W_v \in \mathbb{R}^{\bar{d} \times d}$는 저차원 임베딩 매핑 행렬 ($\bar{d} = 256$)입니다.

#### Timestep 조건화 가중치 (단일 스텝 능력 보존)

Diffusion 경로의 출력 가중치 $w_d$를 다음과 같이 정의합니다:

$$w_d(t) = w_{\theta_d}(t) - w_{\theta_d}(0) \tag{6}$$

이 설계의 핵심: $t=0$일 때 $w_d = 0$이 되어 Diffusion 경로가 LM 출력에 영향을 주지 않음 → **원본 LM의 단일 스텝 성능 완벽 보존**.

#### Classifier-Free Guidance

태스크 레이블 $j$를 조건으로 한 가이드 예측:

$$\hat{x}_g = w_g \times f_\theta(x_t, t, g_j, c) - (1 - w_g) \times f_\theta(x_t, t, g_0, c) \tag{7}$$

where $w_g \geq 1$은 가이던스 강도 파라미터, $g_0$는 null class embedding입니다.

#### 가중 평균 속도 추정 (비교 대상)

Dieleman et al. (2022)의 방법:

$$\hat{x} = \sum_y f_\theta(x_t, t, c)_y \times V_y \tag{8}$$

본 논문에서는 확률적 샘플링이 Diffusion의 self-correcting 특성을 더 잘 활용한다는 것을 실험적으로 보여줍니다.

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────────┐
│                    L2D 아키텍처                          │
│                                                         │
│  입력 컨텍스트 y^{0:k-1}                                 │
│         ↓                                               │
│  [메인 LM 경로 f_{θ_l}] ←── 동결된 사전학습 가중치       │
│  Self-Attention + MLP                                   │
│         ↓ (KV 캐시 저장)                                 │
│                                                         │
│  [Diffusion 경로 f_{θ_d}]                               │
│  - Cross-Attention (Q만 학습, KV는 메인 경로 공유)        │
│  - MLP (LoRA로 최적화)                                   │
│  - Adaptive RMS Norm (timestep t 조건화)                 │
│  - t-Embedder (sinusoidal features)                     │
│         ↓                                               │
│  f_{θ_l} + w_d × f_{θ_d}  ← 최종 레이어에서만 병합       │
│         ↓                                               │
│  LM Head → logits of ŷ^k                               │
└─────────────────────────────────────────────────────────┘
```

**핵심 구조적 특징**:

- **병렬 Diffusion 경로**: 메인 LM 경로와 분리된 병렬 Transformer 블록
- **LoRA 기반 최적화**: Diffusion 경로를 메인 LM 가중치로 초기화 후 LoRA로 미세조정
- **최종 레이어에서만 병합**: KV 캐시 재사용으로 추론 시 메인 경로는 1회만 실행
- **완전한 배치 병렬화**: 각 시퀀스 위치별 독립적 timestep 샘플링 가능

**추론 알고리즘 (Algorithm 1)**:
1. $x_t \sim \mathcal{N}(0, \sigma^2 I)$로 초기화
2. 각 Diffusion 스텝: $y_t \sim f_\theta(x_t, t, c)$ 샘플링 → $\hat{x} \leftarrow V_{y_t}$
3. $dx_t = \frac{\hat{x} - x_t}{1-t}$ 계산 후 Euler 업데이트
4. $T=15$ 스텝 (Midpoint solver, 8 이산화 레벨) 기본 사용

### 2.4 성능 향상

#### 주요 정량적 결과 (Table 1)

| 모델 | 방법 | 평균 점수 | 파라미터 수 |
|------|------|-----------|------------|
| Llama 3.2 1B | Base | 28.54 | - |
| | LoRA | 29.97 | 3M |
| | Full FT | 27.04 | 1,235M |
| | **L2D** | **35.50** | **73M** |
| Llama 3.1 8B | Base | 54.09 | - |
| | LoRA | 55.95 | 13M |
| | Full FT | 47.15 | 8,030M |
| | **L2D** | **61.33** | **281M** |
| Qwen 2.5 7B | Base | 46.65 | - |
| | LoRA | 63.34 | 10M |
| | **L2D** | **67.58** | **233M** |

**핵심 관찰**: Full FT는 Instruct 모델에서 코딩 성능 급락이 빈번하지만, L2D는 일관된 향상을 보임.

#### 확장 실험 (Table 2, Llama 3.2 1B 기준)

| 방법 | 수학 | 코딩 | 전체 |
|------|------|------|------|
| Base | 11.93 | 47.63 | 28.54 |
| L2D (15 steps) | 28.02 | 49.80 | 35.50 |
| L2D (127 steps) | 28.39 | 51.90 | 36.24 |
| L2D (adaptive solver) | 30.26 | 49.53 | 36.34 |
| L2D + token search | **35.95** | **49.79** | **38.57** |
| L2D (guided, tuned $w_g$) | 29.14 | 50.57 | 36.26 |

### 2.5 한계

1. **신뢰도 점수 손실**: Diffusion 멀티스텝 생성 시 모델의 ground-truth 신뢰도 점수(log-likelihood)에 직접 접근 불가
2. **로짓 조작 불가**: 온도 스케일링, top-p/top-k 등 기존 logit 조작 기법 적용 어려움
3. **추론 오버헤드**: 멀티스텝 추론으로 인한 추가 계산 비용
4. **평가 일관성 제약**: 스토캐스틱 생성으로 인해 pass@k 설정에서 평가 방식 조정 필요
5. **최적 스케줄 탐색**: $\sigma$, timestep 스케줄 등 하이퍼파라미터 민감도

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화 메커니즘

L2D의 일반화 성능 향상은 다음 핵심 메커니즘에서 비롯됩니다:

**① 가중치 비변경(Weight Preservation)**

$w_d(t) = w_{\theta_d}(t) - w_{\theta_d}(0)$의 설계로 메인 LM 가중치를 완전히 동결합니다. 이는 기존 파인튜닝에서 빈번히 발생하는 **catastrophic forgetting**을 방지합니다.

> 논문 인용: "augmenting the model to leverage past computation and improve future predictions, without suffering the potential downsides of trying to alter its capabilities and knowledge."

**② 훈련 데이터 외 태스크에서의 일반화**

파인튜닝 데이터는 수학·코딩 중심임에도, MMLU/MMLU-Pro 등 일반 지식 태스크에서도 소폭 향상을 보입니다:

- Llama 1B: MMLU 38.46 → 41.99 (+3.53), MMLU-Pro 13.63 → 15.35 (+1.72)
- Llama 8B: MMLU 63.83 → 66.69 (+2.86), MMLU-Pro 31.85 → 35.28 (+3.43)

저자들은 이를 "multi-step 추론의 귀납적 편향이 모델이 사전 습득한 지식을 더 잘 추출하도록 돕는다"고 해석합니다.

**③ 적응적 연산 배분(Adaptive Computation)**

Adaptive ODE solver 적용 시, 태스크 난이도에 따라 자동으로 Diffusion 스텝 수를 조절합니다 (Figure 4):

- MATH, HumanEval: 더 많은 스텝 (높은 난이도)
- GSM8K: 적은 스텝 (낮은 난이도)
- MMLU, MMLU-Pro (단일 토큰 답변): 가장 많은 스텝

이 메커니즘은 **태스크별 연산 최적화**를 자율적으로 수행하므로, 다양한 난이도 분포를 가진 새로운 태스크에서도 강건한 성능을 기대할 수 있습니다.

**④ Classifier-Free Guidance를 통한 도메인 특화**

$$\hat{x}_g = w_g \times f_\theta(x_t, t, g_j, c) - (1 - w_g) \times f_\theta(x_t, t, g_0, c)$$

이 메커니즘은 프롬프트 엔지니어링 없이도 모델이 특정 도메인(수학, 코딩, 일반 지식)에 전문화된 출력을 생성하도록 합니다. 저자들은 더 세분화된 레이블링이 LM 개인화에도 활용될 수 있음을 제안합니다.

**⑤ 기존 방법과의 시너지**

| 조합 | 전체 점수 |
|------|-----------|
| Base | 28.54 |
| LoRA만 | 29.97 |
| L2D만 | 35.50 |
| Full FT + L2D | 35.84 |
| L2D + Token Search | **38.57** |
| L2D + CoT | **36.00** |

L2D는 Chain-of-Thought, RL 기반 추론(DeepSeek-R1), Token Search와 결합 시 추가적 이득을 보여 **직교적 일반화 방향**을 제시합니다.

**⑥ 다양한 모델 패밀리에서의 범용성**

Llama 3.2 1B, Llama 3.1 8B, Qwen 2.5 1.5B, Qwen 2.5 7B에 걸쳐 일관된 성능 향상을 보여, Cross-Entropy 사전학습된 모든 Foundation Model에 적용 가능함을 실증합니다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4.1 Diffusion Language Model 계열

| 연구 | 방법 | L2D와의 차이점 |
|------|------|---------------|
| **DDPM** (Ho et al., 2020) | MSE 손실, 연속 공간 Diffusion | 언어에 직접 적용 불가 |
| **Diffusion-LM** (Li et al., 2022) | 토큰 임베딩 공간 Diffusion + MSE | MSE 손실 사용, 처음부터 학습 |
| **CDCD** (Dieleman et al., 2022) | Cross-entropy + 토큰 정규화 | 처음부터 학습, 사전학습 LM 미활용 |
| **PLAID** (Gulrajani & Hashimoto, 2024) | 1B 연속 Diffusion LM | 124M GPT-2 수준 성능, 훈련 비용 막대 |
| **SEDD** (Lou et al., 2024) | 이산 공간 Diffusion (score entropy) | 이산 공간, 소규모 LM 수준 근접 |
| **Discrete Flow Matching** (Gat et al., 2024) | 이산 Flow Matching | 이산 공간, 소규모 LM 수준 |
| **L2D (본 논문)** | 사전학습 LM + 소량 파라미터 추가 | **원본 가중치 보존, 멀티스텝 확장** |

### 4.2 Test-Time Compute Scaling 계열

| 연구 | 방법 | L2D와의 비교 |
|------|------|-------------|
| **Chain-of-Thought** (Wei et al., 2022) | 중간 추론 단계 생성 토큰 활용 | 토큰 길이에 종속, L2D와 직교적 |
| **OpenAI o1** (Jaech et al., 2024) | RL 기반 long CoT | 막대한 RL 훈련 비용 필요 |
| **DeepSeek-R1** (Guo et al., 2025) | RL + distillation | 수학 특화, 코딩/일반지식 성능 저하 |
| **s1** (Muennighoff et al., 2025) | 테스트 시간 CoT 스케일링 | 토큰 생성 길이에 의존 |
| **Large Language Monkeys** (Brown et al., 2024) | 반복 샘플링 | 검증기 필요, 포화 현상 |
| **L2D (본 논문)** | Diffusion 멀티스텝 | **토큰 길이 무관, 단계적 정제** |

### 4.3 PEFT(Parameter-Efficient Fine-Tuning) 계열

| 연구 | 방법 | L2D와의 비교 |
|------|------|-------------|
| **LoRA** (Hu et al., 2021) | 저랭크 행렬 분해 | L2D는 LoRA와 호환, 상호 보완적 |
| **ControlNet** (Zhang et al., 2023) | 병렬 제어 경로 | L2D 아키텍처 설계의 참조 |
| **DiT** (Peebles & Xie, 2023) | Scalable Diffusion Transformers | L2D의 timestep 조건화 방식 참조 |

### 4.4 LM + Diffusion 결합 계열

| 연구 | 방법 | L2D와의 차이 |
|------|------|-------------|
| **DiffusionBERT** (He et al., 2022) | BERT로 마스킹 Diffusion 가속 | 인코더 모델, Masked Diffusion |
| **SSD-LM** (Han et al., 2022, 2023) | AR + Diffusion 계층적 결합 | 동시 학습, 서로 다른 언어 레벨 |
| **DGLM** (Lovelace et al., 2024) | 인코더-디코더 잠재 공간 Diffusion | latent space diffusion |
| **L2D (본 논문)** | 사전학습 LM → Diffusion 파인튜닝 | **원본 LM 능력 완전 보존** |

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 연구에 미치는 영향

**① 새로운 LM 스케일링 패러다임 제시**

기존의 "더 많은 파라미터" 또는 "더 긴 생성 토큰"에 의존하던 스케일링에서 벗어나, **추론 단계 수(diffusion steps)**를 통한 스케일링이라는 새로운 차원을 열었습니다. 이는 특히 엣지 디바이스나 제한된 메모리 환경에서 토큰 생성 비용 없이 품질을 향상시킬 수 있는 방향입니다.

**② 오토회귀 + Diffusion 통합 연구 촉진**

L2D는 두 패러다임이 경쟁 관계가 아닌 상호 보완적임을 실증하여, 향후 하이브리드 아키텍처 연구를 촉진할 것입니다.

**③ RL 기반 추론과의 시너지 가능성**

DeepSeek-R1 위에 L2D를 적용한 실험(Table 16)은 RL 훈련된 추론 모델에 Diffusion 스케일링을 추가하는 연구 방향을 시사합니다:

- DeepSeek-R1-Distill-Qwen-1.5B: 전체 점수 37.88
- + L2D: 전체 점수 42.27 (+4.39)

**④ Foundation Model 범용 적용 프레임워크**

Cross-Entropy로 사전학습된 모든 LM에 적용 가능하다는 점에서, 미래의 더 큰 모델(GPT-5, Llama 4 등)에도 직접 적용 가능한 프레임워크입니다.

**⑤ 컴퓨터 비전 기법의 언어 도메인 이전 가능성 확인**

Classifier-free guidance, adaptive ODE solver 등 비전 도메인의 Diffusion 기법이 언어 도메인에서도 효과적임을 실증하여, 양 도메인 간 기법 이전 연구를 촉진합니다.

### 5.2 앞으로 연구 시 고려할 점

**① 신뢰도 추정 문제 해결**

Diffusion 멀티스텝 과정에서 각 토큰의 실제 log-likelihood에 접근이 어렵습니다. 향후 연구에서는 Diffusion 경로 기반의 **신뢰도 프록시 추정기** 개발이 필요합니다.

**② 최적 Diffusion 스케줄 자동화**

현재 $\sigma$, timestep 분포 등을 수동으로 설정합니다. **동적 time-warping**이나 **학습 기반 스케줄 최적화**가 성능 향상의 여지가 있습니다 (논문 Appendix C.2 참조).

**③ RL 통합 탐구**

논문이 제안하는 미래 방향인 L2D + RL (Black et al., 2023; Wallace et al., 2024의 비전 도메인 RL-Diffusion 연구 참조):

$$\mathcal{L}_{RL-L2D} = \mathcal{L}^{CE}(\theta) + \lambda \cdot \mathcal{L}_{reward}(\theta)$$

형태의 결합 훈련이 탐구될 수 있습니다.

**④ 더 세밀한 Guidance 조건화**

현재 math/coding/general knowledge의 3가지 레이블만 사용하지만, 더 세분화된 레이블(예: 미적분, 선형대수, 알고리즘 등)이 더 강력한 전문화를 가능하게 할 것입니다.

**⑤ 대규모 모델에서의 검증**

현재 실험은 최대 8B 파라미터 모델까지 진행되었습니다. 70B, 405B 규모 모델에서의 스케일링 법칙 검증이 필요합니다.

**⑥ 멀티모달 확장**

Diffusion은 이미지, 오디오 등 연속 모달리티에서 강점을 가지므로, L2D 프레임워크의 멀티모달 LM 적용이 유망한 방향입니다.

**⑦ 이론적 보장 연구**

현재 L2D의 효과성은 주로 실험적으로 입증되었습니다. 왜 Cross-Entropy Diffusion이 언어 도메인에서 효과적인지에 대한 이론적 분석이 필요합니다.

**⑧ 평가 프로토콜 표준화**

Diffusion 생성의 스토캐스틱 특성으로 인해 기존 LM 평가 방법(greedy decoding, temperature sampling 등)과의 공정한 비교가 어렵습니다. 새로운 표준 평가 프로토콜 개발이 필요합니다.

---

## 참고 자료

**주요 출처:**
- **Cetin, E., Zhao, T., & Tang, Y. (2025).** "Large Language Models to Diffusion Finetuning." *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267.* arXiv:2501.15781v2
- **GitHub 코드:** https://github.com/SakanaAI/L2D

**논문 내 인용 주요 참고문헌:**
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS 33.*
- Liu, X., Gong, C., & Liu, Q. (2022). Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv:2209.03003
- Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685
- Dieleman, S., et al. (2022). Continuous diffusion for categorical data. arXiv:2211.15089
- Karras, T., et al. (2022). Elucidating the design space of diffusion-based generative models. *NeurIPS 35.*
- Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. arXiv:2207.12598
- Lou, A., Meng, C., & Ermon, S. (2024). Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv:2310.16834
- Gulrajani, I., & Hashimoto, T. B. (2024). Likelihood-based diffusion language models. *NeurIPS 36.*
- Guo, D., et al. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv:2501.12948
- Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. *ICCV 2023.*
