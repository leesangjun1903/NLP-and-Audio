# AR-DIFFUSION: Auto-Regressive Diffusion Model for Text Generation

***

## 1. 핵심 주장과 주요 기여 (간결 요약)

**핵심 주장**

- 기존 텍스트 diffusion LM들은 모든 토큰을 동일한 timestep에서 **동시에(Non-Autoregressive, NAR)** 디노이징하기 때문에 자연어의 강한 순차 의존성을 제대로 활용하지 못한다는 점을 지적한다. [arxiv](https://arxiv.org/html/2305.09515v3)
- 이를 해결하기 위해, **좌→우(auto-regressive) 순서를 diffusion 과정 안에 내재화**한 **AR-DIFFUSION**을 제안한다. [arxiv](https://arxiv.org/abs/2305.09515)
- 위치에 따라 서로 다른 diffusion timestep(denoising depth)을 부여하는 **multi-level (sentence-level + token-level) diffusion**과, 매우 적은 denoising step으로도 동작 가능한 **skipping 메커니즘**으로,  
  - 기존 diffusion text 모델보다 **성능이 높거나 최소 동급**이면서  
  - SeqDiffuSeq 대비 최대 **100×**, GENIE 대비 최대 **600×** 빠른 디코딩을 달성한다. [arxiv](https://arxiv.org/pdf/2305.09515.pdf)

**주요 기여**

1. **Auto-Regressive Diffusion 개념 도입**  
   - 토큰 위치 $\(n\)$ 과 sentence-level timestep $\(t\)$ 에 의존하는 token-level timestep 함수 $\(f(n,t)\)$ 를 정의하여, **왼쪽 토큰은 적은 노이즈(적은 step)**, 오른쪽 토큰은 **많은 노이즈(더 많은 step)**를 거치도록 설계. [arxiv](https://arxiv.org/pdf/2305.09515.pdf)

2. **Multi-Level Diffusion + Dynamic Movement Speed**  
   - 문장 단위 timestep $\(t\)$ 와 토큰 단위 timestep $\(f(n,t)\)$ 를 결합해, 좌측 토큰이 먼저 “결정”되고 그 정보가 우측 토큰 디노이징에 영향을 미치도록 하는 **동적 속도 규칙(dynamic movement speed)** 제안. [arxiv](https://arxiv.org/html/2305.09515v3)

3. **Skipping 기반 초고속 디코딩**  
   - 전체 diffusion trajectory에서 일부 timestep만 **균등 간격으로 샘플링**하는 skipping 알고리즘을 도입,  
   - 2–3 step 수준에서도 GENIE의 20·2000 step에 비견되는 성능을 유지함을 보임. [arxiv](https://arxiv.org/pdf/2305.09515.pdf)

4. **다양한 Seq2Seq 태스크에서의 우수한 실험 결과**  
   - Summarization(XSUM, CNN/DM), MT(IWSLT14), CommonGen에서  
     - **AR/NAR/diffusion 기반 최신 모델들(Diffusion-LM, DiffuSeq, SeqDiffuSeq, GENIE, DINOISER 등)을 대부분 상회**하거나 동급 성능을 내면서  
     - 디코딩 속도는 크게 개선됨을 실험으로 제시. [arxiv](https://arxiv.org/pdf/2210.08933.pdf)

5. **극단적인 적은 step(2·3 step)에서도 강건한 성능**  
   - step을 극단적으로 줄였을 때 GENIE보다 성능 저하폭이 훨씬 작으며,  
   - 이는 구조 자체가 **일반화·강건성 측면에서 유리**할 수 있음을 시사. [arxiv](https://arxiv.org/html/2305.09515v3)

***

## 2. 논문이 다루는 문제·제안 방법·모델 구조·성능 및 한계

### 2.1 해결하고자 하는 문제

1. **자연어의 강한 순차 의존성 미활용 문제**

   - 대부분의 diffusion 기반 텍스트 모델(Diffusion-LM, DiffuSeq, SeqDiffuSeq, GENIE 등)은  
     모든 토큰이 동일 timestep에서 **동시에 디노이징**되는 **NAR 패턴**을 따른다. [arxiv](https://arxiv.org/abs/2205.14217)
   - 이 경우 Transformer 내 self-attention으로 어느 정도 순서 정보를 이용할 수 있으나,  
     **“왼쪽이 먼저 결정되고 그 결과가 오른쪽을 제약하는” 강한 AR 구조**가 사라져,  
     자연스러운 문장 구조·호응·담화 구조 측면에서 AR LM에 비해 성능이 떨어진다는 결과들이 보고되어 왔다. [ijcai](https://www.ijcai.org/proceedings/2023/0750.pdf)

2. **Diffusion-LM류 모델의 느린 디코딩·실용성 문제**

   - Diffusion-LM, DiffuSeq, SeqDiffuSeq, GENIE 등은 보통 수백~수천 step을 디코딩에 사용해야 AR 수준의 품질을 낸다. [openreview](https://openreview.net/pdf?id=jQj-_rLVXsj)
   - 이는 Seq2Seq 작업(요약, 번역 등)에서 실시간 사용을 어렵게 만드는 주요 병목이다.

3. **AR vs Diffusion의 trade-off**

   - AR Transformer는 **고품질·강한 일반화**를 보여주지만, **철저히 순차적**이어서 병렬성이 떨어진다. [arxiv](https://arxiv.org/abs/2205.14217)
   - Diffusion LM은 **병렬성·다양성·제어가능성**이 강점이지만,  
     **순차 의존성과 효율적 디코딩** 측면에서 약점이 있다. [peerj](https://peerj.com/articles/cs-1905/)

> **문제식:**  
> “AR의 순차 구조와 diffusion의 병렬·다양성·제어 가능성을 **동시에** 가진 텍스트 생성 모델을 설계할 수 있는가?”

***

### 2.2 선행: 연속 텍스트 diffusion LM (Diffusion-LM)

AR-DIFFUSION은 Diffusion-LM의 연속 임베딩 공간 기반 formulation을 거의 그대로 계승한다. [arxiv](https://arxiv.org/pdf/2205.14217.pdf)

1. **연속 forward 과정**

   임의의 잠재 변수 $\(z_0\)$ 에서 시작해, Gaussian 노이즈를 점점 더 많이 섞어가는 과정:

$$
   q(z_t \mid z_0; x) = \mathcal{N}\bigl(z_t;\, \sqrt{\bar\alpha_t}\, z_0,\; (1-\bar\alpha_t) I\bigr),
   $$

   여기서 $\(\bar\alpha_t = \prod_{i=1}^t \alpha_i\), \(\alpha_i \downarrow\)$ with $\(t\)$ . [arxiv](https://arxiv.org/pdf/2205.14217.pdf)

2. **역방향 denoising 과정**

$$
   p_\theta(z_{t-1} \mid z_t; x) 
   = \mathcal{N}\!\bigl(
      z_{t-1};\,
      \mu_\theta(z_t, t; x),
      \Sigma_\theta(z_t, t; x)
   \bigr),
   $$

   여기서 평균 $\(\mu_\theta\)$ 는 신경망(decoder) $\(g_\theta(z_t, t; x)\)$ 로부터 추정하고, 분산은 고정 스케줄을 따른다. [arxiv](https://arxiv.org/abs/2205.14217)

3. **텍스트에 적용: embedding + rounding**

   - discrete token $\(y\)$ 를 임베딩 $\(z_0\)$ 로 매핑:

$$
     q_\phi(z_0 \mid y) = \mathcal{N}\!\bigl(z_0;\, \text{Emb}(y), (1-\alpha_0) I\bigr)
     $$

   - 역과정 마지막에 $\(z_0\)$ 를 가장 가까운 단어 임베딩으로 라운딩하여 토큰을 복원:

$$
     p_\theta(y \mid z_0; x) = \prod_{i=1}^N p_\theta(y_i \mid z_{0,i}; x).
     $$

4. **학습 목적(ELBO 기반 단순화)**

$$
   \mathcal{L}
   = \mathbb{E}_{q_\phi(z_{0:T} \mid y)}
     \Bigg[
         - \log p_\theta(y \mid z_0; x)
         + \sum_{t=1}^T
           \bigl\|z_0 - g_\theta(z_t, t; x)\bigr\|^2
     \Bigg].
   $$

이 기본 틀 위에, AR-DIFFUSION은 **“모든 토큰이 같은 timestep”**이라는 가정을 깨고, **위치 의존적 timestep**을 도입한다. [arxiv](https://arxiv.org/pdf/2305.09515.pdf)

***

### 2.3 AR-DIFFUSION의 핵심 아이디어: Multi-Level Diffusion & Dynamic Movement Speed

#### 2.3.1 2D 좌표계 관점

논문은 토큰 위치 $\(n\)$ 과 timestep $\(t\)$ 를 2차원 평면에 배치한다. [arxiv](https://arxiv.org/html/2305.09515v3)

- **가로축:** token position $\(n \in [1, N]\)$
- **세로축:** sentence-level timestep $\(t \in [0, T]\)$

이때, 각 토큰의 “상태”를 $\((n, f(n,t))\)$ 로 나타내며,  
$\(f(n,t)\)$ 는 위치·문장 timestep에 따른 **token-level timestep**이다. [arxiv](https://arxiv.org/pdf/2305.09515.pdf)

1. **기존 Diffusion-LM 스타일**  
   - 모든 토큰이 동일 timestep 공유  
     $\(\Rightarrow f(n,t) = t\)$ (수평선)  
   - 이동 속도

$$
     v(n, t_i, t_{i+1}) = f(n,t_{i+1}) - f(n,t_i) = t_{i+1} - t_i
     $$

   가 모든 $\(n\)$ 에 대해 동일. [arxiv](https://arxiv.org/abs/2205.14217)

2. **순수 AR 관점 (diffusion으로 재해석)**  
   - 이미 생성된 토큰: $\(t = 0\)$  
   - 아직 생성 전 토큰: $\(t = T\)$  
   - 한 step마다 “오른쪽 토큰 하나”가 $\(T \to 0\)$ 으로 점프하는 구조로 이해 가능. [arxiv](https://arxiv.org/html/2305.09515v3)

3. **AR-DIFFUSION의 목표**

   - 각 문장-level step $\(t\)$ 에서,  
     - **왼쪽 토큰은 더 작은 timestep(더 많이 디노이징된 상태)**  
     - **오른쪽 토큰은 더 큰 timestep(덜 디노이징된 상태)**  
   - 즉, **왼쪽이 먼저 결정되고 오른쪽은 이를 조건으로 동시에(병렬) 디노이징**되도록 시간 스케줄을 설계.

#### 2.3.2 Multi-Level Diffusion 수식

1. **문장-level timestep 선택**

   - Diffusion-LM과 동일하게, 전체 범위에서 문장 단위 timestep $\(t\)$ 를 랜덤 샘플:

$$
     t \sim \text{Uniform}(0, N + T)
     $$

   - 이때 forward 시작점을 $\((n_s, t_s)\)$ 로 정의:

$$(n_s, t_s) = \left( \max(0, N - t),\; \max(0, t - N) \right) $$


- $\(t \le N\)$ : 오른쪽에서 왼쪽으로 이동(가로축 방향)  
- $\(t > N\)$ : 왼쪽에서 위로 이동(세로축 방향) [arxiv](https://arxiv.org/pdf/2305.09515.pdf)

2. **Token-level timestep 함수 \(f(n,t)\)**

   - $\((n_s,t_s)\)$ 에서 미리 정해둔 **anchor point $\((n_e, t_e)\)$ **까지의 직선을 이용해, 각 위치별 timestep을 결정:

$$
f(n, t) = \min \left( T, \max \left( 0, \frac{t_e - t_s}{n_e - n_s} (n - n_s) + t_s \right) \right)
$$

   - 구현에서는 $\( (n_e, t_e) = (2N, T)\)$ 로 설정. [arxiv](https://arxiv.org/html/2305.09515v3)
   - 이 직선의 기울기 덕분에, **왼쪽 토큰 $(\(n\)$ 작을수록)**이 더 빠르게 작은 timestep(0 근처)로 이동한다.

3. **문장 표현 \(z_t\)**

   - 각 위치별 latent를 모아 문장-level 상태 $\(z_t\)$ 정의:

$$
     z_t = 
     \bigl[
       z^{(1)}_{f(1,t)},
       z^{(2)}_{f(2,t)},
       \dots,
       z^{(N)}_{f(N,t)}
     \bigr]
     $$

4. **토큰 이동 속도(dynamic movement speed)**

   - 문장-level timestep 두 지점 $\(t_i, t_{i+1}\)$ 에 대해

$$
     v(n, t_i, t_{i+1})
     = f(n, t_{i+1}) - f(n, t_i)
     $$

   - 왼쪽 토큰은 $\(v\)$ 가 커서 더 빨리 “원본 토큰 영역(작은 $\(t\)$ )”으로 이동하고,  
     오른쪽 토큰은 $\(v\)$ 가 작아 더 오랫동안 노이즈 상태를 유지.

> **핵심:**  
> diffusion 과정 전체를 “시공간 격자”로 보고, **시간 축 스케줄을 위치별로 다르게 설계**함으로써,  
> **병렬 디노이징이지만 결과적으로는 AR-like한 좌→우 의존성을 구현**한다.

***

### 2.4 학습 목표 (수식)

Multi-level diffusion을 포함한 전체 학습 objective는 다음과 같다. [arxiv](https://arxiv.org/pdf/2305.09515.pdf)

- 우선, Diffusion-LM 스타일의 ELBO 기반 손실에서,  
  token-level timestep $\(f(n,t)\)$ 를 반영해 각 위치별 reconstruction term을 사용:

```math
  \mathcal{L}
  =
  \mathbb{E}_{(x,y)\sim\mathcal{D}}
  \mathbb{E}_{t, z_{f(n,t)}}
  \Biggl[
    - \log p_\theta(y \mid z_0; x)
    + \sum_{n=1}^N
      \bigl\|
        g_\theta\bigl(
          z^{(n)}_{f(n,t)}, f(n,t); x
        \bigr)
        - z^{(n)}_0
      \bigr\|^2
  \Biggr]
```

여기서

- $\(z^{(n)}_{f(n,t)}\)$ : 위치 \(n\)에서 token-level timestep \(f(n,t)\)의 latent
- $\(g_\theta\)$ : encoder-decoder Transformer 기반 denoiser
- $\(p_\theta(y\mid z_0; x)\)$ : rounding을 통한 token-level likelihood

***

### 2.5 Skipping을 활용한 추론 알고리즘

기본적으로는 모든 sentence-level timestep $\(t = T+N \to 0\)$ 을 순차적으로 거쳐야 하지만,  
AR-DIFFUSION은 **skipping 메커니즘**으로 이를 대폭 줄인다. [arxiv](https://arxiv.org/html/2305.09515v3)

1. **문장-level skipping 시퀀스 선택**

   - $\(T+N\)$ 에서 0까지 균등 간격으로 $\(M+1\)$ 개의 timestep $\(\{t_i\}_{i=0}^M\)$ 선택:

$$
     T + N = t_0 > t_1 > \dots > t_M = 0,\quad
     M \ll T + N
     $$

2. **각 step에서 위치별 token-level timestep 계산**

   - 각 $\(t_i\)$ 에 대해 (6), (7)을 사용해

$$
     (n_s, t_s)
     = \text{식 }(6), \quad
     f(n, t_i), f(n, t_{i+1})
     = \text{식 }(7)
     $$

3. **위치별 독립 역방향 Gaussian 전이**

   - forward 과정이 각 위치에 대해 독립 Gaussian이므로,  
     위치별로 역전이 분포를 다음과 같이 쓸 수 있다:

$$
     p_\theta
     \bigl(
       z^{(n)}_{f(n,t_{i+1})}
       \mid
       z^{(n)}_{f(n,t_i)};\, x
     \bigr)
     =
     \mathcal{N}
     \bigl(
       z^{(n)}_{f(n,t_{i+1})};\;
       \lambda z^{(n)}_{f(n,t_i)} 
       + \mu\, g_\theta(
           z^{(n)}_{f(n,t_i)}, f(n,t_i); x
         ),
       \sigma^2 I
     \bigr)
     $$

   - 계수 $\(\lambda, \mu, \sigma\)$ 는 Diffusion-LM과 유사하게 $\(\bar\alpha_t\)$ 들로 닫힌형식:

$$
     \lambda
     =
     \frac{
       \bar\alpha_{f(n,t_i)}
       (1-\bar\alpha_{f(n,t_{i+1})})
     }{
       \bar\alpha_{f(n,t_{i+1})}
       (1-\bar\alpha_{f(n,t_i)})
     },
     \quad
     \mu
     =
     \frac{
       \bar\alpha_{f(n,t_{i+1})}
       \bigl(
         1 - \frac{
                \bar\alpha_{f(n,t_i)}
              }{
                \bar\alpha_{f(n,t_{i+1})}
              }
       \bigr)
     }{
       1-\bar\alpha_{f(n,t_i)}
     },
     $$

$$
     \sigma^2
     =
     \frac{
       (1-\alpha_{f(n,t_i)})
       (1-\bar\alpha_{f(n,t_{i+1})})
     }{
       1-\bar\alpha_{f(n,t_i)}
     }
     $$

4. **전체 문장-level 전이**

   - 모든 위치에 대해 독립 전이를 적용하면:

$$
     p_\theta(z_{t_{i+1}} \mid z_{t_i}; x)
     =
     \prod_{n=1}^N
       p_\theta\Bigl(
         z^{(n)}_{f(n,t_{i+1})}
         \mid
         z^{(n)}_{f(n,t_i)};\, x
       \Bigr)
     $$

> **효과:**  
> - 한 번의 step에서 각 토큰이 **긴 구간의 timestep을 건너뛰며** 한꺼번에 디노이징  
> - $\(M\)$ 이 매우 작아도(예: 20, 3, 2) 충분히 많은 timestep span을 커버.  
> - 특히, 좌측 토큰은 이미 작은 timestep 영역에 있기 때문에, skipping을 해도 **정보가 안정적으로 누적**된다. [arxiv](https://arxiv.org/pdf/2305.09515.pdf)

***

### 2.6 모델 구조

AR-DIFFUSION의 기본 아키텍처는 **Transformer-base encoder-decoder**로, ProphetNet/GENIE 계열 구현을 재사용한다. [arxiv](https://arxiv.org/pdf/2212.11685.pdf)

- **인코더:**  
  - 입력 시퀀스(요약의 원문, 번역의 source, CommonGen concept set 등)를 인코딩
- **디퓨전 디코더:**  
  - 임의의 Gaussian 노이즈에서 시작해, 각 step마다
    - sentence-level timestep $\(t\)$ 와 token-level timestep $\(f(n,t)\)$ 를 time embedding으로 인코딩
    - cross-attention을 통해 인코더 출력을 조건으로 사용
    - $\(z_0\)$ 또는 noise를 예측해 미분 가능한 denoising 수행 [proceedings.mlr](https://proceedings.mlr.press/v202/lin23d/lin23d.pdf)
- **Diffusion embedding dimension:**  
  - summarization: 128,  
  - MT/COMMONGEN: 64 등, task별로 조정. [arxiv](https://arxiv.org/html/2305.09515v3)
- **Tokenizer:**  
  - MT에 BPE, 나머지에는 BERT-base-uncased 토크나이저 사용. [arxiv](https://arxiv.org/pdf/2305.09515.pdf)

***

### 2.7 성능 향상: 실험 결과 요약

#### 2.7.1 Text Summarization (XSUM, CNN/DM)

- **XSUM** [arxiv](https://arxiv.org/html/2305.09515v3)
  - 기존 NAR/Semi-NAR, AR Transformer, GENIE와 비교:
    - XSUM에서 ROUGE-1/2/L 기준  
      - Transformer (AR): 30.5 / 10.4 / 24.2  
      - GENIE (diffusion, k=50): 29.3 / 8.3 / 21.9  
      - **AR-DIFFUSION (k=50): 31.7 / 10.1 / 24.7**  
      - **AR-DIFFUSION (k=500): 32.2 / 10.6 / 25.2**  
    - k=500에서 Transformer를 모든 지표에서 상회. [arxiv](https://arxiv.org/pdf/2305.09515.pdf)

- **CNN/DAILYMAIL** [arxiv](https://arxiv.org/html/2305.09515v3)
  - Transformer (AR): 39.5 / 16.7 / 36.7  
  - GENIE (k=50): 34.4 / 12.8 / 32.1  
  - **AR-DIFFUSION (k=50): 39.6 / 16.3 / 37.1**  
  - **AR-DIFFUSION (k=500): 40.2 / 17.1 / 37.7**

> **요약:** summarization에서는 **AR 수준의 품질을 유지하거나 능가**하면서, diffusion 기반으로 **MBR + multi-sample**을 통해 다양한 후보를 생성한다. [semanticscholar](https://www.semanticscholar.org/paper/cb648d482dbd1e6ad0b0f4da43aca71c06538d4f)

#### 2.7.2 Machine Translation (IWSLT14 De→En, En→De)

- **BLEU (SeqDiffuSeq setting)** [arxiv](https://arxiv.org/pdf/2210.08933.pdf)
  - CNAT (NAR): 29.81  
  - SeqDiffuSeq (k=1, 2000 step): 29.83  
  - GENIE (k=50, 20 step): 30.08  
  - Transformer (AR): 34.74  
  - **AR-DIFFUSION (k=50, 20 step): 34.95**  
  - **AR-DIFFUSION (k=500, 20 step): 35.62**

- **SacreBLEU (DINOISER setting)** [arxiv](https://arxiv.org/pdf/2302.10025.pdf)
  - DiffusionLM, GENIE 등과 비교 시  
  - AR-DIFFUSION (k=50)은 DINOISER와 유사하거나 약간 우수하고,  
    k=500에서는 DINOISER를 앞서는 결과 보고. [arxiv](https://arxiv.org/pdf/2302.10025.pdf)

> **해석:**  
> - **단일 샘플(k=1)에서도 NAR·SeqDiffuSeq·GENIE를 상회**  
> - 충분한 후보(k=50~500)와 MBR 결합 시, **AR Transformer 이상**의 성능을 확보.

#### 2.7.3 CommonGen (Commonsense Generation)

- CommonGen에서 AR/NAR/GENIE 대비 ROUGE-2/L, BLEU-3/4, METEOR, SPICE 모두에서 최고 또는 두 번째 성능. [aclanthology](https://aclanthology.org/2020.findings-emnlp.165.pdf)
- 특히 **SPICE·METEOR** 향상은 **구조적·상식적 일관성** 측면에서의 개선을 시사.

#### 2.7.4 추론 효율성 및 극저 step 성능

- **Machine Translation에서 SeqDiffuSeq 대비 100× 빠른 NFE** (20 vs 2000 step)에서 동급 혹은 더 나은 성능. [aclanthology](https://aclanthology.org/2024.naacl-long.2.pdf)
- **XSUM에서 step=2/3 실험** [arxiv](https://arxiv.org/pdf/2305.09515.pdf)
  - GENIE: 20 step 대비 avg ROUGE drop ≈ 2.8~4.2
  - AR-DIFFUSION: 20 step 대비 drop ≈ 0.64(3 step), 1.34(2 step)
  - **3 step의 AR-DIFFUSION ≈ 2000 step GENIE 성능**.

***

### 2.8 한계

논문에서 명시·시사하는 한계는 다음과 같다. [arxiv](https://arxiv.org/html/2305.09515v3)

1. **많은 후보 샘플(k)이 필요**

   - summarization에서 k=50과 k=500 사이에도 성능 차이가 상당함(예: CNN/DM ROUGE-2 약 0.8 차이). [arxiv](https://arxiv.org/pdf/2305.09515.pdf)
   - MBR를 쓰려면 **여러 후보 생성 + 후보 간 pairwise 유사도 계산**이 필요해,  
     **디코딩 wall-clock time은 여전히 비싸**다(특히 대형 모델/긴 시퀀스에서).

2. **학습 비용**

   - Diffusion-LM 계열처럼 **2000 diffusion step**으로 학습,  
     - 모든 step에서 forward/backward가 필요해 학습비용이 크다. [arxiv](https://arxiv.org/pdf/2210.08933.pdf)
   - skipping은 **추론에서만** 쓰이므로, 학습비용은 그대로.

3. **중소형 Seq2Seq 태스크에 국한**

   - XSUM, CNN/DM, IWSLT14, CommonGen 등 **비교적 짧은 시퀀스** 위주.  
   - LLaMA, GPT 수준의 **대규모 LM pretraining**에는 적용되지 않았으며,  
     장문 생성·장기 맥락 일반화에 대한 증거는 제한적이다.

4. **구조 복잡성 및 하이퍼파라미터 민감성**

   - anchor point \((n_e, t_e)\), sentence-level timestep 범위 \([0, N + T]\),  
     skipping 간격, diffusion 스케줄 등 하이퍼파라미터가 많다. [arxiv](https://arxiv.org/html/2305.09515v3)
   - anchor 위치 변경에 따라 MT 성능이 달라지는 실험도 보고되어,  
     세밀한 튜닝이 필요함을 시사. [arxiv](https://arxiv.org/pdf/2305.09515.pdf)

***

## 3. 모델의 일반화 성능 향상 가능성에 대한 분석

AR-DIFFUSION이 **일반화(특히 적은 step / 빠른 디코딩 / 데이터 효율성)** 측면에서 갖는 잠재력은 다음 관점에서 해석할 수 있다.

### 3.1 좌→우 AR 구조와 diffusion의 결합

- **ARLM의 강점**: left-to-right factorization  
  $\(\displaystyle p(y\mid x) = \prod_i p(y_i \mid y_{ < i}, x)\)$  
  은  
  - 강한 순차 의존성  
  - 학습·추론 시 일관된 조건화 구조를 제공해 일반화에 유리하다. [arxiv](https://arxiv.org/abs/2205.14217)

- **기존 diffusion LM의 약점**:  
  - 토큰 간 **조건화 순서가 모호**, 모든 토큰이 동일 timestep에서 동시 디노이징.  
  - 이는 **“어느 토큰이 조건이고 어느 토큰이 예측 대상인지”**가 시간축 상에서 명확히 분리되지 않아, 특히 문법·구문 구조 일반화에 불리할 수 있다. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10909201/)

- **AR-DIFFUSION의 장점**:

  1. 각 step에서 **좌측 토큰은 이미 더 많이 디노이징된 상태(작은 timestep)** → 사실상 “조건(outcome)”처럼 작동.
  2. 우측 토큰은 여전히 노이즈가 많아 “예측 대상”에 가까운 상태.
  3. 이 과정이 multi-step으로 반복되면서,  
     **“부분적으로 이미 디코딩된 prefix + 노이즈가 섞인 suffix”를 반복적으로 refine**하도록 학습된다.

- 이는 **exposure bias 완화 효과**도 제공할 수 있다.  
  - ARLM은 항상 정답 prefix를 조건으로 학습하지만, inference에서는 자기 자신의 출력을 조건으로 삼는다.  
  - AR-DIFFUSION에서는 intermediate latent가 noisy이지만 **모든 위치에서 joint denoising**을 하면서 prefix/suffix를 동시에 수정한다.  
  - 이 joint refinement는 “오류가 난 prefix를 나중에라도 수정”할 여지를 갖게 하여, 분포 shift에 대한 일반화에 유리할 수 있다.

### 3.2 극단적으로 작은 step에서의 강건성

- GENIE·DiffuSeq·SeqDiffuSeq 등의 기존 continuous diffusion LM들은 **step 수를 줄이면 품질이 크게 저하**되는 경향이 보고되었다. [openreview](https://openreview.net/pdf?id=jQj-_rLVXsj)
- AR-DIFFUSION은  
  - 3 step에서 20 step 대비 평균 ROUGE drop이 ≈0.64,  
  - 2 step에서도 ≈1.34에 불과해, **“few-step diffusion”에서도 높은 일반화 성능**을 보인다. [arxiv](https://arxiv.org/html/2305.09515v3)

> **시사점:**  
> - diffusion 모델의 일반적인 단점인 **“step 수–성능 trade-off”**가 완화된 구조이며,  
> - 이는 **초기화·동역학·조건화 구조가 학습 시 더 일관되게 일반화되었기 때문**일 가능성이 높다.

최근 마스크드 diffusion LM이 **데이터 제한(regime)에서 AR보다 더 나은 스케일링 특성을 가질 수 있다는 결과**도 보고되고 있어, [arxiv](https://arxiv.org/html/2507.15857v1)
AR-DIFFUSION처럼 **AR 구조를 diffusion 안에 심은 하이브리드 방식**은  
**데이터 효율·step 효율 측면에서도 유리한 설계 공간**임을 시사한다.

### 3.3 위치별 adaptive timestep이 일반화에 주는 효과

- CDCD, SeqDiffuSeq, DINOISER 등도 **adaptive noise schedule / 위치별 noise** 개념을 도입하여 일반화를 향상시키려 했다. [alphaxiv](https://www.alphaxiv.org/es/overview/2212.10325v5)
  - SeqDiffuSeq는 **adaptive noise schedule**로 timestep별 난이도를 균등화하고, 위치별 noise 스케줄을 분리. [aclanthology](https://aclanthology.org/2024.naacl-long.2/)
  - DINOISER는 **noise scale clipping**과 condition-enhanced denoiser로 소스 조건의 활용도를 높인다. [labxing](https://www.labxing.com/files/lab_data/1743-1687083938-i2FXe59J.pdf)

- AR-DIFFUSION은 한 단계 더 나아가,  
  - **“좌측 토큰은 빨리 확실해지고, 우측 토큰은 천천히 확실해지는”**  
    deterministic한 위치별 시간 설계로,  
  - **언어의 자연스러운 정보 흐름(문장 앞부분에서 토픽·주어·프레임 정립, 뒷부분에서 세부 묘사)**을 모사한다.

> **일반화 관점 가설:**  
> - 이런 위치별 시간 설계는 모델이  
>   - 문장의 **global structure**를 먼저 안정적으로 학습하고  
>   - 세부 정보를 점진적으로 refine 하도록 유도해,  
>   - **길이가 길어져도 구조적 일관성을 유지**하는 방향으로 일반화를 돕는다.

대규모 멀티태스크·멀티도메인 실험은 아직 부족하지만,  
이론적 관점과 비교 실험(다양한 task에서 AR·NAR·다른 diffusion 대비 우수/동급 성능)은 [dl.acm](https://dl.acm.org/doi/10.5555/3618408.3619275)
**일반화 잠재력**이 상당함을 뒷받침한다.

***

## 4. 2020년 이후 관련 최신 연구와 비교 분석

다음 표는 AR-DIFFUSION과 주요 관련 diffusion text 모델들을 **목적·아키텍처·AR성질** 관점에서 요약한 것이다.

| 모델 | 연도 | 주요 아이디어 | AR vs NAR 구조 | 비고 |
|------|------|---------------|----------------|------|
| Diffusion-LM [arxiv](https://arxiv.org/abs/2205.14217) | 2022 | 연속 임베딩 diffusion로 controllable generation | 완전 NAR (동일 timestep) | classifier guidance로 fine-grained control |
| DiffuSeq [arxiv](https://arxiv.org/pdf/2210.08933.pdf) | 2022 | Seq2Seq diffusion, encoder-only, 부분 noising | NAR | 높은 다양성, 느린 추론 |
| SeqDiffuSeq [alphaxiv](https://www.alphaxiv.org/es/overview/2212.10325v5) | 2022–24 | encoder-decoder + adaptive noise schedule + self-conditioning | NAR | DiffuSeq보다 빠르고 품질 향상 |
| GENIE [arxiv](https://arxiv.org/pdf/2212.11685.pdf) | 2022–23 | 대규모 pretraining + continuous paragraph denoise | NAR | GLGE·CommonGen 등에서 AR급 품질, 다양성↑ |
| DINOISER [arxiv](https://arxiv.org/pdf/2302.10025.pdf) | 2023 | noise scale 조작으로 조건 활용도↑ | NAR | MT 등 conditional seq 학습 향상 |
| CDCD [arxiv](https://arxiv.org/pdf/2211.15089.pdf) | 2022 | categorical data용 continuous diffusion | NAR | 언어·그래프 등 범용 discrete 데이터 |
| ARDMs [arxiv](https://arxiv.org/abs/2110.02037) | 2021–22 | discrete diffusion + order-agnostic autoregression 통합 | 순열 기반 AR | 텍스트·이미지·압축에서 strong |
| AR-DIFFUSION [arxiv](https://arxiv.org/abs/2305.09515) | 2023 | 위치별 timestep으로 AR-like behavior + 병렬 디코딩 | **연속 임베딩 기반 AR-like diffusion** | Seq2Seq NLG 전용 |

### 4.1 Diffusion-LM, DiffuSeq, SeqDiffuSeq와의 비교

1. **Diffusion-LM** [openreview](https://openreview.net/pdf?id=3s9IrEsjLyk)

   - 목적: controllable generation(구문·스타일·속성 제어)  
   - 구조: continuous diffusion + classifier-guided latent 조정  
   - 한계: **unconditional/단문 중심**, Seq2Seq 태스크에는 직접 적용이 어렵고 속도가 느림.

   **AR-DIFFUSION과의 차이**

   - AR-DIFFUSION은 **conditional Seq2Seq**에 최적화, encoder-decoder 구조. [arxiv](https://arxiv.org/pdf/2305.09515.pdf)
   - AR-like 시간 스케줄링을 통해 **step 수를 극단적으로 줄여도** 강한 성능 유지.

2. **DiffuSeq** [semanticscholar](https://www.semanticscholar.org/paper/DiffuSeq:-Sequence-to-Sequence-Text-Generation-with-Gong-Li/69144d537f90f214d5b07a7c79121d16afd7da16)

   - Seq2Seq 전용 continuous diffusion, encoder-only로 입력과 출력을 합쳐 모델링.  
   - MBR decoding + 여러 후보로 다양한 출력 확보. [semanticscholar](https://www.semanticscholar.org/paper/DiffuSeq:-Sequence-to-Sequence-Text-Generation-with-Gong-Li/69144d537f90f214d5b07a7c79121d16afd7da16)
   - 그러나, 모든 토큰에 **동일 timestep**을 사용해 NAR 특성 유지, 속도·일반화 trade-off 존재.

   **AR-DIFFUSION과의 차이**

   - **토큰별 timestep f(n,t)**를 도입해 AR-like 의존성 부여.  
   - skipping 덕분에 **NFE 면에서 100× 이상 효율적**. [aclanthology](https://aclanthology.org/2024.naacl-long.2.pdf)

3. **SeqDiffuSeq** [arxiv](https://arxiv.org/pdf/2212.10325v4.pdf)

   - encoder-decoder 구조 + self-conditioning + adaptive noise schedule.  
   - DiffuSeq보다 품질·속도 모두 향상, Seq2Seq에서 사실상 표준 continuous diffusion로 자리잡음. [semanticscholar](https://www.semanticscholar.org/paper/SeqDiffuSeq:-Text-Diffusion-with-Encoder-Decoder-Yuan-Yuan/a1186d7d9a9ef258c76afef1177e4f348061a537)
   - adaptive noise schedule은 timestep별 난이도 균등화·위치별 차별화에 초점.

   **AR-DIFFUSION과의 차별점**

   - SeqDiffuSeq는 여전히 **동시 디노이징(NAR)**;  
     adaptive schedule은 있지만, AR같은 **“좌→우 생성 순서”**는 명시적으로 구현하지 않는다.
   - AR-DIFFUSION은  
     - **(N+T) 2D 격자 상의 직선 f(n,t)**로 명시적인 좌→우 progression을 설계했고,  
     - skipping과 결합해 **3 step 수준에서도** strong 성능을 보여준다. [aclanthology](https://aclanthology.org/2024.naacl-long.2.pdf)

### 4.2 Pre-trained diffusion LM: GENIE, DiffusionBERT, SSD-LM

1. **GENIE** [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2212.11685)

   - 연속 paragraph denoise pretraining으로 대규모 코퍼스에서 diffusion decoder를 학습.  
   - GLGE·XSum·CNN/DM·CommonGen 등에서 AR Transformer와 동급 성능 + 높은 다양성. [arxiv](https://arxiv.org/pdf/2212.11685.pdf)
   - 그러나 디코딩 step 수, candidate 수가 크며, AR-like 순차성은 없다.

   **AR-DIFFUSION과의 연결**

   - 코드·구현 레벨에서 AR-DIFFUSION은 GENIE 구현을 상당 부분 재사용한다. [github](https://github.com/microsoft/ProphetNet/blob/master/AR-diffusion/README.md)
   - AR-DIFFUSION은 GENIE-style pretraining 없이도 downsteam에서 **GENIE보다 좋은 품질·속도 trade-off**를 달성했으며,  
     향후 GENIE-style pretraining과 결합하면 **일반화·표현력**이 더 강화될 여지가 크다.

2. **DiffusionBERT**, **SSD-LM** [arxiv](https://arxiv.org/abs/2211.15029)

   - DiffusionBERT는 BERT 기반 discrete diffusion를 통해 **unconditional text** 품질을 향상. [aclanthology](https://aclanthology.org/2023.acl-long.248.pdf)
   - SSD-LM은 simplex 공간 diffusion + semi-autoregressive 구조로, flexible length·modularity를 제공. [arxiv](https://arxiv.org/pdf/2210.17432.pdf)

   **함의**

   - 최근 diffusion LM들은 AR 구조를 부분적으로 다시 도입(SSD-LM 등)하거나,  
     PLM과 결합(DiffusionBERT, Latent Diffusion for Language)하는 방향으로 진화 중이다. [arxiv](https://arxiv.org/pdf/2212.09462.pdf)
   - AR-DIFFUSION은 이러한 흐름 가운데, **“연속 임베딩 + AR-like timestep 스케줄”**이라는 독특한 설계 포인트를 제시한다.

### 4.3 ARDMs, TimeGrad 등 AR×Diffusion 하이브리드

1. **Autoregressive Diffusion Models (ARDMs)** [arxiv](https://arxiv.org/abs/2110.02037)

   - discrete diffusion와 order-agnostic AR을 통합해,  
     - training 시에는 diffusion-style objective  
     - sampling 시에는 **임의의 순서/병렬화된 AR 디코딩**을 수행. [openreview](https://openreview.net/pdf/1e4e9d350d86450e306f14f662b9cdf718fce184.pdf)
   - 이미지·텍스트·압축 등에서 discrete diffusion보다 적은 step으로 동일 성능을 달성. [semanticscholar](https://www.semanticscholar.org/paper/Autoregressive-Diffusion-Models-Hoogeboom-Gritsenko/599bc7cfe98c2b57ddbe111412203a636da57be0)

2. **TimeGrad** [arxiv](https://arxiv.org/abs/2101.12072v1)

   - multivariate time-series에서 RNN hidden state + diffusion으로  
     각 시점의 분포를 autoregressively 예측하는 방법. [proceedings.mlr](http://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf)

**AR-DIFFUSION과의 관계**

- ARDMs/TimeGrad는 **“시간축 AR + diffusion”**을 결합했고,  
- AR-DIFFUSION은 **“문장 내 위치축 AR-like + diffusion”**을 결합했다는 점에서 개념적으로 유사하다.
- 다만, ARDMs는 discrete state-space(흡수 상태 등) 위에서 동작하는 반면,  
  AR-DIFFUSION은 Diffusion-LM 계열의 **연속 임베딩 공간**에 기반하여,  
  rounding을 통해 discrete 토큰으로 매핑한다는 차이가 있다. [arxiv](https://arxiv.org/pdf/2211.15089.pdf)

***

## 5. 향후 연구에 미치는 영향과 연구 시 고려할 점

### 5.1 향후 연구에 대한 긍정적 영향

1. **“시간 스케줄링 설계”를 새로운 설계 축으로 부각**

   - 기존 diffusion 텍스트 모델은 noise schedule(β 스케줄, discrete vs continuous 등)에만 주로 초점을 맞췄다면,  
   - AR-DIFFUSION은 **“timestep을 위치·세그먼트별로 설계하는 것 자체가 모델 구조 설계”**가 될 수 있음을 보여줌. [alphaxiv](https://www.alphaxiv.org/es/overview/2212.10325v5)
   - 이후 Segment-Level Diffusion이나, 위치·세그먼트별 timestep 조절을 통해 **장문·문단 구조**를 더 잘 모델링하려는 시도들과 자연스럽게 연결된다. [arxiv](http://arxiv.org/pdf/2412.11333.pdf)

2. **AR vs diffusion 패러다임 사이의 새로운 중간 지점 제공**

   - ARDMs, SSD-LM, AR-DIFFUSION 등은 [openreview](https://openreview.net/pdf?id=Lm8T39vLDTE)
     서로 다른 방식으로 AR과 diffusion을 결합하고 있다.
   - AR-DIFFUSION은 Seq2Seq NLG에서 **“거의 AR 품질 + 거의 NAR 속도”**라는  
     새로운 trade-off 지점을 실험적으로 제시했고,  
     이는 LLM 가속화, 병렬 디코딩, 서버 비용 절감 등 실용적 측면에서 매력적인 방향이다.

3. **Few-step diffusion / data-limited regime에서의 가능성**

   - 극저 step (2–3 step)에서도 GENIE 2000 step 수준의 성능을 내는 결과는, [arxiv](https://arxiv.org/html/2305.09515v3)
     “**few-step diffusion**” 연구(예: DDIM, implicit models)와 결합될 때 큰 잠재력을 가진다. [arxiv](https://arxiv.org/pdf/2211.15089.pdf)
   - 또한 data-constrained setting에서 diffusion이 AR보다 나은 스케일링을 보인다는 최근 결과와 결합하면, [arxiv](https://arxiv.org/html/2507.15857v1)
     **소규모 도메인 데이터·전문화된 도메인에서 diffusion-AR 하이브리드의 장점**이 부각될 수 있다.

### 5.2 향후 연구 시 고려해야 할 구체적 포인트

1. **대규모 pretraining과 AR-DIFFUSION의 결합**

   - GENIE, Latent Diffusion for Language, DiffusionBERT 등은 diffusion LM도 PLM처럼 **광범위 pretraining**이 가능함을 보였다. [semanticscholar](https://www.semanticscholar.org/paper/Text-Generation-with-Diffusion-Language-Models:-A-Lin-Gong/cb648d482dbd1e6ad0b0f4da43aca71c06538d4f)
   - AR-DIFFUSION도  
     - 대규모 코퍼스에서 **AR-like diffusion decoder pretraining** 후  
     - downsteam fine-tuning 구조로 확장하면,  
       일반화 능력과 zero/few-shot 성능을 크게 높일 수 있을 것이다.

2. **더 효율적인 후보 선택·MBR 대체**

   - 현재 AR-DIFFUSION의 주된 실용적 한계는 **큰 k에 의존하는 MBR**이다. [arxiv](https://arxiv.org/pdf/2305.09515.pdf)
   - 가능성:
     - latent space 내에서의 **다양성-aware pruning**,  
     - learned scoring model을 활용한 **one-pass reranking**,  
     - diffusion step 중간에서의 **adaptive early stopping** 등.
   - 특히, LLM 시대에는 학습된 **reward model / 평가기**와 결합한 **RLHF-style diffusion decoding**도 유망하다.

3. **장문·문서 레벨 구조로의 확장**

   - 현재 결과는 문장·단락 단위에 주로 초점. 향후에는:
     - 문서 레벨에서 **문단별 segment diffusion**,  
     - discourse marker, section 구조를 timestep 스케줄에 반영,  
     - long-context benchmark(GLGE-hard, LOT 등)에서의 평가가 필요하다. [aclanthology](https://aclanthology.org/2021.findings-acl.36.pdf)

4. **이론적 분석: 일반화·step 효율·노이즈 스케줄**

   - AR-DIFFUSION의 동역학이 왜 few-step regime에서 강건한지,  
     - ELBO 관점,  
     - score matching 관점,  
     - information flow(왼→오른) 관점에서의 이론적 분석이 앞으로의 과제다. [arxiv](https://arxiv.org/pdf/2303.06574.pdf)
   - 특히, **위치별 timestep f(n,t)**가 gradient signal의 분산·bias에 어떤 영향을 미치는지 정량적 분석이 필요하다.

5. **다른 modality·멀티모달로의 확장**

   - 이미 AR-Diffusion 개념은 video, motion, 3D 등으로 확장되고 있다. [arxiv](https://arxiv.org/html/2503.07418v1)
   - 텍스트에서는  
     - text-to-motion,  
     - text+image 조건 generation,  
     - text-rich image generation에서 AR-like diffusion이 구조 일관성 유지에 유리할 수 있다. [arxiv](https://arxiv.org/html/2406.12044v1)

***

## 6. 정리

- AR-DIFFUSION은 **연속 임베딩 diffusion**의 틀 안에서,  
  **위치별 timestep 함수 \(f(n,t)\)**와 **dynamic movement speed**, **skipping 기반 추론**을 도입해  
  **“AR처럼 좌→우로 정보를 흘리면서도 병렬로 디노이징”**하는 새로운 텍스트 생성 패러다임을 제시했다. [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7d866abba506e5a56335e4644ebe18f9-Abstract-Conference.html)

- 요약·번역·commonsense generation 등 다양한 Seq2Seq 태스크에서,  
  **AR Transformer와 동급/우월한 품질**과 **기존 diffusion LM 대비 100–600× 빠른 디코딩**이라는 강력한 empirical evidence를 보여주며,  
  **일반화 가능성·few-step diffusion·데이터 효율성** 측면에서도 유망한 구조임을 시사한다. [ijcai](https://www.ijcai.org/proceedings/2023/0750.pdf)

- 2020년 이후의 Diffusion-LM, DiffuSeq, SeqDiffuSeq, GENIE, DINOISER, CDCD, ARDMs 등과 비교해 볼 때,  
  AR-DIFFUSION은 **“시간 스케줄링을 통한 AR/NAR 통합”**이라는 독자적 설계 축을 개척한 작업으로 평가할 수 있다. [dl.acm](https://dl.acm.org/doi/10.24963/ijcai.2023/750)

향후 연구에서는,  
1) 대규모 pretraining과의 결합,  
2) 후보 샘플 효율화,  
3) 장문·복잡 구조로의 확장,  
4) 이론적 일반화 분석,  
5) 멀티모달 AR-diffusion으로의 확장  
을 중점적으로 탐구함으로써, AR-DIFFUSION 계열 모델이 **AR LM과 Diffusion LM 사이의 새로운 표준 구조**로 자리잡을 수 있을 것이다.

<span style="display:none">[^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/html/2305.09515v3

[^1_2]: https://arxiv.org/abs/2205.14217

[^1_3]: https://arxiv.org/pdf/2210.08933.pdf

[^1_4]: https://www.ijcai.org/proceedings/2023/0750.pdf

[^1_5]: https://arxiv.org/abs/2305.09515

[^1_6]: https://arxiv.org/pdf/2305.09515.pdf

[^1_7]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/7d866abba506e5a56335e4644ebe18f9-Abstract-Conference.html

[^1_8]: https://aclanthology.org/2024.naacl-long.2.pdf

[^1_9]: https://arxiv.org/pdf/2302.10025.pdf

[^1_10]: https://www.alphaxiv.org/es/overview/2212.10325v5

[^1_11]: https://aclanthology.org/2024.naacl-long.2/

[^1_12]: https://dl.acm.org/doi/10.5555/3618408.3619275

[^1_13]: https://openreview.net/pdf?id=jQj-_rLVXsj

[^1_14]: https://www.semanticscholar.org/paper/cb648d482dbd1e6ad0b0f4da43aca71c06538d4f

[^1_15]: https://dl.acm.org/doi/10.24963/ijcai.2023/750

[^1_16]: https://peerj.com/articles/cs-1905/

[^1_17]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10909201/

[^1_18]: https://arxiv.org/pdf/2205.14217.pdf

[^1_19]: https://openreview.net/pdf?id=3s9IrEsjLyk

[^1_20]: https://www.semanticscholar.org/paper/Diffusion-LM-Improves-Controllable-Text-Generation-Li-Thickstun/1386b8a11929cf02da291c56aca353e33bbc22ed

[^1_21]: https://arxiv.org/pdf/2212.11685.pdf

[^1_22]: https://proceedings.mlr.press/v202/lin23d/lin23d.pdf

[^1_23]: https://arxiv.org/abs/2302.10025

[^1_24]: https://aclanthology.org/2020.findings-emnlp.165.pdf

[^1_25]: https://aclanthology.org/2020.findings-emnlp.165/

[^1_26]: https://github.com/INK-USC/CommonGen

[^1_27]: https://arxiv.org/html/2507.15857v1

[^1_28]: https://arxiv.org/pdf/2212.10325v4.pdf

[^1_29]: https://arxiv.org/pdf/2212.10325v1.pdf

[^1_30]: https://www.semanticscholar.org/paper/SeqDiffuSeq:-Text-Diffusion-with-Encoder-Decoder-Yuan-Yuan/a1186d7d9a9ef258c76afef1177e4f348061a537

[^1_31]: https://arxiv.org/abs/2212.10325

[^1_32]: https://www.labxing.com/files/lab_data/1743-1687083938-i2FXe59J.pdf

[^1_33]: https://www.semanticscholar.org/paper/020a50f6a7154850ac81e3cde69ad8198ded6751

[^1_34]: https://arxiv.org/pdf/2211.15089.pdf

[^1_35]: https://www.semanticscholar.org/paper/Continuous-diffusion-for-categorical-data-Dieleman-Sartran/22775e58932cdfbd273a2a835a22c5d86800a458

[^1_36]: http://arxiv.org/abs/2211.15089v1

[^1_37]: https://www.semanticscholar.org/paper/DiffuSeq:-Sequence-to-Sequence-Text-Generation-with-Gong-Li/69144d537f90f214d5b07a7c79121d16afd7da16

[^1_38]: http://arxiv.org/abs/2210.08933

[^1_39]: https://ar5iv.labs.arxiv.org/html/2212.11685

[^1_40]: https://www.semanticscholar.org/paper/Text-Generation-with-Diffusion-Language-Models:-A-Lin-Gong/cb648d482dbd1e6ad0b0f4da43aca71c06538d4f

[^1_41]: https://arxiv.org/abs/2212.11685

[^1_42]: https://arxiv.org/pdf/2305.14671.pdf

[^1_43]: https://fugumt.com/fugumt/paper_check/2211.15089v1_enmode

[^1_44]: https://arxiv.org/abs/2110.02037

[^1_45]: https://openreview.net/pdf/1e4e9d350d86450e306f14f662b9cdf718fce184.pdf

[^1_46]: https://www.semanticscholar.org/paper/Autoregressive-Diffusion-Models-Hoogeboom-Gritsenko/599bc7cfe98c2b57ddbe111412203a636da57be0

[^1_47]: https://openreview.net/pdf?id=Lm8T39vLDTE

[^1_48]: http://arxiv.org/abs/2110.02037

[^1_49]: https://github.com/microsoft/ProphetNet/blob/master/AR-diffusion/README.md

[^1_50]: https://openreview.net/pdf?id=2UFmCXQ6zn

[^1_51]: https://github.com/microsoft/ProphetNet

[^1_52]: https://github.com/microsoft/ProphetNet/blob/master/AR-diffusion/eval_utils/README.md

[^1_53]: https://arxiv.org/abs/2211.15029

[^1_54]: https://arxiv.org/pdf/2210.17432.pdf

[^1_55]: https://aclanthology.org/2023.acl-long.248.pdf

[^1_56]: https://arxiv.org/pdf/2212.09462.pdf

[^1_57]: https://arxiv.org/abs/2101.12072v1

[^1_58]: https://www.emergentmind.com/papers/2101.12072

[^1_59]: http://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf

[^1_60]: https://arxiv.org/pdf/2101.12072.pdf

[^1_61]: https://arxiv.org/pdf/2107.03006.pdf

[^1_62]: http://arxiv.org/pdf/2412.11333.pdf

[^1_63]: https://aclanthology.org/2021.findings-acl.36.pdf

[^1_64]: https://microsoft.github.io/glge/

[^1_65]: https://www.semanticscholar.org/paper/GLGE:-A-New-General-Language-Generation-Evaluation-Liu-Yan/5fe78eb0f142902237df11cb67c455787a759172

[^1_66]: https://arxiv.org/pdf/2410.09527.pdf

[^1_67]: https://arxiv.org/pdf/2303.06574.pdf

[^1_68]: https://arxiv.org/html/2503.07418v1

[^1_69]: https://arxiv.org/pdf/2308.16682.pdf

[^1_70]: http://arxiv.org/pdf/2405.17405.pdf

[^1_71]: https://arxiv.org/html/2503.12929v3

[^1_72]: http://arxiv.org/pdf/2409.10847.pdf

[^1_73]: https://arxiv.org/pdf/2304.04681.pdf

[^1_74]: https://arxiv.org/pdf/2307.08849.pdf

[^1_75]: https://arxiv.org/html/2406.12044v1

[^1_76]: 2305.09515v3.pdf

[^1_77]: https://arxiv.org/abs/2508.10875

[^1_78]: https://arxiv.org/abs/2508.08712

[^1_79]: https://ieeexplore.ieee.org/document/8429065/

[^1_80]: https://iopscience.iop.org/article/10.1088/1361-6595/ae0c33

[^1_81]: https://www.semanticscholar.org/paper/f2f0ab28cb2520a18141b4de7e36b7558741edc5

[^1_82]: https://www.semanticscholar.org/paper/9a326fd93275e0c55f2cf1aeab13c3cad8d6a305

[^1_83]: https://www.semanticscholar.org/paper/a8f47dc4c898cbe5f6daa435ad53db3c7a80d0e1

[^1_84]: http://arxiv.org/pdf/2112.10752.pdf

[^1_85]: https://arxiv.org/html/2412.02099v1

[^1_86]: https://arxiv.org/abs/2305.07015

[^1_87]: https://www.youtube.com/watch?v=WwjyXDuoPtk

[^1_88]: https://github.com/microsoft/ProphetNet/issues/74
