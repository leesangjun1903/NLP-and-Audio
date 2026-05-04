# MoBA: Mixture of Block Attention for Long-Context LLMs

MoBA는 Transformer의 어텐션을 **블록 단위 동적 희소 어텐션**으로 일반화한 구조로, 풀 어텐션과 동급의 언어모델·추론 성능을 유지하면서도 1M~10M 토큰 수준의 장문맥에서 연산량을 크게 줄이고, 기존 Transformer와의 호환성을 유지하는 것을 핵심 주장으로 한다. 이 과정에서 MoE 스타일의 게이팅을 어텐션에 적용해 “less structure(사전 정의 sparsity 패턴 최소화)” 원칙을 따르며, MoBA–Full 하이브리드 및 layer‑wise 하이브리드를 통해 학습 효율과 일반화 성능을 동시에 확보하는 점이 주요 기여이다.[^1][^2][^3][^4]

***

## 1. 핵심 주장과 주요 기여 (요약)

- **Mixture-of-Experts(MoE) 기반 블록 어텐션**
전체 시퀀스를 길이 $N$의 $n$개 블록으로 나누고, 각 쿼리 토큰이 top‑k 블록만 선택해 어텐션하도록 하는 **Mixture of Block Attention(MoBA)**를 제안한다.[^2][^1]
- **풀 어텐션과 파라미터 수가 동일한 drop‑in 대체물**
게이팅은 쿼리·키의 통계량만으로 정의되는 **무(無)파라미터 top‑k 게이트**이므로, MoBA로 교체해도 모델 파라미터 수가 변하지 않고 기존 Transformer를 재활용할 수 있다.[^5][^1]
- **장문맥 스케일링에서 풀 어텐션과 거의 동급의 성능**
568M~2.1B 모델에서 LM loss 스케일링 곡선이 풀 어텐션과 $10^{-3}$ 수준 차이에 머무르고, Llama‑8B‑1M‑MoBA는 AGIEval, GSM8K, LongBench, RULER 등에서 Llama‑8B‑1M‑Full과 사실상 동급 성능을 보인다.[^1]
- **효율성: 1M~10M 토큰까지 서브‑쿼드릭 스케일링**
1M 토큰 프리필에서 최대 약 6.5×, 10M 토큰에서 FlashAttention 대비 약 16× 속도 향상을 보고하며, 실 서비스(Kimi) 장문맥 요청에 이미 적용되었다.[^5][^1]
- **하이브리드 학습 전략**
(1) pre‑training 단계에서 MoBA로 90% 토큰을 학습 후 풀 어텐션으로 전환, (2) SFT에서 상위 몇 개 레이어만 풀 어텐션으로 남기는 layer‑wise hybrid로, 효율·성능 트레이드오프를 실제 학습 파이프라인 수준에서 제시한다.[^1]

***

## 2. MoBA가 해결하려는 문제

### 2.1 기존 장문맥 어텐션의 구조적·계산적 한계

- 표준 self‑attention은 시퀀스 길이 $N$에서 연산·메모리가 $\mathcal{O}(N^2)$라 수십만~수백만 토큰으로의 스케일업이 사실상 불가능하다.[^2][^1]
- Longformer, BigBird, LongT5, LongNet 등 정적 희소 어텐션은 윈도우·글로벌·랜덤·블록 패턴을 **고정**해 두어 작업에 특화되기 쉽고, 다양한 도메인·태스크에 대한 일반화 유연성이 낮다.[^6][^1]
- Quest, Minference, RetrievalAttention 등 동적 희소 어텐션은 주로 **추론(pre‑fill) 단계 가속**에 초점을 두고 있어, 장문맥 pre‑training 자체의 계산량을 충분히 줄이지 못한다.[^7][^1]
- Mamba, RWKV, RetNet 같은 선형/SSM 계열은 구조가 Transformer와 크게 달라 기존 모델을 재사용하기 어렵고, 복잡한 reasoning 과제에서의 성능은 아직 제한적으로만 검증되었다.[^3][^1]


### 2.2 논문이 제기하는 핵심 질문

MoBA 논문은 다음과 같은 연구 질문을 명시적으로 제기한다.[^1]

1. 기존 Transformer 프레임워크(소프트맥스 어텐션, KV‑cache 등)를 유지하면서,
2. 슬라이딩 윈도우나 sink 같은 강한 정적 sparsity 패턴 없이(“less structure”),
3. 학습·추론 모두에서 효율적인 **블록 단위 동적 희소 어텐션**을 도입하고,
4. 풀 어텐션과 동급의 스케일링·다운스트림 및 장문맥 reasoning 성능을 유지할 수 있는가?

MoBA는 이를 위해 **블록 분할 + MoE식 top‑k 라우팅 + FlashAttention 기반 구현**을 결합한 아키텍처를 제안한다.[^3][^1]

***

## 3. 제안 방법: MoBA 메커니즘과 수식

### 3.1 표준 어텐션 복습

단일 헤드 어텐션에서 쿼리 $q \in \mathbb{R}^{1 \times d}$, 키·값 $K,V \in \mathbb{R}^{N \times d}$에 대해:[^1]

$$
\text{Attn}(q, K, V) = \text{Softmax}\left(\frac{q K^\top}{\sqrt{d}}\right) V. \quad (1)
$$

여기서 모든 토큰 쌍에 점수 계산이 필요하므로 계산 복잡도는 $\mathcal{O}(N^2)$이다.[^1]

### 3.2 블록 분할과 블록 단위 어텐션

MoBA는 시퀀스를 길이 $N$에서 $n$개의 블록으로 균등 분할하고, 블록 크기를 $B = N / n$이라 한다.[^1]

$$
I_i = [ (i-1)B + 1,\; iB ]\quad (i = 1,\dots,n). \quad (3)
$$

각 쿼리 토큰은 전체 토큰이 아니라, 선택된 블록들의 합집합 $I \subseteq [N]$에 대해서만 어텐션을 수행한다.[^1]

$$
\text{MoBA}(q, K, V) = \text{Softmax}\left( \frac{q K[I]^\top}{\sqrt{d}} \right) V[I]. \quad (2)
$$

즉, **토큰 단위가 아닌 블록 단위로 sparsity를 부여**해 하드웨어 친화적인 block‑sparse 어텐션을 구현한다.[^5][^1]

### 3.3 MoE 스타일 top‑k 게이팅

각 블록 $I_i$에 대한 affinity score $s_i$를 정의하고, top‑k 게이팅으로 활성화할 블록을 선택한다.[^1]

1. 블록 $i$의 대표 키 벡터는 $K[I_i]$의 mean pooling으로 정의한다.

$$
\bar{k}_i = \text{mean pool}(K[I_i]).
$$

2. 쿼리–블록 affinity는 내적으로 계산한다.

$$
s_i = \langle q, \bar{k}_i \rangle. \quad (6)
$$

3. top‑k 게이트는 다음과 같이 정의된다.

$$
g_i = 
\begin{cases}
1, & s_i \in \text{Topk}(\{s_j\}_{j=1}^n, k), \\
0, & \text{otherwise.}
\end{cases} \quad (5)
$$

4. 최종적으로 선택되는 토큰 인덱스 집합은 활성화된 블록의 합집합이다.

$$
I = \bigcup_{g_i > 0} I_i. \quad (4)
$$

중요한 점은 이 게이트가 **추가 학습 파라미터 없이** $q,K$ 텐서에서 직접 계산된다는 것으로, MoE의 “라우터” 역할을 파라미터 없이 수행한다는 점이 MoBA의 큰 특징이다.[^3][^5]

### 3.4 인과성(autoregressive) 보존

언어모델의 autoregressive 특성을 유지하기 위해 두 가지 설계를 도입한다.[^1]

1. **미래 블록 차단**
쿼리 위치를 $\text{pos}(q)$라 할 때, 쿼리 이후에 시작하는 블록(예: $iB > \text{pos}(q)$ )에 대해서는

$$
s_i = -\infty,\quad g_i = 0
$$

으로 설정해 게이트가 미래 블록을 선택하지 못하게 한다.
2. **현재 블록 강제 및 causal masking**
각 쿼리는 항상 자신이 속한 현재 블록에 라우팅되도록 해당 블록의 $g_i = 1$로 강제하고, 그 블록 내에서는 standard causal mask를 사용해 이후 토큰의 정보가 누출되지 않도록 한다.[^1]

이때 “현재 블록”은 MoE 관점에서 **shared expert**에 해당하며, 라우터가 과거 블록만 선택하는 극단으로 치우치는 것을 방지하고 로컬 문맥을 항상 활용할 수 있게 한다.[^1]

### 3.5 구현: FlashAttention + MoE 스타일 루팅

논문은 Algorithm 1 형태로 고성능 구현을 제시하는데, 요지는 다음과 같다.[^1]

1. $K,V$를 길이 $B$의 블록 $\{\tilde{K}_i,\tilde{V}_i\}$로 분할.
2. 블록 평균을 취해 $\bar{K} \in \mathbb{R}^{n \times h \times d}$를 만들고, $S = Q \bar{K}^\top \in \mathbb{R}^{N \times h \times n}$ 계산.
3. causal mask를 포함한 top‑k 연산으로 쿼리–블록 매핑 행렬 $G$를 생성.
4. (a) 현재 블록에 대한 self‑attention, (b) 선택된 과거 블록들에 대한 MoBA 어텐션을 각각 **FlashAttention varlen 커널**로 계산.
5. online softmax(tiling)를 이용해 두 결과를 결합해 최종 출력을 구성.

이를 통해 MoBA는 실제 구현에서 **FlashAttention 수준의 효율성과 MoE 수준의 dynamic routing**을 함께 달성하며, 실험적으로 서브‑쿼드릭에 가까운 연산 스케일링을 보인다.[^5][^1]

***

## 4. 모델 구조와 학습·하이브리드 전략

### 4.1 MoBA를 포함한 Transformer 구조

- Transformer 블록 내 **어텐션 서브레이어만** MoBA 또는 full attention으로 교체 가능한 형태로 두고, FFN·LayerNorm 등 나머지 구조는 동일하게 유지한다.[^1]
- MoBA는 full attention을 그대로 대체하지만 파라미터 수는 변하지 않아, 같은 모델 설정에서 **attention 종류만 바꾼 공정한 비교**가 가능하다.[^3][^1]


### 4.2 스케일링 실험 설정

- 568M, 822M, 1.1B, 1.5B, 2.1B 등 5개 모델을 시퀀스 길이 8K, Chinchilla 스케일링 레짐(토큰 수 충분)을 따르며, MoBA vs 풀 어텐션을 비교한다.[^1]
- MoBA는 블록 크기 512, top‑k=3으로 설정되어, 8K에서는 대략 $1 - \frac{512 \times 3}{8192} \approx 81.25\%$ 희소도를 갖는다.[^1]


### 4.3 MoBA–Full Attention 하이브리드 (pre‑training)

- 1.5B, 32K 컨텍스트 모델에서 세 가지 레시피를 비교한다.[^1]
    - Full: 전체 토큰을 full attention으로 학습.
    - MoBA: 전체 토큰을 MoBA로 학습.
    - MoBA/Full hybrid: 전체 토큰의 90%는 MoBA로, 마지막 10%는 full attention으로 학습.
- position‑wise LM loss를 보면, MoBA only는 trailing 토큰에서 loss가 다소 높지만, MoBA/Full hybrid는 **풀 어텐션과 거의 동일한 손실 곡선**을 보이며, MoBA→Full 전환 시 loss spike가 나타나지 않는다.[^1]


### 4.4 Layer‑wise 하이브리드 (SFT용)

- SFT에서는 prompt 토큰을 loss 계산에서 마스킹하는 경우가 많아, 희소 어텐션 구조(특히 MoBA)에서는 gradient가 전체 컨텍스트로 충분히 퍼지지 못해 성능 저하가 발생할 수 있다는 관찰이 있다.[^1]
- 이를 완화하기 위해 상위 $L_\text{full}$개 레이어는 full attention, 나머지는 MoBA를 사용하는 layer‑wise hybrid를 제안한다.[^1]
- 실험적으로 $L_\text{full}$을 늘릴수록 SFT LM loss와 trailing LM loss가 감소하며, **순수 MoBA 대비 상당한 성능 회복**을 얻을 수 있다.[^1]


### 4.5 1M‑context Llama 3.1 8B MoBA 모델

- Llama 3.1 8B base를 128K까지 길이 확장한 후, positional interpolation을 사용해 256K, 512K, 1M까지 지속적 pre‑training을 수행한다.[^2][^1]
- 1M 컨텍스트에서 MoBA는 블록 크기 4096, top‑k=12로 설정되어 약 $1 - \frac{4096 \times 12}{10^6} \approx 95.31\%$ 희소도를 갖는다.[^1]
- 레이어 구조는 32층 중 마지막 3층은 full attention, 나머지 29층은 MoBA로 구성하는 layer‑wise hybrid를 사용한다.[^1]

***

## 5. 성능 향상 및 한계

### 5.1 스케일링 법칙 관점 성능

- 568M~2.1B 모델에서 validation LM loss 스케일링 곡선을 피팅한 결과, MoBA와 full attention의 loss 차이는 전체 범위에서 **$10^{-3}$ 수준**에 머무르며, scaling exponent 또한 거의 동일하게 추정된다.[^1]
- 32K 시퀀스에서 trailing token(마지막 2K 토큰)의 LM loss를 사용한 long‑context scaling에서도, MoBA는 full보다 약간 높은 loss를 보이지만 모델이 커질수록 두 곡선의 차이가 줄어든다.[^1]
- 이는 **장문맥에서도 MoBA가 full과 유사한 스케일링 거동**을 보여, 모델 크기를 키울수록 성능 격차가 줄어드는 방향으로 일반화함을 시사한다.[^1]


### 5.2 다운스트림 벤치마크

Llama‑8B‑1M‑MoBA vs Llama‑8B‑1M‑Full 결과(일부):[^1]


| Benchmark | MoBA | Full |
| :-- | :-- | :-- |
| AGIEval (0‑shot) | 0.5144 | 0.5146 |
| BBH (3‑shot) | 0.6573 | 0.6589 |
| CEval (5‑shot) | 0.6273 | 0.6165 |
| GSM8K (5‑shot) | 0.7278 | 0.7142 |
| LongBench @32K (0‑shot) | 0.4828 | 0.4821 |
| RULER @128K (0‑shot) | 0.7818 | 0.7849 |

- 대부분의 태스크에서 두 모델의 성능 차이는 $10^{-3}$ ~ $10^{-2}$ 수준이며, CEval, GSM8K, MBPP 등 일부에서는 MoBA가 오히려 소폭 우수하다.[^1]
- 특히 LongBench(32K), RULER(128K) 같은 장문맥 벤치마크에서도 MoBA는 같은 희소도 조건에서 full과 거의 동급 수준을 유지한다.[^1]


### 5.3 Needle‑in‑a‑Haystack 및 길이 일반화

- 32K~1M 컨텍스트 길이에 대해 Needle‑in‑a‑Haystack 평가를 수행한 결과, MoBA 기반 Llama‑8B‑1M 모델은 1M까지 **안정적인 needle retrieval 성능**을 보이며, 길이가 늘어나도 급격한 성능 저하는 관찰되지 않는다.[^1]
- 이 결과는 MoBA가 단지 학습 시 사용된 길이(예: 32K) 근처에서만 동작하는 것이 아니라, **길이 extrapolation 측면에서도 의미 있는 일반화 능력**을 갖고 있음을 시사한다.[^8][^1]


### 5.4 효율성과 스케일 업

- Llama‑8B‑1M‑MoBA와 full 모델의 attention layer forward time 비교에서, MoBA는 8K~1M 범위에서 항상 더 빠르고, 1M 토큰 프리필에서 최대 약 6.5× 속도 향상을 보여 준다.[^1]
- 블록 수를 고정한 채 block size만 늘려 10M 길이까지 확장했을 때, MoBA는 FlashAttention 대비 최대 16× 연산 시간 감소를 기록하며, query head 단위 tensor parallelism으로 GPU 메모리 제약을 완화한다.[^1]


### 5.5 한계 및 제약

- **SFT에서 순수 MoBA의 optimization 어려움**
SFT에서는 prompt 토큰을 loss에서 마스킹하기 때문에, 희소 어텐션에서는 gradient가 전체 컨텍스트로 충분히 전파되지 못해 성능 열화가 나타날 수 있으며, 논문은 이를 layer‑wise hybrid로 완화하지만 fully‑MoBA SFT는 여전히 도전적임을 시사한다.[^1]
- **Drop‑in이 아닌 “continual training 필요” 구조**
저자들은 MoBA가 기존 LLM에 바로 끼워넣어 사용할 수 있는 drop‑in sparse attention이 아니라, **MoBA로 전환 후 추가 continual pre‑training이 필요하다**고 명시한다.[^5]
- **게이팅 품질에 대한 의존성**
평균 pooling 기반 블록 표현과 단순 내적 기반 게이팅이 항상 최선은 아니며, 후속 연구에서는 이 부분을 개선해 routing SNR을 높이는 방향이 제안된다.[^9][^10]

***

## 6. 일반화 성능 향상 가능성에 대한 분석

### 6.1 “Less structure”와 표현력

- MoBA는 슬라이딩 윈도우나 attention sink처럼 정해진 패턴을 강제하지 않고, 라우터가 데이터에 따라 어떤 블록에 집중할지 학습하도록 한다는 점에서 **표현력이 더 크고 task‑agnostic**한 sparse 어텐션 구조이다.[^4][^1]
- 논문은 슬라이딩 윈도우/attention sink가 MoBA의 특수한 게이팅 패턴으로 포함될 수 있음을 보이며, MoBA가 이들 정적 구조를 상위 개념으로 일반화한다고 주장한다.[^1]
- 이는 다양한 태스크·도메인에서 최적 sparsity 패턴이 달라질 때, MoBA가 **데이터 기반으로 sparsity를 학습**함으로써 더 나은 일반화 성능을 낼 수 있는 잠재력을 갖는다는 의미로 해석할 수 있다.[^1]


### 6.2 스케일링 법칙과 위치별 손실

- 여러 모델 크기에 대해 위치별(position‑wise) LM loss scaling law를 분석한 결과, 0~32K 위치 구간 전체에서 MoBA와 full attention의 scaling exponent가 거의 동일하며, trailing 위치에서도 모델 크기가 커질수록 두 곡선이 수렴하는 경향을 보인다.[^1]
- 이는 **모델 크기를 키우면 키울수록 MoBA가 full 어텐션과 동급의 장문맥 언어모델링 능력으로 일반화**할 수 있음을 뒷받침하며, sparsity 자체가 근본적인 성능 한계를 야기하지 않음을 시사한다.[^1]


### 6.3 길이와 태스크 측면 일반화

- LongBench, RULER, Needle‑in‑a‑Haystack 등 학습 시점의 길이보다 긴 시퀀스와 다른 분포를 가진 벤치마크에서 MoBA가 성능을 잘 유지하는 것은, **길이 extrapolation 측면에서도 구조적 편향이 강한 정적 sparsity보다 유리**할 수 있음을 보여 준다.[^6][^1]
- 긴 컨텍스트에서 LLM이 “알고도 말하지 못하는(know but don’t tell)” 현상—중간/후반부 정보 활용 실패—를 보인다는 최근 분석과 비교하면, MoBA의 block‑level routing은 해당 정보를 더 명시적으로 검색·집중하도록 학습되기 때문에, **정보 인코딩과 활용 간의 괴리를 줄이는 방향**으로 작용할 가능성이 있다.[^8]


### 6.4 후속 연구가 보여주는 일반화 잠재력

- **Optimizing Mixture of Block Attention**은 MoBA의 routing 정확도를 SNR(signal‑to‑noise ratio) 관점에서 이론적으로 분석하고, 블록 크기·key aggregation 방식이 성능에 미치는 영향을 정량화하며, 작은 블록과 convolutional key aggregation으로 일반화를 더 향상시킬 수 있음을 보인다.[^10][^9]
- **SpikingBrain2.0**은 MoBA 스타일 sparse softmax attention과 sparse linear attention을 결합한 dual‑space sparse attention으로, 장문맥 효율–성능 균형을 개선하는 뇌 영감 아키텍처를 제안한다.[^11]
- 최근 long‑context efficient attention 서베이와 HISA, HySparse, SeerAttention 등은 MoBA를 대표적인 block‑sparse, trainable long‑context attention으로 인용하며, self‑distillation, 계층적 인덱싱, oracle token selection 등과 결합해 더욱 강력한 일반화 성능을 목표로 한다.[^12][^13][^7][^6]

이 흐름은 MoBA 계열 구조가 **다양한 도메인·모달리티·학습 패러다임에서 reusable한 building block**으로 기능하며, 장문맥 LLM의 일반화 능력 확장에 핵심 역할을 할 수 있음을 보여 준다.

***

## 7. 2020년 이후 관련 연구 비교 (요지)

아래는 2020년 이후 주요 장문맥/효율 어텐션과 MoBA의 위치를 간략히 비교한 것이다(모두 공개 접근).


| 분류 | 대표 방법 | 핵심 아이디어 | 장점 | MoBA와의 차이 |
| :-- | :-- | :-- | :-- | :-- |
| 정적 희소 | Longformer, BigBird, LongT5, LongNet 등[^6][^1] | 윈도우/글로벌/랜덤/블록 패턴을 사전에 고정 | 구현 단순, 안정적 | 패턴이 고정되어 작업·도메인 일반화 유연성이 낮음 |
| 동적 희소 | Reformer, Routing Transformer, Memorizing Transformer, Unlimiformer, CoLT5, Sparse Sinkhorn 등[^1] | LSH, K‑means, kNN, learnable router로 토큰/클러스터 선택 | 데이터 기반 sparsity, 표현력 우수 | 구조·구현 복잡, pre‑training 비용 큼 |
| 블록‑게이트 계열 | SeerAttention, HySparse, HISA 등[^13][^12][^7] | block‑level gate를 학습하거나 oracle token selection, 하드웨어 친화적 계층형 인덱싱 | 기존 LLM에 gate 파라미터만 추가해 적용 가능 | gate 파라미터가 별도로 필요, self‑distillation 등 추가 학습 필요 |
| 구조 변경형 | Performer, Linformer, Mamba, RetNet, Hyena 등[^3][^1] | 선형 어텐션·SSM·convolution 등 Transformer 어텐션 자체를 대체 | 이론적 $\mathcal{O}(N)$ 또는 sub‑quadratic, 매우 긴 시퀀스 가능 | 기존 Transformer와 상호운용성 낮고, full softmax로의 전환이 쉽지 않음 |
| 블록‑MoE형 | **MoBA**[^2][^3][^4] | MoE 게이팅을 block‑sparse softmax에 적용, 무파라미터 gate, full↔sparse 전환 가능 | 기존 Transformer와 높은 호환성, 풀과 동급 성능, 매우 긴 컨텍스트까지 확장 | 추가 continual pre‑training 필요, SFT에서 fully‑MoBA는 최적화 어려움 |
| MoBA 개선형 | Optimizing MoBA, FlashMoBA 등[^9][^10] | SNR 기반 이론 분석 + 작은 블록 + conv key + CUDA 커널 | routing 품질 향상, GPU 효율 추가 개선, 더 큰 speedup | 구현 복잡도 증가, 아직 연구·검증 진행 중 |

MoBA는 **정적 희소와 구조 변경형 사이**에서, Transformer 호환성을 유지하면서 학습 기반 sparsity와 장문맥 효율성을 동시에 제공하는 설계로 위치 지을 수 있다.[^4][^6]

***

## 8. 앞으로의 연구에 미치는 영향과 고려할 점

### 8.1 연구 패턴 측면 영향

- MoBA는 “어텐션도 MoE처럼 block‑level expert로 분리해 동적으로 선택할 수 있다”는 패턴을 정립해, 이후 block‑sparse·MoE·hybrid‑sparse 계열 연구들의 공통 reference가 되고 있다.[^13][^6]
- 또한 full 어텐션을 완전히 제거하기보다는, **학습·SFT·추론 각 단계에서 MoBA와 full을 적절히 섞는 하이브리드 전략**이 실용적으로 유효하다는 것을 경험적으로 보여 주어, 실제 시스템 설계에 큰 영향을 미친다.[^5][^1]


### 8.2 향후 연구 시 구체적인 고려사항

1. **게이팅 함수와 블록 표현의 개선**
    - 평균 pooling 기반 블록 표현 대신, convolution, attention‑based aggregation, learned projection 등을 사용하면 query–block SNR을 높이고, 장문맥 reasoning·코드·수학 등 복잡한 태스크에서의 일반화 성능을 더욱 끌어올릴 수 있다.[^9][^10]
2. **멀티모달·계층적 메모리와의 결합**
    - SpikingBrain2.0이 보여주듯, MoBA와 sparse linear attention(SSM)을 병렬로 두는 dual‑branch 구조나, vision/video에 대한 공간‑시간 블록 어텐션과 결합한 멀티모달 장문맥 모델 설계가 유망하다.[^11][^1]
3. **SFT·RLHF·R1류 장고추론 모델과의 통합**
    - DeepSeek‑R1, OpenAI o1/o3, Kimi k1.5 등 **긴 chain‑of‑thought**을 사용하는 모델에서는 입력·출력이 모두 길어지므로, MoBA 계열 구조가 KV‑cache, gradient 전파, credit assignment에 어떤 이득/문제를 가지는지 체계적인 분석이 필요하다.[^8][^1]
4. **학습 파이프라인 설계**
    - 실제로는 (1) short‑context pre‑training → (2) 길이 확장/positional interpolation → (3) MoBA 전환 및 continual pre‑training → (4) SFT·RLHF에서의 layer‑wise hybrid 튜닝 등 end‑to‑end 파이프라인 관점에서 MoBA의 역할을 최적화해야 한다.[^5][^1]
    - 데이터 길이 분포, position encoding, optimizer, MoBA hyperparameter(블록 크기, top‑k, sparsity target)를 **공동 최적화**하는 연구가 요구된다.[^6]
5. **일반화 평가 지표와 벤치마크 확장**
    - Needle‑in‑a‑Haystack, RULER, LongBench 외에도, 위치별 loss, 정보 추출 시간, “know but don’t tell” 현상 등 장문맥에서의 **정보 인코딩 vs 활용 간 괴리**를 직접 측정하는 벤치마크 설계가 중요하다.[^14][^8]
    - CAB(Comprehensive Attention Benchmark)나 최신 long‑context 서베이에서 제안하는 세분화된 벤치마크와 MoBA 계열 구조를 결합해 비교·분석하면, 구조 선택에 대한 더 정교한 가이드라인을 얻을 수 있다.[^14][^6]

***

## 9. 추가 자료 및 보고서 안내

위 내용을 바탕으로, 첨부하신 논문과 관련 오픈 액세스 논문들을 종합 분석한 보다 긴 마크다운 리포트를 함께 제공했다(대화 상단의 보고서 파일 참조). 그 안에는 MoBA의 수식·실험 세부, ablation과 후속 연구까지 더 많은 그래프·수치와 함께 정리해 두었다.

***

### 주요 참고 자료(모두 오픈 액세스)

- Enzhe Lu et al., “MoBA: Mixture of Block Attention for Long-Context LLMs”, arXiv:2502.13189 / NeurIPS 2025.[^2][^3]
- Guangxuan Xiao et al., “Optimizing Mixture of Block Attention”, arXiv:2511.11571.[^10][^9]
- SpikingBrain2.0: Brain-Inspired Foundation Models for Efficient Long-Context Modeling.[^11]
- “Efficient Attention Mechanisms for Large Language Models: A Survey”.[^6]
- HISA, HySparse, SeerAttention 등 long‑context sparse attention 관련 최근 논문들.[^12][^13][^7]
- “Insights into LLM Long-Context Failures: When Transformers Know but Don’t Tell”.[^8]

이 자료들을 함께 보시면 MoBA를 장문맥 LLM 연구의 큰 흐름 속에서 어떻게 위치 지을 수 있는지 더 잘 파악하실 수 있을 것입니다.
<span style="display:none">[^15][^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32]</span>

<div align="center">⁂</div>

[^1]: 2502.13189v1.pdf

[^2]: https://arxiv.org/abs/2502.13189

[^3]: https://openreview.net/forum?id=RlqYCpTu1P

[^4]: https://github.com/MoonshotAI/MoBA

[^5]: https://huggingface.co/papers/2502.13189

[^6]: https://arxiv.org/html/2502.13189v1

[^7]: https://openreview.net/pdf?id=Nf8yfPDFTl

[^8]: https://openreview.net/pdf?id=RlqYCpTu1P

[^9]: http://arxiv.org/pdf/2406.14673.pdf

[^10]: https://arxiv.org/abs/2511.11571

[^11]: https://arxiv.org/pdf/2511.11571.pdf

[^12]: https://arxiv.org/html/2604.22575

[^13]: https://arxiv.org/html/2507.19595v1

[^14]: https://arxiv.org/html/2603.28458v1

[^15]: https://arxiv.org/html/2602.03560v1

[^16]: http://arxiv.org/pdf/2210.07661.pdf

[^17]: https://arxiv.org/pdf/2502.13189.pdf

[^18]: https://arxiv.org/pdf/2410.01485.pdf

[^19]: https://aclanthology.org/2023.findings-emnlp.370.pdf

[^20]: http://arxiv.org/pdf/2405.17025.pdf

[^21]: https://arxiv.org/pdf/2404.09173.pdf

[^22]: https://arxiv.org/html/2512.12087v2

[^23]: https://arxiv.org/html/2504.16795v2

[^24]: https://arxiv.org/html/2511.11571v1

[^25]: https://arxiv.org/html/2406.14909v3

[^26]: https://arxiv.org/pdf/2409.15355.pdf

[^27]: https://arxiv.org/html/2509.24745v1

[^28]: https://www.youtube.com/watch?v=ZV2v2pzYW5A

[^29]: https://digitalbourgeois.tistory.com/2267

[^30]: https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling

[^31]: https://neurips.cc/virtual/2025/poster/117997

[^32]: https://kimjy99.github.io/논문리뷰/moba/

