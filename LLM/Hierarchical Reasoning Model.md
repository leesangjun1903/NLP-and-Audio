# Hierarchical Reasoning Model

이 논문은 두 계층의 순환 모듈(H/L)과 근사 그래디언트 학습, 적응적 연산 시간(ACT)을 결합한 새로운 “Hierarchical Reasoning Model(HRM)”을 제안해, 매우 적은 데이터(각 태스크 약 1,000 샘플)와 2,700만 파라미터만으로 ARC-AGI, 고난도 스도쿠, 대형 미로 등 복잡한 추론 문제에서 거대 CoT LLM들을 능가하는 것을 보이는 것이 핵심 주장입니다. 특히 HRM은 언어적 체인‑오브‑쏘트 없이 잠재 상태에서 깊은 알고리즘적 추론을 수행하고, 뇌의 계층적/다중 시계열 구조와 유사한 표현 계층(고차원 H, 저차원 L)을 학습한다는 점을 주요 기여로 내세웁니다.[^1_1][^1_2]

***

## 1. 핵심 주장과 주요 기여

- **계층적 순환 추론 아키텍처 제안**
고수준 추상적 계획(high‑level H)과 저수준 세부 연산(low‑level L)이 서로 다른 시간척도에서 상호작용하며, 단일 forward pass 동안 $N\times T$ 단계의 효과적 “깊이”를 갖는 HRM을 제안합니다.[^1_3][^1_1]
- **BPTT 없이 깊은 순환 추론 학습**
Deep Equilibrium Model(DEQ) 이론을 활용한 1‑step 근사 그래디언트와 segment‑기반 deep supervision으로, BPTT 없이 $O(1)$ 메모리로 순환 추론을 안정적으로 학습하는 방법을 제시합니다.[^1_1][^1_3]
- **소량 데이터·소형 모델로 복잡 추론 달성**
약 2,700만 파라미터의 HRM이 ARC‑AGI‑1/2, Sudoku‑Extreme, Maze‑Hard에서 각 1,000개 수준의 학습 예제만으로, 사전학습과 CoT에 의존하는 수십억 파라미터 LLM을 능가하는 성능을 보입니다.[^1_2][^1_1]
- **일반화 가능한 잠재 추론 및 뇌‑유사 표현 구조**
언어 토큰이 아닌 잠재 상태에서 추론을 수행하면서, 상위 모듈이 더 높은 표현 차원(Participation Ratio)을 갖는 뇌 피질의 계층적 차원 구조와 유사한 패턴이 학습 과정에서 자발적으로 나타남을 보입니다.[^1_3][^1_1]

***

## 2. 이 논문이 해결하고자 하는 문제

1. **고정 깊이 Transformer의 계산 복잡도 한계**
표준 Transformer는 고정된 깊이 때문에 AC $^0$ / TC $^0$ 등 낮은 복잡도 계층에 속하며, 다항 시간이나 깊은 트리‑서치가 필요한 문제에 대해 Turing‑complete한 알고리즘 실행이 구조적으로 어렵다는 점을 문제로 지적합니다.[^1_4][^1_1]
2. **Chain‑of‑Thought(CoT)에 대한 과의존**
LLM의 CoT는 언어 수준에서 인간이 설계한 단계 분해에 의존해 brittle하며, 한 단계라도 틀리면 전체 추론이 무너지고, 많은 CoT 예제·긴 토큰 시퀀스를 요구해 데이터·연산 비용이 크고 지연이 크다는 한계를 갖습니다.[^1_5][^1_1]
3. **Recurrent 모델의 조기 수렴과 BPTT 비효율**
일반 RNN은 몇 step 내에 고정점에 빠르게 수렴해 그 이후 step의 업데이트가 사실상 죽어버리고, BPTT는 $O(T)$ 메모리와 긴 credit assignment 체인을 요구해 깊은 순환 추론 학습에 비현실적이라는 점을 지적합니다.[^1_1][^1_3]
4. **언어 외 잠재 공간(latent space)에서의 추론 부재**
인간 두뇌는 언어로 번역하지 않은 잠재 상태에서 긴 연쇄 추론을 수행하지만, 현재 LLM은 언어 토큰 공간에 과도하게 결박되어 있고, 잠재 추론(latent reasoning)을 제대로 활용하지 못한다는 점이 동기입니다.[^1_3][^1_1]

***

## 3. 제안 방법: HRM 수식과 학습 메커니즘

### 3.1 기본 동역학과 계층적 수렴

HRM은 네 개의 모듈로 구성됩니다: 입력 네트워크 $f_I$, 저수준 순환 모듈 $f_L$, 고수준 순환 모듈 $f_H$, 출력 네트워크 $f_O$.[^1_1][^1_3]

- **입력 임베딩**
입력 $x$는 먼저 임베딩으로 사상됩니다.[^1_1]

$$
\tilde{x} = f_I(x; \theta_I)
$$
- **시간 축과 상태**
한 번의 forward pass는 $N$개의 high‑level cycle과 각 cycle 내 $T$개의 low‑level step으로 구성되며, 전체 step index는 $i = 1,\dots, N T$입니다.[^1_3][^1_1]
각 모듈은 은닉 상태 $z_L^i, z_H^i$를 유지하며 초기값 $z_L^0, z_H^0$에서 시작합니다.[^1_1]
- **L‑모듈 업데이트(빠른 세부 연산)**

$$
z_L^i = f_L\bigl(z_L^{i-1}, z_H^{i-1}, \tilde{x}; \theta_L \bigr)
$$

L은 매 step마다 자신의 이전 상태·현재 H 상태·입력을 받아 세부 탐색/정제를 수행합니다.[^1_3][^1_1]
- **H‑모듈 업데이트(느린 추상 계획)**

$$
z_H^i =
\begin{cases}
f_H\bigl(z_H^{i-1}, z_L^{i-1}; \theta_H \bigr), & \text{if } i \equiv 0 \pmod T \\
z_H^{i-1}, & \text{otherwise}
\end{cases}
$$

즉 H는 한 cycle 당 한 번만 업데이트되며, 해당 cycle 끝의 L 상태를 요약해 새로운 전역 컨텍스트를 형성합니다.[^1_3][^1_1]
- **출력**
$N$개의 cycle 후 최종 H 상태에서 출력을 생성합니다.[^1_1]

$$
\hat{y} = f_O\bigl(z_H^{N T}; \theta_O\bigr)
$$

이 구조에서 표준 RNN이 $T$ step 내에 고정점으로 빠르게 수렴해 이후 계산이 거의 죽는 것과 달리, HRM은 각 cycle에서 L이 “국소 수렴”한 뒤 H가 이를 갱신해 L 궤적을 재시작하게 하여, 서로 다른 국소 평형점으로 연속적으로 이동하는 “계층적 수렴(hierarchical convergence)”을 이룹니다. 이로써 효과적인 계산 깊이가 $N T$로 확장되면서도 각 sub‑computation은 안정적으로 수렴합니다.[^1_3][^1_1]

### 3.2 근사 그래디언트: DEQ와 1‑step gradient

이상적인 HRM은 각 high‑level cycle에서 L‑모듈이 고정점 $z_L^\star$까지 수렴한 뒤, 그 값을 이용해 H‑모듈이 한 번 업데이트된다고 가정할 수 있습니다. 이때 고정점 방정식은 다음과 같습니다.[^1_1][^1_3]

$$
z_L^\star = f_L\bigl(z_L^\star, z_H^{k-1}, \tilde{x}; \theta_L\bigr), \quad
z_H^k = f_H\bigl(z_H^{k-1}, z_L^\star; \theta_H\bigr)
$$

이를 $z_H^k = F(z_H^{k-1}; \tilde{x}, \theta)$ 형태의 고정점 $z_H^\star$로 보았을 때, Implicit Function Theorem을 이용하면 정확한 그래디언트는 다음과 같이 쓸 수 있습니다.[^1_1]

```math
\frac{\partial z_H^\star}{\partial \theta}
=
\bigl(I - J_F\bigr)^{-1}
\frac{\partial F}{\partial \theta}
\Big|_{z_H^\star}
```

여기서 $J_F = \frac{\partial F}{\partial z_H}$입니다. 하지만 $(I-J_F)^{-1}$ 계산은 비용이 매우 크므로, Neumann series[^1_1]

$$
(I - J_F)^{-1} = I + J_F + J_F^2 + \dots
$$

를 첫 항 $I$만 남기는 1‑step gradient 근사를 사용합니다. 이때 고정점에 대한 파라미터 그래디언트는 대략 다음과 같이 단순화됩니다.[^1_1]

$$
\frac{\partial z_H^\star}{\partial \theta_H} \approx \frac{\partial f_H}{\partial \theta_H}, \quad
\frac{\partial z_L^\star}{\partial \theta_L} \approx \frac{\partial f_L}{\partial \theta_L}, \dots
$$

즉, 시간 전개 전체를 unroll하지 않고 “마지막 step”의 국소 연산에 대해서만 backprop을 수행하는 꼴이 되어, 메모리 복잡도가 $O(1)$로 떨어지고 BPTT 없이도 학습이 가능합니다.[^1_3][^1_1]

### 3.3 Deep supervision과 Adaptive Computation Time(ACT)

- **Segment‑기반 deep supervision**
하나의 입력 $(x, y)$에 대해 HRM을 여러 segment(반복 forward pass)로 나누어 실행하고, 각 segment의 출력 $\hat{y}_m$에 대해 loss를 계산해 업데이트합니다. 다음 segment로 넘길 때는 은닉 상태 $z_m$를 computation graph에서 detach해, segment 간의 그래디언트 전파를 1‑step 근사로 제한합니다. 이는 DEQ에서 Jacobian regularization 대신 자주 쓰이는 deep supervision과 유사한 안정화 장치입니다.[^1_1]
- **ACT: Q‑learning 기반 동적 정지**
H‑모듈의 최종 상태에서 Q‑head가 “halt/continue”의 Q‑값 $\hat{Q}_m = (\hat{Q}_m^{\text{halt}}, \hat{Q}_m^{\text{cont}})$를 예측하고, 에피소드 수준의 MDP로 Q‑learning을 수행합니다.[^1_1]
    - halt일 때는 예측이 정답이면 보상 1, 아니면 0.
    - continue일 때는 즉시 보상 0, 다음 segment의 Q‑값을 bootstrap 합니다.[^1_1]
행동 선택은 최소 segment 수 $M_{\min}$, 최대 $M_{\max}$ 및 $\varepsilon$-탐색을 섞어 설계해, 쉬운 예제에는 짧게, 어려운 예제에는 길게 “생각”하게 합니다.[^1_1]

이 ACT는 Sudoku‑Extreme처럼 깊은 탐색이 필요한 태스크에서 평균 연산 step은 적게 유지하면서도, 고정 step 모델과 비슷하거나 더 낳은 정확도를 달성하고, inference 시 $M_{\max}$를 늘리면 별도 재학습 없이 성능이 추가로 상승하는 “inference‑time scaling” 특성을 보입니다.[^1_1]

***

## 4. 모델 구조와 구현 세부

- **시퀀스‑투‑시퀀스 구조**
입력/출력은 2D 그리드를 플래튼한 토큰 시퀀스로 표현하며, 입력 임베딩 $f_I$와 출력 헤드 $f_O(z)=\text{softmax}(\theta_O z)$를 사용합니다. 작은 데이터 설정에서는 softmax 대신 안정성을 높이는 stablemax를 사용해 일반화에 도움을 줍니다.[^1_1]
- **H/L 모듈의 내부 아키텍처**
$f_L, f_H$는 모두 **encoder‑only Transformer block**으로 구현되며, 동일한 차원·구조를 갖습니다.[^1_1]
    - RoPE positional encoding, GLU feed‑forward, RMSNorm, bias‑less linear 등 Llama‑계열 현대 LLM의 개선점을 채택합니다.[^1_1]
    - 두 모듈은 입력·상대 모듈 상태를 단순 element‑wise addition으로 결합합니다.[^1_1]
- **초기화와 최적화**
    - LeCun Normal 기반의 truncated 초기화와 Post‑Norm 구조를 사용해 매우 깊은 model에서도 안정적인 학습을 유도합니다.[^1_1]
    - 최적화는 scale‑invariant Adam‑atan2 + AdamW weight decay를 사용해 파라미터 norm을 $L_\infty$ 제약 아래에서 제어함으로써 Q‑learning 안정성 이론 조건을 만족시킵니다.[^1_3][^1_1]
- **Participation Ratio(PR) 분석**
Sudoku‑Extreme‑Full에서 학습된 모델의 상태 궤적 공분산의 고유값 $\{\lambda_i\}$에 대해

$$
\text{PR} = \frac{\bigl(\sum_i \lambda_i\bigr)^2}{\sum_i \lambda_i^2}
$$

를 계산한 결과, H‑모듈 PR $\approx 89.95$, L‑모듈 PR $\approx 30.22$로 고수준 모듈이 훨씬 고차원 표현 공간을 사용함을 보였습니다. 쥐 대뇌 피질에서 계층 상위 영역일수록 PR이 높아지는 실험 결과와 비슷한 비율(약 3배)을 보이며, 학습되지 않은 랜덤 HRM에서는 이런 계층 구조가 나타나지 않습니다.[^1_1]

***

## 5. 성능 향상 및 한계

### 5.1 정량적 성능

논문이 보고하는 주요 결과는 다음과 같습니다.[^1_5][^1_1]

- **ARC‑AGI‑1 (약 960 train examples)**
    - HRM(27M, 30×30 grid context): 40.3%
    - o3‑mini‑high: 34.5%
    - DeepSeek‑R1: 21.0%
    - Claude 3.7 8K: 21.2%
- **ARC‑AGI‑2 (약 1,120 train examples)**
    - HRM: 5.0%
    - DeepSeek‑R1: 0.9%, Claude 3.7 8K: 1.3%, o3‑mini‑high: 3.0% 등보다 우수.[^1_1]
- **Sudoku‑Extreme (9×9, 1,000 train)**
    - HRM: 55.0% 정답률
    - Direct pred Transformer(동일 파라미터 수, CoT/사전학습 없음) 및 CoT LLM baselines: 0%에 수렴.[^1_1]
- **Maze‑Hard (30×30, 1,000 train)**
    - HRM: 74.5% 최단 경로 정답률
    - Direct pred 및 CoT 기반 LLM: 0% 근처.[^1_1]
기존 연구에서 1.75억 파라미터 Transformer를 100만 maze 예제로 학습해도 pass@64 < 20%였던 결과와 비교하면 극적으로 향상된 수치입니다.[^1_1]

이 모든 실험에서 HRM은 **사전학습 없이**, **CoT supervision 없이**, 단순 입출력 예시만으로 학습되며, 이는 잠재 상태에서의 알고리즘 학습·일반화 능력이 강함을 시사합니다.[^1_2][^1_1]

### 5.2 한계와 제약

- **도메인 범위의 한계**
ARC‑AGI, 스도쿠, 미로 등 **기호적·격자 기반** 퍼즐에 초점이 맞춰져 있고, 자연언어/멀티모달 실세계 작업에 대한 실험은 없습니다. 따라서 LLM 대신 HRM을 그대로 대체 가능하다고 보기는 이릅니다.[^1_1]
- **스케일링과 효율성**
HRM은 각 step에서 **full self‑attention**을 사용하므로, 길이가 긴 시퀀스에 대해 메모리/시간이 아직도 $\mathcal{O}(L^2)$입니다. 긴 문맥 언어 모델을 대체하기 위해서는 linear attention/SSM 등과의 결합이 필요하지만 이는 논문에서 “향후 과제”로 남겨둡니다.[^1_1]
- **근사 그래디언트의 이론적 보장 부족**
1‑step gradient는 DEQ 이론에 의해 동기화되지만, 실제 HRM 동역학이 이상적 고정점 조건을 얼마나 만족하는지, 근사가 언제 실패하는지는 실험적으로만 평가되었고, 엄밀한 수렴/일반화 이론은 제시되지 않습니다.[^1_1]
- **설명가능성과 알고리즘 해석**
중간 상태를 시각화해 보니 스도쿠에서는 깊이 우선 탐색+백트래킹, 미로에서는 경로 후보 동시 탐색 후 가지치기 등 인간이 이해할 수 있는 전략이 보이지만, 정확히 어떤 알고리즘을 학습했는지에 대한 체계적 분석은 향후 과제로 남습니다.[^1_1]

***

## 6. 일반화 성능 향상 관점에서의 의미

### 6.1 소량 데이터에서의 강한 일반화

- HRM은 각 태스크에서 **~1,000개의 예시만으로** 학습되며, ARC‑AGI‑1/2와 같은 “작업 단위 태스크 컬렉션”에서 새로운 문제들에 일반화합니다.[^1_5][^1_1]
- 스도쿠와 미로에서 TDoku 난이도·경로 길이가 큰 퍼즐까지 고성능을 유지해, 단순 패턴 암기가 아닌 **탐색 알고리즘 수준의 절차적 일반화**를 달성했음을 시사합니다.[^1_1]

이때 일반화는 (a) 깊은 순환 구조로 충분한 탐색 깊이를 확보하고, (b) deep supervision/ACT로 다양한 난이도의 예시에 대해 적절한 “생각 시간”을 할당할 수 있기 때문에 가능하다고 해석할 수 있습니다.[^1_1]

### 6.2 잠재 추론과 표현 계층 구조

- 언어 CoT 없이 잠재 상태에서 연산을 수행하므로, CoT 텍스트 분해가 잘 정의되지 않거나 언어 외 도메인(이미지, 그리드 등)에서도 동일한 메커니즘으로 작동합니다.[^1_1]
- H‑모듈이 더 높은 Participation Ratio를 가지는 것은, 상위 모듈이 다양한 태스크/전략을 구분할 수 있는 **풍부한 표현 공간**을 제공하고, L‑모듈은 각 단계의 구체적 연산에 특화된 저차원 subspace에서 동작한다는 것을 의미합니다. 이런 구조는 상위 모듈이 새로운 태스크에 대해 “새로운 조합”을 형성하는 데 유리해 일반화에 기여할 수 있습니다.[^1_1]


### 6.3 Inference‑time compute scaling

- ACT와 segment 반복 구조 덕분에, 학습 시의 최대 segment 수 $M_{\max}$보다 큰 값으로 inference 때만 늘려주면 특히 Sudoku‑Extreme에서 정확도가 계속 증가합니다.[^1_1]
- 이는 “테스트 시 추가 연산을 투입하면 더 어려운 예제에 대해 더 깊이 생각하여 성능을 올릴 수 있는” 일반화 스케일링 형태로, CoT‑LLM의 test‑time scaling과 유사하되, 토큰이 아닌 잠재 step에서 이루어진다는 점이 다릅니다.[^1_1]

***

## 7. 2020년 이후 관련 최신 연구 비교

아래는 HRM과 유사하게 **깊은/느린 추론, 계층적/동적 연산 시간, 뇌‑영감 구조**를 탐구하는 최근 연구들입니다.

### 7.1 대표 논문 개요

| 논문 | 연도 | 핵심 아이디어 (요약) | 출처/링크 |
| :-- | :-- | :-- | :-- |
| Hierarchical Reasoning Model (Wang et al.) | 2025 | H/L 두 모듈의 계층적 순환, 1‑step gradient, ACT로 소형 모델이 소량 데이터에서 ARC, 스도쿠, 미로를 해결.[^1_1][^1_2] | arXiv:2506.21734 – https://arxiv.org/abs/2506.21734[^1_2] |
| Dualformer: Controllable Fast and Slow Thinking (Su et al.) | 2024/25 | 랜덤화된 reasoning trace로 학습해, 한 Transformer 내에서 fast/slow/auto 세 모드의 추론을 제어 가능하게 함.[^1_6][^1_7] | arXiv:2410.09918 – https://arxiv.org/abs/2410.09918[^1_6] |
| Continuous Thought Machines (Darlow et al.) | 2025 | neuron‑level temporal processing과 neural synchronization을 통해 내부 “생각 시간” 축을 도입, maze·ImageNet·parity 등에서 순차 추론 및 adaptive compute 달성.[^1_8][^1_9] | arXiv:2505.05522 – https://arxiv.org/abs/2505.05522[^1_8] |
| Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping (Lehnert et al.) | 2024 | A* 검색 궤적을 CoT‑스타일로 사용해 Transformer가 미로 등에서 탐색 알고리즘을 학습하도록 부트스트랩.[^1_1] | arXiv (언급만, HRM 본문에서 인용).[^1_1] |
| Transformers meet Neural Algorithmic Reasoners (Bounsi et al.) | 2024 | GNN 기반 Neural Algorithmic Reasoner를 Transformer와 결합해, 알고리즘 추론을 개선.[^1_1] | arXiv:2406.09308 (HRM에서 인용).[^1_1] |
| Training LLMs to reason in a continuous latent space (Shen et al.) | 2024 | LLM hidden state를 “continuous thought”로 보고 이를 피드백해 연속 잠재 추론(코코넛, CoCoT)을 구현.[^1_1][^1_10] | arXiv (제목·개념만 HRM에서 인용).[^1_1] |

### 7.2 HRM과의 비교 분석

- **CoT 기반 느린 추론 vs 잠재 순환 추론**
Dualformer는 A* 등으로 생성된 reasoning trace를 구조화해 일부를 드롭하는 방식으로 fast/slow 모드를 학습하며, 여전히 언어 CoT에 크게 의존합니다. 반면 HRM은 CoT 없이 순수 입출력 예제만으로 잠재 상태에서 추론 알고리즘을 학습한다는 점에서 **데이터 및 어노테이션 효율성**이 강점입니다.[^1_6][^1_11][^1_1]
- **동일 Transformer vs 계층적 모듈**
Dualformer나 CoT‑LLM들은 단일 Transformer에 모드 제어/추론 trace를 얹는 반면, HRM은 구조적으로 H/L 두 모듈을 분리하여 서로 다른 시계열과 표현 차원을 갖게 함으로써 **표현 계층 구조와 역할 분담**을 보다 명시적으로 구현합니다.[^1_7][^1_3][^1_1]
- **시간 축 도입 방식**
Continuous Thought Machines(CTM)는 각 뉴런에 고유한 temporal kernel과 동기화(synchronization)를 도입해 neuron‑level dynamics로 “생각 step”을 구현합니다. HRM은 뉴런 수준보다는 모듈 수준(H/L, segment, ACT)의 시간 계층을 사용한다는 점에서 추상도가 더 높지만, 둘 다 “추가적인 내부 시간 축을 도입해 복잡 추론·adaptive compute를 실현한다”는 공통 철학을 갖습니다.[^1_12][^1_8][^1_9][^1_1]
- **DEQ/implicit model 계열과의 연결**
HRM의 1‑step gradient와 deep supervision은 DEQ/implicit layer 문헌(Bai et al., Geng et al.)에서 제안된 기법을 순환 추론 아키텍처에 적용한 것으로 볼 수 있습니다. 다만 DEQ는 보통 root‑finding으로 정확한 고정점을 구하는 반면, HRM은 실제로는 유한한 step에서 근사 수렴만 사용하고 1‑step 근사로 그래디언트를 계산한다는 점에서 더 실용적인, 그러나 이론 보장은 약한 설계입니다.[^1_1]
- **알고리즘 학습 vs 표현 학습**
TransNAR, Neural GPU, RRN 등은 명시적으로 알고리즘(정렬, 그래프 문제 등)을 학습하도록 설계된 반면, HRM은 상대적으로 범용적인 seq2seq 아키텍처 안에 계층적 순환 구조를 넣어 “퍼즐‑유형 문제 전반”을 대상으로 훈련합니다. HRM은 ARC‑AGI라는 범용 유도 추론 벤치마크에서의 성능으로 알고리즘/규칙 학습 능력을 간접적으로 보여줍니다.[^1_1]

요약하면, HRM은 **CoT 없는 잠재 순환 추론 + 계층적 모듈 + 근사 implicit 학습 + ACT**를 동시에 결합한 최초의 실용적 아키텍처 중 하나로, Dualformer/CTM/DEQ‑류, Neural Algorithmic Reasoner 계열과 부분적으로 아이디어를 공유하되, “소량 데이터·소형 모델로 범용 퍼즐 추론”이라는 사용 시나리오에 특화되어 있다는 점이 차별점입니다.[^1_9][^1_2][^1_1]

***

## 8. 앞으로의 연구 영향과 고려할 점

### 8.1 연구 방향에 미치는 영향

1. **아키텍처 설계 측면**
    - 단순히 파라미터 수를 키우는 대신, **계층적 순환 구조와 다중 시계열**을 도입해 깊은 계산을 확보하는 방향의 연구를 촉진할 수 있습니다.[^1_5][^1_1]
    - LLM 내부에 HRM‑유형 모듈을 삽입해, 언어 생성은 Transformer가 담당하고, 깊은 알고리즘 추론은 HRM 코어가 담당하는 **하이브리드 구조** 설계도 자연스러운 후속 주제입니다.[^1_4][^1_1]
2. **뇌‑영감 모델링과 신경과학 연결**
    - Participation Ratio 계층 구조, intrinsic timescale 차이 등 신경과학에서 관찰된 현상을 인공 모델에서도 재현했다는 점은 **양방향 신경과학–AI 연구**(예: 실제 피질 데이터와 HRM 상태 비교, 조작 실험 등)를 자극할 수 있습니다.[^1_1]
3. **일반화·표현 분석 연구**
    - HRM이 실제로 어떤 알고리즘(예: DFS, BFS, constraint propagation)을 학습했는지, 어떤 내부 subspace가 어떤 하위 문제에 대응하는지 분석하는 연구는 **설명가능한 알고리즘 추론 모델** 개발에 기여할 것입니다.[^1_1]

### 8.2 향후 연구 시 고려할 점

1. **자연언어·멀티모달 태스크로의 확장 검증**
    - 현재 실험은 주로 합성 퍼즐에 한정되어 있으므로, 수학 추론, 코드, 자연언어 계획, 로봇 플래닝 등으로 HRM을 확장해도 동일한 일반화 이득이 유지되는지 체계적인 검증이 필요합니다.[^1_4][^1_1]
2. **스케일‑업과 효율성·안정성**
    - 더 큰 모델/긴 시퀀스에서 1‑step gradient와 ACT가 여전히 수렴성과 안정성을 유지하는지, exploding/vanishing 문제가 없는지 정밀 실험이 요구됩니다.[^1_1]
    - Linear attention, structured state‑space model(SSM)과 결합해, 긴 문맥에서도 계층적 순환 추론이 가능하도록 만드는 것이 실용적 LLM‑대체를 위한 핵심 과제입니다.[^1_9][^1_1]
3. **학습 신호 설계**
    - HRM은 dense supervised loss를 사용하지만, 실제로는 RL‑style 보상 기반 장기 계획 문제에도 적용하고자 할 수 있습니다. 이 경우 sparse reward와 1‑step gradient 조합의 안정성, credit assignment의 한계를 어떻게 보완할지가 중요합니다.[^1_1]
4. **안전성과 제어 가능성**
    - inference‑time scaling과 ACT 덕분에 “더 오래 생각하면 더 잘 풀리는” 구조는 매력적이지만, 언제/얼마나 오래 생각할지를 잘못 결정하면 latency와 비용이 폭증하거나, 불필요한 over‑thinking이 발생할 수 있습니다.[^1_1]
    - 따라서 실제 시스템 적용 시에는 **사용자‑제어 가능한 compute budget, 신뢰도 기반 조절 정책, 보수적 halting 기준** 등이 함께 설계되어야 할 것입니다.

***

## 참고한 주요 자료 (제목·링크)

- Guan Wang et al., **“Hierarchical Reasoning Model”**, arXiv:2506.21734.[^1_2][^1_1]
- DiJia Su et al., **“Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces”**, arXiv:2410.09918.[^1_6][^1_7]
- Luke Darlow et al., **“Continuous Thought Machines”**, arXiv:2505.05522.[^1_8][^1_9]

위 세 논문과 HRM 본문에 인용된 관련 연구들을 기반으로 내용을 구성했습니다.[^1_3][^1_1]
<span style="display:none">[^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27]</span>

<div align="center">⁂</div>

[^1_1]: Hierarchical-Reasoning-Model.pdf

[^1_2]: https://arxiv.org/abs/2506.21734

[^1_3]: https://arxiv.org/html/2506.21734v1

[^1_4]: https://www.alphaxiv.org/overview/2506.21734v1

[^1_5]: https://papers.cool/arxiv/2506.21734

[^1_6]: https://arxiv.org/abs/2410.09918

[^1_7]: https://huggingface.co/papers/2410.09918

[^1_8]: https://arxiv.org/abs/2505.05522

[^1_9]: https://huggingface.co/papers/2505.05522

[^1_10]: https://www.semanticscholar.org/paper/Dualformer:-Controllable-Fast-and-Slow-Thinking-by-Su-Sukhbaatar/48472a217174053a146b0ebd3821e71b53f05a36

[^1_11]: https://arxiv.org/html/2410.09918v1

[^1_12]: https://arxiv.org/html/2505.05522v4

[^1_13]: https://arxiv.org/html/2510.00355v1

[^1_14]: https://arxiv.org/pdf/2506.21734.pdf

[^1_15]: https://arxiv.org/pdf/2506.02397.pdf

[^1_16]: https://arxiv.org/html/2505.05522v3

[^1_17]: https://arxiv.org/html/2510.08222v1

[^1_18]: https://www.semanticscholar.org/paper/Hierarchical-Reasoning-Model-Wang-Li/6242e10282d7c567af87ca3846bb1f4a1d104ad8

[^1_19]: https://arxiv.org/html/2505.02665v2

[^1_20]: https://arxiv.org/html/2510.03598v1

[^1_21]: https://arxiviq.substack.com/p/hierarchical-reasoning-model

[^1_22]: https://www.themoonlight.io/en/review/dualformer-controllable-fast-and-slow-thinking-by-learning-with-randomized-reasoning-traces

[^1_23]: https://www.rohan-paul.com/p/dualformer-controllable-fast-and

[^1_24]: https://pub.sakana.ai/ctm/

[^1_25]: https://x.com/SakanaAILabs/status/1992909033800716667

[^1_26]: http://arxiv.org/pdf/2410.09918v1.pdf

[^1_27]: https://www.youtube.com/watch?v=oDd9nBqMX20

