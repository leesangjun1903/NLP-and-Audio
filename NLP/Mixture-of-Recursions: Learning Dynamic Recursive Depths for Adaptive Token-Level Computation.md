
# Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation

> **논문 정보**
> - **제목**: Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation
> - **저자**: Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, Se-Young Yun
> - **소속**: KAIST AI, Google DeepMind, Mila (Montreal Institute for Learning Algorithms)
> - **발표 학회**: NeurIPS 2025 (Poster), ICML 2025 (PMLR 267)
> - **arXiv**: [2507.10524](https://arxiv.org/abs/2507.10524) (v1: 2025.07.14, v3: 2025.10.25)
> - **GitHub**: [raymin0223/mixture\_of\_recursions](https://github.com/raymin0223/mixture_of_recursions)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

언어 모델의 스케일링은 강력한 능력을 제공하지만, 이에 따른 계산 및 메모리 비용이 학습과 배포를 모두 매우 비싸게 만든다. 기존 효율성 연구들은 보통 **파라미터 공유(Parameter Sharing)** 또는 **적응적 계산(Adaptive Computation)** 중 하나만을 목표로 하며, 두 가지를 동시에 달성하는 방법은 아직 미해결 문제였다.

이 논문의 핵심 주장은 다음과 같다:

> **Mixture-of-Recursions (MoR)**는 단일 Recursive Transformer 내에서 두 가지 효율성 축을 결합한 통합 프레임워크다. MoR은 재귀 단계 전반에 걸쳐 공유 레이어 스택을 재사용하여 파라미터 효율성을 달성하는 한편, 경량 라우터가 각 토큰에 서로 다른 재귀 깊이를 동적으로 할당함으로써 토큰 수준의 적응적 "사고(thinking)"를 가능하게 한다. 이를 통해 MoR은 특정 재귀 깊이에서 여전히 활성화된 토큰들 사이에서만 이차(quadratic) 어텐션 계산을 수행하며, 해당 토큰들의 KV 쌍만을 선택적으로 캐싱하여 메모리 접근 효율성도 향상시킨다.

### 1.2 주요 기여

**① 통합 언어 모델링 프레임워크**: MoR은 파라미터 공유(§2.1), 토큰 수준 적응적 사고 깊이(§2.2.1), 메모리 효율적 KV 캐싱(§2.2.2)이라는 효율성 패러다임을 단일 프레임워크 내에서 최초로 통합한 아키텍처다.

**② 동적 재귀 라우팅**: 처음부터 학습되어 동적 per-token 재귀 깊이를 할당하는 라우터를 도입한다. 이는 학습과 추론 시의 동작을 일치시키며, 기존 Early-Exit 방법에서 사용되는 비용이 많이 드는 성능 저하적인 사후(post-hoc) 라우팅 단계를 제거한다.

**③ 광범위한 실증 검증**: 135M에서 1.7B 파라미터에 이르는 모델 스케일에서, 동일한 학습 FLOPs와 더 작은 모델 크기로도 검증 퍼플렉서티를 현저히 낮추고 few-shot 정확도를 향상시키며, 기존 바닐라 및 재귀 베이스라인 대비 더 높은 처리량(throughput)을 달성함으로써 새로운 파레토 프런티어(Pareto Frontier)를 형성했다. 이러한 성과는 MoR이 대형 모델 수준의 품질을 대형 모델의 비용 없이 달성하는 효과적인 경로임을 시사한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 연구들이 개별적으로 해결해 온 두 가지 핵심 문제가 있다:

**① 파라미터 비효율성**:
파라미터 효율성으로 가는 검증된 경로 중 하나는 레이어 타이잉(layer tying)으로, 여러 레이어에 걸쳐 공유 가중치 집합이 재사용된다 (Dehghani et al., 2018; Lan et al., 2019; Takase and Kiyono, 2021 등).

**② Early-Exit의 한계 (Missing KV Cache & Batched Inference)**:
Early-exiting을 실제로 적용할 때는 두 가지 주요 병목이 존재한다. **Missing KV Cache 문제**: 토큰이 일찍 종료되면 더 깊은 레이어에 대한 KV 쌍을 계산하지 못하는데, 이 누락된 값들은 미래 토큰 디코딩에 필수적이며, 근사치를 사용하면 성능이 저하된다. **비효율적 배치 추론**: 일찍 종료한 토큰들은 같은 배치 내 다른 토큰들이 전체 계산을 완료할 때까지 대기하게 되어 처리 시간을 낭비한다.

Recursive Transformer는 파라미터 공유를 통해 비효율적인 배치 추론을 완화하려 했지만, 파라미터 공유와 Early-Exiting을 통합하기 위한 두 번의 별도 학습 과정이 성능을 저하시켰고, 여전히 Missing KV Cache 문제를 처리해야 했다.

### 2.2 제안하는 방법

#### 2.2.1 Recursive Transformer 기반 구조

표준 Transformer는 각각 자기 어텐션과 피드포워드 네트워크를 가진 고유 레이어 스택으로 토큰 표현을 구성한다. Recursive Transformer는 레이어 전반에 걸쳐 레이어를 재사용하여 파라미터 수를 줄이고자 한다. 각 레이어마다 별개의 가중치 집합을 갖는 대신, 모델을 재귀 블록으로 분할하여 각 블록이 공유 파라미터 풀을 사용한다.

MoR의 파라미터 공유 전략은 다음과 같이 수식화할 수 있다. 전체 레이어 수를 $L$, 재귀 횟수를 $N_r$라 하면:

$$\mathbf{h}_{t}^{(r)} = \text{RecursionBlock}_{\theta}\!\left(\mathbf{h}_{t}^{(r-1)}\right), \quad r = 1, 2, \ldots, N_r$$

단일 $\text{RecursionBlock}$의 파라미터 $\theta$는 모든 재귀 단계에서 공유된다.

#### 2.2.2 라우팅 메커니즘 (핵심)

MoR은 사전학습과 추론 중 각 토큰의 재귀 단계를 동적으로 조정하는 프레임워크를 제안한다. MoR의 핵심은 두 가지 구성요소에 있다: **라우팅 메커니즘**—더 어려운 토큰에 집중적으로 계산을 할당하기 위해 토큰별 재귀 단계를 지정하는 것—과 **KV 캐싱 전략**—각 재귀 단계에서 KV 쌍이 어떻게 저장되고 선택적으로 활용되는지를 정의하는 것.

**① Expert-Choice Routing (EC)**:
각 재귀 단계에서 라우터가 상위- $k$개의 토큰을 선택하여 계속 진행시키고, 깊이가 깊어질수록 활성 토큰 집합을 점진적으로 좁혀간다.

Expert-Choice Routing의 선택 과정을 수식으로 표현하면:

재귀 단계 $r$에서, 라우터는 토큰의 히든 상태 $\mathbf{h}_t^{(r)}$를 입력으로 받아 스코어를 계산한다:

$$s_t^{(r)} = \text{Router}^{(r)}\!\left(\mathbf{h}_t^{(r-1)}\right) \in \mathbb{R}$$

상위 $k$개 토큰이 선택되어 재귀 블록을 통과하며, 나머지는 종료(exit)한다:

$$\mathcal{S}^{(r)} = \text{Top-}k\!\left(\{s_t^{(r)}\}_{t=1}^{T}\right)$$

활성 토큰 집합 $\mathcal{S}^{(r)}$에만 계산이 적용되므로, 전체 계산량은:

$$\text{FLOPs}_{r} \propto |\mathcal{S}^{(r)}| \cdot C_{\text{block}}, \quad |\mathcal{S}^{(r)}| = k_r \leq T$$

**② Token-Choice Routing (TC)**:
각 토큰은 처음 한 번의 라우팅 결정에 의해 고정된 재귀 단계가 할당되어, 모델을 통한 완전한 계산 경로를 미리 정의한다.

Token-Choice의 수식은:

$$r_t^* = \arg\max_{r \in \{1,\ldots,N_r\}} \text{Router}_{\text{TC}}(\mathbf{h}_t^{(0)})$$

즉, 토큰 $t$는 고정된 $r_t^*$번만큼 재귀 블록을 통과한다.

**라우터 보조 손실 함수**:

Expert-Choice 라우팅의 로드 밸런싱을 위해 보조 손실을 사용한다:

$$\mathcal{L}_{\text{aux}} = \alpha \cdot \sum_{r=1}^{N_r} \left( \frac{|\mathcal{S}^{(r)}|}{T} - \frac{k_r}{T} \right)^2$$

여기서 $\alpha$는 보조 손실의 가중치 하이퍼파라미터이다.

최종 학습 손실:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{aux}}$$

#### 2.2.3 KV 캐싱 전략

**① Recursion-Wise KV Caching**:
재귀 단계별 KV 캐싱에서는, 각 재귀 단계에서 현재 선택된(드롭되지 않은) 토큰들의 키만 캐싱되며, 어텐션은 이 항목들로만 제한된다.

재귀 단계 $r$에서의 어텐션은 다음과 같이 제한된다:

$$\text{Attn}^{(r)}(Q_t, K, V) = \text{softmax}\!\left(\frac{Q_t K_{\mathcal{S}^{(r)}}^{\top}}{\sqrt{d_k}}\right) V_{\mathcal{S}^{(r)}}$$

이차 어텐션 비용이 $O(|\mathcal{S}^{(r)}|^2)$으로 줄어들어 효율적이다.

**② KV Sharing Variant**:
KV 공유 변형은 첫 번째 재귀의 KV 쌍을 재사용하도록 설계되어 메모리 풋프린트를 더욱 줄인다.

수식으로:

$$K^{(r)} = K^{(1)}, \quad V^{(r)} = V^{(1)}, \quad \forall r > 1$$

깊이 간 파라미터를 공유하는 레이어들은 매우 일관된 크기 패턴과 높은 코사인 유사도를 나타내며, 이는 KV 공유가 약간의 성능 저하만 유발하는 이유를 정당화한다.

### 2.3 모델 구조

MoR 모델은 Llama 아키텍처를 기반으로 구축되었으며, 구체적으로는 `LlamaForCausalLM` 클래스를 수정하여 구현된다.

각 재귀 단계는 고정된 레이어 스택과 각 토큰이 통과할지 종료할지를 결정하는 라우터로 구성된다. 이 재귀 블록이 핵심 단위이다. 전체 모델 구조에서 공유 재귀 단계는 라우터 결정에 따라 각 토큰마다 최대 $N_r$번 적용된다. 라우팅 패턴 예시는 토큰별 재귀 깊이를 보여주는데, 어두운 셀이 재귀 블록을 통한 활성 계산을 나타낸다. 각 텍스트 토큰의 재귀 단계 수(1, 2, 3)를 컬러로 표시한다.

파라미터 공유 전략으로는 **Cycle**과 **Middle-Cycle** 방식이 있으며, Middle-Cycle은 첫 번째와 마지막 레이어는 고유한 파라미터를 유지한다.

전체 아키텍처의 구조를 표로 요약하면:

| 구성요소 | 설명 |
|---|---|
| Input/Embedding Layer | 고유 파라미터 (비공유) |
| Recursion Block × $N_r$ | 공유 파라미터 (weight-tied) |
| Router (per recursion) | 경량 선형 레이어 |
| KV Cache | Recursion-Wise or Sharing |
| Output/LM Head | 고유 파라미터 (비공유) |

### 2.4 성능 향상

동일한 학습 컴퓨팅 예산(compute budgets) 하에서 MoR은 바닐라 및 재귀 베이스라인 모두를 일관되게 능가한다. 16.5e18 FLOPs에서 Expert-Choice 라우팅과 2번의 재귀를 사용한 MoR 모델은 더 낮은 검증 손실을 달성하고, **약 50% 적은 파라미터**를 사용함에도 불구하고 평균 few-shot 정확도(43.1% vs. 42.3%)에서 바닐라 베이스라인을 능가했다.

고정된 학습 토큰을 통해 아키텍처 차이만을 고립시켜 분석했을 때, 2번의 재귀를 가진 MoR은 **25% 적은 학습 FLOPs**를 사용하면서도 바닐라 및 재귀 베이스라인 모두를 능가했다.

MoR 모델은 더 낮은 검증 손실과 더 높은 정확도를 달성하면서도 바닐라 베이스라인 대비 **학습 시간 19% 감소** 및 **피크 메모리 사용량 25% 절감**을 달성했다. 이러한 개선들은 계층적 필터링과 재귀별 어텐션 메커니즘에서 비롯된 것으로, 사전학습 중에도 우월한 계산-정확도 트레이드오프를 실현한다.

MoR-4(4번 재귀)는 최대 **2.18배 높은 추론 처리량**을 달성한다.

MoR은 베이스라인 대비 독특한 컴퓨팅 최적 스케일링 동작을 보인다. isoFLOPs 제약 하에서 MoR은 추가 학습 데이터보다 파라미터 수 증가로부터 더 많은 이점을 얻는다.

### 2.5 한계

성능은 라우팅 및 캐싱 전략에 따라 달라진다. Expert-Choice 라우팅이 Token-Choice 라우팅을 일관되게 능가하며, Recursive KV Sharing은 독립적 캐싱 대비 성능을 약간 저하시키지만 향상된 메모리 효율성을 제공한다.

MoR에는 여전히 도전과제가 존재한다. 동적 라우팅 메커니즘은 성능 저하를 방지하기 위한 신중한 학습이 필요하며, Recursive KV Sharing의 트레이드오프는 다양한 태스크에서 추가 탐구가 필요하다.

최근 연구들은 추론 체인 내의 중복성을 강조하며 Early-Exit 메커니즘으로 이를 해결하고자 한다. MoR 프레임워크는 개별 토큰에 필요한 재귀 깊이를 적응적으로 결정함으로써 잠재적 추론(latent reasoning)을 가능하게 하는데, **실제 추론 데이터셋에서 사후 학습(post-training) 시 라우터가 CoT(Chain-of-Thought) 체인의 필요성에 동적으로 적응하는 방법을 탐구하는 것이 중요한 미래 과제로 남아 있다.**

---

## 3. 일반화 성능 향상 가능성

### 3.1 Few-Shot 일반화 성능

MoR은 파라미터 공유, 적응적 재귀 깊이, 효율적 KV 캐싱을 모델 품질 저하 없이 동시에 활용하는 통합 Transformer 아키텍처다. 경량 라우터를 통해 토큰에 재귀 깊이를 동적으로 할당하고 선택된 토큰의 KV 상태를 선택적으로 캐싱함으로써 이차 어텐션 계산과 중복 메모리 접근 비용을 모두 줄인다. 광범위한 실증 평가에서 MoR은 바닐라 및 이전 재귀 베이스라인 대비 검증 퍼플렉서티를 낮추고 평균 few-shot 정확도를 향상시켰으며, 이는 더 높은 추론 처리량에서도 마찬가지였다.

### 3.2 잠재 추론(Latent Reasoning)을 통한 일반화

MoR은 사전학습 프레임워크로서 잠재 공간 추론—단일 파라미터 블록을 반복적으로 적용하여 비언어적 사고를 수행하는—을 제공한다. 기존 방식들이 생성 전 연속 프롬프트를 통해 숙고하는 것과 달리, MoR은 각 토큰 디코딩 중 직접 잠재적 사고를 가능하게 한다. 더 나아가 라우팅 메커니즘은 모델의 수직 축(vertical axis)을 따라 적응적 추론을 촉진하며, 이는 이전 연구들의 균일한 고정 사고 깊이를 넘어선다.

이러한 방법들은 계산이 가장 필요한 곳에 할당하는 유연성이 부족하여 쉬운 입력에서는 불필요한 오버헤드, 복잡한 입력에서는 불충분한 추론을 초래한다. 루프(looping)가 모델 추론 능력을 향상시킨다는 최근 연구 결과들을 기반으로, MoR 프레임워크는 **적응적 계산과 잠재 추론을 연결하는 핵심 기반**을 제공한다고 저자들은 주장한다.

### 3.3 스케일링 일반화

135M에서 1.7B 파라미터에 이르는 모델 스케일에서 MoR은 새로운 파레토 프런티어를 형성했다: 동일한 학습 FLOPs와 더 작은 모델 크기로도 검증 퍼플렉서티를 현저히 낮추고 few-shot 정확도를 향상시키며, 기존 바닐라 및 재귀 베이스라인 대비 더 높은 처리량을 달성했다.

### 3.4 멀티모달 일반화 가능성

MoR의 재귀 블록은 본질적으로 **모달리티 불가지론적(modality-agnostic)**이어서, 텍스트 처리 너머로 적응적 깊이 메커니즘을 확장할 수 있다. 이 중요한 특성은 MoR이 비전, 음성, 통합 멀티모달 Transformer 아키텍처에 쉽게 통합될 수 있게 한다. 장문 맥락 비디오나 오디오 스트림에 토큰 적응형 재귀를 적용하면 훨씬 더 큰 메모리 효율성과 상당한 처리량 향상이 가능하다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

아래는 MoR과 밀접하게 관련된 주요 연구들의 비교 분석이다.

| 연구 | 연도 | 핵심 아이디어 | 파라미터 공유 | 적응적 계산 | MoR과의 차이 |
|---|---|---|---|---|---|
| **Universal Transformer** (Dehghani et al.) | 2018 | 반복 적용 공유 블록 | ✅ | ❌ (고정 깊이) | 적응적 라우팅 없음 |
| **Depth-Adaptive Transformer** (Elbayad et al.) | 2020 | 토큰 수준 Early-Exit | ❌ | ✅ | 파라미터 공유 없음 |
| **CALM** (Schuster et al.) | 2022 | 신뢰도 기반 Early-Exit | ❌ | ✅ | KV 미싱 문제 미해결 |
| **Looped Transformer** (Giannou et al.) | 2023 | 반복 적용으로 프로그래밍 능력 | ✅ | ❌ | 고정 깊이 |
| **Mixture-of-Depths (MoD)** (Raposo et al.) | 2024 | 레이어별 토큰 서브셋 선택 | ❌ | ✅ | 파라미터 공유 없음 |
| **Recurrent Depth** (Geiping et al.) | 2025 | Test-time 스케일링 | ✅ | ❌ | 균일 고정 깊이 |
| **FREE / Recursive Transformer** (Bae et al.) | 2024 | 파라미터 공유 + Early-Exit | ✅ | ✅ | 별도 학습 필요, 성능 저하 |
| **MoR (본 논문)** | 2025 | 통합 프레임워크 | ✅ | ✅ | **최초 통합, E2E 학습** |

최근 Mixture-of-Depths (MoD)는 적응적 깊이를 라우팅 문제로 재구성하여, 각 레이어의 경량 라우터가 전체 계산을 받을 토큰의 서브셋을 선택하고 나머지는 레이어를 우회하도록 하여 더 세밀한 조건부 컴퓨팅을 구현했다.

MoR은 이 MoD 라우팅 아이디어를 재귀 Transformer에 적용하는데: 토큰들이 별개의 레이어를 통과하는 대신 단일 가중치-묶인(weight-tied) 블록의 반복 호출을 통해 동적으로 전송된다. 이러한 전환은 파라미터 수를 일정하게 유지하면서 모델의 물리적 깊이를 넘어서는 임의로 깊은 적응적 계산을 가능하게 한다.

파라미터 공유는 효율성을 향한 직교적인 경로를 제공한다. Universal Transformer는 단일 블록을 반복 적용하는 것이 깊은 비공유 스택의 표현력과 일치할 수 있음을 처음으로 보였다. Looped Transformer는 프로그래밍 가능한 컴퓨터 역할을 할 수 있고, 알고리즘적 태스크에서 훨씬 더 긴 입력으로 일반화할 수 있음을 보였다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

**① 통합 효율성 프레임워크의 새로운 표준**

MoR 프레임워크는 적응적 계산과 잠재 추론을 연결하는 핵심 기반을 제공한다. MoR은 모델 품질을 저해하지 않으면서 파라미터 공유, 적응적 재귀 깊이, 효율적 KV 캐싱을 동시에 활용하는 통합 Transformer 아키텍처를 제시한다.

**② 추론 모델(Reasoning Model)로의 확장 가능성**

이 프레임워크는 추론 응용 분야에서도 가능성을 보인다. 최근 연구들이 추론 체인의 중복성을 지적하고 있으며, MoR의 적응적 계산은 개별 토큰의 필요 재귀 깊이를 결정하여 자연스럽게 잠재 추론을 가능하게 한다. 재귀 깊이를 추론 복잡성에 맞추는 라우팅 전략을 개발하면 정확도와 효율성을 모두 향상시킬 수 있다.

**③ 멀티모달 확장의 청사진**

MoR 방식은 효율적인 대규모 계산을 위한 매력적인 방향을 제시하며, Mixture-of-Experts와 다규모 재귀 모델링의 아이디어를 다양한 도메인으로 일반화한다. 이러한 프레임워크들은 입력 복잡성이나 쿼리 요건에 가장 잘 맞게 계산이나 검색을 적응적으로 할당하며, 자원 제약이 중요한 환경에서 학습과 배포의 비용 절감과 확장성을 제공한다.

### 5.2 미래 연구에서 고려할 점

**① 추론 후 학습(Post-training) 시 라우터 적응**:
MoR 프레임워크는 개별 토큰에 필요한 재귀 깊이를 적응적으로 결정함으로써 잠재 추론을 가능하게 한다. 따라서 중요한 미래 과제는 라우터가 실제 추론 데이터셋에서 사후 학습될 때 CoT 체인의 필요성에 동적으로 적응하는 방법을 탐구하는 것이다.

**② MoE와의 통합 가능성**:
MoR을 기반으로, Mixture-of-Experts(MoE)나 Mixture-of-Depths(MoD) 등 다른 기법들과 통합하는 미래 연구가 가능하다.

**③ 동적 샘플 스케줄링 및 추론 최적화**:
이 프레임워크의 유연성은 동적 샘플 스케줄링 및 연속적 깊이 방향 배칭(continuous depth-wise batching)의 혁신을 위한 문을 열어주며, 이는 처리량을 최대 2.76배까지 향상시킬 수 있다.

**④ 비텍스트 도메인 적용 탐구**:
멀티모달 및 비텍스트 도메인으로의 확장이 유망하다. MoR의 재귀 블록은 본질적으로 모달리티 불가지론적이어서 적응적 깊이 메커니즘이 텍스트 처리를 넘어 확장될 수 있다. 이 중요한 특성은 MoR이 비전, 음성, 통합 멀티모달 Transformer 아키텍처에 쉽게 통합되도록 한다. 장문 맥락 비디오나 오디오 스트림에 토큰 적응형 재귀를 적용하면 훨씬 더 큰 메모리 효율성과 상당한 처리량 향상이 가능하다.

**⑤ 라우터 해석 가능성(Interpretability)**:

라우터가 실제로 어떤 기준으로 토큰의 "어려움(difficulty)"을 학습하는지를 해석하는 연구가 필요하다. 선택된/선택되지 않은 토큰에 대한 라우터 출력 분포와 재귀 깊이 증가에 따른 로그 우도 개선에 대한 테스트 타임 스케일링 분석은 라우터의 동작 방식을 이해하는 데 중요한 분석 도구가 된다.

**⑥ KV 캐싱 최적화 전략의 정밀화**:
KV 캐시가 초기 재귀에서 생성되어 이후 단계에서 재사용될 수 있어 메모리 소비를 더욱 줄일 수 있다. 이는 Cycle 전략에서만 첫 번째 재귀 동안만 프리필(prefill) 단계를 실행하면 된다는 장점을 제공하여, 1백만 토큰 이상의 프롬프트 설정에서 상당한 속도 향상을 약속한다. 두 가지 캐싱 전략은 다양한 배포 환경에 맞게 최적화될 수 있다.

---

## 📚 참고 자료 (출처)

1. **arXiv 원문 논문**: Sangmin Bae et al., "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation," arXiv:2507.10524, 2025. https://arxiv.org/abs/2507.10524

2. **NeurIPS 2025 포스터**: https://neurips.cc/virtual/2025/poster/118085

3. **OpenReview (NeurIPS 2025)**: https://openreview.net/forum?id=QuqsEIVWIG

4. **ICML 2025 (PMLR 267)**: https://openreview.net/pdf?id=YtQtGsNr64

5. **공식 GitHub 레포지토리**: https://github.com/raymin0223/mixture_of_recursions

6. **ResearchGate 논문 PDF**: https://www.researchgate.net/publication/393684588

7. **arXiv HTML 버전 (v1)**: https://arxiv.org/html/2507.10524v1

8. **arXiv HTML 버전 (v2)**: https://arxiv.org/html/2507.10524v2

9. **Hugging Face Papers**: https://huggingface.co/papers/2507.10524

10. **AI Models FYI 논문 요약**: https://www.aimodels.fyi/papers/arxiv/mixture-recursions-learning-dynamic-recursive-depths-adaptive

11. **관련 후속 연구**: "Understanding Dynamic Compute Allocation in Recurrent Transformers," arXiv:2602.08864, 2026. https://arxiv.org/html/2602.08864

12. **MoR-ViT (비전 도메인 확장)**: arXiv:2507.21761 — "MOR-VIT: Efficient Vision Transformer with Mixture-of-Recursions"
