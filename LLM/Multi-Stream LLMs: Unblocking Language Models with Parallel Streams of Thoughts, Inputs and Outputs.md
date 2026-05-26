
# Multi-Stream LLMs: Unblocking Language Models with Parallel Streams of Thoughts, Inputs and Outputs

> **논문 정보**
> - **저자:** Guinan Su, Yanwu Yang, Xueyan Li, Jonas Geiping
> - **소속:** Max Planck Institute for Intelligent Systems, Tübingen AI Center, ETH Zurich, University of Tübingen, ELLIS Institute Tübingen
> - **arXiv:** [2605.12460](https://arxiv.org/abs/2605.12460) (2026년 5월 12일)
> - **코드:** [github.com/seal-rg/streaming](https://github.com/seal-rg/streaming)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

LLM 기반 자율 에이전트의 능력이 크게 향상되었음에도 불구하고, 이 시스템들의 핵심 구조는 초기 ChatGPT 이후 크게 변하지 않았다. 심지어 고급 AI 에이전트조차 사용자, 시스템, 그리고 자기 자신(chain-of-thought)과 도구 간에 메시지를 순차적으로 교환하는 단일 스트림 방식으로 작동한다.

이 단일 스트림 병목으로 인해, 에이전트는 읽는 동안 행동(출력 생성)을 할 수 없고, 쓰는 동안 새로운 정보에 반응할 수 없다. 마찬가지로 생각하는 동안 행동할 수 없고, 읽거나 행동하는 동안 생각할 수 없다.

### 1.2 주요 기여 (Contributions)

저자들은 **다중 스트림 병렬 생성(multi-stream parallel generation)**을 제안한다. 이는 LLM이 단일 순방향 패스(forward pass)에서 여러 병렬 토큰 스트림을 어텐션하고 생성하도록 instruction-tuning 방식을 원칙적으로 변경하는 것이며, 메시지 기반 데이터를 변환하고 기존 챗 모델로부터 새로운 스트림 학습 데이터를 생성하는 데이터 구성 레시피를 함께 제공한다.

이 데이터 중심 변화는 여러 사용성 한계를 해소하고, 병렬화를 통한 모델 효율성 향상, 관심사의 분리(separation of concerns)를 통한 보안 향상, 그리고 모델 모니터링 가능성(monitorability) 개선에 기여함을 주장한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

현재 GPT-4o, Claude, Gemini 등 주요 AI 시스템은 단일 순차 스트림을 기반으로 설계되어 있다. 모델이 메시지를 읽고 (chain-of-thought를 통해) 생각한 후 응답을 작성하는 구조이며, 이는 한 번에 한 가지 일만 처리할 수 있다. 이 핵심 구조는 2022년 말 ChatGPT 출시 이후 모델 능력이 크게 향상되었음에도 거의 변하지 않았다.

이를 정리하면:

| 제약 | 설명 |
|---|---|
| **읽기 ↔ 행동 차단** | 입력을 읽는 동안 출력 불가 |
| **쓰기 ↔ 반응 차단** | 출력 중 새 정보 반영 불가 |
| **생각 ↔ 행동 차단** | 추론(reasoning) 중 행동 불가 |
| **읽기 ↔ 생각 차단** | 입력 처리 중 추론 불가 |

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 핵심 아이디어

저자들은 순차적 메시지 형식을 위한 instruction-tuning에서 **각 역할(role)을 별도의 스트림으로 분리하는 여러 병렬 계산 스트림**을 위한 instruction-tuning으로 전환함으로써 모델의 제약을 해소할 수 있음을 보인다.

언어 모델의 모든 순방향 패스(forward pass)는 이전 타임스텝에 인과적으로(causally) 의존하면서 동시에 여러 입력 스트림을 읽고 여러 출력 스트림에 토큰을 생성한다.

#### (2) 형식화 (Formalization)

논문의 공개 HTML 전문(arxiv.org/html/2605.12460)과 GitHub 코드 저장소에서 확인되는 내용을 바탕으로, 핵심 수식 구조를 아래와 같이 기술합니다.

**기존 단일 스트림 자동회귀 LLM의 생성 방식:**

$$p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

**다중 스트림 병렬 생성:**

$N$개의 스트림 $\{s^{(1)}, s^{(2)}, \ldots, s^{(N)}\}$이 존재할 때, 타임스텝 $t$에서 스트림 $k$의 토큰 $x_t^{(k)}$는 **모든 스트림의 이전 타임스텝 토큰들**에 인과적으로 의존한다:

$$p\!\left(x_t^{(k)} \mid \mathbf{x}_{<t}^{(1)}, \mathbf{x}_{<t}^{(2)}, \ldots, \mathbf{x}_{ < t}^{(N)}\right)$$

즉, 전체 결합 분포는 다음과 같이 표현된다:

$$p(\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(N)}) = \prod_{t=1}^{T} \prod_{k=1}^{N} p\!\left(x_t^{(k)} \mid \mathbf{x}_{ < t}^{(1:N)}\right)$$

여기서 $\mathbf{x}\_{ < t}^{(1:N)} = \{x_1^{(j)}, \ldots, x_{t-1}^{(j)}\}_{j=1}^{N}$는 모든 스트림의 이전 타임스텝 토큰 집합이다.

> ⚠️ **주의:** 위 수식은 논문의 핵심 아이디어인 "각 타임스텝에서 여러 스트림의 토큰이 동시에 생성되며, 서로의 이전 타임스텝 정보를 공유"한다는 개념을 수식화한 것입니다. 논문 전문의 정확한 표기는 [arXiv HTML](https://arxiv.org/html/2605.12460)을 직접 참고하시기 바랍니다.

#### (3) 어텐션 마스크 구조

병렬 스트림 간 인과성을 유지하기 위해, 기존의 인과적 어텐션 마스크(causal attention mask)를 확장한다. 타임스텝 $t$에서 스트림 $k$의 토큰은:
- **같은 타임스텝** $t$에서 **다른 스트림의 토큰**은 볼 수 없음 (동시 생성, 순환 의존성 방지)
- **이전 타임스텝** $t' < t$의 **모든 스트림 토큰**은 참조 가능

$$\text{Mask}_{t,k,t',k'} = \begin{cases} 1 & \text{if } t' < t \\ 0 & \text{otherwise} \end{cases}$$

#### (4) 데이터 구성

기존 메시지 기반 데이터를 스트림 형식으로 변환하고, 기존 챗 모델로부터 새로운 스트림 학습 데이터를 생성하는 레시피를 제공한다.

---

### 2.3 모델 구조

#### 스트림 구성의 예

논문은 아래와 같은 스트림 분리 구조를 실험한다 (논문 GitHub 코드베이스 기반):

| 스트림 | 역할 |
|---|---|
| **User Input Stream** | 사용자 입력 처리 |
| **System/Tool Stream** | 시스템 메시지, 도구 호출 |
| **Thinking Stream** | Chain-of-thought 추론 |
| **Output Stream** | 최종 사용자 응답 생성 |
| **Auxiliary Monitor Stream** | 모니터링용 보조 스트림 |

표준 chain-of-thought와 달리, 이 보조 스트림들은 직접적인 추론에만 집중하도록 암묵적 압력을 받지 않으며, 사용자 대면 메시지나 주된 추론 흐름에 나타나지 않는 의도를 표현하는 공간을 모델에게 제공한다. 이 보조 스트림에서는 모델의 상황 인식(situational awareness)이, 보이는 출력이나 주된 사고 흐름에서는 나타나지 않더라도, 표현된다는 것을 발견했다.

---

### 2.4 성능 향상

실험(Table 3)에서 스트림 모델은 직접 및 간접 공격 성공률(낮을수록 좋음)에서 더 나은 성능을 보이며, 안전하고 도움이 되는 응답(safe-and-helpful) 및 지시 따르기(instruction following) 성능(높을수록 좋음)에서도 전반적으로 더 나은 결과를 보여, 더 나은 관심사의 분리(separation-of-concerns)를 나타낸다.

**성능 향상 요약:**

| 측면 | 향상 내용 |
|---|---|
| **효율성** | 읽기·생각·쓰기 병렬화로 지연(latency) 감소 |
| **보안** | 시스템/사용자/추론 스트림 분리로 간접 공격 저항성 향상 |
| **모니터링** | 보조 스트림을 통한 모델 내부 상태 관찰 가능 |
| **반응성** | 새로운 정보를 출력 중에도 반영 가능 |

추론과 출력이 별도 스트림에 존재할 때, 각 단계에서 모델이 실제로 무엇을 하는지 검사하기 더 쉬워진다. 새로운 정보를 읽으면서 동시에 출력을 업데이트할 수 있는 에이전트는 새 데이터를 흡수하기 전에 먼저 생각을 끝내야 하는 시스템과는 질적으로 다른 종류의 시스템이 된다.

---

### 2.5 한계 (Limitations)

이 연구는 LLM 효율성의 어떤 발전과 마찬가지로 오용의 가능성이 있다고 저자들도 인정하지만, 보안 및 모니터링 가능성의 이점이 배포 시스템의 안전성을 개선할 가능성이 더 높다고 본다.

논문에서 확인 가능한 추가 한계:

1. **학습 데이터 구성 복잡성:** 기존 단일 스트림 데이터를 멀티 스트림 형식으로 변환하는 파이프라인이 필요하며, 이 변환의 품질이 최종 성능에 영향을 준다.
2. **스트림 수 설계의 비자명성:** 최적 스트림 수 $N$ 및 각 스트림의 역할 분배 기준이 아직 명확한 이론적 근거 없이 경험적으로 설계된다.
3. **추론 비용:** 여러 출력 헤드(output head)를 동시에 운용하므로, 동일 파라미터 수 대비 메모리 및 추론 비용이 증가할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

다중 스트림 구조는 일반화 성능 향상과 관련하여 여러 중요한 가능성을 제시한다.

### 3.1 관심사의 분리(Separation of Concerns)에 의한 일반화

기존 단일 스트림 모델에서는 추론, 도구 호출, 사용자 응답 생성이 하나의 시퀀스에 혼재된다. 이로 인해:

$$\text{단일 스트림: } p(x_t) = p(\text{reason}, \text{action}, \text{response} \mid x_{ < t})$$

다중 스트림 분리 시:

$$p(x_t^{(\text{think})}) \cdot p(x_t^{(\text{output})}) \cdot p(x_t^{(\text{monitor})}) = \text{각 스트림이 독립적 역할 학습}$$

이처럼 역할이 명확히 분리되면, 각 스트림이 **전문화된 표현(specialized representation)**을 학습하여 새로운 도메인 및 과제에 대한 일반화가 개선될 수 있다.

### 3.2 보조 스트림의 정규화 효과 (Regularization via Auxiliary Streams)

병렬 스트림의 핵심 동기 중 하나는 모니터링 가능성이다. 보이는 답변과 함께 실행되는 스트림은 외부 관찰자에게 모델이 실제로 고려하는 바를 직접 파악하게 해준다. 저자들은 여러 내부 채널의 존재가 모델이 현재 대화에 대한 고려를 '내적으로 발화(sub-vocalize)'하는 데 도움을 줄 수 있다고 가정하며, 이 보조 스트림들은 수학 문제의 구체적인 기능적 추론 단계에만 집중할 필요가 없기 때문이다.

이 보조 스트림은 다음과 같이 **암묵적 정규화 효과(implicit regularization)**로 작용할 수 있다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + \lambda \sum_{k \in \text{aux}} \mathcal{L}^{(k)}_{\text{auxiliary}}$$

보조 스트림에 대한 학습 손실 $\mathcal{L}^{(k)}_{\text{auxiliary}}$는 모델이 내부 추론 과정을 명시적으로 표현하도록 강제하여, 단순 암기(memorization)보다 일반화 가능한 표현을 학습하도록 유도한다.

### 3.3 분포 외 데이터(Out-of-Distribution)에서의 강건성

관심사 분리 구조는 OOD 시나리오에서도 이점을 제공한다. 예를 들어:

- **도구 스트림**이 새로운 API나 도구를 만날 때, 추론 스트림과 독립적으로 업데이트 가능
- **사용자 입력 스트림**의 분포가 바뀌어도, 추론 스트림은 기존에 학습한 추론 패턴을 유지 가능

이 데이터 중심 변화는 여러 사용성 한계를 해소하고, 병렬화를 통한 효율성 향상, 더 나은 관심사 분리를 통한 보안 개선, 그리고 모델 모니터링 가능성 향상에 기여한다고 주장한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | Multi-Stream LLMs와의 비교 |
|---|---|---|
| **Chain-of-Thought (Wei et al., 2022)** | 단일 스트림 내 추론 단계 명시 | Multi-Stream은 추론을 별도 스트림으로 분리, 상호 간섭 방지 |
| **Tree-of-Thought (Yao et al., 2023)** | 트리 구조 탐색으로 추론 다양화 | Multi-Stream은 실시간 병렬 처리, ToT는 순차 탐색 |
| **Skeleton-of-Thought (Ning et al., 2023)** | 외부 오케스트레이션으로 병렬 생성 | Multi-Stream은 모델 내부에서 병렬 처리, 일관성 손실 최소화 |
| **Parallel Decoder Transformer (2025, arXiv:2512.10054)** | 동결된 모델에 SNC 어댑터 추가 | Multi-Stream은 instruction-tuning 기반, 재학습 필요 |

자동회귀 디코딩은 본질적으로 순차적이어서 출력 길이에 비례하는 지연 병목을 야기한다. Skeleton-of-Thought 같은 외부 오케스트레이션 방식은 스트림 간 교차 통신 부재로 인한 일관성 손실(coherence drift)이 발생한다. PDT는 동결된 사전학습 모델의 추론 과정에 조정 프리미티브를 직접 내장하는 파라미터 효율적 아키텍처로, 기반 모델 재학습 없이 경량 SNC 어댑터를 주입한다.

**Multi-Stream LLMs의 차별점:**
- 기존 연구들이 **추론 품질 향상**에 집중한 반면, 본 논문은 **아키텍처적 병목 자체**를 해소하려 한다.
- 데이터 중심(data-driven) 접근으로, 별도의 아키텍처 변경 없이 instruction-tuning만으로 구현 가능하다.
- 보안 및 모니터링이라는 **안전성(safety)** 관점을 동시에 다루는 점이 독특하다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

1. **에이전트 아키텍처의 패러다임 전환**
   이 논문은 대형 언어 모델이 정보를 처리하는 방식에 대한 근본적인 재고를 제안하며, 만약 이 접근법이 유효하다면 현재 출시 중인 거의 모든 AI 에이전트의 아키텍처를 바꿀 수 있다.

2. **실시간 반응형 에이전트의 구현 가능성**
   코딩 어시스턴트, 컴퓨터 사용 에이전트, 고객 서비스 봇 등은 현재 일종의 인지적 터널 비전 속에서 작동한다. Multi-stream LLM이 설명된 대로 작동한다면, 이러한 시스템들이 단순히 빠른 것을 넘어 진정으로 반응적(reactive)이 될 수 있다.

3. **AI 안전성(AI Safety) 연구의 새 도구**
   병렬 스트림의 핵심 동기 중 하나인 모니터링 가능성은, 외부 관찰자가 모델이 실제로 무엇을 고려하는지 직접 접근하게 해주어, 답변이 의학적 우려나 테스트 중임을 인식하는 것 등을 생략할 수 있는 경우에도 유효하다.

### 5.2 앞으로의 연구 시 고려할 점

1. **스트림 수 및 역할 분배 최적화:** 최적 스트림 수 $N^*$와 각 스트림의 역할 정의는 아직 경험적 설계에 의존하므로, 이를 자동화하거나 이론적으로 정당화하는 연구가 필요하다.

2. **학습 데이터 품질 관리:** 기존 단일 스트림 데이터를 멀티 스트림으로 변환하는 파이프라인의 품질이 모델 성능에 직결된다. 변환 오류가 다중 스트림 모두에 전파될 수 있어, 데이터 검증 메커니즘이 중요하다.

3. **스트림 간 동기화 문제:** 각 스트림이 독립적으로 생성되므로, 스트림 간 의미적 일관성(semantic consistency)을 유지하는 메커니즘이 필요하다. 예를 들어:

$$\mathcal{L}_{\text{consistency}} = \| f(x_t^{(\text{think})}) - g(x_t^{(\text{output})}) \|^2$$

와 같은 일관성 손실 도입을 고려할 수 있다.

4. **긴 시퀀스에서의 확장성:** 스트림 수 $N$에 비례하여 KV-cache 메모리가 증가하므로, 긴 컨텍스트 환경에서의 메모리 효율화 연구가 중요하다.

5. **멀티모달 확장:** 텍스트 스트림 외에 이미지, 오디오 등 다른 모달리티 스트림으로의 확장 가능성이 있으며, 이는 멀티모달 에이전트 연구와 자연스럽게 연결된다.

6. **보안 취약성 연구:** 다중 스트림 구조가 새로운 형태의 공격 표면을 만들 수 있는지 검토가 필요하다. 특히, LLM 효율성의 발전과 마찬가지로 오용 가능성이 있다고 저자들도 인정하므로, 멀티 스트림 환경에서의 새로운 공격 시나리오에 대한 방어 연구가 병행되어야 한다.

---

## 📚 참고 자료 (References)

1. **본 논문 (arXiv):** Su, G., Yang, Y., Li, X., & Geiping, J. (2026). *Multi-Stream LLMs: Unblocking Language Models with Parallel Streams of Thoughts, Inputs and Outputs*. arXiv:2605.12460. https://arxiv.org/abs/2605.12460
2. **본 논문 (HTML 전문):** https://arxiv.org/html/2605.12460
3. **공식 코드 저장소:** https://github.com/seal-rg/streaming
4. **관련 분석 기사:** SquaredTech, "Multi-Stream LLMs: New Breakthrough In AI Parallelization". https://www.squaredtech.co/multi-stream-llms-could-shatter-the-biggest-ai-bottleneck
5. **비교 연구 – PDT:** *Parallel Decoder Transformer: Model-Internal Parallel Decoding with Speculative Invariance via Note Conditioning*. arXiv:2512.10054. https://arxiv.org/pdf/2512.10054
6. **비교 연구 – LLM 한계 서베이:** Kostikova et al. (2025). *LLLMs: A Data-Driven Survey of Evolving Research on Limitations of Large Language Models*. arXiv:2505.19240
7. **Hacker News 토론:** https://news.ycombinator.com/item?id=48227923

> ⚠️ **정확도 고지:** 본 답변은 arXiv 공개 초록, HTML 전문, GitHub 코드 저장소 및 관련 분석 문서를 기반으로 작성되었습니다. 수식의 세부 표기법, 실험의 정확한 수치, 및 논문 내 상세 알고리즘은 논문 PDF 전문을 직접 확인하시기를 강력히 권장합니다. 논문 전문에 접근하지 못한 일부 내용(예: 구체적 실험 수치, 세부 학습 하이퍼파라미터)은 명시적으로 표시하였습니다.
