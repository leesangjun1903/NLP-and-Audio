
# Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design

> **논문 정보**
> - **저자:** Alberto Pepe, Chien-Yu Lin, Despoina Magka, Bilge Acun, Yannan Nellie Wu, Anton Protopopov, Carole-Jean Wu, Yoram Bachrach (FAIR at Meta)
> - **arXiv ID:** [2605.15871](https://arxiv.org/abs/2605.15871)
> - **제출일:** 2026년 5월 15일

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

이 논문은 재귀적 자기 개선(recursive self-improvement)을 목표로, LLM 에이전트가 표준 트랜스포머를 넘어선 파운데이션 모델을 자율적으로 설계할 수 있는지를 연구합니다.

전통적인 방법들이 경직된 최적화 목표에 제한되는 것과 달리, 이 에이전트들은 방대한 조합적 탐색 공간을 창의적이고 체계적으로 탐색합니다.

### 🏆 주요 기여

두 가지 상호보완적인 접근법인 **AIRA-Compose**와 **AIRA-Design**을 통해, Composer, LRA, Autoresearch라는 3개의 서로 다른 프레임워크로부터 파생된 12개의 새로운 다양한 에이전틱 태스크를 수행합니다.

그 결과로 **AIRAformers**와 **AIRAhybrids**라는 14개의 신규 아키텍처를 도입하며, 이들은 우수한 다운스트림 태스크 성능, 강건한 스케일링 특성, 유망한 손실-효율 트레이드오프를 나타냅니다.

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 🔍 2.1 해결하고자 하는 문제

기존 신경망 아키텍처 탐색(NAS) 연구의 주요 한계는 다음과 같습니다:

1. **탐색 공간의 경직성**: 전통적인 NAS 접근법은 전문가가 정의한 탐색 공간 내에서 고성능 신경망 아키텍처를 찾는 것을 목표로 하며, 미리 정의된 탐색 공간에 제한된다.

2. **자동화의 한계**: 전통적인 NAS는 인간이 정의한 공간 탐색에 근본적으로 제한되어 있어, 자동화된 최적화(automated optimization)에서 자동화된 혁신(automated innovation)으로의 패러다임 전환이 필요합니다.

3. **에이전트 반복 설계 능력 부재**: 기존 LLM 기반 NAD 방법들은 독립적으로 작동하며 과거 경험으로부터 학습하는 능력이 부족하여, 반복적 실수와 비효율적 탐색이 발생합니다.

---

### ⚙️ 2.2 제안하는 방법 (수식 포함)

#### 🔷 AIRA-Compose: 고수준 아키텍처 탐색

AIRA-Compose는 고정된 24시간 컴퓨팅 예산 내에서 기본 연산 프리미티브(Attention, MLP, Mamba)의 조합 설계 공간을 탐색하기 위해 11개의 에이전트 앙상블을 배포합니다.

탐색 공간은 이러한 프리미티브들의 배열로 정의되며, 에이전트는 다음과 같은 목적함수를 최적화합니다:

$$\mathcal{A}^* = \arg\min_{\mathcal{A} \in \mathcal{S}} \mathcal{L}_{\text{val}}(\mathcal{A}, \theta^*(\mathcal{A}))$$

여기서:
- $\mathcal{A}$: 후보 아키텍처
- $\mathcal{S}$: 탐색 공간 (Attention, MLP, Mamba 프리미티브의 조합)
- $\theta^*(\mathcal{A})$: 아키텍처 $\mathcal{A}$에 대해 학습된 최적 파라미터
- $\mathcal{L}_{\text{val}}$: 검증 손실 (validation loss)

에이전트들은 두 단계로 운영되며, 수백만 파라미터 규모에서 후보를 반복적으로 설계·평가한 뒤, 상위 성능 설계를 350M, 1B, 3B 파라미터 규모로 확장합니다.

**스케일링 법칙 (Scaling Law):**
Chinchilla 최적 스케일링을 기반으로 손실-연산량 관계를 다음과 같이 모델링합니다:

$$\mathcal{L}(C) = \frac{A}{N(C)^\alpha} + \frac{B}{D(C)^\beta} + \varepsilon$$

여기서:
- $C$: 총 연산량 (FLOPs)
- $N(C)$: 최적 모델 파라미터 수
- $D(C)$: 최적 학습 토큰 수
- $\alpha, \beta$: 스케일링 지수

AIRA-Compose는 고수준 아키텍처 탐색으로, 에이전트가 사전 정의된 연산 프리미티브를 기반으로 소규모에서 아키텍처를 탐색·평가하며, 최상위 성능 아키텍처만 대규모로 확장합니다. 이는 Composer 프레임워크의 소규모 아키텍처 탐색을 에이전틱 태스크로 재구성한 것입니다.

---

#### 🔶 AIRA-Design: 저수준 기계적 설계

AIRA-Design은 저수준 기계적 설계로, 에이전트가 새로운 연산 프리미티브를 직접 구현하고 효율적으로 학습시키며, 이 모델들은 Long Range Arena(LRA) 및 Autoresearch 벤치마크를 해결하도록 설계됩니다.

에이전트가 설계하는 새로운 어텐션 메커니즘은 일반적으로 다음과 같이 표현됩니다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

AIRA-Design 에이전트는 이 기본 구조에서 벗어나, 장거리 의존성(long-range dependencies)에 특화된 변형 메커니즘을 자율적으로 구현합니다. 예를 들어 선형 어텐션 계열의 경우:

$$\text{LinearAttn}(Q, K, V) = \frac{\phi(Q)\left(\phi(K)^\top V\right)}{\phi(Q)\phi(K)^\top \mathbf{1}}$$

여기서 $\phi(\cdot)$는 커널 함수(kernel function)입니다.

원샷(one-shot) 에이전트들은 유효한 제출물을 생성하지 못했으며, 이는 단일 턴 코드 생성이 기계적 설계에는 불충분하고 반복적 개선, 디버깅, 검증이 필수적임을 확인해 줍니다.

---

#### 🔷 에이전트 탐색 정책

에이전트는 (부분적) 해결책의 노드들로 이루어진 탐색 그래프를 유지하며, 각 반복마다 선택 정책을 통해 노드를 선택하고, 오퍼레이터 정책을 통해 오퍼레이터를 선택하여 노드에 적용하고, 피트니스 함수를 통해 결과 솔루션을 평가합니다.

---

### 🏗️ 2.3 모델 구조

탐색 결과, 두 가지 패밀리에 걸쳐 14개의 아키텍처가 생성됩니다: **AIRAformers** (트랜스포머 기반)와 **AIRAhybrids** (트랜스포머-맘바 혼합).

| 아키텍처 패밀리 | 기반 구조 | 특징 |
|---|---|---|
| **AIRAformers** | Transformer | Attention + MLP 최적 배열 |
| **AIRAhybrids** | Transformer + Mamba | Attention + MLP + Mamba 혼합 |

- **탐색 규모:** 수백만 파라미터 (소규모 탐색)
- **스케일업 규모:** 350M, 1B, 3B 파라미터
- **프리미티브:** Attention, MLP, Mamba 블록의 조합

---

### 📊 2.4 성능 향상

1B 규모에서 고정 토큰 예산으로 사전 학습 시, 에이전트가 발견한 최상위 아키텍처들은 Llama 3.2와 Composer 기반 대안 모두를 일관되게 능가합니다. 다운스트림 태스크에서 AIRAformer-D와 AIRAhybrid-D는 Llama 3.2 대비 각각 2.4%, 3.8%의 정확도 향상을 달성합니다.

AIRA-Compose는 더 가파르고 효율적인 계산-최적 스케일링 프론티어를 달성하는 새로운 아키텍처를 발견합니다. AIRAformer-C는 Llama 3.2 및 최고의 Composer 트랜스포머보다 각각 54%, 71% 빠르게 스케일링하며, AIRAhybrid-C는 Nemotron-2와 최고의 Composer 하이브리드보다 각각 23%, 37% 빠르게 스케일링합니다.

Long Range Arena 벤치마크에서 에이전트 설계 아키텍처는 문서 매칭 및 텍스트 분류에서 인간 최신 기술(state-of-the-art) 대비 2.3%, 2.6% 이내의 정확도에 도달합니다.

---

### ⚠️ 2.5 한계점

1. **유효 제출률(VSR) 격차**: 유효 제출률(VSR)은 에이전트 모델에 따라 크게 달라지며, Greedy Opus 4.6은 항상 솔루션을 제출하는 반면 약한 모델들은 10% 미만의 VSR을 보입니다.

2. **확장 설정 난이도**: 구성 가능(Configurable) 설정에서 VSR이 일반적으로 더 낮았으며, 이는 확장된 하이퍼파라미터 탐색 공간이 유효한 코드 생성의 난이도를 높임을 나타냅니다.

3. **일반화 격차(Generalization Gap)**: 기존 연구에서 AI 연구 에이전트의 구조적 성능 병목으로 검증 기반 선택이 확장된 탐색 범위에 걸쳐 성능 저하를 유발하는 일반화 격차(generalization gap)가 확인되었습니다.

4. **단일 턴 한계**: 원샷 에이전트들은 유효한 제출물을 전혀 생성하지 못하였으며, 이는 기계적 설계에서 단일 턴 코드 생성만으로는 불충분함을 보여줍니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 📈 3.1 스케일링을 통한 일반화

에이전트들은 수백만 파라미터 후보를 평가하고 최상위 설계를 350M, 1B, 3B 규모로 외삽(extrapolating)하며, 이를 통해 AIRAformers와 AIRAhybrids의 두 패밀리가 생성되며 1B 규모에서 사전 학습 시 Llama 3.2와 Composer 기반 베이스라인을 일관되게 능가합니다.

즉, 소규모에서 발견된 아키텍처의 우수성이 대규모에서도 유지됨으로써 **스케일-일반화(scale generalization)** 성질을 보입니다.

### 📈 3.2 LRA 벤치마크를 통한 일반화 검증

LRA 벤치마크는 서로 다른 모델들이 장거리 의존성을 어떻게 포착하는지 평가하는 것이 핵심 목적으로, 관계 모델링 능력, 계층적/공간적 구조, 일반화 능력 등 모델의 다양한 측면을 평가합니다.

AIRA-Design이 LRA에서 인간 SOTA에 근접하는 성능을 보임은, 에이전트 설계 아키텍처가 단순한 과적합이 아닌 실질적인 장거리 패턴 포착 능력을 갖추고 있음을 시사합니다.

### 📈 3.3 일반화 격차 문제와 대응

기존 AI 연구 에이전트의 구조적 성능 병목으로, 검증 기반 선택이 과적합(overfitting)을 유발하여 확장된 탐색 범위에서 성능이 저하되는 일반화 격차(generalization gap), 고정 단일 턴 LLM 오퍼레이터가 탐색 성능에 상한을 부과하는 문제가 있습니다.

이에 대한 대응 방향으로, AIRA₂는 이러한 병목을 해결하기 위해 실험 처리량을 선형으로 증가시키는 비동기 멀티-GPU 워커 풀, 신뢰할 수 있는 평가 신호를 제공하는 Hidden Consistent Evaluation 프로토콜, 그리고 행동 범위를 동적으로 조정하고 대화형으로 디버깅하는 ReAct 에이전트를 도입합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 🌟 4.1 연구에 미치는 영향

#### (1) 재귀적 자기 개선(Recursive Self-Improvement)의 현실화
이 연구는 재귀적 자기 개선을 위한 한 단계로서, LLM 에이전트가 표준 트랜스포머 패러다임을 넘어선 파운데이션 모델을 자율적으로 설계하는 능력을 탐구합니다.

이는 향후 AI가 스스로 더 나은 AI를 설계하는 루프를 구성하는 연구의 기반이 됩니다.

#### (2) NAS 패러다임의 전환
전통적인 NAS의 한계를 넘어 자동화된 최적화에서 자동화된 혁신으로의 패러다임 전환이 가능해지며, 에이전트는 새로운 아키텍처 개념을 자율적으로 가설화하고, 실행 가능한 코드로 구현하며, 엄격한 실험을 통해 성능을 경험적으로 검증합니다.

#### (3) 하이브리드 LLM 연구 촉진
AIRA-Compose는 미래 하이브리드 LLM을 위한 에이전틱 신경망 아키텍처 발견의 두 가지 접근법 중 하나로서, 사전 정의된 연산 프리미티브를 기반으로 새로운 모델 아키텍처를 탐색합니다.

#### (4) 멀티-에이전트 협업의 가능성
진화적 에이전트에 인식 기반(cognition base)과 분석기(analyzer)를 결합하는 연구가 이어지고 있으며, AI 주도 발견을 데이터, 아키텍처, 학습 알고리즘의 세 가지 핵심 AI 개발 요소에 걸쳐 통합하는 방향으로 발전하고 있습니다.

---

### 🔬 4.2 앞으로 연구 시 고려할 점

#### ① 일반화 격차(Generalization Gap) 해결
검증-테스트 발산(validation-test divergence)이 탐색 신호를 오도하여 확장된 연구 범위에서 과적합이 발생하는 것이 구조적 병목으로 정형화되었으며, 이를 극복하기 위한 평가 프로토콜 설계가 필수적입니다.

#### ② 연산 효율성 및 탐색 비용
AIRA-Compose는 340회의 24시간 실행과 300회의 60시간 실행에 걸쳐 11개의 에이전트를 배포하며, 이러한 막대한 연산 비용을 줄이기 위한 효율적인 프록시 태스크(proxy task) 설계가 중요한 연구 과제입니다.

#### ③ 에이전트 역량의 편차
유효 제출률이 에이전트 모델에 따라 Greedy Opus 4.6의 100%에서 약한 모델의 10% 미만까지 크게 달라지므로, 에이전트의 기반 모델 역량에 따른 아키텍처 품질의 편차를 줄이는 연구가 필요합니다.

#### ④ 환경 다양성과 분포 외 일반화
일반화 가능한 에이전트는 훈련 분포를 넘어 다양한 태스크와 미지의 환경에 적응할 수 있어야 하며, 이를 위해 고정된 벤치마크 내에서 단순히 경로나 태스크를 늘리는 것이 아닌 환경 스케일링(environment scaling)이 필요합니다.

#### ⑤ 프리미티브 확장 및 저수준 설계 통합
두 접근법은 단일 목표에 대한 상호보완적 관점을 나타내며, 전자(AIRA-Compose)는 사전 정의된 프리미티브에 의존하여 에이전트의 자유를 배열 최적화에만 제한합니다. 따라서 새로운 프리미티브 자동 발견과 고수준 탐색의 통합이 중요한 향후 과제입니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 핵심 방법 | 탐색 방식 | 주요 특징 |
|---|---|---|---|
| **AIRA-Compose/Design** (2026) | LLM 에이전트 기반 아키텍처 탐색 | 고수준 + 저수준 이중 접근 | AIRAformer/AIRAhybrid 14종 발굴, Llama 3.2 대비 최대 3.8% 향상 |
| **NADER** (2024) | 멀티 에이전트 협업 | 그래프 기반 표현 | 기반 아키텍처를 반복 수정하는 특화 에이전트 팀을 활용하며, 과거 경험 학습 부재로 반복 실수가 생기는 문제를 해결하기 위해 즉각적 피드백과 장기 경험으로부터 학습하는 Reflector를 제안합니다. |
| **ASI-ARCH** (2025) | 자율 아키텍처 혁신 | 종단간 과학 연구 | 1,773회의 자율 실험을 통해 20,000 GPU 시간 동안 106개의 혁신적 선형 어텐션 아키텍처를 발견합니다. |
| **ASI-Evolve** (2025) | 진화적 에이전트 + 인식 기반 | AI 가속 AI | 신경망 아키텍처 설계에서 105개의 SOTA 선형 어텐션 아키텍처를 발견하였으며, 최고 모델은 DeltaNet 대비 +0.97점을 기록했습니다. |
| **AIRA₂** (2026) | 비동기 멀티-GPU + ReAct | 병렬 탐색 | 비동기 멀티-GPU 워커 풀, Hidden Consistent Evaluation 프로토콜, ReAct 에이전트의 세 가지 아키텍처 선택을 통해 구조적 병목을 해결합니다. |

---

## 📚 참고 자료 및 출처

1. **논문 원문:** Alberto Pepe et al., *"Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design"*, arXiv:2605.15871, May 2026. → https://arxiv.org/abs/2605.15871
2. **논문 HTML 전문:** https://arxiv.org/html/2605.15871
3. **관련 연구 - NADER:** *"NADER: Neural Architecture Design via Multi-Agent Collaboration"*, arXiv:2412.19206 → https://arxiv.org/pdf/2412.19206
4. **관련 연구 - ASI-ARCH:** *"AlphaGo Moment for Model Architecture Discovery"*, arXiv:2507.18074 → https://arxiv.org/pdf/2507.18074
5. **관련 연구 - ASI-Evolve:** *"ASI-Evolve: AI Accelerates AI"*, arXiv:2603.29640 → https://arxiv.org/pdf/2603.29640
6. **관련 연구 - AIRA₂:** *"AIRA₂: Overcoming Bottlenecks in AI Research Agents"*, arXiv:2603.26499 → https://arxiv.org/pdf/2603.26499
7. **관련 연구 - AIRA (MLE-bench):** *"AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench"*, arXiv:2507.02554 → https://arxiv.org/html/2507.02554v2
8. **LRA 벤치마크:** Tay et al., *"Long Range Arena: A Benchmark for Efficient Transformers"*, arXiv:2011.04006 → https://ar5iv.labs.arxiv.org/html/2011.04006

> ⚠️ **정확도 유의사항:** 본 논문(arXiv:2605.15871)은 2026년 5월 15일 제출된 최신 논문으로, 공개된 arXiv 초록 및 HTML 전문을 기반으로 분석하였습니다. 수식 중 스케일링 법칙 및 어텐션 표현은 논문의 맥락에서 일반적으로 적용되는 수식을 활용한 것으로, 논문 내 명시된 특정 수식과 세부 기호는 원문을 직접 확인하시기를 권장합니다.
