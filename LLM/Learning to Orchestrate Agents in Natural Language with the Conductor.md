
# Learning to Orchestrate Agents in Natural Language with the Conductor

> **논문 정보**: Stefan Nielsen, Edoardo Cetin, Peter Schwendeman, Qi Sun, Jinglue Xu, Yujin Tang (Sakana AI / University of Michigan)
> **arXiv**: [2512.04388](https://arxiv.org/abs/2512.04388) | **학회**: ICLR 2026 accepted

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **RL로 훈련된 새로운 Conductor 모델**을 소개하며, Conductor는 효과적인 에이전트 간 협업을 위한 통신 토폴로지 설계뿐만 아니라, 각 LLM의 개별 역량을 최대한 활용하도록 집중된 지시를 생성하는 프롬프트 엔지니어링도 학습합니다.

더 넓게 보면, 이 논문은 언어 모델 간 협력이 RL을 통해 활성화될 수 있음을 보이는 초기 연구 중 하나이며, 강력한 협조 전략이 순수 end-to-end 보상 최대화를 통해 LLM에서 자연스럽게 출현함을 보여줍니다.

### 📌 세 가지 핵심 기여

**(1) RL Conductor 도입**: 도전적인 문제를 분할하고, 타겟화된 서브태스크를 위임하며, 워커 LLM 집합에 대한 통신 토폴로지를 설계하는 end-to-end 강화학습으로 훈련된 언어 모델.

**(2) 무작위화된 에이전트 풀 학습을 통한 일반화**: 각 학습 스텝마다 무작위화된 에이전트 풀로 학습함으로써, 오픈소스·클로즈드소스 워커의 임의 조합에도 일반화 가능.

**(3) 재귀적 토폴로지 및 테스트 타임 스케일링**: Conductor가 스스로를 워커로 선택할 수 있게 함으로써 재귀적 토폴로지가 발생하며, 온라인 반복 적응을 통한 새로운 형태의 동적 테스트 타임 스케일링을 구현.

---

## 2. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

수동으로 설계된 에이전틱 워크플로우는 상업적 AI 제품의 핵심 구성 요소이지만, 효과적인 프롬프팅과 자기 개선 전략은 여전히 현재 연구의 핵심 과제로 남아있습니다.

또한 서로 다른 모델들은 특정 데이터셋과 도메인에 특화되어 파인튜닝되어 있어, 모든 태스크에 걸쳐 단일한 최적 LLM이 존재하지 않는다는 문제가 있습니다.

기존 방법들은 수동 에이전틱 워크플로우나 제한된 라우팅 정책에 의존하는 반면, Conductor는 도전적인 문제를 동적으로 분할하고, 목표한 서브태스크를 위임하며, 워커 LLM을 위한 통신 토폴로지를 순수 자연어로 설계하는 LLM 자체입니다.

---

### 2-2. 제안하는 방법 (수식 포함)

핵심 방법론은 에이전트 조정을 강화학습으로 최적화되는 순차적 의사결정 문제로 정의합니다. Conductor의 목표는 각 입력 질문 $q_i$에 특화된 에이전틱 워크플로우를 생성하여 간접적으로 태스크를 해결하는 것입니다.

DeepSeek R1 계열의 레시피를 따라, LLM $\pi_{\theta}$를 커스텀 시스템 프롬프트로 최적화하며, 검증 가능한 문제 집합 $D=(q_1,s_1),\ldots,(q_N,s_N)$에 대해 모델 자체의 완성본 $o_i$를 생성하게 합니다.

보상 함수는 다음과 같이 정의됩니다:

각 출력에 대한 보상 $r_i$는 두 조건으로 결정됩니다:
1. **포맷 조건**: 지정된 `<think>/<solution>` 형식을 준수하지 않는 출력에 대해 $r_i = -1$
2. **정확도 조건**: 올바르게 포맷된 출력이 정답 $s_i$와 일치하면 $r_i = 1$, 그렇지 않으면 $r_i = -0.5$

이를 수식으로 정리하면:

$$r_i = \begin{cases} -1 & \text{포맷 불일치 시} \\ 1 & \text{포맷 일치 + 정답 일치 시} \\ -0.5 & \text{포맷 일치 + 오답 시} \end{cases}$$

이 보상을 활용하여 모델은 GRPO(Shao et al., 2024)라는 단순한 온라인 RL 알고리즘으로 학습됩니다.

---

### 2-3. 모델 구조

RL Conductor는 동적으로 문제를 분할하고, 타겟화된 서브태스크를 위임하며, 워커 LLM 에이전트 집합의 통신 토폴로지를 설계하도록 RL로 훈련된 새로운 추론 모델입니다. 모델 자체가 LLM이며, 각 워크플로우 단계(자연어 지시, 담당 에이전트 배정, 에이전트들 간의 가시성 설계)의 시퀀스를 출력합니다.

단순한 사실적 질문에는 모델 하나만 쿼리하고, 어려운 코딩 문제에는 플래너, 코더, 검증자의 전체 파이프라인을 자율적으로 구성합니다.

이 구조는 Conductor가 각 입력 문제에 완전히 맞춤화된 에이전틱 워크플로우를 구성하도록 하며, 프롬프트 엔지니어링, 정제(refinement), 메타 프롬프트 최적화 같은 공통 전략이 end-to-end 보상 최대화로부터 자연스럽게 출현합니다.

학습에는 NVIDIA H100 80GB GPU 2개를 사용합니다.

---

### 2-4. 성능 향상

7B Conductor는 강력한 워커 LLM 풀(Gemini 2.5 Pro, Claude Sonnet 4, GPT-5, DeepSeek-R1-Distill-Qwen-32B 등)을 활용하여, **LiveCodeBench**와 **GPQA Diamond** 같은 어려운 추론 벤치마크에서 SOTA 결과를 달성하며, 개별 워커, 자기 반성 전략, 기존 다중 에이전트 협업 베이스라인을 훨씬 적은 비용으로 능가합니다.

이러한 결과는 훈련 태스크 집합을 훨씬 넘어서, 수학·코딩·자연과학 등 광범위한 도메인에 걸쳐 유지되며, 수동 에이전틱 스캐폴드를 일반적이고 견고한 end-to-end 접근 방식으로 대체할 수 있는 모델의 가능성을 보여줍니다.

---

### 2-5. 한계

더 복잡한 세밀한 제어 방식이 유의미한 성능 향상을 추가적으로 제공하지 못하며, 저자들은 7B를 넘는 더 크고 지능적인 Conductor를 통해 강력한 토폴로지 발견으로 추가적인 성능 향상이 가능할 것이라는 점을 미래 연구 과제로 남겨둡니다.

기본 RL만으로는 소규모 오픈소스 LLM에서 고급 다단계 오케스트레이션의 출현이 쉽지 않으며, 에이전트 풀이 태스크 다양성과 맞지 않을 경우 비용 오버헤드가 지속될 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

**무작위화된 에이전트 풀을 통한 학습**으로 Conductor는 오픈소스·클로즈드소스 에이전트의 임의 집합에 효과적으로 적응하여 사용자 요구 사항을 충족할 수 있습니다.

또한, 사전 훈련된 Conductor를 두 가지 추가 기술로 파인튜닝하여 커스텀 사용자 요구 사항과 테스트 타임 연산 성능을 더욱 향상시킬 수 있으며, 무작위화된 에이전트 풀 훈련으로 오픈소스·클로즈드소스 워커의 임의 조합에 일반화됩니다.

본 프레임워크는 순수 end-to-end RL을 통해 강력한 에이전트 조정 전략을 학습함으로써 Conductor가 자연어로 표현 가능한 모든 전략을 자유롭게 학습할 수 있다는 점에서 기존 접근 방식과 차별화됩니다.

재귀적 토폴로지를 통해, Conductor가 스스로를 워커 LLM으로 지정할 수 있게 함으로써 새로운 종류의 재귀 토폴로지가 발생하고, 추론 모델에서 조정 가능한 새로운 추론 시간 스케일링 축이 열립니다.

이를 그림으로 표현하면:

$$\underbrace{\text{Conductor}(7\text{B})}_{\text{Orchestrator}} \xrightarrow{\text{workflow}} \begin{cases} \text{Worker}_1 \text{(GPT-5)} \\ \text{Worker}_2 \text{(Gemini 2.5 Pro)} \\ \text{Worker}_3 \text{(Conductor itself)} \leftarrow \text{재귀} \end{cases}$$

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 연구 영향

| 영역 | 내용 |
|------|------|
| **RL 기반 조정** | 언어 모델 조정이 RL을 통해 활성화될 수 있으며, 강력한 조정 전략이 순수 end-to-end 보상 최대화를 통해 자연스럽게 출현함을 보인 초기 연구. |
| **테스트 타임 스케일링** | Conductor가 스스로를 워커로 선택하는 재귀적 토폴로지는 온라인 반복 적응을 통한 동적 테스트 타임 스케일링이라는 새로운 형태를 제시. |
| **범용 에이전트 오케스트레이션** | 이 구조는 프롬프트 엔지니어링, 정제, 심지어 메타 프롬프트 최적화 같은 공통 전략이 end-to-end 보상 최대화로부터 자연스럽게 출현하는 완전히 유연한 에이전틱 워크플로우 구성을 가능하게 합니다. |

### 4-2. 관련 최신 연구 비교 (2020년 이후)

최신 관련 연구들을 살펴보면, Wang et al.(2024)과 Du et al.(2023)은 에이전트를 오케스트레이션하기 위한 수동 설계 스캐폴드를 제안했고, Zhuge et al.(2024)는 협력을 학습 가능한 그래프로 처리하였으며, Chen et al.(2024)는 단일 최적 에이전트로 쿼리를 라우팅하는 라우터를 학습하였습니다.

본 프레임워크는 순수 end-to-end RL을 통해 에이전트 조정 전략을 학습한다는 점에서 기존 모든 접근 방식과 차별화되며, Conductor는 자연어로 표현 가능한 어떠한 전략이든 자유롭게 학습할 수 있습니다.

**FlowSteer**(arXiv:2602.01664, 2026)와의 비교:
FlowSteer는 고정된 연산자나 단일 강력한 LLM 백엔드에 의존하는 기존 접근 방식에서 오는 경로 고착(path lock-in) 문제와 희소하고 불안정한 학습 신호 문제를 해결하고자 하며, 이는 Conductor가 다루는 문제와 유사하지만 연산자 구성 측면에서 다른 방향을 취합니다.

### 4-3. 앞으로 연구 시 고려할 점

1. **더 큰 Conductor 규모 탐구**: 7B를 넘는 더 크고 지능적인 Conductor를 통해 강력한 토폴로지 발견으로 추가적인 성능 향상이 가능할 것으로 기대됩니다.

2. **비용 효율성**: Conductor 기반 시스템은 비용 인식 RL 목표와 난이도에 따른 실행 조정(예: 쉬운 쿼리에 대해 깊은 파이프라인 생략)을 통해 비용 효율적 운용을 실현할 수 있습니다.

3. **보상 설계의 정교화**: 표준 RL 접근 방식은 희소하고 이진적인 보상 문제를 겪으며, 이를 해결하기 위한 점진적 보상 설계(graduated reward design)가 중요한 연구 방향으로 부상하고 있습니다.

4. **멀티 도메인 일반화 검증**: 에이전트 능력 프로파일이나 스킬 태그를 라우팅에 명시적으로 반영하면 잘못된 배정을 줄이고 Conductor 성능이 실질적으로 향상됩니다.

---

## 📚 참고자료

| # | 출처 |
|---|------|
| 1 | **arXiv 논문 원문**: Nielsen, S., Cetin, E., Schwendeman, P., Sun, Q., Xu, J., Tang, Y. "Learning to Orchestrate Agents in Natural Language with the Conductor." arXiv:2512.04388, Dec 2025. [https://arxiv.org/abs/2512.04388](https://arxiv.org/abs/2512.04388) |
| 2 | **Sakana AI 공식 블로그**: "Learning to Orchestrate Agents in Natural Language with the Conductor." [https://sakana.ai/learning-to-orchestrate/](https://sakana.ai/learning-to-orchestrate/) |
| 3 | **OpenReview (ICLR 2026)**: [https://openreview.net/pdf/4a133f1e2ca67ceaedb45c3a123cc8125c694ff5.pdf](https://openreview.net/pdf/4a133f1e2ca67ceaedb45c3a123cc8125c694ff5.pdf) |
| 4 | **ResearchGate**: [https://www.researchgate.net/publication/398357486](https://www.researchgate.net/publication/398357486) |
| 5 | **Moonlight Literature Review**: [https://www.themoonlight.io/en/review/learning-to-orchestrate-agents-in-natural-language-with-the-conductor](https://www.themoonlight.io/en/review/learning-to-orchestrate-agents-in-natural-language-with-the-conductor) |
| 6 | **EmergentMind (관련 연구 비교)**: "Conductor-Based Orchestration of LLMs." [https://www.emergentmind.com/topics/conductor-based-orchestration-of-llms](https://www.emergentmind.com/topics/conductor-based-orchestration-of-llms) |
| 7 | **FlowSteer 비교 논문**: arXiv:2602.01664, "FlowSteer: Interactive Agentic Workflow Orchestration via End-to-End Reinforcement Learning." [https://arxiv.org/html/2602.01664](https://arxiv.org/html/2602.01664) |
| 8 | **Shao et al., 2024**: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." (GRPO 알고리즘 출처) |

> ⚠️ **정확도 안내**: 논문의 전체 수식 체계(특히 GRPO의 세부 목적함수)는 arXiv 원문에 명시된 범위 내에서만 인용하였습니다. 세부 아키텍처 하이퍼파라미터 등 원문에서 확인되지 않은 정보는 의도적으로 제외하였습니다.
