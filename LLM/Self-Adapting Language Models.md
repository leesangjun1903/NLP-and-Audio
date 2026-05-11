# Self-Adapting Language Models

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

대형 언어 모델(LLM)은 사전학습 이후 **정적(static)** 상태로 고정되어 있어, 새로운 지식·태스크·예시에 대해 가중치를 스스로 적응시키는 메커니즘이 없다. SEAL(Self-Adapting LLMs)은 LLM이 **자신의 파인튜닝 데이터와 업데이트 지시문(self-edit)을 직접 생성**하여 가중치를 스스로 갱신할 수 있도록 하는 프레임워크를 제안한다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **Self-Edit 개념 도입** | 모델이 자연어로 자신의 훈련 데이터 및 최적화 하이퍼파라미터를 생성 |
| **RL 기반 자기 적응 루프** | 업데이트 후 성능을 보상 신호로 활용하는 이중 루프(outer RL + inner SFT) 설계 |
| **도메인 불가지론적 프레임워크** | 지식 통합(Knowledge Incorporation)과 퓨샷 학습(Few-Shot Learning) 모두 적용 |
| **GPT-4.1 초과 성능** | 7B 모델이 GPT-4.1이 생성한 합성 데이터를 초과하는 성능 달성 |
| **Continued Pretraining 일반화** | 단일 예시 TTT 환경에서 훈련했음에도 대규모 지속 사전학습 설정에 일반화 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

현재 LLM의 적응 방식(파인튜닝 또는 in-context learning)은 다음 한계를 가진다:

- 입력 데이터를 **있는 그대로(as-is)** 학습하여, 데이터가 학습에 최적화된 형태가 아닐 수 있음
- 모델이 **데이터를 어떻게 변환·재구조화**하면 더 효율적으로 학습할 수 있는지 스스로 결정하지 못함
- 별도의 적응 모듈이나 보조 네트워크에 의존하는 기존 접근법은 범용성이 제한적

SEAL은 이를 인간 학생의 노트 필기 비유로 설명한다: 강의 내용을 그대로 암기하는 것보다, 자신만의 방식으로 재해석하고 정리하는 것이 더 효과적인 학습을 가능하게 한다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### 전체 프레임워크

$\theta$를 언어 모델 $\text{LM}_\theta$의 파라미터라 하자. SEAL은 각 태스크 인스턴스 $(C, \tau)$에 대해:

- $C$: 태스크 관련 컨텍스트 (예: 새로운 사실이 담긴 문단)
- $\tau$: 다운스트림 평가 정의 (예: QA 쌍)

모델은 self-edit $\text{SE}$를 생성하고, SFT로 파라미터를 업데이트한다:

$$\theta' \leftarrow \text{SFT}(\theta, \text{SE})$$

#### RL 목적함수 (Outer Loop)

$$\mathcal{L}_{\text{RL}}(\theta_t) := -\mathbb{E}_{(C,\tau) \sim \mathcal{D}} \left[ \mathbb{E}_{\text{SE} \sim \text{LM}_{\theta_t}(\cdot | C)} \left[ r(\text{SE}, \tau, \theta_t) \right] \right] $$

#### 이진 보상 함수

$$r(\text{SE}, \tau, \theta_t) = \begin{cases} 1 & \text{if on } \tau, \text{ adaptation using SE improves LM}_{\theta_t}\text{'s performance} \\ 0 & \text{Otherwise} \end{cases} $$

#### 몬테카를로 그래디언트 추정 (ReSTEM 근사)

$N$개 컨텍스트, 컨텍스트당 $M$개 self-edit 샘플에 대한 추정:

```math
\nabla_{\theta_t} \mathcal{L}_{\text{RL}} \approx -\frac{1}{NM} \sum_{i=1}^{N} \sum_{j=1}^{M} r_{ij} \nabla_{\theta_t} \log p_{\theta_t}(\text{SE}_{ij} | C_i)
```

이를 자기회귀 분해하면:

$$= -\frac{1}{NM} \sum_{i=1}^{N} \sum_{j=1}^{M} r_{ij} \sum_{s=1}^{T} \nabla_{\theta_t} \log p_{\theta_t}(y_s^{(i,j)} | y_{ < s}^{(i,j)}, C_i) $$

여기서 $p_{\theta_t}$는 모델의 자기회귀 분포, $y_s^{(i,j)}$는 $\text{SE}_{ij}$의 $s$번째 토큰이다.

> **ReSTEM (Rejection Sampling + SFT)**: E-step에서 현재 정책으로부터 후보 출력을 샘플링하고, M-step에서 양의 보상을 받은 샘플만 SFT로 강화. $r=0$인 시퀀스는 식 (4)에서 무시되므로, 사실상 "좋은 self-edit에 대한 SFT"가 된다.

#### SEAL 알고리즘 (Algorithm 1)

```
Input: LM_θ, dataset D = {(C, τ)}
for outer iteration t = 1, 2, ... do
    Sample (C, τ) ~ D
    Generate self-edit SE ~ LM_θ(· | C)
    Inner Loop Update: θ'_t ← SFT(θ_t, SE)
    Evaluate: Ans ~ LM_θ'_t(· | τ)
    Compute reward: r ← r(Ans, τ)
    Update: θ_{t+1} ← RL_Update(θ_t, r, SE)
end for
```

---

### 2-3. 모델 구조

SEAL은 두 개의 중첩 루프로 구성된다:

```
┌─────────────────────────────────────────────────┐
│              Outer RL Loop                       │
│  (self-edit 생성 정책 최적화 - ReSTEM)           │
│                                                  │
│   ┌─────────────────────────────────────────┐   │
│   │           Inner Update Loop             │   │
│   │  (생성된 self-edit로 LoRA SFT 수행)      │   │
│   │  θ' ← SFT(θ, SE)                       │   │
│   └─────────────────────────────────────────┘   │
│                                                  │
│   보상: 업데이트된 모델의 τ에 대한 성능          │
└─────────────────────────────────────────────────┘
```

#### 도메인별 Self-Edit 형태

**① 지식 통합 (Knowledge Incorporation)**

- 모델: `Qwen2.5-7B`
- Self-Edit 형태: 문단의 함의(implications) 리스트 생성
- 업데이트 방법: LoRA (rank 64, alpha 128)
- 평가: passage 없이 QA 수행 (no-context SQuAD)

**② 퓨샷 학습 (Few-Shot Learning on ARC)**

- 모델: `Llama-3.2-1B-Instruct`
- Self-Edit 형태: JSON 형식의 증강 전략 + 하이퍼파라미터 명세
  - 데이터 증강: `use_basic_augmentations`, `use_size_augmentations`, `use_chain_augmentations`, `use_repeat_augmentations`
  - 최적화: learning rate, num_train_epochs, loss 계산 방식
- 업데이트 방법: LoRA (rank 128, alpha 16)
- 평가: held-out test input에 대한 정답률

---

### 2-4. 성능 향상

#### 지식 통합 성능 (SQuAD no-context QA)

| 방법 | Single Passage (n=1, LoRA) | CPT (n=200, full-FT) | CPT (n=2067, full-FT) |
|---|---|---|---|
| Base model | 32.7% | 32.7% | 29.0% |
| Train on Passage Only | 33.5% | 36.0% | 31.2% |
| Train on Passage + Synthetic | 39.7% | 50.6% | 43.4% |
| Train on Passage + GPT-4.1 Synthetic | 46.3% | **59.4%** | **49.2%** |
| **SEAL** | **47.0%** | 58.2% | 46.4% |

- SEAL은 단일 패시지 설정에서 GPT-4.1 합성 데이터를 **초과**하는 성능 달성 (47.0% vs 46.3%)
- 단 **2라운드의 ReSTEM**으로 GPT-4.1을 능가

#### 퓨샷 추상 추론 성능 (ARC subset)

| 방법 | 성공률 (%) |
|---|---|
| ICL (no adaptation) | 0% |
| TTT + Self-Edit (RL 사전학습 없음) | 20% |
| **SEAL** | **72.5%** |
| Oracle TTT (인간이 최적 설정) | 100% |

---

### 2-5. 한계점

#### ① 재앙적 망각 (Catastrophic Forgetting)
연속적인 self-edit 적용 시, 이전 태스크에 대한 성능이 점진적으로 저하됨. 논문 Figure 6에서 passage index가 증가할수록 초기 태스크 성능 감소가 관찰됨.

**잠재적 해결책 (논문 제시):**
- Reward shaping으로 이전 태스크 회귀 페널티 부과
- Null-space constrained edits (AlphaEdit)
- Representational Superposition
- SFT 대신 RL 내부 루프 활용 (RL이 SFT보다 망각이 적음)

#### ② 계산 오버헤드
각 self-edit 평가에 전체 모델의 파인튜닝 + 추론이 필요하여, 단일 self-edit 평가에 **30~45초** 소요. 1라운드 ReSTEM에 **약 6시간** (2×H100 기준).

#### ③ 컨텍스트 의존적 평가
현재 구현은 모든 컨텍스트에 명시적 다운스트림 태스크(QA 쌍 등)가 필요. 레이블 없는 코퍼스로의 확장이 제한적.

#### ④ 규모의 제약
실험이 소규모 설정(11개 ARC 태스크, 7B 파라미터 모델)에서만 수행됨. 더 큰 모델 및 더 넓은 도메인에 대한 검증 필요.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 단일 태스크 → 지속적 사전학습 일반화

SEAL의 가장 중요한 일반화 증거는 **훈련 분포를 벗어난 설정에서의 성능 유지**이다:

- **훈련 환경**: 단일 패시지에 대한 단일 self-edit 생성 (single-example TTT)
- **테스트 환경**: 200개 또는 2,067개 문서에 대한 지속적 사전학습 (Continued Pretraining)

이 설정에서 SEAL은 단순히 평균적인 성능을 유지하는 것을 넘어, **n=200 CPT에서 58.2%**로 단일 패시지(47.0%)보다 오히려 향상되는 모습을 보였다. 이는 self-edit 정책이 단일 예시 생성에 특화되지 않고, 다양한 문서에 걸쳐 일반화되는 효율적인 데이터 변환 전략을 학습했음을 시사한다.

$$\text{일반화 이득} = \underbrace{58.2\%}_{\text{CPT } n=200} - \underbrace{47.0\%}_{\text{Single Passage}} = +11.2\%p$$

### 3-2. 퓨샷 설정에서의 일반화

- **훈련**: 11개 ARC 태스크
- **평가**: 훈련에 포함되지 않은 8개 held-out ARC 태스크

SEAL은 훈련하지 않은 태스크에 대해 **72.5%** 성공률을 달성. 이는 단순한 하이퍼파라미터 암기가 아닌, 태스크 구조에 기반한 적응 전략을 학습했음을 의미한다.

### 3-3. 모델 규모 스케일링과 일반화

| 모델 | Base (No Training) | Base Self-Edit | SEAL | SEAL 이득/Base Self-Edit 이득 비율 |
|---|---|---|---|---|
| Qwen2.5-3B | 25.1% | 31.9% | 37.0% | 1.75× |
| Qwen2.5-7B | 32.7% | 39.7% | 47.0% | **2.04×** |

더 큰 모델에서 RL을 통한 self-edit 최적화의 **상대적 이득이 더 크게 나타남**. 이는 모델 용량이 클수록 RL의 자기 적응 효과가 복리적으로 증가할 가능성을 시사한다 (단, 저자들도 더 광범위한 스케일링 실험이 필요하다고 인정).

### 3-4. 다양한 프롬프트 형식에 대한 일반화

| 프롬프트 형식 | RL 이전 | Round 1 | Round 2 | 향상폭 |
|---|---|---|---|---|
| Implications | 39.7% | 43.7% | 47.0% | +7.3%p |
| Implications-long | 49.3% | 52.4% | 54.4% | +5.1%p |
| Rewrite | 49.4% | 55.3% | 55.6% | +6.2%p |
| Self-QA | 37.3% | 42.8% | 48.7% | +11.4%p |

**모든 프롬프트 형식에 대해 ReSTEM이 6~11%p의 일관된 성능 향상을 제공**하므로, SEAL의 일반화 능력이 특정 프롬프트에 과적합된 것이 아님을 보여준다.

### 3-5. 일반화의 메커니즘적 이해

SEAL의 일반화는 메타학습(Meta-Learning) 관점에서 해석할 수 있다:

- **외부 루프(RL)**: "어떻게 효과적인 self-edit를 생성할 것인가"라는 **적응 전략 자체**를 학습
- **내부 루프(SFT)**: 특정 컨텍스트에 대한 실제 적응 수행

이 구조는 MAML(Model-Agnostic Meta-Learning)과 개념적으로 유사하지만, 그래디언트 기반 메타 업데이트 대신 자연어 생성을 통해 적응 과정을 매개변수화한다는 점에서 차별화된다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4-1. Test-Time Training (TTT) 계열

| 논문 | 방법 | SEAL과의 차이점 |
|---|---|---|
| Sun et al. (2020) "Test-Time Training with Self-Supervision..." | 분포 이동 하에서 자기지도 학습으로 가중치 임시 적응 | 적응 전략이 고정; SEAL은 전략 자체를 RL로 학습 |
| Sun et al. (2024) "Learning to (Learn at Test Time)" | TTT 전략을 메타학습 | SEAL과 가장 유사하지만, LLM이 아닌 비전 태스크 중심 |
| Akyürek et al. (2025) "The Surprising Effectiveness of TTT for Few-Shot Learning" | 수동으로 설계된 증강+TTT | SEAL은 이 방법의 증강 선택을 자동화 (SEAL의 내부 루프의 기반) |

### 4-2. 지식 업데이트 / 모델 편집 계열

| 논문 | 방법 | SEAL과의 차이점 |
|---|---|---|
| Meng et al. (2022) "Locating and Editing Factual Associations in GPT" (ROME) | 특정 파라미터를 직접 수정 | 단일 사실에 특화, 확장성 제한; SEAL은 합성 데이터 경유 |
| Meng et al. (2023) "Mass-Editing Memory in a Transformer" (MEMIT) | 대규모 사실 편집 | 여전히 직접 파라미터 조작; SEAL의 유연성 부족 |
| Akyürek et al. (2024) "Deductive Closure Training" | 사실의 논리적 함의를 생성하여 파인튜닝 | SEAL의 지식 통합 기반이 되는 논문; RL 최적화 없음 |
| Yang et al. (2025) "Synthetic Continued Pretraining" (Entigraph) | 엔티티 그래프 기반 합성 데이터 생성 | 휴리스틱 기반; SEAL은 RL로 최적화. n=200 설정에서 SEAL(58.2%) vs Entigraph pairs+triples(56.0%) |
| Park et al. (2025) "New News: System-2 Fine-tuning" | QA 쌍 생성으로 지식 통합 | SEAL 프레임워크와 직접 통합 가능한 접근법 |

### 4-3. RL + LLM 자기 개선 계열

| 논문 | 방법 | SEAL과의 차이점 |
|---|---|---|
| Zelikman et al. (2022) "STaR" | 추론 추적을 통한 부트스트래핑 | 최종 답변 최적화; SEAL은 가중치 업데이트용 데이터 생성 최적화 |
| Singh et al. (2024) "ReST $^{EM}$ " | 거부 샘플링 + SFT의 EM 방식 | SEAL의 외부 루프에 직접 채택된 알고리즘 |
| DeepSeek-AI (2025) "DeepSeek-R1" | RL로 추론 능력 향상 | Chain-of-thought 최적화; SEAL은 가중치 업데이트 전략 최적화 |
| Zuo et al. (2025) "TTRL" | 테스트 시간 강화학습 | 레이블 없이 다수결 보상; SEAL은 실제 성능을 보상으로 사용 |

### 4-4. 메타학습 / 하이퍼네트워크 계열

| 논문 | 방법 | SEAL과의 차이점 |
|---|---|---|
| Hu et al. (2023) "Meta-learning Online Adaptation of LMs" | 소형 모델이 토큰별 가중치를 출력 | 별도 모델 필요; SEAL은 단일 모델로 생성과 적응 통합 |
| Chen et al. (2025) "Generative Adapter" | 하이퍼네트워크로 LoRA 가중치 생성 | Single-passage에서 66.8%로 SEAL(47.0%) 초과, 但 CPT에서 28.0%로 SEAL(58.2%)에 크게 뒤처짐 |
| Sun et al. (2025) "Transformer-Squared" | SVD 기반 자기 적응 LLM | LoRA 대안 제시; SEAL의 접근법과 상호보완적 |

### 4-5. 종합 포지셔닝 맵

```
                    적응 전략 학습 (메타학습)
                            ↑
                           SEAL
                       (이 논문)
                            |
고정 전략 ←────────────────┼──────────────────→ 학습된 전략
(TTT, 직접 FT)             |              (Generative Adapter, Transformer²)
                            |
                    가중치 직접 수정
                            ↓
                    (ROME, MEMIT)
```

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

#### (1) "데이터 벽(Data Wall)" 문제의 새로운 해법
Villalobos et al. (2024)의 예측에 따르면, 프론티어 LLM은 2028년까지 모든 공개 인간 생성 텍스트를 소진할 것으로 전망된다. SEAL은 모델이 **자신만의 고품질 훈련 신호를 무한히 생성**하는 패러다임을 열어, 외부 데이터 의존도를 줄이는 방향을 제시한다.

#### (2) Agentic AI 시스템의 핵심 구성 요소
에이전트가 환경과 장기간 상호작용하며 점진적으로 지식을 축적하는 시나리오에서, SEAL의 구조적 자기 수정 능력은 핵심 메커니즘이 될 수 있다. 특히 상호작용 완료 후 self-edit를 합성하여 가중치를 갱신하는 방식은 에이전트의 지속적 발전을 가능하게 한다.

#### (3) Chain-of-Thought와의 시너지
현재 RL 기반 추론 모델(DeepSeek-R1 등)은 CoT 추적을 최적화하지만, SEAL은 이와 **상보적**으로 작동할 수 있다:
- 추론 **중간**: 현재 궤적을 안내하기 위해 가중치 업데이트 수행
- 추론 **완료 후**: 핵심 통찰을 파라미터에 내재화하여 미래 추론 효율 향상

#### (4) 합성 데이터 생성 패러다임의 전환
기존 합성 데이터는 정적으로 생성되었지만(GPT-4.1 등), SEAL은 **다운스트림 성능을 직접 최적화하는 동적 데이터 생성자**를 학습한다는 점에서 패러다임 전환을 이끌 수 있다.

#### (5) Teacher-Student 프레임워크로의 확장
논문이 언급한 분리된 교사-학생 구조는 대형 모델이 소형 모델을 위한 맞춤형 self-edit를 생성하는 **지식 증류의 새로운 형태**로 발전할 수 있다.

---

### 5-2. 앞으로 연구 시 고려할 점

#### (1) 재앙적 망각 해결 (우선순위: 매우 높음)
현재 SEAL의 가장 큰 미해결 문제이다. 다음 방향이 유망하다:
- **Reward Shaping**: 이전 태스크 성능 회귀에 페널티 부과

  $$r_{\text{total}} = r_{\text{new}} - \lambda \cdot \max(0, \text{perf}\_{\text{prev}} - \text{perf}_{\text{current}})$$

- **Elastic Weight Consolidation (EWC)**: Fisher 정보 행렬 기반 중요 파라미터 보호
- **Null-space Constrained Editing (AlphaEdit)**: 기존 지식의 null space에서만 업데이트
- **RL Inner Loop**: SFT 대신 RL을 내부 루프에 사용 (Shenfeld et al., 2025의 발견 활용)

#### (2) 계산 효율성 개선 (우선순위: 높음)
현재 self-edit당 30~45초의 평가 비용은 대규모 적용의 병목이다:
- **프록시 보상(Proxy Reward)**: GPT-4.1 채점 기반 대리 보상 (논문에서 45.6% 달성, SEAL의 47.0%에 근접, 5분 소요). 더 정교한 보상 설계 필요
- **비동기 RL**: 내부 루프 평가의 병렬화
- **적응형 샘플링**: 불확실성이 높은 컨텍스트에만 집중적으로 self-edit 생성

#### (3) 레이블 없는 코퍼스로의 확장 (우선순위: 높음)
현재 구조는 모든 컨텍스트에 명시적 QA 쌍이 필요하다. 해결 방향:
- 모델이 **자신의 평가 질문을 생성**하는 "자기 출제" 메커니즘 도입
- 합성 테스트 케이스를 자동 생성하여 원래 컨텍스트가 있는 동안 보상 신호 제공
- 모델 confidence나 일관성(consistency)을 보완적 보상으로 활용

#### (4) 모델 규모 스케일링 연구 (우선순위: 중간)
현재 3B와 7B 모델만 실험되었으며, 더 큰 모델에서의 복리 효과 가능성이 관찰되었다:

$$\frac{\text{SEAL 이득}}{\text{Base Self-Edit 이득}} = \begin{cases} 1.75\times & (3B) \\ 2.04\times & (7B) \end{cases}$$

13B, 70B, 405B 등 더 큰 모델에서의 검증이 필요하다.

#### (5) Self-Edit 형식의 자동 발견 (우선순위: 중간)
현재 implications, rewrite, self-qa 등 프롬프트 형식이 수동 설계되어 있다. `no-prompt` 조건에서 18.9%에 그쳐 완전 자유 형식은 아직 어렵다. 계층적 self-edit 탐색(먼저 형식을 결정한 후 내용을 생성)이나 evolutionary search 방법이 유망하다.

#### (6) 다양한 도메인과 모달리티 검증 (우선순위: 중간)
현재 실험은 텍스트 QA와 ARC 시각 추론에만 집중되어 있다:
- 수학적 추론, 코드 생성, 과학적 추론 등 도메인별 효과 검증
- 멀티모달 self-edit (이미지, 오디오 포함) 가능성 탐색
- 도메인 특화 도구(code interpreter, symbolic solver 등)와의 통합

#### (7) Self-Edit의 해석 가능성 (우선순위: 낮음, 하지만 중요)
RL 훈련 후 모델이 어떤 self-edit를 생성하는지, 그 내용이 왜 효과적인지에 대한 체계적 분석이 부족하다. Figure 5에서 질적 분석을 제공하지만, 더 체계적인 분석이 향후 연구에 도움이 될 것이다.

#### (8) 보안 및 안정성 고려
모델이 자신의 가중치를 수정하는 능력은 잠재적인 **탈정렬(misalignment)** 위험을 내포한다:
- Self-edit가 의도치 않게 안전 필터를 우회하거나 편향을 증폭시킬 가능성
- 적응 과정에서의 안전성 제약 조건 통합 방법 연구 필요

---

## 참고 자료

**주요 논문 (본문 직접 인용)**

1. Zweiger, A., Pari, J., Guo, H., Akyürek, E., Kim, Y., & Agrawal, P. (2025). **Self-Adapting Language Models**. NeurIPS 2025. arXiv:2506.10943v2. https://arxiv.org/abs/2506.10943

2. Akyürek, E., et al. (2025). **The Surprising Effectiveness of Test-Time Training for Few-Shot Learning**. arXiv:2411.07279.

3. Singh, A., et al. (2024). **Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models (ReST $^{EM}$ )**. TMLR.

4. Meng, K., et al. (2022). **Locating and Editing Factual Associations in GPT (ROME)**. NeurIPS 2022.

5. Akyürek, A.F., et al. (2024). **Deductive Closure Training of Language Models**. ACL Findings 2024.

6. Yang, Z., et al. (2025). **Synthetic Continued Pretraining (Entigraph)**. ICLR 2025.

7. Chen, T., et al. (2025). **Generative Adapter**. ICLR 2025.

8. Sun, Q., Cetin, E., & Tang, Y. (2025). **Transformer-Squared: Self-Adaptive LLMs**. arXiv:2501.06252.

9. Hu, E.J., et al. (2022). **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022.

10. Villalobos, P., et al. (2024). **Will We Run Out of Data? Limits of LLM Scaling Based on Human-Generated Data**. arXiv:2211.04325.

11. DeepSeek-AI. (2025). **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**. arXiv:2501.12948.

12. Shenfeld, I., Pari, J., & Agrawal, P. (2025). **RL's Razor: Why Online Reinforcement Learning Forgets Less**. arXiv:2509.04259.

13. Fang, J., et al. (2025). **AlphaEdit: Null-Space Constrained Model Editing**. ICLR 2025.

14. Park, C.F., Zhang, Z., & Tanaka, H. (2025). **New News: System-2 Fine-tuning for Robust Integration of New Knowledge**. arXiv:2505.01812.

**프로젝트 웹사이트**: https://jyopari.github.io/posts/seal
