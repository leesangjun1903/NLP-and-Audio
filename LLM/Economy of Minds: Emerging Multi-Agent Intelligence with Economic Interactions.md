# Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

**"중앙 집중식 조율 없이도, 경제적 인센티브 구조만으로 약한 에이전트들이 스스로 강한 집단 지능을 형성할 수 있다."**

이 논문은 Friedrich Hayek의 분산 시장 이론에서 영감을 받아, LLM 기반 에이전트들이 **경매(auction), 지불(payment), 부(wealth) 축적**이라는 경제적 메커니즘을 통해 중앙 오케스트레이터 없이도 자기 조직화(self-organization)하고 자기 적응(self-adaptation)할 수 있음을 보여준다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **새로운 패러다임 제안** | 멀티에이전트 협력을 "설계"하는 대신 "인센티브 구조를 설계"하는 접근법 |
| **경제적 크레딧 할당** | Bucket-brigade 전이 규칙을 통한 분산적 크레딧 할당 메커니즘 |
| **이론적 보장** | 시장 선택의 수렴성, 후회 경계, Shapley-like 크레딧 할당에 대한 형식적 증명 |
| **광범위한 실험 검증** | 수학 추론, 금융 연구, 과학 연구, 가속기 설계, 분산 시스템 최적화 등 5개 도메인 |
| **약한 에이전트의 역설** | 개별적으로 약한 부분 에이전트(partial agent)들이 완전한 에이전트(complete agent)를 능가함을 실증 |

---

## 2. 상세 기술 분석

### 2.1 해결하고자 하는 문제

**기존 중앙 집중식 멀티에이전트 시스템의 두 가지 근본적 한계:**

1. **병목 문제**: 모든 정보와 의사결정이 오케스트레이터를 통해야 하므로 성능 병목 및 단일 실패점(SPOF) 발생
2. **확장성 문제**: 에이전트 수 증가에 따라 조율 복잡도가 선형적으로 증가

**개별 LLM 에이전트의 내재적 한계:**
- 유한한 컨텍스트 길이
- 부분적 지식과 관측
- 제한된 추론 예산
- 도메인별 편향(inductive bias)

### 2.2 문제 설정 (Problem Setup)

과제 환경을 부분 관측 마르코프 결정 과정(POMDP)으로 모델링한다:

$$\mathcal{E} = (\mathcal{S}, \mathcal{A}, P, r, \gamma, \mu_0)$$

여기서:
- $\mathcal{S}$: 상태 공간
- $\mathcal{A}$: 행동 공간  
- $P(s' \mid s, a)$: 전이 커널
- $r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$: 보상 함수
- $\gamma \in (0, 1]$: 할인 인수
- $\mu_0$: 초기 상태 분포

각 에이전트 $a$는 튜플로 정의된다:

$$a = (\phi_a, \pi_a, b_a, W_a)$$

- $\phi_a: \mathcal{O} \to \{0, 1\}$: 트리거 조건 (wake-up predicate)
- $\pi_a: \mathcal{O} \to \Delta(\mathcal{A})$: 행동 정책
- $b_a \in \mathbb{R}_{\geq 0}$: 고정 입찰가
- $W_a \in \mathbb{R}$: 현재 부(wealth)

이들은 동일한 동결된 LLM 백본에 에이전트별 프롬프트 $p_a = (p_a^{\text{trig}}, p_a^{\text{act}})$로 인스턴스화된다:

$$\phi_a(o) = \text{LLM}_\theta(p_a^{\text{trig}}; o) \in \{0, 1\}$$

$$\pi_a(\cdot \mid o) = \text{LLM}_\theta(p_a^{\text{act}}; o) \in \Delta(\mathcal{A})$$

### 2.3 핵심 메커니즘: 경매와 거래

**2.3.1 경매 기반 행동 선택**

시간 $t$에서의 적격 에이전트 집합:

$$\mathcal{E}_t = \{a \in \mathcal{P}_e : \phi_a(o_t) = 1\}$$

경매 승자:

$$a_t^\star \in \arg\max_{a \in \mathcal{E}_t} b_a \quad \text{(동점 시 무작위 선택)}$$

**2.3.2 Bucket-Brigade 전이 규칙 (크레딧 할당)**

경매 승자 $a_t^\star$와 이전 승자 $a_{t-1}^\star$ 간의 부 업데이트:

$$\boxed{W_{a_t^\star} \leftarrow W_{a_t^\star} - b_{a_t^\star} + r_t, \quad W_{a_{t-1}^\star} \leftarrow W_{a_{t-1}^\star} + b_{a_t^\star}}$$

이 규칙의 핵심: 에이전트는 직접 보상을 받을 뿐만 아니라, **후속 에이전트가 높은 입찰가를 지불하고 싶어하는 상태**를 만들어냄으로써도 이익을 얻는다. 즉, 가치가 성공적인 궤적을 따라 역방향으로 전파된다.

**2.3.3 신규 에이전트 입찰가 (Novice Rule)**

신규 주입된 에이전트 $a'$의 입찰가는 다음과 같이 설정된다:

$$b_{a'} = \left(\max_{a \in C_t} b_a\right) + \varepsilon_{a'}, \quad \varepsilon_{a'} \sim \mathcal{D}_\varepsilon$$

여기서 $\max \emptyset := 0$이고 $\mathcal{D}_\varepsilon$는 소규모 양의 섭동 분포. 이를 통해 신규 에이전트가 **최소 한 번은 테스트**됨을 보장한다.

### 2.4 적응 메커니즘 (Exploration & Exploitation)

에피소드 간 집단 업데이트는 세 단계로 이루어진다:

1. **임대료(Rent)**: $W_a \leftarrow W_a - \rho$ (각 에이전트에서 주기적으로 차감)
2. **제거(Removal)**: $W_a < 0$인 에이전트 삭제
3. **주입(Injection)**: 착취(exploitation)와 탐색(exploration)을 통해 새 에이전트 추가

**착취 (Exploitation)**: 부유한 에이전트의 프롬프트를 돌연변이(mutation)하여 성공적인 패턴을 보존하면서 소규모 행동 변이를 도입

**탐색 (Exploration)**: 파산 에이전트의 프롬프트를 수정(amendment)하여 실패 원인을 교정하거나 새로운 행동 공간 탐색

### 2.5 이론적 보장

**정리 1 (시장 선택의 가치 수렴):**

맥락 $x$가 순환적이고, 보상이 정상적이고 유계이며, 신규 에이전트가 novice 입찰 규칙을 따를 때, 거의 확실하게(almost surely):

$$V^\star(x) - \varepsilon_{\max} \leq \beta_\infty(x) \leq V^\star(x)$$

여기서 $V^\star(x) = \sup_{a: \phi_a(x)=1} V(a, x)$는 최적 전문가의 가치이고, $\beta_\infty(x)$는 장기 생존 에이전트의 최대 입찰가.

**정리 2 (결과 보상만으로도 충분):**

경매 승자 $a_t^\star$가 각 이력 $h_t$에서 $\varepsilon$-최적이라면:

$$Q_t^\star(h_t, a_t^\star) \geq \max_{u \in \mathcal{E}_t} Q_t^\star(h_t, u) - \varepsilon$$

유도된 경매 정책 $\pi^{\text{auc}}$는:

$$J^{\text{out}}(\pi^{\text{auc}}) \geq \sup_\pi J^{\text{out}}(\pi) - \varepsilon \cdot \frac{1 - \gamma^H}{1 - \gamma}$$

**정리 3 (오라클 대비 후회 경계):**

입찰 오차 $\beta_e$가 수렴할 때, 누적 후회:

$$\frac{\text{Reg}(E)}{E} = O(E^{-1/2})$$

**정리 4 (DAG 워크플로우에서의 Shapley-like 크레딧):**

비순환 워크플로우에서 bucket-brigade 지불이 각 에이전트의 **순서화된 한계 기여도(ordered marginal contribution)**와 일치:

$$v(S \cup \{u\}) - v(S)$$

---

### 2.6 모델 구조

```
┌─────────────────────────────────────────────────────────┐
│                    EOM 시스템 구조                        │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Planning (에피소드 내)               │    │
│  │                                                  │    │
│  │  관측 o_t → [에이전트 트리거 평가] → 적격 집합 E_t │    │
│  │                     ↓                           │    │
│  │          [경매: arg max b_a] → 승자 a*_t         │    │
│  │                     ↓                           │    │
│  │          [행동 실행] → o_{t+1}, r_t              │    │
│  │                     ↓                           │    │
│  │          [Bucket-Brigade 부 전이]                │    │
│  └─────────────────────────────────────────────────┘    │
│                         ↕                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Adaptation (에피소드 간)             │    │
│  │                                                  │    │
│  │  임대료 차감 → 파산 에이전트 제거                  │    │
│  │  착취: 부유 에이전트 프롬프트 돌연변이 → 새 에이전트│    │
│  │  탐색: 파산 에이전트 프롬프트 수정 → 새 에이전트   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**에이전트 구성 (동결된 LLM 백본 + 역할별 프롬프트):**

| 도메인 | 에이전트 역할 |
|---|---|
| 수학 추론 | Planner, Executor, Verifier |
| 금융 연구 | EDGAR 검색, 웹 검색, HTML 파싱, 정보 검색 |
| 과학 연구 | Literature, Planner, Executor, Verifier |
| 가속기 설계 | Historian, Planner, Executor |
| 분산 시스템 | Reader, Planner, Implementer, Builder, Evaluator, Finalizer |

### 2.7 성능 향상

| 도메인 | 초기 성능 | EOM 성능 | 완전 에이전트 기준선 |
|---|---|---|---|
| 수학 추론 (Llama-3.1-8B) | 15.9% | **57.0%** | 51.9% |
| 수학 추론 (Gemma-2-9B) | 4.2% | **45.1%** | 44.3% |
| 금융 연구 | 45.0% | **60.0%** | 45.0% (ReAct) |
| 과학 연구 (best-run) | — | **20.0%** | 5.0% (GEA) |
| 가속기 설계 (Avg. EDP↓) | — | **39.3** | 80.2 (DOSA) |
| 분산 시스템 (최소 비용↓) | — | **657** | 930 (OpenEvolve) |

### 2.8 한계

1. **프롬프트 공간 제한**: 적응이 동결된 백본의 프롬프트 공간에서만 이루어지므로, 새로운 기술이나 표현이 필요한 태스크에서 능력 성장이 제한됨
2. **하이퍼파라미터 민감성**: 임대료(ρ), 보상 스케일, 탐색/착취 확률 등 경제 파라미터의 균형이 중요
3. **탐색 초기 성능 저하**: 파이낸스 벤치마크에서 초기 탐색 단계에 성능이 일시적으로 하락
4. **이론-실제 간극**: 이론적 보장이 정상적(stationary) 환경을 가정하지만, 실제 태스크는 비정상적
5. **단일 GPU 실험**: H200 1개로 진행, 대규모 배포 시의 확장성 검증 미흡

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 실험적 증거

**3.1.1 쉬운 문제에서 어려운 문제로의 전이 (Easy-to-Hard Generalization)**

MATH 벤치마크에서 Level 1→5 순서로 훈련 시:
- Level 1~3에서 Llama-3.1-8B: 약 55~70% 달성
- **Level 5 (초기에 10%)에서 약 20%로 향상** — 직접 훈련하지 않은 어려운 문제로의 전이

이는 간단한 문제에서 학습된 로컬 추론 루틴이 더 어려운 문제에서 **재조합 가능함(recomposable)**을 시사한다.

**3.1.2 커리큘럼 민감도**

| 커리큘럼 방식 | 최종 성능 |
|---|---|
| Easy→Hard (기본) | ~57% |
| Hard→Easy (역순) | ~47% |

역순 커리큘럼에서도 개선이 일어나므로 완전히 의존적이지 않지만, 로컬 루틴 먼저 습득하는 것이 유리함.

**3.1.3 교차 도메인 전이 (Cross-Domain Transfer)**

과학 연구에서 EXECUTER 에이전트의 추론 루틴이 물리학, 화학, 약리학, 분광학, 생물학, 합성화학에 걸쳐 전이됨 (Appendix E.2):

> "the inherited routine applies directly: identify the governing principle, count equations versus unknowns, check symmetry, expand the scalar relation, and substitute the result back into the observation."

**핵심**: 전이되는 것은 사실적 내용이 아닌 **내용 독립적 추론 전략(content-independent reasoning strategies)**이다.

**3.1.4 재사용 가능한 도메인 구조 발견**

가속기 설계에서 EOM은 DOSA 대비 2.2× 기하평균 EDP 개선, 가장 어려운 커널에서 37.5× 개선. 중요하게도, **output-stationary 데이터플로우 패턴**을 명시적 가이드 없이 스스로 발견 — 이는 재사용 가능한 하드웨어-소프트웨어 공동설계 휴리스틱.

### 3.2 일반화 메커니즘 분석

**왜 일반화가 가능한가?**

```
경제적 선택 → 재사용 가능한 루틴 축적
     ↓
착취 메커니즘 → 성공 패턴의 변이 전파
     ↓
새 태스크에서 기존 루틴 재조합
     ↓
커리큘럼 없이도 전이 학습 효과
```

**토폴로지 진화가 일반화를 강화함:**
- 초기: 다중 역할 검증 루프 (불확실성 높을 때)
- 후기: 압축된 전문가 실행 경로
- 경매 메커니즘이 태스크 상태에 따라 동적으로 워크플로우를 선택

**전문가의 일반화 우위:**
- 일반론자(generalist): 광범위하지만 희석된 절차적 지침으로 진화 → 국소적 정밀도 부족
- 전문가(specialist): 반복적 피드백이 역할별 의사결정 규칙으로 압축 → 경제적으로 경쟁력 있음

### 3.3 일반화 한계

- 동결된 백본으로 인해 완전히 새로운 기술 습득 불가
- 초기 집단 구성(partial agent 설계)에 도메인 지식이 여전히 필요
- 매우 긴 지평선 태스크에서의 일반화는 미검증

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

**4.1.1 패러다임 전환**

기존: "더 강한 개별 에이전트 또는 더 정교한 오케스트레이터 설계"  
EOM 이후: **"적절한 인센티브 구조 설계를 통해 집단 지능이 창발하도록 유도"**

이는 AI 시스템 설계 철학의 근본적 전환을 시사한다.

**4.1.2 경제학과 AI의 융합**

- Hayek의 자생적 질서(spontaneous order) 이론이 AI 에이전트 시스템에 직접 적용 가능함을 실증
- 메커니즘 설계(mechanism design)가 멀티에이전트 시스템의 핵심 도구로 부상할 가능성

**4.1.3 크레딧 할당의 새로운 관점**

Bucket-brigade 전이가 Shapley 값과 Bellman 업데이트를 모두 복원할 수 있음을 보임 → 강화학습의 크레딧 할당 문제에 새로운 접근법 제시

**4.1.4 LLM 에이전트 연구 방향**

- 단일 강력 모델 → 약한 에이전트들의 경제적 사회로의 관심 전환
- 프롬프트 최적화를 진화적 과정으로 바라보는 새로운 프레임워크

### 4.2 향후 연구 시 고려할 점

**4.2.1 기술적 고려사항**

| 연구 방향 | 구체적 과제 |
|---|---|
| **파라미터 공간으로 확장** | 동결 백본 → 파인튜닝 가능한 백본과의 혼합 적응 |
| **입찰 메커니즘 최적화** | 고정 입찰가 대신 학습 가능한 입찰 전략 |
| **다중 모달 에이전트** | 시각, 음성, 코드 등 다양한 모달리티의 에이전트 |
| **구현된 AI (Embodied AI)** | 실물 로봇이나 물리적 환경으로의 확장 |
| **안전성** | 담합(collusion) 방지 및 악의적 에이전트 탐지 메커니즘 강화 |

**4.2.2 이론적 고려사항**

- 비정상(non-stationary) 환경에서의 수렴 보장 연구 필요
- 집단의 최적 크기 ($N_{\min}$, $N_{\max}$) 결정 이론 개발
- 경제 파라미터(ρ, 보상 스케일)의 자동 조정 이론

**4.2.3 실용적 고려사항**

- **계산 비용**: 에이전트 집단 실행 비용이 상당할 수 있으므로 효율적인 에이전트 관리 필요
- **초기 에이전트 설계**: 부분 에이전트의 역할 분할이 여전히 수동 설계에 의존
- **평가 지표**: 부(wealth) 축적이 실제 태스크 성능과 정렬되도록 보상 함수 설계 중요
- **분산 구현**: 실제 대규모 배포 시 에이전트 간 상태 동기화 문제

**4.2.4 사회적·윤리적 고려사항**

- 에이전트 경제에서의 공정성(fairness) — 초기 조건에 민감한 부의 집중
- 자율 에이전트 사회의 예측 불가능한 창발 행동에 대한 안전 가이드라인
- 인간과의 협력 구조 — 인간이 경제 설계자로서의 역할

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 관련 연구 비교 표

| 연구 | 연도 | 접근법 | 조율 방식 | 적응 메커니즘 | EOM 대비 |
|---|---|---|---|---|---|
| **ReAct** (Yao et al.) | 2022 | 단일 완전 에이전트 | 없음 | 없음 | EOM에 비해 성능 낮음 |
| **MetaGPT** (Hong et al.) | 2023 | 중앙 집중식 오케스트레이터 | 중앙화 | 수동 설계 | 확장성 한계 |
| **AutoGen** (Wu et al.) | 2024 | LLM 대화 기반 협업 | 반중앙화 | 제한적 | 경제 메커니즘 없음 |
| **Multi-Agent Debate** (Du et al.) | 2024 | 에이전트 간 토론 | 분산 | 없음 | EOM에 포함된 기준선 |
| **GEA** (Weng et al.) | 2026 | 경험 공유 기반 진화 | 분산 | 경험 공유 | EOM이 과학 연구에서 우세 |
| **CORAL** (Qu et al.) | 2026 | 자율 멀티에이전트 진화 | 분산 | 오픈엔드 탐색 | 방향 유사, 경제 메커니즘 차별화 |
| **AgentNet** (Yang et al.) | 2025 | 분산 진화적 조율 | 분산 | 그래프 연결 적응 | 검색 증강 메모리 방식 차이 |
| **ReSo** (Zhou et al.) | 2025 | 보상 기반 자기조직화 | 분산 | 보상 신호 | 유사 방향, 경제 이론 연결 없음 |
| **COMAS** (Xue et al.) | 2025 | 상호작용 보상 공진화 | 분산 | 상호 진화 | 명시적 경제 메커니즘 없음 |
| **OpenEvolve** (Novikov et al.) | 2025 | 단일 진화적 코딩 에이전트 | 없음 | 단일 에이전트 진화 | EOM이 Cloudcast에서 28% 우세 |
| **EOM (본 논문)** | 2026 | 경제적 인센티브 기반 사회 | 완전 분산 | 경매+부 축적+돌연변이 | — |

### 5.2 주요 차별점

**EOM vs 기존 접근법의 핵심 차별점:**

```
기존 멀티에이전트:
  협력 = 명시적 통신 프로토콜 + 역할 할당 (수동 설계)

EOM:
  협력 = 경제적 신호 (가격 = 정보) → 자발적 창발
```

**이론적 기반의 차별화:**
- 대부분의 기존 연구는 경험적 방법론에 의존
- EOM은 Hayek 기계, Holland의 bucket-brigade, 메커니즘 설계 이론에 기반한 **형식적 이론** 제공

**일반화 관점에서의 비교:**

| 방법 | 일반화 메커니즘 |
|---|---|
| ReAct | 없음 (태스크별 제로샷) |
| MetaGPT | 역할 재사용 (수동) |
| GEA | 경험 공유 (제한적 전이) |
| **EOM** | 경제적 선택 → 재사용 가능한 루틴 축적 → 크로스 도메인 전이 |

---

## 참고 자료

**본 논문:**
- Qi, Z., Su, H., Qu, A., et al. (2026). *Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions*. arXiv:2606.02859v1

**논문 내 인용된 핵심 참고자료:**
- Hayek, F. A. (1945). The use of knowledge in society. *The American Economic Review*, 35(4), 519–530.
- Baum, E. B. (1999). Toward a model of intelligence as an economy of agents. *Machine Learning*, 35(2), 155–185.
- Holland, J. H. (1985). Properties of the bucket brigade algorithm. *Proceedings of the 1st International Conference on Genetic Algorithms*.
- Yao, S., et al. (2022). React: Synergizing reasoning and acting in language models. *ICLR 2022*.
- Du, Y., et al. (2024). Improving factuality and reasoning in language models through multiagent debate. *ICML 2024*.
- Weng, Z., et al. (2026). Group-evolving agents: Open-ended self-improvement via experience sharing. arXiv:2602.04837.
- Hong, S., et al. (2023). MetaGPT: Meta programming for a multi-agent collaborative framework. *ICLR 2024*.
- Wu, Q., et al. (2024). AutoGen: Enabling next-gen LLM applications via multi-agent conversations. *COLM 2024*.
- Hendrycks, D., et al. (2021). Measuring mathematical problem solving with the MATH dataset. arXiv:2103.03874.
- Zhou, H., et al. (2025). ReSo: A reward-driven self-organizing LLM-based multi-agent system. *EMNLP 2025*.

> **⚠️ 정확도 주의**: 본 분석은 제공된 논문 PDF(arXiv:2606.02859v1)에 기반하며, 2026년 6월 1일 제출된 프리프린트입니다. 동료 심사(peer review)를 거치지 않은 논문이므로, 결과의 재현성 및 주장의 검증에 주의가 필요합니다. 비교 분석에 사용된 관련 연구들의 상세 수치는 각 원본 논문을 직접 확인하시기 바랍니다.
