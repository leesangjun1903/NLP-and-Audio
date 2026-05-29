# ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents
---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

ProRL Agent는 **"Rollout-as-a-Service"** 철학을 기반으로, 멀티턴 LLM 에이전트의 강화학습(RL) 훈련에서 **롤아웃(rollout) 오케스트레이션을 RL 학습 루프로부터 완전히 분리**해야 한다고 주장한다.

기존 프레임워크(SkyRL-Agent, VeRL-Tool, Agent Lightning, rLLM, GEM 등)는 롤아웃 로직을 RL 훈련 스택 내부에 **긴밀하게 결합(tightly coupled)** 시켜 두었고, 이는 다음 두 가지 핵심 문제를 야기한다:

1. **충돌하는 시스템 요구사항**: 롤아웃은 I/O 집약적이고, 훈련은 GPU 집약적이어서 결합 시 자원 효율이 저하됨
2. **이식성 및 유지보수 어려움**: 롤아웃 로직이 훈련 코드에 내장되어 새 백엔드로 마이그레이션 시 전체 파이프라인 재구현 필요

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **아키텍처 혁신** | Rollout-as-a-Service: HTTP API를 통한 롤아웃과 훈련의 완전 분리 |
| **HPC 호환 샌드박스** | Docker 없이 Singularity 기반 rootless 실행 환경 제공 |
| **Token-in/Token-out** | 재토크나이제이션 드리프트 방지를 위한 토큰 ID 기반 통신 |
| **효율적 DAPO 구현** | 비동기 보충 메커니즘으로 워커 유휴 시간 최소화 |
| **확장성 검증** | SWE-Bench Verified에서 4B/8B/14B 모델 전반에 걸친 성능 향상 입증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

멀티턴 에이전트 RL 훈련에서 단일 롤아웃은 다음과 같은 복잡한 프로세스를 포함한다:

$$\tau = \{(s_0, a_0, o_0), (s_1, a_1, o_1), \ldots, (s_T, a_T, o_T)\}$$

여기서 $s_t$는 상태(대화 이력), $a_t$는 에이전트 액션(도구 호출), $o_t$는 환경 관찰값이다.

멀티턴 에이전트는 **POMDP(Partially Observable Markov Decision Process)** 로 형식화된다:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{R}, \gamma)$$

- $\mathcal{S}$: 상태 공간 (코드 저장소, 브라우저 상태 등)
- $\mathcal{A}$: 액션 공간 (bash 명령, 파일 편집, 웹 검색 등)
- $\mathcal{O}$: 관찰 공간 (실행 결과, 오류 메시지 등)
- $\mathcal{T}$: 전이 함수
- $\mathcal{R}$: 보상 함수 (검증 가능한 스칼라)
- $\gamma$: 할인 인수

**문제의 핵심**: 기존 시스템에서 롤아웃과 훈련이 결합되면, I/O-bound 롤아웃과 GPU-bound 훈련이 동일 프로세스에서 실행되어 **자원 간섭 및 확장성 저하**가 발생한다.

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 LLM 백엔드 로드 밸런싱 (Min-Heap 기반)

프레임워크에서 사용하는 핵심 수식은 **최소 힙 기반 부하 분산** 알고리즘이다:

```math
s^* = \arg\min_{s} w_s, \qquad w_{s^*} \leftarrow w_{s^*} + 1

```

여기서:
- $s^*$: 선택된 LLM 백엔드 서버
- $w_s$: 서버 $s$에 등록된 이후 할당된 총 추론 호출 수
- 카운터는 **태스크 단위**(호출 단위 아님)로 증가하여 동일 태스크 내 모든 호출이 동일 서버로 라우팅됨 → **KV 캐시 재사용 극대화**

이 방식은 글로벌 동기화 없이 라운드로빈과 유사한 균형을 달성하며, 단일 락으로 고동시성 환경에서 안전하게 동작한다.

#### 2.2.2 DAPO 알고리즘 (Dynamic Sampling Policy Optimization)

DAPO (Yu et al., 2025)를 핵심 RL 알고리즘으로 채택한다. DAPO의 핵심은 **Zero-Variance Prompts 필터링**이다:

```math
\mathcal{D}_{\text{informative}} = \left\{ x_i \;\middle|\; 0 < \frac{1}{G}\sum_{g=1}^{G} r(y_g^{(i)}) < 1 \right\}
```

여기서:
- $x_i$: $i$번째 프롬프트
- $G$: 인스턴스당 생성 롤아웃 수 (논문에서 $G=8$)
- $r(y_g^{(i)})$: $g$번째 롤아웃의 보상 (이진: 해결/미해결)

즉, 보상이 모두 1(너무 쉬움) 또는 모두 0(너무 어려움)인 프롬프트는 **그래디언트 신호가 없으므로** 학습에서 제외된다.

정책 업데이트 목표 (PPO/GRPO 계열):

$$\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

여기서:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$: 중요도 샘플링 비율
- $\hat{A}_t$: 추정 어드밴티지
- $\beta = 1 \times 10^{-4}$: KL 패널티 계수 (논문 실험 설정)
- $\epsilon$: 클리핑 파라미터

#### 2.2.3 효율적 DAPO 비동기 보충 메커니즘

기존 배치-by-배치 방식의 비효율성:

$$T_{\text{naive}} = \left\lceil \frac{n_{\text{target}}}{n_{\text{informative per batch}}} \right\rceil \times T_{\text{batch}}$$

ProRL Agent의 비동기 방식:

$$T_{\text{efficient}} \approx T_{\text{batch}} + \Delta_{\text{overhead}}$$

3가지 메커니즘으로 구현:
1. **지속적 처리량**: 큐가 비는 즉시 보충
2. **조기 종료**: 목표 수의 정보성 프롬프트 수집 시 즉시 종료
3. **크로스-이터레이션 지속성**: 미완료 롤아웃을 다음 이터레이션으로 이월

### 2.3 모델 구조

ProRL Agent는 세 가지 핵심 컴포넌트로 구성된다:

```
┌─────────────────────────────────────────────────────────┐
│                    RL Trainer                            │
│   (veRL / NeMo RL)  ─── HTTP ───▶ POST /process        │
│                     ◀─────────── Trajectory + Reward    │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│              ProRL Agent Server                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │INIT Pool │─▶│ RUN Pool │─▶│    EVAL Pool         │  │
│  │(I/O-bd) │  │(GPU-bd)  │  │(ms ~ minutes)        │  │
│  └──────────┘  └──────────┘  └──────────────────────┘  │
│                                                          │
│  LLM Backend Pool: MinHeap(w_s)                          │
│  REST API: /process /cancel /add_llm_server              │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│              Sandbox Environment                         │
│  SingularityRuntime (rootless, –fakeroot)               │
│  AgentHandler: init() → run() → eval()                  │
│  Tool Backends: Bash(ptyprocess) / IPython / UDS        │
└─────────────────────────────────────────────────────────┘
```

#### 컴포넌트 상세

**① Sandbox Environment**
- `AgentHandler` 추상 인터페이스: `init()`, `run()`, `eval()`
- `SingularityRuntime`: rootless 컨테이너, `127.x.x.x` 루프백 IP 할당
- 최적화된 도구 백엔드:
  - **Bash**: tmux → **ptyprocess** (직접 의사터미널, 지연시간 대폭 감소)
  - **IPython**: Jupyter 게이트웨이 → **in-process API** (네트워크 오버헤드 제거)
  - **UDS**: TCP 루프백 → **Unix Domain Socket** (OS 커널 직접 메시지 전달)

**② ProRL Agent Server**
- 3단계 비동기 파이프라인 (INIT → RUN → EVAL), 각 독립 워커 풀
- Min-heap LLM 백엔드 풀 (동적 등록/교체)
- HTTP REST API 인터페이스

**③ RL Trainer 연결**
- veRL, NeMo RL 모두 지원
- 2단계 계층적 로드 밸런싱 (동일 노드 우선 → 라운드로빈)

### 2.4 성능 향상

#### SWE-Bench Verified 결과

| 모델 크기 | 베이스라인 | SkyRL-v0 (reported) | ProRL Agent (RL) | 향상률 |
|-----------|-----------|---------------------|------------------|--------|
| 4B | Qwen3-4B: 14.8 | — | **21.2** | +43.2% |
| 8B | Qwen3-8B: 9.6 | SkyRL-8B: 9.4 | **18.0** | +87.5% (vs SkyRL) |
| 14B | Qwen3-14B: 15.4 | SkyRL-14B: 21.6 | **23.6** | +9.3% (vs SkyRL) |

#### 다른 에이전트 도메인 성능

| 도메인 | 메트릭 | 초기값 | 최종값 | 향상 |
|--------|--------|--------|--------|------|
| STEM Agent | Mean Reward | ~0.20 | ~0.65 | +225% |
| Math Agent | Pass@1 (AMC) | 0.40 | ~0.90 | +125% |
| Code Agent | Pass@1 (Codeforces) | 0.23 | ~0.42 | +83% |

#### 컴포넌트 Ablation (Qwen3-14B, 8×H100)

| Load Balancing | Efficient Bash | Stale Job Cleanup | Action Time (s) | GPU Util (%) | Throughput (inst/s) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ✓ | ✓ | ✓ | 0.42 | **78** | **0.37** |
| ✓ | ✓ | ✗ | 0.42 | 42 | 0.25 |
| ✓ | ✗ | ✓ | 0.78 | 68 | 0.29 |
| ✗ | ✓ | ✓ | 0.42 | 65 | 0.30 |

#### 확장성
- 노드 수 증가에 따라 처리량이 **거의 선형적으로 증가** (4B/8B/14B 모델 모두)

### 2.5 한계

논문이 명시적으로 언급하거나 구조적으로 드러나는 한계:

1. **평가 도메인 제한**: SWE-Bench Verified, AMC, Codeforces, STEM QA에 국한됨. GUI, 웹 브라우저, OS 에이전트 등 더 복잡한 환경은 미검증
2. **클러스터 규모 견고성**: 논문 결론에서 "improved cluster-scale robustness"를 future work로 명시
3. **보상 설계 의존성**: 새 도메인 적용 시 적절한 도구 구성 및 보상 설계가 필요하며, 이 과정이 자동화되지 않음
4. **비교 공정성**: SkyRL과 정확히 동일한 훈련 데이터/설정이 아닐 수 있음 (논문에서 "reproduced" vs "reported" 구분)
5. **오프-폴리시 문제**: 체크포인트 교체 시 기존 파이프라인 내 롤아웃이 구 모델로 생성될 수 있음 (논문에서 "no interruption to jobs already in the pipeline" 언급 — 이는 의도된 설계이나 오프-폴리시 갭을 초래)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 인프라 수준에서의 일반화 기제

ProRL Agent의 일반화 성능 향상 가능성은 **인프라 설계**와 **RL 알고리즘** 두 층위에서 분석해야 한다.

#### ① 플러그인 아키텍처의 도메인 일반화

`AgentHandler` 추상 인터페이스는 도메인 간 일반화를 가능하게 한다:

$$\text{AgentHandler}_{\text{domain}} = \{\text{init}_d(), \text{run}_d(), \text{eval}_d()\}$$

새 도메인 $d'$ 추가 시 필요한 것은 오직 `AgentHandler` 서브클래스 구현뿐이며, **훈련 코드 변경이 불필요**하다. 이는 빠른 도메인 확장을 가능하게 하여, 다양한 태스크에서의 공동 훈련(multi-task RL training) 가능성을 열어 준다.

#### ② Token-in/Token-out과 훈련 안정성

재토크나이제이션 드리프트가 있을 때 발생하는 문제:

$$\hat{\pi}_\theta(a_t \mid s_t) \neq \pi_\theta(a_t \mid s_t)$$

즉, 롤아웃 시 생성된 토큰과 훈련 시 계산된 로그 확률이 다른 토큰 시퀀스에 기반하게 되어 **오프-폴리시 편향**이 발생한다. ProRL Agent는 토큰 ID를 정준 표현으로 사용함으로써:

$$\log \hat{\pi}_\theta(a_t \mid s_t) = \log \pi_\theta(a_t \mid s_t)$$

이를 보장한다. 이 안정성이 더 긴 훈련에서의 일반화를 지원한다.

#### ③ DAPO를 통한 효율적 탐색과 일반화

Zero-Variance Prompt 필터링으로 **학습 신호가 있는 데이터만** 사용:

$$\hat{A}_t = r(\tau) - b, \quad \text{Var}[r] > 0 \text{ 보장}$$

이는 모델이 경계 난이도 문제(boundary difficulty problems)에서만 학습하게 하여, 과도하게 쉽거나 어려운 태스크로 인한 **과적합을 방지**하고 일반화를 촉진한다.

### 3.2 실험적 일반화 증거

논문은 4가지 독립적 도메인에서 성능 향상을 보여 인프라 수준의 일반화를 실증한다:

$$\text{Generalization Score} = \frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \frac{\text{Performance}_d^{\text{RL}} - \text{Performance}_d^{\text{base}}}{\text{Performance}_d^{\text{base}}}$$

각 도메인 $d \in \{\text{SWE, STEM, Math, Code}\}$에서 모두 양의 향상을 보였다는 점은, 이 인프라가 **도메인-불가지론적(domain-agnostic)** 일반화 능력을 보유함을 시사한다.

### 3.3 일반화를 제한하는 요인

1. **보상 해킹(Reward Hacking)**: 검증 가능한 보상만 사용하므로, 보상이 불완전하게 설계된 도메인에서는 일반화 실패 가능
2. **분포 이동**: SWE-Gym의 293개 인스턴스로만 훈련 → 새로운 저장소 구조에 대한 일반화 미보장
3. **도구 의존성**: 훈련 시 사용한 도구 셋(Bash, IPython 등)에 과적합될 위험

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 단일턴 → 멀티턴 RL 진화

| 연구 | 연도 | 방법 | 특징 |
|------|------|------|------|
| DeepSeekMath (Shao et al.) | 2024 | GRPO | 수학 단일턴 RL |
| DeepSeek-R1 (Guo et al.) | 2025 | RLVR | 추론 체인 강화, 단일턴 |
| Open-ReasonerZero (Hu et al.) | 2025 | RL on base model | 기반 모델 스케일링 |
| SkyRL-v0 (Cao et al.) | 2025 | PPO + Docker | 멀티턴, 결합 설계 |
| DeepSWE (Luo et al.) | 2025 | RL scaling | 코딩 에이전트 |
| **ProRL Agent (본 논문)** | **2026** | **Rollout-as-a-Service** | **분리 설계, HPC 호환** |

### 4.2 에이전트 RL 인프라 비교

| 프레임워크 | 훈련-롤아웃 분리 | Rootless 샌드박스 | 스캐폴드 독립성 | 비고 |
|-----------|:---:|:---:|:---:|------|
| SkyRL-Agent (2025) | ✗ | ✗ | ✓ | Ray 기반, Docker 필요 |
| VeRL-Tool (2025) | ✗ | ✗ | ✓ | veRL 확장, 도구 오프로드 |
| Agent Lightning (2025) | ✗ | ✗ | ✗ | 단일 프로세스 트리 |
| rLLM (2025) | ✗ | ✗ | ✓ | veRL 포크, 모놀리식 |
| GEM (2025) | ✗ | ✗ | ✓ | 인메모리 환경 |
| AgentGym-RL (2026) | ✗ | — | — | 멀티턴 의사결정 특화 |
| **ProRL Agent (2026)** | **✓** | **✓** | **✓** | **HTTP 서비스 분리** |

### 4.3 벤치마크 환경 진화

| 환경 | 연도 | 특징 |
|------|------|------|
| WebArena (Zhou et al.) | 2023 | 웹 브라우저 기반 자율 에이전트 |
| SWE-Bench (Jimenez et al.) | 2023/2024 | GitHub 이슈 해결 |
| OSWorld (Xie et al.) | 2024 | 실제 OS 환경 멀티모달 |
| R2E-Gym (Jain et al.) | 2025 | SWE 에이전트 절차적 환경 |
| VAGEN (Wang et al.) | 2025 | VLM 에이전트 멀티턴 RL |

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### ① 인프라 패러다임 전환

"Rollout-as-a-Service" 개념은 **LLM 추론의 "Inference-as-a-Service"(vLLM, SGLang)에서 영감을 받아** 롤아웃 영역으로 확장한 것이다. 이는 앞으로의 에이전트 RL 프레임워크 설계의 **표준 아키텍처 패턴**이 될 가능성이 높다.

$$\underbrace{\text{Inference-as-a-Service}}_{\text{vLLM, SGLang}} \xrightarrow{\text{확장}} \underbrace{\text{Rollout-as-a-Service}}_{\text{ProRL Agent}}$$

#### ② 대규모 멀티태스크 에이전트 RL

분리 설계로 인해 **서로 다른 도메인의 롤아웃을 하나의 서버에서 동시 처리**하는 멀티태스크 RL이 용이해진다. 이는 도메인 일반화 에이전트 연구를 촉진할 것이다.

#### ③ HPC 환경에서의 대규모 실험 민주화

Singularity 기반 rootless 실행은 **학술 연구기관의 HPC 클러스터에서도** 대규모 에이전트 RL 실험을 가능하게 하여, 자원 접근성의 장벽을 낮춘다.

#### ④ 재현성과 비교 공정성 향상

표준화된 HTTP API 인터페이스는 **다양한 RL 알고리즘을 동일한 롤아웃 인프라 위에서 공정하게 비교**하는 것을 가능하게 한다.

### 5.2 앞으로 연구 시 고려할 점

#### ① 오프-폴리시 갭 관리

체크포인트 교체 시 파이프라인 내 기존 롤아웃이 구 모델로 생성되는 문제를 고려해야 한다:

$$\text{Off-policy ratio} = \frac{\pi_{\theta_{\text{new}}}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

중요도 샘플링 보정이나 **적응적 클리핑** 메커니즘 연구가 필요하다.

#### ② 보상 설계의 자동화

새 도메인마다 수동으로 `eval()` 메서드를 구현해야 하는데, **자동화된 보상 함수 학습** (예: RLHF, Constitutional AI, Process Reward Model)과의 결합을 고려할 필요가 있다.

#### ③ 클러스터 규모 견고성

논문 스스로 "improved cluster-scale robustness"를 future work로 명시한다. 수백~수천 노드 규모에서의 **장애 복구, 네트워크 파티션, 컨테이너 오케스트레이션** 연구가 필요하다.

#### ④ 멀티모달 에이전트 지원

현재는 텍스트 기반 도구(Bash, IPython, 웹 검색) 중심이다. GUI, 비전-언어 에이전트(VAGEN 등)를 위한 **스크린샷/비디오 기반 관찰** 처리를 위한 확장이 필요하다.

#### ⑤ 장기 훈련 안정성

Figure 4의 훈련 곡선에서 STEM 에이전트가 "포화 징후 없이 계속 향상"됨을 보여 주나, **보상 해킹, 분포 붕괴, 훈련 불안정성**에 대한 장기 분석이 필요하다:

$$\text{Reward Collapse Risk} \propto \frac{T_{\text{training}}}{\text{KL penalty strength}}$$

#### ⑥ 샌드박스 보안 강화

`–fakeroot` 옵션은 **시뮬레이션된 루트 권한**을 제공하는데, 악의적 코드 실행 시나리오에서의 보안 취약성 분석이 필요하다.

#### ⑦ 이론적 수렴 보장

실용적 성능 향상은 입증되었으나, 롤아웃 분리 설계가 **RL 알고리즘의 수렴 속성**에 미치는 영향에 대한 이론적 분석이 부재하다. 비동기 업데이트로 인한 수렴 조건 연구가 필요하다.

---

## 참고 자료 (출처)

**논문 원본:**
- Zhang, H., Liu, M., Zhang, S., et al. "ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents." *arXiv:2603.18815v1 [cs.AI]*, 19 Mar 2026. (제공된 PDF 원문)

**논문 내 인용 주요 참고문헌:**
- Yu, Q., et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." *arXiv:2503.14476*, 2025.
- Sheng, G., et al. "HybridFlow: A Flexible and Efficient RLHF Framework." *EuroSys 2025*, pp. 1279–1297.
- Guo, D., et al. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." *arXiv:2501.12948*, 2025.
- Cao, S., et al. "SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning." 2025a.
- Cao, S., et al. "SkyRL-Agent: Efficient RL Training for Multi-Turn LLM Agent." *arXiv:2511.16108*, 2025b.
- Jimenez, C.E., et al. "SWE-Bench: Can Language Models Resolve Real-World GitHub Issues?" *ICLR 2024*.
- Luo, X., et al. "Agent Lightning: Train Any AI Agents with Reinforcement Learning." *arXiv:2508.03680*, 2025c.
- Liu, M., et al. "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models." *NeurIPS 2025*.
- The Agent Lightning (AGL) Team. "No More Retokenization Drift." *vLLM Blog*, 2025. https://blog.vllm.ai/2025/10/22/agent-lightning.html
- Kwon, W. "vLLM: An Efficient Inference Engine for Large Language Models." *PhD Thesis, UC Berkeley*, 2025.
- Zheng, L., et al. "SGLang: Efficient Execution of Structured Language Model Programs." *NeurIPS 2024*, pp. 62557–62583.
- NVIDIA. "NeMo Gym." GitHub: https://github.com/NVIDIA-NeMo/Gym, 2025.
- Wang, K., et al. "VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents." *NeurIPS 2025*.
- Kaelbling, L.P., Littman, M.L., Cassandra, A.R. "Planning and Acting in Partially Observable Stochastic Domains." *Artificial Intelligence*, 101(1-2):99–134, 1998.
- Tan, S., et al. "rLLM: A Framework for Post-Training Language Agents." 2025.
- Jiang, D., et al. "VeRL-Tool: Towards Holistic Agentic Reinforcement Learning with Tool Use." *arXiv:2509.01055*, 2025.
- Liu, Z., et al. "GEM: A Gym for Agentic LLMs." *arXiv:2510.01051*, 2025b.
- Wang, Z., et al. "RAGen-v2: Understanding Reasoning Collapse in Multi-Turn Agent Reinforcement Learning." 2026.
- Xi, Z., et al. "AgentGym-RL: An Open-Source Framework to Train LLM Agents for Long-Horizon Decision Making via Multi-Turn RL." *ICLR 2026*.

> **⚠️ 정확도 고지**: 본 분석은 제공된 논문 PDF 원문(arXiv:2603.18815v1)에 전적으로 기반하며, 논문 외 외부 소스를 참조하지 않았습니다. 논문에 명시되지 않은 내용(예: GRPO 수식의 세부 전개)은 일반적으로 알려진 RL 이론을 바탕으로 맥락에 맞게 보충했으며, 이 부분은 논문의 직접적 주장이 아님을 명시합니다.
