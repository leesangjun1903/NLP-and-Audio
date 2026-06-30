# SIA: Self Improving AI with Harness & Weight Updates 

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

SIA 논문의 핵심 주장은 다음 한 문장으로 요약됩니다:

> **"AI 자기개선(Self-Improvement)의 두 연구 흐름(Harness 업데이트와 가중치 업데이트)을 단일 폐쇄 루프로 통합하면, 어느 하나만 사용할 때보다 일관되게 더 높은 성능을 달성할 수 있다."**

기존 연구는 두 개의 **사일로(Silo)**로 분리되어 있었습니다:

| 사일로 | 방식 | 한계 |
|--------|------|------|
| **Silo 1: Harness/Scaffold 자기개선** | 스캐폴드(프롬프트·도구·재시도 로직)를 메타 에이전트가 반복 개선, 모델 가중치 고정 | 소프트웨어 엔지니어링 수준의 개선에 머물며 도메인 직관 습득 불가 |
| **Silo 2: 테스트 시 훈련(TTT/TTRL)** | RL로 가중치를 업데이트, Harness는 사람이 고정 작성 | 스캐폴드의 구조적 개선 없이 파이프라인이 정적으로 유지 |

SIA는 이 두 사일로를 **Feedback-Agent**라는 단일 언어 모델 에이전트가 동적으로 조율하는 방식으로 통합합니다.

### 1.2 주요 기여

- **기술적 기여**: Harness 업데이트와 LoRA 기반 가중치 업데이트를 동시에 운영하는 **최초의 단일 자기개선 루프** 제안
- **실증적 기여**: 서로 성격이 다른 세 도메인(법률·시스템·생물학)에서 일관된 SOTA 초과 달성
  - LawBench: $+25.1\%$ (70.1% vs 45.0%)
  - AlphaEvolve TriMul: $-12.4\%$ 지연시간 (1,017 vs 1,161 µs)
  - MAGIC scRNA-seq: $+20.4\%$ (0.289 vs 0.240 mse_norm)
- **분석적 기여**: 두 레버(lever)가 서로 다른 변화 공간을 점유함을 어블레이션으로 실증 — Harness는 *외부 인프라*를, 가중치 업데이트는 *내부 도메인 직관*을 변화시킴

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**인간이 병목(Bottleneck)**이라는 전제에서 출발합니다. 현재 AI 개선 과정은:

1. 연구자가 모델을 설계·훈련
2. 엔지니어가 에이전트 스캐폴드를 수동 설계·디버깅

이 두 단계 모두 인간의 개입이 필수적입니다. SIA는 **태스크 명세(Task Specification)와 검증기(Verifier)만 주어지면**, 인간 개입 없이 스캐폴드와 가중치 모두를 자동 개선하는 시스템을 목표로 합니다.

**핵심 갭(Gap)**: 기존 연구는 Harness 업데이트와 가중치 업데이트 중 하나만 수행하며, 이 두 레버를 **하나의 동적 루프**에서 결합한 사례가 없었습니다.

---

### 2.2 제안하는 방법 및 수식

#### 2.2.1 시스템 구성 요소

SIA는 세 개의 LLM 컴포넌트로 구성됩니다:

**① Meta-Agent ($\mathcal{M}$)**: 태스크 명세 $\mathcal{U}$와 참조 구현 $\mathcal{R}$로부터 초기 스캐폴드를 생성

$$A_1 = \mathcal{M}(\mathcal{U}, \mathcal{R})$$

**② Feedback-Agent ($\mathcal{F}$)**: 이전 세대의 스캐폴드 $A_g$, 실행 궤적 $\tau_g$, 성능 지표 $\mathcal{E}_g$를 입력받아 개선된 스캐폴드 생성

$$A_{g+1} = \mathcal{F}(A_g, \tau_g, \mathcal{E}_g, \mathcal{U})$$

**③ Task-Specific Agent**: 세대 $g$에서 실제로 평가 데이터셋 $\mathcal{D}$에 대해 실행되는 스캐폴드 $A_g$

#### 2.2.2 Harness 업데이트 (Scaffold 진화)

각 세대 $g$는 3단계 프로토콜을 따릅니다:

1. **실행(Execution)**: $A_g$가 샌드박스 내에서 $\mathcal{D}$에 대해 실행, 궤적 $\tau_g$ 캡처
2. **분석(Analysis)**: $\mathcal{F}$가 $A_g$의 소스코드, $\tau_g$, 지표 $\mathcal{E}_g$를 수신
3. **개선(Improvement)**: $\mathcal{F}$가 개선 보고서와 차세대 에이전트 $A_{g+1}$ 생성

Harness 업데이트 점화식:

$$A_{g+1} = \mathcal{F}(A_g,\ \tau_g(\pi_\theta),\ \mathcal{E}_g,\ \mathcal{U})$$

여기서 $\tau_g(\pi_\theta)$는 현재 모델 $\pi_\theta$로 스캐폴드 $A_g$를 실행하여 수집된 궤적이며, **가중치 $\theta$는 이 단계에서 고정**됩니다.

#### 2.2.3 가중치 업데이트 — RL 알고리즘 선택

Feedback-Agent는 궤적 관찰에 기반하여 아래의 RL 알고리즘 중 하나를 동적으로 선택합니다:

**① PPO with GAE (LawBench에 적용)**

가치 헤드 $V_\phi$가 per-token 어드밴티지 추정치를 생성:

$$\hat{A}_t = \sum_{l} (\gamma\lambda)^l \delta_{t+l}$$

클리핑된 서로게이트 목적함수:

$$\mathcal{L}^{CLIP} = \min\!\left(r_t \hat{A}_t,\ \text{clip}(r_t, 1 \pm \varepsilon)\hat{A}_t\right)$$

여기서 $r_t = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$는 확률 비율, $\varepsilon$는 클리핑 임계값입니다.

**② GRPO (Denoising에 적용)**

가치 네트워크 없이 롤아웃 그룹 내에서 어드밴티지를 정규화:

$$\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$$

여기서 $G$는 상태당 롤아웃 수, $\bar{r}$과 $\sigma_r$은 그룹 보상의 평균과 표준편차입니다.

**③ 엔트로픽 어드밴티지 가중치 (TriMul에 적용)**

보상이 희소하고 오른쪽 꼬리가 긴 경우, softmax 재분배로 그래디언트 질량을 조절:

$$w_i \propto \exp\!\left(\frac{r_i}{\beta}\right)$$

적응형 온도 $\beta$는 유효 샘플 크기가 하한 임계값 이상 유지되도록 온라인으로 조정됩니다.

**④ REINFORCE + KL-to-base**

기반 모델로의 KL 페널티를 추가한 몬테카를로 리턴:

$$R_t = \sum_{t' \geq t} \gamma^{t'-t} r_{t'}$$

목적함수에 규제항 추가:

$$\mathcal{L} = -R_t + \alpha\, \text{KL}(\pi_\theta \| \pi_{\theta_0})$$

**⑤ DPO (Direct Preference Optimization)**

승리 롤아웃 $y^+$와 패배 롤아웃 $y^-$ 쌍이 주어질 때:

$$\mathcal{L}_{DPO} = -\log \sigma\!\left(\beta \log \frac{\pi_\theta(y^+)}{\pi_{\theta_0}(y^+)} - \beta \log \frac{\pi_\theta(y^-)}{\pi_{\theta_0}(y^-)}\right)$$

**⑥ Best-of-N Behavioural Cloning**

보상이 극도로 희소한 경우($\mathbb{E}[r] \approx 0$) 콜드 스타트 단계로 사용. 상위 $k$개 롤아웃을 교차 엔트로피 손실로 증류.

---

### 2.3 모델 구조

```
Task spec U  ──► Meta-Agent (Claude Sonnet 4.6)
Verifier V   ──►      │
                       ▼
               Task-Specific Agent
               (gpt-oss-120b + LoRA rank=32)
                       │
                       ▼ (실행 → 궤적 τ_g 생성)
               Environment
                       │
                       ▼
               Feedback-Agent (Claude Sonnet 4.6)
                 ┌─────┴─────┐
                 ▼           ▼
          Harness Update  Weight Update
          (스캐폴드 진화)  (LoRA RL 훈련, H100 via Modal)
                 └─────┬─────┘
                       ▼ (다음 세대)
               Task-Specific Agent
```

**핵심 설계 특징**:
- Meta-Agent와 Feedback-Agent: `Claude Sonnet 4.6`
- Task-Specific Agent 기반 모델: `gpt-oss-120b`
- 어댑터: LoRA (rank $r = 32$, 학습률 $4 \times 10^{-5}$)
- 훈련 인프라: Modal (H100 GPU), rollout 생성·보상 할당·그래디언트 업데이트 통합 파이프라인
- 두 레버는 **순차적 단계가 아닌 동적 인터리빙** 방식으로 작동

---

### 2.4 성능 향상

아래 어블레이션 표가 각 레버의 기여를 명확히 보여줍니다:

| 태스크 | 초기값 | 이전 SOTA | SIA-H (Harness만) | SIA-W+H (Harness+가중치) |
|--------|--------|-----------|-------------------|--------------------------|
| LawBench (top-1 acc) | 13.5% | 45.0% | 50.0% | **70.1%** |
| AlphaEvolve TriMul (reward) | 0.105 | 1.292 | 0.120 | **1.475** |
| Denoising (mse_norm) | 0.048 | 0.240 | 0.241 | **0.289** |

**각 태스크별 관찰**:

- **LawBench**: Harness가 TF-IDF+LinearSVC 파이프라인 구축으로 50.0%까지 도달 후 정체 → PPO+GAE 적용으로 70.1% 달성. 191개 세부 범죄 카테고리 구분에 필요한 도메인 직관이 프롬프트로는 불가능했음을 시사

- **TriMul CUDA**: Harness가 메모리 레이아웃 힌트·컴파일 플래그 최적화로 1.14× 속도향상 후 정체 → 엔트로픽 어드밴티지 가중치 적용으로 14.02× 달성 (런타임 12,483 → 1,017 µs). H100 특화 패턴(shared-memory tiling, fp32 레지스터 누적)이 모델 파라미터에 내재화

- **scRNA-seq Denoising**: Harness가 MAGIC 하이퍼파라미터(k, t, α) 탐색으로 0.241까지 도달 후 정체 → GRPO 적용 후 **최초로 `np.clip + np.rint` 후처리 단계**가 등장하여 0.289 달성. 이는 생물학적 불변량(음이 아닌 정수 발현량)을 Harness 어느 버전에서도 발견하지 못한 구조적 통찰

---

### 2.5 한계점

논문이 명시하는 주요 한계는 **"결합 공진화 Goodhart 문제"**입니다:

$$\text{(Goodhart의 법칙)} \quad \text{측정 지표가 목표가 되면, 더 이상 좋은 측정 지표가 아니다}$$

**구체적 위험**:
- Harness 탐색과 RL 가중치 업데이트 **모두** 동일한 고정 검증기 $V$에 대해 최적화
- 두 최적화기가 서로의 업데이트 이력을 알지 못하는 **내쉬 균형(Nash Equilibrium)** 수렴 가능성
- 이 고정점은 훈련 검증기에서 강하지만, **분포 외(out-of-distribution) 스캐폴드나 새로운 정책에 취약**할 수 있음
- 표준 Goodhart 분석은 단일 최적화기를 가정하는데, 두 레버 설정은 더 복잡한 결합 변형을 생성

**기타 암묵적 한계**:
- 단 세 개의 태스크에서만 검증 (도메인 일반화 증거 부족)
- Feedback-Agent의 레버 선택이 **동결된 LLM 사전(prior)**에 의존 (학습되지 않음)
- LoRA rank 32로 고정된 어댑터 용량의 적절성 미검토
- 계산 비용에 대한 분석 부재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 지지하는 근거

#### 3.1.1 태스크 무관성(Task-Agnostic) 설계

SIA는 태스크 명세 $\mathcal{U}$와 검증기 $V$만 있으면 작동합니다. 논문이 **법률(분류)·시스템(코드 최적화)·생물학(수치 회귀)**이라는 성격이 극도로 다른 세 도메인에서 일관된 성능 향상을 보인 것은, 이 프레임워크가 특정 도메인에 하드코딩되지 않았음을 시사합니다.

#### 3.1.2 두 레버의 보완적 역할

일반화 관점에서 두 레버는 서로 다른 일반화 차원을 담당합니다:

| 레버 | 일반화 메커니즘 |
|------|----------------|
| **Harness** | 태스크 구조 탐색 공간을 확장 — 더 나은 도구·파서·재시도 로직이 에이전트가 다양한 입력 패턴에 대처하는 능력을 향상 |
| **가중치 업데이트** | 도메인 지식을 파라미터에 내재화 — 프롬프트 변화에 무관하게 유지되는 표현 학습 |

이 이중 구조는 단일 레버 방법 대비 더 강인한 표현을 형성할 가능성이 있습니다.

#### 3.1.3 Sample-Task Regularisation

Meta-Agent가 **다양한 태스크 명세 집합**에 조건화되어 초기 스캐폴드를 생성하므로, 단일 벤치마크 인스턴스에 대한 과적합을 완화합니다:

$$A_1 = \mathcal{M}(\mathcal{U}_1, \mathcal{U}_2, \ldots, \mathcal{U}_k, \mathcal{R})$$

이는 초기 스캐폴드가 과도하게 특화되지 않도록 하는 명시적 정규화 역할을 합니다.

#### 3.1.4 가중치 업데이트의 내재화 효과

논문에서 가장 설득력 있는 일반화 증거는 **scRNA-seq 태스크**에서 나타납니다. Harness의 어떤 버전도 발견하지 못한 `np.clip + np.rint` 생물학적 불변량이 가중치 업데이트 후 **자발적으로 등장**했습니다. 이는 RL 훈련이 단순 암기가 아닌 **도메인 구조에 대한 진정한 이해**를 파라미터에 인코딩했음을 시사합니다.

LoRA에 의한 내재화된 지식은 이후 **다른 스캐폴드와 결합**될 수 있으며, 이것이 일반화의 핵심 메커니즘입니다:

$$\theta^* = \theta_0 + \underbrace{\Delta W}_{\text{LoRA: 도메인 직관}}$$

$\theta_0$: 사전학습된 기반 지식, $\Delta W$: 태스크별 적응 지식

### 3.2 일반화를 제한하는 요인

#### 3.2.1 검증기 고정의 딜레마

$$\mathcal{J}(\theta, A) = \mathbb{E}_{x \sim \mathcal{D}}\left[V(x, A(x; \theta))\right]$$

시스템 전체가 단일 고정 검증기 $V$를 최대화하도록 최적화됩니다. 이는 **분포 내(in-distribution)** 성능을 극대화하지만, 검증기가 캡처하지 못하는 측면(예: 해석 가능성, 엣지 케이스 처리)에서의 일반화는 보장하지 않습니다.

#### 3.2.2 LoRA의 용량 제한

$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k)$$

rank $r = 32$로 고정된 LoRA는 태스크 복잡도에 따라 표현 용량이 부족할 수 있습니다. 여러 태스크에 걸친 순차적 적응 시 **재앙적 망각(catastrophic forgetting)** 문제가 발생할 수 있습니다.

#### 3.2.3 Coupled Goodhart 문제와 일반화

두 최적화기의 공진화는 훈련 검증기에 과적합된 내쉬 균형으로 수렴할 수 있습니다. 이 균형점에서 생성된 스캐폴드와 가중치는 **새로운 검증기나 새로운 태스크 변형**에 취약할 수 있습니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 앞으로의 연구에 미치는 영향

#### 4.1.1 패러다임 전환: 단일 레버 → 이중 레버

SIA는 자기개선 AI 연구의 패러다임을 "어떤 레버를 최적화할 것인가"에서 "언제 어떤 레버를 선택할 것인가"로 전환시킵니다. 이는 자동화된 ML 연구 파이프라인, 코드 합성, 과학적 발견 시스템 등 다양한 분야의 설계 원칙에 영향을 미칠 것입니다.

#### 4.1.2 Meta-RL 연구 방향 제시

Feedback-Agent의 레버 선택이 현재는 동결된 LLM 사전에 의존하지만, 논문은 이를 **학습 가능한 Meta-RL 정책**으로 발전시킬 것을 제안합니다:

$$\pi_{\text{selector}}^* = \arg\max_\pi \mathbb{E}_{\text{tasks}}\left[\sum_t r_t^{\text{outer}}\right]$$

이는 외부 MDP에서 $(trajectory, action, outcome)$ 트리플을 전이로 처리하여 선택 정책을 훈련하는 **재귀적 자기개선 구조**를 만듭니다. 이 방향은 메타-학습, 다중 태스크 RL, 계층적 RL 커뮤니티에 새로운 연구 질문을 제기합니다.

#### 4.1.3 자동화된 RL 알고리즘 선택

Feedback-Agent가 태스크 특성(보상 밀도, 롤아웃 비용, 희소성)에 따라 PPO/GRPO/엔트로픽 가중치 등을 동적으로 선택하는 행동은, **알고리즘 선택 문제(Algorithm Selection Problem)**를 RL 에이전트가 해결하는 새로운 방향을 제시합니다. 이는 AutoML 연구와 깊이 연결됩니다.

### 4.2 앞으로 연구 시 고려할 점

#### 4.2.1 Coupled Goodhart 문제 해결

두 레버가 공동 최적화되는 상황에서의 수렴 안정성 이론이 필요합니다. 구체적으로:

- **내쉬 균형 분석**: $(A^\*, \theta^*)$ 쌍이 게임 이론적으로 안정적인 균형인지 확인
- **다중 검증기(Multiple Verifiers)**: 단일 검증기 과적합을 방지하기 위해 앙상블 검증기 $\{V_1, V_2, \ldots, V_k\}$를 사용하는 방안
- **홀드아웃 검증기**: 훈련에 사용되지 않는 별도의 검증기로 일반화 측정

#### 4.2.2 더 세밀한 인터리빙 스케줄

현재 SIA 루프는 Harness 탐색과 가중치 업데이트를 **거친 단위로 교대**합니다. 더 세밀한 스케줄이 필요합니다:

$$\text{(제안)} \quad \Delta t_{\text{switch}} = f(\Delta\mathcal{E}_g,\ \text{gradient variance},\ \text{rollout cost})$$

Feedback-Agent가 Harness 탐색 **중간**에 가중치 업데이트를 트리거하거나, 그래디언트 단계 직후 Harness 탐색을 재개하는 세밀한 스케줄은 더 나은 개선 궤적을 가능하게 할 것입니다.

#### 4.2.3 지속적 학습과 망각 방지

순차적 태스크 적응 시 재앙적 망각을 방지하기 위한 연구가 필요합니다:

- **EWC (Elastic Weight Consolidation)**: 이전 태스크에 중요한 파라미터 보호
- **Progressive LoRA**: 태스크별 독립 어댑터 체인 구성
- **Knowledge Distillation**: 이전 태스크의 지식을 새 모델로 증류

#### 4.2.4 계산 효율성과 비용 분석

논문은 각 접근 방식의 계산 비용을 체계적으로 분석하지 않습니다. 실용적 적용을 위해:
- Harness 업데이트 vs 가중치 업데이트의 **계산 비용 트레이드오프** 분석
- 최소한의 가중치 업데이트로 최대 성능을 달성하는 **효율적 스케줄** 탐구
- **조기 종료(Early Stopping)** 기준 정의

#### 4.2.5 안전성 및 정렬(Alignment) 고려

자기개선 시스템은 잠재적인 안전 위험을 내포합니다:
- 검증기 해킹(Verifier Hacking): 검증기의 허점을 이용한 허위 성능 달성
- 스캐폴드 탈주(Scaffold Escape): 샌드박스 제약을 우회하는 스캐폴드 생성
- 가중치 드리프트: 원래 정렬 속성 상실

이에 대한 **형식 검증(Formal Verification)** 및 **컨스티튜션 AI** 방법론의 통합이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | Harness 편집 | 가중치 편집 | 핵심 차별점 vs SIA |
|------|------|:---:|:---:|---------------------|
| **SIA (본 논문)** | 2026 | ✅ | ✅ | 유일하게 두 레버를 단일 루프에서 동적으로 결합 |
| STaR (Zelikman et al.) | 2022 | ❌ | ✅ | 자기생성 근거로 지도 학습 가중치 업데이트; 스캐폴드 고정 |
| Self-Refine (Madaan et al.) | 2023 | 부분 | ❌ | 순수 추론 시 언어적 비평; 가중치 업데이트 없음 |
| Reflexion (Shinn et al.) | 2023 | 부분 | ❌ | 언어 강화학습; 가중치 미업데이트 |
| EUREKA (Ma et al.) | 2023 | 부분 | ✅ | LLM이 보상 함수 생성(스캐폴드 측), RL로 정책 훈련(가중치 측); **단방향** 루프로 보상 생성기가 훈련된 정책의 피드백으로 업데이트되지 않음 |
| Voyager (Wang et al.) | 2023 | ✅ | ❌ | Minecraft 환경에서 개방형 스킬 학습; 모델 고정 |
| FunSearch (Romera-Paredes et al.) | 2024 | 부분 | ❌ | 프로그램 합성을 통한 수학적 발견; 모델 고정 |
| AI Scientist (Lu et al.) | 2024 | 부분 | ❌ | 연구 파이프라인 자동화; 출력이 아티팩트, 스캐폴드 고정 |
| Surprising TTT (Akyürek et al.) | 2024 | ❌ | ✅ | 소수 샘플 학습을 위한 테스트 시 그래디언트 적응; 스캐폴드 인간 작성 고정 |
| Self-play FT (Chen et al.) | 2024 | ❌ | ✅ | 반복적 자기대결 미세조정; 파이프라인 정적 |
| Darwin Gödel Machine (Zhang et al.) | 2025 | ✅ | ❌ | 에이전트 소스코드의 진화적 탐색; 모델 고정 |
| TTRL (Zuo et al.) | 2025 | ❌ | ✅ | 비레이블 테스트 데이터에서 다수결 의사 보상으로 RL; 스캐폴드 없음 |
| AlphaEvolve (Novikov et al.) | 2025 | ✅ | ❌ | 과학적 발견을 위한 코딩 에이전트; 가중치 고정 |
| Discover-TTT (Yuksekgonul et al.) | 2026 | ❌ | ✅ | 엔트로픽 유틸리티 목적으로 테스트 시 가중치 훈련; SIA가 이 손실 함수를 재사용 |
| Meta-Harness (Lee et al.) | 2026 | ✅ | ❌ | 하네스 그래프의 종단간 LLM 기반 최적화; 가중치 고정 |
| Hyperagents (Zhang et al.) | 2026 | ✅ | ❌ | 메타 메커니즘 자체도 편집 가능(에이전트와 에이전트 개선자 공진화); 가중치 고정 |

**핵심 비교 통찰**:

1. **EUREKA vs SIA**: EUREKA는 스캐폴드(보상 함수)와 가중치(정책) 양쪽을 건드리지만, 보상 생성기가 훈련된 정책의 피드백으로 업데이트되지 않는 **단방향** 상호작용입니다. SIA는 **폐쇄 피드백 루프**에서 두 레버를 동적으로 조율합니다.

2. **Hyperagents vs SIA**: 가장 가까운 동시대 연구. Hyperagents는 스캐폴드 편집의 표현력을 확장하지만(메타 메커니즘도 편집 가능), 모델 가중치를 고정합니다. SIA는 이에 더해 **두 번째 가중치 기반 레버**를 추가합니다.

3. **Discover-TTT + Meta-Harness의 사실상 결합**: SIA는 Discover-TTT (가중치 업데이트 방법론)와 Meta-Harness (Harness 최적화 정신)를 단일 에이전트 루프로 통합한 것으로도 볼 수 있습니다.

---

## 참고 자료

**본 논문 (주요 출처)**:
- Hebbar, P., Manawat, Y., Verboomen, S., Ivanova, A., Palanimalai, S., Bhatia, K., & Baskaran, V. (2026). *SIA: Self Improving AI with Harness & Weight Updates*. arXiv:2605.27276v2.

**논문 내 인용 참고문헌 (관련 연구)**:
- Zelikman, E., Wu, Y., & Goodman, N. D. (2022). *STaR: Bootstrapping reasoning with reasoning*. NeurIPS 35.
- Madaan, A., et al. (2023). *Self-refine: Iterative refinement with self-feedback*. NeurIPS 36.
- Shinn, N., et al. (2023). *Reflexion: Language agents with verbal reinforcement learning*. NeurIPS 36.
- Ma, Y. J., et al. (2023). *Eureka: Human-level reward design via coding large language models*. ICLR 2023.
- Wang, G., et al. (2023). *Voyager: An open-ended embodied agent with large language models*. TMLR.
- Yao, S., et al. (2022). *ReAct: Synergizing reasoning and acting in language models*. ICLR 2023.
- Romera-Paredes, B., et al. (2023). *Mathematical discoveries from program search with large language models*. Nature, 625.
- Chen, Z., et al. (2024). *Self-play fine-tuning converts weak language models to strong language models*. ICML.
- Hu, S., Lu, C., & Clune, J. (2024). *Automated design of agentic systems*. ICLR.
- Lu, C., et al. (2024). *The AI scientist: Towards fully automated open-ended scientific discovery*. arXiv.
- Akyürek, E., et al. (2024). *The surprising effectiveness of test-time training for few-shot learning*. ICML.
- Hu, E. J., et al. (2021). *LoRA: Low-rank adaptation of large language models*. ICLR.
- Fei, Z., et al. (2023). *LawBench: Benchmarking legal knowledge of large language models*. arXiv.
- Novikov, A., et al. (2025). *AlphaEvolve: A coding agent for scientific and algorithmic discovery*. arXiv.
- van Dijk, D. V., et al. (2018). *Recovering gene interactions from single-cell data using data diffusion*. Cell, 174(3).
- Zuo, Y., et al. (2025). *TTRL: Test-time reinforcement learning*. arXiv.
- Zhang, J., et al. (2025). *Darwin Gödel machine: Open-ended evolution of self-improving agents*. SuperIntelligence.
- Zhang, J., et al. (2026). *Hyperagents*. arXiv:2603.19461.
- Lee, Y., et al. (2026). *Meta-harness: End-to-end optimization of model harnesses*. ICML.
- Yuksekgonul, M., et al. (2026). *Learning to discover at test time*. arXiv.
- Sheng, G., et al. (2024). *HybridFlow: A flexible and efficient RLHF framework*. EuroSys.
- Zheng, Y., et al. (2024). *LlamaFactory: Unified efficient fine-tuning of 100+ language models*. ACL.
- Cao, S., et al. (2025). *SkyRL-v0: Train real-world long-horizon agents via reinforcement learning*. NovaSky/UC Berkeley Technical Report.
- Karpathy, A. (2026). *autoresearch: AI agents running research on single-GPU nanochat training automatically*.
