# Safety Alignment of LMs via Non-cooperative Games (AdvGame)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **LM의 안전 정렬(Safety Alignment)을 순차적 적대적 훈련(sequential adversarial training)이 아닌, 비협력적 게임(non-cooperative game)으로 재정의해야 한다**는 것입니다.

기존 방식은 다음과 같은 순차적(sequential) 절차를 따릅니다:
1. Attacker LM 훈련 → 적대적 프롬프트 생성
2. Defender LM 훈련 → 해당 프롬프트에 방어
3. 수렴 기준 충족 시까지 반복

저자들은 이 방식이 **교대 최적화의 비효율성과 불안정성**을 내재적으로 가진다고 지적합니다. 대신, Attacker와 Defender를 **동시에(jointly)** 훈련하는 AdvGame 패러다임을 제안합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **새로운 패러다임** | 안전 정렬을 비영합(non-zero-sum) 비협력적 게임으로 정식화 |
| **AdvGame 프레임워크** | Attacker·Defender LM의 온라인 동시 RL 훈련 |
| **선호 기반 보상** | 점수(point-wise) 대신 쌍별 비교(pairwise) 판단으로 reward hacking 감소 |
| **Pareto 프론티어 이동** | 안전성과 유용성을 동시에 향상 |
| **재사용 가능한 Red-teaming Attacker** | 학습된 Attacker LM을 임의 모델 프로빙에 직접 활용 가능 |

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

**핵심 문제:** 기존 안전 후처리(safety post-training) 방식은 적응적 공격(adaptive attack)에 취약합니다. Self-play 기반 방식(예: Self-RedTeam)은 파라미터를 공유하여 다음 문제를 야기합니다:
- 목표 함수 간 간섭(gradient leakage across roles)
- Attacker가 유해한 공격 생성을 거부하는 퇴행적 해(degenerate solution)
- 탐색 다양성 감소

### 2.2 제안 방법 (수식 포함)

#### 게임 정식화

Attacker $\rho$와 Defender $\pi$의 **결합 최적화 문제**를 다음과 같이 정의합니다:

$$\max_{\rho} \mathbb{E}_{\substack{(c,s)\sim\xi \\ x\sim\rho(\cdot|c,s) \\ y\sim\pi(\cdot|x)}} \left[ R_{\text{att}}(c, s, x, y) - \beta \log \frac{\rho(x|c,s)}{\rho_{\text{ref}}(x|c,s)} \right] $$

$$\max_{\pi} \mathbb{E}_{\substack{(c,s)\sim\xi \\ x\sim\rho(\cdot|c,s) \\ y\sim\pi(\cdot|x)}} \left[ R_{\text{def}}(c, s, x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right] $$

여기서:
- $c \in \{h, b\}$: 쿼리 클래스 (harmful/benign)
- $s$: 시드 쿼리, $x$: Attacker가 생성한 공격 쿼리, $y$: Defender 응답
- $\beta$: KL 정규화 강도
- $R_{\text{att}} \neq -R_{\text{def}}$ → **비영합(non-zero-sum)** 게임

#### Bradley-Terry 모델 기반 선호도 연결

$$P_{\text{def}}(y \succ y' | c, s, x) = \sigma\left(R_{\text{def}}(c,s,x,y) - R_{\text{def}}(c,s,x,y')\right) $$

$$P_{\text{att}}\left((x,y) \succ (x',y') | c,s\right) = \sigma\left(R_{\text{att}}(c,s,x,y) - R_{\text{att}}(c,s,x',y')\right)$$

#### 선호 판사 (Preference Judge)

**Defender 판사:**

$$P_{\text{def}}(y \succ y' | c,s,x) := \begin{cases} P_{\text{compl}}(y \succ y'|s) & x \text{ faithful, } c=b \\ P_{\text{deflec}}(y \succ y'|s) & x \text{ faithful, } c=h \\ 0.5 & x \text{ not faithful} \end{cases} $$

**Attacker 판사 (보상 스왑 핵심):**

$$P_{\text{att}}\left((x,y) \succ (x',y')|c,s\right) := \begin{cases} P_{\text{compl}}(y \succ y'|s) & x,x' \text{ faithful, } c=h \\ P_{\text{deflec}}(y \succ y'|s) & x,x' \text{ faithful, } c=b \\ 1 & \text{only } x \text{ faithful} \\ 0 & \text{only } x' \text{ faithful} \\ 0.5 & \text{both not faithful} \end{cases} $$

> **핵심 설계:** Attacker는 harmful 쿼리에서 compliance를, benign 쿼리에서 deflection을 보상받아 Defender가 쿼리 카테고리를 혼동하도록 유도합니다. 이는 gibberish 생성을 방지합니다.

#### AdvGame-DPO 손실 함수

**Defender 손실:**

```math
\ell_{\text{DPO-def}}(y_w, y_l, x) := -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)\pi_{\text{ref}}(y_l|x)}{\pi_{\text{ref}}(y_w|x)\pi_\theta(y_l|x)}\right)
```

```math
\mathcal{L}_{\text{DPO-def}}(\pi_\theta) := \mathbb{E}_{\substack{(c,s)\sim\zeta \\ x\sim\text{SG}[\rho_\phi](\cdot|c,s) \\ y_1,y_2\sim\text{SG}[\pi_\theta](\cdot|x) \\ y_w,y_l\sim\lambda_{\text{def}}(y_1,y_2|c,s,x)}} \left[\ell_{\text{DPO-def}}(y_w,y_l,x)\right]
```

**Attacker 손실:**

$$\ell_{\text{DPO-att}}(x_w, x_l, c, s) := -\log\sigma\left(\beta\log\frac{\rho_\phi(x_w|c,s)\rho_{\text{ref}}(x_l|c,s)}{\rho_{\text{ref}}(x_w|c,s)\rho_\phi(x_l|c,s)}\right) $$

```math
\mathcal{L}_{\text{DPO-att}}(\rho_\phi) := \mathbb{E}_{\substack{(c,s)\sim\zeta \\ x_1,x_2\sim\text{SG}[\rho_\phi](\cdot|s) \\ y_1\sim\text{SG}[\pi_\theta](\cdot|x_1) \\ y_2\sim\text{SG}[\pi_\theta](\cdot|x_2) \\ x_w,x_l\sim\lambda_{\text{att}}((x_1,y_1),(x_2,y_2)|c,s)}} \left[\ell_{\text{DPO-att}}(x_w,x_l,c,s)\right]
```

#### EMA (Exponential Moving Average) 오프-폴리시 안정화

$$\pi^\gamma_{t+1} = (1-\gamma)\pi^\gamma_t + \gamma\pi_\theta $$

$$\rho^\gamma_{t+1} = (1-\gamma)\rho^\gamma_t + \gamma\rho_\phi$$

여기서 $\gamma = 0.95$를 사용합니다.

#### IPO-MD 변형 (BT 모델 가정 불필요)

$$\mathcal{L}_{\text{IPO-MD-def}}(\pi_\theta) = \mathbb{E}\left[\left(\beta\log\frac{\pi_\theta(y_w|x)\pi_{\text{ref}}(y_l|x)}{\pi_{\text{ref}}(y_w|x)\pi_\theta(y_l|x)} - \frac{1}{2\beta}\right)^2\right] $$

### 2.3 모델 구조

```
[시드 프롬프트 s (harmful/benign)]
         ↓
  [Attacker LM ρ_φ]  ←─ 파라미터 비공유 ─→  [Defender LM π_θ]
  두 공격 쿼리 생성                            두 응답 생성
  x₁, x₂ ~ ρ_φ(·|c,s)                      y¹,y² ~ π_θ(·|xᵢ)
         ↓                                          ↓
  [Faithfulness Judge]              [Compliance/Deflection Judge]
         ↓                                          ↓
  [Attacker DPO 손실]              [Defender DPO 손실]
         ↓                                          ↓
         └──────────→ [EMA 업데이트] ←──────────────┘
```

**사용 모델:**
- **Defender:** Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct
- **Attacker:** Qwen2.5-7B (Qwen 실험), Llama3.1-8B-Abliterated (Llama 실험)
- **Judge:** Llama3.3-70B-Abliterated (Llama), Qwen2.5-32B (Qwen)
- **훈련 데이터:** WildJailbreak 데이터셋 (vanilla harmful + vanilla benign, 80/20 분할)
- **평가 judge:** GPT-4o

### 2.4 성능 향상

#### 유용성 (Utility, ↑)

| 모델 | 방법 | MMLU@5 | IFBench | AlpacaEval2 | ArenaHard |
|------|------|--------|---------|-------------|-----------|
| Qwen2.5-7B | Original | 71.8 | 29.4 | 29.9 | 55.5 |
| | Self-RedTeam | 71.9 | **25.9**🔴 | 28.6 | **52.2**🔴 |
| | **AdvGame-DPO-MD** | **71.8** | **30.7** | 27.6 | **61.3** |
| Llama3.1-8B | Original | 69.3 | 28.2 | 29.9 | 33.6 |
| | Self-RedTeam | **64.8**🔴 | **22.3**🔴 | **13.1**🔴 | **13.6**🔴 |
| | **AdvGame-DPO-MD** | **69.1** | 26.4 | **32.8** | **41.0** |

#### 안전성 (Safety, ASR↓)

| 모델 | 방법 | HarmBench | WJB | DAN | WildGuardTest |
|------|------|-----------|-----|-----|---------------|
| Qwen2.5-7B | Original | 31.6 | 69.6 | 36.3 | 38.6 |
| | Self-RedTeam | 16.8 | 41.1 | 36.3 | 22.0 |
| | **AdvGame-DPO-MD** | **4.7** | **8.5** | 10.3 | **1.2** |
| Llama3.1-8B | Original | 25.0 | 58.6 | 49.3 | 29.7 |
| | Self-RedTeam | 14.9 | 11.5 | 32.3 | 9.8 |
| | **AdvGame-DPO-MD** | **7.4** | **6.4** | 42.0 | **2.1** |

#### 적응적 공격에 대한 강건성 (ASR↓, HarmBench 테스트셋)

| 방어 모델 | PAIR | TAP | GCG | AdvGame-Attacker |
|----------|------|-----|-----|-----------------|
| Qwen Original | 45.0 | 48.8 | 61.6 | 55.6 |
| Self-RedTeam | 37.2 | 40.3 | 23.4 | 40.9 |
| **AdvGame-DPO-MD** | **7.2** | **10.0** | 25.3 | **11.3** |

### 2.5 한계

1. **자연어 공격에만 집중:** GCG와 같은 비가독성 토큰 공격은 방어 범위에서 제외됩니다. 퍼플렉시티 기반 분류기로 별도 방어 필요
2. **DAN 벤치마크 취약성:** Llama DAN ASR이 42.0%로 일부 단일 훈련 방어보다 높습니다 (LAT: 0.0%)
3. **컴플라이언스 저하:** Llama의 WJB-benign 컴플라이언스가 98.8% → 69.9%로 크게 감소
4. **높은 계산 비용:** Attacker·Defender 동시 훈련 + 대형 judge 모델 (32B, 70B) 필요
5. **GRPO 불안정성:** GRPO 기반 변형은 DPO/IPO 대비 훈련이 불안정하고 성능이 낮음
6. **Attacker 모델 보안:** 학습된 Attacker 체크포인트는 제어된 환경에서만 관리되어야 함

---

## 3. 일반화 성능 향상 가능성

이 논문에서 일반화 성능 향상과 관련된 핵심 메커니즘들을 상세히 분석합니다.

### 3.1 동적 공격 다양성을 통한 일반화

기존 정적 데이터셋 기반 안전 훈련은 훈련 시 보지 못한 공격 패턴에 취약합니다. AdvGame은 Attacker가 **온라인으로 진화하는 공격 프롬프트를 생성**하여 Defender가 고정된 공격 패턴에 과적합하지 않도록 합니다.

$$x \sim \rho_\phi(\cdot|c,s) \quad \text{(온라인으로 변화하는 공격 분포)}$$

이는 Madry et al. (2018)의 적대적 훈련 원리를 언어 모델에 적용한 것으로, **훈련 시 보지 못한 공격에도 강건성을 제공하는 핵심 기제**입니다.

### 3.2 Nash 균형으로의 수렴과 일반화

저자들은 두 최적화 문제가 **강한 오목성(strongly concave)** 조건을 만족함을 증명합니다:

$$\pi^*(\cdot|c,s,x) = \arg\max_\pi \mathbb{E}_{y\sim\pi(\cdot|x)}\left[R_{\text{def}}(c,s,x,y)\right] - \beta D_{\text{KL}}(\pi\|\pi_{\text{ref}}|x) $$

```math
\rho^*(\cdot|c,s) = \arg\max_\rho \mathbb{E}_{\substack{x\sim\rho(\cdot|c,s)\\y\sim\pi^*(\cdot|c,s,x)}}\left[R_{\text{att}}(c,s,x,y)\right] - \beta D_{\text{KL}}(\rho\|\rho_{\text{ref}}|c,s)
```

유일한 Nash 균형 $(\rho^\*, \pi^*)$의 존재는 **어떤 Attacker 전략에도 최적으로 대응하는 Defender**를 의미하며, 이는 이론적 일반화 보장입니다.

### 3.3 KL 정규화의 일반화 기여

KL 발산 항 $\beta D_{\text{KL}}(\pi\|\pi_{\text{ref}})$은 최적 해를 참조 모델 근방(인간 가독 텍스트 부분공간)으로 제한합니다:

$$\pi^*(y|c,s,x) = \frac{1}{Z(x,s)}\exp\left(\frac{1}{\beta}R_{\text{def}}(c,s,x,y)\right)\pi_{\text{ref}}(y|x) $$

이는 **과최적화(over-optimization)를 방지**하고, 최적화 문제를 인간 가독 텍스트의 저차원 부분공간으로 제한하여 더 tractable하게 만듭니다.

### 3.4 쌍별 선호 판사의 일반화 기여

점수 기반(point-wise) 판사 대비 쌍별(pairwise) 판사의 일반화 이점:

**실험적 증거 (Table 5):**

| 방법 | HarmBench↓ | WJB↓ | ArenaHard↑ |
|------|-----------|------|-----------|
| +point-wise judge | 14.1 | 30.8 | 56.3 |
| **pair-wise judge (AdvGame-DPO-MD)** | **4.7** | **8.5** | **61.3** |

점수 기반 판사는 **reward hacking에 취약**하여 모델이 판사의 약점을 이용해 실제로는 unsafe하지만 높은 점수를 받는 응답을 학습합니다. 쌍별 판사는 절대적 보정(absolute calibration) 없이 **상대적 비교만 필요**하므로 이러한 과적합을 방지합니다.

### 3.5 EMA 오프-폴리시 학습의 일반화 기여

**실험적 증거 (Table 5):**

| 방법 | HarmBench↓ | WJB↓ |
|------|-----------|------|
| +no EMA (on-policy) | 19.9 | 49.5 |
| **EMA (off-policy)** | **4.7** | **8.5** |

EMA는 최근 변화에 과적합하는 on-policy 학습의 불안정성을 완화합니다:

$$\pi^\gamma_{t+1} = (1-\gamma)\pi^\gamma_t + \gamma\pi_\theta$$

이는 **훈련 안정성을 높여 다양한 공격 패턴에 대한 일반화**를 가능하게 합니다.

### 3.6 Attacker 공동 훈련의 일반화 기여

**실험적 증거 (Table 5):**

| 방법 | HarmBench↓ | WJB↓ | DAN↓ |
|------|-----------|------|------|
| +fixed attacker | 5.1 | 16.6 | 15.0 |
| **trained attacker** | **4.7** | **8.5** | **10.3** |

Attacker를 고정하면 Defender가 다양한 공격에 노출되지 않아 일반화가 제한됩니다. **Attacker의 진화적 훈련은 Defender에게 더 다양하고 어려운 공격 환경을 제공**하여 강건성을 향상시킵니다.

### 3.7 Attacker의 범용 Red-teaming 능력 (전이 일반화)

학습된 Attacker는 **단순 LLM 생성** (temperature=1, max_tokens=500)만으로도 강력한 red-teaming 성능을 보입니다:

| 대상 모델 | PAIR | TAP | GCG | **AdvGame-Attacker** |
|---------|------|-----|-----|---------------------|
| Qwen Original | 45.0 | 48.8 | 61.6 | **55.6** |
| Llama Original | 42.5 | 49.0 | 42.8 | 34.4 |

이는 Attacker가 특정 Defender에만 과적합하지 않고 **임의의 목표 모델에 전이 가능한 일반화된 공격 전략**을 학습했음을 보여줍니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 게임 이론적 안전 정렬의 새로운 기준점 수립
AdvGame은 LM 안전 정렬을 Nash 균형 문제로 공식화한 최초의 실용적 프레임워크 중 하나입니다. 이는 향후 연구의 이론적 기준점이 될 것입니다.

#### (2) 선호 최적화와 안전성의 결합
DPO/IPO의 preference 최적화를 adversarial safety training에 결합한 것은, **RLHF 연구와 안전 연구의 융합**을 촉진할 것입니다.

#### (3) Red-teaming 자동화의 새로운 패러다임
학습된 Attacker의 범용 red-teaming 능력은 **AI 안전 평가 파이프라인을 자동화**하는 방향으로 연구를 촉진할 것입니다.

#### (4) 멀티-에이전트 안전 시스템 설계
서로 다른 역할을 가진 여러 LM을 공동 훈련하는 접근법은 **다중 에이전트 AI 시스템의 안전성 연구**에 영향을 줄 것입니다.

### 4.2 앞으로의 연구 시 고려할 점

#### (1) 보상 모델 설계 개선
쌍별 판사가 점수 기반보다 우수하지만, 여전히 보상 해킹의 가능성이 있습니다. **더 정교한 judge 설계** (예: Constitutional AI 기반 judge, 다중 judge 앙상블)가 필요합니다.

#### (2) 비자연어 공격 방어 통합
현재 AdvGame은 자연어 공격에만 집중합니다. **GCG 같은 최적화 기반 공격**, 멀티모달 공격, 프롬프트 인젝션 등으로 방어 범위를 확장할 필요가 있습니다.

#### (3) 컴플라이언스-안전성 균형 최적화
Llama의 WJB-benign 컴플라이언스 저하(69.9%)는 중요한 문제입니다. **다양한 benign-but-jailbreak-structured 프롬프트**로 훈련 데이터를 보강하거나, 컴플라이언스 손실에 대한 별도의 정규화 항을 도입할 필요가 있습니다.

#### (4) 계산 효율화
Attacker(7-8B) + Defender(7-8B) + Judge(32-70B)의 동시 운용은 고비용입니다. **지식 증류(Knowledge Distillation)나 소형 judge** 모델 활용, 비동기 훈련 전략이 연구될 필요가 있습니다.

#### (5) 다중 라운드 및 다중 에이전트 확장
현재는 단일 라운드 공격-방어 구조입니다. **Tree of Attacks (TAP) 스타일의 다중 라운드 게임**이나 여러 Attacker가 협력하는 다중 에이전트 설정으로의 확장이 연구될 수 있습니다.

#### (6) 이론적 수렴 보장
현재 논문은 테이블 설정(tabular setting)에서의 Nash 균형 존재성을 증명하지만, **신경망 파라미터화된 대규모 LM에서의 수렴 보장**은 여전히 열린 문제입니다.

#### (7) Attacker 아티팩트 관리
학습된 Attacker는 강력한 red-teaming 도구이지만, **악용 가능성**도 높습니다. Attacker 체크포인트를 공개할지 여부, 어떤 조건 하에 접근을 허용할지에 대한 연구 윤리 기준 정립이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 방법론 | 게임 구조 | 파라미터 공유 | 판사 유형 | 한계 |
|------|--------|----------|-------------|---------|------|
| **Zou et al. (2023)** GCG | 그래디언트 기반 프롬프트 최적화 | 없음 | N/A | 없음 | 비자연어, 방어 취약 |
| **Paulus et al. (2025)** AdvPrompter | Attacker LM 순차 훈련 | 순차적 | 없음 | 점수 기반 | 순차적 불안정성 |
| **Liu et al. (2025a)** Self-RedTeam | 자기 대결 | 영합(zero-sum) | **공유** | 점수 기반(GRPO) | 목표 함수 간섭, 퇴행해 |
| **Rafailov et al. (2023)** DPO | 오프라인 선호 최적화 | 없음 | N/A | 쌍별 | 정적 데이터셋 |
| **Munos et al. (2024)** Nash-MD | 맥스민 게임 | 영합 | 공유 | 쌍별 | 안전 정렬 미적용 |
| **Wang et al. (2025)** LifelongSA | 순차적 교대 훈련 | 비영합 | **없음** | 점수 기반(RFT) | 매 반복 처음부터 재훈련 |
| **Zou et al. (2024)** Circuit Breakers | 표현 공학 | 없음 | N/A | 없음 | Attacker 동적 진화 미고려 |
| **AdvGame (본 논문)** | 동시 비협력적 게임 | **비영합** | **없음** | **쌍별** | 비자연어 공격 미지원 |

### 핵심 차별점 정리

1. **vs Self-RedTeam:** 파라미터 비공유, 비영합 게임, 쌍별 판사 → 퇴행해 방지
2. **vs LifelongSA:** 온라인 동시 훈련, EMA 오프-폴리시 → 처음부터 재훈련 불필요
3. **vs Nash-MD:** 동일 역할이 아닌 명확히 구분된 Attacker-Defender 역할 분리
4. **vs GCG/PAIR:** Attacker가 LLM 기반 자연어 공격 생성 → 전이 가능성 높음

---

## 참고 자료

**주요 논문 (PDF 원문 직접 참조):**

- **Paulus, A., Kulikov, I., Amos, B., Munos, R., Evtimov, I., Chaudhuri, K., & Zharmagambetov, A. (2026).** "Safety Alignment of LMs via Non-cooperative Games." *Proceedings of the 43rd International Conference on Machine Learning.* PMLR 306. arXiv:2512.20806v3

**논문 내 인용 참고문헌:**

- Rafailov, R., et al. (2023). "Direct preference optimization: Your language model is secretly a reward model." *NeurIPS.*
- Azar, M. G., et al. (2024). "A general theoretical paradigm to understand learning from human preferences." *AISTATS.*
- Munos, R., et al. (2024). "Nash learning from human feedback." *ICML.*
- Calandriello, D., et al. (2024). "Human alignment of large language models through online preference optimisation." *ICML.*
- Liu, M., et al. (2025a). "Chasing moving targets with online self-play reinforcement learning for safer language models." (Self-RedTeam)
- Zou, A., et al. (2023). "Universal and transferable adversarial attacks on aligned language models."
- Zou, A., et al. (2024). "Improving alignment and robustness with circuit breakers." *NeurIPS.*
- Mazeika, M., et al. (2024). "HarmBench: A standardized evaluation framework for automated red teaming and robust refusal." *ICML.*
- Paulus, A., et al. (2025). "AdvPrompter: Fast adaptive adversarial prompting for LLMs." *ICML.*
- Wang, H., et al. (2025). "Lifelong safety alignment for language models." *NeurIPS.*
- Shao, Z., et al. (2024). "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models." (GRPO)
- Bradley, R. A., & Terry, M. E. (1952). "Rank analysis of incomplete block designs." *Biometrika.*
- Nash, J. F. (1950). "Equilibrium points in n-person games." *PNAS.*
- Madry, A., et al. (2018). "Towards deep learning models resistant to adversarial attacks." *ICLR.*

**코드 저장소:** https://github.com/facebookresearch/advgame
