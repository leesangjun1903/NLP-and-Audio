# Deterministic Differentiable Structured Pruning for Large Language Models

> **⚠️ 주의**: 본 논문은 arXiv:2603.08065v2 (2026년 5월 11일 게재)로, ICML 2026 발표 예정 논문입니다. 인용 시 최신 버전을 확인하세요.

---

## 1. Executive Summary (10문장 이내)

이 논문은 대형 언어 모델(LLM)의 추론 비용을 줄이기 위한 **구조적 가지치기(Structured Pruning)** 방법론인 **DDP(Deterministic Differentiable Pruning)**를 제안한다.  
핵심 문제의식은 기존 방법들이 채택한 **하드-콘크리트 확률적 완화(Hard-Concrete Stochastic Relaxation)** 기법이 학습-추론 불일치(Train-Test Mismatch)와 마스크 표현력 제한이라는 두 가지 근본적 결함을 지닌다는 점이다.  
DDP는 확률적 샘플링 없이 결정론적 소프트 대리 함수(Deterministic Soft Surrogate)를 $\ell_0$ 목적함수에 직접 최적화함으로써 이 문제를 해결한다.  
사전 학습된 가중치는 동결(Frozen)하고 마스크 변수만 최적화하는 **마스크 전용 최적화(Mask-Only Optimization)** 방식을 채택해, LoRA보다 훨씬 적은 파라미터(~0.35M)로 효율적인 학습이 가능하다.  
스파스성 제약은 증강 라그랑지안 방법(Augmented Lagrangian Method, ALM)으로 강제하며, 이진화 손실(Binarization Loss)을 추가해 수렴을 가속한다.  
또한, 순전파 마스크와 정규화용 보존 점수를 분리(Decoupling)하여 마스크 값의 범위를 근이진(near-binary) 제약에서 해방시켜 표현력을 높인다.  
LLaMA, Qwen3, DeepSeekMoE 등 다양한 Dense/MoE 모델에서 검증하였으며, 20% 스파스성에서 성능 손실 1% 이내를 달성하고 기존 최강 기준선을 일관되게 능가한다.  
특히 DeepSeekMoE-16B의 60% 스파스성에서 기존 최강 기준선 대비 평균 정확도 +6.6점 향상을 기록했다.  
vLLM 기반 실제 배포 환경에서 LLaMA-7B 50% 스파스성 기준 2.20× 추론 속도 향상을 실증적으로 확인했다.  
이 연구는 경량화와 품질 사이의 실질적 균형점을 제시한다는 점에서 LLM 배포 효율화 분야에서 중요한 기여를 한다.

> 📌 **용어 설명**
> - **구조적 가지치기(Structured Pruning)**: 모델의 어텐션 헤드, MLP 채널 등 전체 구성 요소를 제거하는 방식으로, 특수 하드웨어 없이 표준 연산으로 속도 향상 가능
> - **하드-콘크리트 완화**: 이산(discrete) $\ell_0$ 노름을 미분 가능하게 만들기 위해 확률적 노이즈를 도입하는 기법 (Louizos et al., 2018)
> - **마스크 전용 최적화**: 사전 학습 가중치는 고정하고, 각 구성 요소를 유지/제거할지 결정하는 게이팅 변수(마스크)만 학습

### 1-1. 연구의 목적과 필요성

**배경**: LLM 배포는 막대한 연산·메모리·서빙 자원을 요구하며, 특히 예산 제약 환경에서 실질적 장벽이 됩니다.

**기존 방법의 한계**:

| 기존 방법 유형 | 한계 |
|---|---|
| 원샷 휴리스틱 가지치기 | 수작업 중요도 지표에 의존 → 공격적 스파스성에서 품질 급락 |
| 스파스성 인식 학습 (전체/LoRA 파인튜닝) | 가중치 업데이트 포함 → 수십억 토큰 필요, 비용 과다 |
| 기존 마스크 학습 (하드-콘크리트) | ① 학습-추론 불일치, ② 마스크 표현력 제한 (근이진 범위만 허용) |

**연구 목적**: 사전 학습 가중치를 동결한 채, 소량의 토큰(<30M)으로 고품질 구조적 스파스성 패턴을 **결정론적·미분가능하게** 탐색하는 방법 제안.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 (논문 내) | 위치 |
|---|---|---|
| 하드-콘크리트의 확률성이 학습-추론 불일치를 유발 | 학습 시 샘플링된 마스크 vs. 추론 시 이산화된 마스크 간 불일치로 스파스성 제어 불안정 | p.3, Section 2.3 |
| 하드-콘크리트가 마스크 표현력을 제한 | 마스크를 $[0,1]$ 근이진 범위로 강제 → 검색 공간 축소 | p.3, Section 2.3 |
| ReLU 게이트로 마스크 범위 확장 시 성능 향상 | Ablation: Det. HC → Det. HC+EM으로 LLaMA-7B PPL 16.30→15.36 | p.7, Table 4 |
| 결정론적 어닐링 서로게이트가 수렴 개선 | HC(16.52)→Det.HC(16.30)→Ours(15.20) PPL 순차적 개선 | p.7, Table 4 |
| 마스크 전용 최적화가 LoRA 복구보다 우수 | 30M 토큰 DDP(64.82) > 120M 토큰 LoRAP(62.98) on LLaMA-2-7B | p.8, Table 7 |
| DDP가 Dense/MoE 모두에서 최강 기준선 능가 | LLaMA, DeepSeekMoE, Qwen3 전 모델·스파스성 수준에서 우위 | p.6-7, Tables 2, 3 |
| 실제 배포 환경에서 속도 향상 실증 | LLaMA-7B 50%: 2.20×, Qwen3-30B-A3B 60%: 1.51× | p.9, Tables 10, 11 |
| 지식 증류(KD)가 자연스럽게 통합 가능 | 사전 학습 가중치 동결 → Dense 모델이 추가 비용 없이 교사 역할 | p.5, Section 3.3 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제

기존 하드-콘크리트 기반 마스크 학습의 두 가지 핵심 결함:

**① 학습-추론 불일치 (Train-Test Mismatch)**

$$\min_z \mathbb{E}_{\boldsymbol{u} \sim U(0,1)}\left[\mathcal{L}_{ce}(\theta, \boldsymbol{m}) + \mathcal{L}_{\text{sparsity}}(\|\boldsymbol{m}\|_0)\right] $$

- 학습 시: $\boldsymbol{m} = \Phi(\boldsymbol{z}, \boldsymbol{u})$ (확률적 샘플링)
- 추론 시: 이산(결정론적) 마스크 필요
- 결과: 스파스성 제어 불안정, 성능 저하

**② 마스크 표현력 제한**

하드-콘크리트 매핑:
$$\boldsymbol{u} \sim U(0,1),\quad \boldsymbol{v} = \sigma\!\left(\log u - \log(1-u) + z\right), \quad \bar{v} = v(r-l)+l, \quad \boldsymbol{m} = \text{Clamp}(\bar{v}, 0, 1) $$

마스크를 $[0,1]$ 근이진 범위로 강제 → 중요 구성 요소의 세밀한 스케일링 불가

---

### 제안하는 방법: DDP

#### (A) 순전파 결정론적 ReLU 게이트

$$\boldsymbol{m} = \text{ReLU}(\boldsymbol{z}) $$

- $\boldsymbol{z} \in \mathbb{R}^K$: 최적화되는 잠재 파라미터(마스크 로짓)
- $m_k = 0$: $k$번째 구성 요소 제거; $m_k > 0$: 유지 및 스케일링
- 마스크 범위: $m_k \in [0, +\infty)$ → 표현력 대폭 확장

> 📌 **용어 설명**
> - **잠재 파라미터(Latent Parameter)** $z$: 직접 최적화되는 실수값으로, ReLU를 통과해 실제 마스크 값이 됨
> - **ReLU**: $\text{ReLU}(x) = \max(0, x)$, 음수 입력을 0으로 만들어 자동으로 가지치기 결정

#### (B) 결정론적 어닐링 보존 점수 (Retention Score)

$\ell_0$ 노름의 미분 불가능성을 확률적 샘플링 없이 해결:

$$\boldsymbol{v} = \sigma\!\left(\frac{(z - \mu_t) C_0}{\mu_t}\right),\quad \bar{\boldsymbol{v}} = \boldsymbol{v}(r-l)+l,\quad \boldsymbol{s} = \text{Clamp}(\bar{\boldsymbol{v}}, 0, 1) $$

- $\mu_t$: 어닐링 선명도 파라미터 (훈련 진행에 따라 감소)
- $\sigma(\cdot)$: 시그모이드 함수
- $l = -0.1,\ r = 1.1,\ C_0 \approx 2.4$: 스트레칭 파라미터 (고정값)
- $s_k \in [0,1]$: $k$번째 구성 요소의 보존 강도 측정값
- $s_k(0) = 0,\ s_k(2\mu_t) = 1$ 보장

> 📌 **용어 설명**
> - **어닐링(Annealing)**: 훈련 초기에는 소프트(부드러운) 결정, 후기에는 하드(이진에 가까운) 결정을 내리도록 $\mu_t$를 점진적으로 0에 가깝게 감소시키는 기법

어닐링 스케줄:
$$\mu_t = \mu_0 - (\mu_0 - \mu_T)\sqrt{\frac{t}{T}} $$

- $\mu_0 = 0.5$: 초기값
- $\mu_T \approx 0.05$: 최종값
- $t$: 현재 스텝, $T$: 전체 스텝 수

#### (C) 증강 라그랑지안 스파스성 손실

$$\mathcal{L}_{\text{sparsity}}(\boldsymbol{s}) = \lambda_1(\bar{s} - \rho) + \lambda_2(\bar{s} - \rho)^2 $$

- $\bar{s} = \frac{1}{K}\sum_k s_k$: 평균 보존 점수
- $\rho$: 목표 keep ratio (유지 비율)
- $\lambda_1$: 라그랑주 승수 (Lagrange Multiplier) — 등식 제약 강제
- $\lambda_2$: 이차 페널티 계수 — 제약 위반에 비례하는 추가 패널티
- $\lambda_1, \lambda_2$는 경사 상승(Gradient Ascent)으로 동적 갱신

> 📌 **용어 설명**
> - **증강 라그랑지안 방법(ALM)**: 등식 제약이 있는 최적화를 제약 없는 문제로 변환하는 기법. $\lambda_1(\bar{s}-\rho)$가 1차 제약, $\lambda_2(\bar{s}-\rho)^2$이 2차 페널티 역할

#### (D) 이진화 손실 (Binarization Loss)

$$\mathcal{L}_{\text{bin}}(\boldsymbol{s}) = \lambda_3 \frac{1}{K} \sum_{k=1}^K s_k(1-s_k) $$

- $s_k(1-s_k)$: $s_k \in \{0,1\}$이면 0, 중간값($s_k = 0.5$)이면 최대 0.25
- 각 마스크를 빠르게 0 또는 1로 polarize (양극화)
- 수렴 속도 가속 효과

#### (E) 지식 증류 (Knowledge Distillation)

$$\mathcal{L}_{kl}(\boldsymbol{m}, \theta) = \sum_i D_{KL}\!\left(P_t(\mathbf{X}, i) \,\|\, P_s(\mathbf{X}, i)\right) $$

- $P_t(\mathbf{X}, i)$: 교사(Dense) 모델의 위치 $i$에서의 다음 토큰 분포
- $P_s(\mathbf{X}, i)$: 학생(Pruned) 모델의 다음 토큰 분포
- 추가 메모리/파라미터 없이 Dense 모델을 교사로 활용 가능

#### (F) 최종 학습 목적함수

$$\min_{\boldsymbol{z}} \mathcal{L}_{ce}(\theta, \boldsymbol{m}) + \mathcal{L}_{\text{sparsity}}(\boldsymbol{s}) + \mathcal{L}_{\text{bin}}(\boldsymbol{s}) $$

(증류 포함 시):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{ce}(\theta, \boldsymbol{m}) + \eta \mathcal{L}_{kl}(\theta, \boldsymbol{m}) + \mathcal{L}_{\text{sparsity}}(\boldsymbol{s}) + \mathcal{L}_{\text{bin}}(\boldsymbol{s})$$

- $\eta$: 증류 가중치 (기본값: 2, 20% 스파스성 기준)
- $\boldsymbol{m} = \text{ReLU}(\boldsymbol{z})$: 순전파 마스크
- $\boldsymbol{s} = \phi(\boldsymbol{z}; \mu)$: 정규화용 보존 점수 (**분리된** 경로)

#### (G) MoE 모델 확장

$$\text{MoE}(\mathbf{X}) = \sum_{e=1}^E \pi_e(\mathbf{X}) \sum_{j=1}^C m_{e,j} f_{e,j}^{\text{mlp}}(\mathbf{X}) $$

- $E$: 전문가(Expert) 수
- $\pi_e(\mathbf{X})$: 라우터(Router)가 결정하는 전문가 $e$의 가중치
- $m_{e,j}$: 전문가 $e$의 채널 $j$에 대한 마스크
- 어텐션 블록은 유지, 전문가 MLP 채널만 가지치기

> 📌 **용어 설명**
> - **MoE (Mixture-of-Experts)**: 입력에 따라 여러 전문가 네트워크 중 일부만 활성화하는 구조. DeepSeekMoE, Qwen3-A3B가 이 구조
> - **라우터(Router)**: 각 입력에 대해 어느 전문가를 활성화할지 결정하는 게이팅 네트워크

#### (H) 세밀한 스파스성 제어 (그룹별)

$$\mathcal{L}_{\text{sparsity}}(\boldsymbol{s}) = \frac{1}{|\mathcal{G}|} \sum_{g \in \mathcal{G}} \left[\lambda_1(\bar{s}_g - \rho) + \lambda_2(\bar{s}_g - \rho)^2\right] $$

- $\mathcal{G}$: 그룹 집합 (레이어별, 전문가별 등)
- $\bar{s}\_g = \frac{1}{|g|}\sum_{k \in g} s_k$: 그룹 $g$의 평균 보존 점수

---

### 모델 구조

**Dense 모델 마스킹:**

$$y = \sum_{k=1}^K m_k f_k(\mathbf{X}),\quad m_k \in \mathbb{R} $$

- 멀티헤드 어텐션: $\boldsymbol{m}^{\text{attn}} \in \mathbb{R}^{H \times L}$ ($H$: 헤드 수, $L$: 레이어 수)
- MLP 채널: $\boldsymbol{m}^{\text{mlp}} \in \mathbb{R}^{C \times L}$ ($C$: 중간 채널 폭)

**어텐션 헤드 기여:**

$$f_h^{\text{attn}}(\mathbf{X}) = \text{Attn}\!\left(\mathbf{X}\mathbf{W}_Q^{(h)},\ \mathbf{X}\mathbf{W}_K^{(h)},\ \mathbf{X}\mathbf{W}_V^{(h)}\right)\mathbf{W}_O^{(h)} $$

**MLP 채널 기여 (Gated MLP):**

$$f_j^{\text{mlp}}(\mathbf{X}) = \left(\varphi(\mathbf{X}\mathbf{u}_j) \odot (\mathbf{X}\mathbf{g}_j)\right)\mathbf{v}_j $$

- $\mathbf{u}_j, \mathbf{g}_j$: Up/Gate 투영 열벡터
- $\mathbf{v}_j$: Down 투영 행벡터
- $\varphi(\cdot)$: GELU 등 원소별 비선형 함수
- $\odot$: 원소별 곱

---

### 성능 향상 (p.6-9, Tables 2, 3, 7, 8, 10, 11)

| 모델 | 스파스성 | 기준선 최고 Mean Acc | DDP Mean Acc | 향상폭 |
|---|---|---|---|---|
| LLaMA-7B | 20% | 62.41 (SlimLLM) | **64.13** | +1.72pp |
| LLaMA-7B | 50% | 53.16 (SlimLLM) | **56.07** | +2.91pp |
| LLaMA-13B | 50% | 58.20 (LoRAP) | **62.14** | +3.94pp |
| DeepSeekMoE-16B | 60% | 51.62 (Camera-P) | **58.18** | +6.56pp |
| Qwen3-30B-A3B | 60% | 59.03 (Camera-P) | **63.35** | +4.32pp |
| LLaMA-7B | 50% | — | 2.20× 추론 속도 | — |

### 한계

1. **가중치 업데이트 미포함**: 마스크만 최적화 → 고스파스성(>50%)에서 정확도 회복 한계
2. **일반화 평가 제한**: FineWeb-Edu 데이터 기반 학습 → 데이터셋 불일치 시 성능 저하 (Table 6)
3. **이론적 수렴 조건**: KKT 수렴 보장이 온화한 가정(mild conditions) 하에서만 성립 (Section B)
4. **비교 불공정 가능성**: MoE 기준선들은 학습 없는(training-free) 방법이지만, DDP는 30M 토큰 학습 사용 (Section 5.1)
5. **하드웨어 의존성**: vLLM 기반 속도 향상이 GPU 종류에 크게 의존 (RTX 5090 vs. B200)

---

## 3. 페이지/Figure/Table 번호 매핑

| 주장/내용 | 위치 |
|---|---|
| Unified Masking Formulation ($\ell_0$ 제약 최적화) | p.2, Section 2.1, Eq. (1)-(6) |
| 하드-콘크리트 완화 공식 | p.3, Section 2.2, Eq. (8)-(9) |
| 하드-콘크리트의 두 가지 결함 | p.3, Section 2.3 |
| DDP 결정론적 ReLU 게이트 | p.3, Section 3.1, Eq. (11) |
| 어닐링 보존 점수 매핑 | p.3-4, Eq. (12)-(14), Figure 2 |
| 이진화 손실 | p.4, Eq. (15) |
| 최종 목적함수 | p.4, Eq. (16) |
| KKT 수렴 정리 | p.5, Theorem 3.1 |
| 지식 증류 (KD) | p.5, Eq. (17) |
| MoE 확장 | p.5, Eq. (18) |
| Dense LLM 성능 비교 | p.6, Table 2; p.19, Table 12 |
| MoE LLM 성능 비교 | p.6-7, Table 3; p.21, Table 14 |
| 구성 요소별 Ablation | p.7, Table 4 |
| 스파스성 세분화(Granularity) Ablation | p.7, Table 5 |
| 토큰 수 영향 | p.7-8, Figure 3 |
| 데이터셋 Ablation | p.8, Table 6 |
| 마스크 전용 vs. LoRA 복구 비교 | p.8, Table 7 |
| Tyr-the-Pruner 추가 비교 | p.9, Table 8 |
| 계산 비용 비교 | p.9, Table 9 |
| 추론 속도 향상 | p.9, Tables 10-11 |
| 스파스성 패턴 시각화 | p.21-22, Figures 4-7 |
| Qwen3 Dense 결과 | p.20, Table 13 |

---

## 4. 저자 보고 결과 vs. 내 해석 분리

### 저자가 직접 보고한 결과

**연구 주제**: $\ell_0$ 제약 마스크 최적화로서의 구조적 가지치기, 결정론적 미분가능 최적화 프레임워크

**방법 (저자 보고):**
- Forward: $\boldsymbol{m} = \text{ReLU}(\boldsymbol{z})$ (Eq. 11)
- 보존 점수: $\boldsymbol{s} = \phi(\boldsymbol{z}; \mu_t)$ (Eq. 12)
- 전체 손실: $\mathcal{L}\_{\text{total}} = \mathcal{L}\_{ce} + \eta\mathcal{L}\_{kl} + \mathcal{L}\_{\text{sparsity}} + \mathcal{L}_{\text{bin}}$

**결과 (저자 보고):**
- Table 2: LLaMA-7B 20% 스파스성에서 DDP 64.13 vs. SlimLLM 62.41 (Mean Acc)
- Table 3: DeepSeekMoE-16B 60% 스파스성에서 DDP 58.18 vs. Camera-P 51.62 (Mean Acc)
- Table 7: 30M 토큰 DDP(64.82) > 120M 토큰 LoRAP(62.98) on LLaMA-2-7B
- Table 9: DDP 0.35M 파라미터, 18GB 메모리, 2.1s/스텝 (LoRA 20M 파라미터, 2.4s/스텝)
- Table 10: LLaMA-7B 50% 스파스성 → 2.20× 속도 향상 (RTX 5090)

### 내 해석

**방법론적 통찰**: 순전파 마스크( $\boldsymbol{m}$ )와 정규화용 보존 점수( $\boldsymbol{s}$ )를 **의도적으로 분리**한 것은 이 논문의 핵심 기여 중 하나입니다. 이 분리가 없다면 확장된 마스크 범위( $[0, +\infty)$ )와 $[0,1]$에 한정된 스파스성 제어를 동시에 달성할 수 없습니다.

**어닐링 설계의 중요성**: $\mu_t \to 0$으로 어닐링할수록 $\phi(\boldsymbol{z}; \mu_t) \to \mathbb{I}[z > 0]$이 됨으로써, 학습 종료 시점에 소프트 완화가 정확한 $\ell_0$ 제약과 일치하게 됩니다. 이는 학습-추론 불일치 문제를 이론적으로 해결하는 핵심 메커니즘입니다.

**성능 우위의 진짜 원인**: 저자들은 KD, 확장 마스크, 결정론적 최적화 각각의 기여를 Ablation으로 분리했지만, 이 세 요소의 **시너지 효과**(Table 4에서 순차적 향상)가 성능 우위의 핵심임을 저자들이 직접 언급합니다("contribute additively").

**MoE 결과의 의미**: 훈련 없는(training-free) 기준선 대비 60% 스파스성에서 +6.6pp 향상은 단순 마스크 최적화만으로도 전문가 활성화 패턴을 효율적으로 포착할 수 있음을 시사합니다. 이는 MoE의 라우팅 비균일성이 구조적 가지치기에 특히 유리한 구조임을 의미합니다.

---

## 5. 통계적 취약점 및 비교 불가능한 수치 ⚠️

| 항목 | 문제점 | 위치 |
|---|---|---|
| **MoE 기준선 비교 불공정** | DDP는 30M 토큰 학습 사용, NAEE/D²-MoE/Camera-P/HEAPr는 학습 없음(training-free). 동일 토큰 예산의 MoE 학습 기준선 미제공 | Table 3, Section 5.1 |
| **Dense 기준선 세팅 동일성** | 기준선(LoRAP 등)에 30M 토큰 파인튜닝 적용을 주장하나, 하이퍼파라미터 일치 여부 불명확 | Section 5.1 |
| **단일 시드(Seed) 실험** | 통계적 유의성 검증 없음. 오차 범위(±) 미보고 | Tables 2-3 전체 |
| **스파스성 목표-달성 편차** | "1% 이내 유지"라고 언급했으나, 모든 실험에서 구체적 달성 스파스성 미보고 | Section 5.1 |
| **vLLM 속도 테스트 조건** | 1,000개 ShareGPT 프롬프트 사용 → 다른 프롬프트 분포에서 속도 재현 가능 여부 미검증 | Section 5.8 |
| **Qwen3-32B의 PPL 역전** | Dense 원본 Wiki2 PPL 7.61 → 20% 가지치기 후 7.25로 **오히려 감소** (이론적으로 비직관적) | Table 13 |
| **HEAPr MoE 결과 부재** | Qwen3-30B-A3B에서 HEAPr 결과가 없음 ("-" 표기) → 불완전한 비교 | Table 3 |
| **캘리브레이션 데이터 의존성** | FineWeb-Edu 30M 토큰 기준 결과. C4 사용 시 성능 저하 확인됨(Table 6)이나, 데이터 선택의 일반화 기준 미제시 | Table 6 |

> ⚠️ **Qwen3-32B PPL 역전 주의**: 20% 가지치기 후 WikiText-2 PPL이 7.61 → 7.25로 감소한 현상은 교란 변수(Confounding Factor)나 학습 데이터-평가 데이터 중첩 가능성을 배제할 수 없어 독립적 검증 필요

---

## 6. 논문이 답하지 않는 질문

| 질문 | 이유/맥락 |
|---|---|
| **50% 이상 스파스성에서 마스크 학습 + 가중치 파인튜닝의 결합 효과?** | "future work will explore continued training"으로만 언급 (p.9) |
| **다른 아키텍처(예: Mamba, RWKV)에 적용 가능한가?** | Transformer 기반 모델에만 적용. 비 Transformer 아키텍처 미탐색 |
| **캘리브레이션 데이터 크기와 품질의 최적 조합은?** | 30M 토큰 단일 실험. 데이터 크기-품질 트레이드오프 곡선 미제공 |
| **마스크 값이 1보다 큰 경우의 의미와 안정성?** | Figure 7에서 관찰되나, 이론적/실험적 분석 없음 |
| **전문가 라우팅과 스파스성 패턴의 상호작용?** | Expert sparsity와 routing score 상관관계 관찰만 있고 인과 분석 없음 |
| **스파스성 달성 편차(1% 허용)의 영향은?** | 허용 범위를 명시했으나 실제 편차의 성능 영향 미분석 |
| **온라인(Online) 배포 환경에서 동적 스파스성 적용?** | 정적(Static) 구조 제거만 실험 |
| **다국어/도메인 특화 모델에서의 일반화?** | 영어 중심 벤치마크(WikiText-2, ARC 등)에만 평가 |
| **마스크 초기화 전략의 민감도?** | $z_k = 1$ 초기화만 사용. 다른 초기화 영향 미탐색 |
| **STE(Straight-Through Estimator) 사용의 최적성?** | STE를 사용하나, 이것이 최선인지에 대한 비교 없음 |

> 📌 **용어 설명**
> - **STE(Straight-Through Estimator)**: ReLU나 Clamp 같은 미분 불가능 함수에서 역전파(Backpropagation) 시 그래디언트를 그대로 통과시키는 근사 기법

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.4): DDP 전체 개요도

**해석:**
- **좌측 (Mask Formulation)**: Dense 모델은 어텐션 헤드($Z_{\text{attn}}$)와 MLP 채널($Z_{\text{mlp}}$) 모두에 마스크 적용. MoE 모델은 각 전문가($Z_{\text{expert}}$)의 채널에만 마스크 적용 (어텐션 유지).
- **우측 (Mask Optimization)**: 잠재 파라미터 $z$에서 두 개의 **분리된 경로**가 핵심:
  - $\boldsymbol{m} = \text{ReLU}(\boldsymbol{z})$: 순전파에서 실제 스케일링에 사용 (범위: $[0, +\infty)$ )
  - $\boldsymbol{s} = \phi(\boldsymbol{z}; \mu)$: 스파스성 손실(Sparsity Loss), 이진화 손실(Binary Loss) 계산에 사용 (범위: $[0,1]$)
- **CE Loss + KL Loss** 조합이 모델 품질 유지의 두 축

**핵심 통찰**: 이 분리(Decoupling) 구조가 DDP를 하드-콘크리트와 근본적으로 차별화하는 설계 결정입니다. 하나의 $z$를 두 개의 용도로 다르게 변환한다는 점이 표현력 확장과 스파스성 제어의 동시 달성을 가능하게 합니다.

---

### Figure 2 (p.4): 결정론적 서로게이트 매핑의 어닐링

**해석:**
- X축: 파라미터 $z$ 값, Y축: 스파스성 대리 점수 $s$
- 세 곡선($\mu = 0.5, 0.3, 0.1$)이 $\mu$ 감소에 따른 변화를 보여줌
- $\mu = 0.5$: 완만한 시그모이드 형태 → 소프트한 결정, 그래디언트 풍부
- $\mu = 0.1$: 급격한 계단 함수에 근접 → 정확한 $\ell_0$ 동작

**핵심 통찰**: 훈련 초기에는 부드러운 곡선으로 그래디언트 신호를 풍부하게 유지하고, 훈련 후기에는 날카로운 결정 경계로 정확한 이진 마스크를 유도합니다. 이 점진적 전환이 학습-추론 불일치 없이 $\ell_0$ 목적함수를 달성하는 핵심 메커니즘입니다.

---

### Figure 3 (p.8): 학습 토큰 수에 따른 성능 수렴

**해석:**
- **Zero-shot Acc (파란 선)**: 약 10M 토큰에서 빠르게 포화 → 다운스트림 태스크 일반화는 조기 달성
- **Perplexity (주황 선)**: 60M 토큰까지 지속적으로 감소 → 토큰 수준 분포 매칭은 시간 필요
- LLaMA-7B와 DeepSeekMoE-16B 모두 동일한 패턴

**핵심 통찰**: 
1. Zero-shot 정확도의 빠른 수렴은 지식 증류(KD)가 고수준 능력을 조기에 보존하기 때문으로 저자들이 설명합니다.
2. 30M 토큰이면 실용적으로 충분하며, 추가 토큰은 PPL만 개선 → **효율적 학습 예산 전략** 수립에 활용 가능
3. 마스크 전용 최적화의 저차원 탐색 공간이 빠른 수렴의 근본 이유

---

### Figure 6 (p.22): MoE 전문가 스파스성 패턴 (DeepSeekMoE-16B)

**해석:**
- **Figure 6(a) 20% 스파스성**: 노란색(높은 보존)이 지배적이지만 일부 전문가(어두운 색)가 이미 공격적으로 가지치기됨
- **Figure 6(b) 40% 스파스성**: 가지치기 패턴이 더 구조화됨 → 특정 전문가들이 대부분 레이어에서 일관되게 비활성화
- **Figure 6(c) 60% 스파스성**: 강한 비균일성 → 일부 전문가는 거의 완전히 제거, 나머지는 높은 비율 유지

**핵심 통찰**:
1. 가지치기 패턴이 **레이어에 걸쳐 일관성**을 보임 → 특정 전문가가 전체적으로 중복적임을 의미
2. MoE의 불균일한 라우팅 빈도가 구조적 가지치기에 자연적으로 유리한 구조임을 확인
3. Global sparsity가 layer-wise/expert-wise보다 우수한 이유: 이 비균일성을 자유롭게 활용할 수 있기 때문 (Table 5와 연결)

---

### Figure 7 (p.22): 학습된 마스크 값의 분포

**해석:**
- **Figure 7(a) 어텐션 헤드 마스크**: 좁은 분포, $[0.4, 1.4]$ 범위. 대부분 ~1.0에 집중 → 헤드는 거의 원래 스케일 유지
- **Figure 7(b) 중간 채널(MLP) 마스크**: 넓은 분포, 최대값 ~7.87. 0 근방과 1 이상 모두에서 유의미한 밀도 → 채널 수준에서 중요 채널을 적극 증폭
- **Figure 7(c) MoE 전문가 마스크**: 최대값 ~3.67, 넓은 분포 → 핵심 전문가 채널을 크게 스케일링

**핵심 통찰**:
1. DDP가 단순 0/1 이진 결정을 넘어 **연속적 재스케일링**을 학습함을 직접 증명
2. 마스크 값 > 1은 "이 채널이 특히 중요하니 더 강조"라는 의미로 해석 가능
3. 어텐션 헤드보다 MLP 채널의 마스크가 더 다양한 분포 → MLP 채널 수준이 더 세밀한 중요도 차이를 지님을 시사
4. 하드-콘크리트의 $[0,1]$ 제약이 이러한 유연한 재스케일링을 원천 차단했음을 역으로 보여줌

---

## 8. 결론, 시사점 및 후속 연구

### 8-1. 저자가 제시한 시사점 및 후속 연구

**저자 제시 시사점** (Section 6, p.9):
- 구조적 가지치기를 $\ell_0$ 제약 마스크 최적화 문제로 재정식화
- 결정론적·마스크 전용 접근으로 수렴 안정성과 가지치기 품질을 최소 연산으로 달성
- Dense/MoE 모두에서 기존 대비 큰 마진으로 우위

**저자 제시 후속 연구** (Section 6, p.9):
- "고스파스성에서 정확도 격차를 더 좁히기 위한 **지속 학습(Continued Training)** 탐색"

### 8-1. 모델 일반화 성능 향상 가능성 (심층 분석)

**현재 일반화 관련 발견:**

1. **데이터셋 분포 의존성** (Table 6): FineWeb-Edu(교육용 웹 텍스트)로 학습된 마스크가 C4나 LaMini보다 우수. 이는 마스크 품질이 캘리브레이션 데이터의 표현 다양성에 크게 의존함을 의미합니다.

2. **지식 증류의 일반화 기여** (Figure 3): KD가 zero-shot 태스크에서 빠른 수렴을 유도함 → 특정 태스크에 과적합되지 않고 범용적 언어 능력을 보존하는 역할.

3. **스케일 확장성** (Table 13): 큰 모델(Qwen3-32B)이 작은 모델(Qwen3-4B)보다 가지치기 후 성능 저하가 작음 → 더 큰 모델일수록 중복성이 높아 일반화 유지에 유리.

**일반화 향상을 위한 가능성과 권장사항:**

| 방향 | 근거 | 기대 효과 |
|---|---|---|
| **다양한 도메인 혼합 캘리브레이션** | Table 6에서 단일 도메인 데이터 한계 확인 | 특정 도메인 편향 감소 |
| **마스크 + 경량 LoRA 결합** | Table 7에서 LoRA 복구의 구조적 한계 확인 → 마스크 후 LoRA 적용 가능 | 50%+ 스파스성에서 회복 |
| **태스크 특화 마스크 앙상블** | 현재 단일 마스크만 학습 → 태스크별 마스크 혼합 가능 | 멀티태스크 일반화 |
| **동적 스파스성 (입력 의존)** | 현재 정적 구조 제거 → Contextual masking으로 확장 | 분포 외 입력에 강건성 |

**일반화 성능 한계에 대한 저자 부분 인정:**
- "instruction-style data may bias masks toward behaviors that do not transfer to pretraining-style evaluation" (p.8, Section 5.3.4)
- 이는 마스크 학습이 데이터 분포에 암묵적으로 특화됨을 저자들이 인정하는 것

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 논문 내 인용 정보와 제 학습 데이터를 기반으로 합니다. 2025년 이후 최신 결과는 불완전할 수 있습니다.

| 연구 | 방법 유형 | 핵심 특징 | DDP와의 차이 |
|---|---|---|---|
| **ShearedLLaMA** (Xia et al., ICLR 2024) | 스파스성 인식 학습 | 계속 학습 중 마스크 학습, ~50B 토큰 | DDP는 30M 토큰으로 동등 이상 성능 |
| **Compresso** (Guo et al., 2023) | 마스크+LoRA | 협력 프롬프트로 구조적 가지치기 | 가중치 업데이트 포함으로 비용 높음 |
| **MaskLLM** (Fang et al., NeurIPS 2024) | 마스크 전용 | N:M 반구조 스파스성, 수정된 하드-콘크리트 | DDP는 구조적 스파스성, 결정론적 방식 |
| **PAT** (Liu et al., AAAI 2025) | 마스크+LoRA | 가지치기 인식 튜닝 | DDP는 가중치 동결로 더 경량 |
| **SlimLLM** (Guo et al., ICML 2025) | 원샷 휴리스틱 | 유사도 기반 헤드/채널 점수 | DDP가 일관되게 우위 (Table 2) |
| **AST/CAST** (Huang et al., 2025) | 스파스성 인식 학습 | LLaMA2/3에서 무손실 2:4 모델 | 반구조 스파스성, 가중치 업데이트 포함 |
| **Tyr-the-Pruner** (Li et al., NeurIPS 2025) | 원샷+전역 분포 최적화 | 전역 스파스성 분포 최적화 | DDP가 모든 모델·스파스성에서 우위 (Table 8) |
| **HEAPr** (Li et al., 2025) | 원샷 (MoE) | 헤시안 기반 전문가 가지치기 | DDP가 30M 토큰으로 HEAPr 120M 토큰보다 우위 |
| **Camera** (Xu et al., 2025) | 원샷 (MoE) | 마이크로 전문가 중복 분석 기반 | DDP가 모든 스파스성에서 우위 |

**비교 분석 종합:**

```
                 연산 비용
                 │
    높음         │  ShearedLLaMA, Compresso, PAT
                 │  (가중치 업데이트 + 대량 토큰)
                 │
    중간         │  [DDP - 본 논문] ← 최적 균형
                 │  (마스크 전용, 30M 토큰)
                 │
    낮음         │  SlimLLM, HEAPr, Camera, NAEE
                 │  (원샷, 학습 없음)
                 └─────────────────────────────
                   낮음        성능 품질       높음
```

---

### 이 논문이 앞으로의 연구에 미치는 영향

1. **마스크 전용 최적화의 표준화**: DDP는 가중치 동결 + 마스크 최적화라는 패러다임이 실용적이고 강력함을 증명하여, 후속 연구들이 이 방향을 적극 채택할 것으로 예상됩니다.

2. **하드-콘크리트 이후의 미분가능 이산 최적화**: 결정론적 어닐링 서로게이트라는 아이디어는 가지치기 외에도 NAS(Neural Architecture Search), 양자화(Quantization) 등 다른 이산 최적화 문제에 적용 가능성이 높습니다.

3. **MoE 모델 효율화**: MoE 전문가 채널 수준의 가지치기가 효과적임을 보여줌으로써, 점점 증가하는 MoE 기반 LLM(GPT-4 추정, Mixtral 등)의 경량화 연구에 중요한 기반이 됩니다.

---

### 앞으로 연구 시 고려할 점 (추가 후속 연구 방향)

| 연구 방향 | 구체적 내용 | 예상 기여 |
|---|---|---|
| **동적 마스크 재학습** | 배포 후 새 도메인 데이터로 마스크만 재최적화 (가중치 고정 유지) | 도메인 적응 비용 최소화 |
| **마스크 전이 학습** | 유사 아키텍처 간 마스크 패턴 전이 가능성 탐색 | 마스크 학습 토큰 추가 절감 |
| **이론적 표현력 경계 분석** | 확장된 마스크 범위 $[0, +\infty)$가 실제로 어느 수준까지 표현력을 높이는지 이론화 | 최적 마스크 범위 설계 기준 |
| **다중 목표 스파스성** | 정확도 + 에너지 소비 + 지연 시간 등 다중 목표 동시 최적화 | 실용적 배포 요구사항 반영 |
| **마스크의 해석 가능성** | 학습된 스파스성 패턴을 모델 내부 표현과 연결 | LLM 내부 작동 원리 이해 |
| **양자화와 가지치기 결합** | DDP 마스크 최적화 + 양자화 인식 학습 동시 적용 | 더 극단적인 모델 압축 |
| **비 Transformer 아키텍처** | Mamba, RWKV 등 상태 공간 모델에 DDP 확장 | LLM 경량화 범용성 확보 |

---

## 참고자료 (논문 내 인용)

- Huang, W., Zhang, P., Zhang, X., Zhou, J., Zhu, J., & Chen, J. (2026). *Deterministic Differentiable Structured Pruning for Large Language Models*. ICML 2026. arXiv:2603.08065v2
- Louizos, C., Welling, M., & Kingma, D. P. (2018). *Learning sparse neural networks through L0 regularization*. ICLR 2018.
- Xia, M., Gao, T., Zeng, Z., & Chen, D. (2024). *Sheared LLaMA: Accelerating language model pre-training via structured pruning*. ICLR 2024.
- Fang, G. et al. (2024). *MaskLLM: Learnable semi-structured sparsity for large language models*. NeurIPS 2024.
- Ma, X., Fang, G., & Wang, X. (2023). *LLM-Pruner: On the structural pruning of large language models*. NeurIPS 2023.
- Guo, J. et al. (2025b). *SlimLLM: Accurate structured pruning for large language models*. ICML 2025.
- Li, G. et al. (2025a). *Tyr-the-Pruner: Structural pruning LLMs via global sparsity distribution optimization*. NeurIPS 2025.
- Kwon, W. et al. (2023). *Efficient memory management for large language model serving with PagedAttention*. SOSP 2023.
- Hu, E. J. et al. (2022). *LoRA: Low-rank adaptation of large language models*. ICLR 2022.
- Guo, D. et al. (2025a). *DeepSeek-R1: Incentivizes reasoning in LLMs through reinforcement learning*. Nature 2025.
- Yang, A. et al. (2025). *Qwen3 technical report*.
- Penedo, G. et al. (2024). *The FineWeb datasets: Decanting the web for the finest text data at scale*. NeurIPS 2024.
- Raffel, C. et al. (2020). *Exploring the limits of transfer learning with a unified text-to-text transformer*. JMLR 2020.
- Huang, W. et al. (2025a). *Pruning large language models with semi-structural adaptive sparse training*. AAAI 2025.
- Huang, W. et al. (2025b). *CAST: Continuous and differentiable semi-structured sparsity-aware training for LLMs*. arXiv:2509.25996.
