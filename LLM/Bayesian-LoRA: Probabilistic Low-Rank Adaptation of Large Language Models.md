# Bayesian-LoRA: Probabilistic Low-Rank Adaptation of Large Language Models

> **⚠️ 주의사항**: 본 논문은 arXiv preprint(arXiv:2601.21003v3, 2026년 7월 25일)로, 아직 동료 심사(peer review)를 거치지 않았습니다. 모든 수치와 주장은 이 점을 고려하여 해석해야 합니다.

---

## 1. Executive Summary (10문장 이내)

Bayesian-LoRA는 대규모 언어 모델(LLM)의 파인튜닝 시 발생하는 **과신(overconfidence) 및 교정 오류(miscalibration)** 문제를 해결하기 위해 제안된 확률적 저차원 적응 프레임워크이다.  
핵심 아이디어는 LoRA의 결정론적 가중치 업데이트 $\Delta W = \frac{\alpha}{r}BA$를 Sparse Gaussian Process(SGP) 기반의 확률적 표현으로 재정식화하는 것이다.  
저자들은 Kronecker 분해된 SGP 사후분포와 LoRA의 인수분해 구조 사이에 **구조적 동형(structural isomorphism)** 이 존재함을 발견하였다.  
이를 바탕으로 저차원 유도 행렬(inducing matrix) $U \in \mathbb{R}^{r \times c}$에 대한 변분 사후분포를 정의하고, 정규화 흐름(normalizing flow)으로 표현력을 강화하였다.  
훈련 목표는 흐름 증강 ELBO(Flow-Augmented ELBO)이며, KL 항은 폐쇄형(closed-form)으로 계산 가능하여 Hessian 연산이 불필요하다.  
모델은 약 0.42M의 추가 파라미터와 표준 LoRA 대비 약 1.2배의 훈련 비용만을 요구한다.  
6개 상식 추론 벤치마크, WikiText-2 언어 모델링, MATH 수학 추론 태스크에서 평가되었다.  
ECE 최대 84% 감소, NLL 최대 76% 감소를 달성하면서 경쟁력 있는 정확도를 유지한다.  
분포 외(OoD) 평가에서도 대규모 분포 이동 시 post-hoc 방법보다 우수한 교정 성능을 보인다.  
전체적으로 Bayesian-LoRA는 안전-중요(safety-critical) 응용에서 LLM의 신뢰 가능한 불확실성 정량화를 위한 실용적인 프레임워크를 제공한다.

---

### 1-1. 연구의 목적과 필요성

**문제 배경**: LLM은 기본적으로 정확도 최적화를 위해 훈련되므로, 불확실한 상황에서도 높은 확신으로 예측하는 경향이 있다. 특히 소규모 데이터셋으로 파인튜닝할 경우 과적합(overfitting)과 교정 오류가 심화된다. (p.1)

> 🔑 **용어 설명**
> - **교정(Calibration)**: 모델이 예측한 확률이 실제 정답률과 얼마나 일치하는지를 나타내는 개념. 예를 들어 "90%의 확신"으로 예측한 경우 실제로도 90%의 정답률을 보여야 잘 교정된 모델임.
> - **과신(Overconfidence)**: 모델이 실제 정확도보다 훨씬 높은 확신으로 예측하는 현상.

**필요성**: 의료 진단(Savage et al., 2025), 자율주행(Tu et al., 2025) 등 안전-중요 도메인에서는 교정된 불확실성 추정이 필수적이다. 기존 방법들의 한계는 다음과 같다:
- **Full Bayesian inference**: LLM 규모에서 계산 불가능
- **Laplace 방법**: 훈련 후 교정 보정을 적용하므로 분포 이동(distribution shift)에 취약
- **BLoB**: LoRA 파라미터 전체에 Mean-field 가정 적용, 표현력 제한

> 🔑 **용어 설명**
> - **Mean-field 가정**: 모든 파라미터가 서로 독립적이라고 가정하여 전체 사후분포를 각 파라미터의 사후분포의 곱으로 근사하는 방법. 계산이 간단하지만 파라미터 간 상관관계를 무시함.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 출처(페이지/Table) |
|---|---|---|
| LoRA와 Kronecker-SGP 사후분포 간 구조적 동형 존재 | 조건부 평균 $M_W(U)=T_r U T_c$가 LoRA의 $\frac{\alpha}{r}BA$와 동일한 이중선형(bilinear) 형태를 공유 | p.1, p.4, §4.1 |
| 결정론적 LoRA는 Bayesian-LoRA의 극한 케이스 | Corollary 4.1: $q(U)=\delta(U-U^\*)$, $\lambda \to 0$일 때 $\Delta W = T_r U^* T_c$로 환원. Table 7로 실험적 검증 | p.5, Table 7 |
| 흐름 증강 ELBO로 교정-인식 훈련 가능 | 정규화 흐름이 사후분포의 표현력 향상, 조건부 KL은 폐쇄형 → Hessian 불필요 | p.4, §4.2 |
| 표준 LoRA 대비 최소한의 추가 비용 | +0.42M 파라미터, ×1.229 훈련 시간, ×1.003 메모리 | Table 4 |
| ECE 최대 84% 감소 (WinoGrande-S) | MAP 대비 ECE: 30.80 → 4.90 | Table 1, p.5 |
| NLL 최대 76% 감소 | 여러 벤치마크에서 최우수 NLL 달성 | Table 1, Table 2 |
| 대규모 분포 이동 시 post-hoc 방법보다 우수 | CS(11.1), Law(16.5), Health(12.9)에서 최우수 ECE. LA는 소규모 이동에서 우위 | Table 6, Appendix B |
| 30B MoE 모델까지 확장 가능 | Qwen3-30B-A3B에서 정확도 손실 없이 NLL/ECE 개선 | Table 3 |

---

## 2-1. 상세 분석

### 🔴 해결하고자 하는 문제

파인튜닝된 LLM의 **교정 오류**: LoRA로 파인튜닝된 모델은 소규모 데이터에 과적합되어 체계적으로 과신하는 경향이 있으며, 기존 방법들은 효율성 또는 분포 이동 견고성 측면에서 한계를 가진다.

---

### 🔵 제안하는 방법 및 수식

#### Step 1: LoRA 기본 수식 (§3.1, p.2)

$$\Delta W = \frac{\alpha}{r} BA, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad A \in \mathbb{R}^{r \times d_{\text{in}}} $$

$$y = (W_{\text{pre}} + \Delta W)x = W_{\text{pre}} x + \frac{\alpha}{r} B(Ax) $$

| 기호 | 설명 |
|---|---|
| $\Delta W$ | 가중치 업데이트 행렬 |
| $\alpha$ | 고정 스케일링 인수 |
| $r$ | 저차원 랭크, $r \ll \min(d_{\text{in}}, d_{\text{out}})$ |
| $B \in \mathbb{R}^{d_{\text{out}} \times r}$ | 좌측 저랭크 행렬 |
| $A \in \mathbb{R}^{r \times d_{\text{in}}}$ | 우측 저랭크 행렬 |
| $x \in \mathbb{R}^{d_{\text{in}}}$ | 레이어 입력 |
| $W_{\text{pre}}$ | 사전학습된 동결 가중치 |

> 🔑 **용어 설명**
> - **저랭크(Low-Rank) 분해**: 큰 행렬을 두 개의 작은 행렬의 곱으로 표현하는 기법. $d_{\text{in}} \times d_{\text{out}}$ 파라미터를 $r(d_{\text{in}} + d_{\text{out}})$으로 줄임.

#### Step 2: 유도 변수 사전분포 및 변분 사후분포 (§3.2, p.3)

$$p(U) = \mathcal{N}(\text{vec}(U) \mid \mathbf{0}, K_U), \quad K_U = K_c \otimes K_r $$

$$q(U) = \mathcal{N}(\text{vec}(U) \mid \mathbf{m}, \mathbf{S}) $$

| 기호 | 설명 |
|---|---|
| $U \in \mathbb{R}^{r \times c}$ | 저차원 유도 행렬 |
| $K_r \in \mathbb{R}^{r \times r}$, $K_c \in \mathbb{R}^{c \times c}$ | 학습 가능한 행/열 공분산 인수 |
| $\otimes$ | Kronecker 곱 |
| $\mathbf{m}$, $\mathbf{S}$ | 변분 평균과 공분산 |

> 🔑 **용어 설명**
> - **Kronecker 곱($\otimes$)**: 두 행렬을 결합하는 연산. $A \otimes B$는 $A$의 각 원소에 $B$를 스칼라 배한 블록 행렬. 공분산 행렬의 구조화된 표현에 유용.
> - **유도 행렬(Inducing Matrix)**: Sparse GP에서 전체 가중치 공간의 분포를 제어하는 저차원 "대표점". 계산 효율성의 핵심.
> - **변분 사후분포(Variational Posterior)**: 실제로 계산 불가능한 정확한 사후분포를 근사하기 위해 사용하는 분포족.

#### Step 3: $W$의 $U$에 대한 조건부 분포 및 투영 연산자 (§3.2, p.3)

$$K_r = Z_r Z_r^\top + D_r^2, \quad K_c = Z_c Z_c^\top + D_c^2 $$

$$T_r = Z_r^\top K_r^{-1}, \quad T_c = K_c^{-1} Z_c $$

$$M_W(U) = T_r U T_c $$

$$p(W \mid U) = \mathcal{N}(W \mid M_W(U), \Sigma_W) $$

| 기호 | 설명 |
|---|---|
| $Z_r \in \mathbb{R}^{r \times d_{\text{out}}}$, $Z_c \in \mathbb{R}^{c \times d_{\text{in}}}$ | 학습 가능한 유도 위치 행렬 |
| $D_r \in \mathbb{R}^{r \times r}$, $D_c \in \mathbb{R}^{c \times c}$ | 대각 잡음 행렬 |
| $T_r \in \mathbb{R}^{d_{\text{out}} \times r}$ | 행 공간 투영 연산자 (LoRA의 $B$에 대응) |
| $T_c \in \mathbb{R}^{c \times d_{\text{in}}}$ | 열 공간 투영 연산자 (LoRA의 $A$에 대응) |
| $\Sigma_W \in \mathbb{R}^{d_{\text{out}}d_{\text{in}} \times d_{\text{out}}d_{\text{in}}}$ | 사전 조건부 공분산 |

> 🔑 **용어 설명**
> - **투영 연산자(Projection Operator)**: 유도 공간의 정보를 전체 가중치 공간으로 확장(diffuse)하는 행렬. $T_r$은 행 방향, $T_c$는 열 방향으로 작동.

#### Step 4: 정규화 흐름으로 사후분포 강화 (§4.1, p.4)

$$q_0(U_0) = \mathcal{N}(\text{vec}(U_0) \mid \mathbf{m}, \text{diag}(\boldsymbol{\sigma}^2)), \quad U = T_\phi(U_0) $$

| 기호 | 설명 |
|---|---|
| $T_\phi$ | 역변환 가능한 미분 가능 변환 (정규화 흐름) |
| $q_0$ | 대각 가우시안 기저 분포 |
| $\boldsymbol{\sigma} \in \mathbb{R}^{rc}_{>0}$ | 기저 분포의 표준편차 |

> 🔑 **용어 설명**
> - **정규화 흐름(Normalizing Flow)**: 단순한 분포(예: 가우시안)를 역변환 가능한 변환으로 복잡한 분포로 변형하는 기법. 표현력을 높이면서 확률 밀도 계산이 가능함.
> - **MAF (Masked Autoregressive Flow)**: 정규화 흐름의 한 종류로, 자기회귀적 구조를 이용해 Jacobian 행렬식을 효율적으로 계산.

#### Step 5: 흐름 증강 ELBO (§4.2, p.4)

$$\log q_\phi(U) = \log q_0(T_\phi^{-1}(U)) - \log \left| \det J_{T_\phi}(T_\phi^{-1}(U)) \right| $$

$$\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{U_0 \sim q_0, \varepsilon}[\log p(\mathcal{D} \mid W)]}_{\text{(1) 기대 로그 우도}} - \underbrace{\mathbb{E}_{U_0 \sim q_0}\left[\log q_0(U_0) - \log \left| \det J_{T_\phi}(U_0) \right| - \log p(T_\phi(U_0))\right]}_{\text{(2) 유도 변수 KL}} - \underbrace{\frac{D}{2}(\lambda^2 - 1 - 2\log\lambda)}_{\text{(3) 조건부 KL}} $$

| 기호 | 설명 |
|---|---|
| $\mathcal{D}$ | 훈련 데이터 |
| $J_{T_\phi}$ | 흐름 $T_\phi$의 Jacobian 행렬 |
| $D = \sum_{\ell \in \mathcal{L}} d_{\text{out},\ell} \cdot d_{\text{in},\ell}$ | 교체된 레이어 전체 가중치 수 |
| $\lambda > 0$ | 학습 가능한 공분산 스케일 |

> 🔑 **용어 설명**
> - **ELBO (Evidence Lower BOund)**: 변분 추론에서 로그 주변 우도 $\log p(\mathcal{D})$의 하한(lower bound). 이를 최대화함으로써 사후분포를 근사.
> - **KL 발산(KL Divergence)**: 두 확률분포의 차이를 측정하는 비대칭적 척도. $\text{KL}(q \| p) = 0$이면 $q = p$.
> - **Jacobian 행렬식(Jacobian Determinant)**: 변수 변환 시 확률 밀도의 변화율을 나타냄. 정규화 흐름에서 필수적으로 계산됨.

#### Step 6: 조건부 KL 폐쇄형 (Appendix A.4, p.13-14)

$$\text{KL}(q(W \mid U) \| p(W \mid U)) = \frac{d_W}{2}(\lambda^2 - 1 - 2\log\lambda) $$

이 항은 $U$에 독립적이므로 샘플링이나 Hessian 계산 없이 정확하게 계산됨.

---

### 🟢 모델 구조 (Figure 1, p.4)

```
[표준 LoRA]                    [Bayesian-LoRA]
사전학습 가중치 W_pre           사전학습 가중치 W_pre (동결)
    ↓                               ↓
B, A 직접 최적화            유도 행렬 U ~ q₀ (기저 분포)
ΔW = (α/r)BA                    ↓
                           U' = T_φ(U) (정규화 흐름)
                                ↓
                      Ā = T^A_r U' T^A_c (조건부 평균)
                      B̄ = T^B_r U' T^B_c
                                ↓
                      A = Ā + λΣ^(1/2)_A ε (분산 추가)
                      B = B̄ + λΣ^(1/2)_B ε
                                ↓
                      W_merged = W_pre + (α/r)BA
                      (N번의 MC 샘플 평균)
```

> 🔑 **용어 설명**
> - **Monte Carlo 샘플링(MC Sampling)**: 확률분포에서 여러 번 무작위 샘플을 뽑아 기댓값을 근사하는 방법. N=2~4개의 샘플로 실용적인 불확실성 추정 가능.
> - **인식론적 불확실성(Epistemic Uncertainty)**: 모델 자체의 지식 부족에서 비롯된 불확실성. 데이터를 더 수집하면 줄일 수 있음 (↔ 우연적 불확실성: 데이터 자체의 노이즈).

---

### 🟡 성능 향상

| 지표 | 최고 성과 | 비교 대상 | 출처 |
|---|---|---|---|
| ECE 최대 감소 | 84% (WinoGrande-S: 30.80 → 4.90) | MAP(표준 LoRA) | Table 1 |
| NLL 최대 감소 | 76% (BoolQ: 0.43 → 0.29) | MAP | Table 1 |
| OoD 정확도 | 6개 중 5개 데이터셋 최우수 | LA(post-hoc) | Table 6 |
| MATH(14B) NLL | 0.513 (최우수) | LA: 0.81, BLoB: 1.21 | Table 3 |
| 추가 파라미터 | +0.42M (4.9M vs 4.48M) | 가장 적은 Bayesian 오버헤드 | Table 4 |

---

### 🔴 한계점 (p.9, Limitations)

1. **레이어 독립성**: 각 레이어의 유도 행렬이 독립적으로 모델링되어 레이어 간 상관관계를 포착하지 못함. 계층적 사전분포(hierarchical priors)가 필요.
2. **하이퍼파라미터 민감성**: Bayesian 최적화 분석(Table 14)에서 정확도-교정 트레이드오프가 하이퍼파라미터에 의존함.
3. **모달리티 제한**: 다른 모달리티(이미지, 오디오) 및 instruction-tuning/RLHF로의 확장 미검증.
4. **이론적 한계**: Sparse 유도 근사에 대한 더 엄밀한 이론적 경계 미제시.

> 🔑 **용어 설명**
> - **RLHF (Reinforcement Learning from Human Feedback)**: 인간의 선호도 피드백을 이용해 LLM을 정렬하는 훈련 방법. ChatGPT 등에 사용됨.

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 근거 위치 |
|---|---|
| LoRA-SGP 구조적 동형 | p.1(Abstract), p.4(§4.1), Figure 1(p.4) |
| Corollary 4.1 (결정론적 극한) | p.5(§4, Corollary 4.1), Table 7(p.9) |
| Proposition 3.1 (KL 불변성) | p.3(§3.2), Appendix A.1(p.13-14) |
| 교정 성능 (6개 벤치마크) | Table 1(p.7) |
| WikiText-2 언어 모델링 | Table 2(p.7) |
| 대규모 모델(14B, 30B) 성능 | Table 3(p.8), §5.2(p.5) |
| 효율성 비교 | Table 4(p.8) |
| 흐름 깊이 어블레이션 | Table 5(p.8), Figure 2(p.9) |
| OoD 견고성 | Table 6(p.9), Appendix B(p.15) |
| MC 샘플 수 영향 | Appendix C, Tables 8-9(p.15) |
| C-LoRA, TFB 비교 | Tables 19-20(p.22), Appendix J(p.21) |
| 시각적 Pareto 비교 | Figures 5-6(p.22-23), Appendix K |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 📊 저자가 직접 보고한 결과

**연구 주제**: LLM 파인튜닝의 교정 오류를 SGP 기반 확률론적 LoRA로 해결.

**핵심 수식** (저자 직접 제시):
- LoRA: $\Delta W = \frac{\alpha}{r}BA$ (Eq. 1)
- SGP 조건부 평균: $M_W(U) = T_r U T_c$ (Eq. 8)
- Flow-ELBO: Eq. (16)
- 조건부 KL: $\frac{d_W}{2}(\lambda^2 - 1 - 2\log\lambda)$ (Eq. 31)

**보고된 수치 결과**:
- WinoGrande-S ECE: MAP 30.80 → Bayesian-LoRA 4.90 (84% 감소)
- BoolQ NLL: MAP 0.43 → Bayesian-LoRA 0.29 (76% 감소)
- MATH(Qwen2.5-14B) CoT-NLL: 0.513, CoT-ECE: 5.81, ACC: 51.1%
- 훈련 시간: ×1.229 MAP, 메모리: ×1.003 MAP (Table 4)
- Qwen3-14B ARC-C: ECE 4.30 → 3.30, OBQA: ECE 5.04 → 3.40 (Table 17)

---

### 🔍 리뷰어(필자)의 해석

1. **성과의 비대칭성**: ECE 개선이 벤치마크마다 크게 다르다. WinoGrande-S에서 84% 개선되었지만, ARC-C ECE(9.20)는 C-LoRA(8.83), LA(7.50)보다 높다(Table 1). 저자들은 이를 충분히 부각하지 않는다.

2. **NLL의 혼재적 결과**: BoolQ에서 BLoB(N=10, 0.31)보다 Bayesian-LoRA(N=4, 0.29)가 좋으나, WinoGrande-S NLL(0.79)은 BLoB(0.63)보다 높다. 일관되지 않은 NLL 패턴은 방법의 장점이 특정 태스크에 의존할 수 있음을 시사한다.

3. **정확도 vs 교정 트레이드오프**: Table 14에서 Bayesian Optimization 적용 시 WinoGrande-M ECE가 3.00 → 7.92로 악화되는 사례가 있어, 하이퍼파라미터 최적화 목표에 따라 교정이 역효과를 낼 수 있다.

4. **C-LoRA와의 비교**: ECE에서는 Bayesian-LoRA가 6개 중 2개 벤치마크에서만 우수하고, NLL에서도 4/6에서만 우수하다. 저자들은 "정확도에서 모두 우수"를 강조하지만 교정 측면의 혼재된 결과는 조심스럽게 해석해야 한다(Table 19).

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적으로 취약한 부분

| 항목 | 문제점 |
|---|---|
| **3개 시드 평균** (Table 1) | 일부 수치의 표준편차가 크다. BBB의 WinoGrande-S ECE: 21.81 ± 12.95는 매우 높은 분산으로 통계적 유의성이 낮음 |
| **Qwen3-14B AIME 실험** (Table 18) | 단 30개 문제로 평가. "80.0% ACC"는 24개 문제 정답으로, 통계적 신뢰 구간이 매우 넓음 |
| **TFB 비교** (Table 20) | Bayesian-LoRA는 단일 실험값(표준편차 없음), TFB는 ±0.19 등 표준편차 보고. 직접 비교 불공정 |
| **OoD 평가** (Table 6) | LLLA, LA의 표준편차가 ±0.0으로 보고되어 단일 시드 사용 의심 |
| **BLoB(N=10) vs Bayesian-LoRA(N=4)** | 추론 샘플 수가 달라 직접 비교 시 불공정. Table 4에서는 N=4 기준 통일하지만 Table 1은 N=10 사용 |

> ⚠️ **주의**: 저자들은 BLoB를 "N=10"으로 교정/NLL 표에 제시하고 효율성 비교는 "N=4"로 제시한다고 각주에서 인정하고 있으나, 이는 비교 기준 불일치임.

### ⚠️ 비교 불가능한 수치

| 비교 쌍 | 이유 |
|---|---|
| Qwen2.5-14B-Instruct vs Qwen3-14B 결과 | 서로 다른 태스크(MATH vs ARC-C/OBQA)에서 평가 |
| C-LoRA(M=10) vs Bayesian-LoRA(N=4) | 샘플 수가 다름 (10 vs 4) |
| LLLA/LA vs Bayesian-LoRA | LA는 사후 보정이므로 훈련 방식 자체가 상이 |
| OoD 평가의 LLLA (std=0.0) | 단일 시드 의심, 다중 시드 결과와 직접 비교 불공정 |

---

## 6. 논문이 답하지 않는 질문

1. **레이어 선택의 영향**: Q, K, LM head만 Bayesian으로 교체하는 것의 이론적 근거는? 다른 레이어 조합이 더 나은 교정 성능을 줄 수 있는가? (Appendix L에서 경험적 일부만 다룸)

2. **유도 행렬 초기화**: $U$의 초기화 방법이 수렴 속도 및 최종 교정 성능에 미치는 영향은?

3. **흐름의 이론적 수렴 보장**: MAF 기반 정규화 흐름이 진정한 사후분포에 수렴하는 조건은? 이론적 bound가 제시되지 않음.

4. **레이어 간 상관관계**: 저자들은 레이어 독립성을 한계로 인정하지만, 계층적 사전분포 설계의 구체적 방향을 제시하지 않음.

5. **RLHF/Instruction Tuning**: 선호도 학습(예: DPO) 환경에서 교정 이점이 유지되는가?

6. **다른 아키텍처**: Mamba, RWKV 등 비-Transformer 아키텍처에서의 적용 가능성은?

7. **KL 스케일링의 영향**: KL scaling이 0.2/steps_per_epoch으로 설정된 근거는? 이 값이 교정-정확도 트레이드오프에 미치는 민감도는?

8. **양자화(Quantization)와의 호환성**: QLoRA처럼 4비트 양자화와 결합 시 확률론적 프레임워크가 유지되는가?

9. **실제 배포 시나리오**: 추론 시 N=1(결정론적 모드)과 N=4(불확실성 모드)의 전환 기준은 어떻게 설정해야 하는가?

---

## 7. 가장 중요한 그림 5개 해석

### 📊 Figure 1 (p.4): Bayesian-LoRA 구조 개요

**설명**: (a) 표준 LoRA는 $B$, $A$ 행렬을 직접 최적화하여 결정론적 $\Delta W$를 생성한다. (b) Bayesian-LoRA는 유도 공간에서 $U$를 샘플링하고, 정규화 흐름 $T_\phi$를 통해 변환한 후 투영 연산자 $T_r$, $T_c$로 $B$, $A$의 확률론적 실현값을 생성한다.

**해석**: 이 그림은 Bayesian-LoRA의 핵심 설계 원칙을 시각화한다. 표준 LoRA가 하나의 점 추정치를 생성하는 반면, Bayesian-LoRA는 N번의 MC 샘플을 통해 가중치 분포를 표현한다. $M_W(U) = T_r U T_c$와 LoRA의 $\frac{\alpha}{r}BA$ 사이의 기능적 유사성이 명확히 드러난다. 이 구조가 LoRA의 함수 공간(동일한 이중선형 저랭크 구조)을 유지하면서 불확실성을 추가하는 방식임을 이해하는 데 핵심적이다.

---

### 📊 Figure 2 (p.9): 유도 점 차원 $r=c$ 어블레이션

**설명**: X축은 유도 점 차원($r=c$), Y축은 ACC/NLL(왼쪽)과 ECE(오른쪽). 기본 설정 $r=c=9$가 점선으로 표시.

**해석**: ECE는 $r=4$에서 $r=16$까지 단조 감소하나 그 이후 수렴(diminishing returns). ACC는 $r=32$ 부근에서 소폭 하락하는 경향. 이는 더 높은 유도 차원이 더 풍부한 사후분포 근사를 제공하지만, 파라미터 수 증가 대비 한계 수익이 체감됨을 나타낸다. $r=c=9$는 정확도와 교정의 합리적 절충점이다. 그러나 **이 어블레이션이 OBQA 단일 벤치마크에서만 수행**되었으므로, 이 결과의 일반화 가능성에는 주의가 필요하다.

---

### 📊 Figure 5 (p.22): Llama-2-7B 6개 벤치마크 평균 ACC vs ECE

**설명**: ECE 축이 반전되어 있어 우상단이 이상적(고정확도, 저ECE). Bayesian-LoRA(빨간 별)가 우상단에 위치.

**해석**: 이 그림은 논문의 핵심 주장을 시각적으로 가장 강력하게 지지한다. Bayesian-LoRA가 ACC와 ECE 양 측면에서 모든 기준선을 파레토 지배(Pareto-dominate)한다는 점이 명확히 드러난다. BBB는 낮은 ECE를 얻지만 ACC가 크게 희생되고, 온도 스케일링(Temp Scaling)은 ECE는 낮지만 ACC가 개선되지 않음. BLoB(N=10)은 ACC와 ECE에서 Bayesian-LoRA에 근접하지만 2.5배 더 많은 추론 샘플이 필요하다. 단, **6개 벤치마크의 평균이므로 개별 벤치마크(예: ARC-C)에서의 ECE 열세가 희석**될 수 있음에 주의.

> 🔑 **용어 설명**
> - **파레토 지배(Pareto Dominance)**: 두 목표(정확도, 교정) 모두에서 다른 방법을 동시에 능가하는 상태.

---

### 📊 Figure 6 (p.23): WinoGrande-S 벤치마크 ACC vs ECE

**설명**: 가장 소규모이며 과신 경향이 강한 단일 벤치마크에서의 비교.

**해석**: Figure 5의 평균 결과와 달리, 이 소규모 이진 태스크에서 각 방법의 특성이 극명하게 드러난다. BBB가 거의 무작위 추측 수준(~55% ACC)으로 붕괴되는 것은 Mean-field 가우시안이 복잡한 사후분포를 표현하는 데 실패함을 보여준다. Post-hoc 방법들(LLLA, LA)은 ECE를 낮추지만 ACC가 Bayesian-LoRA보다 낮다. Bayesian-LoRA가 가장 높은 ACC(~70.9%)와 낮은 ECE를 동시에 달성하는 것은, **구조화된 GP 기반 사후분포가 소규모 데이터에서 과적합을 방지하는 정규화 효과**를 가짐을 시사한다.

---

### 📊 Figure 3 (p.18, Appendix H): WinoGrande-M의 NLL vs ECE 파레토 분석

**설명**: 각 점이 (학습률, 가중치 감쇠) 쌍을 나타내는 파레토 전선 분석. 회색 영역은 지배된(dominated) 해. 교차 표시가 최선 선택.

**해석**: 이 그림은 Bayesian-LoRA의 하이퍼파라미터 민감성을 드러내는 중요한 분석이다. ECE와 NLL 간의 트레이드오프가 존재하며, 단일 최적 설정이 없음을 보여준다. **최선 선택(Best choice)** 주변에 여러 파레토 최적 후보가 밀집해 있어, 실용적으로는 학습률 범위 $[3.4 \times 10^{-4}, 6.9 \times 10^{-4}]$ 내에서 비교적 안정적인 성능을 기대할 수 있다. Table 14에서 BO 적용 시 WinoGrande-M ECE가 3.00 → 7.92로 악화되는 것은, **ECE 단독 최적화 없이 다목적 최적화가 필요**함을 명확히 보여준다.

---

## 8. 결론 및 후속 연구

### 8-1. 저자가 제시한 시사점 및 후속 연구 계획

**시사점** (p.8, Conclusion):
- LoRA와 Kronecker-SGP 사후분포의 구조적 동형은 확률론적 파인튜닝을 위한 새로운 이론적 기반을 제공
- End-to-end 교정 훈련은 대규모 분포 이동 시 post-hoc 방법보다 강건함
- Post-hoc 방법과 Bayesian-LoRA는 상보적 관계 (소규모 이동 → post-hoc 우세, 대규모 이동 → Bayesian-LoRA 우세)

**저자가 언급한 미래 연구** (p.9, Limitations):
1. 레이어 간 상관관계를 포착하는 **계층적 사전분포** 설계
2. **다른 모달리티** (비전, 오디오) 및 **instruction-tuning/RLHF** 확장
3. Sparse 유도 근사에 대한 **더 엄밀한 이론적 경계** 도출

---

### 8-1. 모델의 일반화 성능 향상 가능성

논문에서 관련 근거:

1. **OoD 정확도** (Table 6): 6개 중 5개 분포 외 데이터셋에서 최우수 정확도. 대규모 이동(CS: +0.7, Health: +1.5pp vs LA) 시 일반화 우위.

2. **End-to-end Bayesian 훈련의 정규화 효과** (Appendix B, p.15):
> "The end-to-end Bayesian training acts as a form of distributional regularization that prevents overfitting to the in-distribution data."

3. **ELBO의 KL 항 역할**: KL 발산 $\text{KL}(q_\phi(U) \| p(U))$이 유도 행렬 $U$를 GP 사전분포로 당겨 암묵적 정규화 효과 제공.

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}[\log p(\mathcal{D} \mid W)] - \underbrace{\text{KL}(q_\phi(U) \| p(U))}_{\text{정규화 항}} - \text{KL}_{\text{조건부}}$$

4. **일반화 향상을 위한 추가 연구 방향**:
   - **계층적 사전분포**: 레이어 간 상관관계를 포착하는 사전분포는 공유 구조를 학습하여 새로운 도메인에서 일반화 개선 가능
   - **태스크-agnostic 유도 위치**: 여러 태스크에 공유된 유도 위치를 학습하는 메타 학습(meta-learning) 접근
   - **사전분포 선택**: 현재의 등방성 가우시안 사전분포 대신 태스크 관련 정보를 포함한 정보적 사전분포(informative prior) 활용
   - **불확실성 기반 선택적 예측**: 교정된 불확실성을 활용한 selective prediction으로 실제 일반화 오류를 능동적으로 감지

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

#### 주요 관련 연구 계보

| 연구 | 발표 | 핵심 아이디어 | Bayesian-LoRA와의 관계 |
|---|---|---|---|
| **LoRA** (Hu et al., 2022) | ICLR 2022 | 결정론적 저랭크 적응 | Bayesian-LoRA의 기반. 결정론적 극한으로 환원됨 |
| **Laplace Redux** (Daxberger et al., 2021) | NeurIPS 2021 | Laplace 근사의 효율적 구현 | Bayesian-LoRA가 초월하는 post-hoc 기준선 |
| **LLLA/LA** (Yang et al., 2024) | ICLR 2024 | LoRA 파라미터에 Laplace 적용 | Bayesian-LoRA의 직접 경쟁자. ID/소규모 이동에서 ECE 우세 |
| **BLoB** (Wang et al., 2024b) | NeurIPS 2024 | 역전파로 Bayesian LoRA 가중치 학습 | Mean-field 가정 vs. Bayesian-LoRA의 구조화된 GP 사후분포 |
| **C-LoRA** (Rahmati et al., 2025) | NeurIPS 2025 | 문맥적 저랭크 적응으로 불확실성 추정 | Bayesian-LoRA가 정확도에서 모두 우세, ECE는 혼재 |
| **TFB** (Shi et al., 2025) | NeurIPS 2025 | 훈련 없는 Bayesian 변환 (post-hoc) | Bayesian-LoRA가 ECE에서 크게 우세 (50% 감소) |
| **FFG-U** (Ritter et al., 2021) | NeurIPS 2021 | Kronecker 분해 유도 가중치 | Bayesian-LoRA의 이론적 영감. LoRA와의 연결 제시 |
| **Calibration Survey** (Liu et al., 2025) | KDD 2025 | LLM 불확실성 정량화 및 교정 서베이 | Bayesian-LoRA가 해결하는 문제 공간 정의 |

#### Bayesian-LoRA가 앞으로의 연구에 미치는 영향

1. **이론적 기여**: LoRA의 확률론적 해석은 PEFT 방법 전반에 대한 Bayesian 관점을 열어줌. 다른 PEFT 방법(LoRA+, AdaLoRA, DoRA 등)에도 유사한 확률론적 재해석 가능성 제시.

2. **실용적 프레임워크**: $\approx$ 1.2배 훈련 비용으로 교정 개선이 가능하다는 것은 산업 응용에서 즉시 활용 가능한 수준의 효율성을 보여줌.

3. **안전 AI 연구 방향**: 의료, 자율주행 등에서 교정된 불확실성이 결정의 신뢰성을 높인다는 실증 결과는 Trustworthy AI 연구의 실용적 기준점을 제공.

#### 앞으로 연구 시 고려할 점

**방법론적 고려사항**:

1. **비교 공정성**: BLoB(N=10)과 Bayesian-LoRA(N=4)의 추론 샘플 수 불일치 문제를 인지하고, 동일 샘플 수에서의 비교를 반드시 포함해야 함.

2. **교정 지표의 다양화**: 15-bin ECE는 조잡한 요약 지표임. Reliability Diagram, Adaptive ECE, Expected Variance Calibration(EVC) 등 추가 지표 필요.

3. **벤치마크 다양성**: 현재 상식 추론, 언어 모델링, 수학 추론에 집중되어 있음. 장문 생성, 코드 생성, 멀티모달 태스크에서의 교정 연구 필요.

4. **기저 모델 의존성**: Llama-2-7B, Qwen 시리즈에서의 성능이 다른 아키텍처(예: Gemma, Mistral, Phi)에서도 재현되는지 검증 필요.

5. **이론적 보장**: 현재 ELBO 최적화가 실제 사후분포와 얼마나 가까운지에 대한 이론적 bound가 없음. PAC-Bayes 경계 등을 통한 이론적 보장 연구 필요.

**후속 연구 제안**:

| 연구 방향 | 이유 | 난이도 |
|---|---|---|
| 계층적 사전분포 설계 | 레이어 간 상관관계 포착 → 일반화 개선 | 중 |
| QLoRA + Bayesian-LoRA 결합 | 4비트 양자화 환경에서의 교정 | 중 |
| 불확실성 기반 능동 학습 | 교정된 확신을 라벨 효율성에 활용 | 중 |
| 다중 태스크 유도 공간 공유 | 태스크 간 지식 전달로 일반화 향상 | 높음 |
| RLHF/DPO + Bayesian 교정 | 선호도 학습에서의 불확실성 정량화 | 높음 |
| 멀티모달 Bayesian-LoRA | 비전-언어 모델에서의 교정 | 높음 |
| 적응적 유도 점 할당 | 레이어/태스크 중요도에 따른 유도 차원 동적 배분 | 중 |

---

## 참고 자료

### 논문 내 인용 출처
- **원본 논문**: Lin, M., Guan, S., Patane, A., Gregg, D., & Botterweck, G. (2026). "Bayesian-LoRA: Probabilistic Low-Rank Adaptation of Large Language Models." arXiv:2601.21003v3
- **코드**: https://github.com/moulelin/Bayesian-LoRA

### 주요 참고 문헌 (논문 내 인용)
- Hu et al. (2022). "LoRA: Low-rank adaptation of large language models." ICLR 2022.
- Ritter et al. (2021). "Sparse uncertainty representation in deep learning with inducing weights." NeurIPS 2021.
- Yang et al. (2024). "Bayesian low-rank adaptation for large language models." ICLR 2024.
- Wang et al. (2024b). "BLoB: Bayesian low-rank adaptation by backpropagation for large language models." NeurIPS 2024.
- Titsias (2009). "Variational learning of inducing variables in sparse Gaussian processes." AISTATS 2009.
- Guo et al. (2017). "On calibration of modern neural networks." ICML 2017.
- Lin et al. (2026). "Flow-induced diagonal Gaussian Processes." AAAI 2026.
- Lin et al. (2025). "Stochastic weight sharing for Bayesian neural networks." AISTATS 2025.
- Rahmati et al. (2025). "C-LoRA: Contextual low-rank adaptation for uncertainty estimation in large language models." NeurIPS 2025.
- Shi et al. (2025). "Training-free Bayesianization for low-rank adapters of large language models." NeurIPS 2025.
- Daxberger et al. (2021). "Laplace redux—effortless Bayesian deep learning." NeurIPS 2021.
- Papamakarios et al. (2017). "Masked autoregressive flow for density estimation." NeurIPS 2017.
- Rezende & Mohamed (2015). "Variational inference with normalizing flows." ICML 2015.
- Liu et al. (2025). "Uncertainty quantification and confidence calibration in large language models: A survey." KDD 2025.
