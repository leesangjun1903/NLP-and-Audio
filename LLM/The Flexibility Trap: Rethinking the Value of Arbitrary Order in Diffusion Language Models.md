# The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models

---

## 1. Executive Summary (10문장 이내)

본 논문은 Diffusion Large Language Models(dLLMs)의 **임의 순서(arbitrary order) 생성**이 일반적 추론 과제에서 오히려 추론 잠재력을 제한한다는 반직관적 발견을 제시한다.  
dLLMs는 이론적으로 자유로운 토큰 생성 순서를 통해 AR 모델보다 우월한 추론 경로 탐색이 가능하다고 여겨졌다.  
그러나 저자들은 Pass@k 지표를 통해 임의 순서 생성이 AR 순서보다 낮은 해 공간 커버리지를 보임을 실증하였다.  
이 현상의 원인으로 **엔트로피 저하(entropy degradation)** 메커니즘을 제안하는데, 임의 순서 생성 시 모델이 불확실한 논리적 분기 토큰(예: "Therefore", "Since")을 우선 건너뜀으로써 해당 토큰의 엔트로피가 사전에 붕괴된다.  
기존 dLLM용 RL 방법들은 이 임의 순서 유지를 위해 조합론적 복잡도와 근사 불가능한 가능도 계산 등 과도한 복잡성을 감수한다.  
저자들은 이를 "유연성 세금(flexibility tax)"이라 규정하고, 단순히 AR 순서로 GRPO를 적용하는 **JustGRPO**를 제안한다.  
JustGRPO는 RL 학습 시에만 AR 순서를 적용하고, 추론 시에는 dLLM 고유의 병렬 디코딩을 완전히 보존한다.  
GSM8K 89.1%, MATH-500 45.1% 등 기존 복잡한 diffusion-specific RL 방법들을 능가하는 성능을 달성하였다.  
이 연구는 dLLM 개발에서 임의 순서 유연성의 가치에 대한 근본적인 재검토를 촉구한다.

---

### 1-1. 연구의 목적과 필요성

**목적:** dLLMs의 임의 순서 생성 유연성이 일반 추론 과제에서 실제로 이점을 제공하는지 실증적으로 검증하고, 더 단순하고 효과적인 RL 훈련 방법을 제안한다.

**필요성:**
- dLLMs의 RL 적용 연구들이 임의 순서 유지를 당연한 전제로 삼아, 조합폭발적 궤적 처리와 다루기 어려운 가능도 계산 등 불필요한 복잡성을 감수함 (p.3)
- 임의 순서가 실제로 추론 공간을 확장하는지에 대한 실증적 검증 부재
- 단순성과 효과성을 동시에 달성하는 dLLM RL 방법론의 필요

> 💡 **Pass@k**: $k$개의 독립 샘플 중 적어도 하나가 정답인 확률. 모델의 추론 잠재력(solution space coverage)을 측정하는 지표로, RL 학습으로 달성 가능한 성능의 상한선 역할을 함.

> 💡 **dLLMs (Diffusion Large Language Models)**: 텍스트 생성을 확산(diffusion) 과정으로 다루는 언어 모델. 전통적인 왼쪽→오른쪽 AR 생성과 달리, 마스킹된 토큰들을 반복적으로 복원하며 임의 순서로 생성 가능.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|-------|
| 임의 순서 생성이 일반 추론에서 추론 잠재력을 제한 | AR 순서 대비 낮은 Pass@k 스케일링 커브 (3개 dLLM 모델, 4개 벤치마크) | Figure 3, p.4 |
| 임의 순서 해 공간이 AR의 부분집합에 불과 | Pass@1024 분석: HumanEval에서 AO Only 0.6% vs AR Only 21.3% | Figure 4, p.4 |
| 순서 임의성이 클수록 추론 잠재력이 단조 감소 | 블록 크기 B=1(AR)→128(최대 임의)에 따른 Pass@k 단조 감소 | Figure 5, p.5 |
| 엔트로피 저하 메커니즘이 원인 | 논리 연결어("Therefore", "Thus" 등) 디코딩 시 AO의 엔트로피가 AR 대비 급감 | Figure 7, p.6 |
| 기존 dLLM RL 방법들의 복잡성은 불필요 | 조합폭발 궤적, 근사 불가 가능도, 샘플러-학습자 불일치 문제 분석 | Section 4.1, p.7 |
| JustGRPO가 단순하면서도 우월한 성능 | GSM8K 89.1%, MATH-500 45.1%로 기존 방법 초과 | Table 1, 2, p.8-9 |
| AR 훈련 후에도 병렬 디코딩 능력 보존 | EB-Sampler 기반 병렬 디코딩 실험: 고병렬 시 오히려 더 큰 성능 향상 | Figure 8, p.9 |
| 무작위 순서도 해결책이 되지 못함 | JustGRPO-Random이 GSM8K에서 82.2%로 AR(89.1%)에 크게 뒤처짐 | Table 5, p.18-19 |

---

### 2-1. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

#### 해결하고자 하는 문제

1. **임의 순서의 역설**: 이론적으로 더 큰 해 공간을 제공해야 할 임의 순서 생성이 실제 추론에서 AR 순서보다 낮은 Pass@k를 보이는 이유 규명
2. **기존 dLLM RL의 복잡성**: 임의 순서 유지를 위해 감수하는 세 가지 문제:
   - **토큰 수준 분해의 모호성**: $\pi(o_t|s_t)$ 형태의 고유한 조건부 확률 정의 불가
   - **다루기 어려운 시퀀스 가능도**: $\pi_\theta(o|q) = \sum_{\tau \in \mathcal{T}} \pi_\theta(o, \tau|q)$에서 $|\mathcal{T}| = O(N!)$
   - **샘플러-학습자 불일치**: 실제 샘플링 정책 $\pi^{\text{conf}}\_\theta$와 최적화 목표 $\pi_\theta$ 간 불일치

#### 제안하는 방법 (수식 포함)

**Step 1: MDM의 순전파 과정** (Eq. 1, p.3)

$$q(x_{t,k} \mid x_{0,k}) = \begin{cases} [\text{MASK}], & \text{with prob } t \\ x_{0,k}, & \text{with prob } 1-t \end{cases}$$

- $x_{t,k}$: 시간 $t$에서 $k$번째 토큰의 상태
- $x_{0,k}$: 원본 시퀀스의 $k$번째 토큰
- $t \in [0,1]$: 마스킹 비율(연속 시간 변수)

> 💡 **Masked Diffusion Model(MDM)**: 토큰을 무작위로 마스킹했다가 복원하는 방식으로 텍스트를 생성하는 확산 모델의 일종. 이미지 생성의 노이즈 추가/제거 과정을 텍스트의 마스킹/복원 과정으로 대체함.

**Step 2: MDM 학습 손실** (Eq. 2, p.3)

$$\mathcal{L}_{\text{MDM}}(\theta) = -\mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_t \sim q(x_t|x_0)} \left[ \frac{1}{t} \sum_{k=1}^{L} \mathbf{1}[x_{t,k} = [\text{MASK}]] \log p_\theta(x_{0,k} \mid x_t) \right]$$

- $L$: 시퀀스 길이
- $p_\theta(x_{0,k}|x_t)$: 마스킹된 위치 $k$에서 원본 토큰 분포를 추정하는 신경망
- $\mathbf{1}[\cdot]$: 지시 함수(해당 조건 참이면 1, 거짓이면 0)

> 💡 **NELBO (Negative Evidence Lower Bound)**: 확산 모델 학습의 목적함수. 정확한 데이터 가능도 계산이 어렵기 때문에 그 하한선(ELBO)의 음수를 최소화하는 방식으로 학습함.

**Step 3: Pass@k 추론 잠재력 측정** (Eq. 4, p.4)

$$\text{Pass}@k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

- $n$: 총 샘플 수
- $c$: 정답 샘플 수
- $k$: 평가 시 사용하는 샘플 수

**Step 4: AR 정책 정의를 위한 입력 구성** (Eq. 5, p.7)

$$\tilde{x}_k = [\underbrace{o_1, \ldots, o_{k-1}}_{\text{Observed}}, \underbrace{[\text{MASK}], \ldots, [\text{MASK}]}_{\text{Masked}}]$$

- $o_{1}, \ldots, o_{k-1}$: 이미 생성된 토큰들
- $[\text{MASK}]$: 아직 미생성된 위치

**Step 5: dLLM 위에 AR 정책 정의** (Eq. 6, p.7)

$$\pi^{\text{AR}}_\theta(\cdot \mid o_{<k}, q) \triangleq \text{Softmax}(f_{\theta,k}(\tilde{x}_k, q))$$

- $f_{\theta,k}$: 위치 $k$에서의 모델 로짓(logit)
- $q$: 질의(query)

**Step 6: 정확히 계산 가능한 AR 가능도** (Eq. 7, p.7)

$$\pi^{\text{AR}}_\theta(o \mid q) = \prod_{k=1}^{|o|} \pi^{\text{AR}}_\theta(o_k \mid o_{<k}, q)$$

**Step 7: JustGRPO 목적함수** (Eq. 8, p.8)

$$\mathcal{J}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi^{\text{AR}}_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{k=1}^{|o_i|} \left( \min\left(\rho_{i,k}\hat{A}_{i,k}, \text{clip}(\rho_{i,k}, 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,k}\right) - \beta \mathbb{D}_{\text{KL}} \right) \right]$$

- $\rho_{i,k} = \frac{\pi^{\text{AR}}\_\theta(o_{i,k}|o_{i, < k},q)}{\pi^{\text{AR}}\_{\theta_{\text{old}}}(o_{i,k}|o_{i, < k},q)}$ : 현재/이전 정책 간 중요도 비율
- $\hat{A}_{i,k} = (r(o_i) - \mu_G)/\sigma_G$: 그룹 표준화 어드밴티지
- $\varepsilon$: 클리핑 파라미터
- $\beta$: KL 정규화 계수
- $G$: 그룹 크기
- $\mathbb{D}_{\text{KL}}$: KL 다이버전스

> 💡 **GRPO (Group Relative Policy Optimization)**: DeepSeekMath에서 제안된 RL 알고리즘. 별도의 가치함수(value function) 없이, 동일 질의에서 생성된 여러 응답의 보상을 그룹 단위로 표준화하여 어드밴티지를 추정함.

> 💡 **중요도 비율(importance ratio) $\rho_{i,k}$**: PPO 계열 알고리즘에서 현재 정책과 샘플 수집 시 정책의 차이를 보정하기 위한 비율. 클리핑을 통해 너무 큰 정책 변화를 방지함.

> 💡 **KL 다이버전스(KL Divergence)**: 두 확률분포 간 차이를 측정하는 지표. RL에서 학습된 정책이 기준 정책에서 너무 멀리 벗어나지 않도록 정규화하는 데 사용됨.

#### 모델 구조

| 구성 요소 | 세부 내용 |
|-----------|----------|
| 기반 모델 | LLaDA-8B Instruct (Masked Diffusion Model) |
| 아키텍처 | 양방향 어텐션(causal masking 없음), 시퀀스 수준 디노이저 |
| 훈련 시 | AR 정책 $\pi^{\text{AR}}_\theta$ 정의 후 표준 GRPO 적용 |
| 추론 시 | 원래 dLLM 구조 그대로 사용, 병렬 디코딩(EB-Sampler) 가능 |
| AR 제약의 범위 | 훈련 시간(scaffold)에만 적용, causal masking 부과 없음 |

> 💡 **양방향 어텐션(Bidirectional Attention)**: 시퀀스의 모든 위치가 서로를 참조할 수 있는 어텐션 메커니즘. AR 모델은 미래 토큰을 볼 수 없도록 인과적 마스킹(causal masking)을 사용하지만, dLLM은 이를 사용하지 않음.

> 💡 **EB-Sampler (Entropy Bounded Sampler)**: Ben-Hamu et al. (2025)이 제안한 학습 불필요(training-free) 병렬 디코딩 방법. 엔트로피를 기반으로 여러 토큰을 동시에 언마스킹함.

#### 성능 향상

| 벤치마크 | JustGRPO (256) | 최고 기존 방법 | 향상 |
|----------|---------------|--------------|------|
| GSM8K | **89.1%** | SPG: 86.1% | +3.0%p |
| MATH-500 | **45.1%** | SPG: 40.0% | +5.1%p |
| HumanEval | **49.4%** | LLaDOU: 59.1%† | - |
| MBPP | **52.4%** | LLaDOU: 51.6%† | +0.8%p |

† LLaDOU는 추가 보조 모듈 사용, LLaDA-1.5는 대규모 사설 데이터 사용

#### 한계

- **단일 모델 실험**: LLaDA-Instruct 위주, 다른 dLLM 패밀리에 대한 JustGRPO 적용 결과 부족
- **태스크 범위**: 수학, 코딩에 한정; 비구조적 언어 추론, 창의적 작문 등에서의 검증 없음
- **임의 순서의 잠재적 이점 영역 미탐색**: 스도쿠, 제브라 퍼즐 등 특수 과제에서의 임의 순서 우위와의 조화 방법 미제시
- **훈련 오버헤드**: 각 위치를 독립적으로 평가해야 하므로 AR 모델 대비 추가 계산 비용 발생 (단, JustGRPO-Fast로 부분 완화)
- **이론적 정당화 부재**: 왜 AR 순서가 항상 더 나은 탐색을 유도하는지에 대한 이론적 증명 없음

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 근거 위치 |
|------|----------|
| AR 순서가 더 높은 Pass@k를 보임 | Figure 1 (Left), p.2; Figure 3, p.5 |
| 임의 순서 해 공간이 AR 해 공간의 부분집합 | Figure 4, p.4 |
| 블록 크기 증가 시 Pass@k 단조 감소 | Figure 5, p.5 |
| 임의 순서가 논리 연결어를 우선적으로 건너뜀 | Figure 6, p.6 |
| 논리 분기점 토큰의 엔트로피 저하 | Figure 7, p.6 |
| JustGRPO의 벤치마크 우위 (시스템 수준) | Table 1, p.8 |
| 동일 조건 재현 실험에서도 JustGRPO 우위 | Table 2, p.9 |
| AR 훈련 후에도 병렬 디코딩 능력 보존 | Figure 8, p.9-10 |
| JustGRPO가 ESPO 대비 우수한 시간-정확도 트레이드오프 | Figure 9, p.10 |
| 무작위 순서가 커버리지 개선에 도움 안 됨 | Table 4, p.19 |
| JustGRPO-Random이 AR보다 현저히 열등 | Table 5, p.19 |
| 일반 능력 보존 | Table 6, p.19 |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

**연구 주제:** dLLMs의 임의 순서 유연성이 일반 추론 과제에서 갖는 가치와 RL 훈련에서의 역할

**방법:**
- Pass@k 지표 ($\text{Pass}@k = \mathbb{E}\left[1 - \binom{n-c}{k}/\binom{n}{k}\right]$) 기반 실증 비교
- 3개 dLLM(LLaDA-Instruct, Dream-Instruct, LLaDA-1.5) × 4개 벤치마크(GSM8K, MATH-500, HumanEval, MBPP)
- 블록 크기 스위프(B=1,8,32,128)를 통한 임의성 정도 실험
- AR 정책 $\pi^{\text{AR}}_\theta$ 정의 후 표준 GRPO 적용

**저자 보고 수치 (직접 인용):**
- JustGRPO GSM8K 89.1% (seq len 256) (Table 1, p.8)
- JustGRPO MATH-500 45.1% (Table 1, p.8)
- HumanEval에서 AO Only 0.6% vs AR Only 21.3% (Figure 4, p.4)
- 모든 $k \in \{8,32,128\}$에서 블록 크기 증가 시 Pass@k 단조 감소 (Figure 5, p.5)
- 일반 능력 벤치마크에서 MMLU 65.8% (기존 65.5%), MMLU-Pro 36.7% (기존 37.0%) 등 소폭 변동 (Table 6, p.19)

### 리뷰어(나)의 해석 및 평가

1. **결과의 강점**: 3개 모델과 4개 벤치마크에 걸친 일관된 패턴은 설득력이 있음. 특히 단조 감소 패턴(Figure 5)은 단순 이분법을 넘어 연속적 증거를 제공함.

2. **비교 공정성 문제 (⚠️ 통계적 취약)**: Table 1의 baseline들은 서로 다른 LoRA/full fine-tuning, 1/2 토큰/스텝, 공개/사설 데이터 등 이질적 설정을 사용함. Table 2에서 동일 조건 재현을 시도했으나 일부 모델(HumanEval, MBPP 결과 없음)만 포함됨.

3. **엔트로피 저하 인과성 미확립**: Figure 7은 상관관계를 보여주지만, 엔트로피 저하가 Pass@k 감소의 직접 원인임을 인과적으로 증명하지는 않음.

4. **Pass@k 해석의 전제**: RL이 base model의 분포를 날카롭게만 만든다는 전제(Yue et al., 2025)에 기반하여 Pass@k를 RL 상한선으로 사용하는데, 이 전제 자체가 도전받을 수 있음.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 문제 유형 | 해당 내용 | 위치 |
|-----------|----------|------|
| ⚠️ 이질적 설정 비교 | Table 1: LoRA vs. full fine-tuning, 1 vs. 2 토큰/스텝 혼재 | Table 1, p.8 |
| ⚠️ 사설 데이터 사용 | LLaDA-1.5는 "significantly larger scale" 사설 데이터 사용으로 직접 비교 부적절 | Table 1 각주, p.8 |
| ⚠️ 보조 모듈 사용 | LLaDOU는 추가 trainable 모듈 사용, 구조적으로 JustGRPO와 다름 | Table 1 각주, p.8 |
| ⚠️ 단일 시드 결과 | 표준편차, 신뢰구간 미보고, 랜덤 시드 영향 불명확 | 전체 실험 |
| ⚠️ 온도 최적화 불일치 | 임의 순서의 최적 온도(높음)와 AR의 최적 온도(~0.6)가 다름에도 단일 온도로 주요 비교 | Appendix B, p.16 |
| ⚠️ HumanEval/MBPP 재현 부재 | Table 2에서 d1*, SPG*의 HumanEval, MBPP 결과 누락("-" 표시) | Table 2, p.9 |
| ⚠️ 소규모 훈련 스텝 | 125 스텝(GSM8K는 50 스텝에 수렴)으로 모든 태스크 일괄 적용 | Table 3, p.15 |

---

## 6. 문서가 답하지 않는 질문

1. **왜 AR 순서가 이론적으로 더 나은 탐색을 제공하는가?** 엔트로피 저하 현상은 기술되었으나 이것이 Pass@k 감소로 이어지는 인과적 메커니즘에 대한 이론적 증명이 없음.

2. **스도쿠·제브라 퍼즐 등 비순차적 추론이 유리한 과제와의 경계는 어디인가?** 임의 순서가 유리한 과제와 불리한 과제를 구분하는 기준이 제시되지 않음.

3. **JustGRPO를 다른 dLLM 아키텍처(예: Mercury, Gemini Diffusion)에 적용하면?** LLaDA 계열 외 모델에서의 일반화 여부 미검증.

4. **더 긴 시퀀스(512 이상)에서도 동일한 결과가 나타나는가?** 512까지의 결과만 보고되며, 긴 연쇄 추론(chain-of-thought)이 필요한 복잡한 문제에서의 동작 미검증.

5. **JustGRPO-Fast에서 상위 25% 엔트로피 임계값은 어떻게 결정되었는가?** 임계값 선택의 민감도 분석 부재.

6. **임의 순서 유연성을 부분적으로 활용하는 하이브리드 방법은 가능한가?** 훈련 중 점진적으로 AR→임의 순서로 전환하는 커리큘럼 접근 등의 탐색 없음.

7. **역방향 정제(bidirectional refinement)와 JustGRPO의 결합 효과는?** Appendix E에서 가능성을 언급하나 실험적 검증 없음.

8. **보상 함수 설계가 결과에 얼마나 영향을 미치는가?** 이진 보상(수학)과 패스율 기반 보상(코딩) 외 다른 보상 설계의 영향 미탐색.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2) - "Less Flexibility Unlocks Better Reasoning Potential"

**해석:** 두 부분으로 구성된 핵심 요약 그림.
- **(Left)** LLaDA-Instruct에서 AR 순서(■)가 임의 순서(●)보다 높은 Pass@k를 보임. $k=1$에서 두 방법이 유사하지만(각각 약 80-81%), $k$가 커질수록 AR이 더 가파르게 상승하여 $k=128$에서 약 4%p 이상 격차.
- **(Right)** JustGRPO가 d1(81.1%), ESPO(82.3%), SPG(86.1%), GDPO(82.8%)를 GSM8K에서 모두 초과하며 89.1% 달성.
- **의의**: 이 그림은 논문 전체의 핵심 주장을 단일 시각으로 제시. 왼쪽의 원인 분석이 오른쪽의 방법론 제안으로 연결되는 논리 구조를 직관적으로 보여줌.

### Figure 3 (p.5) - "Reasoning Potential Measured by Pass@k"

**해석:** 3×4 격자 형태로 3개 dLLM × 4개 벤치마크에 대한 Pass@k 커브 비교.
- 모든 모델-벤치마크 조합에서 **일관된 패턴**: $k=1$에서 임의 순서가 경쟁력 있거나 우위이나, $k$ 증가에 따라 AR 순서의 스케일링 기울기가 더 가파름.
- LLaDA-Instruct GSM8K: $k=1$에서 AO≈80%, AR≈79%이나, $k=128$에서 AR이 약 95% 이상으로 상승하며 AO를 크게 추월.
- **의의**: 결과의 일반성을 3개 아키텍처에 걸쳐 검증함으로써 LLaDA 특유 현상이 아님을 입증. Pass@1 우위에도 불구하고 탐색 다양성에서 AR이 우월함을 시각화.

### Figure 4 (p.4) - "Solution Space Coverage by Pass@1024"

**해석:** 파이차트 형태로 AO Only / AR Only / Both Solved 비율을 4개 벤치마크에 표시.
- HumanEval: AR Only 21.3% vs AO Only 0.6% — AR이 배타적으로 해결하는 문제가 35배 많음.
- MBPP: AR Only 14.0% vs AO Only 0.8%.
- GSM8K: AR Only 1.2% vs AO Only 0.0%.
- **의의**: 임의 순서 생성이 "이론적으로 더 큰 해 공간"이라는 주장이 실제로는 AR의 부분집합에 불과함을 보여주는 결정적 증거. 특히 AO Only가 극소한 점은 임의 순서만의 고유한 해가 거의 없음을 시사.

### Figure 7 (p.6) - "Entropy Degradation"

**해석:** "Therefore", "Thus", "Since", "To", "Now", "determine", "Given" 등 논리 연결어에 대해 임의 순서(파란 막대)와 AR 순서(주황 점선) 디코딩 시 평균 엔트로피 비교.
- 전체 평균 토큰 엔트로피(점선)는 두 방법이 유사하지만, 논리 연결어 위치에서만 임의 순서의 엔트로피가 AR 대비 급격히 낮음.
- 예: "Therefore"에서 AR ~1.5 vs AO ~0.5.
- **의의**: "유연성 함정"의 메커니즘을 정량적으로 증명. 임의 순서가 전반적 불확실성을 줄이는 것이 아니라, 정확히 분기점 역할을 하는 중요 토큰에서만 선택적으로 엔트로피를 억제함을 보여줌.

> 💡 **엔트로피(Entropy)**: 확률분포의 불확실성을 나타내는 지표. 높은 엔트로피는 여러 가능성이 열려 있음을, 낮은 엔트로피는 하나의 결과가 결정적임을 의미. 추론 맥락에서 높은 엔트로피의 분기 토큰은 탐색 공간을 다양하게 유지하는 역할을 함.

### Figure 8 (p.9) - "JustGRPO Preserves Parallel Decoding Capability"

**해석:** 토큰/스텝(병렬도)에 따른 정확도 변화를 4개 벤치마크에 대해 LLaDA-Instruct(파란 점)와 JustGRPO(주황 점) 비교.
- 모든 병렬도 설정에서 JustGRPO가 기준 모델 대비 향상: GSM8K +10.6%(1 tok/step) → +13.8%(7.5 tok/step).
- MBPP: +10.6%(1 tok/step) → +25.5%(~5 tok/step) — 병렬도 증가에 따라 격차 확대.
- 기준 모델은 병렬도 증가 시 성능이 급감하는 반면, JustGRPO는 상대적으로 안정적.
- **의의**: AR 훈련이 병렬 디코딩 능력을 손상시킨다는 우려를 불식. 오히려 분포 정제 효과로 병렬 샘플링의 근사 오류에 더 강건한 모델을 만들어냄.

---

## 8. 결론, 시사점, 후속 연구

### 8-1. 저자들이 제시한 시사점 및 후속 연구 계획

**시사점 (Section 7, p.11):**
1. dLLM의 임의 순서 유연성이 일반 추론에서는 탐색(exploration)보다 착취(exploitation)를 조장함
2. RL 훈련 시 AR 순서 제약이 오히려 추론 잠재력을 확장하는 반직관적 결론
3. dLLMs 개발에서 임의 vs. AR 순서의 트레이드오프 재검토 필요

**저자 언급 후속 연구 방향 (Appendix E, p.19):**
- 역방향 정제(bidirectional refinement)와 JustGRPO의 결합 (CDLM, ParallelBench 방향성)
- JustGRPO-Fast의 엔트로피 기반 선택적 위치 평가 확장

### 8-1. 모델의 일반화 성능 향상 가능성

본 논문에서 일반화와 관련된 직접적 증거와 가능성:

**현재 일반화 증거:**
- **다중 모델 일반화**: LLaDA-Instruct, Dream-Instruct, LLaDA-1.5 3개 모델에서 동일 패턴 확인 (Figure 3)
- **다중 도메인 일반화**: 수학(GSM8K, MATH-500)과 코딩(HumanEval, MBPP) 두 도메인에서 일관된 성능
- **일반 능력 보존**: MMLU, MMLU-Pro, HellaSwag, ARC-C에서 성능 유지 (Table 6, p.19) — AR scaffold가 reasoning-specific 개선을 달성하면서 일반 지식은 보존

**일반화 향상 가능성 분석:**

AR scaffold 훈련 방식은 구조적으로 다음과 같은 일반화 이점을 제공할 가능성이 있음:

1. **분포 정제 효과**: AR 훈련이 "특정 궤적에 과적합"하지 않고 기저 모델 분포를 전반적으로 정제함 → 병렬 샘플링 근사 오류에 더 강건 (Figure 8 결과가 이를 지지)

2. **도메인 확장 가능성**: 구조화된 순차적 추론이 필요한 모든 과제(법률 추론, 과학적 추론 등)에 적용 가능성 높음. 단, 비선형적 사고가 유리한 과제(예: 역추론)에서는 검증 필요.

3. **스케일 일반화**: JustGRPO-Fast를 통한 연산 효율화는 더 큰 모델(>8B)이나 더 긴 시퀀스에 적용 시 일반화 가능성을 실질적으로 높임.

**한계 및 우려:**
- 특수 추론 과제(스도쿠, 퍼즐)에서의 일반화는 미검증 — 오히려 임의 순서가 유리할 수 있음
- 다국어, 멀티모달 설정에서의 일반화 불명확

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

#### 주요 관련 연구 계보

| 연구 | 연도 | 핵심 기여 | 본 논문과의 관계 |
|------|------|-----------|----------------|
| DDPM (Ho et al.) | 2020 | 연속 도메인 확산 모델 기반 확립 | dLLM의 이론적 토대 |
| Diffusion-LM (Li et al.) | 2022 | 임베딩 공간 확산 텍스트 생성 | 초기 dLLM 시도, 이산 토큰 문제 노출 |
| MDLM/SMDM (Lou et al., Sahoo et al., Shi et al.) | 2024 | 이산 마스킹 확산 모델 확립 | MDM 수식(Eq.1, 2)의 기반 |
| LLaDA (Nie et al.) | 2025 | 대규모 MDM, AR 수준 성능 | 본 논문 기반 모델 |
| Dream (Ye et al.) | 2025 | 추론 특화 dLLM | 비교 대상 모델 |
| d1 (Zhao et al.) | 2025 | dLLM + RL(GRPO 유사) 초기 시도 | JustGRPO의 직접 비교 대상 |
| ESPO (Ou et al.) | 2026 | 시퀀스 수준 관점의 원칙적 RL | Table 1, 2에서 직접 비교 |
| SPG (Wang et al.) | 2026 | 샌드위치 정책 그래디언트 | 복잡한 임의 순서 보존 RL의 대표 사례 |
| LLaDOU (Huang et al.) | 2025 | 보조 위치 선택 정책 추가 | 임의 순서 유지하되 궤적 가능도 직접 추정 |
| Mercury (Inception Labs) | 2025 | 초고속 추론 dLLM | 병렬 디코딩 효율성 측면 비교 |

> 💡 **RLVR (Reinforcement Learning with Verifiable Rewards)**: 수학 계산 결과, 코드 실행 결과 등 자동으로 검증 가능한 보상을 사용하는 강화학습 패러다임. DeepSeek-R1, o1 등에서 추론 능력 향상에 효과적임이 입증됨.

#### 본 논문이 미치는 영향

1. **패러다임 전환**: "더 많은 유연성 = 더 나은 추론"이라는 암묵적 전제에 도전하여, dLLM RL 연구의 설계 철학 재고 촉구

2. **단순성의 복권**: 복잡한 diffusion-specific RL 방법들(ESPO, SPG, GDPO)이 갖는 복잡도의 정당성 약화 — 방법론 단순화 트렌드 가속화 가능

3. **평가 기준 재정립**: Pass@k를 추론 잠재력 측정의 표준 지표로 강조하며 단순 Pass@1 중심 평가의 한계 부각

4. **훈련-추론 분리 원칙**: RL 훈련과 추론 실행의 최적 전략이 다를 수 있음을 보여줌 — 다른 모델 패밀리에서도 이 원칙 적용 가능성

#### 향후 연구 시 고려할 점

1. **적용 범위 경계 탐색**: 임의 순서가 실제로 유리한 과제와 AR이 유리한 과제를 구분하는 분류 기준 수립 필요

2. **이론적 기반 강화**: 엔트로피 저하와 Pass@k 감소 간의 인과관계를 정보이론적으로 증명하는 후속 연구 필요

3. **적응형 순서 전략**: 훈련 중 엔트로피가 높은 분기점에서만 AR을 강제하고 나머지는 임의 순서를 허용하는 하이브리드 방법 탐색

4. **다른 RL 알고리즘과의 결합**: JustGRPO 프레임워크에서 PPO, REINFORCE, DPO 등 다른 RL 방법 적용 비교

5. **더 강한 dLLM과의 결합**: Mercury, Gemini Diffusion 등 더 강력한 기반 모델에 JustGRPO 적용 시 성능 한계 탐색

6. **온도-순서 상호작용 심화 연구**: Appendix B에서 보인 임의 순서의 최적 온도가 더 높다는 발견 — 온도 제어를 통한 임의 순서 탐색 가능성 미완결

7. **역방향 정제와의 통합**: dLLM의 양방향 주의를 활용한 이미 생성된 출력의 반복적 개선과 JustGRPO 결합

8. **멀티모달 dLLM으로의 확장**: MMaDA(Yang et al., 2025) 등 멀티모달 확산 모델에서의 임의 순서 vs. AR 순서 검토

---

## 참고 자료

본 답변은 전적으로 제공된 논문 원문을 기반으로 작성되었습니다:

**논문 원문:**
- Ni, Z., Wang, S., Yue, Y., Yu, T., Zhao, W., Hua, Y., Chen, T., Song, J., Yu, C., Zheng, B., & Huang, G. (2026). **"The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models."** arXiv:2601.15165v4 [cs.CL].
- Project page: https://nzl-thu.github.io/the-flexibility-trap

**논문 내 인용 주요 참고문헌:**
- Nie et al. (2025). "Large Language Diffusion Models." NeurIPS. (LLaDA)
- Shao et al. (2024). "DeepSeekMath." arXiv:2402.03300. (GRPO 원논문)
- Chen et al. (2021). "Evaluating Large Language Models Trained on Code." arXiv:2107.03374. (Pass@k)
- Yue et al. (2025). "Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?" NeurIPS.
- Ben-Hamu et al. (2025). "Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking." NeurIPS. (EB-Sampler)
- Ou et al. (2026). "Principled RL for Diffusion LLMs from a Sequence-Level Perspective." ICLR. (ESPO)
- Wang et al. (2026a). "SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models." ICLR.
- Zhao et al. (2025). "d1: Scaling Reasoning in Diffusion Large Language Models via RL." NeurIPS.

**⚠️ 주의:** 이 논문은 arXiv:2601.15165v4 (2026년 6월 9일 기준)로, 아직 peer-review 최종 출판 전 프리프린트일 가능성이 있으며, 일부 인용된 참고문헌(2026년도 ICLR, NeurIPS 논문 등)은 제 학습 데이터 기준(2024년 초)으로 직접 확인이 불가한 미래 문헌입니다. 해당 참고문헌들의 내용은 본 논문의 기술에 의존하여 서술하였음을 밝힙니다.

# The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장 (Counter-intuitive Claim)

이 논문의 핵심 주장은 **역설적(counter-intuitive)**이다:

> **Diffusion Language Models(dLLMs)의 임의 순서(arbitrary-order) 생성 능력이, 수학/코딩과 같은 일반 추론 과제에서는 오히려 추론 잠재력을 제한한다.**

직관적으로는 임의 순서 생성이 자동회귀(AR) 방식의 엄격한 좌→우 순서를 포함하는 더 큰 해 공간(solution space)을 제공하므로 더 나은 추론을 가능케 해야 하지만, 실험적으로는 **반대 현상**이 관찰된다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **현상 발견** | 임의 순서 생성이 "포킹 토큰(forking token)"을 우회하여 **엔트로피 저하(entropy degradation)**를 유발함을 실증 |
| **메커니즘 분석** | Pass@k 지표와 엔트로피 측정을 통해 솔루션 공간 붕괴 메커니즘 규명 |
| **방법론 제안** | **JustGRPO**: 복잡한 diffusion-specific RL 적응 없이 표준 GRPO를 dLLM에 직접 적용 |
| **실용적 가치** | 병렬 디코딩 능력을 완전히 보존하면서도 GSM8K 89.1%, MATH-500 45.1% 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

#### 문제 ①: 임의 순서 생성의 역효과 — "유연성 함정(Flexibility Trap)"

dLLMs는 추론 과정에서 **"포킹 토큰(forking tokens)"**, 즉 "Therefore", "Since", "Thus"와 같은 논리적 분기점을 우회(bypass)하는 경향이 있다. 이 토큰들은 다음 추론 경로를 결정하는 분기점으로서 원래 높은 엔트로피를 가져야 한다.

임의 순서 생성 메커니즘에서는:
1. 모델이 **확신이 높은(low-uncertainty) 토큰**을 먼저 생성
2. 나중에 bypassed된 포킹 토큰을 채울 때, **이미 확정된 미래 맥락이 해당 토큰의 불확실성을 제거**
3. 결과적으로 **엔트로피 저하**: 포킹 토큰이 더 이상 열린 분기점이 아니라 "채우기" 역할만 수행

이를 **엔트로피 저하(entropy degradation)** 현상이라 명명한다.

#### 문제 ②: 기존 dLLM용 RL 방법의 복잡성

기존 방법들(d1, ESPO, GDPO, SPG 등)은 임의 순서를 보존하기 위해 다음과 같은 어려움을 감수한다:

- **토큰 수준 분해의 모호성**: $\pi(o_t | s_t)$ 형태의 고유한 조건부 확률 정의 불가
- **시퀀스 우도의 비가산성**:
$$\pi_\theta(o \mid q) = \sum_{\tau \in \mathcal{T}} \pi_\theta(o, \tau \mid q), \quad |\mathcal{T}| = O(N!)$$
  → 정확한 우도 계산 불가, ELBO 근사에 의존
- **샘플러-학습자 불일치(sampler-learner mismatch)**: rollout 시 신뢰도 기반 샘플링 $\pi_\theta^{\text{conf}}(o|q)$와 최적화 목표 $\pi_\theta(o|q)$ 간의 괴리

---

### 2.2 제안 방법 (JustGRPO)

#### 핵심 아이디어

RL 훈련 단계에서만 임의 순서를 포기하고 **dLLM을 AR 정책으로 취급**한다. 이는 구조적 변경 없이(causal masking 없이) 훈련 시에만 AR scaffold를 적용하는 것이다.

#### 수식: AR 정책 정의

$k$번째 토큰 $o_k$를 생성하기 위한 입력 시퀀스를 다음과 같이 구성한다:

$$\tilde{x}_k = [\underbrace{o_1, \ldots, o_{k-1}}_{\text{Observed}}, \underbrace{[\text{MASK}], \ldots, [\text{MASK}]}_{\text{Masked}}] $$

이를 바탕으로 대리 AR 정책(surrogate autoregressive policy)을 정의한다:

```math
\pi_\theta^{\text{AR}}(\cdot \mid o_{ < k}, q) \triangleq \text{Softmax}(f_{\theta,k}(\tilde{x}_k, q))
```

여기서 $f_{\theta,k}$는 position $k$에서의 모델 logit이다.

이를 통해 시퀀스 우도를 **정확히** 분해할 수 있다:

$$\pi_\theta^{\text{AR}}(o \mid q) = \prod_{k=1}^{|o|} \pi_\theta^{\text{AR}}(o_k \mid o_{ < k}, q) $$

#### Masked Diffusion Model (MDM) 기반 수식

MDM의 순전파 과정:

$$q(x_{t,k} \mid x_{0,k}) = \begin{cases} [\text{MASK}], & \text{with prob } t \\ x_{0,k}, & \text{with prob } 1-t \end{cases} $$

MDM 훈련 손실 (Negative ELBO):

$$\mathcal{L}_{\text{MDM}}(\theta) = -\mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_t \sim q(x_t|x_0)}\left[\frac{1}{t}\sum_{k=1}^{L} \mathbf{1}[x_{t,k} = [\text{MASK}]] \log p_\theta(x_{0,k} \mid x_t)\right] $$

#### GRPO 목적 함수

기본 GRPO 목적함수:

$$\mathcal{J}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{k=1}^{|o_i|}\left(\min\left(\rho_{i,k}\hat{A}_{i,k}, \text{clip}(\rho_{i,k}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,k}\right) - \beta\mathbb{D}_{\text{KL}}\right)\right] $$

**JustGRPO의 목적함수** (AR 정책 기반):

$$\mathcal{J}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}^{\text{AR}}}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{k=1}^{|o_i|}\left(\min\left(\rho_{i,k}\hat{A}_{i,k}, \text{clip}(\rho_{i,k}, 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,k}\right) - \beta\mathbb{D}_{\text{KL}}\right)\right] $$

여기서 중요도 비율(importance ratio):

$$\rho_{i,k} = \frac{\pi_\theta^{\text{AR}}(o_{i,k} \mid o_{i, < k}, q)}{\pi_{\theta_{\text{old}}}^{\text{AR}}(o_{i,k} \mid o_{i, < k}, q)}$$

어드밴티지:

$$\hat{A}_{i,k} = \frac{r(o_i) - \mu_G}{\sigma_G}$$

Pass@k 지표 (비편향 추정량):

$$\text{Pass@}k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right] $$

---

### 2.3 모델 구조

JustGRPO는 **모델 아키텍처를 변경하지 않는다**. 주요 구조적 특징:

```
훈련 단계:
  LLaDA-8B Instruct (양방향 어텐션 유지)
  + AR scaffold (causal masking 없음)
  → 표준 GRPO 적용

추론 단계:
  훈련된 모델 (양방향 어텐션 보존)
  + 병렬 디코딩 (EB-Sampler 등)
  → 기존 dLLM 추론 효율성 완전 보존
```

훈련 하이퍼파라미터:
- Base Model: LLaDA 8B Instruct
- Learning Rate: $5 \times 10^{-6}$
- Group Size $G$: 16
- Global Batch Size: 64
- Training Steps: 125
- Max Completion Length: 256
- Hardware: $16 \times$ NVIDIA H100 GPUs

---

### 2.4 성능 향상

#### 주요 벤치마크 결과 (LLaDA-Instruct 기준, Seq Len 256)

| 모델 | GSM8K | MATH-500 | HumanEval | MBPP |
|---|---|---|---|---|
| d1 (Zhao et al., 2025) | 81.1 | 38.6 | — | — |
| ESPO (Ou et al., 2026) | 82.3 | 39.0 | 42.1 | 44.6 |
| GDPO (Rojas et al., 2026) | 82.8 | 39.6 | 39.6 | 50.6 |
| SPG (Wang et al., 2026a) | 86.1 | 40.0 | — | — |
| **JustGRPO** | **89.1** | **45.1** | **49.4** | **52.4** |

#### 통일된 실험 환경에서의 비교 (Table 2)

| 모델 | GSM8K | MATH-500 | HumanEval | MBPP |
|---|---|---|---|---|
| d1* | 83.8 | 39.2 | — | — |
| ESPO* | 84.7 | 40.3 | 42.1 | 44.6 |
| SPG* | 86.9 | 41.8 | — | — |
| **JustGRPO** | **89.1** | **45.1** | **49.4** | **52.4** |

#### 병렬 디코딩과의 호환성

- 병렬 토큰 수 증가 시 성능 우위가 **더욱 확대**됨
- MBPP에서 1 token/step: +10.6% → ~5 tokens/step: +25.5%

---

### 2.5 한계

논문에서 명시적으로 언급된 한계 및 추론 가능한 한계:

1. **per-iteration 계산 오버헤드**: dLLM은 각 위치를 독립적으로 평가해야 하므로 단일 causal forward pass가 불가능 → 계산 비용 증가 (단, JustGRPO-Fast로 부분 완화)
2. **일반화 범위**: 수학/코딩에 집중된 실험. 창의적 글쓰기, Sudoku, Zebra puzzle 등 특정 구조화된 과제에서는 임의 순서가 여전히 유리할 수 있음
3. **단일 베이스 모델 의존**: 주로 LLaDA-Instruct에서 실험. 다른 dLLM 아키텍처에서의 일반화 여부 추가 검증 필요
4. **랜덤 순서의 실패**: 랜덤 순서(JustGRPO-Random)도 GSM8K 82.2%에 그침. 즉, 순서의 구조적 특성(인과성) 자체가 중요
5. **bidirectional refinement 미활용**: 훈련 후 모델이 기존 출력을 반복적으로 수정하는 능력 활용 가능성이 미탐구 상태

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Pass@k를 통한 잠재적 추론 공간의 확장

논문의 핵심 발견은 **AR 순서가 더 높은 Pass@k를 달성**한다는 것이다:

$$\text{Pass@k}_{\text{AR}} > \text{Pass@k}_{\text{Arbitrary}} \quad \text{for } k \geq 2$$

이는 AR 순서로 훈련된 모델이 **더 넓은 해 공간을 탐색**할 수 있음을 의미한다. 구체적으로:

- HumanEval에서 Pass@1024 기준: AR만 해결한 문제 21.3% vs. AO만 해결한 문제 0.6%
- 임의 순서로 해결되는 문제들은 AR로 해결되는 문제들의 **부분집합**에 가까움

### 3.2 엔트로피 보존과 일반화

AR 순서는 논리적 포킹 토큰에서 높은 엔트로피를 유지한다:

$$H_{\text{AR}}(\text{forking tokens}) \gg H_{\text{AO}}(\text{forking tokens})$$

이 높은 엔트로피가 **다양한 추론 경로를 샘플링**할 수 있게 하여, RL 훈련 시 더 많은 positive reward signal을 제공한다. 이는 일반화에 직접적으로 기여한다.

### 3.3 병렬 디코딩 하에서의 일반화 강건성

JustGRPO로 훈련된 모델은 병렬 디코딩(Tokens/Step 증가)에서 **더 강건한 성능**을 보인다:

- 기존 모델: 병렬 토큰 수 증가 시 성능 급락
- JustGRPO: 병렬 토큰 수 증가에도 **안정적인 성능** 유지

이는 AR scaffold 훈련이 특정 trajectory에 과적합하는 대신 **기저 모델 분포 자체를 개선**함을 시사한다. 개선된 분포는 병렬 샘플링의 근사(approximation)에 더 탄탄하여, 다양한 추론 조건에서 일반화 성능이 향상된다.

### 3.4 일반 능력(General Capability) 보존

JustGRPO 훈련 후 비추론 벤치마크 결과:

| 벤치마크 | LLaDA-Instruct | JustGRPO | 변화 |
|---|---|---|---|
| MMLU | 65.5% | 65.8% | +0.3% |
| MMLU-Pro | 37.0% | 36.7% | -0.3% |
| HellaSwag | 74.6% | 74.8% | +0.2% |
| ARC-C | 88.5% | 87.5% | -1.0% |

일반 능력이 **거의 보존**됨. 이는 AR scaffold가 모델의 지식 인코딩 방식에는 영향을 주지 않고 **추론 경로 탐색 방식만을 개선**함을 의미한다.

### 3.5 블록 크기와 임의성의 연속적 관계

반자동회귀(semi-autoregressive) 블록 크기 $B$ 실험에서:

$$\text{Pass@}k \downarrow \text{ monotonically as } B \uparrow$$

이는 **임의성이 적을수록 일관되게 추론 잠재력이 높아짐**을 보여주며, 이 관계가 단순히 두 극단 비교가 아닌 **연속적이고 일반적인 현상**임을 지지한다.

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 향후 연구에 미치는 영향

#### ① dLLM 설계 철학의 재검토

이 논문은 "더 많은 자유도 = 더 나은 성능"이라는 dLLM 분야의 암묵적 가정에 근본적인 의문을 제기한다. 향후 dLLM 설계에서:

- **사전학습(pre-training)**: 임의 순서 학습이 데이터 분포를 더 느슨하게 근사한다는 Du et al. (2025)의 분석과 함께, 사전학습에서도 AR 순서의 효용성을 재평가할 필요
- **아키텍처 설계**: 양방향 어텐션을 보존하면서도 생성 순서에 구조적 귀납 편향(inductive bias)을 부여하는 새로운 아키텍처 탐구 가능

#### ② RL for dLLMs 연구 방향 단순화

기존 연구들(ESPO, GDPO, SPG 등)이 복잡한 diffusion-specific 적응을 개발한 것과 달리, JustGRPO는 **단순함(simplicity)이 강력한 기준선(baseline)이 될 수 있음**을 보였다. 이는:

- 새로운 dLLM RL 방법은 JustGRPO 대비 유의미한 향상을 입증해야 함
- 복잡한 trajectory 처리의 실질적 필요성에 대한 재검토 유도

#### ③ 탐색-활용 트레이드오프(Exploration-Exploitation Tradeoff) 이해 심화

엔트로피 저하 현상은 **추론에서의 탐색-활용 트레이드오프**를 더 정밀하게 이해하는 틀을 제공한다:

- 포킹 토큰의 엔트로피 관리가 RL 훈련 성능에 핵심임을 시사
- "Low-probability tokens sustain exploration" (Wang et al., 2025; Huang et al., 2025a)과 같은 연구와 연계하여 **엔트로피 제어 기반 RL** 연구의 발전 가능

#### ④ JustGRPO-Fast의 파급 효과

상위 25% 고엔트로피 위치에서만 $\rho_{i,k}$를 계산하는 JustGRPO-Fast는:

- **희소 엔트로피 구조(sparse entropy structure)**를 활용한 효율적 RL의 가능성을 제시
- 포킹 토큰 식별 → 선택적 최적화 → 효율적 훈련의 패러다임이 AR 모델에도 적용 가능할지 탐구 여지

#### ⑤ 훈련-추론 분리(Decoupling) 원리의 확장

JustGRPO가 보여준 핵심 원리: **훈련 목적(탐색 효율)과 추론 실행(병렬 디코딩)을 분리**할 수 있다. 이 원리는:

- 다른 비자기회귀 모델(Non-AR models) 훈련 전략에도 적용 가능
- 제한적 훈련 분포로 강건한 추론 능력을 학습하는 일반적 접근으로 확장 가능

---

### 4.2 향후 연구 시 고려할 점

#### ① 임의 순서가 유리한 과제 경계 규명

이 논문은 수학/코딩에서 임의 순서의 한계를 보였으나, Ye et al. (2025a), Kim et al. (2025)은 Sudoku, Zebra puzzle에서의 우위를 보였다. 따라서:

- **어떤 과제 특성이 임의 순서의 유/불리를 결정하는가?**
- 과제별 "forking token" 밀도, 구조적 의존성, 인과성의 역할 등을 정밀 분석 필요

#### ② 더 다양한 dLLM 아키텍처에서의 검증

현재 실험은 주로 LLaDA-Instruct (8B)에 집중. 다음 환경에서의 재현성 확인 필요:
- 더 큰 모델 (예: Mercury/Inception Labs, Gemini Diffusion/DeepMind)
- 다른 마스킹 전략을 사용하는 MDM 변형

#### ③ 하이브리드 접근의 탐구

AR과 임의 순서의 이분법을 넘어서:

- **동적 블록 크기 조정**: 포킹 토큰 주변에서는 $B=1$ (AR), 나머지는 $B>1$ (병렬)
- **엔트로피 인식 스케줄링(entropy-aware scheduling)**: 생성 중 실시간 엔트로피 모니터링으로 순서 전략 적응적 전환
- **JustGRPO-Fast의 고도화**: 고엔트로피 위치 탐지를 학습 가능한 모듈로 발전

#### ④ 양방향 정제(Bidirectional Refinement) 활용

JustGRPO 이후 모델이 이미 생성된 출력을 반복적으로 수정하는 능력을 활용하는 연구:
- CDLM (Zhang et al., 2025b)의 수정적 접근
- ParallelBench (Kang et al., 2026)의 양방향 맥락 설정과의 결합

#### ⑤ 보상 함수 설계의 세분화

현재 이진(binary) 보상을 사용하지만:
- 중간 추론 단계에 대한 **과정 보상(process reward)** 도입 가능성
- 포킹 토큰의 다양성을 직접 장려하는 **엔트로피 보너스(entropy bonus)** 설계

#### ⑥ 사전학습과 RL 훈련의 상호작용

Zhang et al. (2025a)의 "사전학습-중간학습-RL 상호작용" 연구와 연계하여:
- 사전학습 단계부터 AR 순서로 훈련된 dLLM이 RL에서 더 나은 출발점을 제공하는지 검토

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 dLLM 기반 연구 계보

```
Continuous Diffusion for Text (2020-2022)
├── DDPM (Ho et al., NeurIPS 2020) — 연속 도메인 확산 기반
├── Diffusion-LM (Li et al., NeurIPS 2022) — 임베딩 공간 적용
└── DiffuSeq (Gong et al., ICLR 2023) — Seq2Seq 생성

Discrete/Masked Diffusion (2024)
├── MDLM (Lou et al., ICML 2024) — 이산 확산 비율 추정
├── MDLM (Sahoo et al., NeurIPS 2024) — 단순화된 마스크 확산
└── SMDM (Shi et al., NeurIPS 2024) — 일반화된 마스크 확산

Large-scale dLLMs (2025)
├── LLaDA (Nie et al., NeurIPS 2025) — 8B 규모 마스크 확산 LM
├── Dream (Ye et al., 2025) — 7B 확산 LM
├── LLaDA 1.5 (Zhu et al., 2025) — 선호도 최적화 개선
└── Mercury (Inception Labs, 2025) — 초고속 추론

RL for dLLMs (2025-2026)
├── d1 (Zhao et al., NeurIPS 2025) — 최초 dLLM RL 확장
├── ESPO (Ou et al., ICLR 2026) — 시퀀스 수준 관점
├── GDPO (Rojas et al., ICLR 2026) — 그룹 확산 정책 최적화
├── SPG (Wang et al., ICLR 2026a) — 샌드위치 정책 그래디언트
├── LLaDOU (Huang et al., NeurIPS 2025b) — 보조 위치 선택 모듈
└── JustGRPO [본 논문, 2026] — 단순 AR GRPO
```

### 5.2 RL 방법론별 상세 비교

| 방법 | 핵심 아이디어 | 우도 계산 | 임의 순서 보존 | GSM8K (256) | 복잡도 |
|---|---|---|---|---|---|
| **d1** (Zhao et al., 2025) | 토큰 수준 GRPO 직접 적용 | ELBO 근사 | ✓ | 81.1 | 중 |
| **ESPO** (Ou et al., 2026) | 시퀀스 수준 관점, 원리적 RL | ELBO 근사 | ✓ | 82.3 | 높음 |
| **GDPO** (Rojas et al., 2026) | 그룹 확산 정책 최적화 | 근사 | ✓ | 82.8 | 높음 |
| **SPG** (Wang et al., 2026a) | 샌드위치 정책 그래디언트 | ELBO 근사 | ✓ | 86.1 | 높음 |
| **LLaDOU** (Huang et al., 2025b) | 위치 선택 보조 모듈 | 직접 추정 | ✓ | 88.1 | 매우 높음 |
| **JustGRPO** [본 논문] | AR scaffold + 표준 GRPO | **정확한 계산** | ✗ (훈련 시) | **89.1** | **낮음** |

### 5.3 Pass@k 관점에서의 탐색 능력 비교

| 방법 | Pass@k 스케일링 | 탐색 효율 | 비고 |
|---|---|---|---|
| Arbitrary Order (dLLM 기본) | **낮음** (flat curve) | 낮음 | 포킹 토큰 우회 |
| AR Order (dLLM에 AR 적용) | **높음** (steep curve) | 높음 | 포킹 토큰 대면 |
| JustGRPO (훈련 후) | 높음 유지 | 높음 | AR 탐색 + 병렬 추론 |

### 5.4 병렬 추론 연구와의 관계

병렬 디코딩을 다루는 연구들과의 위치:

| 연구 | 방향 | JustGRPO와의 관계 |
|---|---|---|
| Fast-dLLM (Wu et al., ICLR 2026b) | KV cache + 병렬 디코딩 | **상호 보완적**: JustGRPO 훈련 후 Fast-dLLM 추론 적용 가능 |
| EB-Sampler (Ben-Hamu et al., NeurIPS 2025) | 엔트로피 경계 언마스킹 | **직접 활용**: JustGRPO는 EB-Sampler와 호환 확인 |
| ParallelBench (Kang et al., ICLR 2026) | 병렬 디코딩 트레이드오프 분석 | 향후 JustGRPO 모델 평가에 활용 권장 |
| CDLM (Zhang et al., 2025b) | 수정적 확산 언어 모델 | 양방향 정제와 결합 가능성 |

### 5.5 기존 dLLM 추론 연구와의 차별점

| 연구 | 비자기회귀 이점 주장 | 본 논문의 입장 |
|---|---|---|
| Ye et al. (ICLR 2025a) | Sudoku, Zebra puzzle에서 비순차 생성 우위 | 특정 구조화 과제에는 동의, **일반 추론에는 반론** |
| Kim et al. (ICML 2025) | 최악 순서 훈련이 일반화 향상 | 훈련 시 구조의 중요성 공유, 방향성은 차이 |
| Du et al. (2025) | 균등 순열 학습이 데이터 분포 근사를 느슨하게 함 | **pre-training에서의 AR 우위** 동일하게 관찰 → 상호 지지 |

---

## 참고 자료

본 답변은 제공된 PDF 논문 원문을 기반으로 작성되었습니다:

**주요 참고 논문 (논문 내 인용 기준)**:

1. **Ni et al. (2026)** — "The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models" *(본 논문, arXiv:2601.15165v4)*
2. **Nie et al. (NeurIPS 2025)** — "Large Language Diffusion Models" (LLaDA)
3. **Zhao et al. (NeurIPS 2025)** — "d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning"
4. **Shao et al. (2024)** — "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (GRPO 원논문)
5. **Ou et al. (ICLR 2026)** — "Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective" (ESPO)
6. **Rojas et al. (ICLR 2026)** — "Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization" (GDPO)
7. **Wang et al. (ICLR 2026a)** — "SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models"
8. **Huang et al. (NeurIPS 2025b)** — "Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models" (LLaDOU)
9. **Ben-Hamu et al. (NeurIPS 2025)** — "Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking" (EB-Sampler)
10. **Ye et al. (ICLR 2025a)** — "Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning"
11. **Kim et al. (ICML 2025)** — "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions"
12. **Ho et al. (NeurIPS 2020)** — "Denoising Diffusion Probabilistic Models"
13. **Chen et al. (2021)** — "Evaluating Large Language Models Trained on Code" (Pass@k 지표)
14. **Yue et al. (NeurIPS 2025)** — "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"
15. **Wang et al. (NeurIPS 2025)** — "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning"
16. **Du et al. (2025)** — "Understanding the Limitations of Diffusion LLMs through a Probabilistic Perspective"
17. **Zhu et al. (2025)** — "LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models"
18. **Zhang et al. (2025b)** — "Corrective Diffusion Language Models" (CDLM)
19. **Kang et al. (ICLR 2026)** — "ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs"
20. **Schulman et al. (ICML 2015)** — "Trust Region Policy Optimization" (TRPO)
