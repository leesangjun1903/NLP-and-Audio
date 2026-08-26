# When Skills Meet Safety: Benchmarking and Characterizing the Adaptive Jailbreak Robustness of Skill-Merged LLMs

---

## 1. Executive Summary (10문장 이내)

본 논문은 **모델 병합(model merging)** 이 LLM에 새로운 기술을 부여하는 표준 방식이 된 현 상황에서, 기존 안전성 평가 방식인 **정적 거부 테스트(static refusal test)** 의 심각한 한계를 지적한다.  
저자들은 정적 안전성이 높은 모델이라도 적응형 공격(adaptive attack)에는 취약할 수 있음을 실증적으로 보인다.  
이를 위해 **SkillSafe-Bench** 라는 통제된 벤치마크를 설계하여, 정적 안전성·적응형 안전성·능력 유지를 동시에 평가한다.  
6개 베이스 모델(5개 패밀리, 2개 규모)에 걸쳐 실험한 결과, 정적 안전성이 낮은 ASR을 보이는 모델도 GCG·템플릿 공격에 60–76% 피탈(jailbreak)됨을 확인했다.  
Qwen과 Gemma 계열은 정적으로는 "안전"해 보이지만 적응형 공격에 취약(fragile)했고, Llama와 Phi-4는 강인(robust)했다.  
이 차이는 모델 패밀리 고유의 정렬 강도(base-conditional)에 기인함을 밝혔다.  
또한 태스크 벡터가 안전 부분공간(safety subspace) $\mathcal{S}$와 얼마나 겹치는지를 데이터 없이 계산하는 **기하학적 신호(geometric signal)** 를 제안하여 안전 침식 스킬을 사전 탐지한다.  
나아가 이 겹침 성분을 투영 제거하는 **SubSafe-Merge** 로 안전성을 복원하면서도 능력을 유지함을 보였다.  
결론적으로, 병합된 LLM의 안전성 평가에 적응형 평가는 선택이 아닌 필수임을 강조한다.

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **배경** | 오픈웨이트 LLM 생태계에서 task arithmetic, TIES, DARE 같은 모델 병합이 재훈련 없이 새 스킬을 추가하는 기본 방식으로 자리잡음 |
| **기존 문제** | 안전성 평가가 "정적 거부 테스트"에만 의존 → 고정된 유해 프롬프트를 한 번 제시하고 거부 여부만 측정 |
| **핵심 간과** | 안전 정렬(safety alignment)은 "얕음(shallow)" — 거부 신호가 초반 몇 토큰에 집중 (Qi et al., 2025). 적응형 공격은 이 얕은 층을 우회 가능 |
| **필요성** | 정적으로 "안전해 보이는" 병합 모델이 실제 공격에 얼마나 취약한지 측정하는 통제된 프레임워크 부재 |
| **목적** | 적응형 견고성까지 포괄하는 벤치마크 설계, 안전 침식 메커니즘 규명, 데이터 없는 사전 탐지·수정 방법 제안 |

> 💡 **용어 설명**
> - **모델 병합(Model Merging)**: 여러 파인튜닝된 모델의 가중치를 가중합 등으로 합치는 기법. 재학습 없이 새 능력을 추가할 수 있어 효율적
> - **정적 거부 테스트(Static Refusal Test)**: 고정된 유해 프롬프트를 그대로 제시하고 모델이 거부하는지만 확인하는 평가 방식
> - **안전 정렬의 얕음(Shallow Alignment)**: 안전 거부 행동이 생성 초반 몇 토큰에 주로 인코딩되어 있어, 조금만 우회해도 무력화되는 현상

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|------|
| A | 정적 안전성은 적응형 견고성을 예측하지 못함 | Qwen은 정적 ASR ≤0.12인데 GCG로 0.28–0.48까지 상승; Llama는 동일 정적 수준에서 0.06–0.12로 유지 | Section 5, Table 1, Fig. 2b |
| B | 병합의 안전 비용은 베이스 모델에 조건부 | 강정렬 베이스(Qwen, Llama)는 수학 스킬 병합 시 정적 ASR이 베이스 이하로 떨어지나, 약정렬 Mistral은 어떤 스킬이든 ASR 상승 | Section 5, Fig. 2a, Table S1 |
| C | 기하학적 신호(서브스페이스 겹침)로 안전 침식 스킬 사전 탐지 가능 | uncensored 벡터의 $\mathcal{S}$ 겹침 ≈0.99; math/code 벡터 ≈0.001 (3 orders of magnitude 차이) | Section 6, Table 2 |
| D | SubSafe-Merge는 능력 유지하며 안전 침식 제거 | Qwen: static ASR 0.46→0.18, adaptive 0.54→0.36, GSM8K 0.80→0.80 | Section 7, Table 3 |
| E | 겹침 신호는 same-recipe abliteration에만 신뢰 가능 | SFT/DPO 무검열 모델(Orion): 겹침 0.001이나 실제 위험(static ASR 0.63); 크로스 레시피에서 오탐 | Section 7, Table S4, S6 |
| F | 취약/강인 순서는 공격 종류와 무관하게 불변 | GCG, 템플릿(best-of-6), PAIR 3가지 공격 모두 Qwen ≥ Llama 순서 유지 | Section 5 |

### 2-1. 해결 문제·제안 방법·모델 구조·성능·한계 상세 설명

#### ① 해결하고자 하는 문제

- 기존 안전 평가는 정적 테스트에만 의존 → 실제 공격자는 적응형 공격 사용
- 병합 방법론 연구들이 안전성을 부수적 지표로만 측정
- 어떤 스킬 벡터가 안전을 침식하는지 병합 전에 알 수 없음

#### ② 제안하는 방법과 수식

**[수식 1] 일반적 모델 병합:**

$$\theta_{\text{merge}} = \theta_{\text{base}} + \sum_i \lambda_i f(\tau_i)$$

| 기호 | 설명 |
|------|------|
| $\theta_{\text{base}}$ | 안전 정렬된 베이스 모델의 파라미터 |
| $\theta_i$ | 스킬 $i$에 파인튜닝된 모델의 파라미터 |
| $\tau_i = \theta_i - \theta_{\text{base}}$ | 태스크 벡터 (스킬의 가중치 변화량) |
| $\lambda_i$ | 병합 계수 (스케일링 팩터) |
| $f(\cdot)$ | 병합 방법별 함수 (Task Arithmetic: identity; TIES: trim-and-sign-elect; DARE: drop-and-rescale; Soup: averaging) |

> 💡 **용어 설명**
> - **태스크 벡터(Task Vector)**: 파인튜닝 후 모델에서 베이스 모델 파라미터를 뺀 값 $\tau = \theta_{\text{ft}} - \theta_{\text{base}}$. 특정 능력의 "방향"을 담음
> - **TIES-Merging**: 여러 태스크 벡터의 간섭을 줄이기 위해 작은 업데이트를 제거(trim)하고 부호 갈등을 해소하는 병합 방식
> - **DARE**: 델타 파라미터를 희소화(sparsify)하고 재스케일하는 병합 방식

**[수식 2] 안전 서브스페이스 추정:**

$$\tau_{\text{safe}}^{(\ell)} = \theta_{\text{base}}^{(\ell)} - \theta_{\text{unsafe}}^{(\ell)}$$

$$\mathcal{S}^{(\ell)} = \text{span}\left(\text{top-}k \text{ left singular vectors of } \tau_{\text{safe}}^{(\ell)}\right)$$

| 기호 | 설명 |
|------|------|
| $\theta_{\text{unsafe}}$ | abliterated(거부 방향 제거) 버전의 베이스 모델 파라미터 |
| $\ell$ | 레이어 인덱스 |
| $\mathcal{S}^{(\ell)}$ | 레이어 $\ell$의 안전 부분공간 |
| $P_{\mathcal{S}}$ | $\mathcal{S}$로의 정사영 연산자 |
| $P_{\mathcal{S}^\perp}$ | $\mathcal{S}$의 직교 보공간으로의 정사영 연산자 |
| $k$ | 사용하는 특이벡터 수 (실험에서 주로 $k=8$, $k=1$도 동일 결과) |

> 💡 **용어 설명**
> - **Abliteration(어블리터레이션)**: 모델에서 특정 방향(여기서는 거부 방향)을 활성화 공간에서 제거하는 기법. 결과적으로 모델이 거부하지 않게 됨
> - **특이값 분해(SVD, Singular Value Decomposition)**: 행렬을 $U\Sigma V^T$로 분해. 여기서 $U$의 상위 열벡터들이 가장 중요한 방향(안전 부분공간)을 형성
> - **정사영(Projection)**: 벡터를 특정 부분공간에 수직으로 내린 그림자. $P_{\mathcal{S}}(\mathbf{v})$는 $\mathbf{v}$에서 $\mathcal{S}$ 방향 성분만 추출

**[수식 3] SubSafe-Merge:**

$$\theta_{\text{merge}} = \theta_{\text{base}} + \sum_i \lambda_i P_{\mathcal{S}^\perp}\!\left(f(\tau_i)\right)$$

| 기호 | 설명 |
|------|------|
| $P_{\mathcal{S}^\perp}(f(\tau_i))$ | 병합 처리된 태스크 벡터에서 안전 서브스페이스 성분을 제거한 나머지 |

> 💡 **용어 설명**
> - **SubSafe-Merge**: 태스크 벡터에서 안전 서브스페이스와 겹치는 성분을 투영 제거한 뒤 병합하는 방법. 능력(직교 성분)은 보존하면서 안전 침식(서브스페이스 내 성분)만 제거

**[수식 4] 서브스페이스 겹침(Subspace Overlap) 측정:**

$$\text{overlap}(\tau_i) = \frac{\|P_{\mathcal{S}}(\tau_i)\|_F^2}{\|\tau_i\|_F^2}$$

| 기호 | 설명 |
|------|------|
| $\|\cdot\|_F$ | 프로베니우스 노름 (행렬 원소의 제곱합의 제곱근) |
| $\text{overlap}(\tau_i)$ | 태스크 벡터 에너지 중 안전 서브스페이스 내에 있는 비율 |

#### ③ 모델 구조

| 컴포넌트 | 상세 |
|----------|------|
| **베이스 모델** | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3 (주요 3개) + Qwen2.5-14B, Phi-4-mini, Gemma-2-9B (확장 3개) |
| **스킬 벡터** | Math LoRA (MetaMathQA), Uncensored/Abliterated 모델, Code, Medicine, Finance, Law |
| **병합 방법** | Task Arithmetic, Linear (Soup), TIES, DARE-TIES |
| **평가 공격** | GCG (100/500 steps), Best-of-6 Template, PAIR, Crescendo (multi-turn) |
| **판정기(Judge)** | HarmBench classifier + Llama Guard 3 (AND 규칙: 둘 다 unsafe여야 성공으로 판정) |

#### ④ 성능 향상

| 설정 | 지표 | Plain Merge | SubSafe-Merge | 개선 |
|------|------|-------------|---------------|------|
| Qwen + uncensored ($\lambda$=0.6) | Static ASR | 0.460 | 0.182 | **-27.8pp** |
| Qwen + uncensored ($\lambda$=0.6) | GCG ASR | 0.540 | 0.360 | **-18pp** |
| Qwen + uncensored ($\lambda$=0.6) | GSM8K | 0.80 | 0.80 | **유지** |
| Llama + uncensored ($\lambda$=0.6) | Static ASR | 0.307 | 0.180 | **-12.7pp** |
| Llama + uncensored ($\lambda$=0.6) | GCG ASR | 0.340 | 0.160 | **-18pp** |
| Llama + uncensored ($\lambda$=0.6) | GSM8K | 0.69 | 0.71 | **+2pp** |

#### ⑤ 한계

| # | 한계 | 세부 설명 |
|---|------|-----------|
| L1 | 안전 서브스페이스 추정의 의존성 | abliterated 모델의 품질에 따라 $\mathcal{S}$ 품질이 결정됨 |
| L2 | 적응형 ASR은 하한값 | 고정 예산 공격이므로 더 강한 공격자는 절댓값을 높일 수 있음 |
| L3 | 실험 규모 제한 | 7–14B 모델, 2개 주요 스킬만 완전 실험; 더 큰 모델과 다양한 스킬은 미래 과제 |
| L4 | 판정기 불완전성 | AND 규칙에서도 Cohen's κ=0.66, 잔류 오류 존재 |
| L5 | S-외부 침식 탐지 불가 | SFT/DPO 방식 무검열화는 겹침 ≈0.001임에도 실제 위험 → 완전한 false negative |
| L6 | 취약성 메커니즘 불명 | 안전 서브스페이스의 기하학적 특성(유효 랭크, 특이값 갭)이 적응형 취약성을 예측하지 못함 |

---

## 3. 각 주장의 위치 표시

| 주장 | 페이지/위치 |
|------|------------|
| 정적 안전성이 적응형 견고성 예측 실패 | p.6 (Section 5), **Table 1**, **Fig. 2b** |
| 스킬-안전성 효과가 베이스 조건부 | p.5–6 (Section 5), **Fig. 2a**, **Table S1** |
| 안전 정렬의 얕음(shallow alignment) 근거 | p.2 (Introduction), p.3 (Related Work) — Qi et al. 2025 인용 |
| 기하학적 신호(겹침) 제안 | p.7–8 (Section 6), **Table 2** |
| SubSafe-Merge 결과 | p.9–10 (Section 7), **Table 3**, **Table S8** |
| 6개 베이스 교차 검증 | p.7 (Section 5), **Table S2**, **Table S3** |
| 겹침 신호의 한계(same-recipe만 탐지) | p.9–10 (Section 7), **Table S4**, **Table S6** |
| 판정기 신뢰성 (Cohen's κ) | p.5 (Section 4) |
| SafeMERGE 비교 | p.10 (Section 7), **Table S8** |
| 통계 검정 (McNemar, bootstrap CI) | p.6 (Section 5), Appendix S5 |

---

## 4. 저자 직접 보고 vs. 독자 해석 분리

### 4-1. 저자가 직접 보고한 결과

**연구 주제:**
> "스킬 병합된 LLM의 적응형 탈옥 견고성이 정적 거부 테스트로 예측 불가능함을 benchmarking으로 규명" (Abstract, p.1)

**방법 (수식 포함):**
- 병합: $\theta_{\text{merge}} = \theta_{\text{base}} + \sum_i \lambda_i f(\tau_i)$ (Eq. 1, p.4)
- SubSafe-Merge: $\theta_{\text{merge}} = \theta_{\text{base}} + \sum_i \lambda_i P_{\mathcal{S}^\perp}(f(\tau_i))$ (Eq. 2, p.9)
- 두 판정기 AND 규칙 (Cohen's κ = 0.66, p.5)
- McNemar 검정: Qwen math merge $\lambda$=0.6에서 $p < 0.008$ (p.6, S5)

**결과 (저자 직접 보고):**
- Qwen + math merge: static ASR 0.10, GCG ASR 0.28–0.38 (gap +20pp, Table 1)
- Llama + math merge: static ASR ≤0.08, GCG ASR ≤0.12 (gap ≈+3pp, CI 포함 zero, Table 1)
- 겹침: uncensored 0.993, math 0.0012, code 0.0012 (Table 2)
- SubSafe-Merge: Qwen static 0.46→0.18, GCG 0.54→0.36, GSM8K 0.80→0.80 (Table 3)

### 4-2. 독자(AI 연구원)의 해석

1. **Qwen-Llama 분리 현상의 함의**: 두 모델이 동일한 정적 ASR임에도 적응형 공격 반응이 3배 이상 차이나는 것은, 안전 정렬의 "깊이"가 모델 패밀리별 아키텍처 및 훈련 절차에 크게 의존함을 시사한다. 이는 단순히 RLHF나 DPO 적용 여부가 아니라, 정렬이 가중치 공간에서 얼마나 분산되어 있는지(distributional robustness)의 차이로 해석할 수 있다.

2. **SubSafe-Merge의 실용적 한계 재해석**: 논문은 S-외부 침식을 "한계"로 분류했지만, 실제 배포 환경에서는 SFT/DPO 방식의 무검열화(예: Orion, overlap 0.001이나 static ASR 0.63)가 더 흔한 공격 벡터일 수 있다. 따라서 SubSafe-Merge의 실제 보호 범위는 논문이 제시하는 것보다 좁을 가능성이 높다.

3. **기하학적 신호의 레시피 의존성 문제**: 겹침 신호가 same-recipe abliteration만 탐지한다는 것은, 공격자가 간단히 다른 abliteration 레시피를 사용하면 탐지를 우회할 수 있음을 의미한다. 이는 보안 도구로서의 실용성을 크게 제한한다.

4. **Gemma의 취약성 원인**: Gemma는 unmerged 상태에서도 template ASR이 0.64로, 병합이 취약성을 "생성"한 것이 아니라 "상속"했다는 저자 해석이 타당하다. 그러나 이는 동시에 정적 평가(static ASR 0.02)가 베이스 모델 자체의 적응형 취약성을 완전히 놓친다는 더 심각한 문제를 드러낸다.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적으로 취약한 부분

| 항목 | 문제점 | 위치 |
|------|--------|------|
| **적응형 평가 샘플 크기** | $n=50$ behaviors만 사용. Wilson 95% CI가 매우 넓음 (예: [0.17, 0.42]). 일부 간격이 0 포함 | Table 1 |
| **다중 비교 미보정** | 21개 셀에 대해 McNemar 검정을 반복 수행했으나 FWER/FDR 보정 없음. 저자 스스로 "we do not correct for multiplicity" 인정 | p.6 |
| **SafeMERGE vs SubSafe GCG 차이** | McNemar $p=0.24$ (비유의), bootstrap CI $[-0.04, 0.28]$. 저자가 SafeMERGE의 낮은 GCG 수치를 "아티팩트"로 해석하는 것은 합리적이지만 통계적으로 불확실 | Section 7, Table S8 |
| **leave-one-base-out Spearman** | 3 bases × 2 skills라는 소규모에서 순열 귀무가설과 구별 불가. 저자 스스로 "not distinguishable from a permutation null" 인정 | Section 6 |
| **Crescendo 절댓값 비교 불가** | 공격자 모델을 3B 소형으로 약화, 휴리스틱 판정기 사용 → 논문의 다른 attack family와 절댓값 비교 불가 | Appendix S5 |
| **초기 first-N 서브셋 편향** | 초기 분석에서 gap ≈25pp였다가 최종 수정 후 17pp로 감소. 전처리 과정의 투명성 우려 | p.7 |

### ⚠️ 비교 불가능한 수치

| 수치 쌍 | 비교 불가 이유 |
|---------|---------------|
| GCG ASR (Gemma) vs. GCG ASR (Qwen/Llama) | GCG가 Gemma에 under-transfer → 절댓값 비교 시 Gemma의 취약성을 과소평가 | Table S2 |
| Crescendo ASR vs. GCG/Template ASR | 공격 강도가 다름 (3B attacker, heuristic judge) |
| Copyright 행동의 ASR | 내부 LLM-classifier 기반 측정 vs. HarmBench 공식 MinHash 기반 — 시스템이 달라 직접 비교 불가 | Appendix S5 |
| 6개 베이스의 GCG 순위 | GCG under-transfer가 베이스마다 비균일하여 cross-family GCG 수치로 모델 순위를 매길 수 없음 | p.7 |

---

## 6. 논문이 답하지 않는 질문

| # | 미해결 질문 | 근거 |
|---|------------|------|
| Q1 | **Qwen은 왜 Llama보다 적응형 공격에 취약한가?** 아키텍처, 훈련 데이터, RLHF 방법 중 어느 요인이 결정적인가? | Section 8 Limitation 5: "we characterize the phenomenon rather than claim a mechanism" |
| Q2 | **SubSafe-Merge가 S-외부 침식(SFT/DPO 무검열화)을 어떻게 처리할 수 있는가?** 현재 false negative 문제 미해결 | Section 7 |
| Q3 | **모델 크기를 키우면(예: 70B+) 결과가 달라지는가?** 14B까지만 실험, 대형 모델은 미래 과제 | Section 8 Limitation 3 |
| Q4 | **다수의 스킬을 동시에 병합할 때 안전성 침식이 어떻게 상호작용하는가?** 현재 1–2개 스킬만 실험 | Section 4 |
| Q5 | **SubSafe-Merge가 데이터 의존적 방법(Thakkar et al., Wu et al.)보다 우월한가?** head-to-head 비교 미실행 | Section 7 |
| Q6 | **멀티모달 또는 에이전트 시나리오에서도 동일한 취약성 패턴이 나타나는가?** 텍스트 모달만 대상 | Section 3 Threat Model |
| Q7 | **안전 서브스페이스 $\mathcal{S}$의 유효 랭크가 적응형 취약성과 무관한 이유는?** 현상 기술에 그침 | Table S3, Section 8 |
| Q8 | **겹침 신호 임계값(threshold)을 어떻게 설정해야 실용적 스크리닝이 가능한가?** 연속 예측 검증 실패 | Section 6 |
| Q9 | **방어자가 adaptive ASR을 실용적 비용으로 측정하려면 어떤 공격 예산이 필요한가?** 최소 필요 예산 분석 없음 | Section 4 |
| Q10 | **병합이 안전성을 향상시키는 조건(Gallego 2024, Zeng et al. 2025)은 무엇인가?** SkillSafe-Bench에서 미탐구 | Section 2 |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2): 전체 프레임워크 개요

```
[정적 테스트] → "안전" 판정 (오해)
[적응형 공격] → 실제 취약성 노출
[기하학적 신호] → 병합 전 위험 탐지
[SubSafe-Merge] → 안전 복원
```

**해석:** 논문 전체의 논리 흐름을 한눈에 보여주는 그림이다. 핵심 메시지는 정적 안전성과 적응형 견고성이 "다른 것"이라는 점이다. 왼쪽 상단 Path A(정적)는 "오해의 소지가 있는 안전"으로, Path B(적응형)는 "진짜 취약성"으로 연결된다. 오른쪽에는 uncensored 벡터($\tau_{\text{uncensored}}$)가 $\mathcal{S}$ 안에 놓이고, math 벡터($\tau_{\text{math}}$)가 $\mathcal{S}$에 직교함을 기하학적으로 시각화한다. 이 직교성이 SubSafe-Merge의 원리적 근거가 된다.

> 💡 **용어 설명**: **직교(Orthogonal)**: 두 벡터가 서로 수직인 관계. 내적이 0. 수학 스킬 벡터가 안전 부분공간에 직교한다는 것은, 수학 능력이 안전성 방향과 무관하다는 의미

---

### Figure 2a (p.6): 스킬-안전성 효과의 베이스 조건부성

**해석:** x축은 병합 계수 $\lambda$, y축은 정적 ASR(400 behaviors, AND rule). 세 가지 베이스(Qwen/Llama = 실선/점선, Mistral = 파선)와 두 스킬(math = 상승/하강, uncensored = 단조 상승)의 교차 패턴이 핵심이다.

- **Qwen/Llama + math**: 모든 $\lambda$에서 베이스 이하로 하강 (최저 0.07–0.09). 이는 직관에 반하는 현상으로, 저자들은 저작권 관련 판정기 아티팩트로 일부 설명
- **Qwen/Llama + uncensored**: $\lambda$에 비례해 단조 상승 (0.6까지 0.46–0.31)
- **Mistral**: 두 스킬 모두 베이스(0.48)보다 높아, 약정렬 베이스는 어떤 병합도 안전성을 훼손

**핵심 시사점**: 병합의 안전성 비용은 task vector의 특성 + 베이스의 정렬 강도의 함수. 알고리즘(4가지 방법 모두 0.035 이내)의 영향은 미미

---

### Figure 2b (p.6): 정적 안전성 ≠ 적응형 견고성 (가장 중요한 그림)

**해석:** x축은 정적 ASR(random 50 behaviors), y축은 적응형 GCG ASR. 대각선($y=x$)은 "정적 = 적응형"인 경우.

- **대각선 위**: 정적으로 안전해 보이지만 적응형으로 취약 (위험한 과신)
- **대각선 아래**: 정적보다 적응형이 낮음 (드물고 대부분 아티팩트)

**Qwen(원형)**: 낮은 정적 ASR(0.10–0.24)에도 GCG ASR 0.28–0.48로 대각선 훨씬 위. "정적-안전하지만 적응형-취약"

**Llama(삼각형)**: 동일하게 낮은 정적 ASR에서 GCG ASR도 낮음(≤0.12). 거의 대각선에 위치.

**Mistral(사각형)**: 높은 정적 ASR이 그대로 높은 적응형 ASR로 이어짐. 대각선 근방.

이 하나의 산점도가 논문의 핵심 주장 전체를 입증한다.

---

### Table 1 (p.8): 정적 vs. GCG 적응형 ASR 완전 비교

**해석:** 3개 베이스 × 2 스킬 × 3 $\lambda$ 값의 정적/GCG/Gap/95% CI를 체계적으로 제시.

주목할 행:

| 행 | 정적 | GCG | Gap | 해석 |
|----|------|-----|-----|------|
| Qwen base | 0.240 | 0.480 | +0.24 | 베이스 자체가 이미 취약 |
| Qwen +math λ=0.6 | 0.100 | 0.280 | +0.18 | 가장 "안전해 보이나" 실제 위험 |
| Llama +math λ=0.6 | 0.060 | 0.120 | +0.06 | CI 포함 zero → 진정 강인 |
| Mistral base | 0.500 | 0.500 | +0.00 | 정적이 이미 높아 gap 없음 |

Llama의 모든 CI가 zero를 포함한다는 점이 "통계적으로 강인"의 근거이고, Qwen의 math merge들은 CI 하한이 0을 훨씬 초과한다.

---

### Table 3 (p.10): SubSafe-Merge 결과

**해석:** SubSafe-Merge의 효과를 plain merge, base와 3-way 비교.

```
Qwen: 0.460(plain) → 0.182(SubSafe) ← 0.195(base)  [정적 복원]
      0.540(plain) → 0.360(SubSafe) ← 0.480(base)   [적응형 부분 복원]
      GSM8K: 0.80 = 0.80 (능력 완전 유지)

Mistral: 0.590(plain) → 0.480(SubSafe) ≈ 0.483(base) [정적 복원]
         0.680(plain) → 0.660(SubSafe) >> 0.500(base) [적응형 복원 불가]
```

Mistral에서 적응형 ASR이 베이스보다 높게 유지되는 이유: "모델은 자신의 베이스보다 더 강인해질 수 없다"는 원리. Mistral 베이스 자체가 GCG ASR 0.50으로 취약하기 때문. SubSafe-Merge는 병합이 **추가한** 침식만 제거하지, 베이스의 **선천적** 취약성은 고칠 수 없다.

---

## 8. 결론 — 시사점, 후속 계획, 추가 방향

### 8-1. 저자들이 제시한 시사점과 후속 계획

**시사점:**
1. 현재 안전 병합 평가들이 위험을 과소보고: 낮은 정적 ASR은 안전한 병합의 증거가 아님
2. 적응형 평가는 병합 LLM의 안전 인증에 필수
3. 겹침 스크린의 장점은 직접 공격 테스트보다 데이터 없이 사용 가능하고 탐지와 수정이 연동된다는 점
4. SkillSafe-Bench 하니스(harness)를 공개하여 커뮤니티가 표준으로 사용 가능케 함

**저자 제시 후속 연구:**
- 더 많은 스킬과 14B 초과 대형 모델로 확장
- SubSafe-Merge와 데이터 의존적 방법(Thakkar et al., Wu et al.)의 head-to-head 비교
- 겹침 특성의 정량적 예측 검증 (더 많은 스킬/베이스 다양성 필요)
- 비-텍스트 모달(멀티모달, 에이전트) 확장

### 8-1 (심화). 모델의 일반화 성능 향상 가능성

본 논문의 SubSafe-Merge가 일반화 성능에 미치는 영향을 분석하면:

**긍정적 측면:**
- 태스크 벡터의 직교 성분( $P_{\mathcal{S}^\perp}(\tau_i)$ )을 보존함으로써 스킬 능력을 유지 (GSM8K, MMLU 변동 ≤0.02)
- $k$에 대한 민감도가 낮아 실용적 안정성 높음 ($k=1$과 $k=8$ 동일 결과)

**일반화 제한 요인:**

$$\text{수정 범위} \propto \text{overlap}(\tau_{\text{donor}})$$

즉, 수정 효과가 same-recipe abliteration에 국한되어 일반화 범위가 좁다. 보다 일반적인 일반화를 위해서는:

1. **다중 안전 서브스페이스 구성**: 여러 abliteration 레시피에서 추출한 $\mathcal{S}_1, \mathcal{S}_2, \ldots, \mathcal{S}_m$의 합집합 부분공간을 사용:

$$\mathcal{S}_{\text{union}} = \mathcal{S}_1 \cup \mathcal{S}_2 \cup \cdots \cup \mathcal{S}_m$$

2. **동적 서브스페이스 학습**: abliterated 모델 대신, 다양한 유해 응답 패턴으로부터 데이터 기반으로 $\mathcal{S}$를 학습

3. **적응형 겹침 임계값**: 고정 임계값 대신 베이스 모델의 적응형 취약성 프로파일에 맞춘 임계값 설정

**베이스 조건부 일반화**: 동일 SubSafe-Merge라도 Qwen(적응형 취약)과 Llama(적응형 강인)에서 효과가 다름. 이는 일반화가 "베이스 모델의 적응형 견고성 × 태스크 벡터의 S-겹침"의 곱으로 결정됨을 시사한다. 강인한 베이스에서는 SubSafe-Merge 없이도 안전 유지; 취약한 베이스에서는 SubSafe-Merge가 static은 복원하나 adaptive는 베이스 상한에 묶인다.

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법 | 평가 방식 | 본 논문 대비 차별점 |
|------|------|------|----------|-------------------|
| Ilharco et al. (Task Arithmetic) | 2023 | 태스크 벡터 덧셈 | 능력 평가만 | 안전성 미고려 |
| Hammoud et al. | 2024 | 병합 안전성 탐구 | **정적 Llama Guard** | 본 논문: 적응형 추가, 팩토리얼 설계 |
| Lermen et al. (LoRA Jailbreak) | 2023 | LoRA 파인튜닝으로 안전 훈련 제거 | 정적 | 파인튜닝 vs. 병합의 다른 시나리오 |
| Arditi et al. (Refusal Direction) | 2024 | 거부가 1D 방향에 집중됨 발견 | 활성화 공간 분석 | 본 논문: 가중치 공간에서 이를 활용 |
| Hsu et al. (Safe LoRA) | 2024 | LoRA 업데이트를 안전 서브스페이스에 투영 | 정적 | 본 논문: 병합에 적용, 역투영($P_{\mathcal{S}^\perp}$) |
| Djuhera et al. (SafeMERGE) | 2026 | 안전 레이어를 선택적으로 병합 | 정적 | 본 논문: head-to-head 비교, 적응형 평가 추가 |
| Zou et al. (GCG) | 2023 | 적대적 서픽스 gradient 탐색 | 단일 모델 | 본 논문: 병합 모델에 체계적 적용 |
| Andriushchenko et al. | 2025 | 간단한 적응형 공격이 정적 안전 모델 파탈 | 단일 모델 | 본 논문: 병합 맥락으로 확장 검증 |
| Wei et al. | 2024 | 안전 파라미터가 희소하고 유틸리티와 분리 | 가지치기/수정 | 본 논문 $\mathcal{S}$ 설계의 이론적 근거 |

**본 논문이 앞으로의 연구에 미치는 영향:**

1. **평가 표준의 전환**: 모델 병합 안전성 연구에서 정적 ASR이 아닌 적응형 ASR을 표준 지표로 채택하게 하는 촉매 역할. SkillSafe-Bench의 공개로 재현 가능한 비교 기반 마련

2. **안전 서브스페이스 패러다임 확산**: 가중치 공간의 기하학적 분석을 통해 병합 전 위험 스크리닝이라는 새로운 연구 방향 제시. Safe LoRA의 아이디어를 병합으로 확장

3. **베이스 모델 선택 기준**: "정적 ASR이 낮은 모델"이 아닌 "적응형 공격에 강인한 모델"을 베이스로 선택해야 한다는 새로운 체크리스트 제시

4. **침식 메커니즘 연구 자극**: Qwen-Llama 분리 현상의 원인 규명을 위한 아키텍처 수준 분석 필요 → 정렬 훈련의 깊이와 분산도 연구 촉발 예상

**앞으로 연구 시 고려할 점:**

1. **평가 다양성**: 단일 공격(GCG 100 steps)은 하한값이므로, 복수 공격의 앙상블로 적응형 ASR 상한을 추정하는 방법 필요

2. **판정기 신뢰성**: Cohen's κ = 0.66은 moderate agreement 수준. 더 신뢰할 수 있는 human-calibrated 판정기 또는 LLM-as-judge 파이프라인 구축 필요

3. **크로스 레시피 $\mathcal{S}$ 추정**: 단일 abliteration 소스 의존에서 벗어나, 여러 소스를 앙상블하거나 데이터 기반으로 $\mathcal{S}$를 구성하는 방법 탐구

4. **대형 모델 확장성**: 70B+ 모델에서 동일한 기하학적 특성이 유지되는지, 그리고 SubSafe-Merge의 계산 비용이 실용적인지 검증 필요

5. **동적 위협 모델**: 공격자가 SubSafe-Merge를 알고 대응하는 적응형 위협 모델(adversarial meta-attack)에 대한 robustness 분석

6. **능력-안전성 Pareto 최적화**: 단순 서브스페이스 투영 외에, 능력-안전성 트레이드오프를 레이어별·헤드별로 세밀하게 최적화하는 방향

---

## 참고 자료

본 분석은 다음 논문 및 자료에 기반합니다:

1. **Yu Ma et al. (2026)** — "When Skills Meet Safety: Benchmarking and Characterizing the Adaptive Jailbreak Robustness of Skill-Merged LLMs" (arXiv:2608.08542v1)

2. **Ilharco et al. (2023)** — "Editing Models with Task Arithmetic" (ICLR) (arXiv:2212.04089)

3. **Yadav et al. (2023)** — "TIES-Merging: Resolving Interference When Merging Models" (NeurIPS) (arXiv:2306.01708)

4. **Yu et al. (2024)** — "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch" (ICML) (arXiv:2311.03099)

5. **Zou et al. (2023)** — "Universal and Transferable Adversarial Attacks on Aligned Language Models" (arXiv:2307.15043)

6. **Qi et al. (2025)** — "Safety Alignment Should Be Made More Than Just a Few Tokens Deep" (ICLR) (arXiv:2406.05946)

7. **Arditi et al. (2024)** — "Refusal in Language Models Is Mediated by a Single Direction" (NeurIPS) (arXiv:2406.11717)

8. **Hsu et al. (2024)** — "Safe LoRA: The Silver Lining of Reducing Safety Risks when Finetuning Large Language Models" (NeurIPS) (arXiv:2405.16833)

9. **Mazeika et al. (2024)** — "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal" (ICML) (arXiv:2402.04249)

10. **Hammoud et al. (2024)** — "Model Merging and Safety Alignment: One Bad Model Spoils the Bunch" (EMNLP) (arXiv:2406.14563)

11. **Andriushchenko et al. (2025)** — "Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks" (ICLR) (arXiv:2404.02151)

12. **Tramer et al. (2020)** — "On Adaptive Attacks to Adversarial Example Defenses" (NeurIPS) (arXiv:2002.08347)

> ⚠️ **정확도 고지**: 본 분석은 제공된 PDF 전문에 기반하며, 논문이 arXiv:2608.08542v1 (2026년 8월 게재)임을 명시합니다. 일부 미래 연구 비교(8-2절)는 논문 내 인용 목록과 AI 연구 커뮤니티의 공개된 연구 동향에 기반하며, 논문 출판 이후 발표된 연구에 대한 분석은 포함되지 않습니다.
