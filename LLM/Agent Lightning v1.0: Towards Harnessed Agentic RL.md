# Agent Lightning v1.0: Towards Harnessed Agentic RL

> **⚠️ 정확도 안내**: 본 분석은 제공된 PDF 원문에 근거합니다. 원문에 명시되지 않은 내용은 "[원문 미기재]"로 표시하며, 추론이 포함된 경우 명시적으로 구분합니다.

---

## 1. Executive Summary (10문장 이내)

Agent Lightning v1.0은 **Harnessed Agentic RL**이라는 새로운 RL 패러다임을 정의하고, 그에 따른 기술적 도전 과제를 체계적으로 규명한 연구이다.  
현대 AI 에이전트는 독립적인 LLM이 아니라, 도구·컨텍스트·제어 흐름을 관리하는 **에이전트 하네스(agent harness)** 안에서 실행된다.  
기존 RL 프레임워크들은 훈련 엔진이 환경 상호작용 루프를 소유하는 구조였으나, Harnessed Agentic RL에서는 하네스가 이 루프를 소유하고 훈련 엔진은 LLM 요청-응답 쌍만 관찰한다.  
이 구조 전환은 **재토크나이제이션(retokenization), 샘플 병합(sample merging), 어드밴티지 계산(advantage calculation), 손실 정규화(loss normalization), 훈련 백엔드 스케줄링**이라는 5가지 핵심 과제를 야기한다.  
저자들은 이 과제들을 처음으로 체계적으로 규명하고, 약 3,500줄의 경량 프레임워크인 Agent Lightning v1.0을 제안한다.  
시스템은 API Gateway, Rollout Controller, Customized Trainer 세 컴포넌트로 구성되며, Kubernetes 클러스터 위에서 임의의 에이전트 하네스를 지원한다.  
**Collocated Async RL**을 도입하여 동기 RL 대비 약 2배의 속도 향상을 GPU 추가 없이 달성하였다.  
코딩 에이전트 실험에서는 SWE-bench Verified 기준 Qwen3.5-9B를 41.8% → 56.4%로 향상(+14.6%p)시켰다.  
전체 코드, 데이터 파이프라인, 훈련 스크립트를 오픈소스로 공개하여 재현 가능한 연구를 지원한다.

### 1-1. 연구의 목적과 필요성

**목적**: 배포 시 사용하는 에이전트 하네스를 RL 훈련에 직접 통합하는 *Harnessed Agentic RL* 패러다임을 정립하고, 이 패러다임에서 발생하는 고유한 기술적 문제들을 해결하는 경량 프레임워크를 제공한다.

**필요성**:
- 기존 RL 프레임워크(verl, AReaL, slime)는 에이전트 루프를 훈련 프레임워크 내부에 직접 구현해야 해서, 독립적으로 유지되는 **mini-SWE-agent, OpenHands, Claude Code** 등의 하네스를 통합하기 어렵다 (Section 1, p.2).
- 하네스가 훈련 루프를 소유하는 구조에서 발생하는 문제들이 기존 프레임워크에서 **미명시(underspecified)** 상태로 방치되어 있어 훈련 불안정성을 초래한다 (Abstract, p.1).
- 코딩 에이전트 훈련을 위한 **완전하고 재현 가능한 파이프라인**이 부재하다 (Section 4, p.11).

> 💡 **에이전트 하네스(Agent Harness)**: LLM을 감싸서 도구 사용, 컨텍스트 관리, 오류 복구 등을 담당하는 실행 환경. LLM 자체가 아닌 "LLM을 이용하는 소프트웨어 시스템"이라고 이해하면 쉽다. 예) mini-SWE-agent는 GitHub 이슈를 해결하기 위해 코드 수정, 테스트 실행 등을 관리하는 하네스이다.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|-------|
| 1 | Harnessed Agentic RL은 전통적 Agentic RL과 근본적으로 다른 패러다임이다 | 하네스가 환경 루프를 소유하여 훈련 엔진은 요청-응답 쌍만 관찰; 상태 공간이 Harness+Environment로 확장됨 | Figure 2, p.2-3 |
| 2 | 재토크나이제이션으로 인한 토큰-프리픽스 비연속성이 샘플 병합을 불안전하게 만든다 | Chat-template 비합성성, Decode-retokenize drift, 추론 시 출력 변환의 3가지 메커니즘 규명 | Figure 3, Section 2.1, p.4-6 |
| 3 | 어드밴티지는 샘플 수준이 아닌 롤아웃 수준에서 계산해야 한다 | 재토크나이제이션은 우연적 현상으로 어드밴티지 할당을 변화시켜선 안 됨; 샘플 수준 시 베이스라인 왜곡 ($\bar{r}\_{sample}=3/4$ vs $\bar{r}_{rollout}=1/2$) | Figure 4, Section 2.2, p.6-7 |
| 4 | 손실 정규화는 롤아웃 수준 토큰 평균 손실($\mathcal{L}_{\text{rollout-mean}}$)이 가장 적합하다 | seq-mean은 샘플 수에 따라 가중치 왜곡; token-mean은 긴 부정 샘플 대량 출현 시 불안정 | Section 2.3, p.7-8 |
| 5 | Collocated Async RL이 동기 RL 대비 ~2배 속도 향상을 동일한 GPU로 달성한다 | GPU를 롤아웃과 업데이트 단계가 시간 공유; 비동기 RL보다 적은 GPU 사용 | Figure 6, Section 3.1, p.9-10 |
| 6 | 롤아웃 수준 어드밴티지 + 롤아웃 수준 정규화가 검증 보상과 엔트로피 안정성을 가장 크게 향상시킨다 | 3가지 설정 비교: 38.2% > 35.0% > 33.1% (step 128 기준) | Figure 9, Section 4.3.3, p.13-14 |
| 7 | 6K 훈련 예제와 적정 컴퓨팅만으로 SWE-bench Verified에서 +14.6%p 향상이 가능하다 | RL만으로 Qwen3.5-9B 41.8% → 56.4% (step 208) | Section 4.3, p.12-14 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제

**핵심 문제**: 배포 시 사용되는 에이전트 하네스를 RL 훈련에 직접 사용할 때, 훈련 엔진이 완전한 롤아웃 기록이 아닌 **LLM 요청-응답 쌍의 시퀀스만** 관찰하는 상황에서 올바른 학습 신호를 구성하는 방법.

$$\mathcal{C}(\rho) = \left((p_1, a_1),(p_2, a_2), \ldots,(p_{T_\rho}, a_{T_\rho})\right) $$

> - $\rho$: 하나의 롤아웃(에이전트가 하나의 태스크를 수행하는 전체 과정)
> - $p_i$: $i$번째 LLM 호출 시 전달된 프롬프트 토큰 시퀀스
> - $a_i$: $i$번째 LLM 호출에 대한 모델의 응답 토큰 시퀀스
> - $T_\rho$: 롤아웃 $\rho$에서 총 LLM 호출 횟수

환경 상호작용 및 하네스 상태 전이 정보는 보이지 않는다.

---

### 제안하는 방법 (수식 포함)

#### (A) 하네스 상태 모델링

잠재 실행 상태를 하네스 상태와 환경 상태의 합성으로 정의:

$$s_t = \left(s_t^{\text{harness}}, s_t^{\text{env}}\right) $$

> - $s_t^{\text{harness}}$: 시점 $t$에서의 하네스 내부 상태 (컨텍스트, 도구 상태 등)
> - $s_t^{\text{env}}$: 시점 $t$에서의 환경 상태 (코드베이스, 파일 시스템 등)

하네스가 컨텍스트를 구성하고 프롬프트로 렌더링:

$$C_t^{\text{msg}} = \text{Context}_H(s_t^{\text{harness}}) $$

$$p_t^{\text{tok}} = \text{Tok}\left(\text{Template}(C_t^{\text{msg}})\right) $$

> - $C_t^{\text{msg}}$: 메시지 수준 컨텍스트 (chat 메시지 리스트)
> - $\text{Context}_H$: 하네스 $H$가 상태로부터 컨텍스트를 구성하는 함수
> - $\text{Template}(\cdot)$: chat template을 적용하는 함수
> - $\text{Tok}(\cdot)$: 토크나이저

각 정책 결정은 호출 수준 전이로 기록:

$$z_t = (p_t^{\text{tok}}, a_t^{\text{tok}}), \quad a_t^{\text{tok}} \sim \pi_\theta\left(\cdot \mid p_t^{\text{tok}}\right) $$

> - $\pi_\theta$: 파라미터 $\theta$를 가진 정책 모델 (LLM)

> 💡 **POMDP (Partially Observable Markov Decision Process)**: 에이전트가 환경의 전체 상태를 직접 관찰하지 못하고, 부분적인 관찰만으로 의사결정을 해야 하는 수학적 프레임워크. 예) 체스에서 상대 패를 모르는 상황과 유사하다.

---

#### (B) 재토크나이제이션 문제

텍스트 수준 프리픽스 조건:

$$(p_i^{\text{text}}, a_i^{\text{text}}) \preceq p_{i+1}^{\text{text}} $$

토큰 수준 프리픽스 조건 (실제로 보장되지 않음):

$$(p_i^{\text{tok}}, a_i^{\text{tok}}) \preceq p_{i+1}^{\text{tok}} $$

> - $\preceq$: 정확한 토큰 수준 프리픽스 관계 (앞 시퀀스가 뒤 시퀀스의 시작 부분과 완전히 일치)

Chat-template 비합성성의 수학적 표현:

$$\text{Template}(A \| B) \neq \text{Template}(A) \| \text{Template}(B) $$

Decode-retokenize drift:

$$\text{Tok}(\text{Decode}(a_i^{\text{tok}})) \neq a_i^{\text{tok}} $$

**해결책 (Agent Lightning v1.0)**: *Best-effort sequence merging* — 식 (9)를 만족할 때만 병합하고, 실패 시 새 시퀀스를 시작한다.

> 💡 **재토크나이제이션(Retokenization)**: 텍스트를 토큰(모델이 처리하는 기본 단위)으로 변환하는 과정을 다시 수행하는 것. 같은 텍스트도 문맥에 따라 다른 토큰 경계로 분할될 수 있어 문제가 발생한다. 예) "having"이 처음엔 "h"+"aving"으로 분리되었다가 나중엔 "hav"+"ing"으로 분리되는 경우 (Figure 3).

---

#### (C) 손실 정규화

**Token-mean loss** (DAPO 방식):

$$\mathcal{L}_{\text{token-mean}} = \frac{\sum_{\rho=1}^{R}\sum_{j=1}^{N_\rho}\sum_{t=1}^{L_{\rho,j}} \ell_{\rho,j,t}}{\sum_{\rho=1}^{R}\sum_{j=1}^{N_\rho} L_{\rho,j}} $$

**Seq-mean-token-mean loss** (GRPO 방식, 문제 있음):

$$\mathcal{L}_{\text{seq-mean}} = \frac{1}{\sum_\rho^R N_\rho} \sum_{\rho=1}^{R}\sum_{j=1}^{N_\rho} \frac{1}{L_{\rho,j}}\sum_{t=1}^{L_{\rho,j}} \ell_{\rho,j,t} $$

**Rollout-level token-mean loss** (Agent Lightning v1.0 채택):

$$\mathcal{L}_{\text{rollout-mean}} = \frac{1}{R}\sum_{\rho=1}^{R} \frac{\sum_{j=1}^{N_\rho}\sum_{t=1}^{L_{\rho,j}} \ell_{\rho,j,t}}{\sum_{j=1}^{N_\rho} L_{\rho,j}} $$

> - $R$: 배치 내 롤아웃 수
> - $N_\rho$: 롤아웃 $\rho$에서 생성된 훈련 샘플 수
> - $L_{\rho,j}$: 롤아웃 $\rho$의 $j$번째 샘플의 응답 토큰 수
> - $\ell_{\rho,j,t}$: 롤아웃 $\rho$, 샘플 $j$, 토큰 위치 $t$에서의 per-token 손실값

> 💡 **GRPO (Group Relative Policy Optimization)**: DeepSeekMath에서 제안된 RL 알고리즘. 동일한 프롬프트에서 여러 롤아웃을 생성하고, 그 그룹 내 상대적 보상으로 어드밴티지를 계산한다. Value network 없이 동작하는 것이 특징이다.

---

#### (D) 훈련 배치 구성

```math
\mathcal{B}_{\text{train}} = \bigcup_{\rho \in \mathcal{B}_{\text{rollout}}} \left\{ (S_{\rho,j}, \rho, g_\rho) \mid 1 \leq j \leq N_\rho \right\}
```

> - $S_{\rho,j}$: 롤아웃 $\rho$에서 구성된 $j$번째 훈련 시퀀스
> - $g_\rho$: 롤아웃 $\rho$가 속한 프롬프트 그룹 식별자 (같은 프롬프트에서 샘플링된 롤아웃들의 집합)

---

### 모델 구조

Agent Lightning v1.0은 단일 모델 구조가 아닌 **분산 시스템 아키텍처**이다 (Figure 1, p.1):

| 컴포넌트 | 역할 | 구현 |
|----------|------|------|
| **API Gateway** | 롤아웃 상태 관리, LLM 프록시 | 롤아웃 ID 기반 상태 머신 (Queuing → Running → Succeeded/Failed) |
| **Rollout Controller** | 에이전트 실행 조율 | K8S Reconciler + Local Reconciler |
| **Customized Trainer** | 샘플 조립 및 RL 훈련 | VERL 기반, Sample Adapter + Monitoring |
| **Inference Engine** | LLM 추론 서버 | 임의의 OpenAI-compatible 서버 |
| **Training Engine** | 모델 파라미터 업데이트 | VERL 백엔드 |

---

### 성능 향상 및 한계

**성능 향상** (저자 보고):

| 에이전트 유형 | 모델 | 지표 | 초기값 | 최종값 | 향상 |
|--------------|------|------|--------|--------|------|
| Search Agent | Llama-3.2-3B-Instruct | 검증 EM | 25.1% | 41.7% | +16.6%p |
| Instruction-Following Agent | Qwen3-4B-Instruct-2507 | 검증 보상 | 51.9% | 70.2% | +18.3%p |
| Coding Agent | Qwen3.5-9B | SWE-bench Verified | 41.8% | 56.4% | +14.6%p |
| Collocated Async vs Sync | - | 처리 속도 | 1x | ~2x | 2배 |

**한계**:
- 코딩 에이전트 훈련에 단일 모델(Qwen3.5-9B)만 사용하여 일반화 검증 부족
- 어드밴티지 설계 선택의 이론적 최적성 미증명
- SWE-bench Verified 결과가 단일 체크포인트(step 208)만 보고됨

---

## 3. 각 주장의 페이지/Figure/Table 위치

| 주장 | 위치 |
|------|------|
| Harnessed vs Traditional Agentic RL 비교 | Figure 2 (p.3), Section 1 (p.2) |
| 재토크나이제이션 문제 (having 예시) | Figure 3 (p.4), Section 2.1 (p.4-6), Eq. 8-11 |
| 어드밴티지 계산 차이 | Figure 4 (p.7), Section 2.2 (p.6-7) |
| 손실 정규화 예시 (Rollout A/B/C) | Figure 5 (p.8), Section 2.3 (p.7-8), Eq. 14-16 |
| Collocated Async RL | Figure 6 (p.10), Section 3.1 (p.9-10) |
| Search Agent 훈련 결과 | Figure 7 (p.12), Section 4.1 (p.11) |
| Instruction-Following Agent 결과 | Figure 8 (p.12), Section 4.2 (p.11-12) |
| 코딩 에이전트 어드밴티지/정규화 비교 | Figure 9 (p.13), Section 4.3.3 (p.13-14) |
| 롤아웃 병합 통계 | Figure 10 (p.14), Section 4.3.3 (p.14) |
| API Gateway 설계 | Figure 11 (p.19), Table 1 (p.20), Appendix A.1 |
| Rollout Controller 설계 | Figure 12 (p.20), Appendix A.2 |
| Reward Hacking 방지 | Section 4.3.2 (p.13) |
| SWE-smith 데이터 필터링 | Section 4.3.1 (p.12-13) |

---

## 4. 저자 보고 결과 vs 해석 분리

### 저자가 직접 보고한 결과

| 항목 | 저자 보고 내용 |
|------|--------------|
| 코딩 에이전트 성능 | "RL improves Qwen3.5-9B on SWE-bench Verified from 41.8% to 56.4%, an absolute 14.6% gain" (Abstract, p.1) |
| 훈련 데이터 규모 | "only 6K training examples and modest compute" (Abstract, p.1) |
| 프레임워크 규모 | "approximately 3,500 lines of code" (Abstract, p.1) |
| Collocated Async 속도 | "roughly a 2x end-to-end speedup over synchronous RL while also using fewer GPUs" (Section 3.1, p.10) |
| Search Agent 검증 보상 | "25.1% to 41.7%, an absolute 16.6% gain" (Section 4.1, p.11) |
| Instruction-Following 검증 보상 | "51.9% to 70.2%, an absolute 18.3% improvement" (Section 4.2, p.12) |
| 코딩 에이전트 정규화 비교 | 38.2% (Rollout+Rollout) > 35.0% (Rollout Adv) > 33.1% (Sample Adv), step 128 (Section 4.3.3, p.14) |
| 롤아웃 병합 통계 | 평균 36%만 단일 샘플로 유지, 평균 2.41개 샘플/롤아웃 (Figure 10, p.14) |
| SWE-smith 데이터 필터링 결과 | 59,136 → ~6,000 훈련 예제, ~400 테스트 예제 (Section 4.3.1, p.12-13) |

### 본 분석자의 해석 (저자 직접 보고와 구분)

- **의의**: 14.6%p 향상은 단순 SFT 개선이 아닌 RL만으로 달성된 것으로, 하네스를 포함한 RL 훈련의 효과를 강력하게 시사한다. 그러나 기존 SFT 베이스라인이나 다른 방법론과의 직접 비교가 없어 절대적 우위 판단에 주의가 필요하다.
- **2x 속도 향상**: "roughly"라는 표현이 사용되어 정확한 측정값이 아님. 실험 조건 상세 정보가 부족하여 일반화에 주의가 필요하다.
- **6K 데이터의 효율성**: 오리지널 SWE-smith 59K 데이터의 약 10%만 사용했음에도 큰 향상을 보인 것은, 데이터 품질 필터링의 중요성을 보여주는 강력한 증거이다.
- **엔트로피 안정화**: Figure 9에서 Rollout-level Norm 추가 시 엔트로피가 더 느리게 증가하는 것은, 손실 정규화가 정책 탐색 안정성에 직접적 영향을 준다는 것을 시사한다.

---

## 5. 통계적 취약점 및 비교 불가능한 수치

> ⚠️ 아래 항목들은 통계적 신뢰성이 낮거나 직접 비교가 불가능한 수치입니다.

| 항목 | 문제점 | 심각도 |
|------|--------|--------|
| SWE-bench Verified 56.4% | 단일 모델(Qwen3.5-9B), 단일 체크포인트(step 208), 단일 실행 결과. 표준편차 미보고 | 🔴 높음 |
| "~2x speedup" | "roughly"라는 표현 사용, 측정 조건(GPU 종류, 배치 크기 등) 미상세화 | 🔴 높음 |
| 코딩 에이전트 비교 (38.2% vs 35.0% vs 33.1%) | 3개 설정의 단일 실행 비교, 통계적 유의성 검정 없음 | 🔴 높음 |
| Search Agent 결과 (25.1%→41.7%) | Search-R1 원 논문의 결과와 직접 비교 없음 | 🟡 중간 |
| Instruction-Following 결과 (51.9%→70.2%) | 배치 수준 훈련 보상이 "noisy"하다고 직접 인정 (p.11) | 🟡 중간 |
| "only 36% remain as single sample" | 특정 설정(Rollout-level Advantage + Rollout-level Norm)에서만 측정된 값 | 🟡 중간 |
| "2.41 training samples on average" | 상동 | 🟡 중간 |
| 3,500 lines of code | 코드 품질, 기능 커버리지 등 다른 프레임워크와의 직접 비교 기준 없음 | 🟢 낮음 |

> 💡 **어드밴티지(Advantage)**: RL에서 특정 행동이 평균 대비 얼마나 더 좋은지를 나타내는 값. $A(s,a) = Q(s,a) - V(s)$로 표현되며, 양수면 평균보다 좋은 행동, 음수면 나쁜 행동을 의미한다.

---

## 6. 논문이 답하지 않는 질문들

| 카테고리 | 미답변 질문 |
|----------|------------|
| **성능 검증** | 다른 LLM 아키텍처(GPT-4, Llama 등)에서도 동일한 향상이 재현되는가? |
| **성능 검증** | SWE-bench Verified 56.4%는 동일 컴퓨팅 예산의 SFT 대비 얼마나 우월한가? |
| **이론적 근거** | 롤아웃 수준 어드밴티지가 최적임을 이론적으로 증명할 수 있는가? |
| **하네스 일반화** | 실험에 사용된 mini-SWE-agent 외의 하네스(OpenHands, Claude Code 등)에서도 동일한 결과가 나오는가? |
| **스케일링** | 더 큰 모델(70B 이상)이나 더 많은 훈련 데이터(60K 이상)로 확장 시 성능 트렌드는? |
| **크레딧 할당** | 하나의 롤아웃에서 생성된 여러 샘플 간 최적의 크레딧 할당 방법은 무엇인가? (p.7에서 Future work로 언급) |
| **재토크나이제이션 영향** | 재토크나이제이션이 실제로 훈련 결과에 미치는 영향을 정량적으로 측정한 실험은 없는가? |
| **Collocated Async 정확도** | Collocated Async의 오프-폴리시 효과가 최종 모델 품질에 미치는 영향은? |
| **하네스 보안** | 네트워크 차단 외의 더 강건한 reward hacking 방지 방법은 무엇인가? |
| **비용 분석** | Kubernetes 기반 자체 호스팅 대비 상용 샌드박스 서비스의 실제 비용 비교는? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1): Agent Lightning v1.0 전체 프레임워크

```
API Gateway ← → Rollout Controller ← → Kubernetes Cluster
     ↕                                        ↕
Inference Engine                        Agent Harnesses
     ↕
Training Engine → Model
```

**해석**: 이 그림은 시스템의 핵심 철학인 **분리(disaggregation)**를 시각화한다. 왼쪽의 에이전트 실행 영역(Kubernetes)과 오른쪽의 훈련 영역(Inference+Training Engine)이 API Gateway를 통해서만 통신한다. 이 구조가 "임의의 하네스 지원"을 가능하게 하는 근본 이유이다. LLM API Proxy는 하네스가 자신이 훈련 시스템과 연결되어 있다는 것을 인식하지 않아도 되게 만드는 투명한 계층이다.

---

### Figure 2 (p.3): Traditional vs Harnessed Agentic RL 비교

**해석**: 이 그림은 논문 전체의 핵심 개념을 단 하나의 다이어그램으로 설명한다.

- **전통적 방식(왼쪽)**: 정책 모델 ↔ (De)Tokenizer ↔ Environment로 직접 연결. 모델은 연속된 하나의 토큰 히스토리를 관찰한다.
- **Harnessed 방식(오른쪽)**: Environment와 Policy Model 사이에 **Agent Harness**가 삽입됨. 모델은 OAI-like API를 통해 각 호출마다 독립적으로 구성된 프롬프트만 받는다.

우측 표에서 State, Model Input, Agents 세 차원에서의 차이가 명확히 드러난다. 이 구조 변화가 이후 4가지 챌린지(재토크나이제이션, 어드밴티지, 손실, 스케줄링)를 모두 야기한다.

---

### Figure 4 (p.7): 어드밴티지 계산 비교

```
Traditional:  Prompt → Rollout1(Sample1, R=1) / Rollout2(Sample1, R=0)
              → baseline = (1+0)/2 = 0.5 ✓

Harnessed:    Prompt → Rollout1(Sample1,2,3, R=1) / Rollout2(Sample4, R=0)
              → rollout-level baseline = (1+0)/2 = 0.5 ✓
              → sample-level baseline = (1+1+1+0)/4 = 0.75 ✗
```

**해석**: 이 그림은 왜 샘플 수준 어드밴티지가 문제인지를 직관적으로 보여준다. 하네스의 내부 동작(서브에이전트 생성, 컨텍스트 요약)으로 롤아웃이 3개 샘플로 분리되면, 샘플 수준 계산은 베이스라인을 0.75로 왜곡시켜 Rollout 1의 모든 샘플이 실제보다 낮은 어드밴티지를 받게 된다. 이는 우연적 분리가 학습 신호를 변질시키는 핵심 문제이다.

---

### Figure 6 (p.10): Sync vs Async vs Collocated Async RL

```
Sync:              [Rollout 1-4] ──────────── [Update 1-4]  (GPU idle 문제)
Async:             [Rollout 1-4] [Rollout 5-8] [Rollout 9-10]
                         [Update 1-4]      [Update 5-8]  (GPU 두 배 필요)
Collocated Async:  [Rollout 1][2][3][4][5] [Update 1,3,4,5][2] [6][7][8] ...
                   (같은 GPU에서 시간 공유)
```

**해석**: 동기 RL은 가장 느린 롤아웃을 기다려야 하는 "최악 케이스 대기" 문제가 있다. 비동기 RL은 이를 해결하지만 GPU를 두 배 사용한다. Collocated Async는 충분한 롤아웃이 수집되면 즉시 업데이트를 시작하고, API Gateway가 새 요청을 잠시 보류하여 GPU를 시간 공유한다. 에이전트 하네스는 이 전환을 인식하지 못한다. 이는 리소스 효율과 훈련 속도를 동시에 달성하는 핵심 공학적 기여이다.

---

### Figure 9 (p.13): 코딩 에이전트 훈련 동역학 비교

**해석**: 세 가지 설정의 검증 보상(왼쪽)과 정책 엔트로피(오른쪽)를 비교한다.

- **Sample-level Advantage** (노란선): 가장 낮은 최종 성능(33.1%), 엔트로피 변동 큼
- **Rollout-level Advantage** (초록선): 중간 성능(35.0%), 엔트로피가 빠르게 증가
- **Rollout-level Advantage + Rollout-level Norm** (파란선): 최고 성능(38.2%), 엔트로피 증가가 가장 느리고 안정적

특히 롤아웃 수준 어드밴티지만 적용했을 때 엔트로피가 급격히 증가하는 현상은, 어드밴티지 교정이 최적화를 너무 공격적으로 만드는 부작용이 있음을 시사한다. 롤아웃 수준 정규화가 이를 완화하여 더 안정적인 훈련을 가능하게 한다.

> 💡 **정책 엔트로피(Policy Entropy)**: 모델이 얼마나 다양한 행동을 생성하는지 측정하는 지표. 엔트로피가 너무 빠르게 낮아지면 모델이 특정 행동에만 집중하는 "조기 수렴" 또는 "exploitation"이 발생하고, 너무 높으면 무작위적 행동을 한다.

---

## 8. 결론: 시사점, 후속 연구 계획 및 추가 방향

### 저자 제시 시사점 (p.14-15)

1. **패러다임 정립**: Harnessed Agentic RL은 전통적 Agentic RL과 근본적으로 다르며, 별도의 설계 원칙이 필요하다.
2. **롤아웃 수준 설계**: 어드밴티지 계산과 손실 정규화 모두 롤아웃 수준에서 처리해야 훈련 안정성이 높아진다.
3. **재현 가능성**: 6K 데이터와 적정 컴퓨팅으로 코딩 에이전트 RL 훈련이 가능함을 보임으로써, 소규모 팀도 연구 참여 가능함을 시사한다.
4. **Reward hacking 대응**: 시스템 수준의 네트워크 차단과 Git 히스토리 은폐가 효과적인 safeguard임을 실증.

### 저자 제시 후속 연구 (p.7, 미래 연구로 명시)

- 롤아웃 내 샘플들 간의 더 나은 크레딧 할당 방법 설계 (Section 2.2, p.7)
- 추가 하네스 유형 및 에이전트 설정으로의 확장 (암묵적으로 시사)

---

### 8-1. 모델의 일반화 성능 향상 가능성

논문이 직접 다루지 않았으나, 다음의 분석이 가능하다:

#### 현재 일반화 한계

1. **단일 모델 검증**: 코딩 에이전트는 Qwen3.5-9B 단 하나의 모델로만 검증되었다. 동일한 훈련 파이프라인이 더 큰 모델(70B)이나 다른 아키텍처에서도 동일한 효과를 보이는지 불명확하다.
2. **단일 하네스 의존**: mini-SWE-agent에 특화된 훈련이 다른 하네스(OpenHands)에서 제로샷으로 작동하는지 검증되지 않았다.
3. **도메인 범위 제한**: SWE-smith는 Python 저장소 중심이며, JavaScript, Rust 등 다른 언어나 다른 소프트웨어 엔지니어링 태스크로의 일반화가 불명확하다.
4. **OOD 일반화**: 훈련 데이터의 난이도 분포(중간 난이도 집중)가 실제 SWE-bench Verified 배포와 다를 수 있어 분포 이동(distribution shift) 문제가 존재한다.

#### 일반화 향상 잠재력 분석

| 방향 | 근거 | 예상 효과 |
|------|------|----------|
| **다양한 하네스로 훈련** | 하네스가 다르면 컨텍스트 구성 방식이 달라 각 하네스에 최적화된 정책 학습 가능 | 특정 하네스 의존성 감소 |
| **다언어/도메인 데이터 확장** | 현재 Python 중심 → TypeScript, Rust 데이터 추가 | 범용 코딩 능력 향상 |
| **커리큘럼 학습 적용** | 현재 난이도 필터링이 단순함 → 점진적 난이도 증가 전략 | 더 어려운 태스크 일반화 |
| **다중 에이전트 하네스 훈련** | Figure 2에서 Multi-agent, subagents, handoffs가 언급됨 → 이 기능 활용 | 복잡한 태스크 분해 능력 |
| **오프라인 데이터 통합** | SWE-smith 외 SWE-bench Lite 등 기존 데이터셋과 혼합 | 데이터 분포 다양화 |

**핵심 관찰**: Best-effort sequence merging 전략은 재토크나이제이션으로 인해 평균 64%의 롤아웃이 2개 이상의 샘플로 분리된다 (Figure 10). 이는 훈련 효율성 손실이지만, 동시에 각 LLM 호출이 독립적으로 학습됨을 의미하여 **다양한 컨텍스트에 대한 robust한 정책 학습**을 가능하게 할 수 있다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 비교 분석은 본 논문의 참고문헌 정보와 일반적 AI 연구 동향에 기반합니다. 각 논문의 정확한 수치는 해당 논문을 직접 확인하시기 바랍니다.

#### 관련 연구 계보 및 비교

| 연구 | 연도 | 핵심 기여 | Agent Lightning v1.0과의 관계 |
|------|------|-----------|-------------------------------|
| **verl (HybridFlow)** [8] | 2025 | 유연한 RLHF 프레임워크, FSDP+Megatron 혼합 | 훈련 백엔드로 활용, 하네스 통합 어려움 |
| **ReAct** [16] | 2023 (ICLR) | 추론과 행동을 결합한 에이전트 패러다임 | 전통적 Agentic RL의 기반 패러다임 |
| **GRPO (DeepSeekMath)** [17] | 2024 | Value network 없는 그룹 상대적 정책 최적화 | 코딩/검색 에이전트 훈련 알고리즘으로 채택 |
| **DAPO** [18] | 2025 | 오픈소스 대규모 RL 시스템, token-mean loss | 손실 정규화 방법 비교 대상 |
| **AReaL** [9] | 2025 | 대규모 비동기 RL 시스템 | 비동기 RL 아이디어 참조, 하네스 통합 방식 차이 |
| **Search-R1** [19] | 2025 | 검색 엔진과 RL을 결합한 훈련 | 검색 에이전트 실험 설정 기반 |
| **SWE-smith** [21] | 2025 (NeurIPS) | 59K 소프트웨어 엔지니어링 태스크 데이터셋 | 코딩 에이전트 훈련 데이터 소스 |
| **SWE-Gym** [33] | 2024 | 소프트웨어 엔지니어링 에이전트 훈련 환경 | 비교 대상 (6TB vs SWE-smith 295GB) |
| **LLM-in-Sandbox** [20] | 2026 | 컴퓨터 환경에서 범용 에이전트 훈련 | 일반 instruction-following 실험 기반 |
| **verl Uni-Agent** [12] | 2026 | 프록시 기반 에이전트 RL 통합 | Agent Lightning 아이디어를 verl에 통합한 후속 |
| **AReaL 2.0** [13] | 2026 | 차세대 에이전트 RL 시스템 | 동일 패러다임 추구, 다른 설계 선택 |
| **slime v0.3.0** [10] | 2025/2026 | LLM 포스트 트레이닝 RL 스케일링 | 샘플 수준 어드밴티지 사용 (본 논문과 견해 차이) |
| **Polar** [14] | 2026 | 대규모 하네스 에이전트 RL | 롤아웃 수준 어드밴티지 사용 (본 논문과 일치) |
| **RLOO** [30] | 2024 | REINFORCE 스타일 LLM 최적화 | Instruction-following 에이전트에 채택 |

#### 이 논문이 향후 연구에 미치는 영향

1. **패러다임 정립 효과**: "Harnessed Agentic RL"이라는 용어와 개념을 최초로 명확히 정의함으로써, 향후 관련 연구의 공통 언어와 분류 기준을 제공한다.

2. **알고리즘 설계 기준 제시**: 롤아웃 수준 어드밴티지와 rollout-mean 손실이 더 원칙적임을 실험적으로 검증하여, 후속 프레임워크들의 기본 설계 지침이 될 가능성이 높다.

3. **재현 가능성 기여**: 6K 데이터와 완전한 스크립트 공개는 소규모 연구팀의 코딩 에이전트 RL 연구 진입 장벽을 크게 낮춘다.

4. **상용 의존성 탈피**: Kubernetes 기반 자체 호스팅 접근법은 Modal, E2B 등 상용 샌드박스 의존성을 제거하여 연구 비용을 절감한다.

#### 앞으로 연구 시 고려할 점

| 고려사항 | 상세 내용 |
|----------|-----------|
| **Retokenization 표준화** | 각 프레임워크가 다른 전략(버퍼 대체 vs best-effort vs 독립 계산)을 사용하므로, 재현 가능한 비교를 위해 명시적으로 전략을 보고해야 한다 |
| **어드밴티지 계산 수준 명시** | 샘플/롤아웃 수준 어드밴티지 중 어느 것을 사용했는지 논문에서 명확히 밝혀야 결과 비교가 가능하다 |
| **다중 하네스 평가** | 단일 하네스에서의 결과는 하네스 특화 오버피팅 가능성이 있으므로, 여러 하네스에서의 크로스 평가가 필요하다 |
| **Reward hacking 체계적 분류** | 4가지 reward hacking 패턴이 발견된 만큼, 더 광범위한 분류 체계 개발이 필요하다 |
| **오프-폴리시 영향 정량화** | Collocated Async의 오프-폴리시 효과가 실제로 얼마나 성능에 영향을 미치는지 이론적·실험적 분석이 필요하다 |
| **동적 샘플 수의 이론적 분석** | 훈련 수렴성과 샘플 효율성에 대한 이론적 보장이 현재 전무하다 |
| **크레딧 할당의 정밀화** | 롤아웃 내 샘플들(특히 서브에이전트 생성 시)에 대한 더 정교한 크레딧 할당이 향후 성능 향상의 핵심 변수가 될 것이다 |

---

## 참고문헌 (논문 내 인용 기준)

본 분석에서 직접 참조한 논문 내 문헌:

- [1] SWE-agent Team. mini-swe-agent. GitHub, 2026.
- [8] Sheng et al. HybridFlow: A Flexible and Efficient RLHF Framework. EuroSys 2025.
- [9] Fu et al. AReaL: A Large-scale Asynchronous Reinforcement Learning System. arXiv:2505.24298, 2025.
- [10] Zhu et al. slime: An LLM Post-training Framework for RL Scaling. GitHub, 2025.
- [11] Luo et al. Agent Lightning: Train ANY AI Agents with Reinforcement Learning. arXiv:2508.03680, 2025.
- [12] Ding et al. Uni-Agent: Build, Run, and Train Agents at Scale. GitHub, 2026.
- [13] Yan et al. Next-generation Agentic Reinforcement Learning Systems. arXiv:2607.01120, 2026.
- [14] Xu et al. Polar: Agentic RL on Any Harness at Scale. arXiv:2605.24220, 2026.
- [16] Yao et al. ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.
- [17] Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning. arXiv:2402.03300, 2024.
- [18] Yu et al. DAPO: An Open-source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, 2025.
- [19] Jin et al. Search-R1: Training LLMs to Reason and Leverage Search Engines. arXiv:2503.09516, 2025.
- [20] Cheng et al. Computer Environments Elicit General Agentic Intelligence in LLMs. arXiv:2601.16206, 2026.
- [21] Yang et al. SWE-smith: Scaling Data for Software Engineering Agents. NeurIPS 2025.
- [30] Ahmadian et al. Back to Basics: Revisiting REINFORCE Style Optimization. arXiv:2402.14740, 2024.
- [33] Pan et al. Training Software Engineering Agents and Verifiers with SWE-Gym. arXiv:2412.21139, 2024.
- [34] Jimenez et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024.

**논문 프로젝트 페이지**: https://github.com/microsoft/agent-lightning
