# MemoHarness: Agent Harnesses That Learn from Experience 

> **⚠️ 중요 고지**: 본 논문은 arXiv:2607.14159v1 (2026년 7월 14일 제출)로, **프리프린트(Under Review)** 상태입니다. 동료 심사를 거치지 않았으며, 저자들 스스로 통계적 강건성과 컴포넌트 귀인에 대한 한계를 명시적으로 인정하고 있습니다.

---

## 1. Executive Summary (10문장 이내)

MemoHarness는 LLM 에이전트를 감싸는 "에이전트 하네스(agent harness)"를 자동으로 최적화하는 프레임워크다.  
기존 방법들이 프롬프트, 파이프라인, 워크플로우 등 부분적 요소만 최적화하는 반면, MemoHarness는 하네스 전체를 6개의 편집 가능한 제어 차원으로 분해한다.  
훈련 시에는 레이블된 탐색 사례를 통해 경험을 누적하고, 이를 이중 레이어 경험 뱅크(dual-layer experience bank)에 저장한다.  
테스트 시에는 레이블, 피드백, 추가 탐색 없이 유사 사례를 검색하여 하네스를 케이스별로 적응시킨다.  
Terminal-Bench 기준 MemoHarness는 0.806을 달성하여 최강 기준선(Codex, 0.722) 대비 +0.084 향상을 보였다.  
LiveCodeBench(0.900→0.967)와 FinanceAgent(0.600→0.767)에서도 개선을 확인했다.  
크로스 모델 전이 실험에서 평균 +0.098의 성능 향상을 7개 모델 모두에서 달성했다.  
경험 뱅크 검색 비용은 캐싱을 통해 경쟁력 있는 수준($6.89)으로 유지될 수 있다.  
다만 18개 태스크 기반의 소규모 평가, 신뢰구간 미제공, 컴포넌트 격리 미실시 등 중요한 통계적 한계가 존재한다.  
저자들은 이 결과를 "실행 경험이 적응형 하네스 구축의 실용적 기반이 될 수 있다는 증거"로 제한적으로 해석한다.

### 1-1. 연구의 목적과 필요성

**문제 배경**: LLM 에이전트의 성능은 기반 모델뿐 아니라, 모델을 감싸는 **에이전트 하네스**(컨텍스트 조립, 도구 접근, 추론 오케스트레이션, 메모리, 출력 처리 등)에 의해 크게 좌우된다. 실무 경험상 하네스 설계만으로도 동일 모델에서 태스크 성공률이 수십 퍼센트포인트 달라질 수 있다(Lopopolo, 2026; LangChain, 2025).

**기존 방법의 한계**: 기존 자동화 개선 방법들은 프롬프트(Fernando et al., 2023), 선언적 파이프라인(Khattab et al., 2023), 워크플로우(Zhang et al., 2024) 등 **부분 요소만** 최적화하며, 배포된 에이전트는 모든 케이스에 단일 전역 하네스를 재사용한다. 가장 근접한 선행 연구인 Meta-Harness(Lee et al., 2026)도 훈련 시점의 정적 산출물을 생성할 뿐, **테스트 시점의 케이스별 적응**을 제공하지 않는다.

**필요성**: ① 하네스는 고차원이고 차원 간 결합도가 높어 단일 차원의 변경이 전체에 영향을 줌, ② 벤치마크 점수만으로는 어느 차원이 실패를 야기했는지 진단 불가, ③ 테스트 시점 적응은 레이블 없이 이루어져야 함 — 이 세 가지 도전 과제를 해결하기 위해 MemoHarness가 제안되었다. (Abstract, pp.1–2)

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 출처 |
|---|-----------|------|------|
| C1 | 6차원 분해가 단일 불투명 프롬프트보다 효과적 | D1~D6 구조화 편집으로 체계적 진단 가능 | §2.3, Table 1 |
| C2 | 이중 레이어 경험 뱅크가 단순 스칼라 점수보다 풍부한 학습 신호 제공 | 케이스별 진단 + 전역 패턴 증류로 반복 실패 패턴 발견 | §2.4 |
| C3 | 테스트 시 케이스별 적응이 단일 전역 하네스보다 우수 | Terminal-Bench: 0.806 vs 0.722(Codex) | §3.2, Figure 2 |
| C4 | 학습된 하네스가 미관측 평가 스위트에 선택적으로 전이 | MMMLU +0.030, StrongReject +0.030, SWE-Pro +0.059 (TB 소스 기준) | §3.2, Table 2 |
| C5 | 학습된 하네스가 재훈련 없이 다른 기반 모델로 전이 | 7개 모델 모두에서 개선, 평균 +0.098 | §3.2, Table 3 |
| C6 | 캐싱 가정 하에 비용 경쟁력 유지 | $6.89 (Codex $10.28 대비 저렴, Terminus/OpenCode 대비 고가이나 정확도 우위) | §3.2, Table 4 |
| C7 | 장기 에이전트 작업일수록 하네스 탐색 효과 큼 | FinanceAgent 42.5%→65.0% (10회 반복), LiveCodeBench는 포화 상태 | §3.2, Figure 4 |

---

## 2-1. 해결 문제·제안 방법·모델 구조·성능 향상 및 한계

### 해결하고자 하는 문제

1. **하네스 전체 최적화의 부재**: 기존 방법은 부분 요소만 최적화
2. **단일 전역 하네스의 경직성**: 케이스 유형(도메인, 모호성, 추론 깊이, 검색 필요성)에 무관하게 동일 하네스 사용
3. **진단 정보 부재**: 스칼라 점수만으로는 어떤 차원이 실패를 유발했는지 알 수 없음
4. **테스트 시점 적응 불가**: 배포 후 레이블 없이 실시간 적응 불가

### 제안하는 방법 (수식 포함)

**탐색 케이스와 평가 케이스 정의** (Def. 2.1, p.4):

```math
\mathcal{D}_{\text{search}} = \left\{x_i^s = (u_i, \phi_i, y_i^\star)\right\}_{i=1}^n \quad \text{(레이블 있음)}
```

```math
\mathcal{D}_{\text{test}} = \left\{x_j^{\text{test}} = (u_j, \phi_j)\right\}_{j=1}^m \quad \text{(레이블 없음)}
```

**하네스 구성 공간** (Def. 2.2, p.4):

$$W \in \mathcal{W} = \mathcal{W}^{(1)} \times \mathcal{W}^{(2)} \times \cdots \times \mathcal{W}^{(6)}, \quad W = \left(W^{(1)}, \ldots, W^{(6)}\right) $$

**실행 궤적 및 비용** (p.4):

$$\tau_i(W) = \left(y_i(W),\ \mathcal{M}_i(W),\ \kappa_i(W)\right), \quad \kappa_i(W) = \left(n_i^{\text{call}}(W),\ n_i^{\text{tok}}(W),\ \ell_i(W)\right) $$

$$c_i(W) = n_i^{\text{tok}}(W) $$

$$r_i(W) = R\left(y_i(W),\ y_i^\star\right) $$

**이중 레이어 경험 뱅크** (§2.4, p.5):

$$\mathcal{B}_t = \left(\mathcal{E}_t,\ \mathcal{G}_t\right) $$

케이스별 실행 항목:

$$\xi_i^{(t)} = \left(i,\ t,\ \phi_i,\ W_t,\ \Delta_i^{(t)},\ \tau_i(W_t),\ r_i(W_t),\ c_i(W_t),\ z_i^{(t)}\right) $$

$$\Delta_i^{(t)} = \Delta\left(W_t,\ W_i^{ < t}\right) $$

진단 연산자:

$$z_i^{(t)} = g\left(x_i^s,\ W_t,\ \tau_i(W_t),\ r_i(W_t)\right) $$

$$z_i^{(t)} = \left(s_i^{(t)},\ d_{i,\text{prim}}^{(t)},\ \mathcal{D}_{i,\text{sec}}^{(t)},\ a_i^{(t)}\right) $$

전역 패턴 증류:

$$\mathcal{G}_t \leftarrow \mathcal{G}_{t-1} \cup \text{Distill}(\mathcal{E}_{\leq t}) $$

$$\mathcal{S}_t(q) = \text{Retrieve}(\mathcal{B}_t,\ q) $$

**훈련 시 하네스 최적화** (§2.5, p.6):

$$q_t = Q(W_{t-1},\ \mathcal{B}_{t-1}), \quad \mathcal{S}_{t-1}(q_t) = \text{Retrieve}(\mathcal{B}_{t-1},\ q_t), \quad W_t = \Pi_{\text{train}}(W_{t-1},\ \mathcal{S}_{t-1}(q_t)) $$

정확도 우선 선택 (lexicographic):

$$\bar{r}_t = \frac{1}{n}\sum_{i=1}^n r_i(W_t), \quad \bar{c}_t = \frac{1}{n}\sum_{i=1}^n c_i(W_t) $$

$$W^\star \in \underset{\text{lex},\ W_t \in \mathcal{C}_{\text{feas}}}{\arg\max}\left(\bar{r}_t,\ -\bar{c}_t\right) $$

**테스트 시 케이스 적응** (§2.6, p.7):

유사도 점수:

$$\rho_\psi(x, \xi) = \cos\left(\psi(u),\ \psi(u_\xi)\right) $$

성공/실패 이웃 검색:

$$\mathcal{N}_K^+(x) = \text{TopK}_{\xi \in \mathcal{E}_T^+}\left[\rho_\psi(x, \xi)\right], \quad \mathcal{N}_K^-(x) = \text{TopK}_{\xi \in \mathcal{E}_T^-}\left[\rho_\psi(x, \xi)\right] $$

테스트 시 증거:

$$\mathcal{S}_{\text{test}}(x) = \left(\mathcal{N}_K^+(x),\ \mathcal{N}_K^-(x),\ \text{Retrieve}(\mathcal{B}_T,\ Q_{\text{test}}(x)),\ \mathcal{G}_T\right) $$

케이스별 하네스 생성:

$$W(x) = \Pi_{\text{test}}\left(W^\star,\ x,\ \mathcal{S}_{\text{test}}(x)\right) $$

### 모델 구조 (6차원 하네스 공간)

| 차원 | 단계 | 기능 |
|------|------|------|
| **D1 Context assembly** | Pre-call 입력 구성 | 명령어, 제약, 검색 자료, 예시로 모델 입력 구성 |
| **D2 Tool interaction** | 외부 도구/검색 사용 | 외부 도구 호출 방식 및 시점 제어 |
| **D3 Generation control** | 디코딩 구성 | 샘플링 파라미터 및 토큰 예산 설정 |
| **D4 Orchestration** | 워크플로우 토폴로지 | 모델 호출 순서 및 중간 추론 단계 선택 |
| **D5 Memory management** | 크로스 콜 상태 유지 | 호출 간 유지/삭제할 상태 결정 |
| **D6 Output processing** | Post-call 출력 처리 | 원시 출력을 최종 답변으로 변환 |

(Table 1, p.5)

### 성능 향상

| 벤치마크 | Base | MemoHarness | 향상 |
|----------|------|-------------|------|
| Terminal-Bench | 0.722 | 0.806 | +0.084 |
| LiveCodeBench | 0.900 | 0.967 | +0.067 |
| FinanceAgent | 0.600 | 0.767 | +0.167 |
| 크로스 모델 평균 | — | — | +0.098 |

### 한계 (Appendix A, p.14)

1. Terminal-Bench 평가가 18개 태스크만 사용 — 신뢰구간·유의성 검정 미제공
2. 일부 기준선이 동일 모델·런타임을 공유하지 않아 순수 스캐폴드 비교 불가
3. 경험 뱅크, 전역 패턴, 테스트 시 적응의 개별 ablation 미실시
4. 비용 분석이 높은 캐싱률 가정에 의존
5. 컨트롤러와 진단 연산자가 휴리스틱으로 구현되어 일반화 한계

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| 하네스 설계가 성능을 수십%p 좌우 | p.1, Introduction |
| 6차원 분해 구조 | p.5, Table 1, §2.3 |
| 이중 레이어 경험 뱅크 정의 | p.5–6, §2.4, Eq.(7)–(13) |
| 훈련 시 최적화 알고리즘 | p.6–7, §2.5, Eq.(14)–(16) |
| 테스트 시 적응 메커니즘 | p.7, §2.6, Eq.(17)–(20) |
| Terminal-Bench 결과 (0.806 vs 0.722) | p.8, Figure 2 |
| 3개 벤치마크 체크포인트 진화 | p.9, Figure 3 |
| FinanceAgent/LiveCodeBench 반복 곡선 | p.9, Figure 4 |
| 크로스 데이터셋 전이 | p.10, Table 2 |
| 크로스 모델 전이 | p.10, Table 3 |
| 비용 분석 | p.11, Table 4 |
| 연산 수준 진단 | p.19–20, Appendix G, Table 6 |
| 한계 사항 | p.14–15, Appendix A |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

- **RQ1**: Terminal-Bench에서 MemoHarness = 0.806, Codex = 0.722, Claude Code = 0.556, OpenCode = 0.389, Terminus = 0.361 (Figure 2, p.8)
- **RQ2**: LiveCodeBench 0.900→0.967, FinanceAgent 0.600→0.767, Terminal-Bench 0.722→0.806 (Figure 3, p.9)
- **RQ3**: Terminal-Bench 소스 하네스로 MMMLU +0.030, StrongReject +0.030, SWE-Pro +0.059 (Table 2, p.10)
- **RQ4**: 7개 모델 모두 개선, 평균 +0.098, GLM-5 최대 +0.233, GPT-4.1 최소 +0.038 (Table 3, p.10)
- **RQ5**: MemoHarness 입력 14.18M 토큰 중 13.32M 캐시됨, 비용 $6.89 (Table 4, p.11)
- **Table 6**: cat 연산 추가 시 긍정적 전환율 72.7% (+59.5pp), curl은 5.3% (−7.9pp) (p.20)

### 분석자(필자)의 해석

- **해석 1**: Terminal-Bench 비교에서 Codex는 GPT-5.3-Codex 기반이나, Claude Code·Terminus·OpenCode는 서로 다른 기반 모델을 사용하므로, 이는 순수 하네스 효과가 아닌 **시스템 수준 비교**임. 저자도 이를 명시적으로 인정함(p.8).
- **해석 2**: FinanceAgent에서 10회 반복으로 42.5%→65.0% 향상은 인상적이나, 유사하게 **소규모 케이스**에서의 분산 미보고로 안정성 판단 불가.
- **해석 3**: 크로스 모델 전이에서 GLM-5의 +0.233 향상은 기저 성능(0.500)이 낮은 모델에서 하네스 효과가 더 두드러질 수 있음을 시사 — **성능 천장 효과(ceiling effect)**의 반대 현상.
- **해석 4**: 캐싱 의존 비용 분석($6.89)은 배포 환경에 따라 크게 달라질 수 있어, 비용 경쟁력 주장에는 조건부 유보가 필요함.
- **해석 5**: Table 6의 연산 수준 분석은 소표본(cat $n_{\text{add}}=11$, sed $n_{\text{add}}=11$)이라 인과성이 아닌 상관성만 보여줌.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적 취약점

| 항목 | 문제점 | 위치 |
|------|--------|------|
| **18개 태스크 평가** | 95% CI 미제공, p-값 없음. 점 추정치만 보고 | Appendix A, p.14 |
| **Table 6 소표본** | cat($n=11$), sed($n=11$), strings($n=4$), head($n=5$) — 신뢰구간 미제공, 다중 비교 보정 없음 | p.20 |
| **크로스 모델 전이** | 모델당 단일 실행 결과로 보임, 반복 실험 여부 불명확 | Table 3, p.10 |
| **FinanceAgent 10회 반복** | 각 반복의 분산 미보고 | Figure 4, p.9 |

### ⛔ 비교 불가능한 수치

| 비교 | 비교 불가 이유 |
|------|---------------|
| MemoHarness(GPT-5.3-Codex) vs Claude Code | Claude Code는 별도 모델 사용 → 모델+하네스 동시 차이 |
| MemoHarness vs Terminus/OpenCode | 기반 모델 차이 존재 |
| 비용 비교 (Table 4) | 캐싱률 가정이 시스템마다 다름; MemoHarness의 높은 캐싱률(94%)이 다른 시스템에는 적용 안 될 수 있음 |
| 크로스 데이터셋 Table 2의 HEFix/RG-Easy | 모든 하네스가 1.000/0.947 → 천장 효과로 차별화 불가 |
| Meta-Harness 직접 비교 | 공개 구현 미확보로 정량 비교 불가 (Appendix E, p.17) |

---

## 6. 논문이 답하지 않는 질문

1. **컴포넌트 ablation 부재**: D1~D6 각 차원, 경험 뱅크(E vs G), 테스트 시 적응의 개별 기여도를 격리하지 않음 → "어떤 차원이 가장 중요한가?" 미답변
2. **하이퍼파라미터 민감도**: $T=10$, $M=5$, $N=3$, $K_{\text{succ}}=K_{\text{fail}}=10$ 선택의 근거와 민감도 분석 없음
3. **유사도 함수 $\psi$의 구체적 구현**: Eq.(17)의 $\psi(u)$ 표현 방식 미공개
4. **비지도 탐색 가능성**: 레이블 없이 탐색이 가능한지 미실험
5. **온라인 경험 누적**: 배포 후 새로운 케이스로부터 지속 학습하는 방식 미연구
6. **실패 진단의 정확도**: 진단 연산자 $g$가 올바른 차원을 얼마나 정확히 식별하는가?
7. **케이스 유사도 측정의 질**: 코사인 유사도 기반 검색이 실제 유사한 하네스 전략을 가진 케이스를 잘 찾는가?
8. **전이 실패 원인**: LawBench에서 일부 하네스가 성능 저하를 보이는 원인 분석 없음
9. **확장성**: 수천 개 케이스, 수백 회 반복에서의 경험 뱅크 확장성 미검증
10. **서로 다른 도메인에서 탐색된 하네스 결합**: 여러 소스 벤치마크에서 학습한 하네스를 앙상블할 경우의 효과

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.3): MemoHarness 전체 파이프라인 개요

**해석**: 전체 시스템을 Phase A(훈련 시 탐색)와 Phase B(테스트 시 적응) 두 단계로 시각화한다. Phase A에서는 레이블된 탐색 케이스에 후보 하네스 $W$를 실행하고, 정확도 우선-비용 차순의 보상 신호로 랭킹하여 이중 레이어 경험 뱅크 $\mathcal{B}_t = (\mathcal{E}_t, \mathcal{G}_t)$에 저장한다. Phase B에서는 레이블 없는 테스트 케이스에 대해 유사 케이스와 전역 패턴을 검색하여 전역 하네스 $W^\star$를 케이스별 하네스 $W(x_j)$로 적응시킨다. **핵심 관찰**: 두 단계의 명확한 분리가 테스트 시 레이블 누출을 방지하는 설계적 특징임.

### Figure 2 (p.8): Terminal-Bench 기준선 비교

**해석**: 5개 시스템의 Terminal-Bench 성능을 비교한다. MemoHarness(0.806) > Codex(0.722) > Claude Code(0.556) > OpenCode(0.389) > Terminus(0.361) 순이다. **중요 주의점**: Codex를 제외한 나머지 기준선은 MemoHarness와 다른 기반 모델을 사용하므로, 순수 하네스 효과의 격리가 어렵다. 저자 스스로도 "closer to isolating the surrounding harness"라고 조건부로 표현하며, 가장 강한 비교 포인트는 동일 GPT-5.3-Codex 기반의 Codex 대비 +0.084 향상임.

### Figure 3 (p.9): 6개 체크포인트에서 하네스 품질 진화

**해석**: base, 4회 중간 반복, 최종 선택 하네스에서의 성능을 3개 벤치마크별로 보여준다. 모든 벤치마크에서 최종값이 base보다 높지만, 훈련 중 피크가 최종값보다 높은 경우가 있다(LiveCodeBench: 1.000 → final 0.967, Terminal-Bench: 0.833 → final 0.806). **핵심 관찰**: 이는 검증 기반 선택이 in-training 피크 대비 보수적으로 작동함을 의미하며, 과적합 방지의 증거이기도 하고 동시에 전반적 성능 상한을 낮추는 트레이드오프이기도 함.

### Figure 4 (p.9): FinanceAgent vs LiveCodeBench 반복별 성공률

**해석**: 왼쪽 FinanceAgent는 10회 반복에 걸쳐 42.5%에서 65.0%로 지속적으로 향상되며, 특히 반복 5 이후 가파른 상승이 나타난다. 오른쪽 LiveCodeBench는 91.2%~95.0% 밴드 내에서 진동하며 포화 상태를 보인다. **핵심 관찰**: 이 대조는 "하네스 탐색 효과가 헤드룸(개선 여지)이 클수록 크다"는 직관을 뒷받침하며, 하네스 최적화가 장기 에이전트 작업에 가장 실용적임을 시사한다. 단, FinanceAgent의 반복별 분산이 미보고되어 급격한 향상이 실제 학습 효과인지 측정 노이즈인지 불명확함.

### Table 6 (p.20): 연산 수준 lift 분석

**해석**: 인접 반복 전환에서 특정 쉘 연산이 새로 추가될 때 보상 향상이 일어날 확률을 기준선(~13.2%) 대비 비교한다. `cat`(+59.5pp), `sed`(+23.2pp), `which`(+20.2pp), `test`(+17.3pp)는 강하게 양의 관계를 보이는 반면, `curl`(−7.9pp), `echo`(−2.5pp)는 음의 관계를 보인다. **핵심 관찰**: 이 분석은 이중 레이어 경험 뱅크가 단순 점수를 넘어 실행 수준의 진단 가능성을 제공함을 보여주는 핵심 증거다. **한계**: 소표본($n_{\text{add}}$가 2~46으로 작음)과 다중 비교 미보정으로 개별 수치의 신뢰도가 낮음.

---

## 8. 결론: 시사점, 후속 연구, 추가 방향

### 8-1. 모델의 일반화 성능 향상 가능성

**저자가 제시한 시사점**:
- 실행 경험을 재사용하면 단일 정적 구성보다 더 적응적인 하네스를 구축할 수 있다
- 하네스 제어 레이어 개선이 모델 스케일링과 수동 하네스 엔지니어링의 실용적 보완책이 될 수 있다

**저자가 제시한 후속 연구**:
1. 완전 비지도(unsupervised) 탐색 연구
2. 대규모 검증(larger-scale validation)
3. 세밀한 컴포넌트 귀인(component attribution)
4. 배포 간 온라인 경험 누적(online experience accumulation)

**일반화 성능 관련 구체 분석**:

MemoHarness의 일반화는 세 축에서 관찰된다:

① **크로스 데이터셋 일반화**: Terminal-Bench에서 학습한 하네스가 MMMLU, StrongReject, SWE-Pro에서 개선을 보인 것은 **장기-도구 중심 작업에서 학습된 제어 결정**(강건한 명령 추종, 소프트웨어 작업 구조)이 도메인을 초월하는 이식 가능성을 시사한다. 단, LawBench에서 혼재된 결과는 **도메인 특화된 제어 결정**은 전이 폭이 좁음을 보여준다.

② **크로스 모델 일반화**: 7개 모델 모두에서 개선이 나타난 것은 학습된 하네스가 특정 모델의 프롬프트 특성에 과적합되지 않았음을 시사한다. 그러나 GPT-4.1(+0.038)과 GLM-5(+0.233)의 큰 차이는 **기반 모델의 기존 보정 수준**이 하네스 효과의 크기를 결정하는 핵심 변수임을 보여준다.

③ **일반화의 근본 메커니즘**: 경험 뱅크가 단순 입출력 매핑이 아닌 **실행 패턴과 실패 진단**을 저장하기 때문에, 학습된 지식이 특정 태스크 표현이 아닌 하네스 제어 전략 수준에서 이식 가능하다는 가설이 그럴듯하다. 그러나 이를 검증하는 실험(예: 전역 패턴만 사용 vs. 케이스별 검색만 사용의 ablation)이 없어 현재 단계에서는 가설 수준에 머문다.

**일반화 향상을 위한 추가 연구 방향**:
- **메타 학습 통합**: MAML(Finn et al., 2017) 등 메타 학습 프레임워크와 결합하여 "빠른 적응"을 위한 하네스 초기화 학습
- **도메인 불변 패턴 추출**: 전역 패턴 $\mathcal{G}_t$에서 도메인 특화 vs. 도메인 불변 패턴을 분리하는 구조화된 증류 방법론
- **Few-shot 하네스 적응**: 새 도메인에서 5~10개 케이스만으로 빠르게 적응하는 메커니즘
- **Negative transfer 방지**: 소스 하네스가 타겟 도메인에서 성능을 저하시키는 경우(예: LawBench)를 사전 감지하는 게이팅 메커니즘

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 고지**: 아래 비교는 논문 내 인용 문헌과 필자의 배경 지식을 기반으로 하며, 일부 2026년 논문(Meta-Harness, Terminal-Bench 등)은 실제 게재 여부가 확인되지 않은 프리프린트입니다. 해당 논문들의 정확한 수치는 원문을 직접 확인하시기 바랍니다.

| 연구 | 연도 | 최적화 대상 | 적응 방식 | MemoHarness와의 관계 |
|------|------|-------------|-----------|----------------------|
| **ReAct** (Yao et al.) | 2023 | 추론+행동 통합 | 고정 | 에이전트 기반 마련, 하네스 최적화 없음 |
| **DSPy** (Khattab et al.) | 2023 | 선언적 파이프라인 | 훈련 시 컴파일 | 파이프라인 수준, 하네스 전체 미포함 |
| **Reflexion** (Shinn et al.) | 2023 | 언어 반성 | 반복 피드백 | 단일 에이전트 루프, 크로스 케이스 경험 없음 |
| **Self-Refine** (Madaan et al.) | 2023 | 출력 개선 | 자기 피드백 | 단일 프롬프트 수준 |
| **OPRO** (Yang et al.) | 2023 | 명령어 최적화 | LLM 최적화기 | 프롬프트만 최적화 |
| **AFlow** (Zhang et al.) | 2024 | 에이전트 워크플로우 코드 | 탐색+실행 피드백 | 워크플로우만, 하네스 전체 미포함 |
| **TextGrad** (Yuksekgonul et al.) | 2024 | 복합 AI 시스템 | 텍스트 역전파 | 시스템 수준이나 하네스 분해 없음 |
| **Meta-Harness** (Lee et al.) | 2026 | 하네스 코드 전체 | 훈련 시 탐색 | 가장 유사하나 테스트 시 적응 없음 |
| **Natural-Language Agent Harnesses** (Pan et al.) | 2026 | 자연어 하네스 아티팩트 | 공유 런타임 | 편집 가능성 강조, 경험 누적 없음 |
| **AlphaEvolve** (Novikov et al.) | 2025 | 코딩 에이전트 | 진화적 탐색 | 평가기 피드백 하 코드 진화, 하네스 미분리 |
| **MemoHarness** (본 논문) | 2026 | 하네스 전체 (6차원) | 훈련+테스트 시 적응 | **이중 레이어 경험 뱅크 + 케이스별 적응 최초 통합** |

**차별점 요약**: MemoHarness의 핵심 차별점은 (1) 하네스를 6차원으로 구조화하여 **진단 가능한 탐색**을 가능하게 하고, (2) 경험 뱅크를 통해 **크로스 케이스 지식을 이중 레이어**로 축적하며, (3) **테스트 시 레이블 없는 케이스별 적응**을 최초로 통합한 점이다.

### 향후 연구에 미치는 영향

1. **하네스 연구의 제도화**: 에이전트 하네스를 독립적 최적화 대상으로 확립함으로써, 향후 연구가 하네스 설계를 1급 시민(first-class citizen)으로 다루도록 유도
2. **경험 기반 적응의 패러다임**: 실행 경험을 재사용 가능한 구조화된 지식으로 변환하는 패러다임을 LLM 에이전트 연구에 도입
3. **평가 프로토콜의 중요성 부각**: 동일 기반 모델을 가정한 순수 하네스 비교를 위한 표준화된 평가 기준의 필요성 제기

### 앞으로 연구 시 고려할 점

1. **엄격한 통계 검증**: 더 큰 평가 세트, 반복 실험, 신뢰구간 보고가 필수적
2. **공정한 기준선 설정**: 기반 모델을 통제한 순수 하네스 효과 측정 프로토콜 확립
3. **컴포넌트 격리**: 6차원 각각과 경험 뱅크 두 레이어의 개별 기여도 ablation
4. **캐싱 의존도**: 실제 배포 환경에서 캐싱률이 다를 경우의 비용 분석
5. **학습 효율**: 10회 반복으로 충분한지, 조기 종료 조건 및 수렴 기준 연구
6. **보안 고려**: 경험 뱅크가 민감한 실행 정보를 저장할 때의 프라이버시·보안 이슈
7. **적대적 케이스 강건성**: 경험 뱅크를 오염시키는 적대적 케이스에 대한 강건성

---

## 참고자료 (논문 내 인용 기준)

본 분석은 다음 자료에 기반합니다:

**Primary Source**:
- Huang, Y., Wang, W., Bao, H., et al. (2026). *MemoHarness: Agent Harnesses That Learn from Experience*. arXiv:2607.14159v1.

**논문 내 인용 주요 참고문헌**:
- Fernando, C., et al. (2023). *Promptbreeder*. arXiv:2309.16797
- Khattab, O., et al. (2023). *DSPy*. arXiv:2310.03714
- Lee, Y., et al. (2026). *Meta-Harness*. arXiv:2603.28052
- Madaan, A., et al. (2023). *Self-Refine*. arXiv:2303.17651
- Merrill, M. A., et al. (2026). *Terminal-Bench*. arXiv:2601.11868
- Novikov, A., et al. (2025). *AlphaEvolve*. arXiv:2506.13131
- Opsahl-Ong, K., et al. (2024). *MIPRO*. arXiv:2406.11695
- Pan, L., et al. (2026). *Natural-Language Agent Harnesses*. arXiv:2603.25723
- Pryzant, R., et al. (2023). *ProTeGi*. arXiv:2305.03495
- Schick, T., et al. (2023). *Toolformer*. arXiv:2302.04761
- Shinn, N., et al. (2023). *Reflexion*. arXiv:2303.11366
- Yang, C., et al. (2023). *OPRO*. arXiv:2309.03409
- Yang, J., et al. (2024). *SWE-Agent*. arXiv:2405.15793
- Yao, S., et al. (2023a). *Tree of Thoughts*. arXiv:2305.10601
- Yao, S., et al. (2023b). *ReAct*. arXiv:2210.03629
- Yuksekgonul, M., et al. (2024). *TextGrad*. arXiv:2406.07496
- Zhang, J., et al. (2024). *AFlow*. arXiv:2410.10762
- Zhou, A., et al. (2023). *LATS*. arXiv:2310.04406
- Anthropic. (2024). *Building Effective Agents*. Anthropic Engineering Blog
- LangChain. (2025). *Context Engineering for Agents*. LangChain Blog
- Lopopolo, R. (2026). *Harness Engineering*. OpenAI Engineering Blog
- GitHub Repository: https://github.com/HowieHwong/MemoHarness
