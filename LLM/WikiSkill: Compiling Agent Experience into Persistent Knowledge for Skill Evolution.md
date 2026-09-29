# WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution

> **⚠️ 정확도 고지**: 본 논문은 arXiv:2608.27454v1 (2026-08-27)에 공개된 실제 문서를 기반으로 작성되었습니다. 참조된 일부 인용 문헌(2026년 발행)은 본 논문과 함께 공개된 최신 연구들이므로, 해당 문헌들의 세부 내용은 직접 확인이 필요합니다.

---

## 1. Executive Summary (10문장 이내)

WikiSkill은 AI 에이전트가 실행 경험으로부터 **영속적 지식 베이스(Wiki)** 를 구축하고, 이를 바탕으로 재사용 가능한 스킬을 진화시키는 프레임워크다.  
기존 스킬 진화 방법(EvoSkill, Trace2Skill, SkillOpt)이 학습된 통찰을 별도의 지식 표현으로 유지하지 않는 문제를 해결한다.  
WikiSkill은 실행 트레이스(Raw Layer), 구조화된 지식(Wiki Layer), 실행 가능한 절차(Skill Layer)라는 3계층 구조로 작업 공간을 분리한다.  
매 이터레이션마다 추론 에이전트가 롤아웃을 수행하고, Wiki Maintainer가 패턴을 집약하며, Skill Proposer가 업데이트를 제안하고, Gating 메커니즘이 검증 성능 기반으로 채택 여부를 결정한다.  
5개 벤치마크(LiveMath, SealQA, SpreadSheet, OfficeQA, ALFWorld)와 5개 모델(Qwen 계열, Gemma, Gemini)에서 기존 방법 대비 일관된 성능 향상을 달성했다.  
스킬 진화의 이점은 모델 스케일과 상보적으로 작용하여, 더 큰 모델일수록 WikiSkill의 혜택이 더 크다.  
소형 모델에 WikiSkill 적용 시 스킬 없는 대형 모델을 능가하는 결과도 확인되었다.  
진화된 스킬은 모델 계열을 넘어 전이되며, 경우에 따라 자기 진화 스킬보다 우수한 성능을 보인다.  
Ablation 연구를 통해 Wiki의 영속적 지식 누적이 성능 향상의 핵심 요인임이 확인되었다.

### 1-1. 연구의 목적과 필요성

**목적**: LLM 에이전트가 실행 경험으로부터 체계적으로 지식을 축적하고, 이를 재사용 가능한 절차적 스킬로 변환하는 지속적·자동화된 메커니즘 구축.

**필요성**:
- 실제 세계 작업에는 도메인 특화된 절차적 지식이 필요하나, 대부분의 에이전트 스킬은 수작업으로 제작됨 (p.2)
- 기존 자동화 방법(EvoSkill 등)은 최적화 이력에서 얻은 통찰이 이터레이션 간에 체계적으로 재활용되지 않고 분산됨 (p.2)
- 모델 파라미터를 업데이트하지 않고도 지식을 축적할 수 있는 경량화된 메커니즘의 필요성 (p.2)

> **💡 용어 설명**
> - **에이전트 스킬(Agent Skill)**: LLM 기반 에이전트가 사용할 수 있도록 도메인 특화 지식, 지침, 스크립트를 파일시스템 기반 모듈(디렉토리)로 패키징한 재사용 가능한 자원. 모델 가중치를 변경하지 않고 에이전트 능력을 확장함.
> - **절차적 지식(Procedural Knowledge)**: "무엇을 아는가"가 아닌 "어떻게 하는가"에 관한 지식. 예: 특정 도구를 사용하는 순서, 오류 복구 방법 등.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 (데이터/실험) | 위치 |
|---|-----------|-------------------|------|
| 1 | WikiSkill은 기존 스킬 진화 방법(EvoSkill, SkillOpt, Trace2Skill)보다 일관되게 우수 | 5개 모델 전체에서 평균 성능 최고치 달성; 경쟁 방법 대비 +3.3~+12.0 포인트 향상 | Table 1, p.7~8 |
| 2 | 스킬 진화 이점은 모델 스케일과 상보적 | Qwen 계열에서 WikiSkill 향상 폭: 4B→+12.3%, 9B→+17.5%, 27B→+23.9% | p.3, Table 1 |
| 3 | 작은 모델+스킬이 큰 모델을 능가 | Qwen-3.5-9B+WikiSkill(47.4%) > Qwen-3.6-27B 스킬 없음(39.4%) | p.3, Table 1 |
| 4 | 진화된 스킬은 모델 계열 간 전이 가능, 때로 자기 진화 스킬보다 우수 | ALFWorld: Qwen-3.5-9B가 자체 스킬(63.4%)보다 27B 스킬(70.2%)로 더 우수 | Table 2, p.9 |
| 5 | 영속적 Wiki 지식 누적이 성능의 핵심 요인 | Proposer에 Wiki 제공 시 평균 +15.0% (48.7%→63.7%) | Table 3, p.11 |
| 6 | 추론 에이전트에게 Wiki 접근 부여는 역효과 | Wiki 접근 시 평균 63.7%→60.9%, LiveMath 72.6%→64.8% | Table 3, p.11 |
| 7 | 스킬 발견 능력과 스킬 실행 능력은 분리된 역량 | OfficeQA: Qwen-3.5-4B 자체 스킬 사용 시 성능 저하(-1.7%), 27B에 적용 시 +10.8% | Table 2, p.10 |

---

## 2-1. 해결 문제·제안 방법·모델 구조·성능·한계 상세 설명

### A. 해결하고자 하는 문제

기존 스킬 진화 방법들은 에이전트가 실행 경험에서 얻은 통찰을 **별도의 진화하는 지식 표현**으로 유지하지 않는다. 최적화 이력 전반에 통찰이 분산되어, 이터레이션 간 체계적 재사용이 불가능하다. (p.2)

> **💡 용어 설명**
> - **롤아웃(Rollout)**: 에이전트가 특정 정책(스킬)을 사용하여 실제 태스크를 처음부터 끝까지 실행하는 과정. 이 과정에서 생성된 실행 이력(트레이스)이 학습 데이터가 됨.

### B. 제안하는 방법 (수식 포함)

**문제 형식화** (p.3):
데이터셋 $\mathcal{D} = \{(x_i, y_i)\}\_{i=1}^{N}$에서 $x_i$는 태스크, $y_i$는 정답. 이를 훈련( $\mathcal{D}\_\text{train}$), 검증($\mathcal{D}\_\text{val}$), 테스트($\mathcal{D}_\text{test}$ )로 분할. 에이전트 $\pi$는 도구 집합 $\mathcal{U}$와 스킬 집합 $S = \{s_1, \ldots, s_M\}$을 보유.

시스템 상태는 이터레이션 $k$에서 튜플 $(S_k, W_k)$로 표현. $S_k$는 활성 스킬 집합, $W_k$는 영속적 지식 베이스(Wiki).

> **💡 용어 설명**
> - **스킬 집합 $S$**: 현재 에이전트가 사용 가능한 모든 스킬의 집합. 초기값은 공집합 $\emptyset$.
> - **Wiki $W_k$**: 누적된 패턴, 실패 분석, 진화 로그 등 구조화된 지식을 담는 영속적 저장소.

**수식 1: 추론 에이전트 실행** (p.5, Eq.1):

$$\tau_i \sim \pi(x_i; S_{k-1})$$

- $\tau_i$: 이터레이션 $k$에서 태스크 $x_i$에 대한 실행 트레이스
- $\pi$: LLM 기반 추론 에이전트
- $S_{k-1}$: 이전 이터레이션의 활성 스킬 집합
- 트레이스 $\tau_i = (o_1, a_1, o_2, a_2, \ldots, o_T, a_T)$: 관찰($o_t$)과 행동($a_t$)의 시퀀스

> **💡 용어 설명**
> - **트레이스(Trace) $\tau_i$**: 에이전트가 하나의 태스크를 수행하는 동안 발생한 관찰(환경 피드백)과 행동(도구 호출, 추론 등)의 전체 기록.

**수식 2: Wiki Maintainer 패턴 집약** (p.6, Eq.2):

$$W'_k \leftarrow \mathcal{M}_\text{WM}(W_{k-1}, \mathcal{T}_\text{sample,k})$$

- $\mathcal{M}_\text{WM}$: Wiki Maintainer 에이전트 (LLM 기반)
- $W_{k-1}$: 이전 이터레이션의 Wiki 상태
- $\mathcal{T}\_\text{sample,k} \subset \mathcal{T}_\text{train,k}$: 현재 이터레이션의 샘플링된 트레이스 부분집합 (최대 8개: 실패 5개 + 성공 3개)
- $W'_k$: 중간 Wiki 상태 (아직 스킬 제안 결과 반영 전)

**수식 3: Wiki 기반 스킬 제안** (p.6, Eq.3):

$$P_k \leftarrow \mathcal{M}_\text{P}(W'_k, S_{k-1}, \mathcal{T}_\text{train,k})$$

- $\mathcal{M}_\text{P}$: Skill Proposer 에이전트 (ReAct 방식으로 동작)
- $P_k$: 이터레이션 $k$에서의 스킬 제안 (단일 스킬 생성 또는 수정)
- ReAct 방식: 추론(Reasoning)과 행동(Acting)을 교차 반복하는 다중 턴 에이전트 패턴

> **💡 용어 설명**
> - **ReAct(Reasoning + Acting)**: Yao et al.(2023)이 제안한 LLM 에이전트 패러다임. 에이전트가 "생각 → 행동 → 관찰"을 반복하며 문제를 해결. WikiSkill의 Skill Proposer는 이 방식으로 Wiki를 탐색하고 스킬을 제안함.

**수식 4: Gating & Rollback** (p.6, Eq.4):

$$S_k \leftarrow \begin{cases} S'_k & \text{if } \mathcal{R}(\mathcal{T}_\text{val,k}) > \mathcal{R}_\text{best} \\ S_{k-1} & \text{otherwise} \end{cases}$$

- $S'\_k = \text{Apply}(S_{k-1}, P_k)$: 제안된 스킬이 적용된 후보 스킬 집합
- $\mathcal{R}(\mathcal{T}_\text{val,k})$: 검증 분할에서의 성능 점수
- $\mathcal{R}_\text{best}$: 현재까지의 최고 검증 성능 임계값
- Wiki $W_k$는 채택 여부에 관계없이 절대 롤백되지 않음

> **💡 용어 설명**
> - **Gating & Rollback**: 새로운 스킬 제안이 검증 성능을 향상시킬 때만 채택하고, 그렇지 않으면 이전 스킬로 되돌리는 안전장치. Wiki는 롤백되지 않아 지식은 항상 누적됨.

**수식 5: WikiSkill API 호출 복잡도** (p.22, Eq.5):

$$\mathcal{C}_\text{WikiSkill} = (1 + T_\text{ReAct})\frac{N_\text{train}}{B}$$

- $T_\text{ReAct}$: Skill Proposer의 ReAct 추론 턴 수 ($10 \le T_\text{ReAct} \le 20$)
- $N_\text{train}$: 훈련 태스크 수
- $B$: 배치 크기 (WikiSkill에서는 $B = N_\text{train}$으로 설정)
- 전체 배치 모드에서: $\mathcal{C}\_\text{WikiSkill} = 1 + T_\text{ReAct}$ (훈련 데이터 크기에 무관한 $O(1)$ )

**타 방법 복잡도 비교** (p.22~23):

$$\mathcal{C}_\text{EvoSkill} = \frac{2N_\text{train}}{B} \quad [O(N_\text{train}/B)]$$

$$\mathcal{C}_\text{SkillOpt} = \frac{K_\text{opt} \cdot N_\text{train}}{B} \quad [O(N_\text{train}/B)]$$

$$\mathcal{C}_\text{Trace2Skill} \approx N_\text{train} + \left(1 + \frac{1}{c-1}\right)\frac{N_\text{train}}{B} + 1 \quad [O(N_\text{train})]$$

- $K_\text{opt}$: SkillOpt의 단계당 반영·병합 호출 수 ( $\approx 6$ ~ $8$ )
- $c$: Trace2Skill의 감소 트리 분기 계수

### C. 모델 구조 (3계층 아키텍처 + 4요소 루프)

```
┌─────────────────────────────────────────────────┐
│          WikiSkill 프레임워크                    │
│                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │  Raw Layer  │  │ Wiki Layer  │  │  Skill   │ │
│  │  (raw/)     │→ │  (wiki/)    │→ │  Layer   │ │
│  │  불변 트레이스│  │  영속 지식  │  │(skills/) │ │
│  │  (Write Once)│  │(Never Reset)│  │(조건부 업)│ │
│  └─────────────┘  └─────────────┘  └──────────┘ │
│                                                  │
│  Step 1: Inference Agent (스킬 주입, 롤아웃)     │
│  Step 2: Wiki Maintainer (트레이스 → 패턴 집약) │
│  Step 3: Skill Proposer (Wiki 기반 스킬 제안)   │
│  Step 4: Gating & Rollback (검증 기반 채택 결정)│
└─────────────────────────────────────────────────┘
```

**Wiki Layer 내부 구조**:
- `wiki/patterns/`: 실패 모드·성공 전략 문서 (마크다운)
- `wiki/logs.md`: 연대순 진화 로그
- `wiki/skill-impact.md`: 스킬 제안 이력 및 채택 결과
- `wiki/index.md`: 패턴 카탈로그 색인

### D. 성능 향상

| 모델 | 기존 최고(평균%) | WikiSkill(평균%) | 향상 |
|------|----------------|-----------------|------|
| Qwen-3.5-4B | 35.2 (SkillOpt) | 38.5 | +3.3 |
| Qwen-3.5-9B | 42.3 (EvoSkill) | 47.4 | +5.1 |
| Qwen-3.6-27B | 53.3 (EvoSkill) | 63.3 | +10.0 |
| Gemma-4-31B | 49.1 (SkillOpt) | 54.9 | +5.8 |
| Gemini-3.5-Flash | 56.1 (EvoSkill) | 68.1 | +12.0 |

*(출처: Table 1, p.8)*

### E. 한계 (p.14)

1. **스킬 검색/트리거 미평가**: 현 연구는 스킬 품질 격리를 위해 모든 스킬을 프롬프트에 직접 주입. 스킬 수가 증가하면 검색 메커니즘이 필요.
2. **엄격한 Gating 기준**: 검증 점수를 즉시 향상시키지 않는 "중립적" 제안은 거부. 이후 이터레이션에 도움이 될 수 있는 제안도 배제될 가능성.
3. **Wiki 가지치기 부재**: Wiki가 지속적으로 누적되나 자동 정리 메커니즘 없음. 장기 실행 시 비대해질 수 있음.
4. **매우 장기 태스크 미지원**: 수백 개 환경 행동이나 수 시간에 걸친 장기 태스크는 벤치마크에 포함되지 않음.

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| WikiSkill 프레임워크 개요 및 3계층 구조 | Figure 2, p.4 |
| WikiSkill 전체 성능 비교 (5모델 × 5벤치마크) | Table 1, p.8 |
| 모델 스케일별 성능 비교 그래프 | Figure 1, p.1 |
| 교차 모델 스킬 전이 결과 | Table 2, p.10 |
| Wiki 접근 Ablation 연구 | Table 3, p.11 |
| Wiki-가이드 스킬 진화 사례 연구 (ALFWorld) | Figure 3, p.12 |
| 스킬 및 Wiki 패턴 통계 | Table 4, p.13 |
| 이터레이션별 스킬 업데이트 수용 분포 | Table 5, p.20 |
| WikiSkill 진화 루프 알고리즘 | Algorithm 1, p.19 |
| API 호출 복잡도 비교 | Table 7, p.22 |
| 벤치마크 통계 및 분할 | Table 6, p.20 |

---

## 4. 저자 직접 보고 vs. 내 해석 분리

### 저자가 직접 보고한 내용

- WikiSkill이 5개 모델 전체에서 평균 성능 최고치를 달성했음 (Table 1)
- Qwen 계열에서 모델 크기에 따른 WikiSkill 향상 폭: 4B→+12.3%, 9B→+17.5%, 27B→+23.9% (p.3)
- Qwen-3.5-9B+WikiSkill(47.4%) > Qwen-3.6-27B 스킬 없음(39.4%) (p.3)
- Proposer에 Wiki 제공 시 평균 +15.0% (48.7%→63.7%) (Table 3)
- Wiki 접근을 추론 에이전트에게 제공 시 성능 저하: 63.7%→60.9% (Table 3)
- Gemini-3.5-Flash의 LiveMath: 33.0%→72.6%, SpreadSheet: 50.5%→76.6% (Table 1)
- 전체 배치 모드에서 WikiSkill의 옵티마이저 API 호출 복잡도는 $O(1)$ (p.22)
- 3회 독립 실행 평균 및 paired bootstrap test ($p < 0.05$) 사용 (p.7)

### 내 해석

- **모델 스케일 상보성의 의미**: 저자들은 큰 모델이 스킬을 더 잘 활용함을 보이나, 이는 단순히 큰 모델의 instruction-following 능력이 더 뛰어나기 때문일 수 있음. 즉, WikiSkill이 더 나은 스킬을 생성하는 것과 더 큰 모델이 동일한 스킬을 더 잘 따르는 것이 혼재될 수 있음.
- **음의 전이(Negative Transfer)의 원인**: 저자들은 Qwen-3.5-4B 스킬이 Gemini-3.5-Flash에 부정적 영향을 준다고 분석하나(SpreadSheet: 50.5%→18.1%), 이는 스킬 내용의 모델 특수성뿐 아니라 컨텍스트 윈도우 활용 패턴의 차이도 원인일 수 있음.
- **Wiki 영속성의 양면성**: Wiki가 누적됨으로써 장기 이득이 있으나, 오래된 잘못된 패턴이 삭제되지 않을 경우 오히려 방해가 될 가능성이 있음. 이는 저자들도 한계로 인정하나 실험적 근거는 제시하지 않음.
- **3회 반복의 통계적 충분성**: 저자들은 3회 실행 평균을 사용하나, 이는 분산 추정에 통계적으로 제한적임 (§5 참조).

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적으로 취약한 부분

| 항목 | 취약점 |
|------|--------|
| **3회 반복 평균** | 실험당 3회 실행은 분산 추정에 통계적으로 제한적. 특히 OfficeQA(24개 검증 샘플)처럼 소규모 검증셋에서는 gating 결정에 상당한 노이즈가 존재할 수 있음 (p.20). |
| **소규모 검증셋** | SealQA(10개), ALFWorld(18개), LiveMath(18개) 검증 분할은 매우 소규모로, gating 결정의 신뢰도가 낮을 수 있음 (Table 6, p.20). |
| **Gemini-3.5-Flash ALFWorld 결과** | 모든 방법이 85.9%로 동일하여 Gemini의 사전 포화(100% 검증 점수) 때문. 이 조건에서의 비교는 의미가 없음 (p.7). |
| **교차 모델 전이 실험** | Table 2에서 일부 소스 모델 조합(Gemini-3.5-Flash → ALFWorld)은 "-"로 표기되어 비교 불완전 (p.10). |

### ⚠️ 비교 불가능한 수치

| 항목 | 이유 |
|------|------|
| **WikiSkill vs. GEPA 등 일반 프롬프트 최적화기** | 저자들이 의도적으로 비교를 제외. 비교가 불공평하다는 prior work의 관점을 따름 (p.7). 실제로는 얼마나 차이가 나는지 불명확. |
| **스킬 길이 vs. 성능 상관관계** | Table 4에서 모델별 스킬 길이 차이(45.1~128.6 줄)가 보고되나, 길이와 성능 간 직접 분석 없음. |
| **API 호출 복잡도와 실제 비용** | Table 7에서 이론적 복잡도를 비교하나, 실제 토큰 비용(LLM 출력 길이)이나 레이턴시는 보고되지 않음. ReAct 턴 수($T_\text{ReAct}$)가 10~20으로 넓은 범위여서 실제 비용 예측이 어려움. |

---

## 6. 문서가 답하지 않는 질문

1. **스킬 검색(Retrieval) 성능**: 현재는 모든 스킬을 프롬프트에 직접 주입하는 방식으로, 스킬 수가 수십~수백 개로 증가할 때 WikiSkill이 어떻게 작동하는지 실험 없음.

2. **Wiki 가지치기(Pruning) 전략**: Wiki가 무한히 누적될 때의 성능 변화나 가지치기 방법에 대한 분석 없음.

3. **WikiSkill의 실시간/온라인 적용 가능성**: 현재 프레임워크는 오프라인 이터레이션 방식. 단일 장기 실행 중 실시간 스킬 적응 방법은 미제시.

4. **Wiki 품질의 정량적 평가**: Wiki에 누적된 패턴의 품질(정확성, 관련성)을 정량적으로 평가하는 방법이 없음. 잘못된 패턴이 누적될 경우의 영향 미분석.

5. **계산 비용 세부 분석**: 실제 토큰 소비량, GPU 시간, 금전적 비용 비교 부재. $T_\text{ReAct}$의 변동(10~20)이 실제 비용에 미치는 영향 불명확.

6. **타 도메인 일반화**: 5개 벤치마크 외 다른 도메인(코드 생성, 의료, 법률 등)에서의 성능 미검증.

7. **Wiki 접근 제한의 최적 구성**: 추론 에이전트의 Wiki 접근이 해로운 이유에 대한 정성적 분석이 가설(§5.1)로만 제시되고 실험적으로 더 깊이 검증되지 않음.

8. **스킬 진화 중단 기준**: $\mathcal{R}_\text{best} = 1.0$이 아닌 다른 조기 종료 기준이나 수렴 판단 방법 미제시.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1): 모델 스케일별 평균 성능 비교

```
정확도(%)
75% │                                    ◆ WikiSkill
    │                               ▲ SkillOpt
60% │                         ■ EvoSkill
    │                    ● No skill
45% │
    │
30% │
    └────────────────────────────────────────
       Qwen3.5-4B  Qwen3.5-9B  Qwen3.6-27B  Gemini 3.5 Flash
```

**해석**: 4개 모델에서 WikiSkill(◆)이 일관되게 최상위를 유지하며, 모델이 커질수록 WikiSkill과 No skill 간의 격차가 급격히 벌어짐. 이는 스킬 진화가 단순히 명령을 따르는 것이 아니라, 복잡한 절차를 이해하고 실행하는 능력이 필요함을 시사. Gemini-3.5-Flash에서 WikiSkill이 약 75%에 달해 두드러진 우위를 보임.

### Figure 2 (p.4): WikiSkill 프레임워크 개요

**해석**: 3계층(Raw/Wiki/Skill)과 4단계(Inference/Wiki Maintainer/Skill Proposer/Gating) 루프의 시각적 표현. 핵심은 화살표 방향으로, 정보가 Raw → Wiki → Skill로 단방향이 아니라 순환하며 누적됨을 보여줌. "Compounding, Never Reset"(Wiki)과 "Reversible, Conditional Update"(Skill)의 대비가 프레임워크의 핵심 설계 철학을 명확히 전달함. 추론 에이전트가 Wiki에 접근하지 않는 설계가 명시적으로 표현되어 있어, Ablation(Table 3)의 근거를 시각적으로 확인 가능.

### Figure 3 (p.12): ALFWorld 사례 연구 - Wiki 가이드 스킬 진화

**해석**: Qwen-3.6-27B의 ALFWorld에서 4번의 이터레이션에 걸친 스킬 진화 과정을 상세히 보여줌. 핵심 통찰:
- **Iteration 0**: `goal-directed-action` 스킬 제안 → 검증에서 거부(val score=0.72). Wiki의 `skill-impact.md`가 이 실패를 기록.
- **Iteration 1**: 이전 거부 이력을 참조하여 더 구체적인 `break-repetition-loop` 제안 → 채택(val score=0.78). Wiki가 없었다면 유사한 실수를 반복했을 가능성 높음.
- **Iteration 4**: 새로운 루프 패턴(`multi-operation-loop.md`)이 누적되어 스킬 추가 개선.
이 그림은 Wiki의 audit trail(감사 추적)이 어떻게 중복 실수를 방지하고 점진적 개선을 유도하는지 가장 명확하게 보여주는 근거.

> **💡 용어 설명**
> - **감사 추적(Audit Trail)**: 과거에 무엇이 시도되었고, 왜 거부/채택되었는지에 대한 완전한 기록. WikiSkill에서는 `skill-impact.md`가 이 역할을 담당.

### Table 1 (p.8): 5모델 × 5벤치마크 성능 비교

**해석**: 본 논문의 핵심 결과 테이블. 주목할 점:
- **일관성**: WikiSkill이 5개 모델 전체 평균에서 1위. 기존 방법들은 특정 모델/벤치마크 조합에서 성능 저하를 보임(EvoSkill: Gemma LiveMath 33.9%→29.8%).
- **도메인 의존성**: SpreadSheet에서의 향상폭이 가장 큼(27B: +40.9%), OfficeQA에서 가장 작음. 이는 절차적 코딩 지식이 문서 탐색 지식보다 스킬로 표현되기 쉬움을 시사.
- **소형 모델의 한계**: Qwen-3.5-4B에서 WikiSkill이 OfficeQA를 오히려 저하(30.2%→28.5%). 소형 모델은 복잡한 다단계 검색 지침을 따르는 능력 자체가 부족.

### Table 3 (p.11): Wiki 접근 Ablation 연구

**해석**: WikiSkill의 핵심 설계 결정을 정당화하는 가장 중요한 Ablation.

| Inference Agent Wiki | Skill Proposer Wiki | 평균 |
|---------------------|---------------------|------|
| ✗ | ✗ | 48.7% |
| ✓ | ✗ | 45.3% |
| ✗ | ✓ | **63.7%** (기본 설정) |
| ✓ | ✓ | 60.9% |

두 가지 핵심 발견:
1. Skill Proposer에 Wiki 제공: +15.0% (48.7%→63.7%) — Wiki의 가치 증명
2. Inference Agent에 Wiki 추가 제공: -2.8% (63.7%→60.9%) — Wiki가 스킬을 대체하면 트레이스 품질 저하

이는 "정보가 많을수록 좋다"는 직관에 반하는 결과로, 역할 분리(Role Separation)의 중요성을 강조함.

---

## 8. 결론: 시사점·후속 연구 계획·추가 방향

### 저자 제시 시사점 (p.14)

1. **지식 누적의 근본적 중요성**: 스킬 진화에서 영속적 지식 베이스가 핵심 요인임을 실험적으로 입증.
2. **스킬 발견과 실행의 분리**: 스킬을 잘 만드는 능력과 스킬을 잘 활용하는 능력은 서로 다른 역량임.
3. **스킬 진화와 모델 스케일링의 상보성**: 하드웨어 확장 없이도 스킬 진화만으로 더 큰 모델과 경쟁 가능.

### 저자 제시 후속 연구 방향 (p.14)

- 스킬 수 증가에 따른 **스킬 검색/트리거** 메커니즘 연구
- **유연한 수용 기준**: 즉각적 성능 향상 없이도 미래 이터레이션에 도움이 되는 제안 허용
- **Wiki 자동 가지치기** 메커니즘 개발
- **온라인 스킬 적응**: 단일 장기 실행 내 실시간 절차적 지식 개선

### 8-1. 모델 일반화 성능 향상 가능성

WikiSkill의 **교차 모델 스킬 전이** 결과(Table 2)는 일반화 성능에 대한 중요한 함의를 가짐:

**긍정적 시사점**:
- Qwen-3.5-4B가 진화시킨 스킬이 Gemma-4-31B의 LiveMath를 56.7%→73.1%로 향상. 작은 모델의 경험이 더 강력한 모델에 전이 가능.
- LiveMath 스킬은 모델 계열 간(Qwen→Gemini) 전이가 잘 되어 **도메인 보편적 절차 지식**을 캡처함을 시사.

**한계 및 도전**:
- SpreadSheet처럼 **모델 특수적 우회전략(Workaround)** 을 포함하는 스킬은 전이 시 성능 저하 발생(4B→Gemini: 50.5%→18.1%)
- OfficeQA처럼 장문 컨텍스트 처리 능력이 필요한 경우, 소형 모델이 생성한 스킬은 소형 모델 자체도 실행하지 못함

**일반화 향상을 위한 제안**:
1. **스킬 추상화 수준 자동 조절**: 스킬 생성 시 모델 독립적(high-level) 절차와 모델 특수적(low-level) 세부사항을 분리하여 저장
2. **전이 가능성 점수(Transferability Score)**: 생성된 스킬이 얼마나 범용적인지를 자동 평가하는 메트릭 도입
3. **다중 모델 앙상블 스킬 진화**: 여러 모델의 경험을 통합하여 보편적 스킬 생성

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 WikiSkill 논문(2026년)에 인용된 문헌들을 중심으로 구성하였습니다. 2026년 발행 문헌들은 본 논문과 동시기 연구로, 해당 논문들의 상세 결과는 직접 확인이 필요합니다.

#### 관련 연구 계보 및 비교

**1. 프롬프트 최적화 계열**

| 연구 | 방법 | WikiSkill과의 차이 |
|------|------|-------------------|
| DSPy/OPRO (2023~2024) | 자연어 피드백으로 프롬프트 최적화 | 스킬이 아닌 단일 프롬프트 최적화, 영속 지식 없음 |
| GEPA (Agrawal et al., 2026) | 반영적 프롬프트 진화 | 범용 프롬프트 최적화; WikiSkill은 재사용 가능 스킬에 특화 |
| TextGrad/ProTeGi (Yuksekgonul et al., 2025) | LLM 피드백 역전파 | 단일 모듈 최적화; WikiSkill은 에이전트 워크플로우 전체 최적화 |

**2. 에이전트 스킬 진화 계열 (직접 비교)**

| 연구 | 핵심 메커니즘 | WikiSkill 대비 |
|------|--------------|---------------|
| EvoSkill (Alzubi et al., 2026) | 실패 트레이스 + 평면적 이력 | 영속 Wiki 없음; 특정 모델에서 성능 저하 발생 |
| Trace2Skill (Ni et al., 2026) | 병렬 트레이스 분석 + 계층적 병합 | O($N_\text{train}$) 복잡도; Wiki 없음 |
| SkillOpt (Yang et al., 2026) | 6단계 ReflACT 파이프라인 | Wiki 없음; 선형 확장 복잡도 |
| SkillRL/Skill1 (Xia/Shi et al., 2026) | 강화학습 기반 스킬 내재화 | 모델 파라미터 업데이트 필요; WikiSkill은 파라미터 동결 |

**3. 메모리·지식 누적 계열**

| 연구 | 방법 | WikiSkill과의 관련성 |
|------|------|-------------------|
| MemGPT (2023) | 계층적 메모리 관리 | 단기/장기 메모리 분리; WikiSkill의 3계층과 유사한 철학 |
| Voyager (Wang et al., 2023) | Minecraft에서 기술 라이브러리 구축 | 게임 환경 특화; WikiSkill은 범용 에이전트에 적용 |
| Skill0 (Lu et al., 2026) | 인-컨텍스트 강화학습으로 스킬 내재화 | 스킬을 모델 내부로 통합 시도; WikiSkill은 외부 스킬 유지 |

#### WikiSkill이 앞으로의 연구에 미치는 영향

1. **지식 계층화의 표준화**: Raw-Wiki-Skill의 3계층 분리는 에이전트 지식 관리의 새로운 패러다임을 제시. 향후 에이전트 프레임워크 설계에 영향 예상.

2. **스킬 발견과 실행의 분리**: 이 개념은 에이전트 능력 평가 체계에 영향을 줄 수 있음. "스킬을 잘 만드는 모델"과 "스킬을 잘 활용하는 모델"을 별도로 평가하는 벤치마크 개발 가능성.

3. **경험 기반 지식 컴파일**: Karpathy의 "LLM Wiki" 개념을 실제 에이전트 시스템에 구현한 첫 사례로, 경험→지식→절차의 변환 파이프라인 연구를 촉진할 전망.

#### 앞으로 연구 시 고려할 점

1. **컨텍스트 창 한계**: 스킬과 Wiki가 모두 프롬프트에 주입되므로, 스킬과 패턴이 많아질수록 컨텍스트 창 포화 문제 발생. 선택적 로딩(Progressive Disclosure)과의 통합 필요.

2. **Wiki 신뢰성 검증**: LLM이 생성한 패턴의 정확성을 어떻게 보장할 것인가? 잘못된 패턴이 누적되면 스킬 품질 저하로 이어질 위험.

3. **멀티-에이전트 확장**: 여러 에이전트가 동시에 Wiki를 업데이트할 때의 충돌 해결 메커니즘 필요.

4. **도메인 전이 학습**: 하나의 도메인에서 진화된 Wiki 지식이 관련 도메인으로 얼마나 전이되는지 체계적 연구 필요.

5. **스킬 버전 관리(Version Control)**: 스킬이 롤백될 때 이전 버전과의 의미적 차이를 추적하여, 향후 동일한 실수를 방지하는 메커니즘 개발.

6. **비용-성능 트레이드오프의 명시적 분석**: 스킬 진화에 투입되는 LLM API 비용 대비 성능 향상 ROI(투자 대비 수익률)의 정량적 분석 필요. 특히 소규모 배포 환경에서의 실용성 검토.

---

## 참고 자료

**본 답변에서 직접 참조한 자료**:
- **Tang, L. et al. (2026).** WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution. *arXiv:2608.27454v1*. https://arxiv.org/abs/2608.27454

**논문 내 인용된 주요 관련 연구**:
- Alzubi et al. (2026). EvoSkill: Automated Skill Discovery for Multi-Agent Systems. arXiv:2603.02766
- Ni et al. (2026). Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills. arXiv:2603.25158
- Yang et al. (2026). SkillOpt: Executive Strategy for Self-Evolving Agent Skills. arXiv:2605.23904
- Yao et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023
- Ma et al. (2024). SpreadsheetBench: Towards Challenging Real World Spreadsheet Manipulation. NeurIPS 2024
- Shridhar et al. (2021). ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. ICLR 2021
- Yuksekgonul et al. (2025). Optimizing Generative AI by Backpropagating Language Model Feedback. *Nature*, 639(8055):609–616
- Karpathy, A. (2026). LLM Wiki. GitHub Gist. https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- Agrawal et al. (2026). GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. ICLR 2026
- Singhvi et al. (2025). Introducing OfficeQA. Databricks Blog
- Pham et al. (2026). SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models. ICLR 2026
