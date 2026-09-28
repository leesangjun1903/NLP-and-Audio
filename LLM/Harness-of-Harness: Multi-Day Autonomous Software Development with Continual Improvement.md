# Harness-of-Harness: Multi-Day Autonomous Software Development with Continual Improvement

> **⚠️ 중요 고지**: 이 논문은 arXiv:2609.01481v1 (2026년 9월 1일)로 제출된 **미래 날짜 논문**입니다. 본 분석은 제공된 PDF 원문에만 근거하며, 논문 외부의 추가 검증은 불가능합니다. 논문에 등장하는 일부 모델명(GPT-5.5, DeepSeek-V4-Pro 등)은 작성 시점(2025년 기준)에서 확인 불가한 미래 모델입니다.

---

## 1. Executive Summary (10문장 이내)

**Harness-of-Harness(HoH)** 는 LLM 기반 코딩 에이전트가 고수준 요구사항만을 입력받아 인간 개입 없이 완전한 소프트웨어를 자율 개발하는 프레임워크다.  
HoH는 기존 코딩 에이전트 하네스(harness) 위에서 동작하며, 반복적인 계획(Planning)–코딩(Coding)–테스트(Testing) 루프를 조직화한다.  
각 루프에서는 Project Planner, Developer, QA Tester 세 역할이 분리 실행되며, 소프트웨어 아티팩트($A_t$)와 실행 증거($\mathcal{E}_t$)가 루프 간에 지속 전달된다.  
HoH는 세 벤치마크(GameCraft-Bench, FrontierSWE, ProgramBench)에서 세 가지 하네스-모델 조합 모두에서 기존 Vanilla 베이스라인을 일관되게 초과달성한다.  
3회 반복 후 평균 상대적 성능 향상률은 **52.25%**, 최대 향상률은 **82.86%**이다.  
단순 반복 실행(Vanilla Continuation)과의 비교에서 HoH는 동일한 개발 패스 예산 내에서 항상 우위를 보인다.  
다중 일(multi-day) 자율 FPS 게임 개발 사례에서는 70번 이상의 반복을 통해 인간이 플레이 가능한 게임(*Fusepoint*)을 완성했다.  
Ablation 연구는 계획 갱신, 증거 피드백, 워밍 스타트 세 요소 모두가 성능에 기여함을 보인다.  
HoH는 모델이나 하네스 구현을 수정하지 않고도 장기 자율 개발을 가능하게 하는 메타-오케스트레이션 프레임워크로서의 실용성을 입증한다.

### 1-1. 연구의 목적과 필요성

**목적**: 인간 개입 없이 고수준 요구사항만으로 완전하고 배포 가능한 소프트웨어를 자율 개발하는 시스템 구축.

**필요성**:

| 문제 | 설명 |
|------|------|
| 기존 코딩 에이전트의 한계 | 대부분 human-in-the-loop 방식으로, 개발자가 지속적으로 개입해야 함 (p.2) |
| 장기 궤적의 문제 | 개발이 길어질수록 에이전트가 초기 요구사항·설계 결정을 망각하거나 지역적 수정이 전체 시스템 제약을 위반함 (p.2) |
| 반복 수리의 함정 | 반복적인 검사-수리 사이클에 갇혀 전체 시스템 진전이 없거나, 불완전한 아티팩트를 완성으로 오인함 (p.2) |
| 기존 하네스의 제약 | 단일 에피소드 내 개발로 한정되어, 검증된 기능·설계 결정·실행 증거를 후속 작업에 보존하지 못함 (p.4) |

> 💡 **용어 설명 - 하네스(Harness)**: LLM이 소프트웨어 개발 환경과 상호작용하는 운영 계층. 어떤 정보를 받고, 어떤 도구를 사용할 수 있으며, 실행 결과가 어떻게 다음 결정에 반영되는지를 결정하는 시스템 (p.2, 4).

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|------|
| 1 | HoH는 기존 하네스 대비 소프트웨어 품질을 일관되게 향상 | 3개 벤치마크, 3개 구성에서 모두 Vanilla 초과. 평균 52.25% 상대 향상 | Table 1, p.11 |
| 2 | HoH의 성능 향상은 단순 반복 실행 증가로 설명되지 않음 | Vanilla Continuation 3회(58.24)보다 HoH@2(64.84)가 더 적은 토큰으로 더 높은 점수 달성 | Table 2, p.13 |
| 3 | 루프가 진행될수록 품질이 단조적으로 향상됨 | GameCraft-Bench Overall이 HoH@1→@3 단조 증가. FrontierSWE에서 10회 반복 시 72.67% Dominance | Figure 5, Table 1 |
| 4 | 계획 갱신, 증거 피드백, 아티팩트 워밍 스타트 모두 필수적 | 각 제거 시 점수 6.28~8.13점 하락. 워밍 스타트 제거 시 토큰 사용량도 11.12M으로 증가 | Table 3, p.14 |
| 5 | 70회 이상 루프에서도 일관된 점진적 발전 유지 | *Fusepoint* FPS 게임 개발: 81개 이슈 중 65개 해결, 인간 플레이 가능 게임 완성 | Section 5, Figure 1 |
| 6 | HoH는 초기 Vanilla 성능 수준에 무관하게 개선 제공 | 가장 낮은 Vanilla 성능의 OpenCode도 HoH@3에서 크게 향상 | p.11, Table 1 |

---

## 2-1. 상세 분석: 문제, 방법, 모델 구조, 성능, 한계

### 해결하고자 하는 문제

**(1) 상태 단절 문제 (State Disconnection)**
장기 개발 궤적에서 이전 요구사항, 설계 결정, 관찰된 실패, 검증된 동작이 후속 변경과 단절됨.

**(2) 범위 결정 문제 (Scope Underdetermination)**
고수준 명세만으로는 다음에 해야 할 작업을 결정할 수 없음. 컴포넌트 의존성과 구현 제약이 지역적으로 합리적인 변경을 기존 동작과 충돌하게 만듦.

**(3) 품질 검증 문제 (Quality Verification)**
기능적·품질 요구사항은 다양한 시나리오별 동작으로 나타나며, 누락되거나 잘못된 동작이 감지되지 않아 불완전한 아티팩트가 완성으로 받아들여질 수 있음.

---

### 제안하는 방법 (수식 포함)

#### 핵심 수식

**수식 (1): HoH 기본 매핑**

$$\text{HoH}_{M,H} : \mathcal{S} \longmapsto A$$

- $\mathcal{S}$: 소프트웨어 명세(specification)
- $M$: 언어 모델(language model)
- $H$: 코딩 하네스(coding harness)
- $A$: 최종 소프트웨어 아티팩트(artifact)

> 💡 **용어 설명 - 아티팩트(Artifact)**: 개발 과정에서 생성되는 소프트웨어 산출물. 소스코드, 설정 파일, 리소스, 프로젝트 메타데이터를 포함 (p.6).

---

**수식 (2): 루프 간 상태 전이**

$$(A_{t-1}, \mathcal{E}_{t-1}) \xrightarrow{\text{loop } t \text{ under } \mathcal{S}} (A_t, \mathcal{E}_t)$$

- $A_t$: 루프 $t$ 이후 소프트웨어 아티팩트 상태 (소스코드, 설정, 리소스, 메타데이터)
- $\mathcal{E}_t$: 루프 $t$ 이후 실행 증거 상태 (검증된 동작, 미해결 실패, 우선순위 갱신)
- $A_0$: 첫 루프 전 빈 프로젝트 워크스페이스
- $\mathcal{E}_0 = \emptyset$: 초기 증거 없음

---

**수식 (3): 한 번의 HoH 반복 구현**

$$D_t = \text{Plan}_H(\mathcal{S}, \mathcal{E}_{t-1})$$

$$A_t = \text{Dev}_H(A_{t-1}; \mathcal{S}, D_t)$$

$$\mathcal{E}_t = \text{Test}_H(A_t; \mathcal{S}, D_t)$$

- $D_t$: 반복 $t$의 개발 문서 (Development Document)
- $\text{Plan}_H$: Project Planner 역할 호출
- $\text{Dev}_H$: Developer 역할 호출
- $\text{Test}_H$: QA Tester 역할 호출

> 💡 **용어 설명 - 개발 문서($D_t$)**: 각 반복에서 Project Planner가 생성하는 구조화된 문서. 달성할 작업 범위, 보존할 기능, 검증 조건을 포함 (p.7-8).

---

**수식 (4): 증거 수집 과정**

$$C_t = \text{Claims}(\mathcal{S}, D_t)$$

$$r_i = \text{Observe}(A_t, c_i)$$

$$s_i = \text{Assess}(c_i, r_i)$$

$$\mathcal{E}_t = \{(c_i, r_i, s_i)\}_{c_i \in C_t}$$

- $C_t$: 검사 가능한 주장들의 집합
- $c_i$: 개별 주장(claim)
- $r_i$: 주장 $c_i$에 대해 수집된 실행 기록
- $s_i$: 정규화된 QA 상태(verified 또는 gap)

---

**수식 (5)-(6): 증거 분할**

$$\mathcal{E}_t^{\text{ver}} = \{(c_i, r_i, s_i) \in \mathcal{E}_t \mid s_i = \text{verified}\}$$

$$\mathcal{E}_t^{\text{gap}} = \{(c_i, r_i, s_i) \in \mathcal{E}_t \mid s_i = \text{gap}\}$$

$$\mathcal{E}_t = \mathcal{E}_t^{\text{ver}} \cup \mathcal{E}_t^{\text{gap}}, \quad \mathcal{E}_t^{\text{ver}} \cap \mathcal{E}_t^{\text{gap}} = \emptyset$$

> 💡 **용어 설명 - 검증됨/갭(Verified/Gap)**: $\mathcal{E}_t^{\text{ver}}$는 증거로 뒷받침된 요구사항, $\mathcal{E}_t^{\text{gap}}$는 미충족·실패·회귀·불충분한 증거의 요구사항. 갭은 다음 계획의 수정 대상이 됨 (p.26-27).

---

**수식 (7): Vanilla Continuation**

$$A_k^{\text{VC}} = \text{Dev}_H\left(A_{k-1}^{\text{VC}}; \mathcal{S}, p_{\text{cont}}\right), \quad k \in \{2, 3\}$$

- $p_{\text{cont}}$: 고정된 계속 지시 프롬프트 ("Continue developing and testing the current game.")
- 별도 계획 문서나 QA Tester 호출 없음

---

**수식 (8): Ablation 변형**

$$\text{w/o Plan Update}: D_t = D_1, \quad t > 1$$

$$\text{w/o Evidence Feedback}: D_t = \text{Plan}_H(\mathcal{S}, \emptyset)$$

$$\text{w/o Warm-Start}: A_t = \text{Dev}_H(A_0; \mathcal{S}, D_t)$$

---

**수식 (9): GameCraft-Bench Overall 점수**

$$\text{Overall} = 100B\,(0.15M + 0.35D + 0.15V + 0.35A)$$

- $B$: 컴파일 및 실행 성공 여부 (성공=1, 실패=0)
- $M$: Core Mechanics 평균 점수
- $D$: Content Depth 평균 점수
- $V$: Functional Visuals 평균 점수
- $A$: Art and Presentation 평균 점수

---

**수식 (10): 태스크 수준 집계**

$$\bar{s}_{\mathcal{B}}(c) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} s_i(c)$$

---

**수식 (11): 누적 토큰 사용량**

$$C_i(c) = 10^{-6} \sum_{j \in \mathcal{I}_i(c)} \left(n_j^{\text{in}} + n_j^{\text{out}}\right), \quad \bar{C}(c) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} C_i(c)$$

- $n_j^{\text{in}}$: $j$번째 호출의 입력 토큰 수
- $n_j^{\text{out}}$: $j$번째 호출의 출력 토큰 수

---

**수식 (12): 토큰 대비 품질 효율**

$$\eta(c) = \frac{\bar{s}_{\text{GC}}(c) - \bar{s}_{\text{GC}}(\text{Vanilla})}{\bar{C}(c) - \bar{C}(\text{Vanilla})}$$

---

**수식 (13)-(16): FrontierSWE Dominance**

$$s(x, y) = \begin{cases} 1, & x > y \\ 0.5, & x = y \\ 0, & x < y \end{cases}$$

$$\text{Dominance}_{d,t}(a) = \frac{1}{11} \sum_{j \neq a} s(r_{a,t}, r_{j,t})$$

$$\text{Dominance}_d(a) = \frac{1}{N_d} \sum_{t=1}^{N_d} \left[\frac{1}{11} \sum_{j \neq a} s(r_{a,t}, r_{j,t})\right]$$

$$\text{Dominance}(a) = \frac{1}{3} \sum_d \text{Dominance}_d(a)$$

- $d$: 도메인 (Implementation, Performance, Research)
- $N_d$: 도메인 $d$의 태스크 수 (각각 4, 9, 2)
- 비교 풀: 3개 시스템 × 4개 조건(Vanilla, HoH@1–3) = 12개 구성

> 💡 **용어 설명 - Dominance Score**: 특정 구성이 비교 풀 내 다른 11개 구성을 동일 태스크에서 이기는 평균 비율. 50%가 기준선 (랜덤과 동일) (p.37).

---

**수식 (17): PXI 평가**

$$s_{p,k} = \frac{1}{3} \sum_{j=1}^{3} x_{p,k,j}$$

- $p$: 평가자
- $k$: PXI 구성 요소 (10개)
- $x_{p,k,j}$: $j$번째 문항 응답 (-3~+3)

> 💡 **용어 설명 - PXI (Player Experience Inventory)**: 게임 플레이어 경험을 측정하는 검증된 설문 도구. 기능적·심리사회적 결과를 포함한 10개 구성 요소로 이루어짐 (p.38, 참고문헌 34).

---

### 모델 구조

```
소프트웨어 명세 S
        │
        ▼
┌─────────────────────────────────────┐
│         Harness-of-Harness          │
│  ┌─────────────────────────────┐    │
│  │     반복 루프 t              │    │
│  │                             │    │
│  │  S + E_{t-1} + A_{t-1}      │    │
│  │         │                   │    │
│  │         ▼                   │    │
│  │  [Project Planner]          │    │
│  │  → 개발 문서 D_t 생성       │    │
│  │         │                   │    │
│  │         ▼                   │    │
│  │  [Developer]                │    │
│  │  A_{t-1} + D_t → A_t 작성  │    │
│  │         │                   │    │
│  │         ▼                   │    │
│  │  [QA Tester] (read-only)    │    │
│  │  A_t 평가 → E_t 생성        │    │
│  └─────────────────────────────┘    │
│           │                         │
│     (A_t, E_t) → 다음 루프          │
└─────────────────────────────────────┘
        │
        ▼
    최종 아티팩트 A_T
```

**Runtime 시스템**:
- 각 역할의 입력/도구/쓰기 권한 제어
- 구조화된 출력 스키마 강제 (위반 시 재시도)
- 아티팩트 생성과 평가 분리 (QA는 frozen read-only 아티팩트만 접근)

**점진적 공개(Progressive Disclosure)**:
- 전용 메모리 모듈 대신 파일 시스템에 계획·보고서·이력 저장
- 관련 시점에만 세부 내용 검색 → 컨텍스트 윈도우 절약

> 💡 **용어 설명 - 점진적 공개(Progressive Disclosure)**: 모든 정보를 한꺼번에 제공하지 않고, 필요한 시점에 관련 정보만 노출하는 방식. LLM의 컨텍스트 윈도우 한계를 극복하기 위한 설계 전략 (p.3).

---

### 성능 향상

| 벤치마크 | 설정 | Vanilla | HoH@3 | 절대 향상 |
|----------|------|---------|-------|-----------|
| GameCraft-Bench | Codex+GPT-5.5 | 49.58 | 71.52 | +21.93 |
| GameCraft-Bench | OpenCode+DS-V4-Pro | 26.90 | 48.98 | +22.08 |
| GameCraft-Bench | Pi+MiniMax-M3 | 42.16 | 58.78 | +16.62 |
| FrontierSWE Dominance | Codex+GPT-5.5 | 44% | 71% | +27%p |
| FrontierSWE Dominance | OpenCode+DS-V4-Pro | 25% | 44% | +19%p |
| FrontierSWE Dominance | Pi+MiniMax-M3 | 35% | 64% | +29%p |
| ProgramBench PassRate | Codex+GPT-5.5 | 60.41 | 66.50 | +6.09 |
| ProgramBench PassRate | OpenCode+DS-V4-Pro | 45.27 | 57.56 | +12.29 |
| ProgramBench PassRate | Pi+MiniMax-M3 | 35.83 | 52.68 | +16.85 |

**FrontierSWE 10회 반복 (Codex+GPT-5.5)**:
- Vanilla: 27.33% → HoH@10: 72.67% (최고: HoH@9에서 76.00%)

**예산 대비 효율 (GameCraft-Bench, 수식 12)**:
- Vanilla Continuation: 2.32점/백만 토큰
- HoH: 3.77점/백만 토큰 (62% 효율 우위)

---

### 한계

| 한계 | 설명 |
|------|------|
| ⚠️ 반복 수 제한 | 주요 실험은 T=3으로 제한. 더 많은 반복에서의 장기 행동 불명확 |
| ⚠️ 비단조적 향상 | Pi+MiniMax-M3의 ProgramBench에서 HoH@2→@3 소폭 하락 (53.57→52.68) |
| ⚠️ 단일 실행 | 태스크-조건별 1회 실행. 노이즈 추정 어려움 |
| ⚠️ 도메인 범위 | GameCraft-Bench와 Godot 게임 개발에 집중. 다른 소프트웨어 유형 일반화 미검증 |
| ⚠️ 토큰 비용 증가 | HoH는 Vanilla 대비 토큰을 3~4배 사용하며, 이는 비용 증가를 의미 |
| ⚠️ 인간 플레이 테스트 규모 | Fusepoint 게임은 소수 평가자만 참여 |

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| HoH@3가 모든 벤치마크·설정에서 Vanilla 초과 | Table 1 (p.11) |
| HoH는 Vanilla Continuation보다 같은 패스 수에서 우수 | Table 2 (p.13), Figure 9 (p.44) |
| FrontierSWE 10회 반복에서 Dominance 72.67% 달성 | Figure 5 (p.12) |
| Ablation: 계획 갱신 제거 시 -8.13점 | Table 3 (p.14) |
| Ablation: 증거 피드백 제거 시 -6.28점 | Table 3 (p.14) |
| Ablation: 워밍 스타트 제거 시 -7.85점, 토큰 11.12M | Table 3 (p.14) |
| 70회 루프 FPS 게임 개발 성공 | Figure 1 (p.1), Section 5 (p.14-16) |
| 게임 4개 품질 차원 모두 개선 | Figure 4 (p.12) |
| 질적 비교: 3개 게임 프레임 | Figure 6 (p.13) |
| HoH 개요 아키텍처 | Figure 3 (p.5), Algorithm 1 (p.9) |
| 하네스-모델 설정 | Table 6 (p.30) |
| GameCraft-Bench 측정 공식 | Eq.(9) (p.35) |
| 증거 수집 공식 | Eq.(4) (p.26) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

**연구 주제**: "자율 소프트웨어 개발을 위한 지속적 개선 가능 반복 프레임워크 HoH"

**저자 보고 수치**:
- HoH@3 평균 상대 향상률: **52.25%**, 최대: **82.86%** (Abstract, p.1)
- GameCraft-Bench 절대 향상: **16.62–22.08점** (p.3)
- FrontierSWE Dominance 향상: **19–29%p** (p.3)
- ProgramBench PassRate 향상: **6.09–16.85점** (p.3)
- HoH@10 FrontierSWE Dominance: **72.67%** vs Vanilla 27.33% (p.12)
- Ablation 점수 감소: Plan Update 없이 **-8.13**, Evidence Feedback 없이 **-6.28**, Warm-Start 없이 **-7.85** (Table 3)

### 분석자 해석

⚠️ **통계적 불확실성**: 대부분의 수치가 태스크-조건별 **단 1회 실행** 기반. 신뢰 구간은 GameCraft-Bench 컴포넌트 분석에만 제공됨 (95% 부트스트랩).

⚠️ **성능 향상의 해석 주의점**:
- "평균 상대 향상률 52.25%"는 어떻게 계산되었는지 명확하지 않음 (Vanilla 기준 상대값인지, 특정 선택 방식인지)
- 82.86% 최대 향상은 특정 저성능 Vanilla 조합에서 발생한 것으로 대표성이 제한됨

✅ **의미 있는 관찰**:
- HoH@2가 3-pass Vanilla Continuation보다 적은 토큰으로 더 높은 점수 달성 → 단순 반복 이상의 메커니즘 기여 확인
- 세 하네스-모델 조합 모두에서 일관된 향상 → 특정 모델/하네스에 종속되지 않는 범용성

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적으로 취약한 부분

| 문제 | 상세 |
|------|------|
| **단일 실행** | 모든 실험이 태스크-조건별 1회만 실행. 분산·표준오차 없음 (p.30, Table 7) |
| **신뢰구간 부재** | Figure 4의 컴포넌트 분석에만 95% 부트스트랩 CI 제공. 나머지 핵심 결과는 없음 |
| **소표본 FrontierSWE** | 15개 태스크 (원래 17개에서 2개 제외). 특히 Research 카테고리는 2개뿐 → 분산 추정 불가 |
| **Dominance 계산 방식** | 비교 풀이 Vanilla+HoH@1–3으로만 구성 → 외부 베이스라인 없어 상대적 의미만 존재 |
| **Fusepoint 사례 연구** | 단일 게임, 단일 팀 개발, 평가자 수 미공개 → 일반화 불가 |
| **비단조적 하락** | Pi+ProgramBench HoH@2→@3 하락 (53.57→52.68), OpenCode 일부 태스크 음수 Δ (e.g., Autobattler -12.12) |

### ⚠️ 비교 불가능한 수치

| 수치 | 이유 |
|------|------|
| 크로스-제공자 토큰 비교 | 캐시 회계 방식이 제공자마다 다름. 논문도 이를 인정 (p.10) |
| FrontierSWE 2개 태스크 제외 | 인프라 이유로 제외 → 전체 벤치마크 점수와 비교 불가 |
| 다중일 Fusepoint vs 벤치마크 | 도구·스킬·버전관리 추가로 설정이 다름 |
| Vanilla vs HoH 시간 비용 | 토큰과 시간이 보고되지만 3-pass HoH는 계획+테스트 추가 비용 포함 |

---

## 6. 문서가 답하지 않는 질문

| # | 미답 질문 |
|---|-----------|
| 1 | HoH는 몇 회 반복에서 수렴하는가? FrontierSWE 이외 벤치마크에서 10회+ 반복 결과는? |
| 2 | 루프 내에서 각 역할(Planner/Developer/Tester)의 개별 기여도는? |
| 3 | 서로 다른 하네스-모델 조합을 혼합하면 (e.g., 계획에 GPT-5.5, 개발에 DeepSeek) 어떤 성능이 나오는가? |
| 4 | 워밍 스타트 없이 토큰이 증가하는 이유는 정확히 무엇인가? (재구성 비용인가, 탐색 증가인가?) |
| 5 | QA Tester의 주관적 판단 오류율은 얼마나 되는가? 잘못된 "검증(verified)" 판정의 영향은? |
| 6 | 소프트웨어 시스템이 복잡해질수록 (예: 수백만 LOC) HoH 루프의 효율성은 어떻게 변하는가? |
| 7 | 다른 유형의 소프트웨어 (웹 앱, ML 파이프라인, 시스템 소프트웨어)에서의 일반화 성능은? |
| 8 | 17개 이슈가 재개되는 회귀(regression)의 근본 원인은? |
| 9 | HoH의 비용(토큰, 시간, API 비용)은 실무에서 허용 가능한 수준인가? |
| 10 | 동일 태스크를 여러 번 실행 시 HoH 결과의 분산은 얼마나 되는가? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1): FPS 게임 자율 개발 궤적

**내용**: 70회 반복에 걸친 *Fusepoint* FPS 게임 개발의 이슈 추적 그래프 (신규 이슈, 종료 이슈, 오픈 이슈 수) + 각 발전 단계 스크린샷.

**해석**:
- **Phase 1 (Loop 1–27)**: 초기 구성 단계. 핵심 기능 구현과 함께 이슈 백로그 증가 (테스트 가능성이 높아지며 더 많은 문제 발견)
- **Phase 2 (Loop 28–49)**: 능력 확장. 새 기능 추가 + 기존 이슈 해결 병행
- **Phase 3 (Loop 50–70)**: 안정화. 기능 추가 감소, 이슈 해결 중심으로 백로그 감소
- **핵심 통찰**: 오픈 이슈 수가 단조적으로 감소하지 않음 → 자율 개발에서 회귀와 새로운 발견이 지속 발생함을 보여줌
- **한계**: 단일 게임 사례, 인간 개입(네트워크/API 복원)이 있었음

---

### Figure 3 (p.5): Harness-of-Harness 전체 구조도

**내용**: HoH의 세 역할(Project Planner, Developer, QA Tester)과 데이터 흐름, 도구 및 산출물을 보여주는 시스템 다이어그램.

**해석**:
- **Project Planner**: $\mathcal{S}$와 $\mathcal{E}_{t-1}$을 받아 $D_t$ 생성. 아티팩트 수정 권한 없음 (read-only)
- **Developer**: $D_t$와 $A_{t-1}$을 받아 $A_t$ 생성. 유일하게 아티팩트 수정 권한 보유
- **QA Tester**: Frozen $A_t$를 읽기 전용으로 접근해 $\mathcal{E}_t$ 생성. 수정 불가
- **핵심 설계 원칙**: 역할 분리(separation of concerns)가 구현과 수락(acceptance)이 동일 결정자에게 귀결되는 것을 방지
- **도구 계층**: 코어 도구(Godot MCP, Git, Asset Provider 등) + 개발 스킬(2D/3D 자산 선택, UI/UX 폴리시, 런타임 디버깅)

---

### Figure 4 (p.12): GameCraft-Bench 4개 품질 차원 비교

**내용**: Vanilla vs HoH@3의 Core Mechanics(M), Content Depth(D), Functional Visuals(V), Art and Presentation(A) 점수 비교 (95% 부트스트랩 CI 포함).

**해석**:
- 세 하네스-모델 조합 모두에서 4개 차원 전부 향상
- **Codex+GPT-5.5**: Functional Visuals 최대 향상 (+25.56: 48.67→74.23)
- **OpenCode+DS-V4-Pro**: 가장 낮은 Vanilla 시작점에서 Mechanics 최대 향상 (+34.63: 34.71→69.34)
- **통계적 주목점**: 오차 막대(95% CI)가 제공되어 이 그림만 통계적 신뢰성을 부분적으로 확인 가능
- **해석 주의**: Content Depth(D, 가중치 0.35)와 Art(A, 가중치 0.35)가 Mechanics(M, 0.15)보다 Overall 점수에 더 큰 영향

---

### Figure 5 (p.12): FrontierSWE 10회 반복 Dominance 추이

**내용**: Codex+GPT-5.5의 FrontierSWE Dominance를 1~10회 반복에 걸쳐 추적 (±1 SE, 최고 체크포인트 표시).

**해석**:
- HoH@3: 39.33% → HoH@9 최고: 76.00% → HoH@10: 72.67%
- Vanilla 기준선: 27.33%
- **지속 향상**: 3회를 넘어서도 성능이 계속 증가 → HoH의 장기 스케일 가능성
- **비단조성**: HoH@9→@10 소폭 하락 (76%→72.67%) → 수렴 또는 회귀 발생 가능성
- **한계**: 비교 풀이 내부 구성으로만 이루어져 (Vanilla+HoH@1–10), 외부 독립 시스템 대비 성능은 알 수 없음

---

### Figure 6 (p.13): Vanilla vs HoH@3 게임 프레임 비교

**내용**: 3개 GameCraft-Bench 태스크(Momentum Lab, Kitchen Rush, Ant Empire)에서 Vanilla와 HoH@3의 게임플레이 프레임 비교.

**해석**:

| 게임 | Vanilla 문제점 | HoH@3 개선 |
|------|---------------|------------|
| Momentum Lab | 평평한 기하학적 플랫폼, 겹치는 수정자 구역 | 테마 지형과 목표, 가이드된 벽 점프 경로 |
| Kitchen Rush | 도식적 자르기 타일, 단일 플레이팅 구역 | 삽화 준비 스테이션, 픽업-플레이팅-쓰레기 워크플로 |
| Ant Empire | 특수 카스트 모두 ×0, 단일 홍수 카운트다운 | 특수 카스트 활성화, 계절·결과 상태 표시 |

- **Overall 점수 향상**: 34.05→70.61, 42.63→73.38, 65.52→87.88
- **해석**: 시각적 개선이 단순 미적 요소가 아니라 게임플레이 정보 전달과 플레이어 경험에 직결됨을 보여줌

---

## 8. 결론과 후속 연구

### 저자들이 제시한 시사점 (p.16)

1. **메타-오케스트레이션**: HoH는 코딩 하네스 구현을 수정하지 않고도 장기 자율 개발을 가능하게 하는 메타 프레임워크를 제시
2. **하네스 참여 구조화**: 코딩 하네스가 모델 작동을 구조화하듯, HoH는 하네스의 장기 개발 참여를 구조화
3. **증거 기반 오케스트레이션**: 지속적·증거 기반 오케스트레이션을 통한 end-to-end 소프트웨어 개발의 실용적 경로 제시

### 저자들이 제시한 후속 연구 (p.16)

- 다양한 유형의 게임과 소프트웨어 시스템으로 HoH 적용 범위 확장 (참고문헌 35, 42–44, 47–48)
- 자율 소프트웨어 개발을 위한 일반 프레임워크 구축

---

### 8-1. 모델의 일반화 성능 향상 가능성

#### 현재 일반화 수준 평가

| 일반화 차원 | 현재 수준 | 평가 |
|------------|----------|------|
| 하네스-모델 독립성 | 3개 조합 모두 일관 향상 | ✅ 강함 |
| 벤치마크 다양성 | 게임개발, SW엔지니어링, 프로그램 재구성 | ✅ 중간 |
| 도메인 다양성 | 주로 Godot 게임 개발 | ⚠️ 제한적 |
| 복잡도 스케일 | 단일 태스크~70회 루프 복잡 게임 | ✅ 일부 검증 |

#### 일반화 성능 향상을 위한 연구 방향

**① 도메인 독립적 증거 수집 메커니즘**

현재 QA Tester는 게임플레이 특화 테스트(화면 캡처, 리플레이 추적)에 의존함. 웹 서비스, ML 파이프라인, 임베디드 시스템 등에서의 자율 테스트 메커니즘이 필요:

$$\mathcal{E}_t^{\text{domain}} = \text{Test}_H^{\text{domain}}(A_t; \mathcal{S}, D_t, \text{DomainAdapter})$$

**② 적응적 루프 깊이**

현재 T는 고정됨. 태스크 복잡도에 따른 동적 루프 수 결정:

```math
T^* = \arg\min_T \left\{ \bar{s}_{\mathcal{B}}(T) \geq \theta \right\}
```

**③ 교차 도메인 스킬 전이**

게임 개발에서 학습된 오류 패턴을 다른 소프트웨어 도메인에 적용하는 스킬 라이브러리 구축 필요.

**④ 메타-학습 통합**

HoH의 반복 구조는 MAML(Model-Agnostic Meta-Learning) 계열 접근과 결합 가능:
- 이전 프로젝트의 증거 패턴을 초기화 $\mathcal{E}_0$에 활용
- Few-shot 적응을 통한 새 도메인 조기 수렴

**⑤ 회귀 방지 강화**

17개 이슈 재개(regression) 사례가 보고됨. 검증된 행동의 불변성을 수식으로 명시:

$$\forall (c_i, r_i, s_i) \in \mathcal{E}_t^{\text{ver}}: \text{Observe}(A_{t+k}, c_i) \supseteq r_i, \quad \forall k > 0$$

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 이 논문이 2026년 9월로 제출되었으며, 인용된 일부 논문도 2026년 발표로 기재되어 있습니다. 아래 비교는 논문 내 인용 정보와 분석자의 일반 지식을 결합했으나, 2025년 이후 논문의 존재는 독립적으로 검증 불가합니다.

#### 관련 연구 비교 표

| 연구 | 방법 | HoH와의 차이 | 참고 위치 |
|------|------|-------------|-----------|
| **SWE-Bench** (Jimenez et al., ICLR 2024) | 저장소 수준 이슈 해결 | 기존 코드베이스 패치 vs HoH의 처음부터 생성 | [12], p.4 |
| **MetaGPT** (Hong et al., ICLR 2024) | 역할 기반 다중 에이전트 워크플로 | 고정 워크플로 vs HoH의 동적 증거 기반 계획 | [8], p.4 |
| **ChatDev** (Qian et al., ACL 2024) | 통신 에이전트 소프트웨어 개발 | 단일 에피소드 vs HoH의 다중 루프 | [32], p.4 |
| **SWE-Agent** (Yang et al., NeurIPS 2024) | 에이전트-컴퓨터 인터페이스 | 단일 pass vs HoH가 위에서 동작 | [39], p.4 |
| **Self-Refine** (Madaan et al., NeurIPS 2023) | 자기 피드백 반복 개선 | 동일 에이전트 자기 평가 vs HoH의 독립 QA | [24], p.2 |
| **Reflexion** (Shinn et al., NeurIPS 2023) | 언어적 강화학습 | 단일 태스크 반성 vs HoH의 다중 루프 상태 관리 | [33], p.2 |
| **AgileCoder** (Nguyen et al., FORGE 2025) | 애자일 방법론 기반 반복 개발 | 스프린트 기반 vs HoH의 증거 기반 루프 | [27], p.4 |
| **EvoMAC** (Hu et al., ICLR 2025) | 테스트 피드백 기반 워크플로 적응 | 워크플로 최적화 vs HoH의 하네스 위 메타 계층 | [9], p.4 |
| **GameCraft-Bench** (Luo et al., 2026) | 게임 개발 평가 벤치마크 | HoH가 사용하는 평가 도구 중 하나 | [23], p.3 |
| **Self-Harness** (Zhang et al., 2026) | 하네스 자가 진단·수정 | 하네스 자체 최적화 vs HoH의 하네스 위 오케스트레이션 | [45], p.4 |

#### HoH의 핵심 차별점

```
기존 연구 (Self-Refine, Reflexion 등)
→ 단일 에이전트, 단일 태스크, 단기 피드백 루프

MetaGPT, ChatDev
→ 다중 에이전트이나 고정 워크플로, 단일 에피소드

SWE-Agent, OpenHands
→ 강력한 하네스이나 단일 개발 패스, 장기 상태 미보존

HoH (이 논문)
→ 기존 하네스 위의 메타 계층 + 루프 간 아티팩트·증거 지속 관리
→ 무한 확장 가능한 장기 개발 지원
```

#### HoH가 앞으로의 연구에 미치는 영향

**① 메타-하네스 연구 방향 개척**
기존 연구가 더 좋은 단일 에이전트·하네스를 만드는 데 집중했다면, HoH는 기존 시스템을 수정 없이 활용하는 **메타-레벨 오케스트레이션** 패러다임을 제시. 이는 향후 연구에서 "어떻게 에이전트를 만드는가"보다 "어떻게 에이전트 시스템을 조직화하는가"에 초점을 옮길 수 있음.

**② 장기 자율 개발 벤치마크 필요성 촉발**
현재 SWE-Bench 계열은 단기·단일 태스크 중심. HoH의 FPS 게임 70루프 사례는 **다중 일 자율 개발을 평가하는 새 벤치마크** 필요성을 시사.

**③ 역할 분리의 중요성 실증**
구현자와 수락자의 분리($\text{Dev}_H \neq \text{Test}_H$)가 성능 향상의 핵심 기여자임을 Ablation으로 확인. 이는 향후 다중 에이전트 설계에서 **독립 검증 에이전트**의 필수화를 촉진.

#### 앞으로 연구 시 고려할 점

| 고려 사항 | 구체적 방향 |
|-----------|------------|
| **실험 반복성** | 단일 실행 결과의 분산 추정을 위한 다중 시드 실행 필수화 |
| **비용 효율성** | HoH의 토큰 비용(Vanilla 대비 3~4×)의 실무적 최적화 필요 |
| **회귀 방지** | 검증된 기능의 보존을 강제하는 형식적 메커니즘 연구 |
| **도메인 확장** | 게임 외 소프트웨어(웹, ML, 임베디드)에서의 도메인 어댑터 설계 |
| **자동 중단 기준** | 최적 반복 수 $T^*$를 동적으로 결정하는 기준 개발 |
| **인간-AI 협업 경계** | 완전 자율과 human-in-the-loop 사이의 최적 개입 지점 연구 |
| **평가 지표 다양화** | 게임 품질 4차원 외 실용성·유지보수성·보안성 등 추가 |
| **멀티모달 QA** | 시각적 증거(스크린샷, 비디오)를 활용한 더 정교한 자동 QA 시스템 |

---

## 참고 자료

본 분석에서 참고한 문헌 (논문 내 인용 목록 기준):

1. **Anthropic. Claude Code for Product Development.** Anthropic technical report, 2025.
2. **Hong et al. "MetaGPT: Meta programming for a multi-agent collaborative framework."** ICLR 2024.
3. **Yang et al. "SWE-Agent: Agent-computer interfaces enable automated software engineering."** NeurIPS 2024.
4. **Jimenez et al. "SWE-Bench: Can language models resolve real-world GitHub issues?"** ICLR 2024.
5. **Madaan et al. "Self-Refine: Iterative refinement with self-feedback."** NeurIPS 2023.
6. **Shinn et al. "Reflexion: Language agents with verbal reinforcement learning."** NeurIPS 2023.
7. **Qian et al. "ChatDev: Communicative agents for software development."** ACL 2024.
8. **Luo et al. "GameCraft-Bench: Can Agents Build Playable Games End-to-End in a Real Game Engine?"** 2026.
9. **Yang et al. "ProgramBench: Can Language Models Rebuild Programs From Scratch?"** 2026.
10. **Chu et al. "FrontierSWE."** Proximal Blog, 2026.
11. **Larman & Basili. "Iterative and incremental developments: a brief history."** Computer, 2003.
12. **Nguyen et al. "AgileCoder."** FORGE 2025.
13. **Hu et al. "Self-evolving multi-agent collaboration networks."** ICLR 2025.
14. **Wang et al. "OpenHands."** ICLR 2025.
15. **Zhao et al. "Commit0."** ICLR 2025.
16. **Vanden Abeele et al. "Development and validation of the player experience inventory."** IJHCS 2020.
17. **Zhang et al. "Self-Harness."** 2026.
18. **Yao et al. "ReAct."** ICLR 2023.

> **본 분석의 한계**: 이 논문은 2026년 9월 날짜로 제출된 arXiv 프리프린트로, 현재 시점에서 독립적인 검증이 불가능합니다. 논문 내 모델(GPT-5.5, DeepSeek-V4-Pro, MiniMax-M3)의 존재 여부, 인용된 2026년 논문들의 실제 발표 여부, 그리고 수치들의 재현 가능성을 확인할 수 없습니다. 분석은 제공된 PDF 내용에만 근거합니다.
