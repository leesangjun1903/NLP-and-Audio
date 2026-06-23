# LEAP: Supercharging LLMs for Formal Mathematics with Agentic Frameworks 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

LEAP(LLM-in-Lean Environment Agentic Prover)는 **특화된 파인튜닝 없이** 범용 LLM만으로도, 적절한 에이전틱 프레임워크를 결합하면 형식 수학(formal mathematics) 분야에서 최고 수준의 성능을 달성할 수 있다는 것을 실증적으로 보여준다.

> **핵심 메시지**: 범용 LLM의 형식 증명 한계는 언어 이해 능력의 부재가 아니라, **구조화된 반복적 상호작용 메커니즘의 부재**에서 비롯된다.

### 세 가지 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **워크플로 기반 에이전틱 설계** | 인간 수학자의 증명 방식을 모사한 AND-OR DAG 기반 Blueprint 구조 |
| **Lean-IMO-Bench 데이터셋** | 60개 IMO급 문제를 Lean으로 형식화한 새로운 벤치마크 |
| **강력한 실험 결과** | Putnam 2025 12/12 해결, Lean-IMO-Bench에서 one-shot 10% → 70% 향상 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**이중적 격차(dual gap) 문제:**

$$\underbrace{\text{범용 LLM}}_{\text{강한 비형식 추론}} \xrightarrow{\text{격차}} \underbrace{\text{형식 증명(Lean)}}_{\text{기계 검증 가능}}$$

- LLM은 자연어 수학에서 강하지만, Lean 같은 형식 언어로 **긴 증명을 한 번에** 생성하는 데 실패함
- 특화 모델(AlphaProof, DeepSeek Prover V2 등)은 형식 증명에 강하지만, 비형식 추론·자기수정 능력이 부족함
- 기존 벤치마크(MiniF2F, PutnamBench)의 포화(saturation) 문제

### 2.2 제안 방법 및 핵심 알고리즘

#### AND-OR DAG 구조

LEAP의 증명 상태는 AND-OR Directed Acyclic Graph(DAG)로 표현된다.

- **OR 노드**: 열린 목표(goal) 또는 보조정리 — 어떤 유효한 증명 전략으로도 해결 가능
- **AND 노드**: 후보 분해(decomposition) — 모든 자식 서브목표가 증명되어야 성립

$$\text{목표 } G \xrightarrow{\text{분해}} \{L_1, L_2, \ldots, L_n\} \quad \text{(AND 노드)}$$

$$\text{모든 } L_i \text{ 증명} \Rightarrow G \text{ 증명}$$

#### 검색 복잡도 비교

Hilbert(기존 재귀적 프레임워크)의 시간 복잡도:

$$O\left((n \cdot b)^d\right)$$

여기서 $n$은 보조정리 재시도 횟수, $b$는 평균 분기 계수(branching factor), $d = 10$은 최대 증명 깊이.

LEAP는 **DAG 기반 메모이제이션**을 통해 이 지수적 복잡도를 완화한다. 공유된 중간 보조정리를 재사용함으로써:

$$\text{중복 도출 감소} \Rightarrow \text{효과적인 탐색 예산 절감}$$

#### LEAP 워크플로 (Figure 1 기반)

```
입력 정리
    │
    ▼
[Direct Formalization 시도]
    ├─ NL Prover → Formal Prover → Lean Verifier
    │   ├─ 성공: Accept
    │   └─ 실패: Reviser (최대 20회 컴파일러 피드백 수정)
    │           └─ 실패 시 → [Blueprint Generation]
    │
    └─ [Blueprint Generation]
         ├─ Blueprint Gen. (비형식 청사진 생성)
         ├─ Sketch Gen. (형식 증명 스케치 생성, sorry 허용)
         ├─ LLM Reviewer (분해 품질 평가)
         └─ Lean Verifier
              ├─ 성공: 새 서브목표를 DAG에 추가 (사이클 검사)
              └─ 실패: Reviser / 분기 포기
```

#### 세 가지 핵심 설계 원칙

**① DAG 기반 계층적 메모이제이션(Hierarchical Memoization via DAG)**

중간 보조정리를 공유 노드로 저장하여 서로 다른 분기에서 재사용. *예측적 보조정리 계획(anticipatory lemma planning)*: 아직 필요하지 않은 보조정리도 미리 제안하여 그래프에 저장.

**② 교차 비형식-형식 계획(Interleaved Informal–Formal Planning)**

$$\text{비형식 증명(NL)} \xrightarrow{\text{번역}} \text{형식 Lean 코드} \xrightarrow{\text{검증}} \text{컴파일러}$$

직접 증명 경로와 Blueprint 분해 경로 모두 **비형식 초안을 먼저 작성**한 후 형식화 진행. 이는 직접적인 코드 생성보다 덜 취약하다.

**③ 검증 기반 증명 탐색(Verification-Guided Proof Search)**

- **Lean 컴파일러**: 구문 및 타입 정확성 검사
- **LLM 리뷰어**: 분해의 *의미론적* 유용성 평가 (서브목표가 원래 목표보다 실질적으로 단순한지 판단)

LLM 리뷰어 없을 경우의 실패 패턴 (Figure 3):

```lean
-- 조부모 목표
theorem perms_a_equiv_b_card ... : Nat.card (PermsA m s k) = Nat.card (PermsB m s k)

-- 중간 보조정리 (정의 전개)
lemma valid_perms_end_eq_sigma_card ...

-- 제안된 서브목표 (조부모 목표와 동일!)
lemma card_perms_a_eq_card_perms_b ... : Nat.card (PermsA m s k) = Nat.card (PermsB m s k) := by sorry
```

이처럼 **형식적으로는 유효하지만 수학적으로 무의미한 순환 분해**를 LLM 리뷰어가 차단한다.

### 2.3 모델 구조

LEAP는 별도의 신경망 아키텍처를 학습하지 않으며, 다음 컴포넌트로 구성된다:

| 컴포넌트 | 역할 | 사용 모델 |
|---|---|---|
| NL Prover | 비형식 증명 생성 | Gemini-3.1-Pro |
| Formal Prover | Lean 코드 생성 | Gemini-3.1-Pro |
| Blueprint Generator | 분해 계획 생성 | Gemini-3.1-Pro |
| Sketch Generator | 형식 스케치 생성 | Gemini-3.1-Pro |
| LLM Reviewer | 분해 품질 평가 | Gemini-3.1-Pro |
| Lean Verifier | 형식 검증 | Lean 컴파일러 |
| LeanSearch | 관련 보조정리 검색 | Gao et al., 2024 |
| State Reader/Writer | DAG 상태 관리 | 규칙 기반 |

### 2.4 성능 향상

#### Putnam 2025 결과

| 방법 | 해결율 (%) | 비고 |
|---|---|---|
| Gemini-3.1-Pro (Pass@128) | 0.0 | 직접 형식화 |
| Goedel-Prover-V2-32B (Pass@128) | 0.0 | 특화 모델 |
| Hilbert (rollout=2) | 33.3 | 오픈소스 에이전틱 |
| Aristotle (rollout=2) | 75.0 | 폐쇄형 특화 시스템 |
| **LEAP (rollout=2)** | **100.0** | **12/12 완전 해결** |

#### Lean-IMO-Bench 결과 (Overall)

| 방법 | Basic Set (%) | Advanced Set (%) |
|---|---|---|
| Gemini-3.1-Pro (Pass@128) | 20.0 | 3.3 |
| Goedel-V2-32B (Pass@128) | 10.0 | 0.0 |
| Hilbert (rollout=2) | 36.6 | 6.6 |
| Aristotle (rollout=2) | 76.7 | 20.0 |
| **LEAP (rollout=2)** | **83.3** | **56.7** |

#### DAG 메모이제이션 Ablation (Lean-IMO-Bench)

| 설정 | Basic Overall (%) | Advanced Overall (%) |
|---|---|---|
| w/o DAG (Naive Tree) | 73.3 | 40.0 |
| Full DAG | **83.3** | **56.7** |

#### One-shot vs. 반복적 형식화 비교

| 모델 | One-shot Pass@128 (%) | Iterative Pass@1 (%) |
|---|---|---|
| Goedel-Prover-V2-32B | 10.0 | 6.6 |
| Gemini-3.1-Pro | 20.0 | **36.6** |

Gemini-3.1-Pro는 반복적 피드백으로 **83% 향상** (20.0% → 36.6%)하지만, Goedel-Prover는 오히려 감소. 이는 컴파일러 오류 해석 및 맥락 유지 능력이 반복적 형식화의 핵심임을 시사한다.

### 2.5 한계

1. **기하(Geometry) 분야 취약**: Lean-IMO-Bench에서 모든 방법이 기하 문제에서 거의 0% 달성. 올림피아드급 기하 형식화에는 도메인 특화 도구가 필요하다고 논문에서 명시.

2. **높은 계산 비용**: Putnam A5 문제의 경우 LLM 호출 수가 3,000회, 활성 노드 170개. 복잡한 문제에서 계산 비용이 급증.

   | 문제 | LLM 호출 | 활성 노드 | 증명 라인 수 |
   |---|---|---|---|
   | a1 | 71 | 8 | 405 |
   | a5 | 3,000 | 170 | 2,000 |
   | b5 | 239 | 211 | 1,900 |

3. **탐색 전략의 단순성**: 현재 DFS + 백트래킹만 사용. 더 정교한 휴리스틱 탐색(MCTS 등) 미적용.

4. **폐쇄형 모델 의존성**: Gemini-3.1-Pro에 의존하여 재현 가능성에 제약.

5. **오픈 문제 케이스 스터디의 부분적 성격**: Knuth의 Hamiltonian 분해 문제의 **핵심 서브문제**만 형식화 완료 (5,000줄 Lean 4 코드), 전체 문제가 아님.

---

## 3. 일반화 성능 향상 가능성

### 3.1 도메인 일반화 증거

LEAP는 다양한 수학 분야에서 강한 일반화 성능을 보인다:

$$\text{Algebra (Basic/Advanced): } 100\% / 100\%$$
$$\text{Number Theory (Basic/Advanced): } 100\% / 100\%$$
$$\text{Combinatorics (Basic/Advanced): } 100\% / 25\%$$
$$\text{Geometry (Basic/Advanced): } 16.7\% / 12.5\%$$

대수학과 정수론에서 **난이도 구분 없이 100%** 달성은 LEAP의 구조적 접근법이 특정 분야에 과적합되지 않았음을 시사한다.

### 3.2 일반화를 가능케 하는 구조적 요인

**① 비형식 추론의 도메인 독립성**: LLM의 비형식 추론 능력은 분야 구분 없이 적용 가능하며, Lean으로의 번역은 컴파일러 피드백을 통해 수정됨.

**② DAG 메모이제이션의 도메인 독립성**: 보조정리 공유 메커니즘은 수학 분야에 무관하게 작동. 특히 고급 대수학과 정수론에서 개선 효과가 두드러짐:

$$\text{Advanced Algebra: } 75\% \xrightarrow{\text{Full DAG}} 100\%$$
$$\text{Advanced Number Theory: } 66.6\% \xrightarrow{\text{Full DAG}} 100\%$$

**③ 오픈 문제 일반화**: LEAP는 훈련 데이터에 없는 새로운 오픈 문제(Knuth's Hamiltonian 분해, Erdős Problem 457)에도 성공적으로 적용됨 — 이는 **분포 외(out-of-distribution) 일반화**의 증거.

**④ 반복적 자기수정의 일반성**: 컴파일러 피드백 기반 수정은 특정 문제 유형에 의존하지 않으며, Gemini-3.1-Pro가 이 과정에서 특히 강점을 보임.

### 3.3 일반화의 한계와 제약

그러나 일반화에는 중요한 한계가 존재한다:

- **기하학적 추론**: 형식 기하의 특수성으로 인해 범용 접근이 효과적이지 않음. 이는 LEAP의 일반화가 *모든* 수학 분야에 균등하지 않음을 시사.

- **극도로 복잡한 증명**: Putnam A5처럼 매우 어려운 문제는 여전히 다수의 rollout과 대규모 계산을 요구함. 즉, **계산 자원 의존성**이 일반화 범위를 제한.

- **Lean Mathlib 라이브러리 의존성**: LeanSearch를 통한 Mathlib 검색에 의존하므로, Mathlib에 없는 수학 이론을 다루는 문제에서는 일반화가 제한될 수 있음. TaoBench(Taylor et al., 2026)는 "Mathlib 범위를 넘어선 일반화"를 측정하는 벤치마크로 이 문제를 제기함.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

**① 패러다임 전환**: "형식 수학에는 반드시 특화 파인튜닝이 필요하다"는 기존 통념을 도전. 범용 LLM + 에이전틱 프레임워크 조합이 특화 모델에 필적하거나 능가할 수 있음을 보임.

**② 에이전틱 AI 연구 방향 제시**: AND-OR DAG + 메모이제이션 + LLM 리뷰어의 조합은 다른 형식 검증 도메인(소프트웨어 검증, 하드웨어 설계 등)에도 확장 가능한 청사진 제공.

**③ 하이브리드 아키텍처 가능성 제시**: 논문 자체에서 "고수준 구조적 추론(범용 LLM) + 저수준 형식 증명 생성(특화 모델)"의 하이브리드가 유망함을 언급. 이는 향후 연구의 자연스러운 방향.

**④ 새로운 평가 기준(Lean-IMO-Bench) 제공**: 포화된 MiniF2F, PutnamBench를 보완하는 더 도전적인 벤치마크를 제공하여, 향후 연구의 평가 기준을 높임.

**⑤ 연구 자동화 실증**: Knuth 문제의 서브문제 형식화(5,000줄 Lean 4)를 자율적으로 수행 — AI가 실제 수학 연구를 보조할 수 있음을 실증.

### 4.2 향후 연구 시 고려할 점

**① 탐색 전략 고도화**

현재 LEAP는 단순 DFS + 백트래킹 사용. 향후 연구는 더 정교한 탐색 전략을 탐구해야 한다:

$$\text{DFS} \rightarrow \text{MCTS} \text{ 또는 } A^* \text{ 기반 휴리스틱 탐색}$$

LLM 리뷰어를 **가치 함수(value function)**로 활용하여 유망한 분기에 계산 자원을 집중하는 방향이 제안됨.

**② 계산 비용 최적화**

복잡한 문제에서 LLM 호출이 수천 회에 달하는 문제를 해결해야 한다:
- 분기 우선순위화(branch prioritization)
- 동적 계산 예산 배분(adaptive compute allocation)
- 중간 결과 캐싱 전략 강화

**③ 기하학 분야 특화 지원**

기하 분야의 near-zero 성능 개선을 위한 도메인 특화 도구(예: E-증명기, 기하 특화 tactic) 통합 연구 필요.

**④ 하이브리드 아키텍처 탐구**

논문이 명시적으로 미래 과제로 제안:

$$\underbrace{\text{범용 LLM}}_{\text{고수준 계획 및 분해}} + \underbrace{\text{특화 모델}}_{\text{저수준 Lean 증명 생성}} \rightarrow \text{최적 하이브리드}$$

**⑤ Mathlib 범위를 넘어선 일반화**

TaoBench(Taylor et al., 2026)가 제기한 문제 — Mathlib 외부 이론을 다루는 문제에서의 성능을 측정하고 개선해야 함.

**⑥ 재현 가능성 및 오픈소스화**

Gemini-3.1-Pro 의존성을 줄이고, 오픈소스 모델로도 유사 성능을 달성할 수 있는지 탐구 필요.

**⑦ 증명 길이 및 복잡도 스케일링**

현재 성공한 증명들도 수백~수천 줄 수준. 더 복잡한 현대 수학(리만 가설, BSD 추측 등)으로의 확장을 위한 스케일링 법칙 연구 필요.

---

## 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 접근 방식 | 주요 특징 | LEAP와의 차이 |
|---|---|---|---|---|
| **Polu & Sutskever** (GPT-f) | 2020 | 신경 언어 모델 기반 증명 생성 | Metamath 환경에서 LLM 최초 적용 | 에이전틱 프레임워크 없음, 단순 생성 |
| **LeanDojo** (Yang et al.) | 2023 | RAG 기반 Lean 증명 | Lean과 LLM 통합, 검색 증강 | 특화 파인튜닝 모델 의존 |
| **Draft, Sketch, Prove** (Jiang et al.) | 2023 | 비형식→형식 안내 | 비형식 증명으로 형식 탐색 안내 | 에이전틱 반복 개선 없음, 단방향 |
| **HyperTree Proof Search** (Lample et al.) | 2022 | MCTS 기반 tactic 탐색 | 트리 탐색으로 대규모 공간 탐색 | 특화 모델 의존, 고수준 계획 없음 |
| **Baldur** (First et al.) | 2023 | 전체 증명 한번에 생성 | LLM으로 완전한 증명 단번에 생성 | 반복적 수정 없음 |
| **DeepSeek-Prover-V1.5** (Xin et al.) | 2025 | RL + MCTS | 증명 보조기 피드백으로 RL, MCTS | 특화 파인튜닝 필수 |
| **DeepSeek-Prover-V2** (Ren et al.) | 2025 | RL 기반 서브목표 분해 | 강화학습으로 서브목표 분해 학습 | 대규모 파인튜닝 필요, 범용성 제한 |
| **Kimina-Prover** (Wang et al.) | 2025 | 대형 형식 추론 모델 + RL | RL로 형식 추론 특화 | 특화 훈련 필수 |
| **Hilbert** (Varambally et al.) | 2025 | 재귀적 비형식+형식 결합 | 범용 LLM(비형식) + 특화 모델(형식) | $O((n \cdot b)^d)$ 지수적 복잡도 |
| **AlphaProof** (Hubert et al.) | 2026 | RL + 자동형식화 | IMO 금메달 수준, 대규모 RL | 막대한 계산/데이터 필요 |
| **Goedel-Prover-V2** (Lin et al.) | 2026 | 스캐폴드 데이터 합성 + 자기수정 | 오픈소스 특화 ATP | 특화 파인튜닝 필수 |
| **Aristotle** (Achim et al.) | 2025 | 특화 ATP 컴포넌트 결합 | IMO 2025 금메달 수준 | 폐쇄형, 특화 ATP 의존 |
| **Lean-STar** (Lin et al.) | 2025 | 사고-증명 교차 학습 | 비형식 사고와 형식 증명을 교차 | 파인튜닝 필요 |
| **BFS-Prover** (Xin et al.) | 2025 | 최선 우선 트리 탐색 | LLM 기반 BFS로 증명 탐색 | 특화 모델 의존 |
| **LEAP** (본 논문) | 2026 | AND-OR DAG + 범용 LLM | **파인튜닝 없이** 범용 LLM만 사용 | **유일하게 특화 없이 SOTA 달성** |

### 핵심 차별점

$$\boxed{\text{LEAP} = \underbrace{\text{범용 LLM}}_{\text{파인튜닝 불필요}} + \underbrace{\text{AND-OR DAG}}_{\text{메모이제이션}} + \underbrace{\text{비형식↔형식 교차}}_{\text{Blueprint}} + \underbrace{\text{LLM 리뷰어}}_{\text{탐색 가지치기}}}$$

기존 연구들이 특화 파인튜닝 또는 대규모 RL에 의존하는 반면, LEAP는 **구조적 에이전틱 설계**만으로 유사하거나 우월한 성능을 달성한다는 점에서 독창성이 있다.

---

## 참고 자료

**본 논문**
- Kung, P.-N., Song, L., Hwang, D., Yoon, J., Li, C.-L., Severini, S., Olšák, M., Lockhart, E., Le, Q.V., Gokturk, B., Luong, T., Pfister, T., & Peng, N. (2026). *LEAP: Supercharging LLMs for Formal Mathematics with Agentic Frameworks*. arXiv:2606.03303v2.

**논문 내 인용 주요 참고문헌**
- Jiang, A.Q. et al. (2023). *Draft, Sketch, and Prove*. ICLR 2023.
- Lample, G. et al. (2022). *HyperTree Proof Search for Neural Theorem Proving*. NeurIPS 2022.
- Yang, K. et al. (2023). *LeanDojo: Theorem Proving with Retrieval-Augmented Language Models*. NeurIPS 2023.
- Ren, Z.Z. et al. (2025). *DeepSeek-Prover-V2*. arXiv:2504.21801.
- Xin, H. et al. (2025a). *DeepSeek-Prover-V1.5*. ICLR 2025.
- Xin, R. et al. (2025b). *BFS-Prover*. ACL 2025.
- Lin, H. et al. (2025). *Lean-STar*. ICLR 2025.
- Varambally, S. et al. (2025). *Hilbert*. NeurIPS 2025 Workshop.
- Achim, T. et al. (2025). *Aristotle: IMO-level Automated Theorem Proving*. arXiv:2510.01346.
- Hubert, T. et al. (2026). *AlphaProof*. Nature, 651:607–613.
- Lin, Y. et al. (2026). *Goedel-Prover-V2*. ICLR 2026.
- Wang, H. et al. (2025). *Kimina-Prover Preview*. arXiv:2504.11354.
- Gao, G. et al. (2024). *A Semantic Search Engine for Mathlib4*. EMNLP 2024 Findings.
- Taylor, A.K. et al. (2026). *TaoBench*. arXiv:2603.12744.
- Zheng, K. et al. (2022). *MiniF2F*. ICLR 2022.
- Polu, S. & Sutskever, I. (2020). *Generative Language Modeling for Automated Theorem Proving*. arXiv:2009.03393.
- Guo, D. et al. (2025). *DeepSeek-R1*. arXiv:2501.12948.
- Luong, T. et al. (2025). *Towards Robust Mathematical Reasoning*. EMNLP 2025.
- First, E. et al. (2023). *Baldur*. ESEC/FSE 2023.

**관련 프로젝트 링크**
- LEAP 코드: https://github.com/google-deepmind/superhuman/tree/main/leap
- Lean-IMO-Bench: https://imobench.github.io
