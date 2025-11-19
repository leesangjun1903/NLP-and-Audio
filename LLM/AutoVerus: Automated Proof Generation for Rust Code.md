# AutoVerus: Automated Proof Generation for Rust Code

### 1. 핵심 주장과 주요 기여

AutoVerus는 **대규모 언어 모델(LLM)을 활용하여 Rust 코드의 형식적 증명을 자동으로 생성하는 시스템**입니다. 본 논문의 핵심 주장은 다음과 같습니다:

**핵심 주장:**
- LLM 기반 증명 생성은 코드 생성만큼 발전하지 못했으나, 적절한 설계를 통해 90% 이상의 증명 성공률을 달성할 수 있습니다.
- 인간 전문가의 지식, LLM의 창의성, 형식 검증 도구의 엄밀성을 결합하면 신뢰할 수 있는 증명 생성이 가능합니다.
- Verus는 Rust 문법을 사용하기 때문에 새로운 언어 학습 없이도 LLM이 활용할 수 있는 독특한 기회를 제공합니다.

**주요 기여:**

1. **구조화된 3단계 워크플로우**: 인간 전문가의 증명 개발 프로세스를 모방한 생성(Generation) → 정제(Refinement) → 디버깅(Debugging) 단계
2. **AutoVerus 시스템**: 150개의 증명 작업으로 구성된 Verus-Bench 벤치마크 첫 구축
3. **높은 성능**: 91.3%의 증명 성공률(137/150 작업) 달성, 반 이상이 30초 이내 또는 3번의 LLM 호출로 완료

***

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하고자 하는 문제

**근본적 과제:**
- **학습 데이터 부족**: Verus는 개발 3년 차인 신규 도구로, GitHub에 10개 미만의 프로젝트만 존재하여 학습 데이터가 극히 부족합니다.
- **언어 문법의 복잡성**: Verus는 Rust 기반이지만 고유의 문법 확장이 있습니다:
  - 추상 정수형(int) 타입과 고정 비트 정수형(u64 등) 간의 타입 캐스팅 필요
  - @연산자를 통한 Rust Vec에서 Verus Seq로의 변환
  - 실행 가능한 함수와 유령 코드(ghost code)의 엄격한 분리
- **증명의 본질**: SMT 기반 검증기에서는 증명 단계별 진전을 측정하기 어려워 LLM의 성능 평가가 복잡합니다.

#### 2.2 제안하는 방법론

**3단계 프레임워크:**

**Phase 1: 초기 증명 생성**

LLM 에이전트가 다음을 수행합니다:

$$\text{GenPrelimProof}(\text{program}) \rightarrow \{p_1, p_2, p_3, p_4, p_5\}$$

여기서:
- 루프 불변식(loop invariants)을 중심으로 5개의 증명 후보 생성
- 필터링: 안전하지 않은(unsafe) 후보 제거
- 순위 지정: 검증 점수 $\{V, E\}$ 기준 정렬
  - $V$: Verus가 검증한 함수 수
  - $E$: 보고된 검증 오류 수
- 병합: 여러 후보의 루프 불변식을 결합하여 더 나은 증명 도출

**Phase 2: 일반적 증명 정제**

4개의 정제 에이전트가 순차 실행됩니다:

- **Constant-Propagation 에이전트**: 함수 전제 조건의 성질을 루프 불변식으로 추가
- **Array-Length 에이전트**: 루프에서 사용되는 배열/컨테이너의 길이 정보 추가
- **Quantifier 에이전트**: 한정자(forall, exists) 사용의 정확성 검증
- **Conditional-Loop-Invariant 에이전트**: 모든 루프 반복에 적용되지 않는 불변식 조정

각 에이전트 $A_i$에 대해:

```math
\text{RefineProof}(P_B) = \begin{cases} P_B & \text{if } \text{accept\_refine}(A_i(P_B)) \\ A_i(P_B) & \text{otherwise} \end{cases}
```

**Phase 3: 오류 기반 증명 디버깅**

Verus의 검증 오류 유형별로 설계된 10개의 복구 에이전트:

$$\text{DebugProof}(P): \text{for } i=1 \text{ to } M_{\text{Max}}$$

각 반복에서:

1. 검증 오류 수집: $\text{errors} \leftarrow \text{verus compile}(P)$
2. 오류 선택: 우선순위에 따라 $e \in \text{errors}$ 선택
3. 복구 에이전트 호출: repair_agent ← select_repair_agent e
4. 후보 생성 및 검증

오류 우선순위:
$$\text{Priority} = [\text{Type Error} \succ \text{Bound Errors} \succ \text{Invariant Errors} \succ \cdots]$$

#### 2.3 핵심 수식

**Houdini 알고리즘을 통한 증명 최소화:**

주어진 증명 주석 집합 $A$에서 올바른 부분집합을 선형 시간에 찾습니다:

$$\text{Houdini}(A, P) \rightarrow A' \subseteq A$$

여기서 $|A'|$은 최소이면서 $\text{verify}(P \cup A') = \text{True}$

**병합 함수:**

두 프로그램 $P_1, P_2$의 증명 주석을 병합:

$$\text{merge}(P_1, P_2) = P_{\text{pure}} \cup (G_1 \cup G_2)$$

여기서:
- $P_{\text{pure}}$: 순수 Rust AST(공통 부분)
- $G_1, G_2$: 각각의 유령 코드

#### 2.4 모델 구조

**AutoVerus 아키텍처:**

```
[입력 프로그램] 
    ↓
[Phase 1: 생성]
    ├─ LoopInvAgent: 5개 후보 생성
    ├─ Filtering: 안전성 검사 (Lynette)
    ├─ Ranking: 점수 기반 순위 지정
    └─ Merging: 후보 병합
    ↓
[Phase 2: 정제]
    ├─ Constant-Propagation Agent
    ├─ Array-Length Agent
    ├─ Quantifier Agent
    └─ Conditional-Loop-Invariant Agent
    ↓
[Phase 3: 디버깅]
    ├─ 10개 복구 에이전트 (타입, 경계, 불변식, 단언, 술어 등)
    ├─ Houdini 알고리즘
    └─ 반복적 복구
    ↓
[Verus 검증기 + Lynette (AST 분석)]
    ↓
[정확한 증명 또는 실패]
```

**보조 도구:**

- **Lynette**: Verus 파서 기반 AST 분석 도구
  - 안전성 검사: 원본 Rust 코드 변경 감지
  - 병합: 유령 코드의 AST 기반 병합
  - Houdini 지원: 증명 주석 최소화
  
#### 2.5 성능 결과

**전체 성능:**

$$\text{Success Rate} = \frac{137}{150} = 91.3\%$$

| 벤치마크 소스 | 작업 수 | AutoVerus 성공 | Phase 1 | Phase 2 | Phase 3 | Baseline |
|---|---|---|---|---|---|---|
| CloverBench | 11 | 11 (100%) | 7 | 2 | 2 | 8 |
| Diffy | 38 | 38 (100%) | 26 | 12 | 0 | 5 |
| MBPP | 78 | 68 (87%) | 36 | 4 | 28 | 43 |
| Misc | 23 | 20 (87%) | 9 | 4 | 7 | 11 |
| **합계** | **150** | **137 (91.3%)** | **78** | **22** | **37** | **67** |

**효율성:**

- **평균 LLM 호출 수**: 작업당 8.8회
- **처리 시간**: 122개 작업은 첫 시도에서 성공
- **비용 효율성**: 3백만 입력 토큰 + 1.5백만 출력 토큰 = 약 $37 (Azure GPT-4o 기준)

**Baseline 대비 비교:**

$$\text{AutoVerus vs Baseline (30초 예산):} 81 \text{ vs } <40 \text{ tasks}$$

***

### 3. 모델의 일반화 성능 향상 가능성

#### 3.1 현재 성능의 일반화 특성

**일반화 강점:**

1. **모델 독립성**: AutoVerus 프레임워크는 GPT-4o, GPT-4-turbo, Deepseek-R1 모두에서 유사한 성능을 보여 일반화성이 우수합니다.

```math
\text{Model Generalization} = \{\text{GPT-4o}: 91.3\%, \text{GPT-4-turbo}: 90.1\%, \text{Deepseek-R1}: 88.5\%\}
```

2. **벤치마크 다양성**: 서로 다른 출처의 벤치마크에서 일관된 성능:
   - Diffy (루프 중심): 100% 성공
   - CloverBench (알고리즘): 100% 성공
   - MBPP (다양한 과제): 87% 성공

3. **필요한 증명 주석의 유사성**: 
   - AutoVerus 생성 증명과 인간 작성 증명의 필요한 루프 불변식 수가 비슷
   - 이는 일반화된 증명 전략을 학습함을 시사

**불필요한 주석 문제:**

$$\text{Unnecessary Invariants:} \begin{cases} \text{AutoVerus} &: 9.6 / \text{task (Diffy)} \\ \text{Human} &: 2.0 / \text{task} \end{cases}$$

이는 AutoVerus가 "안전한" 접근으로 과도한 주석을 생성하지만, 오류 아래로 자동 제거 가능합니다.

#### 3.2 일반화 성능 향상 기법

**1. 자가 진화 접근법 (Self-Evolution)**

최근 연구(SAFE, 2024)에서 제시된 접근법:

$$\text{SAFE Framework} = \begin{cases} \text{Data Synthesis} &: \text{검증기로부터 피드백을 통한 합성 증명 생성} \\ \text{Fine-tuning} &: \text{합성 증명으로 모델 미세조정} \\ \text{Self-debugging} &: \text{부정확한 증명으로부터 학습} \end{cases}$$

결과: Open-source 모델의 성능을 GPT-4o의 14.39%에서 **52.52%로 향상**[1]

**2. 데이터 증강 전략**

$$\text{Data Synthesis Cycle:}$$
- 초기 증명 생성 → 검증기 피드백 수집 → 오류 주석 달기 → 재학습
- 수만 개의 합성 증명을 생성하여 새로운 LLM 모델 학습

**3. 도메인 적응 (Domain Adaptation)**

현재 AutoVerus의 한계:

**산술 오버플로우 문제** (6개 작업 실패):
$$\text{Desired:} \text{sum} \leq i64::\text{MAX} \times \text{index}$$
$$\text{LLM Attempt:} \text{sum} + \text{arr[index]} \leq i128::\text{MAX}$$

→ 경계 근사(Bound Approximation) 전략의 LLM 학습 필요

**Vstd 라이브러리 지식 부족** (5개 작업 실패):
$$\text{Solution:} \text{Vstd API 문서 포함 in-context learning}$$

#### 3.3 일반화 향상 실증 분석

**Houdini 알고리즘의 기여도:**

$$\text{Houdini-Assisted Proofs} = 16 \text{ tasks}$$
- Phase 1: 5개 작업
- Phase 2: 6개 작업  
- Phase 3: 5개 작업

이는 **과도한 주석에서 최소 증명으로의 추상화**를 지원하여 일반화 가능한 증명 코어를 추출합니다.

**수리 복구 에이전트의 효과:**

| 복구 에이전트 | 사용 횟수 | 성공률 | 영향 |
|---|---|---|---|
| AssertFail | 310 | 46.8% | 가장 많이 사용, 낮은 성공률 |
| InvFailEnd | 85 | 91.8% | 높은 성공률 |
| PostCondFail | 26 | 69.2% | 적용 범위 제한적 |

→ 특정 오류 유형(InvFailEnd)에 대한 높은 일반화 능력 입증

***

### 4. 한계 및 실패 사례 분석

#### 4.1 기술적 한계 (13개 작업 실패)

**1. 산술 경계 근사 실패** (6/150 작업):

```math
\text{Challenge:} \sum_{i=0}^{n} \text{arr}[i] \leq \text{MAX\_VALUE}
```

LLM은 세부 항을 직접 검증하려 하고, 전체 합에 대한 경계 추론을 못 함.

**해결책**:
- 경계 명제(Bound Predicate) 라이브러리 구축
- 증명 함수(Proof Function)를 통한 단계적 경계 증명

**2. Vstd 라이브러리 API 이해 부족** (5/150 작업):

```math
\text{APIs}: \text{Seq::no\_duplicates(), Seq::to\_set(), Seq::filter(), ...}
```

이들 API의 공리(axiom)와 보조정리(lemma)를 LLM이 알지 못함.

**해결책**:
- In-context learning에 Vstd 문서 포함
- 검색 기반 구강 생성(RAG) 적용 고려

**3. 복잡한 논리 추론** (2/150 작업):

예: 재귀 사양 함수와 다중 후제 조건의 결합

$$\text{Post-condition:} \forall k: 0 \leq k < n \land \text{condition} \Rightarrow \text{property}(k)$$

***

#### 4.2 확장성 문제

**단일 함수 제한:**

현재 AutoVerus는 단일 함수를 입력 단위로 사용합니다.

$$\text{Future Challenge:} \begin{cases} \text{함수 간 의존성} &: \text{재귀적 호출, 사양 함수 체인} \\ \text{대규모 시스템 코드} &: \text{수백~수천 줄의 복잡한 상호작용} \\ \text{동시성 검증} &: \text{Verus의 선형 타입 활용 미흡} \end{cases}$$

***

### 5. 논문의 영향과 향후 연구 방향

#### 5.1 학계·산업에 미치는 영향

**1. LLM 기반 형식 검증의 새로운 패러다임**[2]

AutoVerus는 다음을 입증했습니다:
- 학습 데이터 부족 상황에서도 전문가 지식 통합으로 높은 성능 달성 가능
- LLM의 창의성과 형식 방법의 엄밀성 결합의 효과성

$$\text{Framework} = \{\text{Human Expertise}, \text{LLM Creativity}, \text{Formal Methods}\}$$

이는 증명 생성 이외에 **프로그램 합성, 버그 탐지, 최적화** 등으로 확장 가능

**2. Verus 검증 도구의 실용화 가속**

$90\%$ 이상의 자동 증명 생성은 Rust 개발자에게 형식 검증의 진입 장벽을 대폭 낮춥니다.
- 현재: 증명 작성에 시간 투자 필요
- AutoVerus 통합 후: 자동 생성 증명 → 개발자 검증 → 최종화

**3. 벤치마크 생태계 구축**

Verus-Bench (150개 작업)는 **첫 공개 Verus 증명 벤치마크**로서:
- 향후 Verus 관련 AI 연구의 기초 제공
- 모델 성능 비교의 표준화

#### 5.2 최신 연구 트렌드 및 고려사항

**1. 자가 진화 모델 (Self-Evolution Models, 2024-2025)**

최근 SAFE 및 RustBrain 등의 연구:[3][1]

$$\text{SAFE:} \text{합성 데이터} + \text{미세조정} + \text{자동 디버깅}$$
- Open-source 모델을 52.52% 성공률로 향상 (vs. GPT-4o 14.39%)
- **저비용 대안 제공**

**2. 에이전틱 형식 검증 (Agentic Formal Verification, 2024)**

ProofWright 등의 새로운 접근:[4]
- LLM 에이전트 네트워크를 통한 점진적 검증
- CUDA, GPU 등 새로운 영역으로 확장
- **멀티 에이전트 조율의 중요성 강조**

**3. 모드별 혼합 (Hybrid Approaches)**

최신 연구들의 공통 방향:

$$\text{Hybrid} = \begin{cases} \text{LLM-based 초기 가설} &: \text{빠른 속도} \\ \text{SMT Solver 검증} &: \text{엄밀성} \\ \text{정적 분석 최적화} &: \text{효율성} \\ \text{신경망 가이드} &: \text{탐색 공간 축소} \end{cases}$$

#### 5.3 향후 연구 시 고려할 점

**1. 데이터 누수(Data Leakage) 미니화**

$$\text{현황:} \begin{cases} \text{Diffy (38개)} &: \text{공개 불가} \\ \text{MBPP (78개)} &: \text{공개 불가} \\ \text{CloverBench (11개)} &: \text{2024년 5월 이후 공개} \\ \text{Misc (23개)} &: \text{온라인 튜토리얼 기반} \end{cases}$$

**권장사항**: 엄격한 데이터 격리 및 재현성 검증

**2. 확장성 전략**

$$\text{3대 과제:} \begin{cases} \text{코드 의존성} &\rightarrow \text{프로그램 분석 + LLM 하이브리드} \\ \text{사양 추론} &\rightarrow \text{반자동 사양 생성} \\ \text{증명 다양성} &\rightarrow \text{고급 Verus 기능 학습} \end{cases}$$

**3. 모델 일반화 강화**

최근 제안된 기법:[5][3]

**기법 A: 동적 In-Context Learning**

$$\text{Prompt} = \text{Task} + \text{Retrieve}(\text{Task}, \text{ProofDB})$$

**기법 B: 점진적 전문화 (Progressive Specialization)**

```math
\text{Model}_t = \text{Finetune}(\text{Model}_{t-1}, \text{Synthetic\_Data}_t)
```

**4. 다중 언어 검증 확장**

최신 동향 (2025):
- **Lean 4**: 정리 증명 자동화
- **Dafny**: 프로그램 검증
- **F∗**: 사양 검증

AutoVerus의 3단계 프레임워크는 이들 도구에도 적용 가능[6][7]

**5. 형식 검증의 신뢰성 평가**

$$\text{질문:} \text{자동 생성된 증명의 신뢰도는?}$$
- 현재: 100% (Verus 검증 통과)
- 고려 사항: Verus 자체의 버그 가능성, SMT 솔버 한계

→ **메타 검증(Meta-verification)** 및 **다중 독립적 검증기 사용** 권고

***

### 결론

AutoVerus는 **LLM 기반 증명 자동화의 새로운 이정표**를 제시합니다. 91.3%의 높은 성공률은 단순히 기술적 성과를 넘어, 다음을 의미합니다:

1. **인간 전문가 지식의 활용**: 3단계 워크플로우로 증명 작성 전문가의 직관을 LLM에 전이
2. **형식 도구와의 협력**: Houdini 알고리즘, 정적 분석을 통한 LLM의 약점 보완
3. **실용적 가능성**: $37의 저비용으로 150개 작업의 증명 자동 생성

앞으로의 연구는 **자가 진화 모델**, **멀티 에이전트 조율**, **도메인 지식 통합**에 초점을 맞춰야 하며, 이를 통해 더욱 복잡한 시스템 코드의 형식 검증 자동화가 가능해질 것으로 예상됩니다.

---

### 참고문헌 (최신 연구 기반)

 Yang, C., et al. (2025). AutoVerus: Automated Proof Generation for Rust Code. Proc. ACM Program. Lang. 9, OOPSLA2, Article 396.[2]

 Chen, T., et al. (2024). SAFE: Automated Proof Generation for Rust Code via Self-Evolution. ICLR 2025.[8]

 Hawblitzel, C., et al. (2023). Verus: Verifying Rust Programs using Linear Ghost Types. Verus-Lang.[5]

 Yang, C., et al. (2023). Leveraging Large Language Models for Automated Proof Synthesis in Rust. arXiv:2311.03739.[9]

 Ghallab, M., et al. (2024). FVEL: Interactive Formal Verification Environment with Large Language Models. arXiv:2406.14408.[10]

 Xu, K., et al. (2025). Towards Automated Formal Verification of Backend Systems with LLMs. arXiv:2506.10998.[6]

 Kobayashi, N., et al. (2025). Toward Neural-Network-Guided Program Synthesis and Verification. Formal Methods in System Design.[11]

 Petrović, M., et al. (2025). Unlocking a New Rust Programming Experience: Fast and Slow Thinking with LLMs. arXiv:2503.02335.[3]

 Hawblitzel, C., et al. Verus Documentation and Tutorial. https://verus-lang.github.io/verus/guide/[12]

 Chen, T., et al. (2024). Automated Proof Generation for Rust Code via Self-Evolution. ICLR 2025 Conference.[1]

 Xu, K., et al. (2025). Towards Automated Formal Verification of Backend Systems with LLMs. Proceedings of ICLR 2025.[13]

 ProofWright Team. (2024). ProofWright: Towards Agentic Formal Verification of CUDA. arXiv:2511.12294.[14]

 Yao, J., et al. (2025). On the Impact of Formal Verification on Software Development. OOPSLA 2025 Interview Study.[4]

 Gupta, R., et al. (2025). RustEvo: An Evolving Benchmark for API Evolution in LLM-based Rust Code Generation. arXiv:2503.16922.[15]

 Chen, X., et al. (2018). Execution-Guided Neural Program Synthesis. OpenReview ICLR.[16]

[1](https://proceedings.iclr.cc/paper_files/paper/2025/file/b2e20d7402c9985eae4ba924c65370a8-Paper-Conference.pdf)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2193484f-5a47-4a7d-bb58-39cce5eeddff/2409.13082v3.pdf)
[3](https://arxiv.org/pdf/2503.02335.pdf)
[4](https://arxiv.org/html/2511.12294v1)
[5](https://arxiv.org/pdf/2410.15756.pdf)
[6](https://arxiv.org/pdf/2303.05491.pdf)
[7](https://arxiv.org/html/2506.10998v1)
[8](http://arxiv.org/pdf/2311.03739.pdf)
[9](https://arxiv.org/pdf/2406.14408.pdf)
[10](https://arxiv.org/pdf/2305.14752.pdf)
[11](http://arxiv.org/pdf/2411.13627.pdf)
[12](https://arxiv.org/html/2503.16922v1)
[13](https://arxiv.org/abs/2103.09414)
[14](https://en.wikipedia.org/wiki/Automated_theorem_proving)
[15](https://tohoku.elsevierpure.com/en/publications/towards-neural-network-guided-program-synthesis-and-verification)
[16](https://verus-lang.github.io/verus/guide/)
[17](https://openreview.net/pdf?id=H1gfOiAqYm)
[18](https://www.sciencedirect.com/topics/computer-science/automated-theorem-proving)
[19](https://ranjitjhala.github.io/static/oopsla25-formal.pdf)
