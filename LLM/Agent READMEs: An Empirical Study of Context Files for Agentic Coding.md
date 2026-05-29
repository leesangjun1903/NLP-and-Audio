# Agent READMEs: An Empirical Study of Context Files for Agentic Coding

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 **Agent Context Files(에이전트 컨텍스트 파일, ACF)**—예: `CLAUDE.md`, `AGENTS.md`, `copilot-instructions.md`—에 대한 **최초의 대규모 실증 연구**를 수행합니다. 이 파일들은 AI 코딩 에이전트(Claude Code, OpenAI Codex, GitHub Copilot)가 프로젝트를 이해하고 동작하는 데 필요한 **지속적(persistent) 프로젝트 수준의 지시사항**을 담고 있습니다.

핵심 주장은 다음 세 가지입니다:

1. **ACF는 단순한 정적 문서가 아니라**, 설정 코드처럼 지속적으로 진화하는 복잡한 아티팩트이다.
2. **개발자들은 기능적 맥락(빌드, 아키텍처, 구현 세부 사항)을 우선시**하며, 보안·성능과 같은 비기능 요구사항(NFR)은 심각하게 부족하다.
3. **LLM 기반 자동 분류(GPT-5)**가 구체적·기능적 카테고리에 대해 micro F1 = 0.79의 높은 성능을 달성하며, 대규모 모니터링 가능성을 제시한다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| 첫 번째 대규모 실증 연구 | 1,925개 저장소의 2,303개 ACF 분석 |
| 구조·가독성 분석 | Flesch Reading Ease(FRE), 단어 수, 마크다운 헤더 계층 분석 |
| 유지보수 패턴 분석 | 커밋 이력 기반 진화 패턴 분석 |
| 16개 명령어 분류 체계 | 수동 레이블링 + LLM 자동 분류 |
| 실무 가이드라인 제시 | 연구자·개발자·도구 개발자를 위한 시사점 제공 |

---

## 2. 해결하려는 문제, 방법, 구조, 성능, 한계

### 2.1 해결하려는 문제

기존 LLM 보조 소프트웨어 공학 연구는 모델 능력, 상호작용 패턴, 즉각적인 프롬프트에 집중되어 있었습니다. 반면 실제 개발 현장에서 팀이 이미 사용하고 있는 **에이전트 컨텍스트 파일**에 대한 실증적 증거는 전무했습니다.

구체적으로 해결하고자 한 문제:

- ACF의 구조적 특성(크기, 가독성, 계층 구조)은 어떤가?
- ACF가 정적 문서처럼 유지되는가, 아니면 코드처럼 진화하는가?
- 개발자들은 어떤 종류의 지시사항을 ACF에 포함시키는가?
- ACF 내용 분류를 자동화할 수 있는가?

### 2.2 연구 방법론

#### 데이터 수집

- **AIDev 데이터셋**에서 GitHub 스타 5개 이상인 8,370개 저장소를 추출
- GitHub API를 통해 `CLAUDE.md`, `AGENTS.md`, `copilot-instructions.md` 검색
- 최종 수집: Claude Code 922개, OpenAI Codex 694개, GitHub Copilot 687개 (총 2,303개)

#### RQ1: 구조적 특성 분석

**가독성 측정**: Flesch Reading Ease (FRE) 지수 사용

$$
\text{FRE} = 206.835 - 1.015 \left(\frac{\text{총 단어 수}}{\text{총 문장 수}}\right) - 84.6 \left(\frac{\text{총 음절 수}}{\text{총 단어 수}}\right)
$$

**단어 수 측정**: 정규식 `\w+` 패턴으로 단어 토큰 카운팅

**구조 분석**: 마크다운 헤더 H1~H5 계층별 카운팅

**통계 검정**: Mann-Whitney U 검정 ($\alpha = 0.05$), Cliff's delta $d$로 효과 크기 측정

$$
\text{Effect size} = \begin{cases} \text{negligible}, & \text{if } |d| \leq 0.147 \\ \text{small}, & \text{if } 0.147 < |d| \leq 0.33 \\ \text{medium}, & \text{if } 0.33 < |d| \leq 0.474 \\ \text{large}, & \text{if } 0.474 < |d| \leq 1 \end{cases} \tag{1}$$

#### RQ2: 유지보수 분석

각 저장소의 커밋 이력에서 ACF 관련 커밋을 추출:
- 타임스탬프, 추가/삭제 라인 수 집계
- 총 커밋 수 및 커밋 간 시간 간격 분석

수집된 커밋 수:
- Claude Code: 5,655개
- OpenAI Codex: 2,767개
- GitHub Copilot: 2,237개

#### RQ3: 내용 분류 (수동)

**2단계 수동 분류 접근법:**

1. **레이블 생성 단계**: H1·H2 제목 추출 → Claude Opus 4.1, Gemini 2.5 Pro, GPT-5로 후보 레이블 생성 → 80개 레이블 → 의미적 유사 레이블 병합 → **16개 핵심 레이블** 도출
2. **레이블 할당 단계**: 332개 Claude Code 파일에 2인 독립 레이블링 (합의율 80.3%), 3인 합의로 최종 2,069개 레이블 결정

#### RQ4: 자동 분류

**다중 레이블 이진 분류 문제**로 프레임화:

$$
\hat{y}_i = \text{GPT-5}\left(\text{context file}_i, \text{category descriptions}\right) \in \{0,1\}^{16}
$$

평가 지표:

$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

$$
\text{Micro-Avg } F_1 = \frac{2 \sum_i TP_i}{\sum_i (2TP_i + FP_i + FN_i)} = 0.79
$$

### 2.3 모델 구조

이 논문은 새로운 딥러닝 모델을 제안하지 않습니다. 대신 **GPT-5를 분류기로 활용한 프롬프트 기반 파이프라인**을 사용합니다:

```
입력: [ACF 전체 내용] + [16개 카테고리 설명 및 예시]
출력: 각 카테고리에 대한 이진 레이블 벡터 {0,1}^16
```

### 2.4 주요 발견(성능)

#### RQ1 결과

| 도구 | 중간값 단어 수 | 중간값 FRE | 해석 |
|------|------------|----------|------|
| GitHub Copilot | 535.0 | 26.6 | 매우 어려움 |
| Claude Code | 485.0 | 16.6 | 매우 어려움 |
| OpenAI Codex | 335.5 | 39.6 | 어려움 |

- Claude Code vs OpenAI Codex 길이 차: $d = 0.22$ (small effect)
- 계층 구조: H1 중간값 = 1 (모든 도구 동일), H2 중간값 6~7, H3 중간값 9~12

#### RQ2 결과

- Claude Code 파일의 **67.4%**가 다중 커밋으로 수정됨
- 커밋 간 중간 간격: Claude Code 24.1시간, OpenAI Codex 22.0시간, GitHub Copilot 70.7시간
- 추가 라인 중간값(Claude Code): 57.0 단어/커밋
- 삭제 라인 중간값: 15.0 단어 미만 (미미함)

#### RQ3 결과 (16개 카테고리 분포)

| 카테고리 | 비율 |
|---------|------|
| Testing | 75.0% |
| Implementation Details | 69.9% |
| Architecture | 67.7% |
| Development Process | 63.3% |
| Build and Run | 62.3% |
| System Overview | 59.0% |
| **Security** | **14.5%** |
| **Performance** | **14.5%** |
| **UI/UX** | **8.7%** |

#### RQ4 결과 (자동 분류 성능)

| 레이블 | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Architecture | 0.89 | **0.97** | **0.93** |
| Testing | **0.91** | **0.96** | **0.94** |
| Build & Run | 0.90 | 0.94 | 0.92 |
| Development Process | **0.92** | 0.76 | 0.83 |
| AI Integration | 0.33 | 0.86 | 0.48 |
| Project Management | 0.40 | 0.44 | 0.42 |
| **Micro Average** | **0.73** | **0.86** | **0.79** |

### 2.5 한계 (Threats to Validity)

| 유형 | 한계 내용 |
|------|----------|
| **내적 타당도** | 수동 분류의 주관적 편향 가능성 (단, 80.3% 합의율로 일부 완화) |
| **구성 타당도** | 이진 분류 방식 — 카테고리 존재 여부만 판단하고 깊이·풍부함은 미측정 |
| **외적 타당도** | 3개 도구(Claude Code, OpenAI Codex, GitHub Copilot)만 분석 — 다른 에이전트 도구로의 일반화 제한 |
| **샘플 편향** | GitHub 스타 5개 이상 필터링 → 소규모·비공개 프로젝트 제외 |

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 "일반화 성능"은 전통적 딥러닝 의미와는 달리, **ACF 분류 시스템과 연구 결과의 다양한 맥락으로의 확장 가능성**을 의미합니다.

### 3.1 현재 일반화의 한계

**도메인 범위 한계:**

$$
\text{현재 커버리지} = \frac{|\text{분석된 도구}|}{|\text{전체 에이전트 도구}|} = \frac{3}{?} \approx \text{제한적}
$$

- 분석 대상: Claude Code, OpenAI Codex, GitHub Copilot 3개 도구만
- Cursor, Windsurf, Devin 등 다른 에이전트 도구 미포함
- 수동 분류는 Claude Code 파일 332개에만 적용 (전체 922개 중)

**추상적 카테고리의 낮은 일반화:**

- `AI Integration` ($F_1 = 0.48$), `Project Management` ($F_1 = 0.42$), `Maintainability` ($F_1 = 0.56$)
- 이러한 카테고리는 **맥락 의존적이고 표현 방식이 다양**하여 GPT-5도 일관된 패턴 인식 실패

**언어·문화 편향:**
- 영어 중심 GitHub 공개 저장소만 분석
- 비영어권 프로젝트나 비공개 엔터프라이즈 저장소로의 일반화 불확실

### 3.2 일반화 성능 향상 가능성

#### (A) 더 넓은 도구 커버리지로 확장

논문이 제안하는 방향:
> *"Future work should extend this analysis to include a larger number of files from various agentic coding systems."*

새로운 도구별 ACF 명명 규칙 (예: `.cursorrules`, `windsurf_rules.md` 등)을 포함하면, 분류 체계의 보편적 적용 가능성을 검증할 수 있습니다.

#### (B) RAG 기반 의미론적 검색으로 분류 정확도 향상

논문이 제안하는 실용적 개선안:

$$
\text{Context}_{\text{agent}} = \text{RAG}\left(\text{task}, \text{ACF}_{\text{sections}}\right) = \arg\max_{\text{section} \in \text{ACF}} \text{sim}\left(\text{task embedding}, \text{section embedding}\right)
$$

- 현재의 단순 텍스트 청킹을 대체
- 16개 카테고리 구조를 활용한 의미론적 우선순위 부여
- 예: 버그 수정 태스크 → Debugging(24.4%), Testing 섹션 우선 검색

이를 통해 **에이전트의 태스크 특화 일반화 성능** 향상 기대:

$$
\text{Agent Performance}_{\text{task}} \propto \text{Precision}\left(\text{retrieved context} | \text{task type}\right)
$$

#### (C) 자동 분류기의 파인튜닝을 통한 일반화

현재 GPT-5 프롬프트 기반 분류의 약점을 보완하기 위해:

1. **Few-shot 예시 확충**: 현재 낮은 F1 카테고리(`AI Integration`, `Project Management`)에 더 많은 대표 예시 추가
2. **지속적 학습(Continual Learning)**: 새로운 ACF가 추가될수록 분류기 업데이트

$$
\mathcal{L}_{\text{classifier}} = -\sum_{k=1}^{16} \left[ y_k \log p_k + (1-y_k)\log(1-p_k) \right]
$$

($y_k$: 실제 레이블, $p_k$: 예측 확률)

#### (D) "Context Debt" 개념의 일반화

논문이 새롭게 제안하는 **컨텍스트 부채(Context Debt)** 개념:

> ACF의 낮은 가독성(FRE 중간값 16.6 — "매우 어려움")과 지속적 증가는, 명확한 지침을 제공하기 위한 메커니즘이 오히려 인지 부하를 높이는 역설을 만든다.

이 개념을 일반화하면:

$$
\text{Context Debt} = f(\text{FRE}^{-1}, \text{file length}, \text{NFR coverage}^{-1}, \text{staleness})
$$

이 지표를 다양한 도구와 프로젝트에 적용하면 **일반적인 ACF 품질 평가 프레임워크** 구축 가능.

#### (E) 비기능 요구사항(NFR) 포함 확대

현재 Security(14.5%), Performance(14.5%)의 극히 낮은 포함률은, 에이전트가 생성하는 코드의 **보안·성능 품질 일반화 실패**를 의미합니다.

$$
P(\text{secure code} | \text{agent}) \approx P(\text{secure code} | \text{context with security guidelines})
$$

NFR 지침이 포함된 ACF로 훈련/프롬프팅된 에이전트는 더 넓은 도메인에서 안전한 코드를 생성할 가능성이 높습니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (A) 새로운 연구 분야 개척: "컨텍스트 공학(Context Engineering)"의 실증화

이 논문은 프롬프트 공학과 구별되는 **지속적 컨텍스트 공학**이라는 새로운 연구 분야를 실증적으로 정초합니다. 향후 연구는:

- ACF 품질과 에이전트 코드 품질 간의 인과 관계 규명
- "최적 ACF"의 정의 및 측정 방법 개발

#### (B) "컨텍스트 부채(Context Debt)" 형식화

기존 기술 부채 연구의 새로운 하위 유형 도입:

> **컨텍스트 부채** = ACF가 복잡하고 불명확해져 에이전트의 동작을 예측할 수 없게 만드는 상태

향후 연구 과제:
- ACF "스멜(smell)" 탐지 방법론 (예: 모순된 지시, 불명확한 역할 정의)
- 가독성 점수를 넘어서는 ACF 유지보수성 메트릭 개발

#### (C) ACF와 코드베이스의 공진화(Co-evolution) 모델링

ACF의 67.4%가 다중 커밋으로 수정된다는 발견은 **ACF-코드 동기화 연구**의 필요성을 제기합니다:

- CI/CD 통합 ACF 린터 개발
- 예: Build and Run 섹션의 명령어와 실제 `package.json`/`Makefile` 일치 검증

$$
\text{Divergence}(t) = d\left(\text{ACF}_{\text{commands}}(t), \text{build scripts}(t)\right)
$$

#### (D) 에이전트 벤치마크 재설계

현재 SWE-bench 등 주요 벤치마크는 기능적 정확성만 측정합니다. 이 논문의 발견은 새로운 벤치마크 설계를 촉구합니다:

$$
\text{Score}_{\text{new}} = \alpha \cdot \text{Functional Correctness} + \beta \cdot \text{Security Compliance} + \gamma \cdot \text{Performance Adherence}
$$

여기서 $\beta, \gamma > 0$으로 설정하여 NFR 준수를 명시적으로 평가.

#### (E) LLM 자동 분류 파이프라인의 확장

micro F1 = 0.79의 자동 분류 결과는 **대규모 ACF 생태계 모니터링**의 가능성을 열어줍니다:

- 시간에 따른 Security·Performance 지침 트렌드 추적
- 도구별·도메인별 ACF 패턴 비교 분석 자동화

### 4.2 앞으로 연구 시 고려해야 할 점

| 고려 사항 | 구체적 내용 |
|-----------|-----------|
| **데이터 대표성 확장** | Cursor, Devin, Windsurf 등 새로운 에이전트 도구 포함 |
| **비영어권 저장소** | 한국어·중국어 등 비영어 ACF 분석 |
| **인과성 연구** | ACF 품질과 에이전트 코드 품질 간의 통제 실험 필요 |
| **NFR 지침 효과 측정** | Security/Performance 지침이 실제로 에이전트 코드 품질을 향상시키는지 검증 |
| **동적 분류 체계** | 새로운 에이전트 도구 등장에 따라 16개 카테고리 지속 갱신 필요 |
| **기업 환경 연구** | 비공개 엔터프라이즈 ACF 패턴은 공개 저장소와 다를 가능성 |
| **자동화 도구 개발** | ACF 작성 지원 IDE 플러그인, 자동 완성 기능 개발 |
| **윤리·보안 연구** | AI 에이전트가 ACF의 보안 지침을 실제로 준수하는지 검증 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 초점 | 본 논문과의 관계 |
|------|------|------|----------------|
| **SWE-bench** (Jimenez et al.) | 2024 | 에이전트의 GitHub 이슈 해결 능력 평가 | ACF 없는 에이전트 평가 — 본 논문은 ACF가 에이전트 성능에 미치는 영향의 측정 필요성 제기 |
| **RepairAgent** (Bouzenia et al.) | 2024 | LLM 기반 자율 프로그램 수리 에이전트 | 에이전트 내부 구조 연구 vs. 본 논문은 외부 설정 아티팩트 연구 |
| **Repository-Level Prompt Generation** (Shrivastava et al.) | 2022 | 저장소 수준 컨텍스트 LLM에 제공 | 본 논문의 ACF 연구와 보완적 — 자동 컨텍스트 vs. 수동 작성 컨텍스트 |
| **AutoCodeRover** (Zhang et al.) | 2024 | 자율 프로그램 개선 에이전트 | 에이전트 워크플로우 최적화 — ACF를 통한 외부 제약 설정 방법론 미연구 |
| **AIDev Dataset** (Li et al.) | 2025 | AI 에이전트 기여 저장소 큐레이션 | 본 논문의 데이터 소스로 직접 활용 |
| **MCP Security** (Hasan et al.) | 2025 | Model Context Protocol 보안 리스크 | ACF의 Security 지침 부재(14.5%)와 연결 — MCP 통합 시 보안 취약점 위험 |
| **Sharp Tools** (Kumar et al.) | 2025 | 실제 SE 태스크에서 에이전트 AI 활용 패턴 | 개발자-에이전트 상호작용 패턴 연구 vs. 본 논문은 설정 아티팩트 연구 |
| **Prompt Management** (Li et al.) | 2025 | GitHub 저장소의 프롬프트 관리 | 본 논문의 직접적 선행 연구 — ACF는 프롬프트 관리의 특수 형태 |
| **Can LLMs Replace Annotation** (Ahmed et al.) | 2025 | LLM의 수동 레이블링 대체 가능성 | 본 논문의 RQ4 동기 제공 — LLM 기반 ACF 분류 정당성 |
| **Human-In-The-Loop Agents** (Takerngsaksiri et al.) | 2024/2025 | 인간 감독 하의 소프트웨어 개발 에이전트 | ACF는 에이전트 자율성과 인간 통제 간의 균형을 위한 핵심 아티팩트 |

---

## 참고 자료

**주요 논문 (출처)**

- **Chatlatanagulchai et al. (2025)**: *Agent READMEs: An Empirical Study of Context Files for Agentic Coding.* arXiv:2511.12884v1 [cs.SE]. ← **본 분석의 주 논문**
- Jimenez et al. (2024): *SWE-bench: Can Language Models Resolve Real-world Github Issues?* ICLR 2024.
- Shrivastava et al. (2022): *Repository-Level Prompt Generation for Large Language Models of Code.* ICML 2022.
- Bouzenia et al. (2024): *RepairAgent: An Autonomous, LLM-Based Agent for Program Repair.* ICSE 2024.
- Hassan et al. (2025): *Agentic Software Engineering: Foundational Pillars and a Research Roadmap.* arXiv:2509.06216.
- Li et al. (2025): *The Rise of AI Teammates in Software Engineering (SE) 3.0.* arXiv:2507.15003.
- Li et al. (2025): *Understanding Prompt Management in GitHub Repositories.* arXiv:2509.12421.
- Hasan et al. (2025): *Model Context Protocol (MCP) at First Glance.* arXiv:2506.13538.
- Kumar et al. (2025): *Sharp Tools: How Developers Wield Agentic AI in Real Software Engineering Tasks.* CoRR abs/2506.12347.
- Ahmed et al. (2025): *Can LLMs Replace Manual Annotation of Software Engineering Artifacts?* MSR'25.
- Zhang et al. (2024): *AutoCodeRover: Autonomous Program Improvement.* ISSTA 2024.
- Chatlatanagulchai et al. (2025, prior): *On the Use of Agentic Coding Manifests: An Empirical Study of Claude Code.* PROFES'25.
- Mann & Whitney (1947): *On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other.* Annals of Mathematical Statistics.
- Romano et al. (2006): *Exploring methods for evaluating group differences.* SAIR.

> **⚠️ 정확도 주의**: 본 답변은 제공된 논문 PDF 전문을 기반으로 작성되었습니다. 논문에 명시되지 않은 내용(예: 다른 최신 논문과의 상세 비교)은 논문 내 참고문헌 목록에 실제로 인용된 연구들만을 활용하였습니다.
