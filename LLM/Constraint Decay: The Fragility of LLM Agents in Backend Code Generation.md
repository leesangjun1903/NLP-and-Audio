# Constraint Decay: The Fragility of LLM Agents in Backend Code Generation

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 **"Constraint Decay"(제약 붕괴)** 현상을 실증적으로 규명합니다. LLM 기반 코딩 에이전트는 구조적 제약이 없는 자유로운 코드 생성에서는 높은 성능을 보이지만, 프로덕션 수준 백엔드 개발에서 요구되는 **아키텍처 패턴, 데이터베이스, ORM 등의 구조적 비기능 요건이 누적될수록 성능이 급격히 저하**된다는 것을 보여줍니다.

### 주요 기여 (4가지)

| 기여 항목 | 내용 |
|-----------|------|
| **평가 방법론** | OpenAPI 명세 기반 평가 파이프라인: 기능 정확성과 구조적 준수를 분리 측정; 80개 greenfield + 20개 feature-implementation 태스크 오픈소스 공개 |
| **Constraint Decay (RQ1)** | 구조적 요건 누적 시 최강 모델도 평균 30 pp A% 하락 실증 |
| **Framework Sensitivity (RQ2)** | 동일 API 계약 하에서 프레임워크 선택이 성능에 25~32 pp 차이 유발 |
| **Root Cause Taxonomy (RQ3)** | 실패의 ~71%가 로직 오류이며, 그 중 데이터 레이어 결함(잘못된 쿼리 구성 + ORM 런타임 오류)이 ~45%를 차지 |

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계 상세 설명

### 2.1 해결하고자 하는 문제

기존 LLM 코드 생성 벤치마크(SWE-Bench, BaxBench 등)는 다음 한계를 가집니다:
- **기능적 정확성만 측정**: 아키텍처 패턴, ORM 통합 등 비기능 요건 무시
- **단일 파일 또는 느슨한 명세**: 멀티파일 레포지토리 수준의 구조적 복잡성 미반영
- **제약 누적 효과 미측정**: 제약이 쌓일수록 성능이 어떻게 변하는지 체계적 분석 부재

**연구 질문(RQ)**:
- **RQ1**: 구조적 제약의 누적이 성능에 어떤 영향을 미치는가?
- **RQ2**: 동일 API 계약 하에서 웹 프레임워크 선택이 성능에 영향을 주는가?
- **RQ3**: 제약된 백엔드 생성에서 에이전트 실패의 주요 원인은 무엇인가?

---

### 2.2 제안하는 방법

#### 2.2.1 태스크 설계

**단일 API 계약 고정**: RealWorld Conduit OpenAPI 3.0 명세(19개 CRUD 엔드포인트, 5개 리소스 그룹)를 모든 태스크에 공통 적용.

**세 가지 직교(orthogonal) 비기능 제약 축**:

$$\mathcal{C} = \{\text{Architecture},\ \text{Database},\ \text{ORM}\}$$

**제약 수준(Constraint Level)**: 제약을 순차적으로 추가하여 4단계 정의:

$$L_k = \text{WF} \cup \{c_1, c_2, \ldots, c_k\}, \quad k \in \{0, 1, 2, 3\}$$

| 수준 | 활성 제약 | 변형 수 |
|------|-----------|---------|
| $L_0$ | WF only (baseline) | 8 |
| $L_1$ | WF+Arch \| WF+SQLite \| WF+PG | 24 |
| $L_2$ | WF+Arch+SQLite \| WF+Arch+PG \| WF+SQLite+ORM \| WF+PG+ORM | 32 |
| $L_3$ | WF+Arch+SQLite+ORM \| WF+Arch+PG+ORM | 16 |

8개 프레임워크(Flask, FastAPI, Django, aiohttp, Express, Fastify, Hono, Koa) × 10개 제약 조합 = **총 80개 generation 태스크**

#### 2.2.2 평가 지표

**Assert% (A%)**:

$$A\% = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \frac{\text{(assertions passed in run } t\text{)}}{\text{(total assertions)}} \times 100$$

**pass@k** (Chen et al., 2021의 불편 추정량):

$$\text{pass@}k = \mathbb{E}\left[1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}\right]$$

여기서 $n$은 총 시도 횟수, $c$는 성공한 시도 횟수 ($n=3$으로 설정).

**태스크 성공 기준**:

$$\text{success}(t) = \mathbb{1}[\text{behavioral valid}(t)] \wedge \bigwedge_{c \in \mathcal{C}_t} \text{verifier}_c(t)$$

#### 2.2.3 Per-constraint 한계 효과 측정 (Matched-pair Design)

각 제약 $c$의 한계 효과:

$$\Delta(A\%)_c = \overline{A\%_{\text{with } c} - A\%_{\text{without } c}}$$

| 제약 | 평균 $\Delta$ (pp) |
|------|-------------------|
| Clean Architecture | $-9.1 \pm 1.6$ |
| PostgreSQL | $-19.3 \pm 2.5$ |
| SQLite | $-14.3 \pm 2.5$ |
| SQLAlchemy | $-1.5 \pm 2.1$ |
| Sequelize | $-0.6 \pm 2.2$ |

#### 2.2.4 Verifier 함수

세 가지 **정적 검증기(static verifier)**를 도입하여 구조적 준수 여부를 독립적으로 측정:

**(i) Architecture Verifier**: Clean Architecture 4계층(routes→services→models→repository) 존재 여부 + 의존성 방향 위반(하위 계층에서 상위 계층 import) 탐지

**(ii) Database Verifier**: 지정 DB 엔진(SQLite/PostgreSQL) 사용 증거 패턴 regex 탐색, 대안 DB 사용 여부 확인

**(iii) ORM Verifier**: 지정 ORM(SQLAlchemy/Sequelize) import 증거 패턴 탐색

---

### 2.3 모델 구조 (에이전트 아키텍처)

두 가지 오픈소스 에이전트 스캐폴드를 사용:

**① Mini-SWE-Agent**: ~100줄 Python으로 구성된 경량 스캐폴드; bash 명령으로만 환경과 상호작용; 최대 300 iteration

**② OpenHands**: 파일 편집, 터미널 실행, 코드 검색, 작업 추적 전용 도구를 갖춘 풀-피처 에이전트 프레임워크; 최대 200 iteration

**평가된 모델 4개 티어**:

| 티어 | 모델 |
|------|------|
| Small open agentic | Devstral-Small (24B), Qwen3-Coder-Next (80B) |
| Large open instruct | Qwen3-235B-Instruct (235B total / 22B active) |
| Large open agentic | MiniMax-M2.5, Kimi-K2.5 |
| Closed | GPT-5-mini, GPT-5.2 |

**실행 파이프라인**: Docker 격리 환경에서 Build Phase(에이전트 코드 생성) → Evaluate Phase(Postman Newman 기반 행동 테스트, 32 requests, 291 assertions) 순차 실행.

---

### 2.4 성능 결과

**전체 성능 저하 (RQ1)**:

$$\overline{\Delta A\%}_{L_0 \to L_3} \approx -30 \text{ pp (상위 8개 구성 기준, 상대적 40\% 손실)}$$

| Agent | Model | $A\%_{L_0}$ | $A\%_{L_3}$ | $\Delta A\%$ |
|-------|-------|------------|------------|-------------|
| Mini-SWE | GPT-5-mini | 51.7 | 23.7 | -28.0 |
| OpenHands | GPT-5-mini | 65.8 | 52.2 | -13.6 |
| Mini-SWE | Qwen3-Coder-Next | 86.4 | 46.1 | -40.2 |
| OpenHands | Qwen3-Coder-Next | 73.0 | 27.6 | -45.5 |
| Mini-SWE | MiniMax-M2.5* | 88.6 | 58.3 | -30.3 |
| Mini-SWE | Kimi-K2.5* | 85.4 | 53.7 | -31.7 |
| Mini-SWE | GPT-5.2* | 78.2 | 48.0 | -30.2 |

**프레임워크 민감도 (RQ2)**:

| 프레임워크 | 평균 $A\%$ | 특성 |
|-----------|-----------|------|
| Express | 51.4 | 경량, 명시적 |
| Koa | 50.7 | 경량, 명시적 |
| Flask | 49.3 | 경량, 명시적 |
| aiohttp | 38.4 | 중간 |
| Fastify | 31.7 | 중간 |
| Django | 25.4 | 관례 중심 |
| FastAPI | 24.2 | 관례 중심 |
| Hono | 18.5 | Edge runtime |

**실패 원인 분류 (RQ3)**:

- 로직 오류: ~71% (양 모델 공통)
- 서버 시작 실패: 12~21%
- 로직 오류 세부: 데이터 레이어 결함(잘못된 쿼리 로직 25.5% + ORM 런타임 오류 21.2% = **~46.7%**)

---

### 2.5 한계

- **단일 API 명세**: RealWorld Conduit API 하나만 사용 → 다른 도메인으로의 일반화 불확실
- **정적 검증기의 한계**: 소스 텍스트만 분석; 런타임에서 실제 ORM을 사용하지 않아도 통과 가능 (false positive)
- **비용 제약**: 일부 모델(MiniMax-M2.5, Kimi-K2.5, GPT-5.2)은 16개 태스크 서브셋만 평가
- **언어/런타임 제한**: Python 3.12, Node.js 20만 포함; Java, Go, Rust 등 미포함
- **n=3 제한**: 각 태스크당 3회만 실행 → pass@1 노이즈 큼

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화의 한계 분석

논문은 LLM 에이전트의 일반화 실패를 다음 차원에서 진단합니다:

**① 구조적 제약 간 일반화 실패**

$$\text{Gap} = A\%_{L_0} - A\%_{L_3} \approx 30 \text{ pp}$$

에이전트가 L0에서 학습한 "기능적 코드 생성 패턴"을 L3의 구조화된 환경에 전이하지 못함. 이는 모델이 **제약 조합(constraint composition)**에 대한 일반화 능력이 결여되어 있음을 의미합니다.

**② 프레임워크 간 일반화 실패**

Flask에서 잘 작동하는 에이전트가 FastAPI/Django에서 25~32 pp 성능 하락을 보임. 이는 모델이 **관례 중심(convention-heavy) 프레임워크의 암묵적 규칙**을 일반화하지 못한다는 증거입니다.

**③ Greenfield → Feature Implementation 전이 실패**

Feature implementation 태스크(기존 코드베이스 기반)에서도 유사하게 낮은 성능:

$$\text{pass@1}_{\text{feature}} \leq 55\% \quad (\text{GPT-5.2, 최고 성능})$$

이는 제약 붕괴가 "처음부터 생성"의 아티팩트가 아니라 **구조적 이해 능력 자체의 한계**임을 시사합니다.

---

### 3.2 일반화 성능 향상을 위한 방향성 (논문에서 제시 또는 도출 가능한 것)

#### 방향 1: Retrieval-Augmented Framework Documentation

논문 저자들이 직접 제안하는 방향: 에이전트가 프레임워크별 공식 문서를 런타임에 검색하여 관례를 학습하는 RAG 기반 접근법.

$$\text{Context}_{\text{agent}} = \text{Prompt} \oplus \text{RAG}(\text{framework docs}, q)$$

이를 통해 FastAPI의 타입 힌트 기반 검증, Django의 자동 발견 메커니즘 등을 명시적으로 학습 가능.

#### 방향 2: Constraint-Oriented Planning

현재 에이전트는 비기능 요건을 후처리하는 경향이 있음. 대신 생성 시작 전 제약 그래프를 명시적으로 구성하는 계획 단계를 추가:

$$\text{Plan} = f_{\text{plan}}(\text{OpenAPI}, \mathcal{C}) \rightarrow \text{Code}$$

예: Clean Architecture → DB → ORM 순으로 의존성을 역방향으로 구현하는 체계적 계획.

#### 방향 3: Targeted Pre-training / Fine-tuning on Convention-Heavy Codebases

관례 중심 프레임워크(FastAPI, Django)의 실제 코드베이스로 파인튜닝:

$$\mathcal{L}_{\text{FT}} = \mathcal{L}_{\text{LM}}(\text{convention-heavy codebases}) + \lambda \cdot \mathcal{L}_{\text{constraint}}$$

Devstral-Small(24B)이 near-zero 성능을 보인 이유 중 하나가 관련 학습 데이터 부족임을 논문이 시사함.

#### 방향 4: 다중 에이전트 분업 구조

데이터 레이어 결함이 ~45% 실패를 유발하므로, ORM/SQL 전문 서브에이전트를 분리:

$$\text{Agent}_{\text{main}} \rightarrow \{\text{Agent}_{\text{arch}},\ \text{Agent}_{\text{db}},\ \text{Agent}_{\text{orm}}\}$$

MetaGPT(Hong et al., 2024) 스타일의 다중 에이전트 협업으로 제약별 전문화 가능.

#### 방향 5: 검증 루프(Verification Loop) 내재화

현재 에이전트는 정적 검증기 피드백을 생성 중에 활용하지 않음. 자기 검증 루프를 추가:

$$\text{Code}^{(t+1)} = \text{Agent}\left(\text{Code}^{(t)},\ \text{Verifier}(\text{Code}^{(t)})\right)$$

이를 통해 구조적 제약 위반을 실시간으로 수정하는 반복적 생성 가능.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### ① 벤치마크 패러다임의 전환

이 논문은 기존의 **"기능적 정확성 중심 평가"에서 "기능 + 구조적 준수의 동시 평가"로의 패러다임 전환**을 촉구합니다. SWE-Bench류 벤치마크가 포화 상태에 이르는 시점에서, 구조적 제약 밀도를 독립 변수로 다루는 새로운 평가 축을 제공합니다.

#### ② 코딩 에이전트 설계에 대한 시사점

에이전트 스캐폴드(Mini-SWE-Agent vs. OpenHands)가 모델에 따라 다른 성능 패턴을 보임:
- GPT-5-mini: OpenHands > Mini-SWE (+14.1 pp at L0)
- Qwen3-Coder-Next: Mini-SWE > OpenHands (+13.4 pp at L0)

이는 **모델-스캐폴드 상호작용(model-scaffold interaction)**이 중요한 연구 변수임을 시사합니다.

#### ③ 데이터 레이어 전문화의 필요성

ORM 런타임 오류와 잘못된 쿼리 구성이 실패의 ~45%를 차지한다는 발견은, **데이터베이스/ORM 특화 파인튜닝 데이터셋** 구축의 필요성을 강력히 뒷받침합니다.

#### ④ 프레임워크별 교육 데이터 불균형 문제

Hono의 낮은 성능(avg. 18.5%)은 Edge runtime 관련 학습 데이터 부족에서 기인할 가능성이 있어, **사전학습 데이터의 프레임워크별 분포** 연구의 필요성을 제기합니다.

---

### 4.2 앞으로 연구 시 고려할 점

#### ① 제약 공간의 확장

현재 연구는 3개 제약 차원(Architecture, DB, ORM)만 다룸. 향후 연구에서 고려할 추가 차원:
- **보안 제약**: 인증/인가 패턴(OAuth2, JWT 구현 규칙)
- **성능 제약**: 캐싱 계층, 비동기 처리 패턴
- **API 버저닝**: 하위 호환성 유지 요건
- **테스트 커버리지**: 특정 테스트 전략(TDD) 준수

#### ② 다국어/멀티런타임 확장

현재 Python + Node.js만 다룸. Java(Spring Boot), Go(Gin/Echo), Rust(Axum) 등 정적 타입 언어로 확장 시 결과가 달라질 수 있음. 특히 정적 타입 언어는 컴파일러가 일부 구조적 제약을 강제하므로 다른 실패 패턴이 나타날 가능성이 있습니다.

#### ③ 동적 검증기 개발

현재 정적 패턴 매칭 기반 검증기는 false positive(import는 했지만 실제로 사용하지 않는 경우) 문제가 있음. 향후 연구에서는:

$$\text{Verifier}_{\text{dynamic}} = \text{Runtime analysis} + \text{Code coverage tracing}$$

와 같은 동적 분석 기반 검증기로 측정 정확도를 높여야 합니다.

#### ④ 모델-제약 상호작용의 세밀한 분석

Pearson $r = 0.976$, Spearman $\rho = 0.948$으로 서브셋 대표성은 높지만, 일부 구성에서 편차(예: OpenHands + Qwen3-Coder-Next의 L3에서 +11.0 pp 차이)가 관찰됨. 더 큰 $n$(현재 $n=3$)으로 신뢰 구간을 좁히는 것이 필요합니다.

#### ⑤ 비용-성능 트레이드오프 분석

OpenHands는 Mini-SWE-Agent 대비 **12.9배 더 많은 토큰** 소비. 성능 향상이 비용 증가를 정당화하는지 체계적 분석 필요:

$$\text{Efficiency} = \frac{\Delta A\%}{\Delta \text{Cost (tokens)}}$$

#### ⑥ 인간-에이전트 협업 시나리오

논문은 "에이전트가 rapid prototyping에는 유용하지만 production에는 부적합"하다고 결론. 향후 연구는 **에이전트가 어느 시점에 인간 개발자에게 제어권을 넘겨야 하는지** 결정하는 신뢰도 추정(uncertainty estimation) 메커니즘을 탐구해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 평가 대상 | 제약 다양성 | 멀티파일 | 비교 포인트 |
|------|------|-----------|------------|---------|------------|
| **SWE-Bench** (Jimenez et al.) | 2024 | GitHub Issue 해결 | 없음 | O | 실제 이슈 기반이나 구조적 제약 미측정; 현재 포화 상태 |
| **BaxBench** (Vero et al.) | 2024 | 백엔드 생성 (OpenAPI 기반) | 없음 | X (단일 파일) | 본 논문과 가장 유사하나 단일 파일 + 직접 prompting; 제약 누적 효과 미연구 |
| **Commit0** (Zhao et al.) | 2025 | 라이브러리 생성 (스켈레톤 제공) | 없음 | O | 스켈레톤 기반이라 greenfield와 다름 |
| **NL2Repo-Bench** (Ding et al.) | 2025 | NL 명세 → 레포지토리 | 없음 | O | Python 코드베이스만; 비기능 요건 미포함 |
| **SWE-Bench Pro** (Deng et al.) | 2025 | Long-horizon SE 태스크 | 없음 | O | SWE-Bench 확장이나 구조적 제약 없음 |
| **MultiSWE-Bench** (Zan et al.) | 2025 | 다국어 Issue 해결 | 없음 | O | 언어 다양성 확장이나 비기능 요건 미포함 |
| **FEA-Bench** (Li et al.) | 2025 | Feature 구현 | 암묵적 | O | Feature 추가 평가; 제약은 기존 코드베이스에 내재 |
| **R2E-Gym** (Jain et al.) | 2025 | SWE 에이전트 학습 | 없음 | O | 환경 생성 자동화 초점 |
| **RPG** (Luo et al.) | 2026 | 라이브러리 생성 | 없음 | O | 비구조화된 NL 프롬프트 기반; 구조 제약 없음 |
| **본 논문** (Dente et al.) | 2026 | 백엔드 생성 (OpenAPI 기반) | **명시적 4단계** | O | **유일하게 제약 누적 효과를 체계적으로 측정** |

### 핵심 차별점

본 논문이 기존 연구 대비 갖는 가장 중요한 차별점은:

1. **단일 API 계약 고정 + 제약 변수 독립 조작**: 내적 타당성(internal validity) 극대화
2. **행동 테스트와 구조적 검증의 분리**: 기능 정확성 ≠ 구조적 준수임을 명확히 구분
3. **제약 수준별 성능 궤적 추적**: L0→L3의 연속적 하락 패턴 규명

---

## 참고 자료 (논문 내 인용 및 본 답변에서 직접 참조)

1. **Francesco Dente, Dario Satriani, Paolo Papotti.** "Constraint Decay: The Fragility of LLM Agents in Backend Code Generation." arXiv:2605.06445v1 [cs.SE], 7 May 2026. *(본 분석의 주 대상 논문)*

2. **Jimenez et al.** "SWE-bench: Can language models resolve real-world github issues?" ICLR 2024.

3. **Mark Vero et al.** "Baxbench: Can LLMs generate correct and secure backends?" ICLR 2025 Third Workshop on Deep Learning for Code, 2024.

4. **Xingyao Wang et al.** "Openhands: An open platform for AI software developers as generalist agents." ICLR 2025.

5. **John Yang et al.** "SWE-Agent: Agent-computer interfaces enable automated software engineering." NeurIPS 2024.

6. **SWE-agent Team.** "mini-swe-agent: The 100 line AI agent that's actually useful." GitHub, 2025.

7. **Wenting Zhao et al.** "Commit0: Library generation from scratch." ICLR 2025.

8. **Mark Chen et al.** "Evaluating large language models trained on code." arXiv:2107.03374, 2021. *(pass@k 추정량)*

9. **Robert C. Martin.** "Clean Architecture: A Craftsman's Guide to Software Structure and Design." Prentice Hall, 2017.

10. **Sirui Hong et al.** "MetaGPT: Meta programming for a multi-agent collaborative framework." ICLR 2024.

11. **Wei Li et al.** "FEA-bench: A benchmark for evaluating repository-level code generation for feature implementation." ACL 2025.

12. **Daoguang Zan et al.** "MultiSWE-bench: A multilingual benchmark for issue resolving." NeurIPS 2025 D&B Track.

13. **Xiang Deng et al.** "SWE-Bench Pro: Can AI agents solve long-horizon software engineering tasks?" arXiv:2509.16941, 2025.

14. **Jane Luo et al.** "RPG: A repository planning graph for unified and scalable codebase generation." ICLR 2026.

15. **An Yang et al.** "Qwen3 technical report." arXiv:2505.09388, 2025.

16. **Abhinav Rastogi et al.** "Devstral: Fine-tuning language models for coding agent applications." arXiv:2509.25193, 2025.

---

> **⚠️ 정확도 관련 고지**: 본 논문은 arXiv:2605.06445v1로 **2026년 5월 7일** 기재된 preprint로, 현재(2025년 기준) 미래 시점의 논문입니다. 논문 원문에 명시된 수치, 실험 결과, 인용 문헌을 그대로 반영하였으며, GPT-5.2, Kimi-K2.5 등 2026년 모델명은 논문 원문 기재 그대로 사용하였습니다.
