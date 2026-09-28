# Context as an Environment: Programmatic Context Management for Long-Horizon Agents

> **⚠️ 중요 고지**: 본 논문은 arXiv:2608.21690v1 (2026년 8월 21일)에 게재된 Technical Report입니다. 일부 참조 문헌(2026년 발행)은 현재(2025년 기준) 존재하지 않거나 확인 불가한 미래 문헌일 수 있습니다. 이 논문 자체가 미래 날짜(2026년)를 기재하고 있어, 논문 내 인용 문헌의 검증이 불가능한 부분이 있음을 명시합니다.

---

## 1. Executive Summary (10문장 이내)

LLM 에이전트가 장기(long-horizon) 작업을 수행할 때 세션 히스토리가 단일 컨텍스트 윈도우를 초과하는 문제가 발생한다.  
기존 접근법(압축·외부 메모리)은 미래 필요를 알기 전에 정보를 손실적으로 변환하는 구조적 한계를 갖는다.  
저자들은 **Scroll** 이라는 컨텍스트 매니저를 제안하며, 각 에이전트 세션을 실행 가능한 **Session Environment** 로 취급한다.  
Scroll은 추가 전용(append-only) **Event Log**, 내구성 있는 스토리지($P_t$), 세션 간 지속되는 Python 커널($V_t$) 세 요소로 구성된다.  
모델이 작성한 코드(`exec`)가 세션 상태를 검색·변환하고, `print`로 출력된 내용만 다음 모델 호출의 작업 뷰(working view)에 노출된다.  
작업 뷰가 예산에 근접하면 오래된 스팬을 제거(evict)하되, 원본은 Event Log에 그대로 보존되고 퇴거 인덱스(eviction index)로 탐색 가능하다.  
Qwen3.8-Max 백본 기준으로 LongMemEval $S$ 94.8%, BEAM ${10M}$ 73.1%(기존 최고 대비 +5.1p), LOCA ${256K}$ 86.7%(기존 최고 대비 +37.4p)를 달성했다.  
컨텍스트 관리를 프로그래밍 태스크로 전환함으로써 LLM의 코딩 능력 향상에 자동으로 편승하는 구조를 갖는다.  
저자들은 향후 프론티어 모델 트레이스를 활용한 SFT 또는 정책 증류(policy distillation)를 계획한다.

---

### 1-1. 연구의 목적과 필요성

**목적**: 장기 에이전트 세션에서 컨텍스트 윈도우 한계를 손실 없이 극복하는 컨텍스트 관리 메커니즘 설계.

**필요성** (p.1–2):
- LLM의 유효 컨텍스트는 명목상 윈도우보다 훨씬 작으며, 입력 길이 증가에 따라 검색·추론 성능이 저하됨 [Modarressi et al., 2025; Zeng et al., 2026].
- 기존 **압축(compression)** 방식(요약·트런케이션)과 **외부 메모리** 방식은 모두 미래 필요를 알기 전에 정보를 변환하므로 구조적으로 손실적(lossy)임.
- 장기 작업(예: 레포지토리 수준 소프트웨어 엔지니어링, 딥 리서치)은 과거 정확한 증거나 시간적으로 분산된 사실들에 대한 비자명한 계산을 요구함.

> 💡 **용어 설명 - Long-horizon agent**: 단일 대화가 아니라 수십~수백 번의 도구 호출, 관찰, 수정을 포함하는 장기 연속 작업을 수행하는 LLM 기반 에이전트.

> 💡 **용어 설명 - Context window**: LLM이 한 번의 추론(inference)에서 처리할 수 있는 최대 토큰 수. 이 한계를 넘는 히스토리는 직접 참조할 수 없음.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|-------|
| 1 | 기존 압축·메모리 방식은 구조적으로 손실적 | 압축 시 미래 필요 정보가 무엇인지 알 수 없음; 두 방식 모두 요약이 원본을 대체 | p.2, §1 |
| 2 | 컨텍스트 관리를 프로그래밍 태스크로 전환 가능 | 모델의 코딩 능력 활용; `exec`/`print` 분리로 선택적 노출 | p.2–4, §2 |
| 3 | 손실 없는 히스토리 보존 + 퇴거 후 복구 가능 | append-only Event Log + 퇴거 인덱스 알고리즘(Algorithm 1) | p.5–6, §2.4 |
| 4 | BEAM $_{10M}$에서 기존 최고 대비 +5.1p | Scroll 73.1 vs. Exabase M-1 68.0 (단, 이종 백본 비교) | p.8, Table 2 |
| 5 | LOCA $_{256K}$에서 기존 최고 대비 +37.4p | Scroll 86.7 vs. MiniMax M3+ReAct 49.3 (이종 백본 비교) | p.16, Table 7 |
| 6 | 강한 모델일수록 Scroll을 더 효과적으로 활용 | 6개 백본 비교; LOCA 256K 격차 64p | p.9, Table 4 |
| 7 | 원본 레코드 폐기가 가장 치명적 | lossy 변형 BEAM $_{10M}$ 전체 점수 19.9 | p.9, Figure 3 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제

에이전트 세션이 성장함에 따라 $|S_t| \gg C$가 되어(여기서 $C$는 명목상 컨텍스트 윈도우 크기) 모든 히스토리를 프롬프트에 직렬화할 수 없는 문제. 기존 접근법은 정보 선택을 **수집 시점(ingestion time)**에 고정하므로, 이후 필요한 세부 정보가 무엇인지 알 수 없는 상황에서 불가역적 손실이 발생한다.

**기존 방법의 형식적 한계** (p.3, §2.1):
- **압축**: $c_{t+1} = \phi(c_t, e_t)$ — 손실 연산자 $\phi$가 각 스팬 압축 시 적용. 이후 필요한 정보는 복구 불가.
- **외부 메모리**: $V_t = \psi(V_{t-1}, e_t)$ — 추출 연산자 $\psi$가 수집 시 정보를 고정. 검색 인터페이스가 제한적.

> 💡 **용어 설명 - 손실 연산자(lossy operator) $\phi$**: 원본 텍스트를 요약본으로 변환하는 함수. 원본을 대체하므로 폐기된 정보는 복구 불가능.

> 💡 **용어 설명 - 추출 연산자(extraction operator) $\psi$**: 대화에서 사실·에피소드를 뽑아 외부 저장소에 저장하는 함수. 저장 시점에 선택되지 않은 정보는 나중에 접근 불가.

---

### 제안하는 방법 및 수식

#### 세션 상태 정의 (Eq. 1, p.3)

$$S_t = (L_t,\ P_t,\ V_t)$$

- $S_t$: $t$ 스텝 후의 전체 세션 상태
- $L_t$: 이벤트 시퀀스(메타데이터 포함) — Event Log
- $P_t$: 각 이벤트가 참조하는 페이로드(원본 도구 출력 등) — Durable Storage
- $V_t$: 보조 파생 상태(Scroll에서는 Python 커널 네임스페이스) — 기존 시스템에서는 메모리 저장소나 요약 버퍼

> 💡 **용어 설명 - 페이로드(Payload)**: 도구 호출 결과의 원본 내용(예: 142행짜리 항공편 데이터프레임). 메타데이터와 분리하여 대용량은 외부 파일로 저장하고 포인터만 유지.

#### 컨텍스트 관리 문제 (p.3)

$$\text{각 스텝 } t \text{에서 } S_t \mapsto c_{t+1} \text{ 선택, } \text{s.t. } |c_{t+1}| \leq C$$

- $c_t$: 현재 모델이 보는 작업 뷰(working view), 토큰 수 $|c_t| \leq C$ 제약
- $C$: 모델의 명목상 컨텍스트 윈도우 크기

**Scroll의 해결책**: 맵 $S_t \mapsto c_{t+1}$을 프로그램 $\pi_t$로 표현. 모델이 스텝 $t$에서 $\pi_t$를 작성하면 하니스가 $S_t$ 위에서 실행하여 $V_t$를 갱신하고 다음 호출을 위한 유계 관측(bounded observation)을 방출.

> 💡 **용어 설명 - 작업 뷰(Working View)**: 실제 모델 프롬프트에 들어가는 내용. 전체 세션 상태($S_t$)의 일부만 선택적으로 노출된 것으로, 토큰 예산($C$) 이내여야 함.

#### 퇴거 알고리즘 (Algorithm 1, p.5)

| 기호 | 의미 |
|------|------|
| $c$ | 현재 작업 뷰 |
| $\mathcal{L}$ | Event Log |
| $\mathcal{I}$ | 퇴거 인덱스 |
| $\rho C$ | 예산 ($0 < \rho \leq 1$, $C$는 컨텍스트 윈도우 크기) |
| $k$ | 티어 너비(tier width) |
| $R$ | 보호된 스팬 집합(현재 턴, 최근 꼬리, 최신 도구 결과) |
| $E$ | 퇴거할 스팬 |
| $\mathcal{H}[E]$ | 스팬 $E$의 헤드라인 맵 |

**알고리즘 흐름**:

```
if |c| > ρC:
    ℒ ← PERSIST(c, ℒ)           # 라이브 턴을 영구 저장
    R ← PROTECTED(c)             # 보호 영역 지정
    c ← R ∪ FOLDPAYLOADS(c \ R) # 페이로드를 seq 포인터로 축소
    E ← SELECTSPAN(c \ R, |c| − ρC)  # 퇴거할 스팬 선택
    c, 𝒮 ← EVICTTOINDEX(c, E, ℋ[E]) # 뷰에서 제거, 인덱스에 헤드라인 기록
    𝒮 ← ROLLUP(𝒮, k)            # 인덱스 롤업
return (c, 𝒮)
```

**인덱스 복잡도**: $n$번 퇴거 후 인덱스 크기 $O(k \log_k n)$ — 최근 히스토리는 세밀한 앵커, 오래된 히스토리는 굵은 범위로 표현.

> 💡 **용어 설명 - BM25**: 키워드 기반 정보 검색 알고리즘. TF-IDF의 개선판으로, 임베딩 없이 결정론적으로 동작하며 인덱스 시점에 모델 호출이 불필요.

> 💡 **용어 설명 - Tiered Index(계층형 인덱스)**: 히스토리를 최신(세밀)→오래된(굵은) 순으로 계층적으로 관리하는 인덱스. 각 계층이 $k$개 블록을 초과하면 오래된 블록들이 병합되어 다음 계층으로 이동.

---

### 모델 구조

Scroll은 세 가지 물리적 구성요소와 네 가지 연산 인터페이스로 구성된다.

**물리적 구성요소** (p.4, §2.2):

| 구성요소 | 설명 | 구현 |
|----------|------|-------|
| Event Log ($L_t$) | append-only, 불변 seq 주소, 역할/세션ID/타임스탬프 메타데이터 | SQLite |
| Durable Storage ($P_t$) | 소형 페이로드 SQLite 인라인, 대형은 JSON/아티팩트 파일시스템 + 복구 포인터 | 파일시스템 |
| Python Kernel ($V_t$) | 세션 간 지속, 타입·크기·출처 메타데이터 포함 변수 네임스페이스, 샌드박스 실행 | Python exec |

**모델-facing 인터페이스** (Table 1, p.4):

| 연산 | 인터페이스 | 역할 |
|------|-----------|------|
| LOCATE | `ms.search(query, k, ...)` | BM25 기반 Event Log 검색, seq 주소 반환 |
| MATERIALIZE | `ms.expand(seq)` / `ms.expand(seq_lo, seq_hi)` | 정확한 이벤트/스팬 복구 |
| COMPUTE | 일반 Python + DB/파일시스템/도구 인터페이스 | 필터·조인·집계·파생 상태 구성 |
| EXPOSE | `print(value)` | 선택된 프로젝션만 다음 컨텍스트에 노출 |

> 💡 **용어 설명 - CodeAct 스타일 인터페이스**: LLM이 도구 호출 대신 Python 코드를 직접 작성해 환경과 상호작용하는 패러다임 [Wang et al., 2024]. 코드의 표현력을 그대로 활용할 수 있음.

> 💡 **용어 설명 - 샌드박스(Sandbox)**: 보안을 위해 격리된 실행 환경. Event Log는 커널에서 읽기 전용이며, 네트워크/파일시스템 접근은 하니스가 명시적으로 허용한 범위로 제한.

---

### 성능 향상

**LongMemEval $_S$ ** (Table 2, p.8):
- Scroll: **94.8%** — Mem0(94.4%), Mastra OM(94.9%)과 경쟁적
- ⚠️ **비교 불가 주의**: 각 시스템이 서로 다른 리더 모델(GPT-5, GPT-5 mini, Gemini 3 등) 사용

**BEAM $_{10M}$ ** (Table 2):
- Scroll: **73.1** — Exabase M-1(68.0) 대비 +5.1p
- ⚠️ **비교 불가 주의**: Exabase M-1은 Gemini 3 Flash 사용, Scroll은 Qwen3.8-Max 사용

**LOCA** (Table 3, p.8):
- 128K: Scroll 89.3% = CodeAct Agent 89.3%
- 256K: Scroll **86.7%** vs. CodeAct Agent 85.3%, Summarization Agent 65.3%
- ∆(128K→256K): Scroll **-2.6p** vs. Summarization Agent -21.4p
- ⚠️ **비교 불가 주의**: Table 7의 published systems 비교는 이종 백본

**카테고리별 강점/약점** (Table 6, p.15):
- 강점: Knowledge update(92.5), Contradiction resolution(88.1), Information extraction(75.0)
- 약점: Summarization(70.5 vs. Exabase M-1 91.9), Temporal reasoning(47.5 vs. Exabase M-1 58.8)

---

### 한계

1. **요약/선호 카테고리 열세**: 수집 시점에 다이제스트를 구축하는 파이프라인이 이미 집약된 뷰를 보유하므로 유리. Scroll은 쿼리마다 재구성 필요 (p.16, Appendix A.2).
2. **약한 모델에서 큰 성능 격차**: LOCA 256K에서 Kimi-K2.7(32.0), Qwen3.6-35B-A3B(22.7) — 복잡한 프로그램 합성이 어려운 모델에서 급격히 성능 저하 (p.9, Table 4).
3. **쿼리 공식화 오류**: preference following 실패(D.3)는 검색 메커니즘이 아닌 질의 축 선택 오류에서 기인.
4. **위치적 샘플링 한계**: summarization 실패(D.4)는 긴 세션의 중간 부분 누락.
5. **베이스라인 독립 재현 없음**: "독립 재현이 반복적으로 평가 설정 불일치를 야기"하여 자체 재현 생략 (p.7, footnote 1). ⚠️ **통계적 취약점**.
6. **각 태스크 단일 평가**: "each task is evaluated once" (p.7) — 분산 추정 불가. ⚠️ **통계적 취약점**.
7. **지연 시간·비용 미보고**: "토큰 수만 보고, 지연/비용은 서빙 설정에 의존" (p.10).

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| 기존 압축·메모리의 구조적 손실 문제 | p.2, §1 |
| 세션 상태 공식화 $S_t = (L_t, P_t, V_t)$ | p.3, Eq.(1) |
| 컨텍스트 관리 = 프로그램 $\pi_t$ 작성 | p.3, §2.1 |
| 모델-facing 인터페이스 4가지 | p.4, Table 1 |
| 프로그래밍 컨텍스트 구성 예시 | p.5, Figure 2 |
| 퇴거 알고리즘 | p.5, Algorithm 1 |
| 메모리 시스템 비교 | p.8, Table 2 |
| LOCA 전략 비교 | p.8, Table 3 |
| 백본 일반화 비교 | p.9, Table 4 |
| 에블레이션 결과 | p.9, Figure 3 |
| 비용/효율 | p.10, Figure 4 |
| LongMemEval 세부 분류 | p.15, Table 5 |
| BEAM 카테고리별 비교 | p.15-16, Table 6 |
| LOCA 발표 결과와 비교 | p.16, Table 7 |

---

## 4. 저자 보고 vs. 해석 분리

### 연구 주제
**저자 보고**: "LLM 에이전트가 장기 실행 작업을 위해 컨텍스트를 관리하는 문제를 다루며, 각 에이전트 세션을 실행 가능한 Session Environment로 취급하는 Scroll을 제안한다." (p.1, Abstract)

**해석**: 이는 MemGPT [Packer et al., 2023] 이후 OS-유사 메모리 관리 패러다임을 코드 실행 인터페이스와 결합한 자연스러운 발전으로 볼 수 있다. 핵심 혁신은 메모리 선택을 **수집 시점에서 쿼리 시점으로 지연**하는 것이며, 이는 데이터베이스 시스템의 지연 평가(lazy evaluation) 원리와 유사하다.

### 방법
**저자 보고**: 
- $c_{t+1} = \phi(c_t, e_t)$ (기존 압축), $V_t = \psi(V_{t-1}, e_t)$ (기존 메모리) 대비, Scroll은 $\pi_t(S_t) \to c_{t+1}$ (쿼리 시점 프로그램) (p.3)
- 인덱스 크기: $O(k \log_k n)$ (p.6)

**해석**: BM25 검색 선택은 실용적이나 동의어·의미적 유사성 처리에 약점이 있다. 에블레이션(Figure 3)에서 인덱스 제거 시 preference following이 89.1→74.9로 하락한 것은 BM25의 어휘적 검색 한계를 시사한다. 임베딩 기반 검색과의 비교가 없어 이 선택의 상대적 우열은 불분명하다.

### 결과
**저자 보고**: BEAM $_{10M}$ 73.1, Exabase M-1 68.0, 차이 +5.1p (Table 2, p.8)

**해석**: 이 5.1p 차이는 **이종 백본 비교**이므로(Gemini 3 Flash vs. Qwen3.8-Max) Scroll 아키텍처 자체의 기여분과 백본 성능 차이를 분리할 수 없다. 동일 백본에서의 통제된 비교 없이는 이 수치를 아키텍처 우위의 증거로 해석하기 어렵다. 반면 Table 3(LOCA)의 내부 비교는 동일 백본(Qwen3.8-Max)을 사용하므로 통제된 비교에 해당하며, Scroll의 컨텍스트 관리 효과를 더 신뢰할 수 있게 보여준다.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적 취약점

| 항목 | 문제 |
|------|------|
| 단일 실행 평가 | "each task is evaluated once with a random seed" (p.7) — 신뢰구간, 표준편차 없음 |
| 베이스라인 미재현 | 저자가 직접 베이스라인 재현 없이 공개된 숫자 인용 (p.7, footnote 1) |
| LLM-as-a-judge 사용 | Qwen3.6-flash를 judge로 사용 (p.7) — judge 모델의 편향 미정량화 |
| 에블레이션 단일 조건 | 각 에블레이션이 단일 조건에서 수행, 상호작용 효과(interaction effect) 미분석 |

### 🚫 비교 불가능한 수치 (Table 2, p.8)

| 비교 쌍 | 이유 |
|---------|------|
| Scroll vs. Mem0 (LongMemEval $_S$ 94.8 vs. 94.4) | Mem0: GPT-5, Scroll: Qwen3.8-Max |
| Scroll vs. Hindsight (BEAM $_{10M}$ 73.1 vs. 64.1) | Hindsight: Gemini 3 Pro, Scroll: Qwen3.8-Max |
| Scroll vs. Exabase M-1 (73.1 vs. 68.0) | Exabase M-1: Gemini 3 Flash |
| Scroll vs. Mastra OM (94.8 vs. 94.9) | Mastra OM: GPT-5 mini |
| Table 7의 모든 비교 | 모든 시스템이 서로 다른 백본 사용 |

> 📝 **저자도 인정**: "These are reference points from the literature rather than a controlled comparison: reader models differ across rows and can substantially affect scores" (Table 2 caption, p.8)

---

## 6. 문서가 답하지 않는 질문

1. **BM25 vs. 임베딩 검색**: 의미적 유사성 기반 검색(dense retrieval)과의 성능 비교 없음. 특히 어휘가 다른 표현의 회상에서 BM25의 한계가 예상되나 정량화 없음.

2. **컨텍스트 예산 $\rho$의 민감도**: $\rho$ 값 선택이 성능에 미치는 영향 미분석. 최적 $\rho$가 태스크 유형에 따라 다른지 불명확.

3. **실제 지연 시간 및 비용**: "we report token counts rather than latency or dollar cost" (p.10). 실용적 배포 결정에 필수적인 정보 부재.

4. **티어 너비 $k$의 선택**: 퇴거 인덱스의 $k$ 값이 성능에 미치는 영향 미분석.

5. **다중 에이전트 환경**: 여러 에이전트가 동일 Session Environment를 공유하는 시나리오 미다루어짐.

6. **보안 모델**: 샌드박스 구현의 구체적 보안 경계, 악의적 코드 생성에 대한 방어 메커니즘 미상세화.

7. **ingestion 비용**: "Ingestion involves no additional LLM calls" (p.10)이지만, SQLite 인덱싱·파일 I/O 등 시스템 수준 비용 미보고.

8. **seq 주소 공간 고갈**: 매우 긴 세션에서 단조 증가 seq가 실용적 한계에 부딪히는 시나리오 미언급.

9. **LongMemEval $_M$ 세부 비교**: Table 5에서 Scroll의 $M$ 분할 결과(89.6%)만 있고 다른 시스템의 $M$ 분할 비교 없음.

10. **정책 증류의 구체적 방법**: 결론에서 "frontier-model traces for SFT or policy distillation"을 언급하지만 구체적 계획 없음.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.3): Scroll 전체 구조 개요

**내용**: 좌측은 전통적 컨텍스트 관리(직렬화된 텍스트 관리, 요약/검색으로 손실), 우측은 Scroll(모델이 코드로 작업 뷰를 구성).

**해석**: 핵심 아이디어의 시각화. 왼쪽에서 프롬프트에 직접 serialization되던 히스토리가 오른쪽에서 Session Environment(Event Log + Python Kernel + Durable Storage)로 분리된다. `exec`(1→2→3)이 환경 접근, `print`(4→5)가 뷰 노출을 담당하는 제어 흐름이 명확히 표현되어 있다. 퇴거 인덱스가 계층적(Tier 1, Tier 2)으로 표현된 부분은 $O(k \log_k n)$ 복잡도의 직관적 이해를 돕는다.

> 💡 **용어 설명 - Serialization(직렬화)**: 구조화된 데이터(예: Python 객체)를 텍스트 문자열로 변환하여 프롬프트에 삽입하는 과정. Scroll에서는 이를 최소화하고 객체를 커널 변수로 유지.

---

### Figure 2 (p.5): 프로그래밍 방식의 컨텍스트 구성

**내용**: 여행 계획 태스크에서 세 번의 `exec` 셀이 어떻게 작업 뷰를 구성하는지 단계별 추적.

**해석**: 
- **Cell 1**: `search_flights()`, `search_routes()` 결과를 커널 변수에 바인딩, 3행만 출력 → 142행 전체 데이터가 컨텍스트에 노출되지 않음
- **Cell 2**: `ms.search("prefer OR avoid", k=20)`으로 사용자 선호도 검색 → seq 주소와 함께 미리보기 획득
- **Cell 3**: `ms.expand(87, 912)`로 정확한 선호도 레코드 복구, 필터링 후 최적 옵션만 출력

이 예시는 **점진적 구체화(progressive refinement)** 패턴을 보여준다. 데이터 대부분은 커널에 상주하고 요약된 결과만 모델 뷰에 들어온다.

---

### Figure 3 (p.9): BEAM $_{10M}$ 에블레이션

**내용**: 4가지 시스템(Lossy summarization, Scroll w/o REPL, Scroll w/o index, Scroll 전체)의 10개 카테고리별 judge score.

**해석**:
- **Lossy summarization(전체 0.20)**: Information extraction, Temporal reasoning, Knowledge update에서 근접 0점 — 원본 폐기의 치명적 영향
- **Scroll w/o REPL(전체 0.66)**: Knowledge update(82.5 vs. 92.5), Instruction following(76.3 vs. 97.5)에서 큰 차이 — 커널에서의 필터링·집계 없이는 다중 레코드 합성 불가
- **Scroll w/o index(전체 0.71)**: Preference following(74.9 vs. 89.1) 최대 손실 — 위치 기반 내비게이션이 키워드 회상보다 중요한 카테고리
- **Full Scroll(0.73)**: 위 모두를 갖출 때 최고

각 구성요소의 기여가 태스크 유형에 따라 다르다는 점이 핵심 인사이트다.

> 💡 **용어 설명 - Ablation Study(에블레이션 연구)**: 시스템의 각 구성요소를 하나씩 제거하여 각 요소의 기여도를 측정하는 실험 방법.

---

### Figure 4 (p.10): 태스크별 비용 분포

**내용**: Scroll의 입력 토큰(a), 출력 토큰(b), 상호작용 턴(c) 분포. IQR 박스플롯 + 중앙값/평균 표시.

**해석**:
- **BEAM $_{10M}$ **: 중앙값 입력 105K 토큰 — 10M 토큰 코퍼스의 약 **1%** 만 모델에 노출 (효율적 필터링)
- **출력 < 입력 (1 order of magnitude)**: 모델이 간결한 코드와 결과를 생성함을 의미
- **LOCA 턴 수(중앙값 35.8–39.0)**: 메모리 벤치마크(5.4–9.1)보다 훨씬 많음 — 실제 환경 조작이 더 많은 상호작용 요구
- **높은 분산**: 일부 태스크에서 극단적으로 높은 비용 발생 가능성(박스 상단 이상치)

⚠️ 단일 실행 결과이므로 분산 추정 신뢰도 제한.

---

### Table 6 (p.15): BEAM $_{10M}$ 카테고리별 비교

**내용**: Mem0, Hindsight, Exabase M-1, Scroll의 10개 메모리 능력 카테고리별 judge score.

**해석**:
- **Scroll 최강 카테고리**: Knowledge update(92.5), Contradiction resolution(88.1) — 원본 이벤트 주소 기반 회수가 시간 순서와 출처를 모두 보존하므로 유리
- **Scroll 최약 카테고리**: Temporal reasoning(47.5), Summarization(70.5) — 수집 시점 다이제스트 구축이 유리한 Exabase M-1(58.8, 91.9) 대비 열세
- **패턴 해석**: Scroll은 "정확한 값의 위치 및 순서"가 중요한 태스크에서 우위, "집약적 프로파일 구축"이 필요한 태스크에서 불리
- **Multi-session reasoning**: 모든 시스템에서 최약(9.6–26.1) — 미해결 공통 과제

⚠️ 백본 모델이 상이하여 직접 비교 한계 존재.

---

## 8. 결론 및 후속 연구

### 8-1. 저자 제시 시사점 및 후속 연구

**저자의 시사점** (p.11, §6):
- 컨텍스트 관리를 명시적 모델 정책으로 전환함으로써 LLM의 코딩 능력 발전에 자동으로 편승
- `exec`(검색·계산)과 `print`(노출) 분리가 효율적 컨텍스트 구성의 핵심
- 이 정책은 작은 모델로 증류 가능

**저자의 후속 연구 계획** (p.11, §6):
- 프론티어 모델 트레이스를 이용한 **Supervised Fine-Tuning(SFT)** 또는 **정책 증류(policy distillation)**
- 두 가지 결정을 지도 신호로 활용:
  1. **컨텍스트 검색**: 언제, 어떻게 에이전트 히스토리에 대한 검색 코드를 작성하는가
  2. **컨텍스트 주입**: 어떤 계산 결과를 작업 윈도우에 다시 프린트할 것인가

---

### 추가 후속 연구 방향

1. **하이브리드 검색**: BM25와 임베딩 기반 검색의 앙상블로 어휘적 불일치 문제 해결
2. **예산 적응적 $\rho$ 학습**: 태스크 유형과 세션 복잡도에 따라 $\rho$를 동적으로 조정하는 메타 정책 학습
3. **다중 에이전트 Session Environment**: 여러 에이전트가 공유 Event Log를 통해 협업하는 시나리오 확장
4. **구조화된 요약과 원본 보존의 하이브리드**: Summarization 카테고리 약점 해결을 위해 수집 시점에 경량 다이제스트를 생성하되 원본도 보존
5. **형식 검증(formal verification)**: 모델 생성 코드의 안전성 보장을 위한 정적 분석 통합

---

### 8-2. 모델의 일반화 성능 향상 가능성

**현재 데이터** (Table 4, p.9):

| 백본 | LongMemEval $_S$ | BEAM $_{10M}$ | LOCA 256K |
|------|----------------|-------------|-----------|
| Qwen3.8-Max | 94.8 | 73.1 | 86.7 |
| Deepseek-v4-pro | 93.2 | 70.2 | 58.7 |
| GLM-5.2 | 93.6 | 70.7 | 62.7 |
| Qwen3.6-35B-A3B | 88.8 | 58.1 | 22.7 |

**패턴 분석**:
- **LongMemEval $_S$ **: 35B 오픈 웨이트 모델도 88.8% 달성 — 단순 검색 태스크에서 높은 일반화
- **LOCA 256K**: 35B 모델 22.7% — 복잡한 프로그램 합성 태스크에서 급격한 성능 저하

**일반화 한계의 원인** (p.9):
- "실패는 프로토콜 수준이 아님: 모든 백본이 CodeAct 인터페이스를 준수하지만, 약한 모델은 집계 중심 태스크에서 더 많은 실행 오류를 범하거나 조기 종료"
- → 일반화 한계는 **컨텍스트 관리 메커니즘**이 아닌 **백본의 코딩 능력과 멀티스텝 계획 능력**에서 비롯됨

**일반화 향상 전략**:

1. **정책 증류(Policy Distillation)**: 저자가 계획한 SFT를 통해 소형 모델이 프론티어 모델의 컨텍스트 관리 패턴을 학습
   - 두 가지 지도 신호: (1) 검색 코드 작성 시점, (2) 프린트 결정

2. **코드 생성 품질 향상**: 집계 중심 태스크에서의 실패가 주원인이므로, Scroll 특화 CodeAct 훈련 데이터 구성이 효과적일 것으로 예상

3. **커리큘럼 학습**: 단순 검색 → 다중 레코드 합성 → 복잡한 집계 순으로 난이도를 점진적으로 높이는 훈련

4. **오류 복구 메커니즘**: 실행 오류 발생 시 재시도 전략을 하니스 수준에서 지원하여 약한 모델의 조기 종료 방지

---

### 8-3. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 아래 비교는 논문 내 인용 문헌과 공개 정보를 기반으로 하며, 2026년 발행 문헌은 검증 불가합니다.

| 연구 | 연도 | 핵심 방법 | Scroll과의 관계 |
|------|------|-----------|----------------|
| MemGPT [Packer et al.] | 2023 | OS-유사 페이징, 외부 메모리 계층 | 선행 연구; 요약/검색 파이프라인으로 손실적 |
| CodeAct [Wang et al.] | 2024 | Python 코드를 에이전트 액션으로 | Scroll의 인터페이스 기반; Scroll은 이를 컨텍스트 관리로 확장 |
| ReAct [Yao et al.] | 2023 | 추론+행동 교차 인터리빙 | Scroll의 baseline 중 하나; 순수 텍스트 기반 |
| LongMemEval [Wu et al.] | 2025 | 장기 메모리 평가 벤치마크 | 평가 벤치마크로 사용 |
| BEAM [Tavakoli et al.] | 2025 | 최대 10M 토큰 메모리 평가 | 평가 벤치마크로 사용 |
| MEM1 [Zhou et al.] | 2026 | 메모리와 추론 시너지, RL 학습 | 보완적; Scroll은 하니스 수준, MEM1은 학습 수준 |
| LOCA [Zeng et al.] | 2026 | 장기 에이전트 컨텍스트 증가 평가 | 평가 벤치마크; Scroll이 +37.4p 달성 |
| Memento [Kontonis et al.] | 2026 | LLM이 자체 컨텍스트 관리 학습 | 유사 방향; 학습 기반 vs. 하니스 기반 차이 |

**본 논문이 미치는 영향**:
1. **컨텍스트 관리의 패러다임 전환**: 손실적 압축에서 손실 없는 지연 평가로의 방향 제시
2. **코딩 능력과 메모리 관리의 통합**: LLM의 코딩 능력 향상이 컨텍스트 관리에 자동으로 이익을 주는 선순환 구조 제시
3. **평가 표준화 과제 부각**: 이종 백본 비교의 한계를 명시적으로 인정함으로써 통제된 비교 필요성 강조

**앞으로 연구 시 고려할 점**:

1. **통제된 비교 설계**: 동일 백본에서 아키텍처만 달리하는 비교가 필수
2. **장기 세션에서의 seq 주소 관리**: 매우 긴 세션(억 단위 토큰)에서의 확장성 검증 필요
3. **도메인 특화**: 코딩 에이전트, 의료, 법률 등 도메인별 컨텍스트 관리 전략 최적화
4. **실시간 적응**: 퇴거 정책($\rho$, $k$)의 동적 조정 메커니즘 연구
5. **멀티모달 확장**: 이미지·오디오 페이로드를 포함한 Session Environment 설계
6. **보안 및 프라이버시**: 민감한 정보가 Event Log에 영구 보존될 때의 접근 제어 및 데이터 보존 정책

---

## 참고 자료

**논문 본문에서 인용된 주요 문헌**:
- Lin, Y., Ang, E., Zhu, E., Ding, B., Zhou, J. "Context as an Environment: Programmatic Context Management for Long-Horizon Agents." arXiv:2608.21690v1, 2026.
- Packer, C. et al. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560, 2023.
- Wang, X. et al. "Executable Code Actions Elicit Better LLM Agents." arXiv:2402.01030, 2024.
- Yao, S. et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR, 2023.
- Wu, D. et al. "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory." ICLR, 2025.
- Tavakoli, M. et al. "Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs." arXiv:2510.27246, 2025.
- Zeng, W., Huang, Y., He, J. "Loca-bench: Benchmarking Language Agents under Controllable and Extreme Context Growth." arXiv:2602.07962, 2026.
- Modarressi, A. et al. "NoLiMa: Long-Context Evaluation beyond Literal Matching." ICML, 2025.
- Sumers, T.R. et al. "Cognitive Architectures for Language Agents." TMLR, 2024.
- Mei, L. et al. "A Survey of Context Engineering for Large Language Models." arXiv:2507.13334, 2025.
- Zhou, Z. et al. "MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents." ICLR, 2026.
- Kontonis, V. et al. "Memento: Teaching LLMs to Manage Their Own Context." arXiv:2604.09852, 2026.
- Jimenez, C.E. et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" ICLR, 2024.
- Yang, J. et al. "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering." NeurIPS, 2024.
