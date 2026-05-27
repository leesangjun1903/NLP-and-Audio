# Is Grep All You Need? How Agent Harnesses Reshape Agentic Search

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **에이전트 기반 검색 시스템에서 검색 전략(lexical vs. semantic)의 효과는 에이전트 하네스(harness) 아키텍처 및 툴 호출 방식(tool-calling paradigm)과 분리될 수 없다.**

구체적으로, 단순한 정규식 기반 lexical 검색(grep)이 **인라인(inline) 툴 결과 전달 방식**에서 모든 하네스-모델 조합에 걸쳐 semantic vector 검색보다 일관되게 높은 정확도를 보였습니다. 그러나 하네스 종류가 달라지면 같은 retriever를 사용해도 성능 차이가 크게 발생합니다.

### 주요 기여 3가지

| 기여 항목 | 설명 |
|-----------|------|
| **검색·하네스·전달 방식의 상호작용** | lexical vs. dense retrieval 선택이 에이전트 오케스트레이션 계층, 인라인 vs. 파일 기반 결과 전달과 결합될 때 어떻게 작용하는지 실증 |
| **노이즈 및 스케일 분석** | 비관련 세션이 증가할수록 end-to-end 성능이 어떻게 변화하는지, 두 retriever 전략의 degradation 패턴 비교 |
| **에이전트 스택 이종성** | 동일한 텍스트 코퍼스를 사용하더라도 아키텍처적으로 다른 하네스(custom vs. provider-native CLI) 간에 retrieval 효과가 안정적이지 않음을 직접 비교 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

기존 RAG 연구는 검색 전략을 **에이전트 아키텍처와 분리하여** 평가해 왔습니다. 이 논문은 다음 세 가지 미탐구 문제를 다룹니다:

1. **검색 전략 × 하네스 상호작용**: lexical과 semantic 검색의 효과가 하네스 종류에 따라 어떻게 달라지는가?
2. **툴 결과 전달 방식의 영향**: 툴 출력이 인라인(inline)으로 주입되는지, 파일(file-based)로 전달되는지에 따라 성능이 어떻게 바뀌는가?
3. **노이즈 증가에 대한 강건성**: 비관련 컨텍스트가 증가할수록 두 검색 전략은 어떻게 저하되는가?

### 2.2 제안하는 방법

논문은 새로운 알고리즘을 제안하기보다는 **체계적 실증 연구(empirical study)** 를 수행합니다. 방법론 핵심 구성은 다음과 같습니다.

#### 2.2.1 데이터셋

- **LongMemEval** 벤치마크 [27]의 116문항 서브셋 사용
- 6개 카테고리: KU(knowledge-update), MS(multi-session), SS-A(single-session-assistant), SS-P(single-session-preference), SS-U(single-session-user), TR(temporal-reasoning)
- 각 질문은 oracle 세션(정답 포함)과 다수의 distractor 세션으로 구성

#### 2.2.2 검색 구현

**Lexical Search (Grep)**:

정규식 기반 패턴 매칭으로, 임베딩 모델 없이 로컬 파일에서 직접 수행:

```math
\text{score}_{\text{grep}}(d, q) = \text{count}(\text{regex\_match}(q, d))
```

여기서 $d$는 문서(대화 턴 또는 시간 이벤트), $q$는 쿼리 패턴입니다.

**Semantic (Vector) Search**:

Dense Passage Retrieval 방식으로 ANN(Approximate Nearest Neighbor) 검색:

$$\text{score}_{\text{vec}}(d, q) = \text{sim}(\mathbf{e}_q, \mathbf{e}_d), \quad \mathbf{e}_q, \mathbf{e}_d \in \mathbb{R}^n$$

여기서 $\mathbf{e}_q$와 $\mathbf{e}_d$는 각각 쿼리와 문서의 dense embedding 벡터이며, $\text{sim}$은 코사인 유사도 또는 내적입니다. 검색 후 reranking 단계를 통해 top- $k$ 결과를 반환합니다.

**Hybrid (이론적 기준)**:

Reciprocal Rank Fusion(RRF)을 활용한 결합 [3]:

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

여기서 $k$는 상수(일반적으로 60), $\text{rank}_r(d)$는 retriever $r$에서 문서 $d$의 순위입니다. 단, 본 논문의 실험에서는 hybrid를 직접 실험하지 않고 grep-only와 vector-only를 비교합니다.

#### 2.2.3 Chronos 구조 및 전처리

Chronos [21]는 LangChain 기반 커스텀 하네스로, 구조적 시간 이벤트를 추출하여 대화 턴과 함께 직렬화합니다:

$$\text{Corpus}_q = \text{serialize}(\text{Turns}_q \cup \text{Events}_q^{\text{temporal}})$$

**동적 프롬프팅(Dynamic Prompting)**: 질문 카테고리에 따라 시스템 프롬프트가 조건부로 구성됩니다:

$$\text{Prompt}(q) = \text{SystemPrompt}(\text{category}(q)) \oplus \text{TopK}_{15}^{\text{vector}}(q) \oplus \text{ToolLoop}$$

초기 broad context로 top-15 vector 결과를 제공한 후, 툴 호출 루프가 시작됩니다.

#### 2.2.4 툴 호출 아키텍처

**인라인(Standard/Inline)**: 검색 결과가 대화 컨텍스트에 직접 주입:

$$\text{Context}_{t+1} = \text{Context}_t \oplus \text{ToolResult}_t$$

**파일 기반(Programmatic/File-Based)**: 검색 결과가 디스크에 저장되고 에이전트가 명시적으로 읽어야 함:

$$\text{ToolResult}_t \rightarrow \text{file}_t, \quad \text{agent reads: } \text{file}_t \xrightarrow{\text{cat/grep}} \text{Context}_{t+1}$$

컨텍스트 압력(context pressure)을 수식으로 표현하면:

$$|\text{Context}_t| = |\text{SysPrompt}| + \sum_{i=1}^{t} |\text{ToolResult}_i| + |\text{ConvHistory}_t|$$

인라인 방식에서는 이 값이 컨텍스트 윈도우 한도 $C_{\max}$에 빠르게 도달합니다.

#### 2.2.5 평가

GPT-4o를 보조 평가 모델(grader)로 사용하여 이진 판정:

```math
\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\text{GPT-4o\_grader}(q_i, \hat{a}_i, a_i^*) = 1]
```

여기서 $N=116$, $\hat{a}_i$는 에이전트 답변, $a_i^*$는 정답입니다.

### 2.3 모델 구조

실험에 사용된 모델 및 하네스 구성:

| 하네스 유형 | 하네스 | 평가 모델 |
|------------|--------|----------|
| Custom | Chronos (LangChain 기반) | Claude Opus 4.6, Claude Haiku 4.5, GPT-5.4, Gemini 3.1 Pro, Gemini 3.1 Flash-Lite |
| Provider-Native CLI | Claude Code (Anthropic) | Claude Opus 4.6, Claude Haiku 4.5 |
| Provider-Native CLI | Codex CLI (OpenAI) | GPT-5.4 |
| Provider-Native CLI | Gemini CLI (Google) | Gemini 3.1 Pro, Gemini 3.1 Flash-Lite |

### 2.4 성능 향상 결과

#### Experiment 1: 인라인 vs. 파일 기반, grep vs. vector

**Table 1 요약**:

| Model | Harness | grep (inline) | vector (inline) | grep (prog.) | vector (prog.) |
|-------|---------|--------------|-----------------|--------------|----------------|
| Claude Opus 4.6 | Chronos | **93.1** | 83.6 | 80.2 | 81.9 |
| Claude Opus 4.6 | Claude Code | **76.7** | 75.0 | 68.1 | **79.3** |
| Claude Haiku 4.5 | Chronos | **83.6** | 76.7 | **83.6** | 81.9 |
| Claude Haiku 4.5 | Claude Code | **55.2** | 44.0 | **37.1** | 32.8 |
| GPT-5.4 | Chronos | **89.7** | 81.9 | **87.1** | 75.0 |
| GPT-5.4 | Codex CLI | **93.1** | 75.9 | 55.2 | **67.2** |
| Gemini 3.1 Pro | Chronos | **91.4** | 82.8 | **79.3** | 76.7 |
| Gemini 3.1 Pro | Gemini CLI | **81.9** | 75.0 | 81.0 | **82.8** |
| Gemini 3.1 Flash-Lite | Chronos | **86.2** | 62.9 | **85.3** | 72.4 |
| Gemini 3.1 Flash-Lite | Gemini CLI | **87.1** | 67.2 | 68.1 | **74.1** |

주요 발견:
- **인라인 전달 시**: 모든 하네스-모델 조합에서 grep > vector
- **최대 격차**: Chronos + Gemini 3.1 Flash-Lite에서 86.2% vs. 62.9% (23.3%p 차이)
- **최소 격차**: Claude Code + Claude Opus 4.6에서 76.7% vs. 75.0% (1.7%p 차이)
- **파일 기반 전달 시**: grep vs. vector 순위가 뒤집히는 경우 다수 발생
- **하네스 변경 효과**: 동일 모델(Claude Opus 4.6)에서 Chronos(93.1%) → Claude Code(76.7%)로 변경 시 16.4%p 하락 — retriever 교체 효과와 비슷한 크기

#### Experiment 2: 노이즈 스케일링

세션 수를 s5 → s10 → s20 → s30 → full(39~66 세션)로 증가시키며 평가:

**grep-only (Table 2 요약)**:

| Model | Harness | s5 | s10 | s20 | s30 | full |
|-------|---------|-----|-----|-----|-----|------|
| Claude Opus 4.6 | Chronos | 89.3 | 89.7 | 90.5 | 85.3 | 89.7 |
| Claude Opus 4.6 | Claude Code | 91.4 | 94.0 | **95.7** | 90.5 | 94.0 |
| GPT-5.4 | Chronos | 83.2 | 82.8 | 81.9 | 78.5 | 78.5 |

**vector-only (Table 3 요약)**:

| Model | Harness | s5 | s10 | s20 | s30 | full |
|-------|---------|-----|-----|-----|-----|------|
| Claude Opus 4.6 | Chronos | 94.0 | **94.8** | 92.2 | 84.5 | 92.2 |
| Claude Opus 4.6 | Claude Code | 77.6 | 72.4 | 75.0 | 78.4 | 72.4 |
| GPT-5.4 | Chronos | 88.8 | **94.0** | 86.2 | 82.8 | 82.8 |

**Figure 1**에서 보이듯, 평균적으로 grep(83.6%)이 vector(78.4%)보다 우위이며 두 방법 모두 노이즈 증가에 대해 최소한의 degradation을 보입니다.

### 2.5 한계

논문이 명시한 주요 한계:

1. **도메인 특수성**: 결론이 long-memory conversational QA에 국한됨. 답변이 verbatim span에 기반하는 특성이 grep에 유리하게 작용
2. **일반화 불가**: 과학 논문 합성, 시각 문서, 코드 의미론 등 paraphrase 기반 증거가 많은 도메인에서는 결과가 다를 수 있음
3. **불완전한 데이터**: Codex vector 중간 설정값 누락으로 vendor-complete 비교 불가
4. **인과관계 부재**: 하네스별 성능 차이의 정확한 원인(프롬프트, 샌드박싱, 컨텍스트 포맷 등) 분리 불가

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문은 일반화 성능과 직접적으로 관련된 여러 통찰을 제공합니다.

### 3.1 노이즈 강건성과 일반화

Experiment 2는 **검색 풀에 비관련 세션이 증가할 때** 두 전략의 degradation을 분석합니다. 일반화 관점에서 중요한 발견:

**전통적 가정의 부분적 반박**:

> *"lexical 검색은 소규모 코퍼스에서만 유효하고, 스케일이 커지면 semantic 검색이 필요하다"*

이 가정과 달리, grep은 full haystack(39~66 세션)에서도 평균적으로 우위를 유지합니다:

$$\Delta_{\text{grep-vector}}^{\text{full}} = 83.6\% - 78.4\% = +5.2\%p$$

그러나 **crossover 지점이 하네스와 백본에 따라 다름**을 발견:

- Chronos + Claude Opus 4.6: s5~s20에서 vector 우위 → full에서 grep 추월 (85.3% vs. 84.5%)
- Gemini CLI + Gemini 3.1 Pro: 모든 설정에서 vector 우위 지속 (full: 89.7% vs. 78.5%)

이는 일반화 성능이 단순히 코퍼스 크기의 함수가 아니라 **하네스-retriever-모델의 3원 상호작용**임을 시사합니다.

### 3.2 일반화를 위한 핵심 메커니즘 분석

#### Lexical Search의 일반화 특성

$$P(\text{correct} | \text{grep}) \propto P(\text{literal span in answer} | \text{task}) \times P(\text{precision match} | \text{grep})$$

- **장점**: verbatim span 기반 답변 태스크에서 embedding bottleneck 없이 정확 매칭 → 일반화 안정성 높음
- **단점**: 어휘 불일치(vocabulary mismatch) 시 완전 실패 → paraphrase 중심 태스크에서 일반화 취약

#### Vector Search의 일반화 특성

$$P(\text{correct} | \text{vector}) \propto P(\text{semantic neighbor} = \text{relevant} | \text{embedding space}) \times P(\text{no topical distractor} | \text{corpus})$$

- **장점**: 저세션 설정(s5, s10)에서 semantic neighborhood 탐색으로 높은 초기 커버리지 → 소규모 코퍼스 일반화 유리
- **단점**: 세션 증가 시 topical false positives(의미적으로 유사하지만 무관한 distractor) 증가 → 노이즈 환경 일반화 저하

### 3.3 일반화 성능 향상을 위한 논문의 시사점

#### (1) 카테고리별 일반화 분석 (Table 4)

Chronos + grep-only + full haystack 기준:

| Category | Claude Opus 4.6 | GPT-5.4 | Gemini 3.1 Pro | Gemini 3.1 Flash-Lite |
|----------|----------------|---------|----------------|----------------------|
| KU (지식 업데이트) | 94.4% | 77.8% | 88.8% | 94.3% |
| MS (멀티세션) | 83.9% | 74.2% | 69.3% | 72.6% |
| SS-A (어시스턴트 생성) | 100.0% | 92.3% | 100.0% | 100.0% |
| SS-P (선호도) | 100.0% | 85.7% | 85.7% | 100.0% |
| SS-U (사용자 사실) | 87.5% | 93.8% | 81.3% | 81.3% |
| TR (시간 추론) | 87.1% | 67.7% | 100.0% | 100.0% |

- **MS(멀티세션) 카테고리**가 전반적으로 가장 낮음 → 복수 세션에 걸친 정보 통합이 일반화의 병목
- **SS-A/SS-P**는 거의 모든 모델에서 높음 → 단일 세션 내 literal span 검색에서 일반화 잘 됨
- **TR(시간 추론)**은 모델별 편차가 큼 → Chronos의 구조적 이벤트 추출이 일반화에 기여

#### (2) 동적 프롬프팅의 일반화 효과

Chronos의 **카테고리 조건부 동적 프롬프팅**:

$$\text{SystemPrompt}(q) = f(\text{category}(q))$$

이 메커니즘은 태스크 유형에 따라 에이전트의 검색 전략을 적응적으로 조정하여, 단일 정적 프롬프트 대비 일반화 성능을 향상시킵니다. 특히 temporal reasoning 카테고리에서 Chronos의 구조적 이벤트 추출이 효과적임을 보여줍니다.

#### (3) 파일 기반 전달과 일반화

파일 기반(programmatic) 전달이 일반화에 미치는 영향은 이중적입니다:

- **이론적 이점**: 컨텍스트 압력 감소 → 대규모 코퍼스 일반화 가능성

$$|\text{Context}_t^{\text{prog}}| \ll |\text{Context}_t^{\text{inline}}|$$

- **실제 결과**: Codex + GPT-5.4에서 inline grep 93.1% → programmatic grep 55.2%로 급락 → 에이전트의 파일 읽기 능력이 일반화의 새로운 병목

이는 **일반화 성능이 retriever 품질만이 아니라 에이전트의 툴 사용 역량에 의존함**을 보여줍니다.

### 3.4 일반화 향상을 위한 hybrid 전략

논문이 직접 실험하지 않았지만 시사하는 hybrid 접근법의 이론적 일반화 이점:

$$\text{score}_{\text{hybrid}}(d, q) = \alpha \cdot \text{score}_{\text{grep}}(d, q) + (1-\alpha) \cdot \text{score}_{\text{vec}}(d, q)$$

또는 RRF:

$$\text{score}_{\text{RRF}}(d) = \frac{1}{k + r_{\text{grep}}(d)} + \frac{1}{k + r_{\text{vec}}(d)}$$

- 소규모 코퍼스: vector가 초기 커버리지 제공 ($\alpha$ 낮게 설정)
- 대규모 코퍼스: grep이 precision 안정성 제공 ($\alpha$ 높게 설정)
- 적응형 $\alpha$ 조정으로 다양한 도메인에서의 일반화 향상 가능

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### 4.1.1 벤치마킹 패러다임의 변화

기존: BM25 vs. ANN을 독립 파이프라인에서 비교
→ 이후: **retrieval + harness + delivery path를 하나의 통합 시스템**으로 평가해야 함

$$\text{End-to-End Performance} = f(\text{retriever}, \text{harness}, \text{delivery}, \text{backbone})$$

단순 검색 지표(NDCG, MRR)만 보고하는 벤치마크는 agentic 설정에서의 실제 성능을 과소추정합니다.

#### 4.1.2 컨텍스트 엔지니어링 연구

"Context rot" [12] 현상과 파일 기반 전달의 상호작용은 새로운 연구 영역을 개척합니다:
- Progressive disclosure 전략의 최적화
- 에이전트 주도 post-retrieval filtering
- 컨텍스트 압력과 검색 품질 간의 tradeoff 정량화

#### 4.1.3 하네스 설계 원칙

Provider-native CLI와 custom harness의 성능 차이가 retriever 교체 효과만큼 크다는 발견은, **하네스 설계가 retriever 선택만큼 중요한 연구 주제**임을 확립합니다.

#### 4.1.4 LLM 에이전트 평가 프레임워크

에이전트 논문은 다음을 모두 보고해야 함을 제안:
1. 검색 전략(retrieval mechanics)
2. 결과 전달 경로(delivery path)
3. 하네스 종류(harness type)
4. 툴 호출 방식(tool-calling paradigm)

### 4.2 앞으로 연구 시 고려할 점

#### 4.2.1 도메인 다양화

**현재 한계**: 결론이 long-memory conversational QA에 특화됨

**향후 필요한 연구**:
- 과학 논문 합성(paraphrase 기반 evidence)
- 코드 시맨틱 검색
- 시각 문서 처리
- 다국어 설정

```math
\text{Generalization}(\text{grep, vector}) = g(\text{domain}, \text{evidence\_type}, \text{paraphrase\_density})
```

#### 4.2.2 Hybrid Retrieval 정책 학습

논문이 직접 다루지 않은 hybrid 접근의 실험적 검증:
- 쿼리 유형에 따른 adaptive retriever 선택
- 온라인 강화학습 기반 retrieval policy
- 질문 카테고리별 최적 $\alpha$ 학습

$$\pi^*(\text{retriever} | q, \text{context}) = \arg\max_{\pi} \mathbb{E}[\text{Accuracy}(q | \pi)]$$

#### 4.2.3 툴 사용 역량과 검색의 분리 평가

파일 기반 전달에서의 성능 급락(Codex: 93.1% → 55.2%)은 **retriever 품질과 툴 사용 역량을 혼동**할 위험을 보여줍니다. 향후 연구는:
- Retriever oracle을 사용한 pure tool-use 능력 평가
- Tool-use 능력과 검색 품질의 독립적 ablation

#### 4.2.4 하네스 내부 메커니즘 해석

Provider CLI harness의 내부 구현이 불투명하여 성능 차이의 인과관계 파악이 어렵습니다. 향후 연구:
- 하네스별 프롬프트 템플릿 영향 분석
- stdout 청킹 방식과 컨텍스트 구성의 영향
- 에이전트 검색 종료 조건(stopping criterion)의 차이 분석

#### 4.2.5 더 넓은 벤더 커버리지

현재 Codex vector의 중간 설정값이 누락되어 있어, 완전한 비교를 위해:
- 모든 harness × model × session-limit 조합의 완전한 실험
- 새로운 provider CLI 에이전트 포함 (예: Mistral, Llama 기반 CLI)
- Open-source 하네스(LlamaIndex, AutoGPT 등) 비교

#### 4.2.6 통계적 유의성 검증

현재 실험은 116문항이라는 제한적 샘플로 진행됩니다. 향후:
- 더 큰 평가 세트 (LongMemEval 전체)
- Bootstrap 신뢰구간 보고
- 다수 랜덤 시드에 걸친 반복 실험

$$\text{Confidence Interval}: \bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

현재 $n=116$ 에서 $\sigma \approx 0.4$이면 95% CI는 약 $\pm 7.3\%$ p로 일부 차이가 통계적으로 유의하지 않을 수 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 주요 주제 | 본 논문과의 관계 |
|------|------|-----------|----------------|
| Lewis et al. (RAG) [10] | 2020 | RAG 원형 제안, dense retrieval + 생성 모델 결합 | 본 논문이 RAG의 agentic 확장에서 lexical 검색의 경쟁력을 보임 |
| Karpukhin et al. (DPR) [8] | 2020 | Dense Passage Retrieval, 이중 인코더 학습 | DPR 기반 vector search가 본 논문에서 비교 대상 |
| Khattab & Zaharia (ColBERT) [9] | 2020 | Token-level late interaction, 효율적 dense search | Hybrid의 이론적 기반 제공 |
| Thakur et al. (BEIR) [23] | 2021 | 제로샷 IR 벤치마크, BM25의 경쟁력 입증 | 본 논문의 lexical 우위 결과와 일치 |
| Formal et al. (SPLADE) [4] | 2021 | Learned sparse representation | Lexical-semantic 중간 지점, 본 논문에서 미실험 |
| Yao et al. (ReAct) [29] | 2023 | 추론-행동 인터리빙 패러다임 | Chronos 커스텀 하네스의 기반 패러다임 |
| Jiang et al. (FLARE) [7] | 2023 | Active Retrieval, 필요시 검색 결정 | 본 논문의 반복적 에이전트 검색과 연관 |
| Asai et al. (Self-RAG) [1] | 2024 | 검색 필요성 자체 판단 및 비판적 반성 | 에이전트가 언제 검색할지 결정하는 메커니즘 |
| Packer et al. (MemGPT) [16] | 2023 | LLM을 OS처럼 운용, 파일 기반 메모리 | 본 논문의 programmatic/file-based 전달 방식과 직결 |
| Wu et al. (LongMemEval) [27] | 2025 | 장기 대화 메모리 벤치마크 | 본 논문의 평가 데이터셋 |
| Sen et al. (Chronos) [21] | 2026 | 시간 인식 대화 에이전트, 구조적 이벤트 추출 | 본 논문의 커스텀 하네스 |
| Liu et al. (Lost in Middle) [12] | 2024 | 긴 컨텍스트에서 LLM의 중간 정보 손실 | 본 논문의 "context rot" 현상의 이론적 기반 |
| Wang et al. (RAG Best Practices) [26] | 2024 | RAG 파이프라인 최적 실천 방법 | 본 논문이 static pipeline 평가의 한계를 보완 |

**핵심 차별점**: 기존 연구들은 검색 전략을 **고정 파이프라인**에서 평가하거나 에이전트 아키텍처를 **단일 설정**에서만 연구했습니다. 본 논문은 **검색 전략 × 하네스 × 전달 방식**의 **3원 factorial 설계**로 agentic 검색을 체계적으로 분석한 최초의 실증 연구입니다.

---

## 참고자료

**본 논문 (직접 분석 대상)**:
- Sen, S., Kasturi, A., Lumer, E., Gulati, A., & Subbiah, V. K. (2026). *Is Grep All You Need? How Agent Harnesses Reshape Agentic Search*. arXiv:2605.15184v1 [cs.CL].

**논문 내 인용 참고문헌 (주요)**:
- [1] Asai et al. (2024). Self-RAG. ICLR 2024.
- [3] Cormack et al. (2009). Reciprocal Rank Fusion. SIGIR 2009.
- [4] Formal et al. (2021). SPLADE v2. arXiv:2109.10086.
- [5] Gao et al. (2024). RAG Survey. arXiv:2312.10997.
- [7] Jiang et al. (2023). Active Retrieval Augmented Generation. EMNLP 2023.
- [8] Karpukhin et al. (2020). Dense Passage Retrieval. EMNLP 2020.
- [9] Khattab & Zaharia (2020). ColBERT. SIGIR 2020.
- [10] Lewis et al. (2020). RAG. NeurIPS 2020.
- [12] Liu et al. (2024). Lost in the Middle. TACL 2024.
- [16] Packer et al. (2023). MemGPT. arXiv:2310.08560.
- [21] Sen et al. (2026). Chronos. arXiv:2603.16862.
- [23] Thakur et al. (2021). BEIR. NeurIPS 2021.
- [27] Wu et al. (2025). LongMemEval. ICLR 2025.
- [29] Yao et al. (2023). ReAct. ICLR 2023.

> **정확도 주의**: 이 논문은 arXiv에 2026년 5월로 표기된 최신 preprint입니다. 논문에 명시된 모델명(GPT-5.4, Claude Opus 4.6 등)은 논문 원문에 기재된 그대로 인용했으나, 실제 공개 모델 버전명과 다를 수 있습니다. 제공된 PDF 원문을 직접 확인하시기를 권장합니다.
