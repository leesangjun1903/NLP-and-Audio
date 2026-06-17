# LiveBrowseComp: Are Search Agents Searching, or Just Verifying What They Already Know?

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **LLM 기반 검색 에이전트들이 실제로 "검색"을 하는 것이 아니라, 자신이 이미 알고 있는 것을 "확인"하는 데 웹을 사용한다**는 것입니다. 저자들은 이를 **Intrinsic Knowledge Dependence (IKD, 내재적 지식 의존성)** 라고 명명합니다.

구체적인 증거로:
- 도구 없이도 최대 **44.5%** 의 BrowseComp 문제를 정답으로 맞힘
- 검색 쿼리의 **50% 이상**이 검색된 결과가 아닌 모델 자체 가설에서 생성됨
- 답변 지원 증거를 제거하면 **폐쇄형(closed-book) 기준선보다 성능이 하락**

### 주요 기여

| 기여 | 설명 |
|---|---|
| **IKD 진단** | 3가지 실험으로 검색 에이전트의 내재적 지식 의존 현상 체계적으로 규명 |
| **LiveBrowseComp 벤치마크** | 지식 경계 밖에서 진정한 검색 능력을 평가하는 335개 문제의 동적 벤치마크 제시 |
| **정적 벤치마크의 한계 폭로** | 기존 벤치마크가 검색 능력이 아닌 기억 기반 검증을 평가하고 있음을 실증 |
| **평가 패러다임 전환 촉구** | 동적·시간 민감형 벤치마크의 필요성과 증거 주도형 검색 훈련 신호의 중요성 제시 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제의 구조

기존 정적 검색 벤치마크(BrowseComp, GAIA 등)는 모델이 훈련됨에 따라 해당 문제들의 답이 점점 모델 파라미터에 흡수됩니다. 이로 인해 발생하는 핵심 문제는 다음과 같습니다:

$$\text{BenchmarkScore} = \underbrace{f(\text{Intrinsic Knowledge})}_{\text{기억 기반 추측}} + \underbrace{g(\text{Search Capability})}_{\text{실제 검색 능력}}$$

두 항을 분리하지 못하면, 높은 벤치마크 점수가 실제 검색 능력의 향상인지 단순히 모델의 지식 팽창인지 알 수 없습니다.

#### IKD의 3가지 진단 실험

**Q1: 폐쇄형 커버리지 (Closed-Book Coverage)**

$$\text{IKD}_{\text{coverage}} = \Pr[\text{correct}(q) \mid \text{Tools} = \emptyset]$$

도구 없이도 정답을 맞힌다면, 그 성공은 검색에 기인할 수 없습니다. 실험 결과 pass@4가 최대 44.5%에 달했습니다.

**Q2: 증거 차단 검색 (Evidence-Blocked Search)**

$$\text{IKD}_{\text{collapse}} = \Pr[\text{correct}(q) \mid \text{index} = D_{\text{irrel}} \cup D_{\text{hard-neg}}] < \Pr[\text{correct}(q) \mid \text{Tools} = \emptyset]$$

정답을 지원하는 문서를 검색 인덱스에서 제거했을 때, **모든 모델에서 폐쇄형 기준선보다 성능이 하락**했습니다.

$$\Delta_{\text{avg}} = \overline{\text{pass@4}}_{\text{closed}} - \overline{\text{pass@4}}_{\text{blocked}} = 26.1 - 6.2 = 19.9 \text{ points}$$

**Q3: 궤적 기반 (Trajectory Grounding)**

각 쿼리의 정보가 어디서 처음 등장하는지를 추적합니다:

$$\text{Model-Originated Rate} = \frac{|\{q_i : \text{source}(q_i) = \text{model reasoning}\}|}{|\{q_i\}|}$$

모든 모델에서 이 비율이 **50% 이상**, 후반 검색 단계에서는 **60% 초과**로 증가했습니다. 또한 정답 지원 증거가 검색되더라도 에이전트가 이를 활용하는 비율은 32% 미만이었습니다:

$$\text{Evidence Use Rate} \leq 32.2\% \quad \forall \text{ evaluated models}$$

---

### 2.2 제안하는 방법: LiveBrowseComp 구축 파이프라인

LiveBrowseComp는 IKD를 억제하기 위한 5단계 파이프라인으로 구성됩니다:

#### Stage 1: 시간적 필터링 (Temporal Filtering)

$$t_{\text{event}} > t_{\text{construction}} - 90 \text{ days}$$

90일 이내에 발생한 사실만 유지하여 모델의 훈련 데이터 커버리지 너머에 위치하도록 합니다.

#### Stage 2: 롱테일 필터링 (Long-Tail Filtering)

소스별 모호도 지수(obscurity metric)를 정의합니다. 예를 들어 TMDB의 경우:

$$S_{\text{TMDB}} = w_1 \cdot \mathbf{1}[\text{popularity} \leq 1] + w_2 \cdot \mathbf{1}[\text{vote count} \leq 20] + w_3 \cdot \mathbf{1}[\text{revenue} = 0] + \text{bonus}$$

$$S_{\text{TMDB}} \geq 2.5 \quad \text{(retention threshold)}$$

각 소스별 임계값:

| 소스 | 필터 지표 | 임계값 |
|---|---|---|
| GDELT | LLM 열기 점수, 기사 길이 | 2.0–4.0, >150자 |
| TMDB | 인기도, 투표수, 수익 | ≥ 2.5 |
| RAWG | 평점수, 추가수, Metacritic | ≥ 2.5 |
| CVE | CVSS, 최근성, 익스플로잇 | ≥ 2.0 |
| Sports | 리그 티어, 출석, 다양성 | ≥ 1.5 |
| USGS | 규모, 유의성, 깊이 | ≥ 1.5 |

#### Stage 3: 답변 안정성 필터링 (Answer Stability Filtering)

90일 창 내에서 답이 변경될 수 있는 후보(예: 박스오피스 누적 수익, 실시간 순위)를 제거합니다.

$$\frac{\partial \text{Answer}(q)}{\partial t} = 0 \quad \forall t \in [t_{\text{event}}, t_{\text{construction}}]$$

#### Stage 4: 질문 구성 (Question Construction)

전문 주석자들이 다음 조건을 만족하는 문제를 생성합니다:
1. 다단계·다출처 추론 필요
2. 단일 단답형 정답 (명확성 보장)
3. **최소 하나의 단서가 90일 이내 사실에 고정**

#### Stage 5: 동료 검토 (Peer Review)

3명의 독립 검증자 + 1명의 교차 검토자에 의한 3가지 병렬 검증:
- **(a) 정확성·유일성**: 증거 사슬 추적 + 다중 LLM으로 후보 답변 생성 후 수동 검증
- **(b) 난이도 교정**: 30분 내 해결 불가 요건
- **(c) 시간성 검증**: 90일 이전 대체 증거 부재 확인

---

### 2.3 모델 구조

논문은 새로운 모델 아키텍처를 제안하는 것이 아니라, **평가 프레임워크 및 벤치마크**를 제안합니다. 실험에서 사용된 에이전트 스캐폴드는 다음과 같습니다:

```
통합 검색 에이전트 스캐폴드 (RedSearcher 기반)
├── search(query): serper.dev API, 최대 10개 결과
├── visit(url, goal): Jina를 통한 전체 페이지 검색
└── code_sandbox: 샌드박스 Python 인터프리터

파라미터:
- temperature = 0.7
- top_p = 0.9
- 최대 컨텍스트: 256k 토큰
- 최대 반복: 250 스텝
```

평가 지표는 pass@k와 avg@k를 사용합니다:

$$\text{pass}@k = \Pr[\exists i \in [k] : \text{correct}(r_i)]$$

$$\text{avg}@k = \frac{1}{k}\sum_{i=1}^{k} \mathbf{1}[\text{correct}(r_i)]$$

---

### 2.4 성능 향상 및 한계

#### 성능 데이터

**정적 벤치마크 vs. LiveBrowseComp 비교 (avg@4)**

| 모델 | BrowseComp | HLE | LiveBrowseComp | 하락폭 |
|---|---|---|---|---|
| Seed 2.0 | 77.3 | 54.8 | 41.5 | ↓35.8 |
| GPT 5.4 | 72.1 | 51.9 | **43.2** | ↓28.9 |
| GLM 5.1 | **68.0** | 43.6 | 33.9 | ↓34.1 |
| DeepSeek v3.2 | 51.4 | 37.1 | 37.6 | ↓13.8 |
| MiniMax M2.5 | 60.4 | 27.1 | 28.0 | ↓32.4 |

**주목할 점**: GLM 5.1은 BrowseComp에서 1위였으나 LiveBrowseComp에서 하위권으로 하락. DeepSeek v3.2는 BrowseComp 최하위에서 LiveBrowseComp에서 상위권으로 도약.

**순위 상관관계 분석:**

$$\rho_{\text{Spearman}}(\text{BC}, \text{BC-ZH}) = 0.87 \quad \text{(정적-정적)}$$

$$\rho_{\text{Spearman}}(\text{BC}, \text{LiveBC}) = 0.74 \quad \text{(정적-동적)}$$

$$r_{\text{Pearson}}(\text{BC}, \text{LiveBC}) = 0.53$$

**폐쇄형 정확도 비교:**

$$\text{CB-Accuracy}_{\text{BrowseComp}} \in [20\%, 44.5\%]$$

$$\text{CB-Accuracy}_{\text{LiveBrowseComp}} < 2\% \quad \forall \text{ models}$$

#### 한계

1. **90일 창의 근사성**: 일부 사실은 90일 이내에 발표되었어도 더 일찍 유출되었을 수 있음. 모델마다 훈련 컷오프가 달라 경계가 불명확함
2. **단일 검색 백엔드 의존**: serper.dev만 사용하여 검색 인덱스 커버리지의 영향을 배제할 수 없음
3. **확장성 문제**: 전문 인간 주석 + 다단계 검증으로 인해 고비용·저확장성
4. **범위 제한**: 335문제라는 비교적 소규모 데이터셋
5. **영어 중심**: 다국어 평가는 제한적

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 IKD가 일반화를 저해하는 메커니즘

기존 정적 벤치마크에서 높은 성능을 보인 모델이 LiveBrowseComp에서 급격히 하락하는 현상은 **과적합(memorization)과 진정한 일반화의 괴리**를 드러냅니다.

$$\text{Generalization Gap} = \text{Score}_{\text{Static}} - \text{Score}_{\text{Live}} \propto \text{IKD}_{\text{severity}}$$

IKD 정도가 높은 모델일수록 일반화 격차가 크며, 이는 모델이 **지식 경계 내 문제에만 특화**되어 있음을 의미합니다.

### 3.2 일반화 성능 향상을 위한 시사점

**① 증거 주도형 훈련 신호 (Evidence-Led Training Signals)**

현재 검색 에이전트 훈련은 최종 정답의 정확성만을 보상합니다. 일반화 향상을 위해서는:

$$\mathcal{R}_{\text{evidence}} = \alpha \cdot \mathbf{1}[\text{correct}] + \beta \cdot \frac{|\text{queries sourced from retrieval}|}{|\text{queries total}|} + \gamma \cdot \text{EvidenceUseRate}$$

즉, 검색 결과로부터 유도된 쿼리 비율과 증거 활용률을 보상에 포함해야 합니다.

**② 지식 경계 인식 메커니즘 (Knowledge Boundary Awareness)**

모델이 자신의 지식 경계를 인지하고, 경계 밖 질문에 대해서는 적극적으로 검색을 수행하도록 유도하는 메타인지적 훈련이 필요합니다:

$$P(\text{search required} \mid q) = f(\text{confidence}_{\text{parametric}}, \text{recency}(q), \text{specificity}(q))$$

**③ 롱테일·시간적 사실에 대한 강건성 훈련**

LiveBrowseComp의 결과를 보면, 도메인별 성능 차이가 두드러집니다. 예를 들어 GLM 5.0은 전반 평균 28.5%임에도 Entertainment에서 52.1%를 기록합니다. 이는 **도메인 특화 지식 커버리지**와 **일반적 검색 전략**이 독립적임을 시사합니다.

$$\text{Score}_{\text{domain}_d} \not\approx \text{Score}_{\text{domain}_{d'}} \quad \text{for } d \neq d'$$

따라서 다양한 도메인의 롱테일 사실에 대한 **광범위하고 균형 잡힌 훈련**이 일반화에 중요합니다.

**④ 검색 루프의 재설계 필요성**

현재 에이전트의 검색 루프는 다음과 같이 모델 주도적입니다:

```
현재 (Model-Led):
내부 가설 생성 → 가설 확인 쿼리 → 확인 실패 → 쿼리 재표현 (루프)
```

일반화 향상을 위한 이상적인 루프:

```
목표 (Evidence-Led):
초기 탐색 쿼리 → 검색 결과 분석 → 결과 기반 다음 쿼리 생성 → 증거 통합 → 답변
```

이 패러다임 전환은 검색이 실패했을 때 **fallback 전략**과 **피벗 능력**을 포함해야 합니다.

**⑤ 인터-모델 간 순위 압축(Compression) 현상의 의미**

$$\Delta_{\text{BC}} = 68.0 - 51.4 = 16.6 \text{ points (open-source spread)}$$

$$\Delta_{\text{LiveBC}} = 38.3 - 28.0 = 10.3 \text{ points (open-source spread)}$$

IKD 제거 후 모델 간 성능 격차가 압축되는 것은, **현재의 성능 차이 상당 부분이 지식 폭의 차이**에 기인함을 보여줍니다. 진정한 검색 전략 능력의 차이는 더 좁습니다. 이는 **검색 전략 최적화**가 모델 크기 증가보다 일반화에 더 큰 기여를 할 수 있음을 시사합니다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 검색 에이전트 평가 벤치마크 계보

| 벤치마크 | 연도 | 특징 | LiveBrowseComp와의 차별점 |
|---|---|---|---|
| **TriviaQA** | 2017 | 단일 회전 독해 QA | 정적, IKD 취약 |
| **NaturalQuestions** | 2019 | 실제 Google 검색 기반 QA | 정적, IKD 취약 |
| **HotpotQA** | 2018 | 다단계 추론 QA | 정적, 고정 코퍼스 |
| **WebArena** | 2024 (ICLR) | 실제 웹 환경 행동 평가 | 동적이나 IKD 진단 없음 |
| **GAIA** | 2024 (ICLR) | 일반 AI 도구 사용 평가 | 정적 문제풀, IKD 혼재 |
| **BrowseComp** | 2025 | 지속적 브라우징 필요 1,266문제 | 정적, IKD 심각 (본 논문이 진단) |
| **FreshQA/FreshLLMs** | 2024 (ACL) | 시간 인식 사실형 QA | 단순 사실 확인, 다단계 추론 없음 |
| **LiveBench** | 2024 | 월별 갱신으로 오염 방지 | 검색 에이전트 특화 아님 |
| **LiveCodeBench** | 2025 (ICLR) | 코딩 문제 지속 갱신 | 코딩 특화 |
| **DeepSearchQA** | 2026 | 심층 연구 에이전트 평가 | 정적 문제풀, IKD 미진단 |
| **OnlineMind2Web** | 2025 | 실시간 웹사이트 평가 | 에이전트 단축키 사용 폭로 (유사 문제의식) |
| **LiveBrowseComp** | 2026 | **IKD 억제 + 동적 갱신** | **검색 능력만 순수 측정** |

### 4.2 데이터 오염 및 파라메트릭 지식 관련 연구

| 연구 | 핵심 발견 | LiveBrowseComp와의 관계 |
|---|---|---|
| Sainz et al. (EMNLP 2023) | LLM 벤치마크 데이터 오염 측정 필요성 | IKD는 오염과 다른 새로운 문제 |
| Jacovi et al. (EMNLP 2023) | 테스트 데이터 공개 방지 전략 | 구조적으로 다른 문제 접근 |
| Du et al. (ACL 2024) | 맥락 증거가 있어도 내부 사전 지식 선호 | IKD의 이론적 근거 지지 |
| Ruis et al. (ICLR 2025) | 절차적 지식 합성을 통한 추론 | IKD의 보완적 관점 |
| Deng et al. (NAACL 2024) | 현대 벤치마크의 오염 조사 | LiveBrowseComp로 우회 가능 |

### 4.3 동적 평가로의 패러다임 전환

```
정적 평가 시대                    동적 평가 시대
(2020~2023)                      (2024~2026)
     │                                 │
TriviaQA, NQ, HotpotQA           LiveBench, LiveCodeBench
     │                                 │
BrowseComp, GAIA (2024~25)  →   LiveBrowseComp (2026)
                                  + IKD 진단 프레임워크
```

LiveBrowseComp는 기존 동적 벤치마크가 단순히 "새로운 문제를 추가"하는 방식과 달리, **IKD라는 구체적인 실패 모드를 정의하고 이를 체계적으로 억제**한다는 점에서 차별화됩니다.

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5.1 향후 연구에 미치는 영향

#### ① 벤치마크 설계 패러다임의 근본적 전환

LiveBrowseComp는 검색 에이전트 평가에서 **정적 → 동적, 기억 기반 → 증거 기반**으로의 전환을 촉구합니다. 향후 검색 에이전트 벤치마크는 반드시 다음을 고려해야 합니다:

$$\text{Valid Benchmark} \Leftrightarrow \text{CB-Accuracy} \approx 0 \land \text{Search-Augmented Score} \gg 0$$

#### ② 훈련 목표 재설계

현재 검색 에이전트 훈련에서 사용되는 RLHF/RLAIF 보상 함수는 정답률만을 고려합니다. 향후 연구는:

$$\mathcal{L}_{\text{new}} = \mathcal{L}_{\text{accuracy}} + \lambda_1 \mathcal{L}_{\text{evidence-grounding}} + \lambda_2 \mathcal{L}_{\text{pivot-ability}}$$

형태의 복합 손실 함수를 탐구해야 합니다.

#### ③ RAG 연구에의 함의

IKD 현상은 RAG(Retrieval-Augmented Generation) 연구에도 직접적인 시사점을 제공합니다. Du et al.(2024)의 발견과 결합하면, **검색된 증거를 실제로 활용하는 메커니즘**이 단순히 검색 능력보다 더 중요할 수 있습니다.

#### ④ 에이전트 평가의 공정성 제고

정적 벤치마크에서의 높은 순위($\rho = 0.87$)가 동적 평가에서는 신뢰할 수 없는 예측자($\rho = 0.74$)가 된다는 발견은, **현재 AI 리더보드의 신뢰성**에 근본적인 의문을 제기합니다.

### 5.2 앞으로 연구 시 고려할 점

#### 단기적 고려사항 (1~2년)

1. **IKD 정량화 지표 개발**: 현재 IKD는 현상으로 정의되었으나, 모델별 IKD 강도를 자동으로 측정하는 지표 개발이 필요합니다.

2. **다중 검색 백엔드 평가**: 단일 검색 엔진(serper.dev)에 대한 의존을 줄이고 여러 인덱스에서의 강건성을 평가해야 합니다.

3. **훈련 데이터 시간적 레이블링**: 모델의 훈련 컷오프를 더 정밀하게 추적하여 "90일 창"의 효과를 모델별로 검증해야 합니다.

#### 중기적 고려사항 (2~5년)

4. **LiveBrowseComp 자동 갱신 파이프라인**: 현재 인간 주석에 의존하는 파이프라인을 반자동화하여 지속적인 벤치마크 갱신을 가능케 해야 합니다.

5. **멀티모달 검색 능력 평가**: 텍스트 기반 검색 외에 이미지, 비디오, 코드 등 다양한 모달리티에서의 IKD 현상을 연구해야 합니다.

6. **증거 통합 메커니즘 연구**: 에이전트가 검색된 증거를 실제로 활용하는 비율($\leq 32\%$)을 높이기 위한 어텐션 메커니즘 또는 명시적 증거 추적 아키텍처 연구가 필요합니다.

#### 장기적 고려사항 (5년 이상)

7. **진정한 정보 발견 능력 정의**: IKD를 넘어, 모델이 "무엇을 모르는지 아는" 메타인지적 능력을 평가하는 프레임워크 개발이 필요합니다.

8. **지식 경계의 동적 관리**: 모델이 자신의 지식 경계를 실시간으로 업데이트하고 관리할 수 있는 지속 학습(continual learning) 연구와의 융합이 필요합니다.

9. **인간-AI 협력 검색 패러다임**: 인간 검색자는 IKD의 영향을 받지 않아 BrowseComp와 LiveBrowseComp에서 유사한 성능($30\%$ vs $31\%$)을 보입니다. 인간의 검색 전략을 AI에 이전하는 연구가 중요합니다.

---

## 참고문헌 (본 답변에서 직접 참조한 자료)

1. **Fan, H., Wang, X., Chu, Z., et al.** (2026). *LiveBrowseComp: Are Search Agents Searching, or Just Verifying What They Already Know?* arXiv:2605.28721v1 [cs.AI] — **본 논문 (첨부 PDF)**

2. **Wei, J., Sun, Z., et al.** (2025). *BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents.* arXiv:2504.12516

3. **Du, K., et al.** (2024). *Context versus Prior Knowledge in Language Models.* ACL 2024.

4. **Ruis, L., et al.** (2025). *Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models.* ICLR 2025.

5. **White, C., et al.** (2024). *LiveBench: A Challenging, Contamination-Free LLM Benchmark.* arXiv:2406.19314

6. **Jain, N., et al.** (2025). *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* ICLR 2025.

7. **Vu, T., et al.** (2024). *FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation.* ACL 2024.

8. **Xue, T., et al.** (2025). *An Illusion of Progress? Assessing the Current State of Web Agents.* arXiv:2504.01382

9. **Chen, Z., et al.** (2025). *BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent.* arXiv:2508.06600

10. **Mialon, G., et al.** (2024). *GAIA: A Benchmark for General AI Assistants.* ICLR 2024.

11. **Zhou, S., et al.** (2024). *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR 2024.

12. **Sainz, O., et al.** (2023). *NLP Evaluation in Trouble.* EMNLP 2023.

> **⚠️ 주의**: 이 논문은 arXiv:2605.28721v1로 2026년 5월 27일 제출된 프리프린트입니다. 모델명(GPT-5.4, Gemini 3.1 Pro, Claude Sonnet 4.6 등)과 성능 수치는 논문 원문에 기재된 내용을 그대로 인용한 것이며, 공개 검증이 완료된 동료 심사 논문이 아닙니다. 일부 모델 버전명이 현재 공개된 제품과 다를 수 있음에 유의하시기 바랍니다.
