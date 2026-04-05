# Towards end-to-end automation of AI research

# 1. 핵심 주장과 주요 기여 요약

## 한줄 요약
이 논문은 **아이디어 생성 → 문헌조사 → 코드 작성 → 실험 실행 → 결과 시각화/분석 → 논문 작성 → 자동 리뷰**까지 이어지는 **AI 연구 전주기(end-to-end)**를 하나의 에이전트 파이프라인으로 자동화한 **The AI Scientist**를 제안하고, 그 산출물 중 일부가 실제 상위권 ML 학회 워크숍의 **1차 블라인드 동료심사 기준을 통과**할 수 있음을 보였습니다.

## 핵심 주장
1. **AI 연구의 개별 단계 자동화는 이미 진전되었지만**, 연구 전과정을 하나의 시스템으로 연결해 자율적으로 수행하는 것은 어려웠다.
2. **The AI Scientist**는 이 전과정을 통합하는 최초 수준의 실증적 시스템이다.
3. 논문의 품질을 대규모로 평가하기 위해 만든 **Automated Reviewer**는 공개 리뷰 데이터에서 **인간 리뷰어 수준에 근접한 판정 성능**을 보였다.
4. **더 좋은 기반 모델**과 **더 많은 test-time compute**를 투입할수록 생성 논문의 질이 높아지는 경향을 확인했다.
5. 따라서 향후 foundation model과 에이전트 설계가 발전하면 **AI 기반 연구 자동화의 성능이 지속적으로 향상될 가능성**이 높다.

## 주요 기여
- **기여 1: 연구 전과정 자동화 파이프라인 제안**
  - 아이디어 생성, novelty checking, 실험, 시각화, 논문 집필, 자동 리뷰까지 연결.
- **기여 2: 두 가지 연구 수행 모드 제시**
  - **template-based**: 사람 제공 코드 템플릿 위에서 연구 수행.
  - **template-free**: 초기 코드도 스스로 만들고, **tree search 기반 실험 탐색** 수행.
- **기여 3: Automated Reviewer 개발**
  - 논문 PDF를 입력받아 structured review와 accept/reject 결정을 생성.
- **기여 4: 실제 peer review 환경에서의 검증**
  - ICLR 2025 워크숍에 AI 생성 논문 3편 제출, 그중 1편이 평균 점수상 **워크숍 수락 기준을 상회**.
- **기여 5: 스케일링 관찰**
  - 모델 성능 향상과 compute 증가가 논문 품질 향상과 상관됨을 보고.

---

# 2. 논문 상세 설명: 문제, 방법(수식 포함), 모델 구조, 성능 향상 및 한계

---

## 2-1. 해결하고자 하는 문제

이 논문이 해결하려는 핵심 문제는 다음과 같습니다.

### 문제 정의
기존의 AI 연구 자동화는 대체로 다음 중 일부에 국한되었습니다.
- 가설 생성
- 문헌 요약
- 코드 생성
- 실험 자동화
- 결과 분석

하지만 실제 연구는 다음과 같은 **연쇄적 의사결정 과정**입니다.

$$
\text{Idea} \rightarrow \text{Novelty Check} \rightarrow \text{Experiment} \rightarrow \text{Analysis} \rightarrow \text{Writing} \rightarrow \text{Review}
$$

기존 시스템은 이 전체 파이프라인을 **끊김 없이 폐루프(closed loop)**로 다루지 못했습니다.  
논문은 이를 다음과 같은 최적화 문제로 볼 수 있습니다.

### 추상적 관점의 문제식
연구 산출물(논문) $p$의 품질을 최대화하는 자동화 시스템을 찾는 문제:

$$
p^* = \arg\max_{p \in \mathcal{P}} Q(p)
$$

여기서
- $\mathcal{P}$: 가능한 연구 아이디어, 코드, 실험, 논문 초안의 공간
- $Q(p)$: 논문의 과학적 품질(기여도, 건전성, 표현력, 수락 가능성 등)

실제로는 직접 $Q(p)$를 측정하기 어렵기 때문에, 저자들은 **Automated Reviewer**를 근사 평가기 $R(p)$로 사용합니다.

$$
R(p) \approx Q(p)
$$

즉, 전체 시스템은 사실상

$$
p^* \approx \arg\max_{p \in \mathcal{P}} R(p)
$$

를 수행하는 에이전트형 탐색 시스템으로 해석할 수 있습니다.

---

## 2-2. 제안하는 방법

논문은 **The AI Scientist**와 **The Automated Reviewer**라는 두 핵심 요소를 제안합니다.

---

### A. The AI Scientist: 전체 파이프라인

논문에 따르면 The AI Scientist는 4단계로 구성됩니다.

#### 1) Ideation
- 특정 ML 하위 분야를 입력받아 연구 아이디어 아카이브를 확장
- 각 아이디어는 다음을 포함
  - 제목
  - 아이디어 설명
  - 왜 흥미로운지에 대한 reasoning
  - 실험 계획
- Semantic Scholar API와 웹 검색을 이용해 **기존 연구와 과도하게 유사한 아이디어를 제거**

이를 추상적으로 쓰면, 아이디어 생성은 다음과 같은 반복적 아카이브 확장입니다.

$$
\mathcal{A}_{t+1} = \mathcal{A}_t \cup \{g_\theta(\mathcal{A}_t, c)\}
$$

여기서
- $\mathcal{A}_t$: 시점 $t$의 아이디어 아카이브
- $c$: 연구 맥락(주제, 제약)
- $g_\theta$: LLM 기반 아이디어 생성기

그리고 novelty filter를 통과한 아이디어만 유지합니다.

$$
\mathcal{A}_{t+1}^{\text{valid}} = \{ a \in \mathcal{A}_{t+1} \mid \text{Novelty}(a) > \tau \}
$$

여기서 $\tau$는 암묵적 novelty 기준입니다.  
논문은 구체적 similarity 수식은 본문에 제시하지 않았고, API 검색 기반 필터링이라고 설명합니다.

---

#### 2) Experimentation
두 모드가 있습니다.

##### (i) Template-based
- 인간이 준 시작 코드 템플릿을 사용
- 실험 계획을 선형적으로 수행
- 실행 실패 시 Aider로 자동 디버깅
- 실험 로그를 experimental journal에 기록

##### (ii) Template-free
- 초기 코드도 모델이 스스로 생성
- 실험을 **병렬 tree search**로 확장
- 4개의 연구 단계:
  1. Preliminary investigation
  2. Hyperparameter tuning
  3. Research agenda execution
  4. Ablation studies

논문이 제시한 구조를 수식적으로 표현하면, 각 노드 $n$은 다음 상태를 가집니다.

$$
n = (s, \pi, e, m, v, c, z)
$$

예를 들어
- $s$: experiment script
- $\pi$: high-level plan
- $e$: execution error trace
- $m$: metrics
- $v$: figure/VLM feedback
- $c$: code critique
- $z \in \{\text{buggy}, \text{non-buggy}\}$: node 상태

트리 탐색은 대략적으로 다음과 같이 볼 수 있습니다.

$$
\mathcal{T}_{t+1} = \mathcal{T}_t \cup \text{Expand}(n_t)
$$

선택 정책은 buggy node를 일정 확률로 우선 선택하고, 그렇지 않으면 non-buggy node 중 best-first로 선택합니다.

$$
n_t \sim 
\begin{cases}
\text{BuggySelection}(\mathcal{T}_t), & \text{with prob. } p_b \\
\text{BestFirstSelection}(\mathcal{T}_t), & \text{with prob. } 1-p_b
\end{cases}
$$

이는 **탐색(exploration)**과 **디버깅/개선(exploitation)**을 결합한 에이전트 탐색으로 이해할 수 있습니다.

또한 replication nodes를 두어 서로 다른 random seed로 반복 실행하고, aggregation nodes로 평균과 표준편차를 시각화합니다. 이는 결과 안정성 평가를 위한 장치입니다.

예를 들어 복제 실험 결과 $y_1,\dots,y_K$가 있을 때,

$$
\bar{y} = \frac{1}{K}\sum_{k=1}^{K} y_k
$$

$$
s = \sqrt{\frac{1}{K-1}\sum_{k=1}^{K}(y_k - \bar{y})^2}
$$

를 통해 mean과 s.d.를 제시합니다.

---

#### 3) Write-up
- LaTeX conference template에 맞춰 논문 섹션별 작성
- 결과 그림 포함
- Semantic Scholar API를 이용해 related work와 citation을 반복 보강
- figure-caption alignment를 VLM으로 점검

이 단계는 본질적으로 다음 생성 문제입니다.

$$
\hat{d} = w_\phi(j, f, l)
$$

여기서
- $j$: experimental journal
- $f$: figures
- $l$: literature retrieval 결과
- $w_\phi$: 논문 작성 LLM

즉, 실험 로그와 문헌 탐색 결과를 바탕으로 문서 $\hat{d}$를 생성합니다.

---

#### 4) Review
- 생성 논문을 Automated Reviewer가 평가
- 수치 점수와 accept/reject 산출

최종적으로 시스템 전반은

$$
\text{Paper Score} = R(\hat{d})
$$

로 귀결됩니다.

---

### B. The Automated Reviewer

Automated Reviewer는 NeurIPS 리뷰 가이드라인을 따르는 자동 심사기입니다.
- PDF 입력
- 5개의 독립 리뷰 생성
- 이후 meta-review에서 area chair 역할의 LLM이 종합 판정

이것은 앙상블 리뷰로 해석할 수 있습니다.

리뷰어 $i$의 평가를 $r_i(d)$라 하면,

$$
\{r_1(d), r_2(d), \dots, r_5(d)\}
$$

를 입력으로 메타 판정기 $M$이 최종 결정을 냅니다.

$$
R(d) = M(r_1(d), r_2(d), r_3(d), r_4(d), r_5(d))
$$

이는 단일 샘플의 불안정성을 줄이고 리뷰 변동성을 완화하기 위한 구조입니다.

---

## 2-3. 모델 구조

논문은 단일 end-to-end neural architecture를 제안한 것이 아니라, **여러 foundation model과 툴을 엮은 agentic system**을 제안합니다.

### 전체 구조
The AI Scientist는 크게 다음 모듈들의 조합입니다.

1. **아이디어 생성 모듈**
   - LLM이 연구 방향/가설/실험 계획 생성

2. **문헌 검색 모듈**
   - Semantic Scholar API
   - novelty checking 및 related work 작성

3. **코드 생성/수정 모듈**
   - template-based에서는 Aider 적극 사용
   - template-free에서는 Claude Sonnet 4 중심 코드 생성

4. **실험 실행 및 관리 모듈**
   - Python 실행
   - 오류 로그 수집
   - runtime 제한
   - journal 기록

5. **트리 탐색 관리자**
   - 단계별(stage-wise) 탐색
   - buggy/non-buggy/hyperparameter/ablation/replication/aggregation 노드 관리

6. **VLM 기반 figure critique 모듈**
   - plot 품질, 축/범례/캡션 정합성 점검

7. **논문 작성 모듈**
   - LaTeX 직접 생성 및 수정

8. **Automated Reviewer**
   - structured review + accept/reject

---

### Template-based와 Template-free의 차이

#### Template-based
- 장점:
  - 재현 가능한 시작점
  - 기존 코드베이스 기반이므로 안정적
- 한계:
  - 탐색 범위가 코드 템플릿에 제한됨

#### Template-free
- 장점:
  - 더 넓은 탐색
  - 스스로 코드와 데이터 접근을 구성
  - tree search를 통해 더 개방형 연구 가능
- 한계:
  - 에러 가능성 증가
  - 계산 비용 증가
  - 탐색 품질이 LLM 능력에 크게 의존

---

## 2-4. 성능 향상 결과

---

### (1) Automated Reviewer의 평가 성능

논문 표 1 기준, Automated Reviewer는 human reviewer consistency와 비교 가능한 성능을 보였습니다.

#### 2017–2024 데이터
- Balanced accuracy:
  - Human (NeurIPS consistency): $0.66$
  - Automated Reviewer: $0.69 \pm 0.04$

#### 2025 데이터(knowledge cutoff 이후)
- Balanced accuracy:
  - Automated Reviewer: $0.66 \pm 0.03$

즉, 학습 데이터 오염 가능성을 줄인 이후에도 성능이 유지되어, 저자들은 **실제 인간 집단 판단을 어느 정도 근사**한다고 주장합니다.

---

### (2) 기반 모델 향상에 따른 성능 개선

그림 1b에서, 더 최신의 LLM일수록 AI Scientist가 생성하는 논문의 점수가 높아졌습니다.  
논문은 release date와 paper quality 사이의 회귀 적합을 제시하며,

$$
R^2 = 0.517,\quad P < 0.00001
$$

이라고 보고합니다.

즉, 논문 품질 점수의 변동 중 상당 부분이 **기반 모델 세대 차이**로 설명될 수 있음을 시사합니다.

---

### (3) test-time compute scaling

그림 3c에서 tree search의 experimental node 수를 늘리면 reviewer score가 상승합니다.  
즉,

$$
\frac{\partial \, \text{PaperScore}}{\partial \, \text{Compute}} > 0
$$

에 해당하는 경험적 경향을 보고한 셈입니다.

이는 연구 자동화가 단순히 “좋은 모델 하나”의 문제가 아니라,
- 더 많은 실험 분기
- 더 많은 디버깅
- 더 많은 hyperparameter/ablation
- 더 많은 replicate

를 허용하는 **추론 시 계산(test-time compute)**에 의해서도 향상된다는 의미입니다.

---

### (4) 실제 인간 peer review 결과

- ICLR 2025 워크숍에 AI 생성 논문 3편 제출
- 그중 1편은 reviewer 점수 평균 $6.33$ (각각 6, 7, 6)
- 워크숍 평균 acceptance threshold를 상회

즉, **완전 AI 생성 논문이 실제 블라인드 peer review에서 통과 가능한 수준의 사례**를 실증했습니다.

다만 이는
- 워크숍 수준
- acceptance rate가 상대적으로 높음(본문에 따르면 70%)
- 최종적으로는 사전 프로토콜에 따라 철회

라는 맥락을 반드시 함께 봐야 합니다.

---

## 2-5. 한계

논문이 매우 솔직하게 인정하는 한계가 중요합니다.

### 1) 최상위 학회 메인 트랙 수준은 아직 아님
저자들 스스로도 다음을 명시합니다.
- 3편 중 1편만 워크숍 기준 통과
- 메인 컨퍼런스 수준에는 미달

### 2) 아이디어의 피상성
실패 모드:
- naive하거나 덜 다듬어진 아이디어
- 깊은 방법론적 엄밀성 부족
- conceptual leap 부족 가능성

### 3) 구현 오류
- 핵심 아이디어를 잘못 구현
- 실험 코드 에러
- 디버깅 미완료

### 4) 논문 작성상의 hallucination
- 부정확한 citation
- figure 중복
- 본문/부록 불일치
- 과도한 자신감의 잘못된 서술

### 5) reviewer surrogate의 한계
Automated Reviewer는 유용하지만 여전히

$$
R(d) \neq Q(d)
$$

일 가능성이 큽니다.  
즉, 자동 리뷰 점수가 높다고 해서 과학적 진실성/중요도가 높다는 보장은 없습니다.

### 6) 연구 영역 제한
현재는 본질적으로 **컴퓨터 안에서 자동 실행 가능한 계산 실험** 위주입니다.
예:
- ML benchmark
- 공개 데이터셋
- 코드 기반 실험

실험실 장비, 긴 시간의 인과 검증, 인간 대상 연구 등에는 그대로 일반화하기 어렵습니다.

### 7) 윤리적 문제
- 리뷰 시스템 과부하
- 연구 credential inflation
- 기존 아이디어의 무단 재포장
- 위험한 자동 연구
- 과학 문헌의 노이즈 증가

---

# 3. 특히 강조: 모델의 일반화 성능 향상 가능성

사용자 요청에 따라 이 부분을 중점적으로 다루겠습니다.

---

## 3-1. 이 논문에서 말하는 “일반화”는 두 층위가 있다

이 논문에서 일반화는 단일 supervised model의 test accuracy 의미를 넘어서, 적어도 두 수준에서 이해해야 합니다.

### A. 개별 연구 결과의 일반화
즉, AI Scientist가 만든 모델/방법이
- 다른 데이터셋,
- 다른 시드,
- 다른 설정

에서도 유지되는가?

### B. 연구 자동화 시스템 자체의 일반화
즉, The AI Scientist가
- 새로운 문제,
- 새로운 코드베이스,
- 새로운 데이터셋,
- 새로운 연구 형식

에서도 잘 작동하는가?

논문에서 더 직접적으로 다루는 것은 사실 **B에 더 가깝습니다**.  
그러나 A에 대한 향상 가능성과도 연결됩니다.

---

## 3-2. 일반화 성능 향상 가능성과 관련된 설계 요소

논문은 “우리 시스템이 일반화 성능을 직접 이론적으로 보장한다”고 주장하지는 않습니다.  
대신 다음 메커니즘들이 **일반화에 유리한 연구 습관**을 시스템에 심으려 합니다.

---

### (1) 다단계 실험 구조: feasibility → tuning → main agenda → ablation
이 구조는 단순히 한 번 성능이 잘 나온 실험을 제출하는 것을 막고,
- 먼저 작동 가능성 확인
- 그 후 hyperparameter tuning
- 본 실험
- 마지막에 ablation

으로 이어집니다.

이는 과적합적 “좋아 보이는 결과 한 번”보다, 보다 안정된 실험 설계를 유도합니다.

이를 일반화 관점에서 쓰면, 특정 단일 측정치 $m$을 극대화하는 대신 여러 검증 단계를 통과한 방법 $h$를 찾는 구조입니다.

$$
h^* = \arg\max_{h \in \mathcal{H}} \Big( \text{Viability}(h) + \text{Stability}(h) + \text{AblationSupport}(h) \Big)
$$

물론 이 식은 논문의 명시적 수식은 아니고, 구조를 수학적으로 해석한 것입니다.

---

### (2) 여러 데이터셋 사용
본문에 따르면 stage 2는 **적어도 두 데이터셋에서 성공적 실행**과 training curve 안정화 등을 기준으로 종료될 수 있습니다.

이는 단일 benchmark 과적합을 줄이는 방향입니다.  
일반화 관점에서, 한 데이터셋 성능만 높은 방법보다

$$
J(h) = \frac{1}{D}\sum_{d=1}^{D} \text{Perf}(h; \mathcal{D}_d)
$$

를 높이는 방식이 더 바람직합니다.

여기서 $D \ge 2$는 복수 데이터셋 수입니다.

논문은 이 목적함수를 명시하진 않았지만, 설계 철학은 이와 유사합니다.

---

### (3) Replication nodes
서로 다른 random seed에서 replicate를 실행하여 평균과 표준편차를 계산합니다.

이는 우연한 seed lucky run을 줄이므로, empirical generalization 및 robustness 관점에서 매우 중요합니다.

예를 들어 성능의 기대값과 변동성을 함께 보게 합니다.

$$
\mu(h) = \mathbb{E}_{s \sim \mathcal{S}}[\text{Perf}(h;s)]
$$

$$
\sigma^2(h) = \mathbb{V}_{s \sim \mathcal{S}}[\text{Perf}(h;s)]
$$

좋은 연구는 보통 $\mu(h)$가 높고 $\sigma(h)$가 지나치게 크지 않아야 합니다.  
AI Scientist의 replication/aggregation은 이 점을 일정 부분 반영합니다.

---

### (4) Ablation studies
Ablation은 성능 향상의 원인이 진짜 핵심 모듈 때문인지 검증하는 장치입니다.  
이것은 일반화에 매우 중요합니다. 왜냐하면 우연한 상관관계나 숨은 confounder 때문에 나온 개선을 걸러낼 수 있기 때문입니다.

모형 $h$의 구성요소를 $c_1, \dots, c_k$라 할 때, ablation은 각 구성요소 제거 시 성능 차이를 비교하는 것으로 볼 수 있습니다.

$$
\Delta_i = \text{Perf}(h) - \text{Perf}(h \setminus c_i)
$$

$\Delta_i$가 일관되게 양수라면 해당 구성요소가 실제 기여할 가능성이 높습니다.  
이런 분석은 “우연한 benchmark 특화 개선”을 줄여 일반화 해석을 돕습니다.

---

### (5) VLM 기반 plot critique
이 부분은 간접적이지만 중요합니다. 잘못된 그래프, 엉성한 축, 누락된 범례는 종종 잘못된 해석을 낳습니다.  
시각화 품질 검토는 결과 해석의 오류를 줄이고, 실제 일반화되지 않는 효과를 “있어 보이게” 포장하는 위험을 낮출 수 있습니다.

---

### (6) novelty checking
기존 논문과 지나치게 비슷한 아이디어를 제거합니다.  
이것은 “일반화”와 직접 연결되지는 않지만, 이미 알려진 특정 benchmark trick의 재발견/재포장을 줄이고, 보다 본질적인 연구 방향을 찾도록 유도할 수 있습니다.

---

## 3-3. 그러나 이 논문이 일반화 향상을 직접 입증했다고 보기는 어렵다

이 점은 분명히 해야 합니다.

논문은 **The AI Scientist 자체의 end-to-end 자동화 능력**을 보여주는 데 초점이 있습니다.  
따라서 다음은 **직접적으로 강하게 입증되지 않았습니다**.

1. AI Scientist가 제안한 방법들이 인간 연구자보다 **더 잘 일반화하는 모델**을 만든다.
2. AI Scientist가 만든 아이디어가 **OOD generalization**에서 일관되게 우수하다.
3. AI Scientist의 탐색이 **spurious correlation을 체계적으로 제거한다**.

즉, 일반화 성능 향상은 **가능성**과 **구조적 유리함**은 있으나,  
현 단계에서 논문이 보여준 것은 어디까지나 **연구 프로세스 자동화의 성숙도**에 더 가깝습니다.

---

## 3-4. 일반화 향상 가능성이 큰 이유

그럼에도 앞으로 일반화 성능 향상에 기여할 가능성이 큰 이유는 다음과 같습니다.

### 이유 1: 더 넓은 탐색
인간 연구자는 시간 제약 때문에 적은 수의 가설만 실험합니다.  
AI Scientist는 더 많은 노드를 실험할 수 있으므로, 일반화가 더 잘 되는 해법을 찾을 가능성이 높습니다.

이를 탐색 규모 관점에서 보면,
탐색 후보 수를 $N$이라 할 때 좋은 일반화 특성을 가진 해법이 희소하더라도,

$$
P(\text{find robust solution}) = 1 - (1-p)^N
$$

처럼 후보 수 $N$이 증가할수록 찾을 확률이 증가할 수 있습니다.

---

### 이유 2: 반복 실험 자동화
일반화 검증에는
- 다중 시드
- 다중 데이터셋
- ablation
- 하이퍼파라미터 민감도 분석

이 필요하지만, 인간에게는 매우 노동집약적입니다.  
AI Scientist는 이 비용을 낮추므로, 일반화 친화적 연구 프로토콜을 더 자주 적용할 수 있습니다.

---

### 이유 3: 메타 수준의 자동 개선 가능성
기반 모델이 좋아질수록, 다음과 같은 메타 능력이 좋아집니다.
- 더 좋은 실험 설계
- 더 엄밀한 baseline 비교
- 더 적절한 failure analysis
- 더 나은 오류 수정

이들은 모두 일반화 향상과 연결될 수 있습니다.

---

### 이유 4: shortcut/robustness 연구에 특히 적합
논문 그림 3b 예시 자체가 “shortcut reliance”를 줄이려는 연구 주제를 보여줍니다.  
즉, 이 시스템은 향후
- spurious feature 억제
- OOD robustness
- compositional generalization
- group robustness

같은 문제를 대규모로 자동 탐색하는 엔진으로 쓰일 수 있습니다.

---

## 3-5. 일반화 측면에서 앞으로 반드시 보완해야 할 점

앞으로 진짜로 “일반화 성능을 높이는 AI 연구 자동화 시스템”이 되려면, 다음이 필요합니다.

### 1) 평가함수 자체를 일반화 중심으로 바꿔야 함
현재는 reviewer score가 핵심 surrogate입니다.  
하지만 reviewer score는 실제 일반화와 완전히 일치하지 않습니다.

향후에는 다음과 같은 목적함수가 더 적합합니다.

```math
J_{\text{gen}}(h) = \mathbb{E}_{d \sim \mathcal{D}_{\text{test}}}[\text{Perf}(h;d)]
- \lambda \, \text{Var}_{d \sim \mathcal{D}_{\text{test}}}[\text{Perf}(h;d)]
```

또는 group worst-case 관점:

$$
J_{\text{worst}}(h) = \min_{g \in \mathcal{G}} \text{Perf}(h; g)
$$

이런 지표가 에이전트 탐색에 직접 들어가야, “보기 좋은 논문”이 아니라 “정말 일반화 잘 되는 방법”을 탐색하게 됩니다.

---

### 2) benchmark leakage와 evaluator overfitting 방지
자동화 시스템은 쉽게 알려진 benchmark의 패턴에 과적응할 수 있습니다.  
이는 연구 자동화 시스템 자신도 일반화 실패를 겪을 수 있음을 뜻합니다.

---

### 3) negative result와 uncertainty를 더 잘 다뤄야 함
일반화 연구에서는 “효과 없음”, “조건부로만 효과 있음” 같은 결론이 매우 중요합니다.  
논문에서도 accepted paper가 negative result였다는 점은 의미 있습니다.  
앞으로는 uncertainty-aware writeup이 중요합니다.

예를 들어 개선치 $\Delta$에 대해 단순 평균뿐 아니라 신뢰구간을 명시하고,

$$
\Delta \pm 1.96 \cdot \frac{s}{\sqrt{n}}
$$

같은 형태로 보고하는 습관을 자동 시스템에 내장할 필요가 있습니다.

---

# 4. 2020년 이후 관련 최신 연구와 비교 분석

아래 비교는 **논문 본문 참고문헌과 본문에서 직접 언급한 관련 연구를 중심으로** 정리합니다.  
제가 현재 확실히 말할 수 있는 범위는, 제공된 본문에 포함된 연구들 및 널리 알려진 대표작들입니다.

---

## 4-1. 관련 흐름 개관

2020년 이후 관련 연구는 대략 네 부류로 나뉩니다.

1. **Foundation model/LLM 자체의 발전**
   - GPT-4, Claude 계열, Llama 3 등
2. **LLM 기반 과학 보조**
   - 아이디어 생성, literature survey, hypothesis generation
3. **LLM 기반 ML experimentation**
   - 코드 생성, 에이전트형 ML 실험 수행
4. **자율 과학/자동 연구 시스템**
   - 데이터 분석부터 논문 작성까지 확장

The AI Scientist의 위치는 이 중 **4번의 가장 통합적 형태**입니다.

---

## 4-2. 주요 비교 대상

---

### (A) GPT-4 Technical Report (2023), Claude 3, Llama 3 등
이들은 **기반 모델**이지 연구 자동화 시스템 자체는 아닙니다.  
하지만 The AI Scientist의 핵심 전제는 다음입니다.

$$
\text{System Quality} \uparrow \quad \text{as} \quad \text{Base Model Quality} \uparrow
$$

논문 그림 1b는 바로 이를 실증하려고 합니다.  
즉, The AI Scientist의 참신함은 “새 foundation model”이 아니라, **foundation model 위에 구축된 연구 자동화 orchestration**에 있습니다.

---

### (B) Toolformer (2023)
- LLM이 외부 도구를 쓰게 하는 방향 제시
- The AI Scientist도 Semantic Scholar API, 웹, 코드 실행기, VLM 등을 사용

차이점:
- Toolformer는 일반적인 도구 사용 학습에 가까움
- The AI Scientist는 **과학 연구 워크플로우**라는 고수준 절차를 설계

---

### (C) Eureka (ICLR 2024)
- 코드 생성형 LLM으로 reward function 설계
- 특정 연구 설계 문제를 자동화

차이점:
- Eureka는 **특정 ML 설계 과제 자동화**
- The AI Scientist는 **논문 수준 연구 파이프라인 전체 자동화**

---

### (D) MLAgentBench (ICML 2024)
- 언어 에이전트의 ML 실험 능력 평가 벤치마크

차이점:
- MLAgentBench는 **평가 프레임워크**
- The AI Scientist는 **실제 연구 산출물 생산 시스템**

의의:
- The AI Scientist는 MLAgentBench류 문제설정이 실제 end-to-end 시스템으로 확장된 모습으로 볼 수 있음

---

### (E) AutoSurvey (NeurIPS 2024)
- LLM이 survey 작성 자동화

차이점:
- AutoSurvey는 **문헌 정리/서술 중심**
- The AI Scientist는 **실험 수행과 신생 연구 결과 생성**까지 포함

---

### (F) ResearchAgent (ACL 2025)
- 반복적 literature-based idea generation

차이점:
- ResearchAgent는 아이디어/문헌 탐색 비중이 큼
- The AI Scientist는 여기에 **실험 코드 작성, 실행, ablation, 논문 집필, 자동 리뷰**를 결합

---

### (G) MLE-bench (ICLR 2025)
- ML engineering에서 에이전트 평가

차이점:
- MLE-bench는 실험/엔지니어링 역량 평가
- The AI Scientist는 “엔지니어링 + 과학적 서사 + 리뷰 대응”까지 확장

---

### (H) AIDE: AI-driven exploration in the space of code (2025)
- 코드 공간에서 AI 주도 탐색

차이점:
- AIDE는 코드 탐색/개선 공간에 초점
- The AI Scientist는 **코드 탐색을 연구 방법론 전체의 한 단계로 포함**

---

### (I) Autonomous LLM-driven research—from data to human-verifiable research papers (NEJM AI, 2025)
- 데이터에서 사람 검증 가능한 연구 논문 생성까지 연결하려는 방향

이 연구와 The AI Scientist는 상당히 가까운 문제의식을 가집니다.  
다만 본 논문은 특히
- 실제 top-tier ML workshop peer review 제출
- Automated Reviewer의 정량 검증
- template-free tree search 실험 구조

를 내세운다는 점에서 차별화됩니다.

---

### (J) Darwin Gödel Machine (ICLR 2026)
- 자기개선 에이전트의 open-ended evolution

The AI Scientist와의 연결:
- 둘 다 open-ended, self-improving, agentic exploration 전통에 가깝습니다.
- 하지만 The AI Scientist는 더 직접적으로 **학술 연구 산출물**을 목표로 합니다.

---

## 4-3. 비교 요약 표

| 연구 | 주된 목표 | 실험 수행 | 논문 작성 | 자동 리뷰 | 실제 peer review 검증 |
|---|---|---:|---:|---:|---:|
| Toolformer (2023) | 도구 사용 | 제한적 | 아니오 | 아니오 | 아니오 |
| Eureka (2024) | 특정 설계 자동화 | 예 | 아니오 | 아니오 | 아니오 |
| MLAgentBench (2024) | ML 에이전트 평가 | 예 | 아니오 | 아니오 | 아니오 |
| AutoSurvey (2024) | survey 자동 작성 | 아니오 | 예 | 아니오 | 아니오 |
| ResearchAgent (2025) | 문헌 기반 아이디어 | 제한적 | 일부 | 아니오 | 아니오 |
| AIDE (2025) | 코드 공간 탐색 | 예 | 아니오 | 아니오 | 아니오 |
| NEJM AI 2025 자율 연구 | 데이터→논문 | 예 | 예 | 제한적/불명확 | 제한적 |
| **The AI Scientist (2026)** | **연구 전주기 자동화** | **예** | **예** | **예** | **예** |

이 표는 제공된 본문에 근거해 보수적으로 작성했습니다. 일부 비교 대상의 세부 구현은 원문을 직접 대조하지 못했으므로, **세부 기능 범주는 넓게 해석하지 않고 보수적으로 표기**했습니다.

---

# 5. 앞으로의 연구에 미치는 영향

이 논문은 AI 연구 방법론 자체에 꽤 큰 영향을 줄 가능성이 있습니다.

---

## 5-1. 연구의 단위가 “모델”에서 “연구 에이전트”로 이동
과거에는 좋은 모델 아키텍처를 만드는 것이 핵심이었습니다.  
이제는 다음을 얼마나 잘 조합하느냐가 중요해집니다.

$$
\text{Research Capability} = f(\text{Reasoning}, \text{Tool Use}, \text{Search}, \text{Execution}, \text{Evaluation}, \text{Writing})
$$

즉, 단일 모델 성능보다 **에이전트 시스템 설계**가 연구 생산성을 좌우하게 될 수 있습니다.

---

## 5-2. “아이디어 탐색의 스케일링”이 가능해짐
인간 연구는 아이디어 탐색 폭이 좁습니다.  
이 시스템은 저비용으로 훨씬 많은 가설을 테스트할 수 있게 하므로,
- negative results 축적
- niche benchmark 탐색
- robustness 아이디어 대량 실험
- small improvement들의 메타분석

이 쉬워질 수 있습니다.

특히 일반화 연구처럼 실험 공간이 큰 분야에서 유리합니다.

---

## 5-3. 연구 보조를 넘어 연구 공동수행으로 이동
지금까지 AI는 코딩 보조, 문헌 요약 보조 역할이 강했습니다.  
이 논문은 AI가 **연구 workflow의 공동 주체**가 될 수 있음을 보여줍니다.

---

## 5-4. peer review와 출판 제도의 재설계 압박
이 논문이 던지는 메시지는 기술적 성과를 넘어 제도적 충격도 큽니다.
- AI 논문은 어떻게 disclosure할 것인가?
- AI 생성 결과의 책임은 누가 지는가?
- reviewer overload를 어떻게 방지할 것인가?
- 자동 생성된 저품질 논문 flood를 어떻게 막을 것인가?

향후 학회는 **AI-assisted / AI-generated research policy**를 명시해야 할 가능성이 높습니다.

---

# 6. 앞으로 연구 시 고려할 점

아래는 특히 중요한 후속 연구 과제입니다.

---

## 6-1. 평가함수 설계: “리뷰어 점수”를 넘어서야 함
현재 시스템은 사실상 reviewer-like quality를 최적화합니다.  
하지만 이는 과학적 진실성, 재현성, 장기적 중요성과 다릅니다.

향후에는 다음을 함께 평가해야 합니다.

- 재현 가능성
- 통계적 엄밀성
- OOD/generalization
- causal validity
- novelty의 실질성
- literature faithfulness

예를 들어 최종 보상을 다음처럼 다목적으로 구성할 수 있습니다.

$$
J = \alpha R_{\text{review}} + \beta R_{\text{repro}} + \gamma R_{\text{gen}} + \delta R_{\text{novel}} - \eta R_{\text{risk}}
$$

여기서
- $R_{\text{review}}$: 리뷰 적합도
- $R_{\text{repro}}$: 재현성
- $R_{\text{gen}}$: 일반화 성능
- $R_{\text{novel}}$: 참신성
- $R_{\text{risk}}$: 윤리/안전 리스크

---

## 6-2. 일반화 중심 자동 연구로 발전시켜야 함
앞서 강조했듯, 진짜 중요한 것은 “논문처럼 보이는 결과”가 아니라 **일반화되는 과학적 통찰**입니다.

향후 시스템은 다음을 내재화해야 합니다.
- cross-dataset validation
- worst-group evaluation
- distribution shift 테스트
- seed sensitivity analysis
- confidence interval 보고
- null result reporting

---

## 6-3. hallucination과 citation fidelity 해결
자동 논문 작성에서 가장 위험한 부분 중 하나가 근거 없는 서술입니다.  
특히 관련 연구 인용은 반드시 citation-grounded generation이 필요합니다.

가능한 방향:
- retrieval-grounded writing
- citation span verification
- claim-evidence alignment checker
- theorem/experiment consistency validator

---

## 6-4. 인간-AI 협업 설계가 더 현실적
완전 자동화도 중요하지만, 가까운 미래에는 **human-on-the-loop**가 더 실용적일 수 있습니다.

예:
- 인간이 문제 정의와 위험 관리 담당
- AI가 아이디어 확장/실험 자동화/초안 작성 담당
- 인간이 최종 검증과 책임 담당

이때 생산성은

$$
P_{\text{hybrid}} > P_{\text{human}}, \quad Q_{\text{hybrid}} > Q_{\text{AI-only}}
$$

가 되도록 설계하는 것이 중요합니다.

---

## 6-5. 안전성 및 윤리 가드레일
자동 연구 시스템이 위험한 실험, 생물학적/화학적 오용, 허위 과학, 리뷰 스팸을 일으키지 않도록 다음이 필요합니다.
- 주제 제한
- 위험도 분류
- human approval gate
- submission throttling
- provenance tracking
- AI-generated disclosure

---

## 6-6. 메타과학적 효과 분석
향후 연구는 단지 “AI가 논문을 쓸 수 있는가?”보다,
- 연구 생태계에 어떤 논문이 늘어나는가?
- novelty 분포는 어떻게 바뀌는가?
- false positive science가 증가하는가?
- negative result publication은 늘어나는가?

를 분석해야 합니다.

---

# 7. 최종 정리

## 간결 결론
**“Towards end-to-end automation of AI research”**는 연구 자동화의 핵심 이정표입니다.  
이 논문은 AI가 단순 보조를 넘어, **아이디어 발굴부터 논문 작성과 리뷰까지 이어지는 연구 전체 파이프라인**을 수행할 수 있음을 보여줍니다. 특히 더 나은 foundation model과 더 많은 test-time compute가 품질 향상으로 이어진다는 점은, 이 분야가 앞으로 빠르게 성장할 가능성을 강하게 시사합니다.

## 일반화 성능 관점의 핵심 해석
이 논문이 직접적으로 “더 잘 일반화하는 ML 방법”을 대량 발견했다고 입증한 것은 아닙니다.  
하지만 다음 이유로 **일반화 향상 가능성은 매우 큽니다**.

- 다중 데이터셋 사용
- replicate 및 aggregation
- ablation 자동화
- 넓은 실험 탐색
- robustness/shortcut 문제에 대한 대규모 탐색 잠재력

즉, 이 시스템은 아직 일반화 그 자체를 보장하지는 않지만,  
**일반화를 더 체계적으로 탐구하고 검증하는 연구 엔진**으로 발전할 가능성이 큽니다.

---

# 참고자료 / 출처

아래는 본 답변 작성 시 직접 참고한, 사용자가 제공한 논문 본문 및 그 안에 포함된 주요 참고문헌 제목입니다.

## 핵심 1차 출처
1. **Lu, C., Lu, C., Lange, R. T., Yamada, Y., Hu, S., Foerster, J., Ha, D., Clune, J.**  
   **“Towards end-to-end automation of AI research”**  
   *Nature*, Vol 651, 26 March 2026.  
   DOI: $10.1038/s41586-026-10265-5$

## 논문 본문에서 직접 언급된 관련 참고문헌
2. **OpenAI.**  
   **“GPT-4 Technical Report”** (2023)

3. **Anthropic.**  
   **“The Claude 3 Model Family: Opus, Sonnet, Haiku”** (2024)

4. **Grattafiori et al.**  
   **“The Llama 3 Herd of Models”** (2024)

5. **Jiang et al.**  
   **“AIDE: AI-driven exploration in the space of code”** (2025)

6. **Chan et al.**  
   **“MLE-bench: evaluating machine learning agents on machine learning engineering”** (ICLR 2025)

7. **Baek et al.**  
   **“ResearchAgent: Iterative research idea generation over scientific literature with large language models”** (ACL 2025)

8. **Wang et al.**  
   **“Autosurvey: large language models can automatically write surveys”** (NeurIPS 2024)

9. **Huang, Vora, Liang, Leskovec.**  
   **“MLAgentBench: evaluating language agents on machine learning experimentation”** (ICML 2024)

10. **Ma et al.**  
    **“Eureka: human-level reward design via coding large language models”** (ICLR 2024)

11. **Ifargan et al.**  
    **“Autonomous LLM-driven research—from data to human-verifiable research papers”** (NEJM AI, 2025)

12. **Hu, Lu, Clune.**  
    **“Automated design of agentic systems”** (ICLR 2025)

13. **Lu, Hu, Clune.**  
    **“Automated capability discovery via model self-exploration”** (2025)

14. **Zhang, Hu, Lu, Lange, Clune.**  
    **“Darwin Gödel machine: open-ended evolution of self-improving agents”** (ICLR 2026)

15. **Wang et al.**  
    **“A survey on large language model based autonomous agents”** (2024)

16. **Shinn et al.**  
    **“Reflexion: language agents with verbal reinforcement learning”** (NeurIPS 2024)

17. **Schick et al.**  
    **“Toolformer: language models can teach themselves to use tools”** (NeurIPS 2023)

18. **Bommasani et al.**  
    **“On the opportunities and risks of foundation models”** (2021)

19. **Ecoffet, Clune, Lehman.**  
    **“Open questions in creating safe open-ended AI: tensions between control and creativity”** (2020)

---
