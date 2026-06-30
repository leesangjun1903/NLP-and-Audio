# Learning to Reason by Analogy via Retrieval-Augmented Reinforcement Fine-Tuning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **기존 RAG(Retrieval-Augmented Generation)의 의미론적 유사성 기반 검색이 복잡한 추론 과제에 부적합**하다는 근본적 문제를 지적한다. 표면적으로 유사한 문제가 완전히 다른 풀이 전략을 요구할 수 있고, 반대로 표면적으로 다른 문제가 동일한 추론 구조를 공유할 수 있다. 이를 해결하기 위해 **유추적 추론(Analogical Reasoning)** 에 기반한 새로운 포스트-트레이닝 프레임워크인 **RA-RFT(Retrieval-Augmented Reinforcement Fine-Tuning)** 를 제안한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **새로운 검색 패러다임** | 의미 유사도가 아닌 **추론 유용성(Reasoning Utility)** 기반 검색 |
| **Gold-Relevance Distillation** | LLM Judge를 활용한 추론 관련성 레이블 자동 생성 |
| **Reasoning-Aware Retriever** | 대조 학습으로 훈련된 추론 구조 인식 검색기 |
| **RA-RFT 프레임워크** | RLVR과 RAG를 통합한 포스트-트레이닝 파이프라인 |
| **실증적 성능 향상** | AIME 2025에서 GRPO 대비 Qwen3-1.7B +7.1점, Qwen3-4B +2.8점 향상 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**문제 1: RLVR의 파라메트릭 지식 병목**

RLVR(Reinforcement Learning from Verifiable Rewards)은 모델의 파라메트릭 지식에만 의존한다. 사전 학습에 충분히 표현되지 않은 추론 전략이 필요한 문제에서는 아무리 샘플링해도 올바른 풀이를 발견하기 어렵고, 결과적으로 **희소한 보상 신호(Sparse Reward)** 와 학습 정체가 발생한다.

**문제 2: 기존 RAG의 추론 과제 부적합성**

표준 RAG는 어휘적·의미론적 유사도로 후보를 순위화하는데, 이는 **추론 유용성(Reasoning Utility)** — 즉, 검색된 콘텐츠가 실제로 목표 문제 풀이에 도움이 되는지 — 와 약한 상관관계를 가진다. 노이즈가 있는 검색은 오히려 성능을 저하시킨다.

### 2.2 제안 방법 (수식 포함)

#### Stage 1: Gold-Relevance Distillation (추론 관련성 레이블 생성)

**추론 관련성(Reasoning Relevance)** 을 다음과 같이 정의한다:

```math
c^*_{\text{reason}} = \arg\max_{c \in \mathcal{C}} \mathbb{P}_{\mathcal{M}}(a = a^* \mid q, c)
```

- $q$: 쿼리 문제
- $c$: 코퍼스의 후보 문제+추론 트레이스
- $a^*$: 정답

이를 직접 계산하는 것은 intractable하므로, **GPT-4o Judge 모델** $\mathcal{M}_{\text{judge}}$를 활용해 근사한다:

각 쿼리-코퍼스 쌍 $(q_i, c)$에 대해 이진 레이블 $y_{i,c} \in \{0, 1\}$을 부여:
- $y_{i,c} = 1$: 두 문제가 표면 내용과 무관하게 **구조적 추론 패턴을 공유**할 때

#### Stage 2: Reasoning-Aware Retriever Training (추론 인식 검색기 훈련)

Dense Retriever $\mathcal{R}_\theta$를 **InfoNCE 대조 학습 목적함수**로 훈련한다:

$$\mathcal{L}_{\text{retrieval}} = -\sum_{i} \sum_{c^+ \in \mathcal{C}^+_i} \log \frac{\exp(\langle \mathbf{e}_{q_i}, \mathbf{e}_{c^+} \rangle / \tau)}{\sum_{c \in \mathcal{C}_i} \exp(\langle \mathbf{e}_{q_i}, \mathbf{e}_c \rangle / \tau)} \tag{2}$$

- $\mathcal{C}^+\_i = \{c \in \mathcal{C} : y_{i,c} = 1\}$: 추론 관련 positive 트레이스 집합
- $\tau$: 온도 하이퍼파라미터 (실험에서 $\tau = 0.05$)
- $\mathbf{e}\_{q_i}$, $\mathbf{e}_{c}$: 각각 쿼리와 코퍼스 항목의 임베딩

#### Stage 3: Reinforcement Fine-Tuning with Retrieved Demonstrations (검색 증강 강화 학습)

각 훈련 문제 $(q, a)$에 대해:
1. $\{c_1, \ldots, c_k\} = \mathcal{R}_\theta(q, \mathcal{C})$ — top- $k$ 추론 트레이스 검색
2. $\{\hat{a}\_1, \ldots, \hat{a}\_G\} \sim \mathcal{M}_\phi(\cdot \mid q, c_1, \ldots, c_k)$ — G개 응답 샘플링
3. $r(\hat{a}_g, a)$ — 정답 정확성 기반 보상 계산
4. **GRPO** 목적함수로 정책 업데이트:

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}_{(q,a) \sim \mathcal{D}} \left[ -\frac{1}{G} \sum_{g=1}^{G} A_g \cdot \log \mathcal{M}_\phi(\hat{a}_g \mid q, \{c_j\}) \right] \tag{3}$$

여기서 정규화된 이점(Normalized Advantage)은:

$$A_g = \frac{r(\hat{a}_g, a) - \bar{r}}{\sigma_r}$$

**이진 결과 기반 보상 함수**는 다음과 같이 정의된다:

$$r(\hat{a}_g, a) = \begin{cases} 1 & \text{if } \texttt{verify}(\texttt{extract}(\hat{a}_g), a) = \text{True} \\ 0 & \text{otherwise} \end{cases} \tag{4}$$

### 2.3 모델 구조

```
[훈련 데이터 D = {(q_i, a_i)}]  [코퍼스 C = {(p_j, t_j)}]
           ↓                              ↓
    [GPT-4o Judge] ──────────────→ [이진 관련성 레이블 y_{i,c}]
           ↓
    [Reason-ModernColBERT 기반 Dense Retriever R_θ]
    (InfoNCE 대조 학습으로 파인튜닝)
           ↓
    [Top-k 추론 트레이스 검색]
           ↓
    [LLM 정책 모델 M_φ (Qwen3-1.7B / 4B)]
    [프롬프트: 검색된 트레이스 + 훈련 쿼리]
           ↓
    [G개 응답 샘플링 → 보상 계산 → GRPO 정책 업데이트]
```

**주요 구성 요소:**

| 구성 요소 | 모델/방법 |
|---|---|
| 추론 트레이스 생성기 | Qwen3-235B-A22B |
| Judge 모델 | GPT-4o |
| 검색기 기반 모델 | Reason-ModernColBERT (late-interaction multi-vector) |
| 정책 모델 | Qwen3-1.7B, Qwen3-4B |
| 정책 최적화 | GRPO (기본), RLOO, DAPO로도 검증 |
| 훈련 데이터 | QuestA 12.5k 문제 |
| 검색 코퍼스 | OpenR1-Math-220K |

### 2.4 성능 향상

**주요 결과 (average@32 accuracy, %):**

| 방법 | AIME24 | AIME25 | HMMT25 | BrUMO25 | Avg.(all) |
|---|---|---|---|---|---|
| **Qwen3-4B** | | | | | |
| Base (Instruct) | 70.5 | 64.3 | 41.3 | 65.5 | 60.4 |
| + GRPO | 74.8 | 66.4 | 46.4 | 69.8 | 64.4 |
| + RA-RFT (Ours) | **75.8** | **69.2** | **47.3** | **75.7** | **67.0** |
| **Qwen3-1.7B** | | | | | |
| Base (Instruct) | 48.1 | 35.9 | 23.4 | 50.9 | 39.6 |
| + GRPO | 50.4 | 41.6 | 26.3 | 54.8 | 43.3 |
| + QuestA | 52.0 | 42.7 | 26.0 | 52.6 | 43.3 |
| + RA-RFT (Ours) | **55.1** | **48.7** | **28.2** | **57.4** | **47.4** |

**알고리즘 직교성 검증 (Qwen3-1.7B):**

| 방법 | Avg. |
|---|---|
| RLOO | 41.8 |
| RA-RLOO | **45.5** (+3.7) |
| DAPO | 42.6 |
| RA-DAPO | **44.4** (+1.8) |

### 2.5 한계

1. **추가 컴퓨팅 비용**: GPT-4o Judge를 활용한 일회성 어노테이션 패스가 필요하여 표준 RLVR 파이프라인보다 초기 레이블링 비용이 발생함
2. **도메인 제한**: 경쟁 수준 수학 추론에서만 검증되었으며, 코드 생성·과학 문제풀이 등 다른 도메인 확장은 미래 과제로 남김
3. **코퍼스 의존성**: 검색 코퍼스의 품질과 커버리지에 성능이 의존함; 코퍼스에 관련 추론 패턴이 없으면 효과가 제한될 수 있음
4. **초기 성능 저하**: 훈련 초기(Step 0)에 RA-RFT는 GRPO보다 낮은 정확도로 시작함 — 모델이 처음에는 낯선 검색 컨텍스트에 의해 방해받기 때문
5. **$k=1$ 검색**: 실험에서 $k=1$ 트레이스만 사용하며, 다중 트레이스 활용 전략은 미탐색

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 추론 구조 전이(Structural Reasoning Transfer)가 일반화의 핵심

RA-RFT의 일반화 성능 향상 메커니즘은 **표면적 특징이 아닌 추론 구조의 전이**에 있다. 논문의 케이스 스터디(Figure 5)는 이를 명확히 보여준다:

- **목표 문제**: 16개 의자 배치 문제 (AIME 2025) — 조합론적 구조
- **검색된 유추 문제**: 볼록 n-각형 색칠 문제 — 표면적으로 완전히 다름
- **공유하는 추론 구조**: "인접 제약 하에서의 유효한 구성 세기"

RA-RFT는 이 구조적 유사성을 학습함으로써, **훈련 시 보지 못한 문제 유형**에도 적용 가능한 추론 전략을 내면화한다.

### 3.2 일반화를 지원하는 설계 요소들

#### (a) 컨텍스트 다양성이 문제별 정확도를 향상

Figure 4의 분석에 따르면, Top-4 검색 트레이스 중 Top-1 컨텍스트가 대부분의 문제에서 GRPO 기준선보다 높은 정확도를 달성한다. 더 중요하게는, **서로 다른 트레이스가 서로 다른 문제에 최적**임을 보여주어, 검색된 컨텍스트들이 상보적(complementary) 해결 전략을 제공함을 시사한다.

#### (b) 검색기 자체의 일반화: Recall@1 향상

| 검색기 | Recall@1 (%) | Avg. 정확도 |
|---|---|---|
| Qwen3-Emb-4B | 2.3 | 38.5 |
| Qwen3-Emb-4B + 파인튜닝 | 14.7 | 40.5 |
| R-ModernColBERT | 7.2 | 40.7 |
| R-ModernColBERT + 파인튜닝 | **43.5** | **47.4** |
| 랜덤 트레이스 | 0.0 | 37.6 |

추론 관련성 감독(Gold Relevance Supervision)으로 파인튜닝한 검색기는 Recall@1이 7.2%에서 43.5%로 급증한다. 이는 검색기 자체가 **추론 패턴 유사성이라는 추상적 개념을 일반화**하여 학습했음을 의미한다.

#### (c) 불완전한 검색에도 강건한 일반화

논문은 검색기가 완벽하게 gold-relevant 트레이스를 복원하지 못하더라도 RA-RFT의 이득이 실질적으로 유지됨을 확인했다. 이는 모델이 **부분적으로 관련된 추론 패턴에서도 유용한 구조를 추출**하는 능력을 학습함을 의미한다 — 즉, 검색 노이즈에 대한 강건성 자체가 일반화 능력이다.

#### (d) 추론 훈련 시 Denser Reward Signal

표준 RLVR에서 어려운 문제들은 **희소한 보상**으로 인해 정책이 학습할 기울기 정보가 부족하다. RA-RFT는 유추적 시연을 통해 어려운 문제에서 모델의 **실질적 성공률을 높여 보상 밀도를 증가**시킨다:

$$\mathbb{P}(\text{correct} \mid q, c_{\text{analogous}}) \gg \mathbb{P}(\text{correct} \mid q)$$

이 denser reward는 더 광범위한 문제 유형에 걸쳐 효과적인 정책 학습을 가능하게 하여 일반화를 개선한다.

#### (e) 다양한 평가 벤치마크에서의 일관된 향상

RA-RFT는 AIME 2024, AIME 2025, HMMT Feb 2025, BrUMO 2025라는 **서로 다른 특성의 4개 벤치마크** 전반에 걸쳐 일관된 향상을 보인다. 특정 벤치마크에 과적합된 것이 아니라 **수학적 추론 일반 능력**이 향상되었음을 시사한다.

#### (f) SFT vs. RL의 일반화 메커니즘 차이

| 방법 | Avg. | 일반화 메커니즘 |
|---|---|---|
| SFT | 39.7 | 고정 타겟 모방 → 검색 컨텍스트 활용 불가 |
| RA-SFT | 40.5 | 미미한 향상 → 수동적 복사로 전략 전이 불가 |
| GRPO | 43.3 | 파라메트릭 지식만 활용 |
| **RA-RFT** | **47.4** | **탐색+선택적 통합으로 능동적 일반화** |

RA-SFT가 SFT 대비 미미한 향상에 그치는 반면, RA-RFT는 큰 향상을 보인다. 이는 **강화 학습의 탐색 과정에서만 모델이 외부 증거를 선택적으로 통합하는 방법을 학습**함을 의미한다. 즉, 일반화는 단순한 컨텍스트 노출이 아니라 **보상 신호에 의한 능동적 전략 추출 학습**에서 비롯된다.

#### (g) 정책 최적화 알고리즘 독립성 → 일반화 가능성

RA-RFT의 이점이 GRPO, RLOO, DAPO 모두에서 일관되게 나타난다는 사실은, **검색 증강의 일반화 효과가 특정 최적화 알고리즘에 의존하지 않음**을 의미한다. 이는 미래의 더 발전된 RL 알고리즘과도 결합 가능하다는 확장성을 시사한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (a) RAG 패러다임의 재정의
이 논문은 RAG의 목적 함수를 **의미 유사도에서 추론 유용성으로 전환**하는 패러다임 전환을 제시한다. 향후 추론 중심 RAG 연구는 검색 품질 지표를 단순 유사도 점수가 아닌 downstream 추론 성능으로 정의할 가능성이 높다.

#### (b) 포스트-트레이닝 방법론의 확장
기존 포스트-트레이닝은 크게 ① 보상 설계, ② 최적화기 설계, ③ 커리큘럼 설계로 구분되었다. RA-RFT는 **④ 외부 지식 활용**이라는 새로운 직교적(orthogonal) 축을 제시한다. 이 축과 기존 세 축의 조합 연구가 활발해질 것으로 예상된다.

#### (c) 검색기-정책 공동 훈련(Joint Training) 연구 촉발
현재 RA-RFT에서는 검색기가 고정(frozen)된 채로 정책만 학습된다. 미래 연구는 **검색기와 정책 모델을 엔드-투-엔드로 공동 훈련**하는 방향으로 발전할 수 있다. 이는 정책이 더 유용한 트레이스를 검색하도록 피드백을 주는 방향으로 발전 가능하다.

#### (d) 도메인 일반화 연구 활성화
논문은 코드 생성, 과학 문제풀이 등 다른 추론 집약적 도메인으로의 확장 가능성을 언급한다. 적절한 추론 트레이스 코퍼스만 구성하면 동일한 프레임워크를 적용할 수 있으므로, 다양한 도메인에서 **추론 코퍼스 구축 방법론** 연구가 증가할 것으로 예상된다.

#### (e) 소규모 모델의 성능 향상 가능성 제시
Qwen3-1.7B와 같은 소규모 모델에서 더 큰 향상폭(AIME25 +7.1점)이 나타난다. 이는 **외부 지식 접근이 소규모 모델의 능력 병목을 해소**하는 데 특히 효과적임을 시사하며, 효율적 AI 연구에 중요한 함의를 갖는다.

### 4.2 향후 연구 시 고려할 점

#### (a) 코퍼스 품질과 편향 관리
- **고려사항**: 검색 코퍼스의 추론 트레이스 품질이 성능을 결정한다. 오류 있는 트레이스나 특정 추론 패턴에 편향된 코퍼스는 모델 편향을 심화시킬 수 있다.
- **연구 방향**: 자동화된 트레이스 품질 검증, 코퍼스 다양성 측정 지표 개발

#### (b) Judge 모델 의존성 탈피
- **고려사항**: GPT-4o를 Judge로 사용하는 것은 비용과 접근성 제한 문제가 있다. 오픈소스 LLM으로 대체 가능한지, 그리고 Judge의 판단 편향이 레이블 품질에 어떻게 영향을 미치는지 연구가 필요하다.
- **연구 방향**: 자가 감독(self-supervised) 방식의 추론 관련성 레이블 생성, 더 경량의 Judge 모델 탐색

#### (c) 다중 검색 트레이스($k > 1$) 활용 전략
- **고려사항**: 현재 $k=1$로 제한되어 있으나, 여러 트레이스를 어떻게 결합하느냐에 따라 성능이 달라질 수 있다.
- **연구 방향**: 트레이스 앙상블 전략, 트레이스 다양성을 극대화하는 다중 검색 방법

#### (d) 검색기와 정책 모델의 공동 적응
- **고려사항**: 정책 모델이 학습되면서 이전에 유용하던 트레이스가 덜 유용해질 수 있다 (distribution shift). 검색기가 정적이면 최적이 아닐 수 있다.
- **연구 방향**: 온라인 검색기 업데이트, 강화 학습 기반 검색기 학습 (검색 자체를 행동으로 취급)

#### (e) 추론 유사성의 정량적 측정
- **고려사항**: 두 문제가 "추론 구조를 공유한다"는 개념이 여전히 다소 직관적이다. 보다 엄밀한 추론 유사성 측정 지표 없이는 Judge 모델의 판단에만 의존해야 한다.
- **연구 방향**: 추론 그래프(Reasoning Graph), 계산 그래프(Computational Graph) 등을 활용한 정형화된 유사성 측정

#### (f) 컨텍스트 길이 제약
- **고려사항**: 검색된 트레이스를 프롬프트에 포함시키면 입력 길이가 증가한다. 현재 최대 32,768 토큰 제한 내에서 $k=1$만 사용 가능했다. 더 긴 추론 트레이스나 $k>1$에서는 컨텍스트 창 관리가 중요해진다.
- **연구 방향**: 트레이스 압축 기법, 선택적 스텝 추출, 긴 컨텍스트 LLM 활용

#### (g) 도메인 전이 시 코퍼스 구축 비용
- **고려사항**: 새로운 도메인에 적용하려면 해당 도메인의 추론 트레이스 코퍼스와 Judge 어노테이션이 필요하다. 이 초기 비용이 새로운 도메인 진입 장벽이 될 수 있다.
- **연구 방향**: 도메인 간 전이 가능한 검색기, 소량의 어노테이션으로 적응하는 few-shot 방식

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 검색 기반 | 학습 방식 | 추론 강화 방식 | RA-RFT와의 차이 |
|---|---|---|---|---|---|
| **RAG (Lewis et al., 2020)** | 지식 집약적 NLP를 위한 RAG | 의미 유사도 | MLE | 외부 지식 검색 | 추론 구조 고려 없음 |
| **REALM (Guu et al., 2020)** | 검색 증강 사전학습 | 의미 유사도 | MLM | 검색 기반 LM 사전학습 | 추론 과제 특화 아님 |
| **DeepSeek-R1 (DeepSeek-AI, 2025)** | RLVR 기반 추론 | 없음 | RLVR | 파라메트릭 지식만 | 외부 검색 없음 |
| **GRPO (Shao et al., 2024)** | 그룹 상대 정책 최적화 | 없음 | RL | 그룹 정규화 이점 | 외부 검색 없음 |
| **QuestA (Li et al., 2026)** | 질문 증강 커리큘럼 | 없음 | RLVR | 부분 솔루션 주입 | 외부 추론 패턴 없음 |
| **OPSD (Zhao et al., 2026)** | 자가 증류 | 없음 | 자가 증류 | 특권 트레이스 | 같은 문제의 트레이스; 검색 없음 |
| **R3-RAG (Li et al., 2025)** | RL 기반 추론+검색 | 추론 인식 | RL | 인터리브드 검색-추론 | 추론 시에만 적용; 정책 훈련 시 미통합 |
| **Retro* (Lan et al., 2025)** | 추론 집약적 문서 검색 최적화 | Rubric 기반 | Distillation | LLM Judge 파인튜닝 | 추론 시간 RAG; 정책 고정 |
| **BRIGHT (Su et al., 2025)** | 추론 집약적 검색 벤치마크 | 추론 유사도 | 평가 | 검색 품질 측정 | 벤치마크; 훈련 방법 아님 |
| **Didolkar et al. (2025)** | 메타인지적 재사용 | 자체 트레이스 | SFT/ICL | 과거 트레이스 재사용 | 외부 코퍼스 없음; RL 루프 없음 |
| **RA-RFT (본 논문, 2026)** | 검색 증강 RL 파인튜닝 | **추론 유용성** | **RLVR** | **유추적 데모 + RL** | **추론 구조 기반 검색 + RL 통합** |

**핵심 차별점 정리:**

RA-RFT는 기존 연구들과 달리 다음 세 요소를 **동시에** 달성한다:
1. **추론 유용성 기반 검색** (단순 유사도 검색 극복)
2. **훈련 시간 통합** (단순 추론 시간 RAG 극복)
3. **검증 가능한 결과 보상으로만 감독** (교사 모델 토큰 수준 증류 극복)

---

## 참고 자료

**주 논문:**
- Zilin Xiao, Qi Ma, Jason Chen, et al. "Learning to Reason by Analogy via Retrieval-Augmented Reinforcement Fine-Tuning." arXiv:2606.13680v1, June 11, 2026.

**논문 내 인용 주요 참고문헌:**
- Lewis, P. et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
- Guu, K. et al. "Retrieval Augmented Language Model Pre-Training." ICML 2020.
- Shao, Z. et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." CoRR, 2024.
- DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, 2025.
- Su, H. et al. "BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval." ICLR 2025.
- Li, J. et al. "QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation." ICLR 2026.
- Zhao, S. et al. "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models." arXiv:2601.18734, 2026.
- Yu, Q. et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." NeurIPS 2025.
- Gentner, D. "Structure-Mapping: A Theoretical Framework for Analogy." Cognitive Science, 7(2), 1983.
- Arabzadeh, N. et al. "Restructuring the Corpus Makes RAG Work for Math." NeurIPS 2025 Workshop.
- Lan, J. et al. "Retro*: Optimizing LLMs for Reasoning-Intensive Document Retrieval." arXiv:2509.24869, 2025.
- Li, Y. et al. "R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning." EMNLP 2025.
- Chaffin, A. "Reason-ModernColBERT." HuggingFace: lightonai/Reason-ModernColBERT, 2025.
- Qwen Team. "Qwen3 Technical Report." arXiv:2505.09388, 2025.
- Brown, T. et al. "Language Models are Few-Shot Learners." NeurIPS 2020.
- Goyal, A. et al. "Retrieval-Augmented Reinforcement Learning." ICML 2022.
- Yang, X. et al. "Learning by Analogy: Enhancing Few-Shot Prompting for Math Word Problem Solving with Computational Graph-Based Retrieval." arXiv:2411.16454, 2024.
- Didolkar, A. et al. "Metacognitive Reuse: Turning Recurring LLM Reasoning into Concise Behaviors." arXiv:2509.13237, 2025.
- Ye, T. et al. "On-Policy Context Distillation for Language Models." arXiv:2602.12275, 2026.
- Hugging Face. "Open R1: A Fully Open Reproduction of DeepSeek-R1." GitHub, 2025.
- Sheng, G. et al. "HybridFlow: A Flexible and Efficient RLHF Framework." EuroSys 2025.
- Zhang, Y. et al. "Qwen3 Embedding: Advancing Text Embedding and Reranking through Foundation Models." arXiv:2506.05176, 2025.
