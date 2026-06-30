# Rethinking Cross-Lingual Gaps via Response Variance

---

## 📌 논문 기본 정보

- **제목**: Rethinking Cross-Lingual Gaps via Response Variance
- **저자**: Vihari Piratla, Purvam Jain, Darshan Singh, Trevor Cohn, Preethi Jyothi, Partha Talukdar (Google DeepMind / Google Research)
- **arXiv**: arXiv:2510.15551v2 [cs.CL], 17 Jun 2026

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

기존 연구들은 LLM의 교차 언어 격차(cross-lingual gap)를 **"지식 장벽(knowledge barrier)"** 또는 **"편향된 오류(biased error)"** 에 의한 것으로 주로 귀인하였다. 그러나 이 논문은 **교차 언어 오류의 지배적 원인이 편향이 아닌 목표 언어에서의 높은 응답 분산(response variance)** 임을 최초로 통계적·실증적으로 입증한다.

즉, LLM이 소스 언어에서는 높은 확신(confidence)으로 정확히 답변하지만, 같은 질문을 타겟 언어로 받을 때는 **정답 주변에서 넓게 분산된 응답**을 보이며, 이 분산을 줄이면 교차 언어 격차도 감소한다는 것이다.

### 📊 주요 기여

1. **최초의 통계적 형식화**: 교차 언어 오류를 편향(biased error)과 비편향(unbiased error) 성분으로 분해하는 이론적 프레임워크를 제시
2. **편향 부재 실증**: 타겟 언어 응답 분포의 평균이 소스 응답과 일치함을 실험적으로 검증 ( $\pi \approx 0.9$ ~ $0.95$ )
3. **분산 비례 관계 발견**: 소스 분산이 감소하면 타겟 분산도 감소하고, 교차 언어 격차도 감소함을 이론과 실험으로 검증
4. **추론 시점 개입(inference-time interventions)**: 분산을 줄이는 앙상블 기법들을 제안하여 소스-타겟 전이 점수를 **최대 12절대점** (상대적 8%~50% 이상) 향상

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

LLM은 특정 지식을 소스 언어(예: 힌디어)로는 정확히 답하지만 타겟 언어(예: 히브리어)로는 자주 틀린다. 이 **교차 언어 격차**의 원인으로 기존 연구들은:

- 언어-사일로된 지식 저장(language-siloed knowledge)
- 잘못 정렬된 표현(misaligned representations)
- 토크나이저·임베딩의 한계

를 주로 지적했다. 그러나 이 논문은 이런 "편향(bias)" 기반 설명이 불완전하다고 주장하며, **분산(variance)** 이라는 대안적 원인을 탐구한다.

### 2-2. 이론적 프레임워크 및 수식

#### (a) 잠재 로짓(Latent Logit) 모델

logit 접근이 불가능한 상황(API 전용 모델 등)을 극복하기 위해, 저자들은 **잠재 유효 로짓 벡터**를 가정한다:

$$\alpha \vec{z} \triangleq M(x), \quad \text{s.t.} \; \|\vec{z}\| = 1, \; \alpha > 0$$

$$\hat{y} \sim \text{Categorical}(\text{softmax}(\alpha \vec{z}))$$

여기서 $\vec{z}$는 단위 방향 벡터(응답의 방향성), $\alpha$는 스케일 파라미터(확신도)이다.

#### (b) 소스-타겟 응답 분포 모델

$$\text{Source:} \quad \alpha_s \vec{z}_s = M(x_s), \quad \hat{y}_s \sim \text{Categorical}(\text{softmax}(\alpha_s \vec{z}_s))$$

$$\text{Target:} \quad \kappa \sim \text{Bernoulli}(\pi), \quad \vec{z}_t \mid \kappa = \begin{cases} \alpha_t \vec{z}_s & \kappa = 1 \; \text{(unbiased error)} \\ \alpha_t \vec{z}_b & \kappa = 0 \; \text{(biased error)} \end{cases}$$

$$\text{where} \quad \arg\max \vec{z}_s \neq \arg\max \vec{z}_b, \; \|\vec{z}_b\| = 1$$

$$\hat{y}_t \sim \text{Categorical}(\text{softmax}(\vec{z}_t))$$

- $\pi$: 비편향 성분의 혼합 계수 (핵심 추정 대상)
- $\kappa = 1$ (비편향): 타겟 로짓이 소스 로짓과 같은 방향, 단지 스케일($\alpha_t < \alpha_s$)만 작음 → 분산 증가
- $\kappa = 0$ (편향): 타겟 로짓이 소스와 다른 방향 → 지식 장벽 존재

#### (c) 편향/비편향 오류에 따른 분산 감소 효과 (Propositions)

**Proposition 1** (편향 오류의 경우):
편향이 있을 때($\arg\max \vec{z}_s \neq \arg\max \vec{z}_t$), 응답 분산이 감소할수록(즉 $\alpha \to \infty$) 소스-타겟 응답 일치 확률은 **점근적으로 0에 수렴**한다:

$$\lim_{\alpha \to \infty} \Pr(\hat{y}_s = \hat{y}_t) \to 0$$

**Proposition 2** (비편향 오류의 경우):
교차 언어 오류가 비편향일 때, 응답 분산이 감소할수록 소스-타겟 공유 응답 확률은 **단조 증가**한다. 이는 다음 도함수 계산으로 증명된다:

$$p_k(\kappa) = \frac{\exp(\kappa s_k)}{\sum_i \exp(\kappa s_i)}, \quad q_k(\kappa) = \frac{\exp(\kappa t_k)}{\sum_i \exp(\kappa t_i)}$$

$$\frac{\partial \Pr(\hat{y}_s = \hat{y}_t)}{\partial \kappa} = Z \left( \mathbb{E}_{pq}[S] - \mathbb{E}_p[S] + \mathbb{E}_{pq}[T] - \mathbb{E}_q[T] \right) > 0$$

비편향 경우($q_k \propto p_k^\alpha$)에서 위 도함수가 양수임이 Jensen 부등식 등으로 증명된다.

**Proposition 3** (소스 신뢰도 → 타겟 신뢰도):

$\gamma = \alpha_t / \alpha_s \in (0, 1]$일 때, 타겟 신뢰도(최빈값 확률)의 하한:

$$\Pr(\hat{y}_t = y^{\text{mode}}) \geq \frac{1}{1 + (m-1)^{1-\gamma} \left( \frac{1 - \Pr(\hat{y}_s = y^{\text{mode}})}{\Pr(\hat{y}_s = y^{\text{mode}})} \right)^\gamma}$$

여기서 $m$은 정규화된 응답 공간의 총 개념 수이다. 이 하한은 소스 신뢰도의 **순증가 함수**이므로, 소스에서 자신감이 높을수록 타겟 신뢰도도 높아진다.

증명의 핵심 단계는 Jensen 부등식(오목 함수 $x^\gamma$에 적용):

$$\sum_{i>1} x_i^\gamma \leq (m-1)^{1-\gamma} \left( \sum_{i>1} x_i \right)^\gamma, \quad x_i = \exp(\alpha_s \Delta_i), \; \Delta_i = \vec{z}[i] - \vec{z}[1]$$

#### (d) Chi-squared 거리 (분포 발산 측정)

MMLU(with mixup)에서 소스-타겟 응답 분포 발산 측정에 사용:

$$\text{Chi-squared}(p, q) = \sum_{x:\, p(x)+q(x)>0} \frac{(p(x) - q(x))^2}{p(x) + q(x)}$$

#### (e) Transfer Score 정의

$$A_{q,l} \triangleq \mathbf{1}[\text{question } q \text{ is correct in both source and target language } l]$$

$$\text{transfer-score} := \mathbb{E}_{q,l}[A_{q,l}]$$

#### (f) $\pi$ 추정 (MMLU with mixup)

$$i = \arg\max_k p_k^{(n)}, \quad j = \arg\max_k q_k^{(n)}$$

$$\text{mismatch}^{(n)} = \begin{cases} 0 & \text{if } i = j \\ |p_i^{(n)} - q_i^{(n)}| & \text{otherwise} \end{cases}$$

$$\pi = 1 - \frac{1}{N} \sum_n \text{mismatch}^{(n)}$$

### 2-3. 모델 구조

이 논문은 새로운 모델 아키텍처를 제안하는 것이 아니라, **기존 SoTA LLM들에 대한 분석 프레임워크와 추론 시점 개입 방법**을 제안한다.

**평가 대상 모델:**
- Gemini 2.5 Flash, Gemini 2.5 Pro (Google DeepMind)
- GPT-5 mini, GPT-5 (OpenAI)
- DeepSeek-R1 (오픈소스)

**제안하는 추론 시점 개입 방법:**

| 방법 | 설명 |
|------|------|
| **Output Ensemble** | 동일 질문에 대해 N회(최대 10회) 응답 샘플링 후 다수결 투표 |
| **Translation Ensemble (TrEn-k)** | 원본 질문 + k개 번역본을 함께 제시하여 단일 응답 유도 |
| **TEA** (baseline) | 질문을 영어로 번역 후 답변, 원래 언어로 재번역 |

**TrEn-k의 핵심 설계:**
- 소스 언어와 다른 스크립트의 번역만 포함 (소스 언어 힌트 주입 방지)
- $k \in \{1, 3, 5\}$ 실험

### 2-4. 성능 향상

**ECLeKTic 벤치마크 (Transfer Score):**

| 방법 | G-2.5-Flash | G-2.5-Pro | GPT-5-mini | GPT-5 | DeepSeek |
|------|------------|-----------|-----------|-------|---------|
| Baseline | 40.2 | 51.3 | 29.0 | 50.9 | 23.7 |
| TEA | 41.9 | 54.7 | 27.9 | 50.1 | 36.2 |
| TrEn-1 | 42.8 | 51.9 | 31.7 | 54.5 | 30.8 |
| TrEn-3 | 44.7 | 50.5 | 33.7 | 54.8 | 35.8 |
| **TrEn-5** | **45.3** | **55.6** | **32.5** | **54.8** | **35.7** |

- Gemini/GPT 계열: 상대적 **8~13%** 향상
- DeepSeek: 상대적 **50%** 향상 (23.7 → 35.7)
- Output Ensembling: 소스-타겟 L2 거리를 최대 **12절대점** 감소

**$\pi$ 추정치:**
- ECLeKTic: $\pi \approx 0.88$ ~ $0.96$ (모델별)
- MMLU (with mixup): $\pi \approx 0.94$ ~ $0.97$
- → **90~97%의 사례에서 교차 언어 오류가 비편향** (분산에 의한 것)

### 2-5. 한계

1. **추론 비용**: Output Ensembling은 N배의 추론 비용 발생 (N=10 시 10배)
2. **훈련 시점 개입 부재**: 분산 감소를 위한 파인튜닝/포스트트레이닝 방법은 미탐구
3. **저자원 언어 한계**: 학습 데이터에 거의 없는 언어에는 적용 불가
4. **TrEn의 실용성 제한**: 오라클 번역 필요 (실제 환경에서는 자동 번역 오류 가능)
5. **분산의 근본 원인 불명**: 왜 타겟 언어에서 분산이 증가하는지의 메커니즘 미규명
6. **LLM-as-Judge 노이즈**: ECLeKTic 자유형 응답 평가 시 비영어 인스턴스에서 판정 오류율 상승

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문의 핵심적 통찰은 교차 언어 일반화 실패의 원인을 재정의한다는 점에서, 일반화 성능 향상에 대한 중요한 함의를 가진다.

### 3-1. 이론적 함의: 일반화는 이미 내재되어 있다

**Proposition 3**은 매우 중요한 일반화 성능 향상 가능성을 시사한다:

$$\Pr(\hat{y}_t = y^{\text{mode}}) \geq f\left(\Pr(\hat{y}_s = y^{\text{mode}})\right) \quad \text{(순증가 함수)}$$

이 결과는 **모델이 실제로 지식을 언어 독립적으로 보유하고 있음**을 강력히 시사한다. 즉, 교차 언어 일반화는 원리적으로 달성 가능하며, 문제는 지식 자체가 아니라 **타겟 언어에서의 확신도(confidence) 전이 실패**다.

### 3-2. 소스 분산 제어를 통한 일반화 향상

실험(Section 4.2, Figure 5)에서 소스 신뢰도( $\Pr(y^{\text{mode}}_s)$ )가 높을수록 소스-타겟 일치도가 단조 증가함이 검증되었다. 이는 다음을 의미한다:

- **소스 언어에서 모델이 더 확신할수록, 타겟 언어에서도 자동으로 더 잘 답한다**
- 따라서 소스 언어의 응답 품질을 높이는 모든 기법(chain-of-thought, self-consistency 등)이 자동으로 교차 언어 성능도 향상시킨다

### 3-3. 도메인 어댑테이션 관점에서의 일반화

교차 언어 격차를 **비지도 도메인 어댑테이션(UDA)의 특수 사례**로 볼 때, Ben-David et al.(2006)의 타겟 위험 상한:

$$R_T(h) \leq R_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(S, T) + \lambda$$

기존 접근은 표현 발산( $d_{\mathcal{H}\Delta\mathcal{H}}$ )을 최소화하는 데 집중했다. 그러나 이 논문의 발견에 따르면 소스 위험( $R_S(h)$ ) 자체를 줄이는 것(= 분산 감소 = 소스 신뢰도 향상)이 자동으로 타겟 위험도 줄인다는 새로운 관점을 제공한다.

### 3-4. Self-Consistency와의 연계

논문이 관련 연구로 언급하는 Wang et al.(2023)의 Self-Consistency 기법과의 연결:
- Self-Consistency는 추론 모델의 분산을 줄여 정확도를 높임
- 이 논문의 발견은 **Self-Consistency가 교차 언어 일반화에도 동일하게 적용됨**을 이론적으로 설명

### 3-5. 엔티티(Entity)가 일반화의 핵심 병목

Appendix A의 SBET(Source Borrowed Entities in Target) 실험:
- 타겟 언어 질문에서 엔티티를 소스 언어 형태로 유지하면 교차 언어 격차의 **60~70%가 해소**
- 엔티티 인식이 일반화 성능의 핵심 병목임을 시사
- 흥미롭게도 엔티티의 다국어 언급 빈도와 다국어 정확도 간 상관관계는 낮음 → 단순히 훈련 데이터를 늘리는 것만으로는 해결 불충분

### 3-6. 포스트트레이닝을 통한 일반화 향상 가능성

저자들은 Discussion에서 명시적으로 제안:
> "post-training to alleviate the issue of high response variance in the target language"

타겟 언어에서 분산을 줄이도록 파인튜닝한다면 추론 비용 없이도 일반화 성능을 향상시킬 수 있다. 이는:
- RLHF/DPO에서 타겟 언어 응답의 일관성을 보상 신호로 활용
- 대조 학습(contrastive learning)으로 소스-타겟 응답 분포 정렬

등의 방향으로 구체화될 수 있다.

### 3-7. 크로스모달 일반화로의 확장 가능성

저자들은 교차 언어 격차 분석이 **교차 모달(cross-modal) 격차** (텍스트 입력 vs. 음성 입력 간 성능 차이)에도 적용 가능하다고 언급한다. 이는 일반화 성능 향상의 범위가 다국어를 넘어 멀티모달로 확장될 수 있음을 시사한다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4-1. 연구 영향

#### 📐 패러다임 전환
이 논문은 교차 언어 격차 연구의 **패러다임을 근본적으로 전환**한다:

| 기존 패러다임 | 새로운 패러다임 |
|-------------|--------------|
| 지식 장벽 → 표현 정렬 문제 | 분산 증가 → 확신도 전이 문제 |
| 비용이 큰 사전훈련 개입 필요 | 추론 시점의 간단한 앙상블로 해결 가능 |
| 편향 제거 중심 | 분산 감소 중심 |

#### 🔬 후속 연구 방향 자극

1. **포스트트레이닝 연구**: 타겟 언어 분산을 줄이는 파인튜닝 방법 (RLHF, DPO 등)
2. **자동 번역 품질과 TrEn 결합**: 오라클 번역 없이도 TrEn을 실용적으로 적용하는 연구
3. **분산 원인 규명**: 왜 타겟 언어에서 분산이 증가하는가? (사전훈련 데이터 분포, 크로스-언어 팩추얼 불일치 등)
4. **Calibration 연구**: 타겟 언어에서 모델 확신도를 높이는 캘리브레이션 기법
5. **저자원 언어**: 프레임워크가 충분히 학습되지 않은 언어로 확장 가능한지 탐구

#### 🌐 산업적 영향

다국어 LLM 서비스(Gemini, GPT 등)의 성능 균등화를 위한 실용적이고 즉시 적용 가능한 방법론을 제공한다. 특히 추가 학습 없이 추론 시점에서만 적용 가능하다는 점에서 높은 실용성을 가진다.

### 4-2. 앞으로 연구 시 고려할 점

#### ⚠️ 방법론적 고려사항

1. **온도(Temperature) 편향 문제**: 온도 조절로 분산을 줄이는 것은 편향을 유발할 수 있다 (논문에서 실증적으로 확인). 따라서 분산 감소 기법은 반드시 **비편향 추정량** 기반이어야 한다.

2. **LLM-as-Judge 신뢰성**: 비영어 인스턴스에서 자동 평가의 정확도 하락 문제. 특히 저자원 언어 평가 시 인간 평가 혹은 규칙 기반 평가(Year-ECLeKTic 방식) 병행 필요.

3. **번역 오류의 영향**: 편향으로 분류된 약 5~10%의 사례 중 상당수가 번역 오류(datasets errors)에서 기인. 데이터셋 품질 검증이 선행되어야 한다.

4. **앙상블 비용 대비 효과 최적화**: N=10 앙상블은 10배의 추론 비용. 실용적 배포를 위해 최적 N을 태스크별로 탐색해야 한다.

#### 🔍 이론적 고려사항

5. **$\pi$의 인과적 해석**: 논문에서 추정된 $\pi \approx 0.9$는 상관 분석이지, 인과적으로 교차 언어 오류의 90%가 분산에 의한 것임을 직접 증명하지는 않는다. 인과 추론 방법론(예: 도구 변수, 반사실 분석)을 통한 검증이 필요하다.

6. **Proposition 3의 가정 강도**: 비편향($\kappa=1$)을 가정할 때 소스-타겟 로짓이 완전히 정렬된다고 가정하지만, 실제로는 부분적 정렬일 가능성. 더 일반적인 설정에서의 이론 확장 필요.

7. **응답 공간 정규화의 한계**: 다언어 응답을 단일 개념 공간으로 정규화하는 과정 자체가 오류를 도입할 수 있으며, 이에 대한 민감도 분석이 필요하다.

#### 📊 실험적 고려사항

8. **저자원 언어 심층 분석**: 현재 실험에서 히브리어, 인도네시아어 등 저자원 언어에서의 결과는 제한적. 더 넓은 언어 커버리지(예: 아프리카 언어, 희소 언어) 실험 필요.

9. **태스크 다양성**: 현재 분석은 지식 집약적 QA 태스크(Closed-book)에 집중. 추론, 수학, 코딩 등 다른 태스크로의 일반화 여부 검토 필요.

10. **모델 크기 효과**: 논문의 모델들이 모두 대형 모델. 소형 모델(예: 7B, 1B)에서 $\pi$ 값과 앙상블 효과가 어떻게 변하는지 분석 필요.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5-1. 비교 연구 매핑

| 연구 | 교차 언어 격차의 원인 | 해결 접근법 | 이 논문과의 관계 |
|-----|---------------------|------------|----------------|
| Jiang et al. (2020) - XFaCTR | 지식 불일치 | 다국어 프로브 | 문제 정의 참조 |
| Kassner et al. (2021) - Multilingual LAMA | 언어별 지식 차이 | 평가 프레임워크 | 문제 정의 참조 |
| Chua et al. (2024) - MMLU with mixup | 지식 장벽, 언어 이해 실패 | TEA (번역-영어-답변) | **이 논문이 반박**: 편향보다 분산이 지배적 |
| Wang et al. (2023) - Self-Consistency | 추론 분산 | 다수결 앙상블 | 이 논문의 분산 감소 아이디어와 연결 |
| Brinkmann et al. (2025) | 언어 무관 표현 존재 | 해석 가능성 분석 | 이 논문 주장 지지 |
| Schäfer et al. (2024) | 언어 불균형 훈련의 역할 | 클론 언어 실험 | 이 논문 주장 지지 |
| Fierro et al. (2024) | LLM이 중간 영어 표현 사용 | 메커니즘 분석 | 표현 수준 해석 |
| Wang et al. (2025a) - "Lost in multilinguality" | 번역 오류(중간→최종) | 해석 분석 | 엔티티 가설과 연관 |
| Lu et al. (2025) - "Paths not taken" | 번역 양방향 오류 | 스티어링 벡터 | 훈련 기반 접근 (이 논문과 상호보완) |
| Liu & Niehues (2025) | 중간층 표현 불일치 | 표현 정렬 파인튜닝 | 훈련 기반 접근 (이 논문과 상호보완) |
| Goldman et al. (2025) - ECLeKTic | 교차 언어 지식 전이 평가 | 벤치마크 | 이 논문의 주요 평가 데이터셋 |
| Hupkes & Bogoychev (2025) - MultiLoKo | 다국어 지역 지식 | 벤치마크 | 이 논문의 보조 평가 데이터셋 |

### 5-2. 핵심 차별점

**기존 주류 연구(Chua et al., 2024; Wang et al., 2024b):**
- 가정: 교차 언어 오류 = 편향(biased error)
- 해결책: 표현 정렬, 사전훈련 개선 등 비용이 큰 개입

**이 논문:**
- 반론: 교차 언어 오류의 90~97%는 비편향(unbiased error) = 순수 분산
- 해결책: 추론 시점의 앙상블 (비용 효율적, 즉시 적용 가능)

이 논문은 **Wang et al.(2023)의 Self-Consistency** 아이디어를 교차 언어 영역으로 확장·이론화한 것으로도 해석될 수 있으며, 동시에 **Fierro et al.(2024), Wang et al.(2025a)의 표현 수준 분석**과는 다른 **행동 수준(behavioral level) 분석**을 제공한다는 점에서 상호보완적이다.

---

## 📚 참고 자료 (논문 내 인용 기준)

**본 논문:**
- Piratla et al. (2026). *Rethinking Cross-Lingual Gaps via Response Variance*. arXiv:2510.15551v2

**논문 내 주요 인용 문헌:**
1. Goldman et al. (2025). *ECLeKTic: a novel challenge set for evaluation of cross-lingual knowledge transfer*. arXiv:2502.21228
2. Chua et al. (2024). *Crosslingual Capabilities and Knowledge Barriers in Multilingual Large Language Models*. arXiv:2406.16135
3. Wang et al. (2023). *Self-consistency improves chain of thought reasoning in language models*. arXiv:2203.11171
4. Jiang et al. (2020). *XFaCTR: Multilingual factual knowledge retrieval from pretrained language models*. arXiv:2010.06189
5. Kassner et al. (2021). *Multilingual LAMA: Investigating knowledge in multilingual pretrained language models*. arXiv:2102.00894
6. Brinkmann et al. (2025). *Large language models share representations of latent grammatical concepts across typologically diverse languages*. arXiv:2501.06346
7. Schäfer et al. (2024). *The role of language imbalance in cross-lingual generalisation*. arXiv:2404.07982
8. Fierro et al. (2024). *How do multilingual language models remember facts?* arXiv:2410.14387
9. Wang et al. (2025a). *Lost in multilinguality: Dissecting cross-lingual factual inconsistency in transformer language models*. arXiv:2504.04264
10. Lu et al. (2025). *Paths not taken: Understanding and mending the multilingual factual recall pipeline*. arXiv:2505.20546
11. Liu & Niehues (2025). *Middle-layer representation alignment for cross-lingual transfer in fine-tuned LLMs*. arXiv:2502.14830
12. Hupkes & Bogoychev (2025). *MultiLoKo: a multilingual local knowledge benchmark for LLMs spanning 31 languages*. arXiv:2504.10356
13. Ben-David et al. (2006). *Analysis of representations for domain adaptation*. NeurIPS 19
14. Holtzman et al. (2019). *The curious case of neural text degeneration*. arXiv:1904.09751
15. Guo et al. (2025). *DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning*
16. Dumas et al. (2024). *How do llamas process multilingual text? A latent exploration through activation patching*. ICML 2024 Workshop
17. Hendrycks et al. (2020). *Measuring massive multitask language understanding*. arXiv:2009.03300
18. Qin et al. (2023). *Cross-lingual prompting: Improving zero-shot chain-of-thought reasoning across languages*. EMNLP 2023
19. Zhou et al. (2023). *Enhancing cross-lingual prompting with dual prompt augmentation*. ACL 2023 Findings
20. Google Vertex AI. *text-multilingual-embedding-002*. cloud.google.com/vertex-ai, 2024
