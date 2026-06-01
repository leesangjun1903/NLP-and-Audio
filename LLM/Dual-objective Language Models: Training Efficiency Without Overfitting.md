# Dual-Objective Language Models: Training Efficiency Without Overfitting

> **참고 자료:**
> - **주요 논문:** David Samuel & Lucas Georges Gabriel Charpentier, "Dual-Objective Language Models: Training Efficiency Without Overfitting," *Published as a conference paper at ICLR 2026*, arXiv:2512.14549v3 [cs.CL], 27 Mar 2026.
> - **관련 인용 문헌 (논문 내 참조):** Muennighoff et al. (2023), Brown et al. (2020), Austin et al. (2021), Ou et al. (2025), Charpentier & Samuel (2024), Prabhudesai et al. (2025), Ni et al. (2025), Nie et al. (2025a, 2025b), Raffel et al. (2020), Devlin et al. (2019), Vaswani et al. (2017), Kaplan et al. (2020), Hoffmann et al. (2022), Villalobos et al. (2024), Lv et al. (2024), Samuel (2025), Katz et al. (2025), Xue et al. (2025), Wu et al. (2023), Arriola et al. (2025), Sahoo et al. (2025), Lou et al. (2024) 외 다수 (논문 참고문헌 목록 전체).

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음 한 문장으로 요약된다:

> **"오토리그레시브(AR) 목표함수와 마스크드 디퓨전(MD) 목표함수를 단일 모델에서 동시에 학습하면, 두 방법의 약점을 상쇄하고 장점만을 결합할 수 있다."**

- **AR 모델의 문제:** 훈련 효율(sample efficiency)은 높지만, 데이터 반복 학습 시 **과적합(overfitting)**에 취약하다.
- **MD 모델의 문제:** 과적합에는 강하지만, **훈련 효율이 낮아** 동일 계산량 대비 성능이 뒤처진다.
- **Dual-Objective의 해결:** 두 목표함수를 적절히 혼합하면 **빠른 수렴 + 과적합 방지**를 동시에 달성하며, **추론(inference) 시 구조 변경이 불필요**하다.

### 주요 기여 (3가지)

| # | 기여 내용 |
|---|-----------|
| 1 | **Dual-Objective 학습 방법 제안:** AR + MD 손실함수를 단일 트랜스포머로 결합, 단방향·양방향 태스크 모두 지원 |
| 2 | **체계적 실증 연구:** 50개 언어 모델 훈련/평가를 통해 데이터 반복 횟수 × 목표함수 비율 × 최종 성능의 관계를 정량적으로 분석 |
| 3 | **실용적 가이드라인 도출:** 일반 데이터 환경(Remark 1)과 데이터 제약 환경(Remark 2)에 맞는 최적 $\alpha$ 설정 권고안 제시 |

---

## 2. 상세 분석: 문제 → 방법 → 구조 → 성능 → 한계

### 2-1. 해결하고자 하는 문제

현대 LLM 훈련의 핵심 딜레마는 **"데이터 장벽(Data Wall)"** 문제다. Villalobos et al. (2024)에 따르면 고품질 훈련 데이터는 유한하며, 스케일링 법칙(Kaplan et al., 2020; Hoffmann et al., 2022)을 충족하려면 지수적으로 많은 데이터가 필요하다. 이에 따라 **동일 데이터를 반복 학습**하는 상황이 불가피해지는데:

- **AR 모델** ($\alpha = 1$): Muennighoff et al. (2023)에 따르면 데이터를 **16회 이상 반복**하면 의미 있는 학습이 불가능하고 과적합이 발생한다.
- **MD 모델** ($\alpha = 0$): 과적합에 강하지만, AR 대비 **훈련 sample efficiency**가 현저히 낮다 (Nie et al., 2025a).

### 2-2. 제안 방법 및 수식

#### (a) 언어 모델의 일반적 목표 (MLE)

$$\underset{\theta}{\text{argmax}} \; \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}} \left[ \log p_{\theta}(\boldsymbol{x}) \right] \tag{1}$$

#### (b) 오토리그레시브(AR) 손실함수

$$\mathcal{L}_{\text{AR}}(\boldsymbol{x}; \theta) = -\sum_{i=1}^{|\boldsymbol{x}|} \log p_{\theta}(x_i \mid \boldsymbol{x}_{<i}) \tag{2}$$

인과적 어텐션(causal attention)을 사용하며, 각 위치에서 이전 토큰들만을 조건으로 다음 토큰을 예측한다.

#### (c) 마스크드 디퓨전(MD)의 순방향 확산 과정

시간 변수 $t \in [0, 1]$에 따라 토큰을 점진적으로 마스킹:

$$q_{t|0}(\boldsymbol{x}^t \mid \boldsymbol{x}) \overset{\text{def}}{=} \prod_{i=1}^{|\boldsymbol{x}|} q_{t|0}(x_i^t \mid x_i); \quad q_{t|0}(x_i^t \mid x_i) \overset{\text{def}}{=} \begin{cases} 1 - t, & x_i^t = x_i \\ t, & x_i^t = \texttt{mask} \end{cases} \tag{3}$$

#### (d) 마스크드 디퓨전(MD) 손실함수

음의 로그우도의 상한(upper bound)을 최소화:

$$\mathcal{L}_{\text{MD}}(\boldsymbol{x}; \theta) = -\int_0^1 \mathbb{E}_{\boldsymbol{x}^t \sim q_{t|0}(\cdot|\boldsymbol{x})} \left[ \frac{1}{t} \sum_{\{i \mid x_i^t = \texttt{mask}\}} \log p_{\theta}(x_i \mid \boldsymbol{x}^t) \right] dt \tag{4}$$

적분은 $t \sim \mathcal{U}(0, 1)$에 대한 기댓값으로 표현되어, 몬테카를로 샘플링으로 추정된다.

#### (e) **Dual-Objective 통합 손실함수 (핵심)**

$$\underset{\theta}{\text{argmin}} \; \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}} \left[ \alpha \mathcal{L}_{\text{AR}}(\boldsymbol{x}; \theta) + (1 - \alpha) \mathcal{L}_{\text{MD}}(\boldsymbol{x}; \theta) \right] \tag{5}$$

여기서 $\alpha \in [0, 1]$은 두 목표함수의 균형을 조절하는 핵심 하이퍼파라미터다:
- $\alpha = 1$: 순수 AR 모델
- $\alpha = 0$: 순수 MD 모델
- $0 < \alpha < 1$: Dual-Objective 모델

> **[Erratum 참고]** 논문의 초기 버전(학습 코드 포함)에서는 AR 항에 계수 2를 추가한 $2\alpha\mathcal{L}\_{\text{AR}} + (1-\alpha)\mathcal{L}_{\text{MD}}$ 형태를 사용했으나, 수학적으로 불필요하여 최종본(v3)에서 수식 (5)로 수정되었다.

#### (f) 평가 지표: Pseudo Log-Likelihood (PLL)

양방향 평가를 위한 PLL 추정 (계산 효율성을 위해 MC 추정 대신 사용):

$$\log p_{\theta}(\boldsymbol{w}) \approx \sum_{i=1}^{|\boldsymbol{w}|} \log p_{\theta}\!\left(w_i \mid \boldsymbol{c} \oplus w_0 \oplus \cdots \oplus w_{i-1} \oplus \underbrace{\texttt{mask} \oplus \cdots \oplus \texttt{mask}}_{n \text{개}} \oplus w_{i+n} \oplus \cdots \oplus w_{|\boldsymbol{w}|}\right) \tag{10}$$

### 2-3. 모델 구조

| 항목 | 상세 |
|------|------|
| **파라미터 수** | 470M (임베딩 제외 360M) |
| **레이어 수** | 24 |
| **Hidden Size** | 1,024 |
| **Attention Heads** | 16 |
| **FFN 중간 크기** | 3,554 |
| **어휘 크기** | 51,200 (BPE) |
| **정규화** | Pre-norm + RMSNorm |
| **위치 임베딩** | RoPE (Rotary Positional Embedding) |
| **활성화 함수** | SwiGLU |
| **옵티마이저** | Muon optimizer (WSD 스케줄) |

**두 모드의 핵심 차이점은 오직 입력과 어텐션 마스크뿐이다:**

- **AR 모드:** 원본 토큰 입력 + **인과적(causal) 어텐션 마스크**
- **MD 모드:** 부분 마스크된 입력 + **전방향(bidirectional) 어텐션 마스크**

이를 가능하게 하는 핵심 기술이 **Masked Next-Token Prediction (MNTP)**으로, 위치 $i$의 히든 스테이트로 위치 $i+1$의 토큰을 예측한다. 이 좌측-이동(left-shift) 파라미터화가 표현력을 손실시키지 않음을 논문은 RASP 언어를 통해 형식적으로 증명한다 (Theorem 1: Left-shift closure).

**훈련 구현:** 256개 GPU 장치를 단일 목표함수에 할당하여 처리량 손실 없이 $\alpha \in \{i/256 \mid i = 0, 1, \ldots, 256\}$ 범위에서 세밀한 조정이 가능하다.

### 2-4. 성능 향상

실험은 32B 토큰 예산, 반복 횟수 $R \in \{1, 2, 4, 8, 16, 32, 64, 128, 256\}$에서 수행되었다.

#### 핵심 성능 결과 (표 2, normalized autoregressive score):

| 데이터 반복 | 모델 (최적 $\alpha$) | 평균 점수 | AR-only ($\alpha=1$) |
|------------|---------------------|-----------|----------------------|
| 1× | Dual ($\alpha = 63/64$) | **26.9** | 26.1 |
| 32× | Dual ($\alpha = 3/4$) | **23.9** | 22.0 |
| 128× | Dual ($\alpha = 1/8$) | **19.1** | 9.4 |

128회 반복 시 Dual 모델은 AR 전용 모델 대비 **약 2배의 성능**을 달성하며, 과적합의 파국적 영향을 차단한다.

#### 실용 가이드라인:

> **Remark 1 (일반 데이터 환경, $R \leq 16$):**
> $\alpha \approx 63/64$로 설정 — AR 성능 유지 + MD 성능 대폭 향상

> **Remark 2 (데이터 제약 환경, $R > 32$):**
> AR 목표함수가 훈련 데이터를 약 16회 반복 학습하도록 $\alpha$ 설정 (예: $R=128$이면 $\alpha \approx 1/8$)

> **Remark 3 (Prefix LM 일반화):**
> 추론 시 프롬프트의 조건부(context) 부분에 양방향 어텐션 마스크를 적용하면 추가 학습 없이도 성능이 1%p 이상 향상

### 2-5. 한계

논문이 직간접적으로 인정하는 한계:

1. **규모 일반화 미검증:** 실험은 470M 파라미터 모델에 한정. 수십억 파라미터 이상의 대형 모델에서 최적 $\alpha$ 권고안이 동일하게 유효한지는 **실험적으로 미검증**이다 (이론적 논증만 제시).
2. **단일 언어(영어) 한정:** HPLT v2 영어 코퍼스만 사용. 다국어 환경이나 저자원 언어에서의 효과는 불명확하다.
3. **학습 단계 한정:** 사전훈련(pretraining)에만 집중. 지시 조정(instruction tuning), RLHF 등 이후 단계에서의 영향은 미연구.
4. **최적 $\alpha$의 데이터셋 의존성:** 최적 비율 탐색을 위해 50개 모델을 훈련해야 하므로 새로운 환경에서의 $\alpha$ 탐색 비용이 존재한다.
5. **이분법적 목표함수:** AR과 MD 두 가지 목표함수만 탐구. 다른 목표함수(예: prefix LM, span corruption)와의 삼중 결합은 미탐구.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

이 논문에서 일반화(generalization) 성능 향상은 다음 세 가지 메커니즘을 통해 나타난다.

### 3-1. 과적합 억제를 통한 일반화

MD 목표함수는 구조적으로 **정규화(regularizer)** 역할을 수행한다. MD 학습 시 모델은 임의의 마스킹 패턴 $\boldsymbol{x}^t$에 대해 원본 토큰을 복원해야 하며, 이는 특정 데이터 패턴의 암기를 어렵게 만든다.

$$\mathcal{L}_{\text{MD}}(\boldsymbol{x}; \theta) = -\int_0^1 \mathbb{E}_{\boldsymbol{x}^t} \left[ \frac{1}{t} \sum_{\{i \mid x_i^t = \texttt{mask}\}} \log p_{\theta}(x_i \mid \boldsymbol{x}^t) \right] dt$$

분모의 $t$는 마스킹 비율에 따른 역가중치로, **높은 마스킹 비율(어려운 복원)에 더 높은 가중치**를 부여한다. 이는 모델이 단순한 국소적 패턴보다 **전역적 언어 구조**를 학습하도록 유도한다.

실험적 증거: 128회 반복 환경에서 AR 전용 모델은 검증 손실이 발산하지만($\times$ 표시), Dual 모델은 최적 $\alpha$ 근방에서 발산 없이 훈련된다 (Figure 4, Figure 6).

### 3-2. 양방향 문맥 이해를 통한 일반화

AR 모델의 구조적 한계인 **"역전 저주(Reversal Curse)"** (Berglund et al., 2024) — "A는 B이다"를 학습해도 "B는 A이다"를 유추 못하는 문제 — 를 MD 목표함수가 완화한다. 양방향 어텐션을 통한 학습이 보다 **대칭적이고 완전한 의미 표현**을 형성하기 때문이다.

실증적 증거: 동일한 데이터 환경($R = 1$)에서도 MD 평가 기준으로는 $\alpha = 1$ (AR 전용)이 가장 낮은 성능을 보이며, $\alpha$ 값이 약간만 낮아져도 (예: $\alpha = 255/256$) 양방향 성능이 대폭 향상된다.

### 3-3. Prefix LM으로의 제로샷 일반화 (Remark 3)

가장 주목할 만한 일반화 능력이다. Dual 모델은 **명시적인 Prefix LM 학습 없이도** 추론 시 프롬프트의 조건부 부분에 양방향 어텐션을 적용하면 성능이 향상된다.

표 10에 따르면:
- 1회 반복: AR 26.9 → Prefix **27.9** (+1.0%p)
- 32회 반복: AR 23.9 → Prefix **25.0** (+1.1%p)
- 128회 반복: AR 19.1 → Prefix **20.5** (+1.4%p)

이는 두 가지 어텐션 패턴(인과적/양방향)에 대한 동시 학습이 모델 내부에 **맥락 처리 유연성**을 내재화했음을 시사한다. Katz et al. (2025)의 연구가 별도 어댑터 훈련을 통해 유사한 효과를 보인 것과 달리, 이 논문의 Dual 모델은 **추가 훈련 없이** 이를 달성한다.

### 3-4. 데이터 효율성 향상을 통한 일반화

Dual 모델은 128회 반복(실효 학습 데이터 약 256M 토큰) 환경에서도 평균 19.1점이라는 비자명한(nontrivial) 제로샷 성능을 달성한다. 이는 **극단적인 데이터 제약 환경에서도 일반화 가능한 표현**을 학습할 수 있음을 의미한다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4-1. 목표함수 결합 관련 연구 비교

| 연구 | 방법 | AR 모드 사용 가능 | 구조 변경 | MD 일반화 | 과적합 분석 |
|------|------|:---:|:---:|:---:|:---:|
| **T5** (Raffel et al., 2020) | Span Corruption + AR | ✗ (인코더-디코더) | ✓ (대규모) | ✗ | ✗ |
| **BART** (Lewis et al., 2020) | 다양한 노이즈 + AR | △ | ✓ (인코더-디코더) | ✗ | ✗ |
| **GLM** (Du et al., 2022) | Blank Infilling + AR | △ | △ (특수 위치 인코딩) | ✗ | ✗ |
| **CM3** (Aghajanyan et al., 2022) | 인과적 마스킹 + AR | ✓ | ✗ | ✗ | ✗ |
| **UL2** (Tay et al., 2023) | 다중 노이저 혼합 | ✓ | ✗ | ✗ | ✗ |
| **AntLM** (Yu et al., 2024) | AR → MLM → AR 커리큘럼 | ✓ | ✗ | ✗ | ✗ |
| **GPT-BERT** (Charpentier & Samuel, 2024) | AR + MLM | ✓ | ✗ | ✗ | ✗ |
| **이 논문** (Samuel & Charpentier, 2026) | AR + **MD** (확산 기반) | ✓ | ✗ | ✓ | ✓ (50개 모델) |

**AntLM과의 차별점:** AntLM은 목표함수를 순차적으로 전환하여 **이전 목표함수를 망각**하는 문제가 발생하지만, 본 논문은 두 목표함수를 **동시에 지속적으로** 학습하여 이를 방지한다.

### 4-2. 마스크드 디퓨전 언어 모델 발전

| 연구 | 주요 기여 | 관련성 |
|------|-----------|--------|
| **Austin et al. (2021)** | 이산 상태공간 구조적 디퓨전 모델 제안 | MD의 이론적 기반 |
| **Lou et al. (ICML 2024)** | 데이터 분포 비율 추정 기반 이산 디퓨전 | MD 개선 방법 |
| **Sahoo et al. (NeurIPS 2024)** | 단순하고 효과적인 MD 언어 모델 (MDLM) | 본 논문과 비교 기준 |
| **Ou et al. (ICLR 2025)** | 흡수 이산 디퓨전의 조건부 분포 모델링 | 수식 (4) 유도의 수학적 기반 |
| **Nie et al. (ICLR 2025a)** | 텍스트 MD 모델 스케일링 | AR 대비 MD의 sample efficiency 비교 |
| **Prabhudesai et al. (2025)** | 데이터 제약 환경에서 MD가 AR 능가 | 본 논문 결과 일부 확인 |
| **Ni et al. (2025)** | MD 모델의 초월적 데이터 학습 능력 | 본 논문 결과 일부 확인 |

### 4-3. 데이터 제약 스케일링 법칙

| 연구 | 핵심 발견 | 본 논문과의 관계 |
|------|-----------|----------------|
| **Kaplan et al. (2020)** | 손실이 컴퓨팅·모델·데이터의 거듭제곱 법칙 따름 | 스케일링의 이론적 배경 |
| **Hoffmann et al. (2022)** | Chinchilla: 모델·데이터 균형 스케일링 필요 | 32B 토큰 (Chinchilla의 4× 초과) 실험 설계 근거 |
| **Muennighoff et al. (2023)** | AR 모델은 16회 이상 반복 학습 시 의미 없음 | 본 논문이 확장: Dual은 이 한계를 **한 자릿수 이상** 상회 |
| **Villalobos et al. (ICML 2024)** | 인간 생성 데이터 고갈 임박 예측 | 데이터 장벽 문제의 실증적 동기 |

### 4-4. 추론 효율성 관련 연구

| 연구 | 방법 | 본 논문과의 차별점 |
|------|------|-----------------|
| **AR-Diffusion** (Wu et al., 2023) | 왼쪽→오른쪽 디노이징 편향 디퓨전 | 여전히 디퓨전 모델; AR 모드 불가 |
| **Block Diffusion** (Arriola et al., 2025) | 청크 단위 AR + 디퓨전 디코딩 | 속도 향상이 목적; AR-MD 일반화 불가 |
| **Xue et al. (2025)** | 인과적 마스크 트랜스포머로 MD 파라미터화 | MD 단독 사용은 여전히 준최적; 본 논문이 결합으로 해결 |
| **Katz et al. (2025)** | GPT에 양방향 마스크 적용 (어댑터 필요) | 추가 훈련 필요; 본 논문은 사전훈련만으로 달성 |

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

#### (a) LLM 사전훈련 패러다임의 재검토

이 논문은 "LLM은 AR로 학습한다"는 지배적 패러다임에 **아키텍처 변경 없이** 도전한다. 특히:

- **데이터 효율 연구의 방향 전환:** 더 많은 데이터 수집보다 **기존 데이터의 반복 학습 효율화**가 연구 방향이 될 수 있다.
- **하이브리드 목표함수의 정상화:** 단일 목표함수가 항상 최선이 아님을 실증적으로 입증함으로써, 다중 목표함수 결합 연구의 정당성을 확립한다.

#### (b) 데이터 장벽 대응 전략

Muennighoff et al. (2023)이 "AR 모델은 16회 이상 반복 시 한계"라고 발견한 것을 Dual 모델이 **한 자릿수 이상** 확장했다. 이는 동일 데이터로 훨씬 더 많은 학습이 가능함을 의미하며, 특히:
- **저자원 언어** LLM 개발에 직접적인 함의
- **합성 데이터(synthetic data)** 활용 연구와 결합 가능성

#### (c) 평가 방법론 기여

PLL이 MC 추정 대비 **10배 이상 빠르면서 더 정확**함을 보인 것(Section J)은 MD 계열 모델 평가의 표준이 될 수 있다.

### 5-2. 앞으로 연구 시 고려할 점

#### ① 대규모 모델에서의 최적 $\alpha$ 검증

```
현재: 470M 파라미터
필요: 7B, 13B, 70B+ 파라미터에서의 실험
핵심 가설: 모델 크기↑ → 두 모드의 파라미터 부담↓ → Dual 효과↑
```

저자들은 이론적 논증만 제시했으므로, **대규모 실증 검증**이 필수적이다.

#### ② 다국어 및 저자원 언어 환경

영어 단일 코퍼스 실험을 넘어, **언어별 최적 $\alpha$가 달라지는지** 탐구해야 한다. 저자원 언어는 본질적으로 데이터 제약 환경이므로 Dual 목표함수의 효과가 더 클 수 있다.

#### ③ 파인튜닝 및 정렬 단계에서의 영향

사전훈련 이후:
- **지시 조정(Instruction Tuning):** MD 성분이 포함된 모델이 AR 전용 모델과 동일한 파인튜닝 방법으로 최적화되는지 불명확하다.
- **RLHF/DPO:** 보상 모델과의 상호작용에서 양방향 표현이 미치는 영향 탐구 필요.

#### ④ 최적 $\alpha$ 탐색 비용 절감

현재 50개 모델을 훈련해 최적 $\alpha$를 찾는 방식은 비용이 크다. **예측 방법(predictive scaling)**이나 **프록시 메트릭** 개발이 필요하다:

$$\alpha^* = f(R, \text{model size}, \text{domain}, \ldots)$$

이러한 함수 $f$를 데이터 기반으로 학습하는 **메타 학습(meta-learning)** 접근도 유망하다.

#### ⑤ 세 가지 이상의 목표함수 혼합

$$\underset{\theta}{\text{argmin}} \; \mathbb{E}_{\boldsymbol{x}} \left[ \alpha_1 \mathcal{L}_{\text{AR}} + \alpha_2 \mathcal{L}_{\text{MD}} + \alpha_3 \mathcal{L}_{\text{Prefix}} + \cdots \right], \quad \sum_i \alpha_i = 1$$

Prefix LM, Span Corruption, Denoising 등 다른 목표함수와의 **다중 결합** 효과 탐구.

#### ⑥ 동적 $\alpha$ 스케줄링

현재는 학습 전 과정에서 $\alpha$가 고정된다. **학습 진행에 따른 동적 $\alpha$ 조정**이 추가 성능 향상을 가져올 수 있다:

$$\alpha(t) = \alpha_0 \cdot f(t), \quad t \in [0, T]$$

초기에는 AR 비율을 높여 빠른 수렴을 유도하고, 후기에는 MD 비율을 높여 과적합을 방지하는 커리큘럼 전략.

#### ⑦ 합성 데이터와의 결합

데이터 장벽 문제의 또 다른 대응으로 합성 데이터(LLM이 생성한 데이터) 활용이 증가하는 추세다. Dual 목표함수가 **합성 데이터의 반복 학습** 환경에서 더욱 강점을 발휘할 수 있는지 탐구가 필요하다.

---

## 요약 정리

```
핵심 기여: AR + MD 동시 학습 → 추론 오버헤드 없이 과적합 방지 + 훈련 효율 유지
핵심 수식: argmin_θ E[α·L_AR + (1-α)·L_MD]
핵심 발견: 모든 데이터 반복 환경에서 Dual > 단일 목표함수
핵심 권고: R≤16이면 α≈63/64, R>32이면 AR이 약 16회 반복되도록 α 설정
핵심 한계: 470M 소규모 + 영어 단일언어 + 사전훈련 단계만 검증
향후 과제: 대규모 검증, 다국어 확장, 동적 α, 다중 목표함수, 파인튜닝 영향
```
