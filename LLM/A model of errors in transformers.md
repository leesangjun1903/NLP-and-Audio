
# A model of errors in transformers

## Executive Summary

"A Model of Errors in Transformers"(Raju & Netrapalli, 2026)는 대규모 언어 모델(LLM)이 산술, 동적 계획법 등 단순하지만 반복적인 작업에서 발생하는 오류의 근본 원인을 체계적으로 분석한 혁신적 연구이다. 저자들은 기존의 "표현력 부족" 또는 "추론 붕괴"라는 해석 대신, **注意 메커니즘의 작은 오류들이 누적되면서 임계값을 초과할 때** 오류가 발생한다는 새로운 이론을 제시하고, 200,000개 이상의 실험을 통해 두 개의 파라미터로만 정확히 모델링할 수 있음을 입증했다. [arxiv](https://arxiv.org/html/2601.14175v1)

***

## 1. 핵심 주장과 주요 기여

### 1.1 연구의 핵심 주장

논문의 중심 가설은 다음과 같다:

**LLM의 오류는 transformer 아키텍처의 내재적 한계가 아니라, 반복적인 토큰 처리 과정에서 attention 가중치의 부정확성이 누적되어 발생한다.** 

기존 문헌에서는 다음과 같이 주장했다:
- **표현력 부족**: LLM이 합성 함수(compositional function) 계산을 표현할 수 없음 (Dziri et al., 2023) [journals.lww](https://journals.lww.com/10.1097/JS9.0000000000001619)
- **추론 붕괴**: 일정 길이 초과 시 LLM의 논리적 추론이 급격히 악화됨 (Shojaee et al., 2025)
- **패턴 매칭**: LLM이 참된 알고리즘을 학습하지 못하고 단순 패턴만 의존 (Nikankin et al., 2024)

논문의 통찰력은 이러한 현상들이 사실은 **정밀한 attention mechanism 구현의 어려움**에서 비롯된다는 것이다. 명시적 알고리즘을 제공하거나 강제하더라도 여전히 오류가 발생하는 실험 결과는 표현력 문제가 아님을 강력히 암시한다. [arxiv](https://arxiv.org/html/2502.11771v1)

### 1.2 주요 과학적 기여

#### 1.2.1 정량적 오류 공식
논문의 가장 중요한 기여는 정확도(accuracy)를 complexity parameter $c$의 함수로 나타내는 폐쇄형 공식을 유도한 것이다:

$$a = 1 - \frac{\Gamma\left(\frac{q}{2}, \frac{q}{2rc^2}\right)}{\Gamma(q/2)}$$

이 공식은 단 **두 개의 해석 가능한 파라미터**로 모든 복잡한 행동을 설명한다:
- **$r$**: 기본 노이즈율(elementary noise rate per token)
- **$q$**: 가능한 오류 방향의 개수(number of plausible error directions)

#### 1.2.2 광범위한 실증 검증
- **8가지 다양한 작업**: 리스트 뒤집기, 중첩 선형 변환, 동적 계획법, Tower of Hanoi, 덧셈(3가지 변형), 곱셈
- **200,000개 이상의 distinct prompts**: 이전 연구 대비 수십배 규모
- **3개의 최신 LLM**: Gemini 2.5 Flash, Gemini 2.5 Pro, DeepSeek R1

#### 1.2.3 물리학적 관점의 도입
유효장 이론(effective field theory)의 개념을 LLM에 적용하여, 수조 개의 파라미터가 단 2개의 유효 파라미터로 자동으로 재구성되는 현상을 설명한다. 이는 물리학에서 유체의 복잡한 미시 구조가 밀도와 점성 두 파라미터로 설명되는 것과 유사하다. [arxiv](https://arxiv.org/html/2601.14175v1)

#### 1.2.4 Prompt Engineering을 통한 개선 증명
다항식 표현을 사용한 modified prompt로 multiplication 작업의 노이즈율 $r$을 약 50% 감소시켰으며, 이로 인해 Flash 모델이 Pro 모델의 표준 prompt보다도 우수한 성능을 달성했다. [arxiv](https://arxiv.org/pdf/2601.14175v1.pdf)

***

## 2. 문제 정의 및 해결 방법

### 2.1 정확히 해결하고자 하는 문제

논문이 다루는 문제는 다음과 같이 정의된다:

**결정론적 출력이 필요한 작업(arithmetic, DP, combinatorics)에서 LLM이 시퀀스 길이(complexity) 증가에 따라 accuracy가 power-law로 급격히 감소하는 현상의 원인 규명**

구체적인 문제의 특징:
1. **입출력**: 작은 토큰 집합(digits, bits)에서 drawn된 토큰 시퀀스
2. **알고리즘**: 각 단계에서 소수의 토큰만 attend하면 되는 부분 해결 가능 문제
3. **현상**: complexity $c$ 증가 → accuracy 급격히 감소 (지수에서 power-law로)

이는 purely academic하지 않은 문제이다. 실제로 LLM을 수학 또는 논리 추론에 사용할 때 매우 중요한 제약이 된다.

### 2.2 제안하는 방법론

#### 2.2.1 이상화된 모델(Idealized Model)

완벽하게 작동하는 참조 모델을 정의한다:

$$M_{id} = O \circ L_{id} \circ E_{id}$$

**구성 요소**:
- **$E_{id}$**: Embedding layer

$$E_{id}(t_1, \ldots, t_n) \rightarrow (v_1, \ldots, v_n) \in \mathbb{R}^{d_{emb} \times n}$$

- **$L_{id}$**: Stacked attention layers
  
  각 layer $\ell$에서:

$$v^{\ell}\_i = \sigma\left(\sum_j A_{ij}^{\ell} v^{\ell-1}_j\right)$$
  
  여기서 attention weights:

$$A_{ij} = \frac{\exp(Qv_i \cdot Kv_j / \sqrt{d})}{\sum_k \exp(Qv_i \cdot Kv_k / \sqrt{d})}$$

- **$O$**: Output layer
  $$O(w) = \arg\max_{t \in \mathcal{V}} \langle w, E(t) \rangle$$

이 모델은 **arbitrary precision과 충분한 chain-of-thought 토큰**이 있으면 임의의 결정론적 작업을 완벽하게 수행할 수 있다 (Turing completeness). [arxiv](https://arxiv.org/abs/2310.16028)

#### 2.2.2 Effective Model의 정의

실제 LLM이 암묵적으로 구현하는 모델을 정의한다:

$$M_{eff} = O \circ L_{eff} \circ E_{eff}$$

**핵심 가정 (A1)**: Effective model의 아키텍처는 idealized model과 동일하나, 파라미터가 약간 다르다.

이 가정은 모델이 **일관된 알고리즘**을 사용한다는 의미이다. (Vanilla addition에서 Pro의 실패 원인)

#### 2.2.3 오류 벡터와 임계값 모델

**오류 벡터 정의**:
$$\delta(S_{in}, n) = L_{eff} \circ E_{eff}(t_1 \ldots t_n) - L_{id} \circ E_{id}(t_1 \ldots t_n)$$

**오류 예측 조건 (A2)**:
$$M_{eff}(t_1 \ldots t_n) \neq M_{id}(t_1 \ldots t_n) \iff |\delta(S_{in}, n)|^2 > \tau^2$$

직관: 오류 벡터가 다른 토큰의 embedding vector와 충돌할 정도로 크면 잘못된 토큰이 선택될 확률이 높아진다.

#### 2.2.4 오류 누적의 수학적 분석

**복합 오류 벡터를 직교 basis로 분해**:
$$|\delta_{max}|^2 = \sum_{i=1}^q E_i^2(S_{in}) w_i^2$$

여기서:
- $q$: 효과적 오류 방향의 개수
- $E_i$: 입력 $S_{in}$에 의존하는 계수

**가정 A3**: 계수들이 Gaussian 분포를 따른다:
$$E_i \sim \mathcal{N}(0, v)$$

정당화: $c$가 충분히 크면 입출력 시퀀스 공간을 continuous space로 근사 가능.

**가정 A4**: 오류 분산의 스케일링
$$v = \sigma^{2\alpha} c^{2\alpha}$$

여기서 $\alpha=1$ (quadratic scaling) 또는 $\alpha=1/2$ (linear scaling).

$\alpha=1$의 정당화: Attention error가 context의 모든 벡터에 propagate되고, positional information이 부정확하면 오류들이 **강하게 상관**되어 quadratic accumulation 발생.

#### 2.2.5 최종 정확도 공식 유도

$$P(|\delta_{max}|^2 < \tau^2) = \int_0^{\tau^2} \frac{1}{(2v)^{q/2} \Gamma(q/2)} x^{q/2-1} e^{-x/(2v)} dx$$

$r = q\sigma^2/\tau^2$로 정의하면:

$$\boxed{a = 1 - \frac{\Gamma(q/2, q/(2rc^2))}{\Gamma(q/2)}}$$

**큰 c에서의 Power-law 근사**:
$$a \approx \left(\frac{q/2}{rc^2}\right)^{q/2}$$

이는 실험에서 관찰되는 $a \sim c^{-\alpha}$ 패턴과 정확히 일치한다.

### 2.3 모델 아키텍처의 세부 사항

#### 2.3.1 Complexity Parameter의 정의

$$c = \text{idealized model이 처리해야 할 최소 토큰 수}$$

여기에 포함되는 것:
1. 입력 시퀀스의 토큰
2. Chain-of-thought (중간 계산) 토큰

예시:
- **덧셈** ($n$자리): $c = n$ (carry 추적 포함)
- **Tower of Hanoi** (첫 $c$번 이동 생성): $c = \text{이동 수}$
- **동적 계획법** (크기 $n$ 리스트): $c = n$

**스케일링 불변성**: $a(c_1, r_1, q) = a(c_2, r_2, q)$ if $c_1 : c_2 = r_1 : r_2$

따라서 절대적인 값보다는 **비율**이 중요하다.

#### 2.3.2 Stacked Attention Layers의 오류 전파

$d$개의 layer가 있을 때:

$$\delta^{(\ell)}_i = \sum_j \Delta A_{ij}^{(\ell)} v^{(\ell-1)}_j + \sum_j A_{ij}^{(\ell)} \delta^{(\ell-1)}_j + \delta_{\sigma}^{(\ell)}$$

여기서:
- 첫 번째 항: Attention weight 오류
- 두 번째 항: 이전 layer의 오류 전파
- 세 번째 항: Nonlinearity 오류

**핵심 관찰**: 각 layer에서 $c$개의 벡터 위에 sum이 있으므로, 오류가 **누적**된다. Autoregressive generation이므로, 각 output token을 생성할 때마다 새로운 오류가 추가되고, 이들이 상관되어 quadratic하게 증가한다.

***

## 3. 성능 향상 및 경험적 검증

### 3.1 8가지 작업에 대한 상세 결과

#### Task 1: List Reversal
**설정**: 길이 $c$인 리스트를 역순으로 출력

**결과**:
- Flash: $\chi^2 = 4.2 \times 10^{-4}$, $r = 2.67 \times 10^{-4}$, $q = 4.2$
- DeepSeek: $\chi^2 = 4.2 \times 10^{-4}$, $r = 2.27 \times 10^{-4}$, $q = 4.2$
- Pro: 더 우수한 성능 ($r = 5.0 \times 10^{-5}$)

**해석**: $q \approx 4$는 각 위치에서 ~4개의 경쟁하는 토큰이 있다는 의미 (likely artifact tokens).

#### Task 2: Nested Linear Transformations
**설정**: $C_{i+1} = A_i \cdot C_i + B_i$ 반복 ($c$번)

**결과**: 모든 모델이 공식을 정확히 따름 ($\chi^2 < 10^{-4}$)

**중요성**: 이는 compositional reasoning을 요구하는 대표적 작업. Explicit algorithm 제공하지 않았는데도 성능이 예측된다는 것은:
- LLM이 **암묵적으로 올바른 알고리즘 습득**
- 오류의 원인이 알고리즘 부족이 아님을 증명

#### Task 3: Dynamic Programming (Maximum Non-Adjacent Sum)
**설정**: 길이 $c$ 리스트에서 인접하지 않은 최대합 부분수열 찾기

**결과**: 
- Explicit pseudocode 제공했을 때도 accuracy 저하 관찰
- Flash: $r = 1.30 \times 10^{-4}$, DeepSeek: $r = 3.85 \times 10^{-4}$

**중요성**: 명시적 알고리즘을 제공하고도 오류가 발생한다는 것은 "reasoning collapse" 가설을 직접 반박한다. 문제는 이해/추론이 아니라 **누적 오류**.

#### Task 4: Tower of Hanoi
**설정**: 10개 디스크를 첫 $c$번 이동으로 옮기기 (explicit algorithm 제공)

**결과**: 명시적 알고리즘 제공했을 때도 accuracy 급격히 감소
- Flash: $r = 2.61 \times 10^{-4}$

**결론**: Shojaee et al. (2025)의 "reasoning collapse" 가설 재검토 필요. 실제는 attention noise 누적.

#### Task 5: Vanilla Addition (완전 실패 사례)
**문제**: Gemini Pro가 공식을 **완전히** 따르지 않음

| 모델 | 결과 | $\chi^2$ |
|------|------|---------|
| Flash | 공식 따름 | $10^{-3}$ |
| DeepSeek | 공식 따름 | $10^{-3}$ |
| Pro | **실패** | 매우 큼 |

**가설**: Assumption A1 위반
- Pro가 다른 길이의 수에 대해 **서로 다른 알고리즘 사용**
- 또는 학습 과정에서 여러 알고리즘 습득

**검증**: "Addition with Algorithm" 변형 (명시적 단계별 지시)
- Pro도 공식 따름! ($\chi^2 = 2.2 \times 10^{-3}$)
- 일관된 알고리즘 강제 → A1 만족

**통찰**: 모델의 "flexibility"가 때로는 inconsistency를 야기할 수 있다.

#### Task 6-8: Binary Addition, Multiplication
**Binary addition**: 모두 공식 따름, decimal보다 성능 저하 (r 약 2배 크게)
- Tokenization 복잡성 증가의 영향일 수 있음

**Multiplication** ($7869 \times n$): 
- 모든 모델 정확히 공식 따름
- 이전에 compositional ability 부족으로 해석됨 (Dziri et al., 2023)
- 본 연구는 **표현력이 아닌 precision 문제**임을 입증 [arxiv](https://arxiv.org/pdf/2601.14175.pdf)

### 3.2 파라미터 해석 및 의미

#### Parameter r (Noise Rate)

$$r = \frac{q\sigma^2}{\tau^2}$$

**물리적 의미**:
- 각 token processing step에서의 기본 노이즈율
- 작을수록 좋음

**작업 간 비교**:
| 작업 | Flash r | DeepSeek r | Pro r |
|------|---------|-----------|-------|
| List reversal | $2.67 \times 10^{-4}$ | $2.27 \times 10^{-4}$ | $5.0 \times 10^{-5}$ |
| Binary addition | $7.84 \times 10^{-3}$ | $4.50 \times 10^{-3}$ | $5.16 \times 10^{-3}$ |
| Multiplication (vanilla) | $7.84 \times 10^{-3}$ | $4.50 \times 10^{-3}$ | $7.9 \times 10^{-3}$ |

**패턴**:
- Pro: 대체로 더 작은 r (더 정밀한 attention)
- Arithmetic이 symbolic task보다 높은 r
- 오류가 누적되는 작업이 더 높은 r

#### Parameter q (Error Direction Count)

**의미**: 각 단계에서 경쟁하는 가능한 오류 토큰의 개수

**관찰되는 값**: O(1), 대부분 1~15 범위
- List reversal: q ≈ 4.2
- Nested linear: q ≈ 13.9
- DP: q ≈ 11.1

**해석**:
1. **작은 q**: Vocabulary의 매우 제한된 부분만 relevant
2. **O(1) 특성**: embedding space에서 정확한 토큰과만 몇 개 경쟁
3. **Task-dependent**: 복잡한 task일수록 q 경향히 크다

### 3.3 Prompt Engineering을 통한 개선

#### 실험: Polynomial Representation for Multiplication

**표준 prompt**:
- $7869 \times 611912436665956692$ 직접 계산
- Flash: $r = 7.84 \times 10^{-3}$, q = 9.5

**개선된 prompt** (polynomial approach):
```
1. 수를 다항식으로 표현: P(x) = Σ p_n x^n
2. 다항식 곱셈: R(x) = P(x) × Q(x)
3. Carry 처리 및 정규화
4. 최종 수로 변환
```

**결과**:
- Flash: $r = 3.82 \times 10^{-3}$, q = 4.6
- **r 감소**: 약 50% 개선!
- Accuracy: 표준보다 현저히 우수
- **Pro보다도 우수한 성능 달성**

**메커니즘**: 각 단계에서 중간 결과에 $x^k$ 태그를 붙임 → attention이 관련 위치에만 집중
- 무관한 토큰에 attend할 오류 감소
- σ (노이즈 크기) 감소 → r 감소

### 3.4 모델 간 차이

#### Gemini 2.5 Pro의 특이성

Pro는 Flash/DeepSeek과 다른 행동 패턴:
- 대부분 작업에서 더 우수한 기본 성능 (작은 c)
- **Vanilla addition에서 완전 실패**
- "Addition with algorithm" 강제 시 회복

**해석**:
- Pro가 더 **flexible/adaptive**한 알고리즘 사용
- 길이에 따라 다른 전략 선택 가능
- A1 가정 (일관된 파라미터) 위반

#### Flash vs DeepSeek

- **상당히 유사한 r 값**: 아키텍처 유사성 시사
- DeepSeek이 조금 더 나은 성능 (특히 arithmetic)
- 모두 공식을 정확히 따름

***

## 4. 일반화 성능 향상의 가능성

### 4.1 현재 공식의 일반화 예측력

#### 강점

**광범위한 검증**:
- 200,000개 이상의 실험
- 8가지 structurally distinct tasks
- 3개의 최신 모델
- 높은 reproducibility

**예측력**:
- Accuracy를 미지의 prompt에 대해 사전 예측 가능
- (r, q) 추정만으로도 성능 범위 파악 가능

#### 한계

**Task 제약**:
- 결정론적 출력 필요 (생성 작업 불가)
- 작은 토큰 집합에서만 (vocabulary size 효과 미분석)
- Deterministic logic chains

**검증 범위**:
- c ≤ 200: 더 긴 시퀀스에서의 거동 불명
- Dense task에만 적용: 희소 구조 작업은 미지수

### 4.2 길이 일반화 (Length Generalization)

#### 현재 공식의 예측

**큰 c에서**:
$$a \approx c^{-q/2}$$

이는 다음을 의미:
- q=2: $a \propto c^{-1}$
- q=4: $a \propto c^{-2}$

**실험 관찰**과의 비교:
- 관찰: power-law decay (exponent 1~2)
- 이론: exponent = q/2 (일치!)

#### 개선 가능성

**1. 초과 길이에서의 적응**:
- 훈련 길이: $c_{train}$
- 테스트 길이: $c_{test} > c_{train}$

**현재 이해**: 
- r, q는 모델과 prompt의 고유 특성
- c의 extrapolation에는 new mechanism 필요할 수 있음

**가능한 메커니즘**:
- Relative scaling: $r(c) = r_0 + \delta \log c$?
- Layer-wise specialization
- Sparse attention patterns

#### 관련 최신 연구

**RASP-Generalization Conjecture** (Zhou et al., 2024): [arxiv](https://arxiv.org/abs/2310.16028)
- Transformer가 "short RASP program"으로 표현 가능한 작업은 길이 일반화 달성
- 본 공식과의 통합 가능성?

**State Passing** (Ruiz & Gu, 2025): [arxiv](https://arxiv.org/pdf/2507.02782.pdf)
- Recurrent models에서 상태 분포 적응으로 길이 일반화
- Transformer의 "effective context state" 개념 도입 가능성

### 4.3 합성 일반화 (Compositional Generalization)

#### 논문의 주요 통찰

**핵심**: 합성 능력 부족이 아니라 **precision 문제**

근거:
1. Explicit algorithm 제공 → 여전히 오류
2. 동적 계획법, 중첩 선형 변환 → 공식 따름
3. 곱셈 → 모든 모델이 정확히 수행

#### 일반화 개선 전략

**전략 1: Structured Prompting**
- 각 합성 단계에 명시적 레이블/태그
- attention이 relevant subtask에만 집중
- Polynomial representation의 일반화

**전략 2: Intermediate Supervision**
- 각 compositional step의 중간 결과 예측
- Loss function: main task + auxiliary tasks
- Dziri et al.도 보이듯이 curriculum 효과 있음

**전략 3: Modular Architecture**
- 각 compositional unit별 separate head
- 학습 가능한 routing
- 최근 mixture-of-experts 아키텍처 활용

### 4.4 2020년 이후 관련 최신 연구 비교

#### 오류 관련 연구

| 논문 | 초점 | 본 논문과의 관계 |
|------|------|----------------|
| LEMMA (2025) [arxiv](http://arxiv.org/pdf/2503.17439.pdf) | Learning from errors | Error data를 직접 활용; 본 논문은 error mechanism 분석 |
| Numerical Error Analysis (2025) [arxiv](https://arxiv.org/pdf/2503.10251.pdf) | Round-off in transformers | Finite precision 관점; 본 논문은 semantic error에 초점 |
| Error-Free Linear Attention (2026) [arxiv](https://arxiv.org/abs/2512.12602) | 이론적 zero error 보장 | 하드웨어/알고리즘 수준; 본 논문은 existing models 분석 |

#### Arithmetic 능력

| 논문 | 주요 결과 | 본 논문과의 비교 |
|------|---------|-----------------|
| Mathematical Reasoning (Shrestha et al., 2025) [arxiv](https://arxiv.org/pdf/2502.08680.pdf) | Logical vs arithmetic error 분류 | Logical error 분석; 본 논문은 accumulation mechanism |
| Validation Gap (Bertolazzi et al., 2025) [arxiv](https://arxiv.org/html/2502.11771v1) | Circuit analysis로 error detection | 검출 메커니즘; 본 논문은 발생 메커니즘 |
| Value-Aware Representations (2025) [arxiv](https://arxiv.org/html/2601.09706v1) | 수치 표현 중요성 | 표현 형식의 영향; 본 논문은 computation의 영향 |

#### Compositional/Length Generalization

| 논문 | 주요 기여 | 통합 가능성 |
|------|---------|-----------|
| RASP-Generalization (Zhou et al., 2024) [arxiv](https://arxiv.org/abs/2310.16028) | Length generalization의 필요조건 | 본 공식에 RASP 복잡도 통합? |
| Provable Length Generalization (2024) [arxiv](https://arxiv.org/html/2402.04875v4) | 이론적 보장 | 학습 가능성과 일반화의 trade-off |
| State Tracking (Li et al., 2025) [arxiv](https://arxiv.org/abs/2503.02854) | Permutation composition의 메커니즘 | Effective state의 dimensionality |
| Compositional Reasoning Questions (Yehudai et al., 2025) [openreview](https://openreview.net/pdf?id=nUZaI7aRb2) | Expressive power 비교 (Transformer vs RNN) | 아키텍처 간 noise 누적 속도의 차이? |

#### Attention 메커니즘 혁신

| 접근 | 특징 | 오류 누적에 미치는 영향 |
|------|------|----------------------|
| Linear Attention + Gating (2025) [arxiv](https://arxiv.org/html/2507.19595v1) | Data-dependent decay | 선택적 attention으로 r 감소? |
| Selective Attention [journals.lww](https://journals.lww.com/10.1097/JS9.0000000000001619) | 무관 토큰 필터링 | σ 직접 감소 |
| Mamba/State Space Models [arxiv](https://arxiv.org/pdf/2509.19633.pdf) | Recurrent formulation | Different error propagation law? |

***

## 5. 핵심 한계 및 향후 연구 과제

### 5.1 현재 공식의 한계

#### 가정의 한계

**A1: Consistent Architecture**
- Gemini Pro의 vanilla addition 실패로 검증됨
- 모델이 길이에 따라 다른 전략 선택 가능

**A2: Threshold Crossing**
- Binary 가정: 오류 또는 정확 (intermediate correctness 없음)
- 실제: soft probability 분포

**A3: Gaussian Error Distribution**
- 특수한 경우 non-Gaussian distribution 가능
- Appendix에서 다른 분포 가능성 논의하나 미검증

**A4: Quadratic Scaling (α=1)**
- Linear scaling (α=1/2)도 좋은 fit
- α를 fixed가 아닌 learnable parameter로?

#### 경험적 편차

**Small c에서의 과대평가**:
- List reversal: Flash에서 작은 c에서 공식이 accuracy 과대예측
- 추가 파라미터 $d$ 도입: $c → c+d$ (offset)

**모델 특이성**:
- Pro의 vanilla addition: 완전 실패
- 아키텍처/학습 특성에 따른 variation 미분석

**작업 특이성**:
- 특정 prompt formulation에 매우 민감
- Robust한 파라미터화 필요

### 5.2 향후 필수 연구 과제

#### 이론적 확장

1. **Assumption A1의 일반화**
   - Model selection mechanism 분석
   - Length-dependent algorithm switching 모델화

2. **오류 분포의 정확한 특성화**
   - 실제 오류가 Gaussian인가?
   - Layer-wise error의 independence 검증
   
3. **Effective Field Theory의 수학화**
   - 물리학의 rigorous renormalization group framework 도입
   - Neural network의 "scaling laws" 재해석

4. **Context의 역할**
   - Irrelevant token에 attend하는 이유?
   - Attention mechanism의 내재적 noise 분석

#### 실무적 개선

1. **실세계 작업으로 확대**
   - Open-domain QA, machine translation
   - 오류 정의가 복잡한 작업

2. **Active Error Correction**
   - Self-correction prompt의 최적화
   - Error detection head의 학습 가능성

3. **아키텍처 개선**
   - Selective attention (관련 토큰만 attend)
   - Hierarchical computation with early exits

#### 일반화 능력 강화

1. **Transfer Learning**
   - 한 작업의 (r, q)가 유사 작업으로 일반화?
   - Cross-task parameter prediction

2. **Domain Shift Analysis**
   - 새로운 도메인에서 r, q의 변화
   - OOD detection으로의 응용

3. **다국어 모델**
   - Language family에 따른 r 변화
   - Universality of q?

### 5.3 해석 불가능한 사례

**Vanilla Addition with Pro**: 왜 실패했나?

**가능한 설명들**:
1. 학습 중 데이터 분포의 암묵적 길이 편향
2. Fine-tuning 단계에서의 catastrophic forgetting
3. Multiple algorithm의 ensemble (각 길이대별)

**필요한 연구**:
- Mechanistic interpretability (attention head 분석)
- Activation geometry (hidden state의 dimension)
- Gradient flow analysis

***

## 6. 구조적 기여의 의미

### 6.1 과학적 패러다임의 전환

본 논문은 **LLM의 오류 분석 패러다임**을 다음과 같이 전환한다:

**기존 패러다임**:
- 오류 → 모델의 구조적 한계
- 해결책 → 새로운 아키텍처, 더 많은 학습 데이터

**새로운 패러다임**:
- 오류 → 정밀성의 누적 손실
- 해결책 → Prompt engineering, attention mechanism 개선, 구조적 입력 재설계

이는 매우 실질적인 함의를 가진다. 수조 파라미터를 추가할 필요 없이, **기존 모델에서도** prompt 디자인으로 오류를 크게 줄일 수 있다는 증거를 제시했기 때문이다.

### 6.2 물리학과 AI의 연결

**Effective Field Theory의 도입**:
- 물리학에서 수질량-수백조 원자의 미시 상태 → 몇 개의 매크로 파라미터
- 본 연구: 수조 개 파라미터 → 2개의 유효 파라미터

이는 다음을 시사한다:

> "AI 시스템도 자연의 다른 복잡한 시스템들과 마찬가지로, 본질적으로 renormalization되는 구조를 가질 수 있다."

장기적으로, 이는 **AI의 이론적 기초**를 물리학과 수학의 성숙한 도구들로 구축할 가능성을 제시한다.

### 6.3 실용적 영향

**즉각적 응용**:
1. **성능 예측**: (r, q) 측정 → 특정 complexity에서의 성능 예측
2. **Prompt 최적화**: r을 최소화하는 입력 재설계
3. **모델 비교**: 동일한 (r, q) framework 아래에서 공정한 비교

**장기적 영향**:
1. **AI Safety**: 오류 예측 가능성 → 신뢰도 보장 가능
2. **Resource Allocation**: 복잡도별 계산량 추정
3. **Hybrid Systems**: LLM + Symbolic reasoning의 optimal mixing

***

## 결론

"A Model of Errors in Transformers"는 단순하지만 강력한 통찰력으로 LLM의 오류 현상을 설명한다: **작은 attention 오류들의 지속적 누적**. 

핵심 성과는:

1. **정량적 공식**: 불과 2개 파라미터로 복잡한 오류 패턴 모델화
2. **광범위한 검증**: 200,000개 실험을 통한 강력한 실증적 지지
3. **실제 개선**: Prompt engineering으로 50% 성능 향상 시연
4. **이론적 심화**: Physics의 effective field theory 개념 도입

한편, 모델의 한계도 명확하다:
- 특정 모델(Pro)과 특정 작업(vanilla addition)에서 실패
- Long-context(c > 200)에서의 거동 미검증
- 합성 일반화의 완전한 메커니즘 규명 필요

향후 연구는 이러한 한계를 해결하면서도, **기존 모델에서 오류를 줄이는 실질적 방법**을 찾는 것에 초점을 맞춰야 한다.

궁극적으로, 이 연구는 AI의 신뢰도를 높이기 위해서는 모델의 **구조적 개선**만큼이나 **입출력의 설계**가 중요함을 보여준다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b68abe6a-a89e-4c6e-ba6e-fc14ff70bd35/2601.14175v1.pdf)

***

## 참고문헌

<span style="display:none">[^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/html/2601.14175v1

[^1_2]: https://arxiv.org/pdf/2601.14175v1.pdf

[^1_3]: https://journals.lww.com/10.1097/JS9.0000000000001619

[^1_4]: https://arxiv.org/html/2502.11771v1

[^1_5]: https://arxiv.org/abs/2310.16028

[^1_6]: https://arxiv.org/pdf/2601.14175.pdf

[^1_7]: https://arxiv.org/pdf/2507.02782.pdf

[^1_8]: http://arxiv.org/pdf/2503.17439.pdf

[^1_9]: https://arxiv.org/pdf/2503.10251.pdf

[^1_10]: https://arxiv.org/abs/2512.12602

[^1_11]: https://arxiv.org/pdf/2502.08680.pdf

[^1_12]: https://arxiv.org/html/2601.09706v1

[^1_13]: https://arxiv.org/html/2402.04875v4

[^1_14]: https://arxiv.org/abs/2503.02854

[^1_15]: https://openreview.net/pdf?id=nUZaI7aRb2

[^1_16]: https://neurips.cc/virtual/2025/poster/116131

[^1_17]: https://arxiv.org/html/2507.19595v1

[^1_18]: https://arxiv.org/pdf/2509.19633.pdf

[^1_19]: 2601.14175v1.pdf

[^1_20]: https://arxiv.org/abs/2505.16270

[^1_21]: https://www.ndss-symposium.org/wp-content/uploads/2025-2287-paper.pdf

[^1_22]: https://aclanthology.org/2024.semeval-1.52

[^1_23]: https://biss.pensoft.net/article/182910/

[^1_24]: https://aclanthology.org/2025.findings-emnlp.1038

[^1_25]: https://www.sciltp.com/journals/hm/articles/2504000541

[^1_26]: https://www.semanticscholar.org/paper/222125ec80c1f6fa3cf00d938685db2cbdd4a626

[^1_27]: https://arxiv.org/abs/2508.09820

[^1_28]: https://jisem-journal.com/index.php/journal/article/view/6615

[^1_29]: https://arxiv.org/abs/2404.06948

[^1_30]: http://arxiv.org/pdf/2410.13857.pdf

[^1_31]: http://arxiv.org/pdf/2410.06638.pdf

[^1_32]: http://arxiv.org/pdf/2307.14995.pdf

[^1_33]: https://arxiv.org/pdf/2311.14737.pdf

[^1_34]: https://www.ijml.org/vol12/1079-R013.pdf

[^1_35]: https://www.microsoft.com/en-us/research/publication/learning-and-generalization-in-rnns/

[^1_36]: https://proceedings.neurips.cc/paper/2021/file/b04c387c8384ca083a71b8da516f65f6-Paper.pdf

[^1_37]: https://www.nature.com/articles/s43588-025-00854-1

[^1_38]: https://aclanthology.org/2021.emnlp-main.448.pdf

[^1_39]: https://aclanthology.org/2025.emnlp-main.411.pdf

[^1_40]: https://huggingface.co/papers/2512.12602

[^1_41]: https://www.oist.jp/sites/default/files/2024-03/cnru_nn08.pdf

[^1_42]: https://magazine.sebastianraschka.com/p/state-of-llms-2025

[^1_43]: https://arxiv.org/pdf/2507.19595.pdf

[^1_44]: https://akiraaptx.blog/2017/08/29/sequence-modeling-recurrent-and-recursive-nets/

[^1_45]: https://arxiv.org/html/2601.18699v1

[^1_46]: https://arxiv.org/html/2409.00894v1

[^1_47]: https://arxiv.org/html/2503.04111v1

[^1_48]: https://arxiv.org/html/2510.00184v1

[^1_49]: https://arxiv.org/html/2505.24187v1

[^1_50]: https://arxiv.org/html/2601.08122v1

[^1_51]: https://arxiv.org/pdf/2507.12379.pdf

[^1_52]: https://arxiv.org/html/2209.01610v3

[^1_53]: https://arxiv.org/html/2403.14932v1

[^1_54]: https://arxiv.org/pdf/1703.01619.pdf

[^1_55]: https://ashpublications.org/blood/article/144/Supplement 1/2263/532909/Retrieval-Augmented-Generation-for-the-Detection

[^1_56]: https://arxiv.org/abs/2512.07109

[^1_57]: https://arxiv.org/abs/2504.15349

[^1_58]: https://www.semanticscholar.org/paper/6c8597eba86c4ed4fc9b5df4d8cf1b2a17d64a69

[^1_59]: https://www.semanticscholar.org/paper/ff62bf8b62028543f612f592b803ebb9da833ef6

[^1_60]: https://link.springer.com/10.1007/s44282-024-00128-7

[^1_61]: https://ashpublications.org/blood/article/146/Supplement 1/5590/556716/Bone-marrow-adipocytes-BMA-as-a-novel-biomarker-of

[^1_62]: https://aclanthology.org/2023.findings-acl.480.pdf

[^1_63]: https://arxiv.org/pdf/2501.19215.pdf

[^1_64]: https://arxiv.org/html/2501.08537v1

[^1_65]: https://arxiv.org/abs/2109.15256

[^1_66]: https://arxiv.org/html/2502.15801v1

[^1_67]: https://arxiv.org/pdf/2309.07624.pdf

[^1_68]: http://arxiv.org/pdf/2410.09918v1.pdf

[^1_69]: https://arxiv.org/pdf/2210.11265.pdf

[^1_70]: https://link.aps.org/doi/10.1103/PhysRevD.109.105007

[^1_71]: https://ai.meta.com/blog/advancing-ai-theory-with-a-first-principles-understanding-of-deep-neural-networks/

[^1_72]: https://kozmath.github.io/papers/strassen.pdf

[^1_73]: https://proceedings.iclr.cc/paper_files/paper/2024/file/45ed1a72597594c097152ef9cc187762-Paper-Conference.pdf

[^1_74]: https://www.youtube.com/watch?v=EKsjYodyuv4

[^1_75]: https://www.sciencedirect.com/science/article/abs/pii/S1566253524002690

[^1_76]: https://aclanthology.org/2024.findings-acl.32/

[^1_77]: https://digitalcommons.wayne.edu/oa_dissertations/2438/

[^1_78]: https://aclanthology.org/2025.naacl-long.420.pdf

[^1_79]: https://machinelearning.apple.com/research/dataset-decomposition

[^1_80]: https://arxiv.org/abs/2502.17665

[^1_81]: https://arxiv.org/pdf/2503.01544.pdf

[^1_82]: https://arxiv.org/html/2510.17742v2

[^1_83]: https://arxiv.org/pdf/2505.23683.pdf

[^1_84]: https://arxiv.org/pdf/2510.17196.pdf

[^1_85]: https://arxiv.org/abs/1710.06570

[^1_86]: https://arxiv.org/html/2508.17298v1

[^1_87]: https://arxiv.org/html/2510.17196v1

[^1_88]: https://arxiv.org/html/2510.17469v1

[^1_89]: https://arxiv.org/abs/2305.02334

[^1_90]: https://arxiv.org/html/2503.01544v1

[^1_91]: https://arxiv.org/html/2510.14826v1

[^1_92]: https://arxiv.org/html/2502.17665v1
