# Embedding by Elicitation: Dynamic Representations for Bayesian Optimization of System Prompts

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 **시스템 프롬프트 최적화**를 **집계(aggregate) 피드백만을 활용하는 샘플 제약적 블랙박스 최적화 문제**로 재정의한다. 기존 자동 프롬프트 최적화(APO) 방법들이 개별 예제 수준의 레이블, 오류 추적, 텍스트 비평 등 세분화된 피드백에 의존하는 반면, 실제 배포 환경(사용자 만족도, 안전사고 발생률, 장기 태스크 완료율 등)에서는 **프롬프트 하나당 단일 스칼라 점수**만 관찰 가능한 경우가 많다.

이 문제를 해결하기 위해 **ReElicit**을 제안한다. 핵심 아이디어는 LLM 자체를 **의미론적 표현 공간의 동적 구축자**로 활용하는 것이다. 즉, LLM은 단순한 프롬프트 생성기가 아니라 **적응형 의미론적 좌표계 설계자**로 기능한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **문제 재정의** | 집계 피드백 기반 시스템 프롬프트 튜닝을 블랙박스 최적화로 형식화 |
| **ReElicit 프레임워크** | LLM 유도 의미론적 특징 공간 + GP 대리 모델 + 배치 획득 함수 통합 루프 |
| **도달 가능성(Reachability) 이론 분석** | 표현 오차가 최적화 품질에 미치는 영향을 RKHS 프레임워크로 형식화 |
| **실증 평가** | 10개 벤치마크, 30회 평가 예산, 4개 베이스라인 대비 최고 성능 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**핵심 설정**: 목적함수 $f: \mathcal{X} \to \mathbb{R}$ 는 프롬프트 $x \in \mathcal{X}$ 를 스칼라 점수 $y = f(x)$ 로 매핑한다. 최적화기는 각 프롬프트에 대해 **오직 하나의 집계 스칼라 점수**만 관찰 가능하며, 개별 예제 레이블, 실패 추적, 텍스트 비평 등은 접근 불가하다.

**배치 최적화 형식화**:

$$\mathcal{D}_t = \mathcal{D}_{t-1} \cup \{(x^{\text{new}}_{t,j}, y^{\text{new}}_{t,j})\}_{j=1}^{q}$$

- 총 평가 예산: $N = qT$ (배치 크기 $q$, 총 배치 수 $T$)
- 목표: 제한된 예산 내에서 $x^* = \arg\max_{(x,y) \in \mathcal{D}_{T-1}} y$ 를 찾는 것

**기술적 장애물**:
1. BO는 고정된 저차원 유클리드 공간에서 동작하지만, 시스템 프롬프트는 이산적(discrete), 가변 길이, 의미론적으로 구조화된 자연어 객체
2. 기성 밀집 텍스트 임베딩(예: Sentence-BERT)은 수천 차원으로 소규모 데이터에서 GP 사후 추정이 불가능
3. 연속 잠재 벡터에서 다시 텍스트로 디코딩하는 역매핑(inverse mapping) 문제

### 2.2 제안하는 방법: ReElicit

#### 전체 알고리즘 구조

ReElicit은 4단계 반복 루프로 구성된다:

**Phase 1: 특징 유도 및 임베딩 구축**

라운드 $t$ 에서 LLM은 $K$ 개의 독립적인 특징 집합 후보 $\mathcal{F}_t^{(k)}$ 를 생성하고, 각각에 대해 교차검증(CV) 오차를 계산한다:

$$g_t: \mathcal{X} \to [0, 1]^{d_t}$$

$$k^* = \arg\min_{k \in \mathcal{K}_t} \text{CV}(\mathcal{Z}_t^{(k)}, \mathcal{Y}_{t-1})$$

이때 $t > 1$ 이면 이전 라운드의 특징 집합 $\mathcal{F}_{t-1}$ (incumbent)도 후보로 포함하여 이전에 효과적이었던 표현이 사라지지 않도록 보장한다.

**중요한 정보 분리 원칙**:
- `DefineFeatures`: 프롬프트 + 점수 + 이전 특징 집합을 모두 관찰 → 성능 근거 기반으로 의미론적 축 제안
- `ExtractFeatures`: 프롬프트 + 특징 정의만 관찰, 점수 미노출 → 정보 누출(leakage) 방지

**Phase 2: 임베딩 공간에서의 베이지안 최적화**

선택된 임베딩 $(\mathcal{Z}\_t, \mathcal{Y}_{t-1})$ 에 GP 대리 모델을 적합:

$$\{z^{\text{new}}_{t,1}, \ldots, z^{\text{new}}_{t,q}\} = \arg\max_{z_1, \ldots, z_q \in [0,1]^{d_t}} \alpha(z_1, \ldots, z_q \mid \mathcal{M}_t)$$

구현에서는 **qLogNoisyExpectedImprovement** 획득 함수와 **Matérn 5/2 커널(ARD)** 을 사용하며, `optimize_acqf`로 20회 재시작 및 512개 원초 샘플을 통해 최적화한다.

**Phase 3: 특징 목표 벡터를 자연어 프롬프트로 실현**

BO가 선택한 목표 벡터 $z^{\text{new}}_{t,j}$ 는 실제 프롬프트가 아니다. LLM이 이를 실제 배포 가능한 시스템 프롬프트로 실현한다:

- **3a. 병렬 초기 생성**: $M_{\text{init}}$ 개의 후보를 병렬로 생성하고, 각 후보의 특징 벡터를 추출하여 목표 벡터와 $\ell_2$ 거리가 가장 작은 것을 선택

$$p^* = \arg\min_p \|z^{\text{new}}_{t,j} - z^{(p)}\|_2$$

- **3b. 순차적 특징 갭 정제**: 각 특징 $\ell$ 에 대한 갭 $\Delta_\ell = (z^{\text{new}}\_{t,j})\_\ell - (z_{\text{best}})\_\ell$ 을 계산하고, 갭이 큰 순서대로 `FeatureGuidedRefine`을 통해 반복적으로 개선. $\|z^{\text{new}}\_{t,j} - z_{\text{best}}\|_2 \leq \tau$ 이면 조기 종료.

**Phase 4: 평가 및 업데이트**

$$\mathcal{D}_t = \mathcal{D}_{t-1} \cup \{(x^{\text{new}}_{t,j}, y^{\text{new}}_{t,j})\}_{j=1}^{q}$$

새로운 프롬프트-점수 쌍이 다음 라운드의 특징 유도 및 GP 적합에 반영된다.

#### 선형 CKA (Cross-Iteration 표현 정렬 측정)

인접 라운드 간 표현의 적응도를 측정하기 위해 선형 CKA를 사용:

$$\text{CKA}(K, L) = \frac{\langle HKH, HLH \rangle_F}{\|HKH\|_F \|HLH\|_F}, \quad H = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$$

여기서 $K = ZZ^\top$, $L = Z'Z'^\top$ 는 선형 그람 행렬이다.

### 2.3 모델 구조

```
[프롬프트 점수 히스토리 D_{t-1}]
         ↓
[LLM: DefineFeatures] → K개의 특징 집합 F_t^(k) 후보 생성
         ↓
[LLM: ExtractFeatures] → 각 프롬프트를 [0,1]^d 벡터로 매핑 (점수 미노출)
         ↓
[CV 오차 기반 최적 F_t 선택] (incumbent 포함)
         ↓
[GP 대리 모델 적합: BoTorch SingleTaskGP, Matérn 5/2]
         ↓
[qLogNEI 획득 함수 최적화] → 목표 특징 벡터 z^new
         ↓
[LLM: InitialGenerate + FeatureGuidedRefine] → 배포 가능한 시스템 프롬프트 생성
         ↓
[목표 LLM 평가: f(x)] → 새로운 스칼라 점수
         ↓
[D_t 업데이트 → 다음 라운드]
```

**구현 세부사항**:
- 옵티마이저 LLM: Llama 3.3 70B Instruct (온도 0.7)
- 목표 LLM: Llama 3.1 8B Instruct (온도 0.0, 그리디 디코딩)
- GP 커널: Matérn 5/2 with ARD, Normalize 입력 변환, Standardize 출력 변환
- 특징 수 $d_t$: 평균 2~3개 (소규모 데이터에서 GP가 유효하게 작동하는 차원)
- LOO-CV (데이터 10개 미만) 또는 10-fold CV (이상) 사용

### 2.4 성능 향상

**Table 1** (주요 성능 비교, N=30 평가 예산):

| 방법 | GSM8K | Boolean Expr. | Disambig. QA | Snarks |
|---|---|---|---|---|
| **ReElicit** | **0.833 ± 0.006** | **0.768 ± 0.009** | **0.551 ± 0.006** | **0.796 ± 0.006** |
| APE | 0.830 ± 0.005 | 0.719 ± 0.013 | 0.514 ± 0.008 | 0.770 ± 0.008 |
| OPRO | 0.818 ± 0.008 | 0.650 ± 0.014 | 0.524 ± 0.008 | 0.783 ± 0.007 |
| PromptBreeder | 0.833 ± 0.005 | 0.643 ± 0.017 | 0.516 ± 0.009 | 0.802 ± 0.008 |
| TextGrad | 0.827 ± 0.008 | 0.669 ± 0.019 | 0.532 ± 0.009 | 0.791 ± 0.007 |

**Table 2** (쌍별 승리-무승부 비율):

| 방법 | 평균 승/무 비율 |
|---|---|
| **ReElicit** | **0.81** |
| OPRO | 0.58 |
| APE | 0.56 |
| TextGrad | 0.49 |
| PromptBreeder | 0.40 |

ReElicit은 **10개 모든 태스크에서 통계적으로 최선이거나 최선과 유의미한 차이 없음**.

**Ablation 결과** (Table 3):

| 변형 | 점수 차이 (vs ReElicit) | p값 |
|---|---|---|
| No Refinement | $-0.009 \pm 0.004$ | < 0.001 |
| No BO | $-0.007 \pm 0.003$ | < 0.001 |
| Static Features | $-0.003 \pm 0.004$ | 0.168 (비유의) |
| Independent Extraction | $-0.002 \pm 0.004$ | 0.329 (비유의) |

### 2.5 한계

1. **평가 범위의 제한**: 오프라인 벤치마크 정확도(GSM8K, MMLU, BBH)를 집계 피드백의 프록시로 사용. 실제 배포 목표(사용자 만족도, 안전사고율)와는 간극 존재.

2. **LLM 의존성**: 특징 유도의 품질이 옵티마이저 LLM의 능력에 크게 의존. LLM이 잘못된 특징 축을 제안하면 성능이 저하될 수 있음.

3. **추가 LLM 호출 비용**: ReElicit은 특징 유도, 추출, 실현, 정제에 추가적인 옵티마이저 측 LLM 호출이 필요하므로 총 LLM 호출 효율성은 베이스라인 대비 낮음.

4. **역매핑의 손실성(Lossiness)**: LLM이 연속 특징 벡터를 텍스트로 실현하는 과정이 손실적(lossy)이며, 이론적으로 완벽한 실현을 보장하지 않음.

5. **비교 범위**: APO/BO 관련 방법들(InstructZero, MIPRO, HbBoPs 등)은 소프트 프롬프트, 유한 후보 풀, 고정 임베딩 등 다른 인터페이스를 가정하므로, 본 논문의 실증 주장은 "집계 전용 하드 프롬프트" 설정 내에서만 유효.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 근거: 도달 가능성(Reachability) 분석

**설정**:

$\mathcal{X}$ 를 유한한 프롬프트 우주라 하고, 존재하지 않는(unknown) 오라클 임베딩:

$$g^*: \mathcal{X} \to \mathcal{Z} \subset \mathbb{R}^d$$

를 가정한다. 오라클 커널:

```math
k^*(x, x') = k_\mathcal{Z}(g^*(x), g^*(x'))
```

**Assumption 1** (RKHS 복잡도 경계):

$$f \in \mathcal{H}_{k^*}, \quad \|f\|_{k^*} \leq B$$

**Assumption 2** (특징 맵의 Lipschitz 연속성):

$$\|\phi(z) - \phi(z')\|_{\mathcal{H}_\mathcal{Z}} \leq L\|z - z'\|, \quad \forall z, z' \in \mathcal{Z}$$

**Assumption 3** (유도된 임베딩의 오차 경계):

$$\forall x \in \mathcal{X}, \quad \|g^*(x) - g_t(x)\| \leq \eta_t$$

**Lemma 1** (점별 표현 오차 경계):

$$|f(x) - f_t(x)| \leq BL\eta_t$$

**증명 스케치**: 오라클 가중치 $w^\* \in \mathcal{H}\_\mathcal{Z}$ 를 유도된 임베딩에 적용하는 구조적 대리 $\bar{f}\_t = \langle w^*, \phi(g_t(x)) \rangle_{\mathcal{H}_\mathcal{Z}}$ 를 구성하고, Cauchy-Schwarz 부등식과 Assumption 1~3을 적용:

```math
\sup_{x \in \mathcal{X}} |f(x) - f_t(x)| \leq \sup_{x \in \mathcal{X}} \|w^*\|_{\mathcal{H}_\mathcal{Z}} \|\phi(g^*(x)) - \phi(g_t(x))\|_{\mathcal{H}_\mathcal{Z}} \leq BL\eta_t
```

**Theorem 1** (주요 결과: 최적성 갭 경계):

$x_t$ 가 근사 목적함수 $f_t$ 에 대해 $\delta$-차선(suboptimal)이라고 하자:

$$\max_{x \in \mathcal{X}} f_t(x) - f_t(x_t) \leq \delta$$

그러면 $x_t$ 는 진짜 목적함수 $f$ 에 대해 $\epsilon$-차선이며:

$$f(x^*) - f(x_t) \leq \underbrace{\delta}_{\text{최적화 오차}} + \underbrace{2BL\eta_t}_{\text{표현 오차}}$$

**증명 개요**: 갭을 세 항으로 분해:

```math
f(x^*) - f(x_t) = \underbrace{(f(x^*) - f_t(x^*))}_{\leq BL\eta_t} + \underbrace{(f_t(x^*) - f_t(x_t))}_{\leq \delta} + \underbrace{(f_t(x_t) - f(x_t))}_{\leq BL\eta_t}
```

### 3.2 일반화 성능 향상 메커니즘

**동적 재유도(Re-elicitation)의 역할**:

Theorem 1에서 핵심은 $\eta_t$ 의 감소가 전체 경계를 개선한다는 것이다. 새로운 배치가 도착할수록:

$$\eta_1 \geq \eta_2 \geq \cdots \geq \eta_{T-1}$$

가 성립하기를 기대할 수 있다. LLM이 더 많은 프롬프트-점수 쌍을 관찰할수록, 어떤 의미론적 특성이 고성능과 저성능을 구분하는지 더 정확하게 식별할 수 있기 때문이다.

**실증적 근거** (Figure 2a):
- 동적 특징(Dynamic Features): 반복이 진행될수록 LOO-CV MSE 지속 감소 (평균 $\approx 0.0055$)
- 정적 특징(Static Features): 초기에는 유사하나 이후 감소 폭이 작음 (평균 $\approx 0.0075$)
- 기준선(no features): 지속적으로 높은 MSE

**특징 벡터 품질과 개선 확률 간 관계** (Figure 2b):

| $\ell_2$ 갭 구간 | 개선 확률 $P(\text{improvement})$ |
|---|---|
| Small [0.00, 0.06] | $\approx 0.53$ |
| Medium [0.06, 0.14] | $\approx 0.45$ |
| Large [0.14, 0.71] | $\approx 0.37$ |

Spearman $\rho = -0.122$ ($p = 2.0 \times 10^{-6}$) — 갭이 작을수록 개선 확률이 통계적으로 유의미하게 높음.

**특징 공간의 안정성과 적응성** (Figure 1):
- 반복 추출 노이즈: 특징값 표준편차 > 0.05인 비율이 3.8%에 불과 → 재현성 확보
- 교차 반복 CKA 평균 0.81 → 공간 보존되면서도 충분한 적응 가능

**특징 차원 수의 적응적 증가** (Figure 4):
선택된 특징 공간의 차원이 반복에 따라 평균 2개에서 3개로 완만하게 증가. 이는 데이터가 쌓일수록 LLM이 더 정교한 구분 축을 제안할 수 있음을 보여준다. 동시에 MSE는 감소하므로 표현의 품질이 향상됨을 의미한다.

### 3.3 일반화의 조건과 범위

Theorem 1이 제시하는 일반화 보장은 다음 조건 하에서 성립한다:

1. **목적함수의 평활성**: $B$와 $L$이 작을수록(즉, 객관 함수가 잠재 의미 공간에서 완만하게 변할수록) 표현 오차의 영향이 제한됨. 집계 메트릭(평균 성능)은 개별 예제보다 훨씬 평활하므로 이 조건에 부합.

2. **오라클 임베딩의 존재**: 실제 성능을 예측하는 저차원 의미론적 특징 공간이 존재한다는 가정. 논문은 "태스크 관련 변동이 소수의 의미론적 축에 집중된다"는 실용적 관찰로 이를 정당화.

3. **LLM의 특징 유도 품질**: 이론은 $\eta_t$를 추상적으로 가정하며, 실제 LLM이 얼마나 정확하게 오라클 임베딩을 근사하는지는 경험적으로만 검증 가능.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

**① 새로운 패러다임: LLM as 표현 공간 설계자**

ReElicit은 LLM을 단순한 텍스트 생성기나 프롬프트 제안자가 아닌, **적응형 의미론적 표현 공간의 동적 구축자**로 사용하는 새로운 패러다임을 제시한다. 이는 LLM의 세계 지식과 추론 능력을 탐색 공간 정의에 활용하는 방향으로 BO 연구의 지평을 넓힌다.

**② 집계 피드백 설정의 표준화**

실제 배포 환경(A/B 테스팅, 사용자 연구)에서는 집계 피드백이 일반적임에도, 기존 연구는 대부분 개별 예제 레이블을 가정했다. 본 논문이 이 설정을 명확히 형식화함으로써, 이 방향의 후속 연구를 위한 기준점(benchmark)을 제공한다.

**③ 구조화된 인공물(artifact) 최적화의 일반화**

논문이 Discussion에서 언급하듯, "embedding by elicitation" 패턴은 시스템 프롬프트를 넘어 다양한 도메인에 적용 가능하다:
- 이미지/오디오/비디오 생성 파라미터 (멀티모달 모델 활용)
- 에이전트 도구 사용 지침 최적화
- 평가 루브릭(evaluation rubric) 설계
- 코드 생성 지침 최적화 (Tomar et al., 2025와의 연계)

**④ BO와 LLM의 통합 연구 활성화**

MIPRO (Opsahl-Ong et al., 2024), BOPRO (Agarwal et al., 2025), HbBoPs (Schneider et al., 2024) 등과 함께 BO-LLM 통합 연구의 흐름을 강화한다. 특히 **동적 표현 학습**이라는 차별화된 요소는 고정 임베딩 기반 방법들의 한계를 보완한다.

**⑤ 안전-신뢰성 연구에의 함의**

시스템 프롬프트 최적화가 자동화됨으로써, 안전 관련 메트릭(가드레일, 거부율 등)의 체계적 최적화가 가능해진다. 반면, 잘못 설계된 목적함수에 대한 과최적화(over-optimization) 위험도 증가한다.

### 4.2 앞으로 연구 시 고려할 점

**① 실제 배포 환경에서의 검증**

현재 실험은 오프라인 벤치마크 정확도를 프록시로 사용한다. 실제 연구에서는:
- 온라인 A/B 테스트 환경에서의 검증 필요
- 시간에 따라 변하는 사용자 분포(non-stationarity) 처리 방법 연구
- 지연(delayed) 피드백 상황에서의 적응형 알고리즘 개발

**② LLM 특징 유도의 신뢰성 및 일관성**

LLM이 제안하는 특징 축의 품질 보장이 어렵다:
- **환각(hallucination)** 문제: LLM이 실제로 성능과 무관한 특징 축을 제안할 수 있음
- **다른 LLM 패밀리에서의 전이 가능성**: 현재는 Llama 3.3 70B만 사용했으므로, GPT-4, Claude, Gemini 등에서의 특징 유도 품질 차이 연구 필요
- **LLM 업데이트 민감성**: 옵티마이저 LLM이 업데이트되면 동일 태스크에서도 다른 특징 축이 제안될 수 있음

**③ 이론적 보완**

현재 Theorem 1의 한계:
- $\eta_t$ 의 실제 감소 속도에 대한 이론적 보장 없음
- 오라클 임베딩 $g^*$ 의 존재 가정이 실용적이지 않을 수 있음
- 유한한 텍스트 우주 $\mathcal{X}$ 가정이 실제 무한한 텍스트 공간에서는 성립 안 됨

후속 연구에서는 확률적 LLM 특징 유도 과정을 직접 모델링하는 이론적 프레임워크 개발이 필요하다.

**④ 다목적 및 제약 최적화 확장**

실제 배포에서는 단일 메트릭 최적화보다 여러 메트릭 간의 트레이드오프가 중요하다:
- 성능 vs. 안전성 (Pareto 프론트 탐색)
- 성능 vs. 비용 (응답 길이, API 토큰 비용)
- 다목적 BO (예: BoTorch의 qNEHVI)와 ReElicit 통합 연구

**⑤ 소규모 언어 모델로의 확장**

현재 옵티마이저가 70B 파라미터 모델이므로 계산 비용이 높다. 더 작은 LLM(예: 7B, 13B)이 효과적인 특징 유도자로 기능할 수 있는지 연구가 필요하다. 효율적인 특징 유도 능력의 최소 요건 규명이 중요한 연구 방향이 될 수 있다.

**⑥ 메타-학습 관점에서의 접근**

여러 태스크에 걸쳐 학습된 특징 유도 패턴을 공유(transfer)하는 메타-학습 프레임워크를 통해:
- 새로운 태스크에서의 콜드 스타트 문제 완화
- 태스크 유형별로 효과적인 특징 축의 사전 지식(prior) 구축

**⑦ 특징 다양성 vs. 예측력 트레이드오프**

현재 CV 오차 최소화로 특징 집합을 선택하지만, 예측력이 높은 특징이 탐색(exploration)에 부적합할 수 있다. 획득 함수와 특징 집합 선택을 결합한 통합 기준(unified criterion) 개발이 필요하다.

---

## 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 방법 | 피드백 유형 | 탐색 공간 | ReElicit과의 차이점 |
|---|---|---|---|---|---|
| **APE** (Zhou et al., 2022) | 2022 | LLM 샘플링 + 점수 기반 선택 | 집계 가능 | 하드 프롬프트 | 불확실성 모델링 없음, 비역사적 |
| **OPRO** (Yang et al., 2023) | 2023 | 점수 정렬 히스토리 조건부 LLM 생성 | 집계 | 하드 프롬프트 | 구조화된 BO 없음 |
| **PromptBreeder** (Fernando et al., 2023) | 2023 | 진화 알고리즘 (변이+재결합) | 집계 | 하드 프롬프트 | 의미론적 표현 공간 없음 |
| **TextGrad** (Yuksekgonul et al., 2024) | 2024 | 텍스트 기반 자동 미분 | 개별 예제 레이블 | 하드/소프트 | 집계 피드백 설정 미지원 |
| **InstructZero** (Chen et al., 2023) | 2023 | 소프트 프롬프트 BO | 집계 | 소프트 → 하드 | 소프트 프롬프트 의존, 인코더 필요 |
| **MIPRO** (Opsahl-Ong et al., 2024) | 2024 | 베이지안 서로게이트 + LM 프로그램 | 집계 | 명령+데모 쌍 | 유한 후보 풀, 고정 임베딩 |
| **HbBoPs** (Schneider et al., 2024) | 2024 | Hyperband + 구조적 딥커널 GP | 집계 | 고정 프롬프트 후보 | 고정 후보 풀 필요, 동적 표현 없음 |
| **BOPRO** (Agarwal et al., 2025) | 2025 | 고정 임베딩 공간에서 BO | 집계 | 언어 기반 해 | 고정 임베딩, 재유도 없음 |
| **ProTeGi** (Pryzant et al., 2023) | 2023 | 경사하강법 + 빔 서치 (텍스트) | 개별 레이블 | 하드 프롬프트 | 집계 피드백 설정 미지원 |
| **GEPA** (Agrawal et al., 2025) | 2025 | 반영적 프롬프트 진화 | 개별 레이블 | 하드 프롬프트 | 집계 피드백 설정 미지원 |
| **Local Latent Space BO** (Maus et al., 2022) | 2022 | 구조화 입력용 로컬 잠재공간 BO | — | 구조화 입력 | 보조 데이터 및 인코더-디코더 훈련 필요 |
| **BOSS** (Moss et al., 2020) | 2020 | 문자열 공간 BO (문자열 커널) | — | 문자열 | 텍스트 열거/샘플링 필요, 역매핑 없음 |
| **ReElicit** (본 논문, 2026) | 2026 | 동적 LLM 유도 임베딩 + GP BO | **집계만** | **하드 시스템 프롬프트** | 동적 표현, 역매핑 내장, 추가 데이터 불필요 |

**핵심 차별점 요약**:

1. **동적 표현 vs. 정적/고정 임베딩**: 기존 대부분의 BO 기반 방법들은 고정된 임베딩 공간이나 사전 훈련된 인코더를 사용. ReElicit은 최적화 과정에서 표현 공간 자체를 동적으로 적응.

2. **역매핑 내장**: BOSS, 잠재공간 BO 등은 연속 공간에서 텍스트로의 역매핑에 별도의 훈련된 디코더가 필요. ReElicit은 LLM의 텍스트 생성 능력으로 역매핑을 자연스럽게 해결.

3. **집계 피드백 특화**: TextGrad, ProTeGi, GEPA 등은 개별 예제 레이블 필요. ReElicit은 프롬프트당 단일 스칼라 점수만으로 동작.

4. **해석 가능성**: 유도된 특징 축이 자연어로 기술되어 최적화 과정이 인간에게 해석 가능. 기존 밀집 임베딩 기반 방법들은 이 해석 가능성 제공 불가.

---

## 참고자료

**주 논문**:
- Lin, Z. J., Letham, B., Dooley, S., Balandat, M., & Bakshy, E. (2026). *Embedding by Elicitation: Dynamic Representations for Bayesian Optimization of System Prompts*. arXiv:2605.19093v1 [cs.AI].

**논문 내 인용 주요 참고문헌**:
- Ramnath et al. (2025). *A systematic survey of automatic prompt optimization techniques*. EMNLP 2025.
- Yang et al. (2023). *Large language models as optimizers*. ICLR 2024. (OPRO)
- Zhou et al. (2022). *Large language models are human-level prompt engineers*. ICLR 2023. (APE)
- Fernando et al. (2023). *Promptbreeder: Self-referential self-improvement via prompt evolution*. arXiv:2309.16797. (PromptBreeder)
- Yuksekgonul et al. (2024). *Textgrad: Automatic "differentiation" via text*. arXiv:2406.07496.
- Pryzant et al. (2023). *Automatic prompt optimization with "gradient descent" and beam search*. EMNLP 2023. (ProTeGi)
- Balandat et al. (2020). *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*. NeurIPS 33. (BoTorch)
- Shahriari et al. (2015). *Taking the human out of the loop: A review of Bayesian optimization*. Proceedings of the IEEE, 104(1).
- Gómez-Bombarelli et al. (2018). *Automatic chemical design using a data-driven continuous representation of molecules*. ACS Central Science, 4(2).
- Maus et al. (2022). *Local latent space Bayesian optimization over structured inputs*. NeurIPS 35.
- Moss et al. (2020). *Boss: Bayesian optimization over string spaces*. NeurIPS 33.
- Opsahl-Ong et al. (2024). *Optimizing instructions and demonstrations for multi-stage language model programs*. EMNLP 2024. (MIPRO)
- Schneider et al. (2024). *Hyperband-based Bayesian optimization for black-box prompt selection*. arXiv:2412.07820.
- Agarwal et al. (2025). *Searching for optimal solutions with LLMs via Bayesian optimization*. ICLR 2025. (BOPRO)
- Chen et al. (2023). *InstructZero: Efficient instruction optimization for black-box large language models*. arXiv:2306.03082.
- Wang et al. (2016). *Bayesian optimization in a billion dimensions via random embeddings*. JAIR, 55.
- Kornblith et al. (2019). *Similarity of neural network representations revisited*. ICML 2019. (CKA)
- Cobbe et al. (2021). *Training verifiers to solve math word problems*. (GSM8K)
- Hendrycks et al. (2021). *Measuring massive multitask language understanding*. ICLR 2021. (MMLU)
- Suzgun et al. (2022). *Challenging big-bench tasks and whether chain-of-thought can solve them*. arXiv:2210.09261. (BBH)
- Agrawal et al. (2025). *GEPA: Reflective prompt evolution can outperform reinforcement learning*. arXiv:2507.19457.
- Wu et al. (2025). *LLM prompt duel optimizer: Efficient label-free prompt optimization*. arXiv:2510.13907.
- Yen et al. (2025). *Data mixture optimization: A multi-fidelity multi-scale Bayesian framework*. arXiv:2503.21023.
