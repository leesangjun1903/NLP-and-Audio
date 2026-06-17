# Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **"LLM은 고전적 최적화기의 대체재가 아니라 보완재로서 가장 효과적이다."**

구체적으로:
- **고정 탐색 공간(fixed search space)** 에서는 CMA-ES, TPE 등 고전적 HPO 알고리즘이 LLM 기반 방법을 일관적으로 능가한다.
- LLM이 소스 코드를 직접 편집하는 **비제약 코드 편집(unconstrained code editing)** 방식은 고전적 방법과의 격차를 좁히지만 닫지는 못한다.
- 두 방식의 장점을 결합한 **하이브리드 방법 Centaur**가 모든 방법 중 최고 성능을 달성한다.

### 주요 기여 (4가지)

| # | 기여 내용 |
|---|-----------|
| 1 | 9가지 HPO 방법(고전 4, LLM 기반 4, 하이브리드 1)을 동일한 24시간 GPU 예산으로 벤치마크 |
| 2 | 고정 탐색 공간에서 고전적 HPO가 LLM 에이전트를 능가함을 실증 |
| 3 | **Centaur** 제안: CMA-ES의 내부 상태를 LLM과 공유하는 하이브리드 최적화기 |
| 4 | 탐색 다양성, OOM 비율, 모델 스케일링(0.8B~frontier) 분석 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**autoresearch** (Karpathy, 2025)는 LLM 에이전트가 훈련 코드를 반복적으로 편집하여 소형 언어 모델(~50M 파라미터)을 개선할 수 있음을 보였습니다. 이로부터 두 가지 연구 질문이 제기됩니다:

1. 고전적 HPO 방법들은 이 태스크에서 어떤 성능을 보이는가?
2. LLM 기반 HPO 방법이 고전적 방법을 능가할 수 있는가?

**핵심 문제**:
- LLM은 도메인 지식을 보유하지만 **trial 간 최적화 상태 추적에 실패**함
- 고전적 방법은 최적화 지형을 학습하지만 **도메인 지식이 없음**
- OOM(Out-of-Memory) 실패 회피가 탐색 다양성보다 성능에 더 중요함

---

### 2.2 평가 태스크 및 메트릭

**태스크**: `nanochat` (Karpathy, 2025b) — GPT-2 규모의 소형 디코더-only 트랜스포머를 FineWeb 데이터셋으로 훈련

**평가 지표**: 검증 bits-per-byte ($\text{val bpb}$) 최소화

$$\text{val bpb} = \frac{\mathcal{L}_{\text{val}}}{\ln 2}$$

여기서 $\mathcal{L}_{\text{val}}$은 검증 손실(nats 단위)이며, 낮을수록 좋습니다.

**실험 조건**:
- 하드웨어: NVIDIA H200 GPU (141 GB HBM3e), trial당 5분
- 총 예산: 24시간, 3 seeds
- OOM 실패 패널티: $\text{val bpb} = 100.0$

---

### 2.3 탐색 공간: AST 자동 추출

**Abstract Syntax Tree(AST) 파싱**으로 `train.py`에서 14개 하이퍼파라미터를 자동 추출합니다.

$$\mathcal{S} = \{x_1, x_2, \ldots, x_{14}\} \in \mathbb{R}^{13} \times \mathcal{C}$$

여기서 $\mathcal{C} = \{\text{SSSL, SSLL, SLSL, LLLL, SSSS, LSSL}\}$는 범주형 WINDOW_PATTERN입니다.

주요 하이퍼파라미터 목록:

| HP | 타입 | 범위 | 로그 스케일 |
|----|------|------|-------------|
| DEPTH | int | 4–24 | - |
| ASPECT_RATIO | int | 32–128 | - |
| HEAD_DIM | int | 64–256 | ✓ |
| DEVICE_BATCH_SIZE | int | 32–256 | ✓ |
| TOTAL_BATCH_SIZE | int | 65,536–2,097,152 | ✓ |
| EMBEDDING_LR | float | 0.01–2.0 | ✓ |
| MATRIX_LR | float | 0.005–0.2 | ✓ |
| WEIGHT_DECAY | float | 0.0–0.5 | - |
| WINDOW_PATTERN | cat. | {SSSL, ...} | - |

---

### 2.4 비교 방법론 (9가지)

| 범주 | 방법 | 탐색 공간 | 설명 |
|------|------|-----------|------|
| **고전** | TPE | Fixed | Tree-structured Parzen Estimator |
| **고전** | CMA-ES | Fixed | Covariance Matrix Adaptation ES |
| **고전** | SMAC | Fixed | 랜덤 포레스트 서로게이트 |
| **고전** | Random | Fixed | 균등 랜덤 샘플링 |
| **LLM** | LLAMBO (Paper) | Fixed | LLM as Bayesian surrogate |
| **LLM** | LLAMBO (Optuna) | Fixed | OptunaHub 포트 버전 |
| **LLM** | Karpathy Agent (14 HPs) | Fixed | LLM이 JSON 설정 제안 |
| **LLM** | Karpathy Agent (Code) | Unconstrained | LLM이 소스코드 직접 편집 |
| **하이브리드** | **Centaur** | Fixed | CMA-ES 내부 상태 + LLM |

---

### 2.5 Centaur: 제안 방법

#### 핵심 아이디어

CMA-ES의 내부 상태 $(\boldsymbol{\mu}, \sigma, \mathbf{C})$를 LLM에 명시적으로 공유합니다.

#### CMA-ES 기본 원리

CMA-ES는 다변량 가우시안 분포에서 후보 설정을 샘플링합니다:

$$\mathbf{x}^{(t)} \sim \mathcal{N}\left(\boldsymbol{\mu}^{(t)},\; \left(\sigma^{(t)}\right)^2 \mathbf{C}^{(t)}\right)$$

여기서:
- $\boldsymbol{\mu}^{(t)} \in \mathbb{R}^{n}$: 평균 벡터 (현재 최선 추정 설정)
- $\sigma^{(t)} \in \mathbb{R}^{+}$: 전역 스텝 크기
- $\mathbf{C}^{(t)} \in \mathbb{R}^{n \times n}$: 공분산 행렬 (HP 간 상관관계)

CMA-ES는 각 세대에서 선택된 상위 $\mu$ 개체의 가중 평균으로 $\boldsymbol{\mu}$를 업데이트합니다:

$$\boldsymbol{\mu}^{(t+1)} = \sum_{i=1}^{\mu} w_i \mathbf{x}_{i:\lambda}^{(t)}$$

$$\sum_{i=1}^{\mu} w_i = 1, \quad w_1 \geq w_2 \geq \cdots \geq w_{\mu} > 0$$

#### Centaur 알고리즘

```
Algorithm 1: Centaur
Input: 탐색 공간 S, 예산 T, LLM 비율 r=0.3
CMA-ES 초기화, 히스토리 H ← ∅
for t = 1, ..., T do
    if LLM turn (확률 r로) then
        CMA-ES에서 μ, σ, C 추출
        x ← LLM(μ, σ, C, H, S)
    else
        x ← CMA-ES.Propose()
    y ← Evaluate(x)
    CMA-ES.Update(x, y)
    H ← H ∪ {(x, y)}
```

**핵심 동작 원리**:

$$\mathbf{x}^{(t)} = \begin{cases} \text{LLM}(\boldsymbol{\mu}^{(t)}, \sigma^{(t)}, \mathbf{C}^{(t)}, \mathcal{H}, \mathcal{S}) & \text{if } u \leq r,\; u \sim \mathcal{U}(0,1) \\ \text{CMA-ES.Propose}() & \text{otherwise} \end{cases}$$

모든 trial 결과가 CMA-ES 업데이트에 사용되므로, LLM이 제안을 override해도 CMA-ES는 전체 궤적에서 학습합니다.

**CMA-ES 내부 상태를 선택한 이유**:
- $\boldsymbol{\mu}$: 구체적인 설정값으로 해석 가능
- $\sigma$: 단일 스칼라로 수렴/탐색 상태 표현
- $\mathbf{C}$: 레이블된 행렬로 HP 간 상관관계 표현

반면 TPE의 밀도 추정기나 GP-BO의 고차원 사후 분포는 자연어로 요약하기 어렵습니다.

---

### 2.6 성능 결과

#### 최종 성능 비교 (val_bpb, 낮을수록 좋음)

| 방법 | Best val_bpb | OOM% | 표준편차 |
|------|-------------|------|---------|
| **Centaur [Opus 4.6]** | **0.9739** | 17% | ±0.0012 |
| Centaur [Qwen 0.8B] | 0.9766 | 13% | ±0.0008 |
| Centaur [Qwen 27B] | 0.9763 | 15% | ±0.0005 |
| TPE | 0.9768 | 11% | ±0.0019 |
| SMAC | 0.9778 | 36% | ±0.0020 |
| CMA-ES | 0.9785 | 16% | ±0.0036 |
| KA (Code) [Opus 4.6] | 0.9770 | 5% | ±0.0027 |
| KA (Code) [Qwen 27B] | 0.9814 | 12% | ±0.0046 |
| LLAMBO (Paper) | 0.9862 | 48% | ±0.0041 |
| Random | 0.9873 | 56% | ±0.0021 |
| KA (14 HPs) [Qwen 27B] | 0.9904 | 1% | ±0.0002 |

#### 핵심 발견사항

**① 고전적 방법 우위 (고정 탐색 공간)**
- CMA-ES, TPE가 고정 탐색 공간에서 LLM 기반 방법을 지속적으로 능가
- Karpathy Agent (Code)만이 고전적 방법에 근접하나 동일 성능 달성에 ~4배 더 긴 시간 소요

**② OOM 회피가 핵심**
- 상위 5개 방법의 OOM 비율: 모두 ≤ 20%
- 하위 4개 방법의 OOM 비율: 모두 ≥ 36%
- LLAMBO (Paper/Optuna)는 전체 trial 히스토리를 관찰하지만 OOM 비율이 랜덤 서치(56%)와 유사 → LLM의 상태 추적 실패를 시사

**③ LLM 스케일링 효과의 한계**
- 0.8B → 27B → Gemini Pro 스케일링: 하이브리드 방법에서는 유의미한 이점 없음
- 단, Centaur + Opus 4.6은 plateau를 돌파 (0.9739)

**④ Centaur의 분산 감소 효과**

$$\text{std(CMA-ES)} = 0.0036 \rightarrow \text{std(Centaur [27B])} = 0.0005$$

LLM이 도메인 지식을 주입함으로써 불리한 seed로 인한 표류(drift)를 방지합니다.

**⑤ LLM 비율 ablation**

| LLM 비율 $r$ | Centaur [0.8B] | Centaur [27B] |
|-------------|----------------|----------------|
| 0.1 | **0.9744** | 0.9758 |
| 0.3 (기본값) | 0.9766 | 0.9763 |
| 0.5 | 0.9768 | **0.9746** |
| 0.8 | 0.9849 | 0.9902 ← CMA-ES보다 나쁨 |

$r=0.8$에서 27B 모델은 CMA-ES 단독보다 성능이 저하됩니다. CMA-ES가 최적화 궤적의 다수를 통제해야 함을 확인합니다.

---

### 2.7 한계점

1. **단일 태스크 평가**: 소형 언어 모델 훈련 태스크에만 집중 → 결과의 일반화 가능성 불명확
2. **도구 없는 LLM**: 논문 검색, 문서 조회, 코드 분석 도구 없이 실험 → 에이전틱 LLM의 잠재력 미반영
3. **단순 하이브리드 구조**: CMA-ES 이외의 고전적 옵티마이저와의 결합 미탐색
4. **고정 검색 공간 범위**: 범위 설정에 여전히 일부 도메인 지식 필요
5. **추론 비용 제외**: LLM 추론 오버헤드를 wall-time에서 제외하여 공정성 논란 가능

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Centaur가 일반화에 기여하는 메커니즘

#### (A) 분산 감소를 통한 안정성 향상

Centaur는 CMA-ES의 seed 간 분산을 대폭 감소시킵니다:

$$\sigma_{\text{CMA-ES}}^2 = 0.0036^2 \quad \rightarrow \quad \sigma_{\text{Centaur}}^2 = 0.0005^2$$

이는 단순한 운 좋은 seed에 의존하지 않는 **재현 가능한 성능**을 의미하며, 실제 배포 환경에서의 일반화 안정성을 향상시킵니다.

#### (B) 도메인 지식 기반 제약 활용

Centaur의 정성적 분석(Appendix A.6)에서 LLM은 다음과 같은 **도메인 지식 기반 개선**을 수행했습니다:

- **어텐션 패턴 선택**: `LLLL → SSSS` (깊이 10에서 메모리 효율적인 local attention 선택)
  - CMA-ES는 스칼라 loss 값으로부터 어텐션 패턴의 의미를 학습 불가
- **하드웨어 친화적 반올림**: 배치 크기를 2의 거듭제곱으로 정렬 (61→64, 133,143→131,072)
  - GPU 텐서 코어 제약에 부합하는 설정 → 실제 하드웨어에서의 일반화
- **학습률 조정**: 좋은 설정들이 클러스터된 영역으로 이동

이러한 도메인 지식은 특정 태스크를 넘어 **트랜스포머 훈련 일반 원칙**에 근거하므로, 다른 태스크에도 일반화 가능성이 있습니다.

#### (C) 소형 LLM(0.8B)으로 충분함

$$\text{Centaur [0.8B]}: 0.9766 \approx \text{Centaur [27B]}: 0.9763$$

이는 고가의 대형 모델 없이도 하이브리드 접근법이 효과적임을 의미하며, **다양한 자원 환경에서의 적용 가능성**을 시사합니다.

#### (D) 탐색 다양성과 OOM 회피의 균형

Centaur는 적절한 탐색 다양성을 유지하면서도 낮은 OOM 비율을 달성합니다:

$$\text{Spread}_{\text{Centaur}} = 0.115 \sim 0.138, \quad \text{OOM}_{\text{Centaur}} = 13\% \sim 20\%$$

반면 LLAMBO는 높은 다양성(Spread = 0.252)에도 불구하고 높은 OOM 비율(48%)로 인해 일반화에 실패합니다. 이는 **실제 실행 가능한 설정의 탐색**이 일반화에 더 중요함을 보여줍니다.

### 3.2 일반화 성능 향상의 한계와 과제

**한계**:
- 현재 실험은 단일 모델 아키텍처(GPT-2 규모)에 한정
- 다른 도메인(이미지, 음성, 강화학습 등)으로의 일반화 검증 미실시
- Centaur + Opus 4.6의 plateau 돌파는 여전히 제한적 (0.9739 vs CMA-ES 최고 seed 0.9741)

**미탐색 영역**:
- 멀티 피델리티 접근법(Hyperband, BOHB)과의 결합
- 전이 학습을 통한 이전 태스크 지식 활용
- 코드 편집 LLM과 CMA-ES의 결합 → 탐색 공간이 최적화 궤적과 함께 진화

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 향후 연구에 미치는 영향

#### (A) LLM-HPO 패러다임의 재정립

이 논문은 LLM을 HPO의 독립적 최적화기로 사용하는 접근법이 현재 한계를 가지고 있음을 실증적으로 보여줍니다. 이는 향후 연구가 **"LLM 단독 vs 고전적 방법"의 경쟁 구도**에서 벗어나 **"LLM과 고전적 방법의 시너지 설계"** 방향으로 전환하는 데 영향을 미칠 것입니다.

#### (B) 해석 가능한 옵티마이저 상태 공유의 중요성

Centaur의 성공은 고전적 최적화기의 내부 상태를 **LLM이 이해할 수 있는 형태**로 공유하는 것이 핵심임을 보여줍니다. 이는 다음 연구 방향을 제시합니다:

- 다른 고전적 최적화기(SMAC, TPE, BOHB)와 LLM의 하이브리드 설계
- 최적화기 상태의 자연어 요약 기법 연구
- 새로운 "LLM-readable" 최적화기 설계

#### (C) AutoML의 인간 편향 감소

AST 파싱을 통한 자동 하이퍼파라미터 추출은 **탐색 공간의 수동 큐레이션을 자동화**하는 방향을 제시합니다. 이는 AutoML 시스템의 인간 편향(human prior) 감소에 기여합니다.

#### (D) 벤치마크로서의 autoresearch

autoresearch를 HPO 벤치마크로 사용하는 방법론을 제시함으로써, 실제 ML 파이프라인 최적화에 가까운 현실적 평가 환경을 제공합니다. 이는 향후 HPO 방법 비교 연구의 표준 벤치마크로 활용될 가능성이 있습니다.

---

### 4.2 향후 연구 시 고려사항

#### ① 다양한 도메인과 태스크로의 확장

현재 연구는 단일 태스크(소형 LM 훈련)에 국한됩니다. 향후 연구는:
- 이미지 분류, 객체 탐지, 강화학습 등 다양한 도메인 검증
- 대형 모델(수억~수십억 파라미터) 훈련 시의 확장성 검증
- 멀티태스크 HPO 환경에서의 Centaur 일반화 검증

#### ② 에이전틱 기능 통합

현재 모든 LLM 방법은 도구 없이 동작합니다. 향후 연구는:

$$\text{성능} \propto f(\text{LLM 능력}, \text{옵티마이저 상태}, \underbrace{\text{외부 지식}}_{\text{미탐색}})$$

- **논문 검색 도구**: 관련 연구에서 최적 하이퍼파라미터 범위 추출
- **문서 조회 도구**: 하드웨어 스펙, 라이브러리 API 참조
- **코드 분석 도구**: 정적/동적 분석을 통한 OOM 사전 예측

#### ③ 탐색 공간의 동적 진화

Centaur의 한계를 극복하기 위해:

$$\mathcal{S}^{(t+1)} = \mathcal{S}^{(t)} \cup \Delta\mathcal{S}_{\text{LLM}}^{(t)}$$

CMA-ES + 코드 편집 LLM의 결합으로 탐색 공간이 최적화 궤적과 함께 진화하는 구조 탐구가 필요합니다.

#### ④ LLM의 상태 추적 능력 개선

LLM이 trial 간 최적화 상태를 추적하지 못하는 근본 원인 분석 및 개선이 필요합니다:

- 구조화된 메모리 메커니즘 (외부 메모리, 벡터 데이터베이스)
- 최적화 궤적에 대한 fine-tuning
- Chain-of-thought를 통한 상태 추론 능력 향상

#### ⑤ 멀티 피델리티와의 결합

현재 연구는 단일 피델리티(5분 trial)만을 사용합니다. 향후:

$$\text{budget}(t) \propto \text{신뢰도}(\mathbf{x}^{(t)})$$

- BOHB, Hyperband와 Centaur의 결합
- LLM이 피델리티 수준도 결정하는 적응적 예산 할당

#### ⑥ 공정한 비교를 위한 방법론

- LLM 추론 오버헤드를 포함한 wall-time 비교 필요
- 더 많은 seed와 다양한 컴퓨팅 예산에서의 검증
- 통계적 유의성 검증 (현재 3 seeds만 사용)

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법론 | LLM 역할 | 옵티마이저 상태 공유 | 코드 편집 | 주요 차별점 |
|------|--------|----------|---------------------|-----------|-------------|
| **LLAMBO** (Liu et al., ICLR 2024) | LLM as Bayesian surrogate | 성능 예측 서로게이트 | ❌ | ❌ | GP 대신 LLM을 서로게이트로 활용 |
| **SLLMBO** (Mahammadli & Ertekin, 2024) | LLM + TPE 결합 | 공동 샘플러 | ❌ | ❌ | LLM과 TPE를 결합하나 내부 상태 미공유 |
| **LLM for HPO** (Zhang et al., 2023) | 직접 프롬프팅 | HP 제안 | ❌ | ❌ | 단순 프롬프팅 방식 |
| **LLaMA-ES** (Kramer, ESANN 2024) | LLM으로 CMA-ES HP 튜닝 | CMA-ES의 HP 최적화 | 부분적 | ❌ | CMA-ES를 최적화하지 않고 CMA-ES의 HP만 튜닝 |
| **HOLLM** (Schwanke et al., ICLR 2026) | 검색 공간 분할 + LLM | 서브영역 후보 제안 | ❌ | ❌ | 공간 분할로 LLM 제약, 상태 정보 미제공 |
| **AlphaEvolve** (Novikov et al., 2025) | 진화적 코드 생성 | 알고리즘 발견 | N/A | ✅ | HPO 특화가 아닌 범용 알고리즘 발견 |
| **autoresearch** (Karpathy, 2025) | LLM 코드 편집 | 전체 훈련 코드 편집 | N/A | ✅ | HPO 벤치마크의 기반이 된 원조 방법 |
| **Centaur** (본 논문, 2026) | CMA-ES + LLM 하이브리드 | 30% trial에서 CMA-ES 상태 기반 제안 | ✅ **완전 공유** | ❌ | CMA-ES의 $\boldsymbol{\mu}, \sigma, \mathbf{C}$ 전체 공유 |

### 핵심 차별점 분석

기존 방법들과 Centaur의 가장 큰 차별점은 **고전적 최적화기의 전체 내부 상태를 LLM에 명시적으로 공유**한다는 점입니다:

$$\text{Centaur}: \mathbf{x}_{\text{LLM}} = f(\underbrace{\boldsymbol{\mu}, \sigma, \mathbf{C}}_{\text{CMA-ES 상태}}, \underbrace{\mathcal{H}}_{\text{히스토리}}, \underbrace{\mathcal{S}}_{\text{탐색 공간}})$$

$$\text{LLAMBO}: \mathbf{x}_{\text{LLM}} = \arg\max_{\mathbf{x}} \alpha_{\text{LLM}}(\mathbf{x} | \mathcal{H})$$

$$\text{HOLLM}: \mathbf{x}_{\text{LLM}} = f(\mathbf{x} | \underbrace{\mathcal{R}_k}_{\text{서브영역 제약}}, \mathcal{H})$$

---

## 참고 자료

**주 논문**:
- Ferreira, F., Wobbe, L., Krishnakumar, A., Hutter, F., & Zela, A. (2026). *Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch*. arXiv:2603.24647v5.

**논문 내 인용 주요 참고문헌**:
- Karpathy, A. (2025a). *autoresearch*. https://github.com/karpathy/autoresearch
- Karpathy, A. (2025b). *nanochat*. https://github.com/karpathy/nanochat
- Hansen, N. (2016). *The CMA evolution strategy: A tutorial*. arXiv:1604.00772
- Bergstra, J., et al. (2011). *Algorithms for hyper-parameter optimization*. NeurIPS'11
- Liu, T., et al. (2024). *LLAMBO: Large language models to enhance Bayesian optimization*. ICLR 2024
- Hutter, F., et al. (2011). *Sequential model-based optimization for general algorithm configuration*. LION'11
- Falkner, S., et al. (2018). *BOHB: Robust and efficient Hyperparameter Optimization at scale*. ICML'18
- Schwanke, A., et al. (2026a). *Improving LLM-based global optimization with search space partitioning*. ICLR 2026
- Mahammadli, K., & Ertekin, S. (2024). *Sequential large language model-based hyper-parameter optimization*. arXiv:2410.20302
- Kramer, O. (2024). *LLaMA tunes CMA-ES*. ESANN 2024
- Zhang, M., et al. (2023). *Using large language models for hyperparameter optimization*. arXiv:2312.04528
- Novikov, A., et al. (2025). *AlphaEvolve: A coding agent for scientific and algorithmic discovery*. arXiv:2506.13131
- Bischl, B., et al. (2023). *Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges*. Wiley DMKD
- Feurer, M., & Hutter, F. (2019). *Hyperparameter Optimization*. In Automated Machine Learning. Springer
- Akiba, T., et al. (2019). *Optuna: A next-generation Hyperparameter Optimization framework*. KDD'19
- Qwen Team. (2026). *Qwen3.5: Towards native multimodal agents*. https://qwen.ai
- Google DeepMind. (2026). *Gemini 3.1 Pro Preview*. https://ai.google.dev
- Anthropic. (2026). *Claude Opus 4.6*. https://www.anthropic.com/claude/opus
