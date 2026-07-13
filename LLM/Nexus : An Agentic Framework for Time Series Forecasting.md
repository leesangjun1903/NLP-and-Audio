# Nexus: An Agentic Framework for Time Series Forecasting 

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

Nexus는 시계열 예측이 단순한 수치 외삽(numerical extrapolation) 문제가 아닌, **구조화된 수치 데이터와 비구조화된 텍스트 문맥을 통합하는 에이전틱(agentic) 추론 문제**라는 점을 주장합니다.

기존 접근법의 한계를 명확히 지적합니다:

| 접근법 | 강점 | 약점 |
|---|---|---|
| TSFMs (TimesFM 등) | 수치 패턴 인식 우수 | 텍스트 문맥 활용 불가 |
| LLM 단독 사용 | 텍스트 추론 우수 | 시계열 특성 포착 부족 |
| **Nexus (제안)** | **수치 + 텍스트 통합** | **두 약점 동시 극복** |

### 1.2 주요 기여

1. **분해적 예측(Decomposed Forecasting)**: 거시(Macro) 수준의 추세와 미시(Micro) 수준의 세부 변동을 분리하여 모델링
2. **멀티에이전트 프레임워크 Nexus 도입**: 4개의 전문화된 에이전트로 구성된 완전 LLM 기반 예측 시스템
3. **도메인 적응형 캘리브레이션**: 과거 오류로부터 학습하는 백테스팅 루프 내장
4. **해석 가능한 추론 생성**: 예측값과 함께 자연어 추론 근거 제공
5. **데이터 누출 방지 평가**: LLM 지식 컷오프(2025년 1월) 이후 데이터만 사용

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

실세계 시계열은 종종 다음과 같은 도전 과제를 포함합니다:

- **구조적 단절(Structural Break)**: 예상치 못한 이벤트(팬데믹, 정책 변화 등)로 인한 패턴 변화
- **정권 전환(Regime Shift)**: 역사적 데이터만으로는 설명되지 않는 새로운 추세
- **멀티모달 신호**: 뉴스, 재무 보고서, 거시경제 요약 등 텍스트가 수치 변화를 주도

이를 해결하기 위해 Nexus는 다음의 공식화된 문제를 정의합니다.

### 2.2 문제 공식화 (수식 포함)

단변량 시계열 $\mathbf{X}\_{1:\tau} = (x_1, x_2, \ldots, x_\tau)$와 연관된 텍스트 시퀀스 $\mathbf{E}\_{1:\tau} = (e_1, e_2, \ldots, e_\tau)$가 주어졌을 때, 멀티모달 컨텍스트를 다음과 같이 정의합니다:

$$\mathcal{C}_{1:\tau} = (\mathbf{X}_{1:\tau}, \mathbf{E}_{1:\tau})$$

**목표**: 향후 $T$ 타임스텝의 예측값 $\mathbf{X}_{\tau+1:\tau+T}$와 자연어 추론 $\mathbf{R}$을 동시에 생성하는 매핑 $\mathcal{F}$를 학습:

$$\mathcal{F}(\mathbf{X}_{1:\tau}, \mathbf{E}_{1:\tau}) \rightarrow (\mathbf{X}_{\tau+1:\tau+T}, \mathbf{R})$$

여기서 $\mathbf{R}\_{\tau+1:\tau+T} = (r_{\tau+1}, \ldots, r_{\tau+T})$는 각 예측값의 인과적 설명입니다.

### 2.3 Nexus 프레임워크 구조

Nexus는 세 단계로 예측 과정을 분해합니다:

```
입력 (X_{1:τ}, E_{1:τ})
        ↓
[Stage 1: Contextualization]
  Historical Context Agent (A_ctx)
        ↓ H_{1:τ} (구조화된 인과 타임라인)
[Stage 2: Dual-Resolution Forecast]
  ┌─────────────────┐
  │ Macro Agent     │ → X^macro, R^macro
  │ (A_macro)       │   (전체 추세)
  └─────────────────┘
  ┌─────────────────┐
  │ Micro Agent     │ → X^micro, R^micro
  │ (A_micro)       │   (단계별 세부)
  └─────────────────┘
        ↓
[Stage 3: Synthesis & Calibration]
  Calibration Agent (A_calib) → G (마스터 가이드라인)
  Synthesizer Agent (A_syn)   → 최종 예측 + 추론
        ↓
출력 (X_{τ+1:τ+T}, R)
```

#### Stage 1: 컨텍스트화 (Contextualization)

**Historical Context Agent** $\mathcal{A}_{ctx}$:

$$\mathcal{A}_{ctx}(\mathbf{X}_{1:\tau}, \mathbf{E}_{1:\tau}) \rightarrow \mathbf{H}_{1:\tau}$$

- 각 타임스텝 $t$에서 수치 $x_t$와 텍스트 $e_t$를 결합
- 노이즈를 필터링하고 인과 관계가 명시된 구조화된 타임라인 $\mathbf{H}_{1:\tau}$ 생성
- **설계 동기**: LLM의 긴 컨텍스트 처리 시 발생하는 "인지 과부하(cognitive overload)" 방지 (Liu et al., 2024a의 "Lost in the Middle" 문제 대응)

#### Stage 2: 이중 해상도 예측 생성 (Dual-Resolution Forecast)

**Macro-Reasoning Agent** $\mathcal{A}_{macro}$:

$$\mathcal{A}_{macro}(\mathbf{H}_{1:\tau}) \rightarrow (\mathbf{X}^{macro}_{\tau+1:\tau+T}, \mathbf{R}^{macro})$$

- 전체 예측 구간 $T$에 대한 **하향식(top-down)** 거시 궤적 분석
- 계절성, 장기 추세, 체제 전환 등 포착

**Micro-Reasoning Agent** $\mathcal{A}_{micro}$:

$$\mathcal{A}_{micro}(\mathbf{H}_{1:\tau}) \rightarrow (\mathbf{X}^{micro}_{\tau+1:\tau+T}, \mathbf{R}^{micro}_{\tau+1:\tau+T})$$

- 각 타임스텝 $t \in [\tau+1, \tau+T]$에 대한 **상향식(bottom-up)** 세부 분석
- 즉각적 촉매, 단기 변동성, 국소적 이벤트 포착
- 각 스텝마다 $x^{micro}_t$와 $r^{micro}_t$ 개별 생성

#### Stage 3: 합성 및 캘리브레이션 (Synthesis & Calibration)

**Forecast Synthesizer Agent** $\mathcal{A}_{syn}$:

$$\mathcal{A}_{syn}(\mathbf{H}_{1:\tau}, \mathbf{X}^{macro}, \mathbf{R}^{macro}, \mathbf{X}^{micro}, \mathbf{R}^{micro}, \mathcal{G}) \rightarrow (\mathbf{X}_{\tau+1:\tau+T}, \mathbf{R})$$

- 거시/미시 관점을 동적으로 통합
- 캘리브레이션으로 학습된 가이드라인 $\mathcal{G}$를 조건으로 사용

**Calibration Agent** $\mathcal{A}_{calib}$:

역사 데이터를 $n$개의 순차적 백테스트 분할로 분리하고, 각 훈련 폴드 $i$에서 비평 규칙 $\mathcal{G}_i$를 생성합니다. 과적합 방지를 위해 교집합 연산으로 마스터 가이드라인을 도출:

$$\mathcal{G} = \bigcap_{i=1}^{n-1} \mathcal{G}_i$$

검증 폴드에서 최소 $k\%$ 성능 향상 시에만 최종 테스트에 적용하는 검증 패스를 수행합니다.

### 2.4 실험 설정

**평가 메트릭**:

$$\text{MAPE} = \frac{1}{T} \sum_{t=\tau+1}^{\tau+T} \left| \frac{x_t - \hat{x}_t}{x_t} \right| \times 100$$

$$\text{RMSE} = \sqrt{\frac{1}{T} \sum_{t=\tau+1}^{\tau+T} (x_t - \hat{x}_t)^2}$$

**데이터셋**:
- **Zillow 부동산 지표**: 15개 미국 대도시 주간 판매 재고, 2025년 2월~10월, 컨텍스트 3년
- **주식 시장**: AAPL, GOOGL, RKLB, JNJ, MSFT, NFLX, NVDA 주간 종가, 2025년 2월~12월, 컨텍스트 1년

### 2.5 성능 향상 결과

#### 멀티모달 컨텍스트 예측 (Table 2)

**Gemini-3.1-Pro 기준 Nexus vs. CoT Baseline**:

| 데이터셋 | MAPE 개선율 | RMSE 개선율 |
|---|---|---|
| Zillow | **↓14.7%** | **↓15.3%** |
| Stock | ↓1.2% | ↓1.7% |

**Claude-4.5-Sonnet 기준**:

| 데이터셋 | MAPE 개선율 | RMSE 개선율 |
|---|---|---|
| Zillow | **↓86.6%** | **↓88.6%** |
| Stock | ↓12.0% | ↓7.7% |

Claude-4.5-Sonnet의 CoT 베이스라인은 긴 컨텍스트 처리에서 심각한 성능 저하를 보였으며 (MAPE 0.2968 vs Nexus 0.0398), Nexus는 이를 효과적으로 해결했습니다.

#### 수치 전용 예측 (Table 3)

**Gemini-3.1-Pro 기준 Nexus vs. TimesFM-2.5**:

| 데이터셋 | Nexus MAPE | TimesFM-2.5 MAPE | 비고 |
|---|---|---|---|
| Zillow | 0.0378 | 0.0387 | Nexus **우수** |
| Stock | 0.1238 | 0.1294 | Nexus **우수** |

**주목할 점**: 텍스트 컨텍스트 없이 수치만으로도 Nexus가 전용 TSFM인 TimesFM-2.5와 동등하거나 우수한 성능을 달성했습니다.

#### 추론 품질 평가 (Table 4)

교차 판단(cross-judge) 방식으로 평가한 Overall Preference에서:
- **Zillow**: Nexus Win 97.1% (Gemini), 88.5% (Claude)
- **Stock**: Nexus Win 63.5% (Gemini), 79.8% (Claude)

#### 컴포넌트 분석 (Table 5)

| 모델 변형 | Zillow MAPE | Stock MAPE |
|---|---|---|
| Nexus (w/o Micro) | 0.0314 | 0.0877 |
| Nexus (w/o Macro) | 0.0317 | 0.0882 |
| Nexus (w/o Calibration) | 0.0309 | 0.0877 |
| **Nexus (Full)** | **0.0306** | **0.0866** |

세 컴포넌트 모두 최종 성능에 기여하며, 완전 파이프라인이 항상 최선임을 확인했습니다.

### 2.6 한계

1. **평가 범위 제한**: Zillow와 주식 데이터셋만 평가 (에너지, 의료, 기후 등 다른 도메인 미검증)
2. **페어링된 데이터 희소성**: 수치값-텍스트 쌍으로 구성된 공개 데이터셋 부족
3. **단일 실행 평가**: 수백억 파라미터 모델의 반복 호출에 따른 비용 문제로 통계적 분산 측정 불가
4. **간접적 데이터 누출 가능성**: LLM이 훈련 데이터에서 관련 정보를 간접적으로 습득했을 가능성 존재
5. **단변량 시계열만 다룸**: 다변량(multivariate) 설정에 대한 평가 부재
6. **계산 비용**: 멀티에이전트 시스템 특성상 단일 예측에 여러 LLM 호출 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 핵심 설계 요소

#### 3.1.1 캘리브레이션 루프의 교집합 기반 일반화

$$\mathcal{G} = \bigcap_{i=1}^{n-1} \mathcal{G}_i$$

단일 역사적 분할에서 도출된 규칙이 특정 시장 이상 현상에 **과적합**될 위험을 방지하기 위해, $n-1$개 훈련 폴드에서 공통적으로 유효한 규칙만을 교집합으로 유지합니다. 이는 통계적 **정규화(regularization)**와 유사한 역할을 수행합니다.

#### 3.1.2 검증 게이팅 메커니즘

가이드라인 $\mathcal{G}$는 숨겨진 검증 폴드에서 $k\%$ 이상의 성능 향상을 보일 때만 적용됩니다. 이는 **조기 중지(early stopping)**와 유사한 과적합 방지 기제로 작동합니다.

#### 3.1.3 모달리티 독립적 구조

Nexus의 에이전트 구조는 입력 도메인에 의존하지 않습니다:
- 텍스트 없이 수치만으로도 작동 (Section 4.3에서 검증)
- 다양한 도메인(부동산, 주식)에서 동일한 파이프라인으로 경쟁력 있는 성능 달성
- 이는 **도메인 전이(domain transfer)** 가능성을 시사합니다

#### 3.1.4 LLM 백본 교체 가능성

실험에서 Gemini-3.1-Pro와 Claude-4.5-Sonnet 모두 Nexus 파이프라인과 결합하여 효과적으로 작동했습니다. 이는 특정 LLM에 종속되지 않는 **모델 불가지론적(model-agnostic)** 설계임을 의미합니다.

### 3.2 일반화 성능의 증거

**수치 전용 설정에서의 경쟁력**: 텍스트 없이 수치만으로도 TimesFM-2.5(대규모 수치 데이터로 사전학습된 전용 TSFM)를 능가했다는 점은 Nexus의 추론 구조 자체가 범용적 시계열 패턴 인식 능력을 내재함을 보여줍니다.

**LLM 컨텍스트 길이 한계 극복**: Claude-4.5-Sonnet의 긴 컨텍스트 처리 약점을 Contextualization 단계에서 구조화된 요약으로 해결함으로써, 모델의 고유 한계를 아키텍처 수준에서 보완합니다.

### 3.3 일반화 성능 향상을 위한 추가 가능성

- **더 많은 백테스트 폴드($n$ 증가)**: 가이드라인의 일반성 향상
- **멀티도메인 캘리브레이션**: 여러 도메인의 가이드라인을 통합하여 보편적 규칙 추출
- **앙상블 가이드라인**: 교집합 대신 가중 다수결(weighted voting) 방식으로 더 유연한 일반화

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 시계열 기반 모델 계열

| 모델 | 연도 | 핵심 방법 | Nexus 대비 차이 |
|---|---|---|---|
| **Informer** (Zhou et al.) | 2021 | 효율적 Transformer, ProbSparse attention | 텍스트 미통합, 단일 모델 |
| **TimesFM** (Das et al.) | 2024 | 패치 기반 디코더-전용, 대규모 수치 사전학습 | 텍스트 미통합, 추론 미생성 |
| **Chronos** (Ansari et al.) | 2024 | 수치 이산화(quantization), 언어모델 구조 | 텍스트 미통합, 확률적 예측 |
| **MOMENT** (Goswami et al.) | 2024 | 마스크 재구성, 시계열 파일 사전학습 | 범용 표현 학습, 텍스트 미통합 |
| **MOIRAI** (Woo et al.) | 2024 | 교차-주파수, any-variate 모델링 | 범용 예측, 텍스트 미통합 |
| **Lag-Llama** (Rasul et al.) | 2023 | 지연 공변량, 디코더-전용 확률적 예측 | 확률적 예측, 텍스트 미통합 |

### 4.2 LLM 기반 시계열 계열

| 모델 | 연도 | 핵심 방법 | Nexus 대비 차이 |
|---|---|---|---|
| **GPT4TS/FPT** (Zhou et al.) | 2023 | 동결된 사전학습 LM + 경량 적응 | 파라미터 효율적, 텍스트 추론 미분리 |
| **Time-LLM** (Jin et al.) | 2024 | 시계열 패치를 텍스트 프로토타입으로 재프로그래밍 | 단일 모델, 멀티에이전트 미적용 |
| **UniTime** (Liu et al.) | 2024 | 언어 강화 크로스-도메인 예측 | 다변량, 멀티에이전트 미적용 |
| **TEMPO** (Cao et al.) | 2024 | STL 분해 + 프롬프트 기반 분포 적응 | 시계열 귀납적 편향 활용, 단일 모델 |
| **LLM Zero-Shot** (Gruver et al.) | 2023 | 수치를 문자열로 인코딩, 차세대 토큰 예측 | 단일 패스, 컨텍스트 통합 미흡 |

### 4.3 에이전틱/적응형 예측 계열

| 모델 | 연도 | 핵심 방법 | Nexus 대비 차이 |
|---|---|---|---|
| **Synapse** (Das et al.) | 2025 | 여러 TSFM 간 동적 중재(arbitration) | 수치 중심, 텍스트 추론 미통합 |
| **TimeCopilot** (Garza et al.) | 2025 | 특성 분석, 모델 선택 자동화 | 수치 워크플로우 중심 |
| **TimeSeriesScientist** (Zhao et al.) | 2025 | 범용 AI 에이전트, 다단계 계획 | 수치/도구 중심 |
| **AlphaCast** (Zhang et al.) | 2025 | 인간-LLM 협업 대화형 예측 | 인간 개입 필요 |
| **Cast-R1** (Tao et al.) | 2026 | 도구 강화 순차적 의사결정, RL | 강화학습 기반 |
| **TimeSAF** (Zhang et al.) | 2026 | 비동기 텍스트-시계열 융합 | 비동기 처리, 단일 패스 |
| **LoFT-LLM** (You et al.) | 2025 | 저주파 시계열 + LLM | 주파수 도메인 특화 |
| **Agentic TS Forecasting** (Cheng et al.) | 2026 | 인식-계획-반성-기억 에이전틱 워크플로우 | 포지션 페이퍼, Nexus와 방향 일치 |

### 4.4 Nexus의 차별점 요약

```
Nexus = 멀티에이전트 분해 + 텍스트-수치 통합 + 도메인 적응 캘리브레이션 + 해석 가능 추론
```

기존 에이전틱 시스템들이 주로 **수치 워크플로우 자동화**에 집중한 반면, Nexus는 **텍스트 증거와 시계열 추론을 예측의 핵심**으로 위치시킵니다. 또한 Tan et al. (2024)의 발견 — "LLM 컴포넌트를 제거해도 성능이 떨어지지 않는다" — 에 정면으로 반박하며, 올바른 구조화(disentanglement)를 통해 LLM의 본질적 예측 능력이 더 강함을 입증합니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### 5.1.1 패러다임 전환: 시퀀스 모델링 → 에이전틱 추론

Nexus는 시계열 예측을 **"시퀀스 패턴 인식 문제"**에서 **"증거 통합 기반 에이전틱 추론 문제"**로 재정의합니다. 이는 향후 연구의 방향을 다음과 같이 제시합니다:

- 예측 모델의 **설계 철학**: 단일 대형 모델 → 전문화된 협력 에이전트
- **평가 기준 확장**: 수치 정확도 → 수치 정확도 + 추론 품질 + 해석 가능성
- **벤치마크 재설계**: 지식 컷오프 이후 데이터를 사용한 데이터 누출 방지 평가 표준화

#### 5.1.2 멀티모달 시계열 연구 촉진

Nexus의 성공은 텍스트-수치 쌍 데이터셋 구축과 멀티모달 시계열 벤치마크(예: TFRBench, Context is Key)에 대한 관심을 높일 것입니다.

#### 5.1.3 LLM 내재적 예측 능력 재평가

"현재 세대 LLM은 이전에 인식된 것보다 더 강한 내재적 예측 능력을 가진다"는 주장은 LLM 기반 예측 연구의 재평가를 촉구합니다. 특히 Tan et al. (2024)의 회의적 시각에 대한 반론으로서 중요한 의미를 가집니다.

#### 5.1.4 해석 가능한 AI 예측 시스템

예측값과 함께 자연어 추론을 생성하는 접근법은 실제 의사결정 지원 시스템(금융, 부동산, 의료 등)에서 **신뢰성 있는 AI(Trustworthy AI)** 구현의 선례를 제공합니다.

### 5.2 앞으로 연구 시 고려할 점

#### 5.2.1 데이터 측면

- **더 다양한 도메인 검증 필요**: 에너지, 의료, 기후, 공급망 등 다양한 도메인에서의 일반화 검증
- **다변량 시계열 확장**: 여러 변수 간 상호작용을 포함하는 멀티변량 설정으로의 확장
- **더 긴 예측 구간**: 현재 최대 26주 → 1년 이상의 장기 예측 검증
- **비정기적 이벤트 처리**: 예측 불가능한 블랙스완 이벤트에 대한 강건성 연구

#### 5.2.2 모델 및 시스템 측면

- **계산 효율화**: 멀티에이전트 시스템의 추론 비용 절감 (증류, 경량화 에이전트 등)
- **통계적 분산 측정**: 단일 실행이 아닌 다중 실행을 통한 신뢰구간 및 분산 분석
- **에이전트 간 상호작용 최적화**: 현재는 직렬/병렬 고정 구조 → 동적 에이전트 협력 구조
- **온라인 학습 통합**: 실시간으로 새로운 데이터를 수신하며 캘리브레이션을 갱신하는 온라인 학습

#### 5.2.3 평가 측면

- **데이터 누출 방지 표준화**: 지식 컷오프 이후 데이터 사용 원칙을 LLM 기반 예측 연구의 표준으로 정착
- **추론 품질 메트릭 개발**: LLM-as-a-Judge를 넘어 객관적인 추론 품질 측정 방법론 개발
- **장기 인과 추론 검증**: 생성된 추론이 실제 인과 관계를 올바르게 반영하는지 검증

#### 5.2.4 실용성 측면

- **비용-성능 트레이드오프 분석**: 상용 LLM API 비용 대비 성능 향상의 ROI 분석
- **레이턴시 최적화**: 실시간 예측이 필요한 애플리케이션에서의 응답 시간 문제
- **도메인 전문가와의 협업**: 금융, 부동산 등 분야별 전문 지식을 캘리브레이션 가이드라인에 주입하는 방법

#### 5.2.5 이론적 측면

- **에이전트 분해의 최적성 이론**: 거시/미시 분해가 왜 효과적인지에 대한 이론적 근거 개발
- **캘리브레이션 수렴성 분석**: 교집합 기반 가이드라인이 최적 전략으로 수렴하는 조건 분석
- **LLM의 시계열 추론 메커니즘 해석**: 어텐션 패턴 분석 등을 통한 내부 작동 원리 이해

---

## 참고자료 (출처)

본 분석의 모든 내용은 다음의 논문 원문에 기반합니다:

**주 논문**:
- **Das, S. S. S. et al. (2026).** "Nexus: An Agentic Framework for Time Series Forecasting." *arXiv:2605.14389v1* [cs.AI], 14 May 2026.

**논문 내 인용 참고자료**:
- Ansari, A. F. et al. (2024). "Chronos: Learning the language of time series." *arXiv:2403.07815*
- Cao, D. et al. (2024). "TEMPO: Prompt-based generative pre-trained transformer for time series forecasting." *ICLR 2024*
- Cheng, M. et al. (2026). "Position: Beyond model-centric prediction – agentic time series forecasting." *arXiv:2602.01776*
- Das, A. et al. (2024). "A decoder-only foundation model for time-series forecasting." *ICML 2024*
- Das, S. S. S. et al. (2025). "Synapse: Adaptive arbitration of complementary expertise in time series foundational models." *arXiv:2511.05460*
- Garza, A. and Rosillo, R. (2025). "Timecopilot." *arXiv:2509.00616*
- Goswami, M. et al. (2024). "MOMENT: A family of open time-series foundation models." *ICML 2024*
- Gruver, N. et al. (2023). "Large language models are zero-shot time series forecasters." *NeurIPS 2023*
- Jin, M. et al. (2024). "Time-LLM: Time series forecasting by reprogramming large language models." *ICLR 2024*
- Kojima, T. et al. (2022). "Large language models are zero-shot reasoners." *NeurIPS 2022*
- Liu, N. F. et al. (2024a). "Lost in the middle: How language models use long contexts." *TACL*
- Liu, X. et al. (2024b). "UniTime: A language-empowered unified model for cross-domain time series forecasting." *WWW 2024*
- Rasul, K. et al. (2023). "Lag-Llama: Towards foundation models for probabilistic time series forecasting." *arXiv:2310.08278*
- Tan, M. et al. (2024). "Are language models actually useful for time series forecasting?" *NeurIPS 2024*
- Tao, X. et al. (2026). "Cast-R1: Learning tool-augmented sequential decision policies for time series forecasting." *arXiv:2602.13802*
- Williams, A. R. et al. (2025). "Context is key: A benchmark for forecasting with essential textual information." *ICML 2025*
- Woo, G. et al. (2024). "Unified training of universal time series forecasting transformers." *ICML 2024*
- You, J. et al. (2025). "LoFT-LLM: Low-frequency time-series forecasting with large language models." *arXiv:2512.20002*
- Zhang, F. et al. (2026). "TimeSAF: Towards LLM-guided semantic asynchronous fusion for time series forecasting." *arXiv:2604.12648*
- Zhang, X. et al. (2024). "Large language models for time series: A survey." *IJCAI 2024*
- Zhang, X. et al. (2025). "AlphaCast: A human wisdom-LLM intelligence co-reasoning framework." *arXiv:2511.08947*
- Zhao, H. et al. (2025). "TimeSeriesScientist: A general-purpose AI agent for time series analysis." *arXiv:2510.01538*
- Zhou, H. et al. (2021). "Informer: Beyond efficient transformer for long sequence time-series forecasting." *AAAI 2021*
- Zhou, T. et al. (2023). "One fits all: Power general time series analysis by pretrained LM." *NeurIPS 2023*
- Ahamed, M. A. et al. (2026). "TFRBench: A reasoning benchmark for evaluating forecasting systems." *arXiv:2604.05364*
- Anthropic. (2025). "System card: Claude Sonnet 4.5." Technical report.
- Google DeepMind. (2026). "Gemini 3.1 Pro model card." Technical report.
- Vodrahalli, K. et al. (2024). "Michelangelo: Long context evaluations beyond haystacks." *arXiv:2409.12640*
- Villalobos, P. et al. (2024). "Will we run out of data? Limits of LLM scaling based on human-generated data."
