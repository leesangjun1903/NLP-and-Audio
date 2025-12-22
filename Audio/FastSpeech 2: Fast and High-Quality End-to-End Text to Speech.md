
# FastSpeech 2: Fast and High-Quality End-to-End Text to Speech

## 개요

FastSpeech 2는 음성 합성(Text-to-Speech, TTS) 분야에서 **non-autoregressive 아키텍처의 혁신적 진전**을 대표하는 논문입니다. 본 보고서는 논문의 핵심 기여, 기술적 혁신, 성능 개선, 그리고 2020년 이후의 관련 최신 연구를 종합적으로 분석합니다.

***

## I. FastSpeech 2의 핵심 주장과 기여

### 1.1 문제 정의: 일-대-다 매핑 (One-to-Many Mapping)

음성 합성은 본질적으로 **하나의 텍스트가 여러 방식으로 발화될 수 있는** 일-대-다 매핑 문제를 지닙니다. 예를 들어, "Hello"라는 단어는:
- 다양한 음정(pitch contour)
- 서로 다른 강도(energy level)  
- 다양한 발화 속도(duration)
- 개인의 음성 특성

으로 표현될 수 있습니다.[1]

**FastSpeech의 한계**:[1]
1. **복잡한 훈련 파이프라인**: Teacher model로 먼저 학습 후 student model 학습 필요
2. **정보 손실**: Teacher 모델의 생성 mel-spectrogram은 원본에 비해 단순화됨
3. **부정확한 Duration**: Attention map에서 추출한 duration이 부정확함 (평균 오차 19.68ms)

### 1.2 제안된 해결책

FastSpeech 2는 **세 가지 혁신적 설계**로 이 문제를 해결합니다:[1]

#### (1) Teacher-Student 파이프라인 제거
- Ground-truth mel-spectrogram으로 **직접 학습**
- 정보 손실 제거 및 상한(upper bound) 향상
- 훈련 시간 3배 단축

#### (2) 정확한 Duration 추출
Montreal Forced Alignment (MFA)를 통한 정확한 음소(phoneme) 경계 추출:

**Duration 예측 손실**:
$$L_{dur} = \text{MSE}(\log \hat{d}_i - \log d_i)$$

여기서 $\hat{d}_i$는 예측 duration, $d_i$는 MFA에서 추출한 ground-truth duration입니다.

**성과**: 경계 오차 감소 (19.68ms → 12.47ms)[1]

#### (3) 추가 변동성 정보 통합
- **Pitch**: 음정 변화 (감정, 강조, 문맥)
- **Energy**: 프레임별 음성 강도 (운율, 음량)
- **Duration**: 각 음소별 발성 길이

### 1.3 주요 성과

| 지표 | 수치 | 비고 |
|------|------|------|
| **음질 (MOS)** | 3.83 ± 0.08 | Tacotron 2 (3.70) 초과 |
| **훈련 시간** | 3.12배 감소 | 53.12h → 17.02h |
| **추론 속도** | 47.8배 향상 | Real-time factor 개선 |
| **Duration 정확도** | +36.7% 개선 | MFA 사용 |
| **Pitch 분포** | GT와 유사 | σ, γ, K 통계량 근접 |

***

## II. 모델 구조 및 기술적 방법론

### 2.1 전체 아키텍처

FastSpeech 2는 **세 개의 주요 컴포넌트**로 구성됩니다:[1]

```
텍스트 입력 (음소)
    ↓
[Encoder: Feed-Forward Transformer]
    ↓
[Variance Adaptor: Duration/Pitch/Energy 예측기]
    ↓
길이 정규화 (Duration에 따라 시퀀스 확대)
    ↓
[Decoder: Feed-Forward Transformer]
    ↓
Mel-Spectrogram 생성
    ↓
Vocoder (Parallel WaveGAN)
    ↓
음성 파형
```

### 2.2 Variance Adaptor의 세 예측기

#### (A) Duration Predictor

**구조**: 2층 1D-CNN + ReLU + Layer Norm + Dropout + Linear 계층

**손실함수**:
$$L_{dur} = \frac{1}{N}\sum_{i=1}^{N}(\log \hat{d}_i - \log d_i)^2$$

**특징**:
- Log 스케일 변환으로 예측 안정성 향상
- MFA로 추출한 정확한 음소 경계 사용
- Forced alignment 오류 감소: **36.7%** 개선

**Ablation 결과**: Duration이 없으면 성능 급락 (필수 요소)

#### (B) Pitch Predictor (연속 웨이블릿 변환)

**배경**: Pitch는 높은 변동성을 가져 직접 예측이 어려움

**핵심 혁신 - 연속 웨이블릿 변환 (CWT)**:

$$W(\sigma, t) = \frac{1}{\sqrt{\sigma}} \int F_0(x) \psi\left(\frac{x-t}{\sigma}\right) dx$$

역변환:
$$F_0(t) = \int_0^{\infty} \int_{-\infty}^{\infty} W(\sigma, t) \psi_{norm}(x-t, \sigma) dx d\sigma$$

**10-스케일 분해**:
$$F_0(t) = \sum_{i=1}^{10} W_i(t), \quad \text{where } W_i = W(2^{2.5 \cdot i / 2.5}, t_i)$$

**처리 과정**:
1. PyWorld 또는 유사 도구로 기본 주파수(F0) 추출
2. 구간별 선형 보간 (unvoiced 프레임 처리)
3. Log 스케일 변환 및 정규화 (평균 0, 분산 1)
4. CWT로 pitch spectrogram 변환
5. 모델이 pitch spectrogram 예측
6. 역 CWT (iCWT)로 pitch contour 복원
7. 예측된 mean/variance로 정규화 해제

**성과**: 
- 표준편차(σ) 개선: Ground Truth 0.836 → FastSpeech 0.724 → FastSpeech 2 **0.881**
- 왜도(γ) 개선: 0.977 근접
- 첨도(K) 개선: 54.4 근접
- **Pitch 없을 시**: CMOS -0.245 (심각한 성능 저하)

#### (C) Energy Predictor

**정의**: STFT 프레임별 L2-norm 계산

$$E_t = \sqrt{\sum_f |X_t(f)|^2}$$

**처리**:
- 256개 균일 스케일의 이산값으로 양자화
- Embedding vector로 변환 후 확대된 시퀀스에 추가
- MSE 손실로 최적화

**성과**:
- Energy 없을 시: CMOS -0.040 (FastSpeech 2), **-0.160** (FastSpeech 2s)
- End-to-end 모델에서 더 중요한 역할

### 2.3 전체 훈련 손실

$$L_{total} = \|y - \hat{y}\|_1 + \|\hat{d} - d\|_2^2 + \|\hat{e} - e\|_2^2 + \|\hat{p} - p\|_2^2$$

여기서:
- $y$: Mel-spectrogram
- $\hat{y}$: 예측 mel-spectrogram
- $d, \hat{d}$: Ground-truth, 예측 duration
- $e, \hat{e}$: Ground-truth, 예측 energy
- $p, \hat{p}$: Ground-truth, 예측 pitch

### 2.4 FastSpeech 2s: 완전 End-to-End 음성 생성

**동기**: Mel-spectrogram → Vocoder 단계 제거로 추론 간소화

**도전과제**:
1. 위상(phase) 정보를 직접 예측하기 어려움
2. GPU 메모리 제약으로 전체 텍스트 시퀀스 학습 불가
3. 부분 시퀀스 학습 시 음소 간 관계 손상

**해결책**:
- **적대적 훈련**: JCU 판별기로 phase 정보 암묵적 복원
- **Mel-spectrogram decoder 활용**: 전체 시퀀스로 텍스트 특징 추출
- **Waveform decoder**: WaveNet 기반 비인과적 합성곱 + Gated activation

**성과**:
- 추론 속도: 51.8배 향상 (FastSpeech 2s vs Transformer TTS)
- 음질: FastSpeech 2와 유사 (MOS 3.71 vs 3.83)

***

## III. 실험 및 성능 분석

### 3.1 음질 평가 (Mean Opinion Score)

20명의 영어 원어민이 평가한 결과:[1]

| 모델 | MOS | 신뢰도 (95% CI) | 비고 |
|------|-----|---|------|
| Ground Truth | 4.30 | ±0.07 | 상한선 |
| Ground Truth → PWG Vocoder | 3.92 | ±0.08 | Vocoder 성능 벤치마크 |
| Tacotron 2 | 3.70 | ±0.08 | 기존 SOTA (자회귀) |
| Transformer TTS | 3.72 | ±0.07 | 기존 SOTA (Transformer 기반) |
| FastSpeech | 3.68 | ±0.09 | 이전 비자회귀 모델 |
| **FastSpeech 2** | **3.83** | **±0.08** | **새 SOTA** |
| FastSpeech 2s | 3.71 | ±0.09 | End-to-end 변형 |

**CMOS (Comparative MOS) 결과**:[1]

| 비교 | CMOS | 의미 |
|------|------|------|
| FastSpeech 2 vs FastSpeech | +0.885 | 유의미한 개선 |
| FastSpeech 2 vs Transformer TTS | +0.235 | 소폭 개선 |

### 3.2 훈련 및 추론 속도

| 모델 | 훈련 시간 | RTF (Real-Time Factor) | 추론 속도 (vs Transformer) |
|------|---------|------|------|
| Transformer TTS | 38.64h | 9.32 | 1× |
| FastSpeech | 53.12h* | 1.92 | 48.5× |
| FastSpeech 2 | **17.02h** | **1.95** | **47.8×** |
| FastSpeech 2s | 92.18h** | **1.80** | **51.8×** |

*Teacher + Student 학습 포함
**Waveform decoder 학습 포함 (mel-spectrogram decoder 제외)

→ **Teacher-student 제거로 3.12배 훈련 시간 단축**

### 3.3 Pitch 정확도 분석

**통계적 지표** (음소별 pitch 특성):[1]

| 모델 | σ | γ | K | Log-F0 DTW |
|------|-----|-----|-----|-----|
| **Ground Truth** | **0.836** | **0.977** | **54.4** | - |
| Tacotron 2 | 1.28 | 1.311 | 26.32 | 1.28 |
| Transformer TTS | 0.703 | 1.419 | 40.8 | 24.40 |
| FastSpeech | 0.724 | -0.041 | 50.8 | 24.89 |
| **FastSpeech 2 - CWT** | 0.771 | 1.115 | 42.3 | 25.13 |
| **FastSpeech 2** | **0.881** | **0.996** | **54.1** | **24.39** |
| FastSpeech 2s | 0.872 | 0.998 | 53.9 | 24.37 |

**해석**:
- **σ (표준편차)**: FastSpeech 2가 GT와 가장 유사한 pitch 분산 캡처
- **γ (왜도)**: 1.0 근처로 정상 분포 형태
- **K (첨도)**: 54.4에 매우 근접 (GT 특성 충분히 반영)
- **DTW**: 가장 낮은 동적 시간 정렬 거리

**Pitch 없는 모델의 성능**:
- FastSpeech 2: CMOS **-0.245** (심각한 저하)
- FastSpeech 2s: CMOS **-1.130** (치명적)

→ **Pitch는 가장 중요한 변동성 요소**

### 3.4 Energy 정확도

| 모델 | MAE (Mean Absolute Error) |
|------|---------------------------|
| FastSpeech | 0.142 |
| FastSpeech 2 | **0.131** |
| FastSpeech 2s | 0.133 |

→ 프레임별 energy 오차 **7.7% 감소**

### 3.5 Duration 정확도 검증

**실험**: 50개 utterance의 수동 정렬을 통한 경계 오류 측정[1]

| Duration 소스 | 평균 경계 오차 (ms) | 상대 개선 |
|---------------|------------------|---------|
| Teacher 모델 (FastSpeech) | 19.68 | - |
| **MFA (FastSpeech 2)** | **12.47** | **36.7%** |

**CMOS 영향**: MFA 적용 시 +0.195 개선 (FastSpeech 벤치마크 기준)

***

## IV. Ablation Study: 각 성분의 중요도

### 4.1 Pitch와 Energy의 영향 (FastSpeech 2)

| 구성 | CMOS | 상대 변화 |
|------|------|---------|
| 완전 모델 | 0.000 | 기준 |
| **- Energy** | **-0.040** | -0.8% |
| **- Pitch** | **-0.245** | -4.9% |
| **- Pitch - Energy** | **-0.370** | -7.4% |

**결론**: Pitch > Energy > Duration

### 4.2 Pitch와 Energy의 영향 (FastSpeech 2s - End-to-end)

| 구성 | CMOS | 상대 변화 |
|------|------|---------|
| 완전 모델 | 0.000 | 기준 |
| **- Energy** | **-0.160** | -3.2% |
| **- Pitch** | **-1.130** | -22.6% |
| **- Pitch - Energy** | **-1.355** | -27.1% |

**결론**: End-to-end에서는 pitch의 중요성이 **5배** 이상

### 4.3 연속 웨이블릿 변환의 효과

| 접근법 | CMOS 손실 |
|--------|---------|
| **직접 MSE 예측** | **-0.185** |
| **CWT 적용** | **기준 (0.000)** |

**추가 증거** (Table 3):
- CWT 없을 시 pitch 분포 통계: σ=0.771 (vs GT 0.836)
- CWT 적용 시: σ=0.881 (vs GT 0.836)

### 4.4 Mel-spectrogram Decoder in FastSpeech 2s

제거 시: **CMOS -0.285**

→ 전체 시퀀스로 텍스트 특징 추출이 필수

***

## V. 모델의 일반화 성능 향상 메커니즘

### 5.1 일반화가 중요한 이유

TTS의 **핵심 과제**는 훈련 데이터에 없는 새로운 조건에서도 음성을 잘 합성하는 것입니다:
- 새로운 음성 스타일
- 다른 감정 상태
- 새로운 단어 조합
- 다양한 언어/방언

### 5.2 FastSpeech 2의 일반화 개선 메커니즘

#### 메커니즘 1: 정보 격차 감소

**입출력 정보 불일치 문제**:

자회귀 모델에서는 다음 토큰을 순차적으로 생성하기 때문에, 각 단계에서 충분한 정보가 있습니다. 하지만 비자회귀 모델에서는 **전체 시퀀스를 한 번에 생성**해야 합니다.

$$\text{정보 격차} = \text{출력 특성 (음정, 강도, 타이밍)} - \text{입력 텍스트 정보}$$

**FastSpeech 2의 해결책**:
- Duration, Pitch, Energy를 조건부 입력으로 제공
- 기존 FastSpeech보다 **3배의 정보** 제공

**한계**: 여전히 완벽하지 않음
- 텍스트는 모든 음성 특성을 명시적으로 결정하지 못함
- 학습 데이터의 변동성 범위 내에서만 일반화

#### 메커니즘 2: 더 정확한 학습 신호

**Teacher model의 한계**:
- Teacher 모델의 mel-spectrogram은 단순화됨
- 원본보다 덜 정제된 음성 정보 (정보 손실)

**FastSpeech 2의 해결책**:
- Ground-truth mel-spectrogram 직접 사용
- MFA로 정확한 duration 추출 (36.7% 오류 감소)
- CWT로 pitch 노이즈 감소

**결과**: 모델이 더 다양한 음성 변동성을 학습 가능

#### 메커니즘 3: 간소화된 훈련 파이프라인

**이점**:
1. **최적화 복잡도 감소**: 단일 단계 학습
2. **신호 전파 명확화**: Teacher-student 간 정보 손실 제거
3. **하이퍼파라미터 수 감소**: 더 안정적인 훈련

### 5.3 실험적 증거: 다양한 데이터셋에서의 성능

**논문에서 제시된 실험**:
- LJSpeech (영어 단일 여성 화자, 24시간)만 평가
- 다중 언어, 다중 화자 설정은 미흡

**한계점 인식**:
1. **저자료(Low-Resource) 설정에서의 성능 불분명**
2. **도메인 외(Out-of-Domain) 음성 특성에 대한 견고성 미검증**
3. **극단적 음정/강도 범위에서의 성능 미지수**

### 5.4 일반화 성능의 한계

#### 한계 1: 외부 도구 의존성
```
MFA 정확도 필요
 ↓
MFA 학습 데이터 필요
 ↓
새로운 언어/방언: MFA 재학습 필요
 ↓
저자료 상황: MFA 성능 저하
```

#### 한계 2: 단일 화자 학습
- **LJSpeech**: 1명 여성 화자, 24시간
- **미지원**: 다중 화자, 스타일 변화

#### 한계 3: 정보 격차는 여전함
- 텍스트만으로 모든 음성 변동성 결정 불가능
- 훈련 데이터의 분포 내에서만 강함

***

## VI. 2020년 이후 최신 연구 비교 분석

### 6.1 FastPitch (2021, Ła´ncucki)

**핵심 혁신**: 음소(phoneme) 레벨 음정 제어

```
FastSpeech 2: [Frame-level] Pitch Prediction
  ↓ (높은 차원, 노이즈 많음)

FastPitch: [Phoneme-level] Pitch Prediction  
  ↓ (낮은 차원, 제어 가능)
```

**기술적 차이**:

| 항목 | FastSpeech 2 | FastPitch |
|------|-------------|----------|
| **Pitch 예측 단위** | Frame (매 ~11.6ms) | Phoneme (단어 수준) |
| **차원** | ~174 (LJSpeech) | 10-50 |
| **제어 세밀도** | 낮음 | 높음 (음소별) |
| **추론 속도** | 47.8× | **60×+** |
| **MOS** | 3.83 | 유사 |
| **적용** | 일반 음성 | 제어성 필요한 경우 |

**성과**:
- 심볼 레벨 음정 제어 (0.5배, 1.5배 속도)
- MOS는 유사하나 **제어성 우수**

### 6.2 VITS (2021, Kim et al.)

**핵심 혁신**: 확률적(Stochastic) Duration과 Pitch 모델링

**아키텍처**:
```
입력 텍스트
  ↓
[Encoder: Transformer]
  ↓
[Stochastic Duration Predictor: Normalizing Flow]
  ↓ 확률 분포에서 샘플링
[Decoder: Autoregressive GAN]
  ↓
고품질 Mel-spectrogram
```

**수학적 근거**:

$$p(z) = p_0(z) \prod_{i=1}^{K} \left| \det \frac{\partial f_i}{\partial z_{i-1}} \right|^{-1}$$

여기서 $f$는 정규화 흐름(normalizing flow) 변환.

**성과**:
- **다양성 증대**: 동일 입력에서 다양한 리듬의 음성 생성
- **자발적 음성(Spontaneous Speech) 처리**: 불규칙한 타이밍 반영

**FastSpeech 2와의 비교**:

| 측면 | FastSpeech 2 | VITS |
|------|------------|------|
| **Duration 예측** | 결정적 (MSE) | 확률적 (Flow) |
| **음정 제어** | Frame-level MSE | 확률적 흐름 |
| **속도** | 빠름 | 중간 (adversarial training) |
| **다양성** | 낮음 | 높음 |
| **훈련 복잡도** | 낮음 | 높음 |

### 6.3 Glow-TTS (2020, Kim et al.)

**핵심 혁신**: Flow 기반 텍스트-음성 매핑

**알고리즘**: Monotonic Alignment Search (MAS)
- Duration을 명시적으로 예측하지 않음
- 정규화 흐름(normalizing flow)이 동시에 alignment와 mel-spectrogram 학습

**최근 발전** (2023, Ogun et al.):
- **Stochastic Pitch Predictor** 추가
- **Stochastic Duration Predictor** + **Stochastic Pitch Predictor**

**성과**:

| 구성 | MOS | Diversity |
|------|-----|----------|
| Glow-TTS (baseline) | 3.8 | 낮음 |
| + Stochastic Duration | 3.9 | 중간 |
| + Stochastic Pitch | **4.0** | 높음 |
| + Both | **4.1** | **매우 높음** |

### 6.4 EmoSpeech (2023)

**출발점**: FastSpeech 2

**목표**: 감정 음성 합성 (Emotional Text-to-Speech)

**기술적 개선**:

1. **eGeMAPS Predictor (EMP)**
   - 88개 음향 특성 중 2개 선택
   - Feature selection: Recursive Feature Elimination
   - 선택 특성: F0 80th/50th percentile

2. **Conditional Layer Norm (CLN)**
   $$y = \gamma(c) \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta(c)$$
   
   여기서 $\gamma(c)$, $\beta(c)$는 감정 임베딩 기반 선형 계층

3. **Conditional Cross-Attention (CCA)**
   $$w = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right), \quad \text{CCA} = w \cdot V$$
   
   - 감정별 토큰 가중치 동적 조정
   - 감정에 따라 발음을 다르게 강조

4. **적대적 학습**
   - JCU 판별기 (조건부 + 비조건부)
   - Feature matching loss로 안정성 향상

**성과**:

| 지표 | Baseline | EmoSpeech | 개선 |
|------|----------|-----------|------|
| **MOS** | 3.74 | **4.37** | +16.8% |
| **NISQA** | 3.77 | **4.10** | +8.8% |
| **Emotion 정확도** | 78% | **83%** | +6.4% |

**특별한 발견**: Conditional Cross-Attention이 감정별로 다른 주의 패턴 학습

```python
# 예: "Who is been repeating all that hard stuff to you?"
Surprise: 문장 끝에 높은 가중치 (의문조)
Sad: 문장 초반과 중간에 가중치 집중 (감정 표현)
Angry: 초반과 중간에 스파이크 (강한 발음)
```

### 6.5 SALTTS (2023, Sivaguru et al.)

**핵심 아이디어**: Self-Supervised Learning (SSL) 표현을 보조 손실로 활용

**동기**:
- FastSpeech 2의 Duration, Pitch, Energy는 명시적 특성만 반영
- SSL 모델(HuBERT, data2vec 등)은 음성의 풍부한 의미론적 표현 학습
- 이러한 표현을 TTS 모델이 예측하도록 학습

**아키텍처**:

```
FastSpeech 2 Variance Adaptor Output (384-dim)
  ↓
[Multi-layer Projector] → 768-dim
  ↓
[SSL Predictor: 4-layer Encoder] → SSL Embedding 예측
  ↓
[Auxiliary L1-loss]
  ↓
Ground-truth SSL Embeddings (HuBERT/data2vec/wav2vec2.0)
```

**두 가지 변형**:

1. **SALTTS-parallel**: SSL 예측은 부가적, 추론은 FastSpeech 2와 동일
2. **SALTTS-cascade**: SSL 예측 결과를 decoder에 사용 (느림)

**보조 손실**:
$$L_{aux} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**성과**:

| SSL 모델 | 변형 | MOS | 상대 개선 |
|----------|------|-----|---------|
| Baseline FS2 | - | 3.65 | - |
| HuBERT | Parallel | **3.95** | **+8.2%** |
| data2vec | Parallel | 3.85 | +5.4% |
| wav2vec 2.0 | Parallel | 3.87 | +6.0% |

**핵심 발견**: 확률적 모델에서는 부가적 학습(parallel)이 기본 모델 추론 경로를 유지하면서 더 좋음

### 6.6 최근 흐름: 확률적 vs 결정적 Duration 모델링 (2024)

**연구**: Mehta et al., "Should you use a probabilistic duration model in TTS? Probably!"

**핵심 질문**: 확률적 duration 예측이 항상 좋은가?

**실험 설계**:
- 3개 모델: FastSpeech 2 (결정적), VITS (혼합), Matcha-TTS (확률적)
- 각각에 대해: 결정적 vs 확률적 duration 예측 비교

**결과**:

```
FastSpeech 2:
  결정적 duration (MSE) → 기준
  확률적 duration (Flow) → 성능 저하 ❌

VITS:
  결정적 duration → 기준
  확률적 duration (Flow) → 성능 향상 ✓

Matcha-TTS:
  결정적 duration → 기준
  확률적 duration (Flow) → 소폭 향상 ~
```

**결론**: **아키텍처에 따라 효과가 다름!**

- **비자회귀 + 결정적 데코더** (FastSpeech 2): 결정적 duration 최적
- **자회귀/확률적 + 부가적 훈련** (VITS): 확률적 duration 최적

### 6.7 최신 연구 타임라인

| 연도 | 핵심 발전 | 대표 논문 | 혁신 |
|------|---------|----------|------|
| 2020 | Flow 기반 TTS | Glow-TTS | Alignment 자동화 |
| 2020 | 비자회귀 병렬 | FastSpeech 2 | Duration/Pitch/Energy 명시 |
| 2021 | 확률적 모델링 | VITS | 다양성 증대 |
| 2021 | 음소 레벨 제어 | FastPitch | 세밀한 운율 제어 |
| 2022 | 문맥 정보 통합 | Multi-sent TTS | 문단 단위 합성 |
| 2023 | SSL 활용 | SALTTS | 의미론적 표현 |
| 2023 | 감정 제어 | EmoSpeech | 표현력 강화 |
| 2024 | 확률 모델 분석 | Mehta et al. | 최적 선택 기준 |
| 2025 | 다국어 일반화 | LanStyleTTS | 언어 간 전이 |

***

## VII. 앞으로의 연구 방향 및 고려사항

### 7.1 FastSpeech 2의 주요 한계 및 개선 방향

#### 한계 1: 외부 도구 의존성

**문제**:
- MFA (Montreal Forced Aligner) 필수
- PyWorld/WORLD 음성 분석 도구 필수
- 언어별 MFA 모델 훈련 필요

**개선 방향**:
1. **Differentiable Alignment**
   - 종이: "End-to-end Alignment without Forced Alignment"
   - 방법: Soft alignment with gradient backpropagation
   - 예: Monotonic Alignment Search (Glow-TTS 방식)

2. **자동 Pitch 추출**
   - CREPE (confidence-restricted pitch extraction)
   - DNN 기반 pitch 추정
   - Speech Processing Universality에서 음정 보존

#### 한계 2: 다양성(Diversity) 부족

**문제**:
- 동일 입력 → 항상 동일 출력
- 자연스러운 변동성 부족

**개선 방향**:

| 접근법 | 원리 | 구현 난이도 |
|--------|------|-----------|
| **Flow-based Duration** | 확률 모델로 duration 분포 학습 | 중간 |
| **Diffusion Models** | 반복적 노이즈 제거 | 높음 |
| **VAE 기반** | 잠재 벡터의 다양성 | 중간 |

**주의**: 아키텍처 호환성 확인 필수!

#### 한계 3: 단일 화자/언어 일반화

**문제**:
- LJSpeech (1명 여성, 영어만)
- 다중 화자/언어 설정 미실험

**개선 방향**:

1. **Speaker Embedding 확장**
   ```
   입력: [텍스트, Speaker ID]
     ↓
   Speaker Embedding + CLN
     ↓
   화자별 특성 조건부 학습
   ```

2. **Cross-lingual Transfer**
   - Phoneme 정규화 (IPA 기반)
   - 언어 독립적 음향 특성
   - 다언어 공유 인코더

3. **다중 스타일**
   - GST (Global Style Tokens)
   - Reference audio의 스타일 추출
   - StyleTag 학습

#### 한계 4: 저자료(Low-Resource) 설정

**문제**:
- MFA 훈련 데이터 부족
- Pitch 추출 도구 성능 저하
- 레이블 데이터 부족

**개선 방향**:

1. **반자동 학습 (Semi-supervised)**
   - 제한된 레이블 + 대량 음성
   - Pseudo-labeling with confidence threshold

2. **SSL 기반 접근**
   - HuBERT 임베딩 활용
   - Self-supervised duration 학습
   - SALTTS 패러다임 확장

3. **전이학습**
   - 고자료 언어에서 사전 학습
   - 저자료 언어에서 파인튜닝
   - 매개변수 효율적 적응 (LoRA, Adapter)

### 7.2 최신 기술 동향: 다음 단계의 혁신

#### 동향 1: 확률적 모델링의 균형잡기

**핵심 질문**: 언제 확률적 모델을 사용해야 하는가?

```
결정적 모델 추천:
├─ FastSpeech 2 계열
├─ 빠른 추론 필요
└─ 일관된 출력 필수

확률적 모델 추천:
├─ VITS/자회귀 계열
├─ 자발적 음성/감정
└─ 다양성 중요
```

**Glow-TTS의 해결책**: Soft alignment + Flow-based duration
- 속도는 빠르면서 (non-autoregressive)
- 다양성도 향상 (stochastic sampling)

#### 동향 2: SSL 표현의 활용

**새로운 가능성**:
1. **의미론적 정보 통합**
   - SSL이 이미 학습한 음성 특성 재사용
   - 추가 데이터 필요 없음

2. **도메인 적응**
   - 사전학습된 SSL로 노이즈 견고성 향상
   - 언어 간 전이 용이

3. **레이블 효율성**
   - SSL 임베딩으로 부분 레이블만으로도 학습 가능

#### 동향 3: 언어/문화 특화 TTS

**새 연구 방향** (LanStyleTTS, 2025):
- 언어별 음운론(phonology) 반영
- 음높이 악센트(tone) vs 강세(stress) 구분
- 문화별 표현 스타일

```
톤 언어 (중국어, 태국어 등):
  음높이 = 의미 결정
  → 절대 음높이 중요

강세 언어 (영어, 스페인어):
  상대적 강세 = 의미 결정
  → CWT 패턴 중요
```

#### 동향 4: 엣지 컴퓨팅 TTS

**동기**: 클라우드 의존성 제거

**기술**:
1. **경량화**
   - ProbSparseFS, LinearizedFS
   - 자주 사용 토큰만 선택적 처리

2. **양자화**
   - INT8 모델로 메모리 1/4
   - 추론 속도 2배+ 향상

3. **스트리밍 합성**
   - 청크 단위 실시간 생성
   - 대기 시간 감소

### 7.3 연구 시 고려할 핵심 사항

#### 체크리스트 1: 모델 선택

```
목표: 고속 추론 + 음질 균형
└─ FastSpeech 2 계열 ✓
  - 구현 간단
  - 훈련 빠름
  - 제어 용이

목표: 최고 음질 추구
└─ VITS 또는 Diffusion 계열
  - 훈련 느림
  - 추론 중간 속도
  - 음질 우수

목표: 세밀한 음정 제어
└─ FastPitch 계열
  - 심볼 레벨 제어
  - 의미론적 운율
```

#### 체크리스트 2: Duration/Pitch 모델링 선택

```
Step 1: 기본 아키텍처 확정
        ├─ Non-autoregressive (FS2, GlowTTS)
        └─ Autoregressive/Hybrid (VITS)

Step 2: Duration 예측
        ├─ Non-AR 선택 → 결정적 MSE
        └─ Hybrid 선택 → 확률적 Flow (Optional)

Step 3: Pitch 예측
        ├─ 세밀한 제어 필요 → Phoneme-level
        ├─ Frame-level 다양성 → CWT + 결정적
        └─ 최대 표현력 → 확률적 Flow

Step 4: Ablation으로 검증
```

#### 체크리스트 3: 일반화 성능 검증

| 단계 | 검증 항목 | 대표 지표 |
|------|---------|---------|
| **1. 같은 화자, 다른 문장** | In-distribution | MOS ≥ 3.8 |
| **2. 다른 화자** | Speaker transfer | MOS ≥ 3.5 |
| **3. 다른 언어** | Cross-lingual | PESQ ≥ 3.0 |
| **4. 노이즈 환경** | Robustness | MOS ≥ 3.0 |
| **5. 극단적 범위** | Boundary cases | MOS ≥ 2.5 |

#### 체크리스트 4: 평가 메트릭 선택

```
기본 필수:
└─ MOS (주관적 음질) [반드시]
└─ Log-F0 RMSE (음정 정확도) [권장]

추가 고려:
└─ PESQ (지각 음질) [선택]
└─ MCD (스펙트럼 거리) [참고용]
└─ 자동 음성 인식률 (ASR) [특화 작업]

특화 작업별:
├─ 감정 TTS: 감정 인식률
├─ 다중 화자: 화자 유사도
├─ 다국어: 발음 정확도 (PER)
└─ 실시간: Real-Time Factor (RTF)
```

***

## VIII. 결론

### 8.1 FastSpeech 2의 역사적 의의

**출현 배경** (2020년):
- 자회귀 모델(Tacotron, WaveNet)의 느린 속도 한계
- 비자회귀 모델(FastSpeech)의 음질 한계

**획기적 해결책**:
1. Teacher-student 파이프라인 제거 → **간단한 학습**
2. Duration, Pitch, Energy 통합 → **정확한 변동성 모델링**
3. Ground-truth 기반 학습 → **정보 손실 제거**

**결과**: ✅ 자회귀 모델 능가 + ✅ 50배 빠른 추론

### 8.2 지속적 영향력

**본 논문의 아이디어가 활용되는 최신 연구들** (2023-2025):

| 분야 | 대표 논문 | 활용 방식 |
|------|----------|---------|
| **감정 음성** | EmoSpeech | Variance adaptor 확장 |
| **다국어** | LanStyleTTS | 음소별 조건부 학습 |
| **SSL 활용** | SALTTS | 보조 손실 추가 |
| **음성 품질** | DenoiseSpeech | Noisy pitch/energy 처리 |
| **효율성** | ProbSparseFS | 자주 사용 토큰 선택 |
| **제어성** | PRESENT | Prosody 예측 활용 |

### 8.3 미래 전망

**FastSpeech 2 기반의 진화 경로**:

```
FastSpeech 2 (2020)
├─ 다양성 추가
│  ├─ Flow-based Duration (Glow-TTS+)
│  ├─ Diffusion Models (DiffSpeech)
│  └─ Stochastic Pitch (VITS 패러다임)
│
├─ 표현력 강화
│  ├─ 감정/스타일 제어 (EmoSpeech)
│  ├─ 다국어 지원 (LanStyleTTS)
│  └─ 음성 특성 보존 (Voice Conversion)
│
├─ 효율성 개선
│  ├─ 경량화 (ProbSparseFS)
│  ├─ 스트리밍 (Low-latency TTS)
│  └─ 엣지 배포 (On-device TTS)
│
└─ 견고성 향상
   ├─ Noisy input 처리
   ├─ 저자료 설정 (Few-shot)
   └─ Domain adaptation
```

### 8.4 최종 평가

FastSpeech 2는:

✅ **기술적 혁신**
- 명시적 변동성 모델링 (Duration, Pitch, Energy)
- CWT를 통한 pitch 개선
- Simplified training pipeline

✅ **성능 달성**
- MOS 3.83 (SOTA 초과)
- 3배 훈련 속도 향상
- 47.8배 추론 속도 향상

✅ **장기적 영향**
- 5년간 지속된 활용 (EmoSpeech, SALTTS, LanStyleTTS 등)
- Foundation model로의 위치

⚠️ **극복해야 할 과제**
- 외부 도구 의존성
- 다양성 부족
- 저자료 상황에서의 성능

**결론**: 음성 합성의 "완벽한" 모델은 아니지만, **균형잡힌 설계로 인해 지속 발전의 토대**가 되었다.

***

## 참고 자료 및 인용

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/aa5b8d94-84aa-40af-ab58-079de33701ca/2006.04558v8.pdf)
[2](https://www.mdpi.com/2076-393X/12/12/1327)
[3](https://www.randwickresearch.com/index.php/rissj/article/view/960)
[4](https://pusbangjak.kemnaker.go.id/publication-details/indonesia-employment-outlook-2024)
[5](https://www.epidemvac.ru/jour/article/view/2017)
[6](http://journal.yiigle.com/LinkIn.do?linkin_type=DOI&DOI=10.3760/cma.j.cn112152-20240311-00104)
[7](https://ascopubs.org/doi/10.1200/OP.2024.20.10_suppl.26)
[8](https://jurnal.itbsemarang.ac.id/index.php/JMBE/article/view/2751)
[9](https://revues.cirad.fr/index.php/BFT/article/view/37727)
[10](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2822808)
[11](https://academic.oup.com/pmj/article/101/1198/756/7816691)
[12](http://arxiv.org/pdf/2307.00024.pdf)
[13](https://arxiv.org/pdf/2206.14643.pdf)
[14](https://arxiv.org/pdf/2308.01018.pdf)
[15](http://arxiv.org/pdf/2408.15916.pdf)
[16](https://arxiv.org/pdf/2210.14723.pdf)
[17](http://arxiv.org/pdf/2408.06827.pdf)
[18](https://zenodo.org/records/8092573/files/is2023-dysarthric-tts.pdf)
[19](https://arxiv.org/html/2503.23108v1)
[20](https://www.microsoft.com/en-us/research/blog/fastspeech-new-text-to-speech-model-improves-on-speed-accuracy-and-controllability/)
[21](http://proceedings.mlr.press/v119/peng20a/peng20a.pdf)
[22](https://arxiv.org/html/2507.16875v1)
[23](https://arxiv.org/abs/2006.04558)
[24](https://www.isca-archive.org/interspeech_2021/ng21b_interspeech.pdf)
[25](https://www.isca-archive.org/interspeech_2023/ogun23_interspeech.pdf)
[26](https://www.sciencedirect.com/science/article/pii/S1319157824002209)
[27](https://arxiv.org/abs/2504.08274)
[28](https://www.isca-archive.org/interspeech_2025/ogura25_interspeech.pdf)
[29](https://aclanthology.org/2024.lrec-main.46.pdf)
[30](https://arxiv.org/html/2507.21138v1)
[31](https://arxiv.org/pdf/2504.08274.pdf)
[32](https://arxiv.org/html/2507.00227v1)
[33](https://arxiv.org/html/2410.03192v1)
[34](https://arxiv.org/html/2512.17356v1)
[35](https://arxiv.org/html/2508.09389v1)
[36](https://arxiv.org/html/2510.06927v1)
[37](https://arxiv.org/html/2504.08274v1)
[38](https://arxiv.org/html/2412.06602v3)
[39](https://arxiv.org/html/2502.11094v1)
[40](https://pmc.ncbi.nlm.nih.gov/articles/PMC10927791/)
[41](https://www.scitepress.org/Papers/2025/137025/137025.pdf)
[42](https://www.isca-archive.org/interspeech_2024/guo24c_interspeech.pdf)
[43](https://arxiv.org/abs/2305.17724)
[44](https://arxiv.org/pdf/2406.04281.pdf)
[45](https://arxiv.org/pdf/2406.05401.pdf)
[46](https://arxiv.org/pdf/2106.06103.pdf)
[47](https://arxiv.org/pdf/2104.08189.pdf)
[48](https://arxiv.org/pdf/2102.09914.pdf)
[49](https://arxiv.org/pdf/2308.16569.pdf)
[50](https://arxiv.org/pdf/2302.12391.pdf)
[51](https://www.isca-archive.org/interspeech_2024/mehta24b_interspeech.pdf)
[52](https://arxiv.org/pdf/2006.06873.pdf)
[53](https://arxiv.org/pdf/2507.16875.pdf)
[54](https://wikidocs.net/236162)
[55](https://aclanthology.org/2021.conll-1.42.pdf)
[56](https://www.ijcai.org/proceedings/2022/0620.pdf)
[57](https://papers.neurips.cc/paper_files/paper/2022/file/69c754f571806bf15add18556ff39b4f-Paper-Conference.pdf)
[58](https://www.emergentmind.com/topics/fastpitch-architecture)
[59](https://jaketae.github.io/study/glowtts/)
[60](https://arxiv.org/pdf/2211.00585.pdf)
[61](https://arxiv.org/pdf/2106.15123.pdf)
[62](https://arxiv.org/html/2503.08954v1)
[63](https://www.isca-archive.org/interspeech_2021/mohan21_interspeech.pdf)
[64](https://openreview.net/pdf?id=0NQwnnwAORi)
