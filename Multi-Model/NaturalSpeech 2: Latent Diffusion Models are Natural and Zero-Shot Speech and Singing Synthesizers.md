# NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers
### I. 핵심 주장 및 주요 기여 개요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

NaturalSpeech 2는 텍스트-음성 합성(TTS) 분야의 근본적인 패러다임 전환을 제시한다. 기존 대규모 TTS 시스템들이 **이산 토큰(discrete tokens) + 자동회귀 모델(autoregressive models)** 조합을 사용하면서 발생하는 구조적 문제를 해결하기 위해, Microsoft Research Asia 팀은 **연속 벡터(continuous vectors) + 확산 모델(diffusion model)** 기반의 혁신적인 아키텍처를 제안했다.

논문의 세 가지 핵심 설계 결정은 다음과 같다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

1. **연속 벡터 표현**: 이산 토큰 기반 접근의 토큰 길이 증대 문제를 회피
2. **비자동회귀 확산 모델**: 오류 누적과 불안정성 제거
3. **음성 프롬팅 메커니즘**: 제로샷 능력 향상을 위한 인컨텍스트 학습

400M 파라미터와 44K시간의 다중 화자 음성 데이터(및 5K시간의 가창 데이터)로 학습된 NaturalSpeech 2는 기존 최고 성능 시스템들을 상당한 차이로 능가한다.

***

### II. 해결하고자 하는 문제의 상세 분석

#### 2.1 기존 TTS 시스템의 근본적 한계

대규모 다중 화자 TTS 시스템(예: AudioLM, VALL-E)들은 다음 구조를 채택한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

1. 음성 파형 → 신경망 코덱 → 이산 토큰 시퀀스
2. 이산 토큰 시퀀스 → 자동회귀 언어 모델 → 생성

이 파이프라인이 야기하는 **근본적 딜레마**는 표 1에서 명확히 드러난다:

| 구분 | VQ 기반 단일 토큰 | RVQ 기반 다중 토큰 |
|------|-----------------|-----------------|
| **파형 재구성** | 어려움 (정보손실) | 용이 (고충실도) |
| **토큰 생성** | 용이 (짧은 시퀀스) | 어려움 (긴 시퀀스) |

RVQ(Residual Vector Quantization)를 사용한 경우, R개의 잔여 양자화기를 활용하면 토큰 시퀀스가 R배 길어진다. 예를 들어 8개의 RVQ를 사용하면, 10초 음성이 수천 개의 토큰으로 변환되어 자동회귀 모델이 처리하기 극히 어려워진다. 이는 다음 문제들을 야기한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

- **Word skipping/repeating**: 자동회귀 생성 과정의 오류 누적
- **Prosody 불안정성**: 음성의 자연스러움 저하
- **Model collapse**: 심각한 경우 합성 실패

#### 2.2 자동회귀 모델의 구조적 취약점

자동회귀 모델은 시퀀스 길이와 오류 누적에 극히 민감하다. TTS의 특성상 음소(phoneme)와 음성 프레임 간 엄격한 단조성(monotonic alignment)이 필요하지만, 길어진 토큰 시퀀스에서 이 제약을 유지하기 어렵다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

***

### III. 제안 방법론 및 수식

#### 3.1 신경 오디오 코덱: 연속 벡터 표현 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

NaturalSpeech 2의 오디오 코덱은 다음 구조를 갖는다:

**인코더 → 잔여 벡터 양자화기 → 디코더**

수식으로 표현하면:

$$h = f_{enc}(x)$$

$$\{e^i_j\}^R_{j=1} = f_{rvq}(h^i), \quad z^i = \sum^R_{j=1} e^i_j, \quad z = \{z^i\}^n_{i=1}$$

$$x' = f_{dec}(z)$$

여기서:
- $x$: 음성 파형
- $h$: 인코더의 숨겨진 시퀀스 (프레임 길이 $n$)
- $z$: 양자화된 벡터 시퀀스
- $e^i_j$: $j$번째 잔여 양자화기의 임베딩 벡터
- $R$: 잔여 양자화기의 총 개수

**핵심 차이점**: 잔여 벡터들의 합 $\sum^R_{j=1} e^i_j$을 각 프레임의 **단일 연속 벡터**로 사용한다. 따라서 토큰 시퀀스 길이가 증가하지 않는다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

#### 3.2 잠재 확산 모델: 비자동회귀 생성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

확산 모델은 확률 미분방정식(SDE)으로 정식화된다:

**순전파(Forward SDE)**:

$$dz_t = -\frac{1}{2}\beta_t z_t dt + \sqrt{\beta_t} dw_t, \quad t \in $$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

여기서 $w_t$는 표준 브라운 운동이다.

해는 다음과 같이 주어진다:

$$z_t = e^{-\frac{1}{2}\int^t_0 \beta_s ds} z_0 + \int^t_0 \sqrt{\beta_s} e^{-\frac{1}{2}\int^s_t \beta_u du} dw_s$$

조건부 분포는 가우시안이다:

$$p(z_t | z_0) \sim \mathcal{N}(\rho(z_0, t), \Sigma_t)$$

여기서:
- $\rho(z_0, t) = e^{-\frac{1}{2}\int^t_0 \beta_s ds} z_0$
- $\Sigma_t = I - e^{-\int^t_0 \beta_s ds}$

**역전파(Reverse SDE)**:

$$dz_t = -\left(\frac{1}{2}z_t + \nabla \log p_t(z_t)\right) \beta_t dt + \sqrt{\beta_t} d\tilde{w}_t$$

혹은 **ODE 형식**:

$$dz_t = -\frac{1}{2}(z_t + \nabla \log p_t(z_t))\beta_t dt$$

**손실 함수**:

$$L_{diff} = \mathbb{E}_{z_0, t} \left[ ||\hat{z}_0 - z_0||^2_2 + ||\Sigma^{-1}_t(\rho(\hat{z}_0, t) - z_t) - \nabla \log p_t(z_t)||^2_2 + \lambda_{ce-rvq} L_{ce-rvq} \right]$$

여기서 신경망 $s_\theta(z_t, t, c)$ 는 점수 대신 데이터 $\hat{z}\_0$ 를 직접 예측한다. 세 번째 항 $L_{ce-rvq}$ 는 **RVQ 기반의 새로운 교차 엔트로피 손실**로, 각 잔여 양자화기 $j \in [1,R]$에 대해: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

$$\hat{z}_0 - \sum^{j-1}_{i=1} e^i$$

와 양자화 코드북의 임베딩들 간 L2 거리를 계산한 후, softmax를 통해 확률분포를 얻고 정답 코드와의 교차 엔트로피를 계산한다. 이는 정량화 정규화 역할을 한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

#### 3.3 선행 모델: 음소 인코더 및 지속시간/음고 예측기 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

텍스트 입력을 처리하기 위한 선행 모델 $c$는 다음 구성요소들로 이루어진다:

- **음소 인코더**: 6층 Transformer, 512 임베딩 차원
- **지속시간 예측기**: 30층 1D 합성곱 + 10 Q-K-V 주의층, L1 손실
- **음고 예측기**: 지속시간 예측기와 동일 구조, L1 손실

학습 시 정확한 지속시간/음고 정보를 사용하고, 추론 시 예측값을 사용한다.

**전체 손실 함수**:
$$L = L_{diff} + L_{dur} + L_{pitch}$$

#### 3.4 음성 프롬팅 메커니즘: 인컨텍스트 학습 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

제로샷 능력을 향상시키기 위해, 논문은 혁신적인 **음성 프롬팅 메커니즘**을 도입한다.

**학습 시**:
- 타겟 음성 잠재 시퀀스 $z = \{z^i\}^n_{i=1}$에서 임의로 세그먼트 $z_{u:v}$ (인덱스 $u$에서 $v$)를 선택하여 프롬프트로 사용
- 나머지 세그먼트 $z_{\backslash u:v} = z_{1:u} \cup z_{v:n}$을 학습 타겟으로 사용

**추론 시**:
- 목표 화자의 참조 음성을 프롬프트로 입력

**구현**:

1. **지속시간/음고 예측기에서**: 합성곱층에 Q-K-V 주의층 삽입
   - Query: 합성곱층의 숨겨진 시퀀스
   - Key/Value: 프롬프트 인코더의 출력

2. **확산 모델에서**: 두 단계 주의 구조
   - **첫 번째 주의**: $m$개의 임의 초기화된 쿼리 임베딩이 프롬프트 숨겨진 시퀀스에 주의 → 길이 $m$ 출력
   - **두 번째 주의**: WaveNet 층의 숨겨진 시퀀스가 첫 번째 주의 결과에 주의
   - 결과를 **FiLM(Feature-wise Linear Modulation)** 층의 입력으로 사용하여 WaveNet 출력을 조정

이 설계는 프롬프트 세부 정보가 과도하게 노출되지 않으면서도 화자 특성을 효과적으로 전달한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

***

### IV. 모델 구조의 상세 설명

#### 4.1 전체 시스템 아키텍처 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

**파이프라인**:
1. **학습**: 음성 파형 → 코덱 인코더 → 잠재 벡터 → 확산 모델 학습
2. **추론**: 텍스트/음소 → 확산 모델 → 잠재 벡터 → 코덱 디코더 → 음성 파형

#### 4.2 신경 오디오 코덱의 상세 구성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

- **인코더**: 여러 합성곱 블록, 총 다운샘플링 비율 200 (16kHz 입력)
  - 각 프레임 = 12.5ms 음성 세그먼트
- **잔여 벡터 양자화기**: R개의 코드북으로 순차적 양자화
- **디코더**: 인코더의 대칭 구조

| 모듈 | 구성 | 파라미터 수 |
|------|------|-----------|
| Audio Codec | 16 RVQ blocks, 256 dim | 27M |
| Phoneme Encoder | 6-layer Transformer, 512 hidden | 72M |
| Duration Predictor | 30-layer Conv1D + 10 attention | 34M |
| Pitch Predictor | 30-layer Conv1D + 10 attention | 50M |
| Speech Prompt Encoder | 6-layer Transformer | 69M |
| Diffusion Model | 40 WaveNet layers, 512 hidden | 183M |
| **Total** | | **435M** |

#### 4.3 WaveNet 기반 확산 모델 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

40개의 WaveNet 블록, 각 블록은:
- 1D 확장 합성곱 (커널 3, 확장 2)
- Q-K-V 주의층 (프롬프트로부터의 key/value)
- FiLM 층 (조건 정보 주입)

시간 단계 $t$와 조건 $c$를 기반으로 데이터 $\hat{z}_0$를 예측한다.

***

### V. 성능 향상 분석

#### 5.1 객관적 성능 평가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

| 평가 지표 | LibriSpeech | VCTK | 의의 |
|----------|-----------|------|------|
| **CMOS vs 그라운드 트루스** | +0.04 | -0.30 | 거의 동등한 자연스러움 |
| **CMOS vs YourTTS** | +0.65 | +0.58 | 기존 최고 성능 모델 능가 |
| **WER (LibriSpeech)** | 2.26% | 6.99% | 매우 낮은 오류율 |
| **Pitch 유사성 (프롬프트)** | 10.11 | 13.29 | 우수한 프로소디 모방 |
| **Duration 유사성** | 0.65 | 0.79 | 타이밍 정확성 |

#### 5.2 주관적 평가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

**SMOS (Speaker Similarity Mean Opinion Score)**:
- LibriSpeech: 3.28 (YourTTS 2.03 vs 그라운드 트루스 3.33)
- VCTK: 3.20 (YourTTS 2.43 vs 그라운드 트루스 3.86)

NaturalSpeech 2는 그라운드 트루스와 거의 동등한 화자 유사성을 달성한다.

#### 5.3 강건성 평가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

**50개의 어려운 문장에 대한 음성 지능 테스트**:

| 모델 | 반복 에러 | 스킵 에러 | 문장 오류 | 오류율 |
|------|---------|---------|---------|-------|
| Tacotron (AR) | 4 | 11 | 12 | 24% |
| Transformer TTS (AR) | 7 | 15 | 17 | 34% |
| FastSpeech (NAR) | 0 | 0 | 0 | 0% |
| NaturalSpeech (NAR) | 0 | 0 | 0 | 0% |
| **NaturalSpeech 2 (NAR)** | **0** | **0** | **0** | **0%** |

비자동회귀 모델(NAR)들은 자동회귀 모델의 고질적 문제인 단어 반복/스킵 문제를 완전히 해결했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

#### 5.4 VALL-E와의 직접 비교 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

| 지표 | VALL-E | NaturalSpeech 2 |
|------|--------|-----------------|
| **SMOS** | 3.53 | 3.83 (+0.30) |
| **CMOS** | -0.31 | 0.00 (+0.31) |

두 항목 모두에서 NaturalSpeech 2가 우수하며, 특히 화자 유사성과 자연스러움에서 유의미한 개선을 보인다.

#### 5.5 프롬프트 길이의 영향 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

| 프롬프트 | 음고 평균 | 음고 표준편차 | 지속 평균 | 지속 표준편차 |
|---------|---------|------------|---------|------------|
| **3s** | 10.11 | 6.18 | 0.65 | 0.70 |
| **5s** | 6.96 | 4.29 | 0.69 | 0.60 |
| **10s** | 6.90 | 4.03 | 0.62 | 0.45 |

프롬프트가 길수록 프로소디 유사성이 향상되므로, 더 많은 화자 정보가 포함된 프롬프트를 사용하면 성능 개선 가능성이 있다.

#### 5.6 절제 연구(Ablation Study) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

| 설정 | 음고 평균 | 결과 해석 |
|------|---------|---------|
| 전체 모델 | 10.11 | 기준 |
| w/o 확산 프롬프트 | 수렴 실패 | 확산 모델의 프롬프트 메커니즘 필수 |
| w/o 지속/음고 프롬프트 | 21.69 | (+113% 악화) 지속/음고 프롬프트 매우 중요 |
| w/o CE 손실 | 10.69 | (+5.7% 악화) RVQ CE 손실 효과적 |
| w/o 쿼리 주의 | 10.78 | (+6.6% 악화) 두 단계 주의 설계 유효 |

모든 설계 요소가 성능에 실질적 기여를 한다.

***

### VI. 모델 일반화 성능 향상의 핵심 메커니즘

#### 6.1 연속 벡터 표현의 이점 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

**문제**: RVQ 기반 이산 토큰의 경우, R개의 잔여 양자화기 사용 시 토큰 시퀀스가 R배 길어짐
- 예: 8개 RVQ 사용 시 10초 음성이 수천 개 토큰으로 변환
- 결과: 자동회귀 모델의 오류 누적 및 불안정성 악화

**해결**: 잔여 벡터들의 합을 **단일 연속 벡터**로 표현
$$z^i = \sum^R_{j=1} e^i_j$$

**이점**:
- 시퀀스 길이 증가 없음 (시간 복잡도 $O(n)$ 유지)
- 정보 손실 최소화 (모든 잔여 양자화기 정보 포함)
- 미분 가능하고 연속적인 표현 (최적화 용이)

#### 6.2 비자동회귀 확산 모델의 강점 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

**자동회귀 모델의 한계**:
- 좌-우 시퀀셜 생성으로 인한 오류 누적
- 긴 시퀀스에서 기울기 소실/폭발 문제
- 음소-음성 정렬의 엄격한 제약 위반 위험

**확산 모델의 장점**:
- **병렬 생성**: 모든 프레임을 동시에 처리 가능
- **오류 누적 없음**: 각 역전파 단계가 독립적
- **강건성**: 단조성 제약(monotonic alignment)을 명시적으로 강화 가능 (지속시간 예측기)

RVQ CE 손실 $L_{ce-rvq}$는 각 양자화 층의 정확도를 개별적으로 정규화하여, 확산 모델이 멀티스케일 정보를 균형있게 학습하도록 돕는다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

#### 6.3 음성 프롬팅 메커니즘의 일반화 능력 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

**핵심 설계**:
1. **학습 시**: 타겟 음성 내에서 임의 세그먼트를 프롬프트로 선택 → 다양한 프롬프트-타겟 조합 학습
2. **추론 시**: 미지의 화자 음성을 프롬프트로 사용 → 제로샷 능력

**일반화 메커니즘**:
- **다중 화자 학습**: 2,742명 남성 + 2,748명 여성 (총 5,490명)으로 학습 후, 테스트 시 완전히 새로운 화자에 대해 3초 음성만으로 합성
- **인컨텍스트 학습**: 프롬프트의 프로소디, 음색, 감정 정보를 동적으로 학습

두 단계 주의 구조(첫 번째: 추상화된 $m$개 쿼리, 두 번째: WaveNet 층의 세부 정보)는 **프롬프트의 필수 정보만 추출하면서 과도한 세부정보 노출을 방지**한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

#### 6.4 규모 확대(Scaling)의 효과 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

- **모델 파라미터**: 435M (분산 모델보다 크지 않음)
- **학습 데이터**: 44K 시간 (60K 시간 VALL-E와 비슷)
- **주목할 점**: 더 큰 모델 크기나 데이터양이 아닌, **구조적 설계**가 성능의 핵심

**성능 향상의 원인**:
- 연속 벡터 표현으로 인한 정보 밀도 향상
- 비자동회귀 생성의 강건성
- 다중 역 양자화 CE 손실을 통한 효율적 정규화

***

### VII. 한계 및 제약사항

#### 7.1 계산 효율성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

**문제**:
- 추론 시 150 단계의 확산 역전파 필요 (ODE/SDE 솔버 사용)
- VALL-E의 자동회귀 생성보다 느릴 수 있음

**향후 개선 방안** (논문에서 제시):
- **일관성 모델(Consistency Models)** 적용으로 단계 감소
- 향후 연구에서 검토 예정

#### 7.2 음질 한계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

**관찰**:
- VCTK에서 CMOS가 -0.30 (그라운드 트루스 대비)
- 이유: VCTK의 배경 노이즈 및 ASR 모델의 VCTK 미세조정 부족

#### 7.3 가창 합성의 제한 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

**현재 성능**:
- 약 30시간의 가창 데이터로 학습 (음성 데이터 44K시간과 비교)
- 음성 데이터와 혼합 학습 (업샘플링)

**문제**: 가창 데이터 부족이 성능 향상의 병목일 가능성

#### 7.4 윤리적 우려사항 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

논문은 명시적으로 지적한다:
> "NaturalSpeech 2는 화자 정체성을 유지한 음성을 합성할 수 있어, 음성 스푸핑이나 특정 화자 사칭 위험이 있다."

**권장 조치**:
- 사용자가 목표 화자로 사용될 것에 동의해야 함
- 합성 음성 탐지 모델 개발 필요

***

### VIII. 2020년 이후 관련 최신 연구 비교 분석

#### 8.1 주요 경쟁 모델들의 특성 비교

| 모델 | 발표 | 핵심 방식 | 데이터양 | 강점 | 약점 |
|------|------|---------|--------|------|------|
| **Glow-TTS** | 2020 | Flow-based + Monotonic Alignment | 소규모 | 병렬 생성, 고속 | 다중 화자 미지원 |
| **AudioLM** | 2022 | 이산 토큰 + 2단계 LM | 다양함 | 음악/음성 통합 | 토큰 길이 문제 |
| **NaturalSpeech** | 2022 | 신경 코덱 + Pitch predictor | 단일 화자 | 휴먼 레벨 품질 | 제로샷 미지원 |
| **VALL-E** | 2023.01 | 이산 토큰 + 자동회귀 LM | 60K 시간 | 강력한 제로샷, 감정 보존 | Word skipping, 느린 속도 |
| **AudioLDM** | 2023.01 | 텍스트-오디오 + Latent Diffusion | 다양함 | 범용 오디오 생성 | 음성에 최적화 X |
| **Voicebox** | 2023.06 | Flow-matching + infill | 50K 시간 | VALL-E보다 빠름, WER 1.9% | 공개 미제공 |
| **NaturalSpeech 2** | 2023.05 | 연속 벡터 + 확산 모델 | 44K 시간 | 강건성, 가창 합성, 음성 변환 | 추론 속도 (150 단계) |

#### 8.2 NaturalSpeech 2의 차별화 요소 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10842513/)

**구조적 우월성**:
1. **연속 벡터**: 이산 토큰의 길이 증대 문제 제거
   - VALL-E: 8-10개 RVQ 층으로 시퀀스 8-10배 증가
   - NaturalSpeech 2: 단일 연속 벡터로 길이 동일

2. **비자동회귀 생성**: 
   - VALL-E: 자동회귀로 인한 word skipping 24% 발생 가능
   - NaturalSpeech 2: 0% (50개 어려운 문장 테스트)

3. **통합 모델**:
   - VALL-E: 음성과 가창 별도 학습 필요
   - NaturalSpeech 2: 단일 모델로 음성+가창 합성

#### 8.3 VALL-E와의 상세 비교 [arxiv](http://arxiv.org/pdf/2406.05370.pdf)

**VALL-E의 설계**:

$$p(C(s') | y, C(s)) = \prod^T_{t=1} p(c'_t | c'_{<t}, y, C(s))$$

여기서:
- $C(s)$: 프롬프트 음성의 이산 코드
- $y$: 텍스트
- $c'_t$: 생성할 코드

**문제점**:
- 자동회귀이므로 오류 누적 위험
- 8-10개 코드층으로 인한 긴 시퀀스
- 강력한 인컨텍스트 학습은 가능하나 안정성 떨어짐

**NaturalSpeech 2의 개선**:
- 확산 모델의 역전파로 모든 프레임 동시 처리
- 단일 연속 벡터로 정보 밀도 향상
- 프롬팅 메커니즘으로 제로샷 능력 유지

**실제 성능 비교**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)
- SMOS: 3.83 vs 3.53 (+0.30)
- CMOS (vs 그라운드 트루스): 0.00 vs -0.31 (+0.31)

#### 8.4 Voicebox와의 비교 [arxiv](https://arxiv.org/abs/2306.15687)

**Voicebox (Meta, 2023)**:
- **생성 방식**: Flow-matching (확산 모델과 유사하나 다른 수학 프레임워크)
- **성능**: LibriSpeech WER 1.9% (NaturalSpeech 2: 2.26%)
- **속도**: VALL-E보다 20배 빠름 (50K+ 시간 데이터)
- **특징**: bidirectional context 지원 (음성 편집/infilling 가능)

**비교**:
- Voicebox가 더 빠르고 약간 더 낮은 WER
- NaturalSpeech 2는 가창 합성과 음성 변환에서 강점
- 둘 다 확산/flow 기반 비자동회귀 모델로 VALL-E보다 강건함

#### 8.5 AudioLDM과의 비교 [arxiv](https://arxiv.org/abs/2301.12503)

**AudioLDM**:
- **목표**: 범용 텍스트-오디오 생성 (음성, 음악, 효과음)
- **방식**: Latent Diffusion Model + CLAP 임베딩
- **데이터**: 다양한 오디오 (음성에 특화X)

**비교**:
- AudioLDM은 더 범용적이나 음성 합성에는 TTS 특화 모델이 우수
- NaturalSpeech 2는 음성 합성에 최적화

***

### IX. 향후 연구에 미치는 영향

#### 9.1 패러다임 전환 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

NaturalSpeech 2는 **TTS의 세 번째 패러다임**을 제시한다:

1. **첫 번째 (2016-2019)**: Tacotron, Transformer-TTS 등 자동회귀 모델
2. **두 번째 (2019-2022)**: FastSpeech, GlowTTS 등 비자동회귀 mel-spectrogram 모델
3. **세 번째 (2023-)**: **신경 코덱 + 비자동회귀 생성 모델** (VALL-E, NaturalSpeech 2, Voicebox)

#### 9.2 구조적 혁신의 영향 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

**1) 연속 벡터 표현의 확산**:
- VALL-E는 여전히 이산 토큰 사용으로 길이 증대 문제
- NaturalSpeech 2는 연속 벡터의 이점 입증
- 향후 다른 모달리티(음악, 음향 효과)에도 적용 가능

**2) 확산 모델의 강건성**:
- 자동회귀 모델의 고질적 word skipping/repeating 문제 해결
- 의료, 교육 등 강건성이 필수적인 분야에서 TTS 활용 확대 가능

**3) 멀티태스크 학습의 가능성**:
- 단일 모델로 TTS, 가창 합성, 음성 변환, 음성 향상 통합
- 모달리티 간 전이학습(transfer learning)의 근거 제시

#### 9.3 제로샷 학습의 한계 극복

**핵심 메커니즘 (Sound-based prompting)**:
- 기존: 스피커 임베딩(고정 차원) 또는 멜-스펙트로그램 기반
- NaturalSpeech 2: **raw 음성 세그먼트** → 프롬프트 인코더 → 동적 시퀀스

이 설계는 **다양한 프로소디, 감정, 음색 정보**를 더 자연스럽게 인코딩한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

#### 9.4 멀티모달 생성 모델의 기초 제공 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

음성과 가창의 통합 학습:
- **공통점**: 기본 메커니즘은 동일 (텍스트→생성)
- **차이점**: 가창은 음악적 구조(음정, 박자) 추가
- 단일 모델 통합은 향후 음악 생성, 음성-음악 하이브리드 모델로 확대 가능

#### 9.5 실제 응용의 확대

**즉시 적용 가능한 분야**:
1. **접근성**: 음성 장애인을 위한 음성 복구
2. **미디어**: 영화/게임의 다국어 음성 더빙
3. **교육**: 개인화된 음성 튜터링
4. **콘텐츠 크리에이션**: 1인 크리에이터의 음성 클로닝

***

### X. 향후 연구 시 고려할 점

#### 10.1 기술적 개선 방향

**1) 추론 속도 개선**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)
- **현재**: 150 단계의 ODE/SDE 솔버
- **향후**: Consistency Models 적용으로 단계 감소
- **목표**: VALL-E 수준의 실시간 성능 달성

$$\text{Consistency Model}: F(z_t, t) \approx z_0 \text{ (직접 예측)}$$

이는 역전파 횟수를 크게 줄일 수 있다.

**2) 멀티레벨 RVQ CE 손실 개선**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)
$$L_{ce-rvq} = \frac{1}{R} \sum^R_{j=1} CE(e^j, \text{softmax}(D(z^j_residual)))$$

각 RVQ 층의 가중치를 동적으로 조정하여, 층별 난이도 차이에 적응 가능.

**3) 프롬프트 길이 적응성**:
- 현재: 고정 길이 프롬프트에 최적화
- 향후: 가변 길이 프롬프트에 대한 강건성 강화
- 메커니즘: 프롬프트 길이에 따른 주의 가중치 동적 조정

#### 10.2 데이터 측면의 고려사항

**1) 가창 데이터 확대**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)
- 현재: 약 30시간 (음성 대비 0.7%)
- 문제: 데이터 부족이 가창 품질 제한
- 해결: 100K+ 시간의 가창 데이터 필요

**2) 다국어 확대**:
- NaturalSpeech 2: 영어만 학습
- VALL-E X 사례: 다국어 확산 모델로 확대 가능
- 언어 간 전이학습 가능성 탐색

**3) 다양한 도메인 데이터**:
- 배경 음성(노이즈)이 있는 in-the-wild 데이터
- 감정 표현이 풍부한 데이터
- 특수 목소리 (나이, 억양 등)

#### 10.3 평가 메트릭의 고도화

**현재 한계**:
1. **WER**: ASR 모델의 정확도에 의존
2. **CMOS/SMOS**: 평가자 수가 제한적 (12명, 6명)
3. **프로소디 유사성**: 통계적 지표(mean, std)만 사용

**개선 방향**:
1. **신경 메트릭**: 음성 유사성을 위한 사전학습 모델 (예: WavLM)
2. **세부 평가**: 음고 곡선, 톤(tone), 강세(stress) 패턴 비교
3. **다중 언어 평가자**: 언어별 음성 특성 반영

#### 10.4 윤리 및 안전성 고려사항 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

**음성 스푸핑 방지**:
- 합성 음성 탐지 모델 개발 (음성 위변조 탐지)
- 메타데이터 태깅 (합성 음성 표시)
- 규제 체계 정립

**사용자 동의 프레임워크**:
- 음성 사용 목적 명시
- 동의철회 메커니즘
- 음성 데이터 보호 정책

#### 10.5 모델 압축 및 경량화

**문제**: 435M 파라미터는 엣지 디바이스 배포에 부담

**해결 방안**:
1. **지식 증류(Knowledge Distillation)**: 큰 모델에서 작은 모델로 전이
2. **양자화(Quantization)**: FP32 → INT8로 감소
3. **프루닝(Pruning)**: 불필요한 파라미터 제거

목표: 50M 파라미터 수준으로 감소하면서 성능 70% 이상 유지

***

### XI. 결론 및 종합 평가

#### 11.1 NaturalSpeech 2의 학술적 기여 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

1. **구조적 혁신**: 연속 벡터 + 비자동회귀 확산 모델의 조합으로 TTS의 새로운 패러다임 제시
2. **기술적 우수성**: 강건성(0% word skipping), 자연스러움(CMOS 0.00), 화자 유사성(SMOS 3.28)에서 모두 우수
3. **다중 모달 확장**: 단일 모델로 음성 합성, 가창 합성, 음성 변환, 음성 향상 모두 지원
4. **제로샷 학습**: 3초 프롬프트만으로 미지의 화자 음성 합성 가능

#### 11.2 산업적 영향

**즉시 활용 가능**:
- Microsoft Azure Speech 서비스 통합
- 다국어 콘텐츠 제작 자동화
- 접근성 기술 (음성 장애인 지원)

**장기 잠재력**:
- 마이크로소프트의 생성 AI 생태계 강화
- OpenAI의 음성 모달리티 확대와 시너지

#### 11.3 한계와 추후 개선

| 영역 | 현재 한계 | 개선 방향 |
|------|---------|---------|
| **속도** | 150 단계 필요 | Consistency Models (단계 감소) |
| **음질** | VCTK CMOS -0.30 | 노이즈 강건성 개선 |
| **가창** | 30시간 데이터 | 100K+ 시간 데이터 확보 |
| **다국어** | 영어만 | VALL-E X 방식 적용 |
| **배포** | 435M 파라미터 | 지식 증류로 50M 목표 |

#### 11.4 최종 평가

**강점**:
- ✅ 강건한 비자동회귀 생성으로 word skipping 0% 달성
- ✅ 인간 수준의 자연스러움 (CMOS 거의 동등)
- ✅ 혁신적인 음성 프롬팅 메커니즘으로 제로샷 능력 극대화
- ✅ 단일 모델로 음성+가창 통합 지원
- ✅ 음성 변환, 음성 향상 등 다중 태스크 지원

**약점**:
- ⚠️ 추론 속도 (VALL-E 대비 느림)
- ⚠️ VCTK 데이터셋에서 약간의 성능 저하
- ⚠️ 가창 데이터 부족
- ⚠️ 435M 파라미터의 높은 계산 비용

**종합 평점**: **★★★★★ (5/5)**

NaturalSpeech 2는 TTS 분야의 획기적인 발전이며, 연속 벡터 + 확산 모델 조합의 우월성을 명확히 입증한다. 강건성, 자연스러움, 기능성 면에서 모두 최고 수준을 달성했으며, 향후 TTS 및 일반 음성 처리 연구의 중요한 기준점이 될 것으로 예상된다.

***

### 참고 자료

 Shen, K., Ju, Z., Tan, X., et al. (2023). NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers. arXiv:2304.09116v3. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c38b11af-eafb-4fc9-9a90-12ff0f292bad/2304.09116v3.pdf)

 Wang, C., Chen, S., Wu, Y., et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers. arXiv:2301.02111. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10447815/)

 Liu, H., Chen, Z., Yuan, Y., et al. (2023). AudioLDM: Text-to-Audio Generation with Latent Diffusion Models. arXiv:2301.12503. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10842513/)

 Kim, J., Kim, S., Kong, J., Yoon, S. (2020). Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search. NeurIPS 2020. [ojs.aaai](https://ojs.aaai.org/index.php/AAAI/article/view/29747)

 Song, Y., Sohl-Dickstein, J., Kingma, D. P., et al. (2020). Score-Based Generative Modeling through Stochastic Differential Equations. ICLR 2021. [arxiv](https://arxiv.org/abs/2306.02982)

 Ren, Y., Ruan, Y., Tan, X., et al. (2021). FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. ICLR 2021. [arxiv](https://arxiv.org/abs/2303.03926)
