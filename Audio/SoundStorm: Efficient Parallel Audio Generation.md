
# SoundStorm: Efficient Parallel Audio Generation

## 요약

"SoundStorm: Efficient Parallel Audio Generation"은 Google Research에서 발표한 혁신적인 오디오 생성 모델로, 비자동회귀(non-autoregressive) 병렬 디코딩 기법을 통해 고품질 오디오를 100배 빠르게 생성합니다. 본 논문은 신경 오디오 코덱의 계층적 토큰 구조를 활용한 아키텍처 설계와 신뢰도 기반 병렬 디코딩 방식을 제시하여, AudioLM과 동등한 음질을 유지하면서도 훨씬 뛰어난 음성 일관성과 스피커 보존을 달성합니다.[1]

***

## 1. 해결하고자 하는 문제

### 1.1 음성 생성의 근본적 도전

고품질 오디오 생성은 두 가지 상충하는 요구를 충족해야 합니다. 첫째, SoundStream과 EnCodec 같은 신경 코덱으로부터 파생된 토큰은 높은 비트레이트를 요구하므로, 고품질을 유지하려면 길고 복잡한 토큰 시퀀스가 필요합니다. 둘째, 트랜스포머 기반의 자동회귀 모델은 시퀀스 길이에 대한 이차 복잡도(O(n²))의 자기주의를 계산하므로, 긴 시퀀스 생성 시 계산 비용이 지수적으로 증가합니다.[1]

### 1.2 AudioLM의 한계

AudioLM은 세 단계의 계층적 자동회귀 구조를 사용하여 긴 기간 일관성과 고품질 음성을 동시에 달성했습니다. 그러나 이 접근법은 다음과 같은 문제를 야기합니다:[2][1]

- **속도 병목**: 30초 음성 생성에 수십 초 소요
- **장시간 생성의 어려움**: 10초 이상 생성 시 "slide-and-prompt" 방식 필요로 메모리 오버헤드 증가
- **일관성 저하**: 장시간 생성에서 음성 특성(speaker identity, acoustic conditions)의 표류(drift) 발생

### 1.3 기존 접근법의 한계

논문에서 제시한 세 가지 잠재적 해결책 각각의 한계:[1]
- 효율적 주의 메커니즘(efficient attention): 음성 생성에 충분하지 않음
- 병렬 디코딩 방식: 신경 코덱 토큰의 계층적 구조를 활용하지 못함
- 커스텀 아키텍처: 계층 구조의 특수성을 충분히 활용하지 못함

***

## 2. 제안하는 방법과 수식

### 2.1 모델 아키텍처

#### 입력 처리 메커니즘[1]

SoundStorm의 핵심은 Residual Vector Quantization(RVQ)의 계층적 구조를 명시적으로 활용하는 것입니다. 입력 프로세싱은 다음과 같이 작동합니다:

$$\text{Input}_{frame} = \sum_{q=1}^{Q} \text{Embed}(\text{Token}_{frame,q}) + \text{Embed}(\text{CondToken}_{frame})$$

여기서:
- $Q$: RVQ 레벨 수 (논문에서 12)
- $\text{Token}_{frame,q}$: 프레임 $f$의 RVQ 레벨 $q$의 토큰
- $\text{CondToken}_{frame}$: 조건부 신호(시맨틱 토큰)
- 동일 프레임의 임베딩을 합산하여 시퀀스 길이를 **RVQ 레벨과 무관하게** 프레임 수로 제한

이는 결정적 개선입니다. 전체 토큰 시퀀스 길이가:

$$L_{flattened} = T \times Q$$

에서:

$$L_{RVQ-aware} = T$$

로 축소됩니다. $T=1500$ (30초 @50fps), $Q=12$일 때, 시퀀스 길이는 18,000에서 1,500으로 **90% 축소**됩니다.[1]

#### Conformer 기반 모델 설계[1]

$$\text{Output}_{q} = \text{Head}_{q}(\text{Conformer}(\text{Input}_{frame}))$$

모델 명세:
- **층**: 12개 Conformer 블록
- **주의 헤드**: 16개
- **임베딩 및 모델 차원**: $d_{model} = 1024$
- **피드포워드 차원**: $d_{ff} = 4096$
- **위치 인코딩**: Rotary Position Embedding (RoPE)
- **파라미터**: 350M

**핵심 설계 결정**: 각 RVQ 레벨 $q$에 대해 별도의 출력 헤드를 사용하여, 동일한 인코더가 모든 레벨의 다양한 의존성 구조를 포착하도록 설계했습니다.

### 2.2 마스킹 스킴

SoundStorm의 마스킹 전략은 추론 절차와 일치하도록 설계되었습니다.[1]

#### 학습 중 마스킹 절차

$$\text{Masking Protocol}:$$

1. 프롬프트 경계선 샘플링:

$$t \sim \mathcal{U}\{0, T-1\}$$

2. 현재 RVQ 레벨 샘플링:


$$q^{*} \sim \mathcal{U}\{1, Q\}$$

3. 코사인 스케줄을 따른 마스킹 비율:

$$p = \cos(u), \quad u \sim \mathcal{U}[0, \pi/2]$$

각 위치 $i$에서:

$$M_i \sim \text{Bernoulli}(p)$$

4. 마스크 적용:

$$\text{Masked Token}_{i,q^*} = \begin{cases}
\text{[MASK]} & \text{if } i > t \text{ and } M_i = 1 \\
\text{Token}_{i,q^*} & \text{otherwise}
\end{cases}$$

5. **중요**: 레벨 $q^*$보다 미세한 모든 레벨의 토큰 마스킹:

$$\text{Masked Token}_{i,q} = \text{[MASK]} \quad \forall q > q^*, \forall i > t$$

#### 손실 함수

손실은 **마스크된 레벨 $q^*$의 토큰에만** 계산됩니다:

$$\mathcal{L} = \sum_{i=1}^{T} M_i \cdot \text{CrossEntropy}(\text{Logits}_{i,q^*}, \text{Target}_{i,q^*})$$

이 설계는 coarse-to-fine 학습을 강제하며, 동시에 더 미세한 레벨의 조건부 독립성을 활용합니다.[1]

### 2.3 신뢰도 기반 병렬 디코딩[1]

#### 추론 절차

초기 상태에서 모든 토큰이 마스크된 상태로 시작됩니다:
$$\text{State}_{0} = \text{[MASK]}^{T \times Q}$$

각 RVQ 레벨 $q$에 대해 순차적으로 처리:

**Level $q$에서의 반복적 정제**:

각 반복 $i \in \{1, ..., I_q\}$에서:

1. 모델 순전파:

$$\text{Logits}_{i,q} = \text{Model}(\text{State}_{i-1})$$

2. 신뢰도 계산:

$$\text{Confidence}_{j,q} = \max_{k \in [0,|\mathcal{V}|)} \text{Softmax}(\text{Logits}_{j,q})_k$$

3. 신뢰도 기반 선택:

$$J_i = \text{TopK}(\text{Confidence}_{j,q}, p_i)$$

여기서 $p_i$는 코사인 스케줄을 따릅니다:

$$p_i = \cos\left(\frac{i-1}{I_q-1} \cdot \frac{\pi}{2}\right)$$

4. 상태 업데이트:

$$\text{State}_{i, j, q} = \begin{cases}
\text{Prediction}_{j,q} & \text{if } j \in J_i \\
\text{[MASK]} & \text{otherwise}
\end{cases}$$

5. **마지막 반복에서는 탐욕적 디코딩**:

$$\text{Final}_{j,q} = \arg\max_{k} \text{Softmax}(\text{Logits}_{j,q})_k$$

#### 디코딩 전략

논문의 핵심 발견:[1]
- **첫 번째 RVQ 레벨**: 16번 반복 필요 (품질이 중요)
- **레벨 2-12**: 탐욕적 디코딩만으로 충분 (미세 세부사항이므로)

이는 총 27번의 순전파로 18,000개 토큰(30초 오디오)을 생성함을 의미합니다:

$$\text{Forward Passes} = 16 + \sum_{q=2}^{12} 1 = 27$$

비교: **자동회귀 모델**은 각 토큰마다 하나의 순전파가 필요하므로 18,000번의 순전파 필요[1]

***

## 3. 성능 향상

### 3.1 속도 개선[1]

|메트릭|SoundStorm|AudioLM|개선율|
|---|---|---|---|
|30초 생성 시간|0.5초|~50초|**100배**|
|실시간 계수(RTF)|0.017|~1.67|**98배**|
|파이프라인 전체|2초|~60초|**30배**|

구성:
- 시맨틱 생성: 1.4초
- SoundStorm: 0.5초  
- SoundStream 디코딩: 0.1초

### 3.2 음성 품질 메트릭[1]

#### Word Error Rate (WER) - 낮을수록 좋음

프롬프트 없는 경우:

|길이|Original|AudioLM|SoundStorm|개선|
|---|---|---|---|---|
|4-10s|2.62|4.65|3.48|**25.2% ↓**|
|10-20s|1.95|3.59|2.55|**28.9% ↓**|
|20-30s|2.20|4.79|3.33|**30.4% ↓**|

프롬프트 있는 경우 추가 개선:

|길이|AudioLM|SoundStorm|개선|
|---|---|---|---|
|4-10s|3.77|2.99|**20.4% ↓**|
|10-20s|3.40|2.43|**28.5% ↓**|
|20-30s|3.75|3.36|**10.4% ↓**|

#### Voice Preservation (음성 보존)[1]

프롬프트를 사용한 3초 음성 샘플로부터 스피커 임베딩 코사인 유사도:

|길이|AudioLM|SoundStorm|
|---|---|---|
|4-10s|0.46|0.57|
|10-20s|0.48|0.59|
|20-30s|0.48|0.59|

**평가**: WavLM 기반 스피커 검증 모델 사용 (Chen et al., 2022)[1]

#### Acoustic Consistency (음성 일관성)[1]

프롬프트와의 일관성을 2초 마다 평가 (Figure 2 데이터):

$$\text{Consistency}(t) = \text{Cosine Similarity}(\text{Embed}_{prompt}, \text{Embed}_{t})$$

- **SoundStorm**: 시간이 지남에 따라 일관성 유지 (0.96-0.91)
- **AudioLM**: 현저한 표류(drift) (0.86으로 저하)

**의의**: 장시간 생성에서 SoundStorm의 우월성을 입증

#### Audio Quality[1]

DNS-MOS 유사 모델 사용:

|길이|Original|AudioLM|SoundStorm|
|---|---|---|---|
|4-10s|3.72|3.93|4.01|
|10-20s|3.91|4.04|4.16|
|20-30s|3.99|4.08|4.20|

**평가**: AudioLM과 동등한 품질 달성, 장시간에서 약간의 개선

### 3.3 디코딩 반복 횟수의 영향[1]

Figure 4 (음성 품질 vs 첫 레벨 반복 횟수):
- 1회 반복: 3.7점
- 16회 반복: 3.85-3.90점
- 32회 반복: 3.90-3.92점 (수렴)

**결론**: 첫 레벨에서 16회 반복이 최적값, 추가 반복의 한계효과 미미

***

## 4. 모델 일반화 성능

### 4.1 강점

#### 음성 보존의 강건성[1]

미학습(unseen) 스피커에 대한 성능:
- 3초 프롬프트로 0.57-0.59의 높은 음성 유사도 달성
- AudioLM 대비 일관되게 우월 (0.46-0.48)

이는 SoundStorm의 병렬 디코딩이 프롬프트 정보를 더 잘 보존함을 시사합니다.

#### 장시간 생성의 일관성[1]

30초까지의 일관성 유지:
- AudioLM: 시간이 지남에 따라 0.96→0.86으로 표류
- SoundStorm: 0.96→0.91로 유지

**가설**: 병렬 처리가 장거리 의존성을 더 효과적으로 모델링

#### 다국어 및 다중 스피커[1]

대화 합성 실험:
- 100,000시간의 대화 말뭉치로 훈련
- 미학습 스피커의 목소리 특성 유지
- 스피커 턴 제어 가능

### 4.2 한계

#### 음악 및 효과음[1]

논문에서 명시적 제한:
- 시맨틱 토큰과 첫 RVQ 레벨이 모든 음향 세부사항을 포착하지 못함
- 음악/효과음에서는 상위 레벨에서도 여러 반복 필요 가능성

#### RVQ 레벨 2-12에서의 디코딩 전략[1]

Figure 4 분석:
- 레벨 2-12에서 여러 반복은 통계적으로 유의미한 개선 미제공
- 이는 이들 레벨의 독립성 가정이 강하다는 것을 암시

#### 미학습 조건에 대한 평가 부족

보고서의 한계:
- 저잡음 환경 (LibriSpeech test-clean)에서만 평가
- 강한 배경음이나 방언에 대한 일반화 미평가

### 4.3 일반화 성능 분석

#### 외삽(Extrapolation) 능력

- **학습**: 0-30초 무작위 윈도우
- **평가**: 4-30초 구간에 대해서만 평가

**문제**: 30초 초과에 대한 일반화 불명확

#### 조건부 신호의 영향

두 가지 설정 비교:[1]
1. 조건 없음: 시맨틱 토큰의 중복 제거 제거 (더 강한 조건)
2. AudioLM과 동일 설정

더 강한 조건이 WER을 현저히 개선하므로, **일반화는 조건부 신호 품질에 매우 의존**.

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 AudioLM (Borsos et al., 2022)[3][4][5][2]

**아키텍처**:
- 3단계 계층적 자동회귀 구조
  1. 시맨틱 토큰 생성 (w2v-BERT)
  2. 거친 음성 토큰 (coarse acoustic)
  3. 미세 음성 토큰 (fine acoustic)
- 각 단계마다 별도의 Transformer 디코더

**성과**:
- 음성 및 피아노 음악 생성 가능
- 길이 일관성과 고품질의 균형 달성
- 장기 구조 모델링 (문법, 멜로디)

**한계**:
- 자동회귀 생성의 순차적 특성으로 인한 느린 속도
- 장시간 생성에 큰 메모리 요구

**vs SoundStorm**:
- SoundStorm이 100배 빠르면서도 동등한 품질 달성
- 더 나은 음성 보존과 일관성

### 5.2 VALL-E (Wang et al., 2023)[6][7][8][9]

**핵심 혁신**:
- 신경 코덱 언어 모델링을 TTS에 처음 도입
- 하이브리드 접근: 첫 RVQ 레벨 자동회귀, 나머지 병렬

**구조**:
- 제로샷 TTS를 조건부 언어 모델링으로 정의
- 60,000시간의 다중 스피커 영어 데이터
- 음성 프롬프트 + 텍스트 → 음성 생성

**성과**:
- 3초 프롬프트로 미학습 스피커의 음성 합성
- 감정과 음성 환경(reverberation) 보존
- 이전 최고 성능 시스템 대비 +0.12 CMOS 개선

**한계**:
- 여전히 텍스트 조건 필요
- 첫 레벨의 자동회귀 병목

**vs SoundStorm**:
| 측면 | VALL-E | SoundStorm |
|------|--------|-----------|
| 조건 | 텍스트 + 음성 프롬프트 | 시맨틱 토큰 |
| 디코딩 | 하이브리드 (AR + 병렬) | 완전 병렬 |
| 속도 | 실시간 대비 느림 | 30초 @ 0.5초 |
| 컨텍스트 | 제로샷 TTS | 음성 연속성 |

### 5.3 MaskGIT (Chang et al., 2022)[10][11][12][13]

**아이디어**:
- 이미지 생성을 위한 비자동회귀 병렬 디코딩
- 마스크된 토큰을 신뢰도 기반으로 반복 정제

**성과**:
- ImageNet에서 최고 성능 Transformer 능가
- 자동회귀 대비 **64배 속도 향상**
- Inpainting, 외삽 등 여러 작업 지원

**영향**:
- SoundStorm의 직접적 영감원
- 신뢰도 기반 스케줄 개념 차용

**vs SoundStorm**:
- MaskGIT: 이미지용 (최대 1024 토큰)
- SoundStorm: RVQ 계층 구조에 특화 (18,000 토큰 처리)

### 5.4 SPEAR-TTS (Kharitonov et al., 2023)[14][15][16][17][18]

**특징**:
- 최소 지도 학습(minimal supervision) TTS
- AudioLM을 텍스트 조건으로 확장
- 두 가지 시퀀스-투-시퀀스 작업:
  1. 텍스트 → 시맨틱 토큰
  2. 시맨틱 토큰 → 음성 토큰

**훈련 효율**:
- 15분의 병렬 데이터로 훈련 가능
- 음성만의 데이터로 "speaking" 모듈 훈련
- 역번역(backtranslation)으로 감독 데이터 증강

**성과**:
- CER 1.92% on LibriSpeech test-clean
- 3초 프롬프트로 미학습 스피커 일반화
- 다양한 음성 합성 가능

**SoundStorm과의 결합**:
- 논문에서 대화 합성에 SPEAR-TTS + SoundStorm 조합 제시
- 빠른 텍스트-투-스피치 파이프라인 가능

### 5.5 VALL-E 2 (Chen et al., 2024)[19][20][21]

**획기적 성과**:
- **인간 수준의 제로샷 TTS 달성** (Human Parity)
- LibriSpeech와 VCTK에서 인간 음성과 구분 불가

**혁신**:
1. **반복 인식 샘플링(Repetition Aware Sampling)**:
   - 이전 샘플링 이력에서 반복된 토큰 추적
   - 무한 루프 회피 및 안정성 향상
   
2. **그룹화된 코드 모델링(Grouped Code Modeling)**:
   - RVQ 코드를 그룹으로 조직화
   - 시퀀스 길이 단축 → 속도 및 메모리 개선

**성과 (정량)**:
- 복잡한/반복 문구에서도 안정적 생성
- 스피커 유사도: 4.1+ (인간 수준)
- WER: 유의미한 개선

**vs SoundStorm**:
| 구분 | SoundStorm | VALL-E 2 |
|------|-----------|----------|
| 출시 | 2023.5 | 2024.6 |
| 조건 | 시맨틱 토큰 | 텍스트 + 음성 |
| 인간 동등 | 아니오 | **예** |
| 디코딩 | 신뢰도 기반 | 반복 인식 샘플링 |
| 속도 | 매우 빠름 | 중간 |
| 강건성 | 중간 | 높음 |

### 5.6 관련 최신 기술 동향 (2024-2025)[22][23][24][25][26]

#### RVQ 개선 방향

1. **VRVQ (Variable Bitrate RVQ, 2024)**[25][26]
   - 프레임별로 동적 비트레이트 조정
   - 침묵이나 단순 음성에서 효율성 향상
   - 고정 비트레이트의 한계 극복

2. **EuleroDec (2025)**[22]
   - 복소수값 RVQ-VAE
   - 위상-크기 결합 유지
   - GAN 없이 고품질 달성

3. **ERVQ (2024)**[23]
   - 코드북 내/간 최적화
   - RVQ 성능 향상

#### 음성 합성의 진화

- **E2-TTS (2024)**: 완전 비자동회귀 접근으로 인간 수준 달성
- **Voicebox (Meta)**: 확산 기반 대안
- **GPT-SoVITS**: 대규모 LLM 통합

***

## 6. 논문이 앞으로의 연구에 미치는 영향

### 6.1 근본적 기여

#### 아키텍처 패러다임 전환

SoundStorm은 **계층적 토큰 구조의 명시적 활용**을 통해 새로운 설계 원칙을 제시합니다:

$$\text{효율성} = f(\text{토큰 구조 이해}, \text{조건부 독립성 활용})$$

이는 다음 세대 모델들에 직접 영향:
- VALL-E 2의 그룹화된 코드 모델링
- 향후 멀티모달 생성 모델의 설계 원리

#### 비자동회귀 생성의 입증

장시간 고품질 시퀀스 생성에서 **비자동회귀 방식이 우월**함을 실증적으로 입증:
- 속도: 100배 향상
- 일관성: 더 나음
- 품질: 동등 이상

이는 음성, 영상, 텍스트 등 다양한 모달리티에서 비자동회귀 연구 활성화를 촉발했습니다.

### 6.2 후속 연구의 방향

#### 단기 연구 (2023-2024)

1. **VALL-E 계열의 진화**
   - VALL-E 2의 반복 인식 샘플링은 SoundStorm의 신뢰도 기반 접근의 개선
   - 그룹화된 코드 모델링은 계층 구조 활용의 진화

2. **멀티모달 통합**
   - AudioPaLM: SoundStorm 기반 멀티모달 음성-텍스트 모델
   - LauraGPT: 오디오와 텍스트 통합

#### 중기 연구 (2024-2025)

1. **강건성 향상**
   - 노이즈 환경, 저자원 언어 등 challenging 시나리오
   - 도메인 외 일반화(OOD generalization)

2. **효율성 개선**
   - RVQ 개선 (VRVQ, ERVQ 등)
   - 더 짧은 시퀀스 길이의 토큰화

3. **조건부 신호 확장**
   - 비시간 정렬 조건 처리
   - 스타일, 감정, 맥락 조건

#### 장기 연구 (2025+)

1. **통합 생성 모델**
   - 음성, 음악, 음향 효과를 통합하는 단일 모델
   - 콘텐츠 전반에 대한 일관성 유지

2. **실시간 생성**
   - 현재: 30초 @ 0.5초 (60배 빠름)
   - 목표: 진정한 실시간 (스트리밍) 생성

3. **제어 가능성**
   - 상세한 음성 특성 제어
   - 정밀한 시간 동기화

### 6.3 학제간 영향

**컴퓨터 비전**: MaskGIT 이후 비자동회귀 이미지/비디오 생성의 확산
- 이후 월드 모델(world model) 생성에도 적용

**자연언어처리**: 비자동회귀 텍스트 생성의 재조명
- Insertion Transformer, Diffusion LM 등과 병렬 발전

**생물정보학**: 서열 생성 모델
- RVQ 구조가 단백질, DNA 서열에도 적용

***

## 7. 앞으로 연구 시 고려할 점

### 7.1 기술적 고려사항

#### 1. 조건부 신호의 품질과 정렬

**현재 한계**:
- 시맨틱 토큰은 시간 정렬 필수
- 25 fps의 시맨틱 토큰을 50 fps SoundStream으로 중복 확대

**개선 방향**:
- 비정렬 조건부 신호 처리 메커니즘 (크로스 어텐션)
- 가변 속도 시맨틱 추출

#### 2. 미세 레벨의 디코딩 전략

**발견**: 레벨 2-12에서 탐욕적 디코딩만으로 충분

**문제**:
- 음악/효과음에서는 이 가정이 위반될 가능성
- 도메인별 최적 반복 횟수 결정 필요

**해결책**:
- 적응형 디코딩: 콘텐츠별 반복 횟수 동적 조정
- 메타 학습: 최적 반복 스케줄 학습

#### 7.2 일반화 성능 개선

#### 1. 도메인 외 평가 확대

**필요 평가 도메인**:
- 저자원 언어 (언어 30+)
- 강한 배경음성 (SNR < 10dB)
- 병렬 음성 (칵테일 파티 효과)
- 음악 및 효과음

#### 2. 미학습 조건에 대한 강건성

**현재 약점**:
- CER (문자 오류율)에서 여전히 원본 대비 격차 (1.24-3.36 vs 0.69-0.89)
- 반복 구문에서의 안정성 미흡

**개선 전략**:
```
강건성 향상 = 
  ① 데이터 증강 (노이즈 추가, 음성 변조)
  + ② 대안적 손실 함수 (Focal Loss, Curriculum Learning)
  + ③ 앙상블 디코딩 (다중 샘플 평균)
  + ④ 불확실성 추정 및 재샘플링
```

#### 7.3 확장성 및 효율성

#### 1. 더 긴 시퀀스 처리

**현재**: 30초 (1,500 프레임)

**문제**: 영화, 팟캐스트 같은 장시간 콘텐츠

**솔루션**:
- 청크 기반 생성 + 스트리밍 재접합
- Hierarchical 구조: 장 단위 → 장면 단위 → 발화 단위
- 메모리 효율적 주의 (Local/Sparse Attention)

#### 2. 계산 복잡도 분석

현재 SoundStorm의 복잡도:
$$\mathcal{O}(\text{Frames} \times Q \times \text{Iterations})$$
$$= \mathcal{O}(1500 \times 12 \times 27) \approx \mathcal{O}(486K)$$

vs 자동회귀:
$$\mathcal{O}(18000 \times 1) = \mathcal{O}(18K) \text{ 순전파 수}$$

**개선 가능성**:
- Flash Attention으로 주의 계산 최적화
- 낮은 정밀도(FP8) 추론
- 모델 경량화 (Distillation, Pruning)

### 7.4 윤리 및 안전성

#### 1. 합성 음성의 탐지

논문에서: 98.5% 탐지율[1]

**미래 위험**:
- 탐지기도 지속적 개선 필요
- 적대적 공격에 대한 강건성

**권장사항**:
- 음성 워터마킹 표준 개발
- 합성 음성 메타데이터 삽입
- 검증 가능한 출처 증명(Proof of Provenance)

#### 2. 개인 정보 보호

**문제**:
- 3초 음성 샘플로 음성 클론 가능
- 학습 데이터의 저작권/동의 문제

**해결책**:
- 페더레이션 학습(Federated Learning)
- 차등 개인정보보호(Differential Privacy)
- 명시적 동의 및 사용 라이선스

#### 7.5 벤치마크 및 평가

#### 권장 평가 프레임워크

| 차원 | 메트릭 | 목표 |
|------|--------|------|
| **정확성** | WER, CER | < 2% (LibriSpeech clean) |
| **음성 보존** | Speaker Similarity | > 0.95 |
| **일관성** | Acoustic Drift over Time | < 0.10 |
| **품질** | MOS, DNS-MOS | ≥ 4.5 |
| **강건성** | Noisy TIMIT, CommonVoice | Multi-language support |
| **효율** | RTF, Memory | RTF < 0.1 |
| **안전성** | Synthetic Speech Detection | > 99% TPR@FPR<1% |

### 7.6 미해결 문제(Open Challenges)

#### 1. Representation Learning

- RVQ 토큰의 의미론적 해석 부족
- 계층별 토큰의 정보 이론적 분석 필요

#### 2. Generalization Bounds

- SoundStorm의 미학습 도메인에 대한 일반화 이론적 보장 부족
- PAC-Bayes 또는 Rademacher complexity 분석 필요

#### 3. Long-Context Modeling

- 30초는 여전히 비교적 짧음
- 멀티 턴 대화(분 단위)의 일관성 유지 기작 불명확

#### 4. Interpretability

- 신뢰도 기반 샘플링의 결정 과정 불투명
- 어떤 토큰이 왜 선택되는지에 대한 설명 메커니즘 부족

***

## 결론

SoundStorm은 신경 오디오 코덱의 계층적 구조를 명시적으로 활용하여 **비자동회귀 병렬 생성의 실현 가능성**을 입증한 획기적 작업입니다. 100배의 속도 향상과 더 우수한 음성 일관성은 AudioLM 이후 음성 생성 분야에서 새로운 패러다임을 제시했습니다.

특히 다음 세대 모델들(VALL-E 2, E2-TTS 등)이 이를 기반으로 인간 수준의 성능을 달성한 점은 SoundStorm의 근본적 기여를 증명합니다. 그러나 음악/효과음 생성, 극단적 노이즈 환경, 30초 이상 장시간 콘텐츠에서의 일반화 성능은 여전히 개선이 필요합니다.

앞으로의 연구는 **강건성, 확장성, 해석 가능성**을 중심으로 진행될 것으로 예상되며, 특히 비자동회귀 생성의 이론적 기초와 안전한 배포 메커니즘의 개발이 시급합니다.

***

## 참고 문헌

 Borsos, Z., et al. (2023). "SoundStorm: Efficient Parallel Audio Generation." arXiv:2305.09636.[1]

 Borsos, Z., et al. (2022). "AudioLM: A Language Modeling Approach to Audio Generation." arXiv:2209.03143.[2]

 AudioLM (Google Research). https://ai.googleblog.com/2022/10/audiolm-language-modeling-approach-to.html[3]

 Wang, C., et al. (2023). "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers." arXiv:2301.02111.[6]

 Chang, H., et al. (2022). "MaskGIT: Masked Generative Image Transformer." CVPR 2022.[10]

 Google Research AudioLM Blog. https://research.google/blog/audiolm-a-language-modeling-approach-to-audio-generation/[4]

[48-49, 55, 70, 72] Kharitonov, E., et al. (2023). "Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision." TACL.

 "EuleroDec: A Complex-Valued RVQ-VAE for Efficient Audio Codecs." arXiv:2601.17517.[22]

 Chen, S., et al. (2024). "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers." arXiv:2406.05370.[19]

 "ERVQ: Enhanced Residual Vector Quantization with Intra-and-Inter-Codebook Optimization." arXiv:2410.12359.[23]

 Chae, Y., et al. (2024). "Variable Bitrate Residual Vector Quantization for Audio Codecs." arXiv:2410.06016.[24]

 "VRVQ: Variable Bitrate Residual Vector Quantization for Audio Compression." arXiv:2410.06016.[26][25]

출처
[1] 2305.09636v1.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e39c806e-9783-48e6-b660-77a1cf662691/2305.09636v1.pdf
[2] AudioLM: A Language Modeling Approach to Audio Generation https://ieeexplore.ieee.org/document/10158503/
[3] AudioLM: a Language Modeling Approach to Audio ... https://arxiv.org/abs/2209.03143
[4] AudioLM: a Language Modeling Approach to Audio ... https://research.google/blog/audiolm-a-language-modeling-approach-to-audio-generation/
[5] [PDF] AudioLM: a Language Modeling Approach to Audio Generation https://david.grangier.info/papers/2023/audio-lm-generation.pdf
[6] Neural Codec Language Models are Zero-Shot Text to ... https://arxiv.org/pdf/2301.02111.pdf
[7] Neural Codec Language Models are Zero-Shot Text to ... https://arxiv.org/abs/2301.02111
[8] Vall E https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e/
[9] In-depth Review of VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers https://www.youtube.com/watch?v=fCtbnhR83UI
[10] MaskGIT: Masked Generative Image Transformer https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.pdf
[11] MaskGIT: Efficient Masked Generative Transformer https://www.emergentmind.com/topics/maskgit
[12] [2202.04200] MaskGIT: Masked Generative Image Transformer - arXiv https://arxiv.org/abs/2202.04200
[13] MaskGIT: Masked Image Generative Transformers https://research.google/pubs/maskgit-masked-image-generative-transformers/
[14] Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal
  Supervision http://arxiv.org/pdf/2302.03540v1.pdf
[15] Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00618/2200655/tacl_a_00618.pdf
[16] arXiv:2302.03540v1 [cs.SD] 7 Feb 2023 https://arxiv.org/pdf/2302.03540.pdf
[17] Speak, Read and Prompt: High-Fidelity Text-to-Speech ... https://aclanthology.org/2023.tacl-1.95/
[18] SPEAR-TTS https://google-research.github.io/seanet/speartts/examples/
[19] VALL-E 2: Neural Codec Language Models are Human ... https://arxiv.org/abs/2406.05370
[20] Microsoft's VALL-E 2: First Time Human Parity in Zero-Shot Text-to ... https://syncedreview.com/2024/06/11/microsofts-vall-e-2-first-time-human-parity-in-zero-shot-text-to-speech-achieved/
[21] Review Microsoft's VALL-E 2 (Achieving Human Parity in Zero-shot TTS) https://www.youtube.com/watch?v=d3Zf_m8MaQw
[22] EuleroDec: A Complex-Valued RVQ-VAE for Efficient and ... https://arxiv.org/html/2601.17517v2
[23] ERVQ: Enhanced Residual Vector Quantization with Intra ... https://arxiv.org/html/2410.12359v2
[24] Variable Bitrate Residual Vector Quantization for Audio ... https://arxiv.org/abs/2410.06016
[25] Variable Bitrate Residual Vector Quantization for Audio ... https://sonyresearch.github.io/VRVQ/
[26] VRVQ: Variable Bitrate Residual Vector Quantization for ... https://ai.sony/publications/VRVQ-Variable-Bitrate-Residual-Vector-Quantization-for-Audio-Compression/
[27] The HCCL-DKU system for fake audio generation task of the 2022 ICASSP ADD Challenge https://www.semanticscholar.org/paper/463d34ab1565ab49e7a331d7e5328c5640ff7693
[28] ADD 2022: the first Audio Deep Synthesis Detection Challenge https://ieeexplore.ieee.org/document/9746939/
[29] The IVI Lab entry to the GENEA Challenge 2022 – A Tacotron2 Based Method for Co-Speech Gesture Generation With Locality-Constraint Attention Mechanism https://dl.acm.org/doi/10.1145/3536221.3558060
[30] Exemplar-based Stylized Gesture Generation from Speech: An Entry to the GENEA Challenge 2022 https://dl.acm.org/doi/10.1145/3536221.3558068
[31] GestureMaster: Graph-based Speech-driven Gesture Generation https://dl.acm.org/doi/10.1145/3536221.3558063
[32] Perceptual Conversational Head Generation with Regularized Driver and Enhanced Renderer https://dl.acm.org/doi/10.1145/3503161.3551577
[33] Automated Audio Captioning and Language-Based Audio Retrieval https://arxiv.org/abs/2207.04156
[34] A Baseline for ViCo Conversational Head Generation Challenge https://dl.acm.org/doi/10.1145/3503161.3551569
[35] Superb @ SLT 2022: Challenge on Generalization and Efficiency of Self-Supervised Speech Representation Learning https://ieeexplore.ieee.org/document/10022770/
[36] AudioLM: a Language Modeling Approach to Audio Generation https://arxiv.org/pdf/2209.03143.pdf
[37] LM-VC: Zero-shot Voice Conversion via Speech Generation based on
  Language Models https://arxiv.org/pdf/2306.10521.pdf
[38] AudioPaLM: A Large Language Model That Can Speak and Listen http://arxiv.org/pdf/2306.12925.pdf
[39] Audio-Agent: Leveraging LLMs For Audio Generation, Editing and
  Composition https://arxiv.org/html/2410.03335
[40] LauraGPT: Listen, Attend, Understand, and Regenerate Audio with GPT http://arxiv.org/pdf/2310.04673.pdf
[41] AudioSetCaps: An Enriched Audio-Caption Dataset using Automated
  Generation Pipeline with Large Audio and Language Models https://arxiv.org/html/2411.18953v1
[42] Make Some Noise: Towards LLM audio reasoning and generation using sound
  tokens https://arxiv.org/html/2503.22275
[43] Takin: A Cohort of Superior Quality Zero-shot Speech Generation Models https://arxiv.org/html/2409.12139
[44] Masked Diffusion Generative Recommendation https://arxiv.org/html/2601.19501v1
[45] AudioPaLM: A Large Language Model That Can Speak ... https://arxiv.org/html/2306.12925v1
[46] SC VALL-E: Style-Controllable Zero-Shot Text to Speech ... https://arxiv.org/abs/2307.10550
[47] Demystifying MaskGIT Sampler and Beyond https://www.arxiv.org/pdf/2510.04525.pdf
[48] A Language Modeling Approach to Audio Generation https://www.semanticscholar.org/paper/AudioLM:-A-Language-Modeling-Approach-to-Audio-Borsos-Marinier/8c870bef01a4fbb20f60722ffc2f6bee3870b18b
[49] VALL-E R: Robust and Efficient Zero-Shot Text-to-Speech ... https://arxiv.org/html/2406.07855v1
[50] Accurate and Efficient World Modeling with Masked Latent ... https://arxiv.org/html/2507.04075v1
[51] Towards Universal Audio Generation by Large-scale ... https://arxiv.org/html/2310.00704v6
[52] VALL-E 2: Neural Codec Language Models are Human ... https://arxiv.org/html/2406.05370v1
[53] Parallel Decoding with Exploration for Diffusion Language ... https://arxiv.org/html/2511.21103v1
[54] [R] Google AudioLM produces amazing quality continuation of voice and piano prompts https://www.reddit.com/r/MachineLearning/comments/xy3zfe/r_google_audiolm_produces_amazing_quality/
[55] VALL-E - Microsoft https://www.microsoft.com/en-us/research/project/vall-e-x/
[56] AudioLM : a Language Modeling Approach to Audio Generation https://dl.acm.org/doi/10.1109/TASLP.2023.3288409
[57] State-of-the-art Zero-shot Speech Synthesis with Vall-E https://www.youtube.com/watch?v=NPfkNmfAeR8
[58] arXiv:2202.04200v1 [cs.CV] 8 Feb 2022 https://arxiv.org/pdf/2202.04200.pdf
[59] MuLanTTS: The Microsoft Speech Synthesis System for Blizzard Challenge 2023 https://arxiv.org/abs/2309.02743
[60] Few-shot Dysarthric Speech Recognition with Text-to-Speech Data Augmentation https://zenodo.org/records/8092573/files/is2023-dysarthric-tts.pdf
[61] M2-CTTS: End-to-End Multi-scale Multi-modal Conversational
  Text-to-Speech Synthesis https://arxiv.org/pdf/2305.02269.pdf
[62] FCTalker: Fine and Coarse Grained Context Modeling for Expressive
  Conversational Speech Synthesis https://arxiv.org/pdf/2210.15360.pdf
[63] Neural Text to Articulate Talk: Deep Text to Audiovisual Speech
  Synthesis achieving both Auditory and Photo-realism https://arxiv.org/pdf/2312.06613.pdf
[64] SR-TTS: a rhyme-based end-to-end speech synthesis system https://www.frontiersin.org/articles/10.3389/fnbot.2024.1322312/pdf?isPublishedV2=False
[65] Towards Lightweight and Stable Zero-shot TTS with Self-distilled
  Representation Disentanglement https://arxiv.org/pdf/2501.08566.pdf
[66] arXiv:2312.05415v1 [cs.SD] 8 Dec 2023 https://arxiv.org/pdf/2312.05415.pdf
[67] RALL-E: Robust Codec Language Modeling with Chain-of- ... https://www.semanticscholar.org/paper/RALL-E:-Robust-Codec-Language-Modeling-with-for-Xin-Tan/691eae8477f8091a0e4222de770ee7fbcbc82e17
[68] Towards Controllable Speech Synthesis in the Era of Large ... https://arxiv.org/html/2412.06602v2
[69] arXiv:2406.18009v2 [eess.AS] 12 Sep 2024 https://arxiv.org/pdf/2406.18009.pdf
[70] CoVoMix: Advancing Zero-Shot Speech Generation for ... https://arxiv.org/html/2404.06690v1
[71] ERVQ: Enhanced Residual Vector Quantization with Intra- ... https://arxiv.org/abs/2410.12359
[72] Pseudo-Autoregressive Neural Codec Language Models ... https://arxiv.org/pdf/2504.10352.pdf
[73] Text to Speech Synthesis via Dual Language Modeling https://arxiv.org/html/2509.22062
[74] MBCodec:Thorough disentangle for high-fidelity audio ... https://arxiv.org/abs/2509.17006
[75] arXiv:2407.12707v3 [eess.AS] 2 Dec 2024 https://www.arxiv.org/pdf/2407.12707v3.pdf
[76] Residual Vector Quantization for Audio and Speech Embeddings https://www.youtube.com/watch?v=Xt9S74BHsvc
[77] Speak, Read and Prompt: High-Fidelity Text-to-Speech ... https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00618/118854/Speak-Read-and-Prompt-High-Fidelity-Text-to-Speech
[78] VRVQ: Variable Bitrate Residual Vector Quantization for ... https://neurips.cc/media/neurips-2024/Slides/98231.pdf
[79] VALL-E 2 成為首個達到人類水準的 TTS，基於風險微軟不打算公開發表 https://technews.tw/2024/07/15/ai-speech-generator-reaches-human-parity/
[80] High-Fidelity Text-to-Speech with Minimal Supervision https://randomsampling.tistory.com/376
[81] Residual Vector Quantization (RVQ) From Scratch https://www.youtube.com/watch?v=ZnyfaQRQ8GI
