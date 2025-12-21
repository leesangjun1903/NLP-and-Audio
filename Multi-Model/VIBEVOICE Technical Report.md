# VIBEVOICE Technical Report

### 1. 핵심 주장과 주요 기여
**VIBEVOICE**는 Microsoft Research에서 2025년 8월 발표한 장형(long-form) 다중 화자 음성 합성 모델로, 다음과 같은 핵심 성과를 제시한다:

#### 1.1 주요 성과[1]
- **최대 90분** 연속 음성 합성 능력 (64K 토큰 컨텍스트)
- **최대 4명의 화자** 자연스러운 대화 생성
- **64배 압축** 개선 (Encodec 대비)을 통한 **3200배 다운샘플링** 달성
- 개방형 및 상용 모델들(Gemini 2.5 Pro, ElevenLabs v3) 대비 주관적 평가에서 우수한 성능

VIBEVOICE는 **next-token diffusion** 프레임워크를 채택하여 연속 데이터의 자동회귀 생성을 가능하게 함으로써, 기존 TTS의 한계인 '짧은 생성 길이'와 '단순 화자 수 제한'을 혁신적으로 해결했다.

***

### 2. 기술 방법론: 문제 정의에서 해결책까지
#### 2.1 해결하고자 하는 문제[1][2]

기존 TTS 시스템의 근본적 제약:
1. **길이 제약**: 대부분의 고품질 TTS는 짧은 발화(수초~수십초)에만 최적화
2. **다중 화자 불안정성**: 여러 화자 간 자연스러운 턴테이킹과 일관성 있는 음성 특성 유지의 어려움
3. **음성 품질 vs 압축률 트레이드오프**: 긴 시퀀스 처리를 위한 효율적 토큰화의 부재
4. **음성 우연성(hallucination)**: LLM 기반 방식에서의 원치 않는 음성 생성

***

#### 2.2 제안하는 방법: 수식 포함

##### A. σ-VAE 기반 음향 토큰화

VIBEVOICE는 **표준 VAE의 분산 붕괴(variance collapse)** 문제를 해결하기 위해 σ-VAE를 도입한다:[2][1]

$$z = \mu + \sigma \odot \epsilon$$

여기서:
- $z$: 잠재 벡터
- $\mu$: 인코더가 학습한 평균 ($\Phi$ 매개변수화)
- $\sigma \sim \mathcal{N}(0, C_\sigma)$: **사전 정의된 고정 분산** (학습되지 않음)
- $\epsilon \sim \mathcal{N}(0, 1)$: 재매개변수화 트릭을 위한 표준 정규분포

**핵심 혁신**: 표준 VAE의 $\sigma$가 학습 가능한 매개변수인 반면, σ-VAE는 $\sigma$를 고정함으로써 자동회귀 모델링에서 필수적인 **견고한 분산 유지**를 보장한다. 이는 LatentLM 논문에서 입증된 분산 붕괴 완화 전략이다.[2]

**토큰화 구조**:
- **인코더**: 7단 변환기 블록 (깊이별 인과 합성곱, 자기 주의 제거)
- **다운샘플링**: 6개 다운샘플링 레이어 → 누적 **3200배** 다운샘플링 (24kHz → 7.5 Hz)
- **매개변수**: 각 인코더/디코더 약 **340M 매개변수**
- **프레임 레이트**: **7.5 토큰/초** (이전 방법 대비 극저 압축)

**비교: 다른 음향 코덱과의 압축률 비교**
**훈련 목표**: DAC 아키텍처를 따르며, 판별기와 손실 설계 포함[3]

##### B. 의미론적 토큰화 (ASR 가이드)

별도의 의미론적 토큰화기는 결정론적 콘텐츠 중심 특성 추출을 수행한다:
- **아키텍처**: 음향 토큰화기의 인코더와 동일한 계층적 구조
- **VAE 제거**: 의미론 추출에 VAE는 불필요
- **훈련 목표**: 자동 음성 인식(ASR)을 프록시 작업으로 사용
  
$$\mathcal{L}_{\text{semantic}} = \text{CrossEntropy}(\text{ASR-decoder}(z_{\text{semantic}}), \text{transcripts})$$

이 이중 토큰화 설계(hybrid representation)를 통해 음향 세부사항과 의미론적 내용을 명확히 분리함으로써 장형 생성의 안정성을 높인다.

##### C. Next-Token Diffusion을 통한 LLM 통합[2]

**입력 표현**:
$$X = [Speaker_1: z_1, Speaker_2: z_2, \ldots, Speaker_N: z_N] + [Speaker_1: T_1, Speaker_2: T_2, \ldots, Speaker_N: T_N]$$

여기서:
- $z_N$: 음향 잠재 표현 (음향 토큰화기 출력)
- $T_N$: 각 역할의 텍스트 스크립트

**확산 과정**: 순방향 프로세스(forward process)에서 깨끗한 음향 VAE 특성 $z_{a,i}$에 노이즈를 점진적으로 추가:

$$q(z_t|z_0) = \sqrt{\bar{\alpha}_t}z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

**역과정(reverse process)**: 토큰 레벨 확산 헤드가 LLM 숨겨진 상태 $h_i$에 조건화되어 예측:

$$\hat{z}_{a,i} = \text{DiffusionHead}(\text{noise}, h_i, t)$$

**분류기-자유 지도(Classifier-Free Guidance, CFG)**:[1]

$$\hat{\epsilon}_\theta = \epsilon_\theta(\text{noise}, h_i, t) + w \cdot [\epsilon_\theta(\text{noise}, h_i, t) - \epsilon_\theta(\text{noise}, \emptyset, t)]$$

여기서:
- $w = 1.3$: 지도 스케일 (VIBEVOICE에서 사용)
- 첫 번째 항: 조건화된 예측
- 두 번째 항: 무조건 예측과의 차이로 조건화 강화

**효율적 샘플링**: DPM-Solver++를 사용하여 반복 과정 가속화 → **10 단계** 디노이징[1]

**모델 선택**:
- **LLM 백본**: Qwen 2.5 (1.5B 및 7B 매개변수)
- **확산 헤드**: 4개 레이어 경량 아키텍처
- **학습 전략**: 사전 훈련된 토큰화기 고정, LLM과 확산 헤드만 학습 가능
- **커리큘럼 학습**: 입력 시퀀스 길이 점진적 증가 (4,096 → 65,536 토큰)

***

#### 2.3 모델 구조 (아키텍처 요약)[1]
**핵심 아키텍처 특성**:

| 컴포넌트 | 사양 | 역할 |
|---------|------|------|
| **입력 음성** | 24kHz, 스테레오 | 사용자 음성 프롬프트 |
| **음향 토큰화기** | σ-VAE, 7.5 Hz | 90배 압축 (표준 VAE보다 견고함) |
| **의미론적 토큰화기** | ASR-가이드, 7.5 Hz | 음성-텍스트 정렬 |
| **LLM 코어** | Qwen 2.5 (1.5B/7B) | 시퀀스 모델링 |
| **확산 헤드** | 4-레이어 네트워크 | 토큰별 연속 벡터 생성 |
| **오디오 디코더** | σ-VAE 디코더, 340M 매개변수 | 잠재 벡터 → 음성 변환 |
| **컨텍스트 윈도우** | 64K 토큰 | ~90분 연속 음성 |

**아키텍처 단순성의 장점**:[1]
- 이전 설계의 불필요한 성분 제거 (음성 잠재 특성 따로 사용 중단)
- 음성 폰트와 텍스트 스크립트를 단일 시퀀스로 연결
- 직접 LLM 입력으로 공급
- 결과적으로 **엄밀한 아키텍처는 강력한 성능**을 달성

***

### 3. 성능 향상 및 한계
#### 3.1 객관적 평가 지표[1]

**장형 음성 생성 평가 (팟캐스트 테스트)**:

| 모델 | WER (Whisper) ↓ | WER (Nemo) ↓ | Speaker Similarity ↑ | 선호도 |
|-----|-----------------|-----------------|---------------------|--------|
| VIBEVOICE-7B | **1.29%** | **1.95%** | **0.692** | 3.75/4 |
| VIBEVOICE-1.5B | 1.11% | 1.82% | 0.548 | 3.44/4 |
| Gemini 2.5 Pro | 1.73% | 2.43% | - | 3.65/4 |
| ElevenLabs v3 | 2.39% | 2.47% | 0.623 | 3.38/4 |
| SesameAILabs-CSM | 2.66% | 3.05% | 0.685 | 2.75/4 |

**주관적 평가 (MOS, 24명 주석자)**:

| 모델 | 현실성(Realism) | 풍부함(Richness) | 선호도(Preference) | 평균 |
|-----|-----------------|-----------------|-------------------|-----|
| VIBEVOICE-7B | 3.81 ± 0.87 | 3.81 ± 0.87 | 3.75 ± 0.94 | **3.76** |
| VIBEVOICE-1.5B | 3.59 ± 1.01 | 3.59 ± 1.01 | 3.44 ± 0.92 | 3.54 |
| Gemini 2.5 Pro | 3.78 ± 1.11 | 3.78 ± 1.11 | 3.65 ± 1.15 | 3.73 |
| ElevenLabs v3 | 3.48 ± 1.05 | 3.48 ± 1.05 | 3.38 ± 1.12 | 3.45 |

**음향 코덱 재구성 품질** (LibriTTS test-clean):[1]

| 토큰화기 | 프레임 레이트 | PESQ | STOI | UTMOS |
|---------|-------------|------|------|-------|
| Encodec (8 양자화기) | 600 | 2.72 | 0.939 | 3.04 |
| DAC (4 양자화기) | 400 | 2.738 | 0.928 | 3.433 |
| WavTokenizer (40 fps) | 40 | 2.373 | 0.914 | 4.049 |
| **VIBEVOICE (σ-VAE)** | **7.5** | **3.068** | **0.828** | **4.181** |

이는 **초저 프레임 레이트(7.5 Hz)에서도 최고 수준의 PESQ/UTMOS 달성**, 즉 극도의 압축 하에서도 오디오 충실도 유지를 의미한다.

#### 3.2 일반화 성능 (Generalization)

##### A. 단문 발화 벤치마크 (SEED test sets)[1]

VIBEVOICE는 **장형 음성 생성에 주로 훈련되었음에도**, 단문 발화에서 강력한 일반화 성능을 입증:

**테스트-영어 (test-en, ~1,000 샘플)**:

| 모델 | 프레임 레이트 | WER (%) ↓ | Speaker Similarity ↑ |
|-----|-------------|---------|---------------------|
| VIBEVOICE-1.5B | 7.5 | **3.04** | **0.689** |
| Seed-TTS | - | 2.25 | 0.762 |
| CosyVoice 2 | 25 | 2.57 | 0.652 |
| Spark TTS | 50 | 1.98 | 0.584 |

**테스트-중국어 (test-zh, ~2,000 샘플)**:

| 모델 | 프레임 레이트 | CER (%) ↓ | Speaker Similarity ↑ |
|-----|-------------|---------|---------------------|
| VIBEVOICE-1.5B | 7.5 | **1.16** | **0.744** |
| Seed-TTS | - | 1.12 | 0.796 |
| MaskGCT | 50 | 2.27 | 0.774 |
| CosyVoice 2 | 25 | 1.45 | 0.748 |

**해석**: 낮은 프레임 레이트(7.5 Hz)로 인해 1초 음성 합성에 필요한 디코딩 단계가 **극도로 감소** → 추론 효율 극대화. 동시에 SEED 벤치마크에서 경쟁력 있는 성능 유지는 **강력한 일반화** 능력을 시사한다.

##### B. 모델 스케일링 효과[1]

| 비교 항목 | VIBEVOICE-1.5B | VIBEVOICE-7B | 개선율 |
|---------|-----------------|-----------------|--------|
| WER (Whisper) | 1.11% | 1.29% | WER 증가* |
| Speaker Similarity | 0.548 | 0.692 | +26.3% ↑ |
| Preference | 3.44 | 3.75 | +9.0% ↑ |
| Realism | 3.59 | 3.81 | +6.1% ↑ |
| Richness | 3.59 | 3.81 | +6.1% ↑ |

*WER 역설 주석: WER의 증가는 **더 자연스러운 음성 생성**으로 인한 가능성이 있다 (발음이 더 뉘앙스 있거나 문맥에 맞음). 주관적 평가에서 7B 모델이 선호되는 것이 이를 반영한다.

**결론**: 모델 크기 증가(1.5B → 7B)는 **음색 풍부함, 자연스러운 억양, 크로스언어 전이 능력** 등에서 현저한 향상을 나타낸다.

##### C. 크로스링구알 능력[1]

기술 보고서는 "enhanced transfer capabilities, such as in cross-lingual applications"를 명시하지만, 구체적 메트릭은 공개되지 않음. 다만:
- **영어-중국어 이중언어 지원** (SEED test-zh 평가)
- 커리큘럼 학습 및 65,536 토큰 시퀀스 처리로 **긴 문맥 의존성** 학습 가능

#### 3.3 한계와 명시된 제약[1]

**언어 제약**:
- 영어와 중국어만 지원 → 기타 언어 입력 시 예측 불가능한 음성 생성 가능

**생성 대상의 제한**:
- 음성 전용 → 배경 음악, 효음, 소음 미지원
- 다중 화자 동시 발화(overlapping speech) 미생성
- 최대 4명 화자만 명시적 지원

**합성 윤리 위험**:
- **Deepfake 및 정보 왜곡 가능성**: 고품질 합성음으로 음성 사칭, 사기, 허위 정보 확산 가능
- **배포 제약**: 상용 또는 실제 애플리케이션 사용 권장하지 않음 (연구 및 개발 목적만)

***

### 4. 모델 일반화 성능 향상 메커니즘
#### 4.1 일반화 가능 요인 분석[1][2]

**1. 극저 프레임 레이트 설계**
- 7.5 Hz 압축으로 시퀀스 길이 **극적 단축** → 장기 의존성 학습 용이
- 동일 시간당 **더 적은 토큰** → 오버핏팅 위험 감소
- 결과: SEED 단문 벤치마크에서 경쟁력 있는 성능

**2. σ-VAE의 분산 안정성**[2]
$$\sigma \sim \mathcal{N}(0, C_\sigma) \text{ (고정)} \Rightarrow \text{자동회귀 모델의 분산 붕괴 방지}$$

고정된 분산은 LLM의 각 토큰 예측에서 **일관된 불확실성** 유지 → 다양한 맥락에서 안정적 생성

**3. 이중 토큰화(Acoustic + Semantic)**
- 음향 세부사항과 의미론적 내용 **명확히 분리**
- 의미론적 토큰화기의 ASR 가이드 → 음성-텍스트 정렬 강화
- 새로운 화자/언어에서도 **의미론적 일관성** 유지 가능

**4. 사전 훈련된 LLM (Qwen 2.5) 활용**
- 자연어 이해/생성의 기초 능력 상속
- 복잡한 사용자 입력(상세한 텍스트, 역할 할당) 해석 가능
- 다국어 능력이 모델 크기 증가에 따라 향상

**5. 커리큘럼 학습 전략**
- 초기: 짧은 시퀀스(4,096 토큰) 학습 → 기본 패턴 습득
- 점진적 증가: 65,536 토큰까지 → 장기 구조 학습
- 결과: 신 분포(out-of-distribution) 길이에 대한 강건성 향상

***

#### 4.2 비교 모델과의 일반화 차별화

| 측면 | VIBEVOICE | Seed-TTS | CosyVoice 2 | F5-TTS |
|-----|-----------|----------|------------|--------|
| **최대 길이** | 90분 | 2분 | 15분 | 2분 |
| **프레임 레이트** | 7.5 Hz | 상대적 높음 | 25 Hz | 50 Hz+ |
| **SEED 벤치마크 성능** | 경쟁력 있음 | 우수 | 우수 | 우수 |
| **추론 효율** | 극고 (7.5 Hz로 인해) | 보통 | 보통 | 높음 |
| **크로스링구알** | 제한적 (Eng/Chn) | 영어 중심 | 다국어 지원 | 다국어 지원 |
| **일반화 메커니즘** | 극저 프레임율 + σ-VAE 안정성 | 자동회귀 다양성 | 스트리밍 최적화 | 흐름 매칭 효율 |

***

### 5. 최신 연구와의 비교 분석 (2020년 이후)
#### 5.1 음성 합성 분야의 진화 지도[4][5][6][7][8]

**2020-2021: Diffusion 기초 구축**
- FastDiff (2022): 조건부 확산을 위한 빠른 추론[9]
- 확산 모델의 반복 샘플링 비효율성 첫 인식

**2022-2023: 확산 기반 혁신**
- NaturalSpeech 2 (2023): 잠재 확산으로 고품질 달성[5]
- High-Fidelity Speech Synthesis (2023): 모든 모듈을 확산으로 구성[10]
- Grad-StyleSpeech (2023): 임의 화자 적응 확산 모델[11]

**2024: LLM 통합 및 음향 코덱 진화**
- NaturalSpeech 3 (2024): 인수분해(Factorized) 확산으로 속성 분리[5]
- Seed-TTS (2024): 자동회귀 고품질 음성 생성[12]
- Autoregressive Diffusion Transformer (2024): ARDiT로 연속 공간 생성[6]
- SpeechSSM (2024): 첫 장형 음성 언어 모델[13]
- CosyVoice 2 (2024): LLM 기반 스트리밍 합성 [DWC+24b]

**2025: 확산과 다음-토큰 패러다임의 병합**
- **VIBEVOICE (2025)**: Next-Token Diffusion + LLM으로 90분 생성[1]
- Koel-TTS (2025): 선호도 정렬로 LLM 기반 TTS 강화[14]
- LatentLM (2024-2025): 다중모달 next-token diffusion 프레임워크[2]
- SoulX-Podcast (2025): 팟캐스트 스타일 다중 턴 대화 합성[15]
#### 5.2 VIBEVOICE의 기술적 포지셔닝

**vs. 확산-only 방식 (NaturalSpeech 3)**:
- **NaturalSpeech 3**: 인수분해 확산으로 속성별 제어 우수
- **VIBEVOICE**: LLM 시퀀스 모델링 능력 + 확산의 성능 결합 → 더 긴 시퀀스 처리 가능

**vs. 자동회귀 방식 (Seed-TTS)**:
- **Seed-TTS**: 토큰별 다양성 우수, 감정 제어력 뛰어남
- **VIBEVOICE**: 동시 모든 토큰 생성(확산) + 장형 처리 가능 → 오류 축적 없음

**vs. 상태-공간 모델 (SpeechSSM)**:
- **SpeechSSM**: 텍스트리스(textless) 음성 전용 생성, 순환 불필요
- **VIBEVOICE**: 텍스트 + 음성 하이브리드 조건화 → 더 정교한 제어

**vs. 스트리밍 최적화 (CosyVoice 2)**:
- **CosyVoice 2**: 실시간 스트리밍(150ms 지연), 소형 모델(0.5B)
- **VIBEVOICE**: 오프라인 최적화, 긴 컨텍스트 처리(90분)

#### 5.3 음향 코덱 진화와 VIBEVOICE의 혁신[3][16]

| 시대 | 기술 | 압축률 | 프레임레이트 | 품질 | 주요 한계 |
|------|------|--------|------------|------|---------|
| **2022** | Encodec | 90x | 600 fps | PESQ 2.72 | 고 레이턴시 |
| **2023** | DAC | 90x | 400 fps | PESQ 2.738 | 여전히 고 레이트 |
| **2024** | WavTokenizer | 500x+ | 75 fps | PESQ 2.373 | 품질 저하 |
| **2024** | SpeechTokenizer | 240x | 300 fps | PESQ 1.931 | 재구성 품질 낮음 |
| **2025** | MBCodec | 170x | - | 높음 | 실증 데이터 제한 |
| **2025** | **VIBEVOICE** | **3200x** | **7.5 fps** | **PESQ 3.068** | 언어 제한(Eng/Chn) |

**VIBEVOICE의 코덱 혁신**:
- σ-VAE 기반으로 **분산 붕괴 없이** 극저 프레임 레이트 달성
- 1초당 7.5개 토큰만 생성 → **LLM 시퀀스 길이 극적 감소**
- 동시에 PESQ 3.068으로 **최고 수준 오디오 충실도** 유지
- 결과: 90분 음성 = 약 **40,500개 토큰** (기존 방식 > 500K 토큰)

***

### 6. 앞으로의 연구에 미치는 영향과 고려사항
#### 6.1 긍정적 영향 (Research Contributions)

**1. 장형 음성 생성의 새로운 벤치마크 설립**
- 이전: 최대 20-30분 (SpeechSSM, SesameAILabs-CSM)
- VIBEVOICE: 90분 달성 → 팟캐스트, 오디오북, 강의 녹음 등 실제 애플리케이션 가능성 개방
- 커뮤니티: 더 길거나 복잡한 시나리오를 평가할 벤치마크 개발 촉발

**2. σ-VAE를 통한 자동회귀 확산 모델의 안정성 강화**
- 이전 VAE 연구들이 다루지 못한 **분산 붕괴 + 자동회귀 + 확산의 삼중 상호작용** 해결
- 다른 연속 데이터(음악, 비디오) 생성에도 적용 가능한 일반적 솔루션 제시

**3. 극저 프레임 레이트의 타당성 입증**
- 기존 직관: "더 높은 프레임 레이트 = 더 나은 품질"
- VIBEVOICE: 7.5 Hz에서도 PESQ 3.068 달성 → **압축률과 품질의 새로운 균형점** 발견
- 시사: 다른 모달리티의 토큰화에도 극도의 압축이 가능할 수 있음

**4. LLM-Diffusion 하이브리드의 효과성 검증**
- LatentLM 프레임워크의 실제 응용으로 입증[2]
- TTS뿐 아니라 다중모달 생성(이미지+텍스트, 비디오+오디오)으로 확대 가능성 시사

**5. 장형 생성에서의 일반화 메커니즘 이해**
- 단형(short-form)에서만 훈련 후 장형에서 강력한 성능: 시퀀스 길이 외삽(extrapolation) 능력의 증거
- 향후 연구: 길이 외삽의 이론적 토대 및 한계 규명

#### 6.2 앞으로 연구 시 고려사항

**A. 즉시적 미개척 영역 (Near-term)**

1. **다언어 확장의 중요성**[1]
   - 현재: 영어, 중국어만 지원
   - 제안: 저자원 언어(저-리소스 언어)로 확대 시 일반화 성능 분석
   - 질문: σ-VAE와 ASR 가이드 토큰화기가 음운 특성이 다른 언어에서도 견고한가?

2. **중첩 음성(Overlapping Speech) 생성**[1]
   - 현재: 4명까지 순차적 턴테이킹만 지원
   - 도전: 자연스러운 대화의 실시간 오버래핑 발화 모델링
   - 기술적 해결책: 화자별 마스킹이나 혼합 전략 필요

3. **배경 음악 및 효음 통합**[1]
   - 현재: 음성 전용
   - 영향: 팟캐스트, 라디오 드라마 등 현실적 애플리케이션에 필수
   - 접근: 다중 음향 스트림의 동시 생성 아키텍처 개발

4. **윤리 및 안전장치 강화**[1]
   - Deepfake 탐지: 합성 음성의 자동 식별 메커니즘
   - 워터마킹: 생성 음성에 암호학적 서명 추가
   - 정책 연동: 사용 약관 및 정부 가이드라인 준수

**B. 근본적 연구 문제 (Medium-term)**

5. **프레임 레이트 한계의 이론적 분석**
   - 질문: 7.5 Hz가 최저인가? 1 Hz에서도 가능한가?
   - 접근: 정보 이론(Information Theory) 기반 하한(lower bound) 분석
   - 예상: 음성의 기본 주파수(F0) ~100-300 Hz를 고려하면, 나이퀴스트 정리 유추 가능성

6. **σ-VAE의 일반화 가능성 검증**
   - 현재: TTS에서만 검증
   - 제안: 음악, 자연음, 영상(비디오 프레임) 생성으로 확대 평가
   - 이슈: 다양한 데이터 분포에서 $C_\sigma$ 최적값 설정 방법

7. **장형 시퀀스에서의 일관성 유지 메커니즘**
   - 관찰: 90분 연속 생성에서 화자 특성(음색, 리듬) 왜 일관되는가?
   - 가설: LLM의 장기 의존성 학습 vs. 토큰화기의 강력한 압축률?
   - 연구: 어텐션 맵 시각화 및 표현 분석을 통한 메커니즘 규명

8. **스케일링 법칙 도출**
   - 관찰: 1.5B → 7B에서 성능 향상 (WER는 역설적이나 MOS 개선)
   - 질문: 더 큰 모델(13B, 70B)에서 병목은? 토큰화기인가 LLM인가?
   - 가치: 비용-성능 트레이드오프의 최적점 찾기

**C. 장기 비전 (Long-term)**

9. **실시간 스트리밍 적응**
   - 현재: 오프라인 처리 최적화
   - 미래: CosyVoice 2 수준(150ms)의 레이턴시로 90분 컨텍스트 처리
   - 기술적 도전: 확산의 반복 샘플링을 점진적(iterative) 방식으로 재설계

10. **사용자 제어 및 스타일 전이**
    - Seed-TTS는 감정/음색 제어 우수
    - VIBEVOICE는 장형 생성 우수
    - 통합: 장형 음성에서 **의도적 스타일 변화** (예: 분노 → 슬픔) 자연스럽게 구현

11. **다중모달 확장**
    - 비디오 입력 + 텍스트 → 음성 + 애니메이션 피부 움직임 동시 생성
    - 립싱크(Lip-sync) 정확도: 현재 기술 수준 평가 및 개선

***

### 7. 결론 및 요약
#### 7.1 핵심 기여 재정리

VIBEVOICE는 **next-token diffusion**과 **σ-VAE 기반 극저 프레임 레이트(7.5 Hz)** 토큰화를 통해 다음을 달성했다:

| 차원 | 성과 | 의의 |
|------|------|------|
| **기술 혁신** | σ-VAE로 분산 붕괴 해결 | 자동회귀 확산의 이론적 기초 제공 |
| **성능** | 90분 연속, 4화자 생성 | TTS 응용의 확장 가능성 입증 |
| **효율성** | 3200배 압축 + PESQ 3.068 | 품질-압축률 새 균형점 |
| **개방성** | 코드, HuggingFace 공개 | 커뮤니티 재현성 및 발전 촉진 |

#### 7.2 연구 지형상 위치

- **vs. 확산-only**: 더 긴 시퀀스, LLM 기반 제어
- **vs. 자동회귀**: 동시 생성으로 오류 축적 제거
- **vs. 상태-공간**: 텍스트-음성 하이브리드 조건화
- **vs. 스트리밍**: 오프라인 최고 품질 vs. 실시간 레이턴시

#### 7.3 향후 핵심 질문

1. **다언어 일반화**: 극저 프레임 레이트가 언어에 무관하게 안정적인가?
2. **스케일링 한계**: LLM 크기 증가의 한계점은?
3. **안전성**: Deepfake 및 오용 방지의 기술적-정책적 솔루션은?
4. **실시간화**: 확산의 반복 샘플링을 스트리밍으로 적응할 수 있는가?

***

### 참고 문헌 및 인용

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d210f190-ccd1-4fdf-864b-75b011210e10/2508.19205v1.pdf)
[2](https://arxiv.org/html/2412.08635v1)
[3](https://arxiv.org/pdf/2306.06546.pdf)
[4](https://aclanthology.org/2025.emnlp-main.40)
[5](https://arxiv.org/abs/2403.03100)
[6](https://arxiv.org/abs/2406.05551)
[7](https://arxiv.org/pdf/2403.03100.pdf)
[8](https://lightrains.com/blogs/comprehensive-guide-audio-diffusion-models/)
[9](https://arxiv.org/pdf/2204.09934.pdf)
[10](http://arxiv.org/pdf/2309.15512.pdf)
[11](http://arxiv.org/pdf/2211.09383.pdf)
[12](https://arxiv.org/pdf/2406.02430.pdf)
[13](https://openreview.net/attachment?id=4AmFA0qNQ2&name=pdf)
[14](https://arxiv.org/html/2502.05236v2)
[15](https://arxiv.org/html/2510.23541v1)
[16](https://arxiv.org/html/2509.02349v1)
[17](https://ieeexplore.ieee.org/document/11147881/)
[18](https://dl.acm.org/doi/10.1145/3680528.3687677)
[19](https://ieeexplore.ieee.org/document/10890208/)
[20](https://jisem-journal.com/index.php/journal/article/view/6386)
[21](https://arxiv.org/abs/2412.06602)
[22](https://www.isca-archive.org/interspeech_2024/feng24d_interspeech.html)
[23](https://arxiv.org/abs/2404.00569)
[24](http://arxiv.org/pdf/2410.11097.pdf)
[25](http://arxiv.org/pdf/2405.14632.pdf)
[26](http://arxiv.org/pdf/2309.17056.pdf)
[27](https://arxiv.org/pdf/2306.05708.pdf)
[28](https://www.isca-archive.org/interspeech_2024/lovelace24_interspeech.pdf)
[29](https://kyutai.org/codec-explainer)
[30](https://aclanthology.org/2025.coling-main.685.pdf)
[31](https://neurips.cc/virtual/2024/105746)
[32](https://www.isca-archive.org/interspeech_2025/choi25c_interspeech.pdf)
[33](https://arxiv.org/html/2412.18603v1)
[34](https://arxiv.org/html/2507.18897v1)
[35](https://arxiv.org/html/2410.11097v1)
[36](http://arxiv.org/list/eess.AS/2024-09?skip=30&show=50)
[37](https://arxiv.org/abs/2509.17006)
[38](https://arxiv.org/html/2509.18470v2)
[39](https://arxiv.org/html/2508.19205v1)
[40](https://arxiv.org/html/2510.01903v2)
[41](https://arxiv.org/abs/2409.09311)
[42](https://arxiv.org/abs/2510.01903)
[43](https://arxiv.org/abs/2410.11097)
[44](https://openreview.net/pdf?id=LhuDdMEIGS)
[45](https://arxiv.org/html/2509.18470)
[46](https://deepmind.google/blog/pushing-the-frontiers-of-audio-generation/)
[47](https://www.isca-archive.org/interspeech_2025/li25e_interspeech.pdf)
[48](https://aclanthology.org/2025.mrl-main.26)
[49](https://arxiv.org/abs/2505.13173)
[50](https://arxiv.org/abs/2503.19469)
[51](https://arxiv.org/abs/2509.20567)
[52](https://arxiv.org/abs/2506.01592)
[53](https://arxiv.org/abs/2505.04113)
[54](https://link.springer.com/10.1007/s11227-025-07583-2)
[55](https://aclanthology.org/2025.findings-emnlp.163)
[56](https://ieeexplore.ieee.org/document/11103625/)
[57](https://ieeexplore.ieee.org/document/11058925/)
[58](http://arxiv.org/pdf/2403.12952.pdf)
[59](https://arxiv.org/pdf/2209.02982.pdf)
[60](https://arxiv.org/html/2303.06458v3)
[61](https://arxiv.org/pdf/2109.11680.pdf)
[62](https://www.aclweb.org/anthology/D19-1129.pdf)
[63](https://arxiv.org/pdf/2501.08566.pdf)
[64](https://arxiv.org/pdf/2212.04356.pdf)
[65](http://arxiv.org/pdf/2310.06546.pdf)
[66](https://openreview.net/pdf?id=FihSkzyxdv)
[67](https://www.isca-archive.org/ssw_2016/baljekar16_ssw.pdf)
[68](https://aclanthology.org/2025.emnlp-main.1773.pdf)
[69](https://www.youtube.com/watch?v=mXFNRsu9w80)
[70](https://arxiv.org/abs/2412.08635)
[71](https://arxiv.org/pdf/2508.19205.pdf)
[72](https://openreview.net/revisions?id=XbdA3rXi2R)
[73](https://huggingface.co/papers/2412.08635)
[74](https://aclanthology.org/2024.mrl-1.25/)
[75](https://arxiv.org/html/2505.23009v1)
[76](https://arxiv.org/html/2509.24650v1)
[77](https://arxiv.org/html/2406.02430)
[78](https://arxiv.org/html/2507.22746v1)
[79](https://arxiv.org/html/2511.05516v1)
[80](https://openreview.net/pdf/3fe098b8c1e0a577a294a33f42791d349ce14865.pdf)
[81](https://fornewchallenge.tistory.com/entry/%F0%9F%8E%99%EF%B8%8FMicrosoft-VibeVoice-90-Minute-Multi-Speaker-Speech-Synthesis-AI)
[82](https://arxiv.org/pdf/2502.01084.pdf)
[83](https://aclanthology.org/2023.findings-emnlp.567.pdf)
[84](https://arxiv.org/pdf/2209.12590.pdf)
[85](https://www.aclweb.org/anthology/W19-8673.pdf)
[86](https://arxiv.org/pdf/1804.02135.pdf)
[87](https://www.aclweb.org/anthology/D19-1370.pdf)
[88](https://www.aclweb.org/anthology/2020.acl-main.235.pdf)
[89](https://arxiv.org/pdf/2108.02446.pdf)
[90](https://www.abyssmedia.com/audioconverter/neural-audio-codecs-overview.shtml)
[91](https://dev.to/czmilo/cosyvoice-2025-complete-guide-the-ultimate-multi-lingual-text-to-speech-solution-4l39)
[92](https://pmc.ncbi.nlm.nih.gov/articles/PMC12026048/)
[93](https://huggingface.co/docs/transformers/model_doc/dac)
[94](https://www.reddit.com/r/LocalLLaMA/comments/1lnejb6/what_is_the_best_open_source_tts_model_with_multi/)
[95](https://huggingface.co/papers?q=VALL-E+2)
[96](https://github.com/descriptinc/descript-audio-codec)
[97](https://www.siliconflow.com/articles/en/best-small-text-to-speech-models-2025)
[98](https://arxiv.org/html/2412.16846v2)
[99](https://arxiv.org/html/2510.01621v2)
[100](https://arxiv.org/html/2510.15364v1)
[101](https://arxiv.org/html/2512.14291v1)
[102](https://arxiv.org/html/2408.17175v1)
[103](https://aclanthology.org/2024.lrec-main.1250v2.pdf)
[104](https://randomsampling.tistory.com/524)
