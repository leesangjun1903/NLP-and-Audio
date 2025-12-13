# AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head

### 1. 핵심 주장과 주요 기여

**AudioGPT** 논문의 핵심 주장은 기존의 **대규모 언어 모델(LLM)인 ChatGPT가 텍스트 처리에는 탁월하지만, 복잡한 오디오 정보를 처리하고 음성 대화를 수행할 수 없다는 문제**를 해결하는 것입니다.[1]

주요 기여는 다음과 같습니다:[1]

- **ChatGPT를 오디오 기반 모델(Foundation Models)로 보강**: 단순히 처음부터 멀티모달 LLM을 학습하는 대신, 기존의 다양한 오디오 기반 모델들을 활용하여 음성, 음악, 사운드 및 토킹헤드에 관한 16가지 AI 작업을 해결할 수 있는 시스템 개발

- **음성 대화 인터페이스 통합**: 자동 음성 인식(ASR)과 텍스트-음성 변환(TTS)을 통한 음성 입출력 인터페이스를 구현하여 음성 기반 상호작용 가능

- **멀티모달 LLM 평가 원칙 제시**: **일관성(Consistency), 능력(Capability), 견고성(Robustness)** 의 세 가지 차원에서 멀티모달 LLM을 평가하는 체계적인 프로임워크 제안

***

### 2. 해결하고자 하는 문제 및 제안하는 방법

#### 2.1 해결하는 문제[1]

AudioGPT가 직면한 문제는 크게 두 가지입니다:

1. **데이터 부족**: 인간 라벨 음성 데이터를 얻는 것이 비용이 높고 시간이 많이 소비되며, 텍스트 데이터의 방대한 양과 비교할 때 제한적
2. **계산 자원 부족**: 처음부터 멀티모달 LLM을 학습하는 것은 계산적으로 비용이 많이 들고 시간이 소요됨

#### 2.2 제안하는 방법과 시스템 공식화[2][1]

AudioGPT는 다음과 같이 공식화됩니다:

$$ \text{AudioGPT}(T, L, M, H, \{P_i\}_{i=1}^{P}) \quad (1) $$

여기서:
- $T$: 모달리티 변환기 (Modality Transformer)
- $L$: 대화 엔진 (Dialogue Engine) - 대규모 언어 모델
- $M$: 프롬프트 관리자 (Prompt Manager)
- $H$: 작업 처리기 (Task Handler)
- $\{P_i\}_{i=1}^{P}$: P개의 오디오 기반 모델 세트

#### 2.3 추론 프로세스[1]

사용자의 새로운 쿼리 $q_n$과 대화 컨텍스트 $C = \{(q_1, r_1), (q_2, r_2), ..., (q_{n-1}, r_{n-1})\}$가 주어졌을 때, AudioGPT는 응답 $r_n$을 다음과 같이 생성합니다:

$$r_n = \text{AudioGPT}(q_n, C) \quad (2) $$

이는 다음 네 단계로 구성됩니다:

**1) 모달리티 변환 (Modality Transformation)**

사용자 입력 쿼리 $q_n$을 다음과 같이 변환합니다:

```math
q_{n}=(q_{d}^{n},q_{sn}^{1},q_{sn}^{2},...,q_{sn}^{k})
```

여기서 $q_d^n$은 쿼리 설명 (텍스트 또는 오디오)이고, $q_{sn}^{i}$ 은 쿼리 관련 리소스입니다.

변환 규칙은:

```math
q_{n}^{\prime }=\begin{cases}(T(q_{d}^{n}),q_{sn}^{1},\dots ,q_{sn}^{k})&\text{if\ }q_{d}^{n}\text{\ is\ audio}\\ (q_{d}^{n},q_{sn}^{1},\dots ,q_{sn}^{k})&\text{if\ }q_{d}^{n}\text{\ is\ text}\end{cases}\quad \text{(3)}

```

**2) 작업 분석 (Task Analysis)**

구조화된 인자 $a_n$을 추출합니다:

$$(P_p, h_{P_p}) = L(H(q'_n), q_d^n, C, H(q'_n)) \quad (4) $$

여기서 $P_p$는 선택된 오디오 기반 모델이고, $h_{P_p}$ 는 해당 작업의 관련 인자입니다.

**3) 모델 할당 (Model Assignment)**

선택된 모델을 실행합니다:

$$o_{P_{p}}=P_{p}(q_{sn}^{1},q_{sn}^{2},...,q_{sn}^{k},h_{P_{p}}) \quad (5) $$

**4) 응답 생성 (Response Generation)**

최종 응답을 생성합니다.

***

### 3. 모델 구조 및 지원 작업[1]

#### 3.1 주요 모델 구조

AudioGPT의 아키텍처는 **4단계 파이프라인 구조**로 설계되었습니다:[1]

| 단계 | 역할 | 주요 컴포넌트 |
|------|------|------------|
| 모달리티 변환 | 음성/텍스트 입력을 일관된 형식으로 변환 | ASR (Whisper), TTS (FastSpeech 2) |
| 작업 분석 | 사용자의 의도를 파악하고 적절한 모델 선택 | 대화 엔진 (ChatGPT), 프롬프트 관리자 |
| 모델 할당 | 선택된 기반 모델 실행 | 16개의 오디오 기반 모델 |
| 응답 생성 | 결과를 사용자에게 반환 | 오디오/비디오/텍스트 포맷팅 |

#### 3.2 지원되는 16가지 작업[1]

| 작업 카테고리 | 입출력 | 주요 모델 | 사용 사례 |
|-------------|-------|---------|---------|
| **음성 인식** | 오디오 → 텍스트 | Whisper | 음성 전사 |
| **음성 번역** | 오디오 → 텍스트 | MultiDecoder | 언어 간 음성 번역 |
| **음성 스타일 전이** | 오디오 → 오디오 | GenerSpeech | 참조 음성의 스타일 적용 |
| **음성 강화** | 오디오 → 오디오 | ConvTasNet | 배경 소음 제거 |
| **음성 분리** | 오디오 → 오디오 | TF-GridNet | 혼합 음성 분리 |
| **모노-바이노럴 변환** | 오디오 → 오디오 | NeuralWarp | 모노를 입체 오디오로 변환 |
| **오디오 인페인팅** | 오디오 → 오디오 | Make-An-Audio | 오디오 영역 채우기 |
| **사운드 추출** | 오디오 → 오디오 | LASSNet | 특정 사운드 추출 |
| **사운드 감지** | 오디오 → 텍스트 | Pyramid Transformer | 음향 이벤트 감지 및 시간 예측 |
| **토킹헤드 합성** | 오디오 → 비디오 | GeneFace | 오디오에 기반한 얼굴 애니메이션 |
| **텍스트-음성 변환** | 텍스트 → 오디오 | FastSpeech 2 | 자연스러운 음성 합성 |
| **텍스트-오디오 생성** | 텍스트 → 오디오 | Make-An-Audio | 텍스트 설명에서 사운드 생성 |
| **이미지-오디오 생성** | 이미지 → 오디오 | Make-An-Audio | 이미지에서 오디오 생성 |
| **음성 캡셔닝** | 오디오 → 텍스트 | MAAC | 오디오 설명 생성 |
| **싱잉 합성** | 악보 → 오디오 | DiffSinger, VISinger | 음악 악보에서 노래 생성 |

***

### 4. 성능 향상 및 한계[1]

#### 4.1 성능 평가 프레임워크[1]

AudioGPT는 **세 가지 핵심 차원**에서 평가됩니다:

**1) 일관성 (Consistency)**

사용자의 의도를 정확히 이해하고 적절한 기반 모델을 할당하는 능력을 평가합니다. 평가 척도는 20-100의 Likert 척도입니다:

| 등급 | 일관성 정의 | 설명 |
|------|-----------|------|
| 20 | 완전히 불일치 | 매우 불쾌한 불일치 |
| 40 | 대부분 불일치 | 불쾌하지만 객관적이지 않은 불일치 |
| 60 | 어느 정도 일치 | 인지 가능하고 약간 불쾌한 불일치 |
| 80 | 대부분 일치 | 거의 인지되지 않지만 약간의 불일치 |
| 100 | 완전히 일치 | 감지할 수 없는 불일치 |

**2) 능력 (Capability)**

각 기반 모델이 다양한 오디오 작업에서 달성할 수 있는 성능입니다. 주요 메트릭:[1]

- **음성 인식**: WER (Word Error Rate), CER (Character Error Rate)
- **음성 번역**: BLEU (Bilingual Evaluation Understudy)
- **음성 스타일 전이**: MCD (Mel Cepstral Distortion), FFE (Fine-grained F0 Error), MOS (Mean Opinion Score)
- **음성 강화/분리**: SNR, PESQ, STOI
- **토킹헤드 합성**: FID (Fréchet Inception Distance), LMD (Lip Sync Accuracy)

**3) 견고성 (Robustness)**

시스템이 다음과 같은 특수한 경우를 처리하는 능력을 평가합니다:[1]

- 긴 작업 체인 처리
- 지원하지 않는 작업에 대한 적절한 피드백 제공
- 모델 오류 발생 시 사용자 안내
- 논리적 순서가 없는 쿼리 처리

#### 4.2 성능 향상의 주요 특징

AudioGPT의 성능 향상은 다음과 같은 특징을 가집니다:

1. **기존 기반 모델의 재사용으로 인한 효율성**: 전체 시스템을 재학습할 필요 없이, 이미 검증된 16개의 전문화된 모델을 활용

2. **멀티라운드 대화 지원**: 사용자와의 장시간 상호작용에서 컨텍스트를 유지하면서 연속적인 작업 수행 가능

3. **크로스 모달 작업**: 음성 → 텍스트, 텍스트 → 오디오, 오디오 → 비디오 등 다양한 모달리티 간 변환 지원

#### 4.3 주요 한계[1]

논문에서 명시된 한계는:

**1) 프롬프트 엔지니어링의 어려움**

AudioGPT는 ChatGPT를 사용하여 다수의 기반 모델을 연결하기 때문에, 오디오 기반 모델을 자연어로 설명하기 위해 프롬프트 엔지니어링이 필요합니다. 이는:
- 시간 소비적이고
- 전문 지식을 요구하며
- 프롬프트 설계 오류 시 작업 실패율 증가

**2) 토큰 길이 제한**

ChatGPT의 최대 토큰 길이 제약으로 인해:
- 멀티턴 대화 길이 제한
- 사용자의 장시간 컨텍스트 지시사항 처리 불가
- 상황에 따라 컨텍스트 손실 가능

**3) 기반 모델 성능에 대한 의존성**

AudioGPT는 각 작업에 대해 기반 모델을 사용하므로:
- 기반 모델의 부정확성이 최종 결과에 직접 영향
- 새로운 오디오 도메인이나 언어에 대한 일반화 성능이 제한적
- 기반 모델의 업데이트가 필요할 때 전체 시스템이 영향을 받음

***

### 5. 모델의 일반화 성능 향상 가능성[1]

#### 5.1 현재 일반화 성능의 한계

AudioGPT의 일반화 성능은 다음과 같은 요인으로 제약을 받습니다:

1. **도메인 편향**: 각 기반 모델이 학습된 도메인에 특화되어 있어, 새로운 도메인에서의 성능 저하

2. **언어 간 전이의 어려움**: 다국어 처리 시 각 언어별 데이터 부족으로 인한 성능 편차

3. **작업 간 간섭**: 하나의 기반 모델이 여러 작업을 처리할 때, 작업 간의 간섭으로 인한 성능 저하

#### 5.2 일반화 성능 향상을 위한 제안

**1) 기반 모델의 다중 작업 학습 (Multi-Task Learning)**

$$ \mathcal{L}_{\text{total}} = \sum_{i=1}^{N} \lambda_i \mathcal{L}_i(P_p; D_i) $$

여기서:
- $N$: 작업 수
- $\lambda_i$: 작업 $i$에 대한 가중치
- $\mathcal{L}_i$: 작업 $i$의 손실 함수
- $D_i$: 작업 $i$의 데이터셋

**2) 프롬프트 최적화**

일관성 평가 프레임워크를 통해 프롬프트를 반복적으로 개선하면, 모델 할당의 정확성을 향상시킬 수 있습니다.

**3) 기반 모델의 아키텍처 개선**

근래 연구에서 보여준 바와 같이:
- **Transformer 기반 아키텍처**: 더 나은 장거리 의존성 모델링으로 성능 향상
- **이산 표현 학습 (Discrete Representation Learning)**: VQ-VAE, RVQ 기반 토큰화를 통한 보다 효율적인 표현
- **자가 감시 학습 (Self-Supervised Learning)**: wav2vec 2.0, HuBERT 등을 활용한 사전 학습

**4) 컨텍스트-인식 모델 선택 메커니즘**

현재의 작업 기반 선택에서 벗어나, 대화 컨텍스트를 고려한 동적 모델 선택:

$$ s = \arg\max_{p} P(P_p | q_n, C, H(q'_n)) $$

여기서 $s$는 선택된 모델의 인덱스입니다.

#### 5.3 제로샷 및 퓨샷 학습의 잠재성

AudioGPT의 주요 강점은 **제로샷 능력**입니다. 새로운 작업에 대해:

1. 기존 기반 모델들의 조합으로 새로운 능력 창출 가능
2. 프롬프트 엔지니어링만으로 새로운 작업 적응 가능
3. 추가 학습 없이 멀티턴 대화에서 맥락 유지

***

### 6. 관련 최신 연구 비교 분석 (2020년 이후)

#### 6.1 주요 유사 시스템과의 비교

| 시스템 | 출시연도 | 아키텍처 | 주요 특징 | 장점 | 한계 |
|--------|---------|---------|---------|------|------|
| **AudioGPT** | 2023 | 기반 모델 오케스트레이션 + LLM | 16가지 작업 지원, 멀티턴 대화 | 광범위한 작업 커버리지, 제로샷 능력 | 프롬프트 엔지니어링 의존성, 기반 모델 성능 의존성 |
| **AudioLM** | 2023 | 계층적 토큰 시퀀스 모델링 | 음성 및 음악 생성, 장기간 일관성 | 높은 음질, 장기 구조 유지 | 계산 비용 높음, 특정 도메인에 특화 |
| **MusicLM** | 2023 | 텍스트-음악 생성 | 텍스트 설명에서 고품질 음악 생성 | 세밀한 음악 제어, 음악 변환 | 음악 특화, 음성 처리 제한 |
| **SpeechGPT** | 2023 | 이산 음성 표현 + LLM | 교차 모달 대화 능력 | 엔드-투-엔드 모달리티, 정보 손실 감소 | 음성만 처리 |
| **AudioPaLM** | 2023 | 텍스트-음성 LM 융합 | 다국어 음성 이해/생성 | 음성 기반 언어 모델링, 음성 전이 | 계산 자원 많음 |
| **UniAudio** | 2024 | 멀티 스트림 토큰화 | 음성, 음악, 사운드 통합 생성 | 다양한 오디오 타입 지원, 다중 작업 | 모델 크기 큼, 학습 복잡도 높음 |
| **LauraGPT** | 2024 | 오디오-텍스트 통합 LLM | 음성 인식, 이해, 생성 | 오디오 중심 접근, 다양한 응용 | 특화된 도메인별 성능 편차 |

#### 6.2 기술 혁신 비교: 이산 vs 연속 표현[3][4][5]

**AudioLM의 하이브리드 토큰 전략** (2023):[4]

AudioLM은 두 가지 종류의 토큰을 사용합니다:

1. **의미론적 토큰 (Semantic Tokens)**:
   - w2v-BERT에서 추출
   - 음성의 국소적 의존성 (음운론)과 전역 구조 (언어 구문) 포착
   - 샘플링 레이트 낮음 (25 Hz)

2. **음향 토큰 (Acoustic Tokens)**:
   - SoundStream 신경 코덱에서 생성
   - 음성 특성, 녹음 조건 등 세부 정보 포착
   - 높은 샘플링 레이트 (50 Hz)

이 하이브리드 접근은 다음 방정식으로 표현됩니다:

$$ \mathcal{L}_{\text{AudioLM}} = \mathcal{L}_{\text{semantic}} + \sum_{i=1}^{n_{\text{stages}}} \mathcal{L}_{\text{acoustic}}^{(i)} $$

반면 **AudioGPT의 기반 모델 조합 접근**은:
- 프리트레이닝된 모델들의 이점을 직접 활용
- 각 작업별 최적화된 모델 사용 가능
- 하지만 모델 간 정보 손실 발생 가능

#### 6.3 멀티모달 LLM 평가 방법론의 진화[6][7][8]

**AudioGPT의 3차원 평가 프레임워크**와 최신 연구의 비교:

| 평가 차원 | AudioGPT 접근 | 최신 연구 추세 | 진전 방향 |
|---------|-------------|--------------|--------|
| **일관성** | 인간-의도 정렬 평가 (Likert scale) | 다차원 평가 (HELM) | 자동화된 평가 지표로의 진화 |
| **능력** | 작업별 도메인 메트릭 (WER, BLEU, MOS) | 다중 벤치마크 (AudioBench, CMI-Bench) | 통일된 평가 프로토콜 개발 |
| **견고성** | 3가지 특수 케이스 | 다양한 도메인/언어 평가 | 적대적 공격 및 분포 외 데이터 평가 |

#### 6.4 제로샷 음성 합성의 발전[9][10]

최근 연구들은 더 효율적인 제로샷 능력을 보여줍니다:

**MultiVerse** (2024):[10]

음성 처리를 다음과 같이 분해합니다:

$$ \text{Speech} = \text{Content} + \text{Prosody} + \text{Speaker} $$

각 요소를 독립적으로 모델링하여, 제한된 데이터로도 높은 일반화 성능 달성:
- 학습 데이터: ~1.2K 시간 (AudioGPT 기반 모델 학습량과 비교)
- 제로샷 성능: 대규모 모델 수준

**Seed-TTS** (2024):
- 맥락 내 학습(In-context learning)으로 스피커 적응
- 단 1초의 음성 샘플로 새로운 스피커 합성 가능

#### 6.5 일반화 성능 관점의 비교

| 모델 | 학습 데이터 | 다국어 지원 | 제로샷 능력 | 멀티태스크 | 일반화 성능 |
|------|-----------|-----------|-----------|---------|-----------|
| AudioGPT | 기반 모델 의존 | 가능 (Whisper 사용) | 우수 | 16가지 작업 | 중간 |
| AudioLM | ~1,000시간 | 제한적 | 우수 | 음성/음악 | 중상 |
| MusicLM | ~280,000시간 | 음악 특화 | 우수 | 음악 | 우수 |
| UniAudio | ~8,000시간 | 다중 언어 | 우수 | 7가지 작업 | 중상 |
| LauraGPT | 대규모 멀티모달 | 가능 | 중상 | 다양한 음성 작업 | 중상 |

***

### 7. 앞으로의 연구 방향 및 영향

#### 7.1 AudioGPT가 미칠 영향

**1) 멀티모달 시스템 아키텍처의 패러다임 변화**

AudioGPT는 "**기반 모델 오케스트레이션**"이라는 새로운 패러다임을 제시합니다:[1]

- 전통적: 단일 대규모 모델 학습 → 높은 계산 비용
- AudioGPT: 기존 모델 조합 → 효율성과 유연성 증가

이는 다음을 시사합니다:

$$ \text{Capability}_{\text{system}} \approx \sum_{i=1}^{P} \alpha_i \cdot \text{Capability}_{\text{model}_i} + \text{Integration Quality} $$

여기서 적절한 기반 모델 조합과 통합 품질이 시스템 능력을 결정합니다.

**2) 음성 상호작용의 민주화**

기존 음성 시스템 (Siri, Alexa)의 폐쇄적 접근에서 벗어나, 개방형 오디오 처리 능력을 제공합니다.

**3) 다중 모달리티 학습의 통합화**

음성 → 텍스트, 텍스트 → 오디오, 오디오 → 비디오 등의 작업을 하나의 시스템으로 처리 가능함을 보여줍니다.

#### 7.2 향후 연구 시 고려할 점

**1) 프롬프트 엔지니어링의 자동화**

현재의 수동 프롬프트 엔지니어링은 병목입니다. 향후 연구는:

- **자동 프롬프트 최적화**: 강화 학습이나 진화 알고리즘을 활용한 자동 프롬프트 생성
- **메타 프롬프팅**: 프롬프트를 생성하는 프롬프트 학습

**2) 기반 모델의 선택 메커니즘 고도화**

$$ P_{\text{selected}} = \arg\max_{p} \text{Relevance}(P_p, q_n, C) + \text{Confidence}(P_p) $$

- 단순 작업 분류 → 문맥 인식 모델 선택
- 하나의 모델 선택 → 다중 모델 앙상블

**3) 기반 모델 간의 정보 손실 최소화**

현재 시스템에서 모달리티 변환 과정에서 정보가 손실됩니다:

- **텍스트 기반 병목**: 음성 → 텍스트 → 음성 과정에서 운율 정보 손실
- **개선 방향**: 하이브리드 표현 학습으로 음향 정보 보존

**4) 동적 기반 모델 추가/제거**

시스템이 실행 중에:
- 새로운 기반 모델 추가 가능
- 성능 저하 모델 교체
- 도메인 특화 모델의 플러그인화

**5) 멀티턴 대화에서의 맥락 이해 개선**

토큰 길이 제한 해결:

$$ \text{Context Compression} = f(C; \theta) $$

- 하이어라키컬 컨텍스트 압축
- 장기-단기 메모리 분리
- 대화 상태 추적 개선

**6) 도메인 일반화 능력 강화**

기반 모델의 일반화 성능을 높이기 위해:

- **도메인 적응**: 새 도메인에 대한 빠른 적응 메커니즘
- **다국어 처리**: 저자원 언어에 대한 전이 학습
- **음성-음악-사운드의 통합**: 각 오디오 타입 간 공유 표현

**7) 윤리 및 안전성 고려**

- **음성 합성의 오용 방지**: 음성 조작 감지 메커니즘
- **개인정보 보호**: 사용자 음성 데이터의 보안
- **편향 완화**: 인종, 성별, 언어별 성능 편차 감소

#### 7.3 학술적 기여

AudioGPT는 다음과 같은 학술적 논의를 촉발합니다:

1. **멀티모달 LLM 평가의 표준화**
   - 일관성, 능력, 견고성의 3차원 평가 프레임워크는 향후 멀티모달 시스템 평가의 기준이 될 가능성

2. **기반 모델 오케스트레이션 이론**
   - 여러 전문 모델의 조합이 단일 대규모 모델보다 효율적일 수 있다는 이론적 기초 제공

3. **음성 기반 인터페이스의 실현성**
   - 현실적인 수준의 음성 상호작용 시스템 구현 가능성 입증

***

### 8. 결론

**AudioGPT**는 기존의 대규모 언어 모델을 "음성 인식" 시스템으로 확장하는 현실적이고 효율적인 접근법을 제시합니다. 전체 시스템을 재학습하지 않고 기존의 검증된 기반 모델들을 오케스트레이션하는 방식으로, 16가지 오디오 작업을 멀티턴 대화 형태로 수행할 수 있게 했습니다.

그러나 프롬프트 엔지니어링에 대한 의존성, 기반 모델 성능에 대한 의존성, 그리고 토큰 길이 제한 등의 한계를 극복해야 합니다. 향후 연구는 이러한 한계를 해결하면서도 현재 시스템의 장점인 **효율성**, **유연성**, **제로샷 능력**을 유지하는 방향으로 진행되어야 할 것입니다.

특히 음성 합성의 자동화, 멀티모달 정보의 손실 최소화, 도메인 일반화 능력 강화는 향후 연구의 핵심 과제가 될 것으로 예상됩니다.

***

### 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c6ab00a-c48f-4f50-b1aa-504c686ff301/2304.12995v1.pdf)
[2](https://arxiv.org/abs/2304.12995)
[3](https://arxiv.org/pdf/2305.15719.pdf)
[4](https://arxiv.org/pdf/2209.03143.pdf)
[5](https://arxiv.org/pdf/2305.09636.pdf)
[6](https://aclanthology.org/2024.genbench-1.12.pdf)
[7](https://wandb.ai/onlineinference/genai-research/reports/LLM-evaluation-Metrics-frameworks-and-best-practices--VmlldzoxMTMxNjQ4NA)
[8](https://arxiv.org/html/2503.22458v1)
[9](https://www.isca-archive.org/interspeech_2025/wang25ba_interspeech.pdf)
[10](https://aclanthology.org/2024.findings-emnlp.533.pdf)
[11](https://www.semanticscholar.org/paper/ae861f35942721ebe116a01998627774fba3afb1)
[12](https://dl.acm.org/doi/10.1145/3577190.3616117)
[13](https://dl.acm.org/doi/10.1145/3587423.3595503)
[14](https://pubs.aip.org/jasa/article/154/4_supplement/A203/2924347/VSpace-A-browser-based-vowel-synthesiser)
[15](https://arxiv.org/abs/2406.07855)
[16](https://arxiv.org/abs/2411.17690)
[17](https://arxiv.org/abs/2509.22062)
[18](https://arxiv.org/abs/2509.24391)
[19](https://dl.acm.org/doi/10.1145/3719027.3765567)
[20](https://arxiv.org/pdf/2304.12995.pdf)
[21](https://arxiv.org/pdf/2310.00704.pdf)
[22](https://arxiv.org/pdf/2502.03128.pdf)
[23](http://arxiv.org/pdf/2310.04673.pdf)
[24](https://aclanthology.org/2023.findings-emnlp.1055.pdf)
[25](http://arxiv.org/pdf/2306.12925.pdf)
[26](http://arxiv.org/pdf/2312.09911.pdf)
[27](https://arxiv.org/pdf/2412.11449.pdf)
[28](https://www.youtube.com/watch?v=nlI9dpp06iA)
[29](https://arxiv.org/html/2408.01319v1)
[30](https://mirascope.com/blog/prompt-evaluation)
[31](https://openai.com/index/introducing-our-next-generation-audio-models/)
[32](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/)
[33](https://arxiv.org/html/2511.01299v1)
[34](https://asee.ro/wp-content/uploads/2025/04/How-to-choose-rightAI-foundation-models_compressed.pdf)
[35](http://arxiv.org/abs/2304.12995)
[36](https://arxiv.org/html/2506.08967v1)
[37](https://arxiv.org/html/2510.22455v1)
[38](https://arxiv.org/html/2511.19829v1)
[39](https://arxiv.org/html/2508.02018v1)
[40](https://arxiv.org/html/2410.13287v4)
[41](https://arxiv.org/html/2509.07526v1)
[42](https://arxiv.org/html/2510.16091v1)
[43](https://arxiv.org/pdf/2307.14335.pdf)
[44](https://www.isca-archive.org/interspeech_2025/singh25b_interspeech.pdf)
[45](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-foundation-model-evaluate-overview.html)
[46](https://www.nature.com/articles/s41746-024-01074-z)
[47](https://arxiv.org/pdf/2306.05284.pdf)
[48](http://arxiv.org/pdf/2406.02315.pdf)
[49](https://arxiv.org/pdf/2301.11325.pdf)
[50](https://ambientartstyles.com/assessing-ai-llms-2/)
[51](https://pmc.ncbi.nlm.nih.gov/articles/PMC12434620/)
[52](https://aclanthology.org/2025.acl-long.682.pdf)
[53](https://arxiv.org/abs/2404.09385)
[54](https://arxiv.org/html/2410.03192v1)
[55](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/audiogpt/)
[56](https://arxiv.org/html/2404.09385v1)
[57](https://www.lgresearch.ai/blog/view?seq=389)
[58](https://arxiv.org/pdf/2507.09834.pdf)
[59](https://arxiv.org/pdf/2404.09385.pdf)
[60](https://arxiv.org/html/2311.09770v3)
[61](https://arxiv.org/html/2410.03335v2)
[62](https://arxiv.org/html/2412.06602v1)
[63](https://arxiv.org/html/2312.14398v2)
[64](https://arxiv.org/html/2406.02430)
[65](https://arxiv.org/html/2503.22275v1)
[66](https://research.google/blog/audiolm-a-language-modeling-approach-to-audio-generation/)
[67](https://palospublishing.com/foundation-models-in-audio-and-speech-tasks/)
[68](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/musiclm/)
[69](https://arxiv.org/abs/2504.04470)
[70](https://arxiv.org/abs/2506.19608)
[71](https://ieeexplore.ieee.org/document/10930109/)
[72](https://ieeexplore.ieee.org/document/11152049/)
[73](https://arxiv.org/abs/2504.20860)
[74](https://arxiv.org/abs/2502.11560)
[75](https://arxiv.org/abs/2407.08966)
[76](https://dl.acm.org/doi/10.1145/3711896.3736911)
[77](https://www.semanticscholar.org/paper/ab4f1985ead4f45b962e623a85aac611243d9c96)
[78](https://ieeexplore.ieee.org/document/10431687/)
[79](https://aclanthology.org/2023.findings-acl.504.pdf)
[80](http://arxiv.org/pdf/2312.00823.pdf)
[81](http://arxiv.org/pdf/2502.11560.pdf)
[82](https://arxiv.org/pdf/2312.15821.pdf)
[83](https://arxiv.org/pdf/2502.09573.pdf)
[84](http://arxiv.org/pdf/2402.09585.pdf)
[85](https://arxiv.org/pdf/2109.00181.pdf)
[86](https://arxiv.org/html/2504.12796v1)
[87](https://research.aimultiple.com/large-language-model-evaluation/)
[88](https://aclanthology.org/2025.findings-acl.851.pdf)
[89](https://openreview.net/pdf/1fc836176ec53017b9e81d5976f650bed4f13c4e.pdf)
[90](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
[91](https://arxiv.org/html/2503.01174v1)
[92](https://whynn.tistory.com/8)
[93](https://arxiv.org/html/2410.18908v6)
[94](https://arxiv.org/html/2506.12285v1)
[95](https://arxiv.org/pdf/2405.08295.pdf)
[96](https://arxiv.org/html/2510.04584v1)
[97](https://arxiv.org/html/2510.07978v1)
[98](https://arxiv.org/html/2507.08128v1)
[99](https://arxiv.org/html/2406.16020v5)
[100](https://arxiv.org/html/2509.23206v1)
[101](https://aisera.com/blog/llm-evaluation/)
[102](https://aclanthology.org/2024.emnlp-main.595.pdf)
