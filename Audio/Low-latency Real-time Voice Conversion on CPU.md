# Low-latency Real-time Voice Conversion on CPU

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
LLVC(Low-latency Low-resource Voice Conversion)는 기존 오디오 조작 및 생성 신경망 아키텍처를 **실시간 any-to-one 음성 변환(voice conversion)** 태스크에 적용하여, **소비자용 CPU에서 20ms 이하의 초저지연(ultra-low-latency)** 으로 동작하면서도 자연스러운 음질과 높은 타겟 화자 유사도를 달성할 수 있음을 보여준다.

### 주요 기여
1. **초저지연 스트리밍 음성 변환**: 16kHz 비트레이트에서 약 20ms 미만의 지연 시간을 달성하며, CPU에서 실시간 대비 약 2.8배 빠르게 동작
2. **Knowledge Distillation 기반 병렬 데이터셋 구축**: 고품질 teacher 모델(RVC v2)로부터 합성 병렬 데이터셋을 생성하여 경량 student 모델을 학습
3. **Waveformer 기반 스트리밍 아키텍처 적용**: 실시간 사운드 추출용으로 설계된 Waveformer의 인과적(causal) 인코더-디코더를 음성 변환에 최초 적용
4. **GAN 아키텍처와 다중 손실 함수의 결합**: VITS 기반 multi-period discriminator와 mel spectrogram, HuBERT 기반 자기지도 표현 손실을 함께 활용
5. **오픈소스 공개**: 코드, 사전학습 가중치, 샘플을 모두 공개하여 접근성 확보

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 실시간 음성 변환 모델들(MMVC, so-vits-svc, DDSP-SVC, RVC, QuickVC 등)은 다음과 같은 근본적 한계를 가지고 있다:

- **스트리밍 비적합성**: 기존 모델들은 저지연 스트리밍 오디오에서 동작하도록 설계되지 않았다. 짧은 오디오 세그먼트를 순차적으로 변환하면 음질이 심각하게 저하되므로, 이전 오디오 컨텍스트를 접두사(prefix)로 붙여야 하며, 이는 연산 효율성을 희생시킨다.
- **높은 연산 자원 요구**: 대부분의 모델이 GPU를 필요로 하며, 사전학습된 인코더(contentvec, hubert-soft 등)의 추론 비용이 높다.
- **고지연**: 피치 추정(F0 estimation) 등의 전처리 단계가 병목을 생성하여 실시간 사용에 부적합하다.

LLVC는 **GPU 없이 소비자용 CPU에서 20ms 미만의 지연으로 스트리밍 방식의 음성 변환**을 수행할 수 있는 모델을 제공하는 것을 목표로 한다.

### 2.2 제안하는 방법

#### (1) Knowledge Distillation을 통한 합성 병렬 데이터셋 생성

비병렬(non-parallel) 데이터에서 학습된 대규모 teacher 모델(RVC v2)을 사용하여 합성 병렬 데이터셋을 생성한다. LibriSpeech clean 360시간 데이터셋의 922명 화자 음성을, LibriSpeech 화자 8312의 39분 오디오로 학습된 RVC v2 모델을 통해 단일 타겟 화자 음성으로 변환한다. 이렇게 생성된 (입력 음성, 타겟 음성) 쌍은 시간 정렬(time-aligned)된 병렬 데이터셋을 구성하며, 경량 student 모델(LLVC)의 학습에 사용된다.

#### (2) 스트리밍 추론 수식

LLVC의 스트리밍 추론은 Waveformer의 청크 기반 추론(chunk-based inference with lookahead)을 따른다. 총 지연 시간(latency)은 다음과 같이 계산된다:

```math
\text{Latency} = \frac{\texttt{dec\_chunk\_len} \times L + 2L}{F_s} \quad \text{(seconds)}
```

여기서:
- $\text{dec chunk len}$: 디코더 청크 길이
- $L$: 기본 샘플 단위 크기
- $F_s$: 오디오 샘플레이트 (Hz, 본 논문에서는 16,000 Hz)
- $2L$: lookahead에 필요한 추가 샘플

$N$개의 청크를 한 번에 처리하면 지연 시간이 증가하지만 Real-Time Factor(RTF)가 개선된다.

#### (3) 손실 함수(Loss Function)

생성자(Generator)의 손실 함수는 다음의 가중 합으로 구성된다:

$$\mathcal{L}_G = \mathcal{L}_{\text{GAN}} + \mathcal{L}_{\text{feature}} + \mathcal{L}_{\text{mel}} + \mathcal{L}_{\text{SSL}}$$

- $\mathcal{L}_{\text{GAN}}$: VITS의 생성자 적대적 손실
- $\mathcal{L}_{\text{feature}}$: VITS의 feature matching 손실
- $\mathcal{L}_{\text{mel}}$: Multi-resolution mel spectrogram 손실 (auraloss 라이브러리 기반)
- $\mathcal{L}_{\text{SSL}}$: 사전학습된 fairseq HuBERT Base 모델의 특징 간 L1 거리 기반 자기지도 표현 손실 (Close et al., 2023에서 영감)

판별자(Discriminator)는 VITS의 판별자 손실을 그대로 사용한다.

### 2.3 모델 구조

#### Generator (생성자)

| 구성 요소 | 세부 사항 |
|---------|---------|
| **Causal Convolution Prenet** | 인과적 합성곱으로 구성된 전처리 네트워크. Gated activation (Tanh × Sigmoid) + Dropout + 잔차 블록(Residual Block) × N |
| **Conv1D** | 입력 웨이브폼의 초기 합성곱 처리 (소량의 미래 컨텍스트 접근 제공) |
| **DCC Encoder** | Waveformer의 Dilated Causal Convolution 인코더 (512차원, 깊이 8 레이어, lookahead 16 샘플) |
| **Transformer Decoder** | Waveformer의 마스크드 트랜스포머 디코더 (256차원, 과거 및 현재 토큰에만 어텐션) |
| **ConvTranspose1D** | Prenet 출력과 디코더 출력의 결합 후 변환된 웨이브폼 생성 |

핵심적으로 Prenet 출력과 Conv1D 출력이 더해지고(skip connection), DCC Encoder 출력과 ConvTranspose1D 출력이 곱해져(masking) 최종 변환 웨이브폼을 생성한다.

#### Discriminator (판별자)

VITS의 Multi-Period Discriminator(MPD)를 채택하되, RVC v2에서 영감을 받아 판별 주기를 $[2, 3, 5, 7, 11, 17, 23, 37]$로 확장하였다.

### 2.4 성능 향상

#### 추론 성능 비교

| 모델 | End-to-End Latency (ms) | RTF |
|------|------------------------|-----|
| No-F0 RVC | 189.772 | 1.114 |
| QuickVC | 97.616 | 1.050 |
| **LLVC (ours)** | **19.696** | **2.769** |
| LLVC-NC (ours) | **18.327** | **3.677** |
| LLVC-HFG (ours) | 19.563 | 2.850 |

LLVC는 기존 최선 모델(QuickVC) 대비 **지연 시간을 약 5배 감소**시키면서도 **RTF를 약 2.6배 향상**시켰다.

#### 주관적 평가 (MOS)

| 모델 | Naturalness | Similarity |
|------|-------------|------------|
| Ground Truth | 3.70 | 3.88 |
| No-F0 RVC | 3.58 | 3.35 |
| QuickVC | 3.28 | 3.26 |
| **LLVC (ours)** | 3.78 | 3.83 |
| **LLVC-HFG (ours)** | **3.88** | **3.90** |

LLVC-HFG 변형은 자연스러움과 유사도 모두에서 **Ground Truth를 상회**하는 결과를 보여주었다.

#### 객관적 평가

| 모델 | Resemblyzer | WVMOS |
|------|------------|-------|
| Ground Truth | 0.898 | 3.854 |
| No-F0 RVC | **0.846** | 2.465 |
| LLVC (ours) | 0.829 | 3.605 |
| LLVC-NC (ours) | 0.821 | **3.677** |

WVMOS 기준으로 LLVC 변형들이 비교 모델 대비 압도적으로 높은 음질 점수를 달성하였다.

### 2.5 한계

1. **영어 단일 언어 한정**: 학습 데이터가 깨끗한 영어 음성(LibriSpeech)으로만 구성되어 다국어 일반화가 검증되지 않음
2. **16kHz 제한**: 샘플레이트가 16kHz로 제한되어 고품질 오디오(24kHz, 48kHz)에 대한 지원 부재
3. **Any-to-one 방식**: 단일 타겟 화자로만 변환 가능하며, any-to-many 변환은 지원하지 않음
4. **Teacher 모델 의존성**: 합성 병렬 데이터의 품질이 teacher 모델(RVC v2)의 변환 품질에 의존
5. **노이즈 환경 미검증**: 깨끗한 음성에서만 학습·평가되어 실환경 노이즈에 대한 강건성이 불확실
6. **소규모 주관적 평가**: MOS 평가가 15명 피험자, 4개 발화로 제한적

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능의 근거

LLVC는 LibriSpeech clean 360시간 데이터셋의 **922명 화자**로부터 학습하여 any-to-one 변환을 수행한다. Dev-clean 분할(학습 시 미포함 화자)에서의 검증을 통해 **미확인 입력 화자(unseen input speakers)** 에 대한 일반화 가능성을 시사한다. 또한 test-clean 데이터셋의 2620개 파일에 대한 객관적 평가에서 일관된 성능을 보여준다.

### 3.2 일반화 성능 향상을 위한 구체적 방안

#### (1) 다국어 데이터 확장
논문은 다음과 같이 언급한다:
> "Our choice of training data contained only clean English speech, even though our method of constructing the parallel dataset is **language-independent** and relatively robust to noise."

Knowledge distillation을 통한 병렬 데이터셋 구축 방법 자체는 언어에 독립적이므로, **다국어 음성 데이터**(예: CommonVoice, VCTK의 다양한 악센트 등)를 포함하면 교차 언어 일반화 성능을 향상시킬 수 있다.

#### (2) 노이즈 강건성 확보
깨끗한 음성뿐만 아니라 **잡음이 포함된 음성 데이터**를 학습에 포함시키면, 실제 사용 환경(카페, 거리, 회의실 등)에서의 일반화 성능이 크게 향상될 수 있다. 논문의 병렬 데이터셋 구축 파이프라인은 노이즈에 상대적으로 강건하다고 저자들은 주장한다.

#### (3) 개인화 파인튜닝
논문은 단일 입력 화자만으로 구성된 데이터셋에 파인튜닝하여 **개인화된 음성 변환 모델**을 만들 수 있다고 제안한다. 이는 특정 사용자에 대한 일반화 성능을 극대화하는 방향이다.

#### (4) 데이터 규모 확장의 용이성
합성 병렬 데이터셋은 teacher 모델의 추론을 통해 **임의의 규모로 확장 가능**하므로, 데이터 양을 늘려 다양한 음성 특성(피치, 속도, 악센트, 감정 등)에 대한 일반화를 개선할 수 있다.

#### (5) Teacher 모델 품질 향상
더 고품질의 teacher 모델을 사용하면 합성 병렬 데이터의 품질이 향상되어, student 모델(LLVC)의 일반화 성능 상한이 올라간다. Any-to-many teacher 모델을 사용하면 다중 타겟 화자로의 확장도 가능하다.

#### (6) 아키텍처적 개선 가능성
- **Causal Convolution Prenet의 깊이 조절**: 잔차 블록 수(N)를 늘려 더 넓은 수용 영역(receptive field)을 확보하면 다양한 입력 특성에 대한 적응력이 향상될 수 있다 (지연 증가와의 트레이드오프 존재)
- **DCC Encoder 깊이 복원**: 현재 10→8로 줄인 인코더 깊이를 복원하거나 조절하여 표현력과 일반화 간의 최적점을 탐색

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (1) Knowledge Distillation 패러다임의 확산
이 논문은 **비병렬 데이터에서 학습된 고품질 teacher 모델 → 합성 병렬 데이터셋 생성 → 경량 student 모델 학습**이라는 파이프라인이 음성 변환에서 매우 효과적임을 입증하였다. 이 패러다임은 음성 합성(TTS), 음성 향상(speech enhancement), 음성 분리(speech separation) 등 관련 분야로 확장될 수 있다.

#### (2) 스트리밍 아키텍처 설계의 중요성 부각
기존 음성 변환 모델들이 오프라인 처리에 최적화된 반면, LLVC는 **인과적 합성곱과 마스크드 트랜스포머를 결합한 스트리밍 네이티브 아키텍처**의 중요성을 보여주었다. 이는 향후 실시간 오디오 처리 모델 설계의 기준점이 될 수 있다.

#### (3) 엣지/모바일 배포 가능성
CPU에서 실시간 대비 2.8배 빠르게 동작한다는 것은 **스마트폰, 태블릿, IoT 기기**에서의 음성 변환이 실용적으로 가능함을 시사한다. 이는 음성 익명화, 실시간 더빙, 접근성 도구 등의 응용에 직접적 영향을 미친다.

#### (4) 오픈소스 생태계 기여
코드와 모델 가중치의 공개는 후속 연구자들이 쉽게 벤치마킹하고 개선할 수 있는 기반을 제공한다.

### 4.2 앞으로 연구 시 고려할 점

1. **샘플레이트 확장**: 16kHz → 24kHz/48kHz로의 확장이 필요하며, 이는 모델 크기와 지연 시간 증가를 수반할 수 있다
2. **Any-to-many 확장**: 단일 타겟 화자 제한을 극복하여 다중 타겟 화자 또는 제로샷(zero-shot) 음성 변환으로 확장하는 연구가 필요
3. **피치(F0) 제어**: 현재 모델은 피치 정보를 명시적으로 다루지 않으므로, 이성 간 음성 변환 등에서 한계가 있을 수 있다
4. **더 강력한 주관적/객관적 평가**: 15명, 4개 발화의 소규모 MOS 평가를 넘어 대규모 청취 실험과 다양한 객관적 지표(PESQ, POLQA, 화자 검증 EER 등)가 필요
5. **Teacher 모델의 오류 전파 문제**: Knowledge distillation 기반 접근에서 teacher 모델의 아티팩트가 student 모델에 전파되는 문제를 체계적으로 분석할 필요가 있다
6. **적대적 강건성**: 실시간 응용에서 비정상적 입력(극단적 잡음, 비음성 오디오)에 대한 안정성 검증이 필요
7. **모델 양자화/경량화**: ONNX, TensorRT, 정수 양자화 등을 통한 추가적인 추론 최적화 연구가 가능

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 접근 방식 | 실시간 여부 | CPU 동작 | 지연 시간 | 주요 특징 |
|------|------|---------|-----------|---------|---------|---------|
| **LLVC** (본 논문) | 2023 | Waveformer + KD + GAN | ✅ 스트리밍 | ✅ | ~20ms | 초저지연, 경량, 오픈소스 |
| **RVC** (RVC-Project) | 2023 | ContentVec + HiFi-GAN + Retrieval | ⚠️ 적응형 | ⚠️ (느림) | ~190ms | Retrieval 기반 품질 향상, 피치 추정 병목 |
| **QuickVC** (Guo et al.) | 2023 | VITS + iSTFT vocoder | ⚠️ 적응형 | ✅ | ~98ms | iSTFT 기반 경량 보코더 |
| **so-vits-svc** | 2023 | SoftVC + VITS | ⚠️ 적응형 | ❌ (GPU 권장) | 높음 | 노래 음성 변환에 특화 |
| **VITS** (Kim et al.) | 2021 | CVAE + Normalizing Flow + GAN | ❌ | ❌ | - | 엔드투엔드 TTS, 많은 VC 모델의 기반 |
| **AutoVC** (Qian et al.) | 2019/2020 | Autoencoder + Bottleneck | ❌ | ❌ | - | 제로샷 스타일 전이, 병목 기반 |
| **kNN-VC** (Baas et al.) | 2023 | k-최근접 이웃 + WavLM | ❌ | ❌ | - | 학습 불필요(training-free), 비병렬 |
| **EnCodec** (Défossez et al.) | 2022 | Neural Audio Codec | ✅ 스트리밍 | ✅ | 낮음 | 오디오 코덱(VC가 아닌 압축 목적) |
| **AGAIN-VC** (Chen et al.) | 2020 | Adaptive Instance Norm | ❌ | ❌ | - | 원샷 음성 변환, 활성화 가이던스 |
| **ContentVec** (Qian et al.) | 2022 | Self-supervised 표현 학습 | - | - | - | 화자 분리된 음성 표현, 다수 VC 모델의 인코더로 사용 |

### 핵심 비교 분석

**1. 스트리밍 네이티브 설계 vs. 적응형 스트리밍**
- 기존 모델(RVC, QuickVC, so-vits-svc)은 오프라인 처리용으로 설계된 후 컨텍스트 버퍼링을 통해 스트리밍에 "적응"된 반면, LLVC는 처음부터 **인과적(causal) 아키텍처**로 설계되어 중간 계산을 캐시할 수 있다. 이것이 지연 시간과 RTF에서의 압도적 차이의 근본 원인이다.

**2. 사전학습 인코더 의존성 탈피**
- RVC, so-vits-svc, QuickVC 등은 ContentVec이나 HuBERT-soft 같은 **대규모 사전학습 인코더**에 의존하여 추론 시 상당한 연산 오버헤드가 발생한다. LLVC는 knowledge distillation을 통해 이러한 의존성을 학습 시간으로 이동시켜, 추론 시에는 경량 아키텍처만 사용한다.

**3. kNN-VC와의 차별점**
- kNN-VC(Baas et al., 2023)는 학습 없이 k-최근접 이웃 검색만으로 음성 변환을 수행하지만, WavLM 특징 추출과 검색 과정의 연산 비용으로 인해 실시간 스트리밍에 부적합하다.

**4. EnCodec과의 관계**
- EnCodec은 스트리밍 오디오 코덱으로서 유사한 저지연 설계 원칙을 공유하지만, 인코더가 화자 정체성을 보존하도록 설계되어 음성 변환에는 직접 사용할 수 없다. LLVC는 이러한 스트리밍 코덱의 설계 원칙을 음성 변환에 성공적으로 차용한 사례이다.

**5. 품질-효율 트레이드오프**
- Resemblyzer 유사도에서는 No-F0 RVC(0.846)가 LLVC(0.829)보다 우수하지만, WVMOS 품질에서는 LLVC(3.605)가 No-F0 RVC(2.465)를 크게 상회한다. 이는 LLVC가 지연 시간과 연산 효율을 극단적으로 줄이면서도 **음질 측면에서는 오히려 향상**을 달성했음을 의미한다.

---

## 참고자료

1. Sadov, K., Hutter, M., & Near, A. (2023). "Low-latency Real-time Voice Conversion on CPU." *arXiv:2311.00873v1 [cs.SD]*. https://arxiv.org/abs/2311.00873
2. Veluri, B. et al. (2023). "Real-time target sound extraction." (Waveformer) - 논문 참고문헌 [26]
3. Kim, J., Kong, J., & Son, J. (2021). "Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech." (VITS) - 논문 참고문헌 [12]
4. Guo, H. et al. (2023). "QuickVC: Any-to-many voice conversion using inverse short-time Fourier transform for faster conversion." - 논문 참고문헌 [8]
5. Baas, M., van Niekerk, B., & Kamper, H. (2023). "Voice conversion with just nearest neighbors." - 논문 참고문헌 [3]
6. Défossez, A. et al. (2022). "High fidelity neural audio compression." (EnCodec) - 논문 참고문헌 [6]
7. Qian, K. et al. (2022). "ContentVec: An improved self-supervised speech representation by disentangling speakers." - 논문 참고문헌 [18]
8. Close, G. et al. (2023). "Perceive and predict: Self-supervised speech representation based loss functions for speech enhancement." - 논문 참고문헌 [5]
9. Gou, J. et al. (2020). "Knowledge distillation: A survey." - 논문 참고문헌 [7]
10. LLVC GitHub Repository: https://github.com/KoeAI/LLVC
11. RVC Project: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
12. Walczyna, T. & Piotrowski, Z. (2023). "Overview of voice conversion methods based on deep learning." *Applied Sciences*, 13(5):3100. - 논문 참고문헌 [27]
13. Panayotov, V. et al. (2015). "Librispeech: An ASR corpus based on public domain audio books." - 논문 참고문헌 [15]
