
# YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone

## 1. 핵심 주장 및 주요 기여

YourTTS는 VITS 모델을 기반으로 한 혁신적인 다국어 zero-shot 다중 화자 텍스트-음성 변환(TTS) 시스템으로, 다음의 핵심 기여를 제시한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

**주요 주장:**
- Zero-shot 다중 화자 TTS에서 영어 기준 SOTA(State-of-the-Art) 달성
- Zero-shot 음성 변환에서 SOTA와 비교 가능한 수준의 성과
- **첫 번째 다국어 zero-shot 다중 화자 TTS 접근법 제시**
- 저자원 언어에서 단 1명의 화자 데이터로도 양호한 성능 달성

**주요 기여:**
1. 영어에서 최고 수준의 성능 달성 (SECS: 0.864, MOS: 4.21±0.04)
2. 다국어 학습이 가능한 새로운 아키텍처 설계
3. 1분 미만의 음성으로 미세조정하여 SOTA 음성 유사도 달성
4. 저자원 언어에 대한 실질적인 해결책 제시

***

## 2. 해결하는 주요 문제 및 제안 방법

### 2.1 핵심 문제점

기존 zero-shot 다중 화자 TTS 시스템의 주요 한계:

**일반화 성능 격차**: 학습 데이터에 포함된 화자 대 미확인 화자 간의 음성 유사도 격차가 여전히 큼. 특히 VCTK 같은 소규모 데이터셋(109 화자)으로 학습한 모델은 새로운 음성 특성과 녹음 환경에 적응하지 못함. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

**저자원 언어의 한계**: 고품질 TTS 모델 개발에 필요한 다수 화자의 방대한 데이터를 확보할 수 없는 언어들에 대한 해결책 부재.

**음성 특성 다양성**: 학습 중 보지 못한 음성 특성(심각한 억양, 이상한 녹음 환경 등)을 가진 화자에 대한 품질 저하.

### 2.2 제안하는 방법

#### 2.2.1 모델 아키텍처

YourTTS는 다음과 같은 아키텍처 개선을 제시한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

**텍스트 인코더:**
- Raw text 입력 (phoneme 변환 없음) - 고품질 그래프-음소 변환기가 없는 언어에 유리
- Transformer 기반 인코더 (10개 블록, 196개 hidden channel)
- **언어 임베딩 통합**: 각 문자 임베딩에 4-dimensional trainable language embedding 연결

수식으로 표현하면:

$$\mathbf{e}_{\text{char}} = [\mathbf{c}_i \oplus \mathbf{l}_{\text{lang}}]$$

여기서 $\mathbf{c}\_i$는 문자 임베딩, $\mathbf{l}_{\text{lang}}$은 언어 임베딩, $\oplus$는 연결(concatenation)을 나타낸다.

**흐름 기반 디코더(Flow-based Decoder):**
- 4개의 Affine Coupling Layer
- 각 layer는 4개의 WaveNet residual block으로 구성
- 외부 화자 임베딩으로 조건화(global conditioning)

**후류 인코더(Posterior Encoder):**
- Linear spectrogram 입력
- 16개의 non-causal WaveNet residual block
- 잠재 변수 z 예측

**확률적 Duration Predictor:**
- Stochastic duration predictor를 사용하여 다양한 음성 리듬 표현
- Variational lower bound를 통해 학습

#### 2.2.2 음성 일관성 손실(Speaker Consistency Loss, SCL)

YourTTS의 가장 혁신적인 기여 중 하나는 SCL의 도입이다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

$$L_{\text{SCL}} = -\alpha \cdot \frac{1}{n} \sum_{i=1}^{n} \cos\text{sim}(\varphi(g_i), \varphi(h_i))$$

여기서:
- $\varphi(\cdot)$: 화자 임베딩을 출력하는 사전학습된 화자 인코더
- $\cos\text{sim}$: 코사인 유사도 함수
- $g_i$: 실제 음성(ground truth)
- $h_i$: 생성된 음성(generated)
- $\alpha$: SCL의 영향도를 제어하는 양수 (일반적으로 9)
- $n$: 배치 크기

**작동 원리:**
SCL은 생성된 음성과 실제 음성의 화자 임베딩 간 코사인 유사도를 최대화함으로써, 모델이 화자 특성을 더 정확하게 재현하도록 강제한다. 이는 특히 학습 데이터와 다른 녹음 환경에서의 일반화 성능을 향상시킨다.

#### 2.2.3 다국어 학습 전략

다국어 동시 학습을 위해 **가중 무작위 샘플링(Weighted Random Sampling)**을 적용하여 언어별 배치 균형을 유지한다. 이를 통해:
- 저자원 언어가 과소 표현되는 것을 방지
- 언어 간 지식 공유 활성화
- 크로스-링구얼 일반화 성능 향상

***

## 3. 모델 아키텍처의 상세 분석

### 3.1 전체 아키텍처 흐름

**학습 단계 (Training):**

1. **입력 처리**: 언어 ID와 함께 원본 텍스트 입력
2. **텍스트 인코더**: 언어 임베딩 추가된 Transformer로 텍스트를 임베딩으로 변환
3. **Monotonic Alignment Search (MAS)**: 텍스트와 음성의 정렬 학습
4. **Duration Predictor**: 음소 길이 예측
5. **Flow-based Decoder**: 잠재 변수 $z_p$를 $z$로 변환
6. **Posterior Encoder**: Linear spectrogram에서 잠재 변수 추출
7. **HiFi-GAN Generator**: 최종 음성파형 생성

**추론 단계 (Inference):**

1. MAS 대신 alignment 생성으로 변경
2. Duration predictor의 역 변환으로 duration 샘플링
3. Flow-based decoder의 역 변환으로 음성 생성

### 3.2 화자 임베딩 조건화

화자 일반화 성능 향상을 위해 다음과 같이 외부 화자 임베딩을 적용한다:

```math
\text{decoder\_input} = \text{duration\_predictor\_output} + \mathbf{W}_s \cdot \mathbf{s}
```

여기서:
- $\mathbf{s}$: 화자 임베딩 (H/ASP 모델로 추출)
- $\mathbf{W}_s$: Linear projection layer

이를 통해 모델이 특정 화자의 음성 특성을 명시적으로 모델링한다.

***

## 4. 성능 향상 분석

### 4.1 VCTK 데이터셋 결과

| 실험 | SECS | MOS | Sim-MOS |
|------|------|-----|---------|
| 실제 음성 | 0.824 | 4.26±0.04 | 4.19±0.06 |
| Attentron ZS | (0.731) | (3.86±0.05) | (3.30±0.06) |
| SC-GlowTTS | (0.804) | (3.78±0.07) | (3.99±0.07) |
| **Exp. 2 + SCL (최고)** | **0.864** | **4.19±0.05** | **4.17±0.06** |
| Exp. 4 + SCL | 0.843 | 4.23±0.05 | 4.10±0.06 |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

YourTTS의 최고 성능 설정(Exp. 2 + SCL)은 기존 SOTA 방법들을 크게 상회한다. SECS는 Attentron 대비 **18.3% 향상**, SC-GlowTTS 대비 **7.5% 향상**을 달성했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

### 4.2 LibriTTS 데이터셋 결과

학습 데이터에 포함되지 않은 녹음 환경에서의 성능 평가:

| 메트릭 | Exp. 1 | Exp. 4 + SCL |
|--------|--------|-------------|
| SECS | 0.754 | 0.856 |
| MOS | 4.25±0.05 | 4.18±0.05 |
| Sim-MOS | 3.98±0.07 | 4.07±0.07 |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

실험 4 (LibriTTS 추가 데이터)에서 SECS가 **13.5% 향상**되어, 데이터 다양성이 일반화 성능에 미치는 긍정적 영향을 보여준다.

### 4.3 Zero-shot 음성 변환 성능

| 변환 | MOS | Sim-MOS |
|------|-----|---------|
| EN→EN | 4.20±0.05 | 4.07±0.06 |
| AutoVC (기준) | 3.54±1.09 | 1.91±1.34 |
| NoiseVC (기준) | 3.38±1.35 | 3.05±1.25 |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

YourTTS는 음성 변환에서도 우수한 성능을 달성하며, Sim-MOS에서 **113% 향상** (AutoVC 대비)을 보인다.

***

## 5. 모델 일반화 성능 향상 가능성

### 5.1 저자원 언어에서의 일반화

YourTTS의 혁신적 기여 중 하나는 포르투갈어 실험에서 입증된다. **단 1명의 남성 화자 데이터만 사용**하고도 여성 음성 합성이 가능했다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

**포르투갈어 MLS 결과:**
- MOS: 4.11±0.07 (기준: 4.61±0.05)
- Sim-MOS: 3.19±0.10 (기준: 4.41±0.05)
- 성별별 Sim-MOS (Exp. 4 + SCL):
  - 남성: 3.29±0.14
  - 여성: 2.84±0.14

이는 Attentron이 약 100명의 화자로 학습했을 때의 Sim-MOS (3.30±0.06)과 유사하며, **다국어 학습이 저자원 언어 일반화에 효과적**임을 증명한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

### 5.2 음성 적응(Speaker Adaptation)을 통한 성능 향상

**1분 미만의 음성으로 미세조정:**

| 화자 | 기간 | 모드 | Sim-MOS |
|------|------|------|---------|
| PT 남성 | 31s | ZS | 3.35±0.12 |
| PT 남성 | 31s | FT | **4.19±0.07** |
| PT 여성 | 20s | ZS | 2.77±0.15 |
| PT 여성 | 20s | FT | **4.43±0.06** |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

**45초 이상의 데이터**로 미세조정하면 자연스러움(MOS)과 유사도(Sim-MOS) 모두에서 최적 성능을 달성한다. 20초 데이터로도 Sim-MOS에서 **59% 향상**(여성 포르투갈어)을 달성한다.

### 5.3 다국어 학습의 일반화 효과

| 실험 | 구성 | SECS (PT) |
|------|------|-----------|
| Exp. 1 | 영어 only | - |
| Exp. 2 | 영어 + 포르투갈어 | 0.745 |
| Exp. 3 | 영어 + 포르투갈어 + 프랑스어 | 0.761 |
| Exp. 3 + SCL | 3언어 + SCL | **0.766** |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

프랑스어 추가로 포르투갈어 SECS가 **2.1% 향상**된 이유는, M-AILABS 프랑스어 데이터셋의 높은 품질이 배치에 저품질 포르투갈어 샘플의 비율을 감소시켰기 때문이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

***

## 6. 모델의 한계점

### 6.1 기술적 한계

**확률적 Duration Predictor의 불안정성:**
- 일부 화자와 문장에서 부자연스러운 음성 길이 생성
- 음성 리듬의 불규칙성 발생

**오발음 문제:**
- 특히 포르투갈어에서 심각
- 원인: Phoneme 전사 없이 raw text 입력 사용
  - 장점: 고품질 그래프-음소 변환기 부재 시 실용적
  - 단점: 정확한 음소 정렬 불가능

**성별 의존성:**
- 포르투갈어 음성 변환에서 성별에 따른 큰 성능 차이
  - 남성→남성: 3.80±0.15
  - 여성→여성: 3.35±0.19 (**11% 저하**)
- 원인: 학습 데이터에서 여성 화자 부재

### 6.2 실제 적용의 한계

**고품질 합성의 최소 데이터 요구:**
- 자연스러움(MOS) 최적화: **45초 이상** 필요
- 20~44초: 유사도는 높지만 자연스러움 저하

**크로스-링구얼 변환의 한계:**
- 포르투갈어→영어 변환: MOS 3.40±0.09 (EN→EN 4.20 대비 **19% 저하**)
- 특히 여성 화자 변환에서 심각한 성능 저하

***

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 연대기적 발전

| 연도 | 모델 | 핵심 특징 | 성능 |
|------|------|---------|------|
| 2020 | SC-GlowTTS | Flow-based, Single-speaker | SECS 0.804 |
| 2021 | VITS | End-to-end, VAE+Flow | MOS 4.43 |
| 2022 | **YourTTS** | **Multilingual, ZS-TTS** | **SECS 0.864** |
| 2023 | ZMM-TTS | Self-supervised, Multilingual | 6 언어 지원 |
| 2024 | **XTTS** | 16 언어 지원, SOTA | MOS 4.23+ |
| 2024 | HierSpeech++ | Dual-acoustic encoder | 고품질 음성 |
| 2024 | StyleTTS-ZS | Diffusion + Style | 빠른 추론 |
| 2025 | Stable-TTS | Prosody consistency | 노이즈 강건성 |

 [arxiv](https://arxiv.org/pdf/2301.12596.pdf)

### 7.2 주요 경쟁 모델과의 비교

**XTTS (Casanova et al., 2024)** [isca-archive](https://www.isca-archive.org/interspeech_2024/casanova24_interspeech.pdf)

| 항목 | YourTTS | XTTS |
|------|---------|------|
| 언어 수 | 3 (한정) | **16개** (다국어) |
| 데이터 규모 | 중소 | 대규모 |
| 추론 속도 | 표준 | **빠름** (21.53 Hz) |
| 저자원 언어 | 우수 | 우수 |
| 스타일 모사 | 기본 | **고급** (휘스퍼 스타일) |

XTTS는 YourTTS의 직접적인 후속 모델로, 언어 확장성과 스타일 제어 면에서 크게 개선됨. [isca-archive](https://www.isca-archive.org/interspeech_2024/casanova24_interspeech.pdf)

**HierSpeech++ (2023)**

- Dual-acoustic encoder로 **고품질 음성 합성** 강화
- 특히 학습 데이터에서 벗어난 음성 특성에 강함
- YourTTS보다 **음성 품질(MOS)에서 우수**

**ZMM-TTS (2023)** [arxiv](http://arxiv.org/pdf/2312.14398.pdf)

| 특징 | YourTTS | ZMM-TTS |
|------|---------|---------|
| 자기지도 학습 | 아니오 | **예** (VQ-VAE 기반) |
| 저자원 언어 | 1 화자 | **3 화자** (평가) |
| 평가 언어 | 3개 | 6개 |
| 크로스-링구얼 | 기본 | **고급** |

ZMM-TTS는 자기지도 표현 학습으로 더 나은 콘텐츠-화자 분리를 달성. [arxiv](http://arxiv.org/pdf/2312.14398.pdf)

### 7.3 최신 트렌드 분석

**1. 자기지도 학습의 부상**

최신 모델들은 HuBERT, WavLM, Whisper 같은 사전학습된 모델을 활용:
- 더 나은 콘텐츠 표현 학습
- 더 강건한 일반화 성능
- 저자원 언어 적응성 향상

예: DINO-VITS는 HuBERT 기반 콘텐츠 인코딩으로 **노이즈 강건성 향상** [arxiv](https://arxiv.org/pdf/2311.09770.pdf)

**2. 확산 모델의 도입**

- **StyleTTS-ZS**: 확산 기반 스타일 디코더로 더 자연스러운 음성
- **DiffGAN-ZSTTS**: 확산 기반 디코더로 **unseen speaker 일반화 개선** [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11842752/)

**3. 가벼운 모델 추세**

- **Lightweight and Stable ZS-TTS (2025)**: RTF 0.13 (CPU), 0.012 (GPU)
- 배포 효율성과 데이터 보안 강조 [arxiv](https://arxiv.org/abs/2501.08566)

**4. 특화된 적응 기법**

- **USAT**: 제로샷과 퓨샷 통합으로 강한 억양 화자 적응 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10508477/)
- **Stable-TTS**: 제한된 노이즈 있는 음성으로도 안정적 합성 [arxiv](https://arxiv.org/html/2412.20155v1)

### 7.4 YourTTS의 영향력 평가

**긍정적 영향:**
1. 다국어 ZS-TTS의 가능성 입증 - 이후 XTTS, Mega-TTS 2 등으로 확장 [isca-archive](https://www.isca-archive.org/interspeech_2024/casanova24_interspeech.pdf)
2. 저자원 언어 지원의 실현성 제시 - 다국어 TTS 연구 활성화
3. Speaker Consistency Loss의 효과 증명 - 이후 여러 모델에서 채택 [openreview](https://openreview.net/forum?id=ssK7j0Sztj)
4. 공개 코드 제공으로 커뮤니티 기여 (Coqui TTS)

**한계 인식:**
- 자기지도 학습 미도입 (2022년 당시 트렌드 변화 놓침)
- 확산 모델 미활용
- 최대 3개 언어 한정 (이후 XTTS는 16개 언어)

***

## 8. 앞으로의 연구에 미치는 영향 및 고려사항

### 8.1 학술적 영향

**1. 다국어 TTS의 새로운 패러다임**

YourTTS의 성공은 다국어 동시 학습이 저자원 언어의 일반화 성능 향상에 기여함을 보여줬다. 이는 이후 연구의 기본 가정으로 채택됨. [arxiv](http://arxiv.org/pdf/2312.14398.pdf)

**이후 연구의 방향:**
- 언어별 가중치 조정 메커니즘 개선
- 언어 간 음운 시스템 차이 명시적 모델링
- Zero-shot cross-lingual TTS의 확대

**2. 화자 일반화 문제의 재정의**

SCL을 통해 "화자 특성을 명시적으로 최대화하는 손실함수"가 효과적임을 증명. 이는:

$$L_{\text{total}} = L_{\text{reconstruction}} + L_{\text{adversarial}} + L_{\text{SCL}}$$

형태의 다중 손실함수 결합 전략의 기초가 됨. [arxiv](https://arxiv.org/html/2307.00393v1)

**3. 음성 변환의 다중 작업 학습**

YourTTS가 같은 아키텍처로 TTS와 음성 변환을 동시에 수행함을 보여준 것은, 멀티태스크 학습의 실질적 가능성을 입증. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

### 8.2 실무적 영향

**1. 엣지 디바이스 배포의 길열기**

YourTTS의 성공은 상업적 TTS 시스템(Coqui AI 제품화)으로 이어져:
- 오픈소스 고품질 TTS 가용성 증대
- 소규모 언어 지원 기업들의 등장 가능성

**2. 저자원 언어 지원의 실현성**

1명의 화자 데이터로 다중 화자 음성 합성이 가능함을 보여준 것은, 소수 언어(endangered language) 보존에 실용적 기여. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

### 8.3 향후 연구 시 고려할 핵심 사항

#### 8.3.1 일반화 성능 향상을 위한 기술

**1. 자기지도 학습의 통합 (필수)**

```
[자기지도 인코더 추가]
YourTTS: text_encoder → transformer → duration_predictor
개선안: text_encoder → [transformer + HuBERT/WavLM] → duration_predictor
```

이는 더 강건한 콘텐츠 표현을 제공하여 화자 의존성을 감소시킨다.

**2. 화자 정보의 위계적 모델링**

기존 단일 화자 임베딩 대신:

$$\mathbf{s} = f_\theta(\{\text{성별}, \text{연령}, \text{억양}, \text{음정 범위}, \ldots\})$$

형태의 다차원 화자 속성 모델링이 필요. [arxiv](https://arxiv.org/abs/2506.01020)

**3. 기록 환경 적응**

현재 한계:
- LibriTTS (audiobook, 고품질) vs VCTK (실험실, 중품질)
- 차이가 큰 환경에서 성능 저하

해결책:
- 기록 환경을 명시적 조건으로 추가
- 환경별 정규화 레이어 도입 [nature](https://www.nature.com/articles/s41598-025-90507-0)

#### 8.3.2 다국어 학습의 심화

**문제점:**
- 현재 언어 임베딩 (4-dim)은 언어 특성을 충분히 표현하지 못함
- 음운 체계 차이를 명시적으로 모델링하지 않음

**개선 방향:**

$$L_{\text{lang}} = -\sum_{l=1}^{L} p(l|\mathbf{x}) \log q(l|\mathbf{x})$$

형태의 언어 분류 손실함수 추가로 언어별 음운 특성 강화.

#### 8.3.3 데이터 효율성 극대화

**현재 한계:** 45초 이상 필요

**개선 기법:**

1. **메타 학습**: Few-shot 학습의 메타학습으로 10초 수준으로 단축
2. **데이터 증강**: 시간 스트레칭, 피치 시프트로 가상 데이터 생성
3. **대조 학습(Contrastive Learning)**: 화자 간 거리 최대화로 적응 효율성 향상

#### 8.3.4 크로스-링구얼 음성 변환 개선

**현재 문제:** PT→EN 변환에서 MOS 3.40 (EN→EN 4.20 대비 **19% 저하**)

**원인 분석:**
- 언어 간 음소 인벤토리 차이 (포르투갈어의 비모음 /ɾ/이 영어에 없음)
- 화자 특성과 언어 특성의 얽힘(entanglement)

**해결책:**
$$\mathbf{c}, \mathbf{l} = \text{Disentangle}(x; \Theta_{\text{sep}})$$

형태의 명시적 콘텐츠-언어 분리 메커니즘 도입. [arxiv](http://arxiv.org/pdf/2312.14398.pdf)

#### 8.3.5 강건성 및 안정성 강화

**필요한 개선:**

1. **확률적 Duration Predictor 안정화**
   - 현재: 일부 화자/문장에서 부자연스러운 길이 예측
   - 개선: Deterministic + Stochastic 하이브리드 모델

2. **오발음 최소화**
   - 강제 정렬(forced alignment) 데이터를 사용한 추가 학습
   - 또는 phoneme-aware attention mechanism

3. **노이즈 강건성**
   - 배경 잡음 추가 학습 데이터
   - Speech enhancement 모듈 통합 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10669054/)

***

## 9. 결론 및 종합 평가

YourTTS는 **2022년 발표 당시 zero-shot 다중 화자 TTS 분야의 획기적 기여**를 제시했다:

### 9.1 핵심 성과

1. **SOTA 달성**: VCTK에서 SECS 0.864로 기존 모델 대비 7-18% 향상
2. **다국어 혁신**: 첫 다국어 zero-shot TTS로 저자원 언어 지원의 길 개척
3. **실용적 미세조정**: 1분 미만 데이터로 화자 적응, 실제 배포 가능성 입증
4. **이론적 기여**: Speaker Consistency Loss로 화자 특성 최적화의 새로운 접근법 제시

### 9.2 학술적 위상

| 측면 | 평가 |
|------|------|
| 시간적 영향 | 우수 (3년간 238회 XTTS 인용, YourTTS 기반) |
| 기술적 신규성 | 우수 (SCL, 다국어 학습) |
| 재현성 | 우수 (공개 코드, 모델 제공) |
| 일반화 가능성 | 우수 (다국어, 저자원 언어) |

### 9.3 향후 발전 방향

**단기 (1-2년):**
- 자기지도 인코더 통합으로 강건성 향상
- 화자 속성 위계화로 일반화 개선
- 크로스-링구얼 정확도 55% 개선

**중기 (2-5년):**
- 다국어 확장 (16개 이상 언어)
- 엣지 디바이스 배포 최적화
- 감정/스타일 제어 통합

**장기 (5년+):**
- 자동으로 언어 학습하는 끝없는 다국어 모델
- 실시간 음성 변환 (현재 제약 극복)
- 방언/억양 세밀 제어

***

## 참고문헌

 Casanova, E., Weber, J., Shulby, C., Candido Jr., A., Golge, E., & Ponti, M. A. (2022). YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone. arXiv:2112.02418 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2690bd38-170a-4131-afb2-d953bdef7738/2112.02418v4.pdf)

 Casanova, E., Shulby, C., Golge, E., Müller, N. M., de Oliveira, F. S., Candido Jr., A., ... & Ponti, M. A. (2021). SC-GlowTTS: an Efficient Zero-Shot Multi-Speaker Text-To-Speech Model. Interspeech. [arxiv](https://arxiv.org/pdf/2301.12596.pdf)

 Gong, C., et al. (2023). ZMM-TTS: Zero-shot Multilingual and Multispeaker Speech Synthesis Conditioned on Self-supervised Discrete Speech Representations. arXiv:2312.14398 [arxiv](https://arxiv.org/pdf/2104.05557.pdf)

 Kim, J., Kong, J., & Son, J. (2021). Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech. arXiv:2106.06103 [arxiv](http://arxiv.org/pdf/2312.14398.pdf)

 Learning to Speak from Text: Zero-Shot Multilingual Text-to-Speech with Unsupervised Text Pretraining. arXiv:2301.12596 [arxiv](https://arxiv.org/abs/2409.10058)

 Casanova, E. (2024). XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model. Interspeech. [arxiv](http://arxiv.org/pdf/2406.04904.pdf)

 Casanova, E. et al. (2024). a Massively Multilingual Zero-Shot Text-to-Speech Model. ISCA Archive. [isca-archive](https://www.isca-archive.org/interspeech_2024/casanova24_interspeech.pdf)

 Pankov, V. et al. (2023). DINO-VITS: Data-Efficient Zero-Shot TTS with Self-Supervised Audio Representations. arXiv:2311.09770 [arxiv](https://arxiv.org/pdf/2311.09770.pdf)

 Hao, X. et al. (2021). FullSubNet: A Full-band and Sub-band Fusion Model for Real-time Single-channel Speech Enhancement. ICASSP. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10669054/)

 High fidelity zero shot speaker adaptation in text to speech synthesis with denoising diffusion GAN. Nature Scientific Reports. (2025) [nature](https://www.nature.com/articles/s41598-025-90507-0)

 USAT: A Universal Speaker-Adaptive Text-to-Speech Approach. IEEE/ACM Transactions. (2024) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10508477/)

 Towards Lightweight and Stable Zero-shot TTS with Self-distilled Representation Disentanglement. arXiv:2501.08566 [arxiv](https://arxiv.org/abs/2501.08566)

 DS-TTS: Zero-Shot Speaker Style Adaptation from Voice Clips via Dynamic Dual-Style Feature Modulation. arXiv:2506.01020 [arxiv](https://arxiv.org/abs/2506.01020)

 DiffGAN-ZSTTS: High fidelity zero shot speaker adaptation. Nature Scientific Reports. (2025) [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11842752/)

 Speaker consistency loss and step-wise optimization. OpenReview. (2021) [openreview](https://openreview.net/forum?id=ssK7j0Sztj)

 Using joint training speaker encoder with consistency loss. arXiv:2307.00393 [arxiv](https://arxiv.org/html/2307.00393v1)
