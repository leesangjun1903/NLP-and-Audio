# ERNIE-Music: Text-to-Waveform Music Generation with Diffusion Models

### 1. 핵심 주장과 주요 기여

ERNIE-Music은 2023년 Baidu Inc.에서 제시한 **자유 형식 텍스트 프롬프트를 이용한 텍스트-음악 파형 생성 모델**로, 확산 모델(diffusion model)을 기반으로 음악 파형을 직접 생성하는 선구적 연구입니다. 이 모델의 핵심 주장은 다음과 같습니다:[1]

**주요 기여**:[1]
- 제약 없는 자유 형식 텍스트를 조건부 요인으로 활용하여 확산 모델 프레임워크 내에서 파형 생성 과정을 안내
- 텍스트-음악 병렬 데이터 부족 문제 해결을 위해 웹 리소스에서 약한 감독(weak supervision) 기법으로 데이터셋 구축
- 음악 태그(music tags)와 비제약적 텍스트 설명 두 가지 프롬프트 형식의 효과 비교를 통해 자유 형식 텍스트가 텍스트-음악 연관성을 크게 향상시킴을 입증
- 다양성(diversity), 품질(quality), 텍스트-음악 연관성(text-music relevance) 측면에서 기존 방법들을 상당한 차이로 초과하는 성능 달성

### 2. 문제, 방법, 모델 구조, 성능 및 한계

#### 2.1 해결하고자 하는 문제

**기존 연구의 한계**:[1]
1. **심볼릭 음악 생성의 한계**: 기존 연구(Wu and Sun, 2022)는 음표, 피치, 동적 속성 등 심볼릭 음악 시퀀스를 예측하지만, 실제 청각 경험을 합성하기 위한 후처리가 필요
2. **제한된 제어 능력**: 오디오 신호를 직접 생성하는 방법(Pasini and Schlüter, 2022)은 성능 속성의 세밀한 조정과 제어에 어려움
3. **텍스트 조건부 생성의 제약**: BUTTER(Zhang et al., 2020), Mubert 등은 비구조화된 자유 형식 텍스트 프롬프트로부터 직접 음악 오디오를 생성하는 능력 부재
4. **병렬 데이터 부족**: 텍스트-음악 쌍 데이터의 희소성

#### 2.2 제안하는 방법

**확산 모델 기반 접근**:[1]

확산 모델은 비평형 열역학에 뿌리를 둔 잠재 변수 모델로, 연속 시간 프레임워크에서 정의됩니다. 

**Forward Process (확산 과정)**:
데이터 샘플 \(x\)에 가우시안 노이즈를 점진적으로 추가하는 마르코프 체인:

$$q(z_t|x) = \mathcal{N}(z_t; \alpha_t x, \sigma_t^2 I)$$

$$q(z_{t'}|z_t) = \mathcal{N}(z_{t'}; (\alpha_{t'}/\alpha_t)z_t, \sigma_{t'|t}^2 I)$$

여기서 $\(t, t' \in [0, 1]\), \(t < t'\), 그리고 \(\sigma_{t'|t}^2 = (1 - e^{\lambda_{t'}-\lambda_t})\sigma_{t'}^2\)$ 이며, log signal-to-noise ratio는 $\(\lambda_t = \log(\alpha_t^2/\sigma_t^2)\)$ 로 정의됩니다[1].

**Reverse Process (역확산 과정)**:
파라미터 $\(\theta\)$ 를 가진 함수 근사 $(\(\hat{x}_\theta(z_t, \lambda_t, t) \approx x\))$ 로 디노이징 과정을 추정:

$$p_\theta(z_t|z_{t'}) = \mathcal{N}(z_t; \tilde{\mu}_{t|t'}(z_{t'}, x), \tilde{\sigma}_{t|t'}^2 I)$$

**훈련 목적 함수**:[1]
velocity 예측 방식을 채택하여 "SNR+1" 가중치를 적용:

$$v_t \equiv \alpha_t \epsilon - \sigma_t x$$

최종 훈련 목표:

$$\mathcal{L}_\theta = (1 + \alpha_t^2/\sigma_t^2) \|x - \hat{x}_t\|_2^2 = \|v_t - \hat{v}_t\|_2^2$$

코사인 스케줄 적용: $\(\alpha_t = \cos(\pi t/2)\), \(\sigma_t = \sin(\pi t/2)\)$ , 그리고 $\(\alpha_t^2 + \sigma_t^2 = 1\)$ .[1]

#### 2.3 모델 구조

**전체 아키텍처**:[1]

1. **텍스트 인코더**: ERNIE-M을 사용하여 다국어(중국어, 영어, 한국어, 일본어 등) 텍스트 입력을 인코딩
   - 입력: 길이 \(n\)의 텍스트 토큰
   - 출력: $\([s_0; S]\)$ , 여기서 $\(S = [s_1, ..., s_n]\), \(s_i \in \mathbb{R}^{d_E}\)$
   - $\(s_0\)$ : 입력 텍스트의 분류 표현

2. **음악 확산 모델 (UNet 기반)**: 예상 velocity $\(\hat{v}_\theta(z_t, t, y)\)$ 모델링
   - 입력: 
     - 잠재 변수 $\(z_t \in \mathbb{R}^{d_c \times d_s}\)$
     - 타임스텝 $\(t\)$ (임베딩 $\(e_t \in \mathbb{R}^{d_t \times d_s}\)$ 로 변환)
     - 텍스트 시퀀스 표현 $\([s_0; S] \in \mathbb{R}^{(n+1) \times d_E}\)$
   - 출력: 예측된 velocity $\(\hat{v}_t \in \mathbb{R}^{d_c \times d_s}\)$

**조건부 통합 방법**:[1]

**텍스트-타임스텝 융합**:
$$e'_t = \text{Fuse}(e_t, s_0) \in \mathbb{R}^{d_t' \times d_s}$$
$$z'_t = (z_t \oplus e'_t) \in \mathbb{R}^{(d_t'+d_c) \times d_s}$$

**조건부 Self-Attention (CSA)**:[1]
중간 표현 $\(\phi(z'\_t) \in \mathbb{R}^{d_a \times d_\phi}\)$ 와 텍스트 표현 $\(S \in \mathbb{R}^{n \times d_E}\)$ 에 대해:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

$$Q_i = \phi(z'_t) \cdot W_i^Q$$

$$K_i = \text{Concat}(\phi(z'_t) \cdot W_i^K, S \cdot W_i^{SK})$$

$$V_i = \text{Concat}(\phi(z'_t) \cdot W_i^V, S \cdot W_i^{SV})$$

$$\text{CSA}(\phi(z'_t), S) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

여기서 $\(W_i^Q \in \mathbb{R}^{d_\phi \times d_q}\), \(W_i^K \in \mathbb{R}^{d_\phi \times d_k}\), \(W_i^V \in \mathbb{R}^{d_\phi \times d_v}\), \(W_i^{SK} \in \mathbb{R}^{d_E \times d_k}\), \(W_i^{SV} \in \mathbb{R}^{d_E \times d_v}\), \(W^O \in \mathbb{R}^{hd_v \times d_\phi}\)$ .[1]

**UNet 아키텍처 상세**:[1]
- 14개 레이어의 합성곱 블록과 어텐션 블록 적용
- 입력/출력 채널: 처음 10개 레이어는 512, 이후 256, 128
- 16x16, 8x8, 4x4 해상도에서 어텐션 적용
- 샘플 크기: 327,680, 샘플 레이트: 16,000Hz, 채널 크기: 2 (스테레오)
- 타임스텝 임베딩: 8x1 형태의 학습 가능한 파라미터, 노이즈 스케줄과 연결하여 $\(e_t \in \mathbb{R}^{16 \times 327,680}\)$ 로 확장

#### 2.4 데이터셋 구축

**Web Music with Text Dataset**:[1]
- 음악 플랫폼에서 높은 투표를 받은 사용자 코멘트 활용
- 훈련 세트: 3,890개 샘플, 평균 텍스트 길이 63.23 토큰
- 테스트 세트: 204개 샘플, 평균 텍스트 길이 64.45 토큰
- 음악 샘플: 20초, 16kHz 샘플레이트, 327,680 샘플 크기
- 실제 훈련 샘플 수는 음악 길이가 2-3분이므로 6-9배 증가 (무작위 20초 샘플링)

#### 2.5 성능 평가

**평가 방법론**:[1]
- 객관적 메트릭의 한계로 인해 **인간 평가(human evaluation)** 채택
- 10명의 평가자가 생성된 음악 평가, 모델 식별 정보 비공개
- 평가 차원:
  1. **텍스트-음악 연관성**: 3점 척도 (3=최고, 2, 1)
  2. **음악 품질**: 5점 척도 (5=최고, 4, 3, 2, 1)

**비교 모델**:[1]
- **TSM** (Text-to-Symbolic Music, Wu and Sun, 2022)
- **Mubert** (검색 기반 방법)
- **Musika** (Pasini and Schlüter, 2022)

**성능 결과**:[1]

| 방법 | 텍스트-음악 연관성 점수 | Top Rate | Bottom Rate |
|------|-------------------------|----------|-------------|
| TSM | 2.05 | 12% | 27% |
| Mubert | 1.85 | 37% | 32% |
| **ERNIE-Music** | **2.43** | **55%** | **12%** |

| 방법 | 음악 품질 점수 | Top Rate | Bottom Rate |
|------|----------------|----------|-------------|
| Musika | 3.03 | 5% | 13% |
| **ERNIE-Music** | **3.63** | **15%** | **2%** |

**주요 성과**:
- 텍스트-음악 연관성에서 TSM 대비 18.5% 향상 (2.05 → 2.43)
- 음악 품질에서 Musika 대비 19.8% 향상 (3.03 → 3.63)
- Top Rate에서 경쟁 모델 대비 현저히 높은 비율 달성

**텍스트 형식 비교 실험**:[1]
50개 테스트 샘플로 음악 태그 조건부와 자유 형식 텍스트 조건부 방법 비교:

| 방법 | 점수 | Top Rate | Bottom Rate |
|------|------|----------|-------------|
| Music Tag Conditioning | 1.7 | 22% | 52% |
| **End-to-End Text Conditioning** | **2.3** | **40%** | **10%** |

자유 형식 텍스트가 사전 정의된 태그보다 35.3% 높은 연관성 달성.[1]

#### 2.6 한계점

**논문에서 명시한 한계**:[1]

1. **고정된 짧은 길이**: 
   - 현재 20초 길이로 제한
   - 계산 리소스 제약으로 긴 시퀀스 훈련 불가능
   - 추론 단계에서 길이 변경 시 성능 저하

2. **느린 생성 속도**:
   - 반복적 생성 프로세스로 인한 속도 제약
   - 최적화 기법 필요

3. **악기 음악에 국한**:
   - 훈련 데이터가 주로 악기 음악으로 구성
   - 인간 목소리 포함 불가능
   - 보컬 음악 데이터셋 확장 필요

### 3. 모델의 일반화 성능 향상 가능성

#### 3.1 일반화 성능 관련 설계 요소

**다국어 지원**:[1]
- ERNIE-M 텍스트 인코더를 통한 중국어, 영어, 한국어, 일본어 등 다국어 입력 지원
- 다양한 언어권의 음악 설명 처리 가능

**자유 형식 텍스트 활용**:[1]
- 사전 정의된 태그 대신 비제약적 텍스트 설명 사용
- 모델이 동적으로 유용한 정보 학습
- 노이즈가 많은 수작업 태그 선택 규칙 배제

**다양한 음악 특성 생성**:[1]
- **멜로디**: 부드럽고 편안한 리듬 vs. 열정적이고 빠른 리듬
- **감정 표현**: 슬픔, 축제, 명랑함
- **악기**: 피아노, 바이올린, 얼후(erhu), 기타 등 다양한 악기 조합

**데이터 증강**:
- 2-3분 음악에서 무작위 20초 샘플링으로 효과적으로 6-9배 데이터 증강[1]

#### 3.2 일반화 성능의 제약 요소

**훈련 데이터 편향**:[1]
- 중국어 플랫폼 중심의 데이터 수집으로 특정 문화권 음악 선호 가능
- 3,890개 훈련 샘플은 최신 대규모 모델(MusicLM: 280,000시간, MusicGen: 20,000시간)에 비해 소규모[2][3]

**장르 및 스타일 다양성**:
- 악기 음악에 국한, 보컬 음악 생성 불가[1]
- 특정 장르에 대한 편향 가능성

**교차 도메인 성능**:
최근 연구에 따르면 음악 생성 모델의 교차 문화적 일반화는 중요한 도전 과제입니다:
- Audio-Flamingo와 Qwen2-Audio는 서양 팝 장르(80s, 90s)에서 상대적으로 강력한 성능을 보이지만, 세계 음악 장르(Bossanova, Celtic, Latin 등)에서는 성능 저하[4]
- 문화적 및 역사적 다양성 부족이 주요 원인[4]

### 4. 앞으로의 연구에 미치는 영향과 고려사항

#### 4.1 연구에 미치는 영향

**1. 텍스트-파형 직접 생성 패러다임 확립**

ERNIE-Music은 **자유 형식 텍스트에서 음악 파형으로의 직접 생성**이라는 새로운 패러다임을 제시했습니다. 이는 후속 연구에 중요한 영향을 미쳤습니다:[1]

- **MusicLM** (2023): Google의 MusicLM은 계층적 시퀀스-투-시퀀스 모델링으로 24kHz에서 수 분간 일관된 음악 생성[2]
- **Stable Audio** (2024): 44.1kHz 스테레오로 최대 3분 길이의 완전한 트랙 생성, 구조화된 작곡(인트로, 전개, 아웃트로) 포함[5][6]
- **MusicGen** (2023): Facebook의 단일 단계 변환기 기반 LM으로 압축된 이산 음악 표현 생성[3][7]

**2. 확산 모델의 음악 생성 적용 확대**

ERNIE-Music의 확산 모델 기반 접근은 음악 생성 분야에서 확산 모델 연구를 촉진했습니다:[8][9]

- **Diffusion Models** (DMs): 점진적 디노이징으로 높은 오디오 충실도와 스타일 다양성 제공[9]
- **주요 모델**: Mousai, Noise2Music, Riffusion[9]
- **AudioLDM** (2023): 잠재 확산 모델로 텍스트-오디오 생성, zero-shot 오디오 조작 지원[10][11]
- **MusicLDM** (2023): Beat-Synchronous Mixup으로 음악 구조 정렬 개선[12]

**3. 다국어 및 다중 모달 조건부 생성**

ERNIE-Music의 ERNIE-M 기반 다국어 지원은 음악 생성의 글로벌 접근성을 높였으며, 후속 연구는 다중 모달 조건부 생성으로 확장되었습니다:[1]

- **M²UGen** (2024): 음악, 이미지, 비디오를 포함한 다중 소스 영감으로 음악 생성[13]
- **MeLFusion** (2024): 텍스트 설명과 이미지에서 음악 합성[14]

#### 4.2 2020년 이후 관련 최신 연구 비교 분석

**생성 아키텍처 비교**:

| 모델 | 연도 | 아키텍처 | 샘플레이트 | 최대 길이 | 특징 |
|------|------|----------|------------|-----------|------|
| **ERNIE-Music**[1] | 2023 | Diffusion + UNet | 16kHz | 20초 | 자유 형식 텍스트, 다국어 |
| **MusicLM**[2] | 2023 | Hierarchical Seq2Seq | 24kHz | 수 분 | 계층적 생성, 텍스트 부착성 우수 |
| **MusicGen**[3][7] | 2023 | Transformer LM | 32kHz | 가변 | 단일 단계, 멜로디 조건부 |
| **Stable Audio 2.0**[5][6] | 2024 | Latent Diffusion | 44.1kHz | 3분 | 구조화된 작곡, 오디오-투-오디오 |
| **Riffusion**[15][16] | 2023 | Spectrogram Diffusion | 가변 | 실시간 | 스펙트로그램 기반, 빠른 생성 |
| **Mustango**[17] | 2024 | Diffusion | 16kHz | 가변 | 음악 특화 프롬프트, 제어성 향상 |
| **MusicFlow**[18] | 2024 | Cascaded Flow Matching | 가변 | 가변 | 의미적/음향적 특징 모델링 |

**성능 지표 비교**:

**객관적 평가 메트릭**:[19]
- **FAD** (Fréchet Audio Distance): 생성 음악과 실제 음악 간의 근접성 평가
- **KL Divergence**: 분포 간 차이 측정
- **CLAP Score**: 텍스트-오디오 정렬 평가

**주관적 평가 차원**:[19]
- **Production Quality (PQ)**: 오디오 충실도, 왜곡 부재, 균형잡힌 주파수 범위
- **Production Complexity (PC)**: 오디오 요소의 수와 상호작용
- **Text Alignment**: 텍스트 프롬프트와의 부합성

**최신 모델의 성능 동향**:
- **Stable Audio 2.0**: 3분 길이의 완전한 트랙 생성, 구조적 일관성 유지[6][5]
- **MusicGen**: 높은 품질의 샘플 생성, 텍스트 설명 및 멜로디 특징 조건부[3]
- **MusicLDM**: Beat-Synchronous Mixup으로 텍스트-음악 정렬 개선[12]

#### 4.3 앞으로 연구 시 고려할 점

**1. 긴 형식 음악 생성 및 구조적 일관성**

**현재 과제**:
- ERNIE-Music의 20초 제한은 실용적 음악 제작에 불충분[1]
- 긴 시퀀스 생성 시 구조적 일관성 유지 어려움

**최신 해결책**:
- **Stable Audio 2.0**: 최대 3분 길이의 구조화된 음악(인트로, 전개, 아웃트로) 생성[5][6]
  - 고도로 압축된 오토인코더로 잠재 표현 달성
  - 긴 시간 스케일에서 성능 향상을 위한 모든 구성 요소 적응
- **Long-form Music Generation**: 잠재 확산으로 최대 4분 45초 음악 생성, 긴 시간적 맥락 훈련[20][21]

**권장 사항**:
- 계층적 생성 전략 채택: 전역 구조 먼저, 세부 사항 나중에 생성[2]
- 메모리 효율적 어텐션 메커니즘 활용 (예: Linear Attention)[22]
- 압축률 높은 오토인코더로 긴 시퀀스를 짧은 잠재 표현으로 압축[6][5]

**2. 제어 가능성 및 세밀한 조정**

**현재 과제**:
- 텍스트만으로는 시간적 음악 특징(코드, 리듬) 정밀 제어 어려움[23]
- 사용자가 원하는 음악적 속성을 정확히 지정하기 어려움

**최신 해결책**:
- **MusiConGen** (2024): 자동 추출된 리듬과 코드를 조건 신호로 통합, 소비자급 GPU에서 효율적 파인튜닝[23]
- **BandControlNet** (2024): 세밀한 시공간적 특징으로 조종 가능한 대중 음악 생성, 병렬 변환기 기반[24]
- **Stable Audio 2.0**: 오디오-투-오디오 생성으로 샘플 업로드 및 자연어 프롬프트로 변환[5]
- **ControlNet-inspired conditioning**: 리릭-투-보컬 합성, 타겟 텍스트-투-샘플 생성 등 다운스트림 작업[22]

**권장 사항**:
- 다중 조건부 입력 통합: 텍스트, 멜로디, 리듬, 코드 진행 등[23]
- LoRA 및 ControlNet 아키텍처로 전문화된 생성 작업 파인튜닝[22]
- 인터랙티브 인터페이스 개발: 사용자가 실시간으로 파라미터 조정[25]

**3. 데이터 효율성 및 일반화**

**현재 과제**:
- ERNIE-Music의 3,890개 샘플은 대규모 모델 대비 소규모[1]
- 특정 도메인/장르에 대한 과적합 위험
- 교차 문화적 일반화 부족[26][4]

**최신 해결책**:
- **대규모 데이터셋 활용**:
  - MusicLM: 280,000시간[2]
  - MusicGen: 20,000시간 (400,000개 녹음)[7][3]
  - Stable Audio: AudioSparx 라이선스 데이터셋[5]
- **Mixed Training**: 노래와 음성 데이터 혼합 훈련으로 데이터 부족 해결[27][28]
- **Data Augmentation**: 긴 음악에서 다중 세그먼트 샘플링, 증강 기법 적용[22]

**권장 사항**:
- **다양한 데이터 소스 통합**: 다양한 장르, 문화권, 언어의 음악 포함[26]
- **Opt-out 요청 및 공정한 보상**: 창작자 저작권 보호[5]
- **Zero-shot Learning**: 새로운 장르/스타일에 대한 일반화 능력 향상[29][27]
- **Cross-domain Transfer Learning**: 음성 데이터 활용으로 노래 생성 개선[28][27]

**4. 평가 방법론 개선**

**현재 과제**:
- 주관적 평가의 높은 비용과 낮은 재현성
- 객관적 메트릭(FAD, KL)은 인간 지각과 불일치[19][1]
- 음악성(musicality), 구조적 일관성 평가 어려움

**최신 해결책**:
- **MusicEval** (2025): 전문가 평가가 포함된 생성 음악 평가 데이터셋, 자동 평가와 인간 지각 정렬[19]
- **MusicBench & MusicCaps**: 검증 및 평가용 표준 데이터셋[30]
- **VERSA Toolkit** (2024): 음성, 오디오, 음악을 위한 다목적 평가 툴킷, 다양한 외부 리소스 활용[31]
- **MiRA Tool** (2024): 음악 복제 평가 도구, 다섯 가지 오디오 유사성 메트릭 기반[32]

**권장 사항**:
- **다차원 평가 프레임워크 구축**:
  - 음악 품질: FAD, KL, CLAP[19]
  - 텍스트-음악 연관성: CLAP Score, 의미적 유사성[19]
  - 음악성: 화성 진행, 리듬 일관성, 멜로디 연속성[33]
  - 구조: 인트로-전개-아웃트로 존재 여부[5]
- **Human-in-the-Loop Training**: 인간 선호 데이터셋으로 보상 모델 훈련[34][22]
- **자동화된 객관적 메트릭 개발**: 인간 지각과 높은 상관관계[34][19]

**5. 윤리적 및 법적 고려사항**

**현재 과제**:
- 훈련 데이터 복제 및 표절 위험[32]
- 저작권 침해 가능성
- 음악 창작자에 대한 공정한 보상

**최신 해결책**:
- **MiRA Tool**: 데이터 복제 평가로 10% 이상 정확한 복제 탐지[32]
- **Content Recognition (ACR)**: Stable Audio 2.0은 Audible Magic과 협력하여 저작권 침해 방지[5]
- **라이선스 데이터셋**: Stable Audio는 AudioSparx 라이선스 데이터셋 사용, Opt-out 요청 존중[5]

**권장 사항**:
- **투명한 데이터 소싱**: 데이터 출처 명확히 하고 저작권 준수[5]
- **생성 음악 워터마킹**: 생성된 콘텐츠 추적 및 식별[32]
- **윤리적 사용 가이드라인**: 적대적 환경 조성 음악 생성 금지[3]
- **창작자 권리 보호**: 공정 사용 정책 및 보상 메커니즘 마련[5]

**6. 계산 효율성 및 접근성**

**현재 과제**:
- 확산 모델의 느린 생성 속도[1]
- 대규모 모델의 높은 계산 비용

**최신 해결책**:
- **Flow Matching**: 확산 모델보다 빠른 생성, FAD, KL, CLAP에서 우수한 성능[18][35]
- **Efficient Diffusion**: MeLoDy는 MusicLM 대비 95.7%-99.6% forward pass 감소로 10초-30초 음악 생성[36]
- **Progressive Distillation**: 빠른 샘플링을 위한 확산 모델 증류[3]
- **Latent Diffusion**: 압축된 잠재 공간에서 작동하여 계산 효율 향상[10][6][5]

**권장 사항**:
- **경량 모델 개발**: 소비자급 하드웨어에서 실행 가능한 모델[23]
- **실시간 생성**: Riffusion의 실시간 음악 생성처럼 즉각적 피드백 제공[15][16]
- **오픈소스 도구**: AudioCraft, Riffusion처럼 커뮤니티가 접근 가능한 모델[7][15][3]

**7. 다중 작업 및 통합 모델**

**현재 과제**:
- 각 작업(생성, 편집, 스타일 전송)마다 별도 모델 필요
- 모델 간 통합 및 일관성 부족

**최신 해결책**:
- **MusicGen-Stem** (2025): 다중 스템(베이스, 드럼, 기타) 생성 및 편집, 반복적 작곡 지원[37]
- **ACE-Step** (2025): 음악 생성 파운데이션 모델, 음악 변주, 오디오 인페인팅, 리릭 편집 등 통합[22]
- **AudioLDM**: zero-shot 오디오 인페인팅, 스타일 전송, 초해상도 지원[10]
- **Instruct-MusicGen** (2024): 명령 튜닝으로 텍스트-음악 편집 잠금 해제[38]

**권장 사항**:
- **통합 프레임워크 개발**: 생성, 편집, 스타일 전송을 단일 모델에 통합[37][22]
- **Instruction Tuning**: 자연어 명령으로 다양한 작업 수행[38]
- **모듈형 아키텍처**: 플러그 앤 플레이 방식으로 기능 추가[25]

**8. 음악 이론 및 도메인 지식 통합**

**현재 과제**:
- 데이터 기반 접근만으로는 음악 이론적 일관성 보장 어려움
- 화성, 리듬, 형식 등 구조적 요소 제어 부족

**최신 해결책**:
- **MusiConGen**: 자동 추출된 리듬과 코드 진행으로 음악 이론적 제어 강화[23]
- **MusicGen-Chord** (2024): 코드 진행과 텍스트 설명으로 음악 생성[39]
- **Domain Knowledge Metrics**: 음계 일관성, 화성 진행, 음정 빈도 등 평가[33]

**권장 사항**:
- **음악 이론 제약 통합**: 화성 규칙, 리듬 패턴을 생성 과정에 통합[33][23]
- **계층적 표현**: 음표 레벨, 마디 레벨, 섹션 레벨의 다층 구조 모델링[2]
- **음악 분석 모듈**: 생성된 음악의 이론적 타당성 검증[33]

### 결론

ERNIE-Music은 자유 형식 텍스트-음악 파형 생성 분야의 선구적 연구로, 확산 모델을 활용한 음악 생성의 가능성을 입증했습니다. 이 모델은 다국어 지원, 다양한 음악 특성 생성, 높은 텍스트-음악 연관성을 달성했으나, 길이 제한, 느린 생성 속도, 악기 음악 한정 등의 한계가 있습니다.[1]

2020년 이후 음악 생성 연구는 급속히 발전하여 **더 긴 생성 길이**(Stable Audio 2.0: 3분, Long-form: 4분 45초), **더 높은 샘플레이트**(44.1kHz), **더 정교한 제어 메커니즘**(MusiConGen, BandControlNet), **더 효율적인 생성 방법**(Flow Matching, Efficient Diffusion)을 달성했습니다.[21][35][18][36][20][24][6][23][5]

향후 연구는 **(1) 긴 형식 음악의 구조적 일관성 유지**, **(2) 다차원 제어 가능성 향상**, **(3) 교차 문화적 일반화 능력 강화**, **(4) 인간 지각과 정렬된 평가 방법론 개발**, **(5) 윤리적 데이터 소싱 및 저작권 보호**, **(6) 계산 효율성 개선**, **(7) 다중 작업 통합 모델 개발**, **(8) 음악 이론적 일관성 보장** 등을 중점적으로 고려해야 합니다.

특히 **Foundation Model 접근**은 Stable Diffusion이 이미지 생성에서 이룬 것처럼 음악 생성 분야에서도 강력하고 유연한 오픈 파운데이션을 제공할 잠재력을 가지고 있으며, LoRA, ControlNet 등을 통한 세밀한 제어, 커스터마이제이션, 반복적 개선을 가능하게 합니다. 이러한 방향으로 연구가 진행된다면, 음악 생성 AI는 창작자에게 진정한 창조적 파트너가 될 수 있을 것입니다.[40][41][22]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b974b2ff-4a15-4fb2-bd09-f516345abffb/2302.04456v2.pdf)
[2](https://arxiv.org/pdf/2301.11325.pdf)
[3](https://arxiv.org/abs/2306.05284)
[4](https://arxiv.org/html/2506.12285v1)
[5](https://stability.ai/news/stable-audio-2-0)
[6](https://arxiv.org/pdf/2402.04825.pdf)
[7](https://ai.meta.com/blog/audiocraft-musicgen-audiogen-encodec-generative-ai-audio/)
[8](http://arxiv.org/pdf/2409.03715.pdf)
[9](https://www.ewadirect.com/proceedings/ace/article/view/21641)
[10](https://proceedings.mlr.press/v202/liu23f/liu23f.pdf)
[11](https://mac.kaist.ac.kr/~juhan/gct634/Slides/13.%20audio-level%20music%20generation.pdf)
[12](https://arxiv.org/abs/2308.01546)
[13](https://arxiv.org/html/2311.11255v3)
[14](https://arxiv.org/html/2406.04673v1)
[15](https://www.unite.ai/riffusion-review/)
[16](https://www.musicful.ai/vs/riffusion-ai-review/)
[17](http://arxiv.org/pdf/2311.08355.pdf)
[18](http://arxiv.org/pdf/2410.20478.pdf)
[19](https://arxiv.org/pdf/2501.10811.pdf)
[20](https://arxiv.org/html/2404.10301v2)
[21](https://arxiv.org/pdf/2404.10301.pdf)
[22](https://arxiv.org/html/2506.00045v1)
[23](https://arxiv.org/abs/2407.15060)
[24](https://arxiv.org/html/2407.10462v1)
[25](https://machinelearning.apple.com/research/controllable-music)
[26](https://arxiv.org/html/2502.07328v1)
[27](http://arxiv.org/pdf/2501.13870.pdf)
[28](https://arxiv.org/html/2501.13870v1)
[29](https://archives.ismir.net/ismir2019/paper/000005.pdf)
[30](https://research.samsung.com/blog/Diffusion-based-Text-to-Music-Generation-with-Global-and-Local-Text-based-Conditioning)
[31](https://arxiv.org/pdf/2412.17667.pdf)
[32](https://arxiv.org/abs/2407.14364)
[33](https://musicinformatics.gatech.edu/wp-content_nondefault/uploads/2018/11/postprint.pdf)
[34](https://arxiv.org/html/2509.00051v1)
[35](https://arxiv.org/pdf/2406.10970.pdf)
[36](https://arxiv.org/pdf/2305.15719.pdf)
[37](http://arxiv.org/pdf/2501.01757.pdf)
[38](http://arxiv.org/pdf/2405.18386.pdf)
[39](https://arxiv.org/pdf/2412.00325.pdf)
[40](https://arxiv.org/pdf/2502.02358.pdf)
[41](https://arxiv.org/html/2508.10949)
[42](https://journals.sagepub.com/doi/10.1177/00368504251383055)
[43](https://www.semanticscholar.org/paper/06ca869b5e1d3904a7bbb1bc2fadfd0e51068ddc)
[44](https://www.semanticscholar.org/paper/deca5f65d7cdad237490b46cd26655791b128903)
[45](https://arxiv.org/html/2409.02845v2)
[46](https://arxiv.org/pdf/2211.09124.pdf)
[47](https://www.cometapi.com/best-3-ai-music-generation-models-of-2025/)
[48](https://ncsoft.github.io/ncresearch/f27188f9c5fdfec1298f8fd78fbf3718125cf5a3)
[49](https://arxiv.org/html/2509.23364v1)
[50](https://arxiv.org/html/2507.20128v1)
[51](https://ieeexplore.ieee.org/iel8/6287639/10820123/10845168.pdf)
[52](https://arxiv.org/html/2511.13936v1)
[53](https://arxiv.org/html/2412.18688v2)
[54](https://arxiv.org/pdf/2511.13936.pdf)
[55](https://arxiv.org/html/2509.24773)
[56](https://arxiv.org/html/2501.15302v1)
[57](https://www.ijfmr.com/research-paper.php?id=51404)
[58](https://isjem.com/download/smart-music-composer-a-web-tool-based-on-transformer-model/)
[59](https://arxiv.org/pdf/2306.05284.pdf)
[60](https://arxiv.org/html/2501.08809v1)
[61](https://arxiv.org/pdf/2504.05690.pdf)
[62](https://github.com/facebookresearch/audiocraft/blob/main/model_cards/MUSICGEN_MODEL_CARD.md)
[63](https://www.audiocipher.com/post/stable-audio-ai)
[64](https://www.youtube.com/watch?v=v-YpvPkhdO4)
[65](https://stability.ai/stable-audio)
[66](https://www.topmediai.com/ai-music/riffusion-ai-review/)
[67](https://audiocraft.metademolab.com/musicgen.html)
[68](https://arxiv.org/html/2507.01022v1)
[69](https://arxiv.org/html/2508.20088v1)
[70](https://arxiv.org/html/2506.08570v1)
[71](https://arxiv.org/abs/2409.20196)
[72](https://arxiv.org/pdf/2311.09094.pdf)
[73](https://ieeexplore.ieee.org/document/10605479/)
[74](https://arxiv.org/abs/2210.14868)
[75](https://arxiv.org/abs/2402.16694)
[76](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-024-00370-6)
[77](https://arxiv.org/abs/2407.21531)
[78](https://dl.acm.org/doi/10.1145/3597493)
[79](https://zenodo.org/doi/10.5281/zenodo.14877299)
[80](http://arxiv.org/pdf/2411.01135v2.pdf)
[81](https://downloads.hindawi.com/journals/cin/2022/3847415.pdf)
[82](http://arxiv.org/pdf/2404.00775.pdf)
[83](https://neurips.cc/virtual/2023/papers.html)
[84](https://ecai2025.org/accepted-papers/)
[85](https://dl.acm.org/doi/10.1145/3769106)
[86](https://proceedings.neurips.cc/paper_files/paper/2022/file/af2bb2b2280d36f8842e440b4e275152-Paper-Conference.pdf)
[87](https://www.nature.com/articles/s41598-025-02792-4)
[88](https://arxiv.org/html/2510.16720v1)
[89](https://arxiv.org/html/2510.22455v1)
[90](https://arxiv.org/pdf/2507.18061.pdf)
[91](https://arxiv.org/html/2407.21633v1)
[92](https://arxiv.org/html/2501.07278v1)
[93](https://arxiv.org/html/2509.03131v1)
[94](https://arxiv.org/html/2506.05104v2)
[95](https://dl.acm.org/doi/10.1145/3705328.3748138)
[96](https://arxiv.org/abs/2501.13870)
