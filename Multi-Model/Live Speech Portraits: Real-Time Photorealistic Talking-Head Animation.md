# Live Speech Portraits: Real-Time Photorealistic Talking-Head Animation

### 1. 핵심 주장과 주요 기여

**Live Speech Portraits (LSP)**는 음성만을 입력으로 하여 실시간(30fps 이상)으로 개인화된 포토리얼리스틱 토킹헤드 애니메이션을 생성하는 시스템을 제시합니다. 이 연구의 핵심 주장은 다음과 같습니다.[1]

**주요 기여:**

1. **실시간 포토리얼리스틱 토킹헤드 생성의 첫 구현**: 이전 연구들은 실시간 성능을 명시적으로 보여주지 않았으나, LSP는 처음으로 실제 라이브 시스템을 구현하여 30fps 이상의 성능을 달성했습니다.[1]

2. **음성 특성 추출 및 일반화 개선**: Autoregressive Predictive Coding (APC) 모델을 활용하고 **Manifold Projection**을 도입하여 야생(wild) 음성으로의 일반화 능력을 크게 향상시켰습니다. 이는 도메인 적응의 관점에서 음성 표현을 대상 인물의 음성 공간으로 투영합니다.[1]

3. **확률적 자기회귀 머리 움직임 모델**: 머리 포즈를 가우시안 분포로 모델링하여 다양하고 자연스러운 머리 움직임을 생성합니다. 이는 기존의 결정론적(deterministic) 접근과 차별화됩니다.[1]

4. **전신 움직임 제어**: 머리 포즈로부터 상체 움직임을 추론하여 더 사실적인 애니메이션을 생성합니다.[1]

***

### 2. 문제 정의, 제안 방법, 모델 구조, 성능 향상 및 한계

#### 2.1 해결하고자 하는 문제

음성 기반 토킹헤드 생성은 다음의 다층적 과제를 내포합니다:[1]

1. **1D에서 고차원으로의 매핑 문제**: 1차원 음성 신호를 고차원 얼굴 운동으로 매핑하면서 개인화된 말하기 특성을 보존해야 합니다.

2. **도메인 차이(Domain Gap)**: 야생 음성과 대상 인물의 음성 공간 간의 차이로 인한 성능 저하

3. **약한 오디오-운동 상관성**: 머리 운동 및 상체 움직임은 음성과 약하게 연관되어 있어 결정론적 예측이 어렵습니다.

4. **신경망 렌더링의 한계**: 학습 데이터 범위 밖의 움직임에서 성능 저하가 발생합니다.

5. **실시간 성능 요구**: 비디오 컨퍼런싱 등 실제 응용에서는 높은 효율성이 필수적입니다.

#### 2.2 제안하는 방법 및 수식

LSP는 3단계 파이프라인으로 구성됩니다:[1]

**Stage 1: 심층 음성 표현 추출 및 Manifold Projection**

음성으로부터 추출되는 표현은 자기회귀 예측 부호화(APC) 모델을 통해 획득됩니다:

$$h = \text{GRU}(x_1, x_2, \ldots, x_t)$$

여기서 $h \in \mathbb{R}^{512}$는 GRU의 최종 계층 은닉 상태입니다.

**Manifold Projection (핵심 개선)**:

추출된 음성 표현을 대상 인물의 음성 특성 공간으로 투영하기 위해 Locally Linear Embedding (LLE)을 적용합니다:[1]

$$\min_{\alpha} \left\| h - \sum_{i=1}^{k} \alpha_i f_i \right\|_2^2, \quad \text{subject to} \sum_{i=1}^{k} \alpha_i = 1$$

투영된 표현은:

$$\bar{h} = \sum_{i=1}^{k} \alpha_i f_i$$

여기서 $f_i$는 대상 음성 데이터베이스 $D \in \mathbb{R}^{N \times 512}$에서 가장 가까운 $k$개($k=10$) 이웃입니다. 이는 본질적으로 도메인 적응 기법으로, 야생 음성의 특성을 학습된 표현 다양체상에서 재구성합니다.[1]

**Stage 2: 음성-얼굴 예측**

**입과 관련된 움직임 (Audio-to-Mouth Motion)**:

LSTM 네트워크를 통해 음성 표현을 3D 입술 변위로 변환합니다:[1]

$$m_0, m_1, \ldots, m_t = \text{LSTM}(h_0, h_1, \ldots, h_{t+\tau})$$

$$v_t = \text{MLP}(m_t)$$

여기서 $\tau = 18$ 프레임(300ms)의 시간 지연을 추가하여 음성의 미래 정보에 접근합니다. $v_t \in \mathbb{R}^{25 \times 3}$은 25개의 입 관련 3D 랜드마크의 변위입니다.

**확률적 머리 포즈 생성 (Probabilistic Head Pose Synthesis)** - **핵심 혁신**:

머리 포즈는 1-다(one-to-many) 매핑 문제를 해결하기 위해 확률적으로 모델링됩니다. 조건부 가우시안 분포를 사용:[1]

$$P(x_t | x_{t-1}, x_{t-2}, \ldots, x_{t-L}, h_t) = \mathcal{N}(\mu_t, \sigma_t^2)$$

네트워크는 다음을 출력합니다:[1]

$$(\mu_t, \sigma_t) = \text{Net}(x_{t-1}, x_{t-2}, \ldots, x_{t-L}, h_t)$$

확률적 샘플링을 통해 머리 포즈를 획득:

$$x_t \sim \mathcal{N}(\mu_t, \sigma_t^2)$$

이 디자인은 두 가지 가정을 기반으로 합니다:[1]
- **가정 1**: 머리 포즈는 음성 정보(표현, 억양)와 부분적으로 연관
- **가정 2**: 현재 머리 포즈는 과거 포즈와 부분적으로 의존

네트워크는 7개 계층의 확대 컨볼루션(dilated convolution) 블록 2개로 구성되며, 확대 비율이 7회 두 배 증가하며, 이후 두 번 반복되어 **255 프레임(4.25초)의 수용 영역**을 달성합니다.[2][3][4][5][6][1]

설명된 아키텍처는 딜레이션 비율을 1, 2, 4, 8, 16, 32, 64의 순서로 구성된 7개의 레이어 묶음을 사용하며, 이 묶음을 전체 아키텍처 내에서 두 번 반복하여 총 14개의 레이어로 구성됩니다.

**손실 함수 - 확률적 헤드 포즈**:

$$\mathcal{L}_{\text{pose}} = -\ln \mathcal{N}(x_t | \mu_t, \sigma_t^2)$$

이는 네트워크가 분포의 평균과 표준편차를 올바르게 예측하도록 강제합니다.[1]

**Stage 3: 포토리얼리스틱 이미지 합성**

조건 특성 맵과 후보 이미지 집합을 이용한 이미지-대-이미지 변환:

$$I_{\text{gen}} = G(C_{\text{feature}}, \{I_{\text{cand}}\})$$

여기서 생성자 손실은:[1]

$$\mathcal{L}_G = \lambda_1 \mathcal{L}_{\text{GAN}} + \lambda_2 \mathcal{L}_{\text{color}} + \lambda_3 \mathcal{L}_{\text{perceptual}} + \lambda_4 \mathcal{L}_{\text{feature}}$$

각 가중치는 $\lambda_1=100, \lambda_2=10, \lambda_3=1, \lambda_4=1$로 설정됩니다.[1]

색상 손실:

$$\mathcal{L}_{\text{color}} = \mathbb{E}[|I_{\text{gen}} - I_{\text{gt}}|_1]$$

지각 손실:

$$\mathcal{L}_{\text{perceptual}} = \sum_{l \in S} \mathbb{E}[|\phi_l(I_{\text{gen}}) - \phi_l(I_{\text{gt}})|_1]$$

여기서 $S = \{1, 6, 11, 20, 29\}$는 VGG19의 선택된 계층입니다.[1]

#### 2.3 모델 구조

LSP의 전체 구조는 Figure 2에서 보이듯이:[1]

| 모듈 | 구성 | 입력 | 출력 |
|------|------|------|------|
| **APC 음성 추출** | 3층 GRU (512D) | 80D log Mel spectrograms | 512D 특성 |
| **Manifold Projection** | LLE 재구성 | 512D 음성 특성 | 512D 투영 특성 |
| **입 움직임 모델** | 3층 LSTM + MLP | 음성 특성 (18프레임 지연) | 25개 랜드마크의 3D 변위 |
| **머리 포즈 모델** | 확대 CNN (2 residual blocks) | 음성 + 과거 포즈 | 6D 머리 포즈 (회전+평행이동) |
| **상체 움직임** | Billboard 변환 | 머리 포즈의 평행이동 | 어깨 랜드마크 |
| **렌더링 네트워크** | 8층 UNet (skip connection) | 조건 맵 + 4 후보 이미지 | 512×512 포토리얼리스틱 이미지 |

#### 2.4 성능 향상

**정량적 성능**:

| 방법 | 음성 지연 최적 | L1 손실 (mm) | PSNR | SSIM | LPIPS |
|------|------------------|------|------|------|-------|
| 우리 방법 (전체) | 300ms | **4.916** | **26.006** | **0.862** | **0.698** |
| L2 회귀 변형 | 300ms | 5.248 | 25.920 | 0.861 | 0.727 |
| 히스토리 없음 | 300ms | 3.900 | 25.100 | 0.850 | 0.875 |
| LSTM 기반 | 300ms | 4.900 | 25.000 | 0.830 | 0.875 |

특히 **300ms의 음성 지연**이 최적이며, 더 짧은 지연은 코아티큘레이션을 모델링할 수 없고, 더 긴 지연은 과도한 정보를 포함합니다.[1]

**사용자 연구**:

48명의 참여자를 대상으로 한 평가에서:[1]

- **사실성** (Realistic?): 4.8/5.0 (가장 높음)
- **입술-음성 동기화** (Lip-sync?): 4.2/5.0
- **머리 움직임 자연성** (Head Motion?): 4.1/5.0

이는 Zhou et al. (2020)의 3.8, 3.4, 2.8을 상회합니다.[1]

#### 2.5 한계

논문에서 명시한 한계점:[1]

1. **폐쇄음 및 비음 자음의 약한 캡처**: /p/, /b/, /m/ 등이 작은 음량으로 발음되거나 소음으로 무시되는 경향

2. **빠른 속도의 음성 처리 부족**: 싸움처럼 매우 빠른 속도의 음성에서는 성능 저하

3. **스펙트럼 구성의 한계**: log Mel spectrogram은 단기 음소를 놓칠 수 있음 (wav2vec 등의 순수 심층 특성으로 개선 가능)

4. **안면 추적 알고리즘의 제약**: 사용된 최적화 기반 추적이 차선(sub-optimal)으로, 더 나은 추적이 입술 동기화 개선 가능

5. **학습 데이터 범위 제약**: 생성된 비디오의 스타일은 3-5분의 학습 비디오에 제한됨

6. **감정 표현 부족**: 중립 스타일로 학습시 감정적 음성에 대한 성능 저하

***

### 3. 모델의 일반화 성능 향상 가능성 **[중점 분석]**

#### 3.1 Manifold Projection의 일반화 효과

LSP의 핵심 혁신인 **Manifold Projection**은 일반화 성능을 크게 개선합니다. t-SNE 시각화(Figure 8)에서:[1]

- **원본 표현**(파란색): 야생 음성 특성이 대상 음성 공간(초록색)과 멀리 분포
- **재구성 표현**(빨간색): 대상 음성 공간에 매우 근접하게 투영됨

이 메커니즘은 **3가지 어려운 시나리오**에서 입 동기화 개선을 입증합니다:[1]
- 다른 성별의 음성 (예: 남성 음성으로 여성 구동)
- 외국어 음성
- 심지어 노래까지

이는 **도메인 적응(Domain Adaptation)** 개념을 음성 표현에 적용한 것으로, 출처 도메인(야생 음성)의 특성을 대상 도메인(대상 인물 음성) 다양체에 재투영합니다.

#### 3.2 확률적 모델의 일반화 강점

확률적 머리 포즈 모델은 결정론적 접근보다 일반화 성능이 우수합니다:[1]

| 변형 | 평면 오류 (D-L) | 속도 오류 (D-V) | 회전/평행 오류 (D-RotPos) |
|------|------|------|------|
| LSTM + L2 | 4.9 | 1.1 | 6.91/2.2 |
| LSTM + 확률적 | 4.9 | 1.0 | 6.71/1.6 |
| **우리 방법 (L2)** | 4.5 | 1.1 | 3.7/9.2 |
| **우리 방법 (확률적, 전체)** | **3.6** | **0.8** | **3.6/8.9** |

**이유**: 확률적 모델은 음성과 역사적 포즈 간의 모호성을 명시적으로 처리합니다. 동일한 문장도 다양한 머리 움직임으로 표현 가능하므로, 분포를 모델링하는 것이 결정론적 회귀보다 적절합니다.[1]

#### 3.3 장시간 의존성 처리

확대 컨볼루션 아키텍처는 **255프레임(4.25초)의 수용 영역**을 가지므로 장기 의존성을 효과적으로 모델링합니다. 이는:[1]

- LSTM의 오버피팅 경향을 완화
- 고정된 수용 영역으로 장시간 정보를 체계적으로 처리
- 작은 데이터셋(3-5분)에서 더 안정적인 학습

#### 3.4 다양한 음성 도메인으로의 일반화 증거

실험에서 **야생(unseen) 오디오**에 대한 성능:[1]

- Obama Weekly Address 데이터셋에서 3-5분으로만 학습 후 새로운 음성 클립으로 테스트
- 사용자 연구에서 **20개의 야생 오디오 클립** 모두에서 사실적 결과 달성
- **Text-to-Speech (TTS) 시스템** 생성 음성에도 성공적으로 적용

#### 3.5 후보 이미지 집합의 역할

렌더링 네트워크의 **후보 이미지 집합**은 일반화에 중요한 역할:[1]

- 배경 변화에 대한 강건성 증가 (카메라 움직임 포함)
- 치아, 모공 등 세부 사항 합성 부담 경감
- 학습 데이터 범위 밖의 포즈에 대한 성능 유지

Table 3에서:[1]
- 후보 이미지 없음: L1 = 6.683, LPIPS = 0.746
- 후보 이미지 포함: L1 = 5.713, LPIPS = 0.698 (**약 15% 개선**)

#### 3.6 일반화 성능 향상의 한계 및 미래 방향

**현재 한계**:[1]

1. **개인화된 말하기 스타일 유지의 딜레마**: Manifold Projection을 통한 도메인 적응은 야생 음성에 강건하지만, 개인화된 스타일 세부사항을 일부 상실할 수 있음

2. **감정-음성 분리 부재**: 현재 시스템은 중립 음성으로 학습된 경우 감정적 음성에 대응 불가

3. **음성 특성 분리 부족**: 완전한 해결책은 음성 분리(Qian et al. 2020) - 내용, 피치, 음색, 리듬 분해가 필요

**제안된 개선 방향**:[1]

- **음성 디엔탱글먼트**: 감정, 스타일, 내용을 분리하는 고급 음성 분석
- **다중 모달 학습**: 텍스트, 음성, 시각 정보의 통합
- **메타-러닝**: 매우 짧은 비디오(초 단위)에서의 신속한 적응

***

### 4. 논문의 후속 연구에 미치는 영향 및 고려사항

#### 4.1 연구 분야에 미친 영향

**LSP는 2021년 발표 이후 토킹헤드 생성 분야에 획기적 영향을 미쳤습니다**:[7][3][8][2]

**1. NeRF 기반 방법론의 대중화**:

- **DFA-NeRF (2022)**: LSP의 확률적 모델 개념을 NeRF와 결합, 음성-랜드마크 매핑을 2단계로 분해[9]
- **GeneFace (2023)**: 일반화 가능한 NeRF 기반 토킹헤드 생성으로 199회 인용[10]
- **S3D-NeRF (2024)**: 단일 이미지에서 시작하는 NeRF 기반 방법으로 LSP의 효율성 개념 발전[11]

**2. 3D Gaussian Splatting으로의 진화 (2024-2025)**:

최근 연구들은 NeRF의 느린 렌더링을 개선하기 위해 3D Gaussian Splatting을 도입:[12][2][7]

| 방법 | 연도 | 핵심 기여 | 렌더링 속도 |
|------|------|---------|----------|
| **Gaussian-Face** | 2025 | FLAME 기반 립 모션 + 하이브리드 밀도 | **160 FPS** |
| **GE-Talker** | 2025 | 의미론적 깊이 인식 샘플링 | **기존 대비 109% 향상** |
| **SynGauss** | 2024 | 입 표현 계수 + 지역 멀티헤드 어텐션 | **실시간** |
| **PGSTalker** | 2025 | 픽셀-인식 밀도 제어 | **뛰어난 립싱크** |

이들은 모두 LSP의 **다양한 모달리티의 명시적 제어 개념**을 계승합니다.[2][7][12]

**3. 확률적/다양성 모델링의 확산**:

- **Probabilistic Speech-Driven 3D Facial Motion Synthesis (CVPR 2024)**: LSP의 확률적 접근을 발전시켜 새로운 벤치마크와 평가 지표 제시[13][14]
- **FaceTalk (2024)**: 잠재 확산 모델로 표현 공간에서의 확률적 합성[15]
- **MoDA (2025)**: 다중 모달 확산 아키텍처로 모션-오디오-보조 조건 간 상호작용 강화[3]

**4. 도메인 적응 개념의 재조명**:

LSP의 Manifold Projection은 음성 표현 학습의 중요성을 강조:[1]

- **음성 표현 학습 개선**: wav2vec, HuBERT 등 자기지도학습(Self-Supervised Learning) 모델의 도입
- **DAMC (Dual Audio-Centric Modality Coupling, 2025)**: 내용-인식 인코더와 동적-동기 인코더의 이원 구조로 LSP의 다중-특성 추출 개념 발전[16]

***

#### 4.2 앞으로의 연구에서 고려할 점

**4.2.1 일반화 성능 향상**:[7][2][1]

1. **보편적 음성 표현 학습**:
   - LSP의 APC에서 진화하여 보다 강건한 사전학습 모델 도입 필요
   - 다국어, 악센트, 감정 변동 등 광범위한 음성 도메인 커버
   - 제안: 대규모 비라벨 음성 데이터(CommonVoice 기반) 활용

2. **적응적 Manifold Projection**:
   - 고정된 k-NN(k=10) 대신 동적 수이웃 선택
   - 소스-타겟 도메인 유사성에 따른 가중치 조정
   - 비선형 투영 메커니즘 탐색 (LLE의 한계 극복)

3. **메타-러닝 기반 빠른 적응**:
   - 매우 짧은 비디오(수초)에서의 신속 파인튜닝
   - LSP의 3-5분 요구사항 단축 가능성

**4.2.2 다중 모달리티 통합**:[17][18][3]

1. **텍스트-음성-비디오 통합**:
   - 텍스트 입력 → 음성 합성 → 토킹헤드 생성 통합 파이프라인
   - OmniTalker (2025): 텍스트에서 동시에 음성과 토킹헤드 생성[17]

2. **감정 및 스타일 제어**:
   - LSP의 중립 음성 한계 극복
   - EmoGene (2024): 감정-인식 랜드마크 생성[19]
   - Emotional Audio-Driven Video Portraits (2021): 감정 벡터를 입력으로

3. **제스처 및 신체 움직임**:
   - LSP의 상체 빌보드 모델 확장
   - 팔 움직임, 손 제스처 등 포함

**4.2.3 렌더링 기술의 진화**:[2][7][1]

1. **암시적 vs 명시적 표현**:
   - NeRF (암시적): 고품질 렌더링, 느린 속도
   - 3D Gaussian Splatting (준-명시적): 실시간 성능, 편집 가능성
   - NLDF (2024): 라이트 필드 기반 30배 속도 향상[20]

2. **신경 렌더러의 강건성**:
   - LSP의 UNet 기반 이미지-이미지 변환 외 새로운 아키텍처
   - 극적 포즈 변화에 대한 성능 개선

**4.2.4 평가 메트릭의 고도화**:[21][18]

현재 L1, PSNR, SSIM은 다음의 한계가 있습니다:[18][21]

- **음성-입술 동기화**: Sync-C, SyncC 등 동기화 특화 지표
- **얼굴 움직임 자연성**: 생체역학적 제약 고려
- **개인화 정도**: 스타일 보존율 정량화
- **인지적 품질**: 사용자 연구를 보완할 생리적 지표 (아이트래킹, 근전도 등)

**4.2.5 윤리 및 안전성 고려**:[22][23][24][1]

LSP와 후속 연구의 중요한 사회적 책임:[1]

1. **딥페이크 위험**:
   - 3-5분의 짧은 비디오로 개인화된 아바타 생성 가능
   - 특히 비유명인(non-celebrity)의 음성/얼굴 악용 위험 높음

2. **대응책**:
   - 얼굴 위변조 탐지 기술(FaceForensics)의 병행 개발
   - 생성 영상에 대한 암호화 인증(Digital Face Forensics)
   - 법적/기술적 프레임워크 수립

3. **책임 있는 공개**:
   - 연구 결과의 공개 시점과 방식 신중히 결정
   - 학술 커뮤니티와의 협력적 안전성 평가

***

### 5. 2020년 이후 관련 최신 연구 탐색

#### 5.1 시간순 연구 진화 맵

```
2020년 하반기: 기초 NeRF 기반 방법 출현
├─ Dynamic Neural Radiance Fields (Dec 2020)[7]

2021년: LSP 발표 및 음성-특성 기반 방법 확산
├─ Live Speech Portraits (Sep 2021)[1]
├─ AD-NeRF (Aug 2021)[44]

2022년-2023년: NeRF 기반 방법의 다양화
├─ DFA-NeRF (Jan 2022)[10]
├─ GeneFace (Jan 2023, 199회 인용)[51]
├─ HiDe-NeRF (May 2023)[45]
├─ From Pixels to Portraits 종합 서베이 (Aug 2023)[12]
├─ Probabilistic Speech-Driven 3D Facial Motion (Nov 2023)[47]

2024년: 3D Gaussian Splatting 혁명
├─ NLDF (Neural Light Dynamic Fields, Jun 2024)[11]
├─ EmoGene (Emotion-aware, Oct 2024)[34]
├─ Landmark-guided Diffusion (Aug 2024)[14]
├─ FaceTalk (Motion Diffusion, Mar 2024)[15]
├─ Audio-Driven Facial Animation Survey (Oct 2024)[21]
├─ Gaussian-Face (Feb 2025)[8]
├─ SynGauss (실시간, Apr 2024)[6]

2025년: 최신 동향
├─ GE-Talker (Jun 2025)[2]
├─ PGSTalker (Sep 2025)[32]
├─ DAMC (Dual Audio-Centric, Mar 2025)[31]
├─ MoDA (Multi-modal Diffusion, Jul 2025)[4]
├─ OmniTalker (Text-driven, 실시간)[20]
├─ MemoryTalker (Personalized, ICCV 2025)[22]
├─ GLDiTalker (Speech-driven with Graph, IJCAI 2025)[25]
```

#### 5.2 주요 기술 발전 축

**축 1: 표현 방식의 진화**

| 기간 | 방법론 | 핵심 특징 | 예시 |
|------|--------|---------|------|
| 2021 | 2D 이미지 + 랜드마크 | 희소 특성, 빠른 처리 | LSP, MakeItTalk |
| 2021-2022 | NeRF (암시적) | 고품질, 느린 렌더링 | AD-NeRF, DFA-NeRF |
| 2023-2024 | NeRF + FLAME | 3D 기하학적 제약 | GeneFace, S3D-NeRF |
| 2024-2025 | 3D Gaussian Splatting | 실시간, 편집 가능 | Gaussian-Face, GE-Talker |

**축 2: 모션 모델링의 고도화**

| 기간 | 접근법 | LSP와의 관계 |
|------|--------|------------|
| 2021 | 결정론적 회귀 | LSP보다 선행 |
| 2021-2023 | 확률적 모델 | **LSP가 도입한 개념의 확산** |
| 2023-2024 | 확산 모델 (Diffusion) | 확률적 모델링의 차세대 |
| 2024-2025 | 다중 모달 확산 | 음성, 텍스트, 포즈의 통합 |

**축 3: 일반화 전략**

| 전략 | 수행 연도 | 기술 세부사항 |
|------|---------|-------------|
| **Manifold Projection** (도메인 적응) | **LSP (2021)** | LLE 기반 음성 표현 재구성 |
| **사전학습 모델** | 2022-2024 | wav2vec, HuBERT 등 자기지도학습 |
| **메타-러닝** | 2023-2024 | 매우 적은 데이터로의 빠른 적응 |
| **지식 증류** | 2024 | 고품질 모델의 경량 모델로의 전이 |

#### 5.3 최신 기술의 구체적 사례

**사례 1: Gaussian-Face (2025) - LSP의 상위 진화**[4]

<div style="background-color: #f0f0f0; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3;">

**LSP와의 비교**:
- **LSP**: 이미지-이미지 변환 네트워크 → 포토리얼리스틱
- **Gaussian-Face**: 3D Gaussian + FLAME + 립 모션 변환기 → 160 FPS 렌더링

**핵심 개선**:

```math
\text{Lip\_Motion\_Translator}: \text{audio features} \to \text{3D lip parameters}
```

FLAME의 명시적 3D 표현과 LSP의 음성-특성 매핑 개념 결합

**성능**: 30fps (LSP) → 160fps (Gaussian-Face) 5배 향상

</div>

**사례 2: MoDA (2025) - 다중 모달 확산**[3]

LSP의 단일 음성 입력 → 확산 모델로 다중 모달리티 처리:

$$\text{Joint Parameter Space}: \text{motion generation} + \text{neural rendering unified}$$

**특징**:
- 흐름 일치(Flow Matching)로 확산 학습 단순화
- 다양한 모션 생성 지원 (LSP의 확률적 모델 개념 발전)
- 실시간 응용 가능

**사례 3: OmniTalker (2025) - 텍스트-음성-영상 통합**[17]

LSP의 음성 기반 → 텍스트에서 완전 생성:

$$\text{Text} \to \text{Multi-modal Transformer} \to [\text{Speech} + \text{Talking Head}]$$

**혁신**:
- 단일 프레임 + 텍스트 입력으로 25fps 실시간 생성
- 스타일 보존: 참조 비디오 1개로 음성/얼굴 스타일 학습
- 영점 학습(Zero-shot) 설정에서 기존 방법 능가

***

### 6. 결론 및 향후 전망

**Live Speech Portraits**는 다음과 같은 이유로 토킹헤드 생성 분야의 **기초적 작업**입니다:[1]

1. **실시간 포토리얼리스틱 생성의 첫 구현**: 30fps 이상의 실시간 성능은 이전에 달성되지 않음

2. **Manifold Projection을 통한 도메인 적응**: 음성 표현 학습에서 도메인 갭을 직접적으로 해결한 최초의 정식화

3. **확률적 모션 모델링**: 1-다 매핑의 모호성을 확률 분포로 처리, 이후 확산 모델 연구의 기초 마련

4. **포괄적 3단계 파이프라인**: 음성 특성 추출 → 모션 생성 → 포토리얼리스틱 합성의 명확한 분해가 이후 연구의 템플릿 제공

**2020년 이후의 후속 연구 추세**:

- **암시적 표현 (NeRF)**: 2021-2023 지배적 패러다임
- **3D Gaussian Splatting**: 2024-2025 실시간 성능의 새로운 기준 수립
- **확산 모델**: 다양한 모션과 다중 모달리티 처리의 새로운 방향
- **텍스트 기반 확장**: LSP의 음성 단일성을 벗어나 완전 생성형 시스템으로 진화

**연구자가 고려할 중요 포인트**:

1. **일반화 vs 개인화의 균형**: 도메인 적응은 개인화 특성 손실 위험
2. **실시간 성능 vs 품질**: GPU 메모리와 처리 속도 간 트레이드오프
3. **윤리적 책임**: 딥페이크 위험에 대한 적극적 대응
4. **평가 메트릭의 개선**: 인지적/생체역학적 정확성을 반영한 새로운 지표 개발

LSP는 단순히 우수한 기술이 아니라, **음성 기반 영상 합성의 새로운 패러다임을 제시한 기원점**으로서 향후 10년 이상의 토킹헤드 생성 연구를 구조화하고 있습니다.

***

### 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e3f48d87-31d7-4e9e-9c2c-d3826a7f8c8e/2109.10595v2.pdf)
[2](https://ieeexplore.ieee.org/document/11210076/)
[3](https://arxiv.org/abs/2507.03256)
[4](https://dl.acm.org/doi/10.1145/3728725.3728752)
[5](https://arxiv.org/html/2305.06225)
[6](https://arxiv.org/abs/2509.16922)
[7](https://ieeexplore.ieee.org/document/10889648/)
[8](https://ieeexplore.ieee.org/document/11145199/)
[9](https://www.semanticscholar.org/paper/664d2fcb440338a313bcf089e804aef64d94c2eb)
[10](https://arxiv.org/abs/2301.13430)
[11](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01618.pdf)
[12](https://ieeexplore.ieee.org/document/10910134/)
[13](https://arxiv.org/abs/2311.18168)
[14](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Probabilistic_Speech-Driven_3D_Facial_Motion_Synthesis_New_Benchmarks_Methods_and_CVPR_2024_paper.html)
[15](http://arxiv.org/pdf/2312.08459.pdf)
[16](https://arxiv.org/abs/2503.22728)
[17](https://arxiv.org/html/2504.02433v1)
[18](https://eprints.bournemouth.ac.uk/40586/)
[19](https://ieeexplore.ieee.org/document/11099460/)
[20](https://arxiv.org/abs/2406.11259)
[21](https://arxiv.org/pdf/2308.16041.pdf)
[22](https://ieeexplore.ieee.org/document/9578714/)
[23](https://github.com/harlanhong/awesome-talking-head-generation)
[24](https://patents.google.com/patent/WO2024010484A1/en)
[25](https://www.semanticscholar.org/paper/34ef1b53a9e0103bd251a8d74ec1a9565b44340c)
[26](https://arxiv.org/html/2312.05572)
[27](http://arxiv.org/pdf/2408.01732.pdf)
[28](http://arxiv.org/pdf/2201.00791v1.pdf)
[29](http://arxiv.org/pdf/2307.03270.pdf)
[30](http://arxiv.org/pdf/2404.19038.pdf)
[31](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_MemoryTalker_Personalized_Speech-Driven_3D_Facial_Animation_via_Audio-Guided_Stylization_ICCV_2025_paper.pdf)
[32](https://arxiv.org/html/2507.02900v1)
[33](https://research.nvidia.com/publication/2017-07_audio-driven-facial-animation-joint-end-end-learning-pose-and-emotion)
[34](https://www.ijcai.org/proceedings/2025/0173.pdf)
[35](https://arxiv.org/abs/2103.11078)
[36](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_One-Shot_High-Fidelity_Talking-Head_Synthesis_With_Deformable_Neural_Radiance_Field_CVPR_2023_paper.pdf)
[37](https://arxiv.org/html/2505.01319v1)
[38](https://www.sciencedirect.com/science/article/abs/pii/S0957417425029598)
[39](https://eprints.bournemouth.ac.uk/40586/1/information-15-00675.pdf)
[40](https://linkinghub.elsevier.com/retrieve/pii/S0262885624002087)
[41](https://ieeexplore.ieee.org/document/10799951/)
[42](https://www.semanticscholar.org/paper/a7dc2299281d0bfb5c8e8fca8a473eabcf21023a)
[43](https://arxiv.org/abs/2308.16041)
[44](https://arxiv.org/html/2501.14646v1)
[45](https://arxiv.org/html/2411.19525v2)
[46](https://arxiv.org/pdf/2312.10921.pdf)
[47](https://arxiv.org/pdf/2211.12368.pdf)
[48](https://arxiv.org/pdf/2203.07931.pdf)
[49](https://arxiv.org/pdf/2308.16576.pdf)
[50](https://people.cs.umass.edu/~mahadeva/papers/sub_528.pdf)
[51](https://jd92.wang/assets/files/a11_mm18.pdf)
[52](https://arxiv.org/html/2507.11949)
[53](https://sail.usc.edu/~cvaz/papers/manifold_is.pdf)
[54](https://liner.com/ko/review/geneface-generalized-and-highfidelity-audiodriven-3d-talking-face-synthesis)
[55](https://diglib.eg.org/bitstream/handle/10.2312/pg20231274/081-088.pdf)
[56](https://www.diva-portal.org/smash/get/diva2:1988881/FULLTEXT01.pdf)
[57](https://proceedings.neurips.cc/paper_files/paper/2024/file/86b3697c4eb7792c951831636bfdacd5-Paper-Conference.pdf)
[58](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Probabilistic_Speech-Driven_3D_Facial_Motion_Synthesis_New_Benchmarks_Methods_and_CVPR_2024_paper.pdf)
