# Symbolic Music Generation with Diffusion Models

### 1. 논문의 핵심 주장 및 주요 기여

**"Symbolic Music Generation with Diffusion Models"** (Mittal et al., 2021)는 이산적 음악 데이터에 대해 확산 모델(Diffusion Models)을 처음으로 성공적으로 적용한 획기적인 연구입니다. 이 논문의 핵심 주장은 다음과 같습니다.[1]

기존 확산 모델들은 연속 영역(continuous domain)에서만 효과적이었으나, 이산적(discrete) 음악 데이터에는 적용이 제한적이었습니다. 본 논문은 사전 훈련된 변분 자동 인코더(Variational AutoEncoder, VAE)의 연속 잠재 공간(continuous latent space)에서 확산 모델을 훈련하는 기법을 제시합니다. 이를 통해 1024개 토큰의 장형 음악 시퀀스를 고품질로 생성할 수 있습니다.[1]

**주요 기여:**
- **비자기회귀 생성(Non-autoregressive Generation)**: 자동 회귀 모델과 달리 순차적 이전 토큰에 의존하지 않으며 병렬 생성이 가능합니다.[1]
- **노출 편향(Exposure Bias) 제거**: 교사 강제(Teacher Forcing)를 사용하지 않아 자동 회귀 기선인 TransformerMDN보다 우수한 성능을 달성합니다.[1]
- **사후 조건부 채우기(Post-hoc Conditional Infilling)**: 훈련 없이 음악 시퀀스의 특정 부분을 창의적으로 생성할 수 있습니다.[1]

***

### 2. 해결하고자 하는 문제 및 제안 방법

#### 2.1 문제 정의

확산 확률 모델(DDPM)은 이미지와 오디오 등 연속 영역에서 우수한 성능을 보였으나, 음악 기호 생성에는 두 가지 근본적인 한계가 있습니다:[2][3]

1. **이산 데이터 불일치**: 음악 기호는 음고, 음높이, 악기 등 이산적 토큰으로 표현되며, DDPM의 가우시안 노이즈 기반 샘플링 프로세스는 연속 공간에 최적화되어 있습니다.[1]

2. **장형 구조 모델링 한계**: 기존 음악 VAE는 짧은 시퀀스(2마디)에만 효과적이며, 장형 음악의 시간적 의존성을 모델링하기 어렵습니다.[1]

#### 2.2 제안하는 방법론

**다단계 계층적 접근(Multi-stage Hierarchical Approach)**

논문은 두 단계 파이프라인을 제안합니다:

**1단계: MusicVAE 인코딩**

원본 MIDI 시퀀스를 사전 훈련된 2마디 MusicVAE로 인코딩합니다:
$$z_i = \text{MusicVAE}_{\text{encoder}}(\text{MIDI}_i), \quad i=1,\ldots,32$$

여기서 각 2마디 프레이즈는 42차원의 연속 잠재 벡터 $z_i$로 변환되며, 이를 통해 64마디 전체 시퀀스를 32개의 잠재 임베딩으로 표현합니다.[1]

**2단계: 확산 모델 훈련**

잠재 임베딩 시퀀스 $x_0 = [z_1, \ldots, z_k]$에 대해 확산 모델을 훈련합니다. 전진 프로세스는 고정 노이즈 스케줄에 따라 이산 가우시안 노이즈를 점진적으로 추가합니다:

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

$$q(x_{1:N}|x_0) = \prod_{t=1}^{N} q(x_t|x_{t-1})$$

여기서 $\beta_1, \beta_2, \ldots, \beta_N$은 노이즈 스케줄이고, $N=1000$, $\beta_1 = 10^{-6}$, $\beta_N = 0.01$입니다.[1]

역방향 프로세스는 파라미터 $\theta$를 가진 마르코프 체인으로 정의됩니다:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_\theta(x_t, t))$$

$$p_\theta(x_{0:N}) = p(x_N) \prod_{t=1}^{N} p_\theta(x_{t-1}|x_t)$$

훈련 목적함수는 간단한 제곱 손실로, 노이즈 예측 네트워크 $\epsilon_\theta$를 최적화합니다:

$$\mathcal{L}(\theta) = \mathbb{E}_{x_0, \epsilon, t} \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \right\|^2$$

여기서 $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod\_{i=1}^{t}\alpha_i$, $\epsilon \sim \mathcal{N}(0, I)$입니다.[1]

$$\text{ELBO}: \mathbb{E}[\log p_\theta(x_0)] \geq \mathbb{E}_q\left[\log p(x_N) + \sum_{t \geq 1} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]$$

***

### 3. 모델 구조

#### 3.1 트랜스포머 확산 네트워크

비원 네트워크 $\epsilon_\theta(x_t, \sqrt{\bar{\alpha}}) : \mathbb{R}^{k \times 42} \times \mathbb{R} \rightarrow \mathbb{R}^{k \times 42}$는 다음과 같이 구성됩니다:[1]

- **입력 처리**: 42차원 전처리된 잠재 임베딩을 128차원 공간으로 투영
- **자기 주의 인코더**: $L = 6$개 층, 각 층은 $H = 8$개 자기 주의 헤드와 잔차 완전 연결 층 보유
- **위치 인코딩**: 정현파 위치 인코딩은 다음과 같이 정의됩니다:

$$\omega = \left[10^{-4 \times 0/63}j, \ldots, 10^{-4 \times 63/63}j\right]$$
$$e_j = [\sin(\omega), \cos(\omega)]$$

여기서 $j$는 잠재 입력 임베딩의 위치 인덱스입니다.[1]

- **노이즈 조건화**: 특성 선형 조절(Feature-wise Linear Modulation, FiLM)을 사용하여 각 잔차 층의 층 정규화 출력에 스케일($\gamma$)과 시프트($\xi$) 파라미터를 적용합니다.[1]
- **출력 층**: $K = 2$개 노이즈 조건화 잔차 완전 연결 층으로 역진행 프로세스 출력 생성 (각 층 2048 뉴런)

#### 3.2 잠재 공간 정제(Latent Trimming)

MusicVAE는 대부분의 잠재 차원을 활용하지 않는 문제(posterior collapse)를 보입니다. 이를 해결하기 위해 훈련 세트 전체에서 평균 표준편차가 1.0 이하인 42개 차원만 유지합니다:[1]

$$\text{Trim}(z) = [z_i : \sigma_i < 1.0], \quad i = 1, \ldots, 512$$

***

### 4. 성능 향상 및 한계

#### 4.1 성능 평가

**프레임별 자기-유사성 메트릭(Framewise Self-Similarity Metric)**

논문은 4마디 슬라이딩 윈도우(2마디 홉 크기)를 사용하여 음정(pitch)과 음가(duration) 통계를 측정합니다. 각 프레임 $k$에 대해 가우시안 확률밀도함수 $\mathcal{N}(\mu_1, \sigma_1^2)$과 $\mathcal{N}(\mu_2, \sigma_2^2)$를 정의하고, 인접 프레임의 겹침 영역(Overlapping Area, OA)을 계산합니다:[1]

$$OA(k, k+1) = 1 - \text{erf}\left(\frac{c-\mu_1}{\sqrt{2\sigma_1^2}}\right) + \text{erf}\left(\frac{c-\mu_2}{\sqrt{2\sigma_2^2}}\right)$$

이를 정규화하여 일관성(Consistency)과 분산(Variance)을 계산합니다:

$$\text{Consistency} = \max\left(0, 1 - \frac{|\mu_{OA} - \mu_{GT}|}{\mu_{GT}}\right)$$

$$\text{Variance} = \max\left(0, 1 - \frac{|\sigma_{OA}^2 - \sigma_{GT}^2|}{\sigma_{GT}^2}\right)$$

**결과:**
- 확산 모델은 비조건 생성에서 음정 일관성 0.99, 음정 분산 0.90을 달성하여 TransformerMDN (0.93, 0.68)을 압도합니다.[1]
- 잠재 공간 평가(Fréchet Distance, MMD)에서는 TransformerMDN이 약간 우수하지만, 실제 음악 품질 측정(프레임별 자기-유사성)에서는 확산 모델이 더 우수합니다.[1]

#### 4.2 주요 한계

1. **계산 복잡도**: $N=1000$ 단계의 반복 정제가 필요하여 자동 회귀 모델보다 샘플링 시간이 길습니다.[1]

2. **다중 악기 제한**: 논문은 단성 선율(monophonic melody)에만 집중하며, 다중 악기 다중 트랙 음악으로의 확장이 명확하지 않습니다.[1]

3. **음악적 규칙**: 확산 모델은 음악 이론(조화, 대위법)을 학습하지 않아 음악적 오류가 발생할 수 있습니다.[1]

4. **데이터 의존성**: MusicVAE의 사전 훈련 품질에 크게 의존합니다.[1]

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 현재 일반화 성능 분석

논문의 모델은 다음과 같은 일반화 강점을 보입니다:

**1. 비자기회귀 구조의 이점**

자동 회귀 모델은 교사 강제(teacher forcing) 훈련으로 인해 노출 편향이 발생합니다. 즉, 훈련 시 정확한 이전 토큰을 사용하지만 추론 시 모델의 예측을 사용하므로 분포 이동이 발생합니다. 반면 확산 모델은 모든 잠재 임베딩을 동시에 모델링하므로:

$$\text{Diffusion: } p_\theta(x_{1:N}) = p(x_N)\prod_{t=1}^{N} p_\theta(x_{t-1}|x_t)$$

는 각 단계에서 전체 시퀀스의 joint distribution을 학습하여 노출 편향이 없습니다.[1]

**2. 계층적 구조의 이점**

VAE의 저수준 특징 학습과 확산 모델의 고수준 시간 의존성 모델링을 결합하여, 훈련 데이터의 분포 외(out-of-distribution) 샘플에 대한 robust한 생성이 가능합니다.[1]

#### 5.2 일반화 성능 향상 가능성

**1. 신경망 아키텍처 개선**

최근 연구들은 트랜스포머 대신 구조화된 상태 공간 모델(Structured State Space Models, SSM)인 Mamba를 활용하여 계산 복잡도를 선형으로 감소시킵니다. SMDIM(Symbolic Music Diffusion with Mamba)은:[4]

- 자기 주의의 이차 복잡도 문제 해결
- 긴 시퀀스(1000+마디)에 대한 확장성 개선
- 중국 민속음악 등 다양한 음악 스타일에 대한 성능 향상 입증[4]

**2. 조건부 생성 메커니즘 강화**

최신 연구는 확산 모델에 명시적 제약을 도입하여 음악적 규칙을 학습합니다:[3]

- **엔트로피 정규화 CRF 기반 조화 제약(ERLD-HC)**: 변분 추론으로 음악 이론 규칙을 학습 가능한 특징 함수로 인코딩하여, 음악 규칙 위반률을 2.35% 감소[3]
- **세밀한 가이던스(Fine-Grained Guidance, FGG)**: 음표 밀도, 음역대, 윤곽, 리듬 복잡도 등 다양한 음악적 속성에 대한 fader 제어[5]

**3. 데이터 증강 및 전이 학습**

최신 모델들은 다양한 음악 스타일 데이터셋을 활용하여 일반화를 개선합니다:[6]

- NotaGen: LLM 패러다임의 사전 훈련, 미세 조정, 강화학습을 적용하여 고전 악보 생성의 음악성 향상
- Direct Preference Optimization(DPO)을 통해 인간 주석 없이도 음악성과 제어성 향상[6]

**4. 다중 신호 및 다중 모드 학습**

최근 연구는 여러 조건부 신호를 통합합니다:

- **PIMG(Progressive Image-to-Music Generation)**: 감정을 교량으로 하여 이미지-음악 교차 모드 생성[7]
- **MusDiff**: 텍스트와 이미지 멀티모달 입력으로 음악 품질과 교차 모드 일관성 향상[8]
- **Multi-Track MusicLDM**: Transformer와 확산 모델 결합으로 다중 트랙 음악의 트랙 간 일관성 개선[9]

#### 5.3 일반화 성능 한계 및 해결 방향

**한계:**

1. **음악적 표현력의 한계**: 단순 음정-음가 통계만으로는 감정, 음악적 표현, 예술적 의도를 담기 어렵습니다.[10]

2. **계산 효율성**: 확산 모델의 반복 정제(iterative refinement)는 여전히 자동 회귀 모델보다 느립니다.[11]

3. **장형 구조 모델링**: 64마디를 넘는 대규모 음악 구조(형식, 반복, 변주)의 모델링이 제한적입니다.[4]

**해결 방향:**

- **고속 샘플링**: DDIM(Denoising Diffusion Implicit Models) 기법으로 스텝 수 감소 (1000→50)
- **적응형 노이즈 스케줄**: 초기/후기 단계에서 스텝 크기 자동 조정
- **계층적 생성**: 음악 형식(verse/chorus) 수준의 의미론적 구조를 먼저 생성 후 세부사항 생성

***

### 6. 최신 관련 연구 (2020년 이후)

#### 6.1 기호 음악 생성 확산 모델 진화

**기본 확산 모델 확장 (2021-2022)**
- DiffWave (Chen et al., 2021): 파형 오디오 생성용 다목적 확산 모델 제시, 자동 회귀/GAN 기반 모델 능가[3]
- 이산 확산 모델(D3PM, 2023): DDPM을 이산 공간에 직접 적용하여 음표 수준의 infilling 지원[12]

**고급 아키텍처 (2023-2025)**
- **Mamba 기반 확산(2025)**: SMDIM은 Mamba의 선형 복잡도로 긴 시퀀스(FolkDB 전통 중국 민속음악) 처리, TransformerMDN 능가[4]
- **프로그레시브 이미지-음악 생성(2025)**: 감정 중개를 통해 이미지에서 음악으로의 교차 모드 생성[7]

#### 6.2 조건부 생성 및 제어 (2023-2025)

**음악 규칙 제약**
- **조화 제약 확산(ERLD-HC, 2025)**: 엔트로피 정규화 CRF로 음악 이론 규칙 학습, 조화 규칙 위반 2.35% 감소[3]
- **SYMPLEX(2024)**: Simplex 확산과 어휘 사전으로 악기, 음역대, 인필링 제어[12]

**세밀한 제어 메커니즘**
- **FGG(Fine-Grained Guidance, 2025)**: 음표 밀도, 음역대, 리듬 복잡도 등 속성별 fader 제어[11]
- **조건부 제약 확산(2025)**: 사전 훈련된 비조건 모델 위에 작은 조건 확산 모델들을 적층하여 다양한 음악 속성 제어[5]

#### 6.3 멀티트랙 및 장형 음악 생성 (2024-2025)

**다중 악기/트랙 생성**
- **Multitrack Music Generation(2025)**: Transformer + 확산 모델 결합, 트랙 정렬 손실로 리듬/다이나믹 일관성 개선[9]
- **Multi-Track MusicLDM(2024)**: 다중 트랙 속성(장르, 분위기) 제어로 교차 모드 생성[13]

**장형 음악 생성**
- **DiffRhythm(2025)**: 보컬+반주 완전한 노래 최대 4분45초 생성, 일관된 음악 구조 유지[14]
- **Long-form Music Generation(2024)**: 21.5Hz 잠재율로 연속 잠재 표현 처리, 자동 회귀 및 이산 확산 모델 능가[15]

#### 6.4 효율성 개선 (2023-2025)

**빠른 샘플링**
- **MeLoDy(2023)**: LM 가이드 확산으로 MusicLM 대비 95.7%-99.6% 전진 패스 감소[16]
- **Low-Latency Symbolic Music(2024)**: 속성 기반 가속으로 실시간 즉흥 연주 및 인간-AI 공동 창작 지원[17]

**간소화된 아키텍처**
- **Musimple(2025)**: GTZAN 소규모 데이터셋, 2D 멜 스펙트로그램으로 VAE 없이 훈련, FAD 5.0 달성[18]

#### 6.5 음악성 및 표현성 강화 (2024-2025)

**강화 학습 통합**
- **NotaGen(2025)**: LLM 패러다임(사전 훈련/미세 조정/강화학습), CLaMP-DPO로 음악성 및 제어성 개선[6]
- **Quality-aware MDT(2024)**: 품질 인식 훈련 전략으로 입력 음악 파형 품질 판별[19]

**텍스트 조건부 생성**
- **Noise2Music(2023)**: 시간적 계층화(생성기 → cascader)로 30초 고화질 음악 텍스트 생성[20]
- **Text-to-Music 의미론(2025)**: 최신 모델들의 음악 의미론 포착 능력 검증[21]

***

### 7. 논문이 앞으로의 연구에 미치는 영향

#### 7.1 패러다임 전환

**이산 데이터에 대한 확산 모델의 개방**

Mittal et al.(2021)의 연구는 **"이산 데이터는 연속 잠재 공간에서 처리한다"**는 중요한 통찰을 제시합니다. 이는 텍스트 생성, 단백질 구조 예측 등 다양한 이산 데이터 도메인에서 확산 모델 적용의 기초가 되었습니다.[22][1]

더 나아가, Song et al.(2021)의 스코어 기반 생성 모델링 프레임워크는 DDPM을 확률적 미분 방정식(SDE)으로 일반화하여, 확산 모델이 단순한 이미지 생성을 넘어 **과학 및 통계적 응용의 근본적인 생성 패러다임**으로 진화하도록 영감을 주었습니다.[22]

#### 7.2 계층적 생성 아키텍처의 정립

Mittal et al.(2021)은 **VAE의 저수준 특징 학습 + 확산 모델의 고수준 시간 구조 모델링**이라는 계층적 접근을 확립했습니다. 이후 연구들은 이를 다양하게 확장합니다:

- **VQ-VAE-2(Razavi et al., 2019)**: 이미지에서 다계층 양자화 VAE 제시[23]
- **Jukebox(Dhariwal et al., 2020)**: 다단계 모델(코드북 → 샘플)로 음악 생성[10]
- **DALL-E(Ramesh et al., 2021)**: 이미지 토큰 → 이미지로 확산 모델 적용[13]

이는 **"복잡한 이산 데이터 생성은 계층적 표현 학습의 연쇄"**라는 원칙을 정립했습니다.[1]

#### 7.3 비자기회귀 시퀀스 생성의 가능성

노출 편향 제거로 TransformerMDN을 능가한 결과는 다음과 같은 후속 연구를 촉발했습니다:

- **Masked Diffusion Transformer(2024)**: 음악 생성에 마스킹 기반 확산 적용[19]
- **양자화 이산 확산(2023)**: 피아노롤 기반 이산 확산으로 다중 악기 음악 생성[12]
- **SIMPLEX(2024)**: Simplex 기하학과 어휘 사전으로 제어 가능한 기호 음악 생성[12]

이들은 비자기회귀 시퀀스 생성이 **자동 회귀 모델의 성능을 능가할 수 있음**을 확증합니다.[1]

#### 7.4 사후 조건부 생성(Post-hoc Conditioning)의 활용

Mittal et al.(2021)이 시연한 **무조건 훈련된 모델의 사후 조건부 infilling**은 창의 응용에서 혁신적입니다. 이는 다음과 같이 확장되었습니다:[1]

- **Plug & Play 생성(2017, 2019)**: 기울기 기반 조건화로 재훈련 없이 이미지/음악 제어[37-39]
- **Latent Constraints(2018)**: 사전 훈련 VAE 위에서 조건부 제어 추가[4]

최신 사후 조건부 연구는 **세밀한 속성 제어(Fine-Grained Guidance)**로 발전하여, 음악가가 음악적 의도를 실시간으로 조정할 수 있습니다.,[5][11]

***

### 8. 앞으로의 연구 시 고려할 점

#### 8.1 아키텍처 설계 최적화

**1. 상태 공간 모델(State Space Models) 활용**

Mamba 기반 확산(SMDIM)이 보여주듯이, 트랜스포머의 이차 자기 주의 복잡도는 장형 음악 생성의 병목입니다. 향후 연구는:[4]

- **선형 시간 복잡도 아키텍처 우선 고려**: 최소한 자기 주의의 O(n²) 대신 O(n log n) 이상의 효율성 목표
- **구조 보존 설계**: 음악의 계층적 구조(음표→마디→악장) 반영 아키텍처

**2. 다중 시간 스케일 모델링**

현재 확산 모델은 단일 샘플링 경로를 사용합니다. 향후는:

$$q_{\text{coarse}}(x_t^{\text{bar}}) \rightarrow q_{\text{fine}}(x_t^{\text{note}})$$

형태의 계층적 확산을 고려하여 악장 구조부터 음표 세부사항까지 다중 시간 스케일에서 생성

#### 8.2 음악 이론 제약 통합

**1. 조화 및 대위법 규칙**

ERLD-HC의 CRF 기반 접근을 확장하여:[3]

- **음악적 그래프 신경망**: 코드 진행(chord progression)과 성부 독립성(voice leading)을 그래프 구조로 인코딩
- **제약 만족 최적화**: 생성 후 음악 규칙 위반을 디코딩 단계에서 경정
- 음악 이론 규칙을 손실 함수에 명시적으로 통합:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}} + \lambda \mathcal{L}_{\text{harmony}} + \mu \mathcal{L}_{\text{rhythm}}$$

**2. 스타일 및 장르 제어**

- **스타일 벡터 학습**: 작곡가 스타일, 시대, 문화적 배경을 명시적 잠재 변수로
- **자동 회귀적 스타일 전이**: 생성 과정 중 음악 스타일 점진적 변화 모델링

#### 8.3 멀티모달 및 교차 모드 학습

**1. 텍스트-음악 일관성**

Noise2Music, DiffRhythm의 성과를 기반으로:,[14][20]

- **의미론적 대정렬**: 음악적 특성(밝음/어두움)과 텍스트 임베딩 공간 정렬
- **음악-언어 사전 훈련**: CLaMP-2와 유사한 멀티모달 음악 정보 검색 모델 활용[6]

**2. 이미지-음악, 이모션-음악 연계**

PIMG의 감정 중개 접근을 확장하여:[7]

- **다중 이모션 표현**: 단순 valence-arousal 평면 대신 음악적 감정의 다차원 공간 모델링
- **추상적 표현**: 파형, 색상, 시각적 동향을 음악적 특성과의 추상적 매핑

#### 8.4 평가 방법론의 혁신

현재 평가는 자동 평가 지표(FAD, MMD)의 한계를 보입니다.,,[24][10][1]

**1. 음악성 평가**

- **음악 이론 기반 메트릭**: 조화 규칙 준수율, 멜로디 일관성 지수, 리듬 복잡도 측정
- **계층적 평가**: 음표 수준, 마디 수준, 악장 수준의 각각 평가
- **인간 평가 표준화**: BLEU, ROUGE와 유사한 음악 품질 메트릭 정립

**2. 창의성 평가**

- **신성성(Novelty)**: 훈련 데이터와의 거리 측정
- **다양성(Diversity)**: 생성 샘플 간 음악적 거리
- **일관성(Coherence)**: 원곡과 조건부 생성 간 의미론적 일관성

**예제 메트릭:**
$$\text{Musicality} = \frac{1}{M}\sum_{m=1}^{M} \text{HarmonyCompliance}_m + \text{RhythmCoherence}_m$$

$$\text{Diversity} = \frac{1}{N(N-1)}\sum_{i \neq j} d_{\text{musical}}(x_i, x_j)$$

#### 8.5 실시간 및 인터랙티브 응용

**1. 지연 시간 최적화**

현재 $N=1000$ 단계는 실시간 음악 생성(악기 연주자와의 상호작용)에 부적합합니다:[17]

- **적응형 스텝 스케줄**: 초기 단계에서는 큰 스텝, 최종 단계에서 작은 스텝
- **병렬 확산**: 여러 잠재 벡터를 동시에 처리하여 batch 병렬화
- **증류(Distillation)**: 큰 확산 모델을 작은 모델로 증류하여 추론 가속

**2. 음악가-AI 공동 창작**

- **점진적 세밀화**: 음악가가 지정한 마디에서만 확산 시작
- **실시간 피드백**: 음악가의 즉각적인 수정을 모델이 학습 및 적용
- **속성 기반 제어**: Continuous fader로 리듬, 음역대, 감정 실시간 조정[11]

#### 8.6 데이터 및 일반화

**1. 다양한 음악 문화 포함**

현재 연구는 서양 음악(MIDI)에 집중합니다.,  향후는:[4][1]

- **비서양 음악**: 인도 클래식, 아랍 음악, 동아시아 음악의 미시적 음정(microtonal) 표현
- **악기 다양성**: 합성음, 타악기, 전자음악 통합
- **문화적 아키텍처**: 특정 음악 문화의 규칙을 모델 설계에 통합

**2. 전이 학습 및 소량 데이터 학습**

- **메타 학습**: 새로운 악기/스타일에 빠르게 적응하는 모델
- **자기지도 학습(Self-supervised)**: 라벨이 없는 MIDI 데이터에서 표현 학습
- **컨텍스트 인-디스트리뷰션 문제**: 훈련과 테스트 분포 차이 완화

***

### 9. 결론 및 향후 방향

"Symbolic Music Generation with Diffusion Models"은 **확산 모델의 이산 데이터 적용 가능성을 최초로 입증**한 연구로, 음악 생성 분야의 패러다임을 전환했습니다. 비자기회귀 생성, 노출 편향 제거, 계층적 아키텍처라는 핵심 통찰은 최근 5년간의 후속 연구를 이끌었습니다.[1]

2020-2025년 최신 연구들은:

1. **아키텍처 진화**: 트랜스포머 → Mamba로의 계산 효율성 개선[4]
2. **음악 규칙 통합**: CRF, 강화학습을 통한 음악 이론 제약,[6][3]
3. **멀티모달 학습**: 텍스트, 이미지, 이모션과의 교차 모드 연계,,[20][7][6]
4. **실시간 응용**: 저지연 생성과 인간-AI 공동 창작,[17][11]

그러나 여전히 도전 과제가 있습니다:

- **음악적 표현력**: 긴 시간 스케일의 형식, 반복, 변주 생성
- **창의성과 다양성**: 훈련 데이터 모방 대신 진정한 창의적 합성
- **평가 방법론**: 음악 이론과 인간 지각 기반의 종합적 평가 지표 정립

향후 연구는 **음악 AI를 단순한 데이터 기반 합성에서 음악 이론, 창의성, 문화적 다양성을 통합한 참된 음악 창작 도구**로 발전시켜야 할 것입니다.

***

### 참고 문헌 및 인용

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d37ff629-94b8-4cdd-a4d7-bc87cdd98efb/2103.16091v2.pdf)
[2](https://ieeexplore.ieee.org/document/11209133/)
[3](https://www.mdpi.com/1099-4300/27/9/901)
[4](https://arxiv.org/abs/2507.20128)
[5](https://www.semanticscholar.org/paper/4082ea505f19d2f7630bb02b2dd001e35a0e2aaf)
[6](https://www.ijcai.org/proceedings/2025/1134.pdf)
[7](https://ieeexplore.ieee.org/document/11071375/)
[8](https://www.sciencedirect.com/science/article/pii/S1110016825006738)
[9](https://ieeexplore.ieee.org/document/11152114/)
[10](https://www.ewadirect.com/proceedings/tns/article/view/30045)
[11](https://arxiv.org/pdf/2410.08435.pdf)
[12](https://www.semanticscholar.org/paper/Symbolic-Music-Generation-with-Diffusion-Models-Mittal-Engel/93d00ea9c87268f867b4addb8043be35d6996d18)
[13](https://arxiv.org/html/2409.02845v2)
[14](https://arxiv.org/html/2503.01183v1)
[15](https://arxiv.org/pdf/2404.10301.pdf)
[16](https://arxiv.org/pdf/2305.15719.pdf)
[17](https://arxiv.org/abs/2510.00395)
[18](https://ieeexplore.ieee.org/document/11011064/)
[19](http://arxiv.org/pdf/2405.15863.pdf)
[20](https://arxiv.org/pdf/2302.03917.pdf)
[21](https://www.nature.com/articles/s41467-025-66731-7)
[22](https://yang-song.net/blog/2021/score/)
[23](https://ieeexplore.ieee.org/document/11228274/)
[24](https://www.nature.com/articles/s41598-025-13064-6)
[25](https://zenodo.org/doi/10.5281/zenodo.17706331)
[26](https://arxiv.org/ftp/arxiv/papers/2301/2301.13267.pdf)
[27](https://www.emergentmind.com/topics/score-based-generative-modeling)
[28](https://arxiv.org/html/2507.20128v1)
[29](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/sbgm/)
