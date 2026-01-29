# SoundStream: An End-to-End Neural Audio Codec

***

### **1. SoundStream의 핵심 주장과 주요 기여**

**핵심 주장**

SoundStream은 신경망 기반 오디오 코덱으로, 3kbps~18kbps의 극저비트레이트에서 음성, 음악, 일반 오디오를 모두 처리하면서도 기존의 표준 코덱(Opus, EVS)보다 우수한 품질을 달성한다는 혁신적인 주장을 제시합니다. 특히 SoundStream 3kbps는 Opus 12kbps 및 EVS 9.6kbps와 동등한 품질을 제공하여, **약 3~4배의 비트레이트 효율성 향상**을 입증했습니다.

**주요 기여**

1. **엔드-투-엔드 신경망 코덱**: 인코더, 양자화기, 디코더를 공동으로 학습하는 완전 미분가능 시스템으로, 전통적인 신호 처리 기반 코덱과 달리 데이터 중심 최적화가 가능합니다.

2. **잔차 벡터 양자화기(RVQ)**: 다단계 벡터 양자화 구조로 codebook 폭증 문제를 해결하며, 이는 이후 모든 신경망 코덱의 표준이 되었습니다.

3. **양자화기 드롭아웃**: 단일 모델에서 1.5kbps~18kbps의 다양한 비트레이트를 지원하는 혁신적 기법으로, 기존 코덱은 각 비트레이트마다 별도 모델을 필요로 했습니다.

4. **범용 모델과 저지연 구현**: 스마트폰 CPU에서 실시간 구동이 가능하면서도 음성과 음악을 구분 없이 처리하는 최초의 모델입니다.

5. **통합 압축-향상(Compression-Enhancement)**: 코덱 내에서 노이즈 제거를 추가 지연 없이 수행 가능합니다.

***

### **2. SoundStream이 해결하는 문제 및 제안 방법**

#### **2.1 해결하는 문제**

| 문제 | 기존 접근 | SoundStream의 해결책 |
|------|---------|-------------------|
| **극저비트레이트에서의 품질 저하** | Opus/EVS는 6kbps 이상에서만 만족스러운 품질 | RVQ + GAN 손실로 3kbps에서도 우수한 품질 달성 |
| **범용 모델의 한계** | 음성 특화 코덱(매개변수 기반)은 음악에 실패 | 학습된 인코더로 모든 오디오 타입에 대응 |
| **다중 비트레이트 지원의 비효율성** | 각 비트레이트마다 별도 모델 필요 | 양자화기 드롭아웃으로 단일 모델 지원 |
| **실시간 구현의 어려움** | WaveNet 기반 코덱은 순차 생성으로 느림 | 완전 합성곱 아키텍처로 13ms 지연 달성 |
| **압축과 향상의 분리 처리** | 인코더-향상-디코더 파이프라인으로 지연 증가 | 조건부 처리로 통합 처리 가능 |

#### **2.2 제안하는 방법: 수식 중심 분석**

**전체 파이프라인:**
$$\hat{x} = \text{Decoder}(\text{RVQ}(\text{Encoder}(x)))$$

where $x \in \mathbb{R}^{T}$ is the input audio signal and $\hat{x}$ is the reconstructed audio.

**인코더 구조:**

인코더는 완전 합성곱 구조로 시간 다운샘플링을 수행합니다:

$$z_{\text{enc}} = \text{Conv1D}_{\text{final}}(\text{ResBlocks}_{\text{Benc}}(\text{Conv1D}_{\text{init}}(x)))$$

여기서:
- 각 ResBlock은 팽창(dilation) 비율 1, 3, 9를 갖는 3개의 구조화된 잔차 단위 포함
- 스트라이딩 시퀀스 $[2,4,5,8]$로 총 **$M=320$배 다운샘플링** 달성 [j-ilkominfo](https://j-ilkominfo.org/index.php/ejournalaikom/article/view/264)
- 24kHz 입력에서 **75Hz 프레임 레이트** 생성

$$S = \lfloor T/M \rfloor, \quad z_{\text{enc}} \in \mathbb{R}^{S \times D}$$

**잔차 벡터 양자화(RVQ):**

RVQ의 핵심은 다단계 양자화를 통한 점진적 정제입니다:

$$\hat{z} = \sum_{i=1}^{N_q} Q_i(r_i)$$

where the residuals are computed as:

$$r_1 = z_{\text{enc}}, \quad r_{i+1} = r_i - Q_i(r_i) \quad \text{for } i = 1, \ldots, N_q-1$$

각 양자화기 $Q_i$는 다음과 같이 작동합니다:

$$Q_i(r_i) = c_{Q_i(k_i(r_i))} = \arg\min_{c \in \mathcal{C}_i} \|r_i - c\|_2$$

여기서:
- $\mathcal{C}_i$는 크기 $N=2^{10}=1024$의 학습된 codebook
- $k_i(r_i)$는 가장 가까운 codebook 인덱스

**비트레이트 결정:**

총 비트레이트는:
$$R = \frac{S \cdot N_q \cdot \log_2(N)}{d \cdot f_s} \text{ bps}$$

where:
- $S$: 프레임 수
- $N_q$: 양자화기 수
- $d$: 오디오 지속시간(초)
- $f_s = 24\text{kHz}$: 샘플링 레이트

예시: $N_q=8$, $N=1024$ (10 bits/codebook)일 때:
$$R = \frac{75 \cdot 8 \cdot 10}{1 \text{ sec} \cdot 1} = 6\text{ kbps}$$

**양자화기 드롭아웃 (Quantizer Dropout):**

훈련 시 각 배치마다 $n_q$를 균등 분포로 샘플링:

$$n_q \sim \text{Uniform}(\{1, 2, \ldots, N_q\})$$

추론 시에는 원하는 비트레이트에 따라 $n_q$ 고정:

$$\hat{z}_{\text{inference}} = \sum_{i=1}^{n_q} Q_i(r_i)$$

이는 structured dropout의 형태로, 모든 가능한 비트레이트에서 작동하는 단일 모델을 가능하게 합니다.

**손실 함수:**

생성기 손실은 다음의 가중치 조합:

$$\mathcal{L}_G = \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}(G) + \lambda_{\text{feat}} \mathcal{L}_{\text{feat}}(x, \hat{x}) + \lambda_{\text{rec}} \mathcal{L}_{\text{rec}}(x, \hat{x})$$

**대적 손실 (Hinge Loss):**

$$\mathcal{L}_{\text{adv}}(G) = \frac{1}{K} \sum_{k=0}^{K-1} \frac{1}{T_k} \sum_{t=0}^{T_k-1} \max(0, 1 - D_k(\hat{x}_t))$$

여기서 $D_k$는 $k$번째 판별자 (다중해상도 및 STFT 기반)

**특징 손실 (Feature Loss):**

$$\mathcal{L}_{\text{feat}}(x, \hat{x}) = \frac{1}{KL} \sum_{k=0}^{K-1} \sum_{l=0}^{L-1} \|D_k^l(x) - D_k^l(\hat{x})\|_1$$

**다중스케일 스펙트로그램 손실:**

$$\mathcal{L}_{\text{rec}}(x, \hat{x}) = \sum_{s \in S} \left[\|S_s(x) - S_s(\hat{x})\|_1 + \|\log S_s(x) - \log S_s(\hat{x})\|_2\right]$$

where $S = \{2^5, 2^6, \ldots, 2^{11}\}$ and $S_s$ is mel-spectrogram at scale $s$.

**디코더 구조:**

인코더의 미러 구조:

$$\hat{x} = \text{ConvTranspose1D}_{\text{final}}(\text{ResBlocks}_{\text{Bdec}}(\text{Conv1D}_{\text{init}}(\hat{z})))$$

역순 스트라이딩으로 업샘플링하여 원래 해상도 복구. [e-journal.hamzanwadi.ac](https://e-journal.hamzanwadi.ac.id/index.php/infotek/article/view/26083)

**조건부 처리 (FiLM레이어):**

노이즈 제거 활성화 신호 $c$에 대해:

$$a_{n,c}' = \gamma_{n,c} \cdot a_{n,c} + \beta_{n,c}$$

where $\gamma_{n,c}$ and $\beta_{n,c}$ are learned from $c$.

***

### **3. 모델 구조 상세 분석**

#### **3.1 인코더 아키텍처**

**구성 요소:**
- 초기 Conv1D: kernel=7, channels= $C_{\text{enc}}=32$
- 4개 인코더 블록 (각각 3개 팽창 잔차 단위)
- 최종 Conv1D: kernel=3, output= $D$ (임베딩 차원)

**차원 변화:**
- 입력: $T = 24000$ (1초)
- 블록 1 (stride=2): $12000 \times 64$ 채널
- 블록 2 (stride=4): $3000 \times 128$ 채널
- 블록 3 (stride=5): $600 \times 256$ 채널
- 블록 4 (stride=8): $75 \times 512$ 채널
- 최종 출력: $75 \times 128$ (D=128)

#### **3.2 RVQ의 Codebook 관리**

**초기화:**
- K-means로 첫 배치에서 초기화하여 codebook이 데이터 분포에 근접
- 미사용 벡터 교체: 지수이동평균(decay=0.99)이 2 이하인 벡터를 현재 배치의 임의 프레임으로 교체

**학습:**
- EMA (Exponential Moving Average) 업데이트:
$$c_i^{(t)} \leftarrow \alpha \cdot c_i^{(t-1)} + (1-\alpha) \cdot \text{mean}(\{z : \arg\min_j \|z-c_j\| = i\})$$

**최적화:**
- Straight-Through Estimator (STE)로 양자화 그래디언트 추정
- 커밋먼트 손실: $\mathcal{L}\_w = \|z_{\text{enc}} - \text{sg}(\hat{z})\|_2^2$

#### **3.3 판별자 (이중 구조)**

**파형 기반 판별자 (Multi-Resolution):**
- 3개 스케일 (원본, 2배 다운샘플, 4배 다운샘플)
- 각 스케일에서 grouped convolution 사용

**STFT 기반 판별자:**
- Window size: 1024, hop size: 256
- 복소값 STFT (실수부, 허수부 연결)
- 2D Conv로 시간-주파수 영역 학습

***

### **4. 성능 향상 및 한계 분석**

#### **4.1 성능 향상 (MUSHRA 점수)**

**대비 기존 코덱:**
$$\text{SoundStream (3kbps)} \approx \text{Opus (12kbps)} \approx \text{EVS (9.6kbps)}$$

| 비트레이트 | 음성 | 노이즈 음성 | 음악 |
|----------|-----|----------|------|
| 3kbps | 67.0 | 62.5 | 89.6 |
| 6kbps | 83.1 | 69.4 | 92.9 |
| 12kbps | 90.6 | 80.1 | 91.8 |

**범위 간 효율성:**
- 음악: 3kbps에서도 Opus 12kbps를 초과
- 노이즈 음성: 저비트레이트에서 도전적이나 여전히 우수

#### **4.2 확장성 성능**

**단일 모델 vs. 비트레이트 특화 모델:**

$$\text{ViSQOL}\_{\text{scalable}} \approx \text{ViSQOL}_{\text{specific}} \text{ at } 9\text{, }12\text{ kbps}$$

실제로 9kbps와 12kbps에서는 양자화기 드롭아웃이 정규화 효과를 통해 특화 모델을 능가합니다.

#### **4.3 한계점**

1. **저비트레이트의 노이즈 음성**: 3~6kbps에서 노이즈 음성 품질 저하 (MUSHRA 41.3~62.5)
2. **엔트로피 코딩 미사용**: 기존 구현에서 추가 25~40% 압축 기회 미활용
3. **프레임 레이트 고정**: 24kHz에서 75Hz의 고정 프레임 레이트로 인한 비효율성
4. **계산 복잡도**: RTF (Real-Time Factor) = 2.3 (스마트폰 CPU에서 실시간이나 고속 아님)
5. **트레이닝 요구사항**: 대규모 다양 데이터셋과 다중 GPU 필요

***

### **5. 모델의 일반화 성능 향상 가능성**

#### **5.1 현재 일반화 성능**

**도메인 외 테스트:**
- Clean speech, Noisy speech, Music, Reverberant speech 모두에서 일관된 성능
- 학습하지 않은 오디오 타입 (예: 음악의 악기 변형)에서도 우수

**원인:**
1. **학습된 인코더의 견고성**: 고정 mel-spectrogram과 달리 적응적 특징 추출
2. **다양한 데이터 혼합**: LibriTTS(음성), Freesound(노이즈), MagnaTagATune(음악) 결합
3. **비트레이트 드롭아웃의 정규화 효과**: 과적합 방지

#### **5.2 일반화 향상 방향**

**아키텍처 개선:**
- **비트레이트 적응형 양자화기**: 음성/음악을 자동 감지하여 $N_q$ 동적 조정

$$N_q^* = f_{\text{detect}}(\text{Content Type}, \text{Audio Features})$$

- **다중 해상도 인코더**: 단일 프레임 레이트 대신 다중 시간 스케일 특징 추출

$$z_{\text{enc}}^{(s)} = \text{Encoder}^{(s)}(x) \text{ for } s \in \{\text{coarse, fine}\}$$

**훈련 개선:**
- **더 큰 다양성 데이터셋**: 현재 음성/음악/노이즈 외에 음성 효과, 악기 고립, 언어 다양성 추가
- **도메인 확대 훈련**: $\mathcal{L}\_{\text{domain}} = \lambda_{\text{domain}} \sum_{\text{domains}} \mathcal{L}_{\text{rec}}$
- **적대적 학습의 강화**: 판별자를 영역 외 데이터에 사전훈련

**양자화 개선:**
- **변수 비트레이트 RVQ (VRVQ)**: 프레임마다 $N_q$ 동적 조정으로 무음 구간에서 비트 절약
  $$N_q[t] = \arg\max_{n} (\text{importance}[t, n] > \tau)$$

***

### **6. 2020년 이후 관련 최신 연구 비교 분석**

#### **6.1 주요 진화 개요**

신경망 오디오 코덱의 진화는 SoundStream을 기준으로 네 가지 주요 방향으로 진행되었습니다:
1. **개선된 RVQ**: 더 효율적인 양자화 체계
2. **멀티스케일 설계**: 다양한 시간 해상도에서 동시 처리
3. **다운스트림 통합**: 생성 모델 친화적 토큰화
4. **계산 효율성**: 경량 아키텍처 추구

#### **6.2 핵심 코덱 비교**

| 항목 | SoundStream (2021) | EnCodec (2022) | DAC (2023) | SNAC (2024) | HILCodec (2024) |
|------|----------------|-------------|---------|----------|------------|
| **샘플 레이트** | 24kHz | 24/48kHz | 44.1kHz | 24/32/44kHz | 24kHz |
| **비트레이트** | 3-18kbps | 1.5-24kbps | 3-24kbps | 0.98-3.9kbps | 3-12kbps |
| **범위** | 범용 | 범용 | 범용 | 범용 (음악 강화) | 음성/음악 |
| **아키텍처** | SEANet | SEANet+LSTM | ConvNeXt | RVQGAN+MS | SEANet 경량 |
| **RVQ 특성** | 기본 RVQ | 기본 RVQ | 기본 RVQ | **멀티스케일** | 기본 RVQ |
| **주요 혁신** | 양자화기 드롭아웃 | 단일 MS-STFT 판별자, 손실 균형 | 고보스콧 인코더 구조 | **다중 프레임 레이트** | **분산 제약 설계**, 노필터 판별자 |
| **MUSHRA (3kbps)** | 67-77 | 67-76 | N/A | 77+ | 75 |
| **실시간성 (CPU)** | ✓ (13ms) | ✓ (13ms) | ~ (높은 계산) | ~ | ✓✓ (경량) |
| **매개변수** | ~18M | ~14.9M | ~75M | ~19.8M | ~4M |

#### **6.3 EnCodec의 진보**

**핵심 기여:**

1. **단일 Multi-Scale STFT 판별자 (MS-STFTD)**
   - 기존 다중 판별자(MSD+Mono-STFTD) 대비 훈련 간편화
   - 5개 스케일로 STFT 윈도우 크기 조정: $\{2048, 1024, 512, 256, 128\}$
   - 2D Conv로 시간-주파수 학습

$$\mathcal{L}_{\text{adv}}^{\text{MS-STFT}} = \frac{1}{|S|} \sum_{s \in S} \mathcal{L}_{\text{hinge}}(D_s(\hat{x}))$$

2. **손실 균형기 (Loss Balancer)**
   - 그래디언트 스케일 정규화로 하이퍼파라미터 해석성 향상:

$$\tilde{g}_i = R \cdot \frac{\lambda_i}{\sum_j \lambda_j} \cdot \frac{g_i}{\langle\|g_i\|_2\rangle_\beta}$$

- 각 가중치를 전체 그래디언트의 비율로 해석 가능

3. **엔트로피 코딩**
   - 작은 Transformer LM (5층, 8헤드)로 상관성 모델링
   - **25~40% 추가 압축**: 6kbps → 4.1kbps

EnCodec은 본질적으로 SoundStream의 정제된 버전으로, 훈련 안정성 개선과 엔트로피 코딩 추가로 실용성 향상.

#### **6.4 DAC의 특수한 역할**

**독특한 특성:**

1. **BigVGAN 기반 구조**
   - Snake activation: $x + \frac{1}{a}\sin^2(ax)$ (주기 신호에 민감)
   - 음악 코덱화에 특화된 아키텍처

2. **높은 압축률**
   - 44.1kHz에서 3kbps: 현저한 효율성
   - 음악: DAC 16kbps ≈ EnCodec 24kbps

3. **언어 모델 친화적**
   - 900 토큰/초 프레임 레이트로 생성 모델 입력 최적화
   - AudioLM, Jukebox 등 기초 코덱

**한계:**
- 토널 콘텐츠(악기) 재구성에 어려움
- 계산 복잡도 높음 (75M+ 매개변수)

#### **6.5 SNAC의 다중스케일 혁신**

**핵심 기여:**

다양한 시간 해상도에서 동시 양자화:

$$\hat{z} = \sum_{s=1}^{S} \sum_{i=1}^{N_{q,s}} Q_{s,i}(r_{s,i})$$

where each scale $s$ has its own quantizer cascade at temporal resolution $t_s$.

**예시 구조:**
- Coarse (10Hz, 2 codebooks): 거시적 구조
- Mid (20Hz, 4 codebooks): 중간 특징
- Fine (47Hz, 8 codebooks): 세밀한 디테일

**효과:**
- 3.9kbps에서 0.98kbps로 비트레이트 40% 감소
- 음악: SNAC 1.9kbps ≈ DAC 3kbps

**한계:**
- 구현 복잡도 증가
- 생성 모델과의 토큰 정렬 어려움

#### **6.6 DualCodec의 의미론적 접근 (2025)**

**혁신:**

SSL(Self-Supervised Learning) 표현 통합:

$$z_{\text{acoustic}}, z_{\text{semantic}} = \text{DualStream}(x)$$

첫 RVQ 레이어가 의미론적 정보 직접 인코딩:

$$Q_1^{\text{semantic}} = \arg\min_{c \in \mathcal{C}_1} \|f_{\text{SSL}}(x) - c\|_2$$

**결과:**
- 0.75kbps에서 MUSHRA 79.5~83.5 달성
- 음성 생성 모델 (TTS) 품질 30% 향상

**의미:**
- 코덱이 순수 음성 복원에서 **의미론적 의도 전달**로 진화
- 매우 저비트레이트 영역에서 의미론적 토큰 우선

#### **6.7 최신 방향: VCNAC (변수채널, 2025)**

**특징:**
- 단일 모델로 모노/스테레오/5.1 서라운드 지원
- 채널 간 상호작용 모델링

***

### **7. SoundStream의 장기적 영향과 미래 연구 방향**

#### **7.1 학술적 영향**

**기초 확립:**
- RVQ는 이후 모든 신경 코덱의 표준이 됨 (EnCodec, DAC, SNAC, HILCodec 등)
- 양자화기 드롭아웃은 조건부 생성의 선례

**인용 수:**
- SoundStream: ~1,200 인용 (2021년 이후)
- 영향 지수: 신경망 오디오 분야에서 기초 논문으로 확립

**개방성:**
- Google이 SEANet 및 코드 공개로 재현성 확보
- 학계의 빠른 추종 가능

#### **7.2 SoundStream 기반 발전**

**직접적 개선:**
| 논문 | 핵심 개선 | 성과 |
|-----|---------|------|
| EnCodec | MS-STFT 판별자, 손실 균형 | 훈련 안정성 + 25% 압축 |
| SNAC | 멀티스케일 RVQ | 40% 비트레이트 감소 |
| HILCodec | 경량 아키텍처 + 노필터 판별자 | 모바일 최적화 |
| DualCodec | 의미론적 토큰화 | 극저비트레이트 성능 |

#### **7.3 미해결 문제**

1. **극저비트레이트 의미론적 손실** (< 1kbps)
   - 현재: 음성 식별만 가능
   - 목표: 감정, 음향 특징 보존

2. **비음성 음향의 일반화**
   - 환경 음향, 악기, 음향 효과
   - 다중 모달리티 (음악+음성+효과) 통합

3. **실시간 적응 양자화**
   - 현재: 고정 프레임 레이트
   - 미래: 동적 $N_q$ 결정으로 무음 구간 최적화

4. **생성 모델과의 공설계**
   - 코덱이 생성 모델의 학습을 촉진하는 구조 설계
   - 토큰 시퀀스 길이 vs 재구성 품질 균형

#### **7.4 향후 연구 시 고려사항**

**아키텍처 측면:**
- **Transformer 인코더 도입**: 장거리 의존성 모델링 (SNAC 주의 레이어)
- **다중 분해(Multi-Decomposition) RVQ**: 음향/의미론적 채널 분리 (DualCodec)
- **적응적 프레임 레이트**: 콘텐츠 복잡도에 따른 동적 조정 (VRVQ)

**훈련 개선:**
- **다중 작업 학습**: 코덱 + 향상 + 분리 + 언어 이해를 통합 목표
- **영역별 최적화**: 음성/음악/효과 각각의 가중 손실
- **메타학습**: 새로운 도메인에 빠른 적응

**평가 체계:**
- **의미론적 품질 지표**: 현재 MUSHRA는 재구성 품질만 측정, 의미 보존도 평가 필요
- **다운스트림 태스크 벤치마크**: TTS, ASR, 음악 정보 검색 성능 통합 평가

**배포 고려사항:**
- **극도의 경량화**: 2MB 이하 모델로 엣지 디바이스 지원
- **양방향 호환성**: 이전 버전과의 호환 보장 메커니즘
- **적응적 해석**: 네트워크 상태에 따른 실시간 비트레이트 조정

***

### **결론**

SoundStream은 **신경망 기반 오디오 코덱의 패러다임 전환점**입니다. RVQ 도입과 양자화기 드롭아웃으로 극저비트레이트 범용 코덱의 가능성을 입증했으며, 이후 4년간의 모든 주요 코덱(EnCodec, DAC, SNAC, HILCodec 등)이 SoundStream의 기초 위에 구축되었습니다.

**주요 진화 방향:**
- **EnCodec**: 훈련 안정성 + 엔트로피 코딩
- **SNAC**: 다중스케일 RVQ로 프레임 레이트 최적화
- **DualCodec/UniCodec**: 의미론적 토큰화로 생성 모델 통합

앞으로의 연구는 **극저비트레이트 의미론적 정보 보존**, **다중 도메인 통합**, **생성 모델과의 공설계**에 집중될 것으로 예상됩니다. 특히 음성 언어 모델이 코덱 토큰 기반으로 수렴하면서, 코덱은 단순 압축 도구에서 **음향 이해의 기초 표현**으로서의 역할이 강화될 것입니다.
