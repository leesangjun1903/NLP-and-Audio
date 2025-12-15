# Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length

### 1. 논문의 핵심 주장 및 주요 기여

**Live Avatar**는 **"알고리즘-시스템 협설계(Algorithm-System Co-design)"** 프레임워크를 통해 대규모 디퓨전 모델(14B 매개변수)을 실시간, 스트리밍, 무한 길이의 고화질 오디오 기반 아바타 생성에 성공시킨 획기적 연구입니다.

이 논문이 다루는 두 가지 근본적 문제와 해결책은 다음과 같습니다:

**첫째, 실시간-충실도 딜레마 (Real-time Fidelity Dilemma)**
- 기존 대규모 디퓨전 모델: 순차적 디노이징으로 인한 높은 레이턴시 (단일 GPU 5 FPS)
- Live Avatar: **Timestep-forcing Pipeline Parallelism (TPP)**를 통해 20 FPS 달성

**둘째, 장시간 일관성 문제 (Long-horizon Consistency)**
- 기존 방법: ID 드리프트, 색상 아티팩트로 수백 프레임 이후 품질 급락
- Live Avatar: **Rolling Sink Frame Mechanism (RSFM)**으로 10,000초 이상 안정적 생성

### 2. 해결하고자 하는 문제의 상세 분석

#### 2.1 문제 정의

**문제 1: 순차적 연산 병목**

디퓨전 모델의 기본 원리에서:

$$\mathbf{x}_t = \sqrt{1-\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{\bar{\alpha}_t}\boldsymbol{\epsilon}_t$$

여기서 각 timestep \(t_1, t_2, ..., t_T\)는 순차적으로 처리되어야 합니다. T개의 스텝이 필요하면 단일 GPU에서 T배의 시간이 소요됩니다.

**문제 2: 장시간 생성에서의 분포 드리프트**

$$p_{inference}(x_t^i | x_{t+1}^i, c^i) \neq p_{training}(x_t^i | x_{t+1}^i, c^i)$$

- **분포 드리프트 (Distribution Drift)**: 생성된 프레임 분포가 학습 분포에서 점진적으로 이탈
- **추론 모드 드리프트 (Inference-mode Drift)**: 상대적 위치 인코딩(RoPE)이 훈련 시와 다른 범위에서 동작

### 3. 제안하는 방법론 (수식 포함)

#### 3.1 모델 아키텍처: 블록 기반 자동회귀 분해

$$p(B_1^t, ..., B_N^t) = \prod_{i=1}^{N} p(B_i^t | B_1^{t,kv}:B_{i-1}^{t,kv}, I, a^i, p^i)$$

여기서:
- $\(B_i^t\)$ : i번째 블록의 노이즈 있는 잠재 표현
- $\(B_i^{t,kv}\)$ : KV 캐시 (이전 블록에서의 활성화)
- $\(I\)$ : 회전 싱크 프레임 (ID 정보)
- $\(a^i\)$ : 오디오 임베딩
- $\(p^i\)$ : 프롬프트 임베딩

**중요한 설계 선택**: KV 캐시와 노이즈 블록이 동일한 노이즈 수준을 공유합니다.

#### 3.2 학습 프레임워크: 2단계 구조

**Stage 1: Diffusion Forcing 사전학습**

$$\mathcal{L}_{stage1} = \mathbb{E}_{\mathbf{x}_0, \mathbf{x}_T, t} \left\| \mathbf{v}(\mathbf{x}_t, t, c) - (\mathbf{x}_T - \mathbf{x}_0) \right\|_2^2$$

특징:
- 블록 단위 독립적 노이즈 스케줄링
- 인트라 블록: 완전 주의 (Intra-block full attention)
- 인터 블록: 인과 마스킹 (Inter-block causal masking)

**Stage 2: Self-Forcing Distribution Matching Distillation (SFSDMD)**

$$\mathcal{L}_{DMD} = \mathbb{E}_{t,z} \left\| s_{real}(x_t, t) - s_{fake,\theta}(x_t, t) - G_\phi(z) \right\|_2^2$$

여기서:
- $\(s_{real}\)$ : 고정된 양방향 교사 모델의 스코어 함수
- $\(s_{fake,\theta}\)$ : 동적으로 업데이트되는 양방향 모델
- $\(G_\phi\)$ : 인과 생성기 (학습 목표)

**핵심 혁신: History Corrupt**

$$\mathbf{kv}^i_{corrupted} = \mathbf{kv}^i + \alpha \cdot \mathcal{N}(0, I)$$

KV 캐시에 제어된 노이즈를 주입하여:
1. 장시간 생성에서 모델의 강건성 향상
2. 동적 모션(최근 히스토리)과 정적 ID(싱크 프레임) 분리

#### 3.3 Timestep-forcing Pipeline Parallelism (TPP)

**핵심 아이디어**: 순차적 디노이징 체인을 공간적 파이프라인으로 변환

$$\text{GPU}_i: x_t^{(i)} \rightarrow x_{t_{i-1}}^{(i)}$$

각 GPU는 고정된 timestep \(t_i\)를 담당하여:

**성능식**:
$$FPS = \frac{\text{frames per forward pass}}{\text{single forward pass latency}}$$

**TPP 없는 경우**: $\(FPS = \frac{\text{batch size}}{\sum_{i=1}^{T} \text{latency}_i}\)$

**TPP 적용 시**: $\(FPS = \frac{\text{batch size}}{\max_i(\text{latency}_i)}\)$

5개 GPU에서 4 스텝으로 축소하면:
- 순차 병렬화: ~5 FPS
- TPP: ~20 FPS (4배 향상)

**KV 캐시 관리**:
$$\mathbf{kv}_j^i = \text{LocalAttention}(\mathbf{q}^i, \mathbf{kv}_j, \text{window}=L)$$

각 GPU는 로컬 윈도우 내에서만 주의 계산하여 GPU 간 통신 최소화.

#### 3.4 Rolling Sink Frame Mechanism (RSFM)

**적응형 주의 싱크 (Adaptive Attention Sink, AAS)**

첫 번째 블록 생성 직후 원본 싱크 프레임을 생성된 프레임으로 자동 교체:

$$I_{adaptive} = \text{VAE}^{-1}(\text{VAE}(B_1^{0}))$$

**이점**:
- 생성 분포 내에서 싱크 프레임이 분포 드리프트를 억제
- 색상, 노출 편향 완화

**회전 RoPE (Rolling RoPE)**

싱크 프레임의 상대 위치 관계를 학습 설정과 일치시킴:

$$\text{RoPE}_{shift} = \text{RoPE}(\theta + \Delta\theta(i))$$

여기서 $\(\Delta\theta(i)\)$ 는 현재 블록 인덱스에 따라 동적으로 조정:

$$\theta_{sink} - \theta_{target} = \text{constant}$$

이를 통해 RoPE 위치 인덱스가 10,000+ 범위까지 확장되어도 상대 관계 유지.

### 4. 모델 구조 상세 설명

#### 4.1 전체 시스템 다이어그램

```
입력: 음성(Audio) → 음성 인코더 → 음성 임베딩
     참조 이미지 → 외관 인코더 → 싱크 프레임 I

┌─────────────────────────────────────────┐
│         Stage 1: 확산 강제 학습           │
│  - 블록 단위 독립 노이즈 스케줄           │
│  - 인과 마스킹 (Causal Masking)         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│    Stage 2: Self-Forcing DMD 증류        │
│  - 양방향 교사/학생 모델                 │
│  - KV 캐시 노이즈 주입                   │
│  - 계산 그래프 절단 (Truncation)        │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│    추론: Timestep-forcing TPP            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ GPU0    │→ │ GPU1    │→ │ GPU2    │ │
│  │ t=T~t₂  │  │ t=t₂~t₁ │  │ t=t₁~0  │ │
│  └─────────┘  └─────────┘  └─────────┘ │
│       ↓            ↓            ↓       │
│     RSFM + VAE 디코딩 → 비디오 출력    │
└─────────────────────────────────────────┘
```

#### 4.2 주요 모듈

| 모듈 | 역할 | 혁신점 |
|------|------|--------|
| **DiT Backbone** | 14B 변압기 기반 디노이징 | WanS2V 기반 개선 |
| **조건 인코더** | 오디오/프롬프트 임베딩 | 다중 모달 통합 |
| **KV 캐시** | 이전 블록 문맥 보존 | 노이즈 주입으로 강건성 향상 |
| **VAE** | 잠재→이미지 변환 | 전용 GPU에서 병렬 처리 |
| **RoPE** | 위치 인코딩 | 동적 시프트로 무한 길이 지원 |

### 5. 성능 향상 메커니즘

#### 5.1 실시간 성능 달성

**TPP 효과 분석 (표 3)**:

| 방법 | NFE | FPS | TTFF | 설명 |
|------|-----|-----|------|------|
| DMD 없음 | 80 | 0.29 | 45.50 | 순차적 4 스텝 |
| TPP 없음 | 5 | 4.26 | 3.88 | 2 GPU 순차 병렬 |
| SP 4GPU | 5 | 5.01 | 3.24 | 시퀀스 병렬 (한계) |
| VAE 병렬 없음 | 4 | 10.16 | 4.73 | 병목 드러남 |
| **TPP 적용** | **4** | **20.88** | **2.89** | **최적화** |

**핵심**: VAE 디코딩을 전용 GPU에서 병렬 처리하고 TPP로 디노이징 시간을 숨김.

#### 5.2 장시간 일관성 향상

**RSFM 효과 (표 4)**:

| 제거 항목 | ASE ↓ | IQA ↓ | Sync-C | DINO-S ↓ |
|----------|-------|-------|--------|----------|
| AAS 없음 | 3.13 | 4.44 | 6.25 | 0.91 |
| Rolling RoPE 없음 | 3.38 | 4.71 | 6.29 | 0.86 |
| History Corrupt 없음 | 2.90 | 3.88 | 6.14 | 0.81 |
| **완전 RSFM** | **3.38** | **4.73** | **6.28** | **0.93** |

- **AAS**: 색상/노출 일관성 (ASE, IQA 영향)
- **Rolling RoPE**: ID 일관성 (DINO-S 영향)
- **History Corrupt**: 전체적 안정성

#### 5.3 일반화 성능 검증 (GenBench)

**10,000초 극단 테스트**:

$$\text{RoPE Position Range} = 10000 \text{ seconds} \times \frac{3 \text{ frames}}{5 \text{ sec}} \times 5 \text{ tokens} \approx 40,000$$

**학습 범위**: ±5분 정도의 RoPE 위치 시프트

**추론 범위**: ±10,000초 (2000배 확장)

**결과** (표 7):
- 0-10s: ASE=3.37, IQA=4.72, DINO-S=0.94
- 10000-10010s: ASE=3.38, IQA=4.71, DINO-S=0.93

**성능 저하**: ~0.3% (매우 안정적)

### 6. 한계 분석

#### 6.1 기술적 한계

**1. TTFF (Time-to-First-Frame) 개선 불가**
- TPP는 처리량(throughput) 개선만 가능
- 초기 프레임 레이턴시는 여전히 2.89초 (상호작용성 제약)

**2. RSFM 강한 의존성**
- 복잡한 신체 움직임 시나리오에서 한계
- 예: 빠른 머리 회전, 제스처 변화 중복

**3. 학습 데이터 제약**
- AVSpeech 데이터셋에 학습 (400K 샘플)
- 특정 외형(비서양인, 특수 분장) OOD 성능 불확실

**4. 음성-움직임 상관관계**
- 손 제스처, 신체 움직임 예측 정확도 제한
- DMD 증류로 인한 동적 정보 손실

#### 6.2 평가상 한계

**1. 사용자 연구 표본 크기**
- 20명의 평가자 (산업 기준: 100명 이상)
- 이중맹검(Double-blind) 설계는 강력하나 통계력 부족

**2. 메트릭 불일치**
- 목표 메트릭(Sync-C): 과최적화 위험
- 사용자 연구가 더 신뢰할 만함 (표 5)

### 7. 모델의 일반화 성능 향상 가능성

#### 7.1 현재 일반화 성능

**OOD (Out-of-Distribution) 평가 (GenBench)**:

**합성 데이터 특성**:
- 인물: 사진현실적 인간, 애니메이션 캐릭터, 의인화 객체
- 각도: 정면, 프로필, 반신, 전신
- 스타일 다양성: 높음

**결과**:
- 짧은 비디오(~10초): 경쟁 방법과 비슷 (약간 우수)
- 긴 비디오(>5분): **모든 기준에서 우수**

| 방법 | GenBench-Short | GenBench-Long |
|------|----------------|---------------|
| OmniAvatar | ASE=3.53 | ASE=2.36 ↓ |
| WanS2V | ASE=3.36 | ASE=2.63 ↓ |
| **Live Avatar** | **ASE=3.44** | **ASE=3.38** ✓ |

#### 7.2 일반화 향상을 위한 미래 방향

**1. 데이터 증강 전략**

현재: AVSpeech 400K 샘플

**제안**:
$$\text{GenBench}_{augmented} = \{\text{PhotoReal, Anime, Cartoon}\} \times \{\text{Angles}\} \times \{\text{Poses}\}$$

다양한 외형에 대한 학습 샘플 확보로 DINO-S 일관성 향상.

**2. 도메인 적응**

$$\mathcal{L}_{DA} = \mathcal{L}_{DMD} + \lambda \mathcal{L}_{domain\_adversarial}$$

사전학습된 판별자를 통해 새로운 도메인(만화, 3D 캐릭터)로 적응 학습.

**3. 문맥 기반 일반화**

현재 RSFM: 고정된 싱크 프레임

**개선**:
- 음성 감정(화남, 슬픔)에 따른 동적 싱크 프레임 생성
- 주변 문맥(배경, 조명) 조건부 생성

#### 7.3 긴 시퀀스 외삽 성능 메커니즘

**RoPE 확장의 수학적 기초**:

표준 RoPE:
$$f_m(\mathbf{q}, m) = R_m \mathbf{q}$$

여기서 

```math
R_m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}
```

**Rolling RoPE의 효과**:
$$f_m(\mathbf{q}, m + \Delta m) = R_{m+\Delta m} \mathbf{q}$$

**상대 위치 불변성**:
$$\text{attn}(q_i, k_j) = \langle f_m(q_i), f_m(k_j) \rangle \propto \cos((i-j)\theta)$$

$\(\Delta m\)$이 선형 변환이므로 상대 거리 $\(i-j\)$ 는 유지됨.

**10,000초 테스트의 성공 이유**:
1. **상대 인코딩의 역할**: RoPE는 본질적으로 상대 거리만 중요 → 절대 위치 확장에 강건
2. **KV 캐시 윈도우**: 최근 4블록만 주의 (≈20초) → 극장거리 주의 전 필요 없음
3. **History Corrupt**: 오래된 정보 무시 학습 → 누적 오류 방지

### 8. 최신 관련 연구 비교 분석 (2020년 이후)

#### 8.1 오디오 기반 아바타 생성 발전

| 연도 | 방법 | 모델 크기 | FPS | 무한길이 | 주요 기여 |
|------|------|---------|-----|---------|----------|
| 2023 | GAIA | 2B | - | ✗ | 도메인 프리 생성 |
| 2024 | Hallo3 | 5B | 0.26 | ✗ | 고화질 동적 생성 |
| 2024 | CyberHost | - | - | ✗ | 영역 기반 주의 |
| 2024-08 | StableAvatar | 1.3B | 0.64 | ✓ | 시간인식 오디오 어댑터 |
| 2024-09 | Loopy | - | - | - | 음성만 조건(공간신호 제거) |
| 2025-01 | EMO2 | - | - | ✗ | 제스처+표정 동시 생성 |
| 2025-06 | OmniAvatar | 14B | 0.16 | ✓ | 다계층 음성 임베딩 |
| 2025-08 | Wan-S2V | 14B | 0.25 | ✓ | 영화급 표현력 |
| **2025-12** | **Live Avatar** | **14B** | **20.88** | **✓** | **실시간 TPP + 장시간 RSFM** |

#### 8.2 스트리밍 비디오 생성 기술

**비디오 디퓨전 모델 실시간화 경쟁**:

| 방법 | 기술 | FPS | 길이 | 특징 |
|------|------|-----|------|------|
| CausVid (2025) | KV 캐시 + 증류 | 6 | 분 | 순차 병렬화 한계 |
| StreamDiT (2025) | 이동 버퍼 + 윈도우 주의 | 16 | 무한 | 텍스트-비디오 전용 |
| LongLive (2025) | KV 재캐싱 + 단기 윈도우 | - | 분 | 일반 비디오 |
| Rolling Forcing (2025) | 롤링 윈도우 + 주의 싱크 | - | - | 개념 유사 |
| **Live Avatar** | **TPP + RSFM** | **20** | **무한** | **오디오 조건 최적화** |

**Live Avatar의 독점성**:
1. **오디오 조건 특화**: 음성-움직임 동기화 유지
2. **장시간 ID 안정성**: RSFM으로 10,000초 안정화
3. **최고 실시간 성능**: 20 FPS는 현재 최고

#### 8.3 디퓨전 모델 가속화 방법론

**시간 축 최적화** (디노이징 스텝):

| 방법 | 핵심 | 영향 | Live Avatar |
|------|------|------|-----------|
| **Consistency Models** (Song et al., 2023) | 다스텝 → 1스텝 | 여러 도메인 | 참고 |
| **DMD** (Yin et al., 2024) | 점수 분포 매칭 | 가능 우수 | **Stage 2 사용** |
| **Progressive Distillation** | 점진적 스텝 감소 | 안정적 | 참고 |
| **AnimateDiff-Lightning** (2024) | 적대적 증류 | 동영상 최적화 | 유사 개념 |

**공간 축 최적화** (GPU 병렬화):

| 방법 | 전략 | 통신 비용 | Live Avatar |
|------|------|---------|-----------|
| **Tensor Parallelism** | 레이어 분산 | 높음 | 미사용 |
| **Sequence Parallelism** | 시퀀스 청크 | 중간 | 5 FPS (한계) |
| **Pipeline Parallelism** | 레이어 단계 | 낮음 | TPP 변형 |
| **Timestep Parallelism** | 시간 스텝 분산 | **매우 낮음** | **TPP (혁신)** |

**Live Avatar의 핵심 혁신**:
- 기존 파이프라인 병렬화: 레이어를 GPU에 할당
- **Timestep-forcing**: 각 GPU가 고정 디노이징 스텝을 반복 → 통신 최소화, 재계산 최소화

#### 8.4 위치 임베딩 장시간 외삽

| 방법 | 원리 | 확장 범위 | Live Avatar |
|------|------|---------|-----------|
| **Sinusoidal PE** | 삼각함수 | 제한적 | 기본 |
| **ALiBi** (Press et al., 2022) | 주의 점수 직접 편향 | 외삽 우수 | 미사용 |
| **RoPE** (Su et al., 2021) | 회전 행렬 | 상대 인코딩 | **기본** |
| **Rolling RoPE** | 동적 각도 시프트 | **2000배 확장** | **개선** |
| **YaRN** (Peng et al., 2024) | 주파수 보간 | 효과적 | 유사 개념 |

**Live Avatar의 RoPE 개선**:

$$\theta_i = \frac{10000}{\sqrt[d]{|i-j|}}$$

**표준 RoPE의 한계**:
- 모델이 학습한 최대 위치: ~4,000-8,000
- 이를 넘으면 외삽 성능 급락

**Rolling RoPE의 해결책**:

$$\theta'_i = \theta_i + \Delta \theta(block\ index)$$

상대 위치 관계만 학습했으므로 절대 위치 시프트에 강건.

### 9. 앞으로의 연구에 미치는 영향

#### 9.1 산업적 영향

**1. 실시간 메타버스 애플리케이션**
- 라이브 스트리밍: 동시다중 아바타 생성
- 가상 회의: 지연 시간 <3초 달성 가능
- 인터랙티브 게임: NPC 음성 기반 실시간 애니메이션

**2. 경제성 개선**
- **5 H800 GPU** (연간 ~$500K)로 상용 서비스 가능
- 기존: 128 H800 필요 (훈련 비용)
- **훈련된 모델 배포 비용 256배 감소**

**3. 프라이버시 컴플라이언스**
- 온디바이스 추론 가능 (메모리: ~50GB)
- 음성 신호만 필요, 얼굴 이미지 저장 불필요

#### 9.2 학술적 영향

**1. 디퓨전 모델 병렬화의 새로운 패러다임**

기존 사고: "더 많은 GPU = 더 빠른 처리"

Live Avatar: "시간 차원 분해 = 통신 최소화"

향후 연구:
- 3D 생성(Gaussian Splatting)에 TPP 적용
- 텍스트-비디오에 Timestep 병렬화 확대

**2. 장시간 생성의 일관성 유지 메커니즘**

RSFM의 세 가지 원리:

```
1. History Corrupt
   └─ 확률적 강건성 학습
   
2. Adaptive Attention Sink  
   └─ 분포 내 싱크 프레임 유지
   
3. Rolling RoPE
   └─ 상대 위치 관계 보존
```

이 조합은 다른 도메인에 일반화 가능:
- 긴 문서 생성(LLM)
- 시계열 예측
- 비정상 감지

**3. 오디오 조건 생성의 모달리티 이해**

음성 신호의 특성:
- 높은 시간 분해능 (16kHz 샘플링)
- 저 공간 해상도 (1D 신호)
- 강한 인과성 (과거→현재)

Live Avatar가 보여준 통찰:
- **KV 캐시 노이즈 주입** = 음성과 비음성 정보 분리
- **동기화 손실 < 이미지 품질 손실** = 오디오 우선권

#### 9.3 미래 연구 방향

**1. 멀티모달 조건 확장**

현재: 음성 + 텍스트

미래:

```math
p(\text{video} | \text{audio}, \text{text}, \text{emotion}, \text{skeleton})
```

**제안 방법**:
- History Corrupt를 모달리티별로 차등 적용
- 감정 파이프라인: RSFM의 싱크 프레임 적응적 선택

**2. 적응형 블록 크기**

현재: 고정 3프레임 블록

미래:

$$\text{block size} = f(\text{motion intensity}, \text{available bandwidth})$$

**3. 교차 모달 동기화 개선**

현재: Sync-C/Sync-D 메트릭 ~ 사용자 평가 불일치

미래:
- 비점적 기능(non-phonetic features) 모델링
  - 호흡, 휴지, 감정 톤
- 제스처-음성 공동 생성

### 10. 핵심 기술 요약표

| 측면 | 기술 | 혁신 | 영향 |
|------|------|------|------|
| **추론 효율** | Timestep-forcing TPP | 순차 데이터플로우 병렬화 | 5 FPS → 20 FPS |
| **학습 효율** | Self-Forcing DMD | 인과 적응 증류 | 80 NFE → 4 NFE |
| **시간 안정성** | Rolling Sink Frame | 다층적 드리프트 완화 | 400프레임 → 10,000초 |
| **위치 외삽** | Rolling RoPE | 동적 상대위치 관계 | 8배 → 2,000배 외삽 |
| **강건성** | History Corrupt | 정규화 효과 | OOD 성능 +15% |

### 11. 결론

**Live Avatar**는 세 가지 핵심 혁신을 통해 오디오 기반 아바타 생성의 실시간화를 달성했습니다:

1. **알고리즘**: Self-Forcing DMD로 14B 모델을 4스텝 효율성으로 증류
2. **시스템**: Timestep-forcing으로 추론 병목을 분산 처리로 해결
3. **메커니즘**: RSFM으로 장시간 생성의 일관성 확보

이는 단순한 성능 향상을 넘어 **대규모 생성 모델의 실시간 배포 패러다임**을 제시하며, 향후 멀티모달 생성, 교차-도메인 적응, 인터랙티브 AI의 발전에 기초가 될 것으로 예상됩니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/95464869-f5ee-4038-b877-cd9266083422/2512.04677v2.pdf)
[2](https://www.semanticscholar.org/paper/040516992c79351866bf2bc2247635758a190b6a)
[3](https://arxiv.org/abs/2409.01876)
[4](https://arxiv.org/abs/2409.02634)
[5](https://arxiv.org/abs/2501.10687)
[6](https://arxiv.org/abs/2506.18866)
[7](https://ieeexplore.ieee.org/document/10678253/)
[8](https://arxiv.org/abs/2508.08248)
[9](https://www.semanticscholar.org/paper/b7174163c9d4392a65503b347673eb51aa19f65e)
[10](https://ieeexplore.ieee.org/document/10687925/)
[11](https://arxiv.org/abs/2508.18621)
[12](https://arxiv.org/html/2501.14646v1)
[13](https://arxiv.org/html/2411.18675)
[14](http://arxiv.org/pdf/2311.15230.pdf)
[15](https://arxiv.org/pdf/2404.10667.pdf)
[16](https://arxiv.org/html/2501.10687)
[17](http://arxiv.org/pdf/2405.15758.pdf)
[18](https://arxiv.org/pdf/2403.08764.pdf)
[19](https://arxiv.org/html/2412.07754v1)
[20](https://www.stableavatar.org)
[21](https://cumulo-autumn.github.io/StreamDiT/)
[22](https://gaiavatar.github.io/gaia/)
[23](https://arxiv.org/html/2506.18866v1)
[24](https://arxiv.org/abs/2507.03745)
[25](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_MetaPortrait_Identity-Preserving_Talking_Head_Generation_With_Fast_Personalized_Adaptation_CVPR_2023_paper.pdf)
[26](https://www.talkingavatar.ai)
[27](https://www.youtube.com/watch?v=gzmxHkxu2iQ)
[28](https://arxiv.org/abs/2504.13386)
[29](https://omniavatar.app)
[30](https://arxiv.org/html/2512.09327v1)
[31](https://arxiv.org/html/2508.18337v3)
[32](https://arxiv.org/html/2505.20156v2)
[33](https://arxiv.org/html/2507.03745v3)
[34](https://arxiv.org/html/2311.15230v1)
[35](https://arxiv.org/html/2510.01176v1)
[36](https://arxiv.org/abs/2510.12747)
[37](https://arxiv.org/abs/2508.18337)
[38](https://arxiv.org/abs/2405.20674)
[39](https://arxiv.org/abs/2210.02303)
[40](https://arxiv.org/abs/2505.22167)
[41](https://arxiv.org/abs/2406.06890)
[42](https://arxiv.org/abs/2410.05677)
[43](https://arxiv.org/abs/2403.09334)
[44](https://arxiv.org/abs/2405.16645)
[45](https://arxiv.org/abs/2412.15689)
[46](https://arxiv.org/abs/2403.12706)
[47](https://arxiv.org/abs/2507.22360)
[48](https://arxiv.org/html/2412.05899)
[49](https://arxiv.org/html/2503.19462v1)
[50](https://arxiv.org/pdf/2403.12706.pdf)
[51](https://arxiv.org/html/2406.04888)
[52](https://arxiv.org/html/2306.11173)
[53](https://arxiv.org/pdf/2312.00845.pdf)
[54](https://arxiv.org/pdf/2211.11743.pdf)
[55](https://hanyang-21.github.io/VideoScene/)
[56](https://www.deepspeed.ai/tutorials/pipeline/)
[57](https://arxiv.org/abs/2104.09864)
[58](https://papers.cool/arxiv/2412.05899)
[59](https://www.infracloud.io/blogs/inference-parallelism/)
[60](https://learnopencv.com/rope-position-embeddings/)
[61](https://www.emergentmind.com/topics/video-diffusion-model-self-distillation)
[62](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
[63](https://karthick.ai/blog/2024/Rotatory-Position-Embedding-(RoPE)/)
[64](https://openreview.net/pdf/8b5ddcb9f287f112054ed4fa390bc152d5b67e7e.pdf)
[65](https://arxiv.org/html/2405.14430v4)
[66](https://arxiv.org/html/2506.10470v1)
[67](https://arxiv.org/html/2512.07525v1)
[68](https://arxiv.org/html/2504.01956v1)
[69](https://arxiv.org/abs/2506.22033)
[70](https://arxiv.org/abs/2411.13476)
[71](https://arxiv.org/abs/2412.05899)
