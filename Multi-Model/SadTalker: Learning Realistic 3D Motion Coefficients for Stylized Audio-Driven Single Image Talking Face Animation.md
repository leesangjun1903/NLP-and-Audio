# SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation

## 종합 분석 보고서

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
SadTalker는 기존의 2D 모션 필드 기반 방법들이 **결합된(coupled) 2D 표현 공간**에서 학습하기 때문에 부자연스러운 머리 움직임, 왜곡된 표정, 정체성(identity) 변형 등의 문제가 발생한다고 주장한다. 이를 해결하기 위해 **3DMM(3D Morphable Model)의 모션 계수를 중간 표현(intermediate representation)으로 사용**하여, 오디오로부터 **분리된(decoupled) 3D 모션 계수(머리 자세, 표정)를 개별적으로 학습**하고, 이를 3D-aware face render를 통해 최종 비디오로 합성하는 시스템을 제안한다.

### 주요 기여
1. **SadTalker 시스템**: 오디오로부터 사실적인 3D 모션 계수를 생성하여 단일 이미지 기반 talking face 애니메이션을 수행하는 새로운 시스템
2. **ExpNet과 PoseVAE**: 표정(expression)과 머리 자세(head pose)를 각각 개별적으로 학습하는 네트워크 설계
3. **3D-aware Face Render**: 의미적으로 분리된(semantic-disentangled) 3D 인식 얼굴 렌더러 제안
4. **State-of-the-art 성능**: 모션 동기화와 비디오 품질 모두에서 최신 성능 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 talking head 생성 방법들의 핵심 문제점:

| 문제 유형 | 구체적 설명 | 관련 기존 방법 |
|----------|----------|-------------|
| 부자연스러운 머리 움직임 | 2D 모션 필드에서 머리 움직임과 표정이 충분히 분리되지 않음 | MakeItTalk, Audio2Head |
| 왜곡된 표정 | 결합된 2D 표현 공간에서의 학습으로 인한 얼굴 왜곡 | Audio2Head, Wang et al. |
| 정체성 변형 | 오디오-to-표정이 일대일 매핑이 아님에도 불구하고 이를 고려하지 않음 | Audio2Head, Wang et al. |
| 입 영역 블러 | 립 영역만 수정하는 방식의 품질 한계 | Wav2Lip |
| 뻣뻣한 표정 | 3D 정보를 명시적으로 사용하더라도 정확한 표정 생성 실패 | DECA 기반 방법들 |

핵심적으로, **오디오와 각 모션 유형(입술 움직임, 머리 자세, 눈 깜빡임) 사이의 관계가 서로 다름**에도 불구하고, 기존 방법들은 이를 하나의 네트워크에서 동시에 학습하려 했기 때문에 높은 불확실성(uncertainty)이 발생했다.

### 2.2 제안하는 방법

#### 2.2.1 3DMM 예비 지식 (Preliminary)

3DMM에서 3D 얼굴 형상 $\mathbf{S}$는 다음과 같이 분해된다:

$$\mathbf{S} = \bar{\mathbf{S}} + \alpha \mathbf{U}_{id} + \beta \mathbf{U}_{exp}$$

여기서:
- $\bar{\mathbf{S}}$: 3D 얼굴의 평균 형상
- $\mathbf{U}\_{id}$, $\mathbf{U}_{exp}$: LSFM morphable model의 정체성(identity)과 표정(expression)의 정규직교 기저(orthonormal basis)
- $\alpha \in \mathbb{R}^{80}$: 정체성 계수
- $\beta \in \mathbb{R}^{64}$: 표정 계수
- $\mathbf{r} \in SO(3)$: 머리 회전, $\mathbf{t} \in \mathbb{R}^3$: 이동

정체성과 무관한 모션 파라미터만 모델링:  $\{\beta, \mathbf{r}, \mathbf{t}\}$, 머리 자세 $\rho = [\mathbf{r}, \mathbf{t}]$와 표정 계수 $\beta$를 개별적으로 학습.

#### 2.2.2 ExpNet (표정 계수 생성)

**설계 동기**: 오디오-to-표정은 (1) 서로 다른 정체성에 대해 일대일 매핑이 아니며, (2) 표정 계수에는 오디오와 무관한 모션(예: 눈 깜빡임)이 포함되어 있어 예측 정확도에 영향을 미침.

**네트워크 구조**:

$$\beta_{\{1,...,t\}} = \Phi_M(\Phi_A(a_{\{1,...,t\}}), z_{blink}, \beta_0)$$

여기서:
- $\Phi_A$: ResNet 기반 오디오 인코더 (Wav2Lip에서 사전학습된 파라미터로 초기화)
- $\Phi_M$: 선형 매핑 네트워크
- $a_{\{1,...,t\}}$: 0.2초 mel-spectrogram 오디오 특징
- $z_{blink} \in [0, 1]$: 눈 깜빡임 제어 신호
- $\beta_0$: 참조 이미지의 초기 표정 계수 (정체성 불확실성 감소)

**손실 함수**:

**(1) Distillation Loss** — Wav2Lip으로 생성한 립온리(lip-only) 표정 계수와의 MSE:

$$\mathcal{L}_{distill} = \frac{1}{T} \sum_{t=1}^{T} \left( \beta_t^g - \beta_t^{lip} \right)^2$$

**(2) Eye Loss** — 눈 영역의 랜드마크를 이용한 깜빡임 제어:

$$E_t^w = \frac{\|P_t^{39} - P_t^{36}\|_2 + \|P_t^{45} - P_t^{42}\|_2}{2}$$

$$E_t^h = \frac{\|P_t^{37} + P_t^{38} - P_t^{40} - P_t^{41}\|_2}{2} + \frac{\|P_t^{43} + P_t^{44} - P_t^{46} - P_t^{47}\|_2}{2}$$

$$R_t = \frac{E_t^h}{E_t^w}$$

$$\mathcal{L}_{eye} = \sum_{t=1}^{T} \|R_t - Z_t^{blink}\|_1$$

**(3) Landmark Loss**:

$$\mathcal{L}_{lks} = \lambda_{eye}\mathcal{L}_{eye} + \frac{1}{T}\frac{1}{N}\sum_{t=1}^{T}\sum_{i=1}^{M}\|P_t^i - P_t^{i'}\|_2^2$$

**(4) Lip Reading Loss** — 사전학습된 lip-reading 네트워크를 이용한 시간적 립리딩 손실:

$$\mathcal{L}_{read} = \text{CrossEntropy}(\mathbf{C_{gt}}, \mathbf{C_p})$$

**ExpNet 전체 손실 함수**:

$$\mathcal{L}_{exp} = \lambda_{distill}\mathcal{L}_{distill} + \lambda_{read}\mathcal{L}_{read} + \lambda_{lks}\mathcal{L}_{lks}$$

($\lambda_{distill}=2$, $\lambda_{read}=0.01$, $\lambda_{lks}=0.01$)

#### 2.2.3 PoseVAE (머리 자세 생성)

**설계 동기**: 머리 자세는 오디오와의 관계가 상대적으로 약하며, 다양한 스타일의 자연스러운 머리 움직임 생성이 필요.

**구조**: Conditional VAE 기반, 인코더-디코더 모두 2-layer MLP.

**핵심 설계**:
- **잔차 학습(residual learning)**: 첫 프레임의 자세 $\rho_0$에 대한 잔차를 학습하여 긴 시퀀스에서도 안정적이고 연속적인 모션 생성:

$$\Delta\rho_{\{1,...,t\}} = \rho_{\{1,...,t\}} - \rho_0$$

- **조건**: 오디오 특징 $a_{\{1,...,t\}}$, 스타일 정체성 $Z_{style}$ (46차원 one-hot 벡터)

**손실 함수**:

**(1) 재구성 손실**:

$$\mathcal{L}_{MSE} = \frac{1}{T}\sum_{t=1}^{T}(\Delta\rho_t' - \Delta\rho_t)^2$$

**(2) KL Divergence**: 잠재 공간 분포와 가우시안 분포 사이의 KL 발산 $\mathcal{L}_{KL}$

**(3) 적대적 손실**: PatchGAN 기반 판별자를 사용한 1D convolution:

$$\mathcal{L}_{GAN} = \arg\min_G \max_D (G, D)$$

**PoseVAE 전체 손실**:

$$\mathcal{L}_{pose} = \lambda_{MSE}\mathcal{L}_{MSE} + \lambda_{KL}\mathcal{L}_{KL} + \lambda_{GAN}\mathcal{L}_{GAN}$$

($\lambda_{MSE}=1$, $\lambda_{KL}=1$, $\lambda_{GAN}=0.7$)

#### 2.2.4 3D-aware Face Render

Face-vid2vid에서 영감을 받아, 명시적 3DMM 모션 계수를 비지도 3D 키포인트 공간으로 매핑하는 **MappingNet**을 제안.

**학습 과정** (2단계):
1. Face-vid2vid를 자기지도(self-supervised) 방식으로 학습
2. Appearance encoder, canonical keypoints estimator, image generator를 고정(freeze)하고 MappingNet만 학습

**MappingNet 손실**: 비지도 키포인트 공간에서의 $L_1$ 정규화:

$$\mathcal{L}_1 = \frac{1}{N}\sum_{n=1}^{N}\|K_n' - K_n\|_1$$

($K_n'$: MappingNet이 생성한 키포인트, $K_n$: face-vid2vid의 원래 모션 생성기가 생성한 키포인트, 가중치 20)

**핵심 차별점**: PIRenderer에서 사용하는 face alignment 계수를 제거하여 더 자연스러운 비디오 생성 (alignment 계수는 오디오와 무관하며, 사용 시 부자연스러운 정렬된 머리 움직임 발생).

### 2.3 모델 구조 요약

```
입력: 단일 얼굴 이미지 + 오디오
    ↓
[3D 얼굴 재구성] → (β₀, ρ₀) 초기 계수 추출
    ↓
[ExpNet] 오디오 → 표정 계수 β{1,...,n}
[PoseVAE] 오디오 + 스타일 → 머리 자세 ρ{1,...,n}
    ↓
[MappingNet] 3DMM 계수 → 비지도 3D 키포인트
    ↓
[Face-vid2vid 기반 렌더러] → 최종 비디오
```

### 2.4 성능 향상

HDTF 데이터셋에서의 정량적 결과 (Table 1):

| 메트릭 | SadTalker | Audio2Head | MakeItTalk | Wang et al. |
|--------|-----------|------------|------------|-------------|
| FID↓ | **22.057** | 24.392 | 28.243 | 22.432 |
| CPBD↑ | **0.335** | 0.281 | 0.283 | 0.295 |
| CSIM↑ | **0.843** | 0.823 | 0.838 | 0.811 |
| Diversity↑ | **0.278** | 0.181 | 0.257 | 0.226 |
| Beat Align↑ | **0.293** | 0.267 | 0.268 | 0.268 |

사용자 연구에서도 전체 자연스러움(54.8%), 비디오 선명도(62.8%), 모션 다양성(57.9%)에서 압도적 우위.

### 2.5 한계

1. **치아와 눈 모델링 한계**: 3DMM이 눈과 치아의 변화를 모델링하지 않으므로, 일부 경우 비현실적인 치아 아티팩트 발생 (GFPGAN 등 blind face restoration으로 부분적 해결 가능)
2. **감정 표현 제한**: 입술 움직임과 눈 깜빡임만 고려하고, 감정(emotion)이나 시선(gaze direction) 등 다른 얼굴 표정은 생성하지 않아 고정된 감정의 비디오 생성
3. **해상도 제한**: 256×256 해상도로 학습 및 평가
4. **학습 데이터 의존성**: PoseVAE와 ExpNet 학습에 VoxCeleb에서 선별된 46명의 1890개 정렬된 비디오만 사용

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능 분석

SadTalker는 여러 측면에서 일반화 가능성을 보여준다:

**(1) Cross-identity 설정에서의 강건성**

논문의 부록 Table A6, A7에서 cross-identity 설정(구동 오디오가 다른 사람의 것)으로도 HDTF와 VoxCeleb2 모두에서 우수한 성능을 보임:
- HDTF cross-ID: FID **20.886**, CSIM **0.846**
- VoxCeleb2 cross-ID: FID **22.738**, CSIM **0.893**

이는 3DMM 기반 중간 표현이 정체성과 모션을 효과적으로 분리함을 시사.

**(2) 다양한 데이터셋 간 전이**

VoxCeleb에서 학습하고 HDTF에서 테스트하는 크로스-데이터셋 평가에서도 일관된 성능을 보여, 학습 데이터와 다른 분포의 데이터에 대한 일정 수준의 일반화 능력을 확인.

**(3) in-the-wild 오디오 지원**

기존의 video portrait 방법들과 달리 특정 비디오에 대한 추가 학습 없이 임의의 사진과 야생(in-the-wild) 오디오에 적용 가능.

### 3.2 일반화 성능 향상을 위한 핵심 전략

**(1) 3DMM 중간 표현의 활용**

3DMM의 분리된(decoupled) 표현 공간이 일반화의 핵심. 정체성($\alpha$)과 표정($\beta$), 자세($\rho$)가 명시적으로 분리되어 있어:
- 오디오 → 모션 매핑에서 정체성 무관(identity-irrelevant) 계수만 생성
- 첫 프레임의 $\beta_0$를 조건으로 사용하여 새로운 정체성에 대한 적응

**(2) 모션 분리 학습**

ExpNet과 PoseVAE를 개별 학습하여 각 모션 유형의 불확실성을 줄임. 이는 새로운 오디오/이미지 조합에 대한 일반화를 개선.

**(3) 지식 증류(Knowledge Distillation)**

Wav2Lip의 립싱크 능력을 3DMM 공간으로 증류하여, Wav2Lip의 강건한 립싱크 일반화 능력을 계승.

### 3.3 향후 일반화 성능 향상 방향

| 방향 | 구체적 방법 | 기대 효과 |
|------|----------|---------|
| 더 풍부한 3D 모델 | FLAME, DECA 등 치아·눈 모델 포함 | 세밀한 표정 일반화 |
| 대규모 학습 데이터 | 다양한 인종, 연령, 언어 데이터 확충 | 인구통계학적 일반화 |
| Self-supervised 사전학습 | 대규모 비디오로부터 얼굴 모션 표현 학습 | 도메인 적응력 향상 |
| 감정 모델링 추가 | 오디오의 감정 정보를 추가 조건으로 활용 | 표현력 있는 일반화 |
| 고해상도 확장 | 512×512 이상 해상도 지원 | 실용적 일반화 |
| 다국어 지원 강화 | 언어별 음소-시각 매핑 학습 | 언어적 일반화 |

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

**(1) 패러다임 전환**: 2D 모션 필드에서 **3D 중간 표현 기반**으로의 패러다임 전환을 촉진. 이후 많은 연구가 3DMM이나 FLAME 등의 3D 모델을 중간 표현으로 채택.

**(2) 모듈식 설계의 유효성 입증**: 모션 생성(ExpNet, PoseVAE)과 렌더링(Face Render)을 분리하는 모듈식 접근이 end-to-end 방법보다 효과적일 수 있음을 보여줌. 각 모듈을 독립적으로 개선 가능.

**(3) 실용적 응용 확대**: 단일 이미지와 오디오만으로 고품질 talking head를 생성할 수 있어 디지털 휴먼 생성, 화상 회의, 교육 콘텐츠, 접근성 도구 등 다양한 응용 가능성을 열어줌.

**(4) 다중 모달리티 확장성**: 예측된 3D 계수를 다른 모달리티에 직접 활용 가능 — 개인화된 2D visual dubbing, 2D 만화 애니메이션, 3D 얼굴 애니메이션, NeRF 기반 4D talking head 생성 등.

### 4.2 앞으로 연구 시 고려할 점

**(1) 윤리적 고려사항**
- 딥페이크 악용 방지를 위한 워터마킹 기술 필수
- 위조 탐지(forgery detection) 연구와의 연계 필요
- 생성된 콘텐츠의 식별 가능한 표시(visible/invisible watermark) 삽입

**(2) 3D 모델의 표현력 한계 극복**
- 3DMM은 치아, 눈, 혀 등의 미세한 변화를 모델링하지 못함
- FLAME + jaw 파라미터, 또는 implicit 3D 표현(NeRF 등)과의 결합 검토

**(3) 감정과 맥락 인식**
- 현재 시스템은 감정(emotion)을 생성하지 않아 고정된 감정의 비디오 생성
- 오디오의 프로소디(prosody), 감정 정보를 활용한 감정 인식 talking face 연구 필요

**(4) 실시간 처리**
- 화상 회의 등 실시간 응용을 위한 추론 속도 최적화 필요
- 모델 경량화 및 효율적 추론 파이프라인 설계

**(5) 고해상도 및 전신 생성**
- 256×256을 넘어 고해상도(512+) 생성
- 얼굴뿐만 아니라 상체, 제스처 등 전신 모션 생성으로의 확장

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근 | SadTalker와의 차이 | 장점 | 한계 |
|------|------|---------|-----------------|------|------|
| **Wav2Lip** (Prajwal et al.) | 2020 | Lip-sync expert discriminator로 입술 영역만 수정 | 입술만 동기화, 머리 움직임 없음 | 강건한 립싱크 | 블러 발생, 정적인 머리/표정 |
| **PC-AVS** (Zhou et al.) | 2021 | 암묵적 잠재 코드로 머리 자세와 표정 분리 | 외부 비디오에서 제어 신호 필요, 저해상도 | 자세-표정 분리 시도 | 낮은 해상도, 외부 신호 필요 |
| **Audio2Head** (Wang et al.) | 2021 | 오디오로부터 dense motion field 생성 | 2D 워핑 기반, 측면 얼굴 생성 어려움 | 자연스러운 머리 움직임 시도 | 정면 편향, 정체성 변형, 얼굴 왜곡 |
| **MakeItTalk** (Zhou et al.) | 2020 | 오디오에서 콘텐츠/화자 정보 분리, 랜드마크 기반 | 2D 랜드마크 기반, 왜곡 발생 | 콘텐츠-화자 분리 | 낮은 비디오 품질, 왜곡 |
| **PIRenderer** (Ren et al.) | 2021 | 3DMM 기반 semantic neural rendering | 정렬 계수 포함으로 부자연스러운 모션, 비디오 구동에 초점 | 제어 가능한 생성 | 오디오 구동에 직접 적용 어려움, 표정 정확도 낮음 |
| **Face-vid2vid** (Wang et al.) | 2021 | 비지도 3D 키포인트 기반 one-shot 합성 | 비디오 구동만 가능, 오디오 구동 불가 | 고품질 3D-aware 렌더링 | 구동 비디오 필요 |
| **AD-NeRF** (Guo et al.) | 2021 | NeRF 기반 오디오 구동 talking head | 개인별 학습 필요, 임의 사진 불가 | 고품질 3D 렌더링 | 일반화 불가, 학습 시간 높음 |
| **FaceFormer** (Fan et al.) | 2022 | Transformer 기반 음성-3D 얼굴 애니메이션 | 3D 메시 생성에 초점, 2D 렌더링 미포함 | Transformer의 시퀀스 모델링 | 3D 메시만 생성, 2D 비디오 미포함 |
| **EAMM** (Ji et al.) | 2022 | 감정 인식 오디오 기반 모션 모델 | 감정 모델링 포함 (SadTalker에 없음) | 감정 표현 가능 | 외부 감정 레이블 필요 |
| **StyleHEAT** (Yin et al.) | 2022 | 사전학습된 StyleGAN 기반 고해상도 talking face | GAN 잠재 공간 활용 | 고해상도 편집 가능 | 비디오 구동 중심 |
| **VideoReTalking** (Cheng et al.) | 2022 | 야생 talking head 비디오의 오디오 기반 립싱크 편집 | 기존 비디오 편집에 초점, 단일 이미지 생성 아님 | 야생 비디오 처리 강건성 | 비디오 필요 |
| **CodeTalker** (Xing et al.) | 2023 | 이산 모션 사전(discrete motion prior) 기반 3D 얼굴 애니메이션 | 3D 메시 애니메이션에 초점 | VQ-VAE 기반 이산 표현 | 3D 메시만 생성 |
| **DPE** (Pang et al.) | 2023 | 자세-표정 분리 기반 비디오 초상화 편집 | 비디오 구동 편집 중심 | 포즈-표정 효과적 분리 | 오디오 구동 미지원 |

### 핵심 비교 인사이트

**SadTalker의 차별적 위치**:
1. **단일 이미지 + 오디오만으로** 완전한 talking head 비디오를 생성하는 few-shot 방식에서 최초로 3DMM 계수를 분리 학습하는 체계적 접근
2. 기존 3DMM 기반 방법(PIRenderer)이 비디오 구동에 초점을 맞춘 반면, SadTalker는 오디오 구동에 최적화
3. 기존 오디오 구동 방법(Wav2Lip, PC-AVS)이 부분적 모션만 생성한 반면, 표정+머리 자세+눈 깜빡임을 종합적으로 생성

**후속 연구 동향** (2023년 이후):
- **Diffusion 기반 방법**: DiffTalk, DreamTalk 등 diffusion model을 활용한 talking face 생성이 활발히 연구되고 있으며, SadTalker의 3DMM 기반 분리 학습 아이디어를 diffusion 프레임워크에 통합하는 시도가 이어지고 있음
- **NeRF/3DGS 기반 방법**: ER-NeRF, GaussianTalker 등 neural radiance field나 3D Gaussian Splatting 기반의 더 사실적인 3D 렌더링 방법과의 결합 연구
- **대규모 사전학습 모델 활용**: 대규모 언어모델이나 비전-언어 모델의 표현력을 활용한 감정 인식 talking face 연구

---

## 참고자료

1. **논문 원문**: Zhang, W., Cun, X., Wang, X., Zhang, Y., Shen, X., Guo, Y., Shan, Y., & Wang, F. (2023). "SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation." arXiv:2211.12194v2 [cs.CV].
2. **프로젝트 페이지**: https://sadtalker.github.io
3. **Wav2Lip**: Prajwal, K.R. et al. "A Lip Sync Expert Is All You Need for Speech to Lip Generation in the Wild." ACM MM, 2020.
4. **PC-AVS**: Zhou, H. et al. "Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation." CVPR, 2021.
5. **PIRenderer**: Ren, Y. et al. "PIRenderer: Controllable Portrait Image Generation via Semantic Neural Rendering." ICCV, 2021.
6. **Face-vid2vid**: Wang, T.-C. et al. "One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing." CVPR, 2021.
7. **Audio2Head**: Wang, S. et al. "Audio2Head: Audio-Driven One-Shot Talking-Head Generation with Natural Head Motion." IJCAI, 2021.
8. **MakeItTalk**: Zhou, Y. et al. "MakeItTalk: Speaker-Aware Talking-Head Animation." ACM TOG, 2020.
9. **FaceFormer**: Fan, Y. et al. "FaceFormer: Speech-Driven 3D Facial Animation with Transformers." CVPR, 2022.
10. **CodeTalker**: Xing, J. et al. "CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior." arXiv:2301.02379, 2023.
11. **EAMM**: Ji, X. et al. "EAMM: One-Shot Emotional Talking Face via Audio-Based Emotion-Aware Motion Model." ACM SIGGRAPH, 2022.
12. **3DMM (Blanz & Vetter)**: Blanz, V. & Vetter, T. "A Morphable Model for the Synthesis of 3D Faces." ACM SIGGRAPH, 1999.
13. **Deep 3D Face Reconstruction**: Deng, Y. et al. "Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set." CVPR Workshops, 2019.
14. **GFPGAN**: Wang, X. et al. "Towards Real-World Blind Face Restoration with Generative Facial Prior." CVPR, 2021.
15. **AD-NeRF**: Guo, Y. et al. "AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis." ICCV, 2021.
