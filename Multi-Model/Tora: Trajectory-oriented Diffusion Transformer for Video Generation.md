
# Tora: Trajectory-oriented Diffusion Transformer for Video Generation

> **논문 정보:**
> - **제목:** Tora: Trajectory-oriented Diffusion Transformer for Video Generation
> - **저자:** Zhenghao Zhang, Junchao Liao, Menghao Li, Zuozhuo Dai, Bingxue Qiu, Siyu Zhu, Long Qin, Weizhi Wang
> - **게재:** CVPR 2025 (pp. 2063–2073)
> - **arXiv:** [2407.21705](https://arxiv.org/abs/2407.21705) (2024년 7월 31일 제출)
> - **소속:** Alibaba Group
> - **공식 코드:** [github.com/alibaba/Tora](https://github.com/alibaba/Tora)

---

## 1. 🔑 핵심 주장 및 주요 기여 요약

최근 Diffusion Transformer(DiT)는 고품질 비디오 콘텐츠 생성에서 뛰어난 성과를 보이고 있지만, 제어 가능한 모션을 가진 비디오를 효과적으로 생성하기 위한 트랜스포머 기반 확산 모델의 잠재력은 여전히 제한적으로 탐구된 영역이다.

이에 Tora는 다음의 핵심 주장과 기여를 제시합니다:

| 기여 | 내용 |
|---|---|
| **최초의 궤적 기반 DiT** | 텍스트·이미지·궤적 조건을 동시 통합 |
| **TE (Trajectory Extractor)** | 임의 궤적을 계층적 시공간 모션 패치로 인코딩 |
| **MGF (Motion-guidance Fuser)** | DiT 블록에 모션 조건을 주입 |
| **2단계 학습 전략** | Dense → Sparse 궤적 학습으로 일반화 성능 향상 |
| **고해상도 장시간 영상** | 720p, 최대 204프레임 생성 가능 |

이 논문은 텍스트, 시각적, 궤적 조건을 동시에 통합하는 최초의 궤적 지향 DiT 프레임워크인 Tora를 소개하며, 효과적인 모션 가이던스를 통한 확장 가능한 비디오 생성을 실현한다.

---

## 2. 🔍 상세 분석

### 2-1. 해결하고자 하는 문제

기존 비디오 확산 모델들은 주로 U-Net 아키텍처를 사용했으며, 약 2초의 짧은 영상 생성에 집중하고 고정된 해상도와 종횡비에 제한되었다.

구체적으로는 다음 세 가지 핵심 문제를 다룹니다:

1. **모션 제어의 부재:** 트랜스포머 기반 확산 모델이 제어 가능한 모션을 가진 비디오를 효과적으로 생성하는 잠재력은 제한적으로 탐구된 상태다.
2. **장시간 영상에서의 성능 저하:** U-Net 모델은 오류가 급격히 증가하는 반면, Tora는 오류가 점진적으로 증가하여 고해상도, 장시간 영상에서도 안정적이다.
3. **물리 세계 동역학 재현:** Tora는 지정된 궤적을 정밀하게 따를 뿐만 아니라 물리 세계에 부합하는 더 부드러운 움직임을 생성한다.

---

### 2-2. 제안 방법 및 수식

#### 🔷 기반 모델: Latent Diffusion Model (LDM)

Tora는 LDM의 표준 공식을 따르며, 기존 U-Net을 Transformer 기반 디노이징 함수로 대체합니다.

LDM의 노이즈 예측 목적함수:

$$\mathcal{L} = \mathbb{E}_{z_0, \tau, \epsilon \sim \mathcal{N}(0,I)} \left[ \| \epsilon - \epsilon_\theta(z_\tau, \tau, c) \|^2 \right]$$

여기서:
- $z_0$: 인코딩된 잠재 벡터 (비디오 패치)
- $\tau$: 디노이징 타임스텝
- $\epsilon$: 가우시안 노이즈
- $c$: 조건 (텍스트, 이미지, **궤적**)
- $\epsilon_\theta$: DiT 기반 디노이징 함수

기존 LDM 프레임워크와 전반적인 공식을 일치시키면서, 패러다임 전환의 핵심은 디노이징 함수를 위한 U-Net을 트랜스포머 아키텍처로 대체한 것이다.

---

#### 🔷 Trajectory Extractor (TE)

TE는 임의의 궤적을 계층적 시공간 모션 패치로 변환하며, 초기에는 궤적의 연속 프레임 간 위치 변위를 흐름 시각화 기법을 통해 RGB 도메인으로 변환한다.

**Step 1 — Flow Visualization:**

연속 프레임 $t$와 $t+1$ 사이의 위치 변위 $\delta_{t}$를 RGB 공간으로 시각화:

$$F_t = \text{FlowVis}(\delta_t), \quad \delta_t = p_{t+1} - p_t$$

여기서 $p_t$는 프레임 $t$에서의 궤적 상의 좌표.

**Step 2 — Gaussian Filtering:**

이 시각화된 변위는 분산 문제를 완화하기 위해 가우시안 필터링을 거친다.

$$\tilde{F}_t = G_\sigma * F_t$$

여기서 $G_\sigma$는 표준편차 $\sigma$의 가우시안 커널.

**Step 3 — 3D VAE 인코딩:**

이후 3D Variational Autoencoder(VAE)가 궤적 시각화를 시공간 모션 잠재 벡터로 인코딩하며, 이 잠재 벡터는 비디오 패치와 동일한 잠재 공간을 공유한다.

$$z_m = \text{Enc}_{3D\text{-VAE}}(\tilde{F}_{1:T}) \in \mathbb{R}^{T' \times H' \times W' \times C}$$

**Step 4 — Hierarchical Feature Extraction:**

Trajectory Extractor는 3D 모션 VAE를 사용하여 궤적 벡터를 비디오 패치와 동일한 잠재 공간에 임베딩하고, 연속 프레임 간 모션 정보를 효과적으로 보존하며, 이후 적층된 합성곱 레이어를 사용하여 계층적 모션 특징을 추출한다.

$$\{m^{(1)}, m^{(2)}, \ldots, m^{(L)}\} = \text{ConvStack}(z_m)$$

여기서 $m^{(l)}$은 $l$번째 레벨의 계층적 모션 특징.

---

#### 🔷 Motion-guidance Fuser (MGF)

MGF는 적응적 정규화 레이어(Adaptive Normalization)를 활용하여 이 다중 레벨 모션 조건을 해당 DiT 블록에 원활하게 주입하며, 정의된 궤적을 일관되게 따르는 비디오 생성을 보장한다.

Adaptive Normalization 수식:

$$\hat{h}^{(l)} = \gamma^{(l)}(m^{(l)}) \cdot \frac{h^{(l)} - \mu}{\sigma} + \beta^{(l)}(m^{(l)})$$

여기서:
- $h^{(l)}$: $l$번째 DiT 블록의 히든 스테이트
- $\gamma^{(l)}, \beta^{(l)}$: 모션 특징 $m^{(l)}$으로부터 예측되는 스케일/시프트 파라미터
- $\mu, \sigma$: 레이어 정규화의 평균/표준편차

---

#### 🔷 2단계 학습 전략

모델은 두 단계로 학습되는데, 먼저 Dense Optical Flow로 학습하고 이후 Sparse Trajectory로 파인튜닝한다.

어댑터 방식과 유사하게, Temporal Block, TE, MGF만 학습하며, 이 전략은 DiT의 내재적 생성 지식과 외부 모션 신호를 원활하게 통합한다.

**Stage 1 — Dense Flow 학습:**

$$\mathcal{L}_1 = \mathbb{E} \left[ \| \epsilon - \epsilon_\theta(z_\tau, \tau, c_{text}, c_{img}, m_{dense}) \|^2 \right]$$

**Stage 2 — Sparse Trajectory 파인튜닝:**

$$\mathcal{L}_2 = \mathbb{E} \left[ \| \epsilon - \epsilon_\theta(z_\tau, \tau, c_{text}, c_{img}, m_{sparse}) \|^2 \right]$$

---

### 2-3. 모델 전체 구조

```
입력: 텍스트 프롬프트 + 이미지 조건 + 사용자 정의 궤적

         ┌──────────────────────────────────────────┐
         │         Trajectory Extractor (TE)         │
         │  궤적 → Flow Vis → Gaussian Filter        │
         │       → 3D VAE Enc → Conv Stack           │
         │   계층적 모션 패치: {m^(1),...,m^(L)}    │
         └──────────────────┬───────────────────────┘
                           │
         ┌──────────────────▼───────────────────────┐
         │         Spatial-Temporal DiT              │
         │  (OpenSora / CogVideoX-5B 기반)           │
         │  ┌─────────────────────────────────────┐  │
         │  │ DiT Block 1                         │  │
         │  │  ← MGF: Adaptive Norm (m^(1))       │  │
         │  └─────────────────────────────────────┘  │
         │  ┌─────────────────────────────────────┐  │
         │  │ DiT Block 2                         │  │
         │  │  ← MGF: Adaptive Norm (m^(2))       │  │
         │  └─────────────────────────────────────┘  │
         │              ...                           │
         └──────────────────┬───────────────────────┘
                           │
         ┌──────────────────▼───────────────────────┐
         │          생성된 비디오                     │
         │  - 최대 204 프레임 @ 720p                 │
         │  - 다양한 종횡비·해상도                   │
         └──────────────────────────────────────────┘
```

Tora는 OpenSora를 DiT 아키텍처의 기반 모델로 채택한다. 공개 버전은 CogVideoX-5B 모델을 기반으로 한 CogVideoX 버전의 Tora다.

---

### 2-4. 데이터셋 구성

Tora 학습을 위해 캡션과 움직임 궤적이 포함된 주석 비디오가 필수적이며, OpenSora의 워크플로우를 활용하여 원시 비디오를 고품질 비디오-텍스트 쌍으로 변환하고 광학 흐름 추정기를 사용하여 궤적을 추출한다. 또한 모션 분할 결과와 흐름 점수를 결합하여 주로 카메라 움직임을 포함하는 인스턴스를 필터링하며, 이를 통해 일관된 모션을 가진 630k 고품질 비디오 클립 데이터셋이 구축된다.

평가 데이터셋은 비디오 객체 분할 데이터셋에서 선택된 다양한 모션 궤적과 장면을 포함한 185개의 클립으로 구성된다.

---

### 2-5. 성능 향상

2단계 학습을 통해 Tora는 다양한 길이, 종횡비, 해상도에 걸쳐 모션 제어 가능한 비디오 생성을 달성하며, 지정된 궤적을 따르는 최대 204프레임, 720p 해상도의 고품질 비디오를 생성할 수 있다.

#### 정량적 성능:
128프레임 영상에서 Tora는 U-Net 기반 모델 대비 3~5배 높은 궤적 정확도와 약 30~40% 더 나은 FVD(Fréchet Video Distance)를 달성한다.

Tora의 궤적 정확도는 128프레임 테스트 설정에서 다른 방법들보다 3~5배 높다.

#### 3D VAE의 우수성:
3D VAE는 서로 다른 궤적 압축 방법 중 가장 우수한 결과를 보인다. 키프레임 샘플링 방식은 빠른 모션이나 오클루전에서 흐름 추정 오류가 발생하고, 평균 풀링 방식은 궤적의 방향과 크기를 균질화하여 중요한 모션 세부 정보를 희석시키는 문제가 있다.

#### 비교 우위:
Tora는 UNet 기반 및 다른 DiT 기반 방식 대비 월등한 모션 제어 효능과 시각적 성능을 보이며, 특히 비디오 길이가 증가할수록 그 우위가 두드러진다.

---

### 2-6. 한계점

Tora가 유망한 결과를 보이지만, 향후 연구는 가속도와 변형과 같은 더 복잡한 모션 단서를 통합하여 모션 현실감을 높이는 것과, 보지 못한 객체 카테고리와 모션 패턴에 대한 모델의 일반화 능력을 조사하는 것이 유익할 것이다.

추가적인 한계점:
- 단순한 프롬프트는 시각적 품질과 모션 제어 효율성에 부정적인 영향을 미칠 수 있다.
- 상업적 계획으로 인해 완전한 버전의 Tora는 오픈소스로 공개되지 않는다.
- 카메라 움직임과 객체 움직임을 명시적으로 분리하는 메커니즘이 부족함

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

### 3-1. 2단계 학습이 일반화에 기여하는 방식

Dense Optical Flow를 먼저 사용하고 이후 사용자 친화적인 Sparse Trajectory로 파인튜닝하는 2단계 학습 전략을 통해 다양한 모션 패턴에 걸쳐 유연하고 정확한 모션 제어를 달성한다.

이 전략은 두 가지 측면에서 일반화를 향상시킵니다:
- **Dense Flow:** 다양한 물리적 움직임 패턴의 저수준 표현 학습
- **Sparse Trajectory:** 사용자가 실제로 제공하는 희소한 포인트로부터 움직임을 일반화하는 능력 학습

### 3-2. DiT 스케일러빌리티와 일반화

이 설계는 DiT의 확장성과 원활하게 일치하여 다양한 길이, 종횡비, 해상도에서 비디오 콘텐츠의 다이나믹스를 정밀하게 제어할 수 있다.

다양한 해상도와 길이에 걸친 궤적 오차 분석은 Tora의 견고성을 강조한다. UNet 모델은 오차가 급격히 증가하는 반면 Tora의 오차는 점진적으로 증가하며, 이는 Tora가 비디오 길이와 복잡성이 증가하더라도 궤적 제어를 효과적으로 유지함을 나타낸다.

### 3-3. 다중 조건 통합의 일반화 효과

Tora는 단일 시작 프레임과 초기/최종 프레임의 조합을 포함한 광범위한 시각적 조건을 수용하며, 여러 궤적을 전문적으로 처리하여 다수의 객체를 정밀하게 조작하고, 다양한 종횡비, 해상도, 길이에 걸쳐 비디오 생성을 지원함으로써 유연하고 적응적인 콘텐츠 생성을 보장한다.

### 3-4. 어댑터 방식의 일반화 전략

어댑터 방식으로, Temporal Block과 TE, MGF만 학습하며, 이 전략은 DiT의 내재적 생성 지식과 외부 모션 신호를 원활하게 통합한다.

기존 대규모 사전 학습 지식을 최대한 보존하면서 모션 제어 능력을 추가하는 방식으로, 새로운 도메인에 대한 일반화 잠재력이 높습니다.

### 3-5. 후속 연구 Tora2가 보여주는 일반화 방향

후속 연구인 Tora2는 ACM MM25에 채택되었으며, Tora의 설계 개선을 통해 다중 개체에 대한 향상된 외관 및 모션 커스터마이징을 가능하게 한다.

이를 통해 다양한 모션 제어를 가지면서 여러 개체의 충실도를 유지하도록 설계된 튜닝 없이 사용 가능한(Tuning-free) DiT 방법인 Tora2가 개발되었다.

---

## 4. 📊 관련 최신 연구 비교 분석 (2020년 이후)

| 모델 | 발표 | 기반 | 제어 방식 | 강점 | 한계 |
|------|------|------|----------|------|------|
| **VideoComposer** | 2023 | U-Net | 모션 벡터 | 다양한 조건 지원 | U-Net 아키텍처 한계 |
| **DragNUWA** | 2023 | U-Net | 텍스트+이미지+궤적 | 세밀한 제어 | 광학 흐름으로 추출된 궤적만 학습하여 전경과 배경의 움직임 구별에 한계 |
| **MotionCtrl** | SIGGRAPH 2024 | U-Net(LVDM) | 카메라+객체 분리 제어 | 카메라·객체 독립 제어 | U-Net 기반, DiT 비호환 |
| **DragAnything** | ECCV 2024 | U-Net | 엔티티 표현 | 객체 레벨 정밀 제어 | 단일 엔티티 중심 |
| **TrailBlazer** | 2024 | Diffusion | 바운딩 박스 | 직관적 입력 | 인스턴스 레벨 한계 |
| **Tora (제안)** | CVPR 2025 | **DiT** | 텍스트+이미지+궤적 | **DiT 확장성, 장시간 고해상도** | 카메라/객체 미분리 |
| **Tora2** | ACM MM 2025 | DiT(CogVideoX) | 다중 개체 제어 | 외관·모션 동시 커스터마이징 | — |

초기 접근법인 Stable Video Diffusion, VideoCrafter, Animate Anything 등은 주로 시간 모델링을 위해 3D 합성곱 레이어를 가진 U-Net 기반 아키텍처를 사용했으나, 이러한 프레임워크는 장시간 시퀀스에서 모션 일관성을 유지하는 데 어려움이 있었다.

DragNUWA는 궤적을 조건으로 카메라와 객체 모션을 모델링하고, DragAnything은 엔티티 레벨 제어를 위해 객체 마스크를 활용하며, Tora는 DiT 프레임워크에 궤적 조건을 도입한다.

Tora의 차별점은 기존 U-Net 기반 방법들과 달리 DiT 스케일러빌리티를 완전히 활용한다는 점입니다:

이 이점은 비디오 길이에 따라 U-Net 기반 접근법보다 더 효과적으로 확장되는 DiT 아키텍처에서 기인한다.

---

## 5. 🔮 앞으로의 연구에 미치는 영향과 고려사항

### 5-1. 연구에 미치는 영향

#### ① DiT 기반 제어 가능 비디오 생성의 표준 확립
이 연구가 모션 가이드 Diffusion Transformer 방법의 미래 연구를 위한 강력한 기준선(Baseline)을 제공하기를 기대한다.

#### ② 멀티모달 조건 통합 방향 제시
Tora는 텍스트, 이미지, 궤적 조건을 통합하는 최초의 궤적 지향 DiT 프레임워크로서 임의의 궤적을 시공간 모션 패치로 효과적으로 인코딩하며, DiT의 확장성과 일치함으로써 물리 세계 움직임의 더 현실적인 시뮬레이션을 가능하게 한다.

#### ③ 응용 분야 확장 가능성
Tora의 궤적 지향 모션 제어 접근법은 비디오 생성 외에도 애니메이션 및 로보틱스 분야에서 상당한 잠재력을 가지며, 캐릭터 애니메이션 분야에서는 사용자가 정의한 궤적으로부터 현실적이고 다양한 동작을 생성하는 능력이 워크플로우를 혁신적으로 바꿀 수 있다.

로보틱스 분야에서는 궤적 기반 접근법이 로봇 시스템의 모션 계획에 통합될 수 있으며, 원하는 궤적을 지정함으로써 장애물과 환경 제약을 고려한 부드럽고 효율적인 동작 시퀀스를 생성할 수 있다.

#### ④ 세계 모델(World Model) 시뮬레이션 방향
광범위한 실험을 통해 Tora는 높은 모션 충실도를 달성하는 동시에 물리적 세계의 움직임을 정밀하게 시뮬레이션하는 데 탁월함을 보인다.

---

### 5-2. 향후 연구 시 고려사항

#### 🔴 해결해야 할 한계
1. **카메라 모션과 객체 모션의 명시적 분리:**
   카메라 포즈와 객체 궤적을 모두 활용하여 생성 비디오의 카메라 및 객체 모션을 유연하게 제어하는 방향이 필요하다.

2. **보다 복잡한 물리 표현:**
   가속도와 변형과 같은 더 복잡한 모션 단서를 통합하여 모션 현실감을 더욱 향상시키는 연구가 필요하다.

3. **미학습 객체·모션 일반화:**
   보지 못한 객체 카테고리와 모션 패턴에 대한 모델의 일반화 능력을 조사하는 것이 유익할 것이다.

4. **텍스트 프롬프트 품질 의존성:**
   텍스트 프롬프트의 세부 향상을 위해 GPT-4 사용이 권장되며, 단순한 프롬프트는 시각적 품질과 모션 제어 효율성에 부정적인 영향을 줄 수 있다.

#### 🟡 연구 확장 방향

| 방향 | 세부 내용 |
|------|----------|
| **자동 궤적 추정** | 사용자 입력 없이 텍스트 → 궤적 자동 생성 |
| **실시간 제어** | 전체 제어 입력이 지정될 때까지 생성이 지연되는 문제를 해결하기 위한 자기회귀(AR) VDM 기반 실시간 제어 |
| **다중 개체 동시 제어** | Tora2가 보여주듯 복수 객체의 독립적 모션·외관 제어 |
| **Force/Physics-based Control** | 물리 법칙 기반의 힘(force) 프롬프트와의 통합 |
| **더 큰 기반 모델 적용** | HunyuanVideo, Wan 등 최신 대형 DiT 모델과의 결합 |

---

## 📚 참고 자료 및 출처

| # | 자료명 | URL/DOI |
|---|-------|---------|
| 1 | **Tora 논문 (arXiv)** | https://arxiv.org/abs/2407.21705 |
| 2 | **Tora 논문 (CVPR 2025 Open Access)** | https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Tora_Trajectory-oriented_Diffusion_Transformer_for_Video_Generation_CVPR_2025_paper.html |
| 3 | **Tora 프로젝트 페이지** | https://ali-videoai.github.io/tora_video/ |
| 4 | **Tora 공식 GitHub (Alibaba)** | https://github.com/alibaba/Tora |
| 5 | **Tora HuggingFace** | https://huggingface.co/papers/2407.21705 |
| 6 | **Tora HTML 논문 (v1)** | https://arxiv.org/html/2407.21705v1 |
| 7 | **Tora2 논문 (arXiv)** | https://arxiv.org/html/2507.05963 |
| 8 | **CVPR 2025 Poster** | https://cvpr.thecvf.com/virtual/2025/poster/34398 |
| 9 | **IEEE Xplore** | https://ieeexplore.ieee.org/document/11094661 |
| 10 | **Tora 분석 (linnk.ai)** | https://linnk.ai/insight/computer-vision/tora-enhancing-diffusion-transformers... |
| 11 | **MotionCtrl (SIGGRAPH 2024)** | https://dl.acm.org/doi/fullHtml/10.1145/3641519.3657518 |
| 12 | **DragAnything (ECCV 2024)** | https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03277.pdf |
| 13 | **Real-Time Motion-Controllable AR Video Diffusion** | https://arxiv.org/html/2510.08131 |
| 14 | **Video Diffusion Generation Survey (Springer 2025)** | https://link.springer.com/article/10.1007/s10462-025-11331-6 |
| 15 | **Motion Prompting** | https://motion-prompting.github.io/ |
| 16 | **Liner Quick Review (Tora)** | https://liner.com/review/tora-trajectoryoriented-diffusion-transformer-for-video-generation |
