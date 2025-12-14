# Motion-Conditioned Diffusion Model for Controllable Video Synthesis

### 1. 핵심 주장과 주요 기여

MCDiff(Motion-Conditioned Diffusion Model)는 **스트로크 기반의 직관적인 모션 제어를 통해 고품질의 동영상을 합성하는 조건부 확산 모델**입니다.[1]

**핵심 주장:**
- 기존의 텍스트 기반이나 조악한 수준의 클래스 기반 동영상 합성은 사용자의 세밀한 의도를 표현하기에 부족합니다.[1]
- 비디오를 **콘텐츠와 모션의 결합**으로 이해하고, 시작 프레임과 스트로크 입력을 통해 각각을 제어할 수 있습니다.[1]
- 단일 단계의 end-to-end 확산 모델은 희소 스트로크의 모호성과 의미론적 이해의 어려움으로 인해 실패합니다.[1]

**주요 기여:**
- **두 단계 프레임워크**: 희소-밀집 흐름 완성(flow completion)과 미래 프레임 예측을 분리하여 각각의 문제를 단순화합니다.[1]
- **자기지도학습 기반 흐름 완성**: 키포인트 추적과 일반 포인트 추적을 결합하여 의미론적 흐름 맵을 생성합니다.[1]
- **확산 기반 미래 프레임 예측**: 조건부 확산 모델을 통해 높은 품질의 프레임을 생성합니다.[1]
- **전역 최적화를 통한 성능 향상**: 두 네트워크의 end-to-end 미세 조정으로 도메인 갭을 감소시킵니다.[1]
- **배경/전경 모션 학습**: 전경 스트로크는 객체 움직임을, 배경 스트로크는 카메라 조정을 제어하는 것을 자동 학습합니다.[1]

***

### 2. 해결하고자 하는 문제

#### **문제 정의**
기존 동영상 합성 방법들의 한계점:[1]
- II2V와 iPOKE는 잠재 공간에서 희소한 제어 신호를 처리하여 저품질의 출력을 생성합니다.[1]
- 모션 전이 기반 방법들은 참고 비디오를 필요로 하여 세밀한 제어가 어렵습니다.[1]
- 텍스트 기반 제어는 복잡하고 다양한 모션을 정확히 표현할 수 없습니다.[1]

#### **핵심 도전 과제**
1. **희소 모션 입력의 모호성**: 단일 픽셀의 스트로크가 손가락 전체의 움직임을 의미해야 하며, 관련 신체 부위까지 일관성 있게 확산되어야 합니다.[1]
2. **의미론적 이해의 필요성**: 모델은 이미지의 의미를 이해하고 인물의 신체 구조 제약을 고려해야 합니다.[1]
3. **다양한 데이터셋에서의 확장성**: 단순한 데이터셋(TaiChi-HD, Human3.6M)을 넘어 다양한 콘텐츠와 활동을 포함하는 MPII Human Pose 데이터셋에서도 동작해야 합니다.[1]

***

### 3. 제안하는 방법 (수식 포함)

#### **3.1 방법 개요**

MCDiff는 자동회귀 구조로 매 시간 단계마다 다음을 수행합니다:[1]

$$\text{Dense Flow} = F(x_i, s_{i \to (i+1)})$$
$$x_{(i+1)} = G(x_i, d_{i \to (i+1)})$$

여기서:
- $F$: 흐름 완성 모델 (Flow Completion Model)
- $G$: 미래 프레임 예측 모델 (Future-Frame Prediction Model)
- $x_i$: 현재 프레임
- $s_{i \to (i+1)}$: 희소 모션 입력 (스트로크)
- $d_{i \to (i+1)}$: 예측된 밀집 흐름 맵

#### **3.2 동영상 동역학 주석 (Video Dynamics Annotation)**

동영상의 모션을 밀집 흐름으로 표현하기 위해:[1]

- 이미지 프레임에 추적 포인트 $P$를 분산시킵니다.
- 키포인트 추적(HRNet 사용)과 일반 포인트 추적(PIP 사용)을 결합합니다.
- 두 프레임 $(x_a, x_b)$ 사이의 밀집 흐름 맵 $\hat{d}_{a \to b}$를 생성합니다.

#### **3.3 흐름 완성 모델 (Flow Completion Model)**

**목표**: 희소 흐름 $s_{a \to b}$를 입력받아 밀집 흐름 $d_{a \to b}$를 예측합니다.[1]

**학습 목표함수**:

$$L_F = \frac{1}{\|P\|} \sum_{p \in P} w_p \cdot \|d_{a \to b}(p) - \hat{d}_{a \to b}(p)\|_2^2$$

여기서 픽셀별 가중치 인자는:

$$w_p = \lambda + \|\hat{d}_{a \to b}(p)\|_2 / \hat{d}_{\max}$$

[1]

**특징**:
- 자기지도학습으로 학습 (ground truth 흐름으로부터 희소 흐름 시뮬레이션)
- 키포인트 흐름(30%)과 일반 포인트(8개)의 가중 샘플링
- 흐름 크기를 기반으로 한 샘플링 확률로 중요한 모션 보존
- 여전 카메라 움직임에서 0 값이 지배하는 분포 문제를 해결하기 위해 가중치 적용[1]

#### **3.4 미래 프레임 예측 모델 (Future-Frame Prediction)**

**목표**: 현재 프레임과 밀집 흐름으로부터 다음 프레임을 합성합니다.[1]

잠재 확산 모델(Latent Diffusion Model) 기반으로 구현:

1. 현재 프레임 $x_i$를 VQ-4 자동인코더로 잠재 공간에 인코딩
2. 노이즈 샘플 $x_T^{(i+1)}$에서 시작하여 점진적 노이즈 제거
3. 각 단계에서 U-Net은 조건으로서 $x_i$와 $d_{i \to (i+1)}$을 concatenate하여 입력받음[1]

**학습 목표함수**:

$$L_G = \mathbb{E}_{x_b, \epsilon \sim \mathcal{N}(0,1), t} \left\|\epsilon - \epsilon_\theta(x_t^b, t, x_a, \hat{d}_{a \to b})\right\|_2^2$$

[1]

여기서:
- $\epsilon_\theta$: U-Net 연산
- $t$: $\{1, ..., T\}$에서 균등 샘플된 타임스텝
- $x_t^b$: 노이징된 미래 프레임

#### **3.5 End-to-End 미세조정 (Fine-Tuning)**

두 모델을 개별적으로 수렴할 때까지 학습한 후, 상호작용을 최적화하기 위해:[1]

$$L = \lambda_F \cdot L_F + \lambda_G \cdot L_G$$[1]

이 과정에서:
- 흐름 완성 모델 $F$의 출력이 미래 프레임 예측 모델 $G$의 입력으로 사용됨
- 전체 파이프라인 $G(x_a, F(x_a, s_{a \to b}))$이 end-to-end 미분 가능
- 하이퍼파라미터: $\lambda_F = 0.05$, $\lambda_G = 1$[1]

***

### 4. 모델 구조

#### **4.1 전체 아키텍처**

MCDiff의 구조는 두 주요 구성 요소로 이루어져 있습니다:[1]

| 구성 요소 | 역할 | 입력 | 출력 |
|---------|------|------|------|
| **흐름 완성 모델 (F)** | 희소 흐름을 밀집 흐름으로 확장 | 현재 프레임 + 희소 스트로크 | 밀집 흐름 맵 |
| **미래 프레임 예측 (G)** | 조건부 확산을 통한 프레임 합성 | 현재 프레임 + 밀집 흐름 | 다음 프레임 |

#### **4.2 흐름 완성 모델의 상세 구조**

- **아키텍처**: UNet (LDM-4와 동일)[1]
- **입력 처리**: 
  - 희소 흐름을 2D 맵으로 재구성
  - 결측값을 학습 가능한 임베딩으로 채우기
  - 현재 프레임과 concatenate[1]
- **특징**: 자기지도 학습을 통해 큰 모션 계산 선호도 해결

#### **4.3 미래 프레임 예측 모델의 상세 구조**

- **아키텍처**: UNet (LDM-4)[1]
- **인코더**: 256×256 프레임을 64×64 잠재 공간으로 인코딩 (VQ-4 자동인코더)
- **디코더**: 잠재 공간으로부터 256×256 출력 비디오로 디코딩
- **조건 입력**: 현재 프레임과 밀집 흐름을 노이징된 변수와 concatenate[1]

#### **4.4 학습 설정**

**모델 구성**:
- 양쪽 U-Net 모두 LDM-4 기반
- 공간 해상도: 256×256
- 잠재 공간 크기: 64×64
- 확산 스텝: T = 1000[1]

**학습 절차**:
1. **단계 1**: $F$와 $G$를 분별적으로 학습
   - 반복: 400k
   - 배치 크기: 40
   - $F$ 학습률: 4.5e-6
   - $G$ 학습률: 7e-5[1]

2. **단계 2**: End-to-end 미세조정
   - 추가 반복: 100k
   - 배치 크기: 20
   - 학습률: 7e-5[1]

**프레임 쌍 샘플링**:
- 간격: $(b-a) \in$ 프레임[2][3]
- 키포인트 샘플: 30% (흐름 크기 기반 확률)
- 일반 포인트: 8개(MPII), 4개(다른 데이터셋)[1]

***

### 5. 성능 향상

#### **5.1 정량적 평가 (Quantitative Results)**

MCDiff는 **TaiChi-HD와 Human3.6M** 데이터셋에서 이전 방법들을 크게 능가합니다:[1]

| 메트릭 | II2V (5 strokes) | iPOKE (5 strokes) | MCDiff (5 strokes) |
|-------|------------------|-------------------|-------------------|
| **FVD** ↓ | 175.48 | 113.12 | **110.80** |
| **LPIPS** ↓ | 0.11 | 0.11 | **0.09** |
| **SSIM** ↑ | 0.76 | 0.75 | **0.77** |
| **PSNR** ↑ | 20.31 | 20.80 | **21.45** |

**Human3.6M 데이터셋:**

| 메트릭 | II2V (5 strokes) | iPOKE (5 strokes) | MCDiff (5 strokes) |
|-------|------------------|-------------------|-------------------|
| **FVD** ↓ | 137.88 | 111.38 | **106.32** |
| **LPIPS** ↓ | 0.09 | 0.05 | **0.05** |
| **SSIM** ↑ | 0.91 | 0.93 | **0.94** |
| **PSNR** ↑ | 23.73 | 23.51 | **24.03** |[1]

#### **5.2 모션 제어 정확도 (Motion Controllability)**

스트로크 개수에 따른 성능 비교 (Average Displacement Error - ADE):[1]

**TaiChi-HD (더 낮은 값이 더 좋음)**:

| 방법 | 1 stroke | 5 strokes | 9 strokes |
|-----|----------|-----------|-----------|
| II2V | 4.17 | 7.51 | 11.36 |
| iPOKE | 2.63 | 5.09 | 8.94 |
| **MCDiff** | **2.77** | **2.72** | **2.90** |

**특징**: 
- MCDiff는 스트로크 개수 증가에도 성능이 거의 유지됨
- 이전 방법들은 스트로크 증가에 따라 선형으로 성능 저하[1]

#### **5.3 두 단계 설계의 필요성 (Ablation Study)**

MPII Human Pose 데이터셋에서의 절제 연구:[1]

| 모델 | FVD ↓ | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|------|-------|---------|--------|--------|
| 흐름 완성 없음 (w/o F) | 273.86 | 0.18 | 0.63 | 18.34 |
| **전체 모델 (Full)** | **194.30** | **0.14** | **0.69** | **19.52** |

**성능 향상**:
- FVD: 29% 개선
- LPIPS: 22% 개선
- SSIM: 9% 개선

이는 흐름 완성 모듈이 희소 모션의 모호성을 효과적으로 해결함을 증명합니다.[1]

***

### 6. 모델의 일반화 성능 향상 가능성

#### **6.1 일반화 능력의 현재 상태**

MCDiff의 MPII Human Pose 데이터셋 실험 결과:[1]

**강점**:
- **다양한 활동**: 410개의 서로 다른 인간 활동을 처리 가능
- **다양한 촬영 조건**: 고정 카메라부터 동적 카메라 조정까지 포함
- **복합 장면**: 다중 인물, 물체 상호작용, 배경 변화 처리[1]
- **카메라-객체 구분**: 전경 스트로크와 배경 스트로크를 자동으로 구분[1]

#### **6.2 일반화 성능 향상 가능성**

**1. 아키텍처 레벨의 개선**
- 흐름 완성 모델의 의미론적 이해 강화
- 더 정교한 세맨틱 분할을 통한 계층별 흐름 예측
- 불확실성 모델링을 통한 견고성 증대

**2. 학습 데이터 확대**
- 텍스처리스(textureless) 표면에 대한 특수 처리
- 시뮬레이션 데이터를 활용한 물리 기반 흐름 학습
- 대규모 동영상 데이터셋(예: YouTube, TikTok) 활용[1]

**3. 제어 신호의 확장**
- 희소 스트로크 외에 밀집 마스크, 깊이 맵, 포즈 정보 결합
- 다중 모드 조건부 생성 지원
- 사용자 의도를 더 정확히 포착하는 인터페이스 개발

**4. 모델 용량 확대**
- 더 큰 U-Net 아키텍처 (예: LDM-8로 업그레이드)
- Transformer 기반 확산 모델(DiT) 도입
- 멀티스케일 동영상 학습

#### **6.3 한계와 제약사항**

**현재 한계**:[1]

1. **분포 외 편집(Out-of-distribution edits)**: 
   - 스코어 버그를 화면의 다른 위치로 이동
   - 선반에서 물건이 떨어지는 큰 구성 변화
   - 데이터-주도 학습의 근본적 제약[1]

2. **광학 흐름 추정의 제약**:
   - 텍스처리스 표면에서의 실패
   - 바버폴 착시(barber-pole illusion) 같은 시각적 착시 처리 불가
   - 패턴 인식 기반 기법의 한계[1]

3. **물리적 이해 부족**:
   - 실제 모션 필드에 대한 물리 센서 정보 부재
   - 인터넷 비디오의 패턴만으로 학습[1]

#### **6.4 향후 일반화 개선 방향**

논문에서 제시된 미래 방향:[1]

- **생성 모델 개선**: 새로운 객체 구성을 처리할 수 있는 일반화 생성 모델 개발
- **물리 기반 학습**: 특수 센서(예: 광학 카메라)나 시뮬레이션을 통한 물리 기반 모션 필드 구축
- **세간지 강화**: 의미론적 이해를 위한 다중 모드 기반 모델 개발

***

### 7. 2020년 이후 관련 최신 연구 비교 분석

#### **7.1 동영상 확산 모델 진화**

| 시기 | 주요 연구 | 특징 | 제한사항 |
|------|---------|------|---------|
| **2020** | VDM (Ho et al.) | 3D U-Net 기반 비디오 확산 모델 | 제어 메커니즘 부족 |
| **2022** | Imagen Video (Ho et al.) | 7개 확산 모델의 캐스케이드 구조 | 높은 계산 비용 |
| **2023** | MCDiff (Chen et al.) | **스트로크 기반 모션 제어** | 희소 입력 모호성 처리 |
| **2023** | VideoFusion | 분해된 확산 모델 | 텍스트 제어만 가능 |
| **2024** | Lumiere (Google) | 공간-시간 U-Net 전체 기간 생성 | 매개변수 수 증가 |
| **2024** | FlowVid | 광학 흐름 기반 비디오-투-비디오 합성 | 흐름 추정 오류 처리 |
| **2024-2025** | VideoComposer, Wan-Move | 다중 조건 제어, 점 궤적 기반 제어 | 확장성 및 효율성 |[4][5][6]

#### **7.2 모션 제어 방법론 비교**

**스트로크/궤적 기반 방법**:

| 방법 | 제어 신호 | 아키텍처 | 품질 | 제어 정확도 |
|-----|---------|---------|------|-----------|
| **II2V** | 희소 스트로크 | RNN 기반 잠재 공간 | 낮음 | 낮음 |
| **iPOKE** | 희소 스트로크 | RNN 기반 잠재 공간 | 낮음 | 낮음 |
| **MCDiff** | 희소 스트로크 | 2단계 흐름-확산 모델 | **높음** | **높음** |
| **FlowVid** | 광학 흐름 | 확산 + 흐름 워핑 | 매우 높음 | 중간 |
| **Wan-Move** | 점 궤적 | 잠재 궤적 가이던스 | **매우 높음** | **매우 높음** |
| **DragEntity** | 다중 궤적 | 트랜스포머 기반 | 높음 | 높음 |[6][7][8][9]

#### **7.3 광학 흐름 기반 접근법**

MCDiff의 두 단계 설계 철학이 다른 연구에 영향을 미침:[10][11]

**FlowVid (2024)**:
- 불완전한 광학 흐름을 소프트 조건으로 처리
- 워핑된 프레임을 참고로 사용
- MCDiff와 유사하게 흐름 정보와 공간 조건을 결합[12]

**MotionPrompt (2025)**:
- 광학 흐름으로 비디오 생성 과정 가이드
- 변별자를 통해 흐름 일관성 학습[13]

**OnlyFlow (2024)**:
- 입력 비디오에서 추출한 광학 흐름으로 모션 제어[14]

#### **7.4 시간 일관성 및 제어 메커니즘의 발전**

**2023년 중반 이전**: 주로 텍스트 기반 제어[4]
- Imagen Video, Make-a-Video

**2023년 후반 ~ 2024년**: 세밀한 공간 제어 도입[15][7]
- VideoComposer: 다중 조건 (텍스트, 모션 벡터, 깊이 맵, 포즈)
- ControlNet 기반 적응

**2024 ~ 2025년**: 정밀한 궤적 제어로 진화[6][8][16][17]
- Wan-Move: 점 궤적을 통한 상업 수준의 제어
- FlexTraj: 유연한 점 궤적 제어
- C³: 불확실성 정량화를 통한 보정된 제어[17]

#### **7.5 MCDiff의 위치와 영향**

**MCDiff (2023년 4월)**의 위치:
- **선구적 역할**: 확산 모델에 스트로크 기반 모션 제어를 처음 성공적으로 도입
- **방법론 기여**: 희소-밀집 흐름 변환 패러다임 확립
- **영향력**: 이후 광학 흐름 기반 방법들이 유사한 두 단계 설계 채택

**한계 극복의 역사**:
1. MCDiff: 희소 스트로크 → 밀집 흐름 (명시적 완성)
2. FlowVid: 불완전한 흐름 처리 (소프트 조건화)
3. Wan-Move: 궤적을 직접 잠재 공간에 매핑 (매개변수 효율성)

***

### 8. 앞으로의 연구에 미치는 영향과 고려사항

#### **8.1 학술 커뮤니티에 미치는 영향**

**1. 방법론적 영향**:
- **두 단계 분해 패러다임**: 복잡한 조건부 생성 문제를 단순화하는 아키텍처 원칙 확립
  - 이후 연구들이 유사한 분해 전략 도입[7][11][12]
  - 비디오 합성뿐만 아니라 이미지 편집, 3D 생성 등으로 확대 적용
  
- **희소-밀집 변환의 중요성**: 
  - 사용자 입력의 모호성을 해결하기 위한 중간 표현의 가치 증명
  - 이후 FlexTraj, Wan-Move 등에서 더 정교한 변환 메커니즘 개발[16][6]

**2. 확산 모델 조건화 메커니즘**:
- 광학 흐름을 명시적 조건으로 사용하는 방식의 유효성 입증
- 이후 FlowVid, MotionPrompt 등에서 흐름 조건화 채택
- Diffusion Transformer(DiT) 기반 모델에서의 조건화 설계에 영향[18]

#### **8.2 실무 응용의 방향**

**1. 비디오 편집 및 제작**:
- VFX 스튜디오: 정밀한 모션 제어를 필요로 하는 애니메이션 제작
- 콘텐츠 크리에이터: 직관적인 스트로크 기반 인터페이스로 재표현 및 편집
- 실시간 인터랙션: 게임 엔진과의 통합을 통한 실시간 캐릭터 조작[1]

**2. 상업적 시스템으로의 진화**:
- MCDiff 이후 점 궤적 기반 방법(Wan-Move, Kling 1.5 Pro Motion Brush)이 상업화
- 더 높은 해상도(480p → 720p 이상)와 더 긴 길이(6초 → 30초 이상) 지원[3]

#### **8.3 향후 연구 시 고려할 기술적 과제**

**1. 일반화 성능 향상**:

**문제점**:
- 데이터셋 분포 내에서만 효과적 작동
- 새로운 객체 구성이나 장면에서 실패[1]

**해결 방향**:
- **메타 러닝**: 신규 도메인에 빠르게 적응하는 능력 개발
- **다중 모드 학습**: 텍스트, 포즈, 깊이, 에지 등 다양한 조건 결합
- **도메인 적응**: 무감독 또는 약한 감독 학습을 통한 도메인 갭 감소[19][15]

**2. 광학 흐름 추정의 개선**:

**현재 문제**:[1]
- 텍스처리스 표면에서의 실패
- 시각적 착시(barber-pole illusion) 처리 불능
- 패턴 인식 기반 방법의 한계

**개선 방안**:
- **물리 기반 흐름 학습**: 센서 데이터(이벤트 카메라, 열화상) 또는 시뮬레이션 활용[1]
- **신경 암묵 흐름 표현**: 연속적이고 가능한 부분에서 미분 가능한 흐름 표현
- **불확실성 추정**: 흐름 추정의 신뢰도를 모델링하여 오류에 강건하게 대응[20]

**3. 계산 효율성**:

**현재 제약**:
- 이미지당 400ms 이상 소요 (8 A100 GPU 필요)
- 매우 긴 비디오(수십 초)는 자동회귀 생성으로 오류 누적 가능

**개선 전략**:
- **축약된 확산(Distilled Diffusion)**: 샘플링 스텝 수 감소로 추론 가속[21]
- **계층적 생성**: TempoMaster처럼 저프레임률 → 고프레임률 점진적 생성[22]
- **조건부 인젠션**: ControlNet 기반의 경량 어댑터 구조[23]

#### **8.4 특정 응용 분야별 고려사항**

**1. 3D 아바타 및 캐릭터 애니메이션**:
- 신체 구조 제약을 더 명시적으로 모델링
- 뼈대(skeleton) 기반 제어 신호 통합
- 관절각 추정으로 현실감 향상[1]

**2. 로봇 정책 학습**:
- 현재 MCDiff의 할루시네이션 문제는 로봇 정책 학습에 위험
- 불확실성 정량화(Confidence Calibration) 필수[17]
- 물리 제약 조건 강제

**3. 장면 이해 및 월드 모델**:
- 뉴럴 라디언스 필드(NeRF)와의 통합
- 깊이 일관성 보장
- 조명 조건 보존[7]

#### **8.5 윤리 및 사회적 고려사항**

**1. 합성 미디어의 신뢰성**:
- 생성된 비디오의 투명성 표시 필요
- 거짓 정보 확산 방지 메커니즘 개발

**2. 데이터 개인정보보호**:
- 인간 활동 데이터셋(MPII, Human3.6M) 사용 시 개인 정보 보호
- 동의 기반 데이터 수집 및 활용

**3. 창작 능력 민주화**:
- 비용 효율적인 배포를 통한 접근성 개선
- 오픈 소스 모델 공개로 커뮤니티 기여 장려

***

### 9. 결론

**MCDiff의 핵심 성취**:
- ✅ 희소 스트로크 입력의 모호성을 흐름 완성으로 해결
- ✅ 확산 모델의 강력한 생성 능력과 모션 제어의 결합
- ✅ 배경/전경 모션의 자동 구분으로 유연한 제어 실현
- ✅ 다양한 콘텐츠와 활동에 대한 일반화 능력 시연

**향후 발전 방향**:
- 더 정교한 의미론적 이해를 통한 분포 외 편집 지원
- 물리 기반 학습으로 광학 흐름의 한계 극복
- 계산 효율성 개선으로 실시간 응용 실현
- 불확실성 정량화로 안전성 강화

MCDiff는 **동영상 합성의 제어 가능성 문제에서 획기적인 진전**을 이루었으며, 이후 관련 연구들이 다양한 방식으로 이를 확장하고 개선하고 있습니다. 특히 희소-밀집 표현의 변환, 두 단계 분해 설계, 그리고 광학 흐름 기반의 모션 제어라는 세 가지 핵심 아이디어는 2024-2025년의 최신 연구에서도 계속 활용되고 있습니다.[8][15][7][19][6][12][17]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5f7426a8-ae17-4cf7-a9f6-7547e63b3396/2304.14404v1.pdf)
[2](https://arxiv.org/abs/2507.20478)
[3](https://badanpenerbit.org/index.php/SEMNASPA/article/view/2335)
[4](https://dl.acm.org/doi/10.1145/3707292.3707367)
[5](https://arxiv.org/abs/2504.16081)
[6](https://arxiv.org/html/2512.08765v1)
[7](https://proceedings.neurips.cc/paper_files/paper/2023/file/180f6184a3458fa19c28c5483bc61877-Paper-Conference.pdf)
[8](https://openaccess.thecvf.com/content/CVPR2025/papers/Geng_Motion_Prompting_Controlling_Video_Generation_with_Motion_Trajectories_CVPR_2025_paper.pdf)
[9](https://arxiv.org/html/2410.10751)
[10](https://arxiv.org/pdf/2312.17681.pdf)
[11](https://arxiv.org/html/2411.15540v1)
[12](https://www.frontiersin.org/articles/10.3389/frai.2025.1649155/full)
[13](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[14](https://arxiv.org/abs/2411.10501)
[15](https://arxiv.org/abs/2507.16869)
[16](https://arxiv.org/html/2510.08527v1)
[17](https://arxiv.org/html/2512.05927)
[18](https://arxiv.org/html/2509.02460v1)
[19](https://arxiv.org/html/2507.16869v1)
[20](http://arxiv.org/pdf/2412.20404.pdf)
[21](https://arxiv.org/html/2503.19462v1)
[22](https://arxiv.org/html/2511.12578v3)
[23](https://openaccess.thecvf.com/content/WACV2025/papers/Huang_Fine-Grained_Controllable_Video_Generation_via_Object_Appearance_and_Context_WACV_2025_paper.pdf)
[24](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01824.pdf)
[25](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[26](https://dl.acm.org/doi/10.1145/3587423.3595503)
[27](https://arxiv.org/abs/2502.17119)
[28](https://arxiv.org/abs/2505.06814)
[29](https://arxiv.org/pdf/2308.03463.pdf)
[30](https://arxiv.org/pdf/2211.13221.pdf)
[31](https://arxiv.org/html/2401.12945v1?s=09)
[32](http://arxiv.org/pdf/2303.08320v3.pdf)
[33](http://arxiv.org/pdf/2405.15364.pdf)
[34](http://arxiv.org/pdf/2405.03150.pdf)
[35](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
[36](https://www.emergentmind.com/topics/latent-video-diffusion-model)
[37](https://www.sciencedirect.com/science/article/abs/pii/S0925231225023434)
[38](https://github.com/showlab/Awesome-Video-Diffusion)
[39](https://arxiv.org/html/2412.18688v2)
[40](https://arxiv.org/abs/2304.14404)
[41](https://arxiv.org/html/2511.21129v1)
[42](https://arxiv.org/html/2509.24353v1)
[43](https://pmc.ncbi.nlm.nih.gov/articles/PMC11086136/)
[44](https://www.arxiv.org/abs/2510.07670)
[45](https://ieeexplore.ieee.org/document/10744093/)
[46](https://ieeexplore.ieee.org/document/10760910/)
[47](https://ieeexplore.ieee.org/document/10333553/)
[48](https://ieeexplore.ieee.org/document/9481904/)
[49](https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0010869400003122)
[50](https://www.semanticscholar.org/paper/6bb013addc9d5d7bd9e5e9f32146a293a7a61cb8)
[51](https://dl.acm.org/doi/10.1145/3746278.3759384)
[52](https://arxiv.org/abs/2212.03250)
[53](http://link.springer.com/10.1007/978-3-030-20887-5_32)
[54](https://dl.acm.org/doi/10.1145/3651671.3651681)
[55](https://arxiv.org/abs/2411.15540)
[56](https://arxiv.org/pdf/1702.02463.pdf)
[57](https://arxiv.org/abs/2207.11075)
[58](https://arxiv.org/pdf/2308.01568.pdf)
[59](http://arxiv.org/pdf/1601.07532.pdf)
[60](https://arxiv.org/pdf/2203.10462.pdf)
[61](http://arxiv.org/pdf/2103.05101.pdf)
[62](https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_FlowVid_Taming_Imperfect_Optical_Flows_for_Consistent_Video-to-Video_Synthesis_CVPR_2024_paper.pdf)
[63](https://openaccess.thecvf.com/content_cvpr_2015/papers/Wulff_Efficient_Sparse-to-Dense_Optical_2015_CVPR_paper.pdf)
[64](https://iccvm.org/2024/papers/lncs/93.pdf)
[65](https://learnopencv.com/optical-flow-using-deep-learning-raft/)
[66](https://cerv.aut.ac.nz/wp-content/uploads/2015/08/MItech-TR-11.pdf)
[67](https://arxiv.org/html/2502.17863v1)
[68](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)
[69](https://stackoverflow.com/questions/11037136/what-is-the-difference-between-sparse-and-dense-optical-flow/11048092)
[70](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/sequence_processing/optical_flow_example.html)
[71](https://arxiv.org/html/2507.15496v1)
[72](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Learning_Video_Stabilization_Using_Optical_Flow_CVPR_2020_paper.pdf)
[73](https://arxiv.org/html/2510.12777v1)
[74](https://arxiv.org/html/2512.07469v1)
[75](https://arxiv.org/abs/2510.12777)
[76](https://arxiv.org/html/2510.19193v2)
[77](https://files.is.tue.mpg.de/black/papers/0738_ext.pdf)
[78](https://viso.ai/deep-learning/optical-flow/)
[79](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Upscale-A-Video_Temporal-Consistent_Diffusion_Model_for_Real-World_Video_Super-Resolution_CVPR_2024_paper.pdf)
