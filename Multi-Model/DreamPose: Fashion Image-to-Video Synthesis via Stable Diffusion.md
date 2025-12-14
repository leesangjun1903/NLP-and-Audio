# DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion

### 1. 핵심 주장과 주요 기여

**DreamPose**는 **프리트레인된 텍스트-이미지 확산 모델(Stable Diffusion)**을 **포즈-이미지 조건화 비디오 합성 모델**로 효율적으로 변환하는 혁신적 접근법을 제시한다.[1]

주요 기여는 다음과 같다:

1. **포즈-이미지 조건화 확산 기반 방법**: 단일 패션 이미지와 포즈 시퀀스로부터 **포토리얼리스틱한 패션 비디오**를 생성[1]

2. **효과적인 포즈 조건화 기법**: 5개의 연속 포즈를 입력 노이즈에 연결하여 **시간적 일관성을 대폭 향상**[1]

3. **분할 CLIP-VAE 인코더**: CLIP 이미지 임베딩과 VAE 잠재 임베딩을 결합하는 어댑터 모듈로 **세부 사항 충실도 증대**[1]

4. **이단계 파인튜닝 전략**: 전체 데이터셋으로 기본 모델 학습, 개별 대상 이미지로 주제별 파인튜닝하여 **이미지 충실도와 포즈 일반화의 균형 유지**[1]

***

### 2. 해결하고자 하는 문제

#### 2.1 기본 문제 정의

패션 산업에서 **여전히 사진은 정적 정보**만 제공하며, 동영상은 **의류의 드레이프, 흐름, 움직임**과 같은 중요한 세부사항을 나타낼 수 없다는 한계가 있었다. 특히:[1]

- **비디오 확산 모델의 품질 문제**: 기존 비디오 확산 모델은 이미지 모델만큼의 높은 품질을 달성하지 못하며, 종종 텍스처 흔들림, 카툰 같은 모습 등을 나타낸다[1]

- **시간적 일관성 부족**: 프레임 간 깜박임, 움직임 떨림, 현실감 부족, 운동 및 상세한 객체 모양 제어 불능[1]

- **조건화 신호의 한계**: 기존 텍스트 조건화는 정확한 포즈와 신원에 대한 풍부하고 상세한 정보를 제공하지 못함[1]

***

### 3. 제안하는 방법 및 수식

#### 3.1 확산 모델의 기초

DreamPose는 **잠재 확산 모델(Latent Diffusion Model, LDM)** 프레임워크를 기반으로 한다:[1]

$$L_{DM} = \mathbb{E}_{z,\mathcal{N}(0,1)}[\|\epsilon - \hat{\epsilon}(z_t, t, c)\|_2^2]$$

여기서:
- \(z_t\): 타임스텝 \(t\)에서의 노이즈가 섞인 잠재
- \(c\): 조건 정보 임베딩
- \(\hat{\epsilon}\): 예측 노이즈

#### 3.2 분류기 없는 지도(Classifier-Free Guidance)

기본 공식:[1]

$$\hat{\epsilon}(z_t, t, c) = \epsilon(z_t, t, \emptyset) + s[\epsilon(z_t, t, c) - \epsilon(z_t, t, \emptyset)]$$

여기서 \(s\)는 지도 가중치

#### 3.3 이중 분류기-없는 지도

DreamPose의 혁신적 확장 - 이미지와 포즈 조건을 독립적으로 제어:[1]

$$\hat{\epsilon}(z_t, c_I, c_p) = \epsilon(z_t, \emptyset) + s_I[\epsilon(z_t, c_I) - \epsilon(z_t, \emptyset)] + s_p[\epsilon(z_t, c_I, c_p) - \epsilon(z_t, c_I)]$$

- \(s_I\): 이미지 지도 가중치
- \(s_p\): 포즈 지도 가중치
- \(c_I\): 이미지 조건화 신호
- \(c_p\): 포즈 조건화 신호

#### 3.4 분할 CLIP-VAE 인코더

이미지 조건화 신호 구성:[1]

$$c_I = \mathcal{A}(c_{CLIP}, c_{VAE})$$

여기서:
- \(\mathcal{A}\): 어댑터 모듈
- \(c_{CLIP}\): CLIP 이미지 인코더에서의 임베딩
- \(c_{VAE}\): VAE 인코더에서의 잠재 임베딩

**초기화 전략**: VAE 임베딩 가중치는 0으로 초기화되어 **네트워크 쇼크 방지**

#### 3.5 포즈 조건화

DreamPose의 핵심 특징 - **5개 연속 포즈 사용**:[1]

$$c_p = [p_{i-2}, p_{i-1}, p_i, p_{i+1}, p_{i+2}]$$

이를 UNet 입력층의 10개 추가 채널에 연결:
- **개별 포즈의 프레임-간 지터 노이즈 감소**
- **생성 프레임의 전체 움직임 매끄러움 증대**

***

### 4. 모델 구조

#### 4.1 아키텍처 개요

DreamPose의 핵심 설계는 세 가지 목표 달성:[1]

1. **입력 이미지에 대한 충실도**
2. **시각적 품질**
3. **생성된 프레임 간 시간적 안정성**

#### 4.2 수정된 UNet

원본 Stable Diffusion의 UNet을 수정:[1]

- **입력 채널 확대**: 10개 추가 채널로 5개 연속 포즈 표현 수용 (초기값: 0)
- **원본 채널 유지**: 프리트레인된 가중치 손상 방지
- **공간 정렬 특성**: 포즈는 입력 이미지와 공간적으로 정렬되어 노이즈 연결 사용

#### 4.3 이단계 파인튜닝 전략

**단계 1: 기본 모델 학습**[1]

```
입력: 전체 학습 데이터셋 (UBC Fashion: 339개 비디오)
학습률: 5e-6
에포크: 5
배치 크기: 16 (4 그래디언트 누적)
목표: UNet과 어댑터 모듈 최적화
```

**단계 2: 주제별 파인튜닝**[1]

```
1단계 - UNet 파인튜닝:
  - 입력: 단일 대상 이미지
  - 스텝: 500
  - 학습률: 1e-5
  - 데이터 증강: 랜덤 크롭핑
  
2단계 - VAE 디코더 파인튜닝:
  - 스텝: 1,500
  - 학습률: 5e-5
  - 목표: 시각적으로 일관된 상세 정보 복구
```

**주요 특징**: **드롭아웃 전략**으로 오버피팅(텍스처-스티킹) 방지

#### 4.4 프리트레인 초기화

가중치 초기화 원칙:[1]

- **CLIP 이미지 인코더**: 별도의 프리트레인 체크포인트에서 로드
- **새로운 조건화 신호 가중치**: 0으로 초기화
- **기존 계층**: Stable Diffusion 프리트레인 가중치 유지
- **목표**: 그래디언트 의미성 최대화, 네트워크 성능 회귀 방지

***

### 5. 성능 향상 및 정량적 분석

#### 5.1 정량적 비교 (UBC Fashion 데이터셋)

**Table 1: 최신 방법과의 정량적 비교**[1]

| 방법 | L1 | SSIM | VGG | LPIPS | FID | FVD 16f | AED |
|------|-----|-------|--------|---------|--------|---------|--------|
| MRAA | 0.0857 | 0.749 | 0.534 | 0.212 | 23.42 | 253.65 | 0.0139 |
| TPSMM | 0.0858 | 0.746 | 0.547 | 0.213 | 22.87 | 247.55 | 0.0137 |
| PIDM | 0.1098 | 0.713 | 0.629 | 0.288 | 30.279 | 1197.39 | 0.0155 |
| **DreamPose** | **0.0256** | **0.885** | **0.235** | **0.068** | **13.04** | **238.75** | **0.0110** |

**성능 개선**:[1]

- **L1 에러**: PIDM 대비 77% 감소
- **SSIM**: PIDM 대비 24% 향상
- **FID**: PIDM 대비 57% 감소 (13.04 vs 30.279)
- **FVD**: 시간적 일관성 대폭 향상 (1197.39 → 238.75)

#### 5.2 수치 지표의 의미

- **L1 에러**: 피클셀 재구성 오류 (낮을수록 좋음) → 조건 이미지와의 충실도
- **SSIM**: 구조적 유사성 지표 (높을수록 좋음) → 전반적 이미지 품질
- **LPIPS**: 인지적 손실 (낮을수록 좋음) → 인간 인지와 일치하는 차이
- **FID**: Fréchet Inception Distance (낮을수록 좋음) → 생성 이미지 분포 품질
- **FVD**: Fréchet Video Distance (낮을수록 좋음) → 시간적 일관성

#### 5.3 절제 연구 결과 (Ablation Study)

**Table 2: DreamPose의 설계 선택 검증**[1]

| 변형 | L1 | SSIM | VGG | LPIPS |
|------|-------|--------|--------|---------|
| DreamPose-CLIP | 0.025 | 0.882 | 0.247 | 0.070 |
| DreamPose-NoVAE-FT | 0.025 | 0.897 | 0.210 | 0.057 |
| DreamPose-1pose | 0.019 | 0.899 | 0.208 | 0.056 |
| DreamPose-smooth | 0.767 | 0.758 | 0.502 | 0.202 |
| **DreamPose (Full)** | **0.019** | **0.900** | **0.207** | **0.056** |

**핵심 발견**:[1]

1. **CLIP-VAE 분할 인코더의 중요성**
   - CLIP 전용 대비 세부 사항 포착 향상
   - 의류 패턴, 얼굴 정체성 보존 개선

2. **VAE 디코더 파인튜닝의 필수성**
   - 포토리얼리스틱 세부 사항 복구
   - 고주파 노이즈 감소
   - 개인 신원 유지 가능

3. **5개 연속 포즈의 효과**
   - 단일 포즈: 주제 형태의 뚜렷한 깜박임 (특히 발, 머리)
   - 5개 포즈: 운동 매끄러움 대폭 향상
   - 시간적 일관성 강화

#### 5.4 사용자 연구

**Table 3: 사용자 선호도 (쌍대비교)**[1]

- **DreamPose vs MRAA**: 65% 선호도
- **DreamPose vs TPSMM**: 57% 선호도
- **평점 분석**: 사용자 85%가 3점 이상 (0-5 스케일)

***

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 현재 일반화 능력

**강점**:[1]

1. **데이터 효율성**
   - 매우 작은 데이터셋에서 학습: UBC Fashion 323개 비디오
   - 계산 효율: 2개 A100 GPU, 2일 학습 (PIDM은 26일)

2. **도메인 외 적용성**
   - **DeepFashion 데이터셋**에서의 제로샷 일반화 성공
   - 학습 데이터에 없는 배경, 모델 신원, 액세서리, 패턴 처리 가능
   - **주제별 파인튜닝 후** 새로운 포즈로의 효과적 적응

3. **다중 입력 이미지 지원**
   - 단일 이미지로 높은 품질 달성
   - 추가 입력 이미지로 충실도 향상
   - 시점 일관성 개선

#### 6.2 일반화 향상 메커니즘

**프리트레인된 기초 모델의 활용**:[1]

DreamPose는 Stable Diffusion의 이미 학습된 자연 이미지 분포 사전 지식을 활용:

$$P(\text{이미지 세트}) = P(\text{자연 이미지 분포}) \cap P(\text{포즈 조건})$$

이를 통해 **이미지 애니메이션 작업이 단순화**됨:
- 자연 이미지 분포 학습: Stable Diffusion이 사전 학습으로 달성
- 추가 학습: 조건화 신호와 일치하는 이미지 부분공간 찾기

#### 6.3 일반화 제약 및 한계

**현재 한계점**:[1]

1. **포즈 추정 오류에 취약**
   - 일부 실패 케이스는 개선된 DensePose로 완화 가능
   - 세분화 마스크 추가로 향상 가능

2. **복잡한 패턴의 시간적 일관성**
   - 대규모 복잡한 패턴: 약간의 깜박임 발생
   - 주제별 파인튜닝 없이 더 나은 시간적 일관성 달성 필요

3. **극단적 포즈 변화**
   - 사지가 기저 직물로 침투하는 현상 발생
   - 특징 환각 (hallucination) 발생
   - 후방 향 포즈에서 전방향 예측 오류

4. **계산 효율성**
   - 개인 주제 파인튜닝: UNet 10분 + VAE 디코더 20분
   - 추론 시간: 프레임당 18초 (GAN/VAE 대비 느림)

#### 6.4 일반화 성능 향상을 위한 방향

**데이터셋 확대의 영향**:[1]

다중 입력 이미지 실험에서:
- 단일 입력: 기본 성능
- 3개 입력: 충실도와 시점 일관성 향상
- 5개 이상 입력: 추가 개선 미미

⟹ **더 큰, 더 다양한 학습 데이터**가 일반화 향상의 핵심

**아키텍처 개선 제안**:[1]

1. **향상된 포즈 표현**: 불확실성을 인코딩하는 포즈 표현
2. **세분화 마스크 통합**: 의류 영역과 신체 경계 명시적 정의
3. **적응형 가중치**: 포즈 오차에 따른 동적 가중치 조정

***

### 7. 2020년 이후 관련 최신 연구 비교 분석

#### 7.1 주요 경쟁 방법 비교

**Table 4: 이미지 애니메이션 방법의 진화**[2][3][4][5]

| 방법 | 연도 | 접근 | 장점 | 단점 |
|------|------|------|------|------|
| **FOMM** | 2020 | GAN | 기초 작업 | 팔다리 부정확 |
| **MRAA** | 2021 | GAN | 신체 부위 인식 | 스타일 변형 어려움 |
| **TPSMM** | 2022 | GAN | 유연한 변형 | 대규모 포즈 변화 약함 |
| **PIDM** | 2023 | 확산 | 고품질 | 시간적 일관성 약함 |
| **DreamPose** | 2023 | 확산 | 시간적 일관성 | 계산 비용 |

#### 7.2 확산 기반 방법의 혁신 (2022-2024)

**1. 비디오 확산 모델의 발전**

| 모델 | 연도 | 특징 | 성능 |
|------|------|------|------|
| **Imagen Video** | 2022 | 텍스트-비디오, 계층 구조 | 높은 품질, 느린 속도 |
| **Make-A-Video** | 2022 | 텍스트-비디오 어댑터 | 품질 vs 속도 트레이드오프 |
| **Latent Video DM** | 2023 | 3D CNN 기반 | 계산 효율성 |
| **Tune-A-Video** | 2023 | 이미지 모델 파인튜닝 | 빠른 학습, 품질 저하 |
| **Lumiere** | 2024 | 시공간 확산 | 향상된 일관성 |

#### 7.3 포즈-유도 생성의 진화

**GAN 기반 방법의 한계 (2018-2021)**:[6][4]

- 광학 흐름 의존으로 복잡한 변형 어려움
- 폐색 영역 합성 부정확
- 의류 스타일 보존 어려움

**확산 기반 방법의 장점 (2022-2024)**:[3]

- **PIDM (2023)**: 텍스처 확산 모듈로 세부사항 보존
- **DreamPose (2023)**: 프리트레인 활용으로 효율성 극대화
- **X-MDPT (2024)**: 마스크 확산 변환기로 유연성 증대
- **Cross-view methods (2024)**: 시점 일관성 개선

#### 7.4 시간적 일관성 강화 기법 (2023-2025)

**주요 도전과제**:[7]

1. **프레임 간 깜박임**: 
   - DiffSynth (2023): 잠재 반복 디플리커링
   - Motion-Guided Latent Diffusion (2024): 움직임 기반 샘플링

2. **시공간 특징 모델링**:
   - VideoFusion (2023): 분해된 확산 프로세스
   - BIVDiff (2024): 이미지-비디오 확산 연결

3. **데이터 기반 정규화**:
   - FluxFlow (2025): 시간적 섭동을 통한 학습
   - 분산 감소와 시간적 다양성 향상

#### 7.5 패션 비디오 생성의 특화 발전

**DreamPose 이후의 진화**:[8]

**ProFashion (2025)**:
- 프로토타입-유도 패션 비디오 생성
- 여러 참조 이미지 활용
- 신원 보존 개선

**M2HVideo (2025)**:
- 마네킹-인간 변환
- 포즈 인식 헤드 인코더
- 머리-신체 움직임 정렬

#### 7.6 일반화 성능 비교

**Table 5: 일반화 능력 비교**

| 기준 | MRAA | PIDM | DreamPose | 최신 방법 |
|------|------|------|-----------|----------|
| 데이터 효율성 | 낮음 | 중간 | **높음** | 매우 높음 |
| 계산 비용 | 낮음 | 중간 | 높음 | 중간-높음 |
| 도메인 외 성능 | 낮음 | 중간 | **높음** | 매우 높음 |
| 시간적 일관성 | 중간 | **낮음** | **높음** | **매우 높음** |
| 세부 충실도 | 중간 | 높음 | **매우 높음** | **매우 높음** |

***

### 8. 혁신성 및 학술적 기여도

#### 8.1 학계에 미치는 영향

**패러다임 전환**:

1. **프리트레인 활용의 효율성 입증**
   - 처음부터 학습하는 대신 기존 확산 모델 미세조정
   - 데이터 요구사항 크게 감소
   - 계산 자원 접근성 확대

2. **조건화 메커니즘의 개선**
   - 이미지와 포즈 조건의 분리된 제어
   - 이중 분류기-없는 지도의 개발
   - 교차주의 기반 조건화의 확대

3. **이단계 학습 전략의 중요성**
   - 일반화와 충실도의 균형
   - 주제별 적응의 효율성
   - 오버피팅 방지 기법

#### 8.2 후속 연구 방향

**DreamPose 이후 대표적 연구들**:

| 연구 | 기여 |
|------|------|
| **I2VGen-XL (2023)** | 계층 인코더를 통한 의미 보존 |
| **VideoCrafter (2023)** | 개방형 비디오 생성 모델 |
| **Motion-Guided LDSR (2024)** | 움직임 기반 가이드 |
| **ProFashion (2025)** | 패션 특화 프로토타입 기반 생성 |

***

### 9. 한계점 및 개선 방향

#### 9.1 주요 한계

**1. 포즈 추정 오류의 전파**[1]

현재: DensePose로부터 노이즈가 포함된 포즈 입력
→ 더 정확한 포즈 추정 네트워크 필요

**2. 복잡 패턴의 시간적 일관성**[1]

- 대규모 패턴에서 프레임 간 깜박임
- 원인: 패턴 추적의 어려움
- 해결책: 패턴 특화 모듈 또는 관심 메커니즘

**3. 극단적 포즈 변화에서의 오류**[1]

- 사지 침투: 기하학적 제약 부족
- 특징 환각: 모델이 부분 영역 추측
- 후방향 예측: 전방향으로의 편향

**4. 계산 효율성**[1]

- 프레임당 18초 렌더링
- 개인 파인튜닝: 30분 이상
- GANs/VAEs 대비 속도 열세

#### 9.2 개선 전략

**단기 개선 (기술적)**:

1. **세분화 마스크 통합**
   - 의류와 신체의 명시적 경계
   - 침투 방지 및 오류 감소

2. **향상된 포즈 표현**
   - 불확실성 인코딩
   - 신뢰도 가중 조건화

3. **적응형 시간적 모듈**
   - 동적 학습 가중치
   - 포즈 복잡도 기반 조정

**장기 개선 (데이터 및 아키텍처)**:

1. **대규모 멀티모달 데이터셋**
   - 더 다양한 의류, 신체, 포즈 커버리지
   - 극단 케이스 포함

2. **효율적인 아키텍처**
   - 경량 확산 모델 (2단계, 1단계)
   - 병렬 프레임 생성

3. **특화된 모듈**
   - 패턴 인식 서브네트워크
   - 신원 보존 메커니즘 강화

***

### 10. 앞으로의 연구에 미치는 영향 및 고려사항

#### 10.1 직접적 영향

**1. 산업 응용**

**온라인 소매**:
- 모델 없이 의류 비디오 생성
- 가상 피팅 시스템 강화
- 배송 시간 단축

**소셜 미디어**:
- 동적 패션 콘텐츠 자동 생성
- 영향력 있는 콘텐츠 생산 간소화
- 접근성 향상

**엔터테인먼트**:
- 애니메이션 프로덕션 가속화
- 가상 배우 생성
- 의류 시뮬레이션

**2. 기술 발전**

- **프리트레인 활용 패러다임**: 다른 생성 작업으로 확대
- **멀티 조건 제어**: 더 복잡한 시나리오 가능
- **효율적인 파인튜닝**: 리소스 제약 환경에서의 적응

#### 10.2 후속 연구시 고려사항

**1. 데이터 및 평가**

```
현황:
- UBC Fashion: 339개 비디오 (충분하지 않음)
- 메트릭: 이미지 기반 (FID, LPIPS)

개선 방향:
- 시간적 메트릭 확대 (FVD, TCD)
- 사람 평가 프레임워크 표준화
- 세분화된 평가 (얼굴, 의류, 움직임)
```

**2. 공정성 및 윤리**

주의사항:
- 신원 보존 기술의 악용 가능성
- 가짜 패션 콘텐츠 구분
- 모델의 신체 유형 편향

**3. 계산 효율성**

연구 과제:
- 추론 시간 단축 (현재 18초/프레임)
- 메모리 효율성 개선
- 다중 GPU 병렬화

**4. 도메인 적응**

일반화 확대:
- 다른 인체 애니메이션 작업
- 비패션 객체 애니메이션
- 장면 생성과의 통합

#### 10.3 이론적 과제

**1. 시공간 모델링 심화**

- 확산 프로세스에서 시간 정보 인코딩 최적화
- 프레임 간 의존성 명시적 모델링
- 시간 일관성의 이론적 보장

**2. 조건 상충 해결**

이미지 충실도 vs 포즈 일관성:

$$\text{최적화: } \max_\theta [f(\text{이미지}) + g(\text{포즈}) - \lambda \cdot \text{충돌}]$$


**3. 일반화 이론**

- 제한된 학습 데이터에서의 일반화 경계
- 프리트레인 지식의 전이 효율성
- 도메인 외 성능 분석

#### 10.4 학제간 협력

**예상되는 협력 분야**:

1. **컴퓨터 비전**: 포즈 추정, 세분화, 추적
2. **머신러닝**: 안정적인 생성, 효율적인 학습
3. **그래픽스**: 물리 기반 시뮬레이션, 렌더링
4. **패션 산업**: 도메인 지식, 평가 기준
5. **윤리/사회**: 책임 있는 AI 개발

***

### 11. 결론

DreamPose는 **확산 기반 생성 모델의 효율적 미세조정** 및 **멀티모달 조건화**를 통해 **패션 비디오 합성 분야의 새로운 기준**을 수립했다.[1]

**핵심 성과**:
- 이전 방법 대비 상당한 품질 향상 (FID: 30.28 → 13.04)
- 데이터 및 계산 자원 효율성 (2일, 2개 GPU)
- 도메인 외 일반화 입증

**나아갈 길**:
- 더 큰 데이터셋과 고도화된 아키텍처
- 시간적 일관성의 이론적 기초 강화
- 산업 응용을 위한 계산 최적화

이 연구는 **기초 생성 모델의 전이 학습 가능성**을 보여주며, 향후 비디오 생성, 3D 콘텐츠 생성, 그리고 구현화된 AI 시스템으로 발전할 수 있는 **중요한 토대를 제공**한다.

***

### 참고 문헌 (인용)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/517254c4-e147-4bd9-9c52-d10c88071423/2304.06025v4.pdf)
[2](https://dl.acm.org/doi/10.1145/3707292.3707367)
[3](https://ieeexplore.ieee.org/document/10204031/)
[4](https://openaccess.thecvf.com/content/CVPR2021/papers/Siarohin_Motion_Representations_for_Articulated_Animation_CVPR_2021_paper.pdf)
[5](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Thin-Plate_Spline_Motion_Model_for_Image_Animation_CVPR_2022_paper.pdf)
[6](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ceyuan_Yang_Pose_Guided_Human_ECCV_2018_paper.pdf)
[7](https://arxiv.org/html/2502.17863v1)
[8](https://arxiv.org/html/2505.06537v1)
[9](https://arxiv.org/abs/2507.20478)
[10](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[11](https://dl.acm.org/doi/10.1145/3587423.3595503)
[12](https://dl.acm.org/doi/10.1145/3664475.3664553)
[13](https://ejournal.polraf.ac.id/index.php/JIRA/article/view/663)
[14](https://arxiv.org/abs/2504.16081)
[15](https://arxiv.org/abs/2502.17119)
[16](https://ieeexplore.ieee.org/document/11141031/)
[17](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[18](https://arxiv.org/pdf/2308.03463.pdf)
[19](http://arxiv.org/pdf/2303.08320v3.pdf)
[20](https://arxiv.org/pdf/2211.13221.pdf)
[21](https://arxiv.org/pdf/2502.07001v1.pdf)
[22](http://arxiv.org/pdf/2312.02813.pdf)
[23](https://arxiv.org/abs/2210.02303)
[24](https://arxiv.org/abs/2310.19512)
[25](http://arxiv.org/pdf/2406.07686.pdf)
[26](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
[27](https://research.google/pubs/dreampose-fashion-video-synthesis-with-stable-diffusion/)
[28](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06048.pdf)
[29](https://arxiv.org/abs/2311.04145)
[30](https://grail.cs.washington.edu/projects/dreampose/)
[31](https://github.com/showlab/Awesome-Video-Diffusion)
[32](https://arxiv.org/abs/2304.06025)
[33](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Upscale-A-Video_Temporal-Consistent_Diffusion_Model_for_Real-World_Video_Super-Resolution_CVPR_2024_paper.pdf)
[34](https://arxiv.org/abs/2312.02813)
[35](https://arxiv.org/html/2503.15417v1)
[36](https://arxiv.org/html/2410.05322v1)
[37](https://arxiv.org/html/2510.16833v1)
[38](https://arxiv.org/html/2406.01493v3)
[39](https://arxiv.org/html/2502.07001v1)
[40](https://arxiv.org/html/2511.12578v1)
[41](https://arxiv.org/html/2401.12945v2)
[42](https://arxiv.org/abs/2211.06235)
[43](https://www.semanticscholar.org/paper/98418bc5178fd6c406f8532bf76cbe7adf5271fc)
[44](https://www.semanticscholar.org/paper/c58f046b1f83689b3e101b7866e2cfd86e2a25d9)
[45](https://jamanetwork.com/journals/jama/fullarticle/2792610)
[46](https://casereports.bmj.com/lookup/doi/10.1136/bcr-2022-249691)
[47](https://ojrd.biomedcentral.com/articles/10.1186/s13023-022-02313-w)
[48](https://www.semanticscholar.org/paper/1f282d9f34be2fc79a5c2353ce20a4477fab5af6)
[49](https://academic.oup.com/jes/article/6/Supplement_1/A100/6787291)
[50](https://www.semanticscholar.org/paper/4c825ea7c1579d259714bac9574db25e72992254)
[51](https://arxiv.org/abs/2211.12500)
[52](https://arxiv.org/pdf/2402.01516.pdf)
[53](https://arxiv.org/html/2412.07333)
[54](http://arxiv.org/pdf/2406.06045.pdf)
[55](https://arxiv.org/pdf/2310.06313.pdf)
[56](https://arxiv.org/html/2502.03426)
[57](https://arxiv.org/pdf/2112.05744v3.pdf)
[58](https://arxiv.org/html/2402.00627v3)
[59](https://openaccess.thecvf.com/content/CVPR2023/papers/Bhunia_Person_Image_Synthesis_via_Denoising_Diffusion_Model_CVPR_2023_paper.pdf)
[60](https://www.semanticscholar.org/paper/Person-Image-Synthesis-via-Denoising-Diffusion-Bhunia-Khan/7ec5b25fbbbf7a83a2a04b3f6ae951d7e488badd)
[61](https://openreview.net/pdf/2f5ed5e8e8d64f9f6d7f0f72b9b00408fca4f9e5.pdf)
[62](https://arxiv.org/abs/2203.14367)
[63](https://github.com/ankanbhunia/PIDM)
[64](https://snap-research.github.io/articulated-animation/)
[65](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model)
[66](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/pidm/)
[67](https://arxiv.org/pdf/2511.15092.pdf)
[68](https://pdfs.semanticscholar.org/2f5c/033cfd47da5d0483e51c6d0dd8aec487ee46.pdf)
[69](https://arxiv.org/html/2406.03035v2)
[70](https://arxiv.org/html/2511.16711v1)
[71](https://arxiv.org/html/2401.09146)
[72](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Multi-focal_Conditioned_Latent_Diffusion_for_Person_Image_Synthesis_CVPR_2025_paper.pdf)
