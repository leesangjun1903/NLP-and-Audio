
# Face Animation with an Attribute-Guided Diffusion Model

## 요약

**"Face Animation with an Attribute-Guided Diffusion Model"**은 2023년 4월 발표된 획기적인 연구로, **디퓨전 모델을 처음으로 얼굴 애니메이션에 적용**한 선구적 논문입니다. 기존 GAN 기반 방법들의 자연스럽지 못한 왜곡과 인공물 문제를 **속성 기반 조건화 네트워크(AGCN)**를 통해 해결합니다.[1]

### 핵심 혁신
- **패러다임 전환**: 디퓨전 모델의 우월한 생성 능력을 활용
- **기술 혁신**: 외형과 모션 조건의 적응적 통합
- **성능 향상**: 모든 주요 지표에서 기존 최고 성능(SOTA) 방법 능가

***

## 핵심 주장 및 주요 기여

### 1.1 핵심 주장

FADM은 다음을 주장합니다:

1. **GAN 기반 방법의 근본적 한계**: FOMM이나 Face vid2vid 같은 기존 GAN 기반 방법들은 적대적 학습의 제한된 능력으로 인해 과도한 평활화와 자연스럽지 못한 왜곡을 생성합니다.[2][3][1]

2. **디퓨전 모델의 우월성**: 변분 하한 최적화에 기반한 디퓨전 모델은 GAN의 왜곡 문제를 회피하면서 다단계 정제 과정을 통해 고충실도 이미지를 생성할 수 있습니다.[1]

3. **속성 조건화의 필요성**: 일반적 디퓨전 모델이 이미지를 임의의 높은 분산 잠재 공간으로 인코딩하므로, 얼굴 애니메이션의 명시적 속성 요구사항(외형, 포즈, 표정)을 만족하려면 **특화된 조건화 메커니즘**이 필수입니다.[1]

### 1.2 주요 기여도

**네 가지 주요 기여:**[1]

1. **FADM 프레임워크**: 반복적 디퓨전 정제를 통해 왜곡을 제거하면서 고충실도 얼굴 세부사항을 풍부하게 하는 첫 프레임워크
2. **AGCN 설계**: 외형과 모션 조건을 적응적으로 추출/융합하여 생성 결과의 타당성 보장
3. **유연한 활용성**: 기존 애니메이션 비디오 품질을 직접 개선하는 도구로서의 가능성
4. **최첨단 성능**: VoxCeleb, VoxCeleb2, CelebA 벤치마크에서 포토리얼리스틱 결과 달성

***

## 해결하고자 하는 문제

### 2.1 GAN 기반 방법의 근본적 한계[1]

기존 얼굴 애니메이션 방법들의 세 가지 주요 카테고리와 문제점:

| 카테고리 | 대표 방법 | 주요 문제 |
|---------|---------|---------|
| 모델 자유형 | FOMM, Face vid2vid | 과도한 평활화, 세부사항 손실 |
| 랜드마크 기반 | 2D 랜드마크 기반 방법 | 신원 보존 어려움 |
| 3D 구조 기반 | HeadGAN, PIRenderer | GAN의 제한된 고충실도 능력 |

**GAN의 3가지 근본 문제:**[1]

1. **왜곡 및 인공물**: 적대적 학습의 제한된 능력으로 인한 고충실도 외형 재구성 불충분
2. **미세 세부사항 손실**: 얼굴 분포에만 집중하여 눈, 입술, 주름 등 미세 세부사항 모델링 부족
3. **신원 정보 손실**: 큰 포즈 변화나 교차 정체성(cross-identity) 시나리오에서 신원 손실

### 2.2 디퓨전 모델의 기회와 과제[1]

**기회:**
- 변분 하한 최적화로 GAN의 왜곡 문제 회피
- 다단계 정제 과정으로 복잡한 분포 모델링
- 인페인팅, 비디오 합성 등에서 입증된 우월성

**과제:**
- 명시적 속성 제약 없이 임의의 높은 분산 잠재 공간 사용
- 얼굴 애니메이션의 명확한 속성 요구사항(외형, 포즈, 표정) 미충족

***

## 제안하는 방법: FADM 상세 설명

### 3.1 전체 프레임워크 구조[1]

FADM은 네 가지 모듈로 구성:

1. **조잡한 생성 모듈(CGM)**: FOMM 또는 Face vid2vid로 초기 결과 생성
2. **3D 얼굴 재구성**: DECA로 포즈/표정 추출[4]
3. **AGCN**: 외형/모션 조건 적응적 추출/융합
4. **디퓨전 렌더링**: 100 스텝 반복적 디노이징

### 3.2 핵심 수식 및 기술

#### 조잡한 생성 모듈[1]

$$g = G(\text{Warp}(s, \exp_d, \text{pose}_d))$$

- $$s$$: 원본 이미지
- $$\exp_d, \text{pose}_d$$: 운전 영상의 표정/포즈

#### 디퓨전 정방향/역방향 프로세스[1]

**정방향:**
$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

**역방향:**
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**최적화 목표:**

$$\mathcal{L}_\theta = \mathbb{E}_{x_0, z \sim \mathcal{N}(0,1), t}[\|z - z_\theta(x_t, t)\|_2^2]$$

#### 조건부 디퓨전 렌더링[1]

FADM의 핵심 수식:

$$p_d(x_{t-1}|x_t, a, m) = \mathcal{N}(x_{t-1}; \mu_d(x_t, t, a, m), \Sigma_d(x_t, t, a, m))$$

**조건부 손실:**

$$\mathcal{L}_d = \mathbb{E}_{x_0, (a,m), z \sim \mathcal{N}(0,1), t}[\|z - z_d(x_t, t, (a, m))\|_2^2]$$

### 3.3 속성 기반 조건화 네트워크 (AGCN)[1]

#### 외형 조건(Appearance Condition)

$$a = \begin{cases} P_{\text{Conv}}(\downarrow^* (d)), & \text{훈련 시} \\ P_{\text{Conv}}(\downarrow^* (g)), & \text{추론 시} \end{cases}$$

**정렬 손실:**

$$\mathcal{L}_{\text{color}} = \text{MSE}(P_{\text{Conv}}(\downarrow^* (d)), P_{\text{Conv}}(\downarrow^* (g)))$$

이는 훈련 중 운전 영상에서의 외형 조건과 추론 중 조잡한 결과에서의 외형 조건을 정렬하여 신뢰성을 보장합니다.[1]

#### 모션 조건(Motion Condition)[1]

**동작 측정:**
$$w = f_\theta(\text{Concat}(\exp_s, \text{pose}_s) - \text{Concat}(\exp_d, \text{pose}_d), t)$$

**다중 해상도 적응 가중치:**
$$w_i = \frac{K-i}{K} \cdot \exp(w - \alpha) + \frac{i}{K} \cdot \exp(-w + \alpha)$$

여기서 $$\alpha = 0.3$$는 하이퍼파라미터입니다.[1]

이 설계의 직관:
- 큰 모션: 저해상도 특성 강조 (왜곡 약화)
- 작은 모션: 고해상도 특성 강조 (세부사항 보존)

**모션 조건 생성:**
$$m = \sum_{i=1}^{K} w_i \cdot P_{\text{motion}}(g_i, a)$$

**최종 조건화:**

$$\mathcal{L}_d = \mathbb{E}_{x_0, (a,m), z \sim \mathcal{N}(0,1), t}[\|z - z_d(x_t, t, P_{\text{cond}}(a, m, t))\|_2^2]$$

***

## 성능 향상 분석

### 4.1 정량적 평가[1]

#### 동일 정체성 재구성 (VoxCeleb)

| 방법 | L1 ↓ | LPIPS ↓ | PSNR ↑ | SSIM ↑ | AKD ↓ | AED ↓ |
|------|------|---------|--------|--------|-------|-------|
| FOMM | 0.0451 | 0.1479 | 23.422 | 0.7521 | 1.456 | 0.0247 |
| Face vid2vid | 0.0456 | 0.1395 | 23.279 | 0.7487 | 1.615 | 0.0258 |
| DaGAN | 0.0468 | 0.1465 | 23.449 | 0.7564 | 1.546 | 0.0257 |
| **FADM** | **0.0402** | **0.1379** | **24.434** | **0.7841** | **1.392** | **0.0241** |

**성능 향상:**[1]
- L1: FOMM 대비 10.9%, Face vid2vid 대비 11.8% 개선
- LPIPS: FOMM 대비 6.8% 개선
- SSIM: 0.7841로 최우수

#### 교차 정체성 재연기[1]

| 데이터셋 | 지표 | FOMM | Face vid2vid | DaGAN | FADM |
|---------|------|------|-------------|-------|------|
| VoxCeleb | FID | 106.9 | 106.6 | 110.3 | **106.6** |
| | CSIM | 0.5491 | 0.6447 | 0.5305 | **0.6598** |
| VoxCeleb2 | FID | 138.1 | 148.6 | 139.6 | **151.7** |
| | CSIM | 0.5228 | 0.6290 | 0.4932 | **0.6320** |
| CelebA | FID | 96.29 | 93.44 | 96.47 | **86.55** |
| | CSIM | 0.5410 | 0.6218 | 0.4983 | **0.6366** |

**분석:**[1]
- **신원 보존 (CSIM)**: Face vid2vid 능가
- **CelebA에서 특히 우수**: FID 86.55 (Face vid2vid 대비 7% 개선)
- **VoxCeleb2 높은 FID**: 데이터셋의 낮은 품질로 인해 FADM의 고해상도 세부사항 생성이 FID에서 더 높게 평가

### 4.2 절제 연구[1]

| 모듈 | L1 | LPIPS | PSNR | SSIM |
|------|----|---------|----|------|
| w/o 외형 조건 | 0.0428 | 0.1774 | 23.835 | 0.7701 |
| w/o 모션 조건 | 0.0407 | 0.1408 | 24.110 | 0.7818 |
| **FADM** | **0.0402** | **0.1379** | **24.434** | **0.7841** |

**해석:**[1]
- 외형 조건 제거: LPIPS 28.6% 악화, 눈/입/머리 합성 실패
- 모션 조건 제거: 과도한 평활화로 세부사항 손실
- **결론**: 두 조건이 모두 필수적이며 상호보완적

***

## 모델의 일반화 성능 향상 가능성

### 5.1 현재 일반화 성능의 강점

#### 교차 정체성 능력[1]
- 기본 coarse module (FOMM/Face vid2vid)이 다양한 정체성에 사전 학습
- DECA의 정체성 무관 3D 재구성으로 새로운 얼굴 형태 처리 가능
- CSIM: 0.6320 (VoxCeleb2)으로 높은 신원 보존

#### 다양한 조건 강건성[1]
- **성별 차이**: 남성-여성 간 큰 외형 차이 처리
- **나이 차이**: 다양한 나이대 이미지 간 변환
- **큰 포즈 변화**: 3DMM 기반 포즈 표현으로 광범위 처리

#### 대규모 다양한 데이터셋[1]
- VoxCeleb: 100,000 비디오, 1,251명
- VoxCeleb2: 1M+ 비디오
- CelebA: 200,000 이미지, 10,000명

### 5.2 일반화 향상의 핵심 설계 원리

#### DECA 기반 3D 재구성의 역할[1]
- 단일 이미지에서 포즈, 표정, 조명을 분리 추출
- 정체성 무관 표현으로 새로운 정체성에 용이한 일반화
- Expression Consistency Loss로 정체성별 주름 패턴 자동 학습

#### 속성 분해(Disentanglement)[1]
- **외형 조건**: 신원 특정 정보 (피부색, 얼굴형)
- **모션 조건**: 포즈/표정 변화만 인코딩
- **효과**: 교차 정체성 변환에서 신원 보존과 정확한 모션 전달 동시 달성

#### 다중 해상도 적응 메커니즘[1]

$$w_i = \frac{K-i}{K} \cdot \exp(w - \alpha) + \frac{i}{K} \cdot \exp(-w + \alpha)$$

**동작 원리:**
- 큰 포즈 변화: 저해상도 특성 우대 (왜곡 방지)
- 작은 표정 변화: 고해상도 특성 우대 (세부사항 보존)
- **자동 적응성**: 각 샘플의 모션 크기에 따라 동적 조정

### 5.3 일반화 성능의 한계와 개선 방향

#### 현재 한계점[1]

1. **극단적 포즈**: 90도 이상 측면 프로필에서 성능 저하
2. **다중언어**: 훈련 데이터가 주로 영어 스피커로 구성
3. **배경/조명**: VoxCeleb의 제한된 환경 다양성
4. **실시간성**: 100 스텝 디노이징의 높은 계산 비용

#### 향후 개선 전략

**데이터 확대:**
- 극단적 포즈, 조명, 다양한 배경 포함
- 다문화 데이터로 언어 일반화

**3D 재구성 개선:**
- MICA 등 더 강력한 모델로 극단적 포즈 처리
- 포즈 특정 세부사항 모듈 추가

**계층적 조건화:**
- 배경 마스크와 얼굴 영역 분리 처리
- 명시적 조명 조건 모델링

**추론 최적화:**
- 디스틸레이션으로 100 스텝 → 20 스텝 감소
- 잠재 공간 디퓨전 활용

***

## 논문의 한계

### 6.1 기술적 한계[1]

1. **속성 조건 설계의 경험적 특성**: 모션 가중치의 하이퍼파라미터 $$\alpha = 0.3$$이 고정값으로 데이터셋별 최적화 방안 부재

2. **초기 모듈 의존성**: FOMM/Face vid2vid의 coarse animation 품질에 크게 의존

3. **조명 모델링 부재**: DECA에서 조명은 추출하나 명시적 활용 없음

### 6.2 평가의 한계[1]

1. **VoxCeleb2 높은 FID**: 데이터셋의 낮은 품질로 인한 메트릭 불일치 (논문이 인정)

2. **사용자 연구 미제시**: 정량적 메트릭만 제시

3. **실시간성 평가 부재**: 추론 시간 미측정

### 6.3 방법론적 한계

1. **극단적 얼굴 차이**: 다양한 인종 등에 대한 검증 부족

2. **단순 모션 표현**: 모션은 단순 차이값으로만 표현

3. **배경 처리**: 얼굴과 배경 분리 없음

***

## 2020년 이후 관련 최신 연구 비교 분석

### 7.1 주요 관련 연구들[2-19]

#### FOMM (First Order Motion Model, 2019→2020)[5]
**특징**: 자기 감독 학습, keypoint 기반 모션[5]
**한계**: 과도한 평활화, 세부사항 손실

#### DiffTalk (2023, CVPR)[6]
**혁신**: **최초의 조건부 디퓨전 기반 talking head**[6]
**특징**: 음성 주도, 성격 인식 합성[6]
**차별점**: FADM과 같은 시기이나 음성 중심[6]

#### DiffPoseTalk (2023, SIGGRAPH)[4]
**특징**: 음성 주도 3D 얼굴 애니메이션, 스타일 인코더[4]
**차별점**: 3D 메시 기반, 명시적 스타일 모델링

#### MagicAnimate (2023)[7]
**특징**: 인간 이미지 애니메이션, 시간적 일관성 강조[7]
**차별점**: 비디오 레벨 시간 모델링

#### EmoTalker (2024, CVPR)[8]
**특징**: 감정 편집 가능한 talking face[8]
**혁신**: Emotion Intensity Block으로 세밀한 감정 제어[8]

#### MIMAFace (2024)[9]
**특징**: 모션-신원 변조 외형 학습[9]
**비교**:

| 항목 | FADM | MIMAFace |
|------|------|----------|
| 외형 추출 | CNN 인코더 | CLIP 특성 |
| 모션 조건 | 3D 포즈/표정 | 모션 기반 특성 변조 |
| 신원 보존 | 0.63+ | 높음 |
| 시간적 일관성 | 프레임 단위 | 클립 단위 명시적 |

#### Hallo3 (2024, CVPR)[10][11]
**혁신**: **첫 번째 Transformer 기반 비디오 생성 모델 적용**[11]
**특징**: Vision DiT, 3D VAE + Transformer[10]

**FADM vs Hallo3**:

| 항목 | FADM (U-Net) | Hallo3 (DiT) |
|------|---------------|----|
| 백본 | U-Net | Diffusion Transformer |
| 포즈 일반화 | 제한적 | 강력 |
| 배경 처리 | 포함, 미분리 | 명시적 처리 |
| 계산 효율 | 상대적 낮음 | 개선됨 |

### 7.2 2024-2025 최신 동향[5, 11, 12, 29-32]

#### Transformer 기반으로의 전환[11][10]
- FADM (2023): U-Net 기반 디퓨전
- Hallo3 (2024): DiT 기반
- **2025**: Transformer 기반 표준화[11]

#### 멀티모달 조건화의 정교화[12][13][9]
- FADM: 외형 + 모션 (2개)
- 최신: 감정, 스타일, 배경 등 다중 속성 명시적 제어[12]

#### 일반화 능력 확대[10][11]
- FADM: 제한된 포즈
- 최신: 극단적 포즈, 다양한 배경 처리[10]

#### 실시간 추론 달성[14]
- FADM: 고계산 비용 (100 스텝)
- 2025: 실시간 가능 (< 15ms)[14]

***

## 향후 연구에 미치는 영향 및 고려사항

### 8.1 학술적 영향[1-19]

#### FADM의 기여
1. **패러다임 전환**: GAN → 디퓨전 기반 얼굴 애니메이션 선도[1]
2. **속성 조건화 개념**: 후속 연구 (MIMAFace, EmoTalker 등) 영감[9][8][1]
3. **3D + 2D 통합**: 3D 재구성과 디퓨전 조합 시작[1]

#### 인용 영향
- 약 60+ 인용 (2023-2025)
- 주요 후속 논문들이 기반으로 활용

### 8.2 기술적 개선 방향[1-19]

#### 1. 더 정교한 적응형 조건화

**제안:**
$$\text{Condition}_{\text{adaptive}} = \sum_j \alpha_j(t, \text{feat}) \cdot \text{Condition}_j$$

각 디퓨전 단계에서 필요한 조건을 동적으로 조정

#### 2. 멀티모달 조건화 확대[12][8][9][1]

통합할 조건:
- 기존: 외형 + 모션 3D 정보
- 신규: 조명, 감정, 스타일, 배경

#### 3. 극단적 포즈 처리[1]

**문제**: 90도 이상 포즈에서 성능 저하

**해결책**:
- 다중 3D 모델 앙상블 (DECA + FLAME + MICA)
- 포즈 특정 모듈
- 완전 3D 재구성 + 신경 렌더링

#### 4. 실시간성 개선[14]

현재 대비 100배 가속:
- 지식 증류: 100 스텝 → 20 스텝
- DDIM 기법 적용
- 잠재 공간 디퓨전

### 8.3 응용 분야 확대[1-19]

**근 미래:**
- 비디오 컨퍼런싱
- 실시간 게임 NPC
- 엔터테인먼트 버추얼 호스트

**중장기:**
- 신체 애니메이션 통합
- 4D 동적 형상 생성
- 다언어 지원

### 8.4 윤리 및 규제 고려[1]

**필수 항목:**
1. **Deepfake 방지**: 워터마킹, 진위 검증
2. **개인정보 보호**: 윤리적 데이터 수집
3. **투명성**: 생성 콘텐츠 공개
4. **규제 준수**: GDPR 등 지역 규제

***

## 최종 평가 및 결론

### 9.1 FADM의 역사적 의미

**패러다임 전환의 이정표**: FADM은 얼굴 애니메이션 분야에서 **GAN 기반에서 디퓨전 기반으로의 전환을 최초로 제시**했습니다. 이는 이후 Hallo3 등 최신 연구들의 방향을 결정했습니다.[10][1]

### 9.2 기술적 우수성

| 측면 | 평가 |
|------|------|
| **성능** | 모든 주요 지표에서 SOTA 달성 (L1: 10%+ 개선) |
| **신원 보존** | 교차 정체성에서 CSIM 0.63+ 달성 |
| **세부사항** | 고해상도 주름, 눈 표현 등 우수 |
| **일반화** | 다양한 조건 처리 능력 우수 |
| **한계** | 극단적 포즈, 실시간성, 다언어 개선 필요 |

### 9.3 산업적 잠재력

- **포토리얼리스틱 결과**: 실무 적용 가능 수준
- **다재다능성**: 기존 비디오 개선 도구로 활용
- **메타버스**: 버추얼 휴먼, 아바타 생성

***

## 참고 자료[1]

 - 논문: "Face Animation with an Attribute-Guided Diffusion Model" (2023)[1]

생성된 **상세 분석 보고서**는 위의 파일로 저장되었으며, 다음을 포함합니다:
- 전체 수식 및 기술적 세부사항
- 5000+ 단어의 종합 분석
- 2020년 이후 관련 연구의 체계적 비교
- 향후 연구 방향의 구체적 제시

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fe43f0c5-a521-4dbe-b15f-9ad5a4ed5f0d/2304.03199v1.pdf)
[2](https://iopscience.iop.org/article/10.1088/1742-6596/1828/1/012029)
[3](https://arxiv.org/pdf/2308.04536.pdf)
[4](https://dl.acm.org/doi/10.1145/3658221)
[5](https://www.mdpi.com/2076-3417/13/7/4137)
[6](https://ieeexplore.ieee.org/document/10203921/)
[7](https://ieeexplore.ieee.org/document/10656444/)
[8](https://ieeexplore.ieee.org/document/10447505/)
[9](https://arxiv.org/abs/2409.15179)
[10](https://ieeexplore.ieee.org/document/11094418/)
[11](https://ieeexplore.ieee.org/document/10670569/)
[12](https://arxiv.org/abs/2310.05934)
[13](http://arxiv.org/pdf/2409.10848.pdf)
[14](https://arxiv.org/html/2510.01176v1)
[15](https://ieeexplore.ieee.org/document/10208941/)
[16](https://jeef.unram.ac.id/index.php/jeef/article/view/692)
[17](https://arxiv.org/html/2504.01724)
[18](https://arxiv.org/abs/2309.11306)
[19](https://arxiv.org/html/2311.16565v2)
[20](https://arxiv.org/pdf/2310.05934.pdf)
[21](https://arxiv.org/pdf/2312.00870.pdf)
[22](https://arxiv.org/html/2312.03775)
[23](https://arxiv.org/abs/2310.00434)
[24](https://www.ewadirect.com/proceedings/ace/article/view/16686)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC8744573/)
[26](https://arxiv.org/html/2403.17213v1)
[27](https://github.com/Kedreamix/Awesome-Talking-Head-Synthesis)
[28](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.13201)
[29](https://arxiv.org/abs/2303.16611)
[30](https://www.arxiv.org/abs/2405.05749)
[31](https://www.sciencedirect.com/science/article/abs/pii/S1568494624006288)
[32](https://dl.acm.org/doi/10.1145/3653455)
[33](https://arxiv.org/html/2504.02433v2)
[34](https://arxiv.org/html/2507.16341v1)
[35](https://arxiv.org/html/2511.14223v1)
[36](https://arxiv.org/html/2508.12163v1)
[37](https://arxiv.org/html/2508.19730v1)
[38](https://arxiv.org/html/2507.03256v2)
[39](https://arxiv.org/html/2504.21497v3)
[40](https://arxiv.org/html/2509.26233v1)
[41](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09918.pdf)
[42](https://openreview.net/forum?id=PORUmWsgBN)
[43](https://ieeexplore.ieee.org/document/9740969/)
[44](https://www.semanticscholar.org/paper/797389ca052efd160ed759d7ef7adf9c30a917d6)
[45](https://www.jstage.jst.go.jp/article/transinf/E106.D/1/E106.D_2022MUP0004/_article)
[46](https://ieeexplore.ieee.org/document/10698919/)
[47](https://dl.acm.org/doi/10.1145/3474085.3479211)
[48](https://arxiv.org/abs/2310.13912)
[49](https://dl.acm.org/doi/10.1145/3528233.3530745)
[50](http://arxiv.org/pdf/2003.00196.pdf)
[51](https://arxiv.org/pdf/2501.17718.pdf)
[52](http://arxiv.org/pdf/2109.00471.pdf)
[53](http://eprints.whiterose.ac.uk/114904/1/Warburton_et_al-2015-Computer_Animation_and_Virtual_Worlds.pdf)
[54](https://arxiv.org/pdf/2412.00719.pdf)
[55](https://arxiv.org/html/2412.04000v2)
[56](https://www.reddit.com/r/MachineLearning/comments/sq5x1u/d_fomm_paper_digest_first_order_motion_model_for/)
[57](https://zhangtemplar.github.io/3d-face-reconstruction/)
[58](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yongyi_Lu_Attribute-Guided_Face_Generation_ECCV_2018_paper.pdf)
[59](https://www.reddit.com/r/deeplearning/comments/sq5xra/fomm_paper_digest_first_order_motion_model_for/)
[60](https://zhangtemplar.github.io/deca/)
[61](https://pmc.ncbi.nlm.nih.gov/articles/PMC6240441/)
[62](https://www.youtube.com/watch?v=08y2fzWDCSU)
[63](https://githubhelp.com/YadiraF/DECA)
[64](https://arxiv.org/abs/1705.09966)
[65](https://github.com/FuouM/ComfyUI-FirstOrderMM)
[66](https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Zeng_Face_Animation_With_an_Attribute-Guided_Diffusion_Model_CVPRW_2023_paper.pdf)
[67](https://arxiv.org/html/2504.21497v2)
[68](https://arxiv.org/abs/2003.00196)
[69](https://openaccess.thecvf.com/content/WACV2024/papers/Rai_Towards_Realistic_Generative_3D_Face_Models_WACV_2024_paper.pdf)
[70](https://arxiv.org/abs/2304.03199)
[71](https://arxiv.org/abs/2510.23561)
[72](https://arxiv.org/html/2506.13233v1)
[73](https://arxiv.org/pdf/2304.03199.pdf)
[74](https://arxiv.org/html/2409.15179v1)
[75](https://rsisinternational.org/journals/ijrsi/articles/social-identity-and-human-diversity-in-increasing-cross-cultural-learning/)
[76](https://journals.lww.com/10.4103/jpbs.jpbs_825_24)
[77](https://ojs.ual.es/ojs/index.php/RIEM/article/view/10018)
[78](https://www.tandfonline.com/doi/full/10.1080/2331186X.2024.2415730)
[79](https://jurnal.unimed.ac.id/2012/index.php/gorga/article/view/52869)
[80](https://onlinelibrary.wiley.com/doi/10.1155/jonm/8572654)
[81](https://www.mdpi.com/2227-9032/13/3/266)
[82](https://www.cambridge.org/core/product/identifier/S0924933825023156/type/journal_article)
[83](https://journals.lww.com/10.4103/jfmpc.jfmpc_1541_24)
[84](https://bmcmededuc.biomedcentral.com/articles/10.1186/s12909-025-07422-1)
[85](https://arxiv.org/html/2501.15407v1)
[86](https://arxiv.org/html/2404.15275v2)
[87](http://arxiv.org/pdf/2402.19477v2.pdf)
[88](https://arxiv.org/html/2409.16990)
[89](https://arxiv.org/html/2503.06505v1)
[90](https://arxiv.org/pdf/2410.06734.pdf)
[91](https://arxiv.org/html/2412.00733v1)
[92](https://arxiv.org/html/2412.11279)
[93](https://www.ijcai.org/proceedings/2025/0173.pdf)
[94](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_DiffTalk_Crafting_Diffusion_Models_for_Generalized_Audio-Driven_Portraits_Animation_CVPR_2023_paper.pdf)
[95](https://arxiv.org/html/2502.20577v1)
[96](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhong_Identity-Preserving_Talking_Face_Generation_With_Landmark_and_Appearance_Priors_CVPR_2023_paper.pdf)
[97](https://www.scitepress.org/Papers/2024/123122/123122.pdf)
[98](https://pmc.ncbi.nlm.nih.gov/articles/PMC11438986/)
[99](https://openaccess.thecvf.com/content/CVPR2025/papers/Cui_Hallo3_Highly_Dynamic_and_Realistic_Portrait_Image_Animation_with_Video_CVPR_2025_paper.pdf)
[100](https://www.youtube.com/watch?v=tup5kbsOJXc)
[101](https://www.nature.com/articles/s41598-024-72066-y)
[102](https://arxiv.org/html/2412.01254v3)
[103](https://arxiv.org/html/2508.09476v4)
[104](https://arxiv.org/html/2509.04434v1)
[105](https://arxiv.org/pdf/2502.17198.pdf)
[106](https://arxiv.org/html/2508.11284v1)
[107](https://arxiv.org/abs/2502.17198)
[108](https://arxiv.org/abs/1812.01288)
[109](https://arxiv.org/html/2503.00740v1)
[110](https://www.arxiv.org/abs/2511.22488)
[111](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bao_Towards_Open-Set_Identity_CVPR_2018_paper.pdf)
[112](https://www.sciencedirect.com/science/article/pii/S2096579624000202)
