# DreamFusion: Text-to-3D using 2D Diffusion

### 1. 핵심 주장과 주요 기여

**DreamFusion**은 Google Research와 UC Berkeley에서 발표한 혁신적인 text-to-3D 생성 방법으로, 다음의 핵심 주장을 제시합니다:[1]

#### 주요 주장

DreamFusion은 **3D 생성을 위해 대규모 3D 데이터셋을 필요로 하지 않는다**는 혁신적 관점을 제시합니다. 기존 접근법이 3D 데이터 부족 문제에 직면했던 반면, DreamFusion은 이미 수십억 개의 이미지-텍스트 쌍으로 훈련된 **사전학습된 2D text-to-image 확산 모델(Diffusion Model)을 3D 도메인으로 전이**함으로써 이 한계를 극복합니다.[1]

#### 주요 기여

1. **Score Distillation Sampling (SDS)**: 확률 밀도 증류(Probability Density Distillation)에 기반한 새로운 손실 함수를 도입하여, 2D 확산 모델이 임의의 매개변수 공간에서 샘플링에 사용될 수 있게 함[1]
2. **3D 이미지 매개변수화**: 미분 가능한 image parameterization(DIP)으로 Neural Radiance Field(NeRF)를 사용하여 텍스트 설명을 실제 렌더링 가능한 3D 자산으로 변환[1]
3. **무감독 3D 생성**: 3D 훈련 데이터나 확산 모델 수정 없이 순수 텍스트만으로 다양한 3D 장면 생성 가능[1]

***

### 2. 문제 정의 및 해결 방법

#### 2.1 해결하고자 하는 문제

DreamFusion이 직면한 핵심 문제는 **3D 및 다중 뷰 데이터의 극심한 부족**입니다.[1]

- 2D 이미지: 수십억 개의 대규모 데이터셋 존재
- 3D 자산: 상대적으로 매우 적은 수량
- 기존 3D 생성 모델: 충분한 학습 데이터 부족으로 인한 성능 저하

따라서 원래의 물음은 다음과 같습니다: **"어떻게 3D 데이터 없이 텍스트 설명만으로 고품질의 일관된 3D 모델을 생성할 수 있을까?"**

#### 2.2 Score Distillation Sampling (SDS) 방법

DreamFusion의 핵심 혁신은 **Score Distillation Sampling (SDS)** 손실 함수입니다.[1]

**SDS의 수학적 정의:**

기본 확산 손실(Diffusion loss)에서 시작하면:

$$L_{\text{Diff}}(\theta, x) = \mathbb{E}_{t \sim U(0,1), \epsilon \sim \mathcal{N}(0,I)} \left[ w_t \left\| \hat{\epsilon}_\theta(z_t, y, t) - \epsilon \right\|_2^2 \right]$$

여기서 $\hat{\epsilon}_\theta$는 U-Net이 예측한 노이즈이고, $w_t$는 시간 단계 $t$에 따른 가중치입니다.[1]

그러나 DreamFusion의 핵심 관찰은 이 손실의 기울기에서 **U-Net 야코비안 항을 제거**하면 더 효과적인 기울기가 얻어진다는 것입니다:

$$L_{\text{SDS}}(\theta, x = g(\theta)) = \mathbb{E}_{t, \epsilon} \left[ w_t (\hat{\epsilon}_\theta(z_t, y, t) - \epsilon) \frac{\partial x}{\partial \theta} \right]$$

여기서 $z_t = \alpha_t x + \sigma_t \epsilon$는 노이즈가 추가된 이미지이고, $x = g(\theta)$는 NeRF로부터의 렌더링입니다.[1]

**확률 밀도 증류 해석:**

이 손실은 다음의 가중 확률 밀도 증류로 해석될 수 있습니다:

$$L_{\text{SDS}}(\theta, x = g(\theta)) = \mathbb{E}_t \left[ \lambda_t w_t \text{KL}\left( q(z_t | g(\theta), y, t) \| p(z_t | y, t) \right) \right]$$

여기서 $\text{KL}$은 Kullback-Leibler 발산이며, $q$는 forward diffusion process, $p$는 역 과정입니다.[1]

**Classifier-Free Guidance (CFG) 적용:**

향상된 품질을 위해 점수 함수에 가이던스 가중치 $\gamma$를 적용합니다:

$$\hat{\epsilon}_\theta^{\text{guided}}(z_t, y, t) = (1 + \gamma) \hat{\epsilon}_\theta(z_t, y, t) - \gamma \hat{\epsilon}_\theta(z_t, \emptyset, t)$$

DreamFusion에서는 $\gamma = 100$의 매우 높은 가이던스 가중치를 사용하여 높은 품질을 달성합니다.[1]

#### 2.3 Neural Radiance Field (NeRF) 기반 3D 렌더링

DreamFusion은 **mip-NeRF 360**을 기반으로 개선된 NeRF 구조를 사용합니다.[1]

**Volume Rendering 방정식:**

NeRF의 기본 렌더링은 다음과 같습니다:

```math
C_i = \sum_j w_i c_i, \quad w_i = \alpha_i \prod_{j < i}(1 - \alpha_j), \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i)
```

여기서 $\sigma_i$는 부피 밀도(volumetric density), $c_i$는 RGB 색상, $\delta_i$는 샘플 간격입니다.[1]

**표면 정규화 벡터:**

표면 법선은 밀도의 기울기로부터 계산됩니다:

$$\mathbf{n} = -\nabla_{\mathbf{x}} \sigma(\mathbf{x})$$

**디퓨즈 쉐이딩(Lambertian Reflectance):**

표면 색상은 다음과 같이 결정됩니다:

$$c = \max(0, \mathbf{n} \cdot (\mathbf{l} - \mathbf{x})) \rho + a$$

여기서 $\rho$는 albedo(표면 색상), $\mathbf{l}$은 광원 위치, $a$는 주변광입니다.[1]

**MLP 아키텍처:**

- **입력**: Integrated positional encoding을 사용한 3D 좌표 및 광선 정보
- **구조**: 5개의 ResNet 블록, 128개의 숨겨진 유닛, SwishSiLU 활성화 함수
- **출력**: 부피 밀도 $\sigma$ (exp 활성화) 및 RGB albedo $\rho$ (sigmoid 활성화)[1]

#### 2.4 최적화 절차

DreamFusion의 각 반복에서 수행되는 작업:[1]

1. **카메라 및 조명 샘플링**: 구면 좌표에서 다양한 관점과 조명 조건을 샘플링
   - 고도각 $\theta_{\text{cam}} \in [10°, 90°]$
   - 방위각 $\phi_{\text{cam}} \in [0°, 360°]$
   - 거리 $r \in [1, 1.5]$ (경계 구 반지름 1.4)

2. **NeRF 렌더링**: 샘플링된 카메라 위치에서 64×64 해상도로 렌더링

3. **뷰 의존적 텍스트 프롬프트 조정**: 카메라 각도에 따라 프롬프트를 동적으로 수정
   - 고도 $\theta_{\text{cam}} > 60°$: "overhead view" 추가
   - 저도: "front view", "side view", "back view" 등 추가

4. **SDS 손실 계산**: 렌더링된 이미지와 노이즈 예측으로부터 손실 계산

5. **매개변수 업데이트**: Distributed Shampoo 최적화기를 사용하여 NeRF 가중치 업데이트
   - 총 15,000 반복
   - 약 1.5시간 소요 (TPUv4 4칩)

***

### 3. 모델 구조 및 아키텍처

#### 3.1 전체 시스템 아키텍처

```
Text Prompt
    ↓
T5-XXL Text Embedding
    ↓
Random NeRF Initialization
    ↓
┌─────────────────────────────────────┐
│  DreamFusion Optimization Loop       │
│  (15,000 iterations)                │
├─────────────────────────────────────┤
│ 1. Sample Camera & Light Position    │
│ 2. Render NeRF at 64×64 resolution  │
│ 3. Compute SDS Loss with Imagen     │
│ 4. Backprop through NeRF MLP        │
│ 5. Update NeRF Parameters           │
└─────────────────────────────────────┘
    ↓
Optimized NeRF Parameters
    ↓
3D Model (exportable as mesh)
```

#### 3.2 NeRF MLP 구조 상세

```
Input: Integrated Positional Encoding
   ↓
ResNet Block (128 units, SwishSiLU)
   ↓
Layer Normalization
   ↓
[Repeat 4 times]
   ↓
Output Head 1: exp() → σ (Density)
Output Head 2: sigmoid() → ρ (Albedo)
```

#### 3.3 Imagen 텍스트-이미지 모델

- **기본 모델**: 64×64 해상도의 조건부 확산 모델
- **텍스트 인코더**: T5-XXL로부터의 텍스트 임베딩
- **노이즈 예측**: U-Net 기반 $\hat{\epsilon}_\theta(z_t, y, t)$
- **제약**: Classifier-Free Guidance with $\gamma = 100$

***

### 4. 성능 향상 및 실험 결과

#### 4.1 정량적 평가 지표

DreamFusion은 **CLIP R-Precision** 메트릭을 사용하여 텍스트-3D 일치도를 평가합니다.[1]

**R-Precision 정의**: 렌더링된 이미지가 주어진 경우 CLIP이 정확한 캡션을 검색할 수 있는 정확도

| 방법 | CLIP B32 | CLIP B16 | CLIP L14 |
|------|----------|----------|----------|
| 색상 이미지 | | | |
| Ground Truth | 77.1 | 79.1 | 79.1 |
| Dream Fields | 68.3 | 74.2 | - |
| DreamFusion | **75.1** | **77.5** | **79.7** |
| 무텍스처 기하학(Geometry) | | | |
| Dream Fields | 42.5 | 46.6 | 58.5 |
| DreamFusion | **42.5** | **46.6** | **58.5** |[1]

#### 4.2 절제 연구(Ablation Study)

DreamFusion의 각 구성요소의 영향을 분석한 결과:[1]

| 구성요소 | Albedo | Shaded | Textureless |
|---------|--------|--------|------------|
| 기본 모델 (Base) | 0.85 | 0.45 | 0.15 |
| + 뷰 증강 (ViewAug) | 0.82 | 0.52 | 0.28 |
| + 뷰 의존 프롬프트 (ViewDep) | 0.80 | 0.58 | 0.42 |
| + 조명 (Lighting) | 0.78 | 0.65 | 0.52 |
| + 무텍스처 렌더링 (Textureless) | 0.75 | 0.68 | **0.58** |

**핵심 발견**: 정확한 기하학 복원을 위해 뷰 의존 프롬프트, 조명 다양성, 무텍스처 렌더링이 모두 필수적입니다.[1]

***

### 5. 일반화 성능 및 한계

#### 5.1 일반화 성능 강점

1. **광범위한 프롬프트 다양성**: 현실적 객체부터 상상력 있는 장면까지 다양한 텍스트 설명 지원
2. **3D 데이터 독립성**: 어떤 3D 훈련 데이터도 필요하지 않음
3. **전이 학습 효과성**: 사전학습된 2D 모델의 강력한 선행(Prior) 활용
4. **모델 불변성**: 원래 Imagen 모델 수정 없이 사용 가능

#### 5.2 근본적 한계

**모드 추구(Mode-seeking) 문제**:[1]

$$\text{역 KL 발산: } D_{\text{KL}}(q||p) = \mathbb{E}_q[\log q - \log p]$$

이는 고-에너지 영역을 무시하고 저-에너지 영역에만 집중하는 경향이 있어:
- 결과의 다양성 부족
- 과도한 평활화(Oversmoothening)
- 색상 포화(Oversaturation)

**구체적 제한사항**:[1]

1. **해상도 제한**: 64×64 Imagen 모델 사용으로 인한 세부 사항 부족
2. **역 렌더링의 근본적 모호성**: 동일한 2D 이미지를 만드는 다양한 3D 구조 존재
3. **국소 최솟값**: 모든 콘텐츠가 평탄한 표면에 그려지는 퇴화된 해석
4. **계산 시간**: 단일 객체 생성에 약 1.5시간 소요
5. **다중 뷰 일관성 부족**: 단일 2D 뷰에서만 최적화되는 문제

***

### 6. 최신 연구 기반 일반화 성능 향상 방향

DreamFusion 이후의 발전 연구들이 제시한 개선 방안을 분석하면 다음과 같습니다:[2][3][4][5][6]

#### 6.1 SDS 개선 방법들

**1. DreamFlow (2024)**:[7][8]
- **개선 사항**: 확률 흐름(Probability Flow) 근사를 통한 결정론적 타임스텝 스케줄
- **성능**: 5배 빠른 최적화 + 고해상도(1024×1024) 생성 가능
- **효과**: 무작위 타임스텝의 높은 분산 문제 해결

**2. Score Distillation via Inversion (SDI)**:[9]
- **개선 사항**: DDIM 반전(Inversion) 활용으로 일관된 노이즈 궤적 유지
- **성능**: 과도한 포화 현상 감소, 품질-다양성 트레이드오프 개선

**3. Denoised Score Distillation (DSD)**:[10]
- **개선 사항**: 이전 반복의 음의 기울도를 사용하여 과도한 평활화 대항
- **성능**: 더 선명하고 의미론적으로 정확한 텍스처

#### 6.2 다중 뷰 일관성 개선

**MVDream (2024)**:[11][12]
- **혁신**: 2D와 3D 데이터로부터 학습된 다중 뷰 확산 모델
- **효과**: SDS 기반 방법의 일관성 및 안정성 대폭 향상
- **기여**: 다중 뷰 제약을 통한 3D 선행 제공

#### 6.3 기하학-외형 분리

**Fantasia3D (2023)**:[13][14][15]
- **혁신**: 기하학과 외형 학습의 완전한 분리
- **방법**: 
  - 기하학: 표면 법선을 확산 모델에 입력
  - 외형: BRDF 모델을 통한 재현실적 렌더링
- **성능**: 더 정확한 기하학 복원 및 사진 현실적 렌더링

#### 6.4 빠른 피드-포워드 생성

**HexaGen3D (2024)**:[16]
- **혁신**: 최적화 없는 직접 생성
- **성능**: 7초 내에 고품질 3D 자산 생성
- **방법**: 잠재 삼면체(Latent Triplane) 사용

**BoostDream (2024)**:[17]
- **혁신**: 다중 뷰 확산 모델로부터의 직접 생성 + 미세 조정
- **효과**: Janus 문제(모순적 뷰) 극복

#### 6.5 다양성 향상

**Repulsive Latent Score Distillation (2024)**:[10]
- **개선**: 입자 앙상블(Particle Ensemble) 간 유사성에 페널티 부여
- **효과**: 모드 붕괴 방지, 다양성 향상

**RewardSDS (2025)**:[10]
- **혁신**: 보상 모델을 사용한 기울도 가중치 조정
- **효과**: 사용자 선호도와의 정렬

#### 6.6 고해상도 생성

**LucidDreamer (2023)**:[18]
- **기술**: 구간 기반 점수 매칭(Interval Score Matching)
- **성능**: 과도한 평활화 방지 + 3D Gaussian Splatting 통합

***

### 7. 향후 연구 시 고려할 점

#### 7.1 기술적 고려사항

1. **3D 선행 강화**: 더 강력한 3D 기하학 선행 도입으로 역 렌더링 모호성 완화
2. **다중 해상도 최적화**: 조율 단계적 해상도 증가를 통한 세부 사항 향상
3. **메모리 효율성**: 대규모 배치 처리 및 실시간 생성을 위한 최적화
4. **다중 텍스처 표현**: 메타머 현상 제거를 위한 더 나은 재질 표현

#### 7.2 방법론적 방향

1. **하이브리드 접근**: 2D 확산 모델과 3D 생성 모델의 결합
2. **조건부 생성 강화**: 스케치, 깊이 맵 등 추가 제약 조건 통합
3. **반복적 개선**: 사용자 피드백을 통한 대화형 3D 생성
4. **크로스 모달 학습**: 텍스트뿐만 아니라 이미지, 음성 등 다중 모달 입력

#### 7.3 응용 및 확장

1. **동적 3D 생성**: 시간 변화하는 애니메이션 자산 생성
2. **제약 기반 생성**: 크기, 스타일, 재료 등의 명시적 제약
3. **3D 편집 및 조작**: 생성된 자산의 후속 수정 지원
4. **신경 렌더링 통합**: 실시간 고품질 렌더링을 위한 신경 렌더러 연계

#### 7.4 평가 지표 개선

현재 CLIP R-Precision의 한계 극복:
1. **기하학 정확도**: Chamfer Distance 등 기하학 메트릭 개선
2. **다양성 측정**: 확률 분포 메트릭 도입
3. **사용자 연구**: 정성적 평가 강화
4. **다중 평가자**: 일관성 있는 종합 평가 체계

***

## 결론

DreamFusion은 **사전학습된 2D 확산 모델을 3D 도메인으로 효과적으로 전이**하는 획기적인 방법입니다. Score Distillation Sampling이라는 새로운 손실 함수를 통해 3D 데이터 없이도 고품질 3D 자산을 생성할 수 있음을 보여주었습니다.[1]

그러나 모드 추구 문제, 해상도 제한, 역 렌더링의 근본적 모호성 등의 한계가 존재하며, DreamFlow, MVDream, Fantasia3D 등의 후속 연구들이 이러한 제한을 체계적으로 극복하고 있습니다. 향후 연구는 **더 강력한 3D 선행**, **다중 뷰 일관성**, **사용자 선호도 정렬** 등을 통해 text-to-3D 생성의 실용성과 품질을 지속적으로 향상시킬 것으로 예상됩니다.

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c5cfcdd9-e61d-4db1-8702-6bd29ed3c64c/2209.14988v1.pdf)
[2](https://arxiv.org/html/2306.07349)
[3](http://arxiv.org/pdf/2310.05375.pdf)
[4](https://arxiv.org/html/2311.01714)
[5](http://arxiv.org/pdf/2503.03664.pdf)
[6](https://arxiv.org/abs/2211.10440)
[7](https://arxiv.org/abs/2403.14966)
[8](https://pure.kaist.ac.kr/en/publications/dreamflow-high-quality-text-to-3d-generation-by-approximating-pro/)
[9](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ded44d59f5094eed0d02132fe75b60d-Paper-Conference.pdf)
[10](https://www.emergentmind.com/topics/score-distillation-sampling-sds)
[11](https://proceedings.iclr.cc/paper_files/paper/2024/file/adbe936993aa7cf41e45054d8b72f183-Paper-Conference.pdf)
[12](https://arxiv.org/abs/2308.16512)
[13](https://www.studocu.com/en-us/document/stanford-university/animals-and-performance/fantasia-3d-min/87029529)
[14](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Fantasia3D_Disentangling_Geometry_and_Appearance_for_High-quality_Text-to-3D_Content_Creation_ICCV_2023_paper.pdf)
[15](https://arxiv.org/abs/2303.13873)
[16](https://arxiv.org/pdf/2401.07727.pdf)
[17](https://arxiv.org/html/2401.16764v3)
[18](http://arxiv.org/pdf/2311.11284.pdf)
[19](https://arxiv.org/pdf/2311.05461.pdf)
[20](https://arxiv.org/abs/2303.13508)
[21](http://arxiv.org/pdf/2310.02977.pdf)
[22](https://proceedings.iclr.cc/paper_files/paper/2024/file/57568e093cbe0a222de0334b36e83cf5-Paper-Conference.pdf)
[23](https://apchenstu.github.io/mvsnerf/)
[24](https://dreamfusion3d.github.io)
[25](https://viso.ai/deep-learning/neural-radiance-fields/)
[26](https://xoft.tistory.com/39)
[27](https://www.ri.cmu.edu/app/uploads/2025/05/Yanbo_Xu_MSR_Thesis_Final.pdf)
[28](https://arxiv.org/html/2210.00379v6)
[29](https://arxiv.org/abs/2209.14988)
[30](https://ieeexplore.ieee.org/document/10925167/)
[31](https://hightechjournal.org/index.php/HIJ/article/view/640)
[32](https://aclanthology.org/2024.semeval-1.49)
[33](https://www.eurekaselect.com/235397/article)
[34](https://www.rrsurg.com/article/10.7507/1002-1892.202406002)
[35](https://photonics.pl/PLP/index.php/letters/article/view/17-20)
[36](https://arxiv.org/html/2402.02972v2)
[37](https://arxiv.org/pdf/2212.14704.pdf)
[38](https://arxiv.org/html/2408.05008)
[39](https://arxiv.org/html/2502.04370v1)
[40](https://openreview.net/forum?id=FUgrjq2pbB)
[41](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09575.pdf)
[42](https://www.semanticscholar.org/paper/DreamFlow:-High-Quality-Text-to-3D-Generation-by-Lee-Sohn/7f70b7a07a11d2931abea88c5116e31c4c0f162c)
