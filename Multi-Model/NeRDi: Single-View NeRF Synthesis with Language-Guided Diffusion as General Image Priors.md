# NeRDi: Single-View NeRF Synthesis with Language-Guided Diffusion as General Image Priors

## 종합 분석 리포트

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
NeRDi는 **단일 이미지(single-view)로부터 3D 감독(supervision) 없이** NeRF를 합성할 수 있는 프레임워크이다. 핵심 아이디어는 대규모 2D 이미지 확산 모델(diffusion model)에 내재된 **일반적 이미지 사전(general image prior)**을 활용하여, 단일 뷰 재구성을 **이미지-조건부 3D 생성 문제(image-conditioned 3D generation problem)**로 재정의하는 것이다.

### 주요 기여 (4가지)
1. **3D 감독 없는 단일 이미지 NeRF 합성 프레임워크**: 단일 뷰 재구성을 조건부 3D 생성 문제로 정식화하고, 대규모 이미지 데이터셋에서 학습된 2D 확산 모델의 사전 정보를 활용
2. **Two-section 언어 가이던스 설계**: 이미지 캡션($\mathbf{s}\_0$)과 텍스트 역변환(textual inversion, $\mathbf{s}_\*$)을 결합한 이중 섹션 의미적 가이던스로, 일반적 사전 지식을 입력 이미지에 맞게 축소(narrow down)
3. **기하학적 정규화**: 추정된 깊이 맵과 NeRF 렌더링 깊이 간의 Pearson 상관관계를 이용한 3D 불확실성 기반 기하학적 정규화 도입
4. **제로샷 일반화 검증**: DTU MVS 데이터셋에서 지도 학습 기반 베이스라인보다 높은 품질의 새로운 뷰 합성을 달성하고, in-the-wild 이미지에 대한 제로샷 NeRF 합성 능력을 입증

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

단일 2D 이미지로부터의 3D 재구성은 본질적으로 **ill-posed 문제**이다. 하나의 2D 이미지에서 3D 장면으로의 일대일 대응이 존재하지 않기 때문이다. 기존 접근법들의 한계는 다음과 같다:

| 기존 방법 | 한계점 |
|---------|--------|
| pixelNeRF, 지도 학습 기반 | 매칭된 다시점 이미지와 보정된 카메라 포즈 필요 |
| 비지도 학습 (GAN 기반) | 테스트 케이스가 훈련 분포를 따라야 하며 일반화 제한 |
| SS3D (지식 증류) | 세밀한 디테일 부족, 미지 카테고리에 대한 일반화 미흡 |

NeRDi는 **3D 감독 데이터 없이**, **특정 카테고리에 한정되지 않고**, in-the-wild 이미지에 대해 일반화 가능한 단일 뷰 NeRF 합성을 목표로 한다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) 문제 정식화

입력 이미지 $\mathbf{x}_0$가 주어졌을 때, NeRF 표현 $F\_\omega : (x, y, z) \rightarrow (\mathbf{c}, \sigma)$를 학습한다. NeRF의 볼륨 렌더링 방정식은 다음과 같다:

$$\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(t) \mathbf{c}(t) \, dt$$

여기서 투과율(transmittance)은:

$$T(t) = \exp\left(-\int_{t_n}^{t} \sigma(s) \, ds\right)$$

간결하게 $\mathbf{x} = f(\mathbf{P}, \omega)$로 표기하면, 단일 뷰 재구성을 **조건부 3D 생성 문제**로 정식화한다:

$$f(\cdot, \omega) \sim \text{3D scene distribution} \mid f(\mathbf{P}_0, \omega) = \mathbf{x}_0$$

3D 장면 분포를 직접 학습하는 대신 **2D 이미지 사전**을 활용하여 목적함수를 재정의한다:

$$\forall \mathbf{P}, \quad f(\mathbf{P}, \omega) \sim \mathbb{P} \mid f(\mathbf{P}_0, \omega) = \mathbf{x}_0$$

최종 최적화 목표는 **조건부 확률의 최대화**이다:

$$\max_\omega \; \mathbb{E}_{\mathbf{P}} \; \mathbb{P}\big(f(\mathbf{P}, \omega) \mid f(\mathbf{P}_0, \omega) = \mathbf{x}_0, \; \mathbf{s}\big)$$

여기서 $\mathbf{s}$는 의미적 가이던스(semantic guidance) 항이다.

#### (B) Novel View Distribution Loss (확산 손실)

Latent Diffusion Model (LDM)을 기반으로 한다. 사전 학습된 이미지 오토인코더의 인코더 $\mathcal{E}(\mathbf{x}) = \mathbf{z}$와 디코더 $\mathcal{D}(\mathcal{E}(\mathbf{x})) = \mathbf{x}$를 사용하며, 잠재 공간에서의 확산 목적 함수는:

$$\mathbb{E}_{\mathbf{z} \sim \mathcal{E}(\mathbf{x}), \, \mathbf{s}, \, \epsilon \sim \mathcal{N}(0,1), \, t} \left[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, t, c_\theta(\mathbf{s}))\|_2^2\right]$$

여기서:
- $t$: 확산 시간 스케일
- $\epsilon \sim \mathcal{N}(0, 1)$: 랜덤 노이즈 샘플
- $\mathbf{z}_t$: 시간 $t$까지 노이즈가 추가된 잠재 코드
- $\epsilon_\theta$: 파라미터 $\theta$를 가진 디노이징 네트워크
- $c_\theta(\mathbf{s})$: 사전 학습된 대규모 언어 모델이 조건부 텍스트 $\mathbf{s}$를 인코딩한 것

사전 학습된 확산 모델에서 $\theta$는 고정되고, 대신 NeRF 렌더링 $\mathbf{x} = f(\mathbf{P}, \omega)$를 통해 NeRF 파라미터 $\omega$에 대해 역전파(backpropagation)를 수행한다.

#### (C) Two-section 의미적 가이던스

**첫 번째 섹션 — 이미지 캡션 $\mathbf{s}_0$**:  
이미지 캡셔닝 네트워크 $\mathcal{S}$를 통해 $\mathbf{s}_0 = \mathcal{S}(\mathbf{x}_0)$를 얻는다. 전체적인 의미(semantics)를 전달하지만, 시각적 세부 정보의 모호성이 크다.

**두 번째 섹션 — 텍스트 역변환 $\mathbf{s}_*$**:  
텍스트 역변환(textual inversion)을 통해 입력 이미지의 시각적 단서를 포착하는 텍스트 임베딩을 최적화한다:

$$\mathbf{s}_* = \arg\min_{\mathbf{s}} \mathbb{E}_{\mathbf{z} \sim \mathcal{E}(\mathbf{x}_0), \, \mathbf{s}, \, \epsilon \sim \mathcal{N}(0,1), \, t} \left[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, t, c_\theta(\mathbf{s}))\|_2^2\right]$$

**결합 특징**: 두 임베딩을 연결하여 공동 특징 $\mathbf{s} = [\mathbf{s}\_0, \mathbf{s}_*]$를 형성하고, 이를 확산 과정의 가이던스로 사용한다. 이를 통해 의미적 일관성과 시각적 일관성을 동시에 달성한다.

#### (D) 기하학적 정규화

Dense Prediction Transformer (DPT) 모델을 사용하여 입력 이미지 $\mathbf{x}\_0$로부터 깊이 맵 $\mathbf{d}_{0,\text{est}}$를 추정한다. NeRF에서 렌더링된 깊이는:

$$\hat{\mathbf{d}}_0 = \int_{t_n}^{t_f} \sigma(t) \, dt$$

추정 깊이의 스케일/시프트 모호성과 오차로 인해 직접적인 L2 손실 대신 **Pearson 상관관계**를 최대화한다:

$$\rho\left(\hat{\mathbf{d}}_0, \mathbf{d}_{0,\text{est}}\right) = \frac{\text{Cov}(\hat{\mathbf{d}}_0, \mathbf{d}_{0,\text{est}})}{\sqrt{\text{Var}(\hat{\mathbf{d}}_0) \, \text{Var}(\mathbf{d}_{0,\text{est}})}}$$

이는 스케일과 시프트에 불변(invariant)한 상대적 깊이 일관성을 보장한다.

### 2.3 모델 구조

전체 구조는 세 가지 손실의 조합으로 NeRF 파라미터 $\omega$를 최적화한다:

```
입력 이미지 x₀
    ├── [입력 뷰] 재구성 손실: ||f(P₀, ω) - x₀||²
    ├── [임의 뷰] 확산 손실: LDM 기반 이미지 분포 손실 (+ two-section 가이던스)
    └── [입력 뷰] 깊이 상관 손실: ρ(d̂₀, d₀,est)
```

- **NeRF 모델**: Multi-resolution grid sampler (Instant-NGP 기반)
- **확산 모델**: Stable Diffusion (LDM), LAION-400M 사전 학습, 512×512 해상도
- **언어 모델**: GPT-2 (캡셔닝), CLIP 텍스트 인코더 (텍스트 역변환)
- **깊이 추정**: DPT (Dense Prediction Transformer), 1.4M 이미지 학습
- **배경 제거**: 이분 이미지 분할(dichotomous image segmentation) 네트워크

**Lambertian NeRF** (뷰 방향 입력 제거)를 사용하여 더 강한 다시점 일관성을 강제한다.

**렌더링 해상도**: 임의 뷰에서 128×128로 렌더링 후 512×512로 리사이즈하여 LDM 인코더에 입력.

### 2.4 성능 향상

#### DTU MVS 데이터셋 정량적 결과

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| NeRF | 8.000 | 0.286 | 0.703 |
| pixelNeRF | 15.550 | 0.537 | 0.535 |
| pixelNeRF, $\mathcal{L}_{\text{MSE}}$ ft | **16.048** | **0.564** | 0.515 |
| DietPixelNeRF | 14.242 | 0.481 | 0.487 |
| **NeRDi (Ours)** | 14.472 | 0.465 | **0.421** |

**핵심 관찰**:
- PSNR/SSIM은 pixelNeRF보다 약간 낮지만, 이 메트릭들은 **픽셀 정렬 유사도** 기반이므로 단일 뷰 3D 추론의 본질적 불확실성 하에서는 덜 적절함
- **LPIPS (지각적 유사도)에서 유의미한 개선** (0.421 vs. 베이스라인 최고 0.487): 확산 모델이 이미지 품질을 개선하고 언어 가이던스가 다시점 의미적 일관성을 유지함
- NeRDi는 **DTU 훈련 세트에 대해 미세조정 없이(zero-shot)** 적용되었음에도 경쟁력 있는 성능

#### 정성적 결과
- DTU: pixelNeRF 초기화에서 시작하여 노이즈와 블러를 제거하고 사실적인 기하학과 외관의 새로운 뷰 합성
- In-the-wild: Google Scanned Objects, COCO 데이터셋, 인터넷 이미지 등 다양한 도메인에서 시각적으로 우수한 결과
- DietNeRF 대비: 글로벌 의미 일관성 우수, SS3D 대비: 세밀한 기하학적 디테일 보존

### 2.5 한계

1. **사전 학습 모델의 편향(bias) 전파**: 확산 모델, GPT-2, DPT 등 여러 사전 학습 모델에 의존하므로, 이들의 편향이 합성 결과에 영향 (예: "a single shoe" 프롬프트에도 두 개의 신발 생성)
2. **높은 변형성(deformable) 객체에 취약**: 언어 가이던스가 의미와 스타일에 집중하나, 물리적 상태/동적 특성의 글로벌 설명 부족 (예: 고양이가 두 개의 머리와 꼬리로 합성)
3. **복잡한 장면 구성에 한계**: 모든 뷰에서 동일한 의미를 가진다는 가정이 객체 중심 이미지에는 적합하나, 대규모 장면에서는 뷰 변화와 가림 현상으로 인해 부적합
4. **텍스트 임베딩 표현력 제한**: 텍스트 역변환의 임베딩이 단일 단어 차원이므로 복잡한 콘텐츠의 미묘한 차이를 표현하기 어려움
5. **계산 비용**: NeRF의 볼류메트릭 렌더링과 확산 모델 역전파의 조합으로 높은 연산량
6. **해상도 제한**: 128×128 렌더링 후 512×512로 업스케일하므로 세밀한 디테일 손실 가능

---

## 3. 모델의 일반화 성능 향상 가능성

NeRDi의 일반화 성능은 이 논문의 **가장 핵심적인 차별점**이며, 여러 차원에서 분석할 수 있다.

### 3.1 현재 일반화 능력의 원천

**(1) 3D 데이터셋 독립성**: 기존 방법(pixelNeRF, 3D-aware GAN 등)은 특정 3D 데이터셋에서 장면 사전을 학습하므로 훈련 분포 외 카테고리에 일반화가 어려움. NeRDi는 LAION-400M (4억 개의 이미지-텍스트 쌍)에서 학습된 2D 확산 모델을 사전으로 사용하므로, **카테고리 불가지론적(category-agnostic)** 일반화가 가능하다.

**(2) Two-section 가이던스의 적응성**: 
- 캡션 $\mathbf{s}_0$: 임의 이미지에 대해 자동 생성 가능 → 의미적 일반화
- 텍스트 역변환 $\mathbf{s}_*$: 각 입력 이미지에 대해 최적화 → 인스턴스 수준 적응

이 조합은 "일반적 사전"에서 "입력 이미지 특화 사전"으로의 효과적인 축소를 가능케 한다.

**(3) 제로샷 평가 결과**: DTU 훈련 세트를 사용하지 않고도, DTU에서 훈련된 방법들(pixelNeRF)과 경쟁력 있는 LPIPS 달성.

### 3.2 일반화 성능 향상을 위한 가능성

**(1) 더 강력한 확산 모델 활용**:
- 논문 발표 시점(2022.12) 이후 SDXL, Stable Diffusion 3, DALL-E 3 등 더 강력한 확산 모델이 등장. 이러한 모델로의 교체는 이미지 사전의 품질과 다양성을 직접적으로 향상시킬 수 있다.
- 특히 **3D 인식 확산 모델**(예: Zero-1-to-3, MVDream)의 활용은 다시점 일관성을 크게 개선할 잠재력이 있다.

**(2) 향상된 언어 가이던스**:
- GPT-2 대신 GPT-4V, LLaVA 등의 **멀티모달 대형 언어 모델(MLLM)**을 사용하면 더 정확하고 상세한 캡션 생성이 가능
- 텍스트 역변환의 단일 단어 차원 한계를 극복하기 위해 **다중 토큰 텍스트 역변환** 또는 **IP-Adapter** 같은 이미지 조건부 기법 활용 가능

**(3) 기하학적 사전 강화**:
- DPT 외에 Metric3D, ZoeDepth 등 최신 메트릭 깊이 추정 모델 활용
- 법선 맵(normal map) 추정 병합으로 표면 기하학 정규화 강화
- 3D 포인트 클라우드나 메시 사전의 통합

**(4) 스케일 확장**:
- 더 높은 해상도의 렌더링(128→256 이상)
- 3D Gaussian Splatting 등 더 효율적인 3D 표현으로의 전환

**(5) 복잡한 장면으로의 확장**:
- 장면 분해(scene decomposition) 기법과의 결합
- 뷰 의존적 조건부 생성으로 대규모 장면의 뷰 변화와 가림 처리

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 연구 패러다임의 전환

NeRDi는 **"2D 사전을 3D 생성에 활용"**하는 패러다임을 체계적으로 제시한 초기 연구 중 하나로, 이후 폭발적으로 성장한 **Score Distillation Sampling (SDS) 기반 3D 생성** 연구의 토대를 마련했다. 특히:

1. **텍스트-3D에서 이미지-3D로의 확장**: DreamFusion이 텍스트→3D를 보여주었다면, NeRDi는 이미지→3D로의 확산 모델 활용을 체계화
2. **조건부 생성의 중요성 부각**: 단순 텍스트 프롬프트 대신 입력 이미지에 충실한 3D 생성을 위한 조건부 설계의 필요성 강조
3. **다중 모달 사전의 통합**: 언어, 시각, 기하학 사전의 체계적 결합 방법론 제시

### 4.2 향후 연구 시 고려 사항

**(1) 다시점 일관성 보장**:
- 2D 확산 모델은 각 뷰를 독립적으로 처리하므로, 다시점 일관성이 본질적으로 보장되지 않음
- 3D 인식 확산 모델 또는 에피폴라 어텐션 등의 기법 필요

**(2) 속도와 효율성**:
- NeRF 최적화 + 확산 모델 역전파는 매우 시간이 많이 소요됨
- 피드포워드 모델(예: LRM, Instant3D)과의 속도-품질 트레이드오프 고려

**(3) 평가 메트릭의 재정의**:
- 단일 뷰 3D 추론의 본질적 모호성 하에서 PSNR/SSIM의 한계가 명확히 드러남
- 지각적 메트릭(LPIPS), FID, 사용자 연구 등 다양한 평가 체계 필요

**(4) 편향과 윤리적 고려**:
- 대규모 확산 모델의 편향이 3D 합성에 직접 전파되므로, 편향 감사(bias audit) 필요

**(5) 실용적 응용 시나리오**:
- 로봇공학(단일 이미지로 물체 3D 파악), AR/VR, 자율주행(Waymo에서의 연구 맥락) 등 실제 응용에서의 강건성 검증 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 표는 NeRDi와 관련된 2020년 이후 주요 연구들을 비교 분석한 것이다:

| 연구 | 연도 | 핵심 기법 | 3D 표현 | 입력 | 3D 감독 | 일반화 | NeRDi와의 관계 |
|------|------|---------|--------|------|--------|------|-------------|
| **NeRF** (Mildenhall et al.) | 2020 | 볼륨 렌더링 + MLP | NeRF | 다시점 | 필요 | 장면 특화 | 기반 3D 표현 |
| **pixelNeRF** (Yu et al.) | 2021 | 이미지 특징 조건부 NeRF | NeRF | 소수 뷰 | 필요 | 훈련 분포 내 | 베이스라인/초기화 |
| **DietNeRF** (Jain et al.) | 2021 | CLIP 특징 일관성 | NeRF | 소수 뷰 | 불필요 | 제한적 | 베이스라인, CLIP 사용 |
| **DreamFusion** (Poole et al.) | 2022 | SDS + Imagen | NeRF | 텍스트 | 불필요 | 높음 | 직접적 영감, 텍스트→이미지 확장 |
| **Zero-1-to-3** (Liu et al.) | 2023 | 뷰 조건부 확산 | NeRF/Mesh | 단일 이미지 | 불필요 | 높음 | 확산 모델에 뷰 조건부 추가 |
| **Magic123** (Qian et al.) | 2023 | 2D+3D 사전 결합 | NeRF→Mesh | 단일 이미지 | 불필요 | 높음 | NeRDi의 2D 사전 + 3D 사전 통합 |
| **One-2-3-45** (Liu et al.) | 2023 | Zero-1-to-3 + SparseNeuS | Mesh | 단일 이미지 | 불필요 | 높음 | 다시점 생성 후 3D 재구성 파이프라인 |
| **SyncDreamer** (Liu et al.) | 2023 | 동기화된 다시점 확산 | NeRF/Mesh | 단일 이미지 | 불필요 | 높음 | 다시점 일관성 확산 모델 |
| **MVDream** (Shi et al.) | 2023 | 다시점 확산 모델 | NeRF | 텍스트 | 불필요 | 높음 | 3D 인식 확산으로 일관성 개선 |
| **Wonder3D** (Long et al.) | 2023 | 크로스 도메인 어텐션 | Mesh | 단일 이미지 | 불필요 | 높음 | 법선 맵 동시 생성 |
| **LRM** (Hong et al.) | 2023 | 대규모 트랜스포머 | TriPlane NeRF | 단일 이미지 | 필요 | 높음 (대규모 데이터) | 피드포워드 방식으로 속도 극복 |
| **DreamGaussian** (Tang et al.) | 2023 | SDS + 3D Gaussian | 3DGS | 단일 이미지/텍스트 | 불필요 | 높음 | 효율적 3D 표현으로 전환 |
| **Instant3D** (Li et al.) | 2023 | 피드포워드 다시점 생성 | TriPlane | 단일 이미지 | 필요 | 높음 | 최적화 없는 빠른 추론 |
| **SV3D** (Voleti et al.) | 2024 | 비디오 확산 기반 | 3DGS/NeRF | 단일 이미지 | 불필요 | 높음 | 비디오 확산으로 궤도 뷰 생성 |

### 주요 비교 분석

#### (1) NeRDi vs. DreamFusion
- **공통점**: 2D 확산 모델을 사전으로 활용, NeRF 파라미터 최적화
- **차이점**: DreamFusion은 텍스트→3D이며 사용자 지정 프롬프트 사용; NeRDi는 이미지→3D이며 입력 이미지에 충실한 3D 생성을 위한 **two-section 가이던스**와 **입력 뷰 제약** 도입
- DreamFusion의 SDS(Score Distillation Sampling)와 NeRDi의 확산 손실은 본질적으로 유사하나, NeRDi는 조건부 설계에서 차별화

#### (2) NeRDi vs. Zero-1-to-3 (2023)
- Zero-1-to-3는 확산 모델 자체를 **상대적 카메라 포즈 조건부 이미지 생성**으로 미세조정하여, 특정 뷰에서의 이미지를 직접 생성
- NeRDi는 사전 학습된 확산 모델을 **고정**하고 언어 가이던스로 조건화하는 반면, Zero-1-to-3는 **3D 인식 미세조정** 수행
- Zero-1-to-3의 접근이 더 직접적이지만 미세조정 데이터(Objaverse)에 의존

#### (3) NeRDi vs. LRM 계열 (2023~2024)
- LRM은 대규모 3D 데이터(Objaverse 등)로 **피드포워드 트랜스포머**를 훈련하여 단일 이미지로부터 ~5초 내 3D 재구성
- NeRDi는 **테스트 시 최적화** 기반이므로 느리지만, 3D 훈련 데이터 불필요
- 일반화 측면에서 LRM은 훈련 데이터 범위에, NeRDi는 확산 모델의 2D 사전 범위에 의존

#### (4) NeRDi vs. 3D Gaussian Splatting 기반 방법 (2023~2024)
- DreamGaussian, GaussianDreamer 등은 3DGS를 3D 표현으로 사용하여 NeRF 대비 훈련/렌더링 속도를 크게 개선
- NeRDi의 프레임워크는 3DGS로도 확장 가능하며, 3D 표현의 발전에 따라 자연스럽게 개선 가능

#### (5) 최근 트렌드와의 관계
- **다시점 확산 모델** (MVDream, SyncDreamer, SV3D): NeRDi의 2D 확산 사전의 다시점 불일관성 문제를 직접 해결
- **대규모 3D 재구성 모델** (LRM, Instant3D): 최적화 기반 접근의 속도 한계를 피드포워드로 극복
- **비디오 확산** (SV3D): 시간적 일관성을 공간적 일관성으로 전환하는 새로운 접근

---

## 참고자료

1. Deng, C., Jiang, C. M., Qi, C. R., Yan, X., Zhou, Y., Guibas, L., & Anguelov, D. (2022). "NeRDi: Single-View NeRF Synthesis with Language-Guided Diffusion as General Image Priors." *arXiv preprint arXiv:2212.03267*.
2. Poole, B., Jain, A., Barron, J. T., & Mildenhall, B. (2022). "DreamFusion: Text-to-3D using 2D Diffusion." *arXiv preprint arXiv:2209.14988*.
3. Mildenhall, B. et al. (2021). "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *Communications of the ACM*, 65(1), 99–106.
4. Yu, A., Ye, V., Tancik, M., & Kanazawa, A. (2021). "pixelNeRF: Neural Radiance Fields from One or Few Images." *CVPR 2021*.
5. Jain, A., Tancik, M., & Abbeel, P. (2021). "Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis." *ICCV 2021*.
6. Rombach, R. et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.
7. Gal, R. et al. (2022). "An Image is Worth One Word: Personalizing Text-to-Image Generation Using Textual Inversion." *arXiv preprint arXiv:2208.01618*.
8. Ranftl, R. et al. (2021). "Vision Transformers for Dense Prediction." *ICCV 2021*.
9. Liu, R. et al. (2023). "Zero-1-to-3: Zero-shot One Image to 3D Object." *ICCV 2023*.
10. Qian, G. et al. (2023). "Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors." *arXiv preprint arXiv:2306.17843*.
11. Hong, Y. et al. (2023). "LRM: Large Reconstruction Model for Single Image to 3D." *arXiv preprint arXiv:2311.04400*.
12. Tang, J. et al. (2023). "DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation." *arXiv preprint arXiv:2309.16653*.
13. Shi, Y. et al. (2023). "MVDream: Multi-view Diffusion for 3D Generation." *arXiv preprint arXiv:2308.16512*.
14. Liu, Y. et al. (2023). "SyncDreamer: Generating Multiview-consistent Images from a Single-view Image." *arXiv preprint arXiv:2309.03453*.
15. Voleti, V. et al. (2024). "SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion." *arXiv preprint arXiv:2403.12008*.
16. Long, X. et al. (2023). "Wonder3D: Single Image to 3D using Cross-Domain Diffusion." *arXiv preprint arXiv:2310.15008*.
17. Alwala, K. V., Gupta, A., & Tulsiani, S. (2022). "Pre-train, Self-train, Distill: A Simple Recipe for Supersizing 3D Reconstruction." *CVPR 2022*.
