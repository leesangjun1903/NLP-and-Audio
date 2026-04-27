
# CAT3D: Create Anything in 3D with Multi-View Diffusion Models

> **논문 정보**
> - **제목**: CAT3D: Create Anything in 3D with Multi-View Diffusion Models
> - **저자**: Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T. Barron, Ben Poole
> - **소속**: Google DeepMind, Google Research
> - **게재**: NeurIPS 2024
> - **arXiv**: [2405.10314](https://arxiv.org/abs/2405.10314) (2024년 5월 16일)
> - **프로젝트 페이지**: [cat3d.github.io](https://cat3d.github.io/)

---

## 1. 핵심 주장과 주요 기여 (Executive Summary)

### 핵심 주장

기존 3D 재구성 기술은 고품질 3D 장면 생성을 위해 사용자가 수백~수천 장의 이미지를 수집해야 했다. CAT3D는 이 실세계 촬영 과정을 **멀티뷰 디퓨전 모델(multi-view diffusion model)**로 시뮬레이션하여 임의 수의 입력 이미지와 목표 시점만으로 고도로 일관된 새로운 뷰(novel views)를 생성하고, 이를 robust한 3D 재구성 파이프라인에 입력해 임의 시점에서 실시간 렌더링 가능한 3D 표현을 생성한다. CAT3D는 1분 이내에 전체 3D 장면을 생성하며 단일 이미지 및 소수 뷰 3D 생성에서 기존 방법을 능가한다.

### 주요 기여 요약

| 기여 항목 | 내용 |
|-----------|------|
| **멀티뷰 잠재 디퓨전 모델** | 임의 수의 입력 뷰로 조건화된 일관된 다중 뷰 생성 |
| **2단계 파이프라인** | 뷰 생성 → 3D 재구성의 명확한 분리 |
| **속도** | 기존 1시간 → 1분 이내로 단축 |
| **일반화** | 인도메인·아웃오브도메인 모두 SOTA 달성 |

---

## 2. 해결하고자 하는 문제

### 2.1 핵심 문제

고품질 3D 콘텐츠 생성은 수백~수천 장의 이미지를 요구하는 노동집약적인 촬영 과정을 필요로 하며 접근성이 낮다. 단일 이미지나 텍스트처럼 제한된 관측만으로 동작하는 기존 3D 생성 솔루션들은 품질, 효율성, 범용성 면에서 한계를 보인다. CAT3D의 핵심 목표는 일관된 새로운 뷰를 생성함으로써 제한된 관측 문제를 해결하고, 풀기 어려운 재구성 문제를 풀 수 있는 생성 문제로 변환하는 것이다.

저자들은 전통적인 3D 재구성 기술이 고충실도 3D 장면을 위해 방대한 촬영 데이터를 요구함을 인식하고, CAT3D는 재구성 문제를 **생성 모델링 작업(generative modeling task)**으로 재정식화하여 제한된 조건부 뷰에서 일관된 새로운 뷰를 생성하는 것을 목표로 삼는다.

---

## 3. 제안하는 방법 (수식 포함)

### 3.1 전체 파이프라인 (2단계)

CAT3D는 두 단계로 구성된다:
1. 입력 뷰와 목표 뷰의 카메라 포즈에 조건화된 멀티뷰 잠재 디퓨전 모델에서 대규모 합성 뷰 집합 생성
2. 관측 뷰 및 생성 뷰에 robust한 3D 재구성 파이프라인을 적용해 NeRF 표현 학습

### 3.2 멀티뷰 잠재 디퓨전 모델

#### 모델 조건화 구조

이 모델은 임의 수의 관측 뷰(입력 이미지와 ray 좌표로 임베딩된 카메라 포즈)로 조건화되며, 여러 일관된 목표 시점을 생성하도록 훈련된다. 이 아키텍처는 비디오 디퓨전 모델과 유사하지만, 시간 임베딩 대신 각 이미지마다 카메라 포즈 임베딩을 사용한다.

#### 핵심 디퓨전 수식

멀티뷰 잠재 디퓨전 모델은 아래 **DDPM(Denoising Diffusion Probabilistic Model)** 기반 목적함수를 따른다.

**Forward Process (가우시안 노이즈 점진적 주입):**

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1-\bar{\alpha}_t)\mathbf{I}\right)$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t} \alpha_s$이며, 노이즈 스케줄에 따라 정의된다.

**Reverse Process (노이즈 제거 학습):**

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0,\, t,\, \boldsymbol{\epsilon}}\!\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\!\left(\mathbf{x}_t,\, t,\, \mathcal{C}\right)\right\|^2\right]$$

여기서:
- $\mathbf{x}_0$: 목표 뷰의 잠재 표현(latent)
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: 주입된 노이즈
- $\boldsymbol{\epsilon}_\theta$: 학습 가능한 denoising 네트워크
- $\mathcal{C}$: 조건부 맥락(입력 뷰 이미지 + 카메라 포즈 ray 맵)

**카메라 포즈 임베딩 (Plucker ray coordinate):**

각 픽셀의 카메라 포즈는 Plucker 좌표계로 표현된다:

$$\mathbf{r} = (\mathbf{d},\; \mathbf{o} \times \mathbf{d})$$

여기서:
- $\mathbf{d} \in \mathbb{R}^3$: 광선(ray) 방향 벡터
- $\mathbf{o} \in \mathbb{R}^3$: 카메라 원점
- $\mathbf{o} \times \mathbf{d}$: 두 벡터의 외적

이 6차원 ray 맵을 각 이미지 잠재와 채널 방향으로 concatenate하여 모델에 입력한다.

**멀티뷰 잠재 인코딩 (VAE):**

모델은 각 조건부 뷰를 이미지와 해당 카메라 포즈로 구성된 입력으로 받으며, 각 입력 이미지는 이미지 변분 오토인코더(Image VAE)를 통해 잠재 표현으로 인코딩된다.

$$\mathbf{z}_i = \text{Enc}_\phi(\mathbf{I}_i), \quad \mathbf{z}_i \in \mathbb{R}^{H' \times W' \times C}$$

**3D Self-Attention을 통한 멀티뷰 일관성:**

전반적으로, 3D 자기-어텐션(spatiotemporal)과 카메라 포즈의 raymap 임베딩을 갖춘 비디오 디퓨전 아키텍처가, robust 재구성 손실과 결합될 때 3D 표현을 복원하기에 충분한 일관성 있는 뷰를 생성함을 발견했다.

모든 $N$개 프레임의 잠재를 joint하게 처리:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

여기서 $Q, K, V$는 모든 뷰(조건부 + 목표)의 잠재를 flatten하여 구성한다.

### 3.3 대규모 뷰 생성 전략 (Anchored Sampling)

CAT3D는 목표 시점을 클러스터링하고 autoregressive 샘플링 방법을 활용하여 장거리 및 로컬 일관성을 모두 보장하는 대규모 합성 뷰 생성 전략을 도입한다.

구체적으로, 먼저 소수의 **앵커 뷰(anchor views)**를 생성하여 장거리 일관성을 확보한 후, 나머지 뷰를 앵커와 함께 조건화하여 로컬 일관성을 보장한다.

### 3.4 Robust 3D 재구성 파이프라인

CAT3D의 3D 재구성 파이프라인은 Zip-NeRF를 기반으로 하며, 생성 뷰의 불일치에 대한 강건성을 높이기 위해 **지각 손실(perceptual loss, LPIPS)**과 **거리 기반 가중치 스케줄(distance-based weighting schedule)**을 도입한다.

재구성 손실 함수:

$$\mathcal{L}_{\text{recon}} = \lambda_1 \mathcal{L}_{\text{MSE}} + \lambda_2 \mathcal{L}_{\text{LPIPS}} + \lambda_3 w(d) \cdot \mathcal{L}_{\text{gen}}$$

여기서:
- $\mathcal{L}_{\text{MSE}}$: 관측 뷰에 대한 픽셀 단위 복원 손실
- $\mathcal{L}_{\text{LPIPS}}$: 지각적 유사도 손실
- $w(d)$: 생성 뷰의 거리 기반 가중치 함수 (불일치에 robust)

최종 3D 표현은 용도에 따라 **Zip-NeRF** 또는 **3D Gaussian Splatting(3DGS)** 중 선택 가능하다.

생성된 뷰들은 robust한 3D 재구성 파이프라인에 입력되어 Zip-NeRF 또는 3DGS 기반 3D 표현을 생성한다.

---

## 4. 모델 구조

CAT3D의 핵심은 일관된 새로운 뷰를 생성하도록 훈련된 멀티뷰 디퓨전 모델이다. 입력 뷰와 목표 뷰의 카메라 포즈에 조건화되어 대규모 합성 뷰를 생성한 후, robust한 3D 재구성 파이프라인이 NeRF 표현을 학습한다.

```
입력 이미지(들) + 카메라 포즈
        │
        ▼
[Image VAE Encoder] → 잠재 표현 z_i
        │
        ▼
[Plucker Ray 임베딩] → 카메라 포즈 조건화
        │
        ▼
[멀티뷰 잠재 디퓨전 U-Net]
  ├── 2D Self-Attention (공간 내)
  ├── 3D Self-Attention (모든 뷰 간)
  └── Cross-Attention (조건부 입력)
        │
        ▼
[앵커 샘플링 전략] → 대규모 일관된 뷰 집합
        │
        ▼
[Robust 3D 재구성]
  ├── Zip-NeRF (고품질)
  └── 3DGS (실시간 렌더링)
        │
        ▼
임의 시점 실시간 렌더링
```

CAT3D는 ImageDream과 유사한 아키텍처를 사용하며, 멀티뷰 의존성은 3D 자기-어텐션을 갖춘 비디오 디퓨전 모델과 유사한 구조로 포착된다.

생성 Prior와 3D 재구성 과정을 분리(decoupling)함으로써, 이전 연구들에 비해 계산 효율성이 향상되고 방법론적 복잡도가 줄어들면서도 이미지 품질이 개선된다.

---

## 5. 성능 향상

### 5.1 정량적 성능

CAT3D는 CO3D, RealEstate10K 같은 인도메인 데이터셋과 DTU, LLFF, mip-NeRF 360 같은 아웃오브도메인 데이터셋을 포함한 다양한 실세계 벤치마크에서 소수 뷰 3D 재구성 분야의 SOTA를 달성하며, 뛰어난 PSNR, SSIM, LPIPS 지표를 기록한다. 이는 강력한 일반화 능력과 다양한 장면 복잡도 및 데이터 분포 처리 능력을 보여준다.

CAT3D는 거의 모든 설정에서 SOTA 성능을 달성하면서 생성 시간을 ZeroNVS와 ReconFusion 대비 1시간에서 수 분으로 단축한다. 특히 CO3D와 mip-NeRF 360 같은 도전적 데이터셋에서 더 큰 격차로 기존 방법을 능가하여, 크고 상세한 장면 재구성에서의 가치를 입증한다.

CAT3D의 성능은 입력 뷰 수에 비례하여 효과적으로 향상되며, 입력 뷰가 3개에서 9개로 늘어날수록 PSNR, SSIM, LPIPS에서 일관된 개선을 보인다. 이는 소수 뷰 환경에서도 효과적이지만, 추가적인 입력 관측으로 재구성 품질이 더욱 향상됨을 시사한다.

| 지표 | CAT3D vs. ReconFusion | CAT3D vs. ZeroNVS |
|------|----------------------|-------------------|
| PSNR | ↑ 향상 | ↑ 향상 |
| SSIM | ↑ 향상 | ↑ 향상 |
| LPIPS | ↑ 향상 | ↑ 향상 |
| 생성 시간 | ~60분 → 수 분 | ~60분 → 수 분 |

### 5.2 정성적 성능

관측되지 않은 영역에서 CAT3D는 기하학과 외관을 입력 뷰에서 보존하면서도 그럴듯한 텍스처 콘텐츠를 hallucinate할 수 있으며, 이전 연구들이 종종 보이던 흐릿한 디테일과 과도하게 부드러운 배경을 방지한다.

---

## 6. 한계점

저자들은 다음과 같은 한계를 인정한다: 여러 다른 카메라 내부 매개변수(camera intrinsics)를 효과적으로 처리하지 못하는 점, 대규모 환경에서 일관된 뷰 생성의 잠재적 어려움, 수동으로 구성된 카메라 궤적에 대한 의존성.

이러한 멀티뷰 디퓨전 모델들은 일반적으로 제한된 참조 이미지로 소규모 공간 영역에 국한되어 복잡한 장면에서 hallucination에 취약하다. 또한 출력물이 엄격한 3D 일관성을 보장하지 않아 재구성 파이프라인과의 직접 통합에 어려움이 있다.

ZeroNVS, Reconfusion, CAT3D 같은 최근 방법들은 현실적 외삽을 위한 3D 뷰 조건화를 도입했지만, 정확한 카메라 포즈에 의존하며 포즈 없이(pose-free) 적용하기 위한 자명한 확장이 불가능하다.

---

## 7. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 7.1 현재의 일반화 성능

저자들은 훈련 데이터셋의 held-out 검증 세트(인도메인 샘플)와 mip-NeRF 360 데이터셋(아웃오브도메인 샘플) 모두에서 새로운 뷰 합성 및 3뷰 재구성을 평가하여 일반화를 검증했다.

이전 연구의 PixelNeRF 방식을 attention 기반 조건화로 대체하고, 조건부 비디오 디퓨전 아키텍처 내에서 이미지당 카메라 포즈 임베딩을 적용함으로써 샘플 품질과 3D 재구성이 향상됨을 확인했다.

### 7.2 일반화 향상을 위한 설계 선택들

저자들은 여러 모델 변형을 고려하고, 인도메인 및 아웃오브도메인 데이터셋 모두에서 샘플 품질과 소수 뷰 3D 재구성 성능을 평가했다.

다음 설계 결정들이 일반화에 특히 중요하다:

1. **Plucker ray 좌표로 카메라 포즈 표현**: 절대 좌표계 대신 상대적 ray 표현을 사용함으로써 다양한 카메라 설정에 대한 범용성 확보

2. **비디오 디퓨전 기반 아키텍처 채택**: 대규모 비디오 사전 지식(prior)을 활용해 다양한 장면 유형에서 일반화

3. **생성과 재구성의 분리(Decoupling)**:

이 연구는 일관된 새로운 뷰 합성에서 멀티뷰 디퓨전 모델의 잠재력을 보여주며 3D 장면 재구성의 경계를 확장한다. 생성 뷰의 불일치를 처리하기 위해 3D 재구성 기법을 정제함으로써, 이러한 방법들을 더욱 일반적으로 적용 가능하고 robust하게 만드는 데 기여한다.

### 7.3 미래 일반화 향상 가능성

CAT3D는 동물 같은 관절 있는 캐릭터를 포함한 다양한 객체, 장면의 3D 모델을 생성할 수 있다. 이러한 유연성은 이전 3D 생성 방법보다 더 확장 가능하고 효율적인 멀티뷰 디퓨전 접근법에 의해 가능해진다.

향후 모델들은 생성 뷰 간 불일치를 더욱 줄여 재구성 과정을 더욱 robust하게 만들 수 있다.

---

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 관련 연구 계보

```
NeRF (2020, Mildenhall et al.)
        │
        ▼
DreamFusion (2022, Poole et al.)       Zero-1-to-3 (2023)
[Text → 3D via SDS]                    [이미지 → 다중뷰]
        │                                    │
        ▼                                    ▼
ReconFusion (2023)                     MVDream / ImageDream (2023)
[Diffusion prior + NeRF]               [멀티뷰 동시 생성]
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
               CAT3D (2024, NeurIPS)
        [2단계: 뷰 생성 → 3D 재구성]
                       │
                       ▼
              CAT4D (2024)
        [4D = 3D + 시간축 확장]
```

### 8.2 주요 방법론 비교표

여러 뷰 간 상관관계를 모델링하면 부분적 관측과 일관된 3D 콘텐츠에 대한 훨씬 강력한 사전 지식을 제공한다. MVDream, ImageDream, Zero123++, ConsistNet, SyncDreamer, ViewDiff 등은 텍스트-이미지 모델을 파인튜닝하여 여러 뷰를 동시에 생성한다.

| 방법 | 입력 | 생성 방식 | 3D 표현 | 속도 | 일반화 |
|------|------|-----------|---------|------|--------|
| **NeRF** (2020) | 다수 이미지 | 최적화 | NeRF | 느림 | 씬 특화 |
| **DreamFusion** (2022) | 텍스트 | SDS 최적화 | NeRF | 매우 느림 | 텍스트 범위 |
| **Zero-1-to-3** (2023) | 1장 이미지 | 단일뷰 디퓨전 | - | 빠름 | 제한적 |
| **MVDream** (2023) | 텍스트 | 멀티뷰 동시 | 3DGS | 중간 | 중간 |
| **ReconFusion** (2023) | 3~9장 | 디퓨전 prior + NeRF | NeRF | ~1시간 | 좋음 |
| **ZeroNVS** (2023) | 1장 | 360° 디퓨전 | NeRF | ~1시간 | 중간 |
| **CAT3D** (2024) | 1~N장 | 멀티뷰 잠재 디퓨전 | Zip-NeRF/3DGS | ~1분 | **매우 좋음** |
| **CAT4D** (2024) | 동영상 | 멀티뷰 비디오 디퓨전 | 4DGS | ~수분 | 동적 장면 |

CAT3D는 단일 이미지 또는 텍스트 같은 제한된 입력으로부터 생성과 재구성을 분리함으로써 효율적으로 3D 장면과 객체를 생성한다. 3D 자기-어텐션을 갖춘 멀티뷰 디퓨전 모델을 사용해 일관된 새로운 뷰를 합성하고, 이를 robust한 3D 재구성 파이프라인으로 처리하여 3D 콘텐츠 생성을 더욱 접근 가능하게 만든다.

### 8.3 CAT3D의 영향: CAT4D

CAT4D는 CAT3D의 디퓨전 모델 위에 구축된 멀티뷰 비디오 디퓨전 모델로, 모든 이미지 잠재를 연결하는 3D 자기-어텐션을 적용한다. CAT3D와 동일한 아키텍처를 채택하면서 추가적인 시간 조건화를 주입한다.

---

## 9. 향후 연구에 미치는 영향과 고려 사항

### 9.1 연구에 미치는 영향

#### ① 3D 생성 패러다임의 전환
CAT3D의 멀티뷰 디퓨전 접근법과 안정적인 3D 생성 기법 같은 핵심 혁신들은 더 접근 가능하고 다용도적인 3D 콘텐츠 생성 도구를 향한 중요한 발걸음이다. CAT3D는 다양한 객체, 장면, 캐릭터의 고품질 3D 모델을 생성할 수 있는 새로운 AI 시스템을 도입한다. 핵심 혁신은 여러 2D 뷰에서 정보를 결합해 3D 콘텐츠를 생성하는 확장 가능하고 유연한 멀티뷰 디퓨전 모델로, 이전 단일 뷰 접근법의 한계를 극복한다.

#### ② 실용적 응용 확대
3D 콘텐츠 생성을 더 접근 가능하게 함으로써 CAT3D는 게임, 가상 현실, 제품 디자인 등의 분야에서 3D 모델링을 민주화할 잠재력을 가진다.

#### ③ 4D 확장 가능성 (시간축)
CAT4D의 핵심에는 카메라와 장면 모션의 제어를 분리하는 멀티뷰 비디오 디퓨전 모델이 있다. 이는 CAT3D 아키텍처의 직접적 확장이며, 정적 3D → 동적 4D 생성으로의 연구 경로를 제시한다.

### 9.2 향후 연구 시 고려할 점

#### ① 카메라 포즈 의존성 극복 (Pose-Free 연구)
CAT3D는 정확한 카메라 포즈에 의존하며 포즈-프리(pose-free) 문제로 자명하게 확장하기 어렵다. 카메라 포즈를 자동으로 추정하거나 포즈 없이 직접 3D 생성하는 연구 방향이 필요하다.

#### ② 대규모 장면 확장성
이러한 모델들은 일반적으로 소규모 공간 영역에 국한되며 복잡한 장면에서 hallucination에 취약하고, 출력물이 엄격한 3D 일관성을 보장하지 않는다. 도시 규모, 실내 전체 등 대규모 환경으로의 확장이 필요하다.

#### ③ 다양한 카메라 내부 매개변수 지원
현재 모델은 여러 다른 카메라 내부 매개변수(camera intrinsics)를 효과적으로 처리하지 못한다. 다양한 렌즈와 센서 설정에 대한 범용 지원이 필요하다.

#### ④ 3D 일관성 강화
3D 일관성을 달성하고, 대규모 장면으로 확장하고, 다양한 실세계 조건에 대한 일반화 등의 과제가 여전히 남아 있다. 기하학적 제약을 명시적으로 통합하는 방향이 중요하다.

#### ⑤ 계산 효율화
CAT3D를 훈련하려면 16개의 A100 GPU가 필요하다. 더 효율적인 훈련 및 추론 파이프라인 개발이 연구 과제다.

#### ⑥ 동적 장면으로의 확장
다중 동기화 카메라의 멀티뷰 비디오 데이터셋은 수집 비용 때문에 대규모로 존재하지 않으며, 합성 자산에서 렌더링하는 것은 가능하지만 기존 합성 4D 데이터셋만으로는 실세계로 일반화하기에 충분히 다양하거나 현실적이지 않다.

#### ⑦ 멀티모달 조건화 강화
텍스트 + 이미지 + 깊이 정보 등 다양한 모달리티를 동시에 활용하는 조건화 방법으로 일반화 폭을 넓히는 연구가 필요하다.

예를 들어, 생성적 디퓨전 모델과 3DGS 기술을 더욱 통합함으로써 더 정확하고 실시간에 가까운 3D 재구성을 달성할 수 있다.

---

## 📚 참고 자료 및 출처

| 번호 | 출처 | 링크/식별자 |
|------|------|------------|
| 1 | **CAT3D 공식 프로젝트 페이지** | [cat3d.github.io](https://cat3d.github.io/) |
| 2 | **arXiv 논문 원문 (v1)** | arXiv:2405.10314 |
| 3 | **arXiv HTML 전체 논문** | [arxiv.org/html/2405.10314v1](https://arxiv.org/html/2405.10314v1) |
| 4 | **NeurIPS 2024 공식 포스터** | [neurips.cc/virtual/2024/poster/95046](https://neurips.cc/virtual/2024/poster/95046) |
| 5 | **NeurIPS 2024 논문집 (ACM DL)** | [dl.acm.org/doi/10.5555/3737916.3740319](https://dl.acm.org/doi/10.5555/3737916.3740319) |
| 6 | **OpenReview** | [openreview.net/forum?id=TFZlFRl9Ks](https://openreview.net/forum?id=TFZlFRl9Ks) |
| 7 | **Liner Quick Review** | [liner.com CAT3D review](https://liner.com/review/cat3d-create-anything-in-3d-with-multiview-diffusion-models) |
| 8 | **Moonlight Literature Review** | [themoonlight.io CAT3D](https://www.themoonlight.io/en/review/cat3d-create-anything-in-3d-with-multi-view-diffusion-models) |
| 9 | **CAT4D 논문 (후속 연구)** | arXiv:2411.18613 |
| 10 | **Sparse-View 3D Reconstruction Survey** | arXiv:2507.16406 |
| 11 | **Diffusion Models for 3D Generation Survey** | [sciopen.com/article/10.26599/CVM.2025.9450452](https://www.sciopen.com/article/10.26599/CVM.2025.9450452) |
| 12 | **GS-Diff (후속 비교 연구)** | arXiv:2504.01960 |
| 13 | **Cascade-Zero123 (관련 연구)** | arXiv:2312.04424 |
| 14 | **HuggingFace Papers** | [huggingface.co/papers/2405.10314](https://huggingface.co/papers/2405.10314) |
| 15 | **Radiance Fields Blog** | [radiancefields.com CAT3D](https://radiancefields.com/cat3d-pounces-on-3d-scene-generation) |

> ⚠️ **정확도 참고사항**: 본 분석에서 제시된 수식 중 일부(특히 재구성 손실의 세부 가중치 계수)는 논문의 공식 HTML 원문에서 확인 가능한 구조를 기반으로 하되, 논문 원문 PDF의 세부 수식 표기와 미세한 차이가 있을 수 있습니다. 정확한 수식은 [arXiv PDF 원문](https://arxiv.org/pdf/2405.10314)을 직접 확인하시기 바랍니다.
