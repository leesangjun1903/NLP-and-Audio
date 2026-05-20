
# Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation

> **논문 정보**
> - **저자**: Tiange Xiang, Kai Li, Chengjiang Long, Christian Häne, Peihong Guo, Scott Delp, Ehsan Adeli, Li Fei-Fei
> - **소속**: Stanford University 외
> - **arXiv ID**: 2503.15877 (2025년 3월)
> - **학회**: ICCV 2025 (포스터 발표 확인)
> - **프로젝트 페이지**: https://cs.stanford.edu/~xtiange/projects/gaussianatlas/
> - **arXiv**: https://arxiv.org/abs/2503.15877

---

## 1. 🔑 핵심 주장 및 주요 기여 요약

### 핵심 주장

최근 텍스트-투-이미지(text-to-image) 확산 모델(diffusion model)의 발전은 방대한 양의 페어드 2D 데이터 덕분에 이루어졌다. 반면, 3D 확산 모델의 발전은 고품질 3D 데이터의 희소성으로 인해 저해되어 2D 모델 대비 경쟁력이 떨어진다. 이 한계를 해결하기 위해, 본 논문은 사전 학습된 2D 확산 모델을 3D 객체 생성에 재활용(repurpose)하는 방법을 제안한다.

### 주요 기여 (Contributions)

| 기여 항목 | 설명 |
|---|---|
| **Gaussian Atlas** | 3D Gaussian을 2D 그리드로 변환하는 표현 방식 도입 |
| **GaussianVerse 데이터셋** | 20만 5천 개 이상의 3D Gaussian 피팅 대규모 데이터셋 구축 |
| **2D→3D 전이 학습** | 사전 학습된 2D 확산 모델의 지식을 3D 생성으로 성공적으로 전이 |
| **SOTA 성능** | 적은 수의 Gaussian으로 최고 수준의 3D 생성 결과 달성 |

Gaussian Atlas는 비정형(unorganized) 3D Gaussian을 구조화된 2D 그리드로 변환하여, 3D 생성을 위한 2D 네트워크의 효과적인 활용을 가능하게 한다. GaussianVerse는 다양한 3D 객체에 대해 205,737개의 3D Gaussian 피팅을 포함하는 대규모 데이터셋으로, 확산 모델 훈련을 지원한다. 실험을 통해 본 방법이 훨씬 적은 수의 Gaussian으로 최첨단 3D 생성 결과를 달성함을 입증한다.

---

## 2. 🔬 상세 설명: 문제 정의 → 제안 방법 → 모델 구조 → 성능 및 한계

---

### 2-1. 해결하고자 하는 문제

3D 생성을 위한 독립적인 확산 모델을 설계하는 것은 간단하지만, 오직 3D 데이터만으로 학습할 경우 고품질 3D 데이터가 2D 이미지에 비해 희소하여 심각한 한계를 가진다. 더 높은 렌더링 충실도를 위해서는 잘 사전 학습된 2D 확산 모델의 사전 지식을 활용하는 것이 바람직하다.

구체적으로 기존 방식의 한계는 두 가지였다:

일부 방법들은 Score Distillation Sampling(SDS)이나 Collaborative Controls 등 복잡한 설계를 통해 동결된 가중치(frozen weights)의 사전 학습 2D 확산 모델을 3D 생성 파이프라인에 통합하려 했다. 그러나 이러한 방법들은 제한된 사전 정의 카테고리의 클래스 조건부 생성만 지원하거나 복잡한 UV 공간 표현에 의존하는 등의 한계가 있다.

역사적으로, 2D 확산 모델을 3D Gaussian으로 직접 파인튜닝(fine-tuning)하는 것은 실현 불가능한 것으로 여겨져 왔다.

---

### 2-2. 제안 방법: Gaussian Atlas

본 논문은 2D 확산 모델을 직접 파인튜닝을 통해 3D 생성에 재활용하는 새로운 관점을 제안한다. 이를 위해 **Gaussian Atlas**라는 3D Gaussian의 새로운 2D 표현을 도입한다.

Gaussian Atlas 변환은 세 단계로 구성된다:

#### Step 1: Sphere Offsetting (구형 최적 전송)

비정형 3D Gaussian을 Optimal Transport(OT)를 이용하여 단위 구(unit sphere)의 표면으로 매핑하여, 구조화된 레이아웃을 확보하고 효율성을 높인다.

수학적으로, 이 단계에서 Optimal Transport 문제는 다음과 같이 정의된다:

$$\min_{\pi \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) \, d\pi(x, y)$$

여기서:
- $\mu$: 원래 3D Gaussian 분포
- $\nu$: 단위 구 위의 균일 분포
- $c(x, y)$: 수송 비용 함수 (일반적으로 $\|x - y\|^2$)
- $\pi$: 결합 분포 (coupling)

#### Step 2: Projection to 2D (등장방형 투영)

구 위에 배치된 3D Gaussian들은 등장방형 투영(equirectangular projection)을 사용하여 평탄화되어 2D 좌표로 변환되며, 공간적 일관성이 보존된다.

등장방형 투영 수식:

$$u = \frac{\phi}{2\pi} \cdot W, \quad v = \frac{\theta - \frac{\pi}{2}}{\pi} \cdot H$$

여기서:
- $\phi \in [0, 2\pi]$: 경도(longitude)
- $\theta \in [0, \pi]$: 위도(latitude)
- $W, H$: 출력 이미지의 너비·높이

#### Step 3: Plane Offsetting (격자 최적 전송)

마지막으로 또 다른 OT 단계를 통해 2D 좌표를 구조화된 정방형 그리드(square grid)로 매핑하여 희소성(sparsity)을 줄인다.

이 최종 단계를 수식으로 표현하면:

$$\mathbf{G}_{atlas} = \text{PlaneOT}\left(\text{EqProj}\left(\text{SphereOT}(\{\mathbf{g}_i\}_{i=1}^{N})\right)\right)$$

여기서 $\mathbf{g}_i = \{\mu_i, \Sigma_i, \alpha_i, \mathbf{c}_i\}$는 각 3D Gaussian의 위치, 공분산, 불투명도, 색상 속성을 의미한다.

---

### 2-3. 모델 구조 (Pipeline)

파이프라인은 두 단계로 구성된다. **3DGS 사전 피팅 단계**에서는 다중 뷰 관측으로부터 다양한 3D 객체에 대해 고품질 3D Gaussian을 피팅한다. 이 대규모 3DGS 피팅들은 각 3DGS 속성에 대한 2D 표현으로 Gaussian Atlas로 재구조화된다. 이후 **확산 모델 학습 단계**에서는 변환된 2D Gaussian Atlas를 활용하여 사전 학습된 잠재 확산 모델(latent diffusion model, 2D UNet 디노이저 $F$)을 재활용하여 3D 콘텐츠 생성을 달성한다.

확산 모델의 역방향(denoising) 목표 함수는 일반적인 잠재 확산 모델 수식에 기반한다:

$$\mathcal{L}_{LDM} = \mathbb{E}_{\mathbf{z}, \boldsymbol{\epsilon} \sim \mathcal{N}(0,1), t} \left[ \left\| \boldsymbol{\epsilon} - F_\theta(\mathbf{z}_t, t, \mathbf{c}) \right\|^2 \right]$$

여기서:
- $\mathbf{z}$: Gaussian Atlas의 인코딩된 잠재 벡터
- $\mathbf{z}_t$: 타임스텝 $t$에서 노이즈가 추가된 잠재 변수
- $\boldsymbol{\epsilon}$: 추가된 가우시안 노이즈
- $F_\theta$: 파인튜닝된 2D UNet 디노이저
- $\mathbf{c}$: 텍스트 조건(conditioning)

---

### 2-4. 데이터셋: GaussianVerse

이전 연구들과 달리, GaussianVerse는 새로운 프루닝(pruning) 전략과 보다 효과적이지만 계산 집약적인 피팅 과정을 통해 최소한의 Gaussian 수로 더 높은 품질의 피팅을 제공한다.

더 높은 품질의 피팅을 위해 완전한 수렴이 이루어질 때까지 다양한 최적화 목표를 결합하여 광범위한 계산을 수행했다. 그 결과 데이터셋 구성에 약 **3.8 A100 GPU 연도(GPU-years)**의 계산이 필요했다.

---

### 2-5. 성능

본 접근법은 사전 학습된 2D 확산 모델의 학습된 지식을 활용하여 3D 콘텐츠 생성의 일반화 성능을 향상시킨다. Zero-shot 텍스트-투-3D 객체 생성 작업에서 평가하여, 생성 품질 및 텍스트 프롬프트와의 의미론적 정렬 측면에서 다른 3D Gaussian 생성 방법들을 능가함을 입증한다.

텍스트-투-이미지 확산 모델에서 학습된 사전 지식은 보편적으로 전이 가능하다. 대규모 2D 데이터셋으로 사전 학습 없이 3D 데이터만으로 훈련하면 모델이 자연어 이해와 콘텐츠 생성에 어려움을 겪지만, 사전 학습된 2D 확산 모델을 효과적으로 재활용하면 **훨씬 빠른 수렴(faster convergence)**과 우수한 일반화 성능을 달성할 수 있다.

---

### 2-6. 한계

기반 2D 확산 모델의 품질에 대한 의존성이 한계로 지적된다. 기존 2D 모델을 재활용하므로 그 편향(bias)과 제한을 그대로 상속받으며, 2D 모델이 특정 개념이나 카테고리에서 어려움을 겪으면 이 약점이 3D 출력에도 나타난다.

또한 3D Gaussian Splatting에 의존하는 방식은 장단점이 있다. Gaussian은 고품질의 효율적인 렌더링을 가능하게 하지만, 가장 간결하거나 편집하기 쉬운 3D 표현은 아니다. 메시(mesh)와 같은 전통적인 3D 포맷으로 변환 시 품질 손실이 발생할 수 있다.

논문은 계산 요구 사항을 충분히 다루지 않는다. 렌더링 자체는 빠르지만, 반복적 정제 과정은 생성 시 상당한 계산 자원을 요구할 가능성이 높으며, 이는 소비자용 하드웨어에서의 실용적 적용을 제한할 수 있다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

Gaussian Atlas는 **일관되고 포착 가능한 시각적 패턴(consistent and capturable visual patterns)**을 생성하며, 이로 인해 사전 학습된 2D 확산 모델이 더 쉽게 일반화할 수 있다.

일반화 성능 향상의 핵심 메커니즘은 세 가지 측면에서 분석할 수 있다:

#### (a) 2D 사전 지식의 전이 (Transfer of 2D Priors)

이 표현 방식은 비정형 3D Gaussian을 표준 3D 구로 이동시킨 후 등장방형 투영을 적용하여 정방형 2D 그리드로 매핑함으로써 직접 파인튜닝을 가능하게 한다. Gaussian Atlas는 2D 확산 모델에 포착된 사전 지식을 3D 생성 작업으로 전이하는 것을 촉진한다.

#### (b) 구조적 규칙성 (Structural Regularity)

비정형 3D Gaussian을 표준 3D 구로 이동시키고 등장방형 투영을 적용하여 정방형 2D 그리드(Gaussian Atlas)를 생성한다. 이 Gaussian Atlas는 2D 확산 모델에서 포착된 사전 지식을 3D 생성 작업으로 전이하는 것을 촉진한다.

#### (c) 대규모 데이터셋을 통한 일반화

본 접근법은 사전 학습된 2D 확산 모델로부터 3D 구조로 평탄화된 2D 다양체(manifold)로의 **성공적인 전이 학습(transfer learning)**을 입증한다.

---

## 4. 📊 2020년 이후 관련 최신 연구 비교 분석

| 논문/방법 | 연도 | 핵심 방식 | 특징 | 한계 |
|---|---|---|---|---|
| **DreamFusion** (Poole et al.) | 2022 | SDS + NeRF | 2D 모델 지식 활용, 텍스트→3D | 느린 최적화, Janus 문제 |
| **GaussianCube** | 2023 | Optimal Transport + 3D 볼륨 | 구조적 3D Gaussian | 3D UNet 사용, 2D priors 미활용 |
| **L3DG** | 2024 | 잠재 그리드 임베딩 + 확산 | 잠재 공간 확산 | 고정된 구조 |
| **GVGen** | 2024 | 볼륨 오프셋 기반 Gaussian | 3D Gaussian 직접 확산 | 볼륨 격자 제약 |
| **Atlas Gaussians Diffusion** (Yang et al.) | 2024 | 로컬 패치 + UV 샘플링 | 이론적으로 무한 개의 Gaussian 생성 가능 | 피드포워드 방식, 복잡한 아키텍처 |
| **GaussianAnything** | 2024 | 포인트 클라우드 공간 오토인코딩 | 2단계 잠재 생성 | 복잡한 2단계 파이프라인 |
| **GSD** | 2024 | 렌더링 가이던스 기반 Gaussian 샘플링 | 2D 관측으로 제약 | 2D priors 직접 활용 불가 |
| **Gaussian Atlas (본 논문)** | **2025** | **OT + 등장방형 투영 + 2D LDM 파인튜닝** | **2D 사전 지식 직접 활용, 구조화된 2D 표현** | **2D 모델 편향 상속, 계산 비용** |

GSD는 2D 관측으로 3D Gaussian 샘플링을 제약하는 렌더링 가이던스를 도입했다. L3DG는 3D Gaussian을 밀집 잠재 그리드에 임베딩하여 잠재 공간에서 확산 모델을 학습한다. GVGen과 GaussianCube는 희소하게 배치된 Gaussian을 더 구조화된 3D 볼륨으로 변환하여 3D Gaussian 확산을 달성한다. GVGen은 오프셋을 갖는 볼륨에 직접 피팅하고, GaussianCube는 OT를 적용하여 3D 격자의 꼭짓점으로 비정형 3D Gaussian을 이동시킨다.

Atlas Gaussians Diffusion(Yang et al., 2024)은 피드포워드 방식의 3D 생성을 위한 새로운 표현을 도입한다. 형상을 로컬 패치의 합집합으로 표현하며 각 패치가 3D Gaussian을 디코딩한다. 패치를 특징 벡터 시퀀스로 매개변수화하고 학습 가능한 함수로 디코딩하며, UV 기반 샘플링을 통해 이론적으로 무한한 수의 3D Gaussian 포인트 생성이 가능하다.

---

## 5. 🔭 향후 연구에 미치는 영향과 고려 사항

### 5-1. 향후 연구에 미치는 영향

**① 2D→3D 전이 학습 패러다임의 확립**

본 연구는 텍스트-투-이미지 생성을 위해 훈련된 2D 확산 모델이 3D 객체 생성에 재활용될 수 있음을 보여준다. Gaussian Atlas는 2D 확산 모델의 파인튜닝을 위한 3D Gaussian의 밀집 2D 그리드 표현이며, 3D 구조에서 평탄화된 2D 다양체로의 성공적인 전이 학습을 보여준다.

**② 파생 연구로의 확장 가능성**

텍스트-투-이미지 확산 모델은 풍부한 2D 데이터로 놀라운 성공을 거뒀지만 3D 확산 모델은 여전히 뒤처진다. 이에 GaussianAtlas로부터 영감을 받은 Shape Atlas와 같은 파생 연구가 등장하고 있으며, 이러한 방법들은 GaussianAtlas와 마찬가지로 사전 학습된 잠재 확산 모델(즉, Stable Diffusion)을 2D 표현으로 파인튜닝한다.

**③ 3D 데이터셋 구축의 중요성 인식 고조**

이전 연구들과 달리 GaussianVerse가 제시한 새로운 프루닝 전략과 효과적인 피팅 방식은 향후 대규모 3D 데이터셋 구축의 기준점이 될 수 있다.

---

### 5-2. 향후 연구 시 고려할 점

**① 기반 모델 편향 극복**

2D 확산 모델의 편향과 제한을 그대로 상속받으므로, 특정 개념이나 카테고리에서 취약점이 3D 출력에 반영된다. 향후 연구에서는 도메인 특화된 사전 학습이나 데이터 증강을 통해 이를 극복해야 한다.

**② 씬(Scene) 수준으로의 확장**

본 논문은 주로 개별 객체 생성에 집중되어 있다. 3D 세계의 이해는 수많은 실세계 응용에 필수적이며, 이 연구는 텍스트 설명으로부터 고품질 3D 자산 생성에 초점을 맞춘다. 향후 연구에서는 복수 객체, 복잡한 씬 수준으로 확장하는 시도가 필요하다.

**③ 메시 변환 및 편집 가능성**

3D Gaussian Splatting에 대한 의존성은 편집 가능성을 제한하며, 메시 등 전통적 포맷 변환 시 품질 손실 문제가 발생할 수 있다. 이를 해결하기 위한 Gaussian-to-Mesh 변환 기술의 발전이 요구된다.

**④ 계산 효율성 개선**

데이터셋 구성에만 약 3.8 A100 GPU 연도의 계산이 필요했다는 점에서, 향후 연구는 데이터 효율적 학습, 경량화된 피팅 전략, 또는 Few-shot 설정에서의 일반화 향상을 중심으로 진행되어야 한다.

**⑤ 조건부 생성으로의 확장**

GaussianAtlas와 달리, 이를 기반으로 한 후속 연구들은 조건부 U-Net을 통합하여 불완전한 포인트 클라우드에서의 실용적인 3D 완성 등 조건부(conditional) 생성으로 확장하려 시도한다. 이는 중요한 방향으로, 단순 텍스트 조건부를 넘어 이미지, 깊이맵, 부분 3D 등 다양한 조건을 통합하는 연구가 필요하다.

---

## 📚 참고 자료 (출처)

1. **arXiv 원문**: Tiange Xiang et al., "Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation," arXiv:2503.15877, 2025. — https://arxiv.org/abs/2503.15877
2. **arXiv HTML (v1)**: https://arxiv.org/html/2503.15877v1
3. **arXiv HTML (v2)**: https://arxiv.org/html/2503.15877v2
4. **arXiv PDF**: https://arxiv.org/pdf/2503.15877
5. **공식 프로젝트 페이지 (Stanford)**: https://cs.stanford.edu/~xtiange/projects/gaussianatlas/
6. **ICCV 2025 포스터 페이지**: https://iccv.thecvf.com/virtual/2025/poster/1130
7. **Papers with Code**: https://paperswithcode.com/paper/repurposing-2d-diffusion-models-with-gaussian
8. **ResearchGate PDF**: https://www.researchgate.net/publication/390038843
9. **AI Models FYI 요약**: https://www.aimodels.fyi/papers/arxiv/repurposing-2d-diffusion-models-gaussian-atlas-3d
10. **Moonlight 리뷰**: https://www.themoonlight.io/en/review/repurposing-2d-diffusion-models-with-gaussian-atlas-for-3d-generation
11. **관련 비교 논문 - Atlas Gaussians Diffusion for 3D Generation** (Yang et al., 2024): arXiv:2408.13055 — https://arxiv.org/abs/2408.13055
12. **관련 파생 논문 - Repurposing 2D Diffusion Models for 3D Shape Completion** (2024): arXiv:2512.13991 — https://arxiv.org/html/2512.13991

> ⚠️ **정확도 주의**: 수식 중 논문에 명시적으로 기재된 수식 외에 OT 문제, LDM 목적함수 등 일부는 논문의 접근법을 기반으로 표준적인 수학 표현으로 재구성한 것입니다. 완전한 수식은 arXiv 원문 PDF를 직접 확인하시길 권장합니다.
