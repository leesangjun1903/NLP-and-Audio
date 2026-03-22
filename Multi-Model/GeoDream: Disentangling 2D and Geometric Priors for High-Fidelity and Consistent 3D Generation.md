# GeoDream: Disentangling 2D and Geometric Priors for High-Fidelity and Consistent 3D Generation

## 종합 분석 리포트

---

## 1. 핵심 주장 및 주요 기여 (요약)

Text-to-3D 생성은 사전 훈련된 대규모 text-to-image 확산 모델을 증류(distillation)하는 방식으로 큰 가능성을 보여왔지만, 여전히 비일관적인 3D 기하학적 구조(Janus problem)와 심각한 아티팩트로부터 문제를 겪고 있습니다. 이러한 문제는 주로 2D 확산 모델이 lifting 과정에서 3D 인식(3D awareness)이 부족하기 때문에 발생합니다.

**GeoDream의 핵심 주장:** 명시적이고 일반화된 3D 프라이어(prior)를 2D 확산 프라이어와 결합하여, 다양성(diversity)이나 충실도(fidelity)를 희생하지 않으면서 모호하지 않은 3D 일관 기하학적 구조를 얻는 능력을 향상시키는 GeoDream이라는 새로운 방법을 제안합니다.

### 주요 기여 (Contributions)

| 기여 | 설명 |
|------|------|
| **Cost Volume 기반 3D 프라이어** | 멀티뷰 확산 모델을 활용하여 포즈된 이미지를 생성하고, 예측된 이미지로부터 cost volume을 구성하여 3D 공간에서 공간적 일관성을 보장하는 네이티브 3D 기하학적 프라이어로 활용합니다. |
| **2D/3D 프라이어 분리 설계** | 3D 기하학적 프라이어를 활용하여 분리된 설계(disentangled design)를 통해 2D 확산 프라이어의 3D 인식 잠재력을 해제합니다. |
| **상호 보완적 개선** | 정제된 3D 기하학적 프라이어가 2D 확산 프라이어의 3D 인식 능력을 돕고, 이것이 다시 3D 기하학적 프라이어의 정제에 우수한 가이던스를 제공한다는 점을 입증합니다. |
| **Uni3D-score 메트릭** | 의미론적 일관성(semantic coherence)을 포괄적으로 평가하기 위해, 측정을 2D에서 3D로 끌어올리는 Uni3D-score 메트릭을 최초로 제안합니다. |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Text-to-3D 생성에서 사전 훈련된 대규모 text-to-image 확산 모델을 증류하는 접근은 큰 가능성을 보여왔으나, 비일관적인 3D 기하학적 구조(Janus problem)와 심각한 아티팩트를 여전히 겪고 있으며, 이는 2D 확산 모델이 lifting 시 3D 인식이 부족한 데서 기인합니다.

구체적으로, 기존 방법(DreamFusion 등)에서 사용하는 **Score Distillation Sampling (SDS)**는 2D 확산 모델만을 prior로 사용하므로, 3D 기하학적 구조에 대한 명시적 감독(supervision)이 없습니다. 이로 인해:

- **Janus Problem**: 객체의 전면(front) 특징이 후면(back)에도 나타나는 다중 얼굴 문제
- **기하학적 아티팩트**: 표면이 평평하거나 비현실적인 기하학 구조 생성
- **뷰 불일관성**: 서로 다른 시점에서의 렌더링이 일관되지 않음

### 2.2 제안 방법 (수식 포함)

#### Stage 1: Cost Volume 구성을 통한 3D 기하학적 프라이어 생성

MVS(Multi-View Stereo) 기반 방법을 따라, 멀티뷰 확산 모델이 예측한 멀티뷰 이미지로부터 cost volume을 네이티브 3D 기하학적 프라이어로 구성하며, 이는 약 2분 내에 완료됩니다.

멀티뷰 확산 모델(MVDream 또는 Zero123)을 사용하여 $N$개의 포즈된 이미지 $\{I_i, P_i\}_{i=1}^{N}$를 생성합니다. 여기서 $I_i$는 이미지, $P_i$는 카메라 파라미터입니다.

**Cost Volume 구성:**

각 복셀 위치 $\mathbf{v} \in \mathbb{R}^3$에 대해, 모든 소스 뷰의 특징(feature)을 역투영(back-projection)하여 비용을 계산합니다:

$$C(\mathbf{v}) = \text{Var}\left(\{F_i(\pi_i(\mathbf{v}))\}_{i=1}^{N}\right)$$

여기서 $F_i$는 이미지 $I_i$에서 추출된 feature map, $\pi_i$는 카메라 $P_i$에 대한 투영 함수, $\text{Var}(\cdot)$는 분산(variance) 연산입니다.

이 cost volume은 3D 공간에서의 occupancy/density를 나타내는 **네이티브 3D 기하학적 프라이어**로 사용됩니다.

#### Stage 2: 2D와 3D 프라이어의 분리 설계 (Disentangled Design)

프라이어 정제 단계에서, 기하학적 프라이어가 2D 확산 모델과 결합하여 렌더링 품질과 기하학적 정확도를 더욱 향상시킬 수 있음을 보여주며, 3D와 2D 프라이어를 분리하는 것이 2D 확산 프라이어의 일반화와 3D 프라이어의 일관성을 모두 유지하는 잠재적으로 흥미로운 방향임을 입증합니다.

**SDS (Score Distillation Sampling) Loss의 기본 형태:**

DreamFusion에서 제안된 SDS loss의 기울기는 다음과 같습니다:

$$\nabla_\theta \mathcal{L}_{\text{SDS}}(\phi, \mathbf{x}) = \mathbb{E}_{t, \epsilon}\left[w(t)\left(\hat{\epsilon}_\phi(\mathbf{z}_t; y, t) - \epsilon\right) \frac{\partial \mathbf{x}}{\partial \theta}\right]$$

여기서:
- $\theta$: 3D 표현의 파라미터 (NeuS 또는 DMTet)
- $\mathbf{x} = g(\theta, c)$: 카메라 $c$에서 렌더링된 이미지
- $\mathbf{z}_t = \alpha_t \mathbf{x} + \sigma_t \epsilon$: 노이즈가 추가된 이미지
- $\hat{\epsilon}_\phi$: 사전 훈련된 확산 모델의 노이즈 예측기
- $y$: 텍스트 프롬프트
- $w(t)$: 시간 의존 가중치 함수

**GeoDream의 분리 손실 함수:**

GeoDream은 3D 기하학적 프라이어와 2D 확산 프라이어를 분리하여 다음과 같은 총 손실 함수를 구성합니다:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{SDS}} \mathcal{L}_{\text{SDS}} + \lambda_{\text{geo}} \mathcal{L}_{\text{geo}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$

여기서:
- $\mathcal{L}_{\text{SDS}}$: 2D 확산 모델로부터의 SDS 손실 (외관/텍스처 가이던스)
- $\mathcal{L}_{\text{geo}}$: Cost volume 기반 3D 기하학적 프라이어 손실 (기하학적 일관성 가이던스)
- $\mathcal{L}_{\text{reg}}$: 정규화 항 (Eikonal loss 등)

**3D 기하학적 프라이어 손실:**

Cost volume $C$로부터 SDF(Signed Distance Function) 또는 density 값을 초기화하고, 이를 기하학적 제약으로 사용합니다:

$$\mathcal{L}_{\text{geo}} = \sum_{\mathbf{v}} \left\| \sigma_\theta(\mathbf{v}) - \hat{\sigma}_C(\mathbf{v}) \right\|^2$$

여기서 $\sigma_\theta(\mathbf{v})$는 학습 중인 3D 표현의 density, $\hat{\sigma}_C(\mathbf{v})$는 cost volume으로부터 추정된 density입니다.

**LoRA 기반 2D 프라이어의 3D-aware 강화:**

2D 확산 모델에 LoRA(Low-Rank Adaptation)를 적용하여, 3D 기하학적 프라이어가 가이드하는 방향으로 2D 모델을 미세 조정합니다. 이를 통해 2D 모델이 뷰에 따른 일관된 가이던스를 제공할 수 있게 합니다.

### 2.3 모델 구조

GeoDream은 정확한 기하학과 섬세한 시각적 디테일을 갖춘 3D 콘텐츠를 생성하는 것에 초점을 맞추며, 2D 확산 프라이어에 일반화 능력을 유지하면서 3D 일관 기하학을 생성하는 능력을 장착합니다. GeoDream은 다음 두 단계로 구성됩니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    GeoDream 전체 파이프라인                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [텍스트 프롬프트] ──→ [Multi-view Diffusion Model]           │
│        │               (MVDream / Zero123)                   │
│        │                      │                              │
│        │              [멀티뷰 이미지 생성]                     │
│        │                      │                              │
│  Stage 1:              [Cost Volume 구성]                    │
│  (~2분)                 (MVS 기반)                           │
│        │                      │                              │
│        │              [3D Geometric Prior]                    │
│        │                      │                              │
│  ──────┼──────────────────────┼──────────────────────────    │
│        │                      │                              │
│  Stage 2:     ┌───────────────┼───────────────┐              │
│  (정제)       │               │               │              │
│           [NeuS]    [Cost Volume Init]  [2D Diffusion        │
│           3D 표현    기하학 초기화        + LoRA]              │
│              │               │               │               │
│              └───────┬───────┘       ┌───────┘               │
│                      │               │                       │
│              [Disentangled           │                       │
│               Optimization]          │                       │
│                      │               │                       │
│              L_geo ←─┤──→ L_SDS ←────┘                       │
│                      │                                       │
│  ──────┬─────────────┼───────────────────────────────────    │
│        │             │                                       │
│  Stage 3:    [DMTet Geometry + Texture 정제]                 │
│  (선택적)     고해상도 렌더링 (1024×1024)                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

구체적으로, 훈련은 "coarse", "geometry", "texture" 세 단계로 나뉩니다.

- **Coarse Stage**: NeuS 기반의 neural SDF 표현을 사용하여 cost volume으로 초기화된 기하학을 SDS와 함께 최적화
- **Geometry Stage**: DMTet(Deep Marching Tetrahedra) 표현으로 전환하여 기하학 정제
- **Texture Stage**: 고해상도 텍스처 최적화 (1024×1024)

### 2.4 성능 향상

수치적/시각적 비교를 통해 GeoDream이 더 3D 일관적인 텍스처 메쉬를 고해상도(1024 × 1024) 사실적 렌더링으로 생성하며, 의미론적 일관성에도 더 밀접하게 부합함을 보여줍니다.

주요 성능 향상:

| 측면 | 개선 내용 |
|------|-----------|
| **Janus Problem** | 명시적 3D 프라이어를 2D 확산 프라이어와 결합하여 Janus problem을 완화합니다. |
| **멀티뷰 일관성** | 일관된 멀티뷰 렌더링 이미지와 풍부한 디테일의 텍스처 메쉬를 생성합니다. |
| **해상도** | 1024×1024의 고해상도 렌더링 달성 |
| **의미론적 평가** | Uni3D-score를 통해 3D 수준의 의미론적 일관성 측정을 최초 도입 |

### 2.5 한계점

논문 및 관련 분석으로부터 추론되는 GeoDream의 주요 한계점:

1. **Multi-view Diffusion Model 의존성**: 초기 멀티뷰 이미지의 품질에 크게 의존하며, 멀티뷰 확산 모델 자체의 한계가 최종 결과에 전파됨
2. **처리 시간**: Cost volume 구성에 ~2분, 전체 파이프라인에 수십 분 소요되어 실시간 생성에 부적합
3. **복잡 장면 한계**: 단일 객체 중심 설계로, 복잡한 다중 객체 장면이나 세밀한 구조 생성에 제한
4. **두 단계 환경 분리**: 멀티뷰 확산 모델의 사전 학습과 cost volume 구성 코드 간의 환경 충돌로 인해, 현재 두 개의 별도 가상환경을 사용해야 합니다.
5. **Cost Volume 해상도**: 복셀 그리드 기반 표현으로 인해 매우 세밀한 기하학적 디테일 표현에 한계가 있을 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

GeoDream의 핵심적 설계 철학은 **일반화 성능(generalizability)**과 **3D 일관성(consistency)**의 균형을 맞추는 것입니다.

3D와 2D 프라이어를 분리하는 것이 2D 확산 프라이어의 일반화와 3D 프라이어의 일관성을 모두 유지하는 잠재적으로 흥미로운 방향입니다.

### 3.1 일반화 성능을 위한 핵심 설계 원칙

**분리(Disentanglement)의 원리:**

$$\underbrace{\mathcal{L}_{\text{SDS}}}_{\text{2D 일반화 (외관/텍스처)}} + \underbrace{\mathcal{L}_{\text{geo}}}_{\text{3D 일관성 (기하학)}}$$

이 분리 설계의 핵심 이점은:

1. **2D 확산 모델의 일반화 능력 보존**: 수십억 장의 이미지로 훈련된 2D 확산 모델(Stable Diffusion 등)의 방대한 의미론적 지식을 그대로 활용
2. **3D 프라이어의 보완적 역할**: 3D 기하학적 프라이어는 구조적 일관성만 제공하므로, 2D 모델의 창의적 다양성을 제한하지 않음

### 3.2 일반화 성능 향상을 위한 구체적 메커니즘

**LoRA 기반 뷰 조건부 적응:**

2D 확산 모델에 LoRA를 적용하여, 원본 모델의 가중치를 보존하면서 3D-aware 능력을 추가합니다:

$$W' = W + \Delta W = W + BA$$

여기서 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$입니다. 이 저랭크 적응은 원래 모델의 일반화 능력을 최대한 보존하면서 3D 인식 능력을 주입합니다.

### 3.3 일반화 성능 향상의 미래 방향

1. **더 강력한 Multi-view Foundation Model**: 더 많은 3D 데이터와 더 큰 모델을 사용한 멀티뷰 확산 모델의 발전으로 cost volume 품질 향상 가능
2. **도메인 적응형 프라이어**: 특정 도메인(건축, 캐릭터, 자연물 등)에 특화된 기하학적 프라이어를 활용한 조건부 일반화
3. **스케일러블 3D 표현**: 해상도에 독립적인 implicit 표현이나 3D Gaussian Splatting과의 결합으로 일반화 능력 강화
4. **대규모 3D 데이터셋 활용**: Objaverse 등의 대규모 3D 데이터셋을 활용한 프라이어 모델의 일반화 범위 확대

---

## 4. 연구 영향 및 향후 연구 시 고려사항

### 4.1 학계 및 산업에 미치는 영향

GeoDream은 Text-to-3D 분야에서 다음과 같은 패러다임적 영향을 미칩니다:

1. **2D-3D 프라이어 분리 패러다임 확립**: 2D 확산 모델의 풍부한 의미론적 지식과 3D 기하학적 일관성을 분리하여 결합하는 방식은 이후 연구의 기본 설계 원칙으로 자리잡을 가능성이 높음
2. **MVS 기술의 생성 모델 도입**: 전통적인 Multi-View Stereo의 cost volume 기법을 생성적 3D 모델링에 도입하여, 컴퓨터 비전의 고전적 기법과 생성 AI의 융합 가능성을 제시
3. **3D 메트릭의 발전**: Uni3D-score는 향후 3D 생성 모델 평가의 표준으로 발전할 수 있는 기반을 마련

### 4.2 향후 연구 시 고려사항

| 고려사항 | 세부 내용 |
|----------|-----------|
| **속도 최적화** | Feed-forward 방식의 3D 생성(LRM, Instant3D 등)과의 결합으로 최적화 기반 방식의 속도 한계 극복 |
| **3D Gaussian Splatting 통합** | 3D Gaussian splatting 표현을 통해 2D와 3D 확산 모델의 힘을 연결하는 방향으로 GeoDream의 프레임워크를 확장 가능 |
| **더 정교한 Score Distillation** | ProlificDreamer가 제안한 VSD(Variational Score Distillation)와 같은 개선된 증류 기법과의 결합 |
| **대규모 Reconstruction Model** | LRM 등 대규모 재구성 모델과의 하이브리드 접근으로 일반화-일관성 트레이드오프 개선 |
| **물리 기반 렌더링(PBR)** | 재조명(relighting) 가능한 재질 추정과의 결합으로 실용성 향상 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 Text-to-3D 생성 방법론 계보

| 연구 | 연도 | 핵심 기법 | 3D 표현 | 장점 | 단점 |
|------|------|-----------|---------|------|------|
| **DreamFusion** | 2022 | 확률적 밀도 증류(probability density distillation) 기반 손실을 도입하여 2D 확산 모델을 파라메트릭 이미지 생성기 최적화의 prior로 활용 | NeRF | SDS 패러다임의 선구적 확립 | Janus problem, 저해상도 |
| **Magic3D** | 2022 | two-stage 최적화 프레임워크를 활용하여, 저해상도 확산 prior로 coarse model을 얻고 sparse 3D hash grid로 가속화한 후, coarse 표현을 초기화로 하여 고해상도 latent 확산 모델과 효율적인 미분 가능 렌더러로 텍스처 3D 메쉬 모델을 추가 최적화 | NeRF → DMTet | 40분 안에 고품질 3D 메쉬 모델 생성 가능, DreamFusion 대비 2배 빠르고 더 높은 해상도 달성 | 여전히 Janus problem 존재 |
| **ProlificDreamer** | 2023 | VSD(Variational Score Distillation) 제안, SDS의 over-smoothing 해결 | NeRF | 뛰어난 텍스처 품질 | 각 뷰별 텍스처 품질은 우수하나 이들을 결합하면 합리적인 3D 객체로 보이지 않는 경향 |
| **MVDream** | 2023 | 일관된 멀티뷰 이미지를 텍스트 프롬프트로부터 생성하는 확산 모델로, 2D와 3D 데이터 모두에서 학습하여 2D 확산 모델의 일반화 능력과 3D 렌더링의 일관성을 달성 | NeRF | 멀티뷰 일관성 크게 향상 | 기하학적 디테일 부족 |
| **Zero-1-to-3** | 2023 | Stable Diffusion 모델에 카메라 뷰포인트 제어를 장착하여, 단일 입력 이미지와 지정된 카메라 변환으로부터 새로운 뷰 합성을 가능하게 하고, 이를 3D 재구성에 적용 | 다양 | 단일 이미지로부터 3D 생성 | 기하학적 일관성 유지에 어려움이 있어 출력 3D 모델에서 흐림 현상이 발생 |
| **DreamGaussian** | 2023 | progressive densification | 3D Gaussians | 10배의 속도 향상 | 디테일 부족 |
| **GaussianDreamer** | 2024 | 3D 확산 모델이 초기화 프라이어를 제공하고 2D 확산 모델이 기하학과 외관을 풍부하게 하는 빠른 3D 객체 생성 프레임워크 | 3D Gaussians | 단일 GPU에서 15분 내에 고품질 3D 인스턴스 또는 아바타 생성 가능, 실시간 렌더링 지원 | 엣지 블러링 |
| **GeoDream** | 2023 | 2D/3D 프라이어 분리, Cost Volume | NeuS → DMTet | 3D 일관성 + 2D 일반화 모두 확보, 고해상도 | 처리 시간, 환경 분리 필요 |
| **DreamPolish** | 2024 | 정제된 기하학과 고품질 텍스처 생성에 뛰어난 text-to-3D 모델로, 기하학 구성 단계에서 여러 신경 표현을 활용하여 합성 안정성을 향상시키고, 뷰 조건부 확산 프라이어에만 의존하는 대신 다양한 시야각으로 조건화된 추가적인 노멀 추정기를 통해 기하학 디테일을 연마 | 다단계 | 표면 연마, 텍스처 품질 향상 | 복잡한 파이프라인 |

### 5.2 핵심 기술 동향 분석

#### (1) Score Distillation 변형 진화

$$\text{SDS (2022)} \rightarrow \text{VSD (2023)} \rightarrow \text{ISM (2024)} \rightarrow \text{Reward-guided (2024-2025)}$$

- **SDS** (DreamFusion): $\nabla_\theta \mathcal{L}\_{\text{SDS}} = \mathbb{E}\_{t,\epsilon}[w(t)(\hat{\epsilon}_\phi(\mathbf{z}_t; y, t) - \epsilon)\frac{\partial \mathbf{x}}{\partial \theta}]$
- **VSD** (ProlificDreamer): 3D 파라미터를 확률 분포로 모델링하여 mode-seeking 문제 완화
- **ISM** (LucidDreamer): DDIM inversion을 통해 역전 가능한 확산 궤적을 생성하여 pseudo-GT 불일관성으로 인한 평균화 효과를 줄이고, 확산 궤적의 두 구간 단계 간 매칭을 통해 단일 단계 최적화의 높은 재구성 오류를 방지

#### (2) 3D 표현의 진화

| 표현 | 대표 연구 | 특징 |
|------|-----------|------|
| NeRF (Implicit) | DreamFusion, MVDream | 부드러운 기하학, 느린 렌더링 |
| DMTet (Hybrid) | Magic3D, GeoDream | 명시적 메쉬, 미분 가능 |
| 3D Gaussian Splatting | DreamGaussian, GaussianDreamer | 빠른 렌더링, 편집 용이 |
| Tri-plane | DIRECT-3D | 효율적인 3D feature 인코딩 |

#### (3) 3D 일관성 확보 전략 비교

| 전략 | 대표 연구 | GeoDream과의 관계 |
|------|-----------|-------------------|
| View-dependent prompting | DreamFusion | 기본적이나 불충분 |
| Multi-view diffusion | MVDream, Zero123 | GeoDream의 Stage 1에서 활용 |
| Geometric prior integration | **GeoDream**, SweetDreamer | 명시적 3D 프라이어 도입 |
| 3D diffusion prior | Shap-E, GaussianDreamer | 3D 데이터 의존적 |
| Normal estimation | DreamPolish | 표면 정제에 특화 |
| Reward-guided | DreamCS (2025) | 3D 보상 모델 기반 정렬 |

### 5.3 GeoDream의 차별화 포지셔닝

GeoDream은 다음과 같은 고유한 위치를 차지합니다:

```
                    높은 일반화
                        ↑
                        │
     DreamFusion ●      │      ● GeoDream
     ProlificDreamer ●  │
                        │      ● DreamPolish
     ────────────────── ┼ ──────────────────→ 높은 3D 일관성
                        │
     DreamGaussian ●    │      ● MVDream
                        │      ● GaussianDreamer
                        │
                    낮은 일반화
```

GeoDream은 2D 확산 프라이어에 일반화 능력을 유지하면서 3D 일관 기하학을 생성하는 능력을 장착하는 것에 초점을 맞추고 있어, 위 그래프에서 우상단의 이상적인 위치를 추구합니다.

---

## 참고 자료 및 출처

1. **GeoDream 원논문**: Ma, B., Deng, H., Zhou, J., Liu, Y.-S., Huang, T., & Wang, X. (2023). "GeoDream: Disentangling 2D and Geometric Priors for High-Fidelity and Consistent 3D Generation." *arXiv preprint arXiv:2311.17971*. [https://arxiv.org/abs/2311.17971](https://arxiv.org/abs/2311.17971)
2. **GeoDream 프로젝트 페이지**: [https://mabaorui.github.io/GeoDream_page/](https://mabaorui.github.io/GeoDream_page/)
3. **GeoDream GitHub**: [https://github.com/baaivision/GeoDream](https://github.com/baaivision/GeoDream)
4. **OpenReview (ICLR 2025 제출)**: [https://openreview.net/forum?id=fIMf9zQo9d](https://openreview.net/forum?id=fIMf9zQo9d)
5. **DreamFusion**: Poole, B., Jain, A., Barron, J. T., & Mildenhall, B. (2022). "DreamFusion: Text-to-3D using 2D Diffusion." *arXiv:2209.14988*. [https://arxiv.org/abs/2209.14988](https://arxiv.org/abs/2209.14988)
6. **Magic3D**: Lin, C.-H. et al. (2023). "Magic3D: High-Resolution Text-to-3D Content Creation." *CVPR 2023*. [https://arxiv.org/abs/2211.10440](https://arxiv.org/abs/2211.10440)
7. **MVDream**: Shi, Y. et al. (2023). "MVDream: Multi-view Diffusion for 3D Generation." *ICLR 2024*. [https://arxiv.org/abs/2308.16512](https://arxiv.org/abs/2308.16512)
8. **GaussianDreamer**: Yi, T. et al. (2024). "GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models." *CVPR 2024*. [https://github.com/hustvl/GaussianDreamer](https://github.com/hustvl/GaussianDreamer)
9. **DreamPolish**: "DreamPolish: Domain Score Distillation With Progressive Geometry Generation." [https://arxiv.org/abs/2411.01602](https://arxiv.org/abs/2411.01602)
10. **DreamCS**: "DreamCS: Geometry-Aware Text-to-3D Generation with Unpaired 3D Reward Supervision." [https://arxiv.org/abs/2506.09814](https://arxiv.org/abs/2506.09814)
11. **3D Generative AI Survey**: "Progress and Prospects in 3D Generative AI: A Technical Overview." [https://arxiv.org/abs/2401.02620](https://arxiv.org/abs/2401.02620)
12. **Recent 3D Generation Survey**: "Recent Advance in 3D Object and Scene Generation: A Survey." [https://arxiv.org/abs/2504.11734](https://arxiv.org/abs/2504.11734)
13. **HuggingFace Papers**: [https://huggingface.co/papers/2311.17971](https://huggingface.co/papers/2311.17971)
14. **Text-to-3D Shape Generation Paper List**: [https://3dlg-hcvc.github.io/tt3dstar/](https://3dlg-hcvc.github.io/tt3dstar/)

---

> **참고**: 본 분석에서 제시된 수식 중 일부(특히 $\mathcal{L}_{\text{geo}}$의 구체적인 형태)는 논문의 공개된 Abstract, 프로젝트 페이지 설명, 코드베이스의 설정 파일 등을 기반으로 기술적으로 합리적인 수준에서 재구성한 것입니다. 정확한 수식은 원논문 PDF의 본문을 직접 참조하시기 바랍니다.
