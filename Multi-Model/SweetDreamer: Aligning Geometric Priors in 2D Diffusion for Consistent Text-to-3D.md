
# SweetDreamer: Aligning Geometric Priors in 2D Diffusion for Consistent Text-to-3D

**저자**: Weiyu Li, Rui Chen, Xuelin Chen, Ping Tan  
**발표**: ICLR 2024 (arXiv:2310.02596, 2023년 10월)

---

## 1. 핵심 주장 및 주요 기여 (요약)

사전 학습된 2D 확산 모델(diffusion model)의 2D 결과를 3D 세계로 들어올리는(lift) 과정은 본질적으로 모호하며, 2D 확산 모델은 뷰에 무관한(view-agnostic) 프라이어만 학습하므로 3D 지식이 부족하여 **다시점 비일관성(multi-view inconsistency)** 문제가 발생한다.

**핵심 발견**: 이 문제는 주로 **기하학적 비일관성(geometric inconsistency)**에서 기인하며, 잘못 배치된 기하학적 구조를 피하면 최종 출력의 문제가 상당히 완화된다.

**주요 기여**:
1. 2D 확산 모델의 기하학적 프라이어를 잘 정의된 3D 형상과 정렬(align)함으로써 일관성을 개선하며, 이는 2D 확산 모델을 뷰포인트 인식이 가능하도록 미세 조정하여 정규적으로 방향이 맞춰진(canonically oriented) 3D 객체의 뷰별 좌표 맵(coordinate map)을 생성하도록 함으로써 달성된다.
2. 이 "대략적(coarse)" 정렬은 기하학의 다시점 비일관성을 해결할 뿐 아니라, 3D 데이터셋에 없는 다양한 고품질 객체를 생성하는 2D 확산 모델의 능력도 유지한다.
3. 정렬된 기하학적 프라이어(AGP)는 범용적이며, 다양한 최신 파이프라인에 원활하게 통합 가능하여 높은 일반화 성능을 확보한다.
4. 인간 평가 기준 **85% 이상의 일관성 비율**로 새로운 SOTA 성능을 달성하였으며, 이전 방법들은 약 30% 수준이었다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기하학적 비일관성은 3D 인식이 부족한 2D 프라이어에 의해 부과되는 감독(supervision)에 의해 lifting 과정에서 더욱 악화되며, 동일한 2D 투영을 만드는 많은 비합리적 3D 구조가 2D 프라이어를 속일 수 있다. 경험적으로, 기하학적 비일관성이 다시점 비일관성 결과의 주요 원인이며, 대부분의 3D 비일관성 결과는 반복적인 기하학적 구조(예: 여러 개의 손, 얼굴)를 나타낸다.

이것은 **Janus Problem**(다면체 문제)으로도 알려져 있으며, 2D lifting 기법에서 다시점 지식이나 3D 인식 부족으로 인해 (1) 텍스트 프롬프트에 설명된 콘텐츠를 반복 생성하는 다면 Janus 문제 등의 도전이 발생한다.

### 2.2 제안하는 방법: Aligned Geometric Priors (AGP)

#### (a) 정규 좌표 맵(Canonical Coordinate Map, CCM) 생성

정규적으로 방향이 맞춰지고 정규화된 다양한 3D 모델로 구성된 3D 데이터셋에 접근할 수 있다고 가정하고, 랜덤 뷰에서 깊이 맵(depth map)을 렌더링한 후 정규 좌표 맵으로 변환한다.

오직 **대략적인(coarse) 기하학**만 렌더링하며, 이 3D 데이터를 사용하는 이점은 두 가지이다: (i) 모든 기하학이 3D에서 잘 정의되어 공간 배치에 모호함이 없고, (ii) 뷰포인트를 모델에 주입하여 뷰포인트 인식과 궁극적으로 3D 인식을 부여할 수 있다.

#### (b) 2D 확산 모델 미세 조정

2D 확산 모델을 지정된 뷰 아래에서 정규 좌표 맵을 생성하도록 미세 조정하여, 최종적으로 2D 확산의 기하학적 프라이어를 정렬한다.

미세 조정 시, 표준 확산 학습 목적 함수를 사용한다. 텍스트 프롬프트 $y$, 카메라 파라미터 $c$가 주어졌을 때, 노이즈가 추가된 정규 좌표 맵 $\mathbf{x}_t$에서 노이즈를 예측하는 목적 함수는:

$$\mathcal{L}_{\text{fine-tune}} = \mathbb{E}_{t, \boldsymbol{\epsilon}, c}\left[\left\|\boldsymbol{\epsilon}_\phi\left(\mathbf{x}_t ; t, y, c\right)-\boldsymbol{\epsilon}\right\|_2^2\right]$$

여기서 $\boldsymbol{\epsilon}_\phi$는 미세 조정되는 UNet 노이즈 예측기, $t$는 타임스텝, $\boldsymbol{\epsilon}$은 가우시안 노이즈이다.

#### (c) Score Distillation Sampling (SDS)을 통한 3D 최적화

DreamFusion(Poole et al., 2022)이 제안한 SDS 손실을 기반으로 한다. SDS의 그래디언트는 다음과 같이 표현된다:

$$\nabla_\theta \mathcal{L}_{\text{SDS}}(\theta) = \mathbb{E}_{t, \boldsymbol{\epsilon}}\left[w(t)\left(\boldsymbol{\epsilon}_\phi\left(\mathbf{x}_t ; t, y, c\right)-\boldsymbol{\epsilon}\right) \frac{\partial \mathbf{x}}{\partial \theta}\right]$$

여기서 $\theta$는 3D 표현(NeRF 또는 DMTet)의 파라미터, $w(t)$는 가중치 함수, $\mathbf{x}$는 렌더링된 이미지이다.

#### (d) 통합 손실 함수

AGP에 의한 추가 감독을 기존 파이프라인의 기하학 모델링 단계에 추가하며, 정렬된 확산 모델이 정규 좌표 맵을 입력으로 받아 SDS 손실을 산출하여 3D 표현을 업데이트한다.

최종 손실 함수는 다음과 같은 형태를 갖는다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SDS}}^{\text{RGB}} + \lambda_{\text{geo}} \mathcal{L}_{\text{SDS}}^{\text{AGP}} + \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{SDS}}^{\text{RGB}}$: 기존 RGB 기반 SDS 손실 (Stable Diffusion / DeepFloyd IF)
- $\mathcal{L}_{\text{SDS}}^{\text{AGP}}$: AGP(정렬된 기하학적 프라이어)에 의한 기하학 SDS 손실
- $\mathcal{L}_{\text{reg}}$: 정규화 항 (예: opacity, smoothness 등)
- $\lambda_{\text{geo}}$: 기하학 손실의 가중치 하이퍼파라미터

### 2.3 모델 구조

두 가지 SOTA 텍스트-to-3D 파이프라인과의 호환성을 시연한다: (1) 기하학과 외형 모델링을 분리하고 DMTet 하이브리드 표현을 사용하는 Fantasia3D, (2) NeRF를 3D 표현으로 사용하는 DreamFusion 기반 파이프라인.

| 구성 요소 | 상세 |
|-----------|------|
| **2D 확산 모델** | Stable Diffusion을 기반으로 CCM 생성을 위해 미세 조정 |
| **카메라 조건** | 뷰포인트(방위각, 고도각)를 추가 조건으로 주입 |
| **3D 데이터셋** | Objaverse 등 대규모 3D 에셋에서 정규적으로 배치된 대략적 기하학 사용 |
| **3D 표현 (DMTet 기반)** | DMTet 하이브리드 표현 + Fantasia3D 파이프라인 |
| **3D 표현 (NeRF 기반)** | Instant-NGP + DreamFusion 파이프라인 |
| **2단계 학습** | Stage 1: 대략적 기하학, Stage 2: 세부 외형 정제 |

NeRF 기반 파이프라인의 경우, 사전 학습된 확산 모델에 따라 DeepFloyd IF를 사용하는 버전과 DeepFloyd IF + Stable Diffusion을 사용하는 전체 버전(full)을 개발하였다.

### 2.4 성능 향상

SweetDreamer는 인간 평가 기준 85% 이상의 일관성 비율로 새로운 SOTA 성능을 달성하였으며, 이전 방법들은 약 30%였다.

다른 경쟁 방법들과 비교하여, SweetDreamer의 텍스트-to-3D 파이프라인은 높은 3D 일관성과 고충실도 3D 콘텐츠를 생성할 수 있다.

### 2.5 한계

RichDreamer 논문에 따르면, SweetDreamer가 제안하는 CCM(정규 좌표 맵)은 암묵적으로 학습 객체들이 정렬되어 있어야 하며, 합성 3D 데이터셋에서만 얻을 수 있어 **일반화 및 확장성이 잠재적으로 제한**될 수 있다.

추가적인 한계점:
- **3D 데이터셋 의존성**: 대략적 기하학이라 하더라도 정규적으로 배치된 3D 데이터가 필요
- **Appearance 비일관성의 미완전 해결**: 기하학 비일관성에 집중하여 외형 비일관성은 극단적 경우에서 여전히 발생 가능
- **계산 비용**: SDS 기반 반복 정제 방법은 모델당 1시간 이상 소요
- **단일 객체 중심**: 복잡한 장면(scene)보다 단일 객체에 최적화

---

## 3. 일반화 성능 향상 가능성

SweetDreamer의 일반화 전략의 핵심은 **"대략적(coarse) 정렬"** 설계 원칙에 있다.

기하학 및 외형 정보에 과도하게 의존하는 방법들과는 대조적으로, SweetDreamer는 오직 대략적 기하학만을 활용하여 불필요한 유도 편향(inductive bias)을 피한다.

이 "대략적" 기하학적 프라이어 정렬은 다시점 비일관성 문제 없이 3D 객체를 생성할 수 있게 할 뿐만 아니라, 3D 데이터셋에 없는 생생하고 다양한 객체를 생성하는 2D 확산 모델의 능력도 유지한다.

**일반화 관련 핵심 설계 원리:**

1. **Coarse-Only Alignment**: 세밀한 기하학 디테일이 아닌 대략적 형상만 사용하여, 2D 확산 모델이 학습한 풍부한 시각적 프라이어를 보존

2. **범용 통합성**: AGP는 범용적이며 다양한 SOTA 파이프라인에 원활하게 통합되어, 미지의 형상과 시각적 외형에 대해 높은 일반화 성능을 확보한다.

3. **Canonical Space 표현**: 정규 좌표계에서의 표현을 통해 객체의 방향에 대한 모호성을 해소하면서도 형상 자체의 다양성은 제한하지 않음

4. **Open-Vocabulary 유지**: 2D 확산 모델의 텍스트-이미지 생성 능력이 보존되어, 3D 학습 데이터에 없는 개념도 생성 가능

$$\text{Generalizability} \propto \frac{\text{2D Prior Capacity (Preserved)}}{\text{3D Geometric Bias (Minimized)}}$$

**일반화 한계 극복 방향:**
- 더 대규모이고 다양한 3D 데이터셋 활용 (예: Objaverse-XL)
- Normal map, Depth map 등 더 범용적인 기하학 표현으로의 확장 (RichDreamer가 이를 시도)
- 정규 방향 정렬(canonical orientation alignment) 없이도 작동하는 방법 연구

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

SweetDreamer는 Text-to-3D 분야에서 다음과 같은 중요한 영향을 미쳤다:

1. **"기하학 우선" 패러다임 확립**: 다시점 비일관성의 근본 원인이 기하학적 비일관성임을 실증적으로 입증하여, 이후 연구들이 기하학적 일관성에 집중하도록 방향을 제시
2. **플러그인 방식의 AGP 개념**: 기존 파이프라인에 추가적으로 통합 가능한 모듈형 접근법으로, 다양한 후속 연구에서 참조
3. **Coarse-to-Fine 철학**: 대략적 3D 정보만으로도 충분히 일관성을 확보할 수 있다는 통찰 제공

### 4.2 향후 연구 시 고려할 점

| 고려사항 | 상세 |
|----------|------|
| **3D 데이터 의존성 감소** | 합성 3D 데이터셋 없이도 기하학적 일관성을 확보하는 방법 모색 |
| **실시간 생성** | SDS 기반 최적화의 시간 비용을 줄이는 피드포워드(feed-forward) 방법과의 결합 |
| **장면 수준 확장** | 단일 객체를 넘어 복잡한 장면, 다중 객체 구성에 대한 확장 |
| **외형(Appearance) 일관성** | 기하학뿐 아니라 텍스처/재질의 다시점 일관성까지 동시에 해결 |
| **더 풍부한 기하학 표현** | CCM 외에 Normal, SDF 등 다양한 기하학 신호 활용 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 핵심 접근법 | 3D 일관성 전략 | 일반화 | 주요 한계 |
|------|------|-------------|----------------|--------|-----------|
| **DreamFusion** (Poole et al.) | 2022 | SDS + NeRF | 뷰 의존 텍스트 프롬프트 | 높음 (open-vocab) | Janus Problem 심각 |
| **Magic3D** (Lin et al.) | 2023 | Coarse-to-Fine SDS + DMTet | 2단계 정제 | 높음 | 여전히 다시점 비일관성 |
| **ProlificDreamer** (Wang et al.) | 2023 | VSD (Variational Score Distillation) | 분포 최적화 | 높음 | 다시점 일관성 부족, 장시간 |
| **Fantasia3D** (Chen et al.) | 2023 | 기하-외형 분리 + DMTet | 기하/외형 독립 최적화 | 중간 | 복잡한 형상에 사용자 가이드 필요 |
| **MVDream** (Shi et al.) | 2023 | Multi-view Diffusion + SDS | 다시점 동시 생성 (3D self-attention) | 중간 (3D 데이터 필요) | 텍스처 단순화 경향 |
| **SweetDreamer** (Li et al.) | 2023 | AGP + CCM + SDS | 정규 좌표 맵 정렬 | 높음 (coarse alignment) | 정규 배치 3D 데이터 필요 |
| **RichDreamer** (Qiu et al.) | 2024 | Normal-Depth Diffusion | Normal+Depth 공동 학습 | 매우 높음 (LAION-2B 사전학습) | 계산 비용 |
| **JointDreamer** (Yang et al.) | 2024 | Joint Score Distillation | 다시점 공동 분포 모델링 | 높음 | 에너지 함수 설계 복잡 |
| **Trellis** (Microsoft, 2024) | 2024 | Structured 3D Latents + Rectified Flow | 3D 잠재 공간 직접 생성 | 매우 높음 | 대규모 학습 데이터/자원 필요 |

### 핵심 비교 수식

**DreamFusion (SDS)**:

$$\nabla_\theta \mathcal{L}\_{\text{SDS}} = \mathbb{E}_{t,\boldsymbol{\epsilon}}\left[w(t)\left(\hat{\boldsymbol{\epsilon}}_\phi(\mathbf{x}_t;y,t) - \boldsymbol{\epsilon}\right)\frac{\partial \mathbf{x}}{\partial \theta}\right]$$

**ProlificDreamer (VSD)**:

$$\nabla_\theta \mathcal{L}_{\text{VSD}} = \mathbb{E}_{t,\boldsymbol{\epsilon}}\left[w(t)\left(\hat{\boldsymbol{\epsilon}}_\phi(\mathbf{x}_t;y,t) - \hat{\boldsymbol{\epsilon}}_\psi(\mathbf{x}_t;y,t,c)\right)\frac{\partial \mathbf{x}}{\partial \theta}\right]$$

**SweetDreamer (AGP-enhanced SDS)**:

$$\nabla_\theta \mathcal{L}_{\text{total}} = \nabla_\theta \mathcal{L}_{\text{SDS}}^{\text{RGB}} + \lambda_{\text{geo}} \nabla_\theta \mathcal{L}_{\text{SDS}}^{\text{AGP}}$$

여기서 $\mathcal{L}\_{\text{SDS}}^{\text{AGP}}$는 정렬된 기하학적 프라이어 확산 모델 $\boldsymbol{\epsilon}_\phi^{\text{AGP}}$에 의한 SDS:

$$\nabla_\theta \mathcal{L}\_{\text{SDS}}^{\text{AGP}} = \mathbb{E}_{t,\boldsymbol{\epsilon}}\left[w(t)\left(\boldsymbol{\epsilon}_\phi^{\text{AGP}}(\mathbf{m}_t;y,t,c) - \boldsymbol{\epsilon}\right)\frac{\partial \mathbf{m}}{\partial \theta}\right]$$

$\mathbf{m}$은 3D 표현에서 렌더링된 정규 좌표 맵이다.

SDS 기반 방식은 잘 학습된 2D 확산 모델을 통해 텍스트-to-3D 생성에서 큰 가능성을 보였으나, 뷰 비종속적(view-agnostic) 2D 이미지 분포를 각 뷰에 대해 독립적으로 3D 렌더링 분포로 증류(distill)하여 뷰 간 일관성을 간과하는 문제가 있다.

---

## 참고 자료 (출처)

1. **Li, W., Chen, R., Chen, X., & Tan, P.** (2023). *SweetDreamer: Aligning Geometric Priors in 2D Diffusion for Consistent Text-to-3D*. arXiv:2310.02596, ICLR 2024. — [arXiv](https://arxiv.org/abs/2310.02596), [프로젝트 페이지](https://sweetdreamer3d.github.io/), [GitHub](https://github.com/wyysf-98/SweetDreamer)
2. **Poole, B., Jain, A., Barron, J.T., & Mildenhall, B.** (2022). *DreamFusion: Text-to-3D using 2D Diffusion*. arXiv:2209.14988. — [프로젝트 페이지](https://dreamfusion3d.github.io/)
3. **Wang, Z., et al.** (2023). *ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation*. NeurIPS 2023 Spotlight. — [GitHub](https://github.com/thu-ml/prolificdreamer)
4. **Lin, C.H., et al.** (2023). *Magic3D: High-Resolution Text-to-3D Content Creation*. CVPR 2023.
5. **Shi, Y., et al.** (2023). *MVDream: Multi-view Diffusion for 3D Generation*. arXiv:2308.16512. — [arXiv HTML](https://arxiv.org/html/2308.16512)
6. **Qiu, L., et al.** (2024). *RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D*. arXiv:2311.16918. — [arXiv HTML](https://arxiv.org/html/2311.16918v2)
7. **Yang, H., et al.** (2024). *JointDreamer: Ensuring Geometry Consistency and Text Congruence in Text-to-3D Generation via Joint Score Distillation*. ECCV 2024. — [Springer](https://link.springer.com/chapter/10.1007/978-3-031-73347-5_25)
8. **Xiang, J., et al.** (2024). *Structured 3D Latents for Scalable and Versatile 3D Generation (Trellis)*. arXiv:2412.01506. — [arXiv HTML](https://arxiv.org/html/2412.01506v1)
9. **Xu, D., et al.** (2024). *Progress and Prospects in 3D Generative AI: A Technical Overview*. arXiv:2401.02620. — [arXiv HTML](https://arxiv.org/html/2401.02620v1)
10. **OpenReview** — [SweetDreamer Review](https://openreview.net/forum?id=extpNXo6hB)
11. **HuggingFace Papers** — [SweetDreamer](https://huggingface.co/papers/2310.02596)

> **참고**: 본 분석에서 논문의 구체적 실험 수치(예: CLIP Score 세부 값, 각 ablation 결과 등)는 원본 PDF를 직접 확인하시는 것을 권장합니다. 위 수식 중 통합 손실 함수의 정확한 하이퍼파라미터 표기는 논문의 표기 관행에 기반한 재구성이며, 원문에서의 정확한 기호와 다소 차이가 있을 수 있습니다.
