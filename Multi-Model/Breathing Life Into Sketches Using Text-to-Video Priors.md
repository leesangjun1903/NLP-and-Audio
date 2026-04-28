# Breathing Life Into Sketches Using Text-to-Video Priors

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **단일 정적 벡터 스케치(vector sketch)**와 **텍스트 프롬프트**만을 입력으로 받아, 사전 학습된 text-to-video diffusion 모델의 모션 사전 지식(motion prior)을 활용하여 스케치를 자동으로 애니메이션화하는 방법("LiveSketch")을 제안합니다.

핵심 발견 중 하나는, 스케치 비디오를 직접 생성하는 데 어려움을 겪는 text-to-video 모델조차도 **추상적 표현(abstract representation)의 시맨틱을 이해**하고, 이를 통해 의미 있는 모션 신호를 추출하는 데 활용될 수 있다는 점입니다.

### 주요 기여

| 기여 | 설명 |
|---|---|
| **벡터 기반 스케치 애니메이션** | 픽셀이 아닌 벡터(SVG/Bézier) 표현을 유지하며 애니메이션 생성 |
| **SDS Loss의 비디오 도메인 적용** | Score Distillation Sampling을 text-to-video 모델에 적용하여 모션 추출 |
| **이중 모션 분리 구조** | Local(미세 변형) + Global(전역 아핀 변환)의 분리된 모션 모델링 |
| **주석 불필요** | 스켈레톤, 키포인트, 레퍼런스 모션 등 수동 주석 없이 동작 |
| **모델 불가지론적 설계** | 다양한 text-to-video 백본과 호환 가능 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

기존 스케치 애니메이션 방법들은 다음의 한계를 가집니다:

- **수동 주석 의존성**: 스켈레톤 키포인트[Smith et al., 2023], 레퍼런스 모션[Su et al., 2018] 등 사람의 개입 필요
- **도메인 제한**: 주로 인간 형태에 특화, 비인간 객체 애니메이션에 취약
- **픽셀 기반 한계**: 픽셀 도메인 image-to-video 모델은 스케치의 추상적 특성을 보존하지 못함

본 논문은 세 가지 목표를 동시에 달성하는 방법을 제안합니다:

1. 출력 비디오가 텍스트 프롬프트와 일치
2. 원본 스케치의 외관(appearance) 보존
3. 자연스럽고 부드러운 모션 생성

---

### 2-2. 제안하는 방법 및 수식

#### (1) 벡터 표현

스케치를 $N$개의 제어점(control point)으로 구성된 집합으로 표현합니다:

$$P = \{p_1, \ldots, p_N\} \in \mathbb{R}^{N \times 2}$$

여기서 각 제어점은 $p = (x, y) \in \mathbb{R}^2$로 2D 좌표를 나타냅니다.

$k$프레임으로 구성된 비디오는 다음과 같이 정의됩니다:

$$Z = \{P^j\}_{j=1}^{k} \in \mathbb{R}^{N \cdot k \times 2}$$

학습 목표는 각 프레임의 각 제어점에 대한 **2D 변위(displacement)**를 예측하는 것입니다:

$$\Delta Z = \{\Delta p_i^j\}_{j \in k, i \in N}$$

---

#### (2) Score Distillation Sampling (SDS) Loss

Poole et al. (DreamFusion, 2022)에서 처음 제안된 SDS Loss를 text-to-video 모델에 적용합니다.

**노이징 과정:**

$$x_t = \alpha_t x + \sigma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

여기서 $\alpha_t$, $\sigma_t$는 diffusion 모델의 노이징 스케줄에 따른 파라미터입니다.

**SDS Gradient:**

$$\nabla_\phi \mathcal{L}_{SDS} = \left[ w(t)(\epsilon_\theta(x_t, t, y) - \epsilon) \frac{\partial x}{\partial \phi} \right]$$

- $w(t)$: $\alpha_t$에 의존하는 가중치 상수
- $\epsilon_\theta(x_t, t, y)$: text-to-video diffusion 모델의 노이즈 예측값
- $\phi$: 모션 예측 네트워크 $\mathcal{M}$의 파라미터
- $y$: 텍스트 프롬프트 컨디셔닝

이를 통해 네트워크 $\mathcal{M}$이 생성한 애니메이션이 텍스트 프롬프트와 정렬되도록 학습합니다.

---

#### (3) 글로벌 모션 (Affine Transformation)

글로벌 경로는 각 프레임 $j$에 대해 아핀 변환 행렬 $T^j$를 예측합니다:

$$\Delta p_{i,\text{global}}^j = T^j \odot p_i^{\text{init}} - p_i^{\text{init}}$$

변환 행렬은 스케일($s_x, s_y$), 전단(shear: $sh_x, sh_y$), 회전($\theta$), 평행이동($d_x, d_y$)의 순차적 적용으로 구성됩니다:

```math
T^j = \underbrace{\begin{bmatrix} s_x & sh_x s_y & d_x \\ sh_y s_x & s_y & d_y \\ 0 & 0 & 1 \end{bmatrix}}_{\text{Scale + Shear + Translation}} \cdot \underbrace{\begin{bmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}}_{\text{Rotation}}
```

각 변환 유형에 대한 사용자 제어 스케일링 파라미터를 추가합니다:

- $\lambda_t$: 평행이동 스케일
- $\lambda_r$: 회전 스케일
- $\lambda_s$: 크기 변화 스케일
- $\lambda_{sh}$: 전단 스케일

예를 들어, 평행이동에 대해:

$$(d_x^j, d_y^j) \rightarrow (\lambda_t d_x^j, \lambda_t d_y^j)$$

---

#### (4) 최종 변위

로컬 변위 $\Delta Z_l$과 글로벌 변위 $\Delta Z_g$의 합으로 최종 변위를 계산합니다:

$$\Delta Z = \Delta Z_l + \Delta Z_g$$

렌더링된 비디오 프레임:

$$F^j = \mathcal{R}(P^j), \quad F = \{F^1, \ldots, F^k\} \in \mathbb{R}^{h \times w \times k}$$

여기서 $\mathcal{R}$은 미분 가능한 래스터라이저(differentiable rasterizer)[Li et al., 2020]입니다.

---

### 2-3. 모델 구조

```
입력: Z^init (초기 제어점 집합)
       ↓
[공유 백본 M_shared]
  - 각 제어점 p_i^j를 잠재 표현으로 투영
  - 프레임 인덱스 + 점 순서 기반 위치 인코딩 추가
       ↓
    ┌──────────────────────────────────┐
    │                                  │
[로컬 경로 M_l]              [글로벌 경로 M_g]
- 소형 MLP                   - T^j (아핀 행렬) 예측
- 각 점에 대한               - 프레임 전체에 균일 적용
  비제약 오프셋 ΔZ_l 예측    - ΔZ_g = T^j ⊙ P^init - P^init
    │                                  │
    └────────────┬─────────────────────┘
                 ↓
          ΔZ = ΔZ_l + ΔZ_g
                 ↓
         미분 가능한 래스터라이저 R
                 ↓
         렌더링된 비디오 F
                 ↓
         SDS Loss (text-to-video 모델 ε_θ 사용)
```

**학습 세부사항:**
- 로컬/글로벌 경로를 교대로 최적화
- 옵티마이저: Adam
  - 로컬 경로 LR: $1 \times 10^{-4}$
  - 글로벌 경로 LR: $5 \times 10^{-3}$
- SDS guidance scale: 로컬 30, 글로벌 40
- 기본 백본: ModelScope text-to-video
- 학습 스텝: 1,000 (A100 GPU, 약 30분)
- 데이터 증강: random crop, perspective transform

---

### 2-4. 성능 향상

#### 정량적 비교 (Table 1a)

| 방법 | Sketch-to-Video Consistency (↑) | Text-to-Video Alignment (↑) |
|---|---|---|
| ZeroScope | $0.754 \pm 0.009$ | - |
| ModelScope | $0.779 \pm 0.009$ | - |
| VideoCrafter | $0.876 \pm 0.007$ | $0.124 \pm 0.005$ |
| **Ours** | $\mathbf{0.965 \pm 0.003}$ | $\mathbf{0.142 \pm 0.005}$ |

- 스케치-비디오 일관성에서 모든 베이스라인을 크게 능가
- 자신의 prior로 사용된 ModelScope 대비 sketch-to-video consistency에서 약 **24% 향상**

#### Ablation Study (Table 1b)

| 설정 | Sketch-to-Video Consistency | Text-to-Video Alignment |
|---|---|---|
| Full | $0.965 \pm 0.003$ | $0.142 \pm 0.005$ |
| No Network | $0.926 \pm 0.007$ | $0.142 \pm 0.005$ |
| No Global | $0.936 \pm 0.006$ | $0.140 \pm 0.005$ |
| No Local | $0.970 \pm 0.002$ | $0.140 \pm 0.004$ |

- 네트워크 제거 시: 지터 증가, 형태 보존 저하
- 글로벌 경로 제거 시: 프레임 전반적 움직임 감소, 비일관적 형태 변환
- 로컬 경로 제거 시: 스케치 형태는 잘 보존되나 비현실적 흔들림, 의미 있는 모션 생성 실패

---

### 2-5. 한계

1. **스케치 표현 의존성**: CLIPasso[Vinker et al., 2022] 기반 cubic Bézier 표현에 최적화. 다른 스케치 표현(TU-Berlin 등)에서는 추가 하이퍼파라미터 튜닝 필요
2. **단일 객체 가정**: 다수 객체나 장면 스케치에서 품질 저하 (e.g., 농구공과 선수 분리 불가)
3. **모션-외관 트레이드오프**: 모션 품질과 스케치 충실도 간 근본적인 균형 문제 존재
4. **Text-to-video prior 한계 상속**: 백본 모델의 편향이나 미지원 모션은 표현 불가
5. **계산 비용**: A100 GPU에서 약 30분 소요 (실시간 응용 불가)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 현재 일반화 성능

본 논문은 다양한 카테고리(인간, 동물, 사물)에 걸쳐 일반화 능력을 보여주며, 이는 주로 다음 요인에서 비롯됩니다:

**① 인터넷 규모 사전 지식 상속**

text-to-video diffusion 모델(ModelScope 등)은 대규모 비디오 데이터로 학습되어, 다양한 객체의 물리적 모션 패턴을 내재하고 있습니다. 본 방법은 이를 직접 활용함으로써 별도의 도메인별 학습 데이터 없이 광범위한 카테고리에 적용 가능합니다.

**② 벡터 표현의 도메인 불변성**

픽셀 기반이 아닌 제어점 기반 표현은 특정 외관 스타일에 덜 의존적이며, 래스터라이저를 통해 임의의 스케치를 동일한 방식으로 처리할 수 있습니다.

**③ 백본 불가지론적 설계**

논문은 ZeroScope의 다양한 버전, ModelScope 등 여러 text-to-video 모델과의 호환성을 실험적으로 검증하였습니다 (Figure 15). 이는 향후 더 강력한 백본 모델 등장 시 성능을 쉽게 향상시킬 수 있음을 의미합니다.

---

### 3-2. 일반화 성능 향상을 위한 잠재적 방향

#### (A) 더 강력한 Text-to-Video 백본 활용

논문은 사용된 ModelScope가 스케치 비디오를 직접 생성하는 데 어려움을 겪음에도 불구하고, 모션 prior로서는 효과적으로 작동함을 보였습니다. 따라서 Sora, CogVideo, Stable Video Diffusion 등 더 표현력 있는 최신 모델을 백본으로 사용하면 일반화 성능이 대폭 향상될 것으로 기대됩니다.

#### (B) 개인화된 모션 모델 활용

논문은 Textual Inversion[Gal et al., 2022]과 같은 개인화 기법으로 증강된 모델을 백본으로 활용할 가능성을 언급합니다. 특정 스타일의 모션이나 새로운, 미관측 모션 패턴을 학습시킨 개인화 모델은 해당 도메인에서의 일반화 성능을 향상시킬 수 있습니다.

#### (C) 메쉬 기반 표현으로의 확장

현재의 Bézier 곡선 기반 표현을 As-Rigid-As-Possible(ARAP)[Igarashi et al., 2005] 손실과 함께 메쉬 기반 표현으로 확장하면, 형태 보존과 모션 표현력 간의 트레이드오프를 더 잘 제어할 수 있어 다양한 스케치 표현에 대한 일반화가 향상될 것입니다.

#### (D) Diffusion Feature Space에서의 일관성 강화

TokenFlow[Geyer et al., 2023]와 같이 diffusion feature space에서 일관성을 강제하는 방법을 도입하면, 특히 복잡한 스케치나 다중 객체 시나리오에서 더 일관된 애니메이션 생성이 가능합니다.

#### (E) 추상화 수준 적응

실험 결과(Figure 11), 4개 스트로크만으로 구성된 매우 추상적인 스케치에서도 어느 정도의 애니메이션이 가능하나 품질이 저하됩니다. 추상화 수준에 따른 적응적 하이퍼파라미터 조정이나 추상화 수준을 조건으로 입력받는 아키텍처 설계가 일반화 성능을 향상시킬 수 있습니다.

#### (F) 다중 객체 시나리오

현재 단일 객체 가정을 완화하기 위해, 객체 분할(instance segmentation) 또는 관계 그래프(relational graph) 기반 모션 모델링을 도입하면 장면 수준의 스케치에서도 일반화 성능을 향상시킬 수 있습니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4-1. 스케치 생성/표현

| 연구 | 연도 | 방법 | 본 논문과의 관계 |
|---|---|---|---|
| CLIPasso [Vinker et al.] | 2022 | CLIP 기반 의미론적 스케치 생성 | 스케치 생성에 직접 활용 |
| CLIPascene [Vinker et al.] | 2022 | 장면 수준 스케치 생성 | 평가 프로토콜 참조 |
| VectorFusion [Jain et al.] | 2022 | SDS를 SVG 최적화에 적용 | SDS 적용의 선행 연구 |
| Word-as-Image [Iluz et al.] | 2023 | 의미론적 타이포그래피를 위한 SVG 최적화 | SDS 기반 벡터 최적화 선행 연구 |

### 4-2. Text-to-Video 생성

| 연구 | 연도 | 방법 | 본 논문과의 관계 |
|---|---|---|---|
| Imagen Video [Ho et al.] | 2022 | 계단식 diffusion 비디오 생성 | 백본 후보 |
| ModelScope [Wang et al.] | 2023 | 잠재 비디오 diffusion 모델 | **본 논문의 기본 백본** |
| VideoCrafter [Chen et al.] | 2023 | 이미지+텍스트 조건부 비디오 생성 | 비교 베이스라인 |
| DynamiCrafter [Xing et al.] | 2023 | 오픈 도메인 이미지 애니메이션 | 유사 목적의 pixel 기반 방법 |

### 4-3. 이미지/스케치 애니메이션

| 연구 | 연도 | 방법 | 본 논문 대비 한계 |
|---|---|---|---|
| Animated Drawings [Smith et al.] | 2023 | 스켈레톤 기반 어린이 그림 애니메이션 | 인간 형태 한정, 수동 주석 필요 |
| Make-It-Move [Hu et al.] | 2022 | 인코더-디코더 이미지-비디오 생성 | 픽셀 기반, 스케치 보존 어려움 |
| Live Sketch [Su et al.] | 2018 | 구동 비디오 기반 스케치 변형 | 레퍼런스 모션 필요 |
| TokenFlow [Geyer et al.] | 2023 | Diffusion feature 일관성 기반 비디오 편집 | 스케치 특화 아님 |

### 4-4. SDS 기반 최적화

| 연구 | 연도 | 방법 | 본 논문과의 관계 |
|---|---|---|---|
| DreamFusion [Poole et al.] | 2022 | SDS를 NeRF에 적용한 text-to-3D | **SDS의 원류** |
| VectorFusion [Jain et al.] | 2022 | SDS를 SVG 최적화에 적용 | 정적 벡터 최적화 선행 연구 |
| 본 논문 | 2023 | SDS를 text-to-video 모델에 적용하여 스케치 모션 추출 | **새로운 기여** |

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

**① 비픽셀 도메인에서의 비디오 생성 패러다임 확립**

본 논문은 SDS Loss를 text-to-**video** 모델에 적용하여 비픽셀(벡터) 표현의 동적 최적화를 최초로 시도한 연구입니다. 이는 향후 NeRF 기반 동적 장면, 3D 애니메이션, 포인트 클라우드 애니메이션 등 다양한 비픽셀 도메인에서의 모션 생성 연구에 영향을 미칠 것입니다.

**② "인터넷 규모 지식의 추상 도메인 전이" 개념의 확장**

"스케치를 이해하지 못하는 모델도 모션 prior로는 유용하다"는 발견은 도메인 갭이 큰 상황에서도 대형 사전 학습 모델의 지식을 활용하는 방법론적 영감을 제공합니다. 이는 의료 이미지 스케치, 공학 도면, 건축 설계도 등의 특수 도메인 애니메이션 연구로 확장될 수 있습니다.

**③ 편집 가능한 AI 생성 콘텐츠(AIGC) 연구 촉진**

벡터 형식의 출력은 후속 편집이 용이하다는 장점이 있습니다. 이는 AI 생성 콘텐츠의 편집 가능성(editability)에 관한 연구, 특히 디자인 워크플로우에서의 AI 활용 연구에 직접적인 영향을 미칩니다.

**④ 멀티모달 생성 모델 활용 방법론**

텍스트-비디오-벡터를 연결하는 본 방법론은 향후 텍스트-오디오-모션 연동, 텍스트-3D 애니메이션 등 멀티모달 생성 파이프라인 설계에 기여할 것입니다.

---

### 5-2. 앞으로 연구 시 고려할 점

**① 더 강력한 평가 지표 개발 필요**

현재 사용된 CLIP 기반 sketch-to-video consistency와 X-CLIP 기반 text-to-video alignment는 모션의 자연스러움, 시간적 일관성, 물리적 타당성 등을 충분히 측정하지 못합니다. 다음과 같은 지표들의 개발이 필요합니다:
- 시간적 일관성 지표 (FVD: Fréchet Video Distance 등)
- 모션 자연스러움의 인지심리학적 평가 기준
- 스케치 충실도의 더 정교한 측정 방법

**② 실시간 추론을 위한 효율화**

현재 30분의 최적화 시간은 실용적 응용을 제한합니다. 향후 연구에서는:
- 사전 학습된 모션 생성 네트워크를 활용한 피드포워드 방식
- 모델 증류(distillation) 기법 적용
- 경량화된 백본 모델 활용

등을 통한 실시간 또는 준실시간 처리가 고려되어야 합니다.

**③ 다중 객체 및 장면 수준 애니메이션**

단일 객체 제약을 완화하기 위한 연구가 필요합니다. 특히:
- 객체 간 상호작용(interaction) 모델링
- 배경과 전경의 분리된 모션 처리
- 물리 기반 시뮬레이션과의 통합

**④ 모션의 시간적 제어성 향상**

현재 방법은 전체 비디오의 모션을 일괄 최적화하여 사용자가 특정 시점의 모션을 세밀하게 제어하기 어렵습니다. 키프레임 기반 제어나 모션 편집 인터페이스의 통합이 필요합니다.

**⑤ 윤리적 고려사항**

- 저작권이 있는 스케치나 아티스트의 스타일 모방에 관한 윤리적 문제
- 딥페이크 수준의 시각적 조작에 대한 오용 가능성
- 훈련 데이터의 편향이 생성 모션에 미치는 영향

**⑥ 다양한 스케치 표현에 대한 강건성**

TU-Berlin 데이터셋 실험에서 확인된 것처럼, 다른 스케치 표현 스타일에서는 추가 하이퍼파라미터 튜닝이 필요합니다. 표현 방식에 무관한 강건한 방법 개발이 필요합니다.

**⑦ 3D로의 확장**

2D 스케치 애니메이션을 넘어, 3D 스케치나 와이어프레임 모델의 애니메이션으로 확장하는 연구도 유망한 방향입니다.

---

## 참고 자료

**논문 원문:**
- Gal, R., Vinker, Y., Alaluf, Y., Bermano, A., Cohen-Or, D., Shamir, A., & Chechik, G. (2023). *Breathing Life Into Sketches Using Text-to-Video Priors*. arXiv:2311.13608v1.

**주요 인용 문헌 (논문 내 참조):**
- Poole, B., Jain, A., Barron, J. T., & Mildenhall, B. (2022). *DreamFusion: Text-to-3D using 2D diffusion*. ICLR 2023.
- Vinker, Y., et al. (2022). *CLIPasso: Semantically-aware object sketching*. ACM Trans. Graph.
- Wang, J., et al. (2023). *ModelScope text-to-video technical report*. arXiv:2308.06571.
- Li, T.-M., et al. (2020). *Differentiable vector graphics rasterization for editing and learning*. ACM Trans. Graph.
- Smith, H. J., et al. (2023). *A method for animating children's drawings of the human figure*. ACM Transactions on Graphics.
- Jain, A., Xie, A., & Abbeel, P. (2022). *VectorFusion: Text-to-SVG by abstracting pixel-based diffusion models*. arXiv.
- Geyer, M., et al. (2023). *TokenFlow: Consistent diffusion features for consistent video editing*. arXiv:2307.10373.
- Chen, H., et al. (2023). *VideoCrafter1: Open diffusion models for high-quality video generation*. arXiv.
- Igarashi, T., Moscovich, T., & Hughes, J. F. (2005). *As-rigid-as-possible shape manipulation*. ACM Trans. Graph.

**프로젝트 페이지:** https://livesketch.github.io/
