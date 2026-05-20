
# Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion (DRF)

> **논문 정보**
> - **저자**: Jiwon Kim, Pureum Kim, SeonHwa Kim, Soobin Park, Eunju Cha, Kyong Hwan Jin
> - **소속**: Korea University, Sookmyung Women's University
> - **발표**: ICCV 2025
> - **arXiv ID**: [2508.09575](https://arxiv.org/abs/2508.09575)
> - **코드**: https://github.com/jwonkm/DRF

---

## 1. 핵심 주장 및 주요 기여 요약

Ctrl-X, FreeControl 등 최근의 controllable T2I diffusion 모델들은 보조 모듈 학습 없이도 강력한 공간적·외형적 제어 능력을 보여주었으나, 이러한 모델들은 공간 구조를 정확하게 보존하지 못하고 객체 포즈 및 장면 레이아웃과 관련된 세밀한 조건을 포착하는 데 실패하는 문제가 있다.

이를 해결하기 위해, 본 논문은 다음을 핵심 기여로 제안합니다:

| 기여 | 내용 |
|---|---|
| **Training-free 설계** | 추가적인 모델 학습 없이 기존 T2I 모델에 플러그인 방식으로 적용 가능 |
| **Dual Recursive Feedback (DRF)** | Appearance Feedback + Generation Feedback의 이중 피드백 구조 |
| **Class-Invariant 제어** | 서로 다른 범주 간(e.g., 인간 동작 → 호랑이)의 포즈/구조 이전 |
| **Latent Manifold 안정화** | 중간 latent를 신뢰할 수 있는 매니폴드로 안내하는 업데이트 메커니즘 |

DRF는 appearance feedback과 generation feedback으로 구성되며, 중간 latent를 재귀적으로 정제하여 주어진 외형 정보와 사용자의 의도를 더 잘 반영한다. 이 이중 업데이트 메커니즘은 latent 표현을 신뢰할 수 있는 매니폴드로 안내하여 구조 및 외형 속성을 효과적으로 통합하며, 인간의 동작을 호랑이의 형태로 이전하는 등 클래스-불변 구조-외형 융합에서도 세밀한 생성을 가능하게 한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

ControlNet은 사전 학습된 T2I diffusion 모델에 공간 조건부 제어를 추가하여 레이아웃, 포즈, 형태 등의 제약 조건을 반영할 수 있지만, 다수의 공간 요소를 제어할 때마다 여러 모델을 재학습해야 하므로 모델 수와 훈련 오버헤드가 증가한다.

반면 FreeControl은 사전 학습된 T2I 모델에 추가 훈련이 필요하지 않지만, latent 공간에서 손실을 최소화하기 위한 그래디언트 기반 조작에만 의존하는 한계가 있다.

기존 방법들은 모델을 매번 재학습해야 하거나 두 이미지 간의 범주적 유사성에 의존하는 문제를 가지며, 구조와 외형의 범주가 크게 다르거나 구조적 불일치가 심한 클래스-불변(class-invariant) 설정에서는 적절한 이미지를 생성하지 못할 수 있다.

---

### 2-2. 제안하는 방법 (수식 포함)

DRF는 **두 가지 피드백 루프**를 재귀적으로 적용하여 diffusion의 중간 latent를 정제합니다.

#### 📌 기본 Diffusion 모델 배경

표준 DDPM/DDIM 기반의 latent diffusion에서, 시간 단계 $t$에서의 denoising 과정은 다음과 같습니다:

$$\mathbf{z}_{t-1} = f\bigl(\mathbf{z}_t,\, \epsilon_\theta(\mathbf{z}_t, t, c)\bigr)$$

여기서 $\mathbf{z}\_t$는 시간 $t$에서의 noisy latent, $\epsilon_\theta$는 denoising network (e.g., UNet), $c$는 텍스트 조건입니다.

#### 📌 DRF의 핵심: Dual Feedback

DRF는 appearance feedback과 generation feedback을 통해 guided noise $\epsilon_\theta^g$를 획득함으로써 반복적으로 업데이트하며, 이 두 noise로부터 유도된 distillation 함수가 결합되어 generation latent를 업데이트한다.

공개된 자료를 기반으로 DRF의 두 피드백을 수식으로 정리하면 다음과 같습니다:

**① Appearance Feedback**

외형 이미지 $\mathbf{z}^a$에 맞게 appearance latent를 정제합니다:

$$\mathcal{L}_{\text{app}} = d\!\left(\mathbf{z}_{0|t}^a,\, \mathbf{z}_{\text{ref}}^a\right)$$

여기서 $\mathbf{z}\_{0|t}^a$는 시간 $t$에서 예측된 clean appearance latent이고, $\mathbf{z}_{\text{ref}}^a$는 참조 외형 이미지의 latent, $d(\cdot, \cdot)$는 거리 함수입니다.

**② Generation Feedback**

Generation feedback은 이전의 이상적인 generation latent에 denoising 궤적을 정렬합니다:

$$\mathcal{L}_{\text{gen}} = d\!\left(\mathbf{z}_{0|t}^g,\, \mathbf{z}_{\text{prev}}^g\right)$$

지수 가중치(exponential weighting) 방식이 적용됩니다.

**③ 결합된 Distillation 업데이트**

두 피드백의 결합 업데이트:

$$\mathbf{z}_{t-1}^g \leftarrow \mathbf{z}_{t-1}^g - \lambda_a \nabla_{\mathbf{z}} \mathcal{L}_{\text{app}} - \lambda_g \nabla_{\mathbf{z}} \mathcal{L}_{\text{gen}}$$

여기서 $\lambda_a$, $\lambda_g$는 각 피드백의 강도를 조절하는 가중치입니다. 이 과정이 각 diffusion timestep에서 **재귀적(recursive)**으로 반복됩니다.

> ⚠️ **주의**: 위 수식은 논문에서 공개된 arXiv HTML 및 emergentmind 자료에 기반한 것입니다. 세부 notation(e.g., 지수 가중치의 정확한 스케줄)은 전체 논문 본문에서 확인이 필요합니다.

---

### 2-3. 모델 구조

DRF는 각 제어 구성요소에 재귀적 피드백을 제공하는 training-free T2I diffusion 프레임워크로서, 외형과 구조 모두를 효과적으로 제어한다. 이 피드백 메커니즘은 최종 생성 출력이 외형 이미지와 텍스트 조건부 score 모두와 일관되게 정렬되도록 보장한다.

전체 파이프라인 구조는 다음과 같이 요약됩니다:

```
[입력]
  - Structure Image (I_s): 포즈/구조 정보 (mesh, skeleton, canny 등)
  - Appearance Image (I_a): 외형/스타일 정보
  - Text Prompt (c): 텍스트 조건

[DRF 파이프라인]
  ┌──────────────────────────────────────┐
  │ Diffusion Denoising Step (t → t-1)   │
  │  ┌─────────────────┐                 │
  │  │ Appearance      │ ← L_app 피드백  │
  │  │ Feedback Loop   │                 │
  │  └─────────────────┘                 │
  │  ┌─────────────────┐                 │
  │  │ Generation      │ ← L_gen 피드백  │
  │  │ Feedback Loop   │                 │
  │  └─────────────────┘                 │
  │  → 두 피드백 결합 → z_t-1 업데이트  │
  └──────────────────────────────────────┘
  (위 과정이 T 스텝 동안 재귀 반복)

[출력]
  - 구조 + 외형이 모두 반영된 고품질 이미지
```

DRF는 이중 피드백 전략을 통해 외형 충실도 손실과 다양한 데이터셋에 걸쳐 안정적인 이미지 생성이 불가능한 기존 T2I 방법의 오랜 문제를 해결하며, 포즈 이전 및 클래스-불변 데이터셋에서의 이미지 합성 등의 작업에서 구조 보존 및 의도된 외형 반영에 있어 강력한 성능을 보인다.

---

### 2-4. 성능 향상

평가 지표로는 다양한 구조(Mesh, Pose, Point cloud, Canny)에 대해 DINO-ViT self-similarity (낮을수록 좋음 ↓), CLIP (높을수록 좋음 ↑), DINO-I (높을수록 좋음 ↑)가 사용되었다.

user study 결과에서도 DRF는 텍스트 프롬프트와의 정렬, 구조 세부 사항 보존, 외형 특성 유지 등 모든 카테고리에서 모든 baselines를 능가하였다.

스타일 이미지에 대한 정성적 결과에서도 DRF는 외형 이미지 스타일을 구조 이미지에 반영하는 데 있어 baselines보다 뛰어난 성능을 보였다.

**비교 대상 baselines:**

| 방법 | 특징 |
|---|---|
| **ControlNet** | Training-based 공간 제어 |
| **FreeControl** | Training-free, gradient 기반 |
| **Ctrl-X** | Training-free, guidance-free |
| **DRF (제안)** | Training-free, dual recursive feedback |

---

### 2-5. 한계

논문에서 명시적으로 언급된 한계로 추정되는 사항은 다음과 같습니다 (공개 자료 기반):

1. **추론 속도**: 재귀적 피드백으로 인해 일반 단일 패스 추론보다 연산 비용이 높을 수 있습니다. training-free 접근 방식들은 훈련 비용은 피하지만, 추론 단계에서 추가적인 역전파로 인해 계산 시간 및 GPU 메모리 요구량이 크게 증가하며, 샘플링 단계가 2–20배 더 길어질 수 있다. (DRF도 이 범주에 해당할 수 있습니다.)

2. **의존성**: DRF는 Ctrl-X, FreeControl 등 기존 T2I 프레임워크 위에서 동작하므로, 기반 모델의 성능에 종속됩니다.

3. **정량적 수치의 세부 미공개**: 전체 수치 비교표는 ICCV 2025 논문 본문에서만 확인 가능합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문의 일반화 성능 향상 가능성은 다음 측면에서 두드러집니다:

### 3-1. Class-Invariant 설정에서의 일반화

DRF의 이중 업데이트 메커니즘은 latent 표현을 신뢰할 수 있는 매니폴드로 안내하여 구조 및 외형 속성을 효과적으로 통합하며, 인간의 동작을 호랑이의 형태로 이전하는 것과 같이 클래스 불변 구조-외형 융합에서도 세밀한 생성을 가능하게 한다.

DRF는 구조(mesh, skeleton)를 기반으로 강력한 포즈 생성을 보여주며, 인간 포즈 구조를 펭귄 외형 이미지에 적용하여 주어진 펭귄 외형과 의미론적으로 일관된 이미지를 생성하고, 인간 skeleton 기반 구조를 호랑이에 적용하여 다양한 포즈 간 모션 합성의 적응성을 시연한다.

### 3-2. 다양한 구조 입력 타입에 대한 범용성

DRF는 스케치, 메시(mesh), 포즈, 3D 등 다양한 구조 입력에 대해 5개의 baseline과 비교 평가되었다. 이는 단일 구조 타입에 의존하지 않는 범용적 일반화를 시사합니다.

### 3-3. Training-free 특성이 주는 일반화 이점

DRF는 각 제어 구성요소에 재귀적 피드백을 제공하는 training-free T2I diffusion 프레임워크로, 최종 생성 출력이 외형 이미지와 텍스트 조건부 score 모두와 일관되게 정렬된 고품질 이미지를 생성한다. Training-free이기 때문에 다양한 도메인의 사전 학습 모델에 즉시 적용 가능하여, 특정 데이터셋에 과적합되지 않는 일반화된 성능을 기대할 수 있습니다.

이 접근 방식은 클래스 불변 구조-외형 융합에서도 세밀한 생성을 가능하게 하며, 광범위한 실험을 통해 고품질의 의미론적으로 일관되고 구조적으로 일관된 이미지 생성에서의 효과성을 검증하였다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 앞으로의 연구에 미치는 영향

**① Training-free Feedback 패러다임의 확장**

DRF는 별도 모듈 학습 없이 재귀적 피드백만으로 고품질 제어를 달성했습니다. 이는 추론 시점의 latent 정제(inference-time latent refinement)라는 연구 방향에 중요한 선례가 됩니다. 향후 Video Diffusion, 3D Generation, Multi-view Synthesis로의 확장 가능성이 열려 있습니다.

**② Class-Invariant 생성의 새로운 기준**

DRF는 이중 피드백 전략을 통해 외형 충실도 손실과 다양한 데이터셋에 걸쳐 안정적 이미지 생성이 불가능한 기존 T2I 방법의 오랜 문제를 해결한다. 이는 크로스-도메인 합성 연구의 새로운 벤치마크 기준점이 될 수 있습니다.

**③ 관련 최신 연구 비교 분석 (2020년 이후)**

| 연구 | 연도 | 특징 | DRF와의 비교 |
|---|---|---|---|
| **ControlNet** (Zhang et al.) | 2023 | Training-based, 구조 조건부 제어 | DRF는 재학습 불필요 |
| **FreeControl** (Mo et al.) | 2024 (CVPR) | Training-free, 선형 subspace 활용 | DRF는 dual feedback으로 포즈 충실도 향상 |
| **Ctrl-X** | 2024 | Training-free, guidance-free | DRF는 재귀 피드백으로 세밀한 포즈 제어 추가 |
| **DRF (본 논문)** | 2025 (ICCV) | Training-free, dual recursive feedback | Class-invariant 설정에서 SOTA |

FreeControl은 중간 diffusion feature의 선형 subspace를 모델링하고 이 subspace에서 guidance를 활용하는 training-free controllable T2I 생성 방법으로, 스케치, 노멀 맵, 뎁스 맵, 엣지 맵, 인간 포즈 등 다양한 제어 조건과 SD 1.5, 2.1, SDXL 등의 모델 아키텍처를 지원하는 최초의 범용 training-free 솔루션이다.

FreeControl 등의 접근 방식은 훈련 데이터가 필요하지 않지만 latent의 그래디언트 계산이 필요하여 상당한 계산 시간과 GPU 메모리가 필요한 반면, Ctrl-X는 추론 단계에서 guidance 없이 직접 feature injection으로 구조를 제어하여 더 빠르고 강력한 이미지 생성을 가능하게 한다.

---

### 4-2. 향후 연구 시 고려할 점

1. **계산 효율성 최적화**: 재귀적 피드백은 반복 연산을 필요로 하므로, 효율적인 early-exit 조건이나 adaptive step 수 조절이 중요한 연구 주제가 됩니다.

2. **피드백 가중치 자동화**: 현재 $\lambda_a$, $\lambda_g$ 등의 가중치 조정이 수동적일 경우, 이를 자동으로 결정하는 meta-learning 또는 adaptive 방법이 필요합니다.

3. **비디오/3D로의 확장**: 현재 2D 이미지에 국한된 DRF를 시간 축을 포함한 비디오 생성 또는 NeRF/3DGS 기반 3D 생성으로 확장하는 연구가 유망합니다.

4. **다중 조건 통합**: 포즈 외에도 깊이(depth), 법선 벡터(normal), 세그멘테이션 등 다중 구조 조건을 동시에 처리하는 확장이 필요합니다.

5. **공정한 비교 프로토콜**: Class-invariant 설정의 벤치마크가 아직 표준화되지 않았으므로, 향후 연구에서는 일관된 평가 프로토콜 수립이 선행되어야 합니다.

6. **이론적 수렴 보장**: 재귀적 업데이트가 안정적으로 수렴하는지에 대한 이론적 분석이 후속 연구에서 요구됩니다.

---

## 📚 참고 자료 및 출처

| # | 제목 / 출처 | 비고 |
|---|---|---|
| 1 | **[arXiv:2508.09575]** Kim et al., "Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion", 2025 | 본 논문 (주 참고) |
| 2 | **[ICCV 2025 Poster]** https://iccv.thecvf.com/virtual/2025/poster/1781 | ICCV 2025 공식 포스터 |
| 3 | **[ICCV 2025 PDF]** openaccess.thecvf.com/content/ICCV2025/papers/Kim_Dual_Recursive_Feedback... | ICCV 2025 공개 논문 PDF |
| 4 | **[arXiv:2406.07540]** "Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance" | 비교 baseline |
| 5 | **[CVPR 2024]** Mo et al., "FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model" | 비교 baseline |
| 6 | **[emergentmind.com]** Diffusion-DRF topic overview (https://www.emergentmind.com/topics/diffusion-drf) | 수식 참고 |
| 7 | **[GitHub]** https://github.com/jwonkm/DRF | 공식 코드 |

> ⚠️ **정확도 관련 고지**: 논문의 세부 수식(특히 지수 가중치 스케줄, 정확한 ablation 수치, 전체 아키텍처 다이어그램)은 ICCV 2025 논문 전문에서만 완전히 확인 가능합니다. 위 수식은 공개된 arXiv HTML(arxiv.org/html/2508.09575v1) 및 emergentmind 요약 자료에 기반한 것으로, 일부 표기는 원문 notation과 다를 수 있습니다. 완전한 수식 확인을 위해서는 논문 원문을 직접 참조하시기 바랍니다.
