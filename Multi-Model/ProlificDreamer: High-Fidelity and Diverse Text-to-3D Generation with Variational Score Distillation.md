# ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation

---

## 1. 핵심 주장 및 주요 기여 요약

Score Distillation Sampling(SDS)은 사전학습된 대규모 text-to-image 확산 모델을 증류하여 text-to-3D 생성에서 큰 가능성을 보여주었으나, **과포화(over-saturation), 과평활화(over-smoothing), 낮은 다양성(low-diversity)** 문제를 겪고 있다.

이 논문의 핵심 주장과 기여는 다음과 같다:

1. **Variational Score Distillation (VSD) 제안**: 3D 파라미터를 SDS에서처럼 상수(constant)가 아닌 **확률변수(random variable)**로 모델링하고, 원칙적인 파티클 기반 변분 프레임워크(particle-based variational framework)인 VSD를 제시하여 text-to-3D 생성의 문제를 설명하고 해결한다.

2. **SDS가 VSD의 특수 사례임을 증명**: SDS는 VSD의 특수한 경우이며, 작은 CFG 가중치와 큰 CFG 가중치 모두에서 낮은 품질의 샘플을 생성한다.

3. **다양한 CFG에서의 안정적 성능**: VSD는 확산 모델의 ancestral sampling과 같이 다양한 CFG 가중치에서 잘 작동하며, 일반적인 CFG 가중치(7.5)로 다양성과 샘플 품질을 동시에 향상시킨다.

4. **설계 공간의 체계적 탐구**: 증류 시간 스케줄(distillation time schedule), 밀도 초기화(density initialization) 등 text-to-3D 설계 공간에서 증류 알고리즘과 직교하면서도 충분히 탐구되지 않은 다양한 개선 사항을 제시한다.

5. **고해상도·고충실도 결과**: ProlificDreamer는 $512 \times 512$의 높은 렌더링 해상도와 풍부한 구조, 복잡한 효과(연기, 물방울 등)를 가진 고충실도 NeRF를 생성할 수 있으며, NeRF에서 초기화된 메쉬를 VSD로 미세조정하면 세밀하고 사실적인 결과를 얻는다.

> 이 논문은 **NeurIPS 2023 Spotlight**으로 채택되었으며, 칭화대학교(Tsinghua University)의 Zhengyi Wang 등이 저술하였다.

---

## 2. 상세 분석: 문제, 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

SDS는 종종 과포화, 과평활화, 낮은 다양성 문제를 겪으며, 이에 대해 아직 충분히 설명되거나 적절히 해결되지 않았다. 또한 렌더링 해상도, 증류 시간 스케줄 등 text-to-3D의 설계 공간에서 직교적인 요소들이 충분히 탐구되지 않아 상당한 개선 가능성이 남아있다.

구체적으로:
- DreamFusion이 제안한 SDS 알고리즘은 과포화, 과평활화, 디테일 부족 등 심각한 문제에 직면한다.
- SDS는 종종 매우 큰 CFG 가중치(CFG=100)를 요구하며, VSD는 정상적인 CFG(=7.5)를 사용할 수 있는 최초의 알고리즘이다.

### 2.2 제안하는 방법: Variational Score Distillation (VSD)

#### (1) 문제 정의 및 변분 공식화

VSD는 텍스트 프롬프트가 주어진 3D 장면을 SDS처럼 단일 포인트가 아닌 확률변수로 취급한다. VSD는 모든 뷰에서 렌더링된 이미지에 의해 유도되는 분포가 사전학습된 2D 확산 모델에 의해 정의된 분포와 KL 발산(KL divergence) 측면에서 가능한 한 밀접하게 정렬되도록 3D 장면의 분포를 최적화한다. 이 변분 공식화 하에서 VSD는 하나의 프롬프트에 대해 여러 3D 장면이 잠재적으로 정렬될 수 있다는 현상을 자연스럽게 특성화한다.

**VSD의 최적화 목적함수**는 다음과 같이 정의된다:

$$\mu^* = \arg\min_{\mu} \mathbb{E}_{t, c} \left[ \frac{\sigma_t}{\alpha_t} \omega(t) \cdot D_{\text{KL}}\left( q_t^{\mu}(x_t | c, y) \,\|\, p_t(x_t | y) \right) \right]$$

여기서:
- $\mu$: 3D 파라미터 $\theta$에 대한 변분 분포 (variational distribution)
- $q_t^{\mu}(x_t | c, y)$: 현재 3D 분포 $\mu$로부터 렌더링된 이미지에 노이즈를 추가한 diffused 분포
- $p_t(x_t | y)$: 사전학습된 2D 확산 모델이 정의하는 목표 분포
- $c$: 카메라 포즈
- $\omega(t)$: 시간 의존 가중치 함수
- $\alpha_t, \sigma_t$: 노이즈 스케줄 파라미터

#### (2) Wasserstein Gradient Flow를 통한 파티클 업데이트

VSD는 효율적으로 문제를 풀기 위해 파티클 기반 변분 추론(particle-based variational inference)을 채택하며, 3D 파라미터들의 집합을 파티클로 유지하여 3D 분포를 표현한다. Wasserstein gradient flow를 통해 파티클에 대한 원칙적인 그래디언트 기반 업데이트 규칙을 유도하며, 최적화가 수렴할 때 파티클이 원하는 분포의 샘플이 될 것을 보장한다(Theorem 2).

**VSD의 그래디언트 업데이트** (핵심 수식, Eq. (9)):

$$\nabla_{\theta} \mathcal{L}_{\text{VSD}} = \mathbb{E}_{t, \epsilon, c} \left[ \omega(t) \left( \epsilon_{\text{pretrain}}(x_t, t, y) - \epsilon_{\phi}(x_t, t, c, y) \right) \frac{\partial x}{\partial \theta} \right]$$

여기서:
- $\epsilon_{\text{pretrain}}(x_t, t, y)$: 사전학습된 확산 모델(예: Stable Diffusion)의 노이즈 예측
- $\epsilon_{\phi}(x_t, t, c, y)$: 변분 분포의 score를 추정하는 추가 확산 모델 (LoRA로 파라미터화)
- $x = g(\theta, c)$: 3D 파라미터 $\theta$를 카메라 포즈 $c$에서 미분 가능하게 렌더링한 이미지
- $x_t = \alpha_t x + \sigma_t \epsilon$: 노이즈가 추가된 이미지

**비교: SDS의 그래디언트** (Eq. (3)):

$$\nabla_{\theta} \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon, c} \left[ \omega(t) \left( \epsilon_{\text{pretrain}}(x_t, t, y) - \epsilon \right) \frac{\partial x}{\partial \theta} \right]$$

핵심 차이점: SDS는 변분 분포를 단일 포인트 디랙 분포 $\mu(\theta|y) \approx \delta(\theta - \theta^1)$로 사용하는 VSD의 특수한 경우이다. SDS에서 $\epsilon_{\phi}$가 순수 노이즈 $\epsilon$으로 대체된다.

#### (3) LoRA 기반 Score 추정

VSD의 업데이트는 diffused 렌더링 이미지 분포의 score 함수 추정을 요구하며, 이는 사전학습된 확산 모델의 Low-Rank Adaptation(LoRA)으로 효율적이고 효과적으로 구현된다.

LoRA의 학습 목적함수:

$$\min_{\phi} \mathbb{E}_{t, \epsilon} \left[ \| \epsilon_{\phi}(x_t, t, c, y) - \epsilon \|_2^2 \right]$$

각 반복(iteration)에서 미분 가능 렌더링을 한 번만 수행하지만, 렌더링된 이미지 $x_0 = g(\theta, c)$를 VSD 그래디언트 계산과 LoRA 학습 두 가지 모두에 사용한다.

최종 알고리즘은 파티클 업데이트와 score 함수 업데이트를 교대로 수행한다.

### 2.3 모델 구조 및 파이프라인

ProlificDreamer의 전체 파이프라인은 **3단계**로 구성된다:

| 단계 | 내용 | 세부사항 |
|------|------|----------|
| **Stage 1** | NeRF 학습 (VSD) | $512 \times 512$ 렌더링 해상도에서 VSD guidance로 NeRF를 25,000 iteration 학습 |
| **Stage 2** | 메쉬 추출 (SDS) | DMTet을 사용하여 Stage 1에서 얻은 NeRF로부터 텍스처 메쉬를 추출하고 미세조정 |
| **Stage 3** | 텍스처 미세조정 (VSD) | VSD guidance로 메쉬 텍스처를 $512 \times 512$ 해상도에서 미세조정 |

**설계 공간의 주요 개선 사항:**

1. **고해상도 렌더링**: 학습 중 $512 \times 512$의 고해상도 렌더링과 annealed distilling time schedule을 제안하여 시각적 품질을 향상시킨다.

2. **Annealed Time Schedule**: 학습 초기에는 $t \in [0.02, 0.98]$의 넓은 범위에서 시작하여 대략적인 의미 구조에 집중한 후, 점진적으로 범위를 좁혀간다.

3. **Scene Initialization**: 단일 객체를 넘어서는 장면의 경우, NeRF에 대해 새로운 "scene initialization"(음의 초기 밀도, 큰 반경)을 적용하여 전체 환경에 대한 최적화를 가능하게 한다.

4. **카메라 포즈 조건화**: 카메라 포즈 $c$는 2층 MLP에 입력되어 U-Net의 각 블록에서 timestep 임베딩에 추가된다.

### 2.4 성능 향상

품질(Quality) 측면에서 ProlificDreamer는 풍부한 미세 구조, 반투명 효과, 복잡한 표면 특성 등 사실적인 3D 자산을 생성한다.

다양성(Diversity) 측면에서 단일 점 추정에서 조건부 분포 $\mu(\theta|y)$로의 전환을 통해 동일한 텍스트 프롬프트가 의미적으로 일관되면서도 시각적으로 다양한 3D 모델을 생성할 수 있다. 실험 결과 3D-FID에서 우수한 성능을 보이며, 사용자 연구에서도 DreamFusion, Magic3D, Fantasia3D에 비해 ProlificDreamer 결과물에 대한 강한 선호가 보고되었다.

Ablation 연구에서 64 렌더링 해상도와 SDS 손실의 일반적인 설정에서 출발하여, 해상도 증가, annealed time schedule 추가, VSD 추가 등 각 개선 사항이 단계적으로 생성 결과를 향상시킨다.

### 2.5 한계점

ProlificDreamer는 뛰어난 text-to-3D 결과를 달성하지만, 생성에 수 시간이 소요되어 확산 모델의 이미지 생성보다 훨씬 느리다.

대규모 장면 생성은 scene initialization으로 달성할 수 있지만, 학습 중 카메라 포즈는 장면 구조와 무관하게 설정되어 있어, 장면 구조에 따른 적응적 카메라 포즈 범위를 고안하면 더 나은 세부 표현이 가능할 것이다.

또한 기반 2D 모델의 제한된 표현력으로 인해 복잡한 프롬프트에 대한 생성이 실패할 수 있으며, 일부 경우에는 멀티페이스 Janus 문제가 발생할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 VSD의 일반화 우수성의 이론적 근거

VSD는 잠재적으로 여러 파티클을 사용할 뿐만 아니라, 단일 파티클(n=1)에서도 파라메트릭 score 함수 $\epsilon_{\phi}$를 학습한다. 경험적으로 학습된 신경망은 SDS의 디랙 분포에 비해 잠재적으로 우수한 일반화 능력을 제공할 수 있으므로, 더 정확한 업데이트 방향을 제공할 수 있다.

이를 수식으로 설명하면:

- **SDS**: $\epsilon_{\phi} \to \epsilon$ (순수 노이즈로 대체, 일반화 능력 없음)
- **VSD**: $\epsilon_{\phi}(x_t, t, c, y)$ (학습된 LoRA가 현재 분포를 추정, 일반화 가능)

$$\underbrace{\epsilon_{\text{pretrain}}(x_t, t, y) - \epsilon}_{\text{SDS: 거친 그래디언트}} \quad \text{vs.} \quad \underbrace{\epsilon_{\text{pretrain}}(x_t, t, y) - \epsilon_{\phi}(x_t, t, c, y)}_{\text{VSD: 정밀한 그래디언트}}$$

### 3.2 일반화 성능을 뒷받침하는 실증적 증거

SDS는 작은 CFG와 큰 CFG 모두에서 과포화되고 과평활화된 이미지를 생성하는 반면, VSD는 CFG 7.5에서 확산 모델의 ancestral sampling과 유사한 사실적인 샘플을 생성한다. 이는 VSD의 원칙적인 변분 공식화와 파라메트릭 score 모델 학습 능력이 SDS의 단일 포인트 디랙 분포 근사에 비해 우수한 일반화와 충실도에 기여함을 나타낸다.

SDS는 과포화되고 과평활화된 그래디언트를 제공하는 반면, VSD는 더 자연스러운 외관의 그래디언트와 더 많은 디테일을 제공하며, 결과적으로 VSD는 더 나은 최종 결과를 제공한다.

### 3.3 다중 파티클과 일반화

SDS와 비교하여, VSD는 주어진 프롬프트에 대해 여러 가능한 3D 장면 표현을 자연스럽게 수용하는 변분 분포 접근법을 활용한다. 다양한 CFG 가중치를 효과적으로 사용할 수 있는 능력이 SDS의 한계를 해결하여 정상적인 CFG 설정에서도 사실적인 렌더링을 가능하게 한다.

### 3.4 일반화 성능 향상을 위한 향후 방향

- **렌더링 해상도 확장**: VSD는 낮은 학습 해상도(128 또는 256)에서도 경쟁력 있는 결과를 제공하여, 512 해상도보다 계산 효율적이다. → 해상도에 따른 일반화 로버스트니스
- **적응적 카메라 전략**: 장면 구조에 따른 적응적 카메라 포즈 범위를 설계하면 더 나은 세부 표현을 얻을 수 있다.
- **기반 2D 모델 강화**: 기반 2D 모델의 제한된 표현력 때문에 복잡한 프롬프트에서 실패하거나 Janus 문제가 발생할 수 있으므로, 보다 강력한 2D 기반 모델(예: SDXL, SD3)과의 결합이 일반화 성능을 더 향상시킬 수 있다.

---

## 4. 연구 영향 및 향후 고려할 점

### 4.1 학계에 미치는 영향

DreamFusion이 SDS를 고용량 2D 확산 prior와 3D 장면 복원 사이의 일반적인 인터페이스로 확립하여 text-to-3D 합성, 효율성, 신뢰성의 빠른 발전을 촉진했다면, ProlificDreamer는 이 SDS 패러다임의 **이론적 한계를 명확히 규명**하고 **변분 추론 기반의 새로운 대안**을 제시했다는 점에서 더 근본적인 기여를 한다.

고충실도, 다양한, 복잡한 3D 장면을 생성하는 능력은 게이밍, 애니메이션, 가상 현실 등 다양한 분야의 응용에 새로운 길을 열었다.

### 4.2 향후 연구 시 고려할 점

1. **생성 속도 가속화**: 품질을 저해하지 않으면서 보다 빠른 text-to-3D 합성을 위해 ProlificDreamer를 최적화하는 것이 주요 과제이다.

2. **장면 이해 및 카메라 포지셔닝**: 생성 과정의 가속화와 더불어, 더욱 세밀한 장면 렌더링을 위한 장면 이해 및 카메라 포지셔닝 통합의 정교화에 초점을 맞출 수 있다.

3. **Janus 문제 해결**: MVDream과 VSD를 결합하여 멀티페이스 문제를 완화하는 접근이 이미 시도되고 있다.

4. **새로운 확산 모델 통합**: ProlificDreamer(VSD)가 개선된 품질을 제공하지만 최적화에 훨씬 더 긴 시간을 필요로 하며, 선형 궤적을 나타내는 간단한 ODE를 활용하는 rectified flow 모델이 대안적 prior로서 가능성을 보여주고 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | ProlificDreamer와의 비교 |
|------|------|-----------|--------------------------|
| **DreamFusion** (Poole et al.) | 2022 | 확률 밀도 증류(probability density distillation) 기반 SDS 손실을 도입하여 2D 확산 모델을 NeRF 최적화의 prior로 사용 | VSD의 이론적 기반이 됨; 과포화/과평활화 문제 존재 |
| **Magic3D** (Lin et al.) | 2022 | 두 단계 최적화 프레임워크를 활용: 저해상도 확산 prior로 coarse 모델을 얻고, 고해상도 latent diffusion으로 텍스처 메쉬를 최적화 | DreamFusion보다 2배 빠르고 더 높은 해상도를 달성하지만, SDS 기반으로 품질 한계 공유 |
| **Score Jacobian Chaining (SJC)** | 2023 | 추정된 score에 chain rule을 적용하여 생성 품질 향상 | VSD와 다른 접근이지만 SDS 변형; ProlificDreamer가 성능 면에서 우위 |
| **LucidDreamer (ISM loss)** | 2023 | 확산 궤적에서 ISM 손실을 제안하지만 같은 스케일의 많은 항을 제거 | SDS 변형 중 하나; VSD의 원칙적 접근과 대비 |
| **Consistent3D** | 2024 | 일관성 증류 샘플링 방법을 설계하여 3D 모델을 학습하지만, 품질 향상이 제한적 | VSD 대비 품질 개선 폭이 작음 |
| **FlowDreamer** | 2024 | Rectified flow 모델을 활용하여 시간 독립적 벡터 필드를 학습함으로써 SDS 프레임워크의 시간 의존적 score로 인한 3D 모델 업데이트 그래디언트의 모호성을 줄이고자 함 | SDS/VSD와 직교하는 새로운 prior 모델 탐구 |
| **DreamCS** | 2025 | GPTEval3D 벤치마크에서 DreamCS가 MVDream, DreamFusion, Magic3D 등의 기하학적 정렬 및 메쉬 품질을 일관되게 향상시킴 | 3D reward guidance를 통한 기하학적 품질 개선에 초점 |
| **Fantasia3D** | 2023 | 기하학과 외관을 분리하여 최적화 | 사용자 연구에서 ProlificDreamer가 Fantasia3D보다 선호됨 |
| **MVDream** | 2023 | Multi-view diffusion prior 사용 | ProlificDreamer의 VSD와 결합하여 Janus 문제 완화에 활용 |

### 종합 비교: 핵심 수식 대조

| 방법 | 그래디언트 수식 |
|------|----------------|
| **SDS** | $\nabla_{\theta}\mathcal{L}\_{\text{SDS}} = \mathbb{E}\_{t,\epsilon,c}\left[\omega(t)\left(\epsilon_{\text{pretrain}}(x_t, t, y) - \epsilon\right)\frac{\partial x}{\partial \theta}\right]$ |
| **VSD (본 논문)** | $\nabla_{\theta}\mathcal{L}\_{\text{VSD}} = \mathbb{E}\_{t,\epsilon,c}\left[\omega(t)\left(\epsilon_{\text{pretrain}}(x_t, t, y) - \epsilon_{\phi}(x_t, t, c, y)\right)\frac{\partial x}{\partial \theta}\right]$ |

SDS에서 $\epsilon$ (순수 노이즈) → VSD에서 $\epsilon_{\phi}$ (학습된 score 추정)으로의 전환이 핵심적 차이이며, 이것이 일반화 성능과 품질 향상의 근본적 원인이다.

---

## 참고 자료 및 출처

1. **Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., & Zhu, J. (2023).** "ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation." *NeurIPS 2023 (Spotlight).* arXiv:2305.16213
   - 논문: https://arxiv.org/abs/2305.16213
   - 프로젝트 페이지: https://ml.cs.tsinghua.edu.cn/prolificdreamer/
   - GitHub: https://github.com/thu-ml/prolificdreamer
   - NeurIPS PDF: https://proceedings.neurips.cc/paper_files/paper/2023/file/1a87980b9853e84dfb295855b425c262-Paper-Conference.pdf

2. **Poole, B., Jain, A., Barron, J.T., & Mildenhall, B. (2022).** "DreamFusion: Text-to-3D using 2D Diffusion." arXiv:2209.14988
   - https://arxiv.org/abs/2209.14988

3. **Lin, C.H. et al. (2023).** "Magic3D: High-Resolution Text-to-3D Content Creation." CVPR 2023.
   - https://arxiv.org/abs/2211.10440

4. **FlowDreamer (2024).** "Exploring High Fidelity Text-to-3D Generation Via Rectified Flow."
   - https://arxiv.org/abs/2408.05008

5. **DreamCS (2025).** "Geometry-Aware Text-to-3D Generation with Unpaired 3D Reward Supervision."
   - https://arxiv.org/abs/2506.09814

6. **OpenReview - ProlificDreamer**: https://openreview.net/forum?id=ppJuFSOAnM

7. **EmergentMind - ProlificDreamer Topic**: https://www.emergentmind.com/topics/prolificdreamer

8. **ar5iv HTML 버전**: https://ar5iv.labs.arxiv.org/html/2305.16213

9. **MarkTechPost 리뷰**: https://www.marktechpost.com/2023/05/31/meet-prolificdreamer-an-ai-approach-that-delivers-high-fidelity-and-realistic-3d-content-using-variational-score-distillation-vsd/

10. **Liner Quick Review**: https://liner.com/review/prolificdreamer-highfidelity-and-diverse-textto3d-generation-with-variational-score-distillation
