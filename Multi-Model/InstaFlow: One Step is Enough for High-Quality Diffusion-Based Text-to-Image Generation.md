
# InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation

> **논문 정보**: Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, Qiang Liu (University of Texas at Austin / Helixon Research)
> **발표**: ICLR 2024
> **arXiv**: [2309.06380](https://arxiv.org/abs/2309.06380)
> **GitHub**: [gnobitab/InstaFlow](https://github.com/gnobitab/InstaFlow)

---

## 1. 핵심 주장 및 주요 기여 요약

Diffusion 모델은 텍스트-이미지 생성에서 탁월한 품질과 창의성으로 혁신을 이끌었지만, 다단계 샘플링 프로세스가 느리다는 근본적인 문제가 있으며, 만족스러운 결과를 얻기 위해 수십 번의 추론 단계가 필요하다.

이전의 증류(distillation) 기반 속도 개선 시도들은 기능적인 1-step 모델을 달성하는 데 실패하였으며, 이 논문은 소규모 데이터셋에만 적용되던 Rectified Flow라는 방법을 탐구한다.

### 주요 기여 3가지

| 기여 | 설명 |
|------|------|
| **1-step 생성 최초 달성** | SD 수준의 이미지 품질을 가진 최초의 1-step 텍스트-이미지 생성기 개발 |
| **대규모 Rectified Flow 파이프라인** | 소규모 데이터셋에 국한되었던 Rectified Flow를 대규모 T2I로 확장 |
| **효율적 훈련** | 순수 지도학습만으로 199 A100 GPU days라는 저비용으로 훈련 |

이 파이프라인을 통해 MS COCO 2017-5k에서 FID $23.3$을 달성하여 이전 최고 기법인 progressive distillation($37.2$)을 크게 앞질렀으며, 1.7B 파라미터의 확장 네트워크를 사용하면 FID가 $22.4$까지 향상된다.

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

단순한 Stable Diffusion의 직접 증류는 완전히 실패한다. 핵심 문제는 노이즈와 이미지 간의 최적화되지 않은 결합(sub-optimal coupling)으로, 이것이 증류 과정을 크게 방해한다.

사전 훈련된 diffusion 모델의 확률 흐름(probability flow) ODE는 **곡선 궤적(curved trajectories)**을 가지며 노이즈와 이미지 사이의 나쁜 결합을 유발한다. 텍스트 조건부 reflow로 미세조정 후, 궤적이 직선화되고 결합이 정제되어 reflow된 모델이 증류에 더 적합해지고, 결과적으로 단 한 단계에서 고품질 이미지를 생성한다.

---

### 2-2. 제안 방법: Rectified Flow + Text-Conditioned Reflow

#### (A) Rectified Flow 기본 원리

Rectified Flow는 두 분포 $\pi_0$ (가우시안 노이즈 분포)와 $\pi_1$ (이미지 분포) 사이를 ODE로 연결하는 흐름 기반 생성 모델입니다.

$\pi_0$를 가우시안 분포, $\pi_1$을 이미지 분포라 할 때, Rectified Flow는 ODE(상미분방정식) 또는 흐름 모델(flow model)을 통해 $\pi_0$를 $\pi_1$로 변환하는 것을 학습한다.

ODE 흐름의 기본 형태:

$$\frac{d\mathbf{Z}_t}{dt} = v(\mathbf{Z}_t, t), \quad t \in [0, 1]$$

여기서 $v(\cdot, t)$는 학습된 속도 필드(velocity field)입니다.

선형 보간(linear interpolation) 기반 학습 목표:

$$\mathbf{X}_t = (1 - t)\mathbf{X}_0 + t\mathbf{X}_1$$

속도 필드 학습 손실:

$$\min_v \int_0^1 \mathbb{E}\left[\left\lVert (\mathbf{X}_1 - \mathbf{X}_0) - v(\mathbf{X}_t, t) \right\rVert^2\right] dt$$

이때 $\mathbf{X}_0 \sim \pi_0$ (노이즈), $\mathbf{X}_1 \sim \pi_1$ (이미지).

#### (B) Reflow 절차

Rectified Flow의 핵심은 reflow 절차로, 확률 흐름의 궤적을 직선화하고, 노이즈와 이미지 간의 결합을 정제하며, 학생 모델의 증류 과정을 용이하게 한다.

**k-Rectified Flow** 절차:

1. **1-Rectified Flow** 훈련: SD로부터 생성된 $(\mathbf{X}_0, \mathbf{X}_1)$ 쌍으로 초기 흐름 훈련
2. **Reflow (2-Rectified Flow)**: 1-Rectified Flow로부터 새로운 쌍 $(\mathbf{Z}_0, \mathbf{Z}_1)$을 재샘플링하여 궤적을 직선화

$$(\mathbf{Z}_0, \mathbf{Z}_1) \sim \text{(1-Rectified Flow로 생성된 결합 분포)}$$

이를 통해 $\mathbf{Z}_0$와 $\mathbf{Z}_1$ 간의 transport cost가 감소하고, 궤적이 직선에 가까워집니다.

직선 흐름(straight flows)은 시뮬레이션에 필요한 단계가 적다는 장점이 있으며, 노이즈 분포와 이미지 분포 사이의 더 나은 결합을 제공하여 성공적인 증류를 가능하게 한다.

#### (C) 텍스트 조건부 CFG 속도 필드

InstaFlow는 텍스트 조건부 Rectified Flow를 위해 Classifier-Free Guidance 속도 필드를 통합하여, 기존 SD 모델과 유사하게 샘플 다양성과 생성 품질 간의 트레이드오프를 가능하게 한다.

CFG 속도 필드:

$$v_{\text{cfg}}(\mathbf{Z}_t, t, c) = v_\theta(\mathbf{Z}_t, t, \emptyset) + w \cdot \left[v_\theta(\mathbf{Z}_t, t, c) - v_\theta(\mathbf{Z}_t, t, \emptyset)\right]$$

여기서 $w$는 guidance scale, $c$는 텍스트 조건.

#### (D) 증류 (Distillation)

2-Rectified Flow를 교사(teacher) 모델로 사용하여, 단일 스텝 학생(student) 모델 훈련:

$$\mathcal{L}_{\text{distill}} = \mathbb{E}\left[\text{LPIPS}\left(f_\phi(\mathbf{Z}_0), \hat{\mathbf{X}}_1\right)\right]$$

증류에는 LPIPS loss (VGG 기반 네트워크)를 사용하여 이미지의 고수준 유사성을 포착한다.

---

### 2-3. 모델 구조 (InstaFlow-0.9B & InstaFlow-1.7B)

#### InstaFlow-0.9B
- Stable Diffusion 1.5의 U-Net 기반 (~0.9B 파라미터)
- 2-Rectified Flow 미세조정 후 단일 스텝 증류
- 추론 시간: **0.09초** (A100 GPU)

#### InstaFlow-1.7B (Stacked U-Net)

모델 크기를 확장하기 위해 두 개의 U-Net을 직렬로 쌓고, 철저한 ablation study를 통해 불필요한 모듈을 제거했다. 이를 Stacked U-Net이라 부르며, 1.7B 파라미터와 약 0.12초의 추론 시간을 가진다.

---

### 2-4. 성능 향상

| 모델 | FID (MS COCO 2017-5k) | FID (MS COCO 2014-30k) | 추론 시간 |
|------|----------------------|----------------------|----------|
| Progressive Distillation (1-step) | 37.2 | - | ~0.1s |
| StyleGAN-T | - | 13.9 | 0.1s |
| **InstaFlow-0.9B** | **23.3** | **13.1** | **0.09s** |
| **InstaFlow-1.7B** | **22.4** | - | **0.12s** |

InstaFlow 모델은 1-step 생성기로서 노이즈를 이미지로 직접 매핑하고, 다단계 샘플링을 완전히 회피한다. A100 GPU 기준 추론 시간은 약 0.1초로, 기존 Stable Diffusion 대비 약 90%의 추론 시간을 절약한다.

이는 증류된 1-step Stable Diffusion 모델이 GAN 기반 모델과 동등하거나 더 나은 성능을 발휘한 최초의 사례이다. InstaFlow-0.9B는 약 199 A100 GPU days가 소요되었으며, 이는 StyleGAN-T나 GigaGAN 등에 비해 훨씬 효율적이다.

---

### 2-5. 한계

InstaFlow는 텍스트 프롬프트의 복잡한 구성(complex compositions)에서 어려움을 겪을 수 있으며, 더 긴 훈련 기간과 더 큰 데이터셋으로의 추가 학습이 이를 완화할 가능성이 높다.

- **복잡한 구성 생성 실패**: 다수의 객체나 복잡한 공간 관계 표현이 약함
- **Reflow의 데이터 생성 비용**: 대규모 합성 데이터셋 생성이 필요하여 스토리지 및 시간 소요가 큼
- **다단계 정제 불가**: 1-step 모델 특성상 점진적 정제가 어려움

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. LoRA 호환성

One-step InstaFlow는 사전 훈련된 LoRA와 호환된다.

이는 도메인 특화 스타일 적용이 가능함을 의미하며, 다양한 스타일로의 일반화 능력을 대폭 확장합니다. 실제로 InstaFlow + dreamshaper-7 조합으로 이미지 품질이 크게 향상되었다.

### 3-2. ControlNet 호환성

One-step InstaFlow는 사전 훈련된 ControlNet과도 완전히 호환된다.

이를 통해 엣지 맵, 깊이 맵, 포즈 등 다양한 조건부 생성으로의 일반화가 가능합니다.

### 3-3. 모델 크기 스케일링에 의한 일반화

훈련 과정이 24.65 A100 GPU days만 소모하므로, 스케일업을 통한 성능 향상 잠재력이 있으며, 더 큰 배치 사이즈로 총 199 A100 GPU days로 훈련을 확장하였다.

### 3-4. 다운스트림 태스크로의 확장

InstaFlow의 대규모 Rectified Flow는 text-to-3D 및 이미지 inversion/editing 등으로도 확장되었다.

InstaFlow는 연속 시간 diffusion 모델과 1-step 생성 모델 사이의 격차를 크게 좁히고, 알고리즘적 혁신을 영감을 주며 3D 생성과 같은 다운스트림 태스크에도 혜택을 준다.

### 3-5. SDXL-Refiner와의 결합

InstaFlow의 잠재적 활용 사례로 프리뷰어(previewer)의 역할이 있다. 빠른 프리뷰어는 저해상도 필터링 과정을 가속화하고, 동일한 계산 예산 하에 더 많은 생성 가능성을 제공하며, 강력한 후처리 모델이 품질과 해상도를 향상시킬 수 있다.

### 3-6. PeRFlow로의 발전 (후속 연구 관점)

PeRFlow는 SD 1.5 기반 미세조정이 4,000번의 훈련 반복만으로 수렴하는 반면, InstaFlow는 동일 배치 사이즈에서 25,000번의 반복이 필요하다. PeRFlow는 reflow를 위한 대규모 데이터 생성이 필요 없다는 장점도 있다.

이는 InstaFlow의 한계를 극복하는 방향으로 연구가 발전함을 보여줍니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법론 | 최소 스텝 수 | 주요 특징 | 한계 |
|------|--------|------------|---------|------|
| **DDPM** (Ho et al., 2020) | Markov chain 기반 확률 모델 | 1000 | 기반 기술 확립 | 매우 느린 추론 |
| **DDIM** (Song et al., 2020) | 결정론적 샘플링 | 20~50 | 빠른 샘플러 | 소규모에선 품질 저하 |
| **Progressive Distillation** (Salimans & Ho, 2022) | 단계적 증류 | 2~4 | 증류 기반 가속 | 1-step 실패 |
| **StyleGAN-T** (Sauer et al., 2023) | GAN 기반 | 1 | 초고속 생성 | GAN 학습의 불안정성 |
| **Consistency Models** (Song et al., 2023) | 일관성 함수 학습 | 1 | 1-step 가능 | 대규모 T2I 미검증 |
| **InstaFlow** (Liu et al., 2023) | Rectified Flow + Reflow + 증류 | **1** | SD 품질 유지 | 복잡 구성 어려움 |
| **SDXL-Turbo** (Sauer et al., 2023) | Adversarial Diffusion Distillation | 1~4 | 실시간 고품질 | ADD 학습 복잡성 |
| **LCM-LoRA** (Luo et al., 2023) | LCM 기반 LoRA 증류 | 2~4 | 범용 플러그인 가속 | 완전 1-step 어려움 |
| **PeRFlow** (Yan et al., 2024) | Piecewise Rectified Flow | 4 | InstaFlow보다 빠른 학습 | 여전히 멀티스텝 |
| **Rectified Diffusion** (2025) | Consistency Distillation + RF | 1 | InstaFlow 대비 3% GPU만 필요 | 신규 연구로 검증 필요 |

Consistency Models는 자연스럽게 1-step으로 동작하는 새로운 생성 모델이지만, 대규모 텍스트-이미지 생성에서의 성능은 아직 불분명하다.

LCM-LoRA는 훈련 없이 다양한 Stable Diffusion 미세조정 모델이나 LoRA에 직접 플러그인할 수 있으며, DDIM, DPM-Solver 같은 기존 수치적 PF-ODE 솔버에 비해 강력한 일반화 능력을 가진다.

Rectified Diffusion(2025, ICLR 2025)은 InstaFlow의 증류 과정 대비 단 3% GPU days만으로 InstaFlow의 Rectified Flow(Distill) 기준선을 능가하는 성능을 보인다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

#### ① 대규모 확률 흐름 직선화의 실현 가능성 검증
Reflow의 효과가 CIFAR10 같은 소규모 데이터셋에서만 검증되었다는 의문이 있었으나, InstaFlow는 이를 대규모 모델과 빅데이터로 성공적으로 확장하였다.

이를 통해 **대규모 generative model의 1-step 생성 가능성**이 입증되었으며, 후속 연구인 PeRFlow, SDXL-Lightning, Rectified Diffusion 등의 직접적 동기가 되었습니다.

#### ② 에너지 효율성과 환경 영향
1-step 모델의 효율성은 에너지 절약과 환경적 혜택으로 이어질 수 있으며, 이러한 생성 모델의 광범위한 사용을 고려할 때 중요한 의미를 가진다.

#### ③ 다운스트림 태스크 영감
InstaFlow는 연속 시간 diffusion 모델과 1-step 생성 모델 사이의 간극을 크게 좁히고, 알고리즘 혁신을 영감하며 3D 생성 등의 다운스트림 태스크에 혜택을 제공한다.

#### ④ 안전 및 윤리적 고려
빠른 생성 모델은 나쁜 행위자들이 유해한 정보와 가짜 뉴스를 더 쉽게 생성하도록 단순화하고 가속화할 수 있으므로, 이러한 초고속 강력 생성 모델은 인간의 가치와 공공 이익에 부합하도록 하는 고급 정렬 기술이 필요하다.

---

### 5-2. 앞으로 연구 시 고려할 점

#### 🔬 기술적 개선 방향

1. **복잡한 구성(complex compositions) 생성 능력 향상**
   - 더 긴 훈련 기간 및 더 큰 규모의 데이터셋 활용
   - 다중 객체, 공간 관계, 속성 바인딩 등에 특화된 데이터 구성

2. **Reflow 데이터 생성 비용 감소**
   - Reflow 절차는 사전 훈련된 확률 흐름 전체를 시뮬레이션하여 합성 데이터셋을 생성해야 하므로, 대규모 저장 공간과 시간이 필요하며, 샘플의 수치적 오류도 유발할 수 있다. 이 병목을 해결하는 연구가 필요합니다.

3. **멀티스텝 정제(multi-step refinement) 지원**
   - Rectified Flow 기반 증류 결과물은 멀티스텝 정제 능력이 부족한 반면, Consistency Distillation 방식은 더 나은 1-step 성능과 멀티스텝 정제 지원을 동시에 달성할 수 있다.

4. **더 효율적인 증류 알고리즘 설계**
   - Naive LPIPS 손실 대신 Adversarial, Consistency, Score 기반 손실 조합 탐색

5. **고해상도 이미지 생성**
   - 현재 512×512 기준이므로, SDXL 수준(1024×1024)의 1-step 생성으로 확장 필요

6. **ControlNet·LoRA와의 공동 최적화**
   - InstaFlow는 SDXL-Turbo보다 높은 다양성을 가지는 것으로 보이며, ControlNet과도 완전히 호환된다. 이를 기반으로 더 긴밀히 통합된 조건부 1-step 생성 연구가 가능합니다.

#### ⚠️ 윤리 및 사회적 고려

- 초고속 생성 모델의 deepfake, 허위 정보 생성 악용 방지 연구
- 워터마킹, 탐지 모델 등 safeguard 기술과의 병행 연구
- 모델 접근성 민주화와 오남용 방지 사이의 균형

---

## 📚 참고자료 및 출처

1. **주 논문 (arXiv)**: Liu, X., Zhang, X., Ma, J., Peng, J., & Liu, Q. (2023). *InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation*. arXiv:2309.06380. https://arxiv.org/abs/2309.06380
2. **ICLR 2024 논문 (공식 PDF)**: https://proceedings.iclr.cc/paper_files/paper/2024/file/4dc37a7bc61057252ce043fa3b83aac2-Paper-Conference.pdf
3. **OpenReview (ICLR 2024 심사 페이지)**: https://openreview.net/forum?id=1k4yZbbDqX
4. **공식 GitHub**: https://github.com/gnobitab/InstaFlow
5. **Hugging Face Paper Page**: https://huggingface.co/papers/2309.06380
6. **HTML 전문 (arXiv v2)**: https://arxiv.org/html/2309.06380v2
7. **비교 연구 - Rectified Diffusion (ICLR 2025)**: https://proceedings.iclr.cc/paper_files/paper/2025/file/4df9a5e6bad9e64ebcea453e031142bb-Paper-Conference.pdf
8. **비교 연구 - PeRFlow (NeurIPS 2024)**: https://github.com/magic-research/piecewise-rectified-flow
9. **비교 연구 - LCM-LoRA**: Luo, S. et al. (2023). *LCM-LoRA: A Universal Stable-Diffusion Acceleration Module*. arXiv:2311.05556. https://arxiv.org/abs/2311.05556
10. **비교 분석 블로그 (Hugging Face)**: https://huggingface.co/blog/Isamu136/insta-rectified-flow
11. **비교 분석 블로그 (zhangtemplar)**: https://zhangtemplar.github.io/insta-flow/
12. **ICLR 2024 발표 슬라이드**: https://iclr.cc/media/iclr-2024/Slides/19575.pdf
13. **few-step 모델 비교 분석**: https://www.baseten.co/blog/comparing-few-step-image-generation-models/
