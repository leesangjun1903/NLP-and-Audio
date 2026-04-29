
# Photorealistic Object Insertion with Diffusion-Guided Inverse Rendering

> **저자:** Ruofan Liang, Zan Gojcic, Merlin Nimier-David, David Acuna, Nandita Vijaykumar, Sanja Fidler, Zian Wang
> **소속:** NVIDIA Toronto AI Lab
> **발표:** ECCV 2024
> **arXiv:** [2408.09702](https://arxiv.org/abs/2408.09702)

---

## 1. 🔑 핵심 주장 및 주요 기여 요약

실제 세계 장면 이미지에 가상 물체를 올바르게 삽입하려면, 장면의 조명(lighting), 기하(geometry), 재질(materials) 및 이미지 형성 과정(image formation process)에 대한 깊은 이해가 필요합니다.

최근 대규모 확산(diffusion) 모델들이 강력한 생성 및 인페인팅(inpainting) 능력을 보여주고 있으나, 현재 모델들은 단일 이미지에서 그림자, 밝은 반사 등 일관된 조명 효과를 생성하거나 합성 물체의 정체성과 디테일을 보존하는 데 충분하지 못합니다.

### 핵심 주장
본 논문은 **DiPIR (Diffusion Prior for Inverse Rendering)** 을 제안하며, 세 가지 핵심 기여에 기반합니다:
- **첫째**, 물리 기반 렌더러(physically based renderer)를 사용하여 빛과 3D 자산 간의 상호작용을 정확하게 시뮬레이션하여 최종 합성 이미지를 생성합니다.
- **둘째**, 알려지지 않은 톤 매핑 곡선(tone-mapping curve)을 카메라 센서 응답으로 모사합니다.
- **셋째**, 입력 이미지와 삽입되는 자산 유형에 기반한 사전 학습된 DM(Diffusion Model)의 경량 개인화(personalization) 방식을 제안합니다.

더불어, 이 개인화를 활용하고 학습 안정성을 개선하는 **SDS(Score Distillation Sampling) 손실의 변형(variant)** 을 설계합니다.

---

## 2. 🔬 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

표준 가상 물체 삽입 파이프라인은 세 가지 핵심 단계를 포함합니다: ① 입력 이미지에서의 조명 추정, ② 3D 프록시 지오메트리(proxy geometry) 생성, ③ 렌더링 엔진에서의 합성 이미지 렌더링. 이 중 첫 번째 단계인 조명 추정이 가장 중요하지만 여전히 해결되지 않은 연구 문제로, 특히 소비자 기기로 촬영된 단일 저다이나믹 레인지(LDR) 이미지 같은 제한적인 입력을 다룰 때 더욱 어렵습니다. 실제로 역 렌더링(inverse rendering)은 근본적으로 **불량 조건 문제(ill-posed problem)** 입니다.

특히, 오프더셀프(off-the-shelf) 확산 모델(DMs)은 종종 가상 물체 삽입에 강건한 가이던스를 제공하지 못하며, 특히 야외 주행 환경과 같은 **분포 외(out-of-distribution) 장면**에서 더욱 그러합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### ① 물리 기반 렌더링 파이프라인

본 방법은 대형 확산 모델의 내재적 장면 이해 능력을 물리 기반 역 렌더링 파이프라인의 가이던스로 활용합니다. 장면별 개인화(scene-specific personalization)를 통한 확산 가이던스 신호와 미분 가능한 역 렌더링 파이프라인을 설계하여 조명 및 톤 매핑 파라미터를 복원합니다.

렌더링 방정식(Rendering Equation)은 다음과 같습니다:

$$L_o(\mathbf{x}, \omega_o) = \int_{\Omega} f_r(\mathbf{x}, \omega_i, \omega_o) \cdot L_i(\mathbf{x}, \omega_i) \cdot (\omega_i \cdot \mathbf{n}) \, d\omega_i$$

여기서:
- $L_o$: 출사 복사(Outgoing Radiance)
- $f_r$: BRDF (쌍방향 반사도 분포 함수)
- $L_i$: 입사 복사(Incoming Radiance)
- $\mathbf{n}$: 표면 법선(Surface Normal)

#### ② 톤 매핑 보정 (Tone-mapping Correction)

입력 이미지의 알 수 없는 톤 매핑을 보상하기 위해 최적화 가능한 톤 보정 함수(optimizable tone correction function)를 도입합니다.

구체적으로 추정된 환경 맵(environment map)을 고정한 후, 배경 이미지에 수동 톤 조정을 적용하고, 확산 가이던스를 사용하여 삽입된 물체가 최종 합성 결과에서 주변 배경과 일치하도록 톤 곡선을 최적화합니다.

#### ③ Score Distillation Sampling (SDS) 기반 최적화

기존 SDS 손실(DreamFusion, Poole et al., 2022)은 다음과 같이 정의됩니다:

$$\nabla_{\theta} \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon} \left[ \omega(t) \left( \hat{\epsilon}_{\phi}(\mathbf{x}_t; y, t) - \epsilon \right) \frac{\partial \mathbf{x}}{\partial \theta} \right]$$

여기서:
- $\theta$: 최적화 파라미터 (환경 맵 및 톤 매핑 파라미터)
- $\omega(t)$: 타임스텝 $t$에 따른 가중치
- $\hat{\epsilon}_{\phi}$: 개인화된 확산 모델의 노이즈 예측값
- $\epsilon$: 추가된 가우시안 노이즈

DiPIR는 이 SDS 손실을 **개인화된 확산 모델**과 결합한 변형 형태로 활용하며:

$$\nabla_{\theta} \mathcal{L}_{\text{DiPIR}} = \mathbb{E}_{t, \epsilon} \left[ \omega(t) \left( \hat{\epsilon}_{\phi_{\text{pers}}}(\mathbf{x}_t; y, t) - \epsilon \right) \frac{\partial \mathbf{x}_{\text{render}}}{\partial \theta} \right]$$

여기서 $\phi_{\text{pers}}$는 장면별로 개인화된 확산 모델의 파라미터입니다.

각 반복(iteration)마다 렌더링된 이미지가 확산(diffused)되어 개인화된 확산 모델에 입력되고, 적응된 Score Distillation 공식의 그래디언트가 미분 가능한 렌더러를 통해 환경 맵과 톤 매핑 곡선으로 역전파됩니다. 수렴 시, 조명 및 톤 매핑 파라미터가 복원되어 단일 이미지에서 가상 물체를 사실적으로 합성할 수 있게 됩니다.

#### ④ 개인화 (Personalization with Concept Preservation)

오프더셀프 DMs은 특히 야외 주행 환경과 같은 분포 외 장면에서 가상 물체 삽입에 강건한 가이던스를 충분히 제공하지 못합니다. 이에 대한 잠재적 해결책으로, 대상 장면의 이미지를 사용하여 DM을 적응시키는 방법을 제안합니다.

이는 **DreamBooth** 또는 **LoRA** 방식의 파인튜닝을 경량화하여 적용하는 방식으로, 다음을 학습합니다:

$$\mathcal{L}_{\text{personalize}} = \mathbb{E}_{\mathbf{x}, \epsilon, t} \left[ \| \epsilon - \epsilon_{\phi}(\sqrt{\bar{\alpha}_t}\mathbf{x} + \sqrt{1-\bar{\alpha}_t}\epsilon, t, c) \|^2 \right]$$

---

### 2.3 모델 구조

DiPIR는 단일 이미지를 입력으로 받아, 가상 물체의 사실적인 삽입을 목표로 장면 조명 및 톤 매핑 파라미터를 복원합니다.

전체 파이프라인은 다음 세 모듈로 구성됩니다:

| 모듈 | 역할 |
|------|------|
| **물리 기반 렌더러 (Mitsuba 3)** | 환경 맵 기반 조명으로 가상 객체 렌더링, 그림자 및 반사 시뮬레이션 |
| **개인화된 확산 모델** | 장면 특성에 맞게 파인튜닝된 DM으로 SDS 가이던스 제공 |
| **미분 가능한 최적화기** | 환경 맵($\mathcal{E}$)과 톤 매핑 곡선($\mathcal{T}$)을 동시 최적화 |

본 방법은 다양한 실내 및 야외 장면에서 효과를 검증하며, Waymo 야외 주행 장면과 언래핑된 실내 HDRI 파노라마를 대상 배경 이미지로 사용하여 평가합니다.

#### 조명 표현: Spherical Gaussians (SG)

환경 맵을 구면 가우시안(Spherical Gaussians)으로 표현합니다:

$$L_{\text{env}}(\omega) = \sum_{k=1}^{K} \mu_k \cdot \exp\left(\lambda_k (\omega \cdot \xi_k - 1)\right)$$

여기서 $\mu_k$는 색상 진폭, $\lambda_k$는 날카로움(sharpness), $\xi_k$는 lobe 방향입니다.

SG 기반 조명 표현은 일반적인 물체에는 적절하지만, **고도 반사(highly specular) 재질에는 사실적으로 동작하지 않을 수 있습니다.** 더 복잡한 조명 표현을 위해 환경 맵에 생성적 사전(generative prior)을 추가하는 것이 탐구할 만한 방향입니다.

---

### 2.4 성능 향상

본 방법은 가상 3D 물체를 배경 이미지에 삽입하기 위한 조명 조건을 더욱 정확하게 추정할 수 있습니다.

**사용자 스터디 결과**: Waymo 야외 스트리트 장면 벤치마크에서, 주간(Daytime), 황혼(Twilight), 야간(Night) 및 전체 장면(All scenes)에 걸쳐 본 방법이 기준선(baselines) 대비 선호되는 비율이 50%를 초과하며 우수한 성능을 보입니다.

또한 가상 물체 삽입뿐만 아니라, 삽입된 물체의 재질(material) 최적화 또는 카메라 간 톤 매핑 불일치 보정 등 다른 장면 파라미터의 최적화도 지원합니다.

---

### 2.5 한계

**주요 한계**:
1. **조명 표현의 한계**: Spherical Gaussians 기반 조명 표현은 고도 반사 재질(highly specular materials)에는 사실적으로 동작하지 않을 수 있습니다.
2. **복잡한 조명 표현 필요성**: 더 복잡한 조명 표현에는 환경 맵에 생성적 사전(generative priors)을 추가하는 방향이 탐구할 만한 미래 연구 과제입니다.

또한 각 연산에는 단순화(simplification)가 포함되어 있으며 (예: 프록시 지오메트리에 의한 이차 조명(secondary lighting)은 고려하지 않음), 남아 있는 시뮬레이션 불완전성은 조명 최적화로 어느 정도 상쇄될 수 있습니다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

손으로 제작된 또는 지도 학습 기반의 데이터 주도 사전(prior)과 달리, 확산 모델(DMs)은 대규모 데이터셋으로 학습되어 세계와 그 기저의 물리적 개념에 대한 놀라운 "이해력"을 보여줍니다. DMs가 생성물에서 그림자 및 반사와 같은 정확한 조명 효과를 만들어내는 데 여전히 종종 실패하지만, 물리 기반 렌더러와 결합되고 장면에 적응될 때 유용한 가이던스를 제공할 수 있음을 관찰했습니다.

### 일반화 성능 향상을 위한 핵심 메커니즘

| 전략 | 설명 |
|------|------|
| **장면별 개인화 (Scene-specific Personalization)** | 입력 이미지로 DM을 파인튜닝하여 분포 외 장면(야외 주행 등)에서도 강건한 가이던스 제공 |
| **실내·야외 모두 지원** | Waymo 야외 + HDRI 실내 파노라마 모두에서 검증 |
| **단일 프레임·비디오 모두 지원** | 단일 이미지에서 비디오 시퀀스까지 확장 가능한 범용 파이프라인 |
| **임의의 가상 물체 삽입** | 특정 카테고리에 한정되지 않고 다양한 3D 자산에 적용 가능 |

DMs이 물리 기반 렌더러와 결합되고 장면에 적응될 때, 단독으로는 실패하는 그림자 및 반사와 같은 조명 효과 생성에 유용한 가이던스를 제공할 수 있습니다.

### 일반화를 저해하는 요인 및 향후 개선 방향

1. **분포 외(OOD) 장면**: 개인화 파인튜닝이 필요하여 추가 계산 비용 발생
2. **고반사 재질**: SG 기반 조명 표현의 표현력 한계
3. **단일 이미지 입력의 모호성**: 역 렌더링은 이미지에서 장면 속성을 추론하는 도전적인 역문제이며, 서로 다른 많은 장면 구성이 동일한 이미지를 만들어낼 수 있어 태스크 자체가 불량 조건(ill-posed) 문제입니다.

---

## 4. 🔭 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

1. **확산 모델과 물리 기반 렌더링의 융합 패러다임 확립**
개인화된 확산 모델이 물리 기반 역 렌더링 과정을 가이드하여 실제 세계 장면에 가상 물체의 사실적인 삽입을 향상시키는 새로운 패러다임을 제시합니다.

2. **확장된 응용 가능성**
가상 물체 삽입 외에도 삽입된 물체의 재질 최적화, 카메라 간 톤 매핑 불일치 보정 등 다양한 응용을 지원하는 기반을 마련합니다.

3. **후속 연구 촉진**
이 연구에 이어, DiffusionRenderer(2025)와 같은 후속 연구들은 강력한 비디오 확산 모델 사전을 활용하여 실제 세계 비디오에서 G-버퍼를 정확히 추정하고, 명시적인 광선 추적(ray tracing) 없이 G-버퍼로부터 사실적인 이미지를 생성하는 방향으로 발전하고 있습니다.

4. **채널별 노이즈 스케줄링 접근법**
역 렌더링의 불량 조건 문제를 해결하기 위해, 채널별 노이즈 스케줄링 방법을 통해 단일 확산 모델 아키텍처가 정확한 단일 해 예측과 다양한 해 생성이라는 두 가지 충돌하는 목표를 달성하고, 이는 물체 삽입 및 재질 편집과 같은 다운스트림 응용에서 향상된 성능으로 이어지고 있습니다.

### 4.2 앞으로 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|-----------|-----------|
| **조명 표현 고도화** | SG 대신 신경망 기반 환경 맵 혹은 NeRF 기반 조명 표현 탐구 |
| **다중 이미지/비디오 입력** | 단일 이미지의 조명 모호성을 다중 뷰 또는 비디오로 완화 |
| **실시간화** | 최적화 기반 접근의 수렴 속도 개선, 실시간 AR/XR 적용 목표 |
| **고반사 재질 처리** | Specular 재질을 위한 더 정밀한 BRDF 모델 및 조명 표현 필요 |
| **개인화 비용 절감** | 장면별 파인튜닝 비용 절감을 위한 메타러닝 또는 제로샷 적응 연구 |
| **데이터 편향 해소** | 다양한 조명 조건(야간, 악천후 등) 데이터 보강 필요 |

---

## 5. 📊 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 방법 | 입력 | 조명 표현 | 차별점 대비 DiPIR |
|------|------|-----------|------|-----------|-------------------|
| **Li et al. (Inverse Rendering for Indoor Scenes)** | 2020 | 신경망 기반 역 렌더링 | 단일 이미지 | SVBRDF + 공간 가변 조명 | 지도 학습 기반, 실내 한정 |
| **DreamFusion (Poole et al.)** | 2022 | SDS + NeRF | 텍스트 | 없음 | 3D 생성 목적, 물리 기반 렌더링 없음 |
| **SDEdit (Meng et al.)** | 2022 | 확률 미분방정식 기반 편집 | 이미지 + 편집 가이드 | 없음 | 조명 물리성 없음 |
| **DiffusionLight (Phongthawee et al.)** | 2023 | 크롬 볼 페인팅으로 조명 추정 | 단일 이미지 | HDR 환경 맵 | 조명 추정에 특화, 물체 삽입 파이프라인 없음 |
| **DiPIR (본 논문)** | 2024 | 개인화 DM + 미분가능 역 렌더링 | 단일 이미지 | SG 기반 환경 맵 + 톤 매핑 | **실내·야외 통합, 물리·DM 하이브리드** |
| **DiffusionRenderer** | 2025 | 비디오 DM 기반 역/순 렌더링 | 단일 비디오 | G-버퍼 | 합성 데이터로 역 렌더링 비디오 DM을 학습하고, 합성 및 자동 레이블된 실세계 데이터로 렌더링 모델을 공동 학습 |
| **Channel-wise Noise Scheduled Diffusion** | 2025 | 채널별 노이즈 스케줄링 | 단일 이미지 | 픽셀별 조명 | 잠재 확산 모델(LDM) 기반 역 렌더링으로, 대규모 이미지-텍스트 데이터에서 학습된 사전(prior)을 활용하여 다양한 해 제공 및 우수한 예측 능력 발휘 |

---

## 📚 참고 문헌 및 출처

1. **Liang, R. et al.** (2024). *Photorealistic Object Insertion with Diffusion-Guided Inverse Rendering*. ECCV 2024. arXiv:2408.09702. [[arXiv]](https://arxiv.org/abs/2408.09702) [[Springer]](https://link.springer.com/chapter/10.1007/978-3-031-73030-6_25)
2. **NVIDIA Toronto AI Lab - DiPIR 프로젝트 페이지**: https://research.nvidia.com/labs/toronto-ai/DiPIR/
3. **Hugging Face Paper Page**: https://huggingface.co/papers/2408.09702
4. **ECCV 2024 Poster**: https://eccv.ecva.net/virtual/2024/poster/1319
5. **Poole, B. et al.** (2022). *DreamFusion: Text-to-3D using 2D Diffusion*. arXiv:2209.14988.
6. **Meng, C. et al.** (2022). *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations*. ICLR 2022.
7. **Phongthawee, P. et al.** (2023). *DiffusionLight: Light Probes for Free by Painting a Chrome Ball*. arXiv.
8. **Li, Z. et al.** (2020). *Inverse Rendering for Complex Indoor Scenes*. CVPR 2020.
9. **DiffusionRenderer** (2025). *Neural Inverse and Forward Rendering with Video Diffusion Models*. arXiv:2501.18590. [[arXiv]](https://arxiv.org/html/2501.18590v1)
10. **Channel-wise Noise Scheduled Diffusion** (2025). *Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes*. CVPR 2025. [[arXiv]](https://arxiv.org/html/2503.09993)
11. **Nimier-David, M. et al.** (2022). *Mitsuba 3 Renderer*. https://mitsuba-renderer.org
12. **Hu, E.J. et al.** (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.

---

> ⚠️ **정확도 안내**: 본 답변에서 수식의 일부(특히 DiPIR 내부 구현 세부 사항)는 공개 논문 전문(arXiv HTML 버전 포함) 및 관련 SDS 문헌을 기반으로 재구성한 것으로, 논문의 구체적인 알고리즘 표현과 완전히 동일하지 않을 수 있습니다. 정확한 수식은 원문(arXiv:2408.09702)의 Section 4를 직접 확인하시기를 권장합니다.
