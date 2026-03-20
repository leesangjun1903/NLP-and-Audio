# DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation

> **논문 정보**: Gwanghyun Kim, Taesung Kwon, Jong Chul Ye (KAIST), **CVPR 2022**, arXiv:2110.02711

---

## 1. 핵심 주장 및 주요 기여 요약

DiffusionCLIP은 GAN 기반 방법의 한계를 극복하고, 실제 이미지의 충실한 조작(faithful manipulation)을 가능하게 하기 위해 **확산 모델(Diffusion Model)**과 **CLIP 손실**을 결합한 텍스트 기반 이미지 조작 방법이다.

### 주요 기여점:

1. **거의 완벽한 역변환(Inversion) 능력 입증**: 확산 모델이 거의 완벽한 역변환 능력 덕분에 이미지 조작에 매우 적합하며, 이는 GAN 기반 모델 대비 중요한 이점이라는 것을 최초로 심층 비교 분석하였다.

2. **빠른 파인튜닝을 위한 새로운 샘플링 전략**: 완벽한 재구성을 유지하면서 속도를 높이는 새로운 샘플링 전략을 제안하였다.

3. **In-domain 및 Out-of-domain 모두 우수한 조작 성능**: 정확한 도메인 내·외부 조작을 가능하게 하고, 의도하지 않은 변화를 최소화하며, SOTA 베이스라인을 크게 능가하였다.

4. **다중 속성 동시 조작**: 여러 파인튜닝된 확산 모델의 노이즈를 결합하여 간단하게 다중 속성 조작을 가능하게 하는 새로운 노이즈 결합 방법을 제안하였다.

5. **ImageNet 수준의 일반 이미지 조작**: 보이지 않는(unseen) 도메인 간에서도 제로샷 이미지 조작을 성공적으로 수행하며, 광범위하게 변화하는 ImageNet 데이터셋의 이미지까지 조작할 수 있어 일반 응용에 한 걸음 더 다가갔다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 GAN 인버전 + CLIP 방법(StyleCLIP, StyleGAN-NADA 등)은 텍스트 프롬프트 기반 제로샷 이미지 조작을 가능하게 했지만, GAN 인버전의 제한된 능력 때문에 다양한 실제 이미지에의 적용이 어려웠다. 특히, 훈련 데이터와 비교하여 새로운 포즈, 시점, 고도로 가변적인 콘텐츠를 가진 이미지의 재구성에 어려움이 있었으며, 객체 정체성의 변경이나 원치 않는 이미지 아티팩트를 생성하는 문제가 있었다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) DDIM 기반 결정론적 순방향/역방향 프로세스

DiffusionCLIP은 DDIM(Denoising Diffusion Implicit Models) 샘플링과 그 역변환을 기반으로, 조작을 가속화할 뿐만 아니라 거의 완벽한 역변환을 가능하게 한다.

**DDPM 학습 목표 (Training Objective):**

$$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right\|^2 \right]$$

여기서 $\boldsymbol{\epsilon}_\theta$는 노이즈 근사 모델, $\mathbf{x}_t$는 시간 $t$에서의 노이즈가 추가된 이미지이다.

**결정론적 DDIM 순방향 (Forward) 프로세스 — Inversion:**

전체 역변환을 위해 결정론적 역방향 DDIM 프로세스와 ODE 근사를 채택하였다.

$$\mathbf{x}_{t+1} = \sqrt{\bar{\alpha}_{t+1}} \, \mathbf{f}_\theta(\mathbf{x}_t, t) + \sqrt{1 - \bar{\alpha}_{t+1}} \, \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

여기서:

$$\mathbf{f}_\theta(\mathbf{x}_t, t) = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}$$

**결정론적 DDIM 역방향 (Reverse) 프로세스 — Generation:**

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \, \mathbf{f}_\theta(\mathbf{x}_t, t) + \sqrt{1 - \bar{\alpha}_{t-1}} \, \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

#### (B) CLIP 손실 함수

CLIP으로부터의 지식을 효과적으로 추출하기 위해 두 가지 손실이 제안되었다: 글로벌 타겟 손실(global target loss)과 로컬 방향 손실(local directional loss).

**Global CLIP Loss:**

$$\mathcal{L}_{\text{global}} = D_{\text{CLIP}}(\mathbf{x}_{\text{gen}}, t_{\text{tar}})$$

여기서 $D_{\text{CLIP}}$은 CLIP 공간에서의 코사인 거리, $t_{\text{tar}}$는 타겟 텍스트, $\mathbf{x}_{\text{gen}}$은 생성 이미지이다.

**Directional CLIP Loss (핵심 손실):**

방향 CLIP 손실로 조작된 이미지는 이미지 표현 간의 방향을 참조 텍스트와 타겟 텍스트 간의 방향에 정렬시키므로 모드 붕괴에 강건하다.

$$\mathcal{L}_{\text{direction}} = 1 - \frac{\Delta I \cdot \Delta T}{\|\Delta I\| \|\Delta T\|}$$

여기서:

$$\Delta T = E_T(y_{\text{tar}}) - E_T(y_{\text{ref}}), \quad \Delta I = E_I(\mathbf{x}_{\text{gen}}) - E_I(\mathbf{x}_{\text{ref}})$$

$E_T$, $E_I$는 각각 CLIP의 텍스트 인코더와 이미지 인코더이다.

#### (C) 전체 파인튜닝 목적 함수

확산 모델 $\boldsymbol{\epsilon}_\theta$의 파인튜닝을 위해 CLIP 손실과 정체성(identity) 손실로 구성된 목적 함수를 사용한다.

$$\mathcal{L}\_{\text{total}} = \mathcal{L}_{\text{direction}} + \lambda_{\ell_1} \mathcal{L}_{\ell_1} + \lambda_{\text{face}} \mathcal{L}_{\text{face}}$$

여기서:
- $\mathcal{L}\_{\ell_1} = \|\mathbf{x}\_{\text{gen}} - \mathbf{x}_{\text{ref}}\|_1$ : 픽셀 레벨의 정체성 보존
- $\mathcal{L}_{\text{face}}$ : 얼굴 정체성 손실 (IR-SE50 기반, 사람 얼굴 조작 시)
- $\lambda_{\ell_1}, \lambda_{\text{face}} \geq 0$ : 각 손실의 가중치

정체성 손실의 필요성은 제어 유형에 따라 다르다. 일부 제어(표정, 머리색 등)에서는 픽셀 유사성과 인간 정체성 보존이 중요하지만, 다른 제어(예술 작품, 종 변경 등)에서는 심한 형태와 색상 변화가 선호된다.

#### (D) 빠른 파인튜닝/추론을 위한 전략

학습 가속화를 위해, 이미지를 마지막 스텝 $T$까지 역변환할 필요가 없으며, $t_0 \in [0, T]$ ('return step')까지만 역변환한다. 역변환 및 생성 과정의 스텝 수를 $S_{\text{inv}}$, $S_{\text{gen}}$으로 줄여 더욱 가속화할 수 있다.

빠른 조작 세팅 $(S_{\text{for}}, S_{\text{gen}}) = (40, 6)$에서, 파인튜닝은 1~7분, 추론은 2초 내에 완료된다.

#### (E) 노이즈 결합을 통한 다중 속성 조작

여러 파인튜닝된 모델 $\{\boldsymbol{\epsilon}_{\theta_i}\}$에서 예측된 노이즈를 결합:

$$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \sum_{i=1}^{N} w_i \, \boldsymbol{\epsilon}_{\theta_i}(\mathbf{x}_t, t)$$

여기서 $w_i$는 각 속성에 대한 가중치이다.

### 2.3 모델 구조

| 구성 요소 | 상세 내용 |
|:---|:---|
| **확산 모델 백본** | CelebA-HQ, LSUN-Church, LSUN-Bedroom의 256×256 이미지에서 사전학습된 DDPM/DDIM 모델 |
| **CLIP 모델** | 사전 훈련된 CLIP (ViT-B/16) — 텍스트·이미지 인코더 |
| **아키텍처** | U-Net 기반의 공유 아키텍처, 모든 타임스텝에서 공유되는 노이즈 예측 네트워크 |
| **얼굴 정체성 보존** | 인간 얼굴 정체성 보존을 위해 사전 훈련된 IR-SE50 모델 사용 |
| **실험 데이터셋** | CelebA-HQ, AFHQ-Dog, LSUN-Bedroom, LSUN-Church, ImageNet |

**전체 파이프라인:**
1. 입력 이미지 $\mathbf{x}\_0$ → **Forward DDIM** (사전학습된 모델) → 잠재 벡터 $\mathbf{x}_{t_0}$
2. CLIP 방향 손실로 확산 모델 파인튜닝
3. 파인튜닝된 모델로 $\mathbf{x}_{t_0}$ → **Reverse DDIM** → 조작된 이미지 $\hat{\mathbf{x}}_0$

### 2.4 성능 향상

일반 사례와 어려운 사례 모두에서 DiffusionCLIP의 결과가 모든 베이스라인보다 선호되었으며(> 50%), 특히 어려운 사례에서는 선호율이 더 증가하여 강건한 조작 성능을 보여주었다. StyleCLIP 대비 약 90%의 선호율을 기록하여 out-of-domain 조작에서 크게 우수하였다.

| 비교 대상 | 주요 결과 |
|:---|:---|
| **StyleCLIP** | Out-of-domain에서 ~90% 선호율 |
| **StyleGAN-NADA** | In/Out-of-domain 모두 우위 |
| **TediGAN** | 재구성 품질 및 조작 정확도 우수 |

### 2.5 한계점

논문 및 후속 연구 분석을 통해 도출된 주요 한계:

1. **파인튜닝 필요성**: 각 새로운 텍스트 속성마다 확산 모델을 파인튜닝해야 하므로, 추가 비용이 발생
2. **해상도 제약**: 실험이 256×256 해상도에 국한됨
3. **추론 속도**: GAN 기반 방법 대비 여전히 느린 추론 시간 (DDIM 다중 스텝 필요)
4. **정밀한 지역적 편집의 어려움**: 이러한 방법들은 정밀하고 국소적인 편집을 달성하는 데 종종 어려움을 겪으며, 비대상 영역에서 아티팩트나 불일치를 자주 도입한다.
5. **CLIP 의존성**: CLIP 모델 자체의 한계(세밀한 시각적 차이 구분의 어려움)가 조작 품질에 영향

---

## 3. 모델의 일반화 성능 향상 가능성

DiffusionCLIP의 일반화 관련 핵심 내용:

### 3.1 확산 모델의 본질적 일반화 우위

확산 모델이 거의 완벽한 역변환 능력 덕분에 이미지 조작에 매우 적합하다는 것을 밝혔으며, 이는 GAN 기반 모델 대비 중요한 이점이다.

GAN 인버전은 학습된 잠재 공간의 표현력에 의존하므로, 훈련 데이터 분포를 벗어난 이미지에서 성능이 급격히 저하된다. 반면, DDIM의 결정론적 순방향-역방향 과정은 다음과 같은 속성을 보장한다:

$$\mathbf{x}_0 \xrightarrow{\text{Forward DDIM}} \mathbf{x}_{t_0} \xrightarrow{\text{Reverse DDIM (same } \boldsymbol{\epsilon}_\theta)} \hat{\mathbf{x}}_0 \approx \mathbf{x}_0$$

이 거의 완벽한 재구성은 **입력 이미지의 도메인에 무관**하게 성립하여, 일반화 성능의 기반이 된다.

### 3.2 Unseen 도메인 간 변환

DiffusionCLIP은 훈련된 도메인 내 이미지뿐만 아니라, 보이지 않는(unseen) 도메인으로의 조작을 성공적으로 수행하며, 심지어 하나의 unseen 도메인에서 다른 unseen 도메인으로의 이미지 변환이나, 스트로크로부터 unseen 도메인에서의 이미지 생성도 가능하다.

이것은 다음 두 가지 메커니즘에 의해 가능하다:
- **CLIP의 제로샷 일반화**: CLIP은 대규모 이미지-텍스트 쌍에서 학습되어, 새로운 도메인의 텍스트 프롬프트에도 의미론적 방향을 제공
- **확산 모델의 도메인 무관 역변환**: DDIM 역변환은 입력 이미지의 구조적 정보를 충실히 보존

### 3.3 ImageNet 수준의 일반 적용

GAN 기반 인버전은 ImageNet의 다양성 때문에 제한된 성능을 보이는 반면, DiffusionCLIP은 제로샷 텍스트 기반 이미지 편집을 가능하게 하여 일반 텍스트 기반 조작에 한 걸음 더 다가갔다.

### 3.4 일반화 향상을 위한 미래 방향

| 방향 | 설명 |
|:---|:---|
| **대규모 사전학습 모델 활용** | Stable Diffusion, Imagen 등 대규모 텍스트-이미지 모델과의 통합 |
| **파인튜닝 없는 방법론** | Classifier-Free Guidance, Attention 조작 기반 접근법 |
| **다중 모달 조건부 일반화** | 텍스트 외에도 이미지, 레이아웃, 스케치 등 다양한 조건 활용 |
| **고해상도 확장** | Latent Diffusion Models(LDM) 기반의 해상도 확장 |

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

DiffusionCLIP은 **확산 모델 기반 이미지 편집**이라는 새로운 패러다임을 선도하였으며, 이후 수많은 후속 연구에 영감을 제공하였다:

1. **확산 모델의 이미지 조작 적용 가능성 입증**: GAN 독점 영역이던 이미지 편집에 확산 모델이 진입할 수 있음을 최초로 체계적으로 보여줌
2. **CLIP + Diffusion 결합 패러다임 확립**: 이후 CLIP 기반 가이던스를 활용한 다양한 방법론의 기반
3. **DDIM Inversion의 중요성 재조명**: DiffusionCLIP이 DDIM 역변환을 사용하여 확산 과정을 역전하고 파인튜닝을 적용하며, CLIP 기반 손실로 생성 이미지를 의도된 편집에 더 잘 정렬시키는 방식은 이후 많은 연구에서 채택되었다.

### 4.2 향후 연구 시 고려할 점

1. **실시간 추론**: 확산 모델의 반복적 샘플링 특성으로 인한 추론 속도 문제 — Consistency Models, Progressive Distillation 등으로 개선 가능
2. **지역적 편집(Local Editing) 정밀도**: Cross-attention 제어, 마스크 기반 편집 등의 통합 필요
3. **사용자 인터페이스**: 텍스트 프롬프트의 모호성 해소 및 직관적 제어 메커니즘 개발
4. **윤리적 고려**: 딥페이크 등 오용 가능성에 대한 대비
5. **다양한 이미지 유형**: 의료 이미지, 위성 이미지 등 특수 도메인으로의 확장

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

DiffusionCLIP(Kim et al., 2022)과 P2P(Hertz et al., 2023) 등 초기 연구가 크로스-어텐션 메커니즘을 활용한 의미론적·스타일 변환을 가능하게 한 이후, InstructPix2Pix(Brooks et al., 2023) 같은 명령어 기반 모델이 일반 텍스트로 편집을 가이드할 수 있도록 발전하였다.

| 방법 | 연도 | 핵심 접근법 | 장점 | 한계 | DiffusionCLIP 대비 |
|:---|:---:|:---|:---|:---|:---|
| **DiffusionCLIP** | 2022 | DDIM Inversion + CLIP 방향 손실로 확산 모델 파인튜닝 | 거의 완벽한 역변환, Unseen 도메인 조작, 다중 속성 | 속성마다 파인튜닝 필요, 256² 해상도, 느린 추론 | — |
| **Prompt-to-Prompt (P2P)** | 2022 | Cross-attention map 제어로 편집 | 파인튜닝 불필요, 구조 보존 우수 | 실제 이미지 적용 시 DDIM Inversion 정확도 의존 | 파인튜닝 없이 가능하나, 실제 이미지 편집에 추가 기법 필요 |
| **InstructPix2Pix** | 2023 | 생성 데이터로 조건부 확산 모델을 훈련, 실제 이미지와 사용자 작성 명령어에 일반화 | Forward pass만으로 편집, 자연어 명령어 | 대규모 합성 데이터 필요, 정밀 제어 어려움 | 추론 속도 크게 향상, 파인튜닝 불필요 |
| **Imagic** | 2023 | 텍스트 임베딩 최적화 + 모델 파인튜닝 | 복잡한 비강체(non-rigid) 편집 가능 | 이미지당 파인튜닝 필요, 높은 비용 | 더 정밀한 편집 가능하나 비용 더 높음 |
| **Null-text Inversion** | 2023 | Null 텍스트 최적화로 DDIM 역변환 정확도 향상 | 높은 재구성 충실도 | CFG와의 호환성 문제 | DiffusionCLIP의 역변환 한계를 보완 |
| **MasaCtrl** | 2023 | Mutual Self-Attention 제어 | 파인튜닝 불필요, 일관된 이미지 합성 | 복잡한 구조 변경 어려움 | 효율적이나 적용 범위 제한 |
| **LEDITS / LEDITS++** | 2023 | DDPM 역변환 + 의미론적 가이던스 | 여러 개념 동시 편집, 파인튜닝 불필요 | 정밀 제어 어려움 | DiffusionCLIP의 다중 속성 아이디어를 발전 |
| **ControlNet** | 2023 | 입력 이미지로부터 추출된 에지·포즈 정보를 조건으로 생성 과정을 제어, 원래 공간 구조를 유지하면서 프롬프트에 따라 스타일링 | 다양한 조건 (에지, 깊이, 포즈) | 사전학습 조건 인코더 필요 | 공간 제어 정밀도 향상 |
| **DreamBooth** | 2023 | 소수 이미지로 개인화 파인튜닝 | 주제 보존 우수 | 과적합 위험, 학습 비용 | 개인화 생성에 특화 |

### 패러다임의 진화

```
[2021] StyleCLIP / StyleGAN-NADA (GAN + CLIP)
    ↓
[2022] DiffusionCLIP (Diffusion + CLIP, 파인튜닝 기반)
    ↓
[2022] Prompt-to-Prompt (Attention 제어, 파인튜닝 불필요)
    ↓
[2023] InstructPix2Pix / Imagic / Null-text Inversion (대규모 학습 / 최적화 기반)
    ↓
[2023-2024] ControlNet / LEDITS++ / SEGA (다중 조건 / 다중 가이던스)
    ↓
[2024+] Training-free 편집, LLM 통합 편집 (RPG, SmartEdit 등)
```

텍스트 기반 이미지 편집 방법의 다수가 Imagen, Stable Diffusion 같은 대규모 확산 기반 생성 모델의 인상적인 능력을 활용하여 최근 개발되었으나, 다양한 유형의 세밀한 편집을 비교하기 위한 표준화된 평가 프로토콜은 아직 존재하지 않는다.

---

## 참고 자료 및 출처

1. **Kim, G., Kwon, T., & Ye, J. C.** (2022). "DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation." *CVPR 2022*, pp. 2426–2435. [arXiv:2110.02711](https://arxiv.org/abs/2110.02711)
2. **GitHub 공식 저장소**: [gwang-kim/DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP)
3. **OpenReview (ICLR 제출본)**: [DiffusionCLIP: Text-guided Image Manipulation Using Diffusion Models](https://openreview.net/forum?id=TKMJ9eqtpgP)
4. **CVPR 2022 Open Access**: [DiffusionCLIP — CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.html)
5. **ar5iv 논문 전문**: [ar5iv.labs.arxiv.org/html/2110.02711](https://ar5iv.labs.arxiv.org/html/2110.02711)
6. **MarkTechPost 해설**: [Researchers From KAIST Propose DiffusionCLIP](https://www.marktechpost.com/2022/04/11/researchers-from-kaist-korea-propose-diffusionclip/)
7. **Semantic Scholar**: [DiffusionCLIP 논문 페이지](https://www.semanticscholar.org/paper/8f8dedb511c0324d1cb7f9750560109ca9290b5f)
8. **Brooks, T., et al.** (2023). "InstructPix2Pix: Learning to Follow Image Editing Instructions." *CVPR 2023*. [arXiv:2211.09800](https://arxiv.org/abs/2211.09800)
9. **Hertz, A., et al.** (2022). "Prompt-to-Prompt Image Editing with Cross Attention Control." *arXiv:2208.01626*
10. **Gomez-Trenado, G., et al.** (2025). "Don't Forget your Inverse DDIM for Image Editing." [arXiv:2505.09571](https://arxiv.org/abs/2505.09571)
11. **EditVal Benchmark** (2023). "Benchmarking Diffusion Based Text-Guided Image Editing Methods." [OpenReview](https://openreview.net/forum?id=nkCWKkSLyb)
12. **SlideShare 프레젠테이션**: [DiffusionCLIP 발표자료](https://www.slideshare.net/ssuser2e0133/diffusionclip-textguided-diffusion-models-for-robust-image-manipulation)
13. **Huang, L., et al.** (2024). "A Survey of Multimodal-Guided Image Editing with Text-to-Image Diffusion Models." [arXiv:2406.14555](https://arxiv.org/abs/2406.14555)
