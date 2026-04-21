
# InfEdit: Inversion-Free Image Editing with Natural Language

> **논문 정보**
> - **제목**: Inversion-Free Image Editing with Natural Language (CVPR 2024)
> - **저자**: Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, Joyce Chai
> - **arXiv**: [arxiv.org/abs/2312.04965](https://arxiv.org/abs/2312.04965)
> - **공식 프로젝트 페이지**: [sled-group.github.io/InfEdit](https://sled-group.github.io/InfEdit/)
> - **GitHub**: [github.com/sled-group/InfEdit](https://github.com/sled-group/InfEdit)
> - **CVPR 2024 Open Access**: [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Inversion-Free_Image_Editing_with_Language-Guided_Diffusion_Models_CVPR_2024_paper.pdf)

---

## 1. 📌 핵심 주장 및 주요 기여 요약

### ✅ 핵심 주장

기존 inversion 기반 편집 방식은 diffusion 모델에서 텍스트 기반 이미지 조작을 어렵게 만드는 세 가지 주요 병목(1) 역변환 과정의 시간 소모, (2) 일관성과 정확도 균형의 어려움, (3) consistency model의 효율적 샘플링 방식과의 비호환성 문제를 내포하고 있었다.

이 논문은 역변환(inversion) 과정 자체를 제거하고, **가상 역변환(virtual inversion)**을 가능하게 하는 샘플링 전략인 **Denoising Diffusion Consistent Model(DDCM)**을 제안함으로써 이를 근본적으로 해결하고자 한다.

### ✅ 주요 기여 (3가지)

| 기여 | 내용 |
|---|---|
| **DDCM** | 명시적 inversion 없이 가상 역변환 구현 |
| **UAC** | Cross-attention + Self-attention 통합 제어 |
| **InfEdit** | 위 두 기법을 결합한 통합 편집 프레임워크 |

DDCM은 역변환 과정을 제거하는 샘플링 전략을 함의하며, UAC(Unified Attention Control)는 텍스트 기반 편집을 위한 튜닝 불필요(tuning-free) 어텐션 제어 방법을 통합한다. 이 둘을 결합한 InfEdit 프레임워크는 강성(rigid) 및 비강성(non-rigid) 의미 변화 모두에서 일관되고 충실한 편집을 허용한다.

---

## 2. 🔬 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

텍스트 기반 diffusion 모델을 실제 이미지 편집에 활용하는 것은 오랫동안 도전적인 과제였다. 초기 방법들은 마스크 레이어나 추가 학습을 필요로 해 제로샷 적용을 제한했다. DDIM inversion에서 영감을 받은 inversion 기반 편집 패러다임이 주류가 되었으며, 최적화 기반 inversion 방식이 지배적이었다.

효율성 병목과 불완전한 일관성 문제 해결을 위해 이중 브랜치(dual-branch) 방식이 도입되었으나, inversion 기반 편집 방식은 여전히 실시간 및 실세계 언어 기반 이미지 편집에서 한계를 지닌다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### 🔷 A. Diffusion 모델 기반 공식 (배경)

확산 모델의 forward process는 다음과 같이 정의된다:

$$z_t = \sqrt{\alpha_t} z_0 + \sqrt{1 - \alpha_t} \varepsilon, \quad \varepsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

여기서 $z_0$는 원본 잠재 코드(latent code), $\alpha_t$는 noise schedule이다.

훈련 목표는:

$$\min_\theta \, \mathbb{E}_{z_0, \varepsilon, t} \left[ d\left(\varepsilon, \varepsilon_\theta(z_t, t)\right) \right]$$

#### 🔷 B. DDIM Sampling (기존 방식)

DDIM의 denoising step은:

$$z_{t-1} = \sqrt{\alpha_{t-1}} \underbrace{\left(\frac{z_t - \sqrt{1 - \alpha_t}\,\varepsilon_\theta(z_t, t)}{\sqrt{\alpha_t}}\right)}_{\text{predicted } z_0} + \underbrace{\sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \varepsilon_\theta(z_t, t)}_{\text{direction to } z_t} + \sigma_t \varepsilon_t$$

DDIM은 재구성 오류가 발생하기 쉽고, 반복적인 역변환을 필요로 한다.

#### 🔷 C. DDCM (제안 방법): Virtual Inversion

초기 샘플이 알려져 있을 때, 특정 분산 스케줄(variance schedule)이 디노이징 스텝을 멀티-스텝 일관성 샘플링과 동일한 형태로 줄인다는 것을 보인다. 이를 **Denoising Diffusion Consistent Model(DDCM)**이라 명명하며, 이는 명시적 inversion 없이 가상 inversion 전략을 함의한다.

$\sigma_t = \sqrt{1 - \alpha_{t-1}}$ 로 설정 시, DDIM step의 두 번째 항이 소멸하여 다음으로 단순화된다:

$$z_{t-1} = \sqrt{\alpha_{t-1}} \cdot f(z_t, t;\, z_0) + \sqrt{1 - \alpha_{t-1}}\, \varepsilon_t$$

여기서 consistent noise predictor는:

$$\varepsilon^{\text{cons}} = \varepsilon'(z_t, t;\, z_0) = \frac{z_t - \sqrt{\alpha_t}\, z_0}{\sqrt{1 - \alpha_t}}$$

이는 $z_0$를 직접 ground truth로 가리키는 **non-Markovian forward process**를 도입한다.

이 과정은 $z_t$가 신경망 예측 없이 직접 ground truth $z_0$를 가리키며, $z_{t-1}$은 consistency model처럼 이전 스텝 $z_t$에 의존하지 않는다.

#### 🔷 D. LCM(Latent Consistency Model) 연동 샘플링

LCM backbone과 결합 시:

$$\hat{z}_{\tau_i} = \sqrt{\alpha_{\tau_i}}\, z_0^{(\tau_{i+1})} + \sigma_{\tau_i}\, \varepsilon$$
$$z_0^{(\tau_i)} = f_\theta(\hat{z}_{\tau_i},\, \tau_i,\, c)$$

여기서 $c$는 텍스트 조건(condition), $\tau_i$는 consistency timestep이다.

#### 🔷 E. UAC (Unified Attention Control)

UAC는 자연어를 통한 튜닝 불필요(tuning-free) 이미지 편집을 위해, **cross-attention**과 **self-attention** 제어를 하나의 통합 프레임워크 안에 결합한다.

U-Net noise predictor의 각 기본 블록은 cross-attention 모듈과 self-attention 모듈을 포함한다.

- **Cross-attention 제어**: 텍스트-이미지 대응 관계(semantic level)를 제어 → Prompt-to-Prompt(P2P) 방식 확장
- **Self-attention 제어**: 공간적 구조(spatial level) 일관성 유지 → MasaCtrl 방식 통합

---

### 2-3. 모델 구조 요약

```
[ Input Image z_0 ]
        ↓
[ DDCM Forward Process ] ← σ_t = √(1-α_{t-1}) 설정으로 가상 inversion
        ↓
[ LCM / DDPM Backbone (U-Net) ]
        ↓
[ UAC: Cross-Attention + Self-Attention 통합 제어 ]
  ├── Cross-Attention: semantic 편집 (텍스트 조건 반영)
  └── Self-Attention: spatial 구조 보존
        ↓
[ Edited Image ]
```

---

### 2-4. 성능 향상

PIE-Bench에서 다양한 편집 방법들과의 비교 실험 결과, 가상 역변환(VI)은 동일한 P2P 어텐션 제어를 사용하는 다른 역변환 방식들과 경쟁하거나 능가한다. UAC와 LCM backbone의 통합은 성능을 더욱 향상시키며, InfEdit은 단일 A40 GPU에서 대부분의 기준 방법보다 약 **한 자릿수(~10배) 빠르게** 동작한다.

InfEdit은 강성(rigid) 및 비강성(non-rigid) 의미 변화 모두에서 일관되고 충실한 편집을 허용하며, 단일 A40에서 **3초 미만**의 워크플로를 유지해 실시간 응용 가능성을 시연한다.

| 지표 | InfEdit (VI + UAC + LCM) | 기존 DDIM Inversion 기반 |
|---|---|---|
| 속도 | **~3초 (A40 1개)** | 수십 초 이상 |
| 편집 일관성 | 최상위 | 누적 오류 문제 존재 |
| 튜닝 필요 여부 | **불필요** | 경우에 따라 필요 |
| Consistency Model 호환 | **호환** | 비호환 |

---

### 2-5. 한계

기존 이미지 편집 모델들은 단일 속성 편집에는 잘 동작하지만, 여러 객체를 포함한 다중 측면(multi-aspect) 편집에 어려움을 겪으며, DirectInversion 및 InfEdit 같은 모델들도 편집 측면의 수가 증가할수록 성능이 저하되는 경향이 있다.

추가적으로 논문에서 함의하는 한계:
- Stable Diffusion 기반 backbone 의존성 → SD 이외의 아키텍처에 대한 일반화 검증 부족
- 극단적인 semantic 변화(예: 완전한 스타일 전환)에서의 충실도 저하 가능성
- 텍스트-이미지 간 미세한 공간적 관계(예: 특정 위치 지정)에 대한 제어의 한계

---

## 3. 🌐 모델 일반화 성능 향상 가능성

### 3-1. 현재 일반화 강점

DDCM 샘플링은 기존 방법들이 필요로 하는 역변환 브랜치 앵커를 제거하여 상당한 연산을 절약하며, InfEdit은 누적 오류 없이 예측된 초기 $z_0^{\text{tgt}}$를 직접 정제한다. 또한 이 프레임워크는 LCM을 사용한 효율적 consistency sampling과 호환되어 매우 적은 스텝으로 대상 이미지를 샘플링할 수 있다.

### 3-2. 일반화 가능 방향

1. **다양한 Backbone 확장**: DDCM의 variance schedule 수정 원리는 이론적으로 SD XL, SDXL-Turbo, FLUX 등 다양한 아키텍처에 적용 가능
2. **멀티모달 조건 확장**: UAC의 어텐션 제어 메커니즘은 텍스트 외 이미지 참조, 마스크, depth map 등 다양한 조건 신호를 통합하도록 확장 가능
3. **Consistency Model 계열 전반 적용**: 초기 샘플이 알려져 있을 때 특정 분산 스케줄이 디노이징 스텝을 멀티-스텝 일관성 샘플링과 동일한 형태로 만들 수 있다는 DDCM의 수학적 특성은, 다양한 consistency model 계열에 일반화될 수 있는 잠재력을 가진다.
4. **Zero-Shot 적용성**: 초기 마스크 레이어나 추가 훈련 없이도 동작하는 zero-shot 방식을 지향하여, 새로운 도메인 이미지에 대한 적응성이 높다.

---

## 4. 📊 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 핵심 기법 | inversion 필요 | 속도 | 다중 편집 |
|---|---|---|---|---|---|
| **DDIM Inversion** (Song et al.) | 2020 | ODE 기반 forward inversion | ✅ 필요 | 느림 | ❌ |
| **Prompt-to-Prompt (P2P)** (Hertz et al.) | 2022 | Cross-attention map 교체 | ✅ 필요 | 중간 | 부분적 |
| **Null-text Inversion** (Mokady et al.) | 2022 | null-text 최적화 기반 inversion | ✅ 필요 | 매우 느림 | ❌ |
| **MasaCtrl** (Cao et al.) | 2023 | Mutual self-attention | ✅ 필요 | 중간 | 부분적 |
| **LCM** (Luo et al.) | 2023 | Consistency distillation | ✅ 생성 전용 | 매우 빠름 | ❌ (편집 불가) |
| **Direct Inversion** (Ju et al., PIE-Bench) | 2023 | dual-branch inversion | ✅ 필요 | 중간 | 부분적 |
| **InfEdit (본 논문)** | 2023/CVPR2024 | DDCM + UAC | ❌ **불필요** | **~3초** | 부분적 |
| **ParallelEdits** (NeurIPS 2024) | 2024 | DDCM 기반 N-branch 확장 | ❌ 불필요 | ~5초 | **✅ 다중** |

기존 inversion 기반 편집의 주요 병목인 (1) 역변환 과정의 시간 소모, (2) 일관성과 정확도의 균형 문제, (3) consistency model과의 비호환성은 InfEdit 이전 방식들의 공통적 한계였다.

InfEdit, PnP, Direct Inversion 등의 최근 편집 모델들은 cross-attention map을 이용하는 Prompt-to-Prompt의 흐름을 따르고 있다.

---

## 5. 🔭 앞으로의 연구에 미치는 영향 및 고려 사항

### 5-1. 향후 연구에 미치는 영향

**① inversion-free 패러다임의 확산**

InfEdit이 제시한 명시적 역변환 없이도 의미적(semantic) 및 공간적(spatial) 수준에서 일관된 편집을 수행하는 접근법은 이후 diffusion 기반 편집 연구의 새로운 기준점을 제시한다.

**② 실시간 편집 연구 촉진**

InfEdit은 기존 방법 대비 약 10배 빠른 속도를 달성함으로써, 실시간 이미지 편집 응용 연구의 가능성을 열었다.

**③ Consistency Model과 편집의 결합**

DDCM이 LCM과의 호환성을 이론적으로 증명함으로써, 향후 Consistency Model 계열 연구(예: iCD, TCD, Hyper-SD 등)가 편집 과제에서도 활용될 수 있는 이론적 토대를 제공한다.

**④ 멀티-에딧 방향으로의 확장**

ParallelEdits(NeurIPS 2024)는 DDCM 샘플링 과정을 N-브랜치로 확장하여, InfEdit의 프레임워크를 다중 측면 동시 편집(multi-aspect parallel editing)으로 발전시켰다.

---

### 5-2. 앞으로 연구 시 고려할 점

1. **다중 속성 편집(Multi-Aspect Editing)**: 현재 모델은 단일 속성 편집에서는 우수하지만, 여러 객체를 포함한 다중 속면 편집에서는 어려움이 있어 이를 해결하는 확장 연구가 필요하다.

2. **더욱 강력한 Backbone과의 결합**: SDXL, FLUX, DiT 기반 모델 등 최신 고해상도 모델과의 통합 시 DDCM의 variance schedule 이론이 그대로 적용되는지 검증이 필요하다.

3. **정량적 평가 지표 다양화**: PIE-Bench 외에도 다양한 편집 시나리오(스타일, 텍스처, 구조 변경 등)에 대한 포괄적 평가 체계가 요구된다.

4. **윤리적 고려**: InfEdit의 능력을 책임감 있게 활용하기 위해 윤리적 고려, 법적 준수, 사회적 복지를 우선시하는 접근이 필수적이다. 딥페이크, 허위 정보 생성 등 악용 가능성에 대한 안전장치 연구가 병행되어야 한다.

5. **비디오/3D 편집으로의 확장**: DDCM의 시간적 일관성 특성은 비디오 편집, NeRF/3D Gaussian Splatting 등으로의 확장 연구에서도 유망한 방향을 제시한다.

6. **LLM 기반 프롬프트 이해 향상**: UAC가 처리하는 텍스트 조건의 질을 높이기 위해, LLM 기반 편집 의도 파악 및 자동 프롬프트 생성 연구와의 결합이 효과적일 수 있다.

---

## 📚 참고 자료 (출처 목록)

| # | 자료명 | URL |
|---|---|---|
| 1 | **InfEdit 공식 프로젝트 페이지** | https://sled-group.github.io/InfEdit/ |
| 2 | **arXiv 논문 (2312.04965)** | https://arxiv.org/abs/2312.04965 |
| 3 | **CVPR 2024 Open Access 논문 PDF** | https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Inversion-Free_Image_Editing_with_Language-Guided_Diffusion_Models_CVPR_2024_paper.pdf |
| 4 | **ar5iv HTML 논문 페이지** | https://ar5iv.labs.arxiv.org/html/2312.04965 |
| 5 | **GitHub 공식 구현** | https://github.com/sled-group/InfEdit |
| 6 | **HuggingFace Paper Page** | https://huggingface.co/papers/2312.04965 |
| 7 | **ParallelEdits (NeurIPS 2024)** — InfEdit의 DDCM 확장 연구 | https://proceedings.neurips.cc/paper_files/paper/2024/file/2847043899e1171183ceadf86bdbb280-Paper-Conference.pdf |
| 8 | **Inverse-and-Edit (arxiv 2025)** — InfEdit 인용 연구 | https://arxiv.org/html/2506.19103 |
| 9 | **Jiayi Pan 개인 논문 페이지** | https://jiayipan.me/publication/infedit/ |
| 10 | **Awesome Text-to-Image Studies (GitHub)** — 관련 연구 목록 | https://github.com/AlonzoLeeeooo/awesome-text-to-image-studies |
