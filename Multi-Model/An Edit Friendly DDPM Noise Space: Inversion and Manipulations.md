
# An Edit Friendly DDPM Noise Space: Inversion and Manipulations

**논문 정보:**
- **저자:** Inbar Huberman-Spiegelglas, Vladimir Kulikov, Tomer Michaeli
- **학회:** CVPR 2024 (pp. 12469–12478)
- **arXiv:** 2304.06140 (v1: 2023년 4월 12일)
- **공식 GitHub:** https://github.com/inbarhub/DDPM_inversion

---

## 1. 핵심 주장 및 주요 기여 요약

DDPM(Denoising Diffusion Probabilistic Models)은 이미지를 생성하기 위해 백색 가우시안 노이즈 샘플의 시퀀스를 사용하며, GAN과의 유사성에서 이 노이즈 맵들은 생성된 이미지에 대응하는 잠재 코드(latent code)로 볼 수 있다. 그러나 이 기본(native) 노이즈 공간은 편리한 구조를 갖추지 못해 편집 작업에 활용하기 어렵다.

이에 저자들은 간단한 수단으로 다양한 편집 작업을 가능하게 하는 **대안적인 잠재 노이즈 공간(alternative latent noise space)**을 DDPM에 제안하고, 임의의 이미지(실제 또는 합성)에서 이 편집 친화적 노이즈 맵을 추출하는 **역변환(inversion) 방법**을 함께 제시한다.

### 핵심 주장 3가지:

1. 편집 친화적 노이즈 맵은 표준 정규 분포를 따르지 않고 타임스텝 간 통계적 독립성도 없지만, 원하는 이미지를 완벽하게 재구성하고 이 맵에 가한 단순한 변환(시프팅, 색상 편집 등)이 출력 이미지의 의미 있는 조작으로 이어진다.

2. 텍스트 조건부 모델에서, 이 노이즈 맵을 고정한 채 텍스트 프롬프트만 변경하면 **구조를 유지하면서 의미론을 수정**할 수 있으며, 이를 통해 비다양적(non-diverse) DDIM 역변환 대신 다양한 DDPM 샘플링 방식으로 실제 이미지의 텍스트 기반 편집을 가능하게 한다.

3. 기존 확산 기반 편집 방법들에 이 방법을 통합하여 품질과 다양성을 향상시킬 수 있다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

실제 이미지를 확산 모델로 편집하려면 생성 과정에서 해당 이미지를 생성할 수 있는 노이즈 벡터를 추출해야 한다. 대다수의 확산 기반 편집 연구는 단일 노이즈 맵에서 생성 이미지로의 결정론적 매핑인 DDIM 방식을 사용하며, 기존 DDIM 논문은 이 방식에 대한 효율적인 근사 역변환을 제안하였다.

DDPM에서는 생성 과정에 $T+1$개의 노이즈 맵이 관여하며, 노이즈 공간의 총 차원이 출력보다 커서 이미지를 완벽히 재구성하는 일관된 역변환이 무한히 존재한다. 그러나 일관된 역변환이 모두 편집 친화적인 것은 아니며, 텍스트 조건부 모델에서 노이즈 맵을 고정하고 텍스트 프롬프트를 변경했을 때 아티팩트 없이 새 텍스트에 대응하는 의미론으로 수정되면서도 원본의 구조가 유지되어야 한다는 요건이 있다.

노이즈 맵이 통계적으로 독립이고 표준 정규 분포를 따르면 편집 친화적일 것이라는 직관적 답변은 CycleDiffusion 등의 연구에서 시도되었으나, 이 기본(native) DDPM 노이즈 공간이 실제로는 편집 친화적이지 않음을 확인하였다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### DDPM의 Forward(가산) 과정

DDPM의 표준 Forward 과정은 다음과 같이 정의된다:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t \mathbf{I})$$

이를 임의의 타임스텝 $t$에 대해 닫힌 형태(closed form)로 표현하면:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \mathbf{I})$$

여기서 $\bar{\alpha}_t = \prod\_{s=1}^{t}(1-\beta_s)$.

#### DDPM의 Reverse(역방향) 과정

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\hat{\epsilon}_\theta(x_t, t)\right) + \sigma_t\, z_t, \quad z_t \sim \mathcal{N}(0, \mathbf{I})$$

여기서 $\hat{\epsilon}_\theta$는 학습된 노이즈 예측 네트워크, $z_t$는 각 스텝에서의 랜덤 노이즈.

#### 편집 친화적 역변환 (Edit-Friendly Inversion)의 핵심 아이디어

저자들은 텍스트 가이드 조작부터 손으로 그린 색상 스트로크 편집까지 다양한 편집 애플리케이션에 더 적합한 대안적 역변환 방법을 제시한다. 이 역변환은 이미지를 노이즈 맵에 보다 강하게 "각인(imprint)"하여, 이를 고정하고 모델의 조건을 바꿀 때 구조가 더 잘 보존되도록 한다.

논문에서 제시하는 편집 친화적 보조 시퀀스 구성은 다음과 같다. $x_0$가 주어졌을 때, 각 타임스텝 $t = 1, \ldots, T$에 대해 독립적인 $\tilde{\epsilon}_t \sim \mathcal{N}(0, \mathbf{I})$를 샘플링하여 보조 시퀀스를 구성한다:

$$\tilde{x}_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \tilde{\epsilon}_t, \quad t = 1, \ldots, T$$

그 후 각 스텝에서의 편집 친화적 노이즈 맵 $z_t$를 역방향 과정 공식으로부터 역산하여 추출한다:

$$z_t = \frac{\tilde{x}_{t-1} - \frac{1}{\sqrt{\alpha_t}}\left(\tilde{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\hat{\epsilon}_\theta(\tilde{x}_t, t)\right)}{\sigma_t}$$

이 방식으로 추출된 $\{x_T, z_T, \ldots, z_1\}$을 역방향 과정에 다시 사용하면 $x_0$를 완벽히 재구성한다.

이 역변환 방법으로 추출된 노이즈 맵은 일반 샘플링에서 사용되는 것과 다르게 분포되어 있다. 타임스텝 간 상관관계가 있고 분산이 더 높다. 그러나 이 맵들은 이미지의 구조를 더 강하게 인코딩하므로 이미지 편집에 더 적합하다.

---

### 2-3. 모델 구조

이 논문은 새로운 신경망 아키텍처를 제안하는 것이 아니라, 기존 사전 학습된 DDPM 모델 위에서 동작하는 **역변환 알고리즘**을 제안한다.

이 방법은 모델을 파인튜닝하거나 어텐션 맵을 수정하지 않고도 실제 이미지의 다양한 편집을 가능하게 하며, Prompt-to-Prompt나 Zero-Shot I2I와 같은 다른 알고리즘에도 쉽게 통합될 수 있다.

공식 저장소에서 지원하는 편집 모드는 다음과 같다:
- **우리의 역변환 단독 사용**: DDPM 역변환 후 타깃 프롬프트로 이미지 생성
- **Prompt-to-Prompt (P2P) + 우리의 역변환**: P2P 방법에 DDPM 역변환을 결합

---

### 2-4. 성능 향상

특히 텍스트 기반 편집 작업에서 단독으로 또는 다른 편집 방법과 결합하여 최신 수준(state-of-the-art)의 결과를 보여줬으며, DDIM 역변환에 의존하는 방법들과 달리 주어진 이미지와 텍스트에 대해 다양한 결과를 생성할 수 있다.

이 DDPM 역변환은 현재 근사 DDIM 역변환에 의존하는 기존 확산 기반 편집 방법에도 쉽게 통합되어 원본 이미지에 대한 충실도를 향상시키며, 노이즈 벡터를 확률적 방식으로 찾기 때문에 텍스트 프롬프트에 부합하는 다양한 편집 이미지 집합을 제공할 수 있다.

구체적 편집 사례:
- ImageNet으로 학습된 무조건부 모델에서 100 추론 스텝으로 생성한 256×256 이미지를 오른쪽으로 d=1,2,4,8,12,16픽셀씩 시프팅할 때, 기본 노이즈 맵이나 CycleDiffusion으로 추출한 노이즈 맵을 사용하면 구조가 손실되지만, 제안 방법의 잠재 공간을 사용하면 구조가 보존된다.

- 색상 조작(color manipulation)에서도 제안 방법(T2=70~T1=20, s=0.05 적용)은 텍스처와 구조를 수정하지 않고 강한 편집 효과를 낸다.

---

### 2-5. 한계 (Limitations)

편집 친화적 노이즈는 소스 이미지를 노이즈 공간에 더 강하게 각인하여 더 나은 재구성을 보장하지만, 노이즈의 감소로 인해 수정 가능한 공간(modification space)이 줄어드는 문제가 있다.

대안적 역변환 기법 중 하나로서, Edit Friendly DDPM은 필수 콘텐츠 보존과 편집 충실도 모두에서 편집 결과의 불안정성을 나타낼 수 있으며, 정성적·정량적 실험에서도 이 불안정성이 확인된다.

고충실도 편집 친화적 역변환은 역변환된 잠재 공간의 노이즈 통계와 생성 prior 간의 정렬에 의존하며, DDIM을 통한 역변환은 과도한 구조를 가진 잠재 변수를 산출하여 조작 자유도를 줄일 수 있다는 실용적 한계가 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 다른 모달리티로의 확장

오디오 영역에서도 이 편집 친화적 DDPM 역변환 방법이 활용되고 있다. 소스 신호를 입력으로 받아 보조 벡터 시퀀스를 생성하고, 추출된 노이즈 벡터들은 원래 생성 과정과 다른 분포를 가지지만 신호의 전역 구조를 더 강하게 인코딩하여 편집 작업에 특히 적합하다.

오디오 도메인에서는 DDPM 역변환이 일반화되어 제로샷 텍스트 기반 오디오 편집(ZETA)과 비지도 주성분 조작(ZEUS)이 가능해졌으며, 악기 참여도, 리듬, 즉흥 연주에 대한 세밀한 제어를 지원한다.

SDE 기반 방법의 한 갈래로서 DDPM 역변환은 확률적 경로를 따라 노이즈나 잔차를 추적하여 입력을 복원하며, 이 개념을 이산 확산 모델로 확장하여 연속 및 이산 설정 모두에서 효과적인 역변환을 가능하게 하는 연구도 등장하였다.

### 3-2. 기존 방법과의 플러그인(Plug-in) 결합

이 DDPM 역변환은 현재 근사 DDIM 역변환에 의존하는 기존 확산 기반 편집 방법들에 즉시 통합될 수 있으며, 이를 통해 원본 이미지에 대한 충실도 보존 능력이 향상된다.

LEDITS는 편집 친화적 DDPM 역변환 기법과 Semantic Guidance를 결합한 경량 접근법으로, 최적화나 아키텍처 확장 없이 미묘한 편집부터 광범위한 편집, 구성 및 스타일 변경까지 다양한 편집을 달성한다.

### 3-3. 일반화를 위한 기술적 개선 방향

편집 친화적 역변환의 일반화를 위한 개선 방향으로는 추가적인 전방 확산 스텝 통합, 스텝별 랜덤 정규직교 변환 적용(FreeInv), 노이즈 스케줄 조정(선형/코사인 대신 로지스틱 스케줄) 등이 연구되고 있으며, 이를 통해 역변환 잠재 변수의 편향 보정, 탈상관화, 가우시안화 및 오차 누적 감소가 가능하다.

비디오 및 오디오 도메인으로의 확장도 이루어지고 있으며, 시간적으로 일관된 비디오 편집(TokenFlow+FreeInv, DAVIS 벤치마크)과 오디오 편집(ZETA, ZEUS)에도 기법이 일반화되고 있다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 기반 | 역변환 방식 | 다양성 | 충실도 | 특징 |
|---|---|---|---|---|---|
| **DDIM Inversion** (Song et al., 2020) | DDIM | 결정론적 | ❌ 낮음 | △ 근사적 | 빠르지만 편집 다양성 없음 |
| **Null-Text Inversion** (Mokady et al., 2023) | DDIM | 최적화 기반 | ❌ 낮음 | ✅ 높음 | 고충실도이나 계산 비용 큼 |
| **CycleDiffusion** (Wu & de la Torre, 2022) | DDPM | 확률적 | ✅ 있음 | △ 구조 손실 | 분포 보존이나 편집에 약함 |
| **EDICT** (Wallace et al., 2023) | DDPM | 수학적 정확 | △ | △ 편집성 감소 | 두 커플 노이즈 벡터 사용 |
| **Edit Friendly DDPM (본 논문)** | DDPM | 편집 친화적 | ✅ 높음 | ✅ 완벽 재구성 | 구조 강하게 인코딩 |
| **LEDITS** (Tsaban et al., 2023) | DDPM+SEGA | 편집 친화적 | ✅ | ✅ | 본 논문 + 의미론적 가이던스 |
| **LEDITS++** (Brack et al., 2024) | DDPM+SEGA | 편집 친화적 | ✅ | ✅ | LEDITS보다 6배 빠름 |

EDICT(Exact Diffusion Inversion via Coupled Transformations)는 어파인 커플링 레이어에서 영감을 받아 교대 방식으로 서로를 역변환하는 두 커플 노이즈 벡터를 유지함으로써 실제 이미지와 모델 생성 이미지의 수학적으로 정확한 역변환을 가능하게 하는 방법이다.

ENM Inversion(Editable Noise Map Inversion)은 콘텐츠 보존과 편집 가능성을 모두 보장하는 최적 노이즈 맵을 탐색하고, 재구성된 노이즈 맵과 편집된 노이즈 맵 간의 차이를 최소화하는 편집 가능한 노이즈 정제를 도입한 새로운 역변환 기법이다.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5-1. 향후 연구에 미치는 영향

편집 친화적 DDPM 역변환은 텍스트 가이드 및 무조건부 확산 모델에서 유연하고 고충실도의 편집을 달성하기 위한 기초적 발전으로 평가되며, 표현력 있는 잠재 표현과 효율적이고 안정적인 역변환 및 편집 워크플로를 조화시키는 데 기여한다.

DDPM 역변환의 범위는 여러 도메인에 걸쳐 있으며, 실제 또는 생성 이미지를 노이즈 맵으로 역변환하고 이를 조작함으로써 기하학적·광학적 편집, 속성 전환, 구성 수정, 프롬프트 조건부 변환 등이 가능해진다.

**구체적 파생 연구 방향:**

1. **LEDITS / LEDITS++**: 편집 친화적 DDPM 역변환 기법과 Semantic Guidance를 결합하여 실제 이미지 편집으로의 확장을 제안한다.
2. **오디오 편집 (ZETA/ZEUS)**: 편집 친화적 DDPM 역변환 방법을 오디오 신호 편집에 활용하여, 추출한 노이즈 벡터를 DDPM 샘플링 과정에서 사용하면서 텍스트 프롬프트 변경을 통해 원하는 편집 방향으로 생성을 유도한다.
3. **비디오 편집**: 시간적으로 일관된 비디오 편집에도 기법이 일반화되고 있다.
4. **이산 확산 모델 일반화**: DDPM 역변환 개념이 이산 확산 모델로 확장되어 연속 및 이산 설정 모두에서 효과적인 역변환이 가능하도록 하는 연구가 진행 중이다.

### 5-2. 향후 연구 시 고려할 점

미래 연구 방향으로는 적응적 스텝별 스케줄 튜닝, 편집 친화적 노이즈 공간으로 직접 모델 학습, 고해상도/비디오 파이프라인의 강건성 확보, 그리고 크로스모달 확장이 포함된다.

의미론적 과제약(Semantic Overconstraint) 문제, 즉 역변환을 과도하게 제약하면 편집 가능성이 저해될 수 있으며, 이는 이중 조건부(dual-conditional) 및 멀티모달 역변환 연구로 이어지고 있다.

이 방법은 아키텍처에 무관(architecture-agnostic)하여 어떤 확산 모델에도 쉽게 사용될 수 있지만, 전반적인 편집 품질은 기반이 되는 사전 학습된 확산 모델의 역량에 크게 의존하며, 더 역량 있는 모델일수록 더 나은 편집 결과를 제공한다. 단, 사용된 확산 모델이 타깃 개념에 대한 충분한 표현력이 없으면 특정 편집 지시가 실패할 수 있다.

---

## 📚 참고 출처

1. **arXiv 원문 (2304.06140)**: https://arxiv.org/abs/2304.06140
2. **CVPR 2024 Open Access**: https://openaccess.thecvf.com/content/CVPR2024/papers/Huberman-Spiegelglas_An_Edit_Friendly_DDPM_Noise_Space_Inversion_and_Manipulations_CVPR_2024_paper.pdf
3. **IEEE Xplore (CVPR 2024)**: https://ieeexplore.ieee.org/document/10657449
4. **공식 GitHub**: https://github.com/inbarhub/DDPM_inversion
5. **공식 프로젝트 페이지**: https://inbarhub.github.io/DDPM_inversion/
6. **Semantic Scholar**: https://www.semanticscholar.org/paper/An-Edit-Friendly-DDPM-Noise-Space:-Inversion-and-Huberman-Spiegelglas-Kulikov/d1c33172c2ffbc038f0598f3ac56bb04af79c904
7. **HuggingFace Papers**: https://huggingface.co/papers/2304.06140
8. **ResearchGate**: https://www.researchgate.net/publication/370001472_An_Edit_Friendly_DDPM_Noise_Space_Inversion_and_Manipulations
9. **LEDITS (arXiv 2307.00522)**: https://arxiv.org/abs/2307.00522 — Tsaban & Passos, 2023
10. **LEDITS++ (CVPR 2024)**: https://openaccess.thecvf.com/content/CVPR2024/papers/Brack_LEDITS_Limitless_Image_Editing_using_Text-to-Image_Models_CVPR_2024_paper.pdf
11. **Zero-Shot Audio Editing Using DDPM Inversion (arXiv 2402.10009)**: https://arxiv.org/abs/2402.10009
12. **PNP Inversion (OpenReview)**: https://openreview.net/pdf/7a08897fea2fe55b08fa202685f17c84e0337fe4.pdf
13. **EmergentMind - Edit-Friendly DDPM Inversion 토픽**: https://www.emergentmind.com/topics/edit-friendly-ddpm-inversion
