
# SEGA: Instructing Text-to-Image Models using Semantic Guidance

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 문제 의식

텍스트-이미지 확산 모델(Text-to-Image Diffusion Models)은 텍스트만으로 고품질 이미지를 생성하는 놀라운 능력으로 최근 큰 주목을 받고 있다. 그러나 사용자의 의도에 맞는 이미지를 단번에 생성하는 것은 사실상 불가능하며, 입력 프롬프트의 작은 변경이 전혀 다른 이미지를 만들어 낸다. 이는 사용자에게 거의 의미론적(semantic) 제어권을 주지 못하는 문제를 야기한다.

### 1.2 핵심 주장 (Core Claim)

이 문제를 해결하기 위해 SEGA는 확산 과정과 상호작용하여 의미론적 방향(semantic directions)을 따라 유연하게 조향(steer)할 수 있는 방법을 제시한다. **SEGA는 Classifier-Free Guidance를 사용하는 모든 생성 아키텍처에 범용적으로 적용 가능하며**, 미묘한 편집부터 광범위한 편집, 구성 및 스타일 변경, 전반적인 예술적 표현의 최적화를 가능하게 한다.

### 1.3 주요 기여 (Key Contributions)

구체적으로 논문은 다음 4가지를 기여한다: (i) Semantic Guidance의 공식적 정의와 해당 시맨틱 공간의 수치적 직관 제시, (ii) 시맨틱 벡터의 강건성(robustness), 유일성(uniqueness), 단조성(monotonicity), 고립성(isolation) 증명, (iii) SEGA의 시맨틱 제어에 대한 광범위한 실험적 평가 제공, (iv) 관련 방법 대비 SEGA의 이점 시연.

SEGA는 추가 학습, 아키텍처 확장, 외부 가이던스가 전혀 필요 없으며, 단일 순전파(forward pass) 내에서 계산된다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

SEGA는 확산 모델에 대한 Semantic Guidance로 제안되어 이미지 생성에 강력한 의미론적 제어를 제공한다. 텍스트 프롬프트의 작은 변경은 보통 완전히 다른 결과 이미지를 만들어 낸다. 그러나 SEGA를 통해 원본 이미지 구성을 유지하면서 직관적이고 쉽게 제어 가능한 다양한 변경이 가능해진다.

### 2.2 제안하는 방법 및 수식

#### Step 1. 확산 모델(Diffusion Model) 기반

확산 모델은 가우시안 분포 변수를 반복적으로 디노이즈하여 학습된 데이터 분포의 샘플을 생성한다. 텍스트-이미지 생성의 경우 모델은 텍스트 프롬프트 $p$에 컨디셔닝되어 해당 프롬프트에 충실한 이미지로 유도된다. DM의 훈련 목표 $\hat{x}_\theta$는 다음과 같이 쓸 수 있다:

$$\mathcal{L} = \mathbb{E}_{t,x,\epsilon}\left[ w_t \| \hat{x}_\theta(z_t, c_p) - x \|^2 \right]$$

여기서 $(x, c_p)$는 텍스트 프롬프트 $p$에 컨디셔닝되고, $t \sim \mathcal{U}([0,1])$, $\epsilon \sim \mathcal{N}(0, I)$이며, $w_t, \omega_t, \alpha_t$는 시간 $t$에 따른 이미지 충실도에 영향을 준다.

#### Step 2. Classifier-Free Guidance (CFG)

가이던스 스케일 $s_g$와 노이즈 예측값 $\epsilon_\theta$를 사용하여, 비조건부 $\epsilon$-예측이 조건부 방향으로 밀려나게 되며 $s_g$가 그 정도를 결정한다:

$$\tilde{\epsilon}_\theta(z_t, c_p) = \epsilon_\theta(z_t) + s_g \cdot \left(\epsilon_\theta(z_t, c_p) - \epsilon_\theta(z_t)\right)$$

#### Step 3. SEGA의 핵심 수식

CFG의 산술 원리를 이용하여, SEGA는 임의의 시맨틱 개념을 인코딩하는 잠재 벡터의 차원들을 식별한다. 이를 위해 개념 설명 $e$에 컨디셔닝된 노이즈 예측값 $\epsilon_\theta(z_t, c_e)$를 계산하고, 비조건부 예측값 $\epsilon_\theta(z_t)$와의 차이를 스케일링한다.

단일 편집 방향에 대한 SEGA의 수정된 노이즈 예측은 다음과 같다:

$$\bar{\epsilon}_\theta(z_t, c_p, c_e) = \tilde{\epsilon}_\theta(z_t, c_p) + s_e \cdot \mu\left(\epsilon_\theta(z_t, c_e) - \epsilon_\theta(z_t)\right)$$

- $s_e$: 편집 가이던스 스케일
- $c_e$: 편집 개념(concept) 프롬프트의 임베딩
- $\mu(\cdot)$: 가장 절댓값이 큰 차원들만을 선택하는 임계값 마스킹 함수 (즉, 상위/하위 $\tau$% tail에 해당하는 차원 선택)

$\mu$ (Eq. 4)는 정의된 편집 프롬프트 $e$와 관련된 프롬프트 조건부 예측의 차원들을 고려한다. $\mu$는 비조건부와 개념-조건부 예측 간의 차이의 절댓값이 가장 큰 값들을 취한다.

다중 편집 방향 $e_1, e_2, \ldots, e_k$에 대해 확장하면:

$$\bar{\epsilon}_\theta(z_t, c_p, \{c_{e_i}\}) = \tilde{\epsilon}_\theta(z_t, c_p) + \sum_{i=1}^{k} s_{e_i} \cdot \mu_i\left(\epsilon_\theta(z_t, c_{e_i}) - \epsilon_\theta(z_t)\right)$$

#### Step 4. 모멘텀 및 웜업 파라미터

편집 임계값(`edit_threshold`, 기본값 0.9), 모멘텀 스케일(`edit_momentum_scale`, 기본값 0.1) 등도 도입된다. 모멘텀 스케일이 0.0으로 설정되면 모멘텀은 비활성화된다. 모멘텀은 웜업 기간 동안 축적되며, `edit_mom_beta`는 이전 모멘텀이 얼마나 유지될지를 정의한다.

편집 웜업 스텝(`edit_warmup_steps`, 기본값 10)은 시맨틱 가이던스가 적용되지 않는 확산 스텝 수를 정의하며, 이 기간 동안 모멘텀이 계산되고 웜업 종료 후 적용된다.

### 2.3 모델 구조 (Architecture)

SEGA는 Classifier-Free Guidance에서 도입된 원리를 실질적으로 확장하여, 오직 모델의 잠재 공간에 이미 존재하는 개념들과만 상호작용한다. 따라서 SEGA는 추가 훈련, 아키텍처 확장, 외부 가이던스가 불필요하다. 오히려 기존 확산 반복 내에서 계산된다. 더 구체적으로, SEGA는 텍스트 프롬프트 $p$에 더하여 목표 개념들을 나타내는 다수의 텍스트 설명 $e_i$를 사용한다.

**SEGA 파이프라인 구성요소 (Stable Diffusion 기준):**

| 구성 요소 | 설명 |
|---|---|
| VAE (AutoencoderKL) | 이미지 인코딩/디코딩 |
| Text Encoder (CLIP) | 텍스트 프롬프트 임베딩 |
| UNet (UNet2DConditionModel) | 노이즈 예측 |
| Scheduler | 디노이징 스케줄 관리 |
| Semantic Guidance Module | 추가 학습 없이 CFG 위에 삽입 |

SEGA의 효과는 Stable Diffusion, Paella, DeepFloyd-IF와 같은 잠재(latent) 기반 및 픽셀 기반 확산 모델 모두에서 다양한 작업으로 검증되었으며, 그 다용성과 유연성에 대한 강력한 증거를 제공한다.

### 2.4 성능 향상

SEGA는 임의의 개념을 원본 이미지에 통합하는 데 있어 강건하게(robustly) 동작한다. 예를 들어 'glasses'라는 프롬프트를 서로 다른 도메인의 이미지들에 적용했을 때, 어떻게 안경을 통합해야 하는지 컨텍스트를 제공하지 않음에도 불구하고 잘 작동한다.

SEGA로 발견된 가이던스 방향은 강건하고, 단조적으로 스케일링되며, 크게 고립(isolated)되어 있다. 이는 이미지에 대한 미묘한 편집, 구성 및 스타일 변화, 예술적 표현 최적화의 동시 적용을 가능하게 한다.

SEGA는 이미지 생성을 위한 word2vec과 유사한 산술 연산(예: King - Male + Female ≈ Queen)을 가능하게 하며, 아키텍처에 어떠한 확장도 필요로 하지 않고 임의의 텍스트 프롬프트에 대해 즉석(ad-hoc)으로 시맨틱 벡터를 생성한다.

### 2.5 한계 (Limitations)

가이던스 벡터의 전이(transfer)는 동일한 초기 시드(seed)로 제한되는데, $\epsilon$-예측이 발산하는 초기 노이즈 잠재 변수에 따라 크게 달라지기 때문이다. 더불어 인간 얼굴에서 동물이나 무생물 객체로의 전환과 같이 이미지 구성의 더 광범위한 변경은 별도의 계산이 필요하다.

$\bar{\epsilon}^*_\theta(z_t, c_e)$는 비제약 네트워크의 출력이며 분류기의 그래디언트가 아니기 때문에, SEGA 성능에 대한 보장이 없다. 그럼에도 불구하고 이 도출은 탄탄한 이론적 기반을 제공하며, SEGA의 효과성은 실험적으로 입증된다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 아키텍처 독립성 (Architecture-Agnostic)

SEGA는 Classifier-Free Guidance를 사용하는 **모든 생성 아키텍처에 범용화(generalize)된다.** 이는 SEGA가 특정 모델 구조에 종속되지 않음을 의미한다.

### 3.2 도메인 전이(Domain Transfer) 가능성

하나의 개념에 대한 가이던스 벡터 $\gamma$는 유일하며, 한 번 계산한 후 다른 이미지에도 적용할 수 있다. 예를 들어 'glasses' 가이던스를 가장 왼쪽 이미지에서 계산하여 다른 프롬프트의 확산 과정에 더하면, 사진 리얼리즘에서 드로잉으로의 상당한 도메인 전환을 포함한 모든 얼굴에 안경이 생성된다.

### 3.3 LEDITS로의 확장 — 실제 이미지 일반화

SEGA의 개념은 텍스트 가이드 확산 모델의 생성 과정에 대한 세밀한 제어를 강화하기 위해 도입되었다. SEGA는 모델의 잠재 공간에 이미 존재하는 개념들과만 상호작용함으로써 Classifier-Free Guidance에서 도입된 원리를 확장한다. SEGA는 크로스 어텐션 맵에 대한 토큰 기반 컨디셔닝이 필요 없으며, 다중 시맨틱 변경의 조합을 허용한다.

DDPM 반전(inversion)과 SEGA 기법의 결합을 LEDITS라고 부르며, SEGA 기법을 실제 이미지로 확장하고 두 방법의 편집 능력을 동시에 활용하는 결합 편집 접근법을 도입하여 최첨단 방법들과 경쟁력 있는 정성적 결과를 보여준다.

### 3.4 시맨틱 공간의 산술적 일반화

표현력 있는 시맨틱 벡터에 관한 연구는 생성적 확산 모델보다 앞서 존재했다. word2vec과 같은 텍스트 임베딩에 대한 덧셈/뺄셈은 자연어에서 시맨틱 및 언어적 관계를 반영하는 것으로 나타났으며, 가장 두드러진 예시는 'King − male + female'의 벡터 표현이 'Queen'에 매우 가깝다는 것이다. SEGA는 이와 유사한 산술을 이미지 생성에도 가능하게 하여 시맨틱 공간의 일반화 능력을 크게 넓혔다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

| 방법 | 발표 | 핵심 접근 | 학습 필요 | 아키텍처 변경 | 다중 편집 | 한계 |
|---|---|---|---|---|---|---|
| **DiffusionCLIP** | CVPR 2022 | CLIP loss로 DM 미세조정 | ✅ 필요 | ❌ | 제한적 | 반복 학습 비용 |
| **Prompt-to-Prompt** | ICLR 2023 | Cross-attention 맵 제어 | ❌ | ❌ | 제한적 | 토큰 기반 의존 |
| **InstructPix2Pix** | CVPR 2023 | GPT-3+SD 합성 데이터로 학습 | ✅ 필요 | ✅ | 가능 | 대규모 재학습 |
| **Imagic** | CVPR 2023 | 텍스트 임베딩 최적화 | ✅ 필요 | ❌ | 제한적 | 이미지별 수분 학습 |
| **SEGA** | NeurIPS 2023 | CFG 확장 시맨틱 벡터 조작 | ❌ | ❌ | ✅ | 시드 의존성 |
| **LEDITS** | arXiv 2023 | SEGA + DDPM 반전 | ❌ | ❌ | ✅ | 실제 이미지 반전 비용 |

InstructPix2Pix (Brooks et al., 2023)는 합성 편집 쌍으로 Stable Diffusion을 미세 조정하여 지시 따르기(instruction-following) 이미지 편집을 시연했다.

DiffusionCLIP (Kim et al., 2022)은 방향성 CLIP 손실로 확산 모델 자체를 미세 조정하는 대안적 접근법을 취하며, 인간 평가에서 StyleCLIP 기준선 대비 60-80% 선호율을 달성한다.

반면 SEGA는 크로스 어텐션 맵에 대한 토큰 기반 컨디셔닝이 필요 없으며, 다중 시맨틱 변경의 조합을 허용한다.

SEGA와 달리, Prompt-to-Prompt와 같은 방법들은 크로스 어텐션 맵을 조작하는 방식이다. SEGA는 아키텍처에 어떠한 확장도 필요로 하지 않으며, 임의의 텍스트 프롬프트에 대해 즉석으로 시맨틱 벡터를 생성한다.

---

## 5. 향후 연구에 미치는 영향 및 고려사항

### 5.1 앞으로의 연구에 미치는 영향

**① 학습-불필요 편집(Training-Free Editing)의 방향 제시**

SEGA는 이미지 생성에 강력한 시맨틱 제어를 제공하며, 원본 이미지 구성을 유지하면서도 직관적이고 쉽게 제어 가능한 다양한 변경이 가능하다. 이는 추가 학습 없이 강력한 편집을 가능하게 하는 패러다임을 제시하였으며, 이후 LEDITS, LEDITS++ 등의 후속 연구에 직접적인 영향을 미쳤다.

**② 시맨틱 공간 탐구의 촉진**

SEGA는 시맨틱 제어가 모델의 노이즈 추정값만을 사용하여 단순 텍스트 설명에서 추론될 수 있음을 증명하며, 이를 통해 이 추정값이 시맨틱 제어에 부적합하다는 이전 연구를 반박한다.

**③ 공정성(Fairness) 및 안전성 연구에의 응용**

편향을 해소하기 위한 전략인 'Fair Diffusion'은 인간의 지시에 기반하여 어떤 방향으로든 편향을 이동시켜 새로운 비율을 생성할 수 있음을 시연한다. 이 도입된 제어는 데이터 필터링과 추가 학습 없이 공정성에 대한 생성 이미지 모델 지시를 가능하게 한다.

### 5.2 향후 연구 시 고려해야 할 점

**① 시드 의존성 문제 해결**

가이던스 벡터 전이는 동일한 초기 시드로 제한된다. $\epsilon$-예측이 서로 다른 초기 노이즈 잠재 변수에 따라 크게 달라지기 때문이다. 시드에 독립적인 시맨틱 벡터 학습 방법 또는 노이즈 공간 정규화 기법이 필요하다.

**② 대규모 도메인 전환의 한계 극복**

도메인 전환은 사진 리얼리즘에서 드로잉으로의 전환과 같은 경우도 포함하지만, 그 전이는 동일한 초기 시드로 제한되며 $\epsilon$-예측이 발산하는 초기 노이즈 잠재 변수에 따라 크게 달라진다.

**③ 하이퍼파라미터 민감도**

Safe Latent Diffusion(SLD)은 하이퍼파라미터 공식화가 복잡하며, DM의 노이즈 예측 공간의 수치적 특성에 대한 더 깊은 이해를 통해 개선될 수 있다. SEGA 역시 `edit_threshold`, `edit_warmup_steps`, `guidance_scale` 등 다수의 하이퍼파라미터를 조율해야 하므로, 자동 하이퍼파라미터 최적화 연구가 필요하다.

**④ 실제 이미지(Real Image) 적용 확대**

텍스트 가이드 편집 도구를 이용한 실제 이미지 편집은 주어진 이미지를 반전(invert)해야 하므로 상당한 도전을 제기한다. 이는 확산 과정에 입력으로 사용되면 입력 이미지를 생성할 노이즈 벡터의 시퀀스를 찾아야 함을 의미한다. DDPM 반전 기법과의 결합(LEDITS)은 하나의 해법이지만, 더 효율적인 역변환(inversion) 알고리즘 개발이 필요하다.

**⑤ 멀티모달 확장**

최근 다중 모달 GPT-4 등과의 협업 연구가 활발히 진행 중이며, 확산 모델을 다른 분야의 모델과 결합하는 연구들이 등장하고 있다. 확산 모델을 이미지 복원, 깊이 추정, 이미지 향상, 분류 등 비전 응용에 적용하는 연구도 있으며, 텍스트-이미지 확산 모델과 활발히 연구 중인 분야와의 추가적인 협업은 탐구할 가치가 있는 흥미로운 주제이다.

---

## 📚 참고자료 및 출처

| # | 자료명 | 출처 |
|---|---|---|
| 1 | **SEGA: Instructing Text-to-Image Models using Semantic Guidance** (Brack et al., NeurIPS 2023) | [arxiv.org/abs/2301.12247](https://arxiv.org/abs/2301.12247) |
| 2 | **NeurIPS 2023 공식 논문 PDF** | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2023/file/4ff83037e8d97b2171b2d3e96cb8e677-Paper-Conference.pdf) |
| 3 | **NeurIPS 2023 Abstract Page** | [proceedings.neurips.cc (Abstract)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4ff83037e8d97b2171b2d3e96cb8e677-Abstract-Conference.html) |
| 4 | **OpenReview — SEGA** | [openreview.net](https://openreview.net/forum?id=KIPAIy329j) |
| 5 | **Hugging Face Diffusers — Semantic Stable Diffusion** | [huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers/api/pipelines/semantic_stable_diffusion) |
| 6 | **ADS Abstract — SEGA** | [ui.adsabs.harvard.edu](https://ui.adsabs.harvard.edu/abs/2023arXiv230112247B/abstract) |
| 7 | **Semantic Scholar — SEGA** | [semanticscholar.org](https://www.semanticscholar.org/paper/SEGA:-Instructing-Diffusion-using-Semantic-Brack-Friedrich/1a984acf57d7d6dd2aa3da0ea1e563598ffad9bd) |
| 8 | **DFKI 공식 페이지 — SEGA** | [dfki.de](https://www.dfki.de/en/web/research/projects-and-publications/publication/14030) |
| 9 | **ResearchGate — SEGA: Instructing Diffusion using Semantic Dimensions** | [researchgate.net](https://www.researchgate.net/publication/367557470_SEGA_Instructing_Diffusion_using_Semantic_Dimensions) |
| 10 | **LEDITS: Real Image Editing with DDPM Inversion and Semantic Guidance** | [arxiv.org/abs/2307.00522](https://arxiv.org/html/2307.00522) |
| 11 | **InstructPix2Pix: Learning to Follow Image Editing Instructions** (Brooks et al., CVPR 2023) | [arxiv.org/abs/2211.09800](https://arxiv.org/abs/2211.09800) |
| 12 | **Prompt-to-Prompt Image Editing with Cross Attention Control** (Hertz et al., ICLR 2023) | [prompt-to-prompt.github.io](https://prompt-to-prompt.github.io/) |
| 13 | **Text-to-image Diffusion Models in Generative AI: A Survey** | [arxiv.org/abs/2303.07909](https://arxiv.org/html/2303.07909v3) |
| 14 | **Diffusion Model-Based Image Editing: A Survey** | [arxiv.org/abs/2402.17525](https://arxiv.org/html/2402.17525v1) |
| 15 | **An overview of classifier-free guidance for diffusion models** | [theaisummer.com](https://theaisummer.com/classifier-free-guidance/) |
| 16 | **NeurIPS 2023 Poster — SEGA** | [neurips.cc/virtual/2023/poster/72016](https://neurips.cc/virtual/2023/poster/72016) |
| 17 | **ACM Digital Library — SEGA** | [dl.acm.org/doi/10.5555/3666122.3667224](https://dl.acm.org/doi/10.5555/3666122.3667224) |
