
# TokenVerse: Versatile Multi-concept Personalization in Token Modulation Space

> **논문 정보**
> - **제목**: TokenVerse: Versatile Multi-concept Personalization in Token Modulation Space
> - **저자**: Daniel Garibi, Shahar Yadin, Roni Paiss, Omer Tov, Shiran Zada, Ariel Ephrat, Tomer Michaeli, Inbar Mosseri, Tali Dekel
> - **arXiv**: [2501.12224](https://arxiv.org/abs/2501.12224) (2025년 1월 21일)
> - **게재**: ACM Transactions on Graphics / ACM SIGGRAPH 2025
> - **프로젝트 페이지**: [token-verse.github.io](https://token-verse.github.io/)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

TokenVerse는 사전 학습된 텍스트-이미지 확산 모델을 활용하는 **다중 개념 개인화(multi-concept personalization)** 방법으로, 단 하나의 이미지에서도 복잡한 시각적 요소와 속성을 분리(disentangle)하고, 여러 이미지에서 추출된 개념들의 조합을 플러그 앤 플레이 방식으로 생성할 수 있다. 기존 연구와 달리 TokenVerse는 여러 이미지 각각에서 복수의 개념을 처리하며, 물체, 액세서리, 재질, 포즈, 조명 등 광범위한 종류의 개념을 지원한다.

### 주요 기여

TokenVerse는 다음 핵심 기여를 제시한다:
1. 여러 이미지에서 분리된 다중 개념 개인화 및 플러그 앤 플레이 이미지 합성을 가능하게 하는 **최초의 방법**
2. 조명 조건, 재질, 포즈 등 **객체를 넘어선 의미론적 개념** 개인화 지원
3. DiT에서 텍스트 토큰 변조(modulation)의 역할 탐구 및 **국소적이고 의미론적으로 풍부한 공간** 입증
4. 개인화된 콘텐츠 창작 및 스토리텔링에의 응용 가능성 시연

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

최근 텍스트-이미지 확산 모델의 발전으로 참조 이미지에서 객체나 스타일을 추출해 새로운 이미지를 합성하는 **개인화된 이미지 생성** 연구가 활발히 진행되었다. 그러나 기존 방법들은 **여러 이미지에서 복수의 개념을 동시에 다루는 것에 한계**가 있으며, 포즈, 재질, 조명 조건과 같은 **비-객체(non-object) 개념을 지원하지 못한다**.

구체적으로 기존 방법들의 문제점은 다음과 같다:

개인화 방법들은 대부분 특정 개념을 표현하는 특수한 텍스트 임베딩을 학습하거나, 모델 자체 레이어를 파인튜닝하는 방식에 의존한다. 이 방식들은 개념의 범위와 개수가 제한되는 문제가 있다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① DiT 모델의 Modulation 메커니즘

TokenVerse는 DiT(Diffusion Transformer) 기반의 텍스트-이미지 모델을 활용한다. 이 모델에서 입력 텍스트는 **어텐션(attention)**과 **변조(modulation, shift & scale)** 두 가지 방식으로 생성에 영향을 미친다. 변조 공간(modulation space)은 의미론적(semantic)이며, 복잡한 개념에 대한 **국소화된 제어**를 가능하게 한다.

각 DiT 블록은 어텐션 레이어, 피드포워드 MLP, 그리고 conditioning 신호를 통합하는 **변조 메커니즘(modulation mechanism)**을 포함한다.

DiT 내 변조 메커니즘은 각 트랜스포머 블록의 활성화를 수정하는 **스케일(scale)과 시프트(shift) 파라미터**를 포함하며, 이는 텍스트 기반 편집을 통해 출력 이미지를 크게 변경할 수 있게 한다.

표준 DiT 블록의 변조는 아래와 같이 표현된다:

$$
\mathbf{h}' = \boldsymbol{\gamma}(\mathbf{y}) \cdot \text{Norm}(\mathbf{h}) + \boldsymbol{\beta}(\mathbf{y})
$$

여기서:
- $\mathbf{h}$: 입력 토큰 (이미지 또는 텍스트)
- $\mathbf{y}$: 풀링된 텍스트 임베딩(pooled text embedding)에서 유도된 변조 벡터
- $\boldsymbol{\gamma}(\mathbf{y})$: 스케일(scale) 파라미터
- $\boldsymbol{\beta}(\mathbf{y})$: 시프트(shift) 파라미터

변조 블록에서 토큰들은 **풀링된 텍스트 임베딩으로부터 유도된 벡터 $\mathbf{y}$**를 통해 변조된다.

#### ② TokenVerse의 $M^+$ 공간 (개인화 변조 공간)

TokenVerse의 핵심 혁신은 **개인화된 변조 공간 $M^+$**의 도입이다. 이 공간에서 변조 벡터는 모든 토큰에 단일 전역 변조 벡터를 적용하는 대신, **각 텍스트 토큰에 대해 개별적으로 수정**될 수 있다. 이를 통해 특정 단어가 조작될 때 이미지의 어떤 요소가 변화하는지에 대한 세밀한 제어가 가능하다.

TokenVerse의 수정된 변조는 다음과 같이 나타낼 수 있다:

$$
\mathbf{y}_i^* = \mathbf{y}_i + \Delta_i
$$

여기서:
- $\mathbf{y}_i$: $i$번째 텍스트 토큰에 해당하는 기본 변조 벡터
- $\Delta_i$: $i$번째 토큰에 대해 학습된 **개인화 방향 벡터(personalized direction vector)**

각 토큰에 대해 이미지와 해당 캡션을 기반으로 최적화를 통해 $M^+$ 공간에서 고유한 방향이 학습되며, 이를 통해 모델이 특정 시각적 표현을 매우 정밀하게 개인화할 수 있다.

#### ③ 최적화 목표 (Reconstruction Objective)

주어진 개념 이미지와 대응하는 캡션으로, 각 텍스트 임베딩에 대한 개인화된 변조 벡터 조정값 $\Delta_i$를 학습하며, 이 조정값들은 **단순한 재구성 목표(reconstruction objective)**를 사용해 학습된다.

학습 손실은 표준 확산 모델의 노이즈 예측 손실로 표현된다:

$$
\mathcal{L}_\text{recon} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\bigl(\mathbf{x}_t,\, t,\, \mathbf{c}(\{\mathbf{y}_i + \Delta_i\})\bigr)\right\|^2\right]
$$

여기서:
- $t$: 확산 타임스텝
- $\mathbf{x}_0$: 원본 이미지
- $\boldsymbol{\epsilon}$: 추가된 노이즈
- $\boldsymbol{\epsilon}_\theta$: 노이즈 예측 네트워크(DiT)
- $\mathbf{c}(\{\mathbf{y}_i + \Delta_i\})$: 개인화된 변조 벡터가 적용된 conditioning

#### ④ Concept Isolation Loss

서로 다른 개념들이 학습 시 간섭하는 문제를 방지하기 위해, TokenVerse는 **개념 고립 손실(concept isolation loss)**을 도입한다. 이 손실은 생성된 이미지와 대응 프롬프트 집합에서 동작하며, 학습된 방향들이 관련 없는 개념에 부정적 영향을 미치지 않도록 보장한다.

개념 이미지와 결합된 캡션으로 모델을 실행하고, 최적화된 방향은 입력 프롬프트의 토큰에만 적용된다. 모델 출력과 기본 모델 출력 사이의 $L_2$ 손실을 개념 이미지에 해당하는 부분에만 적용하여, **학습된 방향이 텍스트에 매칭되는 부분에만 영향을 미치도록** 유도한다.

총 손실 함수는 다음과 같이 구성할 수 있다:

$$
\mathcal{L}_\text{total} = \mathcal{L}_\text{recon} + \lambda \cdot \mathcal{L}_\text{isolation}
$$

여기서 $\lambda$는 isolation loss의 가중치이다.

---

### 2-3. 모델 구조

확산 모델은 초기에는 UNet 기반 아키텍처를 사용했지만, 최신 최고 성능 텍스트-이미지 모델들은 **DiT(Diffusion Transformer)**를 백본으로 사용한다. UNet과 달리 DiT에서는 텍스트 토큰이 이미지 토큰과 함께 트랜스포머로 처리된다. **조인트 어텐션 레이어(joint attention layers)**가 이미지-텍스트 토큰 간의 양방향 상호작용을 촉진하며, 텍스트와 시각 정보의 통합을 가능하게 한다.

TokenVerse는 **Flux-dev** 모델을 기반으로 구현되며, 이 모델은 **58개의 DiT 블록**과 **차원 3072의 변조 벡터**를 가진다.

구조 요약:

| 구성 요소 | 설명 |
|---|---|
| 백본 모델 | Flux-dev (DiT 기반) |
| DiT 블록 수 | 58개 |
| 변조 벡터 차원 | 3072 |
| 학습 대상 | 각 텍스트 토큰의 변조 방향 벡터 $\Delta_i$ |
| 추가 모듈 | 블록별 MLP (per-block MLP) |

2단계 학습 과정을 통해 각 텍스트 토큰에 대한 **전역 변조 벡터 오프셋(global modulation vector offset)**과 **블록별 변조 벡터 오프셋(per-block modulation vector offset)**을 최적화하며, 입력 개념들의 세밀한 시각적 속성을 포착하면서도 높은 충실도를 유지한다.

#### 2단계 최적화 과정:

최적화 과정은 **2단계**로 이루어진다. **1단계**에서는 각 텍스트 토큰에 대한 전역 방향(global direction)을 **800 스텝** 동안 최적화한다.

**2단계 최적화**는 먼저 높은 노이즈 수준에서 거친(coarse) 개념 특징을 효과적으로 포착하고, 이후 낮은 노이즈 수준에서 세밀한 변환에 집중한다. 표현력을 높이기 위해 **블록별 MLP(per-block MLP)**가 토큰 특정 조정을 제공한다.

---

### 2-4. 성능 향상

TokenVerse는 정량적·정성적으로 평가되었으며, 이미지에서 다중 개념을 정확히 추출하고 이를 새로운 이미지 생성에 활용하는 데 있어 **기존 방법들을 능가**하는 것으로 입증되었다.

DreamBench++ 기반 정량적 평가에서 **개념 보존도(Concept Preservation, CP)**와 **프롬프트 충실도(Prompt Fidelity, PF)** 두 지표 모두에서 우수한 성능을 보였다. 사용자 연구에서도 개념 보존과 프롬프트 충실도 모두에서 우월한 성능이 확인되었다.

동시대(contemporaneous) 접근법들과 비교해 TokenVerse는 개념 추출과 다중 개념 합성 태스크 모두에서 우월한 성능을 보이며, DreamBench++ 등의 벤치마크를 통한 정량적 평가로 이를 뒷받침한다.

#### 텍스트 증강을 통한 추가 성능 향상:

실험적으로 **이미지당 여러 텍스트 설명을 할당**하면 모델의 개념 분리 능력이 향상된다는 것을 발견했다. 이를 위해 원본 설명과 유사하지만 단어 순서를 다르게 배치한 프롬프트로 텍스트를 증강하며(LLM으로 생성 가능), **랜덤 플립과 미러링** 같은 이미지 증강도 적용한다.

---

### 2-5. 한계

유사한 개념들이 독립적으로 추출될 때 **블렌딩(blending)** 문제가 발생할 수 있고, 서로 다른 이미지 간의 **토큰 이름 충돌(collision)**이 일어날 수 있다. 이에 대한 완화 방법으로 **조인트 학습(joint training)**과 **맥락적 토큰 차별화(contextual token differentiation)**가 제안되었다.

TokenVerse는 **테스트 시마다 각 새로운 개념 이미지에 대해 파인튜닝이 필요**하여 시간 소모적이며, 단일 학습 이미지에 대해 과적합(overfit)되는 경향이 있어 최적이 아닌 결과를 초래할 수 있다.

---

## 3. 일반화 성능 향상 가능성

### 3-1. 모델 가중치 보존을 통한 일반화

TokenVerse는 **모델의 가중치를 조정하지 않고도** 복잡한 개념을 표현할 만큼 충분히 표현력 있어서 사전 학습 분포(prior)를 보존한다. 또한, 시각적 단서(예: 세그멘테이션 마스크)가 아닌 **의미론적 텍스트 토큰**으로 시각 요소를 정의하기 때문에 겹쳐 있는 객체의 개인화와 포즈, 조명, 재질 같은 비-객체 개념도 지원한다. 이 접근 방식은 **고도로 모듈화**되어 여러 이미지에서 추출된 개념들을 원활하게 조합할 수 있다.

### 3-2. Plug-and-Play 합성과 스케일 가능성

프레임워크의 **모듈화 설계(modular design)**는 서로 다른 이미지에서 시각적 개념을 별도로 학습할 수 있게 하여, 관리하고 조합할 수 있는 개념의 수에서의 **확장성(scalability)**을 촉진한다.

추론 시, 사전 학습된 방향 벡터들이 텍스트 임베딩에 더해져 개인화된 개념을 생성된 이미지에 주입한다. TokenVerse의 즉각적인 응용으로는 동일한 객체와 장면이 등장하는 이미지들로 구성된 내러티브를 생성하는 **스토리텔링**이 있다.

### 3-3. 일반화의 한계와 향후 개선 방향

TokenVerse가 피사체 보존을 위한 변조를 탐구했지만, **파인튜닝 전략에 의존**하는 측면이 있어, 재학습 없이 새롭고 관측되지 않은 피사체에 대한 적응성이 제한될 수 있다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. 주요 선행 연구 요약

| 연구 | 연도 | 접근 방식 | 한계 |
|---|---|---|---|
| **Textual Inversion** (Gal et al.) | 2022 | 새로운 pseudo-word 임베딩 최적화 | 단일 개념, 표현력 제한 |
| **DreamBooth** (Ruiz et al.) | 2022/2023 | 전체 모델 파인튜닝 + 고유 식별자 | 다중 개념 처리 어려움, 언어 드리프트 |
| **Custom Diffusion** (Kumari et al.) | 2023 | Key/Value 레이어 + modifier 토큰 최적화 | 다중 개념 충돌 문제 |
| **Break-A-Scene** (Avrahami et al.) | 2023 | 단일 이미지에서 다중 개념 추출 | 비-객체 개념 미지원 |
| **LoRA-Composer** (Yang et al.) | 2024 | LoRA 기반 훈련 불필요 다중 개념 | 추상적 개념 처리 미흡 |
| **OMG** (Kong et al.) | 2024 | 가림(occlusion)에 강한 다중 개념 생성 | 비-객체 개념 한계 |
| **TokenVerse** (Garibi et al.) | 2025 | DiT 변조 공간 방향 학습 | 테스트 시 파인튜닝 필요, 느린 속도 |

**Textual Inversion**은 각 입력 개념을 나타내는 pseudo-word 텍스트 임베딩을 학습하여 개인화된 텍스트-이미지 생성을 도입했으며, 객체용과 스타일용 두 개의 학습된 pseudo-word를 사용해 합성 생성을 지원한다. 이를 기반으로 **Custom Diffusion**은 modifier 토큰과 함께 모델의 key, value 프로젝션 레이어를 최적화하여 다중 개념 맞춤형 생성을 도입했다.

**DreamBooth**는 텍스트 트랜스포머를 동결한 상태에서 확산 모델의 모든 파라미터를 파인튜닝하고 생성 이미지를 정규화 데이터셋으로 사용한다. **Textual Inversion**은 각 개념에 대해 새로운 단어 임베딩 토큰만 최적화한다.

TokenVerse는 이미지에서 조명 조건, 재질 표면과 같은 **추상적 개념을 분리하는 것을 지원하는 최초의 분리된 다중 개념 개인화 방법**으로 제안되었다.

### 4-2. 후속 연구: Mod-Adapter와의 비교

TokenVerse는 객체와 추상적 개념 모두를 지원하는 다중 개념 개인화 프레임워크이지만, **테스트 시마다 각 새로운 개념 이미지에 대해 파인튜닝이 필요**하여 시간 소모적이고 과적합 경향이 있다. 이에 대응하여 **Mod-Adapter**는 객체와 추상 개념 모두에서 다목적 다중 개념 개인화를 가능하게 하는 **훈련 불필요(tuning-free) 프레임워크**를 제시한다.

---

## 5. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

1. **DiT 변조 공간 탐구의 선구적 역할**

   TokenVerse는 텍스트 프롬프트를 통한 제어된 이미지 생성 분야에서 중요한 진전을 나타내며, 여러 개념의 분리된 개인화를 높은 충실도로 가능하게 한다. 그 모듈화되고 유연한 아키텍처는 콘텐츠 창작에서 스토리텔링에 이르는 다양한 응용을 위한 길을 열어준다.

2. **스토리텔링 및 콘텐츠 창작 분야로의 확장**

   TokenVerse는 이러한 도전들을 극복하는 다중 개념 개인화의 첫 번째 방법으로, DiT의 변조 공간에서 텍스트 토큰별 방향을 추출하는 것이 풍부하고 의미론적임을 보여준다. 이 방법은 스토리텔링에서 개인화된 콘텐츠 창작에 이르는 **다양한 응용의 문을 열어준다**.

3. **후속 연구 파생**

   TokenVerse의 영향으로 **Nested Attention**, **UnZipLoRA**, **OmniPrism**, **DECOR**, **Multi-subject Open-set Personalization in Video Generation** 등의 관련 후속 연구들이 등장하였다.

### 5-2. 향후 연구 시 고려할 점

1. **튜닝 불필요(Tuning-free) 방향 탐색**

   기존의 개인화 생성 방법들은 주로 일반 객체 개념(예: 동물, 사물)에 집중하며 비-객체/추상 개념(포즈, 조명)의 개인화에 어려움이 있다. TokenVerse가 이를 지원하지만, **테스트 시 파인튜닝이 필요해 시간이 많이 소요되고 과적합 경향**이 있으므로, 이를 해결하는 튜닝 불필요 접근법 개발이 중요한 연구 방향이다.

2. **개념 간 충돌 및 블렌딩 문제 해결**

   유사한 개념이 독립적으로 추출될 때의 **블렌딩 문제**와 서로 다른 이미지에서 토큰 이름 충돌 문제가 남아 있으며, **조인트 학습** 및 **맥락적 토큰 차별화** 방안에 대한 추가 연구가 필요하다.

3. **영상(Video) 생성으로의 확장**

   TokenVerse의 다목적 능력은 **스토리텔링 및 개인화된 콘텐츠 창작**에서의 응용을 시사하며, 이는 특히 연속된 비디오 프레임에서의 개념 일관성 유지 연구로 이어질 가능성이 크다.

4. **DreamBench++ 등 표준 벤치마크 및 평가 지표 발전**

   향후 연구는 단일 및 다중 개념 개인화를 **개념 보존도(CP)**와 **프롬프트 충실도(PF)** 두 관점에서 평가해야 하며, 멀티모달 LLM(예: GPT-4o) 기반의 정성적+정량적 통합 평가 방식 도입이 바람직하다.

5. **모델 확장성과 실용성 강화**

   기존 방식은 새로운, 관측되지 않은 피사체에 대한 적응성이 **재학습 없이 제한**되므로, 사전 학습된 표현을 최대한 활용하면서도 새로운 개념에 대한 일반화를 달성하는 연구가 필요하다.

---

## 📚 참고 자료 및 출처

| # | 제목 / 출처 | URL |
|---|---|---|
| 1 | **TokenVerse** (arXiv, 2025) | https://arxiv.org/abs/2501.12224 |
| 2 | **TokenVerse 프로젝트 페이지** | https://token-verse.github.io/ |
| 3 | **TokenVerse** (ACM Transactions on Graphics / SIGGRAPH 2025) | https://dl.acm.org/doi/10.1145/3730843 |
| 4 | **TokenVerse** (HuggingFace Papers) | https://huggingface.co/papers/2501.12224 |
| 5 | **TokenVerse HTML 전문** (arXiv HTML) | https://arxiv.org/html/2501.12224v1 |
| 6 | **TokenVerse Literature Review** (Moonlight) | https://www.themoonlight.io/en/review/tokenverse-versatile-multi-concept-personalization-in-token-modulation-space |
| 7 | **TokenVerse** (EmergentMind) | https://www.emergentmind.com/papers/2501.12224 |
| 8 | **Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization** (arXiv, 2025) | https://arxiv.org/html/2501.12224v1 |
| 9 | **Custom Diffusion: Multi-Concept Customization of Text-to-Image Diffusion** (CVPR 2023) | https://www.cs.cmu.edu/~custom-diffusion/ |
| 10 | **DreamBooth: Fine Tuning Text-to-Image Diffusion Models** (CVPR 2023) | https://arxiv.org/abs/2208.12242 |
| 11 | **Textual Inversion: An Image is Worth One Word** (ICLR 2023) | Gal et al., 2022 |
| 12 | **Break-A-Scene: Extracting Multiple Concepts from a Single Image** (SIGGRAPH Asia 2023) | Avrahami et al., 2023 |
| 13 | **DreamBench++: A Human-Aligned Benchmark for Personalized Image Generation** (arXiv 2024) | arXiv:2406.16855 |
| 14 | **OMG: Occlusion-friendly Personalized Multi-concept Generation** (arXiv 2024) | arXiv:2403.10983 |
| 15 | **LoRA-Composer** (arXiv 2024) | arXiv:2403.11627 |
