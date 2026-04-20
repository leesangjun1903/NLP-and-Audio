
# CLoRA: Contrastive Test-Time Composition of Multiple LoRA Models for Image Generation

---

## 📌 논문 기본 정보

| 항목 | 내용 |
|------|------|
| **제목** | Contrastive Test-Time Composition of Multiple LoRA Models for Image Generation |
| **저자** | Tuna Han Salih Meral, Enis Simsar, Federico Tombari, Pinar Yanardag |
| **발표** | ICCV 2025 (arXiv 최초 공개: 2024년 3월 28일) |
| **arXiv ID** | 2403.19776 |

---

## 1️⃣ 핵심 주장 및 주요 기여 요약

LoRA(Low-Rank Adaptation)는 이미지 생성 분야에서 개인화(personalization)를 위한 강력하고 인기 있는 기법으로 자리 잡았으며, 단일 개념(예: 특정 개 또는 고양이)을 표현하는 데는 탁월한 성능을 보인다. 그러나 복수의 LoRA 모델을 하나의 이미지에서 다양한 개념을 포착하는 데 사용하는 것은 여전히 상당한 도전 과제로 남아 있다.

기존 방법들은 서로 다른 LoRA 모델 내부의 어텐션 메커니즘이 겹치는 문제(attention overlap)로 인해 한 개념이 완전히 무시되거나(예: 개를 생략), 개념이 잘못 결합되는(예: 고양이 한 마리와 개 한 마리 대신 고양이 두 마리가 생성) 문제를 겪는다.

**CLoRA**는 학습 없이(training-free) 테스트 타임(test-time)에 작동하며, 대조 학습(contrastive learning)을 사용하여 여러 개념 및 스타일 LoRA를 동시에 합성한다.

### 🔑 주요 기여 (Key Contributions)

CLoRA의 핵심 기여는 다음과 같다:
- 대조적 목표(contrastive objective)를 기반으로 여러 콘텐츠·스타일 LoRA를 동시에 통합하는 새로운 접근법 제안
- 테스트 타임의 어텐션 맵을 기반으로 잠재 표현을 동적으로 업데이트하고, 별도의 LoRA 모델에 대응하는 크로스-어텐션 맵으로부터 마스크를 유도하여 다중 잠재 표현을 융합
- 추가 학습이나 별도의 제어 입력이 필요 없는 완전 학습-불필요(training-free) 방식

또한 특화된 LoRA 변형이 필요하지 않으며, civit.ai의 커뮤니티 LoRA를 플러그-앤-플레이(plug-and-play) 방식으로 직접 사용할 수 있다.

이 방법은 메모리 사용량 및 런타임 측면에서 매우 효율적이며, 여러 LoRA 모델을 처리할 수 있도록 확장 가능하다.

---

## 2️⃣ 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 향상 및 한계

### 🔴 2-1. 해결하고자 하는 문제

여러 개념의 LoRA를 하나의 이미지에 매끄럽게 혼합하는 것은 상당한 도전 과제이다. 일반적인 접근법들은 서로 다른 LoRA 모델 간의 어텐션 메커니즘이 겹쳐, 한 개념이 완전히 무시되거나 개념들이 잘못 결합되는 문제가 발생한다.

핵심 문제는 두 가지이다:

1. **Attention Overlap (어텐션 중첩)**: 한 LoRA의 어텐션이 다른 LoRA의 어텐션 영역을 침범하여 개념 하나가 완전히 사라지는 문제
2. **Attribute Binding (속성 결합 오류)**: 한 개념의 속성이 다른 개념에 잘못 붙어버리는 문제

### 🟢 2-2. 제안 방법 (수식 포함)

#### ① LoRA의 수식적 표현

LoRA는 기반 레이어를 동결(freeze)하면서 랭크 분해 행렬(rank-decomposition matrices)을 도입하여 대형 모델을 미세조정한다. Stable Diffusion에서 LoRA는 텍스트와 이미지 연결을 담당하는 크로스-어텐션 레이어에 적용된다. 공식적으로, LoRA 모델은 저차원 행렬 쌍 $(W_{out}, W_{in})$으로 표현되며, 이 행렬들은 사전 학습 모델의 $W$ 가중치에 가해지는 조정 사항을 캡처한다. 이미지 생성 중 업데이트된 가중치는 다음과 같이 계산된다:

$$W' = W + W_{in} W_{out}$$

#### ② Stable Diffusion의 Diffusion 과정

Stable Diffusion은 오토인코더의 잠재 공간에서 동작하며, 인코더 $E$와 디코더 $D$로 구성된다. 인코더는 입력 이미지 $x$를 저차원 잠재 코드 $z = E(x)$로 매핑하고, 디코더는 $D(z) \approx x$로 이미지를 복원한다. 확산 과정은 원래의 잠재 코드 $z_0$에 점진적으로 노이즈를 추가하여 타임스텝 $t$에서 $z_t$를 생성하고, UNet 기반 디노이저 $\epsilon_\theta$가 노이즈를 예측하고 제거하도록 학습된다.

Stable Diffusion의 학습 목적 함수:

$$\mathcal{L}_{SD} = \mathbb{E}_{z_0, \epsilon \sim \mathcal{N}(0,I), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right]$$

#### ③ CLoRA의 대조 목적 함수 (Contrastive Objective)

생성이 입력 프롬프트에 충실하도록 보장하기 위해, 추론(inference) 중에 대조 학습을 적용한다. 대조 목적 함수로는 빠른 수렴으로 알려진 **InfoNCE 손실**을 사용하며, 이 손실은 크로스-어텐션 맵의 쌍(pair)들에 대해 동작한다. 동일한 그룹 내의 어텐션 맵 쌍은 **양성(positive)** 쌍으로, 서로 다른 그룹의 항목으로 구성된 쌍은 **음성(negative)** 쌍으로 레이블링된다.

예를 들어, L1 적용 프롬프트의 어텐션 맵들은 서로 가까워지도록 유도되는데, 이는 L1 LoRA를 해당 주제인 고양이와 정렬시키기 위해서다. 반면, 서로 다른 개념 그룹 C1과 C2(예: 원래 프롬프트에서 고양이와 개의 어텐션 맵)는 어텐션 중첩 문제를 방지하기 위해 서로 반발하도록 음성 쌍을 형성한다.

**InfoNCE 손실 (NT-Xent)** 의 일반적 형태:

$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(A_i^+, A_j^+) / \tau)}{\exp(\text{sim}(A_i^+, A_j^+) / \tau) + \sum_{k=1}^{N} \exp(\text{sim}(A_i^+, A_k^-) / \tau)}$$

여기서:
- $A_i^+$: 양성 어텐션 맵 쌍 (같은 LoRA 그룹)
- $A_k^-$: 음성 어텐션 맵 쌍 (다른 LoRA 그룹)
- $\tau$: 온도 파라미터 (논문에서는 $\tau = 0.5$ 사용)
- $\text{sim}(\cdot, \cdot)$: 코사인 유사도

전체 InfoNCE 손실은 모든 양성 쌍에 대해 평균하여 계산된다.

#### ④ Latent Optimization (잠재 표현 최적화)

대조 손실을 기반으로 이미지의 잠재 표현이 업데이트된다. 잠재 표현은 확산 모델이 사용하는 이미지의 압축 버전으로, 손실 함수가 잠재 표현을 올바른 방향으로 유도한다. 업데이트는 학습률 $\alpha_t$를 사용하는 경사 하강법(gradient descent) 방식으로 이루어진다.

잠재 업데이트 규칙:

$$z_t \leftarrow z_t - \alpha_t \cdot \nabla_{z_t} \mathcal{L}_{InfoNCE}$$

#### ⑤ Masked Latent Fusion (마스킹된 잠재 표현 융합)

역방향 단계(backward step) 후, Stable Diffusion에서 생성된 잠재 표현을 추가 LoRA 모델에서 파생된 잠재 표현과 결합한다. 직접 결합이 가능하지만, 각 LoRA가 이미지의 적절한 영역에만 기여하도록 마스킹 메커니즘을 도입한다.

각 LoRA에 대해 "마스크"를 생성하며, 이 마스크는 서로 다른 LoRA들이 과도하게 간섭하는 것을 방지하도록 설계된다.

마스크 기반 잠재 표현 융합의 형태:

$$z_{fused} = M_1 \odot z^{L_1} + M_2 \odot z^{L_2} + \ldots + M_n \odot z^{L_n}$$

여기서 $M_i$는 크로스-어텐션 맵으로부터 유도된 $i$번째 LoRA의 의미론적 마스크이다.

### 🏗️ 2-3. 모델 구조

이 방법은 Stable Diffusion 1.5(SDv1.5) 모델을 기반으로 구현된다.

전체 프로세스는 **프롬프트 분해(prompt breakdown)**, **어텐션 유도 확산(attention-guided diffusion)**, **대조 손실을 통한 일관성 확보(contrastive loss for consistency)** 의 세 단계로 구성된다.

의미론적으로 가장 풍부한 정보를 담고 있는 $16 \times 16$ 어텐션 맵을 활용한다.

#### 최적화 전략

최적화는 $i \in \{0, 10, 20\}$ 반복에서 수행되며, 아티팩트 방지를 위해 $i = 25$ 이후에는 최적화를 중단한다. 대조 학습에서 온도 파라미터는 $\tau = 0.5$로 설정한다. 두 개의 LoRA 모델을 합성하는 데 약 25초가 소요되며, 단일 H100 GPU에서 최대 8개의 LoRA까지 성공적으로 결합 가능하다.

### 📊 2-4. 성능 향상

DINO 기반 유사도 메트릭을 사용하여 모든 방법을 평가한 결과, CLoRA는 LoRA 콘텐츠를 충실하게 병합하는 면에서 기존 베이스라인들을 능가하는 성능을 보였다.

비교 대상 베이스라인으로는 가중 결합 방식의 LoRA-Merge, 새로운 LoRA 모델을 합성하는 ZipLoRA, 특정 LoRA 변형 학습이 필요한 Mix-of-Show, 그리고 Custom Diffusion 등이 포함된다.

**기존 방법들의 한계:**
- MultiLoRA는 특정 LoRA 모델을 닮지 못하고 고양이 두 마리 또는 펭귄 두 마리를 생성하는 실패를 보인다. LoRA-Merge는 의도된 LoRA와 어느 정도 일치하는 고양이는 생성하지만 펭귄을 정확히 담아내지 못한다. ZipLoRA는 다수의 콘텐츠 LoRA 결합에 대한 설계 제약으로 봉제 펭귄을 자주 누락하고 고양이 두 마리를 생성한다. Custom Diffusion은 고양이 LoRA를 완전히 무시하고 봉제 펭귄만 생성한다.

**CLoRA의 성능:**
- CLoRA는 세 개의 LoRA를 사용한 합성 이미지를 성공적으로 생성하고, 인간을 포함한 사실적 합성을 처리하며, 스타일·오브젝트·인물 LoRA를 사용한 이미지를 매끄럽게 합성할 수 있다.

### ⚠️ 2-5. 한계점

CLoRA의 효과는 기반 LoRA 모델의 품질에 의존한다는 점이 논문에서 인정되고 있다. 또한 딥페이크 생성 등 강력한 이미지 생성 도구의 오남용과 관련된 윤리적 우려도 제기된다.

예를 들어, 책 LoRA와 컵 LoRA가 결합될 때 책의 표지가 컵에 나타나는 방식으로 블렌딩되는 경우가 있으며, 논문에서도 책과 컵 오브젝트의 정체성을 묘사하는 데 어려움이 있음을 인정하지만 객체를 혼합하지 않는 합성은 성공한다고 밝힌다.

---

## 3️⃣ 모델의 일반화 성능 향상 가능성

### 🔷 플러그-앤-플레이(Plug-and-Play) 일반화

기존 방법들과 달리, CLoRA는 특화된 LoRA 변형이나 추가 학습이 필요 없으며, civit.ai의 커뮤니티 LoRA를 플러그-앤-플레이 방식으로 직접 사용할 수 있다. 또한 테스트 타임(test-time) 특성으로 인해 계산 효율적이며, 여러 LoRA 모델을 결합하여 이미지를 생성하는 데 약 1분이 소요된다.

### 🔷 스케일 확장성 (Scalability)

이 방법은 메모리 사용량과 런타임 측면에서 매우 효율적이며, 여러 LoRA 모델을 처리할 수 있도록 확장 가능하다.

단일 H100 GPU에서 최대 8개의 LoRA까지 성공적으로 결합할 수 있다.

### 🔷 다양한 LoRA 유형 처리

CLoRA는 스타일 LoRA, 오브젝트 LoRA, 인물 LoRA를 동시에 처리할 수 있어 다양한 유형의 LoRA 조합에 대한 높은 일반화 성능을 보인다.

### 🔷 훈련 불필요(Training-Free) 특성이 주는 일반화 이점

이 접근법은 학습이 필요 없으며 일반적인 방법들과 달리 추가적인 제어가 필요하지 않다.

이는 특정 도메인에 과적합될 위험이 없으며, 다양한 새로운 개념 조합에 자유롭게 적용할 수 있음을 의미한다.

### 🔷 잠재적 일반화 한계

기반 모델이 Stable Diffusion v1.5에 국한되어 있으며, SDXL 등 더 강력한 최신 베이스 모델로의 직접 적용 가능성은 논문에서 별도로 논의되지 않는다. 또한 어텐션 맵 최적화는 여전히 베이스 확산 모델의 표현 능력에 의존하므로, 베이스 모델이 잘 처리하지 못하는 개념에 대해서는 한계가 있을 수 있다.

---

## 4️⃣ 2020년 이후 관련 최신 연구 비교 분석

### 📊 주요 비교 연구 정리

| 방법 | 특징 | CLoRA와의 차이 |
|------|------|----------------|
| **LoRA-Merge** | 가중치 산술 결합 | 학습-불필요이나 어텐션 중첩 해결 못함 |
| **ZipLoRA** (ECCV 2024) | 스타일+콘텐츠 LoRA 병합 | 다수의 콘텐츠 LoRA 결합에 취약 |
| **Mix-of-Show** | EDLoRA 변형 학습 필요 | 추가 학습 및 ControlNet 제어 필요 |
| **Custom Diffusion** | 다중 개념 미세조정 | 재학습 필요, LoRA 무시 문제 |
| **Multi-LoRA Composition** | LoRA Switch/Composite | 어텐션 맵 기반이 아님 |
| **LoRACLR** (2024) | 가중치 공간 대조 병합 | 사전학습 모델 재활용하나 가중치 공간 기반 |

#### ZipLoRA (Shah et al., ECCV 2024)
기존 기법들은 학습된 스타일과 주제의 결합 생성 문제를 신뢰성 있게 해결하지 못하여, 주체 충실도 또는 스타일 충실도가 타협된다. ZipLoRA는 독립적으로 학습된 스타일과 주제 LoRA를 저렴하고 효과적으로 병합하여, 사용자가 제공한 어떤 스타일로든 어떤 주제든 생성하는 것을 목표로 한다. ZipLoRA는 주제 및 스타일 충실도에서 베이스라인 대비 의미 있는 개선을 보이며 재맥락화 능력을 유지하는 결과를 보인다.

그러나 ZipLoRA는 스타일+콘텐츠 LoRA 단일 쌍에 특화되어 있어 다수의 콘텐츠 LoRA 처리에 한계가 있다.

#### Multi-LoRA Composition (Zhong et al., TMLR 2024)
이 연구는 디코딩 중심 관점에서 다중 LoRA 합성 문제를 연구하며, LoRA Switch(각 디노이징 단계에서 서로 다른 LoRA를 번갈아 사용)와 LoRA Composite(모든 LoRA를 동시에 통합하여 더 응집력 있는 이미지 합성 유도)라는 두 가지 학습-불필요 방법을 제시한다.

테스트-타임 LoRA 합성 방법인 Multi LoRA Composite와 Switch 방식도 제안되었으나, 이들은 어텐션 맵을 기반으로 동작하지 않아 만족스럽지 못한 결과를 낳을 수 있다.

#### LoRACLR (2024)
LoRACLR은 각각 별개의 개념에 대해 미세조정된 여러 LoRA 모델을 추가 개별 미세조정 없이 단일 통합 모델로 병합하는 다중 개념 이미지 생성 접근법이다. LoRACLR은 대조 목적 함수를 사용하여 이들 모델의 가중치 공간을 정렬하고 병합하여 호환성을 보장하면서 간섭을 최소화한다. 이로써 고품질 다중 개념 이미지 합성을 위한 효율적이고 확장 가능한 모델 합성이 가능하다.

> CLoRA와 LoRACLR은 모두 대조 학습을 활용하지만, CLoRA는 **어텐션 맵을 직접 조작**하는 테스트-타임 최적화인 반면, LoRACLR은 **가중치 공간에서의 정렬**을 수행한다.

#### Mixture of LoRA Experts (MoLE, Wu et al., ICLR 2024)
MoLE는 계층적 제어와 제약 없는 브랜치 선택을 활용하여 여러 LoRA 컴포넌트를 더 강건하고 다용도로 융합하는 접근법이다. LoRA의 플러그-앤-플레이 특성으로 인해 연구자들은 모델이 다양한 다운스트림 작업에서 뛰어난 성능을 발휘할 수 있도록 여러 LoRA의 합성을 탐구한다.

그러나 이 방법은 각 도메인별로 학습이 필요한 학습 가능한 게이팅 함수(learnable gating functions)가 필요하다는 한계가 있다.

---

## 5️⃣ 앞으로의 연구에 미치는 영향 및 고려사항

### 🔵 5-1. 향후 연구에 미치는 영향

#### ① 테스트-타임 최적화 패러다임의 확장
CLoRA는 **학습 없는 테스트-타임 최적화**가 복잡한 다중 개념 합성 문제를 해결할 수 있음을 증명했다. 이는 재학습 없이 다양한 사용자 요구에 즉각적으로 대응하는 적응형 생성 모델 연구를 촉진할 것이다.

#### ② 대조 학습의 이미지 생성 분야 적용 확대

대조 LoRA 변조 기법은 이미지 합성, 다중 모달 증분 학습, 어댑터 효율적 LLM 배포에서의 적용에 있어 성숙한 기술 방향을 나타내며, 일반화 및 제어 특성을 향상하는 지속적인 연구가 기대된다.

#### ③ 오픈소스 기여와 벤치마크 제공

소스 코드, 벤치마크 데이터셋, 학습된 LoRA 모델을 공유하여 이 주제에 대한 추가 연구를 촉진한다.

이를 통해 후속 연구들이 공정하게 비교 평가할 수 있는 기반이 마련되었다.

#### ④ 커뮤니티 LoRA 생태계와의 통합 가능성

LoRA 모델들은 연구자, 개발자, 예술가들 사이에서 큰 인기를 얻었으며, civit.ai 같은 플랫폼에는 특정 캐릭터, 의상 스타일, 시각적 요소에 맞춤화된 10만 개 이상의 LoRA 모델이 등록되어 있다.
CLoRA의 plug-and-play 특성은 이 방대한 커뮤니티 자원을 즉시 활용 가능하게 한다.

### 🟠 5-2. 앞으로 연구 시 고려할 점

#### ① 더 강력한 베이스 모델로의 확장
현재 CLoRA는 Stable Diffusion v1.5를 기반으로 한다. **SDXL, SD3, FLUX** 등 더 강력한 최신 베이스 모델로의 확장 연구가 필요하다. 더 큰 모델에서의 어텐션 맵 구조 변화가 CLoRA의 최적화 전략에 어떤 영향을 미치는지 탐구해야 한다.

#### ② 동적 개념 수 처리
현재 방법은 최대 8개의 LoRA까지 지원하지만, 더 많은 LoRA를 효율적으로 처리하기 위한 **확장 가능한 메모리·시간 효율성** 연구가 필요하다. 개념 수 증가에 따른 어텐션 맵 분리 효과 저하 문제도 해결해야 한다.

#### ③ 객체 정체성 보존의 한계 극복
현재 방법은 책과 컵 같은 오브젝트의 정체성을 정확히 묘사하는 데 어려움이 있다.
이를 해결하기 위해 **정체성 보존 손실(identity preservation loss)** 또는 더 세밀한 의미론적 마스킹 전략이 필요하다.

#### ④ 적대적 개념 간의 갈등 해결
복수의 LoRA가 시각적으로 유사하거나 같은 공간을 차지하는 개념을 표현할 때 발생하는 갈등(conflict) 해결 메커니즘이 추가로 연구되어야 한다.

#### ⑤ 자동화된 평가 프레임워크 개선
현재 DINO 기반 유사도 메트릭으로 평가가 이루어지는데, 다중 개념 이미지 생성의 복잡한 특성을 더 정확히 평가하는 새로운 메트릭 연구가 필요하다. Multi-LoRA Composition 논문의 GPT-4V 기반 평가 방법론과 결합한 더 종합적인 평가 체계 구축이 요구된다.

#### ⑥ 윤리적 안전장치 연구
이 방법은 최소한의 노력으로 개인화된 이미지 생성을 가능하게 하고 예술과 디자인에서 변혁적 기회의 문을 열지만, 잠재적 오남용 방지를 위한 윤리적 사용에 대한 포괄적이고 신중한 논의가 필요하다.

---

## 📚 참고 자료 및 출처

1. **Meral, T. H. S., Simsar, E., Tombari, F., & Yanardag, P.** (2025). *Contrastive Test-Time Composition of Multiple LoRA Models for Image Generation*. ICCV 2025, pp. 18090-18100.
   - arXiv: https://arxiv.org/abs/2403.19776
   - ICCV Paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Meral_Contrastive_Test-Time_Composition_of_Multiple_LoRA_Models_for_Image_Generation_ICCV_2025_paper.pdf
   - Project Page: https://clora-diffusion.github.io/

2. **Zhong, M. et al.** (2024). *Multi-LoRA Composition for Image Generation*. arXiv:2402.16843 / TMLR 2024.
   - https://arxiv.org/abs/2402.16843

3. **Shah, V. et al.** (2024). *ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs*. ECCV 2024.
   - https://link.springer.com/chapter/10.1007/978-3-031-73232-4_24

4. **Yang, Y. et al.** (2024). *LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models*. arXiv:2403.11627.

5. **LoRACLR** (2024). *LoRACLR: Contrastive Adaptation for Customization of Diffusion Models*. arXiv:2412.09622.
   - https://arxiv.org/html/2412.09622

6. **Wu, X., Huang, S., & Wei, F.** (2024). *MoLE: Mixture of LoRA Experts*. ICLR 2024.
   - https://openreview.net/forum?id=uWvKBCYh4S

7. **OpenReview (CLoRA)**: https://openreview.net/forum?id=Mzz9i4Zf8B

8. **Semantic Scholar (CLoRA)**: https://www.semanticscholar.org/paper/CLoRA:-A-Contrastive-Approach-to-Compose-Multiple-Meral-Simsar/51121c5c73f3a0dd25995fde5bb76c560e41ccd4

9. **Literature Review (themoonlight.io)**: https://www.themoonlight.io/en/review/clora-a-contrastive-approach-to-compose-multiple-lora-models

10. **Emergent Mind — Contrastive LoRA Modulation**: https://www.emergentmind.com/topics/contrastive-lora-modulation-technique
