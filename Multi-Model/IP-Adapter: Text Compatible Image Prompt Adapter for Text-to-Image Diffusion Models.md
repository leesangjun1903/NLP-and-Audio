# IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models

**논문 정보:** Ye, Hu; Zhang, Jun; Liu, Sibo; Han, Xiao; Yang, Wei (Tencent AI Lab, 2023), arXiv:2308.06721

---

## 1. 핵심 주장 및 주요 기여 요약

IP-Adapter는 사전훈련된 text-to-image 확산 모델에 이미지 프롬프트 기능을 부여하는 효과적이고 경량화된 어댑터이다. 핵심 설계는 텍스트 특징과 이미지 특징에 대해 각각 별도의 cross-attention 레이어를 분리하는 **분리형 교차 어텐션(decoupled cross-attention)** 메커니즘이다.

### 주요 기여:
1. **경량 어댑터 설계**: 단순한 방법임에도 불구하고, 약 22M 파라미터만으로 완전히 fine-tuning된 이미지 프롬프트 모델과 동등하거나 더 나은 성능을 달성한다.
2. **일반화 및 호환성**: 사전훈련된 확산 모델을 동결하기 때문에, 동일한 베이스 모델로부터 fine-tuning된 다른 커스텀 모델에도 일반화할 수 있으며, 기존 제어 가능한 도구를 사용한 제어 가능 생성에도 적용할 수 있다.
3. **멀티모달 생성**: 분리형 교차 어텐션 전략의 이점으로, 이미지 프롬프트가 텍스트 프롬프트와 함께 잘 작동하여 멀티모달 이미지 생성을 달성할 수 있다.

---

## 2. 해결하고자 하는 문제

### 2.1 문제 정의

텍스트 프롬프트만으로 원하는 이미지를 생성하는 것은 복잡한 프롬프트 엔지니어링을 수반하기 때문에 매우 까다롭다. 사전훈련된 모델을 직접 fine-tuning하는 기존 방법은 효과적이긴 하지만 대규모 컴퓨팅 자원이 필요하고, 다른 베이스 모델, 텍스트 프롬프트 및 구조적 제어와 호환되지 않는다.

기존 어댑터들은 fine-tuning된 이미지 프롬프트 모델의 성능에 근접하기 어려운데, 그 주요 원인은 이미지 특징이 사전훈련된 모델에 효과적으로 임베딩되지 못하기 때문이다. 대부분의 방법은 단순히 결합된 특징을 동결된 cross-attention 레이어에 전달하여, 확산 모델이 세밀한 특징을 포착하지 못하게 한다.

### 2.2 제안 방법 (수식 포함)

#### (A) 확산 모델 예비 지식

확산 모델의 훈련 손실 함수는 다음과 같다:

$$L_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2\right]$$

여기서 $x_0$는 원본 데이터, $\epsilon$은 가우시안 노이즈, $t$는 타임스텝, $x_t$는 노이즈가 추가된 잠재 벡터, $c$는 텍스트 조건, $\epsilon_\theta$는 노이즈 예측 네트워크이다.

#### (B) 기존 Cross-Attention 메커니즘

표준 text-to-image 확산 모델에서 cross-attention은 다음과 같이 수행된다:

$$Z = \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

여기서:
$$Q = ZW_q, \quad K = c_t W_k, \quad V = c_t W_v$$

$Z$는 UNet의 잠재 특징(latent features), $c_t$는 텍스트 특징이다.

#### (C) IP-Adapter의 분리형 Cross-Attention (핵심 제안)

IP-Adapter는 텍스트 특징과 이미지 특징에 대해 분리형 cross-attention 메커니즘을 채택한다. 확산 모델의 UNet에 있는 모든 cross-attention 레이어마다, 이미지 특징 전용의 추가 cross-attention 레이어를 추가한다.

**텍스트 cross-attention** (기존, 동결):

$$Z_{\text{text}} = \text{Attention}(Q, K_t, V_t) = \text{Softmax}\left(\frac{Q K_t^T}{\sqrt{d}}\right)V_t$$

$$K_t = c_t W_k, \quad V_t = c_t W_v$$

**이미지 cross-attention** (새로 추가, 학습 대상):

$$Z_{\text{image}} = \text{Attention}(Q, K_i, V_i) = \text{Softmax}\left(\frac{Q K_i^T}{\sqrt{d}}\right)V_i$$

$$K_i = c_i W_k', \quad V_i = c_i W_v'$$

여기서 $c_i$는 이미지 특징, $W_k'$와 $W_v'$는 새로 추가된 학습 가능한 프로젝션 가중치이다.

**최종 출력**은 두 attention의 가중합이다:

$$Z_{\text{new}} = Z_{\text{text}} + \lambda \cdot Z_{\text{image}}$$

여기서 $\lambda$는 이미지 프롬프트의 가중치를 조절하는 스케일 파라미터이다.

텍스트 cross-attention은 $\text{Attention}(Q=\text{latent features},\ K=\text{text embeddings},\ V=\text{text embeddings})$로 동작하고, 이미지 cross-attention은 $\text{Attention}(Q=\text{latent features},\ K=\text{image embeddings},\ V=\text{image embeddings})$로 동작한다.

#### (D) 학습 목적 함수

IP-Adapter의 학습 손실은 표준 확산 모델의 노이즈 예측 L2 손실을 그대로 사용한다:

$$L = \mathbb{E}_{x_0, \epsilon, t, c_t, c_i}\left[\|\epsilon - \epsilon_\theta(x_t, t, c_t, c_i)\|^2\right]$$

최적화 과정은 모델의 노이즈 예측과 실제 노이즈 간의 L2 손실을 최소화하며, 그래디언트는 학습 가능한 어댑터 파라미터만 업데이트하고 동결된 베이스 모델은 건드리지 않는다.

### 2.3 모델 구조

IP-Adapter는 두 부분으로 구성된다: (1) 이미지 프롬프트에서 이미지 특징을 추출하는 이미지 인코더, (2) 분리형 cross-attention을 통해 이미지 특징을 사전훈련된 text-to-image 확산 모델에 임베딩하는 적응 모듈이다.

구체적인 구조:

| 구성 요소 | 설명 | 학습 여부 |
|---|---|---|
| **CLIP 이미지 인코더** | OpenCLIP-ViT-H-14 (글로벌 임베딩 또는 패치 임베딩 사용) | ❄️ 동결 |
| **이미지 프로젝션 모듈** | CLIP 이미지 임베딩을 적절한 차원 공간으로 매핑하는 Linear Projection | 🔥 학습 |
| **추가 Cross-Attention 레이어** | 각 UNet cross-attention 블록에 병렬로 추가된 이미지 전용 $W_k'$, $W_v'$ | 🔥 학습 |
| **UNet (Stable Diffusion)** | 기존 text-to-image 확산 모델 | ❄️ 동결 |
| **CLIP 텍스트 인코더** | 기존 텍스트 조건 인코더 | ❄️ 동결 |

학습 과정은 사전훈련된 U-Net, CLIP 텍스트 인코더, CLIP 이미지 인코더를 모두 동결하고 새로 추가된 구성 요소만 학습한다. 이는 이미지 cross-attention 레이어, 이미지 임베딩을 적절한 차원으로 매핑하는 선형 프로젝션 모듈, 관련 LayerNorm 레이어 등 약 22M 파라미터를 포함한다. 이 접근법은 전체 모델 파라미터의 약 3-5%만 학습하게 하여, 계산 요구사항을 크게 줄인다.

**IP-Adapter 변형 모델들:**

다양한 변형이 존재한다: `ip-adapter_sd15` (글로벌 이미지 임베딩), `ip-adapter-plus_sd15` (OpenCLIP-ViT-H-14의 패치 이미지 임베딩 사용, 참조 이미지에 더 가까운 결과), `ip-adapter-plus-face_sd15` (크롭된 얼굴 이미지를 조건으로 사용), `ip-adapter_sdxl` (OpenCLIP-ViT-bigG-14의 글로벌 이미지 임베딩 사용) 등이 있다.

### 2.4 성능 향상

이 방법은 이미지 품질 면에서 다른 방법들을 능가할 뿐만 아니라, 참조 이미지와 더 잘 정렬된 이미지를 생성한다.

기존 방법들과 비교하여, 이미지 품질과 멀티모달 프롬프트 정렬 모두에서 우수한 결과를 생성할 수 있다.

학습 단계에서는 새 cross-attention 레이어의 파라미터만 학습하고 원래 UNet 모델은 동결된다. 약 22M 파라미터만으로도 text-to-image 확산 모델을 완전히 fine-tuning한 이미지 프롬프트 모델에 견줄 만한 생성 성능을 보인다.

### 2.5 한계점

논문 및 관련 분석에서 확인되는 한계점들:

1. **CLIP 인코더 의존성**: CLIP 이미지 인코더는 상대적으로 약하게 정렬된 데이터로 학습되었다. 인코딩된 특징이 세부적인 얼굴 특징을 정확하게 파악하지 못하고, 대신 구성, 스타일, 색상과 같은 넓고 모호한 의미적 정보를 포착한다.

2. **비정사각형 이미지 처리**: CLIP의 기본 이미지 프로세서가 중앙 크롭을 수행하므로 정사각형 이미지에서 가장 잘 작동한다. 비정사각형 이미지에서는 중앙 외부의 정보가 손실될 수 있다.

3. **세밀한 신원 보존(Identity Preservation)의 한계**: 글로벌 이미지 임베딩만으로는 특정 인물의 정확한 신원(identity)을 보존하기 어렵다.

4. **스타일-콘텐츠 분리의 어려움**: 주제와 스타일 합성에서 ControlNet에 의존하거나 콘텐츠와 스타일 누출(leakage) 아티팩트가 발생할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

IP-Adapter의 일반화 성능은 이 논문의 가장 강력한 기여 중 하나이다.

### 3.1 커스텀 모델로의 전이

IP-Adapter는 재사용이 가능하고 유연하다. 베이스 확산 모델에서 학습된 IP-Adapter는 동일한 베이스 확산 모델에서 fine-tuning된 다른 커스텀 모델에도 일반화될 수 있다. 더욱이, IP-Adapter는 ControlNet 등 기존 제어 가능 어댑터와 호환되어 이미지 프롬프트와 구조적 제어를 쉽게 결합할 수 있다.

이 일반화가 가능한 핵심 원리:
- **모델 동결 전략**: 원래 UNet을 동결함으로써 베이스 모델의 가중치 공간을 그대로 유지
- **모듈식 설계**: 추가된 cross-attention 레이어가 플러그인 방식으로 동작
- **공유된 잠재 공간**: 동일 베이스 모델에서 파생된 커스텀 모델들은 유사한 잠재 공간을 공유

### 3.2 다양한 제어 도구와의 호환성

IP-Adapter는 ControlNet, T2I-Adapter 등 기존 제어 가능한 도구들과 완전히 호환된다.

이는 다음과 같은 구조적 조합이 가능함을 의미한다:

$$\text{Output} = \mathcal{F}_{\text{UNet}}(x_t,\ c_{\text{text}},\ c_{\text{image}},\ c_{\text{structure}})$$

여기서 $c_{\text{structure}}$는 ControlNet이나 T2I-Adapter로부터의 구조적 조건이다.

### 3.3 멀티모달 프롬프트 지원

분리형 cross-attention 전략 덕분에, 이미지 프롬프트가 텍스트 프롬프트와 호환되어 멀티모달 이미지 생성을 달성할 수 있다.

스케일 파라미터 $\lambda$를 조절함으로써 텍스트와 이미지 간의 영향력 균형을 조정할 수 있다:

$$Z_{\text{new}} = Z_{\text{text}} + \lambda \cdot Z_{\text{image}}, \quad \lambda \in [0, 1]$$

스케일을 낮추면 더 다양한 이미지를 생성할 수 있지만, 이미지 프롬프트와의 일관성이 떨어질 수 있다. 멀티모달 프롬프트의 경우, 대부분 $\text{scale}=0.5$로 설정하면 좋은 결과를 얻을 수 있다.

### 3.4 일반화 성능 향상을 위한 향후 방향

| 방향 | 설명 |
|---|---|
| 더 강력한 이미지 인코더 | CLIP 외 DINOv2, SigLIP 등 다양한 비전 인코더 활용 |
| 패치 수준 특징 활용 | 글로벌 임베딩 대신 로컬 패치 특징을 활용한 세밀한 제어 |
| 도메인 적응 | 특정 도메인(의료, 위성 등)에 대한 적응적 학습 전략 |
| 다중 참조 이미지 | 여러 참조 이미지의 특징을 집계하여 더 강건한 조건 생성 |

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

IP-Adapter는 이미지 조건부 생성 분야에서 **패러다임적 전환**을 이끈 논문이다:

1. **어댑터 패러다임의 확립**: "모델 전체를 fine-tuning하지 않고, 경량 어댑터로 새로운 모달리티를 통합"하는 설계 패턴이 후속 연구의 표준이 됨
2. **분리형 어텐션의 일반화**: 텍스트와 이미지를 분리하는 cross-attention 설계가 InstantID, PhotoMaker, IP-Adapter-FaceID 등 수많은 후속 연구에 직접 채용됨
3. **모듈식 AI 생태계**: ControlNet, LoRA, IP-Adapter가 결합 가능한 모듈식 생태계 형성에 기여

### 4.2 향후 연구 시 고려할 점

1. **세밀한 신원 보존**: CLIP 글로벌 임베딩의 한계로 인해, 얼굴 인식 모델(InsightFace 등)과의 결합이 필요
2. **의미적 분리**: 스타일, 콘텐츠, 구조 정보의 명시적 분리 메커니즘 연구
3. **동영상 확장**: 시간적 일관성을 유지하면서 이미지 프롬프트를 비디오 생성에 적용
4. **효율성-성능 트레이드오프**: 더 적은 파라미터로 더 높은 충실도를 달성하는 연구
5. **안전성 및 보안**: 최근 연구에서 IP-Adapter가 장착된 T2I 확산 모델이 하이재킹 공격(hijacking attack)이라는 새로운 탈옥 공격을 가능하게 함이 밝혀졌다. 감지 불가능한 이미지 공간 적대적 예제를 업로드함으로써 공격자가 양의적 사용자를 하이재킹하여 이미지 생성 서비스를 탈옥시킬 수 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 차별점 | IP-Adapter 대비 장단점 |
|---|---|---|---|
| **ControlNet** (Zhang et al.) | 2023 | 구조적 조건(엣지, 깊이 등) 기반 제어 | 구조 제어에 강하나 이미지 스타일/콘텐츠 전달 불가 |
| **T2I-Adapter** (Mou et al.) | 2023 | 경량 어댑터로 구조적 제어 | IP-Adapter와 상호 보완적, 이미지 프롬프트 미지원 |
| **BLIP-Diffusion** (Li et al.) | 2023 | 사전훈련된 멀티모달 표현으로 zero-shot 주제 생성 | 주제 주도 생성에 강하나 범용성 부족 |
| **ELITE** (Wei et al.) | 2023 | 글로벌+로컬 CLIP 특징 활용 | 세밀한 매핑 가능하나 학습 비용 높음 |
| **InstantID** (Wang et al.) | 2024 | ID 임베딩(강한 의미적 얼굴 정보 포착), 분리형 cross-attention의 경량 적응 모듈, IdentityNet(참조 얼굴의 상세 특징을 추가 공간 제어와 함께 인코딩) 세 가지 핵심 구성 요소를 포함한다. IP-Adapter보다 더 높은 얼굴 충실도를 달성하는데, 이는 CLIP 임베딩 대신 사전훈련된 얼굴 모델의 강건한 ID 임베딩이 더 풍부한 의미 정보(신원, 나이, 성별)를 포착하기 때문이다. |
| **PhotoMaker** (Li et al.) | 2023 | UNet LoRA 파라미터 학습으로 ID 보존 | 좋은 충실도를 달성하지만 텍스트 제어 능력의 명확한 저하가 관찰됨 |
| **IP-Adapter-FaceID** (Ye et al.) | 2023 | InsightFace 임베딩 + IP-Adapter 아키텍처 | 얼굴 ID 보존 강화, 그러나 스타일 통합 제한 |
| **DreamBooth** (Ruiz et al.) | 2023 | 주제별 fine-tuning으로 개인화 | 높은 품질이나 모델별 fine-tuning 필요 |
| **SubZero** (2024) | 2024 | fine-tuning 없이 모든 주제를 모든 스타일로, 모든 행동을 수행하게 생성하는 프레임워크. 주제 및 스타일 유사성을 향상시키면서 누출을 줄이기 위한 새로운 제약 세트를 제안한다. 디노이징 모델의 cross-attention 블록에서 직교화된 시간적 집계 방식을 제안한다. |
| **ICAS** (2025) | 2025 | IP-Adapter와 ControlNet의 장점을 결합하여 어텐션 메커니즘을 개선하고, 대량의 fine-tuning이나 역변환 없이 다중 주제 콘텐츠와 전체 스타일의 정밀 제어를 달성한다. |
| **Unicron Adapter** | 2024 | cross-attention으로 유의미한 이미지 특징을 추출하고 self-attention으로 이미지와 텍스트를 융합하여 조건 공간을 통합하는 접근법 |

### 핵심 트렌드 분석

IP-Adapter 이후의 연구 흐름은 크게 세 방향으로 전개되고 있다:

1. **더 강력한 신원 인코딩**: CLIP → InsightFace/ArcFace 등 전문 인코더로의 전환
2. **분리의 심화**: 스타일, 콘텐츠, 구조를 더 세밀하게 분리하는 멀티-어댑터 구조
3. **제로샷 일반화**: 추가 학습 없이 다양한 태스크에 적용 가능한 범용 어댑터 설계

---

## 참고 자료 및 출처

1. **Ye, H., Zhang, J., Liu, S., Han, X., & Yang, W.** (2023). *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models.* arXiv:2308.06721. — [https://arxiv.org/abs/2308.06721](https://arxiv.org/abs/2308.06721)
2. **IP-Adapter 공식 프로젝트 페이지** — [https://ip-adapter.github.io/](https://ip-adapter.github.io/)
3. **IP-Adapter GitHub 리포지토리 (Tencent AI Lab)** — [https://github.com/tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
4. **Hugging Face IP-Adapter 모델 페이지** — [https://huggingface.co/h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)
5. **Hugging Face Diffusers IP-Adapter 문서** — [https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter)
6. **ar5iv HTML 전문** — [https://ar5iv.labs.arxiv.org/html/2308.06721](https://ar5iv.labs.arxiv.org/html/2308.06721)
7. **Mercity Research: Understanding and Training IP Adapters** — [https://www.mercity.ai/blog-post/understanding-and-training-ip-adapters-for-diffusion-models/](https://www.mercity.ai/blog-post/understanding-and-training-ip-adapters-for-diffusion-models/)
8. **Wang, Q. et al.** (2024). *InstantID: Zero-shot Identity-Preserving Generation in Seconds.* arXiv:2401.07519. — [https://instantid.github.io/](https://instantid.github.io/)
9. **ICAS (2025)** — [https://arxiv.org/html/2504.13224v1](https://arxiv.org/html/2504.13224v1)
10. **Semantic Scholar IP-Adapter 페이지** — [https://www.semanticscholar.org/paper/2854e5bab8e6f36e54c64456628a9559bf67019e](https://www.semanticscholar.org/paper/2854e5bab8e6f36e54c64456628a9559bf67019e)
