
# EASI-Tex: Edge-Aware Mesh Texturing from Single Image

> **논문 정보**
> - **제목**: EASI-Tex: Edge-Aware Mesh Texturing from Single Image
> - **저자**: Sai Raj Kishore Perla, Yizhi Wang, Ali Mahdavi-Amiri, Hao Zhang
> - **학회/저널**: ACM Transactions on Graphics (Proceedings of **SIGGRAPH 2024**)
> - **Volume/Article**: Vol. 43, No. 4, Article 40
> - **DOI**: [10.1145/3658222](https://dl.acm.org/doi/10.1145/3658222)
> - **arXiv**: [2405.17393](https://arxiv.org/abs/2405.17393) (May 2024)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

EASI-Tex는 단일 RGB 이미지로부터 3D 메시 객체로 텍스처를 seamlessly 전달하기 위해 diffusion 모델에 정교한 조건화(judicious conditioning)를 적용하는 새로운 단일 이미지 메시 텍스처링 방법을 제시한다.

이 방법은 두 객체가 동일 카테고리에 속한다는 가정을 하지 않으며, 같은 카테고리라 하더라도 형상(geometry)과 파트 비율에 상당한 불일치가 존재할 수 있다는 현실적인 문제를 직접 다룬다.

### 주요 기여 3가지

| 기여 | 내용 |
|---|---|
| **① Edge-Aware Conditioning** | 메시 구조를 엣지(edge)로 표현, ControlNet을 통해 조건화 |
| **② IP-Adapter 기반 단일 이미지 활용** | 별도 학습/최적화 없이 단일 이미지를 프롬프트로 활용 |
| **③ Image Inversion** | IP-Adapter의 세부 표현 한계를 보완하는 선택적 fine-tuning 기법 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존의 3D 메시 텍스처링 방법들은 주로 **텍스트 프롬프트** 기반으로 텍스처를 생성하거나, 소스 메시와 타깃 메시가 **동일 카테고리**에 속한다는 강한 가정 하에 동작했습니다. 이로 인해:

1. **이종 카테고리 간 텍스처 전달 불가**: 예를 들어 새의 깃털 패턴을 곰 인형에 적용하는 것이 불가능했습니다.
2. **형상 불일치(Geometry Discrepancy)**: 같은 카테고리라도 파트 비율이나 세부 구조가 달라 텍스처가 자연스럽게 정렬되지 않는 문제가 있었습니다.
3. **추가 학습·최적화 비용**: 기존 방법들은 각 쌍(pair)에 대해 별도의 최적화가 필요했습니다.

엣지 조건화(edge conditioning)는 깊이(depth)나 법선(normals)보다 메시의 "정체성(identity)"을 더 잘 존중하도록 하며, IP-Adapter는 추가 학습이나 최적화 없이 단일 이미지를 프롬프트로 사용할 수 있게 한다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### 전체 파이프라인

텍스처 생성 네트워크는 텍스처가 없는 3D 메시 $\mathbf{M}$, 레퍼런스 텍스처 이미지 $\mathbf{I_{tex}}$, 그리고 서술적 텍스트 프롬프트를 입력으로 받아 텍스처된 뷰(textured view)를 생성한다.

텍스처 생성 과정은 메시를 묘사하는 엣지(edges)를 ControlNet을 통해 조건으로 활용하고, IP-Adapter를 통해 입력 텍스처 이미지를 조건으로 활용한다.

전체 생성 과정은 다음과 같이 수식으로 표현할 수 있습니다:

$$
\hat{x} = \mathcal{G}_{\theta}(\mathbf{z}_t,\, t,\, \mathbf{c}_{text},\, \mathbf{c}_{edge},\, \mathbf{c}_{image})
$$

- $\hat{x}$: 생성된 텍스처 뷰 이미지
- $\mathcal{G}_{\theta}$: 사전학습된 Stable Diffusion (U-Net 기반)
- $\mathbf{z}_t$: 타임스텝 $t$에서의 noisy latent
- $\mathbf{c}_{text}$: 텍스트 프롬프트 조건 (CLIP 텍스트 인코더)
- $\mathbf{c}_{edge}$: 메시 엣지 맵 조건 (ControlNet)
- $\mathbf{c}_{image}$: 입력 텍스처 이미지 피처 (IP-Adapter)

#### Diffusion의 역방향 과정 (Score Matching)

$$
p_\theta(\mathbf{z}_{t-1} \mid \mathbf{z}_t,\, \mathbf{c}_{edge},\, \mathbf{c}_{image},\, \mathbf{c}_{text}) = \mathcal{N}(\mathbf{z}_{t-1};\, \mu_\theta,\, \sigma^2_t \mathbf{I})
$$

여기서 평균 $\mu_\theta$는 노이즈 예측 네트워크 $\epsilon_\theta$로부터 다음과 같이 계산됩니다:

$$
\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{z}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{z}_t, t, \mathbf{c}_{edge}, \mathbf{c}_{image}, \mathbf{c}_{text})\right)
$$

#### ControlNet을 통한 엣지 조건화

ControlNet은 Stable Diffusion의 인코더 블록을 복사한 구조로, 엣지 맵 $\mathbf{e}$ (Canny 등)를 조건으로 받아 U-Net의 중간 피처에 residual connection으로 추가됩니다:

$$
\mathbf{h}^{l}_{out} = \mathbf{h}^{l}_{SD} + \mathcal{F}^{l}_{CN}(\mathbf{h}^{l}_{CN},\, \mathbf{e})
$$

- $\mathbf{h}^{l}_{SD}$: Stable Diffusion $l$번째 레이어의 hidden state
- $\mathcal{F}^{l}_{CN}$: ControlNet의 $l$번째 레이어 출력
- $\mathbf{e}$: 렌더링된 메시의 엣지 맵

#### IP-Adapter를 통한 이미지 조건화

IP-Adapter는 이미지 피처를 decoupled cross-attention 메커니즘으로 U-Net에 주입합니다:

$$
\text{Attn}_{new} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V} + \lambda \cdot \text{softmax}\left(\frac{\mathbf{Q}(\mathbf{K}')^T}{\sqrt{d}}\right)\mathbf{V}'
$$

- $\mathbf{Q}$: U-Net의 query
- $\mathbf{K}, \mathbf{V}$: 텍스트 조건으로부터의 key, value
- $\mathbf{K}', \mathbf{V}'$: 이미지 피처로부터의 key, value (IP-Adapter projection network 출력)
- $\lambda$: 이미지 조건의 가중치

---

### 2-3. 모델 구조

EASI-Tex의 파이프라인은 Image Inversion(선택)과 textured view 생성(필수)의 두 단계로 구성된다. Image Inversion은 선택적 단계로, 네트워크의 일부인 Stable Diffusion의 U-Net과 IP-Adapter의 Projection Network를 fine-tuning하는 것을 포함한다.

전체 구조를 도식화하면:

```
[입력]
  ├─ 3D 메시 M (텍스처 없음)
  ├─ 레퍼런스 이미지 I_tex
  └─ 텍스트 프롬프트 t

     ↓ (멀티뷰 렌더링 & 엣지 추출)

[텍스처 생성 네트워크]
  ├─ ControlNet  ← 메시 엣지 맵 e
  ├─ IP-Adapter  ← CLIP 이미지 인코더 → I_tex 피처
  └─ Stable Diffusion U-Net (frozen backbone)

     ↓ (뷰별 텍스처 생성 & back-projection)

[선택: Image Inversion]
  └─ SD U-Net + IP-Adapter Projection Network 미세조정
     (단일 이미지 I_tex, 소수 iteration)

[출력]
  └─ 텍스처드 3D 메시
```

Image Inversion은 사전학습된 IP-Adapter가 입력 이미지의 모든 세부 사항을 충실하게 포착하지 못하는 경우에 사용하며, 이 방법은 단일 이미지를 통해 diffusion 모델을 특정 개념에 맞게 빠르게 개인화(personalize)한다.

---

### 2-4. 성능 향상

실험 결과는 EASI-Tex의 엣지 인식 단일 이미지 메시 텍스처링 방법이 다양한 3D 객체에서 입력 텍스처의 세부 사항을 보존하면서도 메시의 형상을 존중하는 데 있어 효율성과 효과성을 입증한다.

주요 성능 특징:

| 항목 | 내용 |
|---|---|
| **학습/최적화 불필요** | Inference 시 gradient 최적화 없이 동작 |
| **크로스 카테고리 전달** | 카테고리 경계를 넘나드는 텍스처 전달 가능 |
| **엣지 우월성** | depth, normals 대비 메시 구조 보존 우수 |
| **고해상도 지원** | 1k ~ 3k 텍스처 해상도 지원 |

---

### 2-5. 한계

공개된 정보를 토대로 파악된 한계는 다음과 같습니다:

1. **IP-Adapter의 세부 표현 한계**: 사전학습된 IP-Adapter가 입력 이미지의 모든 세부 사항을 충실하게 포착하지 못하는 경우가 있어, 이를 위해 Image Inversion이라는 별도 기법이 필요하다.

2. **GPU 메모리 요구**: 3k 고해상도 텍스처를 생성할 경우 1k 텍스처 대비 훨씬 더 많은 GPU 메모리가 필요하다.

3. **뷰 일관성 문제**: 멀티뷰 back-projection 기반 접근법의 일반적 한계로, 뷰 간 텍스처 일관성이 완전하지 않을 수 있습니다 (이는 이 분야 전반의 미해결 과제이기도 합니다).

4. **텍스트 프롬프트 의존성**: 텍스트 프롬프트를 여전히 보조 입력으로 사용하여 세밀한 제어가 필요합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화를 가능하게 하는 설계 원칙

EASI-Tex의 일반화 성능이 높은 이유는 세 가지 핵심 설계 선택에 있습니다:

#### (1) 카테고리-비의존적 설계 (Category-Agnostic)

EASI-Tex는 소스 이미지와 타깃 메시가 동일 카테고리에 속한다는 가정을 하지 않으며, 같은 카테고리라도 형상과 파트 비율에 상당한 불일치가 있을 수 있음을 인정한다.

이 설계 덕분에 모델은 동물 ↔ 캐릭터, 자동차 ↔ 트럭 등 이종 카테고리 간 전달이 가능하며, 이는 실제 창작 작업에서의 적용 범위를 크게 확장합니다.

#### (2) 사전학습 모델의 강력한 일반화 지식 활용

EASI-Tex는 사전학습된 Stable Diffusion 생성기를 ControlNet을 통한 엣지 조건화와 IP-Adapter를 통한 이미지 피처 추출로 조건화하여, 별도의 최적화나 학습 없이 메시의 내재적 형상과 입력 텍스처를 모두 존중하는 텍스처를 생성한다.

Stable Diffusion의 대규모 사전학습 지식이 다양한 형상·스타일에 대한 일반화를 기본적으로 보장합니다.

#### (3) 엣지 기반 구조 표현의 우월성

엣지 조건화는 depth나 normals보다 메시의 "정체성"을 더 잘 존중하도록 하며, IP-Adapter는 추가 학습이나 최적화 없이 단일 이미지를 프롬프트로 사용할 수 있게 한다.

엣지는 형상의 위상적(topological) 구조를 인코딩하는 데 탁월하여, 형상이 매우 다른 메시에 대해서도 구조적 일관성을 유지합니다.

### 3-2. Image Inversion의 일반화 기여

Image Inversion은 단일 이미지를 이용하여 diffusion 모델을 특정 개념에 빠르게 개인화하는 기법이며, 이는 사전학습된 IP-Adapter가 입력 텍스처 이미지의 세부 사항을 충실하게 포착하지 못하는 경우에 사용하는 선택적 단계이다.

Image Inversion의 수식은 다음과 같이 표현할 수 있습니다:

$$
\mathcal{L}_{inv} = \mathbb{E}_{\mathbf{z}_t, t, \mathbf{c}} \left[\left\|\epsilon - \epsilon_\theta(\mathbf{z}_t, t, \mathbf{c}_{edge}, \phi(\mathbf{I}_{tex}), \mathbf{c}_{text})\right\|^2_2\right]
$$

여기서 $\phi$는 fine-tuning 대상인 IP-Adapter Projection Network이며, 이 손실을 최소화하는 방향으로 소수의 iteration만 수행합니다. 이 메커니즘은:

- **단 1장의 이미지**로 빠른 개인화 가능
- **추가 데이터셋 불필요** → 임의의 새로운 텍스처 스타일에 즉시 적응 가능
- 결국 사전학습 모델의 일반화 능력을 유지하면서 특정 디테일에 집중하는 효과

### 3-3. 일반화 성능 향상 방향

공개 자료 기반으로 향후 개선 가능한 방향을 정리하면:

| 방향 | 설명 |
|---|---|
| **멀티모달 조건 확장** | 텍스트+이미지+스케치 등 다양한 조건을 통합 |
| **동적 IP-Adapter 가중치 조절** | 각 뷰마다 $\lambda$ 값을 동적으로 최적화하여 일관성 강화 |
| **더 강력한 사전학습 백본** | SD 3.0, SDXL 등 최신 백본으로 교체 시 일반화 성능 직접 향상 |
| **3D 인식 피처 통합** | 3D-aware feature를 IP-Adapter에 통합하여 뷰 일관성 개선 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. 주요 관련 연구 비교 표

| 논문 | 연도/학회 | 입력 | 조건 유형 | 최적화 필요 | 카테고리 제한 |
|---|---|---|---|---|---|
| **Text2Tex** | ICCV 2023 | 텍스트 | Depth | ❌ | 없음 |
| **TEXTure** | SIGGRAPH 2023 | 텍스트 | Depth+Normal | ❌ | 없음 |
| **Mesh2Tex** | ICCV 2023 | 이미지 | 카테고리 피처 | ✅ | **있음** |
| **TexFusion** | ICCV 2023 | 텍스트 | Depth | ❌ | 없음 |
| **EASI-Tex** | SIGGRAPH 2024 | **단일 이미지** | **Edge+IP** | ❌ | **없음** |
| **Meta 3D TextureGen** | 2024 | 텍스트 | Multi-view | ❌ | 없음 |

### 4-2. Text2Tex와의 비교

Text2Tex는 사전학습된 depth-aware 텍스트-이미지 diffusion 모델을 사용하여 3D 객체를 seamlessly 텍스처링하는 방법으로, 목표 메시를 여러 시점에서 렌더링하고 depth-aware 모델로 누락된 외관을 inpainting한다.

Text2Tex는 generate-then-refine 전략을 따르며, 여러 시점에 걸쳐 부분 텍스처를 점진적으로 생성하고 텍스처 공간으로 역투영(backproject)한다.

**핵심 차이**: Text2Tex는 텍스트 프롬프트를 주요 입력으로 사용하는 반면, EASI-Tex는 **단일 RGB 이미지**를 주요 조건으로 활용하여 더 구체적인 텍스처 스타일을 직접 전달할 수 있습니다.

### 4-3. TEXTure와의 비교

TEXTure는 하나의 시점에서 부분 텍스처 맵을 생성한 후 inpainting을 통해 다른 뷰를 완성하는 방식이지만, 뷰 간 전역 정보 부재로 인한 불일치 문제가 발생한다.

### 4-4. 최신 흐름: TriTex (2025)

TriTex는 효율적인 triplane 기반 아키텍처를 사용하여 새로운 타깃 메시로의 semantic-aware 텍스처 전달을 가능하게 하며, 단 하나의 예시만으로 학습하면서도 동일 카테고리 내 다양한 형상에 효과적으로 일반화된다.

### 4-5. Score Distillation Sampling 계열과의 비교

SDS(Score Distillation Sampling) 기반 방법들은 2D diffusion prior를 증류하여 3D 형상에 텍스처를 합성하지만, 높은 계산 비용과 Janus 문제, 부자연스러운 색상 등 본질적인 아티팩트가 존재한다.

EASI-Tex는 이러한 SDS 계열의 문제를 우회하며, 별도의 최적화 루프 없이 단일 forward pass 기반으로 동작합니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5-1. 앞으로의 연구에 미치는 영향

#### ① 단일 이미지 기반 텍스처 전달의 패러다임 전환
기존 텍스트 기반 접근에서 **시각적 레퍼런스(이미지) 기반 접근**으로의 전환을 제시합니다. 이는 아티스트가 원하는 구체적인 스타일이나 질감을 텍스트로 설명하는 어려움을 해소합니다.

#### ② 크로스 카테고리 일반화의 가능성 제시
EASI-Tex는 두 객체가 동일 카테고리에 속한다는 가정 없이도, 사전학습된 Stable Diffusion 생성기를 ControlNet과 IP-Adapter로 조건화함으로써 다양한 3D 객체에 걸쳐 텍스처 세부 사항을 보존할 수 있음을 실험적으로 입증한다.

이는 이후 연구들이 더 자유로운 형태의 텍스처 전달을 탐구할 수 있는 토대를 마련합니다.

#### ③ 경량 개인화 메커니즘의 중요성
Image Inversion은 DreamBooth, Textual Inversion 등 기존 개인화 방법 대비 **단 1장의 이미지 + 소수 iteration**으로 동작하는 경량 방식을 제시합니다. 이는 실시간/인터랙티브 응용으로의 확장 가능성을 시사합니다.

#### ④ 프리트레인 모델 활용의 효율성 증명
EASI-Tex의 코드는 Text2Tex와 Diffusers_IPAdapter를 기반으로 구축되어 있으며, 이는 강력한 사전학습 생성 모델의 구성 요소들을 모듈식으로 조합하는 연구 방향이 효과적임을 보여줍니다.

---

### 5-2. 앞으로 연구 시 고려할 점

#### ① 뷰 일관성 (View Consistency) 강화

멀티뷰 back-projection 과정에서 발생하는 이음새(seam)와 불일치 문제는 여전히 중요한 과제입니다. 향후 연구는:

$$
\mathcal{L}_{consistency} = \sum_{i \neq j} \left\| \mathcal{R}_i(\mathbf{T}) - \mathcal{R}_j(\mathbf{T}) \right\|_2
$$

와 같은 멀티뷰 일관성 손실을 도입하거나, **3D Gaussian Splatting** 등 명시적 3D 표현과 결합하는 방향을 고려할 수 있습니다.

#### ② 더 강력한 이미지 인코더 활용

현재 CLIP 기반 IP-Adapter의 세부 표현 한계는 Image Inversion으로 보완되나, **DINOv2**, **SAM** 등 더 강력한 비전 인코더를 활용하면 Image Inversion 없이도 세부 표현이 가능해질 수 있습니다.

#### ③ 동영상/애니메이션 메시로의 확장

정적 메시에서 **스키닝(skinning)된 동적 메시**로의 텍스처 전달은 변형에 따른 텍스처 왜곡 문제가 추가로 발생합니다. 이를 위해:

$$
\mathbf{T}_{animated} = \mathcal{W}(\mathbf{T}_{static},\, \mathbf{B},\, \mathbf{\theta})
$$

여기서 $\mathbf{B}$는 blend shape, $\mathbf{\theta}$는 포즈 파라미터로, 텍스처의 동적 일관성 유지가 중요한 연구 과제가 됩니다.

#### ④ 물리 기반 렌더링(PBR) 재질 추정과의 결합

현재 방법은 albedo 맵 중심이지만, roughness, metallic, normal map 등 PBR 재질 파라미터까지 추정하는 방향으로 확장할 경우 **포토리얼리스틱(photo-realistic)** 결과를 기대할 수 있습니다.

#### ⑤ 실시간 응용을 위한 경량화

현재 diffusion 기반 방법은 추론 속도가 느린 편입니다. **Consistency Model**, **LCM(Latent Consistency Model)** 등을 결합하여 실시간 게임/VR 파이프라인에 적용 가능한 형태로 경량화하는 연구가 필요합니다.

#### ⑥ 평가 지표의 표준화

단일 이미지 텍스처 전달에서 "좋은 결과"를 정량화하는 것은 LPIPS, FID, CLIP Score 등 기존 지표로는 불충분합니다. **텍스처 충실도(texture fidelity)**, **형상 존중도(geometry adherence)**, **스타일 일관성(style consistency)**을 동시에 평가하는 종합적 벤치마크 구축이 필요합니다.

---

## 참고 자료 및 출처

| 번호 | 자료 |
|---|---|
| 1 | **arXiv**: [2405.17393] EASI-Tex: Edge-Aware Mesh Texturing from Single Image — https://arxiv.org/abs/2405.17393 |
| 2 | **공식 프로젝트 페이지**: https://sairajk.github.io/easi-tex/ |
| 3 | **ACM DL (SIGGRAPH 2024)**: https://dl.acm.org/doi/10.1145/3658222 |
| 4 | **GitHub 레포지토리**: https://github.com/sairajk/easi-tex |
| 5 | **ResearchGate**: https://www.researchgate.net/publication/380929380 |
| 6 | **SIGGRAPH History**: https://history.siggraph.org/learning/easi-tex-edge-aware-mesh-texturing-from-single-image/ |
| 7 | **ADS Abstract**: https://ui.adsabs.harvard.edu/abs/2024arXiv240517393P/abstract |
| 8 | **Text2Tex (ICCV 2023)**: Chen et al., arXiv:2303.11396 — https://arxiv.org/abs/2303.11396 |
| 9 | **TriTex (2025)**: arXiv:2503.16630 — https://arxiv.org/pdf/2503.16630 |
| 10 | **TEXGen**: arXiv:2411.14740 — https://arxiv.org/html/2411.14740v1 |
| 11 | **Meta 3D TextureGen**: arXiv:2407.02430 — https://arxiv.org/html/2407.02430v1 |
| 12 | **IP-Adapter**: Ye et al., arXiv:2308.06721 (2023) |
| 13 | **ControlNet**: Zhang et al., ICCV 2023 (Adding conditional control to text-to-image diffusion models) |

> ⚠️ **정확도 주의 사항**: 논문 내 구체적 수식(특히 Image Inversion의 정확한 손실 함수 형태), 정량적 실험 수치(CLIP Score, FID 등), 비교 실험의 상세 수치는 전체 PDF 원문 접근이 제한된 관계로 공개된 abstract 및 프로젝트 페이지 기반으로 추론하여 작성하였습니다. 정확한 수치 확인을 위해서는 **ACM DL 원문**(DOI: 10.1145/3658222) 직접 열람을 권장합니다.
