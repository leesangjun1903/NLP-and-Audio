# Imagic: Text-Based Real Image Editing with Diffusion Models

> **논문 정보**
> - **제목**: Imagic: Text-Based Real Image Editing with Diffusion Models
> - **저자**: Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, Michal Irani
> - **학회**: CVPR 2023
> - **arXiv**: [2210.09276](https://arxiv.org/abs/2210.09276)
> - **프로젝트 페이지**: [imagic-editing.github.io](https://imagic-editing.github.io/)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

이 논문은 기존 방법들이 특정 편집 유형(예: 오브젝트 오버레이, 스타일 전이)에 제한되거나 합성 이미지에만 적용되거나 동일 오브젝트의 여러 입력 이미지를 필요로 했던 한계를 극복하고, **최초로 단일 실제 이미지에 대해 복잡한(비강체적, non-rigid) 텍스트 기반 시맨틱 편집**을 수행하는 능력을 시연한다.

예를 들어, 이미지 내 하나 또는 여러 오브젝트의 포즈와 구성을 원본 이미지의 특성을 보존하면서 변경할 수 있으며, 서 있는 개를 앉히거나 점프시키고, 새가 날개를 펼치게 할 수 있다.

### 📌 주요 기여 (3가지)

Imagic의 주요 기여는 다음과 같다: **(1)** 단일 실제 고해상도 입력 이미지에 대해 전체적인 구조와 구성을 보존하면서 복잡한 비강체적 편집을 가능하게 하는 최초의 텍스트 기반 시맨틱 이미지 편집 기법을 제안하며, 스타일 변경, 색상 변경, 오브젝트 추가 등 다양한 편집도 수행한다.

**(2)** 두 텍스트 임베딩 시퀀스 사이의 의미론적으로 의미 있는 선형 보간을 시연함으로써, 텍스트-이미지 확산 모델의 강력한 구성적 능력을 드러낸다.

**(3)** 복잡한 비강체적 편집을 설명하는 100쌍의 입력 이미지와 대상 텍스트로 구성된 새로운 벤치마크 **TEdBench (Textual Editing Benchmark)**를 도입하며, 향후 연구를 위해 이를 공개하고 Imagic의 결과도 공개한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 🔴 2-1. 해결하고자 하는 문제

기존 연구들은 단일 입력 이미지와 대상 텍스트(원하는 편집)만을 필요로 하지 않았다. 대부분의 방법이 특정 편집 유형(오브젝트 오버레이, 스타일 전이 등)에 제한되거나, 합성 이미지에만 적용되거나, 동일한 오브젝트의 여러 입력 이미지가 필요했다.

Imagic은 단일 입력 이미지와 대상 텍스트만을 필요로 하며, 실제 이미지에서 동작하고 이미지 마스크나 오브젝트의 추가적인 뷰와 같은 추가 입력을 필요로 하지 않는다.

---

### 🟢 2-2. 제안하는 방법 (수식 포함)

Imagic의 방법은 **3가지 주요 단계**로 구성된다: 텍스트 임베딩 최적화(Text Embedding Optimization), 모델 파인튜닝(Model Fine-Tuning), 보간(Interpolation).

---

#### **Stage 1: 텍스트 임베딩 최적화 (Text Embedding Optimization)**

주어진 실제 이미지와 대상 텍스트 프롬프트에 대해, 대상 텍스트를 인코딩하여 초기 텍스트 임베딩 $e_{tgt}$를 얻고, 이를 입력 이미지를 재구성하도록 최적화하여 $e_{opt}$를 획득한다.

이 단계에서의 최적화 목표는 확산 모델의 일반적인 Denoising Objective를 활용한다. Diffusion Model의 표준 학습 손실은:

$$\mathcal{L}_{DM} = \mathbb{E}_{x_0, \epsilon \sim \mathcal{N}(0,I), t}\left[\|\epsilon - \epsilon_\theta(x_t, t, e)\|^2\right]$$

여기서:
- $x_0$: 입력 이미지
- $x_t$: 시간 $t$에서 노이즈가 추가된 이미지
- $\epsilon$: 추가된 가우시안 노이즈
- $\epsilon_\theta$: 학습된 노이즈 예측 네트워크
- $e$: 텍스트 임베딩 (Stage 1에서 최적화 대상)
- $t$: 확산 타임스텝

Stage 1에서는 **모델 파라미터 $\theta$는 고정**하고, 텍스트 임베딩 $e$를 최적화하여 $e_{tgt} \rightarrow e_{opt}$를 얻는다:

$$e_{opt} = \arg\min_{e} \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t, e)\|^2\right]$$

---

#### **Stage 2: 모델 파인튜닝 (Model Fine-Tuning)**

그 다음, $e_{opt}$를 고정한 채 생성 모델을 파인튜닝하여 입력 이미지에 대한 충실도를 향상시킨다.

이 단계에서는 **임베딩 $e_{opt}$는 고정**하고, **모델 파라미터 $\theta$를 최적화**한다:

$$\theta^* = \arg\min_\theta \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t, e_{opt})\|^2\right]$$

파인튜닝 없이는 원본 이미지를 완전히 재구성하지 못하지만, 파인튜닝을 수행하면 최적화된 임베딩을 넘어 입력 이미지의 세부 사항을 부여함으로써 중간 보간값에서도 이 세부 사항을 유지할 수 있다.

---

#### **Stage 3: 보간 (Interpolation)**

최종적으로, $e_{opt}$와 $e_{tgt}$ 사이를 보간하여 최종 편집 결과를 생성한다.

보간 수식:

$$e_{interp} = (1 - \eta) \cdot e_{opt} + \eta \cdot e_{tgt}, \quad \eta \in [0, 1]$$

여기서:
- $\eta = 0$: 원본 이미지에 가까운 출력 (높은 충실도)
- $\eta = 1$: 대상 텍스트에 가까운 출력 (높은 편집성)
- $\eta$: **편집성(editability)**과 **충실도(fidelity)** 사이의 트레이드오프 파라미터

주어진 실제 이미지와 대상 텍스트 프롬프트에서: (A) 대상 텍스트를 인코딩하고 초기 임베딩 $e_{tgt}$를 얻은 다음, 이를 최적화하여 $e_{opt}$를 획득하고; (B) $e_{opt}$를 고정한 채 생성 모델을 파인튜닝하여 입력 이미지에 대한 충실도를 높이고; (C) 마지막으로 $e_{opt}$와 $e_{tgt}$를 보간하여 최종 편집 결과를 생성한다.

---

### 🟣 2-3. 모델 구조

Imagic은 사전 학습된 텍스트-이미지 확산 모델을 활용하며, **Imagic의 공식화는 확산 모델 선택에 독립적**으로(agnostic), Imagen이나 Stable Diffusion 모두에 동일한 편집 요청을 적용할 수 있다.

구체적으로 모델 구조는 다음 요소로 구성된다:

| 구성 요소 | 역할 |
|---|---|
| **텍스트 인코더** | 텍스트 프롬프트 → 임베딩 (CLIP 또는 T5) |
| **U-Net (확산 모델)** | 노이즈 예측, Stage 2에서 파인튜닝 대상 |
| **텍스트 임베딩 공간** | $e_{tgt}$, $e_{opt}$ 최적화 및 보간이 이루어지는 공간 |
| **디노이징 프로세스** | 보간된 임베딩 $e_{interp}$를 조건으로 최종 이미지 생성 |

예를 들어, Imagen은 T5 언어 모델을 사용하는데, 이 모델은 텍스트의 토큰 수에 따라 길이가 달라지는 임베딩을 출력하여, 보간을 위해 두 임베딩이 동일한 길이여야 하는 조건이 필요하다. 이것이 텍스트 임베딩 최적화 단계의 필요성을 뒷받침한다.

---

### 🟡 2-4. 성능 향상

사용자 연구에서 9,213개의 답변을 집계한 결과, 평가자들은 **모든 고려된 기준선 대비 70% 이상의 선호율**로 Imagic을 강하게 선호하는 것으로 나타났다.

Imagic은 다양한 도메인의 수많은 입력에서 품질과 다양성을 시연하며, 높은 챌린지를 갖는 이미지 편집 벤치마크 TEdBench를 도입하고, 사용자 연구에서 인간 평가자들이 TEdBench에서 기존 주요 편집 방법들보다 Imagic을 선호한다는 결과를 보였다.

1024×1024 픽셀 원본 이미지와 편집된 이미지 쌍을 생성하며, 포즈 변경, 구성 변경, 다중 오브젝트 편집, 오브젝트 추가, 오브젝트 교체, 스타일 변경, 색상 변경 등 다양한 편집 유형을 지원한다.

---

### 🔴 2-5. 한계점

Imagic의 주요 한계는 아래와 같다:

1. **이미지별 파인튜닝의 필요**: 매 입력 이미지마다 Stage 1(임베딩 최적화)과 Stage 2(모델 파인튜닝)를 수행해야 하므로 추론 속도가 느리다. 이는 실시간 응용에 제약이 된다.
2. **편집성-충실도 트레이드오프**: 편집성(editability)과 충실도(fidelity) 사이의 트레이드오프가 존재하며, $\eta$ 파라미터를 수동으로 조정해야 한다.
3. **복잡한 장면 및 다중 오브젝트의 한계**: 복잡한 구도의 이미지나 다수의 오브젝트가 얽힌 경우 원치 않는 변화가 생길 수 있다.
4. **모델 일반화의 부재**: 특정 이미지에 대해 모델을 파인튜닝하기 때문에 다른 이미지에는 적용 불가하다(모델이 일반화되지 않음).
5. **랜덤 시드 민감성**: 논문에서 모델 파인튜닝과 보간 강도에 대한 ablation study를 수행했으며, 텍스트 임베딩 최적화의 필요성 및 랜덤 시드 변화에 대한 민감도에 대한 추가 ablation study도 제시하였다.

---

## 3. 모델의 일반화 성능 향상 가능성

Imagic의 가장 큰 구조적 한계는 **이미지별(per-image) 파인튜닝**으로 인한 일반화 부재이다. 하지만 다음과 같은 방향에서 일반화 성능 향상 가능성을 논의할 수 있다.

### 📌 3-1. 모델 불가지론적(Agnostic) 설계의 장점

Imagic의 공식화는 확산 모델 선택에 독립적으로, Imagen과 Stable Diffusion 모두에 동일한 편집 요청을 적용한 여러 예시를 보여준다. 이는 향후 더 강력한 기반 모델이 등장할수록 Imagic 프레임워크도 자동으로 성능이 향상되는 구조임을 의미한다.

### 📌 3-2. LoRA 등 파라미터 효율적 파인튜닝 기법의 통합

비공식 구현체에서는 LoRA(Low-Rank Adaptation)를 활용하여 24GB VRAM 환경에서 파인튜닝 단계를 수용하고, 파인튜닝 학습률을 LoRA의 기본 설정과 맞추어 조정하는 방법이 시도되었다. 이는 **파인튜닝 비용을 대폭 줄이면서 일반화에 가까운 방향으로 발전**할 수 있음을 시사한다.

### 📌 3-3. FastEdit 등 가속화 연구의 등장

FastEdit은 의미론적 인식 확산 파인튜닝을 통한 빠른 텍스트 기반 단일 이미지 편집 방법으로, 편집 과정을 단 17초로 획기적으로 가속화하고, U-Net에 LoRA를 적용하였다. 이처럼 파인튜닝 속도 향상을 통해 일반화와 실용성을 동시에 높이는 방향이 주목받고 있다.

### 📌 3-4. TEdBench를 통한 표준화된 평가

TEdBench는 복잡한 비강체적 편집을 설명하는 100쌍의 입력 이미지-텍스트 쌍의 새로운 컬렉션이며, 향후 연구가 이 표준화된 평가 세트를 활용할 수 있도록 Imagic의 결과와 함께 공개하였다. 이 벤치마크는 일반화 성능 측정의 기준점이 될 수 있다.

### 📌 3-5. 편집 가능성과 충실도의 동시 개선

텍스트 임베딩 보간의 핵심 수식:

$$e_{interp}(\eta) = (1-\eta) \cdot e_{opt} + \eta \cdot e_{tgt}$$

이 선형 보간 구조는 단순하지만 강력하다. 텍스트-이미지 확산 모델의 강력한 구성적(compositional) 능력을 드러내는, 두 텍스트 임베딩 시퀀스 사이의 의미론적으로 의미 있는 선형 보간을 시연한다. 향후 비선형 보간이나 조건부 보간으로 확장하면 더 정밀한 제어가 가능해질 수 있다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 📊 주요 관련 연구 비교표

| 방법 | 연도 | 입력 | 마스크 필요 | 파인튜닝 | 비강체 편집 | 실제 이미지 |
|---|---|---|---|---|---|---|
| **SDEdit** | 2021 | 이미지+텍스트 | ❌ | ❌ | 제한적 | ✅ |
| **Prompt-to-Prompt** | 2022 | 텍스트 수정 | ❌ | ❌ | ❌ | ❌ (합성) |
| **Textual Inversion** | 2022 | 이미지+텍스트 | ❌ | ✅ | ❌ | ✅ |
| **DreamBooth** | 2023 | 다중 이미지 | ❌ | ✅ | ❌ | ✅ |
| **Imagic** | 2023 | 이미지+텍스트 | ❌ | ✅(per-image) | **✅** | ✅ |
| **InstructPix2Pix** | 2023 | 이미지+명령어 | ❌ | ❌ | 부분적 | ✅ |
| **MagicBrush** | 2024 | 이미지+명령어 | 선택 | ❌ | 부분적 | ✅ |

---

### 🔵 4-1. Prompt-to-Prompt (Hertz et al., 2022)

Prompt-to-Prompt는 사전 학습된 텍스트 조건부 확산 모델에서 이미지 편집을 위한 텍스트 임베딩에 해당하는 어텐션 맵을 조작하는 데 초점을 맞추며, Null-text inversion은 관련 프롬프트와 함께 텍스트 기반 확산 모델의 잠재 공간으로 입력 이미지의 DDIM 역변환을 수행하여 직관적인 텍스트 기반 이미지 편집을 가능하게 한다.

- **Imagic 대비 차이**: Prompt-to-Prompt는 주로 **합성 이미지**에 효과적이며, 실제 이미지 편집은 추가적인 Null-text inversion이 필요하다. 또한 비강체적 편집이 어렵다.

---

### 🟠 4-2. InstructPix2Pix (Brooks et al., CVPR 2023)

주어진 이미지와 해당 이미지를 어떻게 편집할지에 대한 지시가 주어지면 모델이 적절한 편집을 수행하며, **전체 입력 또는 출력 이미지에 대한 상세 설명이 필요 없고, 예시별 역변환(inversion)이나 파인튜닝 없이** 순전파(forward pass)에서 이미지를 편집한다.

InstructPix2Pix는 생성된 데이터로 학습되며, 추론 시 실제 이미지와 사용자가 작성한 명령어로 일반화된다. 순전파에서 편집을 수행하고 예시별 파인튜닝이나 역변환이 필요 없으므로 수 초 내에 빠르게 이미지를 편집한다.

**Imagic vs. InstructPix2Pix 핵심 차이**:
- InstructPix2Pix는 추론 속도가 훨씬 빠르지만 복잡한 비강체적 편집(예: 포즈 변경)에서는 Imagic보다 부족하다.
- Imagic은 이미지별 파인튜닝으로 높은 충실도를 달성하지만 속도가 느리다.

---

### 🟢 4-3. MagicBrush (Zhang et al., 2024)

오프-더-셸프 InstructPix2Pix 체크포인트는 단일 및 다중 턴 시나리오 모두에서 다른 기준선에 비해 경쟁력이 없지만, MagicBrush로 파인튜닝한 후에는 대부분의 지표에서 최우수 또는 차우수 결과를 달성한다. 이는 고품질 수동 주석 데이터의 중요성을 보여주며, Imagic의 파인튜닝 기반 접근법과 상호 보완적 관점을 제시한다.

---

### 🔴 4-4. TurboEdit (SIGGRAPH Asia 2024)

TurboEdit은 단 3번의 확산 단계(A5000 GPU 기준 0.321초)만으로 실제 이미지의 텍스트 기반 편집을 가능하게 한다. 이는 Imagic의 가장 큰 약점인 속도 문제를 해결하는 방향의 연구이다.

---

### 🟡 4-5. LEdits++ 및 아키텍처 독립적 방법

LEdits++의 새로운 역변환 접근법은 튜닝이나 최적화가 필요 없으며, 몇 번의 확산 단계만으로 고충실도 결과를 생성하고, 복수의 동시 편집을 지원하며 아키텍처에 독립적이다.

---

## 5. 앞으로의 연구에 미치는 영향 및 연구 시 고려할 점

### 🚀 5-1. 미래 연구에 미치는 영향

#### ① 텍스트-이미지 잠재 공간의 의미론적 이해 심화

Imagic은 두 텍스트 임베딩 사이의 의미론적으로 의미 있는 선형 보간을 시연함으로써, 텍스트-이미지 확산 모델의 강력한 구성적 능력을 드러낸다. 이는 향후 잠재 공간의 구조를 더 깊이 이해하고 활용하는 연구의 토대를 마련한다.

#### ② TEdBench를 통한 벤치마크 표준화

TEdBench는 SDEdit, DDIB, Text2Live 등 세 가지 방법을 주로 비교하며, 이후 연구들이 이 벤치마크를 기준으로 성능 비교를 수행할 수 있게 하여 분야 발전의 기준점이 되고 있다.

#### ③ 단일 이미지 편집 패러다임의 확산

Imagic이 단일 실제 이미지에서 복잡한 편집을 처음 시연한 이후, 후속 연구로 InstructPix2Pix, Instruct-Diffusion, Imagic, DDPM-Inversion, LEDITS++, ProxEdit 등 다양한 방법들이 이 방향에서 발전하고 있다.

#### ④ 의료·특수 도메인으로의 확장

PRISM 프레임워크는 Stable Diffusion을 활용하여 고해상도의 언어 기반 의료 이미지 counterfactual을 생성하며, 허위 상관관계와 질병 특징을 선택적으로 수정하는 전례 없는 정밀도를 보여준다. 이처럼 Imagic의 아이디어는 의료 영상 분석 등 특수 도메인으로 확장되고 있다.

---

### ⚠️ 5-2. 앞으로 연구 시 고려할 점

#### ① 추론 속도 vs. 편집 품질의 트레이드오프

Imagic은 이미지별 파인튜닝으로 높은 품질을 달성하지만 실용성이 낮다. InstructPix2Pix와 같이 예시별 역변환이나 파인튜닝 없이 순전파에서 이미지를 편집하는 방향과 Imagic의 품질을 결합한 연구가 필요하다.

#### ② 파라미터 효율적 파인튜닝(PEFT) 기법 통합

LoRA를 활용하여 파인튜닝 단계를 24GB VRAM 환경에서 수용하는 접근법처럼, LoRA·Adapter 등 PEFT 기법을 활용하면 파인튜닝 비용을 대폭 줄이면서 품질을 유지할 수 있다.

#### ③ 편집성-충실도 균형의 자동화

보간 파라미터 $\eta$의 수동 조정은 사용자 경험을 저하시킨다. 콘텐츠와 텍스트 프롬프트의 의미적 거리를 기반으로 $\eta$를 자동 추정하는 알고리즘 연구가 필요하다.

#### ④ 다중 편집(Multi-Edit) 지원

LEdits++처럼 복수의 동시 편집을 지원하는 아키텍처 독립적 접근법으로 발전할 필요가 있으며, 현재 Imagic은 단일 텍스트 지시에 의한 단일 편집에 집중되어 있다.

#### ⑤ 데이터 품질 및 평가 메트릭의 고도화

InstructPix2Pix를 MagicBrush로 파인튜닝한 결과가 가장 좋더라도 편집된 이미지는 실제 정답 이미지보다 여전히 현저히 낮은 수준이며, 이는 현재 방법과 실제 편집 요구 사이의 간극을 보여준다. 보다 정교한 평가 메트릭(예: 편집 국소성, 아이덴티티 보존 점수 등)의 개발이 필요하다.

#### ⑥ 비선형 보간 및 조건부 보간 탐구

현재의 선형 보간 $e_{interp} = (1-\eta)e_{opt} + \eta e_{tgt}$ 대신, 곡선형(geodesic) 보간이나 콘텐츠 인식(content-aware) 보간으로 더 자연스러운 편집 결과를 얻는 연구가 유망하다.

---

## 📚 참고 자료 및 출처

| # | 제목 | 출처 |
|---|---|---|
| 1 | Imagic: Text-Based Real Image Editing with Diffusion Models | [arXiv:2210.09276](https://arxiv.org/abs/2210.09276) |
| 2 | Imagic 프로젝트 페이지 | [imagic-editing.github.io](https://imagic-editing.github.io/) |
| 3 | Imagic CVPR 2023 공식 논문 (Open Access) | [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2023/papers/Kawar_Imagic_Text-Based_Real_Image_Editing_With_Diffusion_Models_CVPR_2023_paper.pdf) |
| 4 | ar5iv (arXiv HTML 렌더링) | [ar5iv.labs.arxiv.org/html/2210.09276](https://ar5iv.labs.arxiv.org/html/2210.09276) |
| 5 | Imagic Semantic Scholar | [semanticscholar.org](https://www.semanticscholar.org/paper/Imagic:-Text-Based-Real-Image-Editing-with-Models-Kawar-Zada/23e261a20a315059b4de5492ed071c97a20c12e7) |
| 6 | InstructPix2Pix: Learning to Follow Image Editing Instructions (Brooks et al., CVPR 2023) | [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2023/papers/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.pdf) |
| 7 | MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing | [arxiv.org/html/2306.10012](https://arxiv.org/html/2306.10012v3) |
| 8 | TurboEdit: Text-Based Image Editing Using Few-Step Diffusion Models (SIGGRAPH Asia 2024) | [dl.acm.org](https://dl.acm.org/doi/10.1145/3680528.3687612) |
| 9 | Prompt-to-Prompt Image Editing with Cross-Attention Control (Hertz et al., 2022) | [prompt-to-prompt.github.io](https://prompt-to-prompt.github.io/) |
| 10 | Image Editing with Diffusion Models: A Survey (arXiv 2025) | [arxiv.org/html/2504.13226](https://arxiv.org/html/2504.13226v1) |
| 11 | Papers Decoded — Imagic (Medium) | [medium.com/@chongdashu](https://medium.com/@chongdashu/papers-decoded-imagic-text-based-real-image-editing-with-diffusion-models-b1bda8b2532a) |
| 12 | Unofficial Imagic Implementation (GitHub) | [github.com/sangminkim-99/Imagic](https://github.com/sangminkim-99/Imagic) |
| 13 | Text based Image Editing using Diffusion Model (IJISAE) | [ijisae.org](https://ijisae.org/index.php/IJISAE/article/download/5435/4161/10495) |
| 14 | Instruction-tuning Stable Diffusion with InstructPix2Pix (Hugging Face Blog) | [huggingface.co/blog/instruction-tuning-sd](https://huggingface.co/blog/instruction-tuning-sd) |
| 15 | Textualize Visual Prompt for Image Editing via Diffusion Bridge (arXiv 2025) | [arxiv.org/html/2501.03495](https://arxiv.org/html/2501.03495) |

# Imagic: Text-Based Real Image Editing with Diffusion Models

## 1. 핵심 주장 및 주요 기여

**Imagic**은 텍스트 기반 의미론적 이미지 편집을 위한 혁신적인 방법으로, 다음의 핵심 주장을 제시합니다:[1]

**주요 기여**:
1. **단일 실제 이미지에 대한 복잡한 비강체(non-rigid) 편집의 첫 구현**: 자세 변화, 객체 구성 변경 등 정교한 의미론적 편집을 단 하나의 입력 이미지만으로 수행[1]

2. **의미론적으로 의미 있는 텍스트 임베딩 선형 보간의 발견**: 텍스트-이미지 확산 모델이 강력한 합성 능력을 가지고 있음을 보여줌[1]

3. **TEdBench 벤치마크 소개**: 복잡한 비강체 이미지 편집을 평가하기 위한 최초의 표준화된 100개 이미지-텍스트 쌍 벤치마크 제시[1]

***

## 2. 문제 정의, 제안 방법 및 모델 구조

### 2.1 해결하고자 하는 문제

기존 텍스트 기반 이미지 편집 방법들의 한계:[1]

- **제한된 편집 유형**: 객체 오버레이, 스타일 전이 등 특정 편집만 가능
- **제한된 이미지 도메인**: 합성 생성 이미지나 특정 도메인에만 동작
- **추가 입력 요구**: 마스크, 다중 뷰, 원본 텍스트 설명 등 보조 정보 필수

Imagic은 **단일 입력 이미지**와 **목표 텍스트만**으로 실제 고해상도 이미지에 복잡한 비강체 편집을 적용하는 첫 방법입니다.

### 2.2 제안 방법: 3단계 파이프라인

#### **단계 1: 텍스트 임베딩 최적화**

목표 텍스트를 텍스트 인코더에 통과시켜 초기 임베딩 $$e_{tgt} \in \mathbb{R}^{T \times d}$$을 얻고, 확산 모델의 가중치를 고정한 후 다음 손실 함수로 최적화합니다:[1]

$$L(x, e, \theta) = \mathbb{E}_{t,\epsilon} \left[\left\|\epsilon - f_\theta(x_t, t, e)\right\|^2\right]$$ 

여기서 $t \sim \text{Uniform}[1, T]$이고, $x_t$는 입력 이미지 $x$의 노이즈 버전입니다. 이 과정은 상대적으로 적은 단계 수(100단계)로 진행되어 $$e_{opt}$$를 얻고, 초기 임베딩 근처에 머물게 합니다.[1]

#### **단계 2: 모델 파인튜닝**

최적화된 임베딩 $$e_{opt}$$가 입력 이미지를 정확히 재구성하지 못할 수 있으므로, 임베딩을 고정하고 확산 모델의 가중치 $$\theta$$를 같은 손실 함수로 파인튜닝합니다:[1]

$$L(x, e_{opt}, \theta) = \mathbb{E}_{t,\epsilon} \left[\left\|\epsilon - f_\theta(x_t, t, e_{opt})\right\|^2\right]$$

동시에 초해상도 모델들도 목표 텍스트 임베딩 $$e_{tgt}$$로 1500단계 파인튜닝합니다.[1]

#### **단계 3: 텍스트 임베딩 보간**

최종 편집을 위해 최적화된 임베딩과 목표 임베딩 사이를 선형 보간합니다:[1]

$$\bar{e} = \eta \cdot e_{tgt} + (1-\eta) \cdot e_{opt}$$

여기서 $$\eta \in $$은 편집 강도를 제어하는 하이퍼파라미터로, 일반적으로 0.6~0.8 범위에서 최적의 결과를 얻습니다.[1]

### 2.3 모델 아키텍처

Imagic은 두 가지 최첨단 텍스트-이미지 확산 모델과 호환됩니다:[1]

**Imagen 기반 구현**:
- 64×64 기본 생성 확산 모델
- 64×64 → 256×256 초해상도 모델
- 256×256 → 1024×1024 초해상도 모델
- 분류자 없는 안내(Classifier-free guidance) 통합[1]

**Stable Diffusion 기반 구현**:
- 잠재 공간(4×64×64)에서 작동
- 512×512 이미지 해상도 지원
- 텍스트 임베딩 최적화: Adam 옵티마이저로 1000단계, 학습률 2e-3
- 모델 파인튜닝: 1500단계, 학습률 5e-7[1]

***

## 3. 성능 향상 및 한계

### 3.1 성능 향상

**사용자 연구 결과**:[1]
- SDEdit 대비 **70% 이상** 선호도
- DDIB 대비 **80% 이상** 선호도  
- Text2LIVE 대비 **90% 이상** 선호도
- 총 9213개 사용자 응답 기반 평가

**정량 지표** (150개 입력 기준):[1]
- $$\eta \in [0.6, 0.8]$$ 범위에서 최적 성능
- CLIP 점수(텍스트 정렬도)와 1-LPIPS(이미지 충실도) 간의 최적 균형

**적응성**:[1]
- 이미지 도메인에 관계없이 광범위한 편집 유형 지원
- 스타일 변화, 색상 변화, 자세 변화, 객체 추가/제거 모두 가능
- 확률적 생성 모델의 특성으로 단일 이미지-텍스트 쌍에 대해 여러 편집 옵션 제공

### 3.2 주요 한계

**실패 사례** (Figure 10):[1]

1. **불충분한 편집**: 원하는 편집이 너무 미묘하거나 전혀 적용되지 않음
   - 증가된 $$\eta$$로 해결 가능하지만, 때로 원본 이미지 세부사항 손실

2. **외재적 변경**: 줌, 카메라 각도 등이 원하지 않게 변경
   - $$\eta$$ 값 증가 과정에서 원하는 편집 이전에 발생

**내재적 제약**:[1]
- 기본 확산 모델의 생성 한계 상속 (예: Imagen의 인간 얼굴 표현 부족)
- 모델 편향 상속
- **계산 비용**: Imagen 기준 2개 TPUv4 칩에서 이미지당 약 8분 소요
- **무작위 시드 민감성**: 같은 입력에 대해 다른 시드는 다른 $$\eta$$ 값에서 최적 결과 도출

***

## 4. 모델 일반화 성능 향상 가능성

### 4.1 현재 상태

텍스트 임베딩 최적화 전략은 이미지 특정 최적화이므로, 새로운 이미지마다 100단계의 임베딩 최적화 + 1500단계의 파인튜닝이 필요합니다. 이는 다음을 의미합니다:[1]

- **낮은 직접 일반화**: 단일 이미지 최적화 방식의 특성상 다른 이미지로의 직접 전이 불가
- **도메인 의존성**: 특정 도메인의 이미지에 최적화되지만, 아키텍처 자체는 도메인 무관적

### 4.2 최근 연구 기반 개선 방안

#### **1) Fast Imagic (2024)**[2]

**목표**: Imagic의 느린 최적화 속도와 과적합 문제 해결

**개선 사항**:
- **14배 속도 향상**: 약 8분 → 30초로 단축
- **비전-언어 공동 최적화 프레임워크**: 이미지 인코딩과 텍스트 임베딩을 함께 최적화하여 더 빠른 수렴
- **분리된 UNet 파인튜닝**: UNet 인코더는 공간/구조를, 디코더는 외형/질감을 학습하는 속성 활용
- **망각 메커니즘**: 원본 체크포인트와 최적화된 체크포인트를 병합하여 과적합 해결[2]

#### **2) InstructGIE (2024)**[3]

**강화된 일반화 능력**:
- **In-context 학습 기능 향상**: VMamba 블록과 편집-시프트 매칭 전략으로 시각적 프롬프트 활용
- **언어 명령 통일**: 언어 임베딩을 편집 의미론과 정렬
- **선택적 영역 매칭**: 왜곡된 세부 사항(특히 얼굴 특징) 재정정[3]

#### **3) DragText (2024)**[4]

**텍스트 임베딩 최적화 확장**:
- **포인트 기반 편집과 텍스트 임베딩 동시 최적화**
- **텍스트 임베딩 정규화**: 원본 임베딩으로부터의 발산 방지
- **플러그-앤-플레이 방식**: 다양한 확산 기반 드래그 모델에 적용 가능[4]

#### **4) OmniEdit (2024)**[5]

**지시 기반 편집의 일반화 개선**:
- 자동 합성 또는 수동 주석 이미지 편집 쌍으로 학습
- 기존 방법을 능가하는 성능 달성[5]

#### **5) Attention Interpolation (NeurIPS 2024)**[6]

**텍스트 임베딩 보간의 한계 극복**:
- **발견**: 텍스트 임베딩 보간의 수학적 한계
  - 자기 주의가 교차 주의보다 더 강한 영향력 행사
  - 단순 선형 보간은 일관성 있는 결과 생성 실패
- **개선**: 
  - 교차 주의와 자기 주의 모두의 보간된 주의 메커니즘
  - 베타 분포 기반 샘플 선택으로 부드러운 보간
  - Imagic의 텍스트 임베딩 보간 대비 현저히 개선된 일관성과 충실도[6]

#### **6) 합성 생성성 메커니즘 (2024)**[7]

**조건부 확산 모델의 합성 생성성**:
- 조건부 확산 모델이 훈련 분포 외 조건 조합을 생성할 수 있음을 증명
- **로컬 조건부 점수(Local Conditional Scores)**: 픽셀과 조건자에 대한 희소 의존성
- 이론적으로 길이 일반화(length generalization)가 가능함을 입증[7]

### 4.3 일반화 성능 향상의 핵심 방향

**1. 임베딩 공간 기하학 활용**[8]
- 최적 전송 이론을 적용한 임베딩 보간
- Wasserstein 공간에서의 측지선 기반 보간
- 더 자연스럽고 기하학적으로 부드러운 전환[8]

**2. 세밀한 제어 및 분리**
- 텍스트 임베딩과 이미지 특성의 분리된 학습
- 정체성 보존과 편집 강도의 독립적 제어
- 교차 주의 레이어의 세밀한 조작[9]

**3. 효율성 개선을 통한 확장성**
- 메모리 효율적 파인튜닝[10]
- 양자화된 확산 모델에 대한 최적화
- 훈련 없는(training-free) 방법론 개발[11]

**4. 다중 모달 정렬**
- 언어 명령 통일을 통한 일관된 의미론
- 프롬프트 유도 주의 보간
- 사용자 지정 경로를 통한 보간[6]

***

## 5. 연구의 영향과 미래 고려사항

### 5.1 Imagic의 학계 영향

Imagic은 **텍스트 기반 이미지 편집 분야의 패러다임 전환**을 가져왔습니다:

**학술적 영향**:
- **단일 이미지 편집의 가능성 입증**: 이전까지 불가능하던 복잡한 비강체 편집을 처음 구현
- **텍스트 임베딩 최적화 기법의 확립**: 후속 연구의 기반이 된 핵심 방법론
- **벤치마크 표준화**: TEdBench를 통한 객관적 평가 체계 제시[1]
- **확산 모델 이해 심화**: 텍스트-이미지 확산 모델의 합성 능력 탐색[7]

**산업적 영향**:
- 이미지 편집 소프트웨어의 신로직 제시
- 창작자 도구의 접근성 향상 (마스크 등 추가 입력 불필요)

### 5.2 미래 연구 시 고려사항

#### **1. 계산 효율성 (Urgent Priority)**

현재 **이미지당 8분 소요**는 실제 애플리케이션 배포의 장벽입니다:

- **개선 방안**:
  - Fast Imagic의 30초 달성 기법 활용
  - 양자화 및 경량화 연구[10]
  - 캐싱 및 마이크로프로세싱 기법[11]

#### **2. 객체 정체성 보존**

현재 **얼굴 세부사항 손실** 및 **카메라 각도 변경** 문제 해결:[1]

- **접근법**:
  - 정체성 토큰의 직교성 제약[12]
  - 마스크 기반 관심 영역 제한
  - 크로스 주의 제어 통합[9]

#### **3. 일반화된 프롬프트 처리**

무작위 시드 민감성 감소:[1]

- **해결책**:
  - 자동 $$\eta$$ 선택 메커니즘 개발
  - 여러 시드로부터의 앙상블 기법
  - 강화 학습 기반 최적값 추론

#### **4. 다중 객체 편집의 세밀한 제어**

복잡한 합성 편집에서의 독립적 객체 제어:[13]

- **향후 방향**:
  - 객체 레벨 분해
  - 계층적 편집 프레임워크
  - 기존 주의 제어 기법 통합[9]

#### **5. 크로스 모달 정렬 개선**

**최근 트렌드**:[6]
- 주의 보간을 통한 더 정교한 조건부 제어
- 사용자 가이드 보간 경로
- 멀티모달 입력(텍스트, 이미지, 마스크)의 동시 처리

#### **6. 모델 편향 및 윤리**

기본 모델의 편향 상속 문제:[1]

- **해결 노력**:
  - 값 정렬 프레임워크 (LiVO)[14]
  - 윤리적 이미지 생성 가이드라인
  - 합성 콘텐츠 탐지 기술 발전

#### **7. 비디오 및 3D 확장**

**새로운 패러다임**:[15]
- 시간적 일관성을 유지한 동영상 편집
- 3D 기하학 제약 통합
- 다중 뷰 일관성 보장

***

## 결론

Imagic은 **단일 실제 이미지에 대한 복잡한 텍스트 기반 의미론적 편집을 처음 구현**한 획기적 방법입니다. 텍스트 임베딩 최적화, 모델 파인튜닝, 선형 보간의 3단계 파이프라인을 통해 기존 방법의 한계를 극복했습니다.

현재 **일반화 성능 향상**은 주로 다음 방향으로 진행 중입니다:

1. **계산 효율성 극적 개선** (Fast Imagic)
2. **주의 메커니즘 기반 고급 보간** (Attention Interpolation)  
3. **정체성 보존 및 분리 학습** (S²Edit, DragText)
4. **멀티모달 통합** (InstructGIE, OmniEdit)

Imagic의 핵심 기여인 **텍스트 임베딩 최적화 기법**은 후속 연구의 표준 방법론이 되었으며, 향후 연구는 이를 기반으로 효율성, 세밀한 제어, 확장성 개선에 집중할 것으로 예상됩니다.[12][14][5][11][3][10][2][8][4][7][9][6][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0f357bb2-14ec-4e55-a977-0beef5f5a11a/2210.09276v3.pdf)
[2](https://openreview.net/forum?id=PoLsUIDY0c)
[3](https://arxiv.org/html/2403.05018v2)
[4](https://arxiv.org/html/2407.17843v1)
[5](https://arxiv.org/html/2411.07199)
[6](https://proceedings.neurips.cc/paper_files/paper/2024/file/b12a1d1014e952e676f5d6931d03241a-Paper-Conference.pdf)
[7](https://arxiv.org/abs/2509.16447)
[8](https://openreview.net/pdf/fdcf4c078fecf7bfeb1164a8dd6cd579555acc13.pdf)
[9](https://ieeexplore.ieee.org/document/9706340/)
[10](https://pure.kaist.ac.kr/en/publications/memory-efficient-fine-tuning-forquantized-diffusion-model/)
[11](https://arxiv.org/pdf/2403.12585.pdf)
[12](https://arxiv.org/html/2507.04584v1)
[13](https://arxiv.org/html/2503.12652v1)
[14](https://dl.acm.org/doi/10.1145/3664647.3681652)
[15](https://www.emergentmind.com/topics/diffusion-based-image-editing)
[16](https://link.springer.com/10.1007/s10489-025-06673-1)
[17](http://pubs.rsna.org/doi/10.1148/radiol.240343)
[18](http://pubs.rsna.org/doi/10.1148/rycan.240287)
[19](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[20](https://doi.apa.org/doi/10.1037/emo0001511)
[21](https://aca.pensoft.net/article/151406/)
[22](http://arxiv.org/pdf/2403.05018.pdf)
[23](http://arxiv.org/pdf/2306.14435.pdf)
[24](https://arxiv.org/html/2303.17546v3)
[25](https://arxiv.org/pdf/2402.02583.pdf)
[26](https://arxiv.org/html/2408.08495)
[27](https://aclanthology.org/2023.findings-emnlp.646.pdf)
[28](https://eccv.ecva.net/virtual/2024/poster/1781)
[29](https://www.scitepress.org/publishedPapers/2024/132410/pdf/index.html)
[30](https://arxiv.org/html/2504.13226v1)
[31](https://openreview.net/forum?id=pfS4D6RWC8)
[32](https://www.computer.org/csdl/journal/tp/2025/06/10884879/24j49AKyjO8)
[33](https://arxiv.org/abs/2410.00321)
[34](https://link.springer.com/10.1007/s11760-024-03268-0)
[35](https://arxiv.org/abs/2408.15914)
[36](https://ieeexplore.ieee.org/document/10552817/)
[37](https://www.mdpi.com/2813-2203/4/1/4)
[38](https://www.semanticscholar.org/paper/ce13af4d467c4b01c4af570bd154317ae25ec892)
[39](https://ieeexplore.ieee.org/document/10684754/)
[40](https://ieeexplore.ieee.org/document/10239477/)
[41](https://arxiv.org/pdf/2308.03281.pdf)
[42](https://www.aclweb.org/anthology/2021.naacl-main.457.pdf)
[43](https://arxiv.org/pdf/2307.05610.pdf)
[44](https://arxiv.org/pdf/2401.08472.pdf)
[45](http://arxiv.org/pdf/2412.11652.pdf)
[46](http://arxiv.org/pdf/2305.05665.pdf)
[47](https://arxiv.org/pdf/2402.16829.pdf)
[48](https://arxiv.org/pdf/2311.02084.pdf)
[49](https://pmc.ncbi.nlm.nih.gov/articles/PMC7668300/)
[50](https://www.scribd.com/document/635304588/Untitled)
[51](https://huggingface.co/papers/2307.12560)
[52](https://arxiv.org/html/2506.08844v1)
[53](https://lvelho.impa.br/ip23/proj/slides/Imagic.pdf)
[54](https://www.sciencedirect.com/science/article/pii/S0048733320302225)
