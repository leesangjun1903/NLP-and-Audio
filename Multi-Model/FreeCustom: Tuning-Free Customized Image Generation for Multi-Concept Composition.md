
# FreeCustom: Tuning-Free Customized Image Generation for Multi-Concept Composition

> **논문 정보**
> - **저자**: Ganggui Ding, Canyu Zhao, Wen Wang, Zhen Yang, Zide Liu, Hao Chen, Chunhua Shen (Zhejiang University, China)
> - **발표**: CVPR 2024 (Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9089–9098)
> - **arXiv**: [2405.13870](https://arxiv.org/abs/2405.13870)
> - **프로젝트 페이지**: https://aim-uofa.github.io/FreeCustom/
> - **GitHub**: https://github.com/aim-uofa/FreeCustom

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

대규모 사전학습된 텍스트-이미지(T2I) 생성 모델 덕분에, 사용자가 지정한 개념을 생성하는 커스터마이즈 이미지 생성 분야에서 상당한 진전이 이루어졌다. 그러나 기존 접근법은 단일 개념 커스터마이징에 집중하고 있으며, 다중 개념을 결합하는 복잡한 시나리오에서는 여전히 어려움을 겪는다.

이러한 방법들은 소수의 이미지로 재학습/파인튜닝을 요구하여 시간이 많이 걸리고 빠른 구현을 방해한다. 게다가, 하나의 개념을 표현하기 위해 여러 이미지에 의존하면 커스터마이징의 난이도가 증가한다.

이에 FreeCustom은 **개념당 단 하나의 이미지**만을 입력으로 사용하여, 참조 개념에 기반한 다중 개념 합성의 커스터마이즈 이미지를 생성하는 **새로운 튜닝 프리(tuning-free) 방법**이다.

### 주요 기여 (Contributions)

FreeCustom의 핵심 기여는 세 가지이다: ① **FreeCustom 방법론 제안** — 단일 개념 커스터마이징과 다중 개념 합성 모두에서 고품질 결과를 일관되게 제공하는 새로운 튜닝-프리 방법; ② **MRSA 메커니즘과 가중치 마스크 전략 제안** — 생성 이미지가 입력 개념과 상호작용하고 더 집중할 수 있도록 함; ③ **컨텍스트 인터랙션의 중요성 발견** — 이를 활용하여 고충실도 커스터마이즈 이미지 생성.

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

기존 커스터마이징 방법은 두 가지 범주로 분류된다: (a) 학습 기반 방법과 (b) 범용화를 위한 맞춤형 모델. 학습 기반 방법은 전체 모델을 파인튜닝하거나(Type I) 특정 피사체를 표현하기 위한 텍스트 임베딩을 학습하는(Type II) 방식이며, 맞춤형 모델은 범용 기반을 확립하기 위해 대규모 이미지 데이터셋으로 재학습을 요구한다.

핵심 문제는 다음 세 가지로 요약된다:

| 문제 | 설명 |
|---|---|
| **학습 비용** | 재학습/파인튜닝에 수백~수천 초 소요 |
| **다중 이미지 의존성** | 하나의 개념 표현에 여러 이미지 필요 |
| **다중 개념 합성 한계** | 복수 개념 결합 시 품질 저하 |

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① 전체 파이프라인 (Dual-Path Architecture)

참조 이미지 집합 $\mathcal{I} = \{I_1, I_2, I_3\}$와 대응 프롬프트 $\mathcal{P} = \{P_1, P_2, P_3\}$가 주어지면, 목표 프롬프트 $P$에 맞춘 다중 개념 합성 이미지 $I$를 생성한다. (a) VAE 인코더를 사용해 참조 이미지를 잠재 표현 $\mathbf{z}_0'$로 변환하고, 분할 네트워크로 개념의 마스크를 추출한다. (b) 디노이징 프로세스는 두 경로로 구성된다: 1) 개념 참조 경로(Concepts Reference Path), 2) 개념 합성 경로(Concepts Composition Path).

$$
\mathcal{I} = \{I_1, I_2, \ldots, I_N\}, \quad \mathcal{P} = \{P_1, P_2, \ldots, P_N\}
$$
$$
\xrightarrow{\text{VAE Encoder}} \mathbf{z}_0' \xrightarrow{\text{Forward Diffusion}} \mathbf{z}_t'
$$

#### ② 개념 참조 경로 (Concepts Reference Path)

이 경로에서는 확산 순방향 프로세스를 통해 $\mathbf{z}_0'$를 $\mathbf{z}_t'$로 변환하고, 이를 U-Net $\epsilon_\theta$에 전달한다. 단, $\epsilon_\theta$의 출력은 사용되지 않는다.

$$
\mathbf{z}_0' \xrightarrow{\text{Forward Diffusion}} \mathbf{z}_t' \xrightarrow{U\text{-Net}\ \epsilon_\theta} \text{(Key, Value 특성만 추출)}
$$

#### ③ 개념 합성 경로 (Concepts Composition Path)

이 경로에서는 초기에 $\mathbf{z}_T \sim \mathcal{N}(0, \mathbf{I})$를 샘플링하고, $\mathbf{z}_0$를 얻을 때까지 반복적으로 디노이징한다. 각 시간 단계 $t$마다, 현재 잠재값 $\mathbf{z}_t$를 수정된 U-Net $\epsilon_\theta^*$에 전달하고, MRSA를 사용해 U-Net $\epsilon_\theta$와 $\epsilon_\theta^*$의 마지막 두 블록 특성을 통합한다. 최종적으로 VAE 디코더를 통해 $\mathbf{z}_0$를 최종 이미지 $I$로 변환한다.

$$
\mathbf{z}_T \sim \mathcal{N}(0, \mathbf{I}) \xrightarrow{\text{Iterative Denoising via } \epsilon_\theta^*} \mathbf{z}_0 \xrightarrow{\text{VAE Decoder}} I
$$

#### ④ Multi-Reference Self-Attention (MRSA) 메커니즘 (핵심 수식)

MRSA 메커니즘의 핵심 수식은 다음과 같다:

$$
\text{MRSA}(\mathbf{Q}, \mathbf{K}', \mathbf{V}', \mathbf{M}_w) = \text{Softmax}\!\left(\frac{\mathbf{M}_w \odot (\mathbf{Q}\mathbf{K}'^T)}{\sqrt{d}}\right)\mathbf{V}'
$$

여기서:
- $\mathbf{Q}$: 합성 경로(composition path)에서의 쿼리(Query) 행렬
- $\mathbf{K}'$: 참조 이미지들로부터 연결(concatenate)된 키(Key) 행렬
- $\mathbf{V}'$: 참조 이미지들로부터 연결(concatenate)된 값(Value) 행렬
- $\mathbf{M}_w$: 가중치 마스크(Weighted Mask), 아다마르 곱 ($\odot$)으로 적용
- $d$: Key/Query의 차원

구체적으로 특성 주입(Feature Injection)은 U-Net의 특정 레이어에서 이루어진다. 참조 경로에서 추출된 Key 및 Value 텐서가 합성 경로의 Key, Value 텐서와 **연결(concatenation)**되어 새로운 $\mathbf{K}'$, $\mathbf{V}'$가 구성된다.

$$
\mathbf{K}' = \text{Concat}(\mathbf{K}_{\text{comp}},\ \mathbf{K}_{\text{ref}}^1,\ \mathbf{K}_{\text{ref}}^2,\ \ldots,\ \mathbf{K}_{\text{ref}}^N)
$$
$$
\mathbf{V}' = \text{Concat}(\mathbf{V}_{\text{comp}},\ \mathbf{V}_{\text{ref}}^1,\ \mathbf{V}_{\text{ref}}^2,\ \ldots,\ \mathbf{V}_{\text{ref}}^N)
$$

#### ⑤ Weighted Mask Strategy

어텐션 행렬은 Softmax 함수를 적용하기 전에 가중치 마스크와 **아다마르 곱(Hadamard product)**이 수행된다. 이는 관심 개념에 해당하지 않는 영역을 "마스킹"하여, 어텐션 메커니즘이 중요한 영역에 집중하도록 강제한다.

$$
\mathbf{M}_w = w_i \cdot \mathbf{M}_i + w_{\text{bg}} \cdot \mathbf{M}_{\text{bg}}, \quad \sum_i w_i = 1
$$

- $w_i$: 각 개념 $i$에 대한 가중치
- $\mathbf{M}_i$: 개념 $i$의 분할 마스크(segmentation mask)
- $\mathbf{M}_{\text{bg}}$: 배경 마스크

---

### 2-3. 모델 구조

전체 시스템은 듀얼 패스(dual-path) 아키텍처를 사용하여 다수의 입력 개념의 특성을 추출하고 결합한다. MRSA 메커니즘은 원래의 자기-어텐션(self-attention)을 확장하여 참조 개념의 특성에 접근하고 쿼리할 수 있도록 한다.

모든 자기-어텐션 레이어를 MRSA로 교체하면 아티팩트가 발생할 수 있으므로, 저자들은 U-Net의 **더 깊은 블록(deep blocks)**에 있는 자기-어텐션 모듈만 선택적으로 MRSA로 교체한다. 이는 U-Net의 깊은 레이어의 쿼리 특성이 레이아웃 제어와 의미론적 정보 획득의 역량을 보유하고 있기 때문이다.

전체 구조를 도식화하면:

```
[참조 이미지 I_1,...,I_N]
        ↓
   VAE Encoder → z'_0
        ↓ (Forward Diffusion)
       z'_t → U-Net ε_θ → K', V' 추출 (출력은 사용 X)
        ↑ MRSA로 주입
[합성 경로]
z_T ~ N(0,I) → Modified U-Net ε_θ* → z_0 → VAE Decoder → 최종 이미지 I
```

사용자는 커스터마이징하고자 하는 개념을 표현하는 2~3장의 이미지를 선택하고, 각 개념의 마스크를 얻기 위해 Grounded-Segment-Anything 또는 다른 분할 도구를 사용하여 관련 없는 픽셀을 필터링한다.

---

### 2-4. 성능 향상

FreeCustom은 다중 개념 합성에서 Custom Diffusion 및 Perfusion을 크게 능가하며, 평가된 5가지 지표 전반에서 우수한 점수를 달성했다: **DINOv2** (0.7625 vs 0.6545, 0.6399), **CLIP-I** (0.2871 vs 0.2393, 0.2277), **CLIP-T** (33.7826 vs 29.0702, 22.1371), **CLIP-T-L** (27.8758 vs 23.6657, 16.1719), **CLIP-IQA** (0.9002 vs 0.8921, 0.8624).

42개의 설문지를 포함한 사용자 연구에서도 모든 세 가지 기준에서 FreeCustom의 우월한 성능이 입증되었다: 이미지-프롬프트 정렬(4.40 vs 1.91, 1.50), 개념 일관성(4.65 vs 2.53, 1.70), 전반적인 이미지 품질(4.17 vs 2.48, 1.73).

시간 효율성 측면에서도, 튜닝-프리 특성 덕분에 전처리 시간이 0초인 반면 Custom Diffusion은 287초, Perfusion은 821초를 요구한다. 추론 시간은 경쟁 방법(13~14초)보다 약간 길지만(20초), 방대한 학습/파인튜닝 과정을 제거함으로써 다중 개념 생성 실용적 적용에서 FreeCustom이 훨씬 빠르다.

---

### 2-5. 한계점

저자들은 자신들의 방법이 매우 복잡하거나 추상적인 개념을 처리할 때 여전히 어려움을 겪을 수 있음을 인정하며, 생성된 이미지의 품질이 사전학습된 T2I 모델의 표현력에 의해 제한될 수 있음을 언급한다.

또한, 이 논문은 해로운 콘텐츠나 편향된 콘텐츠를 생성할 위험과 같은 이러한 생성 모델 사용에 관련된 잠재적 윤리적 문제를 다루지 않는다.

추가적으로 문헌 검토를 통해 파악한 한계는 다음과 같다:

| 한계 유형 | 설명 |
|---|---|
| **컨텍스트 이미지 의존성** | 개념이 컨텍스트 없이 단독으로 촬영된 경우 성능 저하 가능 |
| **기반 모델 한계** | Stable Diffusion 등 사전학습 모델의 성능에 종속 |
| **개념 수 제한** | 논문에서는 주로 2~3개 개념 조합을 검증 |
| **속성 누출(Attribute Leakage)** | 유사한 클래스의 여러 개념(예: 개 두 마리) 처리 시 속성 혼용 가능 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 컨텍스트 인터랙션의 역할

듀얼-패스 아키텍처를 통해 다수의 입력 개념의 특성을 추출하고 결합하여 특성 인식을 강화한다. 또한 컨텍스트 인터랙션을 포함한 이미지를 제공하는 것이 개념 정체성 보존에 핵심적임을 입증했으며, MRSA 메커니즘이 이 역량을 활용한다.

이는 실질적인 발견으로, **모자 이미지 단독 제공보다 모자를 착용한 사람의 이미지를 제공하는 것**이 더 나은 결과를 낳는다는 것을 의미한다.

### 3-2. 플러그-앤-플레이(Plug-and-Play) 일반화

FreeCustom은 **ControlNet 및 BLIP Diffusion과 플러그-앤-플레이 방식으로 결합**할 수 있다. FreeCustom을 사용하면 BLIP Diffusion의 출력이 입력 이미지에 더 충실해지고 입력 텍스트와 더 잘 정렬되며, ControlNet은 레이아웃과 정체성이 일관된 결과를 생성할 수 있다.

나아가, 다른 확산 기반 모델에도 쉽게 적용할 수 있다.

### 3-3. 다양한 개념 카테고리 일반화

FreeCustom은 어떠한 모델 파라미터 튜닝 없이도 다중 개념 조합과 단일 개념 커스터마이징에서 고품질 이미지를 신속하게 생성하는 데 탁월하다. 각 개념의 정체성이 놀라울 정도로 잘 보존된다. 다양한 범주의 개념을 처리할 때도 뛰어난 다양성과 견고성을 보인다. 이러한 다양성 덕분에 사용자들은 자신의 필요와 선호에 맞춰 다양한 개념 조합을 포함한 커스터마이즈 이미지를 생성할 수 있다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4-1. 관련 연구 분류 및 비교표

기존 커스터마이징 방법은 크게 두 가지로 분류된다: (a) 학습 기반 방법 — Type I: 전체 모델 파인튜닝, Type II: 텍스트 임베딩 학습; (b) 범용화를 위한 맞춤형 모델 — 대규모 이미지 데이터셋으로 재학습 필요.

| 논문 | 연도 | 방법 유형 | 다중 개념 | 튜닝 필요 | 이미지 수 |
|---|---|---|---|---|---|
| **Textual Inversion** (Gal et al.) | 2022 | 텍스트 임베딩 학습 | ❌ | ✅ | 3~5장 |
| **DreamBooth** (Ruiz et al.) | 2022 | 전체 모델 파인튜닝 | ❌ | ✅ | 3~5장 |
| **Custom Diffusion** (Kumari et al.) | 2023 | Cross-Attention 파인튜닝 | ⚠️ 제한적 | ✅ | 다수 |
| **BLIP-Diffusion** (Li et al.) | 2023 | 멀티모달 사전학습 | ❌ | ❌ | 1장 |
| **Mix-of-Show** (Gu et al.) | 2023 | LoRA 모듈 병합 | ✅ | ✅ | 다수 |
| **FreeCustom** (Ding et al.) | 2024 | MRSA + 가중치 마스크 | ✅ | ❌ | **1장/개념** |
| **FreerCustom** (확장판) | 2025 | MRSA + 이미지/비디오 | ✅ | ❌ | 1장/개념 |

DreamBooth는 확산 모델의 모든 파라미터를 파인튜닝하고 정규화 데이터셋으로 생성된 이미지를 사용하는 반면, Textual Inversion은 각 개념에 대해 새로운 단어 임베딩 토큰만을 최적화한다.

FreeCustom은 어텐션 레이어에서 참조 이미지와 생성 이미지의 키·값 쌍을 연결(concatenation)하여 참조 특성을 주입하고, 어텐션 스코어 편집을 통해 참조 피사체의 시각적 특성을 강조한다.

### 4-2. FreerCustom (FreeCustom의 확장판)

FreerCustom(IJCV 2025)은 단일 객체와 다중 객체 커스터마이징을 모두 포함하는 커스터마이즈 이미지 및 **비디오 생성** 문제를 다루며, 기존 방법들이 단일 객체 커스터마이징에 주로 집중하고 더 복잡한 멀티 객체 시나리오에서 어려움을 겪는다는 한계를 극복한다.

FreerCustom은 MRSA 메커니즘과 가중치 마스크 전략을 결합하여 모델이 생성된 이미지에서 원하는 개념을 더 잘 포착하고 표현할 수 있도록 한다.

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5-1. 향후 연구에 미치는 영향

#### ① 튜닝-프리 패러다임 확산
FreeCustom이 제시한 방법은 방대한 재학습이나 파인튜닝 없이 여러 개념을 포함한 커스터마이즈 이미지를 생성하는 유망한 접근법을 제공한다. 다중 참조 어텐션 및 가중치 마스크와 같은 새로운 메커니즘을 도입함으로써, 사전학습된 T2I 모델을 커스터마이즈 이미지 생성에 더 간단하고 효율적으로 활용하는 방법을 보여준다. 이 연구는 디지털 콘텐츠 창작, 개인화 마케팅, 교육 자원 등의 분야에서 잠재적 응용이 가능한 텍스트-이미지 생성의 접근성과 유연성을 향상시키는 광범위한 노력에 기여한다.

#### ② 주의(Attention) 메커니즘 기반 개념 제어 연구 촉진
FreeCustom은 자기-어텐션 레이어에 참조 특성을 직접 주입하는 다중 참조 자기-어텐션 메커니즘을 도입하여 학습-없는 다중 피사체 생성을 가능하게 하며, 이어서 등장하는 Diptych Prompting은 인페인팅 기반 T2I 모델이 제로샷 피사체 중심 생성을 수행할 수 있음을 보여준다.

#### ③ 비디오 생성으로의 확장
이후 연구들은 FreeCustom의 아이디어를 비디오 생성으로 확장하며, 단일·다중 객체 커스터마이징 모두를 포괄하는 방향으로 발전하고 있다.

---

### 5-2. 앞으로 연구 시 고려할 점

#### ① 개념 충돌 및 속성 누출(Attribute Leakage) 해결
현재 방법들은 적은 수의 샘플로 학습할 때 과적합이 발생하거나, 클래스가 유사한 피사체(예: 특정 두 마리의 개)에 대한 속성 누출 문제에 어려움을 겪고 있다.

이를 해결하기 위해 향후 연구에서는:
- 개념별로 **격리된 어텐션 영역(Localized Attention Region)** 기법 개발
- **마스크 정밀도** 향상을 통한 속성 분리 강화

#### ② 더 많은 개념(3개 이상) 처리
현재 구현에서는 2~3개의 이미지를 선택하도록 권장되고 있어, **4개 이상의 다중 개념** 합성 시 확장성(scalability) 검증이 필요하다. 향후 연구에서는 N개의 임의 개념을 안정적으로 처리하는 메커니즘 개발이 요구된다.

#### ③ 기반 모델(Base Model) 다양화
FreeCustom은 다른 확산 기반 모델에도 쉽게 적용할 수 있다고 주장하지만, **SDXL, DiT 기반 모델(Flux, SD3), 자기회귀 모델(GPT-4V 계열)** 등 최신 아키텍처로의 확장 가능성에 대한 체계적인 평가가 필요하다.

#### ④ 윤리 및 안전성 고려
이 분야가 계속 발전함에 따라, 이러한 생성 기술과 관련된 윤리적 고려 사항과 견고성 문제를 해결하는 것이 중요할 것이다.

특히 다음 사항이 고려되어야 한다:
- 딥페이크(deepfake) 생성에 악용 방지 메커니즘
- 저작권 있는 콘텐츠의 참조 이미지 사용 제한
- 편향된 생성 결과에 대한 fairness 평가

#### ⑤ 평가 지표(Evaluation Metric) 고도화
현재 사용 중인 DINOv2, CLIP-I, CLIP-T 등의 지표는 인간의 지각적 품질을 완전히 반영하지 못할 수 있다. 향후 연구에서는 **다중 개념 합성에 특화된 벤치마크**와 평가 지표의 표준화가 필요하다.

---

## 참고 자료 및 출처

| 번호 | 자료 | 링크/정보 |
|---|---|---|
| 1 | **FreeCustom 공식 arXiv 논문** | https://arxiv.org/abs/2405.13870 |
| 2 | **FreeCustom CVPR 2024 공식 발표** | https://openaccess.thecvf.com/content/CVPR2024/html/Ding_FreeCustom_Tuning-Free_Customized_Image_Generation_for_Multi-Concept_Composition_CVPR_2024_paper.html |
| 3 | **FreeCustom 공식 프로젝트 페이지** | https://aim-uofa.github.io/FreeCustom/ |
| 4 | **FreeCustom 공식 GitHub** | https://github.com/aim-uofa/FreeCustom |
| 5 | **FreeCustom IEEE Xplore** | https://ieeexplore.ieee.org/document/10656576/ |
| 6 | **FreeCustom 상세 리뷰 (themoonlight.io)** | https://www.themoonlight.io/en/review/freecustom-tuning-free-customized-image-generation-for-multi-concept-composition |
| 7 | **FreeCustom 빠른 리뷰 (liner.com)** | https://liner.com/review/freecustom-tuningfree-customized-image-generation-for-multiconcept-composition |
| 8 | **FreerCustom (확장판, IJCV 2025)** | https://link.springer.com/article/10.1007/s11263-025-02623-z |
| 9 | **Custom Diffusion (CVPR 2023)** | https://www.cs.cmu.edu/~custom-diffusion/ |
| 10 | **DreamBooth** | https://dreambooth.github.io/ |
| 11 | **ResearchGate FreeCustom 관련 연구 비교** | https://www.researchgate.net/publication/384206724 |
| 12 | **aimodels.fyi 논문 상세 정보** | https://www.aimodels.fyi/papers/arxiv/freecustom-tuning-free-customized-image-generation-multi |

> ⚠️ **정확도 주의**: 본 답변에서 수식 중 가중치 마스크의 구체적인 분해 수식($\mathbf{M}_w = w_i \cdot \mathbf{M}_i + \ldots$)은 논문의 공개 설명과 GitHub/프로젝트 페이지를 바탕으로 재구성한 것으로, 논문 내부의 정확한 표기와 다소 차이가 있을 수 있습니다. 완전한 수식 확인은 [CVPR 2024 공식 논문 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ding_FreeCustom_Tuning-Free_Customized_Image_Generation_for_Multi-Concept_Composition_CVPR_2024_paper.pdf)를 직접 참조하시길 권장합니다.
