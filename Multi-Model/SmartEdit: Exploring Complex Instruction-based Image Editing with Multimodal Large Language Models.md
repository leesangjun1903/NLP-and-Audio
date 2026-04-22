
# SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal Large Language Models
> **논문 정보:**
> - **제목:** SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal Large Language Models
> - **저자:** Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, Ying Shan
> - **학회:** CVPR 2024 (Highlight)
> - **arXiv:** [arXiv:2312.06739](https://arxiv.org/abs/2312.06739)
> - **공식 코드:** [GitHub - TencentARC/SmartEdit](https://github.com/TencentARC/SmartEdit)
> - **프로젝트 페이지:** [yuzhou914.github.io/SmartEdit](https://yuzhou914.github.io/SmartEdit/)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

기존의 지시 기반 이미지 편집 방법(예: InstructPix2Pix)은 Diffusion Model 내의 단순한 CLIP 텍스트 인코더에 의존하기 때문에 복잡한 시나리오에서 만족스러운 결과를 생성하지 못하는 한계가 있다. 이를 해결하기 위해 본 논문은 MLLM(Multimodal Large Language Model)을 활용하여 이해 및 추론 능력을 강화한 새로운 지시 기반 이미지 편집 접근법인 **SmartEdit**을 제안한다.

### 📌 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **BIM 모듈 제안** | 입력 이미지와 MLLM 출력 간 양방향 정보 교환 |
| **데이터 전략** | 지각 데이터 + 소량의 복잡 편집 데이터 활용 |
| **Reason-Edit 데이터셋** | 복잡한 추론 기반 이미지 편집 평가 벤치마크 신규 구축 |
| **성능 우위** | 정량/정성 평가 모두에서 기존 SOTA 대비 우월한 성능 달성 |

이 논문은 복잡한 지시 시나리오에서의 이미지 편집 성능에 집중하며, 이러한 복잡한 시나리오들은 과거 연구에서 종종 간과되고 탐구가 부족했던 영역이다. SmartEdit은 MLLM을 활용하여 지시를 더 잘 이해하고, BIM을 통해 텍스트와 이미지 특징 간의 상호작용을 개선하며, 복잡한 시나리오에서의 성능을 향상시키는 새로운 데이터 활용 전략을 제안한다.

---

## 2. 문제 정의, 제안 방법(수식 포함), 모델 구조, 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

#### (1) CLIP 텍스트 인코더의 표현력 한계

CLIP 텍스트 인코더를 Diffusion Model에서 사용하는 기존 방법은 복잡한 추론이 필요한 시나리오에서 지시의 의미를 이해하는 데 어려움을 겪는다. 반면, MLLM은 강력한 추론 능력과 세계 지식을 완전히 활용하여 올바른 객체를 식별할 수 있다.

#### (2) UNet의 지각 능력 부족

첫 번째 과제는 SmartEdit이 위치(position)와 개념(concept)에 대한 지각 능력이 부족하다는 것이고, 두 번째 과제는 MLLM이 장착되어 있음에도 불구하고 추론이 필요한 시나리오에서 SmartEdit의 능력이 여전히 제한적이라는 것이다. 요약하면, SmartEdit은 기존의 편집 데이터셋만으로 학습할 경우 복잡한 시나리오를 처리하는 효과가 제한적이다.

#### (3) 복잡 편집 데이터의 부재

기존 지시 기반 편집 방법의 실패에 기여하는 두 번째 이유는 특수한 데이터의 부재이다. InstructPix2Pix와 MagicBrush 등의 편집 데이터셋만으로 학습할 경우, SmartEdit도 복잡한 추론과 이해가 필요한 시나리오를 처리하는 데 어려움을 겪는다.

---

### 2-2. 제안 방법 및 수식

#### 📐 전체 파이프라인

지시(instruction)에 대해, 먼저 $r$개의 `[IMG]` 토큰을 지시 $c$의 끝에 추가한다. 이미지 $x$와 함께 LLaVA에 입력되어 해당 `[IMG]` 토큰에 대응하는 Hidden State를 얻는다. 이 Hidden State는 QFormer에 전달되어 특징 $f$를 얻는다. 이후, 이미지 인코더 $E_\phi$의 출력인 이미지 특징 $v$는 BIM(Bidirectional Interaction Module)을 통해 $f$와 상호작용하여 $f'$과 $v'$을 생성한다. $f'$과 $v'$은 Diffusion Model에 입력되어 지시 기반 이미지 편집 작업을 수행한다.

수식으로 정리하면:

$$
f = \text{QFormer}(\text{LLaVA}(x, c))
$$

$$
v = E_\phi(x)
$$

$$
(f', v') = \text{BIM}(f, v)
$$

$$
\hat{x} = \text{DiffusionModel}(x, f', v')
$$

#### 📐 BIM (Bidirectional Interaction Module) 상세

BIM 모듈에서 입력 정보 $f$와 $v$는 서로 다른 Cross-Attention을 통해 양방향 정보 교환을 수행한다. BIM 모듈은 이미지 특징을 재사용하여 UNet에 보조 정보로 입력한다. 이 모듈의 두 Cross-Attention 블록 구현은 이미지 특징과 텍스트 특징 사이의 강력한 양방향 정보 교환을 촉진한다.

BIM 내부의 양방향 Cross-Attention 수식:

$$
f' = \text{CrossAttn}_1(Q=f,\ K=v,\ V=v)
$$

$$
v' = \text{CrossAttn}_2(Q=v,\ K=f,\ V=f)
$$

BIM의 전체 구조는 Self-Attention, 두 개의 Cross-Attention 블록, MLP로 구성된다:

$$
\text{BIM}(f, v) \rightarrow (f', v')
$$

기존 설계에서는 이미지 특징이 Query 역할을 하고 MLLM 출력이 Key와 Value 역할을 하여, MLLM 출력이 일방향으로만 이미지 특징과 상호작용한다는 문제가 있었다. 이 문제를 완화하기 위해 BIM을 제안하였으며, 이 모듈은 LLaVA의 시각 인코더가 입력 이미지로부터 추출한 이미지 정보를 재사용하고, 이미지와 MLLM 출력 사이의 포괄적인 양방향 정보 교환을 가능하게 하여 복잡한 시나리오에서 더 나은 성능을 발휘하도록 한다.

---

### 2-3. 모델 구조

SmartEdit의 전체 구조는 다음과 같이 세 가지 핵심 구성 요소로 이루어진다:

```
[ Input Image x ] ──► [ LLaVA (7B/13B) ] ──► [ QFormer ] ──► f
       │                                                         │
       └──────────────► [ Image Encoder E_φ ] ──► v             │
                                                  │             │
                                                  └──► [ BIM ] ◄┘
                                                        │
                                              (f', v') │
                                                        ▼
                                              [ Diffusion UNet ]
                                                        │
                                                        ▼
                                               [ Edited Image x̂ ]
```

| 컴포넌트 | 역할 |
|---|---|
| **LLaVA (7B/13B)** | 지시 $c$와 이미지 $x$를 처리하여 `[IMG]` 토큰의 Hidden State 추출 |
| **QFormer** | Hidden State를 압축하여 고품질 텍스트-이미지 정렬 특징 $f$ 생성 |
| **Image Encoder $E_\phi$** | 이미지 $x$를 특징 벡터 $v$로 변환 |
| **BIM** | $f$와 $v$ 간 양방향 교환으로 $f'$, $v'$ 생성 |
| **Diffusion UNet** | $f'$, $v'$을 조건으로 이미지 편집 수행 |

SmartEdit은 시스템 메시지에서 이미지 생성용 특수 토큰 `"img"`와, 대화 시스템에서 이미지 및 텍스트 정보를 요약하기 위한 32개의 토큰(`<img_0>...<img_31>`)을 추가로 사용한다. 따라서 SmartEdit의 어휘 크기는 총 32,035이며, 이 32개의 새로운 토큰만이 QFormer의 유효한 임베딩으로 사용된다.

---

### 2-4. 학습 전략 (데이터 활용 전략)

학습 과정에서는 먼저 지각(perception) 데이터를 통합하여 Diffusion Model의 지각 및 이해 능력을 향상시킨다. 이후, 소량의 복잡한 지시 편집 데이터가 SmartEdit의 더 복잡한 지시에 대한 편집 능력을 효과적으로 자극할 수 있음을 입증한다.

복잡한 시나리오를 처리하는 SmartEdit의 능력을 향상시키는 데 두 가지 핵심 요소가 있다: 첫 번째는 UNet의 지각 능력을 강화하는 것이고, 두 번째는 소수의 고품질 예시를 통해 해당 시나리오에서 모델 역량을 자극하는 것이다.

학습 단계는 아래와 같이 구성된다:

| 학습 단계 | 데이터 | 목적 |
|---|---|---|
| **Stage 1: Alignment** | CC12M | LLaVA-QFormer 정렬 |
| **Stage 2: Perception** | 세그멘테이션 데이터 | UNet의 지각 능력 향상 |
| **Stage 3: Editing** | InstructPix2Pix, MagicBrush + 소량 복잡 편집 데이터 | 복잡한 시나리오 편집 능력 자극 |

---

### 2-5. 평가 지표 및 성능 향상

Reason-Edit에서의 정량적 비교는 PSNR↑, SSIM↑, LPIPS↓, CLIP Score↑(ViT-L/14), Ins-align↑의 다섯 가지 지표를 사용하며, 비교된 모든 방법은 SmartEdit과 동일한 학습 데이터로 파인튜닝된 것이다.

Ins-align 지표 결과에서 SmartEdit은 기존의 지시 기반 이미지 편집 방법들과 비교하여 **유의미한 성능 향상**을 보여준다.

SmartEdit은 기존의 지시 기반 이미지 편집 방법들 대비 유의미한 성능 향상을 달성하였으며, 더 강력한 MLLM 모델을 채택할 경우, SmartEdit-13B가 Ins-align 지표에서 SmartEdit-7B보다 더 나은 성능을 보인다.

SmartEdit은 복잡한 지시(예: "시간을 알려줄 수 있는 물체")에 기반한 객체 식별 및 편집 작업에서, 단순 CLIP 텍스트 인코더에 의존하는 방법들이 어려움을 겪는 시나리오에서도 성공적으로 해당 객체를 인식하고 편집하는 능력을 발휘한다.

Ablation Study에서 BIM을 제거하거나 단방향 상호작용만 사용하면 모든 지표에서 유의미한 성능 하락이 발생함을 확인하였으며, 이는 BIM이 이미지와 텍스트 특징 간의 강력한 정보 교환을 촉진하는 데 핵심적인 역할을 한다는 점을 강조한다.

---

### 2-6. 한계

SmartEdit의 한계로는 위치(position)와 개념(concept)에 대한 지각 능력 부족이 있으며, MLLM이 장착되어 있음에도 불구하고 추론이 필요한 시나리오에서 여전히 능력이 제한적이라는 점이 있다.

복잡한 편집 지시를 처리하기 위해 LLM(Large Language Model)과 DM(Diffusion Model)을 공동으로 파인튜닝해야 하는데, 이는 매우 높은 계산 복잡도와 학습 비용을 수반한다.

최근의 SmartEdit과 MGIE는 MLLM을 사용함에도 불구하고 복잡한 지시 이해 및 객체 정체성 보존(identity preservation) 측면에서 여전히 어려움을 겪는다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 소량 데이터의 효율적 활용

복잡한 시나리오와 유사한 대량의 페어 데이터를 생성하는 방법은 비용이 지나치게 높다. 이 시나리오에 대한 데이터 생성 비용이 매우 크기 때문이다.

학습 시 인식(perception) 데이터를 먼저 통합하여 Diffusion Model의 지각 및 이해 능력을 향상시키고, 이후 소량의 복잡한 지시 편집 데이터가 더 복잡한 지시에 대한 SmartEdit의 편집 능력을 효과적으로 자극할 수 있음을 보여준다.

이는 **데이터 효율적 일반화(data-efficient generalization)** 관점에서, 소수의 고품질 복잡 편집 예시만으로도 모델이 보지 못한 복잡 시나리오로 일반화될 수 있음을 시사한다.

### 3-2. 세그멘테이션 데이터를 통한 지각 능력 강화

세그멘테이션 데이터와 소량의 합성 복잡 편집 데이터의 전략적 활용은 모델의 지각 및 추론 능력 향상에 핵심적이며, 이를 통해 모델이 보지 못한 복잡 시나리오에 더 잘 일반화될 수 있도록 한다.

### 3-3. 더 강력한 MLLM 백본의 활용

추론이 필요한 시나리오에서 단순한 CLIP 텍스트 인코더는 지시의 의미를 이해하는 데 어려움을 겪는 반면, MLLM은 강력한 추론 능력과 세계 지식을 완전히 활용하여 올바른 객체를 식별할 수 있다.

따라서 더 강력한 MLLM(예: LLaMA-3 기반 모델, GPT-4V 수준)을 백본으로 채택할 경우, 일반화 성능의 추가 향상을 기대할 수 있다.

### 3-4. BIM의 구조적 범용성

기존 방법들이 SmartEdit과 동일한 학습 데이터를 파인튜닝에 사용하더라도, LLaVA와 BIM 모듈의 도입이 모델이 더 복잡한 지시를 이해할 수 있도록 하여 우수한 결과를 낳는다.

이는 BIM이 단순히 특정 데이터셋에 과적합되는 것이 아니라, 복잡한 지시에 대한 **구조적 일반화 능력**을 부여함을 의미한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 논문 | 연도 | 핵심 방법 | 주요 한계 |
|---|---|---|---|
| **InstructPix2Pix** (Brooks et al.) | 2023 | 대규모 데이터셋 파인튜닝, CLIP 인코더 | 복잡 추론 불가 |
| **MagicBrush** (Zhang et al.) | 2023 | 수동 주석 고품질 데이터 | 다중 턴 제한 |
| **InstructDiffusion** (Geng et al.) | 2024 | Vision task로의 일반화 | 복잡 지시 약함 |
| **MGIE** (Fu et al.) | 2024 | MLLM으로 표현력 있는 지시 파생 | 단방향 지시 가이드 |
| **SmartEdit** (Huang et al.) | 2024 | BIM + MLLM(LLaVA) + 인식 데이터 | 공동 파인튜닝 고비용 |
| **MCIE-E1** (2025) | 2025 | 지시 분해 + 공간 인식 Cross-Attention | 새로운 평가 벤치 필요 |
| **X-Planner** (2025) | 2025 | MLLM 기반 복합 지시 분해 + 지역 편집 | 아키텍처 복잡도 |

복잡한 편집 지시를 처리하는 한 가지 연구 방향은 MLLM을 Diffusion Model에 통합하여 세밀한 편집 지시의 이해를 향상시키는 것이며, SmartEdit(Huang et al., 2024)과 Fang et al.(2025)이 이 방향의 대표적 연구들이다.

이후 연구들은 SmartEdit(Huang et al., 2024), UltraEdit(Zhao et al., 2024), AnyEdit(Yu et al., 2025) 등을 포함하여 새로운 아키텍처 설계를 통해 다중 모달 상호작용과 지시 충실도(instruction fidelity)를 향상시키는 데 집중하고 있다.

최신 MCIE-E1은 새로운 벤치마크 CIE-Bench에서 지시 준수(instruction compliance)에서 23.96%의 향상을 달성하며 이전 SOTA 방법들을 능가하였다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

#### ✅ MLLM과 Diffusion Model의 통합 패러다임 확립
SmartEdit은 MLLM을 활용하여 지시 기반 편집 방법의 이해 및 추론 능력을 강화하는 지시 기반 이미지 편집 모델로서, MLLM과 생성 모델을 결합하는 새로운 패러다임을 제시하였다.

이후 MGIE, GoT, MCIE-E1 등 다수의 후속 연구가 이 패러다임을 계승·발전시키고 있으며, 이러한 발전들은 시각-언어 이해와 편집을 통합하는 **범용 MLLM으로의 전환**을 보여준다.

#### ✅ 복잡 추론 이미지 편집이라는 새로운 연구 방향 개척
Reason-Edit이라는 복잡한 지시 기반 이미지 편집을 위한 새로운 평가 데이터셋을 구축하였으며, 이 데이터셋에서의 정량적·정성적 결과는 SmartEdit이 이전 방법들을 능가함을 입증하여 복잡한 지시 기반 이미지 편집의 실용적 응용을 위한 길을 열었다.

#### ✅ 소량 데이터 학습 전략의 효과 입증
지각 관련 데이터와 소량의 복잡 편집 데이터가 모델 성능 향상에 핵심적임을 발견하였으며, 이 데이터 활용 전략은 이후 연구에서 데이터 효율적 학습의 중요한 참고 사례가 되었다.

---

### 5-2. 향후 연구 시 고려할 점

#### ⚠️ (1) 공동 파인튜닝 비용 문제 해결
기존 이미지 편집 방법들은 단순한 편집 지시는 잘 처리하나, 복잡한 편집 지시를 위해 LLM과 DM을 공동 파인튜닝해야 하며 이는 매우 높은 계산 복잡도와 학습 비용을 수반한다.

→ **LoRA, QLoRA, MoE**와 같은 경량화 파인튜닝 기법을 통한 효율적 공동 학습 방법 연구가 필요하다.

#### ⚠️ (2) 객체 정체성 보존(Identity Preservation) 개선
SmartEdit과 MGIE는 MLLM을 활용함에도 불구하고 복잡한 지시 이해와 **객체 정체성 보존** 측면에서 여전히 어려움을 겪는다.

→ 편집 영역 외의 배경 일관성 보존을 위한 마스크 기반 편집 및 지역 편집(localized editing) 메커니즘 강화가 필요하다.

#### ⚠️ (3) 평가 지표의 한계 극복
전통적 지표인 PSNR, SSIM, LPIPS, CLIP Score는 복잡한 편집 작업에서 인간의 시각적 지각 및 지시 정렬을 정확하게 반영하는 데 한계가 있다. Ins-align이라는 인간 평가 기반 지표의 도입이 더 신뢰할 수 있는 측정 방법을 제공한다.

→ GPT-4V, VIEScore 등 **다중 모달 LLM 기반의 자동 평가 지표** 개발이 요구된다.

#### ⚠️ (4) 다중 객체 및 다중 지시 처리
사용자가 복잡한 지시를 제공할 때, 기존 접근법들은 하위 지시를 간과하거나 의도하지 않은 변경을 도입하여 배경 일관성을 손상시키는 경우가 많다.

→ 복합 지시 분해(instruction decomposition) 및 단계별 편집 파이프라인 연구가 필요하다.

#### ⚠️ (5) 더 강력한 MLLM 백본 연구
MLLM 기반 다중 모달 추론과 Diffusion 기반 제어 가능 생성을 통합하는 연구 방향이 새로운 가능성을 열고 있으며, Step1X-Edit(Liu et al., 2025)과 Qwen-Image(Wu et al., 2025) 같은 모델들이 이 방향의 대표 연구들이다.

→ GPT-4o 수준의 MLLM을 백본으로 활용하거나 LLaVA-NeXT, InternVL 등 더 강력한 MLLM으로의 교체 실험이 의미 있는 연구 방향이다.

---

## 📚 참고 자료 (출처)

| # | 자료명 | 링크 |
|---|---|---|
| 1 | **[arXiv 원문]** SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal Large Language Models (arXiv:2312.06739) | https://arxiv.org/abs/2312.06739 |
| 2 | **[CVPR 2024 공식 논문]** SmartEdit - CVPR 2024 Open Access | https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_SmartEdit_... |
| 3 | **[프로젝트 페이지]** SmartEdit Official Page | https://yuzhou914.github.io/SmartEdit/ |
| 4 | **[공식 코드]** GitHub - TencentARC/SmartEdit (CVPR-2024 Highlight) | https://github.com/TencentARC/SmartEdit |
| 5 | **[Semantic Scholar]** SmartEdit 논문 정보 및 인용 | https://www.semanticscholar.org/paper/SmartEdit... |
| 6 | **[IEEE Xplore]** SmartEdit - IEEE CVPR 2024 출판본 | https://ieeexplore.ieee.org/document/10656752/ |
| 7 | **[관련 후속 연구]** MCIE: Multimodal LLM-Driven Complex Instruction Image Editing (arXiv:2602.07993) | https://arxiv.org/html/2602.07993 |
| 8 | **[관련 후속 연구]** X-Planner for Complex Instruction-Based Image Editing (arXiv:2507.05259) | https://arxiv.org/pdf/2507.05259 |
| 9 | **[Survey 논문]** Instruction-Guided Editing Controls for Images and Multimedia: A Survey in LLM era (arXiv:2411.09955) | https://arxiv.org/html/2411.09955v1 |
| 10 | **[관련 후속 연구]** MultiEdit: Advancing Instruction-based Image Editing on Diverse and Challenging Tasks (arXiv:2509.14638) | https://arxiv.org/html/2509.14638v1 |
| 11 | **[Quick Review]** Liner.com - SmartEdit 리뷰 | https://liner.com/review/smartedit-... |
