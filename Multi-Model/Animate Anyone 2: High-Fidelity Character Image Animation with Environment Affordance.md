
# Animate Anyone 2: High-Fidelity Character Image Animation with Environment Affordance

> **논문 정보**
> - **제목**: Animate Anyone 2: High-Fidelity Character Image Animation with Environment Affordance
> - **저자**: Li Hu, Guangyuan Wang, Zhen Shen, Xin Gao, Dechao Meng, Lian Zhuo, Peng Zhang, Bang Zhang, Liefeng Bo
> - **arXiv**: [2502.06145](https://arxiv.org/abs/2502.06145) (2025년 2월 10일)
> - **학회**: ICCV 2025 (Open Access)
> - **프로젝트 페이지**: https://humanaigc.github.io/animate-anyone-2/

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

최근 Animate Anyone과 같은 diffusion 모델 기반 캐릭터 이미지 애니메이션 방법들은 일관성 있고 일반화 가능한 캐릭터 애니메이션 생성에 있어 상당한 진보를 이루었습니다. 그러나 이러한 접근 방식들은 **캐릭터와 환경 사이의 합리적인 연관성을 생성하는 데 실패**합니다. 이 한계를 극복하기 위해, Animate Anyone 2는 환경 어포던스(environment affordance)를 갖춘 캐릭터 애니메이션을 목표로 제안되었습니다.

### 1.2 주요 기여 (Key Contributions)

본 논문은 다음과 같은 새로운 환경 공식화(environment formulation) 및 객체 주입(object injection) 전략을 제안하여 캐릭터-환경의 매끄러운 통합을 가능하게 하며, **Pose Modulation**을 통해 모델이 다양한 모션 패턴을 견고하게 처리할 수 있도록 합니다.

구체적인 주요 기여는 다음 세 가지입니다:

| 기여 | 설명 |
|------|------|
| ① Shape-Agnostic Mask 전략 | 캐릭터-환경 경계 표현 및 shape leakage 완화 |
| ② Object Guider + Spatial Blending | 객체 상호작용 충실도 강화 |
| ③ Depth-wise Pose Modulation | 다양한 포즈에 대한 견고한 모션 모델링 |

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

**기존 방법의 문제점:**

Animate Anyone은 골격(skeleton) 표현을 통해 캐릭터의 모션을 포착하고, Pose Guider를 이용해 특징을 모델링합니다. 그러나 이 골격 표현은 팔다리 간 공간적 관계와 계층적 의존성에 대한 명시적 모델링이 부족합니다. 일부 기존 방법들은 SMPL과 같은 3D 메시 표현을 채택하지만, 이는 캐릭터 간 일반화 성능을 저하시키고 밀도 있는 표현으로 인해 **shape leakage**를 유발할 수 있습니다.

이러한 기존 방법들은 캐릭터와 환경 사이의 합리적인 연관성을 생성하지 못하며, 이를 해결하기 위해 환경 어포던스를 갖춘 캐릭터 애니메이션을 목표로 합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### ① 환경 공식화 (Environment Formulation)

소스 비디오에서 모션 신호를 추출하는 것 외에, 추가적으로 환경 표현을 조건부 입력으로 포착합니다. 환경은 **캐릭터를 제외한 영역**으로 공식화되며, 모델은 이 영역에 캐릭터를 생성하면서 환경 맥락과의 일관성을 유지합니다.

수식으로 표현하면, 생성 목표는 다음과 같이 정의됩니다:

$$
\hat{x} = \mathcal{G}(x_{ref}, p_{motion}, e_{env}, o_{obj})
$$

여기서:
- $x_{ref}$: 레퍼런스 캐릭터 이미지
- $p_{motion}$: 소스 비디오에서 추출한 모션 신호 (포즈 시퀀스)
- $e_{env}$: 환경 표현 (캐릭터 제외 영역)
- $o_{obj}$: 상호작용 객체 특징
- $\mathcal{G}$: 생성 네트워크 (diffusion 기반)

#### ② Shape-Agnostic Mask 전략

형태에 무관한(shape-agnostic) 마스크 전략을 제안하여 캐릭터와 환경 간의 관계를 보다 효과적으로 표현합니다. 이 전략은 캐릭터-맥락 통합을 위한 효과적인 학습을 가능하게 하면서 **shape leakage** 문제를 완화합니다.

마스크 연산은 다음과 같이 정의됩니다:

$$
M_{sa}(x) = \mathbb{1}[x \notin R_{char}]
$$

$$
E_{env} = I \odot (1 - M_{char})
$$

여기서 $M_{char}$는 캐릭터 영역 마스크, $E_{env}$는 캐릭터를 제거한 환경 이미지, $\odot$는 element-wise 곱셈입니다.

#### ③ Object Guider & Spatial Blending

경량의 Object Guider를 설계하여 상호작용 객체의 특징을 추출하고, Spatial Blending 메커니즘을 통해 이 특징들을 생성 과정에 주입합니다. 이는 소스 비디오에서 복잡한 상호작용 역학을 보존하는 데 기여합니다.

Spatial Blending 후의 새로운 노이즈 잠재 벡터 $z_{blend}$를 생성합니다. DenoisingNet 디코더의 각 단계에서 캐릭터 특징에 대한 공간적 어텐션(spatial attention)과 객체 특징의 Spatial Blending을 교대로 적용하여, 캐릭터-객체 상호작용의 세밀한 디테일을 갖춘 고충실도 결과를 생성합니다.

수식으로 표현하면:

$$
z_{blend} = z_{char} \odot (1 - M_{obj}) + F_{obj}(z_{obj}) \odot M_{obj}
$$

여기서:
- $z_{char}$: 캐릭터 특징 잠재 벡터
- $z_{obj}$: 상호작용 객체 특징
- $M_{obj}$: 객체 영역 마스크
- $F_{obj}$: Object Guider 추출 함수

#### ④ Depth-wise Pose Modulation

구조적 깊이(structured depth) 정보로 골격 표현을 보강하여 팔다리 간 공간적 관계 표현을 향상시키는 **Depth-wise Pose Modulation**을 제안합니다. 모션 신호로는 Sapiens를 활용해 소스 비디오에서 골격 및 깊이 정보를 추출하며, 깊이 정보는 shape leakage를 완화하기 위해 골격을 통해 구조적으로 처리됩니다.

먼저 골격 이미지를 이진화하여 골격 마스크를 얻고, 이 마스크 내 영역에서 깊이 결과를 추출합니다. 그런 다음 Pose Guider와 동일한 아키텍처의 Conv2D를 사용해 골격 맵과 구조화된 깊이 맵을 처리하고, Cross-Attention 메커니즘을 통해 구조화된 깊이 정보를 골격 특징에 통합합니다. 이 방식의 핵심은 각 팔다리가 다른 팔다리의 공간적 특성을 통합할 수 있도록 하여 더 세밀한 이해를 촉진합니다.

수식으로 표현하면:

$$
F_{pose} = \text{CrossAttn}(F_{skel}, F_{depth\_struct})
$$

$$
F_{depth\_struct} = \text{Conv2D}(D \odot M_{skel})
$$

여기서:
- $F_{skel}$: 골격 특징 (query)
- $F_{depth\_struct}$: 구조화된 깊이 특징 (key, value)
- $D$: 원본 깊이 맵
- $M_{skel}$: 이진화된 골격 마스크

전체 Latent Diffusion 목적 함수는 기존 LDM을 따릅니다:

$$
\mathcal{L}_{LDM} = \mathbb{E}_{z, e, p, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c_{ref}, c_{pose}, c_{env}, c_{obj})\|^2_2\right]
$$

여기서 $c_{ref}$, $c_{pose}$, $c_{env}$, $c_{obj}$는 각각 레퍼런스 캐릭터, 포즈, 환경, 객체 조건입니다.

---

### 2.3 모델 구조 (Architecture)

프레임워크는 3D UNet을 갖춘 self-supervised diffusion 기반 모델을 통해 비디오에서 모션 및 환경 신호를 추출합니다. Shape-Agnostic Mask, 경량 Object Guider + Spatial Blending, Depth-wise Pose Modulation을 통해 캐릭터와 배경 특징을 결합합니다.

```
[입력]
  ┌─────────────────┐
  │ Reference Image │ ──── ReferenceNet (Spatial Attention)
  └─────────────────┘
  
  ┌─────────────────┐
  │  Source Video   │ ──┬── Skeleton + Depth (Sapiens)
  └─────────────────┘   │         ↓ Depth-wise Pose Modulation
                        │   Conv2D → Cross-Attention → F_pose
                        │
                        ├── Environment Region (E_env = I ⊙ (1-M_char))
                        │         ↓ Shape-Agnostic Mask Strategy
                        │
                        └── Object Region (interactive objects)
                                  ↓ Object Guider
                                  F_obj → Spatial Blending

[DenoisingNet (3D UNet)]
  → Spatial-Attention (Character features)
  → Spatial Blending (Object features)
  → Cross-Attention (Env conditioning)
  → Temporal-Attention
  → VAE Decoder → Output Video
```

생성 결과는 캐릭터 일관성 유지뿐만 아니라, 캐릭터와 주변 환경을 매끄럽게 통합하는 고충실도 결과를 생성합니다.

---

### 2.4 성능 향상

PSNR, SSIM, LPIPS와 같은 정량적 지표와 포괄적인 정성적 분석을 통해, Animate Anyone 2가 기존 방법보다 사실적인 환경 어포던스를 갖춘 고충실도 애니메이션 생성에서 우월한 성능을 입증합니다. 이 논문은 캐릭터 모션과 환경 상호작용을 매끄럽게 처리함으로써 캐릭터 애니메이션 기술의 중요한 발전을 이루었습니다.

비교 대상 방법들과의 주요 차별점:

Animate Anyone 2는 기존 방법(Viggle 등)에 비해 더 높은 충실도와 장면 내 캐릭터의 매끄러운 통합을 나타냅니다. MIMO는 깊이 기반으로 인간, 배경, 폐색 구성요소를 분해하여 캐릭터 비디오를 생성하지만, Animate Anyone 2는 특히 다양한 모션이 있는 복잡한 장면에서 더 우월한 견고성과 세밀한 디테일 보존을 보여줍니다.

---

### 2.5 한계

향후 연구 과제로, 처리 속도 향상과 더 복잡한 환경 상호작용 처리에 집중할 필요가 있습니다.

추가적으로 논문에서 암묵적으로 드러나는 한계는 다음과 같습니다:

1. **Shape Leakage 위험**: 마스크 기반 환경 분리 방식은 여전히 정교한 세그멘테이션에 의존하며, 복잡한 경계에서 shape leakage가 발생할 수 있음
2. **코드 비공개**: 코드 및 모델 공개 계획이 불명확함
3. **추론 비용**: Spatial Blending을 디코더의 각 단계마다 적용하므로 추론 시 연산 비용 증가 가능성
4. **단일 캐릭터 한정**: 다중 캐릭터 상호작용 시나리오에 대한 일반화 미검증

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Shape-Agnostic Mask의 일반화 기여

Shape-Agnostic Mask 전략은 캐릭터와 맥락 장면 간의 경계 관계를 더 잘 표현하여, 캐릭터-맥락 통합을 위한 효과적인 학습을 가능하게 하는 동시에 **shape leakage 문제를 완화**합니다.

이는 특정 신체 형태에 과적합되지 않도록 하여 다양한 체형의 캐릭터에 대한 일반화를 가능하게 합니다.

### 3.2 Depth-wise Pose Modulation의 일반화 기여

깊이 기반 포즈 변조 전략은 골격 표현과 연관된 구조적 깊이 정보를 통합함으로써 팔다리 간 공간적 관계 표현을 강화합니다. 이를 통해 모델은 다양한 포즈와 상호작용에 걸쳐 견고한 성능을 유지할 수 있으며, 깊이 정보가 Cross-Attention 메커니즘을 통해 처리되어 팔다리 상호작용이 맥락적으로 관련성을 갖도록 보장합니다.

포즈 변조 전략은 모델이 광범위한 동작을 능숙하게 처리할 수 있도록 하여 **애니메이션의 일반화 가능성 및 신뢰성을 향상**시킵니다.

### 3.3 Zero-shot 일반화 능력

환경 어포던스 모듈은 시스템이 **추가 훈련 없이(zero-shot)** 새로운 캐릭터와 환경에서 작동할 수 있는 능력을 제공합니다.

### 3.4 기존 방법과의 일반화 비교

기존 3D 파라메트릭 모델 기반 방법의 일반화 한계와 비교:

SMPL과 같은 3D 메시 표현을 채택하는 방법들은 캐릭터 간 일반화 성능을 저하시키고, 밀도 있는 표현으로 인해 잠재적으로 shape leakage를 유발합니다.

반면, Animate Anyone 2의 Depth-wise Pose Modulation은 2D 골격의 일반화 장점을 유지하면서 3D 공간 정보를 보완하는 중간 방식을 채택합니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 관련 연구 계보

최근 발전은 주로 diffusion 기반 프레임워크를 채택하여 외관 일관성, 모션 안정성, 캐릭터 일반화 가능성에서 주목할 만한 향상을 달성하고 있습니다.

| 연구 | 연도 | 핵심 기여 | 한계 |
|------|------|-----------|------|
| **Animate Anyone (v1)** | 2023 | ReferenceNet + Pose Guider, 공간 어텐션 기반 외관 일관성 | 환경-캐릭터 연관성 부재 |
| **MagicAnimate** | 2023/2024 | Temporal Consistency + DensePose | 긴 비디오 일관성 부족 |
| **Champ** | 2024 | 3D 파라메트릭 가이던스 (SMPL) | Shape leakage, 일반화 저하 |
| **MimicMotion** | 2024/2025 | Confidence-aware Pose Guidance, 임의 길이 비디오 | 3D 공간 정보 부족 |
| **Animate-X** | 2024/2025 | 범용 캐릭터 (의인화 포함), Pose Indicator | 환경 인식 없음 |
| **Animate Anyone 2** | 2025 | 환경 어포던스 + Object Injection + Depth-wise Pose | 다중 캐릭터 미지원 |

MimicMotion은 Confidence-aware Pose Guidance를 도입하여 높은 프레임 품질과 시간적 부드러움을 확보하고, 포즈 신뢰도 기반 지역적 손실 증폭을 통해 핵심 영역의 이미지 왜곡을 줄이며, Progressive Latent Fusion 전략을 통해 길고 부드러운 비디오를 생성합니다.

Animate-X는 의인화 캐릭터를 포함한 다양한 캐릭터 유형을 위한 범용 LDM 기반 애니메이션 프레임워크를 제안합니다. 모션 표현 강화를 위해 Pose Indicator를 도입하며, CLIP 시각적 특징을 활용한 암묵적 방식과 명시적 방식을 통해 드라이빙 비디오에서 모션 패턴을 포착하고, LDM의 일반화 성능을 강화합니다.

그러나 현재까지 어떤 프레임워크도 모션, 표정, 환경 상호작용 제어를 통합하는 고충실도 캐릭터 애니메이션에 대한 전체적인 해결책을 제공하지 못한다는 중요한 격차가 여전히 존재합니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

**① 환경 인식 애니메이션의 새로운 패러다임 제시**

캐릭터 환경을 존중하는 사실적인 캐릭터 애니메이션 생성 능력은 게임, 영화 제작, 가상현실 응용 등 새로운 가능성을 열어줍니다. 캐릭터 애니메이션에서 환경 인식의 통합은 보다 사실적이고 몰입적인 디지털 경험을 향한 중요한 발전을 의미합니다.

**② 캐릭터-환경 상호작용 연구의 기준점**

환경 표현, 객체 상호작용, 견고한 포즈 처리 전략의 혁신적 통합을 통해 복잡한 비디오 환경 내에서 애니메이션된 캐릭터의 충실도와 적용 가능성에 대한 새로운 기준을 제시합니다.

**③ 후속 연구에서의 기술적 참조**

Animate Anyone 2의 접근 방식(마스킹 메커니즘을 이용한 배경 환경과 상호작용 객체의 구분)은 후속 연구들에서도 참조되고 있으며, 생성된 캐릭터가 환경과 높은 호환성을 보이는 결과를 낳습니다. 이러한 방법들은 **비디오 캐릭터 교체** 작업에도 적용 가능한 것으로 분석되고 있습니다.

### 5.2 향후 연구 시 고려할 점

**① DiT 아키텍처로의 확장 필요성**

DiT(Diffusion Transformer) 기반 캐릭터 이미지 애니메이션은 현재 상당한 연구 관심을 받고 있으며, 사전 훈련된 비디오 기반 모델의 능력을 활용함으로써 생성된 캐릭터 비디오의 현실감과 시간적 일관성이 실질적으로 향상되고 있습니다. 현재 Animate Anyone 2는 UNet 기반 diffusion 프레임워크를 채택하고 있어, DiT 기반 아키텍처로의 전환을 통한 성능 향상 가능성이 있습니다.

**② 다중 캐릭터 및 복잡한 장면 처리**

기존 캐릭터 애니메이션 알고리즘들은 여전히 일반화와 시각적 품질을 제한하는 두 가지 주요 과제에 직면해 있으며, 포즈 표현을 위해 구현된 모델이 여전히 중요한 병목으로 남아 있습니다.

**③ 3D 일관성 강화**

많은 기존 연구들이 OpenPose, DensePose, DWPose와 같은 2D 포즈 추정 모델을 채택하지만, 이러한 2D 표현은 충분한 공간 구조나 깊이 정보가 부족하여 특히 복잡한 포즈에서 애니메이션된 인체의 3D 일관성을 유지하기 어렵습니다.

**④ 실시간 추론 효율성**

처리 속도 향상과 더욱 복잡한 환경 상호작용 처리에 대한 추가 연구가 필요합니다.

**⑤ 데이터 및 윤리적 고려사항**

생성형 AI를 이용한 캐릭터 교체 및 딥페이크 위험성 연구(예: DORMANT 등)와 병행하여, 윤리적 사용 지침 및 검출 기술 개발도 함께 고려해야 합니다.

---

## 참고 자료

1. **[주 논문]** Li Hu et al., "Animate Anyone 2: High-Fidelity Character Image Animation with Environment Affordance," arXiv:2502.06145, 2025. — https://arxiv.org/abs/2502.06145
2. **[프로젝트 페이지]** Animate Anyone 2 공식 페이지 — https://humanaigc.github.io/animate-anyone-2/
3. **[ICCV 2025 Open Access]** Animate Anyone 2, ICCV 2025 — https://openaccess.thecvf.com/content/ICCV2025/papers/Hu_Animate_Anyone_2_High-Fidelity_Character_Image_Animation_with_Environment_Affordance_ICCV_2025_paper.pdf
4. **[전작]** Li Hu et al., "Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation," CVPR 2024, arXiv:2311.17117 — https://arxiv.org/abs/2311.17117
5. **[비교 연구]** Yuang Zhang et al., "MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance," ICML 2025 — https://arxiv.org/abs/2406.19680
6. **[비교 연구]** Shenhao Zhu et al., "Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance," ECCV 2024
7. **[비교 연구]** Tan et al., "Animate-X: Universal Character Image Animation with Enhanced Motion Representation," ICLR 2025 — https://openreview.net/forum?id=1IuwdOI4Zb
8. **[후속 연구]** "Wan-Animate: Unified Character Animation," arXiv:2509.14055, 2025 — https://arxiv.org/abs/2509.14055
9. **[후속 연구]** "MVAnimate: Enhancing Character Animation with Multi-View Optimization," arXiv:2602.08753 — https://arxiv.org/abs/2602.08753
10. **[리뷰]** Literature Review on Animate Anyone 2 — https://www.themoonlight.io/en/review/animate-anyone-2-high-fidelity-character-image-animation-with-environment-affordance
11. **[HuggingFace]** Paper page — https://huggingface.co/papers/2502.06145

> ⚠️ **정확도 주의사항**: 본 답변에서 수식 표현 일부(Spatial Blending, Environment Formulation 등)는 논문의 개념 설명을 기반으로 논문 본문 전체를 직접 열람할 수 없는 한계로 인해 저자의 표기법을 추론하여 작성된 부분이 있습니다. 정확한 수식 검증을 위해서는 ICCV 2025 공식 논문 PDF를 직접 참조하시기 바랍니다.
