# MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation

---

# 1. 핵심 주장 및 주요 기여 요약

MultiDiffusion은 사전 학습된 text-to-image 확산 모델(diffusion model)을 추가 학습이나 파인튜닝 없이 활용하여, 다양하고 제어 가능한 이미지 생성을 가능하게 하는 통합 프레임워크입니다.

이 접근법의 핵심은, 공유된 파라미터 또는 제약 조건을 통해 여러 확산 생성 프로세스를 하나로 묶는 최적화 문제(optimization task) 기반의 새로운 생성 프로세스입니다.

MultiDiffusion은 원하는 종횡비(예: 파노라마), 세밀한 세그멘테이션 마스크부터 바운딩 박스까지 다양한 공간 안내 신호(spatial guiding signals)에 따르는 고품질·고다양성 이미지를 생성할 수 있습니다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **Training-free** | 기존 사전학습 모델을 그대로 사용하며, 재학습·파인튜닝 불필요 |
| **통합 프레임워크** | 파노라마 생성, 영역 기반(region-based) 제어 등 여러 제어 작업을 하나의 프로세스로 통합 |
| **Closed-form 해** | 최적화 문제에 대한 닫힌 형태의 해석적 풀이(closed-form solution) 제공 |
| **추가 연산 비용 미발생** | 추가적인 계산 오버헤드 없이 효율적으로 동작 |

---

# 2. 상세 분석

## 2.1 해결하고자 하는 문제

기존 확산 모델에서 사용자 제어 가능성(user controllability)과 새로운 작업에 대한 빠른 적응은 여전히 미해결 과제로, 주로 값비싸고 긴 재학습/파인튜닝 또는 특정 이미지 생성 작업에 대한 임시 방편(ad-hoc adaptation)으로 대처하고 있었습니다.

구체적으로:
1. **고정 해상도/종횡비 문제**: 사전학습 모델은 고정 크기(예: 512×512)에서만 학습되어, 파노라마 등 임의 크기 생성이 어려움
2. **공간 제어 부재**: 텍스트 프롬프트만으로는 "어디에 무엇을" 배치할지 세밀한 제어가 어려움
3. **기존 방법의 비용**: 이러한 작업들은 curated dataset에 대한 값비싼 대규모 학습을 요구

## 2.2 제안하는 방법 (수식 포함)

### (A) 기본 확산 프로세스 정의

표준 확산 모델의 역방향 생성 프로세스를 다음과 같이 정의합니다:

$$I_T, I_{T-1}, \dots, I_0 \quad \text{s.t.} \quad I_{t-1} = \Phi(I_t \mid y) $$

여기서 $\Phi$는 사전학습된 확산 모델의 단일 디노이징 단계(single denoising step), $y$는 텍스트 프롬프트입니다.

### (B) 다중 확산 경로 (Multiple Diffusion Paths)

$n$개의 서로 다른 확산 경로(diffusion path) $J^i$를 정의합니다:

$$J^i_T, J^i_{T-1}, \dots, J^i_0 \quad \text{s.t.} \quad J^i_{t-1} = \Psi_i(J^i_t \mid z_i) $$

각 경로 $i$는 자체 디노이저 $\Psi_i$와 조건 $z_i$를 가지며, 전체 이미지의 특정 영역(crop)에 대응합니다.

### (C) Fuse-Then-Denoise (FTD) 최적화

MultiDiffusion의 핵심은 **여러 독립적 확산 경로를 하나의 전역 이미지로 융합**하는 것입니다. 각 디노이징 스텝에서, 다음 최적화 문제를 풀어 전역 이미지 $I_{t-1}$을 결정합니다:

```math
I_{t-1} = \argmin_{J} \sum_{i=1}^{n} \left\| W_i \odot (F_i(J) - \Phi(F_i(I_t) \mid y_i)) \right\|^2
```

여기서:
- $F_i$: 전체 이미지에서 $i$번째 크롭(crop) 또는 영역을 추출하는 매핑 함수
- $W_i$: 각 영역에 대한 가중치 마스크 (Hadamard product에 사용)
- $\odot$: Hadamard product (원소별 곱)
- $\Phi(\cdot \mid y_i)$: 프롬프트 $y_i$ 조건하의 디노이징 스텝

이 FTD 비용 함수는 직관적으로, 각 디노이징 샘플링 방향(diffusion direction)을 최소 제곱(least-squares) 의미에서 조화(reconcile)시키는 역할을 합니다.

### (D) Closed-form Solution

위 최적화 문제는 **분석적 닫힌 형태의 해(closed-form solution)**를 가집니다. 이 LS(Least Squares) 문제의 해는 해석적으로 계산됩니다. 구체적으로, 겹치는 영역에 대해 가중 평균(weighted average)으로 귀결됩니다:

$$I_{t-1}(p) = \frac{\sum_{i=1}^{n} W_i(p) \cdot \Phi_i(p)}{\sum_{i=1}^{n} W_i(p)} $$

여기서 $p$는 픽셀 위치, $\Phi_i(p)$는 $i$번째 경로의 디노이징 결과 해당 픽셀 값입니다.

> **Proposition 3.1 (논문 명제)**: 겹치는 모든 영역 $i, j$에 대해 $F_i(I_t) = F_j(I_t)$이면 (즉, 일관된 전역 이미지를 공유하면), 위 수식 (4)의 해에서 FTD 비용 (수식 3)이 0으로 최소화됩니다.

## 2.3 모델 구조

MultiDiffusion은 **별도의 새 모델 아키텍처를 도입하지 않습니다.** 기존 사전학습 모델(예: Stable Diffusion)의 U-Net 구조를 그대로 활용합니다.

**생성 파이프라인:**

```
[전역 노이즈 이미지 I_T]
        │
   ┌────┴────┐
   │  각 영역에 대해 독립적으로 디노이징 수행  │
   │  Φ(F_1(I_t)|y_1), ..., Φ(F_n(I_t)|y_n) │
   └────┬────┘
        │
   [Closed-form Weighted Average로 융합]
        │
   [전역 이미지 I_{t-1} 생성]
        │
   (반복: T → 0)
        │
   [최종 이미지 I_0]
```

### 응용별 구성

1. **파노라마 생성**: 해상도 512×4608의 text-to-panorama 결과를 시연하며, 겹치는 윈도우(overlapping windows)로 전체 파노라마 영역을 커버합니다.

2. **영역 기반 생성 (Region-based Generation)**: 각 영역에 서로 다른 텍스트 프롬프트를 할당하여, 세그멘테이션 마스크 또는 바운딩 박스 기반의 제어된 이미지 생성을 수행합니다.

## 2.4 성능 향상

관련 베이스라인들과 비교하여, 해당 작업에 특화 학습된 방법들과 비교해도 최신 수준(state-of-the-art)의 제어 생성 품질을 달성하는 것으로 나타났습니다.

| 비교 항목 | MultiDiffusion | 기존 베이스라인(BLD, SI) |
|-----------|---------------|----------------------|
| **이음새(Seam)** | 없음 (seamless) | 눈에 보이는 이음새·불연속성 존재 |
| **반복 콘텐츠** | 다양한 콘텐츠 | BLD는 반복적 콘텐츠 생성 |
| **시각 품질** | 전역적으로 일관 | SI는 좌우 시각적 차이 발생 |
| **추가 학습** | 불필요 | 일부 방법은 추가 학습 필요 |

## 2.5 한계

논문에서 식별되거나 추론 가능한 한계점:

1. **미세 제어의 어려움**: 크롭(crop) 단위 독립 디노이징 후 가중 평균을 취하기 때문에, 매우 미세한 객체 배치나 복잡한 상호작용에 한계
2. **레이아웃 결정 시점**: 레이아웃이 확산 과정 초기에 결정되는 것으로 관찰되어, 후반 스텝에서의 레이아웃 수정이 어려움
3. **근사적 최적화**: 각 스텝의 독립적 최소제곱 최적화는 전체 생성 과정에 대한 전역 최적(global optimum)을 보장하지 않음
4. **텍스트-영역 대응의 제한**: 복잡한 장면에서 텍스트 프롬프트와 영역 간 미묘한 의미적 대응이 부족할 수 있음

---

# 3. 모델의 일반화 성능 향상 가능성

MultiDiffusion의 일반화 성능과 관련된 핵심 특성은 다음과 같습니다:

### 3.1 Training-free 특성에 의한 일반화

사전학습 모델을 추가 학습·파인튜닝 없이 사용하므로, 기저 확산 모델의 일반화 능력을 그대로 계승합니다. 이는:

- **모델 교체 유연성**: Stable Diffusion v1.5, v2, SDXL 등 어떤 사전학습 모델이든 적용 가능
- **도메인 독립성**: 특정 도메인에 과적합(overfit)되지 않음
- **향후 더 강력한 기저 모델이 등장할 때 자동으로 성능 향상** 기대

### 3.2 제어 신호의 유연한 일반화

MultiDiffusion은 원하는 종횡비, 또는 거친 영역 기반 텍스트 프롬프트(rough region-based text-prompts) 등 다양한 제어 신호를 통합할 수 있습니다.

- **Tight mask → Rough mask → Bounding box**: 다양한 수준의 공간 가이드를 단일 프레임워크에서 처리
- 거친(rough) 마스크로도 작동하며, 이는 초보 사용자도 직관적으로 제공할 수 있는 입력입니다.

### 3.3 확장성 (Scalability)

- **임의 해상도**: 윈도우 슬라이딩 방식으로 어떤 해상도·종횡비든 생성 가능
- **다중 프롬프트 조합**: 이론적으로 영역 수 $n$에 제한 없이 확장 가능
- **다른 모달리티로의 확장**: 비디오 생성 등으로 확장 가능 (아래 Lumiere 참조)

### 3.4 일반화 향상을 위한 연구 방향

| 방향 | 설명 |
|------|------|
| **적응적 가중치 스케줄링** | $W_i$를 타임스텝별로 동적 조정하여 초기-후기 생성 단계에 따른 최적 융합 |
| **의미 인식(Semantic-aware) 융합** | 단순 가중 평균 대신 의미적 일관성을 고려한 융합 메커니즘 |
| **3D/비디오 확장** | 공간 축 뿐 아니라 시간 축까지 다중 경로 융합으로 확장 |

---

# 4. 향후 연구에 미치는 영향 및 고려할 점

## 4.1 향후 연구에 미치는 영향

1. **Training-free 제어 패러다임의 확립**: MultiDiffusion은 "사전학습 모델을 건드리지 않고 추론 시 제어한다"는 연구 방향을 확립했으며, 이후 많은 training-free 제어 방법들의 기초가 됨

2. **비디오 생성으로의 확장**: Google의 Lumiere 프로젝트에서 MultiDiffusion을 활용하여 비디오 생성의 공간적 일관성을 확보하는 등, 비디오 합성 분야에 직접적 영향

3. **커뮤니티 도구화**: HuggingFace Diffusers 라이브러리에 StableDiffusionPanoramaPipeline으로 통합되어 실용적으로 널리 사용됨

## 4.2 향후 연구 시 고려할 점

1. **전역 일관성 강화**: 크롭 단위 독립 디노이징의 한계를 극복하기 위해, cross-view attention 메커니즘 등 전역 일관성 보장 방법 연구 필요
2. **계산 효율성과 품질의 트레이드오프**: 영역 수가 많아질수록 디노이징 호출 횟수가 선형 증가하므로, 효율적 배치 처리 전략 필요
3. **복잡한 장면 구성**: 객체 간 상호작용(occlusion, interaction)이 복잡한 장면에서의 성능 검증 필요
4. **3D 인식 확장**: 멀티뷰 일관성을 확보하기 위한 3D-aware 융합 메커니즘 연구

---

# 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문/방법 | 연도 | 접근 방식 | 학습 필요 여부 | MultiDiffusion 대비 특징 |
|-----------|------|-----------|-------------|----------------------|
| **DDPM** (Ho et al.) | 2020 | 디노이징 확산 확률 모델 기초 | ✅ 전체 학습 | MultiDiffusion의 기반이 되는 확산 모델 프레임워크 |
| **Stable Diffusion (LDM)** (Rombach et al.) | 2022 | 잠재 공간(latent space) 확산 | ✅ 전체 학습 | MultiDiffusion이 기저 모델로 활용 |
| **ControlNet** (Zhang et al.) | 2023 | 사전학습 모델을 잠그고, zero convolution으로 연결된 학습 가능 복사본으로 공간 제어 조건 추가 | ✅ 파인튜닝 | 엣지, 깊이, 포즈 등 다양한 조건 지원; 학습 필요하나 견고함 |
| **GLIGEN** (Li et al.) | 2023 | 확산 모델 attention layer에 새 파라미터 학습하여 grounded 생성 | ✅ 파인튜닝 | 바운딩 박스 기반 제어에 특화 |
| **SpaText** (Avrahami et al.) | 2023 | 세그멘테이션 마스크를 localized token embeddings으로 매핑하여 확산 모델 제어 | ✅ 전체 학습 | 높은 계산 비용, 코드 비공개 |
| **MVDiffusion** (Tang et al.) | 2023 | SD 모델의 다중 브랜치와 correspondence-aware attention(CAA)으로 다중 뷰 동시 생성 | ✅ CAA 학습 | 1만 장의 파노라마만으로 학습해도 고해상도 사실적 이미지 생성 가능; 멀티뷰 일관성이 더 우수 |
| **ControlNet-XS** (Zavadski et al.) | 2024 | 기저 모델 파라미터의 1%만으로 state-of-the-art 달성, ControlNet 대비 FID 성능 개선 | ✅ 파인튜닝 | 훨씬 경량화된 제어 네트워크 |
| **AMDM** (Yue et al.) | 2024 | 여러 확산 모델의 features를 지정 모델에 통합하여 fine-grained 제어 활성화하는 training-free 알고리즘 | ❌ Training-free | 학습 없이 fine-grained 제어를 유의미하게 개선 |
| **DC-ControlNet** | 2025 | Intra-Element Controller와 Inter-Element Controller로 요소 내·요소 간 조건을 분리 관리 | ✅ 학습 | 다중 조건 간 간섭 방지에 초점 |

### 핵심 비교 인사이트

**MultiDiffusion vs. ControlNet:**
- ControlNet은 사전학습 모델을 잠그고 그 심층 인코딩 레이어를 backbone으로 재활용하여 다양한 조건 제어를 학습하는 반면, MultiDiffusion은 학습 자체가 불필요
- ControlNet은 edge, depth, pose 등 구조적 조건에 강하고, MultiDiffusion은 영역별 텍스트 프롬프트와 해상도 변환에 강점

**MultiDiffusion vs. MVDiffusion:**
- MVDiffusion은 모든 이미지를 동시에 생성하며 전역 인식(global awareness)으로 오류 축적 문제를 효과적으로 해결
- MultiDiffusion은 training-free이지만, MVDiffusion은 cross-view attention 학습을 통해 더 강한 멀티뷰 일관성 확보

**MultiDiffusion vs. AMDM:**
- 둘 다 training-free이나, AMDM은 spherical aggregation과 manifold optimization이라는 두 가지 핵심 요소로, 중간 변수를 최소한의 manifold deviation으로 병합하여 더 정교한 fine-grained 제어 가능

---

# 참고자료

1. **[원 논문]** Bar-Tal, O., Yariv, L., Lipman, Y., & Dekel, T. (2023). "MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation." *Proceedings of Machine Learning Research (PMLR)*, Vol. 202, pp. 1737–1752. ICML 2023. — [arXiv:2302.08113](https://arxiv.org/abs/2302.08113)
2. **[프로젝트 페이지]** MultiDiffusion Project Page — https://multidiffusion.github.io/
3. **[HuggingFace Diffusers 문서]** StableDiffusionPanoramaPipeline — https://huggingface.co/docs/diffusers/en/api/pipelines/panorama
4. **[ControlNet 원 논문]** Zhang, L., Rao, A., & Agrawala, M. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." *ICCV 2023*. — [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)
5. **[MVDiffusion]** Tang, S. et al. (2023). "MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion." — [arXiv:2307.01097](https://arxiv.org/html/2307.01097v7)
6. **[AMDM]** Yue, C. et al. (2024). "Improving Fine-Grained Control via Aggregation of Multiple Diffusion Models." — [arXiv:2410.01262](https://arxiv.org/abs/2410.01262)
7. **[ControlNet-XS]** Zavadski, D., Feiden, J.-F., & Rother, C. (2024). "ControlNet-XS: Rethinking the Control of Text-to-Image Diffusion Models as Feedback-Control Systems." *ECCV 2024*. — https://vislearn.github.io/ControlNet-XS/
8. **[DC-ControlNet]** (2025). "DC-ControlNet: Decoupling Inter- and Intra-Element Conditions in Image Generation with Diffusion Models." — [arXiv:2502.14779](https://arxiv.org/html/2502.14779v1)
9. **[ACM DL]** MultiDiffusion on ACM Digital Library — https://dl.acm.org/doi/10.5555/3618408.3618482
10. **[Weizmann Institute Pure]** 출판 정보 — https://weizmann.elsevierpure.com/en/publications/multidiffusion-fusing-diffusion-paths-for-controlled-image-genera/
11. **[ResearchGate]** MultiDiffusion 전문 — https://www.researchgate.net/publication/368572849
12. **[Lumiere 분석 (Oxen.ai)]** ArXiv Dives - Lumiere — https://ghost.oxen.ai/arxiv-dives-lumiere/

---

> **정확도 관련 공지**: 위 수식 (특히 Eq. 3, 4)은 논문 원문의 핵심 아이디어를 충실히 반영한 것이나, 원문의 정확한 표기(notation)와 세부 인덱싱은 원 논문 PDF를 직접 확인하시기를 권장합니다. 논문에서 사용한 정확한 수식 번호 체계와 세부 증명은 ICML 2023 proceedings (PMLR Vol. 202, pp. 1737–1752)에서 확인하실 수 있습니다.
