
# TEXGen: a Generative Diffusion Model for Mesh Textures

> **논문 정보**
> - 저자: Xin Yu, Ze Yuan, Yuan-Chen Guo, Ying-Tian Liu, JianHui Liu, Yangguang Li, Yan-Pei Cao, Ding Liang, Xiaojuan Qi
> - 학술지: ACM Transactions on Graphics (TOG), Vol. 43, No. 6, Article 213, 2024
> - 발표: SIGGRAPH Asia 2024 (Journal Track, **Best Paper Honorable Mention**)
> - arXiv: [2411.14740](https://arxiv.org/abs/2411.14740)
> - 코드: [https://github.com/CVMI-Lab/TEXGen](https://github.com/CVMI-Lab/TEXGen)

---

## 1. 핵심 주장과 주요 기여 요약

이 연구는 3D 텍스처에 대한 테스트-타임 최적화(test-time optimization)를 위해 사전 학습된 2D 확산 모델(pre-trained 2D diffusion model)에 의존하는 기존의 관행에서 벗어나, UV 텍스처 공간(UV texture space) 자체에서 학습하는 근본적인 문제에 집중합니다.

고품질 텍스처 맵은 사실적인 3D 자산 렌더링에 필수적임에도 불구하고, 특히 대규모 데이터셋에서 텍스처 공간에서 직접 학습을 탐구한 연구는 거의 없었습니다.

### 주요 기여 4가지

| # | 기여 | 설명 |
|---|------|------|
| ① | **최초의 피드-포워드 대형 텍스처 확산 모델** | UV 공간에서 직접 고해상도 텍스처 생성 |
| ② | **하이브리드 네트워크 아키텍처** | UV 맵 컨볼루션과 포인트 클라우드 어텐션의 인터리빙 |
| ③ | **7억 파라미터 확산 모델 훈련** | 텍스트/단일 뷰 이미지 조건부 생성 |
| ④ | **다양한 확장 응용 지원** | 인페인팅, 스파스 뷰 완성, 텍스처 합성 |

최초로, 피드-포워드 방식으로 고해상도 텍스처 맵을 직접 생성할 수 있는 대형 확산 모델을 훈련시켰습니다.

이 논문은 SIGGRAPH Asia 2024 Journal Track에서 Best Paper Honorable Mention을 수상하였습니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

#### 기존 방법의 두 가지 한계

기존의 지배적인 방법은 사전 학습된 2D 확산 모델로 테스트-타임 최적화를 통해 3D 메시를 텍스처링하는 것이었습니다. Score Distillation Sampling(SDS) 등의 기법이 2D 확산 사전(prior)을 3D 형상에 distill하지만, 이 접근 방식은 **높은 연산 비용**과 **야누스 문제(Janus problem)**, **부자연스러운 색상** 같은 고유한 아티팩트라는 심각한 단점이 있습니다.

또 다른 방법은 기하학 조건부 이미지 생성과 인페인팅을 활용해 점진적으로 텍스처를 생성하는 것입니다. TEXTure 등은 하나의 원근 뷰에서 부분 텍스처 맵을 생성한 후, 인페인팅으로 다른 뷰를 완성하는 방식입니다.

UV 맵에서 물리적으로 연결되지 않은 단편(fragment)들은 표면에서 분리되어 있어, 기존 이미지 기반 모델에서 부정확한 특징 추출을 야기합니다. 이 문제를 해결하기 위해 **2D UV 공간의 고해상도·세부 특징 학습 능력**과 **3D 포인트를 통한 전역 일관성 및 연속성 유지**를 결합한 새로운 모델을 제안합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 확산 과정 (Diffusion Process)

TEXGen은 DDPM(Denoising Diffusion Probabilistic Model) 프레임워크를 UV 텍스처 공간에 적용합니다.

**순방향 과정 (Forward Process):**

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1-\bar{\alpha}_t)\mathbf{I}\right)$$

여기서 $\mathbf{x}\_0$는 원본 UV 텍스처 맵, $\bar{\alpha}\_t = \prod_{s=1}^{t}(1-\beta_s)$는 노이즈 스케줄을 나타냅니다.

**역방향 과정 (Reverse Process):**

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c}) = \mathcal{N}\left(\mathbf{x}_{t-1};\, \mu_\theta(\mathbf{x}_t, t, \mathbf{c}),\, \Sigma_\theta(\mathbf{x}_t, t)\right)$$

여기서 $\mathbf{c}$는 텍스트 프롬프트 및 단일 뷰 이미지 조건, $\theta$는 TEXGen 네트워크 파라미터입니다.

**학습 목적 함수 (Training Objective):**

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, t}\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t, \mathbf{c}\right)\right\|^2\right]$$

여기서 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$는 실제 노이즈이고, $\boldsymbol{\epsilon}_\theta$는 모델이 예측하는 노이즈입니다.

> ⚠️ **주의:** 위 수식은 TEXGen이 기반으로 하는 표준 DDPM 공식을 UV 공간에 적용한 형태입니다. 공개된 릴리즈 모델은 논문 버전과 달리 **Flow Matching**으로 훈련되어 더 안정적임이 확인되었습니다. Flow Matching의 목적 함수는 다음과 같이 표현됩니다:
> $$\mathcal{L}\_{\text{FM}} = \mathbb{E}\_{t, \mathbf{x}\_0, \mathbf{x}\_1}\left[\left\|v\_\theta(\mathbf{x}\_t, t, \mathbf{c}) - (\mathbf{x}_1 - \mathbf{x}\_0)\right\|^2\right]$$
> 여기서 $v\_\theta$는 벡터 필드를 예측하는 네트워크이며 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$입니다.

#### (2) 조건부 생성 (Classifier-Free Guidance)

TEXGen은 조건부 생성에 **Classifier-Free Guidance (CFG)**를 활용합니다:

$$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w \cdot \left[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)\right]$$

여기서 $w$는 가이던스 가중치(guidance weight), $\varnothing$는 무조건부(unconditional) 임베딩입니다.

---

### 2.3 모델 구조 — 하이브리드 Point-UV 아키텍처

TEXGen은 2D UV 공간의 강점(고해상도 및 세부 특징 학습)과 3D 포인트를 통한 전역 일관성 및 연속성 유지를 결합한 새로운 모델을 제안합니다. 이 두 컴포넌트는 인터리빙(interleaving)되어 표현을 정제하며, 고해상도 2D 텍스처 맵 생성을 위한 효과적인 학습을 촉진합니다.

```
입력: UV 텍스처 맵 (노이즈) + 포인트 클라우드 + 조건 (텍스트/이미지)
        ↓
┌─────────────────────────────────────────┐
│          Hybrid Block × N               │
│  ┌──────────────────────────────────┐   │
│  │   UV Head Block (2D Conv)        │   │  ← UV 맵에서 지역 세부 특징
│  │   Conv → Norm → Activation       │   │
│  └──────────────────────────────────┘   │
│              ↕ 상호 전파                 │
│  ┌──────────────────────────────────┐   │
│  │   Point Cloud Block              │   │  ← 3D 전역 일관성
│  │   Serialized Attention           │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
        ↓
출력: 고해상도 UV 텍스처 맵
```

위치 인코딩(Position Encoding)은 3D 위치 정보를 모델에 통합하는 데 핵심적인 역할을 하며, 2D 및 3D 블록 모두에서 전역 조건 임베딩을 활용해 중간 특징(intermediate features)을 변조(modulate)함으로써 조건부 정보를 주입합니다.

아블레이션 실험 결과, UV 블록만 있는 모델(B)은 전체적인 의미 및 3D 일관성을 포착하기 어렵고, 포인트 블록만 있는 모델(C)은 세밀한 디테일 표현에 어려움을 겪는 것으로 확인되었습니다. (풀 모델 A 대비)

#### 데이터셋

훈련 데이터로 Objaverse를 원시 데이터 소스로 사용하며, 이는 80만 개 이상의 3D 메시를 포함합니다. 그러나 이 메시들의 텍스처 구조가 균일하지 않아 처리 및 필터링이 필요합니다. 예를 들어, 일부 메시는 여러 텍스처 이미지로 나뉘어 있고, 일부는 텍스처 이미지 없이 기본 색상 정보만 있습니다. 품질이 낮은 텍스처를 필터링한 후, xAtlas를 사용해 UV를 재전개(re-unfold)하여 단일 UV 아틀라스로 표현되도록 합니다. 그런 다음, 원본 메시 파일에서 새로 파라미터화된 메시로 확산 색상(diffuse color)을 베이크(bake)합니다.

---

### 2.4 성능 향상

TEXGen은 사전 학습된 2D 텍스트-이미지 확산 모델을 사용한 테스트-타임 최적화에 의존하는 방법들보다 더 세밀하고 일관된 텍스처를 합성합니다. 또한 3D 데이터셋과 3D 표현으로 훈련되었기 때문에 다른 방법들에서 흔히 발생하는 **야누스 문제(Janus problem)를 방지**합니다.

범용 객체에 대한 텍스처를 생성할 수 있는 최초의 피드-포워드 모델로서, TEXGen은 해당 분야에서 새로운 벤치마크를 수립하였습니다.

텍스처 맵의 임의 영역을 마스킹하고 모델이 빈 부분을 채우는 인페인팅 실험에서, 기존 텍스처와 **매끄러운 통합(seamless integration)** 결과를 보여주었습니다.

---

### 2.5 한계

TEXGen은 인상적인 결과를 보여주지만, **연산 집약적인 특성**으로 인해 실시간 응용에 제약이 있을 수 있습니다.

훈련 과정은 최소 40GB VRAM 이상의 GPU를 필요로 하며, Nvidia A100 GPU로 전체 파이프라인을 테스트하였습니다.

추가적인 한계로는 다음이 있습니다 (연구자 관점 분석):

- **UV 언래핑 의존성:** 모델은 단일 UV 아틀라스를 가정하므로, 복잡한 멀티-파트 UV 구조의 메시에 직접 적용하기 어렵습니다.
- **Albedo 전용:** TEXGen은 UV 도메인에서 직접 albedo 텍스처 맵만 확산시키는 모델로서, PBR(Physically Based Rendering) 재질(roughness, metallic, normal map)은 생성하지 않습니다.
- **데이터 편향:** Objaverse 기반 학습으로 인해 데이터셋 분포에 편향이 존재할 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능을 위한 설계 전략

대형 모델들은 뛰어난 일반화 능력을 보여주는데, 그 성공은 주로 두 가지 핵심 요인에 기인합니다: (1) 모델 크기와 데이터 양이 증가할수록 성능이 향상되는 **확장 가능하고 효과적인 네트워크 아키텍처**, (2) 일반화를 촉진하는 **대규모 데이터셋**.

이 논문은 일반화 가능하고 고품질의 메시 텍스처링을 위해 **모델 크기와 데이터를 스케일업**함으로써 대형 생성 모델을 구축할 가능성을 탐구합니다.

### 3.2 일반화 가능성의 구체적 근거

| 요소 | 일반화 기여 |
|------|------------|
| 7억 파라미터 대형 모델 | 스케일 법칙(scaling law)에 의한 성능 향상 |
| Objaverse 800K+ 메시 | 다양한 카테고리의 3D 자산 커버 |
| 하이브리드 Point-UV 구조 | UV 단편 문제 해결 → 임의 메시에 적용 가능 |
| 피드-포워드 추론 | 테스트-타임 최적화 불필요 → 새로운 객체 범주에 즉시 적용 |

UV 텍스처 맵을 생성 표현으로 활용함으로써, 확장성을 확보하고 고해상도 세부 사항을 보존합니다. 더 중요하게는, 렌더링 손실에만 의존하지 않고 **실측 텍스처 맵(ground-truth texture maps)에서 직접 지도 학습(supervision)**이 가능해, 확산 기반 학습과 호환되며 전반적인 생성 품질을 향상시킵니다.

또한, 모델은 다양한 소스의 부분 텍스처 맵을 유연하게 칠할 수 있으며, 보이지 않는 영역을 효과적으로 칠하고 연속성과 일관성을 보장합니다.

### 3.3 향후 일반화 성능 향상을 위한 연구 방향

1. **더 많은 카테고리와 실세계 스캔 데이터 포함** → 도메인 갭 감소
2. **멀티모달 조건 확장** (예: 재질 특성, 조명 조건 추가)
3. **PBR 재질 생성으로 확장** → roughness, metallic 등 포함
4. **증류(distillation) 기법** 적용으로 추론 속도 향상 → 더 넓은 배포 가능성

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 주요 관련 연구 타임라인

```
2022 ── TEXTure (Richardson et al., 2023): 뷰별 순차적 인페인팅
2023 ── Text2Tex (Chen et al.): Depth-aware 확산으로 텍스처 합성
     ── TexFusion (Cao et al.): 잠재 공간에서 인터리빙 멀티뷰 샘플링
     ── Point-UV Diffusion (Yu et al.): 포인트 클라우드 + UV 결합
2024 ── TexGen (Huo et al., ECCV 2024): 멀티뷰 샘플링/리샘플링 프레임워크
     ── GenesisTex (Gao et al., CVPR 2024): 이미지 디노이징을 텍스처 공간에 적용
     ── Paint3D (Zeng et al.): UV 인페인팅 + UVHD 확산 모델
     ── TEXGen (Yu et al., SIGGRAPH Asia 2024): UV 공간 직접 학습 (본 논문)
```

### 4.2 상세 비교표

| 방법 | 발표 | 접근 방식 | 피드-포워드 | 야누스 문제 | UV 직접 학습 | 주요 한계 |
|------|------|-----------|------------|------------|--------------|----------|
| **TEXTure** | SIGGRAPH 2023 | 뷰별 순차 인페인팅 | ❌ | ❌ 있음 | ❌ | seam 아티팩트 |
| **Text2Tex** | ICCV 2023 | Depth-aware 확산 | ❌ | ❌ 있음 | ❌ | view 불일관성 |
| **TexFusion** | - | 잠재 공간 멀티뷰 | ❌ | △ | ❌ | 과도한 스무딩 |
| **Point-UV** | ICCV 2023 | 포인트 클라우드 → UV | △ | △ | △ | 소규모 데이터 |
| **Paint3D** | CVPR 2024 | UV 인페인팅 확산 | △ | △ | △ | 조명 의존성 |
| **TEXGen** | **SIGGRAPH Asia 2024** | **UV 직접 확산** | ✅ | ✅ 없음 | ✅ | 연산 비용, albedo만 |

Point-UV 및 TexOct와 같은 포인트 클라우드 기반 방법들은 입력 메시와의 3D 전역 일관성 측면에서 더 나은 성능을 보이지만, 소규모 데이터셋으로 학습되어 일반화에 제약이 있습니다.

비교군인 TexGen (Huo et al., ECCV 2024)은 RGB 텍스처 공간에서 뷰 일관성 샘플링을 직접 강제하고, 풍부한 텍스처 세부 사항을 유지하기 위한 노이즈 리샘플링 전략을 개발하였습니다.

> ⚠️ **표기 주의:** "TexGen"(Huo et al., ECCV 2024)과 본 논문 "TEXGen"(Yu et al., SIGGRAPH Asia 2024)은 **서로 다른 논문**입니다. 전자는 멀티뷰 샘플링/리샘플링 방법이고, 후자(본 논문)는 UV 공간에서 직접 학습하는 대형 확산 모델입니다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구 영향

TEXGen은 범용 객체에 대한 텍스처를 생성할 수 있는 최초의 피드-포워드 모델로서 해당 분야에서 새로운 벤치마크를 수립하였으며, 이 기여가 텍스처 생성 및 그 이상의 분야에서 추가 연구와 발전을 촉진할 것으로 기대됩니다.

구체적인 영향 영역:

1. **3D 콘텐츠 생성 파이프라인 혁신**
   고품질 기하 인지(geometry-aware) 텍스처를 생성하는 능력은 게임, 애니메이션, 가상 현실 개발에서 워크플로우를 변화시킬 잠재력이 있으며, 복잡한 예술적 작업을 AI가 자동화할 가능성을 보여줍니다.

2. **UV 공간 생성 모델 연구 흐름 선도**
   TEXGen 이후 등장한 SeqTex(Yuan et al., 2025)는 비디오 확산 모델을 활용하여 합성된 프레임 간 뷰 일관성을 향상시키는 방향으로 연구가 이어지고 있습니다.

3. **대형 3D 생성 모델의 스케일링 법칙 검증**
   이 모델들은 고품질 결과를 생성하고 뛰어난 일반화 능력을 보여주며, 그 성공은 주로 모델 크기와 데이터 양이 증가할수록 성능이 향상되는 확장 가능한 네트워크 아키텍처와 일반화를 촉진하는 대규모 데이터셋이라는 두 가지 핵심 요인에 기인합니다.

### 5.2 향후 연구 시 고려할 점

#### (A) 기술적 확장 방향

| 고려 사항 | 설명 |
|----------|------|
| **PBR 재질 생성** | Roughness, metallic, normal map 통합으로 물리 기반 렌더링 지원 |
| **동적 메시 지원** | 애니메이션 메시(예: 스키닝된 캐릭터)에 대한 시간 일관성 |
| **추론 가속화** | Consistency Model, DDIM, Flow Matching 등으로 샘플링 단계 감소 |
| **해상도 스케일업** | 더 높은 해상도 UV 맵(예: 4K) 생성 지원 |

#### (B) 데이터 및 일반화 관련 고려 사항

- **실세계 스캔 데이터 활용**: Objaverse는 작가가 제작한 합성 자산이 많아, 실세계 스캔(포토그래메트리) 데이터와의 갭이 있을 수 있습니다.
- **다양한 UV 언래핑 전략**: xAtlas 기반 단일 아틀라스 가정에서 벗어나, 다양한 UV 파라미터화(parametrization)에 대한 강인성 확보가 필요합니다.
- **도메인 특화 미세 조정**: 게임, 의료 시뮬레이션 등 특정 도메인에 맞는 파인튜닝 전략 연구가 필요합니다.

#### (C) 평가 지표 관련 고려 사항

텍스트-텍스처 유사도 평가에 Claude 3.5-Sonnet과 같은 MLLM 기반 스코어를 활용하는 방향이 논문에서 제시되었습니다. 이는 새로운 평가 방법론으로, 향후 연구에서 표준화된 벤치마크 개발이 필요합니다.

#### (D) 윤리 및 저작권 고려 사항

- 대규모 3D 자산 데이터셋(Objaverse 등)의 라이선스 및 저작권 문제
- 딥페이크 수준의 사실적 3D 텍스처 생성에 따른 오용 가능성
- 생성된 텍스처의 일관성 검증 도구 필요

---

## 참고자료 목록

| # | 참고 자료 |
|---|-----------|
| 1 | **TEXGen: a Generative Diffusion Model for Mesh Textures** — arXiv:2411.14740 (https://arxiv.org/abs/2411.14740) |
| 2 | **TEXGen 공식 프로젝트 페이지** — https://cvmi-lab.github.io/TEXGen/ |
| 3 | **TEXGen ACM TOG 공식 발표** — ACM Trans. Graph. Vol.43, No.6, Article 213 (https://dl.acm.org/doi/10.1145/3687909) |
| 4 | **TEXGen GitHub (공식 구현)** — https://github.com/CVMI-Lab/TEXGen |
| 5 | **TEXGen Hugging Face Papers 페이지** — https://huggingface.co/papers/2411.14740 |
| 6 | **TEXGen arXiv HTML 전문** — https://arxiv.org/html/2411.14740v1 |
| 7 | **TEXGen ResearchGate PDF** — https://www.researchgate.net/publication/386093769 |
| 8 | **TexGen (Huo et al., ECCV 2024)** — arXiv:2408.01291 (비교 연구) |
| 9 | **TexSpot: 3D Texture Enhancement with Spatially-uniform Point Latent Representation** — arXiv:2602.12157 (후속 연구 인용 포함) |
| 10 | **TexOct: Generating Textures of 3D Models with Octree-based Diffusion** — CVPR 2024 (비교 연구) |
| 11 | **TEXGen AI Models FYI 분석 페이지** — https://www.aimodels.fyi/papers/arxiv/texgen-generative-diffusion-model-mesh-textures |
