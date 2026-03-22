# MVDream: Multi-view Diffusion for 3D Generation

> **논문 정보**: Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, Xiao Yang (ByteDance)
> **발표**: ICLR 2024 | arXiv:2308.16512

---

## 1. 핵심 주장 및 주요 기여 요약

MVDream은 텍스트 프롬프트로부터 일관된 멀티뷰 이미지를 생성할 수 있는 디퓨전 모델로, 2D와 3D 데이터 모두에서 학습하여 2D 디퓨전 모델의 일반화 능력과 3D 렌더링의 일관성을 동시에 달성한다.

**주요 기여 사항:**

1. 이러한 멀티뷰 디퓨전 모델이 3D 표현에 구애받지 않는 일반화 가능한 3D prior임을 암묵적으로 증명한다.
2. Score Distillation Sampling(SDS)을 통해 3D 생성에 적용하여, 기존 2D-lifting 방법의 일관성과 안정성을 크게 향상시킨다.
3. DreamBooth와 유사하게, 소수의 2D 예시로부터 새로운 개념을 학습하여 3D 생성이 가능하다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 2D-lifting 방법들은 3D 인식 부재로 인해 multi-face Janus 문제와 뷰 간 콘텐츠 드리프트 같은 멀티뷰 비일관성 문제를 겪고 있었다.

구체적으로 두 가지 핵심 문제가 존재한다:

- **Janus(다면) 문제**: 말이 두 개의 얼굴을 가지는 것처럼, 시스템이 텍스트에 묘사된 콘텐츠를 각 뷰마다 반복적으로 재생성한다.
- **콘텐츠 드리프트**: 객체 세부사항과 전체 형태가 뷰 간에 비일관적으로 변화하여 비정합적인 3D 기하를 초래한다.

이러한 문제들은 2D 디퓨전 모델이 본질적으로 3D 기하 일관성을 이해하거나 유지할 수 없다는 근본적 한계에서 비롯되며, 저자들은 완벽한 카메라 조건부 2D 모델조차도 불충분할 수 있다고 주장한다.

### 2.2 제안하는 방법 및 수식

#### (A) Score Distillation Sampling (SDS) 확장

SDS 손실은 명시적 형태가 아닌 **그래디언트**로 정의된다. 원래 SDS의 그래디언트는 다음과 같다:

$$\nabla_\phi \mathcal{L}_{\text{SDS}}(\phi) = \mathbb{E}_{t,\epsilon}\left[ w(t)\left(\epsilon_\theta(\mathbf{x}_t; y, t) - \epsilon\right) \frac{\partial \mathbf{x}}{\partial \phi} \right]$$

여기서:
- $\mathbf{x}_t = \alpha_t \mathbf{x} + \sigma_t \epsilon$ 은 노이즈가 추가된 이미지
- $\alpha_t$, $\sigma_t$는 노이즈 스케줄에 의해 제어되는 신호 및 노이즈 스케일 ($\alpha_t^2 + \sigma_t^2 = 1$)
- $\epsilon_\theta(\mathbf{x}_t; y, \mathbf{c}, t)$는 디퓨전 모델의 노이즈 예측 네트워크
- $y$는 텍스트 프롬프트, $\mathbf{c}$는 카메라 파라미터

MVDream은 이를 **멀티뷰 SDS**로 확장한다. 4개의 뷰를 동시에 생성하면서, 각 뷰의 노이즈 예측이 다른 뷰와의 3D self-attention을 통해 상호 참조된다:

$$\nabla_\phi \mathcal{L}_{\text{MV-SDS}}(\phi) = \mathbb{E}_{t,\epsilon}\left[ w(t) \sum_{i=1}^{N} \left(\epsilon_\theta(\mathbf{x}_t^{(1:N)}; y, \mathbf{c}^{(1:N)}, t) - \epsilon^{(i)}\right) \frac{\partial \mathbf{x}^{(i)}}{\partial \phi} \right]$$

여기서 $N=4$개의 직교 뷰를 동시에 처리한다.

또한 SDS 품질 향상을 위해 $\mathbf{x}_0$-reconstruction loss를 사용한다:

$$\nabla_\phi \mathcal{L}_{\text{recon}}(\phi) = \mathbb{E}_{t,\epsilon}\left[ w(t) \frac{\alpha_t}{\sigma_t}\left(\epsilon_\theta(\mathbf{x}_t; y, \mathbf{c}, t) - \epsilon\right) \frac{\partial \mathbf{x}}{\partial \phi} \right]$$

#### (B) 멀티뷰 디퓨전 학습 손실

이미지 데이터셋 $\mathcal{X}$와 멀티뷰 이미지 데이터셋 $\mathcal{X}\_{mv}$가 주어지고, 학습 샘플 $\{x, y, c\} \in \mathcal{X} \cup \mathcal{X}_{mv}$에 대해 멀티뷰 디퓨전 손실이 정의된다:

$$\mathcal{L}_{\text{MV}} = \mathbb{E}_{\{x,y,c\} \in \mathcal{X} \cup \mathcal{X}_{mv}, \epsilon, t} \left[ \| \epsilon_\theta(\mathbf{x}_t; y, \mathbf{c}, t) - \epsilon \|_2^2 \right]$$

여기서 2D 이미지 학습 시에는 **2D self-attention** 모드, 멀티뷰 이미지 학습 시에는 **3D self-attention** 모드로 전환된다.

#### (C) DreamBooth 확장 손실

이미지 파인튜닝 손실과 파라미터 보존 손실 두 가지를 사용한다. $\mathcal{X}_{id}$를 아이덴티티 이미지 집합이라 하면:

$$\mathcal{L}_{\text{DB}} = \mathcal{L}_{\text{LDM}}(\mathcal{X}_{id}) + \frac{\lambda}{N} \|\theta - \theta_0\|_2^2$$

여기서 $\mathcal{L}_{\text{LDM}}$은 이미지 디퓨전 손실, $\theta_0$은 원래 멀티뷰 디퓨전의 초기 파라미터, $N$은 파라미터 수, $\lambda=1$이다.

### 2.3 모델 구조

핵심 혁신은 기존 2D 디퓨전 모델(Stable Diffusion v2.1)을 멀티뷰 이미지 생성을 위해 적응시키는 것이며, 핵심 아키텍처 수정은 "inflated 3D self-attention"의 도입이다.

**구조적 핵심 요소:**

| 구성 요소 | 설명 |
|-----------|------|
| **Base Model** | Stable Diffusion v2.1 (512×512 → 256×256 멀티뷰) |
| **Inflated 3D Self-Attention** | UNet 아키텍처 내에 inflated 3D self-attention을 적용하여 교차 뷰 의존성을 효과적으로 모델링 |
| **Camera Embedding** | 카메라 임베딩을 time embedding에 잔차(residual)로 통합하여 정밀한 카메라 포즈 제어를 제공 |
| **Joint Training** | 3D 렌더링 데이터셋과 LAION의 대규모 2D 이미지-텍스트 쌍을 결합한 공동 학습 전략 |
| **3D 표현** | NeRF (multi-resolution hash-grid) |

MVDream의 멀티뷰 디퓨전 UNet의 각 블록은 4-뷰 이미지 잠재 공간에서 밀집 연결된 3D attention을 포함하여 교차 뷰 상호작용의 학습을 촉진하고 3D 일관성을 높인다.

**Attention 메커니즘 비교 (논문 내 ablation):**

기존 self-attention에 새로운 3D self-attention 레이어를 추가하는 방식은 멀티뷰 이미지의 생성 품질을 저하시켰는데, 이는 기존 파라미터를 재활용하는 것에 비해 새 attention 모듈의 수렴이 느리기 때문이다.

### 2.4 성능 향상

MVDream의 결과물은 곱슬머리, 동물 털 질감 등 객체 디테일에서 더 높은 품질을 보이며, 이는 SDS 손실을 이용한 NeRF 학습 과정에서 더 높은 기하 일관성을 생성하기 때문이다.

**정량적 비교:**

계산 효율 측면에서 MVDream의 멀티뷰 SDS는 기존 베이스라인과 경쟁적이거나 더 빠르며, 단일 V100 GPU에서 약 2시간이 소요된다. 이는 Magic3D-IF-SD(3.5시간)와 ProlificDreamer(10시간 이상)보다 크게 빠르면서도 더 높은 품질과 일관성을 달성한다.

**적용된 품질 향상 기법:**

타임스텝 어닐링(linearly anneal), 저품질 3D 스타일 방지를 위한 고정 네거티브 프롬프트, CFG rescale를 통한 색상 과포화 완화 기법을 제안한다.

### 2.5 한계

1. 데이터셋 크기 증대 및 SDXL과 같은 더 큰 디퓨전 모델로의 교체 가능성이 있으며, 생성된 스타일(조명, 텍스처)이 렌더링된 학습 데이터셋과 유사해지는 경향이 관찰된다.

2. 더 복잡하거나 동적인 3D 장면에서의 성능, 그리고 오클루전 및 기타 어려운 3D 시나리오에 대한 처리 능력이 불분명하다.

3. 멀티뷰 SDS로 일관된 3D 모델을 생성할 수 있으나, 콘텐츠 풍부함과 텍스처 품질은 디노이징 디퓨전 과정에서 직접 샘플링된 이미지에 비해 여전히 부족하다.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화의 핵심 메커니즘

MVDream의 일반화 전략은 **2D와 3D 데이터의 공동 학습**에 핵심이 있다:

사전 학습된 2D 디퓨전 모델의 전이 학습을 활용하여 일반화 능력을 상속받고, 멀티뷰 이미지(3D 에셋)와 2D 이미지-텍스트 쌍을 공동 학습함으로써 일관성과 일반화 모두를 달성한다.

### 3.2 2D 데이터 혼합 학습의 효과

2D 데이터를 포함하여 학습한 것과 포함하지 않은 것의 비교에서, 2D 데이터 추가가 모델 일반화에 확실한 개선을 가져와 더 좋은 이미지 품질과 텍스트-이미지 대응을 이끌어낸다.

### 3.3 랜덤 뷰로의 일반화

직교 뷰로 학습했음에도 랜덤 카메라 뷰로 일반화 가능한지에 대한 실험에서, 제안된 3D self-attention 네트워크가 실제로 랜덤 멀티뷰 이미지를 생성할 수 있으며, 4개 뷰로 학습했음에도 추론 시 64개 이상의 뷰를 생성할 수 있다.

### 3.4 DreamBooth를 통한 개인화 일반화

DreamBooth에서 영감을 받아, 멀티뷰 디퓨전 모델이 2D 이미지 모음으로부터 아이덴티티 정보를 동화할 수 있으며, 파인튜닝 후에도 강건한 멀티뷰 일관성을 보인다.

### 3.5 일반화 향상을 위한 향후 방향

| 방향 | 세부 내용 |
|------|-----------|
| **더 큰 베이스 모델** | SDXL 등 더 큰 디퓨전 모델로 교체 시 성능 향상 기대 |
| **다양한 렌더링** | 더 다양하고 사실적인 렌더링이 더 나은 멀티뷰 디퓨전 모델 학습에 필요하지만 비용이 크다 |
| **SDS 변형과의 결합** | 이론적으로 SJC, VSD 등 다른 SDS 변형과 결합 가능 |

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

1. **멀티뷰 디퓨전 패러다임 확립**: MVDream은 원래 SDS 접근법에 내재된 'Janus-face' 문제를 해결하기 위해 설계된 DreamFusion의 직접적 개선으로 제시된다. 이후 **ImageDream**, **Zero123++**, **SyncDreamer** 등 후속 연구의 기반이 되었다.

2. **하이브리드 학습 전략의 선례**: 2D와 3D 데이터를 혼합하는 전략은 이후 연구에서 표준적으로 채택되었다.

3. **3D prior로서의 디퓨전 모델**: 멀티뷰 디퓨전 모델이 3D 표현에 구애받지 않는 일반화 가능한 3D prior임을 증명한 것은 NeRF, 3DGS, DMTet 등 다양한 표현에 적용 가능성을 열었다.

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 상세 |
|-----------|------|
| **3D 표현의 발전** | 3D Gaussian Splatting (3DGS) 등 최신 표현과의 통합 |
| **생성 속도** | Feed-forward 방식으로의 전환 (LGM, Turbo3D 등) |
| **텍스처 품질** | SDS 기반 최적화의 본질적 한계 극복 |
| **복잡한 장면** | 단일 객체를 넘어 복합 장면으로의 확장 |
| **평가 체계** | 통일된 벤치마크(T3Bench, GPTEval3D) 필요성 |
| **인간 선호도 정렬** | DreamReward, DreamDPO 등 RLHF 기반 3D 품질 향상 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 방법론 계보

```
DreamFusion (2022) → Magic3D (2022) → ProlificDreamer (2023)
        ↓                                        ↓
   MVDream (2023)  ←→  Zero123/Zero123++ (2023)
        ↓                        ↓
 ImageDream (2023)     One-2-3-45++ (2023)
        ↓                        ↓
 DreamCraft3D (2024)   LGM (2024) / Splatter Image (2024)
        ↓
 Turbo3D (2025) / Prometheus (2025) / DreamCS (2025)
```

### 5.2 비교 분석표

| 방법 | 연도 | 핵심 기법 | 3D 일관성 | 일반화 | 속도 | 표현 |
|------|------|-----------|-----------|--------|------|------|
| **DreamFusion** | 2022 | SDS + Imagen | ❌ Janus 문제 | ✅ 높음 | ~1.5h | NeRF |
| **Magic3D** | 2022 | Coarse-to-fine SDS | ❌ 부분적 | ✅ 높음 | ~3.5h | NeRF→DMTet |
| **ProlificDreamer** | 2023 | VSD (Variational SDS) | ⚠️ 개선 | ✅ 높음 | >10h | NeRF |
| **MVDream** | 2023 | Multi-view SDS | ✅ 해결 | ✅ 높음 | ~2h | NeRF |
| **Zero123++** | 2023 | Image-conditioned MV | ✅ 높음 | ⚠️ 이미지 의존 | 빠름 | 다양 |
| **DreamCraft3D** | 2024 | Hierarchical bootstrapped prior | ✅ 높음 | ✅ 높음 | 중간 | NeRF→Mesh |
| **LGM** | 2024 | Large MV Gaussian Model | ✅ 높음 | ⚠️ 중간 | <5초 | 3DGS |
| **HIFA** | 2024 | Holistic sampling + smoothing | ⚠️ 개선 | ✅ 높음 | 중간 | NeRF |
| **DreamFlow** | 2024 | ProlificDreamer 대비 CLIP R-precision에서 우수, 5배 빠른 생성 | ⚠️ 개선 | ✅ 높음 | 빠름 | NeRF→Mesh |
| **Turbo3D** | 2025 | Ultra-fast feed-forward | ✅ 높음 | ⚠️ 중간 | 초고속 | 3DGS |
| **Prometheus** | 2025 | 3D-aware latent diffusion | ✅ 높음 | ✅ 높음 | Feed-forward | Latent 3D |
| **DreamCS** | 2025 | Geometry-aware reward + SDS | ✅ 높음 | ✅ 높음 | 중간 | Mesh |

### 5.3 핵심 트렌드 분석

1. **SDS 최적화 → Feed-forward 생성으로의 전환**: MVDream은 여전히 per-prompt 최적화가 필요하지만, 최근 대규모 재구성 모델들은 입력 이미지를 토큰으로 인코딩하고 NeRF(Triplanes), 3D Gaussians, DMTet/Flexicube 등의 암묵적 표현으로 객체를 재구성하는 강력한 트랜스포머 아키텍처를 학습한다.

2. **3D Gaussian Splatting의 부상**: 3D Gaussian splatting이 환경과 객체 모두를 표현하는 표준이 되어가고 있으며, 명시적이고 미분 가능하며 메모리 효율적인 볼류메트릭 렌더링을 제공한다.

3. **인간 선호도 기반 정렬**: DreamReward는 어노테이션된 렌더링에서 멀티뷰 이미지 기반 보상 모델을 학습하고, DreamDPO는 멀티뷰 비교에서의 인간 선호도를 DPO로 적용하지만, 두 접근 모두 2D 감독에 제한되어 뷰별 외관만 정렬하며 글로벌 3D 구조의 일관성은 부족하다.

---

## 참고 자료 및 출처

1. **[논문 원문]** Shi, Y., Wang, P., Ye, J., Mai, L., Li, K., & Yang, X. (2023). "MVDream: Multi-view Diffusion for 3D Generation." *ICLR 2024*. arXiv:2308.16512 — https://arxiv.org/abs/2308.16512
2. **[ICLR Proceedings]** https://proceedings.iclr.cc/paper_files/paper/2024/file/adbe936993aa7cf41e45054d8b72f183-Paper-Conference.pdf
3. **[GitHub Repository]** ByteDance MVDream — https://github.com/bytedance/MVDream
4. **[논문 HTML 버전]** https://arxiv.org/html/2308.16512v3
5. **[alphaXiv 분석]** https://www.alphaxiv.org/overview/2308.16512v4
6. **[Liner Review]** https://liner.com/review/mvdream-multiview-diffusion-for-3d-generation
7. **[Louis Bouchard 해설]** https://www.louisbouchard.ai/mvdream/
8. **[AI Models FYI 리뷰]** https://www.aimodels.fyi/papers/arxiv/mvdream-multi-view-diffusion-3d-generation
9. **[OpenReview]** https://openreview.net/forum?id=FUgrjq2pbB
10. **[HuggingFace Paper Page]** https://huggingface.co/papers/2308.16512
11. **[ResearchGate]** https://www.researchgate.net/publication/373552037
12. **[MultiImageDream]** https://arxiv.org/html/2404.17419v1
13. **[MVD² - ACM]** https://dl.acm.org/doi/fullHtml/10.1145/3641519.3657403
14. **[Text-to-3D Survey - EG]** https://diglib.eg.org/bitstream/handle/10.1111/cgf15061/v43i2star_02_15061.pdf
15. **[HIFA - ICLR 2024]** https://proceedings.iclr.cc/paper_files/paper/2024/file/178ae4ba29022eb7bf509c2e27bc8ab8-Paper-Conference.pdf
16. **[DreamFlow - ICLR 2024]** https://proceedings.iclr.cc/paper_files/paper/2024/file/57568e093cbe0a222de0334b36e83cf5-Paper-Conference.pdf
17. **[DreamCS - ICLR 2025]** https://openreview.net/pdf/ad6191f1b77f59d05c7a2c3320e1da0da4647196.pdf
18. **[Text-to-3D 비교 (2024-2025)]** https://www.icck.org/article/epdf/ngcst/599
19. **[Semantic Scholar]** https://www.semanticscholar.org/paper/9aa01997226b5c4d705ae2e2f52c32681006654b

> **참고**: 본 분석에서 수식은 논문의 공식 내용을 기반으로 재구성하였으며, 일부 표기는 논문 원본의 표기법을 따랐습니다. 정확한 수식 확인은 ICLR 공식 논문(출처 2)을 참조해 주시기 바랍니다.
