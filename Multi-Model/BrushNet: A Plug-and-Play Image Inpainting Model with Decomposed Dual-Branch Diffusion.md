
# BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion

> **논문 정보**
> - **저자:** Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, Qiang Xu
> - **소속:** ARC Lab, Tencent PCG / The Chinese University of Hong Kong
> - **발표:** ECCV 2024
> - **arXiv:** [2403.06976](https://arxiv.org/abs/2403.06976)
> - **공식 코드:** [GitHub – TencentARC/BrushNet](https://github.com/TencentARC/BrushNet)
> - **Springer DOI:** [10.1007/978-3-031-72661-3_9](https://doi.org/10.1007/978-3-031-72661-3_9)

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

Image Inpainting(이미지 인페인팅), 즉 손상된 이미지를 복원하는 기술은 Diffusion Model(DM)의 발전으로 크게 진보하였으나, 현재의 DM 기반 인페인팅 접근법들은 샘플링 전략 수정이나 인페인팅 전용 DM 개발 등의 방식으로 이루어지며, 의미론적 비일관성(Semantic Inconsistency)과 이미지 품질 저하 문제를 자주 겪는다.

이에 대응하여, BrushNet은 마스킹된 이미지 특징과 노이즈 잠재 표현(noisy latent)을 별도의 브랜치로 분리하는 새로운 패러다임을 도입하며, 이 분리는 모델의 학습 부담을 획기적으로 줄이고, 계층적 방식으로 마스킹된 이미지 정보를 세밀하게 통합한다. BrushNet은 픽셀 수준의 마스킹 이미지 특징을 사전 학습된 임의의 DM에 삽입할 수 있는 플러그 앤 플레이(Plug-and-Play) 이중 브랜치 모델이다.

### 1.2 주요 기여 (Contributions)

BrushNet은 어떤 사전 학습된 Diffusion Model에도 플러그 앤 플레이 방식으로 적용 가능한 텍스트 기반 이미지 인페인팅 모델이며, 핵심 설계 통찰은 두 가지이다: (1) 마스킹된 이미지 특징과 노이즈 잠재 표현의 분리를 통한 학습 부담 감소, (2) 전체 사전 학습 모델에 걸친 밀집 픽셀 단위 제어를 통한 인페인팅 적합성 향상.

추가로, 세그멘테이션 기반 인페인팅 학습과 성능 평가를 위한 **BrushData**와 **BrushBench**를 도입하였으며, 이미지 품질, 마스크 영역 보존, 텍스트 정합성을 포함한 7가지 핵심 지표에서 기존 모델 대비 우월한 성능을 입증하였다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

기존의 ControlNet과 같은 해결책은 인페인팅에 직접 적용 시 정보 추출 및 삽입이 불충분한 문제가 발생하는데, 이는 제어 가능한 이미지 생성과 인페인팅의 근본적인 차이에서 비롯된다: 인페인팅은 텍스트에 의존한 콘텐츠 완성이 아닌 강한 구속 정보를 갖는 픽셀-투-픽셀(pixel-to-pixel) 제약이 필요하다. 결과적으로 ControlNet은 전용 인페인팅 모델에 비해 불만족스러운 결과를 초래한다.

또한 기존의 인페인팅 전용 DM 방식들은 학습된 인페인팅 능력을 임의의 사전 학습 모델로 효과적으로 전달하기 어렵기 때문에 적용 가능성이 제한된다.

### 2.2 제안 방법 및 수식

#### 2.2.1 Diffusion Model 기본 수식

DDPM(Denoising Diffusion Probabilistic Model)의 역방향 과정(Reverse Process):

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)\right)$$

노이즈 예측을 위한 학습 목적함수 (LDM, Latent Diffusion Model 기준):

$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_0, \mathbf{c}, \epsilon \sim \mathcal{N}(0,\mathbf{I}), t}\left[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, \mathbf{c}, t)\|^2\right]$$

여기서 $\mathbf{z}\_t$는 $t$ 시점의 노이즈 잠재 표현, $\mathbf{c}$는 텍스트 조건, $\epsilon_\theta$는 노이즈 예측 네트워크이다.

#### 2.2.2 BrushNet의 입력 구성

모델은 먼저 마스크를 잠재 공간 크기에 맞게 다운샘플링하고, 마스킹된 이미지를 VAE 인코더에 입력하여 잠재 공간 분포에 정렬한다. 이후 노이즈 잠재 표현, 마스킹된 이미지 잠재 표현, 다운샘플링된 마스크를 연결(concatenation)하여 BrushNet의 입력으로 사용한다.

이를 수식으로 표현하면:

$$\mathbf{z}^{BN}_{input} = \text{Cat}(\mathbf{z}_t,\ \mathcal{E}(\mathbf{x}_{masked}),\ \downarrow\mathbf{m})$$

여기서:
- $\mathbf{z}_t$: 시각 $t$에서의 노이즈 잠재 표현
- $\mathcal{E}(\mathbf{x}_{masked})$: VAE 인코더를 통해 처리된 마스킹 이미지의 잠재 표현
- $\downarrow\mathbf{m}$: 다운샘플링된 이진 마스크

#### 2.2.3 이중 브랜치 구조와 Zero Convolution 주입

BrushNet의 특징은 제로 컨볼루션(Zero Convolution)—모든 가중치가 0으로 시작하는 1×1 합성곱—을 통해 BrushNet의 특징을 Base U-Net에 주입하는 것이다. 학습 초기에는 BrushNet의 출력이 0이 되어 Base U-Net에 아무 영향을 주지 않으며, 학습 과정에서 점진적으로 정보를 주입하게 된다.

BrushNet의 출력이 Base U-Net에 더해지는 방식:

$$\mathbf{f}^{base}_{l} \leftarrow \mathbf{f}^{base}_{l} + \mathcal{Z}_l\left(\mathbf{f}^{BN}_{l}\right)$$

여기서:
- $\mathbf{f}^{base}_{l}$: $l$번째 레이어의 Base U-Net 특징
- $\mathbf{f}^{BN}_{l}$: $l$번째 레이어의 BrushNet 특징
- $\mathcal{Z}_l(\cdot)$: $l$번째 레이어의 Zero Convolution

각 주입 지점은 서브레이어별로 총 약 25개의 주입 지점으로 구성되며, 이러한 밀집 주입은 BrushNet에 픽셀 수준의 제어를 부여한다. 이는 생성 콘텐츠와 원본 이미지가 만나는 경계를 정밀하게 처리해야 하는 인페인팅에 매우 중요하다.

#### 2.2.4 텍스트 크로스 어텐션 제거

BrushNet 브랜치에서는 크로스 어텐션 레이어가 제거되어 있다. 이는 의도적 설계로, BrushNet의 역할은 순수하게 공간적—"여기에 구멍이 있고, 주변에는 이런 내용이 있다"—이며, 텍스트 이해는 Base U-Net에서 담당한다. 이 분리 덕분에 BrushNet은 약 480M 파라미터(Base U-Net의 약 520M에 비해 작음)로 더 경량화되며, 무엇을 생성할지보다는 어디에 인페인팅할지에 집중한다.

#### 2.2.5 블렌딩 연산 (Blending Operation)

비마스킹 영역의 일관성 유지를 위해 스무딩된 블렌딩(Blurred Blending)을 적용한다:

$$\mathbf{x}_{inpainted} = (1 - \tilde{\mathbf{m}}) \odot \mathbf{x}_{original} + \tilde{\mathbf{m}} \odot \mathbf{x}_{generated}$$

여기서 $\tilde{\mathbf{m}}$은 가우시안 블러가 적용된 마스크로, 경계 영역의 자연스러운 전환을 유도한다.

### 2.3 모델 구조

BrushNet의 설계 선택의 근거는: (1) 무작위 초기화된 합성곱 레이어 대신 VAE 인코더를 사용하여 마스킹된 이미지를 처리함, (2) BrushNet의 전체 U-Net 특징을 레이어별로 사전 학습된 U-Net에 통합함, (3) BrushNet에서 텍스트 크로스 어텐션을 제거하여 마스킹된 이미지 특징이 텍스트에 영향받지 않도록 함이다.

아블레이션에서 검토된 구성 요소들은: 이미지 인코더(무작위 Conv vs. VAE), 마스크 입력 포함 여부, 크로스 어텐션 레이어 유무, U-Net 특징 추가 방식(전체/절반/ControlNet 방식), 블렌딩 연산 방식(미사용/직접 붙여넣기/블러 블렌딩)이다.

```
[입력 파이프라인]
원본 이미지 → 마스킹 → VAE Encoder → 마스킹 이미지 잠재 표현
                                                    ↓
노이즈 z_t ──────────────────────────────────────→ Cat ──→ [BrushNet Branch]
다운샘플링된 마스크 ────────────────────────────────↗         ↓ (Zero Conv)
                                                        [Base U-Net] ←── 텍스트 조건 c
                                                            ↓
                                                      출력 이미지 (Blending 적용)
```

BrushNet은 완전히 학습 가능한 U-Net을 포함한 이중 브랜치 아키텍처를 활용하여 의미론적 효과를 증폭시키며, PowerPaint는 다른 완성 태스크를 위한 별도의 파라미터를 채택한다.

### 2.4 성능 향상

BrushNet은 (I) 랜덤 마스크 (<50% 마스킹), (II) 랜덤 마스크 (>50% 마스킹), (III) 세그멘테이션 마스크 내부 인페인팅, (IV) 세그멘테이션 마스크 외부 인페인팅 등의 다양한 인페인팅 태스크에서 BLD(Blended Latent Diffusion), SDI(Stable Diffusion Inpainting), HDP(HD-Painter), PP(PowerPaint), CNI(ControlNet-Inpainting)와 비교되어, 스타일, 콘텐츠, 색상, 프롬프트 정렬에서 우월한 일관성을 보인다.

BrushNet은 마스킹 영역 보존, 텍스트 정렬, 이미지 품질 전반에서 탁월한 효율성을 보이며, SDI, HD-Painter, PowerPaint 등의 모델은 내부 인페인팅(inside-inpainting) 태스크에서 강한 성능을 보이지만 외부 인페인팅(outside-inpainting) 태스크에서 텍스트 정렬과 이미지 품질이 크게 저하된다. 전반적으로 BrushNet이 가장 강한 결과를 제공한다.

EditBench에서의 성능 역시 BrushBench에서의 전반적 성능과 일관되게 BrushNet의 우월한 성능을 보여준다.

**비교 결과 정리:**

| 모델 | Plug-and-Play | 픽셀 단위 제어 | 마스크 인식 | 콘텐츠 인식 |
|---|---|---|---|---|
| Blended Latent Diffusion | ✅ | ❌ | ❌ | ❌ |
| Stable Diffusion Inpainting | ❌ | ❌ | ✅ | ✅ |
| ControlNet-Inpainting | ✅ | ❌ | △ | ❌ |
| PowerPaint | ❌ | ❌ | ✅ | ✅ |
| **BrushNet** | ✅ | ✅ | ✅ | ✅ |

### 2.5 한계점

BrushNet은 LAION 데이터셋에서 학습되었기 때문에 일반적인 시나리오에서의 성능만 보장할 수 있으며, 제품 전시(product exhibition), 가상 피팅(virtual try-on) 등 고품질 산업 응용 분야에서는 자체 데이터로 재학습을 권장한다.

또한 노이즈 제거 과정에서 전체 이미지를 생성하는 특성상 배경의 일관성이 본질적으로 저해될 수 있으며, 단순한 블렌딩 연산은 조명이나 색상 불일치와 같은 문제를 해결하기 어렵다는 지적도 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 플러그 앤 플레이를 통한 범용 일반화

BrushNet은 플러그 앤 플레이이며, 콘텐츠 인식(content-aware), 형상 인식(shape-aware)하고, 비마스킹 영역의 보존 정도를 유연하게 조절할 수 있다.

다양한 이미지 도메인—자연 이미지, 연필 소묘(pencil painting), 애니메이션(anime), 일러스트레이션, 디지털 아트, 수채화—에 걸쳐 BrushNet과 기존 인페인팅 방법을 비교한 결과를 제시함으로써, 특정 도메인에 국한되지 않는 일반화 능력을 증명하였다.

이는 BrushNet이 랜덤 마스크, 내부 인페인팅 마스크, 외부 인페인팅 마스크를 포함한 다양한 마스크 유형의 광범위한 인페인팅 태스크에서 강한 성능을 보임을 의미한다.

### 3.2 데이터셋 다양성과 커스터마이징

세그멘테이션 마스크 체크포인트(segmentation_mask_brushnet_ckpt)는 객체 형상과 같은 세그멘테이션 사전 정보(segmentation prior)를 가진 마스크로 학습되었으며, 랜덤 마스크 체크포인트(random_mask_brushnet_ckpt)는 더 일반적인 랜덤 마스크 형상을 위한 체크포인트를 제공한다.

### 3.3 후속 확장(BrushNetX, BrushEdit)

2024년 12월 17일, BrushNetX(더 강력한 BrushNet)가 출시되었다.

BrushNet과 BrushEdit는 자연스러운 스토리를 형성하는데, BrushNet은 **어떻게 인페인팅할지(아키텍처)**를 해결하고, BrushEdit는 **무엇을 인페인팅할지(자연어를 마스크와 캡션으로 변환하는 인텔리전스 레이어)**를 해결한다.

### 3.4 비디오 인페인팅으로의 확장

BrushNet 아키텍처는 비디오 인페인팅에도 응용되고 있는데, VideoPainter는 BrushNet에서 영감을 받아 마스킹된 비디오를 처리하고 배경 가이던스를 사전 학습된 비디오 Diffusion Transformer에 주입하는 경량 컨텍스트 인코더를 특징으로 하는 이중 브랜치 프레임워크를 제안하여, 임의의 마스크 유형에 걸쳐 일반화를 달성한다.

DiffuEraser도 BrushNet의 이미지 인페인팅 아키텍처를 비디오 영역에 적용하고 있다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 주요 관련 연구 비교

| 연구 | 연도 | 방법 | 특징 | 한계 |
|---|---|---|---|---|
| **DDPM** (Ho et al.) | 2020 | 순수 Diffusion Model | 고품질 생성의 기반 | 인페인팅 미특화 |
| **Blended Diffusion** (Avrahami et al.) | 2022 | 샘플링 전략 수정 | 훈련 없음 | 의미론적 비일관성 |
| **Stable Diffusion Inpainting** (Rombach et al.) | 2022 | 전체 UNet 파인튜닝 | 전용 DM | 다른 DM 이식 불가 |
| **ControlNet** (Zhang et al.) | 2023 | 추가 인코더 브랜치 | 구조 조건 제어 | 인페인팅 미특화 |
| **HD-Painter** (Manukyan et al.) | 2023 | 훈련 없는 어텐션 개선 | SDI 기반 개선 | SDI 의존성 |
| **PowerPaint** (Zhuang et al.) | 2023/2024 (ECCV 2024) | 학습 가능한 태스크 프롬프트 | 다목적 인페인팅 | 이식성 제한 |
| **BrushNet** (Ju et al.) | 2024 (ECCV 2024) | 이중 브랜치 + Zero Conv | 플러그 앤 플레이 | 학습 데이터 도메인 한계 |

기존 ControlNet은 텍스트에 의존한 콘텐츠 완성을 위한 희소 구조적 제어에 의존하기 때문에 픽셀 수준의 인페인팅 이미지 특징 주입에 적합하지 않다.

PowerPaint는 학습 가능한 태스크 프롬프트와 맞춤형 파인튜닝 전략을 도입하여 모델이 다양한 인페인팅 대상에 집중할 수 있도록 함으로써, 다양한 인페인팅 태스크에서 최고 수준의 성능을 달성하는 첫 번째 다목적 고품질 인페인팅 모델이다.

그럼에도 불구하고 이러한 방법들은 복잡한 마스크 형상, 텍스트 프롬프트, 이미지 콘텐츠에서 전반적인 일관성 부족이라는 문제를 겪으며, 이는 주로 마스크 경계와 비마스킹 이미지 영역 컨텍스트에 대한 제한된 지각 지식에서 비롯된다.

---

## 5. 앞으로의 연구에 미치는 영향 및 연구 시 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

**① 플러그 앤 플레이 아키텍처 패러다임의 확산**

BrushNet은 Diffusion 프레임워크에 추가 브랜치를 도입하는 방식으로 이미지 인페인팅에 더 적합한 아키텍처를 구축하였으며, 이 이중 브랜치 + Zero Convolution 패러다임은 비디오 인페인팅, 이미지 편집 등 다양한 영역으로 영향을 주고 있다.

**② 사전 학습 모델 활용 가능성 극대화**

BrushNet은 사전 학습된 Diffusion Model에 플러그 앤 플레이로 적용 가능하며, 인페인팅 스케일에 대한 유연한 제어와 마스크 형상 및 비마스킹 콘텐츠 모두에 대한 인식 능력을 제공한다. 이는 앞으로 다양한 기반 모델(SDXL, FLUX 등)에 대한 이식 연구를 촉진할 것이다.

**③ 벤치마크의 표준화 기여**

세그멘테이션 기반 마스크 인페인팅 모델을 학습하고 평가하기 위해 제안된 BrushData와 BrushBench는, LAION-Aesthetic에 추가적인 세그멘테이션 마스크 어노테이션을 더한 것으로, 향후 연구자들이 공정하게 비교할 수 있는 표준 벤치마크로 기능하고 있다.

**④ 멀티모달 에이전트와의 결합**

BrushEdit는 BrushNet에 언어 이해 능력을 결합하여 자연어 명령을 이미지 편집으로 변환하는 에이전트 파이프라인으로, 이는 LLM 기반 이미지 편집 시스템 개발의 방향을 제시한다.

### 5.2 향후 연구 시 고려할 점

**① 도메인 특화 데이터 확보**

BrushNet은 LAION에서 학습되었기 때문에 일반 시나리오의 성능만 보장하며, 제품 전시나 가상 피팅 같은 고품질 산업 응용을 위해서는 해당 도메인의 자체 데이터로의 파인튜닝이 필요하다. 따라서 산업 응용 연구에서는 도메인 특화 데이터셋 구축이 선행되어야 한다.

**② 색상/조명 일관성 문제 해결**

기존의 단순한 블렌딩 연산은 조명이나 색상 불일치 문제를 충분히 해결하지 못한다. 향후 연구에서는 물리 기반 조명 모델이나 색상 적응 모듈을 BrushNet 프레임워크에 통합하는 시도가 필요하다.

**③ 고해상도 인페인팅으로의 확장**

BrushNet은 완전히 학습 가능한 U-Net을 포함한 이중 브랜치 아키텍처로 의미론적 효과를 증폭시키지만, 고해상도(2K, 4K) 이미지에서의 메모리 효율과 품질 유지를 위해 패치 기반 접근법이나 어텐션 효율화 기법과의 결합이 고려되어야 한다.

**④ 공정한 비교를 위한 평가 프로토콜 표준화**

PowerPaint의 비교 결과에는 편향이 있을 수 있는데, 이는 PowerPaint가 로컬 텍스트 설명으로 학습했지만 전역 텍스트 설명으로 테스트되어 완전히 공정한 비교가 아닐 수 있기 때문이다. 향후 연구자들은 동일한 텍스트 조건과 마스크 유형으로 공정하게 비교할 수 있는 평가 프로토콜 설계에 주의를 기울여야 한다.

**⑤ 비디오 및 3D로의 일반화**

BrushNet의 이중 브랜치 아이디어는 이미 VideoPainter 등의 비디오 인페인팅 연구에 영향을 주고 있으며, 향후 3D 포인트 클라우드나 NeRF 기반 장면 인페인팅으로의 확장 연구가 기대된다.

---

## 📚 참고 자료 및 출처

1. **arXiv 논문 원문:** Xuan Ju et al., "BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion," arXiv:2403.06976 (2024). https://arxiv.org/abs/2403.06976
2. **ECCV 2024 Springer 출판:** https://doi.org/10.1007/978-3-031-72661-3_9
3. **공식 GitHub 저장소:** https://github.com/TencentARC/BrushNet
4. **공식 프로젝트 페이지:** https://tencentarc.github.io/BrushNet/
5. **Hugging Face 논문 페이지:** https://huggingface.co/papers/2403.06976
6. **Semantic Scholar:** https://www.semanticscholar.org/paper/BrushNet:-A-Plug-and-Play-Image-Inpainting-Model-Ju-Liu/90c428ba9488c60bd860344d5a1299f01810ae51
7. **Unite.AI 분석 기사:** https://www.unite.ai/brushnet-plug-and-play-image-inpainting-with-dual-branch-diffusion/
8. **Yi's Blog – BrushNet & BrushEdit Explained:** https://wangyi.ai/blog/2026/02/07/brushnet-explained-inpainting/
9. **PowerPaint (비교 연구):** Zhuang et al., "A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting," ECCV 2024. arXiv:2312.03594
10. **ResearchGate – BrushNet 인용 현황:** https://www.researchgate.net/publication/386174419
11. **Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020.**
12. **Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022.**
