
# DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation 

> **논문 정보**
> - **저자**: Chenguo Lin, Panwang Pan, Bangbang Yang, Zeming Li, Yadong Mu
> - **학회**: ICLR 2025 (accepted)
> - **arXiv**: [2501.16764](https://arxiv.org/abs/2501.16764) (2025.01.28)
> - **공식 코드**: [https://github.com/chenguolin/DiffSplat](https://github.com/chenguolin/DiffSplat)
> - **프로젝트 페이지**: [https://chenguolin.github.io/projects/DiffSplat/](https://chenguolin.github.io/projects/DiffSplat/)

---

## 1. 핵심 주장 및 주요 기여 (Executive Summary)

### 🔑 핵심 주장

DiffSplat은 대규모 text-to-image diffusion 모델을 활용하여 3D Gaussian splat을 직접 생성하는 새로운 3D 생성 프레임워크로, 웹 규모의 2D prior를 효과적으로 활용하면서 동시에 하나의 통합 모델에서 3D 일관성을 유지한다는 점이 핵심 주장이다.

DiffSplat은 텍스트 프롬프트 또는 단일 이미지로부터 1~2초 내에 3D Gaussian Splat을 생성하는 프레임워크로, 사전 훈련된 text-to-image diffusion 모델에서 직접 fine-tuning된다.

### 📌 주요 기여 요약

| 기여 항목 | 내용 |
|---|---|
| **스케일러블 데이터 큐레이션** | 경량 재구성 모델(GSRecon)로 빠른 pseudo 3D 데이터셋 생성 |
| **Splat Latent 설계** | Image VAE를 fine-tuning하여 Gaussian splat을 latent 공간에 인코딩 |
| **듀얼 손실 함수** | Diffusion loss + 3D Rendering loss 결합으로 3D 일관성 확보 |
| **이미지 커뮤니티와의 연결** | ControlNet 등 2D 기법을 3D 도메인에 seamless 적용 가능 |

---

## 2. 해결하고자 하는 문제

### 2.1 기존 연구의 한계

텍스트 또는 단일 이미지로부터의 최신 3D 콘텐츠 생성 연구들은 고품질 3D 데이터셋의 부족과 2D 멀티뷰 생성에서의 비일관성이라는 문제를 겪고 있다.

현재 주요 일반화 3D 콘텐츠 생성 방법들은 사전 학습된 2D 확산 모델을 이용해 생성된 멀티뷰 이미지에서 3D 표현을 재구성하는 2단계 방식인데, 이 방법은 업스트림의 멀티뷰 확산 모델에서 생성된 이미지 품질이 낮거나 비일관적일 경우 파이프라인 전체가 무너지는 문제가 있다.

크게 세 가지 문제를 해결하고자 한다:

1. **데이터 부족**: 고품질 3D 데이터셋의 절대적 부족
2. **2단계 파이프라인의 취약성**: 멀티뷰 이미지 생성 → 3D 재구성의 오류 전파
3. **웹 스케일 2D prior 미활용**: 방대한 2D 사전 지식이 3D 생성에 활용되지 못함

---

## 3. 제안하는 방법 (수식 포함)

DiffSplat은 세 가지 핵심 모듈로 구성된다.

### 3.1 모듈 1 — 구조적 Splat 재구성 (GSRecon)

구조화된 멀티뷰 Gaussian splat 그리드를 사용하여 3D 객체를 표현하며, $V_{in}$개의 포즈가 주어진 이미지 $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$로부터 소형 네트워크 $F_\theta$가 픽셀 단위로 splat을 회귀하며, 이를 0.1초 이내에 수행한다.

학습 단계에서 이 그리드는 멀티뷰 이미지에서 0.1초 이내에 즉시 회귀될 수 있어 확장 가능한 고품질 3D 데이터셋 큐레이션을 촉진하며, 2D 그리드의 각 Gaussian splat은 객체의 텍스처와 구조를 내포하는 속성을 가진다.

**GSRecon 학습 손실:**

$$\mathcal{L}_{\text{render}} = \sum_{v=1}^{V} \left( \lambda_{\text{MSE}} \cdot \left\| \hat{I}_v - I_v \right\|_2^2 + \lambda_{\text{LPIPS}} \cdot \mathcal{L}_{\text{LPIPS}}(\hat{I}_v, I_v) \right)$$

- $\hat{I}_v$: 뷰 $v$에서 렌더링된 예측 이미지
- $I_v$: 뷰 $v$의 실제 이미지
- $\lambda_{\text{LPIPS}}$: LPIPS 손실 가중치 (코드 기준 기본값 1.0)

비교적 훨씬 적은 파라미터를 가지면서도 좌표 맵과 법선 맵의 도움으로 제안된 경량 재구성 모델은 3D 생성을 위한 "의사(pseudo)" 정답 표현인 고품질 Gaussian splat 그리드를 즉시 제공할 수 있다.

---

### 3.2 모듈 2 — Splat Latents (GSVAE)

오픈소스 잠재 확산 모델들은 web-scale 이미지 데이터셋으로 훈련되어 있으며 이를 3D 콘텐츠 생성을 위한 Gaussian splat 속성 생성에 직접 재활용한다. 재구성된 splat 그리드가 이미지 확산 모델의 입력 잠재 변수와 같은 형태를 갖도록, 이들의 VAE를 fine-tuning하여 Gaussian splat 속성을 유사한 잠재 공간으로 압축하며 이를 "splat latents"라 부른다.

각 Gaussian splat은 다음의 속성 집합을 가진다:

$$\mathbf{g} = \left( \boldsymbol{\mu} \in \mathbb{R}^3,\ \mathbf{r} \in \mathbb{R}^4,\ \mathbf{s} \in \mathbb{R}^3,\ \sigma \in \mathbb{R},\ \mathbf{c} \in \mathbb{R}^3 \right)$$

- $\boldsymbol{\mu}$: 3D 위치 (position)
- $\mathbf{r}$: 회전 quaternion (rotation)
- $\mathbf{s}$: 스케일 (scale)
- $\sigma$: 불투명도 (opacity)
- $\mathbf{c}$: 색상 (color/spherical harmonics)

**GSVAE 인코딩:**

$$\mathbf{z}_{\text{splat}} = \mathcal{E}_{\text{GS}}(\mathbf{G}), \quad \hat{\mathbf{G}} = \mathcal{D}_{\text{GS}}(\mathbf{z}_{\text{splat}})$$

Gaussian splat 속성 그리드는 자연 이미지와 유사하여, 3D 객체의 색조 및 에지 속성을 반영한다. 이미지 VAE로부터 디코딩된 splat 잠재 변수들은 원본 객체를 "특별한 스타일" 또는 "특별한 환경 조명" 하에서 조명된 것으로 해석될 수 있다.

---

### 3.3 모듈 3 — DiffSplat 생성 모델 및 손실 함수

DiffSplat 훈련 중, splat 잠재 변수에 대한 표준 확산 손실(일반 이미지 확산 모델과 유사)에 더해, 임의 시점에서 Gaussian splat 속성이 네트워크에서 처리되고 미분 가능하게 렌더링될 수 있으므로 생성 모델이 3D 공간에서 작동하고 3D 일관성을 촉진하기 위한 렌더링 손실을 추가로 통합한다.

**전체 학습 목표 함수:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}} + \lambda \cdot \mathcal{L}_{\text{render}}$$

**Diffusion Loss (표준 DDPM/LDM 방식):**

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{\mathbf{z}, \boldsymbol{\epsilon} \sim \mathcal{N}(0,1), t} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c}) \right\|_2^2 \right]$$

- $\mathbf{z}_t$: 타임스텝 $t$에서의 노이즈 가해진 splat latent
- $\boldsymbol{\epsilon}_\theta$: 노이즈 예측 네트워크 (fine-tuned image diffusion backbone)
- $\mathbf{c}$: 텍스트 또는 이미지 조건

**3D Rendering Loss:**

```math
\mathcal{L}_{\text{render}} = \sum_{v} \left( \left\| \hat{I}_v - I_v^{*} \right\|_2^2 + \mathcal{L}_{\text{LPIPS}}(\hat{I}_v, I_v^{*}) \right)
```

- $\hat{I}_v$: 예측된 splat latent를 디코딩하여 임의 뷰 $v$로 렌더링한 이미지
- $I_v^{*}$: 해당 뷰의 실제 이미지

> ⚠️ **주의**: 위 수식들은 논문의 공식 표기를 기반으로 재현한 것이며, 논문의 세부 수식 표기와 일부 차이가 있을 수 있습니다. 정확한 수식은 [ICLR 2025 공식 논문 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/6518f9339196e172fa0ceef48a85543a-Paper-Conference.pdf)를 참조하시기 바랍니다.

---

### 3.4 카메라 포즈 인코딩 (Plücker Embedding)

Plücker 임베딩을 각 splat latent와 feature 차원을 따라 연결(concatenate)하여 상대적 카메라 포즈의 밀집 인코딩을 가능하게 한다. 이는 뷰포인트 선택에서 더 나은 유연성을 제공하고 멀티뷰 데이터셋에 대한 요구사항을 줄인다. 사전 훈련된 모델에 도입된 유일한 새 파라미터는 Plücker 임베딩을 위한 입력 컨볼루션 가중치의 제로 초기화된 새 열(column)뿐으로, 이러한 설계는 다양한 text-to-image 확산 모델에 대한 최소한의 수정과 일반화 가능성을 가져온다.

$$\text{Input} = \text{Concat}(\mathbf{z}_{\text{splat}},\ \text{Plücker}(\mathbf{R}, \mathbf{t}))$$

---

## 4. 모델 구조

```
[Input: Text / Image]
        │
        ▼
┌─────────────────────────────────────────────────┐
│           Stage 1: 데이터 큐레이션 (GSRecon)        │
│  Multi-view RGB + Normal + Coord maps           │
│        → Lightweight F_θ Network               │
│        → Multi-view Gaussian Splat Grids       │
│           (pseudo GT, <0.1s per object)         │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│           Stage 2: GSVAE (Splat Latents)        │
│  Gaussian Properties (μ, r, s, σ, c)           │
│        → Fine-tuned Image VAE Encoder           │
│        → Splat Latents z_splat (same shape      │
│           as image latents)                     │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│        Stage 3: DiffSplat Generation           │
│  Pretrained T2I Diffusion Model (SD1.5/SDXL/   │
│  PixArt-Σ/SD3.5m) + Plücker Embedding          │
│        ↓ L_diffusion + L_render (Tiny Decoder) │
│  Generated Splat Latents → GSVAE Decoder        │
│        → Final 3D Gaussian Splats              │
│        → Real-time Rendering (1~2s)            │
└─────────────────────────────────────────────────┘
```

2D 디노이징 네트워크 아키텍처에 대한 최소한의 수정 덕분에, 다양한 사전 훈련된 text-to-image 확산 모델이 DiffSplat의 베이스 모델로 활용될 수 있으며, 이들을 위해 제안된 기법들이 3D 생성의 영역에 seamlessly 적용될 수 있어 3D 콘텐츠 생성과 이미지 생성 커뮤니티 사이의 다리를 구축한다.

렌더링 손실을 효율적으로 수행하기 위해, DiffSplat 훈련 단계에서는 원본 디코더보다 훨씬 작은 크기의 Tiny Decoder를 훈련하며, 이 Tiny GSVAE 디코더는 DiffSplat 렌더링 손실에서만 사용되고 최종 추론에는 사용되지 않는다.

**지원 백본 모델:**
효율성을 위해서는 DiffSplat (SD1.5), 성능을 위해서는 DiffSplat (SD3.5m), 적절한 균형을 위해서는 DiffSplat (PixArt-Sigma)를 권장한다.

---

## 5. 성능 향상

방대한 실험을 통해 텍스트 및 이미지 조건 생성 작업과 다운스트림 애플리케이션에서 DiffSplat의 우월성이 확인되었으며, 철저한 ablation 연구를 통해 각 핵심 설계 선택의 효과와 기저 메커니즘에 대한 통찰이 검증되었다.

ControlNet이 DiffSplat에 seamlessly 적용되어 법선 맵, 깊이 맵, Canny 에지 등 다양한 형식으로 제어 가능한 텍스트-3D 생성이 가능하다.

매우 얇은 물체를 묘사하는 프롬프트로 생성된 멀티뷰 Gaussian latent의 3D 일관성을 평가한 결과, 제안된 방법이 두드러진 아티팩트나 왜곡 없이 3D 일관된 얇은 물체에 대한 멀티뷰 splatter 이미지를 생성할 수 있음을 보였다.

**훈련 데이터셋:**
G-Objaverse의 약 265K 3D 객체와 10.6M 렌더링 이미지(265K × 40 뷰: RGB, 법선, 깊이 맵 포함)를 GSRecon 및 GSVAE 훈련에 사용하며, LGM이 제공하는 약 83K 3D 객체의 서브셋이 DiffSplat 훈련에 사용된다.

---

## 6. 한계점

DiffSplat은 객체 중심(object-centric) 방법으로, 장면 수준(scene-level)의 재구성에는 적용이 제한된다.

이미지 조건 생성 시, 텍스트 조건 DiffSplat은 입력 이미지 뷰 외의 다른 뷰를 단순히 디노이징하는 것이 불가능한데, 이는 입력 조건(픽셀)과 생성된 출력(splat 속성)이 서로 다른 도메인에 있기 때문이다.

대규모 객체 데이터셋에서의 도메인 갭으로 인해, 생성된 출력들이 합성적이고 가상적으로 보이며, 실제 세계의 텍스처 및 기하학과의 눈에 띄는 차이가 존재한다.

또한 이러한 방법들이 특히 대규모의 실측 3D 장면 데이터가 없는 경우, 더 크고 복잡한 장면 수준의 재구성 작업에 얼마나 잘 확장될 수 있는지는 여전히 불분명하다.

정리하면:

| 한계 유형 | 내용 |
|---|---|
| **도메인 제한** | 객체 중심(object-centric), 장면 수준 생성 어려움 |
| **이미지 조건 생성** | 픽셀-속성 도메인 불일치로 인한 추가 설계 필요 |
| **데이터 편향** | G-Objaverse 기반 학습으로 합성 데이터 편향 존재 |
| **텍스트 민감도** | SD3.5m 모델이 프롬프트 품질에 민감 |

---

## 7. 모델의 일반화 성능 향상 가능성

### 7.1 웹 스케일 2D Prior 활용에 의한 일반화

2D 디노이징 네트워크 아키텍처에 대한 최소한의 수정 덕분에 다양한 사전 훈련된 text-to-image 확산 모델이 DiffSplat의 베이스 모델로 사용될 수 있으며, 이를 위해 제안된 기법들은 프레임워크 내에서 3D 생성 영역에 seamlessly 적용될 수 있어 3D 콘텐츠 생성과 이미지 생성 커뮤니티 사이의 다리를 구축한다.

웹 스케일로 사전 훈련된 이미지 확산 모델이 깊이, 좌표, 법선 등 3D 기하 속성 추정에 효과적이라는 사실에서 착안하여, DiffSplat은 이미지 확산 모델을 직접 3D 콘텐츠 생성에 활용하는 것을 목표로 한다.

### 7.2 Plücker 임베딩에 의한 뷰포인트 일반화

Plücker 임베딩은 뷰포인트 선택에서 더 나은 유연성을 제공하고 멀티뷰 데이터셋에 대한 요구사항을 줄이며, 사전 훈련된 모델에 도입된 유일한 새 파라미터는 제로 초기화된 Plücker 임베딩용 입력 컨볼루션 가중치의 새 열뿐으로, 이러한 설계는 다양한 text-to-image 확산 모델에 대한 최소한의 수정과 일반화 가능성을 가져온다.

### 7.3 스케일러블 데이터 큐레이션에 의한 일반화

학습을 부트스트랩하기 위해 멀티뷰 Gaussian splat 그리드를 즉시 생성할 수 있는 경량 재구성 모델을 통해 확장 가능한 데이터셋 큐레이션을 제안한다.

데이터 필터링이 DiffSplat의 생성 품질에 결정적이며, 더 큰 데이터셋이 GSRecon 및 GSVAE의 성능에 유리하다는 것이 발견되었다.

### 7.4 다양한 T2I 백본 호환성

SD1.5, SDXL, PixArt-Sigma, SD3.5 등 웹 스케일 이미지 데이터셋으로 훈련된 오픈소스 잠재 확산 모델들이 3D 콘텐츠 생성을 위한 Gaussian splat 속성 생성에 직접 재활용된다.

이는 미래의 더 강력한 T2I 모델(예: FLUX, Stable Diffusion 4 등)로 백본을 교체할 경우 자동으로 일반화 성능이 향상될 가능성을 내포한다.

---

## 8. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방식 | 3D 표현 | 2D Prior | 장점 | 단점 |
|---|---|---|---|---|---|
| **NeRF** (2020) | 최적화 기반 | Implicit MLP | ✗ | 높은 품질 | 매우 느린 속도 |
| **3D-GS** (2023, Kerbl et al.) | 최적화 기반 | Gaussian Splat | ✗ | 빠른 렌더링 | 인스턴스별 최적화 |
| **Splatter Image** (2023) | Feed-forward | Gaussian Splat | ✗ | 빠름 | 단일 이미지, 단일 Gaussian |
| **Zero-1-to-3** (2023) | 2D MV 확산 | NeRF | ✓ | 단일 이미지 입력 | 2단계, 불일관성 |
| **LGM** (2024) | Feed-forward | Gaussian Splat | 일부 | 빠른 생성 | 멀티뷰 일관성 부족 |
| **DiffGS** (2024) | 잠재 확산 | Gaussian (연속 함수) | ✗ | 임의 개수 생성 | 2D prior 미활용 |
| **Direct3D** (2024) | 3D 잠재 확산 | Triplane | ✗ | 고품질 | 2D prior 제한적 |
| **DiffSplat** (2025, 본 논문) | 2D 확산 fine-tuning | Gaussian Splat Grid | ✓ (웹 스케일) | 빠름 + 일관성 + 확장성 | Object-centric 한계 |

DiffGS는 3DGS의 이산적이고 비구조적인 특성으로 인한 Gaussian Splatting 생성의 어려움을 해결하기 위해 잠재 확산 모델 기반의 일반 Gaussian 생성기를 제안하며, 임의의 수의 Gaussian 프리미티브를 생성할 수 있다. 핵심 아이디어는 Gaussian 확률, 색상, 변환을 모델링하는 세 가지 새로운 함수를 통해 Gaussian Splatting을 분리된 방식으로 표현하는 것이다.

**DiffSplat의 차별점:**
DiffSplat은 세 가지 방식(직접 최적화, feed-forward 재구성, 2D 다단계 생성)의 장점을 계승하여, 사전 훈련된 이미지 확산 모델을 직접 3DGS 생성에 fine-tuning함으로써 2D 확산 prior를 효과적으로 활용하면서 3D 일관성을 유지한다.

---

## 9. 앞으로의 연구에 미치는 영향과 고려 사항

### 9.1 연구에 미치는 영향

#### ① 2D-3D 연결 패러다임의 정착
이미지 확산 모델과의 호환성은 이미지 생성을 위한 수많은 기법들이 3D 영역에 seamlessly 적용될 수 있게 한다.

이는 향후 연구에서 2D 기술 발전이 자동으로 3D 생성에 이익을 줄 수 있는 새로운 패러다임을 정착시킨다.

#### ② 데이터 큐레이션 방법론의 기여
경량 재구성 모델이 고품질의 구조화된 Gaussian 표현을 "의사(pseudo)" 데이터셋 큐레이션을 위해 제공한다는 접근 방식은 향후 다양한 3D 생성 연구에서 데이터 수집 비용을 획기적으로 줄이는 데 기여할 것이다.

#### ③ 후속 연구로의 직접적 영향
DiffSplat의 직계 후속 연구로 단일 이미지에서 4D 동적 장면 생성(Diff4Splat, CVPR 2026), 단안 비디오에서 4D 동적 재구성(MoVieS, CVPR 2026), 여러 부품으로 3D 객체를 직접 생성하는 3D-native DiT(PartCrafter, NeurIPS 2025) 등이 파생되었다.

### 9.2 향후 연구 시 고려 사항

#### 🔵 장면 수준(Scene-level) 일반화
대규모 커버리지, 멀티 스케일 다양성(객체 중심에서 실내/실외 장면), 풍부한 멀티모달 정보를 동시에 제공하는 3D 데이터셋이 없어 이것이 일반화 가능한 3D 모델 개발의 주요 장애물이 되고 있으며, 기존 데이터셋들은 다양한 장면에서의 일반화 가능하고 멀티모달 학습을 가능하게 하기에 부족하다.

→ 따라서 DiffSplat을 장면 수준으로 확장하기 위해서는 대규모 장면 데이터 수집 및 장면-객체 혼합 학습 전략이 필요하다.

#### 🔵 동적(Dynamic) 4D 생성으로의 확장
Diff4Splat(CVPR 2026)과 같이 단일 이미지에서 4D 동적 장면을 생성하는 방향이 후속 연구로 이어지고 있는 만큼, 시간적 일관성 유지와 동적 Gaussian splat의 latent 표현 설계가 핵심 도전 과제가 될 것이다.

#### 🔵 실세계 도메인 갭 해소
대규모 객체 데이터셋에서의 도메인 갭으로 인해 생성된 출력들이 합성적이고 가상적으로 보이며 실제 세계의 텍스처 및 기하학과의 차이가 존재하므로, 실세계 이미지로 학습하거나 도메인 적응 기법 도입이 필요하다.

#### 🔵 더 강력한 백본 모델로의 전환
이러한 설계는 다양한 text-to-image 확산 모델에 대한 최소한의 수정과 일반화 가능성을 가져오므로, FLUX.1, Stable Diffusion 3.5 Large 등 최신 대형 T2I 모델로 백본을 교체하는 연구가 일반화 성능 개선의 직접적 방법이 될 것이다.

#### 🔵 멀티모달 조건 확장
텍스트와 이미지를 동시에 조건으로 지원하는 현재 방식에서 나아가, 포인트 클라우드, 깊이 맵, 3D 스케치 등 더 다양한 조건부 입력으로의 확장도 의미 있는 연구 방향이다.

---

## 📚 참고 자료 (References)

| 번호 | 제목 및 출처 |
|---|---|
| **[1]** | Lin, C. et al., *"DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation"*, ICLR 2025. arXiv: 2501.16764 — [https://arxiv.org/abs/2501.16764](https://arxiv.org/abs/2501.16764) |
| **[2]** | 공식 GitHub 저장소 — [https://github.com/chenguolin/DiffSplat](https://github.com/chenguolin/DiffSplat) |
| **[3]** | 공식 프로젝트 페이지 — [https://chenguolin.github.io/projects/DiffSplat/](https://chenguolin.github.io/projects/DiffSplat/) |
| **[4]** | OpenReview (ICLR 2025) — [https://openreview.net/forum?id=eajZpoQkGK](https://openreview.net/forum?id=eajZpoQkGK) |
| **[5]** | HuggingFace Paper Page — [https://huggingface.co/papers/2501.16764](https://huggingface.co/papers/2501.16764) |
| **[6]** | ICLR 2025 공식 논문 PDF — [https://proceedings.iclr.cc/paper_files/paper/2025/file/6518f9339196e172fa0ceef48a85543a-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/6518f9339196e172fa0ceef48a85543a-Paper-Conference.pdf) |
| **[7]** | arXiv HTML 전문 — [https://arxiv.org/html/2501.16764v1](https://arxiv.org/html/2501.16764v1) |
| **[8]** | Zhou et al., *"DiffGS: Functional Gaussian Splatting Diffusion"*, arXiv 2410.19657 — [https://arxiv.org/abs/2410.19657](https://arxiv.org/abs/2410.19657) |
| **[9]** | Liao et al., *"Complete Gaussian Splats from a Single Image with Denoising Diffusion Models"*, arXiv 2508.21542 — [https://arxiv.org/abs/2508.21542](https://arxiv.org/abs/2508.21542) |
| **[10]** | Kerbl et al., *"3D Gaussian Splatting for Real-Time Radiance Field Rendering"*, SIGGRAPH 2023 |
| **[11]** | Moonlight Literature Review — [https://www.themoonlight.io/en/review/diffsplat-repurposing-image-diffusion-models-for-scalable-gaussian-splat-generation](https://www.themoonlight.io/en/review/diffsplat-repurposing-image-diffusion-models-for-scalable-gaussian-splat-generation) |

> **⚠️ 정확도 관련 유의사항**: 본 답변은 논문의 공개된 abstract, HTML 전문, GitHub 코드, OpenReview 자료를 기반으로 작성되었습니다. 논문 내부의 일부 세부 실험 수치(PSNR, FID 등 정량 비교)는 검색된 소스에서 직접 확인되지 않아 의도적으로 기술하지 않았습니다. 정확한 수치 비교를 위해서는 반드시 공식 논문 PDF를 직접 참조하시기 바랍니다.
