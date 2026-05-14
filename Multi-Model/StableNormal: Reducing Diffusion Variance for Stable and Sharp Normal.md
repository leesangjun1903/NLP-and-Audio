
# StableNormal: Reducing Diffusion Variance for Stable and Sharp Normal

> **논문 정보**
> - **저자**: Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, Xiaoguang Han
> - **게재지**: ACM Transactions on Graphics (TOG), 2024
> - **학회**: SIGGRAPH Asia 2024 (Journal Track)
> - **arXiv**: [2406.16864](https://arxiv.org/abs/2406.16864)

---

## 1. 핵심 주장과 주요 기여 요약

### 🔑 핵심 주장

이 연구는 단안 컬러 입력(이미지 및 비디오)으로부터 고품질 표면 법선(surface normal)을 추정하는 과제를 다루며, 기존 확산(Diffusion) 기반 접근법은 **확률적 추론(stochastic inference)** 문제와 비용이 큰 앙상블(ensembling) 단계 때문에 속도와 안정성 측면에서 한계를 보인다.

StableNormal은 확산 과정의 확률성을 완화하고 추론 분산을 줄여, **앙상블 없이도 "안정적이고 선명한(Stable-and-Sharp)"** 법선 추정을 달성한다.

### 📌 주요 기여 3가지

| # | 기여 | 설명 |
|---|------|------|
| 1 | **YOSO (You-Only-Sample-Once)** | 1-스텝 법선 추정기로 신뢰성 있는 초기 법선 생성 |
| 2 | **SG-DRN (Semantic-Guided Diffusion Refinement Network)** | DINO 의미론적 특징을 활용한 안정적인 정제 |
| 3 | **Shrinkage Regularizer** | 훈련 분산을 줄이는 새로운 정규화 기법 |

StableNormal은 단안 법선 추정을 위해 확산 사전(diffusion priors)을 조정하며, 기존 확산 기반 연구들과 달리 Stable Diffusion의 내재적 확률성을 줄여 추정 안정성을 향상시킴으로써 여러 기준선을 능가하는 "Stable-and-Sharp" 법선 추정을 가능하게 한다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

GeoWizard 같은 확산 기반 법선 추정기는 후처리 앙상블 없이는 **큰 분산을 가진 출력**을 생성하는 내재적 확률성 문제를 가지며, 반면 1-스텝 접근법은 마르코프 체인을 지나치게 단순화하여 지역 기하학적 세부사항을 평탄화하는 **과평활화(over-smoothing)** 문제를 야기한다. 따라서 확산 모델을 법선 추정 같은 **결정론적 작업에 재활용(repurpose)할 때 "안정성"과 "선명도" 사이의 트레이드오프**가 발생한다.

---

### 2-2. 제안하는 방법 (수식 포함)

StableNormal의 접근법은 Coarse-to-Fine 방식을 따른다: **(1) 신뢰할 수 있는 초기화를 위한 1-스텝 법선 추정**과 **(2) 의미론적으로 인식된 방향으로 법선 맵을 점진적으로 선명하게 하는 의미론적 가이드 확산 정제**로 구성된다.

---

#### 🔷 배경: 확산 모델의 기본 원리

확산 모델(DDPM)의 **순방향 과정(Forward Process)**은 원본 데이터 $x_0$에 단계적으로 가우시안 노이즈를 추가한다:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1},\ \beta_t \mathbf{I})
$$

임의의 타임스텝 $t$에서 직접 샘플링하면:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

여기서 $\bar{\alpha}_t = \prod\_{s=1}^t (1 - \beta_s)$이다.

**역방향 과정(Backward Process)**:

$$
x_{t-1} = B_t\, x_t - \mu_\theta(x_t, t)
$$

확산 확률 모델은 역방향 확산 과정을 통해 가우시안 분포를 순차적으로 변환하여 데이터 분포를 모델링하는 것을 목표로 한다.

---

#### 🔷 Stage 1: YOSO (You-Only-Sample-Once)

YOSO는 표면 법선의 초기 예측을 생성하는 **1-스텝 법선 추정기**로, 가우시안 노이즈 입력을 통해 확률적 요소를 도입하여 선명도와 안정성 간의 균형을 맞추며, 다중 스텝 확산 과정의 고분산 예측 대신 **결정론적 출력**을 목표로 한다.

**핵심 손실함수 — Shrinkage Regularizer (Noise-Decoupled MSE Loss)**:

Shrinkage Regularizer는 1-스텝 법선 추정기를 학습시키기 위해 도입되었으며, 바닐라 확산 손실을 **생성(generative) 항**과 **재구성(reconstruction) 항**으로 분리함으로써 훈련 분산을 줄인다.

이를 수식으로 표현하면 (논문에 기반한 구조적 해석):

$$
\mathcal{L}_{\text{YOSO}} = \mathcal{L}_{\text{gen}} + \lambda \cdot \mathcal{L}_{\text{rec}}
$$

- $\mathcal{L}_{\text{gen}}$: 생성 항 (확산 모델의 기본 노이즈 예측 목표)
- $\mathcal{L}_{\text{rec}}$: 재구성 항 (예측된 $\hat{x}_0$와 GT 법선 $x_0$ 간의 MSE)
- $\lambda$: 두 항 간의 균형 가중치

Shrinkage Regularizer는 출력의 과평활화를 방지하기 위해 통합되며, 예측된 법선 분포가 **디랙 델타 함수(Dirac delta function)에 가깝게 수렴**하도록 유도하여 평균 예측 주변에 출력을 효과적으로 집중시킨다.

이 1-스텝 추정기인 YOSO는 이미 현재 최신 기법인 DSINE과 동등한 수준의 성능을 달성한다.

---

#### 🔷 Stage 2: SG-DRN (Semantic-Guided Diffusion Refinement Network)

SG-DRN은 DINO 의미론적 사전(semantic priors)을 통합하여 확산 기반 정제 과정의 안정성을 강화하며, 이러한 사전이 **샘플링 분산을 줄이는 동시에 지역 세부사항을 향상**시킨다.

SG-DRN은 YOSO의 초기 예측과 동일한 공간 해상도로 정렬된 의미론적 특징을 결합하는 **U-Net 아키텍처**를 활용하며, 이를 통해 복잡한 기하학적 정보를 포함한 장면에서도 정확성이 향상된다. 추론은 잠재 공간에서 샘플링하는 **가이드 확산(guided diffusion)** 방식으로 처리된다.

SG-DRN의 조건부 생성 과정:

$$
p_\theta(x_{t-1} | x_t, c_{\text{sem}}, x_{t^+}) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, c_{\text{sem}}, x_{t^+}),\ \Sigma_\theta)
$$

여기서:
- $c_{\text{sem}}$: DINO에서 추출된 의미론적 특징
- $x_{t^+}$: YOSO로부터 얻은 신뢰할 수 있는 초기 법선

전체 파이프라인은 두 단계로 구성된다: **(1) YOSO는 새로운 Noise-Decoupled MSE Loss를 통해 신뢰할 수 있는 초기화를 생성**하고, **(2) SG-DRN은 DINOv2에서 추출한 강력한 의미론적 제어 정보를 활용하여 안정적인 디노이징**을 수행하며, 두 단계 모두 U-Net의 텍스트 프롬프트로 "The normal map"을 사용한다.

---

### 2-3. 모델 구조

```
입력 이미지 (Monocular RGB)
        │
        ▼
┌──────────────────────┐
│   Stage 1: YOSO      │  ← Stable Diffusion 기반 1-스텝 추정
│  (Shrinkage Reg.)    │  ← Noise-Decoupled MSE Loss
│  → 초기 법선 x_t+    │  ← 비교적 coarse하지만 신뢰성 있음
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   DINOv2             │  ← 의미론적 특징 추출
│  (Semantic Extractor)│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Stage 2: SG-DRN     │  ← U-Net 기반 의미 가이드 확산 정제
│  (Semantic-Guided    │  ← YOSO 출력 + DINO features 조합
│   Diffusion Refine.) │
└──────────┬───────────┘
           │
           ▼
  최종 법선 맵 (Sharp & Stable)
```

---

### 2-4. 성능 향상

StableNormal은 DSINE, Marigold, GeoWizard와 비교하여 4개의 실내 벤치마크(DIODE-indoor, iBims, ScanNetV2, NYUv2)에서 평가되었으며, **iBims, ScanNetV2, DIODE-indoor에서 큰 차이로 우수한 성능**을 달성했다. 다만 NYUv2에서는 DSINE 대비 약간 낮은 성능을 보였다.

YOSO 초기화 대신 DSINE의 초기화를 사용했을 때 DIODE-indoor 데이터셋에서 평균 각도 오차가 13.701°에서 18.453°로 증가하여, **YOSO 초기화의 필요성**을 검증했다.

DINO 추출기를 표준 ResNet-50으로 대체하면 모든 데이터셋에서 성능이 감소하며, DIODE-Indoor에서 평균 각도 오차가 13.701°에서 15.611°로 증가하여 **DINO 시각적 표현의 우수성**이 검증되었다.

| 구성 요소 | DIODE-indoor MAE (낮을수록 좋음) |
|---|---|
| **Full StableNormal** | **13.701°** |
| Ours w/o DINO (ResNet-50) | 15.611° |
| SG-DRN + DSINE Init | 18.453° |

---

### 2-5. 한계점

StableNormal은 1단계에서 단일 스텝 확산으로 거친 법선 맵을 생성하고, **2단계에서는 여전히 반복적 확산(iterative diffusion)으로 정제를 수행하는데 이는 계산 집약적**이라는 한계가 있다.

ScanNet과 NYUv2는 저품질 센서로 촬영되어 **GT(Ground Truth) 법선이 정확하지 않아** 해당 데이터셋에서의 정량적 평가에 불확실성이 존재한다.

추가로 공개 검토 의견을 참고하면:
- 실외(outdoor), 항공, 의료 영상 등 **도메인 외(out-of-domain) 시나리오에서의 일반화**는 명확히 검증되지 않음
- **실시간 추론**이 여전히 어려울 수 있음 (단, `StableNormal-turbo`로 10배 속도 향상 버전 제공)

---

## 3. 일반화 성능 향상 가능성

StableNormal은 극단적인 조명, 블러링, 저품질 등의 어려운 영상 조건 하에서도 강건하게 작동하며, **투명하고 반사적인 표면 및 많은 객체가 있는 혼잡한 장면**에서도 강건성을 보인다.

일반화 성능 향상 관련 핵심 요소:

### (1) DINOv2 의미론적 사전의 역할

의미론적 특징 추출기에 대한 대안이 있음에도 불구하고, DINO 추출기를 ResNet-50으로 교체하면 모든 데이터셋에서 성능이 저하되어 **법선 추정을 위한 의미론적 가이드로서 DINO 시각적 표현의 우수성**이 입증되었다.

대규모 데이터(LAION-5B)로 사전학습된 Stable Diffusion과 DINOv2를 활용함으로써:
- **다양한 도메인 지식**이 모델에 내재
- 새로운 장면 유형에 대한 **Zero-shot 일반화 기대 가능**

### (2) 확산 사전의 강력한 표현력

대규모 데이터셋으로 학습된 확산 기반 이미지 생성기의 발전은 기하학적 또는 내재적 단서(깊이, 법선, 재질 등)를 추정하기 위해 확산 사전을 재활용하는 방향으로 비전 커뮤니티의 관심을 이끌어냈다.

### (3) 다운스트림 태스크에서의 일반화

StableNormal의 결과는 법선 추정의 경계를 확장할 뿐 아니라, **단안 및 다중 뷰 표면 재구성**에서 특히 비-람베르시안(non-Lambertian) 표면에 대한 정확한 법선 매핑이 재구성을 크게 향상시키며, 생성 AI 프레임워크와의 통합을 통한 **3D 객체 생성 프로세스 개선**에서도 의미 있는 성능 향상을 보인다.

### (4) 일반화 성능 한계 및 개선 방향

StableNormal은 확산 사전을 결정론적 추정에 재활용하는 "초기적 시도(baby attempt)"로 자리매김한다. 이는 다음과 같은 일반화 개선 방향을 시사한다:

- **도메인 특화 Fine-tuning**: 의료영상, 위성 이미지 등 특수 도메인 적용 시 추가 훈련 필요
- **더 강력한 Semantic Extractor**: DINOv2 → SAM, CLIP 등 더 풍부한 의미론적 특징 활용
- **동영상 시퀀스 일관성**: 프레임 간 시간적 일관성(temporal consistency) 보장 메커니즘 추가

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 연구 | 방법론 | 특징 | 한계 |
|------|--------|------|------|
| **Omnidata** (Eftekhar et al., 2021) | CNN 회귀 기반 | 대규모 3D 스캔 데이터 활용 | 과평활화 문제 |
| **DSINE** (Bae & Davison, 2024) | 귀납적 편향 재고 | 최고 수준 회귀 기반 | 확산 사전 미활용 |
| **Marigold** (Ke et al., 2024) | 확산 모델 (다중 스텝) | 깊이 추정 확산 선구자 | 앙상블 필요, 느린 속도 |
| **GeoWizard** (Fu et al., 2024) | 확산 모델 | 시각적 선명도 개선 | 정량적 향상 미미, 고분산 |
| **StableNormal** (Ye et al., 2024) | YOSO + SG-DRN | 안정성+선명도 동시 달성 | 2단계로 인한 연산비용 |
| **Lotus** (He et al., 2024) | 확산 재구성 | SoTA 제로샷 깊이/법선 | StableNormal 대비 계산 효율 우위 주장 |

StableNormal은 두 단계 과정을 통해 법선 맵을 예측하며, 1단계는 단일 스텝 확산으로 거친 법선 맵을 생성하고 2단계는 반복적 확산으로 정제를 수행하는 계산 집약적 방식을 택한다.

Lotus는 이에 대응하여 제로샷 단안 깊이 및 법선 추정 두 가지 태스크에 걸쳐 광범위한 평가 데이터셋에서 SoTA 성능을 달성했으며, 단 59K 훈련 샘플로 강력한 확산 사전을 효과적으로 활용하여 놀라운 결과를 제공한다고 주장한다.

---

## 5. 향후 연구에 미치는 영향과 고려할 점

### 🔭 연구에 미치는 영향

1. **확산 → 결정론적 태스크 재활용 패러다임 정립**
   StableNormal은 확산 사전을 결정론적 추정에 재활용하는 초기적 시도로서 하나의 연구 방향을 제시한다. 이 패러다임은 깊이 추정, 재질 추정, 광학 흐름 등 다른 밀집 예측(dense prediction) 태스크로 확장 가능하다.

2. **Coarse-to-Fine 확산 정제 전략의 일반화**
   신뢰할 수 있는 초기화와 안정적인 정제의 결합이 선명하고 안정적인 법선 추정에 필수적임을 입증했다. 이 전략은 다른 기하학적 추정 문제에도 적용 가능한 일반적 프레임워크를 제공한다.

3. **의미론적 사전과 기하학 추정의 결합**
   DINO를 기하 추정의 가이드로 활용한 것은 **시각-언어 모델(VLM)을 기하 추정에 통합**하는 방향의 선구적 시도이다.

4. **공개 코드 및 모델의 기여**
   코드와 모델은 `hf.co/Stable-X`에서 공개적으로 사용 가능하며, 이는 후속 연구자들의 기준선(baseline) 활용을 용이하게 한다.

---

### ⚠️ 앞으로 연구 시 고려할 점

| 고려사항 | 내용 |
|---------|------|
| **연산 효율화** | SG-DRN의 반복적 확산 정제는 계산 비용이 크므로, 일관성 증류(consistency distillation) 등을 통한 가속 필요 |
| **시간적 일관성** | 비디오 입력에서 프레임 간 법선의 일관성 보장 메커니즘 연구 |
| **도메인 일반화 검증** | 실외, 의료, 산업 도메인 등 다양한 In-the-wild 환경에서의 검증 필요 |
| **GT 데이터 품질** | ScanNet, NYUv2 등은 저품질 센서로 촬영되어 GT 법선이 정확하지 않으므로 더 정밀한 GT 생성 방법 연구 필요 |
| **더 강력한 Semantic Prior** | DINOv2를 넘어 SAM2, GPT-4V 등 더 풍부한 의미론적 표현 탐색 |
| **하류 태스크(Downstream) 통합** | 3D Gaussian Splatting, NeRF 등 최신 3D 표현과의 긴밀한 통합 연구 |
| **불확실성 추정** | 현재 결정론적 출력 중심이므로, 예측 불확실성 정량화 연구 병행 필요 |
| **분산-선명도 트레이드오프 이론화** | 결정론적 추정 태스크에 확산 모델을 재활용할 때 발생하는 안정성-선명도 트레이드오프의 이론적 분석 심화 |

---

## 📚 참고자료 및 출처

| 번호 | 출처 |
|------|------|
| 1 | **arXiv 원문**: Chongjie Ye et al., "StableNormal: Reducing Diffusion Variance for Stable and Sharp Normal," arXiv:2406.16864, Jun. 2024. https://arxiv.org/abs/2406.16864 |
| 2 | **ACM TOG (SIGGRAPH Asia 2024)**: https://dl.acm.org/doi/10.1145/3687971 |
| 3 | **공식 프로젝트 페이지**: https://stable-x.github.io/StableNormal/ |
| 4 | **GitHub 공식 저장소**: https://github.com/Stable-X/StableNormal |
| 5 | **arXiv HTML 전문 (v1)**: https://arxiv.org/html/2406.16864v1 |
| 6 | **Hugging Face 논문 페이지**: https://huggingface.co/papers/2406.16864 |
| 7 | **The Moonlight Literature Review**: https://www.themoonlight.io/en/review/stablenormal-reducing-diffusion-variance-for-stable-and-sharp-normal |
| 8 | **Lotus 논문 (비교 연구)**: arXiv:2409.18124, "Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction," 2024. https://arxiv.org/html/2409.18124v2 |
| 9 | **ResearchGate**: https://www.researchgate.net/publication/381668850 |
| 10 | **Ho et al. (2020)**, "Denoising Diffusion Probabilistic Models," NeurIPS 33, 6840–6851. |
| 11 | **Bae & Davison (2024)**, "Rethinking Inductive Biases for Surface Normal Estimation," CVPR 2024. |
| 12 | **Fu et al. (2024)**, "GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation," ECCV 2024. |
| 13 | **Ke et al. (2024)**, "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation (Marigold)," CVPR 2024. |
| 14 | **Oquab et al. (2024)**, "DINOv2: Learning Robust Visual Features without Supervision." |

> ⚠️ **정확도 주의사항**: 수식 내 세부 하이퍼파라미터(예: $\lambda$ 구체적 값, 정확한 손실 함수 표현식)는 논문 전문 접근이 제한되어 개념적 구조로 재구성하였습니다. 정확한 수식 표현은 [공식 arXiv 전문](https://arxiv.org/abs/2406.16864) 및 [HTML 버전](https://arxiv.org/html/2406.16864v1)을 직접 참조하시길 권장합니다.
