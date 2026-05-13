
# Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation

> **논문 정보**
> - **저자**: Haotong Lin, Sida Peng, Jingxiao Chen, Songyou Peng, Jiaming Sun, Minghuan Liu, Hujun Bao, Jiashi Feng, Xiaowei Zhou, Bingyi Kang
> - **학술지/학회**: CVPR 2025, pp. 17070–17080
> - **arXiv**: [arXiv:2412.14015](https://arxiv.org/abs/2412.14015)
> - **프로젝트 페이지**: [promptda.github.io](https://promptda.github.io/)
> - **GitHub**: [DepthAnything/PromptDA](https://github.com/DepthAnything/PromptDA)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

프롬프트(Prompt)는 언어 및 비전 파운데이션 모델의 잠재력을 특정 태스크에 맞게 극대화하는 데 핵심적인 역할을 한다. 이 논문은 **최초로 깊이 파운데이션 모델에 프롬프팅 개념을 도입**하여, **Prompt Depth Anything**이라는 메트릭 깊이 추정의 새로운 패러다임을 제안한다.

구체적으로, 저가형 LiDAR를 프롬프트로 사용하여 Depth Anything 모델이 **정확한 메트릭 깊이(Metric Depth)**를 출력하도록 유도하며, **최대 4K 해상도**를 달성한다.

### 🏆 주요 기여

| # | 기여 |
|---|------|
| 1 | Depth 파운데이션 모델에 프롬프팅을 적용한 새로운 메트릭 깊이 추정 패러다임 제안 |
| 2 | 다중 스케일 Prompt Fusion 아키텍처 설계 |
| 3 | 확장 가능한 학습 데이터 파이프라인(합성 LiDAR 시뮬레이션 + Pseudo GT 생성) |
| 4 | Edge-Aware Depth Loss 제안 |
| 5 | ARKitScenes, ScanNet++ 벤치마크 SOTA 달성 |

이 접근법은 ARKitScenes 및 ScanNet++ 데이터셋에서 새로운 SOTA를 달성하였으며, 3D 재구성과 범용 로봇 파지(Robotic Grasping) 등 다운스트림 애플리케이션에도 기여한다.

---

## 2. 해결 문제 · 제안 방법 · 모델 구조 · 성능 및 한계

### 2-1. 해결하고자 하는 문제

최근 모노큘러 깊이 추정(Monocular Depth Estimation)은 모델 및 데이터의 스케일링을 통해 비약적인 발전을 이루며 깊이 파운데이션 모델들이 등장하였다. 그러나 이 모델들은 **고품질의 상대적 깊이(Relative Depth)**를 출력하는 데는 강점을 보이지만, **스케일 모호성(Scale Ambiguity)**으로 인해 자율주행 및 로봇 조작 등 실용적 적용에 한계가 있다.

Prompt Depth Anything을 학습하기 위해서는 LiDAR 깊이와 정밀한 GT 깊이가 동시에 필요하다. 그러나 기존 합성 데이터는 LiDAR 깊이가 없고, LiDAR가 있는 실제 데이터는 에지(Edge) 품질이 낮은 부정확한 GT 깊이만을 제공한다는 **데이터 부재 문제**가 있다.

---

### 2-2. 제안 방법 및 수식

#### (A) 전체 파이프라인 개요

Prompt Depth Anything은 **ViT 인코더와 DPT 디코더**로 구성된 깊이 파운데이션 모델을 기반으로 하며, 각 스케일에서 메트릭 정보를 융합하는 **다중 스케일 Prompt Fusion 블록**을 추가한다.

#### (B) Prompt Fusion 아키텍처

핵심은 DPT 기반 깊이 파운데이션 모델에 맞춰 설계된 **간결한 Prompt Fusion 아키텍처**이다. 이 아키텍처는 **DPT 디코더 내 다중 스케일에서 LiDAR 깊이를 통합**하여 깊이 디코딩을 위한 LiDAR 피처를 융합한다. 메트릭 프롬프트가 **정밀한 공간 거리 정보**를 제공함으로써, 깊이 파운데이션 모델이 **로컬 형상 학습기(Local Shape Learner)**로 기능하게 되어 정확하고 고해상도의 메트릭 깊이 추정이 가능하다.

구체적으로, DPT 프레임워크의 각 스케일에서 LiDAR 깊이를 피처 맵의 공간 크기에 맞게 조정하고, 얕은 합성곱 네트워크(Shallow CNN)로 깊이 피처를 추출한다. 이렇게 추출된 피처는 이미지 입력과 동일한 피처 공간으로 투영되어 깊이 디코딩 단계에서 효과적으로 통합된다.

이 설계는 **인코더와 디코더를 파운데이션 모델로부터 초기화**하고, 제안된 Fusion 아키텍처는 **Zero-Initialize**하여 초기 출력이 파운데이션 모델과 동일하도록 보장한다.

**계산 오버헤드:**

이 설계는 원래 깊이 파운데이션 모델 대비 **단 5.7%의 추가 계산 오버헤드**(756×1008 이미지 기준 1.789 TFLOPs vs. 1.691 TFLOPs)만을 도입하면서 깊이 파운데이션 모델의 스케일 모호성 문제를 효과적으로 해결한다.

#### (C) 스케일 가능한 데이터 파이프라인

이 도전을 해결하기 위해, **합성 데이터에 대한 저해상도·노이즈 LiDAR 시뮬레이션**과 **재구성 기법을 사용하여 고품질 에지를 가진 Pseudo GT 깊이 생성**을 포함하는 확장 가능한 데이터 파이프라인을 제안한다.

- **합성 데이터**: 정밀한 GT 깊이가 있는 합성 데이터에 LiDAR 깊이를 시뮬레이션
- **실제 데이터**: LiDAR가 있는 실제 데이터에 Zip-NeRF를 활용한 Pseudo GT 깊이 생성

특히, Zip-NeRF와 같은 재구성 기법을 통해 실세계 관측으로부터 Pseudo GT 깊이를 생성한다.

#### (D) Edge-Aware Depth Loss (수식)

Pseudo GT 깊이의 3D 재구성 오류를 완화하기 위해, 에지에서 두드러지는 **Pseudo GT 깊이의 그래디언트만을 활용하는 Edge-Aware Depth Loss**를 도입한다.

Edge-Aware Loss의 핵심 수식은 다음과 같이 구성된다:

$$
\mathcal{L}_{\text{edge-aware}} = \mathcal{L}_{L1} + \lambda \cdot \mathcal{L}_{\text{grad}}
$$

여기서:
- $\mathcal{L}_{L1}$: FARO 어노테이션 GT 깊이를 사용한 전반적인 깊이 학습을 위한 L1 손실
- $\mathcal{L}_{\text{grad}}$: Pseudo GT 깊이의 그래디언트를 활용하는 에지 정보 손실
- $\lambda$: 가중치 하이퍼파라미터 (논문에서 $\lambda = 0.5$로 설정)

깊이 그래디언트는 주로 에지에서 두드러지며, 이는 Pseudo GT 깊이가 탁월한 부분이기도 하다. 그래디언트 손실은 모델이 Pseudo GT 깊이로부터 정확한 에지를 학습하도록 유도하고, L1 손실은 전반적인 깊이를 학습하게 하여 우수한 깊이 예측으로 이어진다.

깊이 정규화(Depth Normalization)의 경우, 입력 LiDAR 깊이를 기준으로 깊이를 정규화하여 프롬프트 스케일에 대응하는 정규화된 깊이를 출력하도록 설계된다:

$$
\hat{d} = \frac{d}{d_{\text{LiDAR}}}
$$

학습 과정의 두 단계:

학습 과정은 2단계로 구성된다. 먼저 **워밍업 단계**에서 모델이 LiDAR 프롬프트 스케일에 대응하는 정규화 깊이를 출력하도록 조정한다. 이후 **종합 학습 단계**에서 다양한 합성·실제 데이터셋을 활용하여 파라미터를 반복적으로 정제한다.

---

### 2-3. 모델 구조 상세

```
[입력 RGB 이미지] + [저해상도 LiDAR 깊이]
         ↓
[ViT Encoder (DINOv2 기반, Depth Anything V2에서 초기화)]
         ↓
[DPT Decoder (4개 스케일)]
    ├── Scale 1: Prompt Fusion Block ← LiDAR Feature (Upsampled)
    ├── Scale 2: Prompt Fusion Block ← LiDAR Feature (Upsampled)
    ├── Scale 3: Prompt Fusion Block ← LiDAR Feature (Upsampled)
    └── Scale 4: Prompt Fusion Block ← LiDAR Feature (Upsampled)
         ↓
[메트릭 깊이 출력 (최대 4K 해상도)]
```

Adaptive LayerNorm, CrossAttention, ControlNet 등 다른 설계들도 실험하였으나, 이 설계들은 제안된 Fusion Block보다 성능이 낮은 것으로 나타났다. 이는 이들이 텍스트 프롬프트 등 이종 모달리티(Cross-modal) 정보를 통합하도록 설계되어, 입력 저해상도 LiDAR와 출력 깊이 간의 픽셀 정렬(Pixel Alignment) 특성을 효과적으로 활용하지 못하기 때문이다.

**추론 속도:**

ViT-L 모델은 A100 GPU에서 768×1024 해상도로 **20.4 FPS**를 달성한다. ARKit6가 4K 이미지 촬영을 지원함에 따라, **2160×3840 해상도에서 2.0 FPS**를 달성한다.

ViT-S 모델로도 구현 가능하며, 해당 속도는 각각 80.0 FPS와 10.3 FPS이다.

---

### 2-4. 성능 향상

방법론은 모든 데이터셋과 메트릭에서 지속적으로 SOTA 성능을 보인다. 심지어 Zero-Shot 모델도 다른 비-Zero-Shot 방법들과 비교하여 더 나은 성능을 달성하며, **파운데이션 모델 프롬프팅의 일반화 능력**을 입증한다.

이 방법은 일관된 깊이 추정을 가능하게 하여 부정확한 스케일과 비일관성으로 고통받는 Metric3D v2의 한계를 극복하며, ARKit LiDAR Depth(240×320)를 크게 능가하는 **4K 정밀 깊이 추정**을 달성한다.

또한, Prompt Depth Anything의 파운데이션 모델과 프롬프트를 **DepthPro 및 Vehicle LiDAR로 대체 가능**함을 보여 확장성을 검증하였다.

**평가 메트릭**: $\text{AbsRel}$, $\text{RMSE}$, $\delta_1$(정확도, $\delta < 1.25$)

$$
\text{AbsRel} = \frac{1}{N}\sum_{i=1}^{N} \frac{|d_i - \hat{d}_i|}{d_i}
$$

$$
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(d_i - \hat{d}_i)^2}
$$

---

### 2-5. 한계점

이 연구에는 몇 가지 알려진 한계가 존재한다. 예를 들어, iPhone LiDAR를 프롬프트로 사용할 경우 **장거리 깊이 처리가 불가능**하다(iPhone LiDAR는 먼 물체에 대해 매우 노이즈가 많은 깊이를 감지하기 때문). 또한, **LiDAR 깊이의 시간적 플리커링(Temporal Flickering)**이 관찰되어 깊이 예측의 플리커링으로 이어진다.

센서 한계 측면에서, 저가형 LiDAR를 이용한 프롬프팅은 2m 이상의 거리에서 성능이 저하되며, 센서 불안정성으로 인한 시간적 깊이 플리커가 발생한다.

---

## 3. 모델 일반화 성능 향상 가능성

### 3-1. 파운데이션 모델 프롬프팅의 일반화 효과

비록 모델이 실내 장면으로만 학습되었지만, 새로운 방, 얇은 구조물이 있는 체육관, 조명이 어두운 박물관, 인간 및 야외 환경 등 **다양한 시나리오에서 잘 일반화**된다. 이는 깊이 파운데이션 모델 프롬프팅의 효과성을 강조한다.

Zero-shot 모델 $\text{Ours}_{\text{syn}}$은 ScanNet++에서 모든 비-Zero-shot 모델들보다 더 나은 성능을 달성하며, **깊이 파운데이션 모델 프롬프팅의 일반화 능력**을 강조한다.

### 3-2. 다양한 센서 및 모델로의 확장성

이 모델은 DPT 구조를 활용하는 다른 깊이 파운데이션 모델(예: Depth Pro)에도 쉽게 적용 가능한 범용 DPT 설계이다. 실험 결과, Depth Pro의 성능도 크게 향상되었음을 보였으나, Depth Anything 기반 선택이 더 우수한 성능을 보였다.

### 3-3. 야외 환경 일반화

차량 LiDAR를 메트릭 프롬프트로 사용한 야외 재구성 실험도 수행하여 옥외 환경으로의 일반화 가능성을 확인하였다.

### 3-4. PromptDA++에서의 패턴-무관 프롬프팅(Pattern-Agnostic Prompting)

후속 연구인 PromptDA++ (arXiv v3)에서는 일반화를 한층 더 강화하였다:

SfM(Structure-from-Motion) 포인트를 프롬프트로 사용하여 **MVS(Multi-View Stereo) 재구성**에 대한 프롬프팅 메커니즘의 잠재력을 조사하였다. 놀랍게도, 이 방법은 어떠한 추가 파인튜닝 없이도 선도적인 MVS 방법들을 큰 차이로 능가하여, **패턴-무관 프롬프팅 접근법의 범용 효과성**을 입증하였다.

입력 깊이 포인트를 토큰으로 직렬화하고 자기-어텐션(Self-Attention)을 사용하여 깊이 파운데이션 모델의 이미지 토큰을 강화하는 **새로운 프롬프팅 메커니즘**을 제안한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. 계보 및 주요 관련 연구

| 연구 | 연도 | 핵심 접근법 | 특징 |
|------|------|------------|------|
| **MiDaS** | 2020 | 다중 데이터셋 학습 | 상대 깊이, 스케일 모호성 |
| **ZoeDepth** | 2022 | 상대→메트릭 변환 | Bin-based metric head |
| **Depth Anything V1** | 2024 | 62M+ 대규모 비라벨 데이터 | DINOv2 기반 파운데이션 모델 |
| **Metric3D v2** | 2024 | 기하학적 재투영 | 스케일/이동 불일치 문제 존재 |
| **Depth Anything V2** | 2024 | 합성→실세계 증류 | 정밀한 상대 깊이, 메트릭 fine-tune 가능 |
| **Depth Pro** | 2024 | 단일 이미지 메트릭 깊이 | DPT 구조, 고해상도 |
| **Prompt Depth Anything** | 2024 | LiDAR 프롬프팅 | 4K 메트릭 깊이, Zero-shot 일반화 |

### 4-2. Depth Anything V2와의 관계

Depth Anything V2는 V1 대비, 1) 모든 라벨링된 실제 이미지를 합성 이미지로 교체, 2) 교사 모델 용량 확대, 3) 대규모 Pseudo-라벨링 실제 이미지를 통한 학생 모델 학습이라는 세 가지 핵심 방식을 통해 훨씬 더 정밀하고 강건한 깊이 예측을 생성한다.

강력한 일반화 능력을 기반으로 메트릭 깊이 라벨로 파인튜닝하여 메트릭 깊이 모델을 얻는다.

그러나 Prompt Depth Anything은 이 Depth Anything V2를 백본으로 삼되, LiDAR 프롬프트를 통해 **메트릭 정확도와 4K 고해상도**라는 두 가지를 동시에 달성한다는 점에서 차별화된다.

### 4-3. 기존 깊이 완성(Depth Completion) 방법과의 차이

최근 연구들은 깊이 완성을 위한 학습 기반 접근법을 채택하고 있다. 그러나 일반적으로 이 방법들은 실제 실내 LiDAR 데이터가 아닌 **시뮬레이션된 희소 LiDAR에 대해서만 테스트**되었다는 한계가 있다.

반면, Prompt Depth Anything은 **실제 iPhone LiDAR를 사용하여 학습·평가**하며, 파운데이션 모델의 강력한 사전 지식을 프롬프트 방식으로 활용한다는 근본적 차이가 있다.

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5-1. 연구에 미치는 영향

**① 새로운 연구 패러다임 제시**

이 논문은 **메트릭 깊이 추정의 새로운 패러다임**을 제시하였다. 깊이 파운데이션 모델에 메트릭 정보를 프롬프팅하는 방식의 실현 가능성을 저가형 LiDAR 깊이를 프롬프트로 선택하여 검증하였다.

**② 다운스트림 태스크 파급 효과**

이 방법은 기존 모노큘러 깊이 추정 및 깊이 완성/업샘플링 방법에 대한 우월성을 입증하였으며, **3D 재구성 및 범용 로봇 파지**를 포함한 다운스트림 태스크에도 기여한다.

**③ 파운데이션 모델-센서 융합의 새 방향**

이러한 기술들은 신속한 적응, 향상된 메트릭 충실도, 그리고 사전 훈련된 깊이 파운데이션 모델의 기본 능력을 넘어서는 **향상된 일반화**를 가능하게 한다.

### 5-2. 향후 연구 시 고려할 점

**① 시간적 일관성 문제 해결**

향후 연구 방향으로는 **시간적 프롬프트 안정성(Cross-frame Filtering/Attention)**, 멀티모달 프롬프트(내재적 파라미터, IMU) 통합, 고주파 세부 정보와 불확실성 정량화를 위한 **생성/확산 스타일 기하학적 헤드와의 융합**이 있다.

**② 다중 모달 프롬프트 확장**

불확실성, 멀티모달 입력(열화상, 레이더), 동적 장면 진화를 처리하도록 프롬프트 설계를 확장하면 적용 가능성이 넓어질 수 있다.

**③ 패턴-무관 프롬프팅의 심화 연구**

패턴-무관 프롬프팅 접근법의 범용 효과성을 보여주는 SfM 포인트를 프롬프팅 입력으로 사용하여 **어떠한 추가 파인튜닝 없이도** MVS 재구성에서 SOTA 성능을 달성한 사례는, 다양한 포인트 클라우드 소스로의 확장 연구를 자극할 것이다.

**④ 경량화 및 엣지 디바이스 배포**

LoRA 및 프롬프트-인젝션 어댑터와 같은 파라미터 효율적 방법들은 SOTA 성능을 파라미터 효율적으로 달성할 수 있음을 보여주며, 이는 모바일 및 엣지 디바이스 배포에 중요한 고려사항이다.

**⑤ 원거리 깊이 추정 한계 극복**

장거리 씬에서의 LiDAR 노이즈 문제를 해결하기 위해, 다중 LiDAR 센서 융합이나 레이더(Radar)와의 결합, 또는 IMU 정보 활용이 유망한 연구 방향으로 고려될 수 있다.

**⑥ 프롬프트 타입 다양화**

현재 연구는 LiDAR와 SfM 포인트에 국한되어 있으나, RGB-D 카메라, TOF 센서, 심지어 **언어(텍스트) 프롬프트와의 결합**이 가능한지에 대한 연구도 향후 중요한 방향이 될 수 있다.

---

## 📚 참고 자료

| # | 출처 |
|---|------|
| 1 | **Lin, H. et al.** (2025). *Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation*. CVPR 2025, pp. 17070–17080. [arXiv:2412.14015](https://arxiv.org/abs/2412.14015) |
| 2 | **프로젝트 페이지**: [promptda.github.io](https://promptda.github.io/) |
| 3 | **CVPR 2025 Open Access**: [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_Prompting_Depth_Anything_for_4K_Resolution_Accurate_Metric_Depth_Estimation_CVPR_2025_paper.html) |
| 4 | **GitHub**: [DepthAnything/PromptDA](https://github.com/DepthAnything/PromptDA) |
| 5 | **arXiv HTML (v1)**: [arxiv.org/html/2412.14015v1](https://arxiv.org/html/2412.14015v1) |
| 6 | **PromptDA++ (v3)**: [arxiv.org/html/2412.14015v3](https://arxiv.org/html/2412.14015v3) |
| 7 | **HuggingFace Paper Page**: [huggingface.co/papers/2412.14015](https://huggingface.co/papers/2412.14015) |
| 8 | **The Moonlight Literature Review**: [themoonlight.io](https://www.themoonlight.io/en/review/prompting-depth-anything-for-4k-resolution-accurate-metric-depth-estimation) |
| 9 | **Yang, L. et al.** (2024). *Depth Anything V2*. NeurIPS 2024. [arXiv:2406.09414](https://arxiv.org/abs/2406.09414) |
| 10 | **Depth Anything V2 Project**: [depth-anything-v2.github.io](https://depth-anything-v2.github.io/) |
| 11 | **IEEE Xplore**: [ieeexplore.ieee.org/document/11095248](https://ieeexplore.ieee.org/document/11095248/) |

> ⚠️ **정확도 참고**: 본 답변의 수식 중 일부(정규화 수식, 손실 함수의 구체적 λ 값)는 논문의 공개된 HTML 버전 및 보충 자료를 기반으로 작성되었으며, 논문 본문에서 직접 확인 가능한 내용은 인용 처리하였습니다. 수식의 완전한 세부 사항은 원문 PDF를 직접 참고하시기를 권장합니다.
