
# AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers

> **논문 정보**
> - **저자**: Sherwin Bahmani\*, Ivan Skorokhodov\*, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B. Lindell, Sergey Tulyakov (\*equal contribution)
> - **소속**: University of Toronto, Vector Institute, Snap Inc., SFU
> - **arXiv**: [2411.18673](https://arxiv.org/abs/2411.18673) (2024.11.27)
> - **게재**: CVPR 2025 (pp. 22875–22889)

---

## 1. 핵심 주장과 주요 기여 요약

최근 수많은 연구들이 3D 카메라 제어를 기반 텍스트-투-비디오 모델에 통합해왔지만, 그 결과는 카메라 제어 정밀도가 낮고 비디오 생성 품질이 저하되는 문제를 안고 있었다.

AC3D는 이 문제를 세 가지 원칙적 분석(First-principles Analysis)을 통해 해결하며 다음과 같은 핵심 주장과 기여를 제시합니다:

| 기여 | 핵심 발견 | 효과 |
|---|---|---|
| ① 저주파 카메라 모션 분석 | 카메라 모션은 저주파 성질 | 학습 수렴 가속 + 품질 향상 |
| ② 레이어별 카메라 정보 탐지 | 초기 레이어만 카메라 정보 보유 | 파라미터 4× 감소, 시각 품질 10% 향상 |
| ③ 정적 카메라 데이터 보강 | 장면 동적성과 카메라 혼동 방지 | 동적 씬 생성 품질 향상 |

이 발견들을 결합하여 Advanced 3D Camera Control (AC3D) 아키텍처를 설계하였으며, 이는 카메라 제어를 포함한 생성적 비디오 모델링의 새로운 SOTA(State-of-the-Art)가 되었다.

---

## 2. 해결하고자 하는 문제, 제안하는 방법(수식 포함), 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

이 논문은 텍스트와 카메라 시퀀스 모두에 컨디셔닝된 비디오 생성 품질을 향상시키기 위해 Video Diffusion Transformer 모델에 정밀한 3D 카메라 제어를 통합하는 문제를 다룬다. 특히 기존 모델에 만연한 세 가지 문제를 탐구한다: 카메라 제어의 낮은 정밀도, 비디오 생성 품질 저하, 그리고 합성 품질을 저하시키지 않고 더 나은 컨디셔닝을 이끌어내는 방법.

또한 인터넷 규모 데이터로 학습된 기반 비디오 확산 모델(VDM)들은 물리 세계에 대한 풍부한 지식을 습득하며, 외형과 그럴싸한 2D 동역학뿐 아니라 3D 구조에 대한 이해도 갖고 있다. 그러나 이 지식의 대부분은 모델 내부에 암묵적으로 저장되어 있으며, 카메라 모션 제어와 같은 세밀한 제어 메커니즘을 노출하지 않는다.

---

### 2.2 제안하는 방법 (세 가지 원칙적 분석)

#### ✅ Analysis 1: 카메라 모션의 저주파 특성 (Motion Spectral Volume, MSV)

저자들은 생성된 비디오에서 카메라에 의한 모션이 주로 저주파 성질을 가짐을 확인했다. 이 발견은 Motion Spectral Volumes (MSV) 분석을 통해 도출되었으며, MSV는 비디오 생성 중 주파수 스펙트럼의 여러 영역에 걸쳐 에너지가 어떻게 분산되는지를 보여준다.

MSV는 주파수 스펙트럼의 서로 다른 부분에 에너지 양을 표시하며(낮은 주파수의 높은 에너지는 부드러운 모션을 의미), 200개의 서로 다른 유형(카메라 모션, 씬 모션, 씬+카메라 모션)의 생성 비디오와 다양한 노이즈 제거 합성 단계에서 측정된다. 카메라 모션은 스펙트럼의 하위 부분에 주로 영향을 미치며, 노이즈 제거 궤적의 초반 약 10% 시점에 나타난다.

이를 바탕으로 노이즈 스케줄을 다음과 같이 조정합니다:

$$
\text{Train-time noise schedule:} \quad t \sim \mathcal{N}_{\text{trunc}}\bigl(\mu=0.8,\ \sigma=0.075,\ [0.6,\ 1.0]\bigr)
$$

$$
\text{Inference-time camera conditioning:} \quad t \in [0.6,\ 1.0] \text{ (first 40\% of reverse diffusion)}
$$

구체적으로, SD3의 표준 로짓-정규 노이즈 레벨 분포(위치 0.0, 스케일 1.0) 대신, 직류 정류 흐름(rectified flow) 노이즈 제거 궤적의 초기 단계를 커버하기 위해 위치 0.8, 스케일 0.075의 [0.6, 1] 구간 절단 정규 분포로 전환한다. 추론 시에도 동일한 [0.6, 1] 구간에서 카메라 컨디셔닝을 적용한다.

이 통찰을 따라, 훈련 시 노이즈 레벨과 테스트 시 카메라 컨디셔닝 스케줄 모두 역방향 확산 궤적의 처음 40%만 커버하도록 제한한다.

---

#### ✅ Analysis 2: 레이어별 카메라 정보 분포 (Linear Probing)

두 번째 분석은 무조건부(unconditional) 비디오 확산 트랜스포머(VDiT)의 내부 표현을 탐침(probing)하여 카메라 포즈 추정을 암묵적으로 인코딩하는지 평가한다. 이는 선형 탐침(linear probing)을 통해 달성되었으며, 카메라 정보가 가장 두드러지는 네트워크 세그먼트를 효과적으로 분리한다. 분석 결과 모델의 초기 레이어(특히 9~21번째 블록 사이)에 카메라 정보가 집중적으로 분리되어 있음이 밝혀졌다.

이에 따라:

$$
\text{Camera conditioning} \rightarrow \text{Only first } K \text{ blocks (e.g., } K=8 \text{ out of 32 DiT blocks)}
$$

이 통찰을 따라 AC3D는 처음 8개 블록에서만 컨디셔닝을 적용한다. 32개 전체 DiT 블록에 컨디셔닝을 시도하면(카메라 컨디셔닝을 8개 블록으로 제한하지 않을 때) 시각 품질이 약 10% 저하되며, 동시에 카메라 제어 품질도 동일하게 유지된다. 이는 중간 및 후기 VDiT 레이어가 이미 처리된 카메라 정보에 의존하기 때문에, 외부 카메라 포즈로 컨디셔닝하면 다른 특성과 간섭할 수 있음을 시사한다.

---

#### ✅ Analysis 3: 정적 카메라 데이터 보강

이 논문은 일반적으로 사용되는 데이터셋(예: RealEstate10K)이 주로 정적인 장면을 포함하고 있어, 이러한 제약 조건에서 훈련된 모델의 성능이 효과적이지 않다는 중요한 한계를 강조한다.

이를 해결하기 위해:

카메라 제어 학습을 위한 일반적인 데이터셋을 정적 카메라를 가진 20K의 다양하고 동적인 비디오로 구성된 큐레이션 데이터셋으로 보완한다. 이는 모델이 카메라와 씬 모션 사이의 차이를 구분하는 데 도움을 주며, 포즈 컨디셔닝된 비디오의 동적성을 개선한다.

---

### 2.3 모델 구조 (VDiT-CC)

AC3D는 VD3D에서 처음 도입된 Plücker 좌표 기반 ControlNet 아키텍처를 따르는 카메라 제어 비디오 생성 파이프라인이다.

**Plücker 임베딩 공식:**

카메라 레이(ray)를 다음과 같이 표현합니다:

$$
\mathbf{p} = (\mathbf{d}, \mathbf{o} \times \mathbf{d}) \in \mathbb{R}^6
$$

여기서 $\mathbf{d}$는 레이 방향, $\mathbf{o}$는 카메라 원점, $\times$는 외적(cross product)입니다. 각 픽셀과 프레임에 대해 Plücker 좌표를 계산한 후 비디오 잠재 공간(latent)에 추가 입력으로 사용합니다.

VDiT-CC 모델은 VDiT 위에 ControlNet 카메라 컨디셔닝을 구축한 것이다. 비디오 합성은 동결된(frozen) VDiT 백본의 대형 4,096차원 DiT-XL 블록에 의해 수행되며, VDiT-CC는 경량 128차원 DiT-XS 블록을 통해 카메라 정보를 처리하고 주입한다(FC는 완전 연결 레이어를 의미).

**모델 구조 요약:**

```
[카메라 Plücker 임베딩]
        ↓
[경량 Camera Branch: 128-dim DiT-XS × 8개 블록]
        ↓ (주입, 처음 8 블록에만)
[동결된 VDiT Backbone: 4096-dim DiT-XL × 32개 블록]
        ↓
[생성 비디오]
```

**Noise Conditioning Schedule:**

$$
\tilde{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_\text{cam}) = \epsilon_\theta(\mathbf{x}_t, t) + w \cdot \bigl[\epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}_\text{cam}) - \epsilon_\theta(\mathbf{x}_t, t)\bigr], \quad t \in [0.6, 1.0]
$$

여기서 $w$는 카메라 CFG(Classifier-Free Guidance) 스케일, $\mathbf{c}_\text{cam}$은 카메라 Plücker 임베딩 시퀀스입니다.

---

### 2.4 성능 향상

AC3D는 약 25% 더 정밀한 카메라 스티어링을 달성한다.

VD3D+FIT(원본 모델) 및 VD3D+DiT(더 큰 비디오 트랜스포머 기반 개선 재구현)와의 사용자 연구를 수행했으며, AC3D는 모든 정성적 측면에서 두 모델을 능가하여 90% 이상의 전체 선호도 점수를 달성했다.

이 접근법은 훈련 파라미터의 4배 감소, 향상된 훈련 속도, 10% 더 높은 시각 품질을 가져왔다.

**일반화 성능:** 이는 FID와 FVD를 MSR-VTT(파인튜닝 분포 외 다양한 씬을 측정하는 데이터셋)에서 약 30% 향상시킨다.

---

### 2.5 한계점

저자들은 논문의 한계를 부록 B에서 논의하고 있으며, 향후 연구에서 데이터 한계를 추가로 개선하고 훈련 분포 외의 카메라 궤적에 대한 제어 메커니즘을 개발할 계획이라고 밝혔다.

AC3D는 RealEstate10K 데이터셋 훈련으로 인해 해당 데이터의 전반적인 미적 특성을 유지하지만, 불균형 프레이밍과 낮은 다이나믹 레인지(과노출 창문 등) 문제가 있다.

또한 큰 카메라 궤적 편차를 처리할 때 기하학적 왜곡이나 일관성 없는 객체 경계가 나타나는 경우가 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 핵심 메커니즘

**① 노이즈 스케줄 조정을 통한 일반화**

역방향 확산 궤적의 처음 40%로만 훈련 및 추론 시 카메라 컨디셔닝 스케줄을 제한함으로써, MSR-VTT(파인튜닝 분포 외 다양한 씬을 측정하는 데이터셋)에서 FID와 FVD를 약 30% 향상시킨다.

이것이 일반화에 핵심적인 이유는, 저주파 단계에서만 카메라 정보를 주입함으로써 **모델이 나머지 노이즈 제거 단계에서 기반 모델(VDiT)의 원래 생성 능력을 온전히 보존**하기 때문입니다.

**② 선택적 레이어 컨디셔닝을 통한 일반화**

무조건부 비디오 확산 트랜스포머의 표현을 탐침함으로써, 이 모델들이 내부적으로 카메라 포즈 추정을 암묵적으로 수행하며, 레이어의 일부분만 카메라 정보를 포함함을 확인했다. 이는 카메라 컨디셔닝 주입을 아키텍처의 하위 집합으로 제한하여 다른 비디오 특성과의 간섭을 방지하도록 유도했으며, 훈련 파라미터의 4배 감소, 향상된 훈련 속도, 10% 더 높은 시각 품질로 이어졌다.

초기 레이어에만 카메라 컨디셔닝을 제한함으로써, 후기 레이어의 **도메인 불변 표현(domain-invariant representations)**이 보존되어 분포 외(out-of-distribution) 씬에 대한 일반화 능력이 유지됩니다.

**③ 정적 카메라 데이터 보강을 통한 일반화**

2D 데이터를 이용한 공동 훈련을 통해 시각 품질 및 씬 모션 저하를 완화하려 했으며, 카메라 입력에 드롭아웃을 적용하여 기반 VDiT 훈련에서 사용된 카메라 어노테이션 없는 2D 비디오 데이터에 대해서도 공동 훈련을 수행했다.

이를 통해 모델이 다양한 씬 콘텍스트에서 카메라와 씬 모션을 구분하는 능력을 일반화할 수 있습니다.

**일반화 성능에 관련된 수식 정리:**

$$
\mathcal{L}_\text{AC3D} = \mathbb{E}_{\mathbf{x}_0, t \sim \mathcal{N}_\text{trunc}(0.8, 0.075, [0.6,1])}\left[\left\|\epsilon - \epsilon_\theta\!\left(\mathbf{x}_t, t, \mathbf{c}_\text{cam}, \mathbf{c}_\text{text}\right)\right\|^2\right]
$$

여기서 고노이즈 구간 $t \in [0.6, 1.0]$에 집중함으로써 저주파 카메라 모션만 학습하고, 나머지 미세한 고주파 특성(장면 디테일, 질감 등)은 기반 모델이 처리합니다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

이 연구 결과는 비디오 확산 모델에서 카메라 모션에 대한 원칙적 분석이 제어 정밀도와 효율성에서 상당한 개선으로 이어진다는 것을 보여준다. 향상된 컨디셔닝 스케줄, 타겟화된 레이어별 카메라 제어, 더 잘 보정된 훈련 데이터를 통해 AC3D는 높은 시각 품질과 자연스러운 씬 동역학을 유지하면서 3D 카메라 제어 비디오 합성에서 SOTA 성능을 달성한다. 이 연구는 텍스트-투-비디오 생성에서 더 정밀하고 효율적인 카메라 제어를 위한 기반을 확립한다.

**구체적인 영향 영역:**

1. **후속 연구에서의 확장**: AC3D의 한 버전이 CogVideoX 위에 구축되었으며, 이는 다양한 기반 모델로의 이식 가능성을 보여줍니다.

2. **비교 연구의 기준점**: UCPE는 파인튜닝 없이 잘 일반화하여 가장 낮은 회전·평이·모션 오류를 달성하고, RealEstate10K(CameraCtrl 및 AC3D)에서 훈련된 모델보다 높은 Q-Align 점수를 보여준다. 이처럼 AC3D는 후속 연구들의 중요한 비교 기준(baseline)으로 기능합니다.

3. **방법론적 영감**: 카메라 제어 성능이 일반적으로 믿어지는 카메라 포즈 표현보다 컨디셔닝 방법의 선택에 크게 의존한다는 연구 결과를 바탕으로, Camera Motion Guidance(CMG)를 도입하여 카메라 제어를 400% 이상 향상시켰다. AC3D의 분석 방법론이 이러한 후속 연구를 촉진했습니다.

---

### 4.2 향후 연구 시 고려할 점

#### 🔴 데이터 관련 한계 극복

향후 연구에서는 데이터 한계를 더욱 개선하고 훈련 분포 외의 카메라 궤적에 대한 제어 메커니즘을 개발하는 데 집중할 계획이다.

- **더 다양한 카메라 궤적 데이터** 수집 및 자동 어노테이션 파이프라인 개발
- 실내 중심(RealEstate10K) 외 야외, 항공, 거리 씬 등 다양한 도메인의 데이터 확보

#### 🔴 일반화 성능의 추가 향상

UCPE는 이 데이터셋에서 파인튜닝되지 않았음에도 불구하고 상대적 카메라 포즈 제어에서 모든 베이스라인을 능가하며, 보지 못한 궤적과 텍스트 프롬프트에 대한 강한 일반화를 보여준다. 이는 **훈련 없는(training-free) 또는 파라미터 효율적인 카메라 제어** 방법론 개발이 중요한 방향임을 시사합니다.

#### 🔴 카메라-씬 모션 분리

카메라 궤적의 각 부분을 엄격히 따르고 더 나은 비디오 동적성을 가져야 한다. AC3D는 궤적 끝부분의 전방 카메라 움직임을 무시하는 경우가 있다. 복잡한 합성 카메라-씬 모션을 처리하기 위한 **모션 분리 알고리즘** 연구가 필요합니다.

#### 🔴 카메라 표현의 확장

UCPE의 한계로서, 훈련 중 카메라 포즈에 의존하며 현재 포즈, 내부 파라미터, 왜곡만 모델링하고 줌, 초점, 피사계 심도 등 더 풍부한 속성은 캡처하지 않는다. UCPE를 이러한 추가 제어로 확장하고 정확한 포즈 감독에 대한 의존도를 줄이는 것이 향후 유망한 연구 방향이다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 방법 | 아키텍처 | 핵심 특징 | AC3D 대비 |
|---|---|---|---|---|---|
| **MotionCtrl** | 2024 | 독립/공동 모션 제어기 | U-Net 기반 | 카메라+객체 모션 독립 제어 | 카메라 정밀도 낮음 |
| **CameraCtrl** | 2024 | Plücker 임베딩 | U-Net 기반 | 카메라 포즈 파라미터 활성화 | DiT 적용 시 정확도 저하 |
| **VD3D** | 2024 | Plücker ControlNet | DiT 기반 | 최초 DiT용 카메라 제어 | AC3D의 기반 아키텍처 |
| **Boosting CMG** | 2024 | Camera Motion Guidance | DiT 기반 | CFG 방식으로 카메라 400% 향상 | 광범위 모델 실험 미비 |
| **AC3D** | 2024/2025 | MSV+Layer Probing+데이터 보강 | DiT 기반 | 원칙적 분석 기반 설계 | **SOTA** |
| **UCPE** | 2024 | Unified Camera Positional Encoding | DiT 기반 | 파인튜닝 없이 일반화 | 파인튜닝 불필요, 일반화 우수 |
| **CameraCtrl II** | 2025 | 클립 단위 순차 생성 | DiT 기반 | 동적 씬 장거리 탐색 | 동적 씬 궤적 추종 우수 |
| **ACD** | 2025 | Attention Supervision | DiT 기반 | Attention Map 직접 제어 | 기하학적 왜곡 개선 |

U-Net 기반 모델들은 카메라 제어에서 유망한 결과를 보였지만, 대규모 비디오 생성에 선호되는 아키텍처인 트랜스포머 기반 확산 모델(DiT)은 카메라 모션 정확도에서 심각한 저하를 겪는다. 이 논문은 이 문제의 근본 원인을 조사하고 DiT 아키텍처에 맞는 해결책을 제안한다.

---

## 📚 참고자료 및 출처

1. **[주 논문]** Bahmani et al., "AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers," CVPR 2025, pp. 22875–22889.
   - arXiv: https://arxiv.org/abs/2411.18673
   - CVPR Open Access: https://openaccess.thecvf.com/content/CVPR2025/html/Bahmani_AC3D_...
   - 프로젝트 페이지: https://snap-research.github.io/ac3d/
   - GitHub: https://github.com/snap-research/ac3d
   - IEEE Xplore: https://ieeexplore.ieee.org/document/11093255/
   - HuggingFace: https://huggingface.co/papers/2411.18673

2. **[비교 논문 1]** "Boosting Camera Motion Control for Video Diffusion Transformers" (2024), arXiv: https://arxiv.org/html/2410.10802v1

3. **[비교 논문 2]** "CameraCtrl II: Dynamic Scene Exploration via Camera-Controlled Video Generation" (2025), arXiv: https://arxiv.org/pdf/2503.10592

4. **[비교 논문 3]** "Unified Camera Positional Encoding for Controlled Video Generation (UCPE)" (2024), arXiv: https://arxiv.org/html/2512.07237v1

5. **[비교 논문 4]** "ACD: Direct Conditional Control for Video Diffusion Models via Attention Supervision" (2025), arXiv: https://arxiv.org/html/2512.21268

6. **[리뷰]** Moonlight Literature Review, "AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers": https://www.themoonlight.io/en/review/ac3d-analyzing-and-improving-3d-camera-control-in-video-diffusion-transformers

7. **[ResearchGate]** https://www.researchgate.net/publication/386335876_AC3D_Analyzing_and_Improving_3D_Camera_Control_in_Video_Diffusion_Transformers
