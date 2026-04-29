
# StableV2V: Stablizing Shape Consistency in Video-to-Video Editing

> **논문 정보**
> - **제목**: StableV2V: Stablizing Shape Consistency in Video-to-Video Editing
> - **저자**: Chang Liu, Rui Li, Kaidong Zhang, Yunwei Lan, Dong Liu (USTC)
> - **arXiv**: 2411.11045 (2024년 11월 17일)
> - **게재**: IEEE TCSVT 2025 채택 확정

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 문제 의식

기존 비디오 편집 연구들은 주로 원본 영상의 내재적 모션 패턴을 편집된 영상으로 전달하는 방식을 취하는데, 전달된 모션과 편집된 콘텐츠 간의 특별한 정렬(alignment) 부재로 인해 사용자 프롬프트와의 일관성이 떨어지는 결과물이 자주 발생한다.

즉, 사용자가 "곰(bear)"을 "너구리(raccoon)"로 바꾸는 프롬프트를 입력하면, 두 객체의 **형태(shape)가 다르기 때문에** 기존 방법들은 원본의 모션 패턴을 새로운 객체 형태에 맞게 조정하지 못해 시각적 불일치가 발생한다.

### 1.2 핵심 주장

StableV2V는 특히 사용자 프롬프트가 편집 콘텐츠에 큰 형태 변화를 초래하는 편집 시나리오를 처리하면서 형태 일관성을 유지하는 방식으로 비디오 편집을 수행하는 새로운 패러다임을 제시한다.

### 1.3 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **새로운 편집 패러다임** | Shape-consistent Video Editing 파이프라인 |
| **3단계 모듈 구조** | PFE → ISA → CIG |
| **새로운 평가 벤치마크** | DAVIS-Edit 구축 |
| **멀티모달 프롬프트 지원** | 텍스트·이미지·스케치 등 다양한 입력 지원 |

또한, StableV2V는 다양한 모달리티의 사용자 프롬프트를 고려하여 광범위한 다운스트림 애플리케이션을 처리하는 데 있어 우수한 유연성을 보인다.

이 논문은 2025년 TCSVT에 게재 승인되었다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 연구들은 주로 원본 영상의 내재적 모션 패턴을 편집된 영상으로 전달하는데, 전달된 모션과 편집 콘텐츠 간의 특정 정렬 부재로 인해 특히 편집된 객체와 원본 객체 사이에 형태 변화가 발생할 수 있는 경우, 사용자 의도와의 불일치 문제가 발생하는 열등한 결과물을 자주 생성한다.

이를 도식화하면 다음과 같다:

$$
\text{문제}: \underbrace{M_{\text{source}}}_{\text{원본 모션}} \xrightarrow{\text{단순 전달}} \underbrace{V_{\text{edit}}}_{\text{편집 영상}} \Rightarrow \text{Shape Mismatch}
$$

기존 방법들은 원본 객체의 깊이 맵(depth map)이나 모션 특징을 그대로 편집된 객체에 적용하려 하지만, 객체의 형태가 달라지면 해당 신호들이 편집된 객체의 실제 형상을 반영하지 못한다.

### 2.2 제안하는 방법 및 모델 구조

StableV2V는 전체 비디오 편집 프로세스를 이미지 편집과 모션 전달로 분해하는 첫 프레임 기반 방법에 기반하며, **Prompted First-frame Editor (PFE)**, **Iterative Shape Aligner (ISA)**, **Conditional Image-to-video Generator (CIG)**의 세 가지 주요 컴포넌트로 비디오 편집 작업을 처리한다.

```
[입력 비디오]
     │
     ▼
┌─────────────────────────┐
│  PFE (첫 프레임 편집기)   │  ← 사용자 프롬프트 (텍스트/이미지/스케치)
│  - InstructPix2Pix       │
│  - Paint-by-Example      │
│  - SD Inpainting         │
└─────────┬───────────────┘
          │ 편집된 첫 프레임 (Î₁)
          ▼
┌─────────────────────────┐
│  ISA (반복 형태 정렬기)   │  ← 원본 깊이 맵 시퀀스 + 형태 마스크
│  - RAFT (광학 흐름)      │
│  - MiDaS (깊이 추정)     │
│  - U²-Net (형태 마스크)  │
│  - Shape-guided Depth    │
│    Refinement Network    │
└─────────┬───────────────┘
          │ 형태 정렬된 깊이 맵 시퀀스
          ▼
┌─────────────────────────┐
│  CIG (조건부 I2V 생성기) │
│  - I2VGen-XL             │
│  - ControlNet (Depth)    │
│  - Ctrl-Adapter          │
└─────────┬───────────────┘
          │
          ▼
     [편집된 비디오]
```

#### 2.2.1 PFE (Prompted First-frame Editor)

PFE는 사용자 프롬프트를 편집 콘텐츠로 변환하는 첫 프레임 이미지 편집기 역할을 하며, 이후의 절차를 통해 전체 비디오로 전파된다.

수식으로 표현하면:

$$
\hat{I}_1 = \mathcal{F}_{\text{PFE}}(I_1, p)
$$

여기서 $I_1$은 원본 첫 번째 프레임, $p$는 사용자 프롬프트, $\hat{I}_1$은 편집된 첫 번째 프레임이다.

#### 2.2.2 ISA (Iterative Shape Aligner)

ISA는 형태 불일치 문제를 주로 처리하며, 깊이 맵(depth map)을 모션 전달을 위한 중간 매개체로 활용하여 편집된 콘텐츠의 형태에 맞게 깊이 맵을 시뮬레이션하고 정렬함으로써 CIG에 정확한 가이던스를 제공한다.

ISA의 핵심 알고리즘은 다음 두 단계로 구성된다:

**Step 1: 광학 흐름 기반 깊이 맵 전파**

$$
D_t^{\text{warp}} = \mathcal{W}(D_1, \{f_{1 \to t}\})
$$

여기서 $D_1$은 첫 번째 프레임의 깊이 맵, $f_{1 \to t}$는 RAFT로 추정된 $1 \to t$ 프레임 간 광학 흐름, $\mathcal{W}$는 warp 연산이다.

**Step 2: 형태 기반 깊이 맵 정제 (Shape-guided Depth Refinement)**

$$
D_t^{\text{refined}} = \mathcal{R}_\theta(D_t^{\text{warp}}, M_{\hat{I}_1})
$$

여기서 $M_{\hat{I}\_1}$은 편집된 첫 프레임에서 추출된 형태 마스크(U²-Net), $\mathcal{R}_\theta$는 학습된 깊이 정제 네트워크이다.

깊이 정제 네트워크는 첫 번째 프레임 형태 마스크를 가이던스로 받기 위해 추가적인 네트워크 채널을 사용한다.

이 정제 네트워크는 YouTube-VOS 데이터셋으로 학습되며, 훈련 손실 함수는 다음과 같이 구성된다:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{shape}} \mathcal{L}_{\text{shape}}
$$

$$
\mathcal{L}_{\text{recon}} = \|D_t^{\text{refined}} - D_t^{\text{GT}}\|_1
$$

**반복(Iterative) 정렬 절차**:

$$
D_t^{(k+1)} = \mathcal{R}_\theta\left(\mathcal{W}(D_t^{(k)}, f_{t}), M_{\hat{I}_1}\right), \quad k = 0, 1, \ldots, K-1
$$

이 과정을 $K$번 반복함으로써 점진적으로 형태 정렬된 깊이 맵을 얻는다.

#### 2.2.3 CIG (Conditional Image-to-Video Generator)

CIG는 편집된 첫 프레임과 정렬된 깊이 맵을 입력으로 받아 깊이 기반 이미지-투-비디오 생성기로서 전체 편집 영상을 생성한다.

$$
V_{\text{edit}} = \mathcal{G}_{\text{CIG}}(\hat{I}_1, \{D_t^{\text{refined}}\}_{t=1}^{T})
$$

CIG는 I2VGen-XL 백본에 ControlNet(depth)을 결합하고 Ctrl-Adapter로 연결하여 구현된다. 이를 디퓨전 모델의 역방향 과정으로 표현하면:

$$
p_\theta(V_{0:T} | \hat{I}_1, D^{\text{refined}}) = \prod_{t=1}^{T} p_\theta(V_{t-1} | V_t, \hat{I}_1, D_t^{\text{refined}})
$$

모델 구성요소는 controlnet-depth (CIG용 ControlNet), ctrl-adapter-i2vgenxl-depth (I2VGen-XL용 Ctrl-Adapter), i2vgenxl (CIG용 기반 모델), instruct-pix2pix, paint-by-example, stable-diffusion-v1-5-inpaint (PFE용), 그리고 shape-guided depth refinement network 등으로 구성된다.

### 2.3 새로운 평가 벤치마크: DAVIS-Edit

DAVIS-Edit는 비디오 편집 연구에 대한 포괄적인 평가를 위해 수동으로 구성된 테스팅 벤치마크로, 텍스트 기반 및 이미지 기반 편집 작업을 모두 포함하며, 서로 다른 형태 차이 정도를 가진 편집 시나리오를 다루기 위한 두 개의 서브셋으로 구성된다.

DAVIS-Edit는 비디오 편집에서 형태 불일치 문제를 해결하기 위한 표준화된 프로토콜을 확립하기 위해 StableV2V 팀이 신중하게 구성하였다. 텍스트 프롬프트 생성을 위해 팀은 비디오의 주요 요소(객체, 전경 등)를 설명하는 특정 단어를 선택적으로 수정하되, 형태 불일치를 강조하는 데 특히 집중하였다. 예를 들어, "blackswan"을 "duck"으로 교체하여 유사한 형태의 객체를 나타내고, 이어서 "duck"을 "rabbit"으로 교체하여 형태 변형이 있는 시나리오를 시뮬레이션한다.

두 서브셋 구성:
- **DAVIS-Edit-S**: 유사한 형태의 객체 간 편집 (Similar shape)
- **DAVIS-Edit-C**: 큰 형태 변화를 수반하는 편집 (Changed shape)

### 2.4 성능 향상

실험 결과 및 분석은 기존 최첨단 연구들과 비교하여 본 방법의 우수한 성능, 시각적 일관성, 추론 효율성을 보여준다.

평가 지표로는 DOVER (비디오 품질), CLIP-Temporal (시간적 일관성), WE (Warp Error) 등이 사용된다. 관련 후속 연구 FlowV2V(2025)의 비교 데이터에 따르면:

I2Edit, AnyV2V, StableV2V는 향상된 성능을 달성하지만, 결과의 시각적 품질은 여전히 제한적이다.

StableV2V는 DAVIS-Edit-C보다 DAVIS-Edit-S에서 대부분의 연구들이 더 나은 성능을 보인다고 지적하는데, 이는 큰 형태 변화를 수반하는 편집 케이스가 본질적으로 더 어렵기 때문이다.

### 2.5 한계점

StableV2V는 유망한 결과를 보이지만 몇 가지 한계가 존재한다: 복잡한 다중 객체 장면에서의 성능은 추가 검증이 필요하고, 매우 긴 영상 처리는 추가적인 최적화가 필요할 수 있으며, 극단적인 카메라 움직임에서의 성능은 아직 탐색되지 않았고, 실시간 편집 기능은 아직 달성되지 않았다.

특히, 깊이 맵 기반 접근의 구조적 한계도 지적된다:

깊이 맵은 이동하는 객체의 모션을 표현하는 데 종종 차선책이며, 깊이 기반 방법들은 강체(rigid body) 모션의 편집 케이스에 근본적으로 제한된다. 비강체 편집 시나리오, 특히 객체 폐색과 회전이 포함된 경우를 처리할 때, 깊이 맵은 그러한 케이스에서 객체 모션을 정확하게 묘사하지 못하기 때문에 편집된 콘텐츠의 전파에 어려움을 겪는다.

인물 초상화 편집 시에는 StableV2V와 AnyV2V 모두 정지된 콘텐츠를 생성하는데, 이는 깊이 맵 시퀀스와 잠재 특징(latent features) 모두 그러한 복잡한 모션 패턴을 모델링하는 데 근본적인 한계가 있기 때문이다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 강점

StableV2V는 다양한 모달리티의 사용자 프롬프트를 고려하여 광범위한 다운스트림 애플리케이션을 처리하는 데 있어 우수한 유연성을 보인다.

구체적으로 지원되는 편집 유형:
- **텍스트 기반 편집**: InstructPix2Pix 활용
- **이미지 기반 편집**: Paint-by-Example, AnyDoor 활용
- **인페인팅 기반 편집**: SD Inpainting 활용
- **스케치 기반 편집**: ControlNet(scribble) 활용

이 구현체는 Diffusers, Ctrl-Adapter, AnyDoor, RAFT를 기반으로 크게 수정되어 있다.

이러한 모듈형(modular) 설계는 **PFE 부분을 임의의 이미지 편집 모델로 교체 가능**하게 하여, 새로운 이미지 편집 모델이 등장할 때마다 전체 파이프라인의 능력을 쉽게 확장할 수 있도록 한다.

### 3.2 일반화 향상을 위한 잠재적 전략

#### 3.2.1 기반 모델(Backbone) 교체를 통한 확장
현재 CIG는 I2VGen-XL에 의존하지만, 이를 최신 Video Diffusion Transformer (예: CogVideoX, Wan2.1, HunyuanVideo 등)로 교체하면 모션 표현 능력이 크게 향상될 수 있다.

$$
V_{\text{edit}} = \mathcal{G}_{\text{new-backbone}}(\hat{I}_1, D^{\text{refined}})
$$

#### 3.2.2 깊이 맵의 대안적 표현
StableV2V는 비디오 편집에서 형태 불일치 문제를 구체적으로 다루며, 형태 정렬된 깊이 맵을 제어 신호로 얻기 위해 반복적 정렬 알고리즘을 활용한다.

깊이 맵의 한계를 극복하기 위해, **광학 흐름(optical flow)** 또는 **포인트 트래킹(point tracking)** 기반의 신호를 제어 매개체로 사용하면 비강체 모션에 대한 일반화를 향상시킬 수 있다. FlowV2V(2025)가 이 방향성을 실제로 검증하였다.

#### 3.2.3 대규모 데이터셋 확장
현재 형태 기반 깊이 정제 네트워크의 훈련에는 YouTube-VOS 데이터셋이 사용된다.

더 다양한 객체 카테고리와 형태 변화를 포함하는 대규모 데이터셋(예: SA-V, VideoX)으로 확장하면 ISA의 형태 정렬 능력을 일반화할 수 있다.

#### 3.2.4 멀티 객체 일반화
현재 파이프라인은 단일 객체 편집에 최적화되어 있다. 멀티 객체 인스턴스 분할(instance segmentation)을 통해 각 객체에 독립적인 ISA를 적용하고 이를 합성하는 방식으로 다중 객체 편집에 대한 일반화를 도모할 수 있다:

$$
V_{\text{edit}} = \text{Composite}\left(\mathcal{G}(\hat{I}_1^{(1)}, D^{(1)}), \mathcal{G}(\hat{I}_1^{(2)}, D^{(2)}), \ldots\right)
$$

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 방법론 비교표

| 연구 | 연도 | 핵심 아이디어 | 형태 변화 대응 | 멀티모달 입력 | 한계 |
|------|------|--------------|---------------|--------------|------|
| **TokenFlow** | 2024 (ICLR) | Diffusion 특징의 일관성 유지 | ❌ | 텍스트 | 시간적 일관성 부족 |
| **FLATTEN** | 2024 (ICLR) | 광학 흐름 기반 어텐션 | ❌ | 텍스트 | 형태 불일치 미해결 |
| **AnyV2V** | 2024 | Plug-and-Play I2V 활용 | △ | 텍스트/이미지 | 복잡한 모션 한계 |
| **VASE** | 2024 | 객체 중심 형태·외형 조작 | △ | 텍스트/이미지 | 확장성 제한 |
| **I2VEdit** | 2024 | 첫 프레임 기반 I2V | △ | 텍스트 | 정렬 미흡 |
| **StableV2V** | 2024 (TCSVT'25) | ISA + 깊이 맵 정렬 | ✅ | 텍스트/이미지/스케치 | 비강체 모션 한계 |
| **FlowV2V** | 2025 | 광학 흐름 기반 편집 | ✅ | 텍스트 | 평가 중 |

TokenFlow와 DMT는 원본 비디오의 충실도 유지 및 시간적 일관성(WE, CLIP-Temporal) 측면에서 결함을 나타낸다.

FlowV2V는 모든 평가 지표에서 우수한 성능을 보이며, 특히 DOVER, WE, CLIP-Temporal에서 뛰어난 샘플 품질과 시간적 일관성을 달성하고, 다른 지표에서도 비교 가능한 성능을 유지한다.

### 4.2 진화 트렌드

```
[2020-2021]              [2022-2023]              [2024-현재]
Tune-A-Video             InstructPix2Pix          StableV2V
(파인튜닝 기반)          (Instruct 기반)          (형태 일관성)
     ↓                        ↓                        ↓
Video-P2P                Prompt2Prompt            FlowV2V
(P2P 어텐션)             (이미지→비디오)          (광학 흐름 기반)
```

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### 5.1.1 새로운 평가 기준 제시
DAVIS-Edit 벤치마크는 다양한 유형의 프롬프트와 난이도를 고려하여 비디오 편집에 대한 포괄적인 평가를 위해 구성되었다.

이는 향후 비디오 편집 연구의 표준 평가 플랫폼으로 기능하며, **형태 변화 정도에 따른 세분화된 평가**라는 새로운 기준을 제시하였다.

#### 5.1.2 모듈형 파이프라인 패러다임 확산
StableV2V의 **PFE → ISA → CIG** 구조는 이미지 편집과 비디오 생성의 강점을 결합한 모듈형 파이프라인 설계의 유효성을 증명하여, 후속 연구들(FlowV2V, V2Edit 등)이 유사한 구조를 채택하도록 영향을 미쳤다.

이 발전은 고품질 비디오 편집을 더욱 접근 가능하고 일관성 있게 만들어 영화 제작부터 소셜 미디어 콘텐츠 제작에 이르는 다양한 분야에 영향을 미칠 수 있다.

### 5.2 앞으로 연구 시 고려할 점

#### 5.2.1 깊이 맵의 대안 탐색
깊이 맵은 이동하는 객체의 모션을 표현하는 데 종종 차선책이며, 깊이 기반 방법들은 강체(rigid body) 모션의 편집 케이스에 근본적으로 제한된다는 점을 인지해야 한다.

따라서 연구자들은 다음과 같은 제어 신호 대안을 탐색해야 한다:
- **광학 흐름(Optical Flow)**: 비강체 모션에 더 유연
- **포인트 트래킹(Point Tracking)**: 객체별 정밀 추적
- **스켈레톤/포즈 맵**: 인체 편집 특화

#### 5.2.2 비강체·다중 객체 시나리오 대응

다중 객체 편집에서 비교 대상 방법들이 아티팩트를 생성하는 경향이 있으므로, 멀티 객체 독립 편집 후 합성(compositing) 전략이나 인스턴스 인식 제어 신호 설계가 중요 연구 방향이다.

#### 5.2.3 실시간 추론 최적화
현재 StableV2V의 파이프라인은 다수의 모델을 순차적으로 실행하여 추론 속도가 느리다. 향후 연구에서는:
- **지식 증류(Knowledge Distillation)** 기반 경량화
- **일관성 모델(Consistency Models)** 기반 빠른 샘플링
- **비디오 전용 End-to-End 학습**

등을 통해 실시간 편집에 근접한 효율성을 달성해야 한다.

#### 5.2.4 더 큰 사전 학습 모델과의 통합
최근 등장한 대형 Video Diffusion Transformer 모델(CogVideoX-5B, HunyuanVideo, Wan2.1 등)을 CIG의 기반 모델로 활용하면, 더 다양하고 복잡한 동작을 자연스럽게 처리할 수 있는 가능성이 높다.

#### 5.2.5 훈련 데이터의 다양성 확보
현재 형태 기반 깊이 정제 네트워크의 훈련은 YouTube-VOS를 사용하는데, 이는 특정 카테고리에 편향될 수 있다. 더 다양하고 대규모의 비디오 데이터셋 및 합성 데이터(synthetic data)를 활용하여 일반화 성능을 높이는 것이 필수적이다.

---

## 📚 참고 자료 출처

| # | 자료명 | 링크/출처 |
|---|--------|----------|
| 1 | **StableV2V 공식 arXiv 논문** | https://arxiv.org/abs/2411.11045 |
| 2 | **StableV2V GitHub 공식 구현체** | https://github.com/AlonzoLeeeooo/StableV2V |
| 3 | **StableV2V 프로젝트 페이지** | https://alonzoleeeooo.github.io/StableV2V/ |
| 4 | **HuggingFace 논문 페이지** | https://huggingface.co/papers/2411.11045 |
| 5 | **HuggingFace 모델 가중치** | https://huggingface.co/AlonzoLeeeooo/StableV2V |
| 6 | **IEEE TCSVT 게재본** | https://ieeexplore.ieee.org/document/11272911/ |
| 7 | **ResearchGate 논문 페이지** | https://www.researchgate.net/publication/385921580 |
| 8 | **AI Models FYI 분석 페이지** | https://www.aimodels.fyi/papers/arxiv/stablev2v-stablizing-shape-consistency-video-to-video |
| 9 | **FlowV2V (후속 연구, arXiv 2506.07713)** | https://arxiv.org/html/2506.07713 |
| 10 | **V2Edit (비교 연구, arXiv 2503.10634)** | https://arxiv.org/html/2503.10634v1 |
| 11 | **Awesome Video Generation (관련 연구 목록)** | https://github.com/AlonzoLeeeooo/awesome-video-generation |

> ⚠️ **정확도 관련 고지**: 논문 내 구체적인 수식(특히 ISA의 손실 함수 세부 항)과 정량적 수치(DOVER, WE 등 정확한 수치)는 공개된 arXiv PDF 원문에서 직접 확인하실 것을 권장합니다. 본 답변에서 제시한 수식들은 논문의 방법론 설명과 공개된 코드베이스를 기반으로 재구성한 것이며, 논문 원문의 표기와 일부 차이가 있을 수 있습니다.
