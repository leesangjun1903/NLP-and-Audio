
# VideoPoet: A Large Language Model for Zero-Shot Video Generation

> **논문 정보**
> - **저자:** Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama 외 (Google Research & CMU)
> - **발표:** arXiv:2312.14125 (2023.12) → **ICML 2024 Best Paper Award** 수상
> - **공식 링크:** [arxiv.org/abs/2312.14125](https://arxiv.org/abs/2312.14125) | [PMLR](https://proceedings.mlr.press/v235/kondratyuk24a.html) | [Project Page](https://sites.research.google/videopoet/)

---

## 1️⃣ 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

VideoPoet은 다양한 조건 신호(conditioning signals)로부터 고품질 비디오와 매칭 오디오를 합성할 수 있는 언어 모델이며, Google 및 Carnegie Mellon University의 연구로, 비디오 생성을 텍스트 생성과 유사한 자기회귀(autoregressive) 시퀀스 모델링 문제로 취급하는 패러다임 전환을 제시합니다.

당시 주요 비디오 생성 모델들은 거의 대부분 확산 모델(diffusion-based)인 반면, LLM은 언어·코드·오디오 등 다양한 모달리티에 걸쳐 뛰어난 학습 능력으로 사실상의 표준으로 인정받고 있었습니다. VideoPoet은 이 간극을 메우려 합니다.

### 📋 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **단일 LLM 멀티태스크** | 별도의 모델 없이 하나의 LLM 내에 다중 비디오 생성 기능 통합 |
| **멀티모달 사전학습 전략** | 텍스트 페어링 없는 비디오도 활용 가능한 Task Mixture 설계 |
| **비디오 토크나이저 활용** | MAGVIT-v2 + SoundStream으로 시각/음성 이산 토큰화 |
| **잠재 공간 초해상도** | 양방향 Transformer 기반 SR 모듈 설계 |
| **Zero-shot 일반화** | 훈련에 없던 새로운 태스크를 태스크 체이닝으로 수행 |

이 논문은 ICML 2024에서 Best Paper Award를 수상하였습니다.

---

## 2️⃣ 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 / 한계

---

### 🔴 (A) 해결하고자 하는 문제

**문제 1: 비디오 생성의 확산 모델 독점**

당시 주요 비디오 생성 모델은 거의 모두 확산 기반이었고, LLM은 언어·코드·오디오에서 사실상 표준으로 자리잡고 있었습니다. 기존 접근법은 각 태스크를 별도로 학습된 컴포넌트에 의존하는 반면, VideoPoet은 이 모든 기능을 단일 LLM 안에 통합합니다.

**문제 2: 비디오의 이산 토큰화 어려움**

LLM은 이산 토큰(discrete tokens)을 처리하므로 비디오 생성이 까다롭지만, 비디오·오디오 토크나이저가 존재하여 클립을 이산 토큰 시퀀스로 인코딩하고 다시 원래 표현으로 변환할 수 있습니다.

**문제 3: 비디오-텍스트 페어 데이터 부족**

VideoPoet의 사전학습 전략은 동일한 비디오를 텍스트 없이도 여러 학습 태스크에 활용할 수 있게 설계되어, 비디오만의 데이터를 대규모로 학습하고 비디오-텍스트 페어에 대한 의존도를 줄입니다.

---

### 🟠 (B) 제안 방법: 핵심 수식 포함

#### ① 자기회귀 언어 모델링 목표

VideoPoet은 비디오를 이산 토큰 시퀀스 $\mathbf{x} = (x_1, x_2, \ldots, x_T)$로 변환한 뒤, 다음 토큰을 조건부 확률로 예측합니다:

$$
p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_1, x_2, \ldots, x_{t-1}, \mathbf{c})
$$

여기서 $\mathbf{c}$는 텍스트 임베딩, 시각 토큰, 오디오 토큰 등으로 구성된 컨디셔닝 입력입니다.

#### ② 학습 손실 함수 (Negative Log-Likelihood)

모델은 각 태스크의 출력 토큰에 대해서만 손실을 계산합니다:

$$
\mathcal{L} = -\sum_{i=1}^{N} \sum_{t \in \mathcal{O}_i} \log p(x_t \mid x_{<t}, \mathbf{c}_i)
$$

여기서:
- $N$: 학습 샘플 수
- $\mathcal{O}_i$: $i$번째 샘플에서 출력 토큰 인덱스 집합 (prefix에는 손실 불포함)
- $\mathbf{c}_i$: $i$번째 샘플의 prefix 입력 (텍스트/시각/오디오 토큰)

각 사전학습 태스크는 정의된 prefix 입력과 출력이 있으며, 모델은 prefix에 조건화되어 오직 출력에 대해서만 손실을 적용합니다.

#### ③ Task Mixture (멀티태스크 사전학습)

LLM 학습 프레임워크에 텍스트-투-비디오, 텍스트-투-이미지, 이미지-투-비디오, 비디오 프레임 연속 생성, 비디오 인페인팅 및 아웃페인팅, 비디오 스타일화, 비디오-투-오디오 등 멀티모달 생성 학습 목표가 혼합하여 도입됩니다.

전체 사전학습 손실은 다음과 같이 각 태스크의 가중합으로 표현됩니다:

$$
\mathcal{L}_{\text{total}} = \sum_{k=1}^{K} w_k \cdot \mathcal{L}_k
$$

여기서 $K$는 태스크 수이고 $w_k$는 각 태스크의 샘플링 가중치입니다.

#### ④ MAGVIT-v2 토크나이저 (시각적 이산 양자화)

MAGVIT-v2는 비디오와 이미지 모두에 대해 간결하고 표현력 있는 토큰을 공통 토큰 어휘로 생성하도록 설계된 비디오 토크나이저로, 이를 활용하면 표준 이미지 및 비디오 생성 벤치마크에서 LLM이 확산 모델을 능가합니다.

비디오 클립은 다음과 같이 정수 시퀀스로 인코딩됩니다:

$$
\mathbf{z} = \text{Quantize}(\text{Encoder}(\mathbf{v})) \in \{0, 1, \ldots, |\mathcal{V}|-1\}^{L}
$$

여기서 $\mathcal{V}$는 코드북(vocabulary), $L$은 토큰 시퀀스 길이입니다.

MAGVIT-v2 토크나이저는 128×128 해상도의 17프레임 비디오를 1,280개의 토큰으로 압축하고, 단일 이미지를 256개의 토큰으로 압축합니다.

#### ⑤ 통합 어휘 공간 설계

통합 어휘는 다음과 같이 구성됩니다: 처음 256개 코드는 특수 토큰 및 태스크 프롬프트용으로 예약되고, 이후 262,144개 코드는 이미지 및 비디오 토크나이제이션에, 그 다음 4,096개 코드는 오디오에 할당됩니다. 텍스트는 처음부터 학습하는 텍스트 토큰보다 성능이 우수한 텍스트 임베딩으로 표현됩니다.

#### ⑥ 초해상도 모듈 (Super-Resolution)

비디오 초해상도 접근법으로, 효율적인 윈도우 로컬 어텐션(windowed local attention)을 사용하는 양방향 트랜스포머(bidirectional transformer)로 잠재 토큰 공간 내에서 공간 해상도를 높입니다.

---

### 🟡 (C) 모델 구조

VideoPoet의 모델은 세 가지 주요 컴포넌트로 구성됩니다: (1) 모달리티별 토크나이저, (2) 언어 모델 백본, (3) 초해상도 모듈.

```
┌───────────────────────────────────────────────────────────┐
│                     VideoPoet 전체 구조                    │
├───────────────────┬───────────────────┬───────────────────┤
│   1. 토크나이저   │  2. LLM 백본      │ 3. SR 모듈        │
│                   │                   │                   │
│ ┌───────────────┐ │ ┌───────────────┐ │ ┌───────────────┐ │
│ │ MAGVIT-v2     │ │ │ Decoder-only  │ │ │ Bidirectional │ │
│ │ (Video/Image) │ │ │ Transformer   │ │ │ Transformer   │ │
│ └───────────────┘ │ │               │ │ │ + Windowed    │ │
│ ┌───────────────┐ │ │ T5 Text Emb.  │ │ │ Local Attn.   │ │
│ │ SoundStream   │ │ │ Causal Attn.  │ │ └───────────────┘ │
│ │ (Audio)       │ │ │               │ │                   │
│ └───────────────┘ │ └───────────────┘ │                   │
└───────────────────┴───────────────────┴───────────────────┘
```

LLM은 텍스트 임베딩, 시각 토큰, 오디오 토큰을 입력으로 받아 멀티모달 멀티태스크 모델링을 수행합니다. VideoPoet은 텍스트 임베딩, 시각 토큰, 오디오 토큰에 조건화되어 시각 및 오디오 토큰을 자기회귀적으로 예측하며, 이후 초해상도 모듈이 비디오 출력의 해상도를 높이고 시각적 디테일을 개선합니다.

**학습 2단계:**

VideoPoet의 학습 과정은 (1) 사전학습과 (2) 태스크 적응(task-adaptation)의 두 단계로 구성됩니다. 사전학습 단계에서는 자기회귀 트랜스포머 프레임워크 내에서 멀티모달 사전학습 목표의 혼합을 통합하며, 사전학습 후에는 텍스트-투-비디오, 이미지-투-비디오, 비디오 편집, 스타일화 등 다양한 멀티태스크 비디오 생성 모델로 기능합니다. 이러한 기능은 텍스트 프롬프트로 제어되는 별도의 생성 모델에 의존하지 않고 단일 LLM에 통합됩니다. 이후 태스크 적응 단계에서 사전학습된 모델은 훈련 태스크의 생성 품질을 향상시키거나 새로운 태스크를 수행하도록 추가 파인튜닝될 수 있습니다.

**학습 데이터:**

이 모델은 10억 개의 이미지-텍스트 쌍과 2억 7천만 개의 비디오를 포함한 2조 개의 토큰으로 학습되었습니다.

---

### 🟢 (D) 성능 향상

VideoPoet은 zero-shot 비디오 생성에서 최첨단(state-of-the-art) 성능을 보이며, 특히 고충실도 모션(high-fidelity motions) 생성 능력을 강조합니다.

예를 들어, 사전학습된 VideoPoet 모델은 MSR-VTT에서 FVD 213을 달성하여 CogVideo(1294) 및 Show-1(538)을 크게 능가하였으며, CLIPSIM 0.3049로 많은 확산 모델과 동등하거나 이를 초과합니다.

VideoPoet은 인간 평가자를 통해 Show-1, VideoCrafter, Phenaki, Pika, Gen2, WALT, Lumiere 등 최근 주요 모델들과 비교 분석되었습니다. Show-1, VideoCrafter, Pika, Gen2, WALT, Lumiere는 비디오 확산 모델이고 Phenaki는 마스크 토큰 모델링을 사용하는 토큰 기반 모델입니다.

VideoPoet을 통해 LLM이 다양한 태스크에서 고도로 경쟁력 있는 비디오 생성 품질을 보임이 입증되었으며, 특히 비디오 내에서 흥미롭고 고품질의 모션을 생성하는 데 있어서 LLM의 비디오 생성 분야에서의 유망한 잠재력을 보여줍니다.

---

### 🔵 (E) 한계점

1. **짧은 기본 생성 길이:** 기본적으로 VideoPoet은 2초 비디오를 출력하며, 긴 비디오 생성은 1초 비디오 클립 입력으로 1초 비디오 출력을 예측하는 방식으로 처리합니다.

2. **낮은 기본 해상도:** MAGVIT-v2 토크나이저가 128×128 해상도로 압축하므로 초해상도 모듈이 별도로 필요합니다.

3. **자기회귀 추론 속도:** 자기회귀 모델의 특성상, 토큰을 순차적으로 생성해야 하므로 확산 모델 대비 추론 지연이 발생할 수 있습니다.

4. **비-인과적 태스크 한계:** LLM은 이산 토큰을 처리하기 때문에 비디오 생성이 도전적이며, 연속적인 시각 표현을 요구하는 일부 고화질 생성에서 제한이 있습니다.

5. **AI 오용 위험:** 저자들은 잠재적 오용 우려를 명시적으로 언급하며 딥페이크나 허위정보 생성 방지를 위한 디지털 워터마킹 및 투명성 조치 도입을 약속합니다.

---

## 3️⃣ 모델의 일반화 성능 향상 가능성

### 🌟 Zero-Shot 일반화의 핵심 메커니즘

VideoPoet이 "zero-shot 비디오 생성"이라고 부르는 이유는, 훈련 데이터 분포와 다른 새로운 텍스트·이미지·비디오 입력을 처리하는 일반화 능력을 보여주기 때문입니다. 또한 VideoPoet은 훈련에 포함되지 않은 새로운 태스크를 처리하는 능력도 나타나기 시작하며, 예를 들어 훈련 태스크를 순차적으로 연결(chaining)함으로써 새로운 편집 태스크를 수행합니다.

### 🔗 태스크 체이닝 (Task Chaining)을 통한 Zero-Shot 능력

이러한 태스크들은 추가적인 zero-shot 기능(예: text-to-audio)을 위해 함께 구성될 수도 있습니다.

예시 체이닝 파이프라인:

```
Text → [T2V] → Video → [V2A] → Audio
Image → [I2V] → Video → [Stylization] → Styled Video
```

연구팀은 여러 연산을 연결함으로써 모델이 여러 창발적(emergent) 능력을 보임을 발견하였으며, 예를 들어 이미지-투-비디오로 단일 이미지를 애니메이션화한 후 스타일화를 적용하거나, 장형 비디오 생성, 일관된 3D 구조 유지, 텍스트 프롬프트로부터 카메라 모션 적용이 가능합니다.

### 📈 멀티태스크 사전학습의 스케일링 효과

GPT-3와 PaLM은 다양한 태스크로 LLM을 훈련하면 zero-shot 및 few-shot 태스크에 긍정적인 스케일링 효과를 가져온다는 것을 보여줍니다. VideoPoet은 이 원리를 비디오 도메인에 그대로 적용합니다.

### 🔄 미페어 데이터 활용에 의한 일반화 강화

사전학습 전략은 텍스트 없이도 동일한 비디오를 여러 훈련 태스크에 사용할 수 있게 하여, 대량의 비디오 전용 예제 학습을 가능하게 하고 비디오-텍스트 페어에 대한 요구를 줄입니다.

### 📷 카메라 모션 제어의 창발적 일반화

VideoPoet의 사전학습의 한 창발적 특성은 텍스트 프롬프트에 촬영 유형을 지정함으로써 고품질의 카메라 모션 커스터마이제이션이 가능하다는 것입니다. 이는 명시적으로 훈련하지 않아도 나타나는 일반화 능력입니다.

### 🎞️ 장기 비디오 자기회귀 확장

1초의 비디오 입력을 조건으로 1초의 비디오 출력을 예측하는 방식으로 더 긴 비디오를 생성할 수 있으며, 이를 반복적으로 체이닝하면 모델이 비디오를 잘 확장할 뿐만 아니라 여러 반복에 걸쳐서도 모든 객체의 외관을 충실하게 보존합니다.

---

## 4️⃣ 앞으로의 연구에 미치는 영향 및 고려 사항

### 🚀 연구적 영향

**① 패러다임 전환: 확산 → LLM 기반 통합 모델**

VideoPoet의 성공은 생성 AI 연구에서의 잠재적인 패러다임 전환을 시사하며, LLM 아키텍처가 여러 모달리티에 걸쳐 특화된 접근법과 경쟁할 수 있음을 보여줍니다. 이는 단일 프레임워크 내에서 다양한 태스크를 처리할 수 있는 더 통합된 범용 AI 시스템 개발에 시사점을 줍니다.

**② Any-to-Any 생성의 새로운 방향**

"any-to-any" 생성 지원, 즉 text-to-audio, audio-to-video, video captioning 등으로 확장이 프레임워크적으로 가능합니다.

**③ 단일 모델 멀티모달 통합의 선례**

LLM 원리를 비디오 생성에 성공적으로 적용함으로써 콘텐츠 창작, 엔터테인먼트, 인간-컴퓨터 상호작용을 혁신할 수 있는 미래 멀티모달 AI 시스템의 기반을 마련합니다.

---

### 🔬 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 출처 | 방식 | VideoPoet 대비 차이점 |
|---|---|---|---|
| **CogVideo** (2022) | Tsinghua | AR Transformer | 텍스트-비디오만; 단일 모달 | 
| **Phenaki** (2022) | Google | Masked Token | 변동 길이 텍스트 시퀀스 지원, 오디오 없음 |
| **MAGVIT-v2** (2023) | Google/CMU | LM 기반 토크나이저 | ImageNet, Kinetics 벤치마크에서 LLM이 확산 모델 능가 입증 |
| **Lumiere** (2024) | Google | Space-Time Diffusion | 단일 패스 전체 시간 생성 가능 |
| **Sora** (2024) | OpenAI | Diffusion Transformer | 최대 60초의 고충실도 비디오 클립 생성 가능 |
| **Show-1** (2023) | NUS | Hybrid Pixel+Latent | Pixel + Latent 확산 결합 |

VideoPoet은 Show-1, VideoCrafter, Phenaki, Pika, Gen2, WALT, Lumiere와 인간 평가를 통해 비교되었으며, Show-1, VideoCrafter, Pika, Gen2, WALT, Lumiere가 비디오 확산 모델인 반면 Phenaki는 마스크 토큰 모델입니다. 가장 최신 버전 기준으로 2024년 1월 시점에 비교 평가가 이루어졌습니다.

---

### 💡 앞으로 연구 시 고려할 점

#### 1. **더 효율적인 시각 토크나이저 설계**
시각 토크나이저는 비디오 생성 품질의 상한선을 결정하는 핵심 요소로, MAGVIT-v2 토크나이저를 선택한 이유는 시각 품질 성능과 높은 압축 능력으로 LLM에 필요한 시퀀스 길이를 효과적으로 줄이기 때문입니다. 향후 연구에서는 더 고해상도에서도 효율적인 토크나이저 개발이 중요합니다.

#### 2. **스케일링 법칙(Scaling Laws) 연구**
비디오 LLM에서의 스케일링 효과를 검증하는 연구가 필요합니다. 언어 모델은 여러 학습 태스크를 쉽게 통합할 수 있고, GPT-3와 PaLM은 다양한 태스크에서 LLM을 학습하면 zero-shot 및 few-shot 태스크에 긍정적인 스케일링 효과가 있음을 보여줍니다.

#### 3. **물리 법칙 준수 및 시공간 일관성 강화**
자기회귀 생성 방식은 장기 비디오에서 물리적 일관성이 누적 오류로 깨질 수 있습니다. 이를 위한 World Model 방향의 연구가 필요합니다.

#### 4. **Any-to-Any 멀티모달 확장**
VideoPoet을 통해 LLM의 높은 경쟁력 있는 비디오 생성 품질이 입증되었으며, 결과는 비디오 생성 분야에서 LLM의 유망한 잠재력을 시사합니다. 향후 방향으로는 text-to-audio, audio-to-video, video captioning 등 "any-to-any" 생성 지원으로의 확장이 가능합니다.

#### 5. **책임 있는 AI 개발 (Responsible AI)**
저자들은 잠재적 오용 우려를 명시적으로 언급하며, 딥페이크나 허위정보 생성 방지를 위한 디지털 워터마킹 및 투명성 조치를 도입하겠다고 공약합니다. 향후 연구에서도 생성물 추적과 윤리적 가이드라인 수립이 중요한 연구 과제입니다.

#### 6. **인터랙티브 편집 및 사용자 제어 강화**
대화형 편집이 가능하며, 입력 비디오를 짧은 시간 연장하고 후보 목록에서 선택함으로써 더 큰 생성 비디오에서 원하는 모션 유형을 세밀하게 제어할 수 있습니다. 이 방향을 더 정교하게 발전시키는 연구가 유망합니다.

---

## 📚 참고문헌 및 출처

| # | 제목 | 출처/링크 |
|---|---|---|
| 1 | **VideoPoet: A Large Language Model for Zero-Shot Video Generation** (원논문) | [arXiv:2312.14125](https://arxiv.org/abs/2312.14125) |
| 2 | **VideoPoet ICML 2024 공식 게재본** | [PMLR, Proceedings of the 41st ICML 235:25105-25124](https://proceedings.mlr.press/v235/kondratyuk24a.html) |
| 3 | **VideoPoet 프로젝트 페이지 (Google Research)** | [sites.research.google/videopoet](https://sites.research.google/videopoet/) |
| 4 | **VideoPoet 공식 블로그 (Google Research Blog)** | [research.google/blog/videopoet](https://research.google/blog/videopoet-a-large-language-model-for-zero-shot-video-generation/) |
| 5 | **arXiv HTML 전문 (v1, v2, v4)** | [arxiv.org/html/2312.14125](https://arxiv.org/html/2312.14125v2) |
| 6 | **ar5iv 렌더링 버전** | [ar5iv.labs.arxiv.org/html/2312.14125](https://ar5iv.labs.arxiv.org/html/2312.14125) |
| 7 | **Language Model Beats Diffusion: Tokenizer is Key (MAGVIT-v2)** | [arXiv:2310.05737](https://arxiv.org/html/2310.05737v2) / [magvit.cs.cmu.edu/v2](https://magvit.cs.cmu.edu/v2/) |
| 8 | **AlphaXiv 분석 페이지** | [alphaxiv.org/overview/2312.14125v4](https://www.alphaxiv.org/overview/2312.14125v4) |
| 9 | **Semantic Scholar 논문 페이지** | [semanticscholar.org](https://www.semanticscholar.org/paper/0c4f46e4dcae5527018e6432fb60cfe8c3354e97) |
| 10 | **ACM DL 게재본** | [dl.acm.org/doi/10.5555/3692070.3693075](https://dl.acm.org/doi/10.5555/3692070.3693075) |
| 11 | **Irfan Essa 교수 개인 페이지 (ICML Best Paper 수상)** | [irfanessa.gatech.edu](https://www.irfanessa.gatech.edu/videopoet-a-large-language-model-for-zero-shot-video-generation/) |
| 12 | **Liner Quick Review: VideoPoet** | [liner.com](https://liner.com/review/videopoet-large-language-model-for-zeroshot-video-generation) |
| 13 | **Google InfoQ 뉴스 (VideoPoet 분석)** | [infoq.com/news/2024/01/google-video-poet](https://www.infoq.com/news/2024/01/google-video-poet/) |
| 14 | **Video Generation Survey (GitHub)** | [github.com/yzhang2016/video-generation-survey](https://github.com/yzhang2016/video-generation-survey/blob/main/video-generation.md) |

# VideoPoet: A Large Language Model for Zero-Shot Video Generation

### 1. 핵심 주장 및 주요 기여 요약

**VideoPoet의 중심 주장**은 확산 모델 중심의 비디오 생성 패러다임에서 벗어나 **대규모 언어모델(LLM) 아키텍처가 비디오 생성에서 경쟁력 있는 성능을 달성할 수 있다**는 것입니다. 이는 LLM이 다중모달 입력(텍스트, 이미지, 비디오, 오디오)을 처리하여 통합된 단일 모델 내에서 다양한 비디오 생성 작업을 수행할 수 있음을 시사합니다.[1]

**주요 기여**는 다음 세 가지입니다:
1. 텍스트 쌍이 있는 비디오와 없는 비디오 데이터를 모두 활용하여 비디오 생성용 LLM을 훈련하는 방법
2. 양방향 트랜스포머와 효율적인 윈도우 로컬 어텐션을 사용하여 잠재 토큰 공간에서 공간 해상도를 증가시키는 기술
3. 훈련 데이터 분포에서 벗어난 새로운 입력에 대한 제로샷 성능 및 작업 체이닝을 통한 새로운 편집 작업 수행 능력 입증[1]

***

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하고자 하는 문제

**문제 1: 확산 모델의 제한성** - 기존 비디오 생성 모델은 확산 기반 방식 위주로, 작업별로 별도의 적응 모듈이 필요하며, 텍스트-이미지 생성 확산 모델에서 출발하여 시간 일관성을 위해 추가 튜닝이 필요합니다.[1]

**문제 2: 단일 통합 모델의 부재** - 텍스트-비디오, 이미지-비디오, 비디오 스타일화, 비디오 편집 등 다양한 작업을 하나의 통합된 프레임워크로 수행하기 어렵습니다.[1]

**문제 3: 제로샷 성능 및 작업 일반화의 한계** - 기존 모델들이 훈련 분포 외의 새로운 입력에 대한 일반화 성능이 제한적입니다.[1]

#### 2.2 제안하는 방법 및 수식

**MAGVIT-v2 토큰화 (시각)**[1]
- 17프레임, 2.125초, 128×128 해상도 비디오를 (5, 16, 16) 형태의 잠재 표현으로 인코딩
- 1280개의 토큰 생성, 어휘 크기: \( 2^{18} = 262,144 \)

**SoundStream 토큰화 (오디오)**[1]
- 2.125초 오디오를 106개 잠재 프레임으로 인코딩
- 4개 레벨 잔여 벡터 양자화(RVQ): 각 레벨 1,024개 코드
- 총 오디오 어휘: \( 4 \times 1,024 = 4,096 \)

**통합 어휘**[1]
- 특수 토큰: 256개, 시각 토큰: 262,144개, 오디오 토큰: 4,096개
- **총 어휘 크기: 약 300,000**

**자동회귀 생성 과정**:[1]

$$P(v_1, v_2, \ldots, v_T | c) = \prod_{t=1}^{T} P(v_t | v_1, \ldots, v_{t-1}, c)$$

여기서 \( v_t \)는 시간 t에서의 토큰, \( c \)는 조건 정보(텍스트, 이미지, 오디오)입니다.

#### 2.3 모델 구조

**디코더 전용 트랜스포머**[1]
- 다중모달 입력(T5 텍스트 임베딩, MAGVIT-v2 시각 토큰, SoundStream 오디오 토큰)을 처리
- 시각 및 오디오 토큰 출력을 자동회귀적으로 생성
- 입력 시퀀스: 양방향 어텐션, 출력 시퀀스: 인과 마스킹

**초해상도 모듈**[1]
- 3개 트랜스포머 레이어 블록: 공간 수직 윈도우, 공간 수평 윈도우, 시간 로컬 어텐션
- 토큰 인수분해(k=2)로 262,144방향 분류를 512방향 분류 2개로 변환
- 캐스케이드 구조: 224×128 → 448×256 → 896×512

**두 단계 샘플링 전략**[1]
- 첫 25% 훈련: 이미지 90%, 비디오 10% (물체 인식 향상)
- 나머지 75%: 이미지 10%, 비디오 90% (동작 학습)

**교대 경사하강법(AGD)**[1]
훈련 시퀀스 길이 그룹별로 교대로 샘플링하여 패딩 비율을 약 0%로 유지합니다.

#### 2.4 성능 향상 결과

**텍스트-비디오 생성 (MSR-VTT)**[1]

| 모델 | CLIP Similarity ↑ | FVD ↓ |
|------|------------------|-------|
| CogVideo (2022) | 0.263 | 1,294 |
| Show-1 (2023) | 0.307 | 538 |
| VideoPoet (Pretrain) | 0.305 | 213 |
| VideoPoet (Task Adapt) | 0.312 | - |

**인적 평가 (T2V 생성)** - VideoPoet는 **동작 흥미도**(48-82% 범위)와 **동작 사실성**(39-84% 범위)에서 특히 강력한 성능을 보였습니다.[1]

**프레임 예측 (Kinetics-600)**[1]

| 방법 | FVD ↓ |
|------|-------|
| T2V 전용 | 759 |
| 모든 태스크 포함 | 729 |

**모델 규모별 성능**[1]

| 모델 크기 | 훈련 데이터 토큰 | FVD (비디오) ↓ |
|----------|------------|------------|
| 300M | 10B | ~1,085 |
| 1B | 37B | ~500 |
| 8B | 58B | 355 |

#### 2.5 한계

VideoPoet의 주요 한계는:[1]

1. **토큰 압축의 한계** - RGB 프레임의 압축 및 양자화로부터 상한선 설정과 미세한 세부사항 손실
2. **정적 장면의 미학 편차** - 프레임 레벨 미학 편차가 기준선과 일치하지 않음
3. **소형 물체 및 세부 묘사** - 큰 동작이 동반된 작은 물체와 세부사항 표현의 어려움
4. **인페인팅 초기 성능** - 디코딩 붕괴(반복적 토큰 예측) 현상

***

### 3. 일반화 성능 향상 가능성

#### 3.1 제로샷 성능의 메커니즘

**다중 작업 사전훈련의 긍정적 효과**:[1]
- T2V 단독: CLIP 유사도 0.244
- 모든 작업 포함: CLIP 유사도 0.240 (전체 벤치마크에서 평균 최고)

교차-작업 일반화를 통한 기본 표현 학습이 핵심입니다.

**자가 감독 학습(SSL) 태스크**[1]
미래 예측, 인페인팅, 음성-비디오 연속 작업이 텍스트 쌍 없는 데이터로부터 모션 다이나믹스에 대한 기초 이해를 제공합니다.

**장시간 비디오 자동회귀 확장**[1]
최근 1초를 조건으로 다음 1초를 자동회귀적으로 확장하여 10초 이상의 비디오를 생성하면서도 반복적 확장으로 시각적 일관성을 유지합니다.

#### 3.2 교차-작업 체이닝을 통한 제로샷 일반화

**새로운 작업 조합**:[1]
- 이미지-비디오 + 비디오-비디오 스타일화: 정적 이미지 → 애니메이션 → 스타일화
- 비디오-비디오 아웃페인팅 + 비디오 편집: 확장 후 추가 효과

각 단계 출력이 다음 단계의 입력으로 사용되며, 출력이 동일 분포 유지로 인해 연쇄적 적용이 가능합니다.

#### 3.3 모델 규모에 따른 일반화 개선

**1B vs 8B 모델 비교**:[1]
8B 모델은 시간 일관성, 프롬프트 충실도, 동작 다이나믹스, 텍스트 렌더링 능력, 공간 이해, 개수 세기 등 모든 면에서 1B 모델을 능가합니다.

**스케일링 법칙**: \( \text{FVD} \propto (D \cdot M)^{-\alpha} \) (α ≈ 0.3-0.4)

#### 3.4 제한된 일반화: "사례 기반" 행동

**최근 연구 발견** (2024 "How Far is Video Generation from World Model"):[2]

비디오 생성 모델의 일반화는 다음과 같이 분류됩니다:[2]
1. **분포 내 일반화**: 우수
2. **조합론적 일반화**: 측정 가능하나 제한적
3. **분포 외 일반화**: 실패

모델이 기본 물리 법칙을 학습하지 않고 유사한 훈련 예제를 모방하며, 단순히 스케일링만으로는 이 문제를 해결할 수 없습니다.[2]

**우선순위 순서** (새로운 케이스 참조 시): 색상 > 크기 > 속도 > 형태[2]

***

### 4. 앞으로의 연구 영향과 고려사항

#### 4.1 비디오 생성 분야에 미치는 영향

**패러다임 전환**[3]

| 연도 | 주요 개발 | 영향 |
|------|---------|------|
| 2023 | Stable Video Diffusion (확산 기반) | 비디오 확산 모델 표준화 |
| 2024 | Sora (텍스트 조건 확산) | 변수 길이/해상도 생성 |
| 2024-2025 | 자동회귀 모델 부흥 | **VAR, Infinity, InfinityStar** |
| 2025 | HunyuanVideo, Wan (확산 + 트랜스포머) | **하이브리드 접근법** |

최근 모델들이 LLM 트랜스포머와 확산의 하이브리드 방식으로 수렴하고 있습니다.[3]

**통합 다중 작업 프레임워크의 영향**:[4][5]
- VideoDirectorGPT (2024): LLM을 통한 다중 장면 비디오 생성 계획
- iVideoGPT (2024): 상호작용 가능한 세계 모델
- InfinityStar (2025 NeurIPS Oral): VAR 기반 텍스트-비디오 생성

#### 4.2 최신 연구 기반 제언사항

**일반화 성능 향상 전략**:[2]

1. **물리 인식 사전훈련** - 합성 데이터 기반 물리 시뮬레이션과 구조화된 동작 표현 학습
2. **부트스트랩 강화** - 인적 피드백 활용(VideoReward, 2025)[6]
3. **다중 규모 표현** - 계층적 토큰 모델링과 조건부 독립성 구조 학습

**토큰 기반 모델의 진화 방향**:[3]
- 벡터 양자화 없는 연속 토큰(Autoregressive Image Generation, 2024)
- 적응형 토크나이저(Pandora 모델)
- 다계층 RVQ 개선

**모델 규모화 전략**(HunyuanVideo, Wan 2025):[7][8]

병렬화 기법의 중요성:
- Fully-Sharded Data Parallel (FSDP)
- Context Parallel (CP): 시퀀스 길이 확장성
- Tensor Parallelism

스케일링 법칙 재정의:
- 확산 모델은 LLM보다 하이퍼파라미터에 민감
- 배치 크기, 학습률에 따른 민감도 높음[3]

**제로샷 성능 극대화**:[1]
1. 다양한 사전훈련 목표(무조건 생성, 조건부 생성, 자가 감독 학습, 마스킹)
2. 프롬프트 엔지니어링(체인-오브-쏘트, 음수 프롬프트, 자동 재작성)
3. 작업 체이닝 최적화(중간 표현의 분포 내 유지, 교사 강제)

#### 4.3 근본적 연구 방향

**의미론적 이해와 인과관계 학습**
- 구조적 인과 모델(SCM) 통합
- 개입적 데이터(Interventional Data) 활용
- 물리 기반 손실 함수 설계

**공정성과 편향 완화**[1]
VideoPoet의 발견: "Young Adults"(18-35), "Male", "Light Skin Tone" 분포 편향
- 다양성 증대 데이터 수집
- 공정성 인식 손실 함수
- 사후 처리 보정 기법

***

### 5. 최종 평가

VideoPoet은 단순한 논문 이상으로 **비디오 생성의 패러다임 전환을 촉발한 선도 연구**입니다. 그 한계가 명확함에도 불구하고, 다중모달 LLM의 유연성과 확장성이 비디오 생성 분야에서도 실현 가능함을 보여주었습니다. 현재의 후속 연구들(VAR, HunyuanVideo, InfinityStar 등)이 이 기초 위에서 더욱 정교해지고 있으며, 향후 5-10년 내 비디오 생성이 이미지 생성 수준의 성숙도에 도달할 것으로 예상됩니다.[3][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e1a6b517-539c-4660-ad8e-c70f7637dbf8/2312.14125v4.pdf)
[2](https://arxiv.org/abs/2411.02385)
[3](https://yenchenlin.github.io/blog/2025/01/08/video-generation-models-explosion-2024/)
[4](https://openreview.net/forum?id=sKNIjS2brr)
[5](https://arxiv.org/html/2405.15223)
[6](https://arxiv.org/pdf/2501.13918.pdf)
[7](https://arxiv.org/html/2412.03603v2)
[8](https://arxiv.org/abs/2503.20314)
[9](https://arxiv.org/pdf/2312.14125.pdf)
[10](https://arxiv.org/html/2503.02341v1)
[11](http://arxiv.org/pdf/2403.15377v4.pdf)
[12](http://arxiv.org/pdf/2412.08879.pdf)
[13](https://arxiv.org/pdf/2306.05424.pdf)
[14](https://arxiv.org/abs/2409.12499)
[15](https://arxiv.org/pdf/2310.12724.pdf)
[16](https://proceedings.mlr.press/v235/kondratyuk24a.html)
[17](https://www.amazon.science/publications/zero-shot-customized-video-editing-with-diffusion-feature-transfer)
[18](https://research.google/blog/videopoet-a-large-language-model-for-zero-shot-video-generation/)
[19](https://huggingface.co/blog/video_gen)
[20](https://ditflow.github.io)
[21](https://dl.acm.org/doi/10.5555/3692070.3693075)
[22](https://homangab.github.io/gen2act/Gen2Act-Paper.pdf)
[23](https://openaccess.thecvf.com/content/CVPR2023W/L3D-IVU/papers/Doshi_Zero-Shot_Action_Recognition_With_Transformer-Based_Video_Semantic_Embedding_CVPRW_2023_paper.pdf)
[24](https://arxiv.org/html/2507.13942v1)
[25](https://arxiv.org/html/2503.17539v1)
[26](https://arxiv.org/pdf/2205.09853.pdf)
[27](https://arxiv.org/pdf/2311.15127.pdf)
[28](http://arxiv.org/pdf/1906.02634v1.pdf)
[29](https://arxiv.org/pdf/2401.09084.pdf)
[30](https://www.ijcai.org/proceedings/2023/0642.pdf)
[31](https://www.sciencedirect.com/science/article/abs/pii/S003132032500562X)
[32](http://pengxi.me/wp-content/uploads/2020/12/2020TNNLS-Deep-Multimodal-Transfer-Learning.pdf)
[33](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Towards_Fast_Adaptation_of_Pretrained_Contrastive_Models_for_Multi-Channel_Video-Language_CVPR_2023_paper.pdf)
[34](https://openai.com/index/video-generation-models-as-world-simulators/)
[35](https://dl.acm.org/doi/10.1145/3078971.3078994)
[36](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04830.pdf)
[37](https://github.com/FoundationVision/VAR)
