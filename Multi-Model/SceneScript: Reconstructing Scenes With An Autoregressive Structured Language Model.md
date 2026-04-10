
# SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model

> **논문 정보**
> - **제목:** SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model
> - **저자:** Armen Avetisyan, Christopher Xie, Henry Howard-Jenkins, Tsun-Yi Yang, Samir Aroudj, Suvam Patra, Fuyang Zhang, Duncan Frost, Luke Holland, Campbell Orme, Jakob Engel, Edward Miller, Richard Newcombe, Vasileios Balntas (Meta Reality Labs)
> - **발표:** ECCV 2024
> - **arXiv:** [arXiv:2403.13064](https://arxiv.org/abs/2403.13064) (2024년 3월 19일)
> - **공식 페이지:** [projectaria.com/scenescript](https://projectaria.com/scenescript)
> - **GitHub:** [facebookresearch/scenescript](https://github.com/facebookresearch/scenescript)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

SceneScript는 오토리그레시브(Autoregressive) 토큰 기반 접근법을 사용하여 장면 모델 전체를 구조화된 언어 명령어(Structured Language Commands)의 시퀀스로 직접 생성하는 방법이다. 이 표현 방식은 Transformer 및 LLM의 최근 성공에서 영감을 받아, 씬을 메시(mesh), 복셀 그리드(voxel grid), 포인트 클라우드(point cloud), 또는 레이디언스 필드(radiance field)로 표현하는 기존의 전통적 방법에서 크게 벗어난다.

핵심 주장을 한 문장으로 요약하면:
> *"3D 실내 장면 이해는 메시나 포인트 클라우드가 아닌, 구조화 언어 명령어 시퀀스를 오토리그레시브하게 생성하는 방식으로 더 효과적으로 해결될 수 있다."*

---

### 1.2 주요 기여 (Contributions)

| 기여 | 설명 |
|------|------|
| **①** 새로운 씬 표현 패러다임 | 3D 장면을 언어 명령어 시퀀스로 표현 |
| **②** 씬 언어 인코더-디코더 아키텍처 | 시각 데이터 → 구조화 명령어 엔드투엔드 추론 |
| **③** 대규모 합성 데이터셋 공개 | Aria Synthetic Environments(ASE), 10만 개 실내 씬 |
| **④** SOTA 달성 | 건축 레이아웃 추정에서 최고 성능 |
| **⑤** 언어 확장성 | 네트워크 변경 없이 새 명령어 추가로 새 태스크 수행 |

SceneScript 훈련을 위해 10만 개의 고품질 실내 씬으로 구성된 **Aria Synthetic Environments(ASE)**라�� 대규모 합성 데이터셋을 생성·공개하였으며, 건축 레이아웃 추정에서 SOTA를 달성하고 3D 객체 검출에서도 경쟁력 있는 결과를 보였다. 또한 구조화 언어에 새로운 명령어를 간단히 추가함으로써 3D 객체 파트 재구성 같은 새로운 태스크에 용이하게 적응할 수 있음을 보여준다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

전통적인 3D 씬 표현은 본질적인 트레이드오프를 가진다: 메시는 상세한 기하구조를 제공하지만 메모리 집약적이고 의미적 수준에서 편집하기 어렵고, 복셀 그리드는 균일한 샘플링을 제공하지만 해상도가 높아질수록 효율성이 낮아지고, 포인트 클라우드는 원시 기하 데이터를 포착하지만 연결성 정보가 부족하며, NeRF 같은 암묵적 표현은 새로운 시점 합성에 뛰어나지만 해석 가능성이 부족하다.

SceneScript는 이러한 문제들을 **구조화 언어 기반 표현**으로 해결하고자 한다:
- 해석 불가능하고 편집 어려운 기존 기하 표현의 한계 극복
- 단일 통합 아키텍처로 다중 태스크(레이아웃 추정 + 객체 검출 + 파트 재구성) 수행
- 합성 데이터로만 학습해도 실제 씬으로 일반화 가능한 시스템 구축

---

### 2.2 제안하는 방법

#### 2.2.1 구조화 씬 언어 (Structured Scene Language)

씬 언어는 다양한 씬 요소를 표현할 수 있는 명령어 및 파라미터 집합을 정의한다: 벽 명령어(`make_wall`)는 위치·방향·높이를 가진 평면 벽 세그먼트를 정의하고, 문·창문 명령어(`make_door`, `make_window`)는 크기·위치 및 부모 벽 참조를 포함하며, 객체 명령어(`make_bbox`)는 위치·방향·크기·카테고리를 가진 3D 바운딩 박스를 정의하고, 파트 명령어(`make_prim`)는 더 세밀한 객체 재구성을 위한 객체 컴포넌트를 정의한다.

각 명령어는 다음과 같은 파라미터 형태를 가진다:

$$\text{command} = \langle \text{cmd type}, p_1, p_2, \ldots, p_n \rangle$$

예를 들어 벽 명령어:

$$\text{make wall}(x_0, y_0, x_1, y_1, z_{\min}, z_{\max})$$

문 명령어:

$$\text{make door}(x_{\text{pos}}, y_{\text{pos}}, \text{width}, \text{height}, \theta)$$

객체 바운딩 박스:

$$\text{make bbox}(x, y, z, w, h, d, \theta, \text{class})$$

여기서 $(x, y, z)$는 중심 좌표, $(w, h, d)$는 크기, $\theta$는 회전각, $\text{class}$는 의미 카테고리이다.

#### 2.2.2 토큰화 (Tokenization)

구조화 언어를 정수 토큰 시퀀스로 직렬화하는 과정을 **토큰화**라 한다. 목표는 구조화 언어 명령어 시퀀스와 Transformer 디코더가 예측할 수 있는 정수 토큰 시퀀스 간의 **전단사(bijective) 매핑**을 구성하는 것이다. 이 스키마는 고정 크기 슬롯이나 패딩 없이 1D 토큰 패킹을 가능하게 하며, 서브시퀀스의 수나 계층 구조에 어떤 제약도 없이 PART 토큰으로 유연하게 분리된다.

학습 목적 함수는 표준적인 **Next-Token Prediction Loss**를 따른다:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(c_t \mid c_1, c_2, \ldots, c_{t-1},\, \mathbf{F}_{\text{scene}})$$

여기서:
- $c_t$: 시각 $t$에서의 토큰
- $T$: 전체 시퀀스 길이
- $\mathbf{F}_{\text{scene}}$: 인코더로 추출된 씬 잠재 코드(latent scene code)
- $\theta$: 모델 파라미터

이 오토리그레시브 접근법은 모델이 장면 요소 간의 복잡한 의존성(예: 문이 유효한 벽 위에 위치하거나 가구가 의미론적으로 유의미한 배치를 갖는 것)을 포착할 수 있게 한다. 학습 목적 함수는 표준적인 next-token prediction loss로서, 주어진 문맥에서 올바른 토큰의 확률을 최대화하도록 모델이 학습된다.

---

### 2.3 모델 구조

#### 2.3.1 전체 파이프라인 개요

원시 이미지 및 포인트클라우드 데이터가 잠재 코드(latent code)로 인코딩되고, 이것이 씬을 설명하는 명령어 시퀀스로 오토리그레시브하게 디코딩된다.

전체 파이프라인은 세 단계로 구성된다:

$$\underbrace{\text{Raw Images} + \text{Point Cloud}}_{\text{입력}} \xrightarrow{\mathcal{E}} \underbrace{\mathbf{F}_{\text{scene}}}_{\text{씬 잠재 코드}} \xrightarrow{\mathcal{D}_{\text{AR}}} \underbrace{c_1, c_2, \ldots, c_T}_{\text{구조화 언어 명령어 시퀀스}}$$

#### 2.3.2 인코더 (Encoder)

시각 데이터를 잠재 표현으로 인코딩하기 위해 SceneScript는 세 가지 인코더 변형을 사용한다: 포인트 클라우드, 포즈 이미지 세트, 그리고 두 가지의 조합이다. 입력 모달리티에 관계없이, 인코딩된 씬 표현은 트랜스포머 기반 디코더에 의해 구조화 언어 명령어 시퀀스로 디코딩된다.

세 가지 인코더 변형을 수식으로 표현하면:

| 인코더 유형 | 수식 표현 |
|------------|----------|
| 포인트 클라우드 인코더 | $\mathbf{F}\_{\text{geo}} = \mathcal{E}_{\text{geo}}(\mathcal{P})$ |
| 이미지(RGB) 인코더 | $\mathbf{F}\_{\text{rgb}} = \mathcal{E}_{\text{rgb}}(\mathcal{I})$ |
| 결합(Lifted Feature) 인코더 | $\mathbf{F}\_{\text{combined}} = \mathcal{E}_{\text{lifted}}(\mathcal{P}, \mathcal{I})$ |

구체적으로, 기하 인코더는 Project Aria의 Machine Perception Services에서 얻은 Semi-dense 포인트클라우드를 사용한다. 포인트 클라우드는 5cm 해상도로 이산화되고, Sparse 3D 합성곱 라이브러리를 활용하여 풀링된 특징을 생성한다. 기하 인코더 $\mathcal{E}_{\text{geo}}$는 일련의 다운 합성곱을 적용하여 최하위 레벨에서 포인트 수를 축소한다.

포인트 클라우드 인코더는 정확한 기하 정보 포착에 특히 효과적이며, 이미지 인코더는 의미적 단서와 세밀한 디테일 추출에 뛰어나다. 결합 인코더는 두 모달리티의 이점을 모두 활용하여 최상의 전반적 성능을 달성한다.

#### 2.3.3 디코더 (Decoder)

토큰 시퀀스는 임베딩 레이어를 통과한 후 위치 인코딩 레이어를 거치고, 인코딩된 씬 코드와 함께 여러 트랜스포머 디코더 레이어에 입력되며, 이때 **인과적 어텐션 마스크(causal attention mask)**를 사용하여 오토리그레시브 생성을 보장한다.

디코더는 GPT 모델과 유사한 트랜스포머 기반 오토리그레시브 아키텍처를 따른다.

디코더의 핵심 연산은 다음과 같다:

$$\hat{c}_t = \text{softmax}(\mathbf{W}_o \cdot \text{TransformerDecoder}(\mathbf{E}(c_{<t}), \mathbf{F}_{\text{scene}}))$$

여기서:
- $\mathbf{E}(c_{<t})$: 이전까지의 토큰 임베딩
- $\mathbf{F}_{\text{scene}}$: 인코더 출력 (크로스 어텐션 컨텍스트)
- $\mathbf{W}_o$: 출력 프로젝션 가중치

인코더는 대규모 포인트 클라우드를 소수의 특징으로 풀링하는 일련의 3D Sparse 합성곱 블록으로 구성된다. 이후 트랜스포머 디코더가 인코더의 특징을 크로스 어텐션의 컨텍스트로 활용하여 오토리그레시브 방식으로 토큰을 생성한다.

---

### 2.4 성능 향상

실험 결과 SceneScript는 F1 기반 메트릭에서 Aria Synthetic Environments 데이터셋의 여러 메트릭에 걸쳐 SOTA 레이아웃 추정 기준선에 비해 상당한 성능 우위를 보였으며, 두 기준선 방법 모두 세밀한 디테일을 다룰 때 정확도가 크게 하락하였다.

비교 기준선 모델 대비 주요 성능 요약:

| 태스크 | 결과 |
|--------|------|
| 건축 레이아웃 추정 | **SOTA** (SceneCAD, RoomFormer 대비 우월) |
| 3D 객체 검출 | 경쟁력 있는 수준 |
| 3D 파트 재구성 | Proof-of-concept 수준에서 확장 가능성 입증 |

비최적화 기본 PyTorch 트랜스포머 기준, 256개 토큰(벽, 문, 창문, 객체 바운딩 박스를 포함하는 중간 크기 씬)을 디코딩하는 데 약 2~3초가 소요된다.

---

### 2.5 한계 (Limitations)

이 표현에 기반한 재구성은 단순하고 거친 기하구조로 귀결되는 경향이 있어, 매우 세밀한 수준에서의 복잡한 뉘앙스를 놓칠 수 있다. 이러한 한계들은 향후 연구 및 최적화의 영역을 잠재적으로 강조하며, 명령어 정의 프로세스의 자동화 및 복잡한 기하 디테일을 정확하게 포착하는 표현 능력 향상 기법의 탐구를 목표로 한다.

현재 모델은 비에고센트릭(non-egocentric) 기기(예: 액션 카메라, 모바일 폰)에서는 작동하지 않는데, 이는 현재 모델이 Project Aria 글래스에서 캡처되는 시퀀스를 시뮬레이션하여 훈련되기 때문이다. 다만 다른 카메라 모델과 다른 종류의 렌즈를 사용하여 파인튜닝될 수 있다.

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 합성→실제 씬 일반화

SceneScript는 에고센트릭 비디오가 주어지면 3D 씬 표현을 직접 예측하며, **합성 실내 환경으로만 훈련되었음에도 다양한 실제 씬으로 일반화**한다는 점이 핵심적이다.

이 연구는 방대한 ASE 합성 데이터셋을 활용하여 실제 씬에 대한 강력한 일반화와 새로운 씬 명령어로의 쉬운 확장성을 입증한다.

$$\text{Generalization Gap} = \mathcal{L}_{\text{real}} - \mathcal{L}_{\text{synthetic}} \approx \text{small} \quad \text{(SceneScript 실험 결과)}$$

### 3.2 언어 확장에 의한 태스크 일반화

SceneScript의 재구성 충실도는 새로운 명령어의 추가를 통해 향상될 수 있다. 초기에는 구조적 룸 레이아웃이 `make_wall`, `make_door`, `make_window` 세 가지 명령어로 표현된다. `make_bbox`의 추가만으로 객체 검출 형태의 장면 콘텐츠가 재구성에 포함된다. 마지막으로 `make_prim`으로 선택된 대상 클래스에 대해 씬의 전반적 구조와 콘텐츠뿐만 아니라 씬 객체 자체의 훨씬 세밀한 재구성이 가능해진다. **중요한 것은, 이러한 디테일의 각 수준이 SceneScript의 네트워크 아키텍처 변경 없이 가능하며, 오직 추론하는 구조화 언어의 표현력을 증가시킴으로써 달성된다.**

시각적 입력과 네트워크 아키텍처를 고정한 채 SceneScript 언어에 대한 간단한 확장을 통해 SceneScript를 새로운 태스크로 쉽게 확장할 수 있음을 증명한다. 이를 3D 객체 검출 문제에서 입증하여 건축 레이아웃과 3D 방향 바운딩 박스를 공동 추론하는 방법을 도출한다. 또한 3D 객체 재구성, 곡선 개체, 개체 구성, 개체 상태 표현 등 새로운 태스크에 대한 진입 장벽을 크게 낮추는 방법의 개념 증명 실험을 추가로 제시한다.

### 3.3 모달리티 결합에 의한 일반화

SceneScript는 Sparse 3D 합성곱 포인트클라우드 인코더, RGB RayTran 기반 특징 볼륨 인코더, 제안된 lifted feature point 인코더의 세 가지 인코더 변형으로 씬 레이아웃 추정을 수행하며, 세 가지 시나리오 모두에서 동일한 트랜스포머 디코더가 사용된다.

$$\mathbf{F}_{\text{scene}} = \begin{cases} \mathbf{F}_{\text{geo}} & \text{(Point Cloud Only)} \\ \mathbf{F}_{\text{rgb}} & \text{(Image Only)} \\ \text{Fuse}(\mathbf{F}_{\text{geo}}, \mathbf{F}_{\text{rgb}}) & \text{(Combined: Best Performance)} \end{cases}$$

### 3.4 LLM과의 결합 가능성

구조화 언어 명령어 기반의 씬 표현을 구축하는 능력은 미래에 복잡하고 효율적인 씬 재구성 방법의 핵심 구성 요소가 될 것이며, 이를 통해 범용 LLM과 결합하여 사용할 수 있게 될 것이라 믿는다.

SceneScript는 순수 언어로 씬을 표현하고 예측하기 때문에, 시뮬레이션 데이터를 설명하는 데 사용되는 언어를 확장함으로써 씬 요소를 손쉽게 확장할 수 있다.

---

## 4. 해당 논문이 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (1) 3D 씬 이해의 언어화 패러다임 전환
SceneScript는 3D 씬 이해 작업에 있어 **언어 모델 패러다임의 도입**이라는 근본적인 방향 전환을 제시하였다. 이미 후속 연구들이 등장하고 있다:

언어 모델 기반의 최근 지각-일반화(perception-generalist) 접근법들이 3D 씬 레이아웃 추정을 포함한 다양한 태스크에서 SOTA를 달성하고 있다. 그러나 이러한 접근법들은 본질적으로 느린 오토리그레시브 next-token 예측에 의존한다. 이에 **Fast SceneScript**는 다중 토큰 예측(MTP)을 사용하여 오토리그레시브 반복 횟수를 줄이고 추론을 크게 가속하는 새로운 구조화 언어 모델을 제안한다. MTP가 속도를 향상시키는 한편, 신뢰도 낮은 토큰 예측은 정확도를 크게 떨어뜨릴 수 있다. 이를 필터링하기 위해 자기 추측 디코딩(SSD)을 구조화 언어 모델에 적용하고 토큰 신뢰도의 향상된 스코어링 메커니즘을 가진 신뢰도 기반 디코딩(CGD)을 도입한다.

3D 지각에서 SceneScript는 오토리그레시브 구조화 언어 모델로서 3D 지각을 시퀀스 생성 태스크로 간주하며, 다양한 태스크의 출력을 표현하도록 구조화 언어가 설계된다. SceneScript는 단일 통합 아키텍처와 인터페이스로 레이아웃 추정, 객체 검출, 거친 3D 객체 파트 재구성을 수행할 수 있다.

#### (2) SpatialLM과 같은 후속 연구 촉발

SpatialLM은 3D 포인트 클라우드 데이터를 처리하고 구조화된 3D 씬 이해 출력을 생성하는 대형 언어 모델로, 레이아웃 추정과 객체 검출 태스크를 모두 지원한다.

#### (3) AR/VR 연구에의 직접적 기여

SceneScript의 모델 가중치는 2024년 9월에 외부 학술 연구자들에게 공개되었다. 이는 AR 글래스, 로봇 내비게이션, 실내 디자인 등 다양한 응용 연구의 가속화를 가능하게 한다.

---

### 4.2 향후 연구 시 고려해야 할 점

#### (1) 추론 속도 최적화
언어 모델 기반 지각-일반화 접근법들이 다양한 태스크에서 SOTA를 달성하고 있지만, 이러한 접근법들은 본질적으로 느린 오토리그레시브 next-token 예측에 의존한다. 따라서 MTP, 스펙큘러티브 디코딩, 모델 압축 등의 기법 연구가 필수적이다.

#### (2) 카메라 도메인 일반화
현재는 Project Aria 글래스와 같은 에고센트릭 카메라에 특화되어 있으며, 다른 카메라 모델이나 렌즈로 파인튜닝이 필요하다. 다양한 카메라 도메인에서의 일반화 방법론 연구가 필요하다.

#### (3) 기하 디테일 정밀도 향상
현재의 표현 기반 재구성은 단순하고 거친 기하구조에 머무르는 경향이 있으며, 매우 상세한 수준의 복잡한 뉘앙스를 놓칠 수 있다. 명령어 정의 자동화 및 복잡한 기하 디테일을 정확하게 포착하는 능력 향상이 향후 연구 방향이다.

#### (4) 명령어 집합 설계의 자동화
현재 `make_wall`, `make_door`, `make_bbox`, `make_prim` 등의 명령어는 수동으로 설계되어 있다. 이를 자동으로 확장하거나 데이터로부터 학습하는 방법론이 필요하다.

#### (5) 실제 데이터와의 격차 해소
실용적인 실내 씬 재구성을 위해 Aria Synthetic Environments를 공개하였으며, 이는 에고센트릭 씬 워크스루와 대응하는 ground truth 명령어 시퀀스의 훈련 쌍으로 구성된다. Transformer가 방대한 양의 데이터를 필요로 하기 때문에 10만 개의 합성 씬을 생성하였으며, 이는 실제 데이터에서는 실현 불가능한 규모이다. 실제 환경 데이터와의 도메인 갭을 줄이는 도메인 적응 연구가 중요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법 | 씬 표현 | 특징 |
|------|------|------|---------|------|
| **PolyGen** (Nash et al.) | 2020 | Autoregressive | 3D 메시 | 저수준 언어, 복잡한 기하 가능 |
| **DeepSVG** (Carlier et al.) | 2020 | Transformer VAE | 2D 벡터 그래픽 | 생성·보간 가능 |
| **DeepCAD** (Wu et al.) | 2021 | Autoregressive | 3D CAD | 저수준 명령, 2D→3D |
| **Pix2Seq** (Chen et al.) | 2022 | Encoder-Decoder | 2D 객체 검출 | SceneScript의 직접 선행 연구 |
| **Point2Seq** (Xue et al.) | 2022 | Recurrent | 3D 바운딩 박스 | 파라미터의 오토리그레시브 순서가 성능 향상 |
| **RoomFormer** | 2022 | Transformer | 다각형 룸 레이아웃 | 병렬 예측, 빠른 추론 |
| **RayTran** (Tyszkiewicz et al.) | 2022 | Ray-traced Transformer | 다중 객체 3D | 다중 비디오 프레임에서 전체론적 추론 |
| **SceneScript** (Avetisyan et al.) | **2024** | Autoregressive Encoder-Decoder | **구조화 언어 명령어** | **고수준 명령어, 해석 가능, 태스크 확장성** |
| **SpatialLM** | 2024 | LLM + 포인트 클라우드 | 구조화 출력 | 범용 LLM 기반 씬 이해 |
| **Fast SceneScript** | 2024/2025 | MTP + SSD + CGD | 구조화 언어 명령어 | SceneScript의 추론 속도 대폭 개선 |

### SceneScript vs. 선행 연구의 핵심 차별점

SceneScript와 가장 가까운 연구는 Pix2Seq이다. Pix2Seq은 SceneScript와 유사한 아키텍처를 제안하지만 2D 객체 검출에서만 실험하여 도메인별 증강 전략이 필요하다. 또 다른 밀접한 관련 연구는 Point2Seq로서, 연속적인 3D 바운딩 박스 파라미터를 오토리그레시브하게 회귀하는 순환 네트워크를 훈련한다. 흥미롭게도 파라미터의 오토리그레시브 순서가 객체 검출의 현재 표준을 능가함을 발견한다.

DeepCAD는 DeepSVG와 유사한 저수준 언어와 아키텍처를 제안하지만 2D 벡터 그래픽 대신 3D CAD 모델에 적용한다. SceneScript의 접근법은 **고수준 명령어를 사용함으로써 해석 가능성과 의미적 풍부함을 제공**한다는 점에서 두드러지며, 저수준 명령어는 임의로 복잡한 기하구조를 표현할 수 있지만 전체 씬 표현 시 시퀀스가 지나치게 길어진다는 단점이 있다.

---

## 참고 자료 (References)

1. **arXiv 원본 논문:** Avetisyan, A. et al. (2024). *SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model*. arXiv:2403.13064. https://arxiv.org/abs/2403.13064

2. **ECCV 2024 출판 버전:** Springer, LNCS vol. 15119. https://doi.org/10.1007/978-3-031-73030-6_14

3. **ECCV 2024 공식 PDF:** https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07833.pdf

4. **Meta AI 공식 연구 페이지:** https://ai.meta.com/research/publications/scenescript-reconstructing-scenes-with-an-autoregressive-structured-language-model/

5. **Project Aria 공식 페이지:** https://www.projectaria.com/scenescript/

6. **GitHub 코드 저장소:** https://github.com/facebookresearch/scenescript

7. **Hugging Face 논문 페이지:** https://huggingface.co/papers/2403.13064

8. **Semantic Scholar:** https://www.semanticscholar.org/paper/SceneScript:-Reconstructing-Scenes-With-An-Language-Avetisyan-Xie/45ecc57010340239b48783957a7e5cbf8b814aeb

9. **후속 연구 — Fast SceneScript (2024/2025):** *Fast SceneScript: Accurate and Efficient Structured Language Model via Multi-Token Prediction*. arXiv:2512.05597. https://arxiv.org/html/2512.05597v1

10. **비교 연구 — Pix2Seq (2022):** Chen, T. et al. *Pix2seq: A Language Modeling Framework for Object Detection*. ICLR 2022.

11. **비교 연구 — RoomFormer:** Transformer-based multi-room polygon generation. Structured3D, SceneCAD 벤치마크 SOTA.

12. **비교 연구 — PolyGen (2020):** Nash, C. et al. *PolyGen: An Autoregressive Generative Model of 3D Meshes*. ICML 2020.

13. **비교 연구 — DeepCAD (2021):** Wu, R. et al. *DeepCAD: A Deep Generative Network for Computer-Aided Design Models*. ICCV 2021.

14. **비교 연구 — SpatialLM (2024):** Large language model for 3D point cloud scene understanding.
