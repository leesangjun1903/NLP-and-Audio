
# Cosmos 3: Omnimodal World Models for Physical AI 

---

## 1. 핵심 주장 및 주요 기여 요약

Cosmos 3는 세계 최초의 완전 오픈 옴니모달(omnimodal) 모델로, 텍스트·이미지·비디오·주변 음향(ambient sound)·액션에 걸쳐 네이티브 비전 추론과 멀티모달 생성을 지원하며, 최첨단 합성 데이터 생성 및 Physical AI 정책 모델 개발을 가능하게 합니다.

**핵심 주장:**
Cosmos 3는 Physical AI의 근본적인 과제, 즉 로봇·자율주행차(AV)·비전 에이전트가 제한된 학습 데이터와 분절된 시뮬레이션 스택 환경에서 실세계에 일반화(generalize)할 수 있도록 하는 문제에 도전합니다.

**주요 기여:**
- 이전 Cosmos 버전들에서 세계 생성(Cosmos Predict), 제어 생성(Cosmos Transfer), 장면 이해(Cosmos Reason), 정책 생성(Cosmos Policy) 등이 별도 모델로 분리되어 있었던 것을, Cosmos 3는 단일 모델에서 서로 다른 모달리티를 하나의 통합 순전파(forward pass)로 추론·생성할 수 있도록 통합합니다.
- 유연한 입력-출력 구성을 지원함으로써 비전-언어 모델, 비디오 생성기, 월드 시뮬레이터, 세계-액션 모델을 단일 프레임워크로 통합합니다.
- 텍스트·이미지·비디오·주변 음향·액션을 네이티브로 이해·생성하며, 선도적인 물리 정확도로 Physical AI 학습 및 평가 주기를 수개월에서 수일로 단축합니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

이것이 로보틱스에서의 '일반화 격차(generalization gap)'입니다. 제너럴리스트 로봇이 아직 제품으로 존재하지 않는 이유이며, 바로 Cosmos 3가 해결하기 위해 구축된 과제입니다.

이전 Cosmos 출시는 세계 생성, 물리적 이해, 제어된 장면 생성을 서로 다른 모델과 워크플로우로 분리해 놓았습니다. 이러한 파편화로 인해 Physical AI 시스템 개발 비용이 높고, 모달리티 간 정보 흐름이 단절되는 문제가 있었습니다.

또한 Sora 2나 Veo 3 같은 창의적 비디오 모델은 물리 위반(오브젝트의 갑작스러운 등장·소멸, 물이 거꾸로 흐르는 등)이 발생하며, 이들은 2초 동안 그럴듯해 보이도록 최적화되어 있습니다. 반면 Cosmos 3는 질량 연속성, 접촉 역학, 마찰 등 보존 법칙을 따르는 출력을 생성하도록 최적화되어 있습니다.

---

### 2-2. 제안하는 방법 및 핵심 수식

#### (a) Mixture-of-Transformers (MoT) 아키텍처

Cosmos 3는 추론을 위한 자기회귀(AR) 트랜스포머와 멀티모달 생성을 위한 확산 트랜스포머(DM)를 결합한 통합 MoT(Mixture-of-Transformers) 아키텍처로 구축된 옴니모달 월드 모델입니다.

입력 시퀀스는 두 개의 서브시퀀스로 분리됩니다: 다음 토큰 예측(next-token prediction)으로 추론·이해를 처리하는 AR 서브시퀀스와, 반복적 노이즈 제거(iterative denoising)로 생성을 처리하는 DM 서브시퀀스. AR과 DM 토큰은 각 트랜스포머 레이어 내에서 별도 파라미터 세트를 사용하지만, 공동 어텐션(joint attention)을 통해 상호작용합니다.

**Dual-Tower 구조:**
Reasoner Tower는 이미지·비디오·텍스트 같은 멀티모달 관찰을 해석하는 VLM으로, 자기회귀 아키텍처를 사용하여 입력을 해석하고 움직임, 오브젝트 상호작용, 물리적 맥락을 이해합니다. 이는 생성 전 세계에 대해 추론하는 '뇌' 역할을 합니다.
Generator Tower는 미래 관찰과 액션 시퀀스를 생성하며, 확산 기반 프로세스를 통해 Reasoner Tower의 이해를 조건으로 하는 물리 인식 비디오 및 액션 출력을 생성합니다. Reasoner는 독립적으로 호출될 수 있지만, Generator는 가이드 생성을 위해 항상 두 타워를 활성화합니다.

**확산 모델의 목적 함수 (Diffusion Objective):**

확산 기반 Generator Tower는 아래와 같은 노이즈 제거 목적 함수를 학습합니다:

$$\mathcal{L}_{\text{DM}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\!\left(\mathbf{x}_t,\, t,\, \mathbf{c}_{\text{AR}}\right) \right\|^2 \right]$$

여기서:
- $\mathbf{x}_t$: 타임스텝 $t$에서 노이즈가 추가된 토큰 (비디오, 오디오, 액션)
- $\boldsymbol{\epsilon}$: 추가된 가우시안 노이즈
- $\boldsymbol{\epsilon}_\theta$: 노이즈 예측 네트워크
- $\mathbf{c}_{\text{AR}}$: AR Reasoner Tower로부터의 조건 컨텍스트

**자기회귀 목적 함수 (AR Objective):**

Reasoner Tower는 표준 언어 모델 목적 함수를 따릅니다:

$$\mathcal{L}_{\text{AR}} = -\sum_{i=1}^{N} \log P_\theta\!\left(x_i \mid x_{<i},\, \mathbf{v}\right)$$

여기서:
- $x_i$: $i$번째 토큰
- $x_{<i}$: 이전 토큰 시퀀스
- $\mathbf{v}$: 비전/이미지/비디오 토큰 컨텍스트

---

#### (b) 3D Multimodal RoPE (mRoPE)

Cosmos 3는 언어·비전·오디오·액션 토큰을 통합 어텐션 프레임워크 내에서 공동 모델링하며, 모달리티 전반에 걸쳐 일관되게 일반화하는 위치 임베딩 체계를 설계하는 것이 핵심 과제입니다. 3D MRoPE(Bai et al., 2025a)에서 영감을 받아, 물리적 시간 축을 따라 비디오·오디오·액션 토큰을 정렬하는 절대 시간 인덱싱을 가진 3D MRoPE를 설계하였습니다.

원래의 3D MRoPE는 각 어텐션 헤드의 히든 차원을 시간(temporal), 높이(height), 너비(width) 성분으로 나누며, 시간 성분은 이산 토큰 인덱스만 기록합니다. 이 설계는 이미지와 비디오 이해 태스크에는 충분하지만, 비디오·오디오·액션 토큰이 서로 다른 프레임율이나 샘플링 속도로 동시에 생성될 수 있는 환경에서는 부적절합니다.

**3D mRoPE 위치 인코딩 수식:**

각 모달리티에 대한 위치 인덱스 트리플렛 $(t, h, w)$는 다음과 같이 할당됩니다:

$$\text{RoPE}(q, k) = q \cdot R(t, h, w) \cdot k^\top$$

$$R(t, h, w) = R_\text{time}(t) \otimes R_\text{height}(h) \otimes R_\text{width}(w)$$

언어 토큰은 $t = h = w$를 사용하고, 비디오 토큰은 세 축 모두에서 변하며, 액션 및 오디오 토큰은 시간 좌표만 사용합니다($h = w = 0$). 모달리티 오프셋 $k$가 텍스트와 비전의 시간 범위를 분리하며, FPS 변조(modulation)는 프레임 인덱스를 스케일된 시간 위치에 매핑하여 16, 24, 30 FPS에서 동일한 실제 지속 시간이 동일한 위치 범위를 차지하도록 합니다(기준 FPS: 24).

---

#### (c) 모달리티 토크나이저

각 모달리티는 전용 인코더에 의해 먼저 인코딩됩니다: 시각적 이해를 위한 ViT, 시각/오디오 생성을 위한 VAE, 액션을 위한 도메인 인식 벡터. 이후 공유 표현 공간으로 프로젝션됩니다.

핵심 기술 혁신 중 하나는 비디오 토크나이저로, 원시 비디오 프레임을 이산 토큰으로 변환합니다(LLM이 텍스트를 토큰으로 변환하는 것과 유사). 이를 통해 비디오는 모델의 1급 입출력 형식이 됩니다. 이 토크나이저는 Cosmos가 장기 비디오 시퀀스를 효율적으로 처리·생성할 수 있게 하며, 이는 장면이 시간에 따라 어떻게 변화하는지 이해해야 하는 로보틱스 응용에서 매우 중요합니다.

---

### 2-3. 모델 구조 상세

Cosmos 3는 Super(32B)와 Nano(8B) 두 가지 변형으로 출시되었습니다.

Cosmos3-Super는 Qwen3-VL 32B 아키텍처를 채택하며, 히든 크기 5120, 어텐션 헤드 64개, KV 헤드 8개, 헤드 차원 128, FFN 차원 25,600을 사용합니다. 모든 모델은 이중 타워 MoT 아키텍처를 공유합니다.

**훈련 데이터:**
사전 학습 데이터 혼합은 Nemotron Nano 2 데이터 컬렉션에서 서브셀렉션된 1,970만 개 샘플과 수학, 비디오, 공간 그라운딩, 지시 따르기 능력 향상을 위해 추가 큐레이션된 230만 개 샘플로 구성됩니다.

**데이터 큐레이션 파이프라인:**
2단계 데이터 큐레이션 파이프라인은 시맨틱 중복 제거(semantic deduplication)에 이어 AI-judge 품질 필터링으로 구성됩니다. 첫 번째 단계는 대화 수준에서 멀티모달 근접 중복(near-duplicate)을 제거합니다.

---

### 2-4. 성능 향상

Cosmos 3는 액션 조건부 비디오 생성(action-conditioned video generation)을 지원하며, 예를 들어 "로봇 팔을 왼쪽으로 15도 이동하여 실린더를 잡아라"와 같은 액션을 지정하면 그 동작이 물리적으로 그럴듯하게 일어나는 비디오를 생성합니다.

하나의 모델에서 텍스트, 이미지, 비디오, 액션 입력으로부터 현실적이고 물리적으로 타당한 비디오 월드 생성, 움직임·인과관계·공간 관계 등 물리적 특성 추론, 현재 상태에 기반한 미래 비디오 및 액션 시퀀스 예측이 가능합니다.

---

### 2-5. 한계

논문 및 관련 자료에서 확인된 주요 한계:

1. **훈련 데이터 편향**: 로봇 시연 데이터는 2026년 현재 여전히 $300,000짜리 원격 조작 장비를 가진 사람이 로봇 팔을 직접 움직이는 방식으로 생성되며, 시간당 $50~150의 비용으로 50~200개의 시연을 생산합니다. 고품질 실제 데이터 수집의 비용 문제가 여전히 존재합니다.

2. **시뮬레이션-현실 격차(Sim-to-Real Gap)**: 합성 환경이 R&D 파이프라인의 표준이 되기 위해서는 물리, 메모리, 안전 제약이 사실감(photorealism)만큼 엄격하게 측정되어야 합니다.

3. **특수 도메인 미세조정 필요**: SFT(Supervised Fine-Tuning)를 통해 개발자가 자체 데이터로 Cosmos 3 모델을 적응시킬 수 있으며, 커스텀 비디오 데이터셋을 위한 비전 생성 포스트 트레이닝과 로보틱스 및 Physical AI 워크플로우를 위한 액션 지향 레시피가 포함됩니다. 개발자는 로보틱스, 자율주행, 창고 자동화 등 대상 도메인에 맞게 커스터마이징할 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 핵심 일반화 메커니즘

Cosmos 3는 제한된 학습 데이터와 분절된 시뮬레이션 스택 환경에서 로봇·AV·비전 에이전트가 실세계에 일반화할 수 있도록 하는 Physical AI의 근본적인 문제를 다룹니다.

**① 시뮬레이터 기반 구현화 공간 추론:**

시각적 공간 QA와 구현화 액션을 실행 가능한 시뮬레이터 상태로부터 레이블을 도출하여 연결합니다. 답변은 카메라, 오브젝트 자세/박스, 깊이, 마스크, 가시성, 실행 가능한 영역으로부터 계산되고, 프로그래밍 방식 및 VLM 크리틱에 의해 검증됩니다. 커리큘럼은 메트릭 기하학, 공간 프레임, 물리적 의미론, 행동 가능한 그라운딩, 시점 역학, 구현화 구성을 포함합니다.

**② 통합 아키텍처에 의한 크로스 모달 일반화:**

핵심은 "비디오"가 아니라 "세계"입니다. 대부분의 생성 비디오 모델이 시각적으로 매력적인 콘텐츠를 생성하도록 설계된 반면, Cosmos 3는 물리 세계가 어떻게 작동하는지를 이해하고 모델링하도록 설계되어 있습니다. 픽셀에만 집중하지 않고 실세계 환경을 지배하는 오브젝트, 액션, 움직임, 인과-결과 관계에 대해 추론합니다.

**③ FPS 변조를 통한 시간적 일반화:**

3D mRoPE에서 FPS 변조 설계($t_{\text{scaled}} = t_{\text{frame}} \times \frac{f_{\text{base}}}{f_{\text{target}}}$)를 통해 16, 24, 30 FPS 등 다양한 프레임률에서 학습된 시간적 패턴이 일관되게 적용되며, 이는 다양한 실제 센서 환경에 대한 일반화를 지원합니다.

**④ 합성 데이터를 통한 데이터 효율성 향상:**

일부 모델은 합성 학습 데이터 생성을, 다른 모델은 기존 데이터 레이블링을, 또 다른 모델은 미래 결과 예측이나 액션 직접 생성을 담당합니다. 이들이 함께 작동하여 더 빠르게 학습하고, 더 잘 일반화하며, 훨씬 적은 비용의 실세계 데이터 수집으로 Physical AI 시스템을 구축하는 툴킷을 형성합니다.

**⑤ RoboLab 벤치마크:**

RoboLab은 태스크 제너럴리스트 로봇 정책을 평가하기 위한 시뮬레이션 벤치마크입니다. 이를 통해 다양한 태스크에 대한 일반화 성능을 체계적으로 측정합니다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

| 모델 | 기관 | 아키텍처 | 주요 특징 | 한계 |
|------|------|---------|---------|------|
| **Cosmos 3** (2026) | NVIDIA | MoT (AR + DM 통합) | 완전 오픈 옴니모달, 액션 생성, 물리 정확도 | 실세계 데이터 수집 비용 |
| **Genie 3** | DeepMind | - | 실시간 3D 환경 생성 | 클로즈드 플랫폼 |
| **V-JEPA 2** | Meta | JEPA | 제로샷 로봇 플래닝, 예측 집중 | 생성 능력 제한 |
| **Sora** | OpenAI | DiT | 고품질 비디오 생성 | 물리 정확도 낮음, 클로즈드 API |

Meta의 V-JEPA는 물리 세계의 표현을 학습하는 비디오 기반 모델로, 생성보다 예측에 집중합니다. 강력한 연구 모델이지만 Cosmos의 완전한 생성-이해 능력보다 범위가 좁습니다.

Sora는 고품질 비디오를 생성하지만 AI 학습을 위한 물리적으로 정확한 시뮬레이션이 아닌 창의적 미디어 제작에 최적화되어 있습니다. 클로즈드 API이며 액션 조건화 또는 로보틱스 파이프라인과의 직접 통합을 지원하지 않습니다.

Cosmos 3는 DeepMind의 Genie 3, Meta의 VIMA-2 같은 모델들과 비교하여 자기 완결적 해석, 추론, 액션을 지원하는 전체론적 세계 시뮬레이션에 집중합니다. 이는 JEPA 스타일의 이론에서 볼 수 있는 추상적 잠재 모델과 Cosmos, Genie 같은 시각적 미래 생성 모델 사이의 격차를 줄이는 중요한 도약을 나타냅니다.

Meta의 V-JEPA 2는 다른 접근 방식을 취합니다: 100만 시간 이상의 인터넷 비디오로 사전 학습하고 62시간 미만의 로봇 궤적으로 미세조정하여 최첨단 액션 예측 및 제로샷 로봇 플래닝을 달성합니다.

Genie 3는 범용 월드 모델링에서 DeepMind의 혁신적 도약을 나타내며, 720p 해상도와 초당 24프레임으로 몰입형 3D 환경을 실시간 생성하며 수 분간 일관성을 유지합니다. 텍스트 프롬프트가 제공되면 1인칭 및 등각 투시 등 다양한 시점에서 사용자와 에이전트가 탐색할 수 있는 대화형 가상 환경을 생성합니다.

---

## 5. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

**① Physical AI 패러다임 전환:**
Cosmos 3는 통합 아키텍처 내에서 언어·이미지·비디오·오디오·액션을 이해하고 생성할 수 있는 옴니모달 월드 모델입니다. 단순한 VLM도 아니고, 비디오 생성기도 아니며, 오디오-비주얼 생성 모델도 아니고, 물리 시뮬레이터/세계-액션 모델도 아닌, 이 모든 것의 통합입니다.

**② 오픈 생태계 형성:**
NVIDIA는 차세대 오픈 월드 모델 발전을 위해 Agile Robots, Black Forest Labs, Generalist, LTX, Runway, Skild AI 등 주요 AI 연구소 및 로보틱스 리더들과 NVIDIA Cosmos Coalition을 출범시켰습니다.

**③ 합성 데이터 생성의 표준화:**
Cosmos 3 출시의 핵심 요소는 완전 오픈 훈련 레시피 세트입니다. 모델 체크포인트를 넘어, 새로운 도메인·구현체·데이터셋에 Cosmos 3를 적응시키기 위한 코드, 설정, 워크플로우를 제공합니다.

**④ 벤치마크 기준 재정의:**
TokenBench는 로봇 조작, 드라이빙, 자기 중심적(egocentric), 웹 비디오 등 광범위한 도메인을 커버하여 Cosmos-Tokenizer 평가를 표준화하는 포괄적 벤치마크입니다.

---

### 5-2. 향후 연구 시 고려할 점

1. **Sim-to-Real 격차 정량화**: 합성 환경이 R&D 파이프라인의 표준이 되려면, 물리·메모리·안전 제약이 사실감만큼 엄격하게 측정되어야 합니다. 이를 위한 표준화된 평가 프로토콜 연구가 필요합니다.

2. **효율적인 실세계 데이터 수집**: 로보틱스 워크플로우, 자율 시스템 데이터셋, 스마트 공간 인식, 합성 물리 세계 훈련 데이터를 구축하는 경우 Cosmos 3는 가장 중요한 릴리스 중 하나입니다. 그러나 실제 데이터와 합성 데이터의 최적 혼합 비율에 대한 연구가 필요합니다.

3. **장기 의존성 모델링**: Genie 3의 비교 모델이 보여주는 장기 시각적 메모리처럼, Cosmos 3의 장기 시나리오에서의 일관성 유지 연구가 중요합니다.

4. **멀티에이전트 확장**: DeepMind가 암시하듯 더 긴 세션, 풍부한 액션 어휘, 멀티에이전트 소셜 물리학으로의 확장은 Cosmos 3 기반 연구의 자연스러운 다음 단계입니다.

5. **윤리·안전 고려**: NVIDIA는 기업 사용을 위해 Cosmos 출력을 안전하게 유지하는 업샘플링 모델과 가드레일을 개발하였습니다. 악용 방지 및 안전성 보장 연구가 병행되어야 합니다.

6. **MoT 아키텍처 최적화**: AR과 DM 타워 간 공동 어텐션 계산의 효율성 향상, 특히 엣지 디바이스에서의 추론 최적화가 중요한 연구 과제입니다.

---

## 📚 참고 자료 (출처)

| # | 제목/출처 | URL |
|---|----------|-----|
| 1 | **Cosmos 3 Technical Report** (NVIDIA, 2026-06-01) | https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf |
| 2 | **NVIDIA Newsroom: NVIDIA Launches Cosmos 3** | https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai |
| 3 | **GitHub - NVIDIA/cosmos** | https://github.com/NVIDIA/cosmos |
| 4 | **Hugging Face Blog: Welcome NVIDIA Cosmos 3** | https://huggingface.co/blog/nvidia/cosmos-3-for-physical-ai |
| 5 | **NVIDIA Technical Blog: Develop Physical AI with Cosmos 3** | https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/ |
| 6 | **MindStudio: What Is NVIDIA Cosmos 3?** | https://www.mindstudio.ai/blog/what-is-nvidia-cosmos-3-world-foundation-model |
| 7 | **Baseten: Nvidia Cosmos 3 - Robots finally take over** | https://www.baseten.co/blog/nvidia-cosmos-3-robots-finally-take-over/ |
| 8 | **WaveSpeed Blog: What Is NVIDIA Cosmos3-Nano?** | https://wavespeed.ai/blog/posts/what-is-nvidia-cosmos3-nano/ |
| 9 | **Frank's World: Unraveling Cosmos 3** | https://www.franksworld.com/2026/06/01/unraveling-cosmos-3-nvidias-revolutionary-step-towards-omnimodal-physical-ai/ |
| 10 | **RD World: Google's Genie 3 comparison** | https://www.rdworldonline.com/googles-genie-3-breaks-through-the-real-time-barrier-for-ai-world-models/ |
| 11 | **ThePromptBuddy: V-JEPA 2 vs NVIDIA Cosmos 2026** | https://www.thepromptbuddy.com/prompts/meta-v-jepa-2-vs-nvidia-cosmos-complete-world-foundation-model-comparison-2026 |
| 12 | **GitHub - NVlabs/TokenBench** | https://github.com/NVlabs/TokenBench |
| 13 | **GitHub - NVIDIA Cosmos Organization** | https://github.com/nvidia-cosmos |

> ⚠️ **정확도 주의 사항**: 일부 목적 함수 수식( $\mathcal{L}\_{\text{DM}}$, $\mathcal{L}_{\text{AR}}$ )은 기술 보고서의 공개된 섹션에서 직접 확인된 것이 아니라, 논문에서 채택한 Diffusion Transformer 및 AR Transformer 방법론에 기반하여 표준적으로 재구성한 것입니다. 정확한 수식은 공식 기술 보고서(PDF) 전문을 직접 확인하시길 권장합니다.
