# Caption Anything: Interactive Image Description with Diverse Multimodal Controls

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

**Caption AnyThing (CAT)**은 기존 Controllable Image Captioning(CIC)의 두 가지 근본적 한계를 해결하고자 합니다:

1. **데이터 의존성 문제**: 기존 CIC 모델은 사람이 직접 주석을 단 `(이미지, 텍스트, 제어신호)` 튜플에 의존하며, 데이터 규모의 한계로 인해 제어 신호 이해 능력이 제한됨
2. **제어 신호의 경직성**: 사전 정의된 단일/소수의 제어 신호만 지원하여 새로운 제어 차원 확장이 어려움

CAT의 핵심 주장은 **"사전 학습된 파운데이션 모델(SAM + BLIP-2 + ChatGPT)을 조합하면, 별도의 학습(training-free) 없이도 다양한 멀티모달 제어가 가능한 이미지 캡셔닝 시스템을 구축할 수 있다"**는 것입니다.

### 1.2 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| ① Training-Free CIC Framework | 파운데이션 모델 기반으로 인간 주석 데이터 의존성 제거 |
| ② 다양한 제어 신호 및 통합 표현 | 시각(3종) + 언어(4종) 제어를 통합 표현으로 일반화 |
| ③ 강력한 사용자 상호작용 능력 | Object-centric chatting, Paragraph captioning 등으로 확장 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

```
[기존 문제]
Vanilla Captioning → 제어 불가, 사용자 의도 반영 불가
Controllable Captioning → 제한적 제어 신호, 대규모 주석 데이터 필요
```

기존 방법론의 한계를 수식으로 표현하면:

기존 CIC 모델은 다음과 같은 조건부 생성 문제를 학습 데이터에 의존하여 풀었습니다:

$$P(C \mid I, s) = \prod_{t=1}^{T} P(c_t \mid c_{ < t}, I, s; \theta)$$

여기서:
- $C = \{c_1, c_2, \ldots, c_T\}$: 생성된 캡션 토큰 시퀀스
- $I$: 입력 이미지
- $s$: 사전 정의된 제어 신호(control signal)
- $\theta$: 학습된 모델 파라미터

이 방식은 $s$가 사전에 고정되어야 하고, 대규모 주석 데이터 $\mathcal{D} = \{(I_i, C_i, s_i)\}_{i=1}^{N}$가 필요하다는 문제가 있습니다.

### 2.2 제안하는 방법

CAT는 다음과 같은 **트리플렛 솔버(triplet solver)**로 구성됩니다:

$$\text{CAT} = \{\mathcal{F}_{\text{seg}},\ \mathcal{F}_{\text{cap}},\ \mathcal{F}_{\text{ref}}\}$$

**전체 파이프라인 수식:**

$$\hat{C} = \mathcal{F}_{\text{ref}}\left(\mathcal{F}_{\text{cap}}\left(I,\ \mathcal{F}_{\text{seg}}(I, v)\right),\ l\right)$$

여기서:
- $v \in \{\text{point, box, trajectory}\}$: 시각적 제어 입력
- $l \in \{\text{sentiment, length, language, factuality}\}$: 언어적 제어 입력
- $\mathcal{F}_{\text{seg}}$: SAM 기반 세그멘터 → 픽셀 수준 마스크 $M$ 생성
- $\mathcal{F}\_{\text{cap}}$: BLIP-2 기반 캡셔너 → 원시 캡션 $C_{\text{raw}}$ 생성
- $\mathcal{F}_{\text{ref}}$: ChatGPT 기반 텍스트 정제기 → 최종 캡션 $\hat{C}$ 생성

#### 2.2.1 Segmenter (SAM)

SAM은 다양한 시각적 프롬프트를 통해 세분화 마스크를 생성합니다:

$$M = \mathcal{F}_{\text{seg}}(I, v), \quad v \in \{\mathbf{p}, \mathbf{b}, \tau\}$$

- $\mathbf{p} = (x, y)$: 포인트 클릭 좌표
- $\mathbf{b} = (x_1, y_1, x_2, y_2)$: 바운딩 박스
- $\tau = \{(x_t, y_t)\}_{t=1}^{T}$: 마우스 궤적(trajectory)

SAM은 SA-1B 데이터셋(11M 이미지, 10억 개 마스크)으로 사전 학습되어 뛰어난 Zero-shot 전이 능력을 보입니다.

#### 2.2.2 Captioner (BLIP-2)

BLIP-2는 동결된(frozen) 이미지 인코더와 LLM을 Q-Former로 연결합니다:

$$C_{\text{raw}} = \mathcal{F}_{\text{cap}}(I \odot M)$$

여기서 $\odot$는 마스크 적용 연산입니다. BLIP-2는 다음의 3단계 학습 구조를 가집니다:

$$\mathcal{L}_{\text{BLIP2}} = \mathcal{L}_{\text{ITC}} + \mathcal{L}_{\text{ITM}} + \mathcal{L}_{\text{LM}}$$

- $\mathcal{L}_{\text{ITC}}$: Image-Text Contrastive Loss
- $\mathcal{L}_{\text{ITM}}$: Image-Text Matching Loss
- $\mathcal{L}_{\text{LM}}$: Language Modeling Loss

#### 2.2.3 Visual Chain-of-Thought (VCoT)

배경 정보로 인한 캡셔닝 오류를 방지하기 위해 단계적 추론을 적용합니다:

$$\text{Step 1: } q_1 = \mathcal{F}_{\text{cap}}(I \otimes M,\ \text{"what is this?"}) \rightarrow a_1$$

$$\text{Step 2: } C_{\text{raw}} = \mathcal{F}_{\text{cap}}\left(I,\ \text{"Describe the } a_1 \text{ in the image."}\right)$$

여기서:
- $I \otimes M$: 배경을 흰색으로 대체한 마스크 적용 이미지
- $a_1$: 관심 객체의 카테고리 (1단계 식별 결과)

이 방식은 NLP의 Chain-of-Thought(CoT) 프롬프팅에서 영감을 받았습니다.

#### 2.2.4 Text Refiner (ChatGPT)

언어 제어 신호를 텍스트 프롬프트로 통합하여 ChatGPT에 입력:

$$\hat{C} = \mathcal{F}_{\text{ref}}\left(C_{\text{raw}},\ \mathbf{l}\right)$$

$$\mathbf{l} = \{l_{\text{sentiment}},\ l_{\text{length}},\ l_{\text{language}},\ l_{\text{factuality}}\}$$

예시 프롬프트 템플릿:
$$\text{Prompt} = \left[\text{"Refine: "} + C_{\text{raw}} + \text{"; Sentiment: "} + l_s + \text{"; Length: "} + l_n + \text{..."}\right]$$

### 2.3 모델 구조

```
사용자 입력
    │
    ├── 시각적 제어 (v) ──→ [SAM Segmenter] ──→ Pixel Mask M
    │   (Point/Box/Trajectory)                        │
    │                                                 ↓
    │                              [BLIP-2 Captioner] + VCoT
    │                                                 │
    │                                          Raw Caption C_raw
    │                                                 │
    └── 언어적 제어 (l) ──────────────────→ [ChatGPT Refiner]
        (Sentiment/Length/                            │
         Language/Factuality)                         ↓
                                              최종 캡션 Ĉ
```

**통합 표현(Unified Representation):**

| 제어 유형 | 통합 표현 |
|----------|----------|
| 시각적 제어 | 픽셀 수준 마스크 $M \in \{0,1\}^{H \times W}$ |
| 언어적 제어 | 텍스트 프롬프트 $\mathbf{l} \in \Sigma^*$ |

### 2.4 성능 향상

논문은 주로 **정성적 평가(qualitative evaluation)**에 집중하며, 공개된 벤치마크에서의 정량적 비교는 제한적입니다. 아래는 논문에서 보고된 질적 성능 향상입니다:

**Visual CoT 효과:**

| 조건 | 예시 결과 |
|------|---------|
| VCoT 없음 | "a brown bear and her cub" (배경 포함) |
| VCoT 있음 | "the bear cub is sleeping on the mother's back" (객체 집중) |

**다양한 언어 제어 효과:**

$$\text{동일 이미지} + l_{\text{sentiment}} \rightarrow \begin{cases} \text{Positive+Factual: "The majestic horse boasts..."} \\ \text{Negative+Imagination: "The horse galloping uncontrollably"} \end{cases}$$

### 2.5 한계점

1. **정량적 평가 부재**: 표준 벤치마크(COCO Captions, NoCaps 등)에 대한 BLEU, CIDEr, METEOR 등 정량적 비교가 없음
2. **파이프라인 지연(Latency)**: SAM → BLIP-2 → ChatGPT 순차 처리로 실시간 응용에서 지연 발생
3. **API 의존성**: ChatGPT API에 의존하여 비용 및 프라이버시 문제 발생 가능 (오픈소스 LLM으로 대체 가능하다고 언급)
4. **오류 전파(Error Propagation)**: 세그멘터의 오분할 → 캡셔너의 오캡션 → 정제기의 오정제로 이어지는 연쇄 오류 가능
5. **복잡한 장면 처리**: 객체가 겹치거나 매우 작은 경우 SAM의 분할 정확도 저하
6. **할루시네이션(Hallucination)**: LLM 기반 정제 과정에서 사실과 다른 내용 생성 가능성

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 Zero-Shot 일반화의 근거

CAT의 일반화 성능은 각 구성 요소의 사전 학습 규모에서 비롯됩니다:

$$\text{일반화 능력} \propto \underbrace{|\mathcal{D}_{\text{SAM}}|}_{\text{SA-1B: 1B masks}} + \underbrace{|\mathcal{D}_{\text{BLIP2}}|}_{\text{129M image-text pairs}} + \underbrace{|\mathcal{D}_{\text{ChatGPT}}|}_{\text{인터넷 규모 텍스트}}$$

**SAM의 일반화**: SA-1B(11M 이미지, 10억 마스크)로 학습되어 의료 이미지, 위성 사진 등 새로운 도메인에서도 강건한 분할 성능을 보입니다.

**BLIP-2의 일반화**: Frozen 인코더와 Q-Former 구조로 다양한 이미지 분포에서 Zero-shot 캡셔닝이 가능합니다:

$$\mathcal{F}_{\text{cap}}(I_{\text{novel}}) \approx \mathcal{F}_{\text{cap}}(I_{\text{train}}), \quad \forall I_{\text{novel}} \notin \mathcal{D}_{\text{train}}$$

**ChatGPT의 일반화**: RLHF(인간 피드백 강화학습)로 튜닝된 ChatGPT는 다국어, 다양한 스타일에 걸쳐 일반화된 텍스트 정제 능력을 보입니다.

### 3.2 일반화 성능 향상 메커니즘

#### 3.2.1 모듈형 설계(Modular Design)를 통한 확장성

통합 표현 덕분에 새로운 제어 신호 추가가 기존 모듈 수정 없이 가능합니다:

$$\text{New Control} \xrightarrow{\text{변환}} \begin{cases} M \in \{0,1\}^{H \times W} & \text{(시각적 제어의 경우)} \\ \mathbf{l} \in \Sigma^* & \text{(언어적 제어의 경우)} \end{cases}$$

예를 들어, 깊이 맵(depth map)이나 열화상(thermal image)도 마스크로 변환하면 즉시 통합 가능합니다.

#### 3.2.2 도메인 전이 가능성

| 응용 도메인 | SAM 일반화 | BLIP-2 일반화 | ChatGPT 일반화 |
|-----------|-----------|--------------|---------------|
| 의료 영상 | 높음 (새 도메인에서도 강건) | 중간 (의료 전문 용어 부족 가능) | 높음 (다양한 지식 보유) |
| 위성 이미지 | 높음 | 중간 | 높음 |
| 예술 작품 | 중간 | 높음 | 높음 |
| 저조도 이미지 | 낮음 | 낮음 | 해당 없음 |

#### 3.2.3 Visual Chain-of-Thought의 일반화 기여

VCoT는 모델이 배경 편향(background bias)에서 벗어나게 하여, 분포 외(out-of-distribution) 이미지에서도 관심 객체에 집중할 수 있게 합니다:

$$P(C \mid I, M)_{\text{VCoT}} = P(C \mid I, M, a_1) \cdot P(a_1 \mid I \otimes M)$$

이는 단순한 조건부 확률 $P(C \mid I, M)$보다 배경 영향을 더 효과적으로 차단합니다.

#### 3.3 일반화 성능 향상을 위한 추가 제안

**앙상블 기반 불확실성 감소:**

$$\hat{M} = \frac{1}{K} \sum_{k=1}^{K} \mathcal{F}_{\text{seg}}^{(k)}(I, v)$$

여기서 $K$개의 서로 다른 시각적 프롬프트를 사용하여 마스크 앙상블을 구성하면 세그멘테이션 불확실성을 줄일 수 있습니다.

**도메인 적응형 프롬프팅:**

$$\mathbf{l}_{\text{adapted}} = \arg\max_{\mathbf{l}} P(\hat{C} \mid C_{\text{raw}}, \mathbf{l}, \mathcal{D}_{\text{domain}})$$

도메인 특화 프롬프트 최적화를 통해 특정 응용 분야에서의 일반화 성능을 향상시킬 수 있습니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 비교 테이블

| 연구 | 발표 연도 | 학습 방식 | 시각적 제어 | 언어적 제어 | Zero-shot | 주요 특징 |
|------|---------|----------|-----------|-----------|----------|---------|
| **SentiCap** (Mathews et al.) | 2016 | 지도학습 | ✗ | 감성(긍/부정) | ✗ | 최초 감성 캡셔닝 |
| **Length-Controllable** (Deng et al.) | ECCV 2020 | 지도학습 | ✗ | 길이 | ✗ | 길이 제어 |
| **Localized Narratives** (Pont-Tuset et al.) | ECCV 2020 | 지도학습 | 마우스 궤적 | ✗ | ✗ | 궤적 기반 주석 |
| **LoopCAG** (Yan et al.) | ACL 2021 | 지도학습 | 공간+시간 | 제한적 | ✗ | 시공간 제어 |
| **Show, Control & Tell** (Cornia et al.) | CVPR 2019 | 지도학습 | 바운딩 박스 | ✗ | ✗ | 지역 시퀀스 제어 |
| **BLIP-2** (Li et al.) | 2023 | 사전학습 | ✗ | 자연어 프롬프트 | ✓ | Q-Former 구조 |
| **ConZIC** (Zeng et al.) | 2023 | Zero-shot | ✗ | 복합 스타일 | ✓ | 샘플링 기반 폴리싱 |
| **Visual ChatGPT** (Wu et al.) | 2023 | Training-free | 제한적 | ✓ | ✓ | 시각 파운데이션 모델 체이닝 |
| **CAT (본 논문)** | 2023 | Training-free | 점/박스/궤적 | 감성/길이/언어/사실성 | ✓ | SAM+BLIP-2+ChatGPT 통합 |

### 4.2 주요 경쟁 연구와의 심층 비교

#### 4.2.1 ConZIC (Zeng et al., 2023, arXiv:2303.02437)

ConZIC는 CAT와 유사하게 Zero-shot 제어 가능 캡셔닝을 시도하지만 다른 접근 방식을 사용합니다:

$$C^* = \arg\max_{C} \left[\lambda_1 \mathcal{S}_{\text{CLIP}}(C, I) + \lambda_2 \mathcal{S}_{\text{style}}(C, l)\right]$$

- **ConZIC**: 샘플링 기반 반복 최적화로 언어 스타일 제어
- **CAT**: 파이프라인 방식으로 시각+언어 통합 제어

**CAT의 우위**: 시각적 제어(SAM) 지원, 객체 중심 캡셔닝 가능

#### 4.2.2 Visual ChatGPT (Wu et al., 2023, arXiv:2303.04671)

Visual ChatGPT는 다양한 시각 파운데이션 모델을 LangChain으로 체이닝하는 방식을 사용합니다. CAT는 이 아이디어에서 영감을 받았으며, 특히 Object-centric chatting 확장에서 LangChain을 활용합니다.

**차별점**: CAT는 이미지 캡셔닝에 특화된 VCoT 메커니즘과 통합 제어 표현을 제공합니다.

#### 4.2.3 GPT-4V / LLaVA 계열 (2023년 이후)

CAT 발표 이후 등장한 **GPT-4V**, **LLaVA** (Liu et al., 2023), **InstructBLIP** (Dai et al., 2023) 등은 엔드-투-엔드 멀티모달 LLM으로 CAT의 파이프라인 방식보다 더 강력한 통합 성능을 보입니다. 그러나 이들은 대규모 파인튜닝 데이터가 필요하다는 점에서 CAT의 Training-free 특성과 대조됩니다.

### 4.3 패러다임 변화 관점

```
2020년 이전: 지도학습 기반 단일 제어 신호
     ↓
2020-2022: 대규모 사전학습 + 제한적 Zero-shot
     ↓
2023 (CAT): Training-free 파운데이션 모델 조합
     ↓
2023 이후: 엔드-투-엔드 멀티모달 LLM (GPT-4V, LLaVA 등)
```

---

## 5. 향후 연구에 미치는 영향과 고려할 점

### 5.1 향후 연구에 미치는 영향

#### 5.1.1 파운데이션 모델 조합 패러다임 확립

CAT는 **"특화된 파운데이션 모델들을 모듈화하여 조합"**하는 접근법이 실용적임을 보여주었습니다. 이는 이후 연구에서:

- **Composed Foundation Models**: 서로 다른 도메인의 전문 모델을 조합하는 방향
- **API-based AI Systems**: 외부 API를 활용한 시스템 설계
- **Training-free Adaptation**: 학습 없이 새로운 태스크에 적응하는 방법론

등의 연구에 영향을 미칩니다.

#### 5.1.2 통합 제어 표현(Unified Control Representation)의 중요성

$$\text{임의의 제어 신호} \xrightarrow{\text{변환 함수}} \text{통합 표현 공간}$$

이 개념은 멀티모달 AI에서 **제어 신호의 추상화와 일반화**를 위한 중요한 설계 원칙으로 자리잡을 수 있습니다.

#### 5.1.3 Visual Chain-of-Thought의 영향

VCoT는 시각-언어 추론에서 단계적 추론의 중요성을 보여주며, 이는:

$$\text{복잡한 시각 추론} = \sum_{k=1}^{K} \text{단계별 서브태스크}_k$$

의 형태로 분해하는 **단계적 멀티모달 추론** 연구에 기여합니다.

#### 5.1.4 객체 중심 상호작용(Object-Centric Interaction) 연구 촉진

전통적인 이미지 전체 이해에서 **특정 객체 중심의 상호작용**으로의 패러다임 전환을 촉진합니다. 이는 로봇공학, 증강현실, 교육 기술 등의 응용 분야에 직접적인 영향을 미칩니다.

### 5.2 향후 연구 시 고려할 점

#### 5.2.1 정량적 평가 체계 구축

현재 CAT는 정성적 평가에 의존합니다. 향후 연구에서는:

$$\text{평가 메트릭} = \alpha \cdot \text{BLEU} + \beta \cdot \text{CIDEr} + \gamma \cdot \text{Control-Adherence Score}$$

**Control-Adherence Score**: 제어 신호와 생성 캡션의 일치도를 측정하는 새로운 메트릭 개발이 필요합니다.

감성 제어 준수도 예시:

$$\text{Sentiment-Adherence} = \frac{|\{C_i : \text{sentiment}(C_i) = l_s\}|}{|C|}$$

#### 5.2.2 오류 전파 최소화

파이프라인 구조에서의 오류 전파를 수식화하면:

$$\epsilon_{\text{total}} \leq \epsilon_{\text{seg}} + \epsilon_{\text{cap}}(\epsilon_{\text{seg}}) + \epsilon_{\text{ref}}(\epsilon_{\text{cap}})$$

이를 최소화하기 위한 **피드백 루프(feedback loop)** 메커니즘 연구가 필요합니다:

$$M^* = \arg\min_{M} \mathcal{L}_{\text{consistency}}(\mathcal{F}_{\text{cap}}(I, M), C_{\text{target}})$$

#### 5.2.3 엔드-투-엔드 통합 가능성 탐구

CAT의 모듈형 파이프라인을 단일 모델로 통합하는 연구:

$$\mathcal{F}_{\text{unified}}(I, v, l) = \hat{C}$$

이는 **지식 증류(Knowledge Distillation)**를 활용하여 파이프라인의 각 모듈을 경량화된 단일 모델로 압축하는 방향으로 발전 가능합니다.

#### 5.2.4 실시간 처리 최적화

현재 CAT의 추론 지연을 분석하면:

$$t_{\text{total}} = t_{\text{SAM}} + t_{\text{BLIP2}} + t_{\text{ChatGPT API}}$$

$t_{\text{ChatGPT API}}$는 네트워크 지연에 종속되므로, **로컬 LLM(LLaMA, Mistral 등)**으로 대체하는 연구가 필요합니다.

#### 5.2.5 할루시네이션(Hallucination) 방지

LLM 기반 정제 과정에서의 사실 오류를 방지하기 위한 **검증 모듈** 추가:

$$\hat{C}_{\text{verified}} = \mathcal{F}_{\text{verify}}(\hat{C}, I, M)$$

이는 생성된 캡션이 이미지 내용과 일치하는지 검증하는 역할을 합니다.

#### 5.2.6 다중 언어 및 문화적 맥락 고려

CAT는 중국어, 프랑스어, 스페인어 등 다국어를 지원하지만, 문화적 맥락에 따른 캡션의 적절성에 대한 연구가 부족합니다. 향후 연구에서는:

- 문화권별 표현 방식의 차이 반영
- 저자원 언어(low-resource languages)에서의 성능 평가
- 문화적 편향(cultural bias) 측정 및 완화 방법론

등을 고려해야 합니다.

#### 5.2.7 프라이버시 및 보안 고려

- **민감한 객체 마스킹**: 얼굴, 개인정보 등을 자동으로 감지하여 캡션에서 제외
- **적대적 공격(Adversarial Attack) 방어**: 시각적 프롬프트 조작을 통한 악의적 캡션 생성 방지
- **API 의존성 최소화**: 오픈소스 모델로의 전환을 통한 데이터 프라이버시 보장

---

## 참고자료

**주요 논문 (본 문서에서 직접 인용):**

1. Wang, T., Zhang, J., Fei, J., et al. "Caption Anything: Interactive Image Description with Diverse Multimodal Controls." *arXiv:2305.02677v3*, 2023.

2. Kirillov, A., Mintun, E., Ravi, N., et al. "Segment Anything." *arXiv:2304.02643*, 2023.

3. Li, J., Li, D., Savarese, S., & Hoi, S. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." *arXiv:2301.12597*, 2023.

4. OpenAI. "GPT-4 Technical Report." 2023.

5. Brown, T., et al. "Language Models are Few-Shot Learners." *NeurIPS*, 2020.

6. Wei, J., et al. "Chain of Thought Prompting Elicits Reasoning in Large Language Models." *arXiv:2201.11903*, 2022.

7. Pont-Tuset, J., et al. "Connecting Vision and Language with Localized Narratives." *ECCV 2020*.

8. Deng, C., et al. "Length-Controllable Image Captioning." *ECCV 2020*.

9. Zeng, Z., et al. "ConZIC: Controllable Zero-Shot Image Captioning by Sampling-Based Polishing." *arXiv:2303.02437*, 2023.

10. Wu, C., et al. "Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models." *arXiv:2303.04671*, 2023.

11. Touvron, H., et al. "LLaMA: Open and Efficient Foundation Language Models." *arXiv:2302.13971*, 2023.

12. Chung, H.W., et al. "Scaling Instruction-Finetuned Language Models (FLAN-T5)." *arXiv:2210.11416*, 2022.

13. Kojima, T., et al. "Large Language Models are Zero-Shot Reasoners." *arXiv:2205.11916*, 2022.

14. Ouyang, L., et al. "Training Language Models to Follow Instructions with Human Feedback (InstructGPT)." *NeurIPS*, 2022.

**GitHub 코드베이스:**
- Caption-Anything: https://github.com/ttengwang/Caption-Anything
- Segment Anything: https://segment-anything.com

> **⚠️ 주의사항**: 본 답변은 제공된 논문 PDF(arXiv:2305.02677v3)를 직접 분석하여 작성하였습니다. GPT-4V, LLaVA, InstructBLIP 등 2023년 중반 이후 발표된 연구와의 비교는 해당 논문 발표 시점 이후의 연구이므로, 구체적인 수치 비교보다는 트렌드 분석 수준에서 기술하였습니다.
