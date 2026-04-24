# Paying More Attention to Image: A Training-Free Method for Alleviating Hallucination in LVLMs
---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문은 Large Vision-Language Models (LVLMs)에서 발생하는 **환각(Hallucination)** 현상의 근본 원인을 **"텍스트 관성(Text Inertia)"** 이라는 새로운 개념으로 정의하고, 추가적인 학습(Training) 없이 이를 완화하는 방법론 **PAI(Pay Attention to Image)** 를 제안합니다.

**텍스트 관성(Text Inertia)**: 이미지 입력이 제거된 상태에서도 LVLMs가 동일한 환각적 설명을 생성하는 현상. 즉, 모델 출력이 시각적 정보가 아닌 언어적 문맥에 의해 주로 결정됨.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **현상 규명** | "Text Inertia" 개념을 최초로 정의 및 실증적 분석 |
| **방법론 제안** | Training-free 추론 개입(Inference Intervention) 방법 PAI 제안 |
| **어텐션 조정** | 이미지 토큰에 대한 셀프-어텐션 가중치를 적응적으로 증폭 |
| **로짓 정제** | 순수 텍스트 입력의 로짓을 차감하여 언어 선행편향 감소 |
| **범용성** | 어떤 디코딩 방법(Greedy, Beam Search, Nucleus)과도 결합 가능 |
| **최초성** | LVLMs에 추론 개입 방식으로 환각 완화를 시도한 최초의 연구 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

LVLMs의 아키텍처는 일반적으로 다음 세 요소로 구성됩니다:

1. **이미지 인코더 (Vision Encoder)**: 이미지를 이미지 토큰으로 변환
2. **프로젝터 (Projector)**: 이미지 토큰을 텍스트 표현 공간으로 매핑
3. **언어 디코더 (Language Decoder)**: LLaMA 계열 모델이 응답 생성

이 과정에서 **규모 불균형 문제(Scale Disparity)** 가 발생합니다. 이미지 인코더보다 훨씬 큰 언어 모델이 멀티모달 이해 과정을 지배하게 되어:

- 이미지 토큰이 많은 시퀀스 비중을 차지함에도 충분한 어텐션을 받지 못함
- LLM의 언어 선행지식(Language Prior)에 의존한 출력이 생성됨
- 결과적으로 시각 입력과 불일치하는 환각 텍스트가 생성됨

논문에서 3개 모델(LLaVA-1.5, Minigpt4, Shikra)을 대상으로 COCO 데이터셋 500개 샘플을 분석한 결과, **전체 환각의 상당 비율이 텍스트 관성에 기인**함을 실증적으로 확인했습니다(Fig. 2).

---

### 2.2 제안하는 방법 (수식 포함)

#### 기본 어텐션 메커니즘

단일 어텐션 헤드 $h$의 연산:

$$\boldsymbol{O}_h = \boldsymbol{A}_h \boldsymbol{V}_h, \quad \boldsymbol{A}_h = \text{softmax}\left(\frac{\boldsymbol{Q}_h \boldsymbol{K}_h^\top}{\sqrt{d_k}}\right) \tag{1}$$

현재 생성 토큰의 어휘 확률 분포:

$$\boldsymbol{y} \sim p_{\text{model}}(\boldsymbol{y} \mid \boldsymbol{X}_I, \boldsymbol{X}_V, \boldsymbol{X}_H) \propto \text{softmax}\left(\text{logit}_{\text{model}}(\boldsymbol{y} \mid \boldsymbol{X}_I, \boldsymbol{X}_V, \boldsymbol{X}_H)\right) \tag{2}$$

여기서 $\boldsymbol{X}_I$는 지시문 표현, $\boldsymbol{X}_V$는 이미지 표현, $\boldsymbol{X}_H$는 이전 생성 토큰 표현입니다.

---

#### **Stage 1: 이미지 어텐션 증폭 (Pay More Attention to Image)**

소프트맥스 이전의 어텐션 가중치 $\tilde{\boldsymbol{A}}$에서 이미지 토큰에 해당하는 부분을 선택적으로 증폭합니다:

$$\tilde{\boldsymbol{A}}_{n,j} = \tilde{\boldsymbol{A}}_{n,j} + \alpha \cdot |\tilde{\boldsymbol{A}}_{n,j}| \quad \text{for } j = m+1 \text{ to } m+n_V \tag{3}$$

- $n$: 현재 생성 중인 마지막 토큰 인덱스
- $j$: 이미지 토큰의 위치 범위 ($m+1$부터 $m+n_V$까지)
- $\alpha$: 개입 강도를 조절하는 하이퍼파라미터
- $|\tilde{\boldsymbol{A}}_{n,j}|$: 원래 어텐션 방향을 신뢰 가능한 방향으로 사용

**설계 철학**: 이미 정렬 훈련(Alignment Training)을 거친 모델의 원래 어텐션 방향이 이미지 기반 신뢰 방향을 제공한다는 가정 하에, 낮은 어텐션 가중치를 가진 헤드는 적게 개입받고 높은 가중치를 가진 헤드는 더 많이 개입받는 **적응적 증폭** 구조를 채택합니다.

**레이어 선행 지식 (Layer Prior)**: BOS 토큰의 Attention Sink 패턴은 얕은 레이어에서는 두드러지지 않고 깊은 레이어에서 나타납니다(Fig. 5). 은닉 상태의 유사도를 계산하여 개입 타이밍을 결정함으로써, 의미적 정보가 안정화된 이후의 레이어에만 선택적으로 개입합니다.

---

#### **Stage 2: 이미지 중심 로짓 정제 (Image-Centric Logit Refine)**

순수 텍스트 입력(이미지 없음)의 로짓을 차감하여 언어 선행편향을 제거합니다:

$$p_{\text{model}} = \gamma \cdot p_{\text{model}}(\boldsymbol{y} \mid \boldsymbol{X}_V, \boldsymbol{X}_I, \boldsymbol{X}_H) - (\gamma - 1) \cdot p_{\text{model}}(\boldsymbol{y} \mid \boldsymbol{X}_I, \boldsymbol{X}_H) \tag{4}$$

- $\gamma$: 언어 선행편향 페널티의 강도를 조절하는 하이퍼파라미터 (기본값 1.1)
- $p_{\text{model}}(\boldsymbol{y} \mid \boldsymbol{X}_I, \boldsymbol{X}_H)$: 이미지 없이 지시문과 이전 응답만으로 생성한 확률 분포

이는 Classifier-Free Guidance(LLM-CFG, Sanchez et al., 2023)와 개념적으로 유사하며, 시각 정보와 텍스트 정보 간의 균형을 잡는 유도 생성 메커니즘입니다.

---

#### CHAIR 평가 메트릭

$$\text{CHAIR}_I = \frac{|\{\text{hallucinated objects}\}|}{\text{all mentioned objects}} \tag{5}$$

$$\text{CHAIR}_S = \frac{|\{\text{captions with hallucinated objects}\}|}{\text{all captions}} \tag{6}$$

---

### 2.3 모델 구조

PAI의 전체 아키텍처(Fig. 4)는 다음과 같이 구성됩니다:

```
[입력]
 ├── 이미지 (X_V) + 지시문 (X_I) + 히스토리 (X_H)
 └── 지시문 (X_I) + 히스토리 (X_H) [이미지 없음 - 비교용]
        ↓
[LVLMs 토크나이저]
        ↓
[Self-Attention 개입 (Stage 1)]
 ├── 소프트맥스 전 어텐션 추출
 ├── 이미지 토큰 어텐션 증폭 (α 파라미터)
 └── 레이어 선행 지식으로 개입 타이밍 결정
        ↓
[LLaMA 디코더]
        ↓
[Image-Centric Logit Refine (Stage 2)]
 └── γ × logit(w/ image) - (γ-1) × logit(w/o image)
        ↓
[최종 출력]
```

**하이퍼파라미터 설정**:
- LLaVA (576 이미지 토큰): $\alpha = 0.5$
- Shikra (긴 이미지 토큰 시퀀스): $\alpha = 0.6$
- Minigpt4 (32 이미지 토큰, 리샘플러 사용): $\alpha = 0.2$
- 모든 모델 공통: $\gamma = 1.1$

---

### 2.4 성능 향상

#### CHAIR 결과 (낮을수록 우수)

| 디코딩 방법 | 모델 | 방법 | $\text{CHAIR}_S$ | $\text{CHAIR}_I$ |
|------------|------|------|-----------------|-----------------|
| Greedy | LLaVA | Vanilla | 46.6 | 13.4 |
| Greedy | LLaVA | **PAI** | **24.8** | **6.9** |
| Beam Search | LLaVA | Vanilla | 46.4 | 14.3 |
| Beam Search | LLaVA | OPERA | 44.6 | 14.4 |
| Beam Search | LLaVA | **PAI** | **21.8** | **5.6** |
| Nucleus | LLaVA | Vanilla | 58.2 | 18.2 |
| Nucleus | LLaVA | VCD | 51.8 | 15.1 |
| Nucleus | LLaVA | **PAI** | **43.4** | **14.7** |

#### GPT-4V 평가 결과

| 모델 | 방법 | Accuracy (C) | Detailedness (D) |
|------|------|--------------|-----------------|
| LLaVA | Greedy | 5.62 | 5.24 |
| LLaVA | **PAI** | **6.46** | **5.36** |
| Minigpt4 | Greedy | 5.8 | 5.74 |
| Minigpt4 | **PAI** | **7.04** | **5.89** |

#### 모델 스케일 확장 실험 (LLaVA-1.5 13B)

| 방법 | $\text{CHAIR}_S$ | $\text{CHAIR}_I$ | POPE Acc | POPE F1 |
|------|-----------------|-----------------|----------|---------|
| Greedy | 44.0 | 12.7 | 85.47 | 86.60 |
| **PAI** | **33.0** | **9.2** | **86.82** | **87.80** |

---

### 2.5 한계점

논문에서 명시한 한계:

1. **LLaMA 종속성**: 현재 오픈소스 LVLMs의 언어 디코더가 주로 LLaMA 계열에 한정되어 있어, 이미지 무시 및 텍스트 관성 문제가 LLaMA 자체에서 기인하는지 여부가 불분명함

2. **성능 상한선 의존성**: PAI는 근본적으로 추론 과정에서 이미지 무시를 완화하는 방법으로, 그 성능 상한은 사전 훈련된 모델의 능력에 의존함. 훈련 과정에서 이 문제를 손실 함수로 반영했을 때 추가적인 성능 향상 가능성이 있음

3. **Nucleus 샘플링 한계**: 핵(nucleus) 샘플링과 결합 시 환각 감소 효과가 상대적으로 미미함. 신뢰할 수 있는 토큰의 우선순위를 높여도 샘플링 풀에 여전히 환각 토큰이 포함될 수 있기 때문

4. **논리적 추론 질문 개선 미흡**: MMHal-Bench 결과(Fig. 6)에서 비교(comparison), 공간 관계(relation) 등 논리적 추론이 필요한 질문 유형에서는 유의미한 개선이 나타나지 않음

5. **하이퍼파라미터 민감도**: 모델마다 적절한 $\alpha$ 값이 다르며, 특히 Minigpt4는 $\gamma$에 매우 민감하게 반응함. 이에 대한 자동화된 설정 방법이 필요

6. **리샘플러 모델의 설명가능성 제한**: Minigpt4처럼 리샘플러를 사용하는 모델은 셀프-어텐션 맵을 원본 이미지 패치로 역추적하기 어려워 시각적 설명가능성(Explainability)이 제한됨

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 다양한 모델에서의 일반화

PAI는 서로 다른 프로젝터 유형을 사용하는 3가지 모델에서 모두 효과를 보였습니다:

| 프로젝터 유형 | 모델 | 일반화 여부 |
|--------------|------|------------|
| 선형 프로젝터 (MLP) | LLaVA-1.5, Shikra | ✅ 효과적 |
| 리샘플러 (Q-Former) | Minigpt4 | ✅ 효과적 (단, α 조정 필요) |

이는 PAI가 프로젝터의 종류에 관계없이 **프로젝터 이후 이미지 토큰에 집중**하는 설계 덕분입니다.

### 3.2 모델 스케일에서의 일반화

LLaVA-1.5 7B → 13B로 모델 스케일을 증가시켰을 때도 PAI의 효과가 유지되었습니다(Table S2):

$$\text{CHAIR}_S: 44.0 \xrightarrow{\text{PAI}} 33.0 \quad (\text{LLaVA-1.5 13B})$$

이는 더 큰 모델에서도 이미지 무시 현상이 존재하며, PAI가 이를 효과적으로 완화함을 보여줍니다.

### 3.3 다양한 디코딩 방법에서의 일반화

PAI는 추론 과정에 개입하는 방식이므로 **디코딩 알고리즘과 독립적**입니다:
- Greedy Decoding ✅
- Beam Search ✅ (OPERA보다 우수)
- Nucleus Sampling ✅ (VCD보다 우수, 다만 효과 제한적)

### 3.4 다양한 태스크에서의 일반화

| 태스크 유형 | 평가 지표 | 일반화 여부 |
|------------|----------|------------|
| 장문 이미지 설명 | CHAIR | ✅ 큰 폭 향상 |
| 단답형 VQA (Single-turn) | POPE | ✅ 소폭 향상 |
| 멀티턴 VQA (Multi-turn) | POPE | ✅ 더 큰 향상 |
| 종합 평가 | MMHal-Bench | ✅ (논리 추론 제외) |
| 인간 수준 평가 | GPT-4V | ✅ |

### 3.5 일반화 성능의 원천 분석

PAI의 일반화 성능이 높은 이유를 다음과 같이 분석할 수 있습니다:

**① 원래 어텐션 방향의 활용**: 정렬 훈련을 통해 학습된 이미지 기반 방향을 그대로 증폭하므로, 별도의 방향 탐색이 필요 없고 모델에 내재된 이미지 이해 능력을 활성화합니다.

**② 적응적 증폭 메커니즘**: $|\tilde{\boldsymbol{A}}_{n,j}|$를 사용하여 낮은 어텐션을 가진 헤드는 덜 개입받고 높은 헤드는 더 개입받으므로, 특정 헤드를 선택하는 복잡한 과정(예: ITI의 신뢰 점수 랭킹) 없이 자연스러운 적응이 이루어집니다.

**③ 레이어 선행 지식의 범용성**: Attention Sink 패턴이 얕은 레이어에서 두드러지지 않는다는 일반적인 트랜스포머 특성을 활용하므로, 다양한 LLaMA 계열 모델에 적용 가능합니다.

**④ Prompt Highlighter 비교 우위**: 이미지 토큰 전체에 균등하게 상수를 더하는 방식(PH)보다 원래 스케일에 기반하여 증폭하는 PAI가 더 우수함(Table S3):

$$\text{CHAIR}_S(\text{PH}) = 52.8 > \text{CHAIR}_S(\text{PAI}) = 23.4$$

### 3.6 일반화의 한계

- **비-LLaMA 아키텍처**: GPT-4V, Gemini 등 비공개 혹은 다른 아키텍처 모델에의 적용 가능성이 검증되지 않음
- **도메인 특수성**: 의료, 위성 이미지 등 특수 도메인에서의 일반화 여부가 불확실함
- **언어 다양성**: 영어 외 다국어 환경에서의 성능이 검증되지 않음

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4.1 환각 완화 방법론 분류 및 비교

```
환각 완화 방법
├── 훈련 기반 (Training-based)
│   ├── 데이터 필터링: CIEM (Hu et al., 2023)
│   ├── RLHF: LLaVA-RLHF / Factually Augmented RLHF (Sun et al., 2023)
│   └── 아키텍처 조정: InternVL (Chen et al., 2023), Monkey (Li et al., 2023)
│
├── 후처리 기반 (Post-processing)
│   ├── LURE (Zhou et al., 2023): 상태 탐지기 + 수정 모델
│   └── Woodpecker (Yin et al., 2023): 외부 시각 모델 활용
│
└── 훈련-없는 기반 (Training-free) ← PAI가 속하는 범주
    ├── 디코딩 방법 개선
    │   ├── OPERA (Huang et al., 2023): 비정상 어텐션 패턴 감지
    │   └── VCD (Leng et al., 2023): 시각적 불확실성 대비 디코딩
    └── 추론 개입 (Inference Intervention)
        └── PAI (Liu et al., 2024) ← 본 논문 (최초 제안)
```

### 4.2 주요 관련 연구 상세 비교

| 방법 | 연도 | 접근법 | 외부 도구 | 추가 학습 | 시간 효율 | 범용성 |
|------|------|--------|----------|----------|----------|--------|
| **LURE** | 2023 | 후처리 + 수정 모델 | ✅ | ✅ | 낮음 | 제한적 |
| **Woodpecker** | 2023 | 외부 시각 모델 | ✅ | ❌ | 낮음 | 제한적 |
| **OPERA** | 2023 | Beam Search 개선 | ❌ | ❌ | 보통 | Beam Search 한정 |
| **VCD** | 2023 | Nucleus Sampling 개선 | ❌ | ❌ | 보통 | Nucleus 한정 |
| **ITI** (Li et al., 2024) | 2024 | 은닉 상태 개입 (LLM) | ❌ | 일부 필요 | 보통 | LLM 중심 |
| **PAI (본 논문)** | 2024 | 어텐션 개입 + 로짓 정제 | ❌ | ❌ | **Vanilla와 동등** | **모든 디코딩** |

### 4.3 핵심 관련 연구 심층 비교

#### OPERA (Huang et al., 2023, NeurIPS)
- **핵심 아이디어**: 멀티모달 LLM 디코딩에서 특정 토큰에 과도한 어텐션이 집중되는 "Over-trust" 패턴을 감지하고 페널티 적용
- **차이점**: Beam Search에 특화된 방법으로, CHAIR에서 LLaVA 기준 44.6/14.4로 PAI(21.8/5.6)에 비해 성능이 낮고 시간 효율도 낮음
- **PAI 대비**: PAI는 모든 디코딩 방법에 적용 가능하며 시간 효율이 Vanilla와 동등

#### VCD (Leng et al., 2023)
- **핵심 아이디어**: 시각적 불확실성(Visual Uncertainty)이 증폭된 이미지를 대비(Contrastive)하여 환각 감소
- **차이점**: Nucleus Sampling 특화, 장문 생성에서 오히려 환각이 증가하는 경우 있음
- **PAI 대비**: PAI는 이미지 토큰에 직접 개입하여 근본 원인을 해결

#### ITI - Inference-Time Intervention (Li et al., 2024, NeurIPS)
- **핵심 아이디어**: LLM에서 신뢰 가능한 방향을 프로브(Probe)로 학습하여 은닉 상태에 개입
- **차이점**: LLM의 사실성(Truthfulness) 향상에 초점, 멀티모달 이미지 정보 활용에는 미흡
- **PAI 대비**: PAI는 별도의 프로브 학습 없이 원래 어텐션 방향을 신뢰 방향으로 활용

#### LLaVA-RLHF / Factually Augmented RLHF (Sun et al., 2023)
- **핵심 아이디어**: 사실적으로 강화된 RLHF로 정렬 학습
- **차이점**: 고품질 데이터와 대규모 컴퓨팅 자원 필요
- **PAI 대비**: PAI는 추가 학습 불필요, 기존 모델에 즉시 적용 가능

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 앞으로의 연구에 미치는 영향

#### ① 추론 개입(Inference Intervention) 패러다임의 확립

PAI는 LVLMs에서 **추론 개입 방식으로 환각을 완화한 최초의 연구**로, 새로운 연구 방향을 개척했습니다. 이는:
- 고비용 재학습 없이 기존 모델의 성능을 향상시킬 수 있다는 가능성 제시
- 어텐션 메커니즘의 직접 조작을 통한 멀티모달 이해 향상이라는 새로운 연구 경로 열기
- 다양한 도메인(의료 영상 분석, 자율주행 등)에서의 Training-free 환각 완화 연구 촉진

#### ② 텍스트 관성(Text Inertia) 개념의 파급 효과

텍스트 관성이라는 현상의 정의와 실증적 분석은 LVLMs 연구 커뮤니티에 중요한 통찰을 제공합니다:
- 환각의 근본 원인 분석 연구를 심화시킬 기반 제공
- 훈련 과정에서 텍스트 관성을 손실 함수로 통합하는 연구 가능성
- 언어 모델과 시각 모델의 균형 잡힌 융합을 위한 아키텍처 설계 연구 자극

#### ③ 멀티모달 모델의 설명가능성(Explainability) 향상

PAI 적용 후 셀프-어텐션 맵이 이미지의 해당 영역과 더 잘 정렬되는 현상(Fig. S1)은:
- LVLMs의 내부 표현 해석 연구에 기여
- Attention 기반 설명가능성(Explainability) 연구의 새로운 방향 제시
- 시각-언어 정렬 상태를 정량적으로 평가하는 새로운 메트릭 개발 가능성

#### ④ 대비 디코딩(Contrastive Decoding) 패러다임 확장

식 (4)의 Image-Centric Logit Refine은 LLM-CFG와 유사하지만, **시각-언어 멀티모달 맥락에서의 대비 디코딩**을 구체화했습니다. 이는:
- 텍스트-전용 모델 대비 멀티모달 모델의 추가적 이점 극대화 연구
- 다양한 모달리티(오디오, 비디오 등)로의 확장 가능성

---

### 5.2 앞으로 연구 시 고려할 점

#### ① LLaMA 외 아키텍처에서의 검증 필요

현재 PAI는 LLaMA 계열 디코더를 사용하는 모델에서만 검증되었습니다. GPT-계열, Mistral, Phi 등 다양한 아키텍처에서도:
- 텍스트 관성 현상이 동일하게 발생하는지 확인
- 어텐션 증폭 방식이 동일하게 효과적인지 검증
- 아키텍처별 최적 하이퍼파라미터 탐색 자동화 필요

#### ② 훈련 과정과의 통합 가능성 탐구

논문에서도 언급했듯이, 추론 시 개입의 상한선은 사전 훈련된 모델 능력에 의존합니다:

$$\text{PAI 성능 상한} \approx f(\text{모델의 잠재적 이미지 이해 능력})$$

따라서:
- 텍스트 관성을 훈련 손실로 포함하는 방법 연구
- Self-supervised 방식으로 이미지 어텐션을 강화하는 사전 훈련 전략
- PAI를 데이터 증강(Data Augmentation)으로 활용하는 방안

#### ③ 하이퍼파라미터 자동 최적화

현재 $\alpha$와 $\gamma$는 모델별로 수동 설정이 필요합니다:
- 이미지 토큰 길이와 원래 어텐션 분포를 기반으로 $\alpha$를 자동 계산하는 방법
- 텍스트 관성의 정도를 실시간으로 측정하여 $\gamma$를 동적으로 조정하는 방법
- 메타 학습(Meta-learning)을 통한 적응적 하이퍼파라미터 설정

#### ④ 복잡한 추론 태스크로의 확장

PAI는 공간적 관계, 비교, 논리적 추론이 필요한 질문에서 제한적인 효과를 보였습니다:
- 이미지의 국소적(Local) 정보와 전역적(Global) 정보를 구분하여 개입하는 방법
- 다단계 추론(Multi-step Reasoning)에서의 이미지 어텐션 분배 전략
- Chain-of-Thought와 PAI를 결합한 시각적 추론 향상 방법

#### ⑤ Nucleus 샘플링과의 호환성 개선

Nucleus 샘플링 사용 시 효과가 제한적입니다:
- 샘플링 풀 자체에서 환각 토큰을 필터링하는 추가 메커니즘 필요
- 어텐션 증폭과 토큰 확률 재분배를 더 긴밀하게 연동하는 방법
- 온도(Temperature) 조절과 PAI의 상호작용 연구

#### ⑥ 비디오, 오디오 등 다중 모달리티로의 확장

시각-언어 모델에서 검증된 PAI의 개념을:
- 비디오-언어 모델(Video-LLM)에서 시간적 어텐션 조절로 확장
- 오디오-언어 모델에서 청각 토큰 어텐션 증폭으로 확장
- 다중 이미지 입력(Multi-image) 시나리오에서의 효과 검증

#### ⑦ 공정한 평가 프레임워크 구축

현재 환각 평가 메트릭들은 각각의 한계를 가지고 있습니다:
- CHAIR: 객체 수준 환각만 측정
- POPE: Yes/No 이진 응답으로 제한
- MMHal-Bench: 소규모 데이터셋 (96개 쌍)

더 포괄적이고 세밀한 환각 평가 방법론 개발이 필요합니다.

---

## 참고 자료 (본 논문 및 비교 논문)

1. **Liu, S., Zheng, K., & Chen, W. (2024)**. "Paying More Attention to Image: A Training-Free Method for Alleviating Hallucination in LVLMs." *arXiv:2407.21771v1* [cs.CV].

2. **Huang, Q. et al. (2023)**. "OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation." *arXiv:2311.17911* [cs.CV].

3. **Leng, S. et al. (2023)**. "Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding." *arXiv:2311.16922* [cs.CV].

4. **Li, K. et al. (2024)**. "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." *Advances in Neural Information Processing Systems 36 (NeurIPS 2024)*.

5. **Sun, Z. et al. (2023)**. "Aligning Large Multimodal Models with Factually Augmented RLHF." *arXiv:2309.14525* [cs.CV].

6. **Zhou, Y. et al. (2023)**. "Analyzing and Mitigating Object Hallucination in Large Vision-Language Models (LURE)." *arXiv:2310.00754* [cs.CV].

7. **Yin, S. et al. (2023)**. "Woodpecker: Hallucination Correction for Multimodal Large Language Models." *arXiv:2310.16045* [cs.CV].

8. **Li, Y. et al. (2023)**. "Evaluating Object Hallucination in Large Vision-Language Models (POPE)." *arXiv:2305.10355* [cs.CV].

9. **Liu, H. et al. (2023)**. "Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)." *arXiv:2310.03744* [cs.CV].

10. **Sanchez, G. et al. (2023)**. "Stay on Topic with Classifier-Free Guidance (LLM-CFG)." *arXiv:2306.17806* [cs.CL].

11. **Xiao, G. et al. (2023)**. "Efficient Streaming Language Models with Attention Sinks (StreamLLM)." *arXiv:2309.17453* [cs.CL].

12. **Rohrbach, A. et al. (2018)**. "Object Hallucination in Image Captioning (CHAIR)." *EMNLP 2018*.

13. **Zhang, Y. et al. (2023)**. "Prompt Highlighter: Interactive Control for Multi-Modal LLMs." *arXiv:2312.04302* [cs.CV].
