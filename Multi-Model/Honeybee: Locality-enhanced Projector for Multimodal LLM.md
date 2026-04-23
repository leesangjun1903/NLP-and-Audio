# Honeybee: Locality-enhanced Projector for Multimodal LLM

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

**Honeybee** 논문의 핵심 주장은 다음과 같습니다:

> *"Multimodal LLM(MLLM)에서 비주얼 프로젝터(visual projector)는 성능과 효율성 모두에 결정적인 역할을 하며, 기존 프로젝터들은 유연성(flexibility)과 지역성 보존(locality preservation)이라는 두 가지 핵심 속성을 동시에 만족하지 못한다. 우리는 이 두 속성을 동시에 달성하는 새로운 locality-enhanced projector를 제안한다."*

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **① Locality-enhanced Projector 설계** | C-Abstractor(합성곱 기반), D-Abstractor(변형 어텐션 기반)를 제안하여 유연성과 지역성 보존을 동시 달성 |
| **② 멀티페이셋 데이터 활용 레시피 제공** | 데이터 조합, 밸런싱, 템플릿 세분성, 멀티턴 전략 등에 대한 체계적인 실험적 분석 제공 |
| **③ SOTA 달성** | MME, MMBench, SEED-Bench, LLaVA-Bench 등 다양한 벤치마크에서 이전 SOTA 대비 큰 폭의 성능 향상 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

#### 기존 프로젝터의 한계

**선형 프로젝터(Linear Projector)**
- ✅ 지역성 보존 우수 (1:1 매핑으로 모든 로컬 컨텍스트 보존)
- ❌ 유연성 부재: 반드시 $M = N$ (입력 토큰 수 = 출력 토큰 수)

**추상화 프로젝터(Abstractor, e.g., Resampler, Q-Former)**
- ✅ 유연성 우수: $M < N$ 가능하여 효율적
- ❌ 지역성 보존 부족: 소수의 두드러진 영역(salient region)에만 집중, 세부 정보 손실

논문에서는 Resampler와 Linear projector의 공간 이해 능력 비교에서:

$$\text{Avg}^N_{\text{Resampler}} = 45.6, \quad \text{Avg}^N_{\text{Linear}} = 52.6$$

(6개 spatial understanding 태스크 기준, Table 3)

이처럼 Resampler가 공간 이해에서 현저히 낮은 성능을 보임을 실험적으로 확인하였습니다.

#### 핵심 문제 정의

> MLLM의 효율성은 비전 인코더나 프로젝터의 연산량이 아닌, **LLM에 입력되는 비주얼 토큰의 수** $M$ 에 의해 결정된다.

따라서 프로젝터 설계의 두 가지 필수 속성:
1. **유연성(Flexibility)**: $M$의 자유로운 조절 가능
2. **지역성 보존(Locality Preservation)**: 시각적 특징의 공간적 맥락 보존

---

### 2-2. 제안하는 방법 (수식 포함)

#### MLLM의 기본 수식

멀티모달 입력 $\mathbf{X}\_{\texttt{img}}$, $\mathbf{X}\_{\texttt{text}}$ 가 주어졌을 때, 응답 $\mathbf{Y} = \{w_i\}_{i=1}^{L}$의 생성 확률:

$$p(\mathbf{Y}|\mathbf{X}_{\texttt{img}}, \mathbf{X}_{\texttt{text}}) = \prod_{i=1}^{L} p(w_i|\mathbf{X}_{\texttt{img}}, \mathbf{X}_{\texttt{text}}, w_{<i}) \tag{1}$$

#### C-Abstractor (Convolutional Abstractor)

**구조**: $L$개의 ResNet 블록 → Adaptive Average Pooling → $L$개의 ResNet 블록

```
Visual Features
     ↓
[ResBlock × L]
     ↓
[Adaptive AvgPool]  ← 여기서 M 결정 (임의의 제곱수 가능)
     ↓
[ResBlock × L]
     ↓
Visual Tokens (M개)
```

- 합성곱(convolution)을 통해 이웃한 특징들의 지역적 맥락을 통합
- Adaptive Average Pooling으로 임의의 $M$ 크기 출력 가능 (유연성)
- ResNet bottleneck block + Squeeze-Excitation 적용 시 최고 성능

#### D-Abstractor (Deformable attention-based Abstractor)

변형 어텐션(deformable attention)을 사용하며, 핵심 수식은 다음과 같습니다:

$$\mathbf{z}^{l+1} = \sum_{k=1}^{K} A_k^l \cdot X_{feat}(p + \Delta o_k^l) \tag{2}$$

여기서:
- $\mathbf{z}^{l+1}$: $l+1$ 번째 레이어의 학습 가능한 쿼리
- $K$: 참조점당 샘플링 오프셋의 수
- $A_k^l$: $l$ 번째 레이어에서 $k$ 번째 샘플링에 대한 어텐션 가중치
- $p$: 2D 참조점(reference point)
- $\Delta o_k^l$: 학습 가능한 샘플링 오프셋
- $X_{feat}$: 비주얼 특징 맵

**추가 기법 (D-Abstractor 성능 향상)**:
1. **v-pooled Q**: 랜덤 초기화 대신 visual feature map의 adaptive average pooling으로 learnable query 초기화
2. **M-RP (Manual Reference Points)**: 참조점을 특징 맵 전체에 균등하게 분포시켜 초기화 (centralized 초기화 대신)

---

### 2-3. 모델 구조

```
이미지 입력
    ↓
[Vision Encoder: CLIP ViT-L/14]
    ↓ (N개의 visual feature)
[Projector: C-Abstractor 또는 D-Abstractor]
    ↓ (M개의 visual token, M은 유연하게 설정)
[Large Language Model: Vicuna-v1.5 7B/13B]
    ↓
텍스트 응답 출력
```

#### 학습 파이프라인 (2단계)

**Stage 1: Pre-training (Vision-Language Alignment)**
- 프로젝터만 학습, 비전 인코더와 LLM은 동결(frozen)
- 데이터: BlipCapFilt, COYO100M (캡셔닝 데이터, ~200M 샘플)
- 목적: 비주얼 특징과 텍스트 간의 정렬 학습

**Stage 2: Visual Instruction Tuning**
- 프로젝터 + LLM 전체 학습 (full fine-tuning)
- 데이터: VQA, 다중선택 VQA, REC, 캡셔닝, LLaVA150K, ShareGPT 등 다양한 태스크

#### 학습 하이퍼파라미터 (최종 모델 기준)

| 설정 | Pre-training | Instruction Tuning |
|------|--------------|---------------------|
| 학습 가능 모듈 | Abstractor | Abstractor + LLM |
| Batch size | 256 | 128 |
| Learning rate | 3e-4 | 2e-5 |
| Training steps | 200k | 10k |
| Optimizer | AdamW ($\beta_1=0.9, \beta_2=0.98$) | 동일 |

---

### 2-4. 성능 향상

#### 프로젝터별 공간 이해 성능 비교 (Table 3)

| Projector | $M$ | s/step | $\text{Avg}^N$ (6 spatial tasks) |
|-----------|-----|--------|----------------------------------|
| Linear | 256 | 3.04 | 52.6 |
| Resampler | 144 | 2.28 | 43.9 |
| Resampler | 256 | 3.12 | 45.6 |
| **C-Abstractor** | **144** | **2.23** | **53.5** |
| **C-Abstractor** | **256** | **3.07** | **56.3** |

→ C-Abstractor (M=144)는 Linear (M=256)보다 적은 연산으로 더 높은 공간 이해 성능 달성

#### SOTA 대비 성능 비교 (Table 1, 6)

| 벤치마크 | 이전 SOTA | Honeybee | 향상폭 |
|---------|-----------|----------|--------|
| MMBench | 67.7 | **73.6** | +5.9 |
| SEED-Bench | 68.1 | **68.6** | +0.5 |
| MME (Perception) | 1531 | **1661** | +130 |
| MME (Total) | 1848 | **1977** | +129 |
| LLaVA-Bench | 70.7 | **77.5** | +6.8 |

---

### 2-5. 한계

논문에서 직접적으로 명시하거나 실험을 통해 확인된 한계:

1. **고해상도 이미지에 대한 의존성**: SEED-Bench처럼 세밀한 시각 이해가 필요한 벤치마크에서는 더 큰 이미지나 더 많은 비주얼 토큰이 필요함 (7B 모델 기준 SEEDI에서 경쟁 모델 대비 열위)

2. **객체 환각(Hallucination)**: POPE 벤치마크에서 7B 규모 Honeybee는 LLaVA-1.5 대비 약간 낮은 성능 (83.2 vs 85.9, Table 16)

3. **LoRA 비효율**: LoRA(r=64, r=256) 적용 시 full tuning 대비 성능이 크게 하락 (AvgN 44.9~48.4 vs 70.6), 더 효율적인 PEFT 방법 탐색 필요

4. **수동 데이터 밸런싱 의존**: 최적 성능을 위해 수동으로 튜닝된 샘플링 비율이 필요 (자동화 미흡)

5. **멀티모달 확장 미탐**: 비디오, 오디오, 3D 등 이미지 외 모달리티에 대한 검증 부재

6. **C-Abstractor 구조 최적화 미탐**: 논문에서 "further architectural variations are explorable under the proposed design principles, we leave them for future investigation"이라고 명시

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 다양한 태스크 다양성이 일반화에 미치는 영향

논문의 Table 4 ablation study는 **태스크 다양성**이 일반화에 핵심적임을 보여줍니다:

$$\text{D1 (모든 태스크)} \gg \text{D10 (Instruction 데이터만)}$$

- D10 (instruction-following 데이터만 사용): MMB 43.7, SEED 0.0(!) → 심각한 성능 저하
- D1 (모든 태스크 포함): MMB 69.2, SEEDI 64.2

각 태스크의 기여:
- **VQA(Open)** → MME 성능에 기여
- **VQA(MC)** → MMBench, SEEDI 성능에 기여
- **캡셔닝 + Instruction-following** → LLaVA-Bench 성능에 기여

### 3-2. 데이터셋 다양성의 역할

단일 데이터셋 사용(D2) vs 다양한 데이터셋 조합(D1):

$$\text{MMB}: 67.4 \text{ (D2)} \rightarrow 69.2 \text{ (D1)}, \quad \text{MMEP}: 1454 \rightarrow 1568$$

동일 태스크 내에서도 여러 데이터셋을 혼합하는 것이 일반화에 유리함을 보여줍니다.

### 3-3. 지역성 보존과 일반화의 연관성

C-Abstractor의 locality preservation은 단순히 공간 이해 태스크뿐만 아니라, **다양한 도메인에서의 세밀한 시각 이해**를 가능하게 합니다:

- 동일한 비주얼 토큰 수 $M=144$에서 C-Abstractor가 Linear(M=256)보다 더 높은 Avg $^N$ 달성
- 이는 더 적은 토큰으로도 더 풍부한 시각 정보를 LLM에 전달함을 의미

$$\text{Avg}^N: \underbrace{53.5}_{\text{C-Abstractor, M=144}} > \underbrace{52.6}_{\text{Linear, M=256}}$$

### 3-4. Instruction Tuning vs Multi-task Learning

| 방식 | 식별자 | MMB | MMEP | AvgN |
|------|--------|-----|------|------|
| **Instruction Tuning** | instruction | **69.2** | **1568.2** | **70.6** |
| Multi-task | dataset name | 66.8 | 1483.1 | 68.4 |
| Multi-task | task name | 68.4 | 1507.5 | 69.3 |

Instruction Tuning이 zero-shot 일반화에 더 유리함을 실증적으로 확인하였으며, 이는 GPT-style 학습 패러다임의 효과를 재확인합니다.

### 3-5. Fine-grained 템플릿의 일반화 효과

| 템플릿 세분성 | 다양성 | MMB | MMEP | AvgN |
|-------------|--------|-----|------|------|
| **Fine-grained** | Single | **69.2** | **1568.2** | **70.6** |
| Coarse-grained | Single | 68.9 | 1553.8 | 70.2 |
| Fine-grained | Multi | 68.1 | 1581.2 | 70.5 |
| Fine-grained | Multi+flip | 67.4 | 1575.9 | 69.8 |

흥미롭게도 **템플릿 다양성(multi)이 성능을 보장하지 않음**을 보여주며, GPT-assisted 데이터가 있는 환경에서는 fine-grained single template이 최적임을 확인하였습니다.

### 3-6. Science QA에서의 Generalist 일반화

Honeybee (C-13B, M=576)는 태스크 특화 fine-tuning 없이도 Science QA에서 94.39%를 달성, LLaVA+GPT-4(judge)(92.53%)를 능가:

$$\text{Honeybee (M=576)}: 94.39 > \text{LLaVA+GPT-4 (judge)}: 92.53$$

이는 locality-enhanced projector + 다양한 instruction tuning 전략이 도메인 특화 미세조정 없이도 강한 일반화를 달성할 수 있음을 보여줍니다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4-1. 주요 MLLM 연구 계보

```
Flamingo (2022, NeurIPS)
    ↓
BLIP-2 (2023, ICML)
    ↓
LLaVA / InstructBLIP (2023)
    ↓
LLaVA-1.5 / Qwen-VL (2023)
    ↓
Honeybee (2023/2024, arXiv)
```

### 4-2. 프로젝터 설계 관점 비교

| 모델 | 프로젝터 | 유연성 | 지역성 보존 | 비고 |
|------|---------|--------|------------|------|
| **Flamingo** (Alayrac et al., NeurIPS 2022) | Perceiver Resampler | ✅ | ❌ | 최초의 대규모 MLLM |
| **BLIP-2** (Li et al., ICML 2023) | Q-Former | ✅ | ❌ | Frozen LLM 활용 |
| **LLaVA** (Liu et al., NeurIPS 2023) | Linear | ❌ | ✅ | 단순하지만 효과적 |
| **LLaVA-1.5** (Liu et al., 2023) | 2-layer MLP | ❌ (M=576 고정) | ✅ | MLP로 약간 개선 |
| **InstructBLIP** (Dai et al., 2023) | Q-Former | ✅ | ❌ | 태스크별 instruction |
| **mPLUG-Owl** (Ye et al., 2023) | Resampler | ✅ | ❌ | 모듈화 설계 |
| **Qwen-VL** (Bai et al., 2023) | Resampler | ✅ | ❌ | 더 큰 ViT-bigG 사용 |
| **Honeybee** (Cha et al., 2024) | C/D-Abstractor | ✅ | ✅ | **두 속성 동시 달성** |

### 4-3. 벤치마크 성능 비교 (7B 스케일)

| 모델 | MMB | MMEP | SEEDI | LLaVAW |
|------|-----|------|-------|--------|
| LLaVA (v1) | 38.7 | 502.8 | 33.5 | - |
| MiniGPT-4 | 24.3 | 581.7 | 47.4 | - |
| InstructBLIP | 36.0 | - | 58.8 | 60.9 |
| Qwen-VL-Chat | 60.6 | 1487.5 | 65.4 | - |
| LLaVA-1.5 | 64.3 | 1510.7 | - | 63.4 |
| **Honeybee (M=144)** | **70.1** | **1584.2** | **64.5** | **67.1** |

### 4-4. 효율성 비교

Honeybee의 특징적 우위: **더 적은 비주얼 토큰으로 더 높은 성능**

$$\underbrace{M=144}_{\text{Honeybee}} \text{ vs } \underbrace{M=576}_{\text{LLaVA-1.5 (13B)}}$$

동일한 시간 대비 성능(AvgN):

$$\text{Honeybee (7B, M=144, 2.23 s/step)}: \text{Avg}^N = 70.6$$
$$\text{Linear (7B, M=256, 3.04 s/step)}: \text{Avg}^N = 70.0$$

### 4-5. Honeybee 이후의 연구 동향 (논문 출판 이후 트렌드)

> **⚠️ 주의**: 다음은 Honeybee 논문 자체에 명시된 내용이 아니라, 제가 알고 있는 2024년 이후 연구 동향입니다. 논문 외부 정보이므로 일부 불확실성이 있음을 밝힙니다.

Honeybee의 locality-aware projector 개념은 이후 연구들에 영향을 미쳤으며:
- **고해상도 처리**: LLaVA-Next, InternVL 등에서 타일링 기반 고해상도 처리로 발전
- **동적 토큰 수**: 이미지 복잡도에 따라 M을 동적으로 조절하는 연구들 등장
- **Projector 재조명**: 프로젝터 설계가 MLLM 연구의 중요 주제로 부상

---

## 5. 연구에 미치는 영향과 향후 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

#### ① 프로젝터 설계의 체계적 분류 프레임워크 제공

Honeybee는 프로젝터를 **유연성(Flexibility)**과 **지역성 보존(Locality Preservation)**이라는 두 축으로 분류하는 체계를 제공했습니다. 이 프레임워크는 향후 새로운 프로젝터 설계 시 명확한 기준점이 됩니다.

$$\text{좋은 프로젝터} = f(\text{Flexibility}, \text{Locality Preservation})$$

#### ② 공간 이해(Spatial Understanding)의 중요성 부각

추상화 프로젝터가 공간 이해에서 취약하다는 것을 정량적으로 증명함으로써, 향후 MLLM 평가 시 공간 이해 능력을 반드시 고려해야 함을 시사합니다.

#### ③ Instruction Tuning 레시피의 체계화

데이터 조합, 밸런싱, 템플릿 설계에 대한 5가지 연구 질문을 체계적으로 분석한 것은 향후 MLLM 학습 설계의 기준이 될 수 있습니다.

#### ④ 효율성-성능 트레이드오프의 재정의

단순히 더 많은 토큰이 좋은 것이 아니라, **locality-aware 설계를 통해 더 적은 토큰으로도 더 좋은 성능**을 달성할 수 있음을 보여준 것은 효율적 MLLM 설계의 새로운 방향을 제시합니다.

### 5-2. 향후 연구 시 고려할 점

#### ① 동적 토큰 할당 (Dynamic Token Allocation)

현재 Honeybee는 $M$을 고정값으로 설정합니다. 이미지의 복잡도나 내용에 따라 $M$을 동적으로 결정하는 메커니즘이 필요합니다:

$$M^* = g(\text{image complexity}, \text{task type}, \text{budget})$$

#### ② 고해상도 이미지 처리

SEED-Bench에서 드러났듯이, 세밀한 시각 이해를 위해서는 더 높은 해상도가 필요합니다. C-Abstractor의 adaptive pooling 구조를 활용한 **타일링(tiling) 또는 다중 스케일** 처리 방식 연구가 유망합니다.

#### ③ 객체 환각(Hallucination) 감소

POPE 벤치마크에서의 상대적 약점을 고려할 때, locality preservation이 오히려 환각을 증가시킬 가능성도 있습니다. 지역 정보 보존과 환각 사이의 관계에 대한 심층 분석이 필요합니다.

#### ④ 다양한 모달리티로의 확장

현재 Honeybee는 이미지만 다루지만, locality-enhanced projector 개념은 비디오(시간적 지역성), 3D 포인트 클라우드(공간적 지역성), 오디오(주파수 지역성) 등에도 적용 가능합니다:

$$\mathbf{z}_{\text{video}}^{l+1} = \sum_{k=1}^{K} A_k^l \cdot X_{feat}(p_t + \Delta o_k^l)$$

(여기서 $p_t$는 시공간 참조점)

#### ⑤ 자동화된 데이터 밸런싱

현재 수동(hand-crafted) 샘플링 비율 조정이 최적이나, 이는 확장성이 떨어집니다. AutoML이나 메타러닝 기반의 자동 데이터 밸런싱 전략 연구가 필요합니다.

#### ⑥ 더 강력한 비전 인코더와의 결합

Honeybee는 CLIP ViT-L/14를 사용하는 반면, Qwen-VL은 ViT-bigG를 사용합니다. C-Abstractor를 더 강력한 비전 인코더(e.g., DINOv2, SigLIP)와 결합했을 때의 효과 분석이 필요합니다.

#### ⑦ PEFT와의 호환성 개선

LoRA 적용 시 성능이 크게 하락하는 문제(AvgN: 70.6 → 48.4)는 실용적 활용에 장애가 됩니다. Locality-aware projector와 함께 효율적으로 동작하는 PEFT 방법 개발이 필요합니다.

#### ⑧ 멀티턴 대화에서의 맥락 유지

현재 De-duplication 전략은 단일 이미지 내 중복 제거에 초점을 맞추지만, 여러 이미지가 포함된 멀티턴 대화에서의 시각 맥락 일관성 유지 방법도 중요한 연구 방향입니다.

---

## 참고 자료

### 논문 원문 (주요 참고 자료)

1. **Honeybee 논문 (주 참고 자료)**
   - Cha, J., Kang, W., Mun, J., & Roh, B. (2024). *Honeybee: Locality-enhanced Projector for Multimodal LLM*. arXiv:2312.06742v2. https://arxiv.org/abs/2312.06742

### 논문에서 인용된 핵심 참고 문헌

2. **Flamingo**: Alayrac, J.-B., et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning*. NeurIPS 2022.

3. **BLIP-2**: Li, J., et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. ICML 2023.

4. **LLaVA**: Liu, H., et al. (2023). *Visual Instruction Tuning*. NeurIPS 2023.

5. **LLaVA-1.5**: Liu, H., et al. (2023). *Improved Baselines with Visual Instruction Tuning*. arXiv:2310.03744.

6. **InstructBLIP**: Dai, W., et al. (2023). *InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning*. arXiv:2305.06500.

7. **Qwen-VL**: Bai, J., et al. (2023). *Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond*. arXiv:2308.12966.

8. **Deformable DETR**: Zhu, X., et al. (2021). *Deformable DETR: Deformable Transformers for End-to-End Object Detection*. ICLR 2021.

9. **CLIP**: Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021.

10. **MMBench**: Liu, Y., et al. (2023). *MMBench: Is Your Multi-modal Model an All-around Player?* arXiv:2307.06281.

11. **SEED-Bench**: Li, B., et al. (2023). *SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension*. arXiv:2307.16125.

12. **MME**: Fu, C., et al. (2023). *MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models*. arXiv:2306.13394.

13. **ScienceQA**: Lu, P., et al. (2022). *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*. NeurIPS 2022.

14. **Vicuna**: Chiang, W.-L., et al. (2023). *Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality.

15. **GitHub (Honeybee 코드)**: https://github.com/kakaobrain/honeybee
