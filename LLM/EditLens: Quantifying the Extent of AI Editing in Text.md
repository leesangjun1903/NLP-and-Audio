# EditLens: Quantifying the Extent of AI Editing in Text

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

**EditLens**는 AI가 텍스트를 *얼마나* 편집했는지를 연속적인 스코어(0~1)로 정량화하는 **최초의 AI 편집 감지 회귀 모델**이다. 기존 연구가 "완전한 인간 vs. 완전한 AI 생성" 이진 분류에 집중한 반면, 실제 LLM 사용의 약 2/3는 사용자 텍스트를 *편집*하는 방식(Chatterji et al., 2025)임을 지적하며, AI 편집 텍스트가 인간 작성 및 완전 AI 생성 텍스트와 **구별 가능함**을 실증적으로 보인다.

### 주요 기여 (5가지)

| # | 기여 내용 |
|---|-----------|
| 1 | 전체 AI 편집 분류체계(taxonomy)를 포괄하는 대규모 **동질적 혼합 텍스트(Homogeneous Mixed Text) 데이터셋** 구축 |
| 2 | **경량 유사도 메트릭**(코사인 거리, Soft N-Grams)으로 AI 편집량 정량화 및 전문 인간 주석자와의 일치도 검증 |
| 3 | 편집된 텍스트만으로 AI 편집 정도를 예측하는 **단일 입력 회귀 헤드** 파인튜닝 |
| 4 | 이진 분류 F1=94.7%, 3분류 F1=90.4%로 **SOTA 달성** (기존 최고 이진 분류 대비 +8%, 3분류 대비 +16%) |
| 5 | 다중 편집, AI→AI 편집, 인간→AI 편집(BEEMO), Grammarly 실제 사용 분석 등 **일반화 사례 연구** 제공 |

---

## 2. 해결 문제 · 제안 방법 · 모델 구조 · 성능 · 한계

### 2.1 해결하고자 하는 문제

**기존 이진 분류 AI 감지기의 근본적 한계:**

- Saha & Feizi (2025)에 따르면 기존 이진 감지기는 AI로 약간 다듬어진 텍스트(AI-polished)를 완전 AI 생성으로 오분류하는 **높은 거짓 양성(False Positive)** 문제 발생
- "가벼운 교정은 허용, 완전 AI 생성은 금지"와 같은 **세분화된 정책 적용 불가**
- 현대적 협업 편집(co-writing)에서 발생하는 **동질적 혼합 저작권(Homogeneous Mixed Authorship)** 문제를 기존 방법론이 다루지 못함

**동질적(Homogeneous) vs. 이질적(Heterogeneous) 혼합 저작:**

| 구분 | 이질적(Heterogeneous) | 동질적(Homogeneous) |
|------|----------------------|---------------------|
| 정의 | 인간/AI가 명확히 분리된 구간 저작 | AI가 인간 텍스트 전체를 편집 |
| 토큰 레이블 | 각 토큰에 Human/AI 레이블 부여 가능 | 저작권이 편집 과정에 **얽혀 있어** 토큰 레이블 불가 |
| 선행 연구 | RoFT, SeqXGPT, PaLD, HaCo-Det | **본 논문이 집중적으로 다룸** |

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 편집 연산자 모델링

편집된 텍스트 $y$를 원본 인간 텍스트 $x$에 편집 연산자 $\mathcal{E}_\lambda$를 적용한 결과로 모델링:

$$y = \mathcal{E}_\lambda(x; z), \quad z \sim p(z), \quad \lambda \in \Lambda$$

여기서 $z$는 삽입·삭제·대체·재배열 등의 마이크로 편집 시퀀스(잠재 변수), $\lambda$는 편집 강도를 요약한 파라미터이다.

#### (2) 변화 크기 함수(Change Magnitude Functional)

유사도 함수 $\text{sim} : \mathcal{X} \times \mathcal{X} \to [0,1]$에 단조 변환을 적용하여 변화 크기를 정의:

$$\Delta(x, y) = g\bigl(\text{sim}(x, y)\bigr), \quad \text{e.g.,} \quad g(s) = 1 - s$$

- $\Delta(x,y) = 0$: 동일 텍스트 (편집 없음)
- $\Delta(x,y) \to 1$: 강한 편집 적용

#### (3) 스케일된 유사도 점수 (회귀 목표값)

임계값 $\tau_\text{low}$, $\tau_\text{high}$를 기준으로 유사도 점수를 정규화:

$$\tilde{s} = \begin{cases} 0.0 & \text{if } s \leq \tau_\text{low} \\ 1.0 & \text{if } s \geq \tau_\text{high} \\ \dfrac{s - \tau_\text{low}}{\tau_\text{high} - \tau_\text{low}} & \text{otherwise} \end{cases}$$

- 코사인 거리 기준: $\tau_\text{low} = 0.03$, $\tau_\text{high} = 0.15$
- Soft N-Grams 기준: $\tau_\text{low} = 0.06$, $\tau_\text{high} = 0.72$

#### (4) 단일 입력 예측기 (추론 시 원본 불필요)

추론 시 편집된 텍스트 $y$만으로 변화 크기를 예측:

$$f^\text{ssi}_\theta : \mathcal{X} \to [0,1], \quad \hat{\Delta}(y) = f^\text{ssi}_\theta(y)$$

학습은 $(x^{(i)}, y^{(i)})$ 쌍으로 목표값 $\Delta^{(i)} = \Delta(x^{(i)}, y^{(i)})$를 계산하되, **추론 시에는 $x$를 사용하지 않음**:

$$\min_\theta \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}\!\left(f^\text{ssi}_\theta\!\left(y^{(i)}\right),\, \Delta\!\left(x^{(i)}, y^{(i)}\right)\right)$$

베이즈 최적 예측기(Bayes-optimal predictor)는 조건부 기대값:

$$f^*(y) = \mathbb{E}[\Delta(X, y) \mid Y = y]$$

그러나 EditLens는 $x$를 재구성하지 않고 $y$만으로 판별적으로 학습하여 이를 근사한다.

#### (5) 회귀 손실 (MSE)

$$\mathcal{L}_\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\tilde{s}_i - \hat{s}_i)^2$$

#### (6) 분류 접근법: 버킷 할당 함수

유사도 점수를 $N$개 버킷으로 이산화 ($N \in \{4, 5, 6\}$):

$$b(s) = \min\!\left(N-1,\; \left\lfloor \frac{s - \tau_\text{min}}{\tau_\text{max} - \tau_\text{min}} \cdot N \right\rfloor\right)$$

버킷 $j$의 중점:

$$m_j = \tau_\text{min} + \frac{(j+0.5) \cdot (\tau_\text{max} - \tau_\text{min})}{N}$$

분류 손실 (Cross-Entropy):

$$\mathcal{L}_\text{CE} = -\frac{1}{n} \sum_{i=1}^{n} \log p\!\left(b(s_i) \mid x_i\right)$$

#### (7) 가중 평균 디코딩 (Weighted-Average Decoding)

추론 시 최종 점수 계산:

$$\hat{s} = \sum_{j=0}^{N-1} p(j \mid x) \cdot m_j$$

이를 통해 전통적인 argmax 대신 **확률 분포를 활용한 세밀한 점수 출력**이 가능하다.

#### (8) 중간 감독 메트릭 (Intermediate Supervision Metrics)

**① 코사인 거리:**
Linq-Embed-Mistral (Choi et al., 2024) 임베딩을 사용한 코사인 유사도의 보수:
$$\Delta_\text{cosine}(x, y) = 1 - \frac{\mathbf{e}_x \cdot \mathbf{e}_y}{\|\mathbf{e}_x\| \|\mathbf{e}_y\|}$$

**② Soft N-Grams:**
원본과 편집 텍스트 간 $a$ ~ $b$ 단어 길이의 모든 구문(구절)을 열거하고, 코사인 유사도가 임계값 $\tau$ 이상인 구문 비율을 계산하는 정밀도 기반 메트릭. $\tau=1$일 때 전통적 n-gram 중복률로 귀결된다.

---

### 2.3 모델 구조

```
[입력: 편집된 텍스트 y]
        ↓
[Mistral/Llama 기반 LLM (3B~24B)]
    (QLoRA로 파인튜닝)
        ↓
[Edit CLS Head: LayerNorm + Linear]
        ↓
[N-way Softmax (4/5/6 버킷)]
        ↓
[가중 평균 디코딩]
        ↓
[출력: AI 편집 정도 스코어 ∈ [0,1]]
```

**주요 하이퍼파라미터:**
- Backbone: Mistral Small (24B) — 최고 성능 모델
- 최적화: AdamW, lr=3e-5, batch size=24, 1 epoch
- LoRA 타겟: 모든 선형 레이어 (self-attention QKV, output, MLP)
- 학습 환경: 8×A100 GPU, 약 8시간

---

### 2.4 성능 향상

#### 이진 분류 성능 (Human vs. Any AI)

| 모델 | 정확도 (%) | F1 |
|------|-----------|-----|
| FastDetectGPT | 69.1 | 80.5 |
| Binoculars | 68.6 | 81.4 |
| Pangram | 80.7 | 83.7 |
| **EditLens (SNG)** | **94.0** | **95.6** |
| **EditLens (Cosine)** | **93.8** | **95.4** |

#### 3분류 성능 (Human / AI-Edited / AI-Generated)

| 모델 | 유형 | Macro-F1 | AI-Edited F1 |
|------|------|----------|--------------|
| GPTZero | Ternary | 72.7 | 50.9 |
| **EditLens (Cosine)** | Regression | **90.4** | **86.8** |

#### APT-Eval 상관계수 (편집 크기와의 피어슨 상관)

| 모델 | 의미적 유사도↓ | Levenshtein↑ | Jaccard↑ |
|------|--------------|--------------|----------|
| Pangram | -0.491 | 0.615 | 0.556 |
| **EditLens** | **-0.606** | **0.799** | **0.781** |

---

### 2.5 한계

1. **단일 편집 패스 가정:** 주요 실험은 인간 텍스트에 AI가 한 번 편집하는 시나리오에 집중됨 (다중 편집 사례 연구는 일부 제시)
2. **Soft N-Grams의 단축 불변성:** 텍스트를 단순 삭제해도 Soft N-Grams 점수가 1이 됨 → 삭제 중심 편집 과소 추정 가능
3. **OOD 성능 저하:** 도메인 OOD 시 Macro-F1 0.904→0.866 (-0.038), LLM OOD 시 0.904→0.850 (-0.054) 저하
4. **비원본 추론의 근본적 한계:** 원본 텍스트 없이 편집된 텍스트만으로 추론하므로, 극단적으로 가벼운 편집 감지에 한계
5. **대규모 파라미터 의존:** 최고 성능이 24B 모델에서 나오므로 경량화 연구 필요
6. **인간 주석자 수 제한:** 3명의 주석자로만 검증 (확장 필요)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 논문에서 확인된 일반화 증거

#### ① 미지 프롬프트(Unseen Prompts)에 대한 일반화
편집 프롬프트 303개를 Train/Val/Test로 분할하여, **모델이 특정 프롬프트에 과적합하지 않도록** 설계. 테스트 시 본 적 없는 프롬프트에도 일관된 성능 유지.

#### ② 미지 LLM(OOD LLM)에 대한 일반화
Llama-3.3-70B-Instruct-Turbo를 홀드아웃 LLM으로 설정:

$$\text{Macro-F1}_{\text{OOD LLM}} = 0.850 \quad (\Delta = -0.054 \text{ vs. in-distribution})$$

#### ③ 미지 도메인(OOD Domain)에 대한 일반화
Enron 이메일 데이터셋을 홀드아웃 도메인으로 설정:

$$\text{Macro-F1}_{\text{OOD Domain}} = 0.866 \quad (\Delta = -0.038 \text{ vs. in-distribution})$$

#### ④ 다중 편집(Multi-Edit) 일반화
단일 인간 텍스트에 5회 연속 AI 편집을 적용했을 때, EditLens 점수의 평균값이 **단조 증가(Monotonically Increasing)**하는 것을 확인 (Figure 5):

$$\bar{s}_1 = 0.01, \quad \bar{s}_2 = 0.39, \quad \bar{s}_3 = 0.52, \quad \bar{s}_4 = 0.63, \quad \bar{s}_5 = 0.65$$

#### ⑤ AI→AI 편집 텍스트 일반화
| 시나리오 | 평균 점수 변화 |
|---------|--------------|
| 인간 텍스트에 AI 편집 | +0.38 |
| AI 텍스트에 AI 편집 | **-0.05** (거의 변화 없음) |

이는 EditLens가 **"AI 편집의 존재"가 아닌 "인간→AI 방향의 변화"를 감지**함을 시사 — 중요한 질적 일반화 증거.

#### ⑥ 인간 편집 AI 텍스트(BEEMO) 일반화
BEEMO 데이터셋에서 AI 텍스트에 인간이 편집을 가하면:

$$\bar{\Delta}_\text{score} = -0.33 \pm 0.30$$

88.9%의 문서에서 점수가 감소 → **역방향(AI→인간) 편집도 감지 가능**

### 3.2 일반화 향상을 위한 추가 가능성

#### (A) 감독 메트릭 다양화
현재 코사인 거리와 Soft N-Grams 두 가지만 사용. 향후 다음을 추가 탐색할 수 있다:
- **BERTScore** (Zhang et al., 2020): 문맥적 의미 유사도
- **BLEURT** (Sellam et al., 2020): 학습된 평가 메트릭
- **MAUVE** (Pillutla et al., 2021): 텍스트 분포 거리

#### (B) 다국어·다도메인 확장
현재 영어 텍스트에 집중. 다국어 사전학습 모델(mBERT, XLM-R)을 백본으로 활용하면 일반화 범위 확장 가능.

#### (C) 멀티태스크 학습 강화
현재 prompt classification + edit prediction 두 헤드를 동시 학습하는 멀티태스크 구조를 사용 중. 저작 귀속(authorship attribution), 표절 감지 등 관련 태스크와의 공동 학습이 일반화에 기여할 수 있음.

#### (D) 더 다양한 편집 행위자 포함
현재는 주로 단일 AI 편집에 집중. 인간-AI 반복 교대 편집, 여러 AI 모델의 순차 편집 등 복잡한 공동 저작 시나리오 데이터를 추가하면 실세계 일반화 성능이 향상될 것.

---

## 4. 미래 연구에 대한 영향 및 고려 사항

### 4.1 앞으로의 연구에 미치는 영향

#### (A) AI 감지 패러다임 전환
- **이진 → 연속 회귀**로의 패러다임 전환을 선도
- AI 사용의 "정도(degree)"를 측정하는 새로운 연구 방향 제시
- 정책 입안자가 "허용 가능한 AI 사용 수준"을 구체적 임계값으로 정의할 수 있게 함 (Jabarian & Imas, 2025의 policy cap 프레임워크 적용 가능)

#### (B) 저작권·저작자 귀속 연구
- 논문, 특허, 창작물의 **저자 기여도 분석**에 활용 가능
- "AI 지원 vs. AI 생성"의 법적·윤리적 경계 설정을 위한 도구 제공

#### (C) 교육 및 학문적 정직성
- 학생의 AI 사용 정도를 연속적으로 측정하여 **단순 탐지가 아닌 정도 평가** 가능
- 거짓 양성(무고한 학생 처벌)을 줄이는 방향의 정책 구현 지원

#### (D) 데이터셋 및 벤치마크 공헌
- 공개된 데이터셋(60k train, 6k test, 2.4k val)과 모델 가중치가 향후 동질적 혼합 텍스트 연구의 **표준 벤치마크** 역할 수행 가능

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 구체적 내용 |
|---------|-----------|
| **거짓 양성 문제** | 비영어권 텍스트, 비원어민 작성 텍스트, 특수 도메인(법률, 의학)에서의 오탐률 분석 필요 |
| **적대적 공격 내성** | "humanizer" 도구(Masrour et al., 2025)나 적대적 패러프레이징(Cheng et al., 2025)에 대한 모델 강건성 검증 필요 |
| **워터마킹과의 통합** | Kirchenbauer et al. (2024)의 LLM 워터마킹과 EditLens를 결합한 하이브리드 감지 시스템 가능성 |
| **경량화 연구** | 현재 24B 모델이 최고 성능이나, 교육 현장 배포를 위해 3B~8B 수준에서의 성능-효율 트레이드오프 연구 필요 |
| **다언어·다문화 확장** | 영어 외 언어, 코드-스위칭 텍스트에서의 일반화 성능 검증 |
| **인과성 분석** | 어떤 언어적 특성이 높은 AI 편집 점수를 유발하는지 **설명 가능성(Explainability)** 연구 병행 필요 |
| **시간적 일반화** | 새로운 LLM(GPT-5, Claude 5 등)이 등장할 때 모델 재학습 없이 적용 가능한지 지속적 평가 필요 |
| **소수 편집 유형 강화** | Grammar & Mechanics 프롬프트가 전체의 2%에 불과 — 클래스 불균형 문제 해소 방안 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 접근법 | 핵심 방법 | EditLens 대비 |
|------|------|--------|---------|--------------|
| **DetectGPT** (Mitchell et al.) | 2023 | 이진 분류 | 확률 곡률 기반 제로샷 | 혼합 텍스트 감지 불가 |
| **FastDetectGPT** (Bao et al.) | 2024 | 이진 분류 | 조건부 확률 곡률 | 효율적이나 AI-edited 감지 취약 |
| **Binoculars** (Hans et al.) | 2024 | 이진 분류 | 두 LLM의 퍼플렉서티 비율 | AI-edited F1=37.4% (EditLens: 86.8%) |
| **SeqXGPT** (Wang et al.) | 2023 | 이질적 혼합 | 문장 수준 분류 | 동질적 혼합에 부적합 |
| **RoFT / AI Boundary** (Kushnareva et al.) | 2024 | 이질적 혼합 | 경계 감지 | 동질적 혼합에 부적합 |
| **PaLD** (Lei et al.) | 2025 | 이질적 혼합 | 부분 LLM 작성 감지 | 토큰별 레이블 필요 → 동질적에 불가 |
| **DetectAIve** (Abassy et al.) | 2024 | 3분류 | 4범주 분류 | Macro-F1=52.5% vs. EditLens 90.4% |
| **GPTZero** (Tian & Cui) | 2023 | 3분류 | 퍼플렉서티 + 버스티니스 | AI-edited F1=50.9% vs. EditLens 86.8% |
| **HERO** (Wang et al.) | 2025 | 3분류 | 조작된 텍스트 감지 | AI-edited 정도 정량화 불가 |
| **Ghostbuster** (Verma et al.) | 2024 | 이진 분류 | 약한 언어모델 특성 | 혼합 텍스트 정량화 불가 |
| **BEEMO** (Artemova et al.) | 2024 | 데이터셋 | 인간 편집 AI 텍스트 벤치마크 | EditLens가 이 데이터셋에서 평가됨 |
| **DAMAGE** (Masrour et al.) | 2025 | 이진 분류 | 적대적 수정 AI 텍스트 감지 | EditLens 팀의 관련 연구 |
| **AI Polish (APT-Eval)** (Saha & Feizi) | 2025 | 평가 데이터셋 | 4단계 AI 광택(polish) 수준 | EditLens가 상관 r=-0.606으로 최고 성능 |
| **Adversarial Paraphrasing** (Cheng et al.) | 2025 | 공격 | 패러프레이징으로 감지 회피 | EditLens의 강건성 한계 노출 가능 |

### 핵심 차별점 요약

```
기존 연구: 이진/이질적/3분류 → 범주 판별
EditLens: 연속 회귀 → 편집 정도 정량화 (최초)
```

EditLens는 **"AI가 관여했는가(Yes/No)"** 가 아닌 **"AI가 얼마나 관여했는가(0~1)"** 를 측정함으로써, 기존 연구의 공백을 채우고 세분화된 정책 적용을 가능하게 하는 핵심 기여를 한다.

---

## 참고 자료

- **Thai, K., Emi, B., Masrour, E., & Iyyer, M. (2025).** *EditLens: Quantifying the Extent of AI Editing in Text.* arXiv:2510.03154v1. https://arxiv.org/abs/2510.03154
- **Saha, S., & Feizi, S. (2025).** *Almost AI, Almost Human: The Challenge of Detecting AI-Polished Writing.* arXiv:2502.15666.
- **Chatterji, A. et al. (2025).** *How People Use ChatGPT.* NBER Working Paper No. 34255.
- **Bao, G. et al. (2024).** *Fast-DetectGPT.* ICLR 2024. arXiv:2310.05130.
- **Hans, A. et al. (2024).** *Spotting LLMs With Binoculars.* arXiv.
- **Mitchell, E. et al. (2023).** *DetectGPT.* ICML 2023. arXiv:2301.11305.
- **Kushnareva, L. et al. (2024).** *AI-Generated Text Boundary Detection with RoFT.* COLM 2024.
- **Lei, E., Hsu, H., & Chen, C.-F. (2025).** *PaLD.* ICLR 2025.
- **Abassy, M. et al. (2024).** *LLM-DetectAIve.* EMNLP 2024 Demo.
- **Wang, P. et al. (2023).** *SeqXGPT.* EMNLP 2023.
- **Artemova, E. et al. (2024).** *Beemo: Benchmark of Expert-edited Machine-Generated Outputs.* arXiv:2411.04032.
- **Verma, V. et al. (2024).** *Ghostbuster.* NAACL 2024.
- **Dettmers, T. et al. (2023).** *QLoRA.* NeurIPS 2023.
- **Choi, C. et al. (2024).** *Linq-Embed-Mistral Technical Report.* arXiv:2412.03223.
- **Jabarian, B. & Imas, A. (2025).** *Artificial Writing and Automated Detection.* SSRN:5407424.
- **Masrour, E., Emi, B., & Spero, M. (2025).** *DAMAGE.* arXiv:2501.03437.
- **Cheng, Y. et al. (2025).** *Adversarial Paraphrasing.* arXiv:2506.07001.
- **Kirchenbauer, J. et al. (2024).** *A Watermark for Large Language Models.* arXiv:2301.10226.
- **Wang, Y. et al. (2025).** *Real, Fake, or Manipulated?* arXiv:2509.15350.
- **Su, Z. et al. (2025).** *HACo-Det.* ACL 2025.
- **Ng, J.-P. & Abrecht, V. (2015).** *Better Summarization Evaluation with Word Embeddings for ROUGE.* EMNLP 2015.
- **Emi, B. & Spero, M. (2024).** *Technical Report on the Pangram AI-Generated Text Classifier.* arXiv:2402.14873.
