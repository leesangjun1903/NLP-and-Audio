# Memorization Dynamics in Knowledge Distillation for Language Models

> **참고 문헌**: Borkar et al. (2026), "Memorization Dynamics in Knowledge Distillation for Language Models", arXiv:2601.15394v2 [cs.CL], 8 Aug 2026. (논문 원문 PDF 직접 분석)

---

## 1. Executive Summary (10문장 이내)

이 논문은 지식 증류(Knowledge Distillation, KD)가 언어 모델의 훈련 데이터 암기(memorization)에 미치는 영향을 최초로 체계적으로 분석한 연구이다.  
저자들은 Pythia, OLMo-2, Qwen-3 세 가지 LLM 패밀리와 FineWeb, Wikitext, Nemotron-CC-v2 세 가지 데이터셋을 사용하여 Teacher-Student-Baseline 삼각 비교 프레임워크를 구성하였다.  
핵심 결과는 네 가지로,  
(1) 증류된 Student 모델은 일반 파인튜닝 대비 암기를 50% 이상 감소시키고,  
(2) 특정 예제는 본질적으로 암기되기 쉬운(easy-to-memorize) 특성을 가지며 Student 암기의 95% 이상을 차지하고,  
(3) 증류 전에 zlib 엔트로피·KL 발산·퍼플렉시티 지표로 암기 예측이 가능하며,  
(4) 소프트 증류와 하드 증류는 유사한 전체 암기율을 보이나 하드 증류는 Teacher로부터의 어려운 예제 상속 위험이 2.7배 높다.  
KD는 Student가 Teacher 일반화 능력의 78%를 회복하면서도 Teacher 전용 암기의 단 2%만 상속하는 효율적인 정규화(regularization) 메커니즘으로 작동한다.  
암기 감소 메커니즘은 KL 발산 손실이 Student로 하여금 불확실한 예제에 대해 평탄한 분포를 출력하게 허용하는 반면, 교차 엔트로피는 강제 암기를 유발하기 때문이다.  
또한, 암기가 예상되는 예제들을 훈련 전 필터링하면 암기 건수가 706개에서 4개로 99.4% 감소함을 확인하였다.

> 💡 **용어 설명**
> - **지식 증류(Knowledge Distillation, KD)**: 대형 Teacher 모델의 출력 분포(소프트 레이블)를 소형 Student 모델이 학습하도록 하는 모델 압축 기법 (Hinton et al., 2015)
> - **암기(Memorization)**: 언어 모델이 훈련 데이터의 특정 시퀀스를 거의 그대로 재생성할 수 있는 현상; 프라이버시 침해 위험과 직결됨
> - **퍼플렉시티(Perplexity, PPL)**: 언어 모델이 주어진 텍스트를 얼마나 잘 예측하는지 나타내는 지표; 낮을수록 모델이 해당 텍스트를 더 잘 이해함

### 1-1. 연구의 목적과 필요성

**목적**: KD 파이프라인에서 훈련 데이터 암기의 역학(dynamics)을 체계적으로 이해하고, 암기 예측·감소 방법을 제시한다.

**필요성**:
1. KD는 DeepSeek-R1, Gemma 등 실용적 LLM 개발에 광범위하게 채택되었으나, 암기 역학은 사전학습/파인튜닝 대비 훨씬 덜 연구되었다 (Section 1, p.1-2).
2. KD는 프라이버시 보호 메커니즘으로 언급되나, 실제 데이터 추출 공격에 대한 저항력이 실증적으로 검증되지 않았다 (Section 1, p.2).
3. 대형 Teacher 모델은 필연적으로 훈련 데이터를 상당량 암기하므로, Student가 이를 얼마나 상속하는지 규명이 필요하다 (Section 3, p.4).

> 💡 **용어 설명**
> - **데이터 추출 공격(Data Extraction Attack)**: 공격자가 모델에 프롬프트를 주입하여 훈련 데이터 일부를 그대로 재생성시키는 프라이버시 공격 기법 (Carlini et al., 2020)
> - **정규화(Regularization)**: 모델이 훈련 데이터에 과적합되지 않도록 일반화 능력을 높이는 기법의 총칭

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 (실험 결과) | 위치 |
|---|-----------|-----------------|------|
| 1 | KD는 파인튜닝 대비 암기를 대폭 감소시키면서 일반화 성능을 개선 | Student(FineWeb): 0.07% vs Baseline: 0.17% vs Teacher: 0.33%; Student 검증 PPL 17.31 < Baseline 17.69 | Table 1, Table 2, Section 3.1 |
| 2 | 특정 예제는 본질적으로 암기되기 쉬우며, Student 암기의 95.7%가 이 범주에 속함 | 706개 중 676개(95.7%)가 Teacher·Baseline 모두가 암기한 easy-to-memorize 예제 | Figure 4, Section 3.2 |
| 3 | 암기 예측은 증류 전에 가능 (AUC-ROC 0.9997) | 로지스틱 회귀 분류기: zlib 엔트로피, Teacher·Baseline PPL, KLD 특징 사용 | Section 4.1, Table 4 |
| 4 | KD는 Shannon 엔트로피·로그 확률 관점에서 강제 암기(forced memorization)를 억제 | Baseline은 고엔트로피 예제에 고확률 강제 부여, Student는 낮은 로그 확률 유지 | Figure 8, Section 5 |
| 5 | 하드 증류는 소프트 증류보다 Teacher 전용 어려운 예제 상속 위험 2.7배 높음 | 하드 증류: 50개 Teacher 전용 상속 vs 소프트 증류: 18개 | Figure 10, 11, Section 6 |
| 6 | 아키텍처 간 easy-to-memorize 예제는 공유되지 않으나, 동일 패밀리 내에서는 일관됨 | Pythia-OLMo2-Qwen3 간 암기 중복 없음; 퍼플렉시티 분포는 모델별 inductive bias에 의존 | Figure 6, 13, Section 3.2.1 |

---

## 2-1. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 해결하고자 하는 문제

KD 파이프라인에서 훈련 데이터 암기 역학이 불명확함: (1) Student가 Teacher의 암기를 얼마나 상속하는가, (2) Student가 동일 크기 Baseline보다 더 암기하는가, (3) 암기를 사전에 예측/차단할 수 있는가, (4) 소프트 vs 하드 증류의 암기 위험 차이는 무엇인가.

---

### 제안하는 방법 및 수식

#### ① Soft Distillation (KL 발산 손실) — 식 (1), (2)

$$\mathcal{L}_{\text{KD}} = T^2 \sum_{i=1}^{|V|} P^{\tau}_{\text{teacher}}(i) \log \frac{P^{\tau}_{\text{teacher}}(i)}{P^{\tau}_{\text{student}}(i)} \tag{1}$$

**기호 설명**:
- $T$: 온도 파라미터 (실험에서 2.0으로 설정); 높을수록 확률 분포가 부드러워짐
- $|V|$: 어휘 크기 (vocabulary size)
- $P^{\tau}_{\text{teacher}}(i)$: Teacher의 온도 스케일된 토큰 $i$에 대한 확률
- $P^{\tau}_{\text{student}}(i)$: Student의 온도 스케일된 토큰 $i$에 대한 확률
- $T^2$ 항: 온도 스케일 보정 인자 (Hinton et al., 2015 표준화)

$$P^{\tau}(i) = \text{softmax}\!\left(\frac{z_i}{T}\right) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)} \tag{2}$$

**기호 설명**:
- $z_i$: 토큰 $i$에 대한 소프트맥스 이전 로짓(pre-softmax logit)
- $z_j$: $j$번째 어휘 항목의 로짓

> 💡 **용어 설명**
> - **KL 발산(Kullback-Leibler Divergence)**: 두 확률 분포의 차이를 측정하는 정보이론적 척도; $\text{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$. 비대칭적이며 항상 0 이상임
> - **로짓(Logit)**: 소프트맥스 함수 적용 이전의 신경망 출력값 (원시 점수)
> - **소프트맥스(Softmax)**: 로짓을 확률 분포로 변환하는 함수

#### ② Hard Distillation (시퀀스 레벨) — 식 (3)

$$\mathcal{L}_{\text{hard}}(\theta) = -\mathbb{E}_{\hat{x} \sim \mathcal{D}_{\text{hard}}} \left[ \sum_{t=1}^{T} \log P_{\theta}(\hat{x}_t \mid \hat{x}_{<t}) \right] \tag{3}$$

**기호 설명**:
- $\theta$: Student 모델 파라미터
- $\mathcal{D}_{\text{hard}}$: Teacher가 생성한 합성 시퀀스로 구성된 데이터셋
- $\hat{x}_t$: 시각 $t$에서의 생성 토큰
- $\hat{x}_{<t}$: 시각 $t$ 이전의 문맥 토큰 시퀀스
- $P_{\theta}(\hat{x}\_t \mid \hat{x}_{ < t})$: Student 모델의 조건부 토큰 확률

> 💡 **용어 설명**
> - **하드 증류(Hard Distillation)**: Teacher가 실제로 생성한 텍스트 시퀀스를 Student의 학습 레이블로 사용하는 방식. 블랙박스 API 상황에서도 사용 가능
> - **소프트 증류(Soft Distillation)**: Teacher의 전체 확률 분포(soft target)를 Student가 모방하는 방식

#### ③ Shannon 엔트로피 (암기 메커니즘 분석용)

$$H_t(x) = -\sum_{v \in V} p_{\theta}(v \mid x_{ < t}) \log p_{\theta}(v \mid x_{ < t})$$

$$\bar{H}_{\theta}(x) = \frac{1}{K} \sum_{t=T-K}^{T-1} H_t(x) \tag{Section 5}$$

**기호 설명**:
- $x = (x_1, \ldots, x_T)$: 토크나이즈된 입력 시퀀스
- $p_{\theta}(v \mid x_{<t})$: 위치 $t$에서 토큰 $v$에 대한 모델 조건부 확률
- $V$: 어휘 집합
- $K = 50$: 평균 계산에 사용되는 마지막 토큰 수
- $\bar{H}_{\theta}(x)$: 시퀀스의 평균 Shannon 엔트로피

> 💡 **용어 설명**
> - **Shannon 엔트로피**: 확률 분포의 불확실성을 측정하는 지표. 높은 엔트로피 = 높은 불확실성 = 모델이 다음 토큰을 예측하기 어려운 상태

#### ④ 발견 가능한 암기(Discoverable Memorization) 평가 기준

$$\mathcal{G}(x_{1:k}) = x_{k+1:L} \quad (k=50, \; L=100) \tag{Section 2}$$

**기호 설명**:
- $x \in \mathcal{D}$: 훈련 시퀀스
- $x_{1:k}$: 50토큰 프리픽스 (입력 프롬프트)
- $x_{k+1:L}$: 50토큰 서픽스 (정답)
- $\mathcal{G}(\cdot)$: 그리디 디코딩 함수 (greedy decoding)
- 생성 결과가 정답 서픽스와 **정확히 일치**하면 해당 예제를 암기된 것으로 분류

---

### 모델 구조

| 모델 | 아키텍처 | 크기 | 학습 방식 |
|------|----------|------|-----------|
| $M_{\text{teacher}}$ | Pythia 12B (주실험) / OLMo-2 7B / Qwen-3 8B | 대형 | Cross-Entropy 파인튜닝 |
| $M_{\text{baseline}}$ | Pythia 1.4B / OLMo-2 1B / Qwen-3 1.7B | 소형 | Cross-Entropy 파인튜닝 |
| $M_{\text{student}}$ | Pythia 1.4B (Base에서 초기화) | 소형 | KL 발산 KD (소프트) 또는 Cross-Entropy (하드) |

- **데이터**: FineWeb (1M 예제, 256 토큰/예제), Wikitext-103, Nemotron-CC-v2
- **학습률**: $5 \times 10^{-5}$, cosine decay
- **암기 평가**: 1M 훈련 예제 전체에 대해 50-토큰 프리픽스로 50-토큰 서픽스 정확 일치 여부 판별

---

### 성능 향상 및 한계

**성능 향상** (Table 1, Table 2):
- Pythia Student: PPL 17.31 < Baseline 17.69 (검증셋)
- 암기율: Student 0.07% vs Baseline 0.17% (FineWeb 기준 ~2.4배 감소)
- 암기 예측 분류기: AUC-ROC $0.9997 \pm 0.0005$, Recall $1.0000 \pm 0.0000$

**한계**:
1. 암기 평가가 **정확 일치(exact match)** 기준에 의존하므로 근사적 암기(paraphrase 수준)는 과소평가될 가능성 존재 (Appendix A.2에서 부분 검증)
2. 실험이 주로 파인튜닝 설정에 집중되어 있으며, 사전학습 스케일에서의 완전한 검증은 제한적 (Section A.7은 예비 실험 수준)
3. 하드 증류 시 Teacher 출력의 perplexity 차이로 인해 직접적 perplexity 비교가 불가능하여 다운스트림 태스크로 대체 평가
4. 아키텍처 간 easy-to-memorize 예제 비중복 원인이 "inductive bias"로만 설명되며, 메커니즘의 세부 규명은 미흡

---

## 3. 각 주장에 페이지 또는 Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| KD가 암기를 50% 이상 감소 (FineWeb 기준 2.4배) | Table 2 (p.5), Section 3.1 (p.4) |
| Student는 Teacher 일반화의 78% 회복, 암기는 2% 상속 | Section 3.1 (p.4, Abstract) |
| Easy-to-memorize 예제가 Student 암기의 95.7% | Figure 4 (p.5), Section 3.2 (p.5-6) |
| 암기가 결정론적: 동일 패밀리 모델 간 96% 일관성 | Section 3.2 (p.5), Figure 3 (p.5) |
| 암기 예측 분류기 AUC 0.9997 | Section 4.1 (p.7), Table 4 (p.20) |
| 예제 제거 시 암기 99.4% 감소 (706개 → 4개) | Section 4.2 (p.8) |
| KD 정규화 메커니즘: 강제 암기 억제 | Figure 8 (p.9), Section 5 (p.8-9) |
| 하드 증류의 Teacher 암기 상속 2.7배 위험 | Figure 10, 11 (p.11), Section 6 (p.10) |
| 온도 상승 시 암기 감소 (T=1→4: 1591→12개) | Figure 2 (p.3) |
| Qwen-3 Student: PPL 25.65 < Baseline 33.23 | Table 1 (p.4) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 연구 주제

| 구분 | 내용 |
|------|------|
| **저자 보고** | KD 파이프라인에서 Teacher/Student/Baseline 세 모델 간 암기 역학을 비교 분석 |
| **내 해석** | 이 연구는 KD를 단순한 모델 압축 기법이 아닌 **자연스러운 프라이버시 방어 레이어**로 재정의하려는 시도이며, 향후 LLM 개발 파이프라인에서 프라이버시-유틸리티 트레이드오프를 재평가하게 하는 연구 |

### 방법

| 구분 | 내용 |
|------|------|
| **저자 보고** | 식 (1)의 Forward KL 발산으로 Student 훈련; 50토큰 프리픽스-서픽스 정확 일치로 암기 판정; 로지스틱 회귀 분류기로 암기 예측 |
| **내 해석** | 암기 판정 기준이 **정확 일치**에 한정되므로, 실제 프라이버시 위험(의미론적 암기, 개인정보 포함 여부 등)보다 보수적으로 측정됨. 분류기의 AUC 0.9997은 거의 완벽에 가깝지만, 이는 극도로 불균형한 클래스(1:3 비율 + 100회 리샘플링)와 매우 단순한 특징(zlib 엔트로피)이 지배적으로 작용했기 때문일 가능성이 높음 |

### 결과

| 구분 | 내용 |
|------|------|
| **저자 보고** | Student FineWeb 암기율 0.07%, Baseline 0.17%; Teacher 전용 암기 상속 0.9%; 암기 예제 제거 시 706→4개 |
| **내 해석** | 암기 절대 건수(예: Qwen-3 Student 약 2600건)가 여전히 적지 않으며, 실제 운영 환경에서는 Teacher 교체 주기·데이터 업데이트 등의 동적 요소가 결과를 변화시킬 수 있음. 특히 OLMo-2의 Teacher 암기율(8.90%)이 Qwen-3(3.45%), Pythia(0.33%)와 큰 차이를 보이는 원인 분석이 부재함 |

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적으로 취약한 부분

| 항목 | 문제점 |
|------|--------|
| **분류기 AUC 0.9997** (Table 4) | 1:3 불균형 비율에서 100회 리샘플링 시 비메모리제이션 예제가 매 회 다르게 샘플링됨. 이는 독립적 테스트가 아닌 반복 샘플링으로, 실제 일반화 성능의 신뢰 구간이 과소 추정될 위험 있음 |
| **78% 일반화 회복률** (Section 3.1, p.4) | 이 수치의 산출 방식과 신뢰 구간이 본문에 명확히 제시되지 않음 |
| **Teacher 전용 암기 상속 0.9% (18/1955개)** (Figure 4) | 3회 독립 실행의 합집합(union)을 사용하므로, 개별 실행에서의 분산이 마스킹됨 |
| **하드 증류 Teacher 상속 2.7배** (Figure 11) | 절대값이 소프트 18개 vs 하드 50개로, 표본 크기가 작아 배율 차이의 통계적 유의성 검정 부재 |
| **Wikitext 결과** (Table 2 Left) | Teacher 1.75%, Baseline 0.21%, Student 0.10%로 다른 데이터셋과 상이한 패턴을 보이나 원인 분석 없음 |

### 🚫 비교 불가능한 수치

| 항목 | 이유 |
|------|------|
| **하드 vs 소프트 학생 PPL** (Section 6) | 하드 Student는 합성 데이터($\mathcal{D}_{\text{hard}}$)로 훈련되므로 동일 검증셋에서의 PPL 직접 비교 불가. 저자도 이를 인지하고 LAMBADA/Winogrande 사용 |
| **OLMo-2 Teacher 암기율 8.90% vs Pythia 0.33%** (Table 2 Right) | 모델 패밀리 간 사전학습 데이터·에폭 수·아키텍처 차이가 혼재되어 있어 직접 비교 불가 |
| **사전학습 설정 실험** (Table 8, Section A.7) | Pythia 2.8B Student 0.06%는 파인튜닝 설정 Pythia 1.4B Student 0.07%와 크기가 달라 직접 비교 불가 |

---

## 6. 논문이 답하지 않는 질문

| # | 미답변 질문 |
|---|-----------|
| 1 | KD의 암기 감소 효과는 **파인튜닝 에폭 수가 다를 때** (Teacher 3 에폭 vs Student 4-5 에폭) 어느 정도 에폭 수 차이에 의한 것인가? |
| 2 | **OLMo-2 Teacher의 암기율(8.90%)**이 다른 Teacher보다 현저히 높은 원인은 무엇인가? |
| 3 | Easy-to-memorize 예제가 아키텍처 간 비중복인 원인으로 "inductive bias"를 지적하나, **구체적 메커니즘(어텐션 패턴, 토크나이저 차이 등)**은 해명되지 않음 |
| 4 | **다단계 증류(cascade distillation)** — Teacher → 중간 모델 → Student — 에서 암기가 어떻게 전파되는가? |
| 5 | 증류 후 **언러닝(unlearning)** 또는 **차분 프라이버시(Differential Privacy)**와 KD를 결합하면 암기를 추가로 얼마나 줄일 수 있는가? |
| 6 | **RLHF/DPO로 후훈련된 모델**에서 KD의 암기 감소 효과가 동일하게 유지되는가? |
| 7 | 암기 예측 분류기를 **새로운 도메인 데이터**에 적용할 때 일반화 성능은 어떠한가? |
| 8 | 하드 증류에서 Teacher가 **잘못된(hallucinated) 정보**를 암기한 경우, Student로의 전달 패턴은? |

> 💡 **용어 설명**
> - **언러닝(Machine Unlearning)**: 훈련 후 특정 데이터의 영향을 모델에서 선택적으로 제거하는 기법
> - **차분 프라이버시(Differential Privacy, DP)**: 출력에 수학적으로 보정된 노이즈를 추가하여 개별 데이터 기여를 보호하는 프라이버시 보장 프레임워크

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 — 실험 프레임워크 (p.1)

**구성**: 왼쪽(훈련 설정), 오른쪽(암기 평가 방식)

**해석**: 세 모델($M_{\text{teacher}}$, $M_{\text{baseline}}$, $M_{\text{student}}$)을 동일 데이터셋에 훈련하되 손실 함수만 다르게 설정한 이 대조 설계는 KD 효과를 명확히 분리할 수 있게 한다. 50-토큰 프리픽스-서픽스 정확 일치 기준은 "discoverable memorization" (Nasr et al., 2023)을 따르며, 재현 가능하고 보수적인 평가를 가능하게 한다. 이 설계의 강점은 Teacher·Student·Baseline을 동시에 비교함으로써 "암기 상속"과 "자체 암기"를 분리할 수 있다는 점이다.

---

### Figure 4 — 암기 예제 중복 (p.5)

**구성**: Teacher(하단), Baseline(좌상), Student(우상) 간 벤 다이어그램

**해석**: 세 가지 핵심 통찰을 동시에 전달한다:
- **676개**: Teacher+Baseline+Student 모두 암기 → *easy-to-memorize* 예제의 핵심 (Student 암기의 95.7%)
- **1,937개**: Teacher만 암기 → Teacher의 대용량 모델 고유 암기; Student에게 전달되지 않음
- **18개**: Teacher+Student 암기 but Baseline 제외 → 매우 소수의 "어려운" Teacher 특유 암기만 상속

이는 KD가 Teacher의 일반화 능력(676개의 공통 패턴)은 흡수하나, Teacher 전용 암기는 효과적으로 차단함을 시각화한다.

---

### Figure 5 — Easy-to-memorize 예제의 본질적 특성 (p.6)

**구성**: X축 = zlib 엔트로피, Y축 = Baseline 퍼플렉시티; 빨간점 = easy-to-memorize, 회색점 = 기타

**해석**: Easy-to-memorize 예제들이 **저엔트로피(낮은 압축 가능성) + 저퍼플렉시티** 영역에 극단적으로 클러스터링됨을 보여준다. zlib 엔트로피가 낮다는 것은 해당 텍스트가 단순하고 반복적인 구조를 가짐을 의미하며, 이러한 텍스트는 모델 크기·아키텍처에 무관하게 쉽게 암기된다. 이 그림은 암기가 확률적(stochastic)이 아닌 **결정론적(deterministic)** 임을 시각화하는 핵심 증거이다. 분류기 설계에서 zlib 엔트로피가 압도적 예측력(계수 −4.50)을 갖는 이유를 직관적으로 설명한다.

> 💡 **용어 설명**
> - **zlib 엔트로피**: 텍스트를 zlib 알고리즘으로 압축했을 때의 길이(바이트). 반복적·단순한 텍스트일수록 압축률이 높고 엔트로피 값이 낮음

---

### Figure 8 — Shannon 엔트로피 vs 로그 확률 분석 (p.9)

**구성**: X축 = Shannon 엔트로피, Y축 = 로그 확률; 녹색=Teacher, 빨간색=Baseline, 파란색=Student(비암기), 주황색=Student(암기)

**해석**: KD가 암기를 억제하는 **메커니즘**을 가장 직접적으로 설명하는 그림:
- **Teacher (녹색)**: 고확률 + 저엔트로피 → 12B 모델은 복잡한 예제도 자연스럽게 모델링
- **Baseline (빨간색)**: 고확률 + **고엔트로피** → 1.4B 모델이 불확실한 예제에 교차 엔트로피로 강제 암기(**Forced Memorization**)
- **Student (파란색)**: 고엔트로피 + **저확률** → KL 발산이 평탄한 분포를 허용하여 강제 암기 방지
- **Student 암기 (주황색)**: Teacher와 겹치는 저엔트로피 영역 → Student는 스스로 확신하는 쉬운 예제만 선택적으로 암기

이 그림은 "KL 발산 = 자연스러운 정규화" 가설의 핵심 실증 증거이다.

---

### Figure 2 — 온도가 암기에 미치는 영향 (p.3)

**구성**: X축 = 증류 온도(T=1,2,3,4), Y축 = 암기 예제 수

**해석**: T=1에서 1,591개, T=2에서 668개, T=3에서 155개, T=4에서 12개로 **지수적 감소**를 보인다. 온도가 높을수록 Teacher의 확률 분포가 더 부드러워져 특정 토큰에 집중되지 않으며, 이로 인해 Student가 학습할 "암기할 만한" 날카로운 패턴이 희석된다. 실용적 시사점: 온도 조정만으로도 암기 위험을 대폭 제어할 수 있으나, 너무 높은 온도는 유용한 지식 전달도 희석시킬 수 있어 최적값 탐색이 필요하다. Jagielski et al. (2023)의 멤버십 추론 공격 결과(저온이 더 취약)와 방향이 다른 점도 주목할 만하다.

> 💡 **용어 설명**
> - **멤버십 추론 공격(Membership Inference Attack, MIA)**: 특정 데이터 샘플이 모델의 훈련 데이터에 포함되었는지 여부를 추론하는 프라이버시 공격

---

## 8. 결론: 시사점, 후속 연구 계획 및 추가 방향

### 저자 제시 시사점 (Section 8, p.12)

1. KD는 모델 유틸리티 향상과 훈련 데이터 암기 감소를 **동시에** 달성
2. 증류 모델은 primarily *easy-to-memorize* 예제를 암기
3. 고위험 예제를 증류 **전에** 예측하고 제거하는 것이 가능하며 매우 효과적
4. Shannon 엔트로피 + 로그 확률 분석으로 KD의 정규화 메커니즘을 기계론적으로 설명
5. 소프트 vs 하드 증류: 전반적 암기율은 유사하나 하드 증류가 어려운 예제 상속 위험이 더 높음

### 저자 제시 후속 연구 계획

논문에는 명시적인 future work 절이 없으나, 본문 곳곳에서 다음 방향이 암시됨:
- 다양한 Teacher-Student 크기 조합에서의 암기 역학 확장 연구
- 사전학습 규모에서의 완전한 KD 암기 분석 (A.7은 예비 수준)
- 암기 예측 분류기의 다양한 아키텍처/도메인 일반화

---

### 8-1. 모델의 일반화 성능 향상 가능성

#### 저자 보고 사실

- Pythia Student: Val Loss 2.85, PPL **17.31** < Baseline PPL 17.69 (Table 1)
- OLMo-2 Student: PPL **28.15** < Baseline PPL 34.61 (Table 1)
- Qwen-3 Student: PPL **25.65** < Baseline PPL 33.23 (Table 1)
- Wikitext에서도 Student PPL **15.36** < Baseline PPL 16.33 (Table 3)

이는 KD가 암기 감소와 일반화 향상을 **동시에 달성**함을 의미한다. 저자들은 Student가 Teacher 일반화 능력의 78%를 회복하면서 Teacher 암기의 2%만 상속한다고 보고하였다.

#### 일반화 향상 메커니즘 분석 (내 해석)

KD의 일반화 향상 메커니즘은 두 가지 관점에서 이해할 수 있다:

**① 소프트 레이블의 정보 이득**: Teacher의 소프트 확률 분포는 잘못된 클래스들 사이의 관계적 유사성 정보(예: "고양이"와 "개"가 "자동차"보다 더 유사)를 포함한다. 이는 단순 one-hot 레이블보다 풍부한 학습 신호를 제공한다.

**② 강제 암기 억제 → 일반화 표현 형성**: Figure 8에서 Baseline의 "강제 암기"는 모델이 실제로 이해하지 못하는 패턴에 과적합하는 것을 의미한다. KD는 이를 억제함으로써 모델이 진정으로 일반화 가능한 표현을 학습하게 유도한다.

**③ 잠재적 한계**: 일반화 향상 효과가 동일 분포 검증셋(FineWeb 검증셋)에서만 측정되었다. 도메인 이동(domain shift)이 있는 데이터셋이나 추론 태스크에서 이 우위가 유지되는지는 추가 검증이 필요하다. 하드 증류 비교에서 저자 스스로 동분포 PPL이 misleading할 수 있음을 인정하고 LAMBADA/Winogrande를 사용한 점이 이를 방증한다.

**실용적 제언**:
- 증류 온도 $T$를 높이면 암기가 추가 감소하지만, 일반화에 미치는 영향을 함께 모니터링해야 함
- 암기 예측 기반 데이터 필터링(Section 4.2)은 암기를 99.4% 줄이면서도 학습 데이터 품질을 유지할 수 있어 **실용적 일반화-프라이버시 파레토 최적화** 도구가 될 수 있음

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 본 논문의 참고문헌 섹션에 인용된 논문들에 한정하여 분석함. 제 훈련 데이터 기반 추가 논문은 정확성 보장이 어려워 포함하지 않음.

| 연구 | 주요 기여 | 본 논문과의 관계 |
|------|-----------|-----------------|
| Carlini et al. (2021/2023), "Quantifying Memorization" [arXiv:2202.07646] | k-암기 정의, 모델 크기·반복 횟수와 암기의 관계 확립 | 본 논문의 기초; "발견 가능한 암기" 정의 참조 |
| Jagielski et al. (NeurIPS 2023), "Students Parrot Their Teachers" | KD에서 Student가 Teacher 멤버십 정보를 누출함을 MIA로 입증 | 본 논문과 비교: 멤버십 누출 ≠ 데이터 추출 암기. 온도 효과 방향 불일치 (저온=취약 vs 본 논문 저온=고암기) |
| Dankers & Raunak (ACL 2025) | 기계번역 시퀀스 증류에서 Student가 Teacher 암기를 상속 | 본 논문은 이와 대조적으로 LLM 파인튜닝 KD에서는 상속이 매우 제한적임을 발견 |
| Singh (L2M2 Workshop, ACL 2025) | 증류가 암기를 감소시킴을 소규모 실험으로 확인 | 본 논문이 이를 대규모(3 패밀리, 3 데이터셋)로 확장·심화 |
| Zhang et al. (EMNLP 2025) | LLM KD에서 멤버십과 암기 연구 | 본 논문과 가장 직접적 경쟁 연구; 32토큰 암기에 집중 |
| Zeng et al. (ACL 2024) | 파인튜닝 LM에서의 암기 탐색 | Baseline 설정의 선행 연구 |
| Lee et al. (2025), "Distillation Robustifies Unlearning" | 증류가 바람직하지 않은 행동(적대적 elicitation 취약성)을 제거하면서 원하는 행동은 유지 | KD의 선택적 정보 필터링 특성을 다른 각도에서 확인 |
| MiniLLM (Gu et al., 2025), GKD (Agarwal et al., 2024) | 역방향 KL 발산, 온-폴리시 증류로 분포 이동 문제 해결 | 본 논문은 순방향 KL만 사용; 역방향 KL에서의 암기 역학 미탐구 |

#### 앞으로의 연구에 미치는 영향

1. **프라이버시-효율성 공동 최적화**: KD가 프라이버시 보호 도구로서 공식적으로 자리매김할 근거를 제공. 향후 Differential Privacy + KD 결합 연구의 기준점이 될 것

2. **Easy-to-memorize 개념의 확장**: zlib 엔트로피 기반 데이터 필터링이 데이터 큐레이션 파이프라인에 통합될 가능성. 이 개념은 능동 학습(active learning), 커리큘럼 학습(curriculum learning)과의 결합 연구로 확장 가능

3. **하드 증류 위험 인식**: 블랙박스 API 기반 증류(ChatGPT 스타일)가 산업에서 일반화됨에 따라, 하드 증류의 Teacher 암기 상속 위험에 대한 정책적 논의에 실증 데이터 제공

4. **역방향 KL(MiniLLM)에서의 암기 역학**: 순방향 KL과 역방향 KL의 암기 억제 차이 연구가 자연스러운 후속 과제로 부상

#### 앞으로 연구 시 고려할 점

1. **개인정보(PII) 포함 예제에 대한 특화 분석**: 암기와 실제 프라이버시 위험(주민번호, 이메일 등)의 관계를 직접 측정해야 함

2. **다단계·반복 증류(Iterative KD)**: 산업에서는 Teacher → 중간 모델 → Student의 다단계 증류가 일반적이므로 암기 누적 효과 연구 필요

3. **RLHF 후 단계 적용**: 실제 배포 모델은 KD 후 RLHF/DPO로 추가 훈련되므로, 이 단계에서 암기 패턴이 어떻게 변화하는지 연구 필요

4. **다국어·도메인 특화 설정**: 본 논문의 영어 중심 결과가 다국어·코드·의학 등 특수 도메인에서도 재현되는지 검증 필요

5. **암기 예측 분류기의 실용화**: 현재 분류기는 Baseline 모델 접근을 가정하지만, Baseline 없이 Teacher만으로의 성능(AUC 0.9998)이 더 높은 점을 활용한 실용적 도구 개발

6. **온도 최적화의 이론적 근거**: 경험적으로 T=2를 선택했으나, 암기 감소와 지식 전달 효율 간의 이론적 최적 온도 도출 연구 필요

---

**참고 자료**:
- 본 분석의 모든 수치, 수식, 주장은 다음 문서에서 직접 추출하였음:
  - Borkar et al. (2026), *Memorization Dynamics in Knowledge Distillation for Language Models*, arXiv:2601.15394v2
- 인용된 선행 연구는 논문 내 참고문헌 섹션(p.13-16)을 기준으로 함
- 본 논문의 참고문헌에 없는 외부 연구는 정확성 불확실성으로 인해 포함하지 않음
