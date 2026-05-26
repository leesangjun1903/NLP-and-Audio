# SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

SlimQwen은 대규모 MoE(Mixture-of-Experts) 모델의 **사전학습 단계**에서 구조적 프루닝(Structured Pruning)과 지식 증류(Knowledge Distillation, KD)를 체계적으로 결합하면, 동일한 학습 예산 하에서 처음부터 학습하는 것보다 훨씬 우수한 성능을 달성할 수 있다는 것을 실증적으로 증명합니다. 구체적으로 Qwen3-Next-80A3B 모델을 약 **4배 압축(23A2B)**하면서도 경쟁력 있는 하위 태스크 성능을 유지합니다.

### 3가지 핵심 연구 질문

1. **초기화(Initialization)**: 사전학습된 MoE를 프루닝한 초기화가 랜덤 초기화보다 유리한가?
2. **압축 전략(Compression Strategy)**: 다양한 전문가 압축 방법이 대규모 지속 사전학습 후 최종 성능에 어떤 영향을 미치는가?
3. **학습 레시피(Training Recipe)**: 최적의 압축 후 학습 전략은 무엇인가?

### 주요 기여

| 기여 | 내용 |
|------|------|
| **체계적 연구** | 깊이·너비·전문가 축에 걸친 대규모 MoE 압축 연구 |
| **부분 보존 전문가 병합 전략** | 새로운 partial-preservation expert merging 제안 |
| **MTP 지식 증류** | Multi-Token Prediction KD 목적함수 제안 |
| **점진적 프루닝 스케줄** | Progressive pruning이 one-shot 압축보다 일관되게 우수함을 입증 |
| **실증적 결과** | Qwen3-Next-80A3B → SlimQwen-23A2B (4× 압축, 경쟁력 있는 성능 유지) |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

현대 MoE LLM은 사전학습 및 서빙 비용이 매우 높습니다. 기존 연구들은 주로 **원샷(one-shot) 성능**에서의 전문가 압축 방법을 비교했지만, **대규모 지속 사전학습 후**의 효과는 미탐구 상태였습니다. 또한:

- 밀집(dense) 모델 압축 기법(Minitron 등)을 MoE에 직접 적용하기 어려움
- MoE는 전문가(expert) 차원이라는 추가적인 압축 차원이 존재
- KD 단독이 LM 손실 결합보다 항상 우수한지 불명확

---

### 2.2 제안 방법 (수식 포함)

#### (1) 모델 기본 구조: MoE 모듈

각 입력 토큰 $x \in \mathbb{R}^{1 \times d}$에 대해 SwiGLU MLP 전문가:

$$\text{Expert}(x) = (\text{SiLU}(xW_{1e}) \odot (xW_{2e}))W_{3e} \tag{1}$$

where $W_{1e}, W_{2e} \in \mathbb{R}^{d \times d_{ff}}$, $W_{3e} \in \mathbb{R}^{d_{ff} \times d}$.

라우터의 top-k 게이팅: $z(x) = \text{softmax}\_{\text{TopK}}(xW^G, k)$, $W^G \in \mathbb{R}^{d \times n_{\text{routed}}}$

공유 전문가 게이트: $z_s(x) = \sigma(xw_{\text{sh}}) \in \mathbb{R}^{n_{\text{shared}}}$

최종 MoE 출력:

$$\text{MoE}(x) = \sum_{e=1}^{n_{\text{routed}}} z_e(x)\text{Expert}_e(x) + \sum_{s=1}^{n_{\text{shared}}} z_s(x)\text{Expert}_s(x) \tag{2}$$

RMSNorm:

$$\text{RMSNorm}(X) = \frac{X}{\text{RMS}(X)} \odot \gamma, \quad \text{RMS}(X)_i = \sqrt{\frac{1}{d}\sum_{j=1}^{d} X_{ij}^2 + \epsilon} \tag{3}$$

---

#### (2) 깊이 프루닝 (Depth Pruning)

$L$개 레이어 중 마지막 $N$개 레이어를 제거:

$$\mathcal{L}_{\text{keep}} = \{1, \ldots, L-N\}, \quad \tilde{L} = L - N \tag{4}$$

> 실험에서는 마지막 25% 레이어를 제거 (48 → 36 블록)

---

#### (3) 너비 프루닝 (Width Pruning)

모듈 출력 활성화 $Z \in \mathbb{R}^{B \times n \times m}$에 대해 평균 절대 활성화:

$$\text{Mean}(Z) := \frac{1}{Bn}\sum_{b=1}^{B}\sum_{t=1}^{n}|Z_{b,t,:}| \in \mathbb{R}^m$$

RMSNorm 출력을 기준으로 한 은닉 차원 중요도:

$$I_{\text{norm}}^{(k)} = \left[\frac{\sum_{i=0}^{L}\text{Mean}(\text{RMSNorm}(X))_{\big|_{L}}}{L}\right]_k, \quad k=1,\ldots,d \tag{5}$$

목표 은닉 크기 $d_t$에서 가장 높은 중요도 점수를 가진 $d_t$개 차원을 유지.

---

#### (4) 전문가 중요도 추정 (Expert Importance Estimation)

**빈도 기반(Frequency-based)**:

$$I_i^{\text{Freq}} = \mathbb{E}_{x \sim \mathcal{C}}\left[\mathbb{I}[i \in \mathcal{A}(x)]\right] \tag{6a}$$

**소프트 로짓(Soft-logits)**:

$$I_i^{\text{Soft}} = \mathbb{E}_{x \sim \mathcal{C}}\left[\frac{\mathbb{I}[i \in \mathcal{A}(x)] \cdot z_i(x)}{\sum_{j \in \mathcal{A}(x)} z_j(x)}\right] \tag{6b}$$

**REAP (Router-weighted Expert Activation)**:

$$I_i^{\text{REAP}} = \frac{1}{|\mathcal{X}_i|}\sum_{x \in \mathcal{X}_i} z_i(x)\|E_i(x)\|_2, \quad i = 1, \ldots, N \tag{7}$$

---

#### (5) 부분 보존 전문가 병합 전략 (Partial-Preservation Expert Merging)

**핵심 아이디어**: 목표 전문가 수 $\tilde{N}$에서 절반($\lfloor \tilde{N}/2 \rfloor$)은 원본 그대로 보존하고, 나머지 절반은 병합으로 구성.

- 보존할 전문가 집합: $\mathcal{S}\_{\text{keep}} = \arg\text{topk}\_{i \in \{1,\ldots,N\}} I_i$, $|\mathcal{S}_{\text{keep}}| = \lfloor \tilde{N}/2 \rfloor$
- 제거 전문가: $\mathcal{S}\_{\text{prune}} = \{1,\ldots,N\} \setminus \mathcal{S}_{\text{keep}}$
- 병합 베이스: $\mathcal{S}_{\text{base}}$ (남은 전문가 중 $\tilde{N}/2$개 선택)

병합된 전문가:

$$\tilde{E}_i = \frac{I_i}{I_i + I_{m(i)}} E_i + \frac{I_{m(i)}}{I_i + I_{m(i)}} E_{m(i)} \tag{8}$$

where $m(i) = \arg\max_{j \in \mathcal{S}_{\text{merge}}} \text{CosineSim}(i, j)$.

---

#### (6) MTP 지식 증류 (Multi-Token Prediction KD)

깊이 $k$에서 $i$번째 토큰의 표현 결합:

$$h_i^{\prime k} = M_k \left[\text{RMSNorm}(h_i^{k-1}); \text{RMSNorm}(\text{Emb}(t_{i+k}))\right] \tag{9}$$

MTP 언어 모델링 손실:

$$\mathcal{L}_{\text{MTP-LM}} = \frac{1}{D}\sum_{k=1}^{D}\left(-\frac{1}{T-k}\sum_{i=1}^{T-k}\log p_{i+k}^k[t_{i+k}]\right) \tag{10}$$

MTP 지식 증류 손실 (교사 분포 $q_{i+k}$와 KL 발산 최소화):

$$\mathcal{L}_{\text{MTP-KD}} = -\frac{1}{D}\sum_{k=1}^{D}\left(\frac{1}{T-k}\sum_{i=1}^{T-k}\sum_{v=1}^{V} q_{i+k}[v]\log p_{i+k}^k[v]\right) \tag{11}$$

**최종 통합 목적 함수**:

$$\mathcal{L} = (1-\lambda)\mathcal{L}_{\text{LM}} + \lambda\mathcal{L}_{\text{KD}} + \beta\left((1-\lambda)\mathcal{L}_{\text{MTP-LM}} + \lambda\mathcal{L}_{\text{MTP-KD}}\right) \tag{12}$$

- $\lambda$: KD 손실 가중치 (1.0→0.75 선형 감소)
- $\beta$: MTP 손실 가중치 (0.3→0.1 코사인 감소)

---

### 2.3 모델 구조

**Qwen3-Next-80A3B (교사 모델)**:
- 48 트랜스포머 블록 (12 Full Attention + 36 Linear Attention)
- 은닉 차원: 2048
- MoE: 레이어당 512 전문가 (top-10 라우팅)
- 총 파라미터: 80B, 활성화 파라미터: ~3.8B

**SlimQwen-23A2B (학생 모델)**:
- 36 트랜스포머 블록 (8 Full Attention + 24 Linear Attention)
- 은닉 차원: 1536 (너비 프루닝)
- MoE: 레이어당 256 전문가 (top-8 라우팅)
- 총 파라미터: 23B, 활성화 파라미터: ~2.0B

```
[아키텍처 요약]
80A3B → 깊이 25% 제거 → 너비 2048→1536 → 전문가 512→256
= 23A2B (약 4× 압축)
```

---

### 2.4 성능 향상

**Q1: 프루닝 초기화 vs 랜덤 초기화 (120B 토큰 학습)**

| 방법 | MMLU | BBH | GSM-8K | Avg. |
|------|------|-----|--------|------|
| Random Init + KD | 65.06 | 56.01 | 73.35 | 61.66 |
| Pruned + LM Loss | 72.76 | 64.94 | 81.84 | 69.96 |
| **Pruned + KD** | **75.67** | **72.29** | **83.17** | **73.45** |
| 교사(80A3B) | 85.22 | 85.12 | 90.07 | 82.68 |

→ 프루닝 초기화 + KD가 랜덤 초기화 대비 **+11.79점** 향상, 교사 성능의 **86.5% 회복**

**Q2: 학습 손실 비교 (23A2B, 120B 토큰)**

| 방법 | MMLU | MMLU-Pro | EvalPlus |
|------|------|----------|---------|
| NTP KD | 74.16 | 50.97 | 67.32 |
| NTP KD + LM Loss | 74.93 | 51.44 | 66.07 |
| NTP KD + MTP KD | 75.13 | 51.94 | **69.32** |
| **Full Objective** | **75.67** | 51.19 | 69.30 |

**Q3: 점진적 프루닝 vs 원샷 프루닝 (400B 토큰)**

| 방법 | MMLU | MMLU-Redux | GSM-8K |
|------|------|-----------|--------|
| One-stage | 75.86 | 75.41 | 85.22 |
| Joint Progressive | 76.30 | 76.93 | 86.05 |
| Width-first | 77.14 | 77.07 | 84.00 |
| **Depth-first (SlimQwen)** | **77.39** | **78.01** | **85.82** |

**효율성**: SlimQwen-23A2B는 vLLM 기준 디코딩 처리량 **142.58 → 210.87 Tok/s** (약 1.48×), 피크 메모리 **156.56 → 43.30 GB** (약 3.6×) 절감

---

### 2.5 한계

1. **절반 보존 비율의 최적성 미검증**: 부분 보존 전략에서 "절반"이라는 비율은 경험적 선택으로, 다양한 압축 비율이나 모델 구조에서의 최적성은 미검증
2. **전문가 압축 방법의 수렴성**: 다양한 원샷 전문가 압축 방법들이 대규모 학습 후 유사한 성능으로 수렴하는 이유에 대한 이론적 설명 부족
3. **아키텍처 특정성**: Qwen3-Next 하이브리드 어텐션(Gated DeltaNet + Gated Attention) 구조에 특화되어 있어 다른 MoE 아키텍처로의 일반화 검증 필요
4. **점진적 단계 수의 한계**: 3단계 이상의 세밀한 스케줄이 추가 이득을 제공하지 않아, 최적 스케줄 설계 원칙이 명확하지 않음
5. **교사 모델 의존성**: KD 방식이 교사 모델의 온라인 추론을 요구하므로, 교사 모델 자체의 메모리/연산 비용이 필요

---

## 3. 일반화 성능 향상 가능성

### 3.1 다양한 벤치마크에서의 일관된 성능

SlimQwen의 일반화 성능 향상은 여러 메커니즘을 통해 달성됩니다:

**① LM 손실과 KD의 혼합**

순수 KD는 교사의 소프트 타깃에만 의존하므로 교사의 편향을 그대로 흡수할 수 있습니다. 반면 LM 손실을 혼합하면:

$$\mathcal{L} = (1-\lambda)\mathcal{L}_{\text{LM}} + \lambda\mathcal{L}_{\text{KD}} + \cdots$$

실험 결과, LM 손실 추가 시 **지식 집약적 벤치마크**(MMLU: 74.16→74.93, MMLU-Pro: 50.97→51.44)에서 일반화가 향상됩니다. $\lambda$를 1.0→0.75로 선형 감소시켜 초기에는 교사 지식을 충분히 흡수하다가 점차 자기 지도 학습 비중을 높이는 방식이 일반화에 유리합니다.

**② MTP KD를 통한 미래 문맥 예측 능력 향상**

MTP KD는 단일 토큰 예측을 넘어 여러 미래 토큰을 동시에 예측하도록 학습:

$$\mathcal{L}_{\text{MTP-KD}} = -\frac{1}{D}\sum_{k=1}^{D}\left(\frac{1}{T-k}\sum_{i=1}^{T-k}\sum_{v=1}^{V} q_{i+k}[v]\log p_{i+k}^k[v]\right)$$

이는 모델이 **장거리 의존성(long-range dependency)**을 더 잘 학습하게 하며, 표현의 품질을 높여 다양한 도메인에서의 일반화에 기여합니다. 투기적 디코딩(speculative decoding)의 수락률 향상도 이를 간접적으로 입증합니다.

**③ 부분 보존 전문가 병합의 다양성 보존**

완전 병합(full merging)은 **표현 동질화(representation homogenization)** 문제를 야기합니다. 부분 보존 전략은:

- 상위 $\lfloor \tilde{N}/2 \rfloor$개 전문가를 원본 가중치로 보존 → **특화된 지식 유지**
- 나머지를 병합 → **보완적 지식 통합**

이 전략은 모델이 다양한 입력 패턴에 대해 서로 다른 전문가를 활성화할 수 있게 하여, MMLU, GSM-8K, CMMLU 등 이질적인 벤치마크에서 **일관된 성능 향상**을 보입니다.

**④ 점진적 프루닝의 부드러운 최적화 궤적**

원샷 압축은 갑작스러운 구조 변화로 인해 최적화 공간에서 큰 도약이 발생합니다. 점진적 프루닝(예: Depth-first: 40B + 360B)은:

```
Stage 1: 깊이 절반 감소 + 40B 토큰 학습 (중간 구조 적응)
Stage 2: 너비/나머지 깊이 감소 + 360B 토큰 학습 (최종 구조 정착)
```

이는 **부드러운 최적화 궤적**을 제공하여 모델이 압축 과정에서 일반화 능력을 더 잘 보존합니다. MMLU가 75.86(원샷)→77.39(깊이우선), MMLU-Redux가 75.41→78.01로 향상된 것이 이를 뒷받침합니다.

**⑤ 다국어 및 다양한 도메인 일반화**

Table 10의 확장 벤치마크 결과에서 Pruned + KD가 랜덤 초기화 대비 모든 도메인에서 우수:

| 벤치마크 | Random Init | Pruned + KD | 향상 |
|---------|------------|------------|------|
| MBPP (코딩) | 57.00 | 67.40 | +10.4 |
| MMMLU (다국어) | 50.90 | 66.40 | +15.5 |
| Mgsm (다국어 수학) | 39.98 | 64.47 | +24.5 |
| KOR-Bench (추론) | 33.36 | 40.80 | +7.4 |

특히 **다국어 수학(Mgsm)**에서의 큰 향상은 프루닝 초기화가 교차 언어적 추론 능력을 보존함을 시사합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

**① MoE 압축의 표준 프레임워크 제시**

SlimQwen은 MoE 모델 압축의 세 축(깊이/너비/전문가)을 통합적으로 다루는 최초의 대규모 연구 중 하나로, 향후 MoE 압축 연구의 기준점(baseline)을 제공합니다. 특히:

- **프루닝 초기화의 우월성 실증**: 대규모 학습에서도 프루닝 기반 초기화가 항상 유리하다는 것이 확립되어, 미래 연구에서 랜덤 초기화를 기본으로 사용하는 관행에 의문을 제기합니다.
- **원샷 전문가 방법의 수렴성**: 다양한 전문가 압축 방법들이 충분한 학습 토큰 후 유사한 성능으로 수렴한다는 발견은, 복잡한 전문가 선택 알고리즘보다 **학습 전략**에 더 집중해야 함을 시사합니다.

**② MTP KD의 적용 확장 가능성**

MTP 지식 증류 개념은 MoE 압축에만 국한되지 않습니다. Dense 모델 압축, 연속 학습(continual learning), 모델 병합(model merging) 등 다양한 설정에서 활용 가능하며, 투기적 디코딩 성능 향상에도 직접적으로 기여합니다.

**③ 사전학습 규모 압축 연구 촉진**

기존 연구들이 주로 파인튜닝 후 압축이나 소규모 실험에 집중했다면, SlimQwen은 **400B 토큰 규모**의 지속 사전학습에서 검증하여, 산업 규모에서의 실용적 가이드라인을 제공합니다.

**④ 하이브리드 어텐션 MoE 압축 연구**

Gated DeltaNet(선형 어텐션) + Gated Attention(전체 어텐션) 하이브리드 구조에서의 압축 방법론은, 향후 선형 어텐션 기반 고효율 모델들의 압축 연구에 기초가 됩니다.

---

### 4.2 앞으로 연구 시 고려할 점

**① 최적 보존 비율의 이론적 근거 탐구**

현재 "절반 보존"은 경험적 선택입니다. 향후 연구에서는:
- 모델 크기, 압축 비율, 전문가 특화 수준에 따른 최적 보존 비율 탐구
- 정보이론적 관점(예: 전문가 간 상호정보량 기반)에서의 이론적 근거 마련

**② 다양한 MoE 아키텍처로의 일반화 검증**

SlimQwen은 Qwen3-Next 특화 결과이므로, Mixtral, DeepSeek-MoE, LLaMA-MoE 등 다양한 MoE 구조에서 검증이 필요합니다.

**③ 압축과 파인튜닝의 통합적 고려**

압축된 모델의 RLHF/SFT 후 성능 변화, 즉 **정렬(alignment) 능력 보존** 여부를 체계적으로 연구해야 합니다.

**④ 적응적/동적 $\lambda$ 스케줄링**

현재는 선형/코사인 감소를 사용하지만, 학습 중 검증 손실이나 벤치마크 성능에 따른 **적응적 $\lambda$ 조정** 전략이 더 최적일 수 있습니다.

**⑤ 구조적 탐색(NAS)과의 결합**

현재는 수동으로 압축 목표(깊이 25% 제거, 너비 2048→1536)를 설정하지만, 신경망 구조 탐색(NAS)과 결합하면 주어진 연산 예산 내 최적 구조를 자동 탐색할 수 있습니다.

**⑥ 연속 학습 시나리오 적용**

사전학습 후 지속적으로 새로운 데이터로 업데이트되는 **평생 학습(lifelong learning)** 시나리오에서 압축된 MoE 모델의 가소성(plasticity)과 안정성(stability) 균형 연구가 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법 | 대상 | 주요 특징 | SlimQwen과 차이 |
|------|------|------|------|---------|--------------|
| **ShearedLLaMA** (Xia et al.) | 2024 | 구조적 프루닝 | Dense LLM | 동적 배치 로딩, 너비+깊이 동시 프루닝 | MoE 미지원, 소규모 실험 |
| **SliceGPT** (Ashkboos et al.) | 2024 | PCA 기반 행/열 삭제 | Dense LLM | 특수 하드웨어 불필요 | MoE 미지원, 대규모 사전학습 미검증 |
| **Minitron** (Muralidharan et al.) | 2024 | 프루닝 + KD | Dense LLM | 활성화 기반 중요도, 고품질 KD | **MoE 미지원** |
| **ShortGPT** (Men et al.) | 2024 | 레이어 중요도 기반 깊이 프루닝 | Dense LLM | 간단하지만 효과적 | MoE 미지원, 단순 깊이 프루닝만 |
| **M-SMoE** (Li et al.) | 2024 | 전문가 병합 | MoE | 라우팅 정책 힌트 활용 | 대규모 사전학습 후 검증 없음 |
| **REAP** (Lasby et al.) | 2025 | 라우터 가중 전문가 활성화 | MoE | 원샷 성능에서 프루닝 우위 입증 | **원샷만 검증, 대규모 학습 후 미검증** |
| **DarwinLM** (Tang et al.) | 2025 | 진화적 구조 프루닝 | Dense LLM | 진화 알고리즘 기반 구조 탐색 | 전문가 내부 차원만 압축 |
| **SlimMoE** (Li et al.) | 2025 | 전문가 슬리밍 + KD | MoE | 전문가 내부 FFN 차원 압축 | **전문가 수 자체는 미감소** |
| **Peng et al.** | 2024 | 사전학습 증류 설계 공간 탐색 | Dense LLM | 로짓 처리, 손실 선택, 스케일링 법칙 | MoE 미지원 |
| **SlimQwen (본 논문)** | 2026 | 깊이+너비+전문가 통합 프루닝 + MTP KD | **MoE LLM** | **대규모 사전학습 검증, MTP KD 제안** | - |

**핵심 차별점 요약**:
- 기존 연구들은 MoE의 세 압축 축(깊이/너비/전문가)을 통합적으로 다루지 않았음
- **400B 토큰 규모**의 대규모 지속 사전학습 후 검증은 SlimQwen이 최초 수준
- MTP KD라는 새로운 증류 목적함수 제안은 기존 연구에 없는 기여
- 원샷 전문가 방법들의 대규모 학습 후 수렴 현상 발견은 새로운 인사이트

---

## 참고 자료

**논문 원문**
- Tang, S., Wang, Z., Zheng, B., et al. (2026). *SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training*. arXiv:2605.08738v2.

**논문 내 인용 문헌 (직접 참조)**
- Muralidharan, S., et al. (2024). *Compact Language Models via Pruning and Knowledge Distillation* (Minitron). arXiv:2407.14679.
- Xia, M., et al. (2024). *Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning*. arXiv:2310.06694.
- Ashkboos, S., et al. (2024). *SliceGPT: Compress Large Language Models by Deleting Rows and Columns*. arXiv:2401.15024.
- Men, X., et al. (2024). *ShortGPT: Layers in Large Language Models are More Redundant than You Expect*. arXiv:2403.03853.
- Lasby, M., et al. (2025). *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*. arXiv:2510.13999.
- Li, P., et al. (2024). *Merge, then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy* (M-SMoE). arXiv:2310.01334.
- Li, Z., et al. (2025). *SlimMoE: Structured Compression of Large MoE Models via Expert Slimming and Distillation*. arXiv:2506.18349.
- Tang, S., et al. (2025). *DarwinLM: Evolutionary Structured Pruning of Large Language Models*. arXiv:2502.07780.
- Gloeckle, F., et al. (2024). *Better & Faster Large Language Models via Multi-Token Prediction*. ICML 2024.
- Peng, H., et al. (2024). *Pre-training Distillation for Large Language Models: A Design Space Exploration*. arXiv:2410.16215.
- Jaiswal, A., et al. (2025). *Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies*. arXiv:2504.05586.
- Yang, A., et al. (2025). *Qwen3 Technical Report*. arXiv:2505.09388.
- Qwen Team. (2025). *Qwen3-Next: Towards Ultimate Training & Inference Efficiency*.
- Sun, W., et al. (2026). *The Curse of Depth in Large Language Models*. arXiv:2502.05795.
- Cao, M., et al. (2025). *Condense, Don't Just Prune: Enhancing Efficiency and Performance in MoE Layer Pruning*. arXiv:2412.00069.
- Yuan, X., et al. (2026). *Accelerating Compound LLM Training Workloads with Maestro*. arXiv:2605.10501.
