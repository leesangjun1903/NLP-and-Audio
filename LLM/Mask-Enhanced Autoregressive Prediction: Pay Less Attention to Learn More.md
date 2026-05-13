# Mask-Enhanced Autoregressive Prediction (MEAP)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

MEAP은 **대규모 언어 모델(LLM)의 핵심 정보 검색 능력 부족 문제**를 해결하기 위해, Masked Language Modeling(MLM)을 Next-Token Prediction(NTP)에 **구조적 변경 없이** 통합한 새로운 학습 패러다임이다. 핵심 직관은 다음과 같다:

> *"더 적은 토큰에 주의를 기울임으로써 더 많은 것을 학습한다 (Pay Less Attention to Learn More)"*

### 주요 기여 (Contributions)

| 기여 항목 | 내용 |
|-----------|------|
| **새로운 훈련 패러다임** | MLM을 decoder-only Transformer의 NTP에 seamlessly 통합 |
| **추가 연산 비용 없음** | 사전학습 및 추론 시 추가 overhead 없음 |
| **데이터 효율성** | 60B 토큰으로 NTP의 200B 토큰 수준 달성 |
| **범용성** | 사전학습 및 파인튜닝 모두에 적용 가능 |
| **기계적 해석** | 어텐션 식별성(attention distinguishability) 향상 메커니즘 규명 |

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

NTP 기반 LLM은 다음 두 가지 한계를 가진다:

1. **정보 검색 능력 부족**: 긴 컨텍스트에서 핵심 정보를 정확히 찾아내지 못함 (Liu et al., 2024b; Kamradt, 2023)
2. **어텐션 식별성 저하**: Softmax 정규화로 인해 긴 시퀀스에서 토큰 간 어텐션 점수 차이가 작아짐 → **Attention Sink 현상** (Xiao et al., 2023)

MLM(BERT류)은 이 문제를 해결하나, 텍스트 생성에 취약하고 encoder-decoder 구조가 필요하다.

기존의 통합 방법(UniLM, UL2)은 복잡한 다중 목적함수와 특수 아키텍처를 요구해 대규모 적용이 어렵다.

---

### 2.2 제안 방법 (수식 포함)

#### 사전학습 (Pre-training)

입력 시퀀스 $X = (x_1, x_2, \ldots, x_{n-1}, x_n)$에 대해, 비율 $P$만큼 토큰을 랜덤 마스킹하여 다음을 얻는다:

$$X' = (x_1, [\text{mask}], \ldots, x_{t-1}, x_t)$$

이후 decoder-only Transformer $\theta$로 표준 NTP를 수행한다:

$$p_\theta(X') = \prod_{t=1}^{T} p_\theta(x_t \mid x_1, [\text{mask}], \ldots, x_{t-1})$$

- 마스킹 비율: $P = 0.15$ (사전학습)
- 마스킹된 위치에서도 **causal attention**을 그대로 유지 (양방향 어텐션 불필요)

#### 파인튜닝 (Fine-tuning)

훈련 샘플을 복제(duplicate)하고, 복제본에만 마스킹을 적용한다. 손실 함수는:

$$\mathcal{L}(\theta) = -\sum_{t \in U_q \cup U_m} \log p_\theta(x_t \mid x_1, \ldots, x_{t-1}; \hat{x}_1, [\text{mask}], \ldots, \hat{x}_{t-1})$$

여기서:
- $\{\hat{x}_i\}$는 원본 시퀀스 $\{x_i\}$의 복제본 ($\hat{x}_i = x_i$)
- $U_q$: 정답(answer) 토큰 집합
- $U_m$: 마스킹된 토큰 집합
- 마스킹 비율: $P = 0.10$ (파인튜닝)
- 답변 길이가 50 미만인 경우 표준 NTP 사용

---

### 2.3 모델 구조

MEAP은 **아키텍처를 변경하지 않는다**. 기반 모델은 LLaMa 스타일의 decoder-only Transformer이며 주요 구성 요소는 다음과 같다:

| 구성 요소 | 세부 내용 |
|-----------|-----------|
| 아키텍처 | Decoder-only Transformer (LLaMa 기반) |
| 모델 크기 | 1.1B (24 layers, 32 heads, hidden=2048) |
| 위치 임베딩 | Rotary Positional Embedding (RoPE), $\theta_{base}=640{,}000$ |
| 정규화 | Pre-Norm + RMSNorm |
| 활성화 함수 | SwiGLU |
| 어텐션 | Grouped-Query Attention (GQA) |
| 컨텍스트 길이 | 4096 (64K로 확장 가능) |
| 옵티마이저 | AdamW ($\beta_1=0.9$, $\beta_2=0.95$) |
| 최대 학습률 | $4 \times 10^{-4}$ |
| 최소 학습률 | $4 \times 10^{-5}$ |
| 가중치 감쇠 | $5 \times 10^{-2}$ |

---

### 2.4 성능 향상

#### (1) Needle-in-a-Haystack (32K context)

| 훈련 토큰 | NTP | MEAP |
|-----------|-----|------|
| 40B | 65.9% | **80.2%** |
| 60B | 52.8% | **85.8%** |
| 200B | 87.1% | **98.2%** |

→ MEAP은 **60B 토큰**으로 NTP의 **200B 수준** 달성 (데이터 효율 3배 이상)

#### (2) Multi-Document QA (사전학습, 상대적 정확도 향상)

| 문서 수 | Position 10 | Position 20 |
|---------|-------------|-------------|
| 10 docs | +30.6% | – |
| 20 docs | +5.1% | +27.2% |

#### (3) 장문 추론 (M-RS Task)
- 8K~32K 모든 컨텍스트 길이에서 평균 **+6.6 percentage point** 향상

#### (4) 파인튜닝 (Llama-3-8B, Alpaca)

| 태스크 | NTP | MEAP |
|--------|-----|------|
| 상식 추론 평균 | 67.30 | **68.42** (+1.12) |
| MDQA 평균 (20문서) | 25.56 | **37.33** (+11.77%) |

#### (5) 환각(Hallucination) 감소

| 데이터셋 | NTP (GPT-4o judge) | MEAP (GPT-4o judge) |
|----------|---------------------|----------------------|
| XSum | 0.14 | **0.16** |
| MultiNews | 0.10 | **0.13** |
| WikiSum | 0.19 | **0.24** |

---

### 2.5 한계 (Limitations)

1. **파인튜닝 시 시퀀스 길이 2배 증가**: 복제 전략으로 인한 메모리 및 초기 훈련 시간 증가 (단, 에포크 수 절반으로 보정됨)
2. **소규모 실험 범위**: 1.1B 파라미터 모델 중심의 사전학습 실험 (수십억~수조 파라미터 스케일 검증 미흡)
3. **마스킹 전략의 민감성**: 랜덤 마스킹이 가장 효과적이나 최적 마스킹 비율이 태스크에 따라 다름 ($P=0.15$ for pre-training, $P=0.10$ for fine-tuning)
4. **텍스트 생성 품질 검증 미흡**: 생성 능력에 대한 체계적 평가 없음
5. **이론적 근거 부족**: 어텐션 식별성 향상의 수학적 증명 없이 실험적 검증에만 의존

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 어텐션 식별성 향상 메커니즘

MEAP의 일반화 성능 향상의 핵심 원리는 **어텐션 분포의 식별성(distinguishability) 향상**이다.

#### 어텐션 점수 감소 (Attention Score Decay)

$$\text{Score Decay} = \frac{Att(X_N[\text{mask}=1]) - Att(X_M[\text{mask}=1])}{Att(X_N[\text{mask}=1])}$$

#### 어텐션 분산 증가 (Attention Variance Increase)

$$\text{Var Increase} = Var(Att(X_M[\text{mask}=0])) - Var(Att(X_N[\text{mask}=0]))$$

실험 결과 (시퀀스 길이 4096):
- 마스킹 위치 어텐션 점수: **53.34% 감소** (p < 1e-6)
- 비마스킹 위치 어텐션 분산: **7.80% 증가** (p < 1e-6)

### 3.2 태스크 관련 토큰에 대한 집중도 향상

Figure 6 및 Tables 16-18에서 확인된 바에 따르면:

| 입력 세그먼트 | NTP 어텐션 | MEAP 어텐션 |
|--------------|------------|-------------|
| Context Before | 73.1% | 49.1% |
| **Answer** | **9.4%** | **34.5%** |
| Context After | ~6% | ~7% |
| Query | ~6% | ~7% |

→ MEAP은 핵심 답변 토큰에 대한 어텐션을 **약 3.67배** 증가시킴

### 3.3 Cross-Model 일반화 (다양한 아키텍처)

다중 아키텍처에서 MDQA 성능 (20문서):

| 모델 | NTP 평균 | MEAP 평균 | 향상 |
|------|---------|-----------|------|
| Llama-3.2-3B | 13.05% | 21.96% | **+8.91%p** |
| Mistral-7B-0.2 | 32.33% | 35.60% | **+3.27%p** |
| Qwen2.5-14B | 57.49% | 59.11% | **+1.62%p** |

→ 모델 크기와 아키텍처에 무관하게 일관된 향상 확인

### 3.4 일반화 성능 향상의 이론적 근거

MEAP이 일반화를 향상시키는 이유를 다음과 같이 해석할 수 있다:

**① 정규화 효과로서의 마스킹**
- 랜덤 마스킹은 드롭아웃과 유사하게 모델이 특정 토큰에 과의존하지 않도록 강제
- 다양한 컨텍스트 패턴에 대한 견고성(robustness) 향상

**② 희소 어텐션과의 유사성**
- Martins et al. (2020)의 sparse attention과 유사한 효과
- 불필요한 토큰에 대한 어텐션을 묵시적으로 억제

**③ Softmax 포화 문제 완화**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

긴 시퀀스에서 Softmax는 모든 토큰에 유사한 낮은 점수를 배분하는 경향이 있다. MEAP은 마스킹된 토큰들이 negligible attention을 받게 함으로써, 나머지 토큰들 간의 점수 차이를 증폭시킨다.

**④ Lost-in-the-Middle 문제 해결**
- 중간 위치 정보 손실 문제에서 MEAP이 최대 +15.22%p 향상 (Position 20)
- 이는 MEAP이 위치에 무관한 일반화 능력을 학습함을 시사

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 비교 연구 표

| 연구 | 연도 | 방법 | 장점 | MEAP 대비 한계 |
|------|------|------|------|----------------|
| **BERT** (Devlin, 2018) | 2018 | Bidirectional MLM | 정보 검색 우수 | 텍스트 생성 불가, encoder-only |
| **UniLM** (Dong et al., 2019) | 2019 | 다중 목적함수 통합 | 이해+생성 동시 | 복잡한 파이프라인, 확장성 저하 |
| **XLNet** (Yang, 2019) | 2019 | Permutation LM | 의존성 모델링 강화 | 복잡한 훈련, decoder-only 미지원 |
| **GPT-3** (Brown, 2020) | 2020 | NTP (few-shot) | 범용성, 확장성 | 정보 검색 취약 |
| **T5** (Roberts et al., 2019) | 2019 | Text-to-Text, MLM 확장 | 분류 태스크 우수 | Encoder-decoder 필수 |
| **UL2** (Tay et al., 2022) | 2022 | Mixture-of-Denoisers | 다양한 패러다임 통합 | 복잡한 목적함수 전환 |
| **LLaMa3** (Dubey et al., 2024) | 2024 | NTP 기반 | SOTA 생성 능력 | 정보 검색 한계 |
| **Differential Transformer** (Ye et al., 2024) | 2024 | 차분 어텐션 메커니즘 | 노이즈 제거 | 아키텍처 변경 필요 |
| **MEAP** (본 논문) | 2025 | MLM+NTP 통합 | 추가 비용 없음, 범용 | 파인튜닝 시 시퀀스 2배 |

### 4.2 핵심 차별점

```
NTP (GPT 계열):    좌→우 단방향, 정보 검색 약점
MLM (BERT 계열):   양방향, 생성 불가
UniLM/UL2:         다중 목적함수, 복잡한 전환
MEAP:              단방향 유지 + 마스킹으로 MLM 효과 달성
                   → 추가 비용 없이 두 패러다임의 장점 결합
```

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

#### (1) 훈련 패러다임 재정의
MEAP은 NTP와 MLM의 이분법을 깨고, **단일 decoder-only 구조 내에서 두 목적의 장점을 결합**할 수 있음을 보였다. 이는 향후 LLM 사전학습 방법론 연구의 새로운 방향을 제시한다.

#### (2) 데이터 효율성 연구 촉진
60B 토큰으로 200B 수준 달성이라는 결과는 **데이터 효율적 훈련(data-efficient training)** 연구에 중요한 영감을 제공한다.

#### (3) 어텐션 메커니즘 연구
마스킹이 어텐션 분포를 개선한다는 발견은 **희소 어텐션(sparse attention)**, **차분 어텐션(differential transformer)** 등과 연계된 후속 연구를 촉진할 것이다.

#### (4) RAG/장문 이해 시스템 개선
Lost-in-the-Middle 문제 해결은 **Retrieval-Augmented Generation(RAG)** 및 장문 문서 처리 시스템 개발에 직접적 영향을 미친다.

---

### 5.2 향후 연구 시 고려할 점

#### (1) 스케일링 검증
현재 실험은 1.1B 모델에 집중되어 있다. **7B, 70B, 수백B 파라미터 스케일**에서의 검증이 필요하다. 특히 다음 질문이 중요하다:
- 모델이 커질수록 마스킹 효과가 유지되는가?
- 최적 마스킹 비율 $P$가 모델 크기에 따라 변하는가?

#### (2) 마스킹 전략 최적화
현재 랜덤 마스킹이 최선이나, 다음을 탐구할 필요가 있다:

$$P_{\text{optimal}} = f(\text{task}, \text{model size}, \text{context length})$$

- **태스크 적응적 마스킹**: 다운스트림 태스크에 맞춘 마스킹 패턴 설계
- **중요도 기반 마스킹**: 정보 이론적 기준(엔트로피 등)으로 마스킹 위치 결정
- **동적 마스킹 비율**: 훈련 단계에 따른 $P$ 스케줄링

#### (3) 멀티모달 확장
MEAP의 원리는 텍스트에 국한되지 않는다. **이미지-텍스트 멀티모달 모델**에서의 적용 가능성을 탐구해야 한다.

#### (4) 이론적 기반 강화
현재 어텐션 분산 증가에 대한 실험적 검증만 존재한다. 다음의 이론적 분석이 필요하다:

$$\mathbb{E}\left[Var\left(Att(X_M)\right)\right] > \mathbb{E}\left[Var\left(Att(X_N)\right)\right]$$

이를 수학적으로 증명하고, 최적 마스킹 비율을 정보 이론적으로 도출하는 연구가 요청된다.

#### (5) RLHF/DPO와의 통합
MEAP의 파인튜닝 효과가 **Reinforcement Learning from Human Feedback(RLHF)** 또는 **Direct Preference Optimization(DPO)**와 결합될 때 시너지 효과가 있는지 검증이 필요하다.

#### (6) 추론 시간 개입
현재 MEAP은 훈련 시간에만 마스킹을 적용한다. **추론 시 동적 마스킹**을 통한 Chain-of-Thought 강화 가능성도 고려할 수 있다.

#### (7) 다국어 일반화
영어 중심 벤치마크만 검증되어 있어, **다국어 환경**에서의 MEAP 적용 효과를 분석해야 한다.

---

## 참고 자료

### 직접 참고한 자료
- **주 논문**: Zhuang, X., Jia, Z., Li, J., Zhang, Z., Shen, L., Cao, Z., & Liu, S. (2025). *Mask-Enhanced Autoregressive Prediction: Pay Less Attention to Learn More*. ICML 2025. arXiv:2502.07490v3
- **코드 저장소**: https://github.com/CharlieZhuang-Code/MEAP

### 논문 내 인용된 주요 참고 연구
- Devlin, J. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805
- Radford, A. (2018). *Improving Language Understanding by Generative Pre-training*.
- Brown, T. B. (2020). *Language Models are Few-Shot Learners*. arXiv:2005.14165
- Dong, L. et al. (2019). *Unified Language Model Pre-training (UniLM)*. NeurIPS 32.
- Tay, Y. et al. (2022). *UL2: Unifying Language Learning Paradigms*. arXiv:2205.05131
- Yang, Z. (2019). *XLNet: Generalized Autoregressive Pretraining*. arXiv:1906.08237
- Liu, N. F. et al. (2024b). *Lost in the Middle: How Language Models Use Long Contexts*. TACL 12.
- Xiao, G. et al. (2023). *Efficient Streaming Language Models with Attention Sinks*. arXiv:2309.17453
- Martins, A. et al. (2020). *Sparse and Continuous Attention Mechanisms*. NeurIPS 33.
- Ye, T. et al. (2024). *Differential Transformer*. arXiv:2410.05258
- Kamradt, G. (2023). *Needle in a Haystack - Pressure Testing LLMs*. Github Repository.
- Dubey, A. et al. (2024). *The LLaMA 3 Herd of Models*. arXiv:2407.21783
- Liu, A. et al. (2024a). *DeepSeek-V3 Technical Report*. arXiv:2412.19437
