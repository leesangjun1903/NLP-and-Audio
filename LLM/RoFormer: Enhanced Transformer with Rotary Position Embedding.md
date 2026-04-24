# RoFormer: Enhanced Transformer with Rotary Position Embedding

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
RoFormer는 **Rotary Position Embedding(RoPE)** 을 통해 절대 위치 정보를 회전 행렬로 인코딩하면서, **자기-어텐션(self-attention) 내에 상대 위치 의존성을 자연스럽게 통합**할 수 있음을 주장합니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| ① 새로운 위치 인코딩 방법 제안 | 기존 덧셈(additive) 방식이 아닌 **곱셈(multiplicative) 회전 행렬** 기반의 RoPE 제안 |
| ② 이론적 성질 분석 | **장거리 감쇠(long-term decay)** 성질 수학적 증명, 선형 어텐션과의 호환성 제시 |
| ③ 다양한 벤치마크 검증 | 기계 번역, GLUE, 중국어 장문 분류 등 다수 태스크에서 일관된 성능 향상 확인 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Transformer 기반 언어 모델의 **self-attention은 본질적으로 위치 불변(position-agnostic)** 입니다. 따라서 위치 정보를 별도로 주입해야 하는데, 기존 방식들은 다음과 같은 한계를 가집니다:

- **절대 위치 임베딩(Absolute PE):** 문맥 표현에 위치 벡터를 단순히 더하는 방식 → 상대 위치 관계 반영 불가
- **상대 위치 임베딩(Relative PE):** 어텐션 점수 분해식을 수정하는 방식 → **선형 어텐션(linear attention)과 호환 불가**
- 공통 문제: 대부분 **위치 정보를 문맥 표현에 덧셈(additive)으로 추가** → 이론적 해석 어려움

### 2.2 제안하는 방법 (수식 포함)

#### 핵심 아이디어: 상대 위치를 내적(inner product)으로 인코딩

Self-attention에서 query $\boldsymbol{q}_m$과 key $\boldsymbol{k}_n$의 내적이 **오직 상대 위치 $m-n$에만 의존하도록** 설계합니다:

$$\langle f_q(\boldsymbol{x}_m, m),\, f_k(\boldsymbol{x}_n, n) \rangle = g(\boldsymbol{x}_m, \boldsymbol{x}_n, m-n) \tag{1}$$

#### 2D 케이스에서의 해 도출

$d=2$인 경우, 복소수의 기하학적 성질을 활용하면 위 조건을 만족하는 해가 존재합니다:

```math
f_q(\boldsymbol{x}_m, m) = (\boldsymbol{W}_q \boldsymbol{x}_m)\, e^{im\theta}
```

```math
f_k(\boldsymbol{x}_n, n) = (\boldsymbol{W}_k \boldsymbol{x}_n)\, e^{in\theta}
```

```math
g(\boldsymbol{x}_m, \boldsymbol{x}_n, m-n) = \mathrm{Re}\left[(\boldsymbol{W}_q \boldsymbol{x}_m)(\boldsymbol{W}_k \boldsymbol{x}_n)^* e^{i(m-n)\theta}\right]
```

이를 2D 회전 행렬로 표현하면:

```math
f_{\{q,k\}}(\boldsymbol{x}_m, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} W^{(11)}_{\{q,k\}} & W^{(12)}_{\{q,k\}} \\ W^{(21)}_{\{q,k\}} & W^{(22)}_{\{q,k\}} \end{pmatrix} \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \end{pmatrix}
```

#### 일반화: d차원 RoPE

$d$가 짝수일 때, $d/2$개의 2D 부분공간으로 분할하여 일반화합니다:

$$f_{\{q,k\}}(\boldsymbol{x}_m, m) = \boldsymbol{R}^d_{\Theta, m} \boldsymbol{W}_{\{q,k\}} \boldsymbol{x}_m $$

여기서 회전 행렬 $\boldsymbol{R}^d_{\Theta, m}$은 다음과 같습니다:

```math
\boldsymbol{R}^d_{\Theta, m} = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\ \sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\ 0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\ 0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2} \end{pmatrix}
```

사전 정의된 파라미터: $\Theta = \{\theta_i = 10000^{-2(i-1)/d},\; i \in [1, 2, \ldots, d/2]\}$

#### Self-attention에 적용

```math
\boldsymbol{q}_m^\top \boldsymbol{k}_n = (\boldsymbol{R}^d_{\Theta,m} \boldsymbol{W}_q \boldsymbol{x}_m)^\top (\boldsymbol{R}^d_{\Theta,n} \boldsymbol{W}_k \boldsymbol{x}_n) = \boldsymbol{x}_m^\top \boldsymbol{W}_q \boldsymbol{R}^d_{\Theta, n-m} \boldsymbol{W}_k \boldsymbol{x}_n 
```

단, $\boldsymbol{R}^d_{\Theta, n-m} = (\boldsymbol{R}^d_{\Theta,m})^\top \boldsymbol{R}^d_{\Theta,n}$이며, $\boldsymbol{R}^d_\Theta$는 **직교 행렬(orthogonal matrix)** 이므로 위치 인코딩 과정에서 안정성을 보장합니다.

#### 계산 효율적 구현

행렬 $\boldsymbol{R}^d_{\Theta,m}$의 희소성(sparsity)을 활용하여:

```math
\boldsymbol{R}^d_{\Theta,m} \boldsymbol{x} = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \\ x_{d-1} \\ x_d \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \cos m\theta_2 \\ \cos m\theta_2 \\ \vdots \\ \cos m\theta_{d/2} \\ \cos m\theta_{d/2} \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \\ \vdots \\ -x_d \\ x_{d-1} \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \sin m\theta_2 \\ \sin m\theta_2 \\ \vdots \\ \sin m\theta_{d/2} \\ \sin m\theta_{d/2} \end{pmatrix}
```

이 방식은 **전체 회전 행렬 곱셈 없이 element-wise 연산만으로** RoPE를 적용할 수 있어 계산 효율적입니다.

### 2.3 모델 구조

RoFormer의 구조는 기본적으로 Transformer와 동일하지만, **self-attention의 query와 key 생성 단계에서만 RoPE를 적용**합니다:

```
입력 토큰 x_m
      ↓
선형 변환: W_q x_m, W_k x_n, W_v x_n
      ↓
RoPE 적용 (query, key에만):
  q_m = R^d_{Θ,m} W_q x_m
  k_n = R^d_{Θ,n} W_k x_n
      ↓
어텐션 계산: q_m^T k_n = x_m^T W_q R^d_{Θ,n-m} W_k x_n
      ↓
출력: o_m = Σ a_{m,n} v_n
```

**Value에는 RoPE를 적용하지 않습니다** (기존 상대 위치 임베딩 연구들과 같은 방식).

### 2.4 성능 향상

#### 기계 번역 (WMT 2014 En-De)

| 모델 | BLEU |
|------|------|
| Transformer-base (Vaswani et al., 2017) | 27.3 |
| **RoFormer** | **27.5** |

#### GLUE 벤치마크 파인튜닝

| 모델 | MRPC | SST-2 | QNLI | STS-B | QQP | MNLI(m/mm) |
|------|------|-------|------|-------|-----|------------|
| BERT | 88.9 | 93.5 | 90.5 | 85.8 | 71.2 | 84.6/83.4 |
| **RoFormer** | **89.5** | 90.7 | 88.0 | **87.0** | **86.4** | 80.2/79.8 |

- RoFormer는 6개 태스크 중 3개(MRPC, STS-B, QQP)에서 BERT를 유의미하게 상회
- SST-2, QNLI, MNLI에서는 BERT보다 낮은 성능 → 모든 태스크에서 일관된 우위는 아님

#### 중국어 장문 분류 (CAIL2019-SCM)

| 모델 | Validation | Test |
|------|-----------|------|
| BERT-512 | 64.13% | 67.77% |
| WoBERT-512 | 64.07% | 68.10% |
| RoFormer-512 | 64.13% | 68.29% |
| **RoFormer-1024** | **66.07%** | **69.79%** |

- 입력 길이 1024에서 WoBERT 대비 **절대 1.5% 향상** (Test 기준)
- 장문 처리 능력이 RoPE의 강점임을 실험적으로 확인

#### 사전 학습 수렴 속도

- RoFormer는 BERT 대비 **MLM loss에서 더 빠른 수렴** 관찰
- Performer + RoPE 조합도 Performer 단독 대비 더 낮은 LM loss 달성

### 2.5 한계

논문에서 명시한 한계:

1. **수렴 속도 가속의 이론적 설명 부재:** 왜 다른 위치 인코딩 방식보다 더 빠르게 수렴하는지에 대한 수학적 설명이 불충분
2. **장문 성능 우위의 메커니즘 미해명:** 장거리 감쇠 성질은 증명했으나, 이것이 왜 장문 태스크 성능 향상으로 이어지는지 명확하지 않음
3. **하드웨어 자원 의존성:** Transformer 기반 사전 학습이므로 대규모 GPU 자원 필요
4. **GLUE 일부 태스크 성능 저하:** SST-2(−2.8%), QNLI(−2.5%), MNLI(−4.4/3.6%) 등에서 BERT 대비 낮은 성능

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 시퀀스 길이 유연성 (Sequence Length Flexibility)

RoPE는 절대 위치 임베딩과 달리 **사전 정의된 최대 시퀀스 길이가 없습니다.** 회전 행렬은 임의의 위치 $m$에 대해 정의되므로, 학습 시 설정한 최대 길이 이상의 시퀀스에도 이론적으로 적용 가능합니다. 이는 다양한 길이의 입력에 대한 일반화 능력을 높여줍니다.

### 3.2 장거리 감쇠 (Long-term Decay) 성질

$\theta_i = 10000^{-2i/d}$로 설정할 때, Abel 변환을 활용한 수학적 증명:

$$\left|\sum_{i=0}^{d/2-1} \boldsymbol{q}_{[2i:2i+1]} \boldsymbol{k}^*_{[2i:2i+1]} e^{i(m-n)\theta_i}\right| \leq \left(\max_i |h_{i+1} - h_i|\right) \sum_{i=0}^{d/2-1} |S_{i+1}| \tag{10}$$

여기서 $\frac{1}{d/2}\sum_{i=1}^{d/2}|S_i|$는 상대 거리 $m-n$이 증가할수록 감소합니다. 즉, **멀리 떨어진 토큰 쌍의 어텐션 가중치가 자연스럽게 줄어드는 귀납적 편향(inductive bias)** 을 갖습니다.

이 성질은 자연어의 특성 — 인접한 단어일수록 강한 의존성 — 과 일치하며, **사전 학습 데이터 분포 외의 도메인에서도 합리적인 일반화 성능을 기대할 수 있게 합니다.**

### 3.3 선형 어텐션과의 호환성

RoPE는 회전이 벡터의 **노름(norm)을 보존**하므로, 선형 어텐션의 비음수 함수 출력에 회전 행렬을 곱하는 방식으로 결합 가능합니다:

$$\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V})_m = \frac{\sum_{n=1}^N \left(\boldsymbol{R}^d_{\Theta,m} \phi(\boldsymbol{q}_m)\right)^\top \left(\boldsymbol{R}^d_{\Theta,n} \varphi(\boldsymbol{k}_n)\right) \boldsymbol{v}_n}{\sum_{n=1}^N \phi(\boldsymbol{q}_m)^\top \varphi(\boldsymbol{k}_n)} \tag{11}$$

이를 통해 **$O(N)$ 복잡도를 유지하면서 상대 위치 정보를 인코딩**하는 것이 가능해집니다. 이는 장문 처리에서의 일반화 능력을 확장합니다.

### 3.4 중국어 사전 학습 실험에서의 일반화 근거

다단계 사전 학습에서 최대 시퀀스 길이를 변경할 때 정확도가 향상되는 양상이 관찰되었습니다:

| Stage | Max seq length | Accuracy |
|-------|---------------|----------|
| 1 | 512 | 65.0% |
| 2 | 1536 | 66.8% |
| 5 | 1536 | **67.4%** |

논문은 이를 "**RoPE의 우수한 일반화 능력(excellent generalizability)**"에 기인한다고 명시하며, 긴 시퀀스로 확장할 때 성능 저하 없이 잘 작동함을 보여줍니다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**① 대형 언어 모델(LLM)의 표준 위치 인코딩으로 채택**

RoPE는 현재 가장 영향력 있는 오픈소스 LLM들의 핵심 구성 요소로 자리잡았습니다:
- **LLaMA** (Meta AI, 2023): RoPE 채택
- **PaLM** (Google, 2022): RoPE 변형 사용
- **Mistral** (Mistral AI, 2023): RoPE 채택
- **Falcon** (Technology Innovation Institute, 2023): RoPE 채택

**② 컨텍스트 길이 확장 연구의 기반**

RoPE의 시퀀스 길이 유연성은 컨텍스트 창(context window) 확장 연구를 촉발했습니다:
- **Position Interpolation (Chen et al., 2023):** RoPE의 각도를 선형 보간하여 컨텍스트 길이 수십 배 확장
- **YaRN (Peng et al., 2023):** RoPE 주파수 조정을 통한 효율적 컨텍스트 확장
- **LongRoPE (Ding et al., 2024):** 비균등 위치 보간을 통한 100만 토큰 이상의 컨텍스트 처리

**③ 멀티모달 및 비언어 도메인 확장**

- 비전 트랜스포머(ViT)에서 2D RoPE 변형 적용 (공간적 위치 관계 모델링)
- 코드 생성, 수학적 추론 등 구조적 위치 관계가 중요한 영역에서 활발히 활용

### 4.2 향후 연구 시 고려할 점

**① 컨텍스트 길이 외삽(Extrapolation) 문제**

RoPE는 이론적으로 임의 길이에 적용 가능하나, **학습 시 보지 못한 위치에 대한 외삽(extrapolation) 성능이 저하**될 수 있습니다. Position Interpolation이나 YaRN 같은 방법이 이를 완화하지만, 근본적 해결책은 아직 연구 중입니다. 따라서:
- 사전 학습 시 다양한 시퀀스 길이를 포함하는 학습 전략 필요
- Interpolation vs. Extrapolation의 트레이드오프 분석 필요

**② $\theta_i$ 파라미터의 최적화**

현재 $\theta_i = 10000^{-2i/d}$는 경험적으로 선택된 값입니다. 태스크에 따라 최적의 base 값이 다를 수 있으며 (예: 코드 생성 모델 Code LLaMA는 base를 100만으로 설정), **학습 가능한 $\theta_i$ 또는 태스크 적응적 주파수 설계**가 연구 과제입니다.

**③ 선형 어텐션과의 완전한 통합**

논문에서 제안한 선형 어텐션 + RoPE 결합(식 11)은 분모를 변경하지 않아 이론적 불완전성이 있습니다. **엄밀한 수학적 완성도를 갖춘 선형 어텐션 + 상대 위치 인코딩 통합** 방법론이 필요합니다.

**④ 장거리 감쇠의 양날의 검 문제**

장거리 감쇠는 일반적으로 바람직하지만, **초장문 문서에서 멀리 떨어진 핵심 정보를 놓치는 원인**이 될 수 있습니다. Retrieval-Augmented Generation(RAG)이나 Sparse Attention과의 결합을 통해 이 문제를 보완하는 연구가 필요합니다.

**⑤ 다차원 위치 인코딩으로의 확장**

텍스트는 1D 시퀀스이지만, 이미지(2D), 비디오(3D), 그래프 데이터 등에서는 다차원 위치 관계가 중요합니다. **2D/3D RoPE 일반화**는 멀티모달 모델 연구에서 중요한 방향입니다.

**⑥ 수렴 가속 메커니즘 규명**

논문 자체에서 인정한 한계 — 왜 더 빠르게 수렴하는가 — 에 대한 이론적 규명이 필요합니다. 이는 **더 나은 위치 인코딩 설계 원칙** 수립에 기여할 것입니다.

---

## 5. 2020년 이후 최신 연구 비교 분석

| 방법 | 논문 | 위치 인코딩 방식 | 특징 | RoPE 대비 |
|------|------|----------------|------|-----------|
| **T5 Relative Bias** | Raffel et al. (2020) | 학습 가능한 스칼라 바이어스 $b_{i,j}$ | 단순, 효율적 | 선형 어텐션 호환 불가 |
| **DeBERTa** | He et al. (2020) | Disentangled Attention (내용+위치 분리) | GLUE 최고 성능 | 상대 위치 명시적 분리, 계산 비용 높음 |
| **ALiBi** | Press et al. (2021) | 어텐션 점수에 선형 페널티 추가 | 외삽에 강함, 파라미터 없음 | 더 단순하고 외삽 우수, 그러나 회전 이론 없음 |
| **Kerple** | Chi et al. (2022) | 커널 기반 상대 위치 인코딩 | 이론적 일반화 | 더 유연하나 복잡 |
| **Position Interpolation** | Chen et al. (2023) | RoPE + 선형 보간 | 컨텍스트 확장 | RoPE의 직접 확장 |
| **YaRN** | Peng et al. (2023) | RoPE 주파수 조정 | 효율적 컨텍스트 확장 | RoPE 개선 |
| **LongRoPE** | Ding et al. (2024) | 비균등 위치 보간 | 100만+ 토큰 | RoPE 대규모 확장 |

### 비교 분석 요약

- **ALiBi (Press et al., 2021):** 학습 파라미터 없이 어텐션 행렬에 상대 거리 기반 선형 바이어스를 빼는 방식으로, 외삽 능력이 RoPE보다 우수하다고 보고됨. 그러나 회전 불변성이나 이론적 해석 가능성은 RoPE가 우위
- **DeBERTa (He et al., 2020):** GLUE 벤치마크에서 강력한 성능을 보이나, 선형 어텐션 호환 불가 및 높은 계산 비용
- **RoPE의 위상:** 현재 LLM 생태계에서 사실상 표준(de facto standard)으로 자리잡으며, 수많은 후속 연구의 기반이 됨

---

## 참고자료

1. **주요 논문:**
   - Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864v5*

2. **비교 연구:**
   - Press, O., Smith, N. A., & Lewis, M. (2021). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." *arXiv:2108.12409*
   - He, P., Liu, X., Gao, J., & Chen, W. (2020). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." *arXiv:2006.03654*
   - Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR, 21, 140:1–140:67*
   - Chen, S., et al. (2023). "Extending Context Window of Large Language Models via Positional Interpolation." *arXiv:2306.15595*
   - Peng, B., et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models." *arXiv:2309.00071*

3. **기반 연구:**
   - Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*
   - Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL-HLT*
   - Katharopoulos, A., et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." *ICML*
   - Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). "Self-Attention with Relative Position Representations." *NAACL-HLT*
   - Dai, Z., et al. (2019). "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." *ACL*

4. **HuggingFace 공식 문서:** https://huggingface.co/docs/transformers/model_doc/roformer
5. **공식 GitHub:** https://github.com/ZhuiyiTechnology/roformer
