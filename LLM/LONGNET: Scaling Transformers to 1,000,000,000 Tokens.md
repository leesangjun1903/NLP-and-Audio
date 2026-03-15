# LONGNET: Scaling Transformers to 1,000,000,000 Tokens

---

## 1. 핵심 주장 및 주요 기여 요약

**LONGNET**은 Microsoft Research와 Xi'an Jiaotong University가 제안한 Transformer 변형 모델로, 시퀀스 길이를 **10억(1B) 토큰**까지 확장하면서도 짧은 시퀀스에서의 성능 저하가 없는 아키텍처이다.

### 핵심 주장
- 기존 Transformer의 self-attention은 $O(N^2d)$의 이차 복잡도를 가지며, 이는 긴 시퀀스 처리의 근본적 병목이다.
- **Dilated Attention**을 통해 **선형 복잡도** $O(Nd)$를 달성하고, 임의의 두 토큰 간 **로그 의존성** $O(\log N)$을 보장한다.

### 주요 기여
1. **Dilated Attention 메커니즘 제안**: 거리에 따라 어텐션 할당이 지수적으로 감소하는 새로운 어텐션 패턴
2. **선형 계산 복잡도**: $O(Nd)$로 기존 대비 획기적 감소
3. **분산 학습 알고리즘**: 시퀀스 차원을 분할하여 다중 GPU에서 초장문 시퀀스 학습 가능
4. **Drop-in replacement**: 기존 Transformer 최적화(FlashAttention, 양자화, 분산 학습 등)와 원활한 통합
5. **스케일링 법칙 유지**: 모델 크기 확장 시 dense Transformer와 유사한 power law를 따름

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

대규모 언어 모델(LLM) 시대에 시퀀스 길이의 확장은 핵심 과제이다. 논문에서는 긴 시퀀스가 필요한 세 가지 이유를 제시한다:

1. **대용량 메모리 및 수용 범위(receptive field)**: 모델이 인간 및 세계와 상호작용하는 데 필요
2. **복잡한 인과관계 및 추론 경로**: 짧은 의존성은 **가짜 상관관계(spurious correlations)**를 유발하여 일반화에 해로움
3. **In-context learning의 한계 탐색**: 극도로 긴 문맥은 **치명적 망각(catastrophic forgetting)** 완화 가능

기존 접근법의 한계:

| 방법 | 계산 복잡도 | 한계 |
|------|-----------|------|
| Recurrent (RNN) | $O(Nd^2)$ | 병렬화 불가 |
| Vanilla Attention | $O(N^2d)$ | 이차 복잡도로 긴 시퀀스 불가 |
| Sparse Attention | $O(N\sqrt{N}d)$ | 1B 토큰까지 확장 불가 |
| State Space Models | 선형 | 일반 길이에서 Transformer 대비 낮은 표현력 |

### 2.2 제안하는 방법: Dilated Attention

#### 기본 Self-Attention (Preliminary)

표준 self-attention은 입력 $Q, K, V \in \mathbb{R}^{N \times d}$에 대해:

$$O = \text{softmax}(QK^T)V $$

#### Dilated Attention의 핵심 연산

입력 $(Q, K, V)$를 세그먼트 길이 $w$로 분할하고, 각 세그먼트를 dilation rate $r$로 희소화(sparsify)한다:

$$\widetilde{Q}_i = [Q_{iw}, Q_{iw+r}, Q_{iw+2r}, \ldots, Q_{(i+1)w-1}] $$

$$\widetilde{K}_i = [K_{iw}, K_{iw+r}, K_{iw+2r}, \ldots, K_{(i+1)w-1}] $$

$$\widetilde{V}_i = [V_{iw}, V_{iw+r}, V_{iw+2r}, \ldots, V_{(i+1)w-1}] $$

희소화된 세그먼트에 대해 병렬로 어텐션을 수행:

$$\widetilde{O}_i = \text{softmax}(\widetilde{Q}_i \widetilde{K}_i^T)\widetilde{V}_i $$

결과를 scatter 및 concatenation하여 최종 출력을 생성:

$$\hat{O}_i = \{\widetilde{O}_{i,j} \mid j \bmod r = 0;\; 0 \mid j \bmod r \neq 0\} $$

$$O = [\hat{O}_0, \hat{O}_1, \ldots, \hat{O}_{\frac{N}{w}-1}] $$

#### 혼합 Dilated Attention (Mixture of Dilated Attentions)

단·장거리 정보를 모두 포착하기 위해 서로 다른 $(r_i, w_i)$ 조합의 dilated attention을 혼합:

$$O = \sum_{i=1}^{k} \alpha_i O\big|_{r_i, w_i} $$

$$\alpha_i = \frac{s_i}{\sum_j s_j} $$

여기서 $s_i$는 어텐션 softmax의 분모(denominator)로, **동적 가중치**가 학습 가능한 고정 가중치보다 우수함을 실험적으로 확인하였다.

세그먼트 크기와 dilation rate는 기하수열로 설정:

$$w = \{w_0, w_1, w_2, \ldots, N\}^k \quad (w_i < w_{i+1} < N) $$

$$r = \{1, r_1, r_2, \ldots, r_k\}^k \quad (1 < r_i < r_{i+1}) $$

**설계 직관**: 로컬 어텐션은 정밀하게(작은 $r$, 작은 $w$), 글로벌 어텐션은 근사적으로(큰 $r$, 큰 $w$) 계산한다.

#### Multi-Head Dilated Attention

$j$번째 헤드에 대해 오프셋 $s_j = j \bmod r$을 적용하여 서로 다른 위치를 희소화:

$$\widetilde{Q}_i = [Q_{iw+s_j}, Q_{iw+s_j+r}, Q_{iw+s_j+2r}, \ldots, Q_{(i+1)w+s_j-1}] $$

$$\widetilde{K}_i = [K_{iw+s_j}, K_{iw+s_j+r}, K_{iw+s_j+2r}, \ldots, K_{(i+1)w+s_j-1}] $$

$$\widetilde{V}_i = [V_{iw+s_j}, V_{iw+s_j+r}, V_{iw+s_j+2r}, \ldots, V_{(i+1)w+s_j-1}] $$

각 헤드가 다른 위치를 커버하므로, 전체적으로 시퀀스의 모든 토큰에 대한 정보를 포괄한다.

### 2.3 계산 복잡도 및 토큰 의존성

단일 dilated attention의 FLOPs:

$$FLOPs = \frac{2N}{w}\left(\frac{w}{r}\right)^2 d = \frac{2Nwd}{r^2} $$

다중 세그먼트/dilation rate에 대해:

$$FLOPs = 2Nd \sum_{i=1}^{k} \frac{w_i}{r_i^2} $$

기하수열 설정 시 (공비 $\alpha > 1$):

$$FLOPs = 2w_0 Nd \sum_{i=0}^{k-1} \frac{1}{\alpha^i} \leq \frac{2\alpha}{\alpha - 1} w_0 Nd \quad (\alpha > 1) $$

따라서 계산 복잡도는 $O(Nd)$이다.

**토큰 간 최대 전파 거리**:

$$D = \sum_{i=0}^{l-1} w_i = w_0 \sum_{i=0}^{l-1} \alpha^i \approx \frac{w_0}{\alpha - 1}\alpha^l $$

**$N$ 토큰 시퀀스의 최대 경로 길이**:

$$L \approx \log_\alpha \frac{N(\alpha - 1)}{w_0} \quad (\alpha > 1) $$

이는 토큰 의존성이 $O(\log N)$임을 증명한다.

### 2.4 모델 구조

- **백본**: MAGNETO (Foundation Transformer) + XPOS 상대 위치 인코딩
- **구성**: Hidden dim 768, 12 어텐션 헤드, 12 디코더 레이어 (base 설정)
- **핵심 변경**: 표준 어텐션을 dilated attention으로 대체 (나머지 구조는 동일)
- **구현 특징**: gather/scatter 연산을 통해 dense attention으로 변환 가능하여 FlashAttention 등 기존 최적화 기술을 직접 활용

### 2.5 분산 학습 알고리즘

시퀀스 차원을 다수 GPU에 분할:

$$X = [X_1, X_2] $$

각 디바이스에서 Q, K, V를 독립적으로 투영:

$$[Q_1, K_1, V_1] = [W_Q, W_K, W_V]X_1, \quad [Q_2, K_2, V_2] = [W_Q, W_K, W_V]X_2 $$

글로벌 세그먼트($w_i > l$)의 경우, 희소화 후 all-gather로 key-value를 수집:

$$\widetilde{K} = [\widetilde{K}_1, \widetilde{K}_2], \quad \widetilde{V} = [\widetilde{V}_1, \widetilde{V}_2] $$

로컬 쿼리와 글로벌 key-value로 cross-attention 수행:

$$\widetilde{O}_1 = \text{softmax}(\widetilde{Q}_1 \widetilde{K}^T)\widetilde{V}, \quad \widetilde{O}_2 = \text{softmax}(\widetilde{Q}_2 \widetilde{K}^T)\widetilde{V} $$

**핵심 특성**: 희소화 후 $\widetilde{K}_i$, $\widetilde{V}_i$의 크기가 시퀀스 길이 $N$에 독립적이므로 **통신 비용이 상수**이다. 이 분산 알고리즘은 데이터, 모델, 파이프라인 병렬화와 **직교적**으로 동작한다.

### 2.6 성능 향상

**언어 모델링 퍼플렉시티 (The Stack 데이터셋)**:

| 모델 | 학습 길이 | 2K | 8K | 32K |
|------|---------|-----|-----|------|
| Transformer | 2K | 4.24 | 5.07 | 11.29 |
| Sparse Transformer | 32K | 5.15 | 4.00 | 3.64 |
| **LONGNET** | **32K** | **4.37** | **3.33** | **3.01** |

주요 결과:
1. **LONGNET은 모든 테스트 길이에서 Sparse Transformer를 능가**
2. 동일 FLOPs에서 LONGNET이 vanilla Transformer보다 낮은 test perplexity 달성
3. **런타임**: 8K → 1B 토큰으로 확장 시 dilated attention은 거의 일정한 지연 시간, vanilla attention은 급격히 증가
4. **스케일링 법칙 유지**: 125M → 2.7B 파라미터까지 dense Transformer와 유사한 power law 곡선
5. **Long Context Prompting**: 컨텍스트 윈도우가 1K → 32K로 증가할수록 test loss가 지속적으로 감소

### 2.7 한계

1. **실험 범위의 제한**: 언어 모델링(The Stack, 소스코드)에 주로 실험. 자연어 다운스트림 태스크(QA, 요약, 분류 등)에 대한 체계적 평가 부재
2. **Dilated Attention의 근사 특성**: 글로벌 어텐션이 근사적이므로, 모든 토큰 쌍 간의 정밀한 어텐션이 필요한 태스크에서는 정보 손실 가능
3. **하이퍼파라미터 의존성**: 세그먼트 크기 $w$와 dilation rate $r$의 기하수열 설정(공비 $\alpha$)이 태스크에 따라 최적화 필요
4. **추론 시 실용성**: 1B 토큰 규모의 추론(inference)에 대한 구체적인 지연 시간·메모리 분석이 제한적
5. **다양한 도메인 검증 부재**: 멀티모달, 게노믹 데이터 등은 향후 과제로만 언급

---

## 3. 모델의 일반화 성능 향상 가능성

LONGNET의 일반화 성능 향상과 관련된 핵심 내용을 심층 분석한다.

### 3.1 짧은 의존성 vs. 긴 의존성과 일반화

논문에서 명시적으로 강조하는 핵심 논점:

> *"A longer context contains more complex causality and reasoning paths that models can exploit in training data. In contrast, short dependency has more spurious correlations, which is harmful to generalization."* (Section 1)

이는 **긴 문맥이 일반화에 유리한 근본적 이유**를 제시한다:
- 짧은 문맥에서는 우연한 통계적 패턴(가짜 상관관계)에 의존할 수 있어 과적합(overfitting) 위험이 높다.
- 긴 문맥은 진정한 인과관계를 학습할 기회를 제공하여, 분포 외(out-of-distribution) 데이터에 대한 일반화 능력을 강화한다.

### 3.2 실험적 근거

1. **외삽(Extrapolation)보다 직접 학습이 우수**: Table 2에서 Transformer(2K 학습)가 32K 입력을 처리할 때 perplexity가 11.29로 급증하지만, LONGNET(32K 학습)은 3.01을 달성. 이는 **긴 시퀀스를 직접 학습하는 것이 외삽보다 일반화에 훨씬 효과적**임을 보여준다.

2. **스케일링 곡선 (Figure 6)**: LONGNET은 동일 FLOPs에서 vanilla Transformer 대비 더 낮은 test perplexity를 달성. 이는 dilated attention이 동일 연산량 대비 **더 효율적으로 장거리 의존성을 학습**하여 일반화 성능이 개선됨을 시사한다.

3. **Long Context Prompting (Figure 7(b))**: 컨텍스트 윈도우 확장(1K → 32K)에 따라 test loss가 단조 감소. 이는 모델이 긴 문맥 정보를 **실제로 활용**하여 일반화하고 있음을 의미한다.

4. **Power Law 유지 (Figure 7(a))**: 125M → 2.7B로 모델 크기를 확장할 때 LONGNET이 power law를 따르므로, dilated attention이 **모델 크기 확장에 따른 일반화 성능 향상을 저해하지 않음**을 확인하였다.

### 3.3 로그 의존성( $O(\log N)$ )의 일반화 관점 해석

수식 (20)에서 도출된 $L \approx \log_\alpha \frac{N(\alpha-1)}{w_0}$는 매우 긴 시퀀스에서도 임의의 두 토큰이 대수적 깊이의 경로로 연결됨을 보장한다. 이는:
- **정보 전파의 효율성**: 1B 토큰에서도 약 $O(\log_{2} 10^9) \approx 30$ 수준의 경로 길이
- **장거리 의존성 학습**: 긴 시퀀스에서의 인과관계 포착 가능
- **일반화 향상**: 더 풍부한 문맥 정보에 기반한 예측

### 3.4 In-context Learning과의 관계

논문은 극도로 긴 문맥이 **many-shot in-context learning의 패러다임 전환** 가능성을 제시하며, 이는 치명적 망각(catastrophic forgetting) 완화와 직결된다. 이 관점에서 LONGNET의 일반화 향상 가능성은:
- 더 많은 예시를 문맥에 포함하여 **few-shot에서 many-shot learning으로 전환**
- 전체 코퍼스나 인터넷 전체를 하나의 시퀀스로 처리하는 **패러다임 수준의 변화**

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

1. **시퀀스 길이의 패러다임 전환**: 512 → 12K → 64K → 1M → 1B로의 지수적 확장 추세를 확립. "전체 코퍼스를 하나의 시퀀스로" 처리하는 비전 제시

2. **효율적 어텐션 연구의 새로운 기준선**: Dilated attention의 $O(Nd)$ 복잡도와 $O(\log N)$ 토큰 의존성은 이후 효율적 어텐션 메커니즘의 벤치마크

3. **분산 학습 전략의 확장**: 시퀀스 차원 병렬화는 기존 데이터/모델/파이프라인 병렬화와 직교적이므로, **5차원 병렬화(5D parallelism)** 가능성을 열음

4. **멀티모달 및 과학 데이터로의 확장**: 논문에서 미래 과제로 제시한 멀티모달 LLM, BEiT 사전학습, 게노믹 데이터 모델링은 초장문 시퀀스가 자연스럽게 등장하는 도메인

5. **스케일링 법칙의 새로운 축**: 기존의 파라미터 수·학습 토큰 수에 더해 **시퀀스 길이**를 스케일링 법칙의 독립적 축으로 확립

### 4.2 향후 연구 시 고려할 점

1. **다운스트림 태스크 평가**: 소스코드 언어 모델링 외에 자연어 이해/생성, QA, 요약, 분류 등 다양한 태스크에서의 체계적 평가 필요

2. **어텐션 근사의 품질 분석**: Dilation으로 인한 정보 손실이 특정 태스크(예: 정밀한 참조 해결, 수학적 추론)에서 성능 저하를 유발하는지 조사

3. **하이퍼파라미터 자동 최적화**: 세그먼트 크기 $w$, dilation rate $r$, 공비 $\alpha$의 태스크별 최적화 전략 연구

4. **추론 효율성**: 학습 시의 효율성은 입증되었으나, 실시간 추론(inference) 시의 지연 시간, 메모리 사용량, KV-cache 관리 전략에 대한 심층 분석 필요

5. **위치 인코딩과의 상호작용**: XPOS 및 기타 상대 위치 인코딩이 dilated attention과 결합될 때의 최적 전략

6. **Retrieval-Augmented Generation(RAG)과의 비교**: 매우 긴 문맥 vs. 검색 기반 증강의 효용 비교

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 최대 시퀀스 길이 | 복잡도 | 핵심 접근법 | LONGNET과의 비교 |
|------|------|-------------|-------|----------|--------------|
| **Longformer** [BPC20] | 2020 | ~16K | $O(N)$ | 슬라이딩 윈도우 + 글로벌 어텐션 | 글로벌 토큰 수가 제한적; 1B 확장 불가 |
| **BigBird** [ZGD+20] | 2020 | ~16K | $O(N)$ | 랜덤 + 슬라이딩 윈도우 + 글로벌 | 이론적 Turing completeness 증명; 실제 길이 제한적 |
| **Performer** [KVPF20] / **Linear Transformer** | 2020 | 이론적 무한 | $O(Nd^2)$ | 커널 근사 | 선형이지만 표현력 저하 문제; 실제 성능 열세 |
| **Linformer** [WLK+20] | 2020 | ~8K | $O(Nd)$ | 저랭크 근사 | Key-Value를 저차원으로 투영; 고정 투영 행렬의 한계 |
| **S4 (Structured State Space)** [GGR22] | 2022 | 이론적 무한 | $O(N \log N)$ | State space model | Long Range Arena에서 우수하나 일반 NLP 성능 열세 |
| **FlashAttention** [DFE+22] | 2022 | 복잡도 불변 ( $O(N^2d)$ ) | $O(N^2d)$ | IO-aware exact attention | 정확한 어텐션의 속도/메모리 개선; 복잡도 자체는 미변경. LONGNET이 이를 내부적으로 활용 |
| **Memorizing Transformers** [WRHS22] | 2022 | ~262K | $O(N^2d)$ (kNN 검색) | 외부 메모리 + kNN | 추가 메모리 스토리지 필요; 1B 확장 어려움 |
| **RMT (Recurrent Memory Transformer)** [BKB23] | 2023 | ~1M | $O(N)$ (세그먼트별) | 메모리 토큰 + 재귀 | 재귀적 처리로 병렬화 제한; 메모리 압축 손실 |
| **Hyena Hierarchy** [PMN+23] | 2023 | ~131K+ | $O(N \log N)$ | 긴 합성곱 | 어텐션 없이 긴 합성곱; 일반 NLP에서 Transformer 대비 열세 |
| **CoLT5** [ALdJ+23] | 2023 | ~64K | 조건부 연산 | 중요 토큰에 heavy attention | 전체 시퀀스에 대한 접근은 light attention; 1B 확장 미검증 |
| **Mamba (S6)** [Gu & Dao, 2023] | 2023 | 이론적 무한 | $O(Nd)$ | 선택적 SSM | 어텐션 없이 선형 복잡도; Transformer 대비 특정 태스크에서 경쟁력. LONGNET과 직접 비교 없음 |
| **Ring Attention** [Liu et al., 2023] | 2023 | 수백만+ | $O(N^2d)$ (분산) | 링 형태 분산 어텐션 | 복잡도는 여전히 이차이나 메모리를 분산; LONGNET은 복잡도 자체를 선형으로 줄임 |

### 핵심 차별화 포인트

1. **복잡도와 표현력의 균형**: LONGNET은 $O(Nd)$ 복잡도와 $O(\log N)$ 토큰 의존성을 **동시에** 달성한 최초의 모델 중 하나
2. **확장성**: 실제로 1B 토큰까지 확장을 시연한 유일한 연구 (2023년 기준)
3. **기존 인프라 호환성**: FlashAttention 등 기존 최적화와 호환되는 drop-in replacement
4. **분산 학습의 체계성**: 시퀀스 차원 병렬화를 위한 체계적 분산 알고리즘 제시

---

## 참고 자료

1. Ding, J., Ma, S., Dong, L., Zhang, X., Huang, S., Wang, W., Zheng, N., & Wei, F. (2023). "LONGNET: Scaling Transformers to 1,000,000,000 Tokens." *arXiv:2307.02486v2* [cs.CL].
2. Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). "Generating Long Sequences with Sparse Transformers." *arXiv:1904.10509*.
3. Beltagy, I., Peters, M. E., & Cohan, A. (2020). "Longformer: The Long-Document Transformer." *arXiv:2004.05150*.
4. Zaheer, M. et al. (2020). "Big Bird: Transformers for Longer Sequences." *NeurIPS 2020*.
5. Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*.
6. Gu, A., Goel, K., & Ré, C. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces." *ICLR 2022*.
7. Bulatov, A., Kuratov, Y., & Burtsev, M. S. (2023). "Scaling Transformer to 1M Tokens and Beyond with RMT." *arXiv:2304.11062*.
8. Poli, M. et al. (2023). "Hyena Hierarchy: Towards Larger Convolutional Language Models." *arXiv:2302.10866*.
9. Katharopoulos, A. et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." *ICML 2020*.
10. Wang, S. et al. (2020). "Linformer: Self-Attention with Linear Complexity." *arXiv:2006.04768*.
11. Kaplan, J. et al. (2020). "Scaling Laws for Neural Language Models." *arXiv:2001.08361*.
12. Wu, Y. et al. (2022). "Memorizing Transformers." *ICLR 2022*.
13. Ainslie, J. et al. (2023). "CoLT5: Faster Long-Range Transformers with Conditional Computation." *arXiv:2303.09752*.
14. Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv:2312.00752*.
15. Liu, H. et al. (2023). "Ring Attention with Blockwise Transformers for Near-Infinite Context." *arXiv:2310.01889*.
