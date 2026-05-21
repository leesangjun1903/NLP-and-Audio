# Fast Inference from Transformers via Speculative Decoding

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Leviathan et al. (2023)의 핵심 주장은 다음 두 가지 관찰에 기반합니다:

1. **난이도 비대칭성**: 언어 모델링의 어려운 태스크에는 작은 모델로도 충분히 근사 가능한 쉬운 서브태스크가 포함되어 있다.
2. **메모리 병목**: 대형 모델의 추론은 산술 연산이 아닌 메모리 대역폭(memory bandwidth)에 의해 병목이 발생하므로, 추가적인 연산 자원이 활용 가능하다.

이 두 관찰을 결합하여, **출력 분포를 전혀 변경하지 않으면서** 2X–3X의 추론 가속을 달성합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **Speculative Sampling** | 확률적 추측 실행(stochastic speculative execution)을 위한 새로운 샘플링 방법 |
| **Speculative Decoding** | 아키텍처 변경·재학습 없이 기존 모델에 즉시 적용 가능한 디코딩 메커니즘 |
| **분포 동일성 보장** | 출력 분포가 타겟 모델 단독 사용과 수학적으로 동일함을 증명 |
| **실증적 성능** | T5-XXL (11B)에서 T5X 대비 2X–3X 속도 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

자기회귀(autoregressive) 모델에서 $K$개의 토큰을 생성하려면 **$K$번의 직렬(serial) 실행**이 필요합니다. 이는 대형 Transformer 모델에서 심각한 추론 지연(latency)을 유발합니다. 기존 솔루션들(蒸留, 양자화, 조기 종료 등)은 대부분 **아키텍처 변경이나 재학습**을 요구하며, 출력 분포가 달라지는 문제가 있었습니다.

### 2.2 제안하는 방법: Speculative Decoding

#### 모델 설정

- $M_p$: 타겟 모델 (대형, 느림), 분포 $p(x_t | x_{<t})$
- $M_q$: 근사 모델 (소형, 빠름), 분포 $q(x_t | x_{<t})$

#### 핵심 알고리즘: Speculative Sampling

토큰 $x \sim p(x)$를 샘플링하기 위해:

1. $x \sim q(x)$에서 샘플링
2. $q(x) \leq p(x)$이면 수락
3. $q(x) > p(x)$이면 확률 $1 - \frac{p(x)}{q(x)}$로 기각하고, 조정된 분포에서 재샘플링:

$$p'(x) = \text{norm}\left(\max\left(0,\ p(x) - q(x)\right)\right)$$

이 절차를 통해 최종 샘플은 항상 $x \sim p(x)$를 따름이 수학적으로 보장됩니다 (Appendix A.1 참조).

#### 알고리즘 1: SpeculativeDecodingStep

```
입력: Mp, Mq, prefix
1. Mq로 γ개의 추측 토큰 x₁,...,xᵧ를 자기회귀적으로 샘플링
2. Mp를 병렬 실행:
   p₁(x),...,pᵧ₊₁(x) ← Mp(prefix),...,Mp(prefix+[x₁,...,xᵧ])
3. 수락 개수 n 결정:
   r₁,...,rᵧ ~ U(0,1)
   n ← min({i-1 | rᵢ > pᵢ(x)/qᵢ(x)} ∪ {γ})
4. 거부 시 분포 조정:
   p'(x) ← norm(max(0, p_{n+1}(x) - q_{n+1}(x)))
5. t ~ p'(x) 샘플링 후 반환
```

### 2.3 핵심 수식

#### 수락률 (Acceptance Rate) $\alpha$

**Definition 3.1**: 수락률 $\beta_{x_{<t}}$는 주어진 prefix $x_{<t}$에서 $x_t \sim q(x_t|x_{<t})$가 수락될 확률.

$D_{LK}$ 발산을 다음과 같이 정의합니다:

$$D_{LK}(p, q) = \sum_x |p(x) - M(x)|, \quad M(x) = \frac{p(x)+q(x)}{2}$$

**Lemma 3.3**:

$$D_{LK}(p, q) = 1 - \sum_x \min(p(x), q(x))$$

**Theorem 3.5**:

$$\beta = 1 - D_{LK}(p, q)$$

**Corollary 3.6**:

$$\alpha = 1 - \mathbb{E}[D_{LK}(p, q)] = \mathbb{E}\left[\sum_x \min(p(x), q(x))\right]$$

#### 생성 토큰 기대값

i.i.d. 가정 하에 단일 알고리즘 실행에서 생성되는 기대 토큰 수:

$$\mathbb{E}(\text{생성 토큰 수}) = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha} \tag{1}$$

#### 벽시계 시간 개선 인수 (Walltime Improvement Factor)

**Definition 3.7**: $c$ = $M_q$ 단일 실행 시간 / $M_p$ 단일 실행 시간 (비용 계수)

**Theorem 3.8**:

$$\text{Walltime 개선 인수} = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}$$

**Corollary 3.9**: $\alpha > c$이면 개선이 존재하며, 최소 개선 인수:

$$\text{최소 개선} = \frac{1 + \alpha}{1 + c}$$

#### 산술 연산 증가율

**Theorem 3.11**:

$$\text{연산 증가 인수} = \frac{(1-\alpha)(\gamma\hat{c} + \gamma + 1)}{1 - \alpha^{\gamma+1}}$$

여기서 $\hat{c}$는 근사 모델의 토큰당 연산 비율.

#### 최적 $\gamma$ 선택

Theorem 3.8을 최대화하는 정수 $\gamma$를 수치적으로 탐색:

$$\gamma^* = \arg\max_{\gamma \in \mathbb{Z}^+} \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}$$

### 2.4 모델 구조

별도의 아키텍처 변경 없이 **기존 off-the-shelf 모델 쌍**을 사용합니다:

| 역할 | 모델 예시 | 파라미터 수 |
|------|-----------|-------------|
| 타겟 모델 $M_p$ | T5-XXL | 11B |
| 근사 모델 $M_q$ | T5-Small | 77M |
| 타겟 모델 $M_p$ | LaMDA | 137B |
| 근사 모델 $M_q$ | LaMDA | 100M~8B |

$M_q$는 $M_p$보다 **약 2 order of magnitude 작은** 모델이 최적 성능 균형을 제공합니다.

### 2.5 성능 향상

실험 결과 (T5-XXL 11B, T5X 대비):

| 태스크 | $M_q$ | 온도 | $\gamma$ | $\alpha$ | 속도 향상 |
|--------|--------|------|----------|----------|-----------|
| En→De | T5-Small | 0 (argmax) | 7 | 0.75 | **3.4X** |
| En→De | T5-Small | 1 | 7 | 0.62 | **2.6X** |
| CNN/DM | T5-Small | 0 | 5 | 0.65 | **3.1X** |
| CNN/DM | T5-Small | 1 | 5 | 0.53 | **2.3X** |

- Chen et al. (2023)의 독립적 구현에서 Chinchilla 70B 모델에 적용 시 유사한 **2X–2.5X** 개선 확인
- Argmax sampling(온도=0)에서 일관적으로 더 높은 $\alpha$ 및 속도 향상 달성

### 2.6 한계점

1. **추가 연산 자원 필요**: 병렬 실행을 위한 추가 GPU/TPU 자원이 없으면 효과 없음
2. **i.i.d. 가정**: $\beta$들이 i.i.d.라는 가정은 실제로는 근사치에 불과
3. **Beam Search 미완성**: 빔 서치와의 결합에 대한 분석이 미완
4. **배치 처리 비효율**: 배치 크기 > 1에서는 이점이 감소할 수 있음
5. **텍스트 모달리티 한정**: 이미지 등 다른 도메인 검증 미완

---

## 3. 일반화 성능 향상 가능성

### 3.1 분포 동일성 보장이 주는 일반화 의미

Speculative Decoding의 가장 강력한 특성은 **출력 분포의 수학적 동일성 보장**입니다. 이는 단순 가속을 넘어 일반화 관점에서 중요한 의미를 갖습니다:

$$P(x = x') = \min(p(x'), q(x')) + p(x') - \min(p(x'), q(x')) = p(x')$$

이는 어떤 근사 모델 $M_q$를 사용하더라도 최종 출력이 $M_p$의 분포를 따르므로, $M_p$의 일반화 능력이 **완전히 보존**됩니다.

### 3.2 다양한 근사 모델을 통한 일반화 확장

논문은 다양한 유형의 근사 모델이 사용 가능함을 보입니다:

#### (a) N-gram 모델 (Negligible-cost)
- $c \approx 0$인 초경량 모델
- En→De 태스크에서 bigram 모델만으로도 $\alpha \approx 0.2$ → **1.25X 속도 향상**
- 비모수적 근사 모델도 유효함을 입증

#### (b) 비자기회귀(Non-autoregressive) 모델
- Stern et al. (2018)과 같은 비자기회귀 모델을 $M_q$로 활용 가능
- 단일 순전파로 여러 토큰을 동시 제안

#### (c) 컨텍스트 복사 휴리스틱
- 요약, 대화 등 장문 반복이 많은 태스크에서 컨텍스트에서 직접 복사하는 전략
- 파라미터 없는 근사 모델로 배포 단순화

#### (d) Lenience 파라미터를 통한 유연한 일반화

엄격한 분포 동일성 대신 **lenience 파라미터** $l \in [0,1]$을 도입:

$$\alpha = \sum_x \min\left(\frac{p(x)}{l}, q(x)\right)$$

| $M_q$ | $l=1$ | $l=0.5$ | $l=0.3$ | $l=0.1$ |
|--------|--------|---------|---------|---------|
| T5-Small | 0.62 | 0.71 | 0.76 | 0.84 |
| T5-Base | 0.68 | 0.80 | 0.83 | 0.90 |

$l=0.1$에서 **5X 속도 향상** 달성 (어떤 토큰도 ground truth 확률의 10배 이상 샘플링되지 않음을 보장).

### 3.3 태스크 독립적 일반화

- **번역** (WMT En-De): $\alpha = 0.62$ – $0.82$
- **요약** (CNN/DM): $\alpha = 0.53$ – $0.74$
- **대화** (LaMDA 137B): $\alpha = 0.57$ – $0.75$
- **무조건 생성** (GPT-like 97M): $\alpha = 0.88$ – $0.89$

태스크 특성이 다양함에도 일관된 성능 향상을 보여 **도메인 일반화**가 우수합니다.

### 3.4 계층적 추론으로의 확장 가능성

논문은 계층적 알고리즘(hierarchical speculative decoding)을 제안합니다:
- $M_q$를 더 빠른 $M_{q'}$으로 가속
- 이론적으로 $\mathbb{E}(\text{토큰 수}) \to \frac{1}{1-\alpha}$ (상한)에 근접 가능
- Oracle $\gamma$ 사용 시 고정 $\gamma$ 대비 **최대 ~60% 추가 향상** 가능

---

## 4. 미래 연구에 미치는 영향 및 고려 사항

### 4.1 미래 연구에 미치는 영향

#### (a) LLM 서빙 인프라의 패러다임 전환

Speculative Decoding은 재학습 없이 적용 가능하므로, 실제 프로덕션 LLM 서빙(ChatGPT, Gemini 등)에 **즉시 통합 가능한 표준 기법**으로 자리잡고 있습니다. 이는 LLM 추론 최적화 연구의 새로운 방향을 제시합니다.

#### (b) 확률론적 추측 실행(Stochastic Speculative Execution)의 이론적 기반 제시

기존 결정론적 추측 실행을 확률론적 설정으로 일반화한 것은 다음 분야로 파급됩니다:
- 물리 시뮬레이션
- 강화학습 (행동 분포 $f(x)$와 환경 $g(y)$의 병렬화)
- 기타 확률론적 파이프라인

#### (c) 소형 모델 활용 생태계 촉진

"작은 모델의 출력을 큰 모델이 검증"하는 패러다임은 소형 특화 모델(small specialized draft models) 연구를 촉진합니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 출력 분포 보존 | 재학습 필요 | 속도 향상 |
|------|------|--------------|------------|-----------|
| **Leviathan et al. (2023)** [본 논문] | Speculative Decoding | ✅ 완전 보존 | ❌ 불필요 | 2X–3X |
| **Chen et al. (2023)** - "Accelerating large language model decoding with speculative sampling" (ArXiv abs/2302.01318) | 독립적 구현, 동일 원리 | ✅ | ❌ | 2X–2.5X (Chinchilla 70B) |
| **Stern et al. (2018)** - "Blockwise Parallel Decoding" (NeurIPS 2018) | 병렬 디코딩 | ❌ (greedy only) | ✅ 필요 | — |
| **Sun et al. (2021)** - "Shallow Aggressive Decoding" (ArXiv abs/2106.04970) | 입출력 복사 활용 | ❌ (stochastic 미지원) | ❌ | — |
| **Schuster et al. (2021)** - "Confident Adaptive Transformers" (EMNLP 2021) | 조기 종료 | ❌ 변경됨 | ✅ 필요 | — |
| **Schwartz et al. (2020)** - "The Right Tool for the Job" (ACL 2020) | 소형 모델 우선 사용 | ❌ 변경됨 | 부분 필요 | — |

#### 주요 차별점 분석

**본 논문 vs. Blockwise Parallel Decoding (Stern et al., 2018)**:
- Blockwise는 greedy decoding(temp=0)만 지원
- 본 논문은 임의의 확률적 샘플링 지원 + 분포 동일성 보장

**본 논문 vs. Chen et al. (2023)**:
- 동일한 원리를 독립적으로 발견
- Chen et al.은 Chinchilla 70B에 적용하여 유사 결과 확인 (상호 검증)

**본 논문 vs. SAD (Sun et al., 2021)**:
- SAD는 입출력이 매우 유사한 태스크(문법 교정 등)만 가능
- 본 논문은 임의의 근사 모델 사용 가능

### 4.3 향후 연구 시 고려 사항

#### ① 최적 Draft 모델 설계

현재 논문은 기존 소형 모델을 그대로 활용하지만, 더 높은 $\alpha$를 위해 다음을 고려해야 합니다:
- **Distillation 기반 Draft 모델**: $M_p$의 soft target으로 $M_q$를 직접 학습
- **$\alpha$ 직접 최적화**: $\alpha$를 목적함수로 두고 $M_q$ 학습

#### ② 동적 $\gamma$ 조정

현재 $\gamma$는 고정값이지만, $\beta$를 예측하여 동적으로 $\gamma$를 변화시키면 이론적으로 **최대 ~60% 추가 향상** 가능:
$$\mathbb{E}(\text{tokens with oracle}) = \frac{1}{1-\alpha}$$

#### ③ Beam Search와의 완전한 통합

Appendix A.4에서 부분적으로만 논의됨. 실제 응용에서는 beam search가 필수적이므로, 다음이 필요합니다:
$$\text{top}_w(M_p) \subseteq \text{top}_u(M_q) \Rightarrow \text{수락 조건}$$
이에 대한 엄밀한 분석 및 효율적 구현이 연구 과제입니다.

#### ④ 멀티모달·다른 도메인 확장

논문은 텍스트에만 적용했으나:
- **이미지 생성** (Parti, DALL-E 등의 자기회귀 모델)
- **오디오/음성** 자기회귀 모델
- **코드 생성**

#### ⑤ 배치 처리 효율화

배치 크기 $> 1$에서의 성능 분석 및 최적화가 필요합니다. 현재 논문은 batch size = 1에서만 평가했습니다.

#### ⑥ 하드웨어 인식 최적화

$c$는 하드웨어 구성에 의존하므로, GPU/TPU별 최적 ($\gamma$, $M_q$) 쌍을 자동으로 선택하는 **하드웨어 인식 스케줄러** 개발이 필요합니다.

#### ⑦ Lenience와 안전성의 트레이드오프

$l < 1$로 lenience를 허용하면 속도는 향상되지만:
- 어떤 토큰도 $\frac{p(x)}{l}$ 이상으로 샘플링되지 않음을 보장
- 그러나 희귀 토큰의 최소 샘플링 확률 보장 없음 → **다양성 감소 우려**
- 안전 critical 응용에서의 분포 변화 영향 연구 필요

#### ⑧ 계층적 Speculative Decoding

$$M_{q_2} \xrightarrow{\text{draft}} M_{q_1} \xrightarrow{\text{draft}} M_p$$

이론적으로 가능하나 실증적 검증 및 최적화 미완성. 각 계층의 $\alpha$, $c$ 균형 분석 필요.

---

## 참고문헌

1. **[주 논문]** Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. ICML 2023. ArXiv:2211.17192
2. **[독립 구현]** Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. M. (2023). *Accelerating large language model decoding with speculative sampling*. ArXiv:2302.01318
3. **[선행 연구]** Stern, M., Shazeer, N., & Uszkoreit, J. (2018). *Blockwise parallel decoding for deep autoregressive models*. NeurIPS 2018.
4. **[선행 연구]** Sun, X., Ge, T., Wei, F., & Wang, H. (2021). *Instantaneous grammatical error correction with shallow aggressive decoding*. ArXiv:2106.04970
5. **[관련 연구]** Schuster, T., Fisch, A., Jaakkola, T., & Barzilay, R. (2021). *Consistent accelerated inference via confident adaptive transformers*. EMNLP 2021.
6. **[관련 연구]** Schwartz, R., Stanovsky, G., Swayamdipta, S., Dodge, J., & Smith, N. A. (2020). *The right tool for the job: Matching model and instance complexities*. ACL 2020.
7. **[기반 모델]** Vaswani, A. et al. (2017). *Attention is all you need*. NeurIPS 2017.
8. **[기반 모델]** Raffel, C. et al. (2020). *Exploring the limits of transfer learning with a unified text-to-text transformer*. JMLR 21(1).
9. **[기반 모델]** Thoppilan, R. et al. (2022). *LaMDA: Language models for dialog applications*. ArXiv:2201.08239
10. **[컴퓨터 구조 참고]** Hennessy, J. L., & Patterson, D. A. (2012). *Computer Architecture: A Quantitative Approach*. Morgan Kaufmann, 5th edition.
11. **[지식 증류]** Hinton, G. E., Vinyals, O., & Dean, J. (2015). *Distilling the knowledge in a neural network*. ArXiv:1503.02531
