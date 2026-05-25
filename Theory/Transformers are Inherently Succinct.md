# Transformers are Inherently Succinct

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **고정 정밀도(fixed-precision) 트랜스포머는 다른 형식 언어 표현 방식들에 비해 놀랍도록 간결(succinct)하다**는 것입니다. 기존 연구들이 트랜스포머의 *표현력(expressivity)*에 집중한 반면, 이 논문은 **간결성(succinctness)**이라는 새로운 렌즈를 제안합니다.

> **간결성(Succinctness)**: 어떤 형식 체계가 언어를 얼마나 *압축적으로* 기술할 수 있는가를 나타내는 척도. 두 형식 체계가 동일한 언어 집합을 인식할 수 있더라도, 하나가 다른 하나보다 훨씬 짧은 표현으로 같은 언어를 기술할 수 있다면, 그것이 더 간결합니다.

### 주요 기여 (5가지)

| 번호 | 기여 내용 |
|---|---|
| ① | UHAT는 LTL보다 **지수적(exponentially)** 으로 더 간결함 (Theorem 15) |
| ② | UHAT는 RNN보다 **지수적**으로 더 간결함 (Corollary 18) |
| ③ | UHAT는 유한 오토마톤보다 **이중 지수적(doubly exponentially)** 으로 더 간결함 (Theorem 17) |
| ④ | UHAT → LTL 변환의 상한을 이중 지수에서 **단일 지수**로 개선 (Proposition 13) |
| ⑤ | UHAT의 비어있음(emptiness) 및 동치(equivalence) 검증 문제가 **EXPSPACE-완전**임을 증명 (Theorem 4, 19) |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

기존 연구들(Yang et al., 2024; Barceló et al., 2024 등)은 트랜스포머가 어떤 언어 *클래스*를 인식하는가에 집중했습니다. 고정 정밀도 트랜스포머는 **별규칙 언어(star-free languages)**의 부분집합만 인식한다고 알려져 있어, RNN보다 표현력이 약합니다.

그러나 이 논문은 다른 질문을 던집니다:

> *"트랜스포머가 인식하는 언어들을 얼마나 **압축적으로** 기술할 수 있는가?"*

이 질문은 실제 성능과 연관이 있습니다. RNN이 더 표현력이 강함에도 불구하고 트랜스포머가 실증적으로 우수한 이유를 설명하는 단서가 될 수 있기 때문입니다.

### 2.2 연구 대상 모델: UHAT (Unique-Hard Attention Transformer)

논문은 **UHAT(Unique Hard-Attention Transformer)**를 연구의 핵심 모델로 사용합니다.

**UHAT의 구성 요소:**

- **심볼 임베딩**: $\text{emb}: \Sigma \to \mathbb{Q}^D$
- **스코어 함수** (어텐션 레이어에서):

$$S(\boldsymbol{v}_n, \boldsymbol{v}_m) \stackrel{\text{def}}{=} \langle \boldsymbol{A}(\boldsymbol{v}_n), \boldsymbol{B}(\boldsymbol{v}_m) \rangle $$

- **마스킹된 위치 집합** 및 **최고 스코어 위치 집합**:

$$U_n \stackrel{\text{def}}{=} \{m \in [N] \mid M(n,m) = 1\} $$

$$B_n \stackrel{\text{def}}{=} \{m \in U_n \mid \forall m' \in U_n: S(\boldsymbol{v}_n, \boldsymbol{v}_m) \geq S(\boldsymbol{v}_n, \boldsymbol{v}_{m'})\} $$

- **어텐션 벡터**: $\boldsymbol{a}\_n \stackrel{\text{def}}{=} \boldsymbol{v}_{\tau(B_n)}$ (타이 브레이킹 함수 $\tau$로 유일 선택)
- **ReLU 레이어**:

$$\rho_r(\boldsymbol{v}) \stackrel{\text{def}}{=} (v_{1:r-1}, \max(0, v_r), v_{r+1:R}) $$

- **언어 인식**: $\mathcal{L}(\mathcal{T}) = \{\mathbf{a} \in \Sigma^+ \mid \langle \boldsymbol{t}, \boldsymbol{v}_N \rangle > 0\}$

### 2.3 중간 표현: B-RASP

EXPSPACE-하드니스 증명의 중간 단계로 **Boolean RASP (B-RASP)**를 사용합니다. B-RASP의 어텐션 연산은:

$$P_{t+1}(i) \stackrel{\text{def}}{=} \blacktriangleleft\!\!\!\blacktriangleright_j \; [M(i,j),\, S(i,j)]\; V(i,j) : D(i) $$

여기서 $o(i)$는:

$$o(i) \stackrel{\text{def}}{=} \begin{cases} \min\{j \in [N] \mid M(i,j)=1 \text{ and } S(i,j)=1\}, & \text{for } \blacktriangleleft \\ \max\{j \in [N] \mid M(i,j)=1 \text{ and } S(i,j)=1\}, & \text{for } \blacktriangleright \end{cases} $$

Yang et al. (2024)에 의해 B-RASP는 UHAT와 표현적으로 동치임이 알려져 있습니다.

### 2.4 간결성의 정의 (수식)

**정의 2 ($f$-more succinct)**: $\mathcal{C}^{(1)}$이 $\mathcal{C}^{(2)}$보다 $f$-더 간결하다는 것은, 언어 패밀리 $\{L_n\}_{n=1}^\infty$와 $R_n^{(1)} \in \mathcal{C}^{(1)}$이 존재하여 모든 $R_n^{(2)} \in \mathcal{C}^{(2)}$에 대해:

$$|R_n^{(2)}| \geq f(|R_n^{(1)}|)$$

- **지수적으로 더 간결**: $f(n) \in \Omega(2^{cn^d})$ for $c, d > 0$
- **이중 지수적으로 더 간결**: $f(n) \in \Omega(2^{2^{cn^d}})$ for $c, d > 0$

**정의 3 ($g$-bounded expansion)**: $\mathcal{C}^{(1)}$이 $\mathcal{C}^{(2)}$에 대해 $g$-유계 팽창을 가진다는 것은, 모든 언어 $L$과 $R^{(2)} \in \mathcal{C}^{(2)}$에 대해 $|R^{(1)}| \leq g(|R^{(2)}|)$인 $R^{(1)} \in \mathcal{C}^{(1)}$이 존재함.

### 2.5 핵심 기술적 요소: 이중 지수 카운터 인코딩

논문의 핵심 기술적 아이디어는 UHAT가 어텐션을 통해 $0$부터 $2^{2^N}$까지 세는 **이중 지수 크기의 카운터**를 구현할 수 있다는 것입니다.

**예시 (N=4 비트 카운터, B-RASP):**

비트 카운터 증분 연산:

```math
C_{+1}(i) \stackrel{\text{def}}{=} \blacktriangleright_j [j < i, Q_\#(j)] \bigvee_{k=1}^{4} \left( \bigwedge_{r=1}^{k-1} (\neg C_r(i) \wedge C_r(j)) \wedge C_k(i) \wedge \neg C_k(j) \wedge \bigwedge_{r=k+1}^{4} (C_r(i) \leftrightarrow C_r(j)) \right) : 1
```

인접 심볼 제약 검사:

$$M_\leftarrow(i) \stackrel{\text{def}}{=} \blacktriangleright_j [j < i, Q_a(j) \vee Q_b(j) \vee Q_c(j)] \bigvee_{(h,h') \in H} Q_h(j) \wedge Q_{h'}(i) : 1 $$

### 2.6 주요 정리들과 증명 구조

**Theorem 4 (EXPSPACE-완전성)**:

> UHAT와 B-RASP 프로그램의 비어있음 문제(non-emptiness problem)는 **EXPSPACE-완전**이다.

증명 구조:
1. **하한 (EXPSPACE-hard)**: $2^N$-타일링 문제 (EXPSPACE-완전, Prop. 7) → Lemma 8 (다항 시간 B-RASP 변환) → Lemma 9 (다항 시간 UHAT 변환)
2. **상한 (EXPSPACE)**: UHAT → LTL (지수 시간, Prop. 13) → LTL 만족 가능성은 PSPACE (Sistla & Clarke, 1985) → 전체 EXPSPACE

**Proposition 12** (정밀도 경계):

> 모든 UHAT $\mathcal{T}$에 대해, 계산 중 발생하는 모든 유리수 값은 최대 $\text{poly}(|\mathcal{T}|)$ 비트로 표현 가능하다.

이 명제가 Prop. 13의 기반이 됩니다. 값의 집합 $F$의 크기가 최대 $2^{\text{poly}(|\mathcal{T}|)}$이므로, LTL 공식 구성이 지수 시간에 가능합니다.

**Proposition 13** (UHAT → LTL 변환):

레이어 $\ell$에서 값 $\boldsymbol{v} \in F^S$에 대한 LTL 공식 $\varphi_\boldsymbol{v}^\ell$을 귀납적으로 구성:

- **기저 (임베딩 레이어)**:

$$\varphi_\boldsymbol{v}^0 \stackrel{\text{def}}{=} \begin{cases} \bigvee_{a \in \text{emb}^{-1}(\boldsymbol{v})} Q_a & \text{if } \text{emb}^{-1}(\boldsymbol{v}) \neq \emptyset \\ \bot & \text{otherwise} \end{cases} $$

- **ReLU 레이어**:

$$\varphi_\boldsymbol{v}^{\ell+1} \stackrel{\text{def}}{=} \bigvee_{\substack{u \in F, \\ \max\{0,u\}=v_k}} \varphi_{(\boldsymbol{v}_{1:k-1},\, u,\, \boldsymbol{v}_{k+1:R})}^\ell $$

- **엄격 마스킹 어텐션 레이어** (미래 마스킹, 오른쪽 타이 브레이킹):

$$\varphi_\boldsymbol{v}^{\ell+1} \stackrel{\text{def}}{=} \bigvee_{\substack{\boldsymbol{u},\boldsymbol{a} \in F^R,\\ \boldsymbol{C}(\boldsymbol{u},\boldsymbol{a})=\boldsymbol{v}}} \varphi_\boldsymbol{u}^\ell \wedge \left( \left(\bigvee_{\substack{\boldsymbol{b}\in F^R,\\ S(\boldsymbol{u},\boldsymbol{b})<S(\boldsymbol{u},\boldsymbol{a})}} \varphi_\boldsymbol{b}^\ell \right) \mathbf{S} \left(\varphi_\boldsymbol{a}^\ell \wedge \neg\mathbf{P}\bigvee_{\substack{\boldsymbol{b}\in F^R,\\ S(\boldsymbol{u},\boldsymbol{b})>S(\boldsymbol{u},\boldsymbol{a})}} \varphi_\boldsymbol{b}^\ell\right)\right) $$

- **최종 수용 공식**:

$$\varphi \stackrel{\text{def}}{=} \bigvee_{\substack{\boldsymbol{v}\in F^S,\\ \langle \boldsymbol{t},\boldsymbol{v}\rangle>0}} \varphi_\boldsymbol{v}^m $$

**Theorem 15** (UHAT는 LTL보다 지수적으로 간결):

증명은 3단계:
1. $\mathcal{M}_n$: $2^n$비트 이진 카운터를 구현하는 튜링 머신 (선형 개 상태, 지수 테이프, 최단 수용 런 길이 $\geq 2^{2^n}$)
2. Van Emde Boas (1997)의 환원: $2^{p(n)}$-타일링 문제 인스턴스 $\mathcal{I}_n$ (다항식 크기, 최소 정확한 타일링 행 수 $\geq 2^{2^n}$)
3. Lemma 8, 9로 다항식 크기 UHAT $\mathcal{T}_n$ 구성

LTL 하한: LTL 공식의 최단 수용 단어 길이는 공식 크기의 지수 이하 → $|\varphi_n| \geq \Omega(2^n)$

**Theorem 17** (UHAT는 FA보다 이중 지수적으로 간결):

같은 witness family $\{L_n\}$ 사용. 비어있지 않은 언어를 인식하는 오토마톤은 오토마톤 크기의 선형 길이 이하 단어를 수용해야 함. 그런데 $L_n$의 최단 단어는 $\geq 2^{2^n}$ → 최소 오토마톤 크기 $\geq 2^{2^n}$.

**Corollary 18** (UHAT는 RNN보다 지수적으로 간결):

**Proposition 1**: 고정 정밀도 $k$인 RNN은 $2^{kD}$개 상태의 FA로 표현 가능 → Theorem 17 + Prop. 1 결합.

**Theorem 19** (동치 문제의 EXPSPACE-완전성):

- **하한**: 비어있음 문제 → 동치 문제 (빈 언어를 인식하는 UHAT $\mathcal{T}_0$와의 동치 검사)
- **상한**: Prop. 13으로 LTL 변환 후 PSPACE 동치 검사

### 2.7 간결성 결과 요약 (표)

| 비교 대상 | 방향 | 간결성 차이 |
|---|---|---|
| UHAT vs. LTL | UHAT $\succ$ LTL | 지수적 (Theorem 15) |
| UHAT vs. RNN | UHAT $\succ$ RNN | 지수적 (Corollary 18) |
| UHAT vs. FA | UHAT $\succ$ FA | 이중 지수적 (Theorem 17) |
| UHAT vs. SSM | UHAT $\succ$ SSM | 지수적 (by extension) |
| LTL vs. UHAT | LTL → UHAT | 다항식 팽창 (Prop. 16) |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 간결성과 일반화의 이론적 연결

논문에서 직접적으로 일반화 성능을 다루는 실험은 없지만, 간결성 결과는 일반화와 관련된 중요한 이론적 함의를 가집니다.

**가설**: 더 간결한 표현은 더 *단순한* 가설 클래스를 의미하고, 단순한 가설 클래스는 (Occam's Razor / PAC 학습 이론에 따라) 더 나은 일반화 성능을 보일 수 있습니다.

구체적으로:

$$\text{일반화 오차} \leq \hat{\mathcal{L}}(h) + O\!\left(\sqrt{\frac{\log|\mathcal{H}|}{m}}\right)$$

트랜스포머가 같은 언어를 LTL이나 FA보다 지수적으로 더 작은 크기로 표현할 수 있다면, 효과적인 가설 공간의 크기 $|\mathcal{H}|$가 작아지고, 이는 더 빠른 일반화로 이어질 수 있습니다.

### 3.2 길이 일반화와의 연결

논문은 길이 일반화(length generalization)와의 연결을 언급합니다:

> "their trainability and ability to generalize to unseen strings of longer lengths (Zhou et al., 2024; Huang et al., 2025; Chiang and Cholak, 2022)"

**핵심 관찰**: UHAT가 $2^{2^n}$ 길이의 최단 수용 단어를 가진 언어를 다항식 크기로 기술할 수 있다는 것은, 트랜스포머가 매우 긴 패턴을 *압축적인 규칙*으로 표현할 수 있음을 의미합니다. 이는 훈련에서 보지 못한 긴 문자열에 대한 일반화 능력과 직결됩니다.

관련 최신 연구인 **Huang et al. (2025)**("A formal framework for understanding length generalization in transformers", ICLR 2025)와 **Yang et al. (2026)**("Length generalization bounds for transformers", ICML 2026)은 이 방향을 직접 탐구합니다. 후자는 본 논문의 저자들이 공동 작성한 후속 연구입니다.

### 3.3 간결성이 시사하는 일반화 메커니즘

트랜스포머의 간결성은 다음과 같은 방식으로 일반화에 기여할 수 있습니다:

**① 압축된 규칙 표현**: 트랜스포머는 어텐션 메커니즘을 통해 이중 지수 크기의 카운터를 다항식 크기로 인코딩합니다. 이는 복잡한 구조적 패턴을 **소수의 파라미터**로 표현할 수 있음을 의미합니다.

**② 별규칙 언어에서의 효율성**: 트랜스포머가 인식하는 언어 클래스(별규칙 언어)에서 트랜스포머는 FA나 RNN보다 훨씬 작은 표현을 사용합니다. 이는 해당 클래스에서 **샘플 효율성**이 높을 수 있음을 시사합니다.

**③ 귀납적 편향(Inductive Bias)**: 동일한 언어를 표현하는 다항식 크기 트랜스포머가 존재한다면, 경사 하강법이 이러한 간결한 해를 선호하는 귀납적 편향을 가질 가능성이 있습니다. 단, 이는 **학습 가능성(learnability)** 문제와 별개이며, 논문 자체는 이를 미해결 과제로 남깁니다:

> "A related open question is the learnability of succinct transformers, on which the empirical evidence remains mixed (Garg et al., 2022; Naim et al., 2025; Huang et al., 2025)."

### 3.4 한계: 간결성과 일반화의 간극

주의해야 할 점은, 간결한 표현의 **존재**와 학습 알고리즘이 그것을 **실제로 찾을 수 있는가**는 별개의 문제입니다.

- EXPSPACE-완전한 비어있음 문제는 모든 학습 알고리즘이 다루기 매우 어렵다는 것을 의미합니다.
- 경사 하강법이 간결한 해를 선호하는지 여부는 이론적으로 보장되지 않습니다.
- 실증 연구들은 혼재된 결과를 보입니다 (Garg et al., 2022; Naim et al., 2025).

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

**① 트랜스포머 이론 연구의 패러다임 전환**

이 논문은 표현력(expressivity)에서 **간결성(succinctness)**으로 분석의 초점을 이동시킵니다. RNN이 표현력은 더 강하지만 트랜스포머가 실증적으로 우수한 역설을 설명하는 새로운 이론적 틀을 제공합니다.

**② 형식 검증(Formal Verification) 연구**

- UHAT의 비어있음과 동치 문제가 EXPSPACE-완전임을 증명함으로써, 트랜스포머 검증의 계산 복잡도 하한을 확립했습니다.
- 이는 실용적인 검증 도구 개발 방향을 제시합니다: 카운터를 인코딩하지 못하는 **제한된 서브클래스** 식별 및 그에 대한 더 효율적인 검증 알고리즘 개발.
- Corollary 14: 엄격한 미래 마스킹 + 왼쪽 타이 브레이킹 UHAT의 비어있음 문제는 **NEXP** 내에 있어, 더 낮은 복잡도의 검증이 가능합니다.

**③ 아키텍처 비교 연구**

트랜스포머 vs. SSM vs. RNN 비교에서 표현력 외에 간결성을 고려해야 한다는 새로운 기준을 제시합니다. 특히 Mamba (Gu & Dao, 2023) 등 SSM 계열 모델과의 이론적 비교가 더 풍부해질 수 있습니다.

**④ UHAT → LTL 변환 개선**

Yang et al. (2024)의 이중 지수 변환을 **단일 지수**로 개선함으로써, LTL 기반 형식 언어 분석 도구를 트랜스포머에 더 효율적으로 적용할 수 있게 되었습니다.

**⑤ 길이 일반화 이론**

후속 연구인 Yang et al. (2026) ("Length generalization bounds for transformers", ICML 2026)은 이 논문의 간결성 결과를 길이 일반화 경계 도출에 직접 활용합니다.

### 4.2 앞으로 연구 시 고려할 점

**① 소프트맥스 트랜스포머로의 확장**

이 논문은 UHAT(unique-hard attention)에 집중합니다. 실제 트랜스포머는 소프트맥스 어텐션을 사용하며, Li & Cotterell (2025)에 따르면 UHAT는 소프트맥스 트랜스포머를 과근사(overapproximate)합니다. 논문 자체도 이를 미래 연구 과제로 명시합니다:

> "We leave the succinctness of fixed-precision softmax and average-hard attention transformers as future work; see Yang et al. (2026) for an initial attempt."

또한, Sälzer et al. (2026)은 위치 인코딩 없이 무한 정밀도의 average-hard-attention 및 softmax 트랜스포머에서 검증이 **결정 불가능(undecidable)**임을 보였습니다.

**② 위치 인코딩의 영향**

이 논문은 위치 정보를 마스킹(strict past/future masking)으로만 표현합니다. Sälzer et al. (2025)처럼 임의의 고정 정밀도 위치 인코딩을 허용하면 간결성이 어떻게 변하는지 탐구가 필요합니다.

**③ 학습 가능성(Learnability) 연구**

간결한 트랜스포머의 **존재**와 학습 알고리즘이 이를 **찾을 수 있는가**는 별개의 문제입니다. 향후 연구에서는:
- 경사 하강법(SGD, Adam)이 간결한 해를 얼마나 잘 찾는지
- 어떤 조건에서 간결성이 학습 효율성을 보장하는지
를 실증적·이론적으로 탐구해야 합니다.

**④ 실용적 검증 도구 개발**

EXPSPACE-완전성은 최악의 경우에 대한 결과이므로, 실용적인 경우에는 더 효율적인 알고리즘이 가능할 수 있습니다. 모델 검사(model checking) 기법, 심볼릭 방법, 추상화(abstraction) 등을 트랜스포머 검증에 적용하는 연구가 필요합니다.

**⑤ 층 수와 헤드 수의 영향**

이 논문의 UHAT 모델은 단순화된 추상이므로, 실제 트랜스포머의 층 수, 헤드 수, 임베딩 차원이 간결성에 미치는 영향을 구체화하는 연구가 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 주제 | 본 논문과의 관계 |
|---|---|---|---|
| Hahn (2020), *TACL* | 2020 | 소프트맥스 트랜스포머의 이론적 한계 (패리티, 역전 불가) | 트랜스포머 표현력 연구의 선구자; 본 논문은 표현력이 아닌 간결성으로 전환 |
| Weiss et al. (2018, 2024), *ACL / ML* | 2018, 2024 | 고정 정밀도 RNN의 언어 인식 능력; 오토마톤 추출 | RNN의 간결성 하한 제공 (Corollary 18의 기반) |
| Merrill et al. (2020), *ACL* | 2020 | RNN 아키텍처의 형식적 계층 구조 | RNN이 모든 정규 언어 인식 가능 → UHAT보다 표현력 강하지만 덜 간결 |
| Pérez et al. (2021), *JMLR* | 2021 | 어텐션은 튜링 완전 (무한 정밀도) | 무한 정밀도 설정; 본 논문은 고정 정밀도에서 더 정밀한 결과 |
| Chiang & Cholak (2022), *ACL* | 2022 | 자기 어텐션의 이론적 한계 극복 | 표현력 관점; 간결성 연구의 배경 |
| Hao et al. (2022), *TACL* | 2022 | 하드 어텐션 트랜스포머의 형식 언어 인식 (회로 복잡도) | UHAT 모델의 직접적 선행 연구 |
| Gu & Dao (2023) (Mamba) | 2023 | SSM: 선택적 상태 공간 모델 | 본 논문 Corollary 18이 SSM으로 확장; RNN과 동일한 간결성 하한 |
| Merrill et al. (2024), *ICML* | 2024 | SSM의 "상태 환상(illusion of state)" | SSM이 RNN처럼 정규 언어 인식 가능 → UHAT보다 덜 간결 |
| Yang et al. (2024), *NeurIPS* | 2024 | 마스킹된 하드 어텐션 트랜스포머가 정확히 별규칙 언어를 인식 | 본 논문의 핵심 선행 연구; UHAT→LTL 변환의 이중 지수 버전 제공 → 본 논문이 단일 지수로 개선 |
| Barceló et al. (2024), *ICLR* | 2024 | 하드 어텐션 트랜스포머의 논리적 언어 | 표현력 연구; 본 논문과 상호 보완 |
| Sälzer et al. (2025), *ICLR* | 2025 | 트랜스포머 인코더 만족 가능성: 복잡도 및 형식 추론 | NEXP-하드 결과 제시; 본 논문은 EXPSPACE-완전으로 강화, 모델도 더 단순 |
| Li & Cotterell (2025), *NeurIPS* | 2025 | 고정 정밀도 트랜스포머 언어 모델의 표현력 특성 | UHAT가 소프트맥스 트랜스포머를 과근사함을 보임; 본 논문의 결과가 실제 트랜스포머에 적용되는 연결고리 |
| Jerad et al. (2025), *ACL* | 2025 | 유일 하드 어텐션: 양면 | UHAT ↔ LTL 단편의 동치 증명 (이중 지수 변환); 본 논문이 단일 지수로 개선 |
| Huang et al. (2025), *ICLR* | 2025 | 트랜스포머의 길이 일반화 이해를 위한 형식적 프레임워크 | 본 논문의 간결성 결과와 직접 연결되는 일반화 연구 |
| Zhou et al. (2024), *ICLR* | 2024 | 트랜스포머가 학습하는 알고리즘: 길이 일반화 연구 | 간결성과 길이 일반화의 실증적 탐구 |
| Sälzer et al. (2026), *ICLR* | 2026 | 위치 인코딩 없는 트랜스포머의 검증 불가능성 | 본 논문의 미래 연구 방향 제시; 무한 정밀도에서 결정 불가능 |
| Yang et al. (2026), *ICML* | 2026 | 트랜스포머의 길이 일반화 경계 | 본 논문 저자들의 후속 연구; 간결성 결과를 직접 활용 |

---

## 참고 자료 (출처)

본 답변은 다음 자료에 기반합니다:

1. **Bergstraßer, P., Cotterell, R., & Lin, A. W. (2025/2026). "Transformers are Inherently Succinct." arXiv:2510.19315v3 [cs.FL], 15 May 2026.** *(제공된 PDF 원문)*

다음은 논문 내에서 인용된 주요 참고문헌들입니다:

2. Yang, A., Chiang, D., & Angluin, D. (2024). "Masked hard-attention transformers recognize exactly the star-free languages." *NeurIPS 2024*.
3. Barceló, P., Kozachinskiy, A., Lin, A. W., & Podolskii, V. V. (2024). "Logical languages accepted by transformer encoders with hard attention." *ICLR 2024*.
4. Jerad, S., Svete, A., Li, J., & Cotterell, R. (2025). "Unique hard attention: A tale of two sides." *ACL 2025*.
5. Li, J. & Cotterell, R. (2025). "Characterizing the expressivity of fixed-precision transformer language models." *NeurIPS 2025*.
6. Sälzer, M., Alsmann, E., & Lange, M. (2025). "Transformer encoder satisfiability: Complexity and impact on formal reasoning." *ICLR 2025*.
7. Sälzer, M., Köcher, C., Kozachinskiy, A., Zetzsche, G., & Lin, A. W. (2026). "The counting power of transformers." *ICLR 2026*.
8. Hahn, M. (2020). "Theoretical limitations of self-attention in neural sequence models." *TACL*, 8.
9. Merrill, W., Petty, J., & Sabharwal, A. (2024). "The illusion of state in state-space models." *ICML 2024*.
10. Merrill, W., Weiss, G., Goldberg, Y., Schwartz, R., Smith, N. A., & Yahav, E. (2020). "A formal hierarchy of RNN architectures." *ACL 2020*.
11. Sistla, A. P. & Clarke, E. M. (1985). "The complexity of propositional linear temporal logics." *Journal of the ACM*, 32(3).
12. Gu, A. & Dao, T. (2023). "Mamba: Linear-time sequence modeling with selective state spaces."
13. Huang, X., Yang, A., et al. (2025). "A formal framework for understanding length generalization in transformers." *ICLR 2025*.
14. Yang, A., Bergstraßer, P., Zetzsche, G., Chiang, D., & Lin, A. W. (2026). "Length generalization bounds for transformers." arXiv:2603.02238. *(ICML 2026)*
15. Zhou, H., Bradley, A., et al. (2024). "What algorithms can transformers learn? A study in length generalization." *ICLR 2024*.
16. Pérez, J., Barceló, P., & Marinkovic, J. (2021). "Attention is Turing-complete." *JMLR*, 22.
17. Strobl, L., Merrill, W., Weiss, G., Chiang, D., & Angluin, D. (2024). "What formal languages can transformers express? A survey." *TACL*, 12.
18. Schwarzentruber, F. (2019). "The complexity of tiling problems." arXiv:1907.00102.
19. Van Emde Boas, P. (1997). "The convenience of tilings." In *Complexity, Logic, and Recursion Theory*. CRC Press.
20. Stockmeyer, L. J. (1974). *The Complexity of Decision Problems in Automata Theory and Logic*. Ph.D. thesis, MIT.
21. Grohe, M. & Schweikardt, N. (2004). "The succinctness of first-order logic on linear orders." *LICS 2004*.
22. Hao, Y., Angluin, D., & Frank, R. (2022). "Formal language recognition by hard attention transformers: Perspectives from circuit complexity." *TACL*, 10.
23. Bergstraßer, P., Köcher, C., Lin, A. W., & Zetzsche, G. (2024). "The power of hard attention transformers on data sequences: A formal language theoretic perspective." *NeurIPS 2024*.
24. Weiss, G., Goldberg, Y., & Yahav, E. (2018). "On the practical computational power of finite precision RNNs for language recognition." *ACL 2018*.
25. Garg, S., Tsipras, D., Liang, P., & Valiant, G. (2022). "What can transformers learn in-context? A case study of simple function classes." *NeurIPS 2022*.
26. Naim, O., Bolte, J., & Asher, N. (2025). "Analyzing limits for in-context learning." *NeurIPS Workshop 2025*.
