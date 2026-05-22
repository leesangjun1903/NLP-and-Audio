# Approximately Aligned Decoding (AprAD) 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

**AprAD**는 LLM의 출력에서 제약 위반(constraint violation)을 피하기 위한 디코딩 방법으로, 기존 방법들이 가진 두 가지 근본적 한계를 동시에 해결하고자 한다:

1. **Constrained Generation**: 계산 효율은 높지만 출력 분포를 심하게 왜곡(probability amplification)
2. **ASAp (Adaptive Sampling with Approximate Expected Futures)**: 출력 분포는 잘 보존하지만 계산 비용이 극단적으로 큼

AprAD는 **Speculative Decoding**의 토큰 수락/거부 메커니즘을 차용하여, 오류 발생 시 얼마나 많은 토큰을 재사용할지를 확률적으로 결정함으로써 위 두 극단의 **중간점(midpoint)**을 제공한다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| 기존 방법 분석 | Constrained Generation, ASAp, Posterior Estimation 방법들의 장단점 체계적 비교 |
| AprAD 제안 | Speculative Sampling을 Prefix Selection Algorithm으로 재해석, 추가 학습 불필요 |
| 실험적 검증 | 합성 환경, Lipogram(vowel exclusion), BigCodeBench API 환상 방지 실험 |
| 일반화된 프레임워크 | 세 방법이 모두 backtracking 전략의 차이임을 보임(Appendix E) |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**문제 정의 (Problem 1)**:

어휘 $\mathcal{V}$에 대한 자기회귀 언어 모델 $P$와 오류 집합 $\mathcal{B} \subset \mathcal{V}^*$가 주어졌을 때, 다음 분포에서 샘플링하는 방법을 제공하라:

$$\hat{P}^{\mathcal{B}}(w) = \begin{cases} 0 & w \in \mathcal{B} \\ \dfrac{P(w)}{\sum_{w' \notin \mathcal{B}} P(w')} & w \notin \mathcal{B} \end{cases} \tag{1}$$

**Dense Error Set 문제**: 오류 발생 확률이 토큰당 $p$라면, 길이 $d$의 출력이 오류 없을 확률은 $(1-p)^d$이므로, 평균적으로 $\frac{1}{(1-p)^d}$번의 생성이 필요하다. 이는 긴 시퀀스에서 기하급수적으로 증가한다.

---

### 2.2 제안 방법 (수식 포함)

#### Speculative Sampling (기반 알고리즘)

AprAD의 핵심은 **Algorithm 2 (SpecSample)**이다. 이는 원래 Speculative Decoding에서 사용되던 것으로, 두 분포 간의 비율을 이용해 토큰을 수락/거부한다:

```
for i ∈ [n+1 ... m]:
    r ← P(xᵢ | x₁...ᵢ₋₁) / S(xᵢ | x₁...ᵢ₋₁)
    with probability min(1, r): accept xᵢ
    else:
        R(t) = max(0, P(t|x₁...ᵢ₋₁) - S(t|x₁...ᵢ₋₁))
        return x₁...ᵢ₋₁, Sample(Normalize(R(·)))
return x₁...m, Sample(P(·|x₁...m))
```

#### AprAD의 핵심 아이디어

ASAp에서 오류 샘플 $x$를 발견하면 이를 $B$에 추가하여 $\hat{P}^{B \cup \{x\}}$를 구성한다. AprAD는:

$$\hat{P}^{B} \approx \hat{P}^{B \cup \{x\}} \quad \text{(두 분포는 매우 유사)}$$

임을 이용하여, $\hat{P}^{B}$에서 샘플된 오류 시퀀스 $x \sim \hat{P}^B$를 이용해 $\hat{P}^{B \cup \{x\}}$에서의 샘플을 근사한다:

$$\text{SpecSample}(\hat{P}^{B \cup \{x\}}, \hat{P}^{B}, n, x_{1...m})$$

**AddBadSample** 절차에서 확률 조정은:

$$\hat{P}^{B \cup \{x\}}(x_i | x_{1...i-1}) \leftarrow \hat{P}^{B}(x_i | x_{1...i-1}) - \hat{P}^{B}(x_{i...m} | x_{1...i-1})$$

그 후 재정규화(renormalize)한다.

#### 확률 증폭 경계 (Appendix C)

AprAD의 첫 번째 반복에서 시퀀스 $y \notin \mathcal{B}$가 생성될 확률은:

$$\text{AprAD}(y | P, \mathcal{B}) = P(y) + \sum_{x \in \mathcal{B}} P(x) \cdot \text{SRS}(y | x, P, \hat{P}^{\{x\}})$$

부분집합 부등식(Subset Inequalities)을 적용하면:

$$\leq \hat{P}^{\mathcal{B}}(y) + \sum_{x \in \mathcal{B}} P(x) \cdot \text{SRS}(y | x, P, \hat{P}^{\mathcal{B}})$$

$$\leq \hat{P}^{\mathcal{B}}(y) + \sum_{x \in \Sigma^*} P(x) \cdot \text{SRS}(y | x, P, \hat{P}^{\mathcal{B}})$$

Speculative Decoding Identity를 적용하면:

$$\leq 2\hat{P}^{\mathcal{B}}(y)$$

즉, **AprAD의 확률 증폭은 반복당 최대 2배**로 유계(bounded)이며, Constrained Generation은 이 증폭이 무한할 수 있다.

#### 하이퍼파라미터 $h$를 통한 연속 제어

SpecSample의 수락 확률 $r$을 다음과 같이 수정하여 연속적인 트레이드오프 제어 가능:

$$r = \left(\frac{P(x_i | x_{1...i-1})}{S(x_i | x_{1...i-1})}\right)^h$$

- $h = 0$: $r = 1$ → 항상 수락 → Constrained Generation과 동일
- $h = 1$: AprAD 기본 동작
- $h \to \infty$: $r \to 0$ → 항상 거부 → ASAp와 유사

---

### 2.3 모델 구조

AprAD는 별도의 모델 아키텍처를 새롭게 설계하지 않는다. 대신 기존 자기회귀 LLM의 **디코딩 단계에서** 작동하는 알고리즘이다.

**구성 요소**:

1. **자기회귀 LLM** $P$: Mistral-7B-Instruct-v0.2, StarCoder2 (7B, 15B) 등
2. **오류 감지기(Black-box Oracle)**: 정규식, 파서(Pyright), 부분 문자열 검색 등
3. **Trie 구조 확률 캐시**: $\hat{P}^B(\cdot | x_{1...m})$을 효율적으로 저장/갱신
4. **SpecSample 알고리즘**: 오류 발생 시 prefix 선택

**방법 분류 관점에서의 위치**:

$$\text{Constrained Gen.} \xrightarrow{\text{점점 더 많은 resampling}} \text{AprAD} \xrightarrow{} \text{ASAp}$$

---

### 2.4 성능 향상

#### 실험 1: 시뮬레이션 (KL-Divergence vs Generation Ratio)

| 오류 집합 | ASAp KL | Const. KL | AprAD KL | ASAp Ratio | Const. Ratio | AprAD Ratio |
|----------|---------|-----------|----------|------------|--------------|-------------|
| $\emptyset$ | 0.0014 | 0.0014 | 0.0014 | 1.000 | 1.000 | 1.000 |
| AAA | 0.0014 | 0.0075 | 0.0046 | 1.020 | 1.000 | 1.004 |
| A** except AAC | 0.0014 | 0.3836 | 0.1540 | 1.232 | 1.113 | 1.205 |
| *** except AAA, BAA | 0.0000 | 0.0000 | 0.0000 | **5.701** | 1.784 | **2.653** |

- AprAD는 **Constrained보다 훨씬 낮은 KL-divergence**를 달성
- AprAD는 **ASAp보다 훨씬 낮은 generation ratio**를 달성

#### 실험 2: Lipogram (모음 제외 생성)

| 방법 | 품질 (1-5) | 제약 의도 (1-3) | Gen. Ratio |
|------|-----------|----------------|-----------|
| Constrained | $3.56 \pm 0.34$ | $2.32 \pm 0.18$ | $1.00 \pm 0.00$ |
| **AprAD** | $\mathbf{4.52 \pm 0.23}$ | $\mathbf{2.84 \pm 0.11}$ | $4.20 \pm 0.91$ |
| ASAp | $1.72 \pm 0.23$ | $2.36 \pm 0.16$ | $321.00 \pm 79.43$ |
| Unconstrained | $4.68 \pm 0.17$ | $1.00 \pm 0.00$ | $1.00 \pm 0.00$ |

- AprAD 품질: Unconstrained에 거의 근접
- ASAp Gen. Ratio: 321 (사실상 불가능한 수준)

#### 실험 3: BigCodeBench (API 환상 방지)

| 크기 | 방법 | Pass@1 | Pass@5 | !NameErr@1 | Gen. Ratio |
|------|------|--------|--------|------------|-----------|
| 15b | AprAD | $0.26 \pm 0.01$ | $0.54 \pm 0.01$ | $0.98 \pm 0.00$ | $1.08 \pm 0.01$ |
| 15b | ASAp | $0.26 \pm 0.01$ | $0.54 \pm 0.01$ | $0.98 \pm 0.00$ | $1.56 \pm 0.12$ |
| 15b | Constrained | $0.22 \pm 0.01$ | $0.51 \pm 0.01$ | $0.93 \pm 0.01$ | $1.02 \pm 0.00$ |

- **AprAD ≈ ASAp** 성능, but **Generation Ratio는 훨씬 낮음** (1.08 vs 1.56)

---

### 2.5 한계

1. **여전히 확률 증폭 존재**: 이상적 분포 $\hat{P}^{\mathcal{B}}$에서 완전히 샘플링하지 못함
2. **Constrained Generation 대비 추가 오버헤드**: 제약이 sparse한 경우 불필요한 비용 발생
3. **극단적 dense error set**: 매우 밀집된 오류 집합에서는 여전히 성능 저하
4. **벽시계 시간(wall-clock time) 미측정**: Generation Ratio로만 비교 (환경 변수 때문)
5. **흑박스 오류 감지기 필요**: 오류가 증가 단조성(monotone prefix-closedness)을 만족해야 함
6. **목표 지향 생성 한계**: 특정 솔루션을 향한 탐색(예: 정리 증명)에는 추가 search 알고리즘 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 분포 왜곡 최소화와 일반화의 관계

AprAD의 핵심 강점은 이상적 분포 $\hat{P}^{\mathcal{B}}$에 더 가깝게 유지한다는 점이다. 이것이 일반화와 연결되는 메커니즘은 다음과 같다:

**Constrained Generation의 일반화 실패 사례**: 만약 $P(B|A) = 0.0001$이더라도, A 접두사 이후에 오류가 발생하면 무조건 B를 선택한다. 이는 극단적으로 낮은 확률 토큰을 강제 선택하게 되어 **모델의 내재적 언어 지식을 완전히 무시**한다.

반면 AprAD에서는:

$$r = \frac{\hat{P}^{B \cup \{x\}}(A)}{\hat{P}^{B}(A)} \approx \frac{1/3}{1/2} = \frac{2}{3}$$

즉, 이전 선택의 적절성을 **모델의 원래 확률 분포로 재평가**하여 불필요한 강제를 피한다.

### 3.2 도메인 간 일반화

AprAD는 **추가 학습 없이** 다양한 도메인에 적용 가능하다:

| 도메인 | 오류 감지 방법 | AprAD 적용 가능성 |
|--------|--------------|-----------------|
| 구조화된 텍스트 (JSON 등) | 파서 | ✅ 즉시 적용 |
| 코드 생성 | 언어 서버 (Pyright) | ✅ 즉시 적용 |
| 시적 제약 (Lipogram) | 정규식 | ✅ 즉시 적용 |
| 도구 호출 형식 | 파서 | ✅ 즉시 적용 |
| 환경 변화 | 런타임 의존 | ✅ 즉시 적응 (FUDGE, Ctrl-G와 달리) |

특히 **환경 변화에 대한 적응성**이 중요하다. FUDGE나 Ctrl-G는 특정 환경에 맞게 학습된 discriminator나 HMM을 사용하므로, 라이브러리 버전 변경이나 새로운 API 추가 시 재학습이 필요하다. AprAD는 오류 감지기만 교체하면 즉시 적응한다.

### 3.3 High-Entropy vs Low-Entropy 시나리오에서의 일반화

논문에서는 다음을 명시적으로 언급한다:

> "Because the distributions of $\hat{P}^B$ and $\hat{P}^{B \cup \{x\}}$ are so close to each other, this prefix is usually most of the length of $x$, **especially when the language model is relatively high-entropy**."

- **고엔트로피 시나리오**: AprAD가 장점이 두드러짐. 토큰 선택의 자유도가 높아 backtracking 후에도 적절한 대안 토큰이 많음
- **저엔트로피 시나리오**: Constrained Generation과의 차이가 더 극적. 저확률 토큰 강제 선택을 피함으로써 생성 품질 보존

### 3.4 확률 증폭 제어와 일반화

확률 증폭이 작을수록 모델이 학습한 **언어의 통계적 구조**를 더 잘 보존한다. KL-Divergence 실험에서 AprAD는 dense error set에서도 Constrained Generation 대비 훨씬 낮은 KL-divergence를 달성하여, 모델의 **원본 표현력**을 더 잘 유지한다.

이는 특히 다음 상황에서 중요하다:
- 길이가 긴 코드 생성 (오류 누적 가능성 ↑)
- 다양한 라이브러리를 사용하는 실용적 코딩 태스크
- 창의적 텍스트 생성 (자연스러운 언어 패턴 유지 필요)

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (1) Speculative Decoding의 재해석
AprAD는 Speculative Decoding을 **단순한 추론 가속** 기법에서 **일반적인 확률 분포 조정 도구**로 재해석한다. 이는 다음과 같은 새로운 연구 방향을 열 수 있다:

- 두 분포 간의 차이를 활용한 다양한 제어 기법 설계
- Speculative Decoding 프레임워크를 안전성(safety) 제약 적용에 활용

#### (2) 하이퍼파라미터 $h$에 대한 체계적 연구
논문에서는 $h$를 제안하지만 분석은 미래 연구로 남겨둔다. 이는 다음을 가능하게 한다:

$$r = \left(\frac{P(x_i | x_{1...i-1})}{S(x_i | x_{1...i-1})}\right)^h, \quad h \in [0, \infty)$$

- **태스크별 최적 $h$ 탐색** (자동 조정 알고리즘 개발)
- **적응적 $h$**: 생성 진행에 따라 $h$를 동적으로 조정

#### (3) Search Algorithm과의 결합
논문은 AprAD와 MCTS 등 탐색 알고리즘의 결합을 미래 연구로 남긴다:

- **정리 증명**: AprAD로 유효하지 않은 tactic 방지 + MCTS로 탐색 방향 결정
- **프로그램 합성**: 문법 제약 하의 탐색
- **단계별 추론(Chain-of-Thought)**: 논리적 오류 방지 + 빔 탐색

#### (4) 분산 오류 집합의 이론적 분석
현재 확률 증폭 경계( $2\hat{P}^{\mathcal{B}}(y)$ )는 엄밀한 증명이 아닌 스케치 수준이다. 엄밀한 이론적 분석이 필요하다.

### 4.2 앞으로 연구 시 고려할 점

#### (1) 오류 감지기 설계의 중요성
AprAD의 성능은 **오류 감지기의 품질**에 크게 의존한다:
- **False Positive 문제**: 유효한 시퀀스를 오류로 판단하면 불필요한 backtracking 발생
- **Prefix-closedness 가정**: 오류 집합이 이 가정을 만족하지 않는 경우(예: 비속어가 정상 단어의 부분 문자열인 경우) 별도 처리 필요
- **지연(latency) 고려**: 복잡한 오류 감지기(예: 전체 파서 실행)는 generation ratio를 향상시켜도 실제 벽시계 시간은 증가할 수 있음

#### (2) 부동 소수점 오류 관리
논문의 Appendix H에서 강조하듯, $\hat{P}^{B*}$ 테이블의 누적 부동 소수점 오류가 실제 구현에서 심각한 문제가 된다. 연구 시 다음을 고려해야 한다:
- 주기적 재계산(periodic recalculation) 전략
- 음수 확률 처리 방법

#### (3) Wall-clock Time 측정
본 논문은 Generation Ratio만 측정했지만, 실제 배포 환경에서는 다음도 중요하다:
- KV-cache 재사용 효율
- 병렬화 가능성
- 오류 감지기 실행 비용

#### (4) 대규모 모델에서의 확장성
논문은 7B-15B 규모에서만 실험했다. 더 큰 모델(70B+)에서의 동작, 특히 고엔트로피 가정이 여전히 성립하는지 검증이 필요하다.

#### (5) 다중 제약 조건 처리
현재는 단일 오류 집합 $\mathcal{B}$만 고려한다. 실제 응용에서는 여러 제약(문법 정확성 + API 환상 방지 + 안전성)을 동시에 다뤄야 하며, 이를 위한 $\mathcal{B}$의 합집합 처리 방법이 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 샘플링 기반 방법 비교

| 연구 | 연도 | 핵심 아이디어 | AprAD 대비 |
|------|------|-------------|-----------|
| **ASAp** (Park et al., 2024) | 2024 | 이미 발견된 오류를 B에 추가하여 정확한 $\hat{P}^{\mathcal{B}}$ 샘플링 | 분포 충실도↑, 계산 비용↑↑ |
| **Grammar-Constrained Decoding** (Geng et al., 2024) | 2024 | CFG를 활용한 효율적 토큰 마스킹 | 계산 효율↑↑, 확률 증폭↑↑ |
| **Gen-C** (Ahmed et al., 2025) | 2025 | 제약을 회로로 변환 후 오류 근방 재샘플링 | 표현력↑, 복잡도↑ |
| **AprAD** (본 논문, 2025) | 2025 | Speculative Sampling 재활용한 적응적 backtracking | 균형점 제공 |

### 5.2 사후 추정 기반 방법 비교

| 연구 | 연도 | 핵심 아이디어 | AprAD 대비 |
|------|------|-------------|-----------|
| **FUDGE** (Yang & Klein, 2021) | 2021 | 미래 제약 위반 확률을 신경망 discriminator로 추정 | 태스크별 학습 필요, 환경 변화 적응 어려움 |
| **SMC Steering** (Lew et al., 2023) | 2023 | Monte Carlo 샘플링으로 사후 확률 추정 | 높은 정확도 but 높은 계산 비용 |
| **Ctrl-G** (Zhang et al., 2024) | 2024 | LLM→HMM 증류 후 DFA와 곱으로 정확한 사후 확률 계산 | 높은 정확도 but 제약이 DFA로 표현 가능해야 함, 환경 변화 시 재증류 필요 |

### 5.3 Speculative Decoding 관련 연구 비교

| 연구 | 연도 | 핵심 아이디어 | AprAD와의 관계 |
|------|------|-------------|--------------|
| **Speculative Decoding** (Leviathan et al., 2023) | 2023 | SSM으로 초안 생성 후 LLM으로 병렬 검증 | AprAD의 직접적 영감 |
| **Medusa** (Cai et al., 2024) | 2024 | 다수의 decoding head로 여러 토큰 동시 예측 | 추론 가속에 집중, 제약 없음 |
| **EAGLE/EAGLE-2** (Li et al., 2024) | 2024 | Feature uncertainty를 고려한 동적 draft tree | 추론 가속에 집중, 제약 없음 |
| **Mentored Decoding** (Tran-Thien, 2024) | 2024 | LLM 분포에서 허용된 편차를 통한 속도 향상 | AprAD와 유사한 철학 (분포 약간 희생) |

### 5.4 종합 포지셔닝

```
          낮은 계산 비용 ←────────────────────→ 높은 계산 비용
               │                                        │
  낮은 분포    │  Constrained Gen.    AprAD    ASAp    │ 높은 분포
  충실도       │  FUDGE(fast)         │        SMC     │ 충실도
               │  Ctrl-G(fast)        │                │
               └────────────────────────────────────────┘
```

AprAD는 **추가 학습 없이** 실용적인 균형점을 제공한다는 점에서 독특하다. FUDGE나 Ctrl-G는 높은 분포 충실도와 낮은 계산 비용을 동시에 달성하지만, **태스크별 학습과 제약의 표현 가능성**이라는 전제 조건을 요구한다.

---

## 참고 자료

**논문 본문 (직접 인용)**:
- Melcer, D., Gonugondla, S., Perera, P., Qian, H., et al. (2025). *Approximately Aligned Decoding*. 39th Conference on Neural Information Processing Systems (NeurIPS 2025). arXiv:2410.01103v2

**논문 내 참조 문헌**:
- Park, K., Wang, J., Berg-Kirkpatrick, T., Polikarpova, N., & D'Antoni, L. (2024). *Grammar-Aligned Decoding*. arXiv:2405.21047
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. arXiv:2211.17192
- Lew, A. K., Zhi-Xuan, T., Grand, G., & Mansinghka, V. K. (2023). *Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs*. arXiv:2306.03081
- Yang, K., & Klein, D. (2021). *FUDGE: Controlled Text Generation With Future Discriminators*. NAACL 2021
- Zhang, H., Kung, P.-N., Yoshida, M., Van den Broeck, G., & Peng, N. (2024). *Adaptable Logical Control for Large Language Models* (Ctrl-G). arXiv:2406.13892
- Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*. arXiv:2401.10774
- Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024). *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*. arXiv:2401.15077
- Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024). *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*. arXiv:2406.16858
- Ahmed, K., Chang, K.-W., & Van den Broeck, G. (2025). *Controllable Generation via Locally Constrained Resampling* (Gen-C). ICLR 2025
- Geng, S., Josifoski, M., Peyrard, M., & West, R. (2024). *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*. arXiv:2305.13971
- Zhuo, T. Y., et al. (2024). *BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions*. arXiv:2406.15877
- Lozhkov, A., Li, R., Ben Allal, L., et al. (2024). *StarCoder 2 and The Stack v2: The Next Generation*. arXiv:2402.19173
- Jiang, A. Q., et al. (2023). *Mistral 7B*. arXiv:2310.06825
- Tran-Thien, V. (2024). *An Optimal Lossy Variant of Speculative Decoding*. HuggingFace Blog

**코드 저장소**:
- https://github.com/amazon-science/Approximately-Aligned-Decoding
