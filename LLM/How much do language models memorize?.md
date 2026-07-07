# How much do language models memorize?

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Morris et al. (2025)은 언어 모델의 암기(memorization)를 정보이론적으로 엄밀하게 정량화하는 새로운 프레임워크를 제안합니다. 핵심 주장은 다음과 같습니다:

1. **암기를 두 성분으로 분리**: *비의도적 암기(unintended memorization)* — 모델이 특정 데이터셋에 대해 보유한 정보 — 와 *일반화(generalization)* — 모델이 진정한 데이터 생성 과정에 대해 획득한 정보 — 로 명확히 구분
2. **GPT 계열 모델의 용량**: GPT 계열 모델은 파라미터당 약 $\alpha \approx 3.6$ bits의 정보를 저장할 수 있음
3. **Grokking의 메커니즘 설명**: 모델이 용량 한계에 도달하면 비의도적 암기 대신 일반화가 시작됨 (Grokking)
4. **멤버십 추론 스케일링 법칙**: 모델 용량과 데이터셋 크기를 기반으로 멤버십 추론 성능을 예측하는 스케일링 법칙 제시

### 주요 기여

| 기여 | 내용 |
|------|------|
| 이론적 기여 | Shannon 정보이론 + Kolmogorov 복잡도 기반의 엄밀한 암기 정의 |
| 실험적 기여 | 500K~1.5B 파라미터 수백 개 트랜스포머 모델 학습 |
| 실용적 기여 | 멤버십 추론 F1 예측 스케일링 법칙 |
| 통찰 기여 | Double descent와 grokking 현상을 용량-데이터 비율로 설명 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 연구들은 암기를 두 가지 방식으로 접근했습니다:

- **추출(extraction) 기반**: 모델이 학습 데이터를 생성할 수 있으면 암기로 판단
- **멤버십 추론(membership inference) 기반**: 데이터가 학습셋에 포함되었는지 분류

**문제점**: 두 접근 모두 **암기와 일반화를 구별하지 못함**. 예를 들어, $2^{100}$의 답을 출력한다고 해서 이것이 암기인지 수학 능력의 일반화인지 판단 불가.

### 2.2 제안 방법 (수식 포함)

#### Step 1: Shannon 정보이론 기반 통계적 정의

학습 알고리즘 $L$이 데이터셋 $x \sim X$로부터 모델 $\hat{\theta} = L(x)$를 학습할 때:

**전체 암기(Total memorization)**:

$$\text{mem}(X, \hat{\Theta}) = I(X, \hat{\Theta}) = H(X) - H(X \mid \hat{\Theta})$$

**비의도적 암기(Unintended memorization)** — 일반화를 제거한 암기:

$$\text{mem}_U(X, \hat{\Theta}, \Theta) = I([X \mid \Theta], \hat{\Theta}) = H(X \mid \Theta) - H(X \mid (\Theta, \hat{\Theta}))$$

**일반화(Generalization, 의도적 암기)**:

$$\text{mem}_I(X, \hat{\Theta}, \Theta) = \text{mem}(X, \hat{\Theta}) - \text{mem}_U(X, \hat{\Theta}, \Theta)$$

여기서 $\Theta$는 진정한 데이터 생성 과정을 근사하는 참조(reference) 모델입니다.

**Proposition 1 (비의도적 암기의 초가산성)**:

$$\sum_{i \in [n]} \text{mem}_U(X_i, \hat{\Theta}, \Theta) \leq \text{mem}_U(X, \hat{\Theta}, \Theta) \leq H(\hat{\Theta})$$

이는 비의도적 암기의 하한과 상한을 제공하며, **암기량은 모델의 총 용량을 초과할 수 없음**을 보장합니다.

#### Step 2: Kolmogorov 복잡도 기반 인스턴스 수준 정의

Shannon 엔트로피는 단일 인스턴스에 적용이 불가하므로, Kolmogorov 복잡도로 전환:

$$H^K(x) = \min_{f(p)=x} |p|, \quad H^K(x \mid \theta) = \min_{f(p,\theta)=x} |p|$$

$$I^K(x, \theta) = H^K(x) - H^K(x \mid \theta)$$

**Definition 3 (Kolmogorov 암기)**:

$$\text{mem}^K_U(x, \theta, \hat{\theta}) = H^K(x \mid \theta) - H^K(x \mid (\theta, \hat{\theta}))$$

$$\text{mem}^K_I(x, \theta, \hat{\theta}) = \text{mem}^K(x, \hat{\theta}) - \text{mem}^K_U(x, \theta, \hat{\theta})$$

**Proposition 4** (Kolmogorov 암기 ≈ Shannon 암기):

$$\left| \mathbb{E}_{\substack{x \sim X \\ \hat{\theta} \sim L(x)}} \left[\text{mem}^K_U(x_i, \hat{\theta}, \theta)\right] - \text{mem}_U(X_i, \hat{\Theta}, \theta) \right| \leq \epsilon$$

여기서 $\epsilon$은 $\ell$, $\ell'$, $n$에 독립적인 상수입니다.

#### Step 3: 산술 코딩(Arithmetic Coding)으로 실용적 추정

Kolmogorov 복잡도는 계산 불가능하므로, 산술 코딩으로 근사:

$$H^K(x \mid \hat{\theta}) \approx -\log p(x \mid \hat{\theta})$$

$$H^K(x \mid \hat{\theta}, \theta) \approx -\log \max\{p(x \mid \hat{\theta}),\ p(x \mid \theta)\}$$

따라서 비의도적 암기의 실용적 추정:

$$\text{mem}^K_U(x, \theta, \hat{\theta}) \approx -\log p(x \mid \theta) + \log \max\{p(x \mid \hat{\theta}),\ p(x \mid \theta)\}$$

#### Step 4: 모델 용량 정의

$$\text{Capacity}(L) = \max_X \text{mem}(X, L(X))$$

합성 랜덤 데이터에 대한 예상 암기량:

$$\text{mem}(X, L(X)) \approx \min(\text{capacity}(L),\ H(X))$$

#### Step 5: 멤버십 추론 스케일링 법칙

$$\text{Membership}_{F_1}(\theta, \mathcal{D}) = \frac{1}{2}\left(1 + c_1 \sigma\!\left(c_2\!\left(\frac{\text{Capacity}(\theta)}{|\mathcal{D}|} + c_3\right)\right)\right)$$

여기서 $\sigma(x) = \frac{1}{1+e^{-x}}$, 실험적으로 $c_1 = 1.34$, $c_2 = -0.034$, $c_3 = -33.14$.

### 2.3 모델 구조

| 항목 | 세부 내용 |
|------|-----------|
| 아키텍처 | GPT-2 스타일 트랜스포머 (Kaplan et al., 2020 기준) |
| 파라미터 범위 | 500K ~ 1.5B (메인 실험: 100K ~ 20M) |
| 레이어 수 | 1, 2, 4, 8 레이어 |
| 히든 차원 | $d_\text{model}$: 32, 64, 128, 256, 512 |
| 어휘 크기 | $V = 2048$ (기본값) |
| 시퀀스 길이 | $S = 64$ 토큰 |
| 학습 설정 | Adam 옵티마이저, 배치 크기 2048, A100 GPU, bfloat16 정밀도 |
| 데이터셋 | 합성: 균등 랜덤 비트스트링 / 텍스트: FineWeb (Penedo et al., 2024) |
| 참조 모델 | 합성: 실제 균등 분포 / 텍스트: 동일 계열 최대 데이터 학습 모델 |

### 2.4 주요 성능 결과

**용량 측정**:
- GPT 계열 모델: $\alpha \approx 3.51$ bpp (bfloat16), $\alpha \approx 3.83$ bpp (float32)
- 16비트 → 32비트 정밀도 2배 증가 시 용량은 약 9% 증가에 불과

**스케일링 법칙 검증 (Table 2)**:

| 모델 | 예측 F1 | 관측 F1 |
|------|---------|---------|
| GPT2-XL (1.5B) | 0.55 | 54.61 ± 1.3 |
| GPT2-XL (1.5B) | 0.75 | 71.08 ± 0.4 |
| GPT2-XL (1.5B) | 0.95 | 95.85 ± 0.8 |
| GPT2-Medium (125M) | 0.55 | 53.44 ± 1.1 |
| GPT2-Medium (125M) | 0.95 | 97.98 ± 0.3 |

예측 오차: 관측값의 약 1~2% 이내.

### 2.5 한계 (Limitations)

1. **실험 환경 특수성**: GPT-2 아키텍처, FineWeb 데이터, 특정 학습 설정에 한정. 다른 아키텍처(예: MoE, State Space Model)나 데이터셋으로의 일반화 불확실
2. **하한 측정**: 경사 하강법이 전역 최적해를 보장하지 않으므로, 측정된 용량은 실제 용량의 **하한**
3. **산술 코딩의 불완전성**: Kolmogorov 복잡도의 진정한 추정값보다 느슨할 수 있음
4. **참조 모델 선택 의존성**: 참조 모델의 품질이 비의도적 암기 측정에 직접 영향
5. **짧은 시퀀스 한정**: 64토큰 시퀀스 실험으로, 긴 문서 암기 특성은 별도 연구 필요
6. **시그모이드 단순성**: 멤버십 추론 스케일링 법칙의 시그모이드 함수가 중간 범위(F1 ≈ 0.75)에서 오차 증가

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 핵심 발견: 용량-데이터 전환점

이 논문의 가장 중요한 일반화 관련 발견은 **암기에서 일반화로의 전환 메커니즘**입니다.

$$\text{mem}(X, L(X)) \approx \min(\text{Capacity}(L),\ H(X))$$

- 데이터셋 크기 $|D| <$ 모델 용량: 모델은 각 샘플을 개별적으로 암기
- 데이터셋 크기 $|D| >$ 모델 용량: 모델은 개별 암기 대신 **공유 가능한 일반화 패턴** 학습 시작

### 3.2 Grokking과 Double Descent의 통합 설명

**Double descent의 원인 규명**:

$$\text{데이터 용량} > \text{모델 용량} \Rightarrow \text{Double Descent 시작}$$

논문은 Double descent가 데이터 크기가 모델 용량(비트 단위)을 초과하는 정확한 시점에 발생함을 실험적으로 확인했습니다. 이는 기존의 현상 설명에서 나아가 **정량적 예측**을 가능하게 합니다.

**Grokking 메커니즘**:

```
[학습 과정]
초기: 모델 용량 미만 데이터 → 완전 암기
전환: 용량 한계 도달 → Grokking 시작
이후: 개별 샘플 암기 감소, 일반화 패턴 증가
결과: 테스트 손실 급격히 감소
```

Figure 2에서 보이듯, 오라클 참조 모델 대비 비의도적 암기는 용량 한계 이후 급격히 감소하며 이는 일반화의 증가와 일치합니다.

### 3.3 일반화 가능성을 높이는 조건

논문이 시사하는 일반화 향상 조건:

**1. 데이터셋 크기와 모델 용량의 최적 비율**:

현대 LLM은 대부분 $\frac{|\mathcal{D}|}{\text{Capacity}(\theta)} \gg 10^2$ 조건에서 학습되므로, 스케일링 법칙에 따르면 멤버십 추론 F1 ≈ 0.5 (무작위 수준) → **사실상 개별 데이터 암기 없이 순수 일반화** 달성

**2. 데이터 중복 제거의 중요성**:

실험에서 완벽한 중복 제거(perfect deduplication) 후 추출률이 거의 테스트 추출률 수준으로 수렴:

$$\lim_{|D| \to \infty} \text{Extraction Rate}_{\text{train}} \approx \text{Extraction Rate}_{\text{test}}$$

이는 충분히 큰 중복 제거된 데이터셋에서 **훈련 데이터 추출이 전적으로 일반화에 기인**함을 의미합니다.

**3. 희귀 토큰 데이터의 암기 집중**:

TF-IDF 분석 결과, 희귀 토큰(특히 비영어 언어)이 포함된 샘플이 집중적으로 암기됨:

$$\text{TF-IDF}(d; \mathcal{D}) = \frac{1}{|d|}\sum_{w \in d} \log \frac{|\mathcal{D}|}{tf(w, \mathcal{D})}$$

이 발견은 **훈련 데이터에서 극단적으로 희귀한 샘플을 제거하거나 하향 샘플링**하면 일반화 효율을 높일 수 있음을 시사합니다.

### 3.4 일반화 성능 향상을 위한 실용적 제안

논문의 발견으로부터 도출되는 일반화 향상 전략:

| 전략 | 근거 |
|------|------|
| **토큰당 파라미터 수 감소** (더 많은 데이터로 학습) | 용량 비율 $\frac{\text{Capacity}}{ \mid D \mid }$ 감소 → 일반화 강제 |
| **완벽한 중복 제거** | 중복 데이터는 비의도적 암기를 비효율적으로 증가시킴 |
| **희귀 패턴 데이터 조정** | 고 TF-IDF 샘플의 하향 샘플링 또는 제거 |
| **더 큰 데이터셋 사용** | $\mid D \mid \gg \text{Capacity}(\theta)$ 달성 시 순수 일반화 |

---

## 4. 연구에 미치는 영향 및 향후 고려사항

### 4.1 앞으로의 연구에 미치는 영향

**① 암기 정량화 연구의 새로운 기준**

기존의 질적 접근("추출 가능하면 암기")에서 비트 단위의 **정량적 암기 측정**으로 패러다임 전환을 이끌 것입니다. 이 프레임워크는 향후 암기 관련 연구의 공통 평가 기준으로 채택될 가능성이 높습니다.

**② 프라이버시 및 보안 연구**

스케일링 법칙 결과는 대규모 LLM에서 개별 데이터 포인트의 멤버십 추론이 사실상 불가능함을 보여줍니다. 이는:
- GDPR 준수 연구에서 "잊혀질 권리(right to be forgotten)"의 실효성 재검토
- Machine Unlearning 연구에서 실제 프라이버시 위험 재평가
- 차분 프라이버시(differential privacy) 적용 필요성의 재조정

**③ 스케일링 법칙 연구의 확장**

Kaplan et al. (2020)의 손실 기반 스케일링 법칙에 **암기 기반 스케일링 법칙**을 추가. 향후 최적 컴퓨팅 배분(optimal compute allocation) 연구 시 프라이버시 고려가 포함될 수 있습니다.

**④ 데이터 효율 연구**

용량-데이터 전환점 발견은 데이터 큐레이션 연구를 더욱 중요하게 만듭니다. 어떤 데이터를 얼마나 사용해야 최적 일반화를 달성하는지에 대한 이론적 근거를 제공합니다.

**⑤ 해석 가능성 연구**

모델이 어떤 정보를 어떻게 저장하는지(파라미터당 3.6비트)에 대한 정량적 이해는 mechanistic interpretability 연구에도 새로운 관점을 제공합니다.

### 4.2 향후 연구 시 고려할 점

**① 아키텍처 다양화**

현재 결과는 GPT-2 스타일에 한정됩니다. 향후 연구 시:
- Mixture of Experts (MoE): 활성화 파라미터 수 vs. 전체 파라미터 수 중 어느 것이 용량을 결정하는가?
- State Space Models (Mamba, etc.): 어텐션 없는 아키텍처의 용량 특성
- 멀티모달 모델: 텍스트와 이미지 데이터의 암기 용량 차이

**② 더 정교한 Kolmogorov 복잡도 추정**

논문 자체가 인정하듯, 산술 코딩은 Kolmogorov 복잡도의 상한 추정치입니다. 향후:
- 훈련 데이터에 특화된 압축 알고리즘 개발
- 프롬프트 최적화(Schwarzschild et al., 2024)와의 통합
- 더 타이트한 Kolmogorov 하한 추정

**③ 학습 알고리즘 의존성**

현재 정의는 최종 모델 $\hat{\theta}$에만 의존하지만, 실제 암기는 학습 알고리즘(SGD vs. Adam), 학습률 스케줄, 배치 크기에도 영향받을 수 있습니다.

**④ 파인튜닝 및 RLHF 환경**

사전학습이 아닌 파인튜닝(fine-tuning), RLHF, Instruction tuning 환경에서의 암기 특성은 별도 연구 필요. 파인튜닝 시 사전학습 용량이 어떻게 재배분되는지 미지수.

**⑤ 연속 학습 및 망각**

모델이 순차적으로 새로운 데이터를 학습할 때 용량이 어떻게 재배분되는지, catastrophic forgetting과 암기 용량의 관계 연구 필요.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 접근법 | Morris et al.과의 비교 |
|------|------|--------|----------------------|
| **Carlini et al., "Quantifying memorization across neural language models"** (arXiv:2202.07646) | 2023 | 추출 기반: $k$-extractability 정의 | 일반화와 암기 미구분. Morris et al.은 이를 정보이론적으로 분리 |
| **Nasr et al., "Scalable extraction of training data from (production) language models"** (arXiv:2311.17035) | 2023 | 대규모 LLM에서 훈련 데이터 추출 공격 | 추출 가능성 = 암기로 정의. Morris et al.은 추출 없이도 암기 측정 가능 |
| **Schwarzschild et al., "Rethinking LLM memorization through the lens of adversarial compression"** (arXiv:2404.15146) | 2024 | 프롬프트 최적화로 압축률 측정 | Morris et al.과 압축 아이디어 공유하나, 일반화 미고려 |
| **Allen-Zhu & Li, "Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws"** (arXiv:2404.05405) | 2024 | 양자화 기반 용량 추정, ~2 bpp | Morris et al.: 3.6 bpp (더 높은 추정치, 엔트로피 기반 측정이 더 직접적) |
| **Zhang et al., "Counterfactual memorization in neural language models"** (arXiv:2112.12938) | 2023 | 반사실적 영향 측정: 특정 샘플 제거 시 예측 변화량 | Morris et al. 관점에서 참조 모델로 동일 계열 모델 사용하는 것과 유사 |
| **Duan et al., "Do membership inference attacks work on large language models?"** (COLM 2024) | 2024 | 대규모 LLM에서 MI 공격 실패 사례 | Morris et al. 스케일링 법칙이 이를 수학적으로 예측 및 설명 |
| **Das et al., "Blind baselines beat membership inference attacks for foundation models"** (arXiv:2406.16201) | 2024 | 단순 베이스라인이 MI 공격 능가 | Morris et al.: 데이터/용량 비율이 충분히 크면 MI 근본적 불가 |
| **Prashanth et al., "Recite, Reconstruct, Recollect: Memorization in LMs as a Multifaceted Phenomenon"** (arXiv:2406.17746) | 2024 | 암기를 다면적 현상으로 분류 | Morris et al.은 단일 정보이론적 척도로 통합 |
| **Nakkiran et al., "Deep Double Descent"** (arXiv:1912.02292) | 2019 | Double descent 현상 발견 | Morris et al.: 용량-데이터 비율로 정확한 발생 시점 예측 (확장) |
| **Delétang et al., "Language modeling is compression"** (arXiv:2309.10668) | 2024 | LLM을 범용 압축기로 활용 | Morris et al.: 압축률을 암기 측정에 역방향으로 활용 |

### 비교 분석 요약

Morris et al. (2025)의 가장 큰 차별점은:

1. **암기와 일반화의 정보이론적 분리** (기존 연구 대부분 미구분)
2. **인스턴스 수준 측정** + **데이터셋 수준 집계** 모두 가능
3. **학습 알고리즘 독립적** 정의 (최종 모델과 샘플만으로 측정)
4. **예측적 스케일링 법칙** (단순 현상 설명 → 정량적 예측)

---

## 참고 자료

**주요 논문 (본문에서 직접 인용)**:

- Morris, J. X., Sitawarin, C., Guo, C., Kokhlikyan, N., Suh, G. E., Rush, A. M., Chaudhuri, K., & Mahloujifar, S. (2025). *How much do language models memorize?* arXiv:2505.24832v3.
- Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramer, F., & Zhang, C. (2023). *Quantifying memorization across neural language models.* arXiv:2202.07646.
- Nasr, M., et al. (2023). *Scalable extraction of training data from (production) language models.* arXiv:2311.17035.
- Schwarzschild, A., Feng, Z., Maini, P., Lipton, Z. C., & Kolter, J. Z. (2024). *Rethinking LLM memorization through the lens of adversarial compression.* arXiv:2404.15146.
- Allen-Zhu, Z., & Li, Y. (2024). *Physics of language models: Part 3.3, knowledge capacity scaling laws.* arXiv:2404.05405.
- Nakkiran, P., et al. (2019). *Deep double descent: Where bigger models and more data hurt.* arXiv:1912.02292.
- Delétang, G., et al. (2024). *Language modeling is compression.* arXiv:2309.10668.
- Zhang, C., et al. (2023). *Counterfactual memorization in neural language models.* arXiv:2112.12938.
- Duan, M., et al. (2024). *Do membership inference attacks work on large language models?* COLM 2024.
- Das, D., Zhang, J., & Tramèr, F. (2024). *Blind baselines beat membership inference attacks for foundation models.* arXiv:2406.16201.
- Prashanth, U. S., et al. (2024). *Recite, reconstruct, recollect: Memorization in LMs as a multifaceted phenomenon.* arXiv:2406.17746.
- Kaplan, J., et al. (2020). *Scaling laws for neural language models.* arXiv:2001.08361.
- Brown, G., et al. (2021). *When is memorization of irrelevant training data necessary for high-accuracy learning?* STOC 2021.
- Shannon, C. E. (1948). *A Mathematical Theory of Communication.*
- Kolmogorov, A. N. (1965). *Three approaches to the quantitative definition of information.* Problems of Information Transmission.
- Grunwald, P., & Vitanyi, P. (2004). *Shannon information and Kolmogorov complexity.* arXiv:cs/0410002.
- Penedo, G., et al. (2024). *The FineWeb datasets: Decanting the web for the finest text data at scale.* arXiv:2406.17557.
- Belkin, M., et al. (2019). *Reconciling modern machine-learning practice and the classical bias-variance trade-off.* PNAS.
- Feldman, V. (2020). *Does learning require memorization? A short tale about a long tail.* STOC 2020.
