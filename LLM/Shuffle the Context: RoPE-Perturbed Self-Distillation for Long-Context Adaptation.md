# Shuffle the Context: RoPE-Perturbed Self-Distillation for Long-Context Adaptation

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

### 핵심 주장

표준 장문맥 파인튜닝(Long-Context Fine-Tuning)은 **위치적으로 취약(positionally brittle)**하다. 즉, 모델의 정확도가 관련 증거가 프롬프트 내 어느 절대적 위치에 놓이는가에 따라 크게 변동한다. 이 논문은 RoPE 인덱스를 교란(perturb)하여 대안적 "뷰(view)"를 생성하고, Self-Distillation을 통해 두 뷰 간의 예측 일관성을 강제함으로써 위치적 강건성을 획득하는 **RoPE-Perturbed Self-Distillation** 방법을 제안한다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **문제 진단** | 장문맥 파인튜닝 후에도 모델이 위치적 취약성(positional brittleness)을 보임을 실증적으로 규명 |
| **방법론 제안** | RoPE 인덱스 교란 + KL 기반 자기 증류(self-distillation) 프레임워크 제안 |
| **성능 향상** | Llama-3-8B: RULER-64K에서 최대 **+12.04%**, Qwen-3-4B: RULER-256K에서 **+2.71%** 향상 |
| **길이 외삽 개선** | 훈련 컨텍스트 창 이상의 길이에서도 강건성 유지(YaRN 기반 외삽 시 일관된 성능 향상) |
| **단문맥 성능 보존** | 장문맥 향상이 단문맥 능력의 희생 없이 이루어짐을 확인 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**위치적 취약성(Positional Brittleness):**

기존 장문맥 파인튜닝은 NIAH(Needle-In-A-Haystack) 등 합성 평가에서 좋은 성능을 보이지만, **동일한 증거 스팬이 프롬프트 내 다른 절대적 위치에 놓일 경우 정확도가 크게 달라지는 문제**가 존재한다. 이는 RAG, 코드베이스 연결, 다중 문서 추론 등 실제 응용에서 치명적이다.

이러한 취약성의 근원은 **RoPE(Rotary Position Embedding)**의 위치 의존적 위상(phase)에 있다. RoPE는 쿼리와 키 벡터를 위치 의존적 각도로 회전시키므로, 장문맥에서 모델 행동이 위치 인덱스 할당 방식에 민감해진다.

### 2.2 제안하는 방법 (수식 포함)

#### 기본 설정

길이 $L$의 토큰 시퀀스 $x_{0:L-1} = (x_0, \ldots, x_{L-1})$에 대해, RoPE 인덱스 벡터 $r = (r_0, \ldots, r_{L-1}) \in \mathbb{Z}^L$을 명시적으로 조작한다.

**표준 CLM 손실:**

$$r_i = i, \quad i = 0, \ldots, L-1 $$

$$\mathcal{L}_{\text{CLM}}(\theta) = \mathbb{E}_{x_{0:L-1} \sim \mathcal{D}} \left[ -\sum_{i=0}^{L-1} \log p_\theta(x_i \mid x_{<i}; r) \right] $$

#### Skip-based RoPE 인덱스 교란

분할점 $s$와 스킵 길이 $y$로 매개변수화된 RoPE 인덱스 할당:

$$r_i^{(s,y)} = \begin{cases} i, & i < s \\ i + y, & i \geq s \end{cases}, \quad i = 0, \ldots, L-1 $$

여기서 $s \in \{0, \ldots, L-1\}$, $y \in \{1, \ldots, Y\}$. 기본 실험에서는 $Y = L$로 설정하고 균등 분포에서 $(s, y)$를 샘플링한다.

**직관:** 이 교란은 인덱스 공간에 "갭(gap)"을 생성하여, 접미사 $[s, L-1]$의 토큰들이 접두사 $[0, s-1]$을 RoPE 공간에서 더 멀리 있는 것으로 인식하게 만든다.

#### Cyclic-shift 교란 (변형)

$$r_i^{\text{cyc}(u)} = (i + u) \bmod L, \quad i = 0, \ldots, L-1 $$

#### 자기 증류 손실 (Self-Distillation Loss)

두 뷰(표준 뷰, 교란 뷰)를 정의한다:

- **표준 뷰:** $p_\theta^{(0,0)}(\cdot \mid x_{<i}) = p_\theta(\cdot \mid x_{<i}; r^{(0,0)})$
- **교란 뷰:** $p_\theta^{(s,y)}(\cdot \mid x_{<i}) = p_\theta(\cdot \mid x_{<i}; r^{(s,y)})$

위치 $i$에서의 역방향 KL 발산(reverse KL divergence):

```math
\ell_i^{(s,y)}(\theta) = \text{KL}\!\left(p_\theta^{(s,y)}(\cdot \mid x_{ < i}) \;\Big\|\; \text{sg}\!\left(p_\theta^{(0,0)}(\cdot \mid x_{ < i})\right)\right)
```

여기서 $\text{sg}(\cdot)$는 stop-gradient 연산자로, 그래디언트가 교란 경로를 통해서만 흐르게 한다. 표준 뷰는 안정적인 교사(teacher) 역할을 한다.

위치 $i < s$에서는 두 뷰가 동일하므로 $\ell_i^{(s,y)}(\theta) = 0$이고, 따라서 증류 손실은 **접미사 위치에만 적용**된다:

$$\mathcal{L}_{\text{distill}}(x_{0:L-1}; \theta) = \mathbb{E}_{(s,y) \sim q(s,y)} \left[ \frac{1}{L-s} \sum_{i=s}^{L-1} \ell_i^{(s,y)}(\theta) \right] $$

$$\mathcal{L}_{\text{KL}}(\theta) = \mathbb{E}_{x_{0:L-1} \sim \mathcal{D}} \left[ \mathcal{L}_{\text{distill}}(x_{0:L-1}; \theta) \right] $$

#### 전체 목적 함수

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{CLM}}(\theta) + \lambda \mathcal{L}_{\text{KL}}(\theta) $$

실험에서는 $\lambda = 1$을 사용한다. 최적 성능은 $\lambda \in [0.5, 1]$ 범위에서 얻어진다.

### 2.3 모델 구조

별도의 새로운 아키텍처를 도입하지 않으며, **기존 Transformer + RoPE 구조를 그대로 유지**한다.

```
훈련 시:
┌─────────────────────────────────────────┐
│  동일 토큰 시퀀스 x_{0:L-1}              │
│                                         │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │ 표준 뷰      │  │ 교란 뷰          │ │
│  │ r^(0,0)     │  │ r^(s,y)          │ │
│  │ → L_CLM    │  │ → L_KL (역KL)    │ │
│  └──────────────┘  └──────────────────┘ │
│           ↓                             │
│    L_total = L_CLM + λ·L_KL            │
└─────────────────────────────────────────┘

평가 시:
→ 표준 인덱스 r^(0,0) 만 사용 (단일 모델)
```

**계산 오버헤드:** 배치당 추가 순전파(forward pass) 1회 → 약 1.6× 벽시계 시간(wall-clock time) 증가

### 2.4 성능 향상

#### Llama-3-8B-Instruct (64K 컨텍스트)

| 방법 | RULER 32K | RULER 64K | HELMET Avg. |
|------|-----------|-----------|-------------|
| Standard | 85.23 | 57.25 | 44.30 |
| LongCE | 86.11 | 52.51 | 45.84 |
| PoSE | 87.20 | 67.47 | 45.65 |
| Ours (cyclic) | 87.33 | **71.30** | 46.12 |
| Ours (skip) | **87.87** | 69.29 | **47.88** |

→ Standard 대비 RULER-64K 최대 **+14.05p** 향상

#### Qwen-3-4B (256K 컨텍스트)

| 방법 | RULER 128K | RULER 256K | HELMET Avg. |
|------|------------|------------|-------------|
| Standard | 74.45 | 67.01 | 51.80 |
| Ours (skip) | 77.51 | 68.10 | 52.99 |
| Ours (skip) + LongCE | 77.50 | **68.95** | **53.59** |

#### 길이 외삽 (YaRN 기반)

| 방법 | Llama 128K | Llama 256K | Qwen 512K | Qwen 1M |
|------|-----------|-----------|----------|---------|
| Standard | 42.03 | 23.90 | 58.11 | 45.31 |
| Ours (skip) | 59.43 | 44.74 | 60.73 | **50.41** |

### 2.5 한계

1. **계산 비용:** 배치당 추가 순전파 필요 → 약 1.6× 훈련 시간 증가
2. **단일 모달리티 검증:** 텍스트 전용 모달리티에서만 검증되어, 멀티모달 장문맥 시나리오로의 확장 가능성은 미검증
3. **Cyclic shift의 순서 민감 태스크 취약성:** Cyclic shift 변형은 순서 민감 태스크(ordinal retrieval)에서 성능 저하 가능
4. **제한적 모델 실험:** Llama-3-8B와 Qwen-3-4B 두 모델에 대해서만 검증; 더 큰 규모(70B+) 모델로의 확장성 미확인
5. **교란 설계의 태스크 의존성:** 최적 교란 방식이 태스크 유형에 따라 달라질 수 있어, 범용 교란 설계가 어려울 수 있음
6. **RoPE 이외의 PE 메커니즘:** 다른 위치 인코딩(예: ALiBi)에 대한 적용 가능성은 이론적으로 언급되나 실험적으로 검증되지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화 성능 향상과 직접적으로 연관되는 요소들을 중점적으로 분석한다.

### 3.1 위치적 일반화 (Positional Generalization)

**"Lost-in-the-Middle" 현상 완화:**

Liu et al. (2024)이 보고한 중간 위치 정확도 저하 현상에 대해, 본 논문의 방법은 이를 실질적으로 완화한다. Qwen-3-4B의 RULER NIAH MultiKey-2 실험에서 표준 CLM 훈련은 U-자형 정확도 곡선(중간 구간 저하)을 보이는 반면, 제안 방법은 **모든 위치에서 고른 정확도**를 달성한다.

**RoPE 다주파수 구조의 활용:**

RoPE의 고주파수(high-frequency) 성분은 위치에 민감하게 변화하고, 저주파수(low-frequency) 성분은 거칠고 완만하게 변화한다. 인덱스 이동은 주로 고주파수 성분을 교란하면서 거친 구조는 상대적으로 안정되게 유지한다. 두 뷰 간의 일관성 강제는 모델로 하여금:

$$\text{위치에 민감한 고주파수 신호} \xrightarrow{\text{감소}} \text{의미론적 신호 + 거친 위치 구조 활용 증가}$$

이 과정이 일반화 성능 향상의 핵심 메커니즘이다.

### 3.2 길이 외삽 일반화 (Length Extrapolation Generalization)

훈련 컨텍스트 창(64K/256K)을 초과하는 길이(128K, 256K, 512K, 1M)에서의 YaRN 기반 외삽 실험은 다음을 보여준다:

- Standard 파인튜닝은 **외삽 시 급격한 성능 저하** (Llama: 64K→256K에서 57.25→23.90)
- 제안 방법은 **외삽 범위에서 일관되게 최고 성능** 유지

이는 RoPE 인덱스 교란에 대한 불변성(invariance)이 훈련 창 이상의 길이 일반화로 **전이(transfer)**됨을 시사한다.

### 3.3 SFT 이후에도 유지되는 일반화

실용적 파이프라인에서는 장문맥 지속 사전훈련(continued pretraining) 후 SFT가 적용된다. Tulu-v3 데이터로 SFT 적용 후에도:

| 방법 | RULER 128K | RULER 256K | HELMET | LongBench-v2 |
|------|------------|------------|--------|--------------|
| Standard (SFT 후) | 75.52 | 64.44 | 52.61 | 27.63 |
| Ours (skip, SFT 후) | 77.28 | 66.86 | 53.48 | 29.40 |
| Ours+LongCE (SFT 후) | **78.68** | **67.15** | **54.02** | **32.22** |

→ SFT에 의해 이득이 희석되지 않고 오히려 약간 강화됨. 이는 **표면적 성능 향상이 아닌 내재적 장문맥 능력의 향상**임을 시사한다.

### 3.4 태스크 일반화 (Task Generalization)

**다양한 장문맥 태스크에서의 일관된 향상:**

- RULER: NIAH, Multi-key retrieval, Multi-value, Multi-query, QA 등 다양한 서브태스크에서 향상
- HELMET: RAG, ICL, LongQA, Rerank 등 실용적 태스크에서 향상
- LongBench-v2: 보다 현실적인 장문맥 멀티태스크에서 향상

**단문맥 능력 유지:**
MMLU, HellaSwag, WinoGrande, OpenBookQA에서 Standard와 동등한 성능 유지 → **장단문맥 간 트레이드오프 없는 일반화**

### 3.5 Mix-length 훈련에서의 일반화

고정 길이 훈련뿐 아니라 **ProLong 스타일 혼합 길이(mix-length) 훈련**에서도 제안 방법이 Standard를 일관되게 상회함(RULER 64K: 52.48 → 65.28)을 확인했다. 이는 고정 길이 설정에 특화된 것이 아닌 **현실적 훈련 레짐에서도 적용 가능**함을 보여준다.

### 3.6 어텐션 패턴 분석을 통한 일반화 메커니즘

64K 토큰 시퀀스에 대한 어텐션 분석(24번째 레이어)에서, 제안 방법으로 훈련된 모델은:

- **장거리 어텐션 질량(long-range attention mass)을 더 많이 할당**
- 장거리 어텐션 분포가 **더 균일(uniform)**함

→ 모델이 가까운 위치의 토큰에만 의존하지 않고 장거리 의미론적 관계를 적극 활용하도록 일반화된 것으로 해석된다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 위치 인코딩 확장 연구

| 연구 | 방법 | 특징 | 본 논문과의 관계 |
|------|------|------|----------------|
| **YaRN** (Peng et al., 2024, ICLR 2024) | RoPE 주파수 보간/스케일링 | 컨텍스트 창 확장 시 OOD 행동 완화 | 평가 시 YaRN을 활용; 본 논문은 훈련 목적함수 수준에서 직교적 접근 |
| **LongRoPE** (Ding et al., 2024, ICML 2024) | 주파수 리스케일링 유연화 | 2M 토큰 이상 확장 | RoPE 메커니즘 자체를 변경 vs. 본 논문은 RoPE 유지하면서 훈련 강건성 향상 |
| **ABF/NTK 스케일링** (Xiong et al., 2024, NAACL 2024) | RoPE 베이스 증가 | 외삽 OOD 완화 | 본 논문의 훈련에서도 ABF 사용; 직교적 조합 가능 |
| **NoPE** (Kazemnejad et al., 2023, NeurIPS 2023) | 위치 인코딩 제거 | 길이 일반화 향상 주장 | 본 논문 ablation에서 NoPE를 교란 뷰에 적용 시 성능 대폭 저하 확인 |
| **RoPE to NoPE** (Yang et al., 2025) | RoPE-NoPE 하이브리드 | 두 방식의 보완적 활용 | 아키텍처 변경 vs. 훈련 목적함수 수준 접근 |

### 4.2 장문맥 학습 방법 연구

| 연구 | 방법 | 특징 | 본 논문과의 비교 |
|------|------|------|----------------|
| **ProLong** (Gao et al., 2025, ACL 2025) | 훈련 레시피 + 데이터 엔지니어링 | 효과적인 장문맥 파인튜닝 방법론 | 본 논문의 기준 베이스라인; 훈련 레시피 공유 |
| **PoSE** (Zhu et al., 2023) | 위치 스킵와이즈 훈련 | 교란된 위치 인덱스로 CLM 훈련 | 본 논문의 직접 비교 대상; KL 일관성 항 없이 교란 뷰만으로 CLM 수행 → 본 논문이 일관되게 우월 |
| **LongCE** (Fang et al., 2025, ICLR 2025) | 토큰 재가중치 CLM | 장문맥 혜택을 받는 토큰 강조 | 동기가 직교적(complementary)이어서 조합(Ours + LongCE)이 최강 성능 |
| **Data Engineering for 128K** (Fu et al., 2024) | 데이터 엔지니어링 | 훈련 데이터 구성 강조 | 데이터 레벨 vs. 목적함수 레벨 접근 |
| **LongRecipe** (Hu et al., 2025, ACL 2025) | 효율적 장문맥 일반화 레시피 | 위치 인덱스 조작 포함 | 위치 시뮬레이션 측면에서 유사하나, 본 논문은 일관성 강제로 차별화 |
| **GrowLength** (Jin et al., 2023) | 점진적 시퀀스 길이 증가 | 훈련 효율성과 안정성 | 커리큘럼 접근 vs. 목적함수 수준 접근 |

### 4.3 위치 편향 완화 연구

| 연구 | 방법 | 특징 | 본 논문과의 관계 |
|------|------|------|----------------|
| **Found in the Middle** (Zhang et al., 2024, NeurIPS 2024) | 플러그앤플레이 위치 인코딩 | 추론 시 위치 인코딩 재맵핑 | 추론 시 개입 vs. 훈련 목적함수 개입; 상호 보완적 |
| **Position Bias Mitigation** (Yu et al., 2025, ACL 2025) | 숨겨진 상태 채널 스케일링 | 위치 편향 추론 시 완화 | 추론 시 개입; 본 논문과 상호 보완적 |
| **An et al. (2024)** | 추론 시 RoPE 인덱스 재맵핑 | 잘 훈련된 위치로 매핑 | 추론 시 개입 vs. 훈련 시 정규화 |

### 4.4 효율적 어텐션 및 아키텍처

| 연구 | 방법 | 본 논문과의 관계 |
|------|------|----------------|
| **Longformer** (Beltagy et al., 2020, arXiv) | 희소 어텐션 | 효율성 중심; 위치 강건성과 직교적 |
| **BigBird** (Zaheer et al., 2020, NeurIPS 2020) | 구조적 어텐션 | 아키텍처 변경 vs. 훈련 목적함수 변경 |
| **LM-Infinite** (Han et al., 2024, NAACL 2024) | 추론 시 KV 유지/스트리밍 | 추론 효율성 중심; 보완적 |
| **StreamingLLM** (Xiao et al., 2023) | 어텐션 싱크 기반 스트리밍 | 효율성 중심; 위치 강건성과 직교적 |

### 4.5 무작위 위치 인코딩 접근법

| 연구 | 방법 | 본 논문과의 비교 |
|------|------|----------------|
| **Randomized Positional Encodings** (Ruoss et al., 2023) | 훈련 시 무작위 위치 인덱스 | 길이 일반화 목적; 일관성 강제 없이 단순 무작위화 → 본 논문은 일관성 강제 추가로 차별화 |
| **Middle-Focused PE** (Wu et al., 2024a, NeurIPS 2024) | 중간 집중 위치 인코딩 | "lost-in-the-middle" 완화 목적; 특정 위치 강화 vs. 위치 불변성 학습 |

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

**① 위치적 강건성 연구의 새로운 패러다임 제시**

기존 연구가 RoPE 확장 방식이나 데이터 엔지니어링에 집중했다면, 이 논문은 **훈련 목적함수 수준의 위치적 불변성 정규화**라는 새로운 방향을 제시한다. 이는 향후 위치 편향 완화 연구의 표준적 접근법 중 하나로 자리잡을 가능성이 있다.

**② Self-Distillation 기반 불변성 학습의 일반화**

동일한 Self-Distillation 프레임워크가 다른 유형의 불변성 학습(예: 다양한 문서 순서, 다양한 언어 표현)에도 적용 가능하다는 가능성을 제시한다.

**③ 복합 방법론 연구 촉진**

Ours + LongCE 조합이 개별 방법보다 강력함을 보임으로써, **직교적 목적함수들의 조합**이 유망한 연구 방향임을 시사한다.

**④ 길이 외삽 연구와의 연계**

RoPE 교란에 대한 불변성 학습이 길이 외삽 성능을 자연스럽게 향상시킨다는 발견은, 위치 인코딩 확장 연구(YaRN, LongRoPE 등)와의 통합 연구를 자극할 수 있다.

**⑤ RAG 및 다중 문서 시스템 개선**

위치적 강건성의 향상은 RAG 파이프라인, 코드베이스 분석, 다중 문서 QA 등 실용적 응용에서 직접적인 성능 개선을 가져올 것이다.

### 5.2 앞으로 연구 시 고려할 점

**① 더 큰 규모 모델로의 확장성 검증 필요**

현재 실험은 4B~8B 규모 모델에 한정되어 있다. 70B 이상의 모델에서도 동일한 효과가 있는지 검증이 필요하다. 특히 이미 더 강력한 장문맥 능력을 가진 대형 모델에서 위치적 취약성이 얼마나 남아있는지 분석이 선행되어야 한다.

**② 태스크 인식적 교란 분포 설계**

현재는 $(s, y)$를 균등 분포에서 샘플링하지만, 특정 태스크(예: RAG에서의 검색 위치 분포, 코드 분석에서의 청크 구조)에 맞춘 **태스크 인식적(task-aware) 교란 분포**가 성능을 더 향상시킬 수 있다.

**③ 순서 민감 태스크와의 트레이드오프 심층 분석**

Cyclic shift의 경우 순서 민감 태스크에서 성능 저하가 관찰되었다. Skip-based shift는 이를 완화하지만, **순서 정보가 매우 중요한 태스크(예: 시간 순서 추론, 코드 실행 순서)**에서의 성능 영향을 더 체계적으로 분석해야 한다.

**④ RoPE 이외 위치 인코딩에 대한 적용 가능성 탐구**

논문은 ALiBi 등 다른 위치 인코딩에도 유사한 교란이 정의될 수 있다고 언급하지만, 실험적 검증이 없다. 다양한 위치 인코딩 메커니즘에서의 적용 가능성 연구가 필요하다.

**⑤ 멀티모달 장문맥으로의 확장**

이미지-텍스트, 비디오-텍스트 등 멀티모달 장문맥 시나리오에서 위치적 취약성은 더욱 복잡하게 나타날 수 있다. RoPE 교란의 개념을 멀티모달 설정으로 확장하는 연구가 중요해질 것이다.

**⑥ 최적 $\lambda$ 선택의 자동화**

현재 $\lambda = 1$로 고정하며, $\lambda \in [0.5, 1]$에서 최적 성능을 보임을 확인했다. 그러나 태스크나 도메인에 따라 최적 $\lambda$가 다를 수 있으므로, **적응적 $\lambda$ 스케줄링** 연구가 필요하다.

**⑦ 이론적 기반 강화**

RoPE 교란에 대한 불변성이 왜 일반화 성능을 향상시키는지에 대한 이론적 분석이 부족하다. 정보 이론적 또는 PAC-학습 이론적 관점에서의 분석이 방법론의 신뢰도를 높이고 설계 원칙을 제공할 것이다.

**⑧ 추론 시 방법과의 결합 효과 검토**

An et al. (2024), Yu et al. (2025) 등의 추론 시 위치 편향 완화 방법과의 결합 효과를 체계적으로 분석할 필요가 있다.

**⑨ 훈련 효율성 개선**

1.6× 계산 오버헤드를 줄이기 위한 연구가 필요하다. 예를 들어, 모든 배치에서 교란 뷰를 계산하지 않고 선택적으로 적용하거나, Gradient checkpointing과 결합하는 방안을 고려할 수 있다.

**⑩ 지시 튜닝(Instruction Tuning) 단계에서의 직접 적용**

현재는 지속 사전훈련(continued pretraining) 단계에서 적용하고 SFT 후에도 이득이 유지됨을 보였지만, **SFT 단계에서 직접 적용**하는 방안의 효과도 탐구할 가치가 있다.

---

## 참고 자료

**주요 논문 (본문 제공 PDF):**
- Zichong Li, Chen Liang, Liliang Ren, Tuo Zhao, Yelong Shen, Weizhu Chen. "Shuffle the Context: RoPE-Perturbed Self-Distillation for Long-Context Adaptation." arXiv:2604.14339v1, April 2026.

**본문에서 인용된 관련 핵심 참고문헌:**
- Gao et al. (2025). "How to train long-context language models (effectively)." ACL 2025.
- Hsieh et al. (2024). "RULER: What's the real context size of your long-context language models?" arXiv:2404.06654.
- Yen et al. (2025). "HELMET: How to evaluate long-context language models effectively and thoroughly." ICLR 2025.
- Su et al. (2024). "RoFormer: Enhanced transformer with rotary position embedding." Neurocomputing.
- Peng et al. (2024). "YaRN: Efficient context window extension of large language models." ICLR 2024.
- Zhu et al. (2023). "PoSE: Efficient context window extension of LLMs via positional skip-wise training." arXiv:2309.10400.
- Fang et al. (2025). "What is wrong with perplexity for long-context language modeling?" ICLR 2025.
- Liu et al. (2024). "Lost in the middle: How language models use long contexts." TACL.
- Ding et al. (2024). "LongRoPE: Extending LLM context window beyond 2 million tokens." ICML 2024.
- Xiong et al. (2024). "Effective long-context scaling of foundation models." NAACL 2024.
- Ruoss et al. (2023). "Randomized positional encodings boost length generalization of transformers." arXiv:2305.16843.
- Kazemnejad et al. (2023). "The impact of positional encoding on length generalization in transformers." NeurIPS 2023.
- Zhang et al. (2024). "Found in the middle: How language models use long contexts better via plug-and-play positional encoding." NeurIPS 2024.
- Bai et al. (2024). "LongBench v2: Towards deeper understanding and reasoning on realistic long-context multitasks." arXiv:2412.15204.
- Lambert et al. (2024). "Tulu 3: Pushing frontiers in open language model post-training."
- Beltagy et al. (2020). "Longformer: The long-document transformer." arXiv:2004.05150.
- Zaheer et al. (2020). "Big Bird: Transformers for longer sequences." NeurIPS 2020.

> **⚠️ 정확도 관련 고지:** 본 답변은 제공된 PDF 원문에 기반하여 작성되었으며, 논문 외부의 추가 정보(논문 출판 이후 인용 현황, 재현 실험 결과 등)는 포함하지 않았습니다. 논문 자체에 명시되지 않은 내용에 대해서는 추론임을 명시하거나 기술하지 않았습니다.
