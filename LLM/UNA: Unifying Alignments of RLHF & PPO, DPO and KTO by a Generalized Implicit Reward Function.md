# UNA: Unifying Alignments of RLHF & PPO, DPO and KTO by a Generalized Implicit Reward Function

### 1. 핵심 주장 및 주요 기여

**UNA (Unifying Alignment)** 논문은 **RLHF/PPO, DPO, KTO**를 하나의 통일된 프레임워크로 통합하는 혁신적인 접근을 제시합니다. 논문의 근본적인 주장은 다음과 같습니다.

대규모 언어 모델(LLM)의 정렬을 위해 기존에 제안된 여러 방법들(RLHF, DPO, KTO)이 각각의 장점과 단점을 가지고 있지만, 이들을 **일반화된 암시적 보상 함수(Generalized Implicit Reward Function)**를 통해 통일할 수 있다는 것입니다. 

**네 가지 핵심 기여:**

1. **수학적 증명**: RLHF 목적함수에서 최적 정책은 다음의 암시적 보상 함수로 유도됨을 증명합니다:

$$r_θ(x,y) = β \log\left(\frac{π_θ(y|x)}{π_{ref}(y|x)}\right)$$

2. **통일된 프레임워크**: RLHF/PPO, DPO, KTO를 **암시적 보상과 명시적 보상의 차이를 최소화하는 감독학습**으로 통합

3. **다양한 피드백 유형 수용**: 쌍방향(pairwise), 이진(binary), 점수기반(score-based) 피드백 모두 처리 가능

4. **온/오프라인 양 모드 지원**: 온라인(RLHF 단순화) 및 오프라인(DPO, KTO 개선) 시나리오 모두 최적화

***

### 2. 해결하는 핵심 문제

#### RLHF의 문제점:
- **복잡성**: 보상 모델 학습과 RL 미세조정의 두 단계 프로세스
- **불안정성**: RL 특성상 학습 과정이 불안정함
- **메모리 부담**: 정책, 참조 정책, 보상 모델, 가치 모델 등 4개 모델 유지 필요

#### DPO의 한계:
- **피드백 제한**: 쌍방향 선호도 데이터만 활용 가능
- **데이터 효율성 저하**: 쌍방향 데이터를 계속 필요로 함 (일회용이 아님)
- **보상 모델 부재**: 명시적 보상 모델 없음으로 인한 세밀한 평가 불가능

#### KTO의 제약:
- **이진 데이터만 처리**: Thumbs up/down만 처리 (점수 기반 피드백 미지원)
- **다른 방식과의 통합 불가**: 여러 피드백 타입을 동시에 활용하지 못함

***

### 3. 제안 방법론 (수식 포함)

#### 3.1 핵심 이론: 암시적 보상 함수 유도

RLHF 목적함수로부터 시작:

$$π^*_θ(y|x) = \max_{π_θ} \mathbb{E}_{x∼D}\left[\mathbb{E}_{y∼π_θ(y|x)}[r_θ(x,y)] - βD_{KL}(π_θ(y|x)∥π_{ref}(y|x))\right]$$

로그-합 부등식(log-sum inequality)을 적용하면, 최적값 조건에서:

$$r(x,y) = β\log\left(\frac{λπ_θ(y|x)}{π_{ref}(y|x)}\right) = β\log\left(\frac{π_θ(y|x)}{π_{ref}(y|x)}\right) \quad (λ=1 \text{일 때})$$

이는 **보상 함수가 정책의 로그 비율로 표현**되며, 외부 보상 모델 없이도 정책 자체에서 암시적 보상을 도출할 수 있음을 의미합니다.

#### 3.2 일반 손실 함수 (General Loss Function)

암시적 보상과 명시적 보상의 차이를 최소화:

$$L_{UNA-reward}(π_θ) = \mathbb{E}_{x∼D}\mathbb{E}_{y∼π_θ(·|x)}[g(r_φ(x,y), r_θ(x,y))]$$

여기서 $g(x_1, x_2)$는 두 보상 간 차이를 측정하는 일반함수 (MSE, BCE 등)

#### 3.3 오프라인 UNA의 세 가지 응용

**A. 쌍방향 피드백 (DPO와 동등)**

$$L_{UNA-pair}(π_θ) = -\mathbb{E}_{(x,y_w,y_l)∼D}(r_θ(x,y_w) - r_θ(x,y_l))$$

DPO와 동등성 증명됨 (로지스틱 함수 $f(x) = \log[σ(x)]$ 적용 시)

**B. 이진 피드백 (KTO 개선)**

긍정 피드백: $r_φ(x,y_w) = 1$, 부정 피드백: $r_φ(x,y_l) = 0$

$$L_{UNA-binary}(π_θ) = -\left[\mathbb{E}_{(x,y_w)∼D}g(r_θ(x,y_w), 1) + \mathbb{E}_{(x,y_l)∼D}g(r_θ(x,y_l), 0)\right]$$

**C. 점수기반 피드백 (RM/LLM 증류)**

연속 보상 $r_φ(x,y) ∈ $에 대해:[1]

$$L_{UNA-score}(π_θ) = -\mathbb{E}_{(x,y)∼D}[(r_θ(x,y) - r_φ(x,y))^2]$$

#### 3.4 온라인 UNA: RLHF의 단순화

PPO 대신 감독학습으로 대체:

$$L_{online}(π_θ) = \mathbb{E}_{x,y∼π_θ}[MSE(r_θ(x,y), r_φ(x,y))]$$

이를 통해:
- 불안정한 RL 제거 → 안정적인 감독학습
- 가치 모델 필요 없음 → 메모리 감소
- 계산 비용 감소 → 훈련 속도 향상 (RLHF 대비 20% 빠름)

***

### 4. 모델 구조

#### 4.1 전체 아키텍처 (Figure 1 참조)

| 구성요소 | UNA | RLHF | DPO | KTO |
|---------|------|------|------|------|
| 프롬프트 데이터 | ✓ | ✓ | - | - |
| 선호도 피드백 | ✓ | ✓ | ✓ | - |
| 이진 피드백 | ✓ | - | - | ✓ |
| 점수 피드백 | ✓ | - | - | - |
| 명시적 보상모델 | ✓ | ✓ | - | - |
| 암시적 보상모델 | ✓ | - | ✓ | ✓ |
| 온라인/오프라인 | 양쪽 | 온라인 | 오프라인 | 오프라인 |

#### 4.2 훈련 파이프라인

**오프라인 UNA**:
```
사전수집 데이터 
  ↓
암시적 보상 계산 (정책에서)
  ↓
명시적 보상 계산 (인간/RM/LLM)
  ↓
MSE/BCE 손실로 최소화
  ↓
정책 업데이트
```

**온라인 UNA**:
```
프롬프트 발송
  ↓
정책에서 응답 생성 & 암시적 보상 계산
  ↓
보상 모델에서 명시적 보상 계산
  ↓
차이 최소화로 정책 업데이트
```

***

### 5. 성능 향상 및 검증

#### 5.1 오프라인 UNA 성능 (Open LLM Leaderboard)

| 방법 | 새 리더보드 평균 | 구 리더보드 평균 | MT-Bench | AlpacaEval |
|------|-----------------|-----------------|----------|------------|
| Mistral 기준 | 28.61 | 60.93 | 3.15 | 0.31 |
| DPO | 28.53 | 62.26 | 6.1 | 3.67 |
| KTO | 28.56 | 62.74 | 5.99 | 4.46 |
| **UNA-binary** | **28.93** | **63.12** | **6.78** | **5.54** |
| **UNA-score** | **30.92** | **64.35** | **6.72** | **8.78** |

**분석**: UNA-score는 점수기반 피드백을 활용하여 DPO/KTO를 크게 초월함 (2-3% 성능 향상)

#### 5.2 온라인 UNA 성능 (RLHF 비교)

| 방법 | bbh | gpqa | mmlu-pro | musr | ifeval | math-hard | 평균 |
|------|-----|------|----------|------|--------|-----------|------|
| Mistral-INST | 42.46 | 29.05 | 24.53 | 38.30 | 38.46 | 2.02 | 29.14 |
| RLHF | 42.50 | 28.99 | 24.60 | 38.29 | 38.53 | 1.79 | 29.12 |
| **UNA (온라인)** | **42.78** | **28.32** | **24.87** | **38.03** | **39.17** | **1.75** | **29.15** |

- 14개 작업 중 8개에서 RLHF 초과 (2개 동등)
- 안정성과 속도는 RLHF 보다 우월

#### 5.3 효율성 지표

| 지표 | RLHF | UNA |
|------|------|-----|
| 훈련 시간 (20k 스텝) | 8시간 | 6.5시간 |
| 속도 향상 | - | **18.75% 빠름** |
| 가치 모델 필요 | 필요 | 불필요 |
| 메모리 효율 | - | **더 효율적** |

***

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 이론적 기초

UNA의 일반화 성능 향상은 다음 세 가지 메커니즘에서 비롯됩니다:

**1. 다양한 피드백 활용**
- 쌍방향만 사용하는 DPO와 달리, 이진/점수 피드백도 활용
- 데이터셋의 정보량 극대화 → 더 견고한 보상신호 학습
- 특히 점수기반 피드백 활용 시 성능 2-3% 향상

**2. 명시적 보상 모델의 역할**
DPO는 명시적 보상이 없으나, UNA는:
- 보상 모델 또는 LLM 평가자를 활용한 명시적 신호 포함
- 암시적 보상(정책에서) vs 명시적 보상(외부 평가)의 정합성 강제
- 이는 정책의 과적합을 방지하고 일반화를 개선

**3. MSE 기반 손실함수의 장점**
- DPO의 이진 분류 손실(BCE)보다 연속적이고 부드러운 학습신호 제공
- 극단값에 덜 민감 → 안정적인 훈련

#### 6.2 실험적 증거

**오프라인 UNA-score의 성능 개선**:
```
- DPO (쌍방향): 28.53 (새 리더보드)
- KTO (이진): 28.56 
- UNA-binary (이진): 28.93 (+0.37)
- UNA-score (점수): 30.92 (+2.39) ← 일반화 성능 대폭 향상
```

**원인 분석**:
- 점수기반 피드백 = 세밀한 품질 신호
- 암시적 보상(정책)과 명시적 보상(점수)의 정렬 강제
- 일반화 오류 경계 개선

#### 6.3 도메인 이동(Distribution Shift) 대응

UNA의 구조적 장점:
1. **명시적 보상 모델 유지**: 하나의 일관된 평가 기준 제공
2. **온/오프라인 혼합**: 온라인 학습으로 새로운 분포 적응 가능
3. **유연한 손실함수**: 다양한 $g(\cdot,\cdot)$ 선택으로 적응적 학습 가능

***

### 7. 논문의 한계 및 미래 연구 방향

#### 7.1 명시된 한계

1. **모델 크기 제약**: 현재 실험은 1-2B 매개변수 모델 중심
   - 더 큰 모델(70B+)에서의 검증 필요
   - "정렬세"(alignment tax) 문제가 여전히 존재 가능

2. **2단계 훈련 필요성**: 온라인 UNA도 여전히 보상 모델 학습 필요
   - 1단계 통합 훈련 미달성
   - 보상 모델 훈련 비용 여전히 존재

3. **f(x) 함수의 최적화 미완료**:
   $$r(x,y) = β\log\left(\frac{π_θ(y|x)}{π_{ref}(y|x)}\right) + f(x) + c$$
   - f(x)와 상수 c의 최적 구성 미탐색
   - 성능 향상 여지 존재

4. **손실함수 다양성 부족**: 현재 MSE, BCE만 사용
   - 다른 거리 메트릭 탐색 여지 있음
   - Wasserstein, TV 거리 등 검토 가능

#### 7.2 미해결 연구 질문

1. **확장성**: 매우 큰 모델(100B+)에서 실제 수렴 속도는?
2. **품질-계산 트레이드오프**: 점수기반 피드백 수집 비용 대비 성능 이득은?
3. **하이퍼파라미터 민감성**: β, 정규화 계수 등의 최적값은?

***

### 8. 2020년 이후 관련 최신 연구 비교 분석

#### 8.1 주요 선행 및 후행 연구

| 연도 | 방법 | 핵심 특징 | UNA와의 관계 |
|------|------|----------|------------|
| 2022 | **RLHF** (Ouyang et al.) | 2단계: 보상모델 + PPO | UNA의 기초 목적함수 |
| 2023 | **DPO** (Rafailov et al.) | 암시적 보상, 감독학습 | UNA가 확장 (다중 피드백) |
| 2023 | **IPO** (Azar et al.) | KL 정규화 개선 | 유사한 통합 시도 |
| 2024 | **KTO** (Ethayarajh et al.) | 이진 피드백, 전망 이론 | UNA가 포함 (일반화) |
| 2024 | **ORPO** (Hong et al.) | 참조모델 제거, SFT 통합 | 직교적 접근 (참조모델 제거) |
| 2024 | **RPO** (Nvidia) | 명시적 보상 활용 | UNA와 매우 유사한 방향 |

#### 8.2 상세 비교

**DPO vs UNA**:
- DPO: 쌍방향 피드백만 → 데이터 효율 낮음
- UNA: 쌍방향/이진/점수 모두 → 유연성 높음
- 수학적 관계: UNA-pair는 DPO와 동등

**KTO vs UNA**:
- KTO: 이진 피드백 최적화, 손실회피 모델링
- UNA: 이진 피드백 포함하되 점수 피드백도 지원
- 성능: UNA-score (30.92) > KTO (28.56)

**RPO vs UNA**:
- RPO: 명시적 보상을 직접 최적화
- UNA: 암시적-명시적 보상 정합성 강제
- 패러다임: RPO는 역KL 기반, UNA는 더 일반적

**ORPO vs UNA**:
- ORPO: 참조모델 제거, SFT에 통합
- UNA: 참조모델 유지, 명시적 보상모델 활용
- 트레이드오프: ORPO는 단순성, UNA는 유연성

#### 8.3 이론적 통합 위치

UNA는 다음과 같은 위치에서 최신 연구를 통합합니다:

```
RLHF (2022)
  ↓
DPO (2023) ← 암시적 보상
  ├─ IPO (2023) ← KL 정규화
  ├─ KTO (2024) ← 이진 피드백
  └─ UNA (2024) ← 모든 피드백 타입 + 명시적 보상
      ├─ ORPO (2024) ← 참조모델 제거 [직교]
      └─ RPO (2024) ← 역KL 기반 [유사]
```

#### 8.4 성능 순위 (최신 연구 기준)

| 방법 | AlpacaEval | 일반화 | 안정성 | 효율성 |
|------|-----------|--------|--------|--------|
| RLHF | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ |
| DPO | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| KTO | ★★★★☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| **UNA** | **★★★★★** | **★★★★★** | **★★★★☆** | **★★★★★** |
| ORPO | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★★★★ |
| RPO | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ |

***

### 9. 앞으로의 연구에 미치는 영향 및 고려점

#### 9.1 패러다임 전환

UNA 논문은 LLM 정렬 연구의 패러다임 전환을 제안합니다:

**기존 패러다임**: 다양한 방법들의 분산
- RLHF, DPO, KTO, ORPO 등이 각각 발전
- 통합 프레임워크 부재

**새 패러다임**: 통합적 접근
- 암시적 보상 함수를 중심으로 통합
- 피드백 타입에 따라 동일한 프레임워크 적용
- 명시적/암시적 보상의 정합성 강제

#### 9.2 향후 연구 방향

**1. 이론적 확장**
- 더 일반적인 보상 함수 형태 규명
- 수렴성 보장(convergence guarantee) 분석
- 표본 복잡도(sample complexity) 분석

**2. 실무적 확장**
- 초대형 모델(100B+) 실험
- 다언어 설정에서의 성능
- 도메인별 최적 피드백 타입 분석

**3. 알고리즘 개선**
- 더 나은 손실함수 설계
- 동적 정규화(adaptive regularization)
- 다목적 정렬(multi-objective alignment)

**4. 응용 확대**
- 이미지 생성 모델 정렬 (Diffusion-UNA?)
- 음성 모델 정렬
- 멀티모달 모델 정렬

#### 9.3 실제 적용 시 고려사항

**1. 데이터 수집 비용 vs 성능**
```
- 쌍방향: 수집 어려움, 정보량 높음
- 이진: 수집 쉬움, 정보량 중간
- 점수: 수집 어려움, 정보량 가장 높음

→ 비용-이득 분석 필요
```

**2. 보상 모델 품질**
- UNA의 성능은 명시적 보상모델 품질에 크게 의존
- 보상 모델 과적합 시 일반화 오류 발생 가능
- 보상 모델 검증 프로토콜 필요

**3. 온/오프라인 선택**
- 온라인: 실시간 데이터 수집 가능, 계산 비용 높음
- 오프라인: 고정 데이터셋, 효율적
- 하이브리드 접근 가능성

**4. 하이퍼파라미터 튜닝**
- β 값: 보상-KL 트레이드오프 조절
- 손실함수 $g(\cdot,\cdot)$ 선택
- 정규화 계수 (이진 피드백의 경우)

#### 9.4 업계 영향

**단기 (1-2년)**
- DPO의 후속 개선으로 인식
- 대규모 모델 훈련에서의 채택
- 다양한 변형 알고리즘 제안

**중기 (2-3년)**
- 산업 표준화 추진 가능성
- 오픈소스 라이브러리 통합
- 다양한 도메인 적용

**장기 (3년 이상)**
- 일반적인 정렬 프레임워크로 정립
- 이론적 통합으로 발전
- 새로운 문제 해결을 위한 기초 이론 제공

***

### 10. 결론: 종합 평가

#### 10.1 강점

1. **이론적 엄밀성**: 로그-합 부등식을 통한 수학적 증명
2. **실무적 유연성**: 다양한 피드백 타입 수용
3. **경험적 우수성**: 일관된 성능 향상 (특히 점수기반)
4. **통합성**: 기존 방법들을 일반화된 틀 내에 포함

#### 10.2 약점

1. **실험 규모**: 주로 7B 이하 모델 중심
2. **계산 비용**: 여전히 보상 모델 필요
3. **민감성**: 보상 모델 품질에 의존
4. **적용 복잡성**: 점수기반 피드백 수집의 실무적 난점

#### 10.3 최종 평가

**UNA는 LLM 정렬 분야의 중요한 진전**을 나타냅니다. 다양한 기존 방법을 통일하면서도 새로운 유연성을 제공합니다. 특히 **점수기반 피드백을 활용한 성능 향상** (2-3%)은 실무적 가치가 높습니다.

그러나 **초대형 모델에서의 검증 필요**, **보상 모델 의존성 개선**, **실제 적용 비용 분석**이 향후 중요한 과제입니다.

이 논문은 다음 세대 LLM 정렬 연구의 기초가 될 가능성이 높으며, DPO와 함께 2023-2024년의 가장 영향력 있는 정렬 방법론으로 평가됩니다.

***

### 참고: 주요 수식 요약

| 개념 | 수식 |
|------|------|
| RLHF 목적함수 | $$\max_{π_θ} \mathbb{E}[\mathbb{E}\_{y∼π_θ}[r(x,y)] - βD_{KL}(π_θ∥π_{ref})]$$ |
| 암시적 보상 | $$r_θ(x,y) = β\log\left(\frac{π_θ(y\|x)}{π_{ref}(y\|x)}\right)$$ |
| 일반 손실 | $$L_{UNA} = \mathbb{E}[g(r_φ(x,y), r_θ(x,y))]$$ |
| DPO 동등성 | $$L_{DPO} = -\mathbb{E}[\log σ(r_θ(y_w) - r_θ(y_l))]$$ |
| MSE 손실 | $$L_{MSE} = \mathbb{E}[(r_θ - r_φ)^2]$$ |

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d330f955-c42c-4d2a-a613-e5e8e51b0fef/2408.15339v3.pdf)
[2](https://ieeexplore.ieee.org/document/10657686/)
[3](https://www.semanticscholar.org/paper/0d1c76d45afa012ded7ab741194baf142117c495)
[4](https://arxiv.org/abs/2311.16839)
[5](https://arxiv.org/abs/2309.16240)
[6](https://aclanthology.org/2024.findings-acl.630)
[7](https://arxiv.org/abs/2310.03708)
[8](https://arxiv.org/abs/2311.08380)
[9](https://arxiv.org/abs/2309.06657)
[10](https://arxiv.org/abs/2312.16430)
[11](https://arxiv.org/abs/2312.10584)
[12](http://arxiv.org/pdf/2404.14723.pdf)
[13](https://arxiv.org/pdf/2502.14356.pdf)
[14](https://arxiv.org/html/2503.01076v1)
[15](https://arxiv.org/html/2410.19720)
[16](https://arxiv.org/pdf/2502.16825.pdf)
[17](http://arxiv.org/pdf/2503.15880.pdf)
[18](https://arxiv.org/pdf/2404.04626.pdf)
[19](https://arxiv.org/html/2410.04350v1)
[20](https://arxiv.org/abs/2305.18290)
[21](https://intuitionlabs.ai/articles/reinforcement-learning-human-feedback)
[22](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/SooWeiKoh.pdf)
[23](https://openaccess.thecvf.com/content/CVPR2024/papers/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.pdf)
[24](https://aws.amazon.com/blogs/machine-learning/fine-tune-large-language-models-with-reinforcement-learning-from-human-or-ai-feedback/)
[25](https://contextual.ai/better-cheaper-faster-llm-alignment-with-kto/)
[26](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dpo/)
[27](https://huyenchip.com/2023/05/02/rlhf.html)
[28](https://winniexu.ca/research/kto)
[29](https://openreview.net/pdf?id=HPuSIXJaa9)
[30](https://openreview.net/pdf?id=WWXjMYZxfH)
[31](https://arxiv.org/abs/2402.01306)
[32](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf)
[33](https://arxiv.org/abs/2203.02155)
[34](https://liner.com/ko/review/kto-model-alignment-as-prospect-theoretic-optimization)
[35](https://arxiv.org/html/2506.08266v1)
[36](https://arxiv.org/pdf/2305.18290.pdf)
[37](https://arxiv.org/abs/2204.05862)
[38](https://arxiv.org/html/2402.01306v1)
[39](https://www.semanticscholar.org/paper/Direct-Preference-Optimization:-Your-Language-Model-Rafailov-Sharma/0d1c76d45afa012ded7ab741194baf142117c495)
[40](https://arxiv.org/html/2504.12501v3)
[41](https://arxiv.org/html/2402.01306v3)
[42](https://arxiv.org/html/2305.18290v3)
[43](https://arxiv.org/pdf/2402.09401.pdf)
[44](https://arxiv.org/html/2402.01306v4)
[45](https://arxiv.org/html/2410.15595v2)
[46](https://arxiv.org/pdf/2402.01306.pdf)
[47](https://www.semanticscholar.org/paper/ebdffb65d5110117c9c44c14697bf3b63e0eceb6)
[48](https://arxiv.org/abs/2406.19185)
[49](https://arxiv.org/html/2402.05749v2)
[50](https://arxiv.org/pdf/2312.16430.pdf)
[51](https://arxiv.org/abs/2503.18454v1)
[52](http://arxiv.org/pdf/2410.04203.pdf)
[53](https://arxiv.org/html/2502.00203v1)
[54](https://arxiv.org/pdf/2310.12036.pdf)
[55](https://arxiv.org/html/2502.02088v3)
[56](https://arxiv.org/pdf/2403.08635.pdf)
[57](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/final-reports/256735149.pdf)
[58](https://huggingface.co/papers/2403.07691)
[59](https://arxiv.org/pdf/2502.00203.pdf)
[60](https://arxiv.org/abs/2403.07691)
[61](https://arxiv.org/html/2508.05170v2)
[62](https://arxiv.org/abs/2310.12036)
[63](https://www.youtube.com/watch?v=52kMBrAI_IM)
[64](https://aclanthology.org/2025.findings-naacl.447.pdf)
[65](https://proceedings.mlr.press/v238/gheshlaghi-azar24a/gheshlaghi-azar24a.pdf)
[66](https://aclanthology.org/2024.emnlp-main.626.pdf)
[67](https://proceedings.neurips.cc/paper_files/paper/2024/file/5e1c255653eb98cef13f45b2d337c882-Paper-Conference.pdf)
[68](https://openreview.net/pdf?id=lo6LdYbx7f)
[69](https://aclanthology.org/2024.emnlp-main.626/)
[70](https://proceedings.iclr.cc/paper_files/paper/2025/file/7f70331dbe58ad59d83941dfa7d975aa-Paper-Conference.pdf)
[71](https://www.arxiv.org/pdf/2508.05170.pdf)
[72](https://arxiv.org/pdf/2405.20830.pdf)
[73](https://arxiv.org/pdf/2509.24159.pdf)
[74](https://arxiv.org/pdf/2506.07492.pdf)
[75](https://arxiv.org/html/2601.05882)
[76](https://arxiv.org/html/2509.24159v1)
[77](https://arxiv.org/pdf/2403.07691.pdf)
[78](https://www.arxiv.org/pdf/2508.05170v1.pdf)
[79](https://arxiv.org/html/2310.12036v1)
[80](https://arxiv.org/html/2403.07691v1)
[81](https://arxiv.org/pdf/2601.08521.pdf)
