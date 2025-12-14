
# RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment

## 1. 논문의 핵심 주장과 주요 기여

### 1.1 핵심 주장
RAFT는 기존의 강화학습(RL) 기반 정렬 방법의 복잡성과 불안정성을 해결하기 위해, **보상 모델을 활용한 반복적 샘플 필터링과 지도학습 미세조정을 결합한 프레임워크**를 제안합니다. 논문의 주요 논의는 다음과 같습니다:

1. **기존 방법의 한계 지적**: PPO 기반 RLHF는 불안정하고 메모리 집약적이며 초매개변수 조정이 어렵습니다.
2. **SFT와의 중간 경로**: 고정된 데이터셋에 대한 SFT보다는 성능이 우수하고, 온라인 RL보다는 안정성이 뛰어난 방법을 제시합니다.
3. **Best-of-K 정책 학습**: 모델은 반복적으로 자신이 생성한 K개 샘플 중 가장 높은 보상을 받은 샘플로부터 학습합니다.

### 1.2 주요 기여
- **안정성과 효율성**: SFT 유사의 학습으로 인한 높은 안정성
- **메모리 효율성**: 데이터 생성과 모델 개선 단계의 분리로 메모리 버든 감소
- **유연성**: LLM과 확산 모델 모두에 적용 가능한 일반적 프레임워크
- **보상 해킹 저항성**: 순위 기반 선택으로 보상 스케일에 덜 민감
- **일반화 가능성**: 다양한 생성 모델에 적용 가능한 방법론 제시

***

## 2. 해결하는 문제와 제안하는 방법

### 2.1 문제 정의

생성 모델의 정렬을 위한 목표 함수는 다음과 같이 정의됩니다:

$$\max_w \mathbb{E}_{x \sim D, y \sim p_g(\cdot|w,x)} r(x, y)$$

여기서:
- \(w\): 모델 매개변수
- \(x\): 프롬프트 (입력)
- \(y\): 생성 출력
- \(r(x, y)\): 보상 함수
- \(D\): 입력 분포

### 2.2 최적 정책의 형태

이상적인 상황에서 최적 정책은 다음 형태입니다:

$$p_g(\cdot|w^*, x) = \begin{cases} 1 & \text{if } y = \arg\max_{y \in Y} r(x, y) \\ 0 & \text{otherwise} \end{cases}$$

그러나 실제로 출력 공간 전체를 탐색하는 것은 불가능하므로, 제한된 샘플링으로부터 고품질 데이터를 구성해야 합니다.

### 2.3 RAFT 알고리즘

RAFT는 다음 세 단계를 반복합니다:

**Step 1: 데이터 수집**
$$y_1, ..., y_K \sim p_g^{1/\lambda}(·|w_t, x_i^t) \text{ for each } x_i^t \in D_t$$

여기서 $\(\lambda\)$ 는 온도 매개변수로 출력 다양성을 조절합니다.

**Step 2: 데이터 순위 매기기 및 필터링**
$$y^* := \arg\max_{y_j \in \{y_1, ..., y_K\}} r(x, y_j)$$

각 프롬프트에 대해 보상이 가장 높은 샘플을 선택합니다.

**Step 3: 모델 미세조정**
선택된 샘플 배치 B에 대해 표준 지도학습으로 미세조정:

$$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \in B} \log p_g(y|w, x)$$

### 2.4 KL 정규화 확장

모델이 과도하게 변하지 않도록 KL 발산 제약을 추가할 수 있습니다:

$$\max_w \left[\mathbb{E}_{x \sim D, y \sim p_g(\cdot|w,x)} r(x, y) - \beta Q(w)\right]$$

여기서 Q(w)는 다음과 같이 정의됩니다:

$$Q(w) = \mathbb{E}_{x \sim D} KL(p_g(\cdot|w,x) \| p_{G_0}(\cdot|w_0,x)) = \mathbb{E}_{x \sim D} \sum_{y \in Y} p_g(y|w,x) \log \frac{p_g(y|w,x)}{p_{G_0}(y|w_0,x)}$$

수정된 보상 함수:

$$\tilde{r}(x, a) = r(x, a) - \beta \log \frac{p_g(y|w,x)}{p_{G_0}(y|w_0,x)}$$

***

## 3. 모델 구조 및 아키텍처

### 3.1 시스템 구성

RAFT는 다음의 핵심 컴포넌트로 구성됩니다:

1. **생성 모델** \(G_t\): 초기 미세조정된 모델 (LLaMA-7B-SFT)
2. **보상 모델** \(r(x,y)\): Bradley-Terry 모델에 기반한 선호도 스코어링
3. **데이터 필터링 메커니즘**: 순위 기반 샘플 선택
4. **SFT 트레이너**: 표준 지도학습 미세조정 엔진

### 3.2 모델 아키텍처 상세

#### 보상 모델 구조
Bradley-Terry 선호도 모델:

```math
p^*(y_w > y_l | x) := \frac{\exp(r^*(x, y_w))}{\exp(r^*(x, y_w)) + \exp(r^*(x, y_l))} := \sigma(-\Delta(r^*(x, y_w) - r^*(x, y_l)))
```

손실 함수:

$$\text{loss}(\theta) = -\mathbb{E}_{x,y_w,y_l \sim D_{train}} \left[\log(\sigma(r_\theta(x, y_w) - r_\theta(x, y_l)))\right]$$

#### 학습 과정의 반복성
- **반복 t**: 현재 모델 \(G_t\)로부터 K개 샘플 생성
- **필터링**: 보상 기준으로 최고 샘플 선택
- **업데이트**: 선택된 샘플로 2 에포크 SFT 진행
- **수렴 판정**: 3 연속 반복에서 보상이 수렴

### 3.3 계산적 효율성

RAFT의 메모리 구조:

| 단계 | 필요 모델 | 메모리 요구 |
|------|---------|----------|
| 데이터 생성 | 생성 모델 1개 | $\(M \times 1\)$ |
| 보상 계산 | 보상 모델 1개 | $\(M \times 1\)$ |
| SFT 미세조정 | 생성 모델 1개 | $\(M \times 1\)$ |
| **PPO 비교** | 4개 모델 동시 | $\(M \times 4\)$ |

PPO와 달리 RAFT는 각 단계에서 1개 모델만 필요하므로 메모리 효율성이 $\(4\times\)$ 우수합니다.

***

## 4. 성능 향상 및 한계

### 4.1 성능 개선 결과

#### HH-RLHF 데이터셋 실험

| 모델 | 보상 | 이상화도(ppl) | MSTTR-100 |
|------|-------|------------|----------|
| LLaMA-7B-SFT | 0.772 | 3.781 | 0.597 |
| PPO (β=0.1) | 2.077 | 4.156 | 0.597 |
| **RAFT-K32-λ1.0** | **2.294** | **4.031** | **0.611** |

주요 성과:
- 보상 점수: SFT 대비 **197% 향상**, PPO 대비 **10.4% 향상**
- 이상화도(Perplexity): PPO보다 **2.9% 개선** (낮을수록 좋음)
- 다양성: 모든 다양성 지표에서 우수한 성능

#### 일반화 성능 분석

**K 값의 영향** (Table 5):
- K=8: 보상 2.180
- K=16: 보상 2.251
- K=32: 보상 2.329 (+7.0% vs K=8)

분석: $\(\sqrt{\log K}\)$ 에 비례하는 한계 수익 체감

**온도 효과** (Table 6):
- λ=0.7: 보상 2.198, 높은 정확도지만 낮은 다양성
- λ=1.0: 보상 2.143, 최고 다양성
- K=32와 λ=1.0 조합: **최적의 균형** (보상 2.294)

#### 배포 외 일반화
확산 모델 실험 (Stable Diffusion v1.5):

| 작업 | DDPO | RAFT | 속도 향상 |
|-----|------|------|---------|
| 해상도 적응 (256×256) | CLIP: 28.8 | CLIP: 27.3 | **50배 빠름** |
| 미학 점수 | 6.04 | **6.14** | 속도 50배 |

### 4.2 알고리즘 한계

#### 4.2.1 이론적 한계

1. **데이터 커버리지 가정**
   - 오프라인 RL 이론에 따르면, 최적 정책과 경쟁하기 위해서는:
   $$\frac{d_{\pi^*}(s,a)}{d_{\text{data}}(s,a)} \leq C$$
   
   여기서 C는 균등하게 유한해야 함. RAFT의 반온라인 특성이 이를 완화하지만 완전히 해결하지는 못함.

2. **Best-of-K의 한계**
   $$\mathbb{E}[r] \leq \mathbb{E}[\max_i r_i] \leq \mathbb{E}[r] + \sqrt{\frac{B^2}{2}\log K}$$
   
   표본 K가 증가해도 한계 수익이 $\(\sqrt{\log K}\)$ 로 체감

#### 4.2.2 실무적 한계

1. **보상 모델 의존성**
   - 보상 모델의 정확도에 크게 의존
   - 보상 모델 검증 정확도: 75.48% (완벽하지 않음)

2. **보상 해킹 문제**
   - 불완벽한 보상 모델을 악용할 가능성
   - 논문의 초기 버전에서 보상 모델이 이모지와 '#' 기호를 선호하는 문제 발생
   - 필터링으로 해결했지만, 체계적 해결 방법은 제시되지 않음

3. **계산 비용**
   - K=32일 때 총 학습 시간: **7.05 시간**
   - PPO 최고 성능: 8.7 시간
   - 동급 성능에도 여전히 상당한 계산 필요

4. **보상 보정 문제**
   - 낮은 보상 예측에서 비관적 편향
   - 높은 보상 예측에서 과신 (Figure 8, ECE=0.082)
   - 이는 PPO보다 RAFT가 덜 영향을 받지만, 완전히 면역은 아님

***

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 배포 외 일반화 분석

#### 5.1.1 LLM의 배포 외 성능

**오픈 도메인 테스트 결과** (Table 3-4):
- GPT-4 평가: RAFT-K32가 PPO-β0.1에 대해 **65:32 승률**
- 인간 평가: 더 높은 '동점' 비율 보여줌 (RAFT의 안정성 시사)

**다양성 개선**:
- Distinct-1: RAFT 0.032 vs PPO 0.033 (유사)
- Unique-1: RAFT 8691 vs PPO 7370 (**18% 향상**)
- 평균 응답 길이: RAFT 156.2 vs PPO 127.8 (**22% 길이 증가**)

#### 5.1.2 증류(Distillation) 실험

**GPT-Neo-2.7B를 LLaMA-7B 샘플로 학습** (Table 8):
- 기본 GPT-Neo 보상: -1.23
- RAFT-LLaMA 학습 후: 0.739 (**161% 향상**)
- 일반화 성공: 더 큰 모델의 데이터로 작은 모델 개선 가능

#### 5.1.3 확산 모델의 일반화

**도메인 외 성능** (Table 9):
| 데이터셋 | DDPO CLIP | RAFT CLIP | 개선 |
|---------|----------|-----------|------|
| In-domain (CIFAR-10) | 28.8±1.2 | 27.3±1.4 | 안정적 |
| **Out-of-domain (CIFAR-100)** | 30.2±1.8 | **26.7±4.5** | 높은 분산 |

**관찰**: Out-of-domain에서 더 높은 분산. RAFT가 학습 데이터에 다소 의존적임을 시사.

### 5.2 일반화 향상 메커니즘

#### 5.2.1 KL 정규화의 효과

**KL 계수 실험** (Table 7, Figure 3):

| KL 계수 | 최종 보상 | KL 발산 | 이상화도 |
|---------|---------|--------|---------|
| 0 | 2.143 | ~34 | 3.921 |
| 0.005 | 2.087 | ~28 | 3.953 |
| 0.01 | 2.038 | ~25 | 3.953 |
| 0.1 | 2.029 | ~15 | 3.953 |

**통찰**:
- KL 계수 증가 → 보상 감소 (-5.3% at 0.1)
- 이상화도는 안정적 (3.95 유지)
- RAFT는 PPO보다 KL 제약에 **덜 민감**
  - PPO와 달리 순위 기반이므로 절대값 변화에 강함

#### 5.2.2 온라인 학습의 이점

**Learning Curve 분석** (Figure 1):
- 에이전트 성능과 Best-of-8 정책이 반복적으로 향상
- 초기 단계에서 빠른 성능 개선 (1-5 반복)
- 10-15 반복에서 수렴
- PPO와 달리 이상화도가 **안정적으로 유지** (3.8-4.0 범위)

**온라인 RL의 기여**:
- 초기 SFT 데이터셋의 '커버리지 부족' 문제 해결
- 매 반복마다 새로운 높은 보상 샘플 생성으로 분포 시프트 완화

### 5.3 일반화 성능의 한계와 개선점

#### 5.3.1 제한 요인

1. **보상 모델 품질**
   - Reward over-optimization 현상 (Figure 10)
   - 약한 보상 모델(GPT2-124M)에서 gold reward가 감소
   - 이는 더 강한 보상 모델이 필수임을 시사

2. **샘플 효율성**
   - K=32에서 각 반복당 32×2048 = 65,536개 샘플 생성
   - 총 학습 샘플: ~1M개 (82K 프롬프트 × 반복 × K)
   - 이는 DPO의 고정 데이터셋 사용보다 샘플 집약적

3. **분포 외(OOD) 성능**
   - 확산 모델 out-of-domain 실험에서 높은 분산
   - 학습 분포에서 먼 샘플에 대해서는 성능 감소

#### 5.3.2 개선 방향

논문에서 제시한 확장 가능성:

1. **전문 생성기 활용**: 프롬프트 엔지니어링이나 더 강력한 모델(GPT-4)에서 데이터 생성
2. **고급 생성 전략**: Beam Search, Top-K/Top-P 샘플링, 대조 탐색(Contrastive Search)
3. **보상 해킹 방지**: 필터링된 데이터셋 수동 검증
4. **전역 순위 매기기**: 프롬프트 간 비교를 허용하는 변형

***

## 6. 2020년 이후 관련 최신 연구 비교 분석

### 6.1 RLHF 계열의 진화

#### 1단계: 기본 RLHF (2020-2021)
**대표 논문**: InstructGPT (Ouyang et al., 2022)
- 3단계: SFT → 보상 모델 학습 → PPO
- **한계**: 불안정성, 메모리 효율성 낮음, 초매개변수 민감성

#### 2단계: DPO 계열 (2023)
**핵심**: Direct Preference Optimization (Rafailov et al., 2023)[1]
- **혁신**: 보상 모델 제거, 폐쇄형 최적 정책 도출
- **수식**:
$$L_{\text{DPO}} = -\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})$$

- **장점**: 간단한 구현, 안정성 향상
- **단점**: 오프라인 방법, 분포 외 성능 한계

#### 3단계: 하이브리드 방법 (2023-2024)

**3-1. RAFT** (본 논문)
- Best-of-K 학습 + SFT
- 온라인성과 오프라인 안정성 결합

**3-2. RRHF** (Yuan et al., 2023)[2]
- DPO와 유사하지만 온라인 샘플 활용
- 다양한 데이터 소스 결합

**3-3. RS-DPO** (2024)
$$L_{\text{RS-DPO}} = L_{\text{DPO}} + L_{\text{RS}}$$
- 거부 샘플링(Rejection Sampling) + DPO 결합
- 더 나은 sample efficiency

#### 4단계: 온라인 메서드의 부활 (2024-2025)

**4-1. Semi-online DPO**
- 온라인과 오프라인 데이터 혼합
- 분포 이동(distribution shift) 완화

**4-2. SAIL** (Self-Improving Efficient Online Alignment)
$$\text{SAIL}: L = L_{\text{online}} + L_{\text{self-improve}}$$
- 자체 개선 메커니즘
- 인간 피드백 필요 제거

**4-3. Online-IPO** (Identity Policy Optimization)
- Nash 평형 기반 온라인 정렬

### 6.2 보상 모델 혁신

#### 진통적 보상 모델 (2022-2023)
**Bradley-Terry 기반**:
$$p(y_w > y_l) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}$$

**문제점**:
- 배포 외(OOD) 일반화 약함
- 보상 해킹에 취약

#### 생성형 보상 모델 (2024-2025)

**GenRM** (Generative Reward Models)
- LLM을 보상 모델로 사용
- 자연어 설명 생성
- OOD 일반화 개선

**Inference-Time Scaling for GRM** (2025)
- 테스트 시간에 계산을 증가시켜 보상 품질 향상
- Best-of-K 분포의 더 정확한 근사

### 6.3 특수 도메인 확장

| 도메인 | 논문 | 주요 기여 |
|--------|------|---------|
| **이미지 생성** | Diffusion-DPO (Wallace et al., 2023) | 확산 모델 정렬 |
| **멀티모달** | RLHF-V (Yu et al., 2024) | 세그먼트 수준 보정 피드백 |
| **다국어** | Multilingual RLHF Survey (2025) | 언어 다양성 |
| **추론** | Step-DPO (2025) | 단계별 보상 |
| **안전** | One-Shot Safety Alignment (2024) | 제약 최적화 |

### 6.4 이론적 발전

#### 오프라인 RL 이론
- **문제**: 데이터 커버리지 요구사항
- **해결**: 
  - 온라인 탐색과 결합 (Leveraging Offline Data in Online RL, 2023)
  - 비관적 추정(Pessimistic Learning) 기반 알고리즘
  - RAFT의 반온라인 설계가 이 문제 부분 해결

#### KL 정규화의 역할
**최근 발견** (Zhang et al., 2025, Liu et al., 2025):
- PPO 대비 RAFT는 KL에 덜 민감
- KL-correct gradient 필요
- 중요 가중치(Importance Weights) 처리의 중요성

### 6.5 RAFT의 위치 맵

```
안정성
  ↑
  |     Online PPO
  |          ↓
  |     Semi-online DPO, SAIL, RAFT
  |          ↑
  |     DPO, RRHF
  |          ↓
  +─────────────────────→ 메모리 효율성
  낮         높
```

**RAFT의 특수성**:
1. **표준 DPO보다 더 나은 성능**: Best-of-K로 더 나은 데이터셋 구성
2. **PPO보다 더 안정적**: SFT 유사 학습
3. **메모리 효율**: 온라인 RL의 1/4
4. **구현 단순성**: DPO보다 복잡하지만 이해하기 쉬움

***

## 7. 연구의 영향과 미래 연구 방향

### 7.1 RAFT의 학계 영향

#### 직접적 영향
1. **후속 연구 자극**
   - Best-of-N 샘플링 최적화 연구 (BoNBoN, 2024)
   - 반온라인 정렬 방법론 발전

2. **산업 응용**
   - LMFlow 오픈소스 패키지에 구현
   - 실제 배포 환경에서 PPO 대체 가능성 제시

3. **이론적 기여**
   - 오프라인 RL과 온라인 정렬의 연계 이해
   - 데이터 커버리지 문제의 실무적 해결

### 7.2 미래 연구 시 고려할 점

#### 7.2.1 보상 모델 개선

**현재 한계**: 
- Bradley-Terry 모델의 이진 선호도 제약
- OOD 일반화 약점

**권장 방향**:
```
1. 생성형 보상 모델 활용
   - LLM 기반 자연어 설명 보상
   - Self-Taught Reasoner (STaR) 방법론 접목

2. 프로세스 보상 모델
   - 중간 단계별 피드백
   - Step-DPO와 결합

3. 앙상블 보상 모델
   - 여러 보상 모델 결합
   - 보상 불확실성 추정
```

#### 7.2.2 데이터 생성 전략

**현재**: 모델 자체 생성 샘플 사용

**개선점**:
```
1. 다양한 데이터 소스
   - 더 강력한 모델 (GPT-4, Claude)
   - 인간 전문가 샘플
   - 합성 데이터 증강

2. 스마트 탐색
   - 활성 학습 기반 샘플 선택
   - 정보 엔트로피 최대화
   - 경계 사례(Edge Case) 우선 탐색
```

#### 7.2.3 분포 시프트 관리

**문제**: 반복 과정에서 정책이 변함에 따른 분포 변화

**해결 방안**:
```
1. 참조 정책 동적 업데이트
   - Wasserstein DPO와 같은 robust 방법
   - 분포 외 성능 보장

2. 데이터 재가중치화
   - Importance sampling 기반
   - Token-level IS (TIS-DPO)

3. 혼합 정책 학습
   - 온라인과 오프라인 정책 혼합
   - Concentrability coefficient 최소화
```

#### 7.2.4 계산 효율성

**현재 한계**: K=32에서 여전히 7시간 학습

**최적화 기회**:
```
1. 추측 디코딩(Speculative Decoding)
   - 2-3배 추론 속도 향상
   - RAFT에 쉽게 통합 가능

2. 배치 병렬 처리
   - 모델 병렬화
   - 데이터 생성과 SFT 파이프라인화

3. 적응형 K 선택
   - 초기에는 작은 K 사용
   - 수렴 후 K 감소
```

#### 7.2.5 일반화 메커니즘 이해

**연구 질문**:
1. Best-of-K로부터의 학습이 왜 우수한가?
2. KL 정규화의 정확한 역할은?
3. 오픈 도메인에서의 성능은?

**권장 연구**:
```
1. 이론적 분석
   - Sample complexity 정식화
   - 수렴율 증명

2. 실험적 확장
   - 다양한 도메인 (수학, 코딩, 추론)
   - 더 큰 모델 (13B, 70B)
   - 다양한 언어

3. 비교 연구
   - Semi-online DPO vs RAFT
   - GenRM vs 전통 보상 모델
```

#### 7.2.6 보상 해킹 방지

**현재**: 사후 필터링에만 의존

**장기 해결**:
```
1. 해석 가능한 보상 모델
   - 자연어 기반 설명
   - 결정 근거 추적

2. 강건한 학습
   - 적대적 훈련(Adversarial Training)
   - Distributionally robust optimization

3. 체계적 검증
   - 자동 검증 테스트
   - 모니터링 프레임워크
```

### 7.3 연구 로드맵 제안

```
단기 (6-12개월)
├── 생성형 보상 모델과 RAFT 결합
├── 더 큰 모델(70B)에서의 성능 검증
└── 다양한 언어/도메인 확장

중기 (1-2년)
├── 온라인 정렬의 이론적 기초 제공
├── 반강화학습(Hybrid RL) 프레임워크 정립
└── 분포 외 일반화 메커니즘 규명

장기 (2년 이상)
├── Foundation Model 정렬의 표준 방법론 확립
├── 멀티모달/다국어 정렬의 일반화
└── 보상 모델 없이도 작동하는 방법 탐색
```

***

## 8. 결론

### 8.1 RAFT의 위치

**RAFT는 RLHF의 진화 과정에서 중요한 이정표**로, 다음과 같은 특징을 갖습니다:

1. **PPO의 복잡성과 불안정성 해결**: SFT 유사의 간단한 학습으로 실무 적용성 향상
2. **DPO의 오프라인 한계 극복**: 온라인 샘플 생성으로 데이터 커버리지 문제 부분 해결
3. **실무적 우수성**: 메모리 효율성과 성능의 우수한 균형

### 8.2 주요 발견

1. **Best-of-K 정책의 강력성**: 간단한 순위 기반 선택이 복잡한 RL보다 효과적
2. **순위 기반의 강건성**: 절대 보상값 변화에 덜 민감해 보상 해킹 저항성 우수
3. **반온라인 학습의 가치**: 완전 온라인도, 완전 오프라인도 아닌 중간 경로의 효과성

### 8.3 한계와 미래

**해결되지 않은 문제**:
- 여전히 보상 모델 의존
- 배포 외(OOD) 일반화의 한계
- 계산 비용 여전히 상당
- 보상 해킹의 근본적 해결 없음

**미래 방향**:
- 생성형 보상 모델과의 결합
- 더 강건한 오프라인 RL 이론 개발
- 멀티모달과 추론 능력 확장
- 보상 모델 없는 정렬 방법 탐색

***

## 참고 문헌

[1](https://arxiv.org/html/2410.19720)
[2](https://arxiv.org/pdf/2304.05302.pdf)
[3](https://arxiv.org/abs/2402.08925)
[4](https://arxiv.org/abs/2406.00832)
[5](https://arxiv.org/abs/2403.08635)
[6](https://arxiv.org/abs/2405.19332)
[7](https://arxiv.org/abs/2402.06147)
[8](https://arxiv.org/abs/2407.19594)
[9](https://arxiv.org/abs/2402.10038)
[10](https://arxiv.org/abs/2404.00934)
[11](https://arxiv.org/pdf/2403.06754.pdf)
[12](https://arxiv.org/pdf/2502.13417.pdf)
[13](http://arxiv.org/pdf/2403.14238.pdf)
[14](https://arxiv.org/pdf/2312.00849.pdf)
[15](http://arxiv.org/pdf/2406.15567.pdf)
[16](http://arxiv.org/pdf/2403.16649.pdf)
[17](https://arxiv.org/pdf/2405.18718.pdf)
[18](https://www.youtube.com/watch?v=Ak0vkBKOz0U)
[19](https://aclanthology.org/2024.emnlp-main.35.pdf)
[20](https://arxiv.org/pdf/2504.02495.pdf)
[21](https://arxiv.org/html/2503.06072v3)
[22](https://ieeexplore.ieee.org/document/10657686/)
[23](https://arxiv.org/abs/2311.16839)
[24](https://arxiv.org/abs/2309.16240)
[25](https://aclanthology.org/2024.findings-acl.630)
[26](https://arxiv.org/abs/2312.16430)
[27](https://arxiv.org/abs/2312.10584)
[28](https://www.semanticscholar.org/paper/44a9d8b0314d34aff91ccff9207d38eed37216ed)
[29](https://arxiv.org/pdf/2502.14356.pdf)
[30](https://arxiv.org/html/2403.19270v1?category=Press+Release)
[31](https://arxiv.org/pdf/2502.01930.pdf)
[32](https://arxiv.org/html/2410.04350v1)
[33](http://arxiv.org/pdf/2503.15880.pdf)
[34](https://aclanthology.org/2024.findings-acl.592.pdf)
[35](https://arxiv.org/abs/2506.21495)
[36](https://arxiv.org/html/2510.23393v1)
[37](https://arxiv.org/pdf/2305.18290.pdf)
[38](https://arxiv.org/pdf/2505.17508.pdf)
[39](https://arxiv.org/html/2509.00347v1)
[40](https://arxiv.org/abs/2311.12908)
[41](https://arxiv.org/pdf/2503.19595.pdf)
[42](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1646532e-c9dc-4672-b35a-c9b0ce2cecfb/2304.06767v4.pdf)
[43](https://www.semanticscholar.org/paper/db32da8f3b075d566a73512f4ccc2c95449c75a1)
[44](https://arxiv.org/abs/2405.19544)
[45](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_RLHF-V_Towards_Trustworthy_MLLMs_via_Behavior_Alignment_from_Fine-grained_Correctional_CVPR_2024_paper.pdf)
[46](https://innodata.com/reward-modeling-for-generative-ai/)
[47](https://incubity.ambilio.com/supervised-fine-tuning-vs-rlhf-for-llms/)
[48](https://proceedings.iclr.cc/paper_files/paper/2024/file/477b39bfb9db99f60914dbfed5f23eb2-Paper-Conference.pdf)
[49](https://invisibletech.ai/blog/supervised-fine-tuning-vs-rlhf-how-to-choose-the-right-approach-to-train-your-llm)
[50](https://www.synthlabs.ai/pdf/Generative_Reward_Models.pdf)
[51](https://arxiv.org/abs/2406.10305)
[52](https://arxiv.org/html/2511.03939v1)
[53](https://arxiv.org/pdf/2505.24119.pdf)
[54](https://arxiv.org/html/2512.01354v3)
[55](https://www.arxiv.org/pdf/2510.03231.pdf)
[56](https://arxiv.org/pdf/2502.19402.pdf)
[57](https://arxiv.org/html/2508.00737v2)
[58](https://arxiv.org/pdf/2505.02686.pdf)
[59](https://arxiv.org/html/2504.07912v2)
[60](https://arxiv.org/pdf/2410.17055.pdf)
[61](https://aclanthology.org/anthology-files/pdf/emnlp/2024.emnlp-main.816.pdf)
[62](https://arxiv.org/abs/2310.03708)
[63](https://arxiv.org/abs/2311.08380)
[64](https://arxiv.org/abs/2309.06657)
[65](http://arxiv.org/pdf/2407.09072.pdf)
[66](https://arxiv.org/html/2503.01076v1)
[67](https://arxiv.org/abs/2305.18290)
[68](https://genai-personalization.github.io/assets/papers/GenAIRecP2024/Sharma.pdf)
[69](https://proceedings.mlr.press/v202/wagenmaker23a/wagenmaker23a.pdf)
[70](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
[71](https://ebbnflow.tistory.com/382)
[72](https://richardli.xyz/post/topk-routing-stability-gap/)
[73](http://nanjiang.cs.illinois.edu/files/STS_Special_Issue_Offline_RL.pdf)
[74](https://github.com/eric-mitchell/direct-preference-optimization)
[75](https://arxiv.org/abs/2406.04274)
[76](https://arxiv.org/html/2410.15595v3)
[77](https://platform.openai.com/docs/guides/direct-preference-optimization)
[78](https://openreview.net/forum?id=BmkOKYfbmV)
[79](https://openreview.net/forum?id=phAlw3JPms)
