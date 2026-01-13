# Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study

### 1. 핵심 주장 및 주요 기여 요약

"Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study"는 **DPO(Direct Preference Optimization)가 PPO(Proximal Policy Optimization)보다 우월하지 않으며, 적절히 튜닝된 PPO가 모든 벤치마크에서 최고의 성능을 달성할 수 있음**을 입증하는 포괄적인 연구입니다.

**주요 기여:**

1. **DPO의 근본적 한계를 이론적으로 증명**: Theorem 4.1을 통해 DPO가 분포 외(out-of-distribution) 응답을 활용하는 편향된 정책을 발견할 수 있음을 보여줌
2. **PPO의 핵심 성능 향상 요소 규명**: 이점 정규화(advantage normalization), 대용량 배치 크기, 참조 모델의 지수 이동 평균(exponential moving average) 업데이트 등 3가지 핵심 요소 발견
3. **광범위한 실증 검증**: 대화 생성부터 도전적인 코드 경쟁까지 다양한 RLHF 벤치마크에서 PPO의 우월성 확인

***

### 2. 문제 정의, 제안 방법, 모델 구조 상세 분석

#### 2.1 해결하고자 하는 문제

RLHF 분야에서 역설적인 현상이 존재합니다:
- **실무 측면**: ChatGPT, Claude 등 성공적인 LLM 애플리케이션은 **보상 기반 PPO 방식** 사용
- **학술 벤치마크**: 최고 성능은 종종 **보상 없는 DPO 방식**에서 달성됨

이러한 불일치를 해명하는 것이 논문의 핵심 목표입니다.

#### 2.2 RLHF 기초 이론

RLHF의 기본 목적함수:

$$J_r(\theta) = \mathbb{E}_{x \sim \text{data}, y \sim \pi_\theta} [r(x,y)] - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{\text{ref}})$$

여기서:
- $$r(x,y)$$ : 보상 함수
- $$\pi_\theta$$ : 현재 정책(학습 대상)
- $$\pi_{\text{ref}}$$ : 참조 정책(SFT 모델)
- $$\beta$$ : KL 페널티 계수

#### 2.3 PPO 방식 (보상 기반)

**1단계: 보상 모델 학습**

선호도 데이터 $$D = \{x, y_w, y_l\}$$ (우승, 패배 응답)에서 Bradley-Terry 모델 가정 하에:

$$P(y_w > y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))$$

보상 모델 손실:

$$\mathcal{L}_R(r) = \mathbb{E}_{x,y_w,y_l \sim D} [-\log \sigma(r(x, y_w) - r(x, y_l))]$$

**2단계: PPO를 이용한 정책 최적화**

$$J_r(\theta) = \mathbb{E} \left[ \min \left( \frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)} A_t, \text{clip}\left(\frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)}, 1-\epsilon, 1+\epsilon\right) A_t \right) \right] - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{\text{ref}})$$

#### 2.4 DPO 방식 (보상 없는)

**핵심 혁신**: 보상을 정책으로부터 직접 도출

최적 정책 조건 (Eq. 5):

$$\pi^*(y|x) = \frac{1}{Z_x} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

이를 역으로 풀면 암묵적 보상:

$$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + C(x)$$

따라서 DPO 손실 함수 (Eq. 7):

$$\mathcal{L}_{DPO} = \mathbb{E}_{x,y_w,y_l \sim D} \left[ -\log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

***

### 3. DPO의 근본적 한계: 이론적 분석

#### 3.1 Theorem 4.1: PPO는 DPO의 진부분집합

**정리**: 주어진 실제 보상 $$r$$과 선호도 데이터 $$D$$에 대해, $$\Pi_{\text{PPO}} \subset \Pi_{\text{DPO}}$$ (진부분집합)

**의미**: 
- PPO가 발견하는 모든 정책은 DPO 손실을 최소화함
- 그러나 DPO는 PPO가 발견할 수 없는 추가 정책을 찾을 수 있음
- 이 추가 정책들이 반드시 좋은 성능을 보장하지 않음

#### 3.2 분포 외(OOD) 데이터 취약성

**핵심 문제**: DPO의 손실함수는 **상대적 확률의 비만 고려**

선호도 데이터에 없는 응답 $$y_{\text{ood}}$$에 대해:
- 보상 모델은 잘못된 높은 값 할당 가능
- DPO는 이 오류를 보상하기 위해 $$\pi_\theta(y_{\text{ood}})$$를 증가시킬 수 있음
- 참조 정책의 KL 제약이 약하면 위험할 수 있음

**구체적 예시 (Table 1, 상태-행동 3개)**:

| 행동 | $$\pi_{\text{ref}}$$ | 선호도 | DPO | PPO |
|------|---------|---------|-----|-----|
| $$y_1$$ | 0.5 | $$>$$ $$y_2$$ | 0.1 | - |
| $$y_2$$ | 0.5 | (없음) | 0.0 | - |
| $$y_3$$ | 0 | (없음) | 0.9 | 0 |

- **DPO**: 데이터에 없는 $$y_3$$에 90% 확률 할당 (위험!)
- **PPO**: KL 제약으로 $$y_3$$에 0% 유지 (안전)

#### 3.3 실제 선호도 데이터에서의 검증

SafeRLHF 데이터셋 실험 (Table 2):

**기저 모델 영향** (분포 시프트):
- SFT-Alpaca 기반: DPO 안전도 55.4% (매우 낮음)
- SFT-Safe 기반: DPO 안전도 71.8% (+16.4% 개선)
- 원인: 선호도 데이터와 기저 모델의 분포 불일치

**반복 DPO (DPO-Iter) 효과**:
- Iteration 4 후 안전도: 99.9% (PPO와 동등)
- 그러나 도움도: -2.96 (PPO의 1.69보다 훨씬 낮음)
- 시사점: DPO는 모델 생성 데이터로 반복 개선 필요, 그래도 부족

***

### 4. PPO의 핵심 성능 향상 요소

#### 4.1 세 가지 핵심 기술 요소

**실험 설정** (Table 3, HH-RLHF 및 코드 작업):

| 기법 | 효과 | OpenAssistant | APPS-Intro | CodeContest |
|-----|------|---|---|---|
| 기저 PPO | 기준 | 0.706 | 10.1 | - |
| +이점 정규화 | 안정성 ↑ | 0.716 | 11.4 | 9.1 |
| +대용량 배치 | **가장 효과적** | 0.716 | **14.6** | **18.0** |
| +참조 EMA | 미세 조정 | 0.718 | 18.0 | 21.4 |

#### 4.2 상세 분석

**1. 이점 정규화 (Advantage Normalization)**

$$A_t^{\text{norm}} = \frac{A_t - \mu(A)}{\sigma(A) + \epsilon}$$

- PPO 훈련을 안정화
- 서로 다른 작업과 배치에서 이점 스케일 일관성 보장
- 특히 작은 배치에서 큰 성능 저하 방지

**2. 대용량 배치 크기 (Large Batch Size)**

배치 크기 64 → 512로 증가 (Figure 2):

- 계산량 증가로 온-정책 데이터 품질 향상
- 분산(variance) 감소로 보상 신호 안정화
- 코드 생성 작업에서 가장 눈에 띄는 개선:
  - 초급: 10.1% → 38.6% (+283%)
  - 경쟁: 3.9% → 38.6% (+890%)
- KL 제약 강화로 분포 시프트 방지

**3. 참조 모델의 지수 이동 평균 업데이트 (Reference EMA)**

$$\pi_{\text{ref}}^{(t)} \leftarrow \tau \pi_\theta^{(t)} + (1-\tau) \pi_{\text{ref}}^{(t-1)}$$

(일반적으로 $$\tau = 0.01$$)

- 정책이 빠르게 변할 때, 고정 참조 모델은 과도한 KL 제약 가능
- EMA 업데이트로 참조 모델이 현재 정책과 같이 진화
- 도전적인 작업(코드)에서 성능 향상:
  - CodeContest: 18.0% → 21.4%
  - 코드 생성은 광범위한 시간/공간 탐색 필요

#### 4.3 초결정적 요소: 대용량 배치의 중요성

배치 크기별 성능 (APPS, CodeLlama-13B, Table 12 참조 내용):
- 배치 64: Intro 33.7%, Inter 8.7%, Comp 3.6%
- 배치 256: 성능 향상 보임
- 배치 512: 최고 성능 달성

**해석**:
$$\text{Effective Step Size} \propto \text{Batch Size}$$

대형 배치는 큰 스텝 크기를 안전하게 허용하여 정책 개선 속도 가속

***

### 5. 모델의 일반화 성능 향상 가능성 심층 분석

#### 5.1 분포 외 일반화: PPO의 우월성

**핵심 기제**: KL 제약의 역할

$$\beta \mathbb{D}_{KL}(\pi_\theta(y|x) || \pi_{\text{ref}}(y|x)) = \beta \sum_y \pi_\theta(y|x) \left[ \log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x) \right]$$

이 항이 보장하는 것:
1. **정책의 급격한 편차 방지**: 참조 정책의 확률이 0인 행동은 높은 확률 불가능
2. **알려진 좋은 행동 보존**: SFT 모델의 확률 구조 유지
3. **OOD 일반화**: 훈련 데이터 외 영역에서도 합리적 행동 가능

**Figure 1 비교**: 합성 시나리오
- DPO: 선호도 데이터에 없는 행동에 높은 확률 할당 (위험한 편차)
- PPO: KL 제약으로 보수적 확률 유지 (안전한 일반화)

#### 5.2 다양한 작업에서의 일반화

**대화 작업 (일반화 어려움)**:
- 성능 차이 작음 (PPO 0.718 vs DPO 0.611)
- 이유: 자연스러운 응답이 학습 데이터 분포 내에 많음
- 일반화 필요성 낮음

**코드 생성 (일반화 매우 중요)**:
- 성능 차이 극대화 (PPO 44.4% vs DPO 34.2%, APPS-pass5)
- 경쟁 수준: PPO 22.4% vs DPO 3.2% (CodeContest-pass100k)
- 이유: 
  - 새로운 알고리즘 패턴 발견 필요 (OOD 영역 광활)
  - 작은 확률 변화가 기능성에 큰 영향
  - 테스트 케이스 커버리지 불완전

#### 5.3 암묵적 보상 모델의 일반화 한계

**최근 연구 비교** (, 2024년 9월):

실험: 분포 시프트가 있는 검증 데이터에서 암묵적 보상(DPO)과 명시적 보상(EXRM) 비교

| 설정 | 명시적(EXRM) | 암묵적(DPO) | 격차 |
|-----|------|------|------|
| 훈련 데이터 | ~98% | ~98% | 정확도 유사 |
| OOD 검증 | ~96% | ~93% | **3% 저하** |
| 극단적 OOD | ~94% | ~87% | **7% 저하** |

**결론**: DPO의 암묵적 보상은 OOD 상황에서 빠르게 성능 저하

#### 5.4 반복 DPO (DPO-Iter)의 한계

**메커니즘**: 모델이 생성한 응답으로 새 선호도 쌍 구성, 보상 모델 라벨링

$$D_{\text{new}} = \{(x, y_{\text{good}}, y_{\text{bad}}) : y_{\text{good}} \text{ scores higher than } y_{\text{bad}}\}$$

**실험 결과**:

SafeRLHF (Table 2, DPO-Iter 반복 4회):
- 안전도: 99.9% (달성! PPO 수준)
- 도움도: -2.96 (PPO의 1.69 대비 **4.65 차이**)
- 해석: 안전만 과학하고, 실제 도움 능력은 여전히 부족

코드 경쟁 (Table 8, CodeContest):
- DPO: 0.0% (완전 실패)
- DPO-Iter: 3.2%
- PPO: **22.4%** (거의 7배 좋음)

**이유**:
1. 보상 모델의 완전성 한계: 올바른 코드 판별은 어려움
2. 반복이 편향 축적: 초기 실수가 증폭될 수 있음
3. 탐사-활용(Exploration-Exploitation) 불균형: DPO는 로컬 최적점에 갇힐 수 있음

***

### 6. 성능 향상 및 한계

#### 6.1 벤치마크별 성능 비교 (종합 결과)

**HH-RLHF (대화, Llama-2-7B)**:

| 방법 | OpenAssistant 보상 | GPT-4 선호도 (PPO vs 상대) |
|-----|------|------|
| SFT | 0.532 | - |
| RRHF | 0.523 | 28% (PPO가 28%, 동등 33%, RRHF가 39%) |
| PRO | 0.529 | 37% vs 26% vs 37% |
| DPO | 0.611 | 55% vs 21% vs 24% |
| DPO-Iter | 0.678 | 55% vs 18% vs 27% |
| **PPO** | **0.718** | **57% vs 21% vs 22%** |

**APPS (코드 생성, CodeLlama)**:

| 모델 | SFT | DPO-Iter | PPO | 상태 |
|-----|------|------|------|------|
| 7B | 30.0% | 20.9% | 29.4% | PPO가 거의 SFT 수준 |
| 13B | 33.7% | 33.0% | 36.4% | PPO만 개선 (+2.7%) |
| **34B** | **38.6%** | **34.2%** | **44.4%** | **PPO가 5.8% 우월** |

**CodeContest (극도로 도전적, CodeLlama-34B)**:

$$\text{pass}_{100k} = \text{100개 샘플 중 10개만 채점}$$

| 방법 | 유효성 | 테스트 |
|-----|------|------|
| AlphaCode-41B | 9.0 | 16.4% |
| SFT | 10.3 | 15.2% |
| DPO-Iter | 3.5 | 3.2% |
| **PPO** | **19.7** | **22.4%** |

**분석**:
- PPO가 16.4% → 22.4% (101k 개선)
- 이는 AlphaCode-41B (41B 파라미터)를 CodeLlama-34B(34B 파라미터)로 능가
- DPO는 완전히 실패 (0.0% → 3.2%)

#### 6.2 PPO의 한계

**1. 계산 비용**:
- 온-정책 샘플링: 재훈련할 때마다 새로운 응답 생성
- 보상 모델 훈련: 추가 학습 단계 필요
- 전체 RLHF 파이프라인이 복잡함

**2. 하이퍼파라미터 민감성**:
- KL 페널티 $$\beta$$: 0.05, 0.1, 0.2 등 많은 시도 필요
- 배치 크기, 학습률 등 많은 튜닝 필요 가능
- 안정성이 DPO보다 낮을 수 있음

**3. 메모리/속도**:
- 참조 모델과 현재 모델 동시 로드 필요 (메모리 2배)
- 온-정책 샘플링으로 인한 속도 저하

#### 6.3 DPO의 한계

**1. OOD 데이터 취약성** (이미 상세 분석):
- 선호도 데이터의 분포 외 영역에서 잘못된 정책 발견 가능
- 대용량 모델일수록 심각할 가능성

**2. 이론적 한계** (Theorem 4.1):
- DPO가 찾는 최적 정책이 실제 RLHF 목적함수를 최대화하지 않을 수 있음
- 암묵적 보상이 명시적 보상과 다를 수 있음

**3. 도전적 작업에서의 성능**:
- 코드 생성처럼 넓은 탐색 공간이 필요한 작업에서 매우 약함
- 반복 DPO도 PPO의 절반 이하 성능

***

### 7. 2020년 이후 관련 최신 연구 비교 분석

#### 7.1 DPO의 진화 (2023-2025)

**원본 DPO (Rafailov et al., 2023)**:[1]
- 최초 제안: 보상 모델 없이 선호도 최적화
- 혁신: 암묵적 보상 개념
- 한계: 오프라인 데이터에 제한

**RS-DPO (Feb 2024)**:[2]
- 거부 샘플링 + DPO 결합
- 모델이 생성한 응답 중 좋은 것만 선택하여 쌍 구성
- 개선: DPO-Iter보다 더 효과적

**Cal-DPO (Dec 2024)**:[3]
- 암묵적 보상의 **절대값 스케일** 보정
- 기존 DPO: 상대적 순서만 맞춤 (예: $$r_w > r_l$$ 여부만)
- Cal-DPO: 절대값도 실제 보상과 비슷하게 → 더 나은 정책 도출
- 개선: 여러 벤치마크에서 기존 방법 능가

**WDPO/Distributionally Robust DPO (Feb 2025)**:[4]
- Wasserstein 거리 기반 강건성
- 분포 시프트에 견딜 수 있는 DPO 변종
- 동기: 정확히 본 논문이 지적한 분포 시프트 문제 해결

**RPO (Feb 2024)**:[5]
- **상대 선호도**: 같은 프롬프트뿐 아니라 유사 프롬프트의 비교도 학습
- 더 많은 훈련 신호 활용
- 개선: 더 나은 일반화

#### 7.2 PPO의 개선 및 분석 (2023-2025)

**Secrets of RLHF in Large Language Models Part I: PPO (July 2023)**:[6]
- 제목: "RLHF에서 PPO의 비결"
- 내용: 구현 디테일이 매우 중요함을 강조
- 본 논문과 유사: PPO의 특정 기법이 성능 결정

**Pairwise PPO (P3O) (2024)**:[7]
- PPO의 문제: 보상 모델이 같은 정보를 제공해도 결과가 다를 수 있음
- 원인: Bradley-Terry 손실은 상수 시프트에 불변이지만 PPO는 아님
- 해결: 비교 정보를 직접 사용하는 P3O
- 성과: PPO와 DPO의 이점 결합

**Value-based Calibration (VCB) (2024)**:[8]
- PPO의 복잡성 지적
- 가치함수 학습 없이 보상만 사용하는 방법
- 성과: DPO와 유사 안정성, PPO 수준의 성능

**Model Averaging for Alignment Tax (2024)**:[9]
- PPO의 문제: 정렬 성능은 높지만 NLP 벤치마크 성능 저하 ("alignment tax")
- 해결: 사전/사후 모델 평균화
- 기여: 이 논문과 다른 각도로 PPO의 트레이드오프 분석

#### 7.3 암묵적 vs 명시적 보상 비교

**"On the Limited Generalization Capability of the Implicit Reward Model Induced by Direct Preference Optimization" (Sep 2024)**:[10]

| 측면 | 명시적(EXRM) | 암묵적(DPO) |
|-----|------|------|
| 훈련 정확도 | ~98% | ~98% |
| ID 테스트 정확도 | ~98% | ~98% |
| 약한 OOD 정확도 | ~96% | ~93% |
| 강한 OOD 정확도 | ~94% | ~87% |
| **평균 저하** | 매우 작음 | **3-7%** |

**결론**: DPO의 암묵적 보상은 분포 시프트에 취약 (본 논문과 일치)

#### 7.4 최신 트렌드 (2025)

**FocalPO (Jan 2025)**:[11]
- DPO 훈련이 실제로 선호도를 거의 개선하지 않는다는 발견
- 손실이 감소해도 성능이 증가하지 않는 모순
- 제안: 올바르게 순위 매겨진 쌍에만 집중

**Improving LLM Alignment via Preference Data Selection (Feb 2025)**:[12]
- DPO의 성능은 선호도 데이터 품질에 매우 의존함
- 노이즈 있는 선호도 제거 필요

**A Comprehensive Survey of Direct Preference Optimization (Oct 2024)**:[13]
- 최근 DPO 논문 수십 개 종합 분석
- 결론: DPO는 개선 중이지만 여전히 온-정책 PPO에 미치지 못함
- 향후 방향: DPO 변종들의 온-정책 성능 개선 필요

#### 7.5 연구 흐름 종합

**시간대별 주요 주제**:

1. **2023년**: 
   - DPO 원본 제안 (Rafailov) 및 빠른 채택
   - PPO 분석 (Zheng et al.)

2. **2024년 상반기**:
   - 본 논문 (Apr): DPO 한계 명확히 함
   - 다양한 DPO 개선 제안 (RS-DPO, RPO, 등)

3. **2024년 하반기**:
   - 암묵적 vs 명시적 보상 비교 연구
   - DPO 강건성 개선 (WDPO, Cal-DPO)
   - P3O 등 대안적 접근

4. **2025년**:
   - DPO 한계 재확인 (FocalPO)
   - 데이터 선택 중요성 강조
   - 하이브리드 방법들 증가 (PP + DPO, etc.)

***

### 8. 앞으로의 연구에 미치는 영향과 고려사항

#### 8.1 학계에 미친 영향

**1. DPO 역사 수정**:
- 초기 DPO 논문: "RLHF와 동등하거나 더 좋음" (Rafailov, 2023)
- 본 논문: "특정 상황에서는 RLHF(PPO)가 우월" (Xu et al., 2024)
- 후속 논문들: 이 발견을 기반으로 더 정교한 분석 진행

**2. PPO 재평가**:
- PPO의 복잡성이 "버그"가 아니라 "기능"임을 입증
- 대용량 배치, EMA 등이 우연이 아니라 필수임을 증명

**3. OOD 일반화의 중요성**:
- LLM 정렬에서 분포 외 일반화가 핵심임을 강조
- 후속 연구들 (WDPO, Cal-DPO)이 이 문제에 집중

#### 8.2 실무(Production)에 미친 영향

**1. 대형 모델 개발**:
- OpenAI, Anthropic 등: 이미 PPO 사용 중 (이론적 정당성 증가)
- 작은 기업: DPO의 단순성으로 접근성은 여전히 높음

**2. 트레이드오프 명확화**:

| 관점 | 선택 | 이유 |
|-----|------|------|
| 최고 성능 추구 | PPO | 코드, 추론 등 도전적 작업에서 필수 |
| 빠른 프로토타입 | DPO | 구현 간단, 안정성 좋음 |
| 제한된 자원 | DPO | 메모리, 계산량 적음 |
| 시간 충분 | PPO + 튜닝 | 최고 성능 달성 가능 |

#### 8.3 미해결 문제와 향후 연구 방향

**1. 좋은 점**: 
- 왜 대형 배치가 PPO 성능을 극적으로 향상시키는가?
- 이론적 분석 필요

**현황**: 본 논문은 경험적 발견만 제시, 깊은 이론 분석 없음

**시도**:
- 최근 "신경망 동역학" 관점에서의 PPO 분석 증가
- 그러나 LLM 규모에서의 분석은 아직 미흡

**2. DPO 개선 이론**:
- Cal-DPO: 보상 크기 조정
- WDPO: 분포 강건성
- 이들의 조합이 PPO 수준에 도달할 수 있을까?

**전망**: 아직 미결정, 2025-2026년 연구 주제

**3. 온-정책 vs 오프라인**:
- PPO는 온-정책: 훈련 중 모델이 생성한 데이터 사용
- DPO는 오프라인: 고정 선호도 데이터만 사용

**질문**: 오프라인 DPO에 온-정책 요소 추가 가능?

**답변**: 네, 이미 시도 (Iterative DPO, Best-of-N, etc.), 그러나 여전히 부족

**4. 보상 모델의 복구성**:
본 논문 한계 (저자들도 인정):
> "The reward model is significant in the training processes of both PPO and DPO-Iter. However, in this paper, we have not delved into the discussion of how to effectively train a robust reward model."

미래 방향:
- 보상 모델 안정성 개선
- 노이즈 있는 선호도 처리
- 다목표 보상 (안전성, 도움도, etc.)

#### 8.4 연구자를 위한 구체적 가이드라인

**LLM 정렬 연구 시 체크리스트**:

```
□ 1. 작업 특성 파악
  ├─ 단순 작업 (QA, 대화)?  → DPO 충분할 가능성
  └─ 복잡 작업 (코드, 수학)?  → PPO 필수

□ 2. 자원 제약 확인
  ├─ GPU 메모리 부족?  → DPO 고려
  └─ 충분함?  → PPO 권장

□ 3. 선호도 데이터 품질 검토
  ├─ 노이즈 많음?  → DPO 위험
  └─ 고품질?  → DPO 및 PPO 모두 가능

□ 4. 벤치마킹 설계
  ├─ ID 성능만?  → DPO도 괜찮음
  ├─ OOD 성능도?  → PPO 권장
  └─ 둘 다?  → P3O, Cal-DPO 등 혼합 고려

□ 5. 하이퍼파라미터 튜닝
  ├─ DPO: β, 학습률, 배치크기 (적음)
  └─ PPO: KL계수, 배치크기, EMA계수 (많음)
```

**논문 제출 시 권장사항**:

1. **OOD 평가 포함**: ID 테스트만으로는 불충분
2. **분포 시프트 분석**: 기저 모델과 선호도 데이터의 일치도 보고
3. **반복 개선 추적**: Iterative 방법인 경우 수렴 곡선 제시
4. **계산 비용 명시**: 메모리, 시간, GPU 사용량 보고
5. **인간 평가 포함**: GPT-4 평가는 참고용, 인간 평가가 더 신뢰성 있음

#### 8.5 실제 적용 시나리오별 권고안

**시나리오 1: 스타트업이 빠르게 모델 배포 필요**
- 선택: DPO
- 이유: 구현 간단, 메모리 효율적
- 주의: 도메인 내 성능만 보장, OOD 성능 낮을 수 있음
- 대비: 후속 PPO 파인튜닝 준비

**시나리오 2: 대형 연구실의 SOTA 달성 목표**
- 선택: PPO (+ 대용량 배치 + EMA)
- 이유: 최고 성능 달성
- 투자: 시간, 계산 자원
- 고도화: P3O, Cal-DPO 등의 수정사항 적용

**시나리오 3: 제한된 라벨 예산**
- 선택: Iterative DPO (또는 RS-DPO)
- 이유: 모델이 생성한 데이터로 반복 개선
- 주의: 여전히 PPO에 못 미칠 가능성
- 보완: 고품질 초기 선호도 데이터 수집에 투자

**시나리오 4: 다국어 또는 낮은 자원 언어**
- 선택: DPO (다국어 전이 학습)
- 이유: 각 언어마다 PPO 튜닝 비싼 비용
- 참고: 최근 DPO 다국어 연구 증가 ()

***

### 9. 결론 및 종합 평가

#### 9.1 주요 발견 종합

| 항목 | 발견 | 영향 |
|-----|------|------|
| **이론적** | DPO는 OOD 해결책이 아님 (Theorem 4.1) | DPO의 한계 명확화 |
| **경험적** | PPO의 3가지 핵심 요소 규명 | 재현 가능한 가이드 제공 |
| **실무적** | 작업 유형에 따라 선택 필요 | 맥락 기반 의사결정 가능 |
| **전망적** | 개선된 DPO 변종 들의 등장 | 하이브리드 접근 증가 |

#### 9.2 본 논문의 강점

✅ 명확한 문제 정의: DPO vs PPO 모순 직접 다룸

✅ 엄밀한 이론: Theorem 4.1로 직관 정량화

✅ 광범위한 실험: 4개 주요 벤치마크 + ablation studies

✅ 실용성: 재현 가능한 기법 제시 (대용량 배치, EMA 등)

✅ 명확한 제시: 표, 그래프로 핵심 결과 강조

#### 9.3 제한사항 및 비판점

❌ 보상 모델 분석 부재: 보상 모델 품질의 역할 미충분

❌ 이론의 한계: Theorem 4.1은 최악의 경우, 평균적 상황 분석 없음

❌ 모델 크기 범위: 최대 34B까지만 실험 (7B, 13B, 34B)

❌ DPO 개선 방법 간단함: 단순 반복(DPO-Iter)만 고려, RS-DPO 등 더 정교한 방법 미반영

❌ 참조 모델 선택: 고정 참조 vs EMA의 영향만 보았지, 다른 참조 선택 미검토

#### 9.4 본 연구가 학계에 제기한 질문들

1. **DPO는 정말 보상 없는가?** 
   → 아니다, 암묵적 보상을 가지는데 이것이 일반화를 잘 하지 못함

2. **왜 학술 벤치마크에서는 DPO가 우수해 보이는가?**
   → 많은 벤치마크가 훈련 분포 내 성능만 평가하기 때문

3. **대용량 배치는 왜 그렇게 중요한가?**
   → 정책 그래디언트의 분산 감소로 온-정책 데이터 품질 향상 → 후속 연구 주제

4. **PPO는 복잡하지만 왜 여전히 사용되는가?**
   → 우월한 성능이 복잡성을 정당화함, 특히 어려운 작업에서

5. **DPO는 미래가 없는가?**
   → 아니다, 개선되고 있지만 온-정책 요소 추가 필요

***

### 10. 최신 연구 동향 최종 정리 (2020-2025)

$$\text{시간축}: \text{DPO 제안} \xrightarrow{\text{2023}} \text{이 논문} \xrightarrow{\text{2024}} \text{DPO 개선들} \xrightarrow{\text{2025}} \text{하이브리드 방법들}$$

**2020-2022**: RLHF 기초 다지기
- InstructGPT (Ouyang et al., 2022): PPO-기반 RLHF 성공

**2023**: DPO 혁명
- Rafailov et al.: "보상 모델 없이도 가능" 주장 (6,889 인용)

**2024 상반기**: 이 논문과 DPO 한계 지적
- Xu et al. (Apr 2024): PPO 우월성 증명
- 동시에 다양한 DPO 개선 시작

**2024 하반기**: 정밀한 분석
- OOD 일반화 비교 (Li et al., Sep 2024)
- 강건한 DPO 제안들 (WDPO, Cal-DPO)

**2025**: 수렴 단계
- DPO의 남은 문제들 체계적으로 해결
- PPO + DPO 혼합 방법들 등장
- 모달리티 확장 (코드, 수학, 비전 등)

***

## 마지막 종합 의견

"Is DPO Superior to PPO for LLM Alignment?"이라는 본 논문의 질문에 대한 답:

### ✅ 결론: **조건부로 아니다 (Conditionally No)**

**단순 대답**: 도전적인 작업(코드, 수학, 복잡한 추론)에서는 **PPO가 확실히 우월**

**미묘한 지점**:
- 대화처럼 자연스러운 작업에서는 차이 적음
- DPO는 구현 단순성과 자원 효율성 제공
- 선호도 데이터 품질이 충분하다면 DPO도 경쟁력

**향후 전망**:
- 개선된 DPO 변종들 (Cal-DPO, WDPO)이 격차 축소 가능
- 하이브리드 방법들(P3O, RS-DPO + PPO)이 최적 선택 가능
- 작업별/자원별로 맞춤형 선택이 새로운 표준

**연구자와 실무자를 위한 최종 권고**:

> "DPO의 단순성은 매력적이지만, 도전적인 작업이나 높은 성능이 필요할 때는 PPO의 우월성을 무시할 수 없다. 향후 2-3년간 개선된 DPO 변종들이 이 격차를 줄일 가능성이 있으므로, 최신 방법들을 면밀히 추적하되, 현재로서는 **작업 특성과 자원 제약을 고려한 의도적 선택**이 최선이다."

***

**참고자료 ID 맵핑**:

[1](https://arxiv.org/abs/2305.18290)
[2](https://arxiv.org/abs/2402.10038)
[3](https://arxiv.org/abs/2412.14516)
[4](https://arxiv.org/pdf/2502.01930.pdf)
[5](https://arxiv.org/abs/2402.10958)
[6](https://arxiv.org/abs/2307.04964)
[7](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-21.pdf)
[8](https://aclanthology.org/2024.emnlp-main.976.pdf)
[9](https://aclanthology.org/2024.emnlp-main.35.pdf)
[10](https://arxiv.org/abs/2409.03650)
[11](https://arxiv.org/pdf/2501.06645.pdf)
[12](https://arxiv.org/html/2502.14560v1)
[13](https://arxiv.org/html/2410.15595v3)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f1bd1d92-fd93-45be-b279-51d98134cc92/2404.10719v3.pdf)
[15](https://arxiv.org/abs/2412.15453)
[16](https://arxiv.org/abs/2404.01258)
[17](https://arxiv.org/abs/2406.09920)
[18](https://arxiv.org/abs/2402.08005)
[19](https://arxiv.org/abs/2402.09320)
[20](https://arxiv.org/abs/2404.10719)
[21](https://arxiv.org/pdf/2407.07880.pdf)
[22](http://arxiv.org/pdf/2408.09834.pdf)
[23](https://arxiv.org/pdf/2502.16825.pdf)
[24](http://arxiv.org/pdf/2410.05939.pdf)
[25](http://arxiv.org/pdf/2404.14723.pdf)
[26](http://arxiv.org/pdf/2410.16586.pdf)
[27](https://cameronrwolfe.substack.com/p/direct-preference-optimization)
[28](https://neurips.cc/virtual/2024/poster/96611)
[29](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/SooWeiKoh.pdf)
[30](https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo)
[31](https://arxiv.org/pdf/2511.03939.pdf)
[32](https://openaccess.thecvf.com/content/CVPR2024/papers/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.pdf)
[33](https://arxiv.org/html/2510.21090v1)
[34](https://rlhfbook.com/c/12-direct-alignment)
[35](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffusion-dpo/)
[36](https://openreview.net/forum?id=7iaAlIlV2H)
[37](https://liner.com/review/reward-difference-optimization-for-sample-reweighting-in-offline-rlhf)
[38](https://arxiv.org/html/2404.10719v1)
[39](https://arxiv.org/pdf/2404.10719.pdf)
[40](https://arxiv.org/html/2508.09937v1)
[41](https://arxiv.org/pdf/2407.16216.pdf)
[42](https://arxiv.org/html/2505.23749v1)
[43](https://arxiv.org/html/2502.01930v1)
[44](https://arxiv.org/html/2410.23726v1)
[45](https://arxiv.org/html/2409.00162v2)
