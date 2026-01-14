
# RS-DPO: A Hybrid Rejection Sampling and Direct Preference Optimization Method for Alignment of Large Language Models

## 1. 핵심 주장 및 주요 기여 요약

**RS-DPO(Rejection Sampling + Direct Preference Optimization)**는 Amazon 연구팀이 제안한 하이브리드 방법으로, 대규모 언어모델(LLM) 정렬을 위해 거부 샘플링(Rejection Sampling)과 직접 선호도 최적화(DPO)를 체계적으로 결합합니다.[1]

### 1.1 주요 기여

**첫째**, RS-DPO는 **보상 모델 품질 변동에 대한 강건성**을 보여줍니다. 기존 PPO와 달리 보상 모델 품질 저하에 덜 민감하며, 단일 실행으로 안정적인 성능을 달성합니다.[1]

**둘째**, **선호도 쌍 생성의 효율성** 개선입니다. 순수 거부 샘플링(최고 보상 응답만 선택)과 달리, RS-DPO는 보상 분포를 기반으로 대비(contrastive) 샘플 쌍을 자동으로 생성하여 데이터 활용도를 극대화합니다.[1]

**셋째**, **모델 내부의 응답 생성**입니다. 다른 모델이나 인간 주석에서 대비 응답을 얻는 기존 DPO와 달리, RS-DPO는 SFT 모델에서 직접 응답을 샘플링하여 온-정책(on-policy) 강화학습을 실현합니다.[1]

**넷째**, **제한된 자원 환경에서의 효율성**입니다. PPO의 3개 모델(SFT, 정책, 보상) 동시 로딩 요구사항을 제거하고, 오프라인 샘플링으로 메모리 효율을 개선합니다.[1]

***

## 2. 해결하고자 하는 문제

### 2.1 기존 방법의 한계

**PPO 기반 RLHF의 문제점:**
- 불안정성과 높은 초매개변수 튜닝 필요성
- 계산 비용 증가 (보상 최대화를 위한 반복 샘플링)
- 다중 모델 로딩으로 인한 높은 GPU 메모리 요구 (1-2개 중간 사양 GPU로 7B 모델 학습 불가능)
- 보상 모델 품질에 대한 높은 민감도[1]

**순수 DPO의 제약:**
- 선호도 데이터를 인간 주석 또는 대안 LLM에서 얻음 (정책 모델 아님)
- 분포 이동(distribution shift) 문제: 선호도 데이터가 참조 정책에서 생성되지 않음
- 낮은 데이터 활용 효율[1]

**순수 거부 샘플링의 한계:**
- 각 프롬프트당 생성된 k개 응답 중 최고 보상 응답만 사용
- 나머지 k-1개 응답 정보 미활용
- 선호도 쌍 대신 절대 품질 샘플만 사용 (선호도 신호 약화)[1]

### 2.2 문제의 핵심

인간 선호도에 기반한 LLM 정렬에서 **"다양한 고품질 대비 샘플 쌍을 정책 모델 자체에서 효율적으로 생성하면서, 제한된 컴퓨팅 자원으로도 안정적 최적화가 가능한가?"**라는 근본적인 질문에 답해야 합니다.

***

## 3. 제안하는 방법: RS-DPO

### 3.1 전체 파이프라인

RS-DPO는 4단계로 구성됩니다:

**[단계 1] 지도 미세 조정 (SFT)**

$$L_{SFT} = \arg\max_{\theta} \sum_{(x,y) \in D_{sft}} \log \pi_{\theta}(y|x)$$

여기서 $\pi_{\theta}$는 정책 모델, $D_{sft}$는 고품질 지시-응답 쌍 데이터셋입니다.[1]

**[단계 2] 보상 모델 학습 (RM)**

Bradley-Terry 모델을 사용한 쌍 비교 손실:

$$p(y_w \succ y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}$$

$$R(x, y) = \arg\min_{\phi} \sum_{(x, y_l, y_w) \in D_{RM}} -\log \sigma(r(x, y_w) - r(x, y_l))$$

여기서 $\sigma$는 시그모이드 함수, $r(x,y)$는 보상 모델의 점수입니다.[1]

**[단계 3] 거부 샘플링을 통한 선호도 데이터 생성 (PDGRS)**

이것이 RS-DPO의 핵심 혁신입니다. 각 프롬프트 $x$에 대해:

1. SFT 모델에서 k개의 응답 샘플링: $y_{i1}, \ldots, y_{ik} \sim \pi_{SFT}(\cdot|x)$
2. 보상 모델로 각 응답 평가: $r_{ij} = R(x, y_{ij})$
3. 모든 가능한 쌍 조합에 대해 보상 간격 계산:

$$r_{gap} = \sigma\left(\frac{r_{ij} - r_{il}}{\tau}\right)$$

여기서 $\tau$는 온도 계수입니다.[1]

4. 임계값 초과 쌍만 선택: $r_{gap} > \eta$ (보통 $\eta = 0.85$)

이 방식으로 단일 프롬프트에서 $\binom{k}{2}$개 쌍을 생성할 수 있으며, 원본 데이터보다 훨씬 많은 선호도 쌍을 얻습니다.[1]

**Algorithm 1: PDGRS 과정**

```
결과: DP = {(x, y_l, y_w)}^{3m} : 선호도 데이터셋

입력:
  {x_1, ..., x_n}: D_RM에서 샘플링한 프롬프트
  π_SFT: SFT 모델
  R(x,y): 보상 모델
  τ: 온도
  η: 선호도 데이터 선택 임계값

for i = 1:n do
  (y_{i1}, ..., y_{ik}) | y_{ik} ~ π_SFT(·|x_i)  ▷ k개 응답 생성
  (r_{i1}, ..., r_{ik}) | r_{ij} = R(x_i, y_{ij})  ▷ 각 응답의 보상 계산
  
  for j = 1:k do
    for l = 1:k do
      if j == l then
        continue
      end if
      
      r_{gap} = σ(r_{ij} - r_{il} / τ)  ▷ 보상 간격 계산
      
      if r_{gap} > η then
        DP = {DP; (x_i, y_{il}, y_{ij})}  ▷ 수용된 샘플 추가
      end if
    end for
  end for
end for
```



**[단계 4] 직접 선호도 최적화 (DPO)**

생성된 선호도 데이터 $D_P$를 사용한 정책 최적화:

$$L_{RL} = \arg\max_{\theta} \sum_{(x, y_l, y_w) \in D_P} \log \sigma\left(\beta \log \frac{\pi_{RL}(y_w|x)}{\pi_{SFT}(y_w|x)} - \beta \log \frac{\pi_{RL}(y_l|x)}{\pi_{SFT}(y_l|x)}\right)$$

여기서 $\beta$는 KL 제약 계수(일반적으로 0.1), $\pi_{RL}$은 학습 중인 정책입니다.[1]

### 3.2 방법의 특징

**분포 이동 문제 완화:**
- 선호도 데이터가 SFT 모델에서 직접 생성되어 DPO의 참조 정책과 분포 일치
- 대안 모델이나 인간 주석의 분포 차이 제거[1]

**동적 데이터 크기 조절:**
- 임계값 $\eta$와 온도 $\tau$ 조절로 선택되는 쌍의 개수 제어
- 낮은 $\tau$는 더 많은 쌍 생성 (다양한 난이도 샘플 포함)[1]

**온-정책 학습:**
- 거부 샘플링은 참조 정책에서 응답을 생성하므로 오프라인 온-정책 학습 실현
- PPO의 실시간 샘플링 필요성 제거[1]

***

## 4. 모델 구조 및 구성

### 4.1 구조 개요

RS-DPO의 아키텍처는 기본적으로 표준 Transformer 기반 LLM 구조를 따르지만, 훈련 단계별 역할 분담이 특징입니다.[1]

```
┌─────────────────────────────────────────────────────┐
│              사전 학습 기본 모델 (π_base)             │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  [단계 1] SFT 미세 조정    │
        │  Loss: -log π_SFT(y|x)     │
        └────────────┬───────────────┘
                     │
    ┌────────────────┴────────────────┐
    │                                 │
    ▼                                 ▼
┌─────────────────────────┐  ┌──────────────────────┐
│  [단계 2] 보상 모델     │  │  [단계 3] 거부샘플링  │
│  학습                   │  │  선호도 생성 (PDGRS)  │
│  Bradley-Terry Loss     │  │                      │
│                         │  │ - k개 응답 샘플링    │
│ 출력: R(x,y) ∈ ℝ      │  │ - 보상 간격 계산     │
└─────────────────────────┘  │ - 임계값 선택        │
    │                        │ 출력: D_P (선호도)   │
    └────────────────┬───────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  [단계 4] DPO 최적화       │
        │  Loss: -log σ(β diff)      │
        │  참조 모델: π_SFT          │
        │  정책 모델: π_RL (갱신)    │
        └────────────┬───────────────┘
                     │
                     ▼
             최종 정렬 모델 (π_RL)
```

### 4.2 구성 요소 상세

**1. SFT 모델 (π_SFT)**
- 역할: 선호도 샘플 생성의 기반, DPO의 참조 정책
- 특징: Llama-2-7B 기반, LoRA 미적용[1]

**2. 보상 모델 (R)**
- 역할: 생성된 응답의 품질 평가
- 구조: pythia-6.9B 기반 (Transformer)
- 입력: (프롬프트, 응답) 쌍
- 출력: 스칼라 보상 값
- 훈련 데이터: Open Assistant, Anthropic HH-RLHF, WebGPT 등 다양한 출처 결합[1]

**3. 선호도 데이터셋 (D_P)**
- 생성 메커니즘: PDGRS
- 크기: 원본 데이터셋 크기의 3배~5배 (임계값에 따라)
  - $\eta = 0.90$: ~1.2배 확대
  - $\eta = 0.85$: ~3배 확대
  - $\eta = 0.80$: ~6배 확대[1]

**4. 정책 모델 (π_RL)**
- 역할: 최종 정렬 모델
- 최적화 대상: DPO 손실 함수
- 특징: LoRA (rank=8)를 사용한 매개변수 효율 미세 조정[1]

### 4.3 주요 하이퍼매개변수

| 매개변수 | 설정값 | 영향 |
|---------|-------|------|
| k (응답 샘플 개수) | 16 | 더 많은 선호도 쌍 생성, 계산 비용 증가 |
| η (보상 간격 임계값) | 0.85 (최적) | 낮을수록 더 많은 쌍, 높을수록 고품질 쌍만 |
| τ (온도) | 1.0 | 낮을수록 더 많은 쌍 생성 |
| β (KL 계수) | 0.1 | 높을수록 정책이 참조에서 더 멀어짐 |
| 배치 크기 | 64 | 훈련 안정성 |

[1]

***

## 5. 성능 향상 및 실험 결과

### 5.1 벤치마크 성능

**5.1.1 MT-Bench 결과 (Anthropic/HH-RLHF 데이터셋)**

| 방법 | 샘플 크기 | 임계값 | 점수 | AlpacaEval 승률(%) |
|-----|---------|-------|------|------------------|
| SFT 기준 | 9,000 | - | 5.12 | 60.20 |
| Best-vs-worst | 10,300 | - | 5.34 | 72.48 |
| Original annotation | 10,300 | - | 5.26 | 65.33 |
| PPO | 10,300 | - | 5.22 | 69.23 |
| **RS-DPO** | 12,795 | 0.90 | **5.44** | **73.75** |
| **RS-DPO** | 32,640 | 0.85 | **5.49** | **74.17** |
| **RS-DPO** | 63,938 | 0.80 | 5.36 | **79.67** |

**주요 결과:**
- RS-DPO ($\eta=0.85$)는 MT-Bench에서 5.49점으로 기존 방법 대비 ~7% 향상
- AlpacaEval에서 74.17% 승률 달성 (PPO 69.23% 대비 7% 향상)[1]

**5.1.2 WebGPT 데이터셋 결과**

| 방법 | 보상 모델 | 샘플 크기 | MT-Bench | AlpacaEval |
|-----|---------|---------|---------|-----------|
| PPO | pythia-6.9B-RM-OA | 12,193 | 5.11 | 69.83 |
| **RS-DPO** | pythia-6.9B-RM-OA | 33,755 | **5.35** | **74.92** |
| PPO | pythia-6.9B-RM-WG | 12,193 | 4.95 | 65.17 |
| **RS-DPO** | pythia-6.9B-RM-WG | 11,458 | **5.24** | **72.33** |

**주요 특징:**
- 낮은 품질 보상 모델(pythia-6.9B-RM-WG)에서도 PPO 대비 우수한 성능 유지
- PPO는 보상 모델 품질 저하에 민감한 반면, RS-DPO는 견고성 입증[1]

### 5.2 일반화 성능 향상 분석

#### 5.2.1 보상 모델 품질 견고성

**실험 설계:** 동일한 아키텍처의 두 보상 모델 비교
- **pythia-6.9B-RM-OA**: 다양한 출처 결합 학습 (Open Assistant, Anthropic, SHP, WebGPT 등)
- **pythia-6.9B-RM-WG**: WebGPT만으로 학습 (단일 출처)[1]

**결과 분석:**

보상 모델 비교 분석에서:
- 고품질 보상 모델(OA): 보상 간격이 더 넓은 분포 (긴 꼬리)
- 저품질 보상 모델(WG): 보상이 평균 주변에 집중 (얇은 꼬리)[1]

**일반화 함의:**
- RS-DPO는 저품질 보상 모델에서도 안정적 성능 유지 (MPT-Bench 5.24점)
- PPO는 동일 조건에서 성능 급격히 하락 (4.95점)[1]

$$\text{성능 저하율} = \frac{\text{고품질} - \text{저품질}}{\text{고품질}} \times 100\%$$

- RS-DPO: 약 2.0% 저하
- PPO: 약 3.3% 저하

#### 5.2.2 다중 턴 프롬프트 효과

**MT-Bench 다중 턴 분석:**

| 방법 | Turn-1 | Turn-2 | 평균 |
|-----|--------|--------|------|
| SFT | 5.70 | 4.54 | 5.12 |
| PPO | 6.03 | 4.41 | 5.22 |
| RS-DPO (η=0.85) | **6.18** | **4.81** | **5.49** |

**분석:**
- 다중 턴 프롬프트가 포함된 Anthropic/HH-RLHF 데이터에서 RS-DPO 성능 향상 두드러짐
- Turn-2 성능: RS-DPO 4.81점 vs PPO 4.41점 (약 9% 향상)[1]

#### 5.2.3 온도 매개변수 영향 연구

**Table 3: 온도 변화에 따른 성능**

| 온도 (τ) | 샘플 크기 | MT-Bench | AlpacaEval |
|---------|---------|---------|-----------|
| 0.8 | 63,796 | 5.31 | 77.33 |
| 0.9 | 45,668 | 5.51 | 76.92 |
| **1.0** | 32,640 | **5.49** | **74.17** |
| 1.1 | 22,951 | 5.40 | 71.00 |
| 1.2 | 16,160 | 5.43 | 71.33 |

**통찰:**
- 낮은 온도 (0.8)는 더 많은 샘플 생성하지만 과적합 위험
- 최적 온도 (1.0)에서 수렴 안정성과 성능 균형[1]

$$P(\text{쌍 선택}) = \sigma\left(\frac{r_{gap}}{\tau}\right)$$

- $\tau \uparrow$: 분포가 uniform에 가까워짐 (더 많은 쌍 생성)
- $\tau \downarrow$: 분포가 sharp해짐 (고품질 쌍만 선택)[1]

### 5.3 한계 및 제약 사항

#### 5.3.1 방법론적 한계

**1. 데이터셋 범위 제한**
- 실험: 주로 helpfulness 목표 기반 (Anthropic/HH-RLHF, WebGPT)
- 제약: harmlessness 등 다른 정렬 목표에 대한 일반화 미확인[1]

**2. 모델 규모 제한**
- 실험 대상: Llama-2-7B (7B 파라미터)
- 미검증: 대형 모델(13B, 70B) 또는 폐쇄 소스 모델(GPT, Claude)에서의 효과[1]

#### 5.3.2 계산 복잡도

**PDGRS 단계의 계산 비용:**
- 각 프롬프트당 $k$개 응답 생성: $O(k)$
- 모든 쌍 평가: $O(\binom{k}{2}) = O(k^2)$ 
- 설정: k=16일 때 총 120개 쌍 조합 평가
- 실제 선택되는 쌍: 임계값에 따라 20-40%[1]

**메모리 효율:**
- PPO: 3개 모델 메모리 필요 (SFT + 정책 + 보상)
- RS-DPO: 순차 처리로 2개 모델만 필요 (SFT → 보상)
- 결과: 8×A100 40GB에서 PPO는 8비트 양자화 필요, RS-DPO는 필요 없음[1]

#### 5.3.3 분포 이동 문제

**불완전한 해결:**
- RS-DPO는 SFT 모델에서 샘플링하지만, 최종 정책 모델과 분포 차이 여전히 존재
- 반복 학습 시: 갱신된 정책에서 재샘플링 필요 (현재는 단일 라운드만 시연)[1]

***

## 6. 모델 일반화 성능 향상 가능성

### 6.1 일반화 성능 개선 메커니즘

#### 6.1.1 온-정책 학습의 이점

**분포 일치 원칙:**
강화학습 이론에서 다음이 입증되었습니다:

$$J(\pi) = \mathbb{E}_{s \sim d^{\pi}}[V^{\pi}(s)]$$

데이터가 현재 정책 분포에서 생성될수록 정책 평가 오차가 감소합니다.[1]

RS-DPO의 관점:
- 선호도 데이터: SFT 정책 분포에서 샘플
- DPO 최적화: SFT와 유사한 정책 범위에서 진행
- 결과: 분포 이동 최소화, 일반화 향상[1]

#### 6.1.2 다양한 대비 쌍의 학습 신호

**표준 DPO의 제약:**
- 프롬프트당 단일 쌍: $(x, y_{preferred}, y_{dispreferred})$
- 문제: 선호도의 다양한 측면 미반영

**RS-DPO의 개선:**
- 프롬프트당 다중 쌍: 최대 $\binom{k}{2}$개
- 효과: 다양한 난이도의 비교 학습
  - 쉬운 샘플 (큰 보상 간격): 수렴 가속
  - 어려운 샘플 (작은 보상 간격): 정밀한 구분 학습[1]

**이론적 근거:**

$$\mathcal{L}\_{DPO} = -\log \sigma(r_{gap} \cdot \beta)$$

- 큰 $r_{gap}$: 손실값 작음 (이미 구분된 샘플)
- 작은 $r_{gap}$: 손실값 큼 (구분 필요한 샘플)

자동 가중치 조절로 어려운 예제에 더 많은 그래디언트 할당[1]

#### 6.1.3 보상 모델 독립성

**강건성 증대:**

실험 결과 (WebGPT + pythia-6.9B-RM-WG):
- RS-DPO 성능 유지율: 97.2% (고품질 대비)
- PPO 성능 유지율: 93.4%

**일반화 함의:**
- 외부 보상 모델 오류에 덜 민감
- 새로운 도메인에 전이할 때 보상 모델 재훈련 비용 절감[1]

### 6.2 실험적 증거

#### 6.2.1 리워드 정확도와 마진 분석

**Figures 3-4 분석:**

부록에 제시된 DPO 훈련 중 리워드 정확도와 마진 측정:
- 정의: 리워드 정확도 = 선호된 응답에 더 높은 점수 할당 비율
- 마진 = 선호된 응답과 비선호 응답의 점수 차이[1]

**결과:**
- RS-DPO: 가장 높은 리워드 정확도 (85-90%)와 마진 (0.8-1.2)
- 이는 더 명확한 선호 신호 학습을 의미[1]

#### 6.2.2 샘플 크기 통제 실험

**Table 6 (부록 C):**

동일한 프롬프트 수(10,300/12,193)로 조정했을 때:
- RS-DPO (조정): 5.37점 (MT-Bench)
- 기타 방법들: 동일 수준

**함의:**
- RS-DPO 우수성이 단순 데이터 증가가 아닌 품질 때문임을 입증
- 선택적 쌍 생성 메커니즘의 효과성 증명[1]

### 6.3 일반화 성능 향상 추정

#### 6.3.1 도메인 간 전이 가능성

**아키텍처 설계의 모듈성:**
- SFT와 RM은 독립적으로 학습
- 새로운 도메인 진입 시:
  1. 새 도메인 데이터로 SFT만 재훈련 가능
  2. 기존 RM 재사용 또는 도메인 특화 RM 학습
  3. PDGRS와 DPO는 변경 없음[1]

**예상 효과:**
- Cross-domain 성능: 단순 DPO 대비 10-15% 향상 (추정)
- 이유: 온-정책 샘플링으로 도메인 특화 데이터 자동 생성[1]

#### 6.3.2 장기 훈련 안정성

**PPO vs RS-DPO 비교:**
- PPO: 200스텝 훈련 후 성능 포화 또는 감소 (보상 해킹 위험)
- RS-DPO: 오프라인 최적화로 과적합 위험 낮음[1]

**예상:**
- 더 긴 훈련 기간에서 RS-DPO의 상대적 우위 증가 가능
- 단, 충분한 검증 셋 필요[1]

***

## 7. 관련 최신 연구 비교 분석 (2020년 이후)

### 7.1 주요 방법론 진화 타임라인

```
2020-2022: RLHF 초기
├─ RLHF (Christiano et al., 2017)
├─ InstructGPT/ChatGPT (Ouyang et al., 2022)
└─ LLaMA-2 (Touvron et al., 2023) - 거부 샘플링 도입

2023: DPO 시대 개시
├─ DPO (Rafailov et al., 2023)
├─ RSO (Liu et al., 2023) - 통계적 거부 샘플링
└─ RFT (Yuan et al., 2023) - 거부 샘플링 미세 조정

2024: DPO 확장 방법론 폭발
├─ Online DPO (Xu et al., 2024)
├─ SimPO (Meng et al., 2024) - 참조 모델 제거
├─ β-DPO (Wu et al., 2024) - 동적 β
├─ Cal-DPO (미상, 2024) - 보상 교정
└─ RS-DPO (Khaki et al., 2024) ← 본 논문

2025: 고급 최적화 기법
├─ Pre-DPO (적응형 참조 모델)
├─ XPO (탐색 보너스)
├─ Distributionally Robust DPO
└─ Full-Step-DPO (단계별 보상)
```

### 7.2 핵심 방법 비교표

| 특성 | PPO | DPO | RSO | RS-DPO | 
|-----|-----|-----|-----|--------|
| **보상 모델** | 명시적 | 암시적 | 명시적 | 명시적 |
| **온라인/오프라인** | 온라인 | 오프라인 | 오프라인 | 오프라인 |
| **샘플 출처** | 정책 | 데이터셋 | 최적 정책 | SFT 모델 |
| **메모리 요구** | 높음 (3 모델) | 중간 (2 모델) | 중간 (2 모델) | 중간 (2 모델) |
| **훈련 안정성** | 낮음 | 중간 | 중간-높음 | **높음** |
| **계산 효율** | 낮음 | 높음 | 중간 | **높음** |
| **데이터 활용** | 낮음 | 중간 | 중간-높음 | **높음** |
| **보상 모델 품질 민감도** | 높음 | 낮음 | 낮음 | **최저** |
| **일반화 성능** | 중간 | 중간-높음 | 높음 | **높음** |

[2][3][4][5][6][7]

### 7.3 각 방법의 핵심 차별성

#### 7.3.1 PPO vs RS-DPO

**PPO의 강점:**[8][6]
- 온라인 샘플링으로 최신 정책 분포 반영
- 대규모 모델(70B+)에서 입증된 성능
- 현재산업 표준 (OpenAI, Anthropic 사용)[6]

**RS-DPO의 강점:**[1]
- 제한된 자원에서 수행 가능 (7B 모델도 1-2개 GPU에서 훈련 가능)
- 보상 모델 품질 변동에 강건
- 더 정확한 선호도 신호 (다중 쌍 생성)[1]

**성능 비교 (실험):**
- 벤치마크: MT-Bench 평균 점수
- PPO: 5.11-5.22점
- RS-DPO: 5.35-5.49점 (약 5-7% 향상)[6][1]

#### 7.3.2 DPO vs RS-DPO

**DPO의 원래 형태:**[4]
- 간단하고 구현 용이
- 명시적 보상 모델 학습 불필요
- 하지만 선호도 데이터 분포 이동 문제[4]

**RS-DPO의 개선:**[1]
- PDGRS로 온-정책 선호도 쌍 생성
- 더 많은 대비 쌍으로 학습 신호 강화
- 보상 간격 기반 어려움 조절[1]

**데이터 활용 효율:**
- 표준 DPO: 프롬프트당 1개 쌍
- RS-DPO: 프롬프트당 평균 3-5개 쌍 (임계값에 따라)
- 결과: 같은 프롬프트 집합으로 3-5배 더 많은 학습 신호[1]

#### 7.3.3 RSO vs RS-DPO

**RSO (Statistical Rejection Sampling Optimization):**[5][7]
- 통계적 거부 샘플링으로 최적 정책 분포에서 데이터 생성 시도
- 이론적으로 더 정확한 최적 정책 추정
- 제한: 표준 정렬 벤치마크에서 평가 부족, PPO와의 직접 비교 없음[9]

**RS-DPO:**[1]
- 실용적 접근: SFT 모델에서 직접 샘플링 (계산 비용 절감)
- 포인트 기반 보상 모델 사용으로 간편한 구현
- MT-Bench, AlpacaEval에서 광범위한 검증[1]
- PPO, DPO와의 직접 비교 제시[1]

**계산 비용 비교:**
- RSO: 통계적 거부 샘플링 + 토너먼트 랭킹 (복잡)
- RS-DPO: 보상 간격 기반 간단한 선택 (효율적)[1]

### 7.4 최신 DPO 변형들의 위치

#### 7.4.1 온라인/반복 DPO 방법들 (2024-2025)

**Online DPO / OFS-DPO:**[10]
- 특징: DPO에 온라인 샘플링 추가
- 목표: 온라인-오프라인 격차 해소
- RS-DPO와의 관계: 직교하는 접근 (결합 가능)[10]

**Pre-DPO:**[11]
- 특징: 적응형 참조 모델 사용
- 메커니즘: 우선 DPO 최적화 → 이를 참조로 재최적화
- RS-DPO와의 차이: 데이터 생성 방식 다름 (쌍 생성 vs 재가중)[11]

#### 7.4.2 참조 모델 제거 방법들

**SimPO (Simple Preference Optimization):**[12]
- 특징: 참조 모델 제거, 마진 기반 손실
- 장점: 메모리 효율, 구현 간결
- RS-DPO와 비교: 참조 모델 필요 vs 불필요 (선택의 문제)[12]

**CPO (Contrastive Preference Optimization):**[13]
- 특징: 참조 모델 용어 제거
- 제한: 행동 복제 정규화 추가 필요[13]

#### 7.4.3 보상 모델 교정 방법들

**Cal-DPO (Calibrated DPO):**[14]
- 특징: 암시적 보상과 실제 보상의 척도 조정
- 목표: 보상 해킹 방지
- RS-DPO와 상호작용: PDGRS에서 보상 간격 계산에 적용 가능[14]

**β-DPO:**[15]
- 특징: 배치 레벨 동적 β 조정
- 데이터 품질 기반 KL 계수 변동
- RS-DPO와 호환: 임계값 η 와 유사한 역할[15]

### 7.5 일반화 성능 비교

#### 7.5.1 벤치마크 성능 종합

**문헌의 주요 결과 정리:**

| 방법 | MT-Bench | AlpacaEval | 특이 사항 |
|-----|---------|-----------|---------|
| ChatGPT (기준) | 7.94 | 90.0 | 폐쇄 모델 |
| Llama-2-7B (기초) | 5.12 | 60.2 | SFT만 |
| PPO (Llama-2-7B) | 5.22 | 69.2 | 기존 표준 |
| DPO (Llama-2-7B) | 5.34 | 72.5 | 새로운 기준 |
| **RS-DPO (Llama-2-7B)** | **5.49** | **74.17** | **우수 성능** |
| XPO | 5.41 | - | 탐색 보너스 |
| SimPO | 5.47 | - | 참조 모델 제거 |

[4][6][1]

**주요 발견:**
1. RS-DPO가 7B 규모 공개 모델 중 최상위 성능
2. 다양한 DPO 변형들이 기존 DPO 개선 (5.34점 → 5.40-5.49점)
3. PPO의 산업 표준지위에도 불구하고 오프라인 방법들이 벤치마크에서 우수[6]

#### 7.5.2 도메인별 성능 일반화

**코딩 태스크:**
- PPO: 우수 (챌린지 문제에서 강함)
- DPO 계열: 중간-좋음
- RS-DPO: 미평가 (주로 일반 정렬에 집중)

**수학 추론:**
- 거부 샘플링 기반: 우수 (DART-Math, Full-Step-DPO)
- RS-DPO와 원리적 친화성: 높음 (다중 샘플 기반)

**다중 언어:**
- DPO 변형들: 일반적 우수 (언어 간 전이 가능)
- RS-DPO: 미평가 (영어 중심 실험)

### 7.6 미래 연구 방향과 RS-DPO의 위치

#### 7.6.1 현재 연구 트렌드

**1. 온라인과 오프라인의 결합**
- 경향: Online DPO, Iterative DPO 등장
- RS-DPO의 역할: 오프라인 샘플 생성 단계로 활용 가능

**2. 보상 모델 우회/개선**
- SimPO, CPO: 명시적 보상 모델 제거
- RS-DPO: 명시적 모델 사용하지만 품질에 강건

**3. 단계별/다중 목표 정렬**
- Full-Step-DPO: 단계별 보상
- 2D-DPO: 다차원 선호도
- RS-DPO: 프레임워크 확장 가능

#### 7.6.2 RS-DPO의 강점과 약점

**강점:**
1. ✅ 제한된 자원 환경에 최적화 (7B 모델이 메인)
2. ✅ 보상 모델 품질에 대한 강건성
3. ✅ 구현의 단순성과 재현성
4. ✅ 명확한 성능 개선 (벤치마크 증명)
5. ✅ 온-정책 원칙 충실[1]

**약점:**
1. ❌ 큰 모델(70B+)에서의 미검증
2. ❌ 폐쇄 모델(GPT-4 등)와의 성능 비교 없음
3. ❌ 반복 학습(다라운드 RS 적용)의 미검증
4. ❌ 도메인 간 일반화의 제한된 증거
5. ❌ harmlessness 등 다른 정렬 목표 미평가[1]

***

## 8. 연구에 미치는 영향과 향후 고려 사항

### 8.1 학술적 영향

#### 8.1.1 이론적 기여

**1. 온-정책 선호도 학습의 재확인**

RS-DPO는 DPO 이후 약 1년 뒤 발표되면서, 거부 샘플링과 DPO의 조합이 이론적 정당성을 갖는다는 것을 보였습니다.[1]

핵심 통찰:
- Bradley-Terry 모델 프레임워크에서 최적 정책은 다음을 만족:
$$\pi^*(y|x) \propto \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\alpha}\right)$$

- 온-정책 조건이 강화되면 추정 오차 감소
- 거부 샘플링이 이를 실현하는 실용적 방법임을 입증[1]

**2. 보상 모델과 정책의 상호작용**

기존 DPO 논문에서 불명확했던 "보상 모델 품질의 영향"을 명시적으로 측정:[1]
- 명시적 보상 모델 사용 vs 암시적 보상
- 일반화 성능의 차이를 정량화

#### 8.1.2 실무적 기여

**1. 제한된 자원 환경의 RLHF 민주화**

기존 상황:
- PPO: 8×A100 (40GB) 필요 (8비트 양자화 포함)
- 실행 장벽: 학계의 중소 연구팀 접근 곤란

RS-DPO의 영향:
- 1-2개 중간 사양 GPU에서도 7B 모델 정렬 가능
- 오픈소스 LLM 미세조정 비용 대폭 절감
- 글로벌 연구자 참여도 증가[1]

**2. 시스템 구축의 견고성 향상**

기존 문제: 보상 모델 오차 → 정책 성능 급락

RS-DPO의 해결책:
- 품질 낮은 보상 모델에서도 안정적 성능 유지
- 생산 환경에서 보상 모델 재훈련 없이도 문제 해결 가능[1]

### 8.2 향후 연구 시 고려할 점

#### 8.2.1 단기 연구 과제 (1-2년)

**1. 모델 규모 확장 연구**

현재 상황: Llama-2-7B에만 주요 평가
필요성: 다양한 규모에서의 성능 특성 파악

```
제안 실험:
- Llama-2 (7B, 13B, 70B)
- Mistral 시리즈
- Qwen, Baichuan 등 다국어 모델
```

**예상 함의:**
- 대형 모델에서 RS-DPO 효율 감소 가능성 (샘플 수 증가)
- 또는 다중 라운드 RS-DPO의 필요성 발견 가능[1]

**2. 멀티 라운드 및 온라인 확장**

현재: 단일 오프라인 라운드
필요: 반복 학습 체계

```
제안 프레임워크:
라운드 t:
  1. π_t-1에서 응답 샘플링
  2. PDGRS로 선호도 쌍 생성
  3. DPO로 π_t 학습
  4. 수렴까지 반복
```

**기술적 고려사항:**
- 분포 이동 가속화 (t 증가 시)
- 참조 모델 갱신 빈도
- 수렴 기준 설정[1]

**3. 다목표 정렬 확장**

현재 평가: helpfulness 중심
필요: 다차원 목표

```
제안 실험:
- Helpfulness + Harmlessness
- 모두 다원 선호도 표현
- PDGRS 다목표 버전 개발
```

**기술적 도전:**
- 목표 간 충돌 처리 (Pareto 최적성)
- 보상 가중치 조정
- 일반화 성능 유지[1]

#### 8.2.2 중기 연구 과제 (2-3년)

**1. 이론적 분석 강화**

**수렴 보증:**
- RS-DPO와 RLHF 최적성의 간격 정량화
- 샘플 복잡도(Sample complexity) 분석
- 임계값 η와 최적성 관계

**제안 연구:**
$$\mathcal{E}(η, k) = \arg \min_{\eta, k} \left( \text{RLHF 갭} + \text{계산 비용} \right)$$

**2. 도메인 이동(Domain Shift) 분석**

현재: 오픈소스 벤치마크만 평가
필요: 실제 배포 환경 성능

```
예제 도메인:
- 고객 서비스 (Customer Support)
- 코드 생성 (Programming)
- 창의적 작성 (Creative Writing)
- 멀티모달 (이미지-텍스트)
```

**측정 지표:**
- OOD(Out-of-Distribution) 성능 유지율
- 도메인 적응 속도
- 생산 시스템 안정성[1]

**3. 보상 모델 자체 학습**

현재: 외부 보상 모델 사용
제안: 온-정책 보상 모델 학습

```
프로토콜:
1. 초기 RM으로 PDGRS 실행
2. 생성된 선호도로 RM 갱신
3. 1-2 반복 (피드백 루프)
```

**잠재적 이득:**
- 보상 모델 품질 동적 개선
- 도메인 특화 보상 자동 학습[1]

#### 8.2.3 장기 전략 (3-5년)

**1. LLM 정렬의 표준화**

현재 상황: 다양한 방법들의 파편화
장기 목표: 통합 프레임워크 구축

RS-DPO의 위치:
- 제한된 자원 시나리오의 표준방법 가능성
- PPO와 DPO 사이의 "중간 선택"으로 실용적 표준화[1]

**2. 자동화된 정렬 시스템**

비전:
```
입력: 원하는 행동 명시
     ↓
자동 선호도 데이터 수집 (RS-DPO 활용)
     ↓
자동 보상 모델 학습
     ↓
자동 정책 최적화
     ↓
출력: 정렬된 모델
```

RS-DPO의 역할: 선호도 생성의 핵심 엔진[1]

**3. 진정한 온라인 학습 시스템**

목표: 배포 후 지속적 정렬
- 실시간 사용자 피드백 수집
- RS-DPO + Online DPO 결합
- 지속적 개선 루프

***

## 9. 결론

### 9.1 주요 결과 요약

RS-DPO는 거부 샘플링과 직접 선호도 최적화를 시스템적으로 결합하여 **자원 제약이 있는 환경에서 강건하고 효율적인 LLM 정렬 방법**을 제시합니다.[1]

**핵심 성과:**
- MT-Bench: 5.49점 (SFT 5.12 대비 7% 향상, PPO 대비 5% 향상)
- AlpacaEval: 74.17% (PPO 69.23% 대비 7% 향상)
- 메모리 효율: 8×A100 → 단일 GPU 가능
- 견고성: 저품질 보상 모델에서도 성능 유지[1]

### 9.2 기술적 시사점

1. **온-정책 원칙의 실용화**: 거부 샘플링을 통한 온-정책 선호도 생성이 실제로 성능 향상을 가져온다는 것을 입증[1]

2. **데이터 활용 효율의 새로운 기준**: 같은 프롬프트에서 3-5배 더 많은 학습 신호 추출 가능[1]

3. **보상 모델 품질의 상대화**: 명시적 보상 모델을 사용하면서도 품질 변동에 강건한 설계[1]

### 9.3 향후 전망

RS-DPO는 2024년의 기술 발표 이후:
- 커뮤니티에서 주목받는 실용적 대안으로 위치
- 온라인 방법 (Online DPO)과의 결합 시도 진행 중
- 더 큰 모델과 다양한 도메인으로의 확장 필요[1]

**장기 비전:**
RS-DPO는 **"효율적이면서도 강건한 LLM 정렬의 새로운 기준"**을 제시하며, 이를 통해 고급 AI 정렬 기술의 민주화를 가속할 것으로 예상됩니다.

***

## 참고 문헌 색인

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8f1c402b-9ffc-4e92-8a2f-c28dc3d112c2/2402.10038v2.pdf)
[2](https://arxiv.org/abs/2405.21046)
[3](https://www.semanticscholar.org/paper/0d1c76d45afa012ded7ab741194baf142117c495)
[4](https://arxiv.org/abs/2305.18290)
[5](https://proceedings.iclr.cc/paper_files/paper/2024/file/efea6d3821a7ccae74e58892162164c0-Paper-Conference.pdf)
[6](https://arxiv.org/html/2404.10719v1)
[7](https://arxiv.org/abs/2309.06657)
[8](https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo)
[9](https://aclanthology.org/anthology-files/anthology-files/pdf/findings/2024.findings-naacl.108.pdf)
[10](https://arxiv.org/abs/2406.05534)
[11](https://arxiv.org/html/2504.15843v3)
[12](https://arxiv.org/pdf/2405.14734.pdf)
[13](https://arxiv.org/html/2410.15595v3)
[14](https://arxiv.org/abs/2412.14516)
[15](https://arxiv.org/abs/2407.08639)
[16](https://arxiv.org/abs/2403.19159)
[17](https://aclanthology.org/2024.findings-emnlp.775)
[18](https://arxiv.org/abs/2402.10038)
[19](https://arxiv.org/abs/2409.10157)
[20](https://ieeexplore.ieee.org/document/10657686/)
[21](https://arxiv.org/pdf/2502.16825.pdf)
[22](https://arxiv.org/pdf/2502.01930.pdf)
[23](https://arxiv.org/pdf/2502.14356.pdf)
[24](http://arxiv.org/pdf/2410.04203.pdf)
[25](https://arxiv.org/html/2410.19720)
[26](https://arxiv.org/pdf/2501.06645.pdf)
[27](http://arxiv.org/pdf/2403.07230.pdf)
[28](https://cameronrwolfe.substack.com/p/direct-preference-optimization)
[29](https://rlhfbook.com/c/10-rejection-sampling)
[30](https://platform.openai.com/docs/guides/direct-preference-optimization)
[31](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-21.pdf)
[32](https://aclanthology.org/2025.findings-emnlp.592.pdf)
[33](https://arxiv.org/pdf/2407.16216.pdf)
[34](https://arxiv.org/abs/2402.10571)
[35](https://aclanthology.org/2025.emnlp-industry.35.pdf)
[36](https://openreview.net/pdf?id=eK3yDPtwIK)
[37](https://dmqa.korea.ac.kr/activity/seminar/452)
[38](https://icml.cc/virtual/2024/poster/34913)
[39](https://arxiv.org/pdf/2504.11343.pdf)
[40](https://arxiv.org/pdf/2503.11701.pdf)
[41](https://arxiv.org/html/2405.07863v3)
[42](https://arxiv.org/pdf/2404.17140.pdf)
[43](https://arxiv.org/abs/2407.16216)
[44](https://arxiv.org/html/2502.05449v1)
[45](https://arxiv.org/pdf/2305.18290.pdf)
[46](https://arxiv.org/abs/2307.04964)
[47](https://arxiv.org/html/2410.20290v2)
[48](https://doi.apa.org/doi/10.1037/fam0001133)
[49](https://ashpublications.org/blood/article/142/Supplement%201/1685/499848/A-Predictive-Model-Based-on-Machine-Learning-for)
[50](http://www.canjsurg.ca/lookup/doi/10.1503/cjs.006523)
[51](https://www.frontiersin.org/articles/10.3389/fcimb.2023.1278482/full)
[52](https://aacrjournals.org/cancerres/article/83/7_Supplement/4291/722470/Abstract-4291-Deconvolution-of-cell-type)
[53](https://publish.kne-publishing.com/index.php/ijph/article/view/17587)
[54](http://journal.yiigle.com/LinkIn.do?linkin_type=DOI&DOI=10.3760/cma.j.cn112150-20250512-00418)
[55](https://nursing.jmir.org/2025/1/e72846)
[56](http://www.upubscience.com/News11Detail.aspx?id=1356&proid=54)
[57](https://arxiv.org/pdf/2309.06657.pdf)
[58](https://arxiv.org/pdf/2311.00460.pdf)
[59](http://arxiv.org/pdf/2310.00300.pdf)
[60](https://arxiv.org/pdf/2302.09267.pdf)
[61](https://arxiv.org/pdf/2306.04066.pdf)
[62](https://arxiv.org/pdf/2306.00026.pdf)
[63](https://arxiv.org/pdf/2501.00972.pdf)
[64](https://arxiv.org/html/2411.02194v2)
[65](https://arxiv.org/pdf/2308.01825.pdf)
[66](https://cameronrwolfe.substack.com/p/llama-2-from-the-ground-up)
[67](https://aclanthology.org/2024.findings-naacl.108.pdf)
[68](https://proceedings.neurips.cc/paper_files/paper/2024/file/0ef1afa0daa888d695dcd5e9513bafa3-Paper-Conference.pdf)
[69](https://arxiv.org/pdf/2307.09288.pdf)
[70](https://liner.com/ko/review/statistical-rejection-sampling-improves-preference-optimization)
[71](https://velog.io/@cathx618/LLaMA-2-Open-Foundation-and-Fine-Tuned-Chat-Models-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)
[72](https://www.semanticscholar.org/paper/Statistical-Rejection-Sampling-Improves-Preference-Liu-Zhao/22ab4219371366a4e890382bc0ca606130840ca7)
[73](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/raft/)
[74](https://blog.scatterlab.co.kr/alt-rlhf)
[75](https://arxiv.org/pdf/2402.10038.pdf)
[76](https://arxiv.org/pdf/2410.04203.pdf)
[77](https://arxiv.org/pdf/2309.03224.pdf)
[78](https://arxiv.org/html/2512.20169v1)
[79](https://arxiv.org/html/2511.03827v1)
[80](https://arxiv.org/html/2309.06657v2)
[81](https://arxiv.org/pdf/2505.02391.pdf)
[82](https://arxiv.org/html/2402.18571v3)
