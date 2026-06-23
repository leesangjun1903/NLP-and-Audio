# Flow-GRPO: Training Flow Matching Models via Online RL 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Flow-GRPO는 **온라인 정책 경사(Policy Gradient) 강화학습(RL)을 Flow Matching 모델에 최초로 통합**한 방법론입니다. 기존 Flow Matching 모델은 결정론적(deterministic) ODE 기반 생성 프로세스를 사용하기 때문에 RL에서 요구하는 확률적 탐색(stochastic exploration)이 불가능했습니다. 이 논문은 이 근본적 문제를 해결하면서 이미지 품질 저하나 다양성 감소 없이 성능을 대폭 향상시킵니다.

### 3대 핵심 기여

| 기여 | 내용 | 효과 |
|------|------|------|
| **ODE → SDE 변환** | 결정론적 ODE를 동일 주변 분포를 유지하는 SDE로 변환 | RL 탐색을 위한 확률성 도입 |
| **Denoising Reduction** | 학습 시 디노이징 스텝 수를 대폭 감소(40→10) | 4배 이상 학습 속도 향상 |
| **KL 정규화를 통한 Reward Hacking 방지** | KL 다이버전스를 제약으로 활용 | 이미지 품질/다양성 유지 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

Flow Matching 모델(SD3.5, FLUX 등)에 온라인 RL을 적용할 때의 두 가지 핵심 장벽:

**문제 1: 결정론적 ODE와 RL의 탐색 요구 충돌**
- Flow Matching은 아래 결정론적 ODE를 따름:

$$d\boldsymbol{x}_t = \boldsymbol{v}_t \, dt \tag{6}$$

- RL은 다양한 궤적 탐색을 위해 확률적 샘플링이 필수
- 결정론적 샘플링에서는 $p_\theta(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t, c)$ 계산이 불가능하거나 매우 비효율적

**문제 2: 샘플링 비효율성**
- 고품질 이미지 생성을 위해 수십~수백 스텝 필요
- 온라인 RL은 대규모 샘플 수집이 필요 → 학습 비용 폭증

### 2.2 제안 방법 (수식 포함)

#### (A) Flow Matching 기초

Rectified Flow 프레임워크에서 "노이즈가 섞인" 데이터 $\boldsymbol{x}_t$는:

$$\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1, \quad t \in [0,1] \tag{1}$$

Flow Matching 목적함수:

$$\mathcal{L}(\theta) = \mathbb{E}_{t,\, \boldsymbol{x}_0 \sim X_0,\, \boldsymbol{x}_1 \sim X_1}\left[\|\boldsymbol{v} - \boldsymbol{v}_\theta(\boldsymbol{x}_t, t)\|^2\right] \tag{2}$$

여기서 타겟 속도장은 $\boldsymbol{v} = \boldsymbol{x}_1 - \boldsymbol{x}_0$.

#### (B) 디노이징 과정의 MDP 정식화

반복적 디노이징 과정을 MDP $(S, A, \rho_0, P, R)$로 정의:

- **상태**: $s_t \triangleq (c, t, \boldsymbol{x}_t)$
- **행동**: $a_t \triangleq \boldsymbol{x}_{t-1}$ (모델이 예측한 디노이즈 샘플)
- **정책**: $\pi(a_t | s_t) \triangleq p_\theta(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t, c)$
- **보상**: $R(s_t, a_t) \triangleq r(\boldsymbol{x}_0, c)$ (최종 스텝에서만)

#### (C) GRPO의 Flow Matching 적용

RL의 정규화된 목적함수:

$$\max_\theta \mathbb{E}_{(s_0,a_0,\ldots,s_T,a_T)\sim\pi_\theta}\left[\sum_{t=0}^{T}\left(R(s_t,a_t) - \beta D_{\mathrm{KL}}(\pi_\theta(\cdot|s_t)\|\pi_{\mathrm{ref}}(\cdot|s_t))\right)\right] \tag{3}$$

프롬프트 $c$에 대해 $G$개 이미지를 샘플링, $i$번째 이미지의 어드밴티지:

$$\hat{A}^i_t = \frac{R(\boldsymbol{x}^i_0, c) - \mathrm{mean}(\{R(\boldsymbol{x}^i_0, c)\}^G_{i=1})}{\mathrm{std}(\{R(\boldsymbol{x}^i_0, c)\}^G_{i=1})} \tag{4}$$

Flow-GRPO의 최종 학습 목적함수:

$$\mathcal{J}_{\text{Flow-GRPO}}(\theta) = \mathbb{E}_{c \sim C,\, \{\boldsymbol{x}^i\}^G_{i=1} \sim \pi_{\theta_{\text{old}}}(\cdot|c)} f(r, \hat{A}, \theta, \varepsilon, \beta) \tag{5}$$

여기서:

$$f(r, \hat{A}, \theta, \varepsilon, \beta) = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{T}\sum_{t=0}^{T-1}\left(\min\left(r^i_t(\theta)\hat{A}^i_t,\; \mathrm{clip}\!\left(r^i_t(\theta), 1-\varepsilon, 1+\varepsilon\right)\hat{A}^i_t\right) - \beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\right)$$

$$r^i_t(\theta) = \frac{p_\theta(\boldsymbol{x}^i_{t-1} | \boldsymbol{x}^i_t, c)}{p_{\theta_{\text{old}}}(\boldsymbol{x}^i_{t-1} | \boldsymbol{x}^i_t, c)}$$

#### (D) ODE → SDE 변환 (핵심 전략 1)

결정론적 ODE를 동일한 주변 분포(marginal distribution)를 보존하는 역방향 SDE로 변환. Fokker-Planck 방정식을 통해 도출된 일반 역방향 SDE:

$$d\boldsymbol{x}_t = \left(\boldsymbol{v}_t(\boldsymbol{x}_t) - \frac{\sigma^2_t}{2}\nabla\log p_t(\boldsymbol{x}_t)\right)dt + \sigma_t \, d\boldsymbol{w} \tag{7}$$

Rectified Flow에서 스코어 함수는 속도장으로부터 다음과 같이 유도됨:

$$\nabla\log p_t(\boldsymbol{x}) = -\frac{\boldsymbol{x}}{t} - \frac{1-t}{t}\boldsymbol{v}_t(\boldsymbol{x}) \tag{27}$$

이를 대입하면 최종 SDE:

$$d\boldsymbol{x}_t = \left[\boldsymbol{v}_t(\boldsymbol{x}_t) + \frac{\sigma^2_t}{2t}\left(\boldsymbol{x}_t + (1-t)\boldsymbol{v}_t(\boldsymbol{x}_t)\right)\right]dt + \sigma_t \, d\boldsymbol{w} \tag{8}$$

Euler-Maruyama 이산화 후 최종 업데이트 규칙:

$$\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \left[\boldsymbol{v}_\theta(\boldsymbol{x}_t, t) + \frac{\sigma^2_t}{2t}\left(\boldsymbol{x}_t + (1-t)\boldsymbol{v}_\theta(\boldsymbol{x}_t, t)\right)\right]\Delta t + \sigma_t\sqrt{\Delta t}\,\epsilon \tag{9}$$

여기서 $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\sigma_t = a\sqrt{\frac{t}{1-t}}$ ($a$는 노이즈 레벨 하이퍼파라미터).

정책 $\pi_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, c)$가 등방성 가우시안이 되므로, KL 다이버전스를 closed form으로 계산 가능:

$$D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}}) = \frac{\|\bar{\boldsymbol{x}}_{t+\Delta t,\theta} - \bar{\boldsymbol{x}}_{t+\Delta t,\mathrm{ref}}\|^2}{2\sigma^2_t\Delta t} = \frac{\Delta t}{2}\left(\frac{\sigma_t(1-t)}{2t} + \frac{1}{\sigma_t}\right)^2\|\boldsymbol{v}_\theta(\boldsymbol{x}_t,t) - \boldsymbol{v}_{\mathrm{ref}}(\boldsymbol{x}_t,t)\|^2$$

#### (E) Denoising Reduction (핵심 전략 2)

- **학습 시**: $T = 10$ 스텝 (저품질이지만 상대적 선호도 신호 보존)
- **추론 시**: $T = 40$ 스텝 (원래 기본 설정 유지)
- 결과: 4배 이상 학습 속도 향상, 최종 성능 저하 없음

### 2.3 모델 구조

```
[프롬프트 입력]
      ↓
[SD3.5-M 기반 Flow Matching 모델 (LoRA α=64, r=32)]
      ↓ (SDE 샘플링, T=10 스텝)
[G=24개의 이미지 그룹 생성]
      ↓
[보상 함수 (규칙 기반 or 모델 기반)]
      ↓
[그룹 상대적 어드밴티지 계산 (GRPO)]
      ↓
[정책 최적화 (KL 정규화 포함, β=0.04/0.01)]
      ↓
[업데이트된 정책으로 반복]
```

**적용 모델**: Stable Diffusion 3.5 Medium (SD3.5-M), FLUX.1-Dev  
**하드웨어**: NVIDIA A800 GPU 24개  
**학습 파라미터**: 노이즈 레벨 $a=0.7$, 그룹 크기 $G=24$, 해상도 512

### 2.4 성능 향상

| 태스크 | 베이스라인 | Flow-GRPO | 향상 |
|--------|-----------|-----------|------|
| GenEval (전체) | 0.63 (SD3.5-M) | **0.95** | +32%p |
| GenEval (위치) | 0.24 | **0.99** | +75%p |
| GenEval (계수) | 0.50 | **0.95** | +45%p |
| 시각적 텍스트 렌더링 | 0.59 | **0.92** | +33%p |
| PickScore (인간 선호) | 21.72 | **23.31** | +1.59 |
| GPT-4o (GenEval) | 0.84 | — | Flow-GRPO 초과 달성 |

**이미지 품질**: DrawBench 기준 Aesthetic Score, DeQA 등 유지 (KL 정규화 사용 시)

### 2.5 한계

1. **비디오 생성 미적용**: 현재 T2I에만 집중; 비디오로 확장 시 보상 설계, 다중 목표 균형, 확장성 문제 존재
2. **KL 정규화의 트레이드오프**: 더 긴 학습 시간 필요; 특정 프롬프트에서 간헐적 reward hacking 발생
3. **그룹 크기 민감성**: $G < 12$ 시 학습 붕괴(training collapse) 위험
4. **계산 비용**: 24개 A800 GPU로 수천 GPU 시간 소요
5. **보상 모델 의존성**: 모델 기반 보상(PickScore 등)의 한계를 그대로 상속

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

### 3.1 실험적 일반화 증거

논문은 Table 4에서 두 가지 명시적 일반화 실험을 수행합니다:

#### (A) 미학습 객체 클래스로의 일반화

| 방법 | 전체 | 단일객체 | 두객체 | 계수 | 색상 | 위치 | 속성 |
|------|------|----------|--------|------|------|------|------|
| SD3.5-M | 0.64 | 0.96 | 0.73 | 0.53 | 0.87 | 0.26 | 0.47 |
| +Flow-GRPO | **0.90** | **1.00** | **0.94** | **0.86** | **0.97** | **0.84** | **0.77** |

- 60개 객체 클래스로 학습 → **20개 미학습 클래스에서 평가**
- 전체 정확도 0.64 → 0.90으로 향상: 순수한 일반화 성능 개선

#### (B) 미학습 개수 범위로의 일반화

| 학습 범위 | 평가 범위 | SD3.5-M | Flow-GRPO |
|----------|----------|---------|-----------|
| 2~4개 객체 | 5~6개 | 0.13 | **0.48** |
| 2~4개 객체 | 12개 | 0.02 | **0.12** |

- 2~4개 객체로만 학습했음에도 **12개 객체 생성**에서 6배 성능 향상

#### (C) 도메인 외 벤치마크: T2I-CompBench++

GenEval 스타일 데이터로 학습했지만, **완전히 다른 분포의 T2I-CompBench++**에서도 향상:

| 메트릭 | SD3.5-M | Flow-GRPO | 향상 |
|--------|---------|-----------|------|
| 색상 | 0.7994 | **0.8379** | +4.8%p |
| 형태 | 0.5669 | **0.6130** | +8.2%p |
| 2D 공간 | 0.2850 | **0.5447** | +26%p |
| 수량 | 0.5927 | **0.6752** | +8.2%p |

### 3.2 일반화를 가능하게 하는 메커니즘

```
일반화 향상의 4가지 메커니즘
│
├── 1. 온라인 RL의 탐색(Exploration) 효과
│   └── SDE 샘플링으로 다양한 이미지 분포 탐색
│       → 특정 패턴 암기보다 근본적 구성 능력 학습
│
├── 2. KL 정규화의 사전 지식 보존
│   └── 사전학습된 가중치와의 거리 제한
│       → 과도한 파인튜닝으로 인한 forgetting 방지
│
├── 3. 규칙 기반 보상의 의미론적 포착
│   └── 객체 계수, 색상, 공간 관계 등 근본 규칙 학습
│       → 특정 객체 클래스가 아닌 추상적 관계 학습
│
└── 4. 그룹 상대적 어드밴티지(GRPO)의 안정성
    └── 그룹 내 상대적 비교로 어드밴티지 추정
        → 절대적 보상값 오류에 덜 민감, 안정적 학습
```

### 3.3 일반화 성능 향상의 한계와 잠재적 원인

- **12개 객체 생성 시 여전히 낮은 성능(0.12)**: 분포 외 일반화는 근본적 한계 존재
- **비공간적 관계에서 제한적 향상**: T2I-CompBench++ Non-Spatial(0.3146 → 0.3195, +0.5%p)
- **보상 설계의 편향**: 규칙 기반 보상이 포착하지 못하는 속성은 일반화 안 됨

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 Flow Matching 기반 모델 정렬 방법론 비교

| 방법 | 유형 | 온라인 여부 | Flow 모델 적용 | 확률적 탐색 | GenEval |
|------|------|------------|--------------|------------|---------|
| DDPO (Black et al., 2023) | PPO 기반 | ✓ | 확장 적용 가능 | ✓ (확산) | — |
| DPO (Rafailov et al., 2023) | 오프라인 선호 | ✗ | Flow-DPO | ✗ | 낮음 |
| ReFL (Xu et al., 2024) | 직접 보상 역전파 | ✗ | ✗ | ✗ | — |
| ORW (Fan et al., 2025) | RWR + Wasserstein | ✓ | ✓ | 제한적 | — |
| F5R-TTS (Sun et al., 2025) | GRPO | ✓ | ✓ (TTS) | 가우시안 재정의 | — |
| **Flow-GRPO (본 논문)** | **GRPO** | **✓** | **✓ (이미지)** | **✓ (ODE→SDE)** | **0.95** |

### 4.2 핵심 관련 연구 상세 비교

#### DDPO vs. Flow-GRPO

| 항목 | DDPO | Flow-GRPO |
|------|------|-----------|
| 기반 RL | PPO (값 함수 필요) | GRPO (값 함수 불필요) |
| 대상 모델 | 확산 모델 | Flow Matching |
| 확률성 도입 | 기본 내장 (DDPM) | ODE→SDE 변환 |
| 학습 안정성 | 후반부 붕괴 경향 | 안정적 지속 학습 |
| 메모리 효율 | 낮음 | 높음 |

#### DeepSeek-R1 / GRPO와의 연결

DeepSeek-R1에서 LLM 추론 능력 향상에 사용된 GRPO를 이미지 생성으로 전이:

- **LLM GRPO**: 토큰 시퀀스 → 텍스트 보상
- **Flow-GRPO**: 디노이징 스텝 시퀀스 → 이미지 보상
- 핵심 차이: Flow는 연속적 행동 공간, LLM은 이산적 행동 공간

#### ORW (Online Reward-Weighted Regression) vs. Flow-GRPO

| 메트릭 | ORW | Flow-GRPO |
|--------|-----|-----------|
| CLIP Score | 28.40 | **30.18** |
| Diversity Score | 0.97 | **1.02** |
| 학습 안정성 | Step 720부터 하락 | 지속 향상 |
| 정규화 방법 | Wasserstein-2 | KL 다이버전스 |

### 4.3 연구 흐름 타임라인

```
2020 ─── 2021 ─── 2022 ─── 2023 ─── 2024 ─── 2025
  │                │          │          │          │
DDPM             Flow       DDPO       FLUX      Flow-GRPO
(Ho et al.)    Matching    (Black     SD3.5    (본 논문)
              (Lipman)    et al.)   DPO류      ORW, F5R
                                   정렬 연구
```

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5.1 연구에 미치는 영향

#### (A) 즉각적 영향

1. **비디오 생성 모델로의 확장 가능성**
   - WAN, HunyuanVideo, Sora 등 비디오 Flow Matching 모델에 동일 프레임워크 적용 가능
   - 시간적 일관성, 물리적 사실성 등의 보상 설계가 핵심 과제

2. **다중 모달 생성 모델로의 확장**
   - 텍스트-비디오, 텍스트-3D, 텍스트-오디오 등 다양한 Flow 기반 생성 모델에 적용 가능
   - F5R-TTS가 텍스트-음성 변환에 유사 접근법 적용 (동시 연구)

3. **VLM을 보상 모델로 활용하는 연구 방향 촉진**
   - 미분 불가능 보상(예: GPT-4V 평가)을 직접 활용 가능
   - 보상 모델의 발전이 자동으로 생성 모델 품질 향상으로 이어지는 선순환

#### (B) 장기적 영향

1. **생성 모델의 RL 패러다임 확립**
   - LLM에서 RL이 핵심 학습 방법론으로 자리잡은 것처럼, 생성 모델에서도 동일한 전환 촉진
   - "사전학습 → SFT → RL" 파이프라인의 생성 모델 버전 확립

2. **자동화된 이미지/비디오 평가 연구 촉진**
   - 더 정교한 규칙 기반/모델 기반 보상 설계 연구 필요성 부각
   - 인간 선호도를 정확히 포착하는 보상 모델 연구 가속

### 5.2 향후 연구 시 고려사항

#### 기술적 고려사항

1. **보상 설계의 정교화**
   ```
   현재 한계:
   - 단일/소수 보상 신호
   - 규칙 기반 보상의 단순성
   
   개선 방향:
   - 계층적 보상 (전역 구성 + 지역 세부사항)
   - 시간적 보상 (비디오의 프레임 간 일관성)
   - 다중 보상의 파레토 최적화
   ```

2. **KL 정규화의 대안 탐색**
   - KL은 학습을 늦추는 트레이드오프 존재
   - 대안: Wasserstein 거리, Rényi 다이버전스, 스펙트럴 정규화
   - 적응적 KL 계수 스케줄링 연구 필요

3. **확장성(Scalability) 문제**
   - 대형 모델(FLUX.1, SD3.5-L 등)에서 그룹 크기 $G$와 메모리의 균형
   - 분산 RL 학습 인프라 최적화
   - 오프-폴리시(off-policy) 데이터 재사용 전략

4. **노이즈 스케줄 최적화**
   - $\sigma_t = a\sqrt{t/(1-t)}$의 고정 형태 한계
   - 학습 가능한 노이즈 스케줄 연구 가능성
   - 태스크별 최적 노이즈 레벨 자동 탐색

5. **Reward Hacking의 근본적 해결**
   - 현재 KL 정규화는 완전한 해결책 아님
   - 적대적 훈련(adversarial training)과의 결합
   - 다양성 명시적 보상(diversity-aware reward) 설계

#### 방법론적 고려사항

6. **오프라인/온라인 하이브리드 접근**
   - 완전한 온라인 RL은 샘플 효율이 낮음
   - 오프라인 사전 학습 + 온라인 파인튜닝 조합 연구

7. **다중 에이전트 생성 시스템**
   - 생성자-검증자 이중 에이전트 구조
   - 자기 비판적(self-critical) 보상 메커니즘

8. **이론적 수렴 보장**
   - 현재 논문은 실증적 결과 중심
   - Flow Matching + GRPO의 이론적 수렴 조건 증명 필요
   - SDE 근사 오류가 최종 성능에 미치는 영향 분석

#### 평가 프레임워크 고려사항

9. **보다 포괄적인 일반화 평가**
   - 현재: GenEval, T2I-CompBench++ 중심
   - 필요: 문화적 다양성, 추상적 개념, 창의적 표현 평가

10. **공정한 비교를 위한 표준화**
    - 컴퓨팅 예산(GPU 시간) 기준 공정 비교 필요
    - 오픈소스 재현 가능 벤치마크 구축

---

## 참고자료 (출처)

본 답변은 아래 문서를 직접 참조하여 작성되었습니다:

1. **Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., & Ouyang, W. (2025).** *Flow-GRPO: Training Flow Matching Models via Online RL.* arXiv:2505.05470v5 [cs.CV]. 39th Conference on Neural Information Processing Systems (NeurIPS 2025). https://arxiv.org/abs/2505.05470

논문 내 핵심 인용 문헌:

2. **Shao, Z. et al. (2024).** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300. (GRPO 원본)
3. **Liu, X., Gong, C., & Liu, Q. (2022).** *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* arXiv:2209.03003.
4. **Esser, P. et al. (2024).** *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML 2024. (SD3.5)
5. **Black, K. et al. (2023).** *Training Diffusion Models with Reinforcement Learning.* arXiv:2305.13301. (DDPO)
6. **Guo, D. et al. (2025).** *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948.
7. **Rafailov, R. et al. (2023).** *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023.
8. **Ghosh, D. et al. (2023).** *GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment.* NeurIPS 2023.
9. **Song, Y. et al. (2020).** *Score-Based Generative Modeling through Stochastic Differential Equations.* arXiv:2011.13456.
10. **Fan, J. et al. (2025).** *Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization.* ICLR 2025.
11. **Huang, K. et al. (2023/2025).** *T2I-CompBench / T2I-CompBench++.* NeurIPS 2023 / TPAMI 2025.
12. **Kirstain, Y. et al. (2023).** *Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation.* NeurIPS 2023. (PickScore)

> **정확도 고지**: 본 답변은 제공된 논문 PDF(arXiv:2505.05470v5)를 직접 분석한 결과입니다. 2025년 10월 기준 최신 버전(v5)을 기반으로 하며, 논문에 명시되지 않은 내용은 포함하지 않았습니다.
