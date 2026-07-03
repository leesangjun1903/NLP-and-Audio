# Think-Then-Generate: Reasoning-Aware Text-to-Image Diffusion with LLM Encoders

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존 T2I(Text-to-Image) 확산 모델들은 LLM을 단순한 **텍스트 인코더(text encoder)**로만 활용하며, LLM이 본래 가진 **추론 능력(reasoning capability)**을 전혀 활용하지 못하고 있다. 이에 본 논문은 **Think-Then-Generate(T2G)** 패러다임을 제안하여 LLM 인코더가 원시 사용자 프롬프트를 **추론하고 재작성(reason & rewrite)**한 후, 재작성된 프롬프트의 임베딩을 확산 조건으로 활용하는 방식을 제안한다.

### 주요 기여 (4가지)

| 기여 항목 | 내용 |
|-----------|------|
| **T2G 패러다임 제안** | LLM을 수동적 인코더 → 능동적 추론 에이전트로 전환 |
| **SFT 데이터 구축** | CoT 추론 + 재작성 프롬프트로 구성된 7,000개 샘플 |
| **Dual-GRPO 설계** | LLM 인코더와 DiT를 동시에 최적화하는 복합 정책 |
| **SOTA 성능 달성** | WISE 0.79 (GPT-4o 수준), T2I-ReasonBench 92.2 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제 1: 텍스트-픽셀 매핑의 한계 (Text-Pixel Mapper Problem)

기존 모델들은 다음과 같은 구조적 한계를 갖는다:

- CLIP, T5 등의 인코더 → 의미론적 추론 불가
- LLM 기반 인코더도 **동결(frozen)** 상태로만 사용 (Qwen-Image, FLUX 등)
- 결과: 추상적·개념적 프롬프트 처리 실패

**예시**: "Holiday celebrating the birth of Jesus Christ" → 바닐라 모델은 예수를 문자 그대로 그리지만, T2G 모델은 크리스마스 축하 장면을 생성

#### 문제 2: 추론-생성 간의 미정렬 (Reasoning-Generation Misalignment)

단순 CoT 추가는 WISE 점수를 0.61 → 0.65로 소폭 향상시키는 데 그침. 이는 CoT 과정이 DiT 디코더와 **연결되지 않은 채(decoupled)** 작동하기 때문이다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: Reasoning-Aware Behavior Activation (SFT)

Gemini-2.5를 이용해 원시 프롬프트에 대한 CoT 추론과 정제 프롬프트를 생성하여 SFT 데이터셋 구축:

$$
\text{raw prompt} \rightarrow [\text{long CoT}] \rightarrow \text{refined prompt}
$$

SFT 후 t-SNE 시각화를 통해 임베딩 분포가 보존됨을 확인 → DiT와의 호환성 유지

#### Step 2: Dual-GRPO

**복합 정책 모델 정의** ($\theta = \{\phi, \lambda\}$):

$$
\pi_\theta(o_t \mid s_t) = \begin{cases} p_\phi(z_t \mid z_{ < t}, q) & t \leq \ell \\ p_\lambda(x_t \mid x_{t-1}, \hat{z}) & t > \ell \end{cases} 
$$

여기서 $\phi$는 LLM 인코더 파라미터, $\lambda$는 DiT 파라미터, $\ell$은 추론 토큰 수, $\hat{z}$는 정제된 프롬프트의 임베딩이다.

**GRPO의 이점 추정 (Advantage Estimation)**:

$$
\hat{A}_g = \frac{R_g - \text{mean}\left(\{R_g\}_{g=1}^G\right)}{\text{std}\left(\{R_g\}_{g=1}^G\right)} 
$$

**전체 최적화 목적 함수**:

$$
\max_{\theta=\{\phi,\lambda\}} \mathbb{E}_{q \sim p(Q), \{z_1,\cdots,x_{\ell+m}\} \sim \pi_\theta} \frac{1}{\ell+m}\left[\sum_{t=1}^{\ell} R_1(z_t, z_{<t}, q) + \sum_{t=\ell+1}^{\ell+m} R_2(x_t, x_{t-1}, \hat{z})\right] 
$$

**Dual-GRPO 목적 함수 (J개 추론 × K개 이미지 계층적 샘플링)**:

$$
\max_{\theta=\{\phi,\lambda\}} \mathbb{E}_{q \sim p(Q), \{z_1,\cdots,x_{\ell+m}\}_{i=1}^{J \times K} \sim \pi_{\theta_{old}}} \frac{1}{l+m}\left[\sum_{t=1}^{l} \mathcal{L}_t(\phi) + \sum_{t=l+1}^{l+m} \mathcal{L}_t(\lambda)\right] 
$$

**LLM 인코더 손실 $\mathcal{L}_t(\phi)$** (그룹 크기 $J$):

$$
\frac{1}{J}\sum_{j=1}^{J}\left[\min\left(r_{j,t}(\phi), \text{clip}(r_{j,t}(\phi), 1-\epsilon, 1+\epsilon)\right)\hat{A}_{j,t}\right] - \beta \mathbb{D}_{KL}\left[p_\phi(z_{j,t}) \| p_{\phi_{ref}}(z_{j,t})\right] 
$$

여기서 $r_{j,t}(\phi) = \frac{p_\phi(z_{j,t}|z_{j,<t},q)}{p_{\phi_{old}}(z_{j,t}|z_{j,<t},q)}$

**DiT 손실 $\mathcal{L}_t(\lambda)$** (Flow-GRPO 기반, 그룹 크기 $K$):

$$
\frac{1}{J}\sum_{j=1}^{J}\frac{1}{K}\sum_{k=1}^{K}\left[\min\left(r_{j,k,t}(\lambda), \text{clip}(r_{j,k,t}(\lambda), 1-\epsilon, 1+\epsilon)\right)\hat{A}_{j,k,t}\right] - \beta \mathbb{D}_{KL}\left[p_\lambda(x_{j,k,t}|x_{j,k,t-1},\hat{z}_j) \| p_{\lambda_{ref}}(x_{j,k,t}|x_{j,k,t-1},\hat{z}_j)\right] 
$$

#### Step 3: 단계별 보상 함수 설계

**LLM 추론 단계 보상 $R_1$** (의미론적 일관성 중심):

$$
R_1(z_{j,t}, z_{j, < t}, q) = \beta_1(\tau) \frac{1}{K}\sum_{k=1}^{K} R_{sem}(\mathbf{x}_{j,k}, q) 
$$

**LLM 단계 이점 계산**:

$$
\hat{A}_{j,t} = \frac{R_1(z_{j,t}, z_{j,<t}, q) - \text{mean}\left(\{R_1(z_{j,t}, z_{j,<t}, q)\}_{j=1}^J\right)}{\text{std}\left(\{R_1(z_{j,t}, z_{j,<t}, q)\}_{j=1}^J\right)} 
$$

**DiT 샘플링 단계 보상 $R_2$** (미적 + 물리적 일관성 + 의미론적 일관성):

$$
R_2(x_{j,k,t}, x_{j,k,t-1}, \hat{z}_j) = \beta_2(\tau)\left(\omega_1 R_{aes}(\mathbf{x}_{j,k}) + \omega_2 R_{con}(\mathbf{x}_{j,k}) + \omega_3 R_{sem}(\mathbf{x}_{j,k})\right) 
$$

**Flow-GRPO의 전이 커널** (결정론적 ODE → SDE 변환):

$$
\pi_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t), g_t^2 \Delta t \mathbf{I}) 
$$

$$
\mu_\theta(\mathbf{x}_t) = \mathbf{x}_t + \left[v_\theta(\mathbf{x}_t, t) + \frac{g_t^2}{2t}\left(\mathbf{x}_t + (1-t)v_\theta(\mathbf{x}_t, t)\right)\right]\Delta t 
$$

---

### 2.3 모델 구조

```
[Raw User Prompt q]
        ↓
[LLM Encoder (Qwen2.5-VL)]
   ├── CoT Reasoning (z₁,...,z_ℓ)  ← SFT + GRPO 학습
   └── Refined Prompt Embedding (ẑ)
        ↓
[DiT Backbone (MM-DiT)]           ← Flow-GRPO 학습
   └── Image Generation (x_{ℓ+1},...,x_{ℓ+m})
        ↓
[Image-grounded Rewards]
   ├── R_sem (의미론적 일관성)
   ├── R_aes (미적 품질)
   └── R_con (물리적 일관성)
        ↓
[Dual-GRPO 업데이트 → 양쪽 동시 최적화]
```

**주요 구성 요소**:

| 컴포넌트 | 세부 사항 |
|---------|-----------|
| LLM 인코더 | Qwen2.5-VL-7B |
| DiT 백본 | MM-DiT (Stable Diffusion 3 기반) |
| SFT 데이터 | 7,000개 (T2I), UniREdit-100K (편집) |
| Dual-GRPO | J=5 추론 × K=16 이미지 계층적 샘플링 |
| 보상 스케줄러 | $\beta_1(\tau) = \beta_2(\tau) = 0.5$ (균형 스케줄러) |

---

### 2.4 성능 향상

#### T2I 생성 (WISE 벤치마크)

| 모델 | Cultural | Time | Space | Biology | Physics | Chemistry | **Overall** |
|------|----------|------|-------|---------|---------|-----------|-------------|
| Qwen-Image | 0.62 | 0.63 | 0.78 | 0.55 | 0.67 | 0.35 | 0.61 |
| Ours (w/o SFT+GRPO) | 0.68 | 0.58 | 0.77 | 0.62 | 0.76 | 0.41 | 0.65 |
| Ours (w/o GRPO) | 0.76 | 0.66 | 0.79 | 0.74 | 0.84 | 0.65 | 0.74 |
| **Ours** | **0.80** | **0.74** | **0.83** | **0.81** | **0.85** | **0.66** | **0.79** |
| GPT-4o | 0.81 | 0.71 | 0.89 | 0.83 | 0.79 | 0.74 | 0.80 |

→ 기존 Qwen-Image 대비 **+30%** 향상, GPT-4o와 동등 수준

#### T2I-ReasonBench

| 모델 | Overall Acc. | Overall Qual. |
|------|-------------|---------------|
| Qwen-Image | 57.8 | 87.5 |
| Gemini-2.0 | 64.8 | 88.7 |
| **Ours** | **68.3** | **92.2** |

#### 이미지 편집 (UniREditBench / RISEBench)

| 모델 | UniREdit ↑ | RISE ↑ |
|------|-----------|--------|
| Qwen-Image-Edit | 56.5 | 8.9 |
| Ours (w/o GRPO) | 61.1 | 20.2 |
| **Ours** | **68.7** | **23.9** |
| Gemini-2.5-Flash-Image | 68.3 | 32.8 |

---

### 2.5 한계점

1. **상업 모델과의 격차**: 특히 Chemistry 도메인(0.66 vs 0.74)과 RISEBench에서 Gemini-2.5-Flash-Image(32.8)에 여전히 뒤처짐
2. **계산 비용**: 계층적 샘플링(J×K)으로 인한 훈련 비용 증가
3. **SFT 데이터 의존성**: Gemini-2.5로 생성된 합성 데이터 의존 → 교사 모델의 오류 전파 가능성
4. **단일 베이스 모델**: Qwen-Image에만 적용됨 → 범용성 검증 필요
5. **보상 함수 설계**: $R_{sem}$, $R_{aes}$, $R_{con}$ 가중치 $\omega_1, \omega_2, \omega_3$의 최적화 기준 불명확

---

## 3. 일반화 성능 향상 가능성

### 3.1 T2G 패러다임의 일반화 메커니즘

T2G가 일반화 성능을 향상시키는 핵심 메커니즘은 **LLM의 세계 지식(world knowledge) 활성화**이다:

$$
\underbrace{q}_{\text{raw prompt}} \xrightarrow{\text{LLM reasoning}} \underbrace{\hat{z}}_{\text{enriched representation}} \xrightarrow{\text{DiT}} \underbrace{x}_{\text{image}}
$$

이 과정에서 LLM은 훈련 데이터에 없는 개념도 **귀납적 추론**으로 처리 가능:

- **예시**: "Einstein's favorite instrument" → 추론 과정에서 "바이올린" 도출 → DiT에 구체적 프롬프트 전달

### 3.2 일반화 성능 향상 근거

#### (1) 임베딩 공간 보존 (t-SNE 분석)

SFT 이후에도 임베딩 분포가 원본과 완전히 겹침(overlap) → DiT와의 호환성 유지 → **새로운 도메인에서도 안정적 시각 출력**

#### (2) 다중 도메인 성능 (WISE 25개 하위 도메인)

| 도메인 유형 | 성능 향상 (Qwen-Image → Ours) |
|------------|------------------------------|
| 문화 상식 | 0.62 → 0.80 (+29%) |
| 시공간 이해 | 0.78 → 0.83 (+6%) |
| 생물학 | 0.55 → 0.81 (+47%) |
| 화학 | 0.35 → 0.66 (+89%) |

특히 **과학적 추론** 도메인에서 급격한 향상 → 일반화 범위 확대 확인

#### (3) 이미지 편집으로의 전이

T2G 패러다임이 단순 생성에서 **이미지 편집 태스크**로도 성공적으로 전이됨:
- Qwen-Image-Edit: 8.9 → Ours: 23.9 (RISE, +168%)
- 이는 reasoning 메커니즘이 도메인 독립적으로 작동함을 시사

#### (4) Dual-GRPO의 정규화 효과

KL 발산 정규화 항:

$$
-\beta \mathbb{D}_{KL}\left[\pi_\theta \| \pi_{\theta_{ref}}\right]
$$

이 항이 과도한 정책 변화를 억제 → **과적합 방지** → 새로운 프롬프트 유형에 대한 일반화 유지

#### (5) 계층적 보상 설계의 역할

```
Stage 1 (LLM): R_sem → 의미론적 다양성 보존
Stage 2 (DiT): ω₁R_aes + ω₂R_con + ω₃R_sem → 다각도 최적화
```

다양한 보상 신호의 조합이 단일 목표로의 과적합을 방지

### 3.3 일반화의 잠재적 한계

| 한계 | 설명 |
|------|------|
| **저빈도 지식** | LLM의 사전 학습 데이터에 없는 개념 처리 어려움 |
| **다국어 일반화** | 주로 영어 프롬프트 기반 → 한국어 등 타 언어 성능 미검증 |
| **분포 외 시각적 스타일** | 매우 특수한 아트 스타일에서 일반화 미확인 |

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 T2I 확산 모델 발전 계보

```
DDPM (Ho et al., 2020)
     ↓
Stable Diffusion (Rombach et al., 2022) - LDM 도입
     ↓
FLUX.1 / SD-3.5 (2023-2024) - DiT + Flow Matching
     ↓
Qwen-Image / LLM 인코더 기반 (2024-2025)
     ↓
Think-Then-Generate (2026) - 추론 인식 T2I
```

### 4.2 주요 관련 연구 비교표

| 연구 | 방법 | 텍스트 인코더 | 추론 | RL 최적화 | 한계 |
|------|------|--------------|------|-----------|------|
| **DDPO** (Black et al., 2023) | 역과정 이산화 | CLIP (동결) | ✗ | Policy Gradient | 복잡한 프롬프트 불안정 |
| **Flow-GRPO** (Liu et al., 2025) | ODE→SDE 변환 | T5 (동결) | ✗ | GRPO (DiT만) | 인코더 동결 |
| **DanceGRPO** (Xue et al., 2025) | 그룹 상대 정책 | 동결 | ✗ | GRPO | 인코더 동결 |
| **T2I-R1** (Jiang et al., 2025) | 양방향 CoT | 통합 멀티모달 | △ | RL | 확산 백본 없음 |
| **Uni-CoT** (Qin et al., 2025) | 매크로+마이크로 CoT | 통합 | ○ | SFT | RL 미적용 |
| **BAGEL** (Deng et al., 2025) | 인터리브드 사전학습 | 통합 | △ | - | 리터럴 생성 편향 |
| **T2G (Ours)** | Think-Then-Generate | LLM (학습) | **○** | **Dual-GRPO** | 계산 비용 |

### 4.3 RL for Diffusion 관점에서의 비교

기존 방법들의 공통 한계:

$$
\underbrace{\text{인코더 동결}}_{\text{Flow-GRPO, DanceGRPO}} \Rightarrow \text{지식 활용 불가}
$$

T2G의 혁신:

$$
\underbrace{\text{LLM 인코더 학습}}_{\phi \text{ 최적화}} + \underbrace{\text{DiT 학습}}_{\lambda \text{ 최적화}} = \text{Dual-GRPO}
$$

### 4.4 멀티모달 통합 모델과의 비교

| 측면 | 통합 모델 (BAGEL, HunyuanImage) | T2G |
|------|--------------------------------|-----|
| 아키텍처 | 단일 자기회귀 | LLM 인코더 + DiT 분리 |
| 추론 활성화 | 사전 학습 의존 | SFT + RL 명시적 활성화 |
| 계산 효율 | 대규모 파라미터 필요 | 경량 SFT 가능 |
| WISE 점수 | 0.70 (BAGEL), 0.58 (HunyuanImage) | **0.79** |

---

## 5. 미래 연구에 미치는 영향과 고려사항

### 5.1 미래 연구에 미치는 영향

#### (1) 인코더 중심 패러다임 전환

기존의 "인코더 동결 + 디코더 최적화" 패러다임에서 벗어나 **인코더 자체를 학습 가능한 추론 에이전트**로 전환하는 방향을 제시. 이는 향후 멀티모달 시스템 설계의 새로운 기준이 될 수 있다.

#### (2) 복합 정책 RL의 일반화

Dual-GRPO의 복합 정책 설계:

$$
\pi_\theta = \underbrace{p_\phi}_{\text{LLM}} \circ \underbrace{p_\lambda}_{\text{DiT}}
$$

이 설계는 **비디오 생성**, **3D 생성**, **음성-이미지 멀티모달** 등 다양한 복합 생성 시스템에 적용 가능하다.

#### (3) 지식 집약적 시각 생성 연구 촉진

WISE, T2I-ReasonBench 등 **추론 기반 벤치마크의 중요성**을 부각시켜, 단순 시각적 품질을 넘어 **의미론적 정확성**을 평가하는 연구 흐름 강화 예상.

#### (4) 통합 모델(Unified Model)로의 로드맵

T2G의 결과는 "추론 + 표현 + 시연" 능력을 갖춘 차세대 통합 모델의 가능성을 보여줌 → AGI 방향 연구에 중요한 중간 단계.

### 5.2 앞으로 연구 시 고려할 점

#### (1) 계산 효율성 문제

$$
\text{총 샘플 수} = J \times K = 5 \times 16 = 80 \text{ (롤아웃당)}
$$

이는 기존 DiT 훈련 대비 훨씬 높은 비용 → **샘플 효율적 RL** 연구 필요:
- 적응적 샘플링 전략
- 오프-폴리시(off-policy) 접근법 탐색

#### (2) 보상 해킹(Reward Hacking) 위험

$R_{sem}$, $R_{aes}$, $R_{con}$이 모두 **프록시 보상(proxy reward)**이므로, 실제 인간 의도와 벗어난 방향으로 최적화될 가능성:
- **인간 피드백(RLHF)** 통합 고려
- 더 강건한 보상 함수 설계 필요

#### (3) 다국어·다문화 일반화

현재 영어 중심 데이터셋 → **다국어 CoT 추론** 능력 검증 및 확장 필요:

$$
q_{\text{한국어}} \rightarrow \hat{z}_{\text{한국어}} \rightarrow x
$$

#### (4) CoT 추론의 신뢰성

SFT 데이터가 Gemini-2.5의 합성 데이터에 의존 → **환각(hallucination) 전파** 가능성:
- 검증된 지식 소스와의 연동
- 사실 검증(Fact Verification) 모듈 통합

#### (5) 추론 길이 최적화

추론 예산 $\ell$이 성능에 미치는 영향 미검증:

$$
\ell \uparrow \Rightarrow \text{더 정교한 추론} \quad \text{vs} \quad \ell \downarrow \Rightarrow \text{빠른 추론}
$$

최적 $\ell$ 값을 동적으로 결정하는 **적응적 추론 길이** 연구 필요.

#### (6) 멀티모달 입력 확장

현재 텍스트 프롬프트 중심 → **이미지+텍스트 복합 입력**에서의 추론 능력 강화:
- Qwen2.5-VL의 VLM 능력을 더 적극적으로 활용
- 참조 이미지 기반 추론 강화

#### (7) 비디오·3D 생성으로의 확장

T2G 패러다임의 시간적(temporal) 일관성 유지 문제:

$$
\pi_\theta^{\text{video}} = p_\phi \circ p_\lambda^{\text{video}} \quad \Rightarrow \text{시간적 추론 통합}
$$

---

## 참고 자료

**본 논문 (주요 출처)**:
- Siqi Kou, Jiachun Jin, Zetong Zhou, et al. "Think-Then-Generate: Reasoning-Aware Text-to-Image Diffusion with LLM Encoders." *arXiv:2601.10332v1*, 2026.

**논문 내 인용 참고자료**:
- Bai et al. "Qwen2.5-VL Technical Report." *arXiv:2502.13923*, 2025.
- Liu et al. "Flow-GRPO: Training Flow Matching Models via Online RL." *arXiv:2505.05470*, 2025.
- Black et al. "Training Diffusion Models with Reinforcement Learning (DDPO)." *arXiv:2305.13301*, 2023.
- Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3)." *ICML 2024*.
- Guo et al. "DeepSeek-R1: Incentivizes Reasoning in LLMs through Reinforcement Learning." *Nature*, 2025.
- Shao et al. "DeepSeekMath (GRPO 원류)." *arXiv:2402.03300*, 2024.
- Niu et al. "WISE: A World Knowledge-Informed Semantic Evaluation." *arXiv:2503.07265*, 2025.
- Sun et al. "T2I-ReasonBench." *arXiv:2508.17472*, 2025.
- Jiang et al. "T2I-R1." *arXiv:2505.00703*, 2025.
- Qin et al. "Uni-CoT." *arXiv:2508.05606*, 2025.
- Deng et al. "BAGEL." *arXiv:2505.14683*, 2025.
- Xue et al. "DanceGRPO." *arXiv:2505.07818*, 2025.
- Schulman et al. "PPO." *arXiv:1707.06347*, 2017.
- Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.
- Zhao et al. "RISEBench." *arXiv:2504.02826*, 2025.

> ⚠️ **정확도 안내**: 본 답변은 제공된 논문 PDF(arXiv:2601.10332v1)를 직접 분석하여 작성되었습니다. 논문에 명시되지 않은 내용(특히 2020년 이후 비교 연구의 세부 수치 중 일부)은 논문 내 인용 정보를 기반으로 재구성하였으며, 억측이나 창작 없이 논문 원문에 충실하게 작성하였습니다.
