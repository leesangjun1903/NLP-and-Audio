# KTO: Model Alignment as Prospect Theoretic Optimization

## 1. 핵심 주장과 주요 기여 (요약)

**핵심 주장**

- RLHF·DPO 등 기존 LLM alignment 목표는 사실상 “인간의 인지 편향(손실회피 등)”을 이미 암묵적으로 모델링하는 **인간-인식 손실(Human-Aware Loss, HALO)** 이며, 이것이 단순 cross-entropy 대비 성능 향상의 주된 이유라는 관점을 제시한다.[1]
- 그러나 기존 방법이 가정하는 인간 효용 함수는 행동경제학의 **Prospect Theory(카너먼–트버스키 이론)** 에서 쓰이는 효용과는 다르다.  
- 이에 따라, **카너먼–트버스키 효용을 직접 최적화하는 HALO인 KTO(Kahneman–Tversky Optimization)** 를 제안하고,  
  - **쌍대 선호(pairwise preference)** 가 아니라, 각 (x, y)에 대한 **이진 피드백(좋음/나쁨)** 만으로 학습하면서도  
  - 1B–30B 규모에서 **DPO 수준 또는 그 이상의 정렬 성능**을 달성함을 보인다.[1]

**주요 기여**

1. **HALO라는 개념 정립**  
   - DPO, SLiC, PPO-Clip 등을 공통된 형식의 **prospect-theoretic value function**으로 해석하고, 이들을 HALO로 분류한다.[1]
2. **Prospect Theory 기반 새로운 손실(KTO) 제안**  
   - KL-제약 RLHF에서 유도된 “암묵적 reward”와 **Kahneman–Tversky 효용**을 결합한 loss를 설계한다.[1]
3. **바이너리 피드백 기반 alignment**  
   - “좋다/나쁘다”만 있는 데이터로 DPO 수준을 달성할 수 있음을 대규모 실험으로 입증한다.[1]
4. **데이터 불균형·비선호 데이터에 대한 견고성**  
   - 긍정 예시를 90% 제거해도 DPO와 비슷하거나 더 나은 성능을 유지하는 등, 극단적 불균형 상황에서의 안정성을 보인다.[1]
5. **이론 분석**  
   - “선호 확률 최대화”가 “인간 효용 최대화”와 generally 같지 않음을 보이고, noisy·비추이(non-transitive) 선호에 대해 **KTO가 DPO보다 더 나은 최악 사례 보장**을 가질 수 있음을 증명한다.[1]

***

## 2. 논문이 해결하고자 하는 문제

### 2.1 문제 정의

1. **데이터 형태의 한계**  
   - RLHF·DPO는 입력 x에 대해 (y_w, y_l) 쌍을 요구(“A가 B보다 좋다”).  
   - 실제 서비스(ChatGPT, Claude 등)에서는 대부분 **thumbs-up / thumbs-down** 형태의 이진 피드백이 수집되며, 선호쌍은 비싸고 희소하다.

2. **목표 함수의 해석 부족**  
   - 왜 DPO·PPO가 SFT보다 훨씬 잘 되는지, “무엇을” 최적화하고 있는지에 대한 이론적 해석이 부족하다.  
   - 특히, 사람의 의사결정을 설명하는 **Prospect Theory**와의 관계가 정교하게 분석된 적이 없다.

3. **일반화(특히 open-ended task)와 노이즈**  
   - 현재의 선호학습은 noisy하고 상충되는 사람들 간의 선호를 어떻게 처리하는지, 그리고 이것이 **일반화·견고성**에 어떤 영향을 주는지에 대한 이해가 부족하다.

논문은 위 문제들을 동시에 겨냥해,  
- “**어떤 손실이 인간 효용에 더 맞는가?**”  
- “**그 손실이 실제로 alignment·일반화에서 더 낫나?**”  
를 Prospect Theory 관점에서 다룬다.[1]

***

## 3. 제안 방법: HALO와 KTO (수식 포함)

### 3.1 기존 RLHF / DPO 복습

KL-제약 RLHF 최적 정책은 다음과 같다.[1]

$`
\pi^{*}(y \mid x)
= \frac{1}{Z(x)}
\pi_{\text{ref}}(y \mid x)
\exp\!\left(\frac{1}{\beta} r^{*}(x,y)\right)
`$

따라서 최적 reward는

$`
r^{*}(x,y)
= \beta \log \frac{\pi^{*}(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
\tag{1}
`$

이를 Bradley–Terry 모형에 넣으면, DPO 손실은

```math
\mathcal{L}_{\text{DPO}}(\pi_\theta,\pi_{\text{ref}})
=
\mathbb{E}_{(x,y_w,y_l)}
\left[
  -\log \sigma\!\left(
    \beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
    - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
  \right)
\right]
```

여기서 $\sigma$는 시그모이드 함수이다.

### 3.2 Prospect Theory 기반 효용 함수

Kahneman–Tversky의 **value function**은 (금전적 결과 z 기준)[1]

```math
v(z, z_{\text{ref}};\lambda,\alpha)
=
\begin{cases}
(z - z_{\text{ref}})^{\alpha} & z > z_{\text{ref}} \\
- \lambda (z_{\text{ref}} - z)^{\alpha} & z < z_{\text{ref}}
\end{cases}
```

- $\alpha \approx 0.88$ : 한계효용 체감 (concave gain, convex loss)  
- $\lambda \approx 2.25$ : 손실회피(loss aversion)

논문은 LLM에 이 복잡한 지수를 그대로 쓰지 않고, **sigmoid를 value function 대체물**로 사용한다.  

### 3.3 HALO (Human-Aware Loss) 정의

HALO는 직관적으로 “보상–참조점의 차이에 concave value function을 씌운 뒤, 음의 affine 변환을 취한 loss”로 정의된다.[1]

정의 3.4에 따르면, 손실 $f$가 HALO이려면

```math
f(x,y;\theta)
=
t\Big(
  v_f\big(r_\theta(x,y)
         - \mathbb{E}_{x',y'}[r_\theta(x',y')]\big)
\Big)
```

- $r_\theta$ : 더 선호되는 (x,y)에 더 큰 값을 주는 reward  
- $v_f$ : (0,∞)에서 concave인 value function  
- $t$ : 음의 기울기를 가진 affine 함수  

논문은 DPO, SLiC(calibration term), PPO-Clip이 이 정의를 만족하는 HALO임을 보인다(Proposition 3.5).[1]

### 3.4 KTO: Prospect-Theoretic Optimization

핵심 아이디어는:  
- RLHF 최적 정책에서 유도되는 **암묵적 reward**  
  $r_{\text{KTO}}(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$  
- 전역 참조점으로서의 **KL 기대값**  
- 그리고 **로지스틱 value function** $\sigma(\cdot)$  
을 결합해, **바이너리 레이블(l = desirable/undesirable)** 을 입력으로 하는 효용을 직접 최적화하는 것이다.[1]

#### 3.4.1 암묵적 reward와 참조점

$$
\begin{aligned}
r_{\text{KTO}}(x,y)
&= \beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \\
z_{\text{ref}}
&= \mathbb{E}_{x' \sim D}
\left[
  \beta \,\mathrm{KL}\!\big(
    \pi_\theta(\cdot\mid x')
    \,\|\, \pi_{\text{ref}}(\cdot\mid x')
  \big)
\right]
\end{aligned}
$$

실제 구현에서는, batch 내 무관한 $(x', y'_U)$들의 log-ratio 평균에 $\max(0,\cdot)$을 취해 근사하고, **역전파는 하지 않는다**.[1]

#### 3.4.2 KTO value function

각 (x,y)가 “바람직한(desirable)” 혹은 “바람직하지 않은(undesirable)” 것으로 라벨링된다고 할 때,

```math
v_{\text{KTO}}(x,y;\beta)
=
\begin{cases}
\sigma\big(r_{\text{KTO}}(x,y) - z_{\text{ref}}\big),
& y \sim y_{\text{desirable}} \mid x
\\[2mm]
\sigma\big(z_{\text{ref}} - r_{\text{KTO}}(x,y)\big),
& y \sim y_{\text{undesirable}} \mid x
\end{cases}
```

즉  
- 좋은 예시는 $r_{\text{KTO}}$가 $z_{\text{ref}}$보다 클수록 utility 증가  
- 나쁜 예시는 $r_{\text{KTO}}$가 $z_{\text{ref}}$보다 작을수록 utility 증가(=나쁘게 만들어야 하므로)

#### 3.4.3 KTO 손실

```math
\mathcal{L}_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}})
=
\mathbb{E}_{(x,y)\sim D}
\big[
  w(y)\,\big(1 - v_{\text{KTO}}(x,y;\beta)\big)
\big]
```

여기서 가중치 $w(y)$는

$$
w(y) =
\begin{cases}
\lambda_D, & y \sim y_{\text{desirable}} \mid x \\
\lambda_U, & y \sim y_{\text{undesirable}} \mid x
\end{cases}
$$

데이터 불균형 $n_D, n_U$에 대해

$$
\frac{\lambda_D n_D}{\lambda_U n_U}
\in \Big[1, \frac{4}{3}\Big]
$$

을 만족하도록 조정함으로써 손실 기여를 균형 있게 맞춘다.[1]

#### 3.4.4 직관

- 좋은 예시의 reward를 “막연히” 키우면 KL도 같이 커져서 $r_{\text{KTO}} - z_{\text{ref}}$가 크게 변하지 않으므로 loss가 거의 줄지 않는다.  
- 따라서 **“정말 좋은 부분”만 선택적으로 키우는 방향**으로 학습이 진행되도록 강제한다.  
- 나쁜 예시에 대해서도 반대 방향의 논리가 작용하며, KL의 비음수성 때문에 손실 포화(saturation) 속도가 비대칭이 된다.[1]

***

## 4. 모델 구조 및 학습 파이프라인

### 4.1 모델 구조

- KTO 자체는 **모델 구조를 바꾸지 않는다.**  
- 베이스 LLM은 Pythia(1.4–12B), LLaMA(7–30B), Mistral-7B 등 표준 decoder-only Transformer로,  
  - pretrain: next-token LM  
  - SFT: instruction-response 데이터  
  - alignment: KTO 또는 DPO / PPO / SLiC 등  
의 **objective만 교체**한다.[1]

따라서  
- **추론 시 구조·복잡도는 SFT LLM과 동일**,  
- 추가 모듈(별도 reward model, value function 등) 없이, **reference 모델과 현재 policy만 필요**하다.

### 4.2 학습 단계

1. **Pretraining**: 대규모 코퍼스에서 LM training  
2. **SFT (선택적)**: instruction-following 데이터로 cross-entropy finetune  
3. **KTO Alignment**:
   - 입력 x, 출력 y, 라벨 $l \in \{\text{desirable}, \text{undesirable}\}$  
   - $r_{\text{KTO}}(x,y)$, $z_{\text{ref}}$ 계산  
   - $v_{\text{KTO}}$, $\mathcal{L}_{\text{KTO}}$로 gradient update

흥미로운 실험 결과는, **충분히 강한 pretrain 모델일 경우 SFT 단계를 완전히 생략하고 바로 KTO를 적용해도 SFT+KTO와 비슷한 품질**이 나온다는 점이다.[1]

***

## 5. 성능 향상과 한계

### 5.1 DPO·기타 HALO 대비 성능

#### 5.1.1 HALO vs Non-HALO

같은 데이터·세팅에서 CSFT, SLiC(정규화 포함), DPO, 오프라인 PPO(±1 dummy reward)를 비교한 결과:[1]

- **HALO(DPO, PPO-Clip 변형)** 가  
  - Non-HALO(CSFT, full SLiC)보다 항상 같거나 더 좋고,  
  - LLaMA 13B 이상에서는 통계적으로 유의미한 성능 차이를 보인다.  
- 즉, “인간 편향을 반영한 value function을 가진 손실(HALO)”가 alignment 품질에 실제로 도움을 준다는 경험적 근거를 제공한다.

#### 5.1.2 KTO ≥ DPO (1B–30B)

동일한 preference 데이터(Anthropic HH, SHP, OpenAssistant 조합)에서 Pythia·LLaMA 패밀리 기준:[1]

- **SFT + KTO ≈ SFT + DPO** : 거의 모든 스케일에서 비슷한 GPT-4 winrate  
- **KTO 단독 vs DPO 단독**:  
  - LLaMA 7B, 13B, 30B에서 **KTO 단독이 DPO 단독보다 유의하게 우수**(p < 0.01).  
- 특히 LLaMA 13B·30B에서는 **SFT 없이도 KTO만으로 SFT+KTO와 비슷한 성능**을 달성. DPO는 SFT를 건너뛰면 성능이 크게 악화된다.

#### 5.1.3 Preference 데이터에 묶이지 않는 KTO

- LLaMA-7B에 대해, desirable 예시의 90%를 제거(1:1 → 1:10 불균형)해도  
  - $\lambda_D, \lambda_U$ 조정만으로 **여전히 DPO보다 높은 winrate**를 유지.[1]  
- OpenAssistant 데이터로 Mistral-7B를 정렬할 때,  
  - DPO: n쌍 사용  
  - KTO(all y per x): 2n 예시  
  - KTO(one y per x): x당 한 개 y만 사용(데이터량 72% 감소)  
  - 결과: **KTO(one y per x)조차 DPO와 공식 Instruct 모델보다 높은 winrate**를 기록.[1]

이는 **KTO가 “원래부터 쌍으로 짝지어져 있지 않은 이진 피드백 데이터”에서도 잘 작동할 수 있다**는 것을 의미한다.

#### 5.1.4 다운스트림 벤치마크 (일반화 측면)

Zephyr-β(Mistral-7B 기반)를 UltraFeedback에서 DPO 대신 KTO로 정렬하면:[1]

- MMLU, GSM8K, HumanEval, BBH 등에서 **전반적으로 DPO보다 우수**  
- 특히 GSM8K(수학적 추론)에서는  
  - DPO 대비 **+13.5 포인트** 향상  
  - one-y-per-x KTO도 DPO보다 큰 폭 우위(+10포인트)

즉, alignment 단계에서의 KTO 교체가 **일반 언어·코드·수학·reasoning 벤치마크 전반의 일반화 성능을 크게 향상**시킬 수 있음을 보인다.

### 5.2 한계와 비판

#### 5.2.1 “어려운” 예시를 무시할 위험 (Underfitting)

Proposition 4.1:[1]

- reward가 너무 높은 나쁜 예시, 너무 낮은 좋은 예시는  
  - 시그모이드 value의 포화 영역에 들어가 gradient가 거의 0이 되며,  
  - KTO 업데이트는 이를 사실상 **무시**하게 된다.  

이는  
- noisy·모순적인 label을 무시하는 면에서는 **견고성**을 주지만,  
- 정말로 배우기 어려우나 필수적인 예시까지 무시한다면 **underfitting**을 유발할 수 있다.

#### 5.2.2 효용과 선호 likelihood의 괴리

Theorem 4.2:[1]

- $r_b(x,y) = r_a(x,y) + h(x)$ 형태로 reward에 입력 의존 상수를 더한 두 reward는  
  - 같은 최적 정책과 같은 Bradley–Terry 선호 분포를 생성하지만,  
  - **인간 효용 분포는 달라질 수 있음**을 보인다.  

→ 즉, **선호 likelihood 최대화(=DPO)** 가 **인간 효용 최대화**와 동일하지 않으며,  
KTO처럼 효용을 직접 최적화하는 접근이 open-ended 평가에서 더 낫게 나올 수 있다는 이론적 근거를 제공한다.[1]

#### 5.2.3 모순 선호에 대한 worst-case 성능

Theorem 4.3:[1]

- 두 annotator a, b가 같은 x에 대해 상충되는 선호  
  - $y_1 \succ_a y_2$, $y_2 \succ_b y_1$ 을 가진다면,  
- 최악의 경우 DPO는  
  - 두 사람 모두의 기대 효용을 **동시에 낮추는** 정책으로 수렴할 수 있음.  
- 반면, 동일 데이터를 “바이너리 예시 두 개”로 쪼개 KTO를 적용하면  
  - default 설정에서 **정책을 거의 바꾸지 않고** noisy한 부분을 회피하는 경향.

이는 여러 annotator가 섞인 실제 preference 데이터(예: SHP, OASST, UltraFeedback)에서  
**KTO가 DPO보다 사람 전체에 대한 평균 효용을 더 잘 유지할 수 있음**을 시사한다.[1]

#### 5.2.4 후속 연구에서 제기된 비판 (특히 일반화 관점)

1. **BCO: Binary Classifier Optimization**[2]  
   - KTO와 마찬가지로 바이너리 피드백만으로 alignment를 수행하면서,  
   - DPO loss에 대한 **BCE 상계(upper bound)** 를 최소화하는 방식(분류기 logit = reward)을 제안.  
   - 이 과정에서 KTO에 대한 두 가지 비판을 제기한다.[2]  

- **gradient 구조**  
        - (단순화를 위해 $z_{\text{ref}}=0$ 가정 시)  

$$
          \nabla_\theta L_{\text{BCO}}
          = \mathbb{E}[\sigma(-r_\theta)\,\nabla_\theta \beta \log \pi_\theta(y|x)]
          $$

$$
          \nabla_\theta L_{\text{KTO}}
          = \mathbb{E}[\sigma(r_\theta)\sigma(-r_\theta)\,\nabla_\theta \beta \log \pi_\theta(y|x)]
$$
        
  - KTO는 추가적인 $\sigma(r_\theta)$ factor로 인해  
          - reward가 작거나 큰 극단 예시에 대해 gradient가 더 쉽게 0으로 사라져,  
          - 데이터를 “공평하게” 학습하지 못한다고 지적.[2]  
- **참조점(z_ref)의 0 클리핑**  
        - 실제 구현에서 $z_{\text{ref}}$를 $\max(0,\cdot)$로 clip하는데,  
        - 평균 reward는 이론적으로 $-\beta \,\mathrm{KL}(\pi_{\text{ref}} \| \pi_\theta)$ 이므로,  
          - alignment가 진행되려면 이 값이 **더 작아져야(=KL 증가)** 함.  
        - 그러나 z_ref를 0 이상으로 고정하면 KL이 충분히 커지지 못해  
          - 모델이 reference에 너무 가까이 머무르고 충분히 학습되지 못한다고 분석.[2]  
   - BCO는 reward shift 기법으로 이를 보완해,  
     - 다양한 벤치마크에서 **KTO보다 일관되게 높은 성능**을 보고한다.[2]

2. **Preference Tuning Generalization Study**[3]  
   - SFT, DPO, KTO, ORPO, PPO, GRPO 등을  
     - **도메인 시프트(요약·QA, 소스/타깃 도메인)** 상황에서 비교.  
   - 결과적으로,  
     - DPO·KTO·ORPO 같은 **offline 정렬 방법은 in-domain 성능은 매우 좋지만, OOD 일반화에는 약하다**고 보고.[3]  
   - 특히 LLaMA-3.1-8B 실험에서,  
     - DPO source winrate는 ~90% 수준이지만, 타깃 도메인에서 큰 성능 저하(일반화 gap ↑)가 관찰되고,  
     - KTO도 ORPO와 유사하게 상당한 OOD 성능 저하를 보인다.[3]  
   → 즉, KTO가 **“alignment 데이터 분포 내” 일반화와 다운스트림 벤치마크**에는 강점이 있지만,  
   **강한 도메인 시프트 하 OOD 일반화에는 PPO/GRPO 등 온라인 방식보다 불리**할 수 있음을 시사한다.

3. **Rethinking Evaluation of Alignment Methods**[4]  
   - DPO, ORPO, KTO, PPO, SFT 등 다양한 방법을  
     - 사실성, 다양성, 간결성, proactivity, safety 등 다차원에서 ID/OOD 모두 평가.  
   - 결과 요약:[4]  
     - **KTO는 낮은 temperature에서 OOD factuality와 safety 측면에서 상대적으로 우수**,  
     - 하지만 proactivity·다양성 등 다른 축에서는 DPO·PPO가 더 강한 경우가 많다.  
   - 이 논문 역시 **offline 방법(특히 ORPO)은 일반화가 가장 약하고, DPO·KTO는 중간 정도, PPO·GRPO가 가장 균형 잡힌 일반화**를 보인다고 정리한다.[4][3]

***

## 6. “모델 일반화 성능 향상 가능성” 관점에서의 KTO

### 6.1 KTO가 일반화에 유리한 구조적 요인

1. **바이너리 피드백 활용**  
   - ±1 같은 coarse signal은 fine-grained 선호점수보다  
     - annotation 노이즈에 덜 민감하고,  
     - 훨씬 대규모 데이터를 수집하기 쉬우므로  
   - 충분히 큰 스케일에서 **데이터 양 자체로 일반화를 끌어올릴 수 있는 잠재력**이 있다.[1]

2. **HALO 인덕티브 바이어스**  
   - concave value function은  
     - 이미 reward가 충분히 좋은 예시에 대해서는 업데이트를 줄여  
       - over-optimization / reward hacking을 완화하고,  
     - noisy extreme 예시를 자연스럽게 down-weight한다.[1]  
   - 이는 특히 **open-ended generation**에서  
     - 과도한 길이 증가(DPO의 길이 폭발)나 특정 패턴으로의 collapse를 줄여  
     - 언어·수학·코드 등 다양한 태스크에서의 **균형 잡힌 성능**으로 이어질 수 있다.

3. **노이즈·비추이 선호에 대한 견고성**  
   - Prop. 4.1, Thm. 4.3이 보여주듯,  
     - 회복 불가능한 noisy 선호(상충 annotator)를 무시하거나,  
     - 최악의 경우에도 효용을 떨어뜨리지 않는 방향으로 작동한다.[1]  
   - 이는 crowd-sourced preference 데이터(SHP, OpenAssistant, UltraFeedback)와 같이  
     - annotator 분포가 다양하고 모순이 많은 환경에서 **실질적인 일반화 이득**을 제공할 수 있다.

4. **SFT 생략 가능성**  
   - 강력한 pretrain 모델(LLaMA-13B/30B)에서는  
     - SFT 없이 KTO만으로도 SFT+KTO 수준의 품질을 얻는다.[1]  
   - 이는 SFT가 유발할 수 있는 일부 overfitting(특정 instruction style에 갇힘)을 줄이고,  
     - 보다 **pretrain 분포에 가까운 범용적 능력**을 유지한 채 alignment를 수행할 수 있음을 시사한다.

### 6.2 그러나: 후속 연구가 보여준 일반화 상한선

- **Task/domain shift** 상황(요약 ↔ QA, 소스 ↔ 타깃 도메인)에서,  
  - DPO·KTO·ORPO 같은 offline 방법은 in-domain winrate는 매우 높지만,  
  - OOD에서는 PPO·GRPO 등 online 방법보다 **크게 성능이 떨어지는 경향**이 관찰된다.[3]  
- **평가 프레임워크 논문들**은 공통적으로  
  - offline 정렬은 “지금 정렬된 도메인”에 대해서는 강하지만,  
  - 새로운 도메인·작업 분포로의 전이는 제한적이라고 보고한다.[4][3]  

**요약하면**:

- KTO는  
  - **동일/근접 분포 내에서의 일반화와 다양한 다운스트림 벤치마크에서 매우 강력**하고,  
  - noisy·다중 annotator 환경에서는 DPO보다 좋은 선택이 될 수 있다.  
- 반면,  
  - **강한 도메인 시프트나 새로운 태스크로의 전이**에서  
  - PPO/GRPO와 같은 online RL 기반 방법만큼의 **OOD 일반화 성능을 보장하지는 못한다**는 것이 현재까지의 실험적 합의에 가깝다.[4][3]

***

## 7. 2020년 이후 관련 최신 연구와의 비교·분석

여기서는 특히 **정렬 방법의 구조와 일반화**를 기준으로 KTO를 위치시킨다.

### 7.1 RLHF 계열

- **Ouyang et al., 2022 (InstructGPT)**[1]  
  - 전통적인 3단계(SFT → reward model → PPO) 구조.  
  - 온라인 policy gradient로 ID/OOD 모두에서 비교적 안정적인 일반화를 보이지만,  
    - 구현·튜닝 난이도와 compute cost가 매우 높음.  
- KTO와의 대비:
  - RLHF는 reward model이 필요, KTO는 필요 없음.  
  - RLHF는 pairwise 혹은 다중 completion 필요, KTO는 **단일 completion + 이진 피드백**만 있으면 됨.  
  - 일반화 측면에서는 RLHF(PPO)가 **도메인 시프트에 더 강한 것으로** 보고된다.[3]

### 7.2 Offline Preference Optimization 계열

1. **DPO (Rafailov et al., 2023)**[1]  
   - Bradley–Terry 기반으로 RLHF objective를 폐형식(supervised objective)으로 변환.  
   - 현재까지 가장 널리 쓰이는 offline alignment 접근법.  
   - 장점: 파이프라인 단순, reward model 불필요.  
   - 단점: preference 쌍 데이터 필요, 길이 폭발, noisy·비추이 선호에 취약.  
   - KTO: 같은 HALO 계열이지만 효용 중심, 바이너리 데이터, 노이즈에 더 견고.

2. **IPO, ORPO, SimPO 등 변형들**[2][3][5]  
   - IPO: regularization을 추가해 DPO의 overfitting 억제.[2]  
   - ORPO: odds-ratio 기반 단순화된 objective로, 구현이 간편하지만 일반화가 가장 약한 편으로 보고.[4]  
   - 이러한 방법들은 대부분 **pairwise preference를 전제로 하며**, noisy·multi-annotator에 대한 이론적 worst-case 보장이 KTO만큼 명확하지 않다.[1][4]

3. **BCO (Binary Classifier Optimization, 2024)**[2]  
   - KTO와 같이 **바이너리 피드백만으로 alignment**를 수행하되,  
   - 분류기의 logit을 reward로 보고 BCE loss가 DPO loss의 상계임을 증명, reward shift로 상계 gap을 줄인다.  
   - 실험적으로  
     - 작은 모델에서는 KTO와 유사하거나 조금 낫고,  
     - 큰 모델·실세계 데이터(Likert-5 → binary 변환)에서는 **DPO·KTO 모두를 능가**.  
   - 일반화 측면에서는  
     - DPO와 유사한 경향(in-domain 강, OOD에서는 online RL보다 약)으로 보고된다.[2][3]

KTO는 이들 중  
- “**Prospect Theory에 뿌리를 둔 HALO**”라는 가장 명확한 인지과학적 해석을 가진 방법이며,  
- 바이너리 피드백 계열의 선구자 역할을 했다.  
이후 BCO 등이 이 선로를 이어받아 더 안정적이고 일관된 성능을 제공하는 방향으로 발전하고 있다.[2]

### 7.3 Diffusion·멀티모달 확장

- **Diffusion-KTO (Li et al., 2024)**[6]  
  - text-to-image diffusion 모델에 KTO 프레임워크를 확장.  
  - per-image binary feedback(좋아요/싫어요)만으로  
    - aesthetics·PickScore·ImageReward·CLIP·HPS v2 등 여러 자동 지표와  
    - 사람 평가에서 **SFT, Diffusion-DPO, AlignProp보다 일관되게 높은 winrate**를 달성.[6]  
  - 이는 KTO식 “효용 최적화 + 바이너리 피드백” 접근이 언어 모델에 한정되지 않고,  
    - 이미지 도메인에서도 강력한 **일반화·안정성**을 제공할 수 있음을 보여준다.

- **Text-to-Image Alignment Paradigm 일반화 연구**[7][8]  
  - f-divergence 기반 통합 프레임워크에서 Diffusion-DPO, Diffusion-KTO 같은 loss들을 비교·일반화.  
  - loss 선택이 다양성–정렬도–안정성 간 트레이드오프에 미치는 영향을 정량 분석.

### 7.4 이론·평가 프레임워크

- **RLHF ↔ Mutual Information / Contrastive Learning 연결**[9][10][11][5]  
  - RLHF와 DPO, KTO 등을  
    - mutual information 최대화,  
    - reward-weighted SFT,  
    - f-divergence 최소화 등 공통 틀로 재해석.  
  - 이 맥락에서 KTO는 “특정 모양의 value function을 가진 contrastive/MI 최대화”로 볼 수 있다.

- **Generalization-oriented Evaluation**[4][3]  
  - Alignment 기법을  
    - factuality, diversity, conciseness, proactivity, safety, OOD gap 등  
    - 다차원 지표로 평가하는 틀을 제안.  
  - 공통 발견:  
    - offline 정렬(DPO, KTO, ORPO)은 높은 in-domain winrate와 안전성, factuality를 가지지만,  
    - 도메인 시프트 하에서는 **online RL 계열(PPO, GRPO 등)이 더 안정적으로 일반화**한다.

***

## 8. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 8.1 영향

1. **“손실 함수 설계”를 alignment의 1급 객체로 부상**  
   - 기존에는 주로 **데이터(더 좋은 preference, 더 많은 label)**에 초점이 있었으나,  
   - KTO는 “같은 데이터라도 **어떤 HALO를 쓰느냐**가 큰 차이를 만든다”는 것을 보여주었다.[1]  
   - 이후 BCO, ORPO, Diffusion-KTO 등 다양한 HALO/비-HALO 설계가 등장하면서  
     - **Loss Engineering**이 alignment의 핵심 축으로 자리 잡았다.

2. **바이너리 피드백의 부상**  
   - “선호쌍”이 아니라 “좋음/나쁨”만으로도 SOTA 정렬이 가능함을 보임으로써,  
   - 실제 서비스에서 쌓이는 thumbs-up/down 로그를  
     - 직접 alignment에 활용하는 연구들이 빠르게 늘어났다.[2][12][13][14][6]

3. **Prospect Theory·행동경제학과 AI Alignment의 접점 확대**  
   - KTO는 Prospect Theory를 직접적으로 참조한 첫 대규모 LLM alignment 작업 중 하나로,  
   - 이후 효용 함수 설계, risk aversion·loss aversion 조정 등을 통해  
     - “어떤 인간 인지모델이 alignment에 적합한가?” 라는 방향의 연구를 촉발했다.[1][6]

4. **멀티모달·개인화 alignment로의 확장**  
   - Diffusion-KTO는 per-image binary feedback으로 text-to-image 모델을  
     - preference·aesthetic·텍스트 정합도 등에서 크게 향상시키면서,  
     - 한 사용자/집단의 취향에 맞춘 **개인화 alignment** 가능성을 보여주었다.[6]

### 8.2 앞으로 연구 시 고려할 점

1. **효용 함수의 선택 (언어 vs 돈)**  
   - 현재 KTO의 value function은  
     - 원래 금전적 결과를 대상으로 추정된 Kahneman–Tversky 함수의 단순화 버전(시그모이드)을 사용한다.[1]  
   - 언어·대화·코드의 “좋음/나쁨”은  
     - 금전적 손익과 다른 구조를 가질 수 있으며,  
   - 따라서 **언어 도메인 전용 value function·HALO 설계**가 필요하다.

2. **ID 성능 vs OOD 일반화의 트레이드오프**  
   - DPO·KTO·BCO 등 offline 정렬은  
     - alignment 데이터 분포 내(in-domain) 성능과 안전성을 크게 향상시키는 반면,  
     - 도메인 시프트 하 OOD 일반화는 PPO·GRPO보다 약하다는 결과가 반복적으로 관찰된다.[4][3]  
   - 실제 시스템 설계에서는  
     - **“내가 노리는 배포 환경이 얼마나 distribution-shifted인가?”** 를 기준으로  
     - KTO/BCO와 PPO/GRPO를 혼합하거나 순차적으로 사용하는 전략이 필요하다.

3. **Multi-annotator·모순 선호를 어떻게 다룰 것인가**  
   - KTO는 noisy·비추이 선호에 대해  
     - “안전한 쪽(업데이트 회피)”으로 기울어져 있다.[1]  
   - 하지만 이는  
     - 많은 annotator가 참여하는 경우 “다수의 취향을 잘 평균적으로 반영하는가?”라는 질문과 연결된다.  
   - 미래 연구에서는  
     - annotator ID·집단 정보를 활용한 **group-specific HALO**나  
     - utility를 explicit하게 multi-criteria로 분해하는 방법 등이 중요해질 것이다.

4. **정렬 평가 지표의 다변화**  
   - KTO는 GSM8K·BBH 등 다양한 벤치마크에서 DPO보다 좋은 성능을 보이지만,[1]  
   - 이는 **특정 세트의 다운스트림 태스크**에 대한 결과일 뿐이다.  
   - 향후 연구는  
     - factuality, helpfulness, safety, diversity, proactivity, calibration, OOD gap 등  
     - 다차원 지표에서의 trade-off를 비교하고,  
   - 각 앱리케이션(코딩 어시스턴트, 검색, 상담 등)에 적합한 HALO를 선택·튜닝해야 한다.[4][3]

5. **데이터 수집 전략과 HALO의 공동 설계**  
   - KTO 계열의 강점은  
     - thumbs-up/down 같은 바이너리 피드백이 매우 싸고 많이 모인다는 점이다.  
   - 그러나  
     - 어떤 분포의 바이너리 피드백을 어떤 비율로 수집할지(양성/음성 비율, 도메인 분포 등)는  
     - HALO의 특성과 강하게 상호작용한다(예: $\lambda_D/\lambda_U$ 조정).[1]  
   - 앞으로는 **데이터 수집 정책 + HALO 설계**를 통합적으로 최적화하는 연구가 필요하다.

***

## 9. 정리

- 이 논문은 **“alignment는 단지 cross-entropy를 바꾸는 문제가 아니라, 인간의 인지 편향을 반영하는 손실 함수를 설계하는 문제”**라는 시각을 제시했고,  
- Prospect Theory를 활용한 KTO를 통해  
  - **바이너리 피드백만으로도 DPO급·그 이상의 성능과 좋은 다운스트림 일반화**를 실현할 수 있음을 보였다.[1]  
- 후속 연구(BCO, Diffusion-KTO, generalization study 등)는  
  - 이 아이디어를 LLM·diffusion·다양한 설정으로 확장하면서,  
  - 동시에 **offline 정렬의 OOD 일반화 한계와 KTO 설계상의 약점**도 밝혀냈다.[2][4][3][6]

향후 alignment 연구에서  
- HALO/효용 함수 설계,  
- 바이너리/점수/선호쌍 데이터의 최적 조합,  
- ID 성능–OOD 일반화–안전성 간의 다차원 trade-off를  
정교하게 탐색하는 것이 핵심 과제가 될 것이다.

[1](https://arxiv.org/html/2402.01306v1)
[2](https://arxiv.org/pdf/2404.04656.pdf)
[3](https://arxiv.org/html/2601.05882v1)
[4](https://arxiv.org/pdf/2509.12936.pdf)
[5](https://arxiv.org/html/2601.06108v1)
[6](https://arxiv.org/html/2404.04465v1)
[7](https://arxiv.org/html/2409.09774v1)
[8](https://openreview.net/pdf/136b48036ad4c7fecd63151c8727f43b5fc5f4d8.pdf)
[9](https://arxiv.org/html/2506.22578v1)
[10](https://arxiv.org/html/2502.11026v2)
[11](https://arxiv.org/html/2405.19320v4)
[12](https://winniexu.ca/research/kto)
[13](https://argilla.io/blog/mantisnlp-rlhf-part-7/)
[14](https://arxiv.org/abs/2502.12485)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8b612ab0-ce7b-42b8-9589-1718b5f61b60/2402.01306v4.pdf)
[16](https://www.semanticscholar.org/paper/c0d8e5ee66c279299012cc3b8d0519011b3f4998)
[17](https://arxiv.org/pdf/2402.01306.pdf)
[18](http://arxiv.org/pdf/2409.10479.pdf)
[19](http://arxiv.org/pdf/2502.14187.pdf)
[20](https://arxiv.org/html/2503.04240v1)
[21](http://arxiv.org/pdf/2405.21023.pdf)
[22](https://arxiv.org/pdf/2502.14096.pdf)
[23](https://arxiv.org/pdf/2305.18470.pdf)
[24](https://www.alphaxiv.org/overview/2402.01306v1)
[25](https://arxiv.org/html/2402.01306v3)
[26](https://contextual.ai/better-cheaper-faster-llm-alignment-with-kto/)
[27](https://yongggg.tistory.com/71)
[28](https://liner.com/ko/review/kto-model-alignment-as-prospect-theoretic-optimization)
[29](https://arxiv.org/abs/2402.01306)
[30](https://www.emergentmind.com/papers/2402.01306)
[31](https://huggingface.co/papers/2402.01306)
[32](https://asidefine.tistory.com/280)
[33](https://ebbnflow.tistory.com/386)
[34](https://arxiv.org/html/2405.11143v4)
[35](https://www.semanticscholar.org/paper/KTO:-Model-Alignment-as-Prospect-Theoretic-Ethayarajh-Xu/c0d8e5ee66c279299012cc3b8d0519011b3f4998)
[36](https://www.themoonlight.io/en/review/kto-model-alignment-as-prospect-theoretic-optimization)
