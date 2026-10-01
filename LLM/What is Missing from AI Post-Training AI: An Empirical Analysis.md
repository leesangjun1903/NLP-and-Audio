# What is Missing from AI Post-Training AI: An Empirical Analysis

> **전제와 검증 한계**
> - 이 보고서는 제공된 PDF(arXiv:2608.19072v2, 25쪽)만을 1차 근거로 삼았습니다. 웹 검색은 하지 않았습니다.
> - 논문이 인용한 2026년 문헌은 제 지식 범위 밖이라 내용을 검증할 수 없습니다. 해당 문헌은 "논문이 이렇게 서술한다"는 수준으로만 다룹니다.
> - 그림은 PDF에 포함된 텍스트 라벨과 캡션 위주로 읽었습니다. 육안 판독이 필요한 값은 "근사"라고 표시했습니다.
> - 표기: **[저자]** 논문이 직접 보고한 내용, **[필자]** 제 해석이나 계산, ⚠️ 통계적 취약, ⛔ 비교 불가.
> - 쪽수는 PDF 쪽 번호입니다. Fig/Table 번호는 논문 기준입니다. (요청 3번은 각 문장 뒤 괄호로 반영했습니다.)

---

## 1. Executive Summary (10문장 이내)

1. 이 논문은 LLM 에이전트가 LLM post-training을 end-to-end로 수행하는 상황에서, 종합 점수가 가리는 **실행 수준(execution-level)**과 **전략 수준(strategy-level)** 능력을 분리해 측정합니다. (p1–3)
2. PostTrainBench 궤적 1,338개(벤치마크 7개, base model 4개, 에이전트 프레임워크 3개)를 분석한 결과, 에이전트는 실행을 안정적으로 수행했습니다. 궤적당 평균 3.82회 학습, 13.80회 평가를 하고 모든 벤치마크에서 평균 점수가 base model보다 올랐습니다. (p3–4, Fig 2)
3. 그러나 에이전트는 초기에 **기본 전략에 잠깁니다**. 기본 전략은 과제가 아니라 에이전트를 따라갑니다(Claude Code는 full SFT 71.9%, Codex CLI는 PEFT 89.5%). 인접 학습 쌍 3,557개 중 74개(2.1%)만 전략을 바꿨습니다. (p4, Table 1·4·5)
4. 원인을 경험, 추론, 결정의 세 가지로 나눠 Qwen3-1.7B-Base, GSM8K·HumanEval·AIME 2025, 각 3회, 10시간, A800 4장 조건에서 검증했습니다. (p5, Sec 4)
5. **경험**(journal, skill library, evaluator)은 실행을 개선했습니다. 지시 모델과의 격차 폐쇄율이 48.6%에서 75.1%로 올랐습니다. 그러나 evaluator의 실행 수준 제안은 22/22를 채택하고 전략 수준 제안은 0/21을 채택했습니다. (p6, Table 2, Fig 4)
6. **추론 compute**(토큰 1.8–9.2배)는 쉬운 과제에서 초반에 몰린 이득만 주었습니다. AIME 2025에서는 약 14M 토큰이 문제 1개 개선에 그쳤습니다. (p6–8, Fig 5)
7. **사람의 개입**: 학습 전 검토는 "어떤 전략에 잠기느냐"만 바꿨습니다. 중간 시점의 단일 전환 지시는 같은 예산에서 에이전트 자체 연속 실행보다 최대 +17.44점(GSM8K 47.76→65.20) 높았습니다. "재고하라"는 지시만으로는 모든 fork에서 기존 전략을 재확인했습니다. (p7–8, Fig 6, Table 8)
8. 결론은 에이전트에게 부족한 것이 **"확정된 전략을 다시 열고 다른 전략을 시도하는 결정"**이라는 것입니다. 저자들은 "continue/switch"를 명시적 결정 노드로 만드는 프로토콜과, fork 쌍에서 얻은 학습 신호를 제안합니다. (p8–9, Fig 7)
9. **[필자 평가]** 두 수준 분리와 switch rate 같은 지표는 유용한 기여입니다. 반면 핵심 인과 근거인 fork 실험은 벤치마크당 n=1이고 저자가 사후 선택했으며, 제안한 처방은 실험으로 검증되지 않았습니다. 따라서 "강한 가설을 뒷받침하는 탐색적 증거" 수준으로 읽는 것이 적절합니다.

> 📘 **용어**
> - **Post-training**: 사전학습된 base model에 SFT, RL, DPO 등을 추가로 적용해 쓸모 있는 모델로 만드는 단계.
> - **RSI(Recursive Self-Improvement)**: AI가 AI 자신을 개선하고, 그 개선된 AI가 다시 개선을 반복하는 구상.
> - **Base model / Instruct model**: 전자는 사전학습만 마친 모델이고, 후자는 대화·지시 따르기 후처리까지 한 공식 모델입니다.
> - **PostTrainBench**: 에이전트가 10시간, H100 1장 안에서 LLM을 post-training하게 하는 벤치마크(논문 인용, Rank et al., 2026). 내용은 제가 검증하지 못했습니다.

### 1-1. 연구의 목적과 필요성

- **목적 [저자]**: 에이전트의 post-training 성공이 "고정 계획을 잘 실행한 결과"인지 "실패 시 계획을 전략적으로 수정한 결과"인지를 가려내고, 부족한 능력이 무엇인지 찾는 것입니다. (p1, Abstract)
- **필요성 [저자]**: 기존 연구는 종합 벤치마크 점수만 보고합니다. 그런데 RSI는 "파이프라인 개선"이 아니라 "에이전트가 연구를 수행하는 방식 자체의 개선"을 요구하므로, 두 능력을 구분해야 합니다. (p1)
- **[필자]**: 점수가 같은 두 에이전트도 전략 선택 능력이 다를 수 있습니다. 한쪽은 우연히 기본값이 과제에 맞았을 뿐일 수 있으므로 과정 수준의 진단 지표가 필요하다는 주장은 타당합니다.

---

## 2. 핵심 주장과 근거 (표)

| # | 핵심 주장 | 근거(수치) | 위치 |
|---|---|---|---|
| C1 | 에이전트는 신뢰할 만한 **실행자** | 평균 3.82 학습/13.80 평가 per 궤적. 평균 점수: AIME 3.9(+3.9), ArenaHard 15.7(+15.7), HealthBench 18.5(+18.5), GPQA 25.1(+16.2), HumanEval 39.7(+17.7), GSM8K 50.9(+26.4), BFCL 67.3(+49.8) | p3–4, Fig 2 |
| C2 | 기본 전략은 **에이전트를 따라가지 과제를 따라가지 않음** | Claude Code full SFT 71.9%(166/231), Codex CLI PEFT 89.5%(274/306), OpenCode full SFT 66.4%(184/277). 7×4 모든 조합에서 같은 방향 | p4, Table 1, Table 4 |
| C3 | 학습 시작 후 전략은 **거의 안 바뀜** | 74/3,557(2.1%). 알고리즘 35, 데이터원 38, 학습단계 1 | p4, Table 5, App A.2 |
| C4 | 경험은 **실행만** 개선 | Gap Closed 48.6%→75.1%. evaluator 제거 시 56.8% | p6, Table 2 |
| C5 | 에이전트는 **전략 수준 제안을 전부 거부** | 실행 제안 22/22 채택, 전략 제안 0/21 | p6, Fig 4, Table 7 |
| C6 | 추론 compute는 **전략을 못 바꿈**(이득이 초반에 집중, 어려운 과제는 천장) | GSM8K 개선의 약 90%가 토큰 예산 전반부. HumanEval +51.8 후 +9.8. AIME 약 14M 토큰=문제 1개 | p6–8, Fig 5 |
| C7 | 학습 전 사람 검토는 **어떤 전략에 잠기는지만** 바꿈 | AIME: 검토 후 최고 3/30 이후 회귀. 후속 실행은 하이퍼파라미터 반복 | p7–8, Fig 6a, App D.1 |
| C8 | 중간 시점 단일 지시는 **같은 예산에서 더 나은 경로를 엶** | GSM8K 47.76→65.20(+17.44), HumanEval 52.0→62.8(+10.8), AIME 0/30→2/30 | p7–8, Fig 6b, Table 8 |
| C9 | 이름 없는 "재고하라" 지시는 **효과 없음** | 3개 fork 모두 SFT 유지. 점수는 48.0/31.7/1/30 | p8, App D.2.3, Table 8 |
| C10 | RSI 루프는 **실행 수준에서만 닫힘** | 개념도 | p9, Fig 7 |

---

## 2-1. 상세 설명

### (a) 해결하려는 문제

에이전트 post-training 성능이 "실행 능력"과 "전략 수정 능력" 중 무엇에서 오는지 분리해서 알 수 없다는 문제입니다. 구체적으로는 에이전트가 전략을 못 바꾸는 이유가 경험 부족, 추론 부족, 결정 부재 중 무엇인지를 가립니다. (p1–2)

### (b) 제안 방법 (수식 포함)

**① 두 수준 능력 프레임워크 (Sec 2.1, p3)**

전략 공간은 다음과 같이 정의합니다.

$$\mathcal{S} := \mathcal{P}\times\mathcal{D}\times\mathcal{G}$$

- $\mathcal{P}$: 학습 알고리즘 (SFT, PEFT, RL, 선호최적화, 증류 등)
- $\mathcal{D}$: 데이터원 (curated, self-generated, mixed)
- $\mathcal{G}$: 학습 단계 구성 (예: 단일 SFT vs SFT→RL)
- $\mathcal{X}$: 실행 설정 공간 (데이터 포맷, 하이퍼파라미터, 구현 세부)
- 궤적: $\tau=\{(s_t,x_t,r_t)\}_{t=1}^{T}$. $s_t\in\mathcal{S}$는 t번째 학습 실험의 전략, $x_t\in\mathcal{X}$는 그 실험의 설정, $r_t$는 평가 결과입니다.
- 전이 $t\to t+1$은 $s_{t+1}\neq s_t$이면 **전략 변경**, $s_{t+1}=s_t$이고 $x_{t+1}\neq x_t$이면 **실행 조정**입니다.

**② 전략–실행 격차 분해 (식 1)**

$$\underbrace{V^\star-V(\tau)}_{\text{total gap}}=\underbrace{V^\star-V^\star(s_1)}_{\text{strategy-level gap}}+\underbrace{V^\star(s_1)-V(\tau)}_{\text{execution-level gap}}$$

- $V(s,x)$: 전략 $s$와 설정 $x$의 벤치마크 점수.
- $V^\star(s):=\max_{x\in\mathcal{X}}V(s,x)$: 전략 $s$의 달성 가능한 최고 점수.
- $V^\star:=\max_{s\in\mathcal{S}}V^\star(s)$: 전체 최고 점수.
- $V(\tau):=\max_t r_t$: 궤적의 최고 점수. 초기 전략에 계속 머무는 경우($\forall t, s_t=s_1$)를 가정합니다.
- **[필자]**: 이 식은 항등식(더하고 빼기)이라 그 자체가 새로운 정리는 아닙니다. 실무적 의미는 " $V^\star(s)$, $V^\star$를 실제로 알 수 없으므로 전략 수준 격차는 직접 추정할 수 없다"는 데 있습니다. 논문은 이를 fork 실험으로 부분 추정합니다.

**③ 락인 지표 (식 2)**

$$\rho(\tau):=\frac{1}{T-1}\sum_{t=1}^{T-1}\mathbf{1}[s_{t+1}\neq s_t],\qquad \kappa_a:=\max_{s\in\mathcal{S}}\pi_a(s)$$

- $\rho(\tau)$: 궤적의 switch rate. $\rho=0$이면 완전 락인입니다.
- $\mathbf 1[\cdot]$: 조건이 참이면 1, 아니면 0인 지시함수.
- $\pi_a(s)=\Pr[s_1=s\mid a]$: 에이전트 $a$의 초기 전략 분포.
- $\kappa_a$: 최빈 전략이 차지하는 비중(default-strategy concentration).
- 풀링 값: $\bar\rho=74/3{,}557\approx 2.1\%$ (전체 인접 쌍 기준).

**④ 학습 알고리즘 목적함수 (App A.2, p15)**

$$J_{i,t}(\theta)=\mathbb{E}_{z\sim D_{i,t}}\big[\ell_{p_{i,t}}(\theta;z,\lambda_{i,t})\big]$$

- $i$: 궤적 인덱스, $t$: 실험 인덱스, $\theta$: 모델 파라미터.
- $D_{i,t}$: 학습 데이터 분포, $\lambda_{i,t}$: 나머지 하이퍼파라미터.
- $\ell_{p_{i,t}}$: 인식된 알고리즘 $p_{i,t}$(SFT, GRPO 등)의 손실.

**⑤ Gap Closed (App B, p17)**

$$\text{Gap Closed}=\frac{\text{Avg}-\text{Avg}_{\text{base}}}{\text{Avg}_{\text{instruct}}-\text{Avg}_{\text{base}}}$$

- 검산 **[필자]**: Experience-driven는 $(48.55-5.44)/(62.83-5.44)=75.1\%$이고, Opus 4.6 단독은 $(33.34-5.44)/57.39=48.6\%$입니다. 표와 일치합니다.

**⑥ 예산 제약 (App D.2.1, p21)**: $T_{\text{guided}}\le B_{t_b}$. 여기서 $B_{t_b}$는 분기점 $t_b$ 시점의 잔여 예산입니다.

**⑦ pass@k (논문에 식 없음, 표준 정의 [필자 보충])**: $\text{pass@}k=\mathbb{E}_{\text{problems}}\!\left[1-\binom{n-c}{k}/\binom{n}{k}\right]$. 문제당 $n$개를 샘플링하고 그중 정답이 $c$개일 때의 값입니다 (Chen et al., 2021). 논문은 AIME에서 문제당 8개 생성으로 pass@8을 씁니다.

> 📘 **용어**
> - **SFT(Supervised Fine-Tuning)**: 정답 예시를 모방하도록 우도를 최대화하는 지도 미세조정.
> - **Full-parameter SFT vs PEFT**: 전자는 모든 가중치를 갱신하고, 후자는 일부 어댑터만 학습합니다.
> - **LoRA/QLoRA**: 저랭크 어댑터로 적은 파라미터만 학습하는 PEFT 기법. QLoRA는 양자화된 모델 위에서 LoRA를 학습합니다.
> - **RL / PPO / GRPO**: 보상 신호로 모델 출력을 최적화하는 학습. PPO는 대표적 정책경사 알고리즘이고, GRPO는 그룹 샘플의 상대 보상으로 이점을 계산하는 변형입니다.
> - **DPO**: 선호/비선호 응답 쌍으로 직접 학습하는 선호최적화.
> - **RFT(Rejection-sampling FT)**: 모델이 생성한 답 중 검증을 통과한 것만 골라 SFT에 쓰는 방식.
> - **DAPO**: GRPO 계열 대규모 RL 개선 레시피(논문 인용).
> - **pass@k**: k번 시도 중 하나라도 정답이면 성공으로 보는 지표. AIME 2025는 30문제라 pass@1은 1문제=3.33%p로 거칩니다.
> - **Lock-in(락인)**: 초기에 정한 방식에서 벗어나지 못하는 현상.

### (c) 모델 구조 (시스템 구성)

이 논문은 새 신경망 구조를 제안하지 않습니다. "모델 구조"에 해당하는 것은 실험 시스템입니다. (p5, App C, p17–19)

```
[Main Agent: Claude Code + Opus 4.6] ──학습/평가 요청──► [Evaluator Agent (동일 설정)]
      │  ▲                                                  │ 체크포인트 진단 + 실행/전략 제안
      ▼  │                                                  ▼
[Experiment Journal: plan / observation / lesson / eval_result / eval_analysis (append-only)]
      ▲
[Skill Library: 908개 문서(≈937K 단어) → 60쪽 wiki(≈20K) → SKILL.md(≈1.2K 단어/파일), seed 5종]
Base model: Qwen3-1.7B-Base | 10시간 | A800 ×4 | 조건당 3회
```

- 모든 학습 결정은 메인 에이전트가 하고, 프레임워크는 정보만 제공합니다. (App C)
- 개입 실험은 위 구조에 사람 검토(학습 전)나 단일 전환 지시(중간 fork)를 추가합니다. (App B, D)

> 📘 **용어**
> - **Claude Code / Codex CLI / OpenCode**: 코딩 에이전트 실행 프레임워크(scaffold).
> - **Fork**: 진행 중인 실행의 특정 체크포인트와 상태를 복제해 서로 다른 지시로 분기 실행하는 것.
> - **Evaluator agent**: 체크포인트를 평가하고 진단하는 별도 에이전트.

### (d) 성능 향상 및 한계

**Table 2 요약 (p6, n=3, 평균, 괄호는 base 대비 증감)**

| 설정 | GSM8K | HumanEval | AIME 2025(pass@8) | Avg | Gap Closed |
|---|---|---|---|---|---|
| Base | 10.84 | 5.48 | 0.00 | 5.44 | 0% |
| Official instruct | 88.70 | 66.46 | 33.33 | 62.83 | 100% |
| Opus 4.6 (CC) | 64.70±9.6 | 32.00±10.4 | 3.33±0.0 | 33.34 | 48.6% |
| GLM-5.2 (CC) | 49.51±7.2 | 44.51±9.8 | 3.33±0.0 | 32.45 | 47.1% |
| GPT-5.2 (Codex) | 43.44±4.1 | 13.41±8.7 | 0.00 | 18.95 | 23.5% |
| **Experience-driven** | **77.30±3.8** | **62.80±6.1** | **5.56±1.6** | **48.55** | **75.1%** |
| w/o journal | 74.50 | 50.20 | 4.44 | 43.05 | 65.5% |
| w/o skill library | 73.10 | 54.50 | 3.33 | 43.64 | 66.6% |
| w/o evaluator | 68.20 | 42.60 | 3.33 | 38.04 | 56.8% |

**성능 향상**
- 경험 프레임워크는 세 벤치마크 모두에서 Opus 4.6 단독 대비 평균이 올랐습니다. (p6)
- 에이전트가 전환 지시를 받으면 비친숙 전략(GRPO)도 구현하고 확장합니다. (p7, App D.2)

**한계 [저자]** (p9)
- base model 1개(Qwen3-1.7B-Base), 벤치마크 3개, 10시간·A800 4장, 3회 반복.
- 중간 비교는 벤치마크당 무작위 선택 궤적 1개이고, 분기점과 대안 전략을 저자가 선택했습니다.
- "더 나은 전략이 도달 가능했음"을 보일 뿐 최적 전략이나 항상 전환이 이득임을 보이지 않습니다.
- 최신 모델(Opus 4.8, Opus 5, Fable 5)은 변동성과 오염(train-on-test) 때문에 제외했습니다. (p5 각주 2, App E.3)

> 📘 **용어**
> - **Reward hacking**: 보상이나 평가 허점을 이용해 실제 성능 개선 없이 점수만 올리는 행위.
> - **Data contamination(train-on-test)**: 평가 데이터가 학습에 섞이는 오염.
> - **Qwen3-1.7B-Base**: 17억 파라미터 사전학습 모델.
> - **GSM8K / HumanEval / AIME 2025**: 초등 수학 문장제 / 코드 생성 / 고난도 수학 경시(30문제).
> - **A800 / H100**: NVIDIA GPU. 두 분석은 하드웨어가 다릅니다(H100 1장 vs A800 4장).

---

## 4. 저자 보고 vs 필자 해석 (분리)

### 4.1 연구 주제
- **[저자]**: AI post-training AI에서 부족한 것은 실행 능력도 자원도 아닌 "전략 재개방 결정"입니다. (p2, p8)
- **[필자]**: "결정이 없다"는 표현은 하나의 설명입니다. 같은 관찰은 컨텍스트 앵커링, 매몰비용, 위험 회피, 지시 순응성 차이, 모델의 RL 경험 부재 등으로도 설명됩니다. 논문은 이들을 구분하는 실험을 하지 않았습니다. 저자도 "long-horizon 에이전트의 알려진 어려움과 일치한다"(p7)고만 표현합니다.

### 4.2 방법
- **[저자]**:
  - 궤적 라벨링은 규칙 스크립트, LLM, 저자 검토를 결합했습니다. (p3, App A)
  - 세 가설은 순차적으로 검증했습니다(경험→추론→결정). (p5–8)
  - 사람 개입은 학습 전 검토와 중간 단일 지시 두 시점으로 했습니다.
- **[필자]**:
  - "추론 compute"는 실제로는 에이전트 누적 토큰 소비를 관찰한 것입니다. thinking 예산을 통제한 실험이 아니라, 에이전트 로그와 툴 출력이 섞인 상관 분석입니다.
  - 경험 프레임워크에는 evaluator 에이전트의 추가 compute가 섞여 있어 "경험 효과"와 "compute 효과"가 분리되지 않습니다.
  - 전략 변경의 정의가 분류체계에 의존합니다(Full SFT와 PEFT를 서로 다른 알고리즘으로 취급). 2.1%라는 수치는 이 정의에 민감할 수 있습니다.

### 4.3 결과
- **[저자]**: 위 C1–C10. 특히 "up to 17.44 points", "22/22 vs 0/21", "2.1%".
- **[필자]**:
  - 강한 근거: 에이전트별 기본 전략 차이(Table 4의 일관된 방향), 제안 채택률의 비대칭, 낮은 switch rate.
  - 약한 근거: fork 실험 (5절 참조). 이 실험이 논문의 가장 중요한 인과 주장을 떠받치는데 표본이 사실상 3건입니다.
  - 해석상 주의: 사람 지시가 효과적인 이유는 "결정을 대신해서"일 수도 있고, "권위 있는 지시라서 따랐기 때문"이거나 "사람이 사후적으로 정답(GRPO)을 알고 골랐기 때문"일 수도 있습니다.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 표시 | 항목 | 내용 | 위치 |
|---|---|---|---|
| ⚠️ | 반복 수 n=3 | 모든 Table 2 비교가 3회 평균입니다. 표준편차 종류(표본/모집단)도 미명시입니다. **[필자]** AIME 5.56±1.6은 문제 수 (1,2,2)/30 기준 모집단 SD(≈1.57)와 일치합니다. 표본 SD면 ≈1.9입니다. | p6, Table 2 |
| ⚠️ | 유의성 검정 부재 | **[필자 암산, 근사]** Welch 기준으로 Experience vs Opus 4.6은 GSM8K에서 t≈2.1(df≈2.6, 비유의 가능성 큼), HumanEval에서 t≈4.4(df≈3, 유의 가능성)입니다. ablation 차이(예: journal 제거 GSM8K −2.8)는 SD 범위 안입니다. | Table 2 |
| ⚠️ | AIME 2025 분해능 | 30문제라 1문제=3.33%p입니다. 5.56 vs 3.33은 평균 약 0.7문제 차입니다. 저자도 "within evaluation variance"라고 인정합니다. | p7, App B |
| ⚠️ | **Fork 실험 n=1/벤치마크** | 분기점과 대안을 저자가 **완료된 궤적을 본 뒤** 선택했습니다(사후 선택 편향). 반복도 신뢰구간도 없습니다. AIME 0/30→2/30은 저자 스스로 "small-sample existence result"라 부릅니다. | App D.2, p21 |
| ⚠️⛔ | **Fork 기준선 vs Table 2** | Fork의 "recorded continuation" GSM8K 47.76은 같은 프레임워크 평균 77.30±3.8보다 약 30점 낮습니다. 따라서 +17.44는 "상대적으로 낮은 한 궤적"과의 비교입니다. **[필자]** 가이드 분기 65.20은 Table 2의 평균(77.30)보다 낮고 Opus 4.6 단독 평균(64.70)과 비슷합니다. HumanEval 가이드 분기 62.8은 Experience 평균 62.80과 같습니다(우연인지 확인 불가). 즉 "가이드가 평균적 자율 실행보다 낫다"는 주장은 뒷받침되지 않습니다. | Fig 6b, Table 2, Table 8 |
| ⚠️ | 제안 채택률 22/22, 0/21 | 벤치마크당 대표 궤적 1개에서 추출했고, 같은 주제가 평가 사이클마다 반복 집계됩니다(독립이 아님). GSM8K의 "majority voting 0/7"은 App C.3의 전략 정의(알고리즘 전환, 단계 추가/제거)에 부합하지 않는 추론 시점 기법으로 보입니다. | Fig 4, Table 7, p19 |
| ⚠️ | switch rate 2.1% | 쌍이 궤적 내에서 상관되어 독립이 아닙니다. 알고리즘 변경 35건이 **16개 궤적**에 집중되고 11개는 2회 이상 전환했습니다(App A.4). 클러스터 부트스트랩 CI가 없습니다. 비교 기준(사람 연구자의 전환율)도 없습니다. | Table 1, App A.4 |
| ⚠️ | 초기 전략 인식률 | 1,338개 중 814개(60.8%)만 인식되었습니다. 결측 메커니즘 불명입니다. "락인 시점"은 측정되지 않았고 계획 단계로만 정의됩니다(각주 1). | p4, App A.2 |
| ⚠️ | 라벨링 신뢰도 | 규칙, LLM, 저자 검토만 있고 평가자 간 일치도가 없습니다. 실험 5,111개 중 733개는 미라벨이며, 인접 쌍은 미라벨 실험을 건너뛰어 연결됩니다("SFT→unknown→GRPO"를 1쌍으로 계산). | App A.2–A.3 |
| ⚠️ | 교란 | "에이전트 기본값"이 모델, scaffold, 시스템 프롬프트와 혼재합니다. OpenCode의 모델 구성은 공개되지 않았습니다. 또한 PEFT 선호는 자원 제약이나 scaffold 습관일 수도 있습니다. | Table 1, Table 4 |
| ⚠️ | 락인의 해악 미입증 | BFCL처럼 전환율 0%이면서 점수가 높은 경우가 있습니다(Claude Code BFCL 0.860). AIME의 낮은 점수는 base model 용량 한계일 수도 있고, 생성 길이 제한 같은 실행 문제일 수도 있습니다(App E.2: 2,048 토큰 캡 수정으로 1→7/30). | Table 4, p23 |
| ⚠️ | Table 6의 전환 결과 | SFT→DPO 사례에 반례가 있고(0.539→0.478), 표본이 작으며 서로 다른 proxy 지표를 씁니다. | Table 6 |
| ⛔ | Gap Closed 합산 | AIME(pass@8)와 GSM8K/HumanEval(pass@1)을 단순 평균합니다. instruct 모델의 평가 모드(thinking 여부 등)는 제공 텍스트에 없습니다. | Table 2, App B |
| ⛔ | 토큰 효율 수치 | "3.7M, 0.9M 토큰/점"의 정의(분모·분자)가 불명확합니다. **[필자]** HumanEval은 전반부 +51.8, 후반부 +9.8로 합 61.6이지만 Table 2의 증가량은 +57.3입니다. GSM8K는 후반부 +2.1이 전체 개선의 약 10%라는 설명이 Table 2의 +66.5와 맞지 않습니다. 대표 run과 평균 run 혼용 가능성이 있습니다. | p6, Fig 5, Table 2 |
| ⛔ | Fig 3의 시간 | 전체 소요가 Experience 4.8h(GSM8K) vs Baseline 9.2h로 더 짧습니다. 본문의 "같은 예산에서 더 많은 실험 완료"를 뒷받침하는 실험 횟수 수치가 본문에 없습니다. 대표 run 1개입니다. | p5, Fig 3 |
| ⛔ | 분석 간 환경 | Appendix A(H100 1장, 20개 구성, 평균 Fig 2)와 Sec 4(A800 4장, Qwen3-1.7B 단일)는 점수를 직접 비교할 수 없습니다. | p3, p5 |
| ⛔ | App E 표 | 평가 표본 크기가 제각각입니다(@50, @150, @164, 전체). Fable 5의 AIME F는 비공식 평가 경로(0.133)입니다. | Table 9–11 |
| ⚠️ | Fig 6a | "best=3/30"은 여러 반복 중 최댓값입니다. baseline(1/30), experience(2/30)와 같은 통계량인지 불명확합니다. 1~2문제 차이입니다. | p8 |
| ⚠️ | 체크포인트 선택 | 제공 텍스트만으로는 test 벤치마크로 선택·진단했는지 불명확합니다. 그렇다면 낙관적 편향이 있습니다. | App C, E |

---

## 6. 문서가 답하지 않는 질문

1. **전환이 항상(또는 언제) 이득인가?** 전환이 해로운 경우의 비율과 최적 전환율이 없습니다(Table 6에 반례만 있음).
2. **락인은 모델의 성질인가, scaffold의 성질인가?** 모델×scaffold 요인설계가 없습니다.
3. **왜 거부하는가?** 컨텍스트 앵커링, 매몰비용, 지시 순응 중 어느 것인지 메커니즘 실험이 없습니다. 예를 들어 컨텍스트 요약 리셋이나 "신선한 눈" 서브에이전트 같은 대조가 없습니다.
4. **처방은 작동하는가?** 명시적 결정 노드와 fork 기반 학습 신호는 제안일 뿐 구현·검증되지 않았습니다.
5. **지시의 어떤 요소가 효과를 냈는가?** 권위(명령)인지 대안 명시인지 분리되지 않았습니다. 동일 대안을 LLM evaluator가 제안했을 때는 0/21이었지만, 사람의 "명령"과 evaluator의 "제안"은 형식이 다릅니다.
6. **프롬프트 민감도**: "재고하라"의 문구 변형, 강도, 반복 효과가 없습니다(n=1 fork).
7. **일반화**: 더 큰 모델, 다른 base, 비검증가능 과제(작문·HealthBench)에서도 같은지 모릅니다. 전환 후 모델의 OOD 성능이나 망각도 측정되지 않았습니다.
8. **최신 에이전트**: Opus 4.8, Opus 5, Fable 5에서도 성립하는지 불명확합니다.
9. **비용·위험**: 전환의 토큰/GPU 비용, 사람 검토의 시간 비용, 검토자 간 일관성이 없습니다.
10. **사람 연구자 기준선**: 인간의 전략 전환율 또는 인간-에이전트 성능 비교가 없습니다.
11. **정의 민감도**: 전략 분류체계를 바꿨을 때(예: PEFT를 실행으로 재분류) 2.1%가 어떻게 변하는지 모릅니다.
12. **Reward hacking·오염의 빈도**(App E.3)는 미래 과제로 남겨졌습니다.

---

## 7. 가장 중요한 그림 5개 해석

**① Figure 1 (p2): 개요**
- 상단은 GSM8K 10시간 세션의 시간–점수 곡선입니다. Claude Code는 SFT v1→v4로 약 0.65까지 올린 뒤, 남은 약 60% 예산에서 lr 조정, 데이터 재샘플링, epoch 추가를 했으나 **+0.00**입니다. Codex CLI는 LoRA v1–v4를 했고 "RL probe → reverted"가 보입니다.
- **[필자]** 단일 run 예시라 대표성은 알 수 없습니다. 흥미로운 점은 Codex가 RL을 시도했다가 되돌렸다는 것입니다. "시도 능력 없음"이 아니라 "유지 편향"을 시사하며, "결정이 없다"보다 "결정이 보수적으로 편향된다"가 더 정확한 서술일 수 있습니다.

**② Figure 4 (p6): 제안 채택 현황**
- 평가 사이클별로 실행 제안은 채택(●), 전략 제안은 거부(×). GSM8K 11사이클, HumanEval 14사이클(SFT→RL 0/11), AIME 5사이클(SFT warm-up 0/3).
- **[필자]** 논문의 가장 설득력 있는 증거입니다. 단 대표 궤적 3개이고 같은 제안이 사이클마다 반복 집계됩니다(독립 표본이 아님). 사이클 수는 같은 주제를 반복 센 것이므로 실질적 서로 다른 제안은 몇 개 안 됩니다. 또한 HumanEval에서 "RL 스크립트를 작성하고도 실행하지 않았다"는 서술(p6)은 인지–행동 괴리를 보여주는 흥미로운 관찰입니다.

**③ Figure 5 (p7): 누적 토큰 vs 평균 정확도**
- 두 곡선(Autonomous, Experience-driven)이 초반 급상승 후 완만해지고, AIME 점(△)은 거의 바닥에 머뭅니다. 음영 밴드가 넓습니다(근사 판독).
- **[필자]** "수확 체감" 서사는 직관적이지만, 평균이 쉬운 과제의 이득에 지배되고 밴드가 넓어 정량 결론은 약합니다. 토큰 소비는 추론 품질의 대리 지표로 약하므로, 인과를 말하려면 thinking 예산을 통제한 실험이 필요합니다.

**④ Figure 6 (p8): 사람 개입**
- (a) AIME 초기 검토: 최고 3/30에 도달한 뒤 이후 반복이 회귀합니다. "no strategy switch, only hparam tuning" 표기가 있습니다.
- (b) 중간 지시: GSM8K 47.76%→65.20%(+17.44pp), HumanEval 52.00%→62.80%(+10.80pp), AIME 0.00→6.67%(+6.67pp). 분기점은 각각 2h51m, 5h14m, 4h48m입니다.
- **[필자]** 가장 인용될 그림이면서 가장 조심해서 읽어야 하는 그림입니다. 5절의 기준선 문제를 보면, +17.44는 "낮은 궤적 1개" 대비이고 Table 2의 평균 성능 대비로는 우위가 사라집니다. 방향성은 시사하지만 효과 크기는 신뢰하기 어렵습니다.

**⑤ Figure 7 (p9): 가정 vs 관찰된 루프**
- (a) 전략→실행→평가→전략 수정으로 전역적으로 닫히는 루프, (b) 실제로는 "repair & retry"만 닫히고 "revision not taken"으로 표시됩니다.
- **[필자]** 개념도이므로 정량적 증거는 아닙니다. 그러나 이 논문의 핵심 메시지(스케일은 닫힌 루프를 깊게 만들 뿐)를 요약하고, 전략 수준 지표 제안(Sec 5)의 근거가 됩니다.

> 📘 **참고**: Fig 2(실행 신뢰성 분포), Fig 3(시간 사용), Fig 8(학습 동역학), Fig 9(AIME 궤적 주석)는 보조 증거입니다. Fig 9에서는 v1–v6 내내 보상 포맷, 엔트로피 계수 같은 국소 수정만 반복되고 "high-level strategy remains anchored"로 끝납니다.

---

## 8. 결론

### 저자의 시사점 (Sec 5, 7)
- **스케일링은 이미 닫힌 루프를 깊게 할 뿐**입니다. 초기 전략의 질이 상한을 정합니다. (p8)
- **RSI 진척 측정에는 전략 수준 지표가 필요**합니다. 제안 지표는 switch rate(식 2), 전략 제안 채택률(Fig 4), fork 기반 전략 격차(Sec 4.3.2)입니다. (p8)
- **근시일 처방은 중간 시점 사람 가이드, 장기 처방은 에이전트 자체 결정**입니다. 이를 위해 (1) 평가 후 "continue or switch"를 명시적 결정 노드로 만드는 상호작용 프로토콜, (2) 같은 체크포인트·같은 잔여 예산에서 continue/switch 분기를 쌍으로 모은 학습 신호가 필요하다고 제안합니다. (p8–9)
- **후속 계획의 명시적 서술**: 별도 Future Work 절은 없습니다. 명시된 것은 Fable 5의 오염과 reward hacking 특성화를 향후 과제로 남긴다는 것(App E.3, p24)이 전부이고, 나머지는 위 처방의 제안입니다.

### 추가 후속 연구 방향 **[필자 제안]**
1. **결정 노드 프로토콜 ablation**: 강제 정당화(구조화된 continue/switch 이유), 대안 N개 열거 의무, 예산 인지 규칙을 비교합니다.
2. **fork 데이터 대규모화**: 자동으로 수백 개 체크포인트에서 다중 대안 분기를 생성하고, 보상 $R=V(\text{switch})-V(\text{continue})$ 같은 선호 데이터로 결정 정책을 학습합니다. (이 식은 제 표기이며 논문에 없음.) 분산이 크므로 반복과 CI가 필수입니다.
3. **외부 컨트롤러/포트폴리오**: 전략을 밴딧의 arm으로 보고 successive halving 등으로 초기에 다수 전략을 병렬 탐색합니다. 에이전트의 자체 전환과 비교합니다.
4. **컨텍스트 개입 대조**: 요약 후 리셋, 매몰비용이 없는 별도 검토 에이전트로 앵커링 가설을 직접 검증합니다.
5. **모델×scaffold 요인설계**로 락인의 원인을 분리합니다.
6. **통계 강화**: 반복 수 확대, 클러스터 부트스트랩, 사전등록된 분기점 선정으로 사후 선택 편향을 제거합니다.
7. **정의 민감도 분석**: 분류체계 변경 시 switch rate 변화를 봅니다.

### 8-1. 모델의 일반화 성능 향상 가능성

**[저자가 보고한 것]**
- 논문은 post-training된 모델의 일반화(OOD)를 직접 측정하지 않습니다. 모든 점수는 에이전트가 최적화 대상으로 삼은 벤치마크 점수입니다.
- Sec 5는 에이전트 수준의 일반화 문제를 언급합니다. "기본값이 우연히 과제에 맞는 에이전트는 과제가 바뀌면 실패할 수 있다"는 것입니다. 이것이 strategy-level 지표를 요구하는 이유 중 하나입니다. (p8)

**[필자 해석]**
- **(i) 에이전트 수준 일반화**: 기본 전략이 에이전트를 따라간다는 결과(Table 4)는 과제 적응적 전략 선택이 약하다는 뜻입니다. 만약 결정 노드와 fork 학습 신호가 작동한다면, 전략 선택 정책이 학습되어 새 과제로 전이될 가능성이 있습니다. 다만 이는 **미검증 가설**입니다. 전이를 보려면 학습에 쓰지 않은 과제와 base model에서 평가해야 합니다.
- **(ii) 모델 수준 일반화**: 가이드 분기가 쓴 RL(GRPO, 검증가능 보상)은 SFT보다 일반화에 유리하다는 문헌 주장이 있습니다(예: Chu et al., 2025 "SFT Memorizes, RL Generalizes"). 반면 RL이 base 능력을 넘어서는지에 대한 반론도 있습니다(Yue et al., 2025). 이 논문의 결과(SFT 정체 후 GRPO로 개선)는 이 논쟁과 방향은 맞지만, **일반화를 입증하지 않습니다**. 평가가 동일 벤치마크의 in-distribution이고 base model이 하나이기 때문입니다.
- **(iii) 위험 요인**: 포맷 교정만으로 base 점수가 오르는 경우가 많습니다(Fig 2에서 base가 0.0인 벤치마크가 3개). 개선의 상당 부분이 "출력 규약 학습"일 가능성이 있고, 이는 일반화 성능과 무관합니다. 또한 Fable 5의 오염과 평가 인터페이스 수정 사례(App E.3)는 점수 상승이 일반화와 무관할 수 있음을 보여줍니다.
- **(iv) 검증 제안**: 전환 전후 모델을 학습에 쓰지 않은 벤치마크(예: 다른 수학·코드·지시 따르기 평가)에서 평가하고, 능력 유지(망각)와 다양한 base model/크기에서 반복해야 합니다. 이렇게 해야 "전략 수정이 더 일반화되는 모델을 만든다"는 주장이 성립합니다.

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> 📘 **검증 범위**: 아래 비교는 (a) 이 논문의 참고문헌에 있는 문헌 중 제가 아는 것과, (b) 제 지식 범위의 공개 문헌을 기반으로 합니다. 2026년 문헌은 제목과 논문의 서술만 언급하고 내용은 검증하지 못했습니다. 세부 수치는 일부러 쓰지 않았습니다.

| 연구(연도) | 초점 | 본 논문과의 관계 |
|---|---|---|
| HumanEval — Chen et al. (2021) | 코드 생성 벤치마크, pass@k | 본 논문이 사용하는 평가와 지표 |
| GSM8K — Cobbe et al. (2021) | 수학 문장제 벤치마크 | 본 논문의 평가 과제 |
| Reflexion — Shinn et al. (2023) | 언어적 반성으로 시행착오 개선 | journal과 유사한 메커니즘. 본 논문은 이런 경험·반성이 **실행은 돕지만 전략은 안 바꾼다**고 보고 |
| Voyager — Wang et al. (2023) | 스킬 라이브러리 기반 체화 에이전트 | skill library의 선행 연구. 본 논문은 스킬 소비는 하지만 **생성은 0건**(App C.2)이라고 보고 |
| ExpeL — Zhao et al. (2024) | 경험 학습 에이전트 | 경험 기반 개선의 선행 연구 |
| Self-Refine (Madaan et al., 2023), "LLMs Cannot Self-Correct Reasoning Yet" (Huang et al., 2024) | 자기 수정의 한계 | 외부 신호 없는 자기 수정의 약함이라는 논의와 결이 같음. 본 논문은 장기 에이전트 설정에서 유사 현상을 관찰 |
| MLAgentBench (2024), MLE-bench (2025), PaperBench (2025) | 에이전트의 ML 엔지니어링/연구 능력 평가 | 최종 점수 중심. 본 논문은 **과정(전략 수정) 수준의 진단**으로 보완 |
| RE-Bench — Wijk et al. (2024) | 인간 전문가 vs 에이전트의 AI R&D 비교 | (제 기억 기준) 짧은 예산에서는 에이전트가 강하고 긴 예산에서는 인간이 시간 투자의 이득을 더 얻는다는 취지. 본 논문의 compute 천장 결과와 방향이 같음 |
| AIDE (2025), AI Scientist-v2 (2025) | 트리 탐색으로 코드·연구 해결책 탐색 | 명시적 분기 구조를 가진 시스템. 본 논문의 단선적 락인에 대한 구조적 대안 후보 |
| Darwin Gödel Machine (2025) | 아카이브 기반 자기수정 에이전트 | 이전 버전으로 돌아가 분기하는 구조. 본 논문의 fork 아이디어와 유사 |
| Snell et al. (2024), Brown et al. "Large Language Monkeys" (2024) | 추론 시점 compute 스케일링 | 과제 난이도별 compute 효과를 분석. 본 논문은 에이전트 토큰이 **전략 탐색이 아닌 국소 정련**에 쓰인다고 주장(단, 설정이 다름: 샘플링+검증 vs 에이전트 장기 로그) |
| DeepSeekMath/GRPO (2024), DeepSeek-R1 (2025), Tulu 3 (2024), DAPO (2025) | 검증가능 보상 RL 등 다단계 post-training 레시피 | 전략 공간 $\mathcal S$(SFT→DPO→RLVR 등)의 실제 후보. 가이드 분기가 쓴 GRPO/DAPO |
| Chu et al. (2025), Yue et al. (2025) | SFT vs RL의 일반화 논쟁 | 8-1 논의의 배경 |
| 논문 내 2026 문헌: PostTrainBench, AI4AI-Bench, RSIBench-Data, Meta-Agent Challenge, Bisht et al., Trehan & Chopra, Abbasi(블로그) | 에이전트 평가, RSI 벤치마크, 연구 에이전트의 한계 비판 | 논문은 이들을 "결과 점수를 측정하거나 결함을 목록화한다"고 서술하고, 자신은 "루프가 어디서 열려 있는지 위치를 짚는다"고 차별화(p9). **제가 내용을 검증하지 못함** |

**이 논문이 앞으로의 연구에 미칠 영향 [필자]**
- **평가 관행**: 최종 점수와 함께 switch rate, 제안 채택률, fork 기반 전략 격차를 보고하는 방향을 촉진할 수 있습니다.
- **에이전트 설계**: 장기 컨텍스트에서 "계속할지 전환할지"를 명시적 결정으로 분리하는 설계와, 분기 기반 학습 신호 연구를 촉진할 수 있습니다.
- **RSI 담론**: "더 많은 compute와 경험이 곧 자기개선"이라는 낙관을 약화시키고 전략 수준 능력을 분리해서 보게 만듭니다.

**앞으로 연구 시 고려할 점 [필자]**
1. **측정 타당성**: 전략의 정의(PEFT/full SFT, 데이터원)에 따라 switch rate가 달라지므로 분류체계를 사전에 고정·공개해야 합니다.
2. **"낮은 전환율 = 나쁨"이 아님**: 기본 전략이 최적인 과제에서는 낮은 전환이 정상일 수 있습니다. 전환의 후회(regret)를 함께 측정해야 합니다.
3. **사후 선택 편향 방지**: 분기점과 대안을 사전 규칙으로 정하고 다수 궤적에서 반복해야 합니다.
4. **교란 통제**: 모델, scaffold, 프롬프트, 추가 에이전트의 compute를 분리해야 합니다.
5. **오염과 보상 해킹 감시**: 점수 개선이 실제 능력인지 확인하기 위해 평가 인터페이스 무결성과 held-out 평가가 필요합니다.
6. **일반화 평가 포함**: in-distribution 점수 외에 OOD 평가와 능력 유지 평가를 넣어야 합니다.
7. **안전·감독**: 에이전트가 스스로 전략을 바꾸게 되면 감독 가능성과 비용·위험(자원 소모, 예기치 않은 행동)을 함께 고려해야 합니다.

---

## 참고자료 (출처)

**1차 근거 (제공 문서)**
- Lim, J. J. Y., Huang, X., Peng, H., Lu, Y., Cong, X., Zhang, Z., Sun, M., Lin, Y. "What is Missing from AI Post-Training AI: An Empirical Analysis." arXiv:2608.19072v2 (제공된 PDF, 2026-09-28 표기). 본 보고서의 모든 수치·쪽수·표/그림 번호의 근거입니다.

**제공 문서의 참고문헌 목록에서 직접 언급한 문헌**
- Chen et al., "Evaluating Large Language Models Trained on Code" (2021)
- Cobbe et al., "Training Verifiers to Solve Math Word Problems" (2021)
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023)
- Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models" (2023)
- Zhao et al., "ExpeL: LLM Agents Are Experiential Learners" (AAAI 2024)
- Huang et al., "MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation" (ICML 2024)
- Chan et al., "MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering" (ICLR 2025)
- Starace et al., "PaperBench: Evaluating AI's Ability to Replicate AI Research" (ICML 2025)
- Jiang et al., "AIDE: AI-Driven Exploration in the Space of Code" (2025)
- Yamada et al., "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search" (2025)
- Zhang et al., "Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents" (arXiv:2505.22954)
- Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025)
- Rank et al., "PostTrainBench: Can LLM Agents Automate LLM Post-Training?" (arXiv:2603.08640, 2026). 인용만 확인했고 내용은 미검증입니다.
- Abbasi, "What We Learned from Letting AI PostTrain AI" (Thoughtful 블로그, 2026). 미검증입니다.
- Chi et al., "AI4AI-Bench" (2026); Meng et al., "RSIBench-Data" (2026); Lu et al., "The Meta-Agent Challenge" (2026); Bisht et al., "Agentic AI Scientists Are Not Built for Autonomous Scientific Discovery" (2026); Trehan & Chopra, "Why LLMs Aren't Scientists Yet" (2026). 모두 논문의 서술에만 의존했고 미검증입니다.

**제 학습 지식에 기반한 보충 문헌** (웹 재확인은 하지 않았습니다. 인용 시 원문 확인을 권장합니다.)
- Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback" (2023)
- Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet" (ICLR 2024)
- Wijk et al., "RE-Bench: Evaluating Frontier AI R&D Capabilities of Language Model Agents Against Human Experts" (2024)
- Snell et al., "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" (2024)
- Brown et al., "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling" (2024)
- Shao et al., "DeepSeekMath" (GRPO 제안, 2024); DeepSeek-AI, "DeepSeek-R1" (2025)
- Lambert et al., "Tulu 3: Pushing Frontiers in Open Language Model Post-Training" (2024)
- Chu et al., "SFT Memorizes, RL Generalizes" (2025); Yue et al., "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" (2025)

**정확성 고지**
- 수치와 인용은 제공 PDF에서 확인한 것만 단정적으로 썼습니다.
- Welch 근사(t값), 표준편차 종류 추정, 수치 불일치 지적은 제가 PDF의 표 값으로 직접 계산한 것이며 원 논문의 보고가 아닙니다.
- 모델명(Opus 4.6, GLM-5.2, GPT-5.2, Fable 5 등)은 논문 표기를 그대로 옮겼고 제가 독립적으로 확인할 수 없습니다.
