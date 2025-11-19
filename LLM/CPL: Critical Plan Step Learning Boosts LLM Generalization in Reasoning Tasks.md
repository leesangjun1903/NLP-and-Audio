# CPL: Critical Plan Step Learning Boosts LLM Generalization in Reasoning Tasks

### 핵심 주장과 주요 기여

CPL(Critical Plan Step Learning) 논문의 **핵심 주장**은 기존의 강화학습(RL) 기반 추론 개선 방법들이 **과제 특화적 해결책(task-specific solutions)**에만 집중하여 다양한 추론 과제로의 일반화 성능을 제대로 달성하지 못한다는 것입니다. 저자들은 이 문제를 해결하기 위해 **추상적이고 고수준의 계획(abstract high-level plans)**을 학습하도록 제안합니다.[1]

논문의 **주요 기여**는 다음 세 가지입니다:[1]

1. 강화학습 스케일링 문제를 탐구하며, 과제 특화적 행동 공간이 아닌 **고수준 추상 계획을 기반으로 한 탐색**을 통해 모델 일반화를 향상시킬 수 있음을 제시합니다.

2. **Critical Plan Step Learning(CPL)**이라는 새로운 접근법을 도입하여, MCTS를 활용한 다양한 계획 단계 탐색과 Step-APO(Step-level Advantage Preference Optimization)를 통한 단계별 계획 선호도 학습을 결합합니다.

3. GSM8K와 MATH만으로 학습했음에도 불구하고, in-domain 과제에서 GSM8K(+10.5%), MATH(+6.5%) 성능 향상을 달성하고, 특히 out-of-domain 과제에서 HumanEval(+12.2%), GPQA(+8.6%), ARC-C(+4.0%), MMLU-STEM(+2.2%), BBH(+1.8%)의 의미 있는 성능 향상을 달성합니다.[1]

***

### 문제 정의 및 배경

#### 기존 방법의 한계

현재의 RL 기반 추론 개선 방법들은 세 가지 주요 문제점을 갖고 있습니다:[1]

1. **과제 특화성**: 기존 방법들(AlphaGo, AlphaMath 등)은 특정 과제나 도메인에 최적화되어 있어 다른 추론 과제로의 전이 학습이 어렵습니다.

2. **무한 행동 공간**: 전통적 RL과 달리 LLM은 무한에 가까운 행동 공간에서 작동하므로, 다양하고 질 높은 추론 경로를 효율적으로 탐색하기 어렵습니다.

3. **세밀한 감독 부족**: 기존 DPO(Direct Preference Optimization)는 문장 수준의 선호도만 학습하여 다단계 추론 과제의 중간 단계 오류를 효과적으로 수정하지 못합니다.[1]

#### 핵심 통찰

논문의 핵심 통찰은 **계획과 해결책의 역할 구분**입니다:[1]

- **과제 특화적 해결책** (예: 수식, 코드): 특정 과제의 구체적인 기술에 의존
- **추상적 계획** (예: 어떤 지식을 적용할지, 문제를 어떻게 분해할지): 과제 불변적(task-agnostic) 문제 해결 전략

저자들은 추상적 계획을 학습하는 것이 더 나은 일반화를 가능하게 한다고 주장합니다.

---

### 제안 방법 상세 분석

#### 1) Plan-based MCTS (Monte Carlo Tree Search)

**기본 구조**: 각 노드는 상태 $$s_t$$를 나타내고, 각 간선은 행동 $$a_t$$를 나타내는 트리를 구성합니다.[1]

**MCTS의 네 가지 주요 연산**:

**선택(Selection)**: PUCT 알고리즘을 사용하여 탐색할 노드를 선택합니다:

$$\arg \max_{a_t} \left[ Q(s_t, a_t) + c_{puct}\pi_\theta(a_t|s_t) \sqrt{\frac{N(s_t)}{1 + N(s_t, a_t)}} \right]$$

여기서 $$Q(s_t, a_t)$$는 상태-행동 쌍의 누적 보상, $$\pi_\theta(a_t|s_t)$$는 정책 모델의 확률, $$N$$은 방문 횟수입니다.[1]

**확장 및 평가**: 여러 후보 행동을 샘플링하고, 종료 노드는 정답과 비교하여 평가하며, 다른 노드는 가치 모델(value model)로 예측합니다.[1]

**백업(Backup)**: 트리의 루트로 역전파하면서 상태 가치, Q값, 방문 횟수를 업데이트합니다:

$$Q(s_t, a_t) \leftarrow r(s_t, a_t) + V(s_{t+1})$$

$$V(s_t) \leftarrow \frac{\sum_a N(s_{t+1})Q(s_t, a_t)}{\sum_a N(s_{t+1})}$$

$$N(s_t) \leftarrow N(s_t) + 1$$

#### 2) Step-APO (Step-level Advantage Preference Optimization)

**배경**: 기존 Step-level DPO는 첫 번째 오류 단계만 식별하는 휴리스틱에 의존하여 광범위한 탐색 공간을 충분히 활용하지 못합니다.[1]

**Step-APO의 도출**:

최대 엔트로피 RL 설정에서 최적 정책은:[1]

```math
\pi^*(a_t|s_t) = e^{(Q^*(s_t,a_t)-V^*(s_t))/\beta}
```

Bellman 방정식을 적용하면:[1]

```math
Q^*(s_t, a_t) = r(s_t, a_t) + \beta \log \pi_{ref}(a_t|s_t) + V^*(s_{t+1})
```

이를 정리하면 **최적 이득 함수(advantage function)**을 얻습니다:[1]

```math
\beta \log \frac{\pi^*(a_t|s_t)}{\pi_{ref}(a_t|s_t)} = r(s_t, a_t) + V^*(s_{t+1}) - V^*(s_t)
```

**Step-level Bradley-Terry 모델**을 적용하면:[1]

$$p^*(a_t^w \succ a_t^l|s_t) = \frac{\exp(r(s_t, a_t^w))}{\exp(r(s_t, a_t^w)) + \exp(r(s_t, a_t^l))}$$

**Step-APO 손실함수**:

$$L_{Step-APO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(s_t, a_t^w, a_t^l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(a_t^w|s_t)}{\pi_{ref}(a_t^w|s_t)} - V(s_{t+1}^w) - \beta \log \frac{\pi_\theta(a_t^l|s_t)}{\pi_{ref}(a_t^l|s_t)} + V(s_{t+1}^l) \right) \right]$$

여기서 $$\sigma$$는 로지스틱 함수입니다.[1]

**그래디언트 분석**:

$$\nabla_\theta L_{Step-APO} = -\beta \mathbb{E}_{(s_t, a_t^w, a_t^l) \sim D} \left[ \sigma(\hat{r}_\theta(s_t, a_t^l) - \hat{r}_\theta(s_t, a_t^w) + V(s_{t+1}^w) - V(s_{t+1}^l)) \right] \left[ \nabla_\theta \log \pi(a_t^w|s_t) - \nabla_\theta \log \pi(a_t^l|s_t) \right]$$

여기서 $$\hat{r}\_\theta(s_t, a_t) = \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$$입니다.[1]

**핵심 차별점**: Step-APO는 단순히 선호도 쌍의 선호도만 고려하는 것이 아니라, **이득 차이(advantage difference)** $$V(s_{t+1}^w) - V(s_{t+1}^l)$$에 의해 가중치가 적용됩니다. 이는 중요한 단계(critical step)를 강조하고 최적화 가중치를 동적으로 할당합니다.[1]

#### 3) 정책 및 가치 모델의 반복 학습

**정책 모델 훈련**:[1]
1. MCTS에서 수집한 올바른 경로로 지도 학습(SFT) 수행
2. MCTS에서 얻은 단계별 선호도 데이터로 Step-APO 적용

**가치 모델 최적화**:[1]
- MCTS의 상태 가치 $$V$$를 레이블로 하여 MSE 손실 사용
- 이득 함수와의 관계 활용:

$$A(s_t, a_t^w) - A(s_t, a_t^l) = Q(s_t, a_t^w) - V(s_t) - (Q(s_t, a_t^l) - V(s_t)) = V(s_{t+1}^w) - V(s_{t+1}^l)$$

---

### 성능 향상 분석

#### In-Domain 성능

| 모델 | MATH | GSM8K |
|------|------|-------|
| DeepseekMath-Base | 35.18 | 63.23 |
| Self-Explore-MATH | 37.86 | 78.39* |
| **CPL-final** | **41.64** | **73.77** |

- MATH에서 **+6.46%** 향상 (Base 대비 +18.2%)
- GSM8K에서 **+10.54%** 향상 (Base 대비 +16.7%)[1]

#### Out-of-Domain 일반화 성능

| 벤치마크 | Base | CPL-final | 향상도 |
|---------|------|-----------|--------|
| HumanEval | 40.90 | 53.05 | **+12.15%** |
| GPQA | 25.75 | 34.34 | **+8.59%** |
| ARC-C | 52.05 | 56.06 | **+4.01%** |
| BBH | 58.79 | 60.54 | **+1.75%** |
| MMLU-STEM | 52.74 | 54.93 | **+2.19%** |

평균 **+5.7%** 향상으로 OOD 일반화 능력을 확인할 수 있습니다.[1]

#### 반복 학습의 효과

| 모델 단계 | MATH | GSM8K | HumanEval |
|----------|------|-------|-----------|
| Round 1 SFT | 36.30 | 63.79 | 42.68 |
| Round 1 Step-APO | 40.56 | 71.06 | 46.34 |
| Round 2 SFT | 39.16 | 69.75 | 48.78 |
| **Round 2 Step-APO** | **41.64** | **73.77** | **53.05** |

각 라운드에서 Step-APO는 SFT 대비 일관되게 3-4% 향상을 제공합니다.[1]

***

### 모델 일반화 성능 향상의 메커니즘

#### 1) 계획 기반 학습의 장점

실험 결과 BBH 벤치마크에서:[1]

| 방법 | BBH 성능 |
|------|---------|
| DeepSeekMath-Base | 58.79 |
| Solution-based SFT | 58.92 |
| **Plan-based SFT** | **59.50** |

계획 기반 학습이 해결책 기반 학습보다 일반화 성능을 더 향상시킵니다.

**이유**: 
- 계획은 **과제 불변적** 문제 해결 전략을 표현
- 다양한 도메인의 문제에서 재사용 가능한 고수준 개념 학습
- 구체적인 수식이나 코드에 과적합되지 않음[1]

#### 2) Step-APO의 세밀한 감독

| 방법 | MATH | GSM8K | HumanEval | GPQA |
|------|------|-------|-----------|------|
| SFT | 36.30 | 63.79 | 42.68 | 28.78 |
| Instance-DPO | 37.72 | 69.29 | 43.90 | 24.24 |
| Step-DPO | 37.89 | 69.83 | 42.68 | 25.25 |
| **Step-APO** | **40.56** | **71.06** | **48.78** | **31.31** |

Step-APO는 다른 방법들에 비해 OOD 과제에서 **6-7% 향상**을 달성합니다.[1]

**메커니즘**:
- 이득 추정치를 사용하여 **중요한 단계를 자동으로 식별**
- 일반적인 오류 단계 식별 휴리스틱의 한계 극복
- 광범위한 탐색 공간 활용으로 더 효과적인 학습[1]

#### 3) 데이터 구성의 최적화

![](chart이터와 해결책 단계 데이터의 비율이 성능에 미치는 영향:[1]

- **모든 계획 쌍 + 단일 해결책 쌍** 조합이 최적
- 과도한 해결책 데이터는 OOD 성능을 저하시킴
- 이는 계획 학습이 일반화에 더 중요함을 시사[1]

***

### 한계 및 고려사항

#### 1) 계산 비용

- Round 1에서 5k 샘플에 대해 **200번의 MCTS 시뮬레이션** 필요
- Round 2에서 15k 샘플에 대해 **100번의 MCTS 시뮬레이션** 필요
- 8개 H100 GPU에서 상당한 훈련 시간 필요[1]

#### 2) 가치 모델 초기화

- 첫 번째 라운드의 **무작위 가치 모델 초기화**가 잠재적으로 성능에 영향
- MCTS 단계에서 종료 노드의 보상이 역전파되며 완화되지만, 여전히 최적과는 거리가 있음[1]

#### 3) 계획의 단순성

- 현재 계획 전략은 "다음 추론 단계로 계속"이라는 기본 옵션만 포함
- 자기 수정, 새로운 아이디어 탐색 등 더 복잡한 계획 전략은 미실장[1]

#### 4) 평가 범위

- 주로 수학, 코딩, 상식 추론 중심의 벤치마크
- 더 광범위한 추론 도메인에 대한 일반화 검증 필요[1]

***

### 앞으로의 연구 방향 및 시사점

#### 1) 최신 연구 기반의 관점

##### 테스트 타임 계산 스케일링의 통합[2][3][4][5]

최신 연구는 테스트 타임에 더 많은 계산을 할당하는 것이 훈련 파라미터 스케일링만큼 효과적일 수 있음을 보여줍니다. CPL과 이를 결합하면:[6]

- **계산-최적 스케일링 전략**: 문제의 난이도에 따라 동적으로 테스트 타임 계산을 할당
- **프로세스 검증 모델(Process Reward Model)** 개발로 MCTS 효율성 향상 가능
- 더 작은 모델도 테스트 타임 계산을 통해 14배 큰 모델을 초과할 수 있음[5]

##### 프로세스 vs 아웃컴 감독[7][8]

최신 이론 연구에 따르면 프로세스 감독과 아웃컴 감독 간의 통계적 어려움이 비슷할 수 있습니다:[7]

- CPL의 **Step-APO는 이득 함수를 최적 프로세스 보상 모델로 사용** 가능
- 이는 이론적으로 근거 있는 접근법
- 앞으로 더 효율적인 알고리즘 설계로 성능 격차 해소 가능[7]

#### 2) 고려할 연구 방향

##### A. 고급 계획 전략의 탐색[1]

논문에서 명시한 바와 같이, 다음과 같은 복잡한 계획 전략 개발이 필요합니다:[1]

- **자기 수정 계획**: 잘못된 방향을 감지하고 대체 접근법으로 전환
- **다중 관점 탐색**: 여러 문제 해석 각도에서 동시 탐색
- **메타-계획**: 어떤 계획 전략이 효과적인지 학습

##### B. 효율성 개선[1]

- **더 효율적인 값 함수 추정**: 초기 무작위 초기화의 영향 감소
- **적응형 MCTS 시뮬레이션 수**: 문제 난이도에 따른 동적 조정
- **분산 학습 최적화**: 계산 비용 감소[1]

##### C. 통합 학습 패러다임[9][10]

최신 연구는 다양한 도메인에서의 RL 기반 추론 개선을 탐색하고 있습니다:[10][9]

- **도메인 간 전이**: 수학 추론에서 배운 일반화 능력을 다른 영역에 적용
- **크로스태스크 노출**: CPL 방식으로 다양한 추론 과제에 동시 노출
- **신흥 능력**: 일반화된 추론 메커니즘의 창발적 능력 연구[10]

##### D. 이론적 기초 강화[8][7]

- **프로세스 감독의 최적화**: 이론적으로 근거 있는 프로세스 보상 모델 설계
- **이득 함수와 정책 관계**: 더 깊은 수학적 분석을 통한 알고리즘 개선
- **수렴 성질 분석**: 반복 학습의 이론적 보장[8]

##### E. 실용적 응용 확대

- **장문맥 추론**: 더 길고 복잡한 추론 과제에서의 성능 평가
- **외부 도구 통합**: API 호출, 코드 실행 등 외부 도구와의 상호작용
- **멀티모달 추론**: 시각 정보를 포함한 추론 과제[9]

***

### 결론

CPL은 **추상적 계획 학습**과 **세밀한 이득 기반 감독**을 통해 LLM의 추론 일반화 문제를 해결하는 혁신적 접근법을 제시합니다. 특히 **정렬되지 않은 강화학습의 스케일링 문제**를 다루면서, 기존의 과제 특화적 방법의 한계를 극복합니다.[1]

논문의 **핵심 기여**는:

1. **계획 vs 해결책의 명확한 구분**: 일반화 성능 향상의 핵심이 고수준 추상 개념 학습임을 입증
2. **이론적으로 근거 있는 Step-APO**: 이득 함수 기반 최적 프로세스 감독의 실현
3. **실증적 성공**: 제한된 학습 데이터로 광범위한 일반화 달성[1]

앞으로의 연구는 **테스트 타임 계산 스케일링**, **프로세스 감독의 이론적 최적화**, **고급 계획 전략의 탐색**을 통해 CPL의 성능을 더욱 향상시킬 수 있을 것입니다.[5][9][7]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fdffd428-4850-46e9-b55d-644a415f2a6d/2409.08642v2.pdf)
[2](http://arxiv.org/pdf/2501.19306.pdf)
[3](http://arxiv.org/pdf/2411.19477.pdf)
[4](https://arxiv.org/pdf/2408.03314.pdf)
[5](https://chatpaper.com/paper/111311)
[6](https://openreview.net/attachment?id=4FWAwZtd2n&name=pdf)
[7](https://arxiv.org/html/2502.10581v1)
[8](https://openreview.net/forum?id=4BfaPHfhJ0&noteId=xuhztU8gs6)
[9](https://arxiv.org/pdf/2502.03671.pdf)
[10](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
[11](https://arxiv.org/pdf/2409.08642.pdf)
[12](https://arxiv.org/pdf/2406.14283.pdf)
[13](http://arxiv.org/pdf/2501.18817.pdf)
[14](https://arxiv.org/pdf/2402.18252.pdf)
[15](http://arxiv.org/pdf/2308.10379.pdf)
[16](http://arxiv.org/pdf/2407.14562.pdf)
[17](https://arxiv.org/html/2503.22732v1)
[18](https://openreview.net/forum?id=r1KcapkzCt)
[19](https://arxiv.org/html/2502.14356v1)
[20](https://aclanthology.org/2025.findings-emnlp.453.pdf)
[21](https://proceedings.iclr.cc/paper_files/paper/2025/file/837ff662214b04e7ea8c43f095fe0dd7-Paper-Conference.pdf)
[22](https://openreview.net/pdf?id=uy39I0axE8)
[23](https://www.ijcai.org/proceedings/2025/1198.pdf)
[24](https://llm-mcts.github.io)
[25](https://www.youtube.com/watch?v=y8uZ-oFjSOs)
[26](https://arxiv.org/html/2508.08636v1)
[27](https://arxiv.org/pdf/2501.14723.pdf)
[28](https://arxiv.org/pdf/2502.09922.pdf)
[29](https://arxiv.org/html/2503.23803v2)
[30](https://arxiv.org/pdf/2502.13575.pdf)
[31](http://arxiv.org/pdf/2412.15287.pdf)
[32](https://scale.com/blog/first-impression-openai-o1)
[33](https://www.toolify.ai/ai-news/revolutionizing-ai-training-the-power-of-process-supervision-969290)
[34](https://huggingface.co/blog/Kseniase/testtimecompute)
[35](https://arxiv.org/abs/2408.03314)
[36](https://blog.iese.edu/artificial-intelligence-management/2024/chain-of-thought-reasoning-the-new-llm-breakthrough/)
[37](https://arxiv.org/html/2408.03314v1)
[38](https://openreview.net/forum?id=BH0Rf3niSQ)
