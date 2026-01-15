
# o1-Coder: an o1 Replication for Coding

## 1. 핵심 주장 및 기여 요약

**o1-Coder**는 OpenAI의 o1 모델을 코딩 태스크에 특화하여 복제하려는 시도로, 강화학습(RL)과 몬테카를로 트리 탐색(MCTS)을 결합하여 **System-2 사고 능력**을 강화하는 혁신적 접근법을 제시합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

논문의 **세 가지 핵심 기여**는 다음과 같습니다:

1. **Test Case Generator (TCG)**: 문제 및 솔루션 코드로부터 자동으로 테스트 케이스를 생성하여 RL의 결과 리워드 신호 제공 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)
2. **Pseudocode-기반 MCTS**: 의사코드를 중간 표현으로 활용하여 추론 과정을 더 효과적으로 정의 및 제어 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)
3. **Self-Play+RL 반복 프레임워크**: 여섯 단계의 자기 놀이 루프로 모델을 지속적으로 개선 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

***

## 2. 해결하고자 하는 핵심 문제들

### 문제 1: 코드 생성 평가의 근본적 어려움

일반적인 게임(바둑)이나 수학 문제와 달리, 코드의 정확성을 평가하려면:
- 코드를 실제로 실행해야 함
- 테스트 케이스 통과 여부를 검증해야 함
- 하지만 대부분의 코드 데이터셋이 충분한 테스트 케이스를 제공하지 않음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

**해결책**: TCG를 훈련하여 문제와 정답 코드로부터 자동으로 고품질 테스트 케이스 생성

### 문제 2: 코드 생성에서 사고 과정 정의의 모호성

코드 생성을 위한 RL을 적용하려면:
- 상태 전이(state transition)와 행동(action)을 명확히 정의해야 함
- 프로세스 리워드의 세분화 수준을 결정해야 함
- 그래야만 RL이 효과적으로 작동 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

**두 가지 후보 접근 방식**:
- "생각 후 행동": 먼저 전체 체인-오브-생각을 생성한 후 답변을 내놓음
- "행동하며 생각": 점진적으로 답변을 생성하면서 동시에 추론 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

**선택된 해결책**: "생각 후 행동" 패러다임 → 먼저 의사코드 생성 후 최종 코드 생성

### 문제 3: 추론 과정 데이터의 극심한 부족

- 공개 데이터는 거의 모두 (질문, 답변) 형태로만 존재
- 중간 추론 단계(reasoning steps) $S_i$가 기록되어 있지 않음
- 이를 수동으로 주석 처리하는 것은 매우 비용이 높고, 특히 복잡한 태스크에서는 더욱 그러움 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

**선택된 해결책**: 강화학습을 활용하여 **자동으로 추론 데이터를 합성** - 이를 통해 높은 비용의 인간 주석 없이도 가능

***

## 3. 제안하는 방법론 상세 분석

### 3.1 Test Case Generator (TCG) 훈련

TCG는 두 단계의 훈련으로 구성됩니다:

#### **단계 1: Supervised Fine-Tuning (SFT)**

기본 모델(DeepSeek-1.3B-Instruct)을 TACO 데이터셋(~10,000개 샘플)으로 미세조정:

```
[SFT 목표]
- 생성된 테스트 케이스가 정해진 형식을 따르도록 학습
- 파싱 및 추출이 정확하게 되도록 보장
```

**결과**: γ $^{\text{sft}}_{\text{TCG}}$ 로 TACO 테스트 데이터셋에서 **80.8% 통과율** 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

#### **단계 2: Direct Preference Optimization (DPO)**

DPO는 선호도 기반 최적화로, 다음의 목적함수를 최소화합니다:

$$L_{\text{DPO}}(\gamma_{\text{TCG}}; \gamma_{\text{TCG}}^{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D_{\text{pref}}} \left[ \log \sigma \left( \beta \log \frac{\gamma_{\text{TCG}}(y_w|x)}{\gamma_{\text{TCG}}^{\text{ref}}(y_w|x)} - \beta \log \frac{\gamma_{\text{TCG}}(y_l|x)}{\gamma_{\text{TCG}}^{\text{ref}}(y_l|x)} \right) \right]$$

여기서:
- $\sigma(x)$: 시그모이드 함수
- $\beta$: 대비 강도 조절 스케일 인수
- $y_w$: 양성 예시 (완벽하게 일치하는 테스트 케이스)
- $y_l$: 음성 예시 (출력을 무작위로 섞은 테스트 케이스 쌍) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

**결과**: γ $^{\text{dpo}}_{\text{TCG}}$ 로 **89.2% 통과율** 달성 (**+8.4%p 향상**) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

**의의**: 선호도 최적화가 테스트 케이스 생성기의 신뢰도를 상당히 향상시킴을 입증

***

### 3.2 의사코드 기반 MCTS 추론 과정 데이터 합성

#### **핵심 개념: 세 가지 행동(Actions)**

논문은 의사코드 정제를 세 가지 명확한 행동으로 구조화합니다:

**[Action 1] 의사코드로 알고리즘 구조 정의**
- 목표: 전체 솔루션의 높은 수준의 구조를 정의
- 포함 내용: 입력, 출력, 각 주요 함수의 핵심 기능
- 제외 내용: 구현 세부사항은 다루지 않음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

**[Action 2] 의사코드 반복 정제**
- 목표: Action 1에서 정의한 의사코드를 점진적으로 상세화
- 추가 내용: 각 함수의 정확한 단계, 로직, 연산 구체화
- 준비 단계: 최종 코드 생성을 위한 준비 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

**[Action 3] 의사코드에서 Python 코드 생성**
- 목표: 정제된 의사코드를 실행 가능한 Python 코드로 변환
- 보장: 입출력 처리 및 요구사항 준수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

#### **의사코드 기반 접근의 두 가지 장점**

1. **적응성(Adaptability)**: 같은 의사코드가 서로 다른 구체적 구현으로 이어질 수 있음
   - 언어별 특성이나 라이브러리 선택에 따른 변형 가능

2. **제어 가능한 세분화(Controllable Granularity)**: 의사코드의 상세 수준을 조절함으로써 추론/탐색 행동의 세분화 조절
   - 더 추상적인 의사코드 → 더 큰 탐색 공간
   - 더 구체적인 의사코드 → 더 정밀한 추론 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

#### **MBPP 벤치마크 실험 결과**

| 모델 | Pass@1(%) | ASPR(%) | 변화 |
|------|----------|---------|------|
| **Qwen2.5-1.5B** (vanilla) | 55.8 | 49.9 | - |
| **Qwen2.5-1.5B** (pseudocode) | 46.7 | 54.5 | **ASPR +4.6%p** |
| **Qwen2.5-Coder-7B** (vanilla) | 57.7 | 49.3 | - |
| **Qwen2.5-Coder-7B** (pseudocode) | 58.2 | 74.9 | **ASPR +25.6%p** |

**핵심 발견**: Pass@1은 감소하지만 **ASPR(Average Sampling Pass Rate)이 대폭 증가** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)
- 이는 의사코드가 **올바른 추론 경로에 도달할 가능성을 크게 향상**시킨다는 의미
- VanillaLLM들은 정확한 의사코드를 생성하는 데 어려움을 겪음
- 하지만 이것이 후속 SFT 초기화와 Self-Play+RL 개선의 목표

#### **MCTS를 통한 데이터 합성**

MCTS로 $D_{\text{process}} = \{(Q_i, \cdots, S^j_i, v^j_i, \cdots, C'_i)\}$ 구성:

- $S^j_i$: j번째 추론 단계
- $v^j_i$: 해당 단계의 평가 값
- $C'_i$: 최종 단계에서 생성된 실행 가능 코드

**리워드 값 계산** (터미널 노드 $S^m_i$):

$$v^m_i = \alpha \cdot \text{compile} + (1-\alpha) \cdot \text{pass}$$

여기서:
- **compile**: 컴파일 성공 여부 (0 또는 1)
- **pass**: 테스트 케이스 통과율 = $\frac{\text{Num}\_{\text{passed}}}{\text{Num}_{\text{test case}}}$
- **α**: 컴파일과 테스트 통과의 상대적 중요도 조절 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

**정확한 솔루션 경로 선택**:
검색이 완료되면 정답을 찾은 터미널 노드($v^m_i = 1$)로부터 전체 추론 경로를 선택하여 정책 모델 초기화 데이터로 활용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

***

### 3.3 정책 모델 초기화 (Policy Model Initialization)

MCTS에서 합성된 추론 데이터 $D^+\_{\text{process}}$ 를 이용하여 정책 모델 $\pi_\theta$ 를 초기화합니다:

$$L_{\text{SFT}} = -\sum_{(Q_i, S^j_i, C'_i) \in D^+_{\text{process}}} \log \pi_\theta(S^{1:m}_i \circ C'_i | Q_i)$$

여기서:
- $S^{1:m}_i \circ C'_i$: 모든 추론 단계와 최종 코드의 연결(concatenation)
- 훈련 데이터는 **검증된 올바른 솔루션**만 포함 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

**의의**: 이 단계는 정책 모델에게 기대되는 "생각 후 행동" 방식을 일관되게 보여줌으로써, 후속 RL 훈련을 위한 최적의 출발점 제공

***

### 3.4 프로세스 리워드 모델 (PRM) 훈련

PRM은 주어진 문제와 현재까지의 추론 경로에서 다음 단계의 품질을 평가합니다.

**두 가지 훈련 방식**:

#### **Point-wise 방식**

$$L^{\text{point-wise}}_{\text{PRM}} = -\mathbb{E}_{(Q_i, S^{1:j-1}_i, S^j_i, v^j_i) \sim D} \left[ v^j_i \log r(Q_i, S^{1:j}_i) + (1-v^j_i) \log(1 - r(Q_i, S^{1:j}_i)) \right]$$

- 각 단계에 **절대적 정확도 점수** 부여
- $v^j_i$: MCTS에서 얻은 단계별 검증 레이블 (0 또는 1) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

#### **Pair-wise 방식 (Bradley-Terry 모델)**

$$L^{\text{pair-wise}}_{\text{PRM}} = -\mathbb{E}_{(Q_i, S^{1:j-1}_i, S^{j_{\text{win}}}_i, S^{j_{\text{lose}}}_i) \sim D_{\text{pair}}} \left[ \log\left( \sigma \left( r(Q_i, S^{1:j-1}_i, S^{j_{\text{win}}}_i) - r(Q_i, S^{1:j-1}_i, S^{j_{\text{lose}}}_i) \right) \right) \right]$$

- 두 단계 간의 **상대적 선호도** 학습
- $\sigma(x)$: 시그모이드 함수
- 정규화되지 않은 점수로 상대 선호도에 집중 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

**선택의 의미**: Point-wise는 절대적 정확도, Pair-wise는 상대적 품질 판단에 강점

***

### 3.5 강화학습으로 정책 모델 개선

코드 생성을 **언어 증강 MDP** $M = (V, S, A, T, R, \phi)$로 모델링:

- **V**: 어휘(vocabulary)
- **S**: 상태 공간 (토큰 시퀀스) - $s_0$는 질문
- **A**: 행동 공간 (토큰 시퀀스 집합) - 각 $a_i$는 추론 단계
- **T**: 상태 전이 함수 - $s_{t+1} = T(s_t, a_t)$ (행동의 토큰 추가)
- **R**: 리워드 함수 - 중간 단계의 품질 평가
- **φ**: 프로세스 기반과 결과 기반 리워드 결합 함수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

#### **리워드 결합 함수 (Reward Aggregation)**

$$\phi(R_i, r^{1:m}_i) = \alpha(t) \cdot R_i + (1-\alpha(t)) \cdot \frac{1}{m} \sum^m_{j=1} \gamma^j r^j_i$$

- **$R_i$**: 최종 리워드 (테스트 케이스 통과 여부)
- **$r^{1:m}_i$**: 각 단계의 프로세스 리워드
- **$\alpha(t)$**: 시간에 따라 변하는 가중치
  - 초기: 최종 리워드를 더 강조
  - 후기: 중간 리워드를 더 강조 (선형 또는 로그 감소 스케줄) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)
- **γ ∈ **: 할인 인수 (미래 리워드의 중요도) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

**훈련 절차**:
1. 각 단계에서 정책 모델이 추론 단계 생성: $S^j_i \sim \pi_\theta(S^j_i | Q_i, S^{1:j-1}_i)$
2. PRM에서 프로세스 리워드 계산: $r^j_i = \rho_{\text{PRM}}(Q_i, S^{1:j}_i)$
3. 최종 코드에 대해 TCG로 테스트 케이스 생성 및 실행
4. 결과 리워드 계산: $R_i = \tau_{\text{pass}}$ (성공) 또는 $\tau_{\text{fail}}$ (실패)
5. 집계된 리워드 $\phi(R_i, r^{1:m}_i)$로 정책 업데이트 (PPO 또는 Iterative DPO) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

***

### 3.6 새로운 추론 데이터 생성 및 자기 놀이 (Self-Play)

개선된 정책 모델 $\pi_\theta$를 사용하여 새로운 추론 데이터 $D'_{\text{process}}$ 생성:

$$D_{\text{process}} \leftarrow D_{\text{process}} \cup D'_{\text{process}}$$

이렇게 확장된 데이터셋을 다시 **4단계(PRM 미세조정)로 돌아가** 반복:

$$\text{4번 단계} \rightarrow \text{5번 단계} \rightarrow \text{6번 단계} \rightarrow \text{4번 단계} \rightarrow \cdots$$

**자기 놀이의 의미**:
- 데이터의 **다양성과 품질 지속적 향상**
- 정책과 리워드 모델의 **공동 진화**
- 모델이 **스스로 학습 신호를 생성**하는 자기 지도 학습 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

***

## 4. 모델 구조 및 6단계 프레임워크

### 알고리즘 1: Self-Play+RL 기반 코더 훈련 프레임워크

```
입력: D_code (문제 Q_i와 솔루션 코드 C_i)
      π_θ (초기 정책 모델)
      γ_TCG (테스트 케이스 생성기)
      ρ_PRM (프로세스 리워드 모델)
      φ (리워드 결합 함수)

출력: π*_θ (최적화된 정책 모델)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① TCG 훈련
  γ_TCG를 D_code로 미세조정하여 다양하고 정확한 테스트 케이스 생성

② 추론 과정 데이터 합성
  MCTS를 사용하여 D_code로부터 추론 과정 포함 데이터셋 생성:
  D_process = {(Q_i, ···, S_j_i, v_j_i, ···, C'_i) | j = 1, ···, m}
  
  여기서 v_j_i ∈ {0, 1}은 검증 지표이고,
  v_m_i = 1 (최종 코드가 테스트 통과)

③ 정책 모델 초기화
  π_θ를 SFT로 D+_process (올바른 단계만)에서 훈련:
  L_SFT = -Σ log π_θ(S^{1:m}_i ◦ C'_i | Q_i)

④ PRM 초기화/미세조정
  반복 루프:
    Point-wise 손실 또는 Pair-wise 손실(DPO)로 PRM 훈련

⑤ RL로 정책 모델 개선
  반복 루프:
    a) 각 단계별로 추론 생성: S_j_i ~ π_θ(···)
    b) PRM으로 프로세스 리워드 계산: r_j_i = ρ_PRM(···)
    c) TCG로 테스트 케이스 생성하여 코드 실행
    d) 결과 리워드 계산: R_i
    e) 집계된 리워드 φ(R_i, r^{1:m}_i)로 π_θ 업데이트

⑥ 새로운 추론 데이터 생성
  업데이트된 π_θ로 새로운 추론 데이터 생성
  D_process ← D_process ∪ D'_process
  
  → ④번 단계로 반복

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

반환: 최적화된 정책 모델 π*_θ
```

**핵심 특징**:
- **모듈식 설계**: TCG, PRM, π_θ가 독립적으로 개선 가능
- **자기 강화 루프**: 4→5→6→4로의 순환 구조로 지속적 개선
- **다중 리워드 신호**: 프로세스 기반(중간)과 결과 기반(최종) 리워드의 조화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

***

## 5. 성능 향상 및 실험 결과

### 5.1 Test Case Generator 성능

| 단계 | 모델 | TACO 테스트 세트 통과율 |
|------|------|---------------------|
| SFT 후 | γ $^{\text{sft}}_{\text{TCG}}$ | **80.8%** |
| DPO 후 | γ $^{\text{dpo}}_{\text{TCG}}$ | **89.2%** (+8.4%p) |

**의의**: DPO 단계가 선호도 기반 최적화로 테스트 케이스 신뢰도를 상당히 향상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

### 5.2 의사코드 기반 MCTS의 효과

코딩 능력 평가 메트릭:
- **Pass@1**: 첫 시도에서 통과 비율
- **ASPR** (Average Sampling Pass Rate): 올바른 추론 경로에 도달한 마지막 단계의 평균 통과율

| 모델 | Vanilla Pass@1 | Pseudocode Pass@1 | Vanilla ASPR | Pseudocode ASPR | ASPR 향상 |
|------|---|---|---|---|---|
| Qwen2.5-1.5B | 55.8% | 46.7% | 49.9% | 54.5% | **+4.6%p** |
| Qwen2.5-3B | 56.3% | 51.3% | 52.0% | 70.6% | **+18.6%p** |
| Qwen2.5-7B | 59.8% | 50.1% | 66.4% | 78.1% | **+11.7%p** |
| Qwen2.5-Coder-7B | 57.7% | 58.2% | 49.3% | 74.9% | **+25.6%p** |

**핵심 통찰**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)
- Pass@1의 감소는 의사코드 생성 단계의 추가로 인한 자연스러운 결과
- **ASPR의 대폭 증가가 더 중요한 지표**: 올바른 추론 경로에 도달할 확률이 크게 향상
- Qwen2.5-Coder-7B에서의 +25.6%p는 **의사코드 기반 추론이 고급 모델일수록 더 효과적**임을 시사

***

## 6. 현재 한계 및 도전 과제

### 6.1 기술적 한계

#### **1. 완전한 성능 평가 데이터 부재**
- 논문은 "기술 보고서"로, 최종 버전에서 업데이트된 실험 결과가 나올 예정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)
- 현재는 주요 컴포넌트(TCG, MCTS)의 성능만 제시

#### **2. 테스트 시간 계산 비용 증가**
- MCTS 기반 탐색으로 인한 높은 추론 시간
- 의사코드 단계 추가로 인한 토큰 수 증가
- **전개 시간(deployment)과 효율성 간의 트레이드오프**

#### **3. 환경 모델 부재 - 실제 세계 적용의 핵심 장애**
논문은 중요한 도전 과제로 **"세계 모델 인코딩"**을 강조: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

- **문제**: o1 유사 모델은 행동 결과를 직접 시뮬레이션해야 함
- **바둑이나 수학**: 규칙이 명확하고 상태 전이가 결정적(deterministic)
  - 바둑: 게임 규칙으로 상태 업데이트 완벽 정의
  - 수학: LLM이 공리와 논리 내재
  - 프로그래밍: 프로그래밍 문법과 의미론 내재
- **현실 응용(기기 사용, 로봇공학)**: 불확실성 높고 외부 시뮬레이터/환경 상호작용 필요
  - 브라우저 페이지 렌더링
  - 네트워크 요청
  - 복잡한 백엔드 상호작용
  - **계산 비용 극증**
  - **온라인 행동 시뮬레이션 불가능** - 이전 상태로 되돌릴 수 없음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

### 6.2 실제 배포 단계의 도전과제

#### **1. 리워드 함수 일반화**
- 현재: 코딩 도메인에서는 테스트 케이스 통과 여부로 명확한 리워드 정의 가능
- 문제: 다른 도메인(의료, 법률, 과학)에서는 리워드 함수 정의 어려움
- 가능한 해결책: Constitutional AI처럼 자연어로 리워드 함수 직접 정의 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

#### **2. 멀티모달 및 함수 호출 미지원**
- o1-preview/o1-mini는 현재 이미지 업로드, 파일 처리, 웹 브라우징 미지원 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)
- 이들 기능은 향후 "완전한 버전"에서 포함될 예정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

#### **3. 적응형 추론 시간**
- 현재: 추론 토큰을 고정 또는 단순 스케줄로 생성
- 필요: 문제 복잡도에 따른 **동적 System 1/2 전환**
  - 간단한 문제 → System 1 (빠른 직관적 반응)
  - 복잡한 문제 → System 2 (깊은 추론) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

***

## 7. 일반화 성능 향상의 가능성과 전략

### 7.1 현재 강점

1. **도메인 기반 강점**: 코딩 도메인에서 명확한 리워드 신호(테스트 통과)
2. **프로세스 감독의 효과**: PRM의 단계별 피드백이 부분 정확도(partial correctness) 감지 가능
3. **자기 놀이 메커니즘**: 모델이 자동으로 더 어려운 문제를 탐색할 수 있음

### 7.2 일반화 향상 전략

#### **전략 1: 다중 도메인 확장**
| 도메인 | 리워드 신호 | 확인 방법 |
|-------|----------|---------|
| 수학 | 답이 정확한가? | 기호 계산(SymPy 등) |
| 논리 | 논증이 타당한가? | 형식적 검증 도구 |
| 과학 | 설명이 물리법칙을 따르는가? | 도메인 특화 검증기 |
| 코딩 | 테스트 통과? | **o1-Coder의 TCG** |

#### **전략 2: 세계 모델(World Model) 구축**
- 모델이 내부 세계 모델을 학습하여 상태 전이 자체를 예측
- AlphaGo Zero → MuZero의 발전 경로를 따르기 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)
- 최근 진전: Google의 Genie 2, 1X의 로봇용 세계 모델 등

#### **전략 3: 멀티모달 확장**
- 이미지, 다이어그램, 그래프 등을 포함한 입력 처리
- 각 모달리티에 대한 특화된 PRM 설계

#### **전략 4: 하이브리드 리워드 설계**
- 규칙 기반 리워드(hard rewards): 객관적 정확도
- 학습 기반 리워드(learned rewards): 뉘앙스 있는 품질 평가
- 두 신호의 **동적 가중합** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

### 7.3 System-2 Alignment로의 확장

논문이 제시하는 중요한 방향: **System-2 기반 안정성 강화** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d34a01-6c3b-4a1f-9edb-6d9426a4c41a/2412.00154v2.pdf)

기존 연구(o1의 Safety Paper)에서 보여준 바와 같이:
- System-2 추론이 모델로 하여금 입력을 더 철저히 평가하도록 유도
- 잠재적 위험을 더 잘 인식
- 치우침 수정에 더 적극적

**미래 방향**:
- Self-Play+RL을 안정성 평가에 적용
- 반사적 추론(reflective reasoning)으로 편향 감지 및 수정
- 전체 추론 과정의 투명성 향상

***

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 주요 경쟁/협력 모델 비교

#### **OpenAI o1 (2024년 9월)** [arxiv](https://arxiv.org/abs/2410.13639)

**특성**:
- 첫 번째 대규모 상용 "추론 모델"
- 일반적 추론 능력 강화

**방법론**:
- RL + Chain-of-Thought
- 비공개 기술 세부사항

**성능** (OpenAI 주요 벤치마크):
- AIME 2024: 74% (vs. GPT-4o 12%)
- Codeforces: 89백분위수 (전문가 수준)
- GPQA (과학): 92% (PhD 수준) [arxiv](https://arxiv.org/abs/2409.18486)

**한계**:
- 폐쇄적 (API 기반만 접근)
- 종종 빌린 논리 포함 가능

***

#### **DeepSeek-R1 (2025년 1월)** - **가장 중요한 비교 대상** [arxiv](https://arxiv.org/abs/2501.12948)

**혁신적 특징**:
- **순수 RL만 사용** (SFT 없이 직접 RL 적용)
  - DeepSeek-R1-Zero: 콜드 스타트 데이터 없음
  - DeepSeek-R1: 최소 콜드 스타트 후 RL
- 첫 번째 **공개 소스** 추론 모델

**훈련 파이프라인** (DeepSeek-R1): [arxiv](https://arxiv.org/html/2501.12948v1)
```
1. SFT 초기 단계 (선택): 거부 샘플링(Rejection Sampling)
2. RL 훈련 1단계: 순수 추론 능력 강화
   - 리워드: 규칙 기반 (수학, 코딩, 논리)
3. 거부 샘플링: 최상의 예시 선택
4. SFT 2단계: 대화 및 비추론 데이터
5. RL 훈련 2단계: 안정성 및 일반 능력 [huggingface](https://huggingface.co/deepseek-ai/DeepSeek-R1)
```

**성능**: o1-preview와 거의 동등 (다양한 벤치마크)

**리워드 함수**: [nature](https://www.nature.com/articles/s41586-025-09422-z)

$$\text{Reward} = \text{Reward}\_{\text{reasoning}} + \text{Reward}\_{\text{general}} + \text{Reward}_{\text{language}}$$

**혁신 의의**:
- SFT 없이도 강화학습만으로 추론 능력 발현 가능 증명
- 세계 최초로 공개하여 재현 가능성 높음

***

#### **Process Reward Models (PRM) 관련 핵심 연구**

**GroundedPRM (2024)** - [yaoz720.github](https://yaoz720.github.io/GroundedPRM/)

**혁신**:
- **MCTS + 도구 검증** 결합
- **실행 기반 정확도 신호** (hallucination 제거)

**구조**:
```
1. MCTS로 구조화된 추론 경로 생성
2. 각 중간 단계를 외부 도구로 검증 (Wolfram Alpha 등)
3. 트리 백프로파게이션으로 단계별 리워드 계산
4. 하이브리드 보상 집계 (도구 기반 + 결과 기반)
```

**결과**:
- 단 40K 자동 라벨 샘플로 훈련 (인간 라벨의 10%)
- 기존 PRM 대비 SOTA 달성

**o1-Coder와의 관계**: o1-Coder의 PRM과 유사하지만, GroundedPRM이 더 정교한 **검증 메커니즘** 제시

***

**ThinkPRM (2024)** - [arxiv](https://arxiv.org/pdf/2504.16828.pdf)

**특징**:
- **생성형 PRM**: 각 단계에 대해 검증 체인-오브-생각(verification CoT) 생성
- 기존 판별형(discriminative) PRM보다 **데이터 효율적**

**이점**:
- 인간이 읽을 수 있는 검증 설명 제공
- 8000개 단계 라벨로만 훈련 (극히 데이터 효율적)

***

**FoVer (2025)** - [arxiv](https://arxiv.org/pdf/2505.15960.pdf)

**혁신**: 형식적 검증 도구(Z3, Isabelle)로 **단계별 오류 자동 라벨링**

**방법**:
- 형식 논리 및 정리 증명(theorem proving) 작업에 적용
- 각 단계를 독립적으로 형식적으로 검증
- 인간 주석 완전 제거

***

#### **MCTS 기반 코드 생성 연구**

**GIF-MCTS (2024)** - [merlerm.github](https://merlerm.github.io/assets/pdf/papers/dainese2024generating.pdf)

**개념**: 코드 세계 모델(Code World Models) 생성을 위한 MCTS 기반 전략

**특징**:
- Generate (생성) → Improve (개선) → Fix (수정)
- MCTS로 블록 단위 생성과 전체 프로그램 편집 결합
- 오프라인 RL 환경에서 평가

**성능**: APPS 벤치마크 경쟁 부문에서 SOTA

***

**RethinkMCTS (2024)** - [aclanthology](https://aclanthology.org/2025.emnlp-main.410.pdf)

**혁신**: 생각 단계에서 **에러 정제**

**절차**:
1. 사고 수준 MCTS 탐색 (코드 생성 전)
2. 피드백 기반 오류 있는 생각 수정
3. 코드 생성

**의의**: o1-Coder의 Action 2 (의사코드 정제)와 개념적 유사

***

#### **LLaMA-Berry (2024)** - [arxiv](https://arxiv.org/abs/2411.14405)

**특화**: 올림피아드 수준의 수학 문제

**방법**: Chain-of-Thought 미세조정 + Pairwise 선호도 최적화(DPO 변형)

**성능**: 올림피아드 벤치마크에서 상당한 향상

***

### 8.2 종합 비교 테이블

| 특징 | **o1-Coder** | OpenAI o1 | DeepSeek-R1 | GroundedPRM | GIF-MCTS |
|------|---|---|---|---|---|
| **출시** | 2024.12 | 2024.09 | 2025.01 | 2024.12 | 2024.11 |
| **공개성** | 오픈소스 ✓ | 비공개 ✗ | 오픈소스 ✓ | 오픈소스 ✓ | 오픈소스 ✓ |
| **코드 특화** | **예** | 아니오 | 아니오 | 아니오 | **예** |
| **System-2** | **예** | **예** | **예** | **아니오** | 부분적 |
| **핵심 기술** | RL+MCTS+TCG | RL+CoT | 순수 RL | MCTS+도구검증 | MCTS 기반 |
| **PRM 방식** | Point/Pair-wise | 비공개 | 규칙 기반 | 하이브리드 | 없음 |
| **리워드 신호** | 프로세스+결과 | 비공개 | 프로세스+결과 | 검증된 단계+결과 | 최종만 |
| **테스트 케이스 생성** | **TCG (자동)** | 비공개 | 없음 | 외부 도구 | 없음 |
| **의사코드 활용** | **핵심 기능** | 아니오 | 아니오 | 아니오 | 아니오 |
| **자기 놀이** | **6단계 루프** | 비공개 | 암묵적 | 없음 | 없음 |
| **다중 RL 단계** | 1개 | 1개 | **2개** | 없음 | 없음 |

***

### 8.3 주요 연구 동향 정리

#### **2024년의 핵심 전환점**
OpenAI o1(9월)과 그 이후의 폭증하는 추론 모델:
- 단순 Next-token Prediction → **System-2 추론**으로의 패러다임 전환
- Test-time Compute(추론 시간 계산) 중요도 급등

#### **2025년의 주요 발전**
1. **공개 소스 모델의 성공**: DeepSeek-R1이 폐쇄 모델과 동등한 성능 입증
2. **순수 RL의 효능**: SFT 없이도 학습 가능함 증명
3. **PRM의 정교화**: 자동 라벨링, 형식적 검증, 생성형 검증 등

#### **일반화 및 확장의 방향**
- 수학, 과학, 논리 등 다양한 도메인으로 확대
- 멀티모달 추론 모델 개발 시작
- 안정성과 정렬(alignment)을 고려한 System-2 기반 접근

***

## 9. 논문의 영향 및 향후 연구 방향

### 9.1 학문적 기여도

#### **첫 번째: 코드 생성 도메인의 System-2 모델 최초 공개**
- o1의 코딩 특화 복제 시도로서 중요한 참고 자료
- 재현 가능하고 공개 소스로 제공

#### **두 번째: TCG의 혁신**
- 테스트 케이스 자동 생성의 체계화
- DPO를 통한 선호도 기반 최적화 적용
- 코드 평가의 "자동화 및 신뢰성" 향상

#### **두 번째: 의사코드 기반 MCTS의 설계 원리**
- 추상도가 조절 가능한 중간 표현 활용
- 단순한 CoT 프롬핑보다 구조화된 접근
- MBPP 실험으로 ASPR 대폭 향상 입증

#### **세 번째: Self-Play+RL의 6단계 프레임워크 체계화**
- 정책 모델, 리워드 모델, 보조 모델의 조화로운 개선
- 자기 강화 루프의 구체적 구현 방식 제시

### 9.2 실제 적용의 영향

#### **산업계에서의 기대**
1. **오픈소스 코드 생성 도구의 개선**: 기존 CodeBERT, CodeT5 등의 후속 세대
2. **자동 코드 리뷰 및 최적화**: TCG + PRM 조합으로 코드 품질 자동 평가
3. **프로그래머 생산성 향상**: 복잡한 로직 문제에서의 추론 능력 강화

#### **교육적 의미**
- 학생들이 복잡한 알고리즘 문제를 해결할 때 "단계별 추론 과정" 학습 가능
- 의사코드의 중요성 재강조

### 9.3 향후 연구 시 고려할 점

#### **1. 데이터 합성의 품질 관리**
- MCTS로 생성된 데이터가 충분히 다양한가?
- **새로운 데이터 추가 시 기존 데이터와의 분포 이동(distribution shift) 문제**
- 거부 샘플링(rejection sampling)이나 커리큘럼 학습(curriculum learning) 고려

#### **2. 리워드 함수의 견고성**
- 현재 리워드 함수가 코드의 모든 측면을 평가하는가?
  - 효율성(시간복잡도, 공간복잡도)
  - 가독성(readability)
  - 유지보수성(maintainability)
- **다목적 최적화(multi-objective optimization)** 필요

#### **3. 계산 효율성**
- MCTS의 높은 계산 비용 완화 방안
- 적응형 깊이(adaptive depth) MCTS 개발
- 추론 시간과 정확도 간의 **파레토 최적점** 탐색

#### **4. 세계 모델 학습**
- 단순 코드뿐 아니라 **로봇, 시뮬레이션, 복잡 환경**으로 확대
- 내부 상태 표현 학습
- AlphaGo → MuZero로의 진화 경로 추적

#### **5. 다중 언어 및 도메인 일반화**
- 현재: Python 중심
- 향후: Java, C++, Rust 등 다양한 언어 지원
- 수학, 과학, 일반 추론으로 확대

#### **6. 안정성 및 정렬**
- System-2 추론이 더 안전한가?
- 긴 체인-오브-생각이 환상(hallucination)을 증가시키지는 않는가?
- 의도치 않은 행동 패턴의 학습 방지

#### **7. 평가 메트릭의 개선**
- Pass@1, ASPR 외에 다른 메트릭 개발
- **부분 정확도(partial correctness)** 평가
- 런타임 복잡도, 메모리 효율성 평가

***

## 10. 결론: o1-Coder의 위치와 미래

### 10.1 학문적 위치

o1-Coder는:
1. **OpenAI o1의 첫 번째 공개 재현 시도**로서 역사적 중요성
2. **코드 생성 특화** System-2 모델로서 도메인 선택의 정당성
3. **체계적 프레임워크**로서 향후 많은 연구의 기초가 될 가능성

### 10.2 기술적 우수성

| 측면 | 강점 | 약점 |
|------|------|------|
| **리워드 설계** | 프로세스 + 결과 결합, TCG 자동화 | 다중 목표 처리 미흡 |
| **추론 구조** | 의사코드 기반으로 추상도 조절 | 다른 도메인 적용 미검증 |
| **학습 프레임워크** | 자기 놀이 루프의 체계화 | 계산 효율성 문제 |
| **공개성** | 전체 코드 및 데이터셋 공개 | 아직 완성 단계 아님 |

### 10.3 미래 전망

#### **단기 (1-2년)**
- 더 정교한 PRM 설계 및 평가
- 다양한 코딩 벤치마크에서의 성능 개선
- 멀티모달 코드 생성(다이어그램 + 코드) 확대

#### **중기 (2-5년)**
- 수학, 과학 등 다른 도메인으로의 일반화
- 세계 모델 학습으로 복잡 환경 대응
- 산업 도구(IDE 통합)로의 실제 적용

#### **장기 (5년 이상)**
- AGI를 향한 **일반화된 System-2 추론**의 핵심 기술 중 하나
- 인간 수준의 복잡한 문제 해결 능력 구현
- 안정성과 정렬의 완성

***

## 종합 평가

**o1-Coder**는 다음의 이유로 중요한 논문입니다:

1. **혁신성**: System-2 기반 코드 생성의 첫 공개 구현
2. **완성도**: TCG, MCTS, PRM, RL을 조화로운 6단계 프레임워크로 통합
3. **재현성**: 공개 소스로 제공되어 학계 및 산업계의 재현과 확장 가능
4. **실용성**: 실제 코딩 문제에서의 가시적 성능 향상

**다만 향후 개선이 필요한 영역**:
- 아직 최종 버전 결과 미발표
- 계산 효율성의 실질적 해결 필요
- 다른 도메인으로의 일반화 검증 요구

***

## 참고문헌 및 인용

<span style="display:none">[^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89]</span>

<div align="center">⁂</div>

[^1_1]: 2412.00154v2.pdf

[^1_2]: https://arxiv.org/abs/2410.13639

[^1_3]: https://arxiv.org/abs/2409.18486

[^1_4]: https://arxiv.org/abs/2501.12948

[^1_5]: https://arxiv.org/html/2501.12948v1

[^1_6]: https://www.nature.com/articles/s41586-025-09422-z

[^1_7]: https://yaoz720.github.io/GroundedPRM/

[^1_8]: https://liner.com/review/groundedprm-treeguided-and-fidelityaware-process-reward-modeling-for-steplevel-reasoning

[^1_9]: https://arxiv.org/pdf/2504.16828.pdf

[^1_10]: https://arxiv.org/pdf/2505.15960.pdf

[^1_11]: https://arxiv.org/abs/2505.15960

[^1_12]: https://merlerm.github.io/assets/pdf/papers/dainese2024generating.pdf

[^1_13]: https://liner.com/ko/review/generating-code-world-models-with-large-language-models-guided-by

[^1_14]: https://aclanthology.org/2025.emnlp-main.410.pdf

[^1_15]: https://arxiv.org/html/2409.09584v1

[^1_16]: https://arxiv.org/abs/2411.14405

[^1_17]: https://arxiv.org/abs/2412.16720

[^1_18]: https://arxiv.org/abs/2410.01792

[^1_19]: https://arxiv.org/abs/2411.06198

[^1_20]: https://www.cureus.com/articles/301598-openai-o1-preview-vs-chatgpt-in-healthcare-a-new-frontier-in-medical-ai-reasoning

[^1_21]: https://www.mdpi.com/2073-431X/13/11/278

[^1_22]: https://arxiv.org/abs/2410.00033

[^1_23]: https://arxiv.org/abs/2412.18925

[^1_24]: https://arxiv.org/abs/2409.13373

[^1_25]: https://arxiv.org/html/2503.10621v1

[^1_26]: http://arxiv.org/pdf/2502.10867.pdf

[^1_27]: https://arxiv.org/pdf/2412.14135.pdf

[^1_28]: http://arxiv.org/pdf/2410.01792.pdf

[^1_29]: http://arxiv.org/pdf/2410.05669.pdf

[^1_30]: http://arxiv.org/pdf/2411.06198.pdf

[^1_31]: https://arxiv.org/html/2412.16720

[^1_32]: https://en.wikipedia.org/wiki/O1_(generative_pre-trained_transformer)

[^1_33]: https://thesciencebrigade.com/jcir/article/view/563

[^1_34]: https://watercrawl.dev/blog/Unlocking-the-Mind-of-AI-System-1-and-System-2

[^1_35]: https://en.wikipedia.org/wiki/OpenAI_o1

[^1_36]: https://www.qeios.com/read/8G8TB2/pdf

[^1_37]: https://www.reddit.com/r/MachineLearning/comments/1g9v7ag/meta_ai_fair_latest_paper_integrates_system1_and/

[^1_38]: https://www.ultralytics.com/blog/openai-o1-a-new-series-of-openai-models-for-ai-reasoning

[^1_39]: https://aclanthology.org/2024.acl-long.251/

[^1_40]: https://blog.nguyenthanh.asia/bridging-generative-models-and-system-1-with-system-2-the-role-of-logical-programming-in-ai-58ca105c2f

[^1_41]: https://encord.com/blog/openai-o1/

[^1_42]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-123.pdf

[^1_43]: https://www.linkedin.com/pulse/dawn-ais-system-2-thinking-why-2025-marks-cognitive-revolution-babu-hpgjc

[^1_44]: https://codefinity.com/blog/Introducing-OpenAI-o1-preview:-The-Future-of-AI-Reasoning

[^1_45]: https://arxiv.org/html/2412.20367v1

[^1_46]: https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/

[^1_47]: https://arxiv.org/html/2507.17548v1

[^1_48]: https://arxiv.org/abs/2406.13975

[^1_49]: https://arxiv.org/html/2505.23387v1

[^1_50]: https://arxiv.org/html/2501.02497v1

[^1_51]: https://arxiv.org/html/2410.07866v5

[^1_52]: https://arxiv.org/abs/2410.07114

[^1_53]: https://arxiv.org/abs/2412.20367

[^1_54]: https://arxiv.org/html/2410.03662v2

[^1_55]: https://arxiv.org/html/2510.18471v1

[^1_56]: https://arxiv.org/html/2505.21432v2

[^1_57]: https://dl.acm.org/doi/10.1145/3643795.3648391

[^1_58]: https://ieeexplore.ieee.org/document/10710209/

[^1_59]: https://arxiv.org/abs/2410.03210

[^1_60]: https://ieeexplore.ieee.org/document/10762291/

[^1_61]: https://dx.plos.org/10.1371/journal.pcbi.1012647

[^1_62]: https://ieeexplore.ieee.org/document/10678348/

[^1_63]: http://www.proceedings.com/079017-2519.html

[^1_64]: https://arxiv.org/abs/2410.06074

[^1_65]: https://ieeexplore.ieee.org/document/10608368/

[^1_66]: https://arxiv.org/abs/2404.03663

[^1_67]: https://arxiv.org/pdf/2405.15383.pdf

[^1_68]: https://arxiv.org/pdf/2402.03289.pdf

[^1_69]: https://arxiv.org/html/2411.11053

[^1_70]: https://arxiv.org/pdf/2502.14693.pdf

[^1_71]: http://arxiv.org/pdf/2409.09584.pdf

[^1_72]: http://arxiv.org/pdf/2404.16364.pdf

[^1_73]: http://arxiv.org/pdf/2502.11476.pdf

[^1_74]: https://arxiv.org/pdf/2308.07738.pdf

[^1_75]: https://www.semanticscholar.org/paper/Make-Every-Move-Count:-LLM-based-High-Quality-RTL-DeLorenzo-Chowdhury/bb8bc1c66e4462ea0f4b457f4adc383bdde69ee2

[^1_76]: https://www.emergentmind.com/topics/process-reward-models-prm

[^1_77]: https://www.vellum.ai/blog/the-training-of-deepseek-r1-and-ways-to-use-it

[^1_78]: https://arxiv.org/abs/2405.15383

[^1_79]: https://huggingface.co/deepseek-ai/DeepSeek-R1

[^1_80]: https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf

[^1_81]: https://arxiv.org/abs/2504.16828

[^1_82]: https://kimjy99.github.io/논문리뷰/deepseek-r1/

[^1_83]: https://arxiv.org/html/2508.05995v1

[^1_84]: https://arxiv.org/html/2503.04548v1

[^1_85]: https://arxiv.org/pdf/2406.07394.pdf

[^1_86]: https://arxiv.org/pdf/2501.12948.pdf

[^1_87]: https://arxiv.org/html/2511.09054

[^1_88]: https://arxiv.org/abs/2305.20050

[^1_89]: https://arxiv.org/abs/2511.19333
