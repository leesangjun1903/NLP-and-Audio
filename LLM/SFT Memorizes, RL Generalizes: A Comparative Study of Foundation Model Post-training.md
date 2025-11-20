# SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training

### 1. 핵심 주장 및 주요 기여

이 논문의 **핵심 주장**은 기초 모델(Foundation Model)의 사후 학습(Post-training) 단계에서 사용되는 두 가지 주요 기법인 지도 학습 기반 미세 조정(SFT, Supervised Fine-Tuning)과 강화 학습(RL, Reinforcement Learning)이 **근본적으로 다른 특성**을 가진다는 것입니다.[1]

**주요 기여:**

SFT는 주로 **훈련 데이터 암기(memorization)**에 의존하는 반면, RL은 **일반화 가능한 규칙 학습(generalization)**을 가능하게 합니다. 이러한 차이는 텍스트 기반 규칙 변형과 시각적 변형 모두에서 일관되게 관찰됩니다.[1]

논문은 두 가지 새로운 평가 환경을 도입합니다: **(1) GeneralPoints** - 산술 추론 능력을 평가하는 카드 게임, **(2) V-IRL** - 실제 네비게이션 환경에서의 공간 추론 능력 평가.[1]

특히, RL은 Llama-3.2-Vision-11B 모델에서 V-IRL 벤치마크에서 **기존 최고 성능 대비 33.8% 향상**(44.0% → 77.8%)을 달성하며, 시각 인식 능력도 동시에 개선합니다.[1]

***

### 2. 해결하고자 하는 문제 및 제안하는 방법

#### 2.1 문제 설정

기초 모델의 사후 학습 단계에서 SFT와 RL의 역할이 불명확합니다. 특히, 다음과 같은 핵심 질문에 답하기 위해 연구를 진행했습니다:[1]

- 모델이 훈련 데이터를 **암기**하는가, 아니면 **일반화 가능한 규칙**을 학습하는가?
- 텍스트 기반 규칙 변형뿐만 아니라 시각적 변형에서도 일반화가 가능한가?
- RL 학습에서 SFT의 역할은 무엇인가?

#### 2.2 제안하는 방법론

**다중 턴 RL 프레임워크 (Multi-turn RL Framework):**

논문은 **순차 수정 공식(Sequential Revision Formulation)**을 기반으로 한 다중 턴 RL 방식을 채택합니다.[1]

$$v_{in}^t = \text{concat}\left(v_{in}^0, [v_{out}^k, v_{ver}^k]_{k=0}^{t-1}\right)$$

여기서:
- $$v_{in}^t$$: 시간 t의 입력 프롬프트
- $$v_{out}^k$$: 모델의 k번째 출력
- $$v_{ver}^k$$: 검증자의 k번째 출력
- $$t$$: 현재 턴 번호

**상태-행동 공간 정의:**

LLM의 경우:
$$S := V^m$$

VLM(Vision-Language Model)의 경우:
$$S := V^m \times O$$

여기서 $$V$$는 어휘(토큰) 공간, $$O$$는 RGB 이미지 공간입니다.[1]

**보상 함수 설계:**

$$\text{VER}(v_{out}^t) \mapsto (r_t, v_{ver}^t)$$

검증자(Verifier)는 모델 출력을 평가하고 **결과 기반 보상(Outcome-based Reward)**을 생성합니다.[1]

GeneralPoints의 보상 함수:
- $$r = +5$$: 정확한 수식 생성
- $$r = -1$$: 올바른 형식이지만 잘못된 값
- $$r = -2$$: 주어지지 않은 숫자 사용
- $$r = -3$$: 기타 불법적 수식

V-IRL의 보상 함수:
- $$r = +1$$: 현재 좌표에서 올바른 행동
- $$r = -1$$: 올바르지 않은 행동
- $$r = -1.5$$: 랜드마크 인식 실패

**정책 최적화:**

PPO(Proximal Policy Optimization)를 사용하여 정책 $$\pi_\theta$$를 최적화합니다:

$$\max_{\pi \in \Pi} \mathbb{E}_\pi\left[\sum_{t=0}^{T} r_t\right]$$

#### 2.3 모델 구조

**백본 모델:** Llama-3.2-Vision-11B (Dubey et al., 2024)[1]

**학습 파이프라인:**
1. **초기화 단계**: SFT를 통해 기본 모델을 초기화
2. **비교 단계**: 동일한 계산 예산으로 SFT와 RL을 각각 학습
3. **평가 단계**: 분포 내(In-Distribution) 및 분포 외(Out-of-Distribution) 성능 평가

***

### 3. 성능 향상 및 결과

#### 3.1 규칙 기반 일반화 (Rule-based Generalization)

**규칙 변형에서의 OOD 성능:**

| 작업 | SFT 변화 | RL 변화 |
|------|---------|---------|
| GP-L (텍스트) | -8.1% (11.5% → 3.4%) | +3.5% (11.5% → 15.0%) |
| V-IRL-L (텍스트) | -79.5% (80.8% → 1.3%) | +11.0% (80.8% → 91.8%) |
| GP-VL (시각-언어) | -5.6% (11.2% → 5.6%) | +3.0% (11.2% → 14.2%) |
| V-IRL-VL (시각-언어) | -33.2% (35.7% → 2.5%) | +9.3% (35.7% → 45.0%) |

**핵심 발견**: RL은 모든 조건에서 일관되게 OOD 성능을 향상시키는 반면, SFT는 모든 경우에 성능 저하를 보입니다.[1]

#### 3.2 시각적 일반화 (Visual Generalization)

**시각 변형에서의 OOD 성능:**

- **GP-VL (카드 색상 변형)**: +17.6% (23.6% → 41.2%) for RL vs -9.9% (23.6% → 13.7%) for SFT
- **V-IRL-VL (도시 간 이동)**: +61.1% (16.7% → 77.8%) for RL vs -5.6% (16.7% → 11.1%) for SFT[1]

#### 3.3 시각 인식 능력 향상

RL 학습이 시각 인식 정확도를 동시에 개선합니다:[1]

$$\text{Visual Recognition Accuracy}_{RL} > \text{Visual Recognition Accuracy}_{SFT}$$

논문은 계산 예산 증가에 따라 RL은 인식 정확도와 전체 성능을 모두 개선하는 반면, SFT는 둘 다 악화됨을 보여줍니다.

#### 3.4 검증 반복(Verification Iterations)의 영향

검증 단계 수 증가에 따른 OOD 성능 향상:[1]

- **1 반복**: +0.48%
- **3 반복**: +2.15%
- **5 반복**: +2.99%
- **10 반복**: +5.99%

***

### 4. 모델 일반화 성능 향상 가능성

#### 4.1 일반화 메커니즘 분석

**RL의 일반화 우월성 이유:**

1. **결과 기반 보상 구조**: RL은 최종 결과에만 초점을 맞춰, 계산 과정의 다양성을 허용합니다.[1]

2. **반복적 오류 수정**: 다중 턴 구조를 통해 모델이 처음 시도에서 실패한 경우 수정할 수 있는 기회를 제공합니다.

3. **시각 인식 개선**: RL 학습이 기저의 시각 인식 능력을 향상시킵니다.[1]

**SFT의 암기 특성:**

- 토큰 레벨의 감독학습이 훈련 데이터의 **정확한 재현**에 초점을 맞춥니다.
- 추론 토큰과 인식 토큰의 빈도 불균형으로 인해 시각 VLM에서 특히 심각한 과적합 발생.[1]

#### 4.2 일반화 확장성

**계산 예산 증가에 따른 성능 궤적:**

RL의 성능은 계산 증가에 따라 **단조 증가(Monotonic Increase)** 추세를 보이는 반면, SFT는 빠르게 포화되고 때로는 악화됩니다.[1]

***

### 5. 한계 및 미해결 과제

#### 5.1 SFT의 시각 도메인 실패

GP-VL에서 SFT가 기대보다 낮은 분포 내 성능을 보입니다. 이는 다음과 같은 이유로 추정됩니다:[1]

- **토큰 빈도 불균형**: 추론 토큰이 시각 인식 토큰보다 훨씬 빈번하게 나타나, SFT가 추론에 과도하게 과적합

$$\text{Freq}(\text{Reasoning Tokens}) >> \text{Freq}(\text{Recognition Tokens})$$

#### 5.2 RL의 한계 사례

**극단적인 과적합 체크포인트에서의 복구 실패:**

매우 과적합된 SFT 모델(OOD 성능 < 1%)에서 시작하면 RL도 성능을 회복할 수 없습니다.[1]

**지침 따르기의 중요성:**

기본 모델이 지침을 따르지 않으면 RL 학습이 실패합니다. 따라서 SFT 초기화가 필수적입니다:[1]

$$\text{RL Success} \propto \text{Instruction Following Capability}$$

#### 5.3 SFT의 필수성

RL의 우월성에도 불구하고, **SFT는 여전히 필수적**입니다:[1]

- 출력 형식 표준화
- 지침 따르기 능력 부여
- RL이 개선할 수 있는 기초 제공

***

### 6. 앞으로의 연구에 미치는 영향

#### 6.1 최신 연구 흐름과의 연계

**1. "Quagmires in SFT-RL Post-Training" (Kang et al., 2025)**[2]

이 연구는 높은 SFT 점수가 반드시 RL 성능 향상으로 이어지지 않음을 보여줍니다. 오히려:

- 일반화 손실(Generalization Loss)과 Pass@large k가 RL 성능의 더 나은 지표
- 일부 경우 SFT 없이 기본 모델에서 RL을 직접 수행하는 것이 나을 수 있음

이는 본 논문의 발견을 확장하여, **SFT의 품질 평가 지표**의 재정의 필요성을 시사합니다.[2]

**2. 추론 시간 계산 확장 (Inference-Time Compute Scaling)**[3][4]

최근 연구들은 추론 단계에서 계산을 증가시켜 성능을 향상시키는 방법을 탐구합니다. 이는 본 논문의 **검증 반복 증가**와 유사한 메커니즘입니다:[4]

- Chain-of-Thought (CoT) 프롬프팅
- 투표 기반 방법 (Self-Consistency)
- 테스트 시간 스케일링

본 논문의 다중 턴 RL 접근법은 **훈련 시간에 이러한 추론 시간 계산을 통합**하는 새로운 관점을 제시합니다.[1]

**3. 멀티모달 모델의 일반화**[5][6]

최근 연구는 VLM의 시각 인식 능력을 개선하는 데 초점을 맞추고 있습니다. 본 논문은 RL이 이를 달성하는 한 가지 효과적인 방법임을 보여줍니다.[1]

#### 6.2 미래 연구 고려사항

**1. 통합 학습 패러다임**

단순한 "SFT → RL" 순차 구조 대신, **적응형 하이브리드 접근법** 개발:

$$\text{Loss}_{Hybrid} = \lambda_1 \cdot \text{Loss}_{SFT} + \lambda_2 \cdot \text{Loss}_{RL}$$

여기서 $$\lambda_1, \lambda_2$$는 학습 과정에서 동적으로 조정되는 가중치입니다.

**2. 토큰 빈도 밸런싱**

시각 도메인에서 SFT 성능 악화 문제를 해결하기 위해:

- 인식 토큰에 높은 가중치 적용
- 보조 손실 함수를 통한 균형 강제

$$\text{Loss}_{Balanced} = \text{Loss}_{Reasoning} + \alpha \cdot \text{Loss}_{Recognition}$$

**3. 동적 체크포인트 선택**

RL 초기화 전 최적의 SFT 체크포인트를 선택하는 메커니즘 개발. 최근 연구에서 제안된 **일반화 손실** 메트릭 활용:[2]

$$\text{Optimal Checkpoint} = \arg\min_t (\text{Generalization Loss}_t)$$

**4. 멀티태스크 일반화**

현재 연구는 특정 작업(GeneralPoints, V-IRL)에 제한되어 있습니다. 향후 연구는:

- 다양한 추론 작업에 대한 일반화
- 언어 작업과 시각 작업 간 전이 학습
- 장기 맥락 이해(Long Context Understanding)

**5. 계산 효율성 분석**

$$\text{Efficiency} = \frac{\text{Performance Gain}}{\text{Computational Cost (GFLOPs)}}$$

RL의 높은 계산 비용(추론 단계 포함)을 고려한 효율성 지표 개발이 필수적입니다.[1]

**6. 인간 피드백 통합**

현재 연구는 자동 보상자(Automatic Verifier)를 사용합니다. 향후 연구는:

- 인간 피드백 기반 RL (RLHF)과의 비교
- 약한 신호(Weak Signal)로부터의 학습

#### 6.3 산업 응용 방향

**1. 생산 시스템에서의 최적화**

- SFT/RL 하이브리드의 비용-효과 분석
- 지연 시간(Latency) vs 정확성(Accuracy) 트레이드오프

**2. 멀티모달 AIAgent 개발**

본 논문의 V-IRL 결과는 로봇 공학, 자율주행 등에 직접 적용 가능합니다:[1]

- 시각 인식 + 공간 추론의 통합
- 분포 외 환경에서의 강건성

**3. 추론 최적화 서비스**

검증 반복 수를 동적으로 조정하여 비용 최적화:

$$\text{Verification Steps} = f(\text{Task Complexity}, \text{Available Budget})$$

***

### 7. 결론

이 논문은 기초 모델의 사후 학습에 있어 **SFT와 RL의 근본적인 차이**를 명확히 합니다. RL의 우월한 일반화 능력은 **결과 기반 보상 신호와 반복적 오류 수정 메커니즘**에서 비롯되며, 이는 텍스트와 시각 도메인 모두에서 검증되었습니다.[1]

특히, **다중 턴 검증 프레임워크**와 **장기 추론 시간 계산** 간의 시너지는 향후 추론 최적화 연구의 중요한 방향을 제시합니다.[3][4]

동시에, SFT의 필수적 역할을 인정하면서도, 현재의 순차 파이프라인보다는 **적응형 통합 학습**으로의 진화가 필요함을 시사합니다.[2]

이 연구는 단순한 비교 연구를 넘어, 미래의 강건하고 일반화 가능한 AI 시스템 구축을 위한 **이론적 토대**와 **실증적 증거**를 제공합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b3136e5c-cbd5-4b11-83ad-9d6f35194880/2501.17161v2.pdf)
[2](https://arxiv.org/abs/2510.01624)
[3](https://arxiv.org/abs/2502.12521)
[4](https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling)
[5](http://arxiv.org/pdf/2406.18043.pdf)
[6](https://arxiv.org/html/2412.09858v1)
[7](https://aclanthology.org/2023.findings-emnlp.40.pdf)
[8](https://arxiv.org/html/2501.17161)
[9](http://arxiv.org/pdf/2309.16382.pdf)
[10](https://arxiv.org/pdf/2402.15567.pdf)
[11](https://arxiv.org/pdf/2312.01939.pdf)
[12](https://arxiv.org/pdf/2210.03821.pdf)
[13](https://icml.cc/virtual/2025/poster/44633)
[14](https://cfp.in.pycon.org/2025/talk/PBLWQR/)
[15](https://arxiv.org/abs/2501.17161)
[16](https://www.linkedin.com/pulse/supervised-fine-tuning-vs-reinforcement-learning-model-sowmya-vivek-txnfc)
[17](https://huggingface.co/papers/2501.17161)
[18](https://tianzhechu.com/SFTvsRL/)
[19](https://openreview.net/forum?id=dYur3yabMj)
