# Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting

**핵심 주장 및 주요 기여**  
Deep pretrained Language Model(LM)의 fine-tuning 과정에서 발생하는 **catastrophic forgetting**을 완화하기 위해, 사전학습(pretraining)과 downstream task를 **동시 학습**하는 multi-task 아이디어를 도입하였다.  
1. **Pretraining Simulation**: 사전학습 데이터를 사용하지 않고도 사전지식을 “recall”하기 위한 **quadratic penalty** 기반 근사화 기법 제안  
2. **Objective Shifting**: fine-tuning 초기에는 사전학습 지식을, 후기로 갈수록 downstream task에 집중하도록 하는 **annealing coefficient** 설계  
3. **RECADAM Optimizer**: Adam에 두 메커니즘을 decoupled 방식으로 통합한 최적화기 공개  

이로써 BERT-base가 BERT-large를 능가하는 성능을 달성하며, GLUE 벤치마크에서 state-of-the-art 결과를 기록한다.[1]

***

## 1. 해결하고자 하는 문제  
순차적 전이학습(sequential transfer learning)은 사전학습 지식을 downstream task로 옮길 때 **기존 지식을 망각**하고 새로운 task에 과도하게 적합(overfitting)되는 문제가 있다. 특히 데이터가 적은 task일수록 망각이 심각해져 최적 성능을 내지 못한다.[1]

## 2. 제안 방법  
### 2.1 Pretraining Simulation  
사전학습 데이터 DS가 없는 상황에서 source task 손실 Loss_S를 대체하기 위해, pretrained 파라미터 θ₀ 주위의 손실 함수를 **Laplace 근사** 및 **독립 파라미터 가정**을 통해 다음과 같은 **quadratic penalty** 형태로 근사한다:  

$$
\text{Loss}_S(\theta) \approx \tfrac{1}{2} \sum_i F_i (\theta_i - \theta_{0,i})^2
$$  

여기서 $$F_i$$는 파라미터 i의 diagonal Fisher 정보, $$\theta_{0,i}$$는 pretrained 값이다.[1]

### 2.2 Objective Shifting  
다중 과제 학습 목적함수  

$$
\text{Loss}_M = \alpha\,\text{Loss}_T + (1-\alpha)\,\text{Loss}_S
$$  

에서 $$\alpha$$를 **시그모이드 형태**의 annealing 함수 $$t(timestep)$$로 대체하여 학습 초기에는 $$\alpha\to0$$로 사전지식 recalling, 후기로 갈수록 $$\alpha\to1$$로 downstream task 학습에 집중하도록 한다:  

$$
t = \frac{1}{1 + \exp(-k\,(step - t_0))},\quad
\text{Loss} = t\,\text{Loss}_T + (1-t)\,\text{Loss}_S
$$  

$$k, t_0$$는 shifting 속도 및 시작점 제어 하이퍼파라미터이다.[1]

### 2.3 RECADAM Optimizer  
Adam의 gradient update 규칙에서 quadratic penalty와 annealing 동작을 **decouple**하여, target task gradient만이 적응적 학습률(adaptive rate)을 적용받게 한다. 이로써 모든 파라미터에 균일한 recalling 강도를 보장한다.[1]

***

## 3. 모델 구조  
기존의 Transformer 기반 BERT 및 ALBERT 아키텍처(12–64 head, 12–4096 hidden dim)를 그대로 사용하며, fine-tuning 시 RECADAM을 통해 위 메커니즘을 적용한다. 사전학습 objective(Masked LM + NSP 등)는 손실 항으로 직접 사용하지 않고, 파라미터 근접성 항(Pretraining Simulation)으로 대체된다.[1]

***

## 4. 성능 향상 및 한계  
- **GLUE benchmark**에서 BERT-base 대비 평균 +1.1, 데이터 적은 task(COLA, MRPC, RTE 등)에서 +1.7까지 개선. BERT-large 대비 절반 파라미터로 동등 이상 성능 달성  
- ALBERT-xxlarge에서도 평균 +0.7 향상, 특히 small-data task에서 +1.5 개선[1]
- **한계**:  
  - Annealing 속도 $$k$$ 설정이 성능 및 수렴 속도에 민감  
  - 극히 작은 데이터셋에서는 초기 recalling이 과도하여 수렴 지연 가능  
  - task 간 유사도가 낮으면 recalling된 사전학습 지식이 방해 요소가 될 수 있음  

***

## 5. 모델의 일반화 성능 향상 가능성  
Pretraining Simulation이 **Fisher 정보 기반 파라미터 정규화**를 통해 사전학습의 분산 정보(robust representation)를 보존하므로, downstream 과제에서도 **과적합 방지**와 **다양한 데이터 분포**에 대한 대응력이 향상된다.  
Objective Shifting은 학습 초기 **general knowledge**를 먼저 강화한 뒤 특화 지식으로 전환하기에, **overfitting 제어** 및 **보다 넓은 함수 공간 탐색**이 가능해져 새로운 데이터 분포에 대한 일반화 성능을 높인다.[1]

***

## 6. 향후 연구 영향 및 고려사항  
- **범용 최적화기 설계**: RECADAM처럼 사전지식 recalling 메커니즘을 통합한 optimizer 개발이 확대될 전망  
- **하이퍼파라미터 자동화**: $$k, t_0, F_i$$ 추정치의 자동 튜닝 및 데이터 의존적 설정 연구 필요  
- **다양한 도메인 적용**: 언어 이외 비전·음성·멀티모달 분야로 Pretraining Simulation 적용 가능성 탐색  
- **효율성 개선**: Fisher 정보 근사 및 annealing 연산 비용 절감 기법 개발이 실용화 관건  

이 논문은 사전학습과 fine-tuning 사이의 간극을 메커니즘 차원에서 메우며, 다양한 전이학습 시나리오에서 **지식 망각 없이 성능 최적화** 연구에 중요한 길잡이가 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/03fac503-b08b-4f7c-8d0c-412e016f1c72/2004.12651v1.pdf)
