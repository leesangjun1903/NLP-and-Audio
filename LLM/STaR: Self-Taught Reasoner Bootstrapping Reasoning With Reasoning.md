# STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning

## 1. 핵심 주장과 주요 기여  
**핵심 주장**  
- 대규모 언어 모델(LLM)이 스스로 “합리화(rationale)”를 생성하고, 이를 통해 반복적으로 자기 자신을 향상시킬 수 있다.[1]

**주요 기여**  
- 소량의 체인오브쏘트(chain-of-thought) 예제만으로 자체 생성한 합리화를 증폭(bootstrapping)하는 STaR 기법 제안.  
- 모델 실패 사례에 ‘정답을 힌트로 제공’하여 합리화를 생성하는 *rationalization* 도입.  
- 수학·commonsenseQA·GSM8K에서 STaR이 합리화 없는 직접 예측 방식 대비 성능 대폭 향상 입증.  

## 2. 문제 정의 및 제안된 방법  
### 2.1 해결 과제  
- 복잡한 추론 과제(수학, 상식 QA 등)에서 체인오브쏘트 학습을 위해 대규모 합리화 데이터셋 구축은 비용·실용성 측면에서 한계.  
- Few-shot 체인오브쏘트는 성능 향상에 기여하나, 대량의 fine-tuning 대비 여전히 성능 부족.  

### 2.2 STaR 알고리즘  
1. **Rationale Generation**: Few-shot 예제 $$P$$를 프롬프트에 붙여 모델 $$M$$이 합리화 $$\hat r_i$$와 답 $$\hat y_i$$ 생성.  
2. **Filtering**: $$\hat y_i=y_i$$인 경우만 $$(x_i,\hat r_i,y_i)$$으로 수집.  
3. **Rationalization**: 실패 사례 $$(\hat y_i\neq y_i)$$에 정답 힌트를 주고 합리화 $$\hat r_i^{\text{rat}}$$ 생성, 다시 정답 맞춘 합리화만 수집.  
4. **Fine-tuning**: 원모델 $$M$$에 수집된 합리화 데이터 전체로 미세조정.  
5. 반복(iteration)  

수식화하면, 합리화 생성은 잠재변수 $$r$$를 샘플링하는 정책 그라디언트에 대응하며, 목표 기대보상은  

$$
J(M)=\sum_i \mathbb{E}_{\hat r_i,\hat y_i\sim p_M(\cdot\mid x_i)}[1(\hat y_i=y_i)]
$$  

그라디언트 추정치는  

$$
\nabla J(M)=\sum_i \mathbb{E}_{\hat r_i,\hat y_i}[1(\hat y_i=y_i)\,\nabla\log p_M(\hat r_i,\hat y_i\mid x_i)].
$$  

Filtering이 $$1(\hat y_i=y_i)$$로 작용한다.[1]

### 2.3 모델 구조  
- GPT-J(6B) 기반, 표준 Transformer 디코더 아키텍처 사용.  
- Few-shot 예제 길이에 따라 동적으로 프롬프트 구성, 매 outer loop마다 원모델로부터 재학습(retrain from scratch)하여 과적합 방지.  

### 2.4 성능 향상 및 한계  
| Task                  | Baseline Direct | STaR w/o Rationalization | STaR w/ Rationalization |
|-----------------------|-----------------|--------------------------|-------------------------|
| CommonsenseQA (dev)   | 60.0%           | 68.8%                    | **72.5%** vs GPT-3 73.0% |
| GSM8K (test)          | 5.8%            | 10.1%                    | **10.7%**               |
| Arithmetic (1–5 digit)| 76.3%           | 89.5%                    | —                       |

- **향상 요인**: 합리화 포함 학습이 직접 예측 대비 일반화 성능·추론 능력 대폭 상승.  
- **한계**:  
  - 초기 few-shot 성능이 무작위 추측 이상이어야 부트스트랩 가능.  
  - 2-choice 등 높은 확률 예측 도메인에선 불량 합리화 필터링 어려움.  
  - 합리화의 *faithfulness*(모델 내부 실제 추론 반영 여부) 보장 불가.  

## 3. 일반화 성능 향상 관점  
- **합리화 샘플 필터링**을 통한 학습이 직접 답 예측보다 *latent variable* 역할을 하는 중간 추론 강화를 유도.  
- *Rationalization*이 실패 사례에도 학습 신호 제공해, 어려운 문제 난이도 관내에서 해결 폭을 넓힘.  
- Arithmetic에서 higher-digit out-of-distribution 문제(9–10 digit)에 대한 일반화 능력 확인.  

## 4. 향후 연구에의 영향 및 고려점  
- **영향**: 소량의 체인오브쏘트로 시작해 대규모 추론 데이터셋을 자동 생성하는 메커니즘 제시, 다양한 추론 도메인 확대 가능성 제시.  
- **고려점**:  
  1. 합리화 신뢰도(transparent reasoning) 및 *bias* 증폭 우려 해결을 위한 필터링·검증 기법 필요.  
  2. *Faithfulness* 확보 위한 내적 설명 기법 연구.  
  3. 초기 few-shot 예제 선택 및 rationalization hint 설계에 따른 성능 민감도 분석.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/dfd97f13-1b01-4b20-9b44-a194f13b79a8/2203.14465v2.pdf)
