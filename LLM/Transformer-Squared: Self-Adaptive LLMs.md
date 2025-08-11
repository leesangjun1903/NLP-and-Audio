# $Transformer^2$ (Transformer-Squared): Self-Adaptive LLMs

## 1. 핵심 주장 및 주요 기여  
Transformer2는 대규모 언어 모델(LLM)에 **실시간으로 과제 특성에 맞춰 동적으로 적응**할 수 있는 **Self-Adaptive** 프레임워크이다.  
- **핵심 주장**: 기존의 고정된 파인튜닝 방식이 아닌, 사전학습된 모델의 가중치 특이값(singular values)에만 최소한의 벡터를 학습해 얹음으로써 적은 파라미터로 다양한 과제에 빠르게 적응 가능하다.  
- **주요 기여**:  
  1. **SVF(Singular Value Fine-tuning)**: 모델 가중치의 특이값만 조정하는 PEFT(parameter-efficient fine-tuning) 기법 제안  
  2. **Expert Vectors**: RL로 학습된 소규모 벡터를 “전문가”로 구성해 모듈화  
  3. **Two-Pass Adaptation**: (1) 과제 분류/특성 파악 → (2) 전문가 벡터 조합 적용의 두 단계 추론 메커니즘  
  4. **세 가지 Adaptation 전략**: Prompt-based, Classification Expert, CEM 기반 Few-Shot

***

## 2. 문제 정의 및 해결 방법 상세

### 2.1 문제 정의  
- **기존 한계**:  
  - 대규모 LLM 전역 파라미터 재학습은 비용·시간 과다  
  - LoRA 등 PEFT도 모듈 수 증가시 과적합·저장 비용 상승  
  - 고정된 파인튜닝으로 동적·다양한 과제 대응 어려움  

### 2.2 제안 기법  
#### 2.2.1 SVF (Singular Value Fine-tuning)  
- 임베딩·어텐션·MLP 각 가중치 $$W=U\Sigma V^\top$$에서  
- **학습 파라미터**: $$\boldsymbol{z}\in\mathbb{R}^r$$  
- **수정**: $$\Sigma' = \Sigma \odot \mathrm{diag}(\boldsymbol{z})$$, $$W' = U\Sigma'V^\top$$  
- **장점**:  
  - **극소 파라미터**($$r$$개만 학습)  
  - **고차원 저차원 경계** 없이 **전 영역(full-rank)** 조정  
  - **과적합 억제**: 기존 특이값 스케일링만, 과도한 변형 방지  

#### 2.2.2 RL 기반 전문가 벡터 학습  
- 정책경사(REINFORCE) + KL 정규화로 직접 과제 성능 보상 최적화  

$$
J(\theta_z)=\mathbb{E}\Big[\log\pi_{\theta_W'}(\hat y|x)\,r(\hat y,y)\Big]\;-\;\lambda\,D_{KL}(\pi_{\theta_W'}\Vert\pi_{\theta_W})
$$  

#### 2.2.3 두 단계 추론(2-Pass Inference)  
1. **1차 추론**: 입력 프롬프트에서 과제 특성 분류 → 적절한 전문가 벡터 $$z'$$ 결정  
2. **2차 추론**: $$z'$$로 가중치 수정 후 실제 답변 생성  

#### 2.2.4 Adaptation 전략  
- **Prompt-Based**: “코딩/수학/추론/기타” 분류 프롬프트 사용  
- **Classification Expert**: 또 다른 SVF-튜닝 벡터로 분류 모델 학습  
- **Few-Shot (CEM)**: 소량의 예시 답안 집합에 CEM 적용해 전문가 벡터 가중치 $$\alpha_k$$ 탐색  

### 2.3 모델 구조  
- **기반 모델**: LLAMA3-8B, MISTRAL-7B, LLAMA3-70B  
- **SVF 적용 위치**: 어텐션 프로젝션 & MLP 프로젝션 계층  
- **전문가 모듈**: 과제별 SVF 벡터 집합 $$z_1,\dots,z_K$$  

***

## 3. 성능 향상 및 한계

### 3.1 성능 향상  
- **SVF vs LoRA**: 거의 모든 과제에서 SVF가 더 높은 성능[Table 1]  
- **Self-Adaptation (Unseen Tasks)**: Few-Shot 전략이 최대 +4% 이상 개선[Table 2]  
- **Vision-Language**: TextVQA→OKVQA 전이에서 SVF+Transformer2 적용 시 +40% 개선  

### 3.2 일반화 성능  
- **Cross-Model Transfer**: LLAMA3-8B 전문가 벡터를 MISTRAL-7B에 적용해 성능 향상  
- **Few-Shot CEM**: 3∼5샷만으로도 효과적, 10샷 이상에서는 성능 포화  
- **내재적 정규화**: RL+특이값 스케일링이 과적합 억제 → 적은 데이터로도 안정적 성능  

### 3.3 한계  
- **추론 오버헤드**: 2-Pass, Few-Shot CEM은 과제당 일회성 계산 비용 발생 (대규모 배치 시 비용 분산)  
- **전문가 벡터 한계**: 사전학습 모델 가중치에 내재된 특성에 의존 → 전혀 새로운 기능 학습에는 제약  
- **확장성**: 전문가 수 증가 시 CEM 탐색 비용 상승  

***

## 4. 미래 연구 영향 및 고려 사항  

- **모델 머징**: 여러 SVF-튜닝 모델 병합 기법을 통해 전이 가능 역량 확대  
- **효율적 탐색 알고리즘**: CEM 대체 최적화 기법(메타휴리스틱) 적용으로 추론 비용 절감  
- **동적 전문가 확장**: 서비스 환경 데이터에 따라 전문가 추가·교체 위한 온라인 학습  
- **범용성 검증**: 타 대규모 LLM(예: GPT 계열) 및 비영어 과제 적용 연구  
- **안정성·안전성**: RL 보상 설계 시 편향 방지, 자가수정 메커니즘의 예측 불가능성 제어  

***
**결론**: Transformer2는 특이값 스케일링 기반 SVF와 모듈식 전문가 벡터, 두 단계 적응 메커니즘을 결합해 LLM의 **실시간·동적 과제 적응**을 실현했다. 특히 소규모 데이터로도 강건한 일반화와 효율적 전이 능력을 보여, Self-Adaptive AI 연구에 새로운 이정표를 제시한다. 앞으로 모델 머징, 효율적 탐색, 온라인 전문가 확장 등 연구가 촉진될 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2bd0db60-6079-4344-a031-9de4bb3b7bea/2501.06252v3.pdf
