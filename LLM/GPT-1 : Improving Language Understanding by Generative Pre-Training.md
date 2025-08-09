# Improving Language Understanding by Generative Pre-Training

**핵심 주장 및 주요 기여**  
“Improving Language Understanding by Generative Pre-Training” 논문은 대규모 비지도 언어 모델의 사전학습(pre-training)을 통해 다양한 자연어 이해(NLU) 과제에서 최첨단 성능을 달성할 수 있음을 보인다.  
- 다양한 NLU 태스크에 대해 하나의 범용 Transformer 기반 언어 모델을 사전학습하고, 최소한의 구조 변경만으로 각 과제에 미세조정(fine-tuning)하여 최상의 성능을 얻음.  
- 12개 벤치마크 중 9개 과제에서 이전 최고 기록을 크게 개선.  
- 공통의 모델 아키텍처와 학습 절차를 사용함으로써 task-specific 설계의 필요성을 최소화하고, 비지도 데이터를 효율적으로 활용할 수 있음을 입증.  

***

## 1. 해결하고자 하는 문제  
- **지도학습 의존성**: 문장 추론, 질문 응답, 문서 분류 등 NLU 과제에 필요한 라벨링 데이터는 희소하고 비용이 높음.  
- **전이 학습의 한계**: 기존에는 word embedding 수준의 전이만 활용하거나, 과제 별로 특수 설계된 모델을 사용해야 했음.  
- **장기 의존성 처리**: LSTM 계열 모델은 긴 문맥을 효과적으로 학습하기 어려움.  

***

## 2. 제안 방법  
### 2.1. 비지도 사전학습 (Unsupervised Pre-Training)  
- 대규모 텍스트 말뭉치(BooksCorpus)를 사용하여 표준 언어 모델링 목표를 최대화:  

$$
L_{\mathrm{LM}}(U) \;=\; \sum_{i=1}^{n} \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)
$$  

- Transformer 디코더 구조(12-layer, 12-head, 768-dim, FFN 3072-dim)를 활용해 long-range dependency 학습.  

### 2.2. 지도 미세조정 (Supervised Fine-Tuning)  
- 사전학습된 파라미터 Θ를 초기화 값으로 사용하고, 과제별 레이블 예측층 $$W_y$$만 추가.  
- 입력 구조가 다른 과제들(문장 쌍, 문맥+질문+답안 후보 등)을 단일 연속 토큰 시퀀스로 변환:  
  - **텍스트 분류**: [⟨s⟩; 문장; ⟨e⟩]  
  - **텍스트 함의**: [⟨s⟩; 전제; \$; 가설; ⟨e⟩]  
  - **질문 응답/스토리 완성**: [⟨s⟩; 문맥; \$; 질문; \$; 답안 후보; ⟨e⟩]  
- 손실 함수에 보조 언어 모델링 항을 결합하여 일반화 성능 및 수렴 가속화:  

$$
L_{\mathrm{FT}}(C) = \sum_{(x,y)\in C} \log P(y\mid x;\Theta) + \lambda \sum_{(x,y)\in C} \sum_{i}\log P(x_i\mid x_{ < i }\Theta)
$$  

***

## 3. 모델 구조  
- **Transformer Decoder-only**:  
  - 12개 self-attention 레이어, 각 레이어 12개 헤드  
  - 임베딩 차원 768, FFN 차원 3072  
  - 위치 임베딩 학습, BPE(40K), 드롭아웃 및 Adam+Cosine 스케줄  
- **미세조정**: 토큰 임베딩과 Transformer 파라미터는 고정 없이 전부 업데이트, 분류층만 추가  

***

## 4. 성능 향상 및 한계  
### 4.1. 주요 성능 향상  
- **Commonsense Reasoning (Story Cloze)**: +8.9% 절대 개선  
- **Question Answering (RACE)**: +5.7% 절대 개선  
- **Textual Entailment (MultiNLI)**: +1.5% 절대 개선  
- **GLUE 종합 점수**: 72.8 → 이전 최고 68.9 (CoLA, QQP, STS-B 등 다수 과제에서 SOTA 달성)  

### 4.2. 한계  
- 소규모 데이터셋(RTE 등)에서는 multi-task 학습이나 앙상블이 필요할 수 있음.  
- 사전학습과 미세조정 모두 계산 비용이 높아 자원 제약 환경에 적용 어려움.  
- 완전 zero-shot 설정에서 일부 과제(예: RTE) 성능 낮음.  

***

## 5. 일반화 성능 향상 관점  
- **보조 언어 모델링**: 미세조정 손실에 언어 모델링 항을 포함함으로써 더 안정적인 파라미터 업데이트 및 과적합 억제.  
- **모든 레이어 전이**: 하위 임베딩부터 상위 표현까지 전부 전이할 때 최상의 전이 성능 관찰.  
- **Transformer inductive bias**: LSTM 대비 낮은 분산과 높은 zero-shot 성능, 긴 문맥 이해 능력 강화.  

***

## 6. 향후 연구에 미치는 영향 및 고려 사항  
- **범용 사전학습 모델**: 이후 BERT, GPT-2/3 등 거대 언어 모델의 발전 방향 제시.  
- **미세조정 전략 최적화**: 보조 목표, 입력 변환, 학습 스케줄 등 다양한 요소의 조합이 일반화 성능에 미치는 영향 추가 연구 필요.  
- **효율성 개선**: 저자원 환경을 위한 경량화 사전학습, 지식 증류(distillation) 및 프루닝 연구 중요.  
- **Zero/Few-Shot 학습**: 사전학습 단계에서 직접 과제 접근성을 높이는 prompting, meta-learning 기법과의 결합 연구 가치.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b5e08da1-6a31-4be3-be6b-29e8f49954b0/language_understanding_paper.pdf
