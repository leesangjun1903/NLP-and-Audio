# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## 핵심 주장 및 주요 기여  
“BERT”(Bidirectional Encoder Representations from Transformers)는 양방향 문맥 정보를 심층적으로 학습할 수 있는 Transformer 기반 언어 모델로, 사전학습(pre-training)과 미세조정(fine-tuning)만으로 다양한 NLP 과제에서 최고 성능을 달성한다. 주요 기여는 다음과 같다.  
- 마스크 언어 모델(MLM)과 차기 문장 예측(NSP)이라는 두 가지 무감독사전학습 과제를 통해 양방향 컨텍스트를 학습  
- 최소한의 태스크별 구조 변경만으로 GLUE, SQuAD, SWAG 등 11개 벤치마크에서 최상위 성능 경신  
- BERTBASE(110M 파라미터)와 BERTLARGE(340M 파라미터) 크기의 모델을 제시하고, 대규모 모델이 소규모 데이터셋에서도 일반화 성능을 크게 향상시킴  

## 1. 해결하고자 하는 문제  
기존 언어모델(OpenAI GPT, ELMo 등)은 문맥을 단방향으로만 인코딩하거나, 양방향 정보를 얕게 결합하여 심층적 연산에서 문맥 정보가 제한된다. 특히 문장 이해, 질의응답 등 토큰 수준 과제에서 좌·우 컨텍스트를 통합할 필요가 있다.

## 2. 제안하는 방법  
### 2.1. 모델 구조  
- Transformer 인코더(L층, H차원, A헤드) 기반  
- 입력: WordPiece 토크나이저(30K 어휘), [CLS], [SEP], 세그먼트·포지션 임베딩  
- 출력:  
  - 문장 분류 → [CLS] 토큰의 최종 히든 상태 $$C \in \mathbb{R}^H$$  
  - 토큰 예측 → 각 토큰 히든 상태 $$T_i \in \mathbb{R}^H$$  

### 2.2. 사전학습 과제  
1. Masked Language Model (MLM)  
   - 입력 문장의 15% 토큰을 임의로 선택해  
     - 80%: [MASK]로 교체  
     - 10%: 무작위 토큰으로 교체  
     - 10%: 원본 토큰 유지  
   - 손실 함수: 마스크된 위치 $$i$$에서 원래 토큰 $$w_i$$ 예측의 교차엔트로피  
2. Next Sentence Prediction (NSP)  
   - 50%: 실제 연속 문장 쌍 (IsNext), 50%: 무작위 문장 쌍 (NotNext)  
   - [CLS] 표현 $$C$$를 통해 문장 연결 여부 이진 분류  

### 2.3. 수식  
마스크 언어 모델 손실:  

$$
\mathcal{L}\_\text{MLM} = -\sum_{i \in M} \log P(w_i \mid \text{context})
$$  

차기 문장 예측 손실:  

$$
\mathcal{L}_\text{NSP} = -\bigl[y\log\sigma(W C) + (1-y)\log(1-\sigma(W C))\bigr]
$$  

전체 손실: $$\mathcal{L} = \mathcal{L}\_\text{MLM} + \mathcal{L}_\text{NSP}$$

## 3. 성능 향상 및 일반화  
- **GLUE**: BERTLARGE 기준 평균 82.1 → 80.5(OpenAI GPT) 대비 +7.3pt 향상  
- **SQuAD v1.1**: F1 93.2, EM 87.4 (단일 모델)  
- **SWAG**: 정확도 86.3 → +8.3%p(OpenAI GPT 대비)  
- 소규모 데이터(RTE, CoLA)에서 대규모 모델일수록 성능 개선폭 큼  
- **일반화 성능**: 사전학습 양방향 모델이 LTR 모델 대비 다양한 다운스트림 과제에 강건한 성능 향상을 보임  

## 4. 한계  
- 대규모 연산 자원 필요(수십 TPU·GPU, 수일간 학습)  
- NSP 과제가 일부 태스크에는 효과적이지만, 모든 경우에 최적은 아님  
- 512 토큰 제약으로 장문 처리 한계  

## 5. 향후 연구에 미치는 영향 및 고려 사항  
- **양방향 사전학습 패러다임 확산**: 후속 모델들(RoBERTa, ALBERT 등)이 학습 데이터·절차 개선  
- **자원 효율화**: 지식 증류, 경량화 모델 연구 시 BERT 구조·학습 목표 활용  
- **긴 문맥 처리**: 512 토큰 이상을 다루는 Longformer, Transformer-XL 연구 고려  
- **사전학습 과제 다양화**: NSP 대체 또는 추가 과제(연속 문단 예측, 패러프레이즈 학습 등) 탐색  

BERT는 심층 양방향 컨텍스트 학습의 중요성을 입증하며 NLP 전 분야의 사전학습 모델 연구 방향을 혁신적으로 전환하였다. 앞으로 효율성과 확장성을 균형 있게 고려한 후속 연구가 필요하다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a171fd07-39b4-483e-9a87-c6cd2ea0533b/1810.04805v2.pdf
