# BiBERT: Accurate Fully Binarized BERT

## 1. 핵심 주장과 주요 기여 간결 요약  
BiBERT는 BERT를 1비트 가중치·임베딩·활성화로 완전 이진화(full binarization)하면서도 대규모 성능 저하를 극복한 최초의 모델이다.  
- **정보 손실 제거**를 위한 Bi-Attention 구조 제안  
- **최적화 방향 불일치**를 해결하는 Direction-Matching Distillation(DMD) 기법 도입  
- GLUE 벤치마크에서 기존 초저비트 양자화 모델 대비 평균 약 20% 이상의 성능 향상  
- FLOPs 56.3×, 모델 크기 31.2× 절감  

## 2. 문제 정의·제안 방법·모델 구조·성능 및 한계 상세 설명  

### 2.1 해결하려는 문제  
- 1비트 이진화 시 주된 성능 저하는  
  1) **Attention 정보 손실**: sign(softmax)로 인해 이진화된 어텐션 가중치가 모두 동일해져 정보 엔트로피가 0으로 붕괴  
  2) **최적화 방향 불일치**: 기존 MSE 기반 지식 증류(distillation)에서 1비트 양자화된 활성화와 풀프리시전 활성화 간의 부호 비교가 방향 오류 누적  

### 2.2 제안 방법  
1) **Bi-Attention**  
   - softmax를 제거하고 attention score $$A = \frac{1}{\sqrt{D}}(B_Q \otimes B_K^\top)$$를 bool 함수  

$$
       \mathrm{BA} = \mathrm{bool}(A) = 
       \begin{cases}
         1 & A \ge 0,\\
         0 & A < 0,
       \end{cases}
     $$
   
   - Bitwise-Affine Matrix Multiplication(BAMM) 연산 $$\boxtimes$$로 효율적 1-bit 연산  
   - 정보 이론적으로 엔트로피 최대화를 보장  

2) **Direction-Matching Distillation (DMD)**  
   - 기존의 attention score 증류 대신 **Q, K, V** 입력 활성화를 거리 행렬(similarity matrix)로 변환하여 지식 전달  

$$
       P_Q = \frac{Q Q^\top}{\|Q Q^\top\|},\quad
       P_K = \frac{K K^\top}{\|K K^\top\|},\quad
       P_V = \frac{V V^\top}{\|V V^\top\|}
     $$
   
   - MSE 대신 유클리드 거리로 최적화 방향 일치 유도  
   - 예측층(distillation) 및 히든 상태 증류와 결합  

### 2.3 모델 구조  
- BERT-BASE(12 layer, hidden=768, heads=12) 완전 이진화  
- 임베딩·Multi-Head Attention·Feed-Forward Network 모두 1비트, 단 위치·토큰 임베딩과 출력층은 FP 유지  

### 2.4 성능 향상  
- GLUE 평균 정확도: 32-32-32 FP 대비 83.9% → BiBERT 63.2% (1-1-1) → 기존 BinaryBERT(1-1-1) 41.0%  
- SST-2: 93.2% → 88.7%(+11.1%p), MRPC: 86.3% → 72.5%(+21.5%p), RTE: 72.2% → 57.4%(+6.6%p) 등  
- TinyBERT4L/6L에서도 BERT-BASE 기존 양자화 모델 성능 상회  
- FLOPs 22.5G → 0.4G(56.3×↓), 모델 크기 418MB → 13.4MB(31.2×↓)  

### 2.5 한계  
- 대규모 파라미터 환경에서 distillation 오버헤드  
- 일부 문장 이해 과제(MNLI, STS-B)에서 2-8-8 비트 양자화 모델에는 소폭 뒤처짐  
- 추가적인 학습 안정화 및 프리트레이닝 단계 부재  

## 3. 일반화 성능 향상 관점  
- **Bi-Attention**으로 다양한 입력에 대한 적응적 헤드 패턴 학습 가능  
- **DMD**의 similarity matrix 증류 방식은 다른 이진화·양자화 모델에도 적용 가능  
- 경량화된 구조로 도메인 적응·전이 학습에서 빠른 튜닝 및 과적합 방지 기대  

## 4. 향후 연구 영향 및 고려점  
- **엔트로피 기반 이진화**: 다른 트랜스포머 모델(e.g., RoBERTa, GPT)에도 Bi-Attention 확장  
- **지식 증류 방법**: similarity matrix 증류를 결합한 다중 테스크·다중 모델 증류 연구  
- **학습 안정화**: 프리트레이닝 단계에서의 이진화 기법 통합, curriculum learning  
- **하드웨어 최적화**: BAMM 연산 ISA 지원 및 극한 저전력 디바이스 적용  

   
BiBERT는 극단적 이진화 극복을 위한 **정보 이론 기반 구조 설계**와 **방향 일치 증류**라는 새로운 패러다임을 제시하며, 향후 초저비트 트랜스포머 연구에 중요한 방향성을 제공한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/66e4654a-6e90-4d1c-a93c-e278888ca894/2203.06390v1.pdf
