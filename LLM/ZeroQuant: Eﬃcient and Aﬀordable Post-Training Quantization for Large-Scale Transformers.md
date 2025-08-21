# ZeroQuant: Eﬃcient and Aﬀordable Post-Training Quantization for Large-Scale Transformers

## 1. 핵심 주장 및 주요 기여 요약  
ZeroQuant는 대규모 BERT·GPT-3 스타일 트랜스포머 모델에 대해 **사후 퀀타이제이션(post-training quantization, PTQ)** 만으로도 FP16 대비 최대 5.2×의 추론 속도 향상과 3×의 메모리 절감 효과를 달성하면서, **정밀도 저하를 최소화**할 수 있음을 보인다. 주요 기여는 다음과 같다:  
- **세분화된 하드웨어 친화적 퀀타이제이션**  
  - 가중치에 그룹 단위(group-wise) INT4/INT8, 활성화에 토큰 단위(token-wise) INT8 적용  
- **레이어별 경량 지식 증류(Layer-by-Layer Knowledge Distillation, LKD)**  
  - 원본 학습 데이터 없이도 각 레이어를 순차적으로 증류하여 극저비트(INT4/8) 모델을 효율적으로 확보  
- **커널 융합 기반 추론 백엔드 최적화**  
  - 퀀타이즈·디퀀타이즈 연산을 커널 융합으로 제거하여 실제 GPU에서의 레이턴시 개선  

## 2. 문제 정의, 제안 기법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
- 대규모 트랜스포머의 메모리·연산 비용 급증  
- 기존 PTQ는 정밀도 저하가 크고, 극저비트(INT4) 적용 시 성능 붕괴  
- QAT(quantization-aware training)는 데이터·컴퓨팅 자원 요구가 과다  

### 2.2 제안 방법  
#### 2.2.1 그룹-/토큰-단위 세분화 퀀타이제이션  
- Weight quant: 행렬 W∈ℝ^{n×m}을 g그룹으로 분할하여 각 그룹별 스케일 S_g 적용  
- Activation quant: 각 토큰 t에 대해 동적 Range를 계산하여 토큰별 스케일 S_t 적용  
- 수식:  

$$
    \hat x = \mathrm{round}\bigl(\mathrm{clamp}(x/S,\,-2^{b-1},\,2^{b-1}-1)\bigr)
  $$  
  
여기서 $$b$$는 비트 수, $$S=\max|x|$$ 또는 토큰·그룹별 값  

#### 2.2.2 레이어별 지식 증류(LKD)  
- 기존 KD와 달리 전체 모델 재학습 없이, 각 레이어 $L_k$만 순차 증류  
- 증류 손실:  

```math
    \mathcal{L}_{\mathrm{LKD},k} = \mathrm{MSE}\bigl(L_k\circ\cdots\circ L_1(X)\;,\;\hat L_k\circ L_{k-1}\circ\cdots\circ L_1(X)\bigr)
```

- 장점: 단일 레이어, 단일 GPU 메모리, 레이블 불필요  

#### 2.2.3 시스템-수준 최적화  
- CUTLASS 기반 INT8 GeMM 스케줄 활용  
- 퀀타이즈·디퀀타이즈 연산을 LayerNorm·GeLU·bias-add 등과 커널 융합  
- CUDA Graph로 작은 배치 오버헤드 제거  

### 2.3 모델 구조  
- BERT-family: BERT-base/large, GLUE 상 8개 테스크  
- GPT-3 스타일: 350M·1.3B·6B·20B 파라미터 모델  
- 퀀타이즈: MHSA는 INT8, FFC는 INT4/INT8 혼합, 활성화는 INT8  

### 2.4 성능 향상  
- **추론 속도**: BERT-base 최대 5.19×, GPT-3-350M 4.16×, NeoX-20B 5.2×  
- **메모리**: 극저비트 혼합 시 FP16 대비 최대 3× 절감  
- **정밀도 유지**:  
  - BERT-base INT8/INT8: −0.2%p GLUE  
  - GPT-3-350M W8A8: zero-shot 정확도 38.9→38.7, PPL 21.5→21.7  

### 2.5 한계  
- **컴퓨터 비전 미검증**: NLP에만 적용  
- **20B 이상 초대형 모델 일반화 미확인**  
- **Self-Attention 활성화(FP16 유지) 등 추가 튜닝 필요**  

## 3. 일반화 성능 향상 관련 고찰  
- **토큰-단위 활성화 퀀타이제이션**이 다양한 입력 분포에 강인하게 작용하여, zero-shot·few-shot 테스크 전반에 걸쳐 정밀도 저하 최소화  
- **레이어별 증류**는 원본 분포에 구애받지 않고 랜덤·위키피디아 데이터로도 내부 표현(structure)을 복원함, 모델이 본 적 없는 데이터에도 일반화 가능성 시사  
- 실험: GPT-3-350M W4/8-A8 증류 시, 랜덤 데이터만으로 zero-shot 정확도 +1.1%p, PPL 92→40  

## 4. 향후 영향 및 연구 시 고려 사항  
ZeroQuant는 **대규모 사전학습 모델의 실용적 서비스**·**온디바이스 추론** 가능성을 크게 확장한다.  
- **영향**:  
  - 오픈소스·프라이빗 모델 모두에 사후 데이터 비의존적 압축 적용  
  - 대규모 모델 상용화 비용·전력 절감 가속화  
- **고려 사항**:  
  - 비전·멀티모달 모델로 확대 검증  
  - 100B 이상 모델의 토큰 분포 민감도 분석  
  - 초저비트(INT2) 적용 한계 및 동적 범위 조정 기법 연구  

***

 GLUE 평균 점수 비교 기반.  
 Table 10; GPT-3-350M W4/8A8 LKD 데이터원별 성능.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/775066fc-3146-4d5d-905c-df2fa8d2b449/2206.01861v1.pdf
