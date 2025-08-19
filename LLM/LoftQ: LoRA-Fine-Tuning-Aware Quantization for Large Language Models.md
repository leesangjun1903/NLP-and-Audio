# LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models

## 핵심 주장 및 주요 기여  
LoftQ는 **양자화(quantization)**와 **LoRA(저차원 적응) 파인튜닝**을 동시에 고려하여 사전학습된 대형 언어 모델의 성능 저하를 완화하는 새로운 프레임워크이다.  
- 사전학습 가중치 $$W$$를 단순히 양자화한 뒤 LoRA를 적용하는 기존 QLoRA와 달리, LoftQ는  

$$
    \min_{Q,A,B} \|W - Q - A B^\top\|_F
  $$
  
  를 목적함수로 하여 양자화 결과 $$Q$$와 LoRA 어댑터 $$(A,B)$$를 **교대로 최적화**한다.  
- 이렇게 얻은 초기화는 양자화로 인한 오차를 크게 줄여, 특히 **2-bit 저정밀도** 환경에서 downstream 일반화 성능을 크게 향상시킨다.

## 해결하고자 하는 문제  
- 저비트 양자화(2-4 bit) 후 LoRA 파인튜닝 시, 양자화로 인한 가중치 왜곡이 LoRA 초기화(제로)와 부조화를 일으켜 성능이 급격히 저하됨.  
- 기존 QLoRA는 Quantize-then-LoRA 방식으로, 양자화 오차를 보정할 메커니즘이 없어 특히 2-bit 환경에서 수렴 실패나 성능 급락을 보임.

## 제안 방법  
### 공동 최소화 문제  
사전학습 가중치 $$W\in\mathbb{R}^{d_1\times d_2}$$에 대하여  

$$
  \min_{Q\in\mathbb{R}_N^{d_1\times d_2},\,A\in\mathbb{R}^{d_1\times r},\,B\in\mathbb{R}^{d_2\times r}}
    \|\,W - Q - A B^\top\|_F^2
$$  

를 최적화한다. 여기서 $$Q$$는 $$N$$-bit 양자화 함수 $$q_N(\cdot)$$의 역함수로 재구성 가능한 정수 표현, $$(A,B)$$는 LoRA 저차원 어댑터이다.

### 교대 최적화 알고리즘 (Alternating Optimization)  
1. 초기화: $$A^{(0)}=0,\ B^{(0)}=0$$  
2. $$t$$-th 단계  
   - **양자화 단계**:  

$$
        Q^{(t)} = q_N\bigl(W - A^{(t-1)}B^{(t-1)\top}\bigr)
      $$  
   
  - **SVD 단계**:

$$
        R^{(t)} = W - Q^{(t)} = U\Sigma V^\top,\quad
        A^{(t)} = U_{[:,1:r]}\sqrt{\Sigma_{1:r,1:r}},\ 
        B^{(t)} = V_{[:,1:r]}\sqrt{\Sigma_{1:r,1:r}}
      $$  

3. 원하는 반복 횟수 $$T$$만큼 수행 후 $$(Q^{(T)},A^{(T)},B^{(T)})$$를 LoRA 파인튜닝 초기화로 사용.

### 모델 구조 적용  
- Transformer의 MHA 및 FFN 내부의 모든 선형층 가중치에 대해 LoftQ를 수행.  
- 2-4 bit 양자화(Uniform, NF4)를 적용하며, LoRA rank는 하이퍼파라미터(예: 8, 16, 32 등)로 설정.

## 성능 향상  
| 모델 유형 | 환경            | 메서드   | 주요 성능 지표                            |
|-----------|-----------------|---------|-------------------------------------------|
| DeBERTaV3 | 2-bit Uniform   | LoftQ    | MNLI-m ↑ 8%p (88.0% vs. 79.9%)             |
|           | 2-bit NF2       | LoftQ    | SQuAD F1 ↑ ~17%p (84.4 vs. 69.5)           |
| BART-large| 4-bit NF4       | LoftQ    | XSum ROUGE-1/2 ↑ +1.1/0.8                  |
|           | 2-bit NF2       | LoftQ    | XSum ROUGE-1 ↑ ~23 points (39.6 vs. N.A.) |
| LLAMA-2   | 2-bit NF4       | LoftQ    | WikiText-2 Perplexity ↓ from N.A. to 7.85  |
|           | mixed-prec.     | LoftQ    | GSM8K Acc ↑ +12.7%p (LLAMA-2-13b)          |

- **저비트(2-bit) 환경에서 특히 두드러진 성능** 개선 및 안정적 수렴.  
- QLoRA가 수렴하지 못하는 경우에도 LoftQ는 일관된 일반화 성능 보장.

## 일반화 성능 향상 요인  
- **양자화 오차와 LoRA 어댑터 보정**을 공동 최적화하여, “초기화 시점”에 사전학습 가중치와의 불일치(Discrepancy)를 최소화.  
- 초기화 차이가 작아질수록 downstream fine-tuning 시 **빠른 수렴**과 **과적합 방지**에 기여.  
- 낮은 비트폭에서의 양자화가 암묵적 정규화 효과를 제공, 대형 모델일수록 과적합 경향 완화에 유리.

## 한계 및 고려 사항  
- **계산 비용**: 대형 모델(예: LLAMA-2-13b)의 경우 개별 행렬에 SVD 수행 시 수십 초 소요.  
- **반복 횟수 $$T$$**: 1~5 회 이상 반복해도 성능 향상은 한계에 다다름(수익 체감).  
- **Rank 선택**: 과도한 낮은 rank는 근사 오차를 키워 성능 저하를 유발할 수 있음.

## 향후 연구 영향 및 고려할 점  
- **PTQ(사후 훈련 양자화) vs. QAT(양자화 인식 훈련)** 사이에 효율적 대안 제시: LoftQ는 경량화된 LoRA와 결합한 PTQ로, 대규모 LLM 서비스 환경에 적합.  
- **혼합 정밀도 양자화** 활용 연구 촉진: 계층별 비트폭 최적화, 자동 비트폭 할당 알고리즘 개발 가능.  
- **확장성 개선**: SVD 병목 완화 위한 근사 SVD 기법, GPU 가속화, 분산 처리 설계 필요.  
- **통합 어댑터 기법** 탐색: LoRA 외 다른 PEFT 기법(예: Prefix-Tuning)과 LoftQ 통합으로 범용성 강화.  
- **응용 분야 확대**: 비자연어(음성, 컴퓨터 비전) 대형 모델 양자화-PEFT에도 LoftQ 원리 적용 가능성.

LoftQ는 **저정밀 양자화와 LoRA 파인튜닝을 통합적으로 고려**함으로써, 대형 언어 모델의 경량화와 일반화 성능 확보 양립을 앞당기는 주요 이정표로 작용할 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/78296418-d2ce-4ae4-a44c-11abf0cf9c7c/2310.08659v4.pdf
