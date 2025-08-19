# 1. 핵심 주장 및 주요 기여 요약  
본 논문은 사후 훈련(Post-Training)만으로 대형 언어 모델(LLM)을 1비트 수준으로 극한 양자화(quantization)하면서도 원본에 가까운 성능(예: LLaMA2-70B에서 WikiText2 perplexity 8.41)을 유지하는 BiLLM 기법을 제안한다.  
BiLLM의 주요 기여는 다음과 같다.  
- **이중 분할 기반 이진화**: 헤시안 기반으로 중요도 높은(즉, “salient”) 가중치를 구조적(컬럼 단위)으로 선택해 이진 잔차 근사(residual approximation)하고, 나머지 비중요 가중치는 최적 분할점(p*)으로 두 구간으로 나눠 각각 이진화함으로써 초저비트화 시 성능 붕괴를 방지.  
- **최적화된 구조적 선택(search)**: 헤시안 민감도 지표로 상위 1–5%의 가중치를 포함하는 컬럼 수를 블록별로 탐색해 양자화 오차를 최소화.  
- **효율성**: 7B급 모델 이진화에 0.5시간 내외로 완료 가능하며, 평균 1.07–1.11비트 수준 달성.  

# 2. 문제 정의와 제안 방법  
## 2.1 해결하고자 하는 문제  
- 기존 PTQ(Post-Training Quantization) 기법은 4–8비트 양자화에서는 성능을 유지하나, 3비트 이하 극저비트화 시 LLM의 perplexity가 급격히 악화됨.  
- 이 진입 장벽을 극복해 “진정한” 1비트 양자화를 달성하면서도 LLM의 언어생성 능력을 보존하고자 함.  

## 2.2 제안 방법 개요  
BiLLM은 각 선형 계층의 가중치 행렬 $$W\in\mathbb{R}^{n\times m}$$를 아래 세 단계로 이진화한다.  

1) 헤시안 기반 구조적 중요도 평가  

$$
   s_i = \frac{w_i^2}{[H^{-1}]_{ii}^2},\quad H=2\,X X^T
   $$  
   
- 값이 큰 상위 $$k$$개 컬럼(또는 행)을 salient 그룹으로 선정.  

2) Salient 가중치 이진 잔차 근사  
   
```math
   \begin{aligned}
   &\alpha_o^*,\,B_o^* = \arg\min_{\alpha,B}\|W_{\text{sal}} - \alpha B\|^2,\\
   &\alpha_r^*,\,B_r^* = \arg\min_{\alpha,B}\|W_{\text{sal}} - \alpha_o^* B_o^* - \alpha B\|^2,\\
   &W_{\text{sal}}\approx \alpha_o^* B_o^* + \alpha_r^* B_r^*,
   \end{aligned}
``` 
  
- 두 단계(bin1, residual bin2)로 이진화해 quantization error를 추가 삭감.  

3) Non-salient 가중치 최적 분할 이진화  
   - 남은 가중치 분포는 종 모양(bell-shaped)을 따르므로, 경계값 $$p$$로 나눠  

$$\{w:|w|\le p\}$$ 그룹과 $$\{w:|w|>p\}$$ 그룹을 각각 이진화.  
   
   - 분할점 $$p^*$$는

$$
     p^*=\arg\min_p\bigl\|W_s-\alpha_s B_s\bigr\|^2 + \bigl\|W_c-\alpha_c B_c\bigr\|^2
     $$  
     
  로 탐색(퍼센타일 서치).  

이 과정을 블록 단위로 수행하며, OBC(block-wise error compensation)를 추가 적용해 가중치 보정.  

## 2.3 모델 구조  
- Transformer의 모든 Linear 계층(Attention Q/K/V/Out, Feed-Forward FC1/FC2)에 적용.  
- 블록 크기(예: 128컬럼)마다 salient 컬럼 수 탐색, 분할점 탐색, 이진화, 보상 수행.  

# 3. 성능 향상 및 한계  
## 3.1 성능 향상  
- WikiText2 perplexity  
  - LLaMA2-70B: FP16 3.53 → BiLLM(1.08bit) 8.41 (기존 PB-LLM(1.7bit)≈28.37, GPTQ(1bit)≫)  
  - OPT-30B: FP16 10.13 → BiLLM(1.11bit) 12.71 (PB-LLM 25.14, GPTQ 15.71)  
- PTB/C4 및 Zero-Shot 과제(PIQA, BoolQ 등)에서도 1비트급 양자화 최상위 성능 달성.  
- 메모리: FP16 대비 약 9–10% 점유율(2비트 GPTQ 대비 ≈70%).  

## 3.2 한계  
- 극소수 salient 컬럼 탐색에 따른 추가 연산 및 저장(≈0.1bit) 발생.  
- 분할점 탐색은 블록별로 이루어져, 모델 크기·블록 크기에 따라 탐색 시간 변동.  
- 이진화 연산(GEMM)에 특화된 하드웨어 지원이 필요하며, 일반 GPU 상에서는 bit-packed 연산 오버헤드 존재.  

# 4. 일반화 성능 향상 가능성  
- Instruction-tuned LLM(Vicuna-7B/13B)에서도 BiLLM(1.08bit)이 GPTQ(2bit), PB-LLM(1.7bit) 대비 전 영역(WikiText2/PTB/C4)에서 perplexity 개선, 제로샷 정확도 향상 확인.  
- 다양한 LLM 패밀리(OPT, LLaMA, LLaMA2, Vicuna)에 공통 적용 가능해 범용성 높음.  
- residual 및 분할 전략은 모델 구조나 크기 변화에도 유연히 적응, 일반화 잠재력 큼.  

# 5. 앞으로 연구에 미치는 영향 및 고려 사항  
- **엣지/온디바이스 배포**: 1비트 양자화로 대형 모델을 극도로 경량화하여 저자원 환경(모바일·IoT)에서도 LLM 서비스 가능성 확대.  
- **합성곱·비(非)Transformer 모델 적용**: BiLLM의 분할·잔차 이진화 전략을 CNN, MLP 등 다양한 아키텍처에 확장 연구 필요.  
- **하드웨어 공동 설계**: bit-packed 연산과 error-compensation을 효율적으로 지원하는 전용 가속기 설계로 실사용 성능 극대화 모색.  
- **양자화 오류 분석**: 비정규 분포·보상 메커니즘이 작용하는 환경에서 quantization error 특성 연구 및 안정성 보장 기법 개발.  
- **지속적 미세조정**: 초저비트화 후 가벼운 QAT 혹은 LoRA 기반 PEFT를 결합해 세밀한 성능 회복 방안 모색.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c7ff2d18-a397-4e33-ad88-1fb94f8b1f06/2402.04291v2.pdf
