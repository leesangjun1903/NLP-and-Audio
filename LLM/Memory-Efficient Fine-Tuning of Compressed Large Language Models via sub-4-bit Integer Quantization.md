# Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
– PEQA(Parameter-Efficient and Quantization-aware Adaptation)는  
  1) 사전 학습된 LLM의 가중치를 서브 4비트 정수로 양자화(quantization)하여 메모리 사용량을 크게 줄이고,  
  2) downstream task별로 오직 양자화 스케일(scale)만 업데이트함으로써 파라미터 효율적(PEFT)인 파인튜닝을 가능케 하며,  
  3) 학습 중·추론 시 모두 메모리·연산 효율을 동시에 달성할 수 있음을 입증한다.

**주요 기여**  
1. **PEQA 방법 제안**  
   – 양자화된 정수 행렬은 고정(frozen)하고, 채널별 스케일 $$s_0$$만 학습  
   – downstream task마다 $$\Delta s$$만 교체해 빠른 task 전환  
2. **메모리·연산 효율성 검증**  
   – LLaMA-65B 기준 DRAM 사용량:  
     -  Full FT 457 GB → PEQA 33 GB  
   – 추론 가속: sub-4비트 행렬-벡터 곱 전용 커널 활용  
3. **성능 비교**  
   – QAT(Quantization-Aware Training) 대비 유사 perplexity 유지  
   – LoRA+PTQ 대비 우수한 성능(3/4비트 모두)  

***

## 2. 문제 정의, 방법론, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
– 대규모 LLM 파인튜닝 시  
  -  옵티마이저 상태 메모리(optimizer state) 부담  
  -  모델 가중치(수백 GB) 메모리 부담  
  -  추론 시 대량의 FP16/FP32 메모리 접근에 따른 지연  

### 2.2 제안 방법: PEQA  
1. **양자화(Quantization)**  

$$
   \widehat W_0 = s_0 \,\bigl(\,\mathrm{clamp}\,\bigl(\lfloor W_0/s_0\rceil + z_0,\;0,\,2^b-1\bigr) - z_0\,
   \bigr)
   $$
   
   – $$W_0$$: 사전 학습된 통상 FP16 가중치  
   – $$s_0$$, $$z_0$$: 채널별 스케일·제로포인트(초기화: $$ \|W_0 - \widehat W_0\|_F $$ 최소화)  
   – $$W^0 = \mathrm{clamp}(\lfloor W_0/s_0\rceil + z_0,0,2^b-1)-z_0$$ 고정  

2. **스케일 업데이트(Adaptation)**  

$$
   \widehat W = (s_0 + \Delta s)W^0
   $$
   
   – $$\Delta s$$: downstream task별 학습 파라미터  
   – 가중치 행렬은 정수($$W^0$$)로 고정, 스케일만 업데이트 → optimizer state 메모리 최소화  

### 2.3 모델 구조  
– 기존 LLM(Transformer 스트럭처) 그대로, 각 FC 레이어 가중치는 sub-4비트 정수 + 채널별 스케일  
– LoRA 등 PEFT와 동일하게 사후 삽입 구조가 아닌, 양자화 기반 ‘스케일’만 학습  

### 2.4 성능 향상  
– **메모리 사용**  
  -  Fine-tuning DRAM: 457 GB→33 GB  
  -  Deployment 모델 사이즈: 131 GB→33 GB  
– **추론 속도**  
  -  sub-4비트 곱셈 전용 커널 적용 시 대기 저하(메모리 I/O) 개선  
– **언어 모델링(perplexity)**  
  -  Wikitext2: LLaMA 65B 기준  
    – LoRA(16비트) 3.82 → PEQA(4비트) 4.02 (소폭 열화)  
    – LoRA+OPTQ(4비트) 4.10 → PEQA(4비트) 4.02 (향상)  
– **확장성**  
  -  GPT-Neo 2.7B~LLaMA 65B에 일관된 성능 복원  
  -  Instruction-tuning(Alpaca) 및 MMLU, commonsense reasoning에서 full-precision 대비 0.5–2%p 차이  

### 2.5 한계  
1. **극저비트(1–2비트)**  
   – sub-3비트까지 가능하나, 2비트 이하에서는 퍼포먼스 급격 저하 우려  
2. **하이퍼파라미터 민감도**  
   – 학습률·에폭 등 세부 튜닝 부족 → 큰 모델(70B)에서 최적화 미흡  
3. **외부 양자화 기법 제약**  
   – RTN 방식 기반, 다른 PTQ 기법과의 결합 효과 확인 필요  

***

## 3. 일반화 성능 향상 가능성

– **In-context learning**: PEQA로 양자화된 LLaMA 7–30B, PIQA·HellaSwag·ARC 등에서 zero-shot → +1.6%p, five-shot → +1.8%p 향상  
– **MMLU**: RTN(quantize only) 대비 instruction-tuning 후 평균 +6–7%p 복원  
– **Natural Instruction**: unseen NLP task 직접 생성 응답(ROUGE-L)에서 LoRA 대비 +2–4%p 우수  
– **모델 크기 제약 완화**: DRAM 한계 내에서도 대형 모델 일반화 평가 가능 → 더 큰 LLM 탑재·평가 확대  

***

## 4. 향후 연구 영향 및 고려 사항

– **연구 영향**  
  -  대규모 LLM 파인튜닝·배포 비용 대폭 절감  
  -  sub-4비트 양자화·PEFT 융합 연구 가속화  
  -  On-device/edge LLM 활용 가능성 제고  

– **고려 사항**  
  1. **초극저비트 안정화**: 2비트, 1비트에서도 성능 보장 방법 연구  
  2. **동적 양자화 그룹화**: 그룹 사이즈 최적화 및 mixed-precision 적용  
  3. **학습 레시피 최적화**: 대형 모델 학습률·스케줄·정규화 기법 세밀 튜닝  
  4. **다양한 PTQ 기법 통합**: SmoothQuant·AWQ 등과 결합한 PEQA 변형 가능성  

**결론**: PEQA는 sub-4비트 양자화를 활용해 LLM 파인튜닝의 메모리·추론 효율을 획기적으로 개선하면서도, 일반화 성능을 거의 완벽히 복원·향상시키는 혁신적인 PEFT 기법이다. 앞으로 더욱 다양한 양자화·튜닝 기법과 결합해, 대규모 언어 모델 활용을 한층 더 확장시킬 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0383c080-572e-412d-892e-4c3ec4caa037/2305.14152v2.pdf
