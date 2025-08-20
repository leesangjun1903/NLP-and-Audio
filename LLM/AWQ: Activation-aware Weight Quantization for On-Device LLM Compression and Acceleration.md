# AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

## 1. 핵심 주장 및 주요 기여
“AWQ” 논문은 **LLM의 모든 가중치를 균일하게 양자화했을 때 발생하는 성능 저하를 줄이기 위해, 활성화 분포에 기반해 ‘중요한’ 채널만 보호하고 나머지는 저비트로 양자화하는** 새로운 방법을 제안한다.  
- **활성화 중요도 기반 채널 선택**: 가중치 크기 대신 해당 채널에 입력되는 활성화의 평균 크기를 기준으로 중요도를 판단.  
- **채널별 스케일링**: 선택된 채널을 스케일업하여 상대적 양자화 오차를 줄이고, 역스케일을 적용해 연산 정확도 유지.  
- **TinyChat 시스템**: AWQ로 양자화된 모델을 데스크톱·모바일·임베디드 GPU/CPU에서 3× 이상의 실제 속도 향상을 보여주는 경량 추론 프레임워크를 구현.

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제
- 대형 LLM은 수백 기가바이트급 메모리를 필요로 해 엣지 디바이스 온디바이스 추론이 불가능.  
- 기존 **Post-Training Quantization**(PTQ)은 4비트 이하 저비트 양자화 시 단순 RTN(round-to-nearest)에서 퍼플렉서티가 크게 악화.  
- **GPTQ** 등은 보상용 재구성(reconstruction) 과정에서 보정 세트에 과적합(over-fitting)되어 도메인 간 일반화가 취약.

### 2.2 제안하는 방법
1) **채널별 활성화 기반 중요도 산정**  
   - 각 입력 채널의 활성화 $$X$$에 대해 평균 절댓값 $$s_X = \frac{1}{|X|}\sum |X|$$ 산출.  
   - 중요도 상위 0.1–1% 채널을 ‘살려두기’ 위한 기준으로 활용.  

2) **스케일링을 통한 양자화 오차 저감**  
   - 일반 양자화:  

$$
       Q(w) = \Delta \cdot \mathrm{Round}\Bigl(\frac{w}{\Delta}\Bigr),\quad
       \Delta = \frac{\max|w|}{2^{N-1}}
     $$
   
   - 스케일 적용: $$w' = w \cdot s,\; x' = x / s$$일 때

$$
       Q(w')(x') 
       = \Delta' \cdot \mathrm{Round}\Bigl(\frac{w\,s}{\Delta'}\Bigr)\cdot \frac{x}{s}
       \approx \Delta \cdot \mathrm{Round}\Bigl(\frac{w}{\Delta}\Bigr)\,x\times\frac{1}{s}
     $$
   
   - $$s>1$$인 경우 **살아남은 채널**의 상대 오차 $$\propto 1/s$$만큼 감소.  
   - 반대로, $$s$$가 너무 크면 비살중 채널에서 오히려 오차가 증가하므로 적정 $$s$$를 탐색해야 함.  

3) **효율적 최적 스케일 탐색**  
   - 스케일 $$s = s_X^\alpha$$ 형태로 단일 하이퍼파라미터 $$\alpha\in[0, 그리드 탐색.[1]
   - 손실 함수  

$$
       L(s) = \big\|Q(W\cdot\mathrm{diag}(s))\,( \,\mathrm{diag}(s)^{-1}X) - W X\big\|_F
     $$
   
   - 오차를 최소화하는 $$\alpha^*$$를 찾음.

### 2.3 모델 구조 및 구현
- **양자화**: 3-bit 및 4-bit 그룹 단위(그룹 크기 128) 양자화.  
- **TinyChat**:  
  - 커널 퓨전, SIMD-aware 패킹, 온더플라이(de)양자화, KV 캐시 통합 등으로 GPU/CPU에서 실전 성능 달성.  
  - LLaMA·Llama-2·OPT·Mistral·Mixtral·Vicuna·OpenFlamingo·LLaVA·CodeLlama 등 다양한 LLM/VLM 지원.

### 2.4 성능 향상
- **언어 모델링**: WikiText-2 퍼플렉시티  
  - RTN 대비 AWQ: OPT-6.7B에서 PPL 23.54→11.39, INT4-g128에서도 SOTA 수준[Table 3][Table 4].  
- **지시문 인식 모델(Vicuna)**: GPT-4 평가에서 RTN/GPTQ 대비 승률 증가.  
- **멀티모달 캡셔닝(OpenFlamingo-9B)**: COCO CIDEr 점수 제로-샷 63.73→62.57, 32-샷 79.74→80.53[Table 6].  
- **코딩·수학 과제**: MBPP, GSM8K에서도 FP16 대비 동등한 pass@1·pass@10 성능 유지[Table 8].  
- **실제 추론 속도**: RTX 4090 데스크톱 GPU에서 Huggingface FP16 대비 3.2–3.9×, Jetson Orin에서 3.5× 가속[Figure 9].

### 2.5 한계
- **초저비트(INT2)**: RTN 단독은 수행 불가, AWQ+GPTQ 조합으로만 한계적 성능 달성[Table 9].  
- **스케일 최적화 민감도**: $$\alpha$$ 그리드 탐색에 따른 오버헤드.  
- **하드웨어 제약**: 플랫폼별 SIMD 폭·커널 특성에 맞춘 추가 최적화 필요.

## 3. 모델의 일반화 성능 향상 관점
- **회귀·재구성 불필요**: GPTQ의 재구성 단계 없이 단순 활성화 통계만 사용 → **Calibration set 과적합 회피**.  
- **데이터 효율성**: 10× 작은 보정 샘플로도 안정적 성능[Figure 8(a)].  
- **분포 강건성**: PubMed↔Enron 간 이종 분포 전이 시 AWQ PPL 변화 0.5에 불과(GPTQ는 2.3–4.9)로 **높은 도메인 일반화**[Figure 8(b)].

## 4. 향후 연구에 대한 시사점 및 고려 사항
- **하이브리드 양자화**: AWQ와 GPTQ를 결합해 **극저비트(INT2)에서도 성능 확보** 가능성.  
- **동적 활성화 기반 스케일링**: 입력 시점마다 $$s_X$$를 실시간 계산해 적응형 스케일링 적용 연구.  
- **모델 구조 확대**: Mixture-of-Experts, Retrieval-Augmented Generation 등 복합 구조에 대한 AWQ 적용성 평가.  
- **하드웨어 공정**: ASIC·NPU 전용 AWQ 친화적 명령어·메모리 레이아웃 설계.  
- **보정 세트 설계**: 소수 샘플로도 더 안전한 $$\alpha$$ 탐색을 위한 메타-러닝 또는 베이지안 최적화 적용.  

AWQ는 **활성화 기반 중요도 추정**과 **채널 스케일링**을 통해 LLM 양자화의 성능·일반화 한계를 효과적으로 극복했으며, 엣지 디바이스 온디바이스 AI 시대 핵심 기술로 자리매김할 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0b258463-53cc-4819-ad51-278aea5e1aca/2306.00978v5.pdf
