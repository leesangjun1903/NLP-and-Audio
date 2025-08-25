# LLaMA: Open and Efficient Foundation Language Models

## 핵심 주장 및 주요 기여  
**LLaMA**(Large Language Model Meta AI)는 7B–65B 파라미터 규모의 공개 가능한 언어 모델로,  
1) **공개 데이터만** 사용해 학습함에도 불구하고, (a) LLaMA-13B가 GPT-3(175B)를 대부분의 벤치마크에서 능가하고, (b) LLaMA-65B가 Chinchilla-70B 및 PaLM-540B 수준에 근접하는 성능을 달성함.  
2) **추론 효율성**을 중시하여, 작은 모델을 더 많은 토큰으로 학습시키면 동일 성능 대비 추론 속도·비용 면에서 가장 효율적인 모델을 설계할 수 있음을 보임.  

## 1. 해결하고자 하는 문제  
기존 대형 언어 모델은  
- 거대 파라미터(수십조 규모)  
- 폐쇄·비공개 데이터  
- 막대한 인퍼런스 비용  
의 한계가 있어, “연구·응용 접근성”과 “실서비스 효율성”을 저해했다.  

## 2. 제안 방법  
### 2.1. 데이터 스케일링 최적화  
- **공개말뭉치만** 활용해 1.4T 토큰 규모로 전처리·중복 제거·품질 필터링  
- 주요 코퍼스 비중: CommonCrawl(67%), C4(15%), GitHub(4.5%), Wikipedia(4.5%) 등  
- “작은 모델을 더 긴 기간 학습” 전략: Hoffmann et al.의 Chinchilla 법칙(**학습 토큰 수 ∝ 파라미터 수**)을 넘어, 7B 모델을 1T→1.4T 토큰까지 학습시켜 성능 지속 향상을 확인  

### 2.2. 모델 구조 및 학습  
- Transformer 기반, 차이점:  
  -  Pre-norm(RMSNorm)  
  -  SwiGLU 활성화  
  -  RoPE(rotary) 위치 임베딩  
- 하이퍼파라미터(예: LLaMA-13B: 5120 차원, 40 layer, batch 4M 토큰, lr 3e-4, 1.0T 토큰 학습)  
- 효율화  
  -  xformers의 causal attention 최적화  
  -  체크포인팅과 모델/시퀀스 병렬화  
  -  GPU-GPU 통신과 연산 오버래핑  

### 2.3. 성능 향상  
- **Zero-shot CommonSense**: BoolQ, PIQA, HellaSwag 등 8개서 GPT-3, Chinchilla 초월 (예: HellaSwag 84.2% vs PaLM-540B 83.4%)  
- **Closed-book QA**: NaturalQ, TriviaQA서 SOTA  
- **Reading Comprehension**: RACE 중·고등급서 PaLM-540B 상회  
- **수리·코드**: GSM8k, MATH, HumanEval 등서 Minerva·PaLM-코더 대비 경쟁력  
- **MMLU**: 65B 모델이 평균 63.4% 정확도로 Chinchilla-70B(67.5%) 대비 근접  

### 2.4. 한계  
- **책·학술 비중 부족** → MMLU 일부 도메인에서 SOTA 모델 대비 격차  
- **편향·독성**  
  -  Respectful prompt toxic score 0.141 (모델 크기↑ 독성↑)  
  -  CrowS-Pairs 평균 66.6% 편향 지수  
  -  WinoGender “gotcha” 사례서 편향 영향 명확  
- **환각(hallucination)** 위험: TruthfulQA 정확도 57%  

## 3. 일반화 성능 향상 관점  
- **데이터-모델 스케일링**: 작은 모델을 장기간 학습(토큰 과잉) → 일반화 지속 개선  
- **아키텍처 개선**(Pre-norm, SwiGLU, RoPE)으로 학습 안정성↑ → 작은 모델서도 고성능 달성  
- **효율화 기법**(메모리 절감, 통신 최적화) 덕분에 대규모 토큰 학습 실현 → 과적합 억제 및 범용성 확대  

## 4. 향후 연구 영향 및 고려사항  
- **열린 연구 생태계**: 공개 가능한 작은 고성능 모델 제공으로, 대형 회사 종속성 ↓  
- **추론 효율성 강조**: inference budget 최적화 연구 활성화  
- **일반화 vs. 특화 균형**: 특정 도메인(책·학술) 보강이 MMLU 같은 지식 집약 태스크에 필수  
- **안전성**: 편향·독성·환각 완화 기술(데이터 필터링, 디버깅·검열 레이어) 통합 필요  
- **지속가능성**: Carbon footprint 최소화 방안(그린 데이터센터, 효율적 하드웨어) 연구 병행  

――  
**요약**: LLaMA는 공개 데이터와 효율화된 아키텍처를 바탕으로, 7B–65B급 모델이 100B+급 대비 동등 수준 성능을 내는 가능성을 입증했다. 앞으로는 소형 모델 학습법, 편향·환각·탄소 배출 저감, 그리고 추론 효율화에 집중해야 한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c8d9fca9-67c8-4214-b687-beb45397ec77/2302.13971v1.pdf)
