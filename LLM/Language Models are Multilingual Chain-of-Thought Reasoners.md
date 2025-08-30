# Language Models Are Multilingual Chain-of-Thought Reasoners

## 1. 핵심 주장 및 주요 기여  
이 논문은 대형 언어 모델(Large Language Models, LLMs)이 단일 언어를 넘어 **다중 언어(multilingual)**에서도 **Chain-of-Thought(이하 CoT)** 추론 능력을 발휘함을 입증한다.  
- 영어 기반 수학 추론 벤치마크 GSM8K를 10개 언어로 수작업 번역한 **MGSM(Multilingual Grade School Math)** 벤치마크를 제안.  
- PaLM-540B 및 GPT-3 모델에 CoT 프롬프트를 적용한 결과, 영­미 고자원 언어뿐 아니라 벵골어·스와힐리어 등 저자원 언어에서도 최대 40–60% 정확도를 달성.  
- **EN-CoT(영어 추론 단계)→NATIVE-CoT(모국어 추론 단계) 간 성능 차이가 크지 않음**을 확인, 영어 CoT가 효과적인 언어 간 전이 베이스라인임을 제시.  
- MGSM 외에도 인과 추론(XCOPA)·어휘 의미 판단(XL-WiC)까지 실험을 확장해, 다중언어 CoT 프롬프트가 전반적 성능을 크게 끌어올림을 보임.

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 해결하고자 하는 문제  
- 기존 CoT 연구는 **영어**에서만 검증되었으나, 실제 응용 환경은 다언어.  
- 비영어권 언어에서 LLM의 복잡한 **다중 단계 추론(multi-step reasoning)** 능력 미검증.  

### 2.2 제안 방법  
1. **MGSM 벤치마크**  
   - GSM8K의 250문제를 10개 언어(영어 제외)로 수작업 번역.  
   - 난이도별(2–8단계) 문제 분포 확보.  
2. **CoT 프롬프트 변형**  
   - DIRECT: 추론 단계 없이 정답 직예측  
   - NATIVE-CoT: 모국어로 단계별 추론 단계 기술  
   - EN-CoT: 모국어 문제→추론 단계는 영어  
   - TRANSLATE-EN: 문제→영어 번역 후 영어 CoT  
3. **Few-shot 샘플**  
   - NATIVE-Exemplars(모국어 예제) vs. Multilingual-Exemplars(6개 주요 언어 예제) vs. English-Exemplars

### 2.3 모델 구조  
- **PaLM**: Pathways 아키텍처 기반, 최대 540B 파라미터  
- **GPT-3**: autoregressive Transformer, 최대 davinci-002  
- 모두 greedy decoding(τ=0)  

### 2.4 수식 (예시)  
- CoT 단계 수 k, 문제 i에 대해  

```math
    \hat y_i = \arg\max_y P(y \mid  
      \underbrace{\text{Prompt} \oplus  
      \sum_{j=1}^k \text{Exemplar}_j}_{\text{few-shot context}})
``` 

- EN-CoT vs. NATIVE-CoT 전이 차이:  

```math
    \Delta_{\text{lang}} = \text{Acc}_{\text{EN-COT}} - \text{Acc}_{\text{NATIVE-COT}}
```

### 2.5 성능 향상  
- **MGSM**  
  - PaLM-540B TRANSLATE-EN: 평균 55% 정확도(언어 빈도와 무관)  
  - EN-CoT vs NATIVE-CoT: 평균 차이 3pp 이내  
  - 저자원 언어(SW·BN): 44–51% 달성  
- **XCOPA** (인과 추론)  
  - 기존 RoBERTa(translate-test) 대비 +13pp 우위, PaLM-540B EN-CoT 89.9%  
- **XL-WiC** (문맥 의미 판단)  
  - XLM-R 대비 다수 언어에서 근접 내지 상회  

### 2.6 한계  
- **프롬프트 의존성**: 최적 예제 수·언어 선택이 실험별 상이  
- **계산 비용**: 대형 모델·CoT 단계 길이 증가 시 추론 비용 급증  
- **추론 해설의 품질**: 단순 산수 외 복잡 추론으로 확장 시 성능 미확인  
- **저자원 언어 번역 오류** 가능성  

## 3. 모델의 일반화 성능 향상 가능성  
- **언어 빈도 독립성**: 저자원 언어 성능 저하폭 < 3pp, 고자원 언어 전이 가능성  
- **모델 스케일**: PaLM-62B 이상에서 CoT 성능 급등, 추가 스케일링 시 일반화 강화 전망  
- **프롬프트 유형 유연성**: Multilingual-Exemplars도 English-Exemplars 대비 경쟁력, 리소스 제한 환경에서도 적용 가능  

## 4. 향후 연구에 미치는 영향 및 고려 사항  
- **Cross-lingual CoT**: 영어 중심 프롬프트 설계가 다언어 추론 연구 표준으로 자리잡을 전망  
- **프롬프트 최적화**: 예제 수·언어 조합·위치 탐색을 통한 자동화 기법 필요  
- **저자원 언어 데이터 확보**: 고품질 수작업 번역 외 언어적 다양성 보강  
- **추론 설명 평가**: CoT 단계 논리 일관성·정확성 자동 평가 메트릭 개발  
- **모델 경량화**: 실시간 다언어 추론을 위한 경량·지연 최소화 아키텍처 연구  

***
**주요 기여**: 다언어에서 CoT 추론 가능성 입증·벤치마크 공개·영어 CoT의 일반화 효용성 확립.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/bbd6cf10-e729-4154-b966-dc2558589e8b/2210.03057v1.pdf)
