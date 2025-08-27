# Can ChatGPT Understand Too? A Comparative Study on ChatGPT and Fine-tuned BERT
# 핵심 요약

**“Can ChatGPT Understand Too?”** 연구는 ChatGPT의 자연어 이해 능력을 GLUE 벤치마크를 통해 평가하고, 전통적인 BERT 계열 모델들과 비교 분석한 최초의 체계적 비교 연구이다.  
주요 기여는 다음과 같다.  
1. ChatGPT는 자연어 추론(NLI) 과제에서 BERT 계열 모델들을 크게 상회하는 탁월한 추론 능력을 보임.  
2. 문장 유사도 및 패러프레이즈 과제에서는 부정 샘플에 취약해 BERT 대비 성능이 크게 저하됨.  
3. 수동 체인오브쏘트(prompting)와 소수 샷 학습 기법을 도입해 ChatGPT의 이해 능력을 평균 7.5%p까지 향상시키고, 일부 과제에서는 RoBERTa-large를 능가함.  

# 1. 문제 정의

기존 연구들은 GPT 스타일 모델이 생성(generation) 능력에서 우수하나 이해(understanding) 과제에서는 BERT 계열 모델에 못 미친다고 보았다.  
이 연구는 “ChatGPT가 NLU 과제에서도 경쟁력 있는 이해 능력을 갖추었는가?”라는 질문을 다음 세부 과제로 나누어 검증한다.

1. GLUE 벤치마크의 8개 대표 과제(CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE)에 대한 ChatGPT의 제로샷 성능 측정  
2. BERT-base, BERT-large, RoBERTa-base, RoBERTa-large의 미세조정(fine-tuning) 성능과 비교  
3. 표준 few-shot prompting, zero-shot CoT, 수동 few-shot CoT 등 고급 프롬프트 전략을 적용해 ChatGPT 이해 능력 개선  

# 2. 제안 방법

## 2.1 모델 구조 및 설정  
- ChatGPT: InstructGPT 기반의 대화형 LLM  
- 비교 대상: BERT-base/large, RoBERTa-base/large (전통적 encoder-only PLM)  

## 2.2 평가 구성  
- 데이터: GLUE 검증 세트에서 클래스별 25개(STS-B는 50개) 샘플 무작위 추출  
- 지표:  
  - 분류 과제: 정확도(Accuracy), MRPC/QQP는 추가 F1  
  - CoLA: Matthew’s correlation  
  - STS-B: Pearson/Spearman 상관계수

## 2.3 고급 프롬프트 전략  
1. **Standard few-shot**: 1-shot/5-shot 예시  
2. **Zero-shot CoT**: “Answer step by step” 템플릿으로 추론 유도  
3. **Manual few-shot CoT**: 인간 설계 중간 추론 단계를 포함한 1-shot/5-shot  

# 3. 주요 실험 결과 및 분석

| 과제 유형           | ChatGPT 제로샷 vs. BERT-base                  | 주요 분석                             |
|---------------------|-----------------------------------------------|---------------------------------------|
| 추론(NLI: MNLI, RTE) | +4–20%p 성능 개선, 모든 BERT 모델 능가         | 강력한 추론 능력, 정확한 관계 판단         |
| 패러프레이즈(MRPC)   | –24%p 급락 (부정 클래스)                      | 세밀한 의미 차이 감지 미흡               |
| 유사도(STS-B)       | –18.5%p (평균 절대 오차)                      | 경계점(2.5점) 부근 극적 오차                |
| 감정분석/QA(CoLA,SST-2,QNLI) | BERT-base와 대등                             | 단일문장 분류 과제는 안정적 성능 유지        |

## 3.1 성능 향상: Manual few-shot CoT  
- 전체 GLUE 평균: 78.7% → **86.2%** (+7.5%p)  
- 일부 과제(SST-2, CoLA, RTE)에서 RoBERTa-large 상회  
- 정밀한 추론 단계 삽입이 ChatGPT 이해 능력 극대화  

## 3.2 일반화 성능 관점  
- Few-shot 예시 선택에 민감: 1-shot 예시와 테스트 샘플 간 의미 유사도에 따라 성능 기복 발생  
- 5-shot 시 예시 노이즈 영향 분산, 안정적 성능 확보  
- 모델이 본질적 규칙과 경계 학습 없이 예시 의존적 패턴 학습에 머무름 → 일반화 한계

# 4. 한계 및 향후 고려 사항

1. **데이터 규모 제약**: 검증 샘플 일부 사용, 전체 검증·테스트로 확대 필요  
2. **과제 범위**: GLUE에 국한, 복잡한 지식 추론·대규모 문맥 처리 과제로 확장 연구 권장  
3. **예시 의존성**: 프롬프트 예시 품질과 유사도에 따라 변동이 커, 자동 예시 선별·정제 방법 개발 시급  
4. **세밀한 의미 구분**: 부정 샘플 및 경계점 판정 강화를 위한 후속 학습 기법 필요  

# 5. 연구의 영향 및 미래 방향

이 연구는 대규모 언어모델의 이해 능력을 정량화하고, **프롬프트 엔지니어링**을 통해 추론·분류 성능을 크게 개선할 수 있음을 입증했다.  
향후 연구에서는  
- **자동화된 CoT 생성**으로 수동 설계 부담 완화  
- **메타 학습** 기반 소수 샷 일반화 성능 제고  
- **혼합 미세조정**(hybrid fine-tuning)으로 프롬프트와 모델 파라미터 동시 최적화  
를 통해 LLM의 **진정한 일반화 능력** 확보를 모색해야 한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9e2f262b-44ef-4e6e-82cd-6ad75acdc9f7/2302.10198v2.pdf)
