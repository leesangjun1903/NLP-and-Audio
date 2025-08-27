# Atlas: Few-shot Learning with Retrieval Augmented Language Models

‘Atlas: Few-shot Learning with Retrieval Augmented Language Models’(Izacard et al., 2022)은 **매우 적은 학습 예시(few-shot)** 환경에서도 대규모 파라미터 기반 모델과 동등하거나 그 이상의 성능을 달성할 수 있음을 보인다. 이를 위해, 모델이 방대한 지식을 파라미터에 내장하는 대신 **외부 지식 소스(위키피디아·Common Crawl)에서 문서를 검색(retrieval)하여 활용**하도록 함으로써, 파라미터 수를 크게 줄이면서도 지식집약적 작업에서 강력한 일반화 능력을 갖춘다.

주요 기여  
1. **Retrieval-augmented pre-training**: Contriever 기반 dense retriever와 Fusion-in-Decoder(T5) 기반 언어 모델을 **Joint pre-training**하여, few-shot 학습 능력을 획기적으로 향상시킴.  
2. **Retriever 학습 손실 함수 비교**: Attention Distillation, EMDR2, Perplexity Distillation, LOOP 등 네 가지 검색기 손실을 비교·분석하고, **Perplexity Distillation**이 효율성과 안정성 면에서 최적임을 확인.  
3. **Few-shot 성능**: 11B 모델이 단 64개 예시로 NaturalQuestions 42.4% 정확도 달성(540B PaLM 대비 +3%p), MMLU 57개 도메인 평균 43.4% 달성(5-shot) 등 파라미터 50×↓에도 동등 이상 성능.[1]
4. **업데이트 가능성(updateability)**: 색인이 변경되면(model 재훈련 없이) 최신 정보 반영이 가능함을 시계열 질문 데이터(TempLAMA)로 검증.  
5. **해석 가능성(interpretability)**: Retrieval된 문서를 직접 확인하여 모델의 답변 근거를 추적·분석할 수 있음.  
6. **색인 압축**: Product Quantization 적용 시 색인 크기를 최대 10× 축소해도 QA 성능 저하 미미.

***

# 1. 해결하고자 하는 문제

- **Few-shot 학습 시 지식 집중화(trade-off)**:  
  - 대규모 언어 모델은 수십억∼수백억 파라미터에 걸쳐 지식을 내장하여 few-shot 일반화 성능을 확보하지만, 파라미터 수·컴퓨팅 비용이 급격히 증가.  
  - **질문 응답·팩트체크 등 지식집약적 태스크**에서 파라미터 메모리 기반 지식 저장의 비효율성과 확장성 한계 존재.

- **목표**: “few-shot 학습” 능력을 유지하면서도 **파라미터 수와 컴퓨팅 비용을 절감**하고, **모델 업데이트·해석 가능성**을 확보하는 것.

***

# 2. 제안하는 방법

## 2.1 모델 구조

Fusion-in-Decoder 기반 Retrieval-Augmented Language Model  
- **Retriever**: Contriever (BERT-base, dual-encoder)  
  - 쿼리·문서를 독립 임베딩→내적(dot-product) 유사도로 Top-K 검색  
- **Language Model**: T5 (seq2seq) + Fusion-in-Decoder  
  - 각 문서를 독립 인코더 처리 후 디코더가 쿼리와 인코더 출력 모두에 cross-attention 수행

## 2.2 학습 손실 함수

Retrieval module을 언어 모델과 **joint-training**하기 위한 4종류 손실:[1]
1. **Attention Distillation** (ADist)  

$$
     \mathrm{KL}\bigl(p^{\text{attn}}(d_k)\|p^{\text{retr}}(d_k)\bigr)
   $$  
   
   – LM의 cross-attention 가중치로 문서 중요도 측정  
2. **EMDR2**  
  
$$
     -\log\sum_{k=1}^K p_{\text{LM}}(a\mid q,d_k)\,p_{\text{retr}}(d_k\mid q)
   $$  
   
   – EM 관점에서 문서를 잠재변수로 모델링  
3. **Perplexity Distillation** (PDist)  

$$
     \mathrm{KL}\bigl(p^{\text{retr}}(d_k)\|p^{\text{LM}}(d_k\mid q,a)\bigr)
   $$  
   
   – 문서가 LM perplexity를 얼마나 개선하는지 역추정  
4. **LOOP**  
   
   – 각 문서를 제외했을 때 LM perplexity 변화 기반 중요도 학습  

**Mask Language Modeling** 과 **Prefix LM** 등의 self-supervised pretext task를 joint-pretrain에 활용.

## 2.3 효율적 색인 갱신

- **Top-L re-ranking**: 미리 검색한 L개를 최신 retriever로 재순위  
- **Query-side fine-tuning**: 문서 인코더 고정, 쿼리 인코더만 업데이트 →색인 불필요  

***

# 3. 성능 향상 및 한계

## 3.1 성능

- **Few-shot (64-shot)**  
  - NaturalQuestions 42.4% (+2.8% vs PaLM-540B)[1]
  - TriviaQA 74.5% (+3.3% vs SOTA)  
  - FEVER 64.3% (+5.1% vs Gopher-280B)  
  - MMLU 43.4% (770M→11B 규모 모두 강점)  
- **Full fine-tuning**  
  - NaturalQuestions 60.4% (+8.1% vs SOTA)  
  - TriviaQA 79.8% (+9.3%)  
  - FEVER 80.1% (FEVER 전용 색인 시)  
  - KILT 8개 태스크 중 5개 SOTA 달성  

## 3.2 한계

- **Retrieval 의존성**: 인덱스 품질·커버리지에 민감. 도메인 미스매치 시 성능 저하.  
- **계산비용**: 대규모 색인 검색·재순위 비용이 고정 비용 대비 무시할 수 없음.  
- **Few-shot 편향(bias)**: MMLU에서 답 옵션 letter bias 제거 위해 **de-biased inference** 필요.  

***

# 4. 일반화 성능 향상 관점

- **외부지식 활용**: Retrieval이 LM의 일반화 능력을 보완, memorization 의존도↓  
- **Joint-pretraining 효과**: retrieval과 generation 모듈을 함께 학습함으로써, **few-shot sample efficiency**와 **cross-task 일반화** 성능↑  
- **Query-side fine-tuning**: 소수 예시에서도 overfitting↓, **robust 적응력** 확보  

***

# 5. 향후 연구에 미치는 영향 및 고려사항

1. **색인 편집·버전 관리 연구**  
   - 지속적 업데이트·지식 수명주기 관리 기법 발전  
2. **도메인 특화 색인 자동 구축**  
   - 의료·법률 등 전문 분야 인덱스 구축 최적화  
3. **Retrieval+Pretraining 시너지**  
   - 대규모 LLM과의 하이브리드 pretraining 전략 연구  
4. **효율적 검색기 손실**  
   - 더 가벼우면서도 성능 유지 가능한 distillation 기법 개발  
5. **Bias·안정성 검증**  
   - De-biasing, leakage 제어를 포함한 **공정성·안정성** 연구 강화  

이 논문은 “few-shot 일반화”를 위해 retrieval-augmented 접근법이 파라미터 확장 외의 **효율적 대안**임을 입증하여, 향후 연구에서 **반응형(semiparametric) 언어 모델** 개발에 중요한 이정표가 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/939a3755-c222-40eb-93bc-c53bccac8387/2208.03299v3.pdf)
