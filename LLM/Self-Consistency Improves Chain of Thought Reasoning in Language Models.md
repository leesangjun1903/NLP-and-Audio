# Self-Consistency Improves Chain of Thought Reasoning in Language Models

# 요약  
**Self-Consistency**는 체인오브-쏘트(chain-of-thought) 추론에서 단일 탐욕적 디코딩을 대체해, 여러 서로 다른 추론 경로를 샘플링한 뒤 최종 답변의 일관성에 따라 다수결로 정답을 선택하는 **“샘플 앤 마지널라이즈(sample-and-marginalize)”** 전략이다. 이로써 기존 CoT 대비 수학·상식 추론 벤치마크에서 최대 17.9%p(GSM8K), 12.2%p(AQuA) 등 **비약적 성능 향상**을 달성했다.[1]

# 1. 문제 정의  
기존 체인오브-쏘트(CoT) 방식은 출력 중 최빈 경로만을 취하는 탐욕적 디코딩(greedy decoding)을 사용해,  
- 지역 최적해에 빠짐  
- 모델 불확실성 과소평가  
등의 한계가 있다.  
**Self-Consistency**는 다수의 추론 경로가 동일 정답에 도달할수록 신뢰도가 높다는 직관에 착안해,  
여러 경로를 생성한 뒤 답변 일관성을 기준으로 정답을 결정한다.[1]

# 2. 제안 방법  
1) CoT 프롬프트 + 체인 전개  
2) 샘플링(sampling)으로 $$m$$개의 $$(r_i, a_i)$$ 생성  
 -  $$r_i$$: 추론 경로 토큰 시퀀스  
 -  $$a_i$$: “The answer is X.”에서 추출한 최종 답  
3) **마지널라이즈(marginalize)**: 다수결(majority vote)으로  

$$
\hat{a} = \arg\max_{a} \sum_{i=1}^m \mathbf{1}(a_i = a)
$$  

또는 토큰 확률로 가중합(weighted sum)  

$$
P(r_i, a_i) = \exp\!\Bigl(\tfrac{1}{K}\sum_{k=1}^K \log P(t_k \mid \text{prompt}, t_{ < k})\Bigr)
$$  

를 활용해 정답을 선정.[1]

# 3. 모델 구조  
기존 대형 언어모델(GPT-3, PaLM, LaMDA, UL2) 여타 CoT 프롬프트 그대로 사용.  
추가 학습·미세조정 없이, **디코더 샘플링**+ **다수결 집계**만으로 적용 가능.  

# 4. 성능 향상  
-  **산술 추론**: GSM8K +17.9%p, SVAMP +11.0%p, AQuA +12.2%p 등[1]
-  **상식·기호 추론**: StrategyQA +6.4%p, ARC-challenge +3.9%p[1]
-  **Prompt 손상·제로샷 CoT**에도 견고  
-  소규모 모델에도 효과적, 대형 모델일수록 절대 향상폭 증가  

# 5. 한계  
- 계산 비용 증가(m 샘플링)  
- 오답 간 일관성 발생 시 오류 확률  
- 추론 경로 품질 보장 어려움(비논리적 혹은 사실 오류 가능)  

# 6. 일반화 성능 향상 가능성  
Self-Consistency는 특정 문제 도메인에 종속되지 않고  
- **추론 다양성**을 통한 모델 불확실성 정량화  
- 일관성 기반 정답 집계로 **OOD(분포 외)** 설정에서도 성능 향상  
등으로 **모델의 일반화·안정성**을 높인다.[1]

# 7. 향후 영향 및 고려사항  
- **효율적 샘플링** 기법 개발: 적은 경로로도 최대 성능 달성  
- **추론 경로 검증** 모듈 결합: 오답 일관성 억제  
- **자기지도 학습**: Self-Consistency 기반 레이블 생성 후 파인튜닝  
- 다양한 언어·비정형 문제로 확장 가능성  

Self-Consistency는 대형 언어모델의 추론 신뢰도와 일반화력을 크게 개선할 **핵심 디코딩 전략**으로 자리잡을 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/63cfee5d-2794-4a6c-9d8a-663611104273/2203.11171v4.pdf)
