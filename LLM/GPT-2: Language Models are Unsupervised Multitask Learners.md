# GPT-2: Language Models are Unsupervised Multitask Learners

## 1. 핵심 주장 및 주요 기여
Alec Radford 등(2019)의 “Language Models are Unsupervised Multitask Learners”는 대규모, 비감독(pre-training) 언어 모델이 자연어 처리의 다양한 과제를 **제로샷(미세조정 없이)** 으로 수행할 수 있음을 보인다.  
- **주요 기여**  
  1. 1.5B 파라미터 규모의 GPT-2 모델 제안  
  2. 인간이 큐레이션한 웹문서(WebText, 40 GB)를 활용한 대규모 언어 모델 학습  
  3. CoQA, LAMBADA, CBT, Winograd Schema, CNN/DailyMail 요약 등 8개 과제에서 제로샷 성능 SOTA 달성  

## 2. 문제 정의, 제안 방법, 모델 구조

### 2.1 해결하려는 문제
- **현황**: 기존 NLP 시스템은 개별 과제별 레이블된 데이터로 슈퍼바이즈드 학습→과제별 미세조정 필요  
- **목표**: 대규모 비감독 학습만으로도 다양한 과제를 즉시 수행(Zero-shot transfer)  

### 2.2 제안 방법
- **언어 모델링**을 범용 과제 학습 메커니즘으로 활용  
- 입력 시퀀스에 ‘task hint’(예: “translate to French: …”)를 자연어로 제시  
- 모델이 다음 토큰을 예측하도록 함으로써 과제를 수행  

#### 2.2.1 학습 데이터  
- Reddit에서 인간이 큐레이션한 4500만 개 링크 중 중복·저품질 제거 → 8백만 문서(WebText)  
- 총 40 GB 텍스트, 엔터티·인용문·질문-답변 예제가 자연스럽게 섞여 있음  

#### 2.2.2 수식적 관점  
언어 모델은 조건부 확률  

$$
p(x)=\prod_{i=1}^n p(s_i\mid s_{ < i }),
$$  

제로샷 과제는 다음과 같이 확장  

$$
p(\text{output}\mid \text{input}, \text{task hint})
=\prod_{t=1}^T p(y_t\mid x, \text{task}, y_{ < t}).
$$

### 2.3 모델 구조
- **Transformer 기반**(Vaswani et al., 2017)  
- GPT 대비: 레이어 수 48, hidden size 1600, context 길이 1024 토큰, BPE 50,257 어휘  
- 잔차 스케일 조정, 전치(normalization-before) layer norm 적용  

## 3. 성능 향상 및 한계

| 과제                      | 메트릭            | GPT-2 성능      | 이전 SOTA     |
|-------------------------|-----------------|--------------|-------------|
| **언어 모델링** (WikiText-103) | Perplexity      | 18.34        | 39.14    |
| **CBT-네임드 엔티티**        | Accuracy        | 89.1%        | 82.3%    |
| **LAMBADA**              | Perplexity/Acc. | 8.6 / 63.2%  | 99.8 / 19%|
| **Winograd Schema**      | Accuracy        | 70.7%        | 63.7%    |
| **CoQA**                 | F1              | 55           | ∼45      |
| **CNN/DailyMail 요약**     | ROUGE-L         | 26.6         | 38.3     |
| **영-불 번역**            | BLEU            | 5.0          | 33.5     |

- **장점**: 제로샷 SOTA, 대규모 데이터+모델 용량에 따른 성능 로그선형 상승  
- **한계**:  
  - 추상적 요약, QA 정답 추출에서는 여전히 전통적 fine-tuning 기법보다 낮은 성능  
  - One-Billion-Word Benchmark에는 부진(문장 셔플링)  
  - 대용량 메모리·연산자원 요구량이 매우 큼  

## 4. 일반화 성능 향상 가능성
- **데이터 다양성**: Reddit 큐레이션→도메인 편향 완화, long-range dependency 학습  
- **모델 용량**: 파라미터 수↑ → zero-shot 성능 일관된 향상(로그선형 경향)  
- **과제 힌트**(task hint) 활용: 자연어 지시문으로 다양 과제 지시 가능 → 강화된 일반화  
- **미미한 과적합 증거**: n-gram 중복율 3% 수준, 학습-검증 perplexity 유사  

## 5. 향후 연구 및 고려 사항
- **미세조정(fine-tuning)**: GPT-2의 양방향 표현 한계 보완, GLUE·decaNLP 벤치마크 적용  
- **데이터 중복 제거**: 더 정교한 유사 문장 필터링(fuzzy matching)  
- **모델 효율화**: 연산 비용·메모리 최적화 연구(압축·지식 증류)  
- **제로샷 한계과제 탐색**: 요약·번역·QA에서 질적 오류 유형 분석 및 개선  
- **안전성·윤리성**: 대규모 비감독 학습이 생성하는 허위 정보·편향 문제 완화 전략  

**결론**: GPT-2는 대규모 비감독 언어 모델이 자연어 지시만으로 다양한 과제를 제로샷 수행할 수 있음을 입증했으며, 향후 fine-tuning 효율화, 일반화·안정성 연구가 필요하다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5e35b656-5597-4c3b-b45b-da8cede02f0d/language_models_are_unsupervised_multitask_learners.pdf
