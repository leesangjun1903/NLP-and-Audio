# Is ChatGPT a Good NLG Evaluator? A Preliminary Study

## 핵심 주장 및 주요 기여
이 논문은 **ChatGPT를 범용 NLG 평가 지표(metric)**로 사용하여, 기존 자동 평가 지표와 인간 평가 간의 상관관계를 비교·분석한 최초의 연구이다.  
- ChatGPT는 **창의적 텍스트 생성**(예: 스토리 생성) 과제에서 높은 상관관계를 보이며, 전통적 유사도 기반 지표를 능가한다.  
- 프롬프트 디자인(평가 형식·참조 사용 여부)에 따라 성능이 크게 달라지므로, **프롬프트 민감도(prompt sensitivity)** 를 철저히 검증해야 함을 강조한다.  
- 메타평가 데이터셋의 **참조 의존 편향(lexical bias)** 이 ChatGPT 평가 성능에 영향을 미치므로, 평가 데이터셋의 생성 방식을 고려하여 해석해야 한다.  

## 문제 정의
기존 NLG 평가 지표들은  
1. **n-gram 중첩 기반(ROUGE, BLEU 등)** – 표면적 중복에 의존  
2. **임베딩 기반(BERTScore, MoverScore 등)** – 의미적 유사도 측정  
3. **LLM 기반(BARTScore 등)** – 사전 학습 모델 활용  
이처럼 표준화된 지표들이 **인간 평가자와의 높은 일치도**를 보이기도 하나,  
- 생성 텍스트의 **창의성**을 제대로 반영하지 못하거나  
- 참조(reference)에 크게 의존해 편향된 평가가 발생하는 문제가 있다.

## 제안 방법
ChatGPT를 **평가자(evaluator)** 로 간주한 후, 다음 네 가지 프롬프트로 점수를 부여한다.  
1. **Direct Assessment (DA) without Reference**  
2. **Star Rating (1–5 Stars) without Reference**  
3. **DA with Reference**  
4. **Star Rating with Reference**  

각 프롬프트는  
- 과제 지시문(task-specific instruction)  
- 평가 측면(aspect: coherence, relevance, fluency 등)  
- (참조 사용 시) Golden Reference  
를 명시하여 ChatGPT로부터 0–100 점 또는 1–5성 평가를 유도한다.  
평가 점수 $$f_\text{auto}$$와 인간 점수 $$f_\text{human}$$ 간의 상관관계를 **샘플 수준(sample-level)** 및 **데이터셋 수준(dataset-level)**에서 다음과 같이 계산한다.  

$$
\text{Sample-level: } 
\frac{1}{n}\sum_{i=1}^n \rho\bigl(\bigl[f_\text{auto}(g_{i,1}),\dots\bigr], \bigl[f_\text{human}(g_{i,1}),\dots\bigr]\bigr)
$$

$$
\text{Dataset-level: }
\rho\bigl(\bigl[f_\text{auto}(g_{1,1}),\dots\bigr], \bigl[f_\text{human}(g_{1,1}),\dots\bigr]\bigr)
$$

여기서 $$\rho$$는 Spearman, Pearson, Kendall’s Tau이다.

## 모델 구조 및 평가 지표
ChatGPT 자체는 평가 지표 역할을 하므로 별도의 모델 구조 변경 없이, **프롬프트 설계(prompt engineering)** 만으로 평가 성능을 조절한다.  
- 참조 무/유(reference-free vs reference-based)  
- 연속 점수 vs 별점  
- 평가 측면별(aspect-specific) 지시문  

## 성능 향상 및 일반화
- **SummEval(요약)** 및 **OpenMEVA-ROC(스토리)** 메타평가 데이터셋에서 ChatGPT는 **현존 최고 상관관계**를 달성.  
- 창의적 생성 과제(스토리)에서 특히 전통적 지표 대비 큰 성능 격차를 보이며, **창의성·다양성**을 평가하는 데 강점을 갖는다.  
- 반면, **NewsRoom** 및 **RealSumm**처럼 참조 집중 방식(reference-oriented annotation)의 데이터셋에서는 **n-gram 기반 지표**가 더 높은 상관관계를 보여, ChatGPT의 상대적 성능이 낮아진다.  

이로부터, **ChatGPT 평가 모델은 “데이터셋 편향”에 민감하나, 제대로 설계된 프롬프트와 평가 데이터셋을 활용하면 범용 평가자로서 높은 **일반화 능력**을 갖출 가능성이 있다.  

## 한계 및 고려사항
- **프롬프트 의존성**: 동일 과제·측면이라도 지시문 문구 하나에 성능이 크게 달라져, 일관성 있는 평가를 위해 최적 프롬프트 탐색이 필요하다.  
- **실험 재현성**: 당시(2023년) API 미공개로 웹 인터페이스 사용, 온도 등 하이퍼파라미터 제어 어려움.  
- **언어·과제 범위 제한**: 영어 NLG 과제에 국한, 대화 생성(dialogue)·보고서 생성(report) 등 다섯 과제 미실험.  
- **데이터셋 편향**: 참조 의존적 평가에서 제시한 메타평가 데이터셋 편향을 해소할 새로운 벤치마크가 필요하다.  

## 향후 연구 영향 및 제언
이 연구는 **LLM 기반 NLG 평가 지표** 가능성을 타진하며, **프롬프트 엔지니어링**과 **메타평가 데이터셋 설계**의 중요성을 부각시켰다.  
- **향후 평가 지표 연구**에서는 ChatGPT/GPT-4 등 확장성 있는 LLM을 활용한 평가 프레임워크가 주류가 될 것으로 전망된다.  
- **데이터셋 구축** 단계에서 참조 의존 편향을 최소화하는 인간 평가 프로토콜을 설계해야, LLM 평가자의 강점을 온전히 발휘할 수 있다.  
- **크로스-언어·크로스-도메인** 일반화 연구를 통해, ChatGPT 평가 지표의 다국어·다중모달 확장 가능성을 검증해야 한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/72fcf0e0-5af2-41be-9186-d5c9e3a8293f/2303.04048v3.pdf)
