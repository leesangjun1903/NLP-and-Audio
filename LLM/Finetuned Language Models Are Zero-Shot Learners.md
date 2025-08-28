# Finetuned Language Models Are Zero-Shot Learners

**핵심 주장 및 주요 기여**  
본 논문은 자연어 지시문(instruction) 형태로 다양한 NLP 과제를 혼합(finetuning) 학습하면, 모델이 이전에 학습하지 않은(unseen) 과제에서도 별도의 샘플 없이(zero-shot) 높은 성능을 달성할 수 있음을 보인다. 137B 파라미터의 LaMDA 기반 언어 모델을 60여 개 이상의 과제에 대해 자연어 지시문으로 미세조정하여, FLAN(FLAN 137B)이라 명명된 모델이 GPT-3 175B의 zero-shot 성능을 20/25개 과제에서 능가함을 입증했다.

***

## 1. 문제 정의  
대형 언어 모델은 few-shot 학습에서는 뛰어난 성능을 보이지만, zero-shot 상황에서는 성능이 현저히 떨어진다.  
- GPT-3 zero-shot vs few-shot 격차  
- 지시문(prompt)만으로 과제 수행이 어려움  

이를 해결하기 위해, **지시문 튜닝(instruction tuning)** 기법을 제안한다.

***

## 2. 제안 방법  
### 2.1. 지시문 튜닝(Instruct-Tuning)  
- **입력**: “이 영화 리뷰의 감성은 긍정인가 부정인가?”, “다음 문장을 프랑스어로 번역하라” 등 자연어 지시문  
- **학습 데이터**: 62개 공개 NLP 데이터셋, 12개 과제 클러스터(예: NLI, QA, 번역, 요약 등)  
- **지시문 템플릿**: 각 데이터셋당 10개 템플릿 수작업 작성  
- **Finetuning**:  
  -  데이터셋별 최대 30k 예제 샘플링  
  -  총 30k gradient steps, batch size 8,192 tokens, Adafactor optimizer, LR=3e-5  
  -  입력 길이 1,024, 출력 길이 256, packing 이용  

### 2.2. 평가 설정  
- **일반화 평가**:(task-cluster hold-out) 특정 과제 클러스터(예: NLI)를 학습에서 완전히 제외 → 해당 클러스터의 zero-shot 성능 측정  
- **비교 모델**:  
  -  LaMDA-PT137B (기본 프리트레인 모델)  
  -  GPT-3 175B zero-shot/few-shot  
  -  GLaM 64B/64E zero/one-shot  

***

## 3. 성능 개선 및 한계  
### 3.1. Zero-Shot 성능 개선  
| 과제 유형           | FLAN 137B | LaMDA-PT137B | GPT-3 175B zero-shot | GPT-3 175B few-shot |
|-------------------|---------:|------------:|--------------------:|-------------------:|
| 자연어 추론 (NLI)   | 56.2%    | 42.9%       | 53.2%               | 56.2%              |
| 읽기 이해 (RC)      | 77.4%    | 63.7%       | 72.6%               | 77.4%              |
| Closed-book QA    | 56.6%    | 49.8%       | 55.7%               | 56.6%              |
| 번역 (EN→FR 등)    | BLEU↑    | BLEU↓       | BLEU↓               | BLEU↑              |

- FLAN은 25개 과제 중 20개에서 GPT-3 zero-shot을 앞서며, 6개 과제(ANLI, RTE, BoolQ, ARC, OBQA, StoryCloze)에서는 GPT-3 few-shot도 능가함.  

### 3.2. 지시문 튜닝 효과 분석  
- 과제 클러스터 수 증가 → 일반화 성능 지속 향상  
- 모델 규모 의존성: 8B 이하 모델은 오히려 성능 저하, 68B·137B 모델에서만 유의미한 개선 확인  
- 지시문 중요성: 입력에 지시문 없이 task 이름만 추가 시 zero-shot 성능 대폭 하락  

### 3.3. 한계 및 실패 사례  
- Commonsense reasoning·coreference resolution 등 언어 모델링 지향 과제에서는 개선폭 작음  
- 컨텍스트 길이(1,024 토큰) 제약으로 대용량 요약 과제 미지원  
- 소규모 모델에선 용량 한계로 오버피팅 우려  

***

## 4. 향후 연구에의 시사점  
- **일반화 능력 확대**: 더 다양한 과제 클러스터와 대규모 다국어 지시문 데이터셋 활용  
- **규모-효율 균형**: 중규모 모델에서도 지시문 튜닝 이점을 살릴 수 있는 경량화 기법 모색  
- **지속적 학습(continual learning)**: 새롭게 등장하는 과제에 대한 순차적 지시문 튜닝 전략  
- **안전성·윤리성**: 지시문 데이터 내 편향이 모델에 전달되지 않도록 편향 감지·교정  

지시문 튜닝은 “다중 과제 학습”과 “프롬프트 기반 학습”의 장점을 결합하여, 대형 언어 모델의 zero-shot 일반화 능력을 크게 확장할 수 있음을 보여줌으로써, 향후 범용 AI 어시스턴트 개발 및 다양한 산업 적용에 핵심적인 기반을 제공할 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8e914468-3422-45f5-8568-f68a4e4ccda6/2109.01652v5.pdf)
