# GPTScore: Evaluate as You Desire

**핵심 주장 및 기여**  
GPTScore는 대규모 생성형 사전학습 언어모델(GPT-3 등)의 **제로샷 지시** 및 **컨텍스트 내 학습(emergent abilities)** 능력을 활용해, 사용자가 원하는 다차원적·맞춤형 평가 지표를 **학습 없이(train-free)** 자동 생성한다. 이를 통해 37개 데이터셋·22개 평가 측면을 단일 프레임워크로 통합하고, 기존 지도학습 기반 평가 지표에 비해 유연성과 확장성을 획기적으로 개선했다.[1]

## 1. 해결하고자 하는 문제  
기존 자동평가(metric)들은  
- 제한된 평가 측면(예: 유창성, 일관성 등)에만 대응  
- 각 측면별 별도 지도학습 또는 수작업 주석 필요  
- 새로운 평가 기준 도입 시 **데이터 수집·모델 재학습**이 필수여서 비용·시간 부담이 큼.[1]

GPTScore는 이 모든 제약을 극복하고자, **사용자 지시문(task specification) + 평가 측면 정의(aspect definition)** 을 템플릿으로 삼아, 대규모 생성 언어모델이 해당 지시문을 따를 확률을 점수로 활용한다.

## 2. 제안 방법  
### 2.1 평가 프레임워크 수식  
주어진 가설문장 $$h=\{h_t\}_{t=1}^m$$, 지시문·측면 정의·컨텍스트 $$T(d,a,S)$$에 대해,  

$$
\text{GPTScore}(h\mid d,a,S)
=\sum_{t=1}^m w_t\log p(h_t\mid h_{ < t},T(d,a,S);\theta)
$$

여기서 $$w_t$$는 토큰 가중치(균등)이며, $$p(\cdot)$$는 생성 모델의 조건부 확률이다.[1]

### 2.2 프롬프트 구성  
- **Task Specification**: 예) “Generate a fluent and grammatical summary for the following text: {Text} Tl;dr {Summary}”  
- **Aspect Definition**: 예) “Is the generated text well-written and grammatical?”  
- **Few-shot Demonstration(선택적)**: 실제 예시를 템플릿에 포함  

### 2.3 모델 구조  
GPTScore는 다양한 사전학습 모델을 백본으로 사용  
- Decoder-only: GPT-3(text-ada, babbage, curie, davinci-001/003), OPT(350M–66B), GPT-J-6B  
- Encoder-decoder: FLAN-T5(small–XXL)  
- GPT-2 variants(355M–1.5B) 등[1]

## 3. 성능 향상 및 일반화  
- **Instruction only**: 지시문 추가 시 Vanilla 대비 평균 Spearman 상관 1.2–2.4pt 상승  
- **Instruction + Demonstration**: 추가 예시 활용 시 평균 2–5pt 추가 상승  
- **모델 크기별 경향**:  
  - 소형 모델도 지시문만으로 기존 지도학습 모델과 동등하거나 우수한 성능 달성  
  - 데코더 전용 모델(GPT3, OPT)에서 지시문 효과가 더욱 안정적  
- **일반화 가능성**  
  - 텍스트 요약·기계 번역·대화 생성·데이터-투-텍스트 등 다중 태스크에 단일 프레임워크 적용  
  - 새로운 평가 측면(예: 흥미도, 심층성 등) 추가 시 **프롬프트만 수정**하면 즉시 대응 가능[1]

## 4. 한계 및 고려사항  
- **API 호출 비용**: 대규모 모델 활용 시 비용 및 지연  
- **지시문 설계 난제**: 프롬프트 최적화에 따른 성능 민감도  
- **언어·도메인 의존성**: 영어 텍스트 중심, 다국어 확장 필요  
- **토큰 확률 기반 한계**: 길이·빈도 편향 보정 연구 필요  

## 5. 향후 영향 및 연구 고려점  
- **동적 평가 지표 설계**: 다양한 평가 관점 요구에 실시간 대응 가능  
- **프롬프트 자동화**: 지시문·예시 자동 생성 기법 연구  
- **비용 효율적 경량화**: 소형·추론 최적화 모델에 GPTScore 기법 통합  
- **다국어·멀티모달 확장**: BLOOM 등 다국어 모델 적용 및 이미지·음성 평가 포함  

GPTScore는 **학습 없는 사용자 맞춤형 평가** 패러다임을 제시하며, 생성 AI 평가의 **유연성, 확장성**을 대폭 확장시킨다는 점에서 향후 **자동평가(metric) 연구**의 중요한 전환점이 될 것으로 기대된다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a48948e9-4ef3-4767-b60b-d41387bdebec/2302.04166v2.pdf)
