# Benchmarking Large Language Models for News Summarization

## 연구의 핵심 주장과 주요 기여

이 논문은 대형 언어 모델(LLM)의 뉴스 요약 능력에 대한 체계적 평가를 통해 **두 가지 중요한 발견**을 제시합니다:[1]

**핵심 주장:**
1. **모델 크기가 아닌 instruction tuning이 LLM의 zero-shot 요약 능력의 핵심 요소**임을 입증
2. **기존 벤치마크의 참조 요약 품질이 매우 낮아** 모델 성능 평가를 저해하고 있음을 발견

**주요 기여:**
- 10개의 다양한 LLM에 대한 포괄적 인간 평가 수행
- 350M 파라미터의 instruction-tuned GPT-3도 175B 파라미터 GPT-3와 동등한 성능 달성 입증
- 프리랜서 작가가 작성한 고품질 요약과 LLM 요약의 동등성 확인
- CNN/DM과 XSUM 데이터셋의 참조 요약 품질 문제 해결을 위한 새로운 평가 기준 제시

## 해결하고자 하는 문제와 제안 방법

### 문제 정의

**1. LLM 성공 요인의 불명확성**
- LLM이 요약 작업에서 좋은 성과를 보이지만, 어떤 설계 요소가 핵심인지 불분명
- 모델 크기, instruction tuning, few-shot learning의 상대적 중요성 파악 필요

**2. 평가 기준의 신뢰성 문제**
- 기존 벤치마크(CNN/DM, XSUM)의 참조 요약이 자동 시스템보다 품질이 낮음[1]
- 자동 메트릭과 인간 평가 간 상관관계 저하

### 제안 방법

**체계적 실험 설계:**

**평가 메트릭:**
- **Faithfulness**: 이진 평가 (사실성)
- **Coherence**: 1-5 Likert 척도 (일관성)  
- **Relevance**: 1-5 Likert 척도 (관련성)

**실험 설정:**
- **Zero-shot**: 지시만으로 요약 생성
- **Five-shot**: 5개 예시와 함께 요약 생성
- **Temperature 0.3**으로 샘플링

**프롬프트 템플릿:**
```
Article: [article]. Summarize the article in three sentences. Summary:
```

## 모델 구조 및 성능 분석

### 평가 모델 구조

논문에서 평가한 10개 LLM은 다음과 같이 분류됩니다:[1]

**Base 모델:**
- GPT-3 (350M, 6.7B, 175B)
- OPT (175B)
- GLM (130B)
- Anthropic-LM (52B)
- Cohere xlarge (52.4B)

**Instruction-tuned 모델:**
- InstructGPT (350M, 6.7B, 175B)

### 성능 향상 결과

**주요 성능 지표 (CNN/DM 데이터셋):**

| 모델 | Faithfulness | Coherence | Relevance |
|------|--------------|-----------|-----------|
| GPT-3 (175B) | 0.76 | 2.65 | 3.50 |
| Instruct Davinci (175B) | 0.99 | 4.15 | 4.60 |
| Instruct Ada (350M) | 0.88 | 4.02 | 4.26 |
| PEGASUS (Fine-tuned) | 0.97 | 3.93 | 4.38 |

**핵심 발견:**
- **Instruction tuning의 효과**: 같은 크기 모델에서 instruction tuning이 모든 평가 지표를 크게 향상[1]
- **크기 vs. Instruction**: 350M Instruct Ada가 175B GPT-3보다 일관성과 관련성에서 우수
- **Five-shot 성능**: 비instruction-tuned 모델도 few-shot으로 성능 개선 가능

### 일반화 성능 향상 요소

**1. Instruction Tuning의 핵심 역할**
- 다양한 NLP 태스크를 프롬프트 형태로 학습
- 지시 따르기 능력의 획기적 개선
- 모델 크기보다 더 중요한 요소로 입증[1]

**2. Multi-task Learning 효과**
- 다양한 입력 분포에서의 학습
- 요약 외 다른 태스크로부터의 지식 전이
- 일반화 능력 향상에 기여

**3. Human Feedback Learning**
- 지도 학습 대신 인간 선호도 기반 학습
- 작성과 평가 간 불일치 해결 가능성
- 더 자연스러운 요약 생성

## 연구의 한계

**1. 평가 방법론적 한계**
- **개인차 존재**: 평가자마다 일관된 선호도를 보이지만, 평가자 간 합의도가 낮음 (Krippendorff's α = 0.07)[1]
- **주관성 문제**: 요약의 "좋음"에 대한 절대적 기준 부재

**2. 데이터셋 품질 문제**
- 기존 참조 요약의 낮은 품질로 인한 메트릭 신뢰성 저하
- Few-shot 및 fine-tuning 성능의 과소평가 가능성

**3. 스타일 차이**
- **LLM**: 추출적 요약 (Coverage: 0.92, Density: 12.1)
- **인간**: 추상적 요약 (Coverage: 0.81, Density: 2.07)[1]
- 스타일 선호도가 평가 결과에 영향

## 연구 영향 및 향후 고려사항

### 연구에 미치는 영향

**1. 평가 패러다임의 전환**
- 참조 기반 메트릭의 한계 인식 확산
- 인간 평가의 중요성 재확인
- 고품질 참조 요약의 필요성 강조[1]

**2. 모델 개발 방향성 제시**
- 모델 크기 확장보다 instruction tuning에 집중
- Multi-task learning과 human feedback의 중요성 부각
- Zero-shot 능력 향상을 위한 새로운 접근법 모색

**3. 벤치마크 개선 요구**
- CNN/DM, XSUM 등 기존 벤치마크의 한계 노출
- 더 나은 평가 기준과 데이터셋 개발 필요성

### 향후 연구 고려사항

**1. 평가 방법론 개선**
- 개인차를 고려한 새로운 평가 프레임워크 개발
- 다운스트림 애플리케이션 기반 평가 방법 모색
- 더 세분화된 품질 측정 지표 개발

**2. 모델 성능 향상**
- Instruction tuning 데이터의 품질 개선
- Human feedback 학습 방법론 연구
- 추상적 요약 능력 강화 방안

**3. 벤치마크 및 데이터셋**
- 고품질 참조 요약을 포함한 새로운 벤치마크 구축
- 도메인별, 언어별 다양한 평가 데이터 개발
- 실제 사용 시나리오를 반영한 평가 환경 조성

이 연구는 LLM의 요약 능력에 대한 깊이 있는 통찰을 제공하며, 향후 자동 요약 연구의 방향성을 제시하는 중요한 기여를 했습니다. 특히 instruction tuning의 중요성 발견과 평가 방법론의 개선 필요성 제기는 이 분야의 발전에 장기적 영향을 미칠 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5143c682-0738-490c-a3fe-c6b4f014a5e0/2301.13848v1.pdf)
