# Challenging BIG-Bench tasks and whether chain-of-thought can solve them

**주요 주장:**  
BIG-Bench Hard(BBH)로 선별된 23개의 난이도 높은 평가 과제에서, 기존의 “답안만(prompting without CoT)” 기준 방식은 언어모델의 잠재력을 과소평가한다. 체인-오브-생각(Chain-of-Thought; CoT) 프롬프트를 적용하면, 대규모 언어모델(특히 Codex code-davinci-002)이 평균 인간 평가자 성능을 17/23개 과제에서 능가함을 보인다.[1]

**주요 기여:**  
1. **BBH 벤치마크 제안:**  
   – 원본 BIG-Bench의 209개 과제 중, 엄격한 필터링(다중선택·정확도, 예제 수, 인간 평가 기준 등)으로 23개 난이 과제 선별  
   – 평가 예제 총 6,511개, 27개 세부과제 포함  
2. **CoT 프롬프트 설계 및 적용:**  
   – 각 과제에 3개 CoT 예시 수작업 작성  
   – “Let’s think step-by-step” 방식으로 중간 추론 유도  
3. **실험 및 분석:**  
   – 모델군: PaLM(8B/62B/540B), InstructGPT(text-davinci-002 외), Codex(code-davinci-002 외)  
   – 평가 방식: 그리디 디코딩, EM(exact match) 측정  
   – 결과:  
     -  Codex+CoT, BBH 평균 73.9% → 인간 평균 67.7% 제치고 17/23 과제 능가  
     -  CoT 최대 +28.5% 향상, 모델 규모↑에 따라 CoT 효과 뚜렷히 증가[1]
     -  CoT로 Emergent Ability(과제별 평탄 스케일링 곡선을 돌파) 관찰 (ex. Multi-Step Arithmetic, Tracking Shuffled Objects)[1]

***

# 1. 해결 문제와 제안 방법

## 1.1 해결하고자 하는 문제  
기존 BIG-Bench 평가에서 few-shot “답안만” prompting은 복합적 추론이 필요한 난도 높은 과제에서 모델 성능을 과소평가한다.  

## 1.2 제안 방법  
– **Chain-of-Thought Prompting (CoT):**  
  – Few-shot 예시 중간에 자연어 추론 과정을 삽입  
  – 프롬프트: “Let’s think step-by-step.”  
– **벤치마크 구축:**  
  1) BIG-Bench 과제 필터링 → clean MCQ or EM, 인간 성능 기록, 충분 예제 확보  
  2) 어려운 36개 과제 중 23개 실용 과제 선정 (BBH)  

## 1.3 모델 구조  
– **Codex:** code-davinci-002 등, code+text 훈련 → 알고리즘적 패턴 강점  
– **InstructGPT:** text-davinci-002 등, instruction-tuned text 모델  
– **PaLM:** 8B, 62B, 540B 규모  

***

# 2. 성능 향상 및 한계

| 모델 (540B)           | Answer-only | CoT     | Δ      | 인간 평균(67.7%) 대비 능가 과제 수 |
|-----------------------|-------------|---------|--------|-------------------------------|
| PaLM                  | 52.3%       | 65.2%   | +12.9% | 10/23                         |
| InstructGPT           | 51.8%       | 68.4%   | +16.6% | 15/23                         |
| Codex (code-davinci)  | 56.6%       | 73.9%   | +17.3% | 17/23                         |  

– **Emergent Ability:**  
  CoT는 PaLM 8B, InstructGPT 소형 모델에서는 효과 미미하나, 대형 모델에서 갑자기 대규모 향상.[1]
– **한계:**  
  – 세계지식·상식 필요 과제(Causal Judgement, Ruin Names 등)에서는 CoT 효과 제한적  
  – 최고 인간 평가자(94.4%)에는 여전히 미치지 못함  

***

# 3. 모델 일반화 성능 향상 가능성

CoT prompting은 다단계 논리·산술·공간·시간 추론 과제 전반에 적용 가능하며,  
– **알고리즘적 과제:** 연쇄적 스텝 분해 통해 정확도 대폭 상승  
– **자연어 이해 과제:** 논리적 설명 유도로 패러그래프 추론 강화  
– **다국어·도메인 확장:** 향후 다언어 CoT, 의학·법률 등 전문 지식 과제에도 활용 기대  

스케일과 CoT 결합은 모델이 훈련 데이터에 명시되지 않은 추론/논리 규칙을 일반화하는 핵심 메커니즘으로 작용할 수 있다.

***

# 4. 향후 연구 영향 및 고려사항

**영향:**  
– CoT 프롬프트는 대규모 모델의 Emergent Ability 연구에 새로운 실험적 증거 제공  
– 복합적 추론 벤치마크 설계 기준으로 BBH 채택 가능  

**고려사항:**  
1. **자동 CoT 생성:** 수작업 예시 제작 대신 자동화 방법 필요  
2. **신뢰도 및 일관성:** CoT 중간 추론의 정합성·설명 가능성 검증  
3. **전문 분야 확장:** 의학·법률·수학 형식적 증명 등 고난도 과제로 확대  
4. **소형 모델에서의 CoT:** 경량화된 CoT 전략 연구  

이상의 방향을 통해 CoT prompting의 범용성·효율성을 높이고 모델 일반화 역량을 극대화할 수 있을 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/43a8de1a-e59c-4684-a191-a86b1da8dacd/2210.09261v1.pdf)
