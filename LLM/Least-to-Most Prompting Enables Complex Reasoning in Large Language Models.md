# Least-to-Most Prompting Enables Complex Reasoning in Large Language Models

## 1. 핵심 주장 및 주요 기여  
이 논문은 **“Least-to-Most Prompting”**(L2M)이라는 새로운 프롬프트 기법을 제안한다. L2M은 복잡한 문제를 난이도 순으로 작은 하위 문제들로 분해(decomposition)하고, 이를 차례로 해결(subproblem solving)함으로써 대형 언어 모델이 “이전에 본 적 없는 더 어려운 문제”에도 일반화(generalize)할 수 있게 한다.  
- 체인오브쏘트(CoT) 프롬프트가 **동일 난이도** 문제에는 잘 작동하지만, 난이도가 더 높은 문제에는 성능이 급락하는 “easy-to-hard generalization” 한계를 가짐.  
- L2M 프롬프트는 **분해 단계**와 **해결 단계**의 두 단계를 통해, 단계적 추론을 학습하며, 별도 훈련 없이 few-shot으로 적용 가능.  
- 다양한 태스크(문자열 조작, SCAN compositional generalization, 수학 문제)에서 CoT 대비 뛰어난 일반화 성능을 보임.

## 2. 해결 문제 및 제안 방법

### 해결하고자 하는 문제  
- **easy-to-hard generalization**: CoT는 프롬프트 예시 수준보다 더 어려운 테스트에 약함.  
  예) 마지막 글자 연결(last-letter concatenation)에서 테스트 리스트 길이가 예시 길이보다 길면 성능 저하.

### Least-to-Most Prompting 개요  
1. **문제 분해 (Decomposition)**  
   Prompt에 “난이도 낮은→높은” 분해 예시를 넣고, 대상을 순차적 하위 문제 리스트로 변환.  
2. **하위 문제 해결 (Subproblem Solving)**  
   각 하위 문제마다 앞선 정답을 포함한 프롬프트로 차례로 해결.  

#### 수식 표현  
테스트 리스트 L 길이일 때, `last-letter-concatenation`에서 L2M은  

$$ S_1, S_2, \dots, S_L $$  

순서대로 하위 리스트로 분해하고,  

$$ \text{ans}(S_1), \text{ans}(S_2),\dots $$  

순차적 재귀(recursive) 풀이를 수행한다.

### 모델 구조  
기존 거대 언어 모델(GPT-3 코드-다빈치 등)에 **추가 학습 없이** few-shot prompt만으로 L2M을 적용. 모델 내부 구조 변경 없음.

## 3. 성능 향상 및 한계

### 성능 향상  
- **문자열 조작**: 리스트 길이 12에서 CoT 정확도 38.4% vs. L2M 74.0%.  
- **SCAN (length split)**: code-davinci-002 기준 CoT 16.2% vs. L2M 99.7%.  
- **수학 문제 (GSM8K, DROP)**: 복잡 단계 문제에서 L2M이 CoT 대비 +6.16%p 상승.

### 한계  
- **도메인 일반화 한계**: 분해 프롬프트는 도메인별 튜닝 필요.  
- **분해 어려움**: 복잡 논리·상식 문제 분해 설계가 쉽지 않음.  
- **추론 단순화 필요**: 분해 품질에 성능 의존.

## 4. 일반화 성능 향상 메커니즘  
- **단계적 추론 학습**: 작은 하위 문제→결합 학습으로, 모델이 “어려운 구조”를 재귀적으로 적용.  
- **자기 일관성 (Self-Consistency)** 결합 시 더욱 안정적.  
- **데모 범용성**: 소수 예시로도 난이도 높은 케이스 일반화.

## 5. 향후 영향 및 연구 시 고려사항  
- **대화형 분해 학습**: 단방향 prompting에서 벗어나, 모델 피드백을 활용한 양방향 학습으로 확장 가능.  
- **자동 분해 생성**: 분해 단계 자동화·최적화 알고리즘 필요.  
- **도메인 중립적 분해 프롬프트** 연구: 분해 전략의 범용성 확보.  
- **추론-분해 융합**: 분해 외 추가 메타정보(검증, 교정) 통합 연구 필요.

L2M prompting은 “few-shot prompting” 패러다임을 넘어, **소수 예시로도 보다 복잡한 문제**를 해결할 수 있는 **프롬프트 기반 추론**의 새로운 방향을 제시한다. 앞으로 언어 모델에게 **더 강력하고 유연한 분해-통합 학습**을 제공하기 위한 연구가 활발해질 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/07d14f24-15a4-44b0-938b-4d38979ffd53/2205.10625v3.pdf)
