# Towards Reasoning in Large Language Models: A Survey

## 1. 핵심 주장 및 주요 기여
이 논문은 대형 언어 모델(LLM)이 보여주는 **추론 능력의 현황**을 체계적으로 정리하고,  
이를 향상·평가하기 위한 **기술, 벤치마크, 실험 결과**, 그리고 **향후 연구 방향**을 제시한다.  
- **추론 능력은 LLM의 규모가 커질수록**(100억 매개변수 이상) **비약적으로 향상**되는 ‘Emergent ability’임을 확인  
- **체인 오브 소트(Chain-of-Thought, CoT) 프롬프트**가 LLM의 다중 스텝 추론 성능을 크게 개선함을 입증  
- **추론 향상 기법**(Fully Supervised Finetuning, In-Context Learning, Problem Decomposition, Hybrid Methods 등)을 총망라하여 비교 분석  
- **추론 평가 지표 및 벤치마크**(GSM8K, CSQA, FOLIO, PrOntoQA 등)와 **심층 분석 도구**(ROSCOE) 소개  

***

## 2. 문제 정의 및 제안 기법

### 2.1 논문이 해결하고자 하는 문제
- LLM이 **실제 ‘추론(reasoning)’**을 수행하는지, 아니면 단순한 패턴 매칭(heuristics)에 그치는지를 명확히 규명  
- 다양한 **추론 유도(prompting) 기법**과 **훈련 방법**이 성능에 미치는 영향을 종합적으로 분석  

### 2.2 제안하는 방법 및 모델 구조
1. **Fully Supervised Finetuning (§3.1)**  
   - 추론 과정을 포함한 라벨링된 데이터(예: CoS-E)로 사전학습된 모델을 미세조정  
   - 한정된 도메인에서 추론 성능 상승  

2. **Prompting & In-Context Learning (§3.2)**  
   - Chain-of-Thought Prompting:  

```math
p = \{(x_i,\,r_i,\,y_i)\}_{i=1}^k \cup \{x^*\} \quad\rightarrow\quad \text{generate }(r^*,y^*)
``` 
   
   - Zero-Shot-CoT (“Let’s think step by step”)  
   - Rationale Engineering: 예제 복잡도 조절·샘플링(Complexity-based, Algorithmic prompting), Self-Consistency(다양한 합리화 샘플링 후 다수결)  

3. **Problem Decomposition (§3.2.3)**  
   - Least-to-Most Prompting: 복잡 문제를 순차적 소문제로 분할  
   - Dynamic Decomposition, Successive Prompting  

4. **Hybrid Methods (§3.3)**  
   - Reasoning-Enhanced Pretraining: 과학·수학 데이터로 추가 사전학습  
   - Bootstrapping & Self-Improving (STaR): 모델의 CoT 출력을 정답이 보장된 데이터로 재훈련  

### 2.3 성능 향상 및 한계
- **CoT Prompting**만으로도 수학·상식·심볼릭 추론 벤치마크에서 2–3배 성능 향상  
- **Self-Consistency**로 불확실성 감소 및 정확도 추가 상승  
- **Hybrid Finetuning** 시, 단순 CoT 프롬프트 대비 **20–30% 추가 개선**  
- 한계:  
  - **복잡 추론(compositional/generalization)** 과제에서는 아직 인간 수준 미달  
  - 라벨링된 추론 데이터의 **획득 비용**  
  - LLM의 **편향·비일관성** 문제  

***

## 3. 일반화 성능 향상 관점
- **Length Generalization**: 짧은 문제 훈련 후 긴 문제 해결 능력은 Few-Shot CoT + Scratchpad Finetuning 조합에서만 관찰됨  
- **Domain Generalization**: 다양한 도메인·언어로 생성된 CoT 예제 활용 시, **영역 밖 과제**에도 견고한 성능  
- **Self-Consistency**와 **Rationale Diversity**가 모델의 OOD(Out-of-Distribution) 강인성 강화  
- **Problem Decomposition** 기법은 복합 합성 과제(compositional tasks)에서 특히 유의미한 개선을 보임  

***

## 4. 향후 연구 영향 및 고려사항
- **실제 응용 과제**(법률·의료·과학적 의사결정)에 맞는 **의미 있는 벤치마크** 개발  
- **추론 신뢰성** 검증: 외부 검증기(verifier) 또는 **공식 논리 체계** 기반 평가 도구(PrOntoQA, FOLIO) 활용  
- **데이터·아키텍처 설계** 단계에서 ‘추론 친화적’ 최적화 목표(loss) 통합  
- **자기 향상(Self-Improving) 루프**의 안정성 보장 및 **편향 누적** 방지  
- **작은 모델로의 지식 증류**를 통한 경량화 및 실시간 추론 응용  

이 논문은 LLM의 추론 능력을 체계적으로 정리·분석함으로써, **추론 중심 AI 연구**의 방향타 역할을 하며, 더 강인하고 신뢰할 수 있는 언어 추론 시스템 개발을 위한 토대를 제공한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/83f16247-a4fd-45b8-9f0d-24e2b11e20d2/2212.10403v2.pdf)

# 2 What is Reasoning?

**추론**은 주어진 정보(증거, 전제, 경험 등)를 바탕으로 논리적·체계적으로 결론이나 판단을 이끌어내는 인지적 과정입니다. 인간 지능의 핵심 요소로, 문제 해결, 의사 결정, 비판적 사고 등 다양한 상황에서 작동합니다.

## 1. 추론의 주요 유형

1. **연역적 추론 (Deductive Reasoning)**  
   - 전제가 참일 때 결론이 필연적으로 참이 되는 방식  
   - 예시:  
     -  전제 1: 모든 포유류는 신장을 가진다.  
     -  전제 2: 고래는 포유류이다.  
     -  결론 : 따라서 고래는 신장을 가진다.  

2. **귀납적 추론 (Inductive Reasoning)**  
   - 개별 관찰에서 일반적 결론을 유도하되, 완전한 확실성은 보장되지 않음  
   - 예시:  
     -  관찰 1: 날개가 있는 생물은 모두 새였다.  
     -  관찰 2: 현재 관찰된 생물은 날개가 있다.  
     -  결론 : 이 생물은 새일 가능성이 높다.  

3. **Abductive 추론 (Abductive Reasoning)**  
   - 관찰된 현상의 ‘최적 설명’을 선택하여 결론으로 삼음  
   - 예시:  
     -  관찰: 차가 시동이 걸리지 않고 엔진 밑에 액체 웅덩이가 있다.  
     -  결론 : 라디에이터 누수일 가능성이 가장 크다.  

4. **기타 추론 형태**  
   - **Analogical 추론**: 유사성을 바탕으로 한 비교 추론  
   - **인과 추론 (Causal Reasoning)**: 원인과 결과 관계로부터 결론 도출  
   - **확률적 추론 (Probabilistic Reasoning)**: 사건의 발생 확률을 기반으로 한 판단  

## 2. 형식적 추론 vs. 비형식적 추론

- **형식적 추론 (Formal Reasoning)**  
  - 수학·논리학에서처럼 엄격한 규칙·공리를 따르는 체계적 과정  
  - 결과의 정확성과 재현성이 높음  

- **비형식적 추론 (Informal Reasoning)**  
  - 직관, 경험, 상식 등을 활용한 덜 구조화된 과정  
  - 일상적 맥락에 유연하지만, 오류에 취약할 수 있음  

## 3. 언어 모델에서의 추론

대형 언어 모델(LLM) 연구에서는 주로 **비형식적 연역 추론**(informal deductive reasoning)을 중심으로 다룹니다. 이는 전제가 참일 때 결론도 참이라는 연역적 특성을 가지되, 표현 방식은 자연어 ‘체인 오브 소트(chain of thought)’와 같이 자유롭게 전개됩니다. 이러한 추론을 통해 모델은 다중 단계 문제 해결, 상식 추론, 수리 계산 등에서 성능을 크게 향상시킬 수 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/83f16247-a4fd-45b8-9f0d-24e2b11e20d2/2212.10403v2.pdf)

# 3. Towards Reasoning in Large Language Models

이 장에서는 대형 언어 모델(LLM)의 **추론 능력**을 향상·유도하기 위한 주요 기법들을 목차별로 상세히 설명한다.

## 3.1 완전 지도(fully supervised) 파인튜닝  
- **핵심 아이디어**  
  사전에 수집된 “추론 과정을 포함하는” 데이터셋(예: CoS-E)을 사용해 LLM을 미세조정(finetuning)  
- **장점**  
  -  명시적 추론 패턴 학습  
  -  특정 도메인(수학, 상식 등)에서 안정적 성능 개선  
- **단점**  
  -  추론 라벨링 데이터 구축 비용이 높음  
  -  한정된 데이터에 과적합(overfitting) 가능성  

## 3.2 프롬프트 기반 학습(Prompting & In-Context Learning)  
### 3.2.1 체인 오브 소트(Chain-of-Thought, CoT) 프롬프트  
- **기법**  
  Few-shot 예시를 ⟨입력, 추론 과정, 출력⟩ 형태로 제시해, 모델이 “추론 과정을 먼저” 생성하도록 유도  
- **변형**  
  -  Zero-Shot CoT: few-shot 없이 “Let’s think step by step”만 추가  
  -  코드 기반 CoT: 추론 과정을 코드 형식으로 출력  
  -  Iterative CoT: 한 번이 아니라 여러 차례 추론 반복  
- **효과**  
  산수, 기호, 상식 문제에서 대폭 성능 향상  

### 3.2.2 추론(근거) 공학(Rationale Engineering)  
- **추론 예시 정제(Rationale Refinement)**  
  -  복잡도 기반 예시 선택  
  -  알고리즘 예시 제공  
- **추론 탐색(Rationale Exploration)**  
  -  Self-Consistency: 여러 추론 경로 생성 후 다수결 방식으로 최종 답 선택  
  -  다양한 예시 샘플링  
- **추론 검증(Rationale Verification)**  
  -  전용 검증기(verifier) 학습  
  -  LLM 자체를 검증기로 활용  

### 3.2.3 문제 분해(Problem Decomposition)  
- **Least-to-Most Prompting**  
  복잡한 문제→차례로 해결 가능한 소문제로 분할→순차 해결  
- **Dynamic Decomposition**  
  입력마다 동적으로 분해 전략 선택  
- **Successive Prompting**  
  매 단계마다 이전 소답안을 참고해 다음 소문제 생성  

### 3.2.4 기타 기법  
- **Selection-Inference Framework**: LLM을 모듈화해 사실 선택→추론 단계 분리  
- **Backward Chaining**: 목표에서 출발해 필요한 전제 역으로 탐색  
- **Abductive Prompting**: 가능한 설명을 재귀적으로 생성  

## 3.3 하이브리드(Hybrid) 접근  
### 3.3.1 추론 강화(pretraining & prompting)  
- 과학·수학·SQL 데이터로 추가 사전학습  
- Instruction-tuned 모델(Flan)처럼 광범위한 추론 태스크로 미세조정  

### 3.3.2 자체 강화(Self-Improving)  
- **STaR(Self-Taught Reasoner)**: 초기 추론 생성→정답 유도된 추론만으로 재학습→반복  
- **Bootstrapping**: 모델 출력의 일관성을 활용해 스스로 추론 능력 강화  

***

이처럼 3장 전반은 **“LLM이 스스로 혹은 유도된 방식으로 다중 단계 추론을 수행하게 하는 다양한 방법론”**을 체계적으로 정리한다. 이러한 기법들의 조합과 발전을 통해, 복잡한 논리·수리·상식 과제에 대한 LLM의 성능이 크게 향상될 수 있다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/83f16247-a4fd-45b8-9f0d-24e2b11e20d2/2212.10403v2.pdf)
