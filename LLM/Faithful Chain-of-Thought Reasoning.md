# Faithful Chain-of-Thought Reasoning

## 1. 핵심 주장 및 주요 기여  
“Faithful Chain-of-Thought (CoT) Reasoning”은 기존 CoT 프롬프트가 생성하는 자연어 추론 과정이 실제 모델의 답변 근거와 일치하지 않는 비(非)신뢰성 문제를 지적하고, 이를 해결하기 위한 **두 단계 프레임워크**를 제안한다.  
- **Translation 단계**: 자연어 질문을 자연어 설명(CNL)과 기호 언어 코드(CSL)를 교차로 포함한 “추론 사슬”로 변환  
- **Problem Solving 단계**: 외부 **결정론적 솔버**(예: Python 인터프리터, Datalog/​PDDL 엔진)를 이용해 CSL을 실행함으로써 최종 답안을 도출  
이로써 추론 사슬이 답변 생성 과정과 **인과적으로 연결되어**, **설명(Explanation)의 신뢰도(faithfulness)를 보장**한다. 실험 결과, 10개 벤치마크 전반에서 기존 CoT 대비 최대 21.4% 상대 성능 향상을 달성했다.

## 2. 문제 정의, 제안된 방법, 모델 구조, 성능 및 한계

### 2.1 해결 과제  
- 기존 **Chain-of-Thought(CoT) 프롬프트**는 자연어로 된 추론 과정을 제공하지만, **생성된 체인이 실제 답변 산출 과정과 일치하지 않음**  
- 비신뢰성(unfaithfulness)은 해석가능성(interpretablity)을 저해하고, 고위험 도메인에서 **과신(trust)** 문제를 유발

### 2.2 제안 방법 개요  
1. **Translation**: 언어 모델(LM)에 NL 질문 $$Q$$를 입력 →  
   -  CNL: 문제 분해를 위한 자연어 소질문 및 의존성 그래프, 근거 표시  
   -  CSL: 각 소질문에 대응하는 기호 언어(Python, Datalog, PDDL 등) 코드  
2. **Problem Solving**: 외부 솔버가 CSL 실행 → 최종 답안 $$A$$ 생성  
   
### 2.3 수식 예시  
- 예: 정수 덧셈 문제  

1. 소질문 $$q_1$$: “초기 값은?” → 변수 선언  

```math
\text{n\_cars\_begin} = 3
```
  
  2. 소질문

```math
q_2: “추가된 값은?” → \text{n\_cars\_arrive} = 2
```
  
  3. 최종 계산:  

```math
       \text{n\_cars\_total} = \text{n\_cars\_begin} + \text{n\_cars\_arrive}
```
     
  → 실행 결과 $$5$$

### 2.4 모델 구조  
- **Translator**: GPT-계열 LM (Codex, GPT-4 등)  
- **Solver**:  
  -  Math Word Problems → Python 인터프리터  
  -  Multi-hop QA → Datalog/​Python  
  -  Planning → PDDL 플래너  
  -  Relational Inference → 커스텀 논리 엔진

### 2.5 성능 향상  
- **10개 벤치마크** 전반에서 표준 CoT 대비 성능 우위  
- 특히:  
  -  Math Word Problems: 최대 +14% 상대 정확도 향상  
  -  Relational Inference: 최대 +21.4%  
  -  Few-shot GPT-4: 7개 데이터셋 SOTA 달성(95%+ 정확도)

### 2.6 한계  
- **Translation 단계** 자체의 해석 가능성 미보장(모델 내부 과정 불투명)  
- Datalog 등 기호 언어 프리트레이닝 부족 시 **구문 오류** 및 **지식 표현 오류** 발생  
- Faithfulness≠Correctness: 여전히 우연 히트 및 잘못된 소논리 체계 발생 가능

## 3. 모델 일반화 성능 향상 가능성

- **다양한 기호 언어(SL)·솔버**와 결합하여 수학, QA, Planning, 관계 추론 등 **여러 도메인**에 적용  
- **모듈화된 구조**로, 새로운 도메인에 최소한의 기호 언어 사양 및 솔버를 추가하면 확장 용이  
- Translation 과정에서 NL·SL 분리를 통해 **사용자 개입**(subquestion 편집·디버깅) 가능  
- Datalog·PDDL 등 기호 언어를 더 폭넓게 학습시켜 **문법·지식 표현** 역량 강화 시, 더욱 견고한 일반화 기대

## 4. 향후 연구 영향 및 고려 사항

- **해석 가능성 연구**: Translation 과정 자체의 내부 논리를 드러낼 수 있는 메커니즘 탐색  
- **기호 언어 학습**: LM이 다양한 SL 구문·표현을 안정적으로 생성하도록 프리트레이닝 또는 파인튜닝 연구  
- **인간-모델 협업**: NL 소질문 단계에서 사용자 피드백을 반영한 상호작용적 디버깅 인터페이스 개발  
- **신뢰도 평가**: Faithfulness 외에도 *Comprehensiveness*, *Causality* 등 추가 평가 지표 정립  
- **리스크 관리**: 우연히 맞힌 경우나 잘못된 서브질문·코드에 대한 **자동 감지·수정** 기법 연구

> **결론**: Faithful CoT는 기호 언어 실행을 통한 설명의 신뢰성을 보장하며, 해석 가능성과 성능 향상을 동시에 달성한 혁신적인 프롬프트 방식으로, 다양한 추론 작업에서의 일반화 및 투명성 연구에 중요한 기반을 제공한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c6e5f1a9-59c1-4b45-b12a-ad987b48da95/2301.13379v3.pdf)
