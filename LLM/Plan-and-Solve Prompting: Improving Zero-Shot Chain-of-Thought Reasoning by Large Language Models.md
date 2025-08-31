# Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models

## 1. 핵심 주장과 주요 기여

**Plan-and-Solve (PS) Prompting**은 대형 언어 모델(LLM)의 zero-shot 추론 능력을 향상시키기 위한 새로운 프롬프팅 전략입니다. 기존 Zero-shot Chain-of-Thought (CoT)의 "Let's think step by step" 접근법을 발전시켜, **계획 수립**과 **단계별 실행**의 두 단계로 구분하여 더 체계적인 추론을 유도합니다.[1]

**주요 기여:**
- Zero-shot-CoT의 세 가지 주요 오류(계산 오류, 단계 누락 오류, 의미 오해 오류)를 분석하고 해결책 제시[1]
- 수동 예시 없이도 Few-shot CoT와 유사한 성능 달성[1]
- 10개 데이터셋에서 일관된 성능 향상 입증[1]

## 2. 해결하고자 하는 문제

### 2.1 Zero-shot-CoT의 한계점
GSM8K 데이터셋 100개 샘플 분석 결과, Zero-shot-CoT는 다음과 같은 오류를 보였습니다:[1]
- **계산 오류 (7%)**: 수치 계산 과정에서 발생하는 실수
- **단계 누락 오류 (12%)**: 복잡한 다단계 추론에서 중간 과정 생략
- **의미 오해 오류 (27%)**: 문제 이해 부족으로 인한 추론 연관성 부족

### 2.2 프롬프트 민감성 문제
GPT-3 모델이 프롬프트 표현에 매우 민감하여 신중한 프롬프트 설계가 필요한 문제점이 존재했습니다.[1]

## 3. 제안하는 방법

### 3.1 Plan-and-Solve (PS) Prompting
**기본 PS 프롬프트**:[1]
```
"Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step."
```

### 3.2 PS+ Prompting (향상된 버전)
**상세 지시사항이 포함된 PS+ 프롬프트**:[1]
```
"Let's first understand the problem, extract relevant variables and their corresponding numerals, and make a plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer."
```

### 3.3 두 단계 접근법

**1단계: 추론 생성**[1]
- 문제 이해 및 계획 수립
- 관련 변수 추출
- 단계별 실행

**2단계: 답안 추출**[1]
- 1단계에서 생성된 추론 텍스트에서 최종 답안 추출
- "Therefore, the answer (arabic numerals) is" 형태의 프롬프트 사용

## 4. 모델 구조 및 실험 설정

### 4.1 백본 모델
- **GPT-3 (text-davinci-003)** 175B 매개변수 모델 사용[1]
- 온도 설정: 0 (argmax sampling)
- Greedy decoding 전략 적용

### 4.2 평가 데이터셋
**수학적 추론 (6개 데이터셋)**:[1]
- GSM8K, SVAMP, MultiArith, AddSub, AQuA, SingleEq

**상식 추론 (2개 데이터셋)**:[1]
- CommonsenseQA, StrategyQA

**기호적 추론 (2개 데이터셋)**:[1]
- Last Letter Concatenation, Coin Flip

## 5. 성능 향상 결과

### 5.1 수학적 추론 성능
PS+ 프롬프팅이 Zero-shot-CoT 대비 **모든 수학 데이터셋에서 향상**을 보였습니다:[1]

| 데이터셋 | Zero-shot-CoT | PS+ (제안방법) | 향상률 |
|----------|---------------|----------------|--------|
| MultiArith | 83.8% | 91.8% | +8.0% |
| GSM8K | 56.4% | 59.3% | +2.9% |
| AddSub | 85.3% | 92.2% | +6.9% |
| AQuA | 38.9% | 46.0% | +7.1% |
| SingleEq | 88.1% | 94.7% | +6.6% |
| SVAMP | 69.9% | 75.7% | +5.8% |
| **평균** | **70.4%** | **76.7%** | **+6.3%** |

### 5.2 Self-Consistency와의 결합
Self-Consistency 기법과 결합했을 때 추가적인 성능 향상을 보였습니다:[1]
- GSM8K: 58.7% → 73.7% (+15.0%)
- SVAMP: 75.7% → 84.4% (+8.7%)

### 5.3 오류 감소 효과
PS+ 프롬프팅은 오류 유형별로 다음과 같은 개선을 보였습니다:[1]

| 오류 유형 | Zero-shot-CoT | PS+ | 개선 |
|-----------|---------------|-----|------|
| 계산 오류 | 7% | 5% | -2% |
| 단계 누락 | 12% | 7% | -5% |
| 의미 오해 | 27% | 27% | 0% |

## 6. 일반화 성능 향상 가능성

### 6.1 도메인 간 일반화
PS+ 프롬프팅은 **세 가지 서로 다른 추론 유형**에서 일관된 성능 향상을 보여 높은 일반화 가능성을 시사합니다:[1]
- 수학적 추론: 평균 6.3% 향상
- 상식 추론: CommonsenseQA 6.7%, StrategyQA 1.6% 향상
- 기호적 추론: Last Letter 10.4%, Coin Flip 2.8% 향상

### 6.2 계획 능력의 일반화
무작위 샘플링한 100개 예측 중 **90개에서 계획이 포함**되어 있어, 최신 LLM들의 강력한 계획 능력 emergence를 확인했습니다.[1]

### 6.3 프롬프트 커스터마이제이션
PS+ 전략은 **다양한 문제 유형에 맞게 쉽게 커스터마이징**할 수 있어, 수학 추론 외에도 상식 추론, 기호적 추론 문제에 적용 가능합니다.[1]

## 7. 한계점

### 7.1 프롬프트 설계의 복잡성
LLM이 정확한 추론 단계를 생성하도록 유도하는 프롬프트 설계에 **상당한 노력이 필요**합니다. GPT-3 모델이 프롬프트 표현에 매우 민감하여 신중한 설계가 요구됩니다.[1]

### 7.2 의미 오해 오류 지속
PS+ 프롬프팅이 계산 오류와 단계 누락 오류는 효과적으로 해결했지만, **의미 오해 오류는 여전히 27%로 개선되지 않았습니다**.[1]

### 7.3 LLM 의존성
제안된 방법은 **LLM의 내재적 능력에 크게 의존**하며, 모델 업그레이드 없이는 근본적인 추론 능력 향상에 한계가 있습니다.[1]

## 8. 향후 연구에 미치는 영향

### 8.1 Zero-shot 프롬프팅 패러다임 발전
- **체계적 계획 수립**의 중요성을 입증하여 향후 프롬프팅 연구의 새로운 방향 제시[1]
- Few-shot 예시 없이도 경쟁력 있는 성능 달성 가능성 확인[1]

### 8.2 프롬프트 엔지니어링 체계화
- **상세한 지시사항**의 효과를 정량적으로 입증[1]
- 프롬프트 구성 요소별 영향 분석을 통한 체계적 접근법 제공[1]

### 8.3 계획 기반 AI 시스템 발전
- **Plan → Execute** 패러다임의 효과성 검증[1]
- 복잡한 추론 작업을 하위 작업으로 분해하는 접근법의 타당성 확인[1]

## 9. 향후 연구 고려사항

### 9.1 의미 오해 오류 해결
프롬프팅만으로 **의미 오해 오류를 해결하는 방법** 탐구가 필요합니다. LLM 업그레이드 외의 접근법 연구가 요구됩니다.[1]

### 9.2 비추론 작업으로의 확장
PS+ 프롬프팅을 **비추론 작업에 적용**하는 연구와 **계획 정제(plan refinement)** 기법 개발이 향후 연구 과제입니다.[1]

### 9.3 자동화된 프롬프트 생성
프롬프트 설계의 복잡성을 해결하기 위한 **자동화된 프롬프트 생성 및 최적화** 연구가 필요합니다.

### 9.4 다양한 LLM에서의 검증
GPT-3 외의 **다양한 LLM 아키텍처**에서의 일반화 성능 검증이 요구됩니다.

이 연구는 **zero-shot 프롬프팅의 새로운 패러다임**을 제시하며, 향후 LLM의 추론 능력 향상을 위한 중요한 기초를 마련했습니다. 특히 **계획 수립의 체계화**와 **상세 지시사항의 효과**를 입증함으로써, 프롬프트 엔지니어링 분야의 발전에 기여할 것으로 예상됩니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b03e7b0f-061f-43f8-869d-4806fc3c6a54/2305.04091v3.pdf)
