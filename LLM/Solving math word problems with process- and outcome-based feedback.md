# Solving math word problems with process- and outcome-based feedback
# 핵심 요약

**Solving Math Word Problems with Process- and Outcome-Based Feedback**는 수학 단어 문제 해결 시[1]
-  **최종 답안(Outcome-based)**과 **추론 과정(Process-based)** 두 가지 감독(signal) 방식을 비교·분석  
-  **최종 답안 정확도**는 두 방식이 유사한 성능을 보이나,  
-  **추론 과정 오류(trace error)** 최소화를 위해서는 **과정 기반 피드백** 또는 이를 모방하는 **보상 모델(reward model)**이 필요함  

***

# 1. 논문이 해결하고자 하는 문제

수학 단어 문제를 풀 때, 모델을  
1) 최종 정답만 맞추도록 학습하는 **Outcome-based**  
2) 중간 추론 단계까지 올바르게 따라가도록 학습하는 **Process-based**  
두 접근법의 장단점을 비교하고,  
- 최종 정답 오류율(final-answer error)  
- 추론 과정 오류율(trace error)  
를 동시에 낮추는 최적의 학습·추론 체계를 제시하는 것  

***

# 2. 제안 방법

## 2.1 모델 구조 및 학습 절차

모두 **대형 사전학습 언어모델(LM)**에 기초하며, 세 가지 주요 구성요소로 이루어짐:  
- **SFT (Supervised Fine-Tuning)**  
  -  Process-based: GSM8K 데이터의 단계별 추론(trace) 전체를 정답으로 학습  
  -  Outcome-based: 최종 정답만 타깃으로 학습  
- **Reward Model (RM)**  
  -  ORM (Outcome-supervised): 최종 정답 일치 여부를 레이블로 학습  
  -  PRM (Process-supervised): 중간 단계 올바름 여부를 레이블로 학습  
- **RL via Expert Iteration**  
  -  ORM-RL / PRM-RL: RM 점수를 보상으로 사용하여 정책(policy)을 강화학습  
  -  Final-Answer RL: 최종 정답 맞추기를 직접 보상으로 사용  

## 2.2 핵심 수식

-  최종 정답 오류율:  

```math
\text{Final-error} = 1 - \frac{\#\text{정답 생성}}{\#\text{전체 문제}}
```

-  추론 과정 오류율:  

```math
\text{Trace-error} = \frac{\#\substack{\text{정답이지만}\\\text{중간 단계 오류}}}{\#\text{정답 생성}}
```  

-  RM 가중 디코딩(RM-weighted decoding):  

```math
f^* =\arg\max_f \sum_{y_i:f(y_i)=f}\text{rm\_prob}(y_i)
```

```math
y^* =\arg\max_{y:f(y)=f^*} \text{rm\_prob}(y)
``` 

***

# 3. 성능 향상 및 한계

| 방법                         | 최종 오류율 ↓ | 추론 오류율 ↓ |
|-----------------------------|-------------:|-------------:|
| SFT + PRM reranking         | 14.1%       | 3.4%        |
| SFT + Outcome-based RL + ORM| 12.7%       | 3.8%        |
| SFT + Final-Answer RL       | 20.2%       | 12.1%       |
| Few-shot + Final-Answer RL  | 23.5%       | 19.8%       |

- **Outcome vs. Process**: 최종 오류율은 **유사**하나, 추론 오류율은 **Process-based** 훨씬 낮음.[1]
- **보상 모델 활용**: ORM도 PRM 레이블을 잘 모방하여(ORM↔PRM 일치율 85% vs. ORM-label 일치율 77%),  
  이를 RL 보상으로 쓰면 trace error 대폭 감소.[1]
- **한계**:  
  1) 수학 문제 특성상 과정·결과 정합성이 강해 일반 도메인으로 일반화 불투명  
  2) 사람의 추론 과정 레이블 수집 비용 높음  

***

# 4. 일반화 성능 향상 관점

- **Selective Prediction**: RM 점수 기준으로 30% 포기 시 최종 오류율 14.1%→2.7% (5× 감소)  
- **OOD 테스트**: MATH 사전대수(pre-algebra) 제로샷 평가에서 SFT+ORM-RL이 63.2% 오류,  
  GPT-3(92.3%) 대비 현저히 우수하나 대규모 수학 특화 학습 모델에는 미치지 못함[1]
- **발견**: 과정 기반 피드백이 **다양한 분포 변동**에도 더 강건할 가능성  

***

# 5. 향후 연구의 영향 및 고려사항

- **피드백 스펙트럼**: 결과 중심↔과정 중심 사이에서 비용·안전성·이해 가능성 균형  
- **보상 모델 설계**: Outcome-based RM이 Process-based를 모방하는 메커니즘 심층 규명 필요  
- **도메인 확장**: 의료·법률 등 인간 설명이 중요한 분야에서 과정 기반 감독 효과 탐구  
- **비용 절감**: 중간 단계 레이블링 비용을 줄이기 위한 준지도·자기지도 학습 기법 개발

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9b4a3b5b-f7dc-4d4f-9a9d-085d3fbb2f77/2211.14275v1.pdf)
