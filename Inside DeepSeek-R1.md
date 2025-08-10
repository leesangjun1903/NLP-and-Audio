# Inside DeepSeek-R1 : AI 추론 혁신의 내막

## 개요

**DeepSeek-R1**은 중국 AI 기업 DeepSeek이 개발한 혁신적인 추론 모델로, OpenAI의 o1 시리즈와 경쟁하는 성능을 보여주면서도 전혀 다른 접근 방식으로 개발되었습니다. 이 모델의 가장 주목할 만한 특징은 **순수한 강화학습(RL)만으로 추론 능력을 학습**했다는 점입니다.[1]

## 기존 AI 훈련의 한계

### 전통적인 지도 학습의 문제점

기존의 대형 언어 모델들은 주로 **지도 학습(Supervised Learning)**을 통해 훈련되었습니다. 이 방식의 핵심 문제점들은 다음과 같습니다:[2][3]

- **패턴 모방에 불과**: 모델이 실제로 추론하는 것이 아니라 인간이 작성한 예시를 단순히 모방
- **인간 데이터의 한계**: 모델의 능력이 훈련 데이터의 품질과 범위에 제한됨
- **일반화 능력 부족**: 훈련 예시와 다른 문제에서는 성능이 급격히 저하[4]

### 전통적인 Chain-of-Thought의 한계

기존의 **Chain-of-Thought(CoT)** 추론도 본질적으로는 인간이 작성한 단계별 사고 과정을 모방하는 것에 불과했습니다. 진정한 추론이 아닌 겉보기에만 논리적인 과정을 따라하는 것이었죠.[5][6]

## DeepSeek-R1의 혁신적 접근법

### 1. DeepSeek-R1-Zero: 순수 강화학습의 시작

DeepSeek의 가장 혁신적인 시도는 **DeepSeek-R1-Zero** 모델입니다. 이 모델은:

- **지도 학습 없이 시작**: 기존과 달리 인간이 작성한 추론 예시를 전혀 사용하지 않음
- **강화학습으로 추론 발견**: 모델이 스스로 추론하는 방법을 찾아냄
- **자연스러운 추론 행동 창발**: 자기 검증, 성찰, 긴 추론 과정 등이 자연스럽게 나타남[7][1]

하지만 R1-Zero는 다음과 같은 문제점들이 있었습니다:
- 가독성 부족
- 언어 혼용 (중국어와 영어 섞임)
- 구조화되지 않은 출력

### 2. GRPO: 핵심 기술 메커니즘

DeepSeek-R1의 핵심 기술은 **Group Relative Policy Optimization(GRPO)**입니다:[8][9][10]

#### GRPO의 작동 원리

1. **여러 답변 생성**: 하나의 질문에 대해 모델이 여러 개의 답변을 생성
2. **그룹 평균 계산**: 이 답변들의 평균 점수를 계산
3. **상대적 우위 결정**: 평균보다 좋은 답변은 강화, 나쁜 답변은 약화
4. **정책 업데이트**: 더 나은 답변을 생성할 확률을 높임

#### 기존 PPO와의 차이점

- **Critic 모델 불필요**: 별도의 가치 평가 네트워크가 필요 없음
- **메모리 효율성**: 훨씬 적은 GPU 메모리로 학습 가능
- **안정성**: 더 안정적인 학습 과정[10][11]

## DeepSeek-R1의 4단계 개발 과정

DeepSeek-R1은 R1-Zero의 문제점들을 해결하기 위해 **4단계의 체계적인 개발 과정**을 거쳤습니다:

### 1단계: Cold-Start 파인튜닝
- **목적**: R1-Zero의 혼란스러운 출력을 안정화
- **방법**: 소량의 고품질 인간 데이터로 출력 형식 구조화
- **결과**: 읽기 가능한 형태의 추론 과정 생성

### 2단계: 추론 중심 강화학습
- **목적**: 추론 정확도 향상 및 언어 일관성 확보
- **방법**: GRPO를 사용한 복합 보상 함수 적용
- **평가 기준**: 수학, 코딩, 과학, 논리 추론 등

### 3단계: Rejection Sampling + SFT
- **목적**: 일반적인 대화 능력 확보
- **방법**: 80만 개의 자체 생성 추론 데이터와 20만 개의 일반 작업 데이터 결합
- **결과**: STEM 전문성과 일반 대화 능력의 균형

### 4단계: 정렬 및 안전성 강화학습
- **목적**: 도움이 되고 안전한 모델로 최종 조정
- **방법**: 혼합 보상 시스템 (규칙 기반 + 선호도 모델)

## 성능과 영향력

### 벤치마크 성능

DeepSeek-R1은 여러 주요 벤치마크에서 OpenAI o1과 경쟁하는 성능을 보여줍니다:[12][1]

- **AIME 2024**: 79.8% (Pass@1)
- **Codeforces**: 96.3 백분위
- **GPQA Diamond**: 71.5%
- **MATH-500**: 97.3%

### 비용 효율성

가장 놀라운 점은 **훨씬 적은 비용으로 개발**되었다는 것입니다:[13]
- 미국의 GPU 수출 제재 하에서도 개발 성공
- 서구 AI 기업 대비 훨씬 작은 GPU 클러스터 사용
- 혁신적인 효율성의 증명

### 오픈소스 기여

DeepSeek은 연구 커뮤니티를 위해 다음을 공개했습니다:[14][7]
- DeepSeek-R1-Zero 모델
- DeepSeek-R1 모델  
- 6개의 증류된 소형 모델 (1.5B~70B 파라미터)

## 모델 증류: 작은 모델의 힘

### 증류 과정의 혁신

DeepSeek-R1의 또 다른 중요한 성과는 **모델 증류(Model Distillation)**를 통한 소형 모델 개발입니다:[15][16][17]

- **교사 모델**: 대형 R1 모델
- **학생 모델**: LLaMA3, Qwen2.5 기반 소형 모델들
- **학습 방식**: 표준 지도학습만으로도 추론 능력 전수 성공

### 증류 모델의 성능

특히 **DeepSeek-R1-Distill-Qwen-32B**는 OpenAI o1-mini를 여러 벤치마크에서 능가하며, 소형 모델의 새로운 가능성을 제시했습니다.[7]

## 한계와 도전과제

### 기술적 한계

1. **토큰 집약적**: 높은 정확도를 위해 많은 토큰 생성 필요[12]
2. **추론의 '단점'**: 특정 추론 길이에서 성능이 오히려 저하되는 현상[18]
3. **안전성 취약점**: 추론 과정이 공개되어 악용 가능성 존재[19]

### 실용성 고려사항

- **속도 vs 정확도**: 빠른 응답이 필요한 용도에는 부적합
- **비용 효율성**: 토큰 생성량이 많아 실제 운영 비용이 높을 수 있음

## 미래에 대한 시사점

### AI 개발 패러다임의 변화

DeepSeek-R1은 다음과 같은 중요한 변화를 시사합니다:[20][4]

1. **강화학습의 우위**: 제한된 데이터 환경에서는 RL이 SFT보다 효과적
2. **체계적 개선의 중요성**: 각 단계에서 특정 문제를 해결하는 순차적 접근
3. **추론의 창발성**: 추론은 가르칠 수 없고 스스로 발견해야 하는 능력

### 글로벌 AI 경쟁력

- **기술 격차 해소**: 서구 기업들과의 기술적 차이가 급속히 줄어들고 있음
- **비용 혁신**: 동일한 성능을 훨씬 적은 비용으로 달성
- **오픈소스 기여**: 전체 AI 생태계 발전에 기여

## 결론

DeepSeek-R1은 단순히 새로운 AI 모델이 아닙니다. 이는 **AI가 추론하는 방법에 대한 근본적인 패러다임 전환**을 보여줍니다. 인간의 예시를 모방하는 것에서 벗어나 스스로 추론 전략을 발견하도록 하는 것, 이것이야말로 진정한 AI의 지능에 한 걸음 더 가까워진 것이라 할 수 있습니다.

더 나아가, 체계적인 개선 과정, 효율적인 강화학습 기법, 그리고 소형 모델로의 성공적인 지식 전수는 앞으로의 AI 개발 방향을 제시하는 중요한 이정표가 될 것입니다. **추론은 창발적이며, 모방이 아닌 인센티브를 통해서만 진정으로 학습할 수 있다**는 DeepSeek의 핵심 통찰은 AI 연구의 새로운 장을 열고 있습니다.

[1] https://arxiv.org/abs/2501.12948
[2] https://arxiv.org/abs/2401.15170
[3] https://thebasics.tistory.com/312
[4] https://arxiv.org/abs/2501.17161
[5] https://arxiv.org/abs/2503.08679
[6] https://orq.ai/blog/what-is-chain-of-thought-prompting
[7] https://huggingface.co/deepseek-ai/DeepSeek-R1
[8] https://aiengineering.academy/LLM/TheoryBehindFinetuning/GRPO/
[9] https://www.oxen.ai/blog/why-grpo-is-important-and-how-it-works
[10] https://huggingface.co/blog/NormalUhr/grpo
[11] https://arxiv.org/abs/2503.06639
[12] https://arxiv.org/abs/2501.18576
[13] https://arxiv.org/abs/2502.02523
[14] https://github.com/deepseek-ai/DeepSeek-R1
[15] https://deepinfra.com/blog/model-distillation
[16] https://lablab.ai/t/building-efficient-ai-models-with-openais-model-distillation-a-comprehensive-guide
[17] https://datasciencedojo.com/blog/understanding-knowledge-distillation/
[18] https://www.semanticscholar.org/paper/cb81fe5812916beb915bdf426d28802df99e6df1
[19] https://arxiv.org/abs/2502.12893
[20] https://predibase.com/blog/how-reinforcement-learning-beats-supervised-fine-tuning-when-data-is-scarce
[21] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/6dff60bf-51b0-4252-b310-44006fd956b3/How_AI_Learns_to_Reason_DeepSeek_R1_Case_Study_1752211947.pdf
[22] https://www.semanticscholar.org/paper/3f0455e9dc4bca6a80bf7c50cd1279249bc65abc
[23] https://tijer.org/tijer/viewpaperforall.php?paper=TIJER2505178
[24] https://ieeexplore.ieee.org/document/11012057/
[25] https://arxiv.org/abs/2503.11655
[26] https://journals.lww.com/10.1097/JS9.0000000000002386
[27] https://arxiv.org/pdf/2501.12948.pdf
[28] https://arxiv.org/pdf/2502.10928.pdf
[29] https://arxiv.org/pdf/2503.11655.pdf
[30] http://arxiv.org/pdf/2503.05132.pdf
[31] https://arxiv.org/html/2504.07615v1
[32] https://arxiv.org/html/2503.10573v1
[33] https://arxiv.org/pdf/2502.17947.pdf
[34] https://arxiv.org/html/2503.17352
[35] https://arxiv.org/pdf/2501.18576.pdf
[36] https://arxiv.org/pdf/2503.16219.pdf
[37] https://anshadameenza.com/blog/technology/google-reasoning-ai-breakthrough/
[38] https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/
[39] https://www.ainvest.com/news/ai-breakthrough-math-reasoning-catalyst-stage-ai-innovation-2507/
[40] https://research.google/blog/google-research-2024-breakthroughs-for-impact-at-every-scale/
[41] https://openai.com/index/learning-to-reason-with-llms/
[42] https://www.seangoedecke.com/deepseek-r1/
[43] https://blog.google/technology/ai/2024-ai-extraordinary-progress-advancement/
[44] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/deepseek-r1/
[45] https://jaylala.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D-with-Python-GRPO%EB%9E%80-Group-Relative-Policy-Optimization
[46] https://finance.yahoo.com/news/top-13-artificial-intelligence-ai-222907215.html
[47] https://littlefoxdiary.tistory.com/131
[48] https://arxiv.org/pdf/2402.03300.pdf
[49] https://arxiv.org/abs/2505.20096
[50] https://arxiv.org/abs/2407.00087
[51] https://arxiv.org/abs/2503.14521
[52] https://arxiv.org/abs/2504.16913
[53] https://dergipark.org.tr/en/doi/10.18621/eurj.1672422
[54] https://ieeexplore.ieee.org/document/11015950/
[55] https://arxiv.org/abs/2506.23678
[56] https://arxiv.org/abs/2504.17091
[57] https://arxiv.org/abs/2410.05695
[58] http://arxiv.org/pdf/2309.15402.pdf
[59] https://arxiv.org/html/2503.05179v1
[60] https://aclanthology.org/2023.findings-emnlp.985.pdf
[61] https://arxiv.org/pdf/2301.11596.pdf
[62] http://arxiv.org/pdf/2502.12134.pdf
[63] https://arxiv.org/pdf/2201.11903v1.pdf%5C.pdf
[64] https://aclanthology.org/2023.findings-emnlp.452.pdf
[65] https://arxiv.org/pdf/2406.12255.pdf
[66] http://arxiv.org/pdf/2410.02167.pdf
[67] https://www.linkedin.com/pulse/supervised-fine-tuning-vs-reinforcement-learning-model-sowmya-vivek-txnfc
[68] https://www.nvidia.com/en-us/glossary/cot-prompting/
[69] https://www.youtube.com/watch?v=Ko0jM3YjprY
[70] https://www.coursera.org/articles/chain-of-thought-prompting
[71] https://platform.openai.com/docs/guides/reinforcement-fine-tuning
[72] https://www.ibm.com/think/topics/chain-of-thoughts
[73] https://arxiv.org/abs/2503.11197
[74] https://labelbox.com/blog/a-pragmatic-introduction-to-model-distillation-for-ai-developers/
[75] https://arxiv.org/abs/2201.11903
[76] https://www.invisible.co/blog/supervised-fine-tuning-vs-rlhf-how-to-choose-the-right-approach-to-train-your-llm
[77] https://labelbox.com/guides/model-distillation/
[78] https://www.promptingguide.ai/techniques/cot
[79] https://www.reddit.com/r/MachineLearning/comments/10rpj0f/d_why_do_llms_like_instructgpt_and_llm_use_rl_to/
