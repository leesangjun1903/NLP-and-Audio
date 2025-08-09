# GPT-3 : Language Models are Few-Shot Learners

## 1. 핵심 주장 및 주요 기여  
이 논문은 대규모 언어 모델이 사전 학습만으로도 소수의 예시(few-shot)만을 제시했을 때 새로운 언어 과제를 빠르게 학습·적용할 수 있음을 보인다.  
- **핵심 주장**: 모델 크기를 크게 늘리면 추가 미세조정 없이도 문장 완성, 번역, 질문응답 등 다양한 과제에서 소수의 예시만으로도 강력한 성능을 낼 수 있다.  
- **주요 기여**:  
  1. 1750억 매개변수의 GPT-3 개발  
  2. zero-/one-/few-shot 설정에서 25여 개 이상의 NLP 벤치마크 평가  
  3. 소수 예시로도 상태-최첨단(fine-tuning) 기법과 비슷한 성능 달성  
  4. 데이터 오염(benchmark contamination) 분석 기법 제안  

## 2. 해결 과제 및 제안 방법  
### 2.1 문제 정의  
전통적으로 대형 사전학습 모델은 특정 과제별로 수만~수십만 개의 라벨 데이터를 이용해 **미세조정(fine-tuning)**해야 했다. 이는 데이터 수집 비용이 크고, 과제별로 재학습이 필요해 실용성이 낮았다.  

### 2.2 in-Context Learning (Few-Shot)  
논문이 제안하는 방식은:  
- 모델 파라미터를 고정한 채, 테스트 시점에 과제 설명과 K개의 예시(‘컨텍스트’)만을 입력으로 제공  
- 모델은 gradient 업데이트 없이 **순전파만**으로 과제를 수행  
- K = 0(zero-shot), 1(one-shot), 10∼100(few-shot) 등 다양한 설정 평가  

### 2.3 모델 구조  
- **Transformer 기반**의 autoregressive 언어 모델  
- GPT-2 아키텍처 확장, 전체 96개 블록, dmodel=12,288, 헤드 수 96, 파라미터 175B  
- 컨텍스트 윈도우 크기 nctx=2048 토큰  
- Sparse attention 적용해 300B 토큰 × 96층 규모 학습  

### 2.4 학습 및 수식  
- **언어 모델링 목표**:  
  $$\max_\theta \sum_{t} \log p_\theta(x_t \mid x_{ < t}) $$  
- 배치 정밀도(half-precision) + Adam(β₁=0.9, β₂=0.95, ϵ=1e-8)  
- weight decay 0.1, gradient clipping, cosine LR decay, warm-up 등  

## 3. 성능 향상 및 일반화  
### 3.1 Few-Shot 성능  
- **LAMBADA**: zero-shot 76.2% → few-shot 86.4% (종전 SOTA 대비 +18%p)  
- **TriviaQA**: zero-shot 64.3% → few-shot 71.2%, open-domain SOTA(RAG)와 동급  
- **Winogrande**: zero-shot 70.2% → few-shot 77.7%  
- **PIQA**: few-shot 82.8% (SOTA 79.4% 초과)  

### 3.2 Zero-/One-Shot 효과  
- zero-/one-shot에서도 상당한 수준 달성: 예시 없이 지시문만 줘도 60~70% 정확도  
- 모델 크기↑에 따라 zero→one→few 사이 격차가 더욱 커져, 대형 모델일수록 in-context 학습 능력 강화  

### 3.3 일반화 개선 관점  
- **스케일링 법칙**: 파라미터 수·학습 토큰 수·컴퓨팅량의 거듭 제곱(power-law) 관계 하에서 언어모델링 손실 및 downstream 성능이 꾸준히 개선됨  
- *Fine-tuning*보다 훨씬 **과제 간 이동성(transferability)**이 좋음  
- 사전학습 데이터 중 일부가 테스트셋에 포함되어도(“데이터 오염”) 성능 왜곡은 미미, 큰 규모 사전학습이 **오버피팅**을 억제  

## 4. 한계  
1. **긴 맥락/비교 과제 약함**: WIC, ANLI, 일부 독해 과제에서 여전히 무작위 수준  
2. **계산 비용**: 추론 단계에서도 거대해 실제 활용성 제약  
3. **Grounding 부재**: 외부 지식·물리 세계와의 연결 취약  
4. **추론 효율**: 순전파로 산술·논리 과제를 풀지만, 정확도는 수치 연산만료 부족  

## 5. 향후 연구 영향 및 고려사항  
- **Few-Shot 일반화 메커니즘 규명**: in-context learning이 ‘새로운 과제 학습’인지, 사전학습 중 본 과제 재식별인지 연구 필요  
- **모델 경량화·증류**: 거대 모델 지식 전달 가능한 소형 모델 개발  
- **다중 모달·행동 지향 학습**: 텍스트 외 시각·행동 맥락 포함한 사전학습  
- **편향·윤리적 사용**: 자동 감시·개입 메커니즘으로 남용·허위정보 생성 방지  
- **효율적 사전학습**: 데이터·컴퓨팅 요구 낮추는 알고리즘 탐색  

이 논문은 “사전학습→Few-Shot” 패러다임이 갖는 강력함을 보여주며, 거대 언어 모델이 과제적응(sample efficiency) 측면에서도 인간에 근접할 수 있음을 시사한다. 후속 연구는 소수 예시 학습의 본질, 비용·윤리적 과제, 그리고 모달리티 확장 가능성을 중점적으로 탐구해야 할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ec30bd9a-6fc6-4edc-9eb8-e7851c197dda/2005.14165v4.pdf
