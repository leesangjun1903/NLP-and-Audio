# Large Language Models Are Reasoning Teachers

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
대형 언어 모델(LLM)을 *추론 교사*(reasoning teacher)로 활용하여, 체인-오브-생각(chain-of-thought, CoT) 추론 데이터를 생성·큐레이션한 뒤 이를 소형 언어 모델에 미세조정(fine-tuning)하면, 소형 모델도 복잡한 다중 단계 추론 능력을 획득할 수 있다는 주장이다.[1]

**주요 기여**  
- **Fine-tune-CoT 방법론 제안**:  
  1) 대형 교사 모델에 제로-샷 CoT 프롬프트를 적용해 다단계 추론 예시 생성  
  2) 정답 필터링 및 특수 토큰을 통한 큐레이션  
  3) 소형 학생 모델 미세조정  
- **Diverse Reasoning**: 동일 샘플에 대해 온도 샘플링을 통해 다양한 추론 경로를 생성·증강하여 학생 모델 성능 대폭 향상.  
- **광범위 실험**: 0.3B~6.7B 파라미터 크기의 소형 모델 다수, 12개 복잡도 높은 추론 과제(arithmetic, symbolic, commonsense 등)에서 Fine-tune-CoT가 기존 제로/Few-shot CoT 및 일반 미세조정 대비 우수함을 입증.  

## 2. 문제 설정 및 제안 방법  
### 2.1 해결하고자 하는 문제  
- 프롬프트 기반 CoT는 **175B 이상의 거대 모델**에 의존하므로, **배포·추론 비용**이 매우 높음.  
- **소형 모델**(≤6.7B)에서는 직관(prompting)만으로는 복잡 추론이 불가능.  

### 2.2 Fine-tune-CoT 방법  
1) **추론 생성(Reasoning Generation)**  
   - 입력 질문 $$q_i$$에 대해 대형 교사 모델 $$T$$에게 “Let’s think step by step.” 지시로 CoT 추론 $$\hat r_i$$과 답 $$\hat a_i$$ 생성:  

$$ \text{Prompt: }Q: q_i\; A: \text{Let’s think step by step.}$$  

2) **큐레이션(Curation)**  
   - $$\hat a_i = a_i$$인 경우만 선별하여,  
     - 프롬프트

```math
p_i = \langle q_i\rangle\;\#\#\#
```
     
  - 완성도 $$c_i = \hat r_i\rightarrow\hat a_i\;\text{END}$$로 재구성  
3) **미세조정(Fine-tuning)**  
   - 소형 학생 모델 $$S$$을 위 큐레이션 데이터 $$\{(p_i,c_i)\}$$로 autoregressive 언어 모델링 목표로 학습  

### 2.3 Diverse Reasoning  
- 각 $$q_i$$에 대해 **온도 샘플링**(temperature $$T=0.7$$)으로 서로 다른 $$D$$개의 $$(\hat r_{ij},\hat a_{ij})$$를 생성·증강  
- **다양한 추론 경로**로 데이터 확장 시 모델 성능 최대화(특히 중간 사고 오류 보완)  

## 3. 모델 구조 및 성능  
### 3.1 모델 구조  
- **교사 모델**: InstructGPT(text-davinci-002, 175B)  
- **학생 모델**: GPT-3(ada, babbage, curie; 0.3B~6.7B), GPT-2, T5/Flan-T5 변형(0.06B~0.7B)  

### 3.2 성능 향상  
- **소형 모델에서도 CoT 능력 획득**: MultiArith에서 6.7B가 5%→33%로, CommonsenseQA에서 52.98%→76.17%로 개선.[1]
- **교사 능력 이상 달성**: Coin Flip, Shuffled Objects 등 단순 과제에서 1.3B·6.7B 학생 모델이 175B 교사 성능 초과.  
- **확장성**  
  - **다양성(D)**: $$D=64$$ 시 6.7B MultiArith 33%→53% 개선  
  - **데이터 규모**: 표본 수 증가 시 성능 선형적 상승(일반 미세조정 대비 안정적 확장)  
  - **교사 성능**: 더 강력한 교사→더 우수한 학생(양성 비례)  
  - **학생 크기**: 0.3B→6.7B 순으로 성능 일관 상승  

### 3.3 한계  
- **복잡 과제 한계**: GSM8K, AQUA 등 난도 높은 과제에서 10% 이하로 실용성 부족  
- **추론 오류 누적**: 중간 연산 오류 및 비논리적 사고 여전  
- **데이터 템플릿 의존**: 일부 데이터셋의 템플릿 유사도로 인한 과도한 일반화 가능성(템플릿별 분할 시에도 기본 성능 확보하나 감소)  

## 4. 일반화 성능 향상 관점  
- **다양한 추론 경로**로 단일 사례의 *다양성* 학습 → 소형 모델이 패턴 이상 인과 추론 가능  
- **교사-학생 규모 비례 학습**: 더 좋은 교사→더 일반화 가능한 학생 모델 형성  
- **데이터 증강 효과**: 소량의 원본 데이터로도 다양한 CoT 경로 생성 시, *데이터 효율* 높은 일반화 달성  

## 5. 향후 영향 및 고려사항  
- **연구 영향**:  
  - 대형 LLM의 CoT 능력을 *지식 증류(distillation)* 관점으로 활용, 소형 모델로 실전 배포 가능성 확대  
  - 다양한 CoT 추론 경로 활용한 데이터 증강 전략의 일반적 채택  
- **고려할 점**:  
  - **교사 모델의 바이어스·독성 전이** 방지: 필터링 강화 및 유해 콘텐츠 제거  
  - **프롬프트 기법 발전**: Self-consistency, self-improvement 등 최신 CoT 기법 통합  
  - **효율적 비용-성능 균형**: Diverse reasoning→추론 비용 상승 고려, PEFT(LoRA) 등 효율적 미세조정 적용  
  - **템플릿 의존 최소화**: 실제 비정형 데이터 일반화 평가 지표 개발 필요  

 attached_file:1[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2d0505c4-c4cb-4704-8041-31b1b0ac6202/2212.10071v2.pdf)
