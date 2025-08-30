# LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models

**핵심 주장**  
LLM-Adapters는 파라미터 효율적 파인튜닝(PEFT) 기법을 대형 언어 모델(LLM)에 적용하기 위한 통합 프레임워크로, 소수의 추가 파라미터만으로도 전체 모델의 성능에 근접하거나 이를 뛰어넘는 효과를 낼 수 있음을 입증한다.[1]

**주요 기여**  
1. **통합 프레임워크 제공**: LLaMA, BLOOM, GPT-J 등 다양한 오픈소스 LLM에 Series Adapter, Parallel Adapter, LoRA, Prompt Tuning 등의 PEFT 모듈을 쉽게 적용할 수 있는 도구를 개발.[1]
2. **종합적 실험 연구**: 14개 수학·상식 추론 데이터셋에 걸쳐 어댑터 종류, 삽입 위치, 하이퍼파라미터 구성을 체계적으로 비교 분석하여 최적 설정을 제시.[1]
3. **소형 모델 성능 검증**: LLaMA-13B 모델에 LoRA를 적용할 경우 GPT-3.5(≈175B)의 성능을 특정 과제(예: MultiArith, AddSub, SingleEq)에서 능가함을 확인.[1]
4. **일반화 분석**: ID(분포 내) 파인튜닝 데이터로 상식 추론 과제에서 ChatGPT, PaLM(540B)을 뛰어넘는 성능을 달성하고, OOD(분포 외) 수학 과제에서도 소형 모델이 경쟁력 있는 성능을 보임을 시사.[1]

# 해결 문제 및 제안 방법

## 해결하고자 하는 문제  
- 전체 파라미터를 업데이트하는 *Full-Model Fine-Tuning*의 높은 연산·저장 비용  
- 다양한 PEFT 모듈의 최적 배치와 설정이 불명확하여 사용자 혼란

## 제안한 방법  
LLM-Adapters 프레임워크 내에서 네 가지 대표 PEFT 모듈을 구현하고, 다음과 같은 수식으로 이들의 동작을 정의 및 평가한다.[1]

1. **Prefix Tuning (프롬프트 기반)**  

$$
   H_o = \text{Attn}(H_i W^Q, [P_K;H_i W^K], [P_V;H_i W^V])
   $$  
   
- $$P_K,P_V \in \mathbb{R}^{L\times d}$$: 학습 가능한 가상 토큰

2. **LoRA (재파라미터화 기반)**  

$$
   H_o = H_i W_0 + H_i \Delta W = H_i W_0 + H_i B A
   $$  
   
- $$B\in \mathbb{R}^{r\times d}, A\in \mathbb{R}^{r\times d}$$, 낮은 랭크 $$r\ll d$$

3. **Series Adapter**  

$$
   H_o \leftarrow H_o + f(H_o W_\text{down}) W_\text{up}
   $$  
   
- 병렬이 아닌 순차적 삽입, 다운/업 프로젝션

4. **Parallel Adapter**  

$$
   H_o \leftarrow H_o + f(H_i W_\text{down}) W_\text{up}
   $$  
   
- 기존 레이어와 병렬로 어댑터 모듈 결합

## 모델 구조 및 설정  
- 베이스 모델: LLaMA (7B, 13B), BLOOMz7B, GPT-J6B  
- 최적 배치:  
  - Prefix Tuning: 입력 임베딩 앞에 가상 토큰 10개  
  - Series/Parallel Adapter: MLP 레이어 뒤, 병목 크기 256  
  - LoRA: Attention 및 MLP 레이어 양쪽, 랭크 32[1]

# 성능 향상 및 한계

## 성능 향상  
- **상식 추론**: LLaMA-13B+Parallel Adapter가 평균 81.5% 정확도로 ChatGPT(≈77.0%) 대비 +4.5%p 우위.[1]
- **수학 추론**: LLaMA-13B+LoRA가 MultiArith, AddSub, SingleEq에서 GPT-3.5를 수치적으로 상회하며(예: AddSub 87.3% vs. 56.4%).[1]

## 한계  
1. **대형 모델 미평가**: LLaMA-33B·65B 등 더 큰 모델에서의 PEFT 효과 미검증  
2. **어댑터 조합 미탐색**: 여러 PEFT 모듈 동시 적용 시 시너지 및 최적 탐색 공간 미고려[1]

# 일반화 성능 향상 관련 내용

- **ID 시나리오**(분포 내): Commonsense170K 데이터로 파인튜닝한 소형 모델이 대형 LLM을 능가하여, 풍부한 ID 데이터가 있으면 소형 PEFT 모델이 일반화에서 우위 가능.[1]
- **OOD 시나리오**(분포 외): Math10K 데이터로 파인튜닝한 경우 GSM8K, AQuA에서는 대형 모델 대비 성능 차 존재하나, MultiArith·AddSub 등 단순 과제에서는 소형 PEFT 모델이 경쟁력 확보.[1]

# 향후 연구 영향 및 고려사항

- **영향**: PEFT 기법이 소규모 연구실 및 산업 현장에서 대형 LLM 활용의 비용장벽을 크게 낮추며, 다양한 태스크 특화 모델 개발을 가속화할 것.  
- **고려사항**:  
  - 어댑터 모듈 간 조합 및 모달리티 확장에 따른 최적화  
  - 대형 베이스 모델에서의 PEFT 적용 효과 및 메모리·추론 속도 트레이드오프 분석  
  - 분포 외 일반화 성능 강화 기법(예: 데이터 증강, 메타러닝) 연구.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e59e5e6c-dd55-4a26-b59f-24f18af14fc3/2304.01933v3.pdf)
