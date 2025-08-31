# ReAct: Synergizing Reasoning and Acting in Language Models

**핵심 주장 및 기여**  
ReAct는 대형 언어 모델(LLM)이 *추론(trace of thought)*과 *행동(action)*을 상호 교차(interleaved) 방식으로 수행하도록 유도하여,  
1) **추론이 행동을 유도**해 계획(plan)을 수립·수정하고  
2) **행동이 추론에 피드백**을 주어 외부 지식을 통합함으로써  
기존의 체인오브쏘트(CoT)나 행동 생성(action-only) 방식 대비 광범위한 언어 이해 및 상호작용 과제에서 성능과 해석 가능성을 동시에 향상시킨다.[1]

***

## 1. 해결하고자 하는 문제  
- **추론–행동 분리**: CoT는 내부 지식에만 의존해 사실 오류(hallucination)와 오차 전파(error propagation)에 취약.  
- **행동 단독**: 웹 인터랙션만 수행해 추론 능력이 결여돼 복잡한 문제 해결에 한계.  

이를 극복하기 위해, *추론과 행동을 유기적으로 결합*해 동적이고 반응적(reason-reactive)인 의사결정 체계를 구축하고자 한다.[1]

***

## 2. 제안하는 방법  
### 2.1. 모델링  
- **확장된 행동 공간** $$\hat A = A \cup L$$  
  - $$A$$: 도메인별 실제 행동(예: `search[query]`, `lookup[string]`, `finish[answer]`)  
  - $$L$$: 언어로 된 추론(“thought”)  
- **정책 학습**: 고정된 LLM을 *few-shot prompting*으로 사용  
  - 입력 $$\mathbf{c}\_t=(o_1,a_1,\dots,o_{t-1},a_{t-1},o_t)$$ 에 대해  

$$
      \pi(\hat a_t \mid \mathbf{c}_t)
    $$  
    
  을 생성하여, $$\hat a_t\in A$$일 때 실행하고, $$\hat a_t\in L$$일 때 *추론문*을 문맥에 추가  
- **ReAct 프롬프트 구성**:  
  - 다중 샷 예시(HotpotQA: 6개, FEVER: 3개) 제공  
  - 각 예시는 $$\lessdot$$사고–행동–관찰$$\gtrdot$$ 반복 구조로 이루어짐  

### 2.2. 수식  
- 동적 계획 수립 및 수정:  

$$
    \mathbf{c}_{t+1} = (\mathbf{c}_t,\,\hat a_t)
  $$  

- CoT–ReAct 결합 전략:  
  - **ReAct→CoT-SC**: 주어진 단계 내에 답을 도출하지 못하면 self-consistency(CoT-SC)로 백오프  
  - **CoT-SC→ReAct**: CoT-SC 앙상블 결과 불확실할 때 ReAct로 백오프  

***

## 3. 모델 구조  
- **기반 모델**: PaLM-540B (및 GPT-3 검증)  
- **프롬프트 구성 요소**:  
  1. **Thought**: 다음 행동을 안내하는 언어적 추론  
  2. **Action**: 외부 지식베이스(Wikipedia API, 텍스트 게임 환경, 웹상 쇼핑 사이트)와 상호작용  
  3. **Observation**: 행동 결과로 얻은 텍스트 정보  
- **미세조정(fine-tuning)**: ReAct 궤적 3,000개로 PaLM-8B/62B finetune, prompting 대비 일반화 성능 대폭 향상  

***

## 4. 성능 향상 및 한계  
### 4.1. 성능 향상  
- **HotpotQA**: CoT-SC 대비 3–8% EM↑, hallucination 대폭 감소[1]
- **FEVER**: 4.2% Acc↑, *사실 정확성* 개선[1]
- **ALFWorld**: Act-only 대비 성공률 34%p↑, 일관된 성능 부스트  
- **WebShop**: Act vs. ReAct 성공률 30.1%→40.0%↑, *복잡한 웹 상호작용*에서 우월  

### 4.2. 한계 및 오류 모드  
- **추론 오류**: 구조적 제약으로 CoT 대비 추론 유연성 감소  
- **검색 오류**: 부적절 검색 시 회복 어려움(전체 오류의 23%)  
- **라벨 불일치**: 데이터셋 라벨의 시대착오성(outdated labels)과의 충돌  
- **반복 루핑**: 과도한 greedy 디코딩으로 thought–action 반복 발생  

***

## 5. 일반화 성능 및 확장성  
- **미세조정**: 소규모 모델(PaLM-8B/62B)에서도 3,000개 traj로 큰 성능 이득 확인  
- **다중 태스크 학습**: 다양한 환경–작업에 범용적으로 적용 가능성  
- **추론–행동 균형**: external acting으로 factual grounding, internal reasoning으로 구조적 유연성 확보  

***

## 6. 향후 연구에 미치는 영향 및 고려사항  
- **인간-기계 협업**: 온-더-플라이 thought 편집을 통한 실시간 정책 수정 가능  
- **안전 및 윤리**: 외부 API·웹 상호작용의 잠재적 위험(사생활·유해행동) 관리 필요  
- **디코딩 전략 개선**: beam search 등 더 정교한 디코딩으로 반복·루핑 완화  
- **대규모 파인튜닝**: 고품질 인간 주석 확충으로 더 높은 일반화 성능 달성  

ReAct는 *추론*과 *행동*을 통합함으로써 LLM의 *실세계 지식* 활용과 *내적 추론* 능력을 동시 개선하며, 향후 범용 지능형 에이전트 설계의 새로운 패러다임을 제시한다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5c8378aa-5145-486a-9c17-3e86b543a0b4/2210.03629v3.pdf)
