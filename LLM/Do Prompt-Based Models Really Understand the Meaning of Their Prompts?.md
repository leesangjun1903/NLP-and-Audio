# Do Prompt-Based Models Really Understand the Meaning of Their Prompts?

## 1. 핵심 주장 및 주요 기여  
이 논문은 **프롬프트 기반 언어모델이 인간처럼 프롬프트 의미를 이해하여 성능을 개선하는지 의문**을 제기한다.  
핵심 발견은 다음과 같다.  
- **무관하거나 오도된(혹은 심지어 병리적) 프롬프트**를 사용해도, 모델은 **정상적인 프롬프트만큼 빠르게 학습**한다.  
- **프롬프트 의미보다는** 오히려 **타겟 단어(yes/no 등)의 선택**이 모델 성능에 더 큰 영향을 미친다.  
- 다양한 크기(235M∼175B 파라미터)와 **instruction-tuned(T0) 모델**에서도 동일한 현상이 관찰된다.  

이로써 “프롬프트는 인간이 이해하는 방식으로 언어모델 학습을 돕는다”는 가정에 **심각한 의문**을 제기했다.

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 문제 정의  
- **목표**: 프롬프트가 언어모델에 *의미 있는 작업 지시(instructions)*를 제공하여 성능을 향상시키는지 검증  
- **평가 과제**: 자연어 추론(NLI), 특히 RTE(Recognizing Textual Entailment) 데이터셋  

### 2.2 실험 방법  
1. **프롬프트 카테고리**  
   - *Instructive*: 인간 지시문과 유사한 정교한 템플릿  
   - *Misleading-Moderate/Extreme*: NLI와 무관·오도된 지시문  
   - *Irrelevant*: 완전히 무관한 문장 삽입  
   - *Null*: 단순 문장 연결, 혹은 마스크만 삽입  
2. **LM 타겟 단어 카테고리**  
   - Yes–No, Yes–No 유사, Arbitrary(‘cat/dog’), Reversed(‘no/yes’) 등  
3. **모델**  
   - ALBERT(235M), T5 variants(770M, 3B, 11B), instruction-tuned T0(3B, 11B), GPT-3 (175B, priming)  
4. **평가 지표**: 0, 4, 8, 16, …, 256샷에서의 validation accuracy 추적  

### 2.3 주요 수식  
- **Rank Classification**  

$$
    \hat{y} = \arg\max_{w \in \{\text{target words}\}} P(w \mid \text{prompt+input})
  $$

- **ANOVA / Kruskal–Wallis 검정** 및 Bonferroni 보정으로 템플릿·타겟별 유의미한 차이 분석  

### 2.4 모델 구조  
- **Discrete Prompt Tuning**: 입력 앞뒤에 자연어 템플릿 삽입 후 전체 모델 파라미터를 업데이트  
- **Priming**(GPT-3): gradient 업데이트 없이 컨텍스트에 예시 삽입  

### 2.5 성능 향상 및 한계  
- **프롬프트 카테고리별**:  
  - *Instructive vs. Irrelevant*: 차이 거의 없음  
  - *Instructive vs. Misleading*: 부분적 차이, 그러나 절대 성능은 병리적 프롬프트에서도 높음  
  - *Null*: 유의하게 낮음  
- **타겟 단어별**:  
  - *Yes/No* 우위, *Arbitrary·Reversed* 현저히 저조  
  - 그러나 “agree/disagree” vs. “yes/no”에서도 큰 성능 격차 발생  
- **한계**:  
  - 프롬프트 의미 이해 여부 판별이 어려운 혼합 결과  
  - 여전히 병리적 조합에서 높은 성능을 보임 → *진정한 의미 이해 vs. 통계적 유사도 학습* 구분 불가  

## 3. 모델 일반화 성능 향상 관점  
- **Instruction Tuning (T0)**:  
  - Zero-shot에서 다양한 프롬프트에 대한 안정성이 크게 향상됨  
  - 그러나 일부 병리적 템플릿에서도 70% 이상의 정확도 획득 → 과도한 *Robustness*  
- **Few-Shot Transfer**:  
  - RTE에서 학습한 모델이 **ANLI, HANS, WSC** 등 타 과제에 zero/few-shot으로 어느 정도 전이 가능  
  - *프롬프트 의미보다는* 데이터 분포 및 타겟 단어 선택이 전이 성능 좌우  

## 4. 향후 연구에 미치는 영향 및 고려사항  
- **의미적 프롬프트 설계의 실효성 재평가**  
- **프롬프트 이해도**를 측정할 수 있는 **새로운 진단 지표 개발** 필요  
- **프롬프트·데이터·타겟 단어** 간의 상호작용 분석을 통한 모델의 *인덕티브 바이어스* 연구  
- **실제 응용**: 사용자 지시가 진정으로 반영되는지 보장하는 *설계·검증 절차* 마련 필수  

> “모델은 프롬프트를 *이해*하기보다, 프롬프트와 훈련 데이터의 통계적 특성을 *이용*할 가능성이 높다.”  

이 논문은 프롬프트 기반 학습의 본질에 대한 근본적 의문을 제기하며, 향후 *프롬프트 설계*와 *모델 해석* 연구에 중요한 이정표를 제시한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5cbd3787-77b3-4da3-8b18-cc1ff786801c/2109.01247v2.pdf)
