# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

## 1. 핵심 주장 및 주요 기여
DeepSeek-R1 논문은 **강화학습(RL)만으로** 대형 언어모델(LLM)의 체인 오브 사고(Chain-of-Thought, CoT) 기반 추론 능력을 획기적으로 향상시킬 수 있음을 증명한다.  
주요 기여는 다음과 같다.  
- **DeepSeek-R1-Zero**: 사전 지도학습 없이 순수 RL로만 베이스 모델을 학습해, AIME 2024 벤치마크 Pass@1을 15.6%→71.0%로 대폭 상승시키고, 다수결(consensus) 평가 시 86.7%로 OpenAI-o1-0912를 능가함.[1]
- **DeepSeek-R1**: 소량의 ‘콜드 스타트’ CoT 데이터를 초기 SFT로 투입하고, 2단계 RL 및 2단계 SFT를 반복하는 멀티스테이지 파이프라인으로 모델 가독성·일관성을 확보하며, OpenAI-o1-1217 수준의 성능 달성.  
- **Distillation**: DeepSeek-R1을 교사 모델로 활용해 Qwen/Llama 시리즈 소형 모델(1.5B~70B)을 SFT만으로 증류하고, 32B 모델이 AIME 72.6%·MATH-500 94.3% 기록, o1-mini급 성능 달성.

## 2. 해결 문제와 제안 방법
### 2.1 해결하고자 하는 문제
- 대형 LLM이 고도의 추론 과제를 풀기 위해선 대규모 지도학습(CoT 예제)이나 추론 시점 스케일링이 필요했으나,  
- RL만으로 추론 역량을 체계적으로 강화하는 방법은 미비하며, RL 초기에 발생하는 불안정성과 결과 가독성 저하 문제 존재.

### 2.2 제안 방법
1) **DeepSeek-R1-Zero: 순수 RL**  
   - 베이스 모델(DeepSeek-V3-Base)에 바로 GRPO(Group Relative Policy Optimization)를 적용.[1]
   - 보상 함수:  
     - 정확도 보상(정답 여부)  
     - 형식 보상(‘…’ 태그 준수)  
   - 학습 목표 함수:  


$$
     J_{GRPO}(θ)=\mathbb{E}\_{q,\,\{o_i\}} \Big[\sum_{i=1}^G 
       \min\Big(\frac{\pi_θ(o_i|q)}{\pi_{old}(o_i|q)}A_i,\,
       \text{clip}\big(\tfrac{\pi_θ}{\pi_{old}},1-\varepsilon,1+\varepsilon\big)A_i\Big)
       -β\,D_{KL}(\pi_θ\|\pi_{ref})\Big]
     $$  
     
  where $$A_i=\frac{r_i-\mu(r)}{\sigma(r)}$$.

2) **DeepSeek-R1: 콜드 스타트 + 멀티스테이지 학습**  
   - **콜드 스타트 SFT**:  수천개의 장문 CoT(+요약) 데이터로 초기 SFT 수행.  
   - **1차 RL**: DeepSeek-R1-Zero와 유사한 RL, 언어 일관성 보상 추가.  
   - **1차 SFT**: RL 체크포인트로부터 리젝션 샘플링한 60만 Reasoning 데이터 + 20만 Non-reasoning 데이터로 재학습.  
   - **2차 RL**: 추론 과제뿐 아니라 다중 도메인(글쓰기·역할극·안전성) 프롬프트에 대한 보상 모델 결합.

### 2.3 모델 구조
- 기본 정책 네트워크는 **DeepSeek-V3-Base** (MoE, 37B 활성화 파라미터)  
- Distillation 대상: Qwen2.5-Math 시리즈(1.5B,7B,14B,32B) 및 Llama-3 시리즈(8B,70B)

## 3. 성능 향상 및 한계
### 3.1 성능 향상
- **추론 벤치마크**  
  - AIME 2024 Pass@1: 16.0%→79.8%  
  - MATH-500 Pass@1: 78.3%→97.3%  
  - Codeforces Percentile: 20.3→96.3  
- **일반 언어 이해**  
  - MMLU: 88.5%→90.8%  
  - AlpacaEval2.0 LC-winrate: 70.0%→87.6%  
- **Distilled 모델**  
  - Qwen-32B 증류: AIME 72.6%, MATH-500 83.3%로 o1-mini 수준 달성.

### 3.2 한계
- **범용성 저하**: DeepSeek-V3보다 함수 호출, 다중 턴, JSON 출력 등 일반 기능에서 미흡.  
- **언어 혼용**: 중국어·영어 외 언어 쿼리에 CoT 혼용 발생.  
- **RL 비용**: SW 엔지니어링 데이터에 대한 대규모 RL은 평가 속도 저하로 미적용.

## 4. 일반화 성능 향상 가능성
- **콜드 스타트+RL 결합**으로 소량의 고품질 CoT 데이터가 학습 초기 분산을 줄여, RL이 발견한 패턴을 더 넓은 도메인에 전이 가능.  
- **멀티스테이지 파이프라인**(SFT↔RL 반복)이 비추론 태스크(글쓰기·QA·안전성)까지 보상 신호를 확장해, **다양한 프롬프트 분포에 대한 적응력**을 강화.  
- **Distillation**을 통해 교사 모델이 학습한 추론 패턴이 소형 모델로 전이됨에 따라, 저연산 환경에서도 추론 일반화 역량을 확보할 수 있음.

## 5. 향후 연구 영향 및 고려 사항
- **영역 확장**: 함수 호출·멀티모달·대화 다중턴 등 CoT를 넘어선 응용 탐색.  
- **언어 확장성**: 다국어 CoT 패턴 설계 및 언어 일관성 보상 조정.  
- **효율적 RL**: 비동기 평가·프롬프트 형식 자동화로 SW 엔지니어링 태스크용 RL 가속.  
- **안정적 보상 모델**: PRM·MCTS 대안으로 스케일 가능한 보상 구조 연구.  
- **강건한 일반화**: 다양한 사용자 입력 분포에 대한 내성 평가 및 도메인 적응 기술 개발.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c44c9c65-939c-41ff-8aeb-37731d2d445f/2501.12948v1.pdf
