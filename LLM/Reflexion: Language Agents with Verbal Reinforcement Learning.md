# Reflexion: Language Agents with Verbal Reinforcement Learning

**핵심 요약**  
Reflexion은 전통적 강화학습처럼 파라미터를 업데이트하지 않고, **언어적 자기반성(self-reflection)** 을 통해 에이전트가 시행착오로부터 빠르게 학습하도록 설계된 프레임워크다. 에이전트는 매 시도 후 환경 피드백을 자연어 요약으로 변환해 장기 메모리에 저장하고, 이를 다음 에피소드의 맥락으로 활용함으로써 행동을 점진적으로 개선한다. 다양한 순차 의사결정, 추론, 프로그래밍 과제에서 기존 기법 대비 최대 22%p 이상의 성능 향상을 달성했다.

## 1. 해결하려는 문제  
대규모 언어모델(LLM)을 에이전트로 활용할 때 전통적 RL은 샘플 비효율성과 고비용의 미세조정(fine-tuning) 문제를 갖는다. 또한 스칼라 보상만으로는 **크레딧 할당 문제**를 정확히 해결하기 어려워, 구체적 개선 방향 제시가 어렵다.

## 2. 제안 방법  
에이전트는 세 가지 LLM 컴포넌트로 구성된다:  
- **Actor** $$M_a$$: 행동 생성  
- **Evaluator** $$M_e$$: 궤적 평가, 스칼라 보상 $$r_t = M_e(\tau_t)$$  
- **Self-Reflection** $$M_{sr}$$: 궤적 $$\tau_t$$와 보상 $$r_t$$을 입력받아 언어적 피드백 $$s r_t$$ 생성  

학습 루프는 다음과 같다:  

$$
\begin{aligned}
&\tau_0 \sim \pi_\theta,\quad r_0 = M_e(\tau_0),\quad sr_0 = M_{sr}(\tau_0, r_0),\\
&\text{메모리 }mem \leftarrow [sr_0],\\
&\text{반복: }t \leftarrow t+1, \;\tau_t \sim \pi_\theta(\cdot\mid mem),\;r_t=M_e(\tau_t),\;sr_t=M_{sr}(\tau_t,r_t),\;mem\mathrel{+=}[sr_t]
\end{aligned}
$$  

Actor는 최신 메모리 $$mem$$를 조건으로 행동을 생성하며, Self-Reflection으로부터 얻은 **언어적 보상**이 에피소드별 의사결정 방향을 제시한다.

## 3. 모델 구조  
- **기억 계층화**: 단기 메모리(현재 궤적)와 장기 메모리(최대 3회 self-reflection)로 구성해 맥락 창 제한에 대응  
- **Evaluator 변형**: 정확도 EM(Reasoning), 휴리스틱(Decision-making), LLM 분류(프로그래밍) 등 과제별 평가기 사용  
- **프로그래밍 과제**: Agent가 자체 생성한 단위 테스트로 코드 실행 후 피드백 → Self-Reflection을 통해 자연어 힌트 생성  

## 4. 성능 향상  
- **ALFWorld**: 12회 시행 후 성공률 97%에 도달, 기존 ReAct 대비 +22%p  
- **HotPotQA**: 6회 시행 후 정확도 +20%p, 사전 Ground-truth 맥락 없이도 개선  
- **HumanEval (Python)**: pass@1 91% 달성(기존 GPT-4: 80%)  
- **Rust, MBPP, LeetcodeHard** 등에서도 SOTA  
이들 과제에서 모두 **자기반성 루프**가 없는 베이스라인은 반복적 개선에 한계가 있었다.

## 5. 한계  
- **지역 최적해 함정**: 탐색 다양성이 필요한 WebShop 과제 등에서는 성능 개선이 미미  
- **테스트 생성을 통한 오류**: 부정확한 테스트가 false positive/negative를 야기할 수 있어 반성의 품질이 성능에 직접 영향  
- **메모리 용량 제한**: 고정된 메모리 크기로 장·단기 기억 관리가 단순하며, 대규모 맥락에는 제약  

## 6. 일반화 성능 향상 관점  
Reflexion의 **언어적 보강**은 과제별 특수 휴리스틱 없이 자연어 경험을 통해 다양한 환경에 적용 가능하다. 특히 LLM 기반 self-evaluation과 반성 문장을 일반화된 피드백 신호로 활용함으로써 새로운 과제에 대해 **0~few-shot** 환경에서도 빠르게 적응할 잠재력을 가진다. 다만 메모리 저장 방식과 반성 생성의 질이 일반화 성능을 좌우하므로, 더 풍부한 구조화된 메모리(벡터 DB 등)와 정교한 반성 생성 전략이 필요하다.

## 7. 향후 연구 영향 및 고려 사항  
향후 연구에서는  
- **메모리 확장**: 벡터 임베딩·데이터베이스 연동을 통한 장기 경험 축적  
- **언어 보상 정량화**: 자연어 피드백을 임베딩 기반 scalar로 전환하여 value-learning 결합  
- **안전성·투명성**: 반성 로그 모니터링을 통한 의도 검증 및 오작동 방지  
등을 고려할 때, 더욱 강건하고 해석 가능한 언어 에이전트를 개발하는 데 Reflexion 패러다임이 중추적 역할을 할 전망이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f0b44f71-90dd-4157-a89a-2a268b933d24/2303.11366v4.pdf)
