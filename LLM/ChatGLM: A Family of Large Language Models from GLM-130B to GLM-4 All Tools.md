# ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools

# 핵심 주장 및 주요 기여 요약

**ChatGLM** 시리즈 논문은 GLM-130B에서부터 최신 GLM-4 All Tools까지, 중국어·영어 중심의 대형 언어 모델(Large Language Models, LLM) 개발 여정을 집대성하여 제시한다.  
1. **핵심 주장**  
   - 단계별 모델 확장을 통해 학습·정렬(alignment) 기술을 축적, GLM-4는 GPT-4 및 Gemini 1.5 Pro와 유사하거나 능가하는 성능을 달성한다.  
   - *All Tools* 버전은 사용자의 의도를 자율 판단하여 웹 브라우징, Python 해석기, 텍스트→이미지 도구 호출 등을 수행하는 지능형 에이전트 역량을 통합한다.  

2. **주요 기여**  
   - **모델 계보**: 2021년 GLM-10B부터 2024년 GLM-4 All Tools까지 네 세대에 걸친 발전 타임라인 제시.  
   - **아키텍처 혁신**: QKV 편향(bias) 최소화, RMSNorm+SwiGLU, 2D RoPE, Group Query Attention 등을 도입해 효율·성능 개선.  
   - **선형적 맥락 확장**: LongAlign 기법으로 컨텍스트 길이를 2K→32K→128K(실험적 1M)까지 확장.  
   - **정렬(Alignment) 기술**: SFT·RLHF의 결합, Self-Contrast 피드백-프리 전략, AgentTuning, APAR 병렬 생성 등 대규모 정렬·에이전팅 방법론 공개.  
   - **다양한 벤치마크**: MMLU, GSM8K, BBH, LongBench-Chat, AlignBench, AgentBench, SafetyBench 등에서 GPT-4 계열과 대등하거나 우수한 결과 입증.  

# 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

## 1. 해결 과제  
- **성능·정렬·에이전팅의 통합**: 기존 LLM은 성능·컨텍스트 길이·안전성·에이전트 기능을 개별적으로 다뤄 왔으나, 이를 하나의 프레임워크에서 종합해야 할 필요.  
- **장문 맥락 처리**: 128K 이상 장문 컨텍스트에서 안정적 언어 이해·추론 성능 확보.  
- **도구 호출 도메인**: 사용자가 지정한 외부 도구(웹, 코드 해석, 이미지 생성 등)를 문맥 추론 과정에 유기적으로 통합할 방법.

## 2. 제안 방법  
1) **아키텍처 개선**  
   - Bias 제거(단, QKV만 유지)  
   - LayerNorm→RMSNorm, GeLU→SwiGLU 교체  
   - Rotary Positional Encoding 확장: 2차원 RoPE 적용  
   - Group Query Attention: KV 캐시 축소  
2) **맥락 확장(LongAlign)**  
   - 연속 학습 및 RoPE 보간(positional interpolation) 기반 확장  
   - 장문 정렬 데이터로 후처리  
3) **정렬 파이프라인**  
   - Supervised Fine-Tuning (SFT)  
   - Reinforcement Learning from Human Feedback (RLHF, PPO/DPO)  
   - Feedback-free Self-Contrast: LLM이 스스로 부정적 샘플 생성  
4) **에이전트화(All Tools)**  
   - 계획(plan)-분석(analyze)-실행(execute) 루프 설계  
   - 도구 호출 함수 인터페이스로 웹 브라우저, Python 인터프리터, 텍스트→이미지 모델 연결  

## 3. 모델 구조  
- 디코더 전용 Transformer, hidden size 및 FFN 크기 조정  
- 150K 어휘(vocabulary)  
- 최대 128K 컨텍스트(실험적 1M)  
- Multi-stage post-training: SFT → RLHF

## 4. 성능 향상  
- **Academic**: GLM-4 (0520) MMLU 83.3%, GSM8K 93.3%, BBH 84.7%, HumanEval 78.5% (GPT-4 대비 96% 이상의 성능)  
- **Instruction Following**: IFEval loose/instruction-level 기준 GPT-4 Turbo와 동등 수준  
- **Alignment**: AlignBench 전체 8.00점으로 GPT-4 Turbo(8.00)와 동일하거나 상회  
- **Long Context**: LongBench-Chat 87.3%(영어), 84.0%(중국어)로 GPT-4 Turbo 유사  
- **Agent & Tools**: AgentBench 전체 3.79점, 브라우저 기반 정보 탐색 78.08%로 ChatGPT-4 All Tools 동등  

## 5. 한계  
- **수학적 추론**: GPT-4 대비 MATH 벤치마크에서 약간 뒤처짐  
- **물리적 상식 안전성**: SafetyBench ‘Physical Health’에서 GPT-4 계열보다 낮은 점수  
- **에이전트 코드 작업**: OS, 지식 그래프, 추리 퍼즐 환경에서 GPT-4 대비 격차 존재  

# 모델의 일반화 성능 향상 가능성

- **Self-Contrast**: 외부 피드백 없이도 대규모 음수 샘플로 정렬 품질 강화  
- **AgentTuning**: 환경별 에이전트 데이터로 범용적 순차 계획 능력 확장  
- **LongAlign**: 맥락 확장과 정렬 결합으로 다양한 입력 길이에서 일관된 성능 보장  
- **모듈화된 도구 호출**: 신규 도구(API) 추가 시 자율 선택 프레임워크로 손쉬운 확장성  

# 향후 연구 영향 및 고려사항

- **통합 에이전트 플랫폼**: 사용자 정의 함수·API·도메인 지식 연계 사례가 풍부해질 전망  
- **장문·멀티모달 정렬**: 1M 토큰급 컨텍스트와 비전·코드 연계 정렬 연구 가속  
- **수학·물리 상식 강화**: ChatGLM-Math 기법 발전 및 외부 체인-오브-생각 결합 필요  
- **안전성 강화**: Health·Ethics·Bias 차원에서 데이터·정렬 정책 고도화 필수  

---  
이 논문은 LLM의 **성능·정렬·에이전트 기능**을 하나의 통합 프레임워크로 제시함으로써, 향후 복합적인 사용자 요구를 자율적으로 처리하는 지능형 에이전트 연구 방향을 주도할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5d88ff9c-cb0e-4855-aadd-fd5b66ba940b/2406.12793v2.pdf)
