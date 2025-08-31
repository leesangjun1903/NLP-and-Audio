# LLM+P: Empowering Large Language Models with Optimal Planning Proficiency

# 핵심 주장 및 주요 기여 요약

**LLM+P** 프레임워크는 **자연어**로 기술된 로봇 계획 문제를 PDDL(planning domain definition language) 형식으로 변환하고, **클래식 플래너**를 활용해 최적(또는 유효) 계획을 생성한 뒤 이를 다시 자연어로 번역함으로써, 대형 언어 모델(LLM)의 *계획 능력*을 획기적으로 향상시킨다.  
주요 기여는 다음과 같다.  
1. LLM을 **PDDL 작성기**로 활용하여 복잡한 계획 문제를 정확한 포맷으로 변환.  
2. 고전적 플래너의 **완전성(soundness, completeness)** 및 **최적성(optimality)** 을 결합해 장기(멀리뻗은) 계획 문제도 신뢰성 있게 해결.  
3. 7개 로봇 도메인(BLOCKSWORLD, BARMAN 등)과 총 140개 과제에서 평가하여, 기존 LLM 단독 수행 방법 대비 성공률을 대폭 향상(예: BLOCKSWORLD 20→90%, GRIPPERS 35→95%)시킴.  
4. 실제 가정용 로봇 시연을 통해 “커피 테이블→팬트리로 병 이동”과 “사이드 테이블의 빈 수프 통 버리기”를 최적 경로로 계획·실행함을 검증.  

# 문제 정의, 제안 방법 및 모델 구조

## 1. 해결 과제  
대형 언어 모델은 **장기 추론(long-horizon reasoning)** 과 **기능적 역량(functional competence)** 에서 한계가 있어, 로봇 물리 계획·일반화 문제에서 유효·최적 계획을 내지 못함.

## 2. LLM+P 개요  
LLM+P는 세 단계로 구성된다(그림 1).  
1) **문제 번역**: 자연어 문제 기술(P) → PDDL 문제 파일  
2) **플래너 호출**: 도메인 PDDL + 생성된 문제 PDDL → 플래너(FAST-DOWNWARD)로부터 최적 계획 πₒₚₜ  
3) **계획 번역**: πₒₚₜ(PDDL) → 자연어 계획 설명  

## 3. 수식적 표현  
- planning problem $$P = \langle S, s_{\mathrm{init}}, S_G, A, f \rangle$$  
- 해 $$\pi = \langle a_1, a_2, \dots, a_N\rangle$$ 는  

$$
    \text{pre}(a_1)\subseteq s_{\mathrm{init}},\quad \text{pre}(a_{i+1})\subseteq f(s_{i},a_i),\quad
    S_G\subseteq s_N
  $$
  
  를 만족.  
- **Metric minimize**: $$\mathrm{cost}(\pi)=\sum_i \mathrm{cost}(a_i)$$ 을 최소화  

## 4. 모델 구조 및 학습  
- **LLM**(GPT-4)에는 **예시 문제⇔PDDL** 페어와 **도메인 PDDL**을 *컨텍스트(prompt)*로 제공.  
- temperature = 0 설정으로 결정론적 출력 보장.  
- PDDL 형식 오류($$ \mathrm{parse\ error}$$) 발생 시 사용자 개입 없이 재생성(in-context learning)만으로 수정.  
- 플래너는 **SEQ-OPT-FDSS-1**을 기본으로, 타임아웃(200 s) 시 대체 래칭(alias) 수행.

# 성능 향상 및 한계

## 성능 향상  
| 도메인          | LLM-AS-P 성공률 | LLM+P 성공률         |
|---------------|---------------|--------------------|
| BLOCKSWORLD   | 20%           | **90%**            |
| GRIPPERS      | 35%           | **95%**            |
| BARMAN        | 0%            | **20%**            |
| STORAGE       | 0%            | **85%**            |
| TYREWORLD     | 15%           | **10%(90% 비최적)** |
| FLOORTILE, TERMES | 0%        | 0–20%              |

- **맥락(context)** 없이 PDDL 변환 시 실패율 급증: 맥락 제공이 *핵심 요소*.  
- LLM+P는 대부분 도메인에서 최적 계획 산출, LLM-AS-P는 사전조건 무시·비효율·타임아웃 발생.

## 한계  
1. **도메인 PDDL 수동 작성**: 각 도메인마다 전문가가 정의해야 함.  
2. **자동 인지 부재**: 언제 LLM+P 파이프라인을 적용할지 LLM이 스스로 인지하지 못함.  
3. **FLOORTILE 등 일부 도메인 오타**: PDDL 초기 조건 일부 누락 시 불해결 사례 존재.  
4. **계산 자원 의존**: 플래너 호출 시 타임아웃 및 비용 문제.

# 일반화 성능 향상 가능성

- **In-Context Learning**을 통해 LLM이 다양한 구조의 PDDL 문제에 적응.  
- **도메인 불변성(domain invariance)**: 동일 도메인 PDDL을 기반으로 새로운 문제 유형·규모에도 유연.  
- **계층적·재귀적 맥락 확장** 연구 시, 더 복잡·미지 도메인에도 일반화 기대.  
- **자동 도메인 구성**(domain induction)과 결합하면, 드문 도메인으로 확장 가능성.

# 향후 연구 영향 및 고려 사항

**영향**:  
- **자연어↔형식 언어 인터페이스** 발전: 로봇·플래너·LLM 융합·확장.  
- **다양한 도메인 자동화**: 수치 계산, 지식검색 툴 등 타 외부 모듈 결합 사례 확산.  

**고려 점**:  
- LLM+P **자동 트리거** 메커니즘 연구.  
- **도메인 PDDL 자동 생성/검증** 기법 통합.  
- 계산 비용·타임아웃 최적화 및 **점진적 계획 재사용** 전략.  
- 도메인 간 전이학습(transferrable prompts) 및 **메타-계획 학습** 고도화.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ace4595e-d3c2-4bb5-9e0e-4453ac2a7085/2304.11477v3.pdf)
