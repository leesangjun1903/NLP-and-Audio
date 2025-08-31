# Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents

## 1. 핵심 주장 및 주요 기여
이 논문은 **DEPS(Describe, Explain, Plan and Select)**라는 대화형 플래닝 프레임워크를 제안하며, 오픈 월드 환경에서 LLM 기반 플래너가 다중 과제를 안정적으로 수행하도록 한다.[1]
주요 기여:
- **인터랙티브 플래닝**: 실패 시 환경 기술(description)과 자기 설명(explanation)을 통해 초기 계획을 점진적으로 수정.  
- **목표 선택(goal selector)**: 병렬 가능한 서브 목표 간 효율성과 실행 용이성을 평가해 최적 순서 결정.  
- **제로샷 멀티태스크 성능**: Minecraft 71개 과제에서 기존 대비 성공률을 거의 2배 이상 향상시켜, 오픈 월드에서의 장기 계획 문제 해결을 입증.[1]

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 문제 정의
1) **장기-다단계 계획**: 오픈 월드의 복잡한 서브 태스크 종속성으로 인한 플래닝 오류 증가.  
2) **계획 실행 가능성**: 현재 상태(state)에 따라 실현 가능한 경로 선택이 어려워 비효율적·실패 확률 증가.[1]

### 제안 방법: DEPS
- **Describe (기술)**: 실행 실패 시 현 상태와 결과를 텍스트로 요약( $d_t = f_{DESC}(s_t−1)$ ).[1]
- **Explain (자기 설명)**: 요약과 이전 계획 정보를 바탕으로 실패 원인을 체인 오브 뎃으로 분석(e_t = f_EX(d_t)).[1]
- **Plan (재계획)**: 설명을 포함한 프롬프트로 LLM이 계획을 재생성( $P_t = f_{LM}(p_t)$ ).[1]
- **Select (선택)**: 병렬 가능한 후보 목표 G_t 중 잔여 단계 수(호라이즌)를 예측해 가장 가까운 목표를 우선(gt ∼ f_S(P_t, s_t−1)).[1]

수식:

$$
f(gt \mid s_t, P_t) = \frac{\exp(-\mu(gt, s_t))}{\sum_{g\in G_t}\exp(-\mu(g, s_t))}
$$

여기서 $$\mu$$는 목표별 남은 단계 수(horizon)를 예측하는 신경망.[1]

### 모델 구조
- LLM 기반 **Planner & Explainer** (Codex, GPT-3 등 활용)  
- **Descriptor**: 환경 관측 및 내부 상태를 구조화  
- **Selector**: Impala CNN 기반 horizon 예측기  
- **Goal-conditioned Controller**: 저수준 정책(behavior cloning)  

구성도:  
1. 초기 계획 생성 → 2. 목표 실행 → 3. 실패 시 기술 → 4. 자기 설명 → 5. 재계획 → 6. 목표 선택 → 7. 액션 수행(반복).[1]

### 성능 향상
- **Minecraft Task101**: 평균 성공률 15.4%→48.6%로 약 2배↑.[1]
- **ObtainDiamond**: 0%→0.6% 성공(제로샷, 10분 제한)  
- **ALFWorld, Tabletop Manipulation**: 타 도메인에서도 최대 76–80% 성공률 달성, 기존 대비 2배 이상 개선.[1]

### 한계
- **LLM 의존성**: GPT-3/ Codex 등 사유 모델 사용으로 접근성·비용 문제.  
- **계획 병목**: 단계별 명시적 계획이 확장성 제한.  
- **컨트롤러 성능**: 고난이도 서브 태스크(low-level)에서 제어 정책의 성공률 제약.

## 3. 일반화 성능 향상 가능성
- **제로샷 프롬프트 설계**: 체인 오브 뎃, 코드 스타일 출력 등으로 다양한 도메인 전이 용이.  
- **Selector 모듈**: 시각언어모델(VLM)과 경험 기반 horizon 예측 결합으로 도메인 특화 없이 상태 인식 강화.  
- **모듈화 아키텍처**: Descriptor, Explainer, Planner, Selector, Controller 분리 설계로 각 요소 교체·확장 가능.[1]

## 4. 향후 연구 영향 및 고려사항
- **오픈소스 LLM 활용**: OPT, BLOOM, LLaMA 등 공개 모델로 대체 연구 필요.  
- **엔드투엔드 학습**: 계획 단계를 신경망에 내재화해 확장성·실시간성 개선.  
- **컨트롤러 고도화**: 데이터 효율적 학습 또는 RL 통합으로 저수준 정책 성능 강화.  
- **Dead-end 회피**: 오픈 월드 특유의 막다른 길(dead end) 탐지·회피 메커니즘 연구.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c90abc7a-0b70-4f97-a933-9fc1b54a9513/2302.01560v3.pdf)
