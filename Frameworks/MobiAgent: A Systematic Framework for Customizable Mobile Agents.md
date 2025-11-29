
# MobiAgent: A Systematic Framework for Customizable Mobile Agents

## 1. 핵심 주장 및 주요 기여

MobiAgent는 **모바일 GUI 기반 자동화 작업의 정확성과 효율성을 동시에 해결하기 위한 풀스택(full-stack) 솔루션**을 제시합니다.[1]

논문의 핵심 주장은 다음 세 가지입니다:

**첫째, 역할 분리 아키텍처의 효과성**: 기존 단일 모델 에이전트의 한계를 극복하기 위해 Planner, Decider, Grounder라는 세 개의 독립적인 역할로 구성된 **MobiMind-series 에이전트 모델**을 제안합니다. 이를 통해 작업 계획, 고수준 추론, 저수준 실행을 명확히 분리함으로써 GUI/XML 기반 다양한 백엔드와의 유연한 통합을 가능하게 합니다.[1]

**둘째, 경험 재활용의 가치**: **AgentRR(Record-Replay-based Agent Acceleration)** 프레임워크를 통해 과거 실행 흔적을 다층 경험으로 추상화하고, 가벼운 잠재 메모리 모델을 통해 재사용 가능한 경험을 판단합니다. 이는 실제 모바일 시나리오에서 2~3배의 성능 개선을 달성합니다.[1]

**셋째, 현실 기반 평가의 필요성**: 기존 벤치마크의 한계(환경 변동성, 다중 정확 경로의 부재, 결정적 검증 메커니즘의 부족)를 해결하기 위해 **MobiFlow** 벤치마크를 개발했습니다. DAG(Directed Acyclic Graph) 기반 다중 경로 정의와 다층 검증 메커니즘을 통해 현실성 있는 평가를 가능하게 합니다.[1]

***

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제

현재 모바일 에이전트가 직면한 주요 문제는:[1]

- **낮은 작업 완료율**: 기존 에이전트들이 실제 모바일 환경에서 불안정한 성능을 보임
- **느린 응답 시간**: 매 단계마다 VLM 추론이 필요하여 효율성 저하
- **예외 상황 대응 능력 부족**: 팝업, 네트워크 오류 등의 예상 외 상황에 취약
- **고품질 학습 데이터 부재**: 주석 비용으로 인한 대규모 데이터 수집 어려움
- **비현실적 평가 벤치마크**: 기존 벤치마크가 현실의 복잡성을 충분히 반영하지 못함

### 2.2 데이터 수집 및 전처리 파이프라인

**고효율 데이터 수집 방식**:[1]

1. **경량 기록 도구**를 통해 실제 사용자 상호작용 캡처
2. **OmniParser**로 불완전한 XML 데이터의 바운딩 박스 재생성
3. **VLM 기반 추론 재구성**: Gemini-2.5를 활용하여 저수준 궤적(trajectory)에서 고수준 의미 추출

기록된 행동 공간:

$$\text{Action Space} = \{\text{Click}(\text{bbox}), \text{Input}(\text{text}), \text{Swipe}(\text{direction}), \text{Wait}(\text{sec}), \text{Done}()\}$$

**데이터 정제 전략**:[1]

- **작업 연결(Task Concatenation)**: 시간적으로 의존하는 궤적 연결로 복합 작업 데이터 생성
- **데이터 재분배(Data Redistribution)**: 접두사(prefix) 작업 샘플링으로 불균형 완화
- **이력 증강(History Augmentation)**: 부분 이력 포함으로 과적합 방지
- **프롬프트 일반화(Prompt Generalization)**: 의미적으로 동등한 다양한 작업 기술 할당
- **코너 케이스 강화(Corner-case Enhancement)**: 팝업 처리 등 특수 상황에 대한 단일 스텝 데이터셋 별도 구성

### 2.3 모델 구조 및 훈련 방법

#### **다중 역할 아키텍처**

$$\text{MobiAgent} = \{\text{Planner}(4B), \text{Decider}(7B), \text{Grounder}(3B)\}$$

- **Planner**: 작업 분해 및 다중 앱 매핑
- **Decider**: 현재 상황에서 다음 고수준 행동 결정 (Qwen2.5-VL-7B 기반)
- **Grounder**: Decider의 고수준 의도를 저수준 바운딩 박스로 변환 (Qwen2.5-VL-3B 기반)

#### **훈련 단계**

**1단계: Warm-up SFT (지도 학습 미세조정)**

Decider와 Grounder를 기본 포맷 준수 및 GUI 에이전트 기능으로 초기화:

$$\mathcal{L}_{\text{SFT}} = -\log P_\theta(\text{action} \mid \text{image}, \text{history}, \text{task})$$

**2단계: 두 단계 커리큘럼 GRPO (강화학습)**

**(1) 그라운딩 GRPO** - 규칙 기반 보상 함수:

$$R_{\text{IoU}} = \begin{cases} \alpha & \text{if } \text{IoU}(b_{\text{pred}}, b_{\text{gt}}) > \beta \\ 0 & \text{otherwise} \end{cases}$$

$$R_{\text{center}} = \begin{cases} 1-\alpha & \text{if Center}(b_{\text{pred}}) \in b_{\text{gt}} \\ 0 & \text{otherwise} \end{cases}$$

$$R = R_{\text{IoU}} + R_{\text{center}}$$

여기서 $b_{\text{pred}}$는 예측 바운딩 박스, $b_{\text{gt}}$는 그라운드 트루스 박스입니다.[1]

**(2) Grounder-as-RM GRPO** - 고정밀 그라운더를 보상 모델로 활용:

클릭이 아닌 행동에 대해:

$$R_{\text{content}} = \begin{cases} 1 & \text{if } a_{\text{pred}} = a_{\text{gt}} \\ 0 & \text{otherwise} \end{cases}$$

클릭 행동에 대해: $R = R_{\text{center}}$ (그라운더가 제공)[1]

**커리큘럼 학습**: 난이도 기반 샘플 레이블링으로 점진적 학습 난이도 상향

**자기 진화(Self-evolution)**: 훈련-테스트 분포 일치 위해 각 훈련 라운드 후 테스트 추적 수집 및 정정된 궤적을 다음 라운드 훈련 데이터에 포함

***

## 3. AgentRR: 다층 경험 기반 가속화 프레임워크

### 3.1 다층 경험 계층

$$\text{Experience Hierarchy} = \begin{cases} \text{High-level: Task Plan (Planner)} \\ \text{Mid-level: Action Primitive (Decider)} \\ \text{Low-level: Operation Action (Grounder)} \end{cases}$$

고수준 경험은 일반화 능력이 강하지만 추론 비용이 크고, 저수준 경험은 효율성은 높지만 특정 상황에 국한됩니다.[1]

### 3.2 ActTree 구조

트리 기반 상태 캐싱 메커니즘:

$$\text{ActTree}: V = \text{UI states}, E = \text{transitions}$$

각 엣지 $e = (u, v)$는 다음 정보 포함:

- $A_e$: 상태 전이를 트리거하는 행동
- $T_e = \{t_1, ..., t_k\}$: 이 전이를 실행한 과거 작업들

**엣지 병합**: 동일한 행동을 가진 엣지는 작업 목록을 합쳐 트리 컴팩트성 유지[1]

### 3.3 잠재 메모리 모델

#### **작업 임베딩 기반 재사용 판정**

$$v_i^{(l)} = f(T_i, l) \in \mathbb{R}^d$$

여기서 $f$는 명령 인식 작업 임베딩 모델, $l$은 ActTree의 계층입니다.

코사인 유사도 기반 판정:

$$\text{Embed-Reuse}^{(l)}(T_i, T_j) = \begin{cases} \text{True} & \text{if } S(v_i^{(l)}, v_j^{(l)}) \geq \tau_1 \\ \text{False} & \text{otherwise} \end{cases}$$

#### **작업 재순위 기반 정밀 판정**

후보 작업 선정:

$$C = \{T \in T_{\text{hist}} \mid \text{Embed-Reuse}^{(l)}(T, T_c)\}$$

최종 재사용 판정 (2단계):

$$\text{Rerank-Reuse}^{(l)} = \begin{cases} \text{True} & \text{if } \exists T_i \in C \text{ such that } g(T_i, T_c, l) \geq \tau_2 \\ \text{False} & \text{otherwise} \end{cases}$$

여기서 $g$는 명령 인식 작업 재순위 모델입니다.[1]

#### **잠재 메모리 모델 훈련**

Qwen-Embedding과 Qwen-Reranker 기반. 임베딩 모델은 InfoNCE 손실 사용:

$$L = \mathbb{E}_v\left[-\ln\frac{e^{S(v,v^+)/\tau}}{e^{S(v,v^+)/\tau} + \sum_{j=1}^M e^{S(v,v_j^-)/\tau}}\right]$$

양성 샘플: $v^+ = f(T_s, l), \text{pre}(T_s, T_i) \geq l$

음성 샘플: $v^- = \{f(T_t, l) \mid \text{pre}(T_t, T_i) < l\}$

여기서 $\text{pre}(·,·)$는 두 작업 사이의 공유 행동 접두사 길이입니다.[1]

***

## 4. MobiFlow: DAG 기반 벤치마킹 프레임워크

### 4.1 구조적 특징

**DAG 기반 작업 정의**: 마일스톤 노드 $M$과 의존성 엣지로 구성

**다중 검증 메커니즘** (진행 순서 기반):

1. **텍스트 매칭**: XML 파일에서 특정 문자열 확인
2. **정규표현식 매칭**: 복잡한 텍스트 패턴 검증
3. **UI 상태 분석**: 뷰 계층 구조(XML) 검토
4. **아이콘 감지**: 컴퓨터 비전으로 특정 아이콘 식별
5. **OCR**: XML 부재 시 스크린샷에서 텍스트 추출
6. **MLLM 판정자**: 멀티모달 LLM의 홀리스틱 판단[1]

### 4.2 다중 경로 처리

AND/OR 로직 지원:

- **AND**: 모든 선행 마일스톤 이벤트 필수
- **OR**: 하나의 마일스톤 이벤트만으로 충분

동기화된 역추적 메커니즘으로 DAG와 실행 추적 노드 간 정확한 매칭[1]

***

## 5. 성능 향상 분석

### 5.1 작업 완료율

MobiAgent는 다음 모델들을 상회:[1]

| 모델 | 전체 평균 | 쉬운 작업 | 어려운 작업 |
|------|---------|---------|-----------|
| GPT-5 | 약 70% | 약 82% | 약 58% |
| Gemini-2.5 Pro | 약 72% | 약 84% | 약 60% |
| UI-TARS-1.5-7B | 약 68% | 약 80% | 약 56% |
| **MobiMind-7B+3B** | **약 78%** | **약 90%** | **약 66%** |

특히 **복합 작업(쇼핑, 음식 배달)에서 우수한 작업 분해 및 예외 처리 능력** 시연[1]

### 5.2 AgentRR 가속화 효과

**균등 분포** 환경: 30~60% 행동 재사용률

**멱 법칙 분포** 환경 (현실성 높음): 60~85% 행동 재사용률[1]

실제 배포 시 **2~3배 성능 개선** 달성 (음식 배달, 온라인 쇼핑 등 복합 작업)[1]

**재사용 정확도**: 99% 이상[1]

### 5.3 작업 종료 신뢰성

- GPT: 11개 앱 카테고리에서 작업 미종료 문제
- Gemini: 3개 앱 카테고리에서 작업 미종료 문제
- **MobiAgent**: 모든 평가 시나리오에서 정확한 종료 달성[1]

***

## 6. 일반화 성능 향상 메커니즘

### 6.1 데이터 레벨 일반화

**프롬프트 일반화**: 동일 궤적에 의미적으로 동등한 다양한 작업 기술 할당으로 모델의 표현 다양성 증대[1]

**이력 증강**: 완전한 이력뿐만 아니라 부분 이력도 훈련 데이터에 포함하여 이력 길이 편향 제거[1]

### 6.2 훈련 레벨 일반화

**자기 진화 메커니즘**: 

$$T^{(n+1)} = T^{(n)} \cup \{\text{Correct}(\mathcal{E}_{\text{test}}^{(n)})\}$$

여기서 $T^{(n)}$은 $n$번째 라운드 훈련 데이터, $\mathcal{E}_{\text{test}}^{(n)}$은 테스트 추적입니다.[1]

**훈련-테스트 분포 일치**: 에러 축적으로 인한 분포 이동 문제를 테스트 시간 데이터를 훈련에 포함시켜 해결[1]

### 6.3 모델 아키텍처 레벨 일반화

**역할 분리**: 각 컴포넌트가 특정 부분에 특화되어 상호 보완

**다층 경험 계층**: 고수준 경험(일반화 높음)부터 저수준 경험(효율성 높음)까지 유연한 선택[1]

***

## 7. 주요 한계

### 7.1 기술적 한계

1. **MobiFlow 절대 점수의 불안정성**: 앱 업데이트나 환경 변동으로 인해 절대 점수는 참고용일 수 있으며, 상대 비교만 신뢰성 있음[1]

2. **오프라인 평가의 거짓 음성(False Negative)**: 에이전트가 새로운 성공 경로를 발견해도 오프라인 추적 기반 평가에서는 실패로 판정될 수 있음[1]

3. **Latent Memory Model의 정확도 의존성**: 재사용 판정이 임베딩 및 재순위 모델의 정확도에 크게 의존하여, 이들 모델의 오류 전파 가능성[1]

4. **한정된 데이터셋**: 현재 중국 내 주요 모바일 앱을 중심으로 구성되어 글로벌 앱 생태계 커버리지 제한[1]

### 7.2 설계상 한계

1. **역할 분리의 오버헤드**: 3개 모델의 연쇄 추론으로 인한 지연 시간 증가 (AgentRR로 완화되나 완전히 제거되지 않음)[1]

2. **재사용 결정의 보수성**: 정확도 우선 정책으로 인해 재사용 가능한 경험을 놓칠 수 있음[1]

***

## 8. 앞으로의 연구에 미치는 영향 및 고려사항

### 8.1 연구 커뮤니티에 대한 기여

**1. 멀티역할 에이전트 아키텍처의 정당성**

최근 연구들(Mobile-Agent-E, MobileSteward, AppAgentX 등)이 **계층적/모듈형 아키텍처로의 이동**을 가속화했습니다. MobiAgent의 성공은 단일 거대 모델보다 **특화된 역할 분리의 효율성**을 입증하여, 향후 에이전트 설계의 표준 패러다임으로 부상할 가능성을 높였습니다.[2][3][4][1]

**2. 강화학습 기반 GUI 에이전트 훈련의 표준화**

MobiAgent의 **GRPO 기반 두 단계 커리큘럼 훈련** 방식은 DeepSeek-R1의 성공 이후 추론 능력 개선 기법으로 주목받고 있습니다. 최근 연구들(AGPO, CPPO, Hybrid GRPO 등)이 GRPO의 개선 방안들을 제시하고 있으며, MobiAgent의 "Grounder-as-RM" 개념은 **도메인 특화 보상 설계의 혁신**으로 평가받고 있습니다.[5][6][7][8][9][1]

**3. 현실 기반 벤치마킹의 중요성 재조명**

MobiFlow의 DAG 기반 다중 경로 및 다층 검증 메커니즘은 기존 정적 벤치마크의 한계를 명확히 드러냈습니다. 최근 출시된 벤치마크들(SPA-Bench, A3, AndroidWorld 등)이 MobiFlow의 개념을 수용하여 동적 환경 평가와 다중 경로 정의를 강화하고 있습니다.[10][11][2][1]

### 8.2 향후 연구 시 고려 사항

**1. 일반화 성능 향상 방향**

현재 Qwen2.5-VL 기반 MobiAgent의 약 78% 완료율은 여전히 개선 여지가 있습니다. 다음 방향이 고려되어야 합니다:[12][4][1]

$$\text{Generalization Improvement} = f(\text{Cross-domain Transfer} + \text{Few-shot Learning} + \text{Domain Adaptation})$$

최근 연구인 Mobile-Agent-E의 **자기 진화 메커니즘**(self-evolution through experience replay)과 MobileSteward의 **객체 지향 멀티에이전트 설계**를 결합하면, 보다 적응적이고 일반화된 에이전트 개발이 가능할 것으로 예상됩니다.[3][4]

**2. 데이터 효율성 개선**

MobiAgent의 AI 보조 데이터 수집은 비용을 줄였으나, 여전히 고품질 라벨 작업이 필요합니다. 향후 연구 방향:[12][1]

- **약지도 학습(Weak Supervision)**: MLLM의 자동 라벨링으로 주석 비용 추가 절감
- **도메인 적응 기법**: 소수 샘플만으로 새로운 앱/환경에 빠르게 적응
- **Active Learning**: 모델의 확신도가 낮은 샘플 우선 수집

**3. 크로스 앱 작업의 복잡성**

MobileSteward 연구에서 드러난 **크로스 앱 협력의 어려움**(복잡한 작업 관계, 다양한 앱 환경, 오류 전파)에 대응하기 위해:[4]

$$\text{Error Propagation} = \sum_{i=1}^{N} P(\text{Error}_i) \cdot \text{Cascade}(\text{Error}_i)$$

- **작업 간 의존성 그래프** 명시적 모델링
- **부분 실패 복구(Partial Recovery)** 메커니즘 개발
- **크로스 앱 상태 추적** 개선

**4. 계산 효율성과 확장성**

AgentRR의 2~3배 가속화에도 불구하고, 실시간 모바일 에이전트 서비스를 위해:[1]

- **모델 경량화**: Grounder 3B 수준으로 Decider도 최적화
- **엣지 디바이스 배포**: Latent Memory Model의 온디바이스 실행
- **조건부 추론**: 작업 복잡도에 따른 동적 모델 선택

**5. 강화학습의 안정성 개선**

최근 "RLV(Reinforcement Learning with Verification)" 연구가 GRPO의 한계를 지적했습니다: GRPO는 정답 여부만 판정하지만, 모델의 **자체 검증 능력 상실**이 발생할 수 있다는 점입니다.[13]

MobiAgent의 Grounder-as-RM을 RLV 개념과 결합하면:

$$\text{RLV-MobiAgent}: \text{Decider} \rightarrow \begin{cases} \text{Generate Action} \\ \text{Verify Action Quality (Grounder)} \end{cases}$$

이를 통해 테스트 시간 추론 스케일링(Test-time Scaling)이 개선될 수 있습니다.[13]

**6. 멀티 모달 이해도 향상**

GUI-Owl 등 최근 GUI 특화 모델의 발전을 고려할 때:[14]

- **화면 내 텍스트-아이콘-공간 정보의 통합 이해** 향상
- **동적 UI 변화에 대한 실시간 적응**
- **앱 특정 스타일/레이아웃에 대한 신속한 적응**

**7. 벤치마크의 진화**

MobiFlow의 성공에 따라 향후 벤치마크 발전 방향:[1]

- **시간 제약 기반 평가**: 반응 시간 고려한 점수 계산
- **비용 메트릭 통합**: 토큰 소비, 모델 호출 횟수 등
- **다양한 지역/언어 커버리지**: 글로벌 앱 생태계 포괄
- **A/B 테스팅 지원**: 모델 간 통계적 유의성 검증

***

## 9. 결론

MobiAgent는 모바일 GUI 에이전트 분야의 **체계적이고 실용적인 솔루션**입니다. 특히 **역할 분리 아키텍처, GRPO 기반 훈련, 경험 재활용 가속화, 현실 기반 벤치마킹**의 네 가지 요소가 시너지를 이루어 기존 모델들을 상회하는 성능을 달성했습니다.[1]

향후 연구는 **일반화 성능 향상, 크로스 도메인 적응, 강화학습의 안정성 개선, 엣지 디바이스 배포**에 집중해야 할 것으로 예상됩니다. 특히 최근의 멀티에이전트 협력 연구들과 RLV 기법의 발전을 통합한다면, 더욱 강력하고 적응적인 모바일 자동화 시스템으로의 진화가 가능할 것입니다.[15][13]

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3db9ce1e-50d1-42d7-b8fd-d9051b072efa/2509.00531v1.pdf)
[2](https://arxiv.org/html/2410.15164)
[3](https://arxiv.org/html/2501.11733v2)
[4](http://arxiv.org/pdf/2502.16796.pdf)
[5](https://arxiv.org/abs/2503.11444)
[6](https://arxiv.org/pdf/2503.06639.pdf)
[7](https://arxiv.org/pdf/2502.01652.pdf)
[8](https://arxiv.org/pdf/2503.22342.pdf)
[9](https://arxiv.org/html/2503.15952v1)
[10](https://arxiv.org/html/2501.01149v1)
[11](http://arxiv.org/pdf/2404.16660.pdf)
[12](http://arxiv.org/pdf/2406.01014.pdf)
[13](https://contents.premium.naver.com/banya/banyacompany/contents/250618104042374kd)
[14](https://marcus-story.tistory.com/269)
[15](https://www.themoonlight.io/ko/review/enhancing-language-multi-agent-learning-with-multi-agent-credit-re-assignment-for-interactive-environment-generalization)
[16](https://arxiv.org/html/2503.02268v1)
[17](https://www.samsungsds.com/kr/insights/2025-dx-prospects.html)
[18](https://www.themoonlight.io/ko/review/a-summary-on-gui-agents-with-foundation-models-enhanced-by-reinforcement-learning)
[19](https://velog.io/@jack_7711/AI-%EC%B0%A8%EC%9B%90-%EC%B6%95%EC%86%8C-%EA%B8%B0%EB%B2%95-%EC%A0%95%EB%A6%AC)
[20](https://blog.naver.com/beyond-zero/223914306480)
[21](https://www.youtube.com/watch?v=k-8CFgBhhu4)
[22](https://www.flowhunt.io/ko/%EC%9A%A9%EC%96%B4%EC%A7%91/dimensionality-reduction/)
[23](https://view6494.tistory.com/entry/2025%EB%85%84-%EC%B5%9C%EC%8B%A0-%EB%AA%A8%EB%B0%94%EC%9D%BC-%EC%95%B1-%EA%B0%9C%EB%B0%9C-%ED%8A%B8%EB%A0%8C%EB%93%9C)
[24](https://www.ibm.com/kr-ko/think/topics/ai-agent-learning)
[25](https://laonpeople.com/blog/%ED%8A%B8%EB%A0%8C%EB%93%9C%EB%A6%AC%ED%8F%AC%ED%8A%B8-2025%EB%85%84-ai-%EC%96%B4%EB%94%94%EA%B9%8C%EC%A7%80-%EC%99%94%EB%82%98/)
[26](https://aisparkup.com/posts/2148)
[27](https://chanmuzi.tistory.com/464)
[28](http://arxiv.org/pdf/1705.06366.pdf)
[29](https://arxiv.org/html/2504.02546)
[30](https://arxiv.org/pdf/2406.18062.pdf)
[31](https://arxiv.org/pdf/2502.18548.pdf)
[32](https://digitalbourgeois.tistory.com/808)
[33](https://energent.ai/use-cases/ko/data-pipeline)
[34](https://dnjswngo.tistory.com/60)
[35](https://blog.naver.com/n_cloudplatform/223938853841?fromRss=true&trackingCode=rss)
[36](https://tech.ktcloud.com/entry/2025-03-ktcloud-ai-agent-%EC%97%90%EC%9D%B4%EC%A0%84%ED%8A%B8-%ED%99%9C%EC%9A%A9%ED%8C%A8%ED%84%B4)
[37](https://discuss.pytorch.kr/t/deep-research-llm/6112)
[38](https://discuss.pytorch.kr/t/anthropic/7100)
[39](https://enterprise.kt.com/bt/dxstory/3467.do)
[40](https://rudaks.tistory.com/entry/LangGraph-%EB%B2%88%EC%97%AD-%EB%A9%80%ED%8B%B0-%EC%97%90%EC%9D%B4%EC%A0%84%ED%8A%B8-%EC%8B%9C%EC%8A%A4%ED%85%9C-Multi-agent-systems)
[41](https://myownproject.tistory.com/70)
