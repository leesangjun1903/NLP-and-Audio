# Mobile-Agent-v3: Fundamental Agents for GUI Automation

### 1. 핵심 주장 및 주요 기여 요약

**Mobile-Agent-v3**는 Alibaba Tongyi Lab에서 2025년 9월 발표한 논문으로, GUI(그래픽 사용자 인터페이스) 자동화 분야에서 혁신적인 접근법을 제시합니다. 이 연구의 핵심 주장은 다음과 같습니다:[1]

**핵심 주장:**
- 오픈소스 기반 GUI 에이전트의 성능을 획기적으로 향상시킬 수 있으며, 이는 폐쇄형 상용 모델(예: GPT-4o, Claude)과 경쟁할 수 있다는 것
- 단일 end-to-end 모델과 다중 에이전트 프레임워크의 조합을 통해 복잡한 GUI 자동화 작업을 효과적으로 처리할 수 있다는 것
- 자동 진화하는 데이터 생성 파이프라인을 통해 수동 주석의 필요성을 최소화할 수 있다는 것

**주요 기여:**

GUI-Owl 모델은 세 가지 주요 혁신을 포함합니다:[1]

1. **대규모 환경 인프라**: 클라우드 기반의 가상 환경(Android, Ubuntu, macOS, Windows)을 구축하여 자동 진화하는 GUI 궤적 생성 파이프라임(Self-Evolving GUI Trajectory Production framework)을 지원
2. **다양한 기본 에이전트 기능 구성**: UI 그라운딩, 작업 계획, 행동 의미론 등 기본 UI 데이터를 통합하여 end-to-end 의사결정과 다중 에이전트 프레임워크로의 모듈화 가능
3. **확장 가능한 환경 강화학습**: 완전 비동기 학습을 지원하고 실제 사용과의 정렬을 개선하는 새로운 **Trajectory-aware Relative Policy Optimization(TRPO)** 알고리즘 제안

***

### 2. 해결하고자 하는 문제 및 제안하는 방법

#### 2.1 문제 정의

기존 GUI 에이전트 개발의 주요 문제점은 다음과 같습니다:[1]

1. **데이터 병목**: GUI 자동화 작업의 특성상 고품질 궤적 데이터를 수집하기 어렵고, 대부분의 데이터는 수동 주석에 의존
2. **크로스 플랫폼 일반화 부족**: 모바일, 데스크톱, 웹 등 다양한 플랫폼에서 일관된 성능 달성이 어려움
3. **장기 다단계 작업 처리의 어려움**: 장기적 시간 지평의 GUI 작업에서 신용 할당 문제(credit assignment problem)가 심각
4. **폐쇄형 API 의존성**: 기존 에이전트 프레임워크가 GPT-4V 등 폐쇄형 모델에 의존하여 적응성과 개인화 제한

#### 2.2 제안하는 방법

**2.2.1 Self-Evolving GUI Trajectory Production Framework**

이 프레임워크는 자동 진화 메커니즘을 통해 고품질 데이터를 대규모로 생성합니다:[1]

**고품질 쿼리 생성:**
- 모바일 앱: 방향성 비순환 그래프(DAG) 구조를 활용하여 현실적인 네비게이션 경로 모델링
- PC 애플리케이션: 원자적 작업(클릭, 스크롤 등)과 소프트웨어 운영 경로의 조합 생성

**궤적 정확성 판단 모듈(Trajectory Correctness Judgment Module):**

단계 수준 평가 함수:
$$\pi^{\text{step}}_{\text{critic}}(\epsilon, a, \epsilon') \rightarrow (a, s, l)$$

여기서:
- $\epsilon$: 사전 행동 상태
- $a$: 실행된 작업
- $\epsilon'$: 사후 행동 상태
- 출력: 분석(a), 요약(s), 레이블 $l \in \{\text{GOOD, NEUTRAL, HARMFUL}\}$

궤적 수준 평가는 텍스트 추론 채널과 다중 모달 추론 채널의 합의 메커니즘으로 결정됩니다:[1]

$$\text{Trajectory Correctness} = \begin{cases} \text{Correct}, & \text{if } \pi^{\text{text}}(T, I) = \text{Correct} \land \pi^{\text{multimodal}}(T, I) = \text{Correct} \\ \text{Incorrect}, & \text{otherwise} \end{cases}$$

**쿼리 특정 지도 생성:**
VLM을 사용하여 참조 궤적의 각 단계에 대한 행동 결과 설명 생성, LLM을 통해 요약

**2.2.2 다양한 기본 데이터 합성**

세 가지 주요 데이터 파이프라인:[1]

**UI 요소 그라운딩:**
- 기능 기반 설명
- 외관 및 배치 기반 설명  
- 세밀한 단어/문자 그라운딩

**작업 계획:**
- 과거 궤적에서 절차 지식 추출
- 대규모 사전 학습된 LLM에서 지식 증류

**행동 의미론:**
사전-사후 상태 전환을 통한 행동의 영향 학습

**2.2.3 강화된 강건한 추론**

**오프라인 힌트 기반 거절 샘플링:**

주어진 궤적 $T = \{(a_0, S_0), (a_1, S_1), \ldots, (a_t, S_t)\}$에 대해 VLM이 각 단계에서 추론을 생성하고, 생성된 추론을 독립적으로 사용하여 행동 예측 검증

**반복 온라인 거절 샘플링:**

$$T^{(k)} = \text{Rollout}\left(M^{(k)}, Q\right) \quad (1)$$

$$M^{(k+1)} = \text{Train}\left(M^{(k)}, T^{(k)}_{\text{filtered}}\right) \quad (2)$$

필터링 전략:
1. 비평가 기반 필터링: $T_{\text{filtered}} = \{s_t | \text{CriticScore}(s_t) \geq \tau_c\}$
2. 사고-행동 일관성 검증
3. 작업 재가중치: 성공률 $p_{\text{succ}}(\text{task})$에 따라 샘플 가중치 조정

***

### 3. 모델 구조 및 성능 향상

#### 3.1 GUI-Owl 모델 구조

**End-to-End GUI 상호작용:**

정책 모델 $\pi$는 현재 관찰값 $S_t$와 과거 작업 이력 $H_t = \{(S_1, a_1), \ldots, (S_{t-1}, a_{t-1})\}$를 바탕으로 행동 공간 위의 확률 분포를 생성합니다:[1]

$$a_t \sim \pi(\cdot | S_t, H_t)$$

모델은 세 단계의 출력을 생성합니다:
1. **추론(Reasoning)**: 현재 상황에 대한 명시적인 사고 과정
2. **결론(Conclusion)**: 추론의 핵심을 30단어 이내로 요약
3. **행동(Action)**: 최종 실행 행동

이러한 구조를 통해 GPU 메모리 사용을 줄이고 추론 속도를 개선하면서도 복잡한 작업 적응 능력을 유지합니다.

**기본 에이전트 기능:**
GUI-Owl은 단순 자율 에이전트로 작동할 뿐만 아니라 Mobile-Agent-v3 다중 에이전트 프레임워크 내에서 특화된 모듈로도 기능합니다.

#### 3.2 훈련 패러다임

세 단계 훈련 프로세스:[1]

1. **사전 훈련 단계**: 기본 UI 이해, 상호작용 궤적, 일반 추론 데이터로 지속적 사전 훈련
2. **반복 튜닝 단계**: 실제 환경에서의 대규모 작업 실행 결과를 수집하여 다양한 추론 데이터셋 구성 후 오프라인 훈련
3. **강화학습 단계**: 비동기 RL 프레임워크를 통해 실제 환경과의 상호작용에서 학습

#### 3.3 확장 가능한 강화학습 인프라

**통합 인터페이스 기반의 다중 작업 훈련:**
- 단일 턴 추론과 다중 턴 에이전트 작업을 통합된 플러그인 인터페이스로 표준화
- 모든 구성 요소가 병렬로 실행되어 높은 처리량 달성

**비결합 제어 롤아웃:**
- 경험 생성과 정책 업데이트 분리
- 정책 준수 정도, 리소스 할당, 데이터 생성 프로세스에 대한 세밀한 제어

#### 3.4 Trajectory-aware Relative Policy Optimization (TRPO)

이는 긴 가변 길이 행동 수열의 온라인 환경 RL 훈련을 위한 핵심 기법입니다:[1]

**궤적 수준 보상:**
$$R(\tau) = \text{정확도 보상} + \text{형식 보상}$$

정확도 보상: 성공 시 1, 실패 시 0
형식 보상: 잘못된 형식 행동에 대해 -0.5

**정규화된 이점 추정:**
$$\hat{A}_\tau = \frac{R(\tau) - \bar{R}}{\sigma_R + \epsilon}$$

여기서 $\bar{R}$과 $\sigma_R$은 여러 궤적에서 관찰된 보상의 실행 평균과 표준편차입니다.

**최종 손실 함수:**

$$L_{\text{TRPO}} = -\frac{1}{N} \sum_{i=1}^{G} \sum_{s=1}^{S_i} \sum_{t=1}^{|o_{i,s}|} \min\left(r_t(\theta) \hat{A}_{\tau_i}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{\tau_i}\right)$$

여기서:
- $N$: 배치 내 토큰의 총 개수
- $\hat{A}_{\tau_i}$: 궤적 $i$의 궤적 수준 이점
- $r_t(\theta) = \frac{\pi_\theta(o_{s,t}|...)}{\pi_{\theta_{\text{old}}}(o_{s,t}|...)}$: 현재와 이전 정책의 토큰 확률 비율

**리플레이 버퍼 메커니즘:**
생성된 궤적이 모두 실패인 경우, 동일한 작업 ID의 성공한 궤적 중 하나를 임의로 선택하여 배치에 주입. 이는 매 배치에서 효과적인 훈련 신호 보장.

#### 3.5 성능 향상 결과

**개별 모델 성능:**[1]
- **GUI-Owl-7B**:
  - OSWorld-Verified: 29.4점 (일반 버전), 34.9점 (온라인 RL 특화)
  - AndroidWorld: 66.4점
  - MMBench-GUI L1 Hard: 90.9점

- **GUI-Owl-32B**:
  - OSWorld-Verified: 34.9점
  - AndroidWorld: 72.8점
  - MMBench-GUI L1 Hard: 94.2점

**다중 에이전트 프레임워크 성능:**[1]
- **Mobile-Agent-v3**:
  - OSWorld-Verified: 37.7점
  - AndroidWorld: 73.3점

이는 개별 모델 성능에 비해 추가 3.3-6.9점의 향상을 보여줍니다.

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 일반화 성능 분석

**크로스 플랫폼 일반화:**

GUI-Owl은 10개의 주요 GUI 자동화 벤치마크에서 테스트되었습니다:[1]

| 벤치마크 | 초점 | GUI-Owl 성능 |
|---------|------|------------|
| ScreenSpot-V2 | 모바일/데스크톱/웹 그라운딩 | 92.8% (7B) |
| MMBench-GUI L2 | 광범위 UI 이해 | 80.49% (7B) |
| OSWorld-G | 정밀 명령 그라운딩 | 55.9% (7B) |
| AndroidWorld | 모바일 end-to-end | 66.4% (7B) |

**세밀한 일반화 능력:**

라디오 버튼 클릭, 메뉴 탐색, 스크롤 기반 선택 등 다양한 UI 패턴에서 일관된 성능 시연

#### 4.2 일반화 성능 향상의 핵심 요인

**1. 다양한 데이터 파이프라인:**

그라운딩, 계획, 행동 의미론 등 다중 관점의 기본 UI 데이터가 모델의 기본 역량을 강화합니다. 이를 통해:[1]

- 서로 다른 도메인의 과제에 대한 전이 학습 가능성 증가
- 사전 학습된 배경 지식이 미지의 UI 환경에서의 적응을 가속화

**2. Self-Evolving Data Generation의 역할:**

온라인 환경에서 지속적으로 데이터를 생성하고 정제함으로써:

- 초기 데이터셋의 편향 완화
- 새로운 UI 패턴 자동 발견 및 학습
- 분포 이동(distribution shift) 극복 가능성

**3. 온라인 강화학습과 오프라인 학습의 결합:**

$$M^{(k+1)} = \text{Train}\left(M^{(k)}, T^{(k)}_{\text{filtered}}\right)$$

이 반복적 개선 루프는:
- 정적 데이터셋의 한계 극복
- 모델 능력 향상에 따라 수집되는 데이터 품질도 함께 개선
- 자기 강화(self-reinforcing) 개선 사이클 형성

#### 4.3 Trajectory-aware TRPO의 일반화 기여

**신용 할당 문제 해결:**

기존 단계 수준 보상의 문제점을 극복하기 위해 궤적 수준 보상 도입:

$$\hat{A}_\tau = \frac{R(\tau) - \bar{R}}{\sigma_R + \epsilon}$$

이를 통해:
- 모든 단계에 균등 분배되는 신호로 균형 잡힌 학습
- 장기 지평 작업에서의 더 안정적인 학습

**리플레이 버퍼의 안정성 기여:**

성공한 궤적을 리플레이 버퍼에 저장하고 재사용함으로써:
- 희소한 성공 피드백의 효율성 극대화
- 훈련 신호 분포의 안정화

실험 결과, 이 메커니즘이 없을 경우 성공률이 31.5%에서 34.9%로의 향상이 31.5% 이상으로 하락하는 것을 관찰했습니다.[1]

#### 4.4 일반화 한계와 향후 개선 방향

**현재 한계:**[1]

1. **도메인 특화 어려움**: 매우 특수한 응용 프로그램(예: 특정 엔터프라이즈 소프트웨어)에 대해서는 추가 미세 조정 필요
2. **장기 기억 제약**: 현재 구조에서 지난 20-30 단계만 메모리에 유지
3. **상황 길이 한계**: Qwen2.5-VL의 32k 토큰 제약으로 매우 긴 상호작용 시퀀스 처리 어려움

**향후 개선 가능 방향:**

1. **메모리 메커니즘 강화**: 장기 작업 기억을 위한 외부 메모리 구조 도입
2. **메타 학습**: 새로운 애플리케이션에 빠른 적응을 위한 메타 학습 프레임워크 통합
3. **동적 프롬프팅**: 작업 특성에 따른 동적 프롬프트 생성 메커니즘
4. **모듈식 기능 라이브러리**: 자동 발견된 재사용 가능한 작업 의존성 라이브러리 구축

***

### 5. Mobile-Agent-v3 프레임워크

#### 5.1 프레임워크 아키텍처

Mobile-Agent-v3는 네 개의 특화된 에이전트로 구성됩니다:[1]

**Manager Agent (M):**
- 역할: 전략적 계획자
- 초기화 함수: $(SS_0, CS_0) = M_{\text{init}}(I, S_0, K_{\text{RAG}})$
- 업데이트 함수: $(SS_t, CS_t) = M_{\text{update}}(I, S_{t-1}, SS_{t-1}, CS_{t-1}, A_{t-1}, F_{t-1}, N_t)$

**Worker Agent (W):**
- 역할: 전술적 실행자
- 함수: $A_t = W(I, S_t, SS_t, F_{t-1}, N_t)$
- 출력: $A_t = (\tau_t, \alpha_t, \sigma_t)$ (사고, 행동, 요약)

**Reflector Agent (R):**
- 역할: 자기 수정 메커니즘
- 함수: $F_t = R(I, S_t, S_{t+1}, A_t)$
- 출력: $F_t = (j_t, \phi_t)$ (판단, 피드백)

**Notetaker Agent (C):**
- 역할: 지속적 맥락 메모리
- 함수: $N_t = C(S_t)$
- 활성화: 성공 시에만

#### 5.2 에이전트 협력 메커니즘

**상태 변수 정의:**[1]

$$\text{Device State: } S_t \in \mathbb{R}^{H \times W \times C}$$
$$\text{Subsequent Subgoals: } SS_t = (g_1, g_2, \ldots, g_k)$$
$$\text{Completed Subgoals: } CS_t = \{\bar{g}_1, \bar{g}_2, \ldots, \bar{g}_m\}$$

**외부 지식 검색(RAG):**

$$Q = \text{GenerateQueries}(I)$$
$$D = \text{SearchEngine}(Q)$$
$$K_{\text{RAG}} = \text{Process}(D)$$

이를 통해 초기 계획 품질 향상

#### 5.3 성능 개선

Mobile-Agent-v3 프레임워크 적용 결과:[1]

| 벤치마크 | GUI-Owl-7B | Mobile-Agent-v3 | 향상 |
|---------|-----------|-----------------|------|
| OSWorld-Verified | 29.4 | 37.7 | +8.3 |
| AndroidWorld | 66.4 | 73.3 | +6.9 |

다중 에이전트 구조의 협력 메커니즘이 개별 모델의 한계를 보완하여 추가 성능 향상 달성.

***

### 6. 한계 및 도전 과제

#### 6.1 알려진 한계

**1. 데이터 의존성:**
- 자동 생성 데이터의 품질이 초기 모델 성능에 의존하여 악순환 가능성
- 매우 특수한 앱은 충분한 훈련 데이터 확보 어려움

**2. 환경 다양성 제약:**
- 현재 가상 환경 기반이므로 실제 디바이스의 모든 특성 반영 불가
- 네트워크 지연, 시스템 예외 등 현실의 변수 불충분

**3. 복잡한 추론 한계:**
- 다단계 의존성이 있는 매우 복잡한 워크플로우에서 성능 저하
- 직관적 도메인 지식이 필요한 작업에서 제한적

**4. 비용과 확장성:**
- 대규모 가상 환경 인프라 유지 비용
- 새로운 앱에 대한 자동 데이터 생성 프로세스 최적화 필요

#### 6.2 향후 극복 필요 과제

1. **하이브리드 접근법**: 수동 주석과 자동 생성의 최적 조합 발견
2. **도메인 특화 모델**: 특정 산업용 GUI 에이전트 개발
3. **설명 가능성 강화**: 에이전트 의사결정 과정의 투명성 증대
4. **안전성 보증**: 비용이 큰 작업에서의 실수 방지 메커니즘

***

### 7. 관련 최신 연구 동향 (2020년 이후)

#### 7.1 GUI 에이전트 분야의 주요 발전

**초기 단계 (2023-2024):**
- **AutoGLM (2024)**: 자율 환경 상호작용을 통한 기초 에이전트 학습
- **ShowUI (2024)**: UI 기반 토큰 선택으로 33% 중복 토큰 제거, 1.4배 성능 가속화
- **GUICourse (2024)**: VLM을 GUI 에이전트로 전환하기 위한 데이터셋 스위트

**중기 단계 (2024-2025):**
- **UI-TARS (2025.1)**: 네이티브 GUI 에이전트 모델로 10+ 벤치마크에서 SOTA 달성
- **Agent S2 (2025.4)**: Generalist-Specialist 조합 프레임워크로 다단계 작업 성능 향상
- **MobileRL (2025.10)**: 난이도 적응형 GRPO로 AndroidWorld 80.2% 달성

**최신 동향 (2025년):**[2][3][4][5]

1. **강화학습의 고도화:**
   - MobileRL: Difficulty-ADAptive GRPO로 어려운 작업 처리 개선
   - Mobile-R1: 작업 수준 보상 기반의 다중 턴 RL
   - UI-TARS-2: 다중 턴 RL과 하이브리드 환경(GUI + 파일 시스템 + 터미널) 통합

2. **일반화 성능 향상:**
   - Breaking the Data Barrier (2025): 작업 일반화를 통한 데이터 부족 극복
   - GUI-Xplore (2025): 탐색 비디오를 통한 크로스 앱 일반화
   - RegionFocus (2025): 시각적 테스트 타임 스케일링으로 28% 성능 향상

3. **데이터 생성의 자동화:**
   - CCAgent (2025): Web3 기반 분산 데이터 수집 시스템
   - MagicGUI (2025): 확장 가능한 데이터 파이프라인으로 7.8M 샘플 수집
   - GUI-explorer (2025): 훈련 없이 자동 탐색 및 지식 추출

#### 7.2 다중 에이전트 시스템의 진화

**협력 메커니즘 발전:**
- 기존의 단순 역할 분담에서 벗어나 **동적 작업 재배분**, **공유 추론 상태**, **상호 학습** 메커니즘으로 진화
- Mobile-Agent-v3의 Manager-Worker-Reflector-Notetaker 구조가 이러한 추세를 잘 반영

**RAG(검색 증강 생성) 통합:**
- 실시간 외부 지식 검색으로 도메인 특화 정보 활용
- 웹 검색, 문서 데이터베이스 등 다양한 지식 소스 통합

#### 7.3 오픈소스 vs 폐쇄형 모델 성능 격차 축소

**성능 비교 추이:**[6][7][1]

| 연도 | 오픈소스 최고 성능 | 폐쇄형 최고 성능 | 격차 |
|------|-----------------|-----------------|------|
| 2024 | UI-TARS (46.6%) | GPT-4o (34.5%) | -12.1% |
| 2025 | MobileRL-9B (80.2%) | Claude-4 (43.9%) | -36.3% |

오픈소스 모델이 폐쇄형 모델을 상당히 초과하는 추세 관찰.

#### 7.4 산업 응용 확대

- **엔터프라이즈 자동화**: RPA(Robotic Process Automation) 분야 수요 증가
- **모바일 앱 테스팅**: 자동화된 GUI 테스트로 개발 효율성 향상
- **접근성 개선**: 장애인용 컴퓨터 보조 기술
- **교육**: 사용자 행동 분석 및 개인화 학습 경로 제시

***

### 8. 향후 연구에 미치는 영향 및 고려사항

#### 8.1 향후 연구에 미치는 영향

**8.1.1 기술적 영향**

1. **자동 데이터 생성 패러다임 확립:**
   - 수동 주석의 병목 극복으로 GUI 에이전트 개발 가속화
   - 다른 도메인(로봇, 비디오 이해 등)의 자동 데이터 생성 방법론에도 영향

2. **다중 에이전트 설계 패턴 정립:**
   - Manager-Worker-Reflector-Notetaker 구조가 복잡한 작업 자동화의 표준 패턴으로 자리잡을 가능성
   - 다른 자동화 도메인(코드 생성, 콘텐츠 생성 등)으로의 확장

3. **강화학습의 새로운 방향:**
   - Trajectory-aware TRPO가 긴 시퀀스 RL의 표준 기법으로 정착
   - 신용 할당 문제의 일반적 해결책 제시

**8.1.2 응용 분야 확대**

1. **엔터프라이즈 시스템:**
   - 복잡한 업무 프로세스의 자동화 (ERP, CRM 시스템 등)
   - 레거시 소프트웨어 현대화를 위한 자동 마이그레이션

2. **접근성 기술:**
   - 장애인 사용자를 위한 적응형 인터페이스
   - 음성 명령 기반의 GUI 자동화

3. **사용자 행동 분석:**
   - A/B 테스트 자동화
   - UI/UX 개선을 위한 사용성 분석

#### 8.2 향후 연구 시 고려할 점

**8.2.1 데이터 및 방법론 관점**

1. **데이터 품질 보증:**
   - 자동 생성된 데이터의 체계적인 품질 검증 메커니즘 강화
   - 도메인 전문가 피드백 루프 통합의 필요성

2. **분포 이동 대응:**
   - 새로운 UI 패턴 출현에 대한 적응 메커니즘 개발
   - 도메인 적응(domain adaptation) 기술의 적극 활용

3. **계산 효율성:**
   - 현재 인프라의 높은 계산 비용 절감 방법
   - 에지 디바이스에서의 배포 가능성 탐색

**8.2.2 안전성 및 신뢰성 관점**

1. **오류 방지 메커니즘:**
   - 중요 작업에서의 실패 모드 분석 및 대응
   - 작업 복구 및 롤백 메커니즘 개발

2. **설명 가능성:**
   - 에이전트 의사결정 과정의 투명성 증대
   - 사용자 감시 하에서의 신뢰 구축

3. **보안:**
   - 악의적 입력에 대한 방어 메커니즘
   - 민감한 정보 보호 및 접근 제어

**8.2.3 평가 및 벤치마킹 관점**

1. **새로운 평가 지표:**
   - 현실성(realism) 있는 벤치마크 개발
   - 사용자 만족도, 작업 완료 효율성 등 실용적 지표

2. **도메인 특화 평가:**
   - 산업별, 앱 종류별 성능 상세 분석
   - 장기 지평 작업에 대한 평가 방법론

3. **공정한 비교:**
   - 동일 조건의 벤치마크 환경 표준화
   - 오픈소스와 폐쇄형 모델의 공정한 비교 체계

**8.2.4 윤리 및 사회적 영향**

1. **일자리 영향:**
   - GUI 자동화로 인한 직업 변화 분석
   - 재교육 및 직업 전환 지원 방안 모색

2. **접근성:**
   - 개발도상국에서의 오픈소스 에이전트 활용 기회 확대
   - 상용 솔루션 비용 절감 효과

3. **규제 준비:**
   - AI 에이전트 사용에 관한 규제 프레임워크 선제 대응
   - 산업별 가이드라인 개발

***

### 결론

**Mobile-Agent-v3**는 GUI 자동화 분야에 세 가지 핵심 기여를 통해 획기적인 진전을 이룬 연구입니다:[1]

1. **자동 진화 데이터 생성** 패러다임을 통해 수동 주석 병목을 해결
2. **다중 에이전트 협력** 구조로 복잡한 다단계 작업 처리 능력 향상
3. **Trajectory-aware TRPO** 알고리즘으로 장기 지평 작업의 신용 할당 문제 극복

관련 최신 연구들(UI-TARS-2, MobileRL, Mobile-R1 등)은 이러한 기초 위에서 지속적으로 성능을 개선하고 있으며, 오픈소스 모델이 폐쇄형 상용 모델을 초과하는 추세를 보이고 있습니다.

향후 연구는 **도메인 특화, 안전성 강화, 실무 배포, 윤리적 고려**라는 네 가지 축에서 진행될 것으로 예상되며, 이들 연구가 활발해질수록 GUI 자동화 기술의 실제 산업 적용이 가속화될 것으로 전망됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fefe6f20-fb65-41c4-9ded-666f88e16392/2508.15144v2.pdf)
[2](https://arxiv.org/abs/2509.18119)
[3](https://arxiv.org/abs/2504.10127)
[4](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_GUI-Xplore_Empowering_Generalizable_GUI_Agents_with_One_Exploration_CVPR_2025_paper.pdf)
[5](https://arxiv.org/abs/2506.20332)
[6](https://arxiv.org/abs/2509.02544)
[7](https://arxiv.org/abs/2501.12326)
[8](https://www.frontiersin.org/articles/10.3389/frai.2025.1649155/full)
[9](https://ijcaonline.org/archives/volume187/number24/joshi-2025-ijca-925428.pdf)
[10](https://ijirem.org/view_abstract.php?title=Review-of-Autonomous-and-Collaborative-Agentic-AI-and-Multi-Agent-Systems-for-Enterprise-Applications&year=2025&vol=12&primary=QVJULTE5MDM=)
[11](https://aacrjournals.org/clincancerres/article/31/2_Supplement/B003/750941/Abstract-B003-Landscape-of-chemoradiation-plus)
[12](https://www.semanticscholar.org/paper/59bbe382b85ceef85b5480e3dd17002524f85c5d)
[13](https://arxiv.org/abs/2412.04454)
[14](http://arxiv.org/pdf/2403.10171.pdf)
[15](http://arxiv.org/pdf/2410.11872.pdf)
[16](https://arxiv.org/html/2411.00820v1)
[17](https://arxiv.org/html/2504.00906v1)
[18](http://arxiv.org/pdf/2406.11317.pdf)
[19](https://arxiv.org/html/2412.18426)
[20](https://arxiv.org/html/2503.02268v1)
[21](https://www.linkedin.com/posts/rakeshgohel01_what-is-the-future-of-gui-ai-agents-in-2025-activity-7270445808754257920-KDaW)
[22](https://arxiv.org/abs/2411.17465)
[23](https://machinelearning.apple.com/research/safe-real-world)
[24](https://www.getaiverse.com/post/ki-gesteuerte-gui-agenten-ueberwindung-der-datenhuerde-durch-aufgaben-generalisierung)
[25](https://arxiv.org/html/2508.15144v2)
[26](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_ShowUI_One_Vision-Language-Action_Model_for_GUI_Visual_Agent_CVPR_2025_paper.pdf)
[27](https://aclanthology.org/2025.acl-long.1065/)
[28](https://openreview.net/forum?id=UXdxYnkJtX)
[29](https://github.com/X-PLUG/MobileAgent)
[30](https://www.semanticscholar.org/paper/55cf15245bbda5e9117c0d1e46a312175c8c9f2a)
[31](https://arxiv.org/abs/2505.00684)
[32](https://dl.acm.org/doi/10.1145/3746252.3761392)
[33](https://arxiv.org/abs/2402.07939)
[34](https://ieeexplore.ieee.org/document/11093438/)
[35](https://arxiv.org/abs/2410.11871)
[36](https://arxiv.org/abs/2508.03700)
[37](https://arxiv.org/abs/2505.16827)
[38](https://arxiv.org/html/2501.12326)
[39](https://arxiv.org/pdf/2402.07939.pdf)
[40](https://arxiv.org/pdf/2501.04575.pdf)
[41](http://arxiv.org/pdf/2407.09018.pdf)
[42](https://arxiv.org/pdf/2403.17918.pdf)
[43](https://www.aclweb.org/anthology/2020.acl-demos.25.pdf)
[44](https://arxiv.org/pdf/2311.08649.pdf)
[45](https://www.emergentmind.com/topics/trajectory-aware-relative-policy-optimization-trpo-5e9d14e3-06ce-4f54-813d-271b71dcd450)
[46](https://arxiv.org/abs/2508.15144)
[47](https://www.themoonlight.io/ko/review/ui-tars-pioneering-automated-gui-interaction-with-native-agents)
[48](https://www.emergentmind.com/topics/trust-region-policy-optimization-trpo)
[49](https://www.emergentmind.com/topics/gui-owl-model)
[50](https://github.com/bytedance/UI-TARS-desktop)
[51](https://www.themoonlight.io/en/review/enhancing-ppo-with-trajectory-aware-hybrid-policies)
[52](https://www.aiworldtoday.net/p/the-dawn-of-self-evolving-ai-agents)
[53](https://agenttars.org/ko)
[54](https://www.sciencedirect.com/science/article/pii/S0925231224004879)
[55](http://proceedings.mlr.press/v97/doerr19a/doerr19a.pdf)
[56](https://arxiv.org/html/2505.12370v2)
