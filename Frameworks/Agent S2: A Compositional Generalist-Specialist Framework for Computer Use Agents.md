# Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents

## 1. 핵심 주장 및 주요 기여도 요약

**Agent S2**는 **합성 프레임워크(Compositional Framework)**를 통해 컴퓨터 사용 에이전트의 성능을 획기적으로 향상시킨다는 것을 핵심 주장으로 한다. 기존의 단일 일반화 모델(Monolithic Generalist Model)에 의존하는 접근방식의 한계를 지적하고, 다양한 전문화된 모듈들로 인지 책임을 분산시키는 방식을 제시한다.[1]

### 기존 컴퓨터 사용 에이전트의 세 가지 핵심 문제점[1]

1. **GUI 요소 정확도 부족(Grounding Bottleneck)**: 텍스트 기반 GUI 요소 설명을 픽셀 수준의 정확한 좌표로 변환하는 데 어려움 존재
2. **장기 작업 계획의 어려움(Long-horizon Task Planning)**: 배경 앱, 팝업, 변화하는 사용자 관찰 등의 방해요소로 인한 계획 수립 곤란
3. **성능 병목(Performance Bottlenecks)**: 계획, 행동 생성, 접지(Grounding) 등 다양한 인지 작업에 단일 모델 사용으로 인한 한계

### Agent S2의 주요 기여도[1]

**기여도 1: 합성 계층적 프레임워크 도입**
- Manager-Worker 아키텍처와 Mixture of Grounding 전문가 모듈의 통합
- 고수준 추론(Manager), 저수준 실행(Worker), 세밀한 접지(Grounding Experts)의 책임 분산

**기여도 2: Mixture-of-Grounding(MoG) 기법 개발**
- 시각적 접지 전문가(Visual Grounding Expert): UI-TARS-72B-DPO 활용
- 텍스트 접지 전문가(Textual Grounding Expert): Tesseract OCR 활용
- 구조적 접지 전문가(Structural Grounding Expert): UNO 인터페이스 활용

**기여도 3: 능동적 계층적 계획(Proactive Hierarchical Planning) 도입**
- 반응적 계획과 달리 매 부분작업 완료 후 계획 갱신
- 환경 변화에 대한 적응성 향상

**기여도 4: 실증적 성능 향상**[1]
- OSWorld 15-step: 27.0% (기준 대비 18.9% 상대 향상)
- OSWorld 50-step: 34.5% (기준 대비 32.7% 상대 향상)
- WindowsAgentArena: 29.8% (NAVI 대비 52.8% 상대 향상)
- AndroidWorld: 54.3% (기존 최고 성능 대비 16.5% 상대 향상)

***

## 2. 문제 정의, 제안 방법, 모델 구조

### 2.1 문제 정의: POMDP 형식화[1]

컴퓨터 사용 에이전트는 다음과 같은 POMDP(Partially Observable Markov Decision Process)로 형식화할 수 있다:

$$M = (S, O, A, T, R)$$

여기서:
- $$S$$: 상태 공간(데스크톱의 현재 상태)
- $$O$$: 관찰 공간(명령, 스크린샷, 접근성 트리 등)
- $$A$$: 행동 공간(클릭, 타이핑 등)
- $$T: S \times A \rightarrow S$$: 상태 전이 함수
- $$R: S \times A \rightarrow $$: 보상 함수[1]

현재 에이전트들이 직면한 세 가지 주요 도전:[1]
1. 정밀 접지의 부재
2. 계획의 경직성
3. 일반화의 한계

### 2.2 제안하는 방법: Manager-Worker 계층 구조[1]

**고수준(High-level) - Manager (시간 단계 T):**

$$\pi_M: (I, \{g'_1, g'_2, \ldots, g'_n\}, o_T) \rightarrow \{g''_1, g''_2, \ldots, g''_m\}$$

여기서 I는 사용자 명령, $$\{g'_1, g'_2, \ldots, g'_n\}$$은 이전 부분목표들, $$o_T$$는 현재 관찰, $$\{g''_1, g''_2, \ldots, g''_m\}$$은 갱신된 부분목표 시퀀스이다.

**저수준(Low-level) - Worker (시간 단계 t):**

$$\pi_W: (g_i, o_t) \rightarrow a_t \in A$$

Worker는 부분목표 $$g_i$$를 달성하기 위해 원자적 행동 $$a_t$$를 순차적으로 생성한다.

### 2.3 Mixture-of-Grounding(MoG) 메커니즘[1]

Worker가 행동 $$a_t = (\text{action type}, d)$$ 를 생성했을 때, 언어 기술자 $$d$$를 기반으로 적절한 접지 전문가로 라우팅된다.

**시각적 접지 전문가:**

$$E_{visual}(o, d) \rightarrow \langle x, y \rangle$$

입력은 스크린샷 $$o$$와 언어 기술 $$d$$이고, 출력은 픽셀 좌표 $$\langle x, y \rangle$$이다.[1]

**텍스트 접지 전문가:**

$$E_{textual}(o, p_1, p_2) \rightarrow \{\langle x_{start}, y_{start} \rangle, \langle x_{end}, y_{end} \rangle\}$$

입력은 스크린샷 $$o$$와 시작/종료 구문 $$p_1, p_2$$이고, 출력은 텍스트 범위의 좌표이다.[1]

**구조적 접지 전문가:**

$$E_{structural}(\{(cell, value)\}, app, sheet) \rightarrow \text{SUCCESS}$$

스프레드시트 셀 업데이트를 프로그래매틱하게 수행한다.[1]

### 2.4 능동적 계층적 계획(Proactive Hierarchical Planning)[1]

**반응적 계획:** Worker가 작업 실패 시에만 Manager가 계획을 갱신
**능동적 계획:** 각 부분목표 완료 후 항상 Manager가 계획을 갱신

이러한 구조적 개선으로 Manager는 이전 부분목표 컨텍스트를 유지하면서 새로운 관찰을 통합할 수 있다.

***

## 3. 성능 향상 및 한계

### 3.1 성능 평가 결과[1]

#### OSWorld 벤치마크 (메인 평가)

| 방법 | 15-step | 50-step | 상대 향상도 |
|------|---------|---------|-----------|
| Claude-3.7-Sonnet (baseline) | 15.5% | 26.0% | - |
| OpenAI CUA | 19.7% | 32.6% | - |
| UI-TARS-72B-DPO | 22.7% | 24.6% | - |
| **Agent S2 (Claude-3.7-Sonnet)** | **27.0%** | **34.5%** | **+18.9% / +32.7%** |

#### 크로스플랫폼 일반화[1]

- **WindowsAgentArena**: 29.8% (이전 SOTA 19.5% 대비 +52.8%)
- **AndroidWorld**: 54.3% (이전 SOTA 46.6% 대비 +16.5%)

### 3.2 Ablation Study 결과[1]

**Mixture-of-Grounding의 기여도:**
- 15-step: 27.69% (기본) → 30.77% (MoG 적용) = +3.08%
- 50-step: 33.85% (기본) → 38.46% (MoG 적용) = +4.61%

**능동적 계획의 효과:**
- 15-step: 26.15% (반응) → 30.77% (능동) = +4.62%
- 50-step: 32.31% (반응) → 38.46% (능동) = +6.15%

### 3.3 오류 분석[1]

실패 원인 분류:

```
계획 오류 (Planning Failures): 41.0% ★★★★★
접지 오류 (Grounding Failures): 20.5% ★★
상호작용 오류 (Interaction Failures): 17.9% ★
네비게이션 오류 (Navigation Failures): 10.3% ★
불가능 작업 (Infeasible Tasks): 10.3%
```

### 3.4 한계[1]

1. **계획 오류의 높은 비율 (41%)**: Manager의 부분목표 분해 정확도 여전히 낮음
2. **도메인 특화 부족**: Office 작업 카테고리에서 특히 낮은 성능 (Calc 25.53%)
3. **계산 비용**: Claude-3.7-Sonnet의 Extended Thinking 사용으로 인한 높은 지연시간
4. **장기 작업의 누적 오류**: 단계 증가에 따른 성능 감소 경향
5. **OSWorld와의 40% 성능 격차**: 인간 수준 성능 달성에 아직 멀어

***

## 4. 모델 일반화 성능 향상 가능성

### 4.1 현재 일반화 성능 분석[1]

**크로스 플랫폼 일반화:**
- Ubuntu(학습) → Windows(평가): WindowsAgentArena에서 52.8% 향상
- Desktop(학습) → Mobile(평가): AndroidWorld에서 54.3% 달성

**도메인별 성능 (OSWorld 50-step):**
- 높은 성능: OS (50%), VS Code (65.22%), Thunderbird (73.33%)
- 중간 수준: Chrome (41.19%), GIMP (50%)
- 도전 과제: Writer (34.77%), Calc (25.53%)

### 4.2 일반화 메커니즘[1]

**MoG를 통한 일반화:**

$$E_{visual} = \text{UI-TARS-72B-DPO} \text{ (25,000+ 다양한 GUI 스크린샷으로 사전학습)}$$

시각적 접지는 픽셀 좌표 예측의 도메인 독립성으로 새로운 애플리케이션에서도 유사하게 작동한다.

**능동적 계획을 통한 일반화:**

$$\text{Generalization}_t = f(\text{Previous Context} + \text{Current Observation} + \text{Original Instruction})$$

이전 부분목표 정보의 지속적 유지로 맥락 혼실을 방지하고 새로운 관찰에 따른 동적 재계획을 수행한다.

### 4.3 향상 가능성

#### 단기 (6-12개월):
- **강화된 도메인 적응**: 도메인 적대적 학습으로 +3-5% 향상 기대
- **동적 라우팅 최적화**: 학습 기반 라우팅 도입으로 +3-5% 향상 기대
- **계획 오류 감소**: 검증 단계 추가로 +5-10% 향상 기대

#### 중기 (1-2년):
- **지식 베이스 확대**: +8-15% 향상 기대
- **멀티태스크 학습**: +10-20% 향상 기대
- **차세대 LLM 활용**: +5-10% 향상 기대

#### 장기 (2-5년):
- **End-to-End 학습**: +15-25% 향상 기대
- **환경 모델 학습**: +10-15% 향상 기대
- **메타 학습**: +20-30% 향상 기대

***

## 5. 학술 커뮤니티에 미치는 영향

### 5.1 패러다임의 전환

**기존 패러다임**: 단일 대규모 LLM이 모든 인지 작업 수행
**새로운 패러다임**: Task 특성에 따른 동적 전문가 모듈 조합[1]

이 패러다임 전환은:
- 컴퓨터 사용 에이전트 분야에 즉시 채택
- 멀티모달 AI 일반에 1-2년 내 광범위 확산 예상
- 기타 에이전트 연구(로봇공학, 게임 AI)에 중장기 영향

### 5.2 기술적 기여

**Mixture-of-Grounding의 일반화 가능성:**
- 의료 영상 분석
- 로봇공학 조작
- 자연어 처리의 문서 이해
- 5년 내 MoE 관련 논문 50+ 발표 예상

**능동적 계획의 체계화:**
- 재계획 빈도의 최적화 알고리즘 개발
- 동적 재계획 조건 학습
- 이론적 최적성 조건 증명 필요

### 5.3 새로운 평가 지표의 필요성[1]

**필요한 새로운 지표:**

$$Q_{plan} = \frac{\text{Successfully Executed Steps}}{\text{Total Generated Steps}}$$

$$A = \frac{\text{Successful Replans}}{\text{Total Replans}}$$

$$G = \frac{\text{Cross-Domain Performance}}{\text{In-Domain Performance}}$$

***

## 6. 향후 연구 시 고려할 점

### 6.1 기술적 고려사항

#### 성능-효율성 트레이드오프
- 현재: 높은 정확도 vs 느린 응답 시간 (50-step 평가 시 수 분 소요)
- 개선 방향: 경량 라우팅 모델, 동적 빔 서치, 병렬 처리

#### 표준화된 인터페이스 개발
- 새로운 전문가 추가 시 표준화된 입출력 인터페이스 필요
- 모듈 호환성 및 상호 운용성 강화

### 6.2 방법론적 고려사항

#### 새로운 벤치마크 설계 원칙
- **다양성**: 30+ 애플리케이션, 다양한 운영 체계
- **구분 가능성**: 성능 차이 5% 이상 구분 가능
- **견고성**: 평가 일관성, UI 변화에 대한 안정성
- **확장 가능성**: 신규 작업 추가 용이성

#### 반자동 라벨링
$$\text{Label} = \text{Model Prediction} + \text{Human Verification}$$

합성 데이터 생성을 통한 비용 절감

### 6.3 윤리 및 사회적 고려사항

#### 편향 및 공정성
- 모든 도메인에서 균등한 성능 추구
- 소수 도메인에 대한 특별한 최적화
- 정기적인 편향 감사

#### 투명성 및 설명 가능성
- 의사결정 추적 및 로깅
- 사용자 개입 메커니즘 제공
- 설명 가능한 AI 원칙 준수

#### 보안 및 프라이버시
- 접근 제어 및 권한 관리
- 민감 정보 자동 마스킹
- 모든 에이전트 행동 기록 및 감시

### 6.4 실무 배포 시 고려사항

#### 호환성 및 통합
- 다양한 환경 지원(온프레미스, 클라우드, 하이브리드)
- 기존 ERP/CRM/레거시 시스템과의 통합

#### 비용 모델

$$\text{TCO} = \text{Development} + \text{Deployment} + \text{Operation}$$

초기 개발, 배포 인프라, 지속적 유지보수 비용 분석 필요

***

## 7. 최신 관련 연구 (2020년 이후)

### 7.1 2024-2025년 주요 진전[2][3][4][5][6][7]

**ComputerRL (2025, Tsinghua):**
- 온라인 강화학습으로 OSWorld 48.9% 달성 (Agent S2의 34.5% 상회)
- Entropulse 전략: RL과 감독 학습의 교대 실행
- 수천 개 병렬 VM을 통한 대규모 학습 인프라 구축

**ScaleCUA (2025, OpenGVLab):**
- 6개 운영 체제와 3개 작업 도메인 포괄
- OSWorld-G: 60.6%, WebArena-Lite-v2: 47.4% 달성
- 자동화 에이전트와 인간 전문가의 폐쇄 루프 파이프라인

**UltraCUA (2025):**
- GUI 원시 액션과 고수준 프로그래밍 도구의 하이브리드 액션
- OSWorld에서 UI-TARS 대비 22% 상대 향상
- 17,000+ 검증 가능 작업으로 학습

**UI-Evol (2025):**
- Agent S2와 결합한 지식 진화 메커니즘
- Retrace Stage와 Critique Stage의 2단계 파이프라인
- 동적 지식 기반 업데이트

**Fara-7B (2025, Microsoft Research):**
- 다중 에이전트 복잡성을 단일 소규모 모델로 압축
- 16단계 평균 (UI-TARS의 41단계 대비)
- 효율성 우선 설계

### 7.2 비전 언어 모델 그라운딩의 발전[8][9][10]

**아키텍처 혁신:**
- "Your Large Vision-Language Models Only Needs A Few Attention Heads" (2025)
- 수천 개 헤드 중 단 3개의 위치 결정 헤드(Localization Heads)만 필요
- 미세조정 없이 경쟁력 있는 성능

**GUI 특화 모델:**
- ScreenAI (2025, Google): UI와 인포그래픽 특화
- ShowUI (2025): UI-Guided Visual Token Selection, 인터리빙 비전-언어-액션 스트리밍

**자기 수정 메커니즘:**
- VLM이 오라클 피드백 없이도 그라운딩 오류 수정 가능
- 반복적 자기 수정으로 최대 8.4% 정확도 향상[11]

### 7.3 계층적 계획 및 작업 분해 연구

**AgentOrchestra (2025):**
- 계획 에이전트(고수준) + 특화 서브 에이전트(저수준)
- 확장 가능한 협력 구조
- 적응형 계획

**HiSOMA (2024):**
- 시간적 및 구조적 계층적 작업 분해
- 장기 시각(Long-horizon) 작업의 신용할당 문제 해결

### 7.4 Mixture-of-Experts 기술 발전[12][13]

**새로운 라우팅 메커니즘:**
- **LLMoE**: 전통 게이팅 네트워크 대신 사전학습 LLM 활용
- **DA-MoE**: 토큰 중요도에 따른 가변 전문가 할당
- **RMoE**: 레이어별 순환 라우터로 의존성 활용

**배포 최적화:**
- ExpertFlow: 동적 라우팅에 적응하는 캐싱 전략
- Read-ME: 라우터 분리 설계(Router-Decoupled MoE)

### 7.5 새로운 벤치마크 및 평가[14][7][15][8]

**MCPWorld (2025):**
- API, GUI, 하이브리드 컴퓨터 사용 에이전트의 통합 평가
- 201개 큐레이션된 작업
- 화이트박스 앱과 동적 코드 계측을 통한 강건한 평가

**WorldGUI (2025):**
- 10개 데스크톱/웹 애플리케이션 포함
- 실용적 평가

**GUI-Xplore (2025, CVPR):**
- 탐색 비디오를 통한 크로스 앱 일반화
- GUI 전이 그래프로 환경 모델링
- 미지의 앱에서 10% 성능 향상

**OSWorld-MCP (2025):**
- MCP(Model Context Protocol) 도구 호출 능력 평가
- 도구 통합으로 성능 향상 (OpenAI o3: 8.3% → 20.4%)

### 7.6 도메인 특화 에이전트 연구

**BIMgent (2025, TUM):**
- 건축 정보 모델링(BIM) 자동화
- 3D 건축 모델링 소프트웨어 제어
- 32% 성공률 (모든 베이스라인 0%)[4]

***

## 8. 결론

Agent S2는 컴퓨터 사용 에이전트 분야에서 **패러다임 전환**을 이끌어낸 획기적인 연구이다. **합성 프레임워크**를 통해 일반화 모델과 전문화 모델의 최적 조합을 실현하였으며, **Mixture-of-Grounding**과 **Proactive Hierarchical Planning**이라는 구체적이고 검증된 방법론을 제시했다.[1]

### 주요 성과
- OSWorld: 27.0% (15-step), 34.5% (50-step) - SOTA 달성[1]
- 크로스플랫폼 일반화: WindowsAgentArena +52.8%, AndroidWorld +16.5%[1]
- 이론적 기여: 모듈화 설계의 효과 실증

### 미해결 과제
- 계획 오류 (41%)의 근본적 해결
- 도메인 적응의 효율성 개선
- 실제 산업 배포를 위한 안정성 강화

### 향후 연구 방향
1. **기술적 진화**: 더 강력한 계획 모듈, 계산 효율성 최적화
2. **학술적 파급**: 모듈화 AI 설계 원칙의 일반화
3. **산업적 영향**: 자동화 기술의 실용화 가속

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3c6834dd-ca7e-458c-ba9d-1b287e6a60a4/2504.00906v1.pdf)
[2](https://arxiv.org/abs/2508.14040)
[3](https://arxiv.org/abs/2509.15221)
[4](https://arxiv.org/abs/2506.07217)
[5](https://arxiv.org/abs/2510.17790)
[6](https://arxiv.org/abs/2505.21964)
[7](https://arxiv.org/abs/2510.24563)
[8](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_GUI-Xplore_Empowering_Generalizable_GUI_Agents_with_One_Exploration_CVPR_2025_paper.pdf)
[9](https://arxiv.org/html/2509.10345v1)
[10](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_ShowUI_One_Vision-Language-Action_Model_for_GUI_Visual_Agent_CVPR_2025_paper.pdf)
[11](https://openreview.net/forum?id=fO1xnmW8T6&noteId=LT2fCTJ3qi)
[12](https://www.rohan-paul.com/p/mixture-of-experts-moe-architectures)
[13](https://arxiv.org/html/2507.11181v1)
[14](https://arxiv.org/abs/2506.07672)
[15](https://arxiv.org/html/2502.08047v3)
[16](https://www.semanticscholar.org/paper/79b9b4120eb4a5bfc9070766dc51e6b794d3de3e)
[17](http://ieeexplore.ieee.org/document/6426341/)
[18](https://arxiv.org/abs/2504.00906)
[19](https://arxiv.org/html/2504.00906v1)
[20](http://arxiv.org/pdf/2410.23218v1.pdf)
[21](https://arxiv.org/pdf/2403.17918.pdf)
[22](http://arxiv.org/pdf/2406.11317.pdf)
[23](https://arxiv.org/html/2412.01268v1)
[24](https://arxiv.org/abs/2412.04454)
[25](https://arxiv.org/html/2411.00820v1)
[26](https://arxiv.org/pdf/2306.00245.pdf)
[27](https://www.tencentcloud.com/techpedia/126570)
[28](https://arxiv.org/html/2504.12679v3)
[29](https://arxiv.org/html/2506.12508v1)
[30](https://openreview.net/forum?id=ZMOLw9eBPm)
[31](https://openaccess.thecvf.com/content/CVPR2025/papers/Kang_Your_Large_Vision-Language_Model_Only_Needs_A_Few_Attention_Heads_CVPR_2025_paper.pdf)
[32](https://www.sciencedirect.com/science/article/abs/pii/S0957417424009837)
[33](https://www.simular.ai/articles/agent-s2)
[34](http://arxiv.org/pdf/2406.00023.pdf)
[35](https://arxiv.org/pdf/2403.07652.pdf)
[36](https://arxiv.org/pdf/2204.08396v1.pdf)
[37](https://arxiv.org/html/2503.16057)
[38](http://arxiv.org/pdf/2409.06669.pdf)
[39](http://arxiv.org/pdf/2408.06793.pdf)
[40](https://arxiv.org/html/2410.17954)
[41](http://arxiv.org/pdf/2410.19123.pdf)
[42](https://www.tencentcloud.com/techpedia/126447)
[43](https://arxiv.org/abs/2508.16271)
[44](https://www.microsoft.com/en-us/research/blog/fara-7b-an-efficient-agentic-model-for-computer-use/)
[45](https://intuitionlabs.ai/articles/mixture-of-experts-moe-models)
[46](https://research.google/blog/screenai-a-visual-language-model-for-ui-and-visually-situated-language-understanding/)
[47](https://arxiv.org/html/2504.21433v1)
[48](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
[49](http://arxiv.org/pdf/2407.21359.pdf)
[50](http://arxiv.org/pdf/2005.07404.pdf)
[51](https://arxiv.org/html/2402.13785v1)
[52](https://arxiv.org/pdf/2303.03339.pdf)
[53](https://arxiv.org/pdf/1911.08453.pdf)
[54](https://arxiv.org/ftp/arxiv/papers/2303/2303.08731.pdf)
[55](https://www.frontiersin.org/articles/10.3389/frobt.2023.1255696/pdf?isPublishedV2=False)
[56](https://scindeks-clanci.ceon.rs/data/pdf/1451-4117/2021/1451-41172101048C.pdf)
[57](https://www.nature.com/articles/s41467-025-56183-4)
[58](https://learn.microsoft.com/en-us/answers/questions/2225221/is-there-a-way-to-use-ground-truth-text-label-data)
[59](https://os-world.github.io)
[60](https://kodexolabs.com/reactive-vs-proactive-ai-agents/)
[61](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06219.pdf)
[62](https://milvus.io/ai-quick-reference/what-is-the-difference-between-reactive-and-proactive-ai-agents)
[63](https://www.themoonlight.io/en/review/towards-visual-text-grounding-of-multimodal-large-language-model)
[64](https://openreview.net/forum?id=JLEneHy8qC)
[65](https://www.geeks.ltd/insights/articles/difference-between-reactive-and-proactive-ai-agents)
[66](https://arxiv.org/html/2504.04974v2)
