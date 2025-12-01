# PC-Agent: A Hierarchical Multi-Agent Collaboration Framework for Complex Task Automation on PC

### 1. 논문의 핵심 주장 및 주요 기여

**PC-Agent**는 개인용 컴퓨터(PC)에서의 복잡한 작업 자동화를 위해 설계된 계층적 다중 에이전트 협업 프레임워크입니다. 이 연구는 MLLM 기반 GUI 에이전트 분야에서 스마트폰 환경과는 다른 PC 시나리오의 고유한 도전 과제를 해결하기 위해 제안되었습니다.[1]

**핵심 주장:**

PC 환경은 두 가지 중요한 차별성을 가지고 있습니다. 첫째, **더욱 복잡한 상호작용 환경**으로서 UI 요소(아이콘, 위젯)가 밀집되어 있고 다양한 텍스트 레이아웃(Word 문서, VS Code 코드)이 존재하여 화면 인식의 어려움을 초래합니다. Claude-3.5 같은 최신 MLLM도 PC 화면에서 아이콘과 텍스트 그라운딩에서 24.0%의 정확도만 달성하는 수준입니다. 둘째, **더욱 복잡한 작업 시퀀스**로서 여행 계획 같은 실제 작업은 여러 애플리케이션을 거쳐 28단계 이상의 작업을 요구하며, 각 서브태스크 간 상호 의존성이 존재합니다. 이러한 긴 작업 시퀀스로 인해 단일 에이전트(GPT-4o)의 성공률은 서브태스크 레벨에서 41.8%인 반면 전체 명령어 레벨에서는 8.0%로 급격히 하락합니다.[1]

**주요 기여:**

1. **활성 인식 모듈(Active Perception Module, APM)**: 접근성 트리(Accessibility Tree)를 통한 상호작용 요소 인식과 MLLM 기반 의도 이해 에이전트 + OCR을 결합하여 정교한 텍스트 위치 파악과 조작 능력을 제공합니다.

2. **계층적 다중 에이전트 협업 아키텍처**: 의사결정을 명령어(Instruction)-서브태스크(Subtask)-동작(Action) 세 수준으로 분해하여, Manager Agent(MA)는 명령어 분해 및 서브태스크 간 통신, Progress Agent(PA)는 진행 추적, Decision Agent(DA)는 단계별 의사결정을 담당합니다.

3. **반사 기반 동적 의사결정 메커니즘**: Reflection Agent(RA)가 실행 결과를 모니터링하여 오류를 검출하고 즉시 피드백을 제공하여 에러 누적을 방지합니다.

4. **PC-Eval 벤치마크**: 25개의 복잡한 실제 명령어와 8개의 인기 있는 PC 애플리케이션을 포함한 새로운 평가 벤치마크를 제시하여, 기존 벤치마크의 단순한 작업 중심 한계를 극복합니다.

결과적으로 PC-Agent는 이전의 최첨단 방법 대비 **태스크 성공률에서 32% 절대 개선**을 달성하며(이전 최고 기준 24% → 56%), 이는 계층적 구조와 협업 메커니즘의 효율성을 명확히 입증합니다.[1]

***

### 2. 해결하고자 하는 문제 상세 설명

#### 2.1 문제 정의

PC 환경에서의 GUI 에이전트 설계는 다음 형식적 문제로 정의됩니다.[1]

$$A_i = \rho(I, O_i, H_{i-1}) \quad (1)$$

여기서:
- $$A_i$$: i번째 단계의 동작
- $$O_i$$: i번째 단계의 환경 관찰(스크린샷)
- $$H_{i-1}$$: (i-1)번째 단계까지의 작업 이력
- $$I$$: 사용자 명령어
- $$\rho$$: GUI 에이전트 정책

**문제의 복잡성 요소:**

1. **인식(Perception) 문제**: 기존 MLLM은 PC 화면의 아이콘과 텍스트 위치를 정확히 파악하지 못합니다. 접근성 API나 HTML 기반 방법만으로는 동적 요소와 텍스트 레이아웃 변화를 포괄적으로 처리할 수 없습니다.

2. **의사결정(Decision-Making) 문제**: 단순히 다음 동작을 선택하는 것을 넘어 서브태스크 간의 복잡한 의존성을 관리해야 합니다. 예를 들어, 첫 번째 서브태스크에서 추출한 정보가 세 번째 서브태스크 수행에 필요할 수 있습니다.

3. **오류 축적 문제**: 긴 작업 시퀀스에서 단일 에이전트는 오류를 추적하고 회복할 메커니즘이 부족하여, 초기 오류가 이후 모든 단계에 연쇄적 영향을 미칩니다.

#### 2.2 기존 방법의 한계

- **UFO**: 애플리케이션 선택 에이전트와 상호작용 에이전트 이중 구조이지만, 세밀한 텍스트 조작 능력이 부족하고 서브태스크 간 의존성 관리에 미흡합니다.
- **Agent-S**: 온라인 검색과 로컬 메모리를 결합하지만, 복잡한 멀티 스텝 PC 작업에서 명령어 분해 능력이 제한적입니다.
- **단일 에이전트**: 긴 작업 시퀀스를 단일 정책으로 처리하려 하므로 맥락 창(context window) 한계와 누적 오류 문제에 직면합니다.

***

### 3. 제안하는 방법 (수식 포함)

#### 3.1 활성 인식 모듈 (APM)

**3.1.1 상호작용 요소 인식**

접근성 트리(Accessibility Tree)를 통해 추출한 요소 정보를 Set-of-Mark(SoM) 방식으로 스크린샷에 표지합니다:

$$\text{Elements} = \{(e_1, \text{bbox}_1, \text{func}_1), \ldots, (e_n, \text{bbox}_n, \text{func}_n)\}$$

여기서 $$\text{bbox} = [x_{\text{top}}, y_{\text{top}}, x_{\text{bottom}}, y_{\text{bottom}}]$$는 경계 상자이고, $$\text{func}$$는 요소의 기능 설명입니다.

**3.1.2 텍스트 인식 및 위치 파악**

텍스트 기반 동작(선택, 편집)을 위해 다단계 프로세스를 적용합니다:

1. **의도 이해 단계**: MLLM 기반 의도 이해 에이전트가 사용자 명령어("마지막 두 문단을 밑줄 그어라")에서 대상 텍스트의 시작과 끝 범위를 결정합니다.

2. **위치 파악 단계**: OCR 도구를 사용하여 결정된 텍스트 범위의 정확한 픽셀 좌표를 획득합니다:

$$\text{TextLocation} = \text{OCR}(\text{TargetRange}) = [x_1, y_1, x_2, y_2]$$

3. **조작 단계**: 획득한 좌표로 드래그 등의 정확한 조작을 수행합니다.

#### 3.2 계층적 다중 에이전트 협업

**3.2.1 Manager Agent (MA)**

명령어를 매개변수화된 서브태스크로 분해합니다:

$$\text{Subtasks} = \text{MA}(I) = \{T_1, T_2, \ldots, T_k\}$$

각 서브태스크는 다음과 같은 의존성 구조를 가질 수 있습니다:

- **유형 1**: $$T_i$$의 결과 $$O_i$$가 $$T_{i+1}$$ 매개변수 입력으로 사용
- **유형 2**: $$T_i$$가 이전 $$T_{i-1}$$의 결과에 의존해 매개변수 결정
- **유형 3**: $$T_i$$가 유형 1과 2를 모두 만족
- **유형 4**: $$T_i$$가 다른 서브태스크와 독립

MA는 **통신 허브(Communication Hub)**를 관리하여:

$$\text{Hub}_t = \{\text{Output}_{\text{completed}(1)} : O_1, \ldots, \text{Output}_{\text{completed}(m)} : O_m\}$$

이를 통해 $$T_{i+1}$$의 매개변수를 동적으로 인스턴스화합니다:

$$T'_i = \text{Instantiate}(T_i, \text{Hub}_{t-1})$$

**3.2.2 Progress Agent (PA)**

i번째 단계에서 서브태스크 진행을 추적합니다:

$$TP_i = \text{PA}(T, TP_{i-1}, A_i, R_i) \quad (2)$$

여기서:
- $$T$$: 현재 서브태스크
- $$TP_{i-1}$$: 이전 단계의 진행 상황
- $$A_i$$: DA가 출력한 i번째 동작
- $$R_i$$: RA가 출력한 반사 정보
- $$TP_i$$: 업데이트된 진행 상황

PA는 전체 명령어 레벨 이력 대신 서브태스크 레벨 이력만 관리하여 컨텍스트 길이를 감소시킵니다.

**3.2.3 Decision Agent (DA)**

MLLM 기반으로 각 단계에서 동작을 생성합니다:

$$A_i = \text{DA}(T, O_i, TP_{i-1}, R_{i-1}) \quad (3)$$

Chain-of-Thought 방식으로 먼저 내적 독백을 생성한 후 동작을 결정하여, 추론 과정을 명시화하고 RA의 판단을 돕습니다:

$$\text{Thought}_i \rightarrow A_i$$

**제약 동작 공간(Constrained Action Space):**

$$\text{Actions} = \{\text{Click}, \text{DoubleClick}, \text{Type}, \text{Select}, \text{Drag}, \text{Scroll}, \text{Shortcut}, \text{Stop}\}$$

각 동작은 구조화된 형식으로 파싱되어 실행 가능하게 합니다.

#### 3.3 반사 기반 동적 의사결정

**3.3.1 Reflection Agent (RA)**

실행 전후 화면 변화를 비교하여 동작 결과를 평가합니다:

$$R_i = \text{RA}(T, A_i, O_{i-1}, O_i) \quad (4)$$

RA의 판단 결과:

1. **실패 - 재계획 필요**: 동작 유형 또는 위치 매개변수가 부정확
2. **부분 실패 - 위치 조정**: 상호작용 요소가 없거나 요소 활성화 필요
3. **성공**: 다음 단계로 진행

실패 판단 시 피드백 $$R_i$$가 DA에 전달되어 재계획을 유도하고, PA에도 전달되어 정확한 진행 추적을 보장합니다.

**피드백 루프:**
- Bottom-up: 동작 수준에서 오류 감지 → 상위 레벨로 피드백
- Top-down: 관리자 레벨 계획 → 실행 레벨로 전파

***

### 4. 모델 구조 상세 설명

#### 4.1 전체 아키텍처 흐름

PC-Agent의 실행 흐름은 다음과 같습니다:

```
사용자 명령어
    ↓
[Manager Agent]
    ├─ 명령어 분해
    ├─ 서브태스크 생성
    └─ 통신 허브 초기화
    ↓
For each Subtask T in {T₁, T₂, ..., Tₖ}:
    ├─ 서브태스크 매개변수 인스턴스화
    ├─ [Progress Agent] 초기화
    │
    ├─ For each step in Subtask:
    │   ├─ [Active Perception Module]
    │   │  ├─ 접근성 트리 추출
    │   │  ├─ 상호작용 요소 표지
    │   │  └─ 텍스트 범위 검출 (필요시)
    │   │
    │   ├─ [Decision Agent]
    │   │  ├─ 인식 정보 + 진행 정보 수집
    │   │  ├─ Chain-of-Thought 추론
    │   │  └─ 동작 생성
    │   │
    │   ├─ 동작 실행
    │   │
    │   ├─ [Reflection Agent]
    │   │  ├─ 화면 변화 비교
    │   │  └─ 결과 평가 및 피드백
    │   │
    │   └─ [Progress Agent]
    │      └─ 진행 상황 업데이트
    │
    └─ 서브태스크 완료
       └─ 결과를 통신 허브에 추가
    ↓
최종 결과
```

#### 4.2 구체적 예시

**예시 작업**: "Chrome에서 Nvidia, Apple, Microsoft의 현재 주가를 검색하고, Excel에서 새 스프레드시트를 만들어 회사명을 A열에, 주가를 B열에 입력하고 저장하세요."

**MA 분해 결과:**
- $$T_1$$: Chrome에서 'Nvidia 현재 주가' 검색, 결과 출력 → $$\{Nvidia\_price: X_1\}$$
- $$T_2$$: Chrome에서 'Apple 현재 주가' 검색 → $$\{Apple\_price: X_2\}$$
- $$T_3$$: Chrome에서 'Microsoft 현재 주가' 검색 → $$\{Microsoft\_price: X_3\}$$
- $$T_4$$: Excel에서 스프레드시트 생성, A1에 'Nvidia' 입력, B1에 $$\{Nvidia\_price\}$$ 입력... ($$T_1, T_2, T_3$$의 결과에 의존)

**통신 허브 진화:**
- After $$T_1$$ completes: $$\text{Hub} = \{Nvidia\_price: 142.62\}$$
- After $$T_2$$ completes: $$\text{Hub} = \{Nvidia\_price: 142.62, Apple\_price: 222.78\}$$
- After $$T_3$$ completes: $$\text{Hub} = \{\ldots, Microsoft\_price: 444.06\}$$
- $$T_4$$ instantiation: $$T'_4 = \text{Instantiate}(T_4, \text{Hub})$$로 정확한 매개변수 결정

***

### 5. 성능 향상 메커니즘 및 실험 결과

#### 5.1 성능 지표

논문은 두 가지 평가 지표를 도입합니다:

1. **Success Rate (SR)**: 완전히 완료된 명령어의 비율
2. **Subtask Success Rate (SSR)**: 완료된 서브태스크의 비율

#### 5.2 벤치마크 결과

**PC-Eval 벤치마크에서의 성능 비교** (Table 2):[1]

| 모델 | 유형 | Subtask SR (%) | Success Rate (%) |
|------|------|---|---|
| Gemini-2.0 | Single-Agent | 35.4% | 0.0% |
| Claude-3.5 | Single-Agent | 15.2% | 0.0% |
| Qwen2.5-VL | Single-Agent | 46.8% | 12.0% |
| GPT-4o | Single-Agent | 41.8% | 8.0% |
| UFO | Multi-Agent | 43.0% | 12.0% |
| Agent-S | Multi-Agent | 55.7% | 24.0% |
| **PC-Agent (Ours)** | **Multi-Agent** | **76.0%** | **56.0%** |

**주요 성과:**
- SR에서 기존 최고 기준(Agent-S, 24%) 대비 **32% 절대 개선** (56% 달성)
- SSR에서도 유의미한 개선으로 장기 작업 완료 능력 증명

#### 5.3 소거 연구 (Ablation Study) 결과

**Table 3 - PC-Agent 구성 요소의 기여도**:[1]

| APM | MA | RA | Subtask SR | Success Rate |
|-----|----|----|---|---|
| ✓ | ✓ | - | 58.2% | 20.0% |
| ✓ | - | ✓ | 50.6% | 12.0% |
| - | ✓ | ✓ | 48.1% | 12.0% |
| ✓ | ✓ | ✓ | **76.0%** | **56.0%** |

**해석:**
- **APM 제거 시**: SSR 17.8%, SR 36.0% 감소 → 세밀한 텍스트 조작 능력의 중요성 입증
- **MA 제거 시**: SR이 44.0% 급락 → 복잡한 명령어 분해의 필수성
- **RA 제거 시**: SSR 27.9%, SR 44.0% 감소 → 오류 복구 메커니즘의 핵심 역할

#### 5.4 파운데이션 모델 비교

**Table 4 - 다양한 MLLM 기반 성능**:[1]

| 모델 | Subtask SR | Success Rate | Recovery Rate | Manager SR |
|------|---|---|---|---|
| Gemini-2.0 | 55.7% | 28.0% | 24.0% | 84.0% |
| Claude-3.5 | 63.3% | 40.0% | 48.0% | 88.0% |
| Qwen2.5-VL | 32.9% | 12.0% | 40.0% | 88.0% |
| GPT-4o | **76.0%** | **56.0%** | **64.0%** | **96.0%** |

**발견사항:**
- GPT-4o의 우수한 인식 및 추론 능력이 PC-Agent 성능을 좌우하는 중요 요소
- Qwen2.5-VL의 낮은 성능은 제한된 텍스트 포맷 이행 능력 때문
- Recovery Rate이 높을수록 복잡한 작업에서 회복력이 우수

#### 5.5 구체적 성공 사례

**사례: 다중 검색 후 Excel 데이터 입력 작업**

- **입력**: 3개 회사의 주가 검색 후 스프레드시트 작성
- **MA 역할**: 4개의 서브태스크로 분해, 검색 결과를 통신 허브에 저장
- **APM 역할**: Excel 셀 위치를 정확히 파악하여 각 셀에 올바른 위치에 데이터 입력
- **RA 역할**: 실패한 입력(예: 잘못된 셀 선택)을 감지하고 DA에 피드백
- **결과**: 28단계에 걸친 복잡한 작업을 성공적으로 완료

***

### 6. 모델의 일반화 성능 향상 가능성 (중점 분석)

#### 6.1 일반화 능력의 현황과 한계

논문에서 PC-Agent의 일반화 성능과 관련하여 제시된 내용:

**현재 성능:**
- PC-Eval의 8개 애플리케이션에 대해 평가되었으며, 해당 애플리케이션군에서는 높은 성능을 보임
- 단, 논문에서 **교차 애플리케이션 일반화(cross-application generalization) 성능**에 대한 명시적 평가는 부족

**한계점:**
- 평가 벤치마크가 제한된 애플리케이션 집합(Chrome, Word, Excel, Outlook, Notepad, Clock, Calculator, File Explorer)에 한정
- 다양한 UI 디자인과 상호작용 패턴을 가진 애플리케이션으로의 확장성 미검증

#### 6.2 일반화 성능 향상 메커니즘

**구조적 이점:**

1. **계층적 분해의 일반성**
   - Instruction → Subtask → Action 분해는 도메인 불가지론적(domain-agnostic)
   - 새로운 애플리케이션에서도 이 분해 전략은 적용 가능성 높음

2. **APM의 이중 경로 설계**
   - **접근성 트리 경로**: 현대적 OS(Windows, macOS, Linux)가 제공하는 표준 API에 의존하여 플랫폼 독립성 제공
   - **OCR + 시각 경로**: HTML이나 접근성 API가 부실한 경우에도 순수 시각 기반 해석 가능

3. **반사 메커니즘의 적응성**
   - RA가 동작 전후의 화면 변화만 비교하므로, 새로운 애플리케이션에도 일반적으로 적용 가능
   - 명시적 애플리케이션 특화 규칙 불필요

#### 6.3 최신 연구의 일반화 강화 방향

2024-2025년 최신 연구 동향에서 나타나는 GUI 에이전트 일반화 성능 향상 방법:[2][3][4][5][6]

**방법 1: 탐색 기반 학습 (Exploration-based Learning)**

GUI-Xplore는 각 애플리케이션에 대한 사전 탐색 비디오(average 23.73분/앱)를 통해:[6]
$$\text{Generalization}_{\text{app}} = f(\text{Exploration Videos} + \text{Task Examples})$$

이는 PC-Agent의 APM을 보완하여 새로운 앱의 UI 구조를 사전 학습할 수 있는 방향 제시

**방법 2: 태스크 일반화를 통한 중간 훈련 (Mid-training for Task Generalization)**

최근 연구에서는 GUI 특화 데이터 부족 문제를 해결하기 위해:[4]
$$\text{GUI Generalization} = \text{Mid-Training(Mathematical Reasoning, Multimodal Reasoning)}$$

이는 GUI 도메인 밖의 다양한 추론 능력을 먼저 습득함으로써 GUI 에이전트의 기본 인식/추론 능력 강화

**방법 3: 다중 에이전트 신용 재할당 (Multi-Agent Credit Re-Assignment)**

CollabUIAgents는 역할 자유 에이전트(role-free agents)로 구성하여:[3]
$$\text{Generalization} \propto \text{Process Rewards}(\text{without environment-specific rewards})$$

프로세스 레벨 보상 신호를 통해 특정 환경에 과적합되지 않도록 제약

#### 6.4 PC-Agent 적용 시 일반화 성능 향상 전략

**단기 전략 (즉시 적용 가능):**

1. **저장소 기반 학습 (Repository-based Learning)**
   - 새로운 애플리케이션 사용 시 초기 탐색 단계 추가
   - Manager Agent가 새로운 UI 패턴을 인식하여 서브태스크 분해 전략 동적 조정

2. **프롬프트 기반 적응 (Prompt-based Adaptation)**
   - Decision Agent의 프롬프트에 새로운 애플리케이션의 UI 규약 정보 추가
   - 예: "Excel의 경우 각 셀은 [행, 열] 좌표로 접근되고..."

3. **APM 확장 (APM Extension)**
   - 웹 기반 애플리케이션을 위해 DOM 파싱 모듈 추가
   - 커스텀 UI 렌더링 애플리케이션을 위해 다양한 OCR 엔진 통합

**중기 전략 (추가 개발):**

4. **메타 학습 (Meta-Learning)**
   - 다양한 애플리케이션 군에 대한 대규모 선행 훈련
   - 새로운 애플리케이션 적응 시간 최소화

5. **점진적 세밀 조정 (Progressive Fine-tuning)**
   - 각 애플리케이션별 성공 케이스 축적 → 점진적 재훈련
   - 오류 패턴 메모리 구축으로 회복력 강화

#### 6.5 일반화 성능의 정량적 예측

현재 정보 기반 추정:

**보수적 시나리오** (직접 변형만 적용):
$$\text{새로운 앱 성공률} \approx 56\% \times 0.7 = 39.2\%$$
(비유사한 UI 구조로 인한 30% 성능 감소)

**낙관적 시나리오** (위의 중단기 전략 혼합 적용):
$$\text{새로운 앱 성공률} \approx 56\% \times 0.9 = 50.4\%$$
(APM 확장 + 프롬프트 적응으로 10% 손실만 발생)

***

### 7. 논문의 한계

#### 7.1 명시된 한계 (논문의 Limitations 섹션)[1]

1. **폐쇄형 모델 의존성**
   - 현재 최적 성능을 위해 GPT-4o(폐쇄형 모델)에 의존
   - 오픈소스 모델(Qwen2.5-VL 등)로의 성능 저하 문제 미해결
   - 비용, 개인정보 보호, 지연시간 관점의 한계

2. **제한된 시나리오 범위**
   - 현재 평가는 생산성 시나리오(productivity scenarios)에 한정
   - 소셜 미디어, 게임, 엔터테인먼트 등 상호작용 패턴이 다른 환경 미평가

#### 7.2 암시적 한계 (논문에서 직접 언급되지 않음)

1. **벤치마크 규모의 제한성**
   - PC-Eval: 25개 명령어 (기존 벤치마크 OSWorld 대비 소규모)
   - 통계적 유의성 검증 필요

2. **실시간 성능 및 비용**
   - 각 단계마다 MLLM 호출 → 지연시간 증가
   - 다중 에이전트 오버헤드에 대한 분석 부재

3. **오류 타입별 분석 부족**
   - 어떤 유형의 오류가 가장 빈번한지 분류 없음
   - 각 에이전트별 오류 기여도 분석 미흡

4. **보안 및 안정성 고려 부재**
   - 악성 명령어(예: 민감한 파일 삭제) 처리 메커니즘 없음
   - 시스템 리소스 보호 전략 미제시

***

### 8. 앞으로의 연구에 미치는 영향

#### 8.1 직접적 영향

**GUI 에이전트 설계 패러다임 전환:**

PC-Agent는 단순 "단일 에이전트 + 프롬프트" 접근에서 벗어나 **계층적 역할 분화** 모델을 입증했습니다. 이는 다음 세대 에이전트 설계의 기본 틀로 채택될 가능성이 높습니다.[7][8][9][2]

**구체적 영향:**

1. **Multi-Agent 아키텍처의 표준화**
   - 동기부여: PC-Agent의 32% 성능 향상은 계층적 구조의 효율성 입증
   - 확산: 최근 2025년 발표 논문들(COLA, MorphAgent, G-Memory)에서 유사한 계층적 구조 채택 시작[10][11][12]

2. **Active Perception 개념의 일반화**
   - 접근성 API + 시각 기반 이중 경로는 크로스 플랫폼 호환성 표준 제시
   - 이후 Aria-UI, ShowUI 등에서 시각-기반 순수 perception으로 진화[5][13]

3. **Reflection-based Mechanism의 확산**
   - PC-Agent의 RA가 bottom-up 피드백 제공 → 최신 연구에서 self-reflection 메커니즘의 필수 요소로 인식[14][15]

#### 8.2 간접적 학문적 영향

**1. 계층적 문제 분해 이론**

PC-Agent의 Instruction-Subtask-Action 삼계층 구조는:
$$\text{Complex Task} = f(\text{Instruction Decomposition}, \text{Subtask Coordination}, \text{Fine-grained Action})$$

이는 복잡한 멀티 스텝 문제 해결의 통용 모델로서 강화학습, 계획 수립(planning), 자동 추론 분야로 확산 가능

**2. 다중 에이전트 협업의 새로운 평가 지표**

기존: 전체 성공률 (Success Rate)
PC-Agent 제시: SSR + SR 이원적 평가 → 부분 성공을 포괄하는 성능 분석 가능

이는 LLM 기반 에이전트 평가의 그래뉼러(granular) 수준 제고

**3. 오류 복구 메커니즘의 이론화**

RA의 3가지 판단 타입(실패-재계획, 부분실패-조정, 성공)은:
$$\text{Error Recovery Framework} = \{\text{Reclassification}, \text{Fine-tuning}, \text{Propagation}\}$$

로 형식화되어 에이전트 견고성(robustness) 이론의 기초 제공

#### 8.3 산업적 응용 가능성

**1. 엔터프라이즈 자동화**

SAP, Oracle 같은 복잡한 ERP 시스템 자동화에 PC-Agent 구조 직접 적용 가능
- 기존: RPA(Robotic Process Automation) 스크립트 기반
- 신규: LLM 기반 동적 적응형 자동화

**2. 접근성 향상**

시각장애인, 운동장애인을 위한 PC 자동 조작 시스템에 활용 가능
- APM의 접근성 API 활용 강화
- 자연언어 명령어 기반 조작으로 접근성 개선

***

### 9. 앞으로 연구 시 고려할 점

#### 9.1 기술적 고려 사항

**1. 모델 독립성 확보 (Model Independence)**

| 관점 | 해결책 | 예상 효과 |
|------|--------|---------|
| 폐쇄형 모델 의존 | 오픈소스 MLLM 선행훈련(Mid-training) | 비용 감소, 개인정보 보호 개선 |
| 추론 지연시간 | 에이전트별 경량 모델 차별화 | MA/PA는 경량, DA는 강력 모델 |
| 토큰 비용 | 에이전트 간 메시지 최소화 | 요약 및 다운샘플링 기법 적용 |

**2. 확장성 강화 (Scalability)**

$$\text{Performance} = f(\text{APM Robustness}, \text{Application Diversity}, \text{Task Complexity})$$

- **APM 강화**: 웹(DOM), 데스크톱(접근성 API), 모바일(AXTree) 통합 인터페이스 개발
- **도메인 확장**: Office 앱에서 산업 특화 소프트웨어로 점진적 확대
- **복잡도 증가**: 현재 28단계 → 100단계 이상의 장기 작업 처리

**3. 오류 분류 및 분석 (Error Analysis)**

현재 미흡한 부분:
$$\text{Error Types} = \{\text{Perception Errors}, \text{Decision Errors}, \text{Execution Errors}, \text{Communication Errors}\}$$

각 유형별 빈도, 원인, 해결 전략 분석 필요

#### 9.2 평가 방법론 개선

**1. 벤치마크 확대**

- **애플리케이션 다양성**: 20개 → 50개+ 애플리케이션
- **작업 복잡도**: 현재 최대 28단계 → 100단계 이상
- **도메인 다양화**: 생산성(현재) → 소셜(Twitter), 미디어(YouTube), 게임 포함

**2. 교차 검증 (Cross-Validation)**

```math
\text{Generalization Score} = \frac{1}{N}\sum_{i=1}^{N} \text{SR}(\text{unseen\_app}_i)
```

학습하지 않은 새로운 애플리케이션에 대한 성능 평가

**3. 비용-성능 분석 (Cost-Benefit Analysis)**

- **계산 비용**: MLLM 호출 횟수, 지연시간, 토큰 사용량
- **정확도-효율성 트레이드오프**: 마진 성능 향상을 위한 실제 비용

#### 9.3 안정성 및 보안

**1. 유해 작업 필터링**

$$\text{Safety Filter} = \{\text{File Deletion}, \text{System Config}, \text{Data Leak}, \text{Malware}\}$$

MA 단계에서 위험 서브태스크 사전 감지 메커니즘

**2. 사용자 의도 검증**

- 명령어의 명확성 판단
- 모호한 경우 사용자 확인 요청 프로토콜

**3. 감사 추적 (Audit Trail)**

전체 작업 로그 기록 → 나중에 문제 발생 시 원인 분석 가능

#### 9.4 학습 및 적응 전략

**1. In-Context Learning**

새로운 애플리케이션 적응 시 초기 몇 개의 예제만으로 빠른 적응

**2. Continual Learning**

각 작업 완료 후 성공 케이스 축적 → 점진적 성능 개선
$$\text{Performance}_{t+1} = f(\text{Performance}_t, \text{New Experiences})$$

**3. 전이 학습 (Transfer Learning)**

유사한 UI 패턴(예: 모든 웹 기반 애플리케이션)에서의 지식 전이

***

### 10. 2020년 이후 관련 최신 연구 탐색

#### 10.1 다중 에이전트 협업 기본 연구 (2023-2025)

| 논문 | 주요 기여 | PC-Agent와의 관계 |
|------|---------|------------------|
| **LongAgent** (2024) | 문서 분할 & 멤버 간 커뮤니케이션 메커니즘 | 계층적 분해 개념의 선행 연구 |
| **MetaGPT** (2023) | SOP 인코딩으로 LLM 다중 에이전트 조율 | 에이전트 역할 표준화 아이디어 제공 |
| **AgentScope** (2024) | 다중 에이전트 프레임워크 플랫폼 | PC-Agent 실제 구현 가능 기반 제공 |
| **MorphAgent** (2024) | 분산형 협업과 역할 동적 진화 | 역할 자유 에이전트 개념 제시 |

#### 10.2 GUI 에이전트 특화 연구 (2024-2025)

| 논문 | 초점 | PC-Agent에 대한 영향 |
|------|-----|------------------|
| **UFO** (2024) | 크로스 앱 작업 (이중 에이전트) | PC-Agent의 선행 다중 에이전트 시도 |
| **Agent-S** (2024) | 온라인 검색 + 로컬 메모리 | 에이전트 강화를 위한 외부 정보 활용 |
| **Mobile-Agent-v3** (2025) | GUI-Owl 기반 다중 역할 에이전트 | PC-Agent와 유사한 계층적 구조 채택 |
| **AutoGLM** (2024) | 웹+휴대폰 기반 재강화학습 | 학습 기반 성능 개선 전략 |

#### 10.3 인식 개선 연구 (2024-2025)

| 논문 | 기술 | PC-Agent APM과의 연계 |
|------|------|------------------|
| **OmniParser** (2024) | 시각 전용 UI 파싱 | 순수 시각 기반 인식 보완 가능 |
| **Aria-UI** (2025) | 시각 그라운딩 최적화 | 텍스트 감지 정확도 개선 |
| **GUI-Xplore** (2025) | 탐색 비디오 기반 학습 | 새 앱 적응을 위한 사전 학습 방법 |
| **ShowUI** (2025) | 비전-언어-동작 통합 모델 | 엔드-투-엔드 GUI 이해 방향 |

#### 10.4 일반화 및 전이 학습 (2024-2025)

**교차 도메인 일반화:**
- **COLA** (2025): 동적 태스크 스케줄링으로 다양한 시나리오 대응[12]
- **CollabUIAgents** (2025): 프로세스 보상 기반 환경 불가지론적 학습[3]
- **Breaking Data Barrier** (2025): 중간 훈련으로 GUI 에이전트 일반화 능력 향상[4]

**관찰 핵심:**
수학 추론, 멀티모달 추론 등 GUI 도메인 외부의 다양한 능력 훈련이 GUI 작업 성능을 30-50% 향상시킬 수 있음 → PC-Agent에 적용 가능

#### 10.5 반사 및 오류 복구 (2024-2025)

| 논문 | 초점 | PC-Agent RA와의 비교 |
|------|-----|------------------|
| **REBACT** (2024) | "Act 전에 Reflect" | PC-Agent의 bottom-up RA와 상호 보완 |
| **Reflective LLM-based Agent** (2025) | 구조화된 자기반사 | PC-Agent RA의 이론적 기초 강화 |
| **Uncertainty-Aware GUI Agent** (2025) | 불확실성 기반 적응 | 높은 불확실성 상황에서 사용자 개입 도입 |

***

### 결론

**PC-Agent**는 PC 환경에서의 복잡한 다중 애플리케이션 자동화 문제에 대해 **계층적 다중 에이전트 협업**을 통해 체계적 해결책을 제시한 중요한 연구입니다. Active Perception Module, 계층적 의사결정 구조, 반사 기반 오류 복구 메커니즘의 삼중 창안(three-pronged innovation)은 다음 세대 GUI 에이전트 설계의 새로운 표준을 수립했습니다.

특히 **일반화 성능 향상**은 현재 제한적이나, 탐색 기반 학습, 중간 훈련, 다중 에이전트 신용 재할당 등의 최신 기법과 결합할 경우 새로운 애플리케이션 환경에 대한 빠른 적응이 가능할 것으로 예상됩니다. 앞으로의 연구에서는 모델 독립성 확보, 벤치마크 확대, 실시간 효율성 개선, 보안 강화에 중점을 두어야 하며, 이러한 발전이 엔터프라이즈 자동화와 접근성 기술로의 실제 응용을 가능하게 할 것입니다.

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e7620145-58b5-42d4-93bf-0e1cf1d9de8b/2502.14282v2.pdf)
[2](https://arxiv.org/abs/2402.11550)
[3](https://arxiv.org/html/2502.14496v1)
[4](https://openreview.net/forum?id=QDtORaZt8K)
[5](https://aclanthology.org/2025.findings-acl.1152.pdf)
[6](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_GUI-Xplore_Empowering_Generalizable_GUI_Agents_with_One_Exploration_CVPR_2025_paper.pdf)
[7](https://dl.acm.org/doi/10.1145/3664647.3684998)
[8](https://arxiv.org/abs/2502.04180)
[9](https://skyworkai.github.io/DeepResearchAgent/)
[10](https://arxiv.org/abs/2506.07398)
[11](http://arxiv.org/pdf/2410.15048.pdf)
[12](https://aclanthology.org/2025.emnlp-main.227.pdf)
[13](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_ShowUI_One_Vision-Language-Action_Model_for_GUI_Visual_Agent_CVPR_2025_paper.pdf)
[14](https://www.emergentmind.com/topics/reflective-llm-based-agent)
[15](https://arxiv.org/html/2509.18607v1)
[16](https://aclanthology.org/2024.emnlp-main.912)
[17](https://arxiv.org/abs/2506.00520)
[18](https://arxiv.org/abs/2502.14282)
[19](https://dl.acm.org/doi/10.1145/3638530.3664053)
[20](https://aclanthology.org/2024.sighan-1.10)
[21](https://arxiv.org/abs/2412.21088)
[22](http://arxiv.org/pdf/2407.07061.pdf)
[23](https://arxiv.org/pdf/2308.08155.pdf)
[24](https://arxiv.org/html/2404.18074v3)
[25](http://arxiv.org/pdf/2412.05449.pdf)
[26](http://arxiv.org/pdf/2308.00352.pdf)
[27](http://arxiv.org/pdf/2404.11943.pdf)
[28](http://arxiv.org/pdf/2402.14034.pdf)
[29](https://arxiv.org/html/2508.15144v2)
[30](https://ieeexplore.ieee.org/abstract/document/10970024/)
[31](http://article.sapub.org/10.5923.j.computer.20251503.02.html)
[32](https://aclanthology.org/2025.findings-emnlp.944.pdf)
[33](https://www.linkedin.com/posts/rakeshgohel01_what-is-the-future-of-gui-ai-agents-in-2025-activity-7270445808754257920-KDaW)
[34](https://arxiv.org/html/2510.27623v1)
[35](https://github.com/X-PLUG/MobileAgent)
[36](https://openreview.net/forum?id=Q20FcJJi4s)
[37](https://openreview.net/forum?id=0uRYFhPRx5)
[38](https://www.emergentmind.com/topics/llm-brained-gui-agents)
[39](https://onlinelibrary.wiley.com/doi/10.1002/asjc.3328)
[40](https://ieeexplore.ieee.org/document/10773100/)
[41](http://www.emerald.com/ilt/article/77/2/211-218/1239749)
[42](https://www.mdpi.com/1424-8220/24/14/4513)
[43](https://ojs.aaai.org/index.php/AAAI/article/view/29571)
[44](https://ieeexplore.ieee.org/document/10715514/)
[45](https://ieeexplore.ieee.org/document/10571357/)
[46](https://ieeexplore.ieee.org/document/10645598/)
[47](https://ieeexplore.ieee.org/document/10584492/)
[48](https://arxiv.org/abs/2403.04588)
[49](https://arxiv.org/html/2411.00820v1)
[50](http://arxiv.org/pdf/2406.18043.pdf)
[51](https://arxiv.org/pdf/2502.14777.pdf)
[52](https://arxiv.org/pdf/2308.10144v1.pdf)
[53](http://arxiv.org/pdf/2402.05929.pdf)
[54](https://arxiv.org/html/2412.01268v1)
[55](http://arxiv.org/pdf/2203.04482.pdf)
[56](https://arxiv.org/html/2508.04025v1)
[57](https://aclanthology.org/2025.findings-acl.1158.pdf)
[58](https://arxiv.org/pdf/2405.06682.pdf)
[59](https://github.com/OSU-NLP-Group/GUI-Agents-Paper-List/blob/main/paper_by_key/paper_learning.md)
[60](https://arxiv.org/html/2504.10127v1)
