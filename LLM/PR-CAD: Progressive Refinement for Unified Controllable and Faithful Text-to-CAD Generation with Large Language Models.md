# PR-CAD: Progressive Refinement for Unified Controllable and Faithful Text-to-CAD Generation with Large Language Models

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

PR-CAD는 기존 text-to-CAD 연구가 **생성(generation)과 편집(editing)을 분리된 작업으로 취급**하는 근본적 한계를 지적하고, 이를 **단일 에이전트 내에서 통합**하는 점진적 정제(Progressive Refinement) 패러다임을 제안합니다. 특히 전문가가 아닌 일반 사용자도 자연어(정성적·정량적 지시)로 CAD 모델을 반복적으로 생성·수정할 수 있는 실용적 프레임워크를 구현합니다.

### 주요 기여 3가지

| 기여 | 설명 |
|------|------|
| **통합 프레임워크** | 생성·편집을 단일 에이전트로 통합한 PR-CAD 제안 |
| **고품질 상호작용 데이터셋** | CAD 전 생애주기를 포괄, 정성·정량 지시 + 다중 편집 연산 포함 |
| **RL 강화 추론 방법** | SFT + SCoT + GRPO 기반 강화학습으로 기하학적 정확도·실행 가능성·의도 정렬 최적화 |

---

## 2. 해결 문제 · 제안 방법(수식) · 모델 구조 · 성능 · 한계

### 2.1 해결하고자 하는 문제

기존 text-to-CAD 시스템의 4가지 핵심 문제:

1. **생성-편집 분리**: 생성 후 수정이 필요하면 처음부터 재시작해야 함
2. **과도하게 기술적인 텍스트 입력 의존**: Text2CAD 기준 단일 S-E 작업 기술에 평균 100단어 필요
3. **비현실적 편집 데이터**: 기존 편집 데이터는 무작위 생성으로 실제 사용자 의도 반영 미흡 (추가·삭제 편중, 수치적 편집 희소)
4. **CAD 직렬화 표현 비호환성**: DSL, ST, GPL 간 변환 어려움

### 2.2 제안 방법 및 수식

#### ① 강화학습 보상 함수 (RL Reward)

총 보상 $R$은 4가지 구성요소의 합:

$$R = R_{\text{chamfer}} + R_{\text{format}} + R_{\text{exec}} + R_{\text{length}} \tag{1}$$

**Chamfer Distance 보상** (주요 밀집 보상, geometric fidelity):

$$R_{\text{chamfer}} = e^{-\alpha D_{\text{CD}}} \tag{2}$$

- $D_{\text{CD}}$: 생성 형상과 정답 간의 Chamfer Distance
- $\alpha$: 감쇠율 제어 하이퍼파라미터 (실험에서 $\alpha = 5.0$)
- $D_{\text{CD}} = 0$ (완벽 일치)일 때 $R_{\text{chamfer}} = 1$ (최대 보상)

**형식 보상** (희소 보상):

$$R_{\text{format}} = \begin{cases} 0 & \text{if format is correct} \\ -0.2 & \text{if format is incorrect} \end{cases} \tag{3}$$

**실행 가능성 보상**:

$$R_{\text{exec}} = \begin{cases} 0 & \text{if executable} \\ -0.1 & \text{if not executable} \end{cases} \tag{4}$$

**길이 패널티** (간결한 출력 장려):

$$R_{\text{length}} = -\beta \cdot L \tag{5}$$

- $L$: 생성된 출력 시퀀스의 길이
- $\beta > 0$: 패널티 스케일 파라미터 (실험에서 $\beta = 0.01$)

#### ② Structured Chain-of-Thought (SCoT)

SCoT는 CAD 설계 프로세스를 4단계로 분해합니다:

```
Step 1: Intent Understanding   → 현재 CAD 모델을 텍스트로 기술, 사용자 지시와 정렬
Step 2: Modeling Analysis      → <sketch>...</sketch> 등 특수 마커로 CAD 요소 관계 표현
Step 3: Parameter Computation  → 좌표계 회전, 스케치 평면 이동, 호 파라미터 계산
Step 4: Position Identification → 편집 대상 부위 특정 및 편집 시퀀스 출력
```

#### ③ SFT (Supervised Fine-Tuning)

크로스 엔트로피 손실 함수로 LLM이 텍스트 → CAD DSL 매핑을 학습:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t}, x) \tag{6}$$

- $x$: 입력 텍스트 지시
- $y_t$: $t$번째 CAD 토큰
- $\theta$: 모델 파라미터

### 2.3 모델 구조

```
[입력: 텍스트 지시 + 현재 CAD 모델]
          ↓
  [DeepSeek-R1-671B]  ← 1,000개 트리플릿으로 SCoT 생성
  Structured CoT 데이터셋 생성
          ↓
  [Qwen2.5-7B-Instruct] ← 기반 모델
          ↓
  [Stage 1: SFT]
  - LLaMA-Factory 프레임워크
  - LoRA rank=8, 전 레이어 적용
  - 시퀀스 최대 4096 토큰
  - 3 에폭, LR=1e-4, BF16
          ↓
  [Stage 2: RL (GRPO)]
  - veRL 프레임워크 사용
  - 보상: R_chamfer + R_format + R_exec + R_length
  - α=5.0, β=0.01
          ↓
[출력: CAD DSL 시퀀스 (LogoUp 3D)]
```

**CAD 직렬화 표현 선택 전략:**
- **생성 작업**: DSL (Domain-Specific Language, LogoUp 3D) → 자연어에 가깝고 인지 패턴과 유사
- **편집 작업 (SCoT 내부)**: ST (Structured Text) → 구조 태그로 정확한 위치 지정 가능
- **GPT (Python 기반)**: 학습 데이터 양은 많으나 CAD 도메인과 표현 공간 간극 큼 → 성능 저조

### 2.4 성능 향상

#### 정량적 비교 (Table 1)

| 방법 | 생성-IR↓ | 생성-CD↓ | 편집-IR↓ | 편집-VLM↑ |
|------|---------|---------|---------|----------|
| GPT-4o (zero-shot) | 74.22 | 133.52 | 27.76 | 61.01 |
| GPT-4o (few-shot) | 55.95 | 77.49 | 13.47 | 66.18 |
| Text2CAD | 0.97 | 27.68 | ✗ | ✗ |
| CAD-Editor | ✗ | ✗ | 5.77 | 69.52 |
| **PR-CAD (Ours)** | **0.62** | **5.87** | **1.71** | **77.83** |

> CD 값은 $\times 10^3$ 스케일

#### 사용자 상호작용 성능 (Table 2)

| 그룹 | 방법 | 성공률 | 평균 턴 | SUS 점수 |
|------|------|--------|---------|---------|
| 전문가 | 단일 생성 | 56.25% | 1 | 38.125 |
| 전문가 | **PR-CAD 다중 대화** | **100%** | 3.375 | **93.125** |
| 초보자 | 단일 생성 | 31.25% | 1 | 25.125 |
| 초보자 | **PR-CAD 다중 대화** | **81.25%** | 4.53 | **80.625** |

#### 절제 연구 (Table 3)

| 훈련 전략 | IR | Mean CD | VLM-Eval |
|----------|-----|---------|----------|
| Qwen2.5-7B (기본) | ✗ | ✗ | ✗ |
| w/o RL | 12.67 | 84.63 | 56.44 |
| w/o SCoT | 2.48 | 11.34 | 68.81 |
| t/o/o Generation | 9.45 | 42.07 | 60.56 |
| **PR-CAD (Ours)** | **1.18** | **8.37** | **70.84** |

### 2.5 한계

1. **복잡한 설계 처리 미흡**: 현재 프레임워크는 고복잡도 설계나 특화 도메인(예: 항공우주, 의료기기)에서 추가 정제 필요
2. **대규모 프로젝트 효율성**: 대규모 CAD 프로젝트에 대한 실시간 적용의 계산 비용 문제
3. **계획 능력의 한계**: 복잡한 다단계 편집 연산에 대한 장기 계획(planning) 역량이 부족
4. **데이터셋 범위**: DeepCAD 기반 데이터에 집중되어 있어 다양한 산업 도메인 CAD 스타일 일반화에 제한

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 GRPO 기반 강화학습의 일반화 기여

PR-CAD는 GRPO (Generalized Reward Policy Optimization)를 활용하여 SFT만으로는 달성하기 어려운 **분포 외(out-of-distribution) 일반화**를 추구합니다.

**일반화에 기여하는 RL 보상 구조:**

$$R_{\text{chamfer}} = e^{-\alpha D_{\text{CD}}}$$

이 지수 감쇠 형태의 보상은 Chamfer Distance가 클 때는 큰 개선 신호를, 작을 때도 미세한 개선을 지속적으로 보상하여 모델이 **분포 경계 근처에서도 최적화 신호를 잃지 않도록** 설계되어 있습니다.

### 3.2 다중 모달리티 상호 강화

절제 연구(Table 3)에서 가장 주목할 만한 일반화 관련 결과:

- **t/o/o Generation** (생성 데이터만 훈련): IR=9.45, Mean CD=42.07
- **t/o/o Editing** (편집 데이터만 훈련): IR=6.94, Mean CD=20.05
- **PR-CAD (통합 훈련)**: IR=1.18, Mean CD=8.37

이는 **생성과 편집 작업이 서로의 일반화를 상호 강화**함을 실증합니다. 생성 작업 학습이 편집의 문맥 이해를 강화하고, 반대로 편집 학습이 생성의 부분 수정 정밀도를 향상시킵니다.

마찬가지로 정성·정량 모달리티 통합 훈련:
- **t/o/o Quantitative**: VLM-Eval=67.19
- **t/o/o Qualitative**: VLM-Eval=66.91
- **PR-CAD (통합)**: VLM-Eval=70.84

### 3.3 SCoT를 통한 구조적 추론의 일반화

SCoT는 복잡한 CAD 작업을 **명시적 추론 단계**로 분해하여, 학습 시 보지 못한 새로운 형상 조합에 대해서도:

1. **Intent Understanding** → 새로운 사용자 표현 해석 가능
2. **Parameter Computation** → 미학습 기하 변환 계산 일반화
3. **Position Identification** → 새로운 편집 위치 추론 가능

이는 Li et al.(2025a)의 코드 생성에서의 SCoT 연구와 일맥상통합니다.

### 3.4 RL의 강건성(Robustness) 부여 효과

부록 A.4에서 보고된 흥미로운 일반화 현상:

> "입력 명령이 잠재적 오류를 포함하거나 충돌을 유발할 가능성이 있을 때, 모델이 편집 의도를 넘어서 관련 부분을 자동으로 조정하여 오류 발생을 방지한다."

이는 $R_{\text{exec}}$ 보상이 모델로 하여금 **훈련 분포 외의 위험 상황에도 실행 가능한 출력을 생성**하도록 일반화된 안전 행동을 학습시켰음을 시사합니다.

### 3.5 스케일링 제거를 통한 수치적 일반화

이전 Transformer 기반 방법들은 파라미터를 고정 범위로 스케일링하였으나, PR-CAD는 **원시(raw) CAD 수치를 직접 사용**합니다. 이는:

$$\text{기존}: \hat{p} = \frac{p - p_{\min}}{p_{\max} - p_{\min}} \in [0, 1] \rightarrow \text{스케일링 오류 누적}$$

$$\text{PR-CAD}: \hat{p} = p \text{ (원시값 직접 사용)} \rightarrow \text{수치적 일반화 향상}$$

스케일링 제거로 학습 데이터와 다른 수치 범위의 실제 CAD 모델에 대한 일반화가 개선됩니다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4.1 연구 계보 및 흐름

```
DeepCAD (Wu et al., 2021, ICCV)
    ├── Text2CAD (Khan et al., 2024, NeurIPS)
    │       ├── Text-to-CADQuery (Xie & Ju, 2025)
    │       └── GeoCAD (Zhang et al., 2025)
    ├── FLEXCAD (Zhang et al., 2024b)
    │       └── CAD-Editor (Yuan et al., 2025)
    ├── CAD-GPT (Wang et al., 2025, AAAI)
    ├── CAD-LLaMA (Li et al., 2025b, CVPR)
    ├── Seek-CAD (Li et al., 2025c)
    └── PR-CAD (An et al., 2026) ← 본 논문
```

### 4.2 주요 방법론 비교 표

| 논문 | 기반 모델 | 생성 | 편집 | 정성 지시 | 정량 지시 | 반복 정제 | 표현 방식 |
|------|-----------|------|------|----------|----------|----------|----------|
| DeepCAD (2021) | Transformer | ✓ | ✗ | ✗ | ✗ | ✗ | DSL |
| Text2CAD (2024) | Transformer | ✓ | ✗ | ✓ | ✓ | ✗ | GPL |
| FLEXCAD (2024) | LLM (FT) | ✗ | ✓(랜덤) | ✓ | ✗ | ✗ | ST |
| CAD-Editor (2025) | LLM (FT) | ✗ | ✓(directed) | ✗ | ✓ | ✗ | ST |
| CAD-GPT (2025) | Multimodal LLM | ✓ | ✗ | ✓ | ✗ | ✗ | DSL |
| Seek-CAD (2025) | DeepSeek | ✓ | ✗(자기개선) | ✓ | ✓ | 제한적 | GPL |
| **PR-CAD (2026)** | **Qwen2.5-7B** | **✓** | **✓(directed)** | **✓** | **✓** | **✓** | **DSL+ST** |

### 4.3 핵심 방법론 차이

#### DeepCAD (Wu et al., 2021)
- **특징**: 트랜스포머 기반 생성 모델, 대규모 CAD 데이터셋 최초 제공
- **한계**: 텍스트 입력 불가, 고정 파라미터 범위 스케일링 필요
- **PR-CAD와의 관계**: 데이터셋 기반으로 활용, 스케일링 문제 해결

#### Text2CAD (Khan et al., 2024, NeurIPS)
- **특징**: 초보자~전문가 수준의 텍스트 프롬프트로 순차적 CAD 생성
- **한계**: 편집 불가, 평균 100단어의 기술적 지시 필요, 스케일링 의존
- **PR-CAD 대비**: Mean CD 27.68 vs **5.87** (PR-CAD 약 4.7배 향상)

#### FLEXCAD (Zhang et al., 2024b)
- **특징**: 미세 조정된 LLM으로 무작위 CAD 편집 수행, ST 표현 사용
- **한계**: 생성 불가, 무작위 편집(랜덤 생성 데이터), 정량 편집 희소
- **PR-CAD 대비**: 질적 편집 VLM-Eval 64.38 vs **77.83** (약 20.8% 향상)

#### CAD-Editor (Yuan et al., 2025)
- **특징**: Locate-then-infill 프레임워크, 자동 훈련 데이터 합성
- **한계**: 생성 불가, 데이터 구성상 추가·삭제 위주, 정량 편집 제한적
- **PR-CAD 대비**: 편집 Mean CD 8.85 vs **6.30** (약 28.8% 향상)

#### CAD-GPT (Wang et al., 2025, AAAI)
- **특징**: 공간 추론 강화된 멀티모달 LLM, CAD 시퀀스 합성
- **한계**: 편집 기능 없음, 반복 정제 패러다임 미지원

#### Seek-CAD (Li et al., 2025c)
- **특징**: DeepSeek 기반 자기 개선(self-refined) 메커니즘, 피드백 기반 반복 정제
- **차이**: 자기 개선은 모델 내부 피드백 루프; PR-CAD는 사용자 지시 기반 외부 정제

### 4.4 패러다임 전환 관점

| 세대 | 특징 | 대표 연구 |
|------|------|-----------|
| **1세대** (2021-2022) | Transformer + 고정 표현, 코드-CAD 매핑 | DeepCAD |
| **2세대** (2023-2024) | LLM 활용, 텍스트 입력 가능화 | Text2CAD, FLEXCAD |
| **3세대** (2025-2026) | 통합 생성-편집, 반복 정제, RL 최적화 | CAD-Editor, **PR-CAD** |

PR-CAD는 3세대 패러다임의 핵심을 집약하며, 특히 **생성-편집 통합 + 다중 모달리티 + RL 최적화**를 동시에 달성한 첫 번째 연구입니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5.1 연구에 미치는 영향

#### 패러다임 측면
- **통합 에이전트 설계의 표준화**: 생성-편집 분리라는 기존 관행을 탈피하여, 향후 CAD AI 연구에서 **통합 워크플로우 설계가 기본 요건**으로 자리잡을 가능성
- **대화형 CAD 시스템 연구 활성화**: ChatCAD 데모는 CAD 도메인에서의 **LLM 기반 대화형 인터페이스** 연구의 새 방향을 제시
- **보상 함수 설계 참조점**: $R = R_{\text{chamfer}} + R_{\text{format}} + R_{\text{exec}} + R_{\text{length}}$ 구조는 구조화된 코드/CAD 생성을 위한 RL 보상 설계의 참조 템플릿으로 활용 가능

#### 기술 측면
- **SCoT + RL 조합의 확산**: 구조화된 추론 체인과 강화학습의 결합이 CAD 외 다른 구조적 생성 작업(예: 회로 설계, 분자 설계)으로 확장 가능
- **멀티모달 데이터 파이프라인**: 9-view 렌더링 + VLM을 통한 자동 지시문 생성 방법론이 다른 3D 생성 도메인에 적용 가능
- **VLM-Eval 메트릭**: 텍스트-3D 정렬 평가에서 VLM 기반 자동 평가가 인간 평가를 대체하는 표준으로 부상 가능성

### 5.2 향후 연구 시 고려할 점

#### 데이터 및 도메인 확장

```
현재: DeepCAD (기계 부품 위주, 상대적으로 단순한 기하학)
      ↓
필요: 전자/의료/건축 특화 도메인 CAD 데이터
      복잡한 어셈블리(Assembly) 수준 모델
      파라메트릭 제약 관계(Constraint) 포함 데이터
```

#### 모델 확장성 (Scalability) 문제

- 현재 7B 파라미터 모델 기준이므로, 더 큰 모델 (70B+)로 스케일 업 시 LoRA 전략 재검토 필요
- 실시간 산업 적용을 위한 **추론 속도 최적화** (양자화, 증류 등) 연구 필요

$$\text{추론 비용} \propto O(L^2) \quad (\text{어텐션 연산, } L: \text{시퀀스 길이})$$

CAD 시퀀스의 특성상 $L$이 크므로 효율적 어텐션 메커니즘 도입 고려

#### 보상 함수 정교화

현재 보상 구조의 개선 방향:

$$R_{\text{current}} = R_{\text{chamfer}} + R_{\text{format}} + R_{\text{exec}} + R_{\text{length}}$$

- **제약 만족 보상** 추가: CAD 설계 원칙(교차 없음, 닫힌 스케치 등) 위반 시 패널티
- **의미적 보상**: VLM 기반 의도 정렬 보상을 RL 루프에 직접 통합

$$R_{\text{extended}} = R_{\text{chamfer}} + R_{\text{format}} + R_{\text{exec}} + R_{\text{length}} + R_{\text{constraint}} + R_{\text{semantic}}$$

- **단계별(step-level) 밀집 보상**: 현재는 에피소드 종료 시 계산되는 방식이므로, 중간 단계 보상으로 학습 안정성 향상 가능

#### 멀티모달 입력 확장

- 현재 텍스트만 지원 → **스케치 이미지**, **3D 점군(Point Cloud)**, **음성** 입력 통합
- 기존 CAD 파일(.STEP, .IGES) 입력으로부터 편집 지시를 받는 **리버스 엔지니어링** 통합

#### 일반화 성능 향상을 위한 추가 연구 방향

1. **도메인 적응 (Domain Adaptation)**: 소수의 도메인별 데이터로 빠른 적응을 위한 메타러닝 접근법
2. **데이터 증강 (Data Augmentation)**: CAD 모델의 기하학적 변환(회전, 미러링 등)을 통한 훈련 데이터 다양성 증가
3. **자기 일관성 (Self-Consistency)**: 여러 출력 후보 중 가장 일관된 CAD 시퀀스 선택하는 앙상블 전략
4. **외부 CAD 엔진 피드백 루프**: STEP/IGES 파일 생성 → 물리 시뮬레이션 → 결과 피드백을 RL 보상에 통합

#### 평가 기준의 다양화

현재 주요 메트릭인 Chamfer Distance의 한계:

$$D_{\text{CD}}(S, S') = \frac{1}{|S|}\sum_{x \in S} \min_{y \in S'} \|x-y\|_2^2 + \frac{1}{|S'|}\sum_{y \in S'} \min_{x \in S} \|x-y\|_2^2$$

- **형상 의미론적 유사도** 미반영 (예: 기능은 동일하나 기하학적으로 다른 설계)
- **제조 가능성(Manufacturability)** 평가 지표 추가 필요
- **인체 공학적(Human-centered) 설계 품질** 평가 체계 구축 필요

---

## 참고 자료

본 답변은 다음 자료를 직접 참조하였습니다:

1. **An, J. et al. (2026).** "PR-CAD: Progressive Refinement for Unified Controllable and Faithful Text-to-CAD Generation with Large Language Models." *arXiv preprint arXiv:2604.19773v1*
2. **Wu, R., Xiao, C., & Zheng, C. (2021).** "DeepCAD: A Deep Generative Network for Computer-Aided Design Models." *ICCV 2021*, pp. 6772–6782.
3. **Khan, M. S. et al. (2024).** "Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts." *NeurIPS 2024*, Vol. 37, pp. 7552–7579.
4. **Zhang, Z. et al. (2024b).** "FlexCAD: Unified and Versatile Controllable CAD Generation with Fine-Tuned Large Language Models." *arXiv:2411.05823*
5. **Yuan, Y. et al. (2025).** "CAD-Editor: A Locate-then-Infill Framework with Automated Training Data Synthesis for Text-Based CAD Editing." *arXiv:2502.03997*
6. **Wang, S. et al. (2025).** "CAD-GPT: Synthesising CAD Construction Sequence with Spatial Reasoning-Enhanced Multimodal LLMs." *AAAI 2025*, pp. 7880–7888.
7. **Xie, H. & Ju, F. (2025).** "Text-to-CADQuery: A New Paradigm for CAD Generation with Scalable Large Model Capabilities." *arXiv:2505.06507*
8. **Zhang, Z. et al. (2025).** "GeoCAD: Local Geometry-Controllable CAD Generation." *arXiv:2506.10337*
9. **Li, X. et al. (2025c).** "Seek-CAD: A Self-Refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek." *arXiv:2505.17702*
10. **Li, J. et al. (2025b).** "CAD-LLaMA: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation." *CVPR 2025*, pp. 18563–18573.
11. **Guo, D. et al. (2025).** "DeepSeek-R1 Incentivizes Reasoning in LLMs through Reinforcement Learning." *Nature*, 645(8081), pp. 633–638.
12. **Hu, E. J. et al. (2022).** "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*
13. **Li, J. et al. (2025a).** "Structured Chain-of-Thought Prompting for Code Generation." *ACM TOSEM*, 34(2), pp. 1–23.
14. **Sheng, G. et al. (2024).** "HybridFlow: A Flexible and Efficient RLHF Framework." *arXiv:2409.19256*
15. **Vaswani, A. et al. (2017).** "Attention Is All You Need." *NeurIPS 2017*, Vol. 30.

> **주의**: 본 분석은 제공된 PDF 원문(arXiv:2604.19773v1)에 기반하였으며, 논문 자체가 arXiv 프리프린트 상태(2026년 3월 27일 제출)임을 감안하여야 합니다. 실험 결과 수치는 논문 본문의 Table 1, 2, 3에서 직접 인용하였습니다.
