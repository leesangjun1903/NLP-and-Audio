# DeepAnalyze: Agentic Large Language Models for Autonomous Data Science

### 1. 핵심 주장과 주요 기여

**DeepAnalyze-8B**는 자율 데이터 과학(Autonomous Data Science)을 위해 설계된 첫 번째 에이전틱 LLM으로, 원시 데이터 소스에서 분석가 수준의 심층 연구 보고서까지 전체 파이프라인을 자동으로 완성할 수 있습니다. 이 연구의 핵심 주장은 기존의 워크플로우 기반 데이터 에이전트들이 사전 정의된 워크플로우에 의존하기 때문에 완전한 자율 데이터 과학을 달성할 수 없다는 점을 지적합니다. DeepAnalyze의 주요 기여는 다음과 같습니다:[1]

1. **에이전틱 모델**: 자율 조율(autonomous orchestration)과 적응적 최적화(adaptive optimization)라는 두 가지 필수 능력을 갖춘 첫 번째 에이전틱 LLM[1]

2. **에이전틱 훈련 패러다임**: 보상 희소성과 궤적 부족 문제를 해결하기 위해 커리큘럼 기반 에이전틱 훈련과 데이터 기반 궤적 합성을 제안[1]

3. **강력한 성능**: 8B 파라미터만으로 12개의 벤치마크에서 대부분의 고급 독점 LLM을 능가하며, 개방형 데이터 연구를 수행하고 분석가 수준의 보고서를 생성할 수 있는 첫 번째 에이전틱 모델[1]

***

### 2. 문제 정의와 제안 방법

#### 2.1 해결하고자 하는 문제

기존 데이터 과학 접근법의 근본적인 한계는 두 가지입니다:[1]

1. **도메인 특화 LLM의 한계**: 데이터 과학 코드 생성, 표 형식 LLM, 데이터베이스 지향 LLM 등은 개별 작업에만 초점을 맞추고 있어 자율 조율 및 적응적 최적화 능력이 부족합니다.

2. **워크플로우 기반 에이전트의 한계**: 수동으로 설계된 휴리스틱과 도메인 특화 규칙에 의존하므로 완전한 자율성과 적응성을 달성할 수 없습니다.

#### 2.2 제안하는 방법: 아키텍처

DeepAnalyze는 5가지 특수 토큰으로 인코딩된 5가지 동작을 도입합니다:[1]

$$\langle \text{Analyze} \rangle \cdots \langle /\text{Analyze} \rangle$$: 계획, 추론, 반성을 포함한 텍스트 분석

$$\langle \text{Understand} \rangle \cdots \langle /\text{Understand} \rangle$$: 데이터베이스, 테이블, 문서 등 데이터 소스의 내용 이해

$$\langle \text{Code} \rangle \cdots \langle /\text{Code} \rangle$$: 환경의 데이터와 상호작용하기 위한 Python 코드 생성

$$\langle \text{Execute} \rangle \cdots \langle /\text{Execute} \rangle$$: 코드 실행 및 환경의 피드백 수집

$$\langle \text{Answer} \rangle \cdots \langle /\text{Answer} \rangle$$: 최종 출력 생성

**추론 과정**은 Algorithm 1으로 표현됩니다:[1]

```
Input: 지시문 Q, 환경 Env, DeepAnalyze 모델 M
Output: 응답 A (상호작용 과정 포함)

초기화: A = ∅
While ⟨Answer⟩···⟨/Answer⟩ not in A do
    y ← M(Q, A)  // 지시문 Q와 현재 응답 A에 기반해 다음 동작 생성
    A ← A + y
    if ⟨Code⟩···⟨/Code⟩ in y then
        code ← extract_code(y)
        feedback ← Env.execute(code)  // 환경의 데이터와 상호작용
        A ← A + ⟨Execute⟩ + feedback + ⟨/Execute⟩
    end if
end while
Return A
```

이 아키텍처는 모든 동작이 모델에 의해 자동으로 생성되므로 인간이 정의한 워크플로우나 규칙 없이 완전한 자율 조율과 최적화를 가능하게 합니다.[1]

#### 2.3 제안하는 방법: 커리큘럼 기반 에이전틱 훈련

데이터 과학 작업의 복잡성으로 인해 기초 LLM이 초기에 성공적으로 작업을 완료하기 어려워 **보상 희소성(reward sparsity)** 문제가 발생합니다. 이를 해결하기 위해 두 단계 훈련 패러다임을 제안합니다:[1]

**단계 1: 단일 능력 미세조정**

기초 LLM의 다양한 단일 능력을 강화합니다:
- 추론 능력 (⟨Analyze⟩에 대응)
- 구조화된 데이터 이해 (⟨Understand⟩에 대응)
- 코드 생성 (⟨Code⟩에 대응)

**단계 2: 다중 능력 에이전틱 훈련**

Group Relative Policy Optimization (GRPO)를 사용하여 다중 능력 통합을 학습합니다. 목적 함수는 다음과 같습니다:[1]

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim D, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}(o_i|q)}} A_i, \text{clip} \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}(o_i|q)}}, 1-\varepsilon, 1+\varepsilon \right) A_i \right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right) \right]$$

여기서:
- $A_i$는 각 그룹 내 출력의 보상으로부터 계산된 이점(advantage)
- $\pi_{\text{ref}}$는 참조 모델
- $\varepsilon$와 $\beta$는 하이퍼파라미터

#### 2.4 하이브리드 보상 모델

데이터 과학 작업의 개방형 특성을 고려하여 규칙 기반 보상과 LLM 판사 보상을 결합합니다.[1]

**데이터 질문 답변 및 데이터 중심 작업의 경우:**

$$R = \frac{1}{2}(1_{\text{acc}}(o) + S_{\text{interaction}}(o))$$

여기서 $1_{\text{acc}}(o) \in \{0, 1\}$는 결과의 정확성을 나타내고, $S_{\text{interaction}}(o) \in $은 상호작용 궤적의 품질을 평가합니다.[1]

**개방형 연구의 경우:**

$$R = \frac{1}{3}\left( S_{\text{report}}(o) + \min\left(\frac{|T|}{N_T}, 1\right) + \frac{1}{|T|} \sum_{T_i \in o} 1_{\text{success}}(T_i) \right)$$

여기서:
- $S_{\text{report}}(o)$는 유용성, 풍부함, 음향성, 해석 가능성, 가독성 등 5가지 측면에서 생성된 보고서를 평가합니다.
- $|T|$는 환경과의 상호작용 턴을 측정하고, $N_T = 10$은 하이퍼파라미터
- $1_{\text{success}}(T_i)$는 각 상호작용 턴의 성공 여부를 나타냅니다.

#### 2.5 데이터 기반 궤적 합성

궤적 부족 문제를 해결하기 위해 두 부분으로 구성된 프레임워크를 제안합니다:

**추론 궤적 합성**:
1. **증류**: 고급 LLM 교사 모델에서 추론 궤적 추출
2. **정제**: 핵심 추론 어휘를 삽입하여 구조화된 데이터에 대한 추론 강화

**상호작용 궤적 합성**:
다중 에이전트 시스템으로 생성:
- **질문자(Questioner)**: 데이터 소스를 관찰하고 데이터 과학 문제 생성
- **해결자(Solver)**: 5가지 동작을 사용하여 작업 완료
- **검수자(Inspector)**: 체크리스트에 대해 궤적 검증

***

### 3. 모델 구조

DeepAnalyze-8B는 **DeepSeek-R1-0528-Qwen3-8B**를 기초 LLM으로 하며, 다음과 같은 구조를 갖습니다:

#### 3.1 입력 형식

기존의 구조화된 데이터 LLM과 달리, DeepAnalyze는 두 가지 모드를 통합합니다:[1]

1. **수동 모드**: 입력에서 텍스트로 표현된 구조화된 데이터 수용
2. **능동 모드**: 사용자 입력에 따라 외부 데이터 소스를 적극적으로 검사

이는 인간 데이터 과학자의 행동을 모방하여 컨텍스트 길이 제한을 극복합니다.

#### 3.2 훈련 데이터: DataScience-Instruct-500K

총 약 500K 샘플로 구성되며:[1]

- **단일 능력 미세조정 단계**: 약 470K 샘플 (수열 길이: 8K)
- **다중 능력 에이전틱 훈련 단계**:
  - 콜드 스타트 단계: 20K 샘플
  - RL 단계: 15K 샘플
  - (수열 길이: 32K)

***

### 4. 성능 향상

#### 4.1 벤치마크 성능

**DataSciBench**에서의 성능:[1]

| 모델 | 점수 | 성공률(%) | 완성률(%) |
|------|------|---------|---------|
| GPT-4o | 66.31 | 68.44 | 3.91 |
| Claude-3-5-Sonnet | 47.48 | 58.11 | 2.14 |
| **DeepAnalyze-8B** | **59.91** | **66.24** | **2.86** |
| Qwen2.5-7B-Instruct | 43.83 | 50.74 | 1.43 |

**DSBench (데이터 분석)**에서 DeepAnalyze-8B는 30.04%의 정확도를 달성하여 이전의 워크플로우 기반 에이전트를 능가했습니다.[1]

**DABStep-Research (개방형 데이터 연구)**에서:[1]

| 작업 | DeepAnalyze-8B | DeepSeek-v3.1 | o3-mini |
|------|-----------------|----------------|---------|
| 데이터 준비 | 3.69 | 2.80 | 2.95 |
| 데이터 분석 | 3.75 | 3.55 | 3.05 |
| 데이터 인사이트 | 3.70 | 3.35 | 2.35 |
| 보고서 생성 | 3.67 | 3.50 | 2.60 |
| 개방형 데이터 연구 | 3.69 | 3.09 | 3.01 |

**TableQA 벤치마크에서의 평균 성능:**[1]

| 모델 | 평균 점수(%) |
|------|------------|
| GPT-4o | 58.96 |
| DeepSeek-R1-0528 | 60.22 |
| **DeepAnalyze-8B** | **64.47** |
| Reasoning-Table (SFT+RL) | 62.62 |

#### 4.2 성능 향상의 핵심 요인

**1. 커리큘럼 기반 훈련의 효과**:[1]

| 훈련 방법 | WikiTQ | MultiHiertt | DS-1000 | DABStep |
|---------|--------|------------|---------|---------|
| 커리큘럼 기반 | 83.24 | 48.29 | 61.70 | 38.88 |
| 단일 능력만 | 81.86 | 44.58 | 54.80 | 15.34 |
| 다중 능력만 | 80.32 | 43.29 | 53.20 | 30.66 |
| 단일 단계 훈련 | 82.13 | 46.23 | 54.80 | 36.89 |

결과는 단계적 훈련이 모든 복잡한 작업에서 우수한 성능을 제공함을 보여줍니다.

**2. ⟨Understand⟩ 동작의 효과**:[1]

| 설정 | WikiTQ | MultiHiertt | DS-1000 | DABStep |
|------|--------|------------|---------|---------|
| 전체 모델 | 83.24 | 48.29 | 61.70 | 38.88 |
| ⟨Understand⟩ 제거 | 80.78 | 45.43 | 61.20 | 31.78 |

구조화된 데이터 이해 동작의 명시적 도입이 성능을 크게 향상시킵니다.

**3. 추론 궤적 합성의 효과**:[1]

| 방법 | WikiTQ | HybridQA | MultiHiertt | HiTab |
|------|--------|----------|------------|-------|
| 원본 데이터 | 75.54 | 34.42 | 39.29 | 72.95 |
| + 증류 | 78.80 | 36.12 | 41.24 | 74.44 |
| + 증류 + 정제 | 80.25 | 38.84 | 43.47 | 75.86 |

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 현재 일반화 성능 분석

DeepAnalyze-8B의 일반화 성능은 여러 측면에서 평가할 수 있습니다:

**1. 다양한 데이터 포맷에 대한 일반화**:

DABStep 벤치마크에서 Markdown, CSV, JSON을 포함한 다양한 데이터 포맷에 대한 성능을 평가했습니다. DeepAnalyze-8B는 일반적인 수준의 작업(쉬운 수준)에서 70.83%의 성공률을 달성했고, 어려운 수준의 작업에서는 32.80%의 성공률을 기록하여, 복잡한 시나리오에서 워크플로우 기반 에이전트를 능가했습니다.[1]

**2. 개방형 연구에 대한 일반화**:

가장 중요한 평가는 DABStep-Research에서 개방형 데이터 연구 작업입니다. DeepAnalyze-8B는 명확한 단계 또는 목표가 없는 완전히 제약 없는 작업에서도 우수한 성능을 보였습니다. 이는 기존의 워크플로우 기반 시스템들이 개방형 작업에서 성능 저하를 보인 것과 대조됩니다.[1]

#### 5.2 일반화 성능 향상을 위한 핵심 메커니즘

**1. 커리큘럼 학습의 역할**

인간 데이터 과학자의 학습 궤적을 모방한 커리큘럼 기반 훈련은 다음과 같은 이유로 일반화를 향상시킵니다:[1]

- **점진적 능력 통합**: 단순 능력에서 복잡한 능력으로 점진적으로 진행하여 과적합을 방지
- **중간 단계 감독**: 각 단계에서의 중간 목표가 의미 있는 탐색 지도를 제공
- **보상 희소성 완화**: 초기 성공 경험이 훈련 안정성을 향상

**2. 환경 기반 학습**

실제 데이터 환경과의 상호작용을 통한 훈련:[1]

- 피드백 루프: 코드 실행 결과의 피드백을 통해 모델이 환경 역학을 학습
- 적응적 최적화: 환경 변화에 따라 전략을 동적으로 조정
- 분포 외 상황 대응: 훈련 중 마주하지 않은 시나리오에 더 잘 대응

#### 5.3 잠재적 일반화 성능 향상 방안

**1. 다중 환경에서의 메타 학습**

현재 연구에서는 특정 유형의 데이터 과학 작업으로 제한되어 있습니다. 향후 연구에서는:[2]

$$\mathcal{L}_{\text{meta}} = \mathbb{E}_{T \sim p(\mathcal{T})} \left[ \mathcal{L}_{\text{task}}(\theta - \alpha \nabla \mathcal{L}_{\text{support}}(\theta)) \right]$$

이런 형태의 메타 학습을 통해 다양한 데이터 과학 도메인에 대한 빠른 적응이 가능할 것입니다.

**2. 분포 외 일반화를 위한 인과 모델 학습**

최신 연구에 따르면 OOD(Out-of-Distribution) 일반화는 근사 인과 모델 학습이 필수적입니다. DeepAnalyze는 다음과 같이 개선될 수 있습니다:[3]

- 데이터 간의 인과 관계를 명시적으로 모델링
- 분포 이동에 강건한 특성 표현 학습

**3. 혼합 전문가 아키텍처 (Mixture of Experts)**

여러 전문화된 모듈의 동적 조합:

$$\text{output} = \sum_{i=1}^{K} g(x)_i \cdot m_i(x)$$

여기서 $g(x)$는 게이팅 네트워크, $m_i(x)$는 전문가 모듈입니다. 이는 다양한 데이터 과학 작업에 대한 일반화를 향상시킬 수 있습니다.

#### 5.4 일반화 성능의 한계

**1. 도메인 특화 지식의 부족**

DeepAnalyze는 일반적인 데이터 과학 작업에 최적화되었지만, 의료, 금융, 과학 분야 같은 특정 도메인의 전문 지식은 제한적입니다.[4]

**2. 시계열 및 공간 데이터의 처리**

현재 구현은 주로 표 형식 데이터와 일반적인 구조화된 데이터에 초점을 맞추고 있습니다. 시계열, 지리공간 데이터 같은 특수한 데이터 타입의 일반화는 향상이 필요합니다.[5]

**3. 실시간 피드백 및 동적 적응**

온라인 학습 환경에서의 성능은 아직 평가되지 않았습니다. WebRL 같은 최신 연구에서 보듯이, 온라인 환경에서의 지속적인 개선은 미래의 중요한 과제입니다.[6][7]

***

### 6. 모델의 한계

#### 6.1 기술적 한계

**1. 컨텍스트 길이 제약**

DeepAnalyze는 여전히 컨텍스트 길이 제약이 있어 대규모 데이터셋 전체를 처리하는 데 어려움이 있습니다.[1]

**2. 복잡한 멀티모달 작업**

현재는 주로 텍스트와 구조화된 데이터에 초점을 맞추고 있으며, 이미지, 음성, 비디오 등을 포함한 진정한 멀티모달 작업은 제한적입니다.[8]

**3. 계산 효율성**

32K 토큰 길이의 상호작용 궤적은 훈련과 추론에서 상당한 계산 비용을 초래합니다.

#### 6.2 데이터 관련 한계

**1. 훈련 데이터의 편향**

DataScience-Instruct-500K는 NL2SQL 데이터셋 (Spider, BIRD) 기반으로 생성되어 특정 유형의 데이터 과학 작업에 편향될 수 있습니다.[1]

**2. 합성 데이터의 한계**

상호작용 궤적 합성이 완벽하지 않아 일부 궤적이 실제 데이터 과학 프로세스를 정확히 반영하지 못할 수 있습니다.

#### 6.3 평가 관련 한계

**1. LLM 판사의 주관성**

개방형 연구 작업의 보상이 LLM 판사를 사용하는데, 이는 특정 LLM의 편향을 반영할 수 있습니다.[1]

**2. 실제 비즈니스 임팩트의 평가 부족**

대부분의 벤치마크는 학술적 성능 메트릭에 기반하며, 실제 비즈니스 의사결정에 미치는 영향은 평가되지 않았습니다.

***

### 7. 논문이 앞으로의 연구에 미치는 영향

#### 7.1 패러다임 전환: 워크플로우 기반 → 에이전틱 모델

DeepAnalyze는 데이터 과학 에이전트 연구의 명확한 패러다임 전환을 표시합니다:[1]

- **기존**: 인간이 정의한 규칙과 워크플로우에 기반한 에이전트
- **새로운**: 실시간 환경에서 학습하고 자율적으로 조율하는 에이전틱 LLM

이러한 전환은 다른 도메인 (과학 연구, 소프트웨어 엔지니어링 등)에서도 유사한 에이전틱 모델 개발로 이어질 것입니다.[9][10]

#### 7.2 에이전틱 훈련 방법론의 발전

DeepAnalyze의 커리큘럼 기반 에이전틱 훈련은 다음과 같은 후속 연구를 촉발했습니다:

**1. 환경 기반 훈련 (Environment Tuning)**

최근 연구는 단순히 모델을 미세조정하는 것이 아니라 훈련 환경 자체를 최적화하는 방향으로 진행 중입니다. 이는 데이터 효율성을 크게 향상시킵니다.[11]

**2. 자체 진화 커리큘럼**

WebRL 같은 최신 연구는 실패한 시도로부터 새로운 작업을 생성하여 커리큘럼을 동적으로 업데이트합니다.[7][6]

$$\mathcal{C}_{t+1} = \text{Curriculum}(\mathcal{C}_t, \{\text{Failed Tasks}\}_t)$$

#### 7.3 멀티모달 에이전틱 시스템의 발전

DeepAnalyze의 기초 위에서 멀티모달 능력이 통합되고 있습니다:[12][8]

- 이미지, 표, 코드를 동시에 처리하는 에이전트
- 공간 지능을 활용한 의사결정 지원 시스템
- 단백질 설계, 이온액체 발견 등 과학적 응용[12]

#### 7.4 일반화 성능 연구의 심화

DeepAnalyze는 다음과 같은 일반화 성능 연구를 촉발했습니다:

**1. 도구 사용의 일반화**

CoreThink Agentic Reasoner는 다양한 도구 호출 환경에서의 일반화를 분석하여, 구조화된 분해를 통해 530% 성능 향상을 달성했습니다.[13]

**2. 분포 외 일반화 메커니즘**

최신 연구는 세계 모델의 내부화를 통해 OOD 상황에서의 성능을 크게 향상시켰습니다.[14]

$$P(\text{success}) \propto \mathbb{E}_{s'}[\mathbb{1}_{\text{goal}}(s')]$$

#### 7.5 실무적 영향

DeepAnalyze의 오픈소스 공개는 다음과 같은 실무적 발전을 주도하고 있습니다:

**1. 엔터프라이즈 데이터 분석의 자동화**

Databricks, H2O.ai 같은 기업 플랫폼에서 에이전틱 분석 기능이 통합되고 있습니다.[15]

**2. 자율 과학 연구**

SR-Scientist 같은 후속 연구는 방정식 발견에 에이전틱 AI를 적용하여 기존 방법 대비 6-35% 성능 향상을 달성했습니다.[9]

**3. 규정 준수 자동화**

금융 서비스 분야에서 agentic AI + RAG 조합이 KYC/AML 규정 준수를 자동화하고 있습니다.[16]

***

### 8. 향후 연구시 고려할 점

#### 8.1 기술적 개선 방향

**1. 효율성 개선**

- **토큰 효율성**: 현재의 32K 토큰 상호작용 길이 감소
- **추론 속도 최적화**: 배포 환경에서의 실시간 처리
- **메모리 효율성**: 대규모 데이터셋 처리 능력 향상

**2. 능력 확장**

- **진정한 멀티모달 지원**: 이미지, 음성, 비디오 처리
- **시계열 분석 특화**: 시간 의존성을 고려한 특수화된 모듈
- **도메인 특화 버전**: 의료, 금융, 과학 분야 특화 모델

#### 8.2 일반화 성능 향상 전략

**1. 메타 학습 적용**

다양한 데이터 과학 도메인에 대한 빠른 적응을 위해 메타 학습을 통합해야 합니다. 이는 새로운 도메인에서의 샘플 효율성을 크게 향상시킬 것입니다.[17]

**2. 인과 추론 통합**

분포 외 일반화를 위해 명시적인 인과 모델링이 필요합니다. 이를 통해:[3]

$$\mathcal{L}_{\text{causal}} = \sum_{i} || \text{Effect}_i - \mathbb{E}_Z[\text{Do}(X=x_i)] ||^2$$

형태의 손실을 최소화하는 훈련이 가능합니다.

**3. 지속적 학습 (Continual Learning)**

온라인 환경에서 새로운 작업에 적응하면서 기존 지식을 보존하는 메커니즘이 필요합니다:[11]

$$\theta_t = \theta_{t-1} - \alpha \nabla \mathcal{L}_{\text{new}} - \beta \nabla ||\theta_t - \theta_{\text{old}}||^2$$

#### 8.3 평가 프레임워크 개선

**1. 실제 임팩트 메트릭**

학술적 벤치마크를 넘어 비즈니스 임팩트를 직접 측정하는 메트릭 개발이 필요합니다:
- ROI (투자 수익률)
- 의사결정 품질 개선도
- 시간 절감

**2. 롱 호라이즌 평가**

장기적 성능 저하(catastrophic forgetting)와 분포 시프트에 대한 평가가 필요합니다.

#### 8.4 안전성 및 신뢰성

**1. 해석 가능성 강화**

에이전틱 모델의 의사결정 과정을 더 잘 이해하기 위해:[18]

- 주의 메커니즘(Attention) 시각화
- 중요도 점수(Saliency) 분석
- 반사실적 설명(Counterfactual Explanations)

**2. 견고성 테스트**

- 적대적 예제에 대한 저항성
- 데이터 오염 시나리오
- 극한 상황 처리

#### 8.5 스케일링 전략

**1. 매개변수 효율적 미세조정**

더 작은 모델로도 강력한 성능을 달성하기 위해:
- LoRA (Low-Rank Adaptation)
- Prefix Tuning
- 어댑터 모듈

**2. 분산 훈련 및 추론**

멀티 에이전트 시스템으로의 확장을 위한 분산 아키텍처 개발:[19]

- 에이전트 간 통신 프로토콜 표준화
- 작업 분해 및 조율 메커니즘
- 결과 통합 및 충돌 해결

#### 8.6 최신 동향과의 통합

**1. 추론 능력 강화**

o1, o3 같은 최신 추론 모델의 기법을 통합하여 복잡한 데이터 과학 추론을 개선합니다:[20]

$$\text{reasoning quality} = f(\text{computation time}, \text{model capability})$$

**2. 검색 증강 생성 (RAG) 통합**

대규모 지식베이스 접근을 통해 일반화 성능을 향상시킵니다. 특히 동적 검색기 라우팅:[21]

$$P(\text{retriever}_i) \propto \text{utility gain}_i$$

**3. 이질적 데이터 처리**

멀티소스 데이터 발견 및 통합 능력 개선:[22]
- 블랙보드 아키텍처 활용
- 스키마 매칭 및 연결
- 데이터 품질 평가

***

### 9. 결론

**DeepAnalyze**는 자율 데이터 과학 분야에서 획기적인 진전을 나타냅니다. 커리큘럼 기반 에이전틱 훈련과 데이터 기반 궤적 합성이라는 두 가지 혁신적 기법을 통해, 8B 파라미터만으로도 대부분의 고급 독점 LLM을 능가하는 성능을 달성했습니다.[1]

**주요 성과**:
- 첫 번째 완전 자율 에이전틱 데이터 과학 모델
- 개방형 데이터 연구 능력 확보
- 분석가 수준의 보고서 생성 가능
- 12개 벤치마크에서 SOTA 성능

**일반화 성능**의 측면에서, DeepAnalyze는 환경 기반 학습과 커리큘럼 학습을 통해 상당한 일반화 능력을 보여주었으나, 도메인 특화 지식, 멀티모달 처리, 시계열 데이터 등에서는 개선의 여지가 있습니다.

**향후 연구의 방향**은 메타 학습, 인과 추론 통합, 지속적 학습, 그리고 멀티 에이전트 시스템으로의 확장 등으로 집약될 것으로 예상됩니다. 특히 온라인 학습 환경에서의 성능 개선과 실제 비즈니스 임팩트 측정이 중요한 향후 과제가 될 것입니다.[6][7][11][1]

이 논문은 LLM 기반 시스템이 단순한 정보 추출을 넘어 복잡한 다중 단계 작업의 자율적 수행으로 진화하고 있음을 명확히 보여주며, 이는 AI 에이전트 연구의 새로운 시대를 예고합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/78f84b7c-6bdb-479b-be5e-c83030f3ce50/2510.16872v1.pdf)
[2](https://milvus.io/ai-quick-reference/how-does-curriculum-learning-help-in-rl)
[3](https://www.linkedin.com/posts/hai-huang-ml_robust-agents-learn-causal-world-models-activity-7198124742359363584-veF2)
[4](http://arxiv.org/pdf/2410.11905.pdf)
[5](https://arxiv.org/html/2503.13868v3)
[6](https://proceedings.iclr.cc/paper_files/paper/2025/file/c66e1fcc9691aae706250638f36f681b-Paper-Conference.pdf)
[7](https://openreview.net/forum?id=oVKEAFjEqv)
[8](https://sparkco.ai/blog/exploring-agent-multimodal-capabilities-in-2025)
[9](https://arxiv.org/abs/2510.11661)
[10](https://www.semanticscholar.org/paper/7e981e3d5cab46803898f4535b4ab5acd69c316f)
[11](https://arxiv.org/abs/2510.10197)
[12](https://www.semanticscholar.org/paper/0aae6e36a113bcfe6c56e9c1945e1949a0535215)
[13](https://arxiv.org/abs/2510.22898)
[14](https://arxiv.org/abs/2510.15047)
[15](https://solutionsreview.com/business-intelligence/the-best-ai-agents-for-data-science/)
[16](https://journalijsra.com/node/1218)
[17](https://milvus.io/ai-quick-reference/how-do-ai-agents-leverage-transfer-learning)
[18](https://labs.adaline.ai/p/evaluating-ai-agents-in-2025)
[19](https://collabnix.com/multi-agent-and-multi-llm-architecture-complete-guide-for-2025/)
[20](https://arxiv.org/abs/2506.08379)
[21](https://arxiv.org/abs/2506.13743)
[22](https://arxiv.org/abs/2510.01285)
[23](https://arxiv.org/abs/2510.16872)
[24](https://www.semanticscholar.org/paper/c351338ab8cbf15499f39af121b5c20a896ab92b)
[25](https://www.irjmets.com/upload_newfiles/irjmets70800040152/paper_file/irjmets70800040152.pdf)
[26](https://arxiv.org/abs/2507.13729)
[27](https://dl.acm.org/doi/10.1145/3764924.3770899)
[28](https://dl.acm.org/doi/10.1145/3712285.3759887)
[29](https://arxiv.org/html/2503.07044v1)
[30](http://arxiv.org/pdf/2410.22457.pdf)
[31](https://arxiv.org/html/2502.05957)
[32](https://arxiv.org/pdf/2309.07870.pdf)
[33](https://arxiv.org/pdf/2503.16734.pdf)
[34](https://arxiv.org/html/2503.18102v1)
[35](https://arxiv.org/html/2502.06589v1)
[36](https://arxiv.org/pdf/2502.14499.pdf)
[37](https://datasciencedojo.com/blog/agentic-llm-in-2025/)
[38](https://openreview.net/forum?id=DSsSPr0RZJ)
[39](https://huggingface.co/papers/2510.16872)
[40](https://arxiv.org/abs/2409.07703)
[41](https://www.youtube.com/watch?v=Q1xBjjgXqms)
[42](https://arxiv.org/abs/2510.01135)
[43](https://metadesignsolutions.com/benchmarking-ai-agents-in-2025-top-tools-metrics-performance-testing-strategies/)
[44](https://www.semanticscholar.org/paper/d137c7ad3adb9b78a4f1caa4e02e7bc0e31c3e98)
[45](https://arxiv.org/abs/2506.10756)
[46](https://arxiv.org/abs/2507.21817)
[47](https://arxiv.org/abs/2510.06790)
[48](https://arxiv.org/pdf/2501.01702.pdf)
[49](https://arxiv.org/pdf/2403.12881.pdf)
[50](https://arxiv.org/pdf/2408.09955.pdf)
[51](https://arxiv.org/pdf/2404.04669.pdf)
[52](http://arxiv.org/pdf/2401.03428v1.pdf)
[53](http://arxiv.org/pdf/2406.14228.pdf)
[54](https://arxiv.org/pdf/2308.11339.pdf)
[55](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_On_the_Out-Of-Distribution_Generalization_of_Large_Multimodal_Models_CVPR_2025_paper.pdf)
[56](https://superagi.com/how-to-leverage-large-agentic-models-for-ai-powered-decision-making-a-step-by-step-guide/)
[57](https://ijcmi.in/index.php/ijcmi/article/view/55)
[58](https://orq.ai/blog/llm-agents)
[59](https://openreview.net/forum?id=eM8Db7ukSB)
