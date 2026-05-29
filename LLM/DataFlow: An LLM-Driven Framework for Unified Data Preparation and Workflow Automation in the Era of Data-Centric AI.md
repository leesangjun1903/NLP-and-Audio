
# DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI

> **논문 정보:** Hao Liang et al. (35명 공저), arXiv:2512.16676v1 [cs.LG], 2025년 12월 18일
> **출처:** [arxiv.org/abs/2512.16676](https://arxiv.org/abs/2512.16676), [huggingface.co/papers/2512.16676](https://huggingface.co/papers/2512.16676), [github.com/OpenDCAI/DataFlow](https://github.com/OpenDCAI/DataFlow), [papers.cool/arxiv/2512.16676v1](https://papers.cool/arxiv/2512.16676v1), [researchgate.net](https://www.researchgate.net/publication/398850920), [arxiv.org/html/2512.16676v1](https://arxiv.org/html/2512.16676v1), [arxiv.org/pdf/2512.16676](https://arxiv.org/pdf/2512.16676)

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

### 🎯 핵심 주장

LLM을 위한 고품질 데이터 수요가 급증하면서 확장 가능하고 신뢰할 수 있는 데이터 준비 파이프라인의 필요성이 커지고 있으나, 현재의 관행은 여전히 임시방편적 스크립트와 느슨하게 정의된 워크플로우에 의존하고 있어 원칙적 추상화가 부족하고 재현성을 저해하며 모델-인-더-루프 데이터 생성 지원이 제한적이다.

이러한 문제를 해결하기 위해 저자들은 DataFlow, 즉 통합적이고 확장 가능한 LLM 기반 데이터 준비 프레임워크를 제안하며, DataFlow는 모듈화·재사용·조합 가능한 데이터 변환을 가능하게 하는 시스템 수준 추상화와 디버깅·최적화 가능한 데이터 흐름을 구축하기 위한 PyTorch 스타일의 파이프라인 구성 API를 제공한다.

### ✅ 주요 기여 요약

핵심 기여는 다음과 같이 요약된다: (1) 조합 가능한 추상화와 LLM 우선 오퍼레이터 실행 모델로 구축된 통합 LLM 기반 데이터 준비 프레임워크, (2) 풍부하고 확장 가능한 오퍼레이터-파이프라인 생태계로서, DataFlow는 텍스트, 수학적 추론, 코드, Text-to-SQL, agentic RAG 데이터, 대규모 QA를 아우르는 약 200개의 재사용 가능한 오퍼레이터와 6개의 SOTA 템플릿 파이프라인을 제공한다.

또한 사용성을 더욱 높이기 위해 자연어 명세를 오퍼레이터 합성, 파이프라인 계획, 반복적 검증을 통해 실행 가능한 파이프라인으로 자동 변환하는 **DataFlow-Agent**를 도입한다.

---

## 2. 상세 분석: 문제 정의 · 제안 방법 · 모델 구조 · 성능 · 한계

### 2.1 해결하고자 하는 문제

통합되고 프로그래밍 가능한 패러다임의 부재로 파이프라인의 재현, 확장, 프로젝트 간 비교가 어렵다. 이 문제는 인스트럭션 튜닝, CoT 생성, 함수 호출과 같은 세밀한 포스트트레이닝 태스크로의 트렌드에 의해 더욱 심화되며, 이런 태스크에서는 데이터 준비의 의미론적 풍부성과 정확성이 정밀한 태스크 레벨 모델 행동을 달성하는 데 필수적이다.

이에 대응하여 NeMo Curator, Data-Juicer 등 여러 시스템이 LLM 데이터 큐레이션 표준화를 목표로 등장했지만, 이들은 여전히 구성 중심의 툴킷에 머물러 있다.

구체적으로, 기존 문제는 다음 세 가지로 정리된다:

| 문제 | 설명 |
|------|------|
| **비원칙적 추상화** | ad-hoc 스크립트, 재현 불가 |
| **모델-루프 미지원** | LLM 기반 데이터 생성 파이프라인 구성 불가 |
| **도메인 분절** | 텍스트/수학/코드/SQL 등 각 도메인별 독립 시스템 |

---

### 2.2 제안하는 방법 및 시스템 설계

#### 2.2.1 핵심 설계 철학

DataFlow가 요구하는 통합 프레임워크의 조건은 다음과 같다:

프레임워크는 (1) 모델-인-더-루프 생성과 의미론적 정제를 위한 세밀하고 조합 가능한 오퍼레이터 제공, (2) LLM 데이터 준비를 위한 도메인-불가지론적 오픈소스 프로토콜 역할을 하는 명시적이고 검증 가능한 파이프라인 정의 지원, (3) 다양한 LLM 엔진과 스토리지 백엔드를 통합하는 백엔드-불가지론적 구조 유지, (4) 에이전트 기반 자동 워크플로우 구성을 추가 지원하면서 모델·태스크·도메인에 걸쳐 원칙적인 워크플로우 구성, 재사용, 최적화 가능이어야 한다.

#### 2.2.2 오퍼레이터 추상화 (핵심 수식적 구조)

오퍼레이터의 `run()` 메서드는 파이프라인 내 실행 단위인 변환 로직을 구현한다. 오퍼레이터를 범용적이고 쉽게 조합할 수 있도록 `run()`은 `DataFlowStorage` 객체와 `input_*` 및 `output_*` 키 집합만 받는다. 키-값 쌍으로 해석할 때, `input_*` 키는 입력 필드로 읽을 스토리지 열을 나타내고, `output_*` 키는 처리된 데이터 항목 각각에 대해 쓸 새 열의 이름을 나타낸다.

이를 형식화하면:

$$\text{Operator: } f_i: \mathcal{S} \times \mathcal{K}_{in} \times \mathcal{K}_{out} \rightarrow \mathcal{S}'$$

여기서 $\mathcal{S}$는 `DataFlowStorage`, $\mathcal{K}\_{in}$은 입력 키 집합, $\mathcal{K}_{out}$은 출력 키 집합이다.

이 설계는 다양한 업스트림 데이터셋에 자연스럽게 적응하는 유연한 I/O 바인딩을 제공하며, 선언된 키들이 오퍼레이터 간 방향성 의존 그래프(DAG)를 형성하여 위상 정렬 스케줄링과 다운스트림 최적화 검사를 가능하게 한다. 구성에서 실행을 분리하고 상태 변경을 공유 스토리지에 대한 명시적 키 기반 읽기/쓰기 연산으로 제한함으로써 오퍼레이터 추상화는 경량적, 결정론적, 조합 용이하다.

파이프라인은 오퍼레이터들의 DAG 합성으로 표현된다:

$$\mathcal{P} = f_n \circ f_{n-1} \circ \cdots \circ f_1, \quad f_i \in \mathcal{O}$$

$$\mathcal{P}(\mathcal{D}) = f_n(f_{n-1}(\cdots f_1(\mathcal{D})\cdots))$$

여기서 $\mathcal{D}$는 원시 데이터셋, $\mathcal{O}$는 약 200개의 오퍼레이터 집합이다.

#### 2.2.3 설계 원칙

PyTorch에서 영감받은 IDE 친화적 프로그래밍 인터페이스로 최소 보일러플레이트로 복잡한 데이터 준비 파이프라인을 구축·디버깅할 수 있으며, `torch.nn.Module`과 유사한 모듈화 추상화를 따라 새 오퍼레이터와 알고리즘을 플러그앤플레이 컴포넌트로 추가하고 기존 워크플로우와 자연스럽게 조합할 수 있다.

DataFlow는 이질적인 데이터 준비 워크플로우를 표준화된 추상화 레이어로 통합하며, 일관성·재현성을 위한 표준화와 도메인 간 필요한 커스터마이제이션 사이의 균형을 맞추어 효율적인 파이프라인 재사용과 적응을 가능하게 한다.

---

### 2.3 모델 구조 (시스템 아키텍처)

DataFlow의 전체 시스템은 다음 세 레이어로 구성된다:

```
┌─────────────────────────────────────────────┐
│          DataFlow-Agent Layer               │
│  (자연어 → 파이프라인 자동 변환)              │
├─────────────────────────────────────────────┤
│         Pipeline Template Layer             │
│  Text | Math | Code | SQL | RAG | KnowExt  │
├─────────────────────────────────────────────┤
│          Operator Ecosystem (~200)          │
│  Filtering | Synthesis | Eval | Transform  │
└─────────────────────────────────────────────┘
```

**① Operator Layer (~200개 오퍼레이터)**

DataFlow는 약 200개의 재사용 가능한 텍스트 특화 오퍼레이터의 풍부한 라이브러리를 기반으로 구축되어 클리닝, 변환, 합성, 평가에 대한 세밀한 제어를 가능하게 한다.

오퍼레이터 유형은 크게 4가지로 분류된다:

$$\mathcal{O} = \mathcal{O}_{filter} \cup \mathcal{O}_{synth} \cup \mathcal{O}_{eval} \cup \mathcal{O}_{transform}$$

**② Pipeline Template Layer**

프레임워크는 텍스트, 수학적 추론, 코드, Text-to-SQL, agentic RAG, 대규모 지식 추출을 아우르는 약 200개의 재사용 가능한 오퍼레이터와 6개의 도메인 범용 파이프라인으로 구성된다.

**③ DataFlow-Agent**

DataFlow-Agent는 오퍼레이터 합성, 파이프라인 계획, 반복적 검증을 통해 자연어 명세를 실행 가능한 파이프라인으로 자동 변환한다.

이 과정을 수식으로 나타내면:

$$\text{DataFlow-Agent}: \mathcal{L}_{nl} \xrightarrow{\text{synthesis}} \mathcal{O}_{new} \xrightarrow{\text{planning}} \mathcal{P} \xrightarrow{\text{verification}} \mathcal{P}^{*}$$

**④ 서빙 레이어**

DataFlow의 서빙 모듈은 오픈소스 모델과 로컬 배포 모델 모두를 지원한다.

---

### 2.4 성능 향상

6가지 대표적 사용 사례에서 DataFlow는 지속적으로 다운스트림 LLM 성능을 향상시킨다. 수학, 코드, 텍스트 파이프라인은 큐레이션된 인간 데이터셋과 특화된 합성 기반선을 능가하며, Text-to-SQL에서 SynSQL 대비 최대 $+3\%$ 실행 정확도, 코드 벤치마크에서 평균 $+7\%$ 향상, MATH·GSM8K·AIME에서 1~3점 향상을 달성한다. 나아가 DataFlow가 생성한 통합 10K 샘플 데이터셋은 베이스 모델이 1M Infinity-Instruct 데이터로 학습한 대응 모델을 능가할 수 있게 한다.

수치 성능을 표로 정리하면:

| 벤치마크 | 향상 수치 | 비교 기준 |
|----------|-----------|-----------|
| **Text-to-SQL** | $+3\%$ 실행 정확도 | 2.5M 샘플 SynSQL |
| **Code 벤치마크** | $+7\%$ 평균 향상 | 공개 코드 인스트럭션 데이터 |
| **MATH / GSM8K / AIME** | $+1 \sim 3$ 점 | Open-R1, Synthetic-1 |
| **Instruct (통합)** | 1M → 10K 데이터 효율 | 1M Infinity-Instruct |

수학 추론 파이프라인의 성능 향상 메커니즘은 다음과 같이 정형화할 수 있다:

$$\text{Score}_{DataFlow} = \mathbb{E}_{x \sim \mathcal{D}^*}[\mathcal{M}(x)] > \mathbb{E}_{x \sim \mathcal{D}_{baseline}}[\mathcal{M}(x)]$$

여기서 $\mathcal{D}^*$는 DataFlow Reasoning Pipeline으로 생성·정제된 데이터셋이다.

DataFlow 합성 수학적 추론 데이터는 MATH, GSM8K, AIME에서 고품질 합성 기준선 대비 1~3점 향상을 제공하고, Text-to-SQL 파이프라인은 2.5M 샘플의 SynSQL 코퍼스보다 $+3\%$ 이상의 실행 정확도 향상을 10만 개 미만의 학습 예제로 달성한다. 또한 DataFlow 생성 텍스트, 수학, 코드 데이터를 통합 코퍼스(DataFlow-Instruct-10K)로 결합하면, 단 10K 샘플로 Qwen2-base와 Qwen2.5-base가 1M Infinity-Instruct 인스턴스로 학습한 모델을 능가하며 해당 Qwen-Instruct 모델 성능에 근접한다.

---

### 2.5 한계점

논문 및 관련 자료를 분석하면 다음과 같은 한계가 암시된다:

1. **멀티모달 지원 미흡:** Data-Juicer 2.0 버전은 텍스트, 이미지, 비디오, 오디오에 걸쳐 100개 이상의 오퍼레이터를 지원하는 반면, DataFlow는 주로 텍스트 기반 오퍼레이터에 집중되어 있어 멀티모달 데이터 처리 측면에서 상대적으로 제한적이다.

2. **LLM 의존성:** DataFlow-Agent와 오퍼레이터 다수가 외부 LLM 호출에 의존하여, LLM 품질 및 비용 변동에 민감하다.

3. **미검증 도메인:** 의료·법률·금융 등 DataFlow가 지원을 명시한 헬스케어, 금융, 법률, 학술 연구 등 특화 도메인에서의 체계적인 벤치마킹은 아직 충분히 보고되지 않았다.

4. **오퍼레이터 조합 최적화:** 약 200개의 오퍼레이터 중 최적 조합을 자동 탐색하는 체계적 방법론(NAS 수준)은 명시적으로 제시되지 않았다.

---

## 3. 모델의 일반화 성능 향상 가능성

DataFlow는 일반화 성능 향상에 직접적으로 기여하는 여러 메커니즘을 제공한다.

### 3.1 데이터 효율적 일반화 (Data-Efficient Generalization)

DataFlow가 생성한 통합 10K 샘플 데이터셋은 베이스 모델이 1M Infinity-Instruct 데이터로 학습된 대응 모델을 능가할 수 있게 한다.

이는 다음과 같은 일반화 효율 공식으로 표현할 수 있다:

$$\text{Generalization}(f_\theta, \mathcal{D}_{DataFlow}^{10K}) \geq \text{Generalization}(f_\theta, \mathcal{D}_{baseline}^{1M})$$

즉, 데이터 양보다 **데이터 품질과 다양성**이 일반화에 결정적임을 입증한다.

### 3.2 다중 도메인 커버리지를 통한 일반화

DataFlow는 6개의 SOTA 템플릿 파이프라인과 대규모 재사용 가능 오퍼레이터 컬렉션을 기반으로 도메인에 걸쳐 원칙적이고 의미론적으로 풍부하며 확장 가능한 워크플로우를 제공하여 프로그래밍 가능성, 재현성, 데이터 품질을 향상시킨다.

이는 다중 도메인 데이터 혼합(Multi-domain Data Mixture)을 통한 일반화를 공식화할 수 있다:

$$\mathcal{D}_{unified} = \bigcup_{d \in \mathcal{D}_{domains}} \mathcal{P}_d(\mathcal{D}_d^{raw})$$

$$\mathcal{L}_{gen} = \mathbb{E}_{(x,y) \sim \mathcal{D}_{unified}}[\ell(f_\theta(x), y)]$$

### 3.3 Reasoning Data 필터링 및 일반화

Synthetic-1의 랜덤 서브셋으로 학습하면 베이스 모델 대비 제한적인 향상만 나타나며, 2 에포크 후 평균이 지시 전용 기준선(47.0 vs. 46.6)과 유사하게 유지된다. 반면 Open-R1 합성 서브셋은 더 강한 학습 신호를 제공한다. 이는 DataFlow의 정제 및 필터링 스택과 결합된 고품질 합성 데이터가 일반화 성능에 기여함을 보여준다.

### 3.4 일반화 성능 향상 메커니즘 요약

$$\text{Generalization} \uparrow \Leftarrow \begin{cases} \text{Quality Filtering} & (\text{노이즈 제거}) \\ \text{Semantic Diversity} & (\text{분포 커버리지} \uparrow) \\ \text{LLM-in-the-loop Synthesis} & (\text{어려운 예제 생성}) \\ \text{Cross-domain Mixture} & (\text{태스크 전이 능력} \uparrow) \end{cases}$$

6개의 DataFlow 구현 파이프라인에 대한 광범위한 실험은 설계 철학이 다양한 데이터 준비 시나리오에서 효과적이며 지속적으로 고품질 학습 데이터를 생산함을 보여주며, 결과 데이터셋은 큐레이션된 인간 데이터셋, 특화된 합성 워크플로우, Qwen2.5-Instruct 시리즈를 포함한 SOTA 기준선과 동등하거나 능가한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 비교 프레임워크

| 시스템 | 연도 | 오퍼레이터 수 | 도메인 | LLM-driven | 에이전트 지원 | 재현성 |
|--------|------|--------------|--------|-----------|-------------|--------|
| **Data-Juicer** | 2023 | 100+ (v2.0) | 멀티모달 | 부분 | ✗ | 보통 |
| **NeMo Curator** | 2023 | 50+ | 텍스트 | 부분 | ✗ | 보통 |
| **DataComp-LM** | 2024 | - | 텍스트 | ✗ | ✗ | 높음 |
| **Infinity-Instruct** | 2024 | - | 인스트럭션 | ✗ | ✗ | 보통 |
| **DataFlow** | 2025 | ~200 | 6개 도메인 | ✓ | ✓ | 높음 |

여러 시스템이 LLM 데이터 큐레이션 표준화를 목표로 등장했으며, NeMo Curator와 Data-Juicer 같은 프레임워크는 캡셔닝, 재작성, 분류, 멀티모달 처리를 포함한 상당한 기능을 제공하여 대규모 코퍼스 구축 효율성을 크게 향상시켰다.

**Data-Juicer (2023):**
대규모 LLM 데이터 큐레이션을 위한 데이터 다운로드·추출(Common Crawl, arXiv, Wikipedia 등), 언어 식별, 텍스트 클리닝, 휴리스틱 및 학습 기반 품질 필터링, 도메인·독성 분류, 문서 및 의미론적 수준 중복 제거, 프라이버시 필터링, 합성 데이터 생성 등을 Dask/RAPIDS 위에 구현하여 멀티-노드·멀티-GPU 환경으로 확장 설계되었다.

**DataComp-LM (NeurIPS 2024):**
DataComp-LM은 언어 모델을 위한 차세대 학습 세트를 탐구하며 NeurIPS 2024에서 발표되었다.

**DataFlow vs. 기존 시스템의 차별점:**
DataFlow는 약 200개의 재사용 가능한 텍스트 특화 오퍼레이터 라이브러리를 기반으로 클리닝, 변환, 합성, 평가에 대한 세밀한 제어를 가능하게 하며, 이 오퍼레이터들로부터 인스턴스화된 다수의 파이프라인이 지속적으로 강한 다운스트림 성능을 제공하고, 다른 파이프라인이 생성한 데이터의 단순 혼합도 매우 효과적이다.

DataFlow는 신뢰할 수 있고 재현 가능하며 확장 가능한 LLM 데이터 준비를 위한 실용적이고 고성능 기반을 제공하며, 미래의 데이터 중심 AI 개발을 위한 시스템 레벨 토대를 확립한다.

---

## 5. 앞으로의 연구 영향 및 고려 사항

### 5.1 앞으로의 연구에 미치는 영향

#### 🔬 데이터 준비의 패러다임 전환
DataFlow는 LLM 데이터 준비의 커뮤니티 표준으로 기능하는 것을 목표로 하며, 통합 추상화는 재현 가능한 파이프라인 공유, 투명한 LLM 백엔드 교체, 통제된 실험을 가능하게 한다. DataFlow는 LLM 중심 데이터 준비의 전체 워크플로우를 아우른다.

#### 📐 데이터 중심 AI의 체계화
데이터 중심 AI 트렌드에서 LLM 기반 데이터 합성을 일급(first-class)의 프로그래밍 가능한 데이터플로우 추상화로 격상시켜야 한다는 주장은 향후 데이터 준비 연구의 방향성을 제시한다.

#### 🤖 에이전트 기반 워크플로우 자동화
DataFlow-Agent의 도입은 자연어 → 파이프라인 자동 합성이라는 새로운 연구 방향을 열며, 이는 AutoML의 데이터 버전(Auto-Data)으로 발전할 가능성이 있다:

$$\text{Auto-Data}: \mathcal{L}_{nl} \rightarrow \mathcal{P}^* = \arg\max_{\mathcal{P}} \mathbb{E}[\text{Downstream LLM Performance}]$$

#### 🔄 소규모 고품질 데이터 패러다임
DataFlow가 생성한 통합 10K 샘플 데이터셋은 베이스 모델이 1M Infinity-Instruct 데이터로 학습된 대응 모델을 능가할 수 있게 한다는 결과는 "더 적지만 더 좋은 데이터(Less But Better Data)" 패러다임을 강화하며 향후 SFT 연구에 영향을 미칠 것이다.

---

### 5.2 앞으로 연구 시 고려할 사항

#### ① 파이프라인 자동 최적화
약 200개 오퍼레이터의 조합 공간은 $2^{200}$에 달하므로 NAS(Neural Architecture Search) 또는 베이지안 최적화 기반의 파이프라인 탐색이 필요하다:

$$\mathcal{P}^* = \arg\max_{\mathcal{P} \in 2^\mathcal{O}} \text{Val}(\mathcal{M}_\theta(\mathcal{P}(\mathcal{D})))$$

#### ② 멀티모달 확장
DataFlow는 현재 텍스트 중심이므로, 이미지·비디오·오디오 데이터까지 오퍼레이터 생태계를 확장하는 연구가 필요하다.

#### ③ 데이터 품질 평가 메트릭의 표준화
LLM 기반 품질 평가 오퍼레이터의 신뢰성을 정량화하는 표준 메트릭 연구가 필요하다:

$$Q(\mathcal{D}) = \frac{1}{|\mathcal{D}|}\sum_{i=1}^{|\mathcal{D}|} \text{LLM-Judge}(d_i) \in [0, 1]$$

#### ④ 도메인 특화 일반화 검증
DataFlow는 헬스케어, 금융, 법률, 학술 연구 등 특화 도메인에서의 LLM 성능 향상을 위한 데이터 준비 시스템으로 설계되었으나, 이들 도메인에서의 체계적 벤치마킹과 개인정보·규제 준수 문제 처리가 향후 연구 과제이다.

#### ⑤ 데이터 오염(Data Contamination) 방지
LLM을 데이터 합성에 사용할 때 벤치마크 데이터 오염 문제가 발생할 수 있으므로, 이를 탐지·방지하는 메커니즘 연구가 필수적이다.

#### ⑥ 재현성 및 버전 관리 표준
DataFlow의 통합 추상화는 재현 가능한 파이프라인 공유, 투명한 LLM 백엔드 교체, 통제된 실험을 가능하게 하지만, 파이프라인 버전 관리(MLflow, DVC 등과의 통합)에 대한 표준화가 추가로 필요하다.

---

## 📚 참고 자료 (References)

| # | 자료명 | 출처 |
|---|--------|------|
| 1 | **DataFlow: An LLM-Driven Framework...** (주 논문) | [arXiv:2512.16676](https://arxiv.org/abs/2512.16676) |
| 2 | DataFlow HTML 전문 | [arxiv.org/html/2512.16676v1](https://arxiv.org/html/2512.16676v1) |
| 3 | DataFlow PDF | [arxiv.org/pdf/2512.16676](https://arxiv.org/pdf/2512.16676) |
| 4 | HuggingFace Papers | [huggingface.co/papers/2512.16676](https://huggingface.co/papers/2512.16676) |
| 5 | GitHub OpenDCAI/DataFlow | [github.com/OpenDCAI/DataFlow](https://github.com/OpenDCAI/DataFlow) |
| 6 | ResearchGate 논문 페이지 | [researchgate.net/.../398850920](https://www.researchgate.net/publication/398850920) |
| 7 | Semantic Scholar | [semanticscholar.org/.../118f3968](https://www.semanticscholar.org/paper/DataFlow:-An-LLM-Driven-Framework-for-Unified-Data-Liang-Ma/118f3968a66f9e3ec1d013d46a0b30beb1ace810) |
| 8 | ChatPaper.AI 요약 | [chatpaper.ai/paper/...](https://www.chatpaper.ai/paper/7e804156-a103-4430-bcdf-0ca9c5c6e4a1) |
| 9 | Cool Papers | [papers.cool/arxiv/2512.16676v1](https://papers.cool/arxiv/2512.16676v1) |
| 10 | Data-Juicer (arXiv:2309.02033) | [arxiv.org/html/2309.02033v3](https://arxiv.org/html/2309.02033v3) |
| 11 | Data-Centric AI in the Age of LLMs (arXiv:2406.14473) | [arxiv.org/pdf/2406.14473](https://arxiv.org/pdf/2406.14473) |
| 12 | Journal of CS&T: Data Preparation for LLMs | [Springer Nature Link](https://link.springer.com/article/10.1007/s11390-026-5948-8) |
| 13 | PyPI: open-dataflow | [pypi.org/project/open-dataflow](https://pypi.org/project/open-dataflow/) |

> ⚠️ **정확도 관련 고지:** 본 답변은 공개된 arXiv 논문 전문 및 관련 소스에 기반하며 모든 수치와 인용은 출처를 명시하였습니다. 논문 내부의 상세 수식(Qurating 임계값 등 일부)은 논문 원문 직접 확인을 권장합니다. DataFlow-Agent의 세부 알고리즘 수식 등 일부 세부 사항은 논문 원문([arXiv:2512.16676](https://arxiv.org/abs/2512.16676))에서 직접 확인하시기 바랍니다.
