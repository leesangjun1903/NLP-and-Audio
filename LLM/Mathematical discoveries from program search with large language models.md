# Mathematical discoveries from program search with large language models

### 1. 핵심 주장과 주요 기여

**FunSearch** (Function Space Searching)는 구글 DeepMind에서 개발한 혁신적인 방법으로, **사전학습된 대규모 언어 모델(LLM)과 체계적인 평가 함수를 결합한 진화 알고리즘**입니다. 이 논문의 핵심 주장은 단순하지만 강력합니다: LLM의 창의적 능력을 진화 알고리즘의 체계적 선택과 결합하면, 기존 최고 기록을 넘어서는 **새로운 수학적 발견과 알고리즘 설계**가 가능하다는 것입니다.[1]

**주요 기여:**

- **개방 문제에 대한 새로운 발견:** 극값 조합론의 cap set 문제에서 20년 만에 최초의 점근 하한 개선을 달성[1]
- **해석 가능한 프로그램 발견:** 수치 나열이 아닌 **프로그램으로 표현된 해**를 통해 인간 연구자가 이해하고 개선할 수 있는 형태의 결과 제공[1]
- **일반화 성능의 실증:** 온라인 빈 패킹 문제에서 훈련 분포 외의 데이터에 우수한 일반화 성능 입증[1]
- **확장 가능한 프레임워크:** 이질적인 수학 및 컴퓨터 과학 문제에 적용 가능한 통일된 방법론 제시[1]

***

### 2. 해결하고자 하는 문제와 제안 방법

#### 2.1 문제 정의

FunSearch가 해결하고자 하는 핵심 문제는 다음과 같습니다:[1]

> **"평가는 쉽지만 해결은 어려운 문제"** - 주어진 후보 해에 대한 품질 평가가 다항식 시간에 가능하지만, 최적 해를 찾는 것은 어려운 문제들

구체적으로:
- **극값 조합론:** Cap set 문제 - $$\mathbb{Z}_3^n$$에서 세 벡터의 합이 영(zero)이 되지 않는 최대 크기의 집합 찾기
- **조합 최적화:** 온라인 빈 패킹 - 들어오는 아이템을 실시간으로 가장 적은 수의 빈에 배치

이들 문제의 공통점은 효율적인 **evaluate** 함수가 존재한다는 것입니다.

#### 2.2 핵심 방법론

FunSearch의 아키텍처는 다음 요소들로 구성됩니다:[1]

** 문제 명세(Specification)**[1]
```
입력: 
- evaluate(candidate_solution) → score
- 초기 프로그램 (자명할 수 있음)
- 프로그램 스켈레톤 (선택사항)
```

** 사전학습 LLM**[2]
- Google의 Codey (PaLM2 기반)[1]
- 또는 StarCoder와 같은 오픈 소스 모델
- **창의적 프로그램 생성 역할**

** 프로그램 데이터베이스 (Island-Based Evolutionary Model)**[3]

다양성 유지를 위해 여러 **섬(島; island)** 구조 사용:

$$P_i = \frac{\exp(s_i / T_{cluster})}{\sum_j \exp(s_j / T_{cluster})} \quad \text{(1)}$$

여기서:
- $$s_i$$: 클러스터 $i$의 점수
- $$T_{cluster}$$: 온도 매개변수
- $T = 1 - \frac{\text{mod}(n, N_{cluster})}{N_{cluster}}$ (시간에 따른 감소)[1]

** Best-Shot Prompting**[4]

상위 $k=2$개의 고성능 프로그램을 함께 프롬프트에 포함:

$$\text{Prompt} = [\text{priority}_{\text{v0}}, \text{priority}_{\text{v1}}, \text{priority}_{\text{v2(생성예정)}}]$$

LLM은 이전 버전들의 패턴을 학습하여 개선된 새 프로그램 생성[1]

** 분산 시스템 구조**[5]
- **샘플러:** LLM 추론 (15개 스레드)
- **평가자:** 프로그램 평가 (150개 CPU 기반)
- **프로그램 데이터베이스:** 중앙 저장소 및 조율

***

### 3. 모델 구조 및 성능 향상

#### 3.1 Cap Set 문제 적용

**프로그램 스켈레톤:**
```python
def main(n):
    solution = solve(n)
    return evaluate(solution, n)

def solve(n):
    elements = get_all_elements(n)
    scores = [priority(el, n) for el in elements]
    elements = elements[argsort(scores)[::-1]]
    capset = []
    for element in elements:
        if can_be_added(element, capset):
            capset.append(element)
    return capset

def priority(element, n):  # ← FunSearch가 진화시키는 부분
    return evolved_heuristic(element, n)
```

**성과:**
- n=8 차원에서 **512-cap** 발견 (기존 최고: 496)[1]
- 이는 더 간단한 구조로 이전 방법들이 필요하던 복잡한 하위 차원 조합 없이 달성[1]

**발견된 Priority 함수의 특성:**
```python
def priority(el: tuple[int,...], n: int) -> float:
    score = n
    in_el = 0
    el_count = el.count(0)
    if el_count == 0:
        score += n**2
    if el[1] == el[-1]:  # 대칭성 발견
        score *= 1.5
    # ... (추가 휴리스틱)
    return score
```

연구자들이 이 코드를 분석한 결과, **4개 트리플 그룹에서의 순환 대칭성** 발견 → 이를 통해 더욱 정교한 구조 설계 가능[1]

#### 3.2 Admissible Sets를 통한 점근 개선

Cap set 용량의 하한: $$C_c = \sup_n \frac{c_n}{n^{1/n}}$$

- 기존 최고: $$C \geq 2.2180$$[1]
- FunSearch 발견: $$C \geq 2.2202$$ (**20년 만의 개선**)
- 발견된 $$I(24,17)$$ admissible set: **237,984개 벡터**[1]

$$\text{하한 진행}: 2.2101 \to 2.2173 \to 2.2180 \to 2.2184 \to 2.2194 \to 2.2202$$[1]

#### 3.3 온라인 빈 패킹 문제

**Heuristic 함수 발견:**

FunSearch가 학습한 휴리스틱의 핵심 전략:
- Best fit 휴리스틱과 다르게, **타이트한 적합에서만 최적 빈 사용**[1]
- 그 외에는 남은 공간이 더 많은 다른 빈 선택
- 작은 간격 생성 회피 → 패킹 효율 향상[1]

**성능 비교 (Table 1):**

| 데이터셋 | First Fit | Best Fit | FunSearch |
|---------|----------|----------|-----------|
| OR1 | 6.42% | 5.81% | 5.30% |
| OR2 | 6.45% | 6.06% | 4.19% |
| OR3 | 5.74% | 5.37% | 3.11% |
| OR4 | 5.23% | 4.94% | 2.47% |
| Weibull 100k | 4.00% | 3.79% | 0.03% |

**일반화 능력:** OR1 크기의 데이터로만 훈련했으나, 더 큰 인스턴스(OR4)에서 더 큰 성능 향상 달성[1]

***

### 4. 일반화 성능 향상 분석

#### 4.1 일반화 메커니즘

FunSearch의 일반화 성능 우수성의 근본 원인:[1]

**Kolmogorov 복잡도 편향:**

전통적 컴퓨터 탐색은 구체적 해의 거대한 숫자 나열을 찾지만, FunSearch는:

$$\text{Kolmogorov Complexity} = \text{min}|p| : \text{프로그램 } p \text{가 해를 생성}$$

짧은 프로그램을 암묵적으로 선호하므로, **일반화되는 구조**를 발견하는 경향[1]

예: Admissible set $$A(24,17)$$
- 수치 표현: 237,984개 벡터
- 프로그램 표현: **단 몇 줄의 코드**[1]

#### 4.2 구체적 일반화 증거

**온라인 빈 패킹에서의 외삽법(Extrapolation):**

- 훈련: OR1 (약 50-100 항목)
- 테스트: OR4 (약 1000+ 항목), Weibull 100k (100,000 항목)[1]
- **결과:** 더 큰 인스턴스에서 더 큰 성능 향상 폭 (2.47% vs 0.03%)[1]

**통계적 안정성:**

모든 테스트에서 FunSearch가 baseline을 일관되게 초과[1]

***

### 5. 방법의 한계

#### 5.1 적용 가능 조건

FunSearch는 다음 조건을 만족하는 문제에 가장 효과적:[1]

1. **효율적 평가 함수 필요:** 평가가 다항식 시간에 가능
2. **풍부한 피드백 신호:** 이진 신호가 아닌 연속적 점수 필요
3. **프로그램 스켈레톤 제공:** 진화할 핵심 부분이 명확해야 함

#### 5.2 해결되지 않은 문제

- **정리 증명:** 증명 생성은 충분히 풍부한 피드백 신호를 정의하기 어려움[1]
- **안정성:** 일부 문제(예: n=8 cap set)에서 140회 중 4회만 성공 (2.9% 성공률)[1]
- **계산 비용:** 약 106 개의 LLM 샘플 필요로 상당한 리소스 소요

#### 5.3 Kolmogorov 복잡도 편향의 한계

모든 문제가 짧은 프로그램으로 표현되는 해를 가지지는 않으며, 임의적 해도 많이 존재[1]

***

### 6. 최신 연구 기반 향후 영향 및 고려사항

#### 6.1 FunSearch의 영향 및 확장 (2024-2025)

** Generative Modeling for Mathematical Discovery (2025):**[1]
새로운 공개 구현이 FunSearch를 여러 LLM에서 테스트하며, **일반화 성능이 모델에 크게 의존하지 않음** 입증[6]

** LLM-SR: 과학 방정식 발견 (2025):**[2]
FunSearch의 원리를 기반으로 **기호 회귀(symbolic regression)** 분야 적용 → 물리학, 생물학 등에서 지배 방정식 발견[7]

** FunBO: 베이지안 최적화 획득함수 설계 (2024):**[3]
FunSearch를 활용해 자동으로 획득함수(acquisition function) 설계 → 커스텀 함수보다 우수한 성능[4]

** 경쟁 프로그래밍에 응용 (2024-2025):**[4]
Hash Code 대회에서 채점함수 최적화에 FunSearch 적용 → 상위 백분위수 달성[3]

#### 6.2 관련 최신 동향

**Evolution Strategies (ES)의 부상 (2025):**
LLM 매개변수 공간에 진화 전략을 직접 적용하는 새로운 패러다임 출현[8]
- 강화학습보다 **표본 효율성 높음** (20% 미만의 평가로 유사 성능)
- 더 **안정적이고 예측 가능한 최적화**[8]

**Hyper-heuristics 분야의 통합:**
전통적 유전 프로그래밍의 문제점(휴리스틱 수작업 설계)을 LLM이 해결하면서, **자동 알고리즘 설계의 새로운 표준** 형성 중[9]

#### 6.3 앞으로의 연구 시 고려사항

** 안정성 개선:**[1]
- 다중 독립 실행 필요성 대폭 감소를 위한 적응형 진화 전략
- 조기 수렴 방지 메커니즘 강화

** 계산 효율성:**[2]
- 현재 106개 샘플이 필수이나, 향후 LLM의 질 향상으로 **더 적은 샘플로 수렴 가능**[10]
- 더 효율적인 평가 함수 설계 연구

** 해석 가능성 극대화:**[3]
- FunSearch 발견 프로그램에 대한 **자동 분석 및 추상화** 도구 개발
- Domain expert와의 상호작용 루프 강화[1]

** 새로운 문제 도메인 확장:**[4]
- 약한 평가함수를 가진 문제들에 대한 **신호 설계 전략**
- 증명 생성, 정책 합성 등 고차 추론 작업

** 모듈화 및 조합:**[5]
- 다양한 문제 도메인의 "기초 프로그램 블록" 라이브러리 구축
- **전이 학습** 적용 (한 문제에서 학습한 휴리스틱을 다른 문제에 전달)

** 하이브리드 접근:**[11]
- LLM 기반 생성과 전통적 수학적 제약 결합
- Neural-symbolic 방법론과의 통합

***

### 결론

FunSearch는 **대규모 언어 모델의 창의성과 진화 알고리즘의 체계성을 결합한 획기적 방법론**으로, 기존에는 불가능하다고 여겨졌던 **수학적 발견과 알고리즘 설계를 LLM 기반 시스템으로 자동화**했습니다. 특히 **프로그램 공간 검색이라는 표현**을 통해 얻은 Kolmogorov 편향이 일반화 성능을 크게 향상시켰으며, 발견된 프로그램의 **해석 가능성**은 인간 연구자와의 협력을 가능하게 합니다.

향후 LLM 품질 향상, 계산 효율성 개선, 그리고 다양한 도메인으로의 확장을 통해 FunSearch는 **자동화된 과학 발견의 새로운 기준**을 제시할 것으로 예상됩니다.

***

### References

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b68d9050-61a7-43b2-b162-20be45fc1b24/s41586-023-06924-6.pdf)
[2](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/)
[3](http://arxiv.org/pdf/2411.19744.pdf)
[4](https://arxiv.org/html/2406.04824)
[5](http://arxiv.org/pdf/2405.05606.pdf)
[6](http://arxiv.org/pdf/2503.11061.pdf)
[7](https://openreview.net/forum?id=m2nmp8P5in)
[8](https://arxiv.org/abs/2509.24372)
[9](https://human-competitive.org/sites/default/files/paper1_2.pdf)
[10](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)
[11](http://arxiv.org/pdf/2405.19749.pdf)
[12](http://arxiv.org/pdf/2410.02992.pdf)
[13](https://arxiv.org/html/2411.06634v1)
[14](https://www.nature.com/articles/s41586-023-06924-6)
[15](https://www.themoonlight.io/en/review/evolution-strategies-at-scale-llm-fine-tuning-beyond-reinforcement-learning)
[16](https://arxiv.org/html/2503.11061v1)
[17](https://aclanthology.org/2025.findings-acl.821.pdf)
[18](https://www.semanticscholar.org/paper/Mathematical-discoveries-from-program-search-with-Romera-Paredes-Barekatain/d32ba88571141ed0ebe7aeefbaa4ccaf8cda7be3)
[19](https://www.sciencedirect.com/science/article/abs/pii/S0925231224020435)
[20](http://arxiv.org/pdf/1101.5851.pdf)
[21](https://arxiv.org/abs/2206.09719)
[22](https://arxiv.org/pdf/2412.03862.pdf)
[23](https://arxiv.org/pdf/2103.06481.pdf)
[24](http://arxiv.org/pdf/2502.06699.pdf)
[25](https://www.nature.com/articles/s41598-024-81749-5)
[26](https://intuitionlabs.ai/articles/mechanistic-interpretability-ai-llms)
[27](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf)
[28](https://arxiv.org/abs/2007.00463)
[29](https://www.lesswrong.com/posts/XGHf7EY3CK4KorBpw/understanding-llms-insights-from-mechanistic)
[30](https://ieeexplore.ieee.org/document/10864950/)
[31](https://arxiv.org/html/2508.11703v1)
[32](https://community.openai.com/t/paper-mathematical-discoveries-from-program-search-with-large-language-models/560495)
