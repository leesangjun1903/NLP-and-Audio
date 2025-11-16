# The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only

### 1. 핵심 주장과 주요 기여

본 논문의 **핵심 주장**은 전통적인 통념을 도전하는 것입니다. 기존에는 대규모 언어모델(LLM) 훈련에 있어 **수동 큐레이션된 고품질 코퍼스(예: 책, 기술 논문, 소셜 미디어)의 혼합이 필수적**이라고 여겨졌습니다. 그러나 이 연구는 **적절히 필터링되고 중복 제거된 웹 데이터만으로도 큐레이션된 코퍼스를 능가하는 성능의 모델을 훈련할 수 있음**을 입증했습니다.[1]

**주요 기여**는 다음과 같습니다:[1]

- **RefinedWeb 데이터셋**: 5조 토큰의 고품질 웹 전용 영어 사전훈련 데이터셋 개발 (공개 버전: 600억 토큰)
- **데이터 품질에 대한 새로운 관점**: 웹 데이터만으로 The Pile(공개 큐레이션 데이터셋)을 능가하는 성능 달성
- **확장 가능한 데이터 처리 파이프라인**: Macrodata Refinement(MDR) 파이프라인 제시로 대규모 웹 데이터 처리 방법 표준화
- **공개 자료**: 1.3B/7.5B 파라미터의 훈련된 언어모델과 함께 600B 토큰 데이터셋 공개

### 2. 문제 정의 및 해결 방법

#### 해결하고자 하는 문제

LLM의 성능 향상을 위해 Hoffmann et al. (2022)의 스케일링 법칙에 따르면 GPT-3 규모(175B 파라미터) 모델 최적 훈련에 **3.5조 토큰**이 필요합니다. 그러나 이는 현존 최대 공개 영어 데이터셋(OSCAR, C4, The Pile)보다 10배 이상 큽니다. 동시에 고품질 데이터의 큐레이션은 노동 집약적이고 라이선스 문제가 있으며 확장성이 제한되어 있습니다.[1]

#### 제안하는 방법: Macrodata Refinement (MDR) 파이프라인

MDR 파이프라인은 세 가지 원칙을 따릅니다:[1]

**1. 규모 우선(Scale First)**: 3-6조 토큰을 목표로 CommonCrawl 중심의 처리
**2. 엄격한 중복 제거(Strict Deduplication)**: 정확한 중복 제거와 퍼지(fuzzy) 중복 제거 결합
**3. 중립적 필터링(Neutral Filtering)**: ML 기반 필터링을 피하고 규칙 기반 휴리스틱 사용

MDR은 다음과 같은 단계로 구성됩니다:[1]

**문서 준비 단계:**
- **URL 필터링**: 4.6M 도메인 차단 목록과 가중치 기반 URL 점수 시스템 사용 (성인 콘텐츠, 도박, 폭력 등 제거)
- **텍스트 추출**: trafilatura 라이브러리 사용으로 메뉴, 광고, 해더/풋터 제거
- **언어 식별**: fastText 분류기로 상위 언어 점수 0.65 이상만 보유

**필터링 단계:**
- **반복 제거**: 행, 문단, n-그램 반복 제거로 경로 적 동작 방지
- **문서 수준 필터링**: 기호-단어 비율, 길이 이상치 등 휴리스틱 기반 필터링
- **행 수준 교정**: 소셜 미디어 카운터("3 likes"), 네비게이션 버튼 등 제거

**중복 제거 단계:**

퍼지 중복 제거와 정확한 부분 문자열 중복 제거를 결합:[1]

$$
\text{MinHash} = \text{Document} \rightarrow \text{Sketch} \rightarrow \text{Similarity Measurement}
$$

- **MinHash 기반 퍼지 중복 제거**: 9,000 해시/문서, 5-그램, 20개 버킷으로 유사 문서 제거
- **정확한 부분 문자열 중복 제거**: 50 토큰 이상의 연속 정확한 일치 제거 (suffix array 사용)
- **URL 중복 제거**: CommonCrawl 덤프 간 재방문 URL 추적 및 제거

#### 파이프라인의 제거율

CommonCrawl 원본 대비 제거율:[1]
- URL 필터링 후: 약 2.2%만 보유
- 언어 식별 후: 약 48% 보유
- 필터링 완료(RW-Filtered): 약 23% 보유  
- 최종(RW, 중복 제거 포함): 약 10% 보유

이는 총 약 90% 문서 제거를 의미하며 엄격한 품질 기준을 반영합니다.[1]

### 3. 모델 구조 및 아키텍처

본 논문의 모델은 **GPT-3 기반의 인과적 디코더 전용(Causal Decoder-Only) 트랜스포머** 구조를 사용합니다.[1]

**주요 아키텍처 특징:**[1]

- **ALiBi(Attention with Linear Biases)**: Press et al. (2021)의 선형 바이어스 기반 위치 인코딩
- **FlashAttention**: Dao et al. (2022)의 메모리 효율적 어텐션 메커니즘
- **모델 사양**:
  - 1B 파라미터 모델: 27GT/60GT 토큰 훈련
  - 3B 파라미터 모델: 60GT 토큰 훈련  
  - 1B/7B 파라미터 모델: 350GT 토큰 훈련 (주요 실험)

모델 아키텍처는 데이터셋의 품질이 주요 초점이므로, **구조 자체보다는 훈련 데이터 품질이 핵심 기여**입니다.

### 4. 성능 향상 및 실험 결과

#### 소규모 연구 (Small-Scale Study)

1B/3B 파라미터 모델을 27GT/60GT 토큰으로 훈련한 결과:[1]

| 데이터셋 | 1B@27GT | 3B@60GT |
|---------|---------|---------|
| OSCAR-21.09 | 55.0% | 59.1% |
| OSCAR-22.01 | 52.7% | 55.9% |
| C4 | 55.7% | 59.6% |
| The Pile | 53.4% | 57.9% |
| **RefinedWeb** | **56.2%** | **59.8%** |

RefinedWeb이 모든 경쟁 데이터셋을 능가합니다.[1]

#### 대규모 실험 (Full-Scale Study)

1B/7B 모델을 350GT 토큰으로 훈련한 경우, **RefinedWeb 모델이 GPT-3 성능에 필적하거나 초과**했습니다:[1]

- **Main-agg 집계**: RefinedWeb 모델의 제로샷 성능이 The Pile 모델과 동일 계산 예산에서 상당히 능가
- **Core-agg/Ext-agg 집계**: 공개 모델 중에서도 최고 성능 달성

#### 단계별 효과 분석

각 단계의 기여도(소규모 실험 기준):[1]

| 파이프라인 단계 | 1B@27GT | 3B@60GT |
|---------------|---------|---------|
| RW-Raw(필터링 전) | 52.7% | 57.4% |
| RW-Filtered(중복 제거 전) | 54.3% | 58.2% |
| **RefinedWeb(최종)** | **56.2%** | **59.8%** |

필터링과 중복 제거가 각각 1.6%p와 1.9%p 개선을 기여합니다.[1]

#### MDR 파이프라인의 다른 데이터셋 적용

기존 데이터셋(Table 5)에 MDR 적용 결과:[1]

| 데이터셋 | Base | 필터링 | 중복제거 | 필터링+중복제거 |
|---------|------|--------|---------|----------------|
| OSCAR-21.09 | 55.0% | +0.4% | +0.6% | +0.5% |
| OSCAR-22.01 | 52.7% | -0.4% | +2.9% | +2.7% |
| C4 | 55.7% | +0.5% | +0.2% | +0.7% |
| The Pile | 53.4% | +0.8% | +1.1% | +1.8% |

**핵심 발견**: 필터링 효과는 데이터셋 의존적이나 **중복 제거는 모든 데이터셋에서 일관되게 개선 제공**합니다.[1]

### 5. 일반화 성능 향상 가능성

일반화 성능은 본 논문의 중요한 측면입니다. 논문은 **제로샷 범용 성능**을 광범위한 벤치마크에서 평가했습니다:[1]

**평가 전략:**
- **18개 작업 기반**: 문장 완성(HellaSwag, LAMBADA), 상식 추론(PIQA, ARC), 자연어 추론(Winogrande, RTE) 등
- **4개 집계**: small(소규모), core(핵심), main(주요), ext(확장)
- **비교 프레임워크**: EleutherAI 평가 하네스 사용으로 공정한 비교

**일반화 성능 향상 요인:**

1. **데이터 다양성 보존**: 엄격한 중복 제거(50% 제거)로 인한 데이터 중복 감소는 모델이 다양한 표현 방식을 학습하도록 강제[1]

2. **품질 기반 필터링의 효과**: 휴리스틱 기반 필터링(ML 기반 아님)으로 불필요한 도메인 제거 없이 노이즈만 제거하여 도메인 편향 최소화[1]

3. **메모리 효과 감소**: Carlini et al. (2022)의 연구에 따르면 중복 제거로 모델 암기(memorization) 감소로 일반화 능력 향상[1]

4. **스케일링 법칙과의 정렬**: Hoffmann et al. (2022)의 Chinchilla 스케일링 법칙에 따른 데이터-모델 균형으로 최적 일반화 달성[1]

**다중 에포크 환경에서의 영향:**

Appendix E.3에서 1-100 에포크 실험 결과, RefinedWeb은 RW-Filtered보다 중복 에포크에서 성능 저하가 적습니다. 이는 엄격한 중복 제거가 제한된 데이터 환경에서도 일반화 개선을 지원함을 시사합니다.[1]

### 6. 논문의 한계

저자들이 명시한 한계사항:[1]

**1. 편향 및 독성 분석의 한계:**
- Perspective API 정의("무례하거나 무례한 콘텐츠")에 한정된 분석
- 사회적 편향, 해로움 등은 미분석
- RefinedWeb의 독성 수준은 The Pile과 유사

**2. 다중 에포크 훈련:**
- Hernandez et al. (2022)에 따르면 100B+ 파라미터 모델은 여러 에포크에 민감
- 본 연구는 단일 에포크 전략 채택
- 중복 제거가 다중 에포크 시 데이터 부족을 완화할지 미해결

**3. 다른 중복 제거 연구와의 불일치:**
- Biderman et al. (2023)의 Pythia 연구에서 The Pile 중복 제거 효과가 제한적
- 웹 데이터와 큐레이션 데이터의 중복 특성이 다를 수 있음을 시사

**4. 다국어 미지원:**
- 공개 버전은 영어만 포함
- 다국어 응용 제한

### 7. 최신 연구에 미치는 영향 및 앞으로의 고려 사항

#### 영향과 발전

**1. 후속 고품질 웹 데이터셋 개발:**

RefinedWeb의 성공 이후, 더욱 정교한 웹 데이터 처리 파이프라인이 개발되었습니다:[2][3][4]

- **FineWeb (2024)**: 15조 토큰의 개선된 CommonCrawl 데이터셋으로 RefinedWeb의 다음 세대 표준 확립[5]
- **FineWeb-Edu (2024)**: LLama3-70B 주석을 사용한 교육적 품질 분류기로 1.3조 토큰의 교육 중심 데이터셋 생성[6]
- **FineWeb2 (2025)**: 1,000개 이상 언어를 지원하는 다국어 확장판으로 20테라바이트 데이터셋[4]

**2. 모델 기반 데이터 필터링으로의 전환:**

초기 휴리스틱 중심에서 신경망 분류기 기반 필터링으로 진화:[3][7]

- **Ultra-FineWeb**: fastText 기반 경량 분류기로 1조 영어 + 1,200억 중국어 토큰 생성[3]
- **FinerWeb-10BT**: GPT-4o mini의 행 수준 라벨링으로 25% 데이터 감소 시 동일 성능 달성[7]

**3. 온라인 데이터 재가중치 방법의 등장:**

오프라인 필터링의 한계를 극복하는 새로운 접근:[8]

- **ADAPT (2025)**: 훈련 중 샘플 중요도를 동적으로 조정하는 온라인 재가중치 프레임워크로 기존 오프라인 방식보다 우수한 일반화 성능[8]

#### 앞으로 연구 시 고려할 점

**1. 데이터 품질 평가의 다층화:**

- **문제**: Perspective API의 "무례함" 정의는 편협함. 미래 연구는 사회적 편향, 성별/나이/종교 편향, 허위정보 등 다각적 평가 필요
- **해결책**: 도메인별 맞춤 평가 메트릭 개발, 공정성 검증 자동화[9][10]

**2. 합성 데이터와의 통합:**

- **문제**: 웹 데이터의 한계로 정보 밀집도 낮음
- **해결책**: 최근 BeyondWeb, WRAP 등 합성 데이터 생성 기법으로 웹 데이터를 보강[11][12]

**3. 온라인 학습 기반 데이터 선택:**

- **문제**: 오프라인 필터링은 모델/작업 변경 시 재실행 필요
- **해결책**: ADAPT처럼 훈련 중 적응적 샘플 재가중치로 유연성 확보[8]

**4. 다국어 및 저자원 언어 지원:**

- **문제**: RefinedWeb은 영어만, FineWeb2도 균형 문제 존재
- **해결책**: 언어별 필터링 임계값 자동 조정, 저자원 언어 중복 제거 최적화[4]

**5. 메모리화 vs 일반화 균형:**

- **문제**: 너무 강한 중복 제거는 정보 손실, 약한 중복 제거는 암기 증가
- **해결책**: Hernandez et al. (2022)의 에포크 수에 따른 중복 제거 강도 적응형 조정[1]

**6. 신흥 아키텍처와의 상호작용:**

- **현황**: Falcon-H1의 Transformer-Mamba 하이브리드 아키텍처 등장
- **고려사항**: 상이한 아키텍처 특성에 맞는 데이터 분포 최적화 필요[13]

**7. 환경 비용 고려:**

- **문제**: 대규모 중복 제거의 계산 비용
- **해결책**: 경량 분류기(fastText) 활용으로 추론 비용 절감, 병렬 처리 최적화[3]

### 결론

RefinedWeb Dataset for Falcon LLM 논문은 **웹 데이터의 품질 향상만으로도 수동 큐레이션을 능가할 수 있다**는 패러다임 전환을 제시했습니다. 엄격한 필터링과 중복 제거 파이프라인을 통해 CommonCrawl에서 5조 토큰이라는 대규모 고품질 데이터셋을 구축함으로써 LLM 사전훈련의 확장성 문제에 현실적 해결책을 제공했습니다.[1]

특히 **일반화 성능의 향상**은 데이터 중복 제거로 인한 다양성 보존, 휴리스틱 필터링의 도메인 편향 최소화, 모델 암기 감소 등 다층적 메커니즘을 통해 달성되었습니다. 이는 단순히 더 많은 데이터가 아닌 **더 나은 품질의 데이터 구축의 중요성**을 강조합니다.[1]

이후 FineWeb, Ultra-FineWeb, 온라인 재가중치 기법 등 지속적인 발전은 이 논문의 기초 위에 모델 기반 필터링, 합성 데이터 통합, 적응형 데이터 선택 등으로 진화하고 있습니다. 앞으로 연구자들은 다국어 지원, 사회적 편향 평가, 신흥 아키텍처와의 최적화, 환경 비용 감소 등의 도전에 집중해야 할 것입니다.[9][5][4][3][8]

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ab8c43a3-d785-4e66-99e7-9fe190f12cb6/2306.01116v1.pdf)
[2](https://ojs.aaai.org/index.php/ICWSM/article/view/35948)
[3](https://arxiv.org/html/2505.05427v1)
[4](https://arxiv.org/html/2506.20920v1)
[5](https://arxiv.org/html/2406.17557v1)
[6](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
[7](https://arxiv.org/pdf/2501.07314.pdf)
[8](https://openreview.net/forum?id=UFwnsmFZ6R)
[9](https://arxiv.org/pdf/2401.13086.pdf)
[10](https://www.gable.ai/blog/llm-data-quality)
[11](https://www.datologyai.com/blog/beyondweb)
[12](https://aclanthology.org/2024.acl-long.757.pdf)
[13](https://arxiv.org/html/2507.22448v1)
[14](http://medrxiv.org/lookup/doi/10.1101/2025.06.06.25329104)
[15](https://revistasinstitutoperspectivasglobales.org/index.php/sanitas/article/view/687)
[16](https://www.semanticscholar.org/paper/a13ee77d3302e573d61b5565fec2944ce31b0ecf)
[17](https://ijonmes.net/index.php/ijonmes/article/view/428)
[18](https://ijsmr.in/doc/ijsmr08_136.pdf)
[19](http://medrxiv.org/lookup/doi/10.1101/2025.08.15.25333781)
[20](https://dl.acm.org/doi/10.1145/3701551.3705706)
[21](https://arxiv.org/abs/2506.13681)
[22](https://www.richtmann.org/journal/index.php/jesr/article/view/14125)
[23](https://arxiv.org/pdf/2306.01116.pdf)
[24](https://arxiv.org/html/2411.16232v1)
[25](https://aclanthology.org/2023.findings-emnlp.608.pdf)
[26](http://arxiv.org/pdf/2407.12858.pdf)
[27](https://arxiv.org/pdf/2311.16867.pdf)
[28](https://arxiv.org/pdf/2401.17645.pdf)
[29](https://arxiv.org/pdf/2307.06616.pdf)
[30](https://arxiv.org/pdf/2411.14513.pdf)
[31](https://www.brainillustrate.com/2025/09/falcon-ascendant-in-depth-analysis-of.html)
[32](https://aws.amazon.com/blogs/machine-learning/an-introduction-to-preparing-your-own-dataset-for-llm-training/)
[33](https://www.labellerr.com/blog/data-collection-and-preprocessing-for-large-language-models/)
[34](https://aiforgood.itu.int/how-falcon-llm-is-reshaping-accessibility-through-model-efficiency/)
[35](https://openreview.net/forum?id=jpSLXoRKnH)
[36](https://aclanthology.org/2025.naacl-long.421.pdf)
[37](https://biss.pensoft.net/article/137867/)
[38](https://arxiv.org/pdf/2503.00808.pdf)
[39](http://arxiv.org/pdf/2502.10361.pdf)
[40](https://arxiv.org/html/2404.01336v1)
[41](http://arxiv.org/pdf/2411.16387.pdf)
[42](https://arxiv.org/pdf/2407.12481.pdf)
[43](https://aclanthology.org/2025.acl-long.123.pdf)
[44](https://aclanthology.org/2025.nodalida-1.27.pdf)
[45](https://commoncrawl.org/blog/common-crawl-foundation-at-colm-2025)
[46](https://arxiv.org/html/2406.17557v2)
