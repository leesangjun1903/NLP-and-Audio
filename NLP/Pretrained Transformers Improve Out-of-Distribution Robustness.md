# Pretrained Transformers Improve Out-of-Distribution Robustness

### 1. 핵심 주장 및 주요 기여

본 논문의 **핵심 주장**은 사전학습된 Transformer 모델(BERT, RoBERTa 등)이 분포 외(Out-of-Distribution, OOD) 예제에 대해 기존 NLP 모델(Bag-of-Words, LSTM, ConvNet)보다 훨씬 뛰어난 강건성을 가진다는 것입니다. 논문은 고정되지 않은 IID(독립동일분포) 가정이 현실 세계와 맞지 않는다는 점에서 출발하여, 실제 분포 변화에 견디는 모델의 성능을 체계적으로 평가합니다.[1]

**주요 기여**는 다음과 같습니다:[1]

첫째, NLP 태스크를 위한 **최초의 포괄적 OOD 강건성 벤치마크**를 구축했습니다. 이는 감정 분석, 텍스트 함축 추론, 읽기 이해, 의미 유사성을 포함한 7개 데이터셋을 대상으로 쓰기 스타일, 주제, 어휘 변화로 인한 현실적인 분포 변화를 측정합니다.[1]

둘째, OOD 강건성을 **일반화 능력**과 **이상 탐지 능력**의 두 가지 차원으로 분해하여 종합적으로 평가합니다.[1]

셋째, **실증적 발견**들을 제시합니다: RoBERTa 모델은 LSTM이 35% 이상 성능 저하를 보일 때 오히려 성능이 증가하고, 모델 크기 증가가 OOD 강건성 향상으로 직결되지 않으며, 다양한 사전학습 데이터가 OOD 강건성 개선에 효과적임을 발견합니다.[1]

***

### 2. 문제 정의 및 제안 방법

#### 2.1 해결하고자 하는 문제

현대 NLP 모델들이 IID 설정에서 높은 정확도를 달성하고 있지만, 실제 배포 환경에서는 훈련 분포와 다른 테스트 분포를 마주합니다. 주요 문제는 다음과 같습니다:[1]

**분포 변화(Distribution Shift)**: 시간에 따른 데이터 변화, 미지의 데이터 등장, 주석 편향(Annotation Artifacts)에 의해 모델이 허위 신호에 의존할 수 있습니다.[1]

**평가 방법론의 한계**: 기존 NLP 연구는 IID 가정에서만 모델을 평가하여, 실제 강건성을 파악하지 못합니다.[1]

#### 2.2 제안하는 평가 방법론

**OOD 일반화 측정**:[1]

연구팀은 두 가지 방식으로 OOD 테스트 세트를 구성합니다:

1. **메타데이터 기반 분할**: 데이터셋의 메타데이터를 활용하여 서로 다른 분포를 만듭니다. 예를 들어, SST-2(전문가 영화 리뷰)와 IMDb(일반 영화 리뷰)를 교차 학습하여 리뷰 형식 변화를 측정합니다.[1]

2. **데이터셋 페어링**: Yelp 리스트를 음식 유형(미국, 중국, 이탈리아, 일본식)으로 분할하거나, Amazon 리뷰를 제품 카테고리(의류, 음악, 영화)로 분할합니다.[1]

**OOD 탐지 측정**:[1]

모델의 예측 확신도(Maximum Softmax Probability)를 이상 탐지 점수로 변환하여 이상 탐지 성능을 평가합니다. 평가 지표는 **FAR95(95% Recall에서의 거짓 경보율)**입니다:

$$ \text{FAR95} = P(\text{경보} | x \in D_{in}) \text{ at 95% OOD 탐지율} $$

낮은 FAR95 값이 더 우수함을 의미합니다.[1]

**AUROC(Area Under ROC Curve)**도 함께 사용합니다:

$$ \text{AUROC} = P(-\max_y p(y | x_{out}) > -\max_y p(y | x_{in})) $$

이는 OOD 예제가 ID 예제보다 높은 이상 점수를 받을 확률을 나타냅니다.[1]

***

### 3. 모델 구조 및 실험 설정

#### 3.1 비교 모델 구조

논문은 **13개 모델**을 평가합니다:[1]

| 모델 유형 | 구체적 모델 | 특징 |
|---------|---------|------|
| **Bag-of-Words (BoW)** | - | 고편향, 저분산 |
| **Word Embedding** | word2vec, GloVe + LSTM/CNN | 전통적 인코더 구조 |
| **Reading Comprehension** | DocQA (GloVe) | 특화된 태스크 모델 |
| **사전학습 Transformer** | BERT-Base/Large, RoBERTa, ALBERT, DistilBERT | 대규모 사전학습된 모델 |

각 모델은 동일한 조건에서 미세조정(Fine-tuning) 되어 공정한 비교를 보장합니다.[1]

#### 3.2 평가 데이터셋 및 태스크

| 태스크 | 데이터셋 | 분포 변화 | 메트릭 |
|------|---------|---------|------|
| **감정 분석** | SST-2, IMDb, Yelp, Amazon | 리뷰 형식, 도메인, 제품 카테고리 | Accuracy |
| **의미 유사성** | STS-B | 텍스트 출처 변화 | Pearson 상관계수 |
| **읽기 이해** | ReCoRD | CNN vs. Daily Mail | Exact Match |
| **텍스트 함축** | MNLI | 텍스트 장르 변화 | Accuracy |

**STS-B 예시**:[1]
- IID: Images 분포에서 학습 후 Images에서 테스트 (성능: 91.8%)
- OOD: Images에서 학습 후 MSRvid에서 테스트 (RoBERTa: 94.3%, LSTM: 38.3%)

***

### 4. 성능 향상 결과

#### 4.1 OOD 일반화 성능 비교

**핵심 발견 1: 사전학습된 Transformer의 우수성**[1]

사전학습 Transformer 모델들은 분포 변화에 훨씬 더 견디는 성능을 보입니다:

- **IMDb 감정 분석**: LSTM(word2vec)은 9.5% 성능 저하 → BERT-Base는 4.4% 저하 → RoBERTa는 2.8% 저하[1]
- **STS-B 의미 유사성**: Average word2vec는 50.1% 저하 → RoBERTa는 0.1% 증가[1]
- **ReCoRD 읽기 이해**: LSTM은 거의 작동 불가 → RoBERTa는 0.7% 증가[1]

표 1은 MNLI의 구체적 예시입니다:[1]

| 학습 분포 | 모델 | Telephone (IID) | Letters (OOD) | Face-to-Face (OOD) |
|---------|------|----------------|---------------|--------------------|
| Telephone | BERT Base | 81.4% | 82.3% (+0.9%) | 80.8% (-0.7%) |

**핵심 발견 2: 모델 크기가 반드시 강건성을 향상시키지 않음**[1]

Figure 3에서 볼 수 있듯이, BERT 및 ALBERT 모델 크기 증가는 SST-2 → IMDb 전이에서 일반화 격차를 줄이지 못합니다. 이는 컴퓨터 비전에서의 패턴(더 큰 모델 = 더 나은 OOD 강건성)과 **다른 결과**입니다.[1]

**핵심 발견 3: 모델 증류(Distillation)의 부정적 영향**[1]

DistilBERT는 Figure 2의 ReCoRD 데이터에서 BERT-Base(53.2%)보다 성능이 낮습니다(45.0%). 이는 강건성 평가에서 모델 압축의 비용을 강조합니다.[1]

**핵심 발견 4: 다양한 사전학습 데이터의 중요성**[1]

RoBERTa는 BERT-Large보다 일관되게 우수한 OOD 성능을 보입니다. RoBERTa가 BERT보다 더 많은 데이터로 사전학습되었기 때문입니다. 이는 **데이터 다양성과 규모가 OOD 강건성 향상의 핵심 요소**임을 시사합니다.[1]

#### 4.2 OOD 탐지 성능 비교

**전통 모델의 취약성**:[1]

Figure 4에서 볼 수 있듯이, BoW, word2vec 평균, LSTM은 이상 탐지에 실패합니다:
- **20 Newsgroups**: 거의 100% FAR95 (무작위 추측과 동일)
- **Multi30K**: 대부분 90% 이상 FAR95
- 이들 모델은 OOD 예제에서 **더 높은 확신도**를 보임[1]

**사전학습 Transformer의 우수성**:[1]

| OOD 데이터셋 | LSTM word2vec | BERT-Large | 개선 |
|----------|--------------|-----------|------|
| 20 Newsgroups | 94% | 29% | 65%p |
| Multi30K | 92% | 23% | 69%p |
| RTE | 93% | 29% | 64%p |
| SNLI | 92% | 28% | 64%p |
| WMT16 | 90% | 44% | 46%p |

RoBERTa는 평균 43% FAR95를 달성하여 더욱 우수합니다.[1]

**한계**: Figure 5에서 보듯이, RoBERTa도 ID/OOD 신뢰도 분포가 겹쳐있어 완벽한 분리가 불가능합니다.[1]

***

### 5. 논문의 한계 및 미해결 문제

#### 5.1 실증적 한계

1. **모델 크기 증가의 역설**: 컴퓨터 비전에서 보이는 패턴(더 큰 모델 = 더 나은 강건성)이 NLP에서는 재현되지 않습니다. 이는 NLP와 비전의 근본적 차이를 시사하지만, 정확한 원인은 규명되지 않았습니다.[1]

2. **불완전한 OOD 탐지**: 사전학습 Transformer도 ID/OOD 경계를 명확하게 구분하지 못하며, Figure 5의 신뢰도 분포 겹침이 이를 보여줍니다.[1]

3. **제한된 분포 변화 유형**: 논문이 다루는 분포 변화는 주로 **공변량 변화(Covariate Shift)**와 **개념 변화(Concept Shift)** 조합이며, 레이블 변화(Label Shift) 등 다른 유형은 제한적으로 다루어집니다.[1]

#### 5.2 메커니즘 이해의 공백

**왜 사전학습 Transformer가 더 강건한가?**[1]

논문은 부분적 설명만 제시합니다:

1. **데이터 규모**: BERT는 3억 단어로 사전학습되었지만, GloVe는 8,400억 단어로 학습되었으므로, 단순 규모가 아님[1]

2. **데이터 다양성**: 사전학습 데이터의 다양성이 중요하지만, 정확한 기여도는 측정되지 않음[1]

3. **자가 감독 학습**: 자가 감독 목표(Masked Language Modeling)가 강건성을 높일 가능성이 제기되었으나, 체계적 검증이 부재함[1]

***

### 6. 모델 일반화 성능 향상에 대한 상세 분석

#### 6.1 일반화 격차 분석

**IID/OOD 격차(Generalization Gap) 정의**:

$$ \text{Gap} = \text{Accuracy}_{IID} - \text{Accuracy}_{OOD} $$

**주요 발견**:[1]

| 태스크 | 모델 | IID | OOD | 격차 |
|------|------|-----|-----|------|
| STS-B (Images→MSRvid) | Average BoW | 39.7% | 4.4% | 35.4%p |
| | LSTM word2vec | 81.8% | 38.3% | 43.5%p |
| | RoBERTa | 94.2% | 94.3% | -0.1%p |

**패턴**:
- 전통 모델: 40-60% 격차
- BERT 계열: 2-10% 격차
- RoBERTa: 종종 음수(OOD 성능 개선)[1]

#### 6.2 일반화 메커니즘

**가설 1: 주석 편향 회피**[1]

OOD 예제가 ID 데이터에 존재하는 허위 신호를 포함하지 않기 때문에, 이러한 신호에 의존하지 않는 모델이 더 잘 일반화합니다. 사전학습은 모델에 견고한 언어 표현을 제공합니다.

**가설 2: 다층 표현 학습**[1]

사전학습 Transformer는 여러 계층의 자기 주의(Self-Attention)를 통해 다양한 추상 수준의 특징을 학습하며, 이는 분포 변화에 대한 불변성(Invariance)을 생성합니다.

**가설 3: 대규모 데이터 효과**[1]

사전학습은 대규모 다양 데이터에 노출되어, 모델이 더 넓은 특징 공간을 커버하게 합니다. 이는 보이지 않은 분포에 대한 외삽(Extrapolation) 능력을 강화합니다.

---

### 7. 최신 연구 기반 영향 및 미래 연구 방향

#### 7.1 본 논문의 영향

이 논문은 NLP 강건성 연구의 **패러다임 전환**을 주도했습니다:[2][3]

1. **벤치마크 정립**: 이 논문의 방법론을 바탕으로 **GLUE-X**(2022)와 **BOSS**(2023) 같은 포괄적 OOD 벤치마크가 개발되었습니다. BOSS는 5개 태스크와 20개 데이터셋을 포함하며, 분포 변화의 명확성과 도전성을 보장합니다.[3][2]

2. **평가 방법론 개선**: 최신 연구는 분포 변화의 **깊이(Severity)**와 **명확성(Clarity)**을 강조하며, 이전 연구의 평가 설정이 충분히 도전적이지 않음을 지적합니다.[3]

3. **LLM 시대의 새로운 질문**: 최근 연구는 대형 언어 모델(GPT, Llama 등)의 OOD 강건성을 재평가하고 있습니다.[3]

#### 7.2 최신 연구 발견 (2023-2025)

**발견 1: 큰 모델과 작은 모델의 상반된 패턴**[3]

2023년 NeurIPS의 BOSS 연구에서는 더 깊이 있는 분석을 제시합니다:[3]

- **소규모 미세조정 모델**: ID 성능이 증가하면 OOD 성능도 증가하는 패턴을 보임
- **LLM(In-Context Learning)**: ID 데이터가 충분할 때 미세조정 모델이 우수하지만, 제한적일 때는 LLM이 우수함
- **중요한 발견**: 모든 방법이 완벽한 해결책이 아니며, 태스크와 데이터 정보에 따라 접근 방식 선택이 중요[3]

**발견 2: 이전 방법들의 한계**[3]

논문은 OOD 강건성을 위해 제안된 5가지 고전적 방법(FreeLB, GroupDRO 등)이 제한적 효과만 가진다고 보고합니다. 기본적인 미세조정이 여전히 강력한 기준선임을 강조합니다.[3]

**발견 3: 데이터 증강의 재평가**[4]

최근 연구(2025년)는 **휴리스틱 데이터 증강**이 동시 분포 변화(Concurrent Distribution Shifts) 상황에서도 최고 성능을 달성한다고 보고합니다. 이는 단순한 접근이 복잡한 최적화 기법을 능가할 수 있음을 시사합니다.[4]

**발견 4: 자가 감독 학습의 효과**[5]

Vision Transformer 연구에서 **자가 감독 학습으로 훈련된 모델**이 감독 학습 모델보다 OOD 탐지에서 더 우수한 성능을 보입니다. 이는 본 논문의 "자가 감독이 강건성 향상에 효과적일 수 있다"는 가설을 뒷받침합니다.[5]

#### 7.3 최근 OOD 탐지 방법론의 진전

최신 연구는 다양한 고급 기법을 제안합니다:[6]

1. **특징 기반 탐지 (Feature-Based Detection)**
   - Mahalanobis 거리 기반 점수
   - 에너지 점수(Energy Score)[6]
   - 다층 지식 증류 기반 앙상블[6]

2. **프로토타입 학습 (Prototype Learning)**
   - LLM과 결합된 프로토타입 손실 함수
   - 2025년 감정 분석 연구에서 효과 검증[7]

3. **맥락 학습 기반 개선**
   - In-Context Rewriting (ICR): LLM을 사용해 OOD 입력을 ID 스타일로 변환[8]
   - ROBOSHOT: 완전 제로샷 방식의 강건성 개선[9]

#### 7.4 앞으로의 연구 과제 및 고려사항

**1. 메커니즘 이해 강화 필요**

현재까지도 "왜 사전학습이 강건성을 향상시키는가"에 대한 완전한 답이 없습니다. 향후 연구는:[10][1]

- 표현 공간의 불변성(Invariance) 특성화
- 중간층 특징의 강건성 기여도 분석
- 주목(Attention) 메커니즘의 일반화 역할 규명

이를 위해 기계 해석 가능성(Mechanistic Interpretability) 분석이 필수적입니다.[10]

**2. 동시 분포 변화 대응**[4]

현실에서는 여러 분포 변화가 동시에 발생합니다(예: 스타일 + 도메인 + 어휘). 2025년 연구는 이 복합적 상황에서 데이터 증강이 가장 효과적이지만, 구체적 메커니즘은 미해명입니다.[4]

**3. 생성 모델 활용**[11]

2024년 연구는 **LLM 기반 데이터 증강**이 읽기 이해 태스크의 OOD 강건성을 현저히 향상시킨다고 보고합니다. 향후 연구는:[11]

- 생성 모델의 증강 품질 제어
- ID/OOD 분포 균형 최적화
- 계산 비용과 성능의 트레이드오프 분석

**4. 기초 모델(Foundation Model)의 OOD 강건성 재평가**

최근 LLM 등장으로 새로운 질문이 제기됩니다:[8][3]

- **크기와 강건성의 관계**: 단순히 더 큰 모델이 강건하지 않을 가능성[3]
- **제로샷 vs. 미세조정**: 서로 다른 OOD 시나리오에서 상반된 효과[3]
- **적대적 강건성 vs. OOD 강건성**: 이 두 강건성이 독립적일 가능성[8]

**5. 평가 방법론의 표준화**

BOSS(2023)는 분포 변화의 **명확성(Clarity)**, **도전성(Challenge)**, **지속성(Durability)**을 강조합니다. 향후 벤치마크는:[3]

- 더욱 현실적인 분포 변화 설계
- 다중 도메인, 다중 태스크 평가
- 장시간 성능 추적(Longitudinal Evaluation)

***

### 요약 및 결론

**Pretrained Transformers Improve Out-of-Distribution Robustness**는 NLP 모델의 강건성 평가를 체계화한 획기적 연구입니다. 주요 성과는:

1. **실증적 발견**: 사전학습 Transformer가 분포 변화에 40-65% 더 강건함을 입증
2. **평가 방법론**: 일반화와 탐지 능력을 분리하여 평가하는 프레임워크 제시
3. **실용적 교훈**: 모델 크기보다 **데이터 다양성**과 **사전학습 방식**이 더 중요

그러나 **근본적 질문들**은 여전히 미해결입니다. 2023-2025년 최신 연구는 이들 질문에 접근하면서도 새로운 복잡성을 드러내고 있습니다. 특히 생성 모델 시대에는:

- **자가 감독 학습**의 역할 재정의
- **동시 분포 변화** 대응 전략 개발
- **기초 모델 규모**와 강건성의 관계 재해석

이 모든 과제는 실제 세계 배포 환경에서 신뢰할 수 있는 NLP 시스템을 구축하기 위한 필수적 과제입니다.[11][3][1]

***

**참고 문헌:**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/037796f5-53a6-4acb-82df-5eb5a7cb8977/2004.06100v2.pdf)
[2](https://aclanthology.org/2023.findings-acl.806.pdf)
[3](https://arxiv.org/pdf/2306.04618.pdf)
[4](https://arxiv.org/html/2501.04288v1)
[5](https://epub.jku.at/download/pdf/10531915.pdf)
[6](https://par.nsf.gov/servlets/purl/10526541)
[7](https://www.sciencedirect.com/science/article/abs/pii/S0950705125000231)
[8](https://arxiv.org/html/2412.10535v1)
[9](https://snorkel.ai/research-paper/zero-shot-robustification-of-zero-shot-models-with-foundation-models-2/)
[10](https://openreview.net/forum?id=Wjgq9ISdP0)
[11](https://aclanthology.org/2024.eacl-srw.20/)
[12](https://aclanthology.org/2023.acl-short.144.pdf)
[13](https://aclanthology.org/2023.emnlp-main.360.pdf)
[14](https://arxiv.org/abs/2211.08073)
[15](https://aclanthology.org/2023.emnlp-main.276.pdf)
[16](http://arxiv.org/pdf/2208.00629.pdf)
[17](https://arxiv.org/html/2410.14899v1)
[18](https://nips.cc/virtual/2023/poster/73407)
[19](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930609.pdf)
[20](https://openaccess.thecvf.com/content/ACCV2022/papers/Sultana_Self-Distilled_Vision_Transformer_for_Domain_Generalization_ACCV_2022_paper.pdf)
[21](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/liu-2024-robust-arxiv.pdf)
[22](https://arxiv.org/html/2404.04452v1)
[23](https://arxiv.org/html/2409.11884v1)
[24](https://aclanthology.org/2024.findings-emnlp.7.pdf)
[25](https://arxiv.org/pdf/2002.08973.pdf)
[26](https://arxiv.org/pdf/2205.12753.pdf)
[27](https://arxiv.org/pdf/2312.01540.pdf)
[28](http://arxiv.org/pdf/2502.11671.pdf)
[29](https://arxiv.org/pdf/2109.01558.pdf)
[30](https://aclanthology.org/2021.findings-acl.84.pdf)
[31](https://arxiv.org/pdf/2110.09506.pdf)
[32](https://proceedings.neurips.cc/paper_files/paper/2023/file/b6b5f50a2001ad1cbccca96e693c4ab4-Paper-Datasets_and_Benchmarks.pdf)
[33](https://cvpr24-advml.github.io/long_paper/20.pdf)
[34](https://openreview.net/pdf?id=TCydh8ywpQ)
[35](https://arxiv.org/html/2511.08250v1)
[36](https://neurips.cc/virtual/2023/workshop/66517)
[37](https://arxiv.org/abs/2309.06358)
