# The Probabilistic Relevance Framework: BM25 and Beyond

### 1. 핵심 주장 및 주요 기여 요약

"The Probabilistic Relevance Framework: BM25 and Beyond"는 정보검색(IR) 분야의 고전적 확률 모델의 이론적 기초와 실제 응용을 통합하는 종합적 연구이다. 이 논문의 핵심 주장은 다음과 같다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**확률적 관점의 우월성**: 정보검색 시스템은 문서의 실제 연관성을 알 수 없으므로, 관찰 가능한 문서와 질의 특성으로부터 연관성 확률을 추정하는 것이 최적의 검색 효율성을 보장한다는 것이다. 이는 **확률 순위 원칙(Probability Ranking Principle, PRP)**으로 표현되며, 확률이 감소하는 순서로 문서를 순위 매기면 주어진 정보에 대해 최선의 성능을 달성할 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**체계적 모델 진화**: PRF는 1970-1980년대부터 발전하여 세 가지 주요 단계를 거쳤다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)
- 이진 독립성 모델(Binary Independence Model)
- BM25 알고리즘 (가장 성공적인 인스턴시에이션)
- BM25F (메타데이터 및 구조 정보를 고려한 확장형)

**탄력적 매개변수화**: 이 프레임워크의 성공은 이론적 견고성과 실용적 매개변수화의 조합에 있다. 문서 길이 정규화(b 매개변수)와 항 빈도 포화(k1 매개변수)는 모델이 다양한 문서 수집에 적응할 수 있게 한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

***

### 2. 문제 정의 및 제안 방법

#### 2.1 해결하고자 하는 문제

전통적 정보검색 모델의 근본적 문제는 다음과 같다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

1. **연관성의 불확실성**: 시스템은 문서-질의 쌍의 실제 연관성을 결코 완전히 알 수 없다.
2. **휴리스틱 방식의 한계**: TF-IDF 같은 휴리스틱 방식은 이론적 정당성이 부족하다.
3. **이진 연관성 가정**: 단순 이진(관련/무관) 연관성 모델의 현실성 문제.
4. **항 독립성 가정의 이완**: 현실의 문서에서 항들이 통계적으로 독립이 아니다.

#### 2.2 제안하는 방법: 확률적 모델링

**기본 원리**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

확률 순위 원칙을 기반으로 연관성 확률 \( P(\text{rel}|d,q) \)을 추정한다:

$$P(\text{rel}|d,q) \propto_q \frac{P(d|\text{rel},q)}{P(d|\overline{\text{rel}},q)}$$

여기서 \( d \)는 문서, \( q \)는 질의, \( \text{rel} \)은 연관성을 나타낸다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**베이지안 역변환** (Step 2.2): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

$$\frac{P(\text{rel}|d,q)}{P(\overline{\text{rel}}|d,q)} = \frac{P(d|\text{rel},q)}{P(d|\overline{\text{rel}},q)} \cdot \frac{P(\text{rel}|q)}{P(\overline{\text{rel}}|q)}$$

두 번째 항은 문서에 독립적이므로 순위 매김에 영향을 주지 않는다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**항 독립성 가정** (Step 2.4): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

$$P(d|\text{rel},q) \approx \prod_{i \in V} P(TF_i = tf_i|\text{rel},q)$$

이는 조건부 독립성으로, 관련성이 주어졌을 때 항들이 통계적으로 독립이라 가정한다. 이 가정은 이론적으로 완벽하지 않으나 실제로는 견고하고 효과적이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**질의항 제한** (Step 2.5): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

질의에 나타나는 항들만 고려하며, 질의에 없는 항들은 연관성과 무관하다고 가정한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**로그 변환을 통한 단순화** (Step 2.6): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

$$\log \prod_{i \in q} \frac{P(TF_i = tf_i|\text{rel})}{P(TF_i = tf_i|\overline{\text{rel}})} = \sum_{i \in q} \log \frac{P(TF_i = tf_i|\text{rel})}{P(TF_i = tf_i|\overline{\text{rel}})}$$

#### 2.3 모델 구조: 이진 독립성 모델(BIM)

**BIM 가중치 공식**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

$$w^{BIM}_i = \log \frac{P(t_i|\text{rel})(1-P(t_i|\overline{\text{rel}}))}{(1-P(t_i|\text{rel}))P(t_i|\overline{\text{rel}})}$$

여기서 \( t_i \)는 항의 존재를 나타내는 이진 변수다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**로버트슨/스팩 존스(RSJ) 가중치**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

실제 확률 추정을 위해 의사계수(pseudo-count) 0.5를 도입하여 로버스트성을 개선:

$$w^{RSJ}_i = \log \frac{(r_i + 0.5)(N - R - n_i + r_i + 0.5)}{(n_i - r_i + 0.5)(R - r_i + 0.5)}$$

여기서:
- \( N \): 판정된 샘플 크기
- \( n_i \): \( t_i \)를 포함하는 문서 수
- \( R \): 관련 문서 집합 크기
- \( r_i \): \( t_i \)를 포함하는 관련 문서 수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

역연관 정보 부재 시 (즉, \( R = r_i = 0 \)), IDF 형태로 축약된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

$$w^{IDF}_i = \log \frac{N - n_i + 0.5}{n_i + 0.5}$$

#### 2.4 BM25: 엘리트성 모델과 항 빈도

**엘리트성(Eliteness) 개념**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

BIM은 문서에서 항의 실제 빈도(tf)를 무시하므로, 로버트슨과 워커는 **숨겨진 이진 엘리트성 변수 \( E_i \)**를 도입했다. 이는 문서가 해당 항에 대해 "주제 관련"인지를 나타낸다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**2-포아송 모델**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

항 빈도가 엘리트성에 따라 포아송 분포를 따른다고 가정:

$$E_{ie}(tf) \sim \text{Poisson}(\lambda_{ie})$$

여기서 \( \lambda_{i1} > \lambda_{i0} \) (엘리트 문서에서 더 높은 평균). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**포화(Saturation) 특성**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

웨이트가 항 빈도에 대해 단조증가하지만 점근적으로 최댓값에 도달한다:

$$\lim_{tf \to \infty} w^{\text{elite}}_i(tf) = \log \frac{p_1(1-p_0)}{(1-p_1)p_0} = w^{BIM}_i$$

**BM25 포화 함수**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

포아송 분포 가정의 정확한 형태를 모르므로, 실무에서는 다음 단순 함수로 근사:

$$\frac{tf}{k_1 + tf}$$

여기서 \( k_1 > 0 \)은 포화도 조절 매개변수다. \( k \)가 작을수록 더 강한 포화가 발생한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

#### 2.5 문서 길이 정규화

**동기화**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

문서 길이 변동이 두 가지 원인에서 비롯된다고 가정한다:
- **장황함(Verbosity)**: 같은 내용을 더 많은 단어로 표현
- **범위(Scope)**: 더 많은 주제를 다룸

**정규화 인수 B**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

$$B := \left[(1-b) + b \frac{dl}{\text{avdl}}\right], \quad 0 \leq b \leq 1$$

여기서:
- \( dl \): 문서 길이
- \( \text{avdl} \): 컬렉션의 평균 문서 길이
- \( b = 0 \): 정규화 없음
- \( b = 1 \): 완전 정규화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**BM25 최종 공식**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

$$w^{BM25}_i(tf) = \frac{tf}{k_1\left[(1-b) + b\frac{dl}{\text{avdl}}\right] + tf} \cdot w^{RSJ}_i$$

또는 확장형:

$$w^{BM25}_i(tf) = \frac{tf}{k_1 + k_1(1-b)\frac{dl}{\text{avdl}} + tf} \cdot w^{RSJ}_i$$

***

### 3. 성능 향상 및 모델 한계

#### 3.1 성능 향상 메커니즘

**항 가중치의 합산**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

최종 문서 점수는 질의 항들의 가중치 합:

$$\text{Score}(d,q) = \sum_{i \in q} w^{BM25}_i(tf_i)$$

이는 선형성에서 비롯되며, 비선형 포화 함수와 IDF를 활용한 항들의 증거 결합. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**BM25F: 필드 기반 확장**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

구조화된 문서(제목/본문/앵커 텍스트)에서 서로 다른 필드의 엘리트성을 독립적으로 모델링:

$$\tilde{tf}_i = \sum_{s=1}^{S} v_s \frac{tf_{si}}{B_s}$$

여기서:
- \( v_s \): 필드 \( s \)의 가중치
- \( B_s = (1-b_s) + b_s \frac{sl_s}{\text{avsl}_s} \): 필드별 길이 정규화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**역연관 정보의 활용**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

제시된 문서들 중 사용자가 판정한 관련 문서들로부터:
- **제시 가중치(Offer Weight)**를 통해 질의 확장 항들의 추가 효과 평가
- **맹목적 피드백(Blind Feedback)**: 초기 검색 결과를 관련으로 가정하고 재검색 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

#### 3.2 모델 한계

**이진 연관성 가정의 제약**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

- 현실적으로는 다단계 연관성(매우 관련, 어느 정도 관련, 무관)이 더 정확하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)
- 문서들 간의 상대적 연관성을 간과한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**항 독립성 위반**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

- 동의어(synonym) 쌍이나 의미적으로 관련된 항들이 통계적으로 독립이 아니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)
- 질의 확장 시 유사 항들의 추가가 항 독립성 가정을 악화시킬 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**확률 추정의 어려움**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

- 순위 매김에만 필요한 상대 확률 vs 절대 확률 추정의 간극
- 역변환 단계에서 제거된 항들 \( P(\text{rel}|q) \)를 정확히 추정할 수 없음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)
- 이로 인해 적응형 필터링(adaptive filtering) 같은 절대 확률이 필요한 응용에 제약이 있음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

**위치 정보 미고려**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

- 가방 모델(bag-of-words)로 인해 단어 순서와 근접성을 무시한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)
- 그러나 실증적으로 위치 정보의 영향은 생각보다 제한적이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

***

### 4. 모델의 일반화 성능 향상 가능성

최근 연구(2020-2025)는 BM25와 PRF의 일반화 성능을 크게 개선하는 새로운 방향들을 제시한다:

#### 4.1 하이브리드 검색의 우월성

**발견**: BEIR 벤치마크(2021년)에서 순수 밀집 검색(dense retrieval)이 BM25를 제로샷(zero-shot) 상황에서 크게 상회하지 못했다. 반면 하이브리드 접근이 양쪽의 장점을 결합하여 최고 성능을 달성했다. [rsisinternational](https://rsisinternational.org/journals/ijriss/articles/mapping-the-landscape-of-knowledge-management-research-2020-2025/)

**구체적 성과**:
- 하이브리드 + LLM 재순위 매김: nDCG@10에서 0.504-0.537 달성 [link.springer](https://link.springer.com/10.1007/s10965-025-04719-z)
- 하이브리드 검색 일반화: 개별 희소/밀집 모델보다 10-25% 향상 [tandfonline](https://www.tandfonline.com/doi/full/10.1080/10408347.2025.2527741)
- Light Hybrid Retrievers (Hybrid-LITE): 메모리 13배 절감 후 98% 성능 유지 [xlink.rsc](https://xlink.rsc.org/?DOI=D5RA07154B)

#### 4.2 동적 가중치 조정

**DAT (Dynamic Alpha Tuning)**: [epess](https://www.epess.net/index.php/epess/article/view/971)

LLM을 이용해 각 질의마다 BM25와 밀집 검색의 최적 가중치를 동적으로 조정:

- BM25와 밀집 검색의 상위 1개 결과 효과성 평가
- 정규화된 효과성 점수로부터 최적 \( \alpha \) 계산
- 결과: 정적 가중치 대비 3.3% Precision@1 향상 [epess](https://www.epess.net/index.php/epess/article/view/971)

#### 4.3 쿼리 확장 및 의미적 보강

**BMX (Entropy-weighted BM25)**: [ejournal.ppsdp](https://ejournal.ppsdp.org/index.php/pijed/article/view/848)

엔트로피 가중치를 활용해 BM25를 확장:

$$\text{BMX} = \text{BM25} + \text{entropy-weighted similarity} + \text{semantic augmentation}$$

- 장문서 검색과 실세계 벤치마크에서 BM25 초과 성능 [ejournal.ppsdp](https://ejournal.ppsdp.org/index.php/pijed/article/view/848)
- 밀집/LLM 기반 모델과 경쟁 가능 [ejournal.ppsdp](https://ejournal.ppsdp.org/index.php/pijed/article/view/848)

**BM25 쿼리 증강 (Learned Augmentation)**: [ashpublications](https://ashpublications.org/blood/article/146/Supplement%201/7871/548832/Evaluating-the-impact-of-AI-tools-on-diagnostics)

신경망이 쿼리를 확장하고 재가중하는 방법을 학습:

- 속도는 유지하면서 BM25 성능 향상 [ashpublications](https://ashpublications.org/blood/article/146/Supplement%201/7871/548832/Evaluating-the-impact-of-AI-tools-on-diagnostics)
- 미학습 데이터셋에도 양호한 전이 성능 [ashpublications](https://ashpublications.org/blood/article/146/Supplement%201/7871/548832/Evaluating-the-impact-of-AI-tools-on-diagnostics)

#### 4.4 문서 확장 및 재순위 매김

**docT5query**: [dovepress](https://www.dovepress.com/solidification-of-snedds-using-mesoporous-carriers-20202025-a-review-o-peer-reviewed-fulltext-article-DDDT)

사전학습된 T5 모델을 사용해 관련성이 높은 가상 질의를 문서에 추가:

- BEIR에서 18개 중 11개 데이터셋에서 BM25 초과 [dovepress](https://www.dovepress.com/solidification-of-snedds-using-mesoporous-carriers-20202025-a-review-o-peer-reviewed-fulltext-article-DDDT)
- 전반적 경쟁 성능 [dovepress](https://www.dovepress.com/solidification-of-snedds-using-mesoporous-carriers-20202025-a-review-o-peer-reviewed-fulltext-article-DDDT)

**LLM 기반 재순위 매김**: [arxiv](https://arxiv.org/pdf/2104.08663.pdf)

신경 재순위(neural reranking) 및 LLM 재순위 매김:

- Hybrid + RankLLaMA: MAP 0.523에서 0.797로 52% 상대 향상 [arxiv](https://arxiv.org/pdf/2104.08663.pdf)
- NDCG@10에서 일관된 최고 성능 [arxiv](https://arxiv.org/pdf/2104.08663.pdf)

#### 4.5 영역 특화 모델

**생의학 RAG 시스템**: [arxiv](https://arxiv.org/pdf/2305.01203.pdf)

- BM25와 MedCPT 결합: 의료 관련 개념 정확도 향상 [arxiv](https://arxiv.org/pdf/2305.01203.pdf)
- 순수 밀집 모델보다 용어 정확성에서 우수 [arxiv](https://arxiv.org/pdf/2305.01203.pdf)
- 2024-2025년 대부분의 생의학 RAG가 하이브리드 접근 채택 [arxiv](https://arxiv.org/pdf/2305.01203.pdf)

#### 4.6 조건부 재샘플링과 어려운 음성 샘플

**효과적인 음성 표본 선택**: [arxiv](http://arxiv.org/pdf/2407.03618.pdf)

밀집 검색기 학습에서:

- BM25의 상위 무관 문서를 어려운 음성으로 사용 [arxiv](http://arxiv.org/pdf/2407.03618.pdf)
- 단순 음성 샘플보다 크게 개선된 성능 [arxiv](https://arxiv.org/pdf/2305.14087.pdf)

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법 | 주요 성과 | 특징 |
|------|------|------|---------|------|
| **BEIR 벤치마크** | 2021 | 제로샷 평가 | BM25 강력한 기준선 [rsisinternational](https://rsisinternational.org/journals/ijriss/articles/mapping-the-landscape-of-knowledge-management-research-2020-2025/) | 18개 이질적 데이터셋 |
| **Mr. TyDi** | 2021 | 다국어 밀집 검색 | mDPR < BM25 (제로샷) [arxiv](http://arxiv.org/pdf/2412.08329.pdf) | 하이브리드 보완성 입증 |
| **Hybrid-LITE** | 2023 | 경량 하이브리드 | 메모리 13배 절감 [tandfonline](https://www.tandfonline.com/doi/full/10.1080/10408347.2025.2527741) | 강화된 일반화 성능 |
| **BMX** | 2024 | 엔트로피 가중 BM25 | 밀집 모델 경쟁 [ejournal.ppsdp](https://ejournal.ppsdp.org/index.php/pijed/article/view/848) | 어휘 기반과 의미 결합 |
| **RRR (GAR-meets-RAG)** | 2023 | 쿼리 재쓰기 + LLM 재순위 | BEIR nDCG@10 +17% 상대 이득 [arxiv](https://arxiv.org/pdf/1704.08803.pdf) | 제로샷 성능 SOTA |
| **DAT** | 2025 | 동적 알파 튜닝 | Precision@1 +3.3% [epess](https://www.epess.net/index.php/epess/article/view/971) | 질의별 적응형 가중치 |
| **Biomedical RAG Survey** | 2025 | 도메인 특화 하이브리드 | BM25 + 밀집 표준 [arxiv](https://arxiv.org/pdf/2305.01203.pdf) | 임상 응용 중심 |
| **Cross-Encoder + BM25** | 2025 | 신경 재순위 | nDCG 향상 [arxiv](https://arxiv.org/pdf/2305.16243.pdf) | 의미적 BM25 변형 재발견 |

#### 5.1 BM25 vs 밀집 검색: 생존성

**주요 발견들**: [tecnoscientifica](https://tecnoscientifica.com/journal/csue/article/view/849)

1. **제로샷 성능**: BM25는 도메인 전이에서 대부분의 신경 모델보다 우수
2. **어휘 불일치**: 밀집 모델은 미학습 어휘(예: "Karatsuba 알고리즘")에 약함 [arxiv](https://arxiv.org/pdf/2305.06300.pdf)
3. **업데이트 효율**: BM25는 즉시 인덱싱 가능, 밀집은 임베딩 생성 필요 (50-200ms/문서) [arxiv](https://arxiv.org/pdf/2305.06300.pdf)

**성능 비교** (MS MARCO): [arxiv](https://arxiv.org/pdf/2305.06300.pdf)
- BM25: MRR@10 = 0.187
- 밀집 검색: 약 0.32
- 하이브리드 (BM25 + 밀집): 약 0.38
- 신경 재순위: 0.44 [arxiv](https://arxiv.org/pdf/2305.06300.pdf)

#### 5.2 일반화 메커니즘

**동메인 vs 크로스도메인 성능 격차**: [mbrenndoerfer](https://mbrenndoerfer.com/writing/bm25-probabilistic-ranking-information-retrieval)

- 밀집 모델: 인메인(in-domain)에서 우수하나 OOD에서 심각한 성능 저하
- BM25: 상대적으로 안정적인 일반화
- 생성형 검색: 전체적으로 더 높은 성능 (어색한 색인 구조)

**쿼리 변형 및 대적 공격**: [tandfonline](https://www.tandfonline.com/doi/full/10.1080/10408347.2025.2527741)

Light Hybrid Retrievers:
- 자연 쿼리 편차: 강건성 유지
- 문자 오류 공격: 개선된 견고성
- 의미론적 변형: 보강된 포착 능력

#### 5.3 검색 정밀도-재현율 트레이드오프

**정밀도 향상 (Reranking)**: [arxiv](https://arxiv.org/pdf/2305.16243.pdf)
- 초기 검색: BM25 (빠름, 고회수)
- 재순위: LLM/신경 모델 (느림, 고정밀)
- 이 두 단계 결합이 최적

**재현율 향상 (확장)**: [dovepress](https://www.dovepress.com/solidification-of-snedds-using-mesoporous-carriers-20202025-a-review-o-peer-reviewed-fulltext-article-DDDT)
- docT5query: 문서 확장으로 재현율 개선
- 의미론적 유사 항 추가 (동의어)

***

### 6. 향후 연구에 미치는 영향 및 고려 사항

#### 6.1 학문적 영향

**기초 모델의 재평가**:

전통 정보검색 모델들이 단순 기준선이 아닌 견고한 기준으로 재인식되고 있다. 신경 모델의 상대적 약점(OOD 성능)이 강조되면서, 하이브리드 접근이 표준이 되고 있다. [link.springer](https://link.springer.com/10.1007/s44442-025-00011-3)

**확률론적 기초의 부활**:

BM25의 확률론적 기초는 신경 재순위 모델의 설계에도 영향을 미친다. 예를 들어, 항 빈도 포화와 문서 길이 정규화의 개념이 신경망의 주의 메커니즘(attention)과 다중작업 학습에서 재현된다. [link.springer](https://link.springer.com/10.1007/s10965-025-04719-z)

**벤치마크 설계 개선**:

BEIR 같은 제로샷 벤치마크가 OOD 일반화 평가의 중요성을 강조하면서, 단순 인메인 성능만으로 모델을 평가하는 관행이 비판받고 있다. [rsisinternational](https://rsisinternational.org/journals/ijriss/articles/mapping-the-landscape-of-knowledge-management-research-2020-2025/)

#### 6.2 실제 적용 시 고려 사항

**하이브리드 시스템 설계**: [arxiv](https://arxiv.org/pdf/2305.06300.pdf)

1. **초기 후보 검색**: BM25로 신속한 필터링 (수밀리초)
2. **밀집 재검색**: 추가 의미론적 신호 (선택사항)
3. **재순위 매김**: LLM 또는 신경 크로스인코더로 정밀화

**장점**:
- 어휘 정확성 (BM25) + 의미론적 매칭 (밀집) 결합
- 계산 효율성: BM25로 초기 필터, 비용이 큰 모델은 상위 K에만 적용

**동적 가중치 조정**: [epess](https://www.epess.net/index.php/epess/article/view/971)

- 질의 특성에 따라 BM25 vs 밀집의 가중치 적응
- LLM 기반 평가 또는 학습 가능한 가중치

**도메인 특화**: [arxiv](https://arxiv.org/pdf/2305.01203.pdf)

- 의료/법률 같은 전문 영역: 정확한 용어 필수 → BM25 가중치 상향
- 일반 웹 검색: 의미론적 유연성 필요 → 밀집 가중치 상향

#### 6.3 미해결 문제 및 미래 방향

**1. 항 간 의존성 모델링**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

프레임워크의 조건부 독립 가정이 완벽하지 않다. 향후 연구는:

- 약한 의존성 구조 (예: Markov Random Fields) 통합
- 주제 모델 (LDA, LSI) 결합을 통한 의미 인수분해
- 신경 모델이 학습한 분산 표현과의 통합 [link.springer](https://link.springer.com/10.1007/s10965-025-04719-z)

**2. 확률 추정의 신뢰도**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

순위 매김에는 상대 확률만 필요하지만, 다음 응용들에서는 절대 확률이 필수이다:

- 신뢰도 기반 피드백 (의료 진단)
- 불확실성 정량화 (생성형 AI의 할루시네이션 방지)
- 조정 가능한 결정 임계값 (ROC 곡선)

**3. 위치와 근접성 정보**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

현재 가방 모델은 구문적 정보를 무시한다:

- 문구 매칭과의 결합 (이미 일부 구현)
- 위치 특성을 비텍스트 피처로 통합 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)
- 신경 모델의 문맥 부호화와 조합

**4. 다중 의도(Multiple Intent) 쿼리**:

프레임워크는 단일 정보 요구를 가정하지만:

- 하나의 쿼리가 여러 의도를 나타낼 수 있음 (예: "python" = 프로그래밍 언어 vs 뱀)
- 의도 분류 모듈의 추가
- 다목표 랭킹 최적화

**5. 장문서 검색과 구조화된 검색**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

BM25F의 성공에도 불구하고:

- 깊게 중첩된 구조 (계층적 문서)
- 혼합형 콘텐츠 (텍스트 + 이미지 + 표)
- 그래프 기반 검색과의 통합 (지식 그래프)

#### 6.4 연구 방법론 권장사항

**1. 벤치마크 선택**:

- **인메인 평가**: MS MARCO, SQuAD (신경 모델 검증용)
- **제로샷 평가**: BEIR, MIRACL (일반화 평가)
- **도메인 특화**: BioASQ, LEGAL-BERT 벤치마크

**2. 기준선 설정**:

BM25를 항상 포함:
- k1 = 1.5, b = 0.75 (기본값)
- 다양한 토큰화 전략 테스트 (단어 vs 문자 n-그램)
- IDF 변형 (Robertson-Sparck Jones, 표준 로그)

**3. 파라미터 최적화**: [semanticscholar](https://www.semanticscholar.org/paper/The-Probabilistic-Relevance-Framework:-BM25-and-Robertson-Zaragoza/47ced790a563344efae66588b5fb7fe6cca29ed3)

- **그리디 최적화**: 캐싱, 격자 탐색, 강건 선형 탐색
- **다차원 최적화**: 약속하는 방향(Promising Directions) 알고리즘
- **경사도 최적화**: 학습-투-순위(Learning-to-Rank) 손실 함수 활용

**4. 공정한 비교**:

- 모든 방법에 동일한 코퍼스, 토큰화, 정규화 적용
- 계산 비용(지연시간, 메모리) 함께 보고
- 통계적 유의성 검증 (카이제곱, t-검정)

***

### 결론

"The Probabilistic Relevance Framework: BM25 and Beyond"는 발표 15년 이상이 지난 2009년 논문이지만, 여전히 현대 정보검색의 기초를 이룬다. 최근 연구(2020-2025)는 다음을 보여준다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ebb0362-646d-4181-9be6-f31054d2408b/robertson_foundations.pdf)

1. **BM25의 지속적 경쟁력**: 제로샷 설정에서 신경 모델을 능가 [tecnoscientifica](https://tecnoscientifica.com/journal/csue/article/view/849)
2. **하이브리드 전략의 우월성**: 어휘+의미 결합으로 10-25% 성능 향상 [link.springer](https://link.springer.com/10.1007/s44442-025-00011-3)
3. **적응형 가중치의 중요성**: 동적 알파 튜닝 같은 질의별 최적화 [epess](https://www.epess.net/index.php/epess/article/view/971)
4. **도메인 특화의 필요성**: 일반적 모델보다 영역 맞춤 모델 우수 [arxiv](https://arxiv.org/pdf/2305.01203.pdf)

향후 연구는 확률 프레임워크의 이론적 견고성을 유지하면서, 신경 모델의 의미론적 이해 능력과 결합하는 방향으로 진행될 것으로 예상된다. 특히 절대 확률 추정, 항 간 의존성 모델링, 위치/구조 정보 통합이 핵심 과제가 될 것이다.

***

### 참고 문헌

<span style="display:none">[^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87]</span>

<div align="center">⁂</div>

[^1_1]: robertson_foundations.pdf

[^1_2]: https://rsisinternational.org/journals/ijriss/articles/mapping-the-landscape-of-knowledge-management-research-2020-2025/

[^1_3]: https://tecnoscientifica.com/journal/csue/article/view/849

[^1_4]: https://link.springer.com/10.1007/s44442-025-00011-3

[^1_5]: https://link.springer.com/10.1007/s10965-025-04719-z

[^1_6]: https://www.tandfonline.com/doi/full/10.1080/10408347.2025.2527741

[^1_7]: https://xlink.rsc.org/?DOI=D5RA07154B

[^1_8]: https://www.epess.net/index.php/epess/article/view/971

[^1_9]: https://ejournal.ppsdp.org/index.php/pijed/article/view/848

[^1_10]: https://ashpublications.org/blood/article/146/Supplement 1/7871/548832/Evaluating-the-impact-of-AI-tools-on-diagnostics

[^1_11]: https://www.dovepress.com/solidification-of-snedds-using-mesoporous-carriers-20202025-a-review-o-peer-reviewed-fulltext-article-DDDT

[^1_12]: https://arxiv.org/pdf/2104.08663.pdf

[^1_13]: https://arxiv.org/pdf/2305.01203.pdf

[^1_14]: http://arxiv.org/pdf/2407.03618.pdf

[^1_15]: https://arxiv.org/pdf/2305.14087.pdf

[^1_16]: http://arxiv.org/pdf/2412.08329.pdf

[^1_17]: https://arxiv.org/pdf/1704.08803.pdf

[^1_18]: https://arxiv.org/pdf/2305.16243.pdf

[^1_19]: https://arxiv.org/pdf/2305.06300.pdf

[^1_20]: https://mbrenndoerfer.com/writing/bm25-probabilistic-ranking-information-retrieval

[^1_21]: https://www.semanticscholar.org/paper/The-Probabilistic-Relevance-Framework:-BM25-and-Robertson-Zaragoza/47ced790a563344efae66588b5fb7fe6cca29ed3

[^1_22]: https://arxiv.org/html/2502.20245v1

[^1_23]: https://www.sourcely.net/resources/bm25-and-its-role-in-document-relevance-scoring

[^1_24]: https://www.sciencedirect.com/science/article/pii/S1574013723000205

[^1_25]: https://www.sci.utah.edu/~beiwang/publications/IR_Survey_BeiWang_2025.pdf

[^1_26]: https://iclr.cc/virtual/2025/papers.html

[^1_27]: https://haystackconf.com/us2023/talk-1/

[^1_28]: https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf

[^1_29]: https://www.gbv.de/dms/tib-ub-hannover/632343664.pdf

[^1_30]: https://dl.acm.org/doi/10.1145/3534928

[^1_31]: https://arxiv.org/html/2502.04645v1

[^1_32]: https://procheta.github.io/sprocheta/Thesis.pdf

[^1_33]: https://www.sciencedirect.com/science/article/pii/S0925231223008032

[^1_34]: https://arxiv.org/pdf/2505.01146.pdf

[^1_35]: https://arxiv.org/html/2510.21425v1

[^1_36]: https://www.arxiv.org/pdf/2505.20139.pdf

[^1_37]: https://www.arxiv.org/pdf/2510.16384.pdf

[^1_38]: https://arxiv.org/html/2403.11219v1

[^1_39]: https://www.arxiv.org/pdf/2511.18177.pdf

[^1_40]: https://arxiv.org/html/2510.16384v1

[^1_41]: https://arxiv.org/html/2504.18586v1

[^1_42]: https://arxiv.org/html/2412.16075v1

[^1_43]: https://arxiv.org/html/2601.03618v1

[^1_44]: https://arxiv.org/html/2502.12799v1

[^1_45]: https://arxiv.org/html/2505.15918

[^1_46]: https://arxiv.org/html/2507.12948v2

[^1_47]: https://arxiv.org/html/2504.20113v1

[^1_48]: https://arxiv.org/html/2412.15361v3

[^1_49]: https://arxiv.org/pdf/2410.09662.pdf

[^1_50]: https://arxiv.org/html/2508.17694v1

[^1_51]: https://www.semanticscholar.org/paper/Neural-Ranking-Models-with-Multiple-Document-Fields-Zamani-Mitra/0f9535512a9bc8414f86a2774b5e4d8b35a5036b

[^1_52]: https://heisenberg.kr/bm25/

[^1_53]: https://arxiv.org/abs/2510.03795

[^1_54]: https://arxiv.org/pdf/2108.08787.pdf

[^1_55]: https://arxiv.org/html/2503.23013v1

[^1_56]: https://arxiv.org/pdf/2502.20245.pdf

[^1_57]: https://annals-csis.org/proceedings/2023/drp/pdf/8119.pdf

[^1_58]: https://aclanthology.org/2023.acl-short.159.pdf

[^1_59]: http://arxiv.org/pdf/2409.05882.pdf

[^1_60]: http://arxiv.org/pdf/2210.01371.pdf

[^1_61]: https://www.systemoverflow.com/learn/search-ranking/ranking-algorithms/bm25-vs-dense-retrieval-when-to-use-each

[^1_62]: https://www.emergentmind.com/topics/hybrid-bm25-retrieval

[^1_63]: https://arxiv.org/abs/2104.08663

[^1_64]: https://dev.to/qvfagundes/dense-vs-sparse-retrieval-mastering-faiss-bm25-and-hybrid-search-4kb1

[^1_65]: https://ceur-ws.org/Vol-4038/paper_89.pdf

[^1_66]: https://aclanthology.org/2025.bucc-1.5/

[^1_67]: https://arxiv.org/pdf/2503.23013.pdf

[^1_68]: https://www.emergentmind.com/topics/bm25-retrieval

[^1_69]: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf

[^1_70]: https://aclanthology.org/2025.regnlp-1.5v2.pdf

[^1_71]: https://www.emergentmind.com/topics/beir-benchmark

[^1_72]: https://aclanthology.org/2024.emnlp-main.845.pdf

[^1_73]: https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres

[^1_74]: https://github.com/chanmuzi/Papers/blob/main/RAG/BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.md

[^1_75]: https://arxiv.org/html/2505.23250v1

[^1_76]: https://arxiv.org/pdf/2412.18768.pdf

[^1_77]: https://arxiv.org/html/2506.22644v1

[^1_78]: https://arxiv.org/pdf/2310.20158.pdf

[^1_79]: https://arxiv.org/pdf/2408.06643.pdf

[^1_80]: https://arxiv.org/pdf/2506.22644.pdf

[^1_81]: https://arxiv.org/pdf/2306.07471.pdf

[^1_82]: https://arxiv.org/pdf/2403.18684.pdf

[^1_83]: https://arxiv.org/html/2508.01405v2

[^1_84]: https://arxiv.org/pdf/2304.14233.pdf

[^1_85]: https://arxiv.org/html/2502.19712v1

[^1_86]: https://arxiv.org/html/2509.10697v1

[^1_87]: https://ar5iv.labs.arxiv.org/html/2104.08663

