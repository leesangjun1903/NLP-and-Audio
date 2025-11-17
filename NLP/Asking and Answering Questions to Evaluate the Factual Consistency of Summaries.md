# Asking and Answering Questions to Evaluate the Factual Consistency of Summaries

### 1. 핵심 주장과 주요 기여

**QAGS(Question Answering and Generation for Summarization)** 논문의 핵심 주장은 **질문 생성과 질답을 활용하면 기존의 n-gram 기반 평가 지표보다 생성된 요약의 사실적 일관성을 훨씬 더 정확하게 평가할 수 있다**는 것입니다.[1]

논문의 주요 기여는 다음과 같습니다:[1]

1. **자동 평가 메트릭 QAGS 제안**: 모델이 생성한 요약 텍스트의 사실적 불일치를 자동으로 감지할 수 있는 평가 프로토콜 개발

2. **인간 판단 데이터 수집**: CNN/DailyMail과 XSUM 두 개 요약 데이터셋에서 인간이 평가한 사실적 일관성 판단 데이터 구축

3. **우월한 상관관계 입증**: QAGS는 ROUGE-2(상관계수 17.72)에 비해 CNN/DailyMail에서 54.52의 피어슨 상관계수를 달성하여 기존 메트릭을 크게 능가

4. **해석가능성 제공**: 생성된 질문과 답변을 통해 요약의 어느 부분이 불일치하고 그 이유가 무엇인지 명확하게 보여줌

5. **강건성 입증**: 모델 품질, 도메인 이동, 질문 개수 등 여러 요인에 대한 탈션 연구로 QAGS의 견고성 확인

---

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능 분석

#### 2.1 해결하고자 하는 문제

추상적 요약 모델의 주요 문제점은 **사실적 불일치(factual inconsistency)** 입니다. 기존 자동 평가 지표의 문제점:[1]

- **n-gram 기반 지표의 한계**: ROUGE, BLEU, METEOR 등은 모든 n-gram을 동등하게 가중치를 부여하여 의미적 오류에 민감하지 않음
- **참조 텍스트 의존성**: 참조 요약이 필수적이나, 고엔트로피 생성 작업에서 단일 참조는 부족함
- **의미적 오류 검출 불가**: "나는 밴쿠버에서 논문을 쓰고 있다"와 "나는 밴쿠버에서 논문을 쓰고 있지 않다"는 거의 모든 유니그램과 바이그램을 공유하지만 의미는 정반대임[1]

#### 2.2 제안하는 방법 및 수식

QAGS의 핵심 프레임워크는 다음 확률 모델을 기반으로 합니다:[1]

$$
E_{Q \sim p(Q|Y)} \left[ D \left( p(A|Q, X), p(A|Q, Y) \right) \right]
\tag{1}
$$

여기서:
- $$X$$: 원본 텍스트
- $$Y$$: 요약 텍스트
- $$p(Q|Y)$$: 요약 Y에 대한 질문의 확률 분포
- $$p(A|Q, X)$$: 원본 X에 주어진 질문 Q에 대한 답변의 확률 분포
- $$p(A|Q, Y)$$: 요약 Y에 주어진 질문 Q에 대한 답변의 확률 분포
- $$D$$: 두 답변 분포의 유사성을 측정하는 함수

이 식이 최대화되면 Y가 X의 정보의 부분집합이면서 두 텍스트에서 유사한 답변을 생성하는 경우, 즉 Y가 X와 사실적으로 일치하는 경우입니다.[1]

#### 2.3 QAGS 점수 계산 방법

실제 구현에서 QAGS는 다음 네 단계를 거칩니다:[1]

**1단계: 질문 생성(Question Generation)**
- spaCy를 사용하여 요약에서 명명된 개체와 명사구를 추출
- 답변 조건부 QG 모델(BART)을 NewsQA에서 fine-tuning
- Beam search(너비 10)를 사용하여 100개 질문 후보 생성
- 휴리스틱 기반 필터링 적용: 짧은 질문, 중복 제거, 답변 불가능 질문 필터링
- K=20개의 가장 확률 높은 질문 선택

**2단계: 질답(Question Answering)**
- BERT를 SQuAD 2.0에서 fine-tuning한 추출식 QA 모델 사용
- 원본 텍스트(X)와 요약(Y) 모두에 대해 동일한 질문에 답변

**3단계: 답변 유사도 계산**
- 토큰 수준 F1 점수로 답변 비교:

$$
F1(\arg\max p(A|Q, X), \arg\max p(A|Q, Y))
$$

**4단계: 최종 점수**
- 모든 질문에 대한 F1 점수의 평균을 QAGS 점수로 계산[1]

#### 2.4 모델 구조

**질문 생성 모델 (QG):**
- 기반 모델: BART-large (사전학습)
- 훈련 데이터: NewsQA
- 입력: [원본 기사] [특수 토큰] [답변] [특수 토큰] [질문]
- 학습: 답변 조건부 방식으로 label smoothed cross entropy (smoothing=0.1) 최적화
- 디코딩: Beam search (크기 10, 길이 페널티 1.0, 삼중어 반복 차단)[1]

**질답 모델 (QA):**
- 기반 모델: BERT-large-uncased (사전학습)
- 훈련 데이터: SQuAD 2.0
- 입력: [질문] [특수 토큰] [문맥]
- 최적화: AdamW (초기 학습률 5e-5, 3 epoch, warmup 비율 0.1)[1]

#### 2.5 성능 향상

**정량적 성능 개선:**

| 메트릭 | CNN/DM | XSUM |
|--------|--------|------|
| ROUGE-1 | 28.74 | 13.22 |
| ROUGE-2 | 17.72 | 8.95 |
| ROUGE-L | 24.09 | 8.86 |
| METEOR | 26.65 | 10.03 |
| BERTScore | 27.63 | 2.51 |
| **QAGS** | **54.53** | **17.49** |

QAGS는 기존 최고 성능 메트릭 대비 **거의 2배의 상관계수** 달성[1]

**재순위 지정 작업에서의 성능:**
- Random: 50.0%
- BERT NLI: 64.1%
- ESIM: 67.6%
- FactCC: 70.0%
- **QAGS: 72.1%** (새로운 SOTA)[1]

#### 2.6 논문의 한계

1. **질문 품질 문제**: 8.75%의 비논리적 질문, 3.00%의 형식적이지만 요약에 대해 답변 불가능한 질문
2. **QA 모델의 전이 문제**: 원본 텍스트에 대해 32.50%의 질문이 잘못 답변됨
3. **어휘 변이 감지 실패**: 8.00%의 경우 같은 의미의 답변이 F1 점수로 구분되지 못함
4. **추출식 QA 한계**: 의역된 답변 매칭 불가능
5. **도메인 이동**: SQuAD-tuned QG 사용 시 CNN/DM에서 54.53에서 51.53으로 성능 저하
6. **적용 범위 제한**: 추출식 요약에서는 거의 완벽한 점수 얻을 가능성 높음[1]

---

### 3. 일반화 성능 향상 가능성에 대한 분석

#### 3.1 강건성 분석 결과

**QA 모델 품질의 영향:**
논문에서는 서로 다른 BERT 변종(bert-base, bert-large, bert-large-wwm)을 사용한 탈션 연구를 수행했습니다. 결과는 놀랍게도:[1]
- SQuAD F1: 75.95 → 84.36으로 크게 향상되었음에도
- CNN/DM 상관계수: 55.20 → 51.36으로 오히려 감소
- XSUM은 변이가 매우 작음 (20.71 → 18.07)

이는 **더 강력한 QA 모델이 반드시 더 나은 평가 지표를 만들지 않으며, 일정 수준 이상의 QA 모델이면 충분**함을 시사합니다.[1]

**질문 생성 모델 품질의 영향:**
NewQA 개발 세트에서 perplexity를 변화시킨 결과:[1]
- 5.48 (최고): CNN/DM 54.53, XSUM 17.49
- 18.56 (최악): CNN/DM 47.92, XSUM 16.38
- CNN/DM에서는 9.7% 성능 저하
- XSUM에서는 6.4% 성능 저하로 오히려 증가

**도메인 이동 효과:**
NewsQA vs SQuAD (다른 도메인) 비교:[1]
- CNN/DM: 54.53 → 51.53 (5.5% 저하)
- XSUM: 17.49 → 15.28 (12.6% 저하)

**약한 QG 모델 사용 시에도** 다른 메트릭을 능가함을 보였고, 이는 일정 수준의 domain mismatch는 수용 가능함을 의미합니다.[1]

**질문 개수의 영향:**
질문 개수를 5개부터 50개까지 변화시킨 결과:[1]

| 질문 수 | CNN/DM | XSUM |
|--------|--------|------|
| 5 | 41.61 | 15.63 |
| 10 | 41.17 | 15.49 |
| 20 | 54.53 | 17.49 |
| 50 | 57.94 | 17.74 |

- 5→20개에서 큰 향상 (31% 개선)
- 20→50개에서는 감소 추세 (한계 효율 감소)

#### 3.2 일반화 성능 향상을 위한 함의

**강건성 기반 일반화 가능성:**
1. **모델 선택의 자유도**: 일정 수준 이상의 표준 QA/QG 모델(BERT, BART)이면 충분하여 고비용 모델 불필요
2. **도메인 적응 가능성**: Domain mismatch가 있어도 충분히 견고한 성능 유지 (5-13% 저하로 제한)
3. **계산 효율**: K=20개 질문으로 대부분의 이점 확보 가능으로 계산 비용 관리 가능[1]

**한계 및 개선 필요 영역:**
논문의 에러 분석에 따르면:[1]
1. **질문 생성 오류 (11.75%)**: 비논리적 또는 답변 불가능한 질문
2. **QA 모델 전이 오류 (32.50%)**: 원본에 대한 높은 오류율
3. **의미적 중복 검출 실패 (8%)**: 어휘 변이로 인한 위음성

이는 **더 강력한 QA 모델, 더 나은 답변 유사도 메트릭(NLI 기반 등), 추상적 QA의 필요**를 시사합니다.[1]

***

### 4. 이후 연구에 미치는 영향 및 고려사항

#### 4.1 후속 연구 및 개선 논문들

**직접적인 개선 연구:**

**QAFactEval (2022)**: QAGS의 직접적 후속 연구로, 다음과 같은 개선을 제시했습니다.[2][3]
- QA 기반 메트릭의 모든 구성요소를 분석하여 최적화
- 특히 질문 생성(QG)과 답변 가능성 분류를 핵심 개선 요소로 식별
- SummaC 벤치마크에서 기존 QA 기반 메트릭 대비 **14% 평균 개선**
- NLI 기반 메트릭(SummaC)과 결합하여 **보완적 신호 활용**으로 추가 성능 향상[3][2]

**MQAG (2023)**: 다중 선택 질문 답변과 생성 방식[4][5]
- 기존 QAGS의 개방형 QA 방식에서 다중 선택형으로 확장
- 네 개의 요약 평가 데이터셋에서 평가: QAG-CNNDM/XSum, XSum-Hallucination, Podcast[5][4]

**FIZZ (2024)**: 세밀한 원자 사실(Atomic Facts) 분해 기반[6][7]
- 원자 사실 분해(fine-grained atomic facts decomposition)를 기반으로 한 새로운 방식
- 적응적 세분화 확장(adaptive granularity expansion)을 통한 원본 문서와의 정렬
- 기존 시스템을 **크게 능가**하는 성능으로 더 높은 해석가능성 제공[7][6]

**ACUEval (2024)**: 원자 콘텐츠 단위(Atomic Content Units) 검증[8]
- 요약을 원자 콘텐츠 단위로 분해하고 각각 원본에 대해 검증
- 세 개 요약 평가 벤치마크에서 **3% 균형 정확도 개선** (다음 최고 메트릭 대비)
- LLM 생성 요약에 대한 편향 감소[8]

#### 4.2 평가 메트릭 연구의 확장

**관련 모달리티로의 확장:**
- **Q² (2021)**: QAGS 프레임워크를 지식 기반 대화에 적용하여 factual consistency 평가[9]
- 기존 토큰 매칭에서 **자연언어 추론(NLI)** 기반 답변 비교로 개선[9]

**종합 벤치마킹:**
- **GO FIGURE (2020)**: 10개 factuality 메트릭에 대한 메타 평가 제시[10]
- QA 메트릭이 일반적으로 표준 메트릭을 능가하지만, 질문 생성 방식에 따라 성능이 **크게 달라짐**을 입증[10]

- **TRUE (2021)**: QAGS의 상관관계 평가 신뢰성에 대한 재검토[11]
- 다양한 evaluation 설정에서 더 공정한 비교 제시[11]

#### 4.3 Hallucination 및 Factuality 개선 연구

**LLM 기반 평가의 새로운 방향:**
- **TrueTeacher**: 합성 NLI 레이블을 통한 대규모 데이터셋(1.4M) 생성으로 NLI 모델 개선
- SummaC와 비교하여 **50배 작은 T5-11B 모델**이 50배 큰 FLAN-PaLM 540B을 능가[12]

- **SelfCheckGPT 활용**: 샘플링 기반 방식으로 LLM의 hallucination 감지[12]

**도메인 적응 및 일반화:**
- **AdaptEval (2023)**: 과학, 의료, 정부 도메인에서 LLM의 도메인 적응 능력 평가[13]
- 작은 7B 파라미터 모델이 두 개 in-context examples만으로도 더 큰 모델과 비슷한 성능 달성 가능[13]

- **DACP (2025)**: 도메인 적응 연속 사전 학습으로 특정 도메인 성능 개선[14]

#### 4.4 최신 연구 동향 (2024-2025)

**원자적 사실 기반 평가의 확산:**
- FIZZ, ACUEval 등 **세밀한 세분화 기반 평가**가 새로운 표준으로 부상
- 단순 F1 점수 기반 매칭에서 **의미적 동등성 검증**으로 진화[8]

**LLM 기반 평가의 부상:**
- Fine-tuning된 T5, LLaMA, GPT 모델을 활용한 평가로 **더 강력한 semantic understanding** 제공
- 그러나 LLM 생성 요약에 대한 **편향성 문제** 지속[8]

**자동화된 스스로 검증:**
- **Self-Memory Alignment (2025)**: 모델이 자체 생성 응답으로부터 학습하는 선호 최적화로 factuality 개선[15]
- 다양한 벤치마크에서 **일관된 향상** 달성[15]

#### 4.5 향후 연구 시 고려할 점

**1. 모델 자체의 개선**
- 현재 추출식 QA 기반 접근의 한계 극복을 위해 **추상식 QA 도입** 검토
- 답변 의역(paraphrase) 감지를 위한 **고급 유사도 메트릭** (NLI, semantic similarity) 활용

**2. 도메인 일반화**
- 특정 도메인(의료, 법률 등) 데이터에서 in-context learning 또는 domain-adaptive continual pretraining으로 **도메인 특화 평가 메트릭 개발**
- 다국어 지원 필요성 증가

**3. LLM 시대의 재평가**
- GPT-4, Claude 등 대형 모델 기반 요약의 특성 파악
- **LLM 편향성 완화** 및 신뢰도 있는 평가 프레임워크 구축

**4. 속도-정확도 트레이드오프**
- 20개 질문으로 대부분의 성능 획득 가능하나, **더 효율적인 질문 선택 전략** 개발
- 경량 모델을 활용한 **저비용 평가 메트릭** 가능성 탐구

**5. 멀티태스크 평가 메트릭**
- Fluency, relevance와 함께 **통합 평가 프레임워크** 개발
- 현재 QAGS는 factual consistency만 측정하는 한계[1]

**6. 합성 데이터 활용**
- TrueTeacher 사례처럼 **LLM 기반 합성 라벨 생성**으로 평가 메트릭 개선 데이터 확대
- 이를 통한 더 강건한 메트릭 개발[12]

---

### 결론

QAGS는 기존 n-gram 기반 평가 지표의 근본적인 문제를 해결하고 **시맨틱 기반의 참조 없는 평가 메트릭**이라는 새로운 패러다임을 제시했습니다. 특히 **높은 해석가능성**과 **강건성**은 이후 많은 후속 연구의 기초가 되었습니다.[6][7][2][8][1]

다만 현재 추출식 QA의 한계, 도메인 이동에 따른 성능 저하, 그리고 LLM 시대에 대한 적응이 향후 연구의 주요 과제입니다. **원자적 사실 분해**, **LLM 기반 의미 이해**, **도메인 적응 기법** 등이 결합된 차세대 평가 메트릭 개발이 앞으로의 방향이 될 것으로 예상됩니다.[7][6][8]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c463edd-2750-4384-9953-0ac3a57e41a1/2004.04228v1.pdf)
[2](https://aclanthology.org/2022.naacl-main.187/)
[3](https://arxiv.org/abs/2112.08542)
[4](https://arxiv.org/abs/2301.12307)
[5](https://aclanthology.org/2023.ijcnlp-main.4.pdf)
[6](https://arxiv.org/abs/2404.11184)
[7](https://aclanthology.org/2024.emnlp-main.3/)
[8](https://aclanthology.org/2024.findings-acl.597/)
[9](https://arxiv.org/abs/2104.08202)
[10](https://aclanthology.org/2021.findings-acl.42.pdf)
[11](https://openreview.net/pdf?id=H5lBXmMBZq)
[12](https://eugeneyan.com/writing/abstractive/)
[13](https://arxiv.org/html/2407.11591v2)
[14](https://aclanthology.org/2025.newsum-main.7.pdf)
[15](https://arxiv.org/pdf/2502.19127.pdf)
[16](https://www.aclweb.org/anthology/2020.acl-main.450.pdf)
[17](https://arxiv.org/pdf/2112.08542.pdf)
[18](https://arxiv.org/pdf/2210.02804.pdf)
[19](https://arxiv.org/pdf/2305.08281.pdf)
[20](https://arxiv.org/pdf/2210.17378.pdf)
[21](https://aclanthology.org/anthology-files/anthology-files/pdf/acl/2020.acl-main.450.pdf)
[22](https://arxiv.org/abs/2309.12546)
[23](https://aclanthology.org/2020.acl-main.450/)
[24](https://yoonschallenge.tistory.com/1016)
[25](https://arxiv.org/abs/2004.04228)
[26](https://publikationen.bibliothek.kit.edu/1000172042/153257524)
[27](https://discuss.pytorch.kr/t/2025-01-06-01-12-ml-top-ml-papers-of-the-week/5835)
[28](https://aclanthology.org/2023.findings-emnlp.525.pdf)
[29](https://arxiv.org/pdf/2311.08401.pdf)
[30](https://www.aclweb.org/anthology/2020.acl-main.499.pdf)
[31](http://arxiv.org/pdf/2410.04002.pdf)
[32](https://arxiv.org/html/2504.19856v3)
[33](https://openreview.net/pdf/327c54dc2d5a30e51a3f70d622fd2e6ef66d8f5d.pdf)
[34](https://www.scitepress.org/Papers/2025/130945/130945.pdf)
[35](https://github.com/salesforce/QAFactEval)
