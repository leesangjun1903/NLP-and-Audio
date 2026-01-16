
# REALM: Retrieval-Augmented Language Model Pre-Training 

## 1. 핵심 주장과 주요 기여 (Executive Summary)

REALM은 2020년 Google Research에서 발표한 획기적인 논문으로, 언어 모델 사전 학습(pre-training)에 신경망 기반 지식 검색 기능을 통합하는 새로운 패러다임을 제시했습니다. 기존 BERT, RoBERTa, T5 같은 언어 모델들이 지식을 모델 파라미터에 암묵적으로 저장하는 방식에서 벗어나, 명시적인 텍스트 코퍼스(Wikipedia)로부터 관련 문서를 동적으로 검색하여 활용합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

**핵심 주장**: 더 큰 네트워크를 만드는 대신, 학습된 검색기와 언어 모델을 결합하면 더 모듈화되고, 해석 가능하며, 확장성 있는 지식 통합이 가능하다는 것입니다. 이 접근 방식은 특히 Open-domain Question Answering(Open-QA) 같은 지식 집약적 작업에서 ORQA 대비 4-16% 절대 정확도 향상을 달성했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

***

## 2. 해결하고자 하는 문제 (Problem Definition)

### 2.1 기존 접근 방식의 한계

기존의 암묵적 지식 저장 방식은 세 가지 근본적인 문제를 야기합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

**1) 무한한 모델 확장의 필요성**: 더 많은 세상의 지식을 담으려면 네트워크를 계속 키워야 합니다. 예를 들어, 영국의 통화가 "파운드"라는 사실을 BERT가 예측하도록 학습되었지만, 수백만 개의 유사한 팩트를 저장하려면 파라미터가 기하급수적으로 증가해야 합니다.

**2) 해석 불가능성 (Interpretability)**: 어떤 지식이 어디에 저장되었는지 추적할 수 없습니다. 잘못된 예측이 나면 모델의 어떤 부분이 책임인지 파악하기 어렵습니다.

**3) 새로운 지식 적응의 어려움**: 세상이 변해도 모델의 파라미터는 고정되어 있으므로, 새로운 정보(예: 새로운 수도, 최신 뉴스)를 반영하려면 전체 모델을 재학습해야 합니다.

### 2.2 Open-QA의 특수성

Open-QA는 다른 작업보다 훨씬 더 넓은 지식 범위를 요구합니다. 전통적인 Reading Comprehension(SQuAD) 과제에서는 모델이 주어진 단일 문서를 이해하면 되지만, Open-QA에서는 위키피디아의 모든 문서에서 답변이 나올 수 있으므로 잠재적으로 수백만 개 문서의 지식이 필요합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

***

## 3. 제안하는 방법 및 모델 구조

### 3.1 Retrieve-then-Predict 생성 과정의 수학적 형식화

REALM의 핵심은 다음과 같은 주변화(marginalization) 기반의 생성 과정입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

$$p(y|x) = \sum_{z \in Z} p(y|z,x) p(z|x) \quad (1)$$

여기서:
- $x$ : 입력 (사전학습 시: 마스킹된 문장, 미세조정 시: 질문)
- $y$ : 출력 (사전학습 시: 원래 토큰, 미세조정 시: 답변)
- $z$ : 검색된 문서
- $Z$: 전체 지식 코퍼스 (Wikipedia)
- $p(z|x)$: 검색 확률 (Query가 주어졌을 때 문서 z의 관련성)
- $p(y|z,x)$: 조건부 확률 (쿼리와 검색된 문서가 주어졌을 때 답변)

이 공식은 두 가지 학습 가능한 컴포넌트를 명시적으로 분리합니다:

**1) 검색 단계** $p(z|x)$
**2) 예측 단계** $p(y|z,x)$

### 3.2 모델 아키텍처: 두 가지 핵심 컴포넌트

#### 3.2.1 Knowledge Retriever (신경망 기반 검색기)

검색 확률 $p(z|x)$는 다음과 같은 밀집 내적 모델(dense inner product model)로 구현됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

$$p(z|x) = \frac{\exp f(x,z)}{\sum_{z' \in Z} \exp f(x,z')}$$

여기서 관련성 점수는:

$$f(x,z) = \text{Embed}_{\text{input}}(x)^T \text{Embed}_{\text{doc}}(z)$$

**구현 세부사항**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)
- $\text{Embed}\_{\text{input}}(x) = W_{\text{input}} \cdot \text{BERT}_{\text{CLS}}(\text{joinBERT}(x))$
- $\text{Embed}\_{\text{doc}}(z) = W_{\text{doc}} \cdot \text{BERT}\_{\text{CLS}}(\text{joinBERT}(z_{\text{title}}, z_{\text{body}}))$
- BERT-style Transformer를 사용하여 입력과 문서를 인코딩
- 문서 제목($z_{\text{title}}$)과 본문($z_{\text{body}}$)을 결합하여 문서 표현 생성
- 선형 투사 행렬 $W$로 차원 감소

#### 3.2.2 Knowledge-Augmented Encoder (지식 강화 인코더)

조건부 확률 $p(y|z,x)$는 Transformer 기반 인코더로 구현되며, 사전학습과 미세조정 단계에서 다르게 작동합니다.

**사전학습 (MLM - Masked Language Modeling)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

$$p(y|z,x) = \prod_{j \in \text{mask}} p(y_j|z,x)$$

$$p(y_j|z,x) = \frac{\exp(w_j^T \text{BERT}_{\text{MASK}_j}(\text{joinBERT}(x,z_{\text{body}})))}{\sum_{y'} \exp(w_{y'}^T \text{BERT}_{\text{MASK}_j}(\text{joinBERT}(x,z_{\text{body}})))}$$

여기서:
- $j$는 마스킹된 토큰의 인덱스
- $\text{BERT}_{\text{MASK}_j}$는 j번째 마스킹 위치의 Transformer 출력 벡터
- $w_j$는 학습된 단어 임베딩

**미세조정 (Open-QA)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

$$p(y|z,x) = \sum_{s \in S(z,y)} \frac{\exp(\text{MLP}([h_{\text{START}_s}; h_{\text{END}_s}]))}{\sum_{s' \in S(z)} \exp(\text{MLP}([h_{\text{START}_{s'}}; h_{\text{END}_{s'}}]))}$$

여기서:
- $S(z,y)$는 문서 $z$에서 답변 $y$와 매치되는 모든 스팬의 집합
- $h_{\text{START}\_s}$와 $h_{\text{END}_s}$는 스팬의 시작과 끝 토큰 표현

***

## 4. 훈련 절차 및 계산 최적화

### 4.1 로그 우도 최대화

양쪽 단계 모두 다음 목표를 최대화하여 훈련됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

$$\mathcal{L} = \log p(y|x) = \log \sum_{z \in Z} p(y|z,x) p(z|x)$$

검색기와 인코더 파라미터에 대한 기울기는:

$$\nabla_{\theta} \log p(y|x) = \sum_{z} p(z|y,x) \nabla_{\theta} f(x,z)$$

$$\nabla_{\phi} \log p(y|x) = \sum_{z} p(z|x) \nabla_{\phi} \log p(y|z,x)$$

### 4.2 MIPS (Maximum Inner Product Search)를 통한 스케일링

핵심 과제는 수백만 개 문서에 대해 marginal likelihood를 계산하는 것입니다. REALM은 다음과 같이 해결합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

**근사 전략**: 상위 k개 문서만 고려
$$p(y|x) \approx \sum_{z \in \text{Top-k}(Z, x)} p(y|z,x) p(z|x)$$

**MIPS 활용**: 내적 최대값 검색 알고리즘으로 선형 시간에 Top-k 문서 찾기

**비동기 갱신**: 검색 인덱스가 "낡지만(stale)" 효율성을 위해 허용됨 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

```
시간 t: MIPS 인덱스 갱신 (새 파라미터로 모든 문서 재인코딩)
시간 t ~ t+500: 훈련 진행 (낡은 인덱스 사용)
시간 t+500: 다시 갱신
```

이 절충(trade-off)은 실증적으로 충분한 빈도(예: 500 스텝마다)로 갱신하면 안정적인 최적화를 보장함을 보여줍니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

### 4.3 검색기가 학습하는 방식

식 (1)의 gradient를 분석하면, 각 문서 $z$는 다음과 같은 업데이트를 받습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

$$\nabla_{\theta} f(x,z) \propto \mathbb{1}[p(y|z,x) > \mathbb{E}_{z' \sim p(z|x)}[p(y|z',x)]]$$

**해석**: 
- 문서 $z$가 예상보다 나은 예측을 제공하면 ($p(y|z,x)$가 높으면) → 양의 기울기 → 검색 점수 증가
- 문서 $z$가 예상보다 나쁘면 → 음의 기울기 → 검색 점수 감소

이는 **무지도(unsupervised)** 학습 신호만 사용하면서도 의미 있는 검색을 학습합니다.

***

## 5. 귀납적 편향 (Inductive Biases) 주입

검색기가 초기에 무작위 문서를 검색하면 인코더가 검색을 무시하도록 학습되고, 이로 인해 검색기가 유용한 기울기를 받지 못하는 "악순환(vicious cycle)"이 발생할 수 있습니다. REALM은 이를 해결하기 위해 여러 전략을 도입합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

### 5.1 Salient Span Masking

모든 마스킹된 토큰이 외부 지식을 필요로 하지는 않습니다. 예를 들어, "[CLS] The cat **[MASK]** on the mat [SEP]"에서 [MASK]는 문맥만으로 "sat"임을 알 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

**해결책**: Named Entity Recognition (NER)과 정규식으로 주요 개체와 날짜만 마스킹합니다.

- "United Kingdom" (명명된 개체) ✓ 검색 필요
- "July 1969" (날짜) ✓ 검색 필요
- 임의의 토큰 ✗ 검색 불필요

실험에서 이는 무작위 토큰 마스킹 대비 현저히 더 나은 성능을 보입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

### 5.2 Null Document (공 문서)

모든 마스킹된 스팬이 검색을 요구하지 않습니다. 따라서 검색이 도움이 되지 않는 경우를 처리하기 위해 "공 문서"를 Top-k에 추가합니다. 이는 모델이 적절히 검색을 "사용하지 않을" 수 있게 해줍니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

### 5.3 Trivial Retrieval 방지

$X$(사전학습 코퍼스)와 $Z$(지식 코퍼스)가 같으면 (예: 모두 Wikipedia), 마스킹된 문장 $x$가 원래 문서 $z$에서 나온 경우, 검색된 문서에 $x$의 마스킹되지 않은 부분이 포함되어 있습니다. 이는 "치팅(cheating)"을 가능하게 하므로, 사전학습 중에는 원본 문서를 제외합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

### 5.4 ICT (Inverse Cloze Task) 초기화

무작위 초기화 상태에서는 검색기가 의미 있는 기울기를 받지 못하는 콜드 스타트 문제가 발생합니다. 이를 해결하기 위해: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

1. **ICT로 검색기 사전학습**: 문장이 주어졌을 때 그 문장이 나온 원본 문서를 검색하도록 학습
2. **BERT로 인코더 사전학습**: 표준 BERT 사전학습 사용
3. **REALM 공동 학습**: 사전학습된 두 컴포넌트로부터 시작

***

## 6. 성능 향상 및 실증적 결과

### 6.1 Open-QA 벤치마크 성능

| 데이터셋 | REALM (Wikipedia) | REALM (CC-News) | ORQA | T5-11B | 개선율 |
|----------|-------------------|-----------------|------|--------|--------|
| **NaturalQuestions-Open** | 40.4% | 39.2% | 33.3% | 34.5% | +6-7pp |
| **WebQuestions** | 40.7% | 40.2% | 36.4% | 37.4% | +3-4pp |
| **CuratedTrec** | 42.9% | 46.8% | 30.1% | - | +12-17pp |

주목할 점: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)
- **효율성**: REALM은 330M 파라미터로 T5-11B (11.3B 파라미터)를 초월 → 약 **34배 작은 모델**
- **corpus 선택의 영향**: Wikipedia와 CC-News 코퍼스 선택에 따른 차이 존재
- **ORQA 대비 우위**: 동일한 하이퍼파라미터와 미세조정 설정에서 순수하게 pre-training 개선으로만 달성

### 6.2 Ablation Study (NaturalQuestions-Open 개발셋)

| 구성 | 정확도 | Top-5 Recall |
|------|--------|--------------|
| **REALM (전체)** | 38.2% | 38.5% |
| REALM retriever + Baseline encoder | 37.4% | 38.5% |
| Baseline retriever + REALM encoder | 35.3% | 13.9% |
| Baseline (ORQA) | 31.3% | 13.9% |
| Random uniform masks | 32.3% | 24.2% |
| Random span masks | 35.3% | 26.1% |
| 30 steps stale MIPS | 28.7% | 15.1% |

**해석**:
- 검색기와 인코더 모두 중요함 (각각 단독으로는 제한적)
- Salient span masking의 중요성: 무작위 마스킹 대비 +2.9pp
- MIPS 갱신 빈도: 30 스텝마다는 너무 성글음 (28.7%)

### 6.3 Retrieval Utility 메트릭

검색의 유용성을 측정하기 위해 다음을 정의합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

$$\text{RU}(z,x) = \log p(y|z,x) - \log p(y|\varnothing,x)$$

여기서 $\varnothing$는 공 문서입니다.

실험 결과, RU는 사전학습 과정에서 꾸준히 증가하며, 전체 로그-우도보다도 다운스트림 Open-QA 성능의 더 좋은 예측지표입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

***

## 7. 일반화 성능과 한계

### 7.1 강점

**1) 모듈성 (Modularity)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)
- 새로운 지식을 추가하려면 문서를 코퍼스에 추가하면 됨
- 모델 재훈련 불필요

**2) 해석 가능성 (Interpretability)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)
- 어느 문서에서 답변이 나왔는지 추적 가능
- 오류 분석 용이

**3) 매개변수 효율성** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)
- 큰 모델보다 작은 모델로 우수한 성능 달성
- 계산 효율성 개선

**4) 지식 업데이트** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)
- 표 4의 예: 2018년 Wikipedia 코퍼스에서 불가능한 답변("Lawrence")이 2020년 업데이트된 코퍼스에서 가능해짐

### 7.2 한계 및 문제점

**1) Multi-hop 추론의 한계** [aclanthology](https://aclanthology.org/2023.findings-emnlp.1036.pdf)
- REALM은 단일 문서에서 답변을 추출하는 데 최적화
- 복잡한 다단계 추론이 필요한 질문에서 성능 저하
- 2023 연구: retrieve-and-read 모델의 multi-hop 추론 능력이 제한적임을 입증

**2) 불완전한 검색에 대한 취약성** [arxiv](http://arxiv.org/pdf/2408.04414.pdf)
- 2024년 연구들이 지적: 검색된 문서가 답변을 포함하지 않으면 생각할 수 없음
- 모순되는 정보에 대한 처리 미흡
- 답변 불가능한 질문(unanswerable questions)에서 환각(hallucination) 발생

**3) Knowledge Corpus 의존성**
- 검색기의 성능이 코퍼스 커버리지에 의존
- Out-of-domain 질문에서 성능 저하

**4) Cold-start 문제** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)
- 검색기가 좋지 않으면 초기 학습이 어려움
- ICT 사전학습이 필수 (별도 데이터 필요)

**5) 계산 비용** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)
- 수백만 개 문서 고려 필요
- 검색 인덱스 유지 및 갱신 오버헤드
- 추론 시 검색 지연

***

## 8. 후속 연구 비교 분석 (2020-2026)

### 8.1 동시대 모델 (2020)

#### **RAG (Retrieval-Augmented Generation, Lewis et al., 2020)** [arxiv](https://arxiv.org/abs/2005.11401)

REALM과 동일 시기에 발표되었으나 다른 접근 방식을 취함:

| 특성 | REALM | RAG |
|------|-------|-----|
| 기본 구조 | BERT-style 양방향 | BART seq2seq |
| 적용 시점 | Pre-training | Fine-tuning |
| 검색 | 학습 가능한 dense retriever | Dense Passage Retriever (DPR) |
| Fusion 방식 | Early fusion (문서 prepend) | Late fusion (marginalization) |
| NQ 성능 | 40.4% | 44.5% [aclanthology](https://aclanthology.org/2023.findings-acl.732.pdf) |

**차이점**: RAG는 생성형(generative)이므로 임의의 문자열 답변 생성 가능하나, REALM은 추출형(extractive)이므로 기존 스팬만 답변.

#### **DPR (Dense Passage Retrieval, Karpukhin et al., 2020)** [pub.towardsai](https://pub.towardsai.net/dense-passage-retrieval-2020-and-contriever-2021-the-models-that-paved-the-way-for-future-8ec140398ead)

더 단순한 검색기 모델:
- Dual-encoder 구조 (query와 document 독립 인코딩)
- REALM의 joint 인코딩보다 계산 효율적
- 검색만 학습하고, 읽기 모듈은 분리된 기존 QA 시스템 사용
- 후속 모델들의 기초 검색기로 광범위하게 채택

***

### 8.2 고급 아키텍처 (2021-2022)

#### **FiD (Fusion-in-Decoder, Izacard & Grave, 2020-2021)** [arxiv](http://arxiv.org/pdf/2205.09226.pdf)

REALM과 완전히 다른 파이프라인:

**구조**:
```
query + retrieved passages
    ↓
[각 passage를 독립적으로 인코딩]
    ↓
[디코더에서 모든 인코딩을 결합하며 토큰 생성]
```

**성능 개선**:
- NQ: 44.5% (REALM 40.4% 대비 +4.1pp)
- TriviaQA: 56.8%
- 생성형이므로 더 유연한 답변 가능

**후속 개선**:
- **FiDO (Fusion-in-Decoder Optimized, 2023)**: 메모리 대역폭 최적화로 7배 빠른 추론 [aclanthology](https://aclanthology.org/2023.findings-acl.732.pdf)
  - 레이어별 스파스 교차-어텐션 도입
  - 멀티-쿼리 어텐션 사용
  - 비대칭 디코더 설계
  - NQ에서 REALM 40.4% → FiDO 대규모 모델로 월등히 우수

#### **RETRO (Retrieval-Enhanced Transformer, Borgeaud et al., 2021)** [proceedings.mlr](https://proceedings.mlr.press/v162/borgeaud22a/borgeaud22a.pdf)

대규모 검색 데이터베이스 활용의 새로운 벤치마크:

**혁신**:
- 2조(2T) 토큰 데이터베이스 사용 (REALM의 Wikipedia는 ~수십 억 토큰)
- 청크 단위 반복 검색 (sequence 생성 중)
- 자동회귀 언어 모델링에 적용

**성능**:
- Pile 벤치마크에서 GPT-3/Jurassic-1과 동등 성능
- **하지만 25배 더 작은 모델** (7B vs 175B)

**차이점**:
- REALM: 단일 query에 대해 한 번만 검색
- RETRO: 각 토큰 생성마다 검색 (더 빈번한 검색)
- RETRO는 이전 청크 기반 "로컬" 검색, REALM은 입력 전체 기반

**한계** (2023 분석): [aclanthology](https://aclanthology.org/2023.findings-eacl.109.pdf)
- 검색 이득이 실제로는 훈련 데이터와의 overlapping tokens에 국한될 가능성
- 진정한 일반화(true generalization)가 아닐 수 있음

#### **ColBERT (Contextualized Late Interaction, Khattab & Zaharia, 2020)** [aclanthology](https://aclanthology.org/2022.naacl-main.272.pdf)

다른 검색 패러다임:

**Late Interaction 개념**:
```
Query: [token1, token2, token3] → [vec1, vec2, vec3]
Doc:   [token1, token2, ...]   → [vec1', vec2', ...]

MaxSim 점수 = Σ_i max_j (vec_i · vec_j')
```

- 단일 벡터 대신 다중 벡터 표현 유지
- 세밀한 상호작용(fine-grained interaction) 가능

**성능**:
- MS MARCO에서 기존 single-vector retrievers (DPR) 능가
- 계산 비용: DPR보다 높지만 cross-encoder보다 낮음

**진화**:
- **ColBERTv2 (2022)**: 잔여 압축(residual compression)으로 6-10배 공간 절감, 정확도는 유지/향상 [aclanthology](https://aclanthology.org/2022.naacl-main.272.pdf)

***

### 8.3 현대적 발전 (2023-2024)

#### **문제 포인트: 검색 견고성 (Robustness)**

**2024년 연구들의 지적**: [arxiv](https://arxiv.org/html/2410.15107v1)

RAG 시스템의 약점:
- "unanswerable queries": 검색된 문맥이 답변을 포함하지 않는 경우
- "adversarial inputs": 의도적으로 잘못된 정보가 있는 경우
- "conflicting documents": 모순되는 정보 다중 문서

**해결책**:
- Self-CRAG (Corrective Retrieval-Augmented Generation): 검색 품질 자체 판단 후 재검색
- RALM 강건화 기법: 검색 신뢰도 추정 개선

#### **하이브리드 검색**

**KG-FiD (Knowledge Graph-FiD, 2022)**: [arxiv](https://arxiv.org/pdf/2110.04330.pdf)
- 텍스트 검색 + 지식 그래프 검색 결합
- 노이즈 감소로 ODQA 성능 향상

**Graph RAG (2025)**: [arxiv](https://arxiv.org/html/2506.00054v1)
- 검색된 passage로부터 엔티티 기반 그래프 구성
- 커뮤니티 요약으로 규모 확장
- Multi-hop QA recall +6.4pp

#### **LLM 기반 검색기 (2024-2025)**

새로운 트렌드: 대형 언어 모델을 검색 모델로 직접 사용

**Debater (2025)**: [arxiv](https://arxiv.org/html/2502.12974v1)
- Chain-of-Deliberation으로 LLM 기반 dense retriever 개선
- 작은 LLM으로 큰 모델과 비교 가능한 성능 달성

**MA-RAG (Multi-Agent RAG, 2025)**: [arxiv](https://arxiv.org/pdf/2505.20096.pdf)
- 여러 전문 검색 에이전트 결합
- 8B 모델로 REPLUG 65B, GPT-4o-mini 능가

***

## 9. 모델의 일반화 성능 향상 가능성

### 9.1 일반화 성능의 정의

일반화 성능은 두 가지 차원에서 평가됩니다:

**1) 개별 데이터셋 간 전이 (Transfer)**
- NaturalQuestions → WebQuestions
- 같은 domain (Open-QA)이지만 다른 질문 분포

**2) 도메인 간 전이 (Domain Transfer)**
- Wikipedia 기반 pre-training → 다른 분야 (의학, 법률, 금융)
- 근본적으로 다른 corpus에 적응

### 9.2 REALM의 일반화 강점

**1) Pre-training 기반 설계** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)
- 비지도 신호만 사용하여 사전학습
- 특정 downstream task에 의존하지 않음
- 다양한 작업으로 미세조정 가능성

**2) 모듈화 구조** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)
- 검색 corpus 교체 가능
- 새로운 domain corpus로 간단히 적응
- 인코더와 검색기 독립적 개선 가능

**3) 실증적 증거**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)
- Wikipedia와 CC-News corpus 모두에서 학습 가능 (Table 1)
- 서로 다른 corpus 선택의 trade-off 존재:
  - Wikipedia: 기존 지식에 강함 (40.4%)
  - CC-News: 최신 정보에 강함 (일부 벤치마크에서 우수)

### 9.3 일반화 성능의 한계

**1) Retrieval Oracle Problem** [arxiv](http://arxiv.org/pdf/2408.04414.pdf)
- 검색 품질이 최고 성능의 상한선(ceiling)을 결정
- Gold document가 상위 5개에 없으면 불가능한 작업
- NaturalQuestions에서:
  - 상위 100: ~91% recall (좋음)
  - 상위 5: ~39% recall (금 문서가 있을 확률 절반 이하)

**2) Domain Mismatch** [aclanthology](https://aclanthology.org/2023.findings-eacl.109.pdf)
- Pre-training corpus와 test corpus의 불일치
- RETRO 분석: 검색 이득의 상당 부분이 overlapping tokens에서만 (진정한 일반화 제한)

**3) Multi-hop Reasoning 부재** [aclanthology](https://aclanthology.org/2023.findings-emnlp.1036.pdf)
- 단일 문서 답변 추출 최적화
- "Who won the Oscar for Best Director in the movie starring [actor]?" 같은 복잡한 질문에서 약함

### 9.4 개선 방향

**1) Adaptive Retrieval**
- 언제 검색할지 동적 결정
- 필요 없으면 검색 스킵 (SELF-RAG 스타일)
- 계산 효율성과 성능 trade-off 개선

**2) Multi-hop Evidence Chain**
- 여러 문서를 순차적으로 검색
- 중간 결과를 기반으로 다음 검색 쿼리 생성
- Graph RAG 같은 구조화된 접근

**3) Cross-lingual Transfer**
- 다국어 corpus로 pre-training
- 저자원 언어로도 Open-QA 가능

**4) Domain-specific Fine-tuning**
- 법률, 의료, 금융 corpus로 검색기 재적응
- 인코더는 공유하되 검색기만 fine-tune

***

## 10. 미래 연구 영향과 고려사항

### 10.1 REALM이 끼친 학문적 영향

**1) Retrieval-Augmented 패러다임의 정립** [research](https://research.google/blog/realm-integrating-retrieval-into-language-representation-models/)
- REALM은 explicit knowledge retrieval을 mainstream으로 만듦
- 수백 개의 follow-up 논문 생성 (3,141회 인용)
- RAG 분야의 기초 이론 제공

**2) Pre-training + Retrieval 분리의 확립**
- 이전: 단순히 retriever-reader pipeline
- 이후: "retriever도 언어 모델 pre-training의 일부"라는 개념
- 후속 모델들 (FiD, RETRO, ColBERT)의 영감

**3) 무지도 학습(Unsupervised Learning) 신호 활용**
- 마스킹된 언어 모델링만으로 검색기 학습 가능 입증
- 별도 supervised 데이터 불필요

### 10.2 현재 (2024-2026) 연구의 초점

**1) 검색 견고성(Robustness)**
- REALM 시대: 검색이 맞다고 가정
- 현재: 검색이 틀릴 수 있다 → 어떻게 처리할 것인가?
- 요구사항: 검색 신뢰도 추정, 자체 수정, 검색 거절

**2) 효율성-정확성 Trade-off**
- FiDO: 7배 빠르면서 성능 유지/향상
- 추론 지연이 중요한 실시간 시스템
- 캐싱, 조기 종료, 스파스 어텐션

**3) LLM과의 통합**
- REALM: encoder-only (BERT-style)
- 현재: decoder-only LLM (GPT-style)과 검색 결합
- RAG × LLM = 새로운 가능성

### 10.3 향후 연구 시 고려할 점

#### **설계 선택사항**

| 측면 | 선택지 | Trade-off |
|------|--------|-----------|
| **검색 구조** | Dual-encoder (DPR) vs Joint (REALM) vs Late-interaction (ColBERT) | 효율성 vs 상호작용 풍부도 |
| **Fusion 시점** | Early (REALM) vs Late (FiD) vs Mid-layer (RETRO) | 계산 위치, 메모리 요구 |
| **Corpus 규모** | Small (1B) vs Medium (100B) vs Large (2T, RETRO) | 메모리, 검색 지연 vs 커버리지 |
| **검색 빈도** | 한 번 vs 매 토큰마다 | 정확성 vs 계산 비용 |

#### **평가 지표의 확대**

REALM과 이후 작업들은 주로 정확도(Accuracy)에 초점:

**새로운 차원**:
- **Faithfulness**: 생성된 답변이 검색된 문서에 기반했는가?
- **Robustness**: 잘못된 검색에 대한 견고성
- **Efficiency**: 처리량(throughput), 지연(latency), 메모리
- **Interpretability**: 예측의 추적 가능성
- **Bias**: 검색 결과의 편향성

#### **Scalability 문제**

**REALM의 제한**:
- Wikipedia: ~13M 문서
- 실제 시스템: 웹 규모 (수십억 문서)

**해결 시도**:
- Dense retrieval with quantization (ColBERTv2: 6-10배 압축)
- Sparse + Dense hybrid (SPLADE, BM25+DPR)
- Multi-hop graph-based (Graph RAG)

### 10.4 특정 기술 영역별 미래 방향

#### **의료/과학 분야** [proceedings.mlr](http://proceedings.mlr.press/v119/guu20a/guu20a.pdf)
- 신문/논문 코퍼스로 REALM-style 모델
- 검색 근거가 중요 (의료 거짓 정보 방지)

#### **다국어 시스템**
- Cross-lingual retrieval
- Zero-shot transfer to low-resource languages
- Multilingual corpus 구성의 challenge

#### **지속적 학습(Continual Learning)**
- 새로운 정보가 계속 추가됨 (뉴스, 새 제품)
- 기존 corpus 업데이트 vs 새 corpus 추가
- REALM의 모듈화가 우월성 → 문서만 추가하면 됨

#### **제약 조건 있는 검색**
- "최근 2년 내 뉴스만"
- "특정 소스만" (신뢰할 수 있는 출처)
- "편향 없는" (다양한 관점 표현)

***

## 11. 결론

REALM은 2020년 "명시적 검색"을 언어 모델 pre-training에 처음 통합한 혁신적 논문입니다. 사전에 모든 지식을 매개변수에 저장하는 대신, 필요할 때 외부 코퍼스를 검색하는 방식은: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/01f47c46-90bf-4b3d-a080-941b4a0176bc/2002.08909v1.pdf)

1. **모듈화와 해석 가능성**을 획기적으로 개선
2. **매개변수 효율성**으로 작은 모델이 큰 모델을 초월 가능
3. **지식 업데이트 유연성**으로 새로운 정보 즉각 반영

이 접근 방식은 이후 6년간의 연구를 주도했으며, RAG, FiD, RETRO, ColBERT 등 다양한 발전을 촉발했습니다. [arxiv](https://arxiv.org/html/2507.18910v1)

**현재 (2024-2026) 연구의 초점**은 REALM이 남긴 문제들(검색 견고성, multi-hop 추론, efficiency)을 해결하는 것이며, **향후의 중요한 과제**는: [arxiv](https://arxiv.org/html/2410.15107v1)

- 실시간 환경에서의 scalable retrieval
- 검색 오류에 대한 defensive 설계
- 도메인과 언어 간 일반화
- 모듈식 아키텍처의 추가 발전

이 과정에서 REALM의 핵심 아이디어—"명시적 검색이 암묵적 저장보다 우월하다"—는 계속 타당성을 유지하고 있으며, 이것이 이 논문의 가장 깊은 기여입니다.

***

## References

<span style="display:none">[^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68]</span>

<div align="center">⁂</div>

[^1_1]: 2002.08909v1.pdf

[^1_2]: https://aclanthology.org/2023.findings-emnlp.1036.pdf

[^1_3]: http://arxiv.org/pdf/2408.04414.pdf

[^1_4]: https://arxiv.org/html/2410.15107v1

[^1_5]: https://arxiv.org/abs/2005.11401

[^1_6]: https://aclanthology.org/2023.findings-acl.732.pdf

[^1_7]: https://pub.towardsai.net/dense-passage-retrieval-2020-and-contriever-2021-the-models-that-paved-the-way-for-future-8ec140398ead

[^1_8]: http://arxiv.org/pdf/2205.09226.pdf

[^1_9]: https://proceedings.mlr.press/v162/borgeaud22a/borgeaud22a.pdf

[^1_10]: https://www.semanticscholar.org/paper/Improving-language-models-by-retrieving-from-of-Borgeaud-Mensch/002c256d30d6be4b23d365a8de8ae0e67e4c9641

[^1_11]: https://aclanthology.org/2023.findings-eacl.109.pdf

[^1_12]: https://aclanthology.org/2022.naacl-main.272.pdf

[^1_13]: https://cs.uwaterloo.ca/~jimmylin/publications/Lin_etal_2021_RepL4NLP.pdf

[^1_14]: https://arxiv.org/pdf/2110.04330.pdf

[^1_15]: https://www.microsoft.com/en-us/research/wp-content/uploads/2021/10/2110.04330.pdf

[^1_16]: https://arxiv.org/html/2506.00054v1

[^1_17]: https://arxiv.org/html/2502.12974v1

[^1_18]: https://arxiv.org/pdf/2505.20096.pdf

[^1_19]: https://research.google/blog/realm-integrating-retrieval-into-language-representation-models/

[^1_20]: https://arxiv.org/html/2507.18910v1

[^1_21]: http://proceedings.mlr.press/v119/guu20a/guu20a.pdf

[^1_22]: http://arxiv.org/pdf/2402.13492.pdf

[^1_23]: https://arxiv.org/pdf/2403.03187.pdf

[^1_24]: https://arxiv.org/pdf/2410.12837.pdf

[^1_25]: https://arxiv.org/pdf/2312.10997.pdf

[^1_26]: https://arxiv.org/pdf/2209.14290.pdf

[^1_27]: http://arxiv.org/pdf/2407.12982.pdf

[^1_28]: https://arxiv.org/html/2502.20245v1

[^1_29]: https://aclanthology.org/2022.naacl-srw.7/

[^1_30]: https://aclanthology.org/2021.findings-emnlp.26/

[^1_31]: https://arxiv.org/abs/2002.08909

[^1_32]: https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART003124697

[^1_33]: https://aclanthology.org/2025.findings-acl.1220.pdf

[^1_34]: https://arxiv.org/pdf/2002.08909.pdf

[^1_35]: https://dl.acm.org/doi/abs/10.5555/3524938.3525306

[^1_36]: https://aclanthology.org/2022.naacl-srw.7.pdf

[^1_37]: https://knowledge-nlp.github.io/kdd2023/papers/Nangi9.pdf

[^1_38]: https://arxiv.org/html/2407.17877v1

[^1_39]: https://arxiv.org/html/2509.10697v1

[^1_40]: https://arxiv.org/html/2507.03958v1

[^1_41]: https://arxiv.org/html/2509.23861v1

[^1_42]: https://arxiv.org/html/2504.14891v1

[^1_43]: https://arxiv.org/html/2504.05216v3

[^1_44]: https://arxiv.org/html/2501.16111v1

[^1_45]: https://arxiv.org/html/2502.19712v1

[^1_46]: https://aclanthology.org/2021.emnlp-main.301.pdf

[^1_47]: http://arxiv.org/pdf/2404.02581.pdf

[^1_48]: https://arxiv.org/pdf/2310.13682.pdf

[^1_49]: https://aclanthology.org/2023.emnlp-main.93.pdf

[^1_50]: https://arxiv.org/html/2403.11803v1

[^1_51]: http://arxiv.org/pdf/2305.17041.pdf

[^1_52]: https://openreview.net/pdf?id=4bCsX2K0KuR

[^1_53]: https://www.reddit.com/r/mlscaling/comments/rhcog6/retrievalenhanced_transformer_retro_improving/

[^1_54]: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00405/107205/Relevance-guided-Supervision-for-OpenQA-with

[^1_55]: https://aclanthology.org/2022.acl-long.340.pdf

[^1_56]: https://arxiv.org/abs/2112.04426

[^1_57]: https://people.eecs.berkeley.edu/~matei/papers/2020/sigir_colbert.pdf

[^1_58]: https://arxiv.org/pdf/2212.08153.pdf

[^1_59]: https://deepmind.google/blog/improving-language-models-by-retrieving-from-trillions-of-tokens/

[^1_60]: https://arxiv.org/html/2505.19274v1

[^1_61]: https://arxiv.org/pdf/2112.01488.pdf

[^1_62]: https://arxiv.org/pdf/2112.04426.pdf

[^1_63]: https://arxiv.org/html/2508.03555v1

[^1_64]: https://www.arxiv.org/pdf/2509.23861.pdf

[^1_65]: https://arxiv.org/html/2410.00004v2

[^1_66]: https://arxiv.org/html/2403.13291v1

[^1_67]: https://arxiv.org/html/2306.13421v2

[^1_68]: https://arxiv.org/html/2408.08444v1
