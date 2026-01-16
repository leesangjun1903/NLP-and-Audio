# Dense Passage Retrieval for Open-Domain Question Answering

# 1. 핵심 주장과 주요 기여 (요약)

이 논문의 핵심 주장은 다음 한 줄로 정리된다.

> **“추가 pretraining 없이, 질문–패시지 쌍만으로 학습한 단순 듀얼 인코더 기반 dense retriever(DPR)만으로도 BM25를 크게 능가할 수 있고, 이 향상이 곧바로 open‑domain QA 전체 성능 향상으로 이어진다.”** [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)

주요 기여는 네 가지이다.

1. **Dense Passage Retriever(DPR) 제안**  
   질문 인코더와 패시지 인코더 두 개의 BERT를 사용하는 듀얼 인코더 구조와, 내적 기반 similarity, FAISS 인덱싱을 결합한 **완전 dense retrieval 파이프라인**을 제안한다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

2. **아주 단순한 학습 목표로 BM25를 크게 상회**  
   inverse cloze task(ICT) 같은 추가 pretraining 없이, 질문–패시지 쌍만으로 **소프트맥스 대조 학습 + in‑batch negatives + BM25 hard negative 1개**를 사용하여, BM25 대비 top‑20 기준 **9–19%p** 절대 향상을 달성한다. [arxiv](https://arxiv.org/abs/2004.04906)

3. **Retrieval 개선이 QA 성능 향상으로 직결됨을 실증**  
   DPR + BERT reader 파이프라인이 ORQA·REALM 등 복잡한 joint training 시스템보다 Natural Questions, TriviaQA 등에서 더 높은 EM을 달성함을 보여, “retriever를 잘 만드는 것만으로 QA 전체 성능을 크게 올릴 수 있다”는 점을 실증한다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)

4. **샘플 효율성과 cross‑dataset 일반화 가능성 제시**  
   약 1k 수준의 질문–패시지 쌍만으로도 BM25를 능가하고, 한 데이터셋(NQ)에서 학습한 DPR이 WebQuestions·TREC 등 다른 QA 데이터셋에서도 BM25를 크게 상회함을 보여, **dense retriever의 일반화 잠재력**을 부각한다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

***

# 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

## 2.1 문제 정의: Open-domain QA에서의 효율적 패시지 검색

입력:

- 대규모 문서 코퍼스  

$$
  C = \{p_1, p_2, \dots, p_M\}
  $$

- 질문 $q$

목표:

- retriever $R$는 질문–코퍼스 쌍에서 상위 $k$개의 후보 패시지 집합 $C_F$를 반환

$$
  R : (q, C) \mapsto C_F,\quad C_F \subset C,\quad |C_F| = k \ll |C|
  $$

- 성능 지표(top‑ $k$ accuracy): $C_F$ 안에 **정답 스팬을 포함하는 패시지가 존재하는 질문 비율** [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)

문제점:

- 전통적인 BM25/TF‑IDF는 **lexical match**에는 강하지만,  
  예: “bad guy” ↔ “villain” 같은 의미적 변이를 잘 잡지 못해, 정답 패시지를 놓치는 경우가 많다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)
- retriever가 정답 문맥을 못 가져오면 reader는 아무리 강력해도 회복 불가능 → **retriever가 end‑to‑end QA 성능 상한을 결정**한다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

이 논문의 질문:

> **질문–패시지 쌍만으로도, dense retriever를 BM25보다 명확히 더 좋게 학습할 수 있는가?** [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

***

## 2.2 제안 방법: Dense Passage Retrieval (DPR)

### 2.2.1 듀얼 인코더 구조

DPR은 질문과 패시지를 각기 독립적인 BERT BASE 인코더로 임베딩하는 **듀얼 인코더** 구조를 사용한다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)

- 질문 인코더

$$
  E_Q : q \mapsto \mathbf{v}_q \in \mathbb{R}^d
  $$

- 패시지 인코더

$$
  E_P : p \mapsto \mathbf{v}_p \in \mathbb{R}^d
  $$

- 차원 $d = 768$ (BERT [CLS] 벡터)

질문–패시지 유사도는 **내적(dot product)**로 정의한다.

$$
\mathrm{sim}(q, p)
= E_Q(q)^\top E_P(p)
$$

내적을 사용하면 다음 장점이 있다.

- 모든 패시지 임베딩 $E_P(p_i)$를 **오프라인으로 미리 계산**해 dense 인덱스로 저장 가능.
- 질의 시점에는 질문 임베딩 $E_Q(q)$만 계산한 뒤, **Maximum Inner Product Search(MIPS)**로 상위 $k$개 패시지를 매우 빠르게 검색 가능. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

### 2.2.2 인덱싱 및 검색

- 코퍼스:  
  2018‑12‑20 영어 위키피디아 덤프를 문서 단위로 전처리 후, **100 단어 길이의 non‑overlapping 패시지 블록**으로 분할.  
  최종 패시지 수:

$$
  M \approx 2.1 \times 10^7
  $$
  
  각 패시지는 제목 + [SEP] + 본문 100단어로 구성. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)

- 인덱스:  
  FAISS HNSW 기반 인덱스에 모든 $\mathbf{v}_{p_i} = E_P(p_i)$를 저장.

- 질의 시:
  1. 질문 임베딩 계산:

$$
     \mathbf{v}_q = E_Q(q)
     $$
  
  2. $\mathbf{v}_q$ 와 내적이 가장 큰 상위 $k$개의 패시지 인덱스 반환:

```math
\mathrm{*}\>{TopK}_p \left( E_Q(q)^\top E_P(p) \right)
```

실험적으로, CPU HNSW 인덱스로 **초당 약 995개의 질문에 대해 top‑100 검색**이 가능하며, BM25/Lucene은 같은 환경에서 **초당 약 23.7개**이다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

***

## 2.3 학습 목표와 수식

### 2.3.1 기본 대조 학습 손실 (Softmax NLL)

학습 데이터셋은 다음과 같이 구성된다.

```math
\mathcal{D} 
= \left\{
\bigl\langle q_i, p_i^+, p_{i,1}^-, \dots, p_{i,n}^- \bigr\rangle
\right\}_{i=1}^m
```

- $q_i$: 질문
- $p_i^+$: 정답 스팬을 포함한 양성 패시지
- $p_{i,j}^-$: 정답을 포함하지 않는 음성 패시지들

손실 함수는 “양성 패시지의 log‑likelihood를 최대화하는 소프트맥스 기반 대조 손실”이다.

```math
L\bigl(q_i, p_i^+, p_{i,1}^-, \dots, p_{i,n}^-\bigr)
= 
- \log
\frac{
  \exp\bigl(\mathrm{sim}(q_i, p_i^+)\bigr)
}{
  \exp\bigl(\mathrm{sim}(q_i, p_i^+)\bigr)
  +
  \sum_{j=1}^n
  \exp\bigl(\mathrm{sim}(q_i, p_{i,j}^-)\bigr)
}
```

이는 metric learning 관점에서, “양성 쌍의 내적 점수를 음성 쌍보다 높게 만들도록 임베딩 공간을 학습”하는 형태이다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)

### 2.3.2 In-batch negatives

Dense retrieval에서 **좋은 음성 샘플링**이 중요하며, DPR은 **in‑batch negatives**를 핵심으로 사용한다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)

배치 크기 $B$인 mini‑batch를 생각하자.

- 질문 임베딩 행렬:

$$
  Q \in \mathbb{R}^{B \times d}
  $$

- 패시지 임베딩 행렬:

$$
  P \in \mathbb{R}^{B \times d}
  $$

유사도 행렬:

$$
S = Q P^\top \in \mathbb{R}^{B \times B}
$$

에서 $S_{ij}$는 질문 $q_i$와 패시지 $p_j$의 similarity이다.

이때:

- $i = j$인 경우: $p_i$는 $q_i$의 **양성**
- $i \neq j$인 경우: $p_j$는 $q_i$의 **음성**

질문 $q_i$에 대한 손실은

$$
L_i
= - \log
\frac{
  \exp\bigl(S_{ii}\bigr)
}{
  \sum_{j=1}^B \exp\bigl(S_{ij}\bigr)
}
$$

배치 전체 손실:

```math
L_{\text{batch}}
=
\frac{1}{B} \sum_{i=1}^B L_i
```

장점:

- 배치 내 다른 질문들의 양성 패시지들이 **자동으로 음성 역할**을 수행.
- 배치 크기 $B$에서 $B^2$개의 질문–패시지 쌍에 대해 contrastive 학습이 일어나, **학습 신호가 매우 풍부**해진다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)

### 2.3.3 Hard negative (BM25 negative) 추가

DPR은 in‑batch negatives에 더해, 각 질문당 하나의 **BM25 기반 hard negative**를 추가한다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)

- BM25로 상위 랭크되지만 정답 스팬을 포함하지 않는 패시지 $p^{\text{BM25}}_i$를 샘플링.
- 이 패시지를 해당 질문뿐 아니라 **배치 내 모든 질문의 음성**으로 사용.

실험 결과:

- “in‑batch gold negatives + BM25 hard negative 1개” 구성이 Natural Questions dev에서 top‑5/20/100 측면에서 가장 좋은 결과를 보였다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)
- hard negative 2개를 넣는 것은 1개 대비 추가 이득이 거의 없어, **복잡도 대비 효율이 떨어진다.**

***

## 2.4 Reader (QA 모듈) 구조

retriever 성능이 QA 성능으로 어떻게 이어지는지 보기 위해, 저자들은 BERT 기반 reader를 사용한다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

각 패시지 $p_i$에 대해 BERT 출력:

$$
P_i \in \mathbb{R}^{L \times h}
$$

- $L$: 최대 토큰 길이
- $h$: hidden 차원

### 2.4.1 시작/끝 위치 확률

패시지 $i$에서 토큰 $s$가 정답 시작 위치일 확률:

```math
P_{\text{start}, i}(s)
=
\mathrm{softmax}\bigl( P_i w_{\text{start}} \bigr)_s
```

정답 끝 위치 $t$에 대한 확률:

```math
P_{\text{end}, i}(t)
=
\mathrm{softmax}\bigl( P_i w_{\text{end}} \bigr)_t
```

여기서

- $w_{\text{start}}, w_{\text{end}} \in \mathbb{R}^h$는 학습 가능한 파라미터 벡터이다.

### 2.4.2 패시지 선택 확률

각 패시지 $p_i$의 [CLS] 표현을 $P_i^{[\text{CLS}]} \in \mathbb{R}^h$라 하면,

```math
\hat{P}
=
\left[
P_1^{[\text{CLS}]},
P_2^{[\text{CLS}]},
\dots,
P_k^{[\text{CLS}]}
\right]
\in
\mathbb{R}^{h \times k}
```

패시지 선택 확률은

```math
P_{\text{selected}}(i)
=
\mathrm{softmax}\bigl(
\hat{P}^\top w_{\text{selected}}
\bigr)_i
```

$$w_{\text{selected}} \in \mathbb{R}^h$$

### 2.4.3 최종 답 선택

패시지 $i$ 내에서 스팬 $(s, t)$의 점수:

```math
\text{SpanScore}_i(s, t)
=
P_{\text{start}, i}(s)
\cdot
P_{\text{end}, i}(t)
```

패시지 선택 점수:

```math
\text{PassageScore}(i)
=
P_{\text{selected}}(i)
```

실제 구현에서는 이 둘을 조합해 (예: 곱이나 로그합) 전체 스코어가 가장 큰 $(i, s, t)$를 최종 스팬으로 선택한다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

***

## 2.5 성능 향상

### 2.5.1 Passage retrieval 성능

테스트 기준, 대표적인 결과는 다음과 같다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

| Dataset | Retriever     | Top‑20 (%) | Top‑100 (%) |
|--------|---------------|-----------:|------------:|
| NQ     | BM25          | 59.1       | 73.7        |
| NQ     | DPR (Single)  | 78.4       | 85.4        |
| NQ     | DPR (Multi)   | 79.4       | 86.0        |
| Trivia | BM25          | 66.9       | 76.7        |
| Trivia | DPR (Single)  | 79.4       | 85.0        |
| Trivia | DPR (Multi)   | 78.8       | 84.7        |

관찰:

- **BM25 대비 절대 9–19%p** 수준의 top‑20 향상이 일관되게 나타난다. [arxiv](https://arxiv.org/abs/2004.04906)
- SQuAD에서는 lexical overlap이 매우 커 BM25가 상대적으로 유리하며, DPR 이득이 작거나 역전되는 경우도 있어, 데이터 편향 이슈가 지적된다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)
- BM25 + DPR hybrid는 일부 데이터셋에서 추가 이득을 주지만, dense‑only DPR만으로도 **BM25를 명확히 상회**한다는 점이 중요하다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)

### 2.5.2 End‑to‑end QA 성능

Reader까지 포함한 전체 QA EM 결과: [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

| Dataset | 시스템                 | EM (%) |
|--------|------------------------|-------:|
| NQ     | BM25 + Reader         | 32.6   |
| NQ     | ORQA                  | 33.3   |
| NQ     | DPR (Single)          | 41.5   |
| NQ     | DPR (Multi)           | 41.5   |
| Trivia | BM25 + Reader         | 52.4   |
| Trivia | DPR (Single)          | 56.8   |

핵심:

- **Retriever 성능 향상 → QA EM 향상이 직접적으로 연결**된다.  
  NQ에서 BM25+Reader(32.6) → DPR+Reader(41.5)로 약 9pt 상승. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)
- ORQA, REALM 등 복잡한 joint training retriever+reader 시스템보다 DPR pipeline이 여러 벤치마크에서 더 높은 성능을 보인다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

### 2.5.3 샘플 효율성과 cross‑dataset 일반화

- Natural Questions dev 기준, **1,000개 질문–패시지 쌍만으로도 BM25를 상회**하고, 10k, 20k, 40k, 59k로 늘릴수록 top‑k accuracy가 꾸준히 증가한다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)
- NQ로만 학습한 DPR을 WebQuestions/TREC에 그대로 적용했을 때:
  - BM25 top‑20: 55.0 / 70.9
  - NQ‑trained DPR top‑20: 69.9 / 86.3
  - dataset‑specific DPR top‑20: 75.0 / 89.1  
  → BM25 대비 큰 이득, same‑dataset fine‑tuning 대비 3–5pt 정도 손실. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)

이는 **dense retriever의 샘플 효율성과 in‑domain/general QA 간 generalization 가능성**을 강하게 시사한다.

***

## 2.6 한계와 비판적 시각

1. **Supervised QA label 의존성**  
   질문–패시지(또는 질문–정답) 쌍이 필요해, 레이블이 적은 도메인에서는 적용이 어렵다. 이후 ART, Contriever 등 label‑free dense retriever 연구가 등장한 배경이다. [direct.mit](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00564/116466/Questions-Are-All-You-Need-to-Train-a-Dense)

2. **도메인 이동에 대한 취약성 (후속 연구 기준)**  
   BEIR 벤치마크는 다양한 도메인에서 zero‑shot 평가를 수행한 결과, DPR류 dense retriever가 **BM25보다 전반적으로 낮은 zero‑shot 성능**을 보인다고 보고한다. [rueckle](https://rueckle.net/publication/BEIR/)
   → DPR는 **동일/유사 QA 도메인 내에서는 잘 generalize**하지만, **도메인 차이가 큰 환경에서는 일반화가 제한적**이다.

3. **인덱스 구축 비용**  
   2,100만 패시지에 대한 dense embedding 계산에 8GPU 기준 약 8.8시간, FAISS 인덱스 구축에 약 8.5시간이 필요해, Lucene inverted index(약 30분)에 비해 오프라인 비용이 크다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

4. **BM25 baseline 과소평가 이슈**  
   독립 재현 연구에서 원 논문의 BM25 성능이 실제보다 낮게 보고되었음을 지적하며, BM25 및 hybrid(BM25+Dense)의 성능이 DPR에 더 근접함을 보였다. [semanticscholar](https://www.semanticscholar.org/paper/89a19523b0cfb587d272b9ceb950c7bc4e8e221e)
   → dense‑only가 항상 최선이라고 보기는 어렵고, **hybrid 구성이 보다 robust**할 수 있다.

5. **희귀 구문·고유명사 처리 한계**  
   의미 기반 매칭(동의어, 패러프레이즈)에는 우수하지만, 매우 희귀한 고유명사 phrase에서는 여전히 BM25가 유리한 사례들이 qualitative 분석에서 보고된다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)

***

# 3. 모델의 “일반화 성능 향상 가능성”에 대한 논의

DPR 논문 자체와 이후 연구를 종합하면, 일반화 관점에서 다음과 같은 평가가 가능하다.

## 3.1 DPR가 보여준 긍정적 시그널

1. **소량의 supervised 예제로도 강력한 retriever 학습 가능**  
   1k–10k 수준의 질문–패시지 쌍만으로 BM25를 넘어서며, 수만 건이면 SOTA 수준에 도달한다는 것은 **dense retriever의 샘플 효율성**을 보여준다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)

2. **QA 데이터셋 간 transfer가 실질적으로 동작**  
   NQ로 학습한 DPR이 WebQuestions/TREC에서 BM25보다 훨씬 높은 top‑20 accuracy를 보이고, dataset‑specific DPR과의 격차도 3–5pt에 불과하다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)
   → **“질문/패시지 분포가 어느 정도 유사한 QA 태스크들 사이에서는 DPR 스타일 supervised dense retriever가 잘 generalize된다”**는 근거.

3. **Multi‑dataset 학습의 효과**  
   여러 QA 데이터셋(NQ, TriviaQA, WQ, TREC)을 합쳐 학습한 multi‑dataset DPR은 특히 소규모 데이터셋(예: TREC)에서 큰 성능 향상을 보인다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)
   → multi‑task·multi‑domain 학습이 dense retriever의 일반화를 향상시킬 수 있음을 시사.

## 3.2 후속 연구가 드러낸 일반화 한계

그러나 2020년 이후 연구(특히 BEIR)는 DPR류 모델의 일반화 한계를 명확히 보여준다.

1. **Heterogeneous domain zero‑shot에서의 취약성**  
   BEIR는 18개 데이터셋·9개 태스크에서 zero‑shot retrieval을 평가한 결과, **BM25가 여전히 매우 강력한 baseline**이며, dense retriever는 효율성은 좋지만 zero‑shot 일반화는 BM25 및 ColBERT류보다 자주 열세라고 보고한다. [datasets-benchmarks-proceedings.neurips](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf)

2. **도메인 적응 필요성**  
   DoDress 및 후속 domain adaptation 연구는 target 도메인에서 BM25·cross‑encoder 기반 pseudo label로 dense retriever를 self‑supervised 적응시키면, zero‑shot 대비 **5–10pt 수준의 nDCG 향상**이 가능함을 보인다. [aclanthology](https://aclanthology.org/2024.lrec-main.467.pdf)
   → DPR‑style supervised 모델만으로는 **진정한 도메인‑불변 generalization**에 한계가 있으며, pseudo‑labeling·self‑training이 필수임을 시사.

정리하면,

- DPR는 **“dense retrieval이 BM25를 대체 가능하고, 적절한 설계만으로 상당한 정도의 일반화가 가능하다”**는 것을 처음 대규모로 보여준 논문이다.
- 동시에, 이후 연구들은 **“그러나 범용 zero‑shot 일반화를 위해서는 추가적인 pretraining, domain adaptation, hybrid 구조 등이 필요하다”**는 점을 명확히 했다.

***

# 4. 2020년 이후 관련 최신 연구 비교 분석 및 향후 연구에의 영향

DPR 이후 dense retrieval·RAG 연구는 대략 다음 방향으로 확장되었다.

- (1) **학습·negative 샘플링 전략 고도화** (ANCE 등)
- (2) **dense 표현 pretraining 개선** (SimLM, ART, Query‑as‑context 등)
- (3) **도메인 일반화/적응** (BEIR, DoDress, domain adaptation)
- (4) **RAG·FiD·KG‑FiD 등 LLM/seq2seq와의 결합**
- (5) **DPR 자체 변형/개선** (multi‑positive, ensemble, control token 등)

## 4.1 학습 전략: ANCE와 Multi‑task DR

### ANCE (Approximate Nearest Neighbor Negative Contrastive Learning) [openreview](https://openreview.net/pdf?id=zeFrfgyZln)

- DPR의 in‑batch negatives는 “local negatives”에 국한되어 gradient 신호가 점점 약해질 수 있다는 분석을 제시.
- ANN 인덱스를 유지하면서 **현재 retriever가 가장 헷갈리는 전역 hard negative**를 선택하는 ANCE를 제안.
- 웹 검색 및 QA에서, ANCE는 DPR‑style 모델 대비 더 높은 MRR/NDCG를 달성하고, BM25 + cross‑encoder cascade에 가까운 성능을 훨씬 빠르게 제공. [arxiv](https://arxiv.org/pdf/2007.00808.pdf)

→ DPR 대비, **hard negative 선택을 전역으로 확장하여 일반화와 수렴 속도를 동시에 개선**하는 방향.

### Multi‑Task Dense Retrieval via Model Uncertainty Fusion [aclanthology](https://aclanthology.org/2021.findings-emnlp.26)

- 여러 QA 데이터셋에 대해 각각 DPR을 학습한 뒤, test 시 질문별 불확실도 기반으로 **전문가 모델들을 가중합**하는 방식.
- Karpukhin et al.의 multi‑dataset DPR보다 더 강한 cross‑dataset 성능을 달성. [aclanthology](https://aclanthology.org/2021.findings-emnlp.26)

→ DPR가 던진 “multi‑dataset 학습” 아이디어를 보다 정교하게 구현한 예로, **일반화 향상과 catastrophic forgetting 완화를 동시에 노린다.**

## 4.2 표현·사전학습: SimLM, ART, Query-as-context

### SimLM: Pre-training with Representation Bottleneck for DR [arxiv](https://arxiv.org/abs/2207.02578)

- dense retrieval을 위한 **표현 병목(rep. bottleneck) pretraining**을 제안, BERT 대신 retrieval 목적에 특화된 사전학습 모델을 사용.
- MS MARCO 등에서 DPR/ColBERTv2보다 우수한 성능을 보고, **label이 적은 환경에서도 높은 성능**을 보인다. [arxiv](https://arxiv.org/abs/2207.02578)

### ART: Questions Are All You Need to Train a Dense Passage Retriever [direct.mit](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00564/116466/Questions-Are-All-You-Need-to-Train-a-Dense)

- supervised QA label 없이, 코퍼스와 질문만으로 dense retriever를 학습하는 **ART**를 제안.
- “코퍼스에서 질문을 생성 → 질문으로 다시 코퍼스를 검색 → autoencoding” 구조로 pretraining을 수행.
- 다양한 QA 벤치마크에서 label‑free pretraining만으로 DPR 수준 또는 그 이상의 성능을 달성. [direct.mit](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00564/116466/Questions-Are-All-You-Need-to-Train-a-Dense)

### Query-as-context Pre-training [arxiv](https://arxiv.org/pdf/2212.09598.pdf)

- 기존 context‑based pretraining의 noise 문제를 지적하고, 문장을 **질문‑역할, 문맥‑역할**로 재구성해, 질문 유사 문장 ↔ 답변 패시지 관계를 학습하는 pretraining을 제안.
- DPR 초기화 대비 여러 retrieval 벤치마크에서 성능 향상을 보고. [arxiv](https://arxiv.org/pdf/2212.09598.pdf)

→ 이들 연구는 모두 DPR의 “단순 supervised fine‑tuning”을 넘어, **pretraining 단계에서부터 dense retrieval 목적을 반영해 일반화를 강화**하려는 흐름이다.

## 4.3 도메인 일반화·적응: BEIR, DoDress, Domain Adaptation

### BEIR 벤치마크 [arxiv](https://arxiv.org/abs/2104.08663)

- 18개 데이터셋, 9개 태스크에서 zero‑shot retrieval을 평가.
- 주요 결과:
  - **BM25는 여전히 매우 강력한 zero‑shot baseline**.
  - dense retriever(DPR, ANCE 등)는 효율성은 우수하지만, zero‑shot 성능은 BM25 및 ColBERT류보다 자주 열세.
  - hybrid sparse+dense, late‑interaction(ColBERT류)이 zero‑shot에서 가장 robust한 편. [arxiv](https://arxiv.org/abs/2104.08663v1)

→ DPR류 supervised dense retriever의 일반화 한계를 수량화하고, **hybrid 및 domain adaptation 연구의 필요성**을 명확히 한다.

### DoDress 및 진보된 domain adaptation [arxiv](https://arxiv.org/pdf/2212.06552.pdf)

- DoDress: target 도메인에서 BM25·cross‑encoder를 이용해 pseudo relevance label을 만든 뒤, 이를 이용해 dense retriever를 self‑supervised로 적응.
- 2024년 domain adaptation DR 연구는 pseudo label + query generation(GPL류) 결합으로, 다양한 BEIR 서브태스크에서 **5–10pt 수준의 nDCG@10 향상**을 보고. [aclanthology](https://aclanthology.org/2024.lrec-main.467.pdf)

→ DPR가 보여준 in‑domain generalization 위에, **pseudo‑label 기반 domain adaptation**으로 zero‑shot generalization 갭을 줄이는 방향이 확립되었다.

## 4.4 RAG·FiD·KG-FiD: DPR와 생성 모델 결합

### RAG (Retrieval-Augmented Generation) [arxiv](https://arxiv.org/abs/2005.11401)

- DPR‑style dense retriever + BART generator를 joint training해, open‑domain QA, fact verification, dialog 등 지식 집약적 태스크에서 SOTA 달성.
- RAG‑Sequence/RAG‑Token 두 변형을 통해, 문장 전체 또는 토큰별로 서로 다른 패시지를 conditioning. [proceedings.neurips](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)
- DPR는 이 구조에서 **비파라메트릭 메모리 인덱스** 역할을 수행. [semanticscholar](https://www.semanticscholar.org/paper/659bf9ce7175e1ec266ff54359e2bd76e0b7ff31)

### FiD & KG‑FiD [arxiv](https://arxiv.org/pdf/2007.01282.pdf)

- FiD(Fusion‑in‑Decoder): DPR 등 retriever가 가져온 여러 패시지를 T5 encoder로 각각 인코딩하고, decoder에서 concat하여 cross‑attention하는 구조로, ODQA에서 강력한 reader로 자리잡음. [arxiv](https://arxiv.org/pdf/2209.14290.pdf)
- KG‑FiD: DPR + FiD 위에 knowledge graph 정보를 추가해, context selection을 개선하고 효율을 높인다. [aclanthology](https://aclanthology.org/2022.acl-long.340.pdf)
- FiDO, FastFiD, token pruning 등은 FiD 구조 내에서 encoder/decoder FLOPs 분배를 조정해 **수배 이상의 추론 효율 향상**을 달성한다. [arxiv](https://arxiv.org/pdf/2212.08153.pdf)

→ DPR는 이들 시스템에서 **retriever 표준**으로 사용되며, **RAG 전체의 factuality·robustness·efficiency**를 좌우하는 중요한 구성요소가 된다.

## 4.5 DPR 자체 개선: Multi-positive, Ensemble, Control Tokens 등

- **Multiple Positive DPR**: 질문당 여러 positive 패시지를 사용해 BCE 기반 multi‑positive loss를 도입, top‑20/100 retrieval accuracy를 추가 향상. [arxiv](https://www.arxiv.org/pdf/2508.09534.pdf)
- **Confidence-Calibrated Ensemble DPR**: granularity가 다른 여러 dense retriever를 ensemble하고 confidence calibration을 적용, NQ/SQuAD 등에서 SOTA 달성. [arxiv](https://arxiv.org/abs/2306.15917)
- **Control Token with DPR**: 특정 도메인/소스 제어를 위한 control token을 도입해, Top‑1/Top‑20 accuracy를 각각 13%/4%p 개선. [arxiv](https://arxiv.org/abs/2405.13008)
- **Aggretriever**: BERT [CLS] 하나에 전체 텍스트를 압축하는 구조의 한계를 지적하며, 다양한 aggregation 전략으로 DPR 성능을 상향. [arxiv](https://arxiv.org/pdf/2208.00511.pdf)

→ DPR의 구조적 철학(듀얼 인코더 + dot product + in‑batch negatives)을 유지한 채, **loss, representation, ensemble, 제어 가능성 측면에서 점진적으로 일반화를 개선**하는 방향이다.

***

# 5. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

## 5.1 DPR가 남긴 구조적 영향

1. **Dense retriever를 “기본 옵션”으로 만든 논문**  
   DPR 이후, open‑domain QA·RAG·지식 집약적 NLP에서 **BM25 단독**이 아닌  
   - DPR/ANCE/SimLM류 dense retriever  
   - 혹은 BM25 + Dense hybrid  
   가 사실상의 기본 베이스라인이 되었다. [arxiv](https://arxiv.org/html/2509.10697v1)

2. **듀얼 인코더 + in‑batch negatives + dot‑product loss 설계의 표준화**  
   이후 dense retriever(ANCE, SimLM, ART 등)는 거의 모두 DPR과 유사한 구조를 채택하며, 차이는 주로 **pretraining·negative mining·domain adaptation**에 있다. [openreview](https://openreview.net/pdf?id=zeFrfgyZln)

3. **Retriever–Reader 모듈 분리 설계의 정당화**  
   ORQA/REALM의 joint training과 달리, DPR는 retriever와 reader를 분리 학습해도 (혹은 그 편이) 더 나은 결과를 낼 수 있음을 보인다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)
   → 오늘날 RAG 시스템에서 retriever와 LLM을 모듈식으로 교체/튜닝하는 설계의 이론적 근거가 된다. [arxiv](https://arxiv.org/abs/2312.10997)

4. **Dense retrieval 일반화 연구의 기준점**  
   BEIR, DoDress, SimLM, ART, domain adaptation 연구 등은 거의 모두 DPR을 출발점·비교 대상으로 사용한다. [rueckle](https://rueckle.net/publication/BEIR/)

## 5.2 향후 연구 시 고려할 핵심 포인트 (특히 일반화 관점)

연구를 설계할 때, DPR 및 이후 연구가 주는 교훈은 다음과 같다.

1. **In-domain vs Out-of-domain 일반화를 명확히 구분할 것**  
   - DPR는 동일/유사 QA 도메인 내 transfer에는 강하다. [scottyih](https://scottyih.org/files/emnlp2020-dpr.pdf)
   - 그러나 BEIR류 heterogeneous 환경에서는 dense가 BM25보다 자주 약하다. [datasets-benchmarks-proceedings.neurips](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf)
   → 반드시 **in‑domain + out‑of‑domain**을 함께 평가하고, zero‑shot generalization을 따로 분석해야 한다.

2. **Hard negatives/negative sampling 설계를 신중히 할 것**  
   - DPR의 in‑batch + BM25 hard negatives는 단순하지만 매우 강력하다. [aclanthology](https://aclanthology.org/2020.emnlp-main.550.pdf)
   - ANCE는 ANN 기반 전역 hard negatives로 이를 한 단계 확장한다. [arxiv](https://arxiv.org/pdf/2007.00808.pdf)
   → 실제 연구에서는 in‑batch, BM25/ColBERT hard, ANN hard negatives를 조합해, **test 시 분포와 최대한 가까운 negative 분포**를 설계해야 한다.

3. **Pretraining 관점에서 dense 표현 학습을 재설계할 것**  
   - DPR 수준의 supervised fine‑tuning만으로도 강력하지만, SimLM, ART, query‑as‑context 등은 pretraining 단계에서부터 dense retrieval 목적을 반영해,  
     **label이 적거나 없는 도메인에서도 뛰어난 일반화**를 달성했다. [arxiv](https://arxiv.org/abs/2207.02578)
   → 앞으로는 “language model pretraining + retrieval pretraining”의 결합이 dense retriever 일반화의 핵심 축이 될 것이다.

4. **Domain adaptation과 pseudo‑labeling 전략을 적극 활용할 것**  
   - DoDress 및 2024년 domain adaptation 연구는 BM25/cross‑encoder를 teacher로 사용한 pseudo‑label self‑training이 dense retriever의 domain shift를 크게 줄일 수 있음을 보여준다. [arxiv](https://arxiv.org/pdf/2212.06552.pdf)
   → 실제 도메인(기업 내 문서, 특수 분야 등)에서는 **pseudo‑labeling 기반 self‑supervised adaptation 파이프라인** 설계가 필수적이다.

5. **Sparse–Dense Hybrid 및 Late‑interaction 구조를 함께 고려할 것**  
   - DPR는 dense‑only지만, BEIR 결과는 hybrid 및 ColBERT류가 zero‑shot에서 가장 robust한 경우가 많음을 보여준다. [semanticscholar](https://www.semanticscholar.org/paper/BEIR:-A-Heterogenous-Benchmark-for-Zero-shot-of-Thakur-Reimers/807600ef43073cd9c59d4208ee710e90cf14efa8)
   - DPR 재현 연구도 BM25 baseline이 생각보다 강하다는 점을 지적한다. [arxiv](https://arxiv.org/pdf/2104.05740.pdf)
   → dense‑only vs sparse‑only vs hybrid vs late‑interaction을 **효율성·효과성·일반화** 측면에서 동시에 비교·최적화할 필요가 있다.

6. **RAG/LLM과의 공동 최적화 관점에서 retriever를 설계할 것**  
   - RAG, FiD, KG‑FiD, FiDO 등은 DPR류 retriever 위에 LLM/seq2seq를 얹어, **factuality·robustness·efficiency**를 종합적으로 개선한다. [arxiv](https://arxiv.org/abs/2005.11401)
   - 최근 survey들은 retrieval‑augmented generation 전체를 대상으로 한 **end‑to‑end generalization 평가 프레임워크**를 제안한다. [arxiv](https://arxiv.org/pdf/2509.10697.pdf)

연구자로서 앞으로는,

- DPR가 제시한 **샘플 효율적인 dense retriever 설계**를 계승하면서,
- BEIR 및 domain adaptation 연구가 드러낸 **domain shift 취약성**을 극복하기 위해,
  - 강력한 **retrieval‑aware pretraining**
  - pseudo‑label 기반 **self‑supervised domain adaptation**
  - **sparse‑dense hybrid 및 late‑interaction** 구조
  - LLM 기반 RAG에서의 **joint/alternating training**

을 통합적으로 설계하는 것이 “모델의 일반화 성능 향상”이라는 관점에서 가장 중요한 연구 방향이 될 것이다.

<span style="display:none">[^2_1][^2_2][^2_3][^2_4][^2_5][^2_6][^2_7][^2_8][^2_9][^2_10][^2_11][^2_12]</span>

[^2_1]: https://aclanthology.org/2020.emnlp-main.550.pdf

[^2_2]: https://scottyih.org/files/emnlp2020-dpr.pdf

[^2_3]: https://arxiv.org/abs/2004.04906

[^2_4]: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00564/116466/Questions-Are-All-You-Need-to-Train-a-Dense

[^2_5]: https://rueckle.net/publication/BEIR/

[^2_6]: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf

[^2_7]: https://arxiv.org/abs/2104.08663

[^2_8]: https://arxiv.org/abs/2104.08663v1

[^2_9]: https://www.semanticscholar.org/paper/89a19523b0cfb587d272b9ceb950c7bc4e8e221e

[^2_10]: https://arxiv.org/pdf/2104.05740.pdf

[^2_11]: https://www.semanticscholar.org/paper/BEIR:-A-Heterogenous-Benchmark-for-Zero-shot-of-Thakur-Reimers/807600ef43073cd9c59d4208ee710e90cf14efa8

[^2_12]: https://aclanthology.org/2024.lrec-main.467.pdf

[^2_13]: https://arxiv.org/pdf/2212.06552.pdf

[^2_14]: https://openreview.net/pdf?id=zeFrfgyZln

[^2_15]: https://arxiv.org/pdf/2007.00808.pdf

[^2_16]: https://arxiv.org/abs/2007.00808

[^2_17]: https://openreview.net/forum?id=zeFrfgyZln

[^2_18]: https://aclanthology.org/2021.findings-emnlp.26

[^2_19]: https://arxiv.org/abs/2207.02578

[^2_20]: https://arxiv.org/pdf/2212.09598.pdf

[^2_21]: https://openreview.net/pdf?id=wCu6T5xFjeJ

[^2_22]: https://arxiv.org/abs/2005.11401

[^2_23]: https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf

[^2_24]: https://arxiv.org/abs/2312.10997

[^2_25]: https://www.semanticscholar.org/paper/659bf9ce7175e1ec266ff54359e2bd76e0b7ff31

[^2_26]: https://arxiv.org/pdf/2007.01282.pdf

[^2_27]: https://aclanthology.org/2022.acl-long.340.pdf

[^2_28]: https://arxiv.org/pdf/2209.14290.pdf

[^2_29]: https://arxiv.org/pdf/2212.08153.pdf

[^2_30]: https://www.semanticscholar.org/paper/FiDO:-Fusion-in-Decoder-optimized-for-stronger-and-Jong-Zemlyanskiy/a7ca1bce0af7fe4703f5c3296db2dcc8dc112f20

[^2_31]: https://arxiv.org/html/2403.14197v1

[^2_32]: https://www.arxiv.org/pdf/2508.09534.pdf

[^2_33]: https://arxiv.org/abs/2306.15917

[^2_34]: https://arxiv.org/html/2306.15917

[^2_35]: https://arxiv.org/abs/2405.13008

[^2_36]: https://arxiv.org/pdf/2208.00511.pdf

[^2_37]: https://arxiv.org/html/2509.10697v1

[^2_38]: https://arxiv.org/pdf/2509.10697.pdf

[^2_39]: https://www.aclweb.org/anthology/2020.emnlp-main.550

[^2_40]: https://arxiv.org/html/2506.00054v1

[^2_41]: https://arxiv.org/html/2507.18910v1

[^2_42]: https://arxiv.org/pdf/2410.12837.pdf

[^2_43]: 2004.04906v3.pdf




