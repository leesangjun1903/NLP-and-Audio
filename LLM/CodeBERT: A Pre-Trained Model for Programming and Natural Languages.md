# CodeBERT: A Pre-Trained Model for Programming and Natural Languages

### 1. 핵심 주장과 주요 기여

CodeBERT는 자연어(NL)와 프로그래밍 언어(PL)를 위한 최초의 대규모 bimodal 사전학습 모델입니다. 이 논문의 핵심 기여는 다음과 같습니다.[1]

**첫째**, NL-PL 쌍의 bimodal 데이터와 코드만 있는 unimodal 데이터를 모두 활용하는 하이브리드 학습 방식을 제안했습니다. **둘째**, 자연어 코드 검색과 코드 문서 생성 작업에서 최고 성능을 달성했습니다. **셋째**, NL-PL probing을 위한 최초의 데이터셋을 구축하여 사전학습 모델이 학습한 지식을 체계적으로 평가했습니다.[1]

### 2. 문제 정의와 제안 방법

#### 해결하고자 하는 문제

기존 NLP 사전학습 모델들은 자연어만을 대상으로 학습되어 프로그래밍 언어와의 의미적 연결을 포착하지 못했습니다. 또한 multi-modal 사전학습 모델들은 bimodal 데이터만 사용하여 대규모로 존재하는 unimodal 코드 데이터를 활용하지 못했습니다.[1]

#### 모델 구조

CodeBERT는 RoBERTa-base와 동일한 multi-layer bidirectional Transformer 구조를 사용하며, 총 125M개의 파라미터를 가집니다. 입력은 `[CLS], w₁, w₂, ..., wₙ, [SEP], c₁, c₂, ..., cₘ, [EOS]` 형태로 자연어와 코드를 연결하여 구성됩니다. 출력은 각 토큰의 맥락적 벡터 표현과 분류/랭킹을 위한 `[CLS]` 토큰의 집계된 표현을 포함합니다.[1]

#### 학습 목적함수 (수식 포함)

**Objective #1: Masked Language Modeling (MLM)**

NL-PL 쌍에서 15%의 토큰을 마스킹하고 원래 토큰을 예측합니다:[1]

$$
m_i^w \sim \text{unif}\{1, |w|\} \text{ for } i = 1 \text{ to } |w|
$$

$$
m_i^c \sim \text{unif}\{1, |c|\} \text{ for } i = 1 \text{ to } |c|
$$

$$
\mathcal{L}_{\text{MLM}}(\theta) = \sum_{i \in m^w \cup m^c} -\log p_{D_1}(x_i | w_{\text{masked}}, c_{\text{masked}})
$$

여기서 $$p_{D_1}$$은 대규모 어휘집에서 토큰을 예측하는 discriminator입니다.[1]

**Objective #2: Replaced Token Detection (RTD)**

Unimodal 데이터를 활용하기 위해 ELECTRA의 RTD를 적용했습니다. NL generator $$p_{G_w}$$와 PL generator $$p_{G_c}$$가 마스킹된 위치에 대해 그럴듯한 대안 토큰을 생성합니다:[1]

$$
\hat{w}_i \sim p_{G_w}(w_i | w_{\text{masked}}) \text{ for } i \in m^w
$$

$$
\hat{c}_i \sim p_{G_c}(c_i | c_{\text{masked}}) \text{ for } i \in m^c
$$

Discriminator는 각 토큰이 원본인지 교체된 것인지를 판별합니다:[1]

$$
\mathcal{L}_{\text{RTD}}(\theta) = \sum_{i=1}^{|w|+|c|} \left[ \delta(i) \log p_{D_2}(x^{\text{corrupt}}, i) + (1 - \delta(i))(1 - \log p_{D_2}(x^{\text{corrupt}}, i)) \right]
$$

여기서 $$\delta(i) = 1$$은 $$x_i^{\text{corrupt}} = x_i$$일 때, 즉 토큰이 원본일 때입니다.[1]

**최종 손실함수:**

$$
\min_{\theta} \mathcal{L}_{\text{MLM}}(\theta) + \mathcal{L}_{\text{RTD}}(\theta)
$$

이 접근법은 generator로 bidirectional n-gram 언어 모델을 사용하며, NL generator는 bimodal 데이터의 문서에서, PL generator는 unimodal 코드에서 학습됩니다.[1]

#### 학습 데이터

Python, Java, JavaScript, PHP, Ruby, Go 등 6개 프로그래밍 언어에서 총 210만 개의 bimodal NL-PL 쌍과 640만 개의 unimodal 코드를 사용했습니다. 데이터는 GitHub 저장소에서 수집되었으며, 각 bimodal 데이터는 함수와 그에 대한 문서의 첫 단락으로 구성됩니다.[1]

### 3. 성능 향상

#### 자연어 코드 검색 (Natural Language Code Search)

CodeBERT(MLM+RTD)는 macro-average MRR 0.7603을 달성하여 RoBERTa(0.6972)를 6.31%p 초과했습니다. 특히 Python에서는 0.8685, Go에서는 0.8400의 높은 성능을 보였습니다. RoBERTa로 초기화한 경우 처음부터 학습한 것보다 성능이 크게 향상되었습니다.[1]

#### 코드 문서 생성 (Code Documentation Generation)

CodeBERT(MLM+RTD)는 전체 평균 BLEU-4 점수 17.83을 기록하여 RoBERTa(16.57)보다 1.26점 높았습니다. 특히 PHP(25.16), Python(19.06), Go(18.07)에서 우수한 성능을 보였습니다.[1]

#### NL-PL Probing (Zero-shot 평가)

파라미터를 고정한 zero-shot 설정에서 CodeBERT(MLM)은 PL probing에서 85.66%, NL probing에서 74.53%의 정확도를 달성했습니다. 이는 RoBERTa(각각 62.45%, 61.21%)를 크게 초과하는 결과입니다. 특히 Python과 Java에서 90% 이상의 정확도를 보였습니다.[1]

### 4. 일반화 성능 향상

#### Zero-shot 전이 능력

CodeBERT는 사전학습 단계에서 본 적 없는 C# 언어에 대한 코드 요약 작업에서 22.36 BLEU 점수를 달성했습니다. 이는 RoBERTa(19.81)보다 2.55점 높으며, 프로그래밍 언어 간 전이 학습 능력이 우수함을 보여줍니다.[1]

#### Multilingual 학습 효과

6개 프로그래밍 언어에 대해 명시적인 언어 마커 없이 하나의 모델로 학습하는 multilingual BERT 방식을 적용했습니다. 이를 통해 각 언어별로 일관되게 높은 성능을 달성했으며, 언어 간 공통된 프로그래밍 패러다임을 효과적으로 학습했습니다.[1]

#### Fine-tuning 초기 단계 성능

Learning curve 분석 결과, CodeBERT는 fine-tuning 초기 단계부터 RoBERTa와 코드 전용 사전학습 모델보다 높은 성능을 보였습니다. 이는 CodeBERT가 downstream 작업에 유리한 초기화를 제공함을 의미합니다.[1]

#### Bimodal 데이터의 효과

NL-PL probing 실험에서 CodeBERT는 코드만으로 사전학습된 모델보다 NL probing(74.53% vs 65.19%)과 PL probing(85.66% vs 74.11%) 모두에서 우수한 성능을 보였습니다. 이는 bimodal 데이터가 자연어와 코드의 의미적 연결을 효과적으로 학습시킴을 보여줍니다.[1]

### 5. 한계점

#### AST 정보 활용 부족

CodeBERT는 코드를 토큰 시퀀스로만 처리하며, Abstract Syntax Tree(AST)의 구조 정보를 활용하지 못합니다. C# 코드 요약 작업에서 code2seq(23.04 BLEU)보다 낮은 성능(22.36 BLEU)을 보인 주된 이유가 AST의 compositional path를 사용하지 않았기 때문입니다. AST의 순회 순서를 따라 CodeBERT를 학습했으나 생성 작업에서는 개선이 없었습니다.[1]

#### 생성 작업을 위한 사전학습 목적함수 부재

CodeBERT의 사전학습 목적함수는 주로 NL-PL 이해 작업에 초점을 맞추고 있어, 생성 기반 학습 목적함수(예: BART의 denoising 목적함수)를 포함하지 않습니다. 비록 강력한 BLEU 점수를 달성했지만, 생성 관련 학습 목적함수를 추가하면 더 개선될 가능성이 있습니다.[1]

#### Generator의 단순성

RTD 목적함수에서 사용하는 generator가 bidirectional n-gram 언어 모델로 제한되어 있습니다. Bimodal 증거를 활용하는 더 복잡한 generator나 Transformer 기반 신경망 구조를 사용하면 성능이 더 향상될 수 있습니다.[1]

#### Code Completion 성능

Preceding context만을 사용한 PL probing에서 CodeBERT의 정확도는 59.12%로, bidirectional context를 사용한 경우(85.66%)보다 크게 낮았습니다. 이는 코드 자동완성과 같은 단방향 예측 작업에서 추가 개선이 필요함을 시사합니다.[1]

#### 언어별 모델 필요성

Multilingual 모델을 fine-tuning했을 때 언어별 모델보다 성능이 낮았습니다. 이는 각 프로그래밍 언어의 특수성을 더 잘 반영하기 위해 언어별 적응 방법이 필요함을 의미합니다.[1]

### 6. 향후 연구에 미치는 영향

#### Code Intelligence 분야의 새로운 기준

CodeBERT는 코드-언어 이해를 위한 사전학습 모델의 표준을 확립했습니다. 이후 연구들은 CodeBERT를 baseline으로 사용하여 더 발전된 모델(예: CodeT5, GraphCodeBERT, CodeGen)을 개발할 수 있는 기반을 마련했습니다.[1]

#### Cross-modal 학습 패러다임 제시

NL과 PL을 서로 다른 modality로 취급하는 접근법은 vision-language 모델(ViLBERT, VideoBERT)의 성공을 코드 도메인으로 확장했습니다. 이는 프로그래밍 언어를 또 하나의 modality로 연구하는 새로운 관점을 제시했습니다.[1]

#### Probing 방법론의 선구적 제안

NL-PL probing 데이터셋은 사전학습 모델이 학습한 지식을 체계적으로 평가하는 방법론을 제시했습니다. 이는 향후 모델 해석가능성 연구에 중요한 도구가 되었습니다.[1]

### 7. 향후 연구 시 고려사항

#### AST 통합 방법 연구

AST의 구조 정보를 Transformer 기반 모델에 효과적으로 통합하는 방법을 연구해야 합니다. 단순한 순회 순서 적용이 아닌, graph neural network나 tree-based attention mechanism을 고려할 수 있습니다.[1]

#### 생성 지향 사전학습 목적함수 추가

코드 생성 작업의 성능을 더욱 향상시키기 위해 denoising autoencoder, span corruption, prefix language modeling 등의 생성 기반 목적함수를 추가 연구해야 합니다.[1]

#### 더 강력한 Generator 설계

RTD 목적함수의 효과를 극대화하기 위해 bimodal 증거를 활용하는 Transformer 기반 generator나 joint training 방식을 연구할 필요가 있습니다.[1]

#### 도메인/언어 적응 방법

더 많은 프로그래밍 언어와 도메인으로 확장하기 위해 효율적이고 강력한 적응 방법(adapter, prompt tuning, few-shot learning)을 개발해야 합니다.[1]

#### 더 다양한 downstream 작업

코드 검색과 문서 생성 외에도 버그 탐지, 취약점 분석, 코드 변환, 프로그램 수정 등 더 다양한 NL-PL 작업에 CodeBERT를 적용하고 평가해야 합니다.[1]

#### 단방향 컨텍스트 활용 개선

코드 자동완성과 같은 실용적 응용을 위해 preceding context만으로도 높은 성능을 달성할 수 있는 학습 전략을 연구해야 합니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4d8797b6-91aa-484f-9bff-d31362410530/2002.08155v4.pdf)
