
# Attention Is All You Need

## 1. 핵심 주장과 주요 기여

**Attention Is All You Need** 논문은 2017년 구글 브레인 팀이 발표한 혁신적인 논문으로, 기존의 **RNN(Recurrent Neural Networks)과 CNN(Convolutional Neural Networks)**에 기반한 시퀀스 모델 아키텍처를 완전히 새로운 패러다임으로 전환했습니다.[1]

이 논문의 가장 중요한 주장은 **"Attention(어텐션)만으로 충분하다"**는 것입니다. 즉, 기존 모델에서 필수적이었던 순환 구조나 합성곱 구조를 제거하고, 순수하게 어텐션 메커니즘만으로 구성된 **Transformer** 아키텍처를 제안했습니다. 이는 다음과 같은 획기적인 성과를 달성했습니다.[1]

- **WMT 2014 영어-독일어 번역**: BLEU 점수 28.4 기록 (기존 최고 성능 모델 대비 2.0 이상 향상)
- **WMT 2014 영어-프랑스어 번역**: BLEU 점수 41.8 달성 (기존 모델의 1/4 미만의 훈련 비용)
- **훈련 효율성**: 8개의 P100 GPU에서 3.5일 내에 완료[1]

***

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 기존 모델의 핵심 문제점

기존의 RNN 기반 시퀀스 모델들은 다음과 같은 본질적 한계를 가지고 있었습니다.[1]

1. **순차 계산의 비효율성**: RNN은 이전 시간 단계의 은닉 상태 $$h_{t-1}$$에 의존하여 현재 시간 단계 $$h_t$$를 계산해야 하므로, 완전한 병렬 처리가 불가능합니다.

2. **장거리 의존성 학습 어려움**: 시퀀스가 길어질수록 초반부 정보가 점진적으로 손실되어 장거리 의존성 학습이 어려워집니다.

3. **계산 복잡도**: 시퀀스 길이 $$n$$에 대해 $$O(n)$$의 순차 연산이 필요합니다.[1]

### 2.2 제안된 핵심 방법: Transformer 아키텍처

Transformer는 **Encoder-Decoder** 구조로 다음과 같이 구성됩니다.

#### **2.2.1 Scaled Dot-Product Attention**

Transformer의 핵심은 다음 수식으로 정의되는 Scaled Dot-Product Attention입니다:[1]

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

여기서:
- $$Q$$: Query (크기 $$d_k$$)
- $$K$$: Key (크기 $$d_k$$)
- $$V$$: Value (크기 $$d_v$$)
- $$\sqrt{d_k}$$: 스케일링 인자 (gradient vanishing 방지)

이 메커니즘은 각 쿼리가 모든 키-값 쌍과의 관계를 계산하여, 시퀀스의 모든 위치 간의 의존성을 **상수 시간(O(1) 순차 연산)**에 포착할 수 있습니다.[1]

#### **2.2.2 Multi-Head Attention**

단일 어텐션 헤드의 한계를 극복하기 위해 Multi-Head Attention을 제안합니다:[1]

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

여기서:
- $$h = 8$$: 병렬 어텐션 헤드의 개수
- $$d_k = d_v = d_{\text{model}}/h = 64$$: 각 헤드의 차원
- $$W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$$, $$W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$$, $$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$$: 학습 가능한 선형 변환
- $$W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$$: 출력 선형 변환

Multi-Head Attention은 모델이 **서로 다른 표현 부분공간에서 정보를 동시에 처리**할 수 있게 하여, 단일 헤드의 평균화 효과로 인한 정보 손실을 방지합니다.[1]

### 2.3 Transformer의 전체 구조

#### **2.3.1 Encoder Stack**

Encoder는 N=6개의 동일한 레이어로 구성되며, 각 레이어는 다음 두 개의 서브층을 포함합니다:[1]

1. **Multi-head self-attention**: 자신의 이전 레이어 출력에 대한 어텐션
2. **Position-wise Feed-Forward Network**:
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

각 서브층 주변에는 **Residual Connection**과 **Layer Normalization**이 적용됩니다:[1]

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

이는 매우 깊은 모델에서 그래디언트 소실 문제를 완화합니다.

#### **2.3.2 Decoder Stack**

Decoder도 N=6개의 동일한 레이어로 구성되며, Encoder와 달리 다음과 같은 특징을 가집니다:[1]

1. **Self-attention** (마스킹됨): 미래 위치를 보지 않도록 masking 적용
2. **Encoder-decoder attention**: Encoder 출력에 대한 어텐션
3. **Position-wise Feed-Forward Network**

자동회귀(auto-regressive) 생성을 보장하기 위해, 디코더의 self-attention에서 위치 $$i$$의 쿼리가 위치 $$i$$보다 이후의 위치에 어텐션할 수 없도록 마스킹됩니다.

### 2.4 Positional Encoding

Transformer는 순환 구조가 없으므로, 토큰의 순서 정보를 명시적으로 주입해야 합니다. 논문에서는 다음의 정현파 위치 인코딩을 사용합니다:[1]

$$PE_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

여기서 $$\text{pos}$$는 위치, $$i$$는 차원입니다. 이 함수들은 **기하급수적으로 증가하는 주기**를 가져, 모델이 **상대 위치 관계**를 쉽게 학습할 수 있도록 합니다.[1]

***

## 3. 모델 구조의 장점과 계산 복잡도 분석

### 3.1 계산 복잡도 비교[1]

| 레이어 타입 | 레이어당 복잡도 | 순차 연산 | 최대 경로 길이 |
|-----------|------------|---------|------------|
| **Self-Attention** | $$O(n^2 \cdot d)$$ | **O(1)** | **O(1)** |
| Recurrent | $$O(n \cdot d^2)$$ | O(n) | O(n) |
| Convolutional | $$O(k \cdot n \cdot d^2)$$ | O(1) | $$O(\log_k(n))$$ |

Transformer의 혁신성:
- **병렬 처리**: 순차 연산 O(1)으로 완전한 병렬 처리 가능
- **장거리 의존성**: 모든 위치 간 상수 경로 길이 (O(1))로 그래디언트 흐름 개선
- **효율성**: 시퀀스 길이 $$n$$이 표현 차원 $$d$$보다 작은 경우 (NMT에서 일반적) RNN보다 빠름[1]

### 3.2 성능 향상 요인

논문의 ablation study (Table 3)에서 각 요소의 영향을 분석했습니다:[1]

- **멀티헤드 어텐션 헤드 수**: 8개 헤드가 최적 (단일 헤드 대비 0.9 BLEU 감소)
- **Key 차원 감소**: $$d_k$$ 감소 시 성능 저하 (호환성 함수의 복잡도 필요)
- **Dropout**: Pdrop=0.1이 과적합 방지에 효과적
- **Label Smoothing**: $$\epsilon_{ls}=0.1$$ 적용 (perplexity 증가하지만 BLEU 향상)

***

## 4. 일반화 성능 향상 메커니즘

### 4.1 Transformer의 일반화 이점[1]

1. **병렬 처리로 인한 더 많은 데이터 처리 가능**
   - RNN은 순차적 구조로 인해 긴 시퀀스에서 메모리 제약
   - Transformer는 배치 크기와 시퀀스 길이를 더 크게 설정 가능
   - 더 다양한 데이터 노출 → 일반화 성능 향상

2. **Multi-Head Attention의 표현 다양성**
   - 서로 다른 어텐션 헤드가 다양한 관계 패턴을 학습
   - 단일 헤드의 평균화 효과 제거
   - 의미론적(semantic)과 구문론적(syntactic) 특징 동시 학습

3. **Residual Connection과 Layer Normalization**
   - 그래디언트 흐름 안정화
   - 훈련 중 더 나은 최적화 수렴
   - 더 깊은 모델 학습 가능

### 4.2 작업 전이 능력[1]

논문은 기계 번역뿐 아니라 **영어 구문 분석(English Constituency Parsing)** 작업에서도 Transformer의 우수한 일반화 능력을 시연했습니다.

| 파서 | 훈련 데이터 | WSJ 23 F1 |
|-----|----------|----------|
| Transformer (4 layers) | WSJ only | 91.3 |
| Transformer (4 layers) | semi-supervised | **92.7** |
| RNN Grammar (최고 성능) | generative | 93.3 |

제한된 훈련 데이터(WSJ: 40K 문장)에서도 기존 방법을 능가하는 성능을 보여, **작은 데이터셋에서의 일반화 능력**을 입증했습니다.[1]

***

## 5. 모델의 한계와 개선 방향

### 5.1 논문에서 지적된 한계

1. **Attention 복잡도**: $$O(n^2)$$ 시간과 메모리 복잡도로 매우 긴 시퀀스 처리 시 비효율
   - 논문에서는 제한된 어텐션(restricted self-attention) 등의 미래 방향 제시[1]

2. **절대 위치 정보**: Positional encoding이 절대 위치에만 의존
   - 상대 위치 관계만으로 충분한가에 대한 검증 필요

3. **해석 가능성**: 블랙박스 특성으로 의사결정 과정 이해 어려움

### 5.2 최신 연구에서의 개선 방향 (2024-2025)[2][3]

#### **5.2.1 효율성 개선**

1. **스파스 어텐션(Sparse Attention)**
   - 모든 토큰 간의 관계를 계산하지 않고, 필요한 관계만 선택적으로 계산
   - Longformer, BigBird 등이 로컬 어텐션과 글로벌 어텐션 결합[2]

2. **선형 어텐션(Linear Attention)**
   - Flash Attention 등으로 계산 복잡도를 $$O(n^2)$$에서 $$O(n)$$으로 감소[2]

3. **RWKV (Receptance Weighted Key Value)**
   - RNN의 선형 시간 복잡도와 Transformer의 병렬 처리 능력 결합
   - 훈련과 추론 효율성 모두 개선[3]

#### **5.2.2 일반화 성능 개선**

1. **길이 일반화(Length Generalization)**
   - Looped Transformers: 훈련 데이터보다 긴 시퀀스에 대한 일반화 개선[4]
   - 알고리즘 문제에서 훈련 길이를 초과하는 시퀀스 처리 가능

2. **파라미터 효율적 파인튜닝**
   - LoRA(Low-Rank Adaptation): 전체 모델 파라미터 대신 저랭크 행렬만 학습
   - 적응 비용 감소로 더 다양한 작업에 일반화 가능[2]

3. **추론 성능의 한계 분석**
   - Deep Reasoning Dataset (DeepRD): 기존 벤치마크보다 복잡한 추론 문제 제시
   - 현재 LLM의 일반화 능력이 일정 복잡도 이상에서 급격히 저하[5]

#### **5.2.3 Vision Transformer의 발전**

1. **Next-ViT**: 산업 배포 최적화로 CNN 대비 지연시간-정확도 트레이드오프 개선[6]

2. **의료 영상 분석**: Multi-modal Transformer로 다양한 의료 데이터 처리 능력 향상[7]

***

## 6. 논문의 학문적 및 산업적 영향

### 6.1 학문적 기여

"Attention Is All You Need" 논문은 **현대 AI의 기초 이론**이 되었습니다:

- **대규모 언어 모델(LLM)의 기반**: GPT-4, ChatGPT, Claude, Gemini 등 모든 최신 LLM이 Transformer 기반[2]
- **컴퓨터 비전의 변혁**: Vision Transformer (ViT)로 CNN 중심 패러다임을 어텐션 기반으로 전환[8]
- **멀티모달 학습**: DALL-E, LLaVA 등 이미지-텍스트 통합 모델의 기초[9]

### 6.2 산업적 임팩트

1. **계산 효율성 개선**: 병렬 처리로 훈련 시간 대폭 단축
2. **확장성**: 파라미터 수와 데이터셋 크기 증가에 따른 성능 향상 (scaling law)
3. **전이 학습**: 사전 학습된 대규모 모델을 다양한 작업에 적용 가능

***

## 7. 앞으로의 연구 방향 및 고려사항

### 7.1 2025년 이후 주요 연구 트렌드[5][4][2]

1. **효율성-성능 균형**
   - 모바일 및 엣지 기기 배포를 위한 경량화 기술 (양자화, 프루닝, 지식 증류) 발전
   - 8비트 양자화와 같은 기법으로 메모리 사용 40-50% 감소

2. **장문 맥락 처리**
   - 기존 $$O(n^2)$$ 복잡도 극복 연구 활발
   - 제한된 어텐션 및 스파스 어텐션 기법 고도화

3. **해석 가능성(Interpretability) 강화**
   - Attention visualization 그 이상의 메커니즘 분석 필요
   - Generalized Attention Flow (GAF) 등 새로운 기여도 분석 방법 개발[10]

4. **추론 능력의 근본적 한계 극복**
   - 단순 in-context learning 뛰어넘기
   - Chain-of-thought 및 자가 검증 메커니즘 발전[5]

5. **로보틱스 등 다양 영역 확대**
   - Robotics Transformer (RT-1): 다양한 로봇 작업에서 제로샷 일반화 시도[11]

### 7.2 연구 시 고려할 점[8][7][2]

1. **데이터 품질과 다양성의 중요성**
   - Transformer는 대규모 데이터로부터 패턴 학습
   - 편향된 데이터는 편향된 모델 생성

2. **전이 학습의 한계**
   - 사전 학습 작업과 목표 작업의 유사성 중요
   - 도메인 이동(domain shift)에 대한 정규화 필요

3. **계산 자원 고려**
   - Vision Transformers의 높은 계산 요구 사항
   - 중소기업의 접근성 문제

4. **모델 해석 가능성 확보**
   - 보건, 법률, 금융 등 고위험 분야에서 필수
   - 현재 어텐션 가중치 시각화 이상의 방법 개발 필요

***

## 결론

**"Attention Is All You Need"** 논문은 단순히 번역 작업의 성능 향상을 넘어, **AI 산업 전체의 패러다임을 전환**시킨 혁명적 논문입니다. 병렬 처리 가능성, 장거리 의존성 학습의 용이성, 강력한 일반화 능력을 갖춘 Transformer는 자연어 처리, 컴퓨터 비전, 멀티모달 학습, 로보틱스 등 극히 다양한 영역에서 기본 아키텍처로 채택되었습니다.

현재 2025년 연구 동향은 이 기초 위에서 **효율성, 확장성, 해석 가능성**을 동시에 추구하고 있습니다. 특히 일반화 능력의 근본적 한계를 인식하고 이를 극복하기 위한 노력이 활발히 진행 중입니다. 향후 연구에서는 Transformer의 강점을 유지하면서 이러한 한계들을 점진적으로 해결해 나가는 것이 핵심 과제로 남아있습니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/15a26092-dfc0-48b9-96f3-00c6f57335b4/1706.03762v7.pdf)
[2](https://funes-days.com/dev/transformer-revolution-2025-ai-evolution/)
[3](https://aclanthology.org/2023.findings-emnlp.936.pdf)
[4](https://www.themoonlight.io/ko/review/looped-transformers-for-length-generalization)
[5](https://discuss.pytorch.kr/t/2025-10-27-11-02-ai-ml/8104)
[6](http://arxiv.org/pdf/2207.05501.pdf)
[7](https://blog.outta.ai/248)
[8](https://www.fortunebusinessinsights.com/ko/vision-transformers-market-112365)
[9](https://real-st-ory.tistory.com/49)
[10](https://bart0401.tistory.com/62)
[11](https://hsejun07.tistory.com/436)
[12](https://arxiv.org/pdf/2106.01548.pdf)
[13](http://arxiv.org/pdf/2408.04413.pdf)
[14](http://arxiv.org/pdf/2405.19284.pdf)
[15](https://arxiv.org/pdf/2502.20525.pdf)
[16](https://arxiv.org/pdf/2306.09364.pdf)
[17](http://arxiv.org/pdf/2207.07827.pdf)
[18](https://arxiv.org/pdf/2302.08374.pdf)
[19](https://arxiv.org/pdf/2310.10930.pdf)
[20](https://hanaoverride.tistory.com/6)
[21](https://ettrends.etri.re.kr/ettrends/204/0905204002/012-022.%20%EA%B9%80%ED%98%9C%EC%A7%80_204%ED%98%B8.pdf)
[22](https://blog.outta.ai/119)
[23](https://www.themoonlight.io/ko/review/analyzing-generalization-in-pre-trained-symbolic-regression)
[24](https://namu.wiki/w/Attention%20Is%20All%20You%20Need)
[25](https://velog.io/@dutch-tulip/decision-transformer)
[26](http://arxiv.org/pdf/2502.03417.pdf)
[27](http://arxiv.org/pdf/2411.03697.pdf)
[28](https://aistudy9314.tistory.com/14)
[29](https://blog.kakaocloud.com/91)
[30](https://datasciencebeehive.tistory.com/105)
[31](https://wikidocs.net/31379)
[32](https://yongproblog.tistory.com/45)
