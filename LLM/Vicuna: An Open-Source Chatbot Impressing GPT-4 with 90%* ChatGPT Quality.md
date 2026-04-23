
# Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality

> **출처 / 참고자료**
> - LMSYS Blog (2023-03-30): https://lmsys.org/blog/2023-03-30-vicuna/
> - GitHub - lm-sys/FastChat: https://github.com/lm-sys/FastChat
> - UC Berkeley Sky Computing Lab – Vicuna: https://sky.cs.berkeley.edu/project/vicuna/
> - Peng et al. (2023), "Instruction Tuning with GPT-4", arXiv:2304.03277
> - Dettmers et al. (2023), "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023
> - Chia et al. (2023), "Flacuna: Unleashing the Problem Solving Power of Vicuna using Flan Fine-Tuning", arXiv:2307.02053
> - Width.ai – "How to Train and Deploy Vicuna and FastChat LLMs"
> - OpenLaboratory.ai – "Vicuna 7B"

---

## 1. 핵심 주장 및 주요 기여 요약

Vicuna-13B는 LLaMA 기반 모델을 ShareGPT에서 수집한 사용자 공유 대화로 파인튜닝한 오픈소스 챗봇으로, GPT-4 평가 기준에서 ChatGPT 및 Google Bard 대비 90% 이상의 품질을 달성하며, 훈련 비용은 약 $300에 불과하다고 주장합니다.

### 주요 기여 (4가지)

| 기여 영역 | 내용 |
|---|---|
| **저비용 오픈소스 챗봇** | $300 수준의 훈련 비용으로 ChatGPT 수준에 근접 |
| **실사용 대화 데이터 활용** | ShareGPT의 실제 사용자 대화 데이터 활용 |
| **멀티턴 대화 파인튜닝** | 기존 Alpaca 스크립트를 멀티턴 대화 및 긴 시퀀스 처리에 맞게 개선 |
| **GPT-4 기반 자동 평가 프레임워크** | LLM-as-a-Judge 방법론 제시 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2-1. 해결하고자 하는 문제

LLM의 급속한 발전으로 ChatGPT와 같은 고성능 챗봇이 등장했지만, ChatGPT의 훈련 및 아키텍처 세부 사항이 공개되지 않아 연구 및 오픈소스 혁신이 저해되는 문제가 있습니다.

이를 해결하기 위해 Meta LLaMA와 Stanford Alpaca 프로젝트에서 영감을 받아, ShareGPT.com에서 수집한 사용자 공유 대화로 LLaMA 기반 모델을 파인튜닝하여 Stanford Alpaca와 같은 다른 오픈소스 모델과 비교해 경쟁력 있는 성능을 보이는 Vicuna-13B를 소개합니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) 데이터 수집 및 전처리

Vicuna는 ShareGPT.com의 공개 API를 통해 수집된 약 70K건의 사용자 공유 대화를 파인튜닝 데이터로 사용하며, 데이터 품질 보장을 위해 HTML을 마크다운으로 변환하고 부적절하거나 저품질 샘플을 필터링합니다. 또한 긴 대화는 모델의 최대 컨텍스트 길이에 맞게 분할합니다.

#### (B) 멀티턴 대화에 대한 훈련 손실 조정

Vicuna의 학습 레시피는 Stanford Alpaca를 기반으로 하되, 멀티턴 대화를 처리하기 위해 파인튜닝 손실을 챗봇의 출력에 대해서만 계산하도록 훈련 손실을 조정합니다.

기존 Alpaca의 경우 전체 시퀀스에 대한 손실을 계산하지만, Vicuna는 아래와 같이 **어시스턴트 응답 토큰에만 손실을 적용**합니다:

$$
\mathcal{L} = -\sum_{t \in \mathcal{T}_{\text{assistant}}} \log P(x_t \mid x_{<t}; \theta)
$$

여기서:
- $x_t$: 시간 스텝 $t$에서의 토큰
- $x_{<t}$: 이전 토큰 시퀀스 (멀티턴 대화 전체 컨텍스트 포함)
- $\theta$: 모델 파라미터
- $\mathcal{T}_{\text{assistant}}$: 챗봇(어시스턴트) 응답에 해당하는 토큰 인덱스 집합

이를 통해 사용자의 발화(Human turn)는 손실 계산에서 제외되어, **모델이 어시스턴트 응답 생성에만 집중**하도록 합니다.

#### (C) 메모리 최적화

긴 컨텍스트 이해를 가능하게 하기 위해 Alpaca의 최대 컨텍스트 길이 512에서 2048로 확장하였으며, 이로 인해 대폭 증가하는 GPU 메모리 요구량을 **Gradient Checkpointing**과 **Flash Attention**을 통해 해결합니다.

**Gradient Checkpointing** 수식으로 표현하면 아래와 같습니다:

$$
\text{Memory}(\text{Backprop}) = O(n) \quad \text{(대신 재계산 비용 } O(n \log n) \text{ 발생)}
$$

일반 역전파는 $O(n^2)$ 메모리를 필요로 하지만, Gradient Checkpointing은 $O(n)$ 메모리만 사용하며 일부 중간 활성화 값을 재계산하여 이를 달성합니다.

#### (D) 훈련 인프라

훈련은 8개의 A100 GPU에서 PyTorch FSDP(Fully Sharded Data Parallel)를 사용하여 하루 만에 완료합니다.

대표적인 훈련 하이퍼파라미터로는 배치 크기 128, 학습률 $2 \times 10^{-5}$, 그리고 일반화 최적화를 위한 다중 에폭이 사용되었습니다.

---

### 2-3. 모델 구조

Vicuna는 GPT-4와 같은 다른 트랜스포머 기반 LLM과 유사한 오토리그레시브(Autoregressive) 디코더 전용(Encoder 없음) 네트워크 아키텍처를 가집니다. 여러 LLaMA 디코더 레이어로 구성되며, 각 디코더 레이어에는 멀티헤드 셀프 어텐션 블록과 MLP(Multi-Layer Perceptron)가 포함됩니다. 멀티헤드 셀프 어텐션 블록은 각 토큰이 다양한 언어적 측면에서 다른 모든 토큰에 얼마나 주의를 기울여야 하는지에 대한 정보를 담은 은닉 상태를 생성합니다. 각 레이어의 MLP 네트워크는 해당 어텐션 블록의 은닉 상태에 비선형 활성화 함수를 적용합니다.

멀티헤드 셀프 어텐션 메커니즘을 수식으로 표현하면:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

여기서 $d_k$는 키 벡터의 차원이며, $W^Q_i, W^K_i, W^V_i, W^O$는 각 헤드의 투영 행렬입니다.

모델의 입력은 최대 2,048개의 임베딩을 포함할 수 있는 토큰 임베딩 리스트이며, 출력은 각 토큰이 이전 토큰의 영향을 받는 다음 예측 토큰입니다.

---

### 2-4. 성능 평가

#### GPT-4 Judge 기반 평가

80개의 다양한 질문 세트를 생성하고, GPT-4를 활용해 모델 출력을 평가합니다. 두 모델을 비교하기 위해 각 질문에 대한 각 모델의 출력을 단일 프롬프트로 결합하고, 이 프롬프트를 GPT-4에게 전송하여 어느 모델이 더 나은 응답을 제공하는지 평가합니다.

GPT-4는 80개 질문에서 각 모델의 응답에 10점 만점의 정량적 점수를 부여하며, 점수를 모두 합산한 결과 Vicuna의 총점은 ChatGPT 총점의 **92%**에 해당합니다.

$$
\text{Score Ratio} = \frac{\sum_{q=1}^{80} s_{\text{Vicuna}}(q)}{\sum_{q=1}^{80} s_{\text{ChatGPT}}(q)} \approx 0.92
$$

80개 질문 중 45%에서 GPT-4는 Vicuna의 응답을 ChatGPT의 응답과 동등하거나 더 낫다고 평가합니다.

---

### 2-5. 한계점

이러한 최근 발전에도 불구하고, 이 챗봇들은 여전히 기본 수학 문제 해결에 어려움을 겪거나 코딩 능력이 제한적인 등의 한계를 가집니다.

이 평가 프레임워크는 챗봇을 평가하는 가능성을 보여주지만, 대형 언어 모델이 환각(hallucination)에 취약하기 때문에 아직 엄격하거나 성숙한 접근 방식은 아닙니다. 챗봇을 위한 포괄적이고 표준화된 평가 시스템을 구축하는 것은 추가 연구가 필요한 열린 과제입니다.

ShareGPT의 크라우드소싱 특성으로 인해 부정확한 답변, 개인 데이터, 악의적 답변이 모델에 유입될 수 있으며, FastChat이 데이터 정제를 시도하지만 자동화된 방식으로 콘텐츠의 부적절성을 고려하지 못합니다.

또한, 현재 Vicuna나 다른 FastChat 모델에는 강화학습 기반 인간 피드백(RLHF) 방법이 적용되지 않은 상태입니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 현재 일반화 성능의 한계

Alpaca와 Vicuna는 LLaMA의 파인튜닝 체크포인트로서 일부 특정 벤치마킹 태스크에서 ChatGPT 수준의 성능에 근접했지만, **전반적인 일반화는 여전히 달성하기 어렵습니다.** InstructEval(Chia et al., 2023)과 같은 최근 연구는 파인튜닝 데이터셋이 태스크별 성능을 결정짓는다는 점을 강하게 시사합니다.

### 3-2. Flacuna: 일반화 향상을 위한 후속 연구

Flan Collection 데이터셋에서 100만 개 인스트럭션을 샘플링한 Flan-mini 데이터셋을 구성하고, 이를 Vicuna의 대화 형식으로 변환한 후 LoRA 어댑터를 사용하여 Vicuna 체크포인트를 파라미터 효율적으로 파인튜닝함으로써 Flacuna를 도출하였습니다. 예상대로 **Flacuna는 대부분의 벤치마크 데이터셋, 특히 추론 집약적 태스크에서 Vicuna를 크게 능가합니다.**

Flacuna의 훈련 목표는 다음과 같이 표현 가능합니다:

$$
\theta_{\text{Flacuna}} = \arg\min_{\theta} \mathcal{L}_{\text{Flan-mini}}(\theta | \theta_{\text{Vicuna}})
$$

$$
\mathcal{L}_{\text{Flan-mini}} = -\sum_{(x,y) \in \mathcal{D}_{\text{Flan-mini}}} \log P_\theta(y|x)
$$

### 3-3. 데이터 스케일 확장을 통한 일반화

GPT-4 데이터 크기는 52K이고 기본 LLaMA 모델 크기는 7B이며, Vicuna는 약 70만 개의 대화 턴(멀티턴 ShareGPT 데이터에서 추정)을 수집하고 13B LLaMA 모델을 사용합니다. 따라서 GPT-4 인스트럭션 데이터를 계속 수집하고, ShareGPT 데이터와 결합하여 더 큰 LLaMA 모델을 훈련하는 것이 성능 향상에 유망합니다.

### 3-4. RLHF를 통한 일반화 강화

RLHF 관점에서, 보상 모델이 디코딩 단계에서만 사용된다는 것은 비교 데이터가 LLM 훈련에 유용한 피드백을 제공할 가능성이 있음을 시사하며, 예를 들어 기계 생성 피드백을 통한 강화학습으로 LLM을 계속 훈련하는 것이 자연스러운 방향입니다.

### 3-5. QLoRA를 통한 파라미터 효율적 일반화

Vicuna 벤치마크 결과에 따르면, QLoRA로 파인튜닝된 Guanaco 65B 모델은 GPT-4 다음으로 최고 성능을 보이며 ChatGPT 대비 **99.3%**의 성능을 달성합니다.

QLoRA의 핵심 수식:

$$
W = W_0 + \Delta W = W_0 + BA
$$

$$
W_0 \in \mathbb{R}^{d \times k}, \quad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d,k)
$$

QLoRA는 메모리를 절약하면서도 성능을 유지하기 위한 여러 혁신을 도입합니다: (a) 정규 분포 가중치에 정보 이론적으로 최적인 새로운 데이터 타입 4비트 NormalFloat(NF4), (b) 양자화 상수를 양자화하여 평균 메모리 사용량을 줄이는 이중 양자화(Double Quantization), (c) 메모리 스파이크를 관리하는 페이지 옵티마이저(Paged Optimizers)입니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 연구에 미치는 영향

#### (1) LLM-as-a-Judge 패러다임 확립
GPT-4의 능력이 벤치마크 생성 및 성능 평가를 위한 자동화 평가 프레임워크를 가능케 할 수준에 이르렀는지 연구하였으며, GPT-4가 챗봇 응답을 비교할 때 매우 일관된 순위와 상세한 평가를 생성한다는 초기 결과가 확인되었습니다.

이로 인해 LLM 평가에서 **LLM-as-a-Judge** 방법론이 후속 연구(MT-Bench, Chatbot Arena 등)의 기반이 되었습니다.

#### (2) 저비용 오픈소스 LLM 파인튜닝 생태계 촉진
LMSYS(Large Model Systems Organization)는 UC Berkeley, Stanford, UCSD, CMU 등 여러 대학의 협력에서 시작된 비영리 단체로, 오픈소스 모델, 데이터셋, 시스템 및 평가 도구를 개발하여 대형 AI 모델에 대한 접근을 민주화하는 데 초점을 맞춥니다.

#### (3) 데이터 품질 중심의 인스트럭션 튜닝 연구 촉진
Stanford Alpaca는 GPT-3.5로 생성된 52K 인스트럭션 샘플을 사용하고, Vicuna는 ShareGPT의 약 70K 대화에서 약 70만 개의 인스트럭션 샘플을 사용합니다. 이를 발전시켜, LLM을 위한 인스트럭션 튜닝의 최신 기술을 발전시키기 위해 GPT-4를 셀프 인스트럭션 튜닝의 교사로 최초 사용하는 연구가 제안되었습니다.

---

### 4-2. 앞으로 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|---|---|
| **평가 방법론의 표준화** | GPT-4 Judge 방식은 LLM의 환각 문제 및 Position Bias에 취약 |
| **데이터 품질 및 안전성** | ShareGPT 크라우드소싱 데이터의 편향, 개인정보, 유해 콘텐츠 포함 가능성 |
| **RLHF 미적용** | 보상 모델 기반 강화학습 미적용으로 인한 안전성 및 alignment 한계 |
| **컨텍스트 길이 한계** | 2,048 토큰 제한 (LongChat은 16,384 토큰으로 확장) |
| **수학 및 코딩 능력 한계** | 기본 수학 문제 해결 및 코딩 능력 부족 |
| **상업적 이용 제한** | LLaMA 라이선스 및 ShareGPT 데이터 이용 조건에 따른 비상업적 사용 제한 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델/방법 | 기반 모델 | 데이터 | 핵심 기법 | 성능 (vs ChatGPT) |
|---|---|---|---|---|
| **Alpaca** (2023) | LLaMA-7B/13B | 52K (GPT-3.5 생성) | Self-Instruct | ~낮음 |
| **Vicuna** (2023) | LLaMA-13B | 70K ShareGPT | 멀티턴 SFT, Flash Attention | ~92% |
| **Flacuna** (2023) | LLaMA (Vicuna 기반) | Flan-mini (1M) | LoRA + Flan 데이터 | Vicuna 대비 추론↑ |
| **QLoRA/Guanaco** (2023, NeurIPS) | LLaMA-65B | OASST1 | 4-bit QLoRA | ~99.3% |
| **LLaMA-GPT4** (2023) | LLaMA-7B | 52K (GPT-4 생성) | Self-Instruct (GPT-4 교사) | Vicuna와 경쟁 |
| **Vicuna v1.5** (2023) | LLaMA 2 | 125K ShareGPT | SFT on Llama 2 | MMLU, TruthfulQA↑ |

Alpaca와 Vicuna는 LLaMA의 파인튜닝 체크포인트로서 일부 특정 벤치마킹 태스크에서 ChatGPT 수준의 성능에 근접했지만, 전반적인 일반화는 여전히 달성하기 어렵습니다.

QLoRA의 결과는 4비트 QLoRA가 효과적이며 ChatGPT에 필적하는 최첨단 챗봇을 생성할 수 있음을 보여주고, Guanaco 33B 모델은 24GB 소비자용 GPU에서 12시간 이내에 훈련이 가능합니다.

Vicuna v1.5는 Llama 2에서 지도 인스트럭션 파인튜닝으로 파인튜닝되었으며, 훈련 데이터는 ShareGPT.com에서 수집된 약 125K 건의 대화입니다.

---

## 요약

Vicuna는 **저비용($300), 오픈소스, 멀티턴 대화 파인튜닝**이라는 세 가지 핵심 축으로 LLM 민주화에 크게 기여하였습니다. 그러나 전반적인 일반화 성능의 한계, RLHF 미적용, 데이터 품질 문제, 평가 방법론의 미성숙이라는 과제가 남아 있으며, 이러한 한계는 이후 QLoRA, Flacuna, LLaMA-GPT4 등 다양한 후속 연구의 출발점이 되었습니다. 향후 연구에서는 **더 엄밀한 평가 체계 확립, RLHF 통합, 데이터 다양성 강화, 그리고 더 긴 컨텍스트 처리 능력**이 핵심 과제로 고려되어야 합니다.
