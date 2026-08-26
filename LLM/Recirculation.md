# Recirculation

> **출처**: Mozer, M. C., Siddiqui, S. A., Sawyer, D., Sanyal, S., & Liu, R. (2026). *Recirculation*. arXiv:2608.17981 [cs.LG] (Google DeepMind / UT Austin). 이하 arXiv 초록·본문 발췌(arxiv.org/abs/2608.17981, arxiv.org/pdf/2608.17981, arxiv.org/html/2608.17981) 및 2차 보도자료(explainx.ai, daily.dev, KuCoin, CryptoBriefing)를 근거로 작성함.

---

## 1. Executive Summary (10문장 이내)

이 논문은 피드포워드 트랜스포머의 상태 업데이트가 모델 깊이에 의해 제한된다는 근본적 한계에서 출발하여, 모델이 동적 시스템처럼 작동하며 belief state를 추적할 수 있게 하는 특정 형태의 순환(recurrence)인 "recirculation" 기법을 제안한다.  
이는 기성 파운데이션 모델에 대해 추론 시점에서 적용하는 아키텍처적 보강으로, perplexity를 뚜렷이 낮추고 생성 및 추론(reasoning) 과제 전반의 정확도를 높인다.  
이 접근법은 생성 단계에서는 추가 지연시간이 거의 없지만, prefill 단계에서는 순차적 처리를 필요로 한다.  
저자들은 이 기법을 복잡한 추론에 적합한 chain-of-thought 계산, 그리고 인기 있는 깊이-순환 기법(looping)이나 비용이 큰 순환 트랜스포머 훈련과 구별한다.  
또한 원본 모델 가중치를 고정한 채 하이퍼파라미터만 가볍게 튜닝하는 적응형(adaptive) recirculation 변형을 제안하고 평가한다.  
기본 베이스라인 대비 적응형 recirculation은 Gemma3 계열에서 9개 데이터셋 평균 23%의 perplexity 감소, GSM8k에서 21%의 정확도 향상, 그 외 다운스트림 과제에서도 안정적인 성능 향상을 달성한다.  
저자들은 이 훈련 불필요(training-free) 접근법이 임의적 설계가 아니라 학습된 네트워크 자체의 속성에 기반한 아키텍처 진화의 새로운 경로를 시사한다고 주장한다.  
핵심 메커니즘은 깊은 층의 활성화(잔차 스트림)를 얕은 층으로 되돌려 섞는 것으로, 이는 마치 은닉 순환 신경망이 시간에 따라 상태를 갱신하는 것과 유사한 효과를 트랜스포머 내부에서 재현한다.  
논문은 2026년 8월 arXiv에 공개되었으며 Google DeepMind와 UT Austin 연구진의 공동 작업이다.

### 1-1. 연구의 목적과 필요성

언어를 이해하고 상황에 대해 추론하며 주변 세계를 모델링하기 위해서는 유동적으로 변화하는 상태(state)를 추적하는 능력이 필수적이며, 순환 신경망이나 칼만 필터 같은 전통적 접근법은 잠재 변수를 반복적·순차적으로 갱신함으로써 동역학을 포착한다. 그러나 트랜스포머는 훈련과 prefill 과정에서 병렬적으로 연산이 이루어지기 때문에 전통적 방식으로 상태를 추적하는 것이 불가능하다. 트랜스포머는 유한한 시퀀스 길이에 대해서는 영리한 해법을 학습하여 상태 추적에 효과적일 수 있지만, 실패하는 경우도 존재한다. 이 문제를 예시하기 위해 논문은 **"Fred가 강둑(bank)에 낚시하러 갔다"는 문맥에서, 후속 질문("ATM이 있을까?")에 모델이 문맥을 무시하고 '은행(bank)'으로 잘못 해석하는 사례(Figure 1)**를 제시하며 이를 "상태 추적 실패로 인한 문맥화 오류(contextualization error)의 예시"로 규정한다. 이러한 한계는 재훈련 없이 기존 모델의 추론 능력을 개선할 필요성을 낳았고, 이것이 recirculation 연구의 핵심 동기이다.

> **용어 설명**: *belief state*는 모델이 지금까지의 입력을 바탕으로 세계(또는 문맥)의 현재 상태에 대해 유지하는 내부 표현을 뜻한다. *contextualization error*는 모델이 이전 문맥 정보를 충분히 반영하지 못해 단어의 의미를 잘못 해석하는 오류를 말한다(위 "bank" 예시처럼 강둑/은행 중 잘못된 의미 선택).

---

## 2. 핵심 주장과 근거 (표)

| # | 핵심 주장 | 근거(출처) |
|---|---|---|
| 1 | 피드포워드 트랜스포머는 깊이에 의해 상태 갱신이 제한되어 순차적 상태 추적이 불가능하다 | "state updates in feedforward transformers are bounded by model depth" (초록) |
| 2 | Recirculation은 깊은 층 활성화를 얕은 층에 되돌려 섞어 순환을 도입, 추가 생성 지연 없음 | "markedly reduces perplexity...essentially no additional latency during generation, though it requires serial processing in the prefill phase" |
| 3 | 적응형 recirculation은 하이퍼파라미터만 가볍게 튜닝, 원본 가중치는 동결 | "adaptive variant of recirculation which requires only light tuning of hyperparameters while freezing the original model weights" |
| 4 | Gemma3 계열에서 perplexity 23% 감소, GSM8k 21% 정확도 향상 | "a 23% reduction in perplexity on a suite of datasets, a 21% increase in accuracy on GSM8k" (Figure 6, Figure 12 관련) |
| 5 | 최적 source-destination 층 쌍이 존재하며 α(혼합계수)가 클수록 효과가 커지지만 해로운 쌍도 늘어남 | "Figure 5 correspond to α ∈{0.04, 0.07, 0.10, 0.16}...Increasing α amplifies the effect of recirculation but results in more source-destination pairs that harm perplexity" |
| 6 | 모델 크기가 커질수록 perplexity 개선 폭도 커짐 (1B/4B: ~16%, 12B: ~35%) | "The 1B and 4B models obtain reductions up to 16% and the 12B model up to 35%" (Table, 각주 3 존재) |
| 7 | 단일 토큰 응답 과제에는 견고한 이득이 없으나, 긴 생성 과제(GSM8k)에는 유의미한 이득 | "neither recirculation nor adaptive recirculation yield robust accuracy gains on single-token response tasks, the gains for GSM8k offer a promising signal for extended generative response tasks" |
| 8 | 독립적으로 재현되었으며 Gemma 외 모델(Llama 3.2 1B)에도 효과가 확인됨 | "has already been independently reproduced...GitHub implementations have confirmed the gains on models outside the Gemma family, including Llama 3.2 1B" (비공식 3자 재현, 원 논문 주장 아님) |

---

## 2-1. 상세 설명: 문제·방법·구조·성능·한계

### (1) 해결하고자 하는 문제
표준(피드포워드) 트랜스포머는 훈련과 prefill 시 병렬 연산을 하기 때문에 전통적 RNN/칼만 필터처럼 순차적으로 상태를 갱신할 수 없다. 이는 다층 대화나 긴 문서에서 애매성이 늦게(깊은 층에서) 해소되더라도, 그 정보가 이미 지나간 얕은 층의 이후 토큰 처리에는 반영되지 못하는 구조적 한계로 이어진다(Figure 1 예시).

### (2) 제안 방법: Recirculation

논문은 "깊은 층(source layer)의 잔차 스트림 활성화를 얕은 층(destination layer)으로 되돌려 섞는" 방식을 제안한다. 검색 결과에서 확인되는 핵심 요소는 다음과 같다.

- source(원본) 층과 destination(목적) 층 사이의 관계를 히트맵으로 표현했으며, 최대 12개 층 간격까지 살펴보았다.
- 혼합 강도를 나타내는 계수 α가 클수록 recirculation의 효과가 커지지만 동시에 성능을 해치는 source-destination 쌍도 늘어난다.
- 특정 층(예: layer 4)은 5~7층 위에서 오는 정보를 받는 destination으로서 특히 바람직한 것으로 나타났다.

이를 개념적으로 표현하면(⚠️ 아래 식은 원문의 정확한 표기를 확인하지 못해 **제가 검색된 설명을 바탕으로 재구성한 근사식**이며, 논문에 실린 원본 수식과 기호가 다를 수 있음을 밝힙니다):

$$
h^{(d)}_{t} \leftarrow (1-\alpha)\, h^{(d)}_{t} + \alpha \, h^{(s)}_{t-1}
$$

- $h^{(d)}_t$: destination 층 $d$에서 시점(토큰 위치) $t$의 잔차 스트림(활성화) 벡터
- $h^{(s)}_{t-1}$: source 층 $s$($s>d$, 더 깊은 층)에서 직전 시점 $t-1$까지 계산된 잔차 스트림
- $\alpha \in [0,1]$: 혼합 계수(recirculation strength) — 클수록 과거 깊은 층 정보를 더 강하게 주입
- $t$: 순차 처리되는 토큰의 시간(순서) 인덱스 — 이 때문에 prefill이 병렬이 아니라 **순차적**이어야 함

적응형(adaptive) 변형은 고정된 α 대신 하이퍼파라미터를 가볍게 튜닝하는 방식이며, 3자 리뷰에 따르면 별도의 MLP를 arXiv·C4·PG19에서 각 250개 문서로 학습시켜 혼합 계수를 예측하도록 하는 방식(9개 평가 데이터셋에서 평균 23.0% perplexity 감소, 고정 계수 방식은 8.5%, 완전 파인튜닝은 21.6%)으로 설명된다. 이는 **논문 자체의 수식이 아니라 3자 요약**이므로 정확한 원문 수식·기호는 원문 확인이 필요하다.

### (3) looped transformer와의 비교(모델 구조)
저자들은 recirculation을 이해시키기 위해 looped transformer와 비교한다. looped transformer는 표준 아키텍처의 파라미터 효율적 변형으로, 표준 트랜스포머가 고유한 블록들을 깊게 쌓는 반면 looped transformer는 공유된 블록 집합을 여러 번 적용한다. Figure 3b는 looped transformer를 깊이 방향으로는 수직으로, 입력 스텝 방향으로는 수평으로 펼쳐서 보여주며, 1단계에서 첫 입력 토큰이 제시되고 활성화 스택이 계산된 뒤 특정 층(loop source)의 활성화가 다른 층(loop destination)으로 전달된다. Recirculation은 이 구조와 유사해 보이지만, 복잡한 추론에는 chain-of-thought가 더 적합하고, looping이나 순환 트랜스포머의 비용이 큰 훈련과는 다른 방식으로 구별된다는 점을 강조한다.

### (4) 성능 향상
- arXiv, PG19, C4 3개 데이터셋에 대해 source·destination 층을 스위핑하고 α=0.10으로 고정했을 때, 최적의 source-destination 쌍은 평균 4.72%의 perplexity 감소를 달성했다(Figure 6).
- 1B와 4B 모델은 최대 16%, 12B 모델은 최대 35%까지 perplexity 감소를 보였다.
- GSM8k에서 적응형 recirculation은 pass@1에서 8.8%, pass@128에서 20.9%의 오류율 감소를 달성했으며, 이는 모델 자체를 건드리지 않았다는 점에서 놀라운 결과로 평가된다(Figure 12).

### (5) 한계
- prefill 단계에서는 순차 처리가 요구되어 처리 비용이 늘어난다.
- 단일 토큰 응답(예: 객관식) 과제에서는 견고한 정확도 향상이 나타나지 않는다.
- 3자 분석에 따르면 완전히 훈련 불필요(training-free)한 것은 두 변형 중 하나뿐이며, GSM8k 관련 상세 결과는 21%p의 정확도 상승이 아니라 상대적 오류율 감소를 보고한 것이라는 지적이 있다.

---

## 3. 주장별 페이지/그림·표 표시

| 주장 | 근거 위치 |
|---|---|
| 상태 추적 한계 및 동기 | Abstract; Figure 1 (Fred/bank 예시) |
| looped transformer 설명 | Figure 3(a), Figure 3(b) |
| α/source-destination 히트맵 | Figure 5 (α ∈ {0.04, 0.07, 0.10, 0.16}) |
| 3개 데이터셋 층 스위핑, 4.72% 최적 감소 | Figure 6 (top row) |
| 모델 크기별 perplexity 감소(16%, 35%) | 본문 각주3 언급, 표(Table, 번호 미확인) |
| GSM8k pass@1/128 결과 | Figure 12 |

> ※ 정확한 페이지 번호는 원문 PDF 페이지 매김 확인이 필요하며, 본 요약에서는 검색 결과에 명시된 Figure/Table 번호만 확정적으로 표기함.

---

## 4. 저자 보고 결과 vs. 저의 해석 분리

**저자가 직접 보고한 것 (원문/초록 근거):**
- "adaptive recirculation achieves remarkable gains on the Gemma3 family, including a 23% reduction in perplexity...a 21% increase in accuracy on GSM8k"
- "Adaptive recirculation greatly benefits GSM8k, yielding 8.8% and 20.9% reductions in error rate with pass@1 and pass@128, respectively"
- "Our training-free approach succeeds by leveraging the model itself to inform architectural modifications, suggesting a route to architectural evolution guided by a trained network's properties"

**저의 해석/종합 (원문에 명시되지 않은 추론):**
- 초록의 "21% 증가"라는 표현과 본문의 "8.8%/20.9% 오류율 감소"라는 표현이 정확히 어떻게 산술적으로 연결되는지는 3자 분석(explainx.ai)에서 "초록은 21% 증가라고 말하지만, 상세 결과는 pass@128에서 20.9% 오류율 감소를 보고하며, 이는 정확도의 퍼센트 포인트 상승이 아니라 남은 오류의 상대적 감소"라는 지적이 있음 — 즉 **초록의 수치와 본문 상세 수치 사이에 해석 차이(상대적 vs 절대적 변화)가 존재할 수 있다는 점**은 원문 자체보다 3자 검증에서 드러난 것이므로, 독자는 이를 구분해서 읽을 필요가 있다.
- 논문이 α=0.10 고정 조건에서의 4.72% 개선을 "대표값"으로 제시했지만, 최고 성능치(16~35%)는 층 쌍과 모델 크기를 별도로 최적화한 조건에서 나온 것이므로, 이 둘을 동일선상에서 비교하는 것은 저의 해석상 주의가 필요하다(§5 참고).

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

1. **하이퍼파라미터 탐색 데이터와 평가 데이터의 경계**: 1B, 4B, 12B 모델 각각에 대해 튜닝 세트에서 평균 percentage perplexity 감소가 가장 큰 최적 source-destination 쌍을 찾았으며, 평가 분할은 하이퍼파라미터 튜닝에 사용된 'training' 분할과는 별도라고 명시되어 있으나, 같은 3개 데이터셋(arXiv, C4, PG19) 내에서의 분할이므로 완전히 독립적인 도메인 일반화 검증이라 보기는 어렵다.
2. **모델 크기별 최대 개선폭(16%, 35%) 비교**: 이는 각 모델별로 최적화된 층 쌍에서 나온 최댓값이므로, 모델 간 "일관된 개선 배율"로 해석하기보다는 최적 조건에서의 상한선으로 봐야 하며, 단순 비교는 통계적으로 취약하다.
3. **적응형 recirculation vs 완전 파인튜닝 비교**: 3자 리포트에 따르면 적응형 recirculation이 23%의 평균 perplexity 감소를, 완전 파인튜닝은 21.6% 감소를 달성했다고 비교되는데, 두 방법은 계산 비용·데이터 활용 방식이 근본적으로 다르므로(파인튜닝은 가중치 전체 갱신, recirculation은 소수 하이퍼파라미터만 조정) 단순 수치 비교는 공정한 비교라 보기 어렵다.
4. **GSM8k pass@128 지표**: pass@128의 20.9% 오류율 감소는 "정확도의 퍼센트 포인트 상승이 아니라 남은 오류의 상대적 감소"이므로, 초록에 언급된 "21% 증가"와 문자 그대로 동일한 지표가 아닐 가능성이 있다.
5. **재현성 관련 수치**: 비공식 GitHub 재현에서 −7.9%의 perplexity 개선을 보고했는데, 이는 원 논문의 −14.4%와 차이가 있으며, 그 이유로 평가 문서 수(전체 테스트셋 50개 문서 vs 재현 실험의 6개 문서, 18개 윈도우)의 차이가 제시된다. 이는 표본 크기가 작아 통계적으로 취약한 재현 결과이며, 원 논문 수치와 직접 비교하기엔 무리가 있다.

---

## 6. 문서가 답하지 않는 질문

- 적응형 recirculation의 하이퍼파라미터(혼합 계수 예측 MLP 등)를 튜닝하는 데 필요한 계산 비용이 전체 파인튜닝 대비 정확히 얼마나 절감되는지에 대한 정량적 총비용(FLOPs, GPU-시간) 비교는 검색 결과에서 확인되지 않는다.
- Gemma3 계열 외의 완전히 다른 아키텍처(예: Mixture-of-Experts, State Space Model 계열)에 recirculation을 적용했을 때의 결과는 원 논문에서 다루었는지 불명확하다(3자 재현에서만 Llama 3.2 1B 언급).
- 매우 긴 컨텍스트(수만~수십만 토큰) 환경에서 recirculation의 순차적 prefill 비용이 실제 서비스 환경에서 어느 정도의 지연을 유발하는지에 대한 실측 벤치마크는 검색 결과에서 확인되지 않는다.
- Recirculation이 안전성(safety), 환각(hallucination) 감소, 편향(bias) 등 비-정확도 지표에 미치는 영향은 언급되지 않는다.
- 왜 특정 층 쌍(예: layer 4를 목적지로, 5~7층 위를 원본으로)이 최적인지에 대한 이론적·기계적 설명(왜 그 위치인지)은 검색된 범위 내에서 완전히 설명되지 않는다.

---

## 7. 가장 중요한 그림 5개의 해석

1. **Figure 1 (Fred/낚시터 대화 예시)**: Fred가 낚시하러 강둑(bank)에 갔다는 문맥에서, 후속 질문("ATM이 있을까?")에 모델이 이를 금융기관 bank로 잘못 해석하여 ATM이 있을 것이라 답하는 예시는 상태 추적 실패(문맥화 오류)를 시각적으로 보여주는 동기 부여 그림이다. 즉, 깊은 층에서 이미 "bank=강둑"이라는 의미 해소가 일어났어도 그 정보가 얕은 층의 후속 처리에 전달되지 않아 오류가 발생함을 보여준다.

2. **Figure 3 (표준 vs looped transformer 구조도)**: looped transformer가 표준 트랜스포머와 달리 공유 블록을 반복 적용하는 구조임을 깊이·입력 스텝 두 축으로 펼쳐 보여주며, recirculation과의 구조적 유사점과 차이점(순환의 방향과 타이밍)을 명확히 하는 데 사용된다.

3. **Figure 5 (α별 source-destination 히트맵)**: 4개의 α 값(0.04, 0.07, 0.10, 0.16)에 대해 source/destination 층 조합별 perplexity 변화를 파랑-빨강 색상으로 나타내며, α가 커질수록 효과는 커지지만 해로운 조합도 늘어나고, layer 4가 5~7층 위의 정보를 받는 destination으로서 바람직함을 시각적으로 보여준다. 이는 recirculation의 하이퍼파라미터(층 쌍, 혼합 강도) 선택이 성능에 얼마나 민감한지를 보여주는 핵심 그림이다.

4. **Figure 6 (3개 데이터셋 층 스위핑 및 평균 % 변화)**: arXiv, PG19, C4 세 데이터셋에서 α=0.10으로 고정하고 층을 스위핑한 뒤, 절대 perplexity를 기준선 대비 백분율 변화로 변환해 평균낸 결과, 최적 쌍은 평균 4.72%의 perplexity 감소를 보였다. 이는 특정 α·층 조합이 여러 도메인(코드/논문/웹텍스트)에 걸쳐 일반화되는지를 검증하는 그림이다.

5. **Figure 12 (GSM8k에서의 적응형 recirculation 성능)**: 각 패널의 녹색 막대가 적응형 recirculation의 GSM8k 성능을 나타내며, pass@1에서 8.8%, pass@128에서 20.9%의 오류율 감소를 보여, 모델 가중치를 전혀 건드리지 않고도 상당한 추론 성능 향상이 가능함을 입증하는 핵심 실증 그림이다.

---

## 8. 결론: 시사점, 후속 연구, 일반화 가능성, 최신 연구 비교

저자들은 recirculation을 훈련 불필요한 추론 시점 아키텍처 수정으로 규정하며, 이를 통해 상태 추적을 개선할 수 있음을 보였다고 결론짓는다. 나아가 이 성과가 임의적 설계가 아니라 학습된 네트워크 자체의 속성에 기반한 아키텍처 진화의 새로운 경로를 시사한다고 주장하며, 이는 향후 모델 설계 시 "사후적으로 모델을 분석해 최적의 소규모 구조적 개입을 찾아내는" 방법론적 전환을 시사하는 시사점으로 볼 수 있다.

### 8-1. 모델의 일반화 성능 향상 가능성
- 모델 크기가 커질수록(1B/4B → 12B) perplexity 개선 폭이 오히려 커지는 경향(최대 16% → 35%)은, recirculation이 단순한 임시방편이 아니라 대형 모델의 표현력을 더 잘 활용하는 방향으로 확장 가능성(scalability)을 가질 수 있음을 시사한다. 다만 이는 특정 도메인(arXiv, C4, PG19)에서의 결과이므로, 코드·다국어·멀티모달 등 다른 데이터 분포에 대한 일반화는 추가 검증이 필요하다.
- 단일 토큰 응답 과제에서는 이득이 없고 긴 생성 과제(GSM8k)에서만 유의미한 이득이 나타난 점은, recirculation의 일반화 가능성이 "과제의 성격(순차적 상태 추적이 필요한 긴 생성 과제)"에 의존적임을 보여준다. 즉 일반화는 전 영역에 균일하게 나타나지 않고, 상태 추적이 실제로 중요한 과제에서 더 크게 나타날 가능성이 높다.

### 8-2. 2020년 이후 관련 연구와의 비교 및 향후 고려사항
- **Looped Transformer 계열**: looped transformer(Dehghani et al., 2019; Giannou et al., 2023)는 파라미터 효율성을 목표로 하지만 recirculation과 달리 학습 단계에서부터 순환 구조를 설계하는 경우가 많다. 본 논문은 이를 대조군으로 명시적으로 구별하며, **훈련 없이 추론 시에만 개입한다는 점**이 차별점이다.
- **Chain-of-Thought(CoT) 계열**: 저자들은 recirculation을 복잡한 추론에 적합한 CoT와 구분하며, recirculation은 기본적인 상태 추적(state tracking)에 특화된 방법으로 자리매김시킨다. 3자 분석 역시 recirculation이 활성화 수준에서 작동해 출력 토큰을 소모하고 지연을 더하는 CoT와는 경쟁적이라기보다 상호보완적이라고 평가한다.
- **State tracking 관련 최신 이론 연구**: 트랜스포머가 유한 시퀀스 길이에서 상태 추적을 위한 영리한 해법을 학습할 수 있음을 보인 연구들(Li et al., 2025a; Piotrowski et al., 2025; Prakash et al., 2026; Shai et al., 2024)이 인용되며, 이러한 구성적 증명은 해법의 존재 가능성만 다룰 뿐 학습 가능성(learnability)은 다루지 않는다는 한계가 지적된다. Recirculation은 이 학습 가능성 문제를 "재훈련" 대신 "추론 시 구조적 개입"으로 우회하려는 시도로 볼 수 있다.
- **향후 고려사항**: (1) prefill 단계의 순차 처리 비용이 실제 서비스 환경(특히 긴 컨텍스트·다중 사용자 서빙)에서 어떤 영향을 미치는지 정량화 필요, (2) 적응형 변형의 하이퍼파라미터/혼합계수 예측기가 도메인 밖 데이터에도 강건한지 검증 필요, (3) Gemma 외 아키텍처(MoE, SSM, Llama 등)로의 일반화는 비공식 3자 재현에서 Llama 3.2 1B에 대한 효과가 확인되었으나 공식적 검증이 필요, (4) 초록의 "정확도 증가"와 본문의 "오류율 상대적 감소"라는 표현 차이가 혼동을 줄 수 있으므로 후속 연구에서는 절대적/상대적 지표를 명확히 구분해 보고할 필요가 있다.

---

## 용어 설명 모음 (본문 등장 순)

- **belief state**: 모델이 지금까지의 입력을 바탕으로 유지하는 세계/문맥 상태에 대한 내부 표현.
- **contextualization error**: 문맥 정보를 충분히 반영하지 못해 단어의 의미를 잘못 해석하는 오류.
- **perplexity**: 언어모델이 다음 토큰을 얼마나 잘 예측하는지를 나타내는 지표로, 낮을수록 예측력이 좋음을 의미.
- **prefill**: 트랜스포머가 사용자 입력(프롬프트) 전체를 한 번에 처리해 KV 캐시를 구성하는 초기 단계.
- **KV cache**: 트랜스포머의 어텐션 계산에서 재사용을 위해 저장해 두는 Key/Value 벡터 캐시.
- **잔차 스트림(residual stream)**: 트랜스포머 각 층을 관통하며 누적되는 정보 흐름(스킵 연결로 유지되는 벡터).
- **looped transformer**: 서로 다른 고유 블록을 깊게 쌓는 대신, 동일한(공유된) 블록을 여러 번 반복 적용하는 구조.
- **chain-of-thought(CoT)**: 모델이 답을 내기 전에 중간 추론 과정을 텍스트로 명시적으로 생성하도록 하는 기법.
- **pass@k**: k번 시도 중 적어도 한 번 정답을 맞히면 성공으로 간주하는 평가 지표(k가 클수록 관대한 지표).
- **adaptive recirculation**: 고정된 혼합 계수 대신, 소규모의 추가 파라미터(예측기)를 학습시켜 혼합 강도를 상황에 맞게 조정하는 recirculation 변형.
