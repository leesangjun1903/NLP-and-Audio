# An Empirical Exploration of Recurrent Network Architectures

### 1. 핵심 주장과 주요 기여

본 논문은 **LSTM 아키텍처의 최적성에 대한 실증적 검증**을 목표로 한다. 저자들의 주요 주장은 다음과 같다:[1]

**핵심 주장:**
- LSTM의 설계가 임의적(ad-hoc)이며, 개별 구성 요소의 중요도가 명확하지 않음
- 광범위한 아키텍처 탐색을 통해 LSTM보다 우수한 구조가 존재할 수 있음
- LSTM의 포겟 게이트 바이어스를 1로 초기화하는 간단한 조정만으로도 성능 격차를 상당히 줄일 수 있음

**주요 기여:**
- 10,000개 이상의 RNN 아키텍처를 평가한 광범위한 아키텍처 탐색 수행[1]
- GRU(Gated Recurrent Unit)와 유사하지만 성능이 우수한 3가지 변형 아키텍처 발견
- LSTM의 각 게이트 구성 요소의 중요도에 대한 절제 실험(ablation study) 수행
- 포겟 게이트가 언어 모델링을 제외한 모든 작업에서 극도로 중요함을 입증

***

### 2. 해결하고자 하는 문제 및 제안 방법

#### 문제 정의

RNN 훈련의 근본적 문제는 **기울기 소실(vanishing gradient) 문제**와 **기울기 폭발(exploding gradient) 문제**다. 표준 RNN에서:[1]

$$\nabla h_t = \frac{\partial L}{\partial h_t} \prod_{s=1}^{t-1} \frac{\partial h_{s+1}}{\partial h_s}$$

여기서 $\frac{\partial h_{s+1}}{\partial h_s} = W_{rh}$이므로, 기울기는 가중치 행렬의 거듭제곱으로 표현되어 지수적으로 증가 또는 감소한다.[1]

#### 제안된 해결 방법

**LSTM의 핵심 아이디어:** 표준 RNN의 상태 $S_t$를 직접 계산하는 대신, 변화량 $\Delta S_t$를 계산하여 누적하는 방식 도입:

$$S_t = S_{t-1} + \Delta S_t$$

이러한 재매개변수화는 모델의 표현력을 증가시키지는 않지만, 기울기가 누적되어 소실되지 않게 한다.[1]

#### LSTM 수식

논문에서 사용한 LSTM 아키텍처는 다음과 같다:[1]

$$i_t = \tanh(W_{xi}x_t + W_{hi}h_{t-1} + b_i)$$

$$j_t = \mathrm{sigm}(W_{xj}x_t + W_{hj}h_{t-1} + b_j)$$

$$f_t = \mathrm{sigm}(W_{xf}x_t + W_{hf}h_{t-1} + b_f)$$

$$o_t = \tanh(W_{xo}x_t + W_{ho}h_{t-1} + b_o)$$

$$c_t = c_{t-1} \odot f_t + i_t \odot j_t$$

$$h_t = \tanh(c_t) \odot o_t$$

여기서 $i_t$는 입력 게이트, $j_t$는 입력 변환, $f_t$는 포겟 게이트, $o_t$는 출력 게이트, $c_t$는 셀 상태, $h_t$는 숨겨진 상태를 나타낸다.[1]

#### 아키텍처 탐색 절차

**탐색 공간 표현:** 계산 그래프로 표현되며, 각 노드는 활성화 함수 또는 원소별 연산(덧셈, 곱셈, 뺄셈)이다.[1]

**돌연변이 연산:** 6가지 변환을 확률 $p$로 적용:
1. 활성화 함수 교체 (tanh, sigmoid, ReLU, Linear 변형)
2. 원소별 연산 교체 (곱셈, 덧셈, 뺄셈)
3. 현재 노드와 부모 사이에 활성화 함수 삽입
4. 입출력이 각각 하나인 노드 제거
5. 현재 노드를 선조 노드로 교체
6. 임의의 노드와 현재 노드의 선조를 선택하여 합, 곱, 차로 결합[1]

**성능 평가 메트릭:**

$$\min_{\text{task}} \frac{\text{architecture's best accuracy on task}}{\text{GRU's best accuracy on task}}$$

이는 모든 작업에서 우수한 성능을 보이는 아키텍처를 찾도록 보장한다.[1]

---

### 3. 발견된 모델 구조

#### 발견된 변형 아키텍처

**MUT1 (최고 성능):**

$$z_t = \mathrm{sigm}(W_{xz}x_t + b_z)$$

$$r_t = \mathrm{sigm}(W_{xr}x_t + W_{hr}h_t + b_r)$$

$$h_{t+1} = \tanh(W_{hh}(r_t \odot h_t) + \tanh(x_t) + b_h) \odot z_t + h_t \odot (1-z_t)$$

**MUT2:**

$$z_t = \mathrm{sigm}(W_{xz}x_t + W_{hz}h_t + b_z)$$

$$r_t = \mathrm{sigm}(x_t + W_{hr}h_t + b_r)$$

$$h_{t+1} = \tanh(W_{hh}(r_t \odot h_t) + W_{xh}x_t + b_h) \odot z_t + h_t \odot (1-z_t)$$

**MUT3:**

$$z_t = \mathrm{sigm}(W_{xz}x_t + W_{hz}\tanh(h_t) + b_z)$$

$$r_t = \mathrm{sigm}(W_{xr}x_t + W_{hr}h_t + b_r)$$

$$h_{t+1} = \tanh(W_{hh}(r_t \odot h_t) + W_{xh}x_t + b_h) \odot z_t + h_t \odot (1-z_t)$$

이 아키텍처들은 모두 GRU와 유사한 구조를 가지고 있으면서도 특정 작업에서 우수한 성능을 보인다.[1]

#### GRU와의 비교

**GRU 방정식:**

$$r_t = \mathrm{sigm}(W_{xr}x_t + W_{hr}h_{t-1} + b_r)$$

$$z_t = \mathrm{sigm}(W_{xz}x_t + W_{hz}h_{t-1} + b_z)$$

$$\tilde{h}_t = \tanh(W_{xh}x_t + W_{hh}(r_t \odot h_{t-1}) + b_h)$$

$$h_t = z_t \odot h_{t-1} + (1-z_t) \odot \tilde{h}_t$$

***

### 4. 성능 향상 및 평가 결과

#### 작업별 성능 비교

| 아키텍처 | 산술(Arithmetic) | XML 모델링 | PTB (언어 모델링) |
|---------|-----------------|-----------|-----------------|
| Tanh RNN | 0.295 | 0.321 | 0.088 |
| **LSTM** | 0.892 | 0.425 | 0.089 |
| **LSTM-f** (포겟 게이트 제거) | 0.293 | 0.234 | 0.088 |
| **LSTM-i** (입력 게이트 제거) | 0.751 | 0.414 | 0.087 |
| **LSTM-o** (출력 게이트 제거) | 0.867 | 0.421 | 0.089 |
| **LSTM-b** (포겟 바이어스=1) | **0.902** | 0.444 | 0.090 |
| **GRU** | 0.896 | 0.460 | 0.091 |
| **MUT1** | 0.921 | 0.475 | 0.090 |
| **MUT2** | 0.898 | 0.473 | 0.090 |
| **MUT3** | 0.907 | 0.465 | 0.092 |

**주요 결과:**[1]
- MUT1이 산술과 XML 작업에서 최고 성능 달성
- LSTM-b (포겟 바이어스 초기화)가 GRU와의 성능 격차를 상당히 감소
- 언어 모델링 작업에서는 전체 아키텍처 간 성능 차이가 미미

#### 포겟 게이트 바이어스의 중요성

**핵심 발견:** 포겟 게이트 바이어스를 적절히 초기화하지 않으면 심각한 문제 발생:

- 표준 초기화: $f_t = \mathrm{sigm}(0) = 0.5$ → 타임스텝당 기울기 감쇠 인수 0.5
- 장기 의존성이 있는 문제에서 기울기 소실 초래
- $b_f = 1$ 초기화: $f_t \approx 0.73$ (시그모이드 함수 근처) → 기울기 흐름 개선[1]

이는 논문 내에서 가장 실용적인 기여로, Gers et al. (2000)에서 제안되었으나 최근 LSTM 논문들에서 언급되지 않았던 방법이다.[1]

#### 절제 연구 결과

**게이트별 중요도:**[1]
- **입력 게이트**: 중요 (제거 시 대부분 작업에서 성능 저하)
- **포겟 게이트**: 매우 중요 (특히 산술, XML 작업에서 극적 성능 저하, 단 언어 모델링 제외)
- **출력 게이트**: 가장 덜 중요 (제거해도 $h_t = \tanh(c_t)$로만 단순화되어 대부분 성능 유지)

언어 모델링에서 포겟 게이트의 중요도가 낮은 이유는 정보를 누적하기 위한 적분 장치와 같은 역할이 불필요하기 때문으로 해석된다.[1]

***

### 5. 모델의 일반화 성능 향상 가능성

#### 일반화 성능 분석

**훈련 외 데이터셋 평가:**

음악 데이터셋(Nottingham, Piano-Midi)에서 아키텍처의 일반화 능력 검증:[1]

| 아키텍처 | Nottingham | Nottingham-Dropout | Piano-Midi |
|---------|-----------|------------------|----------|
| **LSTM** | 3.492 | 3.403 | 6.866 |
| **MUT1** | 3.254 | 3.376 | 6.792 |
| **LSTM-b** | 3.419 | 3.345 | 6.820 |

MUT1이 음악 데이터셋에서도 우수한 성능을 보이며, 이는 아키텍처의 안정적인 일반화 능력을 시사한다.

**드롭아웃과의 상호작용:**

흥미로운 발견으로, 드롭아웃 정규화를 적용할 때는 **LSTM 변형이 최고 성능**을 보였다. 이는 정규화 강도에 따라 아키텍처 선택의 최적성이 변할 수 있음을 의시한다.[1]

**일반화 향상의 핵심 요인:**

1. **아키텍처 구조 자체**: 적절한 게이트 메커니즘이 다양한 작업에서 일관된 성능 제공
2. **초기화 전략**: 포겟 바이어스 초기화가 기울기 흐름과 수렴 안정성을 개선하여 일반화 성능 향상
3. **작업별 특수성**: 언어 모델링 같은 특정 작업은 포겟 게이트의 중요도가 낮아 단순한 구조도 효과적

#### 장기 의존성 학습 능력

**아키텍처 심화 메커니즘:**

LSTM의 메모리 셀 $c_t = c_{t-1} \odot f_t + i_t \odot j_t$는 상태의 누적을 통해:

- 장기 의존성에서 기울기가 "스며들지만" 완전히 소실되지 않음
- 포겟 게이트가 $f_t \approx 1$일 때 정보 보존이 강화됨
- 이를 통해 수백~수천 타임스텝의 의존성도 학습 가능[1]

#### 일반화 성능 한계

**한계점:**

1. **아키텍처 다양성 부족**: 탐색 결과 모두 GRU 유사 구조 (더 다양한 아키텍처 발굴 실패)[1]
2. **작업 특수성**: "모든 작업에서 우수한" 아키텍처가 존재하지 않음 (문제별 최적 구조 상이)
3. **드롭아웃 상황에서 변동성**: 정규화 강도에 따라 최적 아키텍처가 변함

***

### 6. 한계 및 비판

#### 논문의 주요 한계

**1. 제한된 탐색 공간:**[1]
- 계산 그래프 기반 표현으로 인해 혁신적 아키텍처 발견이 어려움
- 대부분의 우수 아키텍처가 GRU와 유사 구조로 수렴

**2. 작업 의존성:**[1]
- 최적 아키텍처가 작업별로 상이 (산술, XML, 언어 모델링 각각 다름)
- 모든 조건에서 LSTM/GRU를 능가하는 단일 아키텍처 불가능

**3. 계산 비용:**[1]
- 230,000개의 하이퍼파라미터 구성 평가로 매우 높은 계산 비용
- 실무적 적용에는 한계

**4. 이론적 근거 부족:**
- 아키텍처 탐색의 결과가 경험적(empirical)일 뿐, 왜 특정 구조가 우수한지에 대한 깊이 있는 이론적 설명 부재

#### 절제 실험의 한계

- 개별 게이트의 중요도만 분석 (게이트 간의 상호작용 미분석)
- 드롭아웃 적용 시 이상 현상의 원인 규명 불완전

***

### 7. 최신 연구 기반 영향 및 고려사항

#### A. 논문의 이후 영향 (2015-2025)

**1. RNN 아키텍처 연구의 기반 마련:**[2][3][4]

논문의 광범위한 아키텍처 탐색 방법론은 이후 신경망 아키텍처 탐색(NAS) 분야의 표준 방법론 수립에 기여했다. 특히 진화 알고리즘 기반 아키텍처 탐색(evolutionary NAS)의 체계적 접근 방식을 제시했다.[3][2]

**2. 포겟 게이트 바이어스 초기화의 확산:**[5][6]

- 이 논문 이후 LSTM 구현에서 포겟 게이트 바이어스를 큰 양수값(1.0, 2.0)으로 초기화하는 것이 표준 관행으로 정착
- 최신 딥러닝 프레임워크(PyTorch, TensorFlow)의 기본 권장사항에 포함[5]

**3. 게이트 메커니즘 개선 연구로 확대:**[4][7]

2020년 Gu et al.은 이 논문의 게이트 초기화 개념을 확장하여 "uniform gate initialization (UGI)"을 제안했다. 이는 포겟 게이트 활성화를 광범위한 분포에서 직접 초기화하여, 다양한 의존성 길이(timescale)를 모델이 학습 가능하도록 했다.[7]

$$f_t^{(\text{UGI})} \sim U[\text{wide range}]$$

이를 통해 기울기 흐름이 대폭 개선되었다.[4]

#### B. Transformer와 Attention 기반 모델의 부상

**장기 의존성 문제의 구조적 해결:**[8][9]

2017년 "Attention is All You Need" 이후, Transformer의 자기 주의(self-attention) 메커니즘이 RNN의 기울기 소실 문제를 구조적으로 해결했다. 이는:

- 타임스텝 $t$와 $s$ 간 직접 연결: $\alpha_{t,s} = \frac{\exp(q_t \cdot k_s)}{\sum_j \exp(q_t \cdot k_j)}$
- 기울기 경로 길이: $O(1)$ (RNN의 $O(t)$에 비해 극적 개선)
- 결과: 장기 의존성 학습에서 RNN 대비 현저한 성능 향상[8]

이는 이 논문의 기울기 흐름 개선 시도를 기술적으로 우회하는 새로운 패러다임을 제시했다.

#### C. RNN의 재평가 및 최신 동향 (2023-2025)

**State Space Models (SSM)와 Mamba의 등장:**[10][11][12]

Transformer의 이차 계산 복잡도 문제(긴 시퀀스에서 $O(n^2)$ )를 해결하기 위해, 최근 RNN 기반 구조가 재조명되고 있다:

- **Mamba (2024):** 선택적 SSM 기반 아키텍처로, RNN의 선형 복잡도( $O(n)$ )와 Transformer 수준의 성능을 결합[11][12]
- **RWKV (2023):** 변수 폭 가중치 기반 RNN으로, 병렬 훈련과 선형 시간 추론 제공[2]
- 음성, 게놈 데이터 등 초장거리 시퀀스 처리에서 RNN 계열의 우월성 회복[11]

**길이 외삽(Length Extrapolation) 연구:**[13][14]

최신 연구에서는 이 논문이 다루지 않은 **길이 일반화(length generalization)** 문제에 집중하고 있다:

- 훈련 시퀀스 길이: 2K, 테스트 시퀀스 길이: 128K 같은 시나리오 처리[14]
- RNN의 잠재 상태 분포 커버리지 증대를 통한 개선 방법 제시[13]

---

### 8. 앞으로 연구 시 고려할 점

#### 1. 아키텍처 설계 원칙

**원칙:**
- **기울기 경로 최소화:** 이 논문의 핵심 교훈인 "기울기 흐름 보존"은 여전히 중요 ( RNN 기반 최신 모델들도 동일 원칙 추구)[3]
- **게이트 초기화의 중요성:** 포겟 바이어스 초기화 같은 "작은 조정"이 큰 성능 개선을 가져올 수 있음 (현재도 미리 고려하지 않는 경우 존재)[5]

#### 2. 일반화 성능 향상을 위한 전략

**과제:**
- **도메인 특수성 인식:** 이 논문처럼 "모든 작업에 최적인 아키텍처"는 존재하지 않음 ( 표 1-3의 작업별 상이한 최고 성능 아키텍처)[1]
- **길이 외삽 능력 강화:** 최신 연구의 주요 초점 ( PANM, RAMba 같은 포인터 기반 메모리 아키텍처 등장)[14][13]
- **정규화 강도별 적응:** 드롭아웃 적용 여부에 따라 최적 아키텍처가 변함 ( 표 3 참조) → 작업 특성과 정규화 강도를 고려한 아키텍처 선택 필요[1]

#### 3. 효율적 아키텍처 탐색 방법론

**현황:**
- 이 논문의 230,000회 평가는 막대한 계산 비용 소모
- 최신 NAS 기법은 성능 예측, 가중치 공유 등으로 탐색 비용을 100배 이상 절감[2][3]

**고려사항:**
- **조기 종료 전략:** 유망하지 않은 아키텍처를 빠르게 제외
- **전이 학습:** 이미 탐색된 아키텍처 정보를 새 작업에 활용
- **멀티태스크 최적화:** 특정 작업이 아닌 작업 클래스 전체에 일반화되는 아키텍처 탐색

#### 4. 이론과 실증의 결합

**필요성:**
- 이 논문은 순수 경험적 접근 (무엇이 작동하는지)만 제시
- 왜 특정 아키텍처가 작동하는지에 대한 이론적 이해 부족

**권장사항:**
- 신경망 과학(neural network science) 관점에서 게이트 구조의 수학적 분석
- 정보 이론(information theory) 기반 기울기 흐름 특성화
- 구조적 인과성(structural causality) 분석을 통한 아키텍처 설계 원칙 도출

#### 5. 하이브리드 아키텍처 탐색

**최신 트렌드:** ([9][15]

- Recurrent Transformer: RNN의 순차 처리 편향과 Transformer의 병렬처리 능력 결합
- 이 논문의 GRU/LSTM 같은 순수 RNN 아키텍처 탐색에서 벗어나, 주의(attention) 메커니즘과의 결합 탐색

**구체적 제안:**
- 계산 효율성이 중요한 분야 (엣지 디바이스, 실시간 처리): RNN 기반 최적화
- 대규모 모델 (LLM): Transformer/SSM 기반 설계 선호

#### 6. 평가 지표의 다양화

**현황:** 이 논문은 주로 정확도/손실만 평가

**개선안:**
- **길이 외삽 성능:** 훈련 길이의 2배, 10배 시퀀스에서의 성능 평가[13][14]
- **계산 효율성:** 지연시간(latency), 메모리, FLOPs 등 자원 효율성 지표 포함
- **견고성(robustness):** 노이즈, 입력 분포 이동(distribution shift) 등에 대한 일반화 능력

***

## 결론

Jozefowicz et al.의 "An Empirical Exploration of Recurrent Network Architectures"는 RNN 아키텍처의 최적성을 체계적으로 검증한 획기적 연구다. 주요 기여는:[1]

1. **포겟 게이트 바이어스 초기화의 재조명**: 간단한 조정만으로 LSTM 성능이 대폭 개선됨을 입증 → 현재 표준 실천법으로 정착[6][5]

2. **광범위한 실증 기반**: 10,000개 아키텍처 평가로 RNN 설계 공간에 대한 통찰 제공[1]

3. **게이트별 중요도 분석**: 입력, 포겟, 출력 게이트의 상대적 중요성을 정량화[1]

그러나 이 논문의 시사점은 동시에 제약성도 드러낸다:
- Transformer의 부상으로 RNN의 장기 의존성 문제가 주의 메커니즘으로 우회됨[8]
- 최신 연구 (2023-2025)는 오히려 **길이 외삽**, **효율성**, **하이브리드 설계**에 집중[10][11][14][13]

이 논문이 제시한 **기울기 흐름 보존** 원칙은 여전히 유효하며, 특히 최근 재조명되는 SSM/Mamba 같은 RNN 기반 구조에서도 동일하게 적용된다. 향후 연구는 이 기본 원칙 위에, 효율성과 일반화 능력을 동시에 달성하는 새로운 아키텍처 설계에 초점을 맞춰야 할 것이다.[12][11]

---

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4468409-9523-4307-8f0a-6d899656d058/jozefowicz15.pdf)
[2](https://arxiv.org/pdf/2108.01854.pdf)
[3](https://arxiv.org/pdf/2412.15978.pdf)
[4](http://arxiv.org/pdf/2404.15622.pdf)
[5](http://arxiv.org/pdf/2206.03010.pdf)
[6](https://arxiv.org/pdf/2402.01313.pdf)
[7](https://aclanthology.org/2023.findings-emnlp.936.pdf)
[8](https://arxiv.org/pdf/2206.09166.pdf)
[9](http://arxiv.org/pdf/2106.15295.pdf)
[10](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2024_Advances%20in%20Neural%20Architecture%20Search.pdf)
[11](https://pmc.ncbi.nlm.nih.gov/articles/PMC3217173/)
[12](https://seducinghyeok.tistory.com/8)
[13](https://academic.oup.com/nsr/article/11/8/nwae282/7740455)
[14](https://aclanthology.org/W19-0128.pdf)
[15](https://apxml.com/courses/rnns-and-sequence-modeling/chapter-4-rnn-training-challenges/weight-initialization-strategies)
[16](https://arxiv.org/html/2209.01610v3)
[17](https://arxiv.org/pdf/2507.02782.pdf)
[18](http://proceedings.mlr.press/v119/gu20a/gu20a.pdf)
[19](https://www.sciencedirect.com/science/article/pii/S0020025524003797)
[20](https://arxiv.org/pdf/1804.09849.pdf)
[21](https://www.aclweb.org/anthology/P18-1008.pdf)
[22](https://ace.ewapublishing.org/media/8858ec6a45004b298956ec9000d323d2.marked.pdf)
[23](https://arxiv.org/pdf/2305.17473.pdf)
[24](https://arxiv.org/html/2411.15671)
[25](http://arxiv.org/pdf/1807.03819.pdf)
[26](https://appinventiv.com/blog/transformer-vs-rnn/)
[27](https://arxiv.org/html/2510.17196v1)
[28](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)
[29](https://openreview.net/pdf?id=6PjZA4Jvge)
[30](https://hungleai.substack.com/p/extending-neural-networks-to-new)
[31](https://openreview.net/forum?id=tEYskw1VY2)
[32](https://arxiv.org/pdf/2402.18510.pdf)
[33](https://openreview.net/forum?id=2OEb20dy7B&noteId=Uv9ExsurEm)
[34](https://velog.io/@euisuk-chung/Paper-Review-Mamba-Linear-Time-Sequence-Modeling-with-Selective-State-Spaces)
[35](https://arxiv.org/html/2402.18510v1)
