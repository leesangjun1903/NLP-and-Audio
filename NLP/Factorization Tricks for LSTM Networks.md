# Factorization Tricks for LSTM Networks

**핵심 주장 및 주요 기여**  
본 논문은 대규모 LSTM 네트워크의 매개변수 수와 훈련 시간을 크게 줄이면서, 유사한 성능(언어 모델 perplexity) 수준에 도달할 수 있는 두 가지 기법을 제안한다.  
1. **Factorized LSTM (F-LSTM)**: LSTM의 큰 가중치 행렬 $$W\in\mathbb{R}^{4n\times2p}$$를 두 개의 작은 행렬 $$W_1\in\mathbb{R}^{r\times2p},\,W_2\in\mathbb{R}^{4n\times r}$$의 곱으로 근사하여 매개변수를 $$\mathcal{O}(r(2p+4n))$$로 감소시킨다.  
2. **Group LSTM (G-LSTM)**: 입력 $$x_t$$와 은닉 상태 $$h_{t-1}$$를 $$k$$개의 그룹으로 분할하고, 각 그룹마다 독립적인 affine 변환 $$T_j: \mathbb{R}^{2p/k}\to\mathbb{R}^{4n/k}$$을 수행하여 매개변수를 $$\mathcal{O}(k\cdot4n\cdot2p/k^2)$$로 감소시킨다.[1]

***

## 1. 해결하려는 문제  
- **매개변수 폭발**: 대규모 LSTM은 수천만 개 이상의 가중치를 가지며, 멀티-GPU 환경에서도 훈련에 수주가 소요된다.  
- **학습 효율**: 대형 행렬 연산 병목으로 인해 분산 동기화 비용이 크다.  

***

## 2. 제안 방법 상세

### 2.1 Factorized LSTM (F-LSTM)  
- **모델 수식**  
  Affine 변환 $$T$$를  

$$
    W\,[x_t,h_{t-1}] + b
    \approx W_2\,(W_1\,[x_t,h_{t-1}]) + b,
  $$  
  
  로 근사하며, $$W_1\in\mathbb{R}^{r\times2p},\,W_2\in\mathbb{R}^{4n\times r}$$이다.  
- **매개변수 절감**  
  원본: $$2p\times4n$$  
  F-LSTM: $$r(2p +4n)$$ (단, $$r<p\le n$$)  

### 2.2 Group LSTM (G-LSTM)  
- **모델 구조**  
  입력과 은닉 상태를 $$k$$개 그룹으로 분할:  

$$
    x_t=(x_t^{(1)},\dots,x_t^{(k)}),\quad
    h_{t-1}=(h_{t-1}^{(1)},\dots,h_{t-1}^{(k)}).
  $$  
  
  각 그룹 $$j$$별 affine 변환:  

```math
    \begin{pmatrix}i^{(j)}\\f^{(j)}\\o^{(j)}\\g^{(j)}\end{pmatrix}
    =T_j\bigl([x_t^{(j)},\,h_{t-1}^{(j)}]\bigr),
    \quad T_j: \mathbb{R}^{2p/k}\to\mathbb{R}^{4n/k}.
``` 

- **매개변수 절감**  
  전체 $$k$$그룹: $$k\times(2p/k)\times(4n/k)=2p\cdot4n/k$$  

***

## 3. 모델 구조 및 성능 개선  
| 모델                      | Perplexity | RNN 파라미터 수    | 학습 속도 (words/sec) |
|---------------------------|:----------:|:------------------:|:----------------------:|
| BIGLSTM (baseline)        | 35.1       | 151M               | 33.8K                 |
| BIG F-LSTM (r=512)        | 36.3       | 52.5M              | 56.5K                 |
| BIG G-LSTM (k=2)          | 36.0       | 83.9M              | 41.7K                 |
| BIG G-LSTM (k=4)          | 40.6       | 50.4M              | 56.0K                 |
| BIG G-LSTM (k=8)          | 39.4       | 33.6M              | 58.5K                 |  
위 결과에서 F-LSTM과 G-LSTM은 매개변수를 2–3배 줄이면서 유사한 perplexity를 달성했고, 더 높은 iteration 속도로 주어진 시간 내에 빠르게 수렴함을 보였다.[1]

***

## 4. 일반화 성능 향상 가능성  
- **매개변수 절감**은 과적합 위험 감소 및 학습 속도 향상으로 이어질 수 있으며, 작은 모델이지만 충분한 표현력을 유지하면 오히려 테스트 성능이 개선될 가능성이 있다.  
- **그룹 분할**은 지역적 특징 학습을 유도해 각 그룹이 독립적인 표현을 학습하게 하므로, 모듈화된 일반화 성능이 기대된다.  

***

## 5. 한계 및 향후 연구 고려사항  
- **근사 오차**: F-LSTM의 행렬 근사는 $$r$$ 값에 따라 표현력 손실이 발생할 수 있다.  
- **그룹 수 결정**: G-LSTM의 최적 $$k$$는 데이터 특성과 모델 크기에 민감하며 자동화된 선택 방법이 필요하다.  
- **심층화 조합**: 계층별 그룹 수 변화(예: hierarchical G-LSTM)는 초기 실험에서 유망하나, 체계적 연구가 요구된다.  

***

## 6. 향후 영향 및 연구 시 고려점  
- **분산 학습 가속**: 매개변수 절감 기법은 대규모 RNN을 더 효율적으로 분산 환경에서 훈련 가능하게 한다.  
- **경량화 모델 설계**: 모바일·엣지 디바이스용 경량화 RNN 설계에 적용될 수 있다.  
- **행렬 근사의 일반화**: F-LSTM 아이디어는 Transformer나 CNN 등 다양한 아키텍처의 가중치 근사에도 확장 가능하며, 자동 rank 선정 및 동적 그룹 분할 연구가 필요하다.  

 1703.10722v3.pdf[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8076b133-728b-4fe2-81da-e86e892deae9/1703.10722v3.pdf)
