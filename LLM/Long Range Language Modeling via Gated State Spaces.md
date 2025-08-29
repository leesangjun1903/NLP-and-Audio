# Long Range Language Modeling via Gated State Spaces

## 1. 핵심 주장 및 주요 기여
이 논문은 **Gated State Space (GSS) 레이어**를 제안하여, 기존의 구조화된 상태 공간 모델(DSS/S4) 대비  
- **학습 속도를 2–3배 가속**  
- **언어 모델링 성능(Perplexity)에서 경쟁력**  
- **제로샷 장문 일반화** 능력을 확보  
했다는 점을 핵심으로 주장한다.[1]

## 2. 해결하고자 하는 문제
Transformer 기반 모델은 $$O(L^2)$$ 복잡도로 인해 매우 긴 문맥을 처리할 때 연산 비용이 급증한다.  
기존 상태 공간 모델(SSM)은 $$O(L\log L)$$ 복잡도로 완전 병렬 학습이 가능하나,  
- DSS는 FFT 병목으로 TPU에서 느리게 동작  
- HiPPO 초기화에 민감하여 구현 복잡  
문제가 있었다.[1]

## 3. 제안 방법
### 3.1 GSS 레이어 수식
GSS는 **입력 $$X\in\mathbb R^{L\times E}$$**에 대해  

$$
U = \mathrm{GELU}(XW_1),\quad V = \mathrm{GELU}(XW_2)
$$  

$$
Y = \mathrm{DSS}(U),\quad U_{\text{ctx}} = Y W_3
$$  

$$
O = (U_{\text{ctx}}\odot V)W_4 + X
$$  

로 정의된다.  
여기서 DSS(단순화된 상태 공간 합성)는  

$$
y = K * u,\quad K = \mathrm{FFT-based\ convolution\ kernel}
$$  

구성이나, 고정 $$\Delta=1$$ 및 랜덤 초기화로 FFT 병목을 완화한다.[1]

### 3.2 모델 구조
- **GSS 블록** 16개 쌓기, 임베딩 차원 1024  
- 하이브리드: 4개마다 Transformer 블록 삽입  
- 학습 시 병렬성 유지, 추론 시 순차 RNN 모드로 60× 빠른 디코딩 가능[1]

## 4. 성능 향상 및 한계
| 데이터셋 | DSS Throughput | GSS Throughput | Perplexity 개선 |
|---|---:|---:|---:|
| PG-19 (4k) | 1.8 steps/s | 5.3 steps/s | 14.51→14.01 |
| ArXiv (4k) | 1.8 steps/s | 5.3 steps/s | 3.65→3.57 |
| GitHub (4k) | 1.8 steps/s | 5.3 steps/s | 3.65→2.68 |  
GPU/TPU 환경에서 **2–3× 학습 속도** 향상, **Perplexity 소폭 개선**을 확인했다.[1]

**한계**  
- 고정 파라미터 예산 대비 Block-Recurrent Transformer에 약간 뒤처짐  
- LaTeX·소스코드 등 특수 어휘 처리 개선 필요  
- 대형 모델에서 장문 일반화 성능 불안정

## 5. 일반화 성능 향상 관련 고찰
- **제로샷 긴 시퀀스**(최대 65k)에서 Perplexity가 오히려 감소: 문맥 활용 능력 강화  
- Gating으로 입력 차원 축소, FFT 연산 부담 감소→긴 문맥 정보 보존  
- 단일 초기화 하에서도 구조적 규모(스케일·형태)가 중요한 역할[1]

## 6. 향후 연구 영향 및 고려 사항
이 논문은 **상태 공간 모델의 실용적 확장**을 제시함으로써,  
- 대규모 언어 모델의 **장문 효율적 학습·추론** 연구 가속  
- **가벼운 하이브리드 아키텍처** 설계 영감 제공  
향후 연구 시  
- **어휘·도메인 특화 초기화** 전략  
- **하이브리드 비율 최적화**와 **추론 병렬화**  
- **장문 안정적 일반화** 위한 정규화·메모리 보강  
등을 고려해야 할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/04d98a65-e559-4ecf-b43c-42015ff90347/2206.13947v3.pdf)
