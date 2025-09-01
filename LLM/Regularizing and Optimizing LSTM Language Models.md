# Regularizing and Optimizing LSTM Language Models

**핵심 주장 및 주요 기여**  
이 논문은 **LSTM 기반 언어 모델의 일반화 성능을 크게 향상**시키기 위해 두 가지 핵심 기법을 제안한다.  
1. **Weight-Dropped LSTM**: 은닉 상태 간 연결에 DropConnect를 적용하여 과적합을 방지  
2. **NT-ASGD (Non-Monotonic Triggered ASGD)**: 평균화 시점을 검증 퍼플렉시티 비모노토닉(non-monotonic) 조건으로 자동 결정  

이와 더불어, 변동 길이 BPTT, 임베딩 드롭아웃, AR/TAR 등 부수적 정규화 기법을 조합하여 PTB와 WikiText-2 데이터셋에서 **최신 최저 퍼플렉시티**를 달성했다.[1]

***

## 1. 해결하고자 하는 문제  
- **과적합**: LSTM의 과잉 파라미터로 인해 장기 의존성 학습 시 과적합 발생  
- **학습 효율성**: 드롭아웃 등 기존 RNN 정규화 기법이 검증 성능과 학습 속도 사이에 trade-off 존재  
- **하이퍼파라미터 민감도**: ASGD의 평균화 시점을 수작업 조정해야 하는 문제  

***

## 2. 제안 방법

### 2.1 Weight-Dropped LSTM  
은닉-은닉 가중치 행렬 $$U_i,U_f,U_o,U_c$$에 DropConnect를 적용하여  

$$
\tilde{U} = U \odot M,\quad M_{jk}\sim\mathrm{Bernoulli}(1-p)
$$  

여기서 $$M$$은 고정된 마스크, $$p$$는 드롭 확률이다. 타임스텝마다 동일 마스크를 사용해 **장기 의존성 유지**와 **속도 저하 최소화**를 동시에 달성한다.[1]

### 2.2 NT-ASGD  
평균화 임계점 $$T$$를 비모노토닉 검증 퍼플렉시티 기준으로 자동 결정하는 ASGD 변형 알고리즘:  
```
if validation_perplexity(현재) > min(validation_perplexity 과거 n번):
    trigger averaging at iteration k
```
학습률 스케줄링 없이 단일 학습률로 안정적 수렴을 보장한다.[1]

### 2.3 기타 정규화 및 최적화 기법  
- **Variable BPTT Length**: 길이 $$L$$ 또는 $$L/2$$ 확률적 선택 후 정규분포로 jitter  
- **Embedding Dropout**: 단어 단위 드롭아웃  
- **Activation Regularization (AR)**: $$\alpha\|\mathbf{h}_t\|_2^2$$  
- **Temporal Activation Regularization (TAR)**: $$\beta\|\mathbf{h}\_t - \mathbf{h}_{t+1}\|_2^2$$  
- **Weight Tying**: 임베딩과 소프트맥스 가중치 공유  
- **Fine-tuning**: NT-ASGD 이후 ASGD로 추가 미세조정  

***

## 3. 모델 구조  
- **3-layer LSTM**: 히든 크기 1150, 임베딩 400  
- **DropConnect**: 은닉-은닉 행렬에 0.5 드롭  
- **드롭아웃**: 입력·중간·출력 0.4–0.65, 임베딩 0.1  
- **AR/TAR 계수**: $$\alpha=2,\ \beta=1$$  
- **배치 크기**: PTB 40, WT2 80  
- **학습**: 750 에폭, 초기 학습률 30, 그래디언트 클리핑 0.25[1]

***

## 4. 성능 향상  
| 데이터셋 | 기존 최저 퍼플렉시티 | 제안 모델 퍼플렉시티 |
|:--------:|:------------------:|:-------------------:|
| PTB(Test)   | 65.4 (Variational RHN) | **57.3** (AWD-LSTM) |
| WT2(Test)   | 65.9 (Skip-LSTM)      | **65.8** (AWD-LSTM) |
| PTB+Cache(Test) | 72.1 (LSTM+cache) | **52.8** (AWD-LSTM+cache) |
| WT2+Cache(Test) | 68.9 (LSTM+cache) | **52.0** (AWD-LSTM+cache) |

- NT-ASGD: SGD 대비 약 6–7 퍼플렉시티 절감  
- Weight-Dropped: 제거 시 11점 이상 성능 저하[1]

***

## 5. 한계 및 고려 사항  
- **하이퍼파라미터 의존성**: AR/TAR, 드롭 확률 등 경험적 튜닝 필요  
- **대규모 임베딩**: 임베딩 크기 확장 시 과적합 발생 가능성  
- **장기 문맥**: 캐시 모델 적용으로 개선됐으나 여전히 한계 존재  

***

## 6. 모델의 일반화 성능 향상 가능성  
제안된 **DropConnect 기반 은닉 정규화**와 **비모노토닉 ASGD 평균화**는  
- 다양한 시퀀스 학습 과제(번역, QA 등)에도 **모듈식**으로 적용 가능  
- **데이터 효율성**과 **추론 속도**를 유지하면서 일반화 성능을 동시에 개선  

***

## 7. 앞으로의 연구에 미치는 영향 및 고려할 점  
- **일반화 기법 확장**: Transformer 등 다른 아키텍처에서 DropConnect 활용  
- **자동 하이퍼파라미터 탐색**: Bayesian 최적화로 AR/TAR 계수 자동 결정  
- **장기 의존성 모델링**: 강화된 캐시 메커니즘 또는 메모리 네트워크 결합  
- **이론적 분석**: NT-ASGD의 수렴 특성 및 안정성 분석 심화  

이 연구는 **블랙박스 LSTM 구현**에도 적용 가능한 정규화·최적화 기법을 제시함으로써, 차후 대규모 시퀀스 학습 모델의 **일반화 성능 향상**과 **효율적 훈련**을 위한 토대를 마련했다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/334d4134-7947-4b6c-bef0-676f891d8bf2/1708.02182v1.pdf)
