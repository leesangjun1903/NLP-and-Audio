# GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding

## 1. 핵심 주장 및 주요 기여  
GShard는 단일 기기 메모리 한계를 훌쩍 뛰어넘는 수조~수조 매개변수급의 거대 신경망을 몇 줄의 어노테이션만으로 자동 병렬화·분산 학습할 수 있도록 하는 모듈이다.  
- **조건부 연산(Conditional Computation)**: 각 입력 토큰마다 활성화되는 서브네트워크를 달리해 전체 계산량을 서브선형(sub-linear)으로 유지  
- **Mixture-of-Experts(MoE) 층 통합**: Transformer의 일부 FFN 레이어를 MoE 레이어로 대체하여 모델 용량 극대화  
- **SPMD 샤딩 자동화**: XLA 컴파일러 확장을 통해 복잡한 통신·그래프 분할 작업을 투명하게 처리  

이로써 600B 매개변수의 다국어 번역 모델을 2048 TPU v3 코어에서 4일 만에 학습 가능하며, 기존 Dense 모델 대비 성능·비용 효율을 대폭 높였다.

***

## 2. 문제 정의  
- **메모리 한계**: 단일 GPU/TPU 메모리로 수십억~수조 매개변수 모델을 다룰 수 없음  
- **계산 비용의 초선형 증가**: 모델 크기 확장 시 학습 시간이 선형 이상으로 늘어나 실용성 저하  
- **프레임워크 제약**: TensorFlow/PyTorch 수준에서 효율적인 모델 병렬화 지원 부족  
- **개발 복잡도**: 수동 모델 분할·통신 코드를 작성해야 해 엔지니어링 부담  

***

## 3. 제안 기법

### 3.1. Sparsely-Gated Mixture-of-Experts Layer  
Transformer의 FFN 레이어를 토큰별로 최대 두 전문가만 활성화하는 MoE 레이어로 대체  
수식:  

$$g_{s,e} = \mathrm{softmax}(W_g x_s)_e$$  

상위 2개 전문가 지수 $$e_1,e_2$$에 대해  

```math
y_s = g_{s,e_1}\,\mathrm{FFN}_{e_1}(x_s) + g_{s,e_2}\,\mathrm{FFN}_{e_2}(x_s)
``` 

부가적 손실로 부하 균형(auxiliary loss) $$\ell_{aux} = \frac1E\sum_e \bigl(\frac{c_e}{S}\bigr)\bar m_e$$ 를 도입해 전문가 간 처리량 편차 최소화

### 3.2. GShard 샤딩 API  
- replicate(tensor): 전체 디바이스에 복제  
- split(tensor, dim, D): dim 축을 D개 파티션으로 분할  
- shard(tensor, device_assignment): 임의 토폴로지 기반 세분화  

컴파일러는 어노테이션을 바탕으로 SPMD 변환을 수행해 AllReduce, AllToAll, CollectivePermute 등을 자동 삽입

***

## 4. 모델 구조  
- Transformer 기반 인코더-디코더  
- 전체 레이어 수 L = {12, 36, 60}  
- MoE 레이어는 “매 두 번째 FFN” 위치에 배치  
- 전문가 수 E = {128, 512, 2048} (디바이스 수와 일치)  
- 모델 차원 d_model = 1024, FFN 차원 = 8192, 헤드 수 = 16  

***

## 5. 성능 향상 및 한계

| 모델          | 파라미터 | TPU 코어 | 학습시간 | 평균 ∆BLEU |
|--------------|---------:|---------:|--------:|----------:|
| MoE(2048E,36L)|   600B  |   2048   |   4일   |   +13.5   |
| MoE(512E,36L) |   150B  |    512   |  11일   |   +12.9   |
| Dense(96L)   |   2.3B  |   2048   |  42일   |    +6.1   |

- **학습 효율**: 600B 모델이 4일·22 TPU년으로 100개 개별 모델(29 TPU년) 합산 대비 우수  
- **메모리 확장성**: 전문가 수 증가에도 디바이스당 메모리 상수 스케일링 달성  
- **연산 부하**: AllToAll 통신 비용이 $$O(\sqrt D)$$로 서브선형 증가  
- **한계**:  
  - Gate 연산 중 Cumsum 등 순차적 처리 병목  
  - Base dilation 등 복잡한 윈도우 연산의 구현 난이도  
  - 1T+ 파라미터 모델에서 수치 불안정성 관측

***

## 6. 일반화 성능 관점  
- **Positive Transfer**: 공유된 MoE 전문가가 데이터 부족 언어에서 성능 상승 견인  
- **Capacity Trade-off**: 전문가 수↑ → 고자원 언어 능력↑, 공유 범위↓ → 저자원 이점 일부 감소  
- **깊이의 역할**: L↑ 시 샘플 효율↑, 동일 파라미터로도 더 빠른 수렴  
- **과매개변수화 효과**: sub-network 활성화로 overfitting 없이 일반화 성능 지속 개선

***

## 7. 향후 연구 영향 및 고려 사항  
- **확장성**: 수조 파라미터 모델 연구에 필수적 툴킷 제시  
- **자동화 병렬화**: 모델 설계와 실행 분리, 생산성·재현성 향상  
- **일반화 최적화**: Gate 구조·손실 설계 개선 통해 부하 균형 및 전이 학습 극대화  
- **수치 안정성**: 극대형 모델 훈련 시 bfloat16 활용·정교한 스케일링 스케줄 필요  
- **응용 분야 확장**: CV, 음성, 멀티모달 등 다양한 도메인에서 GShard 패턴 적용 검토

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a86bdade-e6e6-4db1-bd23-397edd5ee4c5/2006.16668v1.pdf)
