# ST-MoE: Designing Stable and Transferable Sparse Expert Models

## 1. 논문의 핵심 주장과 주요 기여  
이 논문은 대규모 언어 모델의 **연산 효율성**과 **전이 학습 성능**을 동시에 달성하기 위해, 스파스 전문가(Sparse Expert) 모델에 존재하던 **훈련 불안정성**과 **파인튜닝 성능 저하** 문제를 해결하는 디자인 가이드를 제시한다.  
주요 기여는 다음과 같다.  
- **훈련 안정화 기법** 대규모 실험: 다양한 안정화 기법(제곱합 손실, 노이즈 주입, 업데이트 클리핑 등)의 효과–품질 트레이드오프를 체계적으로 분석  
- **Router Z-Loss** 도입: 라우팅 네트워크의 로짓 크기를 제어하는 보조 손실을 제안해, 안정성은 높이면서 질 저하 없이 훈련 불안정성을 해소  
- **파인튜닝 전이 특성 분석**: 스파스 모델의 과적합 경향을 규명하고, 비전문 파라미터만 동결하거나 전문가 드롭아웃 등으로 일반화 성능을 개선하는 기법 검증  
- **효율적 스파스 모델 디자인**: 전문가 수, 용량 계수(capacity factor), 라우팅 알고리즘(top-n routing)의 페어토 최적 설계를 제안  
- **대규모 ST-MoE-32B**: 269B 파라미터 스파스 모델(ST-MoE-32B)을 T5-32B와 유사한 연산량으로 학습해, SuperGLUE, XSum, ARC, ANLI 등 다양한 벤치마크에서 최첨단 성능 달성  

***

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
1. **훈련 불안정성**: MoE 계열 모델은 대규모로 확장할수록 손실 발산(loss spikes)이 자주 발생  
2. **파인튜닝 일반화 격차**: 사전 학습 속도는 빠르나, 소규모·희소 데이터 파인튜닝 시 과적합 우려로 전이 성능이 밀림  

### 2.2 제안 방법  
- **Router Z-Loss**  
  라우팅 네트워크 입력 로짓 $$x\in\mathbb{R}^{B\times N}$$에 대해, 확률 분포 전 softmax 로짓의 제곱합을 보조 손실로 가중합  

$$
    L_{z} = \frac{1}{B}\sum_{i=1}^{B}\Bigl(\log\sum_{j=1}^{N} e^{x_{ij}}\Bigr)^2,
    \quad L_{\text{total}}=L_{\text{CE}}+c_{B}L_{\text{balance}}+c_{z}L_{z}
  $$  
  
  이를 통해 로짓 절댓값을 작게 유도해 지수 함수에서의 반올림 오류를 억제하고, 훈련 발산을 제거  

- **파인튜닝 일반화 강화**  
  -  전체 파라미터 대신 비전문(Non-MoE) 파라미터만 업데이트해 과적합 억제  
  -  전문가 레이어 내 드롭아웃 및 sentinel 토큰 삽입 실험을 통해 세부 프로토콜 최적화  
  -  배치 크기·학습률 최적화: 스파스 모델은 작은 배치·높은 학습률이 일반화에 유리  

- **스파스 모델 설계 원칙**  
  -  Top-2 routing + capacity factor 1.25가 TPU 환경에서 대체로 페어토 효율적  
  -  전문가 수는 accelerator당 ≦1개 권장, 평가 시 capacity factor 조정 가능  
  -  스파스–밀집 레이어 병치 및 전문가 FFN에 multiplicative bias 추가 시 품질 소폭 향상  

### 2.3 모델 구조 비교  
- **ST-MoE-L**: 4.1B 파라미터, 32 전문가, T5-Large(FLOPs 매칭)  
- **ST-MoE-32B**: 269B 파라미터, 64 전문가, T5-32B(FLOPs 매칭)  
  -  Encoder-Decoder, every-4th FFN을 MoE 레이어로 교체  
  -  bfloat16 mixed precision, Adafactor, capacity factor 1.25  

### 2.4 성능 향상  
- ST-MoE-L 대비 FLOPs 동등 T5-Large 대비 SuperGLUE +2.3%p, XSum +10%, ARC-Easy +19%p 등 전 영역에서 우세  
- ST-MoE-32B: SuperGLUE 테스트 91.2점으로 인간 수준 초과, ARC-Easy 95.2%, ARC-Challenge 86.5%, ANLI-R3 74.7% 등 SOTA  

### 2.5 한계 및 관찰  
- 소규모 데이터 과제(CB, WSC)에서는 여전히 과적합 경향  
- 디코더 전문가 특성화 부재, 라우팅 토큰 불균형에도 파인튜닝에 큰 영향 없음  
- 다국어 전문가도 언어별 특화 미미, 로드 밸런싱 손실이 다국어 설정에서 예측 불가성 유발  

***

## 3. 일반화 성능 향상 관련 심층 분석  
- **과적합 경향**: 작은 훈련셋에서 sparse 모델이 train 정확도를 빠르게 달성하지만, validation에서는 dense 모델에 뒤처짐  
- **일부 파라미터만 업데이트**: 전체 대신 비전문 파라미터만 업데이트 시 SuperGLUE 약간 개선, 메모리·속도 효율 상승  
- **드롭아웃 및 하이퍼파라미터**: global dropout 0.1, expert dropout 0.3 정도에서 최적, batch size 65k–262k, learning rate 1e-4–5e-4 사이가 전이 성능 우수  
- **토큰 드롭 내성**: fine-tuning 중 최대 15% 토큰 드롭 불구 성능 저하 미미, 오히려 regularization 효과 가능  

***

## 4. 향후 연구 영향 및 고려 사항  
- **정밀도 형식 연구**: Router Z-Loss가 불필요한 지수 범위를 압축하듯, 더욱 작은 부동소수점 형식 연구 가능  
- **라우팅 알고리즘**: Batch Prioritized Routing 등 새로운 스파스 알고리즘과 상호보완적 실험 필요  
- **다국어 특화**: 언어별 전문가 활성화 메커니즘, 글로벌 로드 밸런싱 전략 재고  
- **디코더 스파스 최적화**: 디코더 전문가 제거 또는 task-aware routing으로 효율·성능 동시 개선  
- **Adaptive Compute**: 이질적 전문가, 입력 난이도에 따른 유동적 계산량 할당 연구  

Sparse Expert 모델은 연산 효율성과 대규모 전이 학습 성능을 양립시키는 중요한 방향이다. 본 논문의 안정화·일반화 가이드는 차세대 거대 언어 모델 설계와 훈련 프로토콜 발전에 핵심 토대가 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e2dae311-732c-4685-9536-61b0e20bbafc/2202.08906v2.pdf)
