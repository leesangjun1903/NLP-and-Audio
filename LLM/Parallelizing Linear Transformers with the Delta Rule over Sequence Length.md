# Parallelizing Linear Transformers with the Delta Rule over Sequence Length

## 1. 핵심 주장 및 주요 기여  
본 논문은 **DeltaNet**이라 불리는 선형 변환기(linear transformer)에 델타 규칙(Δ-rule) 업데이트를 결합한 모델을, **시퀀스 길이에 걸친 병렬 처리 알고리즘**으로 확장하여 대규모 언어 모델 학습에 실용적으로 적용할 수 있음을 보인다.  
- Δ-rule 업데이트를 통해 **기억 용량(memory capacity)** 과 **연관 연상(associative recall)** 성능을 개선  
- Householder 행렬 군의 WY 표현을 활용해, **시퀀스 차원 전체에 병렬화**된 순전파·역전파를 가능케 하는 알고리즘 제안  
- 1.3B 규모 언어 모델(100B 토큰) 학습에서 Mamba, GLA 등 기존 선형시간 모델 대비 **퍼플렉시티 및 제로샷 성능** 우수함을 입증  

***

## 2. 문제 정의·제안 기법·모델 구조·성과·한계

### 문제 정의  
- **선형 변환기**(Linear Transformer)는 KV 캐시 없이 메모리 사용량이 일정하나, **추가적 가중치 갱신(additive update)** 만으로는 시퀀스 길이가 길어질수록 과거 정보 충돌(key collision) 발생  
- 델타 규칙(Δ-rule, Widrow–Hoff 학습법)을 적용한 DeltaNet은 **예측 오차(St₋₁kₜ – vₜ)** 에 기반해 핵심-값 쌍을 갱신하므로 장기 문맥 회상에 유리  

### 제안 기법  
1. **델타 규칙 업데이트**  

$$S_t = S_{t-1} - β_t(S_{t-1}k_t - v_t)k_t^T$$  

여기서 $$β_t∈(0,1)$$은 쓰기 강도(write strength), $$S_t∈ℝ^{d×d}$$  
2. **Pseudo-value** 도입: $$u_t = β_t(v_t - S_{t-1}k_t)$$, $$S_t = \sum_{i=1}^t u_i k_i^T$$  
3. **WY 표현 기반 병렬화**  
   - 델타 업데이트는 일반화된 Householder 변환: $$S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T$$  
   - 시퀀스 차원 Chunk별로 **UT 변환** 이용:  

$$
       T = \bigl(I + \text{tril}(\mathrm{diag}(β)KK^T, -1)\bigr)^{-1}\mathrm{diag}(β),\quad
       U = TV,\ W = TK
     $$  

- 최종 Chunk 단위 전파:  

$$
       S[t+1] = S[t] + \bigl(U[t] - W[t]S[t]^T\bigr)^T K[t],\quad
       O[t] = Q[t]S[t]^T + (Q[t]K[t]^T ⊙ M)(U[t] - W[t]S[t]^T)
     $$

### 모델 구조  
- **DeltaNet Transformer**: 표준 Transformer++ 구조에서 **Self-Attention** 층을 DeltaNet 레이어로 교체  
- 각 레이어마다 Δ-rule 기반 fast-weight, 키/쿼리에 SiLU 활성화 및 L2 정규화 적용  
- **하이브리드**: DeltaNet 레이어와 슬라이딩 윈도우·글로벌 어텐션 레이어를 교차 배치  

### 성능 향상  
- **Synthetic Recall**: MQAR, MAD 벤치마크에서 GLA·Mamba 대비 recall 정확도 최대 수십 % 개선  
- **언어 모델링**:  
  - 340M 모델: Wiki perplexity 28.24 → 27.06(슬라이딩 윈도우 하이브리드), 제로샷 평균 ↑10–20%  
  - 1.3B 모델: Transformer 대비 16.85→16.55, SWDE·SQuAD·FDA recall-intensive 작업에서 유의미한 우위  
- **학습 효율**: 순차적 DeltaNet 대비 2×–16× 속도 가속 (시퀀스 길이 및 헤드 차원 증가 시 효과적)

### 한계  
- **학습 속도**: GLA 대비 여전히 느린 편—헤드 차원 내 I/O·텐서코어 비활용 오버헤드  
- **길이 일반화**: 명시적 감쇠(decay) 부재로 훈련 길이 초과 시 성능 하락  
- **스테이트 크기 확장** 제약: 대규모 모델에서 recall-intensive 작업 성능이 GLA에 일부 뒤처짐  

***

## 3. 일반화 성능 향상 관련 고찰  
- Δ-rule로 **과거 정보 충돌 완화** → 강화된 장기 기억 회상 능력  
- **UT 변환 기반 Chunkwise 병렬** 덕분에 역전파 시에도 숨겨진 상태를 재계산(recompute)하므로 대용량 시퀀스에도 일관성 유지  
- 하이브리드 구조가 국소성(local)·전역성(global) 토큰 상호작용을 보완, **길이 외삽(extrapolation)** 성능 추가 개선 가능성  

***

## 4. 향후 연구에의 영향 및 고려 사항  
- **병렬화 기법 확장**: DPLR(대각+저랭크) 변환 등 다양한 구조적 행렬에 UT/WY 표현 적용 가능성  
- **하이브리드 설계**: Δ-rule과 게이팅, 감쇠 연산 결합으로 학습·일반화 균형 맞춘 새로운 토큰 믹싱 레이어 개발  
- **하드웨어 최적화**: 텐서코어·SRAM 타일링 활용해 Δ-rule 오버헤드 최소화 연구  
- **이론적 분석**: Δ-rule의 표현력 한계와 parallel-expressiveness trade-off 해석적 이해 필요  

이상의 성과는 대규모 기억 기반 모델 설계와 병렬화 전략 연구에 새로운 전기를 제공하며, 차세대 효율적 컨텍스트 모델링 및 메모리 강화 네트워크 개발의 발판이 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a7b180d8-9c04-4117-a10d-49a999202c81/2406.06484v6.pdf
