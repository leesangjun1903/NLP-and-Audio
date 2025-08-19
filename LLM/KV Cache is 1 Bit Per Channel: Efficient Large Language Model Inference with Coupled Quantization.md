# KV Cache is 1 Bit Per Channel: Efficient Large Language Model Inference with Coupled Quantization

## 1. 핵심 주장과 주요 기여  
이 논문은 LLM 추론에서 병목이 되는 KV 캐시(Key/Value cache)의 메모리 대역폭 문제를 해결하기 위해, **채널 간 상호 의존성**을 활용한 저비트 양자화 기법인 **Coupled Quantization (CQ)**을 제안한다.  
- 기존 채널별(혹은 토큰별) 독립 양자화 방식이 1비트 수준에서 품질 붕괴를 겪는 한계를 보완  
- 채널을 그룹으로 묶어 결합 양자화함으로써, 공동 엔트로피(joint entropy)가 개별 엔트로피 합보다 느리게 증가하는 정보 이론적 이점을 활용  
- 1비트 압축(16× 축소)에서도 FP16 대비 유의미한 언어 모델 품질(Perplexity 및 downstream 과제 정확도) 보존  

## 2. 해결 과제  
- **von Neumann 병목**: 긴 문맥·대규모 배치 시 KV 캐시 메모리 읽기 비용이 연산 비용을 압도  
- **고비트 양자화 한계**: 4비트 이하 시 독립적 채널 양자화는 성능 붕괴  

## 3. 제안 방법  
### 3.1 정보 이론적 배경  
- 채널 간 **상호 정보(Mutual Information)**로 인해  

$$H(X_1, …, X_n) ≤ ∑_{i=1}^n H(X_i)$$  

- n개 채널을 함께 인코딩 시 필요한 비트 수가 개별 합보다 작음  

### 3.2 Coupled Quantization (CQ)  
- c개 연속 채널을 그룹화(채널 그룹 크기 c)  
- 각 그룹당 2^b 개의 **다차원 중심 벡터(centroids)** 학습 → CQ-c c b 표기  
- **Uniform k-means** 또는 **Fisher 정보(대각선 근사)** 기반 가중치 k-means로 centroid 학습  
- 양자화: 각 그룹 벡터를 L2 거리 기준 최단 centroid로 매핑  

#### 수식  
- Uniform:  

$$
C_i^* = \arg\min_{|C|=2^b}\|A_{i:\!i+c-1} - q(A_{i:\!i+c-1})\|_F^2
$$

- Fisher-guided:  

$$
C_i^* = \arg\min_{|C|=2^b}\sum_j g_j^\top g_j\;\|A_{i:\!i+c-1,j} - q(A_{i:\!i+c-1,j})\|_2^2
$$  

(여기서 $$g_j$$는 활성화 그라디언트의 요소별 제곱값)  

## 4. 모델 구조 및 구현  
- 대상 모델: LLaMA-7B/13B, LLaMA-2-7B/13B, Mistral-7B  
- Centroid 학습: WikiText-2 캘리브레이션 세트(16×2048 토큰)  
- 압축 설정 예: CQ-8c 8b (1비트/FPN), CQ-4c 8b (2비트/FPN)  

## 5. 성능 향상 및 한계  
|설정|비트/FPN|WikiText-2 Perplexity|C4 Perplexity|Downstream 정확도(PIQA)|
|--|--|--|--|--|
|FP16|16|5.68|7.08|78.67%|
|CQ-4c 8b|2|5.97|7.52|76.11%|
|CQ-8c 8b|1|8.09|12.13|71.16%|

- **Perplexity**: 2비트에서 비압축 대비 ≈5% 이내 열화, 1비트에서도 1-3비트 타법 대비 우수  
- **추론 오버헤드**: Sparse outlier 처리가 필요 없는 단일 dequant 연산  
- **한계**: 1비트에서 문맥 길이·배치 크기 극한 시 품질 추가 저하, centroid 학습 비용  

## 6. 모델 일반화 성능 향상 가능성  
- Fisher-guided centroid 학습이 “중요 활성화” 보존 효과로 **zero-shot** 성능 저하 최소화  
- 채널 결합 규모(c) 증가 시 일반화 지표(ARC Challenge, WinoGrande)에서도 꾸준한 성능 상승  
- 다양한 LLM 구조·도메인별 데이터 분포 차이에 따른 centroid 재학습 필요성  

## 7. 향후 연구 영향 및 고려 사항  
- **향후 연구 영향**  
  - LLM 장문 컨텍스트·대규모 배치 환경에서 추론 확장성(throughput·latency) 개선  
  - 양자화 외 차원 축소 기법이나 신경 컴펙션(neural compression) 결합 가능  
- **연구 시 고려사항**  
  - **Centroid 재학습**: 도메인·모달리티 변화에 따른 분포 drift 대응  
  - **비트-정밀도 균형**: 1비트 이하/비균등 비트 배분(uniform vs. mixed precision) 전략  
  - **추론 하드웨어 최적화**: GPU 외 CPU·특수 가속기 상에서 CQ 효율성 평가  
  - **안정성·재현성**: 양자화 불안정 구간 모니터링 및 자동 중앙값(clipping) 기법 연구

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/924733fb-4564-486f-beaf-94e34d2b4555/2405.03917v1.pdf
