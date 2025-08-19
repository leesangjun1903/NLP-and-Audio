# ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification

**핵심 주장**  
ZipCache는 대규모 언어 모델(LLMs)의 KV(Cache) 메모리를 효율적으로 압축하면서도 정확한 핵심 토큰(salient token)을 식별하여 모델 성능 저하 없이 높은 압축률과 추론 속도 향상을 동시에 달성할 수 있음을 입증한다.

**주요 기여**  
1. **채널 분리 토큰 단위 양자화(Channel-separable tokenwise quantization)**  
   - 채널별 최대값으로 정규화 후 토큰 단위로 양자화하고, 다시 스케일링하여 outlier 영향을 완화.  
   - 그룹 단위 양자화 대비 양자화 파라미터 수를 대폭 줄여 메모리 오버헤드를 절감.  
2. **정규화된 주목도(normalized attention score) 기반 핵심 토큰 식별**  
   - 기존 누적 어텐션 스코어의 초기 토큰 편향을 보정하기 위해, 각 토큰이 받은 총 어텐션을 비제로 개수로 나눠 정확도 향상.  
3. **빠른 어텐션 호환 효율적 근사 기법**  
   - 전체 어텐션 스코어 대신 일부 ‘probe’ 토큰에 대해서만 표준 어텐션을 계산하고, 나머지는 FlashAttention을 활용해 속도·메모리 최적화.  
4. **종합적 성능**  
   - Mistral-7B, LLaMA2/3-8B 등에서 최대 5× 압축, 정확도 손실 <0.5% 및 37% 전·채움(prefill) 단계, 57% 디코딩 단계 지연시간 감소  

***

# 1. 해결 과제 및 제안 방법

## 1.1 해결 과제  
- **KV 캐시 메모리 병목**: 대형 LLM 서비스 시, 긴 문맥·대량 배치에서 KV 캐시가 수백 기가바이트를 차지.  
- **기존 기법의 한계**  
  - 균일 양자화: 비중요 토큰까지 고정 정밀도 적용해 압축률 제한  
  - 적응형 압축: 누적 어텐션 스코어 편향으로 핵심 토큰 식별 정확도 낮음, FlashAttention 비호환  

## 1.2 제안 방법

### 1) 채널 분리 토큰 단위 양자화  
데이터 $$X \in \mathbb{R}^{l \times h d}$$에서,  
채널 $$i$$별 스케일 $$c_i = \max(|X_i|)$$로 정규화 후  

$$
\hat X = \text{TokenQuant}\Bigl(\tfrac{X}{c}\,,\,k\Bigr),\quad
\hat X_i \leftarrow \hat X_i \times c_i
$$  

– 양자화 파라미터 수: $$hd + 2bl$$  

### 2) 정규화된 주목도 기반 핵심 토큰 식별  
어텐션 점수 행렬 $$A\in \mathbb{R}^{l\times l}$$에 대해,  
누적 점수 $$p_i=\sum_k A_{k,i}$$ 편향 보정하여  

$$
\tilde p_i=\frac{\sum_k A_{k,i}}{\mathrm{nnz}(A_{:,i})}
$$  

– 초기 토큰이 과도하게 유리해지는 편향 제거  

### 3) 효율적 근사 기법  
‘probe’ 토큰 인덱스 집합 $$P$$에 대해  

$$
A_{\mathrm{probe}}=\mathrm{Softmax}\bigl(Q_{P}K^T/\sqrt{d_k}\bigr)
$$  

근사 $$\tilde p$$ 계산 후, 나머지 어텐션 출력은 FlashAttention으로 처리  

***

# 2. 모델 구조 및 처리 흐름

1. **Prefill 단계**  
   - Probe 토큰 샘플링 → 어텐션 스코어 계산 → $$\tilde p$$ 산출  
   - FlashAttention으로 전체 출력 계산  
   - 상위 $$r\%$$ 핵심 토큰과 비핵심 토큰 분할 → 각각 고/저 비트 양자화 후 재결합  

2. **Decoding 단계**  
   - 신규 토큰 쿼리로 FlashAttention 수행 → 출력  
   - 100토큰마다 재압축, 5% 최근·5% 랜덤 probe 토큰 갱신  

***

# 3. 성능 향상 및 한계

- **성능**  
  - GSM8k: 5× 압축 시 정확도 손실 0.38% 이내  
  - Line Retrieval: 200라인 문맥에서 Mistral-7B 기준 MiKV 대비 42%p 개선  
  - GPU 메모리 20%↓, Prefill 37%↓, Decoding 57%↓  
- **한계**  
  - 핵심 토큰 비율(r%) 수작업 설정  
  - 생성 모델 안전성(악용 가능성)  

***

# 4. 일반화 성능 향상 가능성

- **적응형 양자화의 일반화**  
  - 정규화된 주목도는 토큰 중요도 편향을 제거하므로 다양한 도메인(문서 요약·대화·코드 생성 등)에서 핵심 정보 보존 강화  
- **근사 기법의 확장**  
  - Probe 샘플링 전략 연구를 통해, 더 작은 샘플링 비율로도 정확도 유지 가능성  
- **전이학습 시 활용**  
  - 사전 학습된 어텐션 분포 특성 이용해 초기 r% 자동 조정 기법 도입 시, 새로운 태스크에서도 압축-정밀도 균형 자동 최적화  

***

# 5. 향후 연구 방향 및 고려 사항

- **핵심 토큰 비율 자동화**: 데이터셋·태스크별 최적 r%를 동적 추정하는 메타 러닝 기법  
- **우발적 편향 탐지**: 정규화된 주목도도 특정 패턴에 취약할 수 있어, 편향 감지·보정 모듈 통합  
- **안전성·공정성 분석**: 압축 과정에서 주요 토큰이 특정 계층·내용에 과도하게 영향을 받는지 검증  
- **하드웨어 친화적 구현**: 다양한 가속기·라이브러리(TransformerXL·Reformer)와의 호환성 및 최적화  

ZipCache는 대용량 문맥 처리에서 메모리·속도·정확도를 동시에 개선하는 혁신적 프레임워크로, 향후 adaptive compression 및 efficient inference 연구에 중요한 이정표가 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d2ef182c-4f78-423e-b9af-e3ad3313fb86/2405.14256v1.pdf
