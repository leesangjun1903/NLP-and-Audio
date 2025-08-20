# Atom: Low-bit Quantization for Efficient and Accurate LLM Serving

**핵심 주장 및 주요 기여**  
이 논문은 대형 언어 모델(LLM) 서비스에서 GPU 자원을 효율적으로 활용해 추론 처리량을 획기적으로 높이면서도 성능 저하를 최소화하는 저비트(4-bit 이하) 양자화 기법인 **Atom**을 제안한다. Atom의 주요 기여는 다음과 같다.  
- **혼합 정밀도 양자화(Mixed-Precision Quantization)**: 활성화(outlier) 채널만 8-bit로 유지하고, 나머지는 4-bit로 양자화해 정밀도를 보존하면서 연산 효율을 극대화  
- **그룹 단위 미세 양자화(Fine-Grained Group Quantization)**: 가중치·활성화를 소그룹(예: 128개 요소)별로 양자화해 표현 오류를 추가로 감소  
- **동적 양자화 프로세스(Dynamic Quantization)**: 입력마다 활성화 분포에 맞춰 스케일을 런타임에 계산하여 정적 양자화 대비 오차를 줄임  
- **KV-캐시 양자화(KV-Cache Quantization)**: 디코드 단계의 메모리 병목을 완화하기 위해 키·값 캐시를 저비트로 저장  

이와 같은 기법들을 CUDA 커널 수준에서 연산에 융합(fusion)·재정렬(reorder)해 추가 오버헤드를 0.5% 이내로 억제하며, 최대 7.7× 높은 토큰 처리량을 달성한다.

***

## 1. 해결하고자 하는 문제  
대형 언어 모델 추론은  
1) 배치(batch) 크기가 클수록 dense layer는 연산 집약적(compute-bound),  
2) 디코딩 단계 self-attention은 메모리 이동(memory-bound)  
이라는 상반된 제약을 가진다.  
기존의 8-bit 양자화(예: INT8 weight-only 또는 weight-activation)는  
- weight-only는 FP16으로 복원 후 연산해 연산 효율이 제한되고,  
- 8-bit weight-activation은 INT4 하드웨어를 활용하지 못해 추가 성능 여지가 크다.  
따라서 저비트(4-bit) 양자화를 통해 연산 성능과 메모리 절감을 동시에 달성하고자 한다.

***

## 2. 제안 방법  
### 2.1 혼합 정밀도 양자화 (Mixed-Precision Quantization)  
- 활성화 채널별 평균값 분포를 분석해 상위 128개 **outlier** 채널만 8-bit(INT8)로, 나머지는 4-bit(INT4)로 양자화  
- 채널 재정렬(reorder) 기법을 이용해 outlier를 행렬 끝으로 모아 메모리 접근의 불규칙성을 제거  
- 가중치도 동일한 순서로 정렬해 정확한 결과 보장  

수식(대칭 양자화):  

$$
\bar X = \mathrm{clamp}\bigl(\lfloor X/s\rceil,\,-2^{n-1},\,2^{n-1}-1\bigr),\quad
s = \frac{2\,\max(|X|)}{2^n-1}\,c
$$  

여기서 $$n$$은 비트 폭, $$c$$는 클리핑 계수이다.

### 2.2 그룹 단위 미세 양자화 (Fine-Grained Group Quantization)  
- 행렬을 그룹(예: 128개 원소)으로 나눠 그룹별 스케일을 계산  
- Tensor Core에서 각 그룹별 INT4 곱셈 수행 후, CUDA Core에서 FP16로 디양자화·합산하도록 GEMM 파이프라인을 커널 융합(fusion)하여 추가 메모리 이동 無  
- 이로써 4-bit 연산의 정확도와 효율을 모두 확보  

### 2.3 동적 양자화 (Dynamic Quantization)  
- 활성화 행렬마다 런타임에 최대값 기반 스케일을 계산해 분포 변동에 적응  
- 양자화 연산을 바로 직전 연산과 융합해 오버헤드를 수 밀리초 단위로 억제  

### 2.4 KV-캐시 양자화 (KV-Cache Quantization)  
- 디코딩 self-attention 단계의 키·값 캐시를 4-bit 또는 비대칭 저비트로 저장  
- Softmax 정규화 특성을 이용해 정밀도 영향 완화  
- 메모리 이동량을 절반 이하로 줄여 throughput 향상  

***

## 3. 모델 구조 및 성능 향상  
Atom은 Llama 계열 모델에 통합되어,  
- **Zero-shot 정확도** 평균 1.4% 이내 손실,  
- **WikiText2 Perplexity** 0.3점 이내 증가  
를 유지하면서,  
- **엔드투엔드 처리량** 최대 7.7×,  
- **디코드 레이턴시** 100ms 미만(배치 256),  
- **동일 메모리 제약** 하 2.5× 처리량 향상  
을 실험적으로 입증했다.

***

## 4. 일반화 성능 향상 가능성  
- **모델 확장성**: Llama-2, Mixtral 등 최신 Transformer 및 Mixture-of-Experts(MoE) 구조에서도 동일한 outlier 정밀도 보존 및 그룹 양자화 기법 적용이 가능  
- **데이터 형식 적응성**: INT4 외에 FP4, MX 포맷 등 차세대 저비트 지원 하드웨어에서도 유사한 성능·정확도 균형 기대  
- **추가 연구 고려점**: MoE의 전문가별 재정렬 인덱스 관리, 클리핑 계수 최적화, 파이프라인 지연 및 메모리 오프로드와의 조합

***

## 5. 한계 및 향후 연구 방향  
- **전처리 비용**: 대형 모델(65B) 기준 4시간 이상 소요되는 오프라인 양자화 및 outlier 식별 과정  
- **그룹 양자화 오버헤드**: 차세대 하드웨어의 MX 연산 지원 전까지는 미세 그룹별 디양자화 비용 존재  
- **모델 특이 분포**: Softmax 이후 분포가 급변하는 비표준 아키텍처(SSM 등)에 대한 안정성 검증 필요  

향후 연구에서는 오프라인 비용 경감, 실시간 적응형 클리핑, 다양한 아키텍처·하드웨어 조합 최적화에 초점을 두어 Atom의 범용성과 효율을 더욱 향상시킬 수 있을 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/329e3395-73a0-401b-90a8-025eeb34e1bd/2310.19102v3.pdf
