# KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization

**핵심 주장 및 주요 기여 요약**  
KVQuant는 대규모 맥락 길이(최대 10백만 토큰)에 대한 디코더형 LLM 추론 시, 메모리 병목의 주요 원인인 Key-Value(KV) 캐시를 초저비트(2–4비트)로 양자화하면서도 perplexity 저하를 0.1 이하로 억제하는 방법을 제안한다.  
1. **채널별 Key 양자화(Per-Channel Key Quantization)**: RoPE 적용 전 Key 활성화의 아웃라이어 채널 분포를 활용해 채널 축을 따라 양자화함으로써 3-비트에서 perplexity 3.82 개선.  
2. **RoPE 전 양자화(Pre-RoPE Quantization)**: 회전 위치 임베딩(RoPE) 적용 전 Key를 양자화하고, 추론 시 RoPE를 온더플라이로 결합해 3-비트에서 perplexity 0.82 개선.  
3. **민감도 가중 비균일 양자화(Sensitivity-Weighted Non-Uniform Quantization)**: 오프라인 칼리브레이션으로 층별 Fisher 정보 기반 k-means signpost 배치를 학습, 기존 균일·비균일 방식보다 최대 0.33 perplexity 개선.  
4. **벡터별 밀집·희소 양자화(Dense-and-Sparse Quantization)**: 채널/토큰별 1% 아웃라이어만 fp16으로 분리 저장해 3-비트에서 추가 0.19 perplexity 개선.  
5. **주의 싱크 인식(Attention Sink-Aware Quantization)**: 첫 토큰을 fp16으로 유지하여 2-비트에서 0.5 이상 perplexity 급락 방지.  
6. **커널 최적화**: A100/A6000 상에서 4-비트 dense-sparse matvec 1.3–1.7× 속도 향상 구현.  

***

## 1. 해결하고자 하는 문제  
- **메모리 병목**: 맥락 길이 $$l$$과 배치 크기 $$b$$에 비례해 KV 캐시 메모리 $$\propto 2·n·h·d·e·b·l$$가 선형 증가.  
- **기존 양자화 한계**: sub-4비트 KV 캐시 양자화 시 아웃라이어와 부적절한 비트 할당으로 perplexity 급등.

***

## 2. 제안 방법  
### 2.1 Per-Channel Key Quantization  
- Key 행렬의 채널별 평균 크기 차이(아웃라이어 채널) 활용  
- 동일 채널 내 값에 스케일·제로포인트 공유  
- 3-비트에서 perplexity 7.05→5.87 개선  

### 2.2 Pre-RoPE Quantization  
- 회전 위치 임베딩 전 Key $$K_n$$ 양자화, 추론 시 dequantization 후 RoPE $$R_{θ,n}$$ 적용  
- 3-비트에서 perplexity 7.05→6.23 개선  

### 2.3 Sensitivity-Weighted Non-Uniform Quantization  

$$
Q^*(A) \approx \arg\min_Q \sum_{i=1}^N F_{ii} (A_i - Q(A_i))^2
$$  

- Fisher 정보 $$F_{ii}$$로 가중치 적용  
- 오프라인 k-means로 signpost 학습  

### 2.4 Dense-and-Sparse Quantization  
- 벡터별 아웃라이어 1% 분리·fp16 저장  
- 나머지는 [−1,1]으로 정규화 후 재양자화  

### 2.5 Attention Sink-Aware Quantization  
- 첫 토큰만 fp16으로 유지해 2-비트 극단적 성능 저하 방지  

***

## 3. 모델 구조 및 커널 구현  
- **커널**: LUT 기반 4-비트 dense matvec + CSR/CSC 포맷 희소 matvec 병합  
- **메모리 포맷**: Key는 CSC, Value는 CSR  
- **성능**: 4-비트에서 Key 1.2–1.6×, Value 1.3–1.7× 속도 향상  

***

## 4. 성능 향상 및 일반화 가능성  
- **Perplexity 저하 억제**  
  - 4-비트: ΔPPL ≤0.02  
  - 3-비트: ΔPPL ≤0.1  
  - 2-비트: ΔPPL ≤0.5  
- **맥락 길이 확장**  
  - 단일 A100-80GB: 7B 모델 1M 토큰  
  - 8-GPU: 7B 모델 10M 토큰  
- **일반화 성능**  
  - 다양한 모델(LLaMA-1/2/3, Mistral)·데이터셋(Wikitext-2/C4)에서도 일관적 개선  
  - 오프라인 칼리브레이션에 데이터 의존도 낮아 타 도메인 강건  

***

## 5. 한계  
- **학습 단계**: 100K+ 토큰 학습은 미해결  
- **배치 처리**: 다중 토큰 압축 시점 효율성 개선 필요  
- **메모리 할당**: 희소 행렬 동적 갱신 최적화 여유  

***

## 6. 향후 연구에 미치는 영향 및 고려 사항  
- **초장문 처리**: 10M 토큰 급확장으로 실제 대규모 문서 요약·분석 연구 가능  
- **양자화-학습 결합**: 양자화 인식 학습(QAT)과 결합 시 추가 성능 상승 탐색  
- **동적 칼리브레이션**: 실행 중 입력 분포 변화 대응 자동화  
- **하드웨어 협업**: ASIC/TPU용 초저비트 양자화 커널 및 메모리 컨트롤러 설계 고려  

---  
KVQuant는 LLM 추론의 메모리 병목을 획기적으로 완화하며, 초장문 응용과 초저자원 환경 실시간 서비스에 길을 열었다. 향후 초장문 학습 기법, 동적 양자화-학습 통합, 맞춤형 하드웨어 협업이 주요 연구 방향이 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ae9edbe0-4064-4fbb-98ec-36ca61b78b1e/2401.18079v6.pdf
