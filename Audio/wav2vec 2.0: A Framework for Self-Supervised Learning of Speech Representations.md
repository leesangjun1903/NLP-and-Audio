# wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

**핵심 주장**  
wav2vec 2.0은 대량의 비라벨 음성 데이터를 이용해 **self-supervised** 방식으로 음성 표현을 학습한 뒤, 소량의 라벨 데이터를 통해 파인튜닝함으로써 최첨단 성능을 달성할 수 있음을 보인다.

**주요 기여**  
1. **엔드투엔드 Self-Supervised 학습 프레임워크**  
   -  음성의 raw waveform에서 컨볼루션 인코더와 Transformer 기반 컨텍스트 네트워크를 이용해 잠재 표현을 추출.  
   -  잠재 표현 중 일부 구간을 마스킹하고, 양자화된(quantized) 표현을 맞추는 contrastive 학습 목표를 도입.  
2. **양자화 모듈**  
   -  Gumbel softmax를 활용해 연속 잠재 벡터를 이산 코드북(codebook) 항목으로 변환.  
   -  코드북 다양성(diversity) 손실을 추가하여 코드 사용 빈도 분포의 엔트로피를 최대화.  
3. **소량 라벨 데이터로도 우수 성능**  
   -  10분 라벨만으로 WER 4.8/8.2%를 달성, 기존 방법 대비 획기적 라벨 효율성 제공.  
4. **범용성 및 확장성**  
   -  1시간, 10시간, 100시간, 960시간 등 다양한 라벨 자원 조건에서 일관되게 성능 향상.  
   -  TIMIT phoneme 인식에서도 종전 대비 23% 상대 PER 개선.

***

# 1. 문제 정의 및 동기

현대 음성 인식은 수천 시간의 전사된 데이터가 필요하나, 전 세계 7,000여 언어 대부분은 라벨 데이터가 부족하다. 인간은 청력을 통해 비라벨 음성을 듣고 언어를 습득하므로, **비라벨 음성으로부터 표현(representation)을 학습**한 뒤 소량의 라벨 데이터로 미세 조정(fine-tune)하는 접근이 효율적이다.

***

# 2. 제안 방법

## 2.1 모델 구조  

Raw waveform X → (1) **Convolutional feature encoder** f: X→Z  
 -  총 7개 컨볼루션 블록, 채널 수 512, 누적 스트라이드 ≈20ms → 시퀀스 z₁…z_T  
(2) **Quantization module** Z→Q  
 -  G 그룹 × V 항목의 코드북, Gumbel softmax로 각 그룹에서 1개 코드 선택 → 이산 qt  
(3) **Context network** g: Z→C  
 -  Transformer 기반, 상대적 위치 인코딩용 컨볼루션 추가  
 -  입력으로는 **마스킹되지 않은** 연속 잠재 z, 출력을 연속 컨텍스트 c_t  

파인튜닝 시에는 c_t → 랜덤 초기화된 선형층 → CTC 손실로 문자 또는 음소 예측.

## 2.2 학습 목표

### 2.2.1 Contrastive loss L_m  
각 마스킹된 time step t에 대해, 참 양자화 벡터 qt를 K개의 distractors {q̃}와 구별하도록 학습:  

$$
L_m = -\sum_{t} \log \frac{\exp(\mathrm{sim}(c_t, q_t)/\kappa)}{\sum_{\tilde q\in Q_t} \exp(\mathrm{sim}(c_t, \tilde q)/\kappa)}
$$

여기서 $$\mathrm{sim}(a,b)=\frac{a^\top b}{\|a\|\|b\|}$$, $$\kappa$$는 온도 파라미터.

### 2.2.2 Diversity loss L_d  
코드북 G×V 항목이 골고루 사용되도록, 그룹별 평균 softmax 확률 $$\bar p_{g,v}$$의 엔트로피 최대화:  

$$
L_d = \frac{1}{G\,V}\sum_{g=1}^G \sum_{v=1}^V \bigl(-\bar p_{g,v}\log\bar p_{g,v}\bigr)
$$

최종 손실:  

$$
L = L_m + \alpha\,L_d
\quad(\alpha=0.1)
$$

***

# 3. 성능 및 한계

## 3.1 라벨 자원별 WER 성능  
-  10분 라벨만으로 **WER 4.8/8.2%** (clean/other) 달성  
-  1시간: 2.9/5.8%, 10시간: 2.4/4.9%, 100시간: 1.9/4.0%  
-  전체 960시간: 1.8/3.3%로 최첨단 기록 경신  

## 3.2 TIMIT PER  
-  사전학습 없이도 **PER 7.4/8.3%**로 종전 대비 23% 상대 PER 개선.

## 3.3 한계 및 고려사항  
- Transformer+CTC 아키텍처로, seq2seq나 워드피스 VOC 사용 시 추가 성능 향상 여지  
- 라벨 데이터가 풍부할수록 LM과 lexicon 결합의 중요성 증가  
- Ultra-low resource 시 자주 발생하는 철자 오류 및 rare 단어 처리 문제  

***

# 4. 일반화 성능 개선 관점

1. **비라벨 음성 도메인 확장성**  
   -  Librivox 53k시간 vs. Librispeech 960시간: 데이터 규모 증가에 비례하는 WER 개선 확인  
2. **양자화와 연속 입력 분리 설계**  
   -  컨텍스트 네트워크 입력은 연속 잠재 유지, 목표(target)만 이산화 → 정보 손실 최소화  
3. **Contrastive objective의 견고성**  
   -  다양한 distractor 샘플링, diversity 제약을 통해 일반화에 유리한 표현 학습  
4. **저자원 언어 및 방언 적용 가능성**  
   -  소수 분량 라벨로도 빠른 적응, 언어·방언 간 표현 전이(transfer)력 우수  

***

# 5. 향후 연구 영향 및 고려사항

**연구 영향**  
- 다양한 언어·도메인에서 **self-supervised 음성 표현 학습** 표준이 될 전망  
- 음성 인식뿐만 아니라 화자 인식, 감정 인식 등 다른 음성 태스크로 확장 가능  

**고려사항**  
- **아키텍처 다양화**: Conformer, seq2seq, 워드피스 통합 등으로 추가 이득  
- **계산 효율성**: 대규모 비라벨 데이터 처리 비용 절감 방안 연구  
- **저자원 언어 적용**: 소수 사용자 데이터에서 사전학습 및 미세조정 전략 최적화  
- **공정성·보안**: 음성 표현 학습 시 방언·소수언어 사용자 과소 대표 문제 검토

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/da0af599-9891-424a-8c47-4e057d911bdb/2006.11477v3.pdf)
