# WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing

## 1. 핵심 주장 및 주요 기여
WavLM은 음성 인식(ASR)뿐만 아니라 화자 인식, 화자 분리, 화자 분할, 감정 인식 등 **풀 스택 음성 처리 태스크 전반에 걸쳐** 범용으로 활용 가능한 사전학습 모델을 제안한다.  
주요 기여는 다음과 같다.  
- **Masked Speech Denoising and Prediction**: 노이즈·중첩 음성을 시뮬레이션하고 이를 마스킹하여 원본 음성의 pseudo-label을 예측함으로써 비-ASR 태스크(화자 관련, 분리, 분할)에 필수적인 정보를 학습.  
- **Gated Relative Position Bias**: Transformer 내 상대 위치 임베딩에 게이트를 도입해 음성 구간에 따라 위치 정보의 중요도를 동적으로 조절.  
- **94k시간 규모의 다양한 도메인 프리트레이닝 데이터**: Libri-Light(60k), GigaSpeech(10k), VoxPopuli(24k)로 구성된 Mix-94k 데이터셋을 활용해 아우디오북 편향을 해소하고 실환경 일반화 성능 강화.  
- **SUPERB 및 대표 벤치마크에서 SOTA 달성**: 15개 태스크 SUPERB 전반에서 HuBERT 대비 +2.4 포인트, 화자 검증, 분리, 분할 테스트셋에서 유의미한 성능 향상.

***

# 상세 설명

## 2. 해결하고자 하는 문제
기존 SSL 기반 음성 모델들은 주로 ASR 혹은 특정 태스크에만 초점을 맞춰 설계되어,  
- **화자 분리·분할**과 같은 멀티스피커 환경에 취약  
- **도메인 불일치(오디오북 → 실환경)**로 비-ASR 태스크 성능 저하  
문제가 있다.

## 3. 제안 방법

### 3.1 Masked Speech Denoising and Prediction
입력의 일부 영역을 마스킹하고, 원본 음성의 pseudo-label $$z_t$$를 예측한다.  
노이즈/중첩 시뮬레이션 후 마스킹된 음성 $$\mathbf{u}'$$를 Transformer에 입력하여 다음과 같은 손실을 최소화:

$$
\mathcal{L} \;=\; -\sum_{t \in \mathcal{M}} \log p(z_t \mid h^L_t)
$$

여기서  
- $$\mathcal{M}$$: 마스크된 인덱스 집합  
- $$h^L_t$$: $$L$$번째 층 Transformer의 출력  
- $$p(c\mid h^L_t)=\frac{\exp(\mathrm{sim}(h^L_tW_P,e_c)/\tau)}{\sum_{c'}\exp(\mathrm{sim}(h^L_tW_P,e_{c'})/\tau)}$$

### 3.2 노이즈/중첩 음성 시뮬레이션
주 배치 음성 $$\{\mathbf{u}_{\mathrm{pri}}\}$$에 무작위로 다른 화자 또는 환경 잡음을 섞는 과정을 통해,  
화자 특성 유지 및 분리·분할 상황을 모델에 노출시킴(논문 Algorithm 1 참조).

### 3.3 Gated Relative Position Bias
Transformer Self-Attention에 상대 위치 편향 $$r_{i-j}$$을 도입하고, 콘텐츠 의존적 게이트를 적용:

```math
\begin{aligned}
q_i,k_j &= h_iW_Q,\;h_jW_K,\\
a_{ij}&\propto \exp\Bigl(\frac{q_i\cdot k_j}{\sqrt{d}} + r_{i-j}\Bigr),\\
r_{i-j}&=d_{|i-j|}+g^{\text{update}}_i\,d_{|i-j|} + (1-g^{\text{update}}_i)\,(w^{\text{reset}}_i\,d_{|i-j|}),
\end{aligned}
```

게이트 $$g^{\text{update}}_i, g^{\text{reset}}_i$$는 $$q_i$$에 기반하여 학습되며, 음성과 무음 구간을 구별하여 위치 중요도를 조절.

### 3.4 모델 구조
- **Conv Encoder**: 7-layer temporal convolution (512채널)  
- **Transformer**: Base(12-layer, 94.7M), Large(24-layer, 316.6M)  
- **Relative Position Bias**: 게이트 적용 bucket embedding

## 4. 성능 향상 및 한계

### 4.1 성능 향상
- **SUPERB 전체**: WavLM Large가 HuBERT Large 대비 +2.4pt[표 I 참조].  
- **화자 검증**: VoxCeleb1 EER 0.986% 달성, ECAPA-TDNN 대비 35% 상대 개선.  
- **화자 분할(분리)**: LibriCSS WER 평균 6.0%, Conformer 기준 27.7% 상대 개선.  
- **화자 분할(분할)**: CALLHOME DER 10.35%로 SOTA 경신.

### 4.2 한계
- **프리트레이닝 데이터 영어 편중**: VoxPopuli 영어만 활용. 다국어 적용성 불확실.  
- **계산 자원**: 316M 파라미터, 94k 시간 데이터로 학습 비용·시간 부담.  
- **Fine-tuning freeze 제약**: 일부 태스크에서 모델 개방 미흡 시 추가 성능 잠재력 미흡.

## 5. 모델 일반화 성능 향상 가능성
- **다양한 도메인 데이터**(팟캐스트, YouTube, EP 등) 포함으로 비아우디오북 음성에 강건.  
- **노이즈·중첩 시뮬레이션**으로 멀티스피커·잡음 환경 적응력 향상.  
- **게이트 relative bias**는 음성 구간 특성에 따라 위치 정보 적응적 반영, 일반화 용이.

***

# 향후 연구에의 영향 및 고려 사항
- **모델 확장**: 8B 파라미터급 대형 모델로 범용성·표현력 향상 가능.  
- **경량화·지연 보장**: 실환경 적용을 위해 지연수준·연산량 최소화를 위한 지식증류, 프루닝 연구 필요.  
- **다국어·크로스모달**: 텍스트·음성 공동 학습, 다국어 데이터 확장으로 글로벌 적용성 및 내용 표현력 강화.  
- **장기 시퀀스 처리**: 긴 대화 및 회의 기록 분석을 위한 메모리 효율적 Transformer 구조 도입.  
- **자기지도 태스크 조합**: Masked prediction, contrastive objective 등을 결합한 멀티태스크 학습으로 특성별 보완적 표현 학습.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a2ee37ea-e144-4c9e-ac63-36da1057e583/2110.13900v5.pdf)
