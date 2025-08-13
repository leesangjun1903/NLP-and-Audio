# PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition

## 1. 핵심 주장 및 주요 기여  
PANNs(Pretrained Audio Neural Networks)는 방대한 AudioSet(1.9M 클립, 527개 클래스)을 활용해 오디오 패턴 인식을 위한 대규모 사전학습 모델을 제안한다.  
- **핵심 주장**: 대규모 데이터로 사전학습된 CNN 계열 네트워크가 다양한 오디오 분류·태깅 과제에서 전통적 소규모 학습 모델을 뛰어넘는 일반화 성능을 보인다.  
- **주요 기여**  
  1. AudioSet 전체 데이터를 사용해 CNN6/10/14, ResNet22/38/54, MobileNetV1/V2, 1D CNN(Res1dNet) 등 다양한 아키텍처로 사전학습  
  2. Wavegram–Logmel-CNN: 파형 기반 Wavegram과 로그 멜 스펙트로그램을 결합해 AudioSet 태깅 mAP 0.439 달성(종전 최고 0.392 대비 +4.7%)  
  3. 사전학습된 PANNs를 ESC-50, DCASE, GTZAN, RAVDESS 등 6개 소규모 태스크에 전이–미세조정(fine-tune) 시 모두 최첨단 성능 경신  
  4. 데이터 불균형 샘플링, Mixup·SpecAugment 활용, 멜 빈·샘플링율·임베딩 차원 등 하이퍼파라미터 최적화  

***

## 2. 문제 정의, 제안 방법 및 모델 구조  

### 2.1 해결하고자 하는 문제  
- 소규모 데이터만으로 학습된 오디오 태깅 모델은 클래스 수·데이터 스케일이 제한적  
- ImageNet·BERT처럼 대규모 사전학습이 오디오 패턴 인식에 미흡

### 2.2 제안 방법  
1. **사전학습**  
   - AudioSet raw waveform → STFT → 64 멜 필터 → log-mel 스펙트로그램  
   - CNN14 기반으로 600k iter 학습, balanced 샘플링 및 Mixup 적용  
2. **Wavegram-Logmel-CNN**  
   - 파형에 1D CNN(필터 k=11, s=5 + dilation 블록×3 + maxpool) 적용 → Wavegram(T×F×C/F)  
   - Wavegram과 log-mel 채널 결합 → 2D CNN14  
3. **전이학습 전략**  
   - Scratch: 랜덤 초기화  
   - Freeze+classifier: 임베딩 추출기 동결, FC 1~3층만 학습  
   - Fine-tune: FC 최종 레이어만 새로 초기화, 전체 네트워크 미세조정  

### 2.3 핵심 수식  
- 이진 크로스엔트로피 손실  

$$
\ell = -\sum_{n=1}^{N}\bigl[y_n\log f(x_n) + (1-y_n)\log\bigl(1-f(x_n)\bigr)\bigr]
$$

### 2.4 모델 구조 개요  
- **CNN14**: VGG 계열 14층, BatchNorm+ReLU, 2×2 pooling ×4, global pooling → FC(2048→527)  
- **ResNet38**: Basic block×16, downsampling后 global pooling → FC  
- **MobileNetV2**: Inverted residual block t=6, lightweight 설계  
- **Wavegram-Logmel-CNN**: 1D CNN → Wavegram; 2D CNN14로 시공간 패턴 학습  

***

## 3. 성능 향상 및 한계  

### 3.1 AudioSet 태깅 성능  
- **CNN14**: mAP 0.431, AUC 0.973  
- **ResNet38**: mAP 0.434  
- **Wavegram-Logmel-CNN**: mAP 0.439(종전비교 +0.047)  
- **MobileNetV1/V2**: mAP 0.389/0.383, 연산량 8.6%/6.7% 수준  

### 3.2 전이학습 성능  
| 태스크      | Scratch  | Freeze_L1 | Freeze_L3 | Fine-tune | 최고 기존 기여  |
|-----------|---------|-----------|-----------|-----------|----------------|
| ESC-50    | 0.833   | 0.918     | 0.908     | **0.947** | 0.865      |
| DCASE19   | 0.691   | 0.764     | 0.589     | **0.764** | 0.851*     |
| DCASE18   | 0.902   | 0.717     | 0.768     | **0.941** | 0.954*     |
| MSoS      | 0.760   | 0.960     | 0.886     | **0.960** | 0.930      |
| GTZAN     | 0.758   | 0.915     | 0.827     | **0.915** | 0.939*     |
| RAVDESS   | 0.692   | 0.397     | 0.401     | **0.721** | 0.645      |

\* 앙상블·스테레오 기반

### 3.3 한계  
- 클래스 불균형 완전 해결 어려움: 희귀 클래스 AP 저조  
- 학습 비용 및 메모리 부담(42G 연산, 80M 파라미터)  
- 일부 도메인(DCASE19·RAVDESS)에서 freeze 방식 저성능  

***

## 4. 일반화 성능 향상 가능성  

- **대규모·다양성 효과**: AudioSet 내 527개 클래스 범용 특징 학습 → 소수 샷(task)의 강건성  
- **Wavegram 활용**: 파형–스팩트럼 융합으로 시간·주파수 특성 동시 포착  
- **Contrastive/self-supervised 확장**: 레이블 노이즈 억제, 레이블 없는 대규모 데이터 활용 가능  
- **경량화·지속학습**: Knowledge distillation, quantization, adapter-based 미세조정으로 모바일·연속학습 지원  

***

## 5. 향후 연구 영향 및 고려 사항  

- **영향**:  
  - 오디오 분야 ImageNet-BERT 역할, 다양한 downstream task 표준 사전학습 모델로 자리매김  
  - Wavegram 개념 확장: 자가학습 패치 입력, 이종 모달리티 융합  

- **연구 시 고려점**:  
  1. **데이터 품질**: 레이블 노이즈·불균형 대비 방안(노이즈-탄력학습, 샘플링 전략)  
  2. **모델 경량화**: 모바일·임베디드 적용 위한 연산·메모리 최적화  
  3. **도메인 적응**: 프롬프트·어댑터 활용해 특수 도메인(task) 소량 데이터 적응  
  4. **자기지도 학습**: 레이블 없는 대규모 오디오에서 표현 학습 후 fine-tune  

PANNs는 대규모 사전학습이 오디오 패턴 인식 전반에 미치는 혁신적 토대를 마련했으며, 경량화·도메인 적응·자가학습 연구로 확장될 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c57207ef-0024-4011-be5a-306892dc9272/1912.10211v5.pdf
