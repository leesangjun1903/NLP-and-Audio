# Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation
# 핵심 요약

**핵심 주장:**  
본 논문은 대규모 음성-텍스트 쌍(633,526쌍, 총 4,325시간)을 활용한 **Contrastive Language-Audio Pretraining**(CLAP) 프레임워크를 제안한다.  
1) 대규모 오디오-텍스트 데이터셋 **LAION-Audio-630K** 공개  
2) 다양한 오디오/텍스트 인코더 실험을 통한 최적 아키텍처 탐색  
3) **Feature Fusion**을 통한 가변 길이 오디오 처리  
4) **Keyword-to-Caption** 자동 캡션 생성 기법을 도입하여 데이터 증강  

이로써, 텍스트-오디오 검색 성능에서 SOTA를 달성하고, zero-shot/슈퍼바이즈드 오디오 분류 성능을 획기적으로 향상시켰다.

# 상세 설명

## 1. 해결 과제  
기존 음성-텍스트 대비 학습 모델들은  
- 소규모 데이터에만 학습하여 일반화 한계  
- 트랜스포머 기반 인코더에서 **가변 길이 오디오** 처리 비효율  
- 레이블 기반 키워드만 이용하여 자연어 표현 부족  
등의 문제를 지닌다.

## 2. 제안 방법

### 2.1 데이터셋 구축: LAION-Audio-630K  
-  8개 공개 소스에서 633,526쌍, 4,325.39시간 오디오-텍스트 확보  
-  AudioSet의 레이블(키워드)을 T5 기반 **Keyword-to-Caption** 모델로 자연어 캡션으로 변환  
-  데이터 전처리: 48kHz FLAC, mono, 최대 토큰 길이 77  

### 2.2 모델 아키텍처

1) **인코더 조합**  
   - 오디오: PANN(CNN) vs. HTSAT(Transformer)  
   - 텍스트: CLIP-Transformer vs. BERT vs. RoBERTa  
2) **Contrastive Loss**  

$$
   L = \frac{1}{2N}\sum_{i=1}^N \bigl[\log\frac{\exp(E^a_i\cdot E^t_i/\tau)}{\sum_{j}\exp(E^a_i\cdot E^t_j/\tau)} + \log\frac{\exp(E^t_i\cdot E^a_i/\tau)}{\sum_{j}\exp(E^t_i\cdot E^a_j/\tau)}\bigr]
   $$  
   
   - $$E^a_i = \mathrm{MLP}\_{\text{audio}}(f_{\text{audio}}(X^a_i))$$  
   - $$E^t_i = \mathrm{MLP}\_{\text{text}}(f_{\text{text}}(X^t_i))$$  
3) **Feature Fusion** (가변 길이 처리)  
   - 입력 길이 $$T\le d$$: 반복 후 패딩  
   - $$T > d$$: 전역 입력(길이 $$d$$) + 3개 로컬 클립(각 $$d$$)  
   - **Attention Feature Fusion**을 통해 합성:  

$$X_{\text{fusion}} = \alpha X_{\text{global}} + (1-\alpha)X_{\text{local}}$$  

4) **Keyword-to-Caption Augmentation**  
   - 템플릿 기반 단순 변환 대신 T5 모델로 더 풍부한 캡션 생성  
   - 생성문 디바이어스(“woman”→“person”) 처리  

## 3. 성능 향상 및 한계

### 3.1 텍스트-오디오 검색  
- 최적 조합: **HTSAT + RoBERTa**  
- LAION-Audio-630K 및 AudioSet(K2C augment) 학습 시 R@1 36.1%→71.8%, mAP@10 대폭 개선  
- Feature Fusion: 길이 가변 클립 처리 성능↑, 특히 Clotho/FreeSound(긴 오디오)에서 두드러짐  

### 3.2 Zero-Shot & Supervised 분류  
- ESC-50, US8K, VGGSound에서 zero-shot 정확도 41.4%→91.0% 등 SOTA 달성  
- FSD50K, VGGSound supervised mAP 64.9%→75.4%로 종전 대비 우수  
- **일반화 성능**: 대규모 대비학습+키워드 증강으로 unseen 클래스 대응력 강화  

### 3.3 한계  
- 데이터 소스 편중: AudioSet 섞이면 특정 도메인 일반화 저하  
- K2C 품질 의존성: 짧은 클립(2초 미만) 캡션 부정확  
- 계산 비용: 대규모 배치 학습 시 높은 메모리 요구  

## 4. 일반화 성능 향상 관점  
- **대규모 다중 소스 학습**으로 다양한 음향 분포 포괄  
- **자연어 캡션 강화**로 텍스트-오디오 매핑 풍부화 → zero-shot에서 강력한 일반화  
- **Feature Fusion**으로 길이 변화에 유연하게 대응, 장단기 패턴 모두 고려  
- 결과: unseen 환경·클래스에서도 일관된 성능 보장

# 향후 연구 영향 및 고려 사항

본 연구는 음성-텍스트 대비 학습 분야에 대규모 데이터셋과 효율적 변형 메커니즘을 제시하여,  
- **오디오 이해**와 **크로스모달 검색** 연구의 기준을 한 단계 끌어올림  
- **Zero-Shot 오디오 분류** 실용화 가능성 확대  

향후 연구 시 고려할 점:  
- **데이터 다양성** 추가 확보(환경, 문화·언어별 캡션)  
- **효율화**: 경량화 모델·메모리 절감 기법 도입  
- **캡션 품질 제어**: 자동 생성 문장 평가·정제 파이프라인 강화  
- **다양한 다운스트림**(오디오 합성·분리·강화) 적용 실험  

이를 통해 더욱 포괄적이고 실용적인 음성-언어 대비 학습 시스템이 발전할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0d3e344f-2834-433e-b449-f4ecfe26a22c/2211.06687v4.pdf)
