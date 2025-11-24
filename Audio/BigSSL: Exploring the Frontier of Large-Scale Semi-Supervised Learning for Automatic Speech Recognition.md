# BigSSL: Exploring the Frontier of Large-Scale Semi-Supervised Learning for Automatic Speech Recognition

### 1. 핵심 주장 및 주요 기여

**BigSSL** 논문의 중심 명제는 다음과 같습니다: **대규모 모델 크기의 확장, 사전 훈련, 그리고 자가 훈련의 조합은 레이블링된 데이터의 효율성을 획기적으로 향상시킨다**는 것입니다.[1]

주요 기여는 다음과 같습니다:

- **8B 파라미터 모델 규모까지 확장**: 600M, 1B, 8B 파라미터를 가진 Conformer 모델을 구축하여 대규모 모델의 실용성을 입증했습니다.[1]
- **100만 시간의 대규모 다양한 미레이블 데이터 활용**: YouTube 기반 YT-U(900k시간)를 사전 훈련 데이터로 사용하여 도메인 다양성을 확보했습니다.[1]
- **데이터 효율성의 극대화**: Voice Search 과제에서 34k 시간 레이블 데이터의 **3% 만으로도 기존 최고 성능(SoTA)과 동일한 결과 달성**했습니다.[1]
- **광범위한 다중 도메인 검증**: 13개의 공개 및 비공개 ASR 데이터셋과 비-ASR 과제에서 일관된 성능 향상을 입증했습니다.[1]

***

### 2. 해결 문제 및 기술 방법

#### 2.1 해결하고자 하는 핵심 문제

기존 반자동학습 연구의 제한점:[1]

1. **도메인 적응성 부족**: Libri-Light(60k 시간) 기반 모델이 다른 도메인에 대한 일반화 능력 부족
2. **상용 규모 데이터셋과의 불일치**: 비공개 산업용 음성 데이터셋 규모에 미치지 못함
3. **실제 과제와의 괴리**: 다운스트림 과제가 사전 훈련 과제보다 훨씬 큼

#### 2.2 제안된 방법론

**$$i) \text{Wav2vec 2.0 사전 훈련}$$**

마스킹된 특성 벡터를 학습합니다:

$$L_{\text{pre}} = -\mathbb{E}\left[\log \frac{\exp(z_t^T c_t / \tau)}{\sum_{\tilde{z} \in Z_t} \exp(z_t^T \tilde{z} / \tau)}\right]$$

여기서:
- $$z_t$$: 마스킹되지 않은 인코더 출력
- $$c_t$$: 마스킹된 입력으로부터 생성된 컨텍스트 벡터
- $$Z_t$$: 대비 손실을 위한 네거티브 샘플 집합
- $$\tau$$: 온도 매개변수[1]

**$$ii) \text{Noisy Student Training (NST)}$$**

반자동학습 절차:

1. 교사 모델 $$T$$가 미레이블 데이터셋 $$U$$에 대해 의사-레이블 $$\hat{y} = T(U)$$ 생성
2. 신뢰도 기반 필터링 (선택적):

$$\text{confidence}_i = \frac{1}{|s_i|}\sum_{j=1}^{|s_i|} P(\hat{y}_{ij})$$

여기서 $$|s_i|$$는 길이 정규화 계수, $$P(\hat{y}_{ij})$$는 단어별 신뢰도[1]

3. 학생 모델 $$M$$을 증강된 데이터 상에서 훈련:

$$\mathcal{L}_{\text{NST}} = \mathbb{E}_{(x,y) \sim T(U)}[\ell(M(x), y)]$$

여기서 $$\ell$$는 CTC 또는 RNN-T 손실[1]

#### 2.3 모델 구조: Conformer 아키텍처

**핵심 구조**:

| 모델 | 파라미터 | 레이어 | 차원 | 어텐션 헤드 |
|------|--------|------|------|----------|
| Conformer XL | 0.6B | 24 | 1024 | 8 |
| Conformer XXL | 1.0B | 42 | 1024 | 8 |
| Conformer G | 8.0B | 36 | 3072 | 16 |

Conformer 블록의 구조 (Figure 3 좌측):

$$\text{ConformerBlock} = \text{Attention} + \text{FeedForward} + \text{Convolution} + \text{FeedForward}$$[1]

각 서브모듈의 동작:

- **Attention**: 상대 위치 인코딩을 사용한 다중 헤드 셀프 어텐션
- **Convolution**: 커널 크기 5의 깊이별 분리 가능 합성곱
- **FeedForward**: 두 개의 선형 계층과 Swish 활성화[1]

#### 2.4 학습 상세사항

**SpecAugment 증강**:

$$\tilde{f} = f - \mathcal{U}[0, F], \quad \text{for } F=27 \text{ frequency bands}$$
$$\tilde{t} = t - \Delta t, \quad \text{where } \Delta t \sim \mathcal{U}[0, T \times p_S], p_S = 0.05$$[1]

**학습률 스케줄** (Transformer 스케줄):

$$\alpha(s) = d^{-0.5} \min(s^{-0.5}, s \cdot w^{-1.5})$$

여기서 $$d$$는 모델 차원, $$w$$는 워밍업 스텝[1]

***

### 3. 성능 향상 결과

#### 3.1 Voice Search 성능 (최주요 과제)

| 데이터셋 | 이전 SoTA | 본 논문 | 상대 개선율 |
|---------|----------|--------|----------|
| VS-100h | 4.8% WER | 6.9% → 5.0% (NST) | - |
| VS-1000h | - | 5.0% WER | - |
| VS-34kh (전체) | 4.1% WER | 4.1% WER | **3% 데이터로 SoTA 달성** |[1]

**데이터 효율성 증명**:
- 100시간으로 3% 데이터만 사용하여 전체 34k 시간 학습 수준의 성능 달성[1]

#### 3.2 멀티 도메인 ASR 과제 (SpeechStew)

| 과제 | 기준선 | 우리 모델 | 향상도 |
|-----|--------|---------|--------|
| AMI-IHM | 9.0% | 8.6% (-Downstream NST: 7.8%) | 13.3% ↓ |
| LibriSpeech-clean | 1.4% | 1.9% | - |
| LibriSpeech-other | 2.6% | 3.5% | - |
| TED-LIUM | 4.3% | 4.5% (NST후 4.5%) | - |[1]

#### 3.3 CHiME-6 (원거리 음성 인식)

| 모델 | 개발 | 평가 | SoTA 대비 |
|------|-----|-----|----------|
| Libri-Light 기준선 | - | 38.9% | - |
| Conformer-RNNT-P | 35.1% | 39.5% | - |
| + SpeechStew 훈련 | 26.2% | 34.4% | **11% 상대 개선** |
| PS-모델 (YT-T 자가훈련) | 26.2% | 31.0% | **20% 상대 개선** |[1]

#### 3.4 비-ASR 과제: NOSS 벤치마크

선형 분류기만으로 달성한 SoTA:

| 과제 | SoTA | 본 논문 (Conformer-XL-P) | 개선 |
|-----|------|------------------------|------|
| Voxforge | 95.4% | 99.7% | +4.3% |
| Speech Commands | 97.9% | 97.5% | - |
| CREMA-D | 74.0% | 88.2% | **+14.2%** |
| SAVEE | 70.0% | 92.5% | +22.5% |
| Masked Speech | 73.0% | 73.4% | +0.4% |[1]

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 도메인 간 일반화 메커니즘

**대규모 다양한 데이터의 효과**:

YT-U(100만 시간)는 900개 이상의 YouTube 카테고리를 포괄하여 도메인 다양성 극대화:
- 강의(Lectures)
- 뉴스(News)
- 인터뷰(Interviews)
- 음악, 스포츠 등 다양한 배경음[1]

이러한 다양성이 **크로스 도메인 일반화 능력**을 향상시킵니다.

#### 4.2 모델 크기와 일반화의 관계

**비직관적 발견** (Figure 5):

기존 가정: 더 큰 모델 = 더 큰 데이터셋 필요

본 논문 발견:
- LibriSpeech 960h에서는 사전 훈련이 필수 (큰 모델이 과적합)
- Voice Search 34kh에서는 600M-1B 범위에서 사전 훈련 없이도 더 큰 모델이 더 나은 성능[1]

$$\text{WER}_{\text{G}} < \text{WER}_{\text{XXL}} < \text{WER}_{\text{XL}} \quad \text{(VS-34kh, 사전훈련 없음)}$$

**결론**: 충분히 큰 레이블 데이터셋(≥34k시간)이 있으면, 모델 크기 확장이 항상 이득

#### 4.3 크로스 언어 일반화

비영어 Voice Search 과제에서:

| 언어 | 100h 데이터 | 크로스 언어 vs 네이티브 |
|-----|-----------|----------------------|
| Hungarian (HU) | 9k | 크로스 언어 도움 됨 |
| Chinese (TW) | 20k | 크로스 언어 이득 감소 |
| Hindi (IN) | 27k | 네이티브 우월 |[1]

**패턴**: 다운스트림 데이터셋이 작을수록 크로스 언어 사전 훈련의 이득이 큼

#### 4.4 인터미디어트 표현의 일반화

AudioSet 벤치마크 결과:
- 최종 출력 계층: 0.192 mAP (약함)
- 중간 계층 (Layer 10): 0.304 mAP (**우월**)[1]

$$\text{mAP}_{\text{best-internal}} = 0.304 > \text{mAP}_{\text{output}} = 0.192$$

**해석**: Wav2vec 2.0 사전 훈련 목표가 음성 인식에는 최적화되었지만, 중간 계층이 범용 음성 표현을 학습

***

### 5. 주요 한계점

#### 5.1 P-모델 vs PS-모델 간의 불일치

**SpeechStew에서의 역설적 발견**:

PS-모델(상류 자가훈련)이 P-모델(사전 훈련만)보다 **더 나쁜 성능** 보임[1]

원인:
- 텍스트 정규화의 차이
- 토크나이제이션 불일치 (상류: 4k WPM vs 다운스트림: 1k WPM)
- <unk> 토큰 생성 문제 (TED-LIUM에서 8.7% → 5.0%로 개선)[1]

$$\text{WER}_{\text{PS}} > \text{WER}_{\text{P}} \quad \text{(특정 도메인에서)}$$

#### 5.2 완전 데이터셋에서의 자가훈련 한계

**Voice Search 전체 데이터셋에서의 실패**:

표준 NST 파이프라인(YT-U의 20% 필터링)이 오히려 성능 저하 유발[1]

**가설**: 레이블 데이터셋이 매우 크면 의사-레이블의 노이즈가 실제 이득을 상쇄

$$\mathcal{L}_{\text{combined}} = \alpha \mathcal{L}_{\text{labeled}} + (1-\alpha) \mathcal{L}_{\text{pseudo}}$$

$$\alpha$$가 클수록 중요 (큰 레이블 데이터셋의 경우)

#### 5.3 대규모 모델의 훈련 불안정성

**8B 모델의 과제**:

- Voice Search 전체 데이터셋(34kh)에서 훈련 불수렴
- 사전 훈련이 훈련 안정성의 **필수 조건**[1]
- 하이퍼파라미터 튜닝의 극도로 높은 계산 비용

#### 5.4 모델 압축 부재

거대 모델(8B)의 실용화를 위한 압축 기법 미제시:
- 지식 증류
- 양자화
- 프루닝
등의 전략이 논문에서 미흡

***

### 6. 논문이 앞으로의 연구에 미치는 영향 및 고려사항

#### 6.1 현재의 연구 영향 (2022-2025)

**$$i) \text{기초 모델 패러다임의 확립}$$**

본 논문 이후 음성 인식 분야에서 **대규모 기초 모델 접근**이 표준화됨:[2][3]

- **Whisper** (OpenAI): 100만 시간 약한 감독 데이터로 훈련된 범용 모델
- **WavLM** (Microsoft): 94k 시간으로 SSL 결합 (마스킹 예측 + 노이징 제거)
- **Conformer-1** (AssemblyAI): 570k 시간 데이터로 훈련하여 실제 세계 견고성 입증[4]

**$$ii) \text{데이터 효율성 연구 촉발}$$**

BigSSL의 **3% 데이터로 SoTA 달성** 결과는 다음 연구들을 자극:[5][6]

- 도메인 적응 최소 레이블 요구량 연구
- 저자원 언어 음성 인식
- 특수 음성(어린이, 악센트, 의료) 인식 개선[7][6]

**$$iii) \text{반자동학습 기법의 정제}$$**

최근 발전:

1. **다중 가설 의사-레이블링** (2025): 1-best 대신 다수의 경쟁 가설 사용[6]
   
$$\hat{y}_i = \{\hat{y}_{i1}, \hat{y}_{i2}, \ldots, \hat{y}_{iK}\} \text{ (K-best 가설)}$$

2. **커리큘럼 학습**: 어려운 샘플을 후반에 학습
   
$$L = \sum_{i=1}^{N} w_i(\text{difficulty}_i) \cdot \ell(M(x_i), y_i)$$

3. **신뢰도-기반 필터링 고도화**: 절대 신뢰도가 아닌 상대 신뢰도 사용[6]

#### 6.2 앞으로의 연구 고려사항

**$$i) \text{모델 압축 및 효율성}$$**

**필수 과제**: 8B 모델을 실제 배포 가능한 크기로 축소

제안 방향:
- 지식 증류: 8B 교사 → 100M-500M 학생
- 동적 양자화: 계층별 다른 비트폭 (상위계층 4-bit, 하위계층 2-bit)
- 구조 프루닝: 어텐션 헤드 선택적 제거[8]

**$$ii) \text{도메인 특이적 일반화}$$**

**한계 분석**: 
- 현재 YT-U는 주로 강연/뉴스(어음 명확)
- 산업 소음, 음악, 음성 효과 등에 대한 평가 부족

개선 방향:
- 도메인 가중치 조정 전략
- 대조학습을 통한 도메인 불변 표현
- 메타러닝 기반 적응[5]

**$$iii) \text{PS-모델의 개선}$$**

**현재 문제**: 상류 자가훈련 데이터의 텍스트 정규화 문제

해결책:
- 텍스트 정규화 대조 학습 (Text Normalization Contrastive Learning)
- 토큰화 독립적 표현 학습
- 양방향 텍스트 매핑[1]

**$$iv) \text{멀티모달 학습}$$**

향후 방향:
- 음성 + 시각 정보 결합 (비디오 음성 인식)
- 음성 + 텍스트의 일관성 제약
- 음성 + 음악 신호 분리

**$$v) \text{연합학습(Federated Learning) 통합}$$**

최근 연구 (2024):[9]

대규모 모델의 FL + 차등 개인정보보호(DP)

$$\min_{\theta} \sum_{u \in \text{clients}} p_u \mathcal{L}_u(\theta) + \lambda \|\theta\|_2^2 + \text{DP-noise}$$

도전: 최적화 안정성, 통신 오버헤드, 개인정보보호-성능 트레이드오프[9]

**$$vi) \text{계산 효율성 표준화}$$**

필요성:
- 에너지 소비 기준 (kWh/WER 개선)
- 지연시간 측정 (온디바이스 배포)
- 메모리 풋프린트 추적

최신 효율성 동향:[8]
- Progressive Downsampling: 입력 길이 단계적 감소
- Efficient Conformer: 주의 복잡도 $$O(n^2) \to O(n\log n)$$로 개선
- 2.8배 속도 향상 달성[8]

#### 6.3 미해결 문제

1. **일반화 한계의 이론적 이해 부족**
   - 왜 특정 도메인에서만 성능 저하?
   - 필요한 최소 데이터 다양성 규모는?

2. **의사-레이블 노이즈에 대한 이론**
   - 필터링 임계값 선택의 원칙화
   - 노이즈 포용 한계 분석

3. **멀티태스크 간 전이 학습**
   - ASR → ASV (화자 검증) 전이 성능
   - ASR → 감정 인식 정량화

***

## 결론

**BigSSL**은 음성 인식 분야에서 **대규모, 다양성, 확장성의 시너지**를 입증한 이정표적 연구입니다. 100만 시간 데이터와 8B 파라미터 모델을 조합하여 3% 데이터로도 기존 최고 성능 달성 가능함을 보여줌으로써, 음성 인식의 **데이터 효율성 새로운 기준**을 제시했습니다.

특히 **모델의 일반화 성능** 측면에서:
- 충분한 레이블 데이터 상황에서도 사전 훈련의 이득 지속
- 크로스 도메인, 크로스 언어 설정에서의 일관된 개선
- 비-ASR 과제로의 표현 전이 가능성 입증

이러한 성과는 후속 **Whisper, WavLM, Conformer-1** 등의 발전을 촉발했으며, **멀티모달 학습**, **연합학습**, **계산 효율성 최적화** 등으로 연구 방향을 확대했습니다. 향후 연구에서는 모델 압축, 도메인 특이적 일반화, PS-모델의 개선, 그리고 이론적 토대의 구축이 핵심 과제로 대두됩니다.

***

### 참고문헌

음성 인식 반자동학습 관련 연구[10][11][12][13][5]
Whisper, 대규모 기초 모델 연구[3][2]
Conformer 효율성 및 실제 구현 연구[4][8]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2edd1630-a48f-4d5d-a14f-37b795bca21e/2109.13226v3.pdf)
[2](https://arxiv.org/abs/2110.13900)
[3](https://cdn.openai.com/papers/whisper.pdf)
[4](https://www.assemblyai.com/research/conformer-1/)
[5](https://arxiv.org/pdf/2110.00165.pdf)
[6](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0333915)
[7](http://arxiv.org/pdf/2406.13431.pdf)
[8](https://www.emergentmind.com/topics/conformer-model)
[9](https://openreview.net/pdf?id=ozN92d7CHX)
[10](https://arxiv.org/pdf/2304.14535.pdf)
[11](https://arxiv.org/pdf/2210.08634.pdf)
[12](https://arxiv.org/pdf/2107.05382.pdf)
[13](https://arxiv.org/pdf/2107.13530.pdf)
[14](https://link.springer.com/10.1007/s13534-025-00484-6)
[15](http://arxiv.org/pdf/1911.08460v2.pdf)
[16](http://arxiv.org/pdf/2310.03938.pdf)
[17](https://arxiv.org/abs/2109.13226)
[18](https://www.isca-archive.org/interspeech_2025/park25f_interspeech.pdf)
[19](https://arxiv.org/html/2307.01546v1)
[20](https://www.semanticscholar.org/paper/BigSSL:-Exploring-the-Frontier-of-Large-Scale-for-Zhang-Park/6fe21b01d2202defb8fcd75c40f306a88bd385dc)
