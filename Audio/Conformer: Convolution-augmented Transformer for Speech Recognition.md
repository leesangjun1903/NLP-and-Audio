
# Conformer: Convolution-augmented Transformer for Speech Recognition

## 1. 핵심 주장과 주요 기여

Conformer은 Google에서 제안한 음성 인식(ASR) 분야의 획기적인 모델로, **자기주의(self-attention)의 전역 모델링 능력과 합성곱(convolution)의 지역 특성 추출 능력을 결합한 하이브리드 아키텍처**입니다. 이 논문의 핵심 주장은 단순 Transformer나 순수 CNN 중 어느 하나만으로는 음성 신호의 복잡한 특성을 효율적으로 포착할 수 없으며, 두 아키텍처의 상보적 특성을 활용할 때 매개변수 효율성을 유지하면서 최고 성능을 달성할 수 있다는 것입니다.[1]

주요 기여는 다음과 같습니다:

- LibriSpeech 벤치마크에서 언어 모델 없이 **WER 2.1%/4.3% (test/test-other)**, 언어 모델 사용 시 **1.9%/3.9%**의 최고 성능 달성[1]

- **매개변수 효율성**: 10M 파라미터의 소형 모델에서 2.7%/6.3% WER 달성으로 동일 크기 경합 모델 대비 **0.7% 향상**[1]

- **설계 선택 사항에 대한 체계적 분석**: 주의 헤드 수, 합성곱 커널 크기, 활성화 함수, Feed-forward 레이어 배치 등 각 요소의 기여도를 정량화[1]

## 2. 해결 문제 및 제안 방법

### 2.1 해결 대상 문제

기존 접근의 제한사항:

- **Transformer의 한계**: 전역 문맥 포착에 탁월하지만, 미세한 지역 특성 패턴 추출에 제한적[1]
- **CNN의 한계**: 지역 정보를 효율적으로 활용하나, 전역 정보를 포착하려면 많은 층이나 매개변수 필요[1]
- **기존 결합 방식의 비효율성**: Wu et al.의 멀티 브랜치 구조는 입력을 두 개 가지로 나눈 후 결과를 연결하는 방식으로 설계 비효율성[1]

### 2.2 제안 모델 구조

Conformer 블록의 수학적 표현은 다음과 같습니다. 입력 $x_i$에 대해 Conformer 블록 $i$의 출력 $y_i$는:[1]

$$\tilde{x}_i = x_i + \frac{1}{2}\text{FFN}(x_i)$$

$$x'_i = \tilde{x}_i + \text{MHSA}(\tilde{x}_i)$$

$$x''_i = x'_i + \text{Conv}(x'_i)$$

$$y_i = \text{LayerNorm}(x''_i + \frac{1}{2}\text{FFN}(x''_i))$$

여기서:
- **FFN**: Feed-forward 모듈
- **MHSA**: Multi-Head Self-Attention 모듈  
- **Conv**: Convolution 모듈

#### 2.2.1 Multi-Head Self-Attention (MHSA) 모듈

상대 위치 인코딩을 포함한 다중 헤드 자기주의를 사용하며, Transformer-XL의 상대 정현 위치 인코딩 기법을 적용합니다. 이는 다양한 입력 길이에 대한 일반화 성능을 향상시킵니다.[1]

#### 2.2.2 Convolution 모듈

다음 단계로 구성됩니다:[1]

1. **점별 합성곱 (Pointwise Convolution)**: 채널 수를 확장 계수 2로 프로젝션
2. **선형 게이팅 단위 (GLU)**: Gated Linear Unit 활성화
3. **1D 심화 합성곱 (1D Depthwise Convolution)**: 공간적 특성 추출
4. **배치 정규화**: 심화 합성곱 이후 적용
5. **Swish 활성화 및 드롭아웃**: 정규화를 위한 확률적 제거

#### 2.2.3 Feed-Forward 모듈

Macaron-Net의 아이디어를 차용하여 두 개의 반단계 (half-step) 잔차 연결을 가진 FFN을 사용합니다:[1]

- 첫 번째 선형 레이어: 확장 계수 4
- 중간 활성화: Swish
- 두 번째 선형 레이어: 모델 차원으로 복원
- Pre-norm 잔차 유닛 사용

### 2.3 아키텍처 설계의 핵심

Conformer 블록의 샌드위치 구조 (FFN-MHSA-Conv-FFN)는 다음의 목적을 갖습니다:[1]

- **매개변수 효율성**: 각 모듈이 명확한 역할 수행
- **기울기 흐름**: 두 개의 FFN 레이어가 더 깊은 네트워크 훈련 촉진
- **성능 향상**: Macaron-style 구조가 단일 FFN 대비 유의미한 개선 제공

## 3. 성능 향상 및 실험 결과

### 3.1 벤치마크 성능

LibriSpeech 데이터셋에서의 성과:[1]

| 모델 | 파라미터 수 | WER (Without LM) | WER (With LM) |
|------|-----------|------------------|---------------|
| Conformer (S) | 10.3M | 2.7%/6.3% | 2.1%/5.0% |
| Conformer (M) | 30.7M | 2.3%/5.0% | 2.0%/4.3% |
| Conformer (L) | 118.8M | 2.1%/4.3% | 1.9%/3.9% |

### 3.2 절제 연구 (Ablation Studies) 결과

각 구성 요소의 기여도 분석:[1]

- **합성곱 모듈 제거**: dev-other에서 0.4% WER 증가
- **Macaron FFN 제거**: dev-other에서 0.7% WER 증가  
- **상대 위치 인코딩 제거**: dev-other에서 1.4% WER 증가
- **Swish 활성화 -> ReLU**: 최소 영향 (0.1% WER 증가)

### 3.3 주의 헤드 및 커널 크기 최적화

- **최적 주의 헤드**: 16개 헤드에서 최고 성능 달성[1]
- **최적 커널 크기**: 32 크기에서 최고 성능 (dev-clean: 1.83%, dev-other: 4.30%)[1]

## 4. 일반화 성능 향상 메커니즘

### 4.1 구조적 특성

Conformer의 일반화 능력 향상 요인:

1. **상대 위치 인코딩의 역할**: 서로 다른 입력 길이에 대한 적응성 증대로 인코더의 견고성 확보[1]

2. **지역-전역 특성의 조화**: 
   - 합성곱으로 인한 이동 등가성 (translation equivariance) 유지
   - 자기주의를 통한 동적 전역 문맥 포착
   - 결과적으로 다양한 음성 특성에 대한 학습 능력 향상

3. **Macaron-style FFN의 효과**: 더 나은 기울기 흐름으로 깊은 네트워크의 최적화 개선[1]

### 4.2 최근 연구에서 확인된 일반화 성능

#### 4.2.1 노이즈 견고성 (Noise Robustness)

AssemblyAI의 Conformer-1 연구 (2023)에서 **노이즈가 많은 환경에서 43% 적은 오류 달성** (인기 있는 상용/오픈소스 ASR 모델 대비)[2]

- 650K 시간의 대규모 음성 데이터로 훈련
- 수정된 희소 주의 (Sparse Attention) 기법 적용으로 배경 노이즈 영향 감소[2]

#### 4.2.2 일반화 능력 향상 기법

**Conformer-R 모델 (R-Drop 구조 적용)**:[3]
- R-Drop 정규화를 통해 드롭아웃으로 생성된 서브모델의 출력 분포 일관성 강제
- 과적합 감소 및 일반화 능력 개선
- AISHELL1 및Wenetspeech 사전훈련 후 컴퓨터 관련 음성 데이터 미세조정
- LAS, Wenet 등 고전 모델 대비 **1-5% CER (Character Error Rate) 향상**[3]

**Conformer-R 모델의 미세조정 성과**:[3]
- 범용 모델: 12.1% dev CER / 13.5% test CER
- 수직 도메인 미세조정: 11.7% dev CER / **6.3% test CER** (45% 상대 개선)

#### 4.2.3 다중 작업 전이 학습

최근 연구 (Interspeech 2025)에서 ASR 사전훈련 Conformer를 통한 전이 학습 효과:[4]

- **음성 감정 인식 (SER) 작업**: 경합 모델 대비 경쟁력 있는 성능 달성
- **화자 검증 작업**: 사전훈련 모델이 과적합을 줄이고 수렴 속도 가속화
- **노이즈 견고성**: ASR 사전훈련이 노이즈 환경에서의 성능 유지에 유리[4]

#### 4.2.4 도메인 적응 및 장시간 음성

**메모리 증강 Conformer** (Carvalho et al., 2023):[5]
- 신경 튜링 기계 (NTM) 기반 메모리 모듈 통합
- 장시간 음성 처리에서 **최대 58% 상대 WER 감소**[5]

**ChunkFormer** (2025):[6]
- 마스킹된 청킹으로 장시간 음성 처리 효율성 개선
- 산업 규모 배포에서 15분 이상 장시간 음성 처리 가능

### 4.3 일반화 성능 향상을 위한 설계 선택

1. **Pre-norm 잔차 단위**: 네트워크 정규화 개선
2. **드롭아웃 (0.1 비율)**: 각 잔차 단위에 적용하여 과적합 방지
3. **변동 노이즈 추가**: 추가 정규화 메커니즘
4. **L2 정규화** ($\lambda = 10^{-6}$): 가중치 크기 제어

## 5. 모델의 한계

### 5.1 구조적 한계

1. **계산 복잡성**: 자기주의의 이차 복잡도 $$O(n^2)$$로 인한 훈련 및 추론 시간 증가[2]

2. **메모리 효율성**: 주의 메커니즘의 메모리 요구가 장시간 음성 처리에 제약[2]

3. **지연 (Latency)**: 실시간 스트리밍 응용에서 낮은 지연 달성의 어려움

### 5.2 데이터 관련 한계

1. **대규모 데이터 의존성**: 최적 성능 달성을 위해 대량의 레이블된 데이터 필요

2. **도메인 이동 (Domain Shift)**: 훈련 데이터와 상이한 새로운 도메인에서의 성능 저하 가능성

## 6. 후속 연구 동향 및 개선 방안

### 6.1 효율성 개선

**Fast Conformer** (NVIDIA, 2023):[7]
- 점진적 다운샘플링으로 **2.8배 속도 향상**
- 선형 복잡도 주의 메커니즘으로 최대 1시간의 장시간 음성 처리 가능
- 원본 모델 대비 유사 또는 더 나은 정확도 유지[7]

**Efficient Conformer**:[8]
- 29% 추론 시간 단축, 36% 훈련 시간 단축
- WER 정확도 동등 또는 향상[8]

**Squeezeformer** (2022):[9]
- Conformer의 거시 및 미시 아키텍처 재검토
- 상태 최고 ASR 모델 성능 달성

### 6.2 구조적 개선

**Nextformer** (2022):[10]
- ConvNeXt 블록으로 시간-주파수 특성 개선

**DCTX-Conformer** (동적 문맥 수행):[11]
- 스트리밍 및 비스트리밍 모드에서 낮은 지연 달성

**Factorised Speaker-environment Adaptive Training**:[12]
- 베이지안 학습으로 화자와 환경 특성 별도 모델링

### 6.3 정규화 및 일반화 기법

**다양성 및 독립성 정규화** (2025):[13]
- 특성 벡터 간 중복성 감소
- 다양하고 독립적인 특성 벡터 육성
- 배경 노이즈와 분산 외 데이터에 대한 견고성 개선

**공유 가중치 메커니즘**:[14]
- 단일 공유 레이어를 인코더로 사용하는 매개변수 효율 변형
- MyST 데이터셋에서 기존 Conformer와 유사한 성능 유지[14]

### 6.4 다중 모드 및 특수 작업

- **음성 향상 (CMGAN)**: Conformer 기반 메트릭 GAN으로 TF 도메인 향상[15]

- **음성 분리**: LibriCSS에서 BLSTM 기준 대비 **23.5% 상대 WER 감소**[16]

- **강건한 음성 인식**: CHiME-4에서 WRBN 대비 **8.4% 상대 WER 감소**, 모델 크기 18.3% 감소, 훈련 시간 79.6% 단축[16]

- **부호 언어 인식**: 다중 스케일 융합 Conformer로 SI 과제에서 **13.07% WER**, 기존 최고 성능 대비 13.53% 감소[17]

## 7. 향후 연구 고려사항

### 7.1 아키텍처 최적화

1. **선형 복잡도 주의**: 더욱 효율적인 주의 메커니즘 탐색으로 장시간 음성 처리 개선

2. **동적 네트워크**: 입력 특성에 따라 동적으로 모델 구조 조정

3. **양자화 및 압축**: 2-bit 양자화로 모델 크기 32-40% 감소 (WER 최소 저하)[16]

### 7.2 학습 전략

1. **자기 감독 학습 확장**: 600M+ 파라미터의 대규모 자기 감독 Conformer로 범용 음성 표현 학습[18]

2. **멀티태스크 학습**: 여러 관련 작업 동시 학습으로 일반화 능력 향상

3. **도메인 적응**: 사전훈련-미세조정 패러다임의 효율화 (Drop4+HFF+Adapter로 메모리 13% 감소)[19]

### 7.3 응용 확장

1. **비 오디오 도메인**: 분자 구조 예측, 생물학적 서열 모델링 등 새로운 분야 탐색[16]

2. **엣지 디바이스 배포**: 소형 모델 (10M 파라미터) 최적화로 스마트폰, 웨어러블 기기, 스마트 홈 기기 지원[16]

3. **스트리밍 모드**: 낮은 지연 스트리밍 ASR 개선 (6.8배 지연 감소, 모델 크기 50% 감소)[16]

### 7.4 이론적 이해 심화

1. **주의 메커니즘 해석**: Conformer의 주의 헤드가 어떻게 상호작용하는지 해석 가능성 개선

2. **합성곱-주의 상호작용**: 두 모듈 간의 정보 흐름 및 상보성에 대한 이론적 분석

3. **일반화 한계**: 표본 복잡도, 과적합 경계 등 이론적 분석

## 결론

Conformer은 음성 인식 분야에서 **획기적인 아키텍처 혁신**을 제시했으며, 합성곱과 자기주의의 결합을 통해 지역 특성 추출과 전역 문맥 포착의 균형을 달성했습니다. 최근 3-5년간의 후속 연구를 통해 노이즈 견고성, 도메인 적응, 효율성 개선, 다중 모드 처리, 비 오디오 도메인 확장 등 다양한 측면에서 발전이 이루어졌습니다.[1]

향후 연구는 계산 효율성 개선, 장시간 음성 처리, 엣지 기기 배포, 그리고 이론적 이해 심화에 집중될 것으로 예상됩니다. 특히 자기 감독 학습과 멀티태스크 학습을 통한 일반화 능력 향상, 그리고 새로운 도메인으로의 확장이 주요 연구 방향이 될 것입니다.

***

**참고 문헌 및 인용:**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ce91358d-6876-48bd-891b-8aeb054c52e0/2005.08100v1.pdf)
[2](https://assemblyai.com/blog/conformer-1)
[3](https://arxiv.org/pdf/2306.08329.pdf)
[4](https://arxiv.org/html/2307.01546v1)
[5](https://arxiv.org/pdf/2309.13029.pdf)
[6](https://arxiv.org/pdf/2502.14673.pdf)
[7](https://research.nvidia.com/labs/conv-ai/blogs/2023/2023-06-07-fast-conformer/)
[8](https://www.assemblyai.com/research/conformer-1/)
[9](https://arxiv.org/pdf/2206.00888.pdf)
[10](https://arxiv.org/ftp/arxiv/papers/2206/2206.14747.pdf)
[11](http://arxiv.org/pdf/2306.08175.pdf)
[12](http://arxiv.org/pdf/2306.14608.pdf)
[13](https://www.isca-archive.org/interspeech_2025/ko25_interspeech.pdf)
[14](https://www.isca-archive.org/interspeech_2025/rolland25_interspeech.pdf)
[15](https://arxiv.org/pdf/2203.15149.pdf)
[16](https://www.emergentmind.com/topics/conformer-model)
[17](https://openaccess.thecvf.com/content/ICCV2025W/MSLR/papers/Haque_A_Signer-Invariant_Conformer_and_Multi-Scale_Fusion_Transformer_for_Continuous_Sign_ICCVW_2025_paper.pdf)
[18](https://arxiv.org/pdf/2110.04621.pdf)
[19](https://www.isca-archive.org/interspeech_2023/huo23b_interspeech.pdf)
[20](https://policyjournalofms.com/index.php/6/article/view/1252)
[21](https://philnauki.mgimo.ru/jour/article/view/624)
[22](https://aacrjournals.org/cancerres/article/85/8_Supplement_1/3390/757249/Abstract-3390-Impact-of-genomic-alterations-in)
[23](https://www.researchprotocols.org/2025/1/e69163)
[24](https://www.barwmedical.com/index.php/BMJ/article/view/205)
[25](https://www.semanticscholar.org/paper/254ab7a6c96de630a3080e0c170c4433da4e5fd4)
[26](https://www.semanticscholar.org/paper/7aa48e434e3304ea013749c2aa071463a9ccdb7e)
[27](https://www.atlantis-press.com/article/125915689)
[28](https://www.semanticscholar.org/paper/264fa8c77512b1cccfe0cdd9ac26caaa66c6fd8e)
[29](https://www.semanticscholar.org/paper/578dd33b84d7b4ad4297f9ee98ba1cbdc97f57cf)
[30](https://arxiv.org/pdf/2309.03019.pdf)
[31](https://arxiv.org/pdf/2209.00260.pdf)
[32](http://arxiv.org/pdf/2312.10359.pdf)
[33](https://arxiv.org/pdf/2508.10456.pdf)
[34](https://arxiv.org/abs/2306.08329)
[35](https://www.isca-archive.org/interspeech_2025/morais25_interspeech.pdf)
[36](https://www.sciencedirect.com/science/article/abs/pii/S0167639324001201)
[37](https://ieeexplore.ieee.org/iel8/6287639/10820123/10924161.pdf)
[38](https://taapublications.com/tijsrat/article/view/716)
[39](https://aacrjournals.org/cancerres/article/85/8_Supplement_1/7429/759411/Abstract-7429-Illuminating-the-dark-genome-in)
[40](http://mtp.knuba.edu.ua/article/view/288934)
[41](https://www.semanticscholar.org/paper/3d855dc16eb41eb1727163d014bbd1a9efe545ef)
[42](https://www.semanticscholar.org/paper/58e9c712c4ae4f397c624b3c8902a70fbcf55f6e)
[43](https://www.semanticscholar.org/paper/a8e09b6589688f70d0b4455b932db5dcb3cc9c9a)
[44](https://arxiv.org/pdf/2305.05084.pdf)
[45](https://arxiv.org/pdf/2304.09325.pdf)
[46](http://arxiv.org/pdf/2311.17932.pdf)
[47](https://openaccess.thecvf.com/content/WACV2023/papers/Burchi_Audio-Visual_Efficient_Conformer_for_Robust_Speech_Recognition_WACV_2023_paper.pdf)
[48](https://velog.io/@e1kim/Paper-Review-Parameter-Efficient-Transfer-Learning-for-NLP)
