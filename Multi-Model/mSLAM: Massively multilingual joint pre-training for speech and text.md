# mSLAM: Massively multilingual joint pre-training for speech and text

### 1. 핵심 주장과 주요 기여

**mSLAM**(Massively multilingual Speech and LAnguage Model)은 음성과 텍스트 데이터를 **공동으로 사전학습**하여 **공유된 표현 공간**에서 **교차-언어, 교차-모달 표현**을 학습하는 다언어 다중모달 모델입니다. 이 논문의 핵심 기여는 다음과 같습니다:[1][2][3]

- **51개 언어의 음성**(약 429,000시간)과 **101개 언어의 텍스트**(약 15TB)를 활용한 대규모 다언어 사전학습
- **음성과 텍스트 간의 표현 정렬 강화**를 위해 CTC 손실함수를 도입한 혁신적 접근
- **영점 샷(Zero-shot) 교차-모달 이동**을 실현하여, 음성 데이터로만 미세 조정한 모델이 **텍스트 번역을 수행**
- 기존 SLAM 모델의 텍스트 관련 성능 저하 문제 해결
- **다중 모달 미세 조정**(speech translation + text translation)을 통한 성능 향상

### 2. 해결하고자 하는 문제 및 제안 방법

#### 2.1 문제 정의[3][1]

다언어 다중모달 모델 개발에서 나타나는 근본적인 문제들:

- **간섭(Interference)**: 여러 언어와 모달리티를 동시에 처리할 때, 고자원 언어의 성능이 저자원 언어 데이터로 인해 악화되는 현상
- **용량 희석(Capacity Dilution)**: 단일 모델이 다양한 언어와 모달리티를 학습하면서 각 부문의 성능이 감소
- **모달리티 갭(Modality Gap)**: 음성과 텍스트는 다양한 특성을 가지고 있어, 두 모달리티를 효과적으로 정렬하기 어려움
- **교차-모달 전이의 한계**: 기존 SLAM 모델은 음성과 텍스트 간 성능 거래(trade-off) 문제 존재

#### 2.2 제안 방법 및 수식

mSLAM은 다음의 **세 가지 손실함수**를 결합한 복합 목표를 사용합니다:[1]

**1) w2v-BERT 손실(음성의 자감독 학습)**

$$\mathcal{L}_{\text{w2v-BERT}} = \mathcal{L}_{\text{contrastive}} + \mathcal{L}_{\text{MLM}}$$

여기서:
- $\mathcal{L}_{\text{contrastive}}$: 미래의 음성 프레임을 부정 샘플에서 구별하는 대조 학습
- $\mathcal{L}_{\text{MLM}}$: 마스크된 음성 토큰 예측

**2) SpanBERT 손실(텍스트의 자감독 학습)**

$$\mathcal{L}_{\text{SpanBERT}} = \text{span-based masked language model}$$

문자 수준의 토크나이제이션(4,096개 토큰으로 101개 언어 포괄)을 사용하며, 마스킹 스팬 길이를 기존 5에서 **20으로 증가**[1]

**3) CTC 손실(교차-모달 정렬)**

$$\mathcal{L}_{\text{CTC}} = -\sum_{t=1}^{T} \log P(y_t | x_1:T)$$

여기서 $y_t$는 음성의 문자 수준 전사이며, 이는 음성-텍스트 쌍에 대해서만 적용[1]

**4) 번역 언어 모델 손실(TLM)**

$$\mathcal{L}_{\text{TLM}} = \text{masked language modeling on parallel text pairs}$$

**통합 손실함수:**

$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{speech}} + 0.3 \cdot \mathcal{L}_{\text{text}} + 0.03 \cdot \mathcal{L}_{\text{CTC+TLM}}$$

손실 가중치는 과적합 방지를 위해 신중하게 설정됩니다[1]

#### 2.3 데이터 구성[1]

**비라벨 음성 데이터(51개 언어, ~429K시간):**
- VoxPopuli: 23개 언어, 372K시간
- Common Voice: 25개 언어, 6K시간
- Multilingual LibriSpeech: 8개 언어, 50K시간
- BABEL: 17개 언어, 1K시간

**비라벨 텍스트 데이터(101개 언어, ~15TB):**
- mC4 데이터셋에서 저자원 언어에 대해 온도 기반 샘플링 적용 ($T = 3.0$)[1]

**음성-텍스트 쌍 데이터(~2.4K시간):**
- VoxPopuli: 1.3K시간 (14개 언어)
- MLS: 80시간 (8개 언어)
- BABEL: 1K시간 (17개 언어)

### 3. 모델 구조

#### 3.1 아키텍처[1]

mSLAM은 **단일 Conformer 인코더**를 기반으로 하며, 다음과 같이 구성됩니다:

**기본 모델 (600M 파라미터):**
- Conformer 계층: 24개
- 모델 차원: 1024
- 자주의 블록: 8개 계층
- MLM 블록: 16개 계층

**대규모 모델 (2B 파라미터, 실제 1.84B):**
- Conformer 계층: 40개로 증가
- 모델 차원: 1408로 증가
- 자주의 블록: 8개 계층 유지
- MLM 블록: 32개 계층[1]

#### 3.2 음성과 텍스트의 입력 처리

음성: 컨볼루션 네트워크 → Conformer 계층 시퀀스
텍스트: 문자 임베딩 → Conformer 계층 시퀀스

두 모달리티는 **공유된 Conformer 파라미터**를 통과하여 동일한 표현 공간에서 표현됩니다[1]

#### 3.3 최적화[1]

- **옵티마이저**: Adam
- **학습률 스케줄**: Transformer 스케줄 (6.0e-4 → 3.6e-4)
- **워밍업 스텝**: 40,000 스텝
- **학습 스텝**: 600M 모델은 1.3M 스텝, 2B 모델은 350K 스텝
- **배치 구성**: 음성 2,048개, 텍스트 8,192개, 쌍 256개 시퀀스

### 4. 성능 향상 및 평가 결과

#### 4.1 음성 번역 (CoVoST-2)[1]

| 모델 | 고자원(High) | 중간자원(Mid) | 저자원(Low) | 평균 |
|------|-------------|-------------|-----------|------|
| XLS-R 2B (기존) | 36.1 | 27.7 | 15.1 | 22.1 |
| w2v-bert-51 0.6B | 35.6 | 25.3 | 13.4 | 20.4 |
| mSLAM-CTC 0.6B | 35.5 | 25.2 | 13.7 | 20.6 |
| mSLAM-CTC 2B | 36.3 | 27.5 | 15.6 | 22.4 |
| **다중모달 미세조정** | | | | |
| mSLAM-CTC 0.6B | 37.6 | 27.8 | 15.1 | 22.4 |
| **mSLAM-CTC 2B** | **37.8** | **29.6** | **18.5** | **24.8** |

다중모달 미세조정(음성 번역 + 텍스트 번역)으로 **0.3 BLEU** 상승하여 XLS-R 2B 초과[1]

#### 4.2 음성 분류 작업[1]

**MINDS-14 (의도 분류):**
- w2v-bert-51: 82.7%
- mSLAM-CTC 0.6B: **86.9%** (+4.2%)
- mSLAM-CTC 2B: 86.6%

**FLEURS (언어 식별):**
- w2v-bert-51: 71.4%
- mSLAM-CTC 0.6B: 73.3%
- mSLAM-CTC 2B: **77.7%** (+6.3%)

#### 4.3 음성 인식 (ASR)[1]

| 데이터셋 | w2v-bert | mSLAM-TLM | mSLAM-CTC | mSLAM-CTC 2B |
|---------|---------|-----------|-----------|-------------|
| VoxPopuli | 9.3 | 9.4 | **9.2** | 9.1 |
| BABEL | 32.8 | 33.2 | 32.9 | **31.3** |
| MLS-10Hr | 9.9 | 10.4 | 10.1 | **9.7** |

텍스트 추가가 ASR 성능에 부정적 영향을 미치지 않으며, CTC 손실이 경쟁력 있는 성능 유지[1]

#### 4.4 텍스트 분류 (XNLI)[1]

다중모달 모델의 텍스트 작업 성능 저하 관찰:

**영점 샷 평가:**
- mT5-Base 0.6B: 73.0%
- mSLAM-CTC 0.6B: 58.9% (용량 희석)
- mSLAM-CTC 2B: **66.1%** (용량 증가로 개선)

**번역 학습 모두 설정:**
- mT5-Base 0.6B: 79.8%
- mSLAM-CTC 0.6B: 70.0%
- mSLAM-CTC 2B: **76.1%** (격차 축소)

### 5. 일반화 성능과 영점 샷 교차-모달 이동

#### 5.1 핵심 발견: 영점 샷 텍스트 번역[1]

**놀라운 성과**: 음성 번역 데이터로만 미세 조정한 모델이 **텍스트 번역을 수행**

| 언어 | 음성→음성(S→S) | 음성→텍스트(S→T) | 텍스트→음성(T→S) | CTC 영점(CAE) |
|-----|---------------|-----------------|----------------|-------------|
| 러시아어(ru) | 41.7 | **21.9** | 0.0 | 85.9 |
| 프랑스어(fr) | 36.7 | **20.0** | 0.0 | 9.4 |
| 스페인어(es) | 39.1 | **21.2** | 0.0 | 7.9 |
| 스웨덴어(sv) | 33.1 | **15.2** | 0.0 | 13.9 |

**중요한 발견**: 13개 언어 중 7개가 5+ BLEU의 영점 샷 성능 달성. 6개 언어는 사전학습 중 음성-텍스트 쌍 데이터 없이도 성공[1]

**러시아어의 특수성**: 키릴 자모로 입력되었으나 라틴 자모로 부분 음차(transliteration)되어, 라틴 문자 언어들과 동일한 인코딩 공간에 매핑[1]

#### 5.2 교차-모달 표현 정렬 검증 (CTC 프로브)[1]

CTC 프로브를 통한 표현 정렬 분석:

$$\text{프로브 작업: 음성 입력 → CTC 디코더 → 문자 시퀀스 출력}$$

결과: 대부분 유럽 언어의 경우 20% 이하의 CER(Character Error Rate)로 영점 샷 수행 가능, 표현이 실제로 정렬되었음을 증명[1]

#### 5.3 텍스트 사전학습의 필요성[1]

분석 결과 (표 5):
- **텍스트 없이 CTC만 사용**: MINDS-14에서 82.7% → 85.0% (기준선 대비)
- **완전한 mSLAM-CTC**: 82.7% → 86.9%

**결론**: 텍스트 사전학습이 필수적이며, CTC 정렬 손실만으로는 불충분[1]

### 6. 모델의 한계

#### 6.1 용량 희석과 간섭[4][1]

**문제**: 음성 데이터의 추가가 텍스트 작업 성능을 저하시킴
- 영점 샷 XNLI에서 mT5 대비 **-14.1% (0.6B)**, **-6.9% (2B)** 성능 저하
- 특히 비유럽 언어(태국어, 중국어)에서 약 20% 정도 성능 하락[1]

**원인 분석**: 
1. 사전학습 중 쌍 데이터가 주로 유럽 언어에 편중
2. 저자원 비유럽 언어의 음성 데이터 부족
3. 두 모달리티 간의 경합(competition)

#### 6.2 스크립트 민감성[1]

러시아어 제외 대부분의 성공적인 영점 샷 전이는 **라틴 문자 언어**에만 국한
- 예: 터키어는 쌍 데이터가 69시간 있음에도 불구하고 제한된 영점 샷 성능 (1.7 BLEU)

#### 6.3 반복 및 빈 출력[1]

영점 샷 텍스트 번역에서 반복 문제 발생:
- 예: "court-metrages" → "short films either short films either short films"
- 음성만으로 미세 조정한 모델이 텍스트 입력을 예상하지 못하여 발생하는 "oscillatory hallucinations"

#### 6.4 방향성 비대칭성[1]

- **음성→텍스트 (S→T)**: 가능 (평균 9.4 BLEU)
- **텍스트→음성 (T→S)**: 불가능 (0.0 BLEU)

텍스트-중심 미세조정이 음성 능력을 전혀 향상시키지 못함

### 7. 최신 연구 기반 영향과 고려 사항

#### 7.1 후속 연구에 미친 영향[5][6][7][8][4]

**1) 모달리티 정렬 개선 연구:**[9][10][11]
- SPECTRA (2023): 대화 문맥을 활용한 음성-텍스트 정렬 강화
- CTC + 최적 운송 결합 (2023): Wasserstein 거리를 이용한 모달리티 갭 최소화

**2) 다중모달 파운데이션 모델 발전:**[7][12]
- SeamlessM4T v2 (2025): 76개 언어, 스트리밍 번역 지원
- SeamlessExpressive (2025): 음성 운율 보존
- SpeechT5 계열 모델: 통합 인코더-디코더 구조

**3) 모델 분석 연구:**[13]
- 교차-모달 표현 수렴 분석: 초기 계층에서 모달리티 특화, 후기 계층에서 수렴
- 길이 적응(length adaptation)의 중요성: 텍스트-음성 갭 감소에 필수
- 언어 갭보다 모달리티 갭이 더 두드러짐

**4) 다중 작업 학습 최적화:**[8]
- 다단계 다중모달 사전학습: 번역 기반 중간 학습 도입으로 최대 38.45% WER 개선

#### 7.2 앞으로 연구 시 고려할 점[14][15][8]

**1) 용량 희석 및 간섭 완화 전략:**
- 모델 용량 증가: mSLAM-CTC 2B에서 텍스트 성능 개선 확인
- 언어별 라우팅(routing): 고자원과 저자원 언어 간 선택적 학습
- 다중 작업 최적화 기법: Gradient Vaccine, PCGrad 등의 적용

**2) 데이터 불균형 해결:**
- 저자원 언어의 고품질 음성-텍스트 쌍 데이터 수집 필요
- 다언어 병렬 데이터셋 확대
- 언어 계열별 통합학습 (예: 드라비다 언어군 통합)

**3) 모달리티 정렬 개선:**
- 더 정교한 정렬 손실함수 개발 (CTC 외의 대안)
- 도메인별/언어별 정렬 가중치 동적 조정
- 영점 샷 역방향 전이(T→S) 달성 방안 모색

**4) 스크립트 다양성 처리:**
- 다양한 필기 체계(비라틴 문자)를 위한 임베딩 개선
- 음차(transliteration) 기반 접근법 활용
- 유니코드 기반 통합 토크나이제이션 개발

**5) 도메인 외 일반화 능력:**
- 도메인 특화 데이터(의료, 법률, 금융)에서의 성능 평가
- 노이즈 음성에 대한 강건성 강화
- 코드 스위칭(code-switching) 처리 능력 개발

**6) 실무적 배포 고려사항:**
- 모델 압축 및 경량화: 엣지 디바이스 배포 방안
- 스트리밍 추론 최적화: 낮은 지연시간 달성
- 개인정보 보호: 연합 학습 적용 가능성

#### 7.3 최신 트렌드 (2024-2025)[16][17][7]

**1) 언어 커버리지 극단적 확장:**
- Google USM: 100+ 언어 ASR
- OWLS: 150개 언어, 최대 18B 파라미터 모델
- MMS: 1,107개 언어의 음성 인식

**2) 스트리밍 다중모달 번역:**
- SeamlessStreaming: EMMA 메커니즘으로 동시 음성→음성/텍스트 번역 가능
- 낮은 지연시간(low-latency) 달성

**3) 음성 기반 LLM 통합:**
- SpeechVerse: 사전학습 음성/텍스트 모델과 LLM 연결
- 다중 작업 학습 프레임워크로 특화된 미세 조정 필요 없음

**4) 표현 학습의 이론적 이해:**
- 다중모달 표현이 언어 수준에서 어떻게 정렬되는지 분석
- 길이 적응의 중요성 증명: 현재 방법들은 고자원 언어에만 효과적

### 결론

mSLAM은 **51개 언어의 음성과 101개 언어의 텍스트를 공동으로 사전학습**하여 **단일 모델에서 음성-텍스트 표현의 통합**을 처음으로 대규모로 성공시킨 모델입니다. CTC 손실을 통한 **교차-모달 정렬 강화**, **영점 샷 교차-모달 전이 실현**, **다중모달 미세조정의 효과**를 입증했습니다.[3][1]

그러나 **용량 희석 문제, 모달리티 갭, 스크립트 민감성, 방향성 비대칭성** 등의 한계를 보여줍니다. 최신 연구는 이러한 문제들을 완화하기 위해 **개선된 정렬 메커니즘, 극단적 언어 확장, 스트리밍 능력, LLM 통합** 방향으로 진화하고 있습니다. 앞으로의 연구는 **저자원 언어 데이터 확충, 정교한 다중 작업 최적화, 모달리티-언어별 동적 가중치 조정**에 초점을 맞춰야 할 것입니다.[17][7][14][8][1]

***

**참고문헌 ID:**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f829c5b2-f6f2-435e-9e54-94ea8b7133c3/2202.01374v1.pdf)
[2](https://arxiv.org/pdf/2202.01374.pdf)
[3](https://ar5iv.labs.arxiv.org/html/2202.01374)
[4](https://arxiv.org/pdf/2212.09553.pdf)
[5](https://europe.naverlabs.com/blog/on-multimodal-speech-text-pre-trained-models/)
[6](https://aclanthology.org/2024.iwslt-1.29/)
[7](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/)
[8](https://aclanthology.org/2024.lrec-main.1045.pdf)
[9](https://aclanthology.org/2023.acl-long.438.pdf)
[10](https://arxiv.org/abs/2301.11716)
[11](https://www.isca-archive.org/interspeech_2024/liu24d_interspeech.pdf)
[12](https://github.com/microsoft/SpeechT5)
[13](https://aclanthology.org/2025.naacl-short.51.pdf)
[14](https://www.emergentmind.com/topics/multilingual-conversational-speech-language-model-challenge-mlc-slm)
[15](https://arxiv.org/abs/2310.03724)
[16](https://arxiv.org/pdf/2305.13516.pdf)
[17](http://arxiv.org/pdf/2502.10373.pdf)
[18](https://www.semanticscholar.org/paper/375fc5e4bfcaf08cbe96d7ca4321a39f4785add3)
[19](https://aclanthology.org/2025.arabicnlp-sharedtasks.104)
[20](https://arxiv.org/abs/2509.17281)
[21](https://dl.acm.org/doi/10.1145/3581783.3612872)
[22](https://www.frontiersin.org/articles/10.3389/frai.2025.1644093/full)
[23](https://rsisinternational.org/journals/ijriss/articles/enhancing-grade-six-learners-descriptive-paragraph-writing-skills-using-padlet-exploring-its-effectiveness-in-interpreting-visual-media/)
[24](https://aacrjournals.org/clincancerres/article/31/12_Supplement/P3-06-14/753355/Abstract-P3-06-14-Evaluating-Spinning-Science-A)
[25](https://www.johs.org.uk/article/doi/10.54531/UYGY8825)
[26](https://aclanthology.org/2023.acl-long.749.pdf)
[27](https://aclanthology.org/2023.acl-long.315.pdf)
[28](https://arxiv.org/pdf/2405.08295.pdf)
[29](https://arxiv.org/abs/2212.00500)
[30](https://arxiv.org/html/2412.13071v1)
[31](http://arxiv.org/pdf/2307.05222.pdf)
[32](https://aclanthology.org/2022.naacl-main.376/)
[33](https://www.arxiv.org/pdf/2502.02603.pdf)
[34](https://arxiv.org/abs/2205.02444)
[35](https://arxiv.org/abs/2506.11160)
[36](https://www.sciencedirect.com/science/article/abs/pii/S1077314225001146)
[37](https://misinforeview.hks.harvard.edu/?p=12405)
[38](https://pubs.aip.org/jasa/article/156/4_Supplement/A55/3331359/The-AnySpeech-Project-Open-vocabulary-keyword)
[39](https://link.springer.com/10.1007/s11227-025-07808-4)
[40](https://arxiv.org/pdf/2104.14830.pdf)
[41](https://arxiv.org/pdf/2401.03689.pdf)
[42](https://arxiv.org/pdf/2408.03900.pdf)
[43](https://arxiv.org/pdf/2303.01037.pdf)
[44](https://aclanthology.org/2022.emnlp-main.391/)
[45](https://www.emergentmind.com/topics/xlm-r)
[46](https://arxiv.org/abs/2205.12216)
[47](https://jmlr.org/papers/volume25/23-1318/23-1318.pdf)
