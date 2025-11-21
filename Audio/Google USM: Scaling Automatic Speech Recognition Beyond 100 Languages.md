# Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages

### 1. 핵심 주장 및 주요 기여

**Google USM(Universal Speech Model)**은 100개 이상의 언어에서 자동 음성 인식을 수행할 수 있는 단일 대규모 모델을 제시합니다. 이 논문의 핵심 주장은 대규모 비레이블 다국어 음성 데이터(12M 시간, 300개 이상 언어)와 더 작은 규모의 레이블이 있는 데이터셀의 조합을 통해 효과적인 범용 ASR 모델을 구축할 수 있다는 것입니다.[1]

주요 기여는 다음과 같습니다:

- Whisper 모델 대비 7분의 1 수준의 레이블된 학습 데이터(90k시간)로 유사하거나 더 우수한 성능 달성[1]
- **BEST-RQ** 기반 음성 전처리와 **MOST**(Multi-Objective Supervised pre-Training)를 통한 음성-텍스트 모달리티 정렬[1]
- **청크 방식 어텐션(Chunk-wise Attention)**을 통한 장형 음성 인식 문제 해결[1]
- 102개 언어의 FLEURS 벤치마크에서 최신 기술 달성[1]
- 저자원 언어에 대한 제로샷 및 소규모 샷 적응 능력 입증[1]

***

### 2. 해결하고자 하는 문제와 제안하는 방법

#### 2.1 주요 문제점

ASR을 수백 개 언어로 확장하는 데 있어 근본적인 난제는 다음과 같습니다:

1. **데이터 부족**: 대부분의 언어에서 전사된 음성 데이터는 매우 부족하지만, 비전사 음성과 텍스트 데이터는 무한에 가까움[1]
2. **적응성 문제**: 저자원 언어에 대한 모델 일반화 성능 저하[1]
3. **장형 음성 처리**: 훈련 중 30초 이하의 짧은 음성 세그먼트로 훈련된 모델이 실제 환경의 수 시간 길이의 음성 처리 시 성능 저하[1]

#### 2.2 제안하는 방법론

**3단계 학습 파이프라인**으로 구성됩니다:

##### (1) **BEST-RQ 기반 비지도 사전학습**

BEST-RQ(BERT-based Speech pre-Training with Random-projection Quantizer)는 BERT 스타일의 마스킹 작업을 음성에 적용하여 이산 라벨을 예측합니다:[2]

```math
\text{discrete\_label} = \arg\min_c \|\text{project}(x) - v_c\|
```

여기서 $x$는 음성 특성, $v_c$는 무작위로 초기화된 코드북 벡터입니다. 다중 소프트맥스 손실을 통해 학습 안정성을 개선합니다:[2]

$$\mathcal{L}_{BEST-RQ} = \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}_{\text{softmax}_i}(\text{predicted}, \text{target}_i)$$

여기서 $N$은 16개의 독립 소프트맥스 계층입니다.[1]

**장점**:
- 코드북 붕괴(codebook collapse) 문제 없음[1]
- wav2vec 2.0 대비 2배 이상의 훈련 시간 단축[3]
- 2B 파라미터 모델로의 확장성 우수[1]

##### (2) **MOST: 다중 목적 지도 사전학습**

MOST는 비레이블 음성, 레이블된 음성-텍스트 쌍, 비레이블 텍스트를 동시에 활용합니다:[1]

$$\mathcal{L}_{MOST} = \lambda_1 \mathcal{L}_{BEST-RQ} + \lambda_2 \mathcal{L}_{ASR} + \lambda_3 \mathcal{L}_{\text{consistency}} + \lambda_4 \mathcal{L}_{\text{reconstruction}}$$

각 손실 항의 역할:
- $\mathcal{L}_{BEST-RQ}$: 음성 인코더 사전학습 유지
- $\mathcal{L}_{ASR}$: 음성-텍스트 정렬 학습
- $\mathcal{L}_{\text{consistency}}$: 음성 인코더와 텍스트 인코더 간의 표현 공간 정렬
- $\mathcal{L}_{\text{reconstruction}}$: 비레이블 텍스트를 마스킹된 음성 특성으로 재구성[1]

##### (3) **청크 방식 어텐션**

장형 음성 인식 성능 저하 문제를 해결하기 위해 제안합니다:

**문제의 본질**: 로컬 셀프 어텐션(128 프레임 좌우 컨텍스트)을 깊은 신경망에 스택하면, 수용 영역이 선형으로 증가합니다:[1]

```math
\text{receptive\_field} = 2 \times 128 \times \text{num\_layers} = 256 \times 32 = 8192 \text{ frames} > 300 \text{ seconds}
```

훈련-테스트 불일치 발생: 훈련 중 최대 30초, 추론 중 수 시간.[1]

**해결책**: 어텐션을 음성 청크 내로 제한합니다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)_{\text{chunk}} V$$

8초 청크로 설정하면 수용 영역이 고정되어 장형 음성에 대한 일관된 성능 유지.[1]

***

### 3. 모델 구조

#### 3.1 기본 아키텍처: Conformer

**Conformer**(Convolution-augmented Transformer)는 트랜스포머의 글로벌 패턴 포착 능력과 컨볼루션의 로컬 패턴 포착 능력을 결합합니다:[1]

**Conformer 블록 구조**:
```
Input → FeedForward → MultiHeadAttention → Conv1D → FeedForward → Output
```

USM에서 사용된 두 가지 모델 크기:[1]
- **Conformer-0.6B**: 24 계층, 1024 차원, 8 어텐션 헤드
- **Conformer-2B**: 32 계층, 1536 차원, 16 어텐션 헤드

#### 3.2 인코더-디코더 구성

- **인코더**: Conformer (BEST-RQ 또는 MOST로 사전학습)
- **디코더**: 세 가지 옵션 제공[1]
  1. **CTC**(Connectionist Temporal Classification): 비자동회귀식, 장형 음성에 최적
  2. **LAS**(Listen, Attend, and Spell): 자동회귀식, 짧은 음성에 최적
  3. **RNN-T**(RNN Transducer): 저자원 언어 적응에 최적

#### 3.3 데이터셋 구성

USM은 다음 네 가지 유형의 데이터를 활용합니다:[1]

| 데이터셋 | 크기 | 설명 | 언어 수 |
|---------|------|------|--------|
| YT-NTL-U | 12.1M 시간 | 비레이블 YouTube 음성 | 300+ |
| Web-NTL | 28B 문장 | 비레이블 웹 텍스트 | 1140 |
| YT-SUP+ | 90k 시간 + 100k 시간 의사레이블 | YouTube 레이블 음성 + 시끄러운 학생 훈련 | 73-75 |
| Pub-U/Pub-S | 429k + 10k 시간 | 공개 비레이블/레이블 음성 | 51-102 |

***

### 4. 성능 향상

#### 4.1 벤치마크 성능[1]

**YouTube 장형 음성 인식** (18개 언어):
- USM-LAS: 14.4% WER (Whisper-longform: 17.7%)
- USM-CTC: 13.7% WER (Whisper-shortform과 비교 불가능 - hallucination)
- **상대 개선율**: 약 23-30%

**CORAAL (AAVE 인식)**:
- USM-LAS: 19.0% WER (Whisper: 27.8%)
- **상대 개선율**: 32%

**FLEURS 다국어 ASR** (102개 언어):
- USM-LAS: 10.5% WER (Whisper 62개 언어: 36.6%)
- USM-M: 12.5% WER → 11.8% WER (MOST 적용)
- **상대 개선율**: Maestro 대비 30%

**CoVoST 2 음성 번역** (21개 언어):
- USM-M: 30.7 BLEU (이전 최고: 29.6 BLEU)
- 약 859시간 데이터로 최고 성능 달성 (이전: 125k시간)

#### 4.2 다중 소프트맥스의 효과[1]

| 소프트맥스 개수 | FLEURS (CER) | CoVoST (BLEU) | 상대 개선율 |
|---------------|-------------|-------------|---------|
| 1 | 7.4 | 27.5 | baseline |
| 16 | 6.9 | 28.7 | >5% |

#### 4.3 모델 및 언어 스케일링[1]

- YT-55 (55개 언어) 대비 YT-NTL-U (300+ 언어) 사용: ~10% 상대 개선율
- 각 새로운 언어가 평균 500시간의 데이터만 포함되어도 성능 향상

***

### 5. 모델의 일반화 성능 향상

#### 5.1 저자원 언어에 대한 강력한 적응성

**보지 않은(Unseen) 언어 성능** (노이지 스튜던트 훈련 적용):[1]

| 언어 | Whisper 성능 | Adapter 적응 후 | 의사 레이블 후 | 상대 개선율 |
|------|------------|------------|-----------|---------|
| Hausa | 88.9% | 24.5% | 22.8% | 7.5% |
| Shona | 121.0% | 29.1% | 22.2% | 31.1% |
| Yoruba | 94.8% | 33.4% | 30.6% | 9.2% |

**핵심 통찰**: 10시간의 역내 도메인 데이터로도 FLEURS 벤치마크에서 유의미한 성능 달성 가능[1]

#### 5.2 Adapter 기반 매개변수 효율적 미세조정[1]

**Frozen Encoder + Adapter**:
- 추가 매개변수: 2% (약 40M)
- 성능 손실: 최소 (1-2% 상대 WER 증가)
- 장점:
  - 100+ 언어의 Adapter를 동시에 로드 가능
  - 동일한 사전학습 가중치 재사용으로 저장소 절감
  - 개별 언어별 빠른 적응 가능[1]

#### 5.3 도메인 간 일반화[1]

- SpeechStew (다중 도메인 en-US): 26.7% WER (거의 모든 도메인에서 SOTA)
- CORAAL (아프리카 미국 영어 방언): 18.7% WER
- YouTube 장형 실제 환경: 13.7% WER

**일관성**: 역내 도메인뿐 아니라 역외 도메인에서도 강건한 성능[1]

#### 5.4 음성-텍스트 모달리티 정렬의 중요성

MOST 적용 효과:[1]
```
Base USM (음성만) → MOST 적용 후
FLEURS: 11.8% CER → 10.5% CER (-11% 상대 개선)
CoVoST: 28.7 BLEU → 30.7 BLEU (+7% 상대 개선)
```

비레이블 텍스트의 기여:
- 28B 문장의 웹-NTL 데이터로 텍스트 인코더 학습
- 음성 표현과의 일관성 손실로 정렬 개선
- 보지 않은 언어에도 일반화 성능 향상[1]

***

### 6. 주요 한계

#### 6.1 구조적 한계

1. **계산 복잡도**: 2B Conformer 모델의 추론 RTF (Real-Time Factor):[1]
   - TPUv4i, 배치 32: RTF = 1.2 (30배 시간 소요 - 실시간 불가)
   - 엣지 디바이스 배포 어려움

2. **마스킹 작업의 제약**: 
   - BEST-RQ의 무작위 프로젝션 코드북은 학습 불가
   - 특정 언어에 최적화된 양자화 불가능

#### 6.2 데이터 기반 한계

1. **YouTube 데이터 편향**:
   - 주로 엔터테인먼트, 교육 콘텐츠 (특정 도메인 과다 대표)[1]
   - 음성 다양성 제한 가능성

2. **저자원 언어 데이터 불균형**:
   - YT-513-U에서 88개 언어는 500시간 이상, 188개 언어는 100시간 미만[1]
   - 극도의 저자원 언어는 성능 개선 미미 가능

#### 6.3 성능 한계

1. **LAS 디코더의 장형 음성 문제**: 
   - Chunk-wise attention 적용에도 CTC보다 성능 열세[1]

2. **특정 언어 군 성능**:
   - 동일 언어족 내에서도 성능 편차 큼
   - 저자원 언어의 의사 레이블 품질 의존도 높음

3. **도메인 이동 문제**:
   - YouTube 중심 훈련 데이터로 인한 타 도메인(전화 음성, 과학 강의 등) 성능 저하 가능[1]

***

### 7. 최신 연구 동향 및 앞으로의 영향

#### 7.1 USM 이후 주요 진전 사항

**1) 언어 식별 통합** (2024-2025)[4]
- Samsung 연구: 다국어 ASR과 언어 식별을 통합 프레임워크로 결합
- **성과**: 언어 식별 오류율 6% 감소, ASR 성능 19.1% WER 감소

**2) 음성-텍스트 다중모달 모델 발전**[5]
- **SPIRIT-LM**: 텍스트와 음성 토큰 인터리빙으로 더 나은 정렬 달성
- **SSR-connector**: 음성-텍스트 모달리티 임베딩 간 간격 감소

**3) 효율적 미세조정 방법론**[6]
- **언어별 LoRA Adapter**: 기존 Adapter 대비 0.4% 매개변수로 12.2% 평균 WER 감소
- 39개 저자원 언어에서 최대 37.5% 개선

**4) 음성 복호 기반 TTS 개선**[7]
- 저자원 언어 TTS에 다국어 ASR 모델 활용
- 30분 데이터로도 8배 성능 개선 가능

#### 7.2 일반화 성능 향상의 새로운 방향

**1) 언어족 기반 전이 학습**[8]
- Turkic 언어 그룹에 대한 다국어 훈련: CER 67.7% → 19.3% (극도의 저자원 상황)
- 언어 간 음운 유사성을 활용한 지능형 그룹화

**2) 매개변수 효율적 튜닝의 일반화**[9]
- **Whistle 모델**: 약음 기반 감독으로 다국어 및 교차언어 성능 강화
- 자음소 단위 학습으로 미보유 언어도 28% 개선

**3) 계층적 표현 학습**[10]
- SSHR(Self-Supervised Hierarchical Representations): 최종 계층에서 내용 정보 강화
- Cross-CTC로 더 나은 표현 분리

#### 7.3 앞으로의 연구 고려사항

**1) 계산 효율성**
- 엣지 디바이스 배포를 위한 모델 압축 필수
- 추론 RTF 1 이하 달성이 실무적 요구사항[1]

**2) 도메인 이동 문제 해결**
- 특정 도메인(의료, 법률, 산업 음성)에 대한 지속적 적응 메커니즘 필요
- 다양한 음성 특성(강조, 방언, 배경음) 포함 훈련 데이터 확충

**3) 저자원 언어의 품질 대 비용 트레이드오프**
- 의사 레이블 품질 보증 시스템 개발
- 인간 피드백 반복 학습(RLHF) 적용으로 품질 개선

**4) 다중 모달 통합의 심화**
- 텍스트-음성 정렬 기반 하위 단어 발견(BPE vs. 음소)
- 시각 정보(입술 움직임)와의 삼중 모달리티 통합 연구

**5) 언어 간 전이의 이론화**
- 언어족, 음운 체계, 형태론의 구체적 영향 분석
- 구조화된 표현 학습으로 더 효과적인 교차언어 전이 달성

**6) 평가 지표의 재고**
- WER/CER의 한계 극복: 의미 보존 정확도(MCA), 불균형 언어 특화 지표 개발
- 실제 사용자 중심 평가 체계 수립

***

### 결론

**Google USM**은 대규모 비지도 사전학습, 다중 목적 지도 사전학습, 그리고 매개변수 효율적 미세조정의 세 가지 핵심 요소를 통해 **100개 이상의 언어로 확장 가능한 음성 인식 모델**을 최초로 실현했습니다. 특히 **BEST-RQ의 단순성**과 **MOST의 음성-텍스트 모달리티 정렬**은 저자원 언어의 일반화 성능을 획기적으로 향상시켰습니다.[1]

그러나 **계산 복잡도**, **도메인 편향**, 그리고 **극도의 저자원 환경**에서의 성능 제약은 여전히 해결해야 할 과제입니다. 향후 연구는 **효율성-정확도 균형**, **도메인 적응 메커니즘**, 그리고 **다중 모달 통합의 심화**에 집중할 필요가 있습니다.[9][10][4][5][6]

USM의 접근 방식은 ASR을 넘어 기계 번역, 음성 합성, 다국어 이해 모델에도 영감을 주고 있으며, **저자원 언어 기술의 민주화**라는 중요한 목표 달성을 위한 토대를 마련했습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e6af4284-9e95-4540-8b1e-db2b72c5c093/2303.01037v3.pdf)
[2](https://proceedings.mlr.press/v162/chiu22a/chiu22a.pdf)
[3](https://arxiv.org/abs/2405.04296)
[4](https://research.samsung.com/blog/A-Unified-Approach-to-Multilingual-Automatic-Speech-Recognition-with-Improved-Language-Identification-for-Indic-Languages)
[5](https://aclanthology.org/2025.acl-long.682.pdf)
[6](https://arxiv.org/abs/2401.08992)
[7](https://arxiv.org/html/2509.21718v1)
[8](https://www.nature.com/articles/s41598-024-64848-1)
[9](https://arxiv.org/abs/2406.02166)
[10](http://arxiv.org/pdf/2309.16937.pdf)
[11](https://arxiv.org/pdf/2108.02034.pdf)
[12](https://arxiv.org/pdf/2412.15299.pdf)
[13](https://arxiv.org/pdf/2401.03689.pdf)
[14](https://arxiv.org/pdf/2303.01037.pdf)
[15](https://arxiv.org/pdf/2205.08014.pdf)
[16](https://arxiv.org/pdf/2301.07851.pdf)
[17](https://www.merl.com/research/highlights/seamless-asr)
[18](https://europe.naverlabs.com/blog/on-multimodal-speech-text-pre-trained-models/)
[19](https://aclanthology.org/2024.eacl-tutorials.5/)
[20](https://www.ijcai.org/proceedings/2023/0761.pdf)
[21](https://ijcaonline.org/archives/volume186/number64/senapati-2025-ijca-924462.pdf)
[22](https://arxiv.org/html/2409.01217v1)
[23](https://www.isca-archive.org/interspeech_2025/li25p_interspeech.pdf)
[24](https://arxiv.org/pdf/2106.09236.pdf)
[25](https://arxiv.org/pdf/2310.14954.pdf)
[26](https://arxiv.org/pdf/2111.00127.pdf)
[27](https://arxiv.org/pdf/2203.00725.pdf)
[28](https://arxiv.org/pdf/2403.08258.pdf)
[29](http://arxiv.org/pdf/2204.03889.pdf)
[30](https://arxiv.org/pdf/2104.06865.pdf)
[31](http://arxiv.org/pdf/2011.04196v1.pdf)
[32](https://www.assemblyai.com/research/conformer-1/)
[33](https://openreview.net/forum?id=tZDhrhUOcs)
[34](https://arxiv.org/abs/2211.07201)
[35](https://arxiv.org/html/2312.09100v1)
[36](https://www.isca-archive.org/interspeech_2022/audhkhasi22_interspeech.pdf)
[37](https://icml.cc/media/icml-2022/Slides/17406.pdf)
[38](https://www.isca-archive.org/interspeech_2023/blau23_interspeech.pdf)
[39](https://betweencloud.tistory.com/114)
[40](https://aclanthology.org/2022.emnlp-main.630.pdf)
[41](https://aclanthology.org/2023.emnlp-main.431.pdf)
[42](https://aclanthology.org/2021.emnlp-main.126.pdf)
[43](https://arxiv.org/pdf/2109.11680.pdf)
[44](http://arxiv.org/pdf/2205.12647.pdf)
[45](https://aclanthology.org/2023.findings-acl.517.pdf)
[46](http://arxiv.org/abs/1912.01214)
[47](https://arxiv.org/pdf/2111.06799.pdf)
[48](https://www.sciencedirect.com/science/article/abs/pii/S030645732200351X)
[49](https://aclanthology.org/2021.acl-short.103/)
[50](https://www.isca-archive.org/interspeech_2022/xu22b_interspeech.pdf)
[51](https://aclanthology.org/2025.fieldmatters-1.3.pdf)
[52](https://arxiv.org/abs/2409.13910)
[53](https://proceedings.mlr.press/v202/yu23l/yu23l.pdf)
[54](https://huggingface.co/blog/mms_adapters)
[55](https://aclanthology.org/2025.naacl-short.4.pdf)
