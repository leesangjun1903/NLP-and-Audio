# Extending Multilingual BERT to Low-Resource Languages

### 1. 핵심 주장 및 주요 기여

본 논문의 핵심 주장은 **Multilingual BERT (M-BERT)의 한계를 극복하고 저자원 언어(Low-Resource Languages)까지 효율적으로 확장할 수 있는 방법을 제시**하는 것입니다.[1]

주요 기여는 다음과 같습니다:[2][1]

1. **간단하면서도 효과적인 M-BERT 확장 방법론 제시** – 새로운 언어를 M-BERT에 추가하는 방식

2. **이중 성능 향상** – M-BERT에 없던 언어뿐만 아니라 기존에 포함된 언어의 성능도 향상

3. **광범위한 실증 검증** – 27개 언어에 대한 포괄적인 실험으로 방법론의 효과성 입증

구체적인 성능 개선 수치는 **M-BERT에 이미 포함된 언어에서 약 6% F1 향상, 새로운 언어에서 23% F1 향상**을 달성했습니다.[1]

---

### 2. 문제점 정의 및 제안 방법

#### 2.1 해결하고자 하는 문제

M-BERT는 Wikipedia의 상위 104개 언어로만 학습되어 있으며, 이는 전 세계 약 4000개 언어 중 3% 미만입니다. 이로 인해:[1]

- **Somali, Uyghur, Oromo, Hausa, Amharic, Akan** 등 2천만 명 이상이 사용하는 주요 언어들이 제외됨
- 저자원 언어의 자연어처리 작업에서 크로스링궤 전이 학습(cross-lingual transfer learning)이 불가능한 상황 발생

기존 해결책의 한계:[1]

- **Scratch로부터 M-BERT 재학습**: 4개의 클라우드 TPU로 4일 이상의 시간 소요로 매우 비효율적
- **Bilingual BERT (B-BERT) 학습**: 여러 언어의 지도학습 데이터를 활용할 수 없음

#### 2.2 제안 방법: EXTEND 알고리즘

논문에서 제안하는 **EXTEND** 방법은 단 **7시간의 단일 TPU 학습**으로 새로운 언어를 추가합니다.[1]

**수학적 모델링**:

M-BERT의 어휘 크기를 $$V_{mbert}$$, 임베딩 차원을 $$d$$, 새로운 언어의 어휘를 $$V_{new}$$ (고정값 30,000)라 할 때:

$$\text{Extended Encoder/Decoder Matrix} = (V_{mbert} + V_{new}) \times d$$

**EXTEND 절차**:[3][1]

1. **어휘 확장**: 목표 언어 말뭉치에서 M-BERT 어휘에 없는 새로운 단어들을 추출하여 30,000개 크기의 새 어휘 생성

2. **인코더/디코더 확장**: M-BERT의 기존 인코더/디코더 행렬에 새 어휘에 대한 $$(V_{new} \times d)$$ 행렬을 추가

3. **가중치 초기화**: 새로 추가된 부분을 M-BERT의 초기화 절차에 따라 초기화하되, **기존 가중치는 사전학습 값으로 유지**

4. **계속 사전학습(Continual Pre-training)**: 목표 언어의 단어형 언어 모델(MLM)과 다음 문장 예측(NSP) 목적으로 500,000 스텝 학습

**핵심 특징**: Weight-tying으로 인해 디코더는 인코더와 동일하지만 추가 편향(bias)을 갖습니다.[1]

***

### 3. 모델 구조 및 실험 설정

#### 3.1 기본 구조

| 컴포넌트 | 상세 |
|---------|------|
| **기반 모델** | Multilingual BERT (Transformer 인코더) |
| **토큰화** | BERT의 WordPiece 방식 |
| **사전학습 목표** | Masked Language Model (MLM) + Next Sentence Prediction (NSP) |
| **하류 작업** | NER (Named Entity Recognition) - Bi-LSTM-CRF 프레임워크 |
| **도구** | AllenNLP를 기반으로 구현 |

#### 3.2 학습 초매개변수[1]

**EXTEND 학습**:
- 배치 크기: 32
- 학습률: 2e-5 (BERT 제안값)
- 학습 스텝: 500,000
- 학습 시간: 단일 TPU에서 7시간 미만

**B-BERT 비교 기준**:
- 배치 크기: 32
- 학습률: 1e-4
- 학습 스텝: 2,000,000 (20배 더 많음)

***

### 4. 성능 향상 및 실험 결과

#### 4.1 M-BERT 대비 성능 비교

**핵심 결과**:[1]

| 언어 범주 | Zero-shot 성능 향상 | 감독 학습 성능 |
|----------|------------------|-------------|
| **M-BERT 포함 언어 (16개)** | 평균 6% F1 증가 | 대부분 개선 |
| **M-BERT 미포함 언어 (11개)** | 평균 23% F1 증가 | 비약적 향상 |

구체적 사례 (Zero-shot NER):[1]
- **Uyghur**: 3.59% → 42.98% (1090% 상대 향상)
- **Sinhala**: 3.43% → 33.97% (890% 상대 향상)
- **Thai**: 22.46% → 40.99% (82% 상대 향상)
- **Arabic**: 37.56% → 40.83% (9% 향상, 이미 포함 언어)

#### 4.2 B-BERT 대비 비교

**E-MBERT의 우수성**:[1]

| 언어 | E-MBERT | B-BERT | 개선 |
|-----|---------|--------|------|
| Somali | 53.63 | 51.18 | +2.45 |
| Uyghur | 42.98 | 21.94 | +21.04 |
| Sinhala | 33.97 | 16.93 | +17.04 |
| Hausa | 24.37 | 26.45 | -2.08 |

**E-MBERT가 B-BERT보다 나은 이유**:[1]

1. M-BERT의 우수한 다언어 특성 활용 – 100개 이상의 언어 정보 포함
2. 빠른 수렴 속도 – 100k 스텝 내 수렴 (B-BERT는 1M+ 스텝 필요)
3. 관련 언어로부터의 지식 전이 – 예: Tamil이 Sinhala 학습에 도움

#### 4.3 추가 실험 분석

**추가 데이터 없이도 성능 유지**:[1]
LORELEI 데이터 대신 M-BERT 학습에 사용된 Wikipedia 데이터로 EXTEND를 수행해도 성능 저하 없음:
- Russian: 56.64 (Wikipedia 사용)
- Thai: 38.35 (Wikipedia 사용)

이는 **어휘 확장과 목표 언어 최적화 자체가 핵심** 임을 시사합니다.

---

### 5. 일반화 성능 향상 가능성

#### 5.1 언어별 성능 향상 분석

**M-BERT 포함 언어의 향상 메커니즘**:[1]

1. **목표 언어 어휘 확장**
   - Wikipedia 기반 M-BERT는 언어별 불균형한 어휘 집합 보유
   - 저자원 언어: 매우 제한된 어휘
   - EXTEND: 30,000개의 새 어휘로 표현 능력 대폭 향상

2. **목표 언어 최적화**
   - 계속 사전학습(500k 스텝)이 해당 언어에 특화된 표현 학습
   - 더 나은 다언어 초기화를 통한 빠른 수렴

3. **추가 모노링궤 데이터**
   - LORELEI 말뭉치의 추가 데이터가 추가 신호 제공

#### 5.2 비영 언어 성능 문제

**중요한 한계 발견**: E-MBERT 확장 시 다른 언어의 성능 저하[4][1]

목표 언어 최적화가 필연적으로 다른 언어의 성능을 희생:
- 하나의 언어에 최적화된 모델이 다른 언어에서는 성능 하락
- 이는 **모델의 용량 희박화(capacity dilution)** 문제를 야기

#### 5.3 수렴 속도 개선 (일반화의 지표)

**EXTEND의 빠른 수렴**:[1]

```
E-MBERT (Hindi):     100k 스텝에서 수렴
E-MBERT (Sinhala):   100k 스텝에서 수렴
B-BERT (Hindi):      1,000k+ 스텝 필요
B-BERT (Sinhala):    1,000k+ 스텝 필요
```

빠른 수렴은 **사전학습된 다언어 표현의 강력한 초기화**를 의미하며, 이는 **더 나은 최초 일반화 성능**을 시사합니다.

---

### 6. 한계 및 제약사항

#### 6.1 방법론적 한계[1]

1. **일 언어 당 하나의 모델**: EXTEND는 한 번에 하나의 언어만 추가 가능
   - 여러 언어를 동시에 확장하려면 각각 별도 모델 필요
   - 컴퓨팅 자원 비효율

2. **비영 언어 성능 저하**: 목표 언어 확장으로 다른 언어의 성능 감소
   - 희박화된 다언어 용량

3. **단순한 토큰 초기화**: 새 토큰을 무작위로 초기화
   - MUSE나 VecMap 같은 정렬 모델을 통한 더 나은 초기화 가능성 미탐색

4. **고정된 어휘 크기**: $$V_{new} = 30,000$$로 고정
   - 언어별 최적 어휘 크기 분석 부재

#### 6.2 영어 성능 유지[1]

긍정적 측면: 대부분의 경우 E-MBERT는 영어 NER 성능을 M-BERT 수준으로 유지
- 평균 영어 성능: 79.37 (M-BERT 기준)
- E-MBERT: 대부분 78+ 범위

#### 6.3 데이터셋 제한

- 단일 작업(NER)에만 평가
- 27개 언어 (다양성 보장하나 모든 언어족 커버 하지 않음)
- LORELEI 데이터셋에만 의존

***

### 7. 최신 연구 기반: 앞으로의 영향과 고려사항

#### 7.1 논문의 학계 영향 (2020 이후)

**높은 인용도**: 165+ 인용 기록으로 저자원 언어 NLP의 중요 연구로 확립[5]

**영향받은 후속 연구 방향들**:

1. **토큰 초기화 최적화** (2023-2025)
   - **FOCUS, OFA, Tik-to-Tok** 등의 동적 초기화 방법 개발
   - 기존의 무작위 초기화 대신 **의미론적으로 관련된 기존 토큰의 가중 조합** 사용
   - 수렴 속도 및 성능 추가 개선[6]

2. **어휘 확장 전략 발전** (2024-2025)
   - **XLM-V**: 공유 어휘를 100만 토큰으로 확장하고 언어 간 불필요한 토큰 공유 제거
   - 저자원 언어 성능 **최대 18% F1 향상**[7]

3. **계속 사전학습(Continual Pretraining)** 표준화
   - **EMMA-500, MaLA-500**: 대규모 언어모델에 대한 계속 사전학습으로 저자원 언어 능력 확장[8]
   - 워크플로우 표준 확립

4. **크로스링궤 정렬(Cross-lingual Alignment)** 강화
   - 2025년 연구: 모델의 중간층에서 **언어 간 의미론적 정렬** 향상
   - 모듈식 접근으로 task-specific과 alignment 모듈 분리 가능[9]

#### 7.2 현재 멀티링궤 모델의 진화

**EuroBERT (2025)**:[10]
- 유럽 및 주요 언어 대상 새로운 인코더 계열
- 기존 대안 모델 초월
- 8,192 토큰까지 시퀀스 처리 가능

**XLM-R 생태계 확장**:[11][7]
- XLM-R Large (550M 파라미터)
- XLM-RoBERTa-Longformer: 4,096 토큰 길이 처리

#### 7.3 미래 연구 시 고려할 점

**1. 용량 희박화(Capacity Dilution) 해결**[12][7]
- 목표 언어 성능과 다른 언어 성능의 균형 문제
- **해결책**: 언어별 모듈식 구조 도입 (예: MEGAVERSE 구조)
- 매개변수 효율적 파인튜닝(LoRA, Adapter) 적용 고려

**2. 다중 언어 동시 확장**[1]
- EXTEND의 주요 미해결 과제
- **방향**: 언어 계층 구조적 학습, 순차적 적응

**3. 토큰 초기화 최적화**[13][6]
- 의미론적 정렬을 고려한 초기화
- 외부 다언어 어휘 자원(VecMap, MUSE) 활용
- 저자원 언어와 고자원 관련 언어 간 매핑

**4. 스크립트 및 언어족 특이성**[14]
- 2024년 연구: 언어족과 스크립트가 저자원 언어 성능의 **중요 결정 요인**
- 언어족별 특화 모델 개발 고려

**5. 일반화 능력 향상**[14]
- 학습되지 않은 언어(unseen languages)에서의 성능이 **스크립트 타입과 언어족에 의존**
- 스크립트 정보를 명시적으로 모델에 통합 필요

**6. 매개변수 효율** (2024-2025 트렌드)
- 4비트 양자화로 소비자 하드웨어에 배포 가능하도록 최적화[15]
- 저자원 지역사회에서의 실제 활용성 향상

**7. 도메인 적응 + 다언어 적응 결합**[9]
- 언어 정렬 + 작업 특화 모듈의 조합
- 사후 조합(post-hoc combination)으로 재학습 없이 성능 향상

**8. 구조적 지식 통합**[7]
- 개체 관계 지식 주입 (XLM-K 형태)
- 사회언어학적 맥락 고려

---

### 결론

**"Extending Multilingual BERT to Low-Resource Languages"**는 저자원 언어를 위한 효율적이고 실용적인 해결책을 제시함으로써 **멀티링궤 NLP의 민주화**를 추진한 중요한 연구입니다. EXTEND 방법은 단순하면서도 효과적이며, 특히 **제한된 계산 자원 환경에서 새로운 언어를 빠르게 추가**할 수 있다는 점이 획기적입니다.

이후 5년간의 발전을 보면, 본 논문의 핵심 아이디어인 **어휘 확장과 계속 사전학습**은 표준 패러다임으로 정착했으며, 토큰 초기화, 용량 관리, 언어 간 정렬 등의 세부 기법들이 지속적으로 고도화되고 있습니다. 앞으로는 **다중 언어 동시 확장, 모듈식 구조, 의미론적 정렬 강화**, 그리고 **스크립트/언어족 특이성 고려**가 핵심 연구 방향이 될 것으로 전향이 될 것으로 전망됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d80361f-fc56-499b-b2a0-578f24b350b2/2004.13640v1.pdf)
[2](https://www.aclweb.org/anthology/2020.findings-emnlp.240.pdf)
[3](https://cogcomp.seas.upenn.edu/papers/WKMR20.pdf)
[4](https://arxiv.org/pdf/2306.01093.pdf)
[5](https://aclanthology.org/2020.findings-emnlp.240.pdf)
[6](https://www.emergentmind.com/topics/token-initialization-method)
[7](https://www.emergentmind.com/topics/xlm-r)
[8](https://arxiv.org/pdf/2409.17892.pdf)
[9](https://slator.com/new-research-explores-how-to-boost-large-language-models-multilingual-performance/)
[10](http://arxiv.org/pdf/2503.05500.pdf)
[11](https://huggingface.co/markussagen/xlm-roberta-longformer-base-4096)
[12](http://arxiv.org/pdf/2412.12500.pdf)
[13](https://arxiv.org/html/2402.14408v1)
[14](https://arxiv.org/html/2404.19159v1)
[15](https://aclanthology.org/2025.wmt-1.100.pdf)
[16](https://arxiv.org/pdf/1909.00100.pdf)
[17](https://www.aclweb.org/anthology/D19-1374.pdf)
[18](http://arxiv.org/pdf/2409.10965.pdf)
[19](https://www.aclweb.org/anthology/2020.sustainlp-1.16.pdf)
[20](http://arxiv.org/pdf/2402.14408.pdf)
[21](https://www.semanticscholar.org/paper/Extending-Multilingual-BERT-to-Low-Resource-Wang-Karthikeyan/dbfbfcc2633ef46c53e2343525ee87c700f2cfc3)
[22](https://ai.meta.com/research/publications/cross-lingual-transfer-learning-for-multilingual-task-oriented-dialog/)
[23](https://arxiv.org/pdf/2509.11570.pdf)
[24](https://openreview.net/forum?id=k7-s5HSSPE5)
[25](https://arxiv.org/abs/2502.02722)
[26](https://dataloop.ai/library/model/peltarion_xlm-roberta-longformer-base-4096/)
[27](https://www.sciencedirect.com/science/article/abs/pii/S095070512501113X)
[28](https://arxiv.org/pdf/2405.18359.pdf)
[29](https://arxiv.org/pdf/2503.10497.pdf)
[30](https://aclanthology.org/2023.emnlp-main.258.pdf)
[31](https://arxiv.org/pdf/2311.07463.pdf)
[32](http://arxiv.org/pdf/2411.10083.pdf)
[33](https://arxiv.org/pdf/2305.17740.pdf)
[34](https://arxiv.org/html/2408.14118)
[35](https://www.nature.com/articles/s41598-024-66472-5)
[36](https://angelica.gitbook.io/hacktricks/todo/llm-training-data-preparation/3.-token-embeddings)
[37](https://aclanthology.org/2024.vardial-1.2.pdf)
