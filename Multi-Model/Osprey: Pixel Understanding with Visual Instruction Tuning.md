
# Osprey: Pixel Understanding with Visual Instruction Tuning

> **논문 정보**
> - **제목**: Osprey: Pixel Understanding with Visual Instruction Tuning
> - **저자**: Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, Jianke Zhu
> - **발표**: CVPR 2024 (arXiv: 2312.10032)
> - **코드/데이터**: [https://github.com/CircleRadon/Osprey](https://github.com/CircleRadon/Osprey)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

Multimodal Large Language Models (MLLMs)는 Visual Instruction Tuning을 통해 인상적인 범용 비전-언어 능력을 달성했지만, 기존 MLLMs는 주로 이미지 수준 또는 바운딩 박스 수준의 이해에 집중하여 픽셀 수준의 세밀한 비전-언어 정렬을 달성하는 데 한계가 있었다.

또한 마스크 기반 지시 데이터의 부족이 모델 발전을 제한했으며, 이에 Osprey는 마스크-텍스트 지시 튜닝(mask-text instruction tuning) 방식을 제안하여, 세밀한 마스크 영역을 언어 지시에 통합함으로써 픽셀 수준의 시각적 이해를 달성하고자 한다.

### 1.2 주요 기여 (3가지)

| 기여 항목 | 내용 |
|---|---|
| **데이터셋 구축** | 724K 규모의 마스크 기반 region-text 데이터셋 (Osprey-724K) |
| **모델 설계** | Convolutional CLIP + Mask-Aware Visual Extractor + LLM 통합 |
| **SAM 연동** | SAM과 원활하게 통합하여 다중 세분도 시맨틱 획득 가능 |

목표 달성을 위해, 먼저 724K 샘플로 구성된 마스크 기반 region-text 데이터셋을 정밀하게 구축하고, Convolutional CLIP 백본을 비전 인코더로 채택하며, 고해상도 입력으로부터 정확한 시각적 마스크 특징을 추출하기 위한 mask-aware visual extractor를 설계했다.

---

## 2. 해결하고자 하는 문제

### 2.1 기존 MLLMs의 한계

대부분의 MLLMs에서 사용되는 ViT 기반 CLIP 모델은 224×224~336×336의 낮은 입력 해상도를 채택하고 있어 픽셀 수준 표현, 특히 소규모 영역의 세밀한 이미지 이해가 어렵다. 입력 해상도를 높이는 것은 ViT 아키텍처의 전역 어텐션과 관련된 계산 부담으로 인해 제한된다.

바운딩 박스와 비교했을 때, 세밀한 마스크를 참조 입력으로 사용하는 것이 객체를 보다 정확하게 표현할 수 있다.

### 2.2 제안하는 방법 및 수식

#### (1) Vision Encoder: Convolutional CLIP

높은 해상도 입력을 처리하기 위해 Convolutional CLIP 백본을 비전 인코더로 활용하며, ViT 기반 모델과 비교하여 Convolutional CLIP은 효율성과 강건성을 갖추고 더 큰 입력 해상도로의 일반화가 우수하다.

#### (2) Mask-Aware Visual Extractor

Osprey는 mask-aware visual extractor를 통해 각 객체 영역 내의 픽셀 수준 특징을 캡처하며, 마스크 수준의 시각 특징을 인코딩하고 각 영역의 공간 위치 정보를 수집한다. 이를 위해 비전 인코더가 생성한 다중 레벨 이미지 특징에 대해 마스크 풀링(mask-pooling) 연산을 수행하고, 각 단일 레벨 특징에 대해 마스크 영역 내의 모든 특징을 풀링한다. 이후 각 특징을 선형 프로젝션 레이어를 통해 인코딩하여 region-level 임베딩을 생성하고, 다중 레벨 특징을 합산(summation)을 통해 융합한다.

이를 수식으로 표현하면 다음과 같다:

**마스크 풀링 연산 (Mask-Pooling)**:

$$f_l^{mask} = \frac{\sum_{(i,j) \in \mathcal{M}} F_l(i, j)}{|\mathcal{M}|}$$

여기서:
- $F_l(i,j)$ : $l$번째 레이어의 비전 인코더 특징 맵
- $\mathcal{M}$ : 입력 마스크 영역
- $f_l^{mask}$ : $l$번째 레이어의 마스크 풀링 특징

**다중 레벨 특징 융합 (Multi-Level Fusion)**:

$$f^{region} = \sum_{l=1}^{L} W_l \cdot f_l^{mask}$$

여기서:
- $W_l$ : 각 레벨의 선형 프로젝션 가중치
- $L$ : 특징 레이어 수
- $f^{region}$ : 최종 region-level 임베딩

**LLM 입력 시퀀스 구성**:

$$\mathbf{X}_{input} = \text{Interleave}(\{f^{region}_k\}_{k=1}^{K},\ \mathbf{E}_{text})$$

여기서:
- $\{f^{region}_k\}$ : $K$개의 마스크 영역 임베딩
- $\mathbf{E}_{text}$ : 텍스트 언어 임베딩
- $\text{Interleave}(\cdot)$ : 마스크 특징과 언어 임베딩을 인터리빙(interleaving)하는 연산

**학습 목적함수 (Language Modeling Loss)**:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t \mid \mathbf{X}_{input},\ y_{<t};\, \Theta)$$

여기서:
- $y_t$ : $t$번째 토큰 (응답)
- $\Theta$ : 모델 파라미터 전체

---

## 3. 모델 구조

Osprey는 이미지 수준의 비전 인코더(image-level vision encoder), 픽셀 수준의 mask-aware visual extractor, 그리고 대규모 언어 모델(LLM)로 구성된다. 입력 이미지, 참조 마스크 영역, 입력 언어가 주어지면 토크나이제이션 및 변환을 수행하여 임베딩을 획득하고, 마스크 특징과 언어 임베딩 시퀀스를 인터리빙하여 LLM에 전달함으로써 세밀한 의미론적 이해를 달성한다.

```
┌─────────────────────────────────────────────────────────┐
│                      Osprey 모델 구조                    │
├───────────────┬──────────────────┬──────────────────────┤
│ Vision Encoder│ Mask-Aware       │  LLM (Vicuna-7B)     │
│ (Conv-CLIP)   │ Visual Extractor │                      │
│               │                  │  ← Interleaved       │
│  Multi-level  │  Mask Pooling    │    [mask feat | text] │
│  Feature Maps │  + Linear Proj   │                      │
│               │  + Summation     │  → Fine-grained       │
│  High-Res     │                  │    Semantic Output   │
│  Input        │ Region Embedding │                      │
└───────────────┴──────────────────┴──────────────────────┘
```

Osprey-724K는 마스크-텍스트 쌍으로 구성된 지시 데이터셋으로, 세밀한 픽셀 수준 이미지 이해를 장려하기 위해 약 724K의 GPT 생성 멀티모달 대화를 포함하며, 강건성과 유연성을 위해 object-level, part-level 및 추가적인 지시 샘플을 포함한다.

**데이터셋 구축 파이프라인**:

세밀한 지역 기반 지시 데이터를 생성하는 데이터 처리 파이프라인을 구축했으며, 객체 카테고리, 객체 유형, 객체 행동, 위치, 색상, 상태 등을 포함한다. LLaVA-115K의 상세 설명을 COCO 이미지의 이미지 수준 설명으로 활용하고, 언어 전용 GPT-4를 통해 각 객체 영역의 시각적 내용을 다양하게 생성하는 지시-따르기 데이터를 구축했다. 바운딩 박스와 간략한 영역 캡션을 최대한 활용하며, 각 박스는 장면 내 객체 개념과 공간 위치를 인코딩한다.

**견고성 향상을 위한 Positive/Negative 샘플**:

MLLMs는 객체 환각(object hallucination) 문제로 고통받으며, 시각 지시에서 자주 등장하거나 다른 객체와 함께 등장하는 객체가 잘못 환각될 수 있다. 정확한 영역 이해를 위한 MLLM의 강건성 강화를 위해 양성/음성 지시 샘플을 추가로 구성하여, 특정 영역이 특정 카테고리에 해당하는지 여부를 묻는 쿼리와 "Yes/No" 응답을 기대하는 방식으로 설계한다.

---

## 4. 성능 향상

Osprey-724K 데이터셋으로 학습된 Osprey 모델은 다양한 영역 이해 태스크에서 우수한 성능을 입증하였으며, 최신 기술 방법들을 능가했다.

구체적으로 Osprey 모델은 METEOR 점수 16.6%, CIDEr 점수 108.3%를 달성하여 최신 GLaMM 접근법을 각각 0.4%, 3.3% 능가하는 경쟁력 있는 성능을 보였다.

**주요 평가 태스크**:

Osprey는 open-vocabulary segmentation, referring object classification, detailed region description, region-level captioning의 4가지 대표적 태스크에 대한 평가를 수행했다.

---

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 Convolutional CLIP의 해상도 일반화

높은 해상도 입력 사용을 촉진하기 위해 Convolutional CLIP 백본을 비전 인코더로 활용하였으며, ViT 기반 모델과 비교하여 Convolutional CLIP은 효율성과 강건성을 바탕으로 더 큰 입력 해상도로 잘 일반화된다.

### 5.2 SAM과의 결합으로 제로샷 일반화

특히 Osprey는 Segment Anything Model(SAM)과 원활하게 통합되어 다중 세분도(multi-granularity) 시맨틱을 획득할 수 있다.

Osprey는 point-prompt, box-prompt, segmentation everything 모드에서 SAM과 원활하게 통합하여 특정 파트 또는 객체와 관련된 시맨틱을 생성한다.

### 5.3 부분 수준(Part-level) 이해를 통한 세밀한 일반화

위의 설계들을 통해 Osprey는 파트 수준 및 객체 수준 영역에 대한 세밀한 의미론적 이해를 달성하며, 주요 객체 카테고리, 상세한 객체 속성, 그리고 더 복잡한 장면 설명을 제공할 수 있다.

### 5.4 도전적 시나리오에서의 견고성

Osprey는 이러한 도전적 시나리오에서 강건한 능력으로 정확한 의미론적 예측을 생성할 수 있다.

### 5.5 한계

Osprey는 픽셀 수준 시각적 이해에 탁월하지만, 사전 첨부된 세그멘테이션 모델(pre-attached segmentation models)에 의존하기 때문에 적용 범위가 제한된다.

또한, 픽셀 수준 이해 데이터와의 결합 학습은 이미지 수준 능력 저하로 이어지는 현상이 LISA와 GLaMM 등 다수 모델에서 관찰되는 공통적 한계이다.

---

## 6. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 입력 참조 방식 | 픽셀/마스크 출력 | 데이터 규모 | 특징 |
|---|---|---|---|---|
| **GPT4RoI** (2023) | Bounding Box | ✗ | Region-text | ROI 특징 + LLM |
| **Shikra** (2023) | 좌표 텍스트 | ✗ | - | 텍스트 좌표 기반 |
| **Ferret** (2023) | Free-shape | 부분 | - | 임의 형상 참조 |
| **LISA** (2023) | 텍스트 | ✓ (SAM 기반) | - | 추론 기반 세그멘테이션 |
| **GLaMM** (CVPR2024) | 이미지+영역 | ✓ | 810M 영역(GranD) | 대화 생성 + 픽셀 그라운딩 |
| **Osprey** (CVPR2024) | **마스크** | ✗(입력 마스크) | 724K | 픽셀 수준 의미 이해 |
| **OMG-LLaVA** (NeurIPS2024) | 통합 | ✓ | 다중 | 이미지·객체·픽셀 통합 |
| **Pixel-SAIL** (2025) | 통합 | ✓ | 1.7M | 단일 트랜스포머, 전문가 모델 불필요 |

GLaMM은 자연어 응답을 객체 세그멘테이션 마스크와 원활하게 통합하여 생성할 수 있는 최초의 모델로, 텍스트 및 선택적 시각 프롬프트를 모두 수용하여 멀티모달 사용자 상호작용을 강화한다.

최신 연구인 Pixel-SAIL-3B는 METEOR 점수 17.6을 달성하여 세밀하게 설계된 더 큰 모델인 Osprey 7B와 GLaMM 7B를 각각 1.0, 1.4포인트 능가한다.

**Osprey vs. GLaMM 구조적 차이**:

- **Osprey**: 마스크를 **입력**으로 받아 픽셀 수준 **이해(언어 출력)** → "이 마스크 영역은 무엇인가?"
- **GLaMM**: 텍스트로부터 픽셀 수준 **출력(마스크 생성)** → "이 개념을 픽셀로 그라운딩하라"

---

## 7. 연구 영향과 앞으로의 연구에서 고려할 점

### 7.1 연구에 미치는 영향

#### (1) 픽셀 수준 MLLM 연구의 기반 제공

Osprey-724K 데이터셋과 Osprey 모델은 실제 응용에서 MLLMs의 픽셀 수준 시각적 이해 발전을 촉진할 것으로 기대된다.

#### (2) 새로운 평가 지표 도입

Osprey가 정의한 Semantic Similarity(Sem. Sim.)와 Semantic IoU(Sem. IoU) 메트릭은 NVIDIA 및 UC Berkeley의 Describe Anything Model에서도 채택되었다.

#### (3) 후속 연구로의 파급

후속 연구인 VideoRefer Suite가 CVPR 2025에 채택되었으며, 이는 비디오 영역 참조(video referring)에 집중하는 연구로 Osprey의 연장선에 있다.

### 7.2 앞으로 연구 시 고려할 점

#### ① 이미지-픽셀 수준 능력의 동시 학습

픽셀 수준 이해 데이터와의 결합 학습은 이미지 수준 능력 저하로 이어지는 현상이 LISA와 GLaMM 등에서 광범위하게 관찰되며, 이 충돌을 제거하기 위한 데이터 구성 방식 연구가 필요하다.

#### ② 전문 세그멘테이션 모델 의존성 탈피

Pixel-SAIL은 단일 트랜스포머로 세밀한 이해를 확장하는 접근법을 제시하며, 처음으로 추가적인 시각 전문가(visual encoder, segmentation model) 없이도 4개의 공개 참조 세그멘테이션 벤치마크에서 더 강력한 성능을 달성할 수 있음을 증명했다.

#### ③ 파트 수준 세그멘테이션 능력 확장

다중 세분도 세그멘테이션 능력의 부족으로 인해 파트 수준 세그멘테이션을 수행할 수 없는 한계가 있으며, 파트 수준 시각 입력을 추가한 더 강력하고 범용적인 지각 모듈을 통해 이 문제를 해결할 수 있다.

#### ④ 비디오 및 3D 도메인으로의 확장

OMG-LLaVA 등의 후속 모델도 여전히 픽셀 수준 공간-시간 추론을 수행하지 못하며, 이는 관련 데이터셋의 부족에 기인한다. 비디오 및 3D 영역으로의 픽셀 수준 이해 확장은 중요한 향후 연구 방향이다.

#### ⑤ 데이터 효율성 및 소규모 데이터 일반화

MLLMs는 상세한 픽셀 수준 정보를 표현하는 데 어려움을 겪으며, 다른 MLLM 기반 방법들과 비교하여 유의미하게 적은 학습 데이터를 사용하는 효율적인 접근법의 필요성이 강조된다.

#### ⑥ 도메인 특화 일반화

Osprey를 포함한 모든 기존 모델들은 자연 장면 데이터로 학습되어 원격 탐지(RS) 이미지 등 특수 도메인 처리에서 열등한 성능을 보이므로, 도메인 적응(domain adaptation) 연구가 필요하다.

---

## 📚 참고 자료 출처

| 번호 | 출처 |
|---|---|
| 1 | Yuan, Y. et al., **"Osprey: Pixel Understanding with Visual Instruction Tuning"**, CVPR 2024, arXiv:2312.10032. [https://arxiv.org/abs/2312.10032](https://arxiv.org/abs/2312.10032) |
| 2 | CVPR 2024 Open Access: [https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Osprey_Pixel_Understanding_with_Visual_Instruction_Tuning_CVPR_2024_paper.html](https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Osprey_Pixel_Understanding_with_Visual_Instruction_Tuning_CVPR_2024_paper.html) |
| 3 | GitHub (공식 코드): [https://github.com/CircleRadon/Osprey](https://github.com/CircleRadon/Osprey) |
| 4 | HuggingFace Paper Page: [https://huggingface.co/papers/2312.10032](https://huggingface.co/papers/2312.10032) |
| 5 | Unite.AI 분석 기사: [https://www.unite.ai/visual-instruction-tuning-for-pixel-level-understanding-with-osprey/](https://www.unite.ai/visual-instruction-tuning-for-pixel-level-understanding-with-osprey/) |
| 6 | Rasheed, H. et al., **"GLaMM: Pixel Grounding Large Multimodal Model"**, CVPR 2024. [https://openaccess.thecvf.com/content/CVPR2024/papers/Rasheed_GLaMM_Pixel_Grounding_Large_Multimodal_Model_CVPR_2024_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Rasheed_GLaMM_Pixel_Grounding_Large_Multimodal_Model_CVPR_2024_paper.pdf) |
| 7 | Zhang, A. et al., **"OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding"**, NeurIPS 2024. arXiv:2406.19389 |
| 8 | **"Pixel-SAIL: Single Transformer For Pixel-Grounded Understanding"**, arXiv:2504.10465 (2025). [https://arxiv.org/html/2504.10465](https://arxiv.org/html/2504.10465) |
| 9 | **"EarthMarker: A Visual Prompting Multi-modal Large Language Model for Remote Sensing"**, arXiv:2407.13596 |
| 10 | **"SAM4MLLM: Enhance Multi-Modal Large Language Model for Referring Expression"**, ECCV 2024 |
| 11 | IEEE Xplore (CVPR 2024): [https://ieeexplore.ieee.org/document/10656218/](https://ieeexplore.ieee.org/document/10656218/) |

> ⚠️ **주의사항**: 본 답변에서 수식(Mask-Pooling, Multi-Level Fusion 등)의 일부 표기는 논문에서 명시적으로 제시된 수식을 기반으로 하되, 논문이 직접 기호를 정의하지 않은 부분은 논문 설명으로부터 합리적으로 형식화한 것임을 밝힙니다. 정확한 구현 세부 사항은 공식 GitHub 코드를 참조하시기 바랍니다.
