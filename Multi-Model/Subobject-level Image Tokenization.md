
# Subobject-level Image Tokenization

> **논문 정보**
> - **제목**: Subobject-level Image Tokenization
> - **저자**: Delong Chen, Samuel Cahyawijaya, Jianfeng Liu, Baoyuan Wang, Pascale Fung
> - **arXiv**: [2402.14327](https://arxiv.org/abs/2402.14327) (v1: 2024.02.22, v3: 2025.03.12)
> - **학회**: **ICML 2025** (공식 채택)
> - **코드**: [GitHub - ChenDelong1999/subobjects](https://github.com/ChenDelong1999/subobjects)
> - **Meta AI Research 공식 페이지**: [ai.meta.com](https://ai.meta.com/research/publications/subobject-level-image-tokenization/)

---

## 1. 🔑 핵심 주장 및 주요 기여 (간결 요약)

기존의 패치 기반 이미지 토크나이제이션은 시각 세계의 형태론적 구조(morphology)를 무시하여 이미지 이해 학습의 효율성을 저해한다. 이에 영감을 받아, 본 논문은 NLP의 **서브워드(subword) 토크나이제이션**에서 착안하여 **서브오브젝트(subobject) 수준의 적응형 토큰 분할**을 도입하고, 슈퍼픽셀, SAM, 그리고 새롭게 제안하는 **EPOC(Efficient and PanOptiC)** 이미지 토크나이저를 탐구한다.

### 🏆 5대 핵심 기여

| # | 기여 내용 |
|---|-----------|
| 1 | 서브오브젝트 레벨 이미지 토크나이제이션 개념 제안 |
| 2 | EPOC (DirectSAM + Watershed) 경량 토크나이저 개발 |
| 3 | 비정형 토큰을 처리하는 SeqAE 아키텍처 설계 |
| 4 | VLM(Vision-Language Model) 통합 프레임워크 구축 |
| 5 | 5개 데이터셋 내재적 + 4개 데이터셋 외재적 평가 체계 수립 |

실험 결과, 서브오브젝트 토크나이제이션은 **더 적은 시각 토큰**을 사용하면서도 **빠른 수렴(faster convergence)**과 **더 나은 일반화(better generalization)**를 달성한다는 것이 입증되었다.

---

## 2. 🔬 상세 분석

### 2-1. 해결하고자 하는 문제

트랜스포머 기반 비전 모델은 통상적으로 이미지를 **고정 크기의 정사각형 패치**로 토크나이즈하는데, 이는 이미지 콘텐츠에 대한 적응성이 없고 픽셀 간의 내재적 그룹핑 구조를 간과한다.

이러한 패치 레벨 이미지 토크나이제이션은 NLP의 **캐릭터 N-gram 레벨 텍스트 토크나이제이션**에 대응되며, 이는 의미론적 경계를 무시하고 어마어마한 어휘 크기로 인해 비효율적이고 비효과적인 경향이 있다. 유사하게, 패치 분할 연산은 객체의 형태론에 적응하지 못하고 이미지 내 픽셀 그룹핑 구조를 무시한다.

본 논문은 NLP의 서브워드 레벨 텍스트 토크나이제이션에서 영감을 받아 **"서브오브젝트(subobject)"** 레벨 이미지 토크나이제이션 개념을 소개한다. 이는 객체(object)와 픽셀(pixel) 사이의 중간 레벨에 위치하며, 단어(word)와 문자(character) 사이의 서브워드와 유사하다. 서브오브젝트는 지각적으로 의미 있는 시각적 구조를 가진 시각 엔티티(예: 객체의 부분들)이다.

---

### 2-2. 제안하는 방법

#### (A) 서브오브젝트 개념의 계층 구조

$$
\text{Pixel} \subset \text{Subobject (= Subword analog)} \subset \text{Object}
$$

서브오브젝트의 개념은 저수준 비전의 슈퍼픽셀(superpixel) 및 고수준 비전의 파트 분할(part segmentation) 개념과 관련되지만, 의미론적으로 유의미하고 오픈-어휘(open-vocabulary)이며 파놉틱(panoptic)한 특성을 강조하며, 이미지 토크나이제이션에의 응용과 NLP의 서브워드와의 연결성을 부각시킨다.

#### (B) EPOC (Efficient and PanOptiC) 토크나이저

EPOC는 **경계 검출(boundary detection)** — 컴팩트한 모델로도 잘 처리될 수 있는 단순한 작업 — 과 픽셀이 분할되지 않은 채 남겨지지 않음을 본질적으로 보장하는 **워터셰드 분할(watershed segmentation)**을 결합한다.

$$
\text{EPOC} = \underbrace{\text{DirectSAM}}_{\text{Boundary Detection}} + \underbrace{\text{Watershed}}_{\text{Non-parametric Segmentation}}
$$

DirectSAM-b0는 단 **3.7M 파라미터**의 경량 모델로, Watershed 알고리즘과 결합되어 EPOC 토크나이저를 구성한다 (EPOC = DirectSAM + Watershed).

Watershed 알고리즘은 비파라메트릭(non-parametric)이며 CPU에서 효율적으로 실행되어, EPOC를 전반적으로 효율적인 이미지 토크나이저로 만든다.

---

#### (C) SeqAE (Sequence-to-sequence AutoEncoder)

비정형 모양의 서브오브젝트를 컴팩트한 임베딩 벡터로 압축하기 위한 핵심 아키텍처:

SeqAE는 **트랜스포머 아키텍처** 기반의 시퀀스-투-시퀀스 오토인코더로, 인코더와 디코더로 구성된다. 각 서브오브젝트 세그먼트의 원시 픽셀이 시퀀스로 평탄화(flatten)되어 입력되며, 이는 가변 크기의 서브오브젝트를 최대 컨텍스트 길이까지 서로 다른 길이의 시퀀스로 처리한다. 더 큰 세그먼트는 다운샘플링된다.

학습 가능한 쿼리 토큰(learnable query tokens)이 입력 시퀀스 앞에 추가된다. 이 쿼리들은 픽셀 토큰들과 상호작용하고, 인코더 레이어를 통과한 후 컴팩트한 잠재 벡터를 추출하는 데 사용된다. 병목 선형 레이어(bottleneck linear layer)가 쿼리 토큰 표현을 압축하고, 디코더는 압축된 쿼리 토큰에 어텐션을 적용한다.

픽셀 재구성에는 범주형 토큰 확률 대신 **정규화된 실수 값 픽셀 강도를 직접 예측**하는 실수값 회귀(real-valued regression)를 사용하며, **MSE(Mean Squared Error) 손실**을 활용한다.

$$
\mathcal{L}_{\text{SeqAE}} = \frac{1}{N} \sum_{i=1}^{N} \left\| \hat{x}_i - x_i \right\|^2
$$

여기서 $\hat{x}_i$는 디코더의 예측 픽셀 값, $x_i$는 실제 픽셀 값이다.

---

#### (D) VLM 통합 구조 및 임베딩 방식

SeqAE에서 생성된 서브오브젝트 임베딩은 **학습 가능한 선형 레이어**를 통해 LLM의 임베딩 공간으로 투영된다. 각 서브오브젝트의 바운딩 박스 좌표 $(x, y, \text{width}, \text{height})$는 또 다른 학습 가능한 선형 레이어를 통해 LLM의 임베딩 공간에 투영된다.

$$
\mathbf{e}_{\text{final}} = \underbrace{W_1 \cdot \mathbf{z}_{\text{SeqAE}}}_{\text{Content Embedding}} + \underbrace{W_2 \cdot \mathbf{b}_{\text{bbox}}}_{\text{Positional Embedding}}
$$

여기서 $\mathbf{z}\_{\text{SeqAE}}$는 SeqAE 잠재 벡터, $\mathbf{b}_{\text{bbox}} = [x, y, w, h]$는 바운딩 박스 좌표이다.

특수 토큰 `<SOI>`와 `<EOI>`가 서브오브젝트 시퀀스의 시작과 끝을 표시하는 데 사용된다. 각 서브오브젝트의 이미지 내 위치를 표현하기 위해 위치 임베딩이 통합되며, 원래 LLM의 위치 임베딩은 1D이므로 2D 이미지를 처리하기 위한 별도의 고안이 필요하다. LVLM 학습 시 서브오브젝트 토큰에 대한 크로스 엔트로피 손실 계산은 생략된다(서브오브젝트는 1D 인과적 구조를 형성하지 않기 때문).

LLM 파라미터의 효율적 적응을 위해 **LoRA**가 사용되며, 손실 함수는 텍스트 토큰에 대해서만 크로스 엔트로피 손실을 계산한다.

$$
\mathcal{L}_{\text{VLM}} = -\sum_{t} \log P(y_t^{\text{text}} \mid y_{<t}, \mathbf{E}_{\text{subobj}})
$$

---

### 2-3. 모델 구조 전체 파이프라인

```
입력 이미지
    │
    ▼
[EPOC Tokenizer]
 DirectSAM (3.7M params)
    → 경계 맵(boundary map) 생성
 Watershed Segmentation
    → 서브오브젝트 마스크 집합 {M_1, ..., M_K}
    │
    ▼
[SeqAE]
 픽셀 시퀀스 평탄화 → Transformer Encoder
    → 학습 가능한 쿼리 토큰 → 잠재 벡터 z
    │
    ▼
[VLM Integration]
 W1·z_SeqAE + W2·b_bbox → LLM 임베딩 공간 투영
 <SOI> [서브오브젝트 임베딩...] <EOI>
    │
    ▼
[LLM (LoRA fine-tuned)]
    → 텍스트 출력 (캡션, 분류 등)
```

---

### 2-4. 성능 향상

#### 내재적 평가 (Intrinsic Evaluation)

내재적 평가는 세 가지 핵심 측면을 포함한다: 1) **형태론(morphology)** — 분할이 의미론적 경계와 일치하는지, 2) **단의미성(monosemanticity)** — 개별 토큰이 여러 의미를 포함하지 않는지, 3) **효율성(efficiency)** — 추가 계산 오버헤드가 얼마나 발생하는지. 5개 데이터셋에 대한 결과는 SAM과 EPOC 모두 강한 형태론적 정렬과 토큰 단의미성을 보이며, EPOC는 **효율성에서 상당한 이점**을 누린다는 것을 보여준다.

패치 기반 토크나이제이션은 이미지 형태론적 구조와 매우 낮은 정렬도를 보이는 반면, 모든 적응형 토크나이제이션 방법들은 이를 명확히 능가한다. 슈퍼픽셀 분할은 다른 학습된 토크나이제이션 방법들보다 성능이 낮은데, 이는 상향식 픽셀 그룹핑이 고수준 의미 구조를 캡처하는 데 필요한 전체론적 이해를 결여하고 있음을 보여준다.

#### 외재적 평가 (Extrinsic Evaluation)

서브오브젝트 레벨 토크나이제이션을 사용한 LVLM은 패치 기반 베이스라인에 비해 **더 빠른 학습(더 낮은 퍼플렉시티)**과 객체 속성(크기, 재질, 형태) 및 객체 수에 대한 설명 생성에서 **현저히 높은 정확도**를 달성한다.

서브오브젝트 레벨 토크나이제이션은 **훈련 퍼플렉시티의 현저히 빠른 감소**를 가능하게 하며, 이는 패치 레벨 토크나이제이션을 서브오브젝트 레벨로 교체할 때 동일한 모델이 훨씬 더 빠르게 학습할 수 있음을 보여준다.

---

### 2-5. 한계점

서브오브젝트 분할 모듈은 **사전 훈련된 인스턴스 및 의미론적 분할 모델에 의존**하며, 이로 인해 편향이나 오류가 발생할 수 있다. 또한 매우 복잡하거나 혼잡한 장면을 처리하는 데 있어 **확장성 문제**가 있을 수 있다.

논문은 서브오브젝트 레벨 토크나이제이션의 **계산 비용 및 런타임 성능에 대한 심층 분석을 제공하지 않는다**. 이는 자원이 제한된 환경에서의 실제 배포를 고려할 때 중요한 실용적 고려 사항이다.

객체 레벨 토크나이제이션은 파놉틱 분할 기반이므로 **어휘 외 문제(out-of-vocabulary problem)**에 취약하며, 슈퍼픽셀 분할은 상향식 픽셀 그룹핑에 의존하기 때문에 기저 구조를 파악하는 능력이 제한된다.

---

## 3. 🌐 일반화 성능 향상 가능성

### 핵심 메커니즘

실험 결과, 서브오브젝트 토크나이제이션은 **더 적은 시각 토큰을 사용하면서도 더 빠른 수렴과 더 나은 일반화**를 가능하게 한다.

객체 레벨 토크나이제이션은 인-도메인(in-domain)에서 강한 성능을 보이지만 **제로샷(zero-shot) 설정에서 어려움을 겪는** 반면, EPOC는 모든 모델이 제로샷 설정에 있는 PPP 및 PIN++ 데이터셋에서 파놉틱 분할 모델과 SAM ViT-B 모델 대비 **명확한 우위**를 보이며 파레토 최적 성능(Pareto optimal performance)을 달성한다.

### 일반화 향상의 세 가지 근거

**① 모노시맨틱 토큰(Monosemantic Token) 형성**

EPOC의 분할은 인간 주석의 객체 및 파트 레벨 시각적 형태론과 잘 정렬되어 **더 단의미적(monosemantic)인 토큰**을 생성하고 상당한 효율성 이점을 제공한다.

하나의 토큰이 하나의 의미 단위에 대응함으로써 모델이 개념 간의 경계를 명확히 학습할 수 있어 일반화 성능이 향상된다:

$$
\text{Monosemanticity} = \frac{|\{t : \text{semantic}(t) = 1\}|}{|T|} \uparrow
$$

**② 형태론적 귀납적 편향(Morphological Inductive Bias)**

서브오브젝트 레벨 토크나이제이션은 이미지 내 실제 형태와 구조에 동적으로 적응하며, NLP에서 단어가 더 작은 의미 단위(서브워드)로 구성되는 것처럼 이미지를 시각적으로 일관된 부분들로 분할한다.

**③ 토큰 수 감소에 의한 샘플 효율성**

서브오브젝트 레벨 토크나이제이션은 전통적인 패치 레벨 토크나이제이션에 비해 이미지를 객체 및 속성 설명으로 번역하는 효율적인 학습을 **현저히 촉진**한다.

더 적은 토큰으로 더 많은 의미 정보를 표현하여 어텐션 메커니즘의 효율이 높아진다:

$$
\text{Attention Complexity} = \mathcal{O}(N^2) \xrightarrow{\text{Subobject}} \mathcal{O}(K^2), \quad K \ll N
$$

여기서 $N$은 패치 수, $K$는 서브오브젝트 수이다.

---

## 4. 📊 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법 | 토큰 유형 | 적응성 | 의미론적 정렬 | 주요 한계 |
|------|------|------|-----------|--------|--------------|-----------|
| **ViT** (Dosovitskiy et al.) | 2020 | 고정 패치 | 정사각형 패치 | ❌ | ❌ | 형태론 무시 |
| **T2T-ViT** (Yuan et al.) | 2021 | 소프트 분할 | 오버래핑 패치 | 부분적 | 부분적 | 여전히 고정 구조 |
| **SAM** (Kirillov et al.) | 2023 | 프롬프트 기반 분할 | 마스크 | ✅ | ✅ | 계산 비용 높음 |
| **BEiT** (Bao et al.) | 2022 | dVAE 토큰 | 이산 코드 | 부분적 | 부분적 | 패치 기반 유지 |
| **Subobject** (**본 논문**) | 2024 | EPOC + SeqAE | 서브오브젝트 | ✅ | ✅ | 분할 오류 가능성 |
| **Adaptive Patch** (Duggal et al.) | 2024 | 순환 할당 | 가변 길이 패치 | ✅ | 부분적 | 학습 가속 제한적 |

ViT는 현재 컴퓨터 비전 백본 아키텍처의 사실상 표준이지만, 언어 모델이 서브워드 토크나이저를 사용하여 토큰당 가변 비트를 처리하는 것과 달리 ViT는 이미지를 동일한 크기의 패치로 분할하며 각각이 토큰이 된다. 이는 특히 고해상도에서 **엄청난 수의 토큰**을 초래할 수 있다.

일부 최근 연구들은 더 복잡한 시각적 입력에 더 많은 토큰을 동적으로 할당하는 **적응형 시각 토크나이저**를 탐구하지만, 학습이나 추론을 의미 있게 가속화하지는 못한다.

---

## 5. 🔭 앞으로의 연구에 미치는 영향과 고려할 점

### 5-1. 연구에 미치는 영향

**① 비전-언어 패러다임 전환**

서브오브젝트 레벨 이미지 토크나이제이션은 비전-언어 학습을 위한 패치 레벨 토크나이제이션의 **실행 가능한 대안**이다. 이는 단순히 기존 방법을 개선하는 것이 아니라, 이미지를 처리하는 근본적인 방식을 재정의한다는 점에서 패러다임 전환적 의미를 갖는다.

**② 비디오 도메인으로의 확장**

이미 관련 후속 연구로 **Subobject Video Tokenization**이 제안되어, 파놉틱 서브오브젝트 궤적(Panoptic Sub-object Trajectory)을 활용한 그라운디드 비디오 토크나이제이션을 탐구하고 있다.

**③ 멀티모달 AI의 토큰 효율성 연구 촉진**

본 연구는 이미지 토크나이제이션의 흥미로운 전진을 나타내며, **제로샷 분할 및 시각적 특징 추출** 등의 분야에서 추가 탐구를 위한 새로운 방향을 열어준다.

**④ NLP-CV 유사성 기반 연구 방법론 확립**

NLP의 서브워드 토크나이제이션의 성공에서 영감을 받아 컴퓨터 비전에 적용하는 이 접근법은, NLP의 방법론을 비전 영역에 체계적으로 이식하는 **연구 방법론적 틀**을 제공한다.

---

### 5-2. 앞으로 연구 시 고려할 점

| 카테고리 | 고려 사항 | 중요도 |
|----------|-----------|--------|
| **확장성** | 혼잡하고 복잡한 장면에서의 분할 안정성 | ⭐⭐⭐⭐⭐ |
| **계산 비용** | 실시간 추론 및 엣지 디바이스 배포를 위한 최적화 | ⭐⭐⭐⭐⭐ |
| **오류 전파** | 분할 오류가 하위 VLM 성능에 미치는 영향 분석 | ⭐⭐⭐⭐ |
| **도메인 적응** | 의료, 위성, 원격 감지 등 특수 도메인에서의 성능 | ⭐⭐⭐⭐ |
| **토큰 수 가변성** | 배치 처리 시 가변 토큰 수 처리 전략 | ⭐⭐⭐⭐ |
| **학습 안정성** | 비정형 토큰 기반 학습의 안정성 보장 방법 | ⭐⭐⭐ |
| **벤치마크 확대** | 더 다양한 VQA, 추론 벤치마크에서의 평가 | ⭐⭐⭐ |

구체적으로 향후 연구에서 집중해야 할 방향:

1. **분할 품질의 견고성(Robustness)**: 저조도, 폐색(occlusion), 고밀도 장면에서의 EPOC 분할 신뢰도 향상
2. **동적 해상도 적응**: 다양한 입력 해상도에 대한 서브오브젝트 수 자동 조정 메커니즘
3. **계층적 서브오브젝트**: 파트→객체→장면 수준의 다단계 계층적 토크나이제이션 탐구
4. **자기 지도 사전 훈련**: EPOC 기반 마스킹을 활용한 MAE(Masked Autoencoder) 스타일 사전 훈련
5. **다중 모달 확장**: 포인트 클라우드, 깊이 맵 등 비RGB 데이터로의 확장

논문이 유망한 결과를 보여주지만, 기술의 한계를 해결하고 더 넓은 응용을 탐구하기 위한 추가 연구가 필요하다. 컴퓨터 비전 분야가 계속 발전함에 따라, 서브오브젝트 레벨 토크나이제이션과 같은 혁신들이 더 강력하고 다재다능한 이미지 이해 시스템 개발에 중요한 역할을 할 수 있다.

---

## 📚 참고 자료 (출처 목록)

1. **arXiv 원문**: Chen, D., Cahyawijaya, S., Liu, J., Wang, B., & Fung, P. (2024). *Subobject-level Image Tokenization*. arXiv:2402.14327. https://arxiv.org/abs/2402.14327
2. **ICML 2025 공식 포스터**: https://icml.cc/virtual/2025/poster/44334
3. **GitHub 공식 저장소**: ChenDelong1999/subobjects. https://github.com/ChenDelong1999/subobjects
4. **Meta AI Research 공식 페이지**: https://ai.meta.com/research/publications/subobject-level-image-tokenization/
5. **OpenReview (ICML 심사 페이지)**: https://openreview.net/forum?id=imkFoKwFwd
6. **HuggingFace Papers**: https://huggingface.co/papers/2402.14327
7. **Papers With Code**: https://paperswithcode.com/paper/subobject-level-image-tokenization
8. **Moonlight Literature Review**: https://www.themoonlight.io/en/review/subobject-level-image-tokenization
9. **AI Models FYI**: https://www.aimodels.fyi/papers/arxiv/subobject-level-image-tokenization
10. **arXiv HTML 전문 (v3)**: https://arxiv.org/html/2402.14327v3
11. **관련 배경 연구 - ViT**: Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words*. arXiv:2010.11929
12. **관련 배경 연구 - SAM**: Kirillov et al. (2023). *Segment Anything*. ICCV 2023
13. **관련 배경 연구 - 적응형 패치**: Duggal et al. (2024). *Adaptive Length Image Tokenization via Recurrent Allocation*. arXiv:2411.02393
