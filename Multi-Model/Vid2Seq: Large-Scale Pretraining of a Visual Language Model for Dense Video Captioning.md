# Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning

---

## 1. 핵심 주장과 주요 기여 요약

**Vid2Seq**는 **dense video captioning**(영상 내 모든 이벤트의 시간적 위치를 파악하고 자연어로 설명하는 과제)을 위한 멀티모달 시각-언어 모델이다. 핵심 주장과 기여는 다음과 같다:

1. **통합 시퀀스 생성 모델**: 언어 모델에 **특수 시간 토큰(time tokens)**을 추가하여, 이벤트의 시간적 경계(temporal boundaries)와 텍스트 캡션을 **하나의 출력 시퀀스**로 동시에 예측한다. 이를 통해 기존의 2-stage 접근법(이벤트 검출 → 캡션 생성)을 단일 단계로 통합했다.

2. **비지도 나레이션 비디오 활용 대규모 사전학습**: 수동 어노테이션 없이도, 나레이션 비디오의 **전사된 음성(transcribed speech)** 문장 경계를 **의사(pseudo) 이벤트 경계**로, 전사된 음성 문장을 **의사 이벤트 캡션**으로 재구성하여 대규모 사전학습을 수행한다.

3. **SOTA 달성 및 범용성**: YouCook2, ViTT, ActivityNet Captions 등 다수의 dense video captioning 벤치마크에서 SOTA를 달성했으며, video paragraph captioning, video clip captioning, few-shot 설정에서도 우수한 일반화 성능을 보여준다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Dense video captioning은 **untrimmed(편집되지 않은) 긴 비디오** 내 모든 이벤트를 (1) **시간적으로 정위(temporal localization)**하고, (2) 각 이벤트를 **자연어로 설명**해야 하는 과제이다.

**기존 방법의 한계**:
- **2-stage 접근법**: 이벤트를 먼저 검출한 후 캡셔닝하는 파이프라인으로, 두 태스크 간 상호작용이 부족하다.
- **제한된 학습 데이터**: 수동 어노테이션 데이터셋(YouCook2: ~2K 영상, ActivityNet: ~20K 영상)의 규모가 작아 효과적인 학습이 어렵다.
- **태스크 특화 컴포넌트**: 이벤트 카운터 등 별도 모듈이 필요하여 일반화가 어렵다.

### 2.2 제안하는 방법

#### 2.2.1 모델 구조 (Vid2Seq Architecture)

Vid2Seq는 **멀티모달 인코더-디코더** 구조로, 세 가지 주요 모듈로 구성된다:

**① Visual Encoder $f$**
- **Spatial Encoder $f^s$**: CLIP ViT-L/14를 사용하여 각 프레임을 독립적으로 인코딩 (frozen)

$$x^s = f^s(x) \in \mathbb{R}^{F \times d}$$

- **Temporal Encoder $f^t$**: Transformer encoder로 프레임 간 시간적 상호작용을 모델링

$$x^t = f^t(x^s + x^p) \in \mathbb{R}^{F \times d}$$

여기서 $x^p \in \mathbb{R}^{F \times d}$는 학습 가능한 시간 위치 임베딩이다.

**② Text Encoder $g$**
- Token Embedder $g^s \in \mathbb{R}^{(V+N) \times d}$로 전사된 음성 시퀀스 $y \in \{1, ..., V+N\}^S$를 임베딩:

$$y^s = g^s(y) \in \mathbb{R}^{S \times d}$$

- Transformer Encoder $g^t$로 문맥화된 음성 임베딩 생성:

$$y^t = g^t(y^s) \in \mathbb{R}^{S \times d}$$

**③ Text Decoder $h$**
- 시각/음성 임베딩 $[x^t, y^t]$에 cross-attend하며 자기회귀적으로 이벤트 시퀀스 $z$를 생성:

$$z^t_k = h^t(h^s(\hat{z}^t_{ < k}), x^t, y^t) \in \mathbb{R}^d$$

- Language Modeling Head $h^l \in \mathbb{R}^{d \times (V+N)}$가 텍스트+시간 토큰에 대한 확률 분포를 예측:

```math
z^l_k = h^l(z^t_k) \in \mathbb{R}^{V+N}
```

**시간 토큰화(Time Tokenization)**: SentencePiece 토크나이저(어휘 크기 $V = 32,128$)에 $N = 100$개의 시간 토큰을 추가한다. 비디오 길이 $T$를 $N$개의 균등 간격 타임스탬프로 양자화한다:

$$t_{\text{start}} = \left\lfloor \frac{s \times N}{T} \right\rfloor$$

예: $s = 3.02$ s, $T = 49.70$ s, $N = 100$ 이면 $t_{\text{start}} = \lfloor \frac{3.02 \times 100}{49.70} \rfloor = 6$

**이벤트 시퀀스 구성**: 모든 이벤트를 시작 시간 순으로 정렬하여 단일 시퀀스로 연결:

$$z = [\text{BOS}, t_{\text{start}_1}, t_{\text{end}_1}, z_{1_1}, \ldots, z_{1_{l_1}}, t_{\text{start}_2}, \ldots, \text{EOS}]$$

#### 2.2.2 사전학습 (Pretraining on Unlabeled Narrated Videos)

두 가지 학습 목표 모두 다음 **최대우도 손실(maximum likelihood loss)**에 기반한다:

$$\mathcal{L}_\theta(x, y, z) = -\frac{1}{\sum_{k=1}^{L-1} w_k} \sum_{k=1}^{L-1} w_k \log p_\theta(z_{k+1} | x, y, z_{1:k})$$

여기서 $L$은 디코더 타겟 시퀀스 길이, $w_k = 1\ \forall k$, $\theta$는 학습 가능 파라미터, $p_\theta$는 텍스트+시간 토큰 어휘에 대한 출력 확률 분포이다.

**① Generative Objective**: 시각 입력 $x$만 인코더에 주어지고, 디코더가 전사 음성 시퀀스 $y$를 예측해야 한다. 텍스트 인코더에는 입력이 주어지지 않으므로, 텍스트 전용 단축(shortcut) 학습을 방지한다.

**② Denoising Objective**: T5에서 영감을 받아, 전사 음성 시퀀스의 토큰 스팬을 확률 $P$, 평균 길이 $M$으로 무작위 마스킹한다. 인코더에 영상 프레임 $x$와 손상된 음성 시퀀스 $\tilde{y}$를 입력하고, 디코더가 마스킹된 스팬 $\bar{y}$를 복원한다. 이 목표는 시각 인코더, 텍스트 인코더, 텍스트 디코더를 공동으로 정렬한다.

**텍스트 초기화**: 텍스트 인코더와 디코더는 **T5-Base** (웹 텍스트 코퍼스에서 denoising loss로 사전학습)로 초기화한다. 토큰 임베딩 레이어를 공유: $g^s = h^s \in \mathbb{R}^{(V+N) \times d}$.

### 2.3 성능 향상

#### Dense Video Captioning (Table 5)

| 데이터셋 | 메트릭 | 기존 SOTA | Vid2Seq | 향상 |
|--------|--------|----------|---------|------|
| YouCook2 | CIDEr | 28.9 (PDVC†) | **47.1** | +18.2 |
| YouCook2 | SODA_c | 4.9 (PDVC†) | **7.9** | +3.0 |
| ViTT | CIDEr | 25.0 (E2ESG) | **43.5** | +18.5 |
| ActivityNet | CIDEr | 29.3 (PDVC†) | **30.1** | +0.8 |

#### Video Paragraph Captioning (Table 7)
- GT proposals를 사용하는 기존 방법들(MART, GVDSup 등)을 **GT proposals 없이도** 능가
- YouCook2 CIDEr: **50.1**, ActivityNet CIDEr: **28.0**

#### Video Clip Captioning (Table 8)
- MSR-VTT CIDEr: **64.6** (YT-Temporal-1B 사전학습 시)
- MSVD CIDEr: **146.2**

#### Few-shot 설정 (Table 9)
- 10%의 학습 데이터만으로도 상당한 성능 달성 (YouCook2 CIDEr: 18.4)

### 2.4 한계

1. **ActivityNet Captions에서의 이벤트 localization**: PDVC나 UEDVC와 비교 시 event localization 성능이 다소 낮다 (Table 6). 이는 Vid2Seq가 이벤트 카운터 등 태스크 특화 컴포넌트를 포함하지 않기 때문이다.

2. **전사 음성의 한계**: 전사 음성이 반드시 시각적 내용을 정확히 기술하지 않으며, 시간적 정렬도 부정확할 수 있다 (약한 감독 신호).

3. **계산 비용**: 64 TPU v4 칩에서 512 배치 크기로 20만 iteration 사전학습에 약 1일이 소요되며, 이는 상당한 컴퓨팅 자원을 요구한다.

4. **환각(Hallucination) 문제**: 논문의 Figure 6 마지막 예시에서 시각적으로 근거 없는 이벤트를 생성하는 경우가 관찰된다 (예: 'one man hats off to the camera').

5. **음성이 없는 비디오**: ActivityNet Captions의 68%는 전사 나레이션이 없어, 텍스트 인코더 입력이 비어있게 된다.

---

## 3. 모델의 일반화 성능 향상 가능성

Vid2Seq의 일반화 성능은 여러 차원에서 검증되었으며, 향상 가능성도 다양하게 존재한다.

### 3.1 검증된 일반화 능력

1. **다중 데이터셋 일반화**: YouCook2(요리), ViTT(일반 교육), ActivityNet(다양한 활동) 등 서로 다른 도메인의 세 데이터셋에서 모두 SOTA를 달성했다.

2. **다중 태스크 일반화**: dense video captioning뿐 아니라, video paragraph captioning과 video clip captioning에서도 우수한 성능을 보인다.

3. **Few-shot 일반화**: 1%~50%의 학습 데이터만으로도 유의미한 성능을 달성한다 (Table 9, 10).

4. **사전학습 데이터 일반화**: YT-Temporal-1B뿐 아니라 HowTo100M에서 사전학습해도 효과적이며, 도메인이 유사한 경우(HowTo100M → YouCook2) 더 높은 성능을 달성한다 (Table 4, row 6).

### 3.2 일반화 향상의 핵심 요인

**① Untrimmed 나레이션 비디오 사전학습**:
- 짧은 클립 대비 긴 비디오에서의 사전학습이 크게 유리하다 (Table 1: row 3 vs row 2).
- 나레이션 문장 수 제한을 제거할수록 성능이 향상된다 (Table 11).

**② 시간 토큰의 역할**:
- 시간 토큰을 사용한 사전학습이 성능을 크게 향상시킨다 (Table 1: row 4 vs row 3, YouCook2 CIDEr 35.0 → 47.1).

**③ Joint captioning + localization**:
- 캡셔닝과 localization을 동시에 수행하는 것이 localization만 수행하는 것보다 우수하다 (Table 3).

**④ 모델 스케일링**:
- 언어 모델 크기 (T5-Small → T5-Base), 시각 백본 크기 (ViT-B/16 → ViT-L/14), 사전학습 데이터 규모를 키울수록 성능이 향상된다 (Table 4, 12).

### 3.3 추가 일반화 향상 가능성

1. **더 큰 언어 모델** (T5-Large, T5-XL 등)로의 확장이 추가 성능 향상을 가져올 수 있다.
2. **오디오 모달리티** 통합: 현재는 전사 음성만 사용하지만, 원시 오디오 신호를 직접 활용하면 음성이 없는 비디오에서도 환경음 등의 추가 단서를 확보할 수 있다.
3. **더 정밀한 시간 토큰** ($N > 100$)과 **계층적 시간 표현**을 통한 세밀한 시간 정위 개선 가능성이 있다.
4. **다국어 나레이션 비디오**로의 확장으로 언어 다양성을 증가시킬 수 있다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

1. **시퀀스-투-시퀀스 프레임워크의 범용성 입증**: dense video captioning을 시퀀스 생성 문제로 정의한 것은 temporally-grounded video QA, temporal action localization 등 다른 시간적 비디오 이해 태스크에도 직접 적용 가능하다.

2. **약한 감독 신호의 대규모 활용**: 나레이션 비디오를 dense captioning의 사전학습 데이터로 활용하는 패러다임은, 어노테이션 비용을 대폭 줄이면서도 높은 성능을 달성할 수 있음을 보여준다.

3. **시간 토큰 개념의 확산**: 텍스트 토크나이저에 시간 토큰을 추가하는 아이디어는 이후 여러 비디오-언어 모델에 영향을 주었다.

### 4.2 향후 연구 시 고려할 점

1. **환각 및 사실적 정확성**: 시각적 근거 없는 이벤트 생성을 줄이기 위한 grounding 강화 기법이 필요하다.
2. **양방향 이벤트 관계 모델링**: 현재 자기회귀 디코딩은 이전 이벤트에만 조건부이므로, 양방향 문맥을 활용하는 방법을 고려해야 한다.
3. **평가 메트릭 개선**: 기존 IoU 기반 매칭 평가는 영상의 스토리 전개를 반영하지 못하므로, SODA_c 등 더 종합적인 메트릭 개발이 필요하다.
4. **실시간 처리**: 현재 모델은 오프라인 처리를 전제하므로, 스트리밍 비디오에 대한 적용을 위해 점진적(incremental) 생성 방법이 필요하다.
5. **프라이버시 및 편향**: 대규모 YouTube 비디오 사전학습에 따른 개인정보 이슈 및 데이터 편향 문제를 고려해야 한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 특징 | Vid2Seq와의 차이점 |
|------|------|---------|-----------------|
| **PDVC** (Wang et al.) [102] | 2021 (ICCV) | Parallel decoding, event counter 사용 | 태스크 특화 컴포넌트(event counter) 필요; Vid2Seq는 범용 시퀀스 생성 |
| **E2ESG** (Zhu et al.) [133] | 2022 (COLING) | 단일 시퀀스 생성, Wikihow 텍스트 사전학습 | 이벤트 위치를 음성 타임스탬프에서 직접 추론하여 음성이 없는 영상 처리 불가; 도메인 특화 텍스트 사전학습에 의존 |
| **UEDVC** (Zhang et al.) [125] | 2022 (ECCV) | 이벤트 검출/캡셔닝 통합, ActivityNet 사전학습 | 같은 데이터셋에서 사전학습; 시각 입력만 사용 |
| **MV-GPT** (Seo et al.) [80] | 2022 (CVPR) | 생성적 사전학습, 클립 레벨 캡셔닝 | 짧은 trimmed 클립 기반 사전학습; dense captioning 미지원 |
| **SwinBERT** (Lin et al.) [63] | 2022 (CVPR) | Sparse attention, 클립 레벨 캡셔닝 | 사전학습 없이 클립 캡셔닝에 특화; dense captioning 미지원 |
| **Flamingo** (Alayrac et al.) [4] | 2022 (NeurIPS) | 대규모 VLM, few-shot 학습 | 범용 VLM이지만 dense temporal grounding에 특화되지 않음 |
| **MERLOT Reserve** (Zellers et al.) [121] | 2022 (CVPR) | YT-Temporal-1B 데이터셋 제공, 멀티모달 스크립트 지식 모델 | Vid2Seq의 사전학습 데이터 소스; 비디오 이해에 초점 (생성 태스크 아님) |
| **Multimodal Pretraining for Dense Video Captioning** (Huang et al.) [35] | 2020 (AACL-IJCNLP) | 나레이션 비디오 사전학습, GT proposals 필요 | Localization을 다루지 않음; 도메인 특화 텍스트 사전학습 |

### 주요 비교 포인트

1. **2-stage vs. 1-stage**: PDVC, UEDVC 등은 여전히 localization과 captioning에 별도 모듈이나 태스크 특화 설계를 사용하는 반면, Vid2Seq는 완전히 통합된 시퀀스 생성 방식을 채택한다.

2. **사전학습 전략**: 기존 연구들은 trimmed 클립 기반 사전학습(MV-GPT, SwinBERT) 또는 도메인 특화 텍스트 사전학습(E2ESG)에 의존하지만, Vid2Seq는 **untrimmed 나레이션 비디오**를 시간 토큰과 함께 활용하여 도메인에 구애받지 않는 사전학습을 수행한다.

3. **스케일**: YT-Temporal-1B(18M 영상)이라는 대규모 다양한 코퍼스에서 사전학습하여, HowTo100M(1.2M 교육 영상)에 국한된 기존 연구보다 넓은 도메인 커버리지를 확보한다.

---

## 참고자료

1. **Yang, A., Nagrani, A., Seo, P. H., Miech, A., Pont-Tuset, J., Laptev, I., Sivic, J., & Schmid, C.** (2023). *Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning*. arXiv:2302.14115v2. (본 논문 원문)
2. **Vid2Seq 프로젝트 웹페이지**: https://antoyang.github.io/vid2seq.html
3. **Wang, T., Zhang, R., Lu, Z., Zheng, F., Cheng, R., & Luo, P.** (2021). *End-to-End Dense Video Captioning with Parallel Decoding*. ICCV 2021. [102]
4. **Zhu, W., Pang, B., Thapliyal, A., Wang, W. Y., & Soricut, R.** (2022). *End-to-End Dense Video Captioning as Sequence Generation*. COLING 2022. [133]
5. **Zhang, Q., Song, Y., & Jin, Q.** (2022). *Unifying Event Detection and Captioning as Sequence Generation via Pre-training*. ECCV 2022. [125]
6. **Seo, P. H., Nagrani, A., Arnab, A., & Schmid, C.** (2022). *End-to-End Generative Pretraining for Multimodal Video Captioning*. CVPR 2022. [80]
7. **Lin, K., Li, L., Lin, C.-C., Ahmed, F., Gan, Z., Liu, Z., Lu, Y., & Wang, L.** (2022). *SwinBERT: End-to-End Transformers with Sparse Attention for Video Captioning*. CVPR 2022. [63]
8. **Alayrac, J.-B., et al.** (2022). *Flamingo: A Visual Language Model for Few-Shot Learning*. NeurIPS 2022. [4]
9. **Zellers, R., et al.** (2022). *MERLOT Reserve: Neural Script Knowledge through Vision and Language and Sound*. CVPR 2022. [121]
10. **Huang, G., Pang, B., Zhu, Z., Rivera, C., & Soricut, R.** (2020). *Multimodal Pretraining for Dense Video Captioning*. AACL-IJCNLP 2020. [35]
11. **Raffel, C., et al.** (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR 2020. [78]
12. **Radford, A., et al.** (2021). *Learning Transferable Visual Models from Natural Language Supervision*. ICML 2021. [77]
