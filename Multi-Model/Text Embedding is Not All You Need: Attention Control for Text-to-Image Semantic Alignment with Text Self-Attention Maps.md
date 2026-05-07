# Text Embedding is Not All You Need: Attention Control for Text-to-Image Semantic Alignment with Text Self-Attention Maps

## 1. 핵심 주장 및 주요 기여 (간결한 요약)

이 논문의 **핵심 주장**은 텍스트 임베딩(text embedding)만으로는 Text-to-Image(T2I) 디퓨전 모델에서 의미적 정합성을 보장하기에 불충분하며, **텍스트 인코더 내부의 self-attention 맵**에 이미 풍부한 구문적 정보가 존재하지만 이것이 임베딩으로 충분히 전달되지 않는다는 것입니다.

**주요 기여**는 다음 네 가지입니다.

첫째, cross-attention 맵의 공간적 정렬이 텍스트 임베딩 간 유사도에 의해 결정된다는 점을 실험·수학적으로 증명하였습니다. 둘째, CLIP 텍스트 임베딩이 'bag-of-words'처럼 구문 관계를 제대로 반영하지 못함을 보였습니다. 셋째, 이러한 한계의 원인을 *attention sink* 현상(즉, `<bos>` 토큰으로 attention이 쏠리는 현상)으로 규명했습니다. 넷째, 외부 파서나 수동 토큰 인덱싱 없이 텍스트 self-attention 맵을 직접 cross-attention의 유사도 구조에 전이시키는 **T-SAM(Text Self-Attention Maps) guidance** 기법을 제안하였습니다.

---

## 2. 문제 정의, 방법론, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

T2I 디퓨전 모델은 종종 (i) **객체 누락(missing objects)** 과 (ii) **속성 잘못된 결합(attribute mis-binding)** 이라는 두 가지 의미적 불일치를 보입니다. 예: *"a black car and a white clock"* 프롬프트에서 시계가 검은색이 되거나, 시계 자체가 사라지는 경우.

### 2.2 진단: 두 가지 핵심 발견

**Finding 1**: 텍스트 임베딩의 코사인 유사도가 cross-attention 맵 유사도와 강하게 연관됩니다. 저자는 **Proposition 1**을 통해 cross-attention 유사도가 본질적으로 키 벡터(텍스트 임베딩)의 거리에 의해 결정됨을 수학적으로 증명했습니다.

$$
\cos(A_i^{(\ell,h)}, A_j^{(\ell,h)}) = \exp\left(-\frac{1}{2}(\mathbf{k}_i - \mathbf{k}_j)^\top W^2 (\mathbf{k}_i - \mathbf{k}_j)\right) + \mathcal{O}\left(\frac{1}{\sqrt{N_c}}\right) + \mathcal{O}(\epsilon)
$$

여기서 $W^2 := W_c^{(\ell,h)\top} \Sigma^{(\ell)} W_c^{(\ell,h)}$이며, $\Sigma^{(\ell)}$은 쿼리 벡터의 공분산 행렬입니다.

**Finding 2**: 그러나 텍스트 임베딩 유사도는 구문적 결합(syntactic binding)을 반영하지 못합니다. *attention sink* 현상이 그 원인이며, **Proposition 2**는 `<bos>` 토큰에 attention이 집중될 때 self-attention 출력 벡터들의 코사인 유사도가 거의 변하지 않음을 증명합니다.

$$
\cos(\mathbf{o}_i^{(\ell,h)}, \mathbf{o}_j^{(\ell,h)}) = 1 - \mathcal{O}(\epsilon)
$$

이는 self-attention 맵이 보유한 구문 정보가 임베딩 단계에서 희석된다는 의미입니다.

### 2.3 제안 방법: T-SAM Guidance

**핵심 아이디어**: cross-attention 맵 간 유사도 행렬 $S$를 텍스트 self-attention 행렬 $T$에 정렬시킵니다.

**(a) 텍스트 self-attention 행렬 (BOS·EOS 제거 후 재정규화)**:

$$
T_{ij} = \frac{T'_{ij}}{\sum_{m=2}^{i} T'_{im}}, \quad T' = \frac{1}{L_e H_e}\sum_{\ell=1}^{L_e}\sum_{h=1}^{H_e} T^{(\ell,h)}
$$

**(b) Cross-attention 유사도 행렬**:

$$
S_{ij} := \frac{C_{ij}}{\sum_{k=1}^{s} C_{ik}}, \quad C_{ij} := \frac{\sum_{a=1}^{N_c} A_{ai} A_{aj}}{\left(\sum_a A_{ai}^2\right)^{1/2}\left(\sum_a A_{aj}^2\right)^{1/2}}
$$

**(c) 최적화 손실 함수**:

$$
\mathcal{L}(z_t) = \sum_{i=1,\, j \leq i}^{s} \rho_i \, \left|T_{ij}^{\gamma} - S_{ij}(z_t)\right|, \quad \rho_i = i/s
$$

여기서 $\gamma$는 큰 값을 증폭하고 작은 값을 압축하는 온도(temperature) 제어 지수입니다.

**(d) 잠재 변수 갱신**:

$$
z'_t = z_t - \alpha \cdot \nabla_{z_t} \mathcal{L}(z_t)
$$

### 2.4 모델 구조

베이스 모델은 **Stable Diffusion v1.5**이며, 추가 학습 없이 추론 시점(test-time optimization)에만 개입합니다. 50 샘플링 스텝 중 1~25 스텝에서 $z_t$를 갱신하며, $M=256$, $\gamma=4$, TIFA에서는 $\alpha=40$, Attend-n-Excite 프롬프트에는 $\alpha=10$을 사용합니다.

### 2.5 성능 향상

TIFA 벤치마크(4,000 프롬프트)에서 TIFA 점수 0.79(SD) → **0.83(T-SAM)** 으로 향상되었으며, 외부 정보를 사용하는 Linguistic Binding(0.80)보다도 우수합니다. 구조화된 프롬프트(Objects, Animals-Objects)에서는 수동 토큰 선택이 필요한 CONFORM과 **거의 동등한 CLIP 유사도**를 달성하면서도 외부 입력이 필요 없습니다.

### 2.6 한계

(1) **CLIP 텍스트 인코더에 종속**: CLIP의 attention sink 특성에 기반하여 설계되었으므로, T5 등 다른 인코더(예: PixArt, SD3)에서는 동작이 다를 수 있습니다.  
(2) **추론 비용 증가**: 매 디노이징 스텝마다 추가 그래디언트 계산이 필요하여 latency가 늘어납니다.  
(3) **하이퍼파라미터 민감성**: $\alpha$, $\gamma$가 데이터셋 특성에 따라 다른 최적값을 가지므로 일반화에는 제약이 있습니다.  
(4) **CLIP 자체의 'bag-of-words' 한계**가 self-attention 맵에도 부분적으로 잔존할 가능성.

---

## 3. 일반화 성능 향상 가능성 (중점)

이 논문이 일반화 측면에서 갖는 가장 중요한 장점은 **외부 의존성 제거(Self-contained)** 와 **다양한 문장 구조에 대한 적용 가능성(Generalizable)** 입니다.

### 3.1 외부 자원 비의존성

기존 방법들은 일반화에 큰 제약이 있었습니다. Linguistic Binding(SynGen)은 SpaCy 같은 텍스트 파서가 (modifier, entity-noun) 쌍을 추출하지 못하면 SD와 동일한 결과를 산출합니다. Attend-and-Excite와 CONFORM은 사람이 직접 positive/negative 토큰 인덱스를 지정해야 하므로 정형화된 템플릿(예: *"[attr1][obj1] and [attr2][obj2]"*)을 벗어난 자유로운 프롬프트에는 확장이 어렵습니다.

T-SAM은 텍스트 인코더 내부에 *이미 존재하는* self-attention 맵을 재사용하므로, 어떤 프롬프트 구조든 추가 정보 없이 처리할 수 있습니다.

### 3.2 복잡한 구문 구조에 대한 적응력

부록 Figure A에서 저자는 *"A deep blue river flowed along the valley, whose banks were dotted with wildflowers"* 와 같이 **관계대명사(whose)** 를 포함한 복잡한 문장에서도 self-attention 맵이 *whose*가 *valley*에 강하게 attend함을 포착함을 보였습니다. 텍스트 임베딩 유사도에서는 이런 패턴이 거의 나타나지 않습니다. 이는 self-attention 맵이 단순 attribute-object 쌍을 넘어 **종속절·전치사구·동격 구조** 등 다양한 구문 관계를 자연스럽게 포착할 수 있음을 시사합니다.

### 3.3 TIFA 카테고리별 일반화

Figure 7에서 LB는 색·모양·재질에서만 향상되었으나, T-SAM은 색·모양·counting·activity·food·location 등 **거의 모든 질문 카테고리**에서 성능이 향상되었습니다. 이는 단순 "수식어-명사" 결합을 넘어 **동사적 관계, 수량 관계, 공간 관계**까지 포괄함을 의미합니다.

### 3.4 향후 일반화의 방향

(i) **다른 텍스트 인코더로 확장**: T5 기반 모델(SD3, PixArt-Σ, FLUX 등)에서 attention sink 패턴이 다르므로 식 (3)의 정규화 방식을 인코더별로 재정의할 수 있습니다.  
(ii) **다중 객체 및 장면 합성**: T-SAM의 smooth한 유사도 정렬은 CONFORM의 binary positive/negative보다 다객체 장면의 자연스러운 공간 배치에 유리할 가능성이 큽니다.  
(iii) **비디오·3D 생성으로의 이식**: 시간적 일관성이 필요한 영상 디퓨전에서도 텍스트 self-attention의 구문 정보는 보편적으로 활용 가능합니다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 미치는 영향

이 논문은 디퓨전 모델 분야에 다음 세 가지 패러다임 전환을 가져올 수 있습니다.

첫째, **"내부 신호 재활용(internal signal reuse)" 관점의 정립**입니다. 외부 파서나 인간 주석에 의존하던 의미 정렬 연구의 흐름을 모델 내부의 잠재 정보 활용으로 옮기는 계기가 될 수 있습니다. 둘째, **attention sink 현상에 대한 진단적 가치**를 제공하여, 이후 텍스트 인코더 자체를 attention sink 완화 방향으로 개선하려는 연구(예: bias 보정, sink-free attention)에 영감을 줄 것입니다. 셋째, **수학적 근거 제공**: Proposition 1, 2와 같은 명시적 분석은 attention 기반 정렬 기법들이 실험에 그치지 않고 이론적 토대를 갖추도록 유도합니다.

### 4.2 향후 연구 시 고려할 점

(1) **Self-attention 맵의 노이즈/편향**: CLIP 자체가 bag-of-words에 가깝다는 한계가 self-attention에도 잔존할 수 있으므로, BERT/T5 기반 인코더와의 비교 분석이 필요합니다.  
(2) **층별·헤드별 차별화**: 모든 layer/head를 단순 평균하는 대신, 구문 정보를 특히 잘 인코딩하는 특정 head를 선별적으로 활용할 수 있는지 탐구할 가치가 있습니다.  
(3) **추론 효율화**: gradient 기반 latent 갱신 대신, 직접 cross-attention의 키/쿼리에 가중치를 부여하는 closed-form 방법으로의 전환 가능성.  
(4) **평가 지표의 한계**: TIFA, CLIP score 모두 VLM 기반이라 자체 편향이 있으므로, 인간 평가나 EPViT 같은 미세 정합성 지표로의 검증이 필요합니다.  
(5) **DiT 계열 아키텍처(Diffusion Transformer)** 로 확장 시 cross-attention 구조가 다르거나 부재할 수 있어, 방법론 재설계가 요구됩니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법론 | 발표 | 외부 의존성 | 핵심 메커니즘 | 일반화 |
|---|---|---|---|---|
| **Prompt-to-Prompt** (Hertz et al., 2022) | arXiv 2022 | ✗ | Cross-attention 맵 직접 수정으로 편집 | 편집 중심, 정합성 ✗ |
| **StructureDiffusion** (Feng et al., 2022) | arXiv 2022 | ✓ (constituency parser) | 구문 트리로 텍스트 임베딩 분해 | 구문 트리 추출 가능한 경우만 |
| **Attend-and-Excite** (Chefer et al., 2023) | ACM TOG 2023 | ✓ (수동 토큰 선택) | 토큰별 최대 attention 값 강화 | 정형 템플릿 한정 |
| **A-STAR** (Agarwal et al., 2023) | ICCV 2023 | ✓ (수동) | Attention 분리(segregation)와 보존(retention) | 수동 그룹화 필요 |
| **SynGen / Linguistic Binding** (Rassin et al., 2024) | NeurIPS 2023/24 | ✓ (SpaCy 의존 구문 분석) | Modifier-entity 쌍 attention 정렬 | 속성-객체 결합에 한정 |
| **CONFORM** (Meral et al., 2024) | CVPR 2024 | ✓ (수동 그룹) | Contrastive loss로 positive/negative 분리 | 정형 템플릿 한정 |
| **Attention Regulation** (Zhang et al., 2024) | ECCV 2024 | ✗ | Attention dominance 완화 | 객체 누락에 초점 |
| **T-SAM** (Kim et al., 2024) | arXiv 2024 | ✗ | 텍스트 self-attention 맵을 cross-attention에 전이 | 자유 형식 프롬프트 |
| **VSC** (Dat et al., 2025) | ICCV 2025 | ✓ (segmentation) | 시각적 prototype 융합 + 분할 기반 학습 | 다중 결합에 강함 |

이 비교를 통해 T-SAM의 차별점이 분명해집니다. 외부 자원 없이(✗) 자유 형식 프롬프트에 적용 가능한 거의 유일한 방법이며, Attention Regulation(2024)이 "지배적 attention 완화"라는 단일 결함에 집중한 것과 달리 **객체 누락과 속성 결합 양쪽**을 단일 손실 함수로 다룬다는 점이 강점입니다. 다만 **VSC(ICCV 2025)** 와 같이 학습 기반 접근이 다중 결합 강건성에서 더 우수할 가능성이 있어, 학습 기반과 추론 시 최적화의 융합이 향후 흥미로운 방향입니다.

---

## 참고 자료

- 본 논문: Kim, J., Esmaeili, E., Qiu, Q. *"Text Embedding is Not All You Need: Attention Control for Text-to-Image Semantic Alignment with Text Self-Attention Maps."* arXiv:2411.15236, 2024. (사용자 업로드 PDF)
- arXiv 페이지: https://arxiv.org/abs/2411.15236
- Hertz et al. *"Prompt-to-Prompt Image Editing with Cross-Attention Control."* arXiv:2208.01626, 2022.
- Chefer et al. *"Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models."* ACM TOG 2023.
- Rassin et al. *"Linguistic Binding in Diffusion Models."* NeurIPS 2024 / arXiv:2306.08877. (https://arxiv.org/abs/2306.08877, https://research.nvidia.com/publication/2023-10_syntactic-binding-diffusion-models)
- Meral et al. *"CONFORM: Contrast is All You Need For High-Fidelity Text-to-Image Diffusion Models."* CVPR 2024 / arXiv:2312.06059. (https://arxiv.org/abs/2312.06059, https://conform-diffusion.github.io/)
- Zhang et al. *"Enhancing Semantic Fidelity in Text-to-Image Synthesis: Attention Regulation in Diffusion Models."* ECCV 2024 / arXiv:2403.06381. (https://arxiv.org/abs/2403.06381)
- Feng et al. *"Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis."* arXiv:2212.05032, 2022.
- Agarwal et al. *"A-STAR: Test-Time Attention Segregation and Retention."* ICCV 2023.
- Rombach et al. *"High-Resolution Image Synthesis with Latent Diffusion Models."* CVPR 2022.
- Hu et al. *"TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering."* ICCV 2023.
- Xiao et al. *"Efficient Streaming Language Models with Attention Sinks."* arXiv:2309.17453, 2023.
- Yuksekgonul et al. *"When and Why Vision-Language Models Behave Like Bags-of-Words."* arXiv:2210.01936, 2022.
- ICCV 2025 VSC: *"Visual Search Compositional Text-to-Image Diffusion Model."* (https://openaccess.thecvf.com/content/ICCV2025/html/Dat_VSC_Visual_Search_Compositional_Text-to-Image_Diffusion_Model_ICCV_2025_paper.html)

> **정확성 안내**: 위 분석 중 본 논문 내용(섹션 1, 2, 3, 4의 핵심 설명·수식·실험 수치)은 업로드된 PDF에 직접 근거합니다. 비교 표와 관련 연구 부분은 검색을 통해 확인된 공식 출판 정보에 기반하였으나, T-SAM이 후속 학습 기반 방법(예: VSC)과 직접 동일 벤치마크에서 비교된 공식 결과는 아직 확인되지 않았으므로 "가능성"·"강점 추정" 차원으로 기술하였습니다.
