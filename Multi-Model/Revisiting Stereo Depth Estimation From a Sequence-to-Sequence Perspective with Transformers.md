# Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective with Transformers

---

## 1. 핵심 주장 및 주요 기여 요약

스테레오 깊이 추정은 좌우 이미지의 에피폴라 라인(epipolar line) 상에서 픽셀 간 최적 대응(correspondence)을 찾아 깊이를 추론하는 문제이며, 본 논문은 이를 시퀀스-투-시퀀스(sequence-to-sequence) 대응 관점에서 재조명하여 기존의 비용 볼륨(cost volume) 구성을 위치 정보와 어텐션 기반의 밀집 픽셀 매칭으로 대체하였다.

제안된 STereo TRansformer(STTR)는 다음의 핵심 이점을 갖는다: 1) 고정된 디스패리티 범위의 제약을 완화하고, 2) 가려진(occluded) 영역을 식별하면서 신뢰도 추정값을 제공하며, 3) 매칭 과정에서 유일성 제약(uniqueness constraint)을 부과한다.

합성 데이터셋과 실세계 데이터셋 모두에서 유망한 결과를 보고하며, STTR이 파인튜닝 없이도 서로 다른 도메인에 걸쳐 일반화됨을 시연하였다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존의 많은 스테레오 깊이 추정 방법들은 사전에 지정된 제한된 디스패리티 범위(fixed disparity range)에 의해 제약을 받아, 가까운 물체나 다양한 카메라 설정에서 성능이 저하되었다. 기하학적 속성인 오클루전 처리 및 유일성 제약이 기존 학습 기반 접근법에서 종종 결여되어 있어, 부정확한 디스패리티 추론과 신뢰도 추정 부재로 이어졌다.

핵심적으로 세 가지 문제를 지적한다:
1. **고정 디스패리티 범위**: 비용 볼륨 기반 방법은 최대 디스패리티 $D_{\max}$를 미리 설정해야 함
2. **오클루전 처리 부재**: 좌영상 픽셀 중 우영상에 대응이 없는 영역 감지 불가
3. **유일성 제약 미적용**: 하나의 좌 픽셀이 하나의 우 픽셀에만 대응되어야 하는 기하학적 제약 미활용

### 2.2 제안하는 방법 (수식 포함)

#### (a) 특징 추출기 (Feature Extractor)

네트워크는 기존의 CNN 특징 추출기와 장거리 관계 포착 모듈인 Transformer를 결합하며, 잔차 연결(residual connections)과 공간 피라미드 풀링(Spatial Pyramid Pooling)을 갖춘 모래시계(hourglass) 형태의 특징 추출기를 사용한다.

좌우 이미지 $I_L, I_R$에서 공유된 특징 추출기를 통해 특징 맵을 추출한다:

$$f_L = \text{FeatureExtractor}(I_L), \quad f_R = \text{FeatureExtractor}(I_R)$$

#### (b) Transformer: Self-Attention 및 Cross-Attention 교대 구조

Transformer 아키텍처는 교대 어텐션 메커니즘을 사용한다. 동일 이미지 내 에피폴라 라인을 따라 픽셀 간 어텐션을 계산하는 Self-Attention과, 좌우 이미지의 대응 에피폴라 라인 간 어텐션을 계산하는 Cross-Attention을 교대로 적용하여, 이미지 문맥과 상대적 위치에 기반하여 특징 디스크립터를 지속적으로 갱신한다.

**Multi-Head Attention** 수식:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

각 어텐션 헤드 $h$에 대해 선형 프로젝션으로 쿼리, 키, 값을 계산한다:

$$Q_h = f \cdot W_h^Q, \quad K_h = f \cdot W_h^K, \quad V_h = f \cdot W_h^V$$

여기서 $d_k = C / N_h$이고, $C$는 특징 차원, $N_h$는 헤드 수이다.

- **Self-Attention**: 같은 이미지 내 에피폴라 라인을 따라 계산

$$\hat{f}_L = \text{SelfAttn}(f_L, f_L, f_L), \quad \hat{f}_R = \text{SelfAttn}(f_R, f_R, f_R)$$

- **Cross-Attention**: 좌우 이미지 간 에피폴라 라인에서 교차 계산

$$\tilde{f}_L = \text{CrossAttn}(\hat{f}_L, \hat{f}_R, \hat{f}_R), \quad \tilde{f}_R = \text{CrossAttn}(\hat{f}_R, \hat{f}_L, \hat{f}_L)$$

#### (c) 상대적 위치 인코딩 (Relative Positional Encoding, RPE)

상대적 위치 인코딩을 통해 위치 정보를 제공하며, 이를 통해 STTR은 특징이 부족한 픽셀로부터 지배적 픽셀(예: 엣지)까지의 상대적 거리를 활용하여 모호성을 해결한다.

어텐션 계산에 상대적 위치 편향 $B$를 추가한다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + B}{\sqrt{d_k}}\right)V$$

#### (d) Optimal Transport를 통한 유일성 제약

엔트로피 정규화된 최적 수송(entropy-regularized optimal transport)을 소프트 할당(soft assignment)에 적용하여 스테레오 매칭의 유연성과 미분가능성을 보장한다.

좌우 특징 벡터 간 비용 행렬 $C$를 계산하고, Sinkhorn 알고리즘을 통해 최적 수송 문제를 반복적으로 풀어 할당 행렬 $T^*$를 구한다:

$$T^* = \arg\min_{T \in \mathcal{U}(a, b)} \langle T, C \rangle - \epsilon H(T)$$

여기서 $\mathcal{U}(a, b)$는 수송 다면체(transport polytope), $H(T) = -\sum_{i,j} T_{i,j} \log T_{i,j}$는 엔트로피, $\epsilon$은 정규화 계수이다.

오클루전 처리를 위해 더스트빈(dustbin) 행/열을 학습 가능한 파라미터 $\phi$로 비용 행렬에 추가한다:

$$\bar{C} = \begin{bmatrix} C & \phi \cdot \mathbf{1} \\ \phi \cdot \mathbf{1}^T & 0 \end{bmatrix}$$

디스패리티는 할당 행렬의 가중 합으로 회귀한다:

$$\hat{d}_i = \sum_{j} T^*_{i,j} \cdot j$$

#### (e) Context Adjustment Layer (CAL)

문맥 조정 레이어는 컨볼루션 블록과 시그모이드 활성화를 사용한 오클루전 정제(상단)와 장거리 스킵 연결을 갖춘 잔차 블록을 사용한 디스패리티 정제(하단)로 구성된다.

$$\hat{o} = \sigma(\text{Conv}([\tilde{f}_L; d_{\text{raw}}; o_{\text{raw}}]))$$

$$\hat{d} = d_{\text{raw}} + \text{ResBlock}([\tilde{f}_L; d_{\text{raw}}; \hat{o}])$$

### 2.3 모델 구조 개요

전체 파이프라인은 다음과 같다:

```
Input: (I_L, I_R)
    ↓
[공유 CNN 특징 추출기 (Hourglass + SPP)]
    ↓
(f_L, f_R) → [Transformer: Self-Attn ↔ Cross-Attn 교대 × N layers]
    ↓          (+ Relative Positional Encoding)
[최종 Cross-Attention + Optimal Transport + Attention Mask]
    ↓
(d_raw, o_raw) → [Context Adjustment Layer]
    ↓
Output: (d_refined, o_refined)
```

STTR은 그래디언트 체크포인팅과 혼합 정밀도 훈련을 활용한 메모리 효율적 구현을 채택하여, 일반 하드웨어에서의 훈련을 가능하게 하고 효율성을 위한 어텐션 스트라이드를 조절한다.

### 2.4 성능 향상

전체 모델은 모든 구성 요소를 통합하여 Scene Flow 데이터셋에서 3px 에러 1.26, EPE 0.45, 오클루전 IOU 0.92의 최고 성능을 달성하며, 이러한 아키텍처 요소들의 시너지 효과를 입증하였다.

**Ablation Study 핵심 결과:**

| 구성 요소 | 3px Error | EPE | Occ IOU |
|---|---|---|---|
| Baseline (AM only) | 2.77 | 0.84 | — |
| + Optimal Transport | — | — | 0.87 |
| Full Model (AM+OT+CAL+RPE) | **1.26** | **0.45** | **0.92** |

최적 수송 레이어는 매칭 시 소프트 유일성 제약을 부과하여 오클루전 감지를 개선하는 데 결정적이며, 어텐션 마스크에 추가 시 오클루전 IOU가 0.77에서 0.87로 크게 증가하였다.

STTR은 Scene Flow와 KITTI 2015에서 정제(refinement)를 적용한 기존 연구들과 비교하여 비슷한 수준의 성능을 보였다.

### 2.5 한계

STTR 모델은 우수한 장점에도 불구하고, 특히 실시간 응용에 있어 상당한 계산 비용을 요구한다.

주요 한계점:
1. **계산 복잡도**: Transformer의 어텐션은 $O(N^2)$ 복잡도를 가지며, 고해상도 이미지에서 메모리와 시간 소모가 큼
2. **다운샘플링 의존성**: 메모리 제약으로 인한 어텐션 스트라이드 적용이 세밀한 디스패리티 추정을 저해
3. **교차 에피폴라 정보 부재**: Transformer가 에피폴라 라인 단위로만 동작하여 수직 방향의 전역 정보 활용이 제한적

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화 성능의 실증적 증거

STTR은 합성 데이터(Scene Flow)만으로 훈련하더라도 MPI Sintel, KITTI 2015, Middlebury 2014, SCARED 등 다양한 실세계 데이터셋으로 일반화할 수 있었다.

### 3.2 일반화를 가능하게 하는 메커니즘

**(1) 암묵적 특징 클러스터링:**

STTR은 Scene Flow 데이터로만 훈련된 차원 축소 임베딩을 통해 특징 맵을 시각화한 결과, 도메인에 상관없이 학습된 표현이 텍스처가 있는 영역, 텍스처가 없는 영역, 그리고 오클루전 영역으로 클러스터링됨을 관찰하였다. 이러한 암묵적으로 학습된 특징 클러스터링이 STTR의 일반화를 향상시키고 Transformer 매칭 과정을 보다 용이하게 만든다고 가설을 세웠다.

**(2) 상대적 위치 인코딩의 역할:**

상대적 위치 인코딩(RPE)은 위치 정보를 제공하여, STTR이 특징이 부족한 픽셀에서 엣지 같은 지배적 픽셀까지의 상대적 거리를 활용하여 모호성을 해결하게 한다.

이는 절대적 위치가 아닌 상대적 관계에 기반하므로, 이미지 해상도나 도메인이 변경되어도 일관성을 유지한다:

$$B_{i,j} = \text{RPE}(i - j)$$

**(3) 고정 디스패리티 범위 제거:**

기존 비용 볼륨 방식은 $D_{\max}$에 의존하여 도메인 변경 시 디스패리티 범위 밖의 대응을 처리할 수 없지만, STTR은 시퀀스 전체에 대해 어텐션을 계산하므로 이론적으로 **무제한 디스패리티 범위**를 처리할 수 있다.

**(4) Optimal Transport의 기하학적 제약:**

유일성 제약을 통해 물리적으로 타당한 매칭만 허용하므로, 새로운 도메인에서도 기하학적으로 일관된 결과를 생성한다.

### 3.3 일반화 성능 향상 방향

후속 연구인 CEST에서는 Context Enhanced Path(CEP)를 제안하여 장거리 전역 정보를 포착함으로써, 기존 솔루션의 일반화 및 견고성을 공통 실패 사례에 대해 개선하였다.

향상 가능성:
- 교차 에피폴라(cross-epipolar) 정보의 통합으로 전역 문맥 강화
- 대규모 사전훈련 (예: DINOv2, Depth Anything V2 등의 Foundation Model 활용)
- 더 효율적인 어텐션 메커니즘 (Linear Attention, Window Attention 등)으로 고해상도 처리 지원

---

## 4. 연구 영향 및 향후 고려 사항

### 4.1 연구 영향

STTR은 스테레오 매칭 분야에서 **패러다임 전환**을 이끈 선구적 연구로 평가받는다:

STTR은 이 분야의 선구적 연구 중 하나로, 전통적인 비용 볼륨 구성과 달리 고정 디스패리티 범위 제약을 완화하고 에피폴라 라인을 따라 Self-Attention과 좌우 쌍 간 Cross-Attention을 교대로 적용하는 Transformer 아키텍처를 도입하여, 장거리 픽셀 연관성을 포착하고 매칭 영역의 모호성을 해결할 수 있게 하였다.

구체적 영향:
1. **Transformer 기반 스테레오 매칭의 가능성 입증** → 후속 다수의 Transformer 기반 방법 촉발
2. **오클루전 감지와 디스패리티 추정의 통합** → end-to-end 학습의 새로운 방향 제시
3. **제로샷 일반화** 가능성 실증 → 도메인 적응 연구 활성화

후속 연구인 STTR-3D는 STTR을 3D로 확장하여 비디오 기반 디스패리티 변화 추정을 수행하며, MLP 기반 상대적 위치 인코딩 정제와 엔트로피 정규화된 최적 수송을 통해 디스패리티 변화 맵을 획득하였다.

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|---|---|
| **계산 효율성** | $O(N^2)$ 어텐션을 선형 근사 또는 윈도우 기반으로 대체 |
| **고해상도 지원** | Coarse-to-Fine 전략 또는 계층적 매칭 도입 |
| **Foundation Model 연계** | DINOv2, Depth Anything V2 등 사전훈련 모델의 표현력 활용 |
| **실시간 추론** | 경량화 (STTR-light) 및 모델 가지치기, 양자화 적용 |
| **자기지도 학습** | 라벨 없는 실세계 데이터 활용을 위한 비지도/자기지도 학습 전략 |
| **멀티태스크 통합** | 광학 흐름, 깊이, 장면 흐름의 통합 프레임워크 구축 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 학회 | 핵심 접근법 | STTR 대비 차별점 |
|---|---|---|---|---|
| **RAFT-Stereo** | 2021 | 3DV | 반복적 갱신 (Iterative GRU) + All-pairs 상관 | GRU 기반 반복 디스패리티 정제, 경량 구조 |
| **CREStereo** | 2022 | CVPR | 계층적 반복 정제 + 적응적 상관 | 계층적 네트워크와 반복 정제를 통해 coarse-to-fine 방식으로 디스패리티를 갱신 |
| **CEST** | 2022 | ECCV | STTR + Context Enhanced Path | CEP를 STTR에 플러그인하여 장거리 전역 정보를 포착함으로써 일반화와 견고성을 개선 |
| **GMStereo** | 2023 | TPAMI | 흐름(Flow), 스테레오, 깊이 추정을 통합하는 Transformer 기반 통합 프레임워크 | 단일 모델로 다중 태스크 수행 |
| **IGEV-Stereo** | 2023 | CVPR | 기하학 인코딩 볼륨(Geometry Encoding Volume)을 구축하여 공간적 단서를 통합하고 기하학 정보를 인코딩한 뒤, 반복적으로 인덱싱하여 디스패리티 맵을 갱신 | 필터링 기반과 반복 최적화 기반의 장점 결합 |
| **IGEV++** | 2025 | TPAMI | 적응적 패치 매칭 모듈로 대규모 디스패리티 범위의 매칭 비용을 효율적으로 계산하고, 선택적 기하학 특징 융합 모듈로 다중 범위/다중 세분도 기하학 특징을 적응적으로 융합한 뒤 ConvGRU로 반복 갱신 | Middlebury, ETH3D, KITTI 2012, KITTI 2015 네 개 벤치마크에서 SOTA 달성 |
| **GOAT** | 2024 | WACV | 전역 오클루전 인식 Transformer를 활용한 견고한 스테레오 매칭 | 오클루전 처리에 특화된 전역 어텐션 |
| **Selective-Stereo** | 2024 | CVPR | 적응적 주파수 정보 선택을 통한 스테레오 매칭 | 주파수 도메인에서의 선택적 정보 처리 |
| **DEFOM-Stereo** | 2025 | CVPR | 깊이 기반 모델(Depth Foundation Model)과 결합하여, DINOv2 백본의 사전훈련된 ViT를 활용한 제로샷 일반화 능력을 스테레오 매칭에 적용 | Foundation Model 시대의 스테레오 매칭 |

### 핵심 트렌드 분석

STTR 이후의 연구 흐름은 크게 세 가지 방향으로 전개되고 있다:

1. **반복 최적화(Iterative Optimization) 패러다임의 부상**: RAFT-Stereo → CREStereo → IGEV → IGEV++로 이어지는 GRU 기반 반복 갱신이 주류가 됨. MonSter, DEFOM-Stereo, IGEV, CREStereo, RAFT-Stereo 등 반복 최적화 기반 스테레오 매칭 모델들이 GRU(Gate Recurrent Units) 메커니즘을 활용하여 다수의 벤치마크에서 지배적 위치를 차지하고 있다.

2. **Foundation Model과의 결합**: 일반화와 견고성은 여전히 도전적인 과제로 남아 있으며, 오클루전, 텍스처 부재, 이미지 블러, 고해상도 등이 핵심 도전 과제이다. 이에 DINOv2, Depth Anything V2 등의 대규모 사전훈련 모델을 활용하는 추세가 부상하고 있다.

3. **효율성과 정확도의 균형**: RT-IGEV와 같은 실시간 버전이 등장하여, 실시간 추론을 달성하면서도 모든 공개된 실시간 방법 중 최고 성능을 제공하고 있다.

---

## 참고자료

1. **Li, Z., Liu, X., Drenkow, N., Ding, A., Creighton, F. X., Taylor, R. H., & Unberath, M.** (2021). *Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective With Transformers*. ICCV 2021. [arXiv:2011.02910](https://arxiv.org/abs/2011.02910)
2. **STTR GitHub Repository**: [github.com/mli0603/stereo-transformer](https://github.com/mli0603/stereo-transformer)
3. **ICCV 2021 Open Access**: [thecvf.com/ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Revisiting_Stereo_Depth_Estimation_From_a_Sequence-to-Sequence_Perspective_With_Transformers_ICCV_2021_paper.html)
4. **IEEE Xplore**: [ieeexplore.ieee.org/document/9711118](https://ieeexplore.ieee.org/document/9711118/)
5. **Liner Quick Review**: [liner.com/review/revisiting-stereo-depth-estimation](https://liner.com/review/revisiting-stereo-depth-estimation-from-sequencetosequence-perspective-with-transformers)
6. **Xu, G. et al.** (2023/2025). *IGEV++: Iterative Multi-range Geometry Encoding Volumes for Stereo Matching*. CVPR 2023 / TPAMI 2025. [arXiv:2409.00638](https://arxiv.org/html/2409.00638)
7. **Xu, H. et al.** (2023). *Unifying Flow, Stereo and Depth Estimation (GMStereo)*. TPAMI 2023. [github.com/autonomousvision/unimatch](https://github.com/autonomousvision/unimatch)
8. **Drenkow, N. et al.** (2022). *Context-Enhanced Stereo Transformer*. ECCV 2022. [ecva.net/papers/eccv_2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920263.pdf)
9. **Jiang, H. et al.** (2025). *DEFOM-Stereo: Depth Foundation Model Based Stereo Matching*. CVPR 2025.
10. **Awesome Deep Stereo Matching**: [github.com/fabiotosi92/Awesome-Deep-Stereo-Matching](https://github.com/fabiotosi92/Awesome-Deep-Stereo-Matching)
11. **LearnOpenCV - ADAS Stereo Vision**: [learnopencv.com/adas-stereo-vision](https://learnopencv.com/adas-stereo-vision/)
12. **ResearchGate - STTR-3D**: [researchgate.net/publication/366996486](https://www.researchgate.net/publication/366996486)
