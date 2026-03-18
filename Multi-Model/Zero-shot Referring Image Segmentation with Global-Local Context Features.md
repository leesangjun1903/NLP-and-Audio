# Zero-shot Referring Image Segmentation with Global-Local Context Features

**저자:** Seonghoon Yu, Paul Hongsuck Seo, Jeany Son (GIST AI Graduate School & Google Research)
**학회:** CVPR 2023, pp. 19456–19465

---

## 1. 핵심 주장 및 주요 기여 (Summary)

Referring Image Segmentation (RIS)은 입력 이미지의 특정 영역에 대응하는 참조 표현(referring expression)이 주어졌을 때 세그멘테이션 마스크를 찾는 것을 목표로 하지만, 이를 위한 레이블 데이터셋 수집은 매우 비용이 높고 노동집약적이다.

이 문제를 극복하기 위해 저자들은 CLIP의 사전학습된 크로스모달(cross-modal) 지식을 활용한 간단하면서도 효과적인 제로샷 RIS 방법을 제안한다.

**주요 기여는 세 가지이다:**

1. **Mask-guided Visual Encoder:** 입력 텍스트에 기반한 세그멘테이션 마스크를 얻기 위해, 입력 이미지의 글로벌 및 로컬 문맥 정보를 포착하는 마스크 기반 시각 인코더를 제안한다.

2. **Global-Local Text Encoder:** 글로벌 특징은 전체 입력 표현의 복잡한 문장 수준 의미를 포착하고, 로컬 특징은 의존성 파서(dependency parser)로 추출된 목표 명사구에 집중하는 글로벌-로컬 텍스트 인코더를 도입한다.

3. **SOTA 성능:** 실험에서 제안된 방법은 여러 제로샷 기준선(baseline)뿐만 아니라 약지도(weakly supervised) RES 방법까지도 상당한 차이로 능가한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

RIS는 참조 표현이 주어졌을 때 해당 이미지 영역의 세그멘테이션 마스크를 찾는 과제이며, 레이블 데이터셋 수집이 매우 비용이 높고 노동 집약적이다. 기존 완전 지도학습(fully-supervised) RIS 방법은 대규모 (이미지, 텍스트, 마스크) 쌍 데이터를 필요로 하므로, 이를 대체할 제로샷 접근이 절실하다.

또한, CLIP의 원래 시각 특징은 이미지 전체를 하나의 단일 특징 벡터로 설명하도록 설계되어 있어, 인스턴스 수준의 세분화된 그라운딩에는 부적합하다는 근본적 문제가 있다.

### 2.2 제안하는 방법 (Global-Local CLIP)

#### (A) 전체 프레임워크 개요

전체 프레임워크(Global-Local CLIP)는 이미지와 표현이 입력으로 주어지면, 마스크 제안(proposal)을 이용하여 글로벌-로컬 문맥 시각 특징을 추출하고, 글로벌-로컬 문맥 텍스트 특징도 추출한 후, 모든 글로벌-로컬 문맥 시각 특징과 텍스트 특징 간 코사인 유사도를 계산하여 가장 높은 점수의 마스크를 선택한다.

#### (B) 마스크 제안 생성 (Mask Proposal)

비지도 인스턴스 세그멘테이션 기법으로 얻은 인스턴스 마스크를 활용하여 세밀한 인스턴스 수준의 그라운딩을 수행한다. 구체적으로 FreeSOLO(비지도 인스턴스 세그멘테이션 모델)를 사용하며, 후속 연구에서는 SAM도 활용된다.

#### (C) Global-context Visual Features (글로벌 문맥 시각 특징)

ResNet과 ViT 두 가지 아키텍처를 시각 인코더로 사용하며 (후자가 약간 더 좋은 성능을 보임), CLIP과 유사하다. ResNet의 경우 Masked Attention Pooling, ViT의 경우 Token Masking 방식으로 마스크 기반 글로벌 문맥 시각 인코더를 구현한다.

**ViT 기반 Token Masking:**

마스크 $M_i$가 주어지면, ViT 패치 토큰 중 마스크 외부의 토큰을 마스킹하여 글로벌 문맥을 유지하면서 특정 인스턴스에 주의를 집중시킨다. 글로벌 문맥 시각 특징은 다음과 같이 정의된다:

$$f_{\text{global}}^{(i)} = \text{CLIP}_{\text{vis}}(I, M_i)$$

여기서 $I$는 원본 이미지, $M_i$는 $i$번째 마스크 제안이다. CLIP 비전 인코더의 self-attention 레이어에서 [CLS] 토큰이 마스크 영역 내부의 패치 토큰에만 주의를 기울이도록 제약하되, 패치 토큰 간에는 전체 이미지에 대한 상호 어텐션을 유지한다.

**ResNet 기반 Masked Attention Pooling:**

$$f_{\text{global}}^{(i)} = \frac{\sum_{(h,w)} M_i^{(h,w)} \cdot F^{(h,w)}}{\sum_{(h,w)} M_i^{(h,w)}}$$

여기서 $F$는 ResNet에서 추출한 특징 맵, $M_i^{(h,w)}$는 마스크의 $(h,w)$ 위치 값이다.

#### (D) Local-context Visual Features (로컬 문맥 시각 특징)

제로샷 시맨틱 세그멘테이션에서 흔히 사용되는 크롭(cropping) 연산을 적용하여 관련 없는 영역을 제거하고, 마스크 부분에 집중하도록 한다.

마스크 $M_i$의 바운딩 박스 영역을 크롭하여 CLIP에 입력한다:

$$f_{\text{local}}^{(i)} = \text{CLIP}_{\text{vis}}(\text{Crop}(I, \text{BBox}(M_i)))$$

#### (E) Global-Local Context Visual Features

글로벌-로컬 문맥 시각 특징은 글로벌 문맥 특징과 로컬 문맥 시각 특징을 결합하여 계산된다:

$$f_{\text{vis}}^{(i)} = \alpha \cdot f_{\text{global}}^{(i)} + (1-\alpha) \cdot f_{\text{local}}^{(i)}$$

여기서 $\alpha$는 글로벌과 로컬 특징 간의 가중치 균형 하이퍼파라미터이다.

#### (F) Global-Local Context Textual Features

CLIP이 이미지 수준 표현만 다루는 한계를 극복하기 위해, spaCy 의존성 파서를 사용하여 텍스트에서 목표 명사(target noun)를 추출하여 로컬 문맥 텍스트 특징을 구성한다.

```math
f_{\text{text}} = \beta \cdot f_{\text{text\_global}} + (1-\beta) \cdot f_{\text{text\_local}}
```

여기서:
- $f_{\text{text global}} = \text{CLIP}_{\text{text}}(T)$ : 전체 참조 표현 $T$의 문장 수준 특징
- $f_{\text{text local}} = \text{CLIP}\_{\text{text}}(T_{\text{noun}})$ : 의존성 파서로 추출한 목표 명사구 $T_{\text{noun}}$의 특징
- $\beta$는 텍스트 글로벌-로컬 가중치 하이퍼파라미터

#### (G) 최종 마스크 선택

모든 마스크 제안 $\{M_1, M_2, ..., M_N\}$에 대해 코사인 유사도를 계산하고 최고 점수의 마스크를 선택한다:

$$M^* = \arg\max_{M_i} \text{cos}(f_{\text{vis}}^{(i)},\ f_{\text{text}})$$

$$\text{cos}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$$

### 2.3 모델 구조 요약

| 구성 요소 | 세부 내용 |
|---|---|
| **Mask Proposal** | FreeSOLO (비지도) / SAM (후속 실험) |
| **Visual Encoder** | CLIP ViT-B/32 또는 ResNet-50 (마스크 기반 수정) |
| **Text Encoder** | CLIP Text Encoder + spaCy 의존성 파서 |
| **특징 결합** | 글로벌-로컬 가중합 (시각 + 텍스트 모두) |
| **마스크 선택** | 코사인 유사도 기반 argmax |

### 2.4 성능 향상

제안된 방법은 여러 제로샷 기준선뿐만 아니라 약지도 참조 표현 세그멘테이션 방법까지도 상당한 마진으로 능가하는 성능을 달성한다.

주요 벤치마크(RefCOCO, RefCOCO+, RefCOCOg)에서의 oIoU 성능이 기존 제로샷 방법 대비 크게 향상되었으며, Global-and-local CLIP (GL-CLIP)은 텍스트 입력으로 제로샷 전이 방식으로 인스턴스를 세그멘테이션하는 최초의 방법으로 제안되었다.

### 2.5 한계

논문 및 후속 연구에서 지적된 주요 한계점은 다음과 같다:

1. **공간적 관계 인식 부족:** 기존 CLIP 기반 모델(GL-CLIP 포함)은 객체의 상대적 공간 관계를 식별하는 능력이 현저히 감소하며, 이는 마스크된 영역별로 유사도를 평가하여 텍스트 입력의 직접적 위치 단서에 대한 민감도가 떨어지기 때문이다.

2. **마스크 제안의 품질 의존성:** 마스크 생성 부분이 더 좋은 성능을 보이고 정답(ground truth)에 가까운 결과를 생성한다면, 이 모델의 성능도 확실히 개선될 것이다.

3. **문맥 토큰 활용 부족:** 대부분의 기존 방법은 로컬 텍스트 특징 추출 시 주요 명사구에만 집중하여 문맥 토큰을 약화시키는 경향이 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문의 핵심 강점 중 하나는 **제로샷 일반화(zero-shot generalization)** 능력이다:

### 3.1 제로샷 전이의 본질적 강점

CLIP의 대규모 이미지-텍스트 사전학습 지식을 활용하므로, 특정 RIS 데이터셋에 대한 학습 없이도 다양한 도메인에 적용할 수 있다. 이는 기존 완전 지도학습 방법이 학습 도메인에 과적합(overfit)되는 문제를 근본적으로 해결한다.

### 3.2 비학습(Unseen) 도메인 성능

GL-CLIP은 비학습 도메인(unseen domain)에서 지도학습 방법보다 더 좋은 성능을 보이며, 퓨샷(few-shot) 학습 설정에서도 k가 작을 때 큰 마진으로 기존 방법을 능가한다.

### 3.3 일반화 성능 향상을 위한 핵심 설계 요소

| 설계 요소 | 일반화 기여 |
|---|---|
| **글로벌 문맥** | 장면의 전체적 구조와 객체 간 관계를 이해하여 새로운 장면에서도 올바른 참조 해석 가능 |
| **로컬 문맥** | 특정 객체의 세밀한 시각적 특성에 집중하여 도메인 독립적 객체 인식 |
| **텍스트 글로벌-로컬** | 복잡한 문장 의미와 핵심 명사를 동시 활용하여 다양한 표현 스타일에 대한 강건성 |
| **사전학습 CLIP 활용** | 추가 미세조정 없이 CLIP의 범용 크로스모달 지식 활용 |

### 3.4 향후 일반화 향상 방향

- 더 강력한 마스크 제안 생성기(예: SAM) 도입 시 성능 상한이 크게 개선됨
- 공간적 관계 추론 능력을 보강하면 위치 기반 표현에 대한 일반화 가능
- 다양한 CLIP 변형(DFN ViT-H/14 등) 사용으로 시각적 표현 품질 개선 가능

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 연구 영향

1. **제로샷 RIS 분야의 개척:** GL-CLIP은 텍스트 입력으로 제로샷 전이 방식의 인스턴스 세그멘테이션을 최초로 제안한 방법으로, 이후 수많은 후속 연구의 기반이 되었다.

2. **글로벌-로컬 패러다임 확산:** 시각과 텍스트 양쪽 모두에서 글로벌-로컬 특징을 결합하는 아이디어는 이후 연구에서 널리 채택되었다.

3. **실용적 접근:** 추가 학습 없이 기존 사전학습 모델을 조합하는 training-free 패러다임의 가능성을 입증하였다.

### 4.2 후속 연구 시 고려할 점

1. **공간 관계 인식 강화:** 이들 방법은 "left", "right" 등의 위치 정보가 포함된 텍스트 입력에서 항상 오작동하는 문제가 있으므로, Grad-CAM 또는 patch-level 특징을 활용한 공간 인식 강화가 필요하다.

2. **마스크 제안의 품질:** 마스크 생성기의 성능이 전체 파이프라인의 상한을 결정하므로, SAM 등 고품질 마스크 생성기와의 통합이 중요하다.

3. **텍스트 표현의 정교화:** 단순한 명사구 추출을 넘어 문맥 토큰의 공간적 단서까지 활용하는 더 정교한 텍스트 처리가 필요하다.

4. **생성 모델과의 결합 가능성:** 생성 모델과 판별 모델을 결합하면 기존 방법을 상당한 마진으로 능가할 수 있으며, 생성 모델이 이 과업에 유익하고 판별 모델을 보완할 수 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도/학회 | 핵심 접근 | GL-CLIP 대비 차별점 |
|---|---|---|---|
| **GL-CLIP (Yu et al.)** | CVPR 2023 | CLIP + FreeSOLO, 글로벌-로컬 시각/텍스트 특징 | 제로샷 RIS 최초 제안 |
| **TAS (Suo et al.)** | EMNLP 2023 Findings | 마스크 제안 네트워크 + 텍스트 증강 시각-텍스트 매칭 점수(P-score, N-score) | surrogate captioning 모델로 시각-텍스트 도메인 갭 해소 |
| **Ref-Diff (Ni et al.)** | arXiv 2023 | Stable Diffusion 등 생성 모델의 관계 이해 능력을 활용한 Referring Diffusional Segmentor | 판별 모델(CLIP) 대신/병행 생성 모델 활용 |
| **IteRPrimE (Wang et al.)** | AAAI 2025 | VLP 모델의 Grad-CAM 히트맵을 활용하고 반복적 정제 전략으로 타겟 영역 집중을 점진적 강화 | 공간 위치 인식 문제를 Grad-CAM 반복 정제로 해결 |
| **HybridGL (Liu & Li)** | arXiv 2025 | 학습 불필요 하이브리드 글로벌-로컬 특징 추출 + 공간 가이던스 증강 전략 | GL-CLIP의 글로벌-로컬 아이디어를 확장하고 SAM + GEM 기반 공간 맵 도입 |
| **CoPatch (An et al.)** | arXiv 2025 | 문맥 토큰의 공간 단서를 통합한 하이브리드 텍스트 특징 + 중간 레이어에서 패치 수준 이미지 특징 추출 → CoMap으로 정밀한 마스크 선택 | CLIP 내부의 잠재된 공간 지식을 직접 추출 |

### 성능 비교 (RefCOCO 계열 mIoU 기준, SAM 마스크 사용)

CoPatch(DFN ViT-H/14)는 TAS와 HybridGL을 포함한 기존 SOTA 제로샷 방법을 능가하며 9개 평가 스플릿 중 8개에서 최고 성능을 달성한다. 아래는 대략적인 mIoU 범위를 나타낸다:

| 방법 | RefCOCOg (val) | RefCOCO (val) | RefCOCO+ (val) |
|---|---|---|---|
| GL-CLIP (ViT-B/32) | ~36 | ~38 | ~33 |
| TAS | ~47 | ~47 | ~44 |
| IteRPrimE | ~46 | ~44 | ~44 |
| HybridGL | ~45 | ~44 | ~38 |
| CoPatch (ViT-B/16) | ~54 | ~50 | ~47 |
| CoPatch (DFN ViT-H/14) | ~56 | ~51 | ~47 |

### 핵심 트렌드

1. **Training-free 패러다임의 진화:** GL-CLIP이 개척한 학습 불필요 접근이 TAS → IteRPrimE → HybridGL → CoPatch로 지속 발전 중
2. **공간 인식 강화:** GL-CLIP의 약점이었던 공간 관계 인식을 후속 연구들이 Grad-CAM, GEM, CoMap 등으로 해결
3. **마스크 생성기 발전:** FreeSOLO에서 SAM, Mask2Former로의 전환이 전체 성능 향상에 기여
4. **VLM 내부 지식 활용:** VLM에 내재된 잠재적 공간 지식을 복원하고 활용하는 것의 중요성이 강조되며, 제로샷 RIS의 새로운 기회를 열고 있다.

---

## 참고자료 및 출처

1. Yu, S., Seo, P. H., & Son, J. (2023). "Zero-Shot Referring Image Segmentation With Global-Local Context Features." *CVPR 2023*, pp. 19456–19465. [arXiv:2303.17811](https://arxiv.org/abs/2303.17811)
2. GitHub 공식 코드: [Seonghoon-Yu/Zero-shot-RIS](https://github.com/Seonghoon-Yu/Zero-shot-RIS)
3. CVPR 2023 Open Access: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Zero-Shot_Referring_Image_Segmentation_With_Global-Local_Context_Features_CVPR_2023_paper.html)
4. Suo, Y., Zhu, L., & Yang, Y. (2023). "Text Augmented Spatial Aware Zero-shot Referring Image Segmentation." *EMNLP 2023 Findings*. [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.73/)
5. Ni, M. et al. (2023). "Ref-Diff: Zero-shot Referring Image Segmentation with Generative Models." [arXiv:2308.16777](https://arxiv.org/abs/2308.16777)
6. Wang, Y. et al. (2025). "IteRPrimE: Zero-shot Referring Image Segmentation with Iterative Grad-CAM Refinement and Primary Word Emphasis." *AAAI 2025*. [arXiv:2503.00936](https://arxiv.org/abs/2503.00936)
7. Liu, T. & Li (2025). "Hybrid Global-Local Representation with Augmented Spatial Guidance for Zero-Shot Referring Image Segmentation." [arXiv:2504.00356](https://arxiv.org/abs/2504.00356)
8. An, N. M. et al. (2025). "CoPatch: Zero-Shot Referring Image Segmentation by Leveraging Untapped Spatial Knowledge in CLIP." [arXiv:2509.23098](https://arxiv.org/abs/2509.23098)
9. ResearchGate 논문 페이지: [ResearchGate](https://www.researchgate.net/publication/369740419)
10. Medium 논문 리뷰 (小連同學): [Medium Review](https://medium.com/@liee9n/paper-review-zero-shot-referring-image-segmentation-with-global-local-context-features-db6334647ff5)
11. Semantic Scholar: [Semantic Scholar](https://www.semanticscholar.org/paper/c70833ab04675e6e339739c11eebd20c60db3d9f)

---

> **참고:** 위 수식들은 논문의 방법론을 기반으로 재구성한 것이며, 논문 원문의 표기와 일부 차이가 있을 수 있습니다. 정확한 수식과 하이퍼파라미터 값은 원논문 및 공식 코드를 직접 참조하시기 바랍니다.
