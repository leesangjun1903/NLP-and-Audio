# LangSplat: 3D Language Gaussian Splatting 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

LangSplat은 기존 NeRF 기반 3D 언어 필드 방법(특히 LERF)의 두 가지 핵심 한계—**낮은 렌더링 속도**와 **부정확한 객체 경계 표현**—를 3D Gaussian Splatting(3D-GS)과 SAM 기반 계층적 의미론 학습을 통해 동시에 해결할 수 있다는 것입니다.

### 주요 기여

| 기여 | 내용 |
|---|---|
| **① 3D-GS 기반 언어 필드** | NeRF 대신 3D Gaussian Splatting을 활용한 최초의 3D 언어 필드 방법 |
| **② Scene-wise Language Autoencoder** | 512차원 CLIP 임베딩을 3차원 잠재 공간으로 압축하여 메모리 폭발 문제 해결 |
| **③ SAM 계층적 의미론** | SAM을 활용하여 subpart/part/whole 3개 계층의 정밀한 세그멘테이션 마스크 기반 학습으로 point ambiguity 해결 |
| **④ 속도/정확도 향상** | LERF 대비 **199× 속도 향상** (1440×1080 해상도), 3D-OVS 데이터셋에서 mIoU **93.4%** 달성 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

#### 문제 1: 느린 렌더링 속도 (NeRF의 구조적 한계)

NeRF는 볼륨 렌더링(volume rendering) 특성상 광선(ray)을 따라 수백 개의 3D 포인트를 적분해야 하므로, 고해상도 실시간 렌더링이 구조적으로 불가능합니다.

#### 문제 2: Point Ambiguity (포인트 모호성)

CLIP 임베딩은 이미지 수준(image-level)으로 정렬되어 있어, 단일 3D 포인트에 할당할 픽셀 정렬(pixel-aligned) 언어 임베딩이 본질적으로 모호합니다.

예: 곰의 코 위의 한 점은 "bear's nose", "bear's head", "bear" 세 가지 텍스트 쿼리 모두에 높은 응답값을 가져야 합니다.

기존 방법(LERF 등)은 이를 해결하기 위해 **다중 절대 스케일(multi absolute scale)** 에서 패치 단위 CLIP 특징을 추출하고, 추론 시 최대 30개의 스케일에서 동시에 렌더링하여 최적 스케일을 선택합니다. 이는 쿼리 시간을 최대 30배 증가시키고, 패치가 객체 경계를 정확히 포함하지 못해 노이즈가 많은 언어 필드를 학습하게 됩니다.

#### 문제 3: 메모리 폭발 (Explicit Modeling의 한계)

3D-GS는 명시적(explicit) 표현 방법이므로, 수백만 개의 Gaussian 포인트 각각에 512차원 CLIP 임베딩을 직접 저장하면 RGB 색상 저장 대비 **35배 이상의 메모리**가 필요합니다.

---

### 2.2 제안하는 방법 및 수식

#### Step 1: SAM 기반 계층적 의미론 학습

$32 \times 32$ 격자 포인트 프롬프트를 SAM에 입력하여 세 개의 계층적 세그멘테이션 맵을 생성합니다:

$$\boldsymbol{M}^s_0, \boldsymbol{M}^p_0, \boldsymbol{M}^w_0$$

여기서 $s$ = subpart, $p$ = part, $w$ = whole을 의미합니다. IoU 점수, 안정성 점수, 마스크 간 중복률을 기반으로 중복 마스크를 제거하여 최종 세그멘테이션 맵 $\boldsymbol{M}^s, \boldsymbol{M}^p, \boldsymbol{M}^w$를 획득합니다.

픽셀 정렬 언어 임베딩은 다음과 같이 정의됩니다:

$$\boldsymbol{L}^l_t(v) = V(\boldsymbol{I}_t \odot \boldsymbol{M}^l(v)), \quad l \in \{s, p, w\} $$

여기서 $V$는 CLIP 이미지 인코더, $\boldsymbol{I}_t$는 $t$번째 입력 이미지, $\boldsymbol{M}^l(v)$는 픽셀 $v$가 속하는 의미론적 레벨 $l$의 마스크 영역, $\odot$는 마스크 적용 연산입니다.

#### Step 2: 3D Gaussian Splatting 기반 언어 필드

3D Gaussian은 평균 $\mu \in \mathbb{R}^3$와 공분산 행렬 $\Sigma$로 특징지어집니다:

$$G(x) = \exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\right) $$

타일 기반 래스터라이저를 사용한 색상 렌더링:

$$C(v) = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j) $$

여기서 $c_i$는 $i$번째 Gaussian의 색상, $\alpha_i = o_i G^{2D}_i(v)$이며, $o_i$는 불투명도(opacity), $G^{2D}_i(\cdot)$는 $i$번째 Gaussian의 2D 투영 함수입니다.

LangSplat은 각 3D Gaussian에 세 가지 언어 임베딩 $\{f^s, f^p, f^w\}$를 추가하여 **3D Language Gaussian**을 구성하고, 언어 특징 렌더링은 다음과 같습니다:

$$\boldsymbol{F}^l(v) = \sum_{i \in \mathcal{N}} \boldsymbol{f}^l_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j), \quad l \in \{s, p, w\} $$

#### Step 3: Scene-wise Language Autoencoder

메모리 절감을 위해 경량 MLP 기반 오토인코더를 학습합니다:

- **인코더** $E$: $D$차원 CLIP 특징 $\boldsymbol{L}^l_t(v) \in \mathbb{R}^D$를 $d$차원 잠재 특징 $\boldsymbol{H}^l_t(v) = E(\boldsymbol{L}^l_t(v)) \in \mathbb{R}^d$로 압축 (실험에서 $d=3$ 사용, $d \ll D$)
- **디코더** $\Psi$: 잠재 특징을 원래 CLIP 임베딩으로 복원

오토인코더 학습 목적 함수 (L1 손실 + 코사인 거리 손실):

$$\mathcal{L}_{ae} = \sum_{l \in \{s,p,w\}} \sum_{t=1}^{T} d_{ae}(\Psi(E(\boldsymbol{L}^l_t(v))), \boldsymbol{L}^l_t(v)) $$

3D Language Gaussian 학습 목적 함수:

$$\mathcal{L}_{lang} = \sum_{l \in \{s,p,w\}} \sum_{t=1}^{T} d_{lang}(\boldsymbol{F}^l_t(v), \boldsymbol{H}^l_t(v)) $$

#### Step 4: 추론 (Open-vocabulary Querying)

관련성 점수(Relevancy Score)는 LERF를 따라 다음과 같이 정의됩니다:

$$\text{Relevancy}(\phi_{img}, \phi_{qry}) = \min_i \frac{\exp(\phi_{img} \cdot \phi_{qry})}{\exp(\phi_{img} \cdot \phi_{qry}) + \exp(\phi_{img} \cdot \phi^i_{canon})}$$

여기서 $\phi^i_{canon}$은 "object", "things", "stuff", "texture" 중에서 선택된 정규 문구의 CLIP 임베딩입니다.

---

### 2.3 모델 구조

```
[Multi-view Images]
        ↓
   [SAM (ViT-H)]
        ↓ (3계층 마스크: subpart / part / whole)
[CLIP Encoder (OpenCLIP ViT-B/16)]
        ↓ (512-dim CLIP embeddings)
[Scene-wise Language Autoencoder (MLP)]
        ↓ (3-dim latent features)
[3D Language Gaussians 학습]
  - 3D-GS RGB 학습 (30,000 iter)
  - Language feature 학습 (30,000 iter, 나머지 파라미터 고정)
        ↓
[Tile-based Rasterizer → 2D 렌더링]
        ↓
[Decoder Ψ → 512-dim CLIP space 복원]
        ↓
[CLIP Text Encoder → Relevancy Score 계산]
        ↓
[3D Object Localization / Semantic Segmentation]
```

**구현 세부사항:**
- GPU: NVIDIA RTX-3090
- 학습 시간: ~25분 (1440×1080 해상도)
- 메모리: ~4GB
- 3D 포인트 수: ~2,500,000개/장면

---

### 2.4 성능 향상

#### LERF 데이터셋 결과

**3D Object Localization (정확도 %)**

| 장면 | LSeg | LERF | LangSplat |
|---|---|---|---|
| ramen | 14.1 | 62.0 | **73.2** |
| figurines | 8.9 | 75.0 | **80.4** |
| teatime | 33.9 | 84.8 | **88.1** |
| waldo kitchen | 27.3 | 72.7 | **95.5** |
| **overall** | 21.1 | 73.6 | **84.3** |

**3D Semantic Segmentation (IoU %)**

| 장면 | LSeg | LERF | LangSplat |
|---|---|---|---|
| ramen | 7.0 | 28.2 | **51.2** |
| figurines | 7.6 | 38.6 | **44.7** |
| teatime | 21.7 | 45.0 | **65.1** |
| waldo kitchen | 29.9 | 37.9 | **44.5** |
| **overall** | 16.6 | 37.4 | **51.4** |

#### 3D-OVS 데이터셋 결과 (mIoU %)

| Method | bed | bench | room | sofa | lawn | overall |
|---|---|---|---|---|---|---|
| LERF | 73.5 | 53.2 | 46.6 | 27.0 | 73.7 | 54.8 |
| 3D-OVS | 89.5 | 89.3 | 92.8 | 74.0 | 88.2 | 86.8 |
| **LangSplat** | **92.5** | **94.2** | **94.1** | **90.0** | **96.1** | **93.4** |

#### 속도 비교 (Ablation Study, ramen 장면)

| AE | 3D-GS | SAM | IoU(%) | Speed(s/q) |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 28.20 | 30.93 |
| ✗ | ✗ | ✓ | 46.74 | 7.77 |
| ✗ | ✓ | ✓ | OOM | OOM |
| ✓ | ✓ | ✓ | **51.15** | **0.26** |

→ 최종 **119× 속도 향상** (988×731), **199× 속도 향상** (1440×1080)

---

### 2.5 한계

1. **Scene-specific 특성**: 오토인코더가 특정 장면에 최적화되어 학습되므로, 새로운 장면에 대해 재훈련이 필요합니다 (일반화 미지원).
2. **SAM 의존성**: SAM의 세그멘테이션 품질에 의존하므로, SAM이 실패하는 장면(반사 재질, 극도로 작은 객체 등)에서 성능이 저하될 수 있습니다.
3. **정적 장면 가정**: 동적 장면(dynamic scene)에 대한 언어 필드 모델링은 다루지 않습니다.
4. **디코더 병목**: 논문에서 언급된 것처럼 계산 시간의 대부분이 디코더에 집중되어 있습니다. 1×1 합성곱 레이어로 대체하면 추가 속도 향상이 가능합니다.
5. **훈련 데이터 규모**: 대규모 다양한 3D 장면 데이터와 언어 주석의 부재로, 각 장면마다 별도 훈련이 필요합니다.
6. **CLIP 임베딩 공간의 한계**: CLIP 자체가 가진 편향(bias)이나 언어 이해의 한계가 그대로 전파됩니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 현재의 일반화 구조와 한계

LangSplat의 핵심 설계 중 하나인 **Scene-wise Language Autoencoder**는 특정 장면의 CLIP 특징 분포를 학습하여 효율적으로 압축하는 장점이 있지만, 이는 동시에 **장면-특화(scene-specific)** 특성을 가져 새로운 장면에 즉시 적용할 수 없는 구조적 한계를 내포합니다.

논문에서 직접 언급하듯이:
> *"The language field Φ we train here is scene-specific, meaning we can leverage scene priors to compress CLIP features."*

즉, 각 장면마다 오토인코더를 새로 훈련해야 하므로 **zero-shot 일반화는 현재 구조에서 지원되지 않습니다.**

### 일반화 성능 향상의 구체적 가능성

#### 방향 1: Universal/Generalizable Autoencoder

여러 장면의 CLIP 특징 분포를 커버하는 **범용 오토인코더**를 사전 학습하면, 새로운 장면에서 오토인코더 재훈련 없이 적용 가능합니다. 다만 이 경우 압축 효율과 정확도가 일부 감소할 수 있습니다.

#### 방향 2: Cross-scene Feature Transfer

메타러닝(Meta-learning) 또는 Few-shot 학습 프레임워크를 적용하여 적은 수의 새로운 장면 뷰만으로도 빠르게 적응(adaptation)하는 구조가 가능합니다.

#### 방향 3: Larger Foundation Model Integration

- **더 강력한 VLM** (e.g., CLIP 대신 GPT-4V, LLaVA, Flamingo 등) 통합 시 언어 이해 범위가 확장됩니다.
- **SAM 2** (2024년 공개, 비디오 세그멘테이션 지원) 활용 시 동적 장면으로의 확장이 가능합니다.

#### 방향 4: 3D Prior 활용

- **대규모 3D 데이터셋**(Objaverse, ScanNet 등)으로 사전 학습된 3D 표현을 활용하여, 새로운 장면에 대한 few-shot 적응이 가능한 구조 설계
- Point cloud나 depth 정보를 추가 입력으로 활용하여 초기화 품질을 높이는 방향

#### 방향 5: Hierarchical Semantics의 확장

현재 SAM의 three-level hierarchy (subpart/part/whole)는 고정된 계층 구조입니다. 이를 더 세밀하거나 동적으로 조정 가능한 계층 구조로 확장하면 다양한 도메인(의료 영상, 위성 영상 등)으로의 일반화가 가능합니다.

#### 방향 6: Online/Incremental Learning

새로운 장면 데이터가 순차적으로 입력될 때 기존 지식을 보존하면서 업데이트하는 **점진적 학습(incremental learning)** 구조를 적용하면 실용적 일반화 능력이 향상됩니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 3D 표현 | 언어 특징 | 세그멘테이션 | 속도 | 주요 특징 |
|---|---|---|---|---|---|
| **Semantic NeRF** (Zhi et al., ICCV 2021) | NeRF | 폐쇄 어휘 | 픽셀 정렬 | 느림 | 의미론+외관 공동 인코딩 |
| **Distilled Feature Fields (FFD)** (Kobayashi et al., NeurIPS 2022) | NeRF | LSeg/DINO | 픽셀 정렬 | 느림 | 2D 특징 증류 |
| **Neural Feature Fusion Fields (N3F)** (Tschernezki et al., 3DV 2022) | NeRF | DINO | 픽셀 정렬 | 느림 | 자기지도 특징 융합 |
| **LERF** (Kerr et al., ICCV 2023) | NeRF | CLIP+DINO | 다중 스케일 | 30.93s/q | 최초 CLIP-NeRF 통합 |
| **3D-OVS** (Liu et al., NeurIPS 2023) | NeRF | CLIP+DINO | 약한 지도 | ~55s/q | 카테고리 목록 필요 |
| **CLIP-Fields** (Shafiullah et al., 2022) | NeRF | CLIP | 픽셀 정렬 | 느림 | 로봇 내비게이션 응용 |
| **ConceptFusion** (Jatavallabhula et al., 2023) | 포인트 클라우드 | CLIP+LSeg | 오픈-셋 | - | 멀티모달 3D 맵핑 |
| **LangSplat (Ours)** (Qin et al., CVPR 2024) | **3D-GS** | **CLIP (SAM 마스크)** | **3계층** | **0.26-0.28s/q** | **199× 속도, 93.4% mIoU** |

### 핵심 비교 분석

**1. 3D 표현 방식의 전환**
- NeRF → 3D Gaussian Splatting의 패러다임 전환이 가장 중요한 기여입니다.
- 3D-GS의 명시적(explicit) 표현은 렌더링 속도에서 압도적 우위를 제공하지만, 메모리 효율성에서 새로운 과제를 제시합니다.

**2. Point Ambiguity 해결 접근법 비교**
- LERF/3D-OVS: 다중 절대 스케일 패치 → 추론 시 30개 스케일 렌더링 필요
- LangSplat: SAM 계층적 마스크 → 3개의 사전 정의된 스케일만 사용

**3. 특징 압축 전략**
- 기존 방법들: CLIP 임베딩을 직접 학습하거나 DINO로 보조 정규화
- LangSplat: Scene-specific autoencoder로 512→3차원 압축

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 연구에 미치는 영향

**① 3D-GS + 언어 필드 패러다임 정립**

LangSplat은 "3D Gaussian Splatting을 언어 필드에 적용한 최초의 방법"으로서 후속 연구의 기반이 됩니다. 이미 Gaussian Grouping, Feature 3DGS, OpenGaussian 등 다수의 후속 연구가 이 방향으로 진행 중입니다.

**② Foundation Model 조합의 가능성 증명**

SAM + CLIP + 3D-GS의 조합이 시너지를 발휘함을 실험적으로 증명함으로써, 다양한 Foundation Model을 3D 표현에 통합하는 연구 방향을 촉진합니다.

**③ 실시간 3D 언어 질의 응답 시스템 가능성**

199× 속도 향상은 로봇공학, AR/VR, 자율주행 등 실시간성이 요구되는 응용 분야에서의 실용적 배포 가능성을 크게 높입니다.

**④ 메모리 효율적 특징 저장 전략**

Scene-specific autoencoder를 통한 특징 압축 아이디어는 다른 고차원 특징(depth, normal, semantic 등)을 3D 표현에 통합하는 후속 연구에도 적용 가능한 범용적 전략입니다.

### 앞으로 연구 시 고려할 점

**① 동적 장면으로의 확장**

현재 LangSplat은 정적 장면만을 다룹니다. 4D Gaussian Splatting (시간 차원 추가)과 언어 필드를 통합하여 동적 장면에서의 언어 질의를 가능하게 하는 연구가 필요합니다.

**② 일반화 가능한 모델 설계**

- Scene-specific autoencoder의 한계를 극복하기 위한 universal autoencoder 또는 meta-learning 기반 빠른 적응 구조 연구
- 단일 이미지 또는 소수의 뷰로부터 언어 필드를 구성하는 few-shot 접근법

**③ 더 강력한 VLM 통합**

CLIP 이상의 최신 Vision-Language Model (GPT-4V, LLaVA-1.6, InternVL 등)과의 통합을 통해 언어 이해 능력 및 긴 텍스트 쿼리 처리 능력을 향상시킬 수 있습니다.

**④ 3D 일관성 강화**

현재 방법은 2D 뷰에서 SAM을 독립적으로 실행하므로, 뷰 간 마스크 일관성이 완전히 보장되지 않습니다. 3D 공간에서 직접 세그멘테이션을 수행하거나, 멀티뷰 일관성 손실을 명시적으로 학습에 포함시키는 연구가 필요합니다.

**⑤ 스케일러블한 장면 표현**

수백만 개의 Gaussian 포인트를 효율적으로 저장하고 검색하기 위한 공간 인덱싱, LOD(Level of Detail) 기반 표현 등의 연구가 필요합니다.

**⑥ 언어 필드의 불확실성 정량화**

쿼리 결과의 신뢰도 추정(confidence estimation) 및 불확실성 정량화를 통해 로봇공학 등 안전이 중요한 응용에서의 신뢰성을 높여야 합니다.

**⑦ 평가 벤치마크 다양화**

현재 LERF 데이터셋과 3D-OVS 데이터셋 위주의 평가에서 벗어나, ScanNet, Matterport3D, 실외 장면 등 더 다양하고 대규모의 데이터셋에서의 성능 검증이 필요합니다.

---

## 참고 자료

1. **Qin, M., Li, W., Zhou, J., Wang, H., & Pfister, H. (2024). LangSplat: 3D Language Gaussian Splatting. arXiv:2312.16084v2 [cs.CV]** *(본 논문, 1차 참고)*
2. Kerr, J., Kim, C. M., Goldberg, K., Kanazawa, A., & Tancik, M. (2023). LERF: Language Embedded Radiance Fields. ICCV 2023.
3. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. TOG, 42(4):1–14.
4. Kirillov, A., et al. (2023). Segment Anything. ICCV 2023.
5. Radford, A., et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP). ICML 2021.
6. Liu, K., et al. (2023). Weakly Supervised 3D Open-Vocabulary Segmentation (3D-OVS). NeurIPS 2023.
7. Kobayashi, S., Matsumoto, E., & Sitzmann, V. (2022). Decomposing NeRF for Editing via Feature Field Distillation (FFD). NeurIPS 2022.
8. Tschernezki, V., et al. (2022). Neural Feature Fusion Fields (N3F). 3DV 2022.
9. Zhi, S., et al. (2021). In-Place Scene Labelling and Understanding with Implicit Scene Representation (Semantic NeRF). ICCV 2021.
10. LangSplat 프로젝트 페이지: https://langsplat.github.io/
