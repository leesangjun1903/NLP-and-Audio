# Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders

## 📌 논문 기본 정보

- **제목**: Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders
- **게재**: ICLR 2025 (Published as a conference paper)
- **저자**: Min Shi, Fuxiao Liu, Shihao Wang 외 (NVIDIA, Georgia Tech, UMD, HKPU, NYU)
- **코드**: https://github.com/NVlabs/Eagle
- **arXiv**: arXiv:2408.15998v2

---

## 1. 핵심 주장과 주요 기여 요약

### 🎯 핵심 주장

Eagle은 **다중 비전 인코더(Mixture of Vision Encoders)** 를 활용하여 멀티모달 LLM(MLLM)의 시각적 인지 능력을 체계적으로 향상시킬 수 있으며, 복잡한 퓨전 아키텍처 없이도 **단순한 채널 연결(Channel Concatenation)** 만으로 최고 수준의 성능을 달성할 수 있다는 것을 실증적으로 보여줍니다.

### 🏆 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **체계적 디자인 공간 탐색** | 비전 인코더 선택, 고해상도 적응, 퓨전 전략, 학습 레시피에 대한 광범위한 ablation study 수행 |
| **단순하고 효과적인 퓨전 전략 발견** | Channel Concatenation이 복잡한 퓨전 방법(LLaVA-HR, Mini-Gemini 등)보다 효율적이고 경쟁력 있음을 증명 |
| **Vision Encoder Unfreezing의 중요성** | SFT 시 비전 인코더를 학습(unfreeze)하는 것이 성능 향상에 핵심임을 발견 |
| **Pre-Alignment 전략 제안** | 비전 전문가별 개별 사전 정렬 단계 도입으로 모달리티 간 표현 불일치 해소 |
| **다중 전문가 선택 방법론** | Round-robin 방식으로 최적 인코더 조합(CLIP, ConvNeXt, SAM, Pix2Struct, EVA-02) 식별 |
| **완전한 오픈소스** | 데이터, 학습 레시피, 코드 모두 공개 |

---

## 2. 세부 분석: 해결하려는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

기존 MLLM은 단일 비전 인코더(주로 CLIP)에 의존하며 다음과 같은 한계가 있었습니다:

1. **저해상도 처리 한계**: CLIP은 $224 \times 224$ 또는 $336 \times 336$ 해상도에서 사전 학습되어 OCR, 문서 이해 등 세밀한 시각 처리에 부족
2. **단일 도메인 표현 한계**: 단일 인코더는 텍스트 인식, 객체 검출, 분할 등 다양한 시각 도메인을 모두 처리하기 어려움
3. **다중 인코더 설계 공간의 체계적 연구 부재**: 여러 비전 인코더를 사용하는 모델들이 있지만, 퓨전 전략, 인코더 선택, 학습 방식에 대한 엄밀한 비교 연구가 없었음
4. **표현 불일치(Representational Inconsistency)**: 비전 전용 태스크(검출, 분할 등)로 사전학습된 인코더와 언어 모델 간의 표현 공간 불일치

### 2.2 제안하는 방법

#### 2.2.1 모델의 기본 구조

Eagle은 LLaVA-1.5 아키텍처를 기반으로 하며, 다음 세 요소로 구성됩니다:

$$\text{MLLM} = f_{\text{LLM}}\left(\text{Projector}\left(\text{Fusion}\left(\{f_{\text{enc}_i}(I)\}_{i=1}^{N}\right)\right) \oplus E_{\text{text}}\right)$$

여기서:
- $f_{\text{enc}_i}(I)$: $i$번째 비전 인코더의 출력 특징 맵
- $\text{Fusion}(\cdot)$: 다중 인코더 특징 통합 함수
- $\text{Projector}(\cdot)$: 시각 임베딩을 텍스트 공간으로 투영하는 레이어
- $E_{\text{text}}$: 텍스트 임베딩
- $\oplus$: 시퀀스 연결(concatenation)

#### 2.2.2 고해상도 적응 (High-Resolution Adaptation)

CLIP 인코더의 위치 임베딩을 보간하여 해상도를 확장합니다:

$$\hat{P}(i, j) = \text{Interpolate}\left(P\left(\frac{i \cdot H_{\text{orig}}}{H_{\text{new}}}, \frac{j \cdot W_{\text{orig}}}{W_{\text{new}}}\right)\right)$$

여기서 $H_{\text{orig}} = W_{\text{orig}} = 336$ (원본), $H_{\text{new}} = W_{\text{new}} = 448$ (목표 해상도)

실험 결과, $448 \times 448$ 보간이 효율성과 성능의 최적 균형점임을 발견했습니다 (Table 2 기준 Avg. 670.5).

#### 2.2.3 퓨전 전략: Channel Concatenation

Eagle이 채택한 채널 연결 방식:

$$F_{\text{fused}} = \text{Concat}\left([f_{\text{enc}_1}(I), f_{\text{enc}_2}(I), \ldots, f_{\text{enc}_N}(I)]\right) \in \mathbb{R}^{T \times (N \cdot d)}$$

여기서:
- $T$: 시각 토큰 수 (각 인코더당 1024개로 고정, bilinear interpolation 또는 pixel shuffle로 조정)
- $d$: 각 인코더의 특징 차원
- $N$: 인코더 수

이는 시퀀스 길이를 증가시키지 않으면서 다양한 인코더의 표현을 통합합니다. 비교한 퓨전 전략들:

| 퓨전 방법 | #Token(V) | #Tokens/s | Avg. |
|---|---|---|---|
| Sequence Append | 2048 | 46.1 | 690.5 |
| **Channel Concat.** | **1024** | **47.3** | **681.5** |
| LLaVA-HR | 1024 | 47.0 | 678.7 |
| Mini-Gemini | 1024 | 45.3 | 672.5 |
| Deformable Attn. | 1024 | 47.3 | 674.3 |

> ⚠️ **주의**: CLIP+ConvNeXt 2개 인코더 조합에서는 Sequence Append가 690.5로 소폭 우세하나, 인코더 수 확장 시 시퀀스 길이가 급증하는 확장성 문제가 있어 Channel Concat.이 최종 선택됨.

#### 2.2.4 Pre-Alignment 전략

논문의 핵심 기여 중 하나인 Pre-Alignment는 3단계 학습 파이프라인으로 구성됩니다:

**Stage 1: Pre-Alignment**

각 비전 전문가 $E_i$를 LLM(Vicuna-7B)이 동결된 상태에서 개별적으로 정렬:

$$\mathcal{L}_{\text{pre-align}} = -\sum_{t=1}^{T} \log P_\theta\left(y_t \mid y_{<t}, f_{\text{enc}_i}(I)\right), \quad \theta = \text{Projector}_i$$

$$\nabla_\theta \mathcal{L} \neq 0, \quad \nabla_{\theta_{\text{LLM}}} \mathcal{L} = 0$$

**Stage 2: Joint Projector Training**

모든 비전 전문가의 특징을 결합하여 통합 프로젝터 학습:

$$\mathcal{L}_{\text{joint}} = -\sum_{t=1}^{T} \log P_\theta\left(y_t \mid y_{<t}, F_{\text{fused}}\right)$$

$$F_{\text{fused}} = \text{Concat}([f_{\text{enc}_1}(I), \ldots, f_{\text{enc}_N}(I)])$$

**Stage 3: Supervised Fine-Tuning (SFT)**

전체 모델(비전 인코더 포함) 엔드-투-엔드 학습:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log P_\theta\left(y_t \mid y_{<t}, F_{\text{fused}}, x_{\text{text}}\right)$$

$$\nabla_{\theta_{\text{all}}} \mathcal{L} \neq 0 \quad \text{(모든 파라미터 업데이트)}$$

Pre-Alignment 효과 (Table 5):

| CLIP | Vision Expert | Unfreeze | Pre-align | Avg. |
|---|---|---|---|---|
| CLIP-448 | ConvNext-1024 | ✗ | ✗ | 652.0 |
| CLIP-448 | ConvNext-1024 | ✗ | ✓ | 670.1 |
| CLIP-448 | ConvNext-1024 | ✓ | ✗ | 681.5 |
| CLIP-448 | ConvNext-1024 | **✓** | **✓** | **686.2** |

#### 2.2.5 최적 인코더 조합 선택: Round-Robin 방식

$$\text{Combo}^* = \arg\max_{\mathcal{S}} \text{Avg}(\mathcal{S}), \quad |\mathcal{S}| = k$$

$$\mathcal{S}_{k+1} = \mathcal{S}_k \cup \arg\max_{e \notin \mathcal{S}_k} \text{Avg}(\mathcal{S}_k \cup \{e\})$$

최종 선정된 최적 5개 인코더 조합 (X5):
- **CLIP** (Vision-Language Alignment, 448px)
- **ConvNeXt** (Vision-Language Alignment, 1024px)  
- **EVA-02** (Object Detection, 1024px)
- **Pix2Struct** (Text Recognition/OCR, 1024px)
- **SAM** (Segmentation, 1024px)

### 2.3 모델 구조 상세

```
[입력 이미지]
      │
      ├──► CLIP-448         → [1024 토큰, d₁차원]  ─┐
      ├──► ConvNeXt-1024    → [1024 토큰, d₂차원]  ─┤
      ├──► EVA-02-1024      → [1024 토큰, d₃차원]  ─┼─► Channel Concat
      ├──► Pix2Struct-1024  → [1024 토큰, d₄차원]  ─┤   [1024 토큰, Σdᵢ차원]
      └──► SAM-1024         → [1024 토큰, d₅차원]  ─┘
                                                         │
                                                    [Projector]
                                                         │
                                               [Text Embedding] ──► 결합
                                                         │
                                               [LLM: Vicuna/Llama3]
                                                         │
                                                    [출력 텍스트]
```

### 2.4 성능 향상

#### 주요 벤치마크 결과 (Vicuna-13B 기준, Table 6, 8):

| 모델 | MME | MMB | SEED | TextVQA | ChartQA | OCRBench | POPE |
|---|---|---|---|---|---|---|---|
| LLaVA-1.5 | 1531 | 67.7 | 61.6 | 61.3 | - | 331 | 85.9 |
| LLaVA-NeXT | 1575 | 70.0 | 71.9 | 67.1 | 62.2 | 514 | 86.2 |
| Mini-Gemini | 1565 | 68.6 | 70.6 | 65.9 | 56.6 | 466 | - |
| Cambrian-1 | 1610 | 75.7 | 74.4 | 72.8 | 76.8 | - | 73.7 |
| **Eagle-X5 (+Pre-Align)** | **1605** | **71.6** | **74.9** | **73.3** | **72.1** | **598** | **89.2** |

#### Cambrian-1 동일 데이터 기준 (Table 8, Llama3-8B):

| 카테고리 | Cambrian-1 | Eagle-X5 | 향상 |
|---|---|---|---|
| Knowledge Avg. | 61.3 | **65.2** | +3.9 |
| General Avg. | 73.0 | **76.2** | +3.2 |
| **OCR and Chart Avg.** | 71.3 | **77.0** | **+5.7** |
| Vision-Centric Avg. | 51.3 | **52.0** | +0.7 |
| **전체 Avg.** | 64.2 | **67.2** | **+3.0** |

### 2.5 한계

논문에서 명시적으로 언급된 한계 및 확인 가능한 제약사항:

1. **계산 비용 증가**: X5 설정 시 비전 인코더 파라미터 수 2,282.8M, FLOPs 6,617.4G, 처리 속도 3.8 Img/Sec (bs=4)으로 단일 인코더 대비 대폭 증가
2. **DINOv2 추가 시 성능 저하**: 6번째 인코더(DINOv2) 추가 시 오히려 성능이 686.8로 하락 (X5: 697.1 대비), 무조건적 인코더 추가가 도움이 되지 않음을 시사
3. **복잡한 학습 파이프라인**: 3단계 학습 과정이 필요하여 재현 난이도 증가
4. **타일링(Tiling) 기법과의 미결합**: 논문 자체적으로 타일링 기법과 혼합 인코더 방식이 호환 가능하다고 언급하나, 이 조합에 대한 체계적 연구 미포함
5. **벤치마크 포화 가능성**: 일부 벤치마크에서 이미 높은 수치를 기록하여 추가 개선 여지 측정의 어려움

---

## 3. 모델의 일반화 성능 향상 가능성

Eagle의 일반화 성능 향상은 다음의 메커니즘을 통해 이루어집니다:

### 3.1 도메인 다양성을 통한 일반화

서로 다른 도메인에서 사전학습된 인코더의 조합이 일반화에 기여합니다:

$$\mathcal{D}_{\text{visual}} = \mathcal{D}_{\text{VL}} \cup \mathcal{D}_{\text{OCR}} \cup \mathcal{D}_{\text{Det}} \cup \mathcal{D}_{\text{Seg}}$$

각 인코더가 서로 보완적인 시각 특징을 제공:

| 인코더 | 사전학습 도메인 | 강점 | 약점 |
|---|---|---|---|
| CLIP | Vision-Language | 전반적 VQA, 의미 이해 | OCR, 세밀한 지역화 |
| ConvNeXt | Vision-Language | 전반적 VQA, 고해상도 | - |
| EVA-02 | Object Detection | 객체 인식, POPE(hallucination↓) | 텍스트 인식 |
| Pix2Struct | OCR/Text Recognition | 텍스트, 문서, 차트 이해 | 일반 객체 인식 |
| SAM | Segmentation | 공간적 구조 이해 | 텍스트 인식 |

### 3.2 Pre-Alignment의 일반화 기여

Pre-Alignment는 각 인코더의 고유한 편향(bias)을 LLM의 언어 공간에 미리 정렬함으로써, SFT 시 모든 인코더가 효과적으로 기여할 수 있게 합니다:

$$\text{Gap}(E_i, \text{LLM}) = \|f_{\text{enc}_i}(I) - \text{Proj}_i \circ f_{\text{text}}(I)\|_2$$

Pre-Alignment는 이 Gap을 최소화하는 방향으로 작동하여, 비전-언어 정렬이 되지 않은 인코더(SAM, EVA-02 등)도 효과적으로 활용 가능하게 합니다.

### 3.3 Cambrian-1 동일 조건 비교에서의 일반화 증거

가장 강력한 일반화 증거는 Table 8입니다. 동일한 사전학습 및 SFT 데이터를 사용했을 때 Eagle이 Cambrian-1 대비 전 카테고리에서 일관된 향상을 보입니다:

- **범용 일반화**: General 카테고리 +3.2%p (Llama3-8B 기준)
- **지식 일반화**: Knowledge 카테고리 +3.9%p
- **OCR/문서 특화**: OCR & Chart 카테고리 +5.7%p (가장 큰 향상)
- **시각 중심 태스크**: Vision-Centric +0.7%p

이는 Eagle의 설계가 데이터 편향에 의존하지 않고, **아키텍처 자체의 우수성으로 일반화**됨을 의미합니다.

### 3.4 인코더 조합의 체계적 일반화 향상

Figure 4에서 확인 가능한 일반화 패턴:

$$\text{Avg}(X_1) = 670.5 \rightarrow \text{Avg}(X_2) = 681.5 \rightarrow \text{Avg}(X_3) = 690.7 \rightarrow \text{Avg}(X_4) = 694.6 \rightarrow \text{Avg}(X_5) = 697.1$$

인코더 수 증가 $N$에 따른 정규화 평균 점수의 단조 증가는 **다양한 시각 도메인 정보의 상호보완성**이 일반화에 직접 기여함을 보여줍니다.

### 3.5 Hallucination 감소를 통한 일반화

EVA-02(객체 검출 사전학습)를 추가하면 POPE 벤치마크 성능이 향상됩니다. POPE는 시각적 환각(hallucination)을 측정하는 벤치마크로, 이 향상은 모델이 **더 신뢰할 수 있는 시각 이해**를 갖추게 됨을 의미합니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 MLLM 아키텍처 계보

```
CLIP (2021) ──► LLaVA-1.5 (2023) ──► LLaVA-NeXT (2024)
                    │                        │
              BLIP-2 (2023)          Mini-Gemini (2024)
                    │                LLaVA-HR (2024)
              Flamingo (2022)        Cambrian-1 (2024)
                    │                        │
              InternVL (2023) ────────► Eagle (2024/ICLR2025)
```

### 4.2 주요 비교 모델과의 차이점

| 모델 | 비전 인코더 | 퓨전 방법 | 고해상도 | Pre-Alignment | 인코더 Unfreeze |
|---|---|---|---|---|---|
| **LLaVA-1.5** (2023) | CLIP (단일) | - | ✗ | ✗ | ✗ |
| **LLaVA-NeXT** (2024) | CLIP (단일) | Tiling | ✓ | ✗ | ✗ |
| **Mini-Gemini** (2024) | CLIP + ConvNeXt | Cross-Attention | ✓ | ✗ | ✗ |
| **LLaVA-HR** (2024) | CLIP + ConvNeXt | MR-Adapter | ✓ | ✗ | ✗ |
| **Prismatic VLM** (2024) | 다중 | Channel Concat | 부분 | ✗ | ✗ |
| **Cambrian-1** (2024) | 다중 | Spatial Pooling | ✓ | ✗ | 부분 |
| **MouSi** (2024) | 다중 | Sequence Append | ✓ | ✗ | ✗ |
| **BRAVE** (2024) | 다중 | Token Concat | ✓ | ✗ | ✗ |
| **MoAI** (2024) | 다중 + 전문가 출력 | Prompt 보강 | ✓ | ✗ | ✗ |
| **Eagle** (2024) | **5개 다중** | **Channel Concat** | **✓ (448/1024)** | **✓** | **✓** |

### 4.3 기술적 비교 심층 분석

#### vs. Cambrian-1 (Tong et al., 2024)

Cambrian-1은 Eagle과 가장 유사한 설계 철학을 가진 동시대 연구입니다:
- **공통점**: 다중 비전 인코더, 비전 중심 설계, 광범위한 데이터
- **차이점**: Cambrian-1은 Spatial Vision Aggregator(SVA)라는 복잡한 퓨전 모듈 사용, Eagle은 단순 채널 연결
- **결과**: 동일 데이터 기준 Eagle이 전 카테고리에서 우세 (Table 8)

#### vs. Prismatic VLM (Karamcheti et al., 2024)

Prismatic도 채널 연결을 디자인 공간 탐색에 포함했으나:
- Prismatic: 인코더 Unfreeze 미실시, Pre-Alignment 없음, 제한적 ablation
- Eagle: 인코더 Unfreeze가 핵심 발견, Pre-Alignment 신규 제안, 광범위한 ablation

#### vs. Mini-Gemini (Li et al., 2024b)

- Mini-Gemini: CLIP을 쿼리로 ConvNeXt에 cross-attend하는 복잡한 구조
- Eagle 발견: 이 복잡한 접근이 단순 채널 연결 대비 성능 열위 (672.5 vs 681.5)

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### 📐 설계 원칙의 재정립
Eagle은 "복잡한 퓨전 = 더 좋은 성능"이라는 통념을 깨고, **체계적인 디자인 공간 탐색과 단순하고 견고한 설계 원칙**이 더 중요함을 실증했습니다. 이는 향후 MLLM 연구에서 구조적 복잡성보다 **기반 설계의 중요성**을 강조하는 흐름을 만들 것입니다.

#### 🔧 인코더 Unfreeze의 표준화
Vision Encoder를 SFT 시 동결(freeze)하던 관행(LLaVA 계열 등)에 반해, **인코더를 학습(unfreeze)하는 것이 핵심**임을 체계적으로 증명했습니다. 이는 후속 연구들의 기본 설정을 바꿀 가능성이 높습니다.

#### 🔗 Pre-Alignment의 확장 가능성
비전 전용 인코더와 LLM 간의 정렬 문제는 더 많은 모달리티(오디오, 비디오, 3D 등)를 결합하는 연구에서도 유사하게 나타납니다. Pre-Alignment 개념은 멀티모달 연구 전반에 적용 가능한 일반적 원칙으로 확장될 수 있습니다.

#### 📊 벤치마크 설계에 대한 시사점
Eagle은 특히 OCR, 문서 이해에서 두드러진 성능 향상을 보여, 기존 일반 VQA 벤치마크만으로는 MLLM의 시각 능력을 충분히 평가하기 어렵다는 것을 시사합니다. **도메인 특화 평가 벤치마크**의 중요성이 부각될 것입니다.

### 5.2 앞으로 연구 시 고려할 점

#### ⚡ 효율성과 성능의 균형 문제
X5 설정에서 처리 속도가 3.8 Img/Sec으로 크게 감소합니다. 실용적 배포를 위해 다음을 고려해야 합니다:

$$\text{효율성 목표}: \max_{\mathcal{S}} \frac{\text{Avg}(\mathcal{S})}{\text{FLOPs}(\mathcal{S})} \text{ subject to } |\mathcal{S}| \leq N_{\max}$$

- 지식 증류(Knowledge Distillation)를 통해 다중 인코더의 지식을 단일 경량 인코더로 압축
- 동적 인코더 선택(Dynamic Expert Selection) 메커니즘 연구 필요

#### 🔢 인코더 수의 최적화 연구
현재 Round-Robin은 그리디 알고리즘으로, 전역 최적해를 보장하지 않습니다. 더 정교한 탐색 방법(강화 학습 기반, 베이지안 최적화 등) 연구가 필요합니다:

$$\mathcal{S}^* = \arg\max_{|\mathcal{S}| \leq N} \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{test}}}[\text{Score}(\mathcal{S}, x, y)]$$

#### 🖼️ 타일링과 혼합 인코더의 시너지
논문은 타일링(LLaVA-NeXT 방식)과 혼합 인코더가 상호 보완적임을 언급하지만 조합 연구를 수행하지 않았습니다. 이 조합은 더 높은 해상도와 더 풍부한 표현을 동시에 달성할 수 있는 유망한 방향입니다.

#### 📏 학습 데이터 구성의 세밀한 분석
Eagle1.8M 데이터의 도메인 구성이 인코더 조합의 효과에 어떤 영향을 미치는지에 대한 연구가 부족합니다. 특정 인코더의 강점을 최대화하기 위한 **데이터-인코더 공동 최적화** 연구가 필요합니다.

#### 🌐 더 넓은 모달리티로의 확장
Eagle의 "전문가 혼합" 개념은 시각 외에도 오디오, 비디오, 포인트 클라우드 등 다양한 모달리티 인코더에 적용 가능합니다:

$$F_{\text{multi-modal}} = \text{Concat}([F_{\text{vision}}, F_{\text{audio}}, F_{\text{video}}, \ldots])$$

#### 🧪 이론적 근거 강화
왜 Channel Concatenation이 더 복잡한 퓨전 방법보다 효과적인지에 대한 이론적 분석이 필요합니다. 각 인코더 특징의 **독립성(independence)** 과 **보완성(complementarity)** 을 정량화하는 지표 개발이 중요합니다.

$$\text{Complementarity}(E_i, E_j) = 1 - \frac{I(F_i; F_j)}{H(F_i) + H(F_j)}$$

여기서 $I(\cdot;\cdot)$은 상호 정보량(Mutual Information), $H(\cdot)$은 엔트로피입니다.

---

## 📚 참고 자료

**1차 출처 (논문 PDF 직접 참조):**
- **Min Shi, Fuxiao Liu et al.** "Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders." *Published as a conference paper at ICLR 2025.* arXiv:2408.15998v2 [cs.CV], 2 Mar 2025.

**논문 내 인용된 주요 참고문헌 (2020년 이후):**
- Liu et al. (2023c). "Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)." arXiv:2310.03744
- Liu et al. (2024a). "LLaVA-NeXT: Improved reasoning, ocr, and world knowledge." llava-vl.github.io
- Li et al. (2024b). "Mini-Gemini: Mining the potential of multi-modality vision language models." arXiv:2403.18814
- Luo et al. (2024). "Feast your eyes: Mixture-of-resolution adaptation for multimodal large language models (LLaVA-HR)." arXiv:2403.03003
- Tong et al. (2024). "Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs." arXiv:2406.16860
- Karamcheti et al. (2024). "Prismatic VLMs: Investigating the design space of visually-conditioned language models." arXiv:2402.07865
- Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision (CLIP)." ICML 2021
- Oquab et al. (2023). "DINOv2: Learning robust visual features without supervision." arXiv:2304.07193
- Fang et al. (2023a,b). "EVA-02: A visual representation for neon genesis." arXiv:2303.11331
- Kirillov et al. (2023). "Segment Anything (SAM)." ICCV 2023
- Lee et al. (2023). "Pix2Struct: Screenshot parsing as pretraining for visual language understanding." ICML 2023
- Kar et al. (2024). "BRAVE: Broadening the visual encoding of vision-language models." arXiv:2404.07204
- Fan et al. (2024). "MouSi: Poly-visual-expert vision-language models." arXiv:2401.17221
- Zong et al. (2024). "MoVA: Adapting mixture of vision experts to multimodal context." arXiv:2404.13046
- Ranzinger et al. (2024). "AM-RADIO: Agglomerative vision foundation model reduce all domains into one." CVPR 2024
- Chen et al. (2023f). "InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks." arXiv:2312.14238
