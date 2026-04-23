
# Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models

> **저자:** Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, Xiang Bai
> **발표:** CVPR 2024 **(Highlight)**
> **arXiv:** [2311.06607](https://arxiv.org/abs/2311.06607)
> **GitHub:** [Yuliang-Liu/Monkey](https://github.com/Yuliang-Liu/Monkey)

---

## 1. 핵심 주장 및 주요 기여 요약

대형 멀티모달 모델(LMM)들은 비전-언어 과제에서 가능성을 보여왔지만, 고해상도 입력 처리와 세밀한 장면 이해에 어려움을 겪고 있다. 이러한 문제를 해결하기 위해 Monkey를 제안한다.

### ✅ 핵심 주장 (Two-fold Contributions)


**기여 1:** 처음부터 사전학습(pretraining)을 새로 하지 않고도, 기존의 비전 인코더(예: ViT-BigHuge) 위에 구축하여 입력 해상도를 $896 \times 1344$ 픽셀까지 효과적으로 향상시킨다.

**기여 2:** 자동으로 풍부한 정보를 제공하여 장면(scene)과 객체(object) 간의 문맥적 연관성 학습을 유도하는 **다단계 설명 생성(Multi-level Description Generation)** 방법을 제안한다.


이 두 가지 전략은 생성된 데이터로부터의 학습을 더욱 효과적으로 만들며, 높은 해상도는 시각 정보를 더 세밀하게 캡처하고 이는 다시 포괄적인 설명의 효과를 높인다.

---

## 2. 해결 문제 / 제안 방법 / 모델 구조 / 성능 / 한계

---

### 🔴 2-1. 해결하고자 하는 문제

기존 LMM들은 지원하는 입력 해상도의 한계(예: $448 \times 448$)와 학습용 이미지-텍스트 쌍의 불완전한 설명으로 인해, 복잡한 장면 이해 및 서술 처리에 어려움을 겪는다.

구체적으로 두 가지 핵심 문제가 존재한다:

| 문제 | 내용 |
|------|------|
| **저해상도 제약** | 기존 ViT 기반 인코더는 $448 \times 448$ 수준의 작은 이미지만 처리 가능 |
| **빈약한 텍스트 레이블** | 단순 캡션 수준의 학습 레이블로는 장면-객체 간 맥락 학습 불충분 |

LMM 학습은 고해상도 이미지로부터 크게 이점을 얻는데, 이는 높은 해상도가 모델이 더 미묘한 시각적 세부 정보를 감지하여 객체, 상호관계, 이미지 내 넓은 맥락을 정확하게 인식하도록 하기 때문이다.

---

### 🟡 2-2. 제안 방법 (수식 포함)

#### 🔹 방법 1: 슬라이딩 윈도우 기반 고해상도 처리 (Patch Division)

Monkey는 ViT를 직접 보간(interpolate)하는 방식 대신, 슬라이딩 윈도우 방식으로 고해상도 이미지를 더 작은 패치들로 분할하는 새로운 모듈을 활용한다. 각 패치는 LoRA 조정과 학습 가능한 시각적 리샘플러(visual resampler)가 적용된 정적(static) 시각 인코더에 의해 독립적으로 처리된다.

입력 이미지 $I \in \mathbb{R}^{H \times W \times 3}$ 를 슬라이딩 윈도우로 분할하면:

$$I \rightarrow \{p_1, p_2, \ldots, p_N\}, \quad p_i \in \mathbb{R}^{h \times w \times 3}$$

각 패치 $p_i$는 고정된(frozen) ViT 인코더 $f_{\text{ViT}}$와 LoRA 어댑터 $\Delta W_i$를 통해 처리된다:

$$\mathbf{v}_i = f_{\text{ViT}}(p_i; W + \Delta W_i), \quad \Delta W_i = B_i A_i$$

여기서 $A_i \in \mathbb{R}^{r \times d}$, $B_i \in \mathbb{R}^{d \times r}$, $r \ll d$ (LoRA 저랭크 분해).

전체 이미지에 대한 전역 특징(global feature)과 지역 패치 특징(local patch features)을 동시에 활용:

$$\mathbf{V}_{\text{global}} = f_{\text{ViT}}(\text{resize}(I)), \quad \mathbf{V}_{\text{local}} = \{\mathbf{v}_1, \ldots, \mathbf{v}_N\}$$

$$\mathbf{V}_{\text{final}} = \text{Concat}(\mathbf{V}_{\text{global}},\; \mathbf{V}_{\text{local}})$$

핵심 아이디어는 이러한 인코더들이 일반적으로 $448 \times 448$과 같은 작은 해상도에서 학습되었으며, 처음부터 다시 학습하는 것은 매우 비용이 크다는 점이다. 각 패치를 지원되는 해상도로 리사이즈함으로써 인코더를 위한 학습 데이터 분포를 유지한다.

#### 🔹 방법 2: 다단계 설명 생성 (Multi-level Description Generation)

단순한 텍스트 레이블과 높은 입력 해상도 사이의 간극을 메우기 위해, 다단계 설명 생성 방법을 제안한다. 이 방법은 자동으로 풍부한 정보를 제공하여 모델이 장면과 객체 간의 문맥적 연관성을 학습하도록 유도한다.

다단계 설명은 다음 세 수준으로 구성된다 (논문 Fig. 3 기반):

$$\text{Description} = \{\mathcal{D}_{\text{scene}},\; \mathcal{D}_{\text{object}},\; \mathcal{D}_{\text{relation}}\}$$

- $\mathcal{D}_{\text{scene}}$: 전체 장면 수준의 설명 (예: BLIP2 생성 캡션)
- $\mathcal{D}_{\text{object}}$: 개별 객체 수준의 설명 (예: 탐지된 영역 기반)
- $\mathcal{D}_{\text{relation}}$: 장면-객체 간 맥락적 연관 서술

최종 학습 목표(Autoregressive Language Modeling Loss):

```math
\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t \mid y_{ < t}, \mathbf{V}_{\text{final}}; \theta)
```

여기서 $y_t$는 $t$번째 토큰, $\theta$는 학습 파라미터이다.

---

### 🟢 2-3. 모델 구조

Monkey의 전체 아키텍처는 원본 이미지의 전역 특징과 분할된 패치들의 지역 특징을 동시에 포착하여 고해상도를 가능하게 한다. 모든 패치는 ViT-BigG (약 20억 파라미터)와 같은 공유된 정적 ViT 인코더를 통해 처리된다.

```
┌─────────────────────────────────────────────────────┐
│                   Input Image (High-Res)             │
│                   H × W (up to 1344×896)             │
└───────────────────┬─────────────────────────────────┘
                    │ Sliding Window Patch Division
          ┌─────────┴──────────┐
          │                    │
   ┌──────▼──────┐      ┌──────▼──────┐
   │ Global Image │      │  Local Patches│
   │ (Resized)   │      │ p1, p2,...pN │
   └──────┬───────┘      └──────┬───────┘
          │                     │ Individual LoRA Adapter
   ┌──────▼──────────────────────▼───────┐
   │     Shared Static ViT Encoder       │
   │     (e.g., ViT-BigG, 2B params)    │
   └──────────────────┬──────────────────┘
                      │ Visual Resampler
   ┌──────────────────▼──────────────────┐
   │        Large Language Model          │
   │    (Qwen-VL 기반 LLM backbone)       │
   └──────────────────┬──────────────────┘
                      │
              ┌───────▼────────┐
              │  Text Output    │
              └────────────────┘
```

주요 구성 요소:

| 컴포넌트 | 설명 |
|----------|------|
| **Shared ViT Encoder** | ViT-BigG (약 2B 파라미터), 가중치 고정(frozen) |
| **LoRA Adapter** | 각 패치별 개별 어댑터 ($r \ll d$의 저랭크 행렬) |
| **Visual Resampler** | 시각 토큰을 LLM 입력에 적합한 형태로 변환 |
| **LLM Backbone** | Qwen-VL 기반 대형 언어 모델 |

Monkey는 8개의 NVIDIA 3090 GPU로 학습할 수 있다. — 즉, 매우 학습-효율적인 구조이다.

---

### 🔵 2-4. 성능 향상

광범위한 절제 실험(ablative results)이 설계의 효과성을 검증한다. 18개 데이터셋에 대한 실험에서 Monkey는 이미지 캡셔닝, 다양한 VQA 형식 등 많은 과제에서 기존 LMM을 능가한다. 특히 밀집 텍스트 질의응답 정성 평가에서 Monkey는 GPT-4V와 비교해 고무적인 결과를 보였다.

표준 $448 \times 448$ 해상도를 훨씬 뛰어넘는 이 해상도 증가는 눈에 잘 띄지 않거나 밀집된 객체 및 고밀도 텍스트를 식별하고 이해하는 능력을 크게 향상시킨다.

평가한 주요 과제별 성능 요약:

| 과제 유형 | 성능 특징 |
|-----------|-----------|
| Image Captioning | 기존 LMM 대비 우수 |
| General VQA | 18개 데이터셋에서 경쟁력 있는 성능 |
| Scene Text-centric VQA | 고해상도 덕분에 탁월한 성능 |
| Document-oriented VQA | GPT-4V 대비 고무적 결과 |
| Dense Text QA | GPT-4V와 대등 혹은 우세 |

---

### 🔴 2-5. 한계 (Limitations)

입력 이미지 처리 능력은 언어 모델의 제한된 입력 길이로 인해 최대 6개의 패치로 제한된다. 이 제약은 입력 해상도의 추가 확장을 방해한다.

또한 다단계 설명 생성 방법은 이미지에 나타난 장면만 설명할 수 있으며, 그 범위는 BLIP2와 원본 CC3M 주석에 포함된 세계 지식에 의해 제한된다. 예를 들어, 특정 국가의 장소 사진이 주어지면 시각적 측면은 설명할 수 있지만 그 장면이 실제로 어느 나라인지 식별하고 명시하는 능력은 부족하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 🌐 3-1. 고해상도가 일반화에 기여하는 이유

LMM 학습은 고해상도 이미지로부터 크게 이점을 얻는데, 높은 해상도가 이러한 모델들로 하여금 더 미묘한 시각적 세부 정보를 감지하여 객체, 상호관계, 이미지 내 더 넓은 맥락을 정확하게 인식하게 해주기 때문이다.

이는 일반화 성능 향상으로 이어지는 핵심 메커니즘이다:

$$\underbrace{\text{고해상도 입력}}_{\text{세밀한 시각 인식}} \rightarrow \underbrace{\text{풍부한 특징 추출}}_{\text{도메인 무관}} \rightarrow \underbrace{\text{일반화 성능 향상}}_{\text{다양한 도메인 적용}}$$

### 🌐 3-2. Multi-level Description이 일반화에 기여하는 이유

다단계 설명 생성 방법은 장면-객체 연관성에 대한 문맥을 풍부하게 한다. 이 두 가지 전략은 생성된 데이터로부터의 더욱 효과적인 학습을 보장하는데, 높은 해상도는 시각 정보를 더 상세하게 캡처하고 이것이 다시 포괄적 설명의 효과성을 높여준다.

- **데이터 다양성 확보:** 자동 생성된 풍부한 설명은 수작업 레이블보다 훨씬 다양한 시나리오를 커버한다.
- **문맥 학습 강화:** 장면-객체-관계의 3단계 기술은 모델이 새로운 도메인에서도 계층적 추론을 가능하게 한다.
- **도메인 이전(Domain Transfer):** 문서 이해, 자연 장면, 밀집 텍스트 등 서로 다른 도메인 전반에 걸쳐 일관된 성능을 보인다.

### 🌐 3-3. Patch-wise 처리의 분산 일반화

$$P(\text{정확한 예측} \mid I_{\text{고해상도}}) > P(\text{정확한 예측} \mid I_{\text{저해상도}})$$

패치 단위의 독립적 처리와 전역 특징의 결합은 서로 다른 해상도와 종횡비를 가진 이미지에 대한 강인성(robustness)을 높인다. 이 방법은 질문에 답할 때 목표물 간의 관계를 더 효과적으로 추론하는 뛰어난 능력을 보여주며, 이는 더욱 포괄적이고 통찰력 있는 결과를 제공한다.

### 🌐 3-4. 학습 효율성 기반 일반화

각 패치는 LoRA 조정과 학습 가능한 시각적 리샘플러로 향상된 정적 시각 인코더에 의해 독립적으로 처리된다. 이 기술은 광범위한 사전 학습의 필요성을 우회하면서 기존 LMM을 활용한다. 이는 적은 학습 데이터로도 다양한 새로운 도메인에 적용 가능함을 의미한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 📌 4-1. 연구에 미치는 영향

#### (A) 고해상도 멀티모달 연구의 표준화

Monkey의 후속 연구로 [ICLR 2025] Mini-Monkey (다중 스케일 적응형 크롭핑), [TPAMI 2026] TextMonkey (OCR-Free 문서 이해 모델), [NeurIPS 2024] MoE Jetpack 등 다양한 파생 연구들이 등장했다. 이는 Monkey가 고해상도 멀티모달 연구의 핵심 기반이 되었음을 보여준다.

#### (B) 패치 분할 패러다임의 확산

패치 분할 방법은 고해상도 이미지를 패치로 분할하고 저해상도 인코더를 재활용한다. Monkey와 SPHINX는 대형 이미지를 더 작은 패치로 분할하여 서브이미지와 다운샘플된 고해상도 이미지를 이미지 인코더로 전달하며, 서브이미지와 저해상도 이미지가 각각 지역 및 전역 특징을 포착한다.

이 패러다임은 LLaVA-NeXT 등 후속 연구에 광범위하게 채택되었다.

#### (C) 데이터 자동 생성 패러다임의 확립

Monkey의 다단계 설명 자동 생성 방법은 LMM 학습 데이터 구축의 새로운 패러다임을 제시하여, 비용이 많이 드는 수동 레이블링 없이도 고품질 학습 데이터를 확보하는 방법론에 영향을 미쳤다.

---

### 📌 4-2. 향후 연구 시 고려할 점

| 고려 사항 | 설명 |
|-----------|------|
| **패치 수 한계 극복** | 현재 최대 6개의 패치로 제한되어 있어 해상도 확장에 한계가 있으므로, 더 긴 컨텍스트를 처리할 수 있는 LLM 백본이나 효율적 토큰 압축 기법 연구가 필요하다. |
| **세계 지식 통합** | 다단계 설명 생성의 범위가 BLIP2와 CC3M 주석의 세계 지식에 제한되어 있으므로, 외부 지식베이스와의 연동 또는 더 강력한 캡셔닝 모델의 활용이 필요하다. |
| **동적 해상도 적응** | 이미지별 최적 해상도를 동적으로 결정하는 메커니즘 연구 (Mini-Monkey에서 일부 해결) |
| **다국어 및 다문화 일반화** | 특정 문화권 지식이 필요한 장면에서의 성능 향상 |
| **비디오 도메인 확장** | 고해상도 프레임 처리 기술을 비디오 이해로 확장 |
| **경량화 연구** | LoRA 어댑터 기반이지만 추론 시 패치별 처리 비용 최소화 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 해상도 처리 방법 | 특징 |
|------|------|----------------|------|
| **CLIP** (OpenAI) | 2021 | $224 \times 224$ 고정 | 대규모 이미지-텍스트 사전학습 |
| **Flamingo** (DeepMind) | 2022 | Perceiver Resampler | 시각-언어 퓨샷 학습 |
| **BLIP-2** (Salesforce) | 2023 | $224 \times 224$ | Q-Former로 LLM 연결 |
| **LLaVA** | 2023 | $336 \times 336$ | 간단한 MLP projection |
| **Qwen-VL** | 2023 | $448 \times 448$ | 커리큘럼 러닝 |
| **🐒 Monkey** | 2023/2024 | **$1344 \times 896$** (패치 분할) | **처음부터 학습 불필요, 다단계 설명** |
| **LLaVA-NeXT** | 2024 | 동적 그리드 패치 | Monkey에서 영향받은 패치 분할 |
| **Mini-Monkey** | 2025 | 다중 스케일 적응형 크롭핑 | Monkey의 경량화 파생 연구 |

Qwen-VL, PaLI-3, PaLI-X 등과 같은 커리큘럼 학습 방법들이 탐색되었으나, 상당한 학습 자원을 요구하고 여전히 더 큰 이미지 크기 처리에 어려움을 겪는다.

Monkey는 이러한 한계를 **학습 효율성을 유지하면서도** 해상도를 대폭 향상시키는 방식으로 극복하였다는 점에서 독보적인 기여를 한다.

---

## 📚 참고 자료 (References)

1. **arXiv 원문:** Zhang Li et al., "Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models," arXiv:2311.06607, 2023. https://arxiv.org/abs/2311.06607
2. **CVPR 2024 Open Access:** https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Monkey_Image_Resolution_and_Text_Label_Are_Important_Things_for_CVPR_2024_paper.pdf
3. **IEEE Xplore (CVPR 2024):** https://ieeexplore.ieee.org/document/10658022
4. **HuggingFace Paper Page:** https://huggingface.co/papers/2311.06607
5. **GitHub (공식 코드):** https://github.com/Yuliang-Liu/Monkey
6. **Semantic Scholar:** https://www.semanticscholar.org/paper/Monkey:-Image-Resolution-and-Text-Label-are-Things-Li-Yang/bf14244669d5505f63343d4365d99d24aa6c5e82
7. **ar5iv (HTML 버전):** https://ar5iv.labs.arxiv.org/html/2311.06607

> ⚠️ **정확도 관련 주의 사항:** 수식 표현 중 다단계 설명의 세부 손실 함수 구성이나 정확한 하이퍼파라미터 값 등 논문 원문 PDF에서만 확인 가능한 세부 정보는 공개된 arXiv HTML 및 공식 GitHub 정보를 바탕으로 재구성하였습니다. 정확한 수식 세부사항은 원문 PDF를 직접 확인하시기를 권장합니다.
