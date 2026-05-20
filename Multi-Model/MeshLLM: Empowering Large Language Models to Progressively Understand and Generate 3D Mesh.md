
# MeshLLM: Empowering Large Language Models to Progressively Understand and Generate 3D Mesh

> **논문 정보**
> - **저자**: Shuangkang Fang, I-Chao Shen, Yufeng Wang, Yi-Hsuan Tsai, Yi Yang, Shuchang Zhou, Wenrui Ding, Takeo Igarashi, Ming-Hsuan Yang
> - **소속**: Beihang University, The University of Tokyo, Atmanity Inc., StepFun Inc., UC Merced
> - **게재**: ICCV 2025, pp. 14061–14072
> - **arXiv**: [2508.01242](https://arxiv.org/abs/2508.01242)
> - **코드**: [GitHub - Fangkang515/MeshLLM](https://github.com/Fangkang515/MeshLLM)

---

## 1. 핵심 주장 및 주요 기여 요약

MeshLLM은 LLM이 텍스트로 직렬화된 3D 메시를 이해하고 생성할 수 있도록 하는 새로운 프레임워크로, 기존 방법들의 핵심 한계—LLM 토큰 길이에 맞춘 데이터셋 규모의 부족, 그리고 메시 직렬화 과정에서의 3D 구조 정보 손실—를 직접적으로 해결합니다.

### 주요 기여 3가지

| # | 기여 내용 |
|---|-----------|
| ① | **Primitive-Mesh 분해 전략**: 3D 메시를 의미 있는 구조적 하위 단위로 분리 |
| ② | **대규모 데이터셋 구축**: 기존 대비 약 50배 규모인 1,500k+ 샘플 생성 |
| ③ | **Progressive Training Paradigm**: Vertex→Face→Assembly 순서로 단계적 학습 |

MeshLLM은 텍스트로 직렬화된 메시를 LLM에 효과적으로 주입하는 방법을 제안하여, 보다 자연스러운 대화형 상호작용을 통해 3D 메시의 이해와 생성을 가능하게 합니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존 방법들은 LLM의 토큰 길이에 맞추기 위해 데이터셋 규모가 크게 제한되었고, 메시 직렬화 과정에서 3D 구조 정보가 손실되는 문제를 갖고 있었습니다.

구체적으로 두 가지 핵심 문제가 있습니다.

**문제 1: 데이터 규모 부족**
- LLM은 긴 시퀀스를 처리하기 어려워, 전체 3D 메시를 직렬화하면 토큰 수가 폭발적으로 증가
- 기존 LLaMA-Mesh 등의 방법들은 학습 데이터 규모가 매우 제한적이었음

**문제 2: 구조 정보 손실**
- 단순히 vertex 좌표를 나열하는 텍스트 직렬화 방식은 메시의 위상(topology)과 공간적 연결성(spatial connectivity)을 LLM이 파악하기 어렵게 만듦

---

### 2-2. 제안하는 방법

#### (A) Primitive-Mesh 분해 전략

3D 메시를 구조적으로 의미 있는 하위 단위(Primitive-Mesh)로 분할하는 전략을 도입하여, 1,500k+ 샘플의 대규모 데이터셋을 생성할 수 있었고, 이는 기존 방법 대비 약 50배 큰 규모로 LLM의 스케일링 법칙(Scaling Law)에 더욱 부합합니다.

분해 방식은 크게 두 종류로 나뉩니다:

- **Geometry-based Primitive-Mesh**: 기하학적 방법으로 분해
- **Semantic-based Primitive-Mesh**: Zero-shot 3D 파트 분할 방법인 SamPart3D를 활용하여 의미론적 수준의 Primitive-Mesh 데이터셋을 구축하며, SamPart3D는 Objaverse에서 사전학습된 3D 백본 네트워크를 이용해 시각적 특징을 추출하고, 경량 MLP를 활용해 2D 분할 마스크를 스케일 조건 그룹으로 정제하여 3D 포인트 클라우드 클러스터링을 수행합니다.

또한 100k개 이상의 고품질 의미론적으로 분할된 메시를 포함하는 세심하게 큐레이션된 데이터셋을 구축하여 LLM의 메시 구조 이해 및 추론 능력을 향상시켰습니다.

#### (B) Progressive Training Paradigm (단계적 학습 패러다임)

구축된 데이터셋을 기반으로, vertex에서 face, mesh assembly까지 메시를 계층적으로 모델링하는 구조화된 학습 패러다임을 제안하여 LLM이 3D 세계를 효과적으로 인식할 수 있게 합니다.

학습 단계를 정리하면 아래와 같습니다:

| 학습 단계 | 주요 태스크 |
|-----------|------------|
| **Phase 1** | Vertex-to-Face 예측 (face connectivity inference) |
| **Phase 2** | Local Mesh Assembly 학습 |
| **Phase 3** | 전체 메시 생성 및 이해(대화 포함) |

Progressive training은 구조적 지식을 먼저 습득한 후 복잡한 생성·대화 태스크를 다루도록 하며, 이전 학습 단계와 ultra-chat 데이터의 교차 샘플링을 통해 catastrophic forgetting을 방지합니다.

#### (C) Face Connectivity Inference (수식 포함)

vertex로부터 face connectivity를 추론하는 방법과 local mesh assembly 학습 전략을 제안하여, LLM이 메시 위상과 공간 구조를 포착하는 능력을 크게 향상시킵니다.

3D 메시의 OBJ 포맷 텍스트 직렬화는 아래와 같이 표현됩니다:

$$\mathcal{M} = \{V, F\}, \quad V = \{v_i \in \mathbb{R}^3\}_{i=1}^{N_v}, \quad F = \{f_j\}_{j=1}^{N_f}$$

각 vertex $v_i = (x_i, y_i, z_i)$는 정규화 후 정수형 토큰으로 양자화되며:

$$\hat{v}_i = \text{round}\left(\frac{v_i - v_{\min}}{v_{\max} - v_{\min}} \times (B-1)\right), \quad B = 128 \text{ (quantization bins)}$$

Face connectivity 예측 태스크는 주어진 vertex 집합 $V^{(k)}$에서 해당 face $F^{(k)}$를 예측하는 조건부 생성 문제로 정의됩니다:

$$P(F^{(k)} \mid V^{(k)}) = \prod_{t=1}^{|F^{(k)}|} P(f_t \mid f_{ < t}, V^{(k)})$$

Local mesh assembly 태스크는 여러 Primitive-Mesh 조각들을 하나의 완전한 메시로 조합하는 학습으로 표현할 수 있습니다:

```math
\mathcal{M}_{\text{full}} = \text{Assemble}\left(\{\mathcal{M}^{(k)}\}_{k=1}^{K}\right)
```

> ⚠️ **주의**: 위 수식 중 논문에 명시된 내용은 개념적 정의 부분이며, 구체적인 수치 파라미터(예: $B=128$)는 논문의 세부 내용에서 가져온 것입니다. 전체 수식 체계의 세부 표기는 논문 원문(arXiv:2508.01242)에서 직접 확인하시기를 권장합니다.

---

### 2-3. 모델 구조

기반 LLM으로 LLaMA-8B-Instruct를 사용하며, 구축된 데이터를 기반으로 80억 개의 전체 파라미터를 파인튜닝합니다.

모델은 Objaverse-XL과 ShapeNet에서 수집한 데이터로 학습되며, 광범위한 데이터 증강과 함께 최대 컨텍스트 길이 8,192 토큰을 사용합니다.

메시 어셈블리, 이해 및 생성 태스크의 경우 PolyGen과 유사한 절차를 따르며, 3,000개 미만의 face를 가진 메시에 대해 평면 단순화(planar simplification)를 적용하고, 최종 전체 메시 표현의 face 수를 800개로 제한하여 LLM의 최대 토큰 길이와의 호환성을 보장합니다.

모델 구조를 요약하면 다음과 같습니다:

```
Input Text / Mesh Description
        ↓
[Primitive-Mesh Tokenization]  ← 3D Mesh → OBJ 텍스트 직렬화
        ↓
[LLaMA-8B-Instruct Backbone]  ← Full Parameter Fine-Tuning
        ↓
  ┌─────────────────────────────┐
  │ Phase 1: Vertex→Face 예측   │
  │ Phase 2: Local Assembly     │
  │ Phase 3: 생성 + 대화        │
  └─────────────────────────────┘
        ↓
   Output: 3D Mesh / Text Description
```

---

### 2-4. 성능 향상

정량적 결과(Table 1)에서 MeshLLM은 모든 생성 메트릭—MMD, COV, 1-NNA, FID, KID—에서 LLaMA-Mesh를 크게 능가하며, MeshXL 및 PolyGen과 같은 특화 모델에 근접하는 성능을 달성했습니다. 이는 Primitive-Mesh 전략이 LLM으로 하여금 고품질의 다양한 메시를 생성하도록 효과적으로 지원함을 보여줍니다.

이해 태스크에서도 BLEU-1, CIDEr, METEOR, ROUGE, CLIP 유사도 지표(Table 2)에서 LLaMA-Mesh를 능가하며, 보다 정확하고 유창한 설명을 생성합니다.

MeshLLM은 질문 답변 및 수학적 추론과 같은 고급 대화 능력을 유지하면서 LLM의 역량을 3D 메시 도메인으로 확장하며, 이를 통해 자연스럽고 직관적인 언어 상호작용으로 3D 메시를 이해하고 생성할 수 있는 기반을 마련합니다.

성능 비교 요약:

| 모델 | 메시 생성 | 메시 이해 | 대화 능력 |
|------|----------|----------|----------|
| LLaMA-Mesh | 기준 | 기준 | ✅ |
| MeshXL | ✅ 우수 | ❌ 없음 | ❌ 없음 |
| PolyGen | ✅ 우수 | ❌ 없음 | ❌ 없음 |
| **MeshLLM** | **✅ LLaMA-Mesh 대비 크게 향상** | **✅ 향상** | **✅ 유지** |

MeshXL과 달리 MeshLLM은 메시를 이해하고 사용자와 대화할 수 있는 능력을 갖추고 있습니다.

---

### 2-5. 한계점

논문에서 확인된 한계점 및 관련 선행 연구 LLaMA-Mesh의 언급 사항:

1. **Face 수 제한**: 최종 전체 메시 표현의 face 수를 800개로 제한하기 때문에, 매우 복잡하고 세밀한 메시(고해상도 3D 자산)를 생성하는 데는 여전히 제약이 있습니다.

2. **컨텍스트 길이 제약**: 최대 컨텍스트 길이가 8,192 토큰으로 제한되어 있어, 초복잡 구조의 3D 객체 전체를 직렬화하기에는 여전히 한계가 있습니다.

3. **데이터 구축 비용**: Semantic-based Primitive-Mesh 데이터셋 생성을 위해 128개의 A800 GPU를 사용하여 약 3일이 소요되어, 데이터 구축에 매우 큰 계산 자원이 필요합니다.

4. **언어 능력 저하 위험**: 선행 연구인 LLaMA-Mesh에서도 지적된 바와 같이, 파인튜닝 후 언어 능력이 약간 저하될 수 있으며, 이는 텍스트 인스트럭션 데이터셋에만 의존하는 것에서 기인할 수 있어 더 다양하고 고품질의 텍스트 인스트럭션 데이터셋을 포함하면 언어 능력 보존에 도움이 될 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

일반화 성능과 관련하여 MeshLLM은 다음과 같은 요소들이 일반화에 긍정적으로 기여합니다:

### (1) LLM 스케일링 법칙과의 정합성

1,500k+ 샘플의 대규모 데이터셋은 기존 방법 대비 약 50배 크며, 이는 LLM 스케일링 법칙의 원칙에 더욱 부합합니다. 스케일링 법칙에 따르면 데이터셋 규모와 모델 파라미터가 커질수록 성능이 지속적으로 향상되므로, 더 큰 LLM 백본과의 결합 시 일반화 성능이 추가적으로 향상될 가능성이 있습니다.

$$\mathcal{L}(N, D) \propto N^{-\alpha} + D^{-\beta}$$

(여기서 $N$은 파라미터 수, $D$는 데이터 규모, $\alpha, \beta$는 스케일링 지수)

### (2) 다양한 카테고리 학습

ShapeNet의 4가지 서브셋(의자, 테이블, 벤치, 램프)과 Objaverse-XL의 1,000개 샘플을 테스트셋으로 사용하여, 다양한 카테고리에 걸친 일반화 능력을 검증했습니다.

### (3) Primitive-Mesh의 다중 도메인 적용성

메시 분해 전략으로 1,500k+ Primitive-Mesh를 생성하여 학습 데이터셋 규모를 약 50배 확장하였으며, vertex-face 예측과 local mesh assembly 학습 태스크를 포함하는 MeshLLM 프레임워크를 통해 LLM의 3D 메시 구조 인식 능력을 향상시켰습니다.

### (4) Catastrophic Forgetting 방지

Progressive training은 구조적 지식을 먼저 습득한 후 복잡한 생성·대화 태스크에 도전하도록 하며, 이전 학습 단계와 ultra-chat 데이터의 교차 샘플링을 통해 catastrophic forgetting을 방지합니다. 이를 통해 새로운 3D 도메인 학습 시에도 기존 언어 능력을 유지하며 일반화됩니다.

### (5) 향후 일반화 향상 방향

- 더 큰 LLM 백본 (예: LLaMA-70B)으로의 확장
- 더 많은 카테고리 및 도메인(건축, 의료, 자동차 등)으로의 데이터 확대
- LLM과 3D 메시 도메인 간의 보다 심층적인 통합을 통해 강력한 멀티모달 지능형 에이전트 개발을 위한 새로운 관점을 연구 커뮤니티에 제공하는 방향으로 일반화 연구가 진행될 수 있습니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 주요 비교 논문들

| 논문 | 연도 | 방법 | 특징 | MeshLLM 대비 |
|------|------|------|------|-------------|
| **PolyGen** | 2020 | Transformer 기반 vertex/face 자동회귀 생성 | 최초의 신경망 기반 다각형 메시 생성 | 메시 이해·대화 불가 |
| **LLaMA-Mesh** | 2024 | LLM(LLaMA) + OBJ 텍스트 직렬화 파인튜닝 | 대화형 3D 생성 최초 시도 | 데이터셋 규모 제한, 구조 정보 손실 |
| **MeshXL** | 2025 | Neural Coordinate Field + 생성 모델 | 3D 기반 생성 모델로 고품질 메시 생성 | 메시 이해·대화 불가 |
| **MeshAnything** | 2024 | Autoregressive Transformer + Artist 메시 | 예술적 품질 메시 생성 특화 | 텍스트 이해·대화 불가 |
| **MeshAnything V2** | 2024 | Adjacent Mesh Tokenization | 토크나이제이션 개선 | 텍스트 이해·대화 불가 |
| **MeshLLM** | **2025** | **Primitive-Mesh + Progressive Training** | **생성+이해+대화 통합** | **–** |

LLaMA-Mesh는 3D 메시를 텍스트로 표현하고 파인튜닝함으로써 LLM이 3D 메시를 이해하고 생성하도록 하며, 단일 모델에서 3D와 텍스트 모달리티를 통합하여 대화형 3D 생성과 메시 이해를 가능하게 했습니다. 그러나 MeshLLM은 이를 데이터 규모와 구조 인식 능력에서 크게 뛰어넘습니다.

MeshXL은 생성적 3D 기반 모델을 위한 신경 좌표 필드 방법으로 NeurIPS 2025에 발표된 전문화된 생성 모델이나, MeshXL과 달리 MeshLLM은 메시를 이해하고 사용자와 대화할 수 있는 통합 능력을 갖추고 있습니다.

---

## 5. 향후 연구에 미치는 영향과 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

**① LLM의 3D 도메인 확장 패러다임 정립**

LLM이 3D 세계를 계층적으로 인식하는 구조화된 학습 패러다임을 제시하였으며, 이러한 발견이 LLM과 3D 메시 도메인 간의 보다 심층적인 통합을 촉진하여 강력한 멀티모달 지능형 에이전트 개발을 위한 새로운 관점을 제공하기를 기대하고 있습니다.

**② 데이터 증강 및 스케일링 방법론의 일반화**

Primitive-Mesh 분해 전략은 3D 메시를 구조적으로 의미 있는 하위 단위로 분할하여 기존 대비 약 50배 큰 1,500k+ 샘플의 대규모 데이터셋 생성을 가능하게 했으며, 이 접근 방식은 다른 3D 표현(예: NeRF, 포인트 클라우드)에도 응용될 수 있습니다.

**③ 멀티모달 에이전트 개발의 새로운 방향**

MeshLLM은 고급 대화 능력을 유지하면서 LLM의 역량을 3D 메시 도메인으로 확장하며, 자연스러운 언어 상호작용으로 3D 메시를 이해하고 생성할 수 있게 하여 LLM을 다목적이고 강력한 도구로 더욱 공고히 합니다.

---

### 5-2. 앞으로 연구 시 고려할 점

1. **토큰 길이 한계 극복**: 현재 face 수 800개 제한을 넘어 더욱 복잡한 메시를 다루기 위해, LLM의 context window 확장(예: 100K+ 토큰) 또는 계층적 압축 인코딩 방법 연구가 필요합니다.

2. **더 다양한 LLM 백본 적용**: LLaMA-Mesh 논문에서도 언급된 바와 같이, 더 큰 LLaMA 모델을 사용하면 성능이 더욱 향상될 것으로 예상되므로, MeshLLM도 더 큰 LLM 백본(70B, 405B)으로의 확장을 고려해야 합니다.

3. **시간적 일관성 및 동적 3D**: 현재 MeshLLM은 정적 메시에 집중하고 있으나, 4D(시간 축 포함) 또는 애니메이션 메시로의 확장이 중요한 연구 방향이 될 수 있습니다.

4. **텍스처 및 재질 통합**: 현재 제안 방법은 기하학(geometry)에 집중하고 있으며, UV 텍스처와 PBR(Physically Based Rendering) 재질 정보를 함께 처리하는 방향으로 확장할 필요가 있습니다.

5. **Catastrophic Forgetting 심층 연구**: Progressive training이 이전 단계와 ultra-chat 데이터의 교차 샘플링을 통해 catastrophic forgetting을 방지한다고 제안하지만, 더 다양한 도메인으로 확장 시 망각 문제에 대한 체계적 연구가 필요합니다.

6. **데이터 구축 비용 절감**: 현재 Semantic Primitive-Mesh 데이터셋 생성에 128개의 A800 GPU로 약 3일이 소요되는 만큼, 더 효율적인 데이터 파이프라인 또는 합성 데이터 생성 방법의 연구가 필요합니다.

7. **평가 지표 다양화**: MMD, COV, 1-NNA, FID, KID 외에도 실제 활용 가능성(manufacturability, editability, physical plausibility)을 평가하는 새로운 지표 개발이 필요합니다.

---

## 참고 자료 / 출처

1. **MeshLLM 논문 (주논문)**
   - Fang, S. et al. "MeshLLM: Empowering Large Language Models to Progressively Understand and Generate 3D Mesh." *ICCV 2025*, pp. 14061–14072.
   - arXiv: https://arxiv.org/abs/2508.01242
   - PDF: https://arxiv.org/pdf/2508.01242
   - HTML: https://arxiv.org/html/2508.01242v1

2. **ICCV 2025 공식 게재 페이지**
   - https://openaccess.thecvf.com/content/ICCV2025/html/Fang_MeshLLM_Empowering_Large_Language_Models_to_Progressively_Understand_and_Generate_ICCV_2025_paper.html

3. **GitHub 공식 코드**
   - https://github.com/Fangkang515/MeshLLM

4. **HuggingFace 논문 페이지**
   - https://huggingface.co/papers/2508.01242

5. **비교 대상 논문: LLaMA-Mesh**
   - Wang, Z. et al. "LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models." arXiv:2411.09595 (2024).
   - https://research.nvidia.com/labs/toronto-ai/LLaMA-Mesh/

6. **비교 대상 논문: MeshXL**
   - Chen, S. et al. "MeshXL: Neural Coordinate Field for Generative 3D Foundation Models." *NeurIPS 2025*, vol. 37, pp. 97141–97166.

7. **비교 대상 논문: MeshAnything**
   - Chen, Y. et al. "MeshAnything: Artist-created Mesh Generation with Autoregressive Transformers." arXiv:2406.10163 (2024).

8. **비교 대상 논문: MeshAnything V2**
   - Chen, Y. et al. "MeshAnything V2: Artist-created Mesh Generation with Adjacent Mesh Tokenization." arXiv:2408.02555 (2024).

9. **ChatPaper 분석**
   - https://chatpaper.com/paper/173227

10. **ResearchGate 논문 페이지**
    - https://www.researchgate.net/publication/394292910
