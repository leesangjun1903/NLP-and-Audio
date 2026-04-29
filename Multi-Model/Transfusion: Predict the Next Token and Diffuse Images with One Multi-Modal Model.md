
# Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

> **논문 정보**: Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., & Levy, O. (2024). *Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model.* arXiv:2408.11039. (Meta AI / ICLR 2025 제출)

---

## 1. 핵심 주장 및 주요 기여 요약

Transfusion은 이산(discrete) 데이터와 연속(continuous) 데이터를 동시에 처리하는 멀티모달 모델을 훈련하는 레시피로, 언어 모델링 손실 함수(다음 토큰 예측)와 diffusion을 결합하여 혼합 모달리티 시퀀스에 대해 단일 트랜스포머를 훈련합니다.

### 주요 기여 (3가지)

| # | 기여 | 내용 |
|---|------|------|
| 1 | **단일 모델 통합** | 텍스트(이산)와 이미지(연속)를 하나의 트랜스포머로 통합 |
| 2 | **스케일링 법칙 확립** | 다양한 단일/크로스 모달 벤치마크에서 스케일링 법칙 수립 |
| 3 | **모달리티 특화 레이어** | 인코딩/디코딩 레이어로 이미지를 16패치까지 압축 |

여러 Transfusion 모델을 최대 7B 파라미터까지 텍스트·이미지 혼합 데이터로 처음부터 사전 학습하여 스케일링 법칙을 수립하였으며, 이미지를 양자화(quantize)하여 언어 모델로 이산 토큰을 학습하는 것보다 Transfusion이 훨씬 더 잘 확장됨을 보여주었습니다.

---

## 2. 논문 상세 분석

### 2-1. 해결하고자 하는 문제

인공지능은 텍스트 생성과 이미지 생성 모두에서 큰 발전을 이루었지만, 이 두 가지 능력은 서로 다른 아키텍처를 가진 분리된 모델에 주로 존재해 왔습니다. 텍스트 생성은 시퀀스에서 다음 토큰을 예측하는 언어 모델에 의존하는 반면, 이미지 생성은 노이즈를 점진적으로 일관된 이미지로 변환하는 확산 모델을 사용합니다. 이 두 접근 방식을 품질을 타협하지 않고 두 모달리티를 모두 처리하는 단일 통합 모델로 결합하는 것은 AI 연구의 과제였습니다.

**기존 접근법의 한계:**

기존에는 언어 모델을 확장해 확산 모델을 도구로 사용하거나(DreamLLM, GILL 등), 연속 모달리티를 양자화(VQ-VAE 등)하여 표준 언어 모델로 이산 토큰을 학습하는 방법이 주로 사용되었습니다. 후자는 모델 아키텍처를 단순화하지만 정보 손실이 발생한다는 단점이 있습니다.

본 연구는 단일 모델이 이산 텍스트 토큰을 예측하는 동시에 연속 이미지를 확산시키도록 훈련함으로써, **정보 손실 없이** 두 모달리티를 완전히 통합할 수 있음을 보여줍니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (1) 데이터 표현

표준 임베딩 레이어는 텍스트 토큰을 벡터로 변환하고, 패치화(patchification) 레이어는 각 이미지를 패치 벡터 시퀀스로 표현합니다. 텍스트 토큰에는 **인과적 어텐션(causal attention)** 을 적용하고, 이미지 패치에는 **양방향 어텐션(bidirectional attention)** 을 적용합니다.

혼합 시퀀스를 처리하기 위해 텍스트 토큰과 이미지 패치를 단일 시퀀스로 결합하며, 특수 토큰인 BOI(Beginning of Image)와 EOI(End of Image)를 추가하여 이미지 데이터의 시작과 끝을 나타내어 모델이 두 모달리티를 구분할 수 있도록 합니다. 이를 통해 모델은 서로 다른 처리 모드 간에 전환하는 시점을 파악합니다.

#### (2) 학습 목적 함수

Transfusion의 핵심은 **두 가지 손실 함수를 결합**하는 것입니다:

**① 언어 모델링 손실 (텍스트)**

$$\mathcal{L}_{LM} = -\sum_{i \in \text{text}} \log P(x_i \mid x_{<i})$$

**② Diffusion 손실 (이미지)**

DDPM(Ho et al., 2020) 방식의 노이즈 예측 손실:

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right) \right\|^2 \right]$$

여기서 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$

**③ 통합 손실 함수**

학습 과정에서 Transfusion은 텍스트 토큰에 대한 언어 모델링 손실과 이미지 패치에 대한 확산 손실의 두 가지 손실 함수를 적용합니다. 이 손실들은 어느 모달리티도 훈련을 지배하지 않도록 균형 계수(balancing coefficient)와 결합됩니다. 이 계수는 각 손실의 상대적 중요도를 제어하여 어느 한 모달리티가 다른 모달리티를 압도하지 않으면서 텍스트와 이미지를 효과적으로 생성하도록 학습할 수 있게 합니다.

$$\mathcal{L}_{\text{Transfusion}} = \mathcal{L}_{LM} + \lambda \cdot \mathcal{L}_{\text{diff}}$$

여기서 $\lambda$는 두 손실 간의 균형을 조정하는 하이퍼파라미터입니다.

#### (3) VAE를 이용한 잠재 표현

이미지는 사전 훈련된 VAE를 통해 잠재 표현으로 변환하고, 이를 다시 패치 표현으로 변환하기 위해 단순 선형 레이어 또는 U-Net 다운 블록을 사용합니다.

이미지 $\mathbf{x}$의 잠재 표현:

$$\mathbf{z} = \text{Encoder}_{\text{VAE}}(\mathbf{x}), \quad \mathbf{x} \approx \text{Decoder}_{\text{VAE}}(\mathbf{z})$$

패치 표현:

$$\mathbf{p}_i = \text{LinearProj}(\mathbf{z}_{i \times P : (i+1) \times P}), \quad i = 1, \ldots, N$$

---

### 2-3. 모델 구조

Transfusion 아키텍처는 멀티모달 콘텐츠를 처리하고 생성하기 위해 함께 작동하는 여러 핵심 구성 요소로 이루어져 있습니다. 주요 구성 요소는 텍스트와 이미지 모두의 처리를 담당하는 트랜스포머입니다. 텍스트 처리를 위해 모델은 임베딩 레이어를 사용하여 각 입력 토큰을 고차원 벡터로 변환합니다. 이 벡터들은 표준 언어 모델링 방식을 따라 인과적 어텐션을 사용하는 트랜스포머 레이어에서 처리됩니다.

**전체 구조:**

```
[텍스트 임베딩]  →  Causal Attention  →  [다음 토큰 예측]
[이미지 VAE→패치] →  Bidirectional Attention  →  [노이즈 예측 (Diffusion)]
       ↑___________단일 Transformer 공유 가중치____________↑
```

7B 트랜스포머에 U-Net 다운/업 레이어(0.27B 파라미터)를 추가하여 2T 토큰(1T 텍스트 토큰과 약 5 에포크의 6억 9천 2백만 개 이미지 및 캡션, 총 1T 패치/토큰)으로 처음부터 학습되었습니다.

**모달리티 특화 레이어:**

모달리티 특화 인코딩 및 디코딩 레이어를 도입함으로써 Transfusion 모델의 성능을 더욱 향상시킬 수 있으며, 각 이미지를 단 16개의 패치로 압축할 수도 있습니다.

**추론 알고리즘:**

추론을 위해 언어 모델의 텍스트 생성 표준 관행과 확산 모델의 이미지 생성을 결합한 디코딩 알고리즘을 도입합니다.

---

### 2-4. 성능 향상

Chameleon의 이산화 접근법(Chameleon Team, 2024)과의 통제된 비교에서 Transfusion 모델이 모든 모달리티 조합에서 더 잘 확장됨을 보여주었습니다.

텍스트-이미지 생성에서 Transfusion은 FID와 CLIP 점수 모두에서 측정했을 때 Chameleon 접근법의 1/3 미만의 연산으로 더 나은 성능을 보였습니다.

GenEval 벤치마크에서 본 모델은 DALL-E 2 및 SDXL과 같은 인기 있는 모델을 능가하였으며, 이러한 이미지 생성 모델들과 달리 텍스트도 생성할 수 있으며 텍스트 벤치마크에서 Llama 1과 동일한 수준의 성능에 도달했습니다.

7B 파라미터와 2T 멀티모달 토큰으로 Transfusion 레시피를 확장하면 유사한 규모의 확산 모델 및 언어 모델과 동등한 수준으로 이미지와 텍스트를 생성할 수 있는 모델이 만들어져 양쪽 세계의 이점을 모두 취할 수 있습니다.

**성능 요약표:**

| 벤치마크 | Transfusion (7B) | 비교 모델 |
|---------|-----------------|---------|
| GenEval | DALL-E 2, SDXL 능가 | 이미지 특화 모델 |
| 텍스트 벤치마크 | Llama 1 동급 | 텍스트 특화 모델 |
| 연산 효율 | Chameleon 대비 1/3 연산으로 동등 | Chameleon |

---

### 2-5. 한계

논문 자체에서 명시된 한계 및 후속 연구에서 발견된 한계:

1. **학습 데이터 제한**: 텍스트 50%, 이미지 50% 데이터로 시연하였으나, 오디오·비디오 등 추가 모달리티 확장은 미검토

2. **연속 모달리티에서의 이해 성능**: 이러한 딜레마는 Transfusion과 유사한 모델이 [이미지 이해 태스크에서] 잘 수행하는 것을 방해합니다. (MMAR, CVPR 2025)

3. **이산 확산의 미성숙**: 이산 텍스트 생성에 대한 확산 접근법은 아직 표준 자기회귀 언어 모델의 성능과 규모를 달성하지 못했습니다. 이 방향의 미래 연구가 단일 모델에서 이산 및 연속 모달리티를 융합하는 새로운 방법을 열 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### (1) 스케일링을 통한 일반화

실험 결과 Transfusion은 효율적으로 확장되며, 파라미터 공유 비용이 거의 없거나 전혀 없으면서 모든 모달리티의 생성을 가능하게 합니다.

### (2) 정보 손실 없는 연속 표현 유지

Chameleon이 이미지를 이산화하여 토큰으로 처리하는 반면, Transfusion은 이미지를 연속 공간에 유지하여 양자화 정보 병목 현상을 제거합니다. 이는 이미지의 세밀한 특징을 보존하여, 보지 못한 이미지 유형에 대한 일반화 능력을 높입니다.

### (3) 모달리티 특화 레이어의 효과

모달리티 특화 인코딩 및 디코딩 레이어를 도입함으로써 이미지를 단 16개의 패치로 압축하면서도 성능이 향상됩니다. 이는 효율적인 표현 학습이 일반화에 기여함을 시사합니다.

### (4) 크로스 모달 전이(Cross-modal Transfer)

단일 트랜스포머에서 언어와 비전을 공동 학습함으로써, 텍스트 지식이 이미지 이해를 돕고 시각적 이해가 언어 추론을 강화하는 **양방향 전이 학습**이 가능합니다:

$$\text{일반화 성능} \propto \underbrace{\text{텍스트}→\text{이미지 전이}}_{\text{언어적 컨텍스트}} + \underbrace{\text{이미지}→\text{텍스트 전이}}_{\text{시각적 이해}}$$

이 방향의 미래 연구가 단일 모델에서 이산 및 연속 모달리티를 융합하는 새로운 방법을 열 수 있습니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

언어 모델은 이산 모달리티에서 다음 토큰 예측 목표로 훈련되어 지배적인 위치를 차지하고 있으며, 확산 모델(Ho et al., 2020; Rombach et al., 2022)과 그 일반화(Lipman et al., 2022)는 연속 모달리티 생성의 최신 기술입니다.

| 모델 | 연도 | 접근법 | Transfusion 대비 |
|------|------|--------|-----------------|
| **DALL-E** (OpenAI) | 2021 | 이미지 이산화 + AR | 정보 손실 발생 |
| **Stable Diffusion** (Rombach et al.) | 2022 | 순수 확산 | 텍스트 생성 불가 |
| **Chameleon** (Meta) | 2024 | 전체 이산화(VQ) | Transfusion 1/3 연산 필요 |
| **LMFusion** | 2024 | 사전학습 LLM 활용 | 이미지 이해 +20%, 생성 +3.6% |
| **MMAR** (CVPR 2025) | 2025 | 경량 확산 헤드 + AR | 이미지 이해 개선 |

LMFusion은 사전 훈련된 Llama-3 8B 모델로 아키텍처를 초기화하고 Transfusion과 동일한 이미지 데이터로 계속 학습하였으며, Transfusion 대비 이미지 이해에서 20%, 이미지 생성에서 3.6% 향상을 달성했습니다.

MMAR은 자기회귀 백본 모델 위에 경량 확산 헤드를 적용하여 확산 프로세스를 분리하는 새로운 멀티모달 자기회귀(Multi-Modal Auto-Regressive) 확률 모델링 프레임워크를 제안합니다.

자기회귀 모델의 추론 및 텍스트 생성 강점과 고품질 이미지 합성을 위한 확산 기반 모델의 견고성을 결합하는 방향으로 발전하고 있으며, 자기회귀 생성을 위해 이미지를 효과적으로 토크나이징하는 방법은 여전히 핵심 미해결 과제입니다.

---

## 5. 연구에 미치는 영향과 앞으로 고려할 점

### 5-1. 연구에 미치는 영향

**① 멀티모달 통합 패러다임 전환**

Transfusion은 언어 모델링과 확산을 단일 모델로 성공적으로 통합함으로써 멀티모달 AI 분야에서 중요한 발전을 이루었습니다. 이는 기존의 "모달리티별 전문 모델" 패러다임에서 "단일 통합 모델" 패러다임으로의 전환을 촉진합니다.

**② 스케일링 법칙의 확장**

최대 7B 파라미터까지의 Transfusion 모델을 텍스트·이미지 혼합 데이터로 사전 학습하여 다양한 단일 및 크로스 모달 벤치마크에 대한 스케일링 법칙을 수립하였으며, 이는 향후 대규모 멀티모달 모델 설계의 기준점이 됩니다.

**③ 후속 연구의 기반**

후속 구현에서는 Black Forest Labs의 Flux 성공에 힘입어 확산 대신 플로우 매칭(flow matching)을 대체하는 시도가 이루어지고 있으며, 임의의 수의 모달리티로 확장하는 연구도 진행 중입니다.

**④ 연속 표현의 중요성 입증**

코드북 크기의 제한 없이 연속 이미지 토크나이저는 이미지를 훨씬 효율적으로 압축할 수 있으며, 최근 연구에서는 128×128 이미지 패치를 단일 연속 이미지 토큰으로 압축하는 것이 가능하여 512×512 이미지당 단 16개의 토큰만이 필요하다는 것을 보여주었습니다.

---

### 5-2. 앞으로 연구 시 고려할 점

**① 다중 모달리티 확장**

현재 텍스트-이미지에 집중되어 있으나, 멀티모달 생성 모델은 텍스트나 코드와 같은 이산적 요소와 이미지, 오디오, 비디오 데이터와 같은 연속적 요소 모두를 인식, 처리, 생성할 수 있어야 합니다. 따라서 오디오·비디오 모달리티로의 확장이 핵심 연구 과제입니다.

**② 손실 함수 가중치($\lambda$) 최적화**

두 손실의 균형을 맞추는 $\lambda$ 값의 자동 조정 메커니즘 개발이 필요합니다:

$$\lambda^* = \arg\min_\lambda \mathcal{L}_{\text{val}}(\mathcal{L}_{LM} + \lambda \cdot \mathcal{L}_{\text{diff}})$$

**③ 이미지 이해 성능 강화**

이러한 딜레마(확산 노이즈 수준과 이미지 이해 성능 간의 트레이드오프)는 Transfusion과 유사한 모델이 [이해 태스크에서] 잘 수행하는 것을 방해합니다. 이해-생성 동시 최적화를 위한 새로운 목적 함수 설계가 필요합니다.

**④ Flow Matching으로의 전환**

이 구현에서는 Black Forest Labs의 Flux 성공에 힘입어 확산 대신 플로우 매칭을 대체하는 방향을 탐색하고 있습니다. Flow Matching은 더 안정적인 학습 궤적을 제공할 수 있습니다:

$$\frac{d\mathbf{x}}{dt} = v_\theta(\mathbf{x}_t, t), \quad \mathbf{x}_0 \sim \mathcal{N}(0, I), \quad \mathbf{x}_1 \sim p_{\text{data}}$$

**⑤ 추론 효율성 개선**

Diffusion 기반 이미지 생성은 여러 디노이징 스텝이 필요하므로, 텍스트 AR 생성(단일 패스)과의 속도 불균형을 해소하는 효율적 샘플링 기법 연구가 필요합니다.

**⑥ 인스트럭션 튜닝 및 RLHF 적용**

파인튜닝된 7B Transfusion 모델로 이미지 편집도 가능함을 보였으나, 사람의 선호도를 반영한 RLHF 기반 멀티모달 정렬 연구가 추가적으로 필요합니다.

---

## 📚 참고 자료 및 출처

| # | 출처 | 링크 |
|---|------|------|
| 1 | **arXiv 원문 (v1)** | https://arxiv.org/abs/2408.11039 |
| 2 | **arXiv HTML 전문** | https://arxiv.org/html/2408.11039v1 |
| 3 | **Meta AI 공식 발표** | https://ai.meta.com/research/publications/transfusion-predict-the-next-token-and-diffuse-images-with-one-multi-modal-model/ |
| 4 | **OpenReview (ICLR 2025)** | https://openreview.net/forum?id=SI2hI0frk6 |
| 5 | **Hugging Face Papers** | https://huggingface.co/papers/2408.11039 |
| 6 | **Semantic Scholar** | https://www.semanticscholar.org/paper/Transfusion:-...-Zhou-Yu/5b3991fe7d8f6fc0a7fbd42938e2988ea37efe14 |
| 7 | **PyTorch 구현 (lucidrains)** | https://github.com/lucidrains/transfusion-pytorch |
| 8 | **ResearchGate PDF** | https://www.researchgate.net/publication/383267014 |
| 9 | **LMFusion (OpenReview)** | https://openreview.net/pdf/14b38a139b47e15b756a5e1cdfa7f9a73fdc10ed.pdf |
| 10 | **MMAR (CVPR 2025)** | https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_MMAR_... |
| 11 | **Unified Multimodal Survey (arXiv 2025)** | https://arxiv.org/abs/2505.02567 |
| 12 | **Medium 해설 블로그** | https://medium.com/@EleventhHourEnthusiast/transfusion-... |

> ⚠️ **정확도 관련 고지**: 본 답변에서 수식 중 일부(특히 $\mathcal{L}\_{\text{Transfusion}} = \mathcal{L}\_{LM} + \lambda \cdot \mathcal{L}_{\text{diff}}$ 형태)는 논문의 개념적 설명을 기반으로 표준적인 수식 형태로 재구성한 것이며, 논문 원문의 정확한 표기와 일부 다를 수 있습니다. 정확한 수식은 arXiv 원문(2408.11039)을 직접 확인하시기 바랍니다.
