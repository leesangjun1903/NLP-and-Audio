
# Lance: Unified Multimodal Modeling by Multi-Task Synergy

> **논문 정보**
> - **제목:** Lance: Unified Multimodal Modeling by Multi-Task Synergy
> - **저자:** Fengyi Fu, Mengqi Huang, Shaojin Wu 외 10인 (ByteDance Intelligent Creation Lab)
> - **arXiv ID:** 2605.18678 (cs.CV), 발표일: 2026년 5월 18~19일
> - **GitHub:** https://github.com/bytedance/Lance
> - **프로젝트 홈:** https://lance-project.github.io

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Lance는 모델 용량 스케일링이나 텍스트-이미지 중심 설계에 의존하는 대신, **협력적 다중 작업(Multi-Task) 학습**을 통한 통합 멀티모달 모델링의 실용적 패러다임을 탐구합니다.

Lance의 핵심 발견은 **멀티태스크 시너지(Multi-Task Synergy)** 가 통합 멀티모달 모델링을 효과적으로 발전시킬 수 있다는 것이며, 이를 통해 다양한 작업들이 공유 프레임워크 내에서 서로를 강화할 수 있음을 보입니다.

### 주요 기여 (5가지)

| 번호 | 기여 내용 |
|------|-----------|
| ① | 이미지·비디오 이해/생성/편집을 단일 모델로 통합 |
| ② | Dual-Stream Mixture-of-Experts (MoE) 아키텍처 제안 |
| ③ | Modality-Aware Rotary Positional Encoding (MaPE) 도입 |
| ④ | 단계적 멀티태스크 학습 패러다임 (PT → CT → SFT) |
| ⑤ | 3B 경량 파라미터로 기존 오픈소스 통합 모델 대비 우수한 성능 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

대부분의 기존 방법(Chameleon, World, SEED, TokenFlow, Janus 등)은 여전히 텍스트-이미지 도메인이나 일부 작업 조합에만 제한되어, 이미지-비디오 이해 및 생성의 전체 공간을 충분히 탐구하지 못했습니다.

최근 통합 모델들(Show-o2, TUNA 등)은 점진적으로 비디오 도메인으로 확장했지만, 편집이나 주제 기반 생성과 같은 다양한 생성 작업들은 통합 멀티태스크 학습 내에서 체계적으로 최적화되기보다 다운스트림 파인튜닝 기술로만 도입되는 경우가 많았습니다.

또한, 단일 자기회귀(autoregressive) 정식화는 생성에 필요한 충실도(fidelity)와 추론에 필요한 추상화(abstraction)를 균형 있게 다루는 데 어려움을 겪습니다.

---

### 2-2. 제안하는 방법

#### (A) 두 가지 핵심 원칙

Lance는 두 가지 핵심 원칙을 기반으로 합니다: **통합 컨텍스트 모델링(Unified Context Modeling)** 과 **분리된 능력 경로(Decoupled Capability Pathways)**.

#### (B) 통합 컨텍스트 시퀀스 구성

통합 컨텍스트를 위해 Lance는 텍스트, 이미지, 비디오 등 모든 입력을 단일 공유 인터리브 멀티모달 시퀀스로 변환합니다. 텍스트 토큰은 Qwen2.5-VL 임베딩 레이어에서 오고, 이해 지향 시각 입력은 Qwen2.5-VL ViT 인코더가 컴팩트한 시맨틱 시각 토큰을 생성합니다. 생성 지향 시각 입력은 Wan2.2 3D 인과 VAE 인코더가 이미지와 비디오를 연속 잠재 표현으로 인코딩하며, 16× 공간 다운샘플링과 4× 시간 다운샘플링을 적용합니다. 이 이질적인 토큰 유형들(텍스트, 시맨틱 시각, 잠재 시각)이 모두 동일한 시퀀스 내에 존재합니다.

#### (C) Dual-Stream Mixture-of-Experts 구조

분리된 경로를 위해 Lance는 Qwen2.5-VL 3B에서 초기화된 **Dual-Stream MoE 아키텍처**를 사용합니다. **이해 전문가($\text{LLM}_\text{UND}$)** 는 텍스트 및 시맨틱 시각 토큰을 처리하여 멀티모달 추론 및 텍스트 생성 출력을 생성합니다. **생성 전문가($\text{LLM}_\text{GEN}$)** 는 시각 합성 및 편집을 위한 VAE 잠재 토큰을 처리합니다. 결정적으로, 두 전문가 모두 동일한 공유 인터리브 시퀀스에서 작동하여 컨텍스트를 공유하지만 동일한 파라미터를 두고 경쟁하지 않습니다.

#### (D) 학습 목적 함수

이해 전문가는 **Next-Token Prediction Loss** 로 학습되고, 생성 전문가는 연속 잠재 공간에서 **Flow Matching Objective** 로 학습됩니다. 두 손실 함수는 학습 전반에 걸쳐 구성 가능한 가중치로 결합됩니다.

수식으로 표현하면:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{und}} \cdot \mathcal{L}_{\text{NTP}} + \lambda_{\text{gen}} \cdot \mathcal{L}_{\text{FM}}$$

여기서:

- $\mathcal{L}\_{\text{NTP}} = -\sum_{t} \log P_{\theta}(x_t \mid x_{<t})$ : 이해 전문가의 **Next-Token Prediction Loss**
- $\mathcal{L}\_{\text{FM}} = \mathbb{E}\_{t, x_0, x_1}\left[\left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2\right]$ : 생성 전문가의 **Flow Matching Loss** ( $x_t = (1-t)x_0 + t x_1$ )
- $\lambda_{\text{und}}, \lambda_{\text{gen}}$ : 훈련 단계별로 조정 가능한 가중치

#### (E) Modality-Aware Rotary Positional Encoding (MaPE)

ViT 시맨틱 토큰, 깨끗한 VAE 조건 토큰, 노이즈가 있는 VAE 타깃 토큰을 동일한 시퀀스에서 처리하는 것은 미묘한 문제를 야기합니다. 표준 3D-RoPE는 시공간적 레이아웃만 기반으로 위치를 인코딩하여 이러한 토큰 그룹들을 구별할 방법이 없습니다.

이를 해결하기 위해 MaPE를 도입합니다:

$$\text{MaPE}(p, m) = \text{RoPE}(p) \oplus \text{ModalityOffset}(m)$$

**Modality-Aware Rotary Positional Encoding (MaPE)** 를 도입하여 이질적인 시각 토큰 간의 간섭을 완화하고 크로스태스크 정렬을 향상시킵니다.

또한 **Generalized 3D Causal Attention** 은 시퀀스를 모달리티별로 분리하여, 텍스트 토큰에는 인과적 어텐션(Causal Attention)을 적용하고 시각 토큰에는 양방향 어텐션(Bidirectional Attention)을 적용합니다.

---

### 2-3. 모델 구조 상세

```
입력 (텍스트 / 이미지 / 비디오)
        │
        ├── 텍스트 → Qwen2.5-VL Tokenizer
        ├── 이해용 시각 → ViT Encoder → Semantic Tokens
        └── 생성용 시각 → Wan2.2 3D Causal VAE → Latent Tokens
        │
        ▼
  [통합 인터리브 시퀀스] + MaPE 적용
        │
        ▼
  Generalized 3D Causal Attention
        │
        ├── LLM_UND (이해 전문가) → Text Output (NTP Loss)
        └── LLM_GEN (생성 전문가) → Visual Latent Output (Flow Matching Loss)
        │
        ▼
  출력: 텍스트 / 이미지 / 비디오
```

MaPE는 이미지와 비디오의 이질적 시각 토큰을 위해 설계된 회전 위치 인코딩으로, 모달리티 간 신호 간섭을 줄입니다.

---

### 2-4. 단계적 학습 전략

**사전훈련(PT) 단계** 는 약 10억 개의 이미지-텍스트 쌍과 1.4억 개의 비디오-텍스트 쌍을 사용하며, 1.5T 학습 토큰을 커버합니다. 이 단계는 기본적인 멀티모달 정렬 및 생성 능력을 확립합니다.

**단계적 멀티태스크 학습** 은 사전훈련(PT), 지속학습(CT), 지도파인튜닝(SFT)의 점진적 레시피를 사용하여 제한된 컴퓨팅 예산 내에서 멀티태스크 협력을 달성합니다.

대규모 쌍 학습이 먼저 핵심 생성 능력을 확립하고, 이후 토큰들은 주로 프롬프트 정렬, 시각적 충실도 및 시간적 일관성을 정제합니다. 더욱이, CT 단계는 순수 생성 데이터를 추가하지 않고 편집 및 명령 추종 데이터와 같은 멀티태스크 데이터를 주로 도입함에도 불구하고 네이티브 생성 능력을 향상시킵니다.

---

### 2-5. 성능 향상

실험 결과, Lance는 이미지 및 비디오 생성에서 기존 오픈소스 통합 모델을 상당히 능가하면서 강력한 멀티모달 이해 능력을 유지합니다.

- **GenEVAL**: 3B의 컴팩트한 모델을 유지하면서 통합 모델 중 최고 종합 점수 타이를 달성했습니다.
- **DPG-Bench**: 글로벌, 엔티티, 속성, 관계, 기타 구성 차원에서 복잡한 프롬프트 따르기를 강조하며, Lance는 특히 관계 그라운딩에서 강합니다.
- **GEdit-Bench**: 배경, 색상, 재질, 주제, 스타일, 톤 변경 등 명령 기반 편집을 평가하며, Lance는 통합 모델 중 최고 평균 점수를 기록합니다.
- 학습 예산이 증가함에 따라, Lance는 프롬프트 정렬, 시각적 충실도, 텍스트 렌더링, 시간적 일관성을 점진적으로 향상시킵니다.

---

### 2-6. 한계

출력 품질은 프롬프트, 해상도, 지속시간, 움직임 복잡성 및 편집 시나리오에 따라 달라질 수 있으며, 사후 학습(post-training) 레시피를 개선할 추가적인 기회가 있습니다.

공개된 체크포인트는 768×768 이미지 생성과 480p, 12 FPS 비디오 생성까지만 학습이 진행되었습니다.

통합 모델은 생성을 개선하면 이해가 손상되고, 그 반대도 마찬가지인 튜닝의 어려움이 있습니다.

Lance는 완성된 제품 모델이 아닌 연구 프로젝트입니다.

---

## 3. 모델의 일반화 성능 향상 가능성

통합 모델에서 더 광범위한 작업 커버리지를 갖는 모델일수록 **미학습(unseen) 작업에 대한 창발적 일반화(Emergent Generalization)** 를 보일 가능성이 높다는 관찰이 있습니다.

CT 단계는 추가적인 순수 생성 데이터가 아닌 편집 및 명령 추종 데이터와 같은 멀티태스크 데이터를 주로 도입함에도 불구하고 네이티브 생성 능력을 향상시킵니다. 이러한 결과는 멀티태스크 통합이 편집 및 명령 추종 동작을 강화할 뿐만 아니라 시각 생성에도 **긍정적 전이(Positive Transfer)** 를 가져옴을 시사하며, 통합 멀티모달 모델링 향상에서 멀티태스크 시너지의 역할을 더욱 검증합니다.

비교적 소규모 모델이 이해, 생성, 편집 전반에 걸쳐 강력하고 균형 잡힌 성능을 달성할 수 있음을 보여줍니다.

### 일반화 향상의 구체적 메커니즘

| 메커니즘 | 설명 |
|----------|------|
| **Task-Level Positive Transfer** | 편집 태스크 학습이 순수 생성 능력에도 긍정적 효과 유발 |
| **Shared Context Learning** | 공유 인터리브 시퀀스를 통한 크로스-태스크 문맥 공유 |
| **MaPE** | 이질적 토큰 간 간섭 감소로 크로스태스크 정렬 향상 |
| **Emergent Generalization** | 더 넓은 태스크 커버리지 → 미학습 태스크 일반화 가능성 증가 |

멀티모달 AI는 이해, 추론, 생성이 통합 프레임워크 내에 통합되는 **네이티브 통합 패러다임**으로 점점 더 나아가고 있습니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

Lance의 연구는 Show-o2, InternVL-U와 같이 이해와 생성을 결합하려는 모델들의 흐름을 따릅니다.

| 모델 | 특징 | Lance와의 차이점 |
|------|------|----------------|
| **Chameleon** (2024, Meta) | 토큰 기반 통합 멀티모달 | 이미지-텍스트 도메인 중심, 비디오 미지원 |
| **Janus** (2024) | 이해/생성 경로 분리 구조 | 이미지 중심, 비디오 편집 미지원 |
| **Show-o2** (2025, NUS+ByteDance) | 자기회귀 + Flow Matching 통합 | 3D Causal VAE 기반, 이미지/비디오 지원 |
| **TUNA / Emerging** (2025) | 비디오 도메인 확장 | 제한적 태스크 커버리지 |
| **Lance (2026, ByteDance)** | X2T, X2I, X2V 전체 통합 | 가장 넓은 태스크 커버리지, MaPE+MoE |

Show-o2는 개선된 네이티브 통합 멀티모달 모델로, 자기회귀 모델링과 Flow Matching을 활용합니다. 3D 인과 변분 오토인코더 공간 위에 구축되어, 공간(-시간) 융합의 이중 경로를 통해 통합 시각 표현을 구성하며, 이미지와 비디오 모달리티에 걸쳐 확장성을 가능하게 합니다. 언어 헤드와 플로우 헤드에 자기회귀 모델링과 Flow Matching이 적용됩니다.

**Lance의 차별점:** Lance는 X2T(이해), X2I(이미지 생성/편집), X2V(비디오 생성/편집) 태스크 전반에 걸친 공동 학습을 체계적으로 통합하며, 이러한 태스크 패밀리들을 단일 네이티브 모델에 통합함으로써 크로스태스크 시너지를 더 잘 활용하고 통합 멀티모달 모델링의 잠재력을 더욱 발전시키는 것을 목표로 합니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

**① 경량 통합 모델의 가능성 입증**

3B 활성 파라미터만으로도 이미지 생성, 이미지 편집, 비디오 생성 벤치마크 전반에서 경쟁력 있는 성능을 달성하여, 대규모 스케일링 없이도 통합 모델링이 가능함을 입증합니다. 이는 엣지 디바이스나 제한된 컴퓨팅 환경에서의 통합 멀티모달 AI 연구에 새로운 방향을 제시합니다.

**② 멀티태스크 시너지 연구의 촉진**

멀티태스크 통합이 편집 및 명령 추종 동작을 강화할 뿐만 아니라 시각 생성에도 긍정적 전이를 가져온다는 것이 검증되면서, 향후 **어떤 태스크 조합이 최대 시너지를 유발하는지** 에 대한 연구가 활발해질 것으로 예상됩니다.

**③ 아키텍처 혁신 방향 제시**

Dual-Stream MoE + MaPE의 조합은 단일 모델 내에서 이질적 능력을 분리·통합하는 새로운 아키텍처 설계 원칙을 제시합니다. 이는 향후 더 많은 모달리티(음성, 3D, 촉각 등)를 포함하는 확장 연구로 이어질 수 있습니다.

**④ 창발적 일반화 연구의 확장**

통합 모델이 미학습 태스크에 대한 창발적 일반화를 보이는지의 여부가 중요한 평가 지표로 부상하면서, 이에 대한 체계적 연구와 측정 방법론 개발이 요구됩니다.

---

### 5-2. 앞으로의 연구에서 고려할 점

**① 해상도 및 품질 한계 극복**
현재 체크포인트는 768×768 이미지 생성과 480p, 12 FPS 비디오 생성까지 학습되어 있어, 고해상도(4K)·고프레임레이트 생성을 위한 확장 연구가 필요합니다.

**② Task Conflict 완화 전략 연구**
통합 모델에서 생성을 개선하면 이해가 손상될 수 있다는 문제를 해결하기 위한 **Gradient Conflict Resolution**, **Task-Specific Learning Rate Scheduling**, **Curriculum Learning** 전략에 대한 심화 연구가 필요합니다.

**③ 적응형 데이터 스케줄링의 자동화**
Lance는 단계적 멀티태스크 학습 패러다임과 능력 지향적 목표 및 적응형 데이터 스케줄링을 채택하지만, 이 스케줄링을 수동으로 설계해야 하는 한계가 있습니다. 강화학습 기반의 **자동 데이터 스케줄링** 연구가 유망합니다.

**④ 포스트 트레이닝 레시피 개선**
출력 품질이 프롬프트, 해상도, 지속시간, 움직임 복잡성 및 편집 시나리오에 따라 달라질 수 있어, 사후 학습 레시피를 개선할 기회가 있습니다. RLHF/RLAIF 등 인간 피드백 기반 강화 방법의 적용이 중요한 후속 연구가 될 것입니다.

**⑤ 멀티모달 평가 기준 표준화**
현재 GenEVAL, DPG-Bench, GEdit-Bench, VBench, MVBench 등 다양한 벤치마크가 혼재하여 있어, 이해·생성·편집을 동시에 평가할 수 있는 **통합 벤치마크 프레임워크** 의 개발이 필요합니다.

**⑥ 오디오·3D 등 추가 모달리티 통합**
이미지와 비디오 양쪽에서 이해와 생성을 네이티브하게 결합하는 통합 멀티모달 모델링은 여전히 활발한 연구 최전선에 있으며, 오디오, 3D 공간, 촉각 데이터 등을 포함하는 확장이 다음 단계의 연구 과제입니다.

---

## 📚 참고 자료 (출처 목록)

| 번호 | 제목 / 출처 | URL |
|------|------------|-----|
| 1 | **[arXiv 논문 원문]** Lance: Unified Multimodal Modeling by Multi-Task Synergy (arXiv:2605.18678) | https://arxiv.org/abs/2605.18678 |
| 2 | **[arXiv HTML 전문]** Lance 논문 HTML 버전 | https://arxiv.org/html/2605.18678v1 |
| 3 | **[arXiv PDF]** Lance 논문 PDF | https://arxiv.org/pdf/2605.18678 |
| 4 | **[공식 GitHub]** bytedance/Lance | https://github.com/bytedance/Lance |
| 5 | **[프로젝트 홈페이지]** lance-project.github.io | https://lance-project.github.io |
| 6 | **[HuggingFace Paper Page]** Lance 모델 페이지 | https://huggingface.co/papers/2605.18678 |
| 7 | **[HuggingFace Model]** bytedance-research/Lance | https://huggingface.co/bytedance-research/Lance |
| 8 | **[MarkTechPost 분석]** One Model, Three Modalities: ByteDance Releases Lance | https://www.marktechpost.com/2026/05/21/one-model-three-modalities-bytedance-releases-lance-for-image-and-video-understanding-generation-and-editing/ |
| 9 | **[Let's Data Science]** ByteDance Releases Lance Unified Multimodal Model | https://letsdatascience.com/news/bytedance-releases-lance-unified-multimodal-model-e012e7ec |
| 10 | **[alphaXiv 강의 오디오]** Lance: Unified Multimodal Modeling by Multi-Task Synergy 분석 | https://www.alphaxiv.org/audio/2605.18678 |
| 11 | **[비교 논문]** Show-o2: Improved Native Unified Multimodal Models (arXiv:2506.15564) | https://arxiv.org/pdf/2506.15564 |

> ⚠️ **정확도 주의:** 본 답변은 arXiv 원문(2605.18678), 공식 GitHub, 프로젝트 홈페이지 및 공신력 있는 분석 자료를 기반으로 작성되었습니다. MaPE 및 Flow Matching 수식의 세부 계수 값 등 논문 내부 실험의 구체적 수치(예: 정확한 $\lambda$ 값, 상세 ablation 수치)는 PDF 원문을 직접 확인하시기 바랍니다.
