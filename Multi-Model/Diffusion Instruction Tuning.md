
# Diffusion Instruction Tuning (Lavender)

> **논문 정보**
> - **제목**: Diffusion Instruction Tuning
> - **저자**: Chen Jin, Ryutaro Tanno, Amrutha Saseendran, Tom Diethe, Philip Teare (AstraZeneca)
> - **arXiv**: [2502.06814](https://arxiv.org/abs/2502.06814) (v1: 2025.02.04, v2: 2025.05.25)
> - **발표**: ICML 2025 (포스터 채택)
> - **공식 구현**: [GitHub - AstraZeneca/vlm](https://github.com/AstraZeneca/vlm)

---

## 1. 핵심 주장 및 주요 기여 요약

Lavender(Language-and-Vision fine-tuning with Diffusion Aligner)는 VLM 트랜스포머 어텐션 레이어를 Stable Diffusion의 어텐션과 직접 정렬하는 **최초의 프레임워크**로, SFT 과정에서 확산 기반 어텐션 분포를 VLM에 전이하여 핵심적인 시각-텍스트 상호작용을 강화합니다.

### 🔑 핵심 주장 3가지

| 주장 | 내용 |
|------|------|
| **이질적 모델 간 지식 전이** | 이미지 생성 모델(Stable Diffusion)의 어텐션 지식을 이미지-텍스트 이해 모델(VLM)에 전이 가능 |
| **어텐션 품질 격차** | 픽셀 수준에서 이미지를 재구성하는 Stable Diffusion과 같은 DM은, 텍스트 토큰 생성만을 위해 최적화된 VLM보다 더 정밀한 텍스트-비전 어텐션 맵을 학습한 것으로 보인다. |
| **정렬이 일반화를 유도** | VLM 트랜스포머 내 텍스트-비전 어텐션을 Stable Diffusion의 어텐션과 정렬함으로써, 모델의 시각적 이해를 풍부하게 하고 분포 내/외 태스크 전반에서 성능을 크게 향상시킨다. |

### 🏆 주요 기여

ICML 2025 포스터 채택. Lavender는 단순한 SFT 기법으로, VLM의 토큰별 어텐션을 Stable Diffusion의 강력한 T2I 어텐션과 정렬하여 image-to-text 생성을 향상시키며, 상당한 성능 향상, 향상된 OOD 견고성, 최소한의 컴퓨팅으로 데이터 효율성을 제공한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

기존 방법들은 LLM 핵심 내부의 트랜스포머 수준 어텐션 정렬의 중요성을 간과하는 경향이 있었으며, 이는 텍스트 기반 모델을 시각 도메인으로 확장하는 데 핵심적인 요소다. 정밀한 시각-텍스트 정렬은 고급 멀티모달 추론에 필수적이다. VLM과 Diffusion 모델(DM) 모두 텍스트와 이미지를 처리하지만, 생성 목적에서 차이를 보인다.

구체적으로:
- **VLM의 시각 이해 부족**: VLM에서 시각적 이해가 언어 능력에 뒤처진다는 최근 연구들에서 영감을 받아, Lavender는 단순하면서도 효과적인 MSE 손실 정렬로 이 격차를 해결한다.
- **어텐션 엔트로피 격차**: DM 어텐션 분포가 VLM의 이상적인 사후 어텐션에 더 가깝다는 핵심 가설 검증에서, DM 어텐션은 VLM 어텐션보다 더 낮은 엔트로피를 가진다는 것이 확인되었다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 전체 학습 목적 함수

Lavender의 전체 손실 함수는 **표준 자기회귀 SFT 손실**에 **어텐션 정렬 정규화 항**을 추가한 형태입니다:

$$\mathcal{L}_{\text{Lavender}} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{align}}$$

여기서:
- $\mathcal{L}_{\text{SFT}}$: 기존 지도 미세조정(cross-entropy) 손실
- $\lambda$: 정렬 항의 가중치 하이퍼파라미터
- $\mathcal{L}_{\text{align}}$: 어텐션 정렬 손실 (MSE 기반)

#### 어텐션 정렬 손실

Lavender는 Stable Diffusion 모델의 텍스트-비전 어텐션 맵 $\text{Attention}\_{SDM}$을 타겟 VLM의 어텐션 $\text{Attention}\_{VLM}$에 대한 가이딩 목적으로 사용한다. 어텐션 정렬 모듈은 3-Layer ConvNet을 활용하여 $\text{Attention}\_{VLM}$을 변환하여 MSE 손실을 통해 $\text{Attention}_{SDM}$과 일치시키며, 이는 지도 미세조정 중 정규화 항으로 작용한다.

수식으로 표현하면:

$$\mathcal{L}_{\text{align}} = \text{MSE}\left(\phi\left(\text{Attention}_{VLM}\right),\ \text{Attention}_{SDM}\right)$$

$$= \frac{1}{N} \sum_{i=1}^{N} \left\| \phi\left(A^{(i)}_{VLM}\right) - A^{(i)}_{SDM} \right\|^2_F$$

여기서:
- $\phi(\cdot)$: 3-Layer ConvNet (Aligner Network) — VLM 어텐션을 SDM 어텐션 공간으로 변환
- $A^{(i)}_{VLM} \in \mathbb{R}^{H \times W}$: VLM의 $i$-번째 토큰에 대한 크로스-어텐션 맵
- $A^{(i)}_{SDM} \in \mathbb{R}^{H \times W}$: Stable Diffusion의 $i$-번째 토큰에 대한 크로스-어텐션 맵 (사전 추출, 오프라인)
- $\|\cdot\|^2_F$: Frobenius 노름의 제곱

#### 오프라인 사전 추출 (Zero Additional Training Cost)

데이터 전처리 단계에서 Lavender는 Stable Diffusion의 시각 능력을 활용하여 기존 데이터로부터 토큰별 어텐션을 추출하고, 미세조정 이전에 정렬을 위한 골든 스탠다드를 설정한다.

$$\text{Attention}_{SDM}^{(i)} = \text{SDM CrossAttn}(\text{image}, \text{token}_i) \quad \text{(offline, no gradient)}$$

#### LoRA 결합 적용

Aligner Network(몇 개의 경량 합성곱 레이어)를 제안하여 원시 VLM 어텐션을 Stable Diffusion 어텐션과 직접 매칭될 수 있는 분포로 변환하며, 파라미터 효율적 미세조정(LoRA)과 함께 사용 시 원래 VLM의 능력을 불안정화하지 않으면서 강력한 결과를 보인다.

---

### 2.3 모델 구조

```
┌──────────────────────────────────────────────────────────┐
│               오프라인 전처리 단계                          │
│  [이미지-텍스트 쌍 130k] ──→ Stable Diffusion           │
│                              ↓                           │
│              AttentionSDM 추출 (토큰별, 저장)             │
└──────────────────────────────────────────────────────────┘
                        ↓ (저장된 어텐션 맵)
┌──────────────────────────────────────────────────────────┐
│               SFT 학습 단계 (Lavender)                   │
│                                                          │
│  [입력 이미지+텍스트]                                     │
│       ↓                                                  │
│  [VLM (Llama-3.2-11B / MiniCPM-v2.5 등)]               │
│       ↓                                                  │
│  AttentionVLM (크로스-어텐션)                             │
│       ↓                                                  │
│  [Aligner Network: 3-Layer ConvNet φ(·)]                │
│       ↓                           ↓                     │
│  φ(AttentionVLM) ←MSE→ AttentionSDM (고정, 오프라인)   │
│                                                          │
│  손실: L_Lavender = L_SFT + λ·L_align                  │
│  파라미터 업데이트: VLM(LoRA 또는 Full) + Aligner         │
└──────────────────────────────────────────────────────────┘
```

파국적 망각(catastrophic forgetting)을 완화하기 위해, 기존 VLM의 역량을 보존하는 여러 어텐션 집계 방법 및 학습 전략도 추가로 제안한다.

---

### 2.4 성능 향상

Stable Diffusion을 활용하여 13만 개의 레이블-이미지 쌍에서 단어별 어텐션을 오프라인으로 추출하는 방식으로, 추가 학습 비용 없이 Lavender는 20개의 다양한 벤치마크에서 자기회귀 미세조정 대비 최대 70%의 성능 향상을 달성했다. Llama 3.2-11B의 경우, in/out-of-distribution 데이터 모두에서 19개 벤치마크에서 최대 30% 향상되었으며, 유사 소규모 오픈소스 모델 대비 50% 이상 성능을 뛰어넘었다.

**태스크별 성능 분포**:
Lavender는 OCR 태스크에서 정밀한 텍스트-시각 정렬에 의존하는 차트, 다이어그램, 문서 이해에서 가장 큰 향상을 보인다. 인식 및 다학제 추론, 환각 분야에서는 적절한 수준의 향상이 관찰되며, 더 넓은 지식이 필요한 실세계 시각 이해에서는 가장 낮은 향상을 보인다.

**데이터 및 컴퓨팅 효율**:
Lavender는 단 13만 개의 학습 예시(일반적인 대규모 SFT 데이터셋의 2.5%)만 필요하며, 표준 하드웨어(GPU 8개)에서 하루 만에 미세조정이 가능하다.

**스케일링 특성**:
Lavender는 자기회귀 미세조정에 비해 더 잘 스케일링되며, 더 큰 데이터셋에서 변동성을 줄여 과적합을 완화한다.

---

### 2.5 한계

논문 및 GitHub 공식 페이지를 통해 확인된 한계점은 다음과 같습니다:

1. **Self-Attention 전용 모델의 제한**: Self-Attention만 사용하는 MiniCPMv2.5에서는 최대 4%의 향상에 그친다. 이는 Lavender가 **크로스-어텐션 구조를 가진 VLM에 더 효과적**임을 시사합니다.

2. **실세계 시각 이해의 한계**: 실세계 시각 이해 분야에서는 더 넓은 지식이 필요하기 때문에 가장 낮은 향상이 관찰된다.

3. **데이터 오버랩의 불완전한 통제**: 이 접근법은 사전 학습 단계에서의 데이터 오버랩 가능성을 배제하지 않으며, 주요 초점은 미세조정 단계에 있다.

4. **라이선스 종속성**: 데이터와 코드는 연구용으로만 사용 가능하며, Llama, Stable Diffusion 등 업스트림 모델의 라이선스 제약을 받는다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화 성능은 핵심 연구 주제 중 하나로, 특히 **OOD(Out-of-Distribution) 일반화**에서 두드러진 결과를 보입니다.

### 3.1 OOD 일반화의 실험적 증거

의료 데이터셋 튜닝 없이도 Lavender는 OOD 벤치마크인 WorldMedQA에서 Llama-3.2-11B의 성능을 68% 향상시켰다.

이는 단순히 학습 데이터의 다양성이 아니라, **어텐션의 질적 향상** 자체가 도메인 불변적 일반화를 이끌어냄을 보여줍니다.

### 3.2 일반화 메커니즘 분석

픽셀 수준에서 이미지를 재구성하는 Stable Diffusion 같은 DM은 VLM보다 더 정밀한 텍스트-비전 어텐션 맵을 학습하는 것으로 보이며, 이러한 DM의 고품질 크로스-어텐션 맵이 SFT 중 VLM의 텍스트-비전 어텐션을 가이드하는 유용한 타겟을 제공하여 단어-영역 정렬과 전체 성능을 향상시킨다.

정렬된 VLM 어텐션 맵은 일반적으로 대응하는 단어의 의미적 영역과 Diffusion 모델과 유사한 방식으로 상관관계를 맺으며, 정렬 후 VLM 어텐션 맵은 Diffusion 모델보다 더 집중된 형태를 보인다.

### 3.3 데이터 오버랩과 일반화의 관계

최신 모델들의 미세조정 데이터셋은 벤치마크 데이터셋과 더 강한 오버랩을 보이는 반면, Lavender의 미세조정 데이터셋은 LLaVA-1.5와 유사하게 낮은 수준의 오버랩 점수를 보이며, 이는 강력한 일반화 능력을 보여준다.

### 3.4 스케일링 및 과적합 저항성

더 큰 미세조정 세트는 Lavender가 자기회귀 기준선보다 더 효과적으로 과적합에 저항하도록 도우며, 정렬된 어텐션 맵은 더 세밀한 시각적 이해를 제공한다.

---

## 4. 향후 연구에 미치는 영향 및 연구 시 고려할 점

### 4.1 향후 연구에 미치는 영향

#### 🔬 새로운 패러다임 제시
Lavender는 두 전문 AI 패러다임을 연결하여 더 견고하고 유능한 비전-언어 시스템을 구축하는 확장 가능한 방법을 제공한다. 이는 **생성 모델의 내부 표현을 이해 모델에 전이**하는 새로운 연구 방향을 열어줍니다.

#### 📈 데이터-효율적 멀티모달 학습의 방향
이미지 생성기의 시각 전문성을 최소한의 감독으로 효율적으로 전이함으로써, Lavender는 더 정확한 비전-언어 시스템을 위한 확장 가능한 솔루션을 제공한다.

#### 🏥 도메인 특화 응용 가능성
WorldMedQA에서의 68% OOD 향상은 **의료 AI, 과학 문서 이해, 전문 분야 VQA** 등 데이터가 부족한 도메인에서 Lavender 방법론의 활용 가능성을 직접적으로 시사합니다.

#### 🔗 관련 최신 연구와의 비교

| 연구 | 방법 | 특징 | Lavender와의 차이 |
|------|------|------|-----------------|
| **InstructCV** (arXiv 2310.00390) | InstructPix2Pix 아키텍처를 따라 구성된 데이터셋으로 T2I 확산 모델에 instruction-tuning을 적용하여, 생성 모델을 instruction-guided 멀티태스크 비전 학습기로 전환 | DM → Vision Generalist | VLM 어텐션을 직접 정렬하지 않음 |
| **LLaDA-V** (arXiv 2505.16933) | 순수 확산 기반 MLLM으로, 마스크 확산 모델과 시각 instruction tuning을 통합 | 완전 확산 기반 MLLM | 기존 VLM을 수정하지 않고 완전히 새로운 아키텍처 |
| **Lavender (본 논문)** | DM 어텐션을 VLM SFT 정규화에 활용 | 기존 VLM 위에 경량 정렬 | 추가 학습 비용 없이 기존 VLM 개선 |

---

### 4.2 후속 연구 시 고려할 점

#### ⚠️ 기술적 고려사항

1. **크로스-어텐션 의존성**: Lavender는 VLM 내 크로스-어텐션 구조를 전제로 하므로, Self-Attention 전용 아키텍처(MiniCPMv2.5 등)에서는 효과가 제한적입니다. 향후 연구에서는 **Self-Attention 구조에도 적용 가능한 정렬 방법**을 개발해야 합니다.

2. **Aligner Network 설계**: 현재 3-Layer ConvNet 구조가 최적인지에 대한 체계적 탐색이 부족합니다. Transformer 기반 Aligner나 더 정교한 구조가 성능을 더 높일 수 있습니다.

3. **어텐션 집계 전략**: 파국적 망각을 완화하기 위해 여러 어텐션 집계 방법과 학습 전략이 제안되었으나, 이들의 최적 조합에 대한 후속 연구가 필요합니다.

4. **확산 모델 버전 의존성**: Stable Diffusion의 특정 버전(SD 1.x / 2.x / SDXL / SD3 등)에 따라 추출되는 어텐션 맵의 품질이 달라질 수 있으며, 더 최신 모델(SDXL, FLUX 등)의 어텐션을 활용하면 추가 향상 가능성이 있습니다.

5. **멀티모달 확장**: 현재는 이미지-텍스트에 집중되어 있으나, **비디오-텍스트, 3D-텍스트** 등 다른 모달리티로의 확장 가능성을 탐색해야 합니다.

#### 📌 실용적 고려사항

- **라이선스 제약**: 데이터와 코드는 연구용으로만 사용 가능하며, Llama, Stable Diffusion 등 업스트림 모델의 라이선스 제약을 받는다. 상용화를 목표로 하는 연구에서는 이 점을 반드시 고려해야 합니다.

- **실세계 지식 통합**: 실세계 시각 이해에서는 더 넓은 지식이 필요하기 때문에 가장 낮은 향상이 관찰되므로, 이 영역을 강화하기 위한 데이터 증강 전략이나 Knowledge Graph와의 결합 연구가 필요합니다.

- **다양한 Diffusion 모델 아키텍처 탐색**: 현재는 Stable Diffusion 기반이지만, DiT(Diffusion Transformer) 계열 모델(PixArt-α, FLUX 등)의 어텐션을 활용하면 더 정밀한 정렬이 가능할 수 있습니다.

---

## 📚 참고 자료 (출처)

| 번호 | 제목 | 출처 |
|------|------|------|
| 1 | **Diffusion Instruction Tuning** (Lavender) | arXiv: https://arxiv.org/abs/2502.06814 |
| 2 | **Diffusion Instruction Tuning** (공식 프로젝트 페이지) | https://astrazeneca.github.io/vlm/ |
| 3 | **Diffusion Instruction Tuning** (공식 구현 코드) | GitHub: https://github.com/AstraZeneca/vlm |
| 4 | **Diffusion Instruction Tuning** (ICML 2025 포스터) | OpenReview: https://openreview.net/forum?id=DVW16DW1Cn |
| 5 | **Diffusion Instruction Tuning** (ICML 2025 발표 페이지) | ICML: https://icml.cc/virtual/2025/poster/46009 |
| 6 | **InstructCV: Instruction-Tuned Text-to-Image Diffusion Models as Vision Generalists** | arXiv: https://arxiv.org/abs/2310.00390 |
| 7 | **LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning** | arXiv: https://arxiv.org/abs/2505.16933 |
| 8 | **Instruction-tuning Stable Diffusion with InstructPix2Pix** (HuggingFace 블로그) | https://huggingface.co/blog/instruction-tuning-sd |
| 9 | **A Diffusion-Based Approach to Diverse Instruction-Tuning** (NeurIPS 2023 Workshop) | https://neurips2023-enlsp.github.io/papers/paper_82.pdf |
