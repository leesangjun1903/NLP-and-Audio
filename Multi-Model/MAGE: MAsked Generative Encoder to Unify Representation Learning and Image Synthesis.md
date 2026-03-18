# MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis

---

## 1. 핵심 주장 및 주요 기여 (요약)

MAGE는 **SOTA 이미지 생성**과 **자기지도 표현 학습(self-supervised representation learning)**을 하나의 프레임워크로 통합하는 최초의 방법을 제안합니다. 핵심 통찰은 마스크드 이미지 모델링(MIM) 사전훈련 시 **가변 마스킹 비율(variable masking ratio)**을 사용함으로써, 매우 높은 마스킹 비율은 생성 훈련을, 상대적으로 낮은 마스킹 비율은 표현 학습을 동일 프레임워크 내에서 가능하게 한다는 것입니다.

MAGE는 VQGAN이 학습한 시맨틱 토큰(semantic tokens)을 입력과 출력으로 사용하며, 이를 마스킹과 결합합니다.

ImageNet-1K에서 단일 MAGE ViT-L 모델이 클래스 비조건부(class-unconditional) 이미지 생성에서 **FID 9.10**, 선형 프로빙(linear probing)에서 **78.9% top-1 정확도**를 달성하여, 두 과제 모두에서 SOTA 성능을 기록했습니다.

**주요 기여:**
1. 이미지 생성과 표현 학습의 최초 통합 프레임워크
2. 가변 마스킹 비율 + 양자화 토큰 조합의 효과 입증
3. 선택적 대조 손실(contrastive loss) 추가로 표현 품질 향상
4. 인페인팅, 아웃페인팅 등 다양한 이미지 합성 과제에 자연스러운 확장

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

생성 모델링(generative modeling)과 표현 학습(representation learning)은 컴퓨터 비전의 두 핵심 과제입니다. 그러나 이 모델들은 보통 독립적으로 훈련되어, 각 과제가 서로를 돕는 잠재력이 무시되고, 훈련 및 모델 유지 비용이 발생합니다.

기존 MIM 기반 자기지도 학습 방법들은 다운스트림 과제에서의 표현 성능을 우선시하여, 재구성 이미지의 품질이 낮았습니다.

핵심 관찰: 생성은 "100% 마스킹된 이미지를 복원"하는 것이고, 표현 학습은 "0% 마스킹된 이미지를 인코딩"하는 것으로 해석할 수 있습니다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) 토큰화 (Tokenization)

통합을 가능하게 하기 위해, 먼저 사전훈련된 VQGAN 모델을 사용하여 입력 이미지를 시맨틱 토큰으로 양자화합니다. 그런 다음 0.5에서 1 사이의 가변 마스킹 비율로 일부 입력 토큰을 무작위로 마스킹하고, 인코더-디코더 트랜스포머 아키텍처를 마스킹되지 않은 토큰에 적용하여 마스킹된 토큰을 예측합니다.

입력 이미지 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$가 VQGAN 인코더 $E$와 양자화기(quantizer) $Q$를 통해 시맨틱 토큰 시퀀스로 변환됩니다:

$$\mathbf{z} = Q(E(\mathbf{x})) = [z_1, z_2, \ldots, z_N], \quad z_i \in \{1, 2, \ldots, K\}$$

여기서 $N$은 토큰 수(예: $16 \times 16 = 256$), $K$는 코드북(codebook) 크기입니다.

#### (B) 가변 마스킹 비율 (Variable Masking Ratio)

마스킹 비율 $\gamma$는 절단된 가우시안 분포(truncated Gaussian distribution)에서 샘플링됩니다: $\mu = 0.55$이고, $[0.5, 1.0]$ 구간으로 클리핑됩니다.

$$\gamma \sim \mathcal{N}_{\text{truncated}}(\mu, \sigma^2), \quad \gamma \in [0.5, 1.0]$$

이 설계의 핵심은:
- **높은 마스킹 비율** ($\gamma \to 1.0$): 생성 능력 학습 (거의 모든 토큰 예측)
- **낮은 마스킹 비율** ($\gamma \to 0.5$): 표현 학습 능력 강화 (충분한 컨텍스트 유지)

#### (C) 재구성 손실 (Reconstruction Loss)

재구성적 크로스엔트로피 손실(reconstructive cross-entropy loss)이 모델이 마스킹된 토큰을 복원하도록 유도합니다.

마스킹된 위치 집합 $\mathcal{M}$에 대해:

$$\mathcal{L}\_{\text{recon}} = -\sum_{i \in \mathcal{M}} \log p_\theta(z_i \mid \mathbf{z}_{\setminus \mathcal{M}})$$

여기서 $\mathbf{z}\_{\setminus \mathcal{M}}$은 마스킹되지 않은 토큰 집합, $p_\theta$는 모델이 예측하는 토큰의 확률 분포입니다.

#### (D) 대조 손실 (Contrastive Loss, 선택적)

인코더 출력에 SimCLR 유사 대조 손실을 추가하여, 학습된 표현의 선형 분리 가능성(linear separability)을 더욱 향상시킬 수 있습니다 (MAGE-C).

$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_j) / \tau)}{\sum_{k=1}^{2B} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_k) / \tau)}$$

여기서 $(\mathbf{h}_i, \mathbf{h}_j)$는 동일 이미지의 서로 다른 증강(augmented) 뷰의 인코더 출력, $\tau$는 온도 파라미터, $B$는 배치 크기입니다.

**총 손실 함수:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda \cdot \mathcal{L}_{\text{contrast}}$$

### 2.3 모델 구조

MAGE 프레임워크: 먼저 VQGAN 토크나이저를 사용하여 입력 이미지를 시맨틱 토큰 시퀀스로 토큰화합니다. 그 후 마스킹 비율을 샘플링하고, 해당 비율에 따라 토큰을 무작위로 마스킹합니다. ViT 인코더-디코더 구조가 마스킹되지 않은 토큰을 처리합니다.

```
┌─────────────────────────────────────────────────────┐
│                    MAGE 아키텍처                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  입력 이미지 ──→ [VQGAN Tokenizer] ──→ 시맨틱 토큰    │
│                        │                            │
│              [Variable Masking: γ ~ N(0.55, σ²)]    │
│                        │                            │
│                        ▼                            │
│           ┌──────────────────────┐                  │
│           │   ViT Encoder        │──→ 표현(h)        │
│           │  (unmasked tokens)   │   ├→ Linear Probe │
│           └──────────┬───────────┘   └→ Contrastive  │
│                      │                  Loss (선택적) │
│                      ▼                              │
│           ┌──────────────────────┐                  │
│           │   ViT Decoder        │                  │
│           │ (predict masked tok) │                  │
│           └──────────┬───────────┘                  │
│                      │                              │
│              Cross-Entropy Loss                     │
│           (masked token 복원)                        │
└─────────────────────────────────────────────────────┘
```

**추론 시 이미지 생성 (Iterative Decoding):**

각 반복(iteration)에서 모델은 나머지 마스킹된 토큰에 대한 예측을 수행합니다. 그 후 예측 확률이 높은 일부 토큰을 샘플링하여 마스킹된 토큰을 대체합니다. 각 반복에서 대체되는 마스킹 토큰 수는 코사인 함수를 따르며, 총 20 스텝으로 이미지를 생성합니다.

### 2.4 성능 비교

| 모델 | 과제 | FID (↓) | Linear Probe Top-1 (↑) |
|------|------|---------|------------------------|
| **MAGE ViT-L** | 비조건부 생성 + 표현 | **9.10** | **78.9%** |
| MAE ViT-L | 표현 학습 | – | 75.8% |
| MaskGIT | 조건부 생성 | 6.18 | – |
| BigGAN | 조건부 생성 | 7.53 | – |

MAGE는 선형 프로빙에서 SOTA 성능을 달성하고, 클래스 비조건부 생성에서 새로운 SOTA를 수립합니다.

MAGE는 양자화된 토큰의 시맨틱 특성 덕분에 모든 트랜스포머 블록에서 MAE보다 일관되게 높은 선형 프로빙 정확도를 보입니다.

### 2.5 한계

논문 및 후속 연구에서 확인되는 주요 한계점:

1. **VQGAN 토크나이저 의존성**: 모델 성능의 상한이 VQGAN의 재구성 품질에 의해 제한됩니다.
2. **ImageNet 중심 평가**: 주요 실험이 ImageNet-1K에 집중되어 있어, 다양한 도메인에 대한 일반화 성능 검증이 부족합니다.
3. **클래스 조건부(class-conditional) 생성 미흡**: 비조건부 생성에서는 SOTA이나, 조건부 생성에서의 Diffusion 모델 대비 경쟁력은 제한적입니다.
4. **해상도 제한**: 256×256 해상도에서만 주요 실험을 수행합니다.
5. 낮은 마스킹 비율에서 양자화 없이 원시 픽셀을 사용하면, 사전훈련 과제가 너무 쉬워져 단축 해법(shortcut solution)으로 이어지고 표현 품질이 저하됩니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 가변 마스킹 비율의 일반화 효과

모델은 생성 능력을 가능하게 하는 높은 마스킹 비율과 표현 학습을 가능하게 하는 낮은 마스킹 비율을 포함하는 광범위한 마스킹 비율에 걸쳐 재구성하도록 훈련됩니다. 이 단순하면서도 매우 효과적인 접근법은 동일한 아키텍처, 훈련 방식, 손실 함수 내에서 생성 훈련과 표현 학습의 부드러운 결합을 가능하게 합니다.

가변 마스킹 비율은 일종의 **정규화(regularization)** 효과를 제공합니다:

$$\mathcal{L}_{\text{total}} = \mathbb{E}_{\gamma \sim p(\gamma)} \left[ -\sum_{i \in \mathcal{M}_\gamma} \log p_\theta(z_i \mid \mathbf{z}_{\setminus \mathcal{M}_\gamma}) \right]$$

이는 모델이 다양한 난이도의 과제에 노출되게 하여, 특정 마스킹 패턴에 과적합되는 것을 방지합니다.

### 3.2 시맨틱 토큰의 역할

토큰화 단계에서 원시 픽셀 대신 시맨틱 토큰을 사용하는 것은 생성과 표현 학습 모두에 유리합니다.

양자화 단계는 좋은 표현을 학습하는 데 필수적입니다. 이는 다음과 같은 이유 때문입니다:

- 시맨틱 토큰은 저수준 픽셀 정보 대신 **고수준 의미 정보**를 인코딩
- 이산적 토큰 예측은 연속적 픽셀 복원보다 **더 풍부한 의미적 이해**를 요구
- 결과적으로 **다운스트림 과제 전이(transfer) 성능** 향상

### 3.3 대조 손실을 통한 추가 일반화

인코더 출력에 대조 손실을 추가하여 표현을 더욱 개선할 수 있습니다. 이는 학습된 표현의 클러스터링 구조를 강화하여, few-shot 전이 학습과 같은 시나리오에서 일반화 성능을 높입니다.

### 3.4 다중 과제 학습의 시너지

MAGE의 핵심 설계 철학은 **생성과 인식(recognition)이 상호 보완적**이라는 것입니다:
- 생성 학습은 데이터 분포의 전체 구조에 대한 이해를 요구 → 표현 품질 향상
- 좋은 표현은 더 정확한 컨텍스트 이해를 가능하게 → 생성 품질 향상

뛰어난 클래스 비조건부 재구성 및 생성 능력을 바탕으로, MAGE는 인페인팅, 아웃페인팅, 언크롭핑과 같은 다양한 이미지 합성 응용에 자연스럽게 활용됩니다.

---

## 4. 향후 연구 영향 및 고려사항

### 4.1 연구 영향

MAGE는 자기지도 표현 학습과 생성 모델링 간의 시너지를 탐구하는 연구 흐름을 촉발했습니다. 이후 M3AE는 마스크드 오토인코딩과 대조 목표를 통합 비전-언어 프레임워크에서 결합했습니다.

더 최근의 멀티모달 모델인 Janus(Wu et al., 2025a), Show-o(Xie et al., 2024), Harmon(Wu et al., 2025b)은 판별적 이해(discriminative understanding)와 텍스트-이미지 생성을 통합하는 것을 목표로 합니다.

### 4.2 향후 연구 시 고려할 점

1. **더 강력한 토크나이저 탐색**: MAGVIT-v2, LlamaGen-VQ 등 최신 토크나이저로의 교체를 통한 성능 상한 개선
2. **고해상도 확장**: 512×512, 1024×1024 해상도로의 확장 및 효율적 추론 전략 연구
3. **멀티모달 통합**: 텍스트-이미지 생성과 이해를 하나의 모델에서 수행하는 방향
4. **Diffusion 모델과의 결합**: MAR(Masked Autoregressive)과 연속 확산 모델의 장점을 결합
5. **도메인 일반화**: 의료, 위성, 비디오 등 다양한 도메인으로의 전이 가능성 검증

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 아이디어 | 생성 FID (ImageNet) | 표현 학습 | 비고 |
|------|------|-------------|--------------------|-----------|----|
| **MAE** (He et al.) | 2022 | 픽셀 재구성 기반 MIM | 생성 불가 | 75.8% LP | 표현 학습 전용 |
| **MaskGIT** (Chang et al.) | 2022 | 양방향 트랜스포머 병렬 디코딩 | 6.18 (조건부) | – | 생성 전용 |
| **MAGE** (Li et al.) | 2023 | 가변 마스킹 + VQ 토큰 통합 | **9.10** (비조건부) | **78.9%** LP | **생성+표현 통합** |
| **DiGIT** (NeurIPS 2024) | 2024 | 판별적 토크나이저 + AR | 4.59 (조건부) | 표현 학습 겸용 | 대규모 모델 스케일링 |
| **eMIGM** (Bai et al.) | 2025 | 통합 마스크 이미지 생성 프레임워크 | 1.57 (조건부) | – | MAR 기반 최적화 |
| **Janus / Show-o / Harmon** | 2024–25 | 멀티모달 이해+생성 통합 | – | – | VLM 기반 확장 |

MaskGIT, MAGE 등의 MAR 생성 모델은 전통적 AR 모델보다 적은 추론 스텝으로 고품질 이미지 생성을 목표로 개발되어 왔습니다.

그러나 훈련과 예측의 내재적 어려움으로 인해, 기존 MAR 모델의 생성 능력은 AR 모델에 비해 여전히 제한적입니다.

최근 eMIGM은 REPA와 같은 최첨단 확산 모델 수준의 FID(1.57 vs. 1.42)를 달성하면서도, 자기지도 특징을 요구하지 않습니다.

### 핵심 트렌드 요약

MAGE 이후의 연구 동향은 크게 세 가지로 정리됩니다:

1. **토크나이저 고도화**: VQGAN → MAGVIT-v2 → LlamaGen-VQ로 이어지는 코드북 품질 향상
2. **생성-이해 통합의 멀티모달 확장**: 이미지뿐 아니라 텍스트, 비디오를 포함하는 범용 모델
3. **샘플링 전략 최적화**: 마스크 스케줄, CFG(Classifier-Free Guidance), DPM-Solver 등을 통한 추론 효율 및 품질 개선

---

## 참고자료

1. **Li, T., Chang, H., Mishra, S., Zhang, H., Katabi, D., & Krishnan, D.** (2023). "MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis." *CVPR 2023*, pp. 2142–2152. [arXiv:2211.09117](https://arxiv.org/abs/2211.09117)
2. **MAGE 공식 GitHub 저장소**: [https://github.com/LTH14/mage](https://github.com/LTH14/mage)
3. **CVPR 2023 Open Access**: [CVPR 2023 Paper PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_MAGE_MAsked_Generative_Encoder_To_Unify_Representation_Learning_and_Image_CVPR_2023_paper.pdf)
4. **Google Research Publication Page**: [research.google/pubs/mage](https://research.google/pubs/mage-masked-generative-encoder-to-unify-representation-learning-and-image-synthesis/)
5. **Chang, H., Zhang, H., Jiang, L., Liu, C., & Freeman, B.** (2022). "MaskGIT: Masked Generative Image Transformer." *CVPR 2022*. [arXiv:2202.04200](https://arxiv.org/abs/2202.04200)
6. **Bai et al.** (2025). "Effective and Efficient Masked Image Generation Models." [arXiv:2503.07197](https://arxiv.org/abs/2503.07197)
7. **"Resurrect Mask AutoRegressive Modeling for Efficient and Scalable Image Generation"** (2025). [arXiv:2507.13032](https://arxiv.org/abs/2507.13032)
8. **Masked Image Modeling: A Survey.** *International Journal of Computer Vision*, Springer, 2025. [SpringerLink](https://link.springer.com/article/10.1007/s11263-025-02524-1)
9. **arXiv HTML 버전 (MAGE 상세 분석)**: [arxiv.org/html/2211.09117](https://arxiv.org/html/2211.09117)
