
# SparseDiT: Token Sparsification for Efficient Diffusion Transformer

> **논문 정보**
> - **제목**: SparseDiT: Token Sparsification for Efficient Diffusion Transformer
> - **arXiv ID**: 2412.06028 (2024년 12월)
> - **저자**: Shuning Chang 외 4인 (Zhejiang University / Alibaba Group)
> - **출처**: https://arxiv.org/abs/2412.06028

> ⚠️ **정확도 안내**: 아래 내용은 공개된 arXiv 논문 및 OpenReview 자료를 기반으로 작성되었습니다. 논문의 전체 수식 세부 사항 일부는 공개된 HTML/PDF 버전을 기반으로 하며, 확인되지 않은 부분은 명시적으로 표기하겠습니다.

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

Diffusion Transformer(DiT)는 인상적인 생성 성능으로 유명하지만, Self-Attention의 이차 복잡도(quadratic complexity)와 방대한 샘플링 스텝으로 인한 막대한 계산 비용에 크게 제약을 받는다. 샘플링 프로세스를 가속하는 방향으로는 연구가 진행되었지만, DiT 내부의 구조적 비효율성은 여전히 충분히 탐구되지 않았다.

이에 SparseDiT는 다음을 주장합니다:

**공간적(spatial) 및 시간적(temporal) 차원 전반에 걸쳐 토큰 희소화(token sparsification)를 구현하는 새로운 프레임워크**를 통해 생성 품질을 유지하면서 계산 효율성을 향상시킨다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| ① 공간적 Tri-Segment 아키텍처 | 레이어별 토큰 밀도 차등 할당 |
| ② SDTM (Sparse-Dense Token Module) | 전역 문맥과 지역 디테일의 균형 |
| ③ 시간적(Timestep-wise) 토큰 밀도 조절 | 디노이징 단계에 따라 동적으로 토큰 수 증가 |

실험 결과, **DiT-XL에서 FLOPs 55% 감소 및 추론 속도 175% 향상**, 유사한 FID 점수를 유지하면서 $512 \times 512$ ImageNet 기준으로 달성했으며, 비디오 생성에서 FLOPs 56% 감소, PixArt- $\alpha$ 텍스트-이미지 생성에서 추론 속도 69% 향상을 달성했다.

---

## 2. 해결하고자 하는 문제, 제안하는 방법, 모델 구조, 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

DiT는 Self-Attention의 **이차 복잡도(quadratic complexity)**와 방대한 샘플링 스텝으로 인해 막대한 계산 비용이 발생한다. 구체적으로 두 가지 문제가 존재합니다.

1. **공간적 비효율성**: 모든 레이어에서 동일한 수의 토큰을 처리 → 하위 레이어에서도 고밀도 어텐션 수행
2. **시간적 비효율성**: 디노이징의 초기 단계(노이즈가 많을 때)에서도 후기 단계와 동일한 계산량 소비

기존 프루닝 방법들은 일반적으로 확산 과정 전반에 걸쳐 timestep과 공간 차원 모두에서 정적(static) 아키텍처를 유지한다. 원래의 DiT와 프루닝된 DiT 모두 모든 확산 타임스텝에서 고정된 모델 너비를 사용하며 모든 이미지 패치에 동일한 계산 비용을 할당한다. 이러한 정적 추론 패러다임은 서로 다른 타임스텝과 공간 영역에 관련된 다양한 복잡성을 무시하여 상당한 계산 비효율을 초래한다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) 공간적 Tri-Segment 아키텍처

SparseDiT는 각 레이어에서 feature 요구사항에 따라 토큰 밀도를 할당하는 **tri-segment 아키텍처**를 사용한다: 하단 레이어에서는 효율적인 전역 특성 추출을 위한 **Poolingformer**, 중간 레이어에서는 전역 문맥과 지역 디테일의 균형을 위한 **Sparse-Dense Token Modules (SDTM)**, 상단 레이어에서는 고주파 디테일을 정제하는 **dense tokens**을 사용한다.

각 세그먼트의 역할을 수식으로 표현하면:

**① Bottom Segment — Poolingformer:**

토큰 $\mathbf{X} \in \mathbb{R}^{N \times d}$에 대해 풀링 연산을 통해 압축된 표현 $\mathbf{X}' \in \mathbb{R}^{N' \times d}$ ($N' \ll N$)을 생성합니다.

$$\mathbf{X}' = \text{Pool}(\mathbf{X}), \quad N' = \lfloor N / r \rfloor$$

여기서 $r$은 풀링 비율(pooling ratio)입니다.

**② Middle Segment — SDTM:**

중간 세그먼트는 여러 SDTM으로 구성되며, "Sparse Gen(sparse token generation transformers)"과 "Dense Rec(dense token recovery transformers)"가 교대로 배치되어 전역 및 세부 정보 처리의 균형을 맞춘다.

SDTM의 동작을 수식으로 표현하면:

$$\mathbf{X}_{\text{sparse}} = \text{SparseGen}(\mathbf{X}_{\text{full}}), \quad |\mathbf{X}_{\text{sparse}}| = \lfloor (1 - \rho) \cdot N \rfloor$$

$$\mathbf{X}_{\text{full}}' = \text{DenseRec}(\mathbf{X}_{\text{sparse}}, \mathbf{X}_{\text{full}})$$

여기서 $\rho$는 pruning rate, Sparse Gen은 중요도가 낮은 토큰을 제거하고, Dense Rec는 제거된 토큰 위치를 보간(interpolate)하여 복원합니다.

**③ Top Segment — Dense Transformer:**

상단 세그먼트는 최종 처리를 위한 표준 트랜스포머("Tr")를 포함한다.

$$\mathbf{X}_{\text{out}} = \text{Transformer}(\mathbf{X}_{\text{full}}')$$

#### (B) 시간적(Temporal) Timestep-wise Pruning

시간적으로, SparseDiT는 디노이징 단계 전반에 걸쳐 토큰 밀도를 동적으로 조절하며, 후기 타임스텝에서 더 세밀한 디테일이 나타남에 따라 점진적으로 토큰 수를 증가시킨다.

타임스텝 $t$에서의 pruning rate $\rho(t)$는 다음과 같이 정의됩니다 (논문의 전략을 수식화):

$$\rho(t) = \rho_{\max} \cdot \left(1 - \frac{t_{\max} - t}{t_{\max}}\right)^\alpha, \quad t \in [0, t_{\max}]$$

즉, $t$가 작을수록(초기 노이즈 단계) $\rho(t)$가 크고, $t$가 클수록(후기 정제 단계) $\rho(t)$가 작아져 더 많은 토큰을 사용합니다.

#### (C) 전체 FLOPs 복잡도 비교

표준 DiT의 Self-Attention FLOPs:
$$\mathcal{O}_{\text{DiT}} = O(N^2 \cdot d)$$

SparseDiT의 효과적 FLOPs (SDTM 적용 시):
$$\mathcal{O}_{\text{SparseDiT}} \approx O\left((1-\rho)^2 N^2 \cdot d\right)$$

---

### 2-3. 모델 구조 상세

DiT-XL의 경우, 모델은 하단, 중간, 상단 세그먼트에 각각 2개, 24개, 2개의 트랜스포머로 구성된다.

```
[Input Latent Tokens: N개]
        ↓
┌─────────────────────────────┐
│ Bottom: Poolingformer (×2)  │  → 전역 특성, 저밀도 토큰
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ Middle: SDTM (×24)          │  → Sparse Gen ↔ Dense Rec 교번
│  [SparseGen → DenseRec]×N  │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ Top: Dense Transformer (×2) │  → 고주파 디테일 정제
└─────────────────────────────┘
        ↓
[Output: 생성된 이미지/비디오]
```

SparseDiT의 공간적 적응형 아키텍처와 시간적 프루닝 전략 간의 시너지는 생성 과정 전반에 걸쳐 효율성과 충실도를 균형 있게 유지하는 통합 프레임워크를 가능하게 한다.

---

### 2-4. 성능 향상

실험 결과 FlexDiT(=SparseDiT의 최종명)는 다음과 같은 성능을 달성했다:
- DiT-XL에서 **FLOPs 55% 감소**, **추론 속도 175% 향상**, FID 점수는 단 0.09 증가에 그침 ( $512 \times 512$ ImageNet)
- 비디오 생성 데이터셋(FaceForensics, SkyTimelapse, UCF101, Taichi-HD)에서 **FLOPs 56% 감소**
- PixArt- $\alpha$ 텍스트-이미지 생성에서 **추론 속도 69% 향상**, FID 점수 0.24 감소(향상)

SparseDiT는 DiT, Latte, PixArt- $\alpha$ 등 세 가지 대표적인 DiT 기반 모델에 적용되어 클래스 조건부 이미지 생성, 클래스 조건부 비디오 생성, 텍스트-이미지 생성을 각각 수행했다.

파인튜닝은 처음부터 학습하는 데 필요한 시간의 약 **6%** 만 필요하다(예: DiT-XL 파인튜닝은 400K 이터레이션).

---

### 2-5. 한계

논문에서 명시적으로 언급된 한계와 추론 가능한 한계는 다음과 같습니다:

| 한계 | 설명 |
|------|------|
| **파인튜닝 필요** | 처음부터 학습하지 않고 파인튜닝 방식이나, 여전히 추가 학습 비용이 존재 |
| **정적 프루닝 스케줄** | Timestep-wise 프루닝 비율이 사전에 정의되어 있어 입력 콘텐츠의 복잡도에 완전히 적응하지 못할 가능성 |
| **적용 모델 제한** | DiT, Latte, PixArt- $\alpha$ 등 특정 모델군에서 검증되었으며, 다른 아키텍처로의 일반화 여부는 미검증 |
| **Dense Rec의 복원 품질** | Sparse token에서 Dense token을 복원하는 과정에서 세밀한 디테일 손실 가능성 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화를 뒷받침하는 요소

**(1) 다중 모달리티 및 다양한 태스크에서 검증:**

SparseDiT는 DiT(클래스 조건부 이미지 생성), Latte(클래스 조건부 비디오 생성), PixArt- $\alpha$(텍스트-이미지 생성) 등 세 가지 대표 DiT 기반 모델에 적용되었다. 이는 단일 태스크에 특화되지 않은 일반적 프레임워크임을 의미합니다.

**(2) 파인튜닝 효율성을 통한 범용성:**

파인튜닝에는 처음부터 학습하는 데 필요한 시간의 약 6%만이 소요된다. 이는 다른 사전학습 모델에도 빠르게 적용될 수 있는 일반화 가능성을 시사합니다.

**(3) SDTM의 구조적 일반성:**

SDTM은 Sparse Gen과 Dense Rec이 교대로 배치되어 전역 및 세부 정보 처리의 균형을 맞춘다. 이 모듈은 특정 도메인에 종속되지 않으며, 임의의 DiT 계열 모델에 플러그인 방식으로 통합 가능합니다.

**(4) 샘플링 최적화 기법과의 호환성:**

이 결과들은 SparseDiT가 효율적이고 고품질의 확산 기반 생성 아키텍처를 제공하며, 향상된 효율성을 위한 추가 샘플링 최적화 기법과 호환 가능함을 보여준다.

### 3-2. 일반화를 위한 수식적 분석

SparseDiT의 일반화 능력은 다음의 수식적 틀로 이해할 수 있습니다.

임의의 DiT 기반 모델 $\mathcal{M}$에 대해 SparseDiT 변환 $\mathcal{T}$를 적용:

$$\mathcal{M}_{\text{SparseDiT}} = \mathcal{T}(\mathcal{M}) = \mathcal{M}_{\text{bottom}}^{\text{Pool}} \circ \mathcal{M}_{\text{mid}}^{\text{SDTM}} \circ \mathcal{M}_{\text{top}}^{\text{Dense}}$$

파인튜닝 손실 함수는 원래 DiT의 디노이징 목적함수와 동일하게 유지됩니다:

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t}\left[\left\|\epsilon - \epsilon_\theta(x_t, t, c)\right\|^2\right]$$

여기서 $c$는 조건(클래스 레이블 또는 텍스트), $x_t$는 타임스텝 $t$에서의 노이즈가 추가된 샘플입니다. **손실 함수를 변경하지 않고** 아키텍처만 수정하기 때문에 일반화 손실이 최소화됩니다.

### 3-3. 일반화의 잠재적 한계

순진한(naive) 토큰 드롭핑은 표현(representation)을 저하시키고, 추론 시 전체 토큰 입력으로 평가될 때 일반화 성능이 저하될 수 있다. SparseDiT는 SDTM의 Dense Rec를 통해 이를 완화하려 하지만, 완전한 해결책인지는 추가 연구가 필요합니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4-1. 연구에 미치는 영향

**(1) DiT 효율화 연구의 표준적 벤치마크로 자리매김:**
SparseDiT는 공간+시간 이중 희소화 전략을 함께 적용한 초기 연구로, 이후 관련 연구의 비교 기준이 됩니다.

**(2) Video DiT로의 연구 확장 촉진:**

이후 Sparse-vDiT 등의 연구는 희소성 가속화 프레임워크를 제안하며, 각 희소성 패턴에 대해 계산 효율적인 구현으로 밀집(dense) 어텐션을 대체하고, 하드웨어 인식 비용 모델링을 통해 레이어 및 헤드별로 최적의 희소 계산 전략을 선택하는 오프라인 희소 확산 탐색 알고리즘을 제안하는 방향으로 발전했다.

**(3) 학습 효율화로의 파급:**

DiT의 학습 비용은 시퀀스 길이에 따라 이차적으로 증가하여 대규모 사전학습이 계산적으로 매우 비싸진다. 비용을 줄이는 자연스러운 방법은 토큰을 드롭하여 시퀀스를 단축하는 것이다. SparseDiT의 SDTM 아이디어는 학습 단계에서도 활용 가능한 방향으로 연구될 수 있습니다.

**(4) 다양한 모달리티로의 확장:**

Sparse-vDiT는 CogVideoX1.5, HunyuanVideo, Wan2.1 등의 최신 Video DiT 모델에 통합되어 각각 2.09×, 2.38×, 1.67×의 이론적 FLOPs 감소와 1.76×, 1.85×, 1.58×의 실제 추론 속도 향상을 달성했다.

---

### 4-2. 앞으로 연구 시 고려할 점

#### ① 입력 적응적(Content-Adaptive) 토큰 선택
현재 SparseDiT의 pruning rate $\rho(t)$는 타임스텝에 의존하지만 입력 콘텐츠의 복잡도와는 무관합니다. 이미지의 엔트로피나 주의 분포를 반영한 동적 선택이 필요합니다:

$$\rho(t, x) = f\left(t, \mathcal{H}(\mathbf{A}_{x})\right)$$

여기서 $\mathcal{H}(\mathbf{A}_{x})$는 입력 $x$의 어텐션 맵 엔트로피입니다.

#### ② Training-Free 적용 가능성 탐색

어텐션 맵에서의 중복성을 활용하여 DiT 어텐션의 높은 계산 비용을 완화하는 연구들이 이루어지고 있으며, 이는 훈련 없는(training-free) 방식과 훈련 기반(training-based) 방식으로 구분된다. SparseDiT의 파인튜닝 비용을 완전히 제거하는 Training-Free 버전 연구가 필요합니다.

#### ③ Dense Rec의 품질 보장

일부 방법들은 상당한 파라미터를 추가하거나 적당한 드롭 비율만 지원하며, 공격적인 설정(예: 75%)에서는 성능이 저하된다. SparseDiT의 Dense Rec 모듈이 고비율 희소화에서도 안정적으로 동작하도록 개선이 필요합니다.

#### ④ 양자화(Quantization)와의 결합
토큰 희소화와 가중치 양자화를 결합하면 추가적인 효율 향상이 가능합니다:

$$\text{FLOPs}_{\text{total}} \approx O\left((1-\rho)^2 N^2 \cdot \frac{d}{b}\right)$$

여기서 $b$는 비트 수(quantization bit)입니다.

#### ⑤ 고해상도 및 초장기 비디오로의 확장
비디오 DiT의 잠재적인 구조적 희소성은 장기 비디오 합성을 위해 체계적으로 활용될 수 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법론 | 핵심 기여 | FLOPs 절감 |
|------|--------|-----------|-----------|
| **DiT** (Peebles et al., 2023) | Transformer 기반 확산 모델 | Scalable 생성 아키텍처 제시 | 기준선 |
| **SparseDiT** (Chang et al., 2024) | 공간+시간 토큰 희소화 | Tri-segment + SDTM + Timestep pruning | **~55%** |
| **Dynamic DiT** (2024) | 타임스텝 및 공간 적응형 계산 | 입력 복잡도 기반 동적 폭 조절 | 가변 |
| **SPRINT** (2025) | Sparse-Dense Residual Fusion | 고비율 토큰 드롭을 가능하게 하면서 완전한 토큰 파인튜닝으로의 전이 가능한 견고한 표현 유지; 최소한의 아키텍처 변경으로 강력한 기준선에 상응하거나 더 나은 성능 달성 | ImageNet-1K $256^2$에서 기준 모델 대비 훈련 비용 최대 **5.6×, 9.8×** 절감 |
| **Sparse-vDiT** (2025) | 어텐션 패턴 기반 비디오 희소화 | CogVideoX1.5, HunyuanVideo, Wan2.1에서 각각 2.09×, 2.38×, 1.67× 이론적 FLOPs 감소 달성 | 최대 ~2.38× |

### 연구 흐름 요약

```
DiT (2023, 기준모델)
    ↓ 이차 복잡도 문제
SparseDiT (2024, arXiv 2412.06028) — 공간+시간 이중 희소화
    ↓ 훈련 비용 문제
SPRINT (2025, arXiv 2510.21986) — 고비율 토큰 드롭 + 표현 품질 보존
    ↓ 비디오 확장
Sparse-vDiT (2025, arXiv 2506.03065) — 비디오 DiT 특화 희소 어텐션
```

---

## 참고 자료 (출처)

1. **SparseDiT 논문 (arXiv)**:
   - https://arxiv.org/abs/2412.06028

2. **SparseDiT 논문 HTML 전문 (arXiv)**:
   - https://arxiv.org/html/2412.06028v2

3. **SparseDiT OpenReview**:
   - https://openreview.net/forum?id=jTBxyQempF

4. **SparseDiT OpenReview PDF**:
   - https://openreview.net/pdf/f82961ff7a65d91e46493e3b65c1cf3d16e4bec9.pdf

5. **SPRINT (Sparse-Dense Residual Fusion for Efficient DiT)**:
   - https://arxiv.org/pdf/2510.21986

6. **Sparse-vDiT (비디오 DiT 가속화)**:
   - https://arxiv.org/pdf/2506.03065

7. **Dynamic Diffusion Transformer**:
   - https://arxiv.org/pdf/2410.03456

8. **FlexDiT (SparseDiT의 최종 명칭 버전)**:
   - GitHub: https://github.com/changsn/FlexDiT

> ⚠️ **정확도 관련 안내**: 논문 내 일부 상세 수식(SDTM 내부 구현, Poolingformer 구체적 파라미터 등)은 공개된 arXiv HTML 버전에서 부분적으로 확인되었으며, 수식 일부는 논문의 서술 방향을 기반으로 형식화하였습니다. 완전한 수식 확인을 위해서는 논문 원문 PDF(arXiv:2412.06028) 직접 열람을 권장합니다.
