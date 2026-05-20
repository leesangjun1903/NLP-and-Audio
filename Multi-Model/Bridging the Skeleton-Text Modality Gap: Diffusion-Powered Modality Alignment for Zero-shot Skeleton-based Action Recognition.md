# Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문은 **Zero-shot Skeleton-based Action Recognition (ZSAR)** 문제에서, 기존 방법들이 스켈레톤 잠재 공간과 텍스트 잠재 공간 사이의 **직접 정렬(direct alignment)** 에 의존함으로써 발생하는 **모달리티 갭(modality gap)** 문제를 지적한다. 이를 해결하기 위해 **확산 모델(Diffusion Model)** 의 크로스-모달리티 정렬 능력을 활용하여, 스켈레톤 피처를 텍스트 프롬프트 조건 하에 역확산 과정을 통해 **암묵적으로(implicitly)** 정렬하는 새로운 프레임워크 **TDSM(Triplet Diffusion for Skeleton-Text Matching)** 을 최초로 제안한다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **① 최초의 확산 기반 ZSAR 프레임워크** | 확산 모델을 ZSAR에 최초 적용; 역확산 과정에서 텍스트 프롬프트로 조건화하여 통합 잠재 공간 형성 |
| **② Triplet Diffusion (TD) Loss 도입** | 올바른 스켈레톤-텍스트 쌍은 당기고 잘못된 쌍은 밀어내는 판별적 손실 함수 설계 |
| **③ SOTA 대비 대폭 성능 향상** | NTU-60, NTU-120, PKU-MMD에서 기존 SOTA 대비 2.36%p~13.05%p 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

ZSAR의 핵심 난제는 **스켈레톤 데이터(시공간적 동작 패턴)** 와 **텍스트 설명(고수준 의미 정보)** 사이의 **모달리티 갭**이다.

- **기존 VAE 기반 방법**: CADA-VAE, SynSE, MSF, SA-DVAE 등은 크로스-재구성 손실로 두 잠재 공간을 직접 정렬 → 모달리티 갭으로 인한 일반화 한계
- **기존 대조학습 기반 방법**: SMIE, PURLS, STAR, DVTA, InfoCPL 등은 특징 거리 최소화 방식 → 동일한 근본 문제 내포

$$\mathcal{Y} \cap \mathcal{Y}^u = \emptyset$$

학습 시 본 클래스 집합 $\mathcal{Y}$와 테스트 시 미지 클래스 집합 $\mathcal{Y}^u$는 완전히 분리되어, 미지 클래스에 대한 강건한 일반화가 필수적이다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 순방향 확산 과정 (Forward Process)

스켈레톤 피처 $\mathbf{z}_x$에 랜덤 타임스텝 $t \sim \mathcal{U}(T)$에서 가우시안 노이즈를 추가:

$$\mathbf{z}_{x,t} = \sqrt{\bar{\alpha}_t}\,\mathbf{z}_x + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t}(1-\beta_s)$는 타임스텝 $t$에서의 노이즈 레벨을 제어한다.

#### (2) 역방향 확산 과정 (Reverse Process)

Diffusion Transformer $\mathcal{T}_\text{diff}$가 글로벌/로컬 텍스트 피처 $\mathbf{z}_g$, $\mathbf{z}_l$로 조건화되어 노이즈를 예측:

$$\hat{\boldsymbol{\epsilon}} = \mathcal{T}_\text{diff}(\mathbf{z}_{x,t},\, t;\, \mathbf{z}_g,\, \mathbf{z}_l)$$

- **양성 샘플(positive)**: GT 레이블 $y_p$의 텍스트 피처 $(\mathbf{z}\_{g,p}, \mathbf{z}_{l,p})$ → $\hat{\boldsymbol{\epsilon}}_p$ 예측
- **음성 샘플(negative)**: 잘못된 레이블 $y_n \in \mathcal{Y} \setminus \{y_p\}$의 텍스트 피처 $(\mathbf{z}\_{g,n}, \mathbf{z}_{l,n})$ → $\hat{\boldsymbol{\epsilon}}_n$ 예측
- 두 경로는 $\mathcal{T}_\text{diff}$의 **공유 가중치(weight sharing)** 를 사용

#### (3) 텍스트 인코딩

CLIP 텍스트 인코더 $\mathcal{E}_d$로 프롬프트 $\mathbf{d}$를 인코딩:

$$[\mathbf{z}_g \mid \mathbf{z}_l] = \mathcal{E}_d(\mathbf{d})$$

- $\mathbf{z}_g \in \mathbb{R}^{1 \times C}$: 전체 문장의 글로벌 텍스트 피처 (고수준 의미)
- $\mathbf{z}_l \in \mathbb{R}^{M_l \times C}$: 토큰 단위 로컬 텍스트 피처 (세밀한 의미)

#### (4) 손실 함수

전체 손실:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{diff} + \lambda \mathcal{L}_\text{TD}$$

**확산 손실 (Diffusion Loss)**:

$$\mathcal{L}_\text{diff} = \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_p\|_2$$

**트리플렛 확산 손실 (Triplet Diffusion Loss)**:

$$\mathcal{L}_\text{TD} = \max\!\left(\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_p\|_2 - \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_n\|_2 + \tau,\; 0\right)$$

- $\tau$: 마진 파라미터 (실험에서 $\lambda = \tau = 1.0$으로 설정)
- $\mathcal{L}_\text{TD}$는 올바른 쌍의 노이즈 예측 거리를 최소화하고, 잘못된 쌍의 거리를 최대화하여 **판별적 피처 융합**을 강화

#### (5) 추론 (Inference)

미지의 스켈레톤 시퀀스 $\mathbf{X}^u$에 대해 고정 노이즈 $\boldsymbol{\epsilon}\_\text{test} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$와 고정 타임스텝 $t_\text{test}$를 사용:

$$\mathbf{z}^u_{x,t} = \sqrt{\bar{\alpha}_{t_\text{test}}}\,\mathbf{z}^u_x + \sqrt{1 - \bar{\alpha}_{t_\text{test}}}\,\boldsymbol{\epsilon}_\text{test}$$

각 후보 레이블 $y^u_k \in \mathcal{Y}^u$에 대해 예측 노이즈 $\hat{\boldsymbol{\epsilon}}_k$를 계산하고, 최소 $\ell_2$ 거리 레이블 선택:

$$\hat{y}^u = \arg\min_k \|\boldsymbol{\epsilon}_\text{test} - \hat{\boldsymbol{\epsilon}}_k\|_2$$

이는 **단일 스텝 추론(one-step inference)** 으로, 반복적 샘플링 없이 효율적으로 동작한다.

---

### 2.3 모델 구조

```
입력 스켈레톤 X → [Skeleton Encoder Ex (GCN: ST-GCN/Shift-GCN)]
                     → zx ∈ R^{Mx×256}
                     → Forward Process → zx,t
                     
텍스트 프롬프트 d → [Text Encoder Ed (CLIP)]
                     → zg ∈ R^{1×1024}, zl ∈ R^{Ml×1024}

Diffusion Transformer Tdiff (DiT 기반):
  - fx,t = Linear(zx,t) + PEx
  - fc   = Linear(TEt) + Linear(zg)
  - fl   = Linear(zl) + PEl
  
  → B=12 CrossDiT Blocks
      각 블록: Scale-Shift 변조 + Multi-Head Self-Attention (12 heads)
      [qx|ql][kx|kl]^T → Softmax → [vx|vl]
  
  → Layer Norm → Linear → ε̂ ∈ R^{Mx×256}
```

**CrossDiT Block 핵심 연산**:

$$[\boldsymbol{\alpha}_x \mid \boldsymbol{\beta}_x \mid \boldsymbol{\gamma}_x \mid \boldsymbol{\alpha}_l \mid \boldsymbol{\beta}_l \mid \boldsymbol{\gamma}_l] = \text{Linear}(\mathbf{f}_c)$$

$$\text{Scale-Shift}: \mathbf{f}_i \leftarrow (1 + \boldsymbol{\gamma}_i) \odot \mathbf{f}_i + \boldsymbol{\beta}_i$$

$$[\mathbf{f}_x \mid \mathbf{f}_l] \leftarrow \text{SoftMax}\!\left([\mathbf{q}_x \mid \mathbf{q}_l][\mathbf{k}_x \mid \mathbf{k}_l]^\top\right)[\mathbf{v}_x \mid \mathbf{v}_l]$$

---

### 2.4 성능 향상

#### SynSE/PURLS 벤치마크 (NTU-60, NTU-120)

| 방법 | NTU-60 55/5 | NTU-60 48/12 | NTU-120 110/10 | NTU-120 96/24 |
|------|------------|-------------|---------------|--------------|
| PURLS (CVPR 2024) | 79.23 | 40.99 | 71.95 | 52.01 |
| SA-DVAE (ECCV 2024) | 82.37 | 41.38 | 68.77 | 46.12 |
| STAR (ACM MM 2024) | 81.40 | 45.10 | 63.30 | 44.30 |
| **TDSM (Ours)** | **86.49** | **56.03** | **74.15** | **65.06** |

#### SMIE 벤치마크

| 방법 | NTU-60 55/5 | NTU-120 110/10 | PKU-MMD 46/5 |
|------|------------|---------------|-------------|
| SA-DVAE | 84.20 | 50.67 | 66.54 |
| STAR | 77.50 | - | 70.60 |
| **TDSM (Ours)** | **88.88** | **69.47** | **70.76** |

- NTU-120 96/24 split에서 2위 대비 **+13.05%p** 향상 (가장 큰 마진)
- Kinetics-200, Kinetics-400 등 더 복잡한 데이터셋에서도 SOTA 달성

---

### 2.5 한계점

논문이 명시적으로 인정한 한계:

1. **추론 시 노이즈 민감성**: 고정 노이즈 $\boldsymbol{\epsilon}_\text{test}$ 에 따라 정확도가 최대 ±2.5%p 변동 → 저자들은 향후 $\mathbf{z}_x$ 직접 예측 방식으로 개선 가능성 언급 (변동폭 5배 감소 확인)

2. **의미적으로 유사한 동작 구별의 어려움**: "전화 받기", "모자 쓰기", "안경 착용" 등 동일한 상향 손 동작을 공유하는 클래스들을 스켈레톤만으로 구별하는 데 한계

3. **맥락 정보 부재**: 스켈레톤 데이터는 배경/객체 정보가 없어 객체-행동 맥락 의존 클래스에서 성능 저하

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 핵심 메커니즘

#### ① 암묵적 정렬을 통한 통합 잠재 공간 형성

기존 직접 정렬 방식은 두 이질적 모달리티 공간 사이의 명시적 매핑을 학습하므로, 학습 시 보지 못한 클래스(unseen)에 대한 매핑이 불안정하다. TDSM은 역확산 과정에서 텍스트 조건화를 통해 **암묵적으로** 두 모달리티를 하나의 잠재 공간으로 융합하므로, 미지 클래스에 대한 의미적 외삽(semantic extrapolation)이 더 자연스럽게 이루어진다.

#### ② 랜덤 가우시안 노이즈의 정규화 효과

학습 시 매 스텝 새로운 랜덤 노이즈 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$를 추가함으로써:

$$\mathbf{z}_{x,t} = \sqrt{\bar{\alpha}_t}\,\mathbf{z}_x + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$$

이 노이즈의 확률적 특성이 **자연스러운 데이터 증강** 역할을 하여 특정 노이즈 패턴에 과적합되는 것을 방지한다. 고정 노이즈 사용 시 성능이 현저히 떨어지는 Ablation(Table 6)이 이를 직접 증명한다:

| 노이즈 유형 | NTU-60 55/5 | NTU-60 48/12 |
|-----------|------------|-------------|
| 고정 | 76.40 | 44.25 |
| **랜덤** | **86.49** | **56.03** |

#### ③ Triplet Diffusion Loss의 판별력 강화

$$\mathcal{L}_\text{TD} = \max\!\left(\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_p\|_2 - \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_n\|_2 + \tau,\; 0\right)$$

이 손실은 학습 데이터의 seen 클래스 내에서 클래스 간 경계를 더욱 명확하게 학습하게 하여, 미지 클래스가 추론 시 등장해도 의미적 유사성을 기반으로 올바른 레이블을 선택할 수 있는 **범용적 판별 능력**을 부여한다.

#### ④ 글로벌+로컬 텍스트 피처의 상보적 활용

| 피처 조합 | NTU-60 55/5 | NTU-60 48/12 |
|---------|------------|-------------|
| $\mathbf{z}_g$ only | 83.41 | 51.50 |
| $\mathbf{z}_l$ only | 83.33 | 52.63 |
| **$\mathbf{z}_g + \mathbf{z}_l$** | **86.49** | **56.03** |

글로벌 피처는 동작 전체의 고수준 의미를, 로컬 피처는 미세한 단어 수준 의미를 포착하여 의미적으로 유사한 클래스 간 구별력을 높인다.

#### ⑤ 최적 타임스텝 T의 역할

| T | NTU-60 55/5 | NTU-60 48/12 |
|---|------------|-------------|
| 1 | 85.03 | 44.10 |
| 10 | 84.51 | 50.89 |
| **50** | **86.49** | **56.03** |
| 100 | 83.48 | 56.27 |
| 500 | 81.34 | 53.43 |

$T$가 너무 작으면 과적합, 너무 크면 노이즈 다양성 과다로 디노이징이 어려워진다. $T=50$이 일반화와 과적합 방지 사이의 최적 균형점이다.

#### ⑥ 단일 텍스트 프롬프트로도 경쟁력 있는 성능

PURLS(7개 텍스트 프롬프트)보다 단일 프롬프트만으로 Kinetics-200/400에서 더 나은 성능을 달성함으로써, 확산 기반 정렬의 근본적인 강건성을 입증한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 발표 | 핵심 기법 | 정렬 방식 | NTU-60 55/5 | NTU-120 110/10 | 한계 |
|------|------|----------|---------|------------|--------------|------|
| SynSE (2021) | ICIP 2021 | 동사/명사 분리 VAE | 직접 (VAE) | 75.81 | 62.69 | 구조적 의미 부족 |
| SMIE (2023) | ACM MM 2023 | 마스킹 + 대조학습 | 직접 (CL) | 77.98 | 65.74 | 모달리티 갭 |
| MSF (2023) | ICIG 2023 | 다중 의미 융합 VAE | 직접 (VAE) | - | - | 제한된 의미 표현 |
| PURLS (2024) | CVPR 2024 | GPT-3, 신체부위별 텍스트 | 직접 (CL) | 79.23 | 71.95 | 다중 프롬프트 의존 |
| SA-DVAE (2024) | ECCV 2024 | 의미 관련/무관 분리 VAE | 직접 (VAE) | 82.37 | 68.77 | 모달리티 갭 |
| STAR (2024) | ACM MM 2024 | GPT-3.5, 학습 가능 프롬프트 | 직접 (CL) | 81.40 | 63.30 | 모달리티 갭 |
| DVTA (2024) | arXiv 2024 | 이중 정렬 + 텍스트 증강 | 직접 이중 (CL) | - | - | 복잡도 증가 |
| InfoCPL (2024) | arXiv 2024 | 100개 고유 문장 생성 | 직접 (CL) | - | - | 프롬프트 생성 비용 |
| **TDSM (2025)** | **arXiv 2025** | **확산 기반 암묵적 정렬** | **간접 (확산)** | **86.49** | **74.15** | 노이즈 민감성 |

**핵심 차별점 정리**:

```
VAE 기반: Skeleton ↔ Text 크로스 재구성 (명시적, 직접적)
CL 기반:  Skeleton ↔ Text 특징 거리 최소화 (명시적, 직접적)
TDSM:    Text 조건 하 Skeleton 노이즈 제거 학습 (암묵적, 간접적)
         → 모달리티 갭을 우회하여 통합 잠재 공간 자연스럽게 형성
```

---

## 5. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 5.1 연구에 미치는 영향

#### ① 확산 모델의 새로운 적용 패러다임 확립

기존 확산 모델 연구는 주로 **생성(generation)** 에 집중되었다. TDSM은 확산 모델을 **판별적 크로스-모달 정렬** 도구로 사용하는 새로운 패러다임을 제시한다. 이는 다음 분야로의 확장을 자극할 것이다:
- 비디오-텍스트 제로샷 인식
- 포인트 클라우드-텍스트 정렬
- 의료 시계열-레이블 정렬

#### ② 소규모 도메인에서의 확산 모델 가능성 입증

LAION-5B 같은 대규모 데이터 없이, 도메인 특화 소규모 데이터(스켈레톤)에서도 확산 모델이 효과적임을 실증적으로 보여준다. 이는 데이터 희소 환경에서의 확산 모델 연구를 촉진할 것이다.

#### ③ 제로샷 학습에서의 모달리티 갭 해결 전략 제시

직접 매핑 대신 확산 과정을 통한 **간접 정렬**이 모달리티 갭 문제를 근본적으로 우회할 수 있음을 보여준다. 이는 이질적 모달리티 간 정렬 연구 전반에 영향을 줄 것이다.

### 5.2 향후 연구 시 고려할 점

#### ① 노이즈 민감성 해결

추론 시 $\boldsymbol{\epsilon}_\text{test}$에 따른 성능 변동(±2.5%p)을 안정화하기 위해:
- **$\mathbf{z}_x$ 직접 예측** 방식 탐색 (논문에서도 변동폭 5배 감소 확인)
- **앙상블 추론** 전략 (현재 10회 평균 사용)
- **결정론적 DDIM 샘플링** 적용 가능성 검토

#### ② 스켈레톤 인코더의 개선

현재 ST-GCN, Shift-GCN에 의존하는데, 더 강력한 스켈레톤 인코더(예: SkateFormer, transformer 기반 모델)와의 결합으로 추가 성능 향상 가능성이 있다.

#### ③ 일반화된 제로샷(Generalized ZSL) 설정 탐구

현재 순수 제로샷 설정만 평가되었으나, seen + unseen 클래스를 모두 추론해야 하는 **GZSL(Generalized Zero-Shot Learning)** 설정에서의 성능 검증이 필요하다.

#### ④ 멀티모달 입력 확장

현재 텍스트-스켈레톤 두 모달리티만 사용하나:
- RGB 영상 피처와의 결합 (BSZSL 비교에서 일부 열세)
- 오디오, 깊이 맵 등 추가 모달리티 통합
- **조건 신호 다양화**: 텍스트 이외에 참조 영상, 동작 설명 그래프 등

#### ⑤ 의미적으로 유사한 동작 클래스 구별

"전화 받기" vs "모자 쓰기" vs "안경 착용" 같이 스켈레톤 패턴이 유사한 클래스들을 구별하기 위해:
- 더 세밀한 신체 부위별 텍스트 설명 활용 (PURLS 방식과의 하이브리드)
- 계층적(hierarchical) 의미 구조를 반영한 프롬프트 설계

#### ⑥ 타임스텝 $t_\text{test}$ 자동 최적화

현재 $t_\text{test} = 25$를 경험적으로 설정하는데, 데이터셋이나 클래스 특성에 따라 동적으로 최적 타임스텝을 선택하는 **적응적 추론 전략** 연구가 필요하다.

---

## 참고 자료

- **논문 원문**: Jeonghyeok Do, Munchurl Kim. "Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition." arXiv:2411.10745v4, 2025. (제공된 PDF 전문)
- **프로젝트 페이지**: https://kaist-viclab.github.io/TDSM_site
- Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020 (논문 내 참조 [22])
- Peebles & Xie, "Scalable Diffusion Models with Transformers (DiT)," ICCV 2023 (논문 내 참조 [48])
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022 (논문 내 참조 [52])
- Radford et al., "Learning Transferable Visual Models from Natural Language Supervision (CLIP)," ICML 2021 (논문 내 참조 [51])
- Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition," AAAI 2018 (논문 내 참조 [71])
- Cheng et al., "Skeleton-Based Action Recognition with Shift Graph Convolutional Network," CVPR 2020 (논문 내 참조 [9])
- Zhu et al., "Part-aware Unified Representation of Language and Skeleton for Zero-shot Action Recognition (PURLS)," CVPR 2024 (논문 내 참조 [79])
- Zhou et al., "Zero-Shot Skeleton-Based Action Recognition via Mutual Information Estimation and Maximization (SMIE)," ACM MM 2023 (논문 내 참조 [77])
