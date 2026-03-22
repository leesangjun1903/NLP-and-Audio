# Language Model Beats Diffusion — Tokenizer is Key to Visual Generation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
**언어 모델(LM)이 시각적 생성(visual generation)에서 확산 모델(diffusion model)에 뒤처지는 근본 원인은 모델 아키텍처가 아니라, 좋은 "시각적 토크나이저(visual tokenizer)"의 부재에 있다.** 우수한 토크나이저를 갖추면, 언어 모델이 동일 데이터·모델 크기·학습 예산 하에서 확산 모델을 능가할 수 있다.

### 주요 기여
| 기여 항목 | 내용 |
|---|---|
| **MAGVIT-v2** | 이미지와 비디오를 공유 어휘(shared vocabulary)로 토크나이징하는 새로운 비디오 토크나이저 |
| **Lookup-Free Quantization (LFQ)** | 코드북 임베딩 룩업 없이 대규모 어휘( $2^{18} \approx 262$ K)를 효과적으로 학습 가능하게 하는 양자화 방법 |
| **LM > Diffusion 최초 증거** | ImageNet 벤치마크에서 동일 조건 하 언어 모델이 확산 모델을 처음으로 능가 |
| **비디오 압축** | HEVC를 초과하고 차세대 코덱 VVC에 필적하는 압축 품질 (인간 평가 기준) |
| **영상 이해** | 비디오 액션 인식에서 기존 최고 토크나이저(MAGVIT) 대비 성능 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 시각적 생성에서 언어 모델은 확산 모델 대비 상당한 성능 격차를 보였다. 예컨대 ImageNet $256 \times 256$에서 최고 언어 모델(Lee et al., 2022)의 FID는 3.41인 반면, 최고 확산 모델(Gao et al., 2023)은 1.79로 약 48% 우위를 보였다.

이 격차의 핵심 원인으로 저자들은 **시각적 토크나이저의 한계**를 지목한다:
1. **어휘 크기 제한**: 기존 VQ-VAE 기반 토크나이저의 코드북은 1K~8K 수준으로, 자연어 어휘(200K+) 대비 매우 작음
2. **어휘 확장 시 생성 품질 저하**: 기존 VQ에서는 어휘를 키우면 재구성(reconstruction)은 개선되지만, 언어 모델의 생성(generation) 품질은 오히려 악화
3. **이미지-비디오 통합 토크나이징 불가**: 기존 MAGVIT의 3D CNN은 시간축 수용장(temporal receptive field) 문제로 단일 이미지 토크나이징이 어려움

### 2.2 제안하는 방법

#### (A) Lookup-Free Quantization (LFQ)

**핵심 아이디어**: VQ-VAE 코드북의 임베딩 차원을 **0**으로 줄여, 코드북 룩업을 완전히 제거한다.

기존 VQ-VAE에서는 인코더 출력 $\mathbf{z} \in \mathbb{R}^d$를 코드북 $\mathbf{C} \in \mathbb{R}^{K \times d}$에서 가장 가까운 엔트리에 매핑한다:

$$q(\mathbf{z}) = \mathbf{c}_i, \quad \text{where} \quad i = \arg\min_{j \in \{1,2,\cdots,K\}} \|\mathbf{z} - \mathbf{c}_j\|_2 $$

LFQ에서는 코드북을 정수 집합 $\mathbb{C}$로 대체하고 ($|\mathbb{C}| = K$), 잠재 공간을 단일 차원 변수들의 데카르트 곱으로 분해한다:

$$\mathbb{C} = \prod_{i=1}^{\log_2 K} \mathbb{C}_i $$

특징 벡터 $\mathbf{z} \in \mathbb{R}^{\log_2 K}$의 각 차원에 대해:

$$q(z_i) = \mathbb{C}_{i,j}, \quad \text{where} \quad j = \arg\min_k \|z_i - \mathbb{C}_{i,k}\| $$

이진 코드북 $\mathbb{C}_i = \{-1, 1\}$을 사용하면 부호 함수로 단순화된다:

$$q(z_i) = \text{sign}(z_i) = -\mathbf{1}\{z_i \leq 0\} + \mathbf{1}\{z_i > 0\} $$

토큰 인덱스는 다음과 같이 계산된다:

$$\text{Index}(\mathbf{z}) = \sum_{i=1}^{\log_2 K} 2^{i-1} \mathbf{1}\{z_i > 0\} $$

코드북 활용률을 높이기 위한 **엔트로피 페널티**:

$$\mathcal{L}_{\text{entropy}} = \mathbb{E}[H(q(\mathbf{z}))] - H[\mathbb{E}(q(\mathbf{z}))] $$

첫째 항은 개별 샘플의 양자화 엔트로피를 최소화(확정적 할당 유도), 둘째 항은 전체 분포의 엔트로피를 최대화(균등 코드북 사용 유도)한다. LFQ에서 차원 독립성을 이용하여 $H(q(\mathbf{z})) = \sum_{i=1}^{\log_2 K} H(q(z_i))$로 분해 가능하다.

**핵심 관찰 (Fig. 1)**: 기존 VQ에서는 어휘 크기 증가 시 재구성 FID는 개선되지만 생성 FID는 악화된다. 반면 LFQ에서는 어휘 크기 증가에 따라 **재구성과 생성 품질이 모두 일관되게 향상**된다.

전체 학습 손실은 다음의 조합이다:
- 재구성 손실 (Reconstruction loss)
- GAN 적대적 손실 (Adversarial loss)
- 지각 손실 (Perceptual loss)
- 커밋먼트 손실 (Commitment loss)
- 엔트로피 페널티 ($\mathcal{L}_{\text{entropy}}$, Eq. 5)
- LeCAM 정규화 (안정성 향상)

기존 VQ의 코드북 손실은 LFQ에서는 적용 불가하므로 제외된다.

#### (B) 시각적 토크나이저 모델 개선

**1) 인과적(Causal) 3D CNN 아키텍처**

이미지와 비디오를 공유 어휘로 토크나이징하기 위해, 기존 3D CNN의 시간축 패딩을 **인과적(causal)**으로 변경한다:
- 기존 3D 컨볼루션 (커널 크기 $(k_t, k_h, k_w)$ ): 시간축으로 앞뒤 $\lfloor\frac{k_t-1}{2}\rfloor$, $\lfloor\frac{k_t}{2}\rfloor$ 프레임 패딩
- 인과적 3D 컨볼루션: 시간축으로 앞쪽에만 $k_t - 1$ 프레임 패딩, 뒤쪽 패딩 없음

이로써 첫 번째 프레임은 항상 다른 프레임과 독립적이므로, 단일 이미지도 토크나이징 가능하다.

**2) 아키텍처 수정 사항**:
- 인코더 다운샘플러: 평균 풀링 → 스트라이드 컨볼루션
- 디코더 업샘플러: 최근접 리사이즈 + 컨볼루션 → Depth-to-Space 연산
- 시간축 다운샘플링: 앞쪽 블록 → 뒷쪽 블록으로 지연
- 판별기 다운샘플링: 3D 블러 풀링 (shift invariance 유도)
- 디코더에 Adaptive Group Normalization 추가 (양자화 잠재 변수를 제어 신호로 사용, StyleGAN 방식)

**3) 토큰 분해(Token Factorization)**:

큰 어휘($2^{18}$)에서 소규모 트랜스포머의 효율적 예측을 위해, LFQ 토큰의 잠재 공간을 동일한 하위 공간으로 분해한다. 예: $2^{18}$ 크기 대신 $2^9$ 크기의 두 개 연결 코드북으로 예측. 각 하위 공간 토큰을 별도 임베딩하고 합산하여 트랜스포머 입력으로 사용하며, weight tying 기법을 적용한다.

### 2.3 모델 구조

MAGVIT-v2의 전체 구조 (Fig. 7 참조):

| 구성 요소 | 세부 사항 |
|---|---|
| **인코더** | Temporally Causal 3D CNN, 채널 배율 [1, 2, 2, 4], 기본 채널 128, 잔차 블록 4개/해상도 |
| **양자화** | LFQ, 어휘 크기 $K = 2^{18}$, 잠재 형상 $5 \times 16 \times 16$ (비디오), $16 \times 16$ (이미지) |
| **디코더** | 인코더와 대칭, Depth-to-Space 업샘플링, Adaptive GroupNorm |
| **판별기** | 3D CNN, 채널 배율 [2, 4, 4, 4, 4], 3D 블러 풀링 다운샘플링 |
| **입력** | 비디오: 17 프레임 $128 \times 128$, 이미지: $256 \times 256$ 또는 $512 \times 512$ |

생성 모델로는 Masked Language Model (MLM, ~307M 파라미터)을 사용하며, MAGVIT과 동일한 트랜스포머 백본을 사용한다.

### 2.4 성능 향상

#### 이미지 생성 (ImageNet)

| 해상도 | 모델 | FID↓ (w/ guidance) | IS↑ | 파라미터 | 디코딩 스텝 |
|---|---|---|---|---|---|
| $512 \times 512$ | VDM++ (Diffusion, SOTA) | 2.65 | 278.1 | 2B | 512 |
| $512 \times 512$ | **MAGVIT-v2 (MLM+LFQ)** | **1.91** | **324.3** | 307M | 64 |
| $256 \times 256$ | MDT (Diff.+VAE, SOTA) | 1.79 | 283.0 | 676M | 250 |
| $256 \times 256$ | **MAGVIT-v2 (MLM+LFQ)** | **1.78** | **319.4** | 307M | 64 |

- $512 \times 512$에서 FID **28% 개선** (1.91 vs. 2.65)
- 모델 크기 **50% 이상 작고**, 디코딩 스텝 **약 4~8배 적음**

#### 비디오 생성

| 벤치마크 | 모델 | FVD↓ | 파라미터 | 스텝 |
|---|---|---|---|---|
| K600 (프레임 예측) | MAGVIT (이전 SOTA) | 9.9±0.3 | 306M | 12 |
| K600 | **MAGVIT-v2** | **4.3±0.1** | 307M | 24 |
| UCF-101 (조건부 생성) | MAGVIT | 76±2 | 306M | 12 |
| UCF-101 | **MAGVIT-v2** | **58±3** | 307M | 24 |

#### 비디오 압축
- LPIPS 기준 HEVC, VVC, MAGVIT 모두 능가 (0.104 vs. 0.153(VVC), 0.199(HEVC))
- 인간 평가(Elo score)에서 VVC에 필적하거나 우위

#### 비디오 이해 (액션 인식)

| 토크나이저 | SSv2 (출력 타겟) | K400 (입력) | K600 (입력) |
|---|---|---|---|
| MAGVIT | 67.22 | 72.29 | 74.65 |
| **MAGVIT-v2** | **67.38** | **75.34** | **77.93** |
| Raw pixel | 64.83 | 76.13 | 78.92 |

MAGVIT-v2의 디토크나이즈된 픽셀을 입력으로 사용할 때, 원본 픽셀 대비 성능 차이가 매우 미미함 (K600: 77.93 vs. 78.92).

### 2.5 한계

1. **CPU 효율성 미검증**: TPU에서 유망한 결과를 보였으나, 표준 코덱처럼 CPU에서 효율적으로 실행되는지 추가 연구 필요
2. **Text-to-Image/Video 미포함**: 논문의 범위가 클래스 조건부/프레임 예측에 한정되며, 텍스트 조건 생성은 다루지 않음
3. **공정 비교의 어려움**: 텍스트 기반 생성 모델들은 서로 다른 데이터셋·학습 조건을 사용하여 과학적 비교가 어려움
4. **LFQ 변형 탐색 부족**: 이진 독립 차원이라는 가장 단순한 형태만 사용; 다변량 코드북 등 다른 LFQ 변형은 추가 연구 필요
5. **PSNR/MS-SSIM**: 비디오 압축에서 지각 품질(LPIPS)은 우수하나, 전통적 왜곡 지표(PSNR, MS-SSIM)에서는 VVC에 열위

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 LFQ를 통한 어휘 확장과 일반화

LFQ의 가장 중요한 기여는 **어휘 크기 확장이 생성 품질과 일관되게 양의 상관관계를 보인다**는 것이다. 기존 VQ에서는 어휘를 키우면 개별 토큰의 표현력이 과도해져 언어 모델이 그 분포를 학습하기 어려웠다. LFQ는 개별 코드 차원의 표현 능력을 이진값으로 제한하면서도, 차원 수를 늘려 전체 어휘 크기를 확장한다. 이 설계는:

- **과적합 방지**: 개별 토큰의 표현 용량을 제한하여 언어 모델이 토큰 분포를 더 잘 일반화
- **코드북 활용률 향상**: 엔트로피 페널티(Eq. 5)가 모든 코드를 균등하게 사용하도록 유도하여 "코드북 붕괴(codebook collapse)" 문제 해결
- **스케일링 가능성**: 자연어 어휘 크기(200K+)에 근접한 시각적 어휘( $2^{18} \approx 262$ K) 사용이 가능해져, 진정한 멀티모달 LLM으로의 통합 기반 마련

### 3.2 공유 어휘를 통한 이미지-비디오 일반화

인과적 3D CNN 설계로 이미지와 비디오가 **동일한 토큰 어휘**를 공유할 수 있다. 이는:
- 이미지에서 학습한 표현이 비디오에 전이(transfer) 가능
- 비디오 토크나이저의 이미지 사전학습 후 인플레이션(inflation) 기법 적용 가능
- 다양한 해상도와 시간 길이에 대한 일반화 향상 (CNN 기반이므로 위치 임베딩에 의한 해상도 제약 없음)

### 3.3 토큰의 다목적 활용과 일반화

MAGVIT-v2 토큰의 일반화 성능은 세 가지 이질적 태스크에서 동시에 검증되었다:
1. **시각적 생성** (이미지/비디오)
2. **비디오 압축** (VVC 수준)
3. **비디오 이해/액션 인식** (BEVT 방식의 사전학습 타겟 및 ViViT 입력)

이는 MAGVIT-v2가 학습한 이산 표현이 **특정 태스크에 과적합되지 않고, 시각적 세계의 핵심 구조를 포착하는 범용 표현**임을 시사한다. 특히 토큰을 입력으로 사용한 액션 인식 실험에서 원본 픽셀 대비 성능 차이가 매우 작다는 점(K400: 75.34 vs. 76.13)은 토크나이징 과정에서의 정보 손실이 미미함을 보여준다.

### 3.4 이산 표현의 강건성 이점

논문에서 인용된 Mao et al. (2021)의 연구에 따르면, 이산 토큰을 모델 입력으로 사용하면 **강건성(robustness)과 일반화(generalization)**가 향상된다. 이산 양자화가 일종의 **정보 병목(information bottleneck)**으로 작용하여 노이즈나 불필요한 변동을 필터링하기 때문으로 해석된다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구 패러다임에 미치는 영향

1. **토크나이저 중심 시각 생성 연구 촉진**: 이 논문은 "모델 아키텍처(확산 vs. 언어 모델)" 논쟁에서 **토크나이저라는 제3의 요소**가 결정적임을 보여, 시각적 토크나이저 연구의 중요성을 재조명

2. **통합 멀티모달 LLM 가능성**: 시각과 언어를 동일한 토큰 공간에서 다룰 수 있어, 이해·생성·추론을 아우르는 진정한 멀티모달 LLM의 기반 기술로 활용 가능

3. **비디오 압축 패러다임 전환**: 학습 기반 토크나이저가 전통 코덱(HEVC/VVC)에 필적하는 압축 품질을 달성함으로써, 생성 모델과 코덱의 경계를 허무는 새로운 연구 방향 제시

4. **자기지도 학습 사전학습 타겟**: BEiT/BEVT 계열 연구에서 더 강력한 사전학습 타겟으로 MAGVIT-v2 토큰 활용 가능

### 4.2 향후 연구 시 고려할 점

1. **텍스트 조건부 생성으로의 확장**: 현재는 클래스 조건부/프레임 예측만 검증; text-to-image/video에서의 성능 검증이 중요한 후속 과제

2. **LFQ 변형 탐색**: 이진 독립 차원 외에 다변량 코드북, 비균등 차원 분할 등 다양한 LFQ 변형 연구 필요

3. **스케일링 법칙**: 토크나이저 어휘 크기, 모델 크기, 데이터 크기 간의 스케일링 법칙(scaling law)에 대한 체계적 연구 필요

4. **하드웨어 효율성**: GPU/CPU에서의 실시간 인코딩/디코딩 효율화 연구 (현재 TPU 기반 결과만 보고)

5. **장기 비디오 일관성**: 긴 비디오에서의 시간적 일관성 및 품질 유지에 대한 추가 검증 필요

6. **평가 지표 개선**: FID/FVD의 한계를 넘어 인간 평가와 더 잘 상관하는 생성 품질 평가 방법론 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연도 | 모델/논문 | 유형 | 핵심 기여 | MAGVIT-v2와의 비교 |
|---|---|---|---|---|
| **2020** | DDPM (Ho et al., 2020) | Diffusion | 고품질 이미지 생성의 확산 모델 기반 확립 | MAGVIT-v2는 이산 토큰 기반으로 확산 모델의 연속 잠재 변수 패러다임과 근본적으로 차별화 |
| **2021** | VQGAN (Esser et al., 2021) | VQ-VAE + GAN | 적대적/지각 손실로 이미지 토크나이저 품질 향상 | MAGVIT-v2가 LFQ로 대체하여 어휘 확장 문제 해결; VQGAN의 코드북 크기 제한(1K-8K) 극복 |
| **2021** | ADM (Dhariwal & Nichol, 2021) | Diffusion | "Diffusion beats GANs" 입증 | MAGVIT-v2가 "LM beats Diffusion"을 입증; ImageNet $512\times512$에서 FID 1.91 vs. 3.85 |
| **2022** | MaskGIT (Chang et al., 2022) | MLM + VQ | 비자기회귀 마스크 언어 모델로 이미지 생성 | MAGVIT-v2가 동일 MLM 프레임워크에서 토크나이저만 교체하여 대폭 개선 (FID 6.18→3.65, $256\times256$) |
| **2022** | LDM/Stable Diffusion (Rombach et al., 2022) | Diff. + VAE | 잠재 공간 확산으로 효율적 고해상도 생성 | LDM의 VAE도 토크나이저에서 파생; MAGVIT-v2는 이산 잠재 포맷의 이점(LLM 호환성, 압축, 이해)을 강조 |
| **2022** | DiT (Peebles & Xie, 2022) | Diff. + Transformer | U-Net을 Transformer로 대체 | 확산과 LM의 경계 모호화; 그러나 잠재 포맷(연속 vs. 이산)이 핵심 차이로 남음. MAGVIT-v2가 DiT-XL/2 대비 우수 (FID 1.78 vs. 2.27, $256\times256$) |
| **2022** | C-ViViT (Villegas et al., 2022) | Transformer 토크나이저 | 인과적 시간 트랜스포머 비디오 토크나이저 | MAGVIT-v2의 인과적 3D CNN이 C-ViViT 대비 우수 (UCF-101 FVD: 96.33 vs. 437.54) |
| **2023** | MAGVIT (Yu et al., 2023a) | MLM + VQ | 3D CNN 비디오 토크나이저, 비디오 생성 SOTA | MAGVIT-v2의 직접적 선행 연구; LFQ + 아키텍처 개선으로 모든 평가에서 대폭 향상 |
| **2023** | VDM++ (Kingma & Gao, 2023) | Diffusion | 가중 ELBO 확산 목적 함수 | 대규모(2B) 순수 확산 모델; MAGVIT-v2가 307M으로 더 적은 파라미터·스텝에서 우위 |
| **2023** | simple diffusion (Hoogeboom et al., 2023) | Diffusion | 고해상도 종단간 확산 | 2B 파라미터 + 512 스텝; MAGVIT-v2가 $512\times512$에서 FID 1.91 vs. 3.02로 대폭 우위 |
| **2023** | MDT (Gao et al., 2023) | Diff. + VAE + Transformer | 마스크 확산 트랜스포머 | $256\times256$에서 MAGVIT-v2(1.78)와 MDT(1.79) 거의 동등; 단 MAGVIT-v2가 절반 크기·1/4 스텝 |
| **2023** | Binary Latent Diffusion (Wang et al., 2023) | Diff. + Binary | 베르누이 분포 확산 | 이진 잠재 사용은 유사하나, 확산 프레임워크 내; MAGVIT-v2는 LM 프레임워크에서 이진 LFQ 활용 |
| **2023** | MUSE (Chang et al., 2023) | MLM | Text-to-image 마스크 생성 트랜스포머 | MAGVIT-v2의 토크나이저를 MUSE 류 모델에 적용하면 텍스트 기반 생성에서도 성능 향상 기대 |

### 핵심 트렌드 분석

1. **확산 모델과 언어 모델의 수렴**: DiT, MDT 등이 트랜스포머 아키텍처를 확산에 도입하면서 두 패러다임의 경계가 모호해지고 있다. MAGVIT-v2는 남은 핵심 차이점(연속 vs. 이산 잠재)에서 이산의 우위를 입증.

2. **토크나이저의 중요성 재인식**: 2022-2023년에 걸쳐 토크나이저 품질이 생성 성능의 병목임이 점차 인식되고 있으며, MAGVIT-v2가 이 관점을 결정적으로 확립.

3. **효율성-품질 트레이드오프**: 최신 확산 모델(VDM++, simple diffusion)은 2B 파라미터와 수백 스텝이 필요한 반면, MAGVIT-v2 기반 MLM은 307M 파라미터와 64 스텝으로 우수한 결과를 달성하여, 실용적 배포 관점에서 큰 이점.

---

## 참고 자료 및 출처

1. **Yu, L., Lezama, J., Gundavarapu, N.B., et al.** "Language Model Beats Diffusion — Tokenizer is Key to Visual Generation." *ICLR 2024*. arXiv:2310.05737v3.
2. **Van Den Oord, A., Vinyals, O., et al.** "Neural Discrete Representation Learning." *NeurIPS 2017*.
3. **Esser, P., Rombach, R., Ommer, B.** "Taming Transformers for High-Resolution Image Synthesis." *CVPR 2021*.
4. **Chang, H., Zhang, H., Jiang, L., et al.** "MaskGIT: Masked Generative Image Transformer." *CVPR 2022*.
5. **Rombach, R., Blattmann, A., Lorenz, D., et al.** "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.
6. **Peebles, W. and Xie, S.** "Scalable Diffusion Models with Transformers." arXiv:2212.09748, 2022.
7. **Dhariwal, P. and Nichol, A.** "Diffusion Models Beat GANs on Image Synthesis." *NeurIPS 2021*.
8. **Yu, L., Cheng, Y., Sohn, K., et al.** "MAGVIT: Masked Generative Video Transformer." *CVPR 2023*.
9. **Kingma, D.P. and Gao, R.** "Understanding the Diffusion Objective as a Weighted Integral of ELBOs." arXiv:2303.00848, 2023.
10. **Hoogeboom, E., Heek, J., Salimans, T.** "simple diffusion: End-to-end Diffusion for High Resolution Images." *ICML 2023*.
11. **Gao, S., Zhou, P., Cheng, M.-M., Yan, S.** "Masked Diffusion Transformer is a Strong Image Synthesizer." arXiv:2303.14389, 2023.
12. **Ho, J., Jain, A., Abbeel, P.** "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*.
13. **Villegas, R., Babaeizadeh, M., et al.** "Phenaki: Variable Length Video Generation from Open Domain Textual Description." arXiv:2210.02399, 2022.
14. **Mao, C., Jiang, L., Dehghani, M., et al.** "Discrete Representations Strengthen Vision Transformer Robustness." *ICLR 2021*.
15. **Bao, H., Dong, L., Piao, S., Wei, F.** "BEiT: BERT Pre-training of Image Transformers." *ICLR 2021*.
16. **Wang, R., Chen, D., Wu, Z., et al.** "BEVT: BERT Pretraining of Video Transformers." *CVPR 2022*.
17. **Chang, H., Zhang, H., Barber, J., et al.** "Muse: Text-to-Image Generation via Masked Generative Transformers." *ICML 2023*.
18. **Wang, Z., Wang, J., Liu, Z., Qiu, Q.** "Binary Latent Diffusion." *CVPR 2023*.
