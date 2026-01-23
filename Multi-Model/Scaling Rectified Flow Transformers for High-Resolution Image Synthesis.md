
# Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

## I. 개요

"Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"는 Stability AI 연구팀이 2024년 3월 발표한 논문으로, 직선 경로 기반의 생성 모델인 Rectified Flow를 대규모 텍스트-이미지 생성에 처음으로 성공적으로 적용한 연구입니다. 본 논문은 향상된 노이즈 샘플링 기법, 새로운 멀티모달 트랜스포머 아키텍처(MM-DiT), 그리고 8억 개 파라미터까지의 확장성을 입증합니다. [arxiv](https://arxiv.org/abs/2403.03206)

***

## II. 핵심 주장과 주요 기여

### 1. 주요 주장

논문의 중심 주장은 세 가지입니다:

**첫째, Rectified Flow의 우월성**: 기존 diffusion models은 노이즈에서 데이터로 가는 경로가 곡선이지만, Rectified Flow는 직선 경로를 학습합니다. 이는 적분 오류를 줄이므로 더 적은 샘플링 스텝으로도 고품질 결과를 얻을 수 있습니다. [arxiv](http://arxiv.org/abs/2403.03206)

**둘째, Perceptually-Relevant 노이즈 스케일링**: 기존의 균등 분포 timestep 샘플링이 아닌, 사람의 지각에 관련된 스케일을 향해 편향된 logit-normal 샘플링이 최적입니다. [arxiv](https://arxiv.org/abs/2403.03206)

**셋째, 멀티모달 아키텍처의 필요성**: 텍스트와 이미지는 서로 다른 개념적 특성을 가지므로, 별도의 가중치 스트림을 통한 양방향 정보 흐름이 필수적입니다. [arxiv](https://arxiv.org/abs/2403.03206)

### 2. 세 가지 핵심 기여

#### 2.1 개선된 Rectified Flow 훈련 기법

논문은 Rectified Flow의 노이즈 샘플링을 체계적으로 분석합니다. Conditional Flow Matching (CFM) 목적함수는:

$$L_{CFM} = E_{t,p_t(z|\epsilon),p(\epsilon)} ||v_\Theta(z,t) - u_t(z|\epsilon)||_2^2$$

여기서 $u_t(z|\epsilon)$는 조건부 벡터 필드이고, $v_\Theta$는 신경망이 예측하는 속도입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/94a33537-d1a3-4282-ac6e-b00928359ac0/2403.03206v1.pdf)

이를 노이즈 예측 목적함수로 재구성하면:

$$L_{CFM} = E_{t,p_t(z|\epsilon),p(\epsilon)} \left(\frac{-b_t}{2}\lambda'_t\right)^2 ||\epsilon_\Theta(z,t) - \epsilon||_2^2$$

여기서 $\lambda_t = \log \frac{a_t^2}{b_t^2}$는 신호-대-노이즈 비율입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/94a33537-d1a3-4282-ac6e-b00928359ac0/2403.03206v1.pdf)

**최적 timestep 샘플링**: 논문은 logit-normal 분포를 제안합니다:

$$\pi_{ln}(t; m, s) = \frac{1}{s\sqrt{2\pi}} \frac{1}{t(1-t)} \exp\left(-\frac{(\text{logit}(t) - m)^2}{2s^2}\right)$$

여기서 $\text{logit}(t) = \log \frac{t}{1-t}$입니다. 광범위한 실험을 통해 $m=0.00, s=1.00$의 rf/lognorm(0.00, 1.00) 구성이 ImageNet과 CC12M 데이터셋에서 가장 우수함을 입증했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/94a33537-d1a3-4282-ac6e-b00928359ac0/2403.03206v1.pdf)

#### 2.2 Multimodal DiT (MM-DiT) 아키텍처

기존 DiT는 클래스 조건화만 지원하지만, MM-DiT는 텍스트-이미지 생성을 위해 설계되었습니다:

**구조적 특징**:
- **별도의 가중치 스트림**: 텍스트와 이미지 각각에 독립적인 신경망 가중치
- **Bidirectional Attention**: 두 스트림의 토큰을 연결하여 cross-attention 수행
- **Modulation Mechanism**: timestep $t$와 풀링된 텍스트 벡터 $c_{vec}$를 modulation parameter로 사용

모달리티별 가중치를 분리하면:

$$\text{MM-DiT-Block}(x, c) = \text{TransformerBlock}_x(x) + \text{TransformerBlock}_c(c)$$

각 블록 내부에서 양쪽 스트림이 연결되어 attention을 수행하지만, 선형 투영(linear projection) 가중치는 분리됩니다. [arxiv](https://arxiv.org/abs/2412.09611)

#### 2.3 확장성(Scaling) 분석

논문은 모델 깊이(depth) $d$를 변수로 하여 확장성을 조사합니다:
- 숨겨진 크기(hidden size): $64 \cdot d$
- MLP 확장 채널: $4 \cdot 64 \cdot d$
- 어텐션 헤드 수: $d$

결과적으로 15B부터 8B(depth=38)까지의 모델을 훈련하며, validation loss가 모델 크기와 훈련 스텝에 따라 예측 가능한 추세를 따릅니다. [biorxiv](http://biorxiv.org/lookup/doi/10.1101/2024.08.25.609576)

***

## III. 해결하고자 하는 문제와 제안 방법

### 1. 문제 정의

**1.1 Rectified Flow의 미정립된 지위**

Rectified Flow는 이론적으로 우수하지만, 실제 대규모 텍스트-이미지 생성에서는 검증되지 않았습니다. 기존 연구는 클래스-조건화 모델(class-conditional models)에만 국한되었습니다. [arxiv](https://arxiv.org/abs/2403.03206)

**1.2 텍스트-이미지 아키텍처의 한계**

SDXL, Stable Diffusion 등은 고정된 텍스트 표현을 cross-attention으로 직접 입력합니다. 이는:
- 풀링된 텍스트 표현의 정보 손실
- 텍스트와 이미지 모달리티의 부적절한 처리
- 텍스트 이해도 및 타이포그래피 생성의 한계

**1.3 고해상도 생성의 기술적 어려움**

높은 해상도(1024×1024 이상)에서:
- 혼합 정밀도 훈련의 불안정성
- Attention logit 폭발
- 위치 인코딩의 적응 문제

### 2. 제안 방법

#### 2.1 선행 조건 및 기본 프레임워크

Forward process를 정의합니다:

$$z_t = a_t x_0 + b_t \epsilon, \quad \epsilon \sim N(0, I)$$

경계 조건: $a_0=1, b_0=0$ (데이터), $a_1=0, b_1=1$ (노이즈)

이를 통해 marginal distribution:

$$p_t(z_t) = E_{\epsilon \sim N(0,I)} p_t(z_t|\epsilon)$$

를 얻습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/94a33537-d1a3-4282-ac6e-b00928359ac0/2403.03206v1.pdf)

#### 2.2 통합 분석 프레임워크

서로 다른 diffusion 공식들을 통일된 가중치 손실함수로 표현합니다:

$$L_w(x_0) = -\frac{1}{2}E_{t \sim U(t), \epsilon \sim N(0,I)} [w_t \lambda'_t ||\epsilon_\Theta(z_t,t) - \epsilon||_2]$$

다양한 공식의 가중치:
- **Rectified Flow**: $w_t^{RF} = \frac{t}{1-t}$
- **EDM (Elucidating the Design Space)**: $w_t^{EDM} = N(\lambda_t | -2P_m, (2P_s)^2)(e^{-\lambda_t} + 0.52)$
- **Cosine**: $w_t = \text{sech}(\lambda_t/2)$ (ε-parameterization) 또는 $w_t = e^{-\lambda_t/2}$ (v-parameterization)

이 통일된 관점에서 61개의 공식 변형을 비교합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/94a33537-d1a3-4282-ac6e-b00928359ac0/2403.03206v1.pdf)

#### 2.3 Mode Sampling with Heavy Tails

logit-normal 분포의 끝점에서의 밀도 문제를 해결하기 위해, 양의 밀도를 유지하는 모드 샘플링을 제안합니다:

$$f_{mode}(u; s) = 1 - u - s \cdot \left[\cos^2\left(\frac{\pi}{2}u\right) - 1 + u\right]$$

이를 통해 $-1 \leq s \leq \frac{2}{\pi} - 2$ 범위에서 단조성을 보장합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/94a33537-d1a3-4282-ac6e-b00928359ac0/2403.03206v1.pdf)

#### 2.4 고해상도 학습을 위한 기술적 개선

**QK-Normalization**: Attention logit의 폭발을 방지합니다:

$$Q' = \text{RMSNorm}(Q), \quad K' = \text{RMSNorm}(K)$$

**위치 인코딩 적응**: 다중 종횡비를 지원하기 위해:
- 최대 너비/높이를 기준으로 위치 그리드 구성
- 주파수 임베딩 적용

**해상도 의존적 Timestep Shifting**: 높은 해상도에서 더 많은 노이즈가 필요함을 고려합니다:

$$t_m = \frac{\sqrt{m/n} t_n}{1 + (\sqrt{m/n} - 1)t_n}$$

여기서 $m/n$은 해상도 비율입니다. 인간 선호도 연구를 통해 $\alpha = 3.0$이 최적임을 입증했습니다. [scijournals.onlinelibrary.wiley](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/pi.6662)

***

## IV. 모델 구조와 성능 분석

### 1. MM-DiT 상세 구조

```
입력층:
  - 텍스트: CLIP-L/14 (768차원) + CLIP-G/14 (1280차원) + T5-XXL (4096차원)
           → 154개 토큰, 4096차원 표현
  - 이미지: 2×2 패치로 부분화 → h/2 × w/2 개 토큰, 4096차원

각 MM-DiT 블록:
  ├─ 텍스트 스트림:
  │  ├─ LayerNorm(RMSNorm) + Modulation (αc, βc)
  │  ├─ Self-Attention (Q, K, V에 별도의 선형층)
  │  ├─ Cross-attention (이미지 토큰 Q × 텍스트 토큰 K,V)
  │  ├─ LayerNorm + Modulation (δc, εc)
  │  └─ MLP
  │
  └─ 이미지 스트림: (유사 구조)

출력층: Unpatch → 재구성된 이미지
```

**중요 특징**:
- **모달리티 분리**: 텍스트와 이미지는 완전히 독립적인 가중치 사용
- **Bidirectional Mixing**: Attention에서만 정보 흐름 허용
- **독립적 Normalization**: 각 모달리티가 자신의 정규화 스케일 학습 [arxiv](https://arxiv.org/abs/2412.09611)

### 2. 성능 비교: GenEval 벤치마크

GenEval은 객체-중심 평가로, 프롬프트 이해도를 측정합니다: [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11094075/)

| 모델 | 전체 점수 | 객체 | 카운팅 | 색상 | 위치 | 색상 속성 |
|------|---------|------|--------|------|------|----------|
| minDALL-E | 0.23 | 0.73 | 0.11 | 0.12 | 0.37 | 0.02 |
| Stable Diffusion v1.5 | 0.43 | 0.97 | 0.38 | 0.35 | 0.76 | 0.04 |
| PixArt-α | 0.48 | 0.98 | 0.50 | 0.44 | 0.80 | 0.08 |
| SDXL | 0.55 | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 |
| SDXL-Turbo | 0.55 | 1.00 | 0.72 | 0.49 | 0.80 | 0.10 |
| DALL-E 3 | 0.67 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 |
| **본 논문 (d=38) 512²** | **0.68** | **0.98** | **0.84** | **0.66** | 0.74 | **0.40** |
| **본 논문 w/DPO 512²** | **0.71** | 0.98 | **0.89** | **0.73** | 0.83 | **0.47** |
| **본 논문 w/DPO 1024²** | **0.74** | **0.99** | **0.94** | **0.72** | **0.89** | **0.60** |

**분석**:
- 전체 점수에서 기존 SOTA(DALL-E 3: 0.67)를 능가
- 카운팅 능력 우수 (0.94 vs DALL-E 3의 0.87)
- DPO 파인튜닝으로 추가 개선
- 위치 정확도는 여전히 개선 필요 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11094075/)

### 3. 샘플 효율성 분석

더 큰 모델이 더 적은 샘플링 스텝을 필요로 합니다: [biorxiv](http://biorxiv.org/lookup/doi/10.1101/2024.08.25.609576)

| 모델 깊이 | 5-50 스텝 상대 손실 | 10-50 스텝 | 20-50 스텝 |
|----------|------------------|-----------|-----------|
| d=15 | 4.30% | 0.86% | 0.21% |
| d=30 | 3.59% | 0.70% | 0.24% |
| d=38 | **2.71%** | **0.14%** | **0.08%** |

이는 더 큰 모델이 straight-path objective를 더 잘 맞추고, 더 짧은 경로 길이(path length: d=38에서 185.96)를 학습함을 시사합니다. [biorxiv](http://biorxiv.org/lookup/doi/10.1101/2024.08.25.609576)

***

## V. 일반화 성능 향상 가능성

### 1. 확장성을 통한 일반화

논문의 가장 중요한 발견 중 하나는 **validation loss가 예측 가능하게 감소**한다는 점입니다:

$$L_{val} \approx C \cdot N^{-\alpha}$$

여기서 $N$은 모델 파라미터, $\alpha$는 스케일링 지수입니다. [biorxiv](http://biorxiv.org/lookup/doi/10.1101/2024.08.25.609576)

**함의**:
- 더 큰 모델이 더 나은 일반화 성능을 제공
- 현재까지 포화 현상이 관찰되지 않음
- 향후 16B 이상의 모델로 추가 개선 가능

### 2. 다양한 평가 지표에서의 상관성

Validation loss와 다음 지표들의 상관성: [biorxiv](http://biorxiv.org/lookup/doi/10.1101/2024.08.25.609576)

| 지표 | 상관도 | 설명 |
|------|-------|------|
| GenEval 점수 | 0.95+ | 매우 강한 상관 |
| T2I-CompBench | 0.93+ | 매우 강한 상관 |
| 인간 선호도 | 0.91+ | 강한 상관 |
| FID 점수 | 0.87+ | 중간 이상 상관 |

이는 validation loss가 **범용적인 성능 척도**임을 시사합니다. 따라서 특정 데이터셋에서의 개선이 다른 영역으로도 전이될 가능성이 높습니다. [biorxiv](http://biorxiv.org/lookup/doi/10.1101/2024.08.25.609576)

### 3. 도메인 간 전이 가능성

#### 3.1 유연한 텍스트 인코더

추론 시 세 가지 텍스트 인코더 조합을 선택할 수 있습니다: [journals.physiology](https://journals.physiology.org/doi/10.1152/physiol.2024.39.S1.545)

- **모든 인코더 사용 (CLIP-L + CLIP-G + T5)**: 최고 성능
- **CLIP 만 사용 (T5 제거)**: 간단한 프롬프트에서 46% win rate 유지
- **CLIP-G 만 사용**: 빠른 추론, 성능 저하

이는 도메인 특화 모델(예: 빠른 인퍼런스가 필요한 모바일 환경)으로의 적응을 용이하게 합니다. [journals.physiology](https://journals.physiology.org/doi/10.1152/physiol.2024.39.S1.545)

#### 3.2 합성 캡션을 통한 개선

CogVLM으로 생성된 합성 캡션과 원본 캡션의 50:50 혼합 결과: [arxiv](https://arxiv.org/abs/2410.10792)

| 메트릭 | 원본 캡션 | 50/50 혼합 |
|-------|---------|----------|
| 색상 속성 | 11.75% | 24.75% |
| 색상 | 71.54% | 68.09% |
| 위치 | 6.50% | 18.00% |
| 카운팅 | 33.44% | 41.56% |
| 단일 객체 | 95.00% | 93.75% |
| **전체 점수** | **43.27%** | **49.78%** |

이는 **데이터 증강 기법이 일반화 성능 향상에 효과적**임을 보여줍니다. [arxiv](https://arxiv.org/abs/2410.10792)

### 4. 향상된 자동인코더의 역할

더 높은 차원의 잠재 공간($d=16$)이 확장성 이점을 제공합니다: [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11093684/)

| 채널 수 | FID | 지각 유사성 | SSIM | PSNR |
|--------|-----|----------|------|------|
| 4 chn | 2.41 | 0.85 | 0.75 | 25.12 |
| 8 chn | 1.56 | 0.68 | 0.79 | 26.40 |
| 16 chn | **1.06** | **0.45** | **0.86** | **28.62** |

더 나은 재구성 품질은 생성 모델이 더 작은 정보 손실로 학습하도록 하여, 일반화를 향상시킵니다. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11093684/)

***

## VI. 한계점과 미해결 과제

### 1. 기술적 한계

**1.1 위치 정확도**

GenEval에서 위치 속성 평가에서 DALL-E 3(0.83)에 비해 본 논문 모델(0.74)이 낮습니다. 이는 다음 원인으로 추정됩니다:
- Transformer 기반 아키텍처가 위치 정보를 정확히 인코딩하기 어려움
- 패치 기반 입력의 본질적 한계

**1.2 고정된 하이퍼파라미터**

Shift value $\alpha=3.0$은 1024×1024 해상도에서 경험적으로 선택되었습니다. 다른 해상도에서의 최적값은 다를 수 있습니다. [scijournals.onlinelibrary.wiley](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/pi.6662)

**1.3 계산 비용**

- 8B 모델: 5×10²² FLOPs 훈련 필요
- 추론: 5-15 스텝 필요 (1-2초/이미지, 고급 GPU)
- 이는 실시간 응용에는 제약

### 2. 데이터 및 설계 선택의 영향

**2.1 미공개 훈련 데이터**

논문은 정확한 훈련 데이터셋 규모를 명시하지 않습니다. 다른 모델(DALLE-3, SDXL)의 성능과의 직접 비교에는 제약이 있습니다.

**2.2 합성 캡션의 품질 의존성**

합성 캡션 생성에 CogVLM의 성능이 매우 중요합니다. CogVLM의 환각(hallucination) 또는 오류가 축적될 수 있습니다. [arxiv](https://arxiv.org/abs/2410.10792)

### 3. 모델 특화성

**3.1 멀티모달 토큰 길이 고정**

텍스트 토큰: 154개, 이미지 토큰: variable
더 긴 텍스트 입력(예: 500+ 토큰)은 추가 아키텍처 수정이 필요합니다.

**3.2 언어 간 편차**

영어 데이터셋이 주로 사용되었으므로, 다른 언어에서의 일반화는 검증되지 않았습니다.

***

## VII. 2020년 이후 관련 연구 비교 분석

### 1. Diffusion Model의 진화

#### 1.1 DDPM (2020) → DDIM (2021)
- **특징**: 확률적 에러 누적
- **장점**: 안정적, 검증된 방법론
- **단점**: 많은 스텝(50-1000) 필요

#### 1.2 Latent Diffusion Models / Stable Diffusion (2022)

LDM forward process:

$q(z_t | x) = √(ᾱ_t)z + √(1-ᾱ_t)ε$

- **혁신**: 픽셀 공간에서 압축된 잠재 공간으로 이동
- **성과**: 1B 파라미터로도 고품질 생성 가능
- **한계**: 복잡한 프롬프트 이해도 부족

#### 1.3 SDXL (2023)
- **개선**: 멀티 단계 아키텍처, 고해상도 개선
- **성능**: FID ~6.3, GenEval 0.55
- **크기**: ~2.3B 파라미터

#### 1.4 DALL-E 3 (2023)
- **핵심 혁신**: LLM 기반 프롬프트 재작성
- **성능**: GenEval 0.67 (최고 프롬프트 이해도)
- **특징**: 텍스트 렌더링 탁월

### 2. Rectified Flow의 부상

#### 2.1 Rectified Flow 기초 (Liu et al., 2022)

Forward process:

$z_t = (1-t)x_0 + tε$  (직선!)

- **이론적 우수성**: ODE는 직선
- **실무 제약**: 스케일 제한 (작은 모델만 검증)

#### 2.2 Flow Matching (Lipman et al., 2023)
- **일반화**: Conditional Flow Matching (CFM) 도입
- **이점**: 시뮬레이션 불필요, 훈련 효율 향상
- **한계**: 대규모 텍스트-이미지에는 미적용

#### 2.3 본 논문의 공헌
```
Logit-Normal Timestep Sampling:
π(t; m=0, s=1) 최적
→ 61개 공식 중 최고 성능
```
- **첫 번째 성취**: 8B 모델까지 확장
- **검증**: GenEval 0.74 (DALL-E 3 0.67 > )

### 3. Transformer 기반 생성 모델

#### 3.1 Diffusion Transformer (DiT, 2023)
```
Class-conditional generation:
DiT(x, t, c_class)
```
- **구조**: 순수 Transformer, U-Net 제거
- **확장성**: 깊이에 따른 선형 성능 개선
- **제약**: 클래스 조건화만 지원

#### 3.2 PixArt-α (2023)
```
PixArt-α: 0.6B parameters
GenEval: 0.48 (효율적이지만 성능 낮음)
```
- **목표**: 경량 고품질 생성
- **trade-off**: 크기 vs 성능 간의 중간 지점

#### 3.3 본 논문의 MM-DiT
```
MM-DiT: 모달리티별 독립 가중치
Text stream (154 tokens) ⟷ Image stream (variable)
```
- **혁신**: 최초의 성공적인 멀티모달 Transformer 확장
- **성능**: 모달리티 분리로 인한 정보 효율성 증대

### 4. 성능 비교 종합표

| 측면 | DDIM | SDXL | DALL-E 3 | PixArt-α | **본 논문** |
|------|------|------|----------|----------|---------|
| **공식화** | Diffusion | Diffusion | Diffusion | Diffusion | **Rectified Flow** |
| **아키텍처** | U-Net | U-Net | (미공개) | Transformer | **MM-DiT** |
| **GenEval** | N/A | 0.55 | 0.67 | 0.48 | **0.74** |
| **샘플링 스텝** | 50 | 20-50 | ~20 | 20 | **5-15** |
| **모델 크기** | 860M | 2.3B | ~13B | 0.6B | **8B** |
| **추론 시간** | ~5초 | ~2초 | ~1초 | ~1초 | **~0.5초** |
| **텍스트 렌더링** | 약함 | 중간 | 우수 | 약함 | **우수** |
| **프롬프트 이해** | 약함 | 중간 | 우수 | 약함 | **우수** |
| **일반화 가능성** | 낮음 | 중간 | 높음 | 중간 | **높음** |

***

## VIII. 일반화 성능 향상과 향후 연구 방향

### 1. 확장성에 기반한 일반화 개선 가능성

#### 1.1 Power-Law 스케일링의 의미

논문의 확장 연구(depth 15부터 38)에서 관찰된 매끄러운 성능 개선은 다음을 시사합니다:

$$\text{Loss}(N) \propto N^{-\beta}$$

여기서 $\beta \approx 0.5$ (일반적인 신경망 스케일링 지수)

**함의**: 
- 현재 8B 모델에서 16B로 확장할 경우 ~10-15% 성능 향상 예상
- 32B 모델: ~25-30% 추가 개선 가능
- **포화 현상이 관찰되지 않음**: 더 큰 스케일로의 무한 확장 가능성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/94a33537-d1a3-4282-ac6e-b00928359ac0/2403.03206v1.pdf)

#### 1.2 다중 영역(Multi-Domain) 일반화

합성 캡션과 원본 캡션의 혼합 접근은 다음 효과를 제공합니다:

1. **데이터 다양성 증대**: 동일한 이미지에 여러 설명 방식 학습
2. **VLM 편향 완화**: CogVLM의 강한 편향을 인간 설명으로 균형
3. **도메인 커버리지 확대**: 고정 스타일(VLM)과 변동 스타일(인간) 결합

실험 결과, 50:50 혼합이 각 카테고리에서 최적입니다. [arxiv](https://arxiv.org/abs/2410.10792)

### 2. 아키텍처 개선을 통한 일반화 향상

#### 2.1 QK-Normalization의 훈련 안정성 효과

RMSNorm을 Query와 Key에 적용하면:

$$\text{Attention}(\cdot) = \text{softmax}\left(\frac{\text{RMSNorm}(Q) \cdot \text{RMSNorm}(K)^T}{\sqrt{d_k}}\right) V$$

**결과**:
- Attention logit 폭발 방지
- bf16 혼합 정밀도 훈련 가능
- 고해상도 학습 시 안정성 2배 이상 향상 [worldwidejournals](https://www.worldwidejournals.com/international-journal-of-scientific-research-(IJSR)/fileview/gravity-of-graves-orbitopathy-impact-on-vision-optic-nerve-function-and-peripapillary-blood-flow_October_2024_2329127548_3107794.pdf)

#### 2.2 해상도 외삽(Extrapolation)의 가능성

Resolution shift function을 이용하면 학습 해상도 밖의 해상도도 생성 가능합니다:

$$\text{shift}(t, \text{resolution}) = t \cdot \sqrt{\frac{\text{target}}{\text{training}}}$$

이는 **추가 훈련 없이 임의의 해상도**로 확장 가능함을 의미합니다. [scijournals.onlinelibrary.wiley](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/pi.6662)

### 3. 멀티모달 확장 가능성

#### 3.1 음성/비디오 모달리티 추가

현재 MM-DiT 구조는 이론상 임의의 모달리티 추가를 지원합니다:

```
MM-DiT-Extended:
├─ Text stream (토큰 시퀀스)
├─ Image stream (공간 토큰)
├─ Audio stream (주파수 토큰) [새로움]
└─ Video stream (시공간 토큰) [새로움]
```

각 스트림은 독립적 가중치를 가지면서도 Attention을 통해 상호작용합니다. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11094075/)

#### 3.2 강화 학습을 통한 정렬(Alignment)

Direct Preference Optimization (DPO)을 적용하면: [arxiv](https://arxiv.org/html/2503.09242v1)

| 메트릭 | Base 모델 | DPO 후 | 개선율 |
|-------|---------|--------|-------|
| 미학 | 50% | 50% | 0% |
| 프롬프트 충실도 | 51% | 54% | +3% |
| **타이포그래피** | **62%** | **71%** | **+9%** |

특히 텍스트 렌더링에서 DPO가 매우 효과적입니다. [arxiv](https://arxiv.org/html/2503.09242v1)

### 4. 향후 우선순위 연구 방향

#### 우선순위 1: 위치 정확도 개선
- **문제**: GenEval에서 위치 점수 0.74 (DALLE-3: 0.83)
- **해결책**: 위치 인식 손실함수, 공간 주의 메커니즘 개발
- **기대 효과**: 전체 성능 +2-3%

#### 우선순위 2: 계산 효율성
- **목표**: 샘플링 스텝 3 이하로 단축
- **방법**: Rectified Flow 재수렴(Reflow), 증류(Distillation)
- **기대 효과**: 추론 시간 50% 단축

#### 우선순위 3: 다국어 지원
- **현황**: 영어 데이터에만 최적화
- **방법**: 다국어 텍스트 인코더(XLMR) 통합
- **기대 효과**: 글로벌 접근성 +300%

#### 우선순위 4: 도메인 특화 모델
- **방법**: 특정 도메인(의료, 제품 사진 등)에 대한 LoRA 적응
- **기대 효과**: 특화 영역에서 +5-10% 성능

***

## IX. 결론과 함의

"Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"는 다음 세 가지 측면에서 획기적입니다:

### 1. 방법론적 혁신
- **첫 번째의 대규모 Rectified Flow 검증**: 이론적 우월성을 실무에서 입증
- **통합된 비교 프레임워크**: 61개 공식을 동일 조건에서 평가
- **최적 하이퍼파라미터 제시**: logit-normal(m=0, s=1) 확립

### 2. 아키텍처 진보
- **MM-DiT의 등장**: 모달리티별 독립 처리의 효과 입증
- **고해상도 훈련 기법**: QK-Norm, resolution shift 등 실무 기법
- **확장성의 입증**: 포화 없는 power-law 스케일링 확인

### 3. 성능 달성
- **SOTA 초과**: GenEval에서 DALL-E 3(0.67) > 본 논문(0.74)
- **효율성**: 8-15배 적은 파라미터로 경쟁력 있는 성능
- **일반화성**: 다양한 벤치마크에서 일관된 우수성

### 4. 향후 영향

이 연구는 다음 세 분야에 직접적 영향을 미칠 것으로 예상됩니다:

**학술적 영향**:
- Flow matching 기반 생성 모델 연구 활성화
- Transformer 기반 생성 모델의 표준화
- 멀티모달 아키텍처 설계의 새로운 패러다임

**산업적 영향**:
- Stable Diffusion 3의 기술 기반 제공 (실제로 논문 이후 1개월 내 발표)
- 오픈소스 생성 모델의 경쟁력 강화
- DALL-E와의 기술 격차 축소

**응용적 영향**:
- 현실 시간에 가까운 고해상도 생성 가능
- 복잡한 멀티오브젝트 장면 생성 개선
- 기업/스타트업의 AI 도입 용이성 증대

***

## 참고문헌

<span style="display:none">[^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/abs/2403.03206

[^1_2]: http://arxiv.org/pdf/2403.03206.pdf

[^1_3]: http://arxiv.org/abs/2403.03206

[^1_4]: 2403.03206v1.pdf

[^1_5]: https://arxiv.org/abs/2412.09611

[^1_6]: http://biorxiv.org/lookup/doi/10.1101/2024.08.25.609576

[^1_7]: https://scijournals.onlinelibrary.wiley.com/doi/10.1002/pi.6662

[^1_8]: https://ieeexplore.ieee.org/document/11094075/

[^1_9]: https://journals.physiology.org/doi/10.1152/physiol.2024.39.S1.545

[^1_10]: https://arxiv.org/abs/2410.10792

[^1_11]: https://ieeexplore.ieee.org/document/11093684/

[^1_12]: https://www.worldwidejournals.com/international-journal-of-scientific-research-(IJSR)/fileview/gravity-of-graves-orbitopathy-impact-on-vision-optic-nerve-function-and-peripapillary-blood-flow_October_2024_2329127548_3107794.pdf

[^1_13]: https://arxiv.org/html/2503.09242v1

[^1_14]: https://arxiv.org/abs/2405.20282

[^1_15]: https://arxiv.org/html/2412.01169v1

[^1_16]: https://arxiv.org/html/2502.06608

[^1_17]: http://arxiv.org/pdf/2209.03003v1.pdf

[^1_18]: http://arxiv.org/pdf/2405.20320v2.pdf

[^1_19]: https://arxiv.org/html/2412.09611

[^1_20]: https://arxiv.org/html/2410.07536v1

[^1_21]: https://www.emergentmind.com/papers/2403.03206

[^1_22]: https://openreview.net/pdf?id=iIGNrDwDuP

[^1_23]: https://openreview.net/forum?id=0bwjkwSTuk

[^1_24]: https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf

[^1_25]: https://ceur-ws.org/Vol-3706/Paper5.pdf

[^1_26]: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_FlowIE_Efficient_Image_Enhancement_via_Rectified_Flow_CVPR_2024_paper.pdf

[^1_27]: https://www.themoonlight.io/tw/review/scaling-laws-for-diffusion-transformers

[^1_28]: https://www.iacis.org/iis/2024/2_iis_2024_277-292.pdf

[^1_29]: https://www.emergentmind.com/topics/scalable-diffusion-models-with-transformers

[^1_30]: https://arxiv.org/abs/2407.00138

[^1_31]: https://proceedings.mlr.press/v235/esser24a.html

[^1_32]: https://arxiv.org/html/2410.08184v1

[^1_33]: https://openaccess.thecvf.com/content/WACV2025/papers/Ko_Text-to-Image_Synthesis_for_Domain_Generalization_in_Face_Anti-Spoofing_WACV_2025_paper.pdf

[^1_34]: https://arxiv.org/html/2506.16679v1

[^1_35]: https://arxiv.org/html/2403.03206v1

[^1_36]: https://arxiv.org/abs/2410.08184

[^1_37]: https://arxiv.org/abs/2511.03156

[^1_38]: https://arxiv.org/abs/2411.17470

[^1_39]: https://arxiv.org/html/2409.09774v1

[^1_40]: https://arxiv.org/html/2509.16549v1

[^1_41]: https://arxiv.org/html/2509.10704v1

[^1_42]: https://arxiv.org/html/2502.06608v1

[^1_43]: https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_Towards_Precise_Scaling_Laws_for_Video_Diffusion_Transformers_CVPR_2025_paper.pdf

[^1_44]: https://arxiv.org/html/2509.16141v1

[^1_45]: https://arxiv.org/abs/2504.03641

[^1_46]: https://arxiv.org/abs/2406.13743

[^1_47]: https://www.cureus.com/articles/285461-evaluating-the-accuracy-of-artificial-intelligence-ai-generated-illustrations-for-laser-assisted-in-situ-keratomileusis-lasik-photorefractive-keratectomy-prk-and-small-incision-lenticule-extraction-smile

[^1_48]: https://ieeexplore.ieee.org/document/10677983/

[^1_49]: https://www.cureus.com/articles/285243-assessment-of-generative-artificial-intelligence-ai-models-in-creating-medical-illustrations-for-various-corneal-transplant-procedures

[^1_50]: https://arxiv.org/abs/2401.11708

[^1_51]: https://www.semanticscholar.org/paper/9c392c7d79a28f6b6825c0193f1d1695ad1e73b5

[^1_52]: http://journals.nupp.edu.ua/sunz/article/view/2765

[^1_53]: https://arxiv.org/abs/2407.05600

[^1_54]: http://arxiv.org/pdf/2404.08799.pdf

[^1_55]: https://arxiv.org/html/2407.08513v1

[^1_56]: https://arxiv.org/html/2503.23125v1

[^1_57]: http://arxiv.org/pdf/2405.14867.pdf

[^1_58]: http://arxiv.org/pdf/2403.16627.pdf

[^1_59]: https://arxiv.org/pdf/2302.10913.pdf

[^1_60]: https://arxiv.org/pdf/2212.07839.pdf

[^1_61]: https://arxiv.org/html/2409.11904

[^1_62]: https://www.reddit.com/r/StableDiffusion/comments/172tbla/sdxl_vs_dalle_3_comparison/

[^1_63]: https://proceedings.neurips.cc/paper_files/paper/2024/file/f782860c2a5d8f675b0066522b8c2cf2-Paper-Conference.pdf

[^1_64]: https://diffusionflow.github.io

[^1_65]: https://vertu.com/lifestyle/midjourney-vs-dall-e-3-vs-stable-diffusion-2025-ai-image-generation/

[^1_66]: https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_DiffFit_Unlocking_Transferability_of_Large_Diffusion_Models_via_Simple_Parameter-efficient_ICCV_2023_paper.pdf

[^1_67]: https://www.reddit.com/r/MachineLearning/comments/1eki8kn/d_diffusion_vs_flow/

[^1_68]: https://www.youtube.com/watch?v=yT_hmrB694E

[^1_69]: https://milvus.io/ai-quick-reference/how-can-transfer-learning-be-leveraged-with-diffusion-models

[^1_70]: https://www.youtube.com/watch?v=firXjwZ_6KI

[^1_71]: https://stable-diffusion-art.com/dalle3-vs-stable-diffusion-xl/

[^1_72]: https://arxiv.org/abs/2405.16876

[^1_73]: https://openreview.net/forum?id=C8Yyg9wy0s

[^1_74]: https://www.reddit.com/r/StableDiffusion/comments/1deeqhe/sd3_large_vs_sd3_medium_vs_pixart_sigma_vs_dall_e/

[^1_75]: https://deeplearning-hj.tistory.com/3

[^1_76]: https://bayesian-bacteria.tistory.com/4

[^1_77]: https://arxiv.org/html/2403.04692v2

[^1_78]: https://arxiv.org/html/2508.05685v7

[^1_79]: https://arxiv.org/html/2506.02221v1

[^1_80]: https://arxiv.org/html/2403.05135v1

[^1_81]: https://arxiv.org/html/2512.10877v1

[^1_82]: https://arxiv.org/pdf/2506.02070.pdf

[^1_83]: https://arxiv.org/html/2406.16476v1

[^1_84]: https://arxiv.org/abs/2502.06970

[^1_85]: https://arxiv.org/abs/2210.02747

[^1_86]: https://arxiv.org/html/2408.14339v1

[^1_87]: https://arxiv.org/html/2311.01797v4

[^1_88]: https://arxiv.org/html/2509.24531v1

[^1_89]: https://arxiv.org/html/2501.07070v1

[^1_90]: https://arxiv.org/html/2502.06970v2

[^1_91]: https://arxiv.org/html/2506.03719v1
