
# Vector Quantized Diffusion Model for Text-to-Image Synthesis

## 1. 핵심 주장 및 주요 기여 (Executive Summary)

**VQ-Diffusion**은 텍스트-이미지 생성 분야에서 기존 자기회귀(Autoregressive, AR) 모델의 근본적인 문제들을 해결하기 위해 제안된 혁신적인 접근법입니다. 이 모델의 핵심 주장은 세 가지입니다: 첫째, **일방향 편향(unidirectional bias) 제거**를 통해 2D 이미지의 구조를 적절히 포착할 수 있으며, 둘째, **누적 오류 회피(error accumulation avoidance)** 전략으로 생성 품질을 향상시킬 수 있고, 셋째, **효율적인 추론 속도** 달성이 가능하다는 것입니다.

주요 기여는 다음과 같습니다:
- **마스크-교체 확산 전략(mask-and-replace diffusion strategy)** 도입으로 이산적 확산 과정을 효과적으로 모델링
- **비자기회귀적 생성** 방식으로 동시에 모든 토큰을 처리하여 전역 문맥 활용
- **15배 빠른 추론 속도** 달성(동일 품질 기준)
- **GAN 기반 방법을 능가**하는 성능 및 **DALL-E, CogView보다 10배 작은 모델**로 경쟁 가능한 결과 달성

## 2. 해결하고자 하는 문제

### 2.1 기존 자기회귀 모델의 한계

기존 텍스트-이미지 생성 AR 모델(DALL-E, CogView, ImageBART 등)은 다음의 심각한 문제를 가지고 있었습니다:

**문제 1: 일방향 편향(Unidirectional Bias)**
- AR 모델은 왼쪽 위에서 오른쪽 아래로 래스터 스캔 순서로 픽셀/토큰을 순차적으로 예측합니다.
- 이는 수평-수직 축 외의 중요한 맥락 정보를 무시합니다.
- 예를 들어, 하단의 객체 정보가 상단 생성에 영향을 미치지 못합니다.
- 결과: **부자연스럽고 구조적으로 일관성 없는 이미지** 생성

**문제 2: 누적 예측 오류(Exposure Bias / Teacher Forcing Mismatch)**
- **학습 단계**: 모델이 정답 토큰을 입력으로 받음(teacher-forcing)
- **추론 단계**: 모델이 자신의 예측 토큰을 입력으로 사용
- 이 불일치로 인해 초기 오류가 다음 단계에 전파되고, 최종적으로 누적됩니다.
- 결과: **해상도가 높을수록 급격한 성능 저하**

**문제 3: 낮은 추론 속도**
- 각 토큰마다 네트워크 전체를 한 번씩 통과해야 함
- 640×480 해상도 이미지의 경우 32×24=768개 토큰이 필요
- 추론 시간이 해상도에 **선형 증가**하여 실용성 부족

### 2.2 이전 GAN 기반 방법의 한계

- **단일 도메인에서만 우수한 성능** (예: CUB-200, Oxford-102 새, 꽃)
- **복잡한 장면 생성 실패** (MSCOCO 같은 다중 객체 데이터셋)
- **합성곱 신경망의 지역성 편향**으로 인한 전역 구조 파악 어려움

## 3. 제안하는 방법론

### 3.1 VQ-VAE 기반 이산 잠재 공간 학습

먼저 **이미지 압축 단계**에서 VQ-VAE를 사용하여 고해상도 이미지를 저해상도 토큰 시퀀스로 변환합니다:

$$z_q = Q(z) = \arg\min_{z_k \in Z} ||z_{ij} - z_k||_2^2$$

여기서 $z_q \in \mathbb{R}^{h \times w \times d}$는 양자화된 잠재 벡터이고, $Z = \{z_k\}_{k=1}^{K}$는 크기 $K$의 코드북입니다.

VQ-VAE의 손실함수는:

$$L_{VQ-VAE} = ||x - G(z_q)||_2^2 + \beta||sg(z_q) - z||_2^2 + ||z_q - sg(z)||_2^2$$

여기서 $sg$는 기울기 정지(stop-gradient) 연산이며, 실제 구현에서는 지수 이동 평균(EMA)으로 대체됩니다.

### 3.2 이산 확산 과정의 정의

**전향 확산 과정**은 마르코프 체인을 통해 정의됩니다:

$$q(x_t|x_{t-1}) = v_{x_t}Q_t v_{x_{t-1}}$$

여기서 $v_x$는 원-핫 벡터이고, $Q_t \in \mathbb{R}^{K \times K}$는 전이 행렬입니다.

마르코프 성질을 이용하면, 중간 단계를 한 번에 건너뛸 수 있습니다:

$$q(x_t|x_0) = v_{x_t}Q_t^t v_{x_0}, \quad \text{with } Q_t^t = Q_t \cdots Q_1$$

**후험 분포**는 다음과 같이 정리 가능합니다:

$$q(x_{t-1}|x_t, x_0) = \frac{q(x_t|x_{t-1}, x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)} = \frac{v_{x_t}Q_t v_{x_{t-1}} v_{x_{t-1}}Q_{t-1}^{t-1}v_{x_0}}{v_{x_t}Q_t^t v_{x_0}}$$

### 3.3 마스크-교체 확산 전략 (핵심 혁신)

기존 균등 노이즈 확산의 문제점을 해결하기 위해, 논문은 **마스크 언어 모델링(MLM)**에서 영감을 받아 새로운 전이 행렬을 제안합니다:

$$Q_t^{mask} = \begin{pmatrix} \alpha_t & \beta_t & \beta_t & \cdots & \beta_t & 0 \\ \beta_t & \alpha_t & \beta_t & \cdots & \beta_t & 0 \\ \vdots & \vdots & \ddots & \ddots & \vdots & \vdots \\ \beta_t & \beta_t & \cdots & \beta_t & \alpha_t & 0 \\ \cdots & \cdots & \cdots & \cdots & \cdots & 1 \end{pmatrix}$$

여기서:
- **마스킹 확률** $\alpha_t$: 토큰이 $[MASK]$ 토큰으로 대체되는 확률
- **균등 확산 확률** $\beta_t$: 모든 토큰이 균등하게 샘플링될 확률  
- **보존 확률** $1 - \alpha_t - K\beta_t$: 토큰이 그대로 유지될 확률

이 전략의 **닫힌 형태 계산**(computational efficiency):

$$Q_t^t v_{x_0} = \bar{\alpha}_t v_{x_0} + \bar{\beta}_t v_{\text{uniform}} + \bar{\gamma}_t v_{K+1}$$

이를 통해 계산 복잡도가 $O(tK^2)$에서 $O(K)$로 감소합니다.

**마스크-교체 전략의 이점**:
1. **손상된 토큰의 명확성**: 네트워크가 어느 위치가 손상되었는지 명시적으로 알 수 있음
2. **이론적 증명**: 마스킹만 사용하면 $x_t = x_0$일 때 자명한 후험 분포가 되므로, 균등 노이즈 필요성을 입증
3. **문맥 강화**: 무작위 토큰 교체로 네트워크가 진정한 문맥 이해 학습
4. **빠른 수렴**: 마스크된 토큰이 신경망의 주의를 집중시켜 수렴 가속화

### 3.4 역 과정 학습

**변분 하한(VLB) 최소화**:

$$L_{vlb} = L_0 + L_1 + \cdots + L_{T-1} + L_T$$

여기서:
- $L_0 = \log p(x_0|x_1, y)$
- $L_{t-1} = D_{KL}(q(x_{t-1}|x_t, x_0) \parallel p(x_{t-1}|x_t, y))$ for $1 \leq t < T$
- $L_T = D_{KL}(q(x_T|x_0) \parallel p(x_T))$

**마스크-교체 확산의 선행(prior)**:

$$p(x_T) = (\alpha_T, \ldots, \alpha_T, \gamma_T)$$

### 3.5 재매개변수화 기법

네트워크 매개변수화는 생성 품질에 중대한 영향을 미칩니다. 이전 작업들과 일치하게, 직접 후험 분포 대신 **노이즈 없는 토큰 분포**를 예측합니다:

$$p(x_{t-1}|x_t, y) = \sum_{x_0=1}^{K} q(x_{t-1}|x_t, x_0)p(x_0|x_t, y)$$

**보조 목표 함수**:

$$L_{x_0} = -\log p(x_0|x_t, y)$$

이를 변분 하한과 결합하면 이미지 품질이 향상됩니다.

### 3.6 모델 아키텍처

**인코더-디코더 트랜스포머** 구조:

```
┌─────────────────────────────────────────────┐
│ 1. 텍스트 인코더 (Text Encoder)              │
│    - CLIP 기반 고정(frozen) 토크나이저      │
│    - 77개 토큰 길이의 조건부 특성 시퀀스   │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│ 2. 확산 이미지 디코더 (Diffusion Image Decoder)│
│    - 여러 개의 트랜스포머 블록              │
│    - 각 블록: Self-Attention + Cross-Attn   │
│    - AdaLN(Adaptive Layer Norm)로 시간 정보 주입 │
│    - Softmax 출력층으로 p(x_0|x_t, y) 산출  │
└─────────────────────────────────────────────┘
```

**두 가지 모델 사양**:
- **VQ-Diffusion-S(Small)**: 18 블록, 192 차원, 34M 파라미터
- **VQ-Diffusion-B(Base)**: 19 블록, 1024 차원, 370M 파라미터
- **VQ-Diffusion-F(Fine-tuned)**: Conceptual Captions으로 사전학습 후 미세조정

### 3.7 빠른 추론 전략

재매개변수화 기법을 활용하여 **시간 스트라이드** $\Delta t$로 샘플링 단계를 건너뜁니다:

$$(x_T, x_{T-\Delta t}, x_{T-2\Delta t}, \ldots, x_0)$$

이를 통해:
- 3-4 추론 단계만으로도 양호한 품질 유지
- 약 1/3-1/4 계산량으로 비교 가능한 성능 달성

## 4. 성능 향상 및 실험 결과

### 4.1 정량적 성능 비교 (FID 점수 기준)

| 데이터셋 | MSCOCO | CUB-200 | Oxford-102 |
|---------|--------|---------|-----------|
| StackGAN | 74.05 | 51.89 | 55.28 |
| DM-GAN | 32.64 | 16.09 | 기타 |
| DF-GAN | 21.42 | 14.81 | 기타 |
| DALL-E | 27.50 | 56.10 | 기타 |
| CogView | 27.10 | 기타 | 기타 |
| **VQ-Diffusion-S** | **30.17** | **12.97** | **14.95** |
| **VQ-Diffusion-B** | **19.75** | **11.94** | **14.88** |
| **VQ-Diffusion-F** | **13.86** | **10.32** | **14.10** |

**주요 결과**:
- VQ-Diffusion-B는 매개변수 수가 비슷한 GAN 기반 방법들을 **대폭 상회**
- VQ-Diffusion-F는 DALL-E/CogView의 1/10 파라미터로 **더 우수한 성능** 달성

### 4.2 추론 속도 비교

| 모델 | 추론 단계 | FID | 처리량(img/s) |
|------|---------|-----|------------|
| VQ-AR-S | - | 18.12 | 0.08 |
| VQ-Diffusion-S (100 steps) | 100 | 12.97 | 0.37 |
| VQ-Diffusion-S (25 steps) | 25 | 15.46 | 1.25 |
| VQ-AR-B | - | 17.76 | 0.03 |
| VQ-Diffusion-B (100 steps) | 100 | 11.94 | 0.13 |
| VQ-Diffusion-B (25 steps) | 25 | 14.03 | 0.47 |

**결론**: VQ-Diffusion은 **15배 빠른 속도**로 더 나은 품질 달성

### 4.3 무조건 및 클래스 조건부 생성

**ImageNet (클래스 조건부)**:
- VQ-Diffusion: FID 11.89 (상위 5% 필터링시 5.32)
- VQGAN: FID 15.78 (상위 5% 필터링시 5.88)
- ADM-G: FID 10.94 (상위 5% 필터링시 4.59)

**FFHQ (무조건 생성)**:
- VQ-Diffusion: FID 6.33
- VQGAN: FID 9.6
- StyleGAN2: FID 3.8

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 현재 강점

**1) 구조적 유연성**
- 비자기회귀적 생성으로 모든 토큰이 동등한 위치에서 생성
- 2D 이미지 구조에 내재된 공간적 편향 없음
- 다양한 해상도와 종횡비 입력에 대한 적응성 향상

**2) 오류 누적 방지**
- 각 단계에서 모든 토큰을 동시 수정 가능
- 초기 오류의 전파 효과 최소화
- 더 깊은 생성 단계에서도 일관된 성능 유지

**3) 전역 맥락 활용**
- 트랜스포머의 self-attention이 모든 위치의 정보 접근
- 장거리 의존성 학습 능력

**4) 데이터 확장성**
- LAION-400M 같은 대규모 데이터셋에서 우수한 확장성
- Conceptual Captions로 사전학습 후 미세조정 전략의 효과성 입증

### 5.2 일반화 성능 향상을 위한 전략

**1) 데이터 다양성 확대**
- 다양한 스타일, 도메인, 해상도의 이미지-텍스트 쌍 학습
- Stable Diffusion이 LAION-5B에서 달성한 것처럼 개방형 생성 능력 향상

**2) 조건부 생성 강화**
- 레이아웃, 스타일 가이드 같은 추가 조건 통합
- 마스크 기반 인페인팅 자연스러운 확장

**3) 전이 학습(Transfer Learning)**
- 대규모 데이터에서 사전학습된 텍스트 인코더(CLIP) 활용
- 소규모 도메인 데이터에서의 빠른 미세조정 가능
- 도메인 간 생성 능력 향상

**4) 구조적 개선**
- 트랜스포머 스케일링으로 더 큰 모델 용량
- Diffusion Transformer(DiT) 같은 순수 트랜스포머 아키텍처 채택
- 효율성 증대(예: 선택적 주의 메커니즘)

### 5.3 한계 및 개선 필요 영역

**1) 텍스트 정렬의 한계**
- 복잡한 합성 개념이나 세부적인 객체 특성 표현 어려움
- 정교한 의미론적 분해 필요

**2) 해상도 제약**
- 현재 256×256 기본 지원(VQ-VAE의 다운샘플링 인수가 32)
- 초고해상도 생성을 위해 해상도 계층적 구조 필요

**3) 계산 비용**
- 대규모 모델 학습에 여전히 상당한 GPU 자원 소요
- 추론도 GAN보다는 다단계 필요

## 6. 2020년 이후 관련 최신 연구 동향

### 6.1 잠재 확산 모델 시대 (2021-2022)

**Latent Diffusion Models (LDM, Stable Diffusion)**
- **핵심 혁신**: 픽셀 공간이 아닌 **잠재 공간에서 확산** 수행
- VAE 기반 자동 인코더로 이미지를 저해상도 잠재 표현으로 압축
- 계산량 약 **50-60배 감소**로 소비자 GPU에서 고해상도 생성 가능
- VQ-Diffusion과 달리 **연속 잠재 공간** 사용
- Cross-Attention을 통한 텍스트 조건화

**DALL-E 2 (2022)**
- CLIP의 이미지-텍스트 정렬 활용
- 두 단계 생성: CLIP 임베딩 생성 후 이미지 생성
- 사전학습된 대규모 텍스트 인코더의 중요성 입증

### 6.2 트랜스포머 기반 아키텍처 (2023)

**Diffusion Transformer (DiT)**
- **U-Net 대체**: U-Net 대신 순수 트랜스포머 사용
- 더 나은 스케일링 특성
- 매개변수 효율성 개선

**Scalability 향상**
- 모델 크기에 따른 성능 개선이 선형적으로 확장
- 더 많은 데이터, 더 큰 모델로 지속적 개선 가능

### 6.3 효율성 개선 (2023-2024)

**Consistency Models**
- **2-4 단계만으로 생성** 가능(기존 50-100+ 단계)
- 약 **50배 속도 향상**
- 사전학습된 확산 모델으로부터 증류

**Flow Matching / Rectified Flows**
- 확산 모델의 수학적 일반화
- 더 직선적인 샘플링 경로
- 더 적은 단계로 동일 품질 달성

**빠른 추론 전략들**
- DDIM(Denoising Diffusion Implicit Models)
- 초기 단계 건너뛰기
- 동적 임계값 처리

### 6.4 도메인 일반화 (2023-2025)

**Cross-Domain Generalization**
- 사전학습된 LDM을 도메인 적응 작업에 활용
- 스타일 이전 및 특성 정렬 개선

**도메인 간 생성**
- LDM 기반 데이터 증강으로 도메인 시프트 완화
- 다중 도메인 학습에서의 일반화 능력 향상

**세계 모델(World Models) 방향**
- 단순 이미지 생성 넘어 영상 생성, 3D 형상 생성 확장
- 시간적 일관성 학습

### 6.5 개방형 생성의 도전과 발전

**대규모 데이터 학습**
- LAION-5B 같은 초대규모 데이터셋 활용
- 오픈 도메인 생성 능력 획기적 향상

**Prompt Engineering**
- 구조화된 프롬프트로 생성 제어성 강화
- 부정적 프롬프트(negative prompts) 도입

**멀티모달 기초 모델**
- 이미지, 텍스트, 3D, 영상을 통합 처리
- 서로 다른 모달리티 간 지식 전이

### 6.6 VQ-Diffusion 이후 발전

**이산 토큰 기반 확산의 한계**
- 연속 잠재 공간(LDM, Stable Diffusion)의 우수성 입증
- 그러나 **이산 표현의 명확성과 해석성** 여전히 가치 있음

**하이브리드 접근**
- 이산 토큰과 연속 잠재의 장점 결합 연구
- 벡터량자화 개선(VQ-VAE-2 등)

**특화된 도메인 모델**
- 의료 이미지, 디자인, 건축 등 특정 도메인 최적화 모델
- 도메인 지식 통합

***

## 7. 논문이 앞으로의 연구에 미치는 영향

### 7.1 긍정적 영향

**1) 비자기회귀 생성의 재평가**
- 자기회귀만이 능사가 아님을 명시적으로 입증
- 동시(parallel) 생성 방식의 가능성 제시

**2) 이산 표현의 가치**
- VQ-VAE 같은 이산 토큰화 방식이 여전히 경쟁력 있음
- 해석성과 효율성의 균형 가능

**3) 트랜스포머의 유연성**
- 트랜스포머가 다양한 생성 패러다임 지원 가능
- 아키텍처의 중요성보다 **학습 목표의 설계**가 더 중요

**4) 효율성과 품질의 트레이드오프 재검토**
- 더 작은 모델로 대규모 모델과 비슷한 성능 달성 가능
- 리소스 제약 환경에서의 실용성 강화

### 7.2 후속 연구 방향

**1) 더 효율적인 확산 과정**
- 마스크-교체 전략의 최적화
- 비균등 노이즈 스케줄 설계

**2) 더 나은 코드북 학습**
- VQ-VAE의 한계(코드북 붕괴, 고주파 정보 손실) 개선
- 다중 레벨 양자화

**3) 조건화 메커니즘 강화**
- 텍스트뿐 아니라 다양한 조건(레이아웃, 속성 등) 통합
- 더 정교한 제어 가능성

**4) 스케일링 법칙 탐색**
- 모델 크기, 데이터 크기, 계산량에 따른 성능 관계
- 최적 리소스 할당 전략

**5) 멀티모달 생성으로의 확장**
- 텍스트-이미지-3D 통합 생성
- 영상 생성에서의 시간적 일관성

### 7.3 실무적 영향

**1) 산업 응용**
- 소비자 수준 하드웨어에서 고품질 생성 가능
- 온디바이스(on-device) 생성 모델 가능성

**2) 창작 도구**
- 인터랙티브 이미지 편집
- 디자인 개념화 도구

**3) 데이터 합성**
- 학습 데이터 부족 도메인에서의 데이터 증강
- 프라이버시 보존 합성 데이터 생성

## 8. 향후 연구 시 고려할 점

### 8.1 기술적 고려사항

**1) 모델 선택의 근거**
- 이산 vs. 연속 잠재 공간 선택 기준 명확화
- 각 방식의 장단점을 작업별로 분석

**2) 데이터 전략**
- 다양성과 품질의 균형
- 도메인 특화 vs. 범용 데이터셋

**3) 평가 지표 확대**
- FID 외에 의미론적 정렬도 측정
- 사용자 연구 및 정성적 평가 중요성

**4) 계산 효율성**
- 추론 단계 수 vs. 품질 곡선 체계적 분석
- 배포 환경에 맞는 최적화

### 8.2 방법론적 고려사항

**1) 일반화 성능 평가**
- 기존 데이터셋과 새로운 도메인에서의 성능 비교
- Zero-shot 생성 능력 평가

**2) 강건성(Robustness)**
- 적대적 입력에 대한 내성
- 분포 외(out-of-distribution) 프롬프트 처리

**3) 공정성(Fairness)**
- 학습 데이터의 편향 추적
- 생성된 이미지의 편향성 평가

**4) 해석성(Interpretability)**
- 이산 토큰의 의미 분석
- 생성 과정의 단계별 변화 이해

### 8.3 미래 방향성

**1) 멀티스케일 생성**
- 일관성 있는 고해상도 생성
- 부분 생성과 전체 일관성의 균형

**2) 동적 조건화**
- 사용자 피드백 기반 반복적 생성
- 대화형 생성 인터페이스

**3) 통합 모델**
- 텍스트, 이미지, 3D, 영상을 한 모델에서 처리
- 모달리티 간 자연스러운 전이

**4) 에너지 효율**
- 더 적은 계산량으로 동등 품질
- 환경 친화적 AI 모델 개발

***

## 결론

**VQ-Diffusion**은 텍스트-이미지 생성 분야에서 **확산 모델과 이산 표현의 조화**를 통해 중요한 전환점을 마련했습니다. 일방향 편향 제거, 누적 오류 방지, 효율적 추론 등 자기회귀 모델의 근본적 한계를 극복하면서도, 더 작은 모델 규모로 경쟁력 있는 성능을 달성했습니다.

이 논문 이후 LDM, Stable Diffusion, DiT 등으로 진화한 확산 모델 기술은 **생성 AI의 민주화**를 이루었으며, 도메인 일반화, 효율성 개선, 멀티모달 통합 등의 새로운 연구 방향을 제시했습니다.

앞으로의 연구는 **일반화 성능, 계산 효율, 해석성, 멀티모달 통합**에 집중할 것으로 예상되며, VQ-Diffusion의 핵심 아이디어들은 이러한 발전의 토대가 될 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5f3123db-d189-4bd2-81b8-c56c54e40246/2111.14822v3.pdf)
[2](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[3](https://e-journal.unair.ac.id/JESTT/article/view/47782)
[4](https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs)
[5](https://iopscience.iop.org/article/10.1149/MA2025-024658mtgabs)
[6](https://arxiv.org/pdf/2211.01324.pdf)
[7](http://arxiv.org/pdf/2211.15388.pdf)
[8](https://arxiv.org/html/2503.01645v1)
[9](https://arxiv.org/pdf/2503.05149.pdf)
[10](http://arxiv.org/pdf/2407.00752.pdf)
[11](https://arxiv.org/html/2412.12888v1)
[12](https://arxiv.org/pdf/2403.04279.pdf)
[13](http://arxiv.org/pdf/2301.09515.pdf)
[14](https://www.edge-ai-vision.com/2023/01/from-dall%C2%B7e-to-stable-diffusion-how-do-text-to-image-generation-models-work/)
[15](https://ffighting.net/deep-learning-paper-review/diffusion-model/dalle2/)
[16](https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Global_Context_With_Discrete_Diffusion_in_Vector_Quantised_Modelling_for_CVPR_2022_paper.pdf)
[17](https://arxiv.org/html/2303.07909v3)
[18](https://mvje.tistory.com/282)
[19](https://papers.miccai.org/miccai-2025/paper/3774_paper.pdf)
[20](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DesignDiffusion_High-Quality_Text-to-Design_Image_Generation_with_Diffusion_Models_CVPR_2025_paper.pdf)
[21](https://dhpark1212.tistory.com/entry/Dall-e-2-%EB%B0%8F-%EC%A3%BC%EB%B3%80-%EA%B8%B0%EC%88%A0-%EB%A6%AC%EB%B7%B0)
[22](https://arxiv.org/html/2511.06863v1)
[23](https://github.com/AlonzoLeeeooo/awesome-text-to-image-studies)
[24](https://arxiv.org/abs/2212.00842)
[25](https://dl.acm.org/doi/10.1145/3592450)
[26](https://arxiv.org/abs/2212.06013)
[27](https://ieeexplore.ieee.org/document/10484137/)
[28](https://ieeexplore.ieee.org/document/10052908/)
[29](https://arxiv.org/abs/2209.14697)
[30](https://arxiv.org/abs/2211.00611)
[31](https://ieeexplore.ieee.org/document/10204618/)
[32](https://ieeexplore.ieee.org/document/10204696/)
[33](https://www.semanticscholar.org/paper/5d45869030a69ae67f47c16d020dc630b9f77a30)
[34](https://arxiv.org/abs/2311.05556)
[35](http://arxiv.org/pdf/2403.16024.pdf)
[36](https://arxiv.org/html/2410.06055)
[37](http://arxiv.org/pdf/2310.04378.pdf)
[38](https://arxiv.org/pdf/2304.08291.pdf)
[39](https://arxiv.org/abs/2210.11058)
[40](https://arxiv.org/pdf/2401.10227.pdf)
[41](http://arxiv.org/pdf/2305.15759.pdf)
[42](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
[43](https://proceedings.neurips.cc/paper_files/paper/2023/file/d6c01b025cad37d5c8bab4ba18846c02-Paper-Conference.pdf)
[44](https://milvus.io/ai-quick-reference/what-techniques-help-improve-the-generalization-of-diffusion-models)
[45](https://arxiv.org/abs/2112.10752)
[46](https://www.scitepress.org/Papers/2024/129378/129378.pdf)
[47](https://openaccess.thecvf.com/content/WACV2024/papers/Niemeijer_Generalization_by_Adaptation_Diffusion-Based_Domain_Extension_for_Domain-Generalized_Semantic_Segmentation_WACV_2024_paper.pdf)
[48](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ldm/)
[49](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf)
[50](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05806.pdf)
[51](https://ieeexplore.ieee.org/document/10491767/)
[52](https://ieeexplore.ieee.org/document/10571357/)
[53](https://ieeexplore.ieee.org/document/10726709/)
[54](https://ojs.aaai.org/index.php/AAAI/article/view/28199)
[55](https://ieeexplore.ieee.org/document/10389779/)
[56](https://www.ijcai.org/proceedings/2024/251)
[57](http://medrxiv.org/lookup/doi/10.1101/2024.07.17.24310565)
[58](https://www.semanticscholar.org/paper/b021962b5ecd1fe2d94b5488ec0ed99004b8585a)
[59](https://dl.acm.org/doi/10.1145/3746027.3755386)
[60](https://ieeexplore.ieee.org/document/11092767/)
[61](https://arxiv.org/html/2307.02138)
[62](https://arxiv.org/pdf/2402.04929.pdf)
[63](https://arxiv.org/html/2312.05387v1)
[64](https://arxiv.org/pdf/2305.18455.pdf)
[65](https://arxiv.org/pdf/2410.16020v1.pdf)
[66](https://arxiv.org/pdf/2504.01521.pdf)
[67](https://arxiv.org/html/2411.19339v2)
[68](https://arxiv.org/html/2306.16425)
[69](https://arxiv.org/html/2506.21042v2)
[70](https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Mofayezi_Benchmarking_Robustness_to_Text-Guided_Corruptions_CVPRW_2023_paper.pdf)
[71](https://www.youtube.com/watch?v=NC9-b7j1Ltc)
[72](https://www.sciencedirect.com/science/article/abs/pii/S0951832025005733)
[73](https://aclanthology.org/2023.findings-emnlp.595.pdf)
[74](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/)
[75](https://openaccess.thecvf.com/content/ICCV2025/papers/He_Boosting_Domain_Generalized_and_Adaptive_Detection_with_Diffusion_Models_Fitness_ICCV_2025_paper.pdf)
[76](https://www.emergentmind.com/topics/zero-shot-text-to-image-generation)
[77](https://arxiv.org/abs/2407.02398)
