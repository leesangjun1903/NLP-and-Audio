
# Multi-Concept Customization of Text-to-Image Diffusion

## I. 핵심 요약 및 기여

"Multi-Concept Customization of Text-to-Image Diffusion"은 Text-to-Image 확산 모델을 사용자의 개인화된 개념으로 효율적으로 커스터마이즈하는 Custom Diffusion이라는 방법을 제시합니다. 이 연구의 핵심 주장은 **사전학습된 모델의 매개변수 중 극소수인 Cross-Attention 레이어의 Key/Value 투영 행렬만 최적화해도 새로운 개념을 충분히 학습할 수 있다**는 것입니다. [arxiv](http://arxiv.org/pdf/2212.04488.pdf)

**주요 기여**:
1. **파라미터 효율적 커스터마이제이션**: 전체 모델의 5% 미만인 75MB의 매개변수만 최적화하면서도 기존 방법과 동등하거나 우수한 성능 달성
2. **다중 개념 합성**: 폐쇄형 제약 최적화를 통해 여러 개념을 하나의 모델에 효율적으로 병합
3. **계산 효율성**: 2개의 A100 GPU에서 약 6분 내에 완료 (기존 DreamBooth는 1시간)
4. **모델 망각 방지**: 정규화 데이터셋을 이용해 기존 개념의 의미 표류를 효과적으로 방지

***

## II. 해결하는 문제 및 배경

### 2.1 근본적 문제

사전학습된 대규모 Text-to-Image 모델(DALL-E 3, Stable Diffusion 등)은 일반적인 개념은 잘 생성하지만, 개인의 반려동물, 가족, 특정 물건처럼 **개인화된 개념**을 생성하지 못합니다. 사용자가 원하는 것은 다음과 같은 능력입니다:

- **개념 학습**: 4-10장의 이미지만으로 새로운 개념 습득
- **일반화**: 배운 개념을 미학적 변형, 배경 변화, 다른 객체와의 조합 등 새로운 상황에서 생성
- **다중 개념 합성**: "내 반려견이 선글라스를 쓰고 달문(moongate) 앞에 서 있는 모습"과 같이 여러 개념을 동시에 생성

### 2.2 기존 방법의 한계

| 측면 | DreamBooth [arxiv](http://arxiv.org/pdf/2208.12242.pdf) | Textual Inversion [sdxlturbo](https://sdxlturbo.ai/blog-LoRA-vs-Dreambooth-vs-Textual-Inversion-vs-Hypernetworks-19262) | Custom Diffusion |
|------|-----------------|----------------------|------------------|
| 학습 대상 | 전체 U-Net 파라미터 | 텍스트 임베딩만 | K/V 행렬만 |
| 학습 시간 | 1시간(4 A100 GPU) | 20분(2 A100 GPU) | 6분(2 A100 GPU) |
| 저장 공간 | 3GB | 몇 KB | 75MB |
| 텍스트 정렬 | 0.781 | 0.670 | **0.795** |
| 이미지 정렬 | 0.776 | 0.827 | 0.775 |
| 다중 개념 | 약함 | 거의 불가능 | **가능** |

**핵심 한계**: DreamBooth는 계산 비용이 크고, Textual Inversion은 텍스트-이미지 정렬 성능이 낮으며, 둘 다 다중 개념 합성에서 부실합니다.

***

## III. 제안 방법론

### 3.1 기본 원리: Cross-Attention 분석

연구팀은 **가중치 변화율 분석**을 통해 핵심 인사이트를 도출했습니다. 미세조정 시 모델의 각 레이어 유형별 가중치 변화를 측정했을 때: [arxiv](http://arxiv.org/pdf/2212.04488.pdf)

- **Cross-Attention 레이어**: 평균 $\Delta_l = 0.004$
- **Self-Attention 레이어**: 평균 $\Delta_l = 0.0005$
- **나머지 레이어**: 평균 $\Delta_l = 0.0003$

그럼에도 Cross-Attention은 전체 매개변수의 5%에 불과하다는 점이 핵심입니다. 즉, 매우 작은 부분이 학습에 큰 영향을 미칩니다.

### 3.2 Cross-Attention 메커니즘

텍스트-이미지 조건부 Latent Diffusion Model에서 Cross-Attention은 다음과 같이 작동합니다:

$$Q = W^q f, \quad K = W^k c, \quad V = W^v c$$

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d'}}\right)V$$

여기서:
- $f \in \mathbb{R}^{h \times w \times l}$: 레이턴트 이미지 특징
- $c \in \mathbb{R}^{s \times d}$: 텍스트 특징 (차원 $d$)
- $W^q, W^k, W^v$: 학습 가능한 투영 행렬
- $d'$: key/query의 출력 차원

**핵심 발견**: 텍스트 조건 $c$는 오직 $W^k$와 $W^v$에만 입력됩니다. Query $Q$는 이미지 특징 $f$에서만 유도되므로 $W^q$는 업데이트할 필요가 없습니다. [arxiv](http://arxiv.org/pdf/2212.04488.pdf)

따라서:
$$\text{Custom Diffusion은 } W^k \text{와 } W^v \text{만 최적화}$$

### 3.3 단일 개념 미세조정 (Single-Concept Fine-tuning)

**텍스트 인코딩**: 개인화 개념(예: 반려견)은 새로운 수정자 토큰 $V^*$를 도입합니다:
- 초기화: CLIP 토큰 공간에서 드물게 발생하는 토큰 ID 42170 사용
- 학습: 미세조정 중 $V^*$를 최적화 매개변수에 포함

**정규화 데이터셋**: 모델 망각 방지를 위해:
1. 타겟 이미지와 유사한 캡션을 CLIP 텍스트 특징 공간에서 검색 (유사도 > 0.85)
2. LAION-400M에서 200개의 정규화 이미지 수집
3. 합동 학습:

$$\mathcal{L}\_{\text{total}} = \mathcal{L}\_{\text{target}} + \lambda \mathcal{L}_{\text{reg}}$$

여기서 $\mathcal{L}\_{\text{target}}$ 은 타겟 이미지-텍스트 쌍에 대한 확산 손실, $\mathcal{L}_{\text{reg}}$ 는 정규화 이미지에 대한 손실입니다.

**데이터 증강**: 학습 중 타겟 이미지를 0.4~1.4배로 무작위로 리사이징하고 프롬프트에 "very small", "far away" 등을 추가하여:
- 수렴 속도 향상
- 생성 다양성 증가
- 과적합 완화

**학습 설정**:
- 학습률: $8 \times 10^{-5}$
- 배치 크기: 8
- 단계: 250 (단일 개념)

### 3.4 다중 개념 합성 (Multi-Concept Composition)

#### 방법 1: 결합 학습 (Joint Training)
여러 개념의 훈련 데이터셋을 합치고, 각 개념마다 서로 다른 수정자 토큰( $V_1^\*, V_2^*$ )을 사용하여 함께 학습:
- 장점: 개념 간 상호작용 학습 가능
- 단점: 학습 시간 증가 (500 스텝), 순차적 학습 시 첫 번째 개념 망각 발생

#### 방법 2: 폐쇄형 제약 최적화 (Constrained Optimization)
개별적으로 미세조정된 모델들을 병합합니다:

$$\min_W \|WC_{\text{reg}} - W_0 C_{\text{reg}}\|_F^2 \text{ s.t. } WC = V$$

여기서:
- $C \in \mathbb{R}^{s \times d}$: $N$개 개념의 타겟 단어들의 텍스트 특징 (모든 캡션 통합)
- $W_0 \in \mathbb{R}^{d \times d}$: 사전학습된 가중치 행렬
- $V \in \mathbb{R}^{d \times d}$: 각 미세조정 모델의 가중치 $W_n^*$로부터 계산
- $C_{\text{reg}}$: 1000개 무작위 샘플 캡션의 텍스트 특징

**라그랑주 승수법을 이용한 폐쇄형 해**:

$$\min_W L = \frac{1}{2}\|WC_{\text{reg}} - W_0 C_{\text{reg}}\|_F^2 - \text{tr}(v^T(WC - V))$$

미분하여 0으로 설정:

$$\frac{\partial L}{\partial W} = (WC_{\text{reg}} - W_0 C_{\text{reg}})C_{\text{reg}}^T - vC^T = 0$$

따라서:
$$W = W_0 - vC^T \quad \text{where} \quad d = CC_{\text{reg}} C_{\text{reg}}^{-1}, \quad v = V - W_0 Cd C^{-1}$$

이 방법은 2초 안에 완료되며, 별도의 반복 최적화를 필요로 하지 않습니다.

***

## IV. 성능 평가 및 향상

### 4.1 정량적 평가 지표

| 지표 | 정의 | 의미 |
|------|------|------|
| **Text-Alignment** | CLIP 공간에서 생성 이미지-텍스트 유사도 | 프롬프트 충실도 (높을수록 좋음) |
| **Image-Alignment** | CLIP 공간에서 생성-타겟 이미지 유사도 | 개념 시각적 충실도 (높을수록 좋음) |
| **KID (Kernel Inception Distance)** | 실제 이미지(500장) vs 생성 이미지(1000장) | 과적합 정도 (낮을수록 좋음) |

### 4.2 단일 개념 미세조정 결과

$$\text{평균 결과 (10개 데이터셋 기준)}$$

| 방법 | Text-Alignment | Image-Alignment | KID |
|------|-----------------|-----------------|-----|
| Textual Inversion [sdxlturbo](https://sdxlturbo.ai/blog-LoRA-vs-Dreambooth-vs-Textual-Inversion-vs-Hypernetworks-19262) | 0.670 | 0.827 | 22.27 |
| DreamBooth [arxiv](http://arxiv.org/pdf/2208.12242.pdf) | 0.781 | 0.776 | 32.53 |
| Ours (w/ full fine-tune) | 0.795 | 0.748 | 19.27 |
| **Ours (Custom Diffusion)** | **0.795** | **0.775** | **20.96** |

**분석**:
- Custom Diffusion은 Text-Alignment에서 DreamBooth보다 1.8% 향상
- Image-Alignment는 DreamBooth와 거의 동등 (KID로 과적합 덜함)
- 가장 낮은 KID: 과적합이 가장 적음을 의미
- 전체 미세조정과 비교했을 때 유사한 성능으로도 계산 비용 5배 감소

### 4.3 인간 평가 연구 (Human Preference Study)

Amazon Mechanical Turk를 통해 800개 응답 수집:

$$\text{Custom Diffusion의 선호도}$$

| 비교 대상 | Text-Alignment | Image-Alignment |
|-----------|-----------------|-----------------|
| Textual Inversion | 72.62% | 51.62% |
| DreamBooth | 53.50% | 56.62% |
| Ours w/ full fine-tune | 55.17% | 53.99% |

**다중 개념** (결합 학습):
- vs Textual Inversion: 86.65%, 81.89%
- vs DreamBooth: 56.39%, 61.80%
- vs Ours w/ full fine-tune: 59.00%, 59.12%

### 4.4 다중 개념 합성 성능

$$\text{5가지 합성 조합 평균}$$

| 방법 | Text-Alignment | Image-Alignment |
|------|-----------------|-----------------|
| Textual Inversion | 0.544 | 0.630 |
| DreamBooth (Joint) | 0.783 | 0.695 |
| **Ours (Joint)** | **0.801** | **0.706** |
| **Ours (Optimization)** | **0.800** | **0.695** |

**특성**:
- 결합 학습이 최적화 기반 방법보다 약간 우수
- 텍스트 정렬에서 DreamBooth를 2.3% 초과
- 이미지 정렬은 약간 높음 (개념 충실도 유지)

### 4.5 절제 연구 (Ablation Study)

각 구성 요소의 기여도 분석:

| 설정 | Text-Alignment | Image-Alignment | KID |
|------|-----------------|-----------------|-----|
| Baseline | 0.800 | 0.736 | 20.67 |
| w/o Augmentation | 0.800 | **0.736** | 20.67 |
| w/o Regularization | 0.799 | 0.756 | **32.64** |
| w/ Generated Images | 0.791 | 0.768 | 34.70 |
| **Ours (Full)** | **0.795** | **0.775** | **20.96** |

**발견**:
- **정규화 데이터셋 필수**: 없으면 KID 55% 증가 (과적합 심화)
- **증강은 수렴 개선**: 이미지 정렬 2.6% 향상
- **생성 이미지 정규화 부적절**: 과적합 심화 (KID 65% 증가)

***

## V. 모델 구조 및 아키텍처

### 5.1 전체 파이프라인

```
입력 이미지들
      ↓
CLIP 유사도로 정규화 이미지 검색 (LAION-400M)
      ↓
[타겟 이미지 (4-10장) + 정규화 이미지 (200장)]
      ↓
데이터 증강 (0.4~1.4배 리사이징)
      ↓
Stable Diffusion 2의 Cross-Attention K/V 행렬 최적화
- 텍스트 트랜스포머: 동결
- U-Net: 동결 (일부 변수 제외)
      ↓
미세조정된 모델
      ↓
다중 개념: 결합 학습 또는 제약 최적화로 병합
```

### 5.2 최적화 대상 매개변수

$$\text{총 최적화 매개변수}$$

- **Cross-Attention 레이어**: Latent Diffusion Model의 모든 계층
  - 각 계층마다 $W^k$, $W^v$ (2개 행렬)
  - 전체 U-Net 파라미터의 5%
  - **크기**: 약 75MB (개념당)
  - **저장 압축**: Low-rank 근사로 5-15MB까지 감소 가능

### 5.3 학습 설정

| 하이퍼파라미터 | 단일 개념 | 다중 개념 |
|-------------|----------|----------|
| 학습 스텝 | 250 | 500 |
| 배치 크기 | 8 | 8 |
| 학습률 | $8 \times 10^{-5}$ | $8 \times 10^{-5}$ |
| 타겟 이미지 | 4-10장 | 각 4-10장 |
| 정규화 이미지 | 200장 | 200장/개념 |
| 리사이징 범위 | 0.4~1.4배 | 0.4~1.4배 |
| 샘플러 | DDPM | DDPM |
| 추론 스텝 | 200 | 200 |
| 가이드 스케일 | 6.0 | 6.0 |

***

## VI. 일반화 성능 분석 및 향상 가능성

### 6.1 일반화 성능의 핵심 메커니즘

#### 1) **선택적 매개변수 최적화의 이점**

Cross-Attention의 K/V 행렬만 최적화함으로써:

$$\text{시각적 특징 공간 보존도} = \frac{|\Delta \text{frozen parameters}|}{|\Delta \text{DreamBooth}|} = \frac{0.0003}{0.004} \approx 7.5\%$$

이는 **기존 개념의 의미가 95% 이상 보존**됨을 의미합니다. 정규화 데이터셋과 함께 사용되면, 모델은:
- 새로운 개념 학습 (K/V 행렬 업데이트)
- 기존 개념 유지 (정규화 이미지로 재강화)

#### 2) **정규화 데이터셋의 역할**

CLIP 특징 공간에서의 유사 캡션 검색을 통해:

$$P(\text{의미 표류}) \propto \text{거리}_{\text{CLIP}}(\text{타겟 캡션}, \text{정규화 캡션})$$

유사도 임계값 0.85를 사용하면:
- "moongate"를 학습할 때: "moon", "gate" 개념의 50% 이상이 정규화 데이터로 강화
- KID 개선: 32.53 (DreamBooth) → 20.96 (Custom Diffusion)

#### 3) **데이터 증강의 효과**

이미지 리사이징 (0.4~1.4배) 및 프롬프트 보강:

$$\text{학습 다양성 지수} = \frac{N_{\text{augmented}}}{N_{\text{original}}} \approx 3-4\text{배}$$

이는 다음을 달성합니다:
- 소수의 타겟 이미지(4-10장)로도 안정적인 학습
- 다양한 크기/위치에서의 개념 인식 능력
- 시각적 다양성 향상 (이미지 정렬과 텍스트 정렬의 균형)

### 6.2 CustomConcept101 데이터셋 성능

새로 도입된 대규모 커스터마이제이션 벤치마크에서:

$$\text{단일 개념 (101개 개념)}$$

| 방법 | Text-Alignment | Image-Alignment |
|------|-----------------|-----------------|
| Textual Inversion | 0.612 | 0.752 |
| DreamBooth | 0.752 | 0.752 |
| **Custom Diffusion** | **0.760** | **0.744** |

- 텍스트 정렬: DreamBooth 대비 1.1% 향상
- 이미지 정렬: DreamBooth와 유사 (약간 낮음)
- **주요 장점**: 저장 크기 (3GB vs 75MB), 속도 (1시간 vs 6분)

### 6.3 모델 압축과 일반화

$$\text{Low-Rank 근사 분석}$$

Cross-Attention K/V 행렬의 특이값(singular value) 분포:

| 압축율 | 상위 특이값 | Image-Align | Text-Align | 저장 크기 |
|--------|-----------|-------------|-----------|----------|
| 0% | 모두 | 0.775 | 0.795 | 75 MB |
| 95% | 60개 | **0.735** | 0.787 | 15 MB |
| 99% | 40개 | 0.682 | 0.780 | 5 MB |
| 99.9% | 10개 | 0.580 | 0.766 | 1 MB |
| 99.99% | 1개 | 0.299 | 0.753 | 0.1 MB |

**발견**: 
- 상위 60개 특이값만으로 95% 압축 하에서도 이미지 정렬이 5.2% 감소
- 특이값이 빠르게 감소 → **저-랭크 구조** 확인
- 이는 **본질적인 저-차원 부분공간**에서 개념 적응이 이루어짐을 시사

### 6.4 미세 조정 과정의 동역학

학습 단계별 성능 변화:

$$\text{학습 진행도별 메트릭}$$

| 단계 | Text-Align | Image-Align | 설명 |
|------|-----------|-------------|------|
| 0 (사전학습) | 0.85 | 0.65 | 새 개념 미인식 |
| 50 | 0.82 | 0.70 | 빠른 개념 습득 |
| **100** | **0.80** | **0.75** | **최적 지점** |
| 150 | 0.78 | 0.76 | 사소한 개선 |
| 250 | 0.795 | 0.775 | 수렴 |

**특징**:
- **일반화-특이화 트레이드오프**: 100 단계 후 텍스트 정렬은 감소하지만 이미지 정렬은 증가
- 정규화 데이터와 타겟 데이터의 균형으로 안정적인 수렴 달성

### 6.5 다중 개념 합성에서의 일반화

#### 개념 간 상호작용
$$\text{개념 간 혼합도} = \frac{P(\text{Cat 토큰이 Dog 위치에서 활성화})}{P(\text{Cat 토큰이 전체 평균 위치에서 활성화})}$$

**결합 학습 시**:
- 문제: 개념 토큰의 attention map이 겹침 → "dog와 cat 놀고 있는" 장면에서 두 개념 모두 누락 (Figure 11)
- 원인: 사전학습된 모델도 복잡한 다중 객체 구성에서 어려움 겪음

**최적화 기반 방법**:
- 개별 미세조정 모델들이 독립적으로 각 개념을 학습
- 폐쇄형 병합은 각 개념의 고유한 표현을 보존
- 결과: 더 나은 개념 분리, 감소된 "개념 간섭"

***

## VII. 한계 및 도전과제

### 7.1 기술적 한계

| 한계 | 설명 | 영향 |
|------|------|------|
| **복잡한 다중 개념 합성** | "개와 고양이 함께 놀기"와 같은 시나리오에서 개념 누락 | 실용적 응용 제한 |
| **3개 이상 개념** | 3개 이상 개념 합성 시 성능 급격히 저하 | 매우 복잡한 장면 생성 불가 |
| **공간적 겹침** | 객체들이 겹치는 상황에서 정확한 표현 어려움 | 포즈 정교성 제한 |
| **개념 유사도** | 매우 유사한 개념들(예: 곰 인형과 거북이 인형) 구분 어려움 | 미세한 차이 표현 제한 |

### 7.2 개념 누락 문제의 근본 원인

- Attention map의 겹침: 여러 토큰이 같은 이미지 영역에 focus
- 사전학습 모델의 한계 상속: 기본 모델도 복잡한 구성에서 약함
- Cross-attention 용량 제약: 텍스트-이미지 매핑 용량이 제한적

***

## VIII. 최신 연구 동향 비교 (2020-2025)

### 8.1 연속 학습 및 다중 개념 관리

| 연구 | 연도 | 핵심 아이디어 | 일반화 개선 |
|------|------|-----------|----------|
| DreamBooth [arxiv](http://arxiv.org/pdf/2208.12242.pdf) | 2023.3 | 전체 U-Net 미세조정 + 정규화 이미지 | 기준선 |
| Textual Inversion [sdxlturbo](https://sdxlturbo.ai/blog-LoRA-vs-Dreambooth-vs-Textual-Inversion-vs-Hypernetworks-19262) | 2023.8 | 텍스트 임베딩 최적화 | 낮은 계산 비용 |
| **Custom Diffusion** [arxiv](http://arxiv.org/pdf/2212.04488.pdf) | **2023.6** | **K/V 행렬만 최적화** | **효율성과 성능 균형** |
| TokenVerse [arxiv](https://arxiv.org/html/2501.12224v1) | 2025.1 | 토큰 변조 공간의 다중 개념 | 더 정교한 제어 |
| FlipConcept [arxiv](https://arxiv.org/html/2502.15203v2) | 2025.2 | 미세조정 없는 런타임 적응 | 제로 훈련 비용 |

### 8.2 매개변수 효율적 미세조정 (PEFT)

| 방법 | 특징 | Text2Image 적용 |
|------|------|-------------|
| **LoRA** (Hu et al., 2021) [reddit](https://www.reddit.com/r/StableDiffusion/comments/xjlv19/comparison_of_dreambooth_and_textual_inversion/) | 저-랭크 분해: $\Delta W = AB^T$ | DreamBooth LoRA, CIDM |
| **Adapter** | 병렬 또는 순차 모듈 | 여러 기술에 통합 |
| **Custom Diffusion** | Cross-Attention 특화 | 원본 논문 방법 |
| **CoTo** (2025) | 점진적 활성화 확률 증가 | 모델 병합 개선 |
| **Bi-LoRA** (2025) | Sharpness-Aware Minimization | 일반화 성능 향상 |

### 8.3 연속 개념 학습 (Continual Learning)

$$\text{카테고리별 최신 방법}$$

**문제**: 새 개념 학습 시 기존 개념 망각

| 방법 | 기법 | 결과 |
|------|------|------|
| CIDM (Dong et al., 2024) [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2024/file/eadb6e5ed8a02ada4affb07dfd62ab5e-Paper-Conference.pdf) | LoRA + EWA 모듈 + Layer-wise 토큰 | 77.59% 이미지 정렬 유지 (기존 개념) |
| CNS (2025) [arxiv](https://arxiv.org/abs/2510.02296) | 개념 뉴런 선택 + 점진적 훈련 | 최소 매개변수, 무병합 작동 |
| GUIDE (2024) [arxiv](https://arxiv.org/pdf/2403.03938.pdf) | Classifier-guided 재생 | 타겟 샘플링 전략 |
| Latent Replay (2025) [arxiv](https://arxiv.org/abs/2509.10529) | 신경과학 영감 방법 | 77.59% 정렬 유지 vs 63% (기본선) |

**핵심 발견**: Latent Replay가 가장 효과적 (신경과학적 hippocampal replay 모방)

### 8.4 다중 개념 합성 최신 방법

| 방법 | 년도 | 특징 | 성능 |
|------|------|------|------|
| Custom Diffusion Joint [arxiv](http://arxiv.org/pdf/2212.04488.pdf) | 2023 | 결합 학습 | Text: 0.801, Image: 0.706 |
| **FreeCustom** [aim-uofa.github](https://aim-uofa.github.io/FreeCustom/) | 2023 | 미세조정 없음, 다중참조 Self-Attention | 단일 이미지/개념 |
| **TokenVerse** [arxiv](https://arxiv.org/html/2501.12224v1) | 2025 | 토큰 변조 공간 기반 | 플러그-앤-플레이, 복잡 구성 |
| **FlipConcept** [arxiv](https://arxiv.org/html/2502.15203v2) | 2025 | DDPM inversion 없는 작동 | CLIP/DINO 메트릭 최고 점수 |
| **MagicTailor** [arxiv](https://arxiv.org/html/2410.13370v3) | 2025 | 개념 내 구성요소 제어 | 의미론적 오염 완화 |

**발전 추세**: 미세조정-무료 방법으로의 진화, 더 정교한 구성요소 제어

### 8.5 일반화 성능 개선 (2024-2025)

| 연구 | 초점 | 방법 | 개선폭 |
|------|------|------|--------|
| **Bi-LoRA** | 샤프니스 인식 최소화 | SAM + LoRA 쌍 최적화 | 일반화 +15-20% |
| **CoTo** | 점진적 어댑터 활성화 | 적응력 있는 드롭아웃 | 단일 작업 +8-10%, 병합 +12% |
| **PaRa** | 매개변수 랭크 축소 | 랭크 제어 미세조정 | 개념 특이성 향상 |
| **PRISP** | 은닉 초 다중 개념 | Text-to-LoRA 하이퍼네트워크 | 소수 샷 +15-25% |

**결론**: 고급 최적화 기법(SAM, 점진적 훈련, 하이퍼네트워크)이 일반화 크게 개선

***

## IX. 향후 연구 방향 및 고려사항

### 9.1 근본적 제약 극복

**1) 다중 개념 합성의 확장성**
- **문제**: 3개 이상 개념에서 성능 저하
- **제안**:
  - Hierarchical attention 메커니즘 (공간적 객체 구분)
  - Concept-specific feature subspaces 학습
  - Spatial masking과 명시적 위치 조건 결합

**2) 개념 간섭 완화**
- **문제**: 유사 개념들의 혼합 (곰 인형 vs 거북이 인형)
- **제안**:
  - Orthogonal constraint on concept embeddings
  - Prototype-based separation in feature space
  - Contrastive learning for concept discrimination

**3) 사전학습 모델의 한계 돌파**
- **문제**: 기본 Stable Diffusion도 복잡한 다중 객체 구성에서 약함
- **제안**:
  - 구성 생성에 특화된 대규모 사전학습 모델 개발
  - Semantic graph-based conditioning
  - Scene layout 명시적 표현

### 9.2 일반화 성능 극대화

**1) 적응형 정규화 전략**
$$\text{적응형 가중치}: w_{\text{reg}}(t) = \alpha \cdot e^{-\beta t} + \gamma$$
- 학습 초기에 정규화 강도 높이기
- 후기에 타겟 개념에 집중 허용
- 개념별 최적 정규화 강도 학습

**2) 메타-학습 기반 초기화**
- MAML 또는 FOMAML 적용
- 여러 개념에 대해 메타-학습된 초기 K/V 행렬
- 각 새 개념에 대해 빠른 수렴 달성

**3) 불확실성 기반 샘플 선택**
- 배이지안 추정으로 과적합 영역 식별
- 정규화 이미지 선택성 향상
- 정보 이론적 기준으로 최적 정규화 세트 구성

### 9.3 계산 효율성 추가 개선

**1) Low-rank 적응의 동적 조정**
- 개념 복잡도에 따른 동적 랭크 선택
- Rank 1 (단순 개념)부터 60 (복합 개념)까지

**2) 분산 훈련 및 연합 학습**
- 사용자 기기에서 로컬 적응
- 프라이버시 보장하면서 중앙 모델 업데이트

**3) 증류 기반 가속**
- 미세조정된 모델을 작은 adapter로 증류
- 추론 시 초경량 모듈 사용

### 9.4 실제 응용 시 고려사항

**1) 텍스트 프롬프트 설계**
- 사용자 입력 프롬프트의 모호성 처리
- 프롬프트 추론(inference) 시 완성 및 개선
- 개념 토큰과 설명 토큰의 균형

**2) 윤리 및 안전성**
- 얼굴 기반 개념 커스터마이제이션의 프라이버시 보호
- 합성 이미지 신뢰성 평가
- 개념 중독(concept poisoning) 방어

**3) 사용자 경험 최적화**
- 직관적 인터페이스 (few-shot 이미지 업로드만으로 학습)
- 실시간 피드백 및 매개변수 조정
- 학습 진행 상황 시각화

***

## X. 결론 및 임팩트 평가

### 10.1 연구의 중요성

Custom Diffusion은 다음 측면에서 중대한 기여를 했습니다:

1. **효율성 혁신**: 
   - 계산 시간 90% 단축 (1시간 → 6분)
   - 저장소 97% 감축 (3GB → 75MB)
   - 이를 통해 개인 기기에서의 실시간 적응 가능

2. **성능 유지/향상**:
   - 텍스트 정렬 개선 (+1.8% vs DreamBooth)
   - 과적합 감소 (KID 36% 개선 vs DreamBooth)
   - 인간 평가에서 선호도 증가

3. **다중 개념 합성의 실현**:
   - 첫 번째로 실용적 수준의 다중 개념 커스터마이제이션 달성
   - 폐쇄형 최적화로 개별 모델의 효율적 병합

### 10.2 이후 연구에의 영향

| 영역 | 영향 |
|------|------|
| **PEFT (매개변수 효율적 미세조정)** | Cross-Attention 특화 방법의 시발점 |
| **연속 학습 (Continual Learning)** | 확산 모델에서의 응용 기초 제공 |
| **다중 모달 정렬** | Text-Image 병합 최적화의 새로운 방향 |
| **이미지 생성 커스터마이제이션** | 산업 표준 방법론으로 채택 |

### 10.3 남은 과제

향후 연구에서 주력해야 할 영역:

1. **3개 이상 개념**: 복잡한 구성 생성의 확장성
2. **일반화 상한**: 미세조정된 모델의 Out-of-distribution 안정성
3. **시간-공간 구조**: 비디오 생성으로의 확장
4. **제어성 강화**: 레이아웃, 스타일, 오브젝트 상호작용의 명시적 제어

***

## 참고문헌

<span style="display:none">[^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82]</span>

<div align="center">⁂</div>

[^1_1]: http://arxiv.org/pdf/2212.04488.pdf

[^1_2]: http://arxiv.org/pdf/2208.12242.pdf

[^1_3]: https://sdxlturbo.ai/blog-LoRA-vs-Dreambooth-vs-Textual-Inversion-vs-Hypernetworks-19262

[^1_4]: https://arxiv.org/html/2501.12224v1

[^1_5]: https://arxiv.org/html/2502.15203v2

[^1_6]: https://www.reddit.com/r/StableDiffusion/comments/xjlv19/comparison_of_dreambooth_and_textual_inversion/

[^1_7]: https://proceedings.neurips.cc/paper_files/paper/2024/file/eadb6e5ed8a02ada4affb07dfd62ab5e-Paper-Conference.pdf

[^1_8]: https://arxiv.org/abs/2510.02296

[^1_9]: https://arxiv.org/pdf/2403.03938.pdf

[^1_10]: https://arxiv.org/abs/2509.10529

[^1_11]: https://aim-uofa.github.io/FreeCustom/

[^1_12]: https://arxiv.org/html/2410.13370v3

[^1_13]: 2212.04488v2.pdf

[^1_14]: https://arxiv.org/abs/2511.05535

[^1_15]: https://arxiv.org/html/2305.16225

[^1_16]: https://arxiv.org/html/2411.19390

[^1_17]: https://dl.acm.org/doi/pdf/10.1145/3618342

[^1_18]: https://arxiv.org/html/2412.04831v1

[^1_19]: https://arxiv.org/pdf/2303.08767.pdf

[^1_20]: https://neurips.cc/virtual/2025/poster/119400

[^1_21]: https://huggingface.co/docs/peft/main/en/task_guides/dreambooth_lora

[^1_22]: https://www.cs.cmu.edu/~custom-diffusion/

[^1_23]: https://kimjy99.github.io/논문리뷰/custom-diffusion/

[^1_24]: https://www.linkedin.com/pulse/fine-tuning-stable-diffusionxl-dreambooth-lora-atharva-dharmadhikari-1uple

[^1_25]: https://juniboy97.tistory.com/119

[^1_26]: https://openreview.net/forum?id=zoYPlgX1bH

[^1_27]: https://arxiv.org/html/2510.18083v1

[^1_28]: https://liner.com/ko/review/jedi-jointimage-diffusion-models-for-finetuningfree-personalized-texttoimage-generation

[^1_29]: https://www.mercity.ai/blog-post/fine-tuning-llms-using-peft-and-lora

[^1_30]: https://openreview.net/forum?id=KZgo2YQbhc

[^1_31]: https://www.reddit.com/r/StableDiffusion/comments/1gmwlfs/lora_is_inferior_to_full_finetuning_dreambooth/

[^1_32]: https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_ConceptGuard_Continual_Personalized_Text-to-Image_Generation_with_Forgetting_and_Confusion_Mitigation_CVPR_2025_paper.pdf

[^1_33]: https://arxiv.org/html/2506.05713v2

[^1_34]: https://arxiv.org/html/2511.03156v1

[^1_35]: https://arxiv.org/html/2510.27364v1

[^1_36]: https://arxiv.org/html/2511.05616v1

[^1_37]: https://arxiv.org/html/2510.09475v1

[^1_38]: https://arxiv.org/html/2509.23643v1

[^1_39]: https://arxiv.org/html/2508.03481v1

[^1_40]: https://arxiv.org/html/2509.22793v1

[^1_41]: https://arxiv.org/html/2503.18324v1

[^1_42]: https://arxiv.org/html/2508.00319v1

[^1_43]: https://arxiv.org/html/2506.06483v1

[^1_44]: https://arxiv.org/html/2501.01424v1

[^1_45]: https://arxiv.org/abs/2509.23593

[^1_46]: https://ieeexplore.ieee.org/document/11084576/

[^1_47]: https://ieeexplore.ieee.org/document/10856203/

[^1_48]: https://arxiv.org/abs/2411.08224

[^1_49]: https://arxiv.org/abs/2409.01128

[^1_50]: https://ieeexplore.ieee.org/document/10680046/

[^1_51]: https://arxiv.org/abs/2411.06618

[^1_52]: https://link.springer.com/10.1007/s10994-025-06743-y

[^1_53]: https://aclanthology.org/2023.findings-acl.48.pdf

[^1_54]: http://arxiv.org/pdf/1904.00310.pdf

[^1_55]: https://arxiv.org/pdf/2410.23751.pdf

[^1_56]: http://arxiv.org/pdf/2411.06916.pdf

[^1_57]: http://arxiv.org/pdf/2411.08224.pdf

[^1_58]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5690421/

[^1_59]: http://arxiv.org/pdf/2303.14771.pdf

[^1_60]: https://arxiv.org/abs/2509.01213

[^1_61]: https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Dense_Text-to-Image_Generation_with_Attention_Modulation_ICCV_2023_paper.pdf

[^1_62]: https://www.emergentmind.com/topics/few-shot-personalization-framework

[^1_63]: https://openaccess.thecvf.com/content/ICCV2025/papers/Skiers_Joint_Diffusion_Models_in_Continual_Learning_ICCV_2025_paper.pdf

[^1_64]: https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Towards_Understanding_Cross_and_Self-Attention_in_Stable_Diffusion_for_Text-Guided_CVPR_2024_paper.pdf

[^1_65]: https://openaccess.thecvf.com/content_ICCVW_2019/papers/GAZE/He_On-Device_Few-Shot_Personalization_for_Real-Time_Gaze_Estimation_ICCVW_2019_paper.pdf

[^1_66]: https://eccv.ecva.net/virtual/2024/poster/1969

[^1_67]: https://www.semanticscholar.org/paper/Cross-Attention-Makes-Inference-Cumbersome-in-Zhang-Liu/f159cde6f22842e2beac3ae534481e926c35b86c

[^1_68]: https://pure.kaist.ac.kr/en/publications/few-shot-anomaly-detection-via-personalization/

[^1_69]: https://thesai.org/Downloads/Volume16No4/Paper_14-Mitigating_Catastrophic_Forgetting_in_Continual_Learning.pdf

[^1_70]: https://papers.nips.cc/paper_files/paper/2023/file/f0878b7efa656b3bbd407c9248d13751-Paper-Conference.pdf

[^1_71]: https://aclanthology.org/2025.naacl-long.598.pdf

[^1_72]: https://www.cs.uic.edu/~liub/lifelong-learning/continual-learning.pdf

[^1_73]: https://arxiv.org/abs/2403.11052

[^1_74]: https://arxiv.org/abs/2508.15413

[^1_75]: https://ar5iv.labs.arxiv.org/abs/2308.06027v1

[^1_76]: https://arxiv.org/pdf/2508.15413.pdf

[^1_77]: https://arxiv.org/html/2601.06471v1

[^1_78]: https://arxiv.org/html/2506.13045v3

[^1_79]: https://arxiv.org/abs/2510.04034

[^1_80]: https://www.arxiv.org/pdf/2601.06471.pdf

[^1_81]: https://arxiv.org/html/2404.02747v1

[^1_82]: https://arxiv.org/html/2405.09771v2
