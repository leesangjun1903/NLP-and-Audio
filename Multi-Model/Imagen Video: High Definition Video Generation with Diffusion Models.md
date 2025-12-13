
# Imagen Video: High Definition Video Generation with Diffusion Models

## 1. 핵심 주장과 주요 기여도 요약

**Imagen Video**는 텍스트 프롬프트로부터 고해상도 비디오를 생성하는 시스템으로, 기존 텍스트-이미지 확산 모델(Diffusion Models)의 성공을 비디오 도메인으로 확장하는 획기적 연구입니다. 본 논문의 핵심 기여는 다음과 같습니다:[1]

**주요 기여도:**
1. **캐스케이드 구조의 효과성**: 저해상도에서 시작하여 순차적으로 해상도를 높이는 7개의 계층화된 확산 모델 파이프라인 개발[1]
2. **비디오-이미지 결합 학습**: 이미지와 비디오 데이터를 동시에 학습하여 모델 성능 향상 입증[1]
3. **v-파라미터화 도입**: 고해상도 생성에서 색상 변이 문제를 해결하고 수치적 안정성 개선[1]
4. **프로그레시브 증류 적용**: 샘플링 단계를 8단계로 단축하면서 품질 유지[1]

논문의 궁극적 성과는 **128프레임, 1280×768 해상도, 24fps의 고정의도 비디오**(약 5.3초)를 생성하며, 이는 당시 64프레임 128×128 비디오 생성 수준에서 약 **126배의 화소 증가**를 달성한 것입니다.[1]

***

## 2. 문제 정의, 방법론, 모델 구조 및 성능

### 2.1 해결하고자 하는 문제

텍스트-비디오 생성 분야의 근본적 어려움은 다음 세 가지입니다:[1]

1. **시간적 일관성(Temporal Coherence)**: 프레임 간 자연스러운 움직임과 객체 지속성 유지
2. **고해상도 생성의 계산 복잡성**: 고해상도 비디오는 이미지 대비 매우 높은 차원의 데이터
3. **제한된 비디오-텍스트 쌍 데이터**: 대규모 텍스트-비디오 학습 데이터 부족

### 2.2 제안하는 방법론 및 수식

#### 기본 확산 모델 프레임워크

Imagen Video는 **연속 시간 확산 모델(Continuous-time diffusion models)**을 기반으로 하며, Kingma et al. (2021)의 공식을 따릅니다.[1]

**전방 과정(Forward Process):**

$$q(z_t|x) = \mathcal{N}(z_t | \alpha_t x, \sigma_t^2 I)$$

$$q(z_t|z_s) = \mathcal{N}(z_t | \alpha_{ts} z_s, \sigma_{ts}^2 I)$$

여기서 $\(0 \leq s \leq t \leq 1\)$ 이고, $\(\sigma_{ts}^2 = 1 - e^{t-s}\alpha_t^2/\alpha_s^2\)$ 입니다.[1]

신호 대 잡음 비(SNR)는 $\(\lambda_t = \log \frac{\alpha_t^2}{\sigma_t^2}\)$ 로 정의되며, $\(t\)$ 가 증가하면서 단조 감소합니다.[1]

**역방 과정(Reverse Process)과 손실 함수:**

생성 모델은 이미지 예측을 통해 학습됩니다:

$$\mathcal{L}(x) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I), t \sim U(0,1)} \|\hat{x}_\theta(z_t, t) - x\|_2^2$$

여기서 $\(\hat{x}\_\theta(z_t, t) = \frac{z_t - \sigma_t \epsilon_\theta(z_t, t)}{\alpha_t}\)$ 입니다.[1]

**조건부 생성(Conditional Generation):**

텍스트 조건 \(c\)가 주어질 때:

$$\hat{x}_\theta(z_t, c, t)$$

**Classifier-Free 가이던스:**

샘플링 시간에 조건부와 무조건부 예측을 결합하여 생성 품질 향상:[1]

$$\hat{x}'_\theta(z_t, c) = (1+w)\hat{x}_\theta(z_t, c) - w\hat{x}_\theta(z_t, \emptyset)$$

여기서 $\(w\)$는 가이던스 강도(guidance weight)입니다.[1]

#### v-파라미터화(v-Prediction)

기존 $\(\epsilon\)$ -파라미터화 대신 v-파라미터화를 적용:[1]

$$v_t = \sigma_t \epsilon - \alpha_t x$$

이를 통해:
- 높은 해상도에서의 **색상 변이 아티팩트 제거**
- **더 빠른 수렴** (Figure 13에서 명확히 드러남)
- **수치적 안정성 향상**

### 2.3 모델 구조

#### 전체 파이프라인 아키텍처

Imagen Video는 **7개의 확산 모델로 구성된 계층적 캐스케이드** 구조를 채택합니다:[1]

```
입력 텍스트 → T5-XXL 인코더 → 베이스 모델 → SSR (3개) → TSR (3개)
                    ↓
            기본 영상 생성        공간 초해상화         시간 초해상화
            (16×40×24@3fps) → (32×80×48@6fps) → (128×1280×768@24fps)
```

각 모델의 사양:[1]

| 모델 | 입력 해상도 | 출력 해상도 | 매개변수 | 역할 |
|------|-----------|-----------|--------|------|
| T5-XXL | - | - | 4.6B | 텍스트 인코더 |
| 베이스 | 16×40×24 | 16×40×24 | 5.6B | 초기 비디오 생성 |
| SSR1 | 32×40×24 | 32×80×48 | 1.4B | 공간 초해상화 |
| SSR2 | 32×80×48 | 128×320×192 | 1.2B | 공간 초해상화 |
| SSR3 | 128×320×192 | 128×1280×768 | 340M | 최종 공간 초해상화 |
| TSR1 | 32×40×24 | 32×40×24 | 1.7B | 시간 초해상화 |
| TSR2 | 64×320×192 | 64×320×192 | 780M | 시간 초해상화 |
| TSR3 | 128×320×192 | 128×320×192 | 630M | 최종 시간 초해상화 |

**전체 매개변수**: 11.6B[1]

#### Video U-Net 아키텍처 (Figure 7)

기본 모델은 **공간-시간 분리 블록(Space-Time Separable Block)**을 활용합니다:[1]

```
입력 프레임
    ↓
[공간 합성곱] × N 프레임 (독립적)
    ↓
[공간 자기주의] × N 프레임 (독립적)
    ↓
[시간 자기주의] (프레임 간 혼합)
    ↓
[공간 합성곱]
    ↓
출력
```

**아키텍처 선택의 근거:**[1]

- **베이스 모델**: 시간적 자기주의(Temporal Self-Attention) 사용 → 장기 시간 의존성 모델링
- **SSR/TSR 모델**: 시간적 합성곱(Temporal Convolution) 사용 → 메모리/계산 효율성
- **높은 해상도 SSR**: 완전 합성곱(Fully Convolutional) 아키텍처 → 1280×768 생성 가능

#### 조건화 메커니즘

1. **T5-XXL 텍스트 인코더**: 동결된 상태로 사용 (Imagen의 성공 원칙 전수)[1]
2. **노이즈 조건화 증강(Noise Conditioning Augmentation)**: 슈퍼해상화 모델 학습 시[1]
   - 학습: 무작위 SNR로 조건 입력에 가우스 노이즈 추가
   - 샘플링: SNR=3 또는 5 고정값 사용
3. **채널 연결**: 입력 비디오를 업샘플링한 후 잡음 있는 데이터에 연결[1]

### 2.4 성능 향상 기법

#### 진동 가이던스(Oscillating Guidance)

큰 가이던스 가중치 사용 시 포화 아티팩트 해결:[1]

- **초기 샘플링 단계**: 고정된 높은 가중치 \(w=15\) (강한 텍스트 정렬)
- **중간-후기 단계**: \(w=15\)와 \(w=1\) 사이에서 진동 (포화 제거)
- **해상도 \(>80×48\)**: 효과 제한적이므로 미적용[1]

#### 프로그레시브 증류(Progressive Distillation)

원래 파이프라인 → 증류 파이프라인 성능 비교 (Table 1):[1]

| 구성 | CLIP 점수 | 샘플링 시간 | 속도 향상 |
|------|----------|----------|---------|
| 원래 (256+128 스텝) | 25.19 | 618초 | 1× |
| 완전 증류 (8+8 스텝) | 25.03 | 35초 | **17.7×** |
| FLOPs 감소 | - | - | **36×** |

**증류 과정:**[1]
1. **1단계**: DDIM 샘플러의 N-스텝을 N/2-스텝으로 증류
2. **2단계**: 모든 7개 모델 순차 증류
3. **샘플러**: 확률적 N-스텝 샘플러 사용 (DDIM 업데이트 + 노이즈 추가)

### 2.5 학습 전략

#### 비디오-이미지 결합 학습

논문의 핵심 전략 중 하나:[1]

**학습 데이터:**
- 14M 비디오-텍스트 쌍 (Google 내부 데이터)
- 60M 이미지-텍스트 쌍 (Google 내부)
- LAION-400M (공개 데이터)

**구현 방법:**[1]
- 개별 이미지를 단일 프레임 비디오로 처리
- 이미지 시퀀스를 패킹 (비디오 길이와 동일)
- **시간 합성곱 레지듀얼 블록**: 마스킹으로 계산 우회
- **교차 프레임 시간 자기주의**: 마스킹으로 비활성화

**효과:**[1]
- 더 크고 다양한 학습 데이터 사용 가능
- 이미지 스타일 학습 → 비디오에서 예술적 스타일 생성 가능 (Figure 8)
- Van Gogh 스타일, 수채화 스타일 등 역동성 생성

***

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 스케일링 분석 (Figure 11, 13)

모델 크기 확대에 따른 성능 변화를 정량적으로 평가:[1]

**매개변수 증가에 따른 효과:**

| 모델 크기 | FVD 점수 | CLIP 점수 | 수렴 특성 |
|----------|---------|---------|---------|
| 500M | ~24 | ~23.5 | 기준선 |
| 1.6B | ~21 | ~24 | 개선 |
| 5.6B | ~18 | ~24.5 | 최고 성능 |

**핵심 발견:**[1]
- FVD(Fréchet Video Distance)에서 **지속적 개선** (500M→5.6B)
- CLIP 점수도 순차적 향상
- **텍스트-이미지 결과와 상이**: Imagen(이미지)은 스케일링 이득이 제한적이었으나, 비디오는 포화되지 않음
- **해석**: 비디오 모델링이 더 어려운 과제이며, 향후 스케일링 효과 예상

### 3.2 v-파라미터화의 일반화 개선

#### 정성적 비교 (Figure 12)

**ε-파라미터화 문제점:**[1]
- 프레임 간 **글로벌 색상 변이**(color shifts) 발생
- 자연스럽지 않은 색상 불일치
- 시간적 일관성 급격히 감소

**v-파라미터화 장점:**[1]
- 일관된 색상 유지
- 높은 해상도에서 안정적

#### 정량적 비교 (Figure 13)

```
FID 점수 vs 학습 단계:
ε-파라미터화: 초기 ~30 → 느린 수렴 → ~15 (1.5M 단계)
v-파라미터화: 초기 ~25 → 빠른 수렴 → ~12 (0.5M 단계) ← 3배 빠름
```

**수치적 안정성 개선 메커니즘:**[1]
- v-공간에서 가이던스 계산 가능: $\(\hat{v}'(z_t, c) = (1+w)\hat{v}(z_t, c) - w\hat{v}(z_t, \emptyset)\)$
- ε-공간이나 x-공간과 달리 **범위 이탈 위험 감소**

### 3.3 텍스트 정렬 및 장면 이해

#### 분석된 일반화 능력

**3D 객체 이해 (Figure 9):**[1]
- 회전하는 객체의 구조 보존 능력
- 예: 빅토리아 시대 집, 초밥으로 만든 자동차, 종이 접기 코끼리
- 3D 일관성이 완벽하지는 않으나, 사전으로 작용 가능

**텍스트 렌더링 (Figure 10):**[1]
- 다양한 애니메이션 스타일의 텍스트 생성
- "Imagen Video"를 가을 낙엽, 붓 획, 빠른 붓질 등으로 표현
- 전통 도구로는 어려운 애니메이션 생성

**예술적 스타일 전이 (Figure 8):**[1]
- Van Gogh 스타일 비디오 (이미지 학습 데이터로부터)
- 수채화 숲 드론 비행
- 픽셀 아트 도시 (정지된 카메라 이동)

### 3.4 성능 메트릭 평가 (Table 1)

생성된 샘플의 품질 평가:[1]

**프롬프트**: "A teddy bear wearing sunglasses playing guitar next to a cactus"

| 모델 구성 | CLIP 점수 | CLIP R-Precision | 샘플링 시간 |
|----------|----------|-----------------|----------|
| 그라운드 트루스 | 24.27 | 86.18 | - |
| 원래 (w=6) | 25.19±0.03 | 92.1±2.53 | 618s |
| 증류 (w=6, 8+8) | 25.03±0.05 | 89.6±3.38 | 35s |
| 증류 (w=15-1, 8+8) | 25.12±0.07 | 90.9±1.46 | 35s |

**해석:**[1]
- **생성 샘플 > 그라운드 트루스**: Classifier-free 가이던스가 분포를 품질 메트릭 방향으로 이동
- **증류로 품질 유지**: 17.7배 속도 향상에도 불구하고 CLIP 점수 0.16만 감소 (< 1%)

***

## 4. 한계 및 제약 사항

### 4.1 기술적 한계

#### 시간적 일관성의 불완전성

논문에서 직접 언급하지 않으나, 4단계 슈퍼해상화 과정으로 인한 **누적 오류**가 가능합니다. 특히:
- TSR 모델이 로컬 시간 일관성에 의존 (시간 합성곱 사용)
- 장시간 의존성 모델링은 베이스 모델에 집중

#### 메모리 및 계산 비용

- **원래 파이프라인**: 618초/샘플 (고해상도에서는 더 많은 시간 필요)
- 전체 7개 모델 앙상블의 순차 실행 필요
- 고해상도 SSR 모델도 여전히 메모리 집약적

### 4.2 사회적 영향 및 윤리적 우려

논문의 섹션 4 (Limitations and Societal Impact)에서:[1]

**잠재적 오용:**
- 가짜, 증오성 또는 명시적 콘텐츠 생성
- 딥페이크 비디오 악용

**완화 조치:**
- 입력 텍스트 프롬프트 필터링
- 출력 비디오 콘텐츠 필터링

**미해결 과제:**[1]
- 학습 데이터의 **사회적 편향 및 고정관념** 존재
- 문제 있는 데이터로 학습된 T5-XXL 인코더
- 명시적/폭력적 콘텐츠는 필터링 가능하나, 사회적 편향은 감지/필터링 어려움

**공개 보류 결정:**[1]
> "우리는 이러한 우려가 완화될 때까지 Imagen Video 모델이나 소스 코드를 공개하지 않기로 결정했습니다."

***

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 학술적 영향

#### 근본적 기여

1. **캐스케이드 확산의 비디오 도메인 확대**: Imagen(이미지) 성공의 직접 확장으로, **확산 모델의 스케일링 가능성** 재증명[1]

2. **비디오-이미지 결합 학습의 효과성**:[1]
   - 제한된 비디오-텍스트 쌍 데이터 문제의 우아한 해결책
   - 이후 연구 (Make-A-Video, Sora 등)의 표준 관행으로 확산

3. **v-파라미터화의 일반적 타당성**:[1]
   - 이미지 생성에서 한정적 효과 → 비디오에서 핵심 역할 입증
   - 고해상도 생성에서 필수 기법 확립

### 5.2 후속 연구와의 비교 (2023-2025)

#### Make-A-Video (Meta AI, 2022년 9월)[2]

| 특성 | Imagen Video | Make-A-Video |
|------|-------------|--------------|
| 텍스트 인코더 | T5-XXL (4.6B) | CLIP 기반 |
| 아키텍처 | Video U-Net (3D 분리) | 공간-시간 분해 |
| 학습 데이터 | 텍스트-이미지 + 텍스트-비디오 | 텍스트-이미지만 (비디오는 무조건부) |
| 해상도 | 1280×768 | 제한적 |
| 프레임 수 | 128프레임 (5.3초) | 상대적으로 짧음 |
| 특징 | v-파라미터화, 프로그레시브 증류 | 라우팅 경로 (Routing Paths) |

**Imagen Video의 우월성:**[2]
- 더 높은 해상도와 길이
- 텍스트 정렬 우수성 (T5-XXL 사용)
- 계산 효율성 (증류)

***

#### Sora (OpenAI, 2024년 2월)[3][4]

| 특성 | Imagen Video | Sora |
|------|-------------|------|
| 기본 구조 | Video U-Net (합성곱) | Diffusion Transformer (DiT) |
| 패치 표현 | - | 공간-시간 패치 |
| 최대 길이 | 128프레임 (5.3초) | 60초+ |
| 해상도 | 1280×768 | 변수 (1920×1080+) |
| 스케일링 | 제한적 표시 | Transformer의 우수한 스케일링 |
| 핵심 혁신 | 캐스케이드 + v-파라미터화 | DiT + 공간-시간 패치 |

**Sora의 진보:**[4][3]
- **Transformer 아키텍처**: 합성곱 기반 U-Net보다 우수한 확장성
- **통합 아키텍처**: 7개 모델 대신 단일 모델 활용 → 장비 최적화
- **공간-시간 패치**: 더 효율적인 토큰화

***

#### CogVideoX (Tsinghua University, 2024년 8월)[5]

| 특성 | Imagen Video | CogVideoX |
|------|-------------|-----------|
| 기본 구조 | 비디오 U-Net | Diffusion Transformer |
| 해상도 | 1280×768 | 768×1360 (5:9) |
| 프레임 수 | 128 (5.3초) | 169 (10.6초) |
| 아키텍처 | 공간-시간 분리 | 전문가 트랜스포머 (Expert) |
| 핵심 기술 | 캐스케이드 | 3D Causal VAE |

**Open-Sora (2024년 12월)[6][7]

|특성 | Imagen Video | Open-Sora |
|------|-------------|-----------|
| 아키텍처 | Video U-Net | Spatial-Temporal DiT (STDiT) |
| 최대 길이 | 128프레임 | 180프레임 (15초) |
| 최대 해상도 | 1280×768 | 720p (임의 종횡비) |
| 오픈소스 | 비공개 | 전체 공개 (가중치 포함) |
| 효율성 | 계산 비용 높음 | STDiT: 효율적 주의 분리 |

**STDiT의 효율성:[7][6]
- 공간과 시간 주의 **분리**: $\(\text{Attention}\_{\text{spatial}} + \text{Attention}_{\text{temporal}}\)$
- 계산 복잡도 감소: $\(O(HWT) \rightarrow O(HW + T)\)$ (근사)

***

### 5.3 미해결 연구 질문 및 향후 방향

#### 1. 일반화 성능의 한계

**문제:**
- Imagen Video는 **특정 스타일과 객체에 편향**될 수 있음 (학습 데이터 분포)
- 장시간 비디오 생성에서 **누적 오류 증가**

**해결책 (최신 연구):[8][9][10]
- **FluxFlow (2025)**: 시간적 정규화로 시간적 일관성 및 다양성 향상[10]
- **USV (Unified Sparsification, 2025)**: 22.7배 end-to-end 가속화[11]
- **StreamingT2V**: 장시간 비디오의 일관성 유지[12]

#### 2. 시간적 일관성 향상

**현재 한계:**[10]
- 프레임 간 깜빡임(flicker)
- 과도하게 단순화된 시간 역학 (temporal dynamics)

**최신 접근법:[13][10]
- **시간적 증강(Temporal Augmentation)**: 학습 시 제어된 시간적 섭동
- **시간적 일관성 가이던스(TCG, 2025)**: 이전-현재 프레임 특성 간 상호작용[13]

#### 3. 계산 효율성

**원본 Imagen Video의 문제:**[1]
- 원래 파이프라인: 618초/샘플
- 증류: 35초 (여전히 실시간 이상)

**최신 솔루션:[14][15][11]
- **USV (2025)**: 22.7배 end-to-end 가속화[11]
- **StreamDiT (2025)**: 실시간 스트리밍 생성[14]
- **On-device Sora (2025)**: 모바일 기기에서 4-8초 만에 49프레임 생성[15]

#### 4. 모델 확장성

**문제:** Imagen Video는 U-Net 기반으로 인해 **확장성 한계** 명시[1]

**해결책 (Sora, Open-Sora):[6][3]
- Transformer 기반 구조로 전환
- **로그 스케일 이득**: 매개변수 증가에 따른 성능 개선이 더 오래 지속

#### 5. 도메인 간 일반화

**현재 한계:**
- 학습 데이터의 편향 (자연 영상 위주)
- 희귀하거나 창의적 시나리오에서 성능 저하

**향후 연구 방향:**[9][8]
- **다중 작업 학습**: 비디오 생성 + 편집 + 추론
- **도메인 적응**: 특정 산업(의료, 애니메이션)으로의 전이 학습
- **제어 가능한 생성**: 카메라 이동, 객체 추적 등 정밀한 제어[16][17]

***

### 5.4 산업적 영향 및 실무 고려 사항

#### 적용 가능 분야

1. **콘텐츠 창작**[18]
   - 스토리보드 시각화
   - 영화/광고 사전 제작
   - 유튜브/소셜 미디어 콘텐츠

2. **의료/교육**[19]
   - 의료 시뮬레이션 (예: FFA Sora - 망막 혈관조영술)[20]
   - 교육용 시각화

3. **3D 콘텐츠 생성**[19]
   - VFusion3D: 비디오 확산 모델로부터 3D 자산 생성
   - 합성 다시점(multi-view) 데이터 생성 → 3D 모델 학습

#### 구현 시 고려사항

**메모리:**
- 원본 7개 모델: 11.6B 매개변수
- 고해상도 생성: GPU 메모리 부족 가능성
- **권장**: 다중 GPU 또는 V100+/A100+

**비용:**
- 학습: 논문에서 공개하지 않음
- 추론: 증류 후 35초/샘플 (TPU/GPU 비용 고려)

**윤리:**
- 데이터 필터링 필수 (편향 제거)
- 출력 콘텐츠 검증 프로세스 구축
- 사용 약관 명확히

***

## 6. 결론 및 종합 평가

### 핵심 성과

**Imagen Video**는 2022년 10월 발표 당시 **최고 수준의 텍스트-비디오 생성 모델**이었으며, 다음을 입증했습니다:[1]

1. **확산 모델의 확장성**: 이미지 생성 기술을 비디오로 우아하게 확대 가능
2. **효율적 학습 전략**: 비디오-이미지 결합 학습으로 데이터 부족 극복
3. **기술적 혁신**: v-파라미터화, 프로그레시브 증류로 품질-효율 트레이드오프 해결

### 연구 지형의 진화

| 연도 | 주요 모델 | 핵심 아키텍처 | 주요 진보 |
|------|----------|-----------|---------|
| 2022-10 | **Imagen Video** | Video U-Net + 캐스케이드 | v-파라미터화, 시간적 초해상화 |
| 2022-09 | Make-A-Video | 공간-시간 분해 | 비디오-이미지 분리 학습 |
| 2024-02 | Sora | Diffusion Transformer | 통합 모델, 우수한 스케일링 |
| 2024-08 | CogVideoX | Expert Transformer | 더 긴 비디오 (10초+) |
| 2024-12 | Open-Sora | STDiT | 오픈소스화, 효율성 |
| 2025-03 | Open-Sora 2.0 | Hybrid MM-DiT | 상업 수준 모델, $200k 학습 비용 |
| 2025+ | 진행 중 | 향상된 DiT + 최적화 | 실시간 생성, 모바일 배포 |

### 학문적 의의

Imagen Video가 후속 연구에 제시한 **프레임워크**:

1. **확산 기반 생성의 정당성**: GAN 대비 확산 모델의 우월성 재확인
2. **캐스케이드 구조의 가치**: 고차원 문제의 **계층적 해결** 원칙 제시
3. **텍스트 조건화의 중요성**: 강력한 언어 모델(T5-XXL)의 필수성

### 앞으로의 방향성

**단기 (1-2년):**
- 실시간 생성(< 1초) 달성
- 모바일/엣지 디바이스 배포
- 더 길어진 비디오(> 1분)

**중기 (2-5년):**
- 정밀한 사용자 제어 (카메라, 객체 추적)
- 멀티모달 입력 (이미지+텍스트+음성)
- 3D 일관성 보장

**장기 (5년+):**
- **세계 시뮬레이터(World Simulators)**: 물리법칙 준수하는 장시간 비디오
- 추론 능력 통합: 인과관계 이해, 논리적 일관성
- 윤리적 생성 표준 확립

***

## 참고 문헌 (핵심 인용 출처)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ba642d66-4c2f-4559-95e4-5449aaa37c95/2210.02303v1.pdf)
[2](https://www.youtube.com/watch?v=AcvmyqGgMh8)
[3](https://openai.com/index/sora/)
[4](https://openai.com/index/video-generation-models-as-world-simulators/)
[5](http://arxiv.org/pdf/2408.06072.pdf)
[6](https://arxiv.org/abs/2412.20404)
[7](http://arxiv.org/pdf/2412.20404.pdf)
[8](https://arxiv.org/html/2410.20502v1)
[9](https://arxiv.org/abs/2507.22360)
[10](https://arxiv.org/html/2503.15417v1)
[11](https://arxiv.org/html/2512.05754)
[12](http://arxiv.org/pdf/2403.14773.pdf)
[13](https://arxiv.org/html/2512.07480v1)
[14](https://arxiv.org/html/2507.03745v3)
[15](https://arxiv.org/html/2502.04363v1)
[16](https://ieeexplore.ieee.org/document/11094661/)
[17](https://arxiv.org/abs/2412.01429)
[18](https://dl.acm.org/doi/10.1145/3707292.3707367)
[19](https://arxiv.org/abs/2403.12034)
[20](https://arxiv.org/abs/2412.17346)
[21](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[22](https://open-publishing.org/publications/index.php/APUB/article/view/2769)
[23](http://arxiv.org/pdf/2305.13840v1.pdf)
[24](http://arxiv.org/pdf/2408.12590.pdf)
[25](http://arxiv.org/pdf/2406.02230.pdf)
[26](https://arxiv.org/abs/2305.18264)
[27](http://arxiv.org/pdf/2406.04277.pdf)
[28](https://en.wikipedia.org/wiki/Text-to-video_model)
[29](https://openaccess.thecvf.com/content/ICCV2023/papers/Khachatryan_Text2Video-Zero_Text-to-Image_Diffusion_Models_are_Zero-Shot_Video_Generators_ICCV_2023_paper.pdf)
[30](https://arxiv.org/html/2502.17863v1)
[31](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
[32](https://ai.meta.com/research/publications/vfusion3d-learning-scalable-3d-generative-models-from-video-diffusion-models/)
[33](https://openreview.net/forum?id=qsffecsbJg)
[34](https://arxiv.org/abs/2408.06072)
[35](https://arxiv.org/html/2412.18688v2)
[36](https://arxiv.org/abs/2411.17470)
[37](https://arxiv.org/html/2505.07652v1)
[38](https://arxiv.org/html/2512.06905v1)
[39](https://arxiv.org/html/2511.00107v1)
[40](https://arxiv.org/html/2503.04606v1)
[41](https://github.com/showlab/Awesome-Video-Diffusion)
[42](https://ieeexplore.ieee.org/document/11093584/)
[43](https://arxiv.org/abs/2405.18326)
[44](https://arxiv.org/abs/2408.12590)
[45](https://arxiv.org/abs/2406.07686)
[46](https://arxiv.org/abs/2406.02540)
[47](https://arxiv.org/abs/2409.01595)
[48](https://arxiv.org/abs/2503.19881)
[49](http://arxiv.org/pdf/2406.07686.pdf)
[50](https://arxiv.org/pdf/2402.17177v1.pdf)
[51](https://arxiv.org/abs/2503.23796)
[52](https://arxiv.org/html/2502.04847v1)
[53](http://arxiv.org/pdf/2405.17405.pdf)
[54](https://blog.metaphysic.ai/native-temporal-consistency-in-stable-diffusion-videos-with-tokenflow/)
[55](https://en.wikipedia.org/wiki/Sora_(text-to-video_model))
[56](https://www.louisbouchard.ai/make-a-video/)
[57](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Varghese_Unsupervised_Temporal_Consistency_Metric_for_Video_Segmentation_in_Highly-Automated_Driving_CVPRW_2020_paper.pdf)
[58](https://www.youtube.com/watch?v=fWUwDEi1qlA)
[59](https://ai.meta.com/blog/generative-ai-text-to-video/)
[60](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Upscale-A-Video_Temporal-Consistent_Diffusion_Model_for_Real-World_Video_Super-Resolution_CVPR_2024_paper.pdf)
[61](https://arxiv.org/html/2507.13343v1)
[62](https://arxiv.org/html/2403.12042v2)
[63](https://arxiv.org/html/2502.04363v2)
[64](https://ar5iv.labs.arxiv.org/html/2209.14792)
[65](https://arxiv.org/html/2508.00144v1)
[66](https://arxiv.org/html/2503.09642v1)
[67](https://arxiv.org/html/2504.05298v1)
[68](https://arxiv.org/html/2406.01493v3)
[69](https://arxiv.org/html/2412.20404v1)
