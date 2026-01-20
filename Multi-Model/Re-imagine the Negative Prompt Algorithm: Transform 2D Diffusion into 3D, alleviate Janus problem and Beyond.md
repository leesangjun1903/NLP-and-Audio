
# Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus problem and Beyond
## 요약
"Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus problem and Beyond"는 텍스트-이미지 diffusion 모델의 부정 프롬프트 알고리즘을 재설계하여 2D 이미지 생성과 3D 객체 생성의 품질을 동시에 향상시키는 논문입니다. Armandpour 등이 2023년 발표한 이 연구의 핵심은 기하학적 성질을 활용한 Perp-Neg 샘플러로, 긍정 프롬프트와 부정 프롬프트의 semantic overlap 문제를 해결하면서 3D 생성의 Janus 문제를 완화합니다. [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

## 1. 핵심 주장과 기여
### 문제의 정의
텍스트-이미지 diffusion 모델은 훈련 데이터의 높은 빈도 텍스트-이미지 쌍에 편향되어 있어, 요청된 텍스트와 정확히 맞는 이미지를 생성하지 못합니다. 예를 들어 "왕관을 쓴 귀여운 코기"라는 프롬프트에서 왕관이 누락되거나 불원한 속성이 포함될 수 있습니다. 기존의 부정 프롬프트 방식을 사용해도, 부정 프롬프트(예: "왕관")가 긍정 프롬프트(예: "코기")의 개념과 겹칠 때 메인 개념 자체가 제거되는 문제가 발생합니다. [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

### 주요 기여
1. **이론적 분석**: 기존 composing diffusion의 한계를 semantic overlap 관점에서 수학적으로 규명 [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
2. **Perp-Neg 알고리즘**: 기하학적 투영 기반의 새로운 샘플링 방식으로, 학습이나 파인튜닝 없이 기존 모델에 직접 적용 가능 [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
3. **2D에서 3D로의 확장**: View conditioning의 개선을 통해 2D 이미지 생성의 view fidelity를 73.1% 향상시켜, 3D DreamFusion의 Janus 다중면 문제를 완화 [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

## 2. 제안하는 방법 (수식 포함)
### 2.1 기존 부정 프롬프트의 문제
기존 composing diffusion 모델은 다음과 같이 표현됩니다: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

$$p(x|c_1, c_2) = p(x) \frac{p(c_1|x) p(c_2|x)}{p(c_1, c_2|x)}$$

가중치 파라미터를 추가한 composing noise predictor는: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

$$\hat{\epsilon}(x_t, t, c) = \hat{\epsilon}(x_t, t) + \sum_{i=1}^{n} w_i \left[\hat{\epsilon}(x_t, t, c_i) - \hat{\epsilon}(x_t, t)\right]$$

부정 프롬프트 처리 시, 기존 방식: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

$$\hat{\epsilon}(x_t, t, c) = \hat{\epsilon}(x_t, t) - w_{neg} \left[\hat{\epsilon}(x_t, t, c_{neg}) - \hat{\epsilon}(x_t, t)\right]$$

이 방식은 긍정 프롬프트($c_{pos}$)와 부정 프롬프트($c_{neg}$)가 겹칠 때 다음과 같은 문제가 발생합니다:

$$\text{If } \nabla_{neg} \approx \nabla_{pos}, \text{ then } \hat{\epsilon} \approx 0$$

즉, 부정 프롬프트의 denoising score가 긍정 프롬프트와 유사하면 전체 신호가 소실됩니다.

### 2.2 Perp-Neg 샘플러
이 문제를 해결하기 위해 Perp-Neg는 부정 프롬프트의 gradient를 긍정 프롬프트에 대해 수직인 성분으로만 제한합니다. [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

**2개 프롬프트의 경우:**

$$\epsilon_{Perp} = \epsilon(x_t, t, c) - w_1 w_2 \frac{\langle \nabla_2, \nabla_1 \rangle}{\|\nabla_1\|^2} \nabla_1$$

여기서: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
- $\nabla_1 = \epsilon(x_t, t, c_{pos}) - \epsilon(x_t, t)$: 긍정 프롬프트 성분
- $\nabla_2 = \epsilon(x_t, t, c_{neg}) - \epsilon(x_t, t)$: 부정 프롬프트 성분
- $\langle \cdot, \cdot \rangle$: 벡터 내적
- 첫 번째 항은 긍정 프롬프트에 기여하는 부정 성분

**일반화된 형태 (다중 부정 프롬프트):** [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

$$\epsilon_{Perp-Neg} = \epsilon(x_t, t, c_{pos}) - \sum_{i=1}^{m} w_i \left[\nabla_{neg_i} - \text{proj}_{\nabla_{pos}}(\nabla_{neg_i})\right]$$

여기서 투영 연산자:

$$\text{proj}_{\nabla_{pos}}(\nabla_{neg_i}) = \frac{\langle \nabla_{neg_i}, \nabla_{pos} \rangle}{\|\nabla_{pos}\|^2} \nabla_{pos}$$

### 2.3 기하학적 직관
Perp-Neg의 핵심은 score space에서의 기하학적 해석입니다. 긍정 프롬프트의 denoising 방향을 기준(basis)으로 설정하고, 부정 프롬프트의 영향을 이 기준에 대해 수직인 방향으로만 제한합니다. 이렇게 하면: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
- 긍정 프롬프트의 의미는 보존됨
- 부정 프롬프트는 원하지 않는 속성만 제거
- 겹치는 부분에서도 안전하게 작동

### 2.4 3D 생성으로의 확장
#### View Conditioning 전략

특정 각도(예: side view)를 생성하기 위해 다음과 같이 설계합니다: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

**Side view 생성:**
- 긍정: "side view"
- 부정: "front view", "back view"

**Back view 생성:**
- 긍정: "back view"  
- 부정: "side view", "front view"

#### Perp-Neg와 SDS의 통합

DreamFusion과 결합하기 위해 Score Distillation Sampling (SDS) 손실을 다음과 같이 수정합니다: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

$$\mathcal{L}_{PN-SDS} = \mathbb{E}_{t, \epsilon} \left[w(t) \left\|\epsilon_{PN}(x_t|c,v,t) - \epsilon\right\|_2^2\right]$$

여기서 $\epsilon_{PN}$은: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

$$\epsilon_{PN}(x_t|c,v,t) = \epsilon_{unc} - w_{guidance}[\epsilon_{pos}^{(v)} - \epsilon_{pos}^{(v)} \perp \epsilon_{neg_i}^{(v)}]$$

- $\epsilon_{unc}$: 무조건부 prediction
- $\epsilon_{pos}^{(v)}$: 특정 view v에서 긍정 프롬프트의 예측
- $\epsilon_{neg_i}^{(v)}$: 부정 프롬프트에 수직인 성분

## 3. 모델 구조
### 3.1 2D Generation Pipeline
```
Text Prompt (Positive) → Text Encoder → CLIP Embedding
                                              ↓
                                        Denoising U-Net
                                              ↑
                                        (Perp-Neg 모듈)
                                              ↑
Text Prompt (Negative) → Text Encoder → CLIP Embedding
```

### 3.2 3D Generation Pipeline (DreamFusion 통합)
```
Text Prompt + View → Perp-Neg 샘플러 → Denoised prediction
                            ↓
                      NeRF Renderer
                            ↓
                    Rendered Image
                            ↓
                      Loss 계산 & 역전파
                            ↓
                      NeRF 파라미터 업데이트
```

### 3.3 알고리즘 구현
알고리즘 1에서 각 timestep t에서의 Perp-Neg 연산: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

```
def PerpNeg(noise_pred_main, noise_pred_neg, weights):
    # 투영 성분 계산
    for each neg_prompt:
        proj = dot(noise_pred_neg, noise_pred_main) / norm(noise_pred_main)^2
        perpendicular = noise_pred_neg - proj * noise_pred_main
        noise_pred_main -= weight * perpendicular
    return noise_pred_main
```

## 4. 성능 향상
### 4.1 2D View Generation 성능
**정량적 평가 (표 1):** [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

| 메서드 | Side View | Back View |
|--------|----------|----------|
| Stable Diffusion | 42.0% | 14.6% |
| CEBM | 12.7% | 2.0% |
| **Perp-Neg** | **73.1%** | **40.4%** |

Perp-Neg는 side view에서 73.1% 성공률을 달성하여 기존 방식 대비 **73% 개선**, back view에서는 40.4%로 **177% 개선**을 보여줍니다. [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

### 4.2 부정 프롬프트 조합 분석
view 생성 성능은 부정 프롬프트 선택에 따라 달라집니다: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
- Side view 생성 시, front view만 부정하기보다는 front view + back view 함께 사용이 더 효과적
- Back view 생성 시, side view 같은 '모호한' 관점을 부정하는 것이 효과적
- 모델이 덜 생성하는 관점(back view)을 부정하는 것이 더 효율적

### 4.3 3D Generation의 Janus 문제 완화
**정성적 결과 (표 2):** [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
- "corgi standing": 0회 → 2회 성공
- "westie": 0회 → 5회 성공  
- "lion": 0회 → 1회 성공
- "Lamborghini": 4회 → 5회 성공
- "cute pig": 0회 → 0회 (어려운 카테고리)
- "Super Mario": 2회 → 4회 성공

Perp-Neg 적용 후 대부분의 프롬프트에서 Janus 문제 없는 3D 생성 성공 사례가 증가합니다. [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

## 5. 한계
### 5.1 프롬프트 설계 의존성
부정 프롬프트의 효과는 정확한 설계에 크게 의존합니다: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
- 부정 가중치 범위: -5 ~ -0.5 (일반적 설정)
- 객체 및 view별로 매개변수 조정 필요
- 자동화된 방법이 없어 수동 조정 부담

### 5.2 기하학적 한계
투영 기반 접근의 한계: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
- 극단적으로 드문 객체(rare categories): 여전히 어려움
- Circular objects: 실험에서 제외됨 (데이터 편향 극심)
- 완전한 3D 일관성 보장 불가능

### 5.3 계산 효율성
추가 연산 오버헤드: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
- 각 timestep에서 추가 projection 계산
- 여러 부정 프롬프트 처리 시 선형 증가
- 실제 성능 상 무시할 수준이지만 이론적 분석 부재

### 5.4 일반화 제한
특정 도메인의 한계: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
- 복합 장면(multi-object): 단일 객체보다 어려움
- 극도로 구체화된 장면: 프롬프트 과결정 가능성
- Text-image alignment가 애초에 나쁜 경우: 부정 프롬프트로도 해결 불가

## 6. 모델의 일반화 성능 향상 가능성
### 6.1 현재 강점
**내재적 일반화 능력:** [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
- 학습 불필요: 사전학습된 모델에 즉시 적용 가능 (zero-shot)
- 아키텍처 불변: Stable Diffusion, Imagen, DALL-E 등 다양한 모델 호환
- 도메인 독립: 동물, 건축물, 추상 개념 등 광범위한 적용 가능

**수학적 견고성:** [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
- 기하학적 원리에 기반한 이론적 정당성
- 부정 프롬프트와 긍정 프롬프트의 관계를 원리적으로 처리

### 6.2 일반화 강화 방법
**1. 자동 부정 프롬프트 생성**

ANSWER나 DNP 같은 방법과 결합: [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Desai_Guiding_Diffusion_Models_with_Adaptive_Negative_Sampling_Without_External_Resources_ICCV_2025_paper.pdf)
- 모델 자체에서 최적 부정 프롬프트 자동 도출
- 각 단계별로 동적 조정
- Perp-Neg의 프롬프트 의존성 해결

**2. 적응형 가중치 학습**

메타러닝 접근:
- 객체 카테고리별 최적 $w_{neg}$ 자동 학습
- view별 가중치 적응화
- 프롬프트 특성에 따른 동적 조정

**3. Multi-view Diffusion과의 통합**

MVDream이나 OrientDream과 결합: [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10887623/)
- MVDream의 강력한 view consistency + Perp-Neg의 부정 프롬프트 효율성
- 이론적 보완성: MVDream은 학습 기반, Perp-Neg는 training-free
- 하이브리드 접근으로 양쪽의 장점 활용

**4. 3D Awareness 추가**

3DFuse 유사 접근: [ku-cvlab.github](https://ku-cvlab.github.io/3DFuse/)
- View-specific depth map으로 3D 구조 정보 제공
- 부정 프롬프트의 효과를 기하학적 제약으로 강화
- Semantic consistency 보존

### 6.3 예상 성능 향상
조합적 접근의 이론적 효과:
- 자동 부정 프롬프트 + Perp-Neg: 프롬프트 설계 부담 제거 + 기하학적 정확성
- Perp-Neg + MVDream: 단일 view bias 해결 + multi-view consistency
- Perp-Neg + VSD: Janus 문제 완화 + 다양성 증대

## 7. 최신 연구와의 비교 (2020년 이후)
### 7.1 Text-to-3D 생성 방법론 비교
#### DreamFusion (2022)
**핵심 기여**: NeRF + Score Distillation Sampling [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10203601/)
- 3D 훈련 데이터 불필요 (혁명적)
- 사전학습된 2D diffusion 모델 활용
- **문제**: Janus 문제 심각, 최적화 느림 (1.5시간/객체)
- **성능**: CLIP R-Precision 0.67

#### Magic3D (2023)
**개선**: 2단계 최적화 프레임워크 [arxiv](https://arxiv.org/abs/2211.10440)
- Stage 1: Low-res coarse model (sparse hash grid)
- Stage 2: High-res mesh 파인튜닝
- **성능**: DreamFusion 대비 2배 빠름 (40분), 해상도 향상
- **평가**: 사용자 선호도 61.7% (DreamFusion 대비)

#### ProlificDreamer (2023)
**혁신**: Variational Score Distillation (VSD) [arxiv](https://arxiv.org/abs/2305.16213)
- 3D 파라미터를 확률 변수로 모델링 (SDS와 차이)
- 과포화, 과평활화, 낮은 다양성 문제 해결
- **성능**: 512×512 고해상도, CLIP R-Precision 0.83

#### MVDream (2023)
**전환점**: Multi-view Diffusion Model [youtube](https://www.youtube.com/watch?v=6HjTgDL97hc)
- 3D 데이터셋으로 훈련한 별도의 multi-view 모델
- 3D attention layer로 cross-view 의존성 학습
- **효과**: 3D consistency 획기적 개선, Janus 문제 크게 완화
- **트레이드오프**: 추가 훈련 필요, 3D 데이터 의존

#### **Perp-Neg (2023)**
**차별성**: 기하학적 부정 프롬프트 재설계 [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
- Training-free (가장 큰 장점)
- 부정/긍정 프롬프트 semantic overlap 해결
- View fidelity 향상 (side view 73.1%, back view 40.4%)
- **제한**: 프롬프트 설계 의존, MVDream처럼 강력하지 않음

#### OrientDream (2024)
**혁신**: 카메라 방향 명시적 조건화 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10887623/)
- MVImgNet으로 사전학습된 2D text-to-image 모델
- Camera extrinsic matrix 직접 입력
- Decoupled backpropagation로 효율성 향상
- **성능**: 다중 view 일관성 극대화, Janus 문제 직접 해결

#### 3DFuse (2023)
**개선**: 2D 모델에 3D awareness 주입 [ku-cvlab.github](https://ku-cvlab.github.io/3DFuse/)
- Coarse 3D 구조 미리 생성
- View-specific depth map으로 조건화
- LoRA로 semantic consistency 유지
- **강점**: Robust multi-view consistency

### 7.2 View Consistency 및 Janus 문제 해결 방법
#### Debiasing Scores and Prompts (2023)
**분석**: View prompt와 user prompt의 충돌 [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2023/file/27725882a88f202e07319abbb3be7693-Paper-Conference.pdf)
- 예: "back view" 프롬프트에 "smiling"이 있으면 front bias 우세
- **해결**: Score debiasing + Prompt debiasing
- 동적 threshold truncation으로 trade-off 관리
- **효과**: View consistency 정량적 개선 입증

#### Score Debiasing의 원리
이 연구는 SDS 과정의 score를 분석하여: [arxiv](https://arxiv.org/html/2412.02287v3)
- View bias의 근본 원인: 훈련 데이터의 long-tailed distribution
- Front view 출현 빈도: Side, Back 대비 2-3배 높음
- **해결책**: Cross-attention map 제어 + CLIP 기반 view 필터링

#### Attention and CLIP Guidance (ACG, 2025)
**최신 접근**: [arxiv](https://arxiv.org/html/2412.02287v3)
- Cross-attention 동적 제어로 view-관련 keywords 강조
- CLIP 기반 view-text 유사도로 잘못된 view 필터링
- **성능**: Pseudo-GT viewpoint 분포 균형 (2:1:1 → 1.2:1:1)
- **장점**: Plug-and-play, 기존 방법과 통합 가능

### 7.3 부정 프롬프트 최적화 연구
#### ANSWER (2024)
**혁신**: Adaptive Negative Sampling Without External Resources [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Desai_Guiding_Diffusion_Models_with_Adaptive_Negative_Sampling_Without_External_Resources_ICCV_2025_paper.pdf)
- 각 diffusion step에서 부정 프롬프트 동적 조정
- Fixed prompt 가정 제거 (기존 방식의 한계)
- **성능**: CFG, DNP 대비 향상된 prompt compliance
- **이점**: Perp-Neg와 이론적으로 보완 가능

#### DNP (Diffusion Negative Prompt)
**방법**: 모델 자체로 최적 부정 프롬프트 도출 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Desai_Guiding_Diffusion_Models_with_Adaptive_Negative_Sampling_Without_External_Resources_ICCV_2025_paper.pdf)
- 부정 이미지 샘플링 후 자동 캡셔닝
- 수동 설계보다 효과적
- **제한**: 추가 forward pass 필요

#### Understanding the Impact of Negative Prompts (2024)
**기여**: 부정 프롬프트의 이론적 이해 [arxiv](https://arxiv.org/html/2406.02965v1)
- Conditional generation에서 negation의 작동 원리 분석
- Classifier-Free Guidance의 역할 재해석
- **영향**: 보다 효과적 음성 프롬프트 설계 지침

### 7.4 Score Distillation Sampling 개선
#### VSD (Variational Score Distillation) [arxiv](https://arxiv.org/abs/2305.16213)
- **문제**: 기본 SDS의 mode collapse, 저다양성
- **해결**: 입자 기반 변분 프레임워크
- **수식**: $\mathcal{L}\_{VSD} = \mathbb{E}\_\phi[w(t)\epsilon_\theta(x_t;\phi) - \epsilon]^2$
- **성능**: ProlificDreamer (CLIP R-Precision 0.83)

#### NFSD (Noise-Free Score Distillation)
- Denoising 잔차항 제거로 더 깨끗한 gradient
- 낮은 CFG scale에서도 작동
- 더 충실한 텍스처 디테일

#### StableDreamer (2023)
**개선**: SDS의 noisy nature 제어 [arxiv](https://arxiv.org/abs/2312.02189)
- Time-annealing noise schedule
- 2단계 훈련 (image-space diffusion for 정확도, latent-space for 색상)
- 3D Gaussians로 NeRF 대체 (더 빠른 렌더링)

#### RewardSDS (2025)
**최신**: Reward model 기반 가중치 [arxiv](https://arxiv.org/abs/2503.09601)
- Reward model으로 정렬도 평가
- 사용자 의도와 일치하는 noise samples 우선
- 양쪽 text-to-image와 text-to-3D에 적용 가능

### 7.5 연구 트렌드 분석
**2022-2023: 기초 및 최적화**
- DreamFusion으로 패러다임 시작
- Magic3D, VSD로 속도/품질 개선

**2023-2024: Multi-view 중심**
- MVDream, LatentNeRF로 consistency 강화
- Score debiasing으로 view bias 분석

**2024-2025: 효율성 및 정교함**
- OrientDream: 카메라 조건 명시화
- ACG: 주의력 제어로 fine-grained 조정
- RewardSDS: 사용자 정렬 최적화

## 8. 논문이 앞으로의 연구에 미치는 영향
### 8.1 이론적 영향
**부정 프롬프트의 수학적 재해석**
Perp-Neg는 부정 프롬프트를 처음으로 기하학적 공간에서 엄밀하게 분석했습니다. 이는 다음과 같은 후속 연구들에 영향을 미쳤습니다: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
- ANSWER의 동적 부정 프롬프트 개념 발전
- Score debiasing 이론의 근거 제공
- Compositional generation의 기하학적 이해

**View-Text Alignment의 새로운 관점**
View instruction을 기하학적으로 처리함으로써, view consistency 문제를 다음 관점에서 재프레이밍했습니다:
- View-related score의 방향성 제어
- Prompt-view 간 semantic overlap 해결
- Multi-objective guidance의 원리적 이해

### 8.2 실용적 영향
**Training-free 방법론의 타당성 입증**
- 복잡한 문제를 training-free로 해결 가능함을 보임
- 이후 ACG, ANSWER 등도 training-free 접근 채택
- 대형 모델 파인튜닝 비용 절감 가능성

**기하학적 조작의 효율성**
- 단순 투영 연산으로 강력한 효과 달성
- 계산 효율성과 효과의 좋은 균형
- 실제 애플리케이션 적용 용이

### 8.3 후속 연구의 방향성
**즉시적 영향**
Perp-Neg 발표 후 여러 방법들이 이를 기반으로 개선:
- 자동 부정 프롬프트 생성 (ANSWER) [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Desai_Guiding_Diffusion_Models_with_Adaptive_Negative_Sampling_Without_External_Resources_ICCV_2025_paper.pdf)
- Multi-view diffusion과의 통합 시도
- 3D-aware 버전 연구

**중장기적 영향**
- **View consistency**: ACG의 주의력 제어 방식에 영향 [arxiv](https://arxiv.org/html/2412.02287v3)
- **Score 분석**: Debiasing 연구에 이론적 토대 제공 [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2023/file/27725882a88f202e07319abbb3be7693-Paper-Conference.pdf)
- **Compositional generation**: 더 정교한 multi-prompt 조합 연구 촉진

## 9. 앞으로 연구 시 고려할 점
### 9.1 이론적 발전
**1. 더 엄밀한 수학적 분석**

현재 Perp-Neg의 이론적 분석 부족:
- Projection이 최적인지 수렴성 증명 필요
- 다양한 프롬프트 조합에 대한 일반화 이론
- Gaussian assumption의 타당성 검증

**개선 방안:**
$$\text{최적화: } \min_{\alpha, \beta} \|\epsilon_{result} - \text{desired}\|^2$$
subject to: $\nabla_{neg} \perp \nabla_{pos}$

**2. 동적 가중치 이론**

현재 고정 가중치 $w_{neg}$의 한계:
- Timestep 별 최적 가중치 변화
- 객체-view 조합별 동적 조정
- Adaptive weighting function 학습

**가능한 형태:**

$$w_{neg}(t, obj, view) = f_\theta(t, \text{embedding}(obj, view))$$

### 9.2 실무적 개선
**1. 자동 부정 프롬프트 생성**

프롬프트 설계 자동화:
```
User input: "a corgi wearing sunglasses"
→ Automatic NLP analysis
→ Detected concepts: [corgi, sunglasses]
→ Auto-generated negatives: [without sunglasses, cat, ...]
→ Perp-Neg sampling
```

ANSWER와의 통합: [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Desai_Guiding_Diffusion_Models_with_Adaptive_Negative_Sampling_Without_External_Resources_ICCV_2025_paper.pdf)
- 각 단계별로 부정 프롬프트 동적 업데이트
- 부정과 긍정의 최적 거리 학습

**2. 멀티 모달 조건화**

단순 텍스트 프롬프트 넘어서:
- 이미지 조건: "이 이미지 스타일 제외"
- 스케치 조건: "이 구조 유지하되..."
- 3D 조건: "이 기하학 변형"

**수식 확장:**

$$\epsilon_{Perp-Neg}^{multi} = \epsilon(x_t|c_{text}, c_{img}, c_{3d}) - \sum w_i \text{proj}_{\nabla_{pos}}(\nabla_{neg_i})$$

**3. 실시간 대화형 제어**

사용자 피드백 루프:
1. 초기 생성 (Perp-Neg)
2. 사용자 평가
3. 프롬프트 자동 조정
4. 재생성

### 9.3 기술적 도전과 해결책
**도전 1: 극단적 Rare 객체**

문제: 훈련 데이터에 거의 없는 객체 (예: 특정 동물의 특정 각도)

**해결책:**
- Few-shot learning: 몇 개 이미지로 적응
- Knowledge distillation: 유사 객체로부터 학습
- 외부 3D prior: Objaverse 같은 데이터 활용

**도전 2: Circular/Symmetric 객체**

문제: 회전 대칭 객체에서 view 구분 어려움

**해결책:**
- 의도적 비대칭 추가 프롬프트
- Depth-based regularization
- Semantic segmentation으로 구조 강제

**도전 3: 복합 장면**

문제: 다중 객체 간 관계 유지

**해결책:**
- Scene-level Perp-Neg
- Object-centric 분해 + 통합
- 관계 그래프 조건화

### 9.4 평가 지표 개선
**현재 평가의 한계:**
- Manual acceptance criteria (주관적)
- CLIP score (항상 정확하지 않음)
- 제한된 prompt/object 세트

**제안되는 개선:**

1. **정량적 지표 확대:**
   - 3D consistency metrics: Normal 일관성, Mesh 품질
   - View accuracy: CLIP과 depth-based 검증
   - Semantic preservation: Feature similarity

2. **자동화 평가:**
   ```python
   def evaluate_perp_neg(generated_image, prompt):
       view_accuracy = compute_view_match(image, prompt)
       semantic_preservation = clip_similarity(image, pos_prompt)
       negative_avoidance = 1 - clip_similarity(image, neg_prompt)
       return weighted_combination(view_accuracy, semantic, 
                                   negative_avoidance)
   ```

3. **인간 평가 체계화:**
   - 더 많은 평가자
   - 구체적 루브릭
   - Inter-rater agreement 측정

### 9.5 하이브리드 접근법
**Perp-Neg + MVDream 통합:**
```
MVDream (multi-view consistency) + Perp-Neg (부정 프롬프트 효율)
= 학습 기반 강점 + training-free 효율성
```

**Perp-Neg + VSD 통합:**
```
VSD (다양성) + Perp-Neg (view fidelity)
= 높은 품질 + 높은 다양성 + 정확한 view
```

**Perp-Neg + 3D-aware 통합:**
```
3DFuse (3D structure) + Perp-Neg (부정 프롬프트)
= 기하학적 정확성 + semantic consistency
```

### 9.6 확장 분야
**1. Video Generation**
- Temporal consistency 유지 + Perp-Neg
- 프레임별 부정 프롬프트 조정

**2. 3D Edit**
- "이 부분 제외하고 수정"
- Semantic inpainting with negative prompts

**3. Cross-modal Generation**
- Text-to-Audio with negative prompts
- 생성 모델 전반으로 확장 가능성

## 결론
Perp-Neg는 부정 프롬프트 문제를 기하학적으로 재해석하여 간단하면서도 강력한 해결책을 제시했습니다. 2D 이미지 생성에서 73.1%의 view 생성 성공률 향상과 3D 생성의 Janus 문제 완화는 실질적 기여입니다. [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)

그러나 이 논문의 진정한 가치는 **training-free 방식으로 복잡한 문제를 수학적으로 해결할 수 있다는 증명**에 있습니다. 이후 ANSWER, ACG 같은 연구들이 이 원리를 발전시키고 있으며, 2024-2025년 연구 트렌드는 다음을 보여줍니다:

1. **이론적 심화**: View bias의 근본 원인 분석 (long-tail distribution)
2. **방법론 발전**: 동적 적응형 접근 (ANSWER, RewardSDS)
3. **통합 경향**: 다양한 방법 간 상호보완 (Multi-view + 기하학적 제어)

앞으로의 연구는 Perp-Neg의 수학적 엄밀성을 강화하고, 자동화 및 적응성을 개선하며, 더 복잡한 생성 문제로 확장하는 방향으로 진행될 것으로 예상됩니다. 특히 **대규모 언어 모델과의 통합**으로 자동 프롬프트 최적화가 가능해진다면, Perp-Neg의 실용성은 더욱 극대화될 것입니다.

## 참고문헌
Armandpour et al., "Transform 2D Diffusion into 3D, alleviate Janus problem and Beyond" (arXiv:2304.04968, 2023) [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/perp-neg/)
 Ban et al., "Understanding the Impact of Negative Prompts" (arXiv:2406.02965, 2024) [arxiv](https://arxiv.org/html/2406.02965v1)
 Zhang et al., "Improving Viewpoint Consistency in 3D Generation via Attention and CLIP Guidance" (arXiv:2412.02287, 2025) [arxiv](https://arxiv.org/html/2412.02287v3)
 Seo et al., "Let 2D Diffusion Model Know 3D-Consistency for Robust Text-to-3D Generation" [ku-cvlab.github](https://ku-cvlab.github.io/3DFuse/)
 Desai et al., "Guiding Diffusion Models with Adaptive Negative Sampling Without External Resources" (ICCV 2025) [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Desai_Guiding_Diffusion_Models_with_Adaptive_Negative_Sampling_Without_External_Resources_ICCV_2025_paper.pdf)
 Xu et al., "ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation" (arXiv:2305.16213, 2023) [arxiv](https://arxiv.org/abs/2305.16213)
 Lin et al., "Magic3D: High-Resolution Text-to-3D Content Creation" (IEEE 2022-2023) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10203601/)
 Kwak et al., "OrientDream: Streamlining Text-to-3D Generation with Explicit Orientation Control" (arXiv:2406.10000, 2024) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10887623/)
 Armandpour et al., arXiv entry (arXiv:2304.04968) [arxiv](https://arxiv.org/abs/2304.04968)
 Lin et al., "Magic3D" (arXiv:2211.10440, 2022) [arxiv](https://arxiv.org/abs/2211.10440)
 Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion" (arXiv:2209.14988, 2022) [arxiv](https://arxiv.org/abs/2209.14988)
 Tagliaferri et al., "StableDreamer: Taming Noisy Score Distillation Sampling for Text-to-3D" (arXiv:2312.02189, 2023) [arxiv](https://arxiv.org/abs/2312.02189)
 Chachy et al., "RewardSDS: Aligning Score Distillation via Reward-Weighted Sampling" (arXiv:2503.09601, 2025) [arxiv](https://arxiv.org/abs/2503.09601)
 Xu et al., "ProlificDreamer" (arXiv:2305.16213, PDF 2023) [arxiv](https://arxiv.org/pdf/2305.16213.pdf)
 Shi et al., "MVDream: Multi-view Diffusion for 3D Generation" (YouTube 2023) [youtube](https://www.youtube.com/watch?v=6HjTgDL97hc)
 Hong et al., "Debiasing Scores and Prompts of 2D Diffusion for View-Consistent Text-to-3D Generation" (NeurIPS 2023) [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2023/file/27725882a88f202e07319abbb3be7693-Paper-Conference.pdf)
 Shi et al., "MVDream: Multi-view Diffusion for 3D Generation" (arXiv:2308.16512, 2023) [arxiv](https://arxiv.org/abs/2308.16512)
