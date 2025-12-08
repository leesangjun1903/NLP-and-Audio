
# GLICHEN: Open-Set Grounded Text-to-Image Generation

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장[1]

**GLIGEN**은 대규모 사전학습된 텍스트-이미지 확산 모델(Diffusion Model)의 가중치를 고정하면서 새로운 **공간 접지(Spatial Grounding) 능력**을 부여하는 혁신적인 접근방식을 제시합니다. 이는 기존의 완전히 새로운 모델을 처음부터 학습하는 방식과 대비되며, 사전학습된 모델의 막대한 개념 지식을 보존하면서도 제어 가능성을 극대화합니다.[1]

### 1.2 주요 기여[1]

1. **새로운 문제 정의**: 기존 텍스트-이미지 생성 모델의 제어 불가능 문제를 해결하기 위해 **개방형 집합 접지 텍스트-이미지 생성(Open-Set Grounded Text-to-Image Generation)** 문제를 정의[1]

2. **혁신적인 아키텍처 설계**: 게이트 자기-주의(Gated Self-Attention) 메커니즘을 통해 사전학습 가중치는 고정하면서 새로운 제어 신호를 주입[1]

3. **개방형 어휘 일반화**: 폐쇄형 집합(Closed-Set)으로만 학습했을 때도 **1,203개 범주의 LVIS 데이터셋에서 영점(Zero-Shot) 성능으로 감독학습 기준선을 큰 폭으로 상회**[1]

***

## 2. 해결하는 문제 및 방법론 상세 설명

### 2.1 문제 정의[1]

#### 기존 방법의 한계[1]

- **텍스트만으로의 제한성**: DALL-E 2, Imagen 등 최신 텍스트-이미지 모델은 자연언어 입력만 허용하여, 객체의 정확한 위치를 명시하기 어려움[1]
- **폐쇄형 집합 제한**: 기존 Layout-to-Image 방법(Layout2Im, LostGAN 등)은 COCO의 80개 객체 범주처럼 훈련 시 본 범주만 생성 가능[1]
- **지식 소실 위험**: 모델을 재학습할 경우 사전학습 단계에서 습득한 방대한 개념 지식 손실 가능성[1]

### 2.2 제안 방법: 수식 중심 설명[1]

#### 2.2.1 기본 확산 모델 학습 목표[1]

표준 잠재 확산 모델(LDM)의 학습 목표:

$$\mathcal{L}_{LDM} = \mathbb{E}_{z, \epsilon \sim \mathcal{N}(0, I), t} \|\epsilon - f_\theta(z_t, t, c)\|^2_2$$

여기서:
- $z_t$: 시간 $t$에서의 잡음 있는 잠재 표현
- $c$: 캡션 조건
- $f_\theta$: 시간 $t$에 조건을 받는 제거 오토인코더[1]

#### 2.2.2 접지 입력 정의[1]

접지된 텍스트-이미지 생성을 위한 입력 구성:

```math
\text{Instruction} = \left\{\begin{array}{l} \text{Caption: } c_1, \ldots, c_L \\ \text{Grounding: } e_1, l_1, \ldots, e_N, l_N \end{array}\right.
```

여기서:
- $L$: 캡션 길이
- $N$: 접지할 엔티티 개수
- $e_i$: 의미 정보(텍스트 또는 이미지)
- $l_i$: 공간 구성(바운딩 박스, 키포인트, 깊이맵 등)[1]

#### 2.2.3 접지 토큰 생성[1]

바운딩 박스가 있는 텍스트 엔티티에 대한 접지 토큰:

$$h_e = \text{MLP}(f_{\text{text}}(e), \text{Fourier}(l))$$

여기서:
- $f_{\text{text}}(e)$: CLIP 텍스트 인코더로 얻은 텍스트 임베딩
- $\text{Fourier}(l)$: 바운딩 박스 좌표 $(x_{min}, y_{min}, x_{max}, y_{max})$의 푸리에 임베딩
- MLP: 다층 퍼셉트론으로 두 입력을 연결[1]

#### 2.2.4 게이트 자기-주의(Gated Self-Attention)[1]

원본 LDM 트랜스포머 블록의 주의 메커니즘:

$$\text{SelfAttn}(v) = v$$
$$\text{CrossAttn}(h^c) = \text{CrossAttn}(h^c)$$

새로운 게이트 자기-주의 층 삽입:

$$\text{TS}[\text{SelfAttn}(v, h_e)] = \text{SelfAttn}([v, h_e]) + \alpha \cdot \text{SelfAttn}([v, h_e])$$

여기서:
- $v$: 시각 특성 토큰
- $h_e$: 접지 토큰
- $\text{TS}$: 토큰 선택 연산 (시각 토큰만 고려)
- $\alpha$: 학습 가능한 스칼라 (0으로 초기화)[1]

**핵심 설계**: 게이트를 0으로 초기화하여 안정적인 훈련을 보장하고, 초반에는 원래 모델의 영향을 최소화[1]

#### 2.2.5 지속적 학습 목표[1]

접지 정보를 포함한 확산 모델 훈련:

$$\mathcal{L}_{\text{Grounding}} = \mathbb{E}_{z, \epsilon \sim \mathcal{N}(0, I), t} \|\epsilon - f_{\theta, \phi}(z_t, t, c_t)\|^2_2$$

여기서:
- $\theta$: 원래 모델의 고정된 가중치
- $\phi$: 게이트 자기-주의 층과 MLP를 포함한 새로운 학습 가능한 파라미터
- $c_t$: 캡션 토큰과 접지 토큰 모두 포함[1]

**중요성**: 새로운 정보를 활용하면 잡음 제거 과정이 더 쉬워지므로, 모델은 자동으로 접지 정보 사용을 학습[1]

#### 2.2.6 스케줄된 샘플링(Scheduled Sampling)[1]

추론 시 성능-제어성 균형을 위한 두 단계 프로세스:

```math
\alpha = \left\{\begin{matrix} 1, & t \leq \alpha_0 T & \text{(접지 추론 단계)} \\ 0, & t > \alpha_0 T & \text{(표준 추론 단계)} \end{matrix}\right.
```

여기서:
- 초기 단계: 전체 접지 토큰 사용 ($\alpha=1$)으로 대략적인 개념 위치 결정
- 후기 단계: 원래 모델만 사용 ($\alpha=0$)으로 세밀한 시각적 품질 개선[1]

**이점**: 원래 Stable Diffusion이 고품질 이미지에 대해 미세조정되었으므로, 후기 단계에서 원래 가중치를 사용하면 이미지 품질 향상[1]

### 2.3 개방형-폐쇄형 전환[1]

#### 폐쇄형 집합 접근[1]

기존 방법: 엔티티당 학습 가능한 벡터 임베딩 사전 $U = \{u_1, \ldots, u_K\}$ 사용
- 한계: 훈련 시 본 $K$개 개념만 생성 가능

#### 개방형 집합 접근[1]

GLIGEN: 동일한 텍스트 인코더 사용
- **핵심 통찰**: 캡션 인코딩에 사용된 것과 동일한 텍스트 인코더로 접지 엔티티 처리
- **효과**: 훈련 데이터에 없는 개념도 생성 가능
- **이유**: 공유된 텍스트 공간에서 게이트 자기-주의가 시각 특성을 재배치하여, 후속 교차-주의 층이 새로운 엔티티 개념도 올바르게 처리[1]

### 2.4 다른 접지 방식으로의 확장[1]

#### 이미지 프롬프트[1]

$f_{\text{image}}(e)$ 사용:

$$h_e = \text{MLP}(P_t(P_i^{-1}(f_{\text{image}}(e))), \text{Fourier}(l))$$

텍스트 특성 공간으로 투영하여 통일성 유지[1]

#### 키포인트[1]

각 키포인트 위치에 푸리에 임베딩 적용하고, 같은 사람에 속한 키포인트를 연결하기 위해 학습 가능한 사람 토큰 사용[1]

#### 공간 정렬 조건(깊이맵, 엣지맵, 의미맵)[1]

ConvNeXt 백본으로 인코딩하여 $h \times w$ 해상도의 접지 토큰으로 변환[1]

***

## 3. 모델 구조 상세 설명

### 3.1 전체 아키텍처 개요[1]

GLIGEN의 아키텍처는 다음과 같은 핵심 구성요소로 이루어집니다:

**고정 컴포넌트 (원래 LDM/Stable Diffusion)**:
- U-Net 기반의 잡음 제거 오토인코더
- CLIP 텍스트 인코더
- 자기-주의 및 교차-주의 층

**새로운 학습 가능 컴포넌트**:
- 접지 토큰 생성 MLP
- 게이트 자기-주의 층 (각 트랜스포머 블록마다)
- 선택적으로 공간 조건 인코더[1]

### 3.2 게이트 자기-주의 층의 세부 구조[1]

각 트랜스포머 블록에서:

1. **원래 경로** (고정):
   - 시각 토큰 $v$의 자기-주의
   - 캡션 특성 $h^c$의 교차-주의

2. **새로운 경로** (학습 가능):
   - 시각 토큰과 접지 토큰의 연결: $[v, h_e]$
   - 자기-주의 적용
   - 시각 토큰만 추출: $\text{TS}[\cdot]$
   - 게이트로 스케일: $\alpha$

3. **통합**:
   - 두 경로의 잔차 연결[1]

### 3.3 U-Net에서의 위치[1]

U-Net 구조:
- **인코더**: 12개 블록 (해상도 감소)
- **중간 블록**: 병목 특성 처리
- **디코더**: 12개 블록 (해상도 증가), 인코더와 스킵 연결

게이트 자기-주의는 모든 트랜스포머 블록에 삽입되어, 서로 다른 해상도에서 접지 정보를 점진적으로 통합[1]

### 3.4 훈련 실무 세부사항[1]

**COCO 관련 실험**:
- 배치 크기: 64
- GPU: V100 16개
- 반복: 100,000회
- 학습률: 5e-5 (Adam)
- 워밍업: 처음 10,000회 반복

**데이터 스케일링 실험**:
- LDM의 경우: 400,000회 반복
- Stable Diffusion의 경우: 500,000회 반복
- 배치 크기: 32

**정규화**:
- 캡션과 접지 토큰을 10% 확률로 무작위 제거 (분류자 없는 지도 활용)[1]

***

## 4. 성능 향상 및 한계

### 4.1 성능 향상[1]

#### 4.1.1 폐쇄형 집합 설정: COCO 2017[1]

| 메트릭 | 기존 방법 | GLIGEN |
|--------|---------|--------|
| FID | 22.15 (TwFA) | **21.04** |
| AP | 13.40 (TwFA) | **22.4** |
| AP50 | 19.70 (TwFA) | **36.5** |
| AP75 | 14.90 (TwFA) | **24.1** |

**개선 사항**: 기존의 최고 성능 Layout2Image 방법들(LostGAN-V2, OCGAN, LAMA 등)을 능가하면서 이미지 품질도 우수[1]

#### 4.1.2 개방형 집합 설정: LVIS (영점 성능)[1]

| 훈련 데이터 | AP | AP_c | AP_f |
|-----------|-----|------|------|
| LAMA (감독학습) | 2.0 | 0.9 | 1.3 |
| GLIGEN (COCO만) | 6.4 | 5.8 | 7.4 |
| GLIGEN (확장 데이터) | 11.1 | 9.0 | 9.8 |

**획기적 결과**: COCO에만 훈련했을 때도 1,203개 카테고리의 LVIS에서 감독학습 기준선 LAMA를 **3배 이상** 능가[1]

#### 4.1.3 다양한 접지 방식 성능[1]

스케줄된 샘플링으로 다음을 달성:

- **키포인트 제어**: 인간 키포인트로만 훈련했을 때도 원숭이, 만화 캐릭터 등으로 일반화
- **의미맵 제어**: 의미적 레이아웃을 정확하게 준수
- **깊이맵/엣지맵 제어**: 복잡한 3D 기하학적 제약 처리[1]

### 4.2 한계 및 과제[1]

#### 4.2.1 구조적 한계[1]

1. **키포인트 일반화의 한계**:
   - 인간 키포인트로만 훈련했을 때 비인간 범주(예: 고양이, 램프)로 일반화 실패
   - **이유**: 바운딩 박스는 모든 범주에 공유 가능하지만, 키포인트(신체 부위)는 범주별로 다름[1]

2. **텍스트 인코더의 제한**:
   - CLIP 텍스트 인코더가 객체 수준 세부사항을 무시하는 경향
   - 예: "우산"을 포함한 다중 객체 문장에서 "우산" 누락 가능[1]

3. **공간 구성 정확도**:
   - 복잡한 다중-객체 장면에서 여전히 공간 관계 오류 발생 가능[1]

#### 4.2.2 계산 및 실용적 한계[1]

1. **추가 계산량**: 새로운 게이트 자기-주의 층으로 인한 추론 오버헤드
2. **메모리 요구**: 다중 접지 조건 처리 시 (예: 이미지+텍스트 그라운딩) 메모리 증가[1]

#### 4.2.3 데이터 품질 의존성[1]

- **의사 라벨 품질**: GLIP으로 생성한 의사 라벨의 정확도가 성능에 영향
- **데이터 불균형**: 희귀 범주에 대한 훈련 샘플 부족 시 일반화 저하[1]

***

## 5. 모델의 일반화 성능 향상 가능성 (심층 분석)[1]

### 5.1 현재 일반화 메커니즘[1]

#### 5.1.1 공유 텍스트 공간을 통한 개념 전이[1]

**핵심 발견**:
- 접지 엔티티와 캡션이 동일한 CLIP 텍스트 인코더로 처리됨
- 이를 통해 훈련 중 보지 못한 개념도 텍스트 임베딩 공간에서 의미적으로 연관된 개념으로 표현 가능[1]

**예시**:
- 훈련: COCO (80개 범주)
- 테스트: LVIS (1,203개 범주) → **3배 이상 성능 향상**[1]

#### 5.1.2 게이트 자기-주의의 역할[1]

게이트 자기-주의가 제공하는 일반화 이점:

1. **점진적 특성 재배치**:
   - 시각 특성을 접지 위치에 따라 동적으로 재배치
   - 레이아웃 구조가 다른 데이터에 적응[1]

2. **경로 가소성 (Path Plasticity)**:
   - 필요에 따라 게이트 강도 $\alpha$ 조정
   - 훈련 초기: 낮은 $\alpha$ (기존 지식 보존)
   - 훈련 진행: 증가하는 $\alpha$ (새로운 기능 학습)[1]

3. **로컬리티 (Locality)**:
   - 각 시각 토큰이 국소적 그라운딩 정보만 처리
   - 전역 구조와 국소 세부사항의 분리[1]

### 5.2 일반화 성능 향상 방안[1]

#### 5.2.1 훈련 데이터 확장의 효과[1]

데이터 규모에 따른 성능:

| 훈련 데이터 | AP | AP_c | AP_f | FID |
|-----------|-----|------|------|-----|
| COCO 2014CD | 6.4 | 5.8 | 7.4 | 22.17 |
| + Objects365 | 8.45 | - | - | - |
| + SBU, CC3M | 11.1 | 9.0 | 9.8 | 10.28 |

**핵심 발견**: 데이터 규모가 두 배 이상 증가할 때마다 성능이 일관되게 향상되며, **희귀 범주(AP_c, AP_f)에서 더 큰 개선**[1]

**이유**: 다양한 도메인과 개념을 포함한 대규모 훈련이 공유 텍스트 공간의 표현력 향상[1]

#### 5.2.2 스케줄된 샘플링의 역할[1]

**일반화 향상 메커니즘**:

1. **도메인 간 전이**:
   - 인간 키포인트로 훈련 → 스케줄된 샘플링으로 원숭이/만화에 일반화
   - 이유: 초기 단계에서 대략적인 구조 결정, 후기에 도메인 특화 세부사항 무시[1]

2. **지식 보존-확장 균형**:
   - 후기 단계에서 원래 모델 가중치 사용 → 사전학습 일반화 능력 활용
   - 초기 단계에서만 새로운 정보 주입 → 과적합 방지[1]

#### 5.2.3 Fourier 임베딩의 효과[1]

**개선 가능성**:
- 기존: MLP 임베딩 사용 시 레이아웃 정확도 큰 폭 감소 (AP: 21.7 → 3.2)
- Fourier 임베딩: 위치 정보의 고주파 성분 캡처로 정확한 공간 제어[1]

**미래 개선**: 더 정교한 위치 인코딩(예: 2D 위치 인식 보간)으로 추가 향상 가능[1]

#### 5.2.4 주의 맵 분석을 통한 인사이트[1]

**시각화 발견**:
- 첫 샘플링 단계(Gaussian 잡음)에서도 접지 토큰이 올바른 공간 대응 학습
- 이는 모델이 **조기에 구조적 제약을 학습**함을 시사
- 후기 단계: 이 대응성이 감소 (스케줄된 샘플링 효과)[1]

### 5.3 이론적 일반화 메커니즘[1]

#### 5.3.1 복합적 일반화[1]

**가설**:
게이트 자기-주의는 **국소성(Locality)** 원리를 따르므로, 훈련 시 보지 못한 합성(Composition)도 처리 가능

예:
- 훈련: 개별적으로 "빨간 자동차", "파란 집" 본 경험
- 테스트: "빨간 집과 파란 자동차" 생성 가능[1]

#### 5.3.2 공유 텍스트 공간의 강점[1]

**연속적 임베딩 공간**:
- 이미지 생성에 사용되는 CLIP 공간이 의미적으로 연속
- 훈련 데이터의 "구멍"을 보간(Interpolation) 가능[1]

***

## 6. 해당 논문이 앞으로의 연구에 미치는 영향[1]

### 6.1 패러다임 전환: 어댑터 기반 확장[1]

#### 6.1.1 영향력[1]

GLIGEN의 성공은 후속 연구에 다음과 같은 새로운 패러다임을 제시:

**기존 패러다임**: 각 새로운 작업마다 처음부터 모델 훈련
**새로운 패러다임**: 사전학습 모델을 고정하고 **경량 어댑터 모듈**로 새 기능 추가[1]

#### 6.1.2 관련 후속 연구들[2][3][4][5]

**ControlNet (Zhang et al., 2023)**: 
- GLIGEN과 유사하게 대규모 확산 모델에 공간 제어 추가
- 영점 초기화 합성곱 사용으로 안정적 훈련 달성
- **확장성**: Canny 엣지, 깊이맵, 포즈, 의미맵 등 다양한 조건 지원[3]

**ControlNet-XS (Kohler et al., 2024)**:
- ControlNet을 피드백 제어 시스템으로 재해석
- 고주파 통신으로 제어 충실도 개선[4]

**FreeControl (Mo et al., 2023)**:
- 훈련 없이 어떤 T2I 모델에도 적용 가능한 공간 제어
- GLIGEN보다 더 유연한 아키텍처[5]

**Ctrl-Adapter (Wei et al., 2024)**:
- 하나의 어댑터로 여러 모델(SDXL, PixArt, DiT 기반 모델) 간 전이 가능[6]

### 6.2 개방형 어휘 문제의 진전[1]

#### 6.2.1 영향[1]

GLIGEN은 **개방형 어휘 생성**이 가능함을 증명하여, 이후 연구에 중요한 통찰 제공:

**핵심 아이디어**: 사전학습된 언어 모델(텍스트 인코더)의 연속 임베딩 공간을 활용하면, 폐쇄형 집합 데이터로만 훈련해도 개방형 어휘 일반화 가능[1]

#### 6.2.2 관련 연구 방향[7][8][1]

**GLIP (Li et al., 2022)**:
- 대규모 텍스트-이미지 쌍 사전학습으로 개방형 어휘 객체 검출 가능
- GLIGEN의 텍스트 인코더 기반이 되는 선행 연구[1]

**ObjectDiffusion (Süleyman et al., 2025)**:
- GLIGEN의 아이디어를 확장하여 의미적/공간적 제어 결합[8]

**Open-Set Recognition (Miller et al., 2024)**:
- 시각-언어 모델의 개방형 집합 취약성 분석
- 공개 어휘 객체 검출기의 신뢰도 평가 프레임워크 제시[9]

### 6.3 조건부 생성의 미래 방향[1]

#### 6.3.1 다중 양식 조건[1]

GLIGEN이 보여준 **다중 조건 지원**이 후속 연구의 방향:

**GLIGEN에서 지원하는 조건들**:
- 텍스트 바운딩 박스
- 이미지 프롬프트
- 키포인트
- 깊이맵, 엣지맵, 법선맵, 의미맵[1]

**Cocktail (Wei et al., 2023)**:
- 여러 양식을 하나의 임베딩으로 혼합
- 일반화된 ControlNet 제시[10]

**FineControlNet (Xiao et al., 2023)**:
- 기하학적 제어(포즈)와 외관 제어(텍스트) 결합
- 인스턴스별 세밀한 제어[11]

#### 6.3.2 텍스트 기반 공간 추론[1]

**제어-GPT (Zhao et al., 2023)**:
- GPT-4를 활용해 텍스트를 TikZ 스케치로 변환
- GLIGEN/ControlNet과 결합해 공간 추론 능력 강화[12]

### 6.4 확산 모델의 일반화 이론[13][14][15][1]

#### 6.4.1 이론적 기여[1]

GLIGEN의 성공은 다음과 같은 이론적 질문을 촉발:

**문제**: 왜 폐쇄형 집합(COCO)으로만 훈련해도 개방형 집합(LVIS)에 일반화되는가?[1]

#### 6.4.2 후속 이론 연구[14][15][13]

**복합적 일반화의 메커니즘 (Kamb et al., 2025)**:
- 조건부 확산 모델의 국소성(Locality)이 복합 일반화를 가능하게 함
- "조건부 사영성 합성"과 "국소 조건부 점수"의 동치성 증명[14]

**확산 모델 일반화의 이론적 경계**:
- 샘플 크기 $n$과 모델 용량 $m$에 대한 일반화 오차 경계 도출
- 수렴 속도: $O(n^{-2/5} + m^{-4/5})$[16]

**확산 모델 일반화의 메커니즘**:
- 로컬 잡음 제거 연산이 훈련 분포를 넘어 좋은 근사 제공
- 이는 GLIGEN의 게이트 자기-주의 국소성과 부합[17]

### 6.5 적응형 학습과 전이 학습[18][19][20][21][1]

#### 6.5.1 어댑터 전이의 새로운 가능성[1]

GLIGEN의 아키텍처는 **어댑터 전이** 연구를 촉발:

**A4A (Tu et al., 2025)**:
- GLIGEN 스타일의 어댑터를 서로 다른 모델 아키텍처(U-Net ↔ Transformer) 간에 전이
- 모든-모든(All-for-All) 매핑으로 통일 공간 구성[18]

**RelationAdapter (Ge et al., 2025)**:
- DiT 기반 모델을 위한 경량 어댑터
- 최소 예시에서 시각적 변환 학습 및 전이[22]

**PEA-Diffusion (Zhang et al., 2024)**:
- 지식 증류를 통한 언어 전이에 어댑터 활용
- 단 6M 파라미터로 다국어 T2I 가능[21]

#### 6.5.2 선호도 정렬[19]

**Diffusion-DPO (Wallace et al., 2024)**:
- Direct Preference Optimization을 확산 모델에 적용
- 인간 피드백으로 정렬하여 미학과 충실도 개선
- 뉴럴 기준선 개선으로 GLIGEN 같은 조건부 모델 개선 가능[19]

### 6.6 응용 분야의 확대[23][24][25][1]

#### 6.6.1 설계 및 편집[1]

**DesignDiffusion (Lin et al., 2025)**:
- 텍스트 기반 디자인 이미지 생성
- 텍스트 및 시각 요소 간 일관성 유지[23]

#### 6.6.2 다중 뷰 생성[1]

**MVDiffusion (Shi et al., 2023)**:
- 대응 인식 주의 층으로 여러 뷰 일관성 유지
- GLIGEN의 교차 뷰 상호작용 메커니즘 활용[25]

#### 6.6.3 의료 이미징[1]

**Chest-Diffusion (Zeng et al., 2024)**:
- 의학 보고서에서 흉부 X-ray 생성
- 도메인 특화 텍스트 인코더로 접지 개선[26]

***

## 7. 앞으로의 연구 시 고려할 점

### 7.1 기술적 개선 방안[1]

#### 7.1.1 아키텍처 최적화[1]

1. **위치 임베딩 강화**:
   - GLIGEN은 게이트 자기-주의가 위치 임베딩 부족 문제 보고
   - **미래 개선**: 사전학습 단계에서부터 위치 임베딩 포함
   - **기대 효과**: 다른 해상도에서의 일반화 개선[1]

2. **적응형 게이트 값**:
   - 현재: 훈련 중 $\alpha=1$ 고정
   - **제안**: 시간 단계별로 적응형 $\alpha(t)$ 사용
   - **이점**: 스케줄된 샘플링을 훈련 단계에 통합, 안정성 향상[1]

3. **멀티-헤드 게이트**:
   - 현재: 단일 스칼라 게이트
   - **제안**: 각 주의 헤드에 다른 게이트 계수
   - **효과**: 다양한 해상도에서의 독립적 제어[1]

#### 7.1.2 훈련 전략 개선[1]

1. **누적 학습(Continual Learning)**:
   - 새로운 도메인이 추가될 때 catastrophic forgetting 방지
   - **방법**: 리플레이 버퍼 또는 메모리 기반 학습
   - **관련 연구**: ExpertDiff (Jaini et al., 2025) 참고[20]

2. **다중 손실 함수**:
   - 현재: 기존 LDM 손실만 사용
   - **제안**: 공간 정확도(YOLO AP) 직접 최적화
   - **문제**: 미분 가능한 공간 손실 설계 필요[1]

3. **의사 라벨 개선**:
   - GLIP 의사 라벨의 품질이 성능 상한 결정
   - **개선**: 자기 학습(Self-training) 또는 교사-학생 방식으로 의사 라벨 정제[1]

### 7.2 일반화 성능 향상 전략[15][16][14][1]

#### 7.2.1 복합 일반화 활용[14][1]

**이론적 기반**:
- Kamb et al. (2025): 국소 조건부 점수로 사영성 합성 달성 가능[14]

**적용 방법**:
1. 훈련 데이터의 기본 조건(예: 기본 객체, 기본 배경)만 포함
2. 모델이 자동으로 이들의 합성에 일반화 가능
3. **테스트**: 매우 드문 조합도 생성 가능[1]

#### 7.2.2 메타-학습 접근[1]

**방향**: Few-shot adaptation
- 새로운 도메인에 소량 데이터만으로 적응
- 기존 어댑터를 메타-학습자로 활용[1]

#### 7.2.3 자기 감독 학습[1]

**아이디어**: 
- 생성된 이미지로부터 객체 검출 모델 훈련
- 신선한 피드백으로 모델 개선[1]

### 7.3 평가 메트릭 개선[1]

#### 7.3.1 공간 일관성 측정[1]

**현재 한계**:
- FID: 전역 이미지 품질만 측정
- YOLO AP: 물체 검출 정확도만 측정 (품질 무시)[1]

**제안**:
1. **공간-의미 일관성(Spatial-Semantic Consistency)**:
   - 객체의 공간적 위치와 의미적 내용이 일치하는 정도 측정
   
2. **분해된 지표(Decomposed Metrics)**:
   - 객체 위치 정확도
   - 객체 크기 정확도
   - 객체 속성(색상, 질감) 일치도
   - 객체 간 공간 관계[1]

3. **인간 평가 프로토콜**:
   - 시각적 품질 vs. 공간 정확도의 우선순위 설정
   - 도메인별(의료, 디자인 등) 커스터마이즈드 평가[1]

### 7.4 데이터 및 확장성 이슈[1]

#### 7.4.1 고품질 데이터셋 구축[1]

**필요성**:
- 현재: COCO의 80개 범주 수준의 주석
- 미래: 더 다양한 범주와 복잡한 공간 관계[1]

**방향**:
1. **합성 데이터**: 3D 렌더링으로 다양한 배경, 조명, 시점 생성
2. **크라우드소싱**: 정확한 공간 주석 수집
3. **약한 감독**: 웹 데이터에서 자동으로 주석 수집[1]

#### 7.4.2 희귀 범주 문제[1]

**현황**:
- LVIS에서 희귀 범주(< 5개 샘플): 성능 급락
- GLIGEN도 희귀 범주에서 기존 방법 대비 소량 우위[1]

**해결책**:
1. **장꼬리 학습(Long-Tail Learning)** 기법 적용
2. **균형 잡힌 샘플링**: 희귀 범주를 오버샘플링
3. **전이 학습**: 유사 범주에서 특성 활용[1]

### 7.5 윤리 및 안전성[1]

#### 7.5.1 편향(Bias) 문제[1]

**우려사항**:
- 웹 데이터의 편향이 학습되어 GLIGEN에 반영
- 사전학습 CLIP 인코더의 고유 편향[1]

**완화 방안**:
1. 훈련 데이터 감시 및 균형화
2. 편향 평가 벤치마크 개발
3. 다양한 표현 확보[1]

#### 7.5.2 악용 방지[1]

**잠재적 문제**:
- 생합성 이미지로 오도성 내용 생성
- 개인 정보 보호 문제[1]

**대책**:
1. 수량 감지(Quantity Detection) 기술 개발
2. 사용자 동의 및 감사 로그
3. 책임 있는 배포 정책[1]

### 7.6 도메인 특화 연구[1]

#### 7.6.1 의료 이미징[1]

**도전**: 정확도 < 이미지 품질
**방향**: 의료 전문가 피드백 기반 정렬[1]

#### 7.6.2 건축 및 디자인[1]

**요구사항**: 엄격한 기하학적 제약 준수
**연구**: 제약 전파(Constraint Propagation) 메커니즘[1]

#### 7.6.3 실시간 응용[1]

**문제**: 현재 추론 시간 너무 김 (여러 초)
**목표**: 실시간 인터랙티브 편집[1]

***

## 8. 2020년 이후 관련 최신 연구 탐색

### 8.1 확산 모델의 기초 연구 (2020-2022)[27][28][29][1]

#### 8.1.1 확산 모델의 출현[28][29][27]

**DDPM (Ho et al., 2020)**:
- 확산 기반 생성 모델의 기초 제시
- 이미지 품질에서 GAN 능가[27]

**Latent Diffusion Model/Stable Diffusion (Rombach et al., 2022)**:
- 잠재 공간에서의 확산으로 계산량 대폭 감소
- GLICHEN의 기본 아키텍처[30][1]

**Imagen (Saharia et al., 2022)**:
- 텍스트-이미지 확산의 획기적 성과
- GLICHEN의 영감원[29]

### 8.2 조건부 제어 연구 (2022-2024)[3][4][5][11][12][10]

#### 8.2.1 ControlNet 계열[4][5][3]

**ControlNet (Zhang et al., 2023)**:
- GLICHEN과 유사 시기에 발표된 동시대 연구
- 영점 합성곱으로 더 강력한 제어 달성
- **비교**: GLICHEN은 게이트 자기-주의, ControlNet은 별도 제어 네트워크[3]

**ControlNet-XS (Kohler et al., 2024)**:
- 제어 시스템 관점의 재해석
- 고주파 통신으로 효율성 개선[4]

**ControlNet++ (Li et al., 2024)**:
- 다양한 조건에서 11-13% 성능 향상
- 일관성 피드백 메커니즘[31]

#### 8.2.2 다양한 조건 제어[11][12][10]

**FineControlNet (Xiao et al., 2023)**:
- 인스턴스별 외관 제어 추가
- 포즈와 텍스트 조건 결합[11]

**Control-GPT (Zhao et al., 2023)**:
- GPT-4 기반 텍스트-투-스케치 변환
- GLICHEN의 텍스트 기반 제어 확장[12]

**Cocktail (Wei et al., 2023)**:
- 다중 양식 조건 혼합
- 공간적 정밀 가이드 샘플링[10]

### 8.3 개방형 어휘 및 일반화 연구 (2022-2025)[7][8][9][1]

#### 8.3.1 개방형 어휘 객체 검출[7]

**GLIP (Li et al., 2022)**:
- 대규모 텍스트-이미지 쌍 사전학습
- GLICHEN의 텍스트 인코더 기반[11][1]

**DetPro (Du et al., 2022)**:
- 학습 가능한 프롬프트로 개방형 어휘 검출[7]

#### 8.3.2 개방형 집합 인식[8][9]

**ObjectDiffusion (Süleyman et al., 2025)**:
- GLICHEN 아이디어 직접 확장
- 의미적 및 공간적 그라운딩 결합[8]

**Open-Set Recognition (Miller et al., 2024)**:
- VLM 기반 개방형 집합 검출 취약성 분석
- 신뢰도 측정 프레임워크 제시[9]

### 8.4 일반화 및 전이 학습 연구 (2023-2025)[13][15][16][18][19][14]

#### 8.4.1 일반화 이론[15][16][13][14]

**"Emergence of Reproducibility" (Kamb et al., 2024)**:
- 국소 조건부 점수의 중요성 입증
- GLICHEN의 국소성이 일반화 제공[14]

**"Diffusion Model Generalization" (Tsigler et al., 2024)**:
- 이론적 일반화 경계 도출
- $O(n^{-2/5} + m^{-4/5})$ 수렴 속도[16]

**"OOD Generalization" (Zhang et al., 2023)**:
- 분포 외 일반화의 종합 리뷰
- GLICHEN의 개방형 어휘는 OOD 일반화의 극단적 사례[32]

#### 8.4.2 어댑터 및 전이 학습[21][22][18][19]

**A4A (Tu et al., 2025)**:
- 크로스 아키텍처 어댑터 전이
- GLICHEN 스타일 모듈의 일반화[18]

**Diffusion-DPO (Wallace et al., 2024)**:
- 직접 선호도 최적화로 정렬
- 기존 조건부 모델 개선 기반 제공[19]

**PEA-Diffusion (Zhang et al., 2024)**:
- 지식 증류로 다국어 전이
- 경량 어댑터의 효율성 입증[21]

**RelationAdapter (Ge et al., 2025)**:
- DiT 기반 모델용 경량 모듈
- 최소 예시 학습[22]

### 8.5 응용 분야 연구 (2023-2025)[26][25][23]

#### 8.5.1 설계 및 의료[23][26]

**DesignDiffusion (Lin et al., 2025)**:
- 설계 이미지 텍스트-투-이미지 생성
- 텍스트 및 시각 요소 일관성[23]

**Chest-Diffusion (Zeng et al., 2024)**:
- 의학 보고서 → 흉부 X-ray
- 도메인 특화 텍스트 인코더[26]

#### 8.5.2 다중 뷰 및 비디오[33][25]

**MVDiffusion (Shi et al., 2023)**:
- 다중 뷰 일관성 생성
- 교차 뷰 주의 메커니즘[25]

**ConditionVideo (Liu et al., 2023)**:
- 훈련 없는 조건부 텍스트-투-비디오
- 3D 제어 네트워크[33]

### 8.6 효율성 및 최적화 연구 (2023-2025)[34][2][31][6][27]

#### 8.6.1 모델 경량화[2][34]

**BK-SDM (2024)**:
- 블록 프루닝과 특성 증류
- 30-50% 크기 감소 유지 성능[34]

**PIXART-δ (2024)**:
- 지연 일관성 모델(LCM) 통합
- 1024px 이미지 0.5초 생성[2]

#### 8.6.2 제어 네트워크 효율화[31][6][27]

**ControlNet++ (Li et al., 2024)**:
- 효율적 일관성 피드백
- 성능-효율 트레이드오프 개선[31]

**Ctrl-Adapter (Wei et al., 2024)**:
- 어댑터 일반화와 전이
- 여러 모델에 한 번의 훈련[6]

**FlexControl (Zhang et al., 2025)**:
- 계산 인식 라우터
- 동적 제어 블록 선택[35]

***

## 결론

**GLIGEN**은 사전학습된 확산 모델의 지식을 보존하면서 새로운 공간 제어 능력을 추가하는 혁신적 접근방식을 제시하며, **개방형 어휘 일반화**를 통해 폐쇄형 데이터셋으로만 훈련해도 광범위한 범주에 적용 가능함을 증명했습니다.[1]

이는 **어댑터 기반 모듈식 확장**, **공유 텍스트 공간 활용**, **국소 제어 메커니즘**이라는 세 가지 핵심 기여로, 후속 연구들(ControlNet, FreeControl, ObjectDiffusion 등)의 기초를 마련했으며, 향후 연구는 **위치 인식 개선**, **복합적 일반화**, **도메인 특화 적응**, **실시간 처리** 등의 방향으로 진행될 것으로 예상됩니다.[5][12][10][15][3][4][11][14][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e8625f98-3272-4590-a956-73db20edae93/2301.07093v2.pdf)
[2](https://arxiv.org/abs/2401.05252)
[3](https://ieeexplore.ieee.org/document/10656779/)
[4](https://ieeexplore.ieee.org/document/10377881/)
[5](https://arxiv.org/abs/2312.06573)
[6](http://arxiv.org/pdf/2404.09967.pdf)
[7](https://openaccess.thecvf.com/content/CVPR2022/papers/Du_Learning_To_Prompt_for_Open-Vocabulary_Object_Detection_With_Vision-Language_Model_CVPR_2022_paper.pdf)
[8](https://arxiv.org/abs/2501.09194)
[9](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05808.pdf)
[10](https://arxiv.org/abs/2306.00964)
[11](https://ieeexplore.ieee.org/document/10943519/)
[12](https://arxiv.org/abs/2305.18583)
[13](https://arxiv.org/abs/2412.14474)
[14](https://arxiv.org/abs/2509.16447)
[15](https://arxiv.org/abs/2502.04549)
[16](https://arxiv.org/pdf/2311.01797.pdf)
[17](https://arxiv.org/html/2411.19339v2)
[18](https://openaccess.thecvf.com/content/CVPR2025/papers/Tu_A4A_Adapter_for_Adapter_Transfer_via_All-for-All_Mapping_for_Cross-Architecture_CVPR_2025_paper.pdf)
[19](https://openaccess.thecvf.com/content/CVPR2024/papers/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.pdf)
[20](https://www.ijcai.org/proceedings/2025/0764.pdf)
[21](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08492.pdf)
[22](https://arxiv.org/html/2506.02528v1)
[23](https://arxiv.org/html/2503.01645v1)
[24](https://arxiv.org/pdf/2403.04279.pdf)
[25](https://arxiv.org/pdf/2307.01097.pdf)
[26](http://arxiv.org/pdf/2407.00752.pdf)
[27](http://arxiv.org/pdf/2502.14779.pdf)
[28](https://openaccess.thecvf.com/content/ICCV2023/papers/Park_Learning_to_Generate_Semantic_Layouts_for_Higher_Text-Image_Correspondence_in_ICCV_2023_paper.pdf)
[29](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/controlnet/)
[30](https://blog.cslee.co.kr/controlnet-spatial-control-text-to-image-diffusion/)
[31](http://arxiv.org/pdf/2404.07987.pdf)
[32](https://arxiv.org/pdf/2108.13624.pdf)
[33](https://arxiv.org/abs/2310.07697)
[34](https://link.springer.com/10.1007/978-3-031-72949-2)
[35](https://arxiv.org/html/2502.10451v1)
[36](http://pubs.rsna.org/doi/10.1148/radiol.231971)
[37](https://www.semanticscholar.org/paper/9c85e6e0f58b480801fe6f1fa09305e2b9c46331)
[38](https://ieeexplore.ieee.org/document/10657216/)
[39](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[40](https://aclanthology.org/2023.semeval-1.301)
[41](https://dl.acm.org/doi/10.1145/3707292.3707367)
[42](https://arxiv.org/abs/2412.06487)
[43](https://www.semanticscholar.org/paper/4f1dcc4fda12072a27c2f2af965b962acb63d1a6)
[44](https://arxiv.org/pdf/2211.01324.pdf)
[45](https://arxiv.org/pdf/2401.10061.pdf)
[46](http://arxiv.org/pdf/2211.15388.pdf)
[47](https://arxiv.org/html/2412.12888v1)
[48](https://proceedings.neurips.cc/paper_files/paper/2024/file/860c1c657deafe09f64c013c2888bd7b-Paper-Conference.pdf)
[49](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08373.pdf)
[50](https://arxiv.org/abs/2303.07909)
[51](https://bequiet-log.vercel.app/alignment-diffusion)
[52](https://velog.io/@ad_official/GLIGEN-Open-Set-Grounded-Text-to-Image-Generation)
[53](https://velog.io/@whdnjsdyd111/Learning-to-Prompt-for-Open-Vocabulary-Object-Detection-with-Vision-Language-Model)
[54](https://liner.com/review/raphael-texttoimage-generation-via-large-mixture-of-diffusion-paths)
[55](https://link.springer.com/10.1007/978-3-031-73223-2_20)
[56](https://ieeexplore.ieee.org/document/10655542/)
[57](https://academic.oup.com/neuro-oncology/article/25/Supplement_2/ii9/7264461)
[58](https://arxiv.org/abs/2302.05543)
[59](https://arxiv.org/html/2410.14279)
[60](https://arxiv.org/pdf/2403.09638.pdf)
[61](https://arxiv.org/html/2312.07536v1)
[62](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf)
[63](https://arxiv.org/pdf/2509.12046.pdf)
[64](https://openreview.net/pdf?id=laWYA-LXlNb)
[65](https://arxiv.org/html/2511.18537v1)
[66](https://www.sciencedirect.com/science/article/abs/pii/S0957417424025120)
[67](https://proceedings.neurips.cc/paper_files/paper/2023/file/b87bdcf963cad3d0b265fcb78ae7d11e-Paper-Conference.pdf)
[68](https://iopscience.iop.org/article/10.1149/MA2025-02412068mtgabs)
[69](https://iopscience.iop.org/article/10.1149/MA2025-024658mtgabs)
[70](http://arxiv.org/pdf/2412.17162.pdf)
[71](https://arxiv.org/html/2310.05264v3)
[72](http://arxiv.org/pdf/2106.04496.pdf)
[73](http://arxiv.org/pdf/2409.10094.pdf)
[74](https://arxiv.org/pdf/2307.13949.pdf)
[75](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Le_Bellier_Detecting_Out-Of-Distribution_Earth_Observation_Images_with_Diffusion_Models_CVPRW_2024_paper.pdf)
[76](https://arxiv.org/abs/2311.12908)
[77](https://arxiv.org/html/2307.04726v4)
[78](https://github.com/xie-lab-ml/awesome-alignment-of-diffusion-models)
[79](https://www.sciencedirect.com/science/article/pii/S1361841524000136)
