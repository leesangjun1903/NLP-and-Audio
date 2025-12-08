# A Novel Sampling Scheme for Text- and Image-Conditional Image Synthesis in Quantized Latent Spaces

### 1. 핵심 주장 및 주요 기여 요약[1]

이 논문은 **Paella**라는 새로운 텍스트-이미지 생성 모델을 제시하며, 기존의 복잡한 확산 모델(diffusion models)과 트랜스포머 기반 접근법의 대안으로 간단하면서도 효율적인 방법을 제안합니다. 핵심 주장은 다음과 같습니다:[1]

첫째, **간단성과 접근성**을 강조합니다. 텍스트-이미지 생성의 복잡성을 낮추어 AI 분야 전문가뿐 아니라 일반인도 이해하고 구현할 수 있도록 하는 것을 목표로 합니다.[1]

둘째, **샘플링 효율성**입니다. 단 **12개의 샘플링 스텝**으로 고품질 이미지를 생성할 수 있으며, 이는 기존 확산 모델의 수백 스텝이나 트랜스포머 기반 방식의 낮은 압축률 문제를 해결합니다.[1]

셋째, **토큰 리노이징(token renoising)** 이라는 혁신적인 샘플링 기법을 도입합니다. MaskGIT과 MUSE의 마스크 토큰 방식과 달리, 샘플링된 토큰을 초기 노이즈로 임의로 교체하여 모델이 예측을 반복적으로 개선할 수 있게 합니다.[1]

넷째, **1B 파라미터 모델로 경쟁력 있는 성능**을 달성합니다. 900만 이미지로 훈련한 Paella는 기존의 더 큰 모델들과 비교 가능한 FID 점수를 얻으면서도 파라미터가 훨씬 적습니다.[1]

다섯째, **다중 조건화 방식**을 지원합니다. 텍스트만, 이미지만, 또는 텍스트와 이미지를 조합하여 조건화할 수 있으며, CLIP 이미지 임베딩을 활용하여 제로샷 이미지 변형과 스타일 전이가 가능합니다.[1]

***

### 2. 해결하고자 하는 문제 및 제안 방법

#### 2.1 문제 정의[1]

기존 텍스트-이미지 생성 방법의 핵심 문제점들:

**확산 모델의 문제점:**[1]
- 높은 계산 비용으로 인한 느린 추론 속도
- 많은 샘플링 스텝 필요 (수백 단계)
- 증가하는 복잡성과 추상화 수준

**트랜스포머 기반 방식의 문제점:**[1]
- 2D 이미지를 1D 시퀀스로 평탄화하는 비자연스러운 방식
- 자기 어텐션의 이차 메모리 증가로 인한 높은 압축률 필요
- 높은 압축률로 인한 세부 손실과 인공물

#### 2.2 제안 방법[1]

**아키텍처 선택:**
- **컨볼루션 기반 모델** 사용: 2D 이미지 구조를 자연스럽게 처리
- **낮은 압축률** (f=4): 이차 메모리 증가 문제 해결
- **U-Net 스타일 인코더-디코더**: 3 레벨, 잔차 블록 기반

#### 2.3 수식 포함 상세 설명[1]

**훈련 과정:**

Step 1) 이미지를 VQGAN으로 인코딩하여 양자화된 토큰 공간으로 변환:
- 입력: H × W × C 해상도 이미지
- VQGAN 인코더: h × w × z 해상도 (h = H/f, w = W/f)
- 양자화: 각 벡터를 학습된 코드북 Q ∈ ℝ^(N_CB × z)에서 가장 가까운 값으로 교체

Step 2) 노이징 과정 - 임의 토큰 교체를 통한 마스킹:

$$\bar{u}_{x,y} = \begin{cases} u_{x,y} & \text{if } m_{x,y} = 0 \\ n_{x,y} & \text{else} \end{cases}$$

여기서:
- t ~ U(0,1): 노이징 비율
- m은 이진 노이즈 마스크 (m_{x,y} ∈ {0,1})
- n_{x,y} ~ U(0, N-1): 코드북의 모든 인덱스에서 샘플된 임의 토큰

Step 3) 모델 예측:
$$\tilde{u} = f_\theta(\bar{u}, c, t)$$

여기서:
- f_θ: 토큰 예측 모델
- c: 조건화 임베딩
- t: 현재 타임스텝

Step 4) 손실 함수 - 교차 엔트로피 + 레이블 스무딩 적용

Step 5) 손실 가중치 스케줄 - 작은 타임스텝에서 학습 불안정성 해결:

$$lw = 1 - (1-m_{x,y}) \cdot ((1-t) \cdot (1-\eta))$$

여기서:
- η: 최소 토큰 손실 기여도 (논문에서는 η = 0.3)
- (1-m_{x,y}): 노이징되지 않은 토큰의 손실 감소

**샘플링 과정 (Algorithm 1):**[1]

입력: 모델 f_θ, 조건 c, 잠재 공간 형태 s, 노이징 비율 수열 t_1 > t_2 > ... > t_T

반복 (i = 1부터 T까지):

1단계) 모델 추론:
$$\tilde{u} = f_\theta(\hat{u}, c, t_i)$$

2단계) 분류기-자유 가이던스(CFG) 적용:
$$\tilde{u} = \tilde{u} \cdot w + f_\theta(\hat{u}, c_\emptyset, t_i) \cdot (1-w)$$

여기서:
- w: CFG 가중치
- c_∅: null 조건화

3단계) 확률 분포로 변환:
$$\tilde{u} = \text{softmax}(\tilde{u}/\tau)$$

여기서 τ는 온도 파라미터 (다양성 조절)

4단계) 다항 샘플링:
$$\hat{u} = \text{multinomial}(\tilde{u})$$

5단계) 토큰 리노이징 (i < T일 때):
$$\hat{u} = \text{renoise}(\hat{u}, t_{i+1}, u_{\text{init}})$$

**토큰 리노이징의 핵심:**[1]
- 샘플링된 토큰의 일부를 초기 노이즈 토큰 u_init으로 교체
- 교체되는 토큰 비율: t_{i+1} (현재 노이징 비율)
- **MaskGIT/MUSE와의 차이**: 신뢰도가 낮은 토큰만 마스킹하는 대신, 임의로 토큰을 리노이징

이러한 접근법의 장점:
- 모델이 이전 예측을 개선할 수 있음
- 초기 예측 오류로부터 복구 가능
- 본질적 다양성 제공 (마스크 토큰이 아닌 임의 토큰 사용)

**조건화 방식:**[1]
- ByT5-XL 임베딩 (95%): 문자 수준 인식으로 텍스트 렌더링 개선
- CLIP 텍스트 임베딩 (5%): 의미적 이해
- CLIP 이미지 임베딩 (5%): 이미지 조건화 가능

### 3. 모델 구조[1]

**토큰 예측기 아키텍처:**

1) **전체 구조**: U-Net 스타일 인코더-디코더
   - 3개 레벨 (L0, L1, L2)
   - 각 레벨은 잔차 블록들로 구성

2) **각 블록의 구성**:
   - 컨볼루션 레이어
   - 자기 어텐션 (lowest 2개 레벨에만)
   - 크로스 어텐션 (조건 임베딩과의 상호작용)
   - 레이어 정규화

3) **메모리 효율화**:
   - 패치 크기 2 사용: 공간 차원 축소, 채널 증가
   - Attention은 lowest 레벨에만 사용

4) **조건화 처리**:
   - ByT5와 CLIP 임베딩을 공유 잠재 공간으로 투영
   - CLIP 풀링 임베딩을 4개로 분리 (각 헤드가 다른 측면 학습)
   - 크로스 어텐션으로 통합

**VQGAN 설정:**[1]
- 압축률: f = 4
- 기본 해상도: 256 × 256 × 3
- 잠재 해상도: 64 × 64 × 인덱스

### 4. 성능 향상 및 한계[1]

#### 4.1 성능 향상[1]

**정량적 성능 비교 (Table 1):**

| 모델 | 파라미터 | 샘플링 스텝 | FID-COCO-30k | 오픈소스 | 데이터 공개 |
|------|---------|-----------|-------------|---------|----------|
| Paella (제안) | 1B | **12** | **11.07** | ✓ | ✓ |
| MUSE-3B | 3B | 24 | 7.78 | - | - |
| LDM-0.4B | 0.4B | 250 | 12.63 | ✓ | ✓ |
| DALL-E 2 | 3.5B | 250 | 10.39 | - | - |
| Imagen | 2B | 1000 | 7.27 | - | - |
| Parti | 20B | 1024 | 7.23 | - | - |

**효율성 측면:**[1]
- 샘플링 스텝: 12 (MUSE 24, 기존 방식 250-1000과 비교)
- 모델 크기: 1B (경쟁사의 2-20B 대비)
- 공개 가능: 소스코드 및 모델 공개

**성능 곡선:**[1]
- **FID 점수**: 12개 스텝에서 최적 (놀랍게도 더 많은 스텝이 오히려 악화)
- **CLIP 점수**: 스텝이 증가할수록 개선 (텍스트 정렬성)
- **CFG 가중치**: 낮은 값이 더 좋은 성능 (기존 관행과 반대)

#### 4.2 한계[1]

**1. 텍스트 렌더링 부족:**[1]
- 논문 저자들의 관찰: 이미지에 텍스트를 그리는 능력 부족
- 원인 가설: VQGAN의 양자화된 토큰이 세밀한 세부 정보 손실
  - 정보를 완전히 파괴하거나 보존만 가능
  - 중간 수준의 정보 손실 불가능
- 개선 방안: 더 낮은 양자화 레벨 또는 다른 토크나이저 연구 필요

**2. FID vs CLIP 스코어의 불일치:**[1]
- **관찰**: FID는 12 스텝에서 최적이나 CLIP 점수는 스텝이 많을수록 개선
- **해석**: 시각적 충실도(fidelity)는 일찍 나타나고, 조건 정렬성(alignment)은 나중에 개선됨
- **함의**: 구성 복잡성 학습이 구조 생성보다 어려움

**3. 분류기-자유 가이던스(CFG)의 역직관적 동작:**[1]
- **기존 관행**: CFG 가중치가 높을수록 좋음
- **Paella의 관찰**: 낮은 CFG 가중치가 CLIP 점수 개선에 더 효과적
- **원인 미명확**: 논문에서 명시적 설명 없음

**4. 평가 메트릭의 한계:**[1]
- FID 점수가 항상 인간 판단과 일치하지 않음
- 더 나은 평가 메트릭(CMMD 등) 고려 필요

### 5. 모델의 일반화 성능 향상 가능성[1]

#### 5.1 현재 일반화 능력[1]

**강점:**

1) **영상 크기 유연성:**[1]
   - 컨볼루션 기반 아키텍처의 장점: 임의의 해상도 생성 가능
   - 트랜스포머는 컨텍스트 윈도우 조정 필요 (Paella는 불필요)

2) **다중 조건화:**[1]
   - 텍스트만 사용: 표준 텍스트-이미지 생성
   - 이미지 임베딩 사용: 제로샷 이미지 변형
   - 텍스트+이미지: 스타일 전이, 이미지 혼합 가능

3) **낮은 압축률의 이점:**[1]
   - f=4 압축으로 세부 정보 보존
   - 높은 압축률 모델의 인공물 감소

4) **효율적 훈련:**[1]
   - 900만 이미지로 충분한 성능 달성
   - 대규모 데이터셋 필요 없음

#### 5.2 일반화 성능 개선 가능성[1]

**1. 데이터 다양성 확대:**
- 현재: LAION-5B 미학 데이터셋
- 제안: 더 다양한 도메인의 데이터 추가
- 효과: 특정 도메인에 과적합되지 않은 더 강건한 모델

**2. 토크나이저 개선:**
- 현재 문제: VQGAN의 양자화로 텍스트 렌더링 실패
- 개선안:
  - ViT-VQGAN이나 개선된 토크나이저 사용
  - 더 낮은 양자화 손실률 탐색
  - 하이브리드 양자화 방식 개발

**3. 아키텍처 확장:**
- 현재: U-Net 기반 컨볼루션
- 개선안:
  - 더 깊은 레이어 구조
  - 계층적 어텐션 (local + global)
  - Mamba 또는 하이브리드 구조 통합

**4. 샘플링 전략 개선:**
- 현재: 균일한 토큰 리노이징
- 개선안:
  - 신뢰도 기반 선택적 리노이징
  - 적응형 노이징 스케줄
  - 토큰 간 의존성 학습

**5. 조건화 메커니즘 강화:**
- 더 풍부한 조건 정보 활용
- 의미적 정렬 개선을 위한 추가 손실 함수
- 다중 모달리티 통합 최적화

**6. 평가 메트릭 개선:**
- FID 점수의 한계 극복
- CMMD(CLIP-based MMD) 등 더 나은 메트릭 도입
- 다양성-충실도 트레이드오프 더 정확한 측정

**7. 제로샷 일반화 강화:**[2][1]
- 현재: MS COCO 벤치마크에서 평가
- 개선 방향:
  - 도메인 외 데이터셋에 대한 성능 평가
  - 도메인 변이(domain shift) 최소화
  - 구성적 일반화(compositional generalization) 강화

**8. 조건부 독립성 가정 개선:**
- 현재: 마르코프 성질 가정 (많은 토큰이 조건부 독립)
- 개선: 토큰 간 의존성을 더 정확히 모델링하는 구조

#### 5.3 최신 연구와의 통합 가능성[3][4][5][6]

**1. MaskBit(2024) 통합:**[3]
- 개선된 VQGAN 적용
- 현대적 토크나이저 기술
- 더 효율적인 양자화

**2. Token-Critic 패러다임:**[7]
- 별도의 신뢰도 모델 도입
- 토큰 간 상관관계 학습
- 더 나은 샘플링 결정

**3. Self-Guidance 방법:**[6]
- 의미적 스무딩을 통한 벡터 양자화
- 샘플 품질과 다양성의 균형
- 파라미터 효율적 파인튜닝

**4. 적응형 가이던스:**[8][9]
- 동적 CFG 스케일 조정
- 주파수 도메인 가이던스
- 더 효과적한 텍스트 정렬

***

### 6. 논문의 앞으로의 연구에 미치는 영향

#### 6.1 학술적 영향[1]

**1. 아키텍처 패러다임의 변화:**
- 트랜스포머 중심에서 컨볼루션 기반 모델의 재평가
- 이차 복잡도 문제 극복의 새로운 방향 제시
- 2D 이미지 구조를 자연스럽게 처리하는 방법론 강조

**2. 샘플링 전략의 혁신:**
- 마스크 토큰 기반 방식의 한계 지적
- 토큰 리노이징의 새로운 가능성 제시
- 반복적 개선 메커니즘의 중요성 강조

**3. 효율성-성능 트레이드오프 재정의:**
- 더 적은 스텝으로도 경쟁력 있는 성능 가능 증명
- 모델 크기 축소의 가능성 제시
- 계산 효율성과 접근성의 균형점 제시

**4. 평가 메트릭에 대한 의문 제기:**
- FID 점수의 절대성 재검토
- 스텝 수와 FID의 비선형 관계 발견
- 평가 메트릭 다원화 필요성 강조

#### 6.2 산업 응용에 미치는 영향[1]

**1. 실시간 애플리케이션 가능성:**[1]
- 12 스텝으로 실용적인 속도 달성
- 엣지 디바이스에서의 실행 가능성 증가
- 모바일/임베디드 시스템 적용 가능

**2. 접근성 향상:**[1]
- 간단한 구조로 인한 이해도 향상
- 구현의 용이성
- 다양한 배경의 연구자들의 참여 유도

**3. 비용 효율성:**
- 더 적은 GPU 메모리 요구
- 훈련 시간 단축
- 상업적 배포 비용 감소

#### 6.3 미래 연구 방향[4][5][6][3][1]

**1. 토크나이저 연구:**
- 양자화 손실을 줄이는 새로운 토크나이저 개발
- 텍스트 렌더링 능력 개선
- 하이브리드 양자화 방식 탐색

**2. 하이브리드 아키텍처:**
- 컨볼루션과 어텐션의 최적 조합
- Mamba와 트랜스포머의 결합 (MaskMamba처럼)
- 계층적 구조 설계

**3. 샘플링 알고리즘:**
- 신뢰도 기반 토큰 선택 통합
- 적응형 노이징 스케줄 학습
- 토큰 간 의존성 모델링

**4. 조건화 메커니즘:**
- 더 풍부한 언어 모델 통합
- 다중 모달 조건화 최적화
- 구조적 제어(레이아웃, 세맨틱) 강화

**5. 확장 가능성:**
- 더 큰 모델 규모로의 확장 (3B, 7B)
- 더 고해상도 이미지 생성
- 비디오 생성으로의 확장

***

### 7. 앞으로 연구 시 고려할 점

#### 7.1 기술적 고려사항[5][7][6][3][1]

**1. 양자화-생성 트레이드오프:**[10]
- 높은 압축률: 계산 효율 ↑, 정보 손실 ↑
- 낮은 압축률: 정보 보존 ↑, 계산 비용 ↑
- **권장**: 각 응용 도메인에 맞는 최적 압축률 탐색

**2. FID 점수의 한계 인식:**[11]
- FID는 Inception 모델 기반으로 편향 가능
- CLIP 기반 메트릭(CMMD) 고려
- 다양한 평가 메트릭 병행 필요
- 인간 평가 포함 권장

**3. CFG의 역직관적 동작 분석:**
- Paella에서 CFG가 다르게 작동하는 원인 규명 필요
- 모델 아키텍처에 따른 CFG 동작 차이 체계적 연구
- 최적 CFG 스케줄 학습 메커니즘 개발

**4. 텍스트 렌더링 문제 해결:**
- 양자화 수준 최적화
- 토크나이저 구조 개선
- 텍스트 렌더링 특화 손실 함수 개발

#### 7.2 데이터 및 평가 고려사항[12][2][1]

**1. 데이터 편향 최소화:**[12]
- 지역적, 인구통계학적 다양성 확보
- 성별, 피부색, 민족성 등에 대한 균형
- 공정성-정확성 트레이드오프 고려

**2. 일반화 성능 평가:**[2]
- 훈련 데이터와 다른 도메인 테스트
- 구성적 일반화(새로운 객체 조합)
- 도메인 외(out-of-distribution) 강건성 평가

**3. 장기 다양성 유지:**
- 반복적 평가로 모델 성능 열화 감지
- 계절, 트렌드 변화에 따른 성능 모니터링

#### 7.3 방법론적 고려사항[7][6][3][1]

**1. 비교 실험의 엄밀성:**
- 동일 하드웨어에서의 공정한 비교
- 하이퍼파라미터 튜닝의 충분성 확보
- 통계적 유의성 검정

**2. 샘플링 전략 체계적 분석:**
- 토큰 리노이징 비율의 최적화
- 신뢰도 기반 선택과의 상세 비교
- 토큰 간 의존성 분석

**3. 모듈성 평가:**
- 각 컴포넌트(토크나이저, 아키텍처, 샘플링)의 기여도 측정
- 제거 실험(ablation studies)의 체계성
- 독립적 개선 방향 도출

#### 7.4 실무적 고려사항

**1. 재현성 확보:**[1]
- 소스코드 공개 (Paella는 MIT 라이선스로 공개)
- 상세한 훈련 로그 및 하이퍼파라미터 공개
- 재현 불가능 요인 명확히 문서화

**2. 확장성 검증:**
- 다양한 이미지 크기에 대한 성능 평가
- 다양한 조건화 방식 호환성 검증
- 추가 모달리티(3D, 비디오 등)로의 확장 가능성

**3. 비용-편익 분석:**
- 훈련 비용 vs 성능 향상
- 추론 시간 vs 품질 트레이드오프
- 상업적 실행 가능성 평가

***

### 8. 2020년 이후 관련 최신 연구 탐색[13][14][15][16][17][18][19][20][4][5][6][3][7]

#### 8.1 마스크 기반 생성 모델의 발전[15][21][16][13]

**MaskGIT (2022)**[21][13][15]
- **핵심**: 양방향 트랜스포머 + 마스크 기반 반복 생성
- **방식**: 마스크 토큰 사용, 신뢰도 기반 토큰 선택
- **성능**: ImageNet 256×256에서 SOTA 달성, 추론 48배 가속
- **영향**: 비자동회귀 병렬 디코딩의 가능성 증명

**MUSE (2023)**[14][16]
- **핵심**: 텍스트-이미지 생성을 위한 마스크 트랜스포머
- **특징**: 대형 언어모델(T5) 활용, 병렬 디코딩
- **성능**: 900M 모델로 FID 6.06 (CC3M), FID 7.88 (COCO zero-shot)
- **효율성**: 24 샘플링 스텝
- **제한**: Paella와 유사한 마스크 토큰 방식의 한계

#### 8.2 토큰 선택 최적화[17][18][22][7]

**Token-Critic (2022)**[23][7]
- **혁신**: 추가 critic 모델로 토큰 신뢰도 학습
- **방식**: Generator가 예측, Critic이 품질 판단, 반복적 개선
- **성능**: MaskGIT 대비 상당한 성능 향상
- **의의**: 토큰 선택의 독립성 가정 극복

**Enhanced Sampling Scheme (2023)**[18]
- **목표**: TimeVQVAE, MaskGIT, Token-Critic의 한계 극복
- **방식**: 3단계 샘플링 (초기 생성 → 개선 → 최적화)
- **특징**: 명시적 다양성과 충실도 보장

**Self-Guidance (2024)**[6]
- **개념**: 벡터 양자화 토큰 공간에서 의미적 스무딩
- **기술**: 보조 작업을 통한 세맨틱 부드러움
- **결과**: 품질-다양성 최적 균형

#### 8.3 아키텍처 혁신[24][4][17][5][3]

**MaskBit (2024)**[3]
- **개선**: 현대화된 VQGAN + 마스크 변환기
- **특징**: 임베딩 없는 구조, Bit 토큰 사용
- **성능**: 개선된 생성 품질 및 표현 학습
- **활용**: 제로샷 이미지 인페인팅, 아웃페인팅, 편집

**MaskMamba (2024)**[4]
- **혁신**: Mamba + Transformer 하이브리드
- **장점**: 이차 복잡도 문제 해결, 효율성 개선
- **구조**: 양방향 Mamba (표준 컨볼루션 사용)
- **의의**: 효율적이면서도 강력한 아키텍처

**Efficient-VQGAN (2023)**[19]
- **목표**: 고해상도 이미지 생성 효율화
- **개선**:
  1. 첫 단계: 로컬 어텐션 기반 인코더-디코더
  2. 두 번째 단계: 다중 그레인 어텐션 + 자동회귀 생성
- **성능**: 효율성과 품질 모두 개선

#### 8.4 샘플링 전략의 고도화[25][26][27][28][29][30]

**Text-Conditioned Token Selection (2023)**[26][20]
- **방식**: 텍스트 정보 기반 토큰 선택
- **기술**: 지역화된 감독 + 주파수 적응 샘플링
- **결과**: 추론 시간 50% 단축, 텍스트 정렬성 개선

**Path Planning for Masked Diffusion (2025)**[28]
- **개념**: 토큰 언마스킹 순서의 최적화
- **기술**: 확장 ELBO + 사전학습 BERT/디노이저 플래너
- **결과**: 대안적 언마스킹 전략으로 성능 향상

**Remasking Discrete Diffusion (2025)**[30]
- **혁신**: 생성 후 토큰 재마스킹 가능
- **의의**: 이전 결정의 수정 가능 (기존 마스크 기반 방식의 한계 극복)

#### 8.5 일반화 성능 향상 연구[31][32][12][2]

**Bias-Free Training Paradigm (2024)**[12]
- **문제**: AI 생성 이미지 감지의 낮은 일반화
- **해결**: Stable Diffusion 조건화 프로세스 활용, 의미적 정렬
- **결과**: 27개 생성 모델에 걸친 강건성 개선

**AnySynth (2024)**[31]
- **목표**: 일반화된 합성 데이터 생성
- **특징**: 작업별 레이아웃 생성, 통일된 이미지 생성, 맞춤형 어노테이션
- **응용**: 적은샷 객체 감지, 도메인 간 일반화

**Compositional Generalization in Diffusion (2025)**[32]
- **연구**: 길이 일반화(더 많은 객체 생성 능력)
- **발견**: 조건부 프로젝티브 구성과 국소적 조건부 점수의 등가성
- **실험**: CLEVR 환경에서 인과 개입으로 생성화 복원

#### 8.6 평가 메트릭의 진화[33][11]

**CMMD (CLIP-based MMD) (2024)**[11]
- **문제**: FID의 한계
  1. Inception 모델의 약한 임베딩
  2. 정규 분포 가정의 부정확성
- **해결**: CLIP 임베딩 + 최대 평균 불일치(MMD)
- **장점**: 분포 가정 없음, 더 풍부한 특성 표현

**생성 모델 평가 프레임워크 확대 (2023-2025)**[34][33]
- **지표**: FID, CLIP 점수, Precision, Recall, Inception Score
- **최신**: GenEval, HPS 점수 추가
- **방향**: 다차원적 평가 (충실도, 정렬성, 다양성)

#### 8.7 조건화 메커니즘의 발전[35][9][36][37][8]

**분류기-자유 가이던스(CFG) 개선들:**

**CFG++ (2024)**[38]
- **혁신**: 매니폴드 제약 조건 추가
- **개선**: 더 부드러운 생성 궤적, 높은 가이던스 스케일에서 품질 유지
- **특징**: 역변환성 개선, 더 낮은 가이던스 스케일 가능

**빈도 도메인 가이던스 (2025)**[9]
- **분석**: 저주파 vs 고주파의 역할 분리
- **방식**: 각각에 다른 가이던스 강도 적용
- **결과**: FID와 회상율 개선

**적응형 가이던스 스케줄 (2025)**[37]
- **발견**: 초기 샘플링 단계의 불안정성
- **해결**: 비율 기반 적응형 가이던스 스케줄
- **성능**: 3배 빠른 샘플링 가능, 품질 유지

***

### 결론

**"A Novel Sampling Scheme for Text- and Image-Conditional Image Synthesis in Quantized Latent Spaces"**는 텍스트-이미지 생성 분야에서 **단순성, 효율성, 접근성**의 삼각형을 이루는 혁신적인 접근을 제시합니다.[1]

핵심적 기여는:
1. **토큰 리노이징** 메커니즘으로 MaskGIT/MUSE의 마스크 토큰 방식의 한계 극복[1]
2. **컨볼루션 아키텍처**로 트랜스포머의 이차 복잡도 문제 해결[1]
3. **12 스텝 샘플링**으로 효율성-품질의 새로운 균형점 제시[1]
4. **낮은 압축률**로 세부 정보 보존[1]

일반화 성능은 **현재 강하면서도 개선 여지가 크다**고 평가됩니다. 토크나이저 개선, 하이브리드 아키텍처, 지능형 샘플링 전략을 통해 텍스트 렌더링, 구성적 일반화, 도메인 외 강건성을 상당히 향상시킬 수 있습니다.

향후 연구는 **양자화-생성 트레이드오프 최적화, FID 점수 한계 극복, CFG의 역직관적 동작 분석** 등을 중점적으로 진행해야 하며, **최신 기술(MaskBit, MaskMamba, Self-Guidance, 적응형 CFG)의 통합**을 통해 이 접근법의 잠재력을 완전히 발휘할 수 있을 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0885d245-720c-490e-93cd-24fe26178981/2211.07292v2.pdf)
[2](https://arxiv.org/abs/2503.10125)
[3](https://arxiv.org/html/2409.16211)
[4](https://arxiv.org/abs/2409.19937)
[5](https://ieeexplore.ieee.org/document/10378581/)
[6](https://arxiv.org/abs/2410.13136)
[7](https://arxiv.org/abs/2209.04439)
[8](https://ieeexplore.ieee.org/document/11094644/)
[9](https://arxiv.org/abs/2506.19713)
[10](https://arxiv.org/html/2412.16326v1)
[11](https://openaccess.thecvf.com/content/CVPR2024/papers/Jayasumana_Rethinking_FID_Towards_a_Better_Evaluation_Metric_for_Image_Generation_CVPR_2024_paper.pdf)
[12](https://ieeexplore.ieee.org/document/11095256/)
[13](https://arxiv.org/abs/2202.04200)
[14](https://arxiv.org/pdf/2301.00704.pdf)
[15](https://www.emergentmind.com/topics/masked-generative-image-transformer-maskgit)
[16](https://arxiv.org/abs/2301.00704)
[17](https://arxiv.org/abs/2509.22925)
[18](https://arxiv.org/pdf/2309.07945.pdf)
[19](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_Efficient-VQGAN_Towards_High-Resolution_Image_Generation_with_Efficient_Vision_Transformers_ICCV_2023_paper.pdf)
[20](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Text-Conditioned_Sampling_Framework_for_Text-to-Image_Generation_with_Masked_Generative_Models_ICCV_2023_paper.pdf)
[21](https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.pdf)
[22](https://papers.nips.cc/paper_files/paper/2024/file/ecd92623ac899357312aaa8915853699-Paper-Conference.pdf)
[23](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830070.pdf)
[24](https://ieeexplore.ieee.org/document/11091878/)
[25](https://arxiv.org/html/2503.17076v1)
[26](https://ieeexplore.ieee.org/document/10377453/)
[27](https://arxiv.org/abs/2510.12231)
[28](https://arxiv.org/html/2502.03540)
[29](https://arxiv.org/pdf/2411.15746.pdf)
[30](https://arxiv.org/html/2503.00307v1)
[31](https://arxiv.org/abs/2411.16749)
[32](https://arxiv.org/abs/2502.14007)
[33](https://softwaremill.com/evaluation-metrics-for-generative-image-models/)
[34](https://huggingface.co/docs/diffusers/en/conceptual/evaluation)
[35](https://www.emergentmind.com/topics/text-to-image-diffusion-models)
[36](https://arxiv.org/abs/2504.13987)
[37](https://arxiv.org/abs/2508.03442)
[38](https://pure.kaist.ac.kr/en/publications/cfg-manifold-constrained-classifier-free-guidance-for-diffusion-m/)
[39](http://pubs.rsna.org/doi/10.1148/radiol.233529)
[40](https://eurradiolexp.springeropen.com/articles/10.1186/s41747-024-00485-7)
[41](https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs)
[42](https://arxiv.org/pdf/2410.00483.pdf)
[43](https://arxiv.org/html/2410.13136v1)
[44](https://arxiv.org/html/2504.06897v1)
[45](https://www.reddit.com/r/MachineLearning/comments/1e37ymt/d_r_in_vqgan_after_quantization_how_is_an_image/)
[46](https://arxiv.org/abs/2502.19716)
[47](https://research.google/blog/vector-quantized-image-modeling-with-improved-vqgan/)
[48](http://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DesignDiffusion_High-Quality_Text-to-Design_Image_Generation_with_Diffusion_Models_CVPR_2025_paper.pdf)
[49](https://arxiv.org/html/2310.05400)
[50](https://liner.com/review/maskgit-masked-generative-image-transformer)
[51](https://arxiv.org/abs/2410.14672)
[52](https://link.springer.com/10.1007/978-3-031-20050-2_11)
[53](http://arxiv.org/pdf/2404.08327.pdf)
[54](https://arxiv.org/pdf/2401.00254.pdf)
[55](https://arxiv.org/pdf/2306.07346.pdf)
[56](https://www.nature.com/articles/s41598-025-90616-w)
[57](https://openaccess.thecvf.com/content/CVPR2024/papers/Jia_Generative_Latent_Coding_for_Ultra-Low_Bitrate_Image_Compression_CVPR_2024_paper.pdf)
[58](https://arxiv.org/html/2511.12032v1)
[59](https://cs231n.github.io/convolutional-networks/)
[60](https://www.sciencedirect.com/science/article/abs/pii/S0031320325005084)
[61](https://arxiv.org/html/2503.07197v2)
[62](https://ieeexplore.ieee.org/document/10743595/)
[63](https://ieeexplore.ieee.org/document/10852217/)
[64](https://arxiv.org/abs/2506.21722)
[65](https://arxiv.org/abs/2407.07860)
[66](https://arxiv.org/abs/2509.16447)
[67](https://dl.acm.org/doi/10.1145/3664647.3680797)
[68](http://arxiv.org/pdf/2403.14370.pdf)
[69](https://arxiv.org/html/2503.12652)
[70](https://arxiv.org/html/2410.02667v1)
[71](https://arxiv.org/pdf/2209.02646.pdf)
[72](https://arxiv.org/html/2503.06132v1)
[73](https://arxiv.org/pdf/2305.18455.pdf)
[74](https://arxiv.org/html/2312.02548)
[75](https://arxiv.org/abs/2408.08306)
[76](https://arxiv.org/html/2503.12652v1)
[77](https://openreview.net/pdf?id=1n1c7cHl3Zc)
[78](https://arxiv.org/html/2502.19716v1)
[79](https://laion.ai/blog/paella/)
[80](https://openaccess.thecvf.com/content/ICCV2025/papers/Fu_UniVG_A_Generalist_Diffusion_Model_for_Unified_Image_Generation_and_ICCV_2025_paper.pdf)
[81](https://velog.io/@sjinu/%EA%B0%84%EB%8B%A8%EC%A0%95%EB%A6%AC-Hierarchical-Text-Conditional-Image-Generation-with-CLIP-LatentsDALL-E2)
[82](https://openreview.net/forum?id=57THeGgNAN)
[83](https://arxiv.org/abs/2502.03726)
[84](https://arxiv.org/abs/2507.18192)
[85](https://www.semanticscholar.org/paper/c2a8ca2ab70ac10725b9660f51ce48e94f98e7c6)
[86](https://arxiv.org/abs/2403.17377)
[87](https://arxiv.org/abs/2506.24108)
[88](https://www.semanticscholar.org/paper/a95388eb8c623b28d23ebef4e068bf3ac067e1f4)
[89](https://arxiv.org/pdf/2311.00938.pdf)
[90](https://arxiv.org/html/2502.07849)
[91](http://arxiv.org/pdf/2406.08070.pdf)
[92](https://arxiv.org/html/2503.17593v1)
[93](https://arxiv.org/html/2503.18886v1)
[94](https://arxiv.org/html/2411.17077)
[95](http://arxiv.org/pdf/2306.00986.pdf)
[96](https://arxiv.org/html/2410.09347v1)
[97](https://www.doptsw.com/posts/post_2024-09-17_05c95f)
[98](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08254.pdf)
[99](https://www.sciencedirect.com/science/article/abs/pii/S0952197625001514)
[100](https://arxiv.org/html/2402.14095v1)
[101](https://proceedings.neurips.cc/paper_files/paper/2024/file/dd540e1c8d26687d56d296e64d35949f-Paper-Conference.pdf)
[102](https://proceedings.mlr.press/v202/chang23b/chang23b.pdf)
[103](https://openaccess.thecvf.com/content/WACV2025/papers/Yu_Image-Caption_Encoding_for_Improving_Zero-Shot_Generalization_WACV_2025_paper.pdf)
[104](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/cfdg/)
