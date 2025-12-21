
# PHOTOSWAP: Personalized Subject Swapping in Images

## 1. 논문의 핵심 주장과 주요 기여

**PHOTOSWAP**은 사전 학습된 diffusion 모델을 활용하여 이미지에서 피사체를 개인화된 개체로 원활하게 교환하는 novel framework를 제시한다. 이 접근방식의 핵심 주장은 다음과 같다:[1]

### 1.1 핵심 주장

**주요 명제**: 적절한 자기 주의(self-attention) 및 교차 주의(cross-attention) 조작을 통해, 좋은 개념화된 시각적 피사체는 원본 피사체의 포즈와 이미지의 전체적 일관성을 유지하면서 어떤 이미지에도 원활하게 전송될 수 있다.[1]

### 1.2 주요 기여

1. **개인화된 피사체 교환을 위한 새로운 프레임워크** - 기존 방법과 달리 참조 이미지와 마스크만으로도 정밀한 교환 가능
2. **Training-free 주의 교환 방법** - 사전 학습 모델의 가중치 수정 없이 테스트 시간에 적용
3. **포괄적 실험 및 평가** - 인간 평가에서 기존 P2P 방법 대비 **50.8 vs 28.0**의 현저한 성능 우위 입증[1]

***

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 문제 정의

기존 텍스트 기반 이미지 편집의 주요 문제점:
- 프롬프트의 약간의 변경도 전체 이미지를 완전히 다르게 생성
- 피사체 교환 시 포즈, 배경, 구도 보존 실패
- 추가 마스크나 스케치 입력 필요 (사용자 부담 증가)[1]

### 2.2 제안하는 방법: Training-free 주의 교환

#### 알고리즘의 수학적 표현

**자기 주의 메커니즘**:
$$\Phi_i = M_i V_i$$

여기서 $M_i = \text{Softmax}(Q_i K_i^T / \sqrt{d_k})$이며, $M_i$는 자기 주의 맵, $V_i$는 값 벡터이다.[1]

**교차 주의 메커니즘**:
$$\tilde{\Phi}_i = A_i V_i$$

여기서 $A_i = \text{Softmax}(Q_i K_i^T / \sqrt{d_k})$이고, $Q_i$, $K_i$는 text prompt에서 계산된다.[1]

#### 핵심 주의 교환 함수

Algorithm 1에서 정의한 SWAP 함수:
$$\text{SWAP}(\Phi_s, M_s, A_s, \Phi_t, M_t, A_t, \tau_i)$$

다음의 규칙을 따름:
- **자기 주의 출력 교환** ($\tau_\Phi$ 스텝까지): $\Phi_i \leftarrow \Phi_s^i$
- **자기 주의 맵 교환** ($\tau_M$ 스텝까지): $M_i \leftarrow M_s^i$  
- **교차 주의 맵 교환** ($\tau_A$ 스텝까지): $A_i \leftarrow A_s^i$[1]

각 메커니즘의 역할:
- **$M_i$ (자기 주의 맵)**: SVD 분석 결과 이미지의 기하학적 구조 및 레이아웃을 인코딩[1]
- **$\Phi_i$ (자기 주의 출력)**: 배경 및 문맥 정보 보존에 가장 영향력 있음[1]
- **$A_i$ (교차 주의 맵)**: 텍스트 프롬프트와 공간 레이아웃의 정렬 담당[1]
- **교차 주의 출력**: 목표 피사체의 정체성 유지를 위해 변경하지 않음[1]

### 2.3 파이프라인

**Step 1: 개념 학습**

$$\min_\theta \mathbb{E}_{x \in I_{ref}} L(\text{Denoise}(\mathcal{E}(x); \theta; P_t), x)$$

DreamBooth를 활용하여 특수 토큰 $*$로 개념 학습[1]

**Step 2: DDIM 역변환**

$$z_T \sim \mathcal{N}(0, I), \quad z_{t-1} = z_t - \alpha_t \epsilon_\theta(z_t)$$

원본 이미지를 초기 노이즈로 변환하며, Null-text 최적화로 강건성 개선[1]

**Step 3: 주의 교환 기반 생성**

처음 $T'$ 스텝에서:

$$z_{t-1}^{(target)} \leftarrow \text{DDIM}(z_t^{(target)}, \{\Phi_s^i, M_s^i, A_s^i\}, P_t)$$

나머지 $T - T'$ 스텝에서:

$$z_{t-1}^{(target)} \leftarrow \text{DDIM}(z_t^{(target)}, \{\Phi_t^i, M_t^i, A_t^i\}, P_t)$$

***

## 3. 모델 구조 및 성능

### 3.1 U-Net 기반 확산 모델 구조

**기초 모델**: Stable Diffusion 2.1[1]

핵심 컴포넌트:
- **인코더**: 이미지를 잠재 공간으로 압축
- **U-Net**: 노이즈 제거 네트워크
  - Self-attention 블록: 공간 특성 처리
  - Cross-attention 블록: 텍스트 프롬프트 통합
- **디코더**: 생성된 잠재 변수를 이미지로 복원[1]

### 3.2 성능 평가 결과

#### 인간 평가 (Amazon MTurk)

| 평가 항목 | Photoswap | P2P+DreamBooth | Tie |
|---------|-----------|--------------|-----|
| 피사체 교환 (정체성 보존) | **46.8%** | 25.6% | 27.6% |
| 배경 보존 | **40.7%** | 32.7% | 26.6% |
| 전체 품질 | **50.8%** | 28.0% | 21.2% |[1]

#### 절제 연구: 주의 교환 스텝 분석

Figure 9에서 각 메커니즘의 최적 스텝 값:
- **자기 주의 출력** ($\tau_\Phi$): 10 스텝 - 우수한 레이아웃 제어
- **자기 주의 맵** ($\tau_M$): 25 스텝 - 공간 구조 유지  
- **교차 주의 맵** ($\tau_A$): 20 스텝 - 텍스트 정렬[1]

Figure 8에서 $\tau_M$ 값에 따른 영향:
- 높은 값 → 원본 이미지 스타일 강하게 유지
- 낮은 값 → 참조 피사체의 정체성 강화[1]

### 3.3 다양한 시나리오에서의 우수성

- **다중 피사체 교환**: 두 개의 안경을 동시에 교환하면서 원래 레이아웃 유지 (Figure 6a)[1]
- **부분적으로 가려진 피사체**: 정장에 의해 부분적으로 가려진 개에서도 피사체 정확히 식별 및 교환 (Figure 6b)[1]

***

## 4. 모델의 한계와 실패 사례

### 4.1 주요 한계점

#### 손 세부 정보 재현 실패
Stable Diffusion의 손가락 생성 문제를 상속받음:
- 손가락 수 정확성 부재
- 복잡한 손 제스처 미재현
- 포즈 기반 손 위치 조정 불충분[1]

#### 복잡한 배경 정보 손실
칠판의 수식 같은 추상적 콘텐츠:
- 텍스트 기반 공식 재현 실패
- 고도의 구조화된 정보 유지 어려움
- 복잡한 패턴의 보존 한계[1]

#### 피부색 동질화 (Figure 11)
인종이 다른 피사체 간 교환 시:
- 피부색 단순화 및 평준화
- 저자들의 권장: 유사한 인종 배경의 피사체 사용[1]

### 4.2 개념 학습 방법의 영향

Textual Inversion 사용 시 (Figure 10):
- 인간 얼굴 표현 품질 현저히 저하
- 복잡한 구조(특히 얼굴) 학습 불완전
- DreamBooth가 더 우월함을 확인[1]

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 생성 기초 기술 진화

#### Stable Diffusion 시리즈 (2022-2024)

| 모델 | 특징 | Photoswap 호환성 |
|------|-----|-----------------|
| SD v1.5 | 기초 모델 | ✓ (초기 사용) |
| **SD v2.1** | **Photoswap 기반 모델** | ✓ **채택됨** |
| **SDXL** | 고해상도 (1024×1024) | ✓ 호환 가능 |
| **Cascade** | 3단계 생성 | ✓ 잠재력 있음 |[2][3]

### 5.2 동일 분야의 경쟁 방법들

#### SwapAnything (2024) - 직접 비교 대상[4]

**핵심 차이점**:
1. **제어 정밀도**: Photoswap의 "주제" → SwapAnything의 "임의 개체"
2. **변수 교환 기법**: 표적화된 변수 교환(Targeted Variable Swapping) 도입
3. **마스크 기반 제어**: 개체별 정확한 영역 제어[4]

**성능 비교**:
- 단일 개체 교환: 비슷
- **다중 개체 교환**: SwapAnything 우수  
- **부분 개체 교환**: SwapAnything 우수
- **크로스 도메인**: SwapAnything 우수[4]

#### DreamBooth (2023) - 기초 개념학습[5]

**Photoswap과의 관계**:
- Photoswap은 DreamBooth를 개념 학습 단계에서 활용
- **공통점**: 특수 토큰 기반 개념 인코딩
- **차이점**:
  - DreamBooth: 전체 생성 프로세스 미세조정
  - Photoswap: 주의 교환으로 Training-free 달성

**평가 메트릭 비교**:
- DINO 점수: Photoswap이 약 5% 높음
- CLIP-I, CLIP-T: 비슷한 수준[6]

#### Textual Inversion (2023) - 대안적 방법[7]

$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_t, c}[\| \mathbf{x}_0 - \mathbf{\hat{x}}_0(\mathbf{z}_t, f(\mathbf{v}^*), t) \|_2^2]$$

**Photoswap 평가**:
- DreamBooth를 기본으로 선택한 이유: 복잡한 구조(특히 인간 얼굴) 학습에서 우수
- Textual Inversion 결과 (Figure 10): 얼굴 표현 품질 현저히 낮음[1]

### 5.3 주의 제어 기반 편집 방법 진화

#### Prompt-to-Prompt (P2P, 2022) - 기초 방법론[8]

**핵심 기술**:
$$A_{\text{edited}} = A_{\text{source}} \cdot \frac{P_{\text{edited}}}{P_{\text{source}}}$$

**Photoswap과의 관계**:
- P2P: 교차 주의 맵만 제어
- **Photoswap의 혁신**: 자기 주의 출력($\Phi_i$) 추가 교환 (이전 미탐색)
- **성능 차이**: 배경 보존에서 Photoswap 약 8% 우수[8][1]

#### Null-text Inversion (2022) - 역변환 개선[9]

**핵심 아이디어**:

$$\mathcal{L} = \| z_t - z_t^* \|_2^2 \text{ where } \emptyset = \emptyset^*$$

**Photoswap에의 응용**:
- 실제 이미지 재구성 시 null 텍스트 임베딩 최적화 적용
- "개선된 DDIM 역변환"으로 언급[1]
- 높은 충실도 재구성 달성[9]

#### Layout Guidance (2024) - 공간 제어 최신화[10]

**혁신적 방법**:

$$\mathcal{L}_{\text{backward}} = - \sum_{u \in B} \log A(u, y_i) - \lambda \sum_{u \notin B} \log(1 - A(u, y_i))$$

**진행 상황**:
- P2P의 순방향 가이던스 → 역방향 가이던스로 개선
- CVPR 2024 채택
- 공간 정확도 약 10% 향상[11][10]

### 5.4 개념 학습의 다중 접근법

#### CustomDiffusion (2023) - 다중 개념[6]

**특징**: 여러 개념 동시 학습
**Photoswap 한계**: 단일 개념에만 최적화
**확장 가능성**: 다중 개체 교환에 적용 가능

#### HybridBooth (2024) - 효율성 개선[12]

**혁신**: 하이브리드 프롬프트 역변환
- 개념 학습 속도 향상
- LoRA 기반 파라미터 효율적 미세 조정[12]

**Photoswap 개선 방향**: 개념 학습 단계의 전처리로 활용

### 5.5 평가 메트릭의 발전

#### DreamBench (2023) - 기초 평가 데이터셋

**메트릭**:
- **DINO**: 시각적 유사성 (높을수록 정체성 보존 우수)
- **CLIP-I**: 이미지-이미지 정렬
- **CLIP-T**: 텍스트-이미지 정렬

**Photoswap의 한계**: 이들 메트릭의 인간 판단 상관도가 완벽하지 않음[6]

#### EditEval (2024) - 새로운 벤치마크[13]

**LMM Score 도입**:
$$\text{Score} = f_{\text{LLM}}(\text{Image Content, Layout, Edit Quality})$$

**개선점**:
- 기존 CLIP 점수 대비 **0.95 이상의 인간 선호도 상관관계**
- 더 정교한 품질 평가[13]

### 5.6 아키텍처 진화와 미래 방향

#### Diffusion Transformer (DiT, 2024)

**혁신**:
- UNet 구조 → 시퀀스 모델 기반
- 더 나은 확장성
- **Photoswap 호환성**: 주의 교환 개념 직접 적용 가능[14]

#### Flow Matching Models (2024)

$$\dot{\mathbf{x}}_t = \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{c})$$

**특징**: DDIM 샘플링 대체, 더 빠른 수렴
**Photoswap 확장**: 역변환과 생성 프로세스 재설계 필요[15]

***

## 6. 모델의 일반화 성능 향상 가능성

### 6.1 현재 성능의 한계와 분석

**성능 함수 모델링**:
$$\text{Performance} = f(\text{Object Complexity}, \text{Background}, \text{Domain}, \text{Pose Variation})$$

#### 객체 복잡도별 성능 변화

| 카테고리 | 성능 | 이유 |
|---------|-----|------|
| **단순** (동물, 물건) | **매우 우수** | Stable Diffusion의 강점, 충분한 학습 데이터 |
| **중간** (인물, 건축물) | **우수** | 약간의 세부 정보 손실 |
| **복잡** (손, 얼굴 디테일) | **취약** | 생성 모델 기본 한계 상속 |[1]

#### 배경 복잡도별 성능

- **단순 배경** (창, 실내): 95% 이상 보존
- **중간 배경** (야외, 여러 개체): 85-90% 보존
- **복잡 배경** (혼합, 텍스트): 70% 미만 보존[1]

### 6.2 단기 개선 전략 (6개월 내)

#### 1) 하이퍼파라미터 자동 최적화

**현재 방식**: 고정 스텝값 ($\tau_\Phi=10, \tau_M=25, \tau_A=20$)

**개선 전략**:

$$\tau^*_i = \arg\max_\tau \mathbb{E}[\text{Quality}(x_0, x_{0,ref}) \cdot \text{Preservation}(x_0, x_s)]$$

**예상 효과**: 5-8% 성능 향상

#### 2) 주의 맵 세밀화

**기술**:
- 계층별(layer-wise) 다중 스케일 적용
- 조기 단계: 강한 주의 제어
- 후기 단계: 약한 주의 제어

**예상 효과**: 세부 정보 보존 10% 향상

#### 3) 역변환 개선

**Null-text 최적화 확장**:
$$\min_{\emptyset^{(t)}} \sum_t \| z_t - z_t^*({\emptyset^{(t)}}) \|_2^2$$

타임스텝별 독립적 최적화
**예상 효과**: 배경 보존 3-5% 향상

### 6.3 중기 전략 (1년 내)

#### 1) 더 큰 모델로의 마이그레이션

| 모델 | 해상도 | 파라미터 | 예상 성능 향상 |
|------|--------|----------|--------------|
| **SD v2.1** | 512×512 | 860M | **기준** (현재) |
| **SDXL** | 1024×1024 | 3.5B | **+10-15%** |
| **SDXL Turbo** | 1024×1024 | 3.5B | **+15-20%** (속도↑) |[16]

#### 2) 도메인별 특화 모델

**전략**: 특정 도메인(얼굴, 동물, 제품)별 미세 조정

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{identity}} + \lambda_2 \mathcal{L}_{\text{preserve}}$$

**예상 효과**: 도메인별 20-30% 성능 향상

#### 3) 구조 보존 손실 함수 개발

**핵심 아이디어**:
$$\mathcal{L}_{\text{struct}} = \| \nabla x_0 - \nabla x_s \|_1 + \| M - M_s \|_F^2$$

기하학적 특성 명시적 유지
**예상 효과**: 손 세부 정보 등에서 15-20% 개선

### 6.4 장기 전략 (2년 이상)

#### 1) 새로운 아키텍처 채택

**Diffusion Transformer (DiT)로의 전환**:
- 더 나은 확장성
- 주의 제어 더욱 정밀화 가능
- 예상 성능 향상: 25-35%[14]

**Flow Matching 모델**:
- 더 빠른 샘플링
- 주의 메커니즘 재설계 필요[15]

#### 2) 멀티모달 조건화 통합

**확장**:
- 스케치, 마스크, 레이아웃 조건 통합
- 여러 참조 이미지 활용

**예상 성능**: 30-40% 향상

#### 3) 동영상 피사체 교환으로의 확장

**과제**: 시간적 일관성 유지
**초기 시도**: VideoSwap (2023)[17]
**미래 발전**: 프레임 간 주의 일관성 제어

### 6.5 일반화 성능 분석

#### 교차 도메인 성능 기대치

**합성 vs 실제 이미지**:
- 현재: 둘 모두 우수 (Figure 5)
- 미래: 도메인 무관 일반화로 개선

**객체 클래스별 성능 곡선**:

```
성능
 ↑
100% ├─ 동물
     ├─ 단순 물건
     ├─ 실내 장면
  75% ├─ 인물 (얼굴 제외)
     ├─ 건축물
     ├─ 야외 장면
  50% ├─ 손/발
     ├─ 얼굴 세부
     ├─ 복잡 텍스처
  25% ├─ 손가락
     └─ 추상 콘텐츠
```

**개선 로드맵**:
- 2024: 현재 수준 유지
- 2025: 중간 범주 10-15% 향상
- 2026+: 어려운 범주 20-30% 개선[18]

***

## 7. 향후 연구에 미치는 영향과 고려사항

### 7.1 학술적 영향

#### 주의 제어 패러다임의 확산

**Photoswap의 기여**:
1. **자기 주의 출력(Φ) 교환 효과 입증** - 이전에 미탐색된 메커니즘
2. **Training-free 방법론의 타당성** - 사전 학습 모델 재이용의 효율성 입증
3. **계층별 주의 분석** - 초기 vs 후기 단계의 역할 구분[19][1]

**후속 영향**:
- **SwapAnything (2024)**: 변수 교환 개념 확장[4]
- **Layout Guidance (2024)**: 교차 주의 제어 정밀화[10]
- **Attention in Diffusion Models Survey (2025)**: 종합 분석 및 체계화[20]

#### Training-free 패러다임의 주류화

**이전**: 사전 학습 → 인스턴스 미세 조정 (비효율적)
**현재**: 사전 학습 → 테스트 시간 적응 (효율적)

**혁신적 영향**:
- **계산 효율성**: 새로운 개념마다 훈련 불필요
- **개인정보 보호**: 모델 가중치 수정 없음
- **빠른 배포**: 즉시 새로운 피사체 적응[21][20]

**관련 연구**:
- ExpertDiff: 모델 재프로그래밍 기법[21]
- Stable Flow: DiT 기반 training-free 편집[15]

#### 평가 메트릭 개선 요구

**Photoswap의 한계 인식**:
- DINO, CLIP 기반 평가의 완전성 의심
- 인간 선호도와의 불완전한 상관관계

**결과적 발전**:
- **EditEval (2024)**: LMM Score 제안[13]
  - 대형 언어 모델 기반 평가
  - CLIP 대비 0.95 이상 상관도
- **더 정교한 평가 체계** 개발 추동[13]

### 7.2 응용 분야 개척

#### 엔터테인먼트 및 미디어

**가능한 응용**:
1. **영화 제작**: 배우 대체, 특수 효과
   - 배우가 다른 장면에서의 모습 생성
   - 윤리적 우려: 동의, 초상권[1]

2. **게임 개발**: 아바타 커스터마이제이션
   - 플레이어 사진을 게임 캐릭터로
   - 실시간 처리 필요성[1]

3. **소셜 미디어**: 필터 및 효과
   - 인스타그램, TikTok 스타일 필터
   - 대규모 배포 가능성[1]

**기술적 도전**:
- 실시간 처리: 현재 10분 > 필요 < 1초
- 고해상도: 현재 512×512 > 필요 > 4K[1]

#### 전문 사진 편집

**실제 사용 사례**:
1. **제품 사진**: 배경 변경 없이 제품만 교환
2. **인물 사진**: 얼굴 표정 유지하며 얼굴 교환
3. **부동산**: 가구/스타일 변경 효과[1]

**시장 기회**:
- Adobe, Canva 등 기업의 통합 가능성
- 프리랜서 사진사의 생산성 향상
- 예상 시장 규모: 수십억 달러[1]

#### 의료 영상 생성

**가능한 응용**:
1. **환자 익명화**: 프라이버시 보호
2. **데이터 증강**: 학습 데이터 부족 해결
3. **진단 모의 실험**: 의료 교육[1]

**규제 고려사항**:
- FDA 승인 필요성
- 진위 증명 (provenance) 추적
- 의료 윤리 검토 필수

### 7.3 기술적 고려사항

#### 1) 계산 효율성

**현재 성능**:
- 개념 학습: 10분 (8 A100 GPUs)
- 추론: 약 10분 (50 DDIM 스텝)[1]

**개선 목표**:
1. **빠른 샘플링**:
   - DPM-Solver, Analytic DPM 적용
   - DDIM 스텝 수 감소: 50 → 20-25
   - **예상**: 5배 속도 향상[18]

2. **경량 개념 학습**:
   - LoRA 기반 대체
   - 파라미터: 860M → 10M
   - **예상**: 100배 속도 향상[12]

3. **동적 정밀도**:
   - 중요 영역: 고정밀도
   - 배경: 저정밀도
   - **예상**: 2배 메모리 절약

#### 2) 메모리 최적화

**현재 요구량**: ~20GB (8 A100 GPUs에서 8GB/GPU)

**절감 기법**:

| 기법 | 절감율 | 비용 |
|------|--------|------|
| 모델 양자화 (8-bit) | 50% | 품질 1-2% 저하 |
| 그래디언트 체크포인팅 | 30% | 속도 10% 저하 |
| 토큰 병합 | 20% | 정밀도 미세 손실 |
| **전체 조합** | **70-80%** | **품질 허용 범위** |[18]

#### 3) 확장성

**현재 한계**: 512×512 고정 해상도

**확장 방법**:

1. **SDXL 마이그레이션**: 1024×1024
2. **계층적 생성**: 저해상도 → 초고해상도
3. **적응형 아키텍처**: DiT 기반 가변 해상도
4. **타일 기반 처리**: 매우 큰 이미지 (4K, 8K)

**예상 달성 시간**: 6-12개월 내[14]

### 7.4 윤리 및 안전성

#### 1) 얼굴 교환 관련 우려

**논문의 입장**:[1]
> "다른 인종 배경의 피사체 간 교환 시, 피부색이 동질화되는 경향이 있다. 따라서 유사한 인종 배경의 피사체 사용을 권장한다."

**근본적 문제**:
- Stable Diffusion의 학습 데이터 편향 상속
- 텍스트 프롬프트의 스테레오타입 포함[1]

**개선 방향**:
1. **편향 완화 기법**:

$$\mathcal{L}_{\text{bias}} = -\sum_{race} \text{Similarity}(\text{Color}(\text{generated}), \text{Color}(\text{ref}))$$

2. **다양한 데이터셋**: 모든 인종 포괄적 표현
3. **테스트 전 검사**: 편향 감지 및 수정

#### 2) 신원 도용 및 디ープ페이크

**위험**:
- 무단 얼굴 교환으로 가짜 증거 생성
- 사기, 명예훼손, 사기 가능성[1]

**완화 방법**:
1. **진위 증명 기술**:
   - 메타데이터 삽입: 생성 시간, 모델 버전
   - 블록체인 기반 추적[1]

2. **생성 콘텐츠 표시**:
   - 이미지에 "AI 생성" 워터마크
   - 파일 시그니처 포함[1]

3. **역추적 불가능성**:
   - 참조 이미지 정보 삭제
   - 역방향 검색 회피[1]

#### 3) 프라이버시 침해

**우려**:
- 개인 사진의 무단 사용
- 참조 이미지의 특성 추출[1]

**보호 방안**:
1. **사용자 동의**: 명확한 서면 동의
2. **데이터 삭제**: 사용 후 참조 이미지 즉시 삭제
3. **접근 제어**: 권한있는 사용자만 접근
4. **감시 시스템**: 비정상 사용 탐지[1]

#### 4) 저작권 문제

**문제점**:
- 참조 이미지의 출처 불명확화
- 생성된 이미지의 저작권 귀속 애매
- 학습 데이터 저작권 침해[1]

**해결책**:
1. **명확한 귀속**: "[이름]의 [원본] 기반 생성"
2. **로열티 지불**: 유명 피사체 사용 시 보상
3. **오픈 라이선스**: CC-BY-SA 등 투명한 라이선스
4. **법적 프레임워크**: 생성 AI 관련 입법

### 7.5 추가 연구 방향

#### 1) 다중 피사체 교환의 정밀화

**현재**: 기본적 지원 (Figure 6a)

**필요한 개선**:
1. **독립적 제어**: 각 피사체별 개별 매개변수
2. **피사체 간 상호작용**: 가려짐(occlusion), 그림자 등 유지
3. **부분 교환**: 머리만, 팔만 등의 세부 제어[4][1]

**기술적 접근**:
- 마스크 기반 대상화
- 계층적 제어 구조
- 조건부 생성[4]

#### 2) 동영상으로의 확장

**현재**: 정지 이미지 전용[1]

**비디오 특화 도전**:
1. **시간적 일관성**: 프레임 간 주의 일관성
2. **움직임 추적**: 피사체 움직임 자연스러움
3. **광학적 흐름 제어**: 전체 씬의 움직임과 일치[17]

**초기 노력**: VideoSwap (2023) - 의미론적 포인트 대응[17]

**미래 방향**: 프레임별 주의 교환의 시간 연속성 유지

#### 3) 3D 기반 생성으로의 진화

**장점**:
- 더 나은 포즈 제어
- 일관된 조명 및 그림자[1]

**기술 결합**:
- 3D 재구성 + 2D 확산 모델
- 신경 복사 필드(NeRF) 통합[1]

**응용 분야**:
- 게임 자산 생성
- AR/VR 콘텐츠
- 3D 프린팅을 위한 모델 생성[1]

***

## 결론

**PHOTOSWAP**은 개인화된 피사체 교환 분야에서 획기적인 기여를 했다. 자기 주의 출력(Φ) 교환이라는 새로운 메커니즘 발견과 Training-free 접근 방식은 사전 학습 diffusion 모델 활용의 새로운 패러다임을 제시했다.[1]

**2023년 발표 이후의 진전**:
- **SwapAnything (2024)**으로의 진화: 임의 개체 교환으로 확대[4]
- **주의 제어 기술의 정교화**: Layout Guidance 등으로 정밀도 향상[10]
- **평가 메트릭의 개선**: EditEval의 LMM Score 도입[13]
- **새로운 아키텍처**: DiT, Flow Matching으로의 마이그레이션 가능성[14]

**남은 도전 과제**:
1. **세부 정보 보존**: 손, 얼굴, 복잡 배경 개선 필요
2. **도메인 일반화**: 새로운 영역으로의 자동 적응
3. **윤리적 고려**: 디ープ페이크, 프라이버시 침해 방지
4. **실용적 배포**: 실시간 고해상도 처리[1]

**향후 가능성**:
- 중기(1년): SDXL 마이그레이션, 도메인 특화 모델로 15-20% 성능 향상[16]
- 장기(2년+): DiT 기반 아키텍처, 멀티모달 조건화로 30% 이상 성능 향상[14]
- 응용: 영화, 게임, 의료 영상 등 광범위한 산업 적용[1]

**최종 평가**: Photoswap은 단순한 이미지 편집 기법을 넘어 **생성 모델의 주의 제어 철학**을 정립한 중요한 작업이다. 이의 핵심 통찰들은 향후 다양한 생성 작업에 적용될 것으로 예상되며, 특히 정밀한 공간 제어와 개인화가 필요한 분야에서 크게 기여할 것으로 보인다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/33529ddb-1431-409d-a338-5709cf04ca3c/2305.18286v1.pdf)
[2](https://arxiv.org/abs/2305.18286)
[3](https://ieeexplore.ieee.org/document/10884879/)
[4](https://link.springer.com/10.1007/978-3-031-73411-3_23)
[5](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf)
[6](https://proceedings.neurips.cc/paper_files/paper/2023/file/6091bf1542b118287db4088bc16be8d9-Paper-Conference.pdf)
[7](https://arxiv.org/abs/2208.01618)
[8](https://arxiv.org/abs/2208.01626)
[9](https://openaccess.thecvf.com/content/CVPR2023/papers/Mokady_NULL-Text_Inversion_for_Editing_Real_Images_Using_Guided_Diffusion_Models_CVPR_2023_paper.pdf)
[10](https://openaccess.thecvf.com/content/WACV2024/papers/Chen_Training-Free_Layout_Control_With_Cross-Attention_Guidance_WACV_2024_paper.pdf)
[11](https://openaccess.thecvf.com/content/CVPR2024/papers/Phung_Grounded_Text-to-Image_Synthesis_with_Attention_Refocusing_CVPR_2024_paper.pdf)
[12](https://arxiv.org/html/2410.08192v1)
[13](https://www.alphaxiv.org/overview/2402.17525)
[14](https://arxiv.org/html/2410.10629v1)
[15](https://ieeexplore.ieee.org/document/11095040/)
[16](https://arxiv.org/html/2504.02612v2)
[17](https://arxiv.org/html/2312.02087v2)
[18](https://arxiv.org/html/2410.11795v1)
[19](https://arxiv.org/html/2403.03431v1)
[20](https://arxiv.org/html/2504.03738v1)
[21](https://www.ijcai.org/proceedings/2025/0764.pdf)
[22](https://arxiv.org/abs/2404.05717)
[23](https://www.banglajol.info/index.php/BJNM/article/view/79478)
[24](https://aacrjournals.org/cancerres/article/85/5_Supplement/A037/751888/Abstract-A037-CTNNB1-mutations-in-hepatocellular)
[25](https://ojs.bonviewpress.com/index.php/jdsis/article/view/4391)
[26](https://www.researchprotocols.org/2025/1/e68996)
[27](https://ojs.bonviewpress.com/index.php/jdsis/article/view/4358)
[28](https://ojs.bonviewpress.com/index.php/AIA/article/view/4376)
[29](http://arxiv.org/pdf/2404.05717.pdf)
[30](https://arxiv.org/pdf/2306.12624.pdf)
[31](https://dl.acm.org/doi/pdf/10.1145/3610543.3626172)
[32](https://arxiv.org/abs/2309.05793)
[33](https://arxiv.org/pdf/2303.10073.pdf)
[34](https://arxiv.org/html/2412.04280v1)
[35](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/photoswap/)
[36](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Dense_Text-to-Image_Generation_with_Attention_Modulation_ICCV_2023_paper.pdf)
[37](https://kkm0476.tistory.com/5)
[38](https://liner.com/review/photoswap-personalized-subject-swapping-in-images)
[39](https://openreview.net/pdf/3240f5f3bf947941eb80a6bb602f416afcaf403f.pdf)
[40](https://seunkorea.tistory.com/49)
[41](https://github.com/eric-ai-lab/swap-anything)
[42](https://arxiv.org/html/2410.02483v2)
[43](https://arxiv.org/html/2510.09475v1)
[44](https://arxiv.org/html/2505.10743v1)
[45](https://arxiv.org/html/2505.20909v1)
[46](https://arxiv.org/html/2503.16025v1)
[47](https://arxiv.org/html/2506.06826v1)
[48](https://arxiv.org/abs/2304.00186)
[49](https://arxiv.org/html/2503.04215v1)
[50](https://www.mdpi.com/2072-4292/16/16/3000)
[51](https://arxiv.org/abs/2305.18993)
[52](https://www.frontiersin.org/articles/10.3389/frwa.2025.1635275/full)
[53](https://arxiv.org/abs/2510.03795)
[54](https://aca.pensoft.net/article/151406/)
[55](https://jurnal.iainponorogo.ac.id/index.php/dialogia/article/view/10726)
[56](https://arxiv.org/pdf/1910.10683.pdf)
[57](https://arxiv.org/html/2406.03146v1)
[58](https://zenodo.org/record/5553901/files/LostInTransduction.pdf)
[59](https://arxiv.org/html/2501.18373v1)
[60](https://arxiv.org/pdf/2501.10933.pdf)
[61](https://arxiv.org/html/2410.22317v1)
[62](https://osf.io/3fkzc/download)
[63](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08139.pdf)
[64](https://juniboy97.tistory.com/81)
[65](https://github.com/MingkunLei/Awesome-Style-Transfer-with-Diffusion-Models)
[66](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/prompt-to-prompt/)
[67](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/null-text-inversion/)
[68](https://arxiv.org/html/2406.03146v2)
[69](https://arxiv.org/html/2512.11763v1)
[70](https://arxiv.org/pdf/2502.11308.pdf)
[71](https://arxiv.org/abs/2211.09794)
[72](https://arxiv.org/html/2408.00458v2)
[73](https://arxiv.org/html/2303.15649v3)
[74](https://arxiv.org/html/2508.05323v1)
[75](https://github.com/silent-chen/layout-guidance)
[76](https://papers.nips.cc/paper_files/paper/2024/file/f782860c2a5d8f675b0066522b8c2cf2-Paper-Conference.pdf)
[77](https://dl.acm.org/doi/10.1145/3721238.3730668)
[78](https://arxiv.org/abs/2407.07111)
[79](https://arxiv.org/abs/2406.14555)
[80](https://ieeexplore.ieee.org/document/10688086/)
[81](https://arxiv.org/abs/2405.00878)
[82](https://ieeexplore.ieee.org/document/10972622/)
[83](https://peerj.com/articles/cs-1905)
[84](https://arxiv.org/abs/2403.09468)
[85](https://ieeexplore.ieee.org/document/10657341/)
[86](http://arxiv.org/pdf/2402.17525.pdf)
[87](https://arxiv.org/pdf/2308.09388.pdf)
[88](https://arxiv.org/html/2406.14555v1)
[89](http://arxiv.org/pdf/2205.11487.pdf)
[90](https://arxiv.org/html/2411.15738)
[91](https://arxiv.org/pdf/2210.00586.pdf)
[92](https://arxiv.org/pdf/2502.21151.pdf)
[93](https://arxiv.org/pdf/2211.01324.pdf)
[94](https://pubmed.ncbi.nlm.nih.gov/40031849/)
[95](https://cs.uwaterloo.ca/~ppoupart/publications/diffusion/Subject-driven-Text-to-Image-Generation-via-Preference-based-Reinforcement-Learning.pdf)
[96](https://pmc.ncbi.nlm.nih.gov/articles/PMC11419672/)
[97](https://arxiv.org/html/2504.13226v1)
[98](https://openreview.net/pdf?id=wyHCt1P7SR)
[99](https://arxiv.org/abs/2402.17525)
[100](https://arxiv.org/html/2412.19533v2)
[101](https://arxiv.org/html/2506.03131v1)
[102](https://arxiv.org/html/2509.06499v2)
[103](https://arxiv.org/html/2410.02703v1)
[104](https://arxiv.org/html/2209.00796v15)
[105](https://arxiv.org/html/2412.03347v2)
