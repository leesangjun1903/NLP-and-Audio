# eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers

### 1. 핵심 주장과 주요 기여

**eDiff-I의 핵심 주장**은 텍스트-이미지 확산 모델의 생성 과정이 **잡음 수준(noise level)에 따라 질적으로 다른 동작**을 보인다는 관찰에 기반합니다.[1]

구체적으로:
- **높은 잡음(σ = 80)**: 모델이 텍스트 프롬프트에 크게 의존하여 텍스트 정렬된 콘텐츠 생성
- **낮은 잡음(σ = 0.0005)**: 텍스트 조건을 거의 무시하고 시각적 충실도 향상에만 집중

이 관찰에 기초하여 **잡음 수준별로 전문화된 여러 개의 디노이저로 구성된 앙상블**을 제안합니다.[1]

**주요 기여**:
1. 텍스트-이미지 확산 모델의 시간적 동역학에 대한 이론적 통찰
2. 효율적인 이진 트리 분기 전략으로 전문가 디노이저 훈련
3. T5, CLIP 텍스트, CLIP 이미지 임베딩 앙상블 활용
4. 훈련 없는 "Paint-with-words" 공간 제어 기능[1]

***

### 2. 해결하고자 하는 문제

**기존 방법의 한계**:

표준 확산 모델은 동일한 디노이저를 전체 잡음 범위에서 공유하며, 이는 다음 문제를 야기합니다:

1. **역할의 혼재**: 단일 모델이 텍스트 정렬과 시각적 충실도라는 상충하는 목표를 동시 추구
2. **용량 제약**: 제한된 모델 용량으로 두 가지 이질적 작업을 모두 수행 곤란
3. **시간 임베딩 부족**: 단순 시간 임베딩만으로는 복잡한 동역학 표현 불가능

**경험적 증거** (Figure 3-4):[1]

교차주의 맵 분석:
- 높은 잡음: 텍스트 토큰 주의력 강함
- 낮은 잡음: 텍스트 주의력 약하고 Null 토큰 주로 주의

프롬프트 전환 실험:
- 마지막 7% 디노이징에서 프롬프트 변경 → 출력 불변
- 첫 40%에서 프롬프트 변경 → 완전히 다른 출력

***

### 3. 제안 방법

#### 3.1 기본 훈련 목적 함수

```math
\mathbb{E}_{p_{\text{data}}(x_{\text{clean}}, e), p(\epsilon), p(\sigma)}\left[\lambda(\sigma)\|D(x_{\text{clean}} + \sigma\epsilon; e, \sigma) - x_{\text{clean}}\|_2^2\right]
```

[1]

**디노이저 전조건화**:

```math
D(x; e,\sigma) := \left(\frac{\sigma_{\text{data}}}{\sigma^*}\right)^2 x + \sigma \cdot \frac{\sigma_{\text{data}}}{\sigma^*} F_\theta\left(\frac{x}{\sigma^*}; e, \ln(\sigma)\right)
```

[1]

여기서 $\sigma^* = \sqrt{\sigma^2 + \sigma_{\text{data}}^2}$, $\sigma_{\text{data}} = 0.5$

**샘플링(생성 ODE)**:
$$\frac{dx}{d\sigma} = -\sigma \nabla_x \log p(x|e, \sigma) = \frac{x - D(x; e, \sigma)}{\sigma}$$[1]

#### 3.2 전문가 디노이저 앙상블 구조

**세 개의 전문가 구성**:[1]

1. **고잡음 전문가** ($E^9_{511}$): 높은 잡음 범위에서 텍스트 조건 처리 최적화
2. **저잡음 전문가** ($E^3_0$): 낮은 잡음 범위에서 시각적 세부사항 추가 최적화
3. **중간 범위 전문가** ($M_C$): 나머지 모든 잡음 수준 처리

**효율적 훈련: 이진 트리 분기 전략**:[1]

- **단계 1**: 기본 모델 훈련 ($M(0,0)$ on $p(\sigma)$ for 500K 반복)
- **단계 2-N**: 재귀적 분기
  - 각 단계에서 부모 모델을 두 자식 모델로 분할
  - 각 자식은 부모로 초기화 후 특정 잡음 구간에서만 추가 훈련
  - 양 끝노드(고잡음 및 저잡음)에 집중

**효율성**: 처음부터 모든 전문가를 훈련하는 것과 달리 **전이 학습 활용**하며, **추론 시간은 불변** (각 단계에서 하나의 모델만 실행)

#### 3.3 다중 조건 입력

**텍스트 임베딩 앙상블**:[1]

- **T5-XXL**: 높은 용량의 언어 모델, 개별 객체 속성 정확히 파악
- **CLIP L/14 텍스트**: 이미지-텍스트 정렬 최적화, 전체 이미지 시각적 모양 파악

$$e_{\text{combined}} = [\text{proj}_{\text{T5}}(e_{\text{T5}}), \text{proj}_{\text{CLIP}}(e_{\text{CLIP-text}}), \text{proj}_{\text{CLIP-image}}(e_{\text{CLIP-image}})]$$

효과:
- **T5만**: 높은 구성성, 정확한 세부사항
- **CLIP만**: 정확한 객체, 세부사항 부족
- **T5 + CLIP**: 최고 성능 (둘의 장점 결합)

**CLIP 이미지 임베딩**:
- 참조 이미지의 스타일 정보 추출
- 텍스트 프롬프트와 결합하여 스타일 전이 수행

#### 3.4 Paint-with-Words: 훈련 없는 공간 제어

**주의력 행렬 수정**:[1]

$$\text{Attention} = \text{softmax}\left(\frac{QK^T + wA}{\sqrt{d_k}}\right)V$$

여기서:
- $Q$: 이미지 토큰 쿼리
- $K, V$: 텍스트 토큰 키, 값
- $A \in \mathbb{R}^{N_i \times N_t}$: 사용자 지정 주의력 행렬

**동적 가중치 스케줄**:[1]
$$w = w' \cdot \log(1 + \sigma) \cdot \max(QK^T)$$

- 높은 잡음에서 강한 영향
- 낮은 잡음에서 자동으로 약해짐

***

### 4. 모델 구조와 성능 향상

#### 4.1 전체 파이프라인

**계단식 구조**:[1]
- **기본 모델**: 64×64 해상도 생성 (전문가 앙상블)
- **SR256 모델**: 256×256로 업샘플링 (2-expert 앙상블)
- **SR1024 모델**: 1024×1024로 업샘플링

각 모델은 T5, CLIP 텍스트, CLIP 이미지 임베딩으로 조건화

#### 4.2 정량적 성능 향상

**벤치마크 비교 (MS-COCO 2014)**:[1]

| 모델 | 파라미터 | FID-30K |
|------|---------|---------|
| GLIDE | 5B | 12.24 |
| DALL·E 2 | 6.5B | 10.39 |
| Stable Diffusion | 1.4B | 8.59 |
| Imagen | 7.9B | 7.27 |
| **eDiff-I-Config-D** | **9.1B** | **6.95** |

**앙상블 효과** (2-Expert vs Baseline):[1]
- 동일 훈련 샘플(600K vs 800K)로 비교
- **전체 FID-CLIP 트레이드오프 곡선에서 일관된 개선**
- 텍스트 정렬(CLIP 점수) 향상: ~0.01-0.02
- 시각적 충실도(FID) 향상: ~1-2

**임베딩의 영향** (Figure 8):[1]

MS-COCO (평균 10.62 단어):
- CLIP만: FID ~22.5
- T5만: FID ~23.0  
- **T5 + CLIP: FID ~20.0** (최우수)

Visual Genome (평균 61.92 단어):
- T5 단독: 더 나은 성능
- **T5 + CLIP: 여전히 최고**

#### 4.3 정성적 개선

**다중 객체 처리** (Figure 10):[1]
- Stable Diffusion: 속성 혼동
- DALL·E 2: 일부 객체 누락
- **eDiff-I**: 모든 객체와 속성 정확히 생성

**텍스트 생성** (Figure 11):[1]
- Stable Diffusion: 텍스트 미생성
- DALL·E 2: 오타 ("NIDCKA VIDA")
- **eDiff-I**: 정확한 텍스트 ("NVIDIA ROCKS")

**장문 설명 처리** (Figure 12):[1]
- 기존: 여러 요소 누락
- **eDiff-I**: 모든 요소 정확히 포함

**스타일 전이** (Figure 16):[1]
- CLIP 이미지 임베딩으로 참조 스타일 적용
- Rembrandt, Van Gogh, 연필 스케치 등 다양한 스타일
- 콘텐츠와 스타일 독립적 제어

**Paint-with-Words** (Figure 17):[1]
- 사용자가 텍스트 구문의 공간 위치 직관적 지정
- 거친 스크리블로도 정확한 배치 생성
- 완전히 새로운 사용자 제어 패러다임

#### 4.4 추론 효율성

**추론 시간 비교** (Figure 13):[1]
- 깊이 증가: 모델 깊이를 늘리면 feedforward 시간 증가
- **eDiff-I 앙상블**: 각 단계에서 하나의 모델만 실행 → **시간 불변**

이는 **매개변수 증가 없이 계산 효율성 유지**하는 혁신적 설계입니다.

***

### 5. 모델의 일반화 성능 향상

#### 5.1 영점(Zero-Shot) 일반화의 강화

**기존 성과**:[1]

- 훈련 데이터에 없는 새로운 개념 조합에 우수한 적응
- COCO와 Visual Genome의 새로운 캡션 대상 평가
- Visual Genome의 장문 캡션 처리 우수성

**강화 메커니즘**:

1. **전문화된 디노이저**:[1]
   - 각 전문가가 특정 잡음 수준에 집중
   - 과잉적합 위험 감소
   - 암묵적 정규화 효과

2. **다중 임베딩 활용**:[1]
   - T5와 CLIP의 상보적 특성
   - 서로 다른 관점의 텍스트 이해로 견고성 향상
   - 임베딩 드롭아웃의 정규화 효과

3. **이진 트리 훈련**:[1]
   - 단계적 세분화로 각 수준의 과제 복잡도 감소
   - 전이 학습의 자연스러운 적용
   - 더 나은 초기화

#### 5.2 구조적 일반화 근거

**정보 병목 이론**:
- 확산 모델의 자연스러운 정보 병목
- 각 잡음 수준에서 최적 정보 흐름 유지
- 불필요한 정보 손실 방지

**표현 학습의 다양성**:
- 전문화된 모델들의 서로 다른 표현 학습
- 집단 지식의 견고성 향상

#### 5.3 향후 일반화 개선 방향

**1. 이진 트리 깊이 확장**:
- 현재 최대 깊이 9 (3개 전문가)
- 더 깊은 트리: 미세한 잡음 수준 세분화
- 적응적 깊이: 데이터셋 특성에 따른 자동 결정

**2. 조건 임베딩 확장**:
- 다국어 인코더: 언어 간 일반화 향상
- 시각 특징 인코더: 풍부한 공간 정보
- 의미 그래프: 개념 간 관계 명시화

**3. 동적 라우팅**:
- 각 샘플에 최적 전문가 자동 선택
- Mixture-of-Experts 개념 적용
- 계산 효율성 추가 개선

**4. 메타 학습**:
- 새로운 도메인에 빠른 적응
- 기존 전문가 기반 가중치 학습
- 몇 샷 학습(few-shot) 능력

***

### 6. 한계

#### 6.1 훈련 복잡도[1]

- 기본 모델: 256개 A100 GPU 필요
- 초해상화: 각각 128개 GPU
- 전체 반복: 1.9M + 2M + 1.7M = 5.6M
- **학계 재현 어려움**

#### 6.2 추론 유연성[1]

- 고정된 전문가 구조
- 훈련 후 전문가 수 변경 불가
- 새로운 데이터셋에 맞춰 재구성 필요

#### 6.3 해석 가능성

- 각 전문가의 정확한 학습 내용 불명확
- 특정 전문가 제거의 영향 분석 부족
- 블랙박스 모델의 한계

#### 6.4 윤리 및 사회적 영향[1]

**악의적 활용**:
- 고급 이미지 조작
- 가짜 콘텐츠 생성
- 딥페이크 기술의 악용 우려

**훈련 데이터 편향**:
- 약 10억 개 데이터셋의 내재된 편향
- 생성 이미지에 반영되는 사회적 편견
- 다양성 부족

***

### 7. 앞으로의 연구에 미치는 영향

#### 7.1 확산 모델 설계 철학의 변화

**"One Model Fits All" 패러다임 탈피**:

기존: 단일 모델로 전체 생성 과정 처리
eDiff-I 이후: **모듈식, 전문화된 설계의 우월성**

**영향 범위**:
- 비디오 생성: 레이아웃, 텍스처, 디테일 단계 분리
- 3D 생성: 글로벌 구조 vs 로컬 세부사항 분리
- 음성 생성: 음성학적 구조 vs 음질 분리

#### 7.2 조건부 생성 모델 연구 확장

**교차주의(Cross-Attention) 메커니즘의 심화**:
- 시간 의존적 행동 이해
- Paint-with-Words의 훈련 없는 제어 가능성

**공간 제어의 정밀화**:
- 특정 시각적 속성의 미세 조정
- 의료 영상에서의 정밀 제어

#### 7.3 실제 응용 분야 확대

**콘텐츠 생성 도구**:
- Paint-with-Words로 예술가의 창의성 증강
- 스타일 참조로 일관된 시각 언어
- A/B 테스트용 변형 자동 생성

**의료 및 과학**:
- 훈련 데이터 부족 분야의 합성 데이터
- 희귀 질병 시뮬레이션
- 프라이버시 보호

#### 7.4 이론적 기여

**생성 과정의 역학 이해**:
- 잡음 수준에 따른 생성 작업 변화 공식화
- 텍스트 조건화의 시간 역학
- 시각적 충실도의 점진적 향상 모델화

**앙상블 학습의 새로운 관점**:
- 시간적(잡음 수준적) 분해
- 추론 시간 증가 없는 용량 확장

***

### 8. 2020년 이후 관련 최신 연구 탐색

#### 8.1 텍스트-이미지 생성의 발전

**주요 모델들** (2022-2023):[2][3]
- **Imagen** (2022): 계층적 텍스트 조건, SOTA 달성
- **Stable Diffusion** (2022): 잠재 공간 확산, 계산 효율성 혁신
- **Parti** (2022): 20B 매개변수 자동회귀 모델

**효율성 연구** (2024):[2]
- **BudgetFusion** (2024/12): 지각적으로 최적의 확산 단계 자동 예측
- **DPM-Solver**: 10 단계 내 고품질 샘플링
- **Flow Matching**: 단일 단계 생성 가능

#### 8.2 주의력 메커니즘의 심화 분석

**Cross-Attention 분석** (CVPR 2024):[2]
- "Towards Understanding Cross and Self-Attention in Stable Diffusion"
- 교차주의의 핵심 역할 규명
- 이미지 편집에서의 중요성 확인
- eDiff-I의 Paint-with-Words와 유사한 발견

**주의력 기반 제어**:[2]
- Prompt-to-Prompt (2022): 프롬프트 수정으로 이미지 편집
- DiffEdit (2022): 의미론적 이미지 편집

#### 8.3 일반화 능력 연구

**영점(Zero-Shot) 학습** (2023-2024):[2]

1. **분류기로서의 활용**:
   - "Your Diffusion Model is Secretly a Zero-Shot Classifier" (2023/03)
   - 밀도 추정을 통한 제로샷 분류[4]
   - CLIP 대비 높은 조합론적 추론

2. **도메인 적응**:
   - **ZoDi** (2024/09): 제로샷 도메인 적응[5]
   - **RevCD** (2024/08): 일반화 제로샷 학습[6]
   - **ZeroDiff** (2025/02): 시각-의미 상관관계 강화[7]

3. **3D 생성**:
   - **Zero-1-to-3** (2023/03): 단일 이미지 3D 재구성[8]
   - 합성 데이터 훈련 후 실제 이미지에 강한 일반화

#### 8.4 일반화 메커니즘 이론

**최신 이론 연구** (2024-2025):[2]

- **"Towards a Mechanistic Explanation of Diffusion Model Generalization"** (2025/02):
  - 확산 모델의 일반화 행동을 설명하는 단순 메커니즘[9]
  - 네트워크 아키텍처 전반의 공유된 귀납 편향

- **"No 'Zero-Shot' Without Exponential Data"** (2024/04):
  - 개념의 사전학습 빈도가 영점 성능 결정[10]
  - 로그-선형 스케일링 법칙 발견
  - 다중모드 모델의 진정한 영점 일반화 한계

#### 8.5 모듈식 및 계층적 설계

**최신 경향** (2024-2025):[2]

- **Mixture-of-Experts in Diffusion**:
  - "Regularized Neural Ensemblers" (2024/10): 동적 앙상블링의 중요성[11]
  - eDiff-I의 시간적 분해와 유사한 공간적 분해

- **적응적 아키텍처**:
  - **DaptDiffusion** (2025/06): 픽셀 수준 대응[12]
  - 사용자 입력에 동적으로 응응하는 구조

#### 8.6 응용 분야 확대

**비디오 생성** (2024-2025):[2]
- "Survey of Video Diffusion Models" (2025/04)
- 확산 기반 비디오의 혁명적 발전
- 시간적 일관성과 움직임 제어

**로봇 조작** (2024-2025):[13]
- Diffusion Policy (2023): CNN 기반 로봇 제어
- Diffusion Transformers (2023): 트랜스포머 기반
- 데이터 증강으로 일반화 향상

**의료 영상** (2024-2025):[2]
- 이미지 이상 탐지
- 제로샷 부재 영역 채우기
- 저선량 이미징

***

### 9. 결론

#### 9.1 eDiff-I의 획기적 기여

1. **개념적 혁신**: 생성 과정의 이질적 단계를 인식하고 전문화된 설계로 해결
2. **기술적 효율성**: 추론 비용 증가 없이 훈련 용량 확장
3. **실무적 성과**: 최고 수준의 정량적 성능 달성 (FID 6.95)
4. **사용자 경험**: Paint-with-Words 등 직관적 제어 인터페이스

#### 9.2 연구 커뮤니티 영향

- **패러다임 전환**: 단일 모델에서 모듈식 아키텍처로
- **이론적 기초**: 생성 과정의 역학에 대한 새로운 이해
- **응용 확대**: 비디오, 로봇, 의료 등 다양한 분야로 파급

#### 9.3 미래의 방향성

1. **깊이 확장**: 더 세밀한 잡음 수준 분해
2. **동적 라우팅**: 샘플별 최적 전문가 선택
3. **메타 학습**: 새로운 도메인에 빠른 적응
4. **이론 심화**: 일반화 원리의 수학적 공식화

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/14102a6c-9170-492a-9740-c4cdb866931f/2211.01324v5.pdf)
[2](https://arxiv.org/abs/2504.16081)
[3](https://arxiv.org/abs/2506.17324)
[4](https://ieeexplore.ieee.org/document/10376944/)
[5](https://arxiv.org/html/2403.13652)
[6](https://arxiv.org/abs/2409.00511)
[7](https://arxiv.org/pdf/2406.02929.pdf)
[8](https://ieeexplore.ieee.org/document/10378322/)
[9](https://arxiv.org/html/2411.19339v2)
[10](https://arxiv.org/abs/2404.04125)
[11](https://arxiv.org/abs/2410.04520)
[12](https://link.springer.com/10.1007/s10489-025-06673-1)
[13](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1606247/full)
[14](https://arxiv.org/abs/2507.20478)
[15](https://arxiv.org/abs/2505.09364)
[16](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-025-01752-8)
[17](https://link.springer.com/10.1007/s00330-025-11871-z)
[18](https://www.semanticscholar.org/paper/b99b1775d0c1506577cab0846c9886844f9c54a5)
[19](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[20](http://medrxiv.org/lookup/doi/10.1101/2025.08.11.25333418)
[21](http://arxiv.org/pdf/2412.17162.pdf)
[22](https://arxiv.org/pdf/2306.01984.pdf)
[23](https://aclanthology.org/2023.acl-long.248.pdf)
[24](http://arxiv.org/pdf/2410.11795.pdf)
[25](http://arxiv.org/abs/2401.13162)
[26](https://arxiv.org/html/2412.05780v3)
[27](https://arxiv.org/abs/2408.08306)
[28](https://arxiv.org/pdf/2107.00630.pdf)
[29](https://diffusion.kaist.ac.kr)
[30](https://users.aalto.fi/~laines9/publications/ediffi2022_paper.pdf)
[31](https://pubmed.ncbi.nlm.nih.gov/39314702/)
[32](https://arxiv.org/abs/2211.01324)
[33](https://www.sciencedirect.com/science/article/abs/pii/S0925231224006441)
[34](https://arxiv.org/abs/2209.00796)
[35](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ediff-i/)
[36](https://peerj.com/articles/cs-2332.pdf)
[37](https://www.youtube.com/watch?v=yHQv0d9McvQ)
[38](https://dl.acm.org/doi/10.1145/3625687.3625798)
[39](https://arxiv.org/abs/2403.10967)
[40](https://arxiv.org/abs/2310.10639)
[41](https://ieeexplore.ieee.org/document/10865513/)
[42](https://arxiv.org/abs/2305.13831)
[43](https://ieeexplore.ieee.org/document/10656544/)
[44](https://ieeexplore.ieee.org/document/10981725/)
[45](https://arxiv.org/html/2402.11424v1)
[46](https://arxiv.org/html/2504.01689v1)
[47](http://arxiv.org/pdf/2412.03771.pdf)
[48](https://arxiv.org/html/2412.17219v2)
[49](https://www.sciencedirect.com/science/article/abs/pii/S095219762502189X)
[50](https://proceedings.neurips.cc/paper_files/paper/2024/file/9647157086adf5aa2c0217fb7f82bb19-Paper-Conference.pdf)
[51](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Zero-Shot_Structure-Preserving_Diffusion_Model_for_High_Dynamic_Range_Tone_Mapping_CVPR_2024_paper.pdf)
[52](https://www.nature.com/articles/s41598-024-79476-y)
[53](https://arxiv.org/html/2409.19365v3)
[54](https://liner.com/ko/review/texttoimage-diffusion-models-are-zeroshot-classifiers)
[55](https://s-space.snu.ac.kr/handle/10371/210033)
[56](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Towards_Understanding_Cross_and_Self-Attention_in_Stable_Diffusion_for_Text-Guided_CVPR_2024_paper.pdf)
[57](https://arxiv.org/html/2409.00511v1)
