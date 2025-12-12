# NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation

### 1. 핵심 주장과 주요 기여 요약

**NUWA-XL**은 텍스트 설명에서 극도로 긴 비디오(3376프레임)를 생성하기 위한 혁신적인 "Diffusion over Diffusion" 아키텍처를 제시합니다. 이 논문의 핵심 주장은 기존의 "Autoregressive over X" 방식의 근본적인 한계를 해결한다는 것입니다.[1]

**주요 기여:**

- **훈련-추론 간격 제거**: 첫 번째로 긴 비디오(3376프레임)에서 직접 훈련하여 훈련-추론 불일치 문제 해결[1]
- **병렬 추론 가능**: 순차적 생성에서 벗어나 모든 세그먼트를 병렬로 생성 가능하며, 1024프레임 생성 시 94.26% 추론 시간 단축[1]
- **지수적 확장성**: O(L^m) 길이의 비디오 생성 가능 (L: 지역 확산 길이, m: 깊이)[1]
- **새로운 벤치마크**: FlintstonesHD 데이터셋 구축으로 장시간 비디오 생성 평가 기준 제시[1]

***

### 2. 해결하고자 하는 문제, 제안 방법, 모델 구조

#### 2.1 문제 정의

기존 "Autoregressive over X" 아키텍처는 두 가지 핵심 문제를 야기합니다:[1]

1. **훈련-추론 간격**: 16프레임 이하로 훈련하면서 1024프레임 생성을 요구하는 극심한 분포 불일치로 인해 장기간 불일치성과 비현실적인 장면 전환 발생
2. **순차 생성의 비효율성**: 슬라이딩 윈도우 기반 생성으로 인해 병렬화 불가능하며, TATS는 1024프레임 생성에 7.5분, Phenaki는 4.1분 소요[1]

#### 2.2 제안 방법: "Coarse-to-Fine" Diffusion over Diffusion

**핵심 아이디어**: 비디오를 두 단계로 생성합니다:[1]

1. **전역 확산 (Global Diffusion)**: L개의 프롬프트로부터 L개의 키프레임 생성 → "대강의 줄거리" 형성
2. **지역 확산 (Local Diffusion)**: 인접한 키프레임들 사이의 중간 프레임 재귀적으로 채우기 → "세밀한 세부사항" 추가

수식으로 표현하면:

$$v_0^1 = \text{GlobalDiffusion}(p_1, v_c^{01})$$

여기서 초기에 $v_c^{01}$은 모두 영(zero)입니다.[1]

첫 번째 지역 확산:

$$v_0^2 = \text{LocalDiffusion}(p_2, v_c^{02})$$

여기서 $v_c^{02}$는 인접한 키프레임들로부터 마스킹된 조건입니다.[1]

반복적 적용:

$$v_0^3 = \text{LocalDiffusion}(p_3, v_c^{03})$$

최종 비디오 길이: $$O(L^m)$$[1]

#### 2.3 모델 구조 상세 분석

**2.3.1 Temporal KLVAE (T-KLVAE)**

압축 효율성을 위해 픽셀 공간이 아닌 잠재 공간에서 확산 과정을 수행합니다. 기존 사전훈련된 KLVAE를 확장하여 시간적 정보를 모델링합니다.[1]

시간적 컨볼루션 초기화:

$$W_{\text{conv1d}}[i, i, (k-1)//2] = 1$$

여기서 모든 다른 가중치는 0으로 초기화되어, 항등 함수로 작동하여 기존 지식 보존.[1]

시간적 어텐션 초기화:

```math
W_{\text{att\_out}} = 0
```

**2.3.2 Mask Temporal Diffusion (MTD)**

핵심 확산 모델로, 표준 확산 과정을 따릅니다:

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)\mathbf{I})$$

노이즈 처리:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0,\mathbf{I})$$

훈련 목표 (L2 손실):

$$L_\theta = ||\epsilon - \epsilon_\theta(x_t, p, t, x_c^0)||_2^2$$

**핵심 특징**: 전역 확산은 모든 프레임이 마스킹되고(조건 없음), 지역 확산은 첫 번째와 마지막 프레임만 조건으로 제공.[1]

**UpBlock 상세 구조**:

1. Skip 연결 연결:
$$h := [s; h_{in}]$$

2. 타임스텝 임베딩 추가:
$$h := h + t$$

3. 공간 컨볼루션:
$$h := \text{SpatialConv}(h)$$

4. 시간적 컨볼루션:
$$h := \text{TemporalConv}(h)$$

5. 시각적 조건 주입:
$$h := w_c \cdot h + b_c + h$$
$$h := w_m \cdot h + b_m + h$$

6. 주의 메커니즘:
   - 공간적 자체 주의 (SA):
$$Q^{SA} = hW^{SA}_Q; \quad K^{SA} = hW^{SA}_K; \quad V^{SA} = hW^{SA}_V$$
$$\hat{h}^{SA} = \text{Selfattn}(Q^{SA}, K^{SA}, V^{SA})$$

   - 프롬프트 교차 주의 (PA):
$$Q^{PA} = hW^{PA}_Q; \quad K^{PA} = pW^{PA}_K; \quad V^{PA} = pW^{PA}_V$$
$$\hat{h}^{PA} = \text{Crossattn}(Q^{PA}, K^{PA}, V^{PA})$$

   - 시간적 자체 주의 (TA): SA와 동일하나 시간 축을 시퀀스 길이로 처리[1]

**추론 단계**:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, p, t, x_c^0)\right) + \sqrt{\frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t}} \cdot \epsilon$$

여기서 $\epsilon \sim \mathcal{N}(0,\mathbf{I})$[1]

***

### 3. 성능 향상 및 한계

#### 3.1 정량적 성능 향상

| 메트릭 | 해상도 | 길이 | NUWA-XL 성능 | 개선 사항 |
|--------|--------|------|-------------|---------|
| 추론시간 | 128 | 256f | 17s | 85.09% 단축 |
| 추론시간 | 128 | 1024f | 26s | 94.26% 단축 |
| Avg-FID | 128 | 1024f | 35.79 | Phenaki(48.56) 대비 향상 |
| B-FVD-16 | 128 | 1024f | 572.86 | Phenaki(622.06) 대비 향상 |

기존 방식과의 비교: "Autoregressive over X" 모델들은 길이 증가에 따라 품질 저하가 심했으나, NUWA-XL은 병렬 생성으로 인해 더 안정적인 성능 유지[1]

#### 3.2 정성적 분석

**지역 일관성**: 질적 비교(Figure 4)에서 NUWA-XL은 의류 같은 세부 사항에서 장기간 일관성 유지(22프레임과 1688프레임의 옷감 일관성)[1]

**현실적 장면 전환**: AR over Diffusion의 비현실적 전환(17-20프레임)을 NUWA-XL은 자연스럽게 해결[1]

#### 3.3 제거 실험 (Ablation Study) 분석

**T-KLVAE 효과**:[1]
- KLVAE (독립 이미지): FID 4.71, FVD 28.07
- T-KLVAE-R (무작위 초기화): FID 5.44, FVD 12.75
- T-KLVAE (항등 초기화): FID 4.35, FVD 11.88 ← **최적**

**MTD 설정** (MI: 다중 스케일 주입, SI: 대칭 주입):[1]
- MTD w/o MS: FID 39.28, FVD 548.90
- MTD w/o S: FID 36.04, FVD 526.36
- MTD (완전): FID 35.95, FVD 520.19 ← **최적**

**깊이 영향** (L=16 고정):[1]
- 깊이 1: 16f에서 527.44, 1024f에서 719.23
- 깊이 3: 16f에서 520.19, 1024f에서 572.86 ← **최적 균형**

**지역 확산 길이** (m=3 고정):[1]
- L=8: 1024f에서 727.22
- L=16: 1024f에서 572.86 ← **최적**
- L=32: 메모리 부족 (OOM)

#### 3.4 주요 한계

**데이터 제한**: Flintstones 카툰만 검증 가능, 개방형 장시간 비디오(영화, TV) 부재[1]

**데이터 요구량**: 장시간 비디오에서 직접 훈련하는 것이 훈련-추론 간격을 해결하나, 이는 더 많은 데이터 필요

**병렬 추론 자원**: GPU 자원이 충분하지 않으면 병렬화 이점 미흡[1]

**해상도 제약**: 최고 256 해상도 생성 (현대 기준으로는 제한적)

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 일반화 개선 메커니즘

**계층적 구조의 이점**:

NUWA-XL의 계층적 구조는 일반화 성능 향상을 자연스럽게 지원합니다:[2][3][1]

1. **점진적 학습**: 전역-지역 확산의 계층 구조는 간단한 패턴부터 복잡한 패턴까지 학습하도록 유도[3]

2. **도메인 이동 감소**: 장시간 비디오에서 직접 훈련하므로 단시간 비디오-장시간 비디오 간의 분포 차이 최소화[1]

#### 4.2 최신 연구의 관련성

**계층적 방식의 인정**:

최근 "Hierarchical Patch Diffusion Models for High-Resolution Video Generation" (CVPR 2024)는 유사한 계층적 접근으로 고해상도 비디오 생성에서 효과성을 입증했습니다:[4]

- "Deep Context Fusion"을 통해 저해상도에서 고해상도로의 맥락 정보 전파[4]
- 적응적 계산으로 조잡한 세부 사항에 더 많은 네트워크 용량 할당[4]

**"ARLON: Boosting Diffusion Transformers with Autoregressive Models for Long Video Generation"** (2024년):[5]

- AR 모델과 확산 트랜스포머 결합으로 장기 일관성 개선[5]
- 조잡한 공간-시간 정보로 세밀한 생성 유도[5]

#### 4.3 일반화 성능 향상 가능성

**긍정적 인자**:

1. **훈련-추론 일치**: 직접 장시간 비디오 훈련이 분포 이동 최소화[1]

2. **계층적 분해**: 전역-지역 구조로 다양한 입력에 적응 가능[1]

3. **데이터 효율성**: 사전훈련된 KLVAE 활용으로 필요 데이터량 감소[1]

4. **주의 메커니즘**: 공간/시간/교차 주의의 조합으로 복잡한 의존성 학습[1]

**개선 기회**:

1. **다양한 도메인 확대**: Flintstones 이상의 다양한 영상(영화, 애니메이션, 자연 영상) 포함 훈련[1]

2. **해상도 확장**: 고해상도(512×512 이상) 비디오에 적응적 계산 적용[4]

3. **조건부 확산 강화**: BERT/ViT 기반 시맨틱 임베딩 확충으로 프롬프트 이해 개선[7]

4. **적응적 깊이 선택**: 장시간 동적 학습으로 동적 깊이 조정 가능성[5]

***

### 5. 해당 논문이 앞으로의 연구에 미치는 영향 및 고려 사항

#### 5.1 학술적 영향

**패러다임 전환**:

1. **"Autoregressive over X" 대안 제시**: 순차 의존성 제거로 새로운 설계 방향 제시[1]

2. **계층적 확산의 효율성**: 후속 연구들이 유사한 계층 구조(Hierarchical Patch Diffusion, HPDM), (Hierarchical VAE, Hi-VAE)를 채택[8][4]

3. **훈련-추론 불일치 문제의 해결책**: 최근 "LongDiff", "MovieDreamer", "ARLON" 등이 유사 원리 적용[9][10][5]

#### 5.2 기술적 영향

**확산 아키텍처 개선**:

- Temporal KLVAE의 항등 초기화 기법: 사전훈련 지식 보존의 구체적 방안 제시[1]
- MTD의 다중 스케일 주입: 마스크 조건 처리의 효과적 방식[1]

**평가 메트릭 개선**:

- Block-FVD 제안: 긴 비디오 평가의 세분화된 접근[1]
- FlintstonesHD 데이터셋: 장시간 비디오 생성 벤치마크 제공[1]

#### 5.3 산업 응용 가능성

**실시간 생성 가능성**:

- 94.26% 추론 시간 단축은 대화형 비디오 생성 애플리케이션 실현성 증대[1]
- 병렬 추론으로 엣지 컴퓨팅 적용 가능성[1]

**콘텐츠 제작 자동화**:

- 장시간 코히어런트 비디오 생성으로 영상 제작 프로세스 자동화 기초 마련[1]

#### 5.4 향후 연구 시 고려할 점

**5.4.1 데이터 관점**

1. **다양성 확보**: 
   - 문제: Flintstones만 검증으로 도메인 외 성능 미정[1]
   - 방안: 영화, 자연 다큐멘터리, UGC 등 다양한 소스 포함[1]

2. **주석의 질**:
   - 현재: 자동 생성 캡션 사용으로 오류 가능성[1]
   - 개선: 수동 검수된 세밀한 주석 시스템 구축

3. **장시간 비디오 데이터 수집**:
   - 저작권, 프라이버시 문제 해결 필요[1]

**5.4.2 모델 구조 개선**

1. **다중 모달 조건**:
   - 텍스트만 아닌 음향, 음악, 모션 캡처 데이터 통합

2. **적응적 깊이 선택**:
   - 입력 내용 복잡도에 따른 동적 깊이 조정
   - 학습 가능한 라우팅 메커니즘

3. **고해상도 확장**:
   - Hi-VAE의 고압축 아이디어 적용[8]
   - 패치 기반 계층 구조와의 결합[11]

**5.4.3 성능 최적화**

1. **메모리 효율성**:
   - 현재 L=32에서 OOM[1]
   - 그래디언트 체크포인팅, 플래쉬 주의 등 기법 적용

2. **추론 가속**:
   - 진행 증류(Progressive Distillation) 적용으로 한 번에 필요한 단계 수 감소
   - 일단계 생성 (One-step generation)[13]

3. **일반화 강화**:
   - Domain Adaptation 기법 적용[14]
   - Meta-learning으로 다양한 도메인에 빠른 적응

**5.4.4 평가 방법론**

1. **시간적 일관성 메트릭**:
   - 현재 B-FVD만으로 부족
   - 광학 흐름 기반 시간적 안정성 지표 추가[15]

2. **인간 평가 확대**:
   - 자동 메트릭의 한계 극복
   - 장시간 비디오의 스토리 일관성, 캐릭터 일관성 평가[16]

3. **도메인별 성능 분석**:
   - 애니메이션, 실사, 합성 환경별 상세 분석

**5.4.5 제약 조건 완화**

1. **개방형 도메인 데이터**:
   - 문제: "개방형 장시간 비디오 부재"[1]
   - 현황: 2024-2025년 HunyuanVideo, Seaweed-7B 등이 대규모 데이터로 확대[17][18]

2. **컴퓨팅 자원 의존성**:
   - 병렬 추론 이점이 GPU 자원에 의존[1]
   - 양자화, 프루닝 기법 적용으로 저자원 환경 지원

***

### 6. 2020년 이후 관련 최신 연구 탐색

#### 6.1 기초 이론 발전

**Diffusion Probabilistic Models의 확장** (2022-2023):

- "Video Diffusion Models" (Ho et al., 2022): 3D U-Net 기반 비디오 확산의 기초 제시[19]
- "Latent Video Diffusion Models" (Blattmann et al., 2023): 잠재 공간 확산의 효율성 입증[15]

**핵심 개선**:
- 시간적 주의 메커니즘 통합으로 프레임 간 일관성 확보[19][15]
- 사전훈련된 이미지 모델 활용으로 데이터 효율성 증대[15]

#### 6.2 긴 비디오 생성 연구 진화

**초기 접근**: "Flexible Diffusion Modeling of Long Videos" (Harvey et al., 2022)[20]
- 마스킹된 조건부 확산 도입
- 자동회귀 확장으로 가변 길이 비디오 지원

**계층적 접근의 부상**:

1. **"Hierarchical Patch Diffusion Models for High-Resolution Video Generation"** (Skorokhodov et al., CVPR 2024):[4]
   - 패치 기반 피라미드 구조로 고해상도 훈련 가능
   - "Deep Context Fusion"으로 계층 간 정보 전파 효율화
   - FVD 66.32로 당시 SOTA 달성[4]

2. **"Hi-VAE: Efficient Video Autoencoding with Global and Detailed Motion"** (2025):[8]
   - Global Motion (조잡)과 Detailed Motion (세밀)의 계층 분해
   - 1428× 압축률로 매우 효율적[8]

**최신 통합 접근**:

1. **"ARLON: Boosting Diffusion Transformers with Autoregressive Models for Long Video Generation"** (2025):[5]
   - 확산 Transformers(DiT)와 AR 모델 결합
   - 조잡한 공간-시간 정보로 세밀한 생성 유도[5]

2. **"LongDiff: Training-Free Long Video Generation in One Go"** (CVPR 2025):[9]
   - 기존 모델 재학습 없이 장시간 생성
   - Position Mapping과 Informative Frame Selection으로 일반화 강화[9]

3. **"MovieDreamer: Hierarchical Generation for Coherent Long Visual Sequence"** (2024):[10]
   - 영상급 장시간 비디오 생성
   - 캐릭터 일관성 및 복잡한 내러티브 유지

#### 6.3 확산 모델의 성능 가속화

**진행 증류**:
- Imagen Video의 7개 모델을 8 스텝으로 압축[15]
- 일단계 생성 기법 개발[12]

**"Diffusion Adversarial Post-Training for One-Step Video Generation"** (2025):[12]
- 2초, 1280×720, 24fps 비디오 실시간 생성 달성
- 1024×1024 이미지도 한 번에 생성[13]

#### 6.4 조건부 확산 개선

**동적 언어 모델 통합**:

- **"Dysen-VDM: Empowering Dynamics-aware Text-to-Video Diffusion with LLMs"** (CVPR 2024):[6]
  - LLM으로 세밀한 시공간 특징 추출
  - 복잡한 동작 인식 강화[7]

**제어 가능성 강화**:

- **"Diffusion as Shader: 3D-aware Video Diffusion for Versatile Video Generation Control"** (2025):[21]
  - 카메라 조작, 콘텐츠 편집 등 다양한 제어
  - 단일 모델로 여러 제어 유형 지원[21]

#### 6.5 특화 응용

**수술 비디오 생성**:
- **"HieraSurg: Hierarchy-Aware Diffusion Model for Surgical Video Generation"** (2025):[22]
  - 의료 도메인 계층화: 수술 단계 → 액션 → 패노픽 분할
  - 전문성 요구 작업에 계층 구조 적용[22]

**초상화 비디오**:
- **"ChatAnyone: Stylized Real-time Portrait Video Generation with Hierarchical Motion Diffusion Model"** (2025):[11]
  - 고음질, 낮은 지연 실시간 생성
  - 상반신 동작 동기화[23]

**4D 비디오 합성**:
- **"MV-Performer: Taming Video Diffusion Model for Faithful and Synchronized Multi-view Performer Synthesis"** (2025):[16]
  - 360도 뷰 생성
  - 카메라 의존 법선 맵으로 명확한 조건[19]

#### 6.6 효율성과 확장성

**비용 효율적 훈련**:

- **"Seaweed-7B: Cost-Effective Training of Video Generation Foundation Model"** (2025):[17]
  - 70억 매개변수로 665,000 H100 GPU시간
  - 대규모 모델 수준의 성능을 1/3 비용으로[24]

**메모리 효율성**:

- **"FreeLong: Training-Free Long Video Generation"** (2024):[25]
  - 기존 모델 적응으로 추가 훈련 제거
  - 메모리 제약 상황 대응[26]

#### 6.7 평가 및 벤치마크 발전

**전문가 평가 프레임워크**:

- **"Stable Cinemetrics: Structured Taxonomy and Evaluation for Professional Video Generation"** (2025):[27]
  - 76개 세밀한 제어 노드 정의 (Setup, Event, Lighting, Camera)[27]
  - 80+ 영상 전문가 주석, 20K 비디오 평가[27]

**시간적 일관성 분석**:

- **"Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution"** (2024):[28]
  - 지역 및 전역 시간적 일관성 분리
  - 광학 흐름 기반 시간적 안정성[29]

#### 6.8 미래 방향 분석

**조사 및 리뷰**:

- **"Bibliometric analysis and review of AI-based video generation: research dynamics and application trends (2020–2025)"** (2025):[30]
  - 2020-2025 비디오 생성 연구의 동향 분석
  - 확산 모델이 주요 패러다임으로 확립[30]

***

### 결론

**NUWA-XL**은 "Diffusion over Diffusion" 아키텍처로 장시간 비디오 생성의 근본적 한계를 해결했습니다. 계층적 구조, 직접 훈련, 병렬 추론을 통해 **훈련-추론 불일치를 제거**하고 **94.26% 추론 시간을 단축**했습니다.[1]

이 논문은 이후 연구에 **계층적 확산 설계 패러다임**을 제시했으며, HPDM, ARLON, MovieDreamer 등 후속 연구들이 유사 원리를 확대 적용했습니다.[10][5][11]

향후 연구는 **다양한 도메인 확보**, **고해상도 확장**, **계산 효율성 개선**, **일반화 강화**를 중점으로 진행될 것으로 예상되며, 2024-2025년 새로운 모델들(HunyuanVideo, Seaweed, LongDiff)이 NUWA-XL의 아이디어를 기반으로 산업급 성능을 달성하고 있습니다.[17][9]

***

### 참고 문헌 (검색 결과 기반)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e73870d0-5c36-4659-835d-db8f03dc5aaa/2303.12346v1.pdf)
[2](https://www.semanticscholar.org/paper/4d4a407ba02f93bd04c1cd7c33bf692fa13553e7)
[3](https://arxiv.org/abs/2106.02719)
[4](https://openaccess.thecvf.com/content/CVPR2024/papers/Skorokhodov_Hierarchical_Patch_Diffusion_Models_for_High-Resolution_Video_Generation_CVPR_2024_paper.pdf)
[5](https://arxiv.org/html/2410.20502v1)
[6](https://openaccess.thecvf.com/content/CVPR2024/papers/Fei_Dysen-VDM_Empowering_Dynamics-aware_Text-to-Video_Diffusion_with_LLMs_CVPR_2024_paper.pdf)
[7](https://ieeexplore.ieee.org/document/11093820/)
[8](https://arxiv.org/abs/2506.07136)
[9](http://openaccess.thecvf.com/content/CVPR2025/papers/Li_LongDiff_Training-Free_Long_Video_Generation_in_One_Go_CVPR_2025_paper.pdf)
[10](https://arxiv.org/html/2407.16655v2)
[11](https://arxiv.org/abs/2503.21144)
[12](https://arxiv.org/abs/2501.08316)
[13](https://arxiv.org/html/2306.11173)
[14](https://pmc.ncbi.nlm.nih.gov/articles/PMC10218961/)
[15](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
[16](https://arxiv.org/abs/2510.07190)
[17](https://arxiv.org/abs/2504.08685)
[18](https://huggingface.co/blog/video_gen)
[19](https://arxiv.org/pdf/2204.03458.pdf)
[20](https://proceedings.neurips.cc/paper_files/paper/2022/file/b2fe1ee8d936ac08dd26f2ff58986c8f-Paper-Conference.pdf)
[21](https://arxiv.org/html/2501.03847v2)
[22](https://arxiv.org/abs/2506.21287)
[23](https://ieeexplore.ieee.org/document/11193318/)
[24](http://arxiv.org/pdf/2312.02813.pdf)
[25](https://arxiv.org/abs/2407.19918)
[26](https://en.wikipedia.org/wiki/Text-to-video_model)
[27](https://arxiv.org/abs/2509.26555)
[28](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Upscale-A-Video_Temporal-Consistent_Diffusion_Model_for_Real-World_Video_Super-Resolution_CVPR_2024_paper.pdf)
[29](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_Grid_Diffusion_Models_for_Text-to-Video_Generation_CVPR_2024_paper.pdf)
[30](https://link.springer.com/10.1007/s10791-025-09628-9)
[31](https://ieeexplore.ieee.org/document/11093640/)
[32](https://aclanthology.org/2023.acl-long.73.pdf)
[33](https://arxiv.org/pdf/2203.09481.pdf)
[34](https://arxiv.org/abs/2503.22622)
[35](https://arxiv.org/pdf/2311.11325.pdf)
[36](http://arxiv.org/pdf/2305.13840v1.pdf)
[37](https://geometry.cs.ucl.ac.uk/courses/diffusion_ImageVideo_sigg25/)
[38](https://learnopencv.com/video-generation-models/)
[39](https://aclanthology.org/2023.acl-long.73/)
[40](https://pmc.ncbi.nlm.nih.gov/articles/PMC10606505/)
[41](https://www.youtube.com/watch?v=KRTEOkYftUY)
[42](https://ieeexplore.ieee.org/document/10483854/)
[43](https://ieeexplore.ieee.org/document/10363131/)
[44](https://ieeexplore.ieee.org/document/9885204/)
[45](https://arxiv.org/abs/2505.12667)
[46](https://dl.acm.org/doi/10.1145/3411764.3445721)
[47](https://www.semanticscholar.org/paper/1a3284e3c7bc58a5f453e6573d9107bfb3686b9e)
[48](https://ieeexplore.ieee.org/document/11149948/)
[49](https://arxiv.org/abs/1809.03316)
[50](https://arxiv.org/pdf/2206.00735.pdf)
[51](https://arxiv.org/html/2502.21314)
[52](https://www.aclweb.org/anthology/2020.coling-main.220.pdf)
[53](https://www.mdpi.com/1424-8220/23/23/9452/pdf?version=1701147042)
[54](https://arxiv.org/pdf/2403.17935.pdf)
[55](https://arxiv.org/html/2510.09553)
[56](https://westlake-autolab.github.io/delphi.github.io/)
[57](https://openreview.net/forum?id=y0SRR9XGlZ)
[58](https://arxiv.org/abs/2406.01349)
[59](https://blog.metaphysic.ai/native-temporal-consistency-in-stable-diffusion-videos-with-tokenflow/)
[60](https://snap-research.github.io/hpdm/)
