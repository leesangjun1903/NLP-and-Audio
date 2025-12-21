# PersonaLive! Expressive Portrait Image Animation for Live Streaming

### 1. 논문의 핵심 주장 및 기여

**PersonaLive**는 실시간 라이브 스트리밍 환경에서 고품질의 표현적 인물 애니메이션을 생성하기 위해 설계된 Diffusion 기반 프레임워크입니다.[1]

논문의 핵심 주장은 기존 Diffusion 기반 인물 애니메이션 모델들이 시각적 품질과 표현 사실성 향상에만 집중하면서 **생성 지연시간과 실시간 성능을 간과했다**는 점입니다. 이는 라이브 스트리밍과 같은 실제 응용 분야에서 이들 방법의 실용성을 심각하게 제한합니다.[1]

**주요 기여:**

1. **Hybrid Motion Control**: 암시적 얼굴 표현(implicit facial representations)과 3D 암시적 키포인트(3D implicit keypoints)를 결합하여 미세한 얼굴 역학과 머리 움직임을 동시에 제어할 수 있게 함[1]

2. **Fewer-step Appearance Distillation**: 기존 20단계 Denoising을 4단계로 축소하면서도 시각 품질을 보존하는 전략을 제안[1]

3. **Micro-chunk Streaming Generation**: 슬라이딩 훈련 전략(Sliding Training Strategy, ST)과 과거 키프레임 메커니즘(Historical Keyframe Mechanism, HKM)을 통해 노출 편향(exposure bias)과 오류 누적을 완화[1]

***

### 2. 해결하는 문제 및 제안 방법 (수식 포함)

#### 2.1 문제 정의

스트리밍 인물 애니메이션의 목표는 주어진 참조 이미지 $I_R$과 연속적인 S개의 운전 프레임 $\{I_1^D, I_2^D, ..., I_S^D\}$에서 애니메이션 시퀀스 $A_{\{1,2,...,S\}}$를 생성하는 것입니다:[1]

$$A^i = D(M(I_i^D), R(I_R)), \quad i = 1, 2, ..., S$$

여기서 $D$는 Denoising 백본, $M$은 운동 추출기, $R$은 외관 추출기입니다.[1]

#### 2.2 3D 암시적 키포인트 변환

3D 파라미터 추출 후 다음과 같이 변환됩니다:[1]

$$k_d = s_d \cdot k_{c,s}R_d + t_d$$

여기서 $k_c$는 정규 키포인트, $R$, $t$, $s$는 각각 회전, 이동, 스케일 파라미터입니다.[1]

#### 2.3 Fewer-step Appearance Distillation

원본 이미지 $x$는 먼저 VAE 인코더로 잠재 표현 $z = V_e(x)$로 변환됩니다. Diffusion 과정은:

$$z_t = \sqrt{\bar{\alpha}_t}z + \sqrt{1-\bar{\alpha}_t}\epsilon$$

$$t \in $$

여기서 $\bar{\alpha}_t$는 사전정의된 노이즈 스케줄입니다.[1]

Distillation 손실 함수는:

$$L_{distill} = L_2(\hat{x}, x_{gt}) + \lambda_{lpips}L_{lpips}(\hat{x}, x_{gt}) + \lambda_{adv}L_{adv}(\hat{x})$$

여기서 $\lambda_{lpips} = 2.0$, $\lambda_{adv} = 0.05$입니다.[1]

#### 2.4 Micro-chunk 스트리밍 세대

Denoising 윈도우는 N개의 Micro-chunk로 구성됩니다:

$$W_s = \{C_1^s, C_2^s, ..., C_N^s\}$$

$$C_n^s = \{z_{t_i}^n | i = 1, 2, ..., M\}, \quad t_1 < t_2 < ... < t_N$$

여기서 각 Micro-chunk는 M개의 프레임으로 구성됩니다.[1]

#### 2.5 Sliding Training Strategy

초기 Denoising 윈도우의 처음 N-1개 청크는:

$$C_n^0 = \{\sqrt{\bar{\alpha}_{t_n}}z_{gt}^i + \sqrt{1-\bar{\alpha}_{t_n}}\epsilon_i\}^M_{i=1}$$

여기서 $\epsilon_i \sim N(0, I)$, $\alpha_{t_n}$은 노이즈 스케줄링 파라미터입니다.[1]

마지막 청크는 순수 가우시안 노이즈 $C_{noise}$로 초기화됩니다.[1]

#### 2.6 Historical Keyframe Mechanism (HKM)

현재 모션 임베딩 $m_f$와 모션 뱅크 $B_{mot}$의 유사성:

$$d = \min_{i=0,1,...}||m_f - m_i||_2$$

임계값 $\tau = 17$을 초과하면 현재 프레임이 키프레임으로 선택됩니다.[1]

#### 2.7 Motion-Interpolated Initialization (MII)

초기 Denoising 윈도우의 i번째 프레임의 보간된 임베딩:

$$m_{f,i} = (1-\omega_i)m_{f,s} + \omega_i m^1_{f,d}$$

$$\omega_i = \frac{i-1}{MN-1}$$

3D 변환 파라미터의 보간:

$$R_i = R((1-w_i)\theta_s + w_i\theta_d)$$

$$s_i = (1-w_i)s_s + w_is_d$$

$$t_i = (1-w_i)t_s + w_it_d$$

$$k_i = s_iK_{c,s}R_i + t_i$$

여기서 $\theta = (\text{pitch}, \text{yaw}, \text{roll})$는 오일러 각도입니다.[1]

***

### 3. 모델 구조 및 아키텍처

PersonaLive는 3단계 파이프라인으로 구성됩니다:[1]

#### Stage 1: Image-level Hybrid Motion Training
- **얼굴 모션 추출**: 운전 이미지에서 1D 얼굴 모션 임베딩 $m_f = E_f(I_D)$를 추출하여 Cross-attention을 통해 주입[1]
- **3D 키포인트 제어**: 3D 파라미터 $\{k_c, R, t, s\}$를 추출하여 PoseGuider를 통해 주입[1]

#### Stage 2: Fewer-step Appearance Distillation
- 컴팩트 샘플링 스케줄 $\{t_i\}^N_{i=1}$ 도입 (N=4)[1]
- 예측 이미지 $\hat{x} = V_d(\hat{z}_0)$에 하이브리드 손실 함수 적용[1]
- 메모리 효율성을 위해 최종 Denoising 단계에만 역전파 실시[1]

#### Stage 3: Micro-chunk Streaming Video Generation
- **Temporal Module**: Diffusion 백본에 통합[1]
- **Sliding Window**: 각 단계 후 모든 청크가 낮은 노이즈 수준으로 이동하고, 첫 청크에서 M개의 깨끗한 프레임 방출[1]
- **History Bank**: 이전 생성된 결과에서 추출한 참조 특징 { $h_0, h_1, ...$ }과 모션 임베딩 { $m_0, m_1, ...$ } 유지[1]

#### 네트워크 아키텍처 특성
- **참조 네트워크**: 사전 훈련된 Diffusion 모델의 생성 사전(generative prior) 활용[1]
- **판별자**: StyleGAN2 아키텍처, FFHQ 데이터셋으로 사전 훈련[1]
- **학습 환경**: 8개 Nvidia H100 GPU, AdamW optimizer (학습률: $1 \times 10^{-5}$ , 가중치 감쇠: 0.01)[1]

***

### 4. 성능 향상 및 한계

#### 4.1 성능 향상 결과

| 지표 | PersonaLive | X-NeMo | HunyuanPortrait | Follow-YE | Megactor-Σ |
|------|-------------|---------|-----------------|-----------|-----------|
| **FPS** ↑ | **15.82** | 1.281 | 1.443 | 1.558 | 2.216 |
| **Latency (s)** ↓ | **0.253** | 15.32 | 14.91 | 7.793 | 6.918 |
| **LPIPS** ↓ | **0.129** | 0.267 | 0.137 | 0.144 | 0.183 |
| **tLP** ↓ | **21.31** | 25.11 | 22.33 | 26.92 | 23.55 |
| **FVD** ↓ | **520.6** | 639.1 | 620.4 | 696.5 | 585.3 |

**TinyVAE 사용 시 20 FPS 달성** (비용은 약간의 품질 저하)[1]

#### 4.2 Ablation Study 결과 분석

**Micro-chunk Streaming 요소별 기여도:**

| 설정 | ID-SIM ↑ | AED ↓ | APD ↓ | FVD ↓ | tLP ↓ |
|------|----------|-------|-------|-------|-------|
| w/ ChunkAttn | 0.689 | 0.709 | 0.032 | 537.0 | 12.83 |
| ChunkSize=2 | 0.660 | 0.713 | 0.031 | 520.2 | 12.14 |
| w/o MII | 0.680 | 0.703 | 0.031 | 511.5 | 13.06 |
| w/o HKM | 0.728 | 0.710 | 0.031 | 535.6 | 13.27 |
| **w/o ST** | **0.549** | **0.785** | **0.040** | **678.8** | **10.05** |
| **Ours** | **0.698** | **0.703** | **0.030** | **520.6** | **12.83** |

Sliding Training Strategy 제거 시 ID-SIM이 0.549로 급격히 하락하여 ST의 중요성을 확인.[1]

#### 4.3 모델의 일반화 성능

**자기-재현(Self-Reenactment) vs 크로스-재현(Cross-Reenactment):**

- **자기-재현**: 첫 프레임을 참조 이미지로 사용한 재현 능력
- **크로스-재현**: 다른 인물로의 일반화 능력 (LV100 벤치마크 사용)

PersonaLive는 크로스-재현에서 FVD 520.6, tLP 12.83으로 우수한 시간적 일관성을 달성하며, 다양한 신원에 대한 우수한 일반화 능력을 입증.[1]

#### 4.4 주요 한계

1. **시간적 중복성 미활용**: 연속 프레임 간의 시간적 중복성을 명시적으로 활용하지 않음[1]
2. **도메인 일반화의 제약**: 
   - 훈련 데이터가 주로 인간 얼굴 데이터에 기반
   - **만화, 동물 등 도메인 외(out-of-domain) 이미지에서 생성 실패** (Fig. 8 참고)[1]
   - 훈련 도메인 외 인물(예: 비인간 외관)의 경우 눈, 입 등이 흐릿하거나 왜곡될 수 있음[1]

***

### 5. 일반화 성능 향상 가능성

#### 5.1 현재 제약 사항

논문에서 명시된 일반화 성능의 제약:

- **데이터셋 편향**: VFHQ, NerSemble, DH-FaceVid-1K는 모두 인간 얼굴 데이터에 집중[1]
- **Domain Shift**: 비인간 얼굴(만화, 그림체, 동물)에 대한 성능 저하 심각[1]

#### 5.2 향후 개선 방향

1. **다양한 도메인 데이터 확대**
   - 만화/애니메이션 캐릭터 데이터셋 추가
   - 동물 얼굴 데이터 학습

2. **Domain Adaptation 기법 도입**
   - Style Transfer 기반 적응
   - Few-shot Learning을 통한 빠른 적응

3. **Self-Supervised Learning 활용**
   - 레이블 없는 다양한 도메인 데이터 활용
   - Contrastive Learning으로 표현 개선

4. **확장 가능한 아키텍처**
   - 도메인-특화 모듈 추가
   - 적응형 특징 추출

***

### 6. 최신 연구와의 비교 분석 (2020년 이후)

#### 6.1 주요 경쟁 방법들

| 방법 | 출시연도 | 주요 특징 | 성능 |
|------|---------|---------|------|
| **PersonaLive** | 2025 | Micro-chunk streaming, 4-step denoising | **15.82 FPS, 0.253s latency** |
| **X-NeMo** | 2025 | Disentangled latent attention | 1.281 FPS, 15.32s |
| **HunyuanPortrait** | 2025 | Implicit condition control | 1.443 FPS, 14.91s |
| **Follow-your-Emoji** | 2024 | Progressive generation strategy | 1.558 FPS, 7.79s |
| **Megactor-Σ** | 2025 | Mixed-modal control | 2.216 FPS, 6.92s |
| **Hallo3** | 2024 | Video Diffusion Transformer | ~10 FPS (추정) |
| **StreamDiT** | 2025 | T2V streaming | **16 FPS** |
| **MotionStream** | 2025 | Motion-controlled streaming | **29 FPS** |

#### 6.2 PersonaLive의 위치

**강점:**
- **인물 애니메이션 특화**: 최고의 실시간 성능 (15.82 FPS)[1]
- **저지연시간**: 0.253초로 라이브 스트리밍에 최적화[1]
- **우수한 시간적 일관성**: tLP 12.83으로 장시간 안정성 입증[1]

**약점:**
- **도메인 외 일반화**: 비인간 얼굴에 성능 저하[1]
- MotionStream (29 FPS)보다 느린 속도[2]

#### 6.3 새로운 트렌드 분석

1. **Streaming Paradigm의 부상** (StreamDiT, MotionStream, PersonaLive)
   - 기존 프레임 단위 생성 → 청크 단위 스트리밍 생성으로 전환
   - 상태 추적(stateful) 아키텍처 도입

2. **Diffusion Transformer 확대**
   - CNN 기반에서 Transformer 기반으로 전환 (Hallo3, Hallo4)
   - 더 나은 long-range dependency 모델링

3. **Multi-modal Control 강화**
   - 음성 + 비디오 + 텍스트 결합 (Hallo4, MegActor-Σ)[3][4]

4. **효율성 최적화 경쟁**
   - Distillation 기법 확대 (Few-step denoising)
   - 양자화 및 모델 압축

#### 6.4 Audio-Driven 애니메이션의 발전

- **JoyVASA** (2024): 오디오 주도 얼굴 역학 생성[3]
- **Hallo4** (2025): Direct Preference Optimization으로 인간 선호도 정렬[5]
- **MEDTalk** (2025): 감정 제어 기능 추가[6]

***

### 7. 향후 연구에 미치는 영향 및 고려사항

#### 7.1 학술적 영향

1. **Streaming 패러다임의 표준화**
   - PersonaLive의 Micro-chunk 접근은 향후 비디오 생성 모델의 표준 방식이 될 가능성
   - Autoregressive 생성에서 노출 편향 제거 전략이 다른 모달리티로 확산

2. **Distillation 기법의 재정의**
   - Few-step denoising이 품질을 유지하면서 성능을 극적으로 개선할 수 있음을 증명
   - 다른 생성 모델 (T2I, T2V, T2A)에 적용 가능성

3. **Hybrid Control 신학**
   - 암시적 표현 + 3D 기하학 결합이 유효함을 입증
   - 향후 다중 조건 제어의 표준 접근

#### 7.2 실무 적용 전망

1. **라이브 스트리밍 인플루언서**
   - 0.253초의 지연시간으로 실시간 상호작용 가능
   - Virtual Influencer 시장 확대 가능성

2. **메타버스/게임 엔진**
   - 실시간 얼굴 재현
   - NPC 애니메이션 자동 생성

3. **원격 교육/회의**
   - 진정한 의미의 실시간 아바타 시스템
   - 접근성 향상

#### 7.3 향후 연구 시 고려할 점

1. **도메인 일반화 개선 필수**
   ```
   - 인간 외의 도메인에 대한 전이학습 전략 개발
   - 소량 데이터로 빠른 적응 가능한 메커니즘
   - Multi-domain 동시 학습 (meta-learning)
   ```

2. **신체 동작 확장**
   - 현재: 포트레이트(얼굴 중심)
   - 향후: 상체, 전신 애니메이션으로 확장
   - 배경 제어 추가

3. **효율성 극대화**
   ```
   - Mobile Device 배포 (현재: H100 GPU 필요)
   - 엣지 컴퓨팅 최적화
   - 동적 양자화 기법
   ```

4. **시간적 중복성 활용**
   - 논문에서 명시된 향후 연구 방향
   - 연속 프레임의 구조적 유사성 활용한 재귀적 특징 공유
   - Temporal-aware 인코딩

5. **윤리 및 안전성**
   - 비동의 콘텐츠 생성 방지
   - 딥페이크 탐지 기법과의 협력
   - 생성 콘텐츠의 투명한 표시 메커니즘

6. **평가 메트릭 개선**
   - 현재: FVD, tLP, LPIPS 등 객관적 지표
   - 향후: 인간 선호도 기반 평가 (Hallo4의 DPO 접근 참고)
   - 도메인별 특화 평가 척도

***

### 8. 결론

**PersonaLive**는 Diffusion 기반 인물 애니메이션에서 **실시간성과 품질의 균형을 처음으로 달성한 방법**입니다. Fewer-step appearance distillation과 micro-chunk streaming paradigm은 향후 비디오 생성 연구의 중요한 기준점이 될 것으로 예상됩니다.

그러나 도메인 외 일반화의 제약, 신체 전체로의 확장 필요성, 그리고 모바일 배포의 어려움은 차세대 연구가 극복해야 할 과제입니다. 특히 **시간적 중복성의 명시적 활용**과 **다양한 도메인으로의 적응**이 다음 세대 방법의 핵심이 될 것으로 판단됩니다.

***

### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/500389c3-1149-4da6-a2d4-110e33600b92/2512.11253v1.pdf)
[2](https://www.emergentmind.com/papers/2511.01266)
[3](https://arxiv.org/html/2411.09209)
[4](https://arxiv.org/html/2408.14975)
[5](https://arxiv.org/html/2505.23525v4)
[6](https://arxiv.org/html/2507.06071v3)
[7](https://arxiv.org/abs/2403.15931)
[8](https://arxiv.org/html/2502.10841v1)
[9](https://arxiv.org/html/2406.01900)
[10](https://cumulo-autumn.github.io/StreamDiT/)
[11](https://liner.com/ko/review/hallo3-highly-dynamic-and-realistic-portrait-image-animation-with-video)
[12](https://arxiv.org/html/2406.01188)
[13](https://arxiv.org/pdf/2311.16498.pdf)
[14](https://arxiv.org/html/2504.01724)
[15](https://cvpr.thecvf.com/virtual/2025/poster/34789)
[16](https://pmc.ncbi.nlm.nih.gov/articles/PMC9203171/)
[17](https://huggingface.co/papers/2501.09756)
[18](https://pmc.ncbi.nlm.nih.gov/articles/PMC12316851/)
[19](https://joonghyuk.com/motionstream-web/)
[20](https://deepai.org/publication/expressive-speech-driven-facial-animation-with-controllable-emotions)
[21](https://arxiv.org/abs/2412.00733)
[22](https://arxiv.org/abs/2507.03745)
[23](https://arxiv.org/html/2305.03216v3)
[24](https://arxiv.org/html/2512.11645v1)
[25](https://arxiv.org/abs/2511.01266)
[26](https://arxiv.org/html/2409.13180v1)
[27](https://arxiv.org/html/2512.16900)
[28](https://arxiv.org/html/2508.13009v3)
[29](https://arxiv.org/html/2508.05115v1)
[30](https://z.ai/blog/realvideo)
