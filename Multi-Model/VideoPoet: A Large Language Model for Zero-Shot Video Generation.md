# VideoPoet: A Large Language Model for Zero-Shot Video Generation

### 1. 핵심 주장 및 주요 기여 요약

**VideoPoet의 중심 주장**은 확산 모델 중심의 비디오 생성 패러다임에서 벗어나 **대규모 언어모델(LLM) 아키텍처가 비디오 생성에서 경쟁력 있는 성능을 달성할 수 있다**는 것입니다. 이는 LLM이 다중모달 입력(텍스트, 이미지, 비디오, 오디오)을 처리하여 통합된 단일 모델 내에서 다양한 비디오 생성 작업을 수행할 수 있음을 시사합니다.[1]

**주요 기여**는 다음 세 가지입니다:
1. 텍스트 쌍이 있는 비디오와 없는 비디오 데이터를 모두 활용하여 비디오 생성용 LLM을 훈련하는 방법
2. 양방향 트랜스포머와 효율적인 윈도우 로컬 어텐션을 사용하여 잠재 토큰 공간에서 공간 해상도를 증가시키는 기술
3. 훈련 데이터 분포에서 벗어난 새로운 입력에 대한 제로샷 성능 및 작업 체이닝을 통한 새로운 편집 작업 수행 능력 입증[1]

***

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하고자 하는 문제

**문제 1: 확산 모델의 제한성** - 기존 비디오 생성 모델은 확산 기반 방식 위주로, 작업별로 별도의 적응 모듈이 필요하며, 텍스트-이미지 생성 확산 모델에서 출발하여 시간 일관성을 위해 추가 튜닝이 필요합니다.[1]

**문제 2: 단일 통합 모델의 부재** - 텍스트-비디오, 이미지-비디오, 비디오 스타일화, 비디오 편집 등 다양한 작업을 하나의 통합된 프레임워크로 수행하기 어렵습니다.[1]

**문제 3: 제로샷 성능 및 작업 일반화의 한계** - 기존 모델들이 훈련 분포 외의 새로운 입력에 대한 일반화 성능이 제한적입니다.[1]

#### 2.2 제안하는 방법 및 수식

**MAGVIT-v2 토큰화 (시각)**[1]
- 17프레임, 2.125초, 128×128 해상도 비디오를 (5, 16, 16) 형태의 잠재 표현으로 인코딩
- 1280개의 토큰 생성, 어휘 크기: \( 2^{18} = 262,144 \)

**SoundStream 토큰화 (오디오)**[1]
- 2.125초 오디오를 106개 잠재 프레임으로 인코딩
- 4개 레벨 잔여 벡터 양자화(RVQ): 각 레벨 1,024개 코드
- 총 오디오 어휘: \( 4 \times 1,024 = 4,096 \)

**통합 어휘**[1]
- 특수 토큰: 256개, 시각 토큰: 262,144개, 오디오 토큰: 4,096개
- **총 어휘 크기: 약 300,000**

**자동회귀 생성 과정**:[1]

$$P(v_1, v_2, \ldots, v_T | c) = \prod_{t=1}^{T} P(v_t | v_1, \ldots, v_{t-1}, c)$$

여기서 \( v_t \)는 시간 t에서의 토큰, \( c \)는 조건 정보(텍스트, 이미지, 오디오)입니다.

#### 2.3 모델 구조

**디코더 전용 트랜스포머**[1]
- 다중모달 입력(T5 텍스트 임베딩, MAGVIT-v2 시각 토큰, SoundStream 오디오 토큰)을 처리
- 시각 및 오디오 토큰 출력을 자동회귀적으로 생성
- 입력 시퀀스: 양방향 어텐션, 출력 시퀀스: 인과 마스킹

**초해상도 모듈**[1]
- 3개 트랜스포머 레이어 블록: 공간 수직 윈도우, 공간 수평 윈도우, 시간 로컬 어텐션
- 토큰 인수분해(k=2)로 262,144방향 분류를 512방향 분류 2개로 변환
- 캐스케이드 구조: 224×128 → 448×256 → 896×512

**두 단계 샘플링 전략**[1]
- 첫 25% 훈련: 이미지 90%, 비디오 10% (물체 인식 향상)
- 나머지 75%: 이미지 10%, 비디오 90% (동작 학습)

**교대 경사하강법(AGD)**[1]
훈련 시퀀스 길이 그룹별로 교대로 샘플링하여 패딩 비율을 약 0%로 유지합니다.

#### 2.4 성능 향상 결과

**텍스트-비디오 생성 (MSR-VTT)**[1]

| 모델 | CLIP Similarity ↑ | FVD ↓ |
|------|------------------|-------|
| CogVideo (2022) | 0.263 | 1,294 |
| Show-1 (2023) | 0.307 | 538 |
| VideoPoet (Pretrain) | 0.305 | 213 |
| VideoPoet (Task Adapt) | 0.312 | - |

**인적 평가 (T2V 생성)** - VideoPoet는 **동작 흥미도**(48-82% 범위)와 **동작 사실성**(39-84% 범위)에서 특히 강력한 성능을 보였습니다.[1]

**프레임 예측 (Kinetics-600)**[1]

| 방법 | FVD ↓ |
|------|-------|
| T2V 전용 | 759 |
| 모든 태스크 포함 | 729 |

**모델 규모별 성능**[1]

| 모델 크기 | 훈련 데이터 토큰 | FVD (비디오) ↓ |
|----------|------------|------------|
| 300M | 10B | ~1,085 |
| 1B | 37B | ~500 |
| 8B | 58B | 355 |

#### 2.5 한계

VideoPoet의 주요 한계는:[1]

1. **토큰 압축의 한계** - RGB 프레임의 압축 및 양자화로부터 상한선 설정과 미세한 세부사항 손실
2. **정적 장면의 미학 편차** - 프레임 레벨 미학 편차가 기준선과 일치하지 않음
3. **소형 물체 및 세부 묘사** - 큰 동작이 동반된 작은 물체와 세부사항 표현의 어려움
4. **인페인팅 초기 성능** - 디코딩 붕괴(반복적 토큰 예측) 현상

***

### 3. 일반화 성능 향상 가능성

#### 3.1 제로샷 성능의 메커니즘

**다중 작업 사전훈련의 긍정적 효과**:[1]
- T2V 단독: CLIP 유사도 0.244
- 모든 작업 포함: CLIP 유사도 0.240 (전체 벤치마크에서 평균 최고)

교차-작업 일반화를 통한 기본 표현 학습이 핵심입니다.

**자가 감독 학습(SSL) 태스크**[1]
미래 예측, 인페인팅, 음성-비디오 연속 작업이 텍스트 쌍 없는 데이터로부터 모션 다이나믹스에 대한 기초 이해를 제공합니다.

**장시간 비디오 자동회귀 확장**[1]
최근 1초를 조건으로 다음 1초를 자동회귀적으로 확장하여 10초 이상의 비디오를 생성하면서도 반복적 확장으로 시각적 일관성을 유지합니다.

#### 3.2 교차-작업 체이닝을 통한 제로샷 일반화

**새로운 작업 조합**:[1]
- 이미지-비디오 + 비디오-비디오 스타일화: 정적 이미지 → 애니메이션 → 스타일화
- 비디오-비디오 아웃페인팅 + 비디오 편집: 확장 후 추가 효과

각 단계 출력이 다음 단계의 입력으로 사용되며, 출력이 동일 분포 유지로 인해 연쇄적 적용이 가능합니다.

#### 3.3 모델 규모에 따른 일반화 개선

**1B vs 8B 모델 비교**:[1]
8B 모델은 시간 일관성, 프롬프트 충실도, 동작 다이나믹스, 텍스트 렌더링 능력, 공간 이해, 개수 세기 등 모든 면에서 1B 모델을 능가합니다.

**스케일링 법칙**: \( \text{FVD} \propto (D \cdot M)^{-\alpha} \) (α ≈ 0.3-0.4)

#### 3.4 제한된 일반화: "사례 기반" 행동

**최근 연구 발견** (2024 "How Far is Video Generation from World Model"):[2]

비디오 생성 모델의 일반화는 다음과 같이 분류됩니다:[2]
1. **분포 내 일반화**: 우수
2. **조합론적 일반화**: 측정 가능하나 제한적
3. **분포 외 일반화**: 실패

모델이 기본 물리 법칙을 학습하지 않고 유사한 훈련 예제를 모방하며, 단순히 스케일링만으로는 이 문제를 해결할 수 없습니다.[2]

**우선순위 순서** (새로운 케이스 참조 시): 색상 > 크기 > 속도 > 형태[2]

***

### 4. 앞으로의 연구 영향과 고려사항

#### 4.1 비디오 생성 분야에 미치는 영향

**패러다임 전환**[3]

| 연도 | 주요 개발 | 영향 |
|------|---------|------|
| 2023 | Stable Video Diffusion (확산 기반) | 비디오 확산 모델 표준화 |
| 2024 | Sora (텍스트 조건 확산) | 변수 길이/해상도 생성 |
| 2024-2025 | 자동회귀 모델 부흥 | **VAR, Infinity, InfinityStar** |
| 2025 | HunyuanVideo, Wan (확산 + 트랜스포머) | **하이브리드 접근법** |

최근 모델들이 LLM 트랜스포머와 확산의 하이브리드 방식으로 수렴하고 있습니다.[3]

**통합 다중 작업 프레임워크의 영향**:[4][5]
- VideoDirectorGPT (2024): LLM을 통한 다중 장면 비디오 생성 계획
- iVideoGPT (2024): 상호작용 가능한 세계 모델
- InfinityStar (2025 NeurIPS Oral): VAR 기반 텍스트-비디오 생성

#### 4.2 최신 연구 기반 제언사항

**일반화 성능 향상 전략**:[2]

1. **물리 인식 사전훈련** - 합성 데이터 기반 물리 시뮬레이션과 구조화된 동작 표현 학습
2. **부트스트랩 강화** - 인적 피드백 활용(VideoReward, 2025)[6]
3. **다중 규모 표현** - 계층적 토큰 모델링과 조건부 독립성 구조 학습

**토큰 기반 모델의 진화 방향**:[3]
- 벡터 양자화 없는 연속 토큰(Autoregressive Image Generation, 2024)
- 적응형 토크나이저(Pandora 모델)
- 다계층 RVQ 개선

**모델 규모화 전략**(HunyuanVideo, Wan 2025):[7][8]

병렬화 기법의 중요성:
- Fully-Sharded Data Parallel (FSDP)
- Context Parallel (CP): 시퀀스 길이 확장성
- Tensor Parallelism

스케일링 법칙 재정의:
- 확산 모델은 LLM보다 하이퍼파라미터에 민감
- 배치 크기, 학습률에 따른 민감도 높음[3]

**제로샷 성능 극대화**:[1]
1. 다양한 사전훈련 목표(무조건 생성, 조건부 생성, 자가 감독 학습, 마스킹)
2. 프롬프트 엔지니어링(체인-오브-쏘트, 음수 프롬프트, 자동 재작성)
3. 작업 체이닝 최적화(중간 표현의 분포 내 유지, 교사 강제)

#### 4.3 근본적 연구 방향

**의미론적 이해와 인과관계 학습**
- 구조적 인과 모델(SCM) 통합
- 개입적 데이터(Interventional Data) 활용
- 물리 기반 손실 함수 설계

**공정성과 편향 완화**[1]
VideoPoet의 발견: "Young Adults"(18-35), "Male", "Light Skin Tone" 분포 편향
- 다양성 증대 데이터 수집
- 공정성 인식 손실 함수
- 사후 처리 보정 기법

***

### 5. 최종 평가

VideoPoet은 단순한 논문 이상으로 **비디오 생성의 패러다임 전환을 촉발한 선도 연구**입니다. 그 한계가 명확함에도 불구하고, 다중모달 LLM의 유연성과 확장성이 비디오 생성 분야에서도 실현 가능함을 보여주었습니다. 현재의 후속 연구들(VAR, HunyuanVideo, InfinityStar 등)이 이 기초 위에서 더욱 정교해지고 있으며, 향후 5-10년 내 비디오 생성이 이미지 생성 수준의 성숙도에 도달할 것으로 예상됩니다.[3][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e1a6b517-539c-4660-ad8e-c70f7637dbf8/2312.14125v4.pdf)
[2](https://arxiv.org/abs/2411.02385)
[3](https://yenchenlin.github.io/blog/2025/01/08/video-generation-models-explosion-2024/)
[4](https://openreview.net/forum?id=sKNIjS2brr)
[5](https://arxiv.org/html/2405.15223)
[6](https://arxiv.org/pdf/2501.13918.pdf)
[7](https://arxiv.org/html/2412.03603v2)
[8](https://arxiv.org/abs/2503.20314)
[9](https://arxiv.org/pdf/2312.14125.pdf)
[10](https://arxiv.org/html/2503.02341v1)
[11](http://arxiv.org/pdf/2403.15377v4.pdf)
[12](http://arxiv.org/pdf/2412.08879.pdf)
[13](https://arxiv.org/pdf/2306.05424.pdf)
[14](https://arxiv.org/abs/2409.12499)
[15](https://arxiv.org/pdf/2310.12724.pdf)
[16](https://proceedings.mlr.press/v235/kondratyuk24a.html)
[17](https://www.amazon.science/publications/zero-shot-customized-video-editing-with-diffusion-feature-transfer)
[18](https://research.google/blog/videopoet-a-large-language-model-for-zero-shot-video-generation/)
[19](https://huggingface.co/blog/video_gen)
[20](https://ditflow.github.io)
[21](https://dl.acm.org/doi/10.5555/3692070.3693075)
[22](https://homangab.github.io/gen2act/Gen2Act-Paper.pdf)
[23](https://openaccess.thecvf.com/content/CVPR2023W/L3D-IVU/papers/Doshi_Zero-Shot_Action_Recognition_With_Transformer-Based_Video_Semantic_Embedding_CVPRW_2023_paper.pdf)
[24](https://arxiv.org/html/2507.13942v1)
[25](https://arxiv.org/html/2503.17539v1)
[26](https://arxiv.org/pdf/2205.09853.pdf)
[27](https://arxiv.org/pdf/2311.15127.pdf)
[28](http://arxiv.org/pdf/1906.02634v1.pdf)
[29](https://arxiv.org/pdf/2401.09084.pdf)
[30](https://www.ijcai.org/proceedings/2023/0642.pdf)
[31](https://www.sciencedirect.com/science/article/abs/pii/S003132032500562X)
[32](http://pengxi.me/wp-content/uploads/2020/12/2020TNNLS-Deep-Multimodal-Transfer-Learning.pdf)
[33](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Towards_Fast_Adaptation_of_Pretrained_Contrastive_Models_for_Multi-Channel_Video-Language_CVPR_2023_paper.pdf)
[34](https://openai.com/index/video-generation-models-as-world-simulators/)
[35](https://dl.acm.org/doi/10.1145/3078971.3078994)
[36](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04830.pdf)
[37](https://github.com/FoundationVision/VAR)
