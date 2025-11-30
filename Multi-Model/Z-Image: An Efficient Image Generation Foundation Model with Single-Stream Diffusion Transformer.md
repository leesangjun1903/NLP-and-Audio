# Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer

### 1. 핵심 주장 및 주요 기여

**Z-Image**는 대규모 매개변수 확대(scale-at-all-costs) 패러다임에 도전하는 **6B 매개변수 기반의 효율적 이미지 생성 기초 모델**입니다. 이 논문의 핵심 주장은 체계적인 최적화를 통해 **상태 최고 수준의 성능을 훨씬 적은 계산 비용으로 달성할 수 있다**는 것입니다.[1]

주요 기여는 다음과 같습니다:

- **비용 효율성**: 전체 훈련을 **314K H800 GPU 시간(약 $630K)**에 완료하여, 20B~80B 매개변수의 경쟁 모델 대비 획기적으로 낮은 비용 달성[1]

- **S3-DiT 아키텍처**: 단일 스트림 멀티모달 확산 트랜스포머로 매개변수 효율성 극대화[1]

- **Z-Image-Turbo**: **8단계 추론으로 부 1초 지연시간**을 달성하면서도 품질 유지[1]

- **포토리얼리즘 및 이중언어 텍스트 렌더링**: 상업용 최고급 모델과 견줄 수 있는 성능[1]

***

### 2. 해결하고자 하는 문제 및 제안 방법

#### 2.1 문제점

현재 이미지 생성 모델 생태계는 두 가지 극단으로 양극화되어 있습니다:[1]

1. **폐쇄 소스 상업 모델** (Nano Banana Pro, Seedream 4.0): 높은 성능이지만 투명성 부족
2. **오픈소스 모델** (Qwen-Image 20B, FLUX.2 32B, Hunyuan-Image-3.0 80B): 수십억 개의 매개변수로 인해 훈련 및 추론이 비현실적

이로 인해 자원 제한 연구가 다른 모델에서 합성 데이터를 증류하는 폐쇄 루프에 의존하게 되어 에러 누적과 데이터 동질화를 초래합니다.[1]

#### 2.2 제안 방법: 네 가지 전략적 기둥

Z-Image는 모델 생명주기의 모든 단계를 최적화하는 통합 솔루션을 제시합니다.[1]

**① 효율적 데이터 인프라**[1]

네 개의 시너지적 모듈로 구성:

- **데이터 프로파일링 엔진**: 이미지 메타데이터, 기술적 품질, 미적 품질, AIGC 콘텐츠 탐지 등 다차원 특성 추출
- **교차모달 벡터 엔진**: k-NN 기반 의미론적 중복 제거로 **10억 항목당 약 8시간** 처리 속도 달성
- **세계 지식 위상 그래프**: Wikipedia 엔티티 및 이미지 캡션 기반 계층적 개념 조직
- **능동적 큐레이션 엔진**: 모델 성능 진단으로 장꼬리 개념 식별 및 동적 데이터 보강

**② 효율적 아키텍처: S3-DiT (Scalable Single-Stream DiT)**[1]

기존 이중 스트림 아키텍처와 달리, 모든 모달리티를 **단일 입력 스트림으로 연결**:

$$x_{\text{unified}} = [\text{text tokens}, \text{VAE tokens}, \text{semantic tokens}]$$

이는 다음과 같은 이점을 제공합니다:[1]

- **매개변수 효율성**: 쌍방향 모달 상호작용이 모든 계층에서 발생하여, 6B 매개변수로 20B~32B 모델과 경쟁
- **3D Unified RoPE**: 이미지 토큰은 공간 차원 확장, 텍스트 토큰은 시간 차원 증가로 혼합 시퀀스 모델링
- **정규화 기법**: QK-Norm, Sandwich-Norm, RMSNorm으로 훈련 안정성 확보[1]

**③ 효율적 훈련 전략: 프로그레시브 커리큘럼**[1]

$$L = \mathbb{E}_{t, x_0, x_1, y}\left[\|u(x_t, y, t; \theta) - (x_1 - x_0)\|_2\right]$$

여기서 $x_t = t \cdot x_1 + (1-t) \cdot x_0$ (플로우 매칭 목표)

세 가지 단계로 구성:[1]

- **저해상도 사전훈련** ($256^2$ 해상도): 기초 시각-의미론적 정렬 및 지식 주입
- **전방위 사전훈련**: 
  - 임의 해상도 훈련
  - 텍스트-이미지 및 이미지-이미지 결합 학습
  - 다단계, 이중언어 캡션 훈련
- **감독된 미세조정 (SFT)**: 분포 좁히기로 고품질 세부 초점 달성

**④ 효율적 추론: 분리된 DMD 및 DMDR**[1]

표준 DMD의 두 가지 독립적 메커니즘을 분리:

$$\text{DMD Loss} = \lambda_{\text{CA}} \cdot L_{\text{CA}} + \lambda_{\text{DM}} \cdot L_{\text{DM}}$$

- **CFG-증강 (CA)**: 소수 단계 생성 능력 개발
- **분포 매칭 (DM)**: 훈련 안정성 및 아티팩트 제거

이를 통해 Z-Image-Turbo는 **8 NFE(함수 평가 회수)에서 100 NFE 모델 수준의 품질** 달성[1]

추가적으로 DMDR은 RL을 분포 매칭 정규화와 결합하여 보상 해킹 방지[1]

***

### 3. 모델 구조 상세 분석

#### 3.1 아키텍처 구성

**S3-DiT의 핵심 구조:**[1]

```
Input Modalities
    ↓
Modality-Specific Processors (각 2개 트랜스포머 블록)
    ↓
Unified Single-Stream Backbone (30 계층)
    ├─ Single-Stream Attention Block
    │  ├─ QK-Norm (attention 정규화)
    │  └─ Sandwich-Norm (입출력 신호 제약)
    ├─ Single-Stream FFN Block
    │  ├─ 저랭크 분해 조건 프로젝션
    │  └─ 스케일 및 게이트 모듈레이션
    └─ Conditioning (시간步 + 조건 임베딩)
```

**구체적 구성:**[1]

| 항목 | 값 |
|------|-----|
| 총 매개변수 | 6.15B |
| 계층 수 | 30 |
| 숨겨진 차원 | 3840 |
| 어텐션 헤드 | 32 |
| FFN 중간 차원 | 10240 |

**텍스트 인코더**: 경량 Qwen3-4B (이중언어 능력)[1]
**이미지 토큰화**: Flux VAE (높은 재구성 품질)[1]
**편집 작업용**: SigLIP 2 (추상적 시각 의미론)[1]

#### 3.2 훈련 효율성 최적화[1]

- **분산 훈련**: 고정 VAE/텍스트 인코더에는 DP, DiT에는 FSDP2 (그래디언트 체크포인팅 포함)
- **배치 구성**: 시퀀스 길이 기반 배치 그룹화로 패딩 최소화
- **동적 배치 크기**: 긴 시퀀스는 작은 배치, 짧은 시퀀스는 큰 배치

#### 3.3 이미지 캡셔닝: Z-Captioner

5가지 캡션 유형 생성으로 다양한 사용자 맥락 대응:[1]

1. **태그 캡션**: 간결한 메타데이터 태그
2. **짧은 캡션**: 완전하고 포괄적인 설명
3. **긴 캡션**: OCR 결과와 세계 지식 포함한 상세 설명
4. **비모달 사용자 프롬프트**: 불완전한 프롬프트로 실제 사용 시나리오 모사
5. **차이 캡션**: 이미지 편집을 위한 3단계 CoT 프로세스
   - Step 1: 원본 및 대상 이미지 상세 캡셔닝
   - Step 2: 비교 분석 (시각 및 텍스트 관점)
   - Step 3: 편집 지시 합성

***

### 4. 성능 향상 및 평가

#### 4.1 정량적 평가

**① 텍스트 렌더링 (CVTG-2K 벤치마크):**[1]

| 모델 | 평균 단어 정확도 | 순위 |
|------|-----------------|------|
| Z-Image | 0.8671 | **1위** |
| Z-Image-Turbo | 0.8585 | 2위 |
| GPT-Image-1 | 0.8569 | 3위 |
| Qwen-Image | 0.8288 | 4위 |

**② 영문/중문 긴 텍스트 렌더링 (LongText-Bench):**[1]

| 벤치마크 | Z-Image | Z-Image-Turbo | 순위 |
|--------|---------|----------------|------|
| LongText-Bench-EN | 0.935 | 0.917 | 2-3위 |
| LongText-Bench-ZH | **0.936** | 0.926 | **2위** |

**③ 세밀한 지시 준수 (OneIG 벤치마크):**[1]

- **영문 전체 점수**: 0.546 (최고)
- **영문 텍스트 점수**: **0.987** (최고 - 거의 완벽)
- **중문 텍스트 점수**: **0.988** (최고)
- **중문 전체 점수**: 0.535 (2위)

**④ 객체 생성 (GenEval):**[1]

Z-Image: 0.84 (2위, Qwen-Image 0.87과 동등 수준)

#### 4.2 인간 선호도 평가

**Alibaba AI Arena (독립적 벤치마크):**[1]

| 순위 | 모델 | 회사 | Elo 점수 | 승률 |
|------|------|------|---------|------|
| 1 | Imagen 4 Ultra | Google | 1048 | 48% |
| 2 | Gemini-2.5-flash | Google | 1046 | 47% |
| 3 | Seedream 4.0 | ByteDance | 1039 | 46% |
| **4** | **Z-Image-Turbo** | **Alibaba** | **1025** | **45%** |
| 5 | Seedream 3.0 | ByteDance | 1012 | 41% |
| 6 | Qwen-Image | Alibaba | 1008 | 41% |

**결론**: **오픈소스 모델 중 1위, 전체 4위** 달성[1]

**FLUX 2 dev와의 비교 (222개 샘플, 3명 평가자):**[1]

| 지표 | Z-Image |
|------|---------|
| G Rate (좋은 생성) | 46.4% |
| S Rate (같은 수준) | 41.0% |
| B Rate (나쁜 생성) | 12.6% |
| **G+S Rate** | **87.4%** |

참고: Z-Image는 FLUX 2 dev의 **1/5 매개변수(6B vs 32B)**로 이 결과 달성[1]

#### 4.3 정성적 평가[1]

- **포토리얼리즘**: 피부 질감, 의류 디테일에서 우수
- **복잡한 포즈 렌더링**: 인물의 미묘한 표정 및 자세 표현 능력
- **이중언어 텍스트**: 중영문 혼합 테스트에서 높은 정확도
- **회복력 있는 일반화**: 다양한 프롬프트 스타일에 일관된 성능

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 일반화 성능을 제한하는 요소[1]

Z-Image 논문은 **명시적 일반화 한계**를 언급하지 않으나, 아키텍처 및 훈련 설계로부터 유추 가능한 강점이 있습니다:

**강점:**

1. **다중 데이터 소스 활용**: 세계 지식 위상 그래프로 개념적 다양성 보장
2. **능동적 큐레이션**: 장꼬리 개념 식별으로 분포 편향 완화[1]
3. **다단계 캡션**: 5가지 캡션 유형으로 다양한 입력 스타일 대응
4. **모델 병합**: 여러 SFT 변형의 선형 보간으로 기능 간 균형 달성[1]

#### 5.2 제안되는 일반화 향상 방법

**① 이미지 편집 전이 (Z-Image-Edit):**[1]

임의 해상도 훈련과 이미지-이미지 결합 학습이 편집 작업으로의 자연스러운 전이 가능

**② 프롬프트 강화기 (PE) 모듈:**[1]

CoT(Chain-of-Thought) 기반 추론으로 복잡한 사용자 프롬프트 이해:

$$\text{PE}(\text{prompt}) = \text{VLM}(\text{prompt}, \text{reasoning chain})$$

예시: "위도 30° 9' 36" N, 경도 120° 7' 12" E의 사진"
- **PE 없음**: 좌표 텍스트 렌더링만 수행
- **PE 포함**: 해당 위치(서호)를 추론하고 관련 장면 생성[1]

**③ RLHF 기반 정렬:**[1]

**Stage 1: Direct Preference Optimization (DPO)**
- 객관적 차원(텍스트 렌더링, 객체 개수)에서 선호도 쌍 학습
- VLM 기반 자동 주석 후 인간 검증으로 확장성 확보

**Stage 2: Group Relative Policy Optimization (GRPO)**
- 복합 이점 함수로 여러 차원 동시 최적화:

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}[\log \sigma(\beta(r_{\text{chosen}} - r_{\text{rejected}} - m))]$$

여기서 $r$은 다차원 보상 모델의 합계

#### 5.4 최근 연구의 일반화 트렌드

2020년 이후 관련 최신 연구를 종합하면:[2][3][4][5]

**① 효율적 아키텍처 트렌드:**

- **PixArt-Σ**: 0.6B 매개변수로 4K 이미지 생성, 토큰 압축 어텐션 도입[2]
- **HiDream-I1**: 17B 매개변수, 동적 MoE(Mixture-of-Experts) 기반 듬성 DiT, 3가지 변형 제공[5]
- **SANA**: 0.6B 매개변수, 선형 어텐션, 32배 압축 오토인코더, **1초 이내 1024×1024 생성**[4]

**② Flow Matching 전환:**

- **Diff2Flow**: 확산 모델에서 Flow Matching으로 효율적 전이[6]
- **FlowTurbo**: 8~16 단계 추론으로 FLUX 수준 품질 달성[7]
- **ShortCutting Flow Matching**: 하루 미만의 A100 시간으로 3단계 FLUX 증류[8]

**③ 교차모달 정렬 개선:**

- **OneReward**: 단일 보상 모델로 여러 편집 작업(채우기, 확장, 제거, 텍스트) 통합 처리[9]
- **CUSA/FMFA**: 교차모달 soft-label 정렬로 fine-grained 일치도 향상[10][11]

**④ 픽셀 공간 생성:**

- **PixelDiT**: 오토인코더 제거, 픽셀 공간에서 직접 DiT 훈련, ImageNet 256×256에서 FID 1.61[12]

***

### 6. 한계 (Limitations)

논문에서 명시된 한계:[1]

1. **6B 매개변수 제약**: 세계 지식과 복잡한 추론에 제한 (PE 모듈로 부분 보완)
2. **데이터 의존성**: 고품질 데이터 인프라 구축의 높은 초기 비용
3. **편집 데이터 희소성**: 이미지-이미지 쌍이 텍스트-이미지 데이터 대비 훨씬 적음
4. **실시간 배포**: H800 GPU 필수 (소비자 GPU는 Z-Image-Turbo 8단계로 가능하지만 추가 최적화 필요)

***

### 7. 향후 연구에 미치는 영향 및 고려사항

#### 7.1 학술 및 산업 영향

**① 효율성 패러다임 전환:**

기존의 "더 크면 더 좋다" 명제를 도전하며, **체계적 최적화가 무분별한 확대보다 효과적**임을 입증[1]

이는 후속 연구에 다음을 권장:
- 아키텍처 혁신에 집중 (S3-DiT의 단일 스트림 설계)
- 데이터 품질 및 큐레이션 인프라 투자
- 훈련 알고리즘 세밀화

**② 오픈소스 생태계 활성화:**

Z-Image의 공개 모델/코드/데모 제공으로, 소규모 연구팀도 고성능 모델 구축 가능한 기반 제공[1]

후속 연구 기회:
- 모델 미세조정 프레임워크 개발
- 도메인 특화 변형 (의료, 예술 등)
- 경량 변형 (3B, 1.5B 등) 탐색

#### 7.2 핵심 고려사항

**① 일반화 능력 연구:**

- **도메인 외 평가**: 학습 분포와 다른 데이터셋에서의 성능 측정 필요
  - 의료 이미지 생성
  - 과학 시각화
  - 취약/긴꼬리 개념 처리

- **언어 다양성**: 현재 중영문 중심, 다언어(한국어, 일본어, 인도 언어 등) 성능 평가 필수

**② 교차모달 정렬 강화:**

최신 연구(CUSA, FMFA)에서 제시하는 fine-grained 정렬 기법을 S3-DiT에 통합하여 text-image 일치도 향상[11][10]

**③ Flow Matching 통합:**

Diff2Flow와 유사한 파이프라인으로 Z-Image를 Flow Matching 프레임워크로 전환하여:[6]
- 추론 속도 추가 가속
- 더 직선적인 생성 궤적
- 기존 diffusion 이점 보존

**④ 멀티모달 확장:**

- **비디오 생성**: 시간 차원 어텐션 추가로 4D 생성으로 확장
- **오디오-비주얼**: AV-DiT 방식의 경량 어댑터 추가[13]
- **3D 생성**: Direct3D의 triplane 기반 latent 활용[3]

**⑤ RLHF 고도화:**

- **다차원 보상**: 현재 3차원(지시 추종, AIGC 감지, 미적 품질)에서 더 세분화
- **온라인 학습**: GRPO의 실시간 인간 피드백 루프 구현
- **보상 해킹 방지**: Adjoint Matching의 memoryless 노이즈 스케줄 응용[14]

**⑥ 데이터 효율성 극한:**

최근 Flow Matching의 이론적 진전(WGF 관점)과 Gaussian Mixture Flow Matching 같은 다모달 속도 필드 모델링을 적용하여 더욱 적은 데이터로 고성능 달성[15][16]

**⑦ 배포 및 실용성:**

- **엣지 배포**: 모바일/임베디드 환경용 2B-3B 변형 개발
- **개인정보 보호**: 온디바이스 추론으로 데이터 개인정보 보호
- **해석성**: Attention 시각화 및 token 기여도 분석으로 의사결정 투명성 향상

***

### 8. 결론

**Z-Image는 이미지 생성 모델의 개발 방식을 근본적으로 재정의합니다.** 단순한 대규모화가 아닌 **데이터 인프라, 아키텍처 설계, 훈련 전략의 정교한 조화**로 6B 매개변수에서 80B 모델 수준의 성능을 달성했습니다.[1]

특히 **효율적 데이터 큐레이션(4개 모듈 구성)**, **S3-DiT의 단일 스트림 설계**, **프로그레시브 훈련 커리큘럼**, **Decoupled DMD/DMDR**은 후속 연구의 벤치마크가 될 것으로 예상됩니다.[1]

모델의 일반화 성능은:
- **강점**: 다양한 데이터 소스, 능동적 큐레이션, 다단계 캡션, 모델 병합으로 우수
- **개선 여지**: Flow Matching 통합, fine-grained 교차모달 정렬, 다언어/도메인 외 평가 필요

최신 2020년 이후 트렌드(PixArt-Σ, HiDream-I1, SANA, PixelDiT, Diff2Flow 등)와 통합할 경우, **1B 이하 매개변수로도 고품질 생성이 가능한 미래**를 열 수 있을 것으로 판단됩니다.[12][4][5][2][6]

***

### 참고 문헌 (선택)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/152a7f67-76cf-4127-ae93-40d1dcf1da10/Z_Image_Report.pdf)
[2](https://arxiv.org/abs/2403.04692)
[3](https://arxiv.org/abs/2405.14832)
[4](https://openreview.net/forum?id=N8Oj1XhtYZ)
[5](https://arxiv.org/html/2505.22705v1)
[6](https://ieeexplore.ieee.org/document/11092660/)
[7](https://arxiv.org/html/2409.18128)
[8](https://arxiv.org/abs/2510.17858)
[9](https://openreview.net/forum?id=osnAy1yTHu)
[10](https://aclanthology.org/2023.eacl-main.250/)
[11](https://arxiv.org/abs/2509.13754)
[12](https://arxiv.org/html/2511.20645v1)
[13](https://arxiv.org/abs/2406.07686)
[14](https://arxiv.org/abs/2409.08861)
[15](https://arxiv.org/abs/2509.00336)
[16](https://arxiv.org/html/2504.05304v1)
[17](https://arxiv.org/abs/2412.03859)
[18](https://arxiv.org/abs/2408.12236)
[19](https://ieeexplore.ieee.org/document/11092615/)
[20](https://ieeexplore.ieee.org/document/11094418/)
[21](https://dl.acm.org/doi/10.1145/3687980)
[22](https://arxiv.org/abs/2411.03286)
[23](https://arxiv.org/abs/2405.17405)
[24](http://arxiv.org/pdf/2312.04557.pdf)
[25](https://arxiv.org/html/2411.04168v3)
[26](https://arxiv.org/abs/2410.13925v1)
[27](http://arxiv.org/pdf/2112.10752.pdf)
[28](http://arxiv.org/pdf/2405.14854.pdf)
[29](http://arxiv.org/pdf/2407.01425.pdf)
[30](https://arxiv.org/html/2405.14430)
[31](https://arxiv.org/html/2503.06132v1)
[32](https://hiringnet.com/image-generation-state-of-the-art-open-source-ai-models-in-2025)
[33](https://arxiv.org/abs/2505.22705)
[34](https://arxiv.org/html/2408.06741v1)
[35](https://www.youtube.com/watch?v=kO7_HMbMfwA)
[36](https://proceedings.neurips.cc/paper_files/paper/2023/file/5aca18e0192b2c1300479e5b700c76a9-Paper-Conference.pdf)
[37](https://github.com/NVlabs/DiffiT)
[38](https://z-image.ai)
[39](https://www.sciencedirect.com/science/article/abs/pii/S1046202323001329)
[40](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffit/)
[41](https://arxiv.org/abs/2411.07625)
[42](https://arxiv.org/abs/2507.14575)
[43](https://arxiv.org/abs/2506.02070)
[44](https://www.semanticscholar.org/paper/c278423a5b7422a679ab26320f02ceb2311f28d0)
[45](https://www.semanticscholar.org/paper/aa080bc1ecb7e88f96476e2334d139449912a3ac)
[46](https://ieeexplore.ieee.org/document/11177205/)
[47](http://arxiv.org/pdf/2302.00482.pdf)
[48](http://arxiv.org/pdf/2303.08797v3.pdf)
[49](https://arxiv.org/html/2402.14017v1)
[50](https://arxiv.org/pdf/2307.03672.pdf)
[51](https://arxiv.org/pdf/2310.05297.pdf)
[52](https://arxiv.org/pdf/2311.13443.pdf)
[53](https://github.com/CompVis/diff2flow)
[54](https://ai-scholar.tech/en/articles/alignment/image_reward_for_text_to_image_generation)
[55](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Schusterbauer_Diff2Flow_Training_Flow_Matching_Models_via_Diffusion_Model_Alignment_CVPR_2025_paper.pdf)
[56](https://www.ijcai.org/proceedings/2024/0088.pdf)
[57](https://diffusion.kaist.ac.kr)
[58](https://intuitionlabs.ai/articles/reinforcement-learning-human-feedback)
[59](https://arxiv.org/abs/2403.05261)
[60](https://dmqa.korea.ac.kr/activity/seminar/486)
[61](https://diffusion.csail.mit.edu)
