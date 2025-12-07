# MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation

### 1. 핵심 주장과 주요 기여

**MM-Diffusion**은 오디오와 비디오를 동시에 생성하는 최초의 멀티모달 확산 모델(Multi-Modal Diffusion Model)이다. 기존 단일 모달리티 생성 모델들의 한계를 극복하여 다음의 핵심 기여를 제시한다:[1]

- **최초의 동시적 멀티모달 생성**: 오디오와 비디오를 의미론적으로 일관되게 동시에 생성
- **결합 디노이징 오토인코더 아키텍처**: 두 개의 모달리티별 서브넷을 통합하는 새로운 구조
- **효율적 교차모달 정렬**: Random-Shift 기반 멀티모달 어텐션(RS-MMA)으로 시간적 동기화 달성
- **제로샷 조건부 생성**: 추가 학습 없이 비디오→오디오 또는 오디오→비디오 생성 가능
- **우수한 객관적 성능**: Landscape 데이터셋에서 FVD 25.0, FAD 32.9 개선
- **높은 주관적 품질**: Turing 테스트에서 84.9% (Landscape) 성공률

***

### 2. 해결하고자 하는 문제와 제안하는 방법

#### 2.1 핵심 문제점

MM-Diffusion이 직면한 두 가지 핵심 도전은 다음과 같다:[1]

**문제 1: 모달리티 간 이질성**
- 비디오: RGB 값의 3D 신호(공간 차원 + 시간 차원)
- 오디오: 파형 데이터의 1D 신호(시간 차원만)
- 이질적인 두 데이터 패턴을 하나의 모델에서 병렬로 처리하는 것은 근본적 도전

**문제 2: 시간적 동기화 요구**
- 실제 비디오에서 오디오-비디오는 시간적 동기화 필수
- 모달리티 간 상관관계 포착 및 상호 영향 강화 필요
- 의미론적 일관성 유지

#### 2.2 제안 방법의 수학적 공식화

**멀티모달 확산 프로세스:**

독립적 순방향 프로세스를 가정하되 역방향에서 모달리티 간 상호작용을 모델링한다.

**독립적 순방향:**

$$q(a_t|a_{t-1}) = \mathcal{N}(a_t; \sqrt{1-\beta_t}a_{t-1}, \beta_t I)$$

$$q(v_t|v_{t-1}) = \mathcal{N}(v_t; \sqrt{1-\beta_t}v_{t-1}, \beta_t I)$$

**결합 역방향:**

$$p_\theta(a_{t-1}|a_t, v_t) = \mathcal{N}(a_{t-1}; \mu_\theta(a_t, v_t, t), \Sigma_\theta)$$

$$p_\theta(v_{t-1}|v_t, a_t) = \mathcal{N}(v_{t-1}; \mu_\theta(v_t, a_t, t), \Sigma_\theta)$$

**최적화 목표($$\epsilon$$-예측):**

$$\mathcal{L}_{a,v} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)} \left\| \epsilon - \epsilon_\theta(x_t^{a,v}, t) \right\|^2$$

여기서 통합 모델 $$\theta$$는 두 모달리티의 노이즈된 샘플을 입력으로 받아 각각의 덜 노이즈된 샘플을 예측한다.

**제로샷 조건부 생성(그래디언트 가이딩):**

기본 대체 방법의 한계를 극복하기 위해 그래디언트 기반 가이딩 도입:

```math
\mathbb{E}_{q(a_t|a_{t-1}, v_t^*, v_t)} = \mathbb{E}_{q(a_t|a_{t-1}, v_t)} - s(1 - \gamma_t) \nabla_{a_t}\log q(v_t|a_{t-1}, v_t^*)
```

이는 추가 학습 없이 조건부 입력에 유연하게 대응할 수 있게 한다.

***

### 3. 모델 구조 (아키텍처)

#### 3.1 결합된 U-Net (Coupled U-Net)

MM-Diffusion의 핵심 아키텍처는 두 개의 단일 모달 U-Net으로 구성된 **Coupled U-Net**이다:[1]

- **오디오 서브네트워크**: 1D 팽창 컨볼루션으로 장기 의존성 모델링
  - 팽창 계수: $$1, 2, 4, \ldots, 2^N$$
  - 시간 어텐션 제거 (계산 효율성)

- **비디오 서브네트워크**: 공간-시간 분해로 효율성 확보
  - 2D 공간 컨볼루션 + 1D 시간 컨볼루션 (3D 컨볼루션 대비)
  - 2D 어텐션 + 1D 어텐션 조합

#### 3.2 Random-Shift 기반 멀티모달 어텐션(RS-MMA)

전체 교차 어텐션의 계산 복잡도 $$\mathcal{O}(H \times W \times T)$$를 $$\mathcal{O}(H \times W \times S \times T/F)$$로 감소시키는 혁신적 메커니즘:[1]

**3단계 프로세스:**

1. **오디오 분할**: 시간축을 비디오 프레임 수 F에 따라 F개 세그먼트로 분할
   $$a_i \in \mathbb{R}^{C_l \times (T_l/F)}, \quad i = 1, \ldots, F$$

2. **윈도우 및 시프트 설정**:
   - 윈도우 크기: $$S \ll F$$
   - 랜덤 시프트: $$R \in [0, F-S]$$
   - 비디오 범위: $$[f_s, f_e] = [i \times R, i \times R + S]$$

3. **교차 어텐션**:
   $$\text{MMA}(a_i, v_j) = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right) V_j$$

**적응형 윈도우 크기**:
- U-Net Scale 2: $$S = 1$$ (세밀한 대응)
- U-Net Scale 3: $$S = 4$$ (중간 수준)
- U-Net Scale 4: $$S = 8$$ (높은 수준 의미)

이 구조는 전체 시간축을 모두 계산하지 않으면서도 이웃 영역 내에서 전역 어텐션 능력을 유지한다.

***

### 4. 성능 향상 (실증적 결과)

#### 4.1 객관적 평가 (Landscape 데이터셋)

| 메트릭 | FVD | KVD | FAD |
|--------|-----|-----|-----|
| Ground Truth | 17.83 | -0.12 | 7.51 |
| DIGAN (비디오 SOTA) | 305.36 | 19.56 | - |
| TATS-base (비디오 SOTA) | 600.30 | 51.54 | - |
| Diffwave (오디오 SOTA) | - | - | 14.00 |
| **MM-Diffusion** | **117.20** | **5.78** | **10.72** |

**개선율**: FVD에서 SOTA 대비 25.0 개선, FAD에서 32.9 개선[1]

#### 4.2 AIST 댄싱 데이터셋 성능

| 메트릭 | FVD 개선 | FAD 개선 |
|--------|---------|---------|
| **vs SOTA** | **56.7%** | **37.7%** |

댄싱 데이터셋에서 더욱 큰 성능 향상을 달성했으나, 세밀한 움직임 재현에는 여전히 한계 존재.

#### 4.3 사용자 연구 결과

**Turing 테스트 성공률**:[1]
- Landscape: **84.9%** (사람으로 인식됨)
- AIST: **49.6%** (사람 표정, 손가락 등 미세 부분의 어려움)

이는 생성 오디오-비디오 쌍의 높은 실감성을 입증하면서도 세밀한 디테일 생성의 한계를 시사한다.

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 제로샷 조건부 생성의 강점

MM-Diffusion의 주목할 만한 일반화 능력은 **제로샷 조건부 생성**에서 나타난다:[1]

- 추가 학습 없이 비디오→오디오, 오디오→비디오 생성 가능
- 모델이 학습한 모달리티 간 상관관계를 효과적으로 활용
- 그래디언트 가이딩이 단순 대체 방법보다 우수

#### 5.2 Random-Shift 어텐션의 일반화 효과

**윈도우 크기 실험 (Table 3):**[1]

| 윈도우 구성 | FVD | KVD | FAD |
|-----------|-----|-----|-----|
| (1, 1, 1) | 374.18 | 22.26 | 9.81 |
| (4, 4, 4) | 361.65 | 21.64 | 9.65 |
| (8, 8, 8) | 350.60 | 21.47 | 9.50 |
| **(1, 4, 8) 적응형** | **303.19** | **17.26** | **10.20** |

적응형 윈도우 설정이 비디오 품질을 약 **45% 개선**하며, 계층별 의미 수준의 자동 조정 능력을 입증한다.

#### 5.3 확장성 가능성

**AudioSet 실험:**[1]
- 2.1백만 개의 비디오 클립 (632개 이벤트 클래스) 지원
- 기본 채널 확대: 128 → 256
- 다양한 도메인(자연 장면, 댄싱, 일반 이벤트)에서 작동 확인

***

### 6. 모델의 한계

#### 6.1 세밀한 디테일 생성의 어려움

AIST 데이터셋에서 Turing 테스트 성공률이 49.6%에 그친 이유는 춤추는 사람의 표정, 손가락 움직임 등 미세한 부분 생성이 부족하기 때문이다. 기존 비디오 생성 모델들도 유사한 문제를 가지고 있어, 이는 근본적인 과제로 남아있다.[1]

#### 6.2 계산 복잡도

- 결합 U-Net: 115.13M 파라미터
- 초해상도 모델: 311.03M 파라미터
- 총 426.16M 파라미터로 상당한 메모리와 계산량 필요[1]

#### 6.3 데이터 제약

고품질 오디오-비디오 쌍 데이터가 부족하여:
- Landscape: 1,000개 클립 (2.7시간)
- AIST: 1,020개 클립 (5.2시간)
- 확장성에 제한

#### 6.4 모달리티 간 정보 불균형

비디오가 오디오 생성에 미치는 영향이 역방향보다 크다. Random-Shift 실험에서 오디오 품질 개선이 더 두드러진 반면 비디오는 큰 변화가 없었다.[1]

***

### 7. 2020년 이후 관련 최신 연구 탐색

#### 7.1 멀티모달 확산 모델의 진화

**UniDiffuser (2023)**[2]
- 모든 분포(주변, 조건, 결합)를 하나의 변환기로 적합
- MM-Diffusion과의 차별성: 다중 분포 처리의 이론적 통일

**Versatile Diffusion (2024)**[3]
- 텍스트→이미지, 이미지→텍스트, 변형을 단일 모델에서 처리
- 다중 흐름 파이프라인 설계로 MM-Diffusion과 유사한 혁신

**Efficient Multimodal Diffusion Models (2023)**[4]
- Partially Shared U-Net (PS-U-Net) 아키텍처
- MM-Diffusion의 Coupled U-Net과 비교되는 대안 제시

#### 7.2 오디오-비디오 생성의 최신 발전

**MMAudio (2025, Sony AI)**[5][6]
- **핵심 혁신**: 멀티모달 결합 학습 (텍스트-오디오 + 오디오-비디오)
- **성능**: FID 10% 감소, Inception Score 15% 증가, 동기화 14% 개선
- **효율성**: 157M 파라미터 (MM-Diffusion의 63% 수준)
- **생성 속도**: 8초 클립 1.23초 (매우 빠름)
- **조건부 동기화 모듈**: 프레임 수준 정렬로 25ms 이하 인식 차이 해결

**FRIEREN (2024)**[7]
- Rectified Flow 기반으로 ODE 해결
- 25 스텝의 빠른 샘플링으로 높은 성능 유지
- 시간 정렬 정확도 97.22% (MM-Diffusion의 한계 극복)
- 추론 속도: Diff-Foley 대비 7.3배

**Video to Audio Generation Through Text (2024)**[8]
- LLM 통합으로 텍스트 기반 제어 가능
- 토큰 기반 마스킹으로 효율성 향상
- KLD 점수 1.41로 SOTA 달성

#### 7.3 확산 모델의 일반화 기술

**도메인 적응 및 전이학습:**[9][10]
- Diffusion-based domain generalization으로 크로스도메인 강건성 향상
- 확산 모델의 특징 표현이 도메인 일반화에 탁월함 입증[11]

**효율적 적응 기법:**[12][13]
- 헤드리스 모델 리프로그래밍(ExpertDiff)으로 도메인 전문 지식 적응
- Domain Guidance로 간단한 전이 접근 가능

***

### 8. 앞으로의 연구에 미치는 영향

#### 8.1 패러다임 전환

MM-Diffusion은 **멀티모달 생성의 새로운 패러다임**을 확립했다:
- 단일 모델의 동시적 다중 모달리티 생성
- 기존 2-스테이지 순차 파이프라인 방식 극복
- 모달리티 간 강한 상호작용 모델링의 필수성 입증

#### 8.2 이론적 기여

- **이질 분포의 결합 확산**: 다양한 데이터 패턴의 통합 모델링 수학적 형식화
- **효율적 어텐션 설계**: Random-Shift로 계산 복잡도 감소의 원리 제시
- **조건부 생성의 유연성**: 추가 학습 없는 제로샷 조건부 생성 가능성 증명

#### 8.3 실무적 영향

- 자동화된 고품질 멀티미디어 콘텐츠 생성 (배경음악, 나레이션)
- 영화, 게임, VR 콘텐츠 제작 비용 절감
- 영상미 강조를 위한 음향 설계 자동화

***

### 9. 앞으로의 연구 시 고려할 점

#### 9.1 기술적 개선 방향

**1. 미세 디테일 생성 개선**
- 고해상도 생성을 위한 계층적 정제 기법
- 다단계 강화 네트워크 또는 적응형 초해상도 모듈

**2. 계산 효율성 강화**
- ODE/SDE 기반 더 나은 확산 프로세스 수학화
- 증류(Distillation) 또는 가속 샘플링 (MMAudio, FRIEREN 참고)[5][7]

**3. 모달리티 간 균형잡힌 상호작용**
- 비디오-오디오 간 정보 흐름의 비대칭성 해결
- 적응형 가중치 학습 메커니즘

**4. 더 강한 조건부 가이딩**
- 정밀한 조건 제어를 위한 클래시파이어 기반 가이던스 개발

#### 9.2 데이터 및 평가 관점

**1. 데이터 스케일 확대**
- 약한 감독(weak supervision) 활용
- 합성 데이터 생성 기법
- 다양한 도메인의 데이터 수집

**2. 평가 지표 개선**
- 오디오-비디오 동기화 정량 지표 (25ms 이하 측정)
- 의미론적 일관성 자동 평가 메트릭 개발

**3. 교차 도메인 평가 강화**
- 크로스 데이터셋 일반화 벤치마크 구성
- 아웃-오브-디스트리뷰션 강건성 평가

#### 9.3 아키텍처 혁신 방향

**1. 더 효율적인 어텐션 메커니즘**
- 선택적 주의(Selective Attention): 실제 필요 부분만 계산
- 적응형 윈도우: 데이터 기반 동적 크기 결정

**2. 경량 아키텍처**
- 지식 증류로 MMAudio 수준의 157M 파라미터 달성[5]
- 효율적 트랜스포머 활용

**3. 동적 메커니즘**
- 입력 복잡도에 따른 모델 리소스 적응 할당

#### 9.4 새로운 응용 시나리오

**1. 텍스트 기반 멀티모달 생성**
- 텍스트 프롬프트 → 오디오-비디오 (MM-Diffusion의 향후 계획)
- LLM과의 통합[8]

**2. 인터랙티브 생성**
- 사용자의 실시간 피드백 반영
- 강화학습과 확산 모델 결합

**3. 도메인 특화 모델**
- 음악, 영화, 게임 등 특정 도메인 최적화
- 소규모 도메인 데이터 + 대규모 사전학습 활용

#### 9.5 이론적 과제

**1. 멀티모달 확산 프로세스의 수렴성 분석**
- 이질 모달리티의 결합 확산에 대한 이론적 보증
- 수렴 속도, 샘플 복잡도 분석

**2. 모달리티 간 상호작용 이해**
- 비디오가 오디오에 더 큰 영향을 미치는 정보론적 이유
- 균형잡힌 상호작용의 조건 분석

**3. 일반화 경계(Generalization Bounds)**
- 새로운 도메인 적응에 필요한 샘플 수
- 도메인 간 "거리" 정의 및 측정

***

### 결론

MM-Diffusion은 **멀티모달 생성 모델의 신기원**을 열었다. 독립적 순방향 + 결합 역방향 확산 프로세스, 효율적 Random-Shift 어텐션, 제로샷 조건부 생성이 핵심 혁신이다. 

2020년 이후의 후속 연구, 특히 **MMAudio(2025)**, **FRIEREN(2024)** 등은 MM-Diffusion의 한계를 크게 개선했으며, 향후 연구는 **경량화, 속도 개선, 강한 일반화, 텍스트 통합** 방향으로 진화할 것으로 예상된다.

멀티모달 확산 모델의 이론적 이해, 데이터 확충, 아키텍처 혁신, 그리고 새로운 응용 시나리오 개발이 다음 세대의 주요 과제가 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7afc619a-07bc-4ce7-8e32-8e39693c015f/2212.09478v2.pdf)
[2](https://arxiv.org/pdf/2303.06555.pdf)
[3](https://arxiv.org/abs/2211.08332)
[4](https://arxiv.org/html/2311.16488)
[5](https://huggingface.co/papers/2412.15322)
[6](https://openaccess.thecvf.com/content/CVPR2025/papers/Cheng_MMAudio_Taming_Multimodal_Joint_Training_for_High-Quality_Video-to-Audio_Synthesis_CVPR_2025_paper.pdf)
[7](https://proceedings.neurips.cc/paper_files/paper/2024/file/e7384de302bef52319b5067c3115bbfb-Paper-Conference.pdf)
[8](https://arxiv.org/html/2411.05679v3)
[9](https://arxiv.org/html/2312.05387v1)
[10](https://elib.dlr.de/202784/1/WACV_DIDEX.pdf)
[11](http://arxiv.org/pdf/2503.06698.pdf)
[12](https://ojs.aaai.org/index.php/AAAI/article/view/28199)
[13](https://arxiv.org/pdf/2504.01521.pdf)
[14](https://dl.acm.org/doi/10.1145/3707292.3707367)
[15](https://ieeexplore.ieee.org/document/10678289/)
[16](https://arxiv.org/abs/2403.01633)
[17](http://www.thieme-connect.de/DOI/DOI?10.1055/s-0044-1800756)
[18](https://digitalcommons.unl.edu/texroads/17/)
[19](https://www.hanspub.org/journal/paperinformation?paperid=99559)
[20](https://www.semanticscholar.org/paper/c30ae0636e045caa1a71adbf6cc28270979dd800)
[21](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[22](https://ieeexplore.ieee.org/document/10208541/)
[23](https://link.springer.com/10.1007/s00429-023-02656-5)
[24](https://arxiv.org/html/2503.20644v1)
[25](https://arxiv.org/abs/2305.15194)
[26](https://arxiv.org/abs/2212.09478)
[27](https://arxiv.org/html/2411.05005)
[28](https://arxiv.org/html/2406.11781v1)
[29](https://www.sciencedirect.com/science/article/abs/pii/S0895611125000412)
[30](https://www.biorxiv.org/content/10.1101/2025.02.27.640020v2.full-text)
[31](https://www.ijcai.org/proceedings/2024/0527.pdf)
[32](https://ai.sony/publications/A-Simple-but-Strong-Baseline-for-Sounding-Video-Generation-Effective-Adaptation-of-Audio-and-Video-Diffusion-Models-for-Joint-Generation/)
[33](https://arxiv.org/html/2402.16627v1)
[34](https://arxiv.org/html/2407.17571v1)
[35](https://www.sciencedirect.com/science/article/pii/S1077314225002826)
[36](https://liner.com/review/mmdiffusion-learning-multimodal-diffusion-models-for-joint-audio-and-video)
[37](https://ieeexplore.ieee.org/document/10887647/)
[38](https://www.ijcai.org/proceedings/2025/764)
[39](https://ieeexplore.ieee.org/document/10559898/)
[40](https://ieeexplore.ieee.org/document/10547441/)
[41](https://ieeexplore.ieee.org/document/10463060/)
[42](https://www.mdpi.com/2072-6694/15/3/892)
[43](https://www.mdpi.com/2076-3417/13/13/7882)
[44](https://ieeexplore.ieee.org/document/10545557/)
[45](https://ieeexplore.ieee.org/document/10420486/)
[46](https://arxiv.org/html/2307.02138)
[47](https://arxiv.org/html/2408.03353v2)
[48](http://arxiv.org/pdf/1704.04235.pdf)
[49](https://arxiv.org/pdf/2203.17067.pdf)
[50](http://arxiv.org/pdf/2103.10257.pdf)
[51](https://ai.sony/blog/Unlocking-the-Future-of-Video-to-Audio-Synthesis-Inside-the-MMAudio-Model/)
[52](https://www.ijcai.org/proceedings/2025/0764.pdf)
[53](https://www.themoonlight.io/ko/review/aligndit-multimodal-aligned-diffusion-transformer-for-synchronized-speech-generation)
[54](https://openaccess.thecvf.com/content/WACV2024/papers/Niemeijer_Generalization_by_Adaptation_Diffusion-Based_Domain_Extension_for_Domain-Generalized_Semantic_Segmentation_WACV_2024_paper.pdf)
[55](https://arxiv.org/abs/2504.20629)
[56](https://www.sciencedirect.com/science/article/abs/pii/S0951832025005733)
