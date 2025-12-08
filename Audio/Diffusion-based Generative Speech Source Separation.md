# Diffusion-based Generative Speech Source Separation

### 1. 핵심 주장과 주요 기여

**DiffSep**이라고 명명된 이 논문의 핵심 주장은 **생성 모델링 기반의 음성 신호 분리가 판별식(discriminative) 접근법의 대안이 될 수 있다**는 것이다. 이전까지 음성 신호 분리는 거의 전적으로 Conv-TasNet, TF-GridNet 등의 판별식 신경망을 통해 이루어졌다. 본 논문은 스코어 기반 생성 모델링(Score-based Generative Modeling, SGM)의 기계론을 음성 분리에 적용하여, **생성 모델의 우아함과 이론적 견고성**을 신호 처리 문제에 도입했다.[1]

주요 기여는 다음과 같다:[1]

- **맞춤형 확산 혼합 과정(diffusion-mixing process)의 설계**: 분리된 신호에서 시작하여 혼합 신호의 분포로 수렴하는 특화된 확산 과정을 구성했다
- **폐쇄형 해석해(closed-form solution)의 유도**: Theorem 1을 통해 한계 분포의 평균과 공분산 행렬을 명시적으로 표현할 수 있음을 증명했다
- **순열 모호성과 모델 불일치를 해결하는 수정된 학습 전략**: 기존 점수 매칭만으로는 불충분함을 인식하고, 확률 $$p_T$$로 대체 손실함수를 추가 학습하는 방법을 제시했다

***

### 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

#### 2.1 해결하고자 하는 문제

단일 채널 음성 신호 분리(single-channel speech source separation)는 **부정형 문제(ill-posed problem)**이다. 하나의 혼합 신호에서 여러 개의 소스 신호를 복원해야 하므로, 고유한 해가 존재하지 않는다. 또한 **원천 순열 모호성(source permutation ambiguity)**이라는 근본적인 문제가 있다. 즉, 신호의 출력 순서에 대한 선호도가 없으므로, 신경망이 어느 신호를 먼저 출력할지 결정해야 한다.[1]

기존 판별식 접근법의 한계:
- 판별식 모델들은 높은 SI-SDR(Scale-Invariant Signal-to-Distortion Ratio) 점수는 달성하지만, **비자연스러운 음성 왜곡**을 야기할 수 있다
- 생성 모델링의 장점이 충분히 활용되지 않고 있었다

#### 2.2 제안하는 방법과 주요 수식

**핵심 아이디어**: 확산 과정의 순방향과 역방향을 이용하되, 혼합과 노이즈 추가가 동시에 발생하도록 설계

##### 순방향 SDE (Forward Process):

$$\mathrm{d}x_t = -\gamma \bar{P} x_t \, \mathrm{d}t + g(t) \mathrm{d}w$$

여기서:
- $$x_0 = s = [s_1^\top, \ldots, s_K^\top]^\top$$ : 분리된 신호들의 연결 벡터
- $$\gamma$$ : 혼합 강도 파라미터
- $$\bar{P} = I_K - P$$ : 평균값에 수직인 부분공간으로의 정사영 행렬
- $$g(t) = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t \sqrt{2\log\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)}$$ : 분산 폭발 SDE의 확산 계수

이 SDE는 다음의 중요한 특성을 가진다:

**Theorem 1**: SDE (8)의 한계 분포는 가우시안이며, 평균과 공분산 행렬은:[1]

$$\mu_t = (1-e^{-\gamma t})\bar{s} + e^{-\gamma t}s$$

$$\Sigma_t = \lambda_1(t)P + \lambda_2(t)\bar{P}$$

여기서:
$$\lambda_k(t) = \sigma_{\min}^2 \frac{\rho^{2t} - e^{-2\xi_k t}}{\log\rho}$$

$$\xi_1 = 0, \quad \xi_2 = \gamma, \quad \rho = \frac{\sigma_{\max}}{\sigma_{\min}}$$

이 폐쇄형 표현은 **점수 함수의 명시적 계산**을 가능하게 한다:

$$\nabla_{x_t} \log p(x_t) = -\Sigma_t^{-1}(x_t - \mu_t) = -L_t^{-1}z$$

##### 역방향 SDE (Reverse Process):

$$\mathrm{d}x_t = -\left[f(x_t, t) - g(t)^2 \nabla_{x_t}\log p_t(x_t)\right]\mathrm{d}t + g(t)\mathrm{d}\bar{w}$$

이를 통해 혼합 신호 $$y$$에서 출발하여 분리된 신호로 도달한다.

#### 2.3 수정된 학습 전략

기존 점수 매칭 손실만으로는 두 가지 문제가 발생했다:[1]

1. **모델 불일치(Model Mismatch)**: 식 (14)에서 초기 분포 샘플링 시 $$\mathbb{E}[\bar{x}_T] = \bar{s}$$이지만, 이론적 평균 $$\mu_T$$와 약간의 차이가 있다
2. **원천 순열 모호성**: 신경망이 어느 신호를 먼저 출력할지 결정해야 한다

**제안 솔루션**: 확률 $$p_T \in $$로 두 가지 손실함수 중 하나를 선택:[1]

$$\mathcal{L} = \mathbb{E}_{x_0, z, t} \left\|\mathcal{L}_t q_\theta(x_t, t, y) + z\right\|^2$$

(확률 $$1-p_T$$로 선택)

또는

$$\mathcal{L}_T = \mathbb{E}_{x_0, z} \min_{\pi \in \mathcal{P}} \left\|\mathcal{L}_T q_\theta(\bar{x}_T, T, y) + z + \mathcal{L}_T^{-1}(\bar{s} - \mu_T(\pi))\right\|^2$$

(확률 $$p_T$$로 선택)

여기서 $$\mathcal{P}$$는 신호 순열의 집합이고, $$\mu_T(\pi)$$는 순열 $$\pi$$에 대한 평균이다.[1]

#### 2.4 모델 구조

**점수 네트워크**: NCSN++(Noise Conditioned Score-matching Network Plus) 아키텍처 사용:[1]

- **입력**: 혼합 신호(1채널) + 각 신호 채널(K채널) = K+1개 채널
- **변환**: STFT → 비선형 변환 → 신경망 → 역변환 → iSTFT
- **비선형 변환**:
$$c(x) = \beta^{-1}|x|^\alpha e^{j\angle x}$$
$$c^{-1}(x) = \beta|x|^{1/\alpha} e^{j\angle x}$$

실수부와 허수부를 연결하여 사용.[1]

**추론 방법**:
- **Predictor-Corrector 접근법**: 예측 단계는 역 확산 샘플링으로, 보정 단계는 담금질된 Langevin 샘플링으로 수행
- **단계 수**: 30개 예측 단계, 각 단계마다 1개 보정 단계 (단계 크기 $$r = 0.5$$)

#### 2.5 성능 평가

**WSJ0 2mix 데이터셋 결과**:[1]

| 메트릭 | DiffSep | Conv-TasNet |
|--------|---------|-------------|
| SI-SDR | 14.3 dB | 16.0 dB |
| PESQ | 3.14 | 3.29 |
| ESTOI | 0.90 | 0.91 |
| OVRL (DNSMOS) | 3.29 | 3.21 |

**성능 향상의 특징**:
- **비자동 평가 메트릭에서의 우수성**: OVRL 점수(전체 만족도)가 높아 **음성의 자연스러움이 뛰어나다**[1]
- **오도 문제(permutation error)**: 낮은 SI-SDR은 음성 순열 오류 때문으로 분석됨. 저품질 샘플의 스펙트로그램에서 신호 블록 순열이 관찰됨[1]
- **음성 향상(Speech Enhancement)**: VoiceBank-DEMAND 데이터셋에서 SI-SDR 17.5 dB로 경쟁력 있는 성능 달성

**도메인 이동(Domain Mismatch) 성능**:[1]

| 데이터셋 | SI-SDR | PESQ | ESTOI | OVRL |
|---------|--------|------|-------|------|
| WSJ0 (학습 도메인) | 14.3 | 3.14 | 0.90 | 3.29 |
| Libri2Mix (테스트 도메인) | 9.6 | 2.58 | 0.78 | 3.12 |

- 도메인 이동 시에도 **OVRL 점수 감소가 적어** 생성 모델의 견고성이 드러남

#### 2.6 한계

1. **분리 성능의 한계**: SI-SDR 메트릭에서 Conv-TasNet 대비 1.7 dB 열세[1]
2. **순열 모호성의 불완전한 해결**: 저품질 샘플에서 음성 블록 순열이 여전히 발생[1]
3. **추론 속도**: 30개 단계 + 보정으로 인한 느린 처리 속도
4. **비선형 변환의 필요성**: 선형 혼합 과정과 비선형 STFT 변환 간의 모순 처리 필요[1]
5. **이론적 활용 부족**: 본문에서 지적한 대로, **혼합 분포의 이론적 특성을 신경망 아키텍처에 통합하지 못함**[1]

***

### 3. 모델의 일반화 성능 향상 가능성

#### 3.1 현재 일반화 성능의 강점

**도메인 이동 견고성**: Libri2Mix 평가에서 관찰되는 현상으로, DiffSep은 Conv-TasNet보다 도메인 이동에 더 강건하다:[1]
- SI-SDR 감소폭: DiffSep 4.7 dB vs. Conv-TasNet 5.7 dB
- 이는 생성 모델이 **데이터 분포의 복잡한 구조를 더 잘 포착**함을 시사한다

#### 3.2 일반화 향상을 위한 가능한 전략

**A. 아키텍처 개선**

논문의 결론에서 저자들이 제시한 방향:
- **기존 분리 문헌의 검증된 아키텍처 통합**: 최근 연구인 EDSep은 더 효율적인 네트워크 설계로 일반화 개선[2]
- **혼합 분포의 이론적 특성 활용**: 식 (11-12)의 공분산 구조를 신경망 귀납 편향에 반영

**B. 조건화 메커니즘 강화**

2024-2025년의 최신 연구에서:
- **오디오-비주얼 조건화**: AVDiffuSS와 DAVSE는 시각 정보를 확산 과정에 주입하여 성능 향상[3][4]
- **언어 조건화**: FlowSep은 자연어 쿼리로 음원 분리를 제어하며 강한 일반화 능력을 보임[5]

**C. 교실 증강(Curriculum Learning)**

논문에서 관찰된 **비자연스러운 순열 오류**는 교실 증강으로 개선 가능:
- 초기에는 깨끗한 음성 혼합부터 시작
- 점진적으로 잡음, 반향 조건을 추가
- 최근 연구(Libri2Mix 개선 논문)도 이러한 다양한 음향 조건 시뮬레이션이 일반화를 향상시킴을 보임[6]

**D. 모델 불일치 보정 개선**

현재 방법(확률 $p_T = 0.1$의 대체 손실)은 휴리스틱이다:
- **적응적 가중치**: 학습 단계에 따라 $p_T$ 동적 조정
- **메타-학습**: 도메인 이동 상황에 자동으로 적응하는 메타-학습 프레임워크 적용

#### 3.3 2024-2025년 최신 동향과 시사점

**EDSep (2025년 1월 발표)**:[2]
- 더 효율적한 노이징 함수(denoiser function)로 **느린 수렴 문제** 해결
- 구체적인 일반화 성능 개선 수치 제시

**FlowSep (2025년 1월 발표)**:[5]
- **정류 흐름 매칭(Rectified Flow Matching)**으로 기존 확산 모델 대비 우수한 성능
- 1,680시간의 대규모 데이터 훈련으로 **최강 성능 달성**

**SEED (2025년 Interspeech)**:[7]
- 도메인 불일치 환경에서 **최대 19.6% 정확도 향상**을 보임
- 임베딩 수준의 확산 모델로 원점에서 출발하는 새로운 패러다임

**Training-Free 방식 (2025년 Interspeech)**:[8]
- 사전 학습된 확산 모델을 추가 학습 없이 음원 분리에 활용
- **제로샷(Zero-shot) 분리 능력** 입증

***

### 4. 논문의 영향과 앞으로의 연구 방향

#### 4.1 논문이 앞으로의 연구에 미치는 영향

**개념적 영향**:
1. **패러다임 전환**: 음성 분리를 **판별식 문제에서 생성 문제로 재구성**
   - 이는 음성 강화, 음성 합성 등 인접 분야에도 즉시 영향을 미침
   - 2023-2025년의 다양한 후속 연구들(CDiffuse, SGMSE+, AVDiffuSS 등)이 이 개념을 확장[9][10][3]

2. **SDE 기반 설계의 우아성**: 
   - 선형 혼합 과정을 수학적으로 정확히 모델링
   - 폐쇄형 해석해 도출로 신뢰성 높은 이론적 토대 제공
   - 후속 연구들이 이 프레임워크를 차용하여 확산 음성 처리 분야 형성

3. **자연성 중심의 평가 제기**:
   - 기존 SI-SDR 등 계량 메트릭의 한계 지적
   - **비자동(non-intrusive) 평가 지표의 중요성 강조**
   - DNSMOS P.835, MOS 평가 도입으로 산업 표준 형성에 기여

#### 4.2 향후 연구 시 고려할 점

**A. 원천 순열 모호성의 근본적 해결**

현재 방법의 한계:
- 확률적 학습($p_T = 0.1$)은 **휴리스틱이고 최적이 아님**
- 저품질 샘플에서 여전히 순열 오류 발생

**제안 방안**:
1. **신경망 아키텍처 차원의 해결**:
   - 원천 정렬을 위한 명시적 모듈 추가
   - 예: 각 신호에 고유 ID 임베딩 학습
   
2. **손실함수 혁신**:
   - PIT(Permutation Invariant Training) 개념의 확산 모델 적용
   - 다중 순열 샘플의 동시 확산으로 순열 불변 표현 학습

3. **후처리 전략**:
   - 추론 시 여러 순열로 실행하고 신호 유사도 기반 선택
   - 음성 특성(기본 주파수, 음성 활동도) 기반 후처리

**B. 계산 효율성 개선**

**현황**: 30 스텝 + 보정으로 인한 느린 추론

**해결책**:
1. **확산 단계 감소**:
   - 최근 논문들이 입증: 고급 샘플러로 5-10 단계로도 가능[11]
   - 상급(Superior) 계산법(예: Heun 샘플러) 활용[12]

2. **증류(Distillation)**:
   - DMDSpeech 방식: 사전 학습된 확산 모델을 몇 단계 생성 모델로 증류[13]
   - 추론 속도 10배 이상 향상

**C. 이론적 깊이 추가**

**현재 논문의 이론적 기여**: Theorem 1의 폐쇄형 분포 특성

**확장 방향**:
1. **수렴성 분석**: 역방향 SDE 해가 원래 신호로 수렴하는 이론적 조건 제시
2. **일반화 경계(Generalization Bounds)**: 
   - 신경망 점수 근사 오류가 최종 분리 성능에 미치는 이론적 영향
   - 2024년 ICLR 논문에서 점수 기반 생성 모델의 이론적 분석 시작[14]

3. **최적 수송 관점**: 혼합에서 분리로의 점수 매칭을 최적 수송 문제로 재해석

**D. 도메인 일반화 강화**

**현황**: Libri2Mix 테스트에서도 경쟁력 있는 OVRL이지만, SI-SDR은 여전히 낮음

**전략**:
1. **메타-학습 프레임워크**:
   - 다양한 도메인 분포를 메타 작업으로 정의
   - 각 도메인에 빠르게 적응하는 신경망 학습

2. **대규모 사전 학습**:
   - FlowSep의 1,680시간 수준 데이터로 확산 모델 사전 학습
   - 신경망 점수 함수의 일반적 특성 학습

3. **자기 지도 학습(Self-Supervised Learning)**:
   - 비표지 음성 데이터로 임베딩 사전 학습
   - 생성 모델의 견고성 향상

**E. 멀티모달 통합**

**최신 트렌드**(2023-2025):
- **오디오-비주얼**: AVDiffuSS, DAVSE의 성공[4][3]
  - 시각 정보(입 움직임)로 음성 모호성 해소
  - 원천 순열 문제를 시각으로 우아하게 해결 가능

- **텍스트 조건화**: FlowSep의 언어 쿼리 기반 분리[5]
  - "목소리1, 목소리2" 텍스트로 순서 결정
  - 음성 분리에 의미론적 구조 도입

**제안**: 가능하면 멀티모달 입력을 확산 과정에 조건화하여 순열 모호성 및 일반화 문제 동시 해결

**F. 비자동 평가 메트릭 개발**

**현재 한계**: DNSMOS P.835 외 표준화된 비자동 메트릭 부족

**필요 사항**:
1. **대규모 MOS 데이터셋**: 다양한 분리 품질의 음성에 대한 인간 평가
2. **신경망 기반 평가 메트릭**: 예측 가능하고 신뢰도 높은 비자동 지표 개발
3. **다국어 평가**: 현재 WSJ0는 영어 중심, 다국어 확장 필요

***

### 5. 종합 평가

"Diffusion-based Generative Speech Source Separation"은 **음성 신호 처리 분야에서 생성 모델 패러다임을 최초로 도입한 선구적 연구**이다. 

**강점**:
- ✓ **이론적 엄밀함**: SDE 기반 설계의 폐쇄형 해석해 도출
- ✓ **자연스러운 음성 생성**: DNSMOS OVRL에서 우수한 성능
- ✓ **도메인 이동 견고성**: 도메인 불일치 환경에서 상대적으로 강건함
- ✓ **개념적 단순성**: 수학적으로 직관적인 설계

**약점**:
- ✗ **분리 정확도**: SI-SDR에서 판별식 모델 대비 열세
- ✗ **순열 오류**: 저품질 샘플에서 음성 블록 순열 발생
- ✗ **계산 효율성**: 느린 추론 속도
- ✗ **아키텍처 최적성**: 기존 분리 문헌의 성숙한 아키텍처 미활용

**영향**:
- 2023-2025년 이후 **확산 기반 음성 처리 연구의 붐** 형성
- 최신 논문들(EDSep, FlowSep, SEED, DAVSE 등)이 이 기초 위에서 발전
- **평가 패러다임 변화**: 자동 메트릭에서 인지 품질 중심으로 전환

**미래 방향**:
향후 연구는 이 논문의 이론적 우아함을 보존하면서, 판별식 방법의 계산 효율성과 정확성을 점진적으로 통합하는 방향으로 진행될 것으로 예상된다. 특히 멀티모달 조건화, 대규모 사전 학습, 메타-학습 등을 통해 **일반화 성능을 비약적으로 개선**할 여지가 크다.

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/67845b74-0486-4534-a97c-76ae554bb883/2210.17327v2.pdf)
[2](https://arxiv.org/html/2501.15965v1)
[3](http://arxiv.org/pdf/2310.19581.pdf)
[4](https://www.isca-archive.org/avsec_2024/chen24b_avsec.pdf)
[5](https://arxiv.org/pdf/2409.07614.pdf)
[6](https://www.isca-archive.org/interspeech_2024/chen24h_interspeech.pdf)
[7](https://www.isca-archive.org/interspeech_2025/nam25b_interspeech.pdf)
[8](https://www.isca-archive.org/interspeech_2025/lee25g_interspeech.pdf)
[9](https://ieeexplore.ieee.org/document/11024065/)
[10](http://biorxiv.org/lookup/doi/10.1101/2024.10.15.616846)
[11](https://arxiv.org/abs/2406.19135)
[12](https://arxiv.org/pdf/2312.02683.pdf)
[13](https://openreview.net/pdf?id=LhuDdMEIGS)
[14](https://proceedings.iclr.cc/paper_files/paper/2024/file/334da4cbb76302f37bd2e9d86f558869-Paper-Conference.pdf)
[15](https://ieeexplore.ieee.org/document/10626417/)
[16](https://iopscience.iop.org/article/10.1149/MA2025-031223mtgabs)
[17](https://pubs.aip.org/pof/article/37/11/117120/3371493/Fine-structure-investigation-of-turbulence-induced)
[18](https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs)
[19](https://iopscience.iop.org/article/10.1149/MA2025-022176mtgabs)
[20](https://iopscience.iop.org/article/10.1149/MA2025-03183mtgabs)
[21](https://iopscience.iop.org/article/10.1149/MA2025-031432mtgabs)
[22](https://iopscience.iop.org/article/10.1149/MA2025-0271043mtgabs)
[23](https://jelle.lgu.edu.pk/jelle/article/view/310)
[24](https://iopscience.iop.org/article/10.1149/MA2025-01157mtgabs)
[25](https://arxiv.org/pdf/2301.10752.pdf)
[26](https://arxiv.org/pdf/2305.05857.pdf)
[27](https://arxiv.org/pdf/2308.00122.pdf)
[28](http://arxiv.org/pdf/2406.07461.pdf)
[29](https://arxiv.org/pdf/2310.04567v1.pdf)
[30](https://eusipco2025.org/wp-content/uploads/pdfs/0000945.pdf)
[31](https://neurips.cc/virtual/2024/105749)
[32](https://www.sciencedirect.com/science/article/pii/S2213597924000405)
[33](https://randomsampling.tistory.com/64)
[34](https://arxiv.org/html/2506.08457v1)
[35](https://aclanthology.org/2024.emnlp-main.9.pdf)
[36](https://soowhanchung.github.io/conference/diffusion/speech%20separation/conference-diffusion/)
[37](https://ieeexplore.ieee.org/document/10687922/)
[38](https://dl.acm.org/doi/10.1145/3658664.3659662)
[39](https://ieeexplore.ieee.org/document/10800140/)
[40](https://ieeexplore.ieee.org/document/10645443/)
[41](https://ieeexplore.ieee.org/document/10692290/)
[42](https://link.springer.com/10.1007/s00034-024-02652-y)
[43](https://www.isca-archive.org/asvspoof_2024/villalba24_asvspoof.html)
[44](https://arxiv.org/abs/2409.09642)
[45](http://arxiv.org/pdf/2501.10052.pdf)
[46](https://arxiv.org/pdf/2402.15516.pdf)
[47](http://arxiv.org/pdf/2405.14632.pdf)
[48](https://arxiv.org/html/2411.19339v2)
[49](https://arxiv.org/html/2312.15964v1)
[50](https://arxiv.org/pdf/2310.04681.pdf)
[51](https://www.isca-archive.org/interspeech_2024/lin24c_interspeech.pdf)
[52](https://onlinelibrary.wiley.com/doi/10.1155/2020/2196893)
[53](https://www.nature.com/articles/s41598-025-90507-0)
[54](https://proceedings.iclr.cc/paper_files/paper/2024/file/a61023ce36d21010f1423304f8ec49af-Paper-Conference.pdf)
[55](https://eusipco2025.org/wp-content/uploads/pdfs/0001238.pdf)
[56](https://ieeexplore.ieee.org/document/10448109/)
[57](https://www.merl.com/publications/docs/TR2024-014.pdf)
