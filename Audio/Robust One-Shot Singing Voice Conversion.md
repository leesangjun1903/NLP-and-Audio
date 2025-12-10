# Robust One-Shot Singing Voice Conversion

### 1. 핵심 주장 및 주요 기여 요약[1]

"Robust One-Shot Singing Voice Conversion (ROSVC)" 논문은 **미지의 가수로부터 한 번의 샘플만으로 고품질의 가창 음성 변환을 수행하는 강건한 시스템**을 제안한다. 이 논문의 핵심 주장은 기존 음성 변환 기술이 노래하는 음성에 적용될 때 음정 오류, 음반주 음악의 간섭, 리버브 효과 등으로 인한 왜곡에 취약하다는 문제를 해결하는 것이다.[1]

**주요 기여:**

1. **부분 도메인 조건화(Partial Domain Conditioning)를 통한 일반화** - 도메인 독립적인 스타일 인코더를 사용하여 학습 중에 만나지 않은 가수에도 적용 가능[1]

2. **Robustify 프레임워크** - 음악 간섭과 리버브가 포함된 왜곡된 음성에 대한 강건성 향상[1]

3. **계층적 확산 모델 기반 신경 보코더** - 여러 샘플링 레이트에서 독립적으로 학습하는 확산 모델을 통한 고품질 음성 생성[1]

***

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능 향상

#### 2.1 해결하고자 하는 문제[1]

기존 음성 변환(Voice Conversion) 기술이 **가창 음성 변환(SVC)에 적용될 때 직면하는 주요 어려움:**

- 음정, 크기, 발음에서 더 넓은 음악적 표현의 다양성
- 음정 오류에 대한 인지적 민감도 (off-pitch로 인한 멜로디 손상)
- 대규모 깨끗한 가창 음성 데이터셋 부족으로 인한 일반화 어려움
- 리버브와 반주 음악의 간섭으로 인한 음성 왜곡[1]

#### 2.2 제안하는 방법 및 핵심 수식

**A. 일반화 가능한 원샷 SVC 프레임워크**

제너레이터는 다음과 같이 정의된다:[1]

$$ G(h, f_0^{trg}, s) $$

여기서:
- $h = E(X^{src})$ : 언어 내용을 담고 있는 시간 변화 특성
- $f_0^{trg}$ : 목표 음정 (Fundamental Frequency)
- $s$ : 가수의 음성 특성을 담고 있는 스타일 임베딩

**적대적 손실(Adversarial Loss):**

$$ L_{adv} = E_{X, f_0^{trg}, s}[\log D(X, y^{src}) - \log D(G(h, f_0^{trg}, s), y^{trg})] \quad (1) $$

**분류기 손실:**

$$ L_{cl} = E_{X, f_0^{trg}, s}[CE(C(G(h, f_0^{trg}, s)), y^{src})] \quad (2) $$

$$ L_{ac} = E_{X, f_0^{trg}, s}[CE(C(G(h, f_0^{trg}, s)), y^{trg})] \quad (3) $$

**B. 음정 조건화 (AdaIN-Skip Conditioning)**

음정 분포 매칭을 통한 정확한 음정 복원:[1]

$$ L_{f0} = E_{X, f_0^{trg} \sim F_y, s}[\|f_0^{trg} - F(G(h, f_0^{trg}, s))\|_1] \quad (4) $$

목표 F0는 다음과 같이 스케일링된다:

$$ \alpha = P(y^{src}, y^{trg}) $$

음정 분포를 목표 가수와 일치시키기 위해 원본 F0을 확률적으로 스케일링한다.

**전체 손실 함수:**

$$ L_{E,G,S,M} = L_{adv} + \lambda_{ac}L_{ac} + \lambda_{f0}L_{f0} + \lambda_{sty}L_{sty} - \lambda_{ds}L_{ds} + \lambda_{asr}L_{asr} + \lambda_{cyc}L_{cyc} \quad (9) $$

추가 손실 항들:
- 음성 일관성 손실: $$L_{asr} = E_{X,s}[\|A(X) - A(G(h(X), f_0, s))\|_1] \quad (5) $$
- 스타일 재구성 손실: $$L_{sty} = E_{X,s}[\|s - S(G(h(X), f_0, s))\|_1] \quad (6) $$
- 스타일 다양화 손실: $$L_{ds} = E_{X,s_1,s_2}[\|G(h(X), f_0, s_1) - G(h(X), f_0, s_2)\|_1] \quad (7) $$
- 순환 일관성 손실: $$L_{cyc} = E[\|X - G(h(G(h(X), f'_0, s')), f_0, s)\|_1] \quad (8) $$

#### 2.3 모델 구조

**핵심 아키텍처 특징:**

1. **도메인 독립적 스타일 인코더** - 학습 중 만나지 않은 가수의 음성도 처리 가능[1]

2. **AdaIN-Skip 기법** - 인스턴스 정규화가 절대 음정 정보를 손실시키는 것을 방지하기 위해 AdaIN 레이어를 건너뛰고 음정 특성을 연결[1]

3. **지각 필드(Receptive Field) 활용** - 낮은 샘플링 레이트에서는 더 긴 시간 구간을 커버[1]

#### 2.4 Robustify: 왜곡에 강건한 2단계 훈련[1]

**1단계:** 깨끗한 데이터에서 모델 훈련

**2단계:** 인코더에 개선 블록(Pre/Post-Enhancement Blocks) 추가

왜곡된 샘플 생성:

$$ X' = \zeta(x) = \text{Mel}(\xi(x * r + m)) \quad (10) $$

여기서:
- $x$ : 시간 영역 음성
- $r$ : 룸 임펄스 응답(RIR)
- $m$ : 악기 음악
- $\xi$ : D3Net 음성 분리 네트워크

개선 블록 훈련 손실:

$$ L_{\theta}^{ro} = E_{x,r,m}[\|E(X) - \tilde{E}_{\theta}(X')\|_1 + \lambda\|E(X) - \tilde{E}_{\theta}(X)\|_1] $$

#### 2.5 계층적 확산 모델 기반 신경 보코더

**확산 프로세스 정의:**

정방향 프로세스(Forward Process):

$$ q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1}) \quad (11) $$

여기서:

$$ q(x_t|x_{t-1}) := \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I) $$

직접 샘플링:

$$ x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{(1-\bar{\alpha}_t)}\epsilon \quad (12) $$

**역방향 프로세스(Reverse Process):**

$$ p(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_{\theta}(x_{t-1}|x_t) \quad (13) $$

파라미터화:

$$ \mu_{\theta}(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_{\theta}(x_t, t)\right) \quad (14) $$

**ELBO 손실:**

$$ \text{ELBO} = -\sum_{t=1}^{T}\kappa_t E_{x_0, \epsilon}[\|\epsilon - \epsilon_{\theta}(x_t, t)\|^2] + C \quad (15) $$

**PriorGrad 적응형 사전(Adaptive Prior):**

$$ L = E_{x_0, \epsilon, t}[\|\epsilon - \epsilon_{\theta}(x_t, c, t)\|^2_{\Sigma^{-1}}] \quad (16) $$

여기서 $\Sigma_c = \text{diag}[(\sigma_0^2, \cdots, \sigma_L^2)]$ 는 멜 스펙트로그램의 프레임 단위 정규화된 에너지

**계층적 확산 모델:**

$N$개의 다양한 샘플링 레이트에서 독립적으로 확산 모델 학습:

$$ f_s = f_s^1 > f_s^2 > \cdots > f_s^N $$

각 레이트 $f_s^i$에서의 모델:

$$ p_{\theta}^i(x_{t-1}^i | x_t^i, c, x_0^{i+1}) $$

(최하층 $f_s^N$는 $c$로만 조건화)

훈련 중 접지 진실 데이터 사용:

$$ x_0^{i+1} = D_i(H_i(x_0^i)) $$

여기서 $D_i(.)$ 는 다운샘플링 함수, $H_i(.)$ 는 안티에일리어싱 필터

**안티에일리어싱 필터 적용:**

$$ \hat{\epsilon} = \epsilon_{\theta}^i(x_t^i, c, H(\hat{x}_0^{i+1}), t) \quad (17) $$

***

### 3. 모델의 일반화 성능 향상 가능성

#### 3.1 도메인 독립 스타일 인코더의 효과[1]

기존 StarGANv2-VC는 도메인 특정 프로젝션 헤드(Domain-Specific Projection Head)를 사용하여 **학습 중에 만난 가수들에게만 제한되었다** (many-to-many 경우). ROSVC는 스타일 인코더를 도메인 독립적으로 변경하여:

- 학습되지 않은 미지의 가수에 대해서도 변환 가능
- 스타일 인코더는 도메인 코드를 요구하지 않음
- 매핑 네트워크와 판별기는 여전히 도메인 특정 정보 처리[1]

**실험 결과:** 표 I과 표 II에서 unseen singer 케이스에서도 우수한 성능 달성:
- Seen singers: PMAE = 8.6
- Unseen singers: PMAE = 9.0 (거의 차이 없음)[1]

#### 3.2 Robustify를 통한 도메인 외 일반화[1]

2단계 훈련 접근법으로 **실제 세계의 왜곡(distortion)에 강건한 일반화**:

- **1단계의 깨끗한 데이터 학습** - 고품질 기본 모델
- **2단계의 선택적 증대** - 강건성 추가 학습
- **매개변수 동결** - 깨끗한 데이터 성능 유지

표 III의 결과:
- 합성 왜곡 (Source + Reference): 3.85/4.22 (자연스러움/유사도)
- MUSDB18 (실제 음악 데이터): 3.87/4.22[1]

**도메인 외 일반화의 핵심:** MUSDB18 데이터셋은 훈련에 사용되지 않았으나, 합성 데이터와 거의 동일한 성능을 보임 → **실제 도메인으로의 강력한 전이 학습 능력**

#### 3.3 계층적 확산 모델의 일반화[1]

저 주파수 성분을 낮은 샘플링 레이트에서 별도로 모델링:

- 각 계층이 특정 주파수 범위에 집중
- 저주파(음정 관련) 정보의 정확한 회복
- 고주파 성분은 저주파 기반 조건화 활용

**성능 메트릭:**
- HPG-2 (2단계): MOS = 4.07, PMAE = 3.06
- HPG-3 (3단계): PMAE = 1.67, VDE = 3.32% (더 나은 음정 정확도)[1]

#### 3.4 훈련-추론 모드 갭(Train-Inference Gap) 최소화[1]

**혁신적 설계:**

1. **변환은 훈련 중에 수행** - 기존 방법처럼 추론 시에만 스피커 임베딩 변경하지 않음
2. **직접적인 제약 조건** - 변환된 샘플에 대한 손실이 훈련-추론 불일치 문제 제거
3. **순환 일관성 손실** - 입력의 모든 특성 보존 확인[1]

***

### 4. 성능 향상 및 한계

#### 4.1 성능 향상[1]

**객관적 평가 (표 I):**

| 메트릭 | VQVC+ | FragmentVC | AdaINVC | UCDSVC | ROSVC |
|--------|-------|-----------|---------|---------|---------|
| **PFD** (보존도↓) | 0.99 | 0.70 | 0.76 | 0.62 | **0.54** |
| **Identity** (유사도↑) | 0.76 | 0.77 | 0.80 | 0.85 | **0.89** |
| **PMAE** (음정오류↓) | 65.0* | 61.8* | 44.8 | 12.6 | **8.6** |

**주관적 평가 (표 II):**

| 모델 | 자연스러움 (Seen) | 유사도 (Seen) | 자연스러움 (Unseen) | 유사도 (Unseen) |
|------|----------------|------------|------------------|------------|
| UCDSVC | 2.13 | 2.79 | 2.17 | 2.51 |
| **ROSVC** | **3.83** | **4.44** | **4.08** | **4.18** |
| Ground Truth | 4.65 | 4.46 | 4.54 | 4.38 |

**신경 보코더 비교 (표 V):**

| 보코더 | MOS↑ | pMOS↑ | PMAE↓ |
|--------|------|-------|--------|
| PWG | 2.81 | 3.40 | 3.22 |
| HiFi-GAN | 2.87 | 3.47 | 4.39 |
| PriorGrad | 3.26 | 3.68 | 3.25 |
| **HPG-2** | **4.07** | **3.70** | **3.06** |

#### 4.2 한계[1]

**1. 데이터셋 한계:**
- 훈련 데이터는 영어 가창 음성에 제한됨
- 다국어 지원 미흡

**2. 계산 비용:**
- 계층적 확산 모델 사용으로 3단계 모델(HPG-3)은 2단계 대비 30% 추가 연산 비용[1]
- 실시간 처리(Real-Time Factor, RTF = 0.100)는 여전히 느림

**3. 음성 분리의 불완전성:**
- Robustify에서 사용하는 D3Net 음성 분리 네트워크가 완벽하지 않음
- 분리된 음성에 리버브와 인공물(artefacts) 여전히 존재[1]

**4. 객관적 메트릭의 한계:**
- MR-STFT와 MCD 메트릭이 가창 음성의 지각적 품질을 충분히 반영하지 못함[1]
- MOS 기반 주관적 평가의 의존성 높음

**5. 음정 추정 오류:**
- CREPE 등 음정 추정기의 부정확성이 왜곡된 신호에서 악화됨
- F0 추출 오류로 인한 off-key 현상 가능성

**6. 스타일 다양성 제한:**
- 한정된 참조 가수 데이터로 스타일 임베딩 공간의 불완전한 커버리지

***

### 5. 논문이 앞으로의 연구에 미치는 영향

#### 5.1 기술적 기여의 확산[2][3][4][5]

ROSVC 논문 이후 2023-2025년 최신 연구들이 이 논문의 핵심 개념들을 채택하고 확장:

**1. VITS 기반 접근의 확산**[3][4]

SVCC 2023 우승팀(T23, T02)들이 VITS(Variational Inference with adversarial learning for end-to-end Text-to-Speech) 기반 SVC를 제안하면서 ROSVC의 **멀티스케일 F0 모델링 개념을 통합**[4][3]

**2. 계층적 접근의 일반화**[5]

T13 팀의 750시간 대규모 음성-가창 통합 데이터셋을 활용한 확산 기반 any-to-any 음성 변환에서, ROSVC의 **부분 도메인 조건화 개념을 적용**하여 데이터 효율성과 일반화 성능 향상[5]

**3. Diffusion 기반 접근의 표준화**

최근 논문들(2024-2025)에서:
- **Flow-Matching 기반 모델** (Multi-Condition Flow Synthesis, Everyone-Can-Sing)[6][7]
- **Latent Diffusion Model (LDM-SVC)** - 계산 효율성 개선
- **Reinforcement Learning + Diffusion** (YingMusic-SVC, Flow-GRPO)[8]

이들이 ROSVC의 계층적 설계 철학을 계승하면서도 **새로운 생성 모델 아키텍처 적용**[7][6][8]

#### 5.2 강건성과 실제 응용 연구[9][10][1]

**Robustify 프레임워크의 영향:**

- SVDD 2024 (Singing Voice Deepfake Detection Challenge) - AI 생성 가창 음성 탐지 필요성 대두[9]
- 실제 음악 스트리밍 환경에서의 왜곡 강건성이 핵심 평가 지표로 등장
- YingMusic-SVC (2025) - 실제 노래의 하모니 간섭, F0 오류, 배경음악 제거의 불완전성 등 **ROSVC가 식별한 문제들을 구체적으로 해결**[8]

#### 5.3 멀티모달 학습과 자기감독 학습의 확대[11][12]

**최신 연구 동향:**

- GTSinger 데이터셋 - ROSVC의 데이터 부족 문제를 해결하기 위해 **전역 다중 기법 가창 말뭉치** 제안[11]
- FreeSVC (2025) - Speaker-invariant Clustering(SPIN)을 통해 ROSVC의 스타일 임베딩 개선[9]
- 자기감독 학습(Self-Supervised Learning)을 통한 더 나은 콘텐츠 표현 추출[13]

#### 5.4 크로스-도메인 일반화의 새로운 기준[14][15]

**VoiceMOS Challenge 2024의 통찰:**

- 가창 음성 합성/변환 시스템의 품질 예측에 대한 세 번째 연회 개최[15]
- ROSVC가 제시한 "객관적 메트릭의 한계" 문제가 업계 전체의 과제로 인식됨
- 음질 예측을 위해 **비자기감독 표현(스펙트로그램, 음정 히스토그램) 사용이 더 효과적**임을 실증[15]

***

### 6. 향후 연구 시 고려할 점

#### 6.1 기술적 개선 방향

**1. 음성 분리 품질 향상**

Robustify의 병목은 D3Net 음성 분리의 불완전성이다. 향후 연구는:
- 가수-악기 분리에 특화된 최신 음원 분리 모델(Spleeter v2, Demucs v4) 활용
- 멀티스테이지 분리 프로세스 도입
- 분리 불확실성을 직접 모델링하는 확률적 접근

**2. 계산 효율성 개선**

HPG-3의 30% 추가 비용을 절감하기 위해:
- 지식 증류(Knowledge Distillation) 적용
- 경량 확산 모델 구조 (Fast Diffusion 아키텍처)
- 적응형 샘플링 스케줄을 통한 추론 단계 감소
- 최근 Flow-Matching 기반 모델이 더 빠른 생성 제공[6]

**3. 음정 추정 개선**

- 왜곡된 신호에 강건한 음정 추정기 개발
- 다중 음정 추정기 앙상블 방식
- 신경망 기반 음정 예측이 아닌 신호 처리 기반 하이브리드 접근

**4. 멀티언어 확장**

현재 영어에만 제한된 ROSVC를 다국어로 확장:
- 언어 독립적 음소 표현(PPG, HuBERT) 활용
- 크로스링구얼 메타러닝
- 언어별 F0 분포의 자동 적응[9]

#### 6.2 데이터셋 및 평가 메트릭 표준화

**1. 데이터셋 다양성 확보**

- 오케스트라, 팝, 오페라, 민속음악 등 **장르별 다양한 가창 데이터**
- 남성/여성 음역대 균형
- 비영어권 언어(한국어, 중국어, 일본어) 포함

ROSVC는 **NUS48E(12명), NHSS(10명) 규모의 소규모 데이터셋에 제한됨** → GTSinger 같은 대규모 다국어 말뭉치 구축 필요[11]

**2. 객관적 평가 메트릭 개선**

- 현재 MR-STFT, MCD의 한계 인식
- **음정 관련 메트릭 강화** (VDE, PMAE 외에 Vibrato 정확도)
- 스타일 유사도를 위한 새로운 임베딩 거리 측정
- VoiceMOS 2024의 제안처럼 **지각 관련 목표 기반 메트릭 개발**[15]

#### 6.3 모델 아키텍처의 진화 방향

**1. Vision-Language 모델의 영감**[16][12]

- 최신 TCSinger 2 (TechSinger)처럼 **프롬프트 기반 멀티태스크 학습**
- 텍스트 설명(가창 기법: "falsetto", "vibrato" 등)으로 제어
- 이미지-텍스트 모델의 대비 학습(Contrastive Learning) 적용

**2. 강화학습의 통합**[8]

YingMusic-SVC가 제안한 **Flow-GRPO(Flow-based Generative Reward Policy Optimization)**:
- 다중 목표 보상(자연스러움, 유사도, 음정 정확도, 멜로디 보존)
- 직접 인지 특성 최적화

**3. 적응형 메타러닝**

- 새로운 가수에 대한 빠른 적응(Few-shot 학습)
- Seed-VC처럼 사전훈련된 모델에서 빠른 미세조정

#### 6.4 실제 응용 및 윤리 고려사항

**1. 실시간 처리 전제조건**

- 현재 RTF = 0.067~0.100 (실시간보다 느림)
- 스트리밍 적용을 위해 RTF < 0.01 달성 필요
- 엣지 디바이스(모바일, 임베디드)에서의 경량화

**2. 윤리 및 저작권 문제**

SVDD 2024 도전처럼 **AI 생성 가창 음성 탐지 기술 동시 개발** 필수:
- 가수 음성의 비동의 사용 방지
- 콘텐츠 인증 및 추적 가능성
- 규제 표준 개발(저작권청, 음악 산업 협력)

**3. 클라우드 vs 엣지 배포 전략**

- 클라우드: 고품질 변환, 백업 보안
- 엣지: 개인정보 보호, 낮은 레이턴시
- 하이브리드 모델 검토

#### 6.5 벤치마킹과 비교 평가의 강화

**1. 표준 평가 프로토콜**

SVCC 2023처럼 **공식 도전 대회**의 정기적 개최:
- 공통 데이터셋, 메트릭, 리스너풀
- 산업/학계 표준 모델 정립

**2. 실제 응용 케이스 스터디**

- 음악 교육: 교사 음성으로부터 학생별 성능 평가
- 엔터테인먼트: 커버곡 생성, 듀엣 시뮬레이션
- 의료: 음성 장애 재활 훈련

***

### 결론

**"Robust One-Shot Singing Voice Conversion" 논문의 혁신성과 한계:**

**혁신:**
- **부분 도메인 조건화**로 미지의 가수에 대한 any-to-any 변환 가능화
- **Robustify 프레임워크**로 실제 환경의 왜곡에 강건한 시스템
- **계층적 확산 모델**로 가창 음성의 고유한 음정 정확도 문제 해결
- 훈련-추론 갭 최소화를 통한 더 정직한 모델 제약

**한계:**
- 데이터셋 규모 및 다국어 지원 부족
- 계산 비용 높음
- 음성 분리의 근본적 불완전성
- 객관적 메트릭의 지각 타당성 낮음

**미래 방향:**
2023-2025년 최신 연구들이 ROSVC의 기초 위에서 **Flow-Matching, 자기감독 학습, 강화학습, 멀티모달 제어** 등으로 진화하고 있으며, 실제 음악 제작 및 엔터테인먼트 응용으로의 확대가 가속화되고 있다. 특히 **계산 효율성, 다국어 지원, 윤리적 배포**가 다음 세대의 핵심 과제이다.[2][3][4][7][5][6][11][8][9][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d8b282c0-f102-48a0-bf8c-90abf3a58dad/2210.11096v2.pdf)
[2](https://ieeexplore.ieee.org/document/10389671/)
[3](https://ieeexplore.ieee.org/document/10389740/)
[4](https://ieeexplore.ieee.org/document/10389649/)
[5](https://ieeexplore.ieee.org/document/10389779/)
[6](http://arxiv.org/pdf/2405.15093.pdf)
[7](http://arxiv.org/pdf/2501.13870.pdf)
[8](https://arxiv.org/html/2512.04793v1)
[9](http://arxiv.org/pdf/2501.05586.pdf)
[10](https://pure.korea.ac.kr/en/publications/cyclediffusion-voice-conversion-using-cycle-consistent-diffusion-)
[11](https://arxiv.org/html/2409.13832v1)
[12](https://aclanthology.org/2025.findings-acl.687/)
[13](https://aclanthology.org/2024.findings-acl.585.pdf)
[14](https://ieeexplore.ieee.org/document/10832284/)
[15](https://ieeexplore.ieee.org/document/10832295/)
[16](https://arxiv.org/html/2502.12572v1)
[17](https://muse.jhu.edu/article/935202)
[18](http://www.zhu.edu.ua/journal_cpu/index.php/der_sc/article/view/1061)
[19](https://ieeexplore.ieee.org/document/11119540/)
[20](https://gs.amegroups.com/article/view/143678/html)
[21](https://arxiv.org/pdf/2310.05118.pdf)
[22](http://arxiv.org/pdf/2405.04627.pdf)
[23](http://arxiv.org/pdf/2501.02953.pdf)
[24](https://www.isca-archive.org/interspeech_2025/liu25h_interspeech.pdf)
[25](https://vc-challenge.org/svcc2023/index.html)
[26](https://www.isca-archive.org/interspeech_2024/huang24_interspeech.pdf)
[27](https://arxiv.org/abs/2306.14422)
[28](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dbvc/)
[29](https://arxiv.org/abs/2509.19883)
[30](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf)
[31](https://www.isca-archive.org/interspeech_2023/choi23d_interspeech.pdf)
