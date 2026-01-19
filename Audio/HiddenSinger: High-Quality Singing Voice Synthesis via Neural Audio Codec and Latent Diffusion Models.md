
# HiddenSinger: High-Quality Singing Voice Synthesis via Neural Audio Codec and Latent Diffusion Models

## 요약 (Executive Summary)

**HiddenSinger**는 신경 오디오 코덱(Neural Audio Codec)과 잠재 확산 모델(Latent Diffusion Models)을 창의적으로 결합하여 고품질의 가창 음성 합성(SVS)을 실현한 혁신적 시스템입니다. 기존 SVS 시스템이 직면한 세 가지 핵심 문제—높은 계산 복잡도, 학습-추론 불일치(Training-Inference Mismatch), 데이터 수집의 어려움—를 체계적으로 해결합니다. 특히 **HiddenSinger-U**라는 비지도 학습 프레임워크를 통해 쌍을 이루지 않은 데이터만으로도 학습 가능하게 함으로써, 음성 합성 분야에 새로운 패러다임을 제시합니다.

***

## 1. 논문의 핵심 주장 및 주요 기여

### 핵심 주장 (Core Claims)

HiddenSinger가 제시하는 핵심 관점:

1. **저차원 잠재 공간의 효율성** - 고차원 오디오 직접 확산 대신, 신경 코덱을 통해 압축된 저차원 공간에서 안정적인 확산 모델 운영이 가능하다는 것을 입증

2. **데이터 기반 사전의 효과** - 표준 가우시안 노이즈 대신 실제 데이터 분포에 기반한 사전을 사용하면 학습-추론 불일치 문제를 완화할 수 있음

3. **비지도 학습의 가능성** - 자기 지도 음성 표현과 대조 학습을 활용하여 쌍 데이터 없이도 고품질 합성이 가능함

### 주요 기여 (Key Contributions)

| 기여 항목 | 설명 | 실제 성과 |
|---------|------|---------|
| **HiddenSinger** | 신경 코덱 + 잠재 확산 결합 | nMOS 3.80 (VISinger 대비 +9.5%) |
| **RVQ 정규화 활용** | 안정적 잠재 공간 구성 | 확산 모델 학습 안정화 |
| **HiddenSinger-U** | 비지도 학습 프레임워크 | 쌍 데이터 없이 nMOS 3.83 유지 |
| **데이터 기반 사전** | 실제 분포 기반 시작점 | 표준 가우시안 대비 4.4% 성능 향상 |

***

## 2. 해결하는 문제, 제안 방법, 모델 구조

### 2.1 해결하는 문제

#### 문제 1: 높은 계산 복잡도
- **현황**: 기존 SVS는 고차원 오디오(waveform) 또는 선형 스펙트로그램(1025개 빈)에서 작동
- **문제점**: 고차원 공간의 연산 비용이 기하급수적으로 증가
- **HiddenSinger의 해결책**: 신경 오디오 코덱으로 **128배 차원 축소** (예: 1025차원 → 8차원 코드)

#### 문제 2: 학습-추론 불일치 (Training-Inference Mismatch)
- **현황**: 음향 모델이 학습한 후진 분포(posterior)와 악보로부터 추정한 사전 분포(prior) 간 갭 존재
- **결과**: 부정확한 음정, 발음 오류 등
- **해결책**: 음악 악보에서 직접 데이터 기반 평균 $\hat{\mu}$ 예측 → $N(\hat{\mu}, I)$에서 시작

#### 문제 3: 데이터 수집의 어려움
- **현황**: 음악 악보와 오디오의 정렬된 쌍 필요 → 매우 비용이 많이 듦
- **HiddenSinger-U의 해결책**: 
  - 자기 지도 표현(wav2vec 2.0)으로 가사 정보 추출
  - 기본주파수(F0) 추출로 멜로디 정보 추출
  - 대조 학습으로 두 표현 공간 정렬

### 2.2 제안 방법의 수식

#### Forward SDE (정규화된 잠재에 대한 확산 과정)

$$dz_t' = \frac{1}{2}(\hat{\mu} - z_t')\beta_t dt + \sqrt{\beta_t}dW_t \quad \text{[Eq. 9]}$$

여기서:
- $z_t'$: t번째 타임스텝의 정규화된 잠재 벡터
- $\hat{\mu}$: 조건 표현으로부터 예측한 평균 (데이터 기반 사전)
- $\beta_t$: 미리 정의된 노이즈 스케줄
- $W_t$: 표준 브라운 운동

**해석**: 표준 가우시안이 아닌 $N(\hat{\mu}, I)$에서 시작하므로, 실제 데이터에 더 가까운 점에서 확산 과정이 시작됨

#### Reverse SDE (역확산 과정)

$$dz_t' = \left[\frac{1}{2}(\hat{\mu} - z_t') - s_\theta(z_t', \hat{\mu}, h_{cond}, t)\right]\beta_t dt + \sqrt{\beta_t}d\tilde{W}_t \quad \text{[Eq. 13]}$$

**점수 추정 네트워크 학습 손실**:

$$L_{diff} = \mathbb{E}_{z_0', z_t', t}\left[||s_\theta(z_t', \hat{\mu}, h_{cond}, t) - \nabla_{z_t'}\log p_t(z_t'|z_0')||_2^2\right] \quad \text{[Eq. 14]}$$

#### Residual Vector Quantization (RVQ) 손실

$$L_{emb} = \sum_{c=1}^{C} ||z_{0,c} - q_c(z_{0,c})||_2^2 \quad \text{[Eq. 3]}$$

**의미**: 연속 표현을 C개의 양자화기를 통해 단계적으로 이산화하며, 각 단계에서의 잔여(residual)를 다음 양자화기가 처리

#### 음성-U 인코더의 대조 학습

```math
L_{cont*} = \sum_{t=1}^{T} \frac{e^{\cos(h_*^{(t)}, \tilde{h}_*^{(t)})/\tau_{cont}}}{\sum_{\xi[k \neq t]}e^{\cos(h_*^{(t)}, h_*^{(k)})/\tau_{cont}}} + \sum_{t=1}^{T} \frac{e^{\cos(\tilde{h}_*^{(t)}, h_*^{(t)})/\tau_{cont}}}{\sum_{\xi[k \neq t]}e^{\cos(\tilde{h}_*^{(t)}, \tilde{h}_*^{(k)})/\tau_{cont}}} \quad \text{[Eq. 16]}
```

**역할**: 쌍을 이룬 표현(음악 악보 기반)과 비지도 표현(오디오만 사용)을 같은 공간에 정렬

### 2.3 모델 구조

HiddenSinger의 전체 구조는 세 개의 독립적인 모듈로 구성:

```
┌─────────────────────────────────────────────────────────┐
│              1. Condition Encoder                       │
│  (음악 악보 → 조건 표현 h_cond + 평균 μ̂ 예측)          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│           2. Latent Generator (확산)                    │
│  (조건 기반 → 잠재 표현 ẑ₀ 생성)                         │
│  • Forward SDE: N(μ̂, I) → 노이즈                       │
│  • Reverse SDE: 점수 추정으로 역확산                     │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│    3. Audio Autoencoder (고정된 파라미터)              │
│  (잠재 표현 → 고충실도 음성 복원)                       │
│  • RVQ Blocks: 30개 양자화기                           │
│  • Decoder: HiFi-GAN V1 + MS-STFT 판별자              │
└─────────────────────────────────────────────────────────┘
```

**세 모듈의 독립적 학습**:
1. **Audio Autoencoder** (사전 학습): 고품질 음성 복원 능력 학습
2. **Condition Encoder + Latent Generator**: 음악 악보 조건에서 잠재 표현 생성

이러한 분리는 **계산 효율성**과 **안정성**을 모두 확보합니다.

***

## 3. 성능 향상 및 한계

### 3.1 성능 비교 결과

| 모델 | nMOS | Pitch Error | MAE |
|------|------|-----------|-----|
| VISinger | 3.47±0.09 | 44.441 | 0.439 |
| DiffSinger | 3.36±0.09 | 45.726 | 0.431 |
| **HiddenSinger** | **3.80±0.08** | **43.247** | 0.467 |
| **HiddenSinger-U** | **3.83±0.08** | **43.536** | 0.454 |

**성능 향상 분석**:
- **자연성(nMOS)**: +9.5% (가장 중요한 지표, 인간 청취 평가)
- **음정 정확도**: 43.247 (경쟁 모델 중 최고 수준)
- **객관적 메트릭(MAE)**: 약간 높음 (확산 모델의 확률적 특성 때문)

### 3.2 비지도 학습의 안정성

HiddenSinger-U는 비지도 데이터 비율이 증가해도 뛰어난 안정성을 보입니다:

| 비지도 비율 | nMOS | Pitch Error | 특성 |
|-----------|------|-----------|------|
| 0% (지도만) | 3.81±0.11 | 43.247 | 기준 |
| 10% | 3.68±0.11 | 43.536 | **안정적** |
| 50% | 3.70±0.13 | 49.084 | 여전히 음정 유지 |

**중요한 관찰**: 음성 유사도(sMOS)는 감소하지만, 자연성과 음정 정확도는 유지됨

### 3.3 주요 한계

#### 한계 1: 계산 효율성
- **문제**: 다단계 확산 필요 (T=1.5 시간 단계)
- **영향**: 추론 속도가 기존 방법보다 느림
- **제안된 해결책**: Consistency Models를 통한 증류 (논문에서 명시)

#### 한계 2: 확률적 변동성
- **특징**: 같은 입력으로도 매번 다른 음정 변동성 생성
- **긍정적 측면**: 자연스러운 음성(nMOS 높음)
- **부정적 측면**: 객관적 오류 지표가 높음(MAE, Periodicity)

```
실제 F0 (5회 추론 시각화):
┌──────────────────────────────────────┐
│ 1회: [음악 악보 음정] + 자연스러운 변동  │
│ 2회: [음악 악보 음정] + 다른 변동      │
│ 3회: [음악 악보 음정] + 또 다른 변동   │
│ ...                                  │
└──────────────────────────────────────┘
```

#### 한계 3: 스타일 적응 미흡
- **현황**: 새로운 창작적 스타일 적응 어려움
- **원인**: 모델이 음악 악보에 충실하도록 설계됨
- **미래 개선**: 제로샷 싱잉 스타일 전이 구현 필요

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 RVQ 정규화의 일반화 효과

**비교 실험** (동일한 확산 모델 사용):

| 정규화 방식 | nMOS | 특징 | 일반화 |
|----------|------|------|--------|
| RVQ-reg. | 3.86±0.09 | **안정적** | ★★★★★ |
| KL-reg. | 3.87±0.10 | 불연속성 | ★★★☆☆ |
| None | 3.86±0.09 | 학습 실패 | ★★☆☆☆ |

**원리**:
1. RVQ는 다층 코드북으로 계층적 정량화 제공
2. 확산 모델이 각 계층의 잠재 공간을 명확히 학습
3. 새로운 화자/곡에서도 일반화 성능 유지

### 4.2 데이터 기반 사전의 효과

**표준 가우시안 vs 데이터 기반 사전**:

```
표준 가우시안 사전 N(0, I):
XT ~ N(0, I) ─[많은 확산 단계]─> X0
            (데이터에서 멀 수 있음)

데이터 기반 사전 N(μ̂, I):
XT ~ N(μ̂, I) ─[적은 확산 단계]─> X0
             (데이터에 가까운 시작점)
```

**성능 향상**: nMOS 3.64 → 3.80 (**+4.4%**)

### 4.3 강화 조건 인코더의 중요성

제거 시 성능 저하 폭:

| 지표 | 전체 시스템 | 제거 후 | 저하율 |
|------|----------|--------|--------|
| nMOS | 3.86 | 3.24 | -16.1% |
| Pitch Error | 43.247 | 49.495 | -14.5% |
| Periodicity | 0.172 | 0.216 | -25.6% |

**역할**: 가사와 멜로디 정보를 음프레임 단위로 통합하여 조건 신호 강화

### 4.4 비지도 학습의 일반화 메커니즘

**3단계 프로세스**:

1. **자기 지도 표현 추출**
   - Wav2Vec 2.0: 음성 → 음소 정보 (화자 정보 제거)
   - F0 추출: 기본주파수 → 멜로디 정보 (양자화)

2. **표현 공간 정렬**
   - 대조 학습: 쌍 표현 vs 비지도 표현
   - 결과: 같은 음향 정보를 다른 방식으로 표현하는 두 모델이 동일 공간에서 작동

3. **새로운 화자에 대한 일반화**
   - 학습되지 않은 화자의 오디오 → 자기 지도 표현으로 자동 인코딩
   - 대조 학습으로 정렬된 공간에서 자동으로 올바른 의미로 해석

***

## 5. 최신 연구 비교 분석 (2020년 이후)

### 5.1 경쟁 모델 비교표

| 모델 | 발표연도 | 기본 구조 | nMOS | 특징 | 장점 | 단점 |
|------|--------|---------|------|------|------|------|
| **DiffSinger** | 2022 | Mel-Spec + 확산 | 3.36 | 얕은 확산 | 빠른 추론 | 자연성 낮음 |
| **VISinger** | 2022 | VAE 기반 E2E | 3.47 | 직접 파형 생성 | 안정성 | 객관적 메트릭 |
| **HiddenSinger** | 2023 | 신경코덱 + 확산 | 3.80 | 저차원 확산 | **최고의 자연성** | 추론 속도 |
| **NaturalSpeech 2** | 2023 | 신경코덱 + LM | 높음 | 44K시간 데이터 | 제로샷 | 대규모 데이터 필요 |
| **RDSinger** | 2024 | 참고기반 확산 | 높음 | 입력 참고 | 안정성 | 추가 입력 필요 |
| **TCSinger** | 2024 | 스타일 제어 | 높음 | 다중 스타일 | 표현성 제어 | 복잡한 구조 |
| **MakeSinger** | 2024 | 반지도 학습 | 높음 | TTS 데이터 활용 | 데이터 효율 | 새로운 기법 |

### 5.2 신경 오디오 코덱 기술 발전

```
2021: SoundStream (Google)
├─ 혁신: Residual Vector Quantization 도입
└─ 영향: 효율적인 다층 양자화 표준화

2022: Encodec (Meta)
├─ 혁신: MS-STFT 판별자로 고주파 음질 개선
└─ 영향: 더 나은 음성 충실도

2023: HiddenSinger (Korea University)
├─ 혁신: 신경 코덱을 확산 모델 입력으로 활용
└─ 영향: 신경 코덱의 새로운 활용 패러다임 제시
       → 후속 모델들이 동일 구조 채택

2024+: 확산
├─ RDSinger: 참고기반 확산
├─ SmoothSinger: 조건부 확산 기반 전체 오디오
├─ LCM-SVC: 일관성 증류로 가속화
└─ 예상: 신경 코덱 + 확산이 표준 구조로 정착
```

### 5.3 2024년 최신 모델들의 진화 방향

**1차 진화**: HiddenSinger의 기본 구조 채택
- **RDSinger (Oct 2024)**: 참고 멜-스펙트로그램 추가로 추론 안정성 개선
- **SmoothSinger (Dec 2024)**: 보코더 제거하고 전체 오디오 직접 생성

**2차 진화**: 표현성/제어성 강화
- **TCSinger (Sept 2024)**: 클러스터링 VQ로 스타일 정보 압축, 다중 스타일 제어
- **ExpressiveSinger (Oct 2024)**: 성능 제어 신호(timing, F0 curve, amplitude) 추가

**3차 진화**: 데이터 효율성
- **MakeSinger (Jun 2024)**: 반지도 학습으로 TTS 데이터도 활용
- **LDM-SVC (Jun 2024)**: 싱잉 음성 변환에도 적용

***

## 6. 논문이 미치는 영향과 향후 연구 방향

### 6.1 학술적 영향

#### (1) 신경 오디오 코덱의 재평가
- **이전**: 보코더로만 사용되는 보조적 기술
- **HiddenSinger 이후**: 생성 모델의 중추적 구성 요소
- **결과**: 2024년 이후 다수 모델이 신경 코덱 기반 구조 채택

#### (2) 잠재 공간 확산의 검증
- **이전**: 음성 분야에서 실패 사례 많음
- **HiddenSinger**: 저차원 잠재 공간에서의 성공 입증
- **후속**: 2024년 다수 SVS/TTS 모델이 동일 원칙 적용

#### (3) 비지도 학습의 새로운 패러다임
- **기존**: 자기 지도 표현의 직접 사용
- **혁신**: 대조 학습으로 쌍/비쌍 표현 공간 정렬
- **영향**: MakeSinger, VISinger2+ 등에서 비지도 학습 활용

### 6.2 현업 적용 전망

#### 가능한 응용 분야
1. **게임/엔터테인먼트**: 배경음악 생성, 캐릭터 음성 합성
2. **음악 제작**: 작곡가의 아이디어 빠른 표현
3. **언어 교육**: 다중언어 발음 연습 데이터 생성
4. **팟캐스트/오디오북**: 실시간 음성 생성 (개선 필요)

#### 현재 제약 사항
- **추론 속도**: T=1.5 타임스텝 필요 → 게임 등 실시간 처리 어려움
- **언어 제한**: 현재 한국어만 지원
- **스타일 제어**: 제한적 (음색 제어만 가능)

### 6.3 향후 연구 방향

#### 단기 (1-2년)
1. **추론 가속화**
   - Consistency Models 증류
   - **예상 효과**: 현재의 10배 이상 속도 향상

2. **스타일 제어 강화**
   - 감정, 가창 기법(vibrato) 조절
   - **참고 모델**: TCSinger의 다중 스타일 제어

3. **다중언어 확장**
   - 영어, 중국어, 일본어 등 지원
   - **과제**: 각 언어별 음성 특성 학습

#### 중기 (2-3년)
1. **제로샷 스타일 전이**
   - 단 3-5초의 참고 음성으로 스타일 적응
   - **목표**: TCSinger 수준의 표현성

2. **완전 end-to-end 아키텍처**
   - 오토인코더 파라미터도 함께 최적화
   - **효과**: 약간의 성능 향상 가능

3. **영상 기반 합성**
   - 입술 움직임을 조건으로 활용
   - **응용**: 가수 아바타, 더빙

#### 장기 (3년 이상)
1. **신경 오디오 코덱의 해석**
   - 학습된 코드의 의미 분석
   - **가능성**: 더 효율적인 코덱 설계

2. **멀티모달 생성 모델 통합**
   - 음악 생성 + SVS + 영상
   - **목표**: 완전 자동 음악 비디오 제작

3. **엣지 컴퓨팅 배포**
   - 개인 휴대폰에서 실행 가능한 경량 모델
   - **응용**: 음성 인터페이스, 창작 도구

### 6.4 향후 연구 시 고려 사항

#### 데이터 윤리
- **문제**: 유명 가수 음성 데이터 활용 시 저작권/개인정보 이슈
- **권장**: OpenCpop, OpenSinging 등 오픈소스 데이터셋 우선 사용
- **필수**: 명시적 동의 기반 데이터셋 구축

#### 모델 편향 (Bias)
- **문제**: 특정 언어/성별/연령 데이터 과잉 표현
- **결과**: 소수 음성에 대한 성능 저하
- **해결책**: 데이터 밸런싱, 다양한 평가 셋 확보

#### 객관적 메트릭과 주관적 평가의 불일치
- **현상**: nMOS는 높으나 특정 화자 모방 어려움
- **원인**: 확산 모델의 확률적 변동성
- **개선**: 온도 파라미터 조절, 화자 임베딩 강화 등 필요

***

## 7. 결론

### 종합 평가

HiddenSinger는 **신경 오디오 코덱과 잠재 확산 모델의 창의적 결합**을 통해 가창 음성 합성 분야에 세 가지 중요한 기여를 합니다:

| 측면 | 기여도 | 영향력 |
|------|--------|--------|
| **기술 혁신** | 높음 | 2024년 이후 표준 구조로 확산 |
| **성능 향상** | 높음 | nMOS 3.80 달성 (동급 최고) |
| **실용성 확대** | 중간 | 비지도 학습으로 데이터 효율성 개선 |
| **학술 영향** | 높음 | 신경 코덱 기반 음성 합성의 새로운 패러다임 |

### 강점 요약
✓ 획기적인 기술 결합으로 자연스러운 음성 생성  
✓ 비지도 학습으로 데이터 수집 비용 감소  
✓ 확산 모델의 안정적 운영으로 훈련 안정성 확보  
✓ 확장 가능한 아키텍처로 후속 연구 용이  

### 약점 및 개선 과제
✗ 추론 속도 개선 필수 (Consistency Models 적용)  
✗ 스타일 제어 능력 제한 (별도 스타일 인코더 필요)  
✗ 고주파 해상도 한계 (24kHz, 44.1kHz 필요)  
✗ 단일 언어 지원 (다중언어 확장 필요)  

### 최종 전망

HiddenSinger는 2024년 이후 **신경 코덱 기반 음성 합성의 표준 구조**로 자리 잡았으며, RDSinger, TCSinger, MakeSinger 등 다수의 후속 연구에 영감을 제공했습니다. 추론 속도 개선, 표현성 제어 강화, 다중언어 확장 등의 과제를 해결할 경우, **게임, 음악, 미디어 엔터테인먼트 분야에서 실용적 영향력**을 발휘할 것으로 예상됩니다.

특히 **비지도 학습 프레임워크(HiddenSinger-U)**는 저자원 언어나 새로운 도메인에의 빠른 적응을 가능하게 하여, 음성 합성 기술의 **민주화(democratization)**에 크게 기여할 것으로 기대됩니다.

<span style="display:none">[^1_1][^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_4][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_5][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_6][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_7][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_8][^1_80][^1_81][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2306.06814v1.pdf

[^1_2]: https://arxiv.org/abs/2410.21641

[^1_3]: https://arxiv.org/abs/2407.02049

[^1_4]: https://arxiv.org/abs/2406.05325

[^1_5]: https://ieeexplore.ieee.org/document/10800358/

[^1_6]: https://arxiv.org/abs/2406.05965

[^1_7]: https://aclanthology.org/2024.emnlp-main.117

[^1_8]: https://ojs.aaai.org/index.php/AAAI/article/view/34541

[^1_9]: https://ieeexplore.ieee.org/document/10626580/

[^1_10]: https://ieeexplore.ieee.org/document/10447981/

[^1_11]: https://dl.acm.org/doi/10.1145/3664647.3681642

[^1_12]: https://arxiv.org/pdf/2306.06814.pdf

[^1_13]: https://arxiv.org/pdf/2409.15977.pdf

[^1_14]: https://arxiv.org/pdf/2406.05965.pdf

[^1_15]: http://arxiv.org/pdf/2406.05325.pdf

[^1_16]: https://arxiv.org/abs/2304.09116

[^1_17]: https://arxiv.org/html/2503.01183v1

[^1_18]: http://arxiv.org/pdf/2410.21641.pdf

[^1_19]: https://arxiv.org/pdf/2311.08667.pdf

[^1_20]: https://pubmed.ncbi.nlm.nih.gov/39368276/

[^1_21]: https://arxiv.org/abs/2301.02111

[^1_22]: https://machinelearning.apple.com/research/controllable-music

[^1_23]: https://pure.korea.ac.kr/en/publications/hiddensinger-high-quality-singing-voice-synthesis-via-neural-audi/

[^1_24]: https://www.microsoft.com/en-us/research/wp-content/uploads/2024/12/SLT2024_CodecInvestigation.pdf

[^1_25]: https://www.ijcai.org/proceedings/2023/0648.pdf

[^1_26]: https://www.isca-archive.org/interspeech_2024/chen24e_interspeech.pdf

[^1_27]: https://www.emergentmind.com/topics/neural-audio-codecs-nacs

[^1_28]: https://proceedings.neurips.cc/paper_files/paper/2023/file/38b23e2328096520e9c889ae03e372c9-Paper-Conference.pdf

[^1_29]: https://aclanthology.org/2025.findings-acl.687.pdf

[^1_30]: https://kimjy99.github.io/논문리뷰/vall-e/

[^1_31]: https://kimjy99.github.io/논문리뷰/ernie-music/

[^1_32]: https://arxiv.org/html/2406.05298v1

[^1_33]: https://openaccess.thecvf.com/content/CVPR2024/papers/Chowdhury_MeLFusion_Synthesizing_Music_from_Image_and_Language_Cues_using_Diffusion_CVPR_2024_paper.pdf

[^1_34]: https://arxiv.org/pdf/2410.21641.pdf

[^1_35]: https://arxiv.org/abs/2501.06320

[^1_36]: https://arxiv.org/abs/2303.08385

[^1_37]: https://arxiv.org/html/2410.21641v1

[^1_38]: https://arxiv.org/abs/2310.14044

[^1_39]: https://arxiv.org/abs/2501.08238

[^1_40]: https://arxiv.org/abs/2302.02257

[^1_41]: https://arxiv.org/html/2511.20470v1

[^1_42]: https://arxiv.org/abs/2502.12759

[^1_43]: https://arxiv.org/abs/2305.09489

[^1_44]: https://arxiv.org/abs/2506.04492

[^1_45]: https://arxiv.org/html/2507.20128v1

[^1_46]: https://ojs.aaai.org/index.php/AAAI/article/view/21350

[^1_47]: https://www.semanticscholar.org/paper/b122a5a5aaf21f982f85d57c42c7c4cbc4af2b28

[^1_48]: https://ieeexplore.ieee.org/document/10509739/

[^1_49]: https://arxiv.org/pdf/2406.05692.pdf

[^1_50]: https://arxiv.org/html/2412.08918v2

[^1_51]: http://arxiv.org/pdf/2501.13870.pdf

[^1_52]: https://arxiv.org/pdf/2212.01546.pdf

[^1_53]: http://arxiv.org/pdf/2409.09988.pdf

[^1_54]: https://arxiv.org/pdf/2305.05401.pdf

[^1_55]: https://arxiv.org/pdf/2105.13871.pdf

[^1_56]: https://arxiv.org/abs/2105.02446v1

[^1_57]: https://www.isca-archive.org/interspeech_2023/zhang23e_interspeech.pdf

[^1_58]: https://arxiv.org/abs/2010.05646

[^1_59]: https://vocalsynth.fandom.com/wiki/DiffSinger

[^1_60]: https://www.semanticscholar.org/paper/VISinger:-Variational-Inference-with-Adversarial-Zhang-Cong/7353bcb7ef870c53a0318d3bf7d5b42c1d58b8d8

[^1_61]: https://www.dialora.ai/blog/hifi-gan-ai-audio-generation-guide

[^1_62]: http://github.com/MoonInTheRiver/DiffSinger

[^1_63]: https://arxiv.org/abs/2211.02903

[^1_64]: https://randomsampling.tistory.com/51

[^1_65]: https://kimjy99.github.io/논문리뷰/diffsinger/

[^1_66]: https://ieeexplore.ieee.org/document/9747664/

[^1_67]: https://github.com/jik876/hifi-gan

[^1_68]: https://arxiv.org/abs/2105.02446

[^1_69]: https://arxiv.org/pdf/2211.02903.pdf

[^1_70]: https://kakaoenterprise.github.io/papers/neurips2020-hifi-gan

[^1_71]: https://arxiv.org/html/2506.21478v1

[^1_72]: https://arxiv.org/pdf/2506.21478.pdf

[^1_73]: https://www.semanticscholar.org/paper/VISinger-2:-High-Fidelity-End-to-End-Singing-Voice-Zhang-Xue/1a0a78cf184a80aaab8150f8435493c379633b2d

[^1_74]: https://arxiv.org/pdf/2010.05646.pdf

[^1_75]: https://www.semanticscholar.org/paper/DiffSinger:-Singing-Voice-Synthesis-via-Shallow-Liu-Li/fe92f3f7ceec008118842d42b578dc25bcba63f9

[^1_76]: https://arxiv.org/html/2502.12759v1

[^1_77]: https://arxiv.org/html/2406.08761v2

[^1_78]: https://arxiv.org/abs/2210.15533

[^1_79]: https://arxiv.org/html/2507.04966v1

[^1_80]: https://arxiv.org/abs/2110.08813

[^1_81]: https://arxiv.org/html/2512.20211v1
