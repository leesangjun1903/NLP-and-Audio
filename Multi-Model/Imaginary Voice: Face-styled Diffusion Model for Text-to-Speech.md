# Imaginary Voice: Face-styled Diffusion Model for Text-to-Speech

### 1. 핵심 주장과 주요 기여 요약

**FACE-TTS(Imaginary Voice)**는 얼굴 이미지를 활용하여 제로샷(zero-shot) 음성 합성을 수행하는 획기적인 연구입니다. 이 논문의 핵심 주장은 다음과 같습니다.[1]

이 연구는 **"사람이 얼굴을 보면 그 사람의 목소리를 상상할 수 있다"는 자연스러운 인간의 인식 능력에 영감**을 받아, 얼굴 이미지의 시각적 특성으로부터 음성 스타일과 음성을 학습하는 최초의 엔드-투-엔드 확산 기반 TTS 모델을 제안합니다.[1]

주요 기여는 다음과 같습니다:

1. **최초의 얼굴 조건화 TTS 모델**: 얼굴 이미지를 TTS 모델 학습 단계에서 직접 조건화로 사용한 최초 사례[1]

2. **교차-모달 생체인식 활용**: 얼굴과 음성 간의 강한 상관관계를 활용하여 화자 정체성 유지[1]

3. **화자 특성 결합 손실(Speaker Feature Binding Loss) 도입**: 합성 음성과 실제 음성 간의 화자 임베딩 공간에서의 유사성을 강제[1]

4. **제로샷 적응 가능**: 등록(enrollment) 과정이 없어도 미출현(unseen) 화자의 음성 생성 가능[1]

***

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하고자 하는 문제

FACE-TTS는 다음의 두 가지 핵심 문제를 해결하고자 합니다.[1]

**문제 1: 다중 화자 TTS로 확장의 어려움**
- 모든 사람은 서로 다른 말하기 스타일, 음정, 액센트를 가짐
- TTS 모델이 다양한 화자 스타일을 학습하기 위해서는 매우 많은 학습 데이터 필요

**문제 2: 미출현 화자의 음성 생성 어려움**
- 기존 다중 화자 TTS도 미출현 화자의 음성을 생성하려면 상당한 양의 목표 화자 음성 샘플 필요
- 깨끗한 등록(enrollment) 발화 수집이 어려움

기존 접근법의 한계:[1]
- Face2Speech 관련 선행연구(2020, 2022)는 얼굴 이미지를 학습 단계가 아닌 **추론 단계에서만** 사용
- 얼굴 정체성 인코더와 음성 인코더를 TTS 모델과 독립적으로 학습

#### 2.2 제안하는 방법

**핵심 아이디어**: 얼굴 이미지가 더 쉽게 수집 가능하므로, 음성 대신 얼굴 이미지를 등록 데이터로 사용

**교차-모달 생체인식 기반 접근**[1]
선행연구에 따르면 얼굴과 음성 간에 강한 상관관계가 있으므로, 이를 활용하여:
- 얼굴 임베딩과 음성 임베딩을 공유 임베딩 공간으로 정렬
- 얼굴 이미지로부터 화자 정체성 특성 추출
- 다중 화자 TTS에서 말하기 스타일 조건화

**화자 특성 결합 손실(Speaker Feature Binding Loss)**[1]

모델이 짧은 길이의 음성 세그먼트로도 얼굴-음성 연관성을 학습하기 위해 제안된 손실함수:

$$L_{spk} = \sum_{B} |F_b(X_0) - F_b(X'_t)|$$

여기서:
- $$X_0$$: 목표 화자의 실제 멜-스펙트로그램
- $$X'_t$$: 네트워크에서 제거된(denoised) 출력
- $$F_b$$: 오디오 네트워크의 컨볼루션 계층으로부터 추출한 잠재 임베딩
- $$B$$: 오디오 네트워크의 컨볼루션 블록 개수 (처음 두 블록 제외)
- 오디오 네트워크는 이 손실로 업데이트되지 않도록 동결(freeze)

#### 2.3 모델 구조

FACE-TTS는 **Grad-TTS** 기반의 확산 모델(score-based diffusion model)을 기반으로 구성됩니다.[1]

**기본 확산 과정**[1]

정방향 프로세스(Forward Process):
$$dX_t = -\frac{1}{2}X_t\beta_t dt + \sqrt{\beta_t} dW_t$$

역방향 프로세스(Reverse Process):
$$dX_t = -\left(\frac{1}{2}X_t + S(X_t, t)\right)\beta_t dt + \sqrt{\beta_t} d\tilde{W}_t$$

이산화된 역방향 샘플링(Discretized Reverse Sampling):

```math
X_{t-\frac{1}{N}}=X_{t}+\frac{\beta _{t}}{N}\left(\frac{1}{2}X_{t}+S(X_{t},t)\right)+\sqrt{{\beta _{t}}} \tilde{W}_{t}
```

여기서:
- $$W_t$$: 표준 브라운 운동(standard Brownian motion)
- $$\beta_t$$: 노이즈 스케줄(noise schedule)
- $$S(X_t, t)$$: 스코어 함수(score function) - 노이즈가 있는 데이터의 log-density 기울기 추정
- $$N$$: 이산화 역방향 프로세스의 단계 수
- $$t \in \{\frac{1}{N}, \frac{2}{N}, ..., 1\}$$

**모델 아키텍처 구성 요소**[1]

1. **텍스트 인코더**: 텍스트 시퀀스로부터 음향 특성 생성

2. **지속시간 예측기(Duration Predictor)**: 목표 화자의 음성을 자연스럽게 발음하기 위해 예측된 음성 지속시간으로 특성 색칠(colorize)

3. **시각 네트워크(Visual Network)**: 
   - 목표 화자의 얼굴 이미지 입력
   - 화자 표현(speaker representation) 생성
   - 사전 학습된 교차-모달 생체인식 모델에서 초기화

4. **확산 모델(Diffusion Model - Noise Predictor)**:
   - 노이즈가 있는 데이터에서 데이터 분포의 기울기 추정
   - 화자 표현의 조건화로 합성 음성이 화자의 음성으로 생성되도록 유도

5. **음성 네트워크(Audio Network)**:
   - 학습 단계에서만 사용
   - 교차-모달 생체인식용 음성 임베딩 추출
   - 화자 특성 결합 손실 계산에 사용

**전체 학습 목표 함수**[1]

$$L = L_{prior} + L_{duration} + L_{diff} + \gamma L_{spk}$$

여기서:
- $$L_{prior}$$: 정규 분포로부터 평균 추정 손실
- $$L_{duration}$$: 음성-텍스트 간 단조 정렬(monotonic alignment)을 통한 발음 지속시간 제어 손실
- $$L_{diff}$$: 데이터 분포의 기울기 추정 손실
- $$\gamma$$: 가중치 계수 (경험적으로 $$1e^{-2}$$로 설정)
- $$L_{spk}$$: 화자 특성 결합 손실

#### 2.4 주요 차이점: Grad-TTS와의 비교[1]

기존 Grad-TTS는 사전 정의된 화자 코드북에 의존하여 새로운 화자 제시가 어려웠으나, FACE-TTS는:
- 얼굴 이미지로부터 직접 화자 표현 생성
- 음성 임베딩(speech embedding)의 복잡한 분포 대신 얼굴-음성의 연관 표현만 사용
- 자연스러운 화자 임베딩 일반화 달성

#### 2.5 성능 향상 및 평가 결과

**실험 설정**[1]

- **학습 데이터**: LRS3 데이터셋 (TED 비디오 기반)
  - 14,114 발화 / 2,007명 화자
  - 검증: 50 발화
  - 테스트: 412명 화자
  - 화자당 평균 약 34초(LibriTTS의 550초 대비 매우 제한적)

- **교차-모달 생체인식 모델**: VoxCeleb2(5,994명)에서 사전 학습

**정성 평가 - MOS(Mean Opinion Score) 테스트**[1]

| 방법 | 화자 조건 | 스펙트로그램 | MOS 점수 |
|------|---------|-----------|---------|
| Ground Truth | - | - | 4.865±0.001 |
| Mel.+HiFi-GAN (상한) | - | - | 4.653±0.035 |
| Grad-TTS† | 보인 화자 | 임베딩 | 3.718±0.318 |
| FACE-TTS | 보인 화자 | 음성 | 3.547±0.331 |
| FACE-TTS | 보인 화자 | **얼굴** | 3.706±0.154 |
| FACE-TTS | 미출현 화자 | 음성 | 3.218±0.249 |
| FACE-TTS | **미출현 화자** | **얼굴** | **3.282±0.219** |

**핵심 성능 지표**:
- 음향 품질: 보인 화자에서 Grad-TTS와 경쟁 수준 달성[1]
- 미출현 화자: 제로샷 적응으로도 우수한 음질(3점 이상) 유지[1]
- 얼굴 조건화가 음성 조건화보다 약간 더 나은 품질 (녹음 환경의 강건성)[1]

**객관적 평가 - 5-way 교차-모달 강제 짝 맞춤**[1]

| 방법 | 화자 정체성 | 정확도(%) |
|------|-----------|---------|
| Mel.+HiFi-GAN (상한) | - | 48.6 |
| Grad-TTS | 임베딩 | 19.4 |
| FACE-TTS ($$L_{spk}$$없음) | 얼굴 | 35.4 |
| FACE-TTS ($$L_{spk}$$포함) | **얼굴** | **38.0** |

**해석**:
- $$L_{spk}$$ 손실이 2.6% 성능 향상 기여[1]
- 여전히 상한(48.6%)보다 10% 개선 여지 있음[1]

**선호도 테스트(Preference Test)**[1]

- **AB 강제 짝 맞춤**: 합성 음성과 두 얼굴 이미지 중 맞는 이미지 선택
  - 정확도: 약 61.5%[1]

- **ABX 선호도 테스트**: 두 합성 음성과 한 얼굴 이미지 중 더 맞는 음성 선택
  - 정확도: 59.6% / 동일: 5.5%[1]

**가상 인간 음성 생성(Virtual Human Speech Generation)**[1]

Stable Diffusion으로 생성한 가상 얼굴에 대한 평가:
- 가상 얼굴-음성 쌍에 4점 Likert 척도 적용
- **평균 점수: 2.941±0.462** (Good 근처)
- 실제 LRS3 얼굴-음성: **3.471±0.291**

#### 2.6 논문의 한계

논문에서 명시적으로 언급하거나 암시적으로 드러나는 한계:[1]

1. **교차-모달 강제 짝 맞춤에서 성능 격차**: 38.0% 대 48.6% (상한) - 여전히 10% 이상의 개선 여지

2. **데이터셋의 제한성**: LRS3은 in-the-wild 음성으로 데이터셋 자체가 LibriTTS(550초/화자) 대비 불리(34초/화자)

3. **추론 단계 수의 증가**: 10개의 제거(denoising) 단계 사용으로 인한 추론 시간 증가 가능성

4. **선호도 테스트의 적당한 성능**: 60% 정확도는 기술이 여전히 개선 필요함을 시사

***

### 3. 모델의 일반화 성능 향상 가능성

#### 3.1 현재 일반화 성능의 강점

**얼굴 기반 조건화의 일반화 우월성**[1]

논문의 중요한 발견은 얼굴 조건화가 음성 조건화보다 더 나은 일반화 성능을 보인다는 것입니다:

- 얼굴 임베딩이 복잡한 음성 임베딩의 분포를 대신
- 녹음 환경의 영향을 받지 않는 안정적인 특성 표현
- 제한된 데이터(34초/화자)로도 학습 가능

**제로샷 성능**[1]

FACE-TTS는 미출현 화자에 대해 완전한 제로샷 적응 가능:
- 별도의 파인튜닝 없음
- 추가 학습 단계 불필요
- MOS 3.282점(미출현) vs 3.706점(보인 화자) - 상대적으로 작은 성능 저하(약 11.5%)

**강건성(Robustness)**[1]

다양한 각도와 표정의 얼굴 이미지로 학습하여 실제 세계 얼굴 이미지에 더 강건:
- LibriTTS만으로 학습한 기존 방법보다 우월
- in-the-wild 환경의 노이즈와 다양성 수용

#### 3.2 일반화 성능 향상의 가능성

**이론적 기반**

FACE-TTS의 일반화 향상 가능성은 다음의 이론적 근거를 가집니다:

$$L_{spk} = \sum_{B} |F_b(X_0) - F_b(X'_t)|$$

이 손실이 합성 음성의 화자 관련 잠재 분포를 실제 음성과 유사하게 만들도록 함으로써, **화자-불변 특성 공간(speaker-invariant feature space)** 구축을 강화합니다.

**개선 가능 영역 분석**[1]

1. **교차-모달 매칭 성능**: 현재 38% → 48.6% 목표
   - 더 강력한 교차-모달 생체인식 모델 사용
   - 손실 함수 개선 (현재 L1 거리 → 더 정교한 메트릭)
   
2. **다양한 언어 데이터로의 일반화**:
   - LRS3는 주로 영어 기반
   - 다국어 데이터로 학습하면 언어 불변 특성 학습 가능
   
3. **더 짧은 참조 음성으로의 적응**:
   - 현재는 LRS3의 전체 발화 사용
   - 극단적 제로샷(단 1~2초)에서의 성능 개선 가능
   
4. **스타일 및 운율(Prosody) 제어**:
   - 현재 모델은 주로 화자 정체성 중심
   - 정서, 강조, 속도 등의 부수적 특성(paralinguistic features) 명시적 모델링

#### 3.3 관련 최신 연구에서의 발전 방향 (2020년 이후)

**관련 연구 1: DEX-TTS (2024)**[2]

확산 기반 표현 음성 TTS는 다음의 개선을 제안합니다:
- 참고 음성으로부터의 스타일 표현 강화
- 시간 변동성(time variability)을 명시적으로 모델링
- FACE-TTS보다 더 풍부한 스타일 재현

**관련 연구 2: StyleTTS 2 (2023)**[3]

스타일 확산과 적대적 학습의 조합으로:
- 참고 음성 없이도 텍스트에 최적의 스타일 생성
- 대규모 음성 언어 모델(SLM)과의 결합
- 인간 수준의 TTS 합성 달성

**관련 연구 3: DiTTo-TTS (2024)**[4]

확산 변환기(Diffusion Transformer) 기반으로:
- 도메인 특화 요소(음소, 지속시간) 제거로 확장성 증대
- 대규모 잠재 확산 모델(LDM) 활용
- 더 일반화된 아키텍처

**관련 연구 4: FVTTS (2024)**[5]

FACE-TTS를 개선한 후속 연구:
- 이미지 특성 융합으로 음성 특성 추출 향상
- VITS 기반 아키텍처 사용으로 더 빠른 추론
- 다양한 데이터셋에서 검증

**관련 연구 5: 제로샷 적응 연구들 (2023-2024)**[6][7][8]

- **YourTTS (2023)**: 다국어 제로샷 TTS의 SOTA
- **StyleTTS-ZS (2024)**: 경경량 제로샷 TTS로 빠른 추론
- **DS-TTS (2025)**: 이중 스타일 인코더로 스타일 표현 강화

#### 3.4 제안되는 일반화 성능 향상 전략

**단기 개선 (1-2년)**:

1. **향상된 손실 함수**:
   - 현재 L1 거리 대신 코사인 유사도 또는 대조 손실(contrastive loss) 사용
   - 다중 레벨 특성 공간에서의 정렬 강제

2. **데이터 증강**:
   - LRS3의 얼굴 이미지 변환(각도, 밝기, 표정)으로 강건성 증대
   - 음성 증강(잡음 추가, 피치 변경)을 통한 다양성 확대

3. **더 강력한 교차-모달 모델**:
   - 최신 비전 트랜스포머(Vision Transformer) 기반 얼굴 인코더
   - 대조 학습(contrastive learning) 기반 사전 학습

**중기 개선 (2-3년)**:

1. **다국어 학습**:
   - 여러 언어의 오디오-비주얼 데이터셋으로 확장
   - 언어 불변 특성 학습으로 제로샷 교차 언어 TTS 가능

2. **계층적 표현 학습**:
   - 화자 정체성: 고수준
   - 운율 특성: 중수준  
   - 음향 세부: 저수준
   로 구분하여 명시적 모델링

3. **메타-러닝 통합**:
   - 극단적 제로샷(1~2초) 성능 개선
   - 몇 가지 예제로 빠른 적응

**장기 개선 (3년 이상)**:

1. **멀티-모달 융합**:
   - 얼굴뿐 아니라 음소 정보, 입 모양, 머리 움직임 활용
   - 완전한 비디오-투-스피치 시스템으로 확장

2. **어댑티브 아키텍처**:
   - 입력 특성과 데이터 특성에 따라 동적으로 모델 복잡도 조정
   - 하드웨어 제약 환경에 대한 자동 최적화

***

### 4. 논문의 영향력과 향후 연구 고려 사항

#### 4.1 학술적 영향력

**혁신적 기여 측면**:

1. **패러다임 전환**: 
   - 기존 음성 기반 화자 제어 → **얼굴 기반 화자 제어**로 전환
   - 더 접근 가능한 데이터(얼굴 이미지)로 음성 합성 가능하게 함[1]

2. **교차-모달 학습의 새로운 응용**:
   - 얼굴과 음성의 강한 상관성을 TTS에 처음 활용[1]
   - 멀티모달 학습 분야의 확장[9][10]

3. **확산 모델의 고급 응용**:
   - Grad-TTS 기반 조건화 메커니즘 개선
   - 새로운 특성 바인딩 손실 도입[1]

**실용적 영향력**:

1. **가상 인간 합성**:
   - 가상 아바타(synthetic human)에 일관성 있는 음성 할당
   - 메타버스, 게임, 영상 제작의 활용 가능성[1]

2. **접근성 개선**:
   - 음성 장애인을 위한 개인화 음성 합성
   - 청각 장애인을 위한 시각 피드백 기반 음성 생성

3. **프라이버시 이점**:
   - 음성 샘플 수집 불필요
   - GDPR 등 규제 준수 용이

#### 4.2 영향을 받은 후속 연구 (2023-2025)

**직접적 확장 연구**:

1. **FVTTS (2024)**: FACE-TTS의 개선 버전
   - 얼굴 특성 융합 강화[5]
   - VITS 기반으로 빠른 추론[5]
   - 다양한 데이터셋 검증[5]

2. **HFSD-V2C (2025)**: 계층적 얼굴-스타일 확산 모델[11]
   - 교차-모달 생체인식으로 미출현 화자 모델링
   - 텍스트, 음성, 비디오 수준의 계층적 운율 모델링

3. **Face2VoiceSync (2025)**: 경량 얼굴-음성 일관성 모델[12]
   - 경량 VAE로 모달리티 간격 제거
   - 유연한 음성 커스터마이제이션

**관련 분야 발전**:

1. **제로샷 음성 복제(Zero-Shot Voice Cloning)**: 
   - YourTTS, StyleTTS-ZS 등으로 급속히 발전
   - 단일 음성 샘플로도 고품질 합성 가능[7][8]

2. **음성 변환(Voice Conversion)**:
   - 기능 분리 기반 방법 발전[13]
   - 교차-언어 적응 가능[14]

3. **멀티모달 TTS**:
   - ViT-TTS: 시각 정보를 TTS에 통합[15]
   - 환경 인식 TTS (DAIEN-TTS): 배경 음성 환경 포함[16]

#### 4.3 향후 연구 시 고려할 핵심 사항

**기술적 고려사항**:

1. **데이터 품질 표준화**:
   - 얼굴 이미지의 해상도, 조명, 각도 표준화 필요
   - 얼굴 인식 실패 사례에 대한 대응 메커니즘

2. **교차-모달 정렬 개선**:
   $$\text{더 나은 손실 함수: } L = \text{Triplet Loss} + \text{Contrastive Loss} + L_{spk}$$
   - 단순 L1 거리보다 정교한 메트릭 개발

3. **추론 효율성**:
   - 확산 모델의 느린 추론(10 스텝) 개선
   - 지식 증류(knowledge distillation) 적용[17]
   - 가속화된 샘플링 알고리즘(ODE 솔버 등)

4. **조건화 메커니즘 개선**:
   - 얼굴의 여러 특성(나이, 성별, 감정)을 명시적으로 분리
   - 세분화된 제어(fine-grained control) 가능하게

**평가 방법론 개선**:

1. **더 엄격한 교차-모달 평가**:
   - 현재 5-way 강제 매칭 → 10-way 또는 20-way로 확장
   - 음성 특성 다양성 평가 메트릭 개발

2. **사용자 연구 강화**:
   - 대규모 MOS 테스트 (현재 17명 → 100명 이상)
   - 장시간 음성 품질 평가 (현재 10 발화)

3. **일반화 평가**:
   - 훈련 데이터와 다른 민족/성별/연령 분포 테스트
   - 다양한 녹음 환경에서의 강건성 평가

**윤리 및 사회적 고려사항**:

1. **딥페이크 방지**:
   - 생성된 음성의 출처 명확히
   - 워터마크 또는 인증 메커니즘 도입
   - 생성 음성의 법적 지위 명확화

2. **개인정보 보호**:
   - 얼굴 데이터 사용 동의 체계
   - 개인 음성 모델의 소유권 규정

3. **공정성**:
   - 모든 민족/성별에 대한 균등한 성능
   - 접근성 불평등 해소

4. **투명성**:
   - 생성 음성 공개 시 고지 의무
   - 모델 가중치 및 학습 데이터 공개 검토

#### 4.4 특정 응용 분야에서의 고려사항

**게임 및 메타버스**:
- 실시간 처리 성능 요구
- 다양한 캐릭터 페르소나 표현

**의료 및 접근성**:
- 개인 음성 재현의 정확도 및 지속성
- 환자 데이터 보호 및 규제 준수

**영화/비디오 제작**:
- 립싱크 정확도 향상 필요
- 복잡한 감정 표현의 정확한 전달

**교육**:
- 다국어 학습 지원
- 발음 피드백 기능 통합

***

### 5. 2020년 이후 주요 관련 연구 탐색 및 동향

#### 5.1 확산 모델 기반 TTS의 발전 궤적

**2020-2021년: 기초 확립**
- Grad-TTS (2021): 확산 모델을 TTS에 처음 적용[18]
- Score-based SDE (2021): 이론적 기초 제공[19]

**2022년: 다양한 확산 기반 방법 등장**
- Guided-TTS (2022): 분류기 가이던스 기반 TTS[20]
- SATTS (2022): 화자 어트랙터를 활용한 제로샷 적응[21]
- Grad-StyleSpeech (2023): 임의 화자 적응형 TTS[22]

**2023년: 고급 스타일 모델링**
- StyleTTS 2 (2023): 스타일 확산과 적대적 학습 조합[3]
- Guided-TTS 2 (2022): 고품질 적응형 TTS[23]
- LightGrad (2023): 경량 확산 모델[24]
- DCTTS (2023): 이산 확산 모델[25]
- ViT-TTS (2023): 시각 정보 통합[15]

**2024-2025년: 효율성과 확장성 개선**
- DEX-TTS (2024): 향상된 스타일 모델링[2]
- DiTTo-TTS (2024): 도메인 특화 요소 제거[4]
- StyleTTS-ZS (2024): 경량 제로샷 TTS[8]
- DiFlow-TTS (2025): 이산 흐름 매칭[26]
- DAIEN-TTS (2025): 환경 인식 TTS[16]

#### 5.2 제로샷 및 적응형 TTS의 발전

**핵심 트렌드**:

1. **화자 임베딩 고도화**:
   - 단순 고정 차원 벡터 → 가변 길이 시퀀스
   - 이중 스타일 인코더 (메� + MFCC)[27]

2. **특성 분리(Disentanglement)**:
   - 화자 정체성, 운율, 감정 등 명시적 분리
   - 독립적 제어 가능[28][13]

3. **메타-러닝 통합**:
   - 소수 샘플 적응 가능[29]
   - MAML 등의 기법 적용

**성능 개선**:

| 연도 | 모델 | 제로샷 성능(MOS) | 화자 유사도 |
|------|------|-----------------|-----------|
| 2020 | Attentron | ~3.5 | 높음 |
| 2021 | Grad-TTS | ~3.6 | 중상 |
| 2023 | StyleTTS 2 | ~4.0+ | 매우 높음 |
| 2023 | YourTTS | ~3.9 | 높음 |
| 2024 | StyleTTS-ZS | ~3.85 | 높음 |

#### 5.3 교차-모달 학습의 발전

**기초 연구**:
- Seeing Voices and Hearing Faces (2018): 교차-모달 생체인식 기초[30]
- Face-voice correlation 확인

**응용 발전**:
- FACE-TTS (2023): 얼굴 기반 TTS[1]
- FVTTS (2024): 개선된 얼굴 인코더[5]
- Face2VoiceSync (2025): 경량 모달리티 정렬[12]
- HFSD-V2C (2025): 계층적 얼굴-스타일 모델[11]

**비디오-기반 확장**:
- 립 동기(lip-sync) 정보 활용[31][14]
- 완전한 비디오-투-음성 시스템으로 진화

#### 5.4 핵심 트렌드 요약

**성능 측면**:
- MOS 점수: 3.5~3.7 (2020) → 4.0+ (2023)
- 화자 유사도: 지속적 개선
- 추론 속도: 점진적 가속화

**기술 측면**:
- 확산 모델의 안정성 확보
- 특성 분리 및 세분화 제어
- 교차-모달 학습의 실용화

**응용 측면**:
- 실시간 처리 가능 모델 등장
- 다국어 및 교차 언어 지원
- 가상 인간 합성의 현실화

***

### 결론

**FACE-TTS (Imaginary Voice)**는 얼굴 이미지를 사용하여 제로샷 음성 합성을 가능하게 한 획기적인 연구입니다. 확산 모델, 교차-모달 생체인식, 그리고 새로운 화자 특성 결합 손실의 조합으로, 미출현 화자의 음성도 고품질로 합성할 수 있음을 보였습니다.[1]

현재의 제한점(교차-모달 매칭 38%, 제한된 데이터셋)에도 불구하고, 후속 연구들(FVTTS, HFSD-V2C, Face2VoiceSync)이 계속 이를 개선하고 있으며, 더 강력한 교차-모달 모델, 데이터 증강, 그리고 계층적 표현 학습을 통해 일반화 성능의 상당한 향상이 가능합니다.

향후 연구는 (1) 기술적으로는 추론 효율성과 교차-모달 정렬 개선에, (2) 평가 방면에서는 대규모 다양한 사용자 연구에, (3) 윤리적으로는 딥페이크 방지와 개인정보 보호에 초점을 맞춰야 합니다. 음성 합성 기술이 점점 더 현실적이고 광범위하게 사용될수록, 이러한 책임 있는 개발이 더욱 중요해질 것입니다.

***

### 참고 문헌 목록 (2020년 이후 주요 논문)

1. Imaginary Voice: Face-styled Diffusion Model for Text-to-Speech (2023)[1]
2. DEX-TTS: Diffusion-based EXpressive Text-to-Speech with Style Modeling on Time Variability (2024)[2]
3. DiTTo-TTS: Diffusion Transformers for Scalable Text-to-Speech without Domain-Specific Factors (2024)[4]
4. LightGrad: Lightweight Diffusion Probabilistic Model for Text-to-Speech (2023)[24]
5. DCTTS: Discrete Diffusion Model with Contrastive Learning for Text-to-speech Generation (2023)[25]
6. ViT-TTS: Visual Text-to-Speech with Scalable Diffusion Transformer (2023)[15]
7. Grad-StyleSpeech: Any-speaker Adaptive Text-to-Speech Synthesis with Diffusion Models (2023)[22]
8. StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training (2023)[3]
9. Guided-TTS: A Diffusion Model for Text-to-Speech via Classifier Guidance (2022)[20]
10. Guided-TTS 2: A Diffusion Model for High-Quality Adaptive Text-to-Speech with Untranscribed Data (2022)[23]
11. Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech (2021)[18]
12. YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone (2023)[7]
13. SATTS: Speaker Attractor Text to Speech (2022)[21]
14. FVTTS: Face Based Voice Synthesis for Text-to-Speech (2024)[5]
15. Attentron: Few-Shot Text-to-Speech Utilizing Attention-Based Multiple Reference Encoders (2020)[32]
16. Seeing Voices and Hearing Voices: Learning Discriminative Embeddings using Cross-Modal Self-Supervision (2020)[10]
17. Seeing Voices and Hearing Faces: Cross-Modal Biometric Matching (2018)[30]
18. Optimizing Feature Fusion for Improved Zero-shot Adaptation in Text-to-Speech Synthesis (2024)[6]
19. Zero-shot Voice Cloning Using Variational Embedding with Attention Mechanism (2021)[31]
20. StyleTTS-ZS: Efficient High-Quality Zero-Shot Text-to-Speech Synthesis (2024)[8]
21. DS-TTS: Zero-Shot Speaker Style Adaptation from Voice (2025)[27]
22. DiFlow-TTS: Discrete Flow Matching with Factorized Speech Tokens for Low-Latency Zero-Shot TTS (2025)[26]
23. DAIEN-TTS: Disentangled Audio Infilling for Environment-Aware Text-to-Speech Synthesis (2025)[16]
24. HFSD-V2C: Hierarchical Face-Styled Diffusion Model for Zero-Shot Visual Voice Cloning (2025)[11]
25. Face2VoiceSync: Lightweight Face-Voice Consistency for Text-to-Speech (2025)[12]
26. Zero-shot Voice Conversion Based on Feature Disentanglement (2024)[13]
27. Speeding Up Text-To-Speech Diffusion Models by Distillation (2023)[17]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02887122-a388-4747-94e5-ed1d5030b3de/2302.13700v1.pdf)
[2](https://arxiv.org/abs/2406.19135)
[3](https://arxiv.org/pdf/2306.07691.pdf)
[4](http://arxiv.org/pdf/2406.11427.pdf)
[5](https://www.isca-archive.org/interspeech_2024/lee24_interspeech.pdf)
[6](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-024-00351-9)
[7](https://arxiv.org/abs/2112.02418)
[8](https://arxiv.org/abs/2409.10058)
[9](https://aclanthology.org/2025.coling-main.352.pdf)
[10](https://www2.informatik.uni-hamburg.de/wtm/publications/2021/PWQW21/hearing_faces_asru_final_header.pdf)
[11](https://aclanthology.org/2025.ccl-1.77.pdf)
[12](https://www.isca-archive.org/interspeech_2025/kang25c_interspeech.pdf)
[13](https://www.sciencedirect.com/science/article/abs/pii/S0167639324001146)
[14](https://www.isca-archive.org/interspeech_2021/maniati21_interspeech.html)
[15](https://aclanthology.org/2023.emnlp-main.990.pdf)
[16](https://arxiv.org/abs/2509.14684)
[17](https://developer.nvidia.com/blog/speeding-up-text-to-speech-diffusion-models-by-distillation/)
[18](https://facetts.github.io)
[19](https://www.isca-archive.org/interspeech_2020/paul20_interspeech.html)
[20](https://arxiv.org/pdf/2111.11755.pdf)
[21](https://arxiv.org/abs/2207.06011)
[22](http://arxiv.org/pdf/2211.09383.pdf)
[23](https://www.isca-archive.org/interspeech_2025/chen25b_interspeech.pdf)
[24](https://arxiv.org/pdf/2308.16569.pdf)
[25](http://arxiv.org/pdf/2309.06787.pdf)
[26](https://arxiv.org/abs/2509.09631)
[27](https://arxiv.org/html/2506.01020v1)
[28](https://aclanthology.org/2024.findings-emnlp.533.pdf)
[29](https://ieeexplore.ieee.org/document/9756900)
[30](https://openaccess.thecvf.com/content_cvpr_2018/papers/Nagrani_Seeing_Voices_and_CVPR_2018_paper.pdf)
[31](https://ieeexplore.ieee.org/document/9660599/)
[32](https://www.isca-archive.org/interspeech_2020/choi20c_interspeech.pdf)
[33](https://s-space.snu.ac.kr/handle/10371/210120)
[34](https://pmc.ncbi.nlm.nih.gov/articles/PMC10708733/)
[35](https://liner.com/ko/review/dittotts-diffusion-transformers-for-scalable-texttospeech-without-domainspecific-factors)
[36](https://ieeexplore.ieee.org/document/10317526/)
[37](http://arxiv.org/pdf/2407.01291.pdf)
[38](http://arxiv.org/pdf/2207.06011.pdf)
[39](https://arxiv.org/pdf/2501.08566.pdf)
[40](http://arxiv.org/pdf/2310.03538.pdf)
[41](https://arxiv.org/pdf/2202.10712.pdf)
[42](https://arxiv.org/html/2401.13921v1)
[43](https://www.isca-archive.org/interspeech_2024/gusev24_interspeech.pdf)
[44](https://arxiv.org/abs/2012.07252)
[45](https://arxiv.org/abs/2204.00990)
