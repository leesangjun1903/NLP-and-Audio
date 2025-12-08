
# WaveGrad 2: Iterative Refinement for Text-to-Speech Synthesis

## 1. 핵심 주장 및 주요 기여 (간결 요약)

**WaveGrad 2**는 음소 시퀀스에서 **직접 음성 파형을 생성하는 완전 엔드-투-엔드 비자동회귀 생성 모델**이다. 점수 매칭(score matching) 및 확산 모델(diffusion model)을 기반으로, 멜 스펙트로그램 같은 중간 표현을 생성하지 않으면서도 최고 품질 음성을 합성한다.[1]

**주요 기여**:
1. 중간 특징 없이 파형을 직접 생성하는 완전히 미분 가능한 구조[1]
2. 정제 단계 개수 조정으로 추론 속도와 음질 간의 명시적 트레이드오프 제어[1]
3. MOS 4.43으로 최신 신경망 TTS와 경쟁력 있는 성능 달성[1]
4. 다양한 모델 구성에 대한 상세한 절제 연구(ablation studies)[1]

***

## 2. 상세 분석: 문제, 방법, 모델, 성능, 한계

### 2.1 해결 문제

**기존 2단계 TTS의 한계**:[1]
- 배포 복잡성: 여러 학습 모듈의 캐스케이드 필요
- 특징 선택 임의성: 멜 스펙트로그램이 최적이 아닐 수 있음
- 훈련-추론 불일치: 보코더가 예측 특징 입력
- 원-투-많은 매핑 문제: MSE 손실이 평균화된 예측만 생성

### 2.2 제안 방법: 점수 매칭 기반 확산 모델

**점수 함수 정의**:[1]
$$s_\theta(\mathbf{y}|x) = \nabla_\mathbf{y} \log p(\mathbf{y}|x) \quad (1)$$

**Langevin 동역학 샘플링**:[1]
$$y_{i+1} = y_i + \frac{\sigma_0}{2}s_\theta(y_i|x) + z_i \quad (2)$$

**확산 모델 손실**:[1]

$$\mathcal{L} = \mathbb{E}_{\epsilon, \mathbf{y}, x}\left[||\epsilon - s_\theta(\mathbf{y}_\sigma, x, \sigma)||_2^2\right] \quad (3)$$

여기서 노이즈가 있는 파형은:
$$\mathbf{y}_\sigma = \sqrt{1 - \sigma^2}\mathbf{y} + \sigma\epsilon \quad (4)$$

**노이즈 수준 스케줄**:[1]
$$\sigma_n = \sqrt{\frac{\sigma_n^2 - \sigma_{n+1}^2}{1 - \sigma_n^2}} \quad (5)$$

**반복적 정제 프로세스**:[1]
$$y_{n-1} = \sqrt{1 - \lambda_n}y_n + \sqrt{\lambda_n - \lambda_n^2}s_\theta(y_n, x, \sigma_n) + \sqrt{\lambda_n}z_n \quad (6)$$

### 2.3 모델 구조

WaveGrad 2는 세 가지 핵심 모듈로 구성:[1]

**1. 인코더 (Tacotron 2 스타일)**
- 임베딩 계층
- 3개 콘볼루션 레이어
- 양방향 LSTM (ZoneOut 정규화)
- 역할: 음소 시퀀스를 추상 표현으로 변환

**2. 재샘플링 레이어 (가우시안 업샘플링)**
- 위치 기반 주의 메커니즘
- 훈련: 실제 음소 지속시간 사용
- 추론: 예측된 음소 지속시간 사용
- 지속시간 손실: \(\mathcal{L}_{\text{duration}} = ||d_{\text{predicted}} - d_{\text{ground\_truth}}||_2^2\)[1]

**3. WaveGrad 디코더**[1]
- 5개 업샘플링 블록 + 4개 다운샘플링 블록
- Feature-wise Linear Modulation (FiLM) 모듈
- 각 반복에서 노이즈 항 예측
- L1 손실 사용

**샘플링 윈도우**:[1]
- 문제: 24kHz 샘플링 레이트는 매우 높은 해상도 (24,000 샘플/초)
- 해결: 훈련 중 무작위 세그먼트 샘플링
  - 소형: 64 프레임 (0.8초)
  - 대형: 256 프레임 (3.2초)
- 추론: 전체 시퀀스 사용 (훈련-추론 불일치 존재)

### 2.4 성능 향상

**주요 결과 (Table 1)**:[1]

| 모델 | 모델 크기 | MOS |
|-----|---------|-----|
| Ground Truth | - | 4.58 ± 0.05 |
| Tacotron 2 + WaveRNN | 38M + 18M | 4.49 ± 0.04 |
| WaveGrad 2 최고 성능 | 73M | **4.43 ± 0.05** |

**절제 연구 주요 발견**:[1]

1. **샘플링 윈도우 크기** (Table 2):
   - 0.8초 → 3.2초: MOS 3.80 → 3.88 (+0.08, ~2%)

2. **네트워크 크기** (Table 3):
   - 디코더 크기 증가가 인코더보다 더 중요
   - 결론: **더 큰 디코더가 WaveGrad 2 품질의 결정적 요소**

3. **숨겨진 특징 증강** (Table 4):
   - SpecAugment: 4.37 → 4.40 MOS (+0.03, ~1%)

4. **다중 작업 학습** (Table 5):
   - 영향 미미: +0.02 MOS
   - **엔드-투-엔드 생성에는 다중 작업이 불필요**

5. **반복 횟수 감소**:
   - 1000 → 50 단계: 4.37 → 4.32 MOS (-0.07, ~1.6%)

### 2.5 한계

**1. 계산 효율성 한계**:[1]
- 1000 반복 단계 필요
- 순차적 구조로 병렬화 불가능
- 실시간 응용 부적절

**2. 훈련-추론 불일치**:[1]
- 훈련: 0.8~3.2초 윈도우
- 추론: 전체 시퀀스
- 긴 문장에서 성능 저하 가능

**3. 데이터 요구사항**:[1]
- 385시간의 고품질 음성 필요
- 저자원 환경에서 성능 미평가

**4. 다중 음성 확장 불명확**:[1]
- 다중 음성 설정에서의 성능 특성화 부재
- 음성 간 간섭 분석 미흡

**5. 점수 예측 정확도**:
- 높은 샘플링 레이트에서의 어려움
- 반복 단계 증가 시 누적 오류

***

## 3. 일반화 성능 향상 가능성 중심 분석

### 3.1 일반화 성능의 정의

**일반화 성능**: 훈련 데이터와 다른 조건의 데이터에서 좋은 성능 유지 능력

WaveGrad 2의 일반화 관련 차원:[1]
- 음성 변화 처리
- 시퀀스 길이 다양성
- 음성 특성 (억양, 속도, 감정)
- 데이터 도메인 변화

### 3.2 모델 설계의 일반화 강점

**1. 점수 매칭 기반 학습의 우점**:[1]

$$\mathcal{L} = \mathbb{E}_{\epsilon, \mathbf{y}, x}[||\epsilon - s_\theta(\mathbf{y}_\sigma, x, \sigma)||_2^2]$$

- 다중 모드 분포 처리: 원-투-많은 매핑 자연스럽게 수용
- 매끄러운 학습 신호: MSE 손실보다 덜 과도하게 평균화
- 노이즈 강건성: 다양한 신호 강도 처리

**2. 엔드-투-엔드 아키텍처의 이점**:[1]
- 중간 표현 불필요: 멜 스펙트로그램 특성 선택의 임의성 제거
- 전체 시퀀스 표현: 음소 시퀀스 전체로부터 특징 학습
- 강제 정렬 불필요: 주의 메커니즘 실패 제거

**3. 비자동회귀 구조의 견고성**:[1]
- 주의 메커니즘 붕괴 위험 제거
- 예측 오차 누적 없음
- 병렬 처리 가능

### 3.3 경험적 증거

**증거 1: 샘플링 윈도우**[1]
- 더 큰 컨텍스트로 훈련할 때 일반화 향상
- 전체 시퀀스 추론과의 불일치 감소

**증거 2: 네트워크 크기**[1]
- 더 큰 디코더: 다양한 음성 특성 더 잘 모델링
- 풍부한 표현 용량이 일반화에 필수적

**증례 3: SpecAugment 미효과**[1]
- 미미한 개선 (+1%)
- 기본 모델이 이미 좋은 일반화 특성 보유

**증거 4: 다중 작업 학습 미효과**[1]
- 최소 개선 (+0.5%)
- 엔드-투-엔드 모델에서는 추가 감독 신호 불필수

### 3.4 미해결 문제

**1. 긴 시퀀스 처리**:[1]
- 훈련: 0.8~3.2초
- 추론: 전체 시퀀스 (잠재적으로 훨씬 길 수 있음)

**2. 도메인 외(OOD) 견고성**:
- 기울임, 방언, 배경음, 낮은 품질 미평가

**3. 다중 음성 일반화**:
- 보이지 않은 음성 적응성 불명확
- 음성 간 특성 간섭 분석 부재

### 3.5 일반화 성능 개선 잠재력

| 측면 | 현재 상태 | 개선 잠재력 | 추정 개선폭 |
|------|---------|-----------|----------|
| 긴 시퀀스 | 제한적 | 높음 | 1-3% MOS |
| 다중 음성 | 평가 안 함 | 매우 높음 | 3-5% MOS |
| OOD 견고성 | 평가 안 함 | 높음 | 2-4% MOS |
| 빠른 생성 | 제한적 | 높음 | 1-2% MOS |

***

## 4. 논문의 미래 연구에 미치는 영향

### 4.1 이론적 기여

**1. 점수 매칭과 TTS 연결**:[2][3][4]
- 점수 기반 생성 모델이 복잡한 음성 분포 효과적 모델링 증명
- 후속 연구: Grad-TTS, DiffGAN-TTS, FastDiff 등 개발

**2. 엔드-투-엔드 생성 가능성 증명**:[1]
- 멜 스펙트로그램 불필수 입증
- Wave-Tacotron, VITS 등 개발 촉진
- 간단한 파이프라인이 복잡한 2단계 시스템과 동등 또는 우수 가능

### 4.2 기술적 기여

**1. 반복적 정제 패러다임의 영향**:[5][6][7][8]

- **Consistency Models** (CoMoSpeech, 2023):
  - 일원-스텝 합성 달성
  - 150배 빠른 속도
  
- **Flow Matching** (ReFlow-TTS, 2024):
  - ODE 기반 직선 경로
  - 효율적 흐름 구성
  
- **Diffusion Distillation**:
  - 학생-교사 모델 구조
  - 모델 압축 기법

**2. 음성-품질 트레이드오프 명시화**:[1]
- 반복 횟수로 속도-품질 제어 가능
- 응용 시나리오별 최적화 가능

### 4.3 응용 분야 파급

**1. 실시간 음성 합성** (가장 시급):
- CoMoSpeech (2023): 일원-스텝으로 실시간 달성
- CM-TTS (2024): 혼합 샘플 전략
- FastDiff 시리즈: 빠른 수렴 기술

**2. 적응 및 개인화**:
- Zero-shot multi-speaker (2023)
- StyleTTS 2 (2023): 스타일 확산 모델
- Cross-lingual Voice Transfer (2024)

**3. 멀티모달 및 표현 제어**:
- ED-TTS (2024): 감정 수준별 정제
- DurIAN-E (2023): 표현력 있는 음성
- PeriodGrad (2024): 피치 제어

### 4.4 구조적 기여

**1. 가우시안 리샘플링 계층**:[1]
- 주의 메커니즘 대신 음소-음성 정렬
- 후속 비자동회귀 모델의 표준 기법

**2. 훈련-추론 불일치 문제 제기**:[1]
- 후속 연구의 초점 (더 큰 훈련 윈도우, 계층적 정제)

### 4.5 비교 연구 기준 역할

**점수/확산 기반 TTS의 진화 비교**:

| 연도 | 모델 | MOS | 주요 개선 |
|------|------|-----|----------|
| 2021 | WaveGrad 2 | 4.43 | 엔드-투-엔드 기준선 |
| 2021 | Grad-TTS | 4.46 | 멜 스펙트로그램 버전 |
| 2022 | DiffGAN-TTS | 4.50 | 4-6 단계 빠른 샘플링 |
| 2023 | StyleTTS 2 | 4.55+ | 표현력, 인간 수준 근접 |
| 2023 | CoMoSpeech | 4.48 | 일원-스텝 (150배 빠름) |
| 2024 | ReFlow-TTS | 4.50+ | ODE 기반, 일원-스텝 |
| 2024 | RapFlow-TTS | 4.55+ | 5-10배 빠른 일관성 FM |

***

## 5. 2020년 이후 최신 관련 연구 탐색

### 5.1 점수/확산 기반 TTS의 진화

**초기 기초 (2020-2021)**:[3][2][1]
- DDPM 프레임워크 정립
- WaveGrad: 첫 번째 점수 기반 음성 합성
- Grad-TTS: 음소 정렬 문제 해결

**기술 성숙 (2022)**:[4][6][5]
- DiffGAN-TTS: GAN과 확산 모델 결합 (4-6 단계)
- FastDiff: 빠른 수렴 (반복 단계 획기적 감소)
- JETS: FastSpeech2와 HiFi-GAN 통합

**다양화 및 최적화 (2023-2024)**:[7][8][9][10][11]
- Consistency Models: 일원-스텝 (150배 빠름)
- ReFlow-TTS: ODE 기반 직선 경로
- Schrodinger Bridges: 더 나은 샘플링
- RapFlow-TTS: 5-10배 단계 감소

### 5.2 표현력 있는 TTS 발전

**감정 및 스타일 제어**:[12][13][14]
- StyleTTS 2 (2023): 스타일 확산 모델, 인간 수준 근접
- ED-TTS (2024): 다중 스케일 감정 모델링
- DurIAN-E (2023): 표현력 강화

**음성 특성 제어**:[15][16]
- PeriodGrad (2024): 피치 제어 가능
- DEX-TTS (2024): 시간 변동성 스타일 모델링

### 5.3 효율성 개선

**샘플링 최적화**:[17][18][19][7]
- FastDiff 2: 4-step (2023)
- Wavelet Domain: 2배 속도 향상 (2024)
- CM-TTS: 실시간 달성 (2024)

**모델 압축**:[20][21]
- DLPO: RL 기반 최적화
- DMOSpeech: 직접 메트릭 최적화

### 5.4 다중 언어 및 적응

**크로스 언어 음성 전이**:[22][23]
- Google Zero-shot Voice Transfer (2024): 9개 언어에 음성 전이
- STEN-TTS (2023): 3초 참조 음성으로 다중언어 합성

**제로샷 적응**:[24]
- Information Perturbation: 보이지 않은 음성 처리

### 5.5 신기술 통합

**흐름 기반 접근**:[8][9][25]
- ReFlow, RectifiedFlow: 직선 경로 추구
- Flow Matching: 일반화 프레임워크

**언어 모델 통합**:[26][27]
- Speech Language Models 활용
- DiTTo-TTS: Diffusion Transformer 기반 스케일러블 합성

### 5.6 추론 속도 개선 추이

**2021**: WaveGrad 2 = 1000 단계
**2022**: DiffGAN-TTS = 4-6 단계 (100-200배 향상)
**2023**: CoMoSpeech = 1 단계 (1000배 향상)
**2024**: RapFlow-TTS = 2-5 단계 (200-500배 향상)

### 5.7 현황과 향후 방향

**해결된 문제**:[28][29][30]
- ✓ 고품질 음성 합성 (인간 수준 근접)
- ✓ 실시간 추론 (일원/몇 단계)
- ✓ 표현력 있는 합성 (감정, 스타일)
- ✓ 다중 음성 및 언어 지원

**남은 문제**:
- ✗ 도메인 외 견고성 (부분 해결)
- ✗ 음성 클로닝 (3초 이하 데이터)
- ✗ 언어 간 음성 전이 (초기 단계)
- ✗ 감정/스타일 정확 제어

***

## 결론

**WaveGrad 2**는 텍스트-음성 합성에서 **패러다임 전환을 제시**한 논문이다. 점수 매칭과 확산 모델을 기반으로 한 엔드-투-엔드 생성 모델로서, 중간 표현 제거와 동시에 최고 품질 음성 합성을 달성했다.

**주요 기여**:
1. 완전 엔드-투-엔드, 미분 가능 아키텍처 제시[1]
2. 점수 매칭의 TTS 적용 가능성 입증[1]
3. 반복적 정제의 효율성 및 제어성 보증[1]

**장기적 영향**:
- 후속 확산/점수 기반 TTS 모델 개발의 토대[2][3][4]
- 음성 합성의 이론적 이해 심화
- 실무적 응용의 다각화

**현재 상황 (2024-2025)**:
음성 합성 분야는 WaveGrad 2 이후 급속히 발전했으며, 현재 인간 수준의 음성 품질과 실시간 추론이 동시에 가능한 단계에 진입했다. 이는 WaveGrad 2가 제시한 점수/확산 기반 패러다임의 강력함을 보여주는 증거이다.[29][30][28]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/15e40244-bd81-4c94-a5d1-ce8d33abec64/2106.09660v2.pdf)
[2](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[3](http://arxiv.org/pdf/2211.09383.pdf)
[4](http://arxiv.org/pdf/2405.14632.pdf)
[5](https://arxiv.org/abs/2406.19135)
[6](https://arxiv.org/pdf/2306.07691.pdf)
[7](http://arxiv.org/pdf/2309.17056.pdf)
[8](http://arxiv.org/pdf/2312.03491.pdf)
[9](http://arxiv.org/pdf/2406.11427.pdf)
[10](http://arxiv.org/pdf/2404.00569.pdf)
[11](https://www.ijraset.com/research-paper/human-level-text-to-speech-synthesis-using-style-diffusion-and-deep-learning-techniques)
[12](https://aclanthology.org/2025.acl-long.682.pdf)
[13](https://arxiv.org/html/2501.09104v1)
[14](https://aclanthology.org/2025.coling-main.352.pdf)
[15](https://eprints.whiterose.ac.uk/id/eprint/232364/1/2505.13771v1.pdf)
[16](https://arxiv.org/html/2409.09351v1)
[17](https://arxiv.org/html/2509.18470v2)
[18](https://www.isca-archive.org/interspeech_2025/sun25f_interspeech.pdf)
[19](https://www.isca-archive.org/interspeech_2020/wang20da_interspeech.html)
[20](https://arxiv.org/abs/2303.13336)
[21](https://www.semanticscholar.org/paper/5a5bcfda3b753f8266b9ba27d34fc86b6d374a1b)
[22](https://www.isca-archive.org/interspeech_2022/liu22x_interspeech.html)
[23](https://www.mdpi.com/2076-3417/13/7/4237)
[24](https://ieeexplore.ieee.org/document/10038120/)
[25](https://www.isca-archive.org/interspeech_2020/paul20_interspeech.html)
[26](https://ieeexplore.ieee.org/document/9413880/)
[27](https://www.isca-archive.org/interspeech_2020/du20c_interspeech.html)
[28](https://dl.acm.org/doi/10.1145/3474085.3475437)
[29](https://ieeexplore.ieee.org/document/10664004/)
[30](https://www.mdpi.com/2078-2489/13/3/103)
[31](http://arxiv.org/pdf/2203.16852v2.pdf)
[32](https://arxiv.org/pdf/1905.09263.pdf)
[33](https://arxiv.org/pdf/2102.01991.pdf)
[34](https://arxiv.org/pdf/2203.10473.pdf)
[35](https://www.mdpi.com/1424-8220/23/7/3461/pdf?version=1679739050)
[36](http://arxiv.org/abs/2306.01442)
[37](https://arxiv.org/pdf/2501.13465.pdf)
[38](https://arxiv.org/pdf/2308.01018.pdf)
[39](http://papers.neurips.cc/paper/6889-deep-voice-2-multi-speaker-neural-text-to-speech.pdf)
[40](https://jaketae.github.io/study/glowtts/)
[41](https://arxiv.org/abs/2105.06337)
[42](https://sython.org/papers/APSIPA/nakai22apsipa.pdf)
[43](https://docs.coqui.ai/en/latest/_modules/TTS/tts/models/glow_tts.html)
[44](https://grad-tts.github.io)
[45](https://arxiv.org/pdf/2006.04558.pdf)
[46](https://coqui-tts.readthedocs.io/en/latest/models/glow_tts.html)
[47](https://proceedings.mlr.press/v139/popov21a/popov21a.pdf)
[48](https://ieeexplore.ieee.org/iel8/6287639/6514899/11080147.pdf)
[49](https://arxiv.org/pdf/2005.11129.pdf)
[50](https://aclanthology.org/2023.findings-acl.437)
[51](https://arxiv.org/abs/2410.12279)
[52](https://ieeexplore.ieee.org/document/10446467/)
[53](https://arxiv.org/abs/2402.10642)
[54](https://dl.acm.org/doi/10.1145/3581783.3612061)
[55](https://arxiv.org/abs/2309.12792)
[56](https://ieeexplore.ieee.org/document/10448502/)
[57](https://ieeexplore.ieee.org/document/10447822/)
[58](https://www.mdpi.com/1424-8220/23/23/9591)
[59](https://arxiv.org/abs/2201.11972)
[60](https://arxiv.org/pdf/2204.09934.pdf)
[61](https://aclanthology.org/2023.findings-acl.437.pdf)
[62](https://arxiv.org/pdf/2212.14518.pdf)
[63](https://arxiv.org/pdf/2305.06908.pdf)
[64](http://arxiv.org/pdf/2402.10642.pdf)
[65](http://arxiv.org/pdf/2410.11097.pdf)
[66](https://arxiv.org/pdf/2211.09707.pdf)
[67](https://arxiv.org/html/2509.18470v1)
[68](https://www.isca-archive.org/interspeech_2025/park25b_interspeech.pdf)
[69](https://huggingface.co/papers/2409.13910)
[70](https://developer.nvidia.com/blog/speeding-up-text-to-speech-diffusion-models-by-distillation/)
[71](https://aclanthology.org/2024.findings-naacl.240.pdf)
[72](https://research.google/blog/restoring-speaker-voices-with-zero-shot-cross-lingual-voice-transfer-for-tts/)
[73](https://aclanthology.org/2024.emnlp-main.9.pdf)
[74](https://arxiv.org/abs/2404.00569)
[75](https://google.github.io/tacotron/publications/zero_shot_voice_transfer/index.html)
[76](https://dl.acm.org/doi/10.1609/aaai.v37i11.26597)
[77](https://arxiv.org/pdf/2402.10642.pdf)
[78](https://randomsampling.tistory.com/514)
[79](https://www.isca-archive.org/interspeech_2023/tran23d_interspeech.html)
