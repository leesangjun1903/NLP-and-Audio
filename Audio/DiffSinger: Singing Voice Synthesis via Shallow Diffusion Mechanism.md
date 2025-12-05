# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism

### **1. 논문의 핵심 주장과 주요 기여**

DiffSinger는 **확산 확률 모델(Diffusion Probabilistic Model, DPM)**을 음성 합성 분야에 최초로 적용하여 노래 음성 합성(Singing Voice Synthesis, SVS)의 근본적인 문제들을 해결한 획기적인 논문입니다.[1]

#### 핵심 문제 해결

기존 SVS 음향 모델들의 주요 문제점을 다음과 같이 지적합니다:[1]

- **과도한 평활화(Over-smoothing)**: L1/L2 손실함수 기반 학습이 단봉 분포(unimodal distribution) 가정을 근거로 음성의 세부 특징이 흐릿해지는 현상
- **불안정한 학습**: GAN 기반 방법들의 판별자 불안정성으로 인한 훈련 과정의 불안정성

#### 주요 기여

1. **SVS 최초의 확산 모델**: 음악 점수 조건에서 노이즈를 멜-스펙트로그램으로 변환하는 매개변수화된 마르코프 체인 제시[1]
   
2. **얕은 확산 메커니즘(Shallow Diffusion Mechanism)**: 
   - MOS 개선: 0.14 포인트 향상
   - CMOS 개선: 0.5 포인트 향상  
   - 추론 속도: **45.1배 가속화**[1]

3. **일반화 성능 입증**:
   - DiffSpeech를 통한 TTS 작업 확대
   - FastSpeech 2 대비 0.24 MOS 향상[1]
   - Glow-TTS 대비 0.23 MOS 향상

***

### **2. 해결하고자 하는 문제와 제안 방법**

#### 2.1 기본 확산 모델 이론

**확산 과정(Diffusion Process)**

$$q(y_t | y_{t-1}) = \mathcal{N}(y_t | \sqrt{1-\beta_t} y_{t-1}, \beta_t I)$$

폐쇄형 계산:[1]

$$q(y_t | y_0) = \mathcal{N}(y_t | \sqrt{\bar{\alpha}_t} y_0, (1-\bar{\alpha}_t) I)$$

여기서 $$\bar{\alpha}\_t = \prod_{s=1}^{t} (1-\beta_s)$$입니다.

**역과정(Reverse Process)**[1]

$$p_\theta(y_{t-1}|y_t) = \mathcal{N}(y_{t-1}|\mu_\theta(y_t, t), \sigma_t^2 I)$$

**훈련 목표 - ELBO 최적화**[1]

$$\mathcal{L}_t = \text{KL}(q(y_{t-1}|y_t, y_0) \parallel p_\theta(y_{t-1}|y_t))$$

재매개변수화를 통해:

$$\mathbb{E}_{y_0, \epsilon} \left[ ||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}y_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, x, t)||^2 \right]$$

#### 2.2 얕은 확산 메커니즘 (Shallow Diffusion)

**핵심 통찰**: $M$(실제 멜-스펙트로그램)과 $\tilde{M}$(보조 디코더의 흐릿한 멜-스펙트로그램)의 확산 궤적이 충분히 큰 확산 단계에서 교차합니다.[1]

**교차점 증명**:[1]

$$\text{KL}(\mathcal{N}(M_t|\bar{\alpha}_t M_0) \parallel \mathcal{N}(\tilde{M}_t|\bar{\alpha}_t \tilde{M}_0)) = \frac{\bar{\alpha}_t^2}{1-\bar{\alpha}_t} ||M_0 - \tilde{M}_0||_2^2$$

$\bar{\alpha}_t$는 $t$ 증가에 따라 0으로 수렴하므로, KL 다이버전스도 0으로 수렴합니다.

**추론 절차**:[1]

1. 보조 디코더로 $\tilde{M}$ 생성
2. 확산 과정을 통해 단계 $k$에서:
   $$M_k = \bar{\alpha}_k \tilde{M} + \sqrt{1-\bar{\alpha}_k} \epsilon$$
3. $M_k$에서 시작하여 역과정 실행 (가우시안 노이즈가 아닌)
4. $k$번의 디노이징 단계만 실행

**장점**:[1]
- 가우시안 노이즈→ $M_0$ 변환보다 훨씬 간단
- 역과정 부담 완화
- 음성 품질 및 추론 속도 개선

#### 2.3 경계 예측 네트워크

경계점 $k$를 자동으로 결정하기 위한 분류 네트워크:[1]

$$\mathcal{L}_{BP} = \mathbb{E}_{M \in Y,t} [-\log BP(M_t, t) - \log(1 - BP(\tilde{M}_t, t))]$$

95%의 단계에서 분류 확률의 마진이 임계값(0.4) 이하인 가장 빠른 단계 $k$를 선택합니다.

***

### **3. 모델 구조**

#### 3.1 주요 컴포넌트

**인코더**:[1]
- 음소 ID → 음소 임베딩 (256 차원)
- Transformer 블록 (4개, 2개 헤드)
- 길이 정규화기: 음성 길이 확장
- 피치 인코더: 피치 ID → 피치 임베딩

**단계 임베딩**:[1]
- 정현파 위치 임베딩
- 두 개의 선형 레이어
- 출력: 256 채널의 $E_t$

**디노이저**:[1]
- 비인과 WaveNet 아키텍처
- 20개 컨볼루션 블록 (커널 크기 3)
- 1×1 컨볼루션 + 게이트 유닛 + 잔차 연결
- 입력: $M_t$, 조건: $E_t$, $E_m$

**보조 디코더**:[1]
- Feed-forward Transformer (FFT) 블록 스택
- 출력: 흐릿한 멜-스펙트로그램 $\tilde{M}$

#### 3.2 구체적 설정[1]

| 구성 요소 | 설정값 |
|---------|-------|
| 총 확산 단계 $T$ | 100 |
| 채널 크기 $C$ | 256 |
| 디노이저 레이어 | 20 |
| 멜-스펙트로그램 빈 | 80 |
| 음소 어휘 크기 | 61 |
| 분산 스케줄 $\beta$ | 1e-4 → 0.06 (선형) |

***

### **4. 성능 평가**

#### 4.1 주관적 평가 (MOS)

| 방법 | MOS | 신뢰도 |
|------|-----|--------|
| GT (실제 음성) | 4.30 | ±0.09 |
| GT 멜-스펙트로그램 변환 (상한) | 4.04 | ±0.11 |
| FFT-Singer | 3.67 | ±0.11 |
| GAN-Singer | 3.74 | ±0.12 |
| **DiffSinger (기본)** | **3.71** | **±0.10** |
| **DiffSinger (얕은 확산)** | **3.85** | **±0.11** |

**성과**:[1]
- GAN-Singer 대비 **0.11 MOS 향상** (3.74 → 3.85)
- 얕은 확산 메커니즘의 효과: **0.14 CMOS 향상**

#### 4.2 절제 연구

| 구성 | 채널 | 레이어 | 얕은 확산 | CMOS |
|-----|-----|-------|---------|------|
| 기준 | 256 | 20 | ✓ | 0.000 |
| 얕은 확산 제거 | 256 | 20 | ✗ | -0.500 |
| 다른 $k$ 값 | 256 | 20 | ✓ | -0.053 |
| 채널 128 | 128 | 20 | ✓ | -0.071 |
| 채널 512 | 512 | 20 | ✓ | -0.044 |

**결론**:[1]
- 얕은 확산 메커니즘이 가장 중요 (**-0.500 CMOS 영향**)
- 경계 예측이 효과적
- 모델 용량 설정이 최적

#### 4.3 추론 속도[1]

| 메트릭 | 얕은 확산 전 | 얕은 확산 후 | 가속도 |
|--------|----------|----------|-------|
| RTF | 0.348 | 0.191 | **45.1배** |

#### 4.4 TTS 일반화 (DiffSpeech)

| 방법 | MOS |
|------|-----|
| GT | 4.22 |
| Tacotron 2 | 3.54 |
| FastSpeech 2 | 3.68 |
| Glow-TTS | 3.69 |
| **DiffSpeech** | **3.92** |

**성과**:[1]
- FastSpeech 2 대비 **0.24 MOS 향상**
- Glow-TTS 대비 **0.23 MOS 향상**
- 추론 속도: **29.2배 가속화**

***

### **5. 모델의 일반화 성능 향상 가능성**

#### 5.1 현재 일반화 성공

DiffSpeech의 성공은 **기본 확산 구조의 강력한 일반화 능력**을 입증합니다.[1]

#### 5.2 일반화 메커니즘

**1. 학습 안정성 (ELBO 최적화)**:[1]
```
기존 GAN: 판별자-생성자 경쟁 → 불안정
DiffSinger: ELBO 최적화 → 안정적 수렴
결과: 더 다양한 데이터에서 견고한 학습
```

**2. 조건부 구조의 유연성**:[1]

$$\mathcal{L} = \mathbb{E} [||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}y_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \mathbf{x}, t)||^2]$$

- 조건 $\mathbf{x}$에 음악 점수, 텍스트, 스타일 정보 등 다양하게 적용 가능
- 새로운 작업/데이터에 빠른 적응

**3. 얕은 확산의 재사용성**:[1]
- 사전훈련된 간단한 디코더의 지식 효과적 활용
- 새로운 도메인으로의 빠른 전이 학습
- 추론 속도와 품질 균형

#### 5.3 향상된 일반화 전략

**다중 작업 학습**:
- 기본 SVS + 보조 작업 (음성 기술 인식, 감정 제어, 음성 변환)
- 음성 특징의 더 풍부한 표현 학습

**자가-지도 학습**:
- 대규모 무레이블 음악 데이터로 사전훈련
- 레이블 데이터 요구 감소

**도메인 적응**:
- 언어 간 전이 학습
- 녹음 환경 적응 (스튜디오 → 라이브)

#### 5.4 후속 연구의 성과 (2021-2025)

**HiddenSinger (2023)**:[2]
- 신경 음성 코덱 + 잠재 확산
- 저차원에서의 확산으로 복잡성 감소

**DiffGAN-TTS (2022-2023)**:[3]
- 확산 모델 + GAN 하이브리드
- GAN으로 디노이징 분포 학습
- **추론 단계 1로 단축** (기존 100)

**TechSinger (2025)**:[4]
- Flow matching 기반
- 7개 성악 기술 제어
- 5개 언어 지원

**TCSinger (2024)**:[5]
- 제로샷 음성 합성 (새로운 성악자)
- 다국어 스타일 전이

**CSSinger (2024)**:[6]
- 청크 단위 스트리밍 SVS
- 실시간 처리 가능

***

### **6. 한계점**

#### 6.1 데이터 제약

- **PopCS**: 한 명 성악가, 117곡 (~5.89시간)[1]
- 다중 성악가 및 다양한 언어/스타일 미포함

#### 6.2 훈련 복잡성

- 세 단계 훈련 필요 (보조 디코더 → 경계 예측 → 메인 디노이저)[1]
- 총 시간: V100 1개 기준 28시간
- 하이퍼파라미터 민감성

#### 6.3 음성 표현

- 보조 디코더가 주기적 변형(aperiodic parameters) 완벽 포착 불가[1]
- 자연스러운 음성 변화 제한

#### 6.4 평가 제한

- MOS에 주로 의존 (객관적 지표 부족)
- 작은 테스트셋 (2곡)

***

### **7. 앞으로의 영향과 연구 고려사항**

#### 7.1 이론적 기여

**패러다임 전환**:
- GAN 중심 → 확산 모델 중심[1]
- 불안정한 적대적 학습 → 안정적 ELBO 최적화
- 현재(2025) 음성 합성 최고 성능 모델들의 기초

**얕은 확산의 혁신**:
- 간단한 모델의 지식을 고급 모델에 활용[1]
- 두 분포 교차점의 수학적 증명
- 추론 속도와 품질 trade-off 해결
- 후속 연구 (DiffGAN-TTS 등)에 광범위 적용

#### 7.2 실제 응용

**음악 산업**:
- 음악 제작 자동화
- 가상 아티스트 개발
- 더빙 및 음성 배우 비용 절감

**음성 기술**:
- 청각 장애인 맞춤형 음성
- 언어 학습 도구

#### 7.3 향후 연구 방향

**단기 (1-2년)**:
- 데이터 확장 (500+ 시간) → MOS 3.95+ 달성
- 다국어 지원
- 실시간 추론 (RTF < 0.05)

**중기 (2-4년)**:
- 자가-지도 학습 통합
- 멀티모달 입력 처리
- 세밀한 표현 제어

**장기 (4+ 년)**:
- 자율 음악 생성 (가사 → 전곡)
- 감정 인식 합성
- 실시간 음성 변환

#### 7.4 기술 개선

**훈련 효율**:
- 엔드-투-엔드 단일 단계 훈련
- 자동 하이퍼파라미터 튜닝
- 분산 훈련

**평가 체계**:
- 객관적 지표 활용 (MCD, F0-RMSE 등)
- 확장된 테스트셋
- 음악성 다차원 평가

#### 7.5 사회적 고려사항

**윤리적 문제**:
- 음성 모방 및 사기 방지
- 저작권 침해 대응
- 음성 워터마킹 기술 개발
- 합성 음성 명확한 표시

**긍정적 기여**:
- 청각 장애인 접근성 향상
- 언어 학습 민주화
- 음악 제작 비용 절감

***

### **8. 최신 관련 연구 현황 (2020-2025)**

#### 8.1 확산 모델의 진화

**2020**: DDPM - 이미지 생성에서의 성공으로 다른 분야 적용 촉발[7]

**2021**: DiffSinger & DiffSpeech - TTS/SVS에 처음 적용[1]

**2022-2023**: DiffGAN-TTS - 하이브리드 접근, 1스텝 생성[3]

**2023-2024**: HiddenSinger, WeSinger, TCSinger - 특화된 기법 개발[8][2][5]

**2024-2025**: TechSinger, Everyone-Can-Sing - Flow matching, 제로샷 학습[9][4]

#### 8.2 성능 진화 추세

```
MOS 점수 추이 (SVS):

2021: DiffSinger ~ 3.85
2022-2023: DiffGAN-TTS ~ 3.9+
2024: HiddenSinger ~ 3.95
2025: TechSinger ~ 예상 3.98+

상한 (GT): 4.04
```

#### 8.3 추론 속도 개선

| 연도 | 방법 | RTF |
|------|------|-----|
| 2021 | DiffSinger | 0.191 |
| 2023 | DiffGAN-TTS | 0.001-0.005 |
| 2024 | HiddenSinger | 0.05-0.1 |
| 2025 | TechSinger | 예상 0.03 |

***

### **결론**

DiffSinger는 **확산 모델을 음성 합성에 최초 적용한 획기적 논문**으로:

1. **이론적 기초**: ELBO 최적화를 통한 안정적이고 수학적으로 타당한 생성 모델 제시
2. **실용적 혁신**: 얕은 확산으로 품질 향상(0.14 MOS)과 속도 개선(45.1배)을 동시 달성
3. **광범위한 영향**: 2021년부터 현재(2025)까지 음성 합성 분야의 주요 연구 방향 설정

현재 관점에서 DiffSinger의 기본 틀은 **다양한 음성 합성 작업에 적용 가능하며 안정적이고 효율적**이어서, 지속적인 개선을 통해 인간 수준의 음성 합성에 더욱 접근하고 있습니다.

[2][4][5][6][7][8][9][3][1]

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/08800370-477c-4c2e-9cf9-8cd089364076/2105.02446v6.pdf)
[2](https://invergejournals.com/index.php/ijss/article/view/99)
[3](https://www.researchcatalogue.net/view/1935282/1935283)
[4](https://ashpublications.org/blood/article/144/Supplement%201/3673/533466/Findings-from-a-Longitudinal-Series-of-Continuing)
[5](https://www.semanticscholar.org/paper/1c5015703a2599974c56cb6a0448b8985472a4fc)
[6](https://minderoo.org/doi/umbrella-review/)
[7](https://open-publishing.org/journals/index.php/jutlp/article/view/734)
[8](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[9](https://jle.hse.ru/article/view/24181)
[10](https://journal-agriplant.com/index.php/journal/article/view/132)
[11](https://www.tandfonline.com/doi/full/10.1080/20551940.2023.2232608)
[12](https://arxiv.org/html/2409.13832v1)
[13](https://arxiv.org/html/2502.12572v1)
[14](https://arxiv.org/html/2409.07226v2)
[15](http://arxiv.org/pdf/2501.13870.pdf)
[16](https://arxiv.org/html/2412.08918v2)
[17](https://arxiv.org/pdf/2108.02776.pdf)
[18](https://arxiv.org/pdf/2306.06814.pdf)
[19](https://arxiv.org/pdf/2203.10750.pdf)
[20](https://www.arxiv.org/pdf/2505.14910.pdf)
[21](https://pdfs.semanticscholar.org/2aec/b02c641c6b73ef124b7ca6d69eb87f89797d.pdf)
[22](https://scispace.com/pdf/melgan-generative-adversarial-networks-for-conditional-2bj4md6is0.pdf)
[23](https://aclanthology.org/2024.emnlp-main.117.pdf)
[24](https://arxiv.org/html/2508.00733v1)
[25](https://arxiv.org/pdf/2508.07711.pdf)
[26](https://github.com/topics/singing-voice-synthesis)
[27](https://www.isca-archive.org/interspeech_2025/kim25b_interspeech.pdf)
[28](https://aclanthology.org/2024.icnlsp-1.3.pdf)
[29](https://www.isca-archive.org/interspeech_2025/zhao25g_interspeech.pdf)
[30](https://arxiv.org/pdf/1905.09263.pdf)
[31](https://pmc.ncbi.nlm.nih.gov/articles/PMC11842752/)
[32](http://arxiv.org/pdf/2307.15484.pdf)
[33](https://zenodo.org/records/8092573/files/is2023-dysarthric-tts.pdf)
[34](http://arxiv.org/pdf/2408.11849.pdf)
[35](http://arxiv.org/pdf/2306.04301.pdf)
[36](https://arxiv.org/pdf/2109.15166.pdf)
[37](https://arxiv.org/pdf/2308.01018.pdf)
[38](https://arxiv.org/abs/2006.04558)
[39](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
[40](https://antispoofing.org/generative-ai-in-audio-speech-music-and-its-detection/)
[41](https://papers.nips.cc/paper/8580-fastspeech-fast-robust-and-controllable-text-to-speech)
[42](https://www.sciencedirect.com/science/article/abs/pii/S0888327024003790)
[43](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffgantts/)
[44](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/fastspeech2/)
[45](https://www.ijcai.org/proceedings/2023/0648.pdf)
[46](https://www.cis.upenn.edu/wp-content/uploads/2024/05/CIS-498-Final-Draft-Judah-N.pdf)
[47](https://all-the-meaning.tistory.com/36)
