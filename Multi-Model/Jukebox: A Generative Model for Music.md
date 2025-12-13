# Jukebox: A Generative Model for Music

### 1. 핵심 주장과 주요 기여

**Jukebox**는 OpenAI가 2020년에 발표한 혁신적인 음악 생성 모델로, 다음과 같은 핵심 주장과 기여를 제시합니다:[1]

**핵심 주장**: 계층적 VQ-VAE(Vector Quantized Variational Autoencoder)와 자동회귀 Transformer를 결합하여 원본 오디오 도메인에서 고충실도의 다양한 음악을 여러 분 길이로 일관성 있게 생성할 수 있다는 것입니다.

**주요 기여**:
- **원본 오디오 생성**: 심볼릭(MIDI) 표현이 아닌 원본 오디오 파형에서 직접 음악을 생성하는 첫 번째 대규모 시스템 구현
- **장시간 일관성**: 기존 20-30초 수준의 음악 생성에서 벗어나 수분 길이의 음악을 생성 가능
- **다중 컨디셔닝**: 아티스트, 장르, 가사 등 다양한 조건으로 음악 생성 제어 가능
- **오픈 소스 공개**: 모델 가중치, 훈련 코드, 샘플링 코드를 모두 공개하여 커뮤니티 활용 촉진

***

### 2. 상세 기술 분석

#### 2.1 해결하고자 하는 문제

음악 생성의 근본적인 도전 과제들:[1]

1. **원본 오디오의 높은 차원성**: 44.1kHz 샘플링 레이트의 4분 곡은 약 1,000만 개의 샘플을 포함하고 있으며, 각 위치마다 16비트 정보를 담고 있음
2. **장거리 의존성**: 음악의 전체 구조와 일관성을 유지하려면 극도로 긴 시간 범위의 의존성을 모델링해야 함
3. **다중 시간 척도**: 음색(timbre)부터 전체 구성(global coherence)까지 여러 시간 척도에서의 음악 구조 포착의 필요성
4. **고품질과 다양성의 균형**: 높은 충실도 유지 와중에도 다양한 음악 스타일 생성

#### 2.2 제안하는 방법론

**계층적 아키텍처**:

Jukebox는 세 단계의 VQ-VAE를 사용하여 음악을 압축합니다:[1]

$$L = L_{recons} + L_{codebook} + \beta L_{commit}$$

여기서:
- $$L_{recons} = \frac{1}{T}\sum_t \|x_t - D(e_{z_t})\|_2^2$$ (재구성 손실)
- $$L_{codebook} = \frac{1}{S}\sum_s \|sg[h_s] - e_{z_s}\|_2^2$$ (코드북 손실)
- $$L_{commit} = \frac{1}{S}\sum_s \|h_s - sg[e_{z_s}]\|_2^2$$ (커밋 손실)

여기서 $$sg$$는 기울기 중단(stop-gradient) 연산입니다.[1]

**VQ-VAE 개선사항**:

1. **무작위 재시작 (Random Restarts)**: 코드북 붕괴(codebook collapse) 현상을 방지
   - 코드북 벡터의 평균 사용도가 임계값 아래로 떨어지면 무작위로 재초기화

2. **분리된 오토인코더**: 계층적 VQ-VAE 대신 다양한 홉 길이(hop length)를 가진 별도의 오토인코더 사용
   - 상위 레벨의 붕괴 문제 해결

3. **스펙트럼 손실 추가**:

$$L_{spec} = \||\text{STFT}(x)| - |\text{STFT}(\hat{x})|\|_2$$

- STFT 파라미터를 다양하게 변경하여 다중 스펙트럼 손실을 계산하여 과적합 방지[1]

**선행(Prior) 모델과 업샘플러**:

세 레벨의 VQ-VAE 코드에 대해 자동회귀 모델링을 수행:

$$p(z) = p(z_{top})p(z_{middle}|z_{top})p(z_{bottom}|z_{middle}, z_{top})$$

각 레벨에서 Sparse Transformer를 사용하여 최대 가능도(maximum likelihood estimation)로 훈련합니다.[1]

**확장 가능한 Transformer**:

논문에서 제안한 단순화된 Sparse Transformer는:[1]
- 축 정렬 주의 패턴(axis-aligned attention patterns): 마스크된 행 주의(row attention), 마스크된 열 주의(column attention), 마스크되지 않은 이전 행 주의(previous-row attention)
- 반정밀도(half-precision, fp16) 파라미터와 동적 스케일링으로 메모리 최적화

#### 2.3 컨디셔닝 메커니즘

**아티스트, 장르, 타이밍 컨디셔닝**:[1]
- 아티스트와 장르 임베딩을 학습하여 시퀀스의 첫 번째 토큰으로 제공
- 곡의 총 지속 시간, 시작 시간, 경과 비율을 나타내는 타이밍 신호로 음악 구조 제어

**가사 컨디셔닝 (Lyrics-to-Singing, LTS)**:[1]
- 인코더-디코더 Transformer 구조: 가사 텍스트를 인코딩하고 음악 디코더가 주의(attention)를 통해 가사에 조건화
- Spleeter와 NUS AutoLyricsAlign을 사용하여 가사-음성 정렬을 개선

**디코더 사전훈련**:
- 사전훈련된 무조건 최상위 선행 모델을 사용하여 가사 인코더를 추가
- 출력 프로젝션 가중치를 영으로 초기화하여 모델 수술(model surgery) 구현[1]

#### 2.4 모델 구조 세부사항

**VQ-VAE 아키텍처**:

세 단계 계층적 구조:[1]
- **하위 레벨**: 홉 길이 8 (약 1.5초 원본 오디오)
- **중간 레벨**: 홉 길이 32 (약 6초 원본 오디오)
- **상위 레벨**: 홉 길이 128 (약 24초 원본 오디오)

각 단계는 비인과적 1D 확장 합성곱(dilated convolution)을 사용하며, 코드북 크기는 각 레벨에서 2048입니다.[1]

**선행 및 업샘플러 모델 사양**:[1]

| 모델 | 파라미터 | 훈련 기간 | 계산 리소스 |
|------|---------|---------|-----------|
| 1B 업샘플러 | 10억 | 2주 | 128 V100 |
| 5B 최상위 선행 | 50억 | 4주 | 512 V100 |

***

### 3. 성능 향상 및 한계

#### 3.1 성능 향상

**일관성 (Coherence)**:[1]
- 최상위 선행의 컨텍스트 길이(약 24초)에서 뛰어난 일관성 유지
- 윈도우 샘플링으로 임의 길이 음악 생성 가능

**음악성 (Musicality)**:[1]
- 친숙한 음악 화음과 자연스러운 가사 배치 학습
- 인간 가수가 강조할 단어에서 가장 높거나 긴 음표 자동 생성

**다양성 (Diversity)**:[1]
- 동일한 아티스트-가사 조합에 대해 여러 샘플 생성 시 서로 다른 음악 스타일 창출
- 기존 곡의 프라이밍(priming)에도 약 30초 후부터 새로운 음악 재료로 이탈

**신규성 (Novelty)**:[1]
- 새로운 스타일 생성: 특정 아티스트를 특이한 장르와 결합하여 해석
- 새로운 목소리: 아티스트 임베딩 보간으로 새로운 음성 합성
- 새로운 가사: 훈련 분포 밖의 새로운 가사에 대응 가능

**VQ-VAE 재구성 충실도**:[1]

표 1의 스펙트럼 수렴(Spectral Convergence) 결과:

| 레벨 | 홉 길이 | 재시작 없음 | 재시작 포함 |
|------|--------|-----------|-----------|
| 하위 | 8 | -21.1 dB | -23.0 dB |
| 중간 | 32 | -12.4 dB | -12.4 dB |
| 상위 | 128 | -8.3 dB | -8.3 dB |

#### 3.2 한계와 제약사항

**단거리 품질 문제**:[1]
- 최상위 선행이 전체 곡의 맥락을 갖지 않아 반복되는 후렴구(chorus)나 질문-응답 구조의 멜로디 부재
- 음향 노이즈나 스크래칠 현상 때때로 관찰

**언어 제약**:[1]
- 영어 주요 곡(600만 곡)으로만 훈련되어 다국어 음악 생성 불가
- 특정 장르(예: 힙합)에서 가사 정렬 실패로 인한 낮은 합창 생성 확률

**생성 속도**:[1]
- 최상위 레벨 토큰 1분 생성: 약 1시간
- 업샘플링: 1분 음악당 약 8시간 소요

**임베딩 일반화**:[1]
- 아티스트 임베딩이 목소리 특성을 강하게 지배하여 새로운 스타일 생성 시 원래 스타일이 지배적
- 예: Alan Jackson의 목소리를 힙합/펑크와 조합 시 여전히 컨트리 스타일 유지

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 현재 성능의 주요 제약 분석

Jukebox의 일반화 성능은 여러 차원에서 분석할 수 있습니다:[1]

**데이터 다양성 문제**:
- 훈련 데이터 불균형: 영어 600만 곡 vs 비영어 600만 곡
- 장르별 학습 편향: 대중음악 중심의 훈련 데이터 구성

**아키텍처의 제약**:
- 최상위 선행의 제한된 컨텍스트 길이(약 24초) → 긴 악곡 구조 학습 불가능
- 가사 정렬의 불완전성: 선형 정렬 휴리스틱은 빠른 가사(예: 랩)에 실패

#### 4.2 향상 방안

**데이터 관점에서의 개선**:
1. **다국어/다문화 확장**: 비영어 음악 데이터셋 비율 증가
2. **장르 균형화**: 언더리프리젠트된 장르(클래식, 월드 뮤직 등)의 샘플 보강
3. **고품질 데이터 선별**: 합성된 또는 저품질 음악 제거를 통한 훈련 신호 개선

**모델 아키텍처 개선**:
1. **더 깊은 계층 구조**: 현재 3레벨에서 4-5레벨로 확대하여 다양한 시간 척도 포착
2. **변수 길이 컨텍스트**: 동적 컨텍스트 길이 조정으로 전체 곡 일관성 학습
3. **멀티태스크 학습**: 음악 생성과 음악 분류, 음악 변환 등을 결합한 멀티태스크 훈련

**컨디셔닝 메커니즘 개선**:
1. **세밀한 감정 컨디셔닝**: 현재 아티스트/장르에서 감정, 에너지 수준, 템포 등으로 확장
2. **악기별 제어**: 개별 악기의 존재/부재를 명시적으로 제어 가능하도록
3. **동적 가사 정렬**: 고정 선형 정렬 대신 학습 가능한 정렬 메커니즘 도입

**훈련 전략 개선**:
1. **증분 훈련(Curriculum Learning)**: 짧은 음악부터 시작하여 점진적으로 긴 음악으로 전환
2. **도메인 적응(Domain Adaptation)**: 새로운 언어/장르에 효율적으로 적응하기 위한 전이 학습
3. **약한 감독(Weak Supervision)**: 완벽한 가사 정렬 대신 약한 라벨 활용

#### 4.3 VQ-VAE 기반 아키텍처의 일반화 한계

**벡터 양자화의 병목**:[1]
- 코드북 크기가 고정되면 정보 용량 제약
- 해결책: 적응형 코드북 크기 또는 더 큰 코드북 사용

**계층적 정보 흐름**:
- 상위 레벨의 정보 손실이 하위 레벨 생성에 영향
- 해결책: 유산 연결(skip connections) 또는 잔여 학습(residual learning)으로 정보 손실 최소화

***

### 5. 해당 논문이 미치는 영향과 앞으로의 연구 고려 사항

#### 5.1 음악 생성 분야에 미친 영향

**패러다임 전환**:[2][1]

Jukebox는 음악 생성을 심볼릭 표현에서 **원본 오디오 도메인으로 전환**하는 데 결정적 역할을 했습니다. 이는 다음과 같은 영향을 미쳤습니다:

1. **고충실도 생성 가능성 입증**: 원본 오디오에서도 고품질 음악 생성이 가능함을 증명
2. **다중 시간 척도 모델링**: VQ-VAE의 계층적 구조가 다양한 시간 척도를 효과적으로 처리할 수 있음을 보여줌
3. **컨디셔닝의 유연성**: 가사, 아티스트, 장르 등 다양한 방식의 제어 가능성 제시

#### 5.2 후속 연구의 진화 방향

**2020년 이후 관련 최신 연구 비교 분석**

| 연구 | 년도 | 주요 기법 | 특징 | 성과 |
|------|------|---------|------|------|
| **AudioLM** (Google)[3][4] | 2022 | 계층적 토큰화 + Transformer | 음성 계속 및 피아노 음악 생성 | 장기 일관성 우수, 음성 정체성 유지 |
| **MusicGen** (Meta)[5][6][7] | 2023 | EnCodec + 단일 단계 Transformer | 텍스트 조건부 음악 생성 | Jukebox보다 빠르고 효율적, 12초 생성 |
| **MusicFlow**[8][9][10] | 2024 | Flow Matching + 계단식 구조 | 텍스트 유도 음악 생성 | 더 나은 텍스트 정렬, 더 빠른 생성 |
| **Multi-Track MusicLDM**[11] | 2024 | 잠재 확산 모델 | 다중 트랙 생성 | 트랙 간 일관성 향상 |
| **MuMu-LLaMA**[12] | 2024 | 멀티모달 LLM + MusicGen | 텍스트/이미지/비디오-음악 | 다중 모드 입력 처리 |
| **MusicGen-Chord**[13] | 2024 | MusicGen + 화음 조건화 | 화음 진행 제어 | 음악 구조 세밀한 제어 |
| **DiffRhythm**[14] | 2025 | 단계 확산 + 자동회귀 | 전체 길이 곡 생성 | 더 긴 형식, 일관된 리듬 |
| **MR-FlowDPO**[15] | 2025 | Flow Matching + 직접 선호도 최적화 | 다중 보상 기반 정렬 | 인간 선호도와의 정렬 개선 |

**주요 기술적 진화**:

1. **생성 패러다임의 변화**:[8][16][2]
   - Jukebox의 자동회귀(Autoregressive) 방식에서 확산(Diffusion), 플로우 매칭(Flow Matching)으로 진화
   - 생성 속도 개선: Jukebox의 1분당 1시간에서 실시간 생성 수준으로 개선

2. **단계화 구조의 개선**:[11][17][2]
   - 계층적 토큰화에서 단일 단계 생성(MusicGen)으로 단순화
   - 최근 다시 계단식 구조로 복귀하여 품질 향상(Multi-Track MusicLDM)

3. **조건화 메커니즘의 고도화**:[18][12][19]
   - 단순 텍스트 조건에서 멀티모달 입력(이미지, 비디오, 음악 분석)으로 확장
   - 화음, 리듬 등 음악 특화 조건화 도입

4. **데이터 및 평가 개선**:[20][21]
   - 더 큰 규모의 라이센스 음악 데이터셋 구축
   - 음악 특화 평가 지표(음악성, 텍스트 정렬도, 리듬 안정성) 개발

#### 5.3 앞으로의 연구 시 고려할 점

**1. 인간-AI 협업 메커니즘**:[22]
- 전문 음악가뿐 아니라 초보자도 효과적으로 이용할 수 있는 인터페이스 개발 필요
- 음성 차선(Voice Lanes), 의미적 슬라이더(Semantic Sliders) 등의 제어 도구 개선

**2. 저작권 및 윤리 문제**:[20]
- 훈련 데이터의 출처 추적 및 크레딧 부여 메커니즘
- 오리지널 아티스트의 음성/스타일 모방에 대한 규범 개발
- 불균형 데이터의 편향성 완화

**3. 다국어/다문화 음악 생성**:[23][1]
- 현재 영어 위주의 한계 극복
- 비서방 음악 전통(인도 고전음악, 중국 전통음악 등)의 특수성을 반영한 모델 개발

**4. 긴 형식 음악 생성**:[14][24][1]
- 완전한 곡(예: 3-5분 길이)의 전체적 일관성 유지
- 음악 구조(도입부-절-후렴구-아웃트로)의 자동 학습

**5. 실시간 생성 및 인터랙티브 시스템**:[25][1]
- 사용자 입력에 대한 즉각적 반응
- 협업 작곡 환경에서 인간 뮤지션과 실시간 상호작용

**6. 음악적 표현성의 향상**:[26][23]
- 감정, 에너지, 템포, 악기 배치 등의 세밀한 제어
- 음악 이론적 제약(조성, 음정, 화음 진행)의 명시적 모델링

**7. 평가 지표의 개선**:[21][27][2]
- 주관적 평가(음악성, 창의성)의 객관화 방법 개발
- 음악 전문가와 일반 청취자의 선호도 차이 분석

**8. 성능 확장성 분석**:[28][29]
- 더 큰 모델 크기와 훈련 데이터의 영향 분석
- 컴퓨팅 효율성 향상을 통한 접근성 확대

**9. 도메인 특화 적응**:[2][26]
- 게임 음악, 영화 점수, 광고 음악 등 특정 분야를 위한 전이 학습
- 음악 요법, 교육 등 특수한 응용 분야의 요구사항 반영

**10. 모델 투명성과 해석성**:[30]
- 생성된 음악의 특정 특성(예: 악기 선택, 멜로디 패턴)이 어떻게 결정되는지 설명 가능성 강화
- 활성화 개입(activation steering)을 통한 세밀한 제어 메커니즘 개발

***

### 결론

Jukebox는 원본 오디오 도메인에서 높은 충실도의 음악을 생성하고 다양한 조건으로 제어 가능함을 처음 대규모로 입증한 획기적 연구입니다. 계층적 VQ-VAE와 자동회귀 Transformer의 조합은 여러 시간 척도에서의 음악 구조를 효과적으로 모델링하는 방법을 제시했습니다.

그러나 생성 속도, 언어 다양성, 전체 곡 길이의 장기 일관성 등에서의 한계는 명확합니다. 이러한 한계는 이후 MusicGen, MusicFlow, DiffRhythm 등의 후속 연구들이 개선하려던 목표였습니다.

앞으로의 음악 생성 연구는 **단순한 품질 향상을 넘어** (1) 다국어/다문화 음악 표현, (2) 음악 이론적 제약의 명시적 모델링, (3) 인간-AI 협업의 직관적 인터페이스, (4) 윤리적 고려사항의 통합 등으로 진화해야 할 것으로 보입니다. 특히 실제 음악 제작 워크플로우와의 통합과 음악 전문가의 요구사항을 반영한 설계는 학술적 성과를 실무적 가치로 전환하는 핵심이 될 것입니다.

***

**참고 문헌**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7d8e8d12-7f37-4ec3-8fb9-b7fdf9963d64/2005.00341v1.pdf)
[2](https://ieeexplore.ieee.org/document/10845168/)
[3](https://research.google/blog/audiolm-a-language-modeling-approach-to-audio-generation/)
[4](https://arxiv.org/pdf/2209.03143.pdf)
[5](https://techcrunch.com/2023/06/12/meta-open-sources-an-ai-powered-music-generator/)
[6](https://www.audiocipher.com/post/meta-musicgen)
[7](https://arxiv.org/abs/2306.05284)
[8](http://arxiv.org/pdf/2410.20478.pdf)
[9](https://musicflowicml.github.io)
[10](https://arxiv.org/abs/2410.20478)
[11](https://arxiv.org/html/2409.02845v2)
[12](https://arxiv.org/html/2412.06660v1)
[13](https://arxiv.org/pdf/2412.00325.pdf)
[14](https://arxiv.org/html/2503.01183v1)
[15](https://arxiv.org/abs/2512.10264)
[16](https://arxiv.org/abs/2506.08570)
[17](https://arxiv.org/pdf/2306.05284.pdf)
[18](https://arxiv.org/html/2311.11255v3)
[19](https://arxiv.org/html/2407.15060v1)
[20](https://arxiv.org/abs/2506.18312)
[21](https://dl.acm.org/doi/10.1145/3769106)
[22](https://dl.acm.org/doi/10.1145/3313831.3376739)
[23](https://www.ewadirect.com/proceedings/tns/article/view/30045)
[24](http://arxiv.org/pdf/2503.00084.pdf)
[25](https://www.emergentmind.com/topics/flow-matching-based-jam)
[26](https://www.ewadirect.com/proceedings/ace/article/view/23588)
[27](https://arxiv.org/pdf/2511.13936.pdf)
[28](https://www.frontiersin.org/articles/10.3389/fdgth.2025.1653369/full)
[29](https://arxiv.org/html/2509.14912v1)
[30](https://arxiv.org/html/2506.10225v1)
[31](https://arxiv.org/abs/2506.10927)
[32](https://www.semanticscholar.org/paper/02391e9532629bed0e37c2e4cb2d6ee6fef41f32)
[33](https://arxiv.org/abs/2502.12489)
[34](https://journals.gaftim.com/index.php/ijtim/article/view/525)
[35](https://arxiv.org/pdf/2210.13944.pdf)
[36](https://arxiv.org/pdf/2312.08723.pdf)
[37](https://www.ijfmr.com/papers/2024/3/20239.pdf)
[38](http://arxiv.org/pdf/2501.09972.pdf)
[39](https://www.cometapi.com/best-3-ai-music-generation-models-of-2025/)
[40](https://arxiv.org/pdf/2110.04005.pdf)
[41](https://arxiv.org/html/2507.20128v1)
[42](https://kingy.ai/blog/the-future-of-music-creation-the-best-ai-music-generators-of-2025/)
[43](https://www.reddit.com/r/MLQuestions/comments/1mpdk1n/unconditional_music_generation_using_a_vqvae_and/)
[44](https://research.samsung.com/blog/Diffusion-based-Text-to-Music-Generation-with-Global-and-Local-Text-based-Conditioning)
[45](https://www.sciencedirect.com/science/article/abs/pii/S0952197625021396)
[46](https://arxiv.org/abs/2509.11976)
[47](https://arxiv.org/abs/2302.03917)
[48](https://arxiv.org/html/2510.24990)
[49](https://arxiv.org/pdf/2509.11976.pdf)
[50](https://arxiv.org/pdf/2503.08565.pdf)
[51](https://arxiv.org/html/2509.11898v2)
[52](https://pdfs.semanticscholar.org/45e4/8c9b14d8788584cc55a92afd1d99e561835d.pdf)
[53](https://arxiv.org/html/2509.11898v1)
[54](https://arxiv.org/html/2512.01537v1)
[55](https://arxiv.org/pdf/2309.02057.pdf)
[56](https://koreascience.kr/article/JAKO202412043241400.page)
[57](https://onlinelibrary.wiley.com/doi/10.1002/cl2.1355)
[58](https://www.semanticscholar.org/paper/06ca869b5e1d3904a7bbb1bc2fadfd0e51068ddc)
[59](https://ieeexplore.ieee.org/document/10570408/)
[60](https://ieeexplore.ieee.org/document/10544569/)
[61](https://oaskpublishers.com/assets/article-pdf/dementia-disorders-a-narrative-review.pdf)
[62](https://arxiv.org/html/2501.08809v1)
[63](http://arxiv.org/pdf/2310.19180.pdf)
[64](https://www.marktechpost.com/2022/10/09/this-google-ais-new-audio-generation-framework-audiolm-learns-to-generate-realistic-speech-and-piano-music-by-listening-to-audio-only/)
[65](https://the-decoder.com/google-shows-generative-ai-model-for-speech-and-music/)
[66](https://www.reddit.com/r/ArtificialInteligence/comments/145d0kr/meta_just_released_musicgen/)
[67](https://musicgen.com)
[68](https://arxiv.org/abs/2209.03143)
[69](https://arxiv.org/html/2411.05679v3)
[70](https://arxiv.org/abs/2504.13535)
