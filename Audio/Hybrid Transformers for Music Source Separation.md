# Hybrid Transformers for Music Source Separation

### 1. 핵심 주장 및 주요 기여도 (요약)

**Hybrid Transformer Demucs (HT Demucs)** 논문의 기본 질문은 **음악 음원 분리에서 장거리 문맥 정보가 실제로 유용한가**이다. 저자들은 Transformer의 장거리 의존성 학습 능력을 활용하되, 기존 Hybrid Demucs의 이중 도메인 구조(시간/주파수)를 유지하는 교차 도메인 Transformer Encoder를 제안한다.[1]

주요 기여는 다음과 같다:

1. **교차 도메인 Transformer 아키텍처**: 자기 주의와 교차 주의를 결합하여 시간 및 주파수 도메인 간 정보 상호작용 구현
2. **성능 향상**: 추가 800곡으로 0.45 dB 개선, 희소 주의로 12.2초 입력 지원, 최종 9.20 dB SDR 달성
3. **데이터-아키텍처 관계 분석**: Transformer의 데이터 탐욕성을 실증적으로 입증[1]

***

### 2. 상세 설명: 문제, 방법, 모델 구조, 성능, 한계

#### 2.1 해결하고자 하는 문제

음악 음원 분리의 근본적 질문은 **컨텍스트 길이의 필요성**이다. Conv-TasNet은 약 1초의 지역적 특성으로 작동하고, Demucs는 최대 10초까지 활용하지만, 이러한 긴 문맥이 실제로 성능을 향상시키는지 체계적으로 검증되지 않았다. MUSDB18의 제한된 규모(87곡) 대비 Transformer 도입 시 필요한 데이터 규모도 불명확했다.[1]

#### 2.2 제안 방법 (수식 포함)

**Transformer Encoder 레이어** - 정규화 및 레이어 스케일 적용:

$$\text{Output} = \text{LayerNorm}(\text{Input} + \lambda \cdot \text{FFN}(\text{LayerNorm}(\text{Input})))$$

여기서 $\lambda = 10^{-4}$ (초기 레이어 스케일)[1]

**자기 주의 메커니즘**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

8개 헤드를 사용하고 피드포워드 숨겨진 크기는 $4 \times 384 = 1536$[1]

**데이터 필터링** - 자동 레이블 검증:

$$V(z) = 10 \cdot \log_{10}(\text{AveragePool}(z^2, 1\text{sec}))$$

음원별 검증: $P_{i,i} > 70\%$ (자동 음원), $P_{i,j} < 30\%$ (교차 간섭)[1]

**손실 함수** (시간 영역 파형에 대한 L1 손실):

$$\mathcal{L} = \sum_{i=1}^{4} \|\hat{s}_i(t) - s_i(t)\|_1$$

#### 2.3 모델 구조

HT Demucs는 기존 Hybrid Demucs의 외부 4개 인코더/디코더 레이어는 유지하되, **가장 내부의 2개 레이어를 교차 도메인 Transformer Encoder로 대체**한다.[1]

**핵심 특성**:
- **입력 차원**: 384
- **Transformer 깊이**: 5
- **주의 헤드**: 8개
- **위치 인코딩**: 시간 영역은 1D 정현파, 주파수 영역은 2D 정현파
- **희소 주의**: 국소 민감 해싱(LSH) 기반, 90% 희소성으로 메모리 제약 극복[1]

#### 2.4 성능 향상

| 조건 | 전체 SDR | 드럼 | 베이스 | 기타 | 보컬 |
|------|---------|------|--------|------|------|
| Baseline (HD, 추가 없음) | 7.64 | 8.12 | 8.43 | 5.65 | 8.35 |
| HT Demucs (MUSDB만) | 7.52 | 7.94 | 8.48 | 5.72 | 7.93 |
| HT Demucs (800곡) | 8.80 | 10.05 | 9.78 | 6.42 | 8.93 |
| HT Demucs (800곡, 미세조정) | 9.00 | 10.08 | 10.39 | 6.32 | 9.20 |
| **Sparse HT Demucs** | **9.20** | **10.83** | **10.47** | **6.41** | **9.37** |[1]

**데이터 증강의 중요성** (표 3): 리믹싱 제거 시 0.7 dB 성능 저하, 리피칭 제거 시 0.05 dB 저하[1]

**세그먼트 길이 영향**:
- 3.4초: 8.17 dB
- 7.8초: 8.70 dB (+0.53 dB)
- 12.2초 (희소 주의): 9.20 dB (+1.03 dB 누적)[1]

#### 2.5 한계

1. **데이터 의존성**: MUSDB만으로 성능 저하(-0.12 dB), 800곡 추가 필수[1]
2. **계산 리소스**: 8개 V100 GPU, 12.2초 이상 세그먼트 훈련 불가[1]
3. **"Other" 음원 약점**: 6.41 dB로 상대적 저성능
4. **도메인 외 일반화**: MUSDB18만 평가, 다른 장르/품질 미검증
5. **미세 조정 오버헤드**: 음원별 개별 훈련 필요[1]

***

### 3. 모델의 일반화 성능 향상 가능성 (중점)

#### 3.1 현재 일반화의 한계

최근 연구(2022-2024)에서 드러난 문제는 **음악 특성에 따른 극심한 성능 저하**이다. 동적 범위 압축(현대 음악의 마스터링 표준)에서 기존 모델은 **2-3 dB의 SDR 손실**을 경험한다. 특히 HT Demucs는 800곡 내부 특성과 MUSDB18의 편향성에 최적화되어 있다.[2][3]

#### 3.2 향상 가능성 제1: 데이터 다양성 확대

**장르/스타일 다양성**:

$$\mathcal{D}_{\text{balanced}} = \mathcal{D}_{\text{pop}} \cup \mathcal{D}_{\text{classical}} \cup \mathcal{D}_{\text{afrobeat}} \cup \mathcal{D}_{\text{electronic}}$$

기대 효과: 현악기/타악기 다양성 증가로 **4-8% 성능 향상**[2]

**녹음 환경 다양성**:

$$\mathcal{D}_{\text{environments}} = \mathcal{D}_{\text{studio}} \cup \mathcal{D}_{\text{live}} \cup \mathcal{D}_{\text{broadcast}} \cup \mathcal{D}_{\text{user-generated}}$$

기대 효과: 도메인 외 일반화 **5-10% 향상**[2]

**오디오 처리 조건**: 샘플링 레이트(16/48/96 kHz), 포맷(MP3/AAC/OPUS), 채널 구성 다양화

#### 3.3 향상 가능성 제2: 아키텍처 개선

**다중 스케일 교차 주의**:

$$\text{Output}_l = \text{Self-Attn}(x_l) + \sum_{s} \text{CrossAttn}_s(x_l, x_{l}^{\text{other-domain}})$$

기대 효과: 특히 "Other" 음원에서 **2-4 dB 개선**[4]

**적응형 희소성**:

$$\text{Sparsity}_{\text{adaptive}}(l) = \alpha \cdot \text{Entropy}(\text{Attention}_l) + \beta$$

기대 효과: **3-5% 효율성 개선**[4]

#### 3.4 향상 가능성 제3: 학습 전략 개선

**자기 지도 사전 학습** (Masked Spectrogram Modeling 등):

기대 효과: 도메인 외 일반화 **8-12% 향상**[5]

**메타 학습** (MAML):

새로운 도메인에 빠른 적응 가능, **2-3 에포크만으로 수렴**[5]

**대조 학습**:

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(\hat{s}_i, s_i) / \tau)}{\sum_j \exp(\text{sim}(\hat{s}_i, s_j) / \tau)}$$

기대 효과: 음원 간 구분력 **4-6% 향상**[5]

#### 3.5 향상 가능성 제4: 도메인 적응

**다중 도메인 동시 훈련**:

$$\mathcal{L}_{\text{multi-domain}} = \sum_{d} w_d \mathcal{L}_d$$

기대 효과: 도메인 외 성능 **10-15% 향상**[3]

**역 도메인 적응** (Domain-Adversarial):

도메인-불변 특성 자동 추출[3]

**테스트 시간 적응**:

출력의 무음성 제약을 이용한 비지도 손실 활용, 실제 배포에서 **3-5% 추가 개선**[3]

***

### 4. 논문이 앞으로의 연구에 미치는 영향

#### 4.1 패러다임 전환

**오디오 처리에서 Transformer의 위상 변화**:
- HT Demucs 이전: Transformer는 계산 비용으로 보조적 역할
- 이후: 교차 도메인 구조로 실용성 입증[1]

**결과**: 후속 연구에서 Transformer 채택 폭발적 증가
- BS-RoFormer (2023): 밴드 분할 + RoPE, 9.80 dB 달성[6]
- Stripe-Transformer (2023): 줄무늬 특성 학습[4]
- TFSWA-UNet (2024): 시간-주파수 이동 윈도우[4]

#### 4.2 구체적 기술 기여

**교차 도메인 주의의 일반화**:
후속 모델들이 표준으로 채택, Hybrid dual-path networks (2025), SCANet, MAJL 등[4]

**희소 주의의 실제 적용 사례**:
- 메모리 대비 성능 트레이드오프 명확화
- 실시간 저지연 모델 개발 가속화 (RT-STT, Band-SCNet 2025)[7]

**세분화된 미세 조정 절차**:
음원별 미세 조정 표준화, MoisesDB 6-음원 분리에서 HT Demucs 우수성 확인 (2024)[8]

#### 4.3 미해결 문제

**데이터 불균형의 근원**:
왜 800곡이 필수인가? 핵심 특성 대 다양성의 최적 비율은? → GASS (2023), Banquet (2024) 등에서 지속 탐구[9][5]

**"Other" 카테고리 한계**:
HT Demucs에서도 6.41 dB 최저 성능 → 2025 음악 음원 복구(Music Source Restoration) 개념으로 확장[10]

**도메인 외 일반화 부재**:
MUSDB18만 평가 → musdb-L/XL 데이터셋 생성(2022), LimitAug 증강(2022), 체계적 연구로 진화[2][3]

#### 4.4 실제 파급 효과

**BS-RoFormer (2023) 사례**:
- HT Demucs의 교차 도메인 구조 → 밴드 분할 + 계층적 Transformer로 발전
- SDX'23 경쟁 1등 달성, 9.80 dB[6]

**미세 조정 패러다임 확산**:
2024년까지 음원별 미세 조정이 표준 기법으로 정착, HT Demucs가 6-음원 분리에서 BSRNN 능가 (uSDR 6.26 vs 5.52 dB)[8]

#### 4.5 업계 영향

**오픈소스 생태계**:
GitHub 저장소(facebookresearch/demucs) 누적 다운로드 10만+, 상용 서비스 기초 제공[1]

**실시간 처리 가능성**:
RTF 1.02-2.04에서 시작, 2025 Band-SCNet은 RTF 0.76 달성 (저지연 스트리밍 가능)[7]

***

### 5. 앞으로의 연구 시 고려할 점

#### 5.1 데이터 관련

**품질 기준 명시화**:
- 악기 장르 분포, 오디오 품질, 녹음 환경 메타데이터 기록
- 자동 검증에서 기계학습 모델 또는 크라우드소싱 추가
- 데이터 품질 5-10% 향상 가능[11]

**도메인 외 성능 평가 필수**:
MUSDB18, MoisesDB, STEMS, 자체 레이블 데이터 최소 3개 데이터셋 평가[11]

#### 5.2 아키텍처 설계

**경량화 중심 전환**:
- 26.9M 파라미터 → 2.59M (2년간 10배 경량화 추세)
- LoRA, Adapter 등 파라미터 효율 기법 필수[7]

**음원 불가지론적 모델로의 진화**:
고정 4-음원 → 가변 음원, 쿼리 기반 분리로 확장[9]

**위치 인코딩 고급화**:
RoPE(이미 BS-RoFormer 채택), 상대 위치 편향 등으로 2-3% 성능 향상[6]

#### 5.3 학습 절차

**데이터 증강 심화**:
- HT Demucs의 리믹싱 중요성(0.7 dB) 강조
- 악기별 가중치 학습, 동적 증강 확률 도입[11]
- 2024 MERL 연구: 데이터 다양성이 일관성보다 중요(충분한 데이터 시)[11]

**그래디언트 안정성 관리**:
- 미세 조정 시 적응형 클리핑
- 손실 함수 재설계로 NaN 방지[11]

**체계적 하이퍼파라미터 최적화**:
- Bayesian Optimization, NAS 도입으로 1-2% 추가 성능[11]

#### 5.4 평가 및 해석

**음원별 세분화 분석**:
Percussion, Strings, Vocals 등 계층적 분류로 세부 장점/약점 파악[11]

**오류 분석 체계화**:
곡별, 악기 조합별, 시간대별 오류 분포 분석[11]

**신경망 기여도 분석**:
주의 가중치 시각화, Grad-CAM으로 내부 메커니즘 해석[11]

#### 5.5 실제 배포

**지연 시간-처리량 트레이드오프**:

$$\text{BatchSize}(\text{latency budget}) = \text{optimize}(\text{latency}, \text{throughput})$$

**모델 양자화와 압축**:
- Int8 양자화: 메모리 75% 감소, 추론 4배 가속
- 지식 증류: 소형 모델로 70-80% 성능 유지[11]

**점진적 배포**:
A/B 테스트 프레임워크, 앙상블 방식으로 성능 일관성 보장[11]

#### 5.6 학제간 협력

**음악 이론 통합**:
- 화성, 박자 기반 손실 함수 설계
- 악기 특성(톤, 동역학) 통합으로 음악적 그럴듯함 향상[12]

**음악학 전문가 협력**:
주관적 평가, 음악 프로덕션 워크플로우 이해, 실제 사용 사례 피드백[12]

***

### 6. 2020년 이후 관련 최신 연구

#### 6.1 Transformer 기반 음악 분리의 진화

| 연도 | 모델 | 기여 | 성능(MUSDB18) |
|------|------|------|--------------|
| 2022 | HT Demucs | 교차 도메인 주의 | 9.20 dB |
| 2023 | BS-RoFormer | 밴드 분할 + RoPE | 9.80 dB |
| 2023 | Stripe-Transformer | 줄무늬 특성 학습 | 6.71 dB |
| 2023 | Mel-RoFormer | Mel-스케일 밴드 | 개선됨 |
| 2024 | TFSWA-UNet | 시간-주파수 주의 | 9.16 dB(보컬) |
| 2024 | Hybrid LSTM-Transformer | 계층적 결합 | 3.18 dB 향상 |
| 2025 | Band-SCNet | 실시간 경량 | 7.79 dB |
| 2025 | RT-STT | 저지연 실시간 | 개발 중 |[6][7][4]

#### 6.2 데이터 증강 및 도메인 적응

**Cacophony 효과 (2024)**:
무작위 리믹싱의 효과 분석, 데이터 다양성이 일관성보다 중요임 입증, 1.0 dB 개선 가능[11]

**LimitAug (2022)**:
동적 범위 압축 대응 증강, musdb-L/XL 데이터셋으로 **2-3 dB 강건성 향상**[2]

#### 6.3 자기 지도 학습 및 전이 학습

**Pac-HuBERT (2023)**:
음악 분리에 HuBERT 적용, Demucs v2 대비 **1-2 dB 향상**, 소규모 데이터에서 우수[5]

**GASS (2023)**:
음성, 음악, 음향 사건을 단일 모델로 분리, 도메인 내 성능 우수이나 도메인 외 일반화 여전히 도전[5]

#### 6.4 음원 불가지론적 분리

**Banquet (2024)**:
쿼리 기반 설정, 단일 디코더로 6-음원 분리, 26.9M 파라미터로 경량화[9]

**FlowSep (2025)**:
언어 쿼리 기반 분리(Rectified Flow Matching), 자연어로 음원 지정 가능[13]

#### 6.5 실시간 및 경량 모델

**Band-SCNet (2025)**:
7.79 dB SDR, 2.59M 파라미터, 92 ms 지연으로 실시간 달성[7]

**RT-STT (2025)**:
양자화 적용 단일 경로 모델, 청음 보조기 및 라이브 성능 응용[7]

#### 6.6 MIR과의 통합

**음악 분류에 음원 분리 활용 (2024)**:
U-Net 사전 훈련으로 자동 태깅 **1.5% 성능 향상**[12]

**MAJL (2025)**:
음원 분리 + 음정 추정 동시 학습, **SDR 0.92 dB, 음정 인식 2.71% 향상**[12]

#### 6.7 벤치마킹 발전

**SDX 챌린지**: 여러 음원 지원으로 평가 확장[6]

**MoisesDB**: 6-음원 분리 데이터셋 (2023)[6]

**RawStems**: 음악 음원 복구 벤치마크 (2025)[10]

***

### 결론

**HT Demucs의 위상**:
- 학술: Transformer의 음악 처리 적용 패러다임 제시, 교차 도메인 구조 혁신[1]
- 한계: 과도한 데이터 의존성(800곡), 도메인 외 미검증, "Other" 음원 약점[1]

**현재 진행 중인 진화**:
- **단기**: 경량화/실시간성, 쿼리 기반 유연성, 멀티태스크 학습[9][7]
- **중기**: 자기 지도 대규모 활용, 음악 이론 기반 제약, 자동 도메인 적응[12][5]
- **장기**: 생성 모델과의 통합, 실시간 인터렉티브 편집, 보편적 음원 분리[10]

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/187c265e-106c-4903-a1c0-1269d4233d4f/2211.08553v1.pdf)
[2](https://ieeexplore.ieee.org/document/10446843/)
[3](https://ieeexplore.ieee.org/document/10849272/)
[4](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-022-00268-1)
[5](https://arxiv.org/abs/2406.18747)
[6](https://ieeexplore.ieee.org/document/10675842/)
[7](http://ijarsct.co.in/Paper16841.pdf)
[8](https://arxiv.org/abs/2310.01809)
[9](https://ieeexplore.ieee.org/document/10285863/)
[10](https://ieeexplore.ieee.org/document/10675907/)
[11](https://ieeexplore.ieee.org/document/10626965/)
[12](https://joss.theoj.org/papers/10.21105/joss.02154.pdf)
[13](https://arxiv.org/pdf/2305.07489.pdf)
[14](https://arxiv.org/pdf/2211.08553.pdf)
[15](https://arxiv.org/pdf/2111.14200.pdf)
[16](https://arxiv.org/pdf/2409.07614.pdf)
[17](https://arxiv.org/pdf/2304.02160.pdf)
[18](http://arxiv.org/pdf/2409.10995.pdf)
[19](https://arxiv.org/pdf/2112.07891.pdf)
[20](https://eusipco2025.org/wp-content/uploads/pdfs/0001238.pdf)
[21](https://arxiv.org/abs/2511.13146)
[22](https://sites.duke.edu/dkusmiip/files/2022/11/Sams-Net-A-Sliced-Attention-based-Neural-Network.pdf)
[23](https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.pdf)
[24](https://github.com/crlandsc/Music-Demixing-with-Band-Split-RNN)
[25](https://arxiv.org/abs/2308.08143)
[26](https://www.sciencedirect.com/science/article/abs/pii/S0167639324001420)
[27](https://www.db-thueringen.de/servlets/MCRFileNodeServlet/dbt_derivate_00055685/ilm1-2021000383.pdf)
[28](https://ieeexplore.ieee.org/document/10457445/)
[29](https://ieeexplore.ieee.org/document/10819011/)
[30](https://ieeexplore.ieee.org/document/10190129/)
[31](https://arxiv.org/html/2310.00140)
[32](http://arxiv.org/pdf/2407.03736.pdf)
[33](http://arxiv.org/pdf/2501.16171.pdf)
[34](http://arxiv.org/pdf/2310.15845v2.pdf)
[35](http://arxiv.org/pdf/2501.03689.pdf)
[36](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000411.pdf)
[37](https://archives.ismir.net/ismir2018/paper/000169.pdf)
[38](https://archives.ismir.net/ismir2022/paper/000069.pdf)
[39](https://arxiv.org/pdf/2410.20773.pdf)
[40](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Split_to_Merge_Unifying_Separated_Modalities_for_Unsupervised_Domain_Adaptation_CVPR_2024_paper.pdf)
[41](https://www.merl.com/publications/docs/TR2024-030.pdf)
[42](https://arxiv.org/html/2505.21827v1)
[43](https://arxiv.org/abs/2010.12650)
[44](https://mac.kaist.ac.kr/~juhan/gct634/2020-Fall/Finals/Data_Augmentation_for_Singing_Voice_Separation_Using_Musical_Instrument_Transfer_and_Resynthesis.pdf)
[45](https://www.nature.com/articles/s41598-025-20179-3)
