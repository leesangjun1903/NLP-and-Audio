# Guided-TTS: A Diffusion Model for Text-to-Speech via Classifier Guidance

### 1. 논문의 핵심 주장 및 주요 기여

**Guided-TTS**는 **전사본(transcript) 없이** 고품질 음성합성을 수행하는 혁신적인 확산 기반 텍스트음성변환(TTS) 모델입니다. 이 논문의 핵심 주장은 다음과 같습니다:[1]

**주요 기여:**

1. **무조건부 확산 모델 기반 음성 생성**: 기존 TTS 모델들이 텍스트-음성 쌍 데이터로 조건부 모델을 학습하는 것과 달리, Guided-TTS는 전사되지 않은 음성 데이터로부터 무조건부 DDPM(Denoising Diffusion Probabilistic Model)을 학습합니다.[1]

2. **분류기 지도(Classifier Guidance) 활용**: 분리된 음성 인식 데이터(LibriSpeech)에서 학습한 프레임별 음소 분류기를 활용하여, 무조건부 확산 모델을 텍스트 조건으로 지도합니다.[1]

3. **노름 기반 지도 방법 제시**: 기존 분류기 지도 방식의 발음 오류를 줄이기 위해 **노름 기반 스케일링(norm-based guidance)** 방법을 제안합니다.[1]

4. **다양한 데이터 세트에서의 일반화**: LJSpeech, Hi-Fi TTS, Blizzard 2013 등 다양한 특성의 데이터셋에서 뛰어난 성능을 보여주며, 특히 긴 형태의 전사되지 않은 데이터에도 적용 가능함을 입증합니다.[1]

***

### 2. 해결하고자 하는 문제, 제안 방법, 모델 구조

#### **2.1 해결하고자 하는 문제**

기존 TTS 모델들의 주요 제약사항:

- **데이터 수집의 어려움**: 고품질 음성합성을 위해 대상 화자의 **정확한 전사본**이 필수적입니다.[1]
- **자동 분할 및 전사의 부담**: 팟캐스트, 오디오북 등 장형의 전사되지 않은 음성 데이터는 활용하기 어렵습니다.[1]
- **적응형 TTS의 제한**: 기존 적응형 방식들은 사전학습된 다중 화자 TTS 모델에 의존하며, 성능이 단일 화자 모델에 미치지 못합니다.[1]

#### **2.2 제안하는 방법 및 수식**

**조건부 스코어 추정의 핵심:**

Guided-TTS는 Bayes 규칙을 통해 조건부 스코어를 다음과 같이 분해합니다:[1]

$$\nabla_{X_t}\log p(X_t|\hat{y}, \text{spk} = S) = \nabla_{X_t}\log p_\theta(X_t|\text{spk} = S) + \nabla_{X_t}\log p_\phi(\hat{y}|X_t, \text{spk} = S)$$

여기서:
- 첫 번째 항: 무조건부 DDPM의 스코어 추정
- 두 번째 항: 음소 분류기의 그래디언트

**역확산 과정:**

$$X_{t-1/N} = X_t + \frac{\beta_t}{N}\left(\frac{1}{2}X_t + \nabla_{X_t}\log p(X_t|\hat{y})\right) + \sqrt{\frac{\beta_t}{N}}z_t$$

**노름 기반 지도(Norm-based Guidance):**

기본 분류기 지도에서 $t \to 0$일 때 무조건부 스코어의 노름이 급격히 증가하여 분류기 그래디언트의 효과가 무시되는 문제를 해결합니다:[1]

$$\alpha_t = \frac{\|\nabla_{X_t}\log p_\theta(X_t)\|}{\|\nabla_{X_t}\log p_\phi(\hat{y}|X_t)\|}$$

이 비율을 그래디언트 스케일 $s$로 조정하면:[1]

$$X_{t-1/N} = X_t + \frac{\beta_t}{N}\left(\frac{1}{2}X_t + \nabla_{X_t}\log p_\theta(X_t) + s \cdot \alpha_t\nabla_{X_t}\log p_\phi(\hat{y}|X_t)\right) + \sqrt{\frac{\beta_t}{N}}z_t$$

#### **2.3 모델 구조**

Guided-TTS는 **4개의 모듈**로 구성됩니다:[1]

| 모듈 | 역할 | 상세 설명 |
|------|------|---------|
| **무조건부 DDPM** | 음성 생성 | U-Net 기반, 32×32 이미지용 아키텍처 사용, 멜스펙트로그램 생성[1] |
| **음소 분류기** | 텍스트 정보 인코딩 | WaveNet 유사 구조, 시간 임베딩과 화자 임베딩을 글로벌 조건으로 사용[1] |
| **지속시간 예측기** | 텍스트 토큰 정렬 | Glow-TTS와 동일 아키텍처, 텍스트와 화자 임베딩 연결[1] |
| **화자 인코더** | 화자 정보 추출 | VoxCeleb2 데이터로 GE2E 손실로 학습한 2층 LSTM[1] |

**학습 데이터:**
- 무조건부 DDPM: 대상 화자의 전사되지 않은 음성 (임의의 5초 청크)
- 음소 분류기, 지속시간 예측기: LibriSpeech (약 982시간)
- 화자 인코더: VoxCeleb2 (100만 개 이상의 발화)

***

### 3. 성능 향상 및 한계

#### **3.1 성능 향상**

**주관적 평가 (MOS - Mean Opinion Score):**

Guided-TTS는 전사본 없이도 기존 최고성능 모델과 경쟁력 있는 성능을 달성합니다:[1]

| 모델 | LJSpeech 전사본 사용 | MOS | CER(%) |
|------|------------------|-----|--------|
| Glow-TTS | ✓ | 4.14 ± 0.08 | 0.66 |
| Grad-TTS | ✓ | 4.25 ± 0.07 | 1.09 |
| **Guided-TTS** | **×** | **4.25 ± 0.08** | **1.03** |

**다양한 데이터셋에서의 일반화:**

| 데이터셋 | 모델 | MOS | CER(%) | 비고 |
|---------|------|-----|--------|------|
| Hi-Fi TTS (ID: 92) | Grad-TTS-ASR | 4.11 ± 0.08 | 1.33 | ASR 전사 오류 |
| | **Guided-TTS** | **4.20 ± 0.08** | **0.81** | **우수한 일반화** |
| Blizzard 2013 (장형) | **Guided-TTS** | **4.24 ± 0.09** | **0.24** | 세그멘테이션 없음 |

**노름 기반 지도의 효과:**

기존 분류기 지도 방식과의 비교에서 노름 기반 지도가 현저히 우수합니다:[1]

$$\text{CER(기존 분류기 지도)} \approx 12-15\% \quad \text{vs} \quad \text{CER(노름 기반)} \approx 1\%$$

**음소 분류기 데이터 의존성:**

LibriSpeech의 크기에 따른 성능 변화:[1]

| LibriSpeech 사용량 | CER(%) |
|------------------|--------|
| 1% (9시간) | 4.24 |
| 10% (96시간) | 2.28 |
| 100% (960시간) | **1.03** |

#### **3.2 모델의 한계**

**발음 정확성 문제:**

1. **초기 샘플링 단계의 약점**: 음소 분류기가 노이즈가 많은 단계 ($t \to 1$)에서 분류 정확도가 낮아져, 그래디언트의 효과가 감소합니다.[1]

2. **음소 분류기의 의존성**: 성능이 음소 분류기의 품질에 크게 의존하며, 대규모 ASR 데이터셋(LibriSpeech)의 가용성이 필수적입니다.[1]

**일반화의 한계:**

1. **언어 간 일반화**: LibriSpeech(영어)에서만 음소 분류기를 학습하므로, 다른 언어로의 확장이 제한적입니다.

2. **OOD(Out-of-Distribution) 텍스트**: 대상 화자 데이터에 없는 단어나 문맥에서의 성능 저하 가능성이 있습니다.

**계산 비용:**

- 샘플링 속도: RTF(Real-Time Factor) 0.486 (N=50 역확산 스텝)[1]
- 추론 시간: 무조건부 스코어 계산 0.184, 분류기 그래디언트 계산 0.291로 구성[1]

**데이터 요구사항:**

- 무조건부 DDPM을 위해 여전히 대량의 대상 화자 음성 데이터가 필요합니다.
- 전사본 제거의 이점이 음소 분류기 학습 데이터 필요성으로 상쇄됩니다.

***

### 4. 모델의 일반화 성능 향상 가능성

#### **4.1 현재 일반화 성능**

**다중 화자 일반화:**

Guided-TTS의 핵심 강점은 **화자 독립적 설계**에 있습니다:[1]
- 음소 분류기와 지속시간 예측기가 화자 임베딩을 입력받으므로, 보지 못한 화자에 대한 적응이 가능합니다.
- 단일 음소 분류기로 여러 데이터셋 (LJSpeech, Hi-Fi TTS, Blizzard)을 처리합니다.[1]

**데이터셋 간 일반화의 우수성:**

Grad-TTS-ASR(ASR로 생성된 전사본 사용)과의 비교에서:[1]

| 시나리오 | Grad-TTS-ASR | Guided-TTS | 우수성 |
|---------|--------------|-----------|--------|
| Hi-Fi TTS 남성 (ID: 9017) | 3.83 ± 0.09 | **4.04 ± 0.09** | **+5.5%** |
| Blizzard (장형) | 미테스트 | **4.24 ± 0.09** | **가능** |

#### **4.2 일반화 성능 향상을 위한 접근법**

**1) 확산 모델 구조의 개선**

최신 연구(2023-2025)에서 제시된 개선 방향:

- **DiTTo-TTS (2025)**: Diffusion Transformer(DiT) 기반으로 음소와 지속시간 의존성 제거. 이는 **더 나은 스케일러빌리티**를 제공합니다.[2]

- **Flow Matching 기반 모델**: OptimalTransport Conditional Flow Matching(OT-CFM)으로 더 효율적인 샘플링.[2]

**2) 분류기 지도 방법의 개선**

- **Classifier-Free Guidance(CFG) 통합**: Song et al. (2021)의 CFG를 TTS에 적용하여 분류기 의존성 감소.[3]

- **Retrieval-Augmented Guidance**: 동적 검색 기반 분류기 지도로 다양한 음성 특성에 더 잘 대응.[3]

**3) 멀티모달 학습**

- **UnDiff (2023)**: 무조건부 확산 모델이 대역폭 확장, 클리핑 제거, 음원 분리 등 다양한 역태스크에 적용 가능함을 보여줍니다. 이는 Guided-TTS의 일반화 가능성을 시사합니다.[4]

- **멀티태스크 학습**: 음성 향상, 음성 변환, 스타일 전이 등을 동시에 학습하여 일반화 성능 향상.[4]

**4) 스타일 및 운율 모델링**

- **DEX-TTS (2024)**: 참고 음성 기반 스타일 모델링으로 운율 다양성 증가.[5]

- **Prosody-TTS (2023)**: 마스크드 오토인코더로 운율 표현을 학습하여 보지 못한 화자의 운율도 일반화.[6]

**5) 강화학습을 통한 세밀한 조정**

- **DLPO (2024)**: Diffusion Model Loss-Guided Policy Optimization으로 확산 모델을 강화학습으로 최적화. MOS 예측 모델과 결합하여 품질 향상.[7][8]

#### **4.3 예상 성능 향상 시나리오**

**시나리오 1: 하이브리드 가이던스 전략**
```
기존 성능: MOS 4.25 (전사본 없음)
+ Classifier-Free Guidance 통합
+ 적응형 노름 스케일링
예상 성능: MOS 4.35-4.40
```

**시나리오 2: 다중 언어 음소 분류기**
```
현재: 영어 LibriSpeech만 사용
+ 다중 언어 ASR 모델 (e.g., Multilingual LibriSpeech)
+ 언어별 음소 사전 학습
예상 성능: 10개 언어 이상 지원, 언어별 CER < 2%
```

**시나리오 3: 적응형 음성 인코더**
```
현재: 고정된 화자 임베딩
+ 컨텍스트 기반 동적 화자 임베딩
+ 음성 특성 기반 실시간 적응
예상 성능: 다양한 음성 특성(방언, 감정)에 MOS +0.15-0.25
```

***

### 5. 논문이 앞으로의 연구에 미치는 영향

#### **5.1 학술적 영향**

**1) 패러다임 전환: 조건부에서 무조건부로**

Guided-TTS는 TTS 학습 방식의 근본적인 변화를 제시합니다:[1]

- **기존**: 텍스트-음성 쌍 데이터 필수 → 큰 수집 비용
- **새로운**: 무조건부 음성 모델 + 외부 분류기 → 데이터 효율성 증대

이는 **저자원 언어 및 방언의 TTS 개발**을 가능하게 합니다.[1]

**2) 분류기 지도의 재조명**

- Song et al. (2021b, 2021c)의 이미지 생성용 분류기 지도를 **음성 도메인에 맞게 적응**시킴.[1]
- **노름 기반 스케일링**이 도메인별 특성을 고려한 가이던스 설계의 중요성을 강조합니다.[1]

**3) 후속 연구의 기초**

**직접적인 후속 연구:**

- **Guided-TTS 2 (2022)**: 화자 조건부 확산 모델과 적응형 미세조정으로 10초 음성만으로 새로운 화자 적응.[9]
- **UnitSpeech (2023)**: 자감독 단위 표현을 사용하여 음소 분류기 의존성 감소.[10]

#### **5.2 실제 응용 가능성**

**1) 저자원 언어 TTS 개발**

- 전사본 없이 대량의 음성 데이터(오디오북, 팟캐스트)를 활용 가능.[1]
- 희귀 언어의 TTS 개발 비용 대폭 감소.

**2) 개인화된 음성 합성**

- 사용자의 몇 분 분량 음성 데이터로 개인화된 TTS 구축.[1]
- 마이크로소프트 Cortana, Apple Siri 등의 성우 맞춤화 강화.

**3) 엔터테인먼트 산업**

Guided-TTS 2에서 "Gollum" 음성 생성 사례를 통해, 가상 캐릭터의 독특한 음성 합성이 가능함을 증명.[9]

***

### 6. 앞으로 연구 시 고려할 점

#### **6.1 기술적 고려사항**

**1) 분류기 품질 관리**

- 음소 분류기의 노이즈 강건성 향상 필요:[1]
  - Noisy 음성에서의 분류 정확도 개선
  - 시간별 신뢰도 가중치 학습

**2) 다중 언어 확장**

- **언어별 음소 인벤토리의 차이** 처리:
  - 언어별 음소 분류기 개발 vs. 다중언어 통합 분류기
  - 문자 체계가 다른 언어(CJK)의 처리 방안

**3) 계산 효율성**

- 현재 RTF 0.486은 실시간(RTF < 1.0)이나, 더 빠른 추론 필요:
  - 역확산 스텝 감소: 50 → 10-20 스텝으로 최적화
  - 증류(Distillation) 기법 적용[11]
  - ODE 기반 고속 샘플러 활용

#### **6.2 평가 지표 개선**

**1) CER(Character Error Rate)의 한계**

- CER은 음성 품질(음성美)을 측정하지 못함.
- **제안**: MOS + CER + 화자 유사도(SECS) + 자연스러움(UTMOS) 조합 평가.[12]

**2) 일반화 능력 평가**

- OOD 텍스트, 보지 못한 음성 특성에 대한 체계적 평가 필요.[1]
- Cross-lingual, cross-domain 평가 프로토콜 확립.

#### **6.3 데이터 및 훈련 전략**

**1) 대규모 무조건부 데이터 활용**

- 현재: 각 화자별로 별도의 DDPM 학습 필요
- **개선안**: 다중 화자 무조건부 DDPM으로 전이학습(Transfer Learning) 적용[9]

**2) 반감독 학습(Semi-supervised Learning)**

- 일부 전사본 + 대량 무조건부 데이터 혼합 학습
- 점진적 레이블링 전략으로 데이터 효율성 향상

#### **6.4 새로운 응용 영역**

**1) 음성 변환(Voice Conversion)**

- Guided-TTS의 구조를 활용한 화자 변환.[4]
- 감정, 방언 변환 등으로 확장 가능.

**2) 음성 향상 및 복원**

- UnDiff와 같이 무조건부 확산 모델의 다목적 활용.[4]
- 저품질 음성의 자동 향상.

**3) 멀티모달 학습**

- 음성 + 텍스트 + 비전 정보 결합
- 표정, 제스처 동기화된 TTS[13]

***

### 7. 2020년 이후 최신 관련 연구 탐색

#### **7.1 확산 모델 기반 TTS의 진화 (2021-2025)**

| 연도 | 모델명 | 주요 특징 | 성능 향상 |
|------|-------|---------|---------|
| 2021 | **Grad-TTS**[14] | SDE 기반 스코어 매칭, 품질-속도 트레이드오프 | MOS 4.25 |
| 2022 | **Guided-TTS** (본 논문) | 무조건부 DDPM + 분류기 지도, 노름 기반 스케일링 | **전사본 없이 MOS 4.25** |
| 2022 | **Guided-TTS 2**[9] | 화자 조건부 DDPM, 10초 적응 | **제로샷 다화자 지원** |
| 2023 | **StyleTTS 2**[15] | 스타일 확산 + 대규모 음성 언어 모델 | **인간 수준 MOS 4.5+** |
| 2023 | **UnDiff**[4] | 무조건부 음성 복원, 다양한 역태스크 지원 | **다목적 활용** |
| 2024 | **DEX-TTS**[5] | 시간 변동성 스타일 모델링 | **운율 다양성 증가** |
| 2024 | **DPI-TTS**[16] | 방향성 패치 상호작용, DiT 기반 | **빠른 수렴** |
| 2025 | **DiTTo-TTS**[2] | DiT 기반, 음소/지속시간 제거 | **확장성 개선** |

#### **7.2 주요 기술 트렌드**

**1) Transformer 기반 모델로의 전환**

- **DPI-TTS (2024)**: U-Net 대신 Diffusion Transformer(DiT) 사용.[16]
- **이점**: 더 나은 스케일러빌리티, 장기 의존성 모델링 개선.[2]

**2) Flow Matching의 부상**

- **최신 모델들**: Optimal Transport Conditional Flow Matching(OT-CFM)으로 더 효율적 학습.[2]
- **DiTTo-TTS (2025)**: 음소/지속시간 없이 raw 텍스트에서 직접 음성 생성.[2]

**3) 강화학습을 통한 미세조정**

- **DLPO (2024)**: MOS 예측 모델을 보상으로 사용한 강화학습.[7]
- **성과**: 기존 모델 대비 MOS +0.3-0.4 향상.[8]

**4) 멀티모달 일반화**

- **음성 + 텍스트 + 비전**: CoCoGesture(2024)는 음성 기반 제스처 생성으로 멀티모달 일관성 강화.[13]

#### **7.3 저자원 언어 및 적응형 TTS**

**Guided-TTS의 영향:**

| 연구 | 접근법 | 성과 |
|-----|-------|------|
| **UnitSpeech (2023)**[10] | 자감독 단위 표현 | 전사본 없이 화자 적응 |
| **BnTTS (2025)**[17] | 저자원 벵갈어 | 50시간 데이터로 MOS 3.8+ |
| **Voice Filter (2022)**[18] | 극저자원 (1분) | 1분 음성으로 음성 필터링 |

#### **7.4 확산 모델의 이론적 발전**

**최신 이론 연구:**

- **"What happens to diffusion model likelihood when your model is conditional?" (2024)**: 조건부 확산 모델의 가능도(likelihood) 특성 규명. Guided-TTS 같은 모델의 이론적 기초 제공.[19]

- **"Critical Windows in Diffusion Models" (2024)**: 확산 과정 중 특정 시간 구간에서만 특정 특성이 나타나는 현상 규명. Guided-TTS의 **노름 기반 지도**가 이를 해결하는 방법으로 해석 가능.[20]

- **"Schrodinger Bridges for TTS (2023)"**: 가우시안 사전 대신 명확한 음성 정보를 포함한 사전 사용으로 샘플링 효율성 향상.[21]

#### **7.5 실제 배포 사례**

**2023-2025 최신 응용:**

1. **AI 성우 서비스**: 다양한 언어의 자동 더빙.[15]
2. **접근성 기술**: 언어 장애인을 위한 음성 합성.[17]
3. **게임/영화 제작**: 캐릭터 음성 자동 생성 (Gollum 사례).[9]
4. **개인용 AI 어시스턴트**: 사용자 목소리 기반 맞춤형 음성.[12]

***

### 8. 종합 결론

**Guided-TTS**는 다음 이유로 **TTS 분야의 분수령**이 됩니다:

1. **패러다임 전환**: 조건부 모델 중심 → 무조건부 모델 + 분류기 지도 방식으로 데이터 요구 조건 완화.[1]

2. **실용성 강화**: 전사본 없이 고품질 음성 합성 가능하여 저자원 언어 및 개인화 TTS 개발 가능.[1]

3. **기술 혁신**: **노름 기반 지도** 방법으로 기존 분류기 지도의 문제 해결, 도메인 적응 가이던스의 좋은 사례 제시.[1]

4. **후속 연구의 촉발**: Guided-TTS 2, UnitSpeech, DiTTo-TTS 등 다양한 후속 연구로 지속적인 발전.[10][9][2]

**향후 개선 방향**:
- **다언어 지원**: 언어별 음소 분류기 개발
- **계산 효율성**: 역확산 스텝 감소 및 증류 기법 적용
- **강화학습 활용**: MOS 기반 최적화로 품질 향상
- **멀티모달 일반화**: 음성 외 다른 모달리티와의 통합

2025년 현재, Guided-TTS의 원리는 **DiTTo-TTS**, **StyleTTS 2**, **DLPO** 등에 의해 더욱 정교하게 발전하고 있으며, 확산 모델 기반 음성 생성의 **표준 패러다임**으로 자리잡고 있습니다.

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a1f0bba0-78ea-4d16-817e-561dfa346724/2111.11755v4.pdf)
[2](http://arxiv.org/pdf/2406.11427.pdf)
[3](https://www.isca-archive.org/interspeech_2024/choi24c_interspeech.pdf)
[4](https://arxiv.org/abs/2306.00721)
[5](https://arxiv.org/abs/2406.19135)
[6](https://aclanthology.org/2023.findings-acl.508)
[7](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[8](http://arxiv.org/pdf/2405.14632.pdf)
[9](https://arxiv.org/abs/2205.15370)
[10](https://www.isca-archive.org/interspeech_2023/kim23k_interspeech.pdf)
[11](https://www.isca-archive.org/interspeech_2022/vovk22_interspeech.html)
[12](https://ieeexplore.ieee.org/document/10974277/)
[13](https://arxiv.org/abs/2405.16874)
[14](https://www.semanticscholar.org/paper/2e32cde6e080f990873638f2e113767a6a19c824)
[15](https://arxiv.org/pdf/2306.07691.pdf)
[16](https://arxiv.org/html/2409.11835)
[17](https://aclanthology.org/2025.findings-naacl.279.pdf)
[18](https://arxiv.org/abs/2202.08164)
[19](https://arxiv.org/abs/2409.06364)
[20](https://arxiv.org/abs/2403.01633)
[21](http://arxiv.org/pdf/2312.03491.pdf)
[22](https://arxiv.org/pdf/2308.16569.pdf)
[23](http://arxiv.org/pdf/2309.06787.pdf)
[24](http://arxiv.org/pdf/2211.09383.pdf)
[25](https://arxiv.org/abs/2409.10058)
[26](https://arxiv.org/html/2505.19595v1)
[27](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/guided-tts/)
[28](https://www.inf.uni-hamburg.de/en/inst/ab/sp/research/diffusion-models.html)
[29](https://www.isca-archive.org/interspeech_2025/chen25b_interspeech.pdf)
[30](https://liner.com/review/guidedtts-diffusion-model-for-texttospeech-via-classifier-guidance)
[31](https://www.emergentmind.com/open-problems/generalization-lower-parameterized-diffusion-se-models)
[32](https://aclanthology.org/2025.coling-main.352.pdf)
[33](https://arxiv.org/abs/2509.19668)
[34](https://www.siam.org/publications/siam-news/articles/generalization-of-diffusion-models-principles-theory-and-implications/)
[35](https://developer.nvidia.com/blog/speeding-up-text-to-speech-diffusion-models-by-distillation/)
[36](https://ieeexplore.ieee.org/document/10694254/)
[37](https://dl.acm.org/doi/10.1145/3707292.3707367)
[38](https://dl.acm.org/doi/10.1145/3610661.3616554)
[39](https://arxiv.org/abs/2310.01381)
[40](https://www.semanticscholar.org/paper/ddd8dde080ad1f906d2c9f8f859621cf3505a6f7)
[41](https://dl.acm.org/doi/10.1145/3610661.3616552)
[42](https://aclanthology.org/2023.emnlp-main.709.pdf)
[43](https://arxiv.org/ftp/arxiv/papers/2301/2301.13267.pdf)
[44](https://arxiv.org/pdf/2309.10457.pdf)
[45](https://aclanthology.org/2023.findings-acl.437.pdf)
[46](https://arxiv.org/pdf/2111.11755.pdf)
[47](https://arxiv.org/pdf/2306.00721.pdf)
[48](http://arxiv.org/pdf/2406.08203.pdf)
[49](https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Bansal_Universal_Guidance_for_Diffusion_Models_CVPRW_2023_paper.pdf)
[50](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/guided-tts2/)
[51](https://arxiv.org/abs/2303.15669)
[52](https://randomsampling.tistory.com/178)
[53](https://ieeexplore.ieee.org/iel8/6570655/10304349/10704960.pdf)
[54](https://arxiv.org/abs/2211.02448)
[55](https://arxiv.org/abs/2303.01849)
[56](https://ieeexplore.ieee.org/document/10446203/)
[57](https://arxiv.org/abs/2310.18169)
[58](http://arxiv.org/pdf/2211.02448.pdf)
[59](https://arxiv.org/pdf/2305.10891.pdf)
[60](https://arxiv.org/pdf/2212.14518.pdf)
[61](http://arxiv.org/pdf/2402.12423.pdf)
[62](https://arxiv.org/abs/2308.10428)
[63](https://arxiv.org/pdf/2205.15370.pdf)
[64](https://proceedings.mlr.press/v139/popov21a/popov21a.pdf)
[65](https://www.isca-archive.org/interspeech_2022/welker22_interspeech.pdf)
[66](https://grad-tts.github.io)
[67](https://randomsampling.tistory.com/385)
[68](https://arxiv.org/abs/2203.17004)
[69](https://music-audio-ai.tistory.com/40)
[70](https://github.com/sp-uhh/sgmse)
[71](https://randomsampling.tistory.com/87)
