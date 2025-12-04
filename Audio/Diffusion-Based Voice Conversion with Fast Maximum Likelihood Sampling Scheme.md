# Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme

### 1. 핵심 주장 및 주요 기여 요약[1]

본 논문은 **원샷(one-shot) 다대다(many-to-many) 음성 변환**을 위한 확산 확률 모델 기반의 고품질 음성 변환 시스템을 제시합니다. 핵심 혁신은 **최대우도 확률 미분방정식(Maximum Likelihood SDE) 솔버**의 개발로, 이를 통해 모델 재훈련 없이도 추론 단계를 수백 개에서 단 6단계로 감소시킬 수 있습니다. 이 접근 방식은 모든 유형의 확산 모델과 생성 작업에 적용 가능한 **범용적 특성**을 보유하고 있으며, 이론적 분석과 실증적 연구로 정당화됩니다.[1]

### 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

#### 2.1 문제 정의[2][1]

원샷 다대다 음성 변환은 **단 하나의 참조 발화**만으로 목표 음성을 학습하면서도 **훈련 데이터에 없는 화자**로부터의 변환을 수행해야 하는 극도로 어려운 문제입니다. 기존 오토인코더 기반 음성 변환 모델들의 주요 과제는:[1]

- **음성-내용 분리(disentanglement)**: 인코더에서 화자 정체성과 음성 내용을 분리하기
- **느린 추론**: 확산 모델의 반복적 특성으로 인한 수백 번의 역확산 단계 필요[1]
- **낮은 일반화 성능**: 훈련 데이터와 다른 음성 도메인에서의 성능 저하

#### 2.2 제안 방법론[2][1]

**인코더 설계**[1]

"평균 음성(average voice)"을 예측하도록 훈련된 고유한 인코더를 사용합니다. 이는 Montreal Forced Aligner를 통해 LibriTTS 데이터셋의 모든 음소 멜-스펙트로그램을 집계하여 얻어진 음소 수준 평균 멜 특성으로 입력을 변환합니다. 이 설계는 화자 정체성을 효과적으로 제거하면서도 음성 내용을 보존합니다.[1]

**확산 모델 기반 디코더**[1]

$$dX_t = \frac{1}{2}\beta_t X_t dt + \sqrt{\beta_t} dW_t \quad \text{(Forward SDE)} \quad (1)$$

$$d\tilde{X}_t = -\left(\frac{1}{2}\beta_t \tilde{X}_t + \beta_t \nabla_{\tilde{X}_t} \log p_{\tilde{X}_t}(\tilde{X}_t) \right) dt + \sqrt{\beta_t} d\bar{W}_t \quad \text{(Reverse SDE)} \quad (2)$$

여기서 $\beta_t$는 노이즈 스케줄, $s_\theta(\tilde{X}_t, \psi, t)$는 스코어 함수, $g_t(Y)$는 화자 조건 네트워크입니다.[1]

인코더 출력 $X^* = \phi(X_0)$를 확산 모델의 사전분포(prior)로 설정합니다:[1]

$$\mathcal{L}(\theta) = \int_0^1 \lambda_t \mathbb{E}_{X_0, X_t \sim p_t(X_0)} \|\nabla_{\tilde{X}_t} \log p_t(\tilde{X}_t|\tilde{X}_0) - s_\theta(\tilde{X}_t, \psi, t)\|_2^2 dt \quad (3)$$

#### 2.3 최대우도 SDE 솔버 (Maximum Likelihood SDE Solver)[3][1]

논문의 **핵심 이론적 기여**인 Theorem 1은 다음과 같은 최적화된 리샘플링 스킴을 제시합니다:[1]

$$X_{t-h} = X_t - \alpha_{t,h} \frac{1}{2}\beta_{t,h} X_t - \alpha_{t,h} \sigma_{t,h} s_\theta(X_t, t) + \beta_{t,h} \epsilon_t \quad (4)$$

여기서 최적 계수는:[1]

$$\alpha_{t,h} = \mu_{t,h}, \quad \sigma_{t,h} = \sigma_{t,h}, \quad \beta_{t,h} = \rho_{t,h}$$

구체적으로:[1]

$$\mu_{t,h} = \frac{\mu_{t,h}}{1 - 2\mu_{t,h}}, \quad \sigma_{t,h} = \frac{\sigma_{t,h}}{1 - 2\mu_{t,h}}$$

$$\rho_{t,h}^2 = \frac{2\sigma_{t,h}^2 - 2\mu_{t,h}^2}{1 - 2\mu_{t,h}} - \frac{n}{2\rho_{t,h}} \mathbb{E}_{X_t}\text{Tr}[\text{Var}(X_0|X_t)]$$

이 솔버는 **이산 표본 경로의 로그우도**를 최대화하며, 표준 Euler-Maruyama 솔버와는 다음과 같이 다릅니다:[1]

| 특성 | Euler-Maruyama | Maximum Likelihood |
|------|-----------------|-------------------|
| $N=6$ 단계에서 성능 | 열악 (FID 높음) | 우수 |
| 계산 오버헤드 | 최소 | 데이터 의존 항 추정 필요 |
| 이론적 보장 | 일반적 | 로그우도 최적성 증명[1] |

#### 2.4 모델 구조[1]

**화자 조건 네트워크 $g_t(Y)$의 세 가지 변형**[1]

1. **d-only**: 사전학습된 화자 검증 네트워크에서 추출한 화자 임베딩만 사용
2. **wodyn**: 화자 임베딩 + 노이즈 있는 목표 멜-스펙트로그램 $Y_t$
3. **whole**: 화자 임베딩 + 전체 확산 궤적 $\{Y_s : s=0.515, 1.515, ..., 14.515\}$

실험 결과 **wodyn**이 최적 성능을 제공합니다.[1]

#### 2.5 성능 향상[2][1]

**객관적 평가 - Fréchet Inception Distance (FID)**[1]

| 방법 | VP DPM (N=10) | VP DPM (N=100) |
|------|----------------|----------------|
| Euler-Maruyama | 229.6 | 19.68 |
| Probability Flow | 88.92 | 5.70 |
| **Maximum Likelihood (λ=0.5)** | **24.45** | **7.82** |

**주관적 평가 - VCTK 데이터셋 (MOS 스코어)**[1]

| 모델 | 자연성 (VCTK) | 화자 유사도 (VCTK) |
|------|---|---|
| AGAIN-VC | 1.98±0.05 | 1.97±0.08 |
| FragmentVC | 2.20±0.06 | 2.45±0.09 |
| VQMIVC | 2.89±0.06 | 2.60±0.10 |
| **Diff-VCTK-ML-6** | **3.73±0.06** | **3.47±0.09** |
| **Diff-VCTK-ML-30** | **3.73±0.06** | **3.57±0.09** |
| 원본 음성 | 4.55±0.05 | 4.52±0.07 |

**LibriTTS 대규모 데이터셋 결과 (25개 미지의 화자)**[1]

| 모델 | 자연성 | 화자 유사도 |
|------|-------|----------|
| Diff-LibriTTS-EM-6 | 1.57±0.02 | 1.47±0.03 |
| Diff-LibriTTS-PF-6 | 2.99±0.03 | 2.50±0.04 |
| **Diff-LibriTTS-ML-6** | **3.80±0.03** | **3.27±0.05** |
| **Diff-LibriTTS-ML-30** | **4.02±0.03** | **3.39±0.05** |
| BNE-PPG-VC (기준 모델) | 3.83±0.03 | 3.03±0.05 |

특히 **6단계 추론에서도 30단계 모델과 유사한 품질**을 달성하면서 **실시간 계수(RTF) 0.1**로 10배 빠른 추론을 실현합니다.[1]

#### 2.6 한계 및 제한사항[2][1]

1. **스코어 함수 근사 오류**: 최적성 보장은 스코어 함수가 완전히 학습되었을 때만 성립하므로, 실제로는 $\lambda$ 하이퍼파라미터 조정이 필요합니다.[1]

2. **데이터 의존 항 추정의 어려움**: 식 (4)의 분산 항 $\mathbb{E}_{X_t}\text{Tr}[\text{Var}(X_0|X_t)]$ 계산이 비자명하여 실험에서는 0으로 설정합니다.[1]

3. **음운 오류 증가**: 기준 모델 BNE-PPG-VC 대비 발음 오류가 더 많이 발생하며, 이는 음소 사후 확률(PPG) 특징 통합으로 개선 가능합니다.[1]

4. **미지 화자 도메인 일반화**: VCTK에서 훈련한 모델이 다른 화자에 적용할 때 성능 저하 경험 (MOS 3.6 → 3.5).[1]

### 3. 모델의 일반화 성능 향상 가능성[2][1]

#### 3.1 현재 달성한 일반화 능력[1]

논문의 모델은 **원샷 임의-대-임의(any-to-any) 변환 설정**에서 강력한 일반화를 보여줍니다:[1]

- **훈련 셋 내 화자 (VCTK 9명)**: MOS 3.73±0.06 (자연성)
- **훈련 셋 외 미지 화자 (16명 추가)**: MOS 3.39±0.04 (자연성)
- **성능 저하율**: ~9% (상대적으로 적음)

대규모 LibriTTS 데이터셋에서의 훈련이 일반화를 크게 개선합니다:[1]

- **미지 화자 평가**: MOS 3.80±0.03 (자연성), 3.27±0.05 (유사도)
- **기준 모델과 동등 또는 우월한 성능**

#### 3.2 일반화 성능 향상의 이론적 근거[2][1]

**1. 데이터 의존 사전분포의 역할**

확산 모델의 사전분포가 입력 $X^* = \phi(X_0)$에 의존한다는 사실이 중요합니다:[1]

```math
p_1(X_1 | X^*) \approx \mathcal{N}(X^*, I)
```

이는 **강한 조건화(strong conditioning)**를 의미하며, 조건화된 분포의 분산이 작아져 최적 SDE 솔버가 더욱 효과적입니다. 이론의 조건 ii에서 보듯이:[1]

> "X₀가 상수 또는 대각 이등방 공분산을 가진 가우시안 분포이면, 생성 모델은 정확합니다."[1]

강한 조건화 극한에서 $X_0|c = X_0(c)$가 상수에 가까워져 정확성이 보장됩니다.

**2. 평균 음성 표현의 안정성**

인코더가 "평균 음성"으로 변환하는 설계는 화자 간 변동성을 제거하면서도 **음성 내용의 본질적 구조**를 보존합니다. 이는 다양한 도메인의 화자에 대한 로버스트한 표현을 생성합니다.[1]

#### 3.3 향상 가능한 방향[2][1]

**1. 음소 사후 확률(PPG) 통합**

논문에서 언급하듯이:[1]

> "PPG 특징을 제안된 확산 VC 프레임워크에 통합하는 것이 유망한 미래 연구 방향입니다."

**2. 대규모 데이터셋 활용**

- LibriTTS 기반 모델 (1,100개 화자)이 VCTK 기반 모델 (109개 화자)보다 우월한 일반화 능력 시현
- **더 다양한 음성 특성 데이터 수집이 일반화 개선의 핵심**

**3. 적응형 하이퍼파라미터 조정**

식 (4)의 $\lambda$ 매개변수를 테스트 도메인 특성에 맞게 동적으로 조정하면 성능 향상 가능:[1]

- 현재: 고정 $\lambda = 0.5$ 사용
- 개선: 도메인 특정 $\lambda$ 값 학습

**4. 크로스 도메인 데이터 증강**

음성 변환 자체를 **데이터 증강 기법**으로 활용하여 모델의 도메인 로버스트성 강화. 최신 연구에서는 음성 변환을 통한 데이터 증강이 **34% 이상의 크로스 도메인 정확도 개선**을 달성합니다.[4]

### 4. 논문이 앞으로의 연구에 미치는 영향 및 고려사항

#### 4.1 학술적 영향[5][2][1]

**확산 모델 이론에 대한 기여**

논문의 Maximum Likelihood SDE 솔버는 음성 변환을 넘어 모든 확산 모델 기반 생성 작업에 적용 가능한 **범용적 가속화 기법**을 제시합니다. 이는:[1]

- CIFAR-10 이미지 생성에서도 검증됨 (Table 4)
- VP, MR-VP, sub-VP, VE 등 모든 주요 확산 모델 타입 지원

**최근 SDE 기반 확산 모델 이론 발전**과의 연계[5]

최신 기술 튜토리얼에서도 확산 모델의 **샘플링 효율성과 스코어 매칭**을 핵심 연구 주제로 다루며, 본 논문의 접근이 현재의 연구 동향과 부합합니다.[5]

#### 4.2 실제 응용 측면의 영향[6][7][1]

**실시간 음성 변환 실현**

- 논문: 6단계 추론으로 **RTF 0.1** 달성[1]
- 최신 진전: FastVoiceGrad (2024)와 FasterVoiceGrad (2025)는 **원스텝 음성 변환**으로 진화[7][8][6]

이들은 본 논문의 가속화 원리를 확장하여 **대규모 실시간 음성 응용**을 가능케 합니다.[7]

**엣지 디바이스 배포**

- 경량화 모델: LHQ-SVC (2024)는 CPU 호환성과 함께 확산 기반 음성 변환의 **모바일 배포** 가능성을 입증[9]

#### 4.3 데이터셋과 학습 전략 개선[10][4][2]

**크로스 도메인 로버스트성 강화**

2025년 음성 변환 도전 대회에서 보듯이, 현재 연구는:[11]

1. **다양한 데이터 소스 통합**: 음성 + 노래 데이터 결합[2]
2. **강화된 조건 조건화**: 내용 표현과 음향 특성 동시 제어[10]
3. **도메인 특정 미세조정**: 제한된 목표 음성 데이터로 효율적 적응[2]

**구체적 성과**

- T13 시스템 (SVCC 2023): 대규모 데이터 (750시간)로 훈련 후 미세조정하여 **크로스 도메인 음성 변환에서 경쟁력 있는 성능** 달성[2]
- 2025년 도전에서도 이 원칙이 지배적 전략으로 유지[11]

#### 4.4 일반화 성능 향상을 위한 향후 연구 방향[12][10][1]

**1. 감정 정보 통합**

최근 정서 음성 변환(Emotional Voice Conversion)이 부상하며, **감정 분리 손실과 표현 가이던스**를 통합한 확산 기반 프레임워크가 개발 중입니다. 이를 통해:[12]

- 화자 특성 + 감정 표현의 **동시 제어** 가능
- 더 풍부한 음성 다양성 생성 가능

**2. 자연언어 프롬프트 기반 제어**

PromptVC (2023)에서 제시된 **자연언어 프롬프트로 음성 스타일 제어**하는 방식이 확산 모델과 결합되어:[13]

- 직관적이고 해석 가능한 음성 변환 제어
- 복잡한 음성 특성의 세밀한 조절

**3. Diffusion Transformers (DiT) 기반 구조**

최신 DiTVC는 U-Net 대신 Transformer 기반 확산을 도입하여:[10]

- **더 나은 장거리 의존성 모델링**
- 환경 음향 특성까지 **정확한 모방** 가능[10]

#### 4.5 이론적 한계와 극복 방안[2][1]

**Theorem 1의 제약 조건**

논문의 최적성 보장은 다음을 가정합니다:[1]

1. 역확산이 최적 상태로 훈련됨 ( $\nabla_{\tilde{X}\_t} \log p_{\tilde{X}\_t}(\tilde{X}\_t) = s_\theta^*$ )
2. 노이즈 무관 항 추정 가능

**실제 대응**

- $\lambda$ 하이퍼파라미터 도입으로 부분 최적성 유지[1]
- 최근 연구에서 **더 정교한 분산 추정 기법** 개발 중[5]

#### 4.6 2020년 이후 최신 연구 동향[6][4][12][11][7][10][2]

| 연도 | 논문/시스템 | 핵심 기여 | 일반화 관련 진전 |
|------|----------|---------|-----------|
| 2021 | **본 논문** | ML-SDE 솔버 제시 | 6단계로 고품질 음성 변환 |
| 2023 | T13 SVCC | 대규모 데이터 + 미세조정 | 750시간 데이터로 크로스 도메인 성능 개선 |
| 2023 | PromptVC | 자연언어 스타일 제어 | 해석 가능성 강화 |
| 2024 | FastVoiceGrad | 조건부 확산 증류 | 원스텝 음성 변환 달성 |
| 2024 | LHQ-SVC | 경량화 + CPU 호환 | 엣지 디바이스 배포 가능 |
| 2025 | FasterVoiceGrad | 개선된 일스텝 방법 | 효율성과 품질 균형 |
| 2025 | DiTVC | Diffusion Transformer | 환경 음향 특성 모방 |
| 2025 | Singing Voice Conversion Challenge | 다양한 도메인 통합 | 음성-노래 크로스 도메인 |

### 5. 향후 연구 시 고려할 핵심 사항

1. **데이터 다양성**: 대규모 다중 도메인 데이터셋이 일반화 성능의 핵심 결정요소
2. **이론-실제 간극 해소**: 완전 최적 스코어 함수 학습 또는 적응형 $\lambda$ 조정 기법 개발
3. **구조적 혁신**: Transformer 기반 확산 모델로의 전환으로 장거리 의존성 향상
4. **다중 목표 최적화**: 화자 유사도, 자연성, 내용 정확성, 음향 환경 특성을 동시에 최적화
5. **실시간 응용**: 원스텝 또는 소수 단계 추론으로 임베디드 시스템 지원

이 논문은 **확산 모델 기반 음성 처리의 새로운 패러다임**을 제시하며, 이후 수년간의 관련 연구에서 가속화, 일반화, 구조적 개선의 기초가 되었습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/223d93e5-4280-4cc4-8b90-a8b6247e772d/2109.13821v2.pdf)
[2](https://ieeexplore.ieee.org/document/10389779/)
[3](https://ieeexplore.ieee.org/document/11198028/)
[4](https://www.isca-archive.org/interspeech_2025/abdullah25_interspeech.pdf)
[5](https://arxiv.org/abs/2402.07487)
[6](http://arxiv.org/pdf/2409.02245.pdf)
[7](https://arxiv.org/abs/2409.02245)
[8](https://www.isca-archive.org/interspeech_2025/kaneko25_interspeech.html)
[9](http://arxiv.org/pdf/2409.08583.pdf)
[10](https://pixl.cs.princeton.edu/pubs/Wang_2025_OVC/Wang-WASPAA-2025.pdf)
[11](https://arxiv.org/html/2509.15629v1)
[12](https://arxiv.org/html/2409.03636v1)
[13](http://arxiv.org/pdf/2309.09262.pdf)
[14](https://pubs.aip.org/pof/article/37/11/117119/3371491/Fine-structure-investigation-of-turbulence-induced)
[15](https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs)
[16](https://pubs.aip.org/pof/article/37/11/117120/3371493/Fine-structure-investigation-of-turbulence-induced)
[17](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[18](https://bmjopen.bmj.com/lookup/doi/10.1136/bmjopen-2024-093884)
[19](https://indonesianfeministjournal.org/index.php/IFJ/article/view/1191)
[20](https://www.mdpi.com/2413-8851/9/11/488)
[21](https://doi.apa.org/doi/10.1037/emo0001511)
[22](https://arxiv.org/pdf/2109.13821.pdf)
[23](https://aclanthology.org/2023.emnlp-main.990.pdf)
[24](https://arxiv.org/pdf/2304.11750.pdf)
[25](http://arxiv.org/pdf/2409.09401.pdf)
[26](https://www.sciencedirect.com/science/article/abs/pii/S0003682X22003887)
[27](https://arxiv.org/abs/2405.00930)
[28](https://www.sciencedirect.com/science/article/abs/pii/S1566253525001022)
[29](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dddm-vc/)
[30](https://github.com/PecholaL/MAIN-VC)
[31](https://ai.meta.com/research/publications/unsupervised-cross-domain-singing-voice-conversion/)
[32](https://liner.com/ko/review/diffusionbased-voice-conversion-with-fast-maximum-likelihood-sampling-scheme)
[33](https://ieeexplore.ieee.org/document/11129906/)
[34](https://arxiv.org/abs/2404.15766)
[35](https://arxiv.org/abs/2405.20750)
[36](https://arxiv.org/abs/2410.04760)
[37](https://arxiv.org/abs/2405.14250)
[38](https://dl.acm.org/doi/10.1145/3664647.3680999)
[39](https://journals.jps.jp/doi/10.7566/JPSJ.94.031010)
[40](https://arxiv.org/abs/2409.18959)
[41](https://ieeexplore.ieee.org/document/10657160/)
[42](https://arxiv.org/pdf/2209.11215.pdf)
[43](https://arxiv.org/html/2410.03108v1)
[44](http://arxiv.org/pdf/2409.07032.pdf)
[45](https://arxiv.org/html/2406.13652v1)
[46](http://arxiv.org/pdf/2402.07487.pdf)
[47](http://arxiv.org/pdf/2210.04872v2.pdf)
[48](https://arxiv.org/html/2411.07233v1)
[49](https://arxiv.org/html/2411.18702v1)
[50](https://www.emergentmind.com/topics/score-based-diffusion-model)
[51](https://openaccess.thecvf.com/content/CVPR2024/papers/Xue_Accelerating_Diffusion_Sampling_with_Optimized_Time_Steps_CVPR_2024_paper.pdf)
[52](https://arxiv.org/abs/2106.03802)
[53](https://arxiv.org/abs/2508.17868)
[54](https://projecteuclid.org/journals/statistics-surveys/volume-19/issue-none/Score-based-diffusion-models-via-stochastic-differential-equations/10.1214/25-SS152.full)
[55](https://openreview.net/forum?id=LOz0xDpw4Y)
[56](https://www.themoonlight.io/ko/review/score-based-diffusion-models-via-stochastic-differential-equations-a-technical-tutorial)
