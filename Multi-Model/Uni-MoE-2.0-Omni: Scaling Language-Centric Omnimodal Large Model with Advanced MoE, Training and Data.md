# Uni-MoE-2.0-Omni: Scaling Language-Centric Omnimodal Large Model with Advanced MoE, Training and Data

### 1. 핵심 주장과 주요 기여 요약[1]

**Uni-MoE-2.0-Omni**는 언어-중심의 옴니모달 대규모 모델(OLM)로서, 완전히 오픈소스로 공개된 시스템입니다. 이 모델은 Qwen2.5-7B 기반의 기본 LLM에서 출발하여, 세 가지 핵심 기여를 통해 이미지, 텍스트, 음성, 비디오 등 10개의 교차모달 입력을 처리할 수 있게 발전했습니다.[1]

**핵심 기여:**

1. **동적 수용력 혼합 전문가(Dynamic-Capacity MoE) 설계**: 공유 전문가, 라우팅 전문가, 널 전문가를 결합하여 계산 효율성과 모델 능력의 균형을 이루며, 10개의 교차모달 입력을 처리합니다.[1]

2. **점진적 훈련 전략**: 교차모달 사전학습, 전문가 웜업, MoE 미세조정, 강화학습, 생성 훈련의 단계적 진행을 통해 훈련 안정성을 보장합니다.[1]

3. **멀티모달 데이터 매칭 기법**: 약 75B 토큰의 오픈소스 멀티모달 데이터를 활용하여 음성 및 이미지 생성 토큰을 특수화함으로써 언어 단서에 따른 조건부 생성을 가능하게 합니다.[1]

**성능 하이라이트**: 85개 벤치마크를 통한 평가에서 1.2T 토큰으로 훈련된 Qwen2.5-Omni를 76개 벤치마크 중 50개 이상에서 초과 달성했으며, 특히 비디오 이해(+7% 평균), 옴니모달성 이해(+7% 평균), 음향시각적 추론(+4%)에서 강력한 성능을 보입니다.[1]

***

### 2. 문제 정의, 제안 방법, 모델 구조, 성능 향상 및 한계

#### **2.1 해결하고자 하는 문제**[2][3][1]

현존하는 옴니모달 모델들은 다음과 같은 핵심 과제들을 직면하고 있습니다:

**구조적 한계**: 많은 기존 옴니모달 모델들은 깊은 맥락 이해와 고충실도 콘텐츠 생성 사이의 내재적 긴장을 해결하지 못합니다. 예를 들어, 일부 모델은 오디오 전용 Qwen-Omni나 Baichuan-Omni처럼 강력한 맥락 이해는 가능하지만 생성 능력이 부족하며, 반대로 OmniGen이나 Janus-Pro 같은 모델들은 생성에 우수하지만 좁은 모달리티 범위에만 제한됩니다.[1]

**효율성 문제**: 간단히 기존의 조밀한 변환기(dense transformer)를 확장하는 것은 계산상 비현실적이며, 수십 개의 작업과 다중 모달 상호작용을 균형있게 처리하기 어렵습니다.[1]

**훈련 불안정성**: 이질적인 데이터에 대한 대규모 훈련 시 MoE 기반 옴니모달 모델의 훈련 불안정성이 주요 장애물입니다.[1]

**모달리티 바인딩**: 모든 모달리티를 통합된 대기공간(latent space)에 효과적으로 정렬하는 것이 어렵습니다.[4]

#### **2.2 제안하는 방법 및 수식**[5][6][7][1]

**A. 라우팅 그래디언트 추정(Routing Gradient Estimation)**[7][1]

전통적인 MoE 모델의 핵심 문제는 Top-K 연산의 미분 불가능성입니다. 이를 해결하기 위해 직선통과 그래디언트 추정기(straight-through gradient estimator)를 상미분방정식(ODE) 프레임워크 내에서 통합합니다.[7][1]

$$\text{TopK}(z)_i := \begin{cases} 1 & \text{if } z_i \text{ is among TopK coordinates} \\ 0 & \text{otherwise} \end{cases}$$

이 방법은 Grin-MoE에서 제안된 전략을 채택하여 라우터와 전문가 모두의 엔드-투-엔드 최적화를 가능하게 합니다.[7]

**B. 동적 용량 라우팅(Dynamic Capacity Routing)**[1]

기존 MoE는 모든 토큰에 고정된 수의 전문가를 적용하지만, 토큰의 복잡도와 지식 요구는 다양합니다. 동적 용량 라우팅은 Top-P 샘플링을 기반으로 합니다:[1]

라우터가 라우팅 전문가에 대한 확률 벡터를 생성한다고 하면:

```math
p^{(i)} = [p^{(i)}_1, p^{(i)}_2, \ldots, p^{(i)}_{N_r}], \quad \sum_{j=1}^{N_r} p^{(i)}_j = 1
```

여기서 $N_r$은 라우팅 전문가의 수입니다. 전문가들을 $p^{(i)}_j$의 내림차순으로 정렬하면:

$$p^{(i)}_{\pi^{(i)}(1)} \geq p^{(i)}_{\pi^{(i)}(2)} \geq \cdots \geq p^{(i)}_{\pi^{(i)}(N_r)}$$

토큰 $i$에 대해 활성화된 라우팅 전문가의 집합은:

```math
R_i = \left\{ \pi^{(i)}(1), \ldots, \pi^{(i)}(k_i) \right\}
```

```math
k_i = \min \left\{ k \mid \sum_{j=1}^{k} p^{(i)}_{\pi^{(i)}(j)} \geq P \right\}
```

여기서 $P$는 누적 확률 임계값(예: $P = 0.7$)입니다.[1]

**C. 옴니-모달 3D RoPE(Rotary Positional Embedding)**[8][1]

모든 모달리티를 통합적으로 처리하기 위해 원래의 회전 임베딩을 시간, 높이, 너비의 세 가지 성분으로 분해합니다:[8][1]

텍스트 입력의 경우, 표준 1D-RoPE와 동등하게 작동하지만, 음성 입력의 경우 시간 위치 ID를 절대 시간과 정렬합니다. 20개 토큰을 최소 시간 단위로 정의하면 3초에 해당합니다.[1]

비디오 입력의 경우, 각 프레임의 시간 ID는 절대 시간에 따라 증가하고, 높이와 너비 성분은 패치 방식으로 할당됩니다. 예를 들어, 120초 비디오와 오디오 입력의 경우:[1]

- 첫 번째 비디오 프레임: RoPE ID = $(x, x, x)$에서 $(x, x+p, x+p)$로 증가 (여기서 $p$는 비디오 프레임의 높이와 너비 토큰 수)
- 두 번째 프레임: $(x + 2\theta, x, x)$에서 $(x + 2\theta, x+p, x+p)$로 증가 (여기서 $\theta$는 절대 시간의 특정 스케일 팩터)

**D. 혼합 전문가 아키텍처의 세 가지 전문가 역할**[1]

1. **라우팅 전문가(Routed Experts)**: 동적으로 활성화되는 작업별 전문가로, 도메인 특정 지식을 모델링합니다.
2. **공유 전문가(Shared Experts)**: 모든 토큰에 대해 지속적으로 활성화되는 도메인 독립적 전문가로, 공통 지식 백본을 유지합니다.
3. **널 전문가(Null Experts)**: 출력이 정확히 0인 "빈" 전문가로, 선택적 망각을 위해 기능합니다.

**E. 언어 중심 하이브리드 훈련**[9][1]

이미지 편집, 이미지 생성, 음성 합성 등의 생성 작업을 언어 생성 프레임워크 내에서 통합합니다:[1]

$$L_{\text{total}} = L_{\text{understanding}} + \lambda_{\text{gen}} L_{\text{generation}}$$

여기서 이해 작업과 생성 작업이 시너지적으로 상호 강화됩니다.[1]

#### **2.3 모델 구조**[1]

모델은 다음과 같은 핵심 구성 요소로 이루어져 있습니다:[1]

| 모듈 | 아키텍처 | 파라미터 |
|------|---------|---------|
| 음성 인코더 | Whisper-Large-v3 | 637M |
| 비전 인코더 | SigLIP-So400M | 398M |
| MoE-LLM | MoE 변환기 | 26B |
| MoE-TTS | MoE 변환기 | 0.7B~1.2B |
| 작업-인식 DiT | 조밀 변환기 | 1.5B |
| 공유 전문가 | MLP | 712M |
| 라우팅 전문가 | MLP | 5.7B |
| 활성화된 전문가 | 2 공유 + 0~3 라우팅 | 1.5B~18B |

**음성 생성 (Context-Aware MoE-TTS)**[1]

음성 생성은 세 가지 단계를 거칩니다:[1]

$$s_0 = [P_1, \ldots, P_n; T_1, \ldots, T_m]$$

$$S_S^l = \text{MSA}(\text{LN}(s_{l-1})) + s_{l-1}$$

$$S_M^l = \text{MoE}(\text{LN}(X_s^l)) + X_s^l$$

$$s_l = \text{LN}(S_M^l)$$

여기서 $P_i$는 프롬프트 텍스트, $T_i$는 목표 텍스트, MSA는 멀티헤드 자기주의, MoE는 혼합 전문가입니다.[1]

**이미지 생성 (작업-인식 Diffusion Transformer)**[1]

Task-Aware Diffusion Transformer는 경량 브릿지 역할을 하며, 두 가지 학습 가능한 토큰 집합을 사용합니다:[1]

- **작업 토큰 (<TASK[i]>)**: 텍스트-이미지, 편집, 저수준 이미지 처리 등의 고수준 명령 인코딩
- **이미지 토큰 (<IMG[i]>)**: Uni-MoE-2.0-Omni에서 원하는 출력의 의미론적 본질 캡처

#### **2.4 성능 향상 분석**[10][11][12][1]

**비전-언어 이해:**[1]
- 일반 이미지 이해: MMBench-EN 80.50%, GQA 62.18% 달성
- 영상 이해: Video-MME에서 66.4% (Ming-Lite-Omni-1.5 대비 +3.85%), VSI-Bench에서 56.0% (Qwen2.5-Omni 대비 +36.7% 초과)
- STEM 이해: MathVision 36.61% 달성

**음성-언어 작업:**[1]
- 영어 ASR: LibriSpeech-clean 1.66% WER, LibriSpeech-clean-long 2.04% WER
- 중국어 ASR: Aishell1 3.23% WER, Aishell2 4.94% WER
- 텍스트-음성: LibriTTS-clean 5.85% WER
- 음성-음성 QA: LlamaQA(s→s) 75.33% 정확도

**옴니모달 이해:**[1]
- WorldSense: 44.7% (SOTA)
- OmniVideoBench: 35.1% (SOTA)
- StreamingBench (Omni): 48.1% (두 번째)
- OmniBench: 47.1% (두 번째)

**이미지 생성 및 편집:**[1]
- GEdit-Bench: 6.02 (PixWizard 대비 +88.1%)
- Emu Edit: 0.076 (PixWizard 대비 +94.8%)

**한계:**[1]
- 문서/OCR 작업에서 성능 격차 존재 (사전 훈련 중 제한된 데이터)
- 음악 이해 전문 영역에서 높은 품질의 음악 주석 데이터 부족으로 인한 성능 한계
- 순수 이미지 생성의 자동 메트릭상 제한된 성능 (외부 확산 모델 제약)

***

### 3. 모델의 일반화 성능 향상 가능성

#### **3.1 동적 라우팅을 통한 일반화 개선**[13][14][1]

동적 라우팅은 토큰 복잡도에 따라 활성화되는 전문가 수를 조정함으로써 모델의 일반화 능력을 향상시킵니다.[14][13][1]

**Figure 8 분석**: Top-P 라우팅 메커니즘의 동적 계산 예산 할당 시각화에서 "피크-트로-피크-폴" 패턴이 관찰되어 다양한 모달리티에 맞춤형 자원 할당이 가능함을 보여줍니다:[1]

- **초기 층(1-3)**: 일반 목적의 특성 추출을 위한 높은 부하
- **중간 층(7-21)**: 복잡한 추론과 특성 통합을 위한 주요 피크
- **심층(21-27)**: 출력으로의 수렴에 따른 최종 감소

시간 입력(비디오, 음성)은 정적 이미지보다 초기에 더 강한 피크를 보이며 더 많은 병렬 자원이 필요함을 시사합니다.[1]

#### **3.2 멀티모달 상호작용을 통한 일반화**[13][14][1]

최근 연구에 따르면 MoE 기반 모델의 라우팅을 시간 변화하는 멀티모달 상호작용으로 안내할 수 있으며, 이는 전문가 특화를 개선하고 작업별 특성 학습보다는 상호작용 처리 기술을 보다 일반화 가능하게 학습하도록 격려합니다.[14][13]

#### **3.3 점진적 훈련 전략의 효과**[15][9][1]

Uni-MoE-2.0-Omni의 점진적 훈련 전략은 다음과 같은 일반화 개선을 제공합니다:[1]

1. **교차모달 사전학습**: 이미지-텍스트, 음성-텍스트 등의 쌍형 데이터를 통해 기본 정렬 학습
2. **전문가 웜업**: 모달리티별 전문가를 먼저 독립적으로 훈련
3. **MoE 미세조정**: 모든 모달리티 혼합 데이터에 대한 통합 훈련
4. **강화학습(GSPO-DPO)**: 추론 능력 개선
5. **생성 훈련**: 음성 및 이미지 생성 능력 추가

이러한 단계적 접근은 약 75B 토큰이라는 제한된 데이터로도 1.2T 토큰 모델보다 우수한 성능을 달성할 수 있게 합니다.[1]

#### **3.4 텍스트 기반 추론 강화의 효과**[16][17]

최근 연구(OThink-MR1, LMM-R1)에 따르면, 텍스트만 포함된 작업에서의 강화학습 훈련이 멀티모달 추론의 일반화 능력을 크게 향상시킵니다:[17]

- **같은 작업 내 개선**: GRPO-D를 통해 Qwen2-VL-2B에서 평균 13.59% 개선
- **교차 작업 일반화**: SFT 대비 평균 61.63%의 상대 개선
- **추론에서 이해로의 일반화**: +63.53%
- **이해에서 추론으로의 일반화**: +368.99%

이는 Uni-MoE-2.0-Omni의 GSPO-DPO 반복 최적화 전략과 유사한 메커니즘입니다.[1]

#### **3.5 장시간 맥락 이해와 프레임 샘플링**[18]

최근 LongInsightBench 연구에 따르면, 모델의 일반화 성능은 시각적 프레임 샘플 밀도에 따라 달라집니다:[18]

- **Ola-7B**: 32프레임에서 53.64%에서 128프레임에서 56.96%로 일관된 개선 (3.32%p 향상)
- **VideoLLaMA3**: 39.29%에서 43.45%로 개선 (4.16%p 향상)
- **Qwen2.5-Omni-7B**: 51.35%에서 51.98%로 포화 (0.63%p 개선 후 정체)

Uni-MoE-2.0-Omni는 동적 라우팅의 Top-P 메커니즘을 통해 다양한 복잡도의 입력을 효율적으로 처리할 수 있는 구조로 설계되어, 더 나은 일반화 성능이 예상됩니다.[1]

#### **3.6 도메인 일반화를 위한 모달리티별 구성**[19][20][1]

Uni-MoE-2.0-Omni의 MoE 아키텍처는 SimMMDG 프레임워크의 원칙을 따릅니다:[20][19]

모든 모달리티를 동일한 임베딩 공간에 매핑하는 것이 일반화를 방해한다는 관찰에 기반하여, Uni-MoE-2.0-Omni는 모달리티별 라우팅 전문가를 유지함으로써 다음을 가능하게 합니다:[1]

1. **모달리티 특화**: 각 모달리티의 고유한 특성 보존
2. **교차모달 전이**: 공유 전문가를 통한 일반 지식 전이
3. **도메인 적응**: 새로운 작업이나 분포에 대한 빠른 적응

***

### 4. 향후 연구에 미치는 영향과 고려사항

#### **4.1 아키텍처 설계에 미치는 영향**[3][4][1]

**언어 중심 설계의 정당성**: Uni-MoE-2.0-Omni는 언어를 세계의 구조적 표현과 모달리티 간 중재자로 사용하는 기본 설계 원칙을 확립합니다. 이는 다음을 시사합니다:[1]

1. 순수 엔드-투-엔드 구조보다는 언어 중심의 하이브리드 접근이 이해와 생성의 균형 유지에 더 효과적
2. 생성 작업(이미지 생성, 음성 합성)은 별도의 전문화된 모듈로 유지하면서 기본 모델과 느슨하게 결합하는 것이 유리
3. 음성 생성의 경우 음소 단계 예측(next-token prediction)과는 다른 시간 세분성을 가지므로 외부화 필요[1]

**MoE의 적응적 활성화**: 고정된 전문가 수 대신 동적 할당은 다음과 같은 연구 방향을 제시합니다:[13][1]

- 토큰 복잡도 측정 메커니즘 개선
- 모달리티별, 작업별 맞춤형 활성화 패턴 학습
- 계산 효율성과 성능 간의 최적 균형점 찾기

#### **4.2 훈련 방법론 발전**[9][17][1]

**점진적 훈련의 일반화**: Uni-MoE-2.0-Omni의 성공은 다음과 같은 훈련 패러다임 변화를 시사합니다:[9][1]

$$\text{Loss}_{\text{total}} = \sum_{i=1}^{n} \alpha_i L_i + \lambda_{RL} L_{RL}$$

여기서 각 단계 $i$는 서로 다른 손실 가중치 $\alpha_i$를 가지며, 강화학습 단계는 특히 불안정성을 제어하기 위해 DPO 정규화가 필요합니다:[1]

1. **다단계 최적화**: 단순 엔드-투-엔드 훈련보다는 검증된 다단계 접근이 더 안정적
2. **반복적 RLHF**: GSPO-DPO 반복 최적화가 MoE 기반 모델의 추론 능력을 특히 개선
3. **데이터 균형 및 품질**: 75B 토큰이라는 제한된 데이터에서도 신중한 데이터 매칭과 품질 필터링을 통해 우수한 성능 달성 가능

**교차모달 정렬 개선**: 최근 연구(Revisiting Multimodal Positional Encoding)에 따르면, RoPE 설계에서 다음이 중요합니다:[8]

1. **위치 일관성**: 모든 모달리티에서 명확하고 구분되는 좌표 할당
2. **주파수 완전 활용**: 모든 위치 축이 전체 주파수 스펙트럼에 접근
3. **텍스트 선행 보존**: 사전학습된 LLM의 표준 RoPE 유지로 전이학습 효율성 보장[8]

#### **4.3 멀티모달 일반화의 새로운 과제**[4][18][1]

**퓨전 결손(Fusion Deficit) 해결**: LongInsightBench 연구에서 발견된 패러독스는 구체적인 연구 방향을 제시합니다:[18]

시각 정보가 고품질 텍스트 설명으로 제공될 때 성능이 더 나음 (0.6965)이 원본 시각+음성 데이터 (0.6517)보다 높음을 보이며, 이는 다음을 시사합니다:[18]

1. 현재 MoE 기반 멀티모달 융합 메커니즘의 근본적 한계 존재
2. 시간 변화하는 멀티모달 상호작용을 명시적으로 고려한 라우팅이 필요
3. 모달리티별 압축 수준의 최적화 필요

**시간적 정렬 및 장시간 맥락**: 비디오와 음성의 정렬은 다음과 같은 개선이 필요합니다:[8][1]

1. **Omni-Modality 3D RoPE의 확장**: 음성-비디오 동기화 메커니즘 강화
2. **프레임 샘플링 적응화**: 콘텐츠 중요도에 따른 동적 샘플링
3. **장시간 추론**: 3분 이상의 장시간 음성 및 비디오 이해 성능 향상

#### **4.4 오픈소스 생태계에 미치는 영향**[2][3][1]

**재현성과 신뢰성**: Uni-MoE-2.0-Omni의 완전한 오픈소스 공개(코드, 체크포인트, 데이터 리스트)는 다음을 가능하게 합니다:[1]

1. 옴니모달 모델의 투명한 개발 및 검증
2. 커뮤니티 기여 및 개선
3. 기술 격차 해소를 통한 공정한 AI 발전

**지속적 발전 방향**:[1]

1. **전문가 특화 최적화**: 대규모 조밀 모델 증류를 통한 더 효율적인 MoE 구성
2. **조건부 라우팅**: 더 효율적이고 제어 가능한 다중 화자 음성 합성
3. **비디오 데이터 확장**: 옴니모달 이해 능력의 추가 향상
4. **새로운 위치 인코딩**: 더 나은 멀티모달 정렬을 위한 혁신적 방법 탐색

#### **4.5 실제 응용 고려사항**[21][22][23][1]

**저수준 이미지 처리와 생성의 정교화**: Uni-MoE-2.0-Omni는 이미지 편집과 저수준 처리에서 특히 강력하지만:[1]

1. 다양한 이미지 처리 작업에 대한 모델의 해석 능력 향상 필요
2. 조건부 생성의 더 정밀한 제어 메커니즘 개발
3. 다중 단계 확산 과정과 단일 토큰 예측 모델 간의 격차 해소

**음성 생성의 실시간 상호작용**: 스트리밍 음성 생성의 품질 향상을 위해:[1]

1. 문맥을 고려한 MoE-TTS 개선
2. 길이 처리 및 음성 연속성 최적화
3. 실시간 음성 상호작용에서의 응답 지연 최소화

#### **4.6 벤치마크 및 평가 방법론의 발전**[24][25][18][1]

새로운 도전과제 제시:

1. **OmniBench와 같은 통합 벤치마크의 확장**: 더 복잡한 크로스모달 추론 작업 포함
2. **영상-음성 동기화 평가**: VSI-Bench와 같은 공간-시간 추론 벤치마크의 개선
3. **일반화 능력 평가**: 학습 분포 외 성능 측정의 체계화
4. **다언어 옴니모달 평가**: 언어 간 모달리티 전이 능력 평가

#### **4.7 한계 극복을 위한 미래 연구**[1]

**데이터 부족 문제 해결**:

1. 고품질 문서/OCR 데이터 수집의 필요성
2. 음악 이해 및 전문 오디오 작업에 대한 공개 데이터셋 개발
3. 합성 데이터 생성 기법의 개선

**모달리티 간 불균형**:

- 현재 모델은 약 75B 토큰을 사용하여 훈련되었으나, 모달리티별 데이터 분포가 불균등
- 각 모달리티의 내재적 특성을 고려한 훈련 데이터 비율 최적화 필요

**계산 효율성 극대화**:

- 동적 라우팅의 추론 시간 오버헤드 감소
- 모바일 및 엣지 장치 배포를 위한 양자화 및 압축 기법 개발

***

### 결론

**Uni-MoE-2.0-Omni**는 오픈소스 옴니모달 모델 개발의 새로운 기준을 제시합니다. 동적 용량 MoE, 점진적 훈련 전략, 언어 중심의 하이브리드 설계를 통해, 제한된 자원(75B 토큰)으로도 대규모 모델(1.2T 토큰)을 초과하는 성능을 달성합니다. 특히 비디오 이해, 장시간 음성 처리, 크로스모달 추론에서의 우수한 성능은 미래 옴니모달 AI 시스템의 설계 원칙을 제시합니다.

그러나 문서 처리, 음악 이해, 순수 이미지 생성과 같은 특정 도메인에서의 성능 한계는 향후 연구의 명확한 방향을 제시합니다. 시간 변화하는 멀티모달 상호작용의 명시적 활용, 퓨전 결손의 이론적 해결, 그리고 더욱 효율적인 동적 라우팅 메커니즘은 다음 세대 옴니모달 모델 개발의 핵심 과제입니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0dbb3b5a-c45e-486a-b43f-a27b2d48d218/2511.12609v2.pdf)
[2](https://arxiv.org/pdf/2501.04561.pdf)
[3](https://arxiv.org/pdf/2310.18390.pdf)
[4](https://www.emergentmind.com/topics/omni-modal-large-language-models-ollms)
[5](https://arxiv.org/pdf/2310.00811.pdf)
[6](https://arxiv.org/pdf/2403.07652.pdf)
[7](https://arxiv.org/html/2511.12609v2)
[8](https://arxiv.org/html/2510.23095v1)
[9](https://peninsula-press.ae/Journals/index.php/EDRAAK/article/view/188)
[10](https://journal.aripafi.or.id/index.php/tritunggal/article/view/706)
[11](https://jurnal.sttarastamarngabang.ac.id/index.php/Corammundo/article/view/398)
[12](http://pubs.rsna.org/doi/10.1148/radiol.250617)
[13](https://arxiv.org/abs/2509.25678)
[14](https://openreview.net/forum?id=qF9WJxvHX8)
[15](https://arxiv.org/abs/2509.19745)
[16](https://aiflower.tistory.com/183)
[17](https://changwangzhang.github.io/files/2025-arxiv-othink-mr1.pdf)
[18](https://arxiv.org/html/2510.17305v2)
[19](https://openreview.net/forum?id=RiSMijlsLT)
[20](https://proceedings.neurips.cc/paper_files/paper/2023/file/f88bec15cc4cb56b432ee040bb63f94f-Paper-Conference.pdf)
[21](https://arxiv.org/pdf/2502.04328.pdf)
[22](https://arxiv.org/pdf/2309.05519.pdf)
[23](https://arxiv.org/html/2502.15803v1)
[24](https://arxiv.org/pdf/2409.15272v3.pdf)
[25](https://vision-x-nyu.github.io/thinking-in-space.github.io/)
[26](https://arxiv.org/abs/2503.00025)
[27](https://conference.bicone.id/index.php/bicone/article/view/46)
[28](https://journals.lww.com/10.1097/AOG.0000000000006121)
[29](https://academic.oup.com/icesjms/article/doi/10.1093/icesjms/fsaf054/8120027)
[30](https://jurnal.stkippersada.ac.id/jurnal/index.php/JPDP/article/view/5427)
[31](https://ejurnal.stpkat.ac.id/index.php/jutipa/article/view/368)
[32](https://arxiv.org/html/2407.11895)
[33](http://arxiv.org/pdf/2404.06212.pdf)
[34](https://proceedings.neurips.cc/paper_files/paper/2024/file/4a3a14b9536806a0522930007c5512f7-Paper-Conference.pdf)
[35](https://www.sciencedirect.com/science/article/abs/pii/S1566253525006578)
[36](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_Multi-Modal_Gated_Mixture_of_Local-to-Global_Experts_for_Dynamic_Image_Fusion_ICCV_2023_paper.pdf)
[37](https://huggingface.co/papers/2502.04328)
[38](https://openreview.net/forum?id=NdSygrpDPZ)
[39](https://editoncpublishing.org/ecpj/index.php/ECJECS/article/view/627)
[40](https://ramlai.com/index.php/ramlai/article/view/18)
[41](http://arxiv.org/pdf/2409.12136.pdf)
[42](http://arxiv.org/pdf/2409.06669.pdf)
[43](https://arxiv.org/pdf/2206.03382.pdf)
[44](https://arxiv.org/html/2501.10714v1)
[45](https://arxiv.org/pdf/2312.09877.pdf)
[46](https://arxiv.org/html/2503.16057)
[47](https://aclanthology.org/2025.emnlp-main.997.pdf)
[48](https://www.cs.cmu.edu/~leili/pubs/ye2021end.pdf)
[49](https://arxiv.org/html/2511.12609v1)
[50](https://pmc.ncbi.nlm.nih.gov/articles/PMC12558867/)
[51](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1590994/full)
[52](https://huggingface.co/papers?q=multimodal-to-speech+generation)
[53](https://www.marktechpost.com/2025/11/17/uni-moe-2-0-omni-an-open-qwen2-5-7b-based-omnimodal-moe-for-text-image-audio-and-video-understanding/)
[54](https://arxiv.org/html/2509.15964v1)
[55](https://dl.acm.org/doi/10.1145/3711896.3737409)
[56](https://bmcoralhealth.biomedcentral.com/articles/10.1186/s12903-025-06619-6)
[57](https://aacrjournals.org/cancerres/article/85/8_Supplement_1/684/755181/Abstract-684-Multi-site-external-reproducibility)
[58](https://jamanetwork.com/journals/jamaophthalmology/fullarticle/2841079)
[59](https://arxiv.org/abs/2505.20612)
[60](https://dl.acm.org/doi/10.1145/3746270.3760219)
[61](https://arxiv.org/abs/2509.00731)
[62](https://ieeexplore.ieee.org/document/11206330/)
[63](https://www.semanticscholar.org/paper/f2866ca95b940f8c7da1901c6817674343b2938b)
[64](https://arxiv.org/pdf/2309.16609.pdf)
[65](https://arxiv.org/abs/2503.20215)
[66](https://arxiv.org/pdf/2412.15115.pdf)
[67](https://arxiv.org/pdf/2409.12186.pdf)
[68](https://arxiv.org/pdf/2502.16137.pdf)
[69](https://arxiv.org/pdf/2502.07374.pdf)
[70](http://arxiv.org/pdf/2411.07238.pdf)
[71](https://arxiv.org/html/2412.05210v1)
[72](https://slashdot.org/software/comparison/Qwen2.5-Max-vs-Qwen3-Omni/)
[73](https://www.emergentmind.com/topics/uni-moe-2-0-omni-model)
[74](https://fornewchallenge.tistory.com/entry/%F0%9F%91%80%F0%9F%91%82%F0%9F%97%A3%EF%B8%8F%E2%9C%8D%EF%B8%8FQwen25-Omni-%EB%B3%B4%EA%B3%A0-%EB%93%A3%EA%B3%A0-%EB%A7%90%ED%95%98%EA%B3%A0-%EC%93%B0%EB%8A%94-%EC%B0%A8%EC%84%B8%EB%8C%80-%EB%A9%80%ED%8B%B0%EB%AA%A8%EB%8B%AC-%EB%AA%A8%EB%8D%B8)
[75](https://arxiv.org/html/2509.14142v1)
[76](https://github.com/QwenLM/Qwen2.5-Omni)
[77](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)
[78](https://openaccess.thecvf.com/content/ICCV2025W/MARS2/papers/Xu_MARS2_2025_Challenge_on_Multimodal_Reasoning_Datasets_Methods_Results_Discussion_ICCVW_2025_paper.pdf)
[79](https://apidog.com/blog/qwen2-5-omni-7b/)
[80](https://research.ibm.com/publications/on-the-generalization-capacity-of-neural-networks-during-generic-multimodal-reasoning)
