# A Closer Look at Parameter-Efficient Tuning in Diffusion Models

### 1. 논문의 핵심 주장과 주요 기여

**"A Closer Look at Parameter-Efficient Tuning in Diffusion Models"** (Xiang et al., 2023)의 핵심 주장은 대규모 확산 모델(Stable Diffusion)을 개인화된 작업에 맞게 효율적으로 조정할 때, **어댑터(adapter)의 입력 위치가 성능을 결정하는 가장 중요한 요소**라는 것입니다.[1]

**주요 기여는 다음과 같습니다:**

- **체계적 설계 공간 분해**: U-Net 기반 확산 모델의 어댑터 배치를 입력 위치, 출력 위치, 함수 형태의 세 가지 직교 요소로 분해[1]
- **ANOVA를 통한 정량 분석**: Analysis of Variance 통계 기법을 적용하여 각 설계 요소의 상대적 영향도 측정 (입력 위치의 F-통계값 약 10으로 가장 높음)[1]
- **최적 배치 발견**: cross-attention 블록 이후에 어댑터를 삽입할 때 최고 성능 달성[1]
- **파라미터 효율성**: 전체 U-Net 파라미터의 **0.75%만을 추가**하면서 DreamBooth의 완전 미세조정과 동등 이상의 성능[1]

***

### 2. 해결하고자 하는 문제

**문제의 배경:**
- Stable Diffusion 같은 대규모 확산 모델의 **전체 파라미터 미세조정**은 메모리 비효율적이고 계산 비용이 높음[1]
- 사용자가 개별 물체나 스타일로 모델을 개인화하려면 수천 개의 파라미터 업데이트 필요[1]
- 기존 NLP의 어댑터 기법이 확산 모델의 복잡한 U-Net 아키텍처(residual 블록, 다중 attention, down/up-sampling)에는 충분히 탐색되지 않음[1]

**구체적 과제:**
- U-Net 아키텍처에서 10개의 입력 위치 후보, 7개의 출력 위치, 여러 함수 형태 조합으로 인한 **큰 설계 공간**[1]
- 어떤 조합이 downstream 작업 성능에 가장 큰 영향을 미치는지 불명확[1]

***

### 3. 제안하는 방법

#### 3.1 어댑터 설계 공간 분해

**입력 위치(Input Position):** 10가지 옵션
- SAin (Self-Attention 입력)
- CAin, CAc, CAout (Cross-Attention 관련)
- FFNin, FFNout (Feed-Forward Network)
- Resin, Resout (Residual 블록)
- Transout (Transformer 블록 출력)[1]

**출력 위치(Output Position):** 7가지 옵션 (덧셈 교환 법칙으로 축약)[1]

**함수 형태(Function Form):**
- **Transformer 블록 어댑터**: 저랭크 행렬 사용
  
$$W_{down} \in \mathbb{R}^{d \times r}, \quad W_{up} \in \mathbb{R}^{r \times d}$$

- **Residual 블록 어댑터**: 3×3 컨볼루션 계층
  
$$Conv_{down}, \quad Conv_{up}, \quad \text{Group Normalization}$$

- **활성화 함수**: ReLU, Sigmoid, SiLU, Identity[1]
- **스케일 팩터**: 0.5, 1.0, 2.0, 4.0[1]

#### 3.2 ANOVA 기반 요소 분석

모델 성능을 $Y$라 하고, 설계 요소를 $X_1, X_2, ..., X_k$라 할 때:

$$F\text{-statistic} = \frac{MSB}{MSE} = \frac{\sum_i n_i(\bar{Y}_i - \bar{Y})^2 / (k-1)}{\sum_i \sum_j (Y_{ij} - \bar{Y}_i)^2 / (n-k)}$$

여기서 $MSB$는 그룹 간 분산, $MSE$는 그룹 내 분산[1]

**실험 결과:**
- 입력 위치: F-통계값 $\approx 10$ (매우 유의미)[1]
- 출력 위치: F-통계값 낮음 (약한 상관관계)[1]
- 함수 형태(활성화, 스케일): F-통계값 $\approx 1$ (무의미)[1]

#### 3.3 최적 어댑터 배치

**최고 성능 설정:**
- **입력 위치**: CAout (Cross-Attention 출력)[1]
- **출력 위치**: FFNin (Feed-Forward Network 입력)[1]
- **이유**: 텍스트 조건 변화를 최대한 감지하고, prompt 인식을 강화[1]

**시각화 검증:**
노이즈 예측 차이를 시각화하면, CAout/CAin에 위치한 어댑터가 prompt 변화에 가장 민감하게 반응[1]

$$\Delta \epsilon = \epsilon(x_t | \text{prompt}_A) - \epsilon(x_t | \text{prompt}_B)$$

***

### 4. 모델 구조 (U-Net 기반 Stable Diffusion)

**전체 아키텍처:**

$$\text{UNet} = \text{EncoderBlocks} + \text{BottleneckBlock} + \text{DecoderBlocks}$$

**각 기본 블록 구조:**

$$\text{BasicBlock} = \text{ResidualBlock} + \text{TransformerBlock}$$

**Transformer 블록 내부:**

$$\text{TransformerBlock} = \text{SelfAttention} + \text{CrossAttention} + \text{FFN}$$

**Self-Attention:**

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

여기서 $Q, K, V \in \mathbb{R}^{n \times d_k}$[1]

**Cross-Attention:**

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

여기서 $Q \in \mathbb{R}^{n \times d_k}$ (이미지 특성), $K, V \in \mathbb{R}^{m \times d_k}$ (텍스트 조건)[1]

**Feed-Forward Network:**

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

여기서 $W_1 \in \mathbb{R}^{d \times d_m}$, $W_2 \in \mathbb{R}^{d_m \times d}$[1]

**Residual 블록:**
- 시간 임베딩 주입
- 컨볼루션 계층
- 활성화 함수

***

### 5. 성능 향상

#### 5.1 정량적 성과

**DreamBooth 개인화 작업:**
- **CLIP 유사도**: 우수 방법 0.899 vs DreamBooth 0.841 (+6.9%)[1]
- **파라미터 수**: 0.75% (약 1.5M 파라미터)[1]
- **메모리 사용량**: 약 30% 감소[1]
- **훈련 시간**: 약 30% 단축[1]

**꽃 데이터셋 미세조정 작업:**
- **FID 점수**: 24.49 (우수) vs 28.15 (완전 미세조정)[1]
- **수렴 속도**: 더 빠른 수렴[1]

#### 5.2 시각적 품질

생성된 이미지에서:
- 개인화된 객체의 **충실도(fidelity)** 우수[1]
- **배경 오염 최소화** (regularization 데이터로 인한 혼동 방지)[1]
- **다양성 유지** (모델 과적응 방지)[1]

***

### 6. 한계

#### 6.1 이론적 한계

- **설명 가능성 부족**: CAout이 최적인 이유에 대한 깊은 이론적 분석 미흡[1]
- **아키텍처 특화**: Stable Diffusion U-Net에만 최적화 (다른 아키텍처에 미지수)[1]
- **정적 배치**: 타임스텝별 동적 어댑터 위치 조정 미탐색[1]

#### 6.2 실험적 한계

- **제한된 작업**: 개인화와 꽃 데이터셋 미세조정만 평가 (더 다양한 작업 필요)[1]
- **모델 규모**: Stable Diffusion (1.4B)에만 검증 (더 큰 모델은 미검증)[1]
- **적응 형태**: 단일 어댑터 삽입만 고려 (다중 어댑터 상호작용 미탐색)[1]

#### 6.3 방법론적 한계

- **함수 형태 효과 미미**: 활성화 함수나 스케일 팩터의 영향이 제한적 (더 나은 선택 가능성)[1]
- **출력 위치 약한 상관**: 출력 위치의 역할 명확하지 않음[1]
- **계산 오버헤드**: ANOVA 분석 자체 비용[1]

***

### 7. 모델의 일반화 성능 향상 가능성

#### 7.1 Zero-shot 능력 보존

**핵심 발견**: 어댑터 기반 미세조정이 **원본 모델의 생성적 사전지식을 더 잘 보존**[참고, ]

- 완전 미세조정(DreamBooth)은 모델 과적응으로 인한 분포 외 성능 저하[참고]
- 어댑터 방식은 0.75% 파라미터만 수정하므로 기저 모델의 범용성 유지[1]

**정량 지표:**
- CLIP 유사도 (in-distribution): 약간 우수 (0.899 vs 0.841)[1]
- 다양한 프롬프트 테스트에서 일관된 성능 유지[1]

#### 7.2 구성적 일반화(Compositional Generalization)

**CLIP 기반 분석:**
- Diffusion 모델의 **다중모달 구성 추론 능력**: CLIP보다 우수[참고, ]
- 새로운 속성-객체 조합에 대한 외삽 능력[참고]

**관찰:**
- Cross-attention 이후 어댑터가 **텍스트 조건 변화를 명시적으로 학습**하므로, 새로운 프롬프트에 더 잘 적응[1]

```math
\Delta L_{\text{noise}} = \epsilon_{\text{adapted}}(x_t, \text{new\_prompt}) - \epsilon_{\text{base}}(x_t, \text{base\_prompt})
```

#### 7.3 분포 외 강건성

**Few-shot 설정에서:**
- 적은 훈련 데이터로 인한 과적응 위험 감소[참고]
- 원본 모델의 사전 지식이 정규화 역할[1]

**예상 성능:**
- 테스트 분포가 훈련 분포와 다를 때도 개인화 기능 유지 가능[1]

***

### 8. 최신 관련 연구 비교 분석 (2020년 이후)

| 연구 | 연도 | 방법 | 파라미터 비율 | 특징 | 일반화 성능 |
|------|------|------|---------------|------|-----------|
| **본 논문 (우리)** | 2023 | Cross-attention 후 어댑터 | **0.75%** | ANOVA 기반 설계 공간 분석 | 우수한 구성 일반화 |
| **DreamBooth** | 2022 | 완전 미세조정 + regularization | ~100% | 개인화 기준선 | 과적응 위험 |
| **LoRA** | 2023 | 저랭크 행렬 (Hu et al.) | 0.17-0.72% | 1-6MB 모델 크기 | 그룹 내 성능 우수 |
| **DiffFit** | 2023 | 바이어스+스케일 팩터만 | **0.12%** | 최소 파라미터 | 안정적 |
| **Diff-Tuning** | 2024 | Chain of Forgetting 이론 | ~0.5% | 역방향 과정 분석 | 26% 개선 |
| **StyleInject** | 2024 | 병렬 저랭크 행렬 | ~0.6% | 스타일 다양성 유지 | 텍스트-이미지 정렬 우수 |
| **SuperLoRA** | 2024 | LoRA 확장 프레임워크 | **0.1% 이하** | 그룹화, 텐서 분해 | 10배 효율 개선 |
| **T-LoRA** | 2024 | 타임스텝 의존 LoRA | 0.2-0.4% | 동적 랭크 조정 | 시간별 적응화 |
| **IP-Adapter** | 2023 | 분리된 cross-attention | 22M | 이미지 프롬프트 | 멀티모달 호환성 |

**비교 분석:**

1. **파라미터 효율성**: DiffFit (0.12%) > SuperLoRA (0.1%) > 본 논문 (0.75%)
   - 그러나 본 논문은 **체계적 분석 방법론** 제시

2. **성능**: 본 논문 (CLIP 0.899) ≈ DreamBooth (0.841) < LoRA/DiffFit
   - 파라미터 효율성과 성능의 최적 균형

3. **이론적 기여**: 본 논문 **> 대부분의 경쟁 방법**
   - ANOVA 기반 설계 공간 탐색이 **유일한 체계적 분석**

4. **일반화 능력**:
   - **본 논문**: 구성적 일반화 우수 (prompt 민감성)
   - **DiffFit**: 안정적이나 일반화 능력 평가 미흡
   - **Diff-Tuning**: 이론 기반 강건성 (+26%)
   - **IP-Adapter**: 멀티모달 일반화 (텍스트+이미지)

***

### 9. 미래 연구에 미치는 영향

#### 9.1 설계 공간 탐색의 선례

- **모더니즘 전환**: 임의적 선택 → **정량적 분석** 기반 설계[1]
- 다른 생성 모델 (VAE, GAN 등)의 어댑터 설계에도 ANOVA 방법론 적용 가능[1]
- **멀티모달 모델**의 어댑터 배치 최적화 연구 촉진[1]

#### 9.2 파라미터 효율적 학습의 새 방향

**동적 어댑터 배치:**
- 타임스텝 $t$별 다른 위치 사용 (현재 미탐색)
- 조건(condition)의 복잡도에 따른 적응형 배치[1]

**다중 어댑터 상호작용:**
- CAout + FFNin 외에 보조 어댑터 추가 시 효과 연구 필요[1]
- 어댑터 간 특성 충돌 가능성 분석[1]

#### 9.3 구성적 일반화 심화

**새로운 질문들:**

1. **어댑터 유연성**: 훈련되지 않은 속성-객체 조합에 일반화 가능한가?
   - 예: "붉은 강아지"로 훈련 → "붉은 고양이" 생성 가능?[참고]

2. **제로샷 능력 정량화:**
   - 어댑터 기반 모델의 zero-shot 성능을 체계적으로 평가[참고]
   - Diffusion Classifier의 분류 능력과 비교[참고]

3. **분포 외 강건성:**
   - 극단적인 도메인 시프트에 대한 적응력[참고]
   - Certification-based robustness 보장 가능성[참고]

#### 9.4 실무적 영향

**이미지 생성 서비스:**
- 클라우드 기반 개인화 모델 (효율적 배포)
- 엣지 디바이스에서의 실시간 적응 모델[1]

**데이터 희소 환경:**
- 의료 영상, 위성 이미지 등 특화 도메인 적응[1]
- 다언어 프롬프트 지원[1]

***

### 10. 향후 연구 시 고려할 점

#### 10.1 방법론적 개선

1. **설계 공간 확장**
   - 더 세밀한 위치 구분 (예: attention head별 어댑터)
   - 비선형 어댑터 함수 탐색[1]

2. **적응 전략 강화**
   - **태스크 조건부 어댑터**: 특정 작업에 맞춘 동적 배치
   - **계층적 어댑터**: 서로 다른 깊이에서 다른 전략 적용[참고]

3. **이론적 분석**
   - CAout 최적성의 기하학적/정보론적 설명[1]
   - 어댑터의 **수렴 속도** 이론 분석[1]

#### 10.2 평가 확대

1. **다양한 아키텍처**
   - DiT (Diffusion Transformer)[참고]
   - Autoregressive diffusion 모델
   - 이산 확산 모델 (discrete diffusion)[참고]

2. **광범위한 작업**
   - 이미지-이미지 번역 (정렬 미세조정)
   - 텍스트-비디오 생성[참고]
   - 3D 생성 모델 (Dream Fusion)[1]

3. **일반화 능력 벤치마크**
   - Zero-shot 평가 (새로운 도메인, 프롬프트)
   - Out-of-distribution 강건성 측정[참고]
   - Compositional 일반화 정량화[참고]

#### 10.3 실용적 고려사항

1. **성능-효율 트레이드오프**
   - 0.75% vs 0.12% (DiffFit) 선택 기준?
   - 대규모 모델 (Imagen, DALL-E 3)에서도 성립?[1]

2. **적응 안정성**
   - 훈련 후 모델 드리프트 (model drift) 모니터링
   - 재훈련 없이 온라인 적응 가능성[1]

3. **해석 가능성**
   - 어댑터가 학습한 특성 시각화 및 분석[1]
   - Attention map을 통한 프롬프트-이미지 상호작용 이해[참고]

#### 10.4 연구 커뮤니티 협력

- **오픈소스 공개**: GitHub (https://github.com/Xiang-cd/unet-finetune)[1]
- **벤치마크 개발**: 표준화된 평가 프로토콜
- **재현성**: 상세한 하이퍼파라미터 공개[1]

***

### 11. 결론

**"A Closer Look at Parameter-Efficient Tuning in Diffusion Models"**은 대규모 생성 모델의 효율적 적응을 위한 **체계적 설계 방법론**을 제시합니다. Cross-attention 이후 위치의 어댑터가 텍스트 조건에 명시적으로 반응하게 되어 **0.75% 파라미터로 완전 미세조정 성능 달성**이라는 실질적 성과를 거두었습니다.[1]

**핵심 의의:**
- **정량적 설계**: ANOVA를 통한 첫 체계적 분석[1]
- **경제적 효율성**: 메모리 30% 감소, 속도 개선[1]
- **일반화 보존**: 원본 모델의 구성적 일반화 능력 유지[1]

**미래 전망:**
- 더 나은 파라미터 효율성 (SuperLoRA 0.1%)과 본 연구의 **설계 공간 분석**의 결합
- 동적 어댑터, 타임스텝 의존성, 멀티태스크 학습으로 진화
- **의료, 위성, 다언어** 등 실무 도메인 적용 확대

***

### 참고문헌 표기

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4e16335f-a441-4495-b2ab-3b557066efd2/2303.18181v2.pdf)
[2](https://dl.acm.org/doi/10.1145/3730403)
[3](https://ieeexplore.ieee.org/document/10678270/)
[4](https://arxiv.org/abs/2403.11887)
[5](https://arxiv.org/abs/2406.00773)
[6](http://thesai.org/Publications/ViewPaper?Volume=15&Issue=12&Code=ijacsa&SerialNo=25)
[7](https://arxiv.org/abs/2411.11516)
[8](https://arxiv.org/abs/2407.01531)
[9](https://ieeexplore.ieee.org/document/10657194/)
[10](https://link.springer.com/10.1007/978-3-031-43412-9_5)
[11](https://arxiv.org/abs/2306.07290)
[12](https://arxiv.org/abs/2412.12953)
[13](http://arxiv.org/abs/2110.04366)
[14](http://arxiv.org/pdf/2405.15020.pdf)
[15](https://arxiv.org/pdf/2305.10924.pdf)
[16](http://arxiv.org/pdf/2303.07910.pdf)
[17](https://arxiv.org/pdf/2301.11660.pdf)
[18](https://arxiv.org/html/2412.12444v1)
[19](https://arxiv.org/pdf/2305.18455.pdf)
[20](https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_DiffFit_Unlocking_Transferability_of_Large_Diffusion_Models_via_Simple_Parameter-efficient_ICCV_2023_paper.pdf)
[21](https://github.com/cloneofsimo/lora/)
[22](https://kesbangpol.biakkab.go.id/wxc5gnzh/lora-dreambooth-vs-fine-tuning-vs-stable-diffusion.html)
[23](https://openreview.net/forum?id=6emETARnWi)
[24](https://www.reddit.com/r/MachineLearning/comments/zfkqjh/p_using_lora_to_efficiently_finetune_diffusion/)
[25](https://www.reddit.com/r/DreamBooth/comments/1c9634p/dreambooth_vs_full_finetune/)
[26](https://arxiv.org/html/2405.16876v2)
[27](https://huggingface.co/blog/lora)
[28](https://www.reddit.com/r/StableDiffusion/comments/11nnt1o/sd_finetuning_methods_compared_a_benchmark/)
[29](https://neurips.cc/virtual/2024/poster/95124)
[30](https://arxiv.org/html/2512.10877v1)
[31](https://arxiv.org/html/2507.05964v1)
[32](https://arxiv.org/html/2410.14265v1)
[33](https://arxiv.org/pdf/2502.04491.pdf)
[34](https://arxiv.org/html/2512.02899)
[35](https://arxiv.org/html/2505.03557v2)
[36](https://arxiv.org/pdf/2512.10877.pdf)
[37](https://arxiv.org/html/2510.09561v2)
[38](https://arxiv.org/html/2511.03156v1)
[39](https://ar5iv.labs.arxiv.org/html/2303.18181)
[40](https://topai.tools/alternatives/dreambooth)
[41](https://arxiv.org/abs/2505.12427)
[42](https://www.semanticscholar.org/paper/77b60fdaf00ba3287c07d5584df9be38bf8dcabc)
[43](https://arxiv.org/abs/2507.03026)
[44](https://www.ewadirect.com/proceedings/ace/article/view/26532)
[45](https://arxiv.org/abs/2407.14302)
[46](https://ieeexplore.ieee.org/document/11081342/)
[47](https://ieeexplore.ieee.org/document/11019514/)
[48](https://ieeexplore.ieee.org/document/10655892/)
[49](https://kilthub.cmu.edu/articles/thesis/Mitigating_Negative_Transfer_for_Better_Generalization_and_Efficiency_in_Transfer_Learning/21728726/1)
[50](https://arxiv.org/abs/2411.10268)
[51](https://arxiv.org/abs/2312.08733)
[52](https://aclanthology.org/2021.acl-short.108.pdf)
[53](https://arxiv.org/pdf/2108.02340.pdf)
[54](https://arxiv.org/pdf/2304.04947.pdf)
[55](https://arxiv.org/pdf/2310.01217.pdf)
[56](https://aclanthology.org/2021.eacl-main.39.pdf)
[57](https://dl.acm.org/doi/pdf/10.1145/3616855.3635805)
[58](https://arxiv.org/pdf/2305.15036.pdf)
[59](https://arxiv.org/abs/2311.11077)
[60](https://openaccess.thecvf.com/content/CVPR2024W/PV/papers/Chen_Conv-Adapter_Exploring_Parameter_Efficient_Transfer_Learning_for_ConvNets_CVPRW_2024_paper.pdf)
[61](https://www.alphaxiv.org/overview/2308.06721v1)
[62](https://s-space.snu.ac.kr/handle/10371/210459)
[63](https://ngp9440.tistory.com/143)
[64](https://arxiv.org/html/2504.03738v1)
[65](http://github.com/diffusion-classifier/diffusion-classifier)
[66](https://liner.com/review/efficient-transfer-learning-for-videolanguage-foundation-models)
[67](https://wikidocs.net/280646)
[68](https://www.sciencedirect.com/science/article/abs/pii/S095219762502189X)
[69](https://openaccess.thecvf.com/content/ICCV2025/papers/Ji_Customizing_Domain_Adapters_for_Domain_Generalization_ICCV_2025_paper.pdf)
[70](https://openaccess.thecvf.com/content/CVPR2025/papers/Tu_A4A_Adapter_for_Adapter_Transfer_via_All-for-All_Mapping_for_Cross-Architecture_CVPR_2025_paper.pdf)
[71](https://www.arxiv.org/abs/2511.18537)
[72](https://arxiv.org/abs/2404.12588)
[73](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.pdf)
[74](https://arxiv.org/abs/2508.08604)
[75](https://arxiv.org/html/2506.07986v1)
[76](https://arxiv.org/abs/2303.16203)
[77](https://arxiv.org/html/2410.15858v1)
[78](https://arxiv.org/abs/2407.05897)
[79](https://arxiv.org/abs/2411.19339)
[80](https://ieeexplore.ieee.org/document/10672612/)
[81](https://ieeexplore.ieee.org/document/10692900/)
[82](https://ieeexplore.ieee.org/document/10376944/)
[83](https://ieeexplore.ieee.org/document/10655065/)
[84](https://arxiv.org/abs/2403.18525)
[85](https://www.semanticscholar.org/paper/00a23e8e07a0eba393627a0c0fe00af0000616ab)
[86](https://arxiv.org/abs/2410.08309)
[87](https://ieeexplore.ieee.org/document/10689422/)
[88](https://arxiv.org/html/2412.14580v1)
[89](https://arxiv.org/pdf/2309.12530.pdf)
[90](http://arxiv.org/pdf/2311.15145.pdf)
[91](http://arxiv.org/pdf/2405.06914.pdf)
[92](https://arxiv.org/pdf/2407.20171.pdf)
[93](https://arxiv.org/pdf/2302.09251.pdf)
[94](https://arxiv.org/pdf/2211.01324.pdf)
[95](https://arxiv.org/html/2407.05897)
[96](https://openreview.net/pdf/65b114f3f8bce74467ea97bc9afece55a7b91539.pdf)
[97](https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_Few-shot_Learner_Parameterization_by_Diffusion_Time-steps_CVPR_2024_paper.pdf)
[98](https://en.wikipedia.org/wiki/U-Net)
[99](https://www.arxiv.org/abs/2508.20783)
[100](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/fsdm/)
[101](https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/)
[102](https://arxiv.org/html/2407.05897v2)
[103](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ddpm-pa/)
[104](https://www.digitalocean.com/community/tutorials/unet-architecture-image-segmentation)
[105](https://www.themoonlight.io/en/review/evaluating-compositional-generalisation-in-vlms-and-diffusion-models)
[106](https://arxiv.org/pdf/2508.20783.pdf)
[107](https://arxiv.org/pdf/2510.08659.pdf)
[108](https://arxiv.org/pdf/2506.21298.pdf)
[109](https://arxiv.org/html/2506.04713v3)
[110](https://arxiv.org/html/2505.17955v3)
[111](https://arxiv.org/html/2506.04713v1)
[112](https://arxiv.org/pdf/2504.15991.pdf)
[113](https://arxiv.org/html/2502.09507v1)
[114](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.pdf)
[115](https://openaccess.thecvf.com/content/ICCV2023/papers/Singh_Benchmarking_Low-Shot_Robustness_to_Natural_Distribution_Shifts_ICCV_2023_paper.pdf)
[116](https://openreview.net/pdf?id=rqKTms-YHAW)
[117](https://www.ultralytics.com/blog/a-guide-on-u-net-architecture-and-its-applications)
