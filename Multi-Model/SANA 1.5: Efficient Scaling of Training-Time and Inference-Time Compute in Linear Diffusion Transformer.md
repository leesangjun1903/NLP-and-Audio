# SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer

### 핵심 요약

SANA 1.5는 텍스트-이미지 생성 분야에서 근본적인 스케일링 패러다임 전환을 제시합니다. SANA-1.0을 기반으로 하는 이 연구는 단순히 모델 크기를 증가시키는 전통적 접근에서 벗어나, **더 나은 최적화 궤적**을 통해 효율적인 스케일링을 달성합니다. 3가지 핵심 혁신—훈련 시간 스케일링, 깊이 기반 가지치기, 추론 시간 스케일링—을 통합함으로써, 연구팀은 4.8B 파라미터 모델로 24B 파라미터 모델과 비교 가능한 또는 더 우수한 성능을 달성하면서 훈련 비용을 60% 감축했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

### 해결하는 문제

텍스트-이미지 생성 모델의 급속한 규모 확대(PixArts 0.6B → Playground v3 24B)는 심각한 계산 병목을 야기합니다. 기존 접근법들은 다음의 한계를 가집니다:

1. **훈련 효율성**: 대규모 모델을 처음부터 학습하려면 엄청난 계산 자원 필요
2. **메모리 제약**: 최적화기(예: AdamW)의 메모리 오버헤드로 인해 고급 GPU 필요
3. **배포 유연성**: 고정된 모델 크기만 지원, 다양한 하드웨어 환경에 부적합
4. **품질-계산 트레이드오프**: 모델 크기 증가가 유일한 성능 향상 방법

이러한 문제들은 고품질 이미지 생성을 연구 기관과 상업적 주체에게만 접근 가능하게 만듭니다.

### 제안하는 방법 및 수식

#### 1. Efficient Model Growth (효율적 모델 성장)

**Partial Preservation Initialization (부분 보존 초기화):**

SANA-1.5는 N개 블록의 사전학습 모델을 N+M 블록으로 확장합니다 (N=20, M=40). 핵심 초기화 전략:

$$
R_\ell = \begin{cases} 
R_{\text{pre}}(\ell) & \text{if } \ell < N-2 \\
\mathcal{N}(0, \sigma^2) & \text{if } \ell \geq N
\end{cases}
$$

여기서 새로운 블록의 출력 프로젝션은 항등함수로 작동하도록:

$$
W_{\text{out}}^{\text{(new)}} \leftarrow 0
$$

이는 새로운 블록이 초기에 항등 변환을 수행하도록 하여 **안정적인 최적화 경로**를 보장합니다. 마지막 2개의 사전학습된 블록을 제거하는 이유는 블록 중요도 분석(Figure 5)에서 태스크 관련도가 높은 블록들이 잘 학습된 특성을 방해하기 때문입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

**메모리 효율적 CAME-8bit 최적화기:**

$$
\text{saved memory} = \sum_{\ell \in \Omega} 24 \cdot n_\ell \text{ bytes}
$$

여기서 $\Omega$는 정량화된 계층, $n_\ell$은 파라미터 수입니다. 블록 단위 정량화 함수:

$$
\tilde{v} = \text{round}\left(\frac{v - v_{\min}}{v_{\max} - v_{\min}} \times 255\right)
$$

**결과**: AdamW-32bit 대비 메모리 사용 8배 감소 (43GB → 36GB 효과적 감소), 25% 추가 절감 달성. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

#### 2. Model Depth Pruning (깊이 기반 모델 압축)

**블록 중요도 분석 (Block Importance Analysis):**

$$
BI_\ell = 1 - E_{X_{t,\ell}, X_{t,\ell+1}} \left[ \text{similarity}(X_{t,\ell}, X_{t,\ell+1}) \right]
$$

이 메트릭은 각 블록의 입출력 특징 유사성을 측정하여 정보 변환량을 정량화합니다. 확산 시간단계와 보정 데이터셋(100개 다양한 프롬프트)에 걸쳐 평균화됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

**주요 발견:**
- **헤드 블록** (1-5): 높은 중요도 (0.7-0.8) - 잠상(latent)을 확산 분포로 변환
- **테일 블록** (55-60): 높은 중요도 (0.6-0.7) - 확산 분포를 원래 이미지로 역변환
- **중간 블록** (20-50): 낮은 중요도 (0.3-0.5) - 점진적 세부사항 정제

**가지치기 후 미세조정:**

100 훈련 스텝(단일 GPU)으로 60블록 → 40/30/20블록 압축 가능, GenEval 성능 유지: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)
- 60블록 → 40블록: 0.693 → 0.684
- 60블록 → 30블록: 0.693 → 0.675  
- 60블록 → 20블록: 0.693 → 0.672 (SANA-1.0 1.6B의 0.665 초과)

#### 3. Inference-Time Scaling (추론 시간 스케일링)

**Denoising Steps vs. Repeated Sampling:**

저자들은 **반복 샘플링이 Denoising 스텝 증가보다 우월**함을 입증합니다. 이유: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

1. Denoising 초기 오류는 자동 수정 불가능 (Figure 3a 참조)
2. 품질 개선이 빠르게 포화 (20 스텝에서 최적화)

따라서 여러 샘플 생성 후 VLM 기반 검증자로 최적 이미지 선택:

$$
\text{score} = P_{\text{VILA}}(\text{"match"} | \text{image, prompt})
$$

**VILA-Judge 설계:**

2M 프롬프트-이미지 데이터셋으로 미세조정된 NVILA-2B 모델. 토너먼트 형식 비교:

- 두 이미지 쌍을 반복 비교
- VILA 응답이 "yes/no"로 갈라지면 "yes" 이미지 선택
- 동일 응답이면 로그프롭(log probability) 기반 선택

**성능 개선:**

$$
\text{GenEval Score} = 0.81 \text{ (단일)} \to 0.96 \text{ (2048샘플)}
$$

세부 성능 향상: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)
- 위치(Position): 0.59 → 0.96
- 색상 속성(Color Attribution): 0.65 → 0.87
- 계산(Counting): 0.86 → 0.97

### 모델 구조

**Linear Diffusion Transformer 아키텍처:**

선형 자기-어텐션과 바닐라 크로스-어텐션 결합. 훈련 안정성을 위해 **RMSNorm** 적용:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma
$$

쿼리와 키에 RMSNorm 적용하여 선형 어텐션의 로짓 폭발 방지 (FP16에서 $\geq 65504$ 오버플로우 회피). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

**하이브리드 정밀도 CAME-8bit:**

- **1차 모멘트**: 블록 단위 8비트 정량화
- **2차 모멘트**: 32비트 유지 (행렬 인수분해로 이미 메모리 효율적)

이는 최적화 안정성 보존과 메모리 절감 간 균형을 달성합니다.

### 성능 향상 및 일반화

#### 정량적 성능 개선

| 메트릭 | SANA-1.0 1.6B | SANA-1.5 4.8B (Pre) | SANA-1.5 4.8B (Ours) | 추론 스케일 |
|--------|---------------|-------------------|----------------------|-----------|
| GenEval | 0.66 | 0.72 | 0.81 | **0.96** |
| FID ↓ | 5.76 | 5.42 | 5.99 | - |
| CLIP | 28.67 | 29.16 | 29.23 | - |
| 처리량 (img/s) | 1.0 | 0.26 | 0.26 | - |
| 레이턴시 (s) | 1.2 | 4.2 | 4.2 | - |

**Playground v3 (24B, GenEval 0.76) 대비:** SANA-1.5는 5.5배 낮은 레이턴시, 6.5배 높은 처리량, 0.05 높은 GenEval 스코어. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

#### 일반화 성능 향상의 메커니즘

**1. 더 나은 최적화 궤적 발견:**

모델 성장 전략이 더 큰 최적화 공간을 탐색하여 더 좋은 특징 표현을 발견합니다. 작은 모델의 사전학습된 특징이 새로운 블록의 학습을 위한 견고한 기반(foundation)이 됩니다.

**2. 지식 전이의 효과:**

무작위 초기화 대비 부분 보존 초기화 사용 시:
- **수렴 시간 60% 단축** (같은 성능 도달)
- **훈련 안정성 향상** (Cyclic/Block Replication 시 NaN 손실 발생 방지)

**3. 블록 중요도 기반 지능형 설계:**

블록 중요도 분석이 두 방향으로 활용:
- **성장**: 새 블록을 마지막에 추가할 때 테일 블록(높은 중요도) 2개 제거
- **가지치기**: 중간 블록(낮은 중요도)를 안전하게 제거 후 미세조정

**4. 다양한 평가 메트릭에서 일관된 개선:**

고품질 데이터(GenEval 스타일 144K 프롬프트) 미세조정 후:

$$
\text{GenEval}_{v1} = 0.72 \to \text{GenEval}_{v2} = 0.81 
$$

(3% 개선)

**5. 다중언어 지원으로 일반화 확대:**

100K 영어 프롬프트를 GPT-4로 4가지 형식으로 확장:
- 순수 중국어
- 영어-중국어 혼합
- 이모지 포함

10K 미세조정 후 **다국어 및 이모지 프롬프트에서 안정적 출력** 달성. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

### 최신 연구와의 비교 분석 (2020년 이후)

#### 1. 스케일링 법칙 연구

**Scaling Laws for Diffusion Transformers (Liang et al., 2024):** [arxiv](https://arxiv.org/abs/2502.12048)
- 훈련 손실이 계산 예산과 멱법칙 관계 준수 발견
- 최적 모델/데이터 크기 예측 프레임워크 제시
- **SANA 1.5의 위치**: 실무적 검증과 추가 최적화 (QK normalization, CAME-8bit)

**Towards Precise Scaling Laws for Video Diffusion Transformers (Yin et al., 2024):** [semanticscholar](https://www.semanticscholar.org/paper/1a65219f0d3852b55d1fadf58e1ca75c1090805e)
- 비디오 생성에서 하이퍼파라미터 민감도 강조
- 학습률, 배치 크기 정확한 모델링 필요
- **SANA 1.5의 차별성**: 이미지 생성의 블록 구조적 스케일링에 집중

#### 2. 모델 압축 및 효율화

**Minitron: LLM Pruning and Distillation in Practice (Sreenivas et al., 2024):** [jst.tnu.edu](https://jst.tnu.edu.vn/jst/article/view/13790)
- LLM의 구조화된 가지치기 (블록 제거)
- 추론 후(post-training) 압축 가능성 입증
- **SANA 1.5의 혁신**: Diffusion Transformer에 처음 적용, 입출력 유사성 기반 중요도 계산

**DiT-MoE: Scaling Diffusion Transformers to 16 Billion Parameters (Fei et al., 2024):** [arxiv](https://arxiv.org/abs/2511.05535)
- Mixture-of-Experts를 통한 희소 확장 (16.5B)
- 전문가 선택이 공간 위치와 시간 스텝에 의존
- **비교**: SANA는 **밀집 모델로 더 효율적 접근**, 추론 시간 스케일링으로 보상

#### 3. 추론 시간 스케일링

**Large Language Monkeys: Scaling Inference with Repeated Sampling (Brown et al., 2024):** [academic.oup](https://academic.oup.com/bib/article/26/Supplement_1/i44/8378055)
- LLM에서 샘플 반복을 통해 추론 계산으로 능력 확장 가능 증명
- 로그-선형 스케일링 패턴 발견
- **SANA 1.5의 기여**: 이미지 생성에 **처음 적용**, VLM 판정자로 검증 강화

**Inference-Time Scaling for Diffusion Models Beyond Scaling Denoising Steps (Ma et al., 2025):** [academic.oup](https://academic.oup.com/bib/article/26/Supplement_1/i24/8378044)
- Feynman-Kac 프레임워크로 입자 기반 샘플링 제시
- 보상 함수 기반 유연한 조종(steering) 가능
- **SANA 1.5와의 관계**: 동시 발전 연구로, SANA가 실용적 VILA-Judge 활용

#### 4. 텍스트-이미지 정렬 평가 및 벤치마크

**GenEval: An Object-Focused Framework (Ghosh et al., 2023):** [e-journal.unair.ac](https://e-journal.unair.ac.id/JESTT/article/view/47782)
- 객체 중심 평가 (공동 발생, 위치, 개수, 색상)
- 83% 인간-어그리먼트 달성
- **SANA 1.5의 중요성**: GenEval에서 **0.96 새로운 최고 성능**

**GenEval 2: Addressing Benchmark Drift (Kamath et al., 2025):** [arxiv](https://arxiv.org/pdf/2401.10061.pdf)
- GenEval의 **포화 문제** 지적 (벤치마크 드리프트 17.7% 오차)
- 구성성 강조 (3-10 "atoms" per prompt)
- Soft-TIFA 메트릭 제안 (AUROC 94.5% 인간 정렬)
- **시사점**: SANA 1.5의 0.81 → 0.96 개선은 **진정한 능력 향상** 증명

**On the Scalability of Diffusion-based Text-to-Image Generation (Li et al., 2024):** [arxiv](http://arxiv.org/pdf/2312.04557.pdf)
- 0.4B~4B 범위에서 크로스 어텐션과 블록 수의 영향 분석
- 텍스트 정렬에는 **블록 깊이가 채널 수보다 효과적**
- **SANA 1.5의 검증**: 블록 중요도 기반 가지치기로 이론 실증

#### 5. 다른 효율적 생성 모델들과의 비교

| 모델 | 파라미터 | 처리량 (img/s) | 레이턴시 (s) | GenEval |
|-----|---------|--------------|-----------|---------|
| FLUX-dev | 12.0B | 0.04 | 23.0 | 0.67 |
| Playground v3 | 24.0B | 0.06 | 15.0 | 0.76 |
| SANA-1.5 4.8B | 4.8B | **0.26** | **4.2** | **0.81** |
| SANA-1.5 + Inference | 4.8B | 0.001 | - | **0.96** |

SANA-1.5는 **파라미터 효율성 (24B 대비 1/5), 속도 (5.5배), 품질 (GenEval 0.81)**에서 우수합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fc575fd-e6f1-4416-9aff-8f7495cea32b/2501.18427v4.pdf)

### 일반화 성능의 한계 및 미래 연구 방향

#### 현재 한계

1. **텍스트 렌더링**: 복잡한 장면에서 정확한 텍스트 생성 여전히 미흡
2. **추론 시간 스케일링 계산 비용**: 2048샘플 생성 + VILA-Judge = 149 GFLOPs (모델 1회 인퍼런스의 ~30배)
3. **공간 관계 이해**: 다중 객체 간의 3D 공간 관계에서 약점
4. **색상 정확도**: 특정 RGB 값의 미세한 색상 제어 제한

#### 앞으로의 연구 고려사항

**1. 추론 스케일링 효율화:**
- 더 가벼운 검증 모델 개발 (VILA-Judge보다 작은 모델)
- 캐싱 및 조기 종료 메커니즘 도입
- 적응형 샘플 수 결정 (쉬운 프롬프트는 적게, 어려운 프롬프트는 많게)

**2. 일반화 성능 확대:**
- **도메인 확장**: 의료 영상, 3D 생성으로의 이전
- **제로샷 능력**: 새로운 스타일/개념에 대한 일반화
- **적대적 입력**: 모자이크된 또는 노이즈 있는 프롬프트 강건성

**3. 멀티모달 일반화:**
- 비디오 생성으로 자연스러운 확장 (이미 SANA-Video 연구 진행 중)
- 이미지-텍스트 쌍방향 생성
- 조건부 생성의 정밀 제어

**4. 블록 선택의 동적화:**
- 입력 프롬프트의 복잡도에 따라 **동적으로 블록 활성화** 조정
- 조기 종료(early exit) 메커니즘으로 불필요한 계산 회피
- 각 프롬프트에 최적화된 블록 조합 학습

**5. 최적화 궤적 이론:**
- 왜 부분 보존 초기화가 더 나은 솔루션을 발견하는가에 대한 이론적 분석
- 로스 랜드스케이프(loss landscape) 시각화
- 신경 탱젠트 커널(NTK) 이론과의 연결

**6. 안전성 강화:**
- NSFW 내용 필터링 (현재 ShieldGemma-2B 적용)
- 편향 평가 및 공정성 분석
- 저작권 보호된 콘텐츠 생성 방지

#### 산업적 임플리케이션

SANA 1.5의 효율성 개선은 다음을 가능하게 합니다:

- **엣지 디바이스 배포**: RTX 4090에서 4.8B 모델 미세조정 가능
- **기업 맞춤화**: 리소스 제약 환경에서의 도메인 특화 모델 개발
- **민주화**: 오픈소스 커뮤니티의 고품질 모델 접근성 대폭 향상

### 결론

SANA 1.5는 텍스트-이미지 생성의 스케일링에 대한 근본적 재고를 제시합니다. **"더 큰 모델이 항상 더 낫다"는 기존 믿음에 도전**하며, 지능적 설계(깊이 성장, 블록 가지치기, 추론 시간 스케일링)를 통해 더 작은 모델이 더 큰 모델을 능가할 수 있음을 입증합니다. 

세 가지 기술의 조화로운 통합으로, SANA 1.5는:
- **60% 훈련 시간 단축**
- **1.6배 성능 향상** (GenEval 0.66 → 0.81)
- **5.5배 속도 향상** (FLUX 대비)
- **2배 처리량 증가** 

을 동시에 달성했습니다. 이러한 효율성 개선이 고품질 이미지 생성을 더 많은 연구자와 실무자에게 접근 가능하게 함으로써, 생성 AI의 민주화에 기여합니다.

***

### 참고 문헌

<span style="display:none">[^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97][^1_98]</span>

<div align="center">⁂</div>

[^1_1]: 2501.18427v4.pdf

[^1_2]: https://arxiv.org/abs/2502.12048

[^1_3]: https://www.semanticscholar.org/paper/1a65219f0d3852b55d1fadf58e1ca75c1090805e

[^1_4]: https://jst.tnu.edu.vn/jst/article/view/13790

[^1_5]: https://arxiv.org/abs/2511.05535

[^1_6]: https://academic.oup.com/bib/article/26/Supplement_1/i44/8378055

[^1_7]: https://academic.oup.com/bib/article/26/Supplement_1/i24/8378044

[^1_8]: https://e-journal.unair.ac.id/JESTT/article/view/47782

[^1_9]: https://arxiv.org/pdf/2401.10061.pdf

[^1_10]: http://arxiv.org/pdf/2312.04557.pdf

[^1_11]: https://www.sciltp.com/journals/hm/articles/2504000541

[^1_12]: https://www.mdpi.com/2076-3417/15/20/11150

[^1_13]: https://jisem-journal.com/index.php/journal/article/view/6615

[^1_14]: https://www.frontiersin.org/articles/10.3389/fenvs.2025.1659344/full

[^1_15]: https://ieeexplore.ieee.org/document/11147513/

[^1_16]: https://pubs.aip.org/pof/article/37/11/117119/3371491/Fine-structure-investigation-of-turbulence-induced

[^1_17]: https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs

[^1_18]: https://iopscience.iop.org/article/10.1149/MA2025-02121107mtgabs

[^1_19]: https://pubs.aip.org/pof/article/37/11/117120/3371493/Fine-structure-investigation-of-turbulence-induced

[^1_20]: https://biss.pensoft.net/article/181733/

[^1_21]: https://arxiv.org/html/2410.02098

[^1_22]: https://arxiv.org/html/2404.09976

[^1_23]: https://arxiv.org/html/2501.18427v3

[^1_24]: https://arxiv.org/html/2407.11633v1

[^1_25]: http://arxiv.org/pdf/2404.02883.pdf

[^1_26]: http://arxiv.org/pdf/2212.09748v2.pdf

[^1_27]: https://arxiv.org/abs/2410.13925v1

[^1_28]: https://arxiv.org/abs/2301.09474

[^1_29]: https://openreview.net/pdf?id=iIGNrDwDuP

[^1_30]: https://liner.com/review/scaling-laws-synthetic-images-for-model-training-for-now

[^1_31]: https://liner.com/review/inferencetime-scaling-for-diffusion-models-beyond-scaling-denoising-steps

[^1_32]: https://neurips.cc/virtual/2025/poster/117664

[^1_33]: https://proceedings.iclr.cc/paper_files/paper/2025/file/f8e7248f3e659cfe70c6debcdae1b023-Paper-Conference.pdf

[^1_34]: https://arxiv.org/abs/2501.09732

[^1_35]: https://kimjy99.github.io/논문리뷰/stable-diffusion-3/

[^1_36]: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Scaling_Down_Text_Encoders_of_Text-to-Image_Diffusion_Models_CVPR_2025_paper.pdf

[^1_37]: https://velog.io/@jojo0217/논문리뷰-Inference-Time-Scaling-for-Diffusion-Modelsbeyond-Scaling-Denoising-Steps

[^1_38]: https://velog.io/@guts4/Scaling-Rectified-Flow-Transformers-for-High-Resolution-Image-SynthesisStableDiffusion3-2024-arXiv

[^1_39]: https://arxiv.org/html/2410.13863v1

[^1_40]: https://arxiv.org/abs/2507.08390

[^1_41]: https://openaccess.thecvf.com/content/ICCV2025/papers/Hou_Dita_Scaling_Diffusion_Transformer_for_Generalist_Vision-Language-Action_Policy_ICCV_2025_paper.pdf

[^1_42]: https://www.amazon.science/publications/on-the-scalability-of-diffusion-based-text-to-image-generation

[^1_43]: https://huggingface.co/papers/2501.09732

[^1_44]: https://arxiv.org/abs/2411.17470

[^1_45]: https://arxiv.org/abs/2503.09443

[^1_46]: https://arxiv.org/abs/2501.06848

[^1_47]: https://arxiv.org/html/2505.15270v2

[^1_48]: https://arxiv.org/pdf/2001.08361.pdf

[^1_49]: https://arxiv.org/abs/2505.22524

[^1_50]: https://arxiv.org/html/2512.01426v1

[^1_51]: https://arxiv.org/abs/2503.00307

[^1_52]: https://arxiv.org/html/2410.15959v3

[^1_53]: https://arxiv.org/html/2312.04567v1

[^1_54]: https://arxiv.org/html/2510.24711v1

[^1_55]: https://arxiv.org/html/2410.08184v1

[^1_56]: https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.pdf

[^1_57]: https://arxiv.org/abs/2503.07265

[^1_58]: https://www.semanticscholar.org/paper/e8f84138900a916be4476abbeb474fd89ce49e45

[^1_59]: https://arxiv.org/abs/2510.02987

[^1_60]: https://arxiv.org/abs/2506.08835

[^1_61]: https://arxiv.org/abs/2508.06152

[^1_62]: https://arxiv.org/abs/2505.21347

[^1_63]: https://arxiv.org/abs/2412.18150

[^1_64]: https://arxiv.org/abs/2310.11513

[^1_65]: https://arxiv.org/abs/2409.10695

[^1_66]: https://arxiv.org/abs/2410.05664

[^1_67]: https://arxiv.org/pdf/2310.11513.pdf

[^1_68]: http://arxiv.org/pdf/2403.04321.pdf

[^1_69]: https://arxiv.org/pdf/2307.06350.pdf

[^1_70]: https://arxiv.org/html/2503.21745v1

[^1_71]: https://arxiv.org/html/2406.03070

[^1_72]: https://arxiv.org/html/2412.18150

[^1_73]: https://arxiv.org/abs/2406.13743

[^1_74]: http://arxiv.org/pdf/2503.07265.pdf

[^1_75]: https://www.emergentmind.com/topics/geneval-2

[^1_76]: https://openaccess.thecvf.com/content/CVPR2024/papers/Lin_VILA_On_Pre-training_for_Visual_Language_Models_CVPR_2024_paper.pdf

[^1_77]: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_DiG_Scalable_and_Efficient_Diffusion_Models_with_Gated_Linear_Attention_CVPR_2025_paper.pdf

[^1_78]: https://www.themoonlight.io/en/review/geneval-2-addressing-benchmark-drift-in-text-to-image-evaluation

[^1_79]: https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/

[^1_80]: https://arxiv.org/abs/2405.18428

[^1_81]: https://proceedings.neurips.cc/paper_files/paper/2023/file/a3bf71c7c63f0c3bcb7ff67c67b1e7b1-Paper-Datasets_and_Benchmarks.pdf

[^1_82]: https://arxiv.org/html/2312.07533v2

[^1_83]: https://kimjy99.github.io/논문리뷰/sana/

[^1_84]: https://liner.com/review/geneval-an-objectfocused-framework-for-evaluating-texttoimage-alignment

[^1_85]: https://github.com/NVlabs/VILA

[^1_86]: https://liner.com/ko/review/dig-scalable-and-efficient-diffusion-models-with-gated-linear-attention

[^1_87]: https://velog.io/@kirby_id/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-VILA-On-Pre-training-for-Visual-Language-Models

[^1_88]: https://www.themoonlight.io/en/review/dig-scalable-and-efficient-diffusion-models-with-gated-linear-attention

[^1_89]: https://arxiv.org/html/2512.22374

[^1_90]: https://www.semanticscholar.org/paper/GenEval-2:-Addressing-Benchmark-Drift-in-Evaluation-Kamath-Chang/e8f84138900a916be4476abbeb474fd89ce49e45

[^1_91]: https://arxiv.org/html/2507.23682v3

[^1_92]: https://arxiv.org/html/2310.11513

[^1_93]: https://arxiv.org/html/2503.16726v1

[^1_94]: https://arxiv.org/abs/2512.16853

[^1_95]: https://arxiv.org/html/2411.12915v2

[^1_96]: https://arxiv.org/abs/2509.24695

[^1_97]: https://arxiv.org/html/2409.04429v3

[^1_98]: https://arxiv.org/html/2509.24899v2
