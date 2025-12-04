# Improved Vector Quantized Diffusion Models

### 1. 핵심 주장 및 주요 기여 요약

"Improved Vector Quantized Diffusion Models"는 **VQ-Diffusion의 두 가지 근본적 문제**를 식별하고 해결하는 논문입니다.[1]

**핵심 주장:**
- VQ-Diffusion은 **후행 문제(Posterior Issue)**로 인해 텍스트 입력과의 상관성이 약함
- **결합 분포 문제(Joint Distribution Issue)**로 인해 구조적으로 부자연스러운 이미지 생성

**주요 기여:**
1. **이산 분류기 없는 가이던스**: 사전과 사후 분포를 동시에 고려하는 개선된 구현
2. **고품질 추론 전략**: 순수도 기반 샘플링으로 위치 의존성 모델링
3. **획기적 성능 향상**: MSCOCO에서 FID 13.86 → 8.44 (37.8% 개선), ImageNet에서 11.89 → 4.83 (59.3% 개선)[1]

***

### 2. 해결하는 문제 및 제안 방법

#### 2.1 후행 문제 (Posterior Issue)

**문제의 본질:**
조건부 이미지 생성(텍스트-이미지)에서 네트워크가 손상된 입력($x_t$)이 충분한 정보를 가지고 있어서 조건 정보(y)를 무시하는 현상[1]

**베이즈 정리 기반 최적화 공식:**

기존 목표: $\max_x \log p(x|y)$

제안 목표: 
$$\max_x [\log p(x|y) + s \log p(y|x)]$$

베이즈 정리를 적용하면:
$$\max_x \left[\log p(x_{t-1}|x_t, y) + (s-1)\left[\log p(x_{t-1}|x_t, y) - \log p(x_{t-1}|x_t)\right]\right]$$ ... (Eq. 7)

여기서 $s$는 사후 제약의 강도를 조절하는 하이퍼파라미터[1]

**핵심 개선:**
- GLIDE의 연속 공간 방식을 이산 공간에 맞게 재구성
- **학습 가능한 벡터** 사용으로 널 벡터보다 더 효과적 (CLIP 점수: 0.302 vs 0.298)[1]

#### 2.2 결합 분포 문제 (Joint Distribution Issue)

**문제의 본질:**
각 디노이징 스텝에서 여러 토큰이 독립적으로 샘플링되어 위치 간 종속성을 무시[1]

**순수도 기반 해결책:**

**순수도 정의** (위치 $i$, 시점 $t$):
$$\text{purity}(i, t) = \max_{j \in \{1,...,K\}} p(x_i^0 = j | x_i^t)$$ ... (Eq. 8)

높은 순수도 = 높은 신뢰도 = 정확한 토큰 선택 가능성

**확률 조정 (선택적):**
$$\hat{p}(x_i^0|x_i^t) = \text{softmax}\left(\frac{1}{\text{purity}(i,t)^r} \log p(x_i^0|x_i^t)\right)$$ ... (Eq. 11)

**전략:**
1. 각 스텝에서 $z$개의 토큰만 샘플 (다중 토큰의 독립성 문제 완화)
2. 순수도 기반 중요도 샘플링으로 높은 신뢰도 토큰 우선 선택[1]

***

### 3. 모델 구조

**VQ-VAE 기반 이산 생성 파이프라인:**

$$\text{이미지} \xrightarrow{\text{VQ-VAE}}^{\text{인코더}} \text{이산 토큰} \xrightarrow{\text{Diffusion}}^{\text{개선됨}} \text{생성 토큰} \xrightarrow{\text{VQ-VAE}}^{\text{디코더}} \text{이미지}$$

**핵심 개선 통합:**
- **조건 정보 처리:** 조건부 로짓 $p(x_0|x_t, y)$ + 비조건부 로짓 $p(x_0|x_t, \mathbf{v}_{\text{learnable}})$ 
- **가이던스 적용:** Equation 7의 혼합 계산
- **순수도 계산:** 각 위치의 최대 확률값 추정
- **적응형 샘플링:** 순수도 기반 중요도 샘플링 적용[1]

**모델 규모:**
- **Base**: 370M 파라미터, 다양한 데이터셋 파인튜닝
- **Large**: 1.27B 파라미터, ITHQ-200M에서 학습 (200M 이미지 쌍)[1]

***

### 4. 성능 향상 결과

| 데이터셋 | FID (기존) | FID (개선) | 개선율 |
|---------|----------|----------|-------|
| MSCOCO | 13.86 | 8.62 | **-37.8%** |
| ImageNet | 11.89 | 4.83 | **-59.3%** |
| Conceptual Captions | 33.65 | 15.58 | **-53.7%** |
| ITHQ-200M | 25.87 | 11.89 | **-54.0%** |

**추가 메트릭 개선 (MSCOCO):**
- 이미지 품질(QS): 0.841 → 0.866 (**+2.9%**)
- 텍스트 정렬(CLIP): 0.267 → 0.304 (**+13.9%**)[1]

**절제 연구 주요 발견:**
1. **제로샷 가이던스**: 재학습 없이도 12.12 FID 달성 (기존 13.86)
2. **최적 가이던스 스케일**: MSCOCO $s=3$, CC $s=5$에서 최적
3. **순수도 샘플링**: 추가 학습 없이도 1-3% 개선[1]

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 데이터셋 규모에 따른 일반화

$$\text{개선율} \propto \log(\text{데이터셋 크기})$$

- **소규모** (CUB-200): 1% 미만 (기존도 이미 양호)
- **중규모** (MSCOCO): 37.8% 개선
- **대규모** (CC, ITHQ-200M): 50-60% 개선

**해석:** 대규모 데이터에서 텍스트 조건의 미활용 문제가 더 큰 것으로 추정[1]

#### 5.2 다양한 작업 간 일반화

✓ **텍스트-이미지 합성**: MSCOCO (8.62 FID), CC (15.58 FID), ITHQ-200M (11.89 FID)
✓ **클래스 조건부 생성**: ImageNet (4.83 FID) - BigGAN 능가
✓ **다양한 평가 지표**: QS, CLIP 점수, 다양성(DS)에서 일관된 개선[1]

#### 5.3 하이퍼파라미터 견고성

| 파라미터 | 범위 | 최적값 | 민감도 |
|---------|------|--------|--------|
| 가이던스 스케일 $s$ | [1][2] | 3-5 | 중간 |
| 순수도 스케일 $r$ | [0.5, 2] | 1.0 | 낮음 |
| 토큰 샘플 개수 $z$ | 1-8 | 데이터셋 의존 | 중간 |

**결론:** 적절한 범위 내에서 견고한 성능[1]

#### 5.4 일반화의 메커니즘

**메커니즘 1 - 사후 제약 (Posterior Constraint):**
$$\text{조건 무시 문제} \xrightarrow{\text{해결}} p(y|x) \text{ 최대화} \xrightarrow{\text{적용}} \text{모든 조건부 작업}$$

**메커니즘 2 - 위치 의존성 모델링:**
$$\text{독립 샘플링} \xrightarrow{\text{해결}} \text{순차 + 순수도 기반} \xrightarrow{\text{적용}} \text{구조화된 출력}$$

***

### 6. 모델의 한계

#### 6.1 이론적 한계

1. **부분적 해결만 가능**: 순수도 기반 휴리스틱은 결합 분포를 완벽하게 해결하지 못함[1]
2. **독립성 가정**: VQ-Diffusion 디코더 레벨에서의 독립성 가정 여전히 존재
3. **선형 성능 증가의 한계**: 대규모 모델에서 수렴 속도 감소

#### 6.2 실무적 한계

| 측면 | 현황 |
|------|------|
| **추론 시간** | 1.5-2배 증가 (품질-속도 트레이드오프) |
| **메모리** | 고해상도에서 무시할 수 없는 오버헤드 |
| **하이퍼파라미터** | 데이터셋별 수동 조정 필요 |
| **극소규모 데이터** | CUB-200에서 개선율 < 1% |

#### 6.3 일반화 한계

**잘 작동:** 조건부 생성, 대규모 데이터, 구조화된 출력
**미검증:** 도메인 이동(의료, 위성), 극대규모 데이터, 비영어 텍스트[1]

***

### 7. 앞으로의 연구에 미치는 영향

#### 7.1 이론적 기여

1. **이산 공간 가이던스의 정규화**: Continuous 공간 방식을 이산 공간에 맞게 재구성한 첫 정확한 공식[1]

2. **위치 의존성의 중요성 입증**: 
   ```
   기존 가정: 각 토큰 독립적 샘플 가능
   이 논문: 위치 간 상관성 존재 및 중요성 입증
   ```

3. **순수도 개념의 도입**: 신뢰도 기반 샘플링의 새로운 관점 제시

#### 7.2 방법론적 확산

**직접 적용 가능 분야:**
- 텍스트 생성 (D3PM, MaskGIT 개선)
- 음성/음악 생성 (VQ-VAE 기반 모델)
- 멀티모달 생성 (이미지 + 텍스트 + 오디오)

**관련 최신 모델들:**
- **DiMO (2025)**: MaskGIT를 1-스텝으로 증류[3]
- **MD4 (2024)**: 마스크된 확산의 단순화[4]
- **DDPD (2025)**: 계획된 디노이징으로 순서 최적화[5]

#### 7.3 한계 극복을 위한 향후 연구 방향

**1. 결합 분포 문제의 완전한 해결:**
```
현재: 순차 샘플링으로 부분 해결
미래: Generalized Interpolating Discrete Diffusion (GIDD, 2025)
      또는 완전한 구조 의존성 명시 모델링
```

**2. 추론 속도 최적화:**
```
현재: 1.5-2배 느림
미래: 지식 증류 또는 1-스텝 생성
      (DiMO: 1 스텝, FLUX Schnell: 4 스텝)
```

**3. 자동 하이퍼파라미터 선택:**
```
메타 학습 기반 동적 s(t) 예측
강화 학습으로 최적 정책 학습
```

***

### 8. 2020년 이후 관련 최신 연구

#### 기초 이론 발전

| 연도 | 논문 | 기여 | 관계 |
|------|------|------|------|
| 2020 | DDPM (Ho et al.) | 확산 모델 기초 | VQ-Diffusion의 근간 |
| 2021 | D3PM (Austin et al.) | 이산 공간 확산 | 이론적 기초 |
| 2021 | Classifier-Free Guidance | 가이던스 방법 | **이 논문의 기초** |
| 2021 | LDM (Rombach et al.) | 잠재 공간 확산 | 유사 아이디어 |

#### 텍스트-이미지 생성 SOTA

| 모델 | 연도 | 기술 | 특징 |
|------|------|------|------|
| DALL-E | 2021 | 자기회귀 | 초기 대규모 모델 |
| DALL-E 2 | 2022 | 확산 + CLIP | 고품질 이미지 |
| Stable Diffusion | 2022 | 잠재 확산 | 개방형 모델 |
| DALL-E 3 | 2023 | 개선된 텍스트 이해 | 현재 SOTA 중 하나 |
| FLUX.1 | 2024 | Diffusion Transformer | **현재 최고 성능** |

#### 이산 생성 모델의 최신 발전 (2024-2025)

| 논문 | 기여 | 관계 |
|------|------|------|
| MaskGIT (2022) | 마스크 기반 반복 재정의 | 이산 생성의 새 패러다임 |
| Discrete Interpolants (2024) | 통합 프레임워크 | 이 논문의 이론적 확장 |
| DiMO (2025) | MaskGIT 1-스텝 증류 | 속도 최적화 |
| GIDD (2025) | 일반화된 이산 확산 | 결합 분포 완전 해결 |
| DDPD (2025) | 계획된 디노이징 | 순서 최적화 |

#### 최신 연구 트렌드

```
2024-2025 핵심 방향:
1. 하나의 포괄적 프레임워크로 수렴
2. 초고속 생성 (1-4 스텝)
3. 멀티모달 일반화 (이미지, 비디오, 텍스트, 오디오)
4. 해석 가능성 및 명시적 제어성
```

**예:** 
- Critical Windows in Diffusion (2024): 특정 스텝에서 특정 특성 결정[6]
- Plug-and-Play Controllable Generation (2024): 제약 조건 적용[7]

***

### 9. 연구 시 고려할 점

#### 9.1 이론적 고려사항

1. **확률 모델의 정확성**: 사후 제약과 순수도 정의가 근사치 기반 → 베이즈 최적성 증명 필요

2. **결합 분포의 완전한 해결**: 순차 샘플링은 극도의 경우 여전히 문제 → 자기회귀 모델 통합, GNN 의존성 모델링, 흐름 일치로의 전환 검토

3. **수렴 성질 분석**: DDPM와 제안 방법의 수렴 속도 비교 분석 필요

#### 9.2 실험 설계 시 고려사항

```
✓ 공정한 비교: 모델 크기, 학습 데이터, 계산 예산 동일 유지
✓ 다양한 메트릭: FID + CLIP + 인간 평가
✓ 광범위한 검증: 소규모, 중규모, 대규모 데이터 모두
✓ 다양한 도메인: 이미지 + 의료 + 위성 + 예술
✓ 다국어 실험: 영어 + 한국어 + 중국어 등
```

#### 9.3 방법론적 개선

| 개선 방향 | 현황 | 제안 |
|---------|------|------|
| 가이던스 스케일 | 고정된 $s$ | 시간 의존적 $s(t)$ 또는 메타 학습 |
| 하이퍼파라미터 | 그리드 탐색 | 베이즈 최적화, 강화 학습 |
| 조건 처리 | 단일 조건 | 다중 조건 처리 (텍스트 + 스타일) |
| 속도 | 1.5-2배 느림 | 지식 증류로 1-4 스텝 |

#### 9.4 윤리 및 책임성

```
필요한 검토:
1. 데이터셋 편향 분석
2. 프라이버시 위협 평가 (차등 프라이버시)
3. 공정성 메트릭 추가
4. 투명성 보고서 작성
```

***

### 결론

**"Improved Vector Quantized Diffusion Models"는:**

✓ 이산 공간 생성의 **후행 문제**와 **결합 분포 문제**를 정확하게 식별
✓ 이산 공간에 맞게 재구성된 **분류기 없는 가이던스** 제시
✓ **순수도 기반 샘플링**으로 위치 의존성 부분 해결
✓ **광범위한 실험**으로 일관된 성능 향상 입증 (FID 최대 59% 개선)
✓ 기존 모델에 **즉시 적용 가능**한 실용적 방법론

그러나 여전히 **순수도의 휴리스틱 성질**, **결합 분포의 부분적 해결만**, **추론 속도 증가** 등의 한계가 있으며, 향후 연구는 이를 극복하면서 초고속 생성과 멀티모달 생성을 지향해야 합니다.

**2024-2025 최신 연구** (GIDD, DiMO, DDPD 등)는 이 논문의 기초 위에서 더욱 이론적이고 실용적인 개선을 이루고 있습니다.[8][4][3][5][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e50913ef-4338-438a-8cdd-2a688d8cacab/2205.16007v2.pdf)
[2](https://ieeexplore.ieee.org/document/11064518/)
[3](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_DiMO_Distilling_Masked_Diffusion_Models_into_One-step_Generator_ICCV_2025_paper.pdf)
[4](https://arxiv.org/pdf/2406.04329.pdf)
[5](http://arxiv.org/abs/2410.06264)
[6](https://arxiv.org/abs/2403.01633)
[7](https://arxiv.org/pdf/2410.02143.pdf)
[8](https://papers.baulab.info/papers/also/Ho-2022.pdf)
[9](https://ieeexplore.ieee.org/document/11162241/)
[10](https://ieeexplore.ieee.org/document/10972642/)
[11](https://ieeexplore.ieee.org/document/11147881/)
[12](https://ieeexplore.ieee.org/document/11147618/)
[13](https://arxiv.org/abs/2402.09052)
[14](https://arxiv.org/abs/2402.05210)
[15](https://ieeexplore.ieee.org/document/10656241/)
[16](https://arxiv.org/abs/2402.10210)
[17](https://ieeexplore.ieee.org/document/10678118/)
[18](https://arxiv.org/abs/2302.08113)
[19](https://arxiv.org/html/2406.11713v1)
[20](https://arxiv.org/abs/2408.08306)
[21](https://arxiv.org/html/2412.14422)
[22](http://arxiv.org/pdf/2409.19589.pdf)
[23](https://arxiv.org/pdf/2209.00796v8.pdf)
[24](https://arxiv.org/pdf/2503.05149.pdf)
[25](https://dl.acm.org/doi/pdf/10.1145/3618342)
[26](https://hiringnet.com/image-generation-state-of-the-art-open-source-ai-models-in-2025)
[27](https://www.emergentmind.com/topics/vector-quantization-vq-vae)
[28](https://arxiv.org/abs/2207.12598)
[29](https://geometry.cs.ucl.ac.uk/courses/diffusion_ImageVideo_sigg25/)
[30](https://www.krafton.ai/en/vision-animation/6907/)
[31](https://diffusion.kaist.ac.kr)
[32](https://arxiv.org/abs/1711.00937)
[33](https://www.doptsw.com/posts/post_2024-09-17_05c95f)
[34](https://arxiv.org/abs/2403.18103)
[35](https://link.springer.com/10.1007/s00285-024-02099-4)
[36](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[37](https://dl.acm.org/doi/10.1145/3707292.3707367)
[38](https://onepetro.org/armaigs/proceedings/IGS24/IGS24/ARMA-IGS-2024-0455/632580)
[39](https://www.semanticscholar.org/paper/91d0b1987e0de1b00eb5fd8fdead8d1595800169)
[40](https://dl.acm.org/doi/10.1145/3610661.3616556)
[41](https://www.semanticscholar.org/paper/9e73a3beffc299ccabedc98512b3dc234d2b0350)
[42](https://ieeexplore.ieee.org/document/11093011/)
[43](https://azbuki.bg/uncategorized/linguistic-models-of-mass-media-genres-stylistic-diffusion-in-the-communicative-space-of-ukraine-and-bulgaria/)
[44](http://arxiv.org/pdf/2502.06768.pdf)
[45](https://aclanthology.org/2023.acl-long.248.pdf)
[46](https://arxiv.org/pdf/2406.07524.pdf)
[47](https://arxiv.org/html/2412.06787v1)
[48](https://arxiv.org/pdf/2503.04482.pdf)
[49](http://arxiv.org/abs/2304.04746)
[50](https://www.microsoft.com/en-us/microsoft-copilot/for-individuals/do-more-with-ai/ai-art-and-creativity/image-creator-improvements-dall-e-3)
[51](http://proceedings.mlr.press/v80/pu18a/pu18a.pdf)
[52](https://papers.nips.cc/paper_files/paper/2024/file/ecd92623ac899357312aaa8915853699-Paper-Conference.pdf)
[53](https://hblabgroup.com/master-dall-e-3-complete-guide/)
[54](https://en.wikipedia.org/wiki/Generative_model)
[55](https://www.instancy.com/how-to-use-dalle-3-for-best-image-generation-results/)
[56](https://www.ai-bites.net/what-does-a-cat-do/)
[57](https://openreview.net/pdf/90e47e4e3fb6ca51e582cdd699b2ba8522905bcf.pdf)
[58](https://www.semanticscholar.org/paper/91b32fc0a23f0af53229fceaae9cce43a0406d2e)
[59](https://www.semanticscholar.org/paper/95f5bafba97beb9b4f8c1fe607f04ec28efab7f9)
[60](https://www.semanticscholar.org/paper/a456a4ef8c2b7537810cb32c40a048a0e2906d60)
[61](https://source.asnt.org/enhanced-ultrasonic-characterization-of-defects-in-polycrystalline-materials-using-total-focusing-method-and-denoising-diffusion-probabilistic-models/)
[62](https://arxiv.org/abs/2302.05259)
[63](https://iopscience.iop.org/article/10.1088/1361-6560/ad209c)
[64](https://arxiv.org/abs/2311.17673)
[65](https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs)
[66](https://www.ewadirect.com/proceedings/ace/article/view/17684)
[67](https://link.springer.com/10.1007/s10851-025-01265-7)
[68](https://arxiv.org/pdf/2107.03006.pdf)
[69](http://arxiv.org/pdf/2209.15421.pdf)
[70](http://arxiv.org/pdf/2102.09672.pdf)
[71](http://arxiv.org/pdf/2405.16387.pdf)
[72](https://arxiv.org/pdf/2310.03337.pdf)
[73](https://arxiv.org/html/2405.13540v1)
[74](https://arxiv.org/pdf/2312.08153.pdf)
[75](https://letter-night.tistory.com/207)
[76](https://arxiv.org/html/2507.19002v1)
[77](https://arxiv.org/abs/2112.10752)
[78](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
[79](https://www.sciencedirect.com/science/article/abs/pii/S0925231225000943)
[80](https://mvje.tistory.com/282)
[81](https://arxiv.org/abs/2006.11239)
[82](https://iclr.cc/virtual/2023/session/13349)
[83](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ldm/)
[84](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ddpm/)
