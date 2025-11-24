# PaLI: A Jointly-Scaled Multilingual Language-Image Model

### 1. 핵심 주장과 주요 기여[1]

**PaLI(Pathways Language and Image model)**는 비전과 언어를 결합한 대규모 다중 모달 모델로, 다음과 같은 핵심 주장을 제시합니다:[1]

1. **균형잡힌 모달리티 확장의 중요성**: 기존 모델들이 언어 모델을 중심으로 확장한 것과 달리, PaLI는 비전 컴포넌트(ViT-e, 4B 파라미터)와 언어 컴포넌트(mT5-XXL, 13B 파라미터)를 균형있게 확장합니다. 이는 PaLI-17B 모델의 총 파라미터 중 약 25%를 비전에 할당하는 구조로 반영됩니다.[1]

2. **다중 목표 학습의 효과성**: 8가지 사전학습 목표를 혼합하여 학습하면 다양한 비전-언어 작업에서 우수한 성능을 달성할 수 있습니다.[1]

3. **100개 이상 언어에 대한 확장성**: WebLI 데이터셋(10억 개 이미지, 100개 이상 언어)을 통해 다중언어 기능을 갖춘 모델을 만들 수 있습니다.[1]

### 2. 해결하는 문제와 제안하는 방법

#### 2.1 문제점[1]

- **파라미터 불균형**: 기존 모델들(예: GIT2는 비전에 4.8B, 언어에 0.3B)에서 비전과 언어 컴포넌트 간 파라미터 분배가 불균등합니다[1]
- **영어 중심 데이터셋**: 기존 대규모 이미지-텍스트 데이터셋(CC3M, ALIGN 등)이 영어 중심이어서 다중언어 능력이 제한됩니다[1]
- **작업 특화 모델의 필요성**: 이미지 분류, VQA, 캡션링 등 각 작업마다 다른 아키텍처가 필요하던 문제[1]

#### 2.2 제안 방법

**모델 아키텍처**:[1]
$$\text{PaLI} = \text{Vision Encoder (ViT)} + \text{Text Encoder-Decoder (mT5)}$$

모델은 입력으로 이미지와 텍스트 스트링을 받아 텍스트를 생성하는 통합 인터페이스를 사용합니다. 비전 트랜스포머의 출력 패치 특성(visual tokens)이 크로스 어텐션을 통해 텍스트 엔코더-디코더에 전달됩니다.[1]

**손실함수**:[1]
$$\mathcal{L} = \mathcal{L}_{\text{cross-entropy}}(\hat{y}, y)$$

각 작업에 대해 표준 소프트맥스 크로스 엔트로피 손실을 사용하는 teacher forcing 방식입니다.[1]

**사전학습 목표 혼합**(1.6B 예제):[1]
1. **Span Corruption** (텍스트 전용): 100M 예제
2. **Split-Captioning** (WebLI): 1,000M 예제  
3. **Captioning** (CC3M-35L): 100M 예제
4. **OCR** (WebLI): 100M 예제
5. **VQA** (다국어): 100M 예제
6. **VQG** (시각적 질문 생성): 100M 예제
7. **Object-Aware VQA**: 100M 예제
8. **Object Detection**: 50M 예제

### 3. 모델 구조와 주요 컴포넌트

#### 3.1 비전 컴포넌트: ViT-e[1]

새로운 **ViT-e(enormous)** 모델을 제안합니다:
- **파라미터**: 4B (기존 ViT-G는 1.8B)[1]
- **구조**: 
  - Width: 1792
  - Depth: 56
  - MLP Dimensions: 15360
  - Heads: 16
- **성능**:
  - ImageNet: 90.9% (ViT-G 대비 미미한 개선)[1]
  - ObjectNet: 84.9% (영역외 데이터에서 큰 개선)[1]
  
ViT-e는 ImageNet 같은 단순 분류 작업에서는 포화 현상을 보이지만, 비전-언어 작업에서는 COCO 캡셔닝에서 약 3 CIDEr 포인트의 개선을 달성합니다.[1]

#### 3.2 언어 컴포넌트: mT5[1]

- **mT5-XXL (13B 파라미터)**: 다중언어 텍스트 인코더-디코더
- **사전학습된 체크포인트 재사용**: 기존 언어 이해 및 생성 능력 유지[1]
- **부분 동결 전략**: 사전학습 중 비전 컴포넌트만 동결하고 언어 파라미터만 업데이트[1]

#### 3.3 WebLI 데이터셋[1]

- **규모**: 10억 개 이미지, 120억 개 대체 텍스트
- **언어 커버리지**: 109개 언어
- **품질 필터링**: 크로스 모달 유사도 기반 상위 10%만 선택 (약 1B 예제)
- **중복 제거**: 68개 벤치마크 데이터셋에 대한 이미지 중복 제거[1]

### 4. 성능 향상 및 실험 결과

#### 4.1 이미지 캡셔닝[1]

| 벤치마크 | PaLI-17B | SOTA 기준 | 개선 |
|---------|----------|---------|-----|
| COCO (Karpathy) | 149.1 CIDEr | GIT2: 145.0 | +4.1 |
| NoCaps | 124.4 CIDEr | GIT2: 124.8 | -0.4 |
| TextCaps | 160.4 CIDEr | GIT2: 145.0 | +15.4 |
| Crossmodal-3600 (35개 언어 평균) | 53.6 CIDEr | 기준: 28.9 | +24.7 |

#### 4.2 시각적 질문 답변(VQA)[1]

| 벤치마크 | PaLI-17B | 기준값 |
|---------|----------|-------|
| VQAv2 | 84.3% | Flamingo: 82.0% (2.2% 개선) |
| OKVQA | 64.5% | KAT: 54.4% (10.1% 개선) |
| TextVQA | 73.06% | GIT2: 67.27% |
| xGQA (다국어) | 52.1% 평균 | MPT: 39.8% |
| MaXM (다국어) | 56.3% 평균 | MPT: 36.6% |

#### 4.3 모델 확장 효과[1]

$$\text{평균 성능 개선} = 3.2 \text{ (ViT-G} \to \text{ViT-e)} + 3.1 \text{ (mT5-L} \to \text{mT5-XXL)}$$

비전 컴포넌트 확장(2B 파라미터 추가)이 언어 컴포넌트 확장(12B 추가)보다 단위 파라미터당 더 큰 개선을 제공합니다.[1]

$$\text{Return on Investment (ViT)} = 2.2 \text{ points/1B parameters}$$
$$\text{Return on Investment (mT5)} = 0.4 \text{ points/1B parameters}$$

#### 4.4 영역외(Out-of-Distribution) 일반화[1]

| 데이터셋 | PaLI-17B | 기준값 |
|---------|----------|-------|
| ImageNet (제로샷) | 72.11% | Flamingo 1-shot: 71.9% |
| ImageNet-A | 44.70% | 이전 모델들 기준값 대비 크게 개선 |
| ImageNet-Sketch | 63.83% | - |
| ObjectNet | 42.62% | - |

고해상도 사전학습(588×588) 추가 단계가 2.0 포인트의 추가 개선을 제공합니다.[1]

### 5. 일반화 성능 향상 가능성 분석

#### 5.1 다중언어 능력[1]

**Crossmodal-3600 벤치마크** (35개 언어):
- 영어: 98.1 CIDEr
- 프랑스어: 75.5 CIDEr  
- 힌디어: 31.3 CIDEr
- 히브리어: 46.8 CIDEr
- 태국어: 72.1 CIDEr
- 중국어: 36.5 CIDEr

**역번역 검증**: 비영어 언어의 캡션을 영어로 역번역하여 비교하면, 언어 간 성능 일관성이 상당히 높음을 확인할 수 있습니다.[1]

#### 5.2 다중 작업 일반화[1]

**ablation 연구** 결과:
- Split-captioning 목표가 가장 중요 (-6.3 CIDEr 제거 시)
- 객체 인식 컴포넌트 제거 시 -2.9 CIDEr (Crossmodal-3600)
- OCR 목표는 TextVQA에는 +1.7, 캡셔닝에는 -1.7 (트레이드오프)

#### 5.3 언어 이해 능력 유지[1]

PaLI-17B는 mT5-XXL과 비교하면:
- **SuperGLUE** (영어 전용): 88.2% vs 89.2% (매우 근접)
- **XNLI** (다국어): 84.9% vs 84.5% (동등 수준)
- **XQuAD/TyDiQA** (QA 작업): 동등한 성능

비전-언어 학습이 기본 언어 능력을 거의 손상시키지 않습니다.[1]

### 6. 한계와 제약사항[1]

#### 6.1 아키텍처 수준의 한계

1. **복잡한 장면 이해 부족**: 많은 객체가 있는 복잡한 장면에서 상세한 설명을 제공하지 못합니다[1]
2. **다국어 미세조정의 문제**: 영어 전용 데이터로 미세조정하면 다언어 능력이 손실됩니다[1]

#### 6.2 평가 관련 제약

1. **오픈 어휘 VQA 엄격함**: 생성된 답변이 정확히 일치해야 하므로, 동의어나 의역은 오류로 간주됩니다[1]
2. **서방 중심 편향**: 벤치마크가 서방식 편향을 반영할 수 있습니다[1]

#### 6.3 다중성 할루시네이션[1]

최신 연구(2024-2025)에서 다중 객체 할루시네이션이 보고되었습니다:
- 다국어 입력에서 영어보다 높은 할루시네이션 비율[2]
- 중간 계층의 언어별 주의 패턴 차이[2]

### 7. 최신 연구 기반 영향 및 향후 고려사항 (2024-2025)

#### 7.1 후속 연구와 발전[3][4]

**PaLI-X (2024)**:[3]
- PaLI를 기반으로 한 확장 모델
- 더 큰 용량과 넓은 작업 혼합
- 이미지 기반 문서 이해(DocVQA) 등 새로운 작업 추가
- 영상 이해 능력 추가[5]

#### 7.2 확장 법칙 연구 (2024-2025)[6][7][8]

**ViT-22B 개발** (2024):
- ViT-e의 5.5배 규모 (22B 파라미터)[8]
- QK 정규화, 비동기 병렬 선형 연산으로 안정성 개선[8]
- OOD 일반화 성능 지속적 개선[8]

**확장 법칙 발견**:
- 초기 융합(early fusion) vs 후기 융합(late fusion) 아키텍처 비교[6]
- 초기 융합이 동일 FLOPs에서 더 나은 훈련 효율성 제공[6]

#### 7.3 일반화 성능 개선 방향[9][10]

**도메인 외 일반화의 문제점** (2024):
- MLLMs는 훈련 도메인과 다른 분포에서 성능 저하[10][9]
- **원인 분석**: 
  - 의미론적 오역해석 (semantic misinterpretation)
  - 시각적 특성 추출 부족 (visual feature extraction insufficiency)
  - **매핑 부족** (mapping deficiency) - 주요 원인[9]

**해결책**:
- In-context learning (ICL): 성능을 크게 개선할 수 있음[9]
- 도메인별 후훈련 (domain-specific post-training)[11][12]
- 언어별 계층 미세조정 (PLAST 방법)[13]

#### 7.4 멀티모달 적응 및 일반화 연구 (2025)[14][15]

**5가지 멀티모달 적응 시나리오**:[14]
1. 멀티모달 도메인 적응
2. 멀티모달 테스트타임 적응
3. 멀티모달 도메인 일반화
4. 기초 모델의 지원을 받은 단일 모달 적응
5. 멀티모달 기초 모델의 적응

#### 7.5 할루시네이션 및 편향 완화[16][17][2]

**다중 객체 할루시네이션** (2024):
- 이질적 쿼리에서 더 많은 할루시네이션 발생[16]
- 객체 클래스 분포가 할루시네이션 행동에 영향[16]

**다국어 할루시네이션 완화** (2024):
- CLAIM: 언어별 주의 패턴 정렬[2]
- 스페인어에서 최대 30% 개선[2]

#### 7.6 과제 특화 모델의 부상[18][19]

**일반주의자 모델의 한계**:
- VisionLLM v2: 수백 개 작업에 대한 통합 학습[19]
- Flamingo: 몇 샷 학습 능력 강화[1]
- 작업 특화 모델과의 성능 간격 여전히 존재[18]

#### 7.8 향후 연구 시 고려사항

**1. 매핑 부족 해결**:
- 더 나은 시각-언어 정렬 메커니즘 개발
- 도메인 적응 기법의 효율성 개선

**2. 계산 효율성**:
- 모달리티 간 균형 유지하면서 경량화
- 토큰 압축 기술 고도화

**3. 공정성 및 편향**:
- 다중언어 데이터의 품질 및 표현성 개선
- 저자원 언어에 대한 성능 격차 해소

**4. 긴급 작업별 미세조정**:
- 다중 작업 학습 시 작업 간 간섭 최소화
- 지속적 학습 (continual learning) 메커니즘

**5. 평가 메트릭 개선**:
- 정확도 일치를 넘어선 의미론적 유사성 평가
- 비서방식 관점의 벤치마크 개발

### 결론

PaLI는 **균형잡힌 다중 모달 확장**, **다중 목표 학습**, **광범위한 다중언어 데이터** 등을 통해 비전-언어 모델의 일반화 성능을 크게 향상시켰습니다. 특히 ViT-e의 도입과 WebLI 데이터셋은 향후 멀티모달 모델 개발의 중요한 이정표가 되었습니다. 

그러나 여전히 도메인 외 일반화, 다중언어 할루시네이션, 계산 효율성 등의 과제가 남아있으며, 2024-2025년 최신 연구는 이들 문제를 해결하기 위한 방향으로 진행되고 있습니다. 특히 인-컨텍스트 학습, 도메인별 후훈련, 언어별 적응 등의 기법들이 PaLI 이후 모델들의 일반화 성능을 한 단계 더 향상시키고 있습니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/57ac9995-775f-4f47-8514-8b69079dd397/2209.06794v4.pdf)
[2](https://arxiv.org/html/2506.11073v1)
[3](https://arxiv.org/pdf/2305.18565.pdf)
[4](https://research.google/pubs/pali-x-on-scaling-up-a-multilingual-vision-and-language-model/)
[5](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_On_Scaling_Up_a_Multilingual_Vision_and_Language_Model_CVPR_2024_paper.pdf)
[6](https://openaccess.thecvf.com/content/ICCV2025/papers/Shukor_Scaling_Laws_for_Native_Multimodal_Models_ICCV_2025_paper.pdf)
[7](https://proceedings.mlr.press/v202/dehghani23a.html)
[8](https://research.google/blog/scaling-vision-transformers-to-22-billion-parameters/)
[9](https://arxiv.org/abs/2402.06599)
[10](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_On_the_Out-Of-Distribution_Generalization_of_Large_Multimodal_Models_CVPR_2025_paper.pdf)
[11](https://aclanthology.org/2025.findings-emnlp.17)
[12](http://arxiv.org/pdf/2411.19930.pdf)
[13](https://aclanthology.org/2025.findings-emnlp.666.pdf)
[14](https://arxiv.org/html/2501.18592)
[15](https://github.com/donghao51/Awesome-Multimodal-Adaptation)
[16](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ea4a1ea4d9ff273688c8e92bd087112-Paper-Conference.pdf)
[17](https://aclanthology.org/2024.findings-acl.937.pdf)
[18](http://arxiv.org/pdf/2406.08394v1.pdf)
[19](https://arxiv.org/abs/2406.08394)
[20](https://oarjst.com/node/710)
[21](https://kinetik.umm.ac.id/index.php/kinetik/article/view/2305)
[22](https://invergejournals.com/index.php/ijss/article/view/161)
[23](https://arxiv.org/abs/2402.14818)
[24](http://arxiv.org/pdf/2407.07726v1.pdf)
[25](http://arxiv.org/pdf/2206.11091.pdf)
[26](https://arxiv.org/abs/2209.06794)
[27](http://arxiv.org/pdf/2501.02189.pdf)
[28](http://arxiv.org/pdf/2305.11175.pdf)
[29](https://pmc.ncbi.nlm.nih.gov/articles/PMC11599617/)
[30](https://arxiv.org/abs/2404.08589)
[31](https://openaccess.thecvf.com/content/CVPR2024W/PV/papers/Ozdemir_Enhancing_Visual_Question_Answering_through_Question-Driven_Image_Captions_as_Prompts_CVPRW_2024_paper.pdf)
[32](https://arxiv.org/html/2504.10462v1)
[33](https://www.sciencedirect.com/science/article/abs/pii/S0957417425001009)
[34](https://ieeexplore.ieee.org/document/10666846/)
[35](https://arxiv.org/abs/2407.21534)
[36](https://arxiv.org/abs/2405.01591)
[37](https://arxiv.org/abs/2412.03665)
[38](https://arxiv.org/abs/2406.15534)
[39](https://arxiv.org/abs/2409.08543)
[40](https://aclanthology.org/2024.naacl-long.97)
[41](https://arxiv.org/pdf/2410.05608.pdf)
[42](https://arxiv.org/html/2409.15657)
[43](https://arxiv.org/pdf/2308.11217.pdf)
[44](https://arxiv.org/abs/2409.03444)
[45](http://arxiv.org/pdf/2402.06599.pdf)
[46](https://arxiv.org/html/2304.14178v2)
[47](https://www.linkedin.com/pulse/future-vision-language-models-scaling-efficiency-performance-thia-8korc)
[48](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3076133f08b40607d00a8f48f6acd71c-Abstract-Conference.html)
