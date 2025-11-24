# CoCa: Contrastive Captioners are Image-Text Foundation Models

### 1. 핵심 주장 및 주요 기여

CoCa는 **단일 사전학습 단계에서 세 가지 기초 모델 패러다임을 통합하는 최소한의 설계**를 제시합니다. 기존 연구는 이들을 분리된 접근 방식으로 다루었으나, CoCa는 다음을 달성합니다:[1]

- **단일 인코더 분류**: 시각적 인식 작업에 최적화된 표현 학습
- **이중 인코더 대조 학습**: 이미지-텍스트 정렬을 통한 영상-텍스트 검색 및 제로샷 분류
- **인코더-디코더 생성**: 멀티모달 이해 및 이미지 캡셔닝 능력

CoCa의 핵심 혁신은 **분리된 텍스트 디코더 아키텍처**로, 전체 사전학습 파이프라인을 단일 계산 그래프에서 효율적으로 수행합니다.[1]

***

### 2. 해결하고자 하는 문제

기존 방법들의 한계:[1]

1. **단일 인코더 모델**: 자유형 텍스트를 활용하지 않아 시각-언어 멀티모달 작업에 제한적
2. **이중 인코더 모델 (CLIP 등)**: 전체 텍스트 표현만 제공하여 VQA 같은 융합 작업에 부적합
3. **인코더-디코더 모델**: 멀티모달 이해에 강하지만 정렬된 텍스트 표현 부재로 교차모달 검색 불가
4. **다단계 사전학습**: 기존 방법들 (LiT, BASIC, ALBEF 등)은 여러 단계가 필요하여 계산 비용 증가

***

### 3. 제안하는 방법과 모델 구조

#### 3.1 자연언어 감독 원칙

**단일 인코더 분류 손실:**
$$L_{Cls} = -p(y) \log q_\theta(x)$$

여기서 $$p(y)$$는 원-핫 또는 평활화된 라벨 분포이며, $$q_\theta(x)$$는 예측 확률입니다.[1]

**이중 인코더 대조 손실:**
$$L_{Con} = -\frac{1}{N}\left(\sum_{i=1}^{N} \log \frac{\exp(x_i^\top y_i / \sigma)}{\sum_{j=1}^{N} \exp(x_i^\top y_j / \sigma)} + \sum_{i=1}^{N} \log \frac{\exp(y_i^\top x_i / \sigma)}{\sum_{j=1}^{N} \exp(y_i^\top x_j / \sigma)}\right)$$

여기서 $$x_i, y_j$$는 정규화된 이미지-텍스트 임베딩이고, $$\sigma$$는 온도 스케일입니다.[1]

**인코더-디코더 캡셔닝 손실:**
$$L_{Cap} = -\sum_{t=1}^{T} \log P_\theta(y_t|y_{<t}, x)$$

여기서 모델은 이전 토큰을 조건으로 하여 다음 토큰을 자동회귀적으로 예측합니다.[1]

#### 3.2 CoCa 통합 아키텍처

**결합 손실함수:**
$$L_{CoCa} = \lambda_{Con} \cdot L_{Con} + \lambda_{Cap} \cdot L_{Cap}$$

기본 설정: $$\lambda_{Con} = 1.0$$, $$\lambda_{Cap} = 2.0$$[1]

#### 3.3 핵심 구조적 혁신

**디커플된 텍스트 디코더 설계:**

표준 인코더-디코더와 달리, CoCa의 디코더는 두 부분으로 구성됩니다:[1]

1. **단일모달 디코더 (처음 $$n_{uni}$$개 층)**
   - 크로스-어텐션 제거
   - 입력 텍스트를 잠재 벡터로 인코딩
   - 학습 가능한 [CLS] 토큰 사용하여 단일모달 텍스트 임베딩 생성

2. **멀티모달 디코더 (나머지 $$n_{multi}$$개 층)**
   - 단일모달 디코더의 출력을 입력으로 수신
   - 시각적 인코더 출력에 크로스-어텐션 적용
   - 멀티모달 이미지-텍스트 표현 생성
   - 자동회귀적 캡셔닝 손실 적용

기본값: $$n_{uni} = n_{multi} = L/2$$ (디코더 층의 절반씩)[1]

**주의 깊은 풀링 (Attentional Pooling):**

작업별 맞춤형 풀링을 위해 다중 헤드 주의를 사용합니다:[1]

- **대조 손실용**: $$n_{query} = 1$$ (단일 글로벌 임베딩)
- **생성 손실용**: $$n_{query} = 256$$ (세밀한 특성)

```math
\text{Pooled} = \text{Attention}(\text{Query}, \text{Encoder\_Output})
```

#### 3.4 비전 모델 아키텍처

모델 크기 변형:[1]

| 모델 | 인코더 층 | 인코더 매개변수 | 단일모달 디코더 | 멀티모달 디코더 | 전체 매개변수 |
|------|----------|-----------------|-----------------|-----------------|----------------|
| CoCa-Base | 12 | 86M | 6 | 6 | 383M |
| CoCa-Large | 24 | 303M | 6 | 6 | 787M |
| CoCa | 40 | 1B | 9 | 9 | 2.1B |

모두 Vision Transformer (ViT) 기반 인코더 사용[1]

***

### 4. 성능 향상 및 실증적 결과

#### 4.1 시각 인식 작업

**ImageNet 분류:**[1]

| 평가 방식 | CoCa | 기존 최고 성능 |
|-----------|------|----------------|
| 제로샷 | 86.3% | 85.7% (BASIC) |
| 고정 인코더 | 90.6% | 90.1% (Florence) |
| 미세조정 | 91.0% | 90.9% (CoAtNet) |

**비디오 인식 (Kinetics 데이터셋):**[1]

| 데이터셋 | CoCa (고정) | CoCa (미세조정) |
|---------|-----------|-----------------|
| Kinetics-400 | 88.0% | 88.9% |
| Kinetics-600 | 88.5% | 89.4% |
| Kinetics-700 | 81.1% | 82.7% |

#### 4.2 교차모달 정렬 작업

**제로샷 이미지-텍스트 검색 (MSCOCO):**[1]

| 작업 방향 | R@1 | R@5 | R@10 |
|---------|-----|-----|------|
| Image → Text | 66.3% | 86.2% | 91.8% |
| Text → Image | 51.2% | 74.2% | 82.0% |

ALIGN, Florence 등을 상회합니다.

**제로샷 다중 분포 일반화:**[1]

CoCa는 여러 분포 편이에서 우수한 일반화를 보입니다:
- ImageNet: 86.3%
- ImageNet-V2: 80.7%
- ImageNet-R: 96.5%
- ObjectNet: 82.7%
- **평균**: 85.7% (BASIC의 83.7% vs 향상)

#### 4.3 멀티모달 이해 작업

**VQA v2, SNLI-VE, NLVR2:**[1]

| 작업 | CoCa | SimVLM | METER |
|------|------|--------|-------|
| VQA | 82.3% | 80.0% | 80.3% |
| SNLI-VE | 87.1% | 86.3% | - |
| NLVR2 | 87.0% | 85.2% | - |

#### 4.4 이미지 캡셔닝

**NoCaps 벤치마크 (최고 난이도):**[1]

- **CIDEr Score**: 120.6 (선행 방법들 110-112 vs)
- 메트릭 최적화 없이 달성
- 기존 SOTA 메서드 (OFA, LEMON)를 초과

***

### 5. 일반화 성능 향상 분석

#### 5.1 제로샷 견고성

CoCa의 설계가 일반화를 개선하는 이유:[1]

1. **이중 목표 시너지**
   - 대조 손실: 글로벌 표현 정렬 강화
   - 캡셔닝 손실: 세밀한 지역 수준 특성 강화
   - 결과: 다양한 추론 필요 작업에 적응

2. **단일 단계 사전학습**
   - JFT-3B (라벨 텍스트) + ALIGN (웹 스케일 alt-text) 동시 학습
   - 각 데이터셋의 강점을 즉시 활용
   - 다단계 방법 (LiT, BASIC)보다 효율적

3. **주의 깊은 풀링의 역할**
   - 작업별 특정 적응 가능
   - 사전학습된 표현을 손상시키지 않음
   - 고정 특성 평가에서 경쟁력 유지

#### 5.2 축척 법칙 (Scaling Laws)

CoCa는 모델 크기에 따른 일관된 성능 개선을 보입니다:[1]

- **제로샷 분류**: 기울기 감소는 완만 (로그 스케일)
- **미세조정**: 모델 크기에 따른 수렴 개선 지속

#### 5.3 분포 외 일반화

ImageNet 분포 편이 실험에서:[1]

CoCa의 평균 성능 85.7%는 감독형 ViT-G (90.5% → 편이에서 저하)보다 **더 견고함**을 시사합니다.

***

### 6. 모델의 한계

#### 6.1 사전학습 데이터 편향

CoCa는 JFT-3B와 ALIGN 데이터를 사용하며, 이들 데이터셋의 편향을 상속합니다:[1]

- 웹 스케일 이미지-텍스트 데이터의 고유 편향
- 특정 도메인 (의료, 위성 이미지 등)에 대한 부족한 표현

#### 6.2 합성 분포 편이에 대한 취약성

최근 연구 (2024)에 따르면, CLIP 기반 모델은 합성 분포 편이 및 적대적 공격에 약함:[2]

- 자연 분포 편이: 경쟁력 있음
- 합성 분포 편이: 유의미한 성능 저하
- 적대적 공격: 감독형 모델보다 취약

#### 6.3 저해상도 이미지 취약성

2025년 연구에 따르면, 파운데이션 모델의 일반화 한계:[3][4]

- 저해상도 이미지에서 성능 급격한 하락
- 모델 크기 증가해도 완전히 해결 불가
- 이미지 초기 층에 더 큰 영향

***

### 7. 삭제 실험 (Ablation Study)

#### 7.1 훈련 목표 효과[1]

| 손실 함수 | 제로샷 분류 | VQA |
|---------|-----------|-----|
| $$L_{Con}$$ 만 | 70.7% | 59.2%* |
| $$L_{Cap}$$ 만 | - | 68.9% |
| $$L_{CoCa}$$ | 71.6% | 69.0% |

*VQA는 단일 대조 손실로는 추가 융합 필요

#### 7.2 손실 가중치 (Table 8c)[1]

$$\lambda_{Cap} : \lambda_{Con} = 2:1$$

이 비율이 최적이며, 1:1보다 모든 작업에서 우수합니다.

#### 7.3 디코더 분할[1]

$$n_{uni}$$의 최적값은 6-9 사이이며, 전체 절반으로 설정하면:
- 제로샷 분류 충분히 강함
- 멀티모달 이해 최적화

***

### 8. 전산 효율성

**사전학습 비용:**[1]
- 2,048개 CloudTPUv4 칩에서 약 5일
- 배치 크기: 65,536 (JFT:ALIGN = 1:1)
- 500k 단계 (약 5 에포크 JFT, 10 에포크 ALIGN)

**효율성 비교 (Table 8b):**[1]

| 방법 | TPU 비용 |
|------|----------|
| 순수 캡셔닝 | 1.0× |
| CoCa | 1.18× |

단일 순전파-역전파로 두 목표 모두 계산 가능하여 오버헤드 최소화

***

### 9. 논문의 앞으로의 영향 및 최신 동향

#### 9.1 후속 연구

**1. VideoCoCa (2023)**[5]
- CoCa를 비디오-텍스트 도메인으로 확장
- 사전학습된 CoCa를 최소 추가 훈련으로 적응

**2. SyCoCa (2024)**[6]
- 대칭적 대조 학습으로 CoCa 개선
- 주의 깊은 마스킹을 통한 멀티모달 정렬 강화

**3. 3D CoCa (2025)**[7]
- 점 클라우드 3D 장면에 CoCa 패러다임 확장
- 3D 캡셔닝과 3D 검색 능력

**4. 의료 이미지 응용**[8][9]
- fMRI 캡셔닝 (뇌 활동 → 텍스트)
- 의료 이미지 분할에 비전-언어 의미 집계

#### 9.2 최신 연구 동향 (2024-2025)

**1. 데이터 품질의 중요성**

2024년 Carnegie Mellon 연구에 따르면:[10]

파운데이션 모델 훈련에서:
- 데이터셋 크기보다 **품질이 더 중요**
- 저품질 샘플 반복 사용 시 유틸리티 급감
- 고품질 일부 반복 vs 저품질 신규 샘플 간 트레이드오프

**2. 시뮬된 분포 편이에 대한 취약성**

2024년 벤치마크 연구:[2]
- CLIP 기반 모델의 제로샷 견고성 실증
- 합성 편이 + 적대적 공격에 심각한 성능 저하
- 자연 분포 편이에서의 강인성은 **데이터 중복**으로 설명 가능

**3. 저해상도에서의 강인성 (2025)**

LR0.FM 벤치마크:[4][3]
- 모델 크기와 해상도 견고성 간 양의 상관관계
- 미세조정된 고해상도 모델이 저해상도에 더 취약
- 단순 전략 (LR-TK0)으로 개선 가능

**4. 파운데이션 모델의 연합 학습 (2024)**

비이드 학습에서의 CoCa 활용:[11]

$$\text{성능} = \text{신호 학습 - 노이즈 암기}$$

프롬프트 포트폴리오를 통해 일반화-개인화 균형 유지

**5. 비전-언어-행동 모델 (2025)**

로봇 조작에서 CoCa 기반 모델 활용:[12]

- 사전학습된 특성 보존이 일반화의 핵심
- Dual-encoder 설계로 표현 강화
- 공동 훈련 전략이 견고성 향상

***

### 10. 향후 연구 시 고려할 점

#### 10.1 견고성 개선

1. **합성 분포 편이 해결**
   - 적대적 훈련 통합
   - 다양한 인공 변환에 대한 사전학습 데이터 확대

2. **저해상도 적응**
   - 다중 해상도 사전학습
   - 해상도-인식 풀링 전략

3. **도메인 특화**
   - 의료, 위성 이미지 등 전문 도메인 데이터 통합
   - 도메인별 수정 전략

#### 10.2 스케일링 효율성

1. **계산 효율 최적화**
   - 주의 메커니즘의 희소성 도입
   - 혼합 정밀도 훈련

2. **데이터 효율**
   - 고품질 데이터 큐레이션 자동화
   - 저자원 언어/도메인 지원

#### 10.3 멀티모달 확장

1. **더 많은 모달리티**
   - 오디오 통합 (음성, 음악)
   - 포인트 클라우드 (이미 3D CoCa로 시작)

2. **동적 아키텍처**
   - 작업에 따른 자동 구조 조정
   - 신경 아키텍처 탐색

#### 10.4 공정성 및 윤리

1. **편향 분석**
   - 사전학습 데이터 편향 체계적 평가
   - 성능 간격 분석 (인구통계 기반)

2. **미용 영향**
   - 모델 투명성 문서화
   - 책임 있는 배포 프레임워크

***

### 결론

CoCa는 시각-언어 기초 모델 설계에서 획기적 단계를 나타냅니다. **단일 사전학습 단계에서 세 가지 패러다임을 통합**함으로써 계산 효율성과 성능을 동시에 달성합니다. 제로샷 이미지 분류에서 86.3%, 미세조정 후 91.0%의 ImageNet 성능은 단일 모델이 여러 특화 모델을 능가할 수 있음을 증명합니다.

그러나 **합성 분포 편이, 저해상도 이미지, 도메인 편향** 등의 한계가 존재합니다. 2024-2025년의 최신 연구는 데이터 품질 최적화, 견고성 개선, 멀티모달 확장이 향후 핵심 방향임을 보여줍니다. CoCa의 설계 원칙—특히 분리된 디코더와 작업별 풀링—은 향후 기초 모델 개발의 중요한 참고점이 될 것입니다.[5][6][11][8][7][3][4][2][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/18b35baf-3b81-49e7-b217-11455807f5e6/2205.01917v2.pdf)
[2](https://arxiv.org/abs/2403.10499)
[3](https://openreview.net/forum?id=AsFxRSLtqR)
[4](https://arxiv.org/abs/2502.03950)
[5](https://arxiv.org/pdf/2212.04979.pdf)
[6](https://arxiv.org/pdf/2401.02137.pdf)
[7](https://arxiv.org/html/2504.09518v1)
[8](https://arxiv.org/abs/2509.08570)
[9](https://papers.miccai.org/miccai-2025/paper/2049_paper.pdf)
[10](https://www.deeplearning.ai/the-batch/scaling-laws-reveal-the-impact-of-data-quality-in-vision-language-model-training/)
[11](https://openreview.net/forum?id=Y4L8GQXZZO)
[12](https://seohyun00.tistory.com/53)
[13](https://qims.amegroups.com/article/view/126997/html)
[14](https://ashpublications.org/blood/article/144/Supplement%201/3306/532952/Implementing-Revised-Reference-Intervals-for-Free)
[15](https://link.springer.com/10.1007/s13300-024-01595-6)
[16](https://ascopubs.org/doi/10.1200/JCO.2024.42.16_suppl.e23321)
[17](https://aacrjournals.org/cancerres/article/84/6_Supplement/3785/738443/Abstract-3785-Clinical-and-genomic)
[18](https://aacrjournals.org/cancerres/article/84/6_Supplement/776/736775/Abstract-776-Change-in-background-contrast)
[19](https://academic.oup.com/jes/article/doi/10.1210/jendso/bvae163.1347/7812380)
[20](https://aacrjournals.org/cancerres/article/84/6_Supplement/4798/735381/Abstract-4798-MRI-screening-for-brain-metastases)
[21](https://academic.oup.com/eurjpc/article/doi/10.1093/eurjpc/zwaf236.340/8137089)
[22](https://aseestant.ceon.rs/index.php/jomb/article/view/56983)
[23](http://arxiv.org/pdf/2205.01917v1.pdf)
[24](https://aclanthology.org/2023.findings-acl.120.pdf)
[25](https://arxiv.org/pdf/2406.07584.pdf)
[26](http://arxiv.org/pdf/2311.14977.pdf)
[27](https://arxiv.org/abs/2107.09990)
[28](https://arxiv.org/pdf/2205.05357.pdf)
[29](https://arxiv.org/abs/2205.01917)
[30](https://www.sciencedirect.com/science/article/abs/pii/S1566253525009182)
[31](https://aclanthology.org/2025.naacl-short.26.pdf)
[32](https://aclanthology.org/2023.findings-eacl.169/)
[33](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/coca/)
[34](https://ieeexplore.ieee.org/document/10571357/)
[35](http://www.emerald.com/ilt/article/77/2/211-218/1239749)
[36](https://aacrjournals.org/clincancerres/article/30/21_Supplement/PR019/749437/Abstract-PR019-Transfer-learning-for-accurate)
[37](https://ieeexplore.ieee.org/document/10543109/)
[38](https://intelektn.donnuet.edu.ua/index.php/intelekt/article/view/98)
[39](https://ieeexplore.ieee.org/document/10933367/)
[40](https://journals.lww.com/10.1097/JS9.0000000000002278)
[41](https://link.springer.com/10.1007/s12530-025-09686-w)
[42](https://aacrjournals.org/clincancerres/article/31/12_Supplement/P4-09-02/752346/Abstract-P4-09-02-Multiparametric-MRI-and-Transfer)
[43](https://arxiv.org/abs/2507.14239)
[44](http://arxiv.org/pdf/2502.17744.pdf)
[45](https://arxiv.org/pdf/2304.04947.pdf)
[46](https://arxiv.org/abs/1909.01331)
[47](http://arxiv.org/pdf/2407.02542.pdf)
[48](http://arxiv.org/pdf/2210.02655.pdf)
[49](https://arxiv.org/html/2204.12833v3)
[50](http://arxiv.org/pdf/2207.05510.pdf)
[51](https://arxiv.org/pdf/2501.10933.pdf)
[52](https://arxiv.org/abs/2207.05377)
[53](https://aclanthology.org/2025.acl-long.1595/)
[54](https://aclanthology.org/2024.emnlp-main.124.pdf)
[55](https://proceedings.neurips.cc/paper_files/paper/2024/file/aee5298251a418aad89618cf6b5e7ccc-Paper-Conference.pdf)
[56](https://journals.sagepub.com/doi/abs/10.1177/02783649241273565)
[57](https://proceedings.neurips.cc/paper_files/paper/2024/file/6b7e1e96243c9edc378f85e7d232e415-Paper-Conference.pdf)
