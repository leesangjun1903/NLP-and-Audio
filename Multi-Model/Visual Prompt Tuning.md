# Visual Prompt Tuning

**Visual Prompt Tuning(VPT)**는 2022년 ECCV에 발표된 획기적인 논문으로, 대규모 Vision Transformer 모델을 다운스트림 작업에 효율적으로 적응시키는 방법을 제시합니다. 이 논문의 핵심 주장은 다음과 같습니다.[1]

**핵심 주장**: 기존의 Full Fine-tuning 방식을 대체하여, 백본 모델은 고정하면서 입력 공간에 소량의 학습 가능한 매개변수(프롬프트)만 추가함으로써 **효율성과 성능을 동시에 달성할 수 있다**는 것입니다.[1]

**주요 기여**:

1. **파라미터 효율성**: 모델 파라미터의 1% 미만(0.04~0.53%)의 학습 가능 파라미터만 추가하면서도 완전 미세조정 성능을 초과합니다.[1]

2. **실증적 우수성**: 24개의 다양한 다운스트림 태스크에서 VPT-Deep이 20개 경우에서 완전 미세조정을 능가합니다.[1]

3. **저장 비용 감소**: 각 작업별로 프롬프트와 분류 헤드만 저장하면 되므로, 멀티태스크 환경에서 저장 공간을 대폭 절감합니다.[1]

4. **NLP와의 차이점 증명**: NLP의 프롬프트 튜닝이 완전 미세조정 성능과 동등한 수준에 그친 반면, 비전 분야에서는 **프롬프트 튜닝이 완전 미세조정을 상회**한다는 독특한 특성을 발견했습니다.[1]

***

## 문제 정의, 제안 방법 및 모델 구조

### 1. 해결하고자 하는 문제

**기본 문제**: Vision Transformer(ViT)와 같은 대규모 사전학습 모델(ViT-Huge: 630M 파라미터)을 새로운 작업에 적응시키려면 전체 백본 파라미터를 업데이트해야 하므로:[1]

- 각 작업마다 전체 모델 복사본을 저장해야 함
- 극단적인 저장 비용 증가 (ViT-Base의 경우 작업당 86M 파라미터)
- 배포 및 관리의 실용성 저하

기존 매개변수 효율적 방법들(Adapter, BiTFit, 부분 미세조정)은 저장 비용 절감하지만 성능이 저하되는 trade-off 문제가 있습니다.[1]

### 2. 제안하는 방법 (VPT)

#### VPT-Shallow 방식의 수식 표현

입력 이미지를 패치로 나누고 임베딩한 후, 첫 Transformer 레이어의 입력에만 프롬프트를 추가합니다:[1]

$$x_1, Z_1, E_1 = L_1(\textcolor{white}{x_0, P}, E_0)$$

여기서:
- $E_0 \in \mathbb{R}^{m \times d}$: 패치 임베딩
- $P = \{p_k \in \mathbb{R}^d | k = 1, 2, \ldots, p\}$: 학습 가능한 프롬프트 토큰
- $x_0$: CLS 토큰
- $m$: 패치 개수, $d$: 임베딩 차원

이후 레이어들은 다음과 같이 표현됩니다:

$$x_i, Z_i, E_i = L_i(x_{i-1}, Z_{i-1}, E_{i-1}), \quad i = 2, 3, \ldots, N$$

최종 출력은 마지막 레이어의 CLS 토큰을 사용하여 분류 헤드로 전달됩니다:

$$y = \text{Head}(x_N)$$

#### VPT-Deep 방식의 수식 표현

모든 Transformer 레이어에 프롬프트를 추가하는 방식입니다:[1]

$$x_i, Z_i, E_i = L_i(x_{i-1}, \textcolor{white}{P_{i-1}}, E_{i-1}), \quad i = 1, 2, \ldots, N$$

여기서 $P_{i-1} = \{p_k^i \in \mathbb{R}^d | k = 1, 2, \ldots, p\}$는 $i-1$ 번째 레이어의 프롬프트입니다.

#### 프롬프트 저장 효율성

ViT-Base (d=768)의 경우:[1]

- **VPT-Shallow** (50개 프롬프트): $p \times d = 50 \times 768 = 0.038$ M (0.04%)
- **VPT-Deep** (50개 프롬프트): $N \times p \times d = 12 \times 50 \times 768 = 0.46$ M (0.53%)

### 3. 모델 구조의 핵심 특징

#### 왜 Latent Space에 프롬프트를 추가하는가?

논문에서는 여러 프롬프트 위치 전략을 비교합니다:[1]

| 전략 | VTAB-Natural | VTAB-Specialized | VTAB-Structured |
|------|--------------|-----------------|-----------------|
| **Prepend (Latent)** | 78.5 | 82.4 | 55.0 |
| Prepend-pixel | 71.5 | 78.0 | 44.5 |
| Concat-channel | 74.0 | 80.2 | 38.6 |
| Add (Element-wise) | 77.5 | 79.7 | 47.0 |

Latent 공간에서 프롬프트를 연결(Prepend)하는 것이 최적인 이유는:

1. **위치 불변성**: Vision Transformer에서는 위치 인코딩이 이미 적용되었으므로, 프롬프트의 위치가 연산 결과에 영향을 주지 않습니다.[1]

2. **학습 효율성**: Latent 공간에서는 더 응축된 작업 특화 신호를 학습할 수 있습니다.[1]

#### Transformer 레이어의 구조 활용

각 Transformer 레이어는 다음과 같은 구조를 가집니다:[1]

$$L_i = \text{LayerNorm} + \text{FFN} \circ (\text{LayerNorm} + \text{MSA})$$

프롬프트는 이 구조 내에서 병렬 시퀀스로 작동하므로, Multihead Self-Attention(MSA)에서 다른 패치 토큰과 상호작용합니다.

***

## 성능 향상 분석

### 1. 종합 성능 비교

ViT-B16 백본으로 24개 다운스트림 작업에서의 결과:[1]

| 방법 | 전체 파라미터 | FGVC | VTAB-Natural | VTAB-Specialized | VTAB-Structured | 승리 |
|------|-------------|------|-------------|-----------------|-----------------|----|
| **Full** | 24.02M | 88.54 | 75.88 | 83.36 | 47.64 | - |
| Linear | 1.02M | 79.32 | 68.93 | 77.16 | 26.84 | 0 |
| Adapter | 1.23M | 85.66 | 70.39 | 77.11 | 33.43 | 2 |
| **VPT-Deep** | **1.18M** | **89.11** | **78.48** | **82.43** | **54.98** | **8** |

### 2. 저데이터 레짐에서의 우수성

데이터 규모별 성능 비교에서 VPT-Deep이 일관되게 우월합니다:[1]

- 데이터 10% 사용: VPT-Deep > Adapter > Linear > Full
- 데이터 50% 사용: VPT-Deep > Full > Adapter > Linear
- 데이터 100% 사용: VPT-Deep > Full

이는 프롬프트 기반 학습이 **과적합을 자연스럽게 방지**함을 시사합니다.

### 3. 모델 규모별 확장성

서로 다른 ViT 규모(Base, Large, Huge)에 대한 성능:[1]

| 백본 | VTAB-Natural | VTAB-Specialized | VTAB-Structured |
|-----|-------------|-----------------|-----------------| 
| ViT-B16 | 78.48 | 82.43 | 54.98 |
| ViT-L16 | 81.64 | 84.02 | 58.54 |
| ViT-H14 | 83.15 | 85.48 | 60.27 |

VPT-Deep은 모델 규모가 커질수록 Full Fine-tuning 대비 성능 격차가 증가합니다.

### 4. 계층적 Transformer (Swin)에서의 성능

Swin Transformer 적용 결과:[1]

| 방법 | VTAB-Natural | VTAB-Specialized | VTAB-Structured |
|-----|-------------|-----------------|-----------------|
| **Full** | 79.10 | 86.21 | 59.65 |
| Linear | 73.52 | 80.77 | 33.52 |
| **VPT-Deep** | 76.78 | 84.53 | 53.35 |

Swin에서는 VPT의 이점이 감소하지만, 여전히 경쟁력 있는 성능을 유지합니다.

***

## 일반화 성능 향상 메커니즘

### 1. t-SNE 시각화를 통한 분석

논문은 VPT-Deep이 생성하는 CLS 임베딩의 특성을 분석합니다.[1]

세 가지 VTAB 작업(SVNH, EuroSAT, Clevrcount)에서:

- **VPT-Deep**: 선형 분리 가능한 클래스 경계 형성 (과적합 방지)
- **Full Fine-tuning**: 과도하게 복잡한 결정 경계 (과적합 위험)
- **VPT-Shallow**: 중간 수준의 분리

### 2. 프롬프트 깊이(Depth)의 영향

프롬프트를 삽입하는 레이어 선택의 중요성:[1]

$$\text{Best Performance} \approx \text{Prompts at Early Layers}$$

- **1-9층**: 최고 성능
- **6-12층**: 중간 성능
- **9-12층**: 성능 저하

조기 레이어의 프롬프트가 더 중요한 이유:
- 초기 레이어가 일반적인 특징 추출 담당
- 후기 레이어는 작업 특화 정보 포함 (수정 필요 없음)

### 3. 프롬프트 길이(Length)의 최적화

작업별 최적 프롬프트 길이는 상이합니다:[1]

| 작업 그룹 | 최적 길이 | 성능 |
|---------|---------|------|
| VTAB-Natural | 50-100 | 78.5% |
| VTAB-Specialized | 50-200 | 82.4% |
| VTAB-Structured | 10-50 | 55.0% |

**흥미로운 발견**: 단 1개 프롬프트만으로도 VPT-Deep은 전체 미세조정을 능가할 때가 있습니다.

### 4. 입력 시퀀스 길이 확장의 영향

프롬프트 추가로 입력 시퀀스 길이가 확장되는데, 이것이 성능 향상의 주요 원인인지 검증:[1]

| 설정 | VTAB-Natural | VTAB-Specialized | VTAB-Structured |
|-----|------------|-----------------|-----------------|
| Prompt-Fixed (학습 X) | 70.5 | 78.4 | 34.1 |
| CLS-Learned (1 프롬프트) | 72.6 | 78.3 | 35.1 |
| **Prompt-Learned** (기본) | **76.8** | **79.7** | **47.0** |
| Full Fine-tuning | 75.9 | 83.4 | 47.6 |

프롬프트 학습 여부가 결정적이므로, **입력 길이 확장 자체보다 학습 가능한 파라미터가 핵심**입니다.

***

## 일반화 성능의 한계 및 제한 사항

### 1. 자기지도 학습 백본에서의 약화

MAE 및 MoCo v3 사전학습 모델에서의 성능:[1]

| 방법 | MAE (Natural) | MoCo v3 (Natural) |
|-----|--------------|------------------|
| Full | 59.29 | 71.95 |
| **VPT-Deep** | **36.02** | 70.27 |
| Partial-1 | 58.44 | 72.31 |

**문제점**: 자기지도 학습 모델에 대해 VPT의 이점이 상당히 감소합니다.

**가설**: 자기지도 학습과 지도 학습의 특성 차이로 인한 것 (세부 원인은 여전히 미개방 문제)

### 2. 의미론적 분할 작업에서의 한계

ADE20K 의미론적 분할 실험에서:[1]

| 방법 | mIoU (Single-Scale) | 파라미터 |
|-----|-------------------|---------|
| Full Fine-tuning | 50.07 | 318.31M |
| **VPT-Deep** | 44.06 | 13.43M |
| Head-only | 37.46 | 13.18M |

의미론적 분할에서는 VPT가 Full Fine-tuning을 능가하지 못합니다.

### 3. ConvNet에서의 제한적 효과

ResNet-50 (ImageNet-1k 사전학습)에 대한 VPT 적용:[1]

| 방법 | VTAB-Natural | VTAB-Specialized | VTAB-Structured |
|-----|-------------|-----------------|-----------------|
| Full | 59.72 | 76.66 | 54.08 |
| **VPT** | 66.25 | 77.32 | 37.52 |
| Partial-1 | 64.34 | 78.64 | 45.78 |

ConvNet에서는 효과가 미미하며, **VPT는 Transformer 아키텍처에 특화된 방법**입니다.

### 4. 계산 복잡도 상세 분석

실제 연산 비용 분석 결과:[1]

| 프롬프트 수 | 훈련 레이턴시 | 추론 레이턴시 | GPU 메모리 |
|-----------|----------|----------|-----------|
| p=1 | 213.6 ms/img | 69.4 ms/img | 10.3 GB |
| p=50 | 350.6 ms/img | 25.8 ms/img | 25.8 GB |
| p=200 | 360.1 ms/img | 25.8 ms/img | 140.8 GB |
| Full Fine-tuning | 358.7 ms/img | 69.7 ms/img | 11.7 GB |

프롬프트 수가 증가하면 메모리 사용량이 급증합니다 (p=200일 때 약 12배).

***

## 최신 연구에 미친 영향 및 미래 연구 방향

### 1. VPT 이후의 발전 연구들 (2023-2025)

#### A. LSPT (Long-term Spatial Prompt Tuning, 2024)

VPT의 한계를 개선한 최신 연구:[2][3]

- 공간적 프롬프트 정보를 활용하여 VPT 성능 개선
- FGVC 및 VTAB-1K 벤치마크에서 새로운 기준 설정
- VPT보다 더 강력한 일반화 성능

#### B. iVPT (크로스층 동적 연결, 2024)

작업 특화 정보 공유 메커니즘 개선:[4]

- 주목할 이미지 토큰 자동 식별
- 프롬프트 토큰과의 가산 방식 향상
- 24개 이미지 분류 및 의미론적 분할 작업에서 성능 향상

#### C. VAPT (Visual Adaptive Prompt Tuning, 2025)

프롬프트 표현 방식의 개선:[5]

- 기존 VPT의 제한된 함수 표현력 문제 해결
- 프롬프트를 입력의 적응 함수로 재정의
- 이론적 분석 제시

#### D. LoR-VP (저랭크 시각 프롬프팅, 2025)

매개변수 효율성 극대화:[6]

- 패치 간의 공유 정보에서 귀납적 편향 활용
- 기존 VPT의 패드 기반 접근법 한계 극복
- 시각 프롬프트와 원본 이미지 간 상호작용 강화

### 2. 연관 기술 분야의 발전

#### A. Federated Learning과의 통합 (2024)

프롬프트 튜닝을 연합 학습에 적용:[7]

```
블록 좌표 하강법(BCD) 활용
- 공유 프롬프트: 공통 특징 정보
- 그룹 프롬프트: 특화된 지식
→ 클라이언트의 로컬 파인튜닝 불필요
```

#### B. 지속 학습 (Continual Learning)과의 연계 (2024-2025)

비전 프롬프트 튜닝의 지속 학습 적용:[8]

- 이전 작업과의 일관성 유지 조건 도출
- Null 공간에서의 프롬프트 튜닝 개발
- 새로운 작업 학습 시 기존 지식 보존

#### C. 비전-언어 모델에의 확장 (2024)

SEP (Self-Enhanced Prompt Tuning for VLM):[9]

- CLIP 같은 비전-언어 모델에 최적화된 프롬프트 튜닝
- Context Optimization (CoOp) 기반 방법론

### 3. 매개변수 효율적 미세조정(PEFT) 분야의 광범위한 발전

VPT는 PEFT 분야의 확대 생태계에 기여:[10]

| 기법 | 주요 특징 | VPT와의 관계 |
|------|---------|-----------|
| **LoRA** | 저랭크 분해 | 서로 다른 패러다임 (입력 vs 가중치) |
| **Adapter** | 모듈 삽입 | VPT가 더 효율적 (Vision 영역) |
| **Prefix-Tuning** | 프리픽스 추가 | VPT와 유사한 입력 공간 접근법 |
| **BitFit** | 편향항만 학습 | VPT보다 성능 낮음 |

### 4. 비전 Transformer 아키텍처의 미래

2025년 기준 최신 트렌드:[11][12]

#### A. 효율성 중심의 진화

```
Sparse Attention → Flash Attention → 선형 복잡도
O(n²) 복잡도 해결
```

#### B. 계층적 구조 도입

Swin Transformer 같은 계층적 아키텍처의 확대로 VPT 적응이 점진적으로 개선되고 있습니다.

#### C. 멀티모달 기반 모델의 부상

CLIP, LLaVA 등 비전-언어 모델에서 프롬프트 튜닝의 중요성 증가

***

## 앞으로의 연구 고려 사항

### 1. 이론적 해석의 필요성

현재까지 명확하지 않은 부분:

- **왜 비전에서는 프롬프트 튜닝이 Full Fine-tuning을 능가하는가?** (NLP와의 본질적 차이)
- 자기지도 학습 백본에서 성능 저하의 정확한 원인
- 프롬프트 초기화 전략의 최적성

### 2. 작업 특수성 대응

현재 VPT의 한계:

| 작업 영역 | 성능 | 개선 방향 |
|---------|------|---------|
| 분류 작업 | ⭐⭐⭐⭐⭐ | - |
| 의미론적 분할 | ⭐⭐⭐ | 디코더 설계 개선 |
| 객체 감지 | 미평가 | 바운딩 박스 회귀 적응 |
| 자기지도 학습 | ⭐⭐⭐ | 메커니즘 규명 필요 |

### 3. 계산 효율성 극대화

VPT-prefix 같은 최적화 방법 개발:

- 프롬프트를 Key-Value 어레이에 직접 추가
- 추론 시간 50% 이상 감소
- 메모리 사용량 개선

### 4. 멀티태스크 학습 강화

프롬프트 공유 및 엙상블 기법:

- 5개 프롬프트 앙상블: Single 대비 평균 2.5% 성능 향상
- 작업 간 프롬프트 전이 학습 연구

### 5. 도메인 특화 모델 개발

특정 분야에 최적화된 VPT 변형:

- **의료 영상**: 프롬프트 깊이 최적화
- **위성 영상**: 멀티 스펙트럼 데이터 적응
- **자율주행**: 연속 학습 통합

***

## 결론

Visual Prompt Tuning은 비전 분야의 전이 학습 패러다임을 근본적으로 재정의한 방법입니다. 매개변수 1% 미만으로 완전 미세조정을 초과하는 성능을 달성함으로써, 대규모 모델의 실용적 배포 가능성을 제시했습니다.[1]

2023-2025년 이후의 연구들은 VPT의 한계를 보완하면서도 **자기지도 학습, 의미론적 분할, 지속 학습** 등의 어려운 문제들에 접근하고 있습니다. 향후 연구는 VPT의 성공을 기반으로 더욱 **통합된 PEFT 프레임워크**, **이론적 근거 마련**, 그리고 **멀티모달 기반 모델**과의 결합을 통해 진화할 것으로 예상됩니다.[3][7][6][4][5]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5a4e488f-c85a-46b1-9561-318394b48fb2/2203.12119v2.pdf)
[2](https://arxiv.org/pdf/2203.17274.pdf)
[3](http://arxiv.org/pdf/2402.17406.pdf)
[4](https://arxiv.org/html/2404.05207v1)
[5](https://arxiv.org/abs/2501.18936)
[6](https://arxiv.org/html/2502.00896v1)
[7](https://arxiv.org/pdf/2310.18285v2.pdf)
[8](https://www.themoonlight.io/ko/review/visual-prompt-tuning-in-null-space-for-continual-learning)
[9](http://arxiv.org/pdf/2405.15549v2.pdf)
[10](https://www.ibm.com/kr-ko/think/topics/parameter-efficient-fine-tuning)
[11](https://funes-days.com/dev/transformer-revolution-2025-ai-evolution/)
[12](https://upself.tistory.com/140)
[13](https://arxiv.org/pdf/2402.02382.pdf)
[14](https://seom-j.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-22%E2%80%99ECCV-Visual-Prompt-Tuning)
[15](https://jiankim3293.tistory.com/28)
[16](https://discuss.pytorch.kr/t/deep-research-test-time-compute-test-time-scaling/6153)
[17](https://johyeongseob.tistory.com/80)
[18](https://www.calluscompany.com/blog/kr/what-is-fine-tuning)
[19](https://seo.goover.ai/report/202408/go-public-report-ko-95a7926f-53ff-4e90-8182-b4d0ec0e442a-0-0.html)
