
# VL-ADAPTER: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks

## 1. 핵심 주장 및 주요 기여

VL-ADAPTER 논문의 핵심 주장은 **비전-언어(Vision-and-Language, V&L) 모델의 완전한 매개변수 미세조정(full fine-tuning)을 하지 않고도 경쟁력 있는 성능을 달성할 수 있다**는 것입니다. 구체적으로, 이미지-텍스트 작업에서는 전체 매개변수의 4.18%, 비디오-텍스트 작업에서는 3.39%만 업데이트하면서 전체 미세조정과 유사한 성능을 낼 수 있음을 입증했습니다.[1]

**세 가지 주요 기여**는 다음과 같습니다:[1]

1. **다양한 매개변수 효율적 미세조정(PETL) 기법의 벤치마킹** - Adapter, Hyperformer, Compacter 같은 어댑터 기반 방법들을 VQA, GQA, NLVR2, MSCOCO 캡셔닝, 비디오 작업 등 다양한 하위 작업에서 평가

2. **공유 가중치(weight-sharing) 기법의 도입** - 다중 작업 학습 환경에서 작업 간 정보 공유를 통해 전체 미세조정 성능을 달성하면서도 매개변수를 80-90% 감소

3. **포괄적인 분석** - CLIP 동결 여부, 다양한 아키텍처 구성요소의 기여도, 작업별 프롬프트와 사전학습의 영향 분석

## 2. 문제 정의 및 제안 방법

### 2.1 해결하는 문제

최신 비전-언어 모델들의 급격한 성장으로 인해 다음과 같은 문제가 발생했습니다:[1]

- GPT-3 기반 V&L 모델은 1750억 개의 매개변수를 가지며 메모리상 700GB가 필요
- 이러한 대규모 모델의 전체 미세조정은 계산 비용과 저장소 측면에서 비실용적
- 기존 적응 방법들이 텍스트 분류와 이미지-텍스트 정렬에 제한되어, 더 복잡한 V&L 작업(VQA, 비디오 질의응답, 캡셔닝)에 적용되지 않음

### 2.2 제안 모델 구조

논문은 **CLIP-BART 또는 CLIP-T5** 아키텍처를 기반으로 합니다. 이 구조는:[1]

- **시각 인코더**: CLIP (ResNet101 또는 ViT-B32)
- **언어 모델**: BART 또는 T5 (인코더-디코더 구조)
- **시각 투영층**: 시각 표현을 언어 모델에 맞게 변환

이들을 결합하여 V&L 작업을 **텍스트 생성 문제로 통일**합니다.[1]

### 2.3 손실 함수 및 기본 수식

기본 손실 함수는 다음과 같습니다:[1]

$$\mathcal{L}(x_I, x_S, y; \theta_L, \theta_V, \theta_{V \rightarrow L}) = -\sum_{i=1}^{M} y_i \log f_\theta(x_{V \rightarrow L}, x_S)_i$$

여기서:
- $x_I$ = 이미지 또는 비디오 입력
- $x_S$ = 문장/질문 입력
- $y = (y_1, y_2, ..., y_M)$ = M개 토큰의 정답 레이블
- $x_{V \rightarrow L} = f_{\theta_{V \rightarrow L}}(f_{\theta_V}(x_I))$ = 투영된 시각 표현

다중 작업 설정에서는:[1]

$$\mathcal{L}_D(\theta_L, \theta_V, \theta_{V \rightarrow L}) = \sum_{(x_I, x_S, y) \in D} \mathcal{L}(x_I, x_S, y; \theta_L, \theta_V, \theta_{V \rightarrow L})$$

### 2.4 어댑터 메커니즘

**표준 어댑터**는 다음과 같이 작동합니다:[1]

$$h = f(Ux(f(Dx) + x))$$

여기서:
- $x \in \mathbb{R}^{d_i}$ = 어댑터 입력
- $D \in \mathbb{R}^{d \times d_i}$ = 다운샘플링 가중치 행렬
- $U \in \mathbb{R}^{d_i \times d}$ = 업샘플링 가중치 행렬
- $d$ = 숨겨진 차원, $d_i$ = 입력 차원
- 매개변수 복잡도: $O(d \cdot d_i)$ (전체 모델의 약 2-3%)

**Hyperformer** 메커니즘:[1]

$$D, U = f_H(f_T(t_j, l_i))$$

여기서:
- $t_j, l_i$ = 작업 $j$와 층 $i$의 임베딩
- $T \in \mathbb{R}^{d_e \times 2d_p}$ = 작업 투영 네트워크
- $H \in \mathbb{R}^{d_p \times d(d_i)}$ = 하이퍼네트워크
- 조건: $d_p \approx N_T \cdot N_L$ (더 효율적)

**Compacter - 매개변수화된 초복소 승법(PHM) 층**:[1]

$$D = \sum_{i=1}^{k} A_i \otimes B_i$$

여기서:
- $A_i \in \mathbb{R}^{k \times k}$, $B_i \in \mathbb{R}^{\frac{d_i}{k} \times \frac{d}{k}}$
- $\otimes$ = 크로네커 곱
- 매개변수 복잡도 감소: $O(\frac{d \cdot d_i}{k})$

**저차 매개변수화 초복소 승법(LPHM)** 개선:[1]

$$D = \sum_{i=1}^{k} A_i(u_i v_i^T)$$

여기서 $u_i \in \mathbb{R}^{\frac{d_i}{k} \times r}$, $v_i \in \mathbb{R}^{r \times \frac{d}{k}}$, $r=1$ 최적

### 2.5 공유 가중치 어댑터(Shared-Weight Adapters)

논문의 핵심 창신점 중 하나는 다중 작업 환경에서 어댑터 가중치 공유입니다:[1]

- **Multiple Adapters**: 각 작업별 독립적 어댑터 $D_i, U_i$
- **Half-shared Adapters**: 다운샘플링 층은 공유 ($D$), 업샘플링은 작업별 ($U_i$)
- **Single Adapter**: 모든 작업이 동일 어댑터 ($D_i = D_j, U_i = U_j$)

이를 통해 저자원 작업(NLVR2)에서 정보 공유 이점이 나타났습니다.

## 3. 성능 향상 및 실험 결과

### 3.1 이미지-텍스트 작업 성능

표 1의 핵심 결과:[1]

| 방법 | 업데이트 매개변수 | VQA | GQA | NLVR2 | COCO | 평균 |
|------|-----------------|------|------|-------|-------|------|
| 완전 미세조정 | 100% | 67.6 | 56.7 | 73.0 | 112.9 | 77.6 |
| 단일 어댑터 | 4.18% | 65.9 | 54.5 | 74.2 | 114.9 | **77.4** |
| Hyperformer | 5.79% | 65.1 | 53.4 | 72.3 | 114.6 | 76.4 |
| 단일 Compacter | 2.70% | 64.2 | 53.3 | 71.7 | 114.1 | 75.8 |
| 단일 LoRA | 5.93% | 65.2 | 53.6 | 71.9 | 115.3 | 76.5 |
| 단일 프롬프트 | 2.00% | 44.0 | 36.3 | 51.8 | 103.9 | 59.0 |

**핵심 발견**:[1]
- 단일 어댑터가 **4.18%의 매개변수만으로 완전 미세조정(77.6)과 거의 동등한 77.4 성능** 달성
- Hyperformer는 더 효율적(5.79%)이지만 성능 감소 (76.4)
- Compacter는 BART 모델과 호환성 문제로 성능 저하

### 3.2 비디오-텍스트 작업 성능[1]

| 방법 | 업데이트 매개변수 | TVQA | How2QA | TVC | YC2C | 평균 |
|------|-----------------|------|-------|------|------|------|
| 완전 미세조정 | 100% | 76.3 | 73.9 | 45.7 | 154.0 | 87.4 |
| 단일 어댑터 | 3.39% | 76.6 | 73.9 | 46.3 | 152.9 | **87.4** |
| 단일 LoRA | 5.17% | 75.5 | 72.9 | 44.6 | 140.9 | 83.4 |

**비디오 작업에서 단일 어댑터는 완전 미세조정과 정확히 동등한 성능을 유지하면서 96.61%의 매개변수 감소를 달성**했습니다.

### 3.3 CLIP 동결 여부의 영향

표 5의 실험 결과:[1]

| CLIP 상태 | BART 상태 | VQA 정확도 |
|----------|----------|-----------|
| 미세조정 | 미세조정 | 65.6 |
| 미세조정 | 동결 | 39.4 |
| 동결 | 미세조정 | 64.7 |
| 동결 | 동결 | 39.1 |

결론: **언어 모델 미세조정이 결정적으로 중요하며, 시각 인코더는 동결해도 성능 손실이 미미**(65.6 vs 64.7)하면서 메모리 효율성이 우수합니다.

### 3.4 비용 효율성 분석

표 7 ablation 결과:[1]

| 구성요소 | 업데이트 매개변수 | VQA | GQA | NLVR2 | COCO | 평균 |
|---------|-----------------|------|------|-------|-------|------|
| $\theta_{V \rightarrow L}$만 | 1.14% | 32.2 | 25.6 | 52.1 | 78.5 | 47.1 |
| +계층 정규화 | 1.18% | 49.5 | 40.1 | 52.4 | 109.6 | 62.9 |
| +단일 어댑터 | 4.18% | 65.9 | 54.5 | 74.2 | 114.9 | 77.4 |

**계층 정규화 업데이트는 필수적**이며, 어댑터의 추가가 최종 성능의 핵심입니다.

### 3.5 VL 사전학습의 영향

표 9 결과:[1]

| 방법 | VQA | GQA | NLVR2 | COCO | 평균 |
|-----|------|------|-------|-------|------|
| 완전 미세조정 (사전학습 후) | 69.2 | 57.5 | 75.0 | 112.1 | 78.5 |
| 단일 어댑터 (사전학습 후) | 69.4 | 58.1 | 73.7 | 115.7 | **79.2** |

**사전학습 가중치를 사용할 경우, 단일 어댑터가 완전 미세조정을 초과하는 성능** 달성

## 4. 모델의 한계

논문에서 명시한 한계점들:[1]

1. **작업 특이성**: 4개 V&L 작업에 대한 실험이므로 다른 아키텍처나 작업 분포에는 일반화되지 않을 수 있음

2. **제한된 어댑터 변형**: Adapter, Hyperformer, Compacter만 평가했으며, 다른 PETL 기법은 미포함

3. **프롬프트 튜닝 성능**: 프롬프트 튜닝이 부진(59.0% 평균)한 이유가 충분히 분석되지 않음 - 사전학습과 하위 작업의 큰 괴리 추정

4. **Compacter의 불일치**: Kronecker 곱 가정이 BART 기반 V&L 작업에 너무 제한적일 수 있음

## 5. 일반화 성능 향상 가능성

### 5.1 현재 논문의 일반화 분석

**교차 데이터셋 전이 능력**:[1]
- 단일 어댑터는 VQA Karpathy 테스트 세트에서 77.4%, 테스트-표준에서 68.3% 달성
- 모델이 다양한 데이터 분포에 적응 가능함을 시사

**저자원 작업에서의 이점**:[1]
- NLVR2 (86.4K 샘플)에서 공유 가중치 기법이 특히 효과적
- 반공유 어댑터(8.36%)와 단일 어댑터(4.18%)의 성능 향상이 저자원 환경에서 정보 공유의 가치를 증명

**판별적 아키텍처로의 확장**:[1]
- CLIP-ViL (SOTA 판별 모델)에서 4.3-6.2% 매개변수로도 효과적 (표 10)
- 이는 어댑터 방법의 광범위한 적용 가능성을 보여줍니다

### 5.2 최신 연구에 기반한 일반화 개선 방향

**최근 연구 트렌드 (2023-2025)**에 따른 향상 가능성:[2][3][4][5]

1. **Sharpness-Aware Minimization (SAM) 적용**[3][4][2]
   - GLAD 프레임워크: LoRA와 SAM을 결합하여 few-shot 시나리오에서 일반화 개선
   - 기울기 정규화를 통해 안정적인 매개변수 영역 탐색으로 분포 변화에 강건

2. **Out-of-Distribution (OOD) 강건성**[6][5]
   - OGEN 방법: 클래스 조건부 특성 생성기로 OOD 샘플 합성
   - 자기 증류(self-distillation)를 통한 신뢰도 보정
   - V&L 모델의 OOD 정확도 개선 입증

3. **테스트 타임 적응**[3]
   - LoRA-TTT: 테스트 시점에 LoRA만 업데이트하여 분포 변화 대응
   - CLIP-ViT-B/16에서 OOD 벤치마크 평균 5.79% 개선

4. **다중 작업 프롬프트 어댑터**[7]
   - 2025 연구: 잔여 연결 기반 프롬프트 어댑터로 작업 간 간섭 최소화
   - 공유 학습과 작업별 독립성 동시 달성으로 일반화 성능 향상

5. **저차 적응의 이론적 근거**[8]
   - EFlat-LoRA: LoRA의 예리함(sharpness)과 일반화의 상관관계 입증
   - 일반화된 LoRA (GLoRA): 가중치뿐 아니라 활성화 차원도 적응하여 전이 학습 개선

### 5.3 VL-ADAPTER의 일반화 강점

원본 논문의 강점 측면에서:[1]

| 특성 | 성능 | 강점 |
|------|------|------|
| 단일 어댑터 (4.18%) | 77.4 | 강 - 전체 미세조정과 동등하면서도 단순함 |
| 공유 가중치 | NLVR2 +2.0% | 저자원 작업에서 명확한 개선 |
| 사전학습 통합 | 79.2 (완전 69.2) | 다양한 사전학습 가중치와의 호환성 |
| 판별 모델 적용 | 4.3-6.2% | 생성/판별 모델 모두 효과적 |

## 6. 앞으로의 연구에 미치는 영향

### 6.1 이론적 기여

1. **매개변수-성능 트레이드오프의 새로운 경계**
   - V&L 도메인에서 4% 미만의 매개변수로 완전 미세조정 성능 달성 가능함을 입증
   - 다중 작업 설정에서의 정보 공유 메커니즘 규명

2. **어댑터 아키텍처의 비교 프레임워크 제시**
   - Adapter > Hyperformer > Compacter 성능 순서는 도메인과 모델 특성에 따라 달라질 수 있음을 시사
   - 단순성과 효율성의 균형이 복잡한 설계보다 우수함을 실증

### 6.2 실무적 영향

1. **대규모 모델 배포의 경제성**
   - 메모리와 저장소 비용을 80-96% 감소시킬 수 있어 엣지 디바이스 배포 가능
   - 모바일/IoT 환경에서의 V&L 모델 활용 확대

2. **다중 작업 학습의 효율화**
   - 작업 간 공유 어댑터 사용으로 총 매개변수 증가를 선형적에서 아-선형적으로 제어

3. **산업 적용 사례**
   - 자동 캡셔닝 시스템, 멀티모달 검색, 머신 러닝 운영(MLOps)

### 6.3 앞으로의 연구 방향

**향후 고려할 사항들:**[9][4][6][2][3][1]

| 연구 방향 | 핵심 내용 | 관련 최신 연구 |
|----------|----------|--------------|
| OOD 일반화 | 도메인 변화에 견딜 수 있는 어댑터 설계 | GLAD (2025), OGEN (2024) |
| Few-shot 적응 | 극단적 저자원 시나리오 해결 | AdvCLIP-LoRA (2025), Low-Rank Few-Shot (2024) |
| 연속 학습 | 새 작업 추가 시 기존 성능 유지 | Multi-task prompt adapter (2025) |
| 강건성 | 적대적 공격에 대한 어댑터 강건성 | AdvCLIP-LoRA (2025) |
| 이론적 분석 | SAM과 LoRA의 연결고리 규명 | EFlat-LoRA (2025), GLAD (2025) |
| 크로스 모달 최적화 | 시각-언어 정렬의 효율적 적응 | LCMHA (2025), HeGraphAdapter (2024) |

## 7. 결론

**VL-ADAPTER**는 비전-언어 모델의 매개변수 효율적 전이 학습에 대한 포괄적인 벤치마크를 제시합니다. 단순한 **단일 어댑터 구조가 복잡한 하이퍼네트워크 기반 방법들보다 우수한 정확도-효율성 트레이드오프를 달성**한다는 발견은 PETL 분야에 중요한 시사점을 제공합니다.

특히 **다중 작업 학습 환경에서 가중치 공유 기법**의 도입은 저자원 작업에서의 성능 향상과 전체 매개변수 감소를 동시에 실현하여, 이 논문이 후속 연구의 기초가 되었습니다. 

2024-2025년 최신 연구들은 VL-ADAPTER의 한계를 보완하기 위해 **SAM 기반 정규화, OOD 강건성, 테스트 타임 적응** 등을 추가하고 있으며, 이들의 결합은 더욱 강력한 일반화 능력을 갖춘 매개변수 효율적 V&L 시스템으로 이어질 것으로 전망됩니다.[5][9][2][3]

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ae0e63a-b0d1-48b1-8550-ea363b037533/2112.06825v2.pdf)
[2](https://arxiv.org/abs/2507.13089)
[3](https://arxiv.org/abs/2502.02069)
[4](https://openaccess.thecvf.com/content/ICCV2025W/MMFM/html/Peng_GLAD_Generalizable_Tuning_for_Vision-Language_Models_ICCVW_2025_paper.html)
[5](https://arxiv.org/abs/2311.01723)
[6](https://openreview.net/forum?id=PKICZXVY9M)
[7](https://www.sciencedirect.com/science/article/abs/pii/S1568494625005654)
[8](https://arxiv.org/abs/2508.00522)
[9](https://openaccess.thecvf.com/content/ICCV2025W/MMFM/papers/Peng_GLAD_Generalizable_Tuning_for_Vision-Language_Models_ICCVW_2025_paper.pdf)
[10](https://ieeexplore.ieee.org/document/11192338/)
[11](https://www.semanticscholar.org/paper/f02241105c2a72943e24c37ae58a22c46db88720)
[12](https://arxiv.org/abs/2306.05642)
[13](https://arxiv.org/html/2504.00691v1)
[14](http://arxiv.org/pdf/2309.01479.pdf)
[15](http://arxiv.org/pdf/2311.15569.pdf)
[16](https://aclanthology.org/2023.findings-emnlp.483.pdf)
[17](https://aclanthology.org/2023.findings-emnlp.356.pdf)
[18](https://arxiv.org/abs/2210.00788)
[19](https://arxiv.org/pdf/2303.11866.pdf)
[20](http://arxiv.org/pdf/2308.12509.pdf)
[21](https://openaccess.thecvf.com/content/CVPR2022/papers/Sung_VL-Adapter_Parameter-Efficient_Transfer_Learning_for_Vision-and-Language_Tasks_CVPR_2022_paper.pdf)
[22](https://proceedings.neurips.cc/paper_files/paper/2023/file/80e354fdac2c7fbf439a51f4853edbac-Paper-Conference.pdf)
[23](https://aclanthology.org/2025.acl-long.1229.pdf)
[24](https://www.semanticscholar.org/paper/VL-ADAPTER:-Parameter-Efficient-Transfer-Learning-Sung-Cho/55a19318cc93714802c7ac59e07651789749b20c)
[25](https://arxiv.org/abs/2410.07854)
[26](https://www.sciencedirect.com/science/article/abs/pii/S1566253525006955)
[27](https://dl.acm.org/doi/10.1007/s00530-025-01878-3)
[28](https://arxiv.org/abs/2504.16054)
[29](https://ebooks.iospress.nl/doi/10.3233/FAIA251319)
[30](https://arxiv.org/abs/2505.05895)
[31](https://arxiv.org/abs/2508.20830)
[32](https://arxiv.org/abs/2507.03283)
[33](https://arxiv.org/abs/2506.06600)
[34](https://arxiv.org/abs/2505.15130)
[35](http://arxiv.org/pdf/2405.18541.pdf)
[36](https://arxiv.org/html/2504.07615v1)
[37](https://arxiv.org/pdf/2311.03079v1.pdf)
[38](http://arxiv.org/pdf/2406.20095.pdf)
[39](https://arxiv.org/pdf/2306.07967.pdf)
[40](http://arxiv.org/pdf/2311.12327.pdf)
[41](http://arxiv.org/pdf/2311.17091.pdf)
[42](https://arxiv.org/pdf/2503.24354.pdf)
[43](https://www.isca-archive.org/interspeech_2025/fang25_interspeech.pdf)
[44](https://chatpaper.com/paper/164801)
[45](https://liner.com/review/overcoming-the-pitfalls-of-visionlanguage-model-finetuning-for-ood-generalization)
[46](https://dl.acm.org/doi/10.1016/j.asoc.2025.113254)
[47](https://arxiv.org/pdf/2507.13089.pdf)
