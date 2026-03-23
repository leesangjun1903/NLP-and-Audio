# 2D-Guided 3D Gaussian Segmentation

---

# 1. 핵심 주장 및 주요 기여 (Summary)

3D Gaussian Splatting(3D-GS)은 NeRF 대비 복잡한 장면 표현과 학습 속도 면에서 강한 경쟁력을 보여주고 있으며, 이러한 장점은 3D 이해 및 편집에 광범위한 활용 가능성을 제시한다. 그러나 3D Gaussian의 세그멘테이션은 아직 초기 단계이며, 기존 방법들은 번거롭고 짧은 시간 안에 다중 객체를 동시에 분할하는 능력이 부족하다.

이에 본 논문은 2D 세그멘테이션을 지도(supervision)로 활용하여 3D Gaussian 세그멘테이션을 수행하는 방법을 제안한다. 입력 2D 세그멘테이션 맵을 통해 3D Gaussian에 추가된 시맨틱 정보의 학습을 가이드하며, 최근접 이웃(nearest neighbor) 클러스터링과 통계적 필터링으로 세그멘테이션 결과를 정제한다.

**주요 기여:**

효율적인 3D Gaussian 세그멘테이션 방법을 제안하여, 3D 장면의 시맨틱 정보를 2분 이내에 학습하고 주어진 시점에서 다중 객체를 1–2초 내에 분할할 수 있다.

LLFF, NeRF-360, Mip-NeRF 360 데이터셋에서 광범위한 실험을 통해 mIOU 86% 수준의 성능을 달성하며 유효성을 입증하였다.

---

# 2. 상세 분석: 문제 정의, 제안 방법, 모델 구조, 성능, 한계

## 2.1 해결하고자 하는 문제

기존 Gaussian Grouping은 약 15분의 확장된 학습 시간이 필요하며, SAGA는 구현이 복잡하고 다중 객체를 동시에 세그멘테이션하기 어렵다. 또한 3D Gaussian의 명시적(explicit) 표현 특성으로 인해 스토리지 오버헤드가 발생하여, NeRF 세그멘테이션처럼 2D 시맨틱 특징을 3D로 직접 전이하기 어렵다. 마지막으로 데이터셋의 부족과 어노테이션 결핍이 지도 학습 방식의 적용을 방해한다.

## 2.2 제안 방법 (Method)

### 파이프라인 개요

포즈가 주어진 학습 이미지로부터 먼저 인터랙티브 모델을 이용해 2D 사전 세그멘테이션 정보를 획득하고, 3D Gaussian에 시맨틱 정보를 추가한 뒤 이 정보를 2D 평면에 투영하여 2D 사전 지식과의 손실(loss)을 계산하며, 최종적으로 KNN 클러스터링과 통계적 필터링으로 세그멘테이션 결과를 정제한다.

### (1) Object Code 할당

각 3D Gaussian에 대해 object code $\mathbf{o} \in \mathbb{R}^K$를 할당하여 현재 3D Gaussian이 각 카테고리에 속할 확률 분포를 나타낸다. 여기서 $K$는 카테고리의 수이다.

### (2) α-blending 기반 시맨틱 렌더링

3D Gaussian Splatting의 기본 렌더링 수식(α-blending)에서, 각 픽셀의 색상 $C$는 다음과 같이 계산된다:

$$C = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

여기서 $c_i$는 $i$번째 3D Gaussian의 색상, $\alpha_i$는 불투명도(opacity)이다.

α-blending에 영감을 받아, 렌더링된 2D 세그멘테이션 맵의 픽셀 카테고리를 현재 레이를 따라 있는 다중 3D Gaussian 카테고리의 가중합으로 간주한다. 첫 번째 3D Gaussian이 가장 큰 기여를 하며, 이후 각 Gaussian의 기여는 렌더링 평면으로부터의 거리에 따라 감소하고 Gaussian의 크기에 비례한다. 렌더링 이미지의 각 픽셀의 카테고리는 식 (1)에서 색상 $c$를 각 3D Gaussian의 object code로 대체하여 표현할 수 있다.

이를 수식으로 나타내면:

$$\hat{O}(p) = \sum_{i \in \mathcal{N}} \mathbf{o}_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

여기서 $\hat{O}(p)$는 픽셀 $p$에서의 렌더링된 시맨틱 맵, $\mathbf{o}_i$는 $i$번째 3D Gaussian의 object code이다.

### (3) 손실 함수 (Loss Function)

2D 세그멘테이션 맵과 렌더링된 세그멘테이션 맵 간의 오차를 최소화하여 각 3D Gaussian의 분류를 가이드한다.

Ground truth 2D 세그멘테이션 맵 $M$과 렌더링된 시맨틱 맵 $\hat{O}$ 사이의 Cross-Entropy Loss를 사용한다:

$$\mathcal{L} = - \sum_{p} \sum_{k=1}^{K} M_k(p) \log\left(\text{softmax}(\hat{O}_k(p))\right)$$

여기서 $M_k(p)$는 픽셀 $p$에서의 ground truth 카테고리 $k$에 대한 원-핫 인코딩 값이다.

### (4) 후처리: KNN 클러스터링 + 통계적 필터링

시맨틱 모호성이 있는 3D Gaussian의 확률 분포를 추론하여 최대 확률값 $\max(\text{softmax}(\mathbf{o})) < \beta$인 3D Gaussian을 선택한다. 이 선택된 Gaussian들의 object code와 중심 좌표를 KNN에 입력하여, 쿼리 Gaussian에 대해 거리가 가장 가까운 $k$개의 3D Gaussian을 선택하고, 쿼리 Gaussian의 object code를 이들의 object code 평균으로 설정한다.

$$\mathbf{o}_q = \frac{1}{k} \sum_{i \in \mathcal{N}_k(q)} \mathbf{o}_i$$

여기서 $\mathcal{N}_k(q)$는 쿼리 Gaussian $q$의 $k$-최근접 이웃 집합이다.

최종적으로 KNN 클러스터링으로 3D Gaussian의 시맨틱 모호성을 해소하고, 통계적 필터링으로 잘못 세그멘테이션된 3D Gaussian을 제거한다.

## 2.3 모델 구조 요약

| 단계 | 설명 |
|------|------|
| **입력** | 사전 학습된 3D-GS 장면, 렌더링 이미지, 카메라 파라미터 |
| **2D 사전 정보** | 인터랙티브 2D 세그멘테이션 모델(SAM 등)로 2D 마스크 생성 |
| **Object Code 학습** | 각 3D Gaussian에 $\mathbf{o} \in \mathbb{R}^K$ 추가, α-blending으로 2D 투영 후 CE Loss 최적화 |
| **후처리** | KNN 클러스터링(시맨틱 모호성 해소) + 통계적 필터링(오류 Gaussian 제거) |
| **출력** | 각 3D Gaussian에 카테고리 라벨 할당 → 임의 시점 세그멘테이션 렌더링 가능 |

## 2.4 성능

2D 세그멘테이션의 지도를 받는 효율적인 3D Gaussian 세그멘테이션 방법으로, 3D 장면의 시맨틱 정보를 2분 이내에 학습하며, LLFF, NeRF-360, Mip-NeRF 360 데이터셋에서 mIOU 86% 수준을 달성하였다.

기존의 단일 객체 세그멘테이션 방법들과 비교했을 때, 다중 객체 세그멘테이션에서 mIOU 및 mAcc 면에서 비견할 만한(comparable) 성능을 달성한다.

## 2.5 한계

기존 Gaussian 기반 세그멘테이션 방법들은 특히 제로샷 세그멘테이션에서 컴팩트한 마스크를 생성하는 데 어려움이 있으며, 이는 각 Gaussian에 학습 가능한 파라미터를 단순 할당하는 방식이 교차 시점(cross-view)에서 비일관적인 2D 기계 생성 라벨에 대한 강건성이 부족하기 때문이다.

본 논문의 구체적 한계:
- **2D 세그멘테이션 모델 의존성**: 2D 사전 지식의 품질에 전적으로 의존하므로, 2D 모델의 오류가 3D에 전파됨
- **Cross-view 일관성**: 시점 간 2D 마스크의 불일치가 학습에 노이즈를 유발
- **대규모 장면 확장성**: 수십만~수백만 개의 Gaussian으로 구성된 대규모 장면에서의 확장성 제한
- **시맨틱 깊이 부족**: 단순 카테고리 분류 수준이므로 계층적(hierarchical) 또는 fine-grained 시맨틱 이해에 한계

---

# 3. 모델의 일반화 성능 향상 가능성

본 논문의 일반화(generalization) 성능과 관련하여 다음과 같은 측면을 분석할 수 있다:

### 3.1 2D Foundation Model 활용의 장점

사전 학습된 2D 세그멘테이션 모델을 활용하여 3D Gaussian 세그멘테이션을 가이드하는 접근법을 제안하고 있다. SAM과 같은 범용 2D 세그멘테이션 모델의 강력한 일반화 능력을 3D로 전이(transfer)하므로, **새로운 장면에 대한 적응이 별도의 대규모 3D 라벨 없이도 가능**하다는 장점이 있다.

### 3.2 일반화 성능 향상을 위한 방향

object-centric(객체 중심) 장면과 360° 장면 모두에서 실험을 통해 유효성을 검증하였다. 그러나 일반화 성능 향상을 위해 다음이 추가적으로 고려되어야 한다:

1. **Cross-view 일관성 강화**: 온라인 시스템에서 2D foundation model들은 카메라 시점이 변할 때 종종 불안정한 결과를 생성한다. 이를 해결하기 위해 비디오 기반 모델(예: SAM 2)의 시간적 일관성을 활용하는 전략이 유망하다.

2. **Feature Distillation 고도화**: 단순 파라미터 할당의 비강건성을 극복하기 위해, Dual Feature Fusion Network 같은 구조로 Gaussian의 세그멘테이션 필드를 구성하는 접근이 제안되고 있다.

3. **Geometry-aware 최적화**: 3D 세그멘테이션의 충실도는 근본적으로 기하 표현의 품질에 의존하며, 원래 3DGS 최적화는 물리적 표면 충실도가 부족한 비구조화된 볼류메트릭 "구름"이나 "떠다니는 요소(floaters)"를 생성하는 경우가 많아, 이러한 아티팩트는 세그멘테이션에 치명적이다.

4. **Open-Vocabulary 확장**: CLIP 등의 language-vision 모델과 결합하면, 사전 정의된 카테고리 없이도 자연어 기반 3D 세그멘테이션이 가능해져 일반화 성능이 크게 향상될 수 있다.

---

# 4. 향후 연구에 미치는 영향 및 고려사항

## 4.1 연구적 영향

본 논문은 **"2D→3D 시맨틱 리프팅(lifting)"** 패러다임의 초기 작업 중 하나로서, 다음과 같은 후속 연구에 기반을 제공하였다:

- **간결한 파이프라인**: 복잡한 특징 추출이나 추가 학습 없이, object code + α-blending + KNN이라는 간결한 구조로 3D 세그멘테이션을 달성함으로써, 후속 연구의 베이스라인 역할을 수행
- **다중 객체 동시 세그멘테이션**: 기존 단일 객체 중심 방법론에서 다중 객체 처리로의 전환을 촉진
- **실시간 응용 가능성**: 2분 이내 학습, 1–2초 세그멘테이션이라는 효율성은 로보틱스, AR/VR 등 실시간 응용의 가능성을 열어줌

## 4.2 향후 연구 시 고려할 점

| 고려 사항 | 설명 |
|-----------|------|
| **2D 모델 발전 통합** | SAM 2, GroundedSAM 등 최신 2D 모델의 일관성 향상을 3D에 적극 반영 |
| **3D 기하 품질** | 2DGS, SuGaR 등 geometry-aware Gaussian 표현으로 세그멘테이션 경계 정밀도 향상 |
| **벤치마크 확장** | 현재 NVOS, SPIn-NeRF 같은 벤치마크는 주로 단일 주요 객체만 포함하여 복잡한 환경에서의 인스턴스 분리 평가에 한계가 있으며, 2D 투영 기반 메트릭만으로는 진정한 3D 공간 일관성을 포착할 수 없다. |
| **계층적 세그멘테이션** | Part-level ↔ Object-level ↔ Scene-level의 다중 granularity 지원 |
| **대규모 장면** | 도시 규모, 자율주행 등 대규모 장면에 대한 확장성 연구 |

---

# 5. 2020년 이후 관련 최신 연구 비교 분석

아래 표는 3D Gaussian Splatting 기반 세그멘테이션 주요 연구들의 비교이다:

| 논문 | 연도/학회 | 핵심 방법 | 특징 | 한계 |
|------|----------|----------|------|------|
| **2D-Guided 3D-GS Seg** (본 논문) | 2023/ASIANComNet | Object code + α-blending + KNN | 2분 이내 학습, 1–2초 세그멘테이션 | Cross-view 불일치, 단순 카테고리 분류 |
| **Gaussian Grouping** (Ye et al.) | 2024/ECCV | 3D-GS를 확장하여 SAM 2D 리프팅을 통해 오픈월드 3D 장면에서 무엇이든 재구성·세그멘테이션·편집 | 편집 기능 포함 | 약 15분의 확장된 학습 시간 필요 |
| **SAGA** (Cen et al.) | 2025/AAAI | Scale-gated affinity feature를 각 3D Gaussian에 부착하여 다중 granularity 세그멘테이션 지원. Scale-aware contrastive training으로 SAM의 능력을 증류하고, soft scale gate 메커니즘으로 다중 granularity 모호성 처리 | 4ms 이내 3D 세그멘테이션 | 다중 객체 동시 세그멘테이션에 어려움 |
| **FlashSplat** (Shen et al.) | 2024/ECCV | 반복적 경사 하강법 대신, 전역적 최적(globally optimal) 솔버를 제안; 2D 마스크 렌더링이 각 Gaussian 라벨에 대한 선형 함수임을 활용 | 단일 단계 LP로 해결, 추가 학습 불필요 | 대규모 3D 장면에서 확장성 도전 |
| **GaussianCut** | 2024/NeurIPS | 장면을 그래프로 표현하고 graph-cut 알고리즘으로 에너지 함수를 최소화하여 Gaussian을 전경/배경으로 파티셔닝 | 클릭, 스크리블, 텍스트 등 직관적 입력 지원 | 특정 장면에서 feature 기반 방법 대비 성능 열세 |
| **CoSSegGaussians** (Dou et al.) | 2024 | Dual Feature Fusion Network를 세그멘테이션 필드로 사용하며, DINO 특징의 explicit unprojection과 포인트 클라우드 처리 네트워크의 공간 특징을 결합 | Zero-shot 세그멘테이션 강화 | 추가 네트워크 학습 필요 |
| **SAGOnline** | 2025 | 제로샷 프레임워크로 SAM 2 통합, Rasterization-aware Geometric Consensus 메커니즘으로 2D 예측을 실시간으로 명시적 3D 라벨에 매핑 | NVOS 92.7%, SPIn-NeRF 95.2% mIoU, 기존 대비 15–1500배 빠른 추론 속도 (27ms/frame) | 장면 별 학습 불필요하나 기하 품질 의존 |
| **ReferSplat** | 2025 | 자연어 설명 기반으로 3D Gaussian 장면에서 대상 객체를 세그멘테이션하는 새로운 태스크(R3DGS) 정의; 가려졌거나 직접 보이지 않는 객체의 식별 요구 | 언어-공간 관계 모델링 | 별도 데이터셋(Ref-LERF) 필요 |

---

# 참고자료 및 출처

1. **Kun Lan et al., "2D-Guided 3D Gaussian Segmentation"**, arXiv:2312.16047, 2023; IEEE 2024 ASIANComNet.
   - 출처: https://arxiv.org/abs/2312.16047 / https://arxiv.org/html/2312.16047v1
   - IEEE: https://ieeexplore.ieee.org/iel8/10811009/10810511/10811031.pdf
   - ResearchGate: https://www.researchgate.net/publication/387549788
2. **Ye et al., "Gaussian Grouping: Segment and Edit Anything in 3D Scenes"**, ECCV 2024.
   - 출처: https://github.com/lkeab/gaussian-grouping
3. **Cen et al., "Segment Any 3D Gaussians (SAGA)"**, AAAI 2025.
   - 출처: https://arxiv.org/abs/2312.00860 / https://github.com/Jumpat/SegAnyGAussians
4. **Shen et al., "FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally"**, ECCV 2024.
   - 출처: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03300.pdf
5. **"GaussianCut: Interactive Segmentation via Graph Cut for 3D Gaussian Splatting"**, NeurIPS 2024.
   - 출처: https://proceedings.neurips.cc/paper_files/paper/2024/file/a26f3dc32a913b77fa70c33ffa5dcb37-Paper-Conference.pdf
6. **Dou et al., "CoSSegGaussians"**: https://david-dou.github.io/CoSSegGaussians/
7. **"SAGOnline: Segment Any Gaussians Online"**, 2025.
   - 출처: https://arxiv.org/html/2508.08219
8. **"ReferSplat: Referring Segmentation in 3D Gaussian Splatting"**, 2025.
   - 출처: https://openreview.net/forum?id=reuShgiHdg
9. **HackerNoon 해설 시리즈**: https://hackernoon.com/effortless-2d-guided-3d-gaussian-segmentation-rendering-and-semantic-information
10. **Awesome 3DGS Applications Survey**: https://github.com/heshuting555/Awesome-3DGS-Applications

> **주의**: 본 답변의 수식은 논문의 공개된 방법론 설명을 기반으로 재구성한 것이며, 논문 원문의 수식 번호 체계와 정확히 일치하지 않을 수 있습니다. 정확한 수식 확인은 원문(arXiv:2312.16047)을 참조하시기 바랍니다.
