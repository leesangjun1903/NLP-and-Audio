# FlexiViT: One Model for All Patch Sizes

---

## 1. 핵심 주장 및 주요 기여 요약

**핵심 주장:** Vision Transformer(ViT)의 패치 크기를 훈련 시 무작위로 변화시키면, **단일 가중치 세트**로 다양한 패치 크기에서 우수한 성능을 달성할 수 있으며, 이를 통해 배포 시점에 연산 예산에 맞춰 모델을 조절할 수 있다.

**주요 기여:**
1. **FlexiViT 학습 방법론:** 훈련 중 패치 크기를 랜덤 샘플링하고, 패치 임베딩 가중치와 위치 임베딩을 적응적으로 리사이즈하는 간단한 프레임워크 제안
2. **PI-resize(Pseudoinverse Resize):** 패치 임베딩 가중치를 리사이즈할 때 토큰 norm의 급격한 변화를 방지하는 수학적으로 원리화된 리사이즈 연산 제안
3. **지식 증류와의 시너지:** FlexiViT는 교사 모델의 가중치로 학생 모델을 초기화할 수 있어 증류 성능을 크게 향상
4. **다양한 다운스트림 태스크 검증:** 분류, 이미지-텍스트 검색, open-vocabulary 탐지, panoptic/semantic segmentation 등에서 고정 패치 ViT와 동등하거나 우월한 성능 달성
5. **자원 효율적 전이학습:** 큰 패치(낮은 연산)로 파인튜닝 후 작은 패치(높은 정확도)로 배포하는 전략 제시

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

ViT에서 패치 크기 $p$는 속도/정확도 트레이드오프를 제어하는 핵심 변수이다. 예를 들어:
- ViT-B/8: 85.6% ImageNet top-1 정확도, 156 GFLOPs
- ViT-B/32: 79.1% 정확도, 8.6 GFLOPs

두 모델은 거의 동일한 파라미터 수를 가지지만, **표준 ViT는 훈련된 패치 크기에서만 잘 작동**하며, 다른 패치 크기에서는 성능이 급격히 저하된다(Figure 3). 패치 크기를 변경하려면 모델을 처음부터 재훈련해야 하므로, 다양한 연산 예산에 대응하려면 여러 모델을 따로 훈련해야 하는 비효율이 발생한다.

### 2.2 제안하는 방법

#### (a) 기본 ViT 표기법

입력 이미지 $x \in \mathbb{R}^{h \times w \times c}$를 패치 크기 $p$로 분할하면 시퀀스 길이는:

$$s = \lfloor h/p \rfloor \cdot \lfloor w/p \rfloor$$

각 패치 $x_i \in \mathbb{R}^{p \times p \times c}$에 대해 패치 임베딩을 계산한다:

$$e_i^k = \langle x_i, \omega_k \rangle = \text{vec}(x_i)^T \text{vec}(\omega_k)$$

여기서 $\omega_k \in \mathbb{R}^{p \times p \times c}$는 패치 임베딩 가중치이다. 학습된 위치 임베딩 $\pi_i \in \mathbb{R}^d$를 더하여 토큰을 구성한다:

$$t_i = e_i + \pi_i$$

Self-attention의 연산량은 시퀀스 길이에 대해 $\mathcal{O}(s^2) = \mathcal{O}(h^4)$로 스케일링된다.

#### (b) FlexiViT 학습

핵심 변경은 단 두 가지:

1. **패치 크기 랜덤 샘플링:** 매 학습 스텝에서 패치 크기 $p$를 사전 정의된 집합 $\{8, 10, 12, 15, 16, 20, 24, 30, 40, 48\}$에서 균등분포 $\mathcal{P}$로 샘플링
2. **파라미터 리사이즈:** 학습 가능한 기본(underlying) 파라미터 형태를 $32 \times 32$ (패치 임베딩)과 $7 \times 7$ (위치 임베딩)으로 정의하고, forward pass 시 현재 패치 크기에 맞게 on-the-fly 리사이즈

이미지 해상도는 $240 \times 240$ px을 사용하여 다양한 패치 크기가 정확히 타일링되도록 한다.

#### (c) PI-resize (Pseudoinverse Resize)

bilinear interpolation으로 패치와 임베딩 가중치를 동시에 리사이즈하면 토큰의 크기가 크게 변한다:

$$\langle x, \omega \rangle \approx \frac{1}{4} \langle \text{resize}_p^{2p}(x), \text{resize}_p^{2p}(\omega) \rangle$$

이 문제를 해결하기 위해, bilinear resize를 선형 변환으로 표현한다:

```math
\text{resize}_p^{p_*}(o) = B_p^{p_*} \, \text{vec}(o)
```

여기서 $B_p^{p_\*} \in \mathbb{R}^{p_*^2 \times p^2}$이다.

리사이즈된 패치의 토큰이 원래 패치의 토큰과 일치하도록 새로운 가중치 $\hat{\omega}$를 구하는 최적화 문제를 정의한다:

$$\hat{\omega} \in \arg\min_{\hat{\omega}} \mathbb{E}_{x \sim \mathcal{X}} \left[ (\langle x, \omega \rangle - \langle Bx, \hat{\omega} \rangle)^2 \right]$$

**업스케일링** ($p_* \geq p$)의 경우, $\hat{\omega} = P\omega$로 정확한 해를 구할 수 있다:

$$P = B(B^T B)^{-1} = (B^T)^+$$

이를 통해:

$$\langle Bx, \hat{\omega} \rangle = x^T B^T B(B^T B)^{-1} \omega = x^T \omega = \langle x, \omega \rangle$$

**다운스케일링** ($p_* < p$)의 경우, 패치 분포 $\mathcal{X} = \mathcal{N}(0, I)$를 가정하면 역시 pseudoinverse가 최적 해가 된다. 최종적으로 **PI-resize**를 다음과 같이 정의한다:

```math
\text{PI-resize}_p^{p_*}(\omega) = \left( (B_p^{p_*})^T \right)^+ \text{vec}(\omega) = P_p^{p_*} \, \text{vec}(\omega)
```

여기서 $P_p^{p_\*} \in \mathbb{R}^{p_*^2 \times p^2}$이다.

일반적인 공분산 $\Sigma = \mathbb{E}_{x \sim \mathcal{X}}[xx^T]$에 대해서는:

$$\|\omega - B^T \hat{\omega}\|_\Sigma^2 = \|\sqrt{\Sigma}\omega - \sqrt{\Sigma}B^T \hat{\omega}\|^2$$

최적 해는:

$$(\sqrt{\Sigma}B^T)^+ \sqrt{\Sigma}\omega \in \arg\min_{\hat{\omega}} \|\omega - B^T \hat{\omega}\|_\Sigma^2$$

#### (d) 지식 증류를 통한 학습

강력한 ViT-B/8 교사 모델의 가중치로 FlexiViT 학생을 초기화하고, FunMatch 방식으로 KL-divergence를 최소화한다:

$$\mathbb{E}_{x \in \mathcal{D}} \mathbb{E}_{p \sim \mathcal{P}} \text{KL}\left( f_{\text{FlexiViT}}(x, p) \| f_{\text{ViT-B/8}}(x) \right)$$

여기서 $f_{\text{FlexiViT}}(x, p)$는 패치 크기 $p$에서의 FlexiViT 예측 분포, $f_{\text{ViT-B/8}}(x)$는 교사의 예측 분포이다.

### 2.3 모델 구조

FlexiViT는 **표준 ViT 아키텍처를 그대로 사용**한다. 아키텍처 변경 없이 forward pass에서 패치 임베딩 가중치와 위치 임베딩만 리사이즈하므로, 기존 ViT와 완전히 호환된다. Algorithm 1에서 보듯이 기존 ViT 코드에 대한 변경은 매우 최소한이다:

- 패치 임베딩 가중치: 기본 형태 $(32, 32, 3, d)$ → 현재 패치 크기로 리사이즈
- 위치 임베딩: 기본 형태 $(7, 7, d)$ → 현재 그리드 크기로 리사이즈
- 나머지 Transformer 인코더 가중치는 모든 패치 크기에 걸쳐 **공유**

### 2.4 성능 향상

**ImageNet-1k 결과 (Figure 2, Tables 1-4):**
- FlexiViT-L/8 (1200 epochs): **86.1%** top-1 정확도 (13 ms/img)
- FlexiViT-L/48: **77.8%** (0.1 ms/img 이하)
- 단일 FlexiViT-L이 DeiT III의 S/B/L 세 모델 모두와 EfficientNetV2를 매칭하거나 능가

**다운스트림 전이 (Figure 7):**
- 분류 (SUN397), panoptic segmentation (COCO PQ), open-vocabulary detection (LVIS AP), image-text retrieval (COCO R@1), semantic segmentation (Cityscapes mIoU) 등 모든 태스크에서 고정 패치 ViT와 동등하거나 우수

**자원 효율적 전이학습 (Figure 8):**
- 8×8 그리드로 저렴하게 파인튜닝: 81.8% → 24×24로 평가 시 **85.3%** (+3.5% 무추가비용)

### 2.5 한계

1. **외삽(Extrapolation) 불가:** 훈련 중 보지 못한 패치 크기(예: 6, 5 이하)로 외삽하면 성능이 서서히 저하된다 (Figure 30, Appendix Q)
2. **모델 너비 변경 대비 한계:** 매우 큰/작은 스케일에서는 모델 너비(width)를 변경하는 것이 패치 크기 변경보다 더 효과적인 지점이 존재
3. **훈련 시간:** 작은 패치 크기에서의 최고 성능을 위해서는 긴 훈련(1000 epochs)이 필요하며, 큰 패치 크기에서 주로 이득을 봄
4. **앙상블 한계:** 동일 FlexiViT를 여러 스케일로 앙상블하더라도, 동일 연산 예산의 단일 스케일보다 거의 항상 나쁨
5. **데이터 증강 효과 미탐구:** 패치 크기 랜덤화가 갖는 데이터 증강 효과에 대한 분석이 부족

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 다양한 패치 크기에서의 일반화

FlexiViT의 가장 핵심적인 일반화 특성은 **단일 모델이 넓은 범위의 패치 크기에서 동시에 높은 성능을 보인다**는 점이다. Figure 3에서 표준 ViT-B/16은 패치 크기 16에서 벗어나면 성능이 급락하지만, FlexiViT-B는 패치 크기 8~48 전 범위에서 일관된 성능을 유지한다.

### 3.2 내부 표현의 일관성 (CKA 분석)

Figure 6의 CKA 분석 결과:
- **초기 레이어~Block 6 MLP:** 다양한 그리드 크기에서 feature map 표현이 유사
- **Block 6 MLP 이후:** 표현이 잠시 분기하다가 최종 블록에서 다시 수렴
- **CLS 토큰 표현:** 모든 그리드 크기에서 일관되게 정렬

이는 FlexiViT가 **출력 수준에서 패치 크기에 관계없이 유사한 의미적 표현을 학습**함을 보여준다.

### 3.3 전이학습 후에도 유연성 보존

가장 주목할 만한 결과 중 하나는, **고정 패치 크기로 파인튜닝한 후에도 유연성이 상당 부분 유지**된다는 점이다 (Section 4.2, Figure 8). 이는:
- 8×8 그리드로 저렴하게 파인튜닝 → 24×24 그리드로 배포 시 +3.5% 정확도 향상
- LiT(Locked-image Tuning)에서 고정 패치로 전이한 FlexiViT도 다른 패치 크기에서 양호한 성능 유지 (Figure 19)

### 3.4 다양한 태스크로의 일반화

FlexiViT는 단순 분류를 넘어 5가지 이상의 다양한 비전 태스크(분류, LiT, OWL-ViT 탐지, UViM panoptic segmentation, Segmenter semantic segmentation)에서 고정 ViT와 동등하거나 우수한 성능을 보여, **태스크 일반화 능력**을 입증하였다.

### 3.5 Shape/Texture 편향

Figure 25에서 FlexiViT가 각 패치 크기에서 보이는 shape/texture 편향은 해당 패치 크기로 훈련된 표준 ViT와 유사하다. 이는 FlexiViT가 각 스케일에서 **해당 스케일에 적합한 특징을 적응적으로 활용**함을 시사한다.

### 3.6 스케일 간 토큰 표현의 대응

Figure 13 하단에서 패치 크기 16의 중심 토큰과 다른 패치 크기의 토큰 간 cosine similarity를 측정하면, **동일 공간 위치의 토큰들이 스케일에 걸쳐 높은 유사도**를 보인다. 이는 FlexiViT가 일종의 **스케일 등변(equivariant) 표현**을 학습했음을 시사한다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구 영향

1. **Compute-Adaptive 모델의 새로운 패러다임:** FlexiViT는 단일 모델로 다양한 연산 예산에 대응하는 간단하고 효과적인 방법을 제시하여, 엣지 디바이스부터 클라우드까지 동일 모델을 배포할 수 있는 가능성을 열었다.

2. **ViT 기반 파이프라인에 대한 Drop-in 개선:** 기존 ViT 학습 코드에 최소한의 변경만으로 적용 가능하므로, 향후 ViT 기반 모든 연구에서 기본적으로 채택될 수 있다.

3. **효율적 전이학습 전략:** 큰 패치로 저렴하게 파인튜닝 후 작은 패치로 배포하는 "fast transfer" 전략은 특히 대규모 모델의 실용적 배포에 중요한 시사점을 제공한다.

4. **멀티모달 확장의 가능성:** FlexiCLIP, FlexiLiT 실험이 보여주듯, 이미지-텍스트 멀티모달 학습에서도 유연한 패치 크기가 효과적으로 작동하여, 향후 대규모 멀티모달 모델에 적용 가능하다.

5. **NAS 및 효율적 추론 연구와의 융합:** FlexiViT의 패치 크기 유연성은 토큰 드롭핑(MAE, DynamicViT 등)이나 early exiting과 결합될 수 있어, 더욱 정교한 적응적 추론 시스템 구축이 가능하다.

### 4.2 향후 연구 시 고려할 점

1. **외삽 능력 부재:** 훈련 중 보지 못한 패치 크기로의 일반화가 어려우므로, 훈련 시 패치 크기 범위를 충분히 넓게 설정해야 한다. 이를 극복하기 위한 연속적 패치 크기 학습이나 meta-learning 접근이 필요하다.

2. **Self-supervised Learning과의 결합:** 본 논문은 주로 지도학습과 증류에 초점을 맞추었으며, MAE, DINO 등 자기지도학습에서의 FlexiViT 효과는 충분히 탐구되지 않았다.

3. **비정사각형 및 비정규 패치:** 현재는 정사각형 패치만 다루지만, 비정사각형 패치나 이미지를 완벽히 타일링하지 않는 경우에 대한 확장이 필요하다.

4. **더 큰 모델 스케일에서의 검증:** ViT-B/L 수준에서 검증되었으나, ViT-H, ViT-G 등 초대형 모델에서의 효과 검증이 필요하다.

5. **동적 패치 크기 선택:** 입력 이미지의 복잡도에 따라 패치 크기를 동적으로 선택하는 전략(예: 쉬운 이미지는 큰 패치, 어려운 이미지는 작은 패치)은 추론 효율성을 더욱 높일 수 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 아이디어 | FlexiViT와의 관계 |
|---|---|---|---|
| **ViT** (Dosovitskiy et al.) | 2021 | 이미지를 고정 패치로 분할, Transformer 적용 | FlexiViT의 기반 아키텍처. 고정 패치 크기에만 작동 |
| **DeiT / DeiT III** (Touvron et al.) | 2021/2022 | ViT의 데이터 효율적 학습 및 증류 | FlexiViT의 ImageNet-1k 실험에서 교사 모델 및 베이스라인으로 사용 |
| **MAE** (He et al., 2022) | 2022 | 마스킹 기반 자기지도학습, 랜덤 토큰 드롭 | FlexiViT와 결합 가능: 토큰 드롭은 시퀀스 길이 감소, FlexiViT는 패치 크기 변경 |
| **DynamicViT** (Rao et al., 2021) | 2021 | 중요도 기반 동적 토큰 가지치기 | FlexiViT는 모든 토큰을 유지하는 반면, DynamicViT는 토큰을 제거. 상호 보완적 |
| **EfficientNetV2** (Tan & Le, 2021) | 2021 | 점진적 해상도 증가로 학습 가속 | FlexiViT는 해상도가 아닌 패치 크기를 변경. Figure 2에서 FlexiViT-L이 능가 |
| **Swin Transformer V2** (Liu et al., 2021) | 2021 | 윈도우 기반 self-attention, 해상도 스케일링 | 해상도 변경 시 재학습 필요. FlexiViT는 단일 모델로 다양한 연산량 대응 |
| **SuperViT** (Lin et al., 2022) | 2022 | 다중 스케일 패치화 + 랜덤 토큰 드롭 | FlexiViT와 가장 유사하나, 더 복잡한 아키텍처 변경 필요. FlexiViT는 기존 ViT와 완전 호환 |
| **Once-for-All (OFA)** (Cai et al., 2019) | 2019 | 하나의 슈퍼넷에서 다양한 서브넷 추출 (NAS) | FlexiViT는 패치 크기만 변경하여 훨씬 단순. NAS 기반 방법은 다수의 아키텍처 차원 변경 필요 |
| **Matryoshka Representations** (Kusupati et al., 2022) | 2022 | 출력 벡터의 부분 벡터가 유의미하도록 학습 | FlexiViT의 보완적 접근: Matryoshka는 출력 차원, FlexiViT는 입력 시퀀스 길이의 유연성 |
| **A-ViT** (Yin et al., 2022) | 2022 | 적응적 토큰 개수로 효율적 추론 | 입력별 토큰 수 조절 vs. FlexiViT의 글로벌 패치 크기 변경. 결합 가능성 존재 |
| **LiT** (Zhai et al., 2022) | 2022 | 이미지 인코더 고정, 텍스트 타워만 학습 | FlexiViT가 LiT의 이미지 인코더로 사용될 때 다양한 패치 크기에서 작동 (FlexiLiT) |
| **OWL-ViT** (Minderer et al., 2022) | 2022 | ViT 기반 open-vocabulary 객체 탐지 | FlexiViT 백본 사용 시 패치 크기별 추론 시간 조절 가능. 태스크별 최적 패치 크기가 다름을 발견 |
| **FunMatch** (Beyer et al., 2022) | 2022 | 일관된 증류: 교사와 학생에게 동일 증강 적용 | FlexiViT의 증류 학습에 직접 사용된 방법론 |

### 주요 차별점 요약

FlexiViT는 위 연구들과 비교하여 다음과 같은 고유한 장점을 가진다:

1. **아키텍처 변경 없음:** 표준 ViT와 완전 호환, 기존 사전학습 모델 활용 가능
2. **구현 단순성:** 기존 학습 코드에 몇 줄의 변경만으로 적용 가능 (Algorithm 1)
3. **이론적 근거:** PI-resize를 통해 패치 임베딩 리사이즈의 수학적 최적성 보장
4. **범용성:** 분류, 탐지, 세그먼테이션, 멀티모달 등 다양한 태스크에서 일관된 효과

---

## 참고자료

1. Beyer, L., Izmailov, P., Kolesnikov, A., Caron, M., Kornblith, S., Zhai, X., Minderer, M., Tschannen, M., Alabdulmohsin, I., & Pavetic, F. (2023). *FlexiViT: One Model for All Patch Sizes*. arXiv:2212.08013v2 [cs.CV]. CVPR 2023.
2. Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
3. Touvron, H., Cord, M., & Jégou, H. (2022). *DeiT III: Revenge of the ViT*. arXiv:2204.07118.
4. He, K., et al. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022.
5. Beyer, L., et al. (2022). *Knowledge Distillation: A Good Teacher is Patient and Consistent*. CVPR 2022.
6. Zhai, X., et al. (2022). *LiT: Zero-Shot Transfer with Locked-Image Text Tuning*. CVPR 2022.
7. Minderer, M., et al. (2022). *Simple Open-Vocabulary Object Detection with Vision Transformers*. ECCV 2022.
8. Kusupati, A., et al. (2022). *Matryoshka Representations for Adaptive Deployment*. arXiv:2205.13147.
9. Rao, Y., et al. (2021). *DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification*. NeurIPS 2021.
10. Lin, M., et al. (2022). *Super Vision Transformer*. arXiv:2205.11397.
11. Tan, M. & Le, Q.V. (2021). *EfficientNetV2: Smaller Models and Faster Training*. ICML 2021.
12. Steiner, A.P., et al. (2022). *How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers*. TMLR 2022.
13. Cai, H., Gan, C., & Han, S. (2019). *Once for All: Train One Network and Specialize It for Efficient Deployment*. arXiv:1908.09791.
14. Yin, H., et al. (2022). *A-ViT: Adaptive Tokens for Efficient Vision Transformer*. CVPR 2022.
15. GitHub Repository: [github.com/google-research/big_vision](https://github.com/google-research/big_vision)
