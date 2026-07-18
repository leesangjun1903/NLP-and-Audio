# Visual Prompt Tuning (VPT)

---

## 1. 핵심 주장과 주요 기여 (요약)

**Visual Prompt Tuning(VPT)**는 Jia 등(Cornell University, Meta AI)이 ECCV 2022에서 발표한 논문으로, 대규모 비전 트랜스포머(ViT)를 다운스트림 태스크에 적응시키는 새로운 패러다임을 제시합니다.

이 논문은 비전 분야의 대규모 트랜스포머 모델에 대한 풀 파인튜닝(full fine-tuning)의 효율적이고 효과적인 대안으로 Visual Prompt Tuning(VPT)을 제안하며, 언어모델의 효율적 튜닝 기법에서 영감을 받아 모델 백본을 고정한 채 입력 공간에 전체 파라미터의 1% 미만에 해당하는 소량의 학습 가능한 파라미터만 도입한다. 

다양한 다운스트림 인식 태스크에 대한 광범위한 실험을 통해 VPT가 다른 파라미터 효율적 튜닝 방식들에 비해 유의미한 성능 향상을 달성함을 보였으며, 가장 중요하게는 VPT가 모델 규모와 학습 데이터 규모에 걸쳐 많은 경우 풀 파인튜닝보다도 더 뛰어난 성능을 보이면서 태스크별 저장 비용까지 절감한다.

핵심 기여를 정리하면:
- 언어 모델의 prompt tuning 개념을 비전 트랜스포머에 최초로 체계적으로 적용
- VPT-Shallow, VPT-Deep 두 가지 변형 제시
- 24개 이상의 다운스트림 인식 태스크(FGVC, VTAB-1k)에서 광범위한 검증
- 파라미터 효율성과 성능을 동시에 달성

---

## 2. 문제 정의, 방법론, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

기존 사전학습 모델을 적응시키는 일반적인 방식은 백본 파라미터 전체를 업데이트하는 풀 파인튜닝이었다. 이는 다음과 같은 문제를 야기합니다:
- ViT-Huge와 같은 대규모 모델의 경우 태스크마다 전체 모델 사본을 저장해야 하므로 저장 비용이 막대함
- 데이터가 적은 다운스트림 태스크에서는 과적합 위험이 큼
- 자원이 제한된 환경에서 배포가 어려움

이 논문은 "**언어 모델에서 성공한 prompt tuning을 이미지 인코더에도 적용할 수 있는가?**"라는 질문을 던집니다. 이 논문에서 저자들은 동일한 방법을 이미지 인코더에도 성공적으로 적용할 수 있는지를 묻는다.

### 2.2 제안하는 방법 (수식 포함)

사전학습된 ViT 모델이 주어지면, VPT는 원본 파라미터 전체를 고정한 채 D차원의 프롬프트 임베딩 집합 $P=\{p_k\}$와 태스크별 예측 헤드(들)라는 두 종류의 새로운 학습 가능 파라미터를 도입한다.

**(1) VPT-Shallow**

VPT-Shallow는 모델의 첫 부분에만 프롬프트를 도입한다. $N_p$개의 학습 가능한 d차원 프롬프트 집합을 $\mathbf{P} = \{ \boldsymbol{p}^i \in \mathbb{R}^d \mid i = 1, 2, \dots, N_p \}$라 할 때, $z_i \in \mathbb{R}^{N_x \times d}$와 $v_i \in \mathbb{R}^{N_p \times d}$는 각각 레이어 $L_i$가 처리하는 인스턴스와 프롬프트 특징을 나타내며, $N_x$는 인스턴스 시퀀스 길이이다.

수식으로는:

$$[\boldsymbol{z}_1, \boldsymbol{v}_1] = L_1([\boldsymbol{x}_0, \mathbf{P}])$$

$$[z_i, v_i] = L_i([z_{i-1}, v_{i-1}]), \quad i=2,\dots,N$$

최종적으로 다층 퍼셉트론 헤드(Head)가 N번째 레이어의 클래스 토큰 $x_N$을 예측 클래스 확률 분포 $y$로 매핑한다.

$$y = \text{Head}(x_N)$$

**(2) VPT-Deep**

VPT-Deep은 각 레이어마다 추가적인 프롬프트 토큰을 도입하여 더 큰 파인튜닝 용량을 가지며, 더 우수한 전이 성능을 보이는 것으로 입증되었다.

$$[z_i, v_i] = L_i([z_{i-1}, \mathbf{P}_{i-1}]), \quad i=1,\dots,N$$

여기서 각 레이어마다 서로 다른 프롬프트 집합 $\mathbf{P}_{i-1}$이 삽입되며, 이전 레이어의 프롬프트 출력은 버려지고 새로운 프롬프트로 대체됩니다.

**두 변형의 트레이드오프**: VPT-Shallow는 레이어 간에 프롬프트를 공유함으로써 연속적인 의미를 포착하여 우수한 적응성과 일반화를 보이지만 파라미터 용량이 제한적이다. 반면 VPT-Deep은 더 큰 학습 가능 파라미터 용량을 제공하지만 레이어 간 의미적 연결을 무시하여 최적화를 혼란시키고 하이퍼파라미터에 민감하게 만들 수 있으며 사전학습 효과성을 감소시킬 가능성이 있다.

### 2.3 모델 구조

전형적인 ViT 모델은 여러 개의 동일한 트랜스포머 레이어가 쌓인 구조로, 입력 이미지가 먼저 고정 크기의 패치 시퀀스로 분할되는 처리 흐름을 가진다. VPT는 이 구조 위에 다음을 추가합니다:
- 학습 가능한 프롬프트 토큰(D차원 벡터, K개)을 패치 임베딩과 [CLS] 토큰 사이에 prepend
- VPT-Shallow 모드에서는 모든 프롬프트가 첫 번째 트랜스포머의 입력 시퀀스 레벨에만 삽입되며, K개의 프롬프트 임베딩이 원본 [CLS] 토큰 및 시각 토큰과 결합되어 확장된 입력 $[x^0,P,E^0]\in \mathbb{R}^{(1+K+M)\times D}$을 형성한다.
- 백본은 완전히 동결(frozen)되고, 프롬프트와 분류 헤드만 학습됨

### 2.4 성능 향상

24개 인식 태스크에 걸친 실험에서 visual prompt tuning은 종종 풀 파인튜닝을 능가하며, 저데이터 환경과 멀티태스크 배포에서 성능을 향상시킨다.

VPT의 효과성은 ViT Base, Large, Huge 등 다양한 모델 규모에 걸쳐 유지되며, 다운스트림 라벨 데이터의 양이 달라져도 견고하다.

태스크별 적응을 위해 VPT는 전체 모델의 사본이 아닌 학습된 프롬프트와 최종 헤드만 저장하면 되므로 저장 공간을 줄이고 많은 다운스트림 태스크의 신속한 배포를 가능하게 한다.

논문 원문의 실험 섹션에서는 ViT-B, ViT-L, ViT-H 등 다양한 모델 규모와 3개의 VTAB 태스크 그룹에 걸쳐 VPT와 풀 파인튜닝을 비교했으며, VPT-deep과 풀 파인튜닝 간의 정확도 차이를 강조 표시했다. 또한 FGVC 태스크에서 학습 데이터 크기가 정확도에 미치는 영향을 살펴보기 위해 학습 데이터를 10%에서 80% 사이로 변화시키며 모든 방법을 비교했으며, 동일한 사전학습 ViT-B를 다운스트림 학습에 사용했다.

이후 후속 연구들의 비교를 통해서도 VPT의 성능이 재확인됩니다. 예: VPT는 적은 수의 학습 가능 파라미터로 특정 태스크에 맞춰 사전학습된 비전 트랜스포머를 조정하기 위해 제안되었으며, 특히 학습 데이터가 제한적일 때 일반 파인튜닝보다 일반화 능력에서 우위를 보인다.

### 2.5 한계

후속 연구들이 지적한 VPT의 한계는 다음과 같습니다:

1. **VPT-Deep의 레이어 간 정보 단절**: VPT-Deep은 이전 레이어의 모든 정보를 버리고 매 레이어마다 완전히 새로운 프롬프트를 도입하여 파라미터 수가 크게 증가한다. 이는 이전 레이어에서 나온 유용한 인스턴스 관련 정보를 손실시킬 수 있는 반면, 모든 것을 그대로 보존하면 레이어별 적응의 여지가 없어진다는 이중의 문제를 낳는다.

2. **VPT-Shallow의 낮은 성능**: VPT-Shallow는 파라미터 수가 훨씬 적지만, 무작위로 초기화된 학습 가능 파라미터가 입력 레이어에만 적용되기 때문에 성능이 훨씬 낮다.

3. **하이퍼파라미터 민감성**: 또한 판별적 의미 특징을 명시적으로 추출하는 메커니즘이 부족한데, 이는 시각 인식 태스크에 중요한 요소이다.

4. **파라미터 효율성의 한계**: E2VPT 등 후속 연구에서 VTAB-1k에서 풀 파인튜닝 대비 5.85%, VPT 대비 1.99% 향상된 정확도를 보였으며, 평균적으로 백본 파라미터의 0.32%만을 사용한 반면 VPT는 평균 0.68%를 필요로 했다.는 점에서 VPT가 상대적으로 더 많은 파라미터를 요구함을 알 수 있습니다.

---

## 3. 일반화 성능 향상 가능성

VPT의 일반화 성능은 이 논문의 가장 중요한 강점 중 하나로 평가됩니다.

### 3.1 저데이터(low-data) 환경에서의 일반화

VPT는 특히 학습 데이터가 제한적인 상황에서 일반 파인튜닝 대비 우수한 일반화 능력을 보인다. 이는 전체 백본을 고정하고 소수의 프롬프트만 학습하기 때문에 과적합 위험이 낮아지는 구조적 이점에서 비롯됩니다.

### 3.2 모델 규모와 데이터 규모에 걸친 견고성

VPT의 효과성은 ViT Base, Large, Huge 등 다양한 모델 규모에 걸쳐 유지되며, 다운스트림 라벨 데이터 양이 달라져도 견고하다. 이는 VPT가 특정 조건에서만 작동하는 것이 아니라 광범위한 조건에서 일반화 가능한 방법론임을 시사합니다.

### 3.3 분포 이동(distribution shift)에 대한 강건성

후속 연구인 PETL 통합 분석에서는 다운스트림 정확도 외에도 PETL 방법들의 분포 이동에 대한 강건성을 평가했다.는 점에서, VPT류 방법이 단순 정확도뿐 아니라 도메인 변화에 대한 일반화 잠재력도 연구 대상이 되고 있음을 알 수 있습니다.

### 3.4 구조적 근거: 얕은 레이어 vs 깊은 레이어

더 깊은 레이어에 프롬프트를 삽입하는 VPT-Deep이 추가적인 성능 향상을 가져오는데, 이는 모델 깊은 곳에서 인코딩된 더 큰 태스크 특이성 및 표현력과 일치한다. 이는 프롬프트가 각 레이어의 표현 공간에 맞춰 적응적으로 태스크 정보를 주입함으로써, 사전학습된 일반적 표현을 보존하면서도 다운스트림 태스크에 특화된 일반화를 가능케 함을 시사합니다.

다만 Natural 태스크 그룹에서는 VPT-deep이 VPT-shallow에 비해 갖는 이점이 오히려 줄어들어, VPT-shallow가 더 나은 정확도를 보이는 경우도 있었다.는 점에서, 일반화 성능 향상이 태스크 특성에 따라 다르게 나타날 수 있음도 확인됩니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항 (2020년 이후 연구 비교 분석)

### 4.1 연구 영향

VPT는 비전 분야 PEFT(Parameter-Efficient Fine-Tuning) 연구의 초석이 되었습니다. 이 연구는 프롬프트 튜닝 기법이 비전 트랜스포머 아키텍처 내에서 적용 가능함을 최초로 체계적으로 탐구하여, 시각 태스크에서의 파라미터 효율적 전이학습을 위한 새로운 경로를 열었다.

이후 다음과 같은 다양한 방향으로 후속 연구가 확산되었습니다:

| 연구 | 학회/연도 | 핵심 개선점 |
|---|---|---|
| E2VPT | ICCV 2023 | 키-값 프롬프트 및 프루닝 도입 |
| LPT | ICLR 2023 | 롱테일 클래스 대응 |
| SA2VP | AAAI 2024 | 공간 정렬 2D 맵 방식 |
| LSPT | CVPR 2024 | 장기 공간 프롬프트 생성 |
| BlackVIP | CVPR 2023 | 제로스오더 블랙박스 방식 |
| DA-VPT | CVPR 2025 | 의미 기반 가이드 프롬프트 |
| Visual Fourier Prompt Tuning (VFPT) | 2024 | 다양한 다운스트림 태스크에서 일관되게 더 나은 성능 |

또한 Adapter 계열(AdaptFormer), 재매개변수화 계열(LoRA, SSF) 등과의 비교 연구도 활발합니다. 통합 벤치마크 연구에서는 대부분의 PETL 방법이 다양한 도메인에서 유사한 정확도를 달성함을 보이면서도, 서로 다른 PETL 접근법이 서로 다른 귀납적 편향(inductive bias)을 가지므로 학습 방식이 다를 수 있다는 가설을 제시했다.는 점에서, VPT가 촉발한 "프롬프트 기반 vs 어댑터 기반 vs 재매개변수화 기반" PEFT 방법론 간 비교·분석이 지속적인 연구 주제로 자리잡았음을 알 수 있습니다.

### 4.2 향후 연구 시 고려할 점

1. **레이어 간 정보 전달 메커니즘 설계**: VPT-Shallow와 VPT-Deep의 트레이드오프(공유 vs 완전 대체)를 넘어서는 균형 잡힌 방식이 필요합니다. 이전 레이어에서 가장 유용한 정보를 선택적으로 유지하면서도 레이어별 적응의 여지를 남기는 균형 잡힌 프레임워크 개발이 동기가 되었다.

2. **인스턴스 인식(instance-aware) 프롬프트**: 정적인 프롬프트가 아닌 입력별로 적응하는 동적 프롬프트 생성 방식에 대한 연구가 증가하고 있습니다 (예: Instance-aware Prompt Tuning).

3. **파라미터 효율성과 성능의 균형**: 파라미터 수를 더욱 줄이면서도 성능을 유지/향상시키는 방향(E2VPT의 pruning, SSF의 scale&shift 등)에 대한 지속적 탐구가 필요합니다.

4. **자기지도학습(SSL) 백본과의 결합**: MAE, MoCo-v3와 같은 자기지도 사전학습 접근법에서 VPT, GateVPT 등과 비교하는 연구가 이루어지고 있으며, GateVPT는 자기지도 사전학습 모델에 특화된 적응 방법이다.는 점에서, 지도학습 사전학습 모델에 최적화된 VPT를 SSL 모델에도 효과적으로 적용하는 방법이 중요한 고려사항입니다.

5. **다른 도메인/모달리티로의 확장**: VPT의 강력한 다운스트림 적응성과 최소한의 저장 파라미터로 인해 오디오 연구자들도 VPT 패러다임을 오디오 태스크에 적용하기 시작했다.는 사례처럼, 비전을 넘어 오디오, 시계열(DAS 신호), 의료 영상 등 다양한 도메인으로의 일반화 가능성에 대한 검증이 계속되어야 합니다.

6. **분포 이동에 대한 강건성 검증**: 단순 벤치마크 정확도뿐 아니라 실제 배포 환경의 분포 변화에 대한 강건성 평가가 표준 평가 프로토콜로 자리잡을 필요가 있습니다.

---

## 참고문헌 (출처)

1. Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N. (2022). *Visual Prompt Tuning*. ECCV 2022. (arXiv:2203.12119)
2. GitHub - KMnP/vpt (공식 코드 저장소)
3. ECVA 공식 논문 PDF (ecva.net)
4. Xin, Z. et al. *E2VPT: An Effective and Efficient Approach for Visual Prompt Tuning*. ICCV 2023 (arXiv:2307.13770)
5. *Revisiting the Power of Prompt for Visual Tuning* (arXiv:2402.02382)
6. *Visual Fourier Prompt Tuning* (arXiv:2411.01327)
7. *DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers*, CVPR 2025
8. *CVPT: Cross Visual Prompt Tuning*, ICCV 2025
9. *Visual Instance-aware Prompt Tuning* (arXiv:2507.07796)
10. *Lessons Learned from a Unifying Empirical Study of Parameter-Efficient Transfer Learning (PETL) in Visual Recognition* (arXiv:2409.16434)
11. *Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey* (arXiv:2402.02242)
12. *Semantic Hierarchical Prompt Tuning for Parameter-Efficient Fine-Tuning* (arXiv:2412.16956)
13. *Pear: Pruning and Sharing Adapters in Visual Parameter-Efficient Fine-Tuning* (arXiv:2409.19733)
14. *One-for-All: Generalized LoRA for Parameter-Efficient Fine-Tuning (GLoRA)* (arXiv:2306.07967)
15. *SegPrompt: Using Segmentation Map as a Better Prompt* (arXiv:2303.08303)
16. Awesome-Visual-Prompt-Tuning (GitHub, yunbeizhang)
17. EmergentMind - Visual Prompt Tuning 요약 페이지
18. Oreate AI Blog - Visual Prompt Tuning 분석
19. *Detect All-Type Deepfake Audio: Wavelet Prompt Tuning* (arXiv:2504.06753)
20. *A Foundation Model for DAS Signal Recognition and Visual Prompt Tuning* (arXiv:2508.04316)
21. *Progressive Visual Prompt Learning with Contrastive Feature Re-formation* (arXiv:2304.08386)

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
