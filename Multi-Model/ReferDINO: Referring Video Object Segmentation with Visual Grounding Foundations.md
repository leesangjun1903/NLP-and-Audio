
# ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations

> **논문 정보**: Tianming Liang, Kun-Yu Lin, Chaolei Tan, Jianguo Zhang, Wei-Shi Zheng, Jian-Fang Hu
> **발표**: ICCV 2025 (arXiv: 2501.14607)
> **프로젝트 페이지**: https://isee-laboratory.github.io/ReferDINO

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

RVOS(Referring Video Object Segmentation)는 텍스트 설명을 기반으로 비디오 전체에서 대상 객체를 분할하는 태스크로, 심층적인 비전-언어 이해, 픽셀 수준의 밀집 예측, 시공간 추론을 동시에 요구하는 매우 도전적인 문제입니다.

기존 RVOS 모델들은 제한된 비디오-언어 이해 능력으로 인해 복잡한 객체 설명을 처리하는 데 어려움을 겪고 있습니다. 이를 해결하기 위해 **ReferDINO**는 사전 학습된 시각적 그라운딩 파운데이션 모델로부터 강력한 비전-언어 이해 능력을 상속받고, 효과적인 시간적 이해와 객체 분할 능력을 추가로 갖춘 엔드-투-엔드 RVOS 모델입니다.

ReferDINO는 파운데이션 시각적 그라운딩 모델을 RVOS에 적용한 최초의 엔드-투-엔드 접근법입니다.

---

### 1.2 주요 기여 (3가지 핵심 혁신)

ReferDINO는 파운데이션 모델을 RVOS에 효과적으로 적응시키기 위해 세 가지 기술적 혁신을 제안합니다:
1. **Object-Consistent Temporal Enhancer**: 사전 학습된 객체-텍스트 표현을 활용하여 시간적 이해와 객체 일관성을 향상
2. **Grounding-Guided Deformable Mask Decoder**: 텍스트와 그라운딩 조건을 통합하여 정확한 객체 마스크 생성
3. **Confidence-Aware Query Pruning**: 성능 저하 없이 객체 디코딩 효율성을 크게 향상

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능

### 2.1 해결하고자 하는 문제

기존 RVOS 모델들은 복잡한 객체 속성 및 공간 위치와 관련된 설명을 처리하는 데 어려움을 겪고 있습니다. 이러한 문제는 현재 모델의 불충분한 비전-언어 능력에서 비롯되며, 이는 가용 RVOS 데이터의 제한적인 규모와 다양성 때문입니다.

GroundingDINO로 대표되는 파운데이션 모델들은 광범위한 이미지-텍스트 사전 훈련을 통해 복잡한 객체-텍스트 연관성 이해에서 강력한 잠재력을 보여주지만, 시간적 이해와 픽셀 단위 분할 능력이 부족하다는 것이 RVOS에 적응하는 핵심 과제입니다.

구체적으로, 기존 방법이 실패하는 두 가지 주요 시나리오:

① 기존 RVOS SOTA 방법은 복합 속성 설명(모양+색상)에 기반한 유사 객체 구별에 어려움을 겪는 반면, ReferDINO는 이를 극복합니다. ② GroundingDINO는 "꼬리를 흔드는" 같은 동작 참조를 이해하지 못하지만, ReferDINO는 크로스-모달 시공간 추론을 통해 올바른 예측을 수행합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 전체 파이프라인

ReferDINO의 파이프라인은 다음과 같습니다:
1. **Confidence-Aware Query Pruning**으로 낮은 신뢰도의 쿼리를 점진적으로 제거하여 컴팩트한 객체 특징 집합 획득
2. 모든 프레임의 객체 특징을 모아 **Object-Consistent Temporal Enhancer**로 크로스-모달 시간 추론 및 일관성 보장
3. **Grounding-Guided Deformable Mask Decoder**로 각 객체의 박스 예측을 위치 조건으로 사용하여 Deformable Cross-Attention과 Cross-Modal Attention으로 특징 점진적 정제
4. 최종적으로 각 객체에 해당하는 출력 특징과 프레임별 특징 맵의 내적(dot product)으로 마스크 시퀀스 $\{\mathbf{m}^t\}_{t=1}^{T}$ 생성

#### 2.2.2 Confidence Score 계산 (Query Pruning)

신뢰도 점수 $s_j$는 다음과 같이 정의됩니다:

$$s_j = \frac{1}{N_l - 1} \sum_{\substack{i=1, i \neq j}}^{N_l} \mathbf{A}^s_{ij} + \max_k \mathbf{A}^c_{kj}$$

여기서 $\mathbf{A}^s_{ij}$는 셀프 어텐션 가중치, $\mathbf{A}^c_{kj}$는 크로스 어텐션 가중치, $N_l$은 쿼리 수를 나타냅니다.

각 크로스-모달 디코더 레이어에서 모델은 다른 쿼리로부터 받는 어텐션 양과 입력 텍스트와의 관련성에 기반하여 각 쿼리의 신뢰도 점수를 계산합니다. 낮은 신뢰도의 쿼리는 제거되어, 다양한 객체 정보를 활용하면서도 계산 비용을 관리 가능하게 유지합니다.

#### 2.2.3 학습 손실 함수

훈련 목적 함수(Matching Cost)는 다음과 같이 구성됩니다:

$$\mathcal{L}(\mathbf{y}, \mathbf{p}_i) = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}(\mathbf{y}, \mathbf{p}_i) + \lambda_{\text{box}} \mathcal{L}_{\text{box}}(\mathbf{y}, \mathbf{p}_i) + \lambda_{\text{mask}} \mathcal{L}_{\text{mask}}(\mathbf{y}, \mathbf{p}_i)$$

- $\mathcal{L}_{\text{cls}}$: 분류 손실 (참조 텍스트와 객체의 이진 매칭 확률)
- $\mathcal{L}_{\text{box}}$: 바운딩 박스 예측 손실
- $\mathcal{L}_{\text{mask}}$: 마스크 예측 손실

#### 2.2.4 Binary 참조 확률 정의

점수는 객체 특징과 텍스트 토큰 간의 유사도로 정의되며, RVOS에 적응하기 위해 텍스트에 의해 참조되는 객체의 이진 확률을 모든 토큰에 걸친 객체의 최대 점수로 정의합니다:

$$P_{\text{ref}}(o) = \max_{k} \text{sim}(\mathbf{f}_o, \mathbf{t}_k)$$

여기서 $\mathbf{f}_o$는 객체 특징, $\mathbf{t}_k$는 $k$번째 텍스트 토큰입니다.

---

### 2.3 모델 구조

ReferDINO의 전체 아키텍처는 파란색 모듈(GroundingDINO에서 차용)과 빨간색 모듈(새롭게 도입)로 구성됩니다.

#### (1) Backbone & Feature Extraction (GroundingDINO 기반)

ReferDINO는 GroundingDINO로부터 정적 객체 인식 능력을 상속받아, 픽셀 단위 밀집 분할과 시공간 추론 능력을 추가로 갖춥니다.

훈련 설정과 관련하여, GroundingDINO의 Swin-T 및 Swin-B 두 체크포인트를 모두 실험에 사용하며, 백본을 동결(freeze)하고 LoRA 기법(rank=32)으로 크로스-모달 트랜스포머를 파인튜닝합니다.

#### (2) Confidence-Aware Query Pruning

신뢰도 인식 쿼리 정제 전략은 낮은 신뢰도의 쿼리를 점진적으로 제거하여 중요한 객체 특징의 컴팩트한 집합 $O_t$만을 도출합니다.

쿼리 정제에서 드롭 비율(drop rate)은 50%로 설정됩니다.

#### (3) Object-Consistent Temporal Enhancer

Object-Consistent Temporal Enhancer는 **Memory-Augmented Tracker**와 **Cross-Modal Temporal Decoder** 두 가지 하위 구성요소로 이루어집니다.

메모리 증강 트래커는 파운데이션 모델의 크로스-모달 텍스트 표현을 활용하여 프레임 간 객체 상호작용을 촉진하고, 시간적 이해와 객체 일관성을 향상시킵니다.

메모리 증강 트래커를 추가하면 시간적 일관성이 $\mathcal{J}$ & $\mathcal{F}$ 기준 0.4% 향상되며, 시간적 디코더는 시간적 동역학과 객체 움직임을 이해하는 데 필수적으로, $\mathcal{J}$ & $\mathcal{F}$ 기준 2.2%의 추가 향상을 제공합니다.

#### (4) Grounding-Guided Deformable Mask Decoder

Grounding-Guided Deformable Mask Decoder는 그라운딩 유도 적응형 샘플링을 위한 **Deformable Cross-Attention (DCA)**과 샘플링 노이즈의 영향을 완화하는 **Cross-Modal Attention (CMA)**을 활용합니다.

마스크 디코더는 원래의 박스 예측 브랜치와 마스크 예측 브랜치를 병렬로 추가하는 대신, 두 브랜치를 **그라운딩-변형-분할 파이프라인**으로 직렬 연결합니다. 사전 학습된 박스 예측을 위치 사전 정보로 활용하여 변형 어텐션 메커니즘을 통해 마스크 예측을 점진적으로 정제하며, 이 과정은 미분 가능하여 마스크 학습이 박스 예측 브랜치에 역전파되는 협업 태스크 학습을 가능하게 합니다.

어블레이션 연구에서, Cross-Attention(CMA)과 Deformable Cross-Attention(DCA)을 제거하면 각각 $\mathcal{J}$ & $\mathcal{F}$ 기준 0.4%와 2.7% 성능 저하가 발생합니다.

아키텍처를 수식으로 표현하면:

**DCA (Deformable Cross-Attention):**

$$\mathbf{q}'_i = \text{DCA}(\mathbf{q}_i, \mathbf{F}_{img}, \mathbf{b}_i)$$

여기서 $\mathbf{q}\_i$는 $i$번째 객체 쿼리, $\mathbf{F}_{img}$는 이미지 특징 맵, $\mathbf{b}_i$는 박스 예측값입니다.

**CMA (Cross-Modal Attention):**

$$\mathbf{q}''_i = \text{CMA}(\mathbf{q}'_i, \mathbf{F}_{text})$$

**최종 마스크 생성:**

$$\mathbf{m}^t_i = \mathbf{q}''_i \cdot \mathbf{F}^t_{hr}$$

여기서 $\mathbf{F}^t_{hr}$는 고해상도 프레임 특징 맵입니다.

---

### 2.4 성능 향상

#### 5개 공개 RVOS 벤치마크 실험

5개의 공개 RVOS 벤치마크에서 광범위한 실험을 통해 ReferDINO의 SOTA 대비 현저한 개선이 입증되었습니다. 예를 들어, Ref-DAVIS17 데이터셋에서 Swin-B 백본을 사용한 ReferDINO는 SOTA 대비 $\mathcal{J}$ & $\mathcal{F}$ 기준 4.0% 성능을 초과합니다.

또한 GroundingDINO와 SAM2를 결합한 경쟁 기반선과 비교할 때, Ref-YouTube-VOS 데이터셋에서 Swin-T 백본의 ReferDINO가 기준선을 12.0% 향상시킵니다.

ReferDINO는 실시간 추론 속도(51 FPS)로도 YouTube-VOS에서 경쟁력 있는 성능을 보여줍니다.

**평가 벤치마크 목록:**
ReferDINO는 5개의 공개 벤치마크에서 평가됩니다: **Ref-YouTube-VOS**, **Ref-DAVIS17**, **A2D-Sentences**, **JHMDB-Sentences**, **MeViS**.

---

### 2.5 한계

대규모 분할 데이터에 대한 학습 부재로 인해 마스크 품질이 만족스럽지 않을 수 있습니다.

동작 세부 사항이 포함된 쿼리(예: "뛰고 있는 흰 토끼")에 대해 ReferDINO는 토끼에 대한 거친 정적 그라운딩을 제공하지만 뛰는 동작의 세부 사항을 놓칠 수 있습니다.

Long-RVOS 결과에 따르면, 현재 RVOS 방법들은 장기 비디오 시나리오에서 심각한 성능 저하를 보입니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 파운데이션 모델 상속을 통한 일반화

기존 RVOS 모델의 불충분한 비전-언어 능력은 가용 RVOS 데이터의 제한된 규모와 다양성에서 비롯됩니다. ReferDINO는 파운데이션 시각적 그라운딩 모델로부터 강력한 비전-언어 이해와 공간 그라운딩 능력을 상속받아 이 문제를 효과적으로 해결합니다.

이전 RVOS 모델들은 특정 데이터셋에서는 주목할 만한 진전을 이루었으나, 불충분한 비전-언어 이해로 인해 미지의 객체나 시나리오에서 어려움을 겪는 반면, ReferDINO는 GroundingDINO의 오픈-월드 지식을 활용합니다.

### 3.2 LoRA 기반 파인튜닝의 효율성

백본을 동결하고 LoRA(rank=32) 기법으로 크로스-모달 트랜스포머만 파인튜닝함으로써 파운데이션 모델의 사전 학습 지식을 최대한 보존하면서 RVOS에 적응할 수 있습니다.

이 접근법은 파운데이션 모델의 **오픈-세계 일반화 능력**을 유지하면서 RVOS 특화 능력을 추가하는 효율적인 전략입니다.

### 3.3 ReferDINO-Plus를 통한 추가 확장 가능성

ReferDINO는 사전 학습된 파운데이션 이미지 모델로부터 객체 수준 비전-언어 지식을 적응시켜 유망한 성능을 보여주며, SAM2를 통합하여 마스크 품질과 객체 일관성을 향상시키는 방향으로 확장이 가능합니다. 또한 단일 객체와 다중 객체 시나리오 간의 성능 균형을 맞추기 위한 조건부 마스크 융합 전략을 도입합니다.

ReferDINO-Plus(ReferDINO와 SAM2의 앙상블 접근법)는 CVPR 2025의 PVUW 챌린지 RVOS 트랙에서 2위를 달성했습니다.

### 3.4 복잡한 텍스트 설명 처리 능력

ReferDINO는 참조 객체 식별, 안정적인 객체 추적, 고품질 객체 분할을 효과적으로 처리하며, 이는 다양한 실제 시나리오에서의 일반화 능력을 시사합니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 발표 | 핵심 접근법 | 주요 특징 | 한계 |
|------|------|-------------|-----------|------|
| **ReferFormer** (Wu et al.) | CVPR 2022 | DETR 기반, 텍스트를 쿼리로 사용 | 최초의 언어-쿼리 기반 RVOS 통합 프레임워크 | 비전-언어 이해 제한, 긴 훈련 필요 |
| **MTTR** (Botach et al.) | CVPR 2022 | 시퀀스 예측 방식의 멀티모달 트랜스포머 | DETR 패러다임 RVOS 최초 도입 | 시간적 정보 제한 |
| **OnlineRefer** (Wu et al.) | ICCV 2023 | 온라인 프레임 단위 쿼리 전파 | 오프라인 방식의 한계 극복, 실시간 처리 | 클립 내 시간적 연관 제한 |
| **MUTR** (Yan et al.) | AAAI 2024 | 멀티모달(텍스트+오디오) 통합 시간 트랜스포머 | 멀티모달 통합 RVOS, 시간적 일관성 향상 | 복잡한 파이프라인 |
| **Grounded-SAM2** | 2024 | GroundingDINO + SAM2 조합 | 강력한 마스크 품질 | 시간 동역학 이해 부재, 시공간 변화에 취약 |
| **ReferDINO** (Liang et al.) | **ICCV 2025** | GroundingDINO 기반 엔드-투-엔드 RVOS 적응 | 파운데이션 모델 상속 + 시공간 추론, 51FPS | 대규모 분할 데이터 훈련 미비, 동작 세부 파악 한계 |

MTTR은 DETR 패러다임을 RVOS에 최초로 도입했으며, ReferFormer는 텍스트로부터 쿼리를 생성하도록 제안했습니다. 이러한 파이프라인 위에 후속 연구들은 프레임 간 일관성과 시간적 이해를 개선하기 위한 모듈식 개선에 집중했습니다. 그러나 특정 데이터셋에서의 주목할 만한 진보에도 불구하고, 이 모델들은 불충분한 비전-언어 이해로 인해 한계를 보입니다.

크로스-어텐션 기반 방법의 성공과 함께 쿼리 기반 아키텍처가 RVOS에 도입되어 ReferFormer와 MUTR 같은 주목할 만한 사례를 낳았습니다. 그러나 분할 어노테이션이 있는 페어드 비디오-언어 데이터의 제한된 규모는 항상 RVOS의 주요 한계였습니다.

---

## 5. 향후 연구에 미치는 영향과 고려 사항

### 5.1 향후 연구에 미치는 영향

#### ① 파운데이션 모델 기반 RVOS 패러다임 확립

ReferDINO는 파운데이션 시각적 그라운딩 모델을 RVOS에 적응시키는 최초의 엔드-투-엔드 접근법으로서, 새로운 연구 방향(파운데이션 모델 → 비디오 이해 태스크)을 개척했습니다.

#### ② 장기 비디오 RVOS 연구 자극

ReferDINO의 접근법은 ReferMo와 같은 후속 연구에서 Hungarian 알고리즘 기반 클립 단위 객체 정렬 방식으로 참조되며, Long-RVOS와 같은 새로운 벤치마크 개발에 영향을 주고 있습니다.

#### ③ 대규모 분할 데이터와의 결합 연구 촉진

ReferDINO의 한계인 마스크 품질 문제를 극복하기 위해 SAM2의 마스크 품질 및 객체 일관성 장점을 통합하고, 단일-다중 객체 시나리오 간 성능 균형을 위한 조건부 마스크 융합 전략 같은 후속 연구가 촉진됩니다.

---

### 5.2 향후 연구 시 고려할 점

#### ① 동작 이해 강화

MeViS와 같은 데이터셋은 유사한 정적 외형을 가지지만 서로 다른 움직임 속성을 가진 여러 객체를 포함하며, 객체 설명이 주로 동작 및 시간적 표현에 초점을 맞춥니다. 따라서 광학 흐름(optical flow)이나 동작 표현을 통합한 접근법이 필요합니다.

#### ② 장기 비디오 처리

현재 RVOS 방법들이 장기 비디오 시나리오에서 심각하게 어려움을 겪는다는 결과가 있으므로, 메모리 효율적인 시간 모델링과 장기 의존성 포착이 중요한 연구 과제입니다.

#### ③ 더 강력한 파운데이션 모델과의 통합

ReferDINO가 GroundingDINO를 기반으로 성공적인 적응을 보여줬으므로, GPT-4V, InternVL, 최신 VLM(Vision-Language Model)과의 통합이 유망한 방향입니다.

#### ④ 다중 객체 참조 처리

SAM2는 다중 객체 마스크 프롬프트에 대해 단일 객체 마스크로 퇴화하는 경향이 있어 후속 프레임에서 상당한 대상 손실을 초래합니다. 이는 다중 객체 참조를 처리하는 보다 강건한 전략 개발이 필요함을 시사합니다.

#### ⑤ 데이터 효율성 및 약지도 학습

분할 어노테이션이 있는 페어드 비디오-언어 데이터의 제한된 규모는 이러한 접근법의 확장성을 저해합니다. 따라서 약지도(weakly-supervised) 혹은 자기지도(self-supervised) 방식의 연구가 중요합니다.

#### ⑥ 실시간 적용 최적화

ReferDINO의 실시간 비디오 응용 잠재력을 실제 배포 환경에서 실현하기 위해 모델 경량화(Knowledge Distillation, Quantization) 연구가 필요합니다.

---

## 📚 참고 자료 및 출처

1. **논문 원문 (arXiv)**: Tianming Liang et al., "ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations," arXiv:2501.14607, 2025. https://arxiv.org/abs/2501.14607

2. **ICCV 2025 공식 논문**: https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_ReferDINO_Referring_Video_Object_Segmentation_with_Visual_Grounding_Foundations_ICCV_2025_paper.pdf

3. **프로젝트 페이지**: https://isee-laboratory.github.io/ReferDINO/

4. **GitHub 공식 코드**: https://github.com/iSEE-Laboratory/ReferDINO

5. **HuggingFace 논문 페이지**: https://huggingface.co/papers/2501.14607

6. **ReferDINO-Plus (CVPR 2025 챌린지 보고서)**: arXiv:2503.23509, "ReferDINO-Plus: 2nd Solution for 4th PVUW MeViS Challenge at CVPR 2025"

7. **비교 관련 연구**: ReferFormer (Wu et al., CVPR 2022), MTTR (Botach et al., CVPR 2022), OnlineRefer (Wu et al., ICCV 2023), MUTR (Yan et al., AAAI 2024)

8. **후속 영향 관련 논문**: "Long-RVOS: A Comprehensive Benchmark for Long-term Referring Video Object Segmentation," arXiv:2505.12702

9. **themoonlight.io 리뷰**: https://www.themoonlight.io/en/review/referdino-referring-video-object-segmentation-with-visual-grounding-foundations
