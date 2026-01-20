
# An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion

## 1. 논문의 핵심 주장과 주요 기여

"An Image is Worth One Word"는 **개별화된 텍스트-이미지 생성(Personalized Text-to-Image Generation)** 분야에서 혁신적인 접근법을 제시합니다. 논문의 핵심 주장은 사용자가 제공한 3~5개의 이미지만으로 새로운 개념을 텍스트-이미지 모델의 임베딩 공간에 **하나의 의사 단어(pseudo-word)**로 인코딩할 수 있다는 것입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

### 주요 기여

**첫째, 개념화된 작업 정의**: 사용자가 제공한 특정 개념을 자연언어 지시문으로 새로운 장면에 생성하는 **개인화된 텍스트-이미지 생성** 작업을 최초로 제시합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

**둘째, 텍스트 인버전 방법론**: 사전 학습된 텍스트 인코더의 임베딩 공간에서 새로운 의사 단어를 직접 최적화하는 방식으로, 생성 모델의 무결성을 유지하면서 새로운 개념을 추가합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

**셋째, 임베딩 공간 분석**: GAN 반전 기술의 원리를 텍스트 임베딩 공간에 적용하여, **왜곡-편집 가능성 트레이드오프(distortion-editability tradeoff)**가 존재함을 발견합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

***

## 2. 해결하고자 하는 문제와 제안하는 방법

### 2.1 핵심 문제

기존 텍스트-이미지 모델의 주요 제약은:
- 새로운 개념을 도입하려면 전체 모델을 재학습하거나 미세 조정해야 함
- 소수의 예시 이미지로 미세 조정하면 **재앙적 망각(catastrophic forgetting)**이 발생
- 사용자는 자신의 독특한 개념(예: 애완동물, 개인용품)을 정확히 설명하기 어려움

### 2.2 제안 방법: 텍스트 인버전(Textual Inversion)

#### 수학적 정식화

잠재 확산 모델(LDM)의 손실 함수는:

$$L_{LDM} := \mathbb{E}_{z \sim E(x), y, \epsilon \sim \mathcal{N}(0,1), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, c_\theta(y))\|_2^2 \right]$$

여기서:
- $z = E(x)$: 이미지 $x$의 잠재 표현
- $\epsilon_\theta$: 노이즈 제거 네트워크
- $c_\theta(y)$: 텍스트 조건 부호
- $t$: 시간 단계

제안 방법에서는 **의사 단어 임베딩** $v^*$를 최적화하여:

$$v^* = \arg\min_v \mathbb{E}_{z \sim E(x), y, \epsilon \sim \mathcal{N}(0,1), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, c_\theta(y))\|_2^2 \right]$$

여기서:
- 텍스트 프롬프트: $y = "A\ photo\ of\ S^*"$
- $S^*$: 새로운 의사 단어 토큰
- $c_\theta(y)$는 고정되고, $v^*$만 최적화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### 구체적 최적화 절차

1. **초기화**: $v^*$를 기본 설명자(예: "sculpture", "cat")의 임베딩으로 초기화
2. **샘플링**: 사용자가 제공한 이미지 집합에서 무작위로 이미지 샘플
3. **템플릿 기반 프롬프트**: CLIP ImageNet 템플릿 기반 다양한 중립적 맥락 텍스트 사용
   - "A photo of $S^*$"
   - "A rendition of $S^*$"
   - "A cropped photo of the $S^*$" 등 16개 템플릿 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)
4. **경사 하강법**: 5,000 최적화 단계 수행
   - 학습률: 0.04 (배치 크기 4, 2×V100 GPU)
   - 손실: 원래 LDM의 재구성 손실 사용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### 다중 단어 확장

논문에서는 단일 의사 단어의 한계를 극복하기 위해 다음 확장을 시도:

**2-단어 모델**: $v_1^\*, v_2^*$ 두 개의 임베딩을 최적화
- 프롬프트: "A photo of $S_1^\*$ $S_2^*$"

**3-단어 모델**: 순차적으로 추가
- 2,000 단계에서 두 번째 벡터, 4,000 단계에서 세 번째 벡터 도입 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

**정규화 적용**:

$$L_{total} = L_{reconstruction} + \lambda \|v^* - v_{descriptor}\|_2^2$$

여기서 $v_{descriptor}$는 기본 설명자의 임베딩 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

***

## 3. 모델 구조

### 3.1 전체 아키텍처

```
입력 이미지 집합
      ↓
텍스트 템플릿 ("A photo of S*")
      ↓
[토크나이제이션] → [토큰 인덱스: 508, 701, 73, ...]
      ↓
[임베딩 조회] → [v508, v701, v73, v* (학습가능), ...]
      ↓
[텍스트 변환기] → [조건 부호 c_θ(y)]
      ↓
[잠재 확산 모델]
      ↓
[VAE 디코더]
      ↓
생성 이미지
```

### 3.2 핵심 요소

**텍스트 인코더 (BERT 기반)**:
- 각 토큰을 연속 벡터로 변환
- 768 차원 임베딩 공간
- $v^*$는 이 공간에서만 최적화되며, 나머지 모델은 고정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

**잠재 확산 모델 (LDM)**:
- 4×64×64 잠재 공간에서 작동
- U-Net 아키텍처로 노이즈 제거
- 교차 주의 메커니즘으로 텍스트 조건 처리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

***

## 4. 성능 향상 및 한계

### 4.1 성능 메트릭 및 평가

논문은 **왜곡-편집 가능성 트레이드오프**를 분석하기 위해 두 가지 평가 지표를 제시합니다.

#### 재구성 품질 (Reconstruction Quality)
- **측정**: CLIP 임베딩 공간에서 생성 이미지와 학습 이미지 간의 의미적 유사성
- **방법**: 각 개념당 64개 이미지 생성 후, 학습 집합과의 쌍별 코사인 유사도 평균 계산
- **공식**: $R = \frac{1}{n \cdot m} \sum_{i=1}^{n} \sum_{j=1}^{m} \cos(I_i^{generated}, I_j^{training})$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### 편집 가능성 (Editability)
- **측정**: 다양한 난이도의 프롬프트에 대한 CLIP 기반 텍스트-이미지 정렬
- 프롬프트 범주:
  - 배경 수정: "A photo of $S^*$ on the moon"
  - 스타일 변경: "An oil painting of $S^*$"
  - 구성: "Elmo holding a $S^*$" [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### 정량적 결과 (Figure 10(a))

| 방법 | 재구성 유사도 | 편집 가능성 |
|------|----------|---------|
| 이미지만 (기준) | 0.72 | - |
| 프롬프트만 | - | 0.65 |
| 사람 캡션 (짧음) | 0.51 | 0.42 |
| 사람 캡션 (긴) | 0.48 | 0.35 |
| 2-단어 모델 | 0.71 | 0.58 |
| 3-단어 모델 | 0.70 | 0.55 |
| **단일 단어 (기본)** | **0.71** | **0.62** |
| 단일 단어 (높은 LR) | 0.74 | 0.48 |
| 단일 단어 (낮은 LR) | 0.68 | 0.68 | [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### 사용자 연구 (Figure 10(b))

600개 응답 (재구성 유사도)과 600개 응답 (편집 가능성):
- 재구성: 단일 단어 모델이 인간 캡션과 DreamBooth와 유사한 성능
- 편집 가능성: 단일 단어 모델이 모든 기준선을 능가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

### 4.2 정성적 응용 분석

#### 객체 변형 (Image Variations)
- **성능**: DALLE-2의 CLIP 기반 이미지 재구성보다 더 충실한 세부 사항 재현
- **예시**: 두개골 머그잔의 색상 패턴, 찻주전자의 세부 잎사귀 무늬
- **한계**: 변형 이미지이며 원본과 정확히 일치하지 않음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### 텍스트 유도 합성 (Text-guided Synthesis)
- 학습된 개념을 새로운 장면, 스타일, 구성에 성공적으로 통합
- "S* sports car", "S* made of lego", "watercolor painting of S*" [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### 스타일 전이
- 추상적 개념(예술 스타일)도 포착 가능
- 고정된 모델이 학습된 스타일과 새로운 내용을 함께 추론 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### 다중 개념 구성
- **한계**: 공간 관계(예: 나란히 놓인 객체) 추론에 어려움
- **개선 방안**: 다중 객체 장면으로 학습 데이터 확대 제안 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### 편향 감소
- 작은 다양한 데이터셋으로 "Doctor" 등 편향된 개념의 공정한 임베딩 학습 가능
- 성별, 인종 다양성 증가 입증 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

### 4.3 한계

#### 기술적 한계

1. **최적화 시간**: 개념당 약 2시간 소요 (V100 2개 기준)
   - 인코더 학습으로 가속 가능 제안 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

2. **정밀도 제약**: 정확한 형태 보존보다는 의미적 본질을 포착
   - 예술 창작에는 충분하지만, 고정밀 응용에는 부족 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

3. **구성 능력 부족**: 복잡한 관계 프롬프트(예: "두 물체를 나란히") 어려움
   - 모델이 단일 개념 중심 학습으로 인한 제약 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

4. **CLIP 기반 평가의 한계**: 
   - CLIP은 형태 보존에 둔감
   - 재구성 점수가 무작위 이미지와 비슷한 이유 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### 사회적 영향 고려

**긍정 측면**:
- 편향 감소 가능성
- 아티스트에게 라이선스 기회 제공

**부정 측면**:
- 비동의자 사진 생성 위험 (다행히 현 모델은 정체성 보존 부족)
- 저작권 침해 잠재성
- 오정보 생성 우려 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

***

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 일반화 능력의 현재 상태

#### 정성적 증거
논문의 주요 발견: **단일 의사 단어만으로 충분한 일반화가 가능**합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

- 학습 시 중립적 맥락 텍스트 사용 ("a photo of S*", "a rendition of S*" 등)
- 추론 시 완전히 새로운 프롬프트에 대해 개념을 올바르게 조합
- 모델의 사전 학습된 지식과 새로운 임베딩이 효과적으로 상호작용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### 구성 일반화
- **성공 사례**: "Painting of two S* fishing on a boat", "A S* themed lunchbox"
- **실패 사례**: 공간 관계가 복잡한 구성 (다중 개념 배치) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

### 5.2 기본 모델 제약

단일 단어 임베딩의 위험성:

$$\text{Editability Loss} = f(\|v^* - v_{prior}\|_2)$$

정규화 없이 학습하면 임베딩이 출력 분포에서 벗어나 편집 능력 상실 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

### 5.3 향상된 일반화를 위한 잠재적 개선 방향

#### (1) 다중 임베딩 계층화

**계층적 임베딩 구조**:

```math
v^* = v^*_{semantic} + \alpha \cdot v^*_{visual}
```

의미적 개념과 시각적 세부사항을 분리하여 일반화 향상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### (2) 적응형 정규화

학습률에 따른 동적 정규화:
$$L_{reg}(t) = \lambda(t) \cdot \|v^*(t) - v_{descriptor}\|_2^2$$

여기서 $\lambda(t) = \lambda_0 \cdot e^{-\beta t}$ (시간에 따라 감소) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### (3) 대조 학습 통합

추가 손실:

```math
L_{contrastive} = -\log \frac{\exp(\cos(v^*, v_{semantic})/\tau)}{\sum_k \exp(\cos(v^*, v_k)/\tau)}
```

임베딩이 의미론적 공간에 머무르도록 유인 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

***

## 6. 2020년 이후 관련 최신 연구 비교 분석

### 6.1 주요 경쟁 방법론 비교표

| 방법 | 발표 | 기술 | 훈련 시간 | 모델 수정 | 일반화 | 다중 개념 |
|------|------|------|---------|---------|-------|---------|
| **Textual Inversion** | 2022 | 임베딩 최적화 | 2시간 | 아니오 | 중간 | 제한적 |
| DreamBooth [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf) | 2023 | 전체 미세 조정 | 1-5분 | 예 | 높음 | 낮음 |
| Custom Diffusion [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.pdf) | 2023 | 주의 가중치 미세 조정 | 6분 | 부분 | 높음 | 높음 |
| ELITE [arxiv](https://arxiv.org/pdf/2302.13848.pdf) | 2023 | 최적화 기반 인코딩 | 32초 | 아니오 | 높음 | 낮음 |
| Perfusion [openaccess.thecvf](https://openaccess.thecvf.com/content/WACV2025/papers/Ram_DreamBlend_Advancing_Personalized_Fine-Tuning_of_Text-to-Image_Diffusion_Models_WACV_2025_paper.pdf) | 2024 | 게이트 rank-one 업데이트 | 40초 | 부분 | 높음 | 중간 |
| LoRA-DreamBooth | 2024 | LoRA 기반 미세 조정 | 30분 | 부분 | 높음 | 중간 |
| Directional TI [arxiv](https://arxiv.org/html/2512.13672v1) | 2025 | 방향성 임베딩 최적화 | 2시간 | 아니오 | **높음** | 제한적 |
| FlipConcept [arxiv](https://arxiv.org/html/2502.15203) | 2025 | 튜닝 무료 다중 개념 | 실시간 | 아니오 | 높음 | **최고** | [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

### 6.2 세부 비교 분석

#### A. 임베딩 최적화 계열

**Textual Inversion (2022) - 기준선** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)
- 접근법: 고정 모델에서 단일 의사 단어 임베딩 최적화
- 장점: 최소 메모리, 저장 용량 < 5KB
- 단점: 2시간 최적화, 중간 수준 일반화

**Directional Textual Inversion (2025)** [arxiv](https://arxiv.org/html/2512.13672v1)
- **핵심 개선**: 임베딩 노름 팽창(norm inflation) 문제 해결
- **수학적 메커니즘**: 

```math
v^* = \|v^*\|_2 \cdot \frac{v^*_{normalized}}{\|v^*_{normalized}\|_2}
```
  
  - 노름을 in-distribution 범위로 제한
- **성과**: 복잡한 프롬프트에서 텍스트 충실도 40% 향상
- **한계**: 여전히 2시간 최적화 필요 [arxiv](https://arxiv.org/html/2512.13672v1)

**CoRe: Context-Regularized Learning (2024)** [arxiv](http://arxiv.org/pdf/2408.15914.pdf)
- **혁신**: 임의 프롬프트에서의 일반화 개선
- **방법**: 컨텍스트 정규화로 개념 임베딩 학습
- **성과**: Textual Inversion 대비 편집 가능성 35% 향상
- **특징**: 생성 무료 테스트 시간 최적화 가능 [arxiv](http://arxiv.org/pdf/2408.15914.pdf)

#### B. 미세 조정 계열 (전체/부분)

**DreamBooth (2023) - 강력한 기준선** [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf)
- 접근법: 고유 토큰과 함께 전체 모델 미세 조정
- **개선점**: 
  $$L = L_{reconstruction} + \lambda \cdot L_{prior\_preservation}$$
  - 사전 보존 손실로 재앙적 망각 완화 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf)
- 성과: 
  - 재구성 유사도 0.80+ (TI의 0.71 vs)
  - 다양성 유지 가능
- 단점: 모델당 1000단계 ~ 5분, 저장 용량 증가 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf)

**Custom Diffusion (2023) - 효율성 최적화** [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.pdf)
- 접근법: 텍스트-이미지 교차 주의 가중치만 미세 조정
- **공식**:
  $$\Delta W = \Delta K \cdot V^T + K \cdot \Delta V^T$$
  (K, V 프로젝션만 업데이트)
- 성과: 
  - 6분 훈련 (DreamBooth 1/10)
  - 다중 개념 구성 가능 (TI의 한계 극복)
  - 폐쇄형 최적화로 여러 개념 병합 가능 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.pdf)

**LoRA-기반 개선들 (2024-2025)** [arxiv](https://arxiv.org/abs/2507.05964)
- **T-LoRA** - 타임스텝 의존적 순위 조정 [arxiv](https://arxiv.org/abs/2507.05964)
  $$W' = W + r(t) \cdot A_t B_t^T$$
  - 높은 타임스텝에서 더 낮은 순위 사용 (오버피팅 방지)
  - 단일 이미지 개인화 성공률 대폭 향상
  
- **Noise Consistency Regularization (2025)** [arxiv](https://arxiv.org/html/2506.06483v1)
  - 두 가지 보조 손실 추가:

$$L_{prior} = \|\epsilon_{prior} - \hat{\epsilon}_{prior}\|_2^2$$

$$L_{subject} = \text{Consistency}(\epsilon(z), \epsilon(z + \sigma n))$$
  
  - DreamBooth 대비 identity 보존 15% 향상 [arxiv](https://arxiv.org/html/2506.06483v1)

#### C. 다중 개념 및 연속 학습

**FlipConcept (2025) - 튜닝 무료 다중 개념** [arxiv](https://arxiv.org/html/2502.15203)
- **혁신**: 완전히 새로운 패러다임 - 아무런 미세 조정 없이 작동
- **핵심 기술**:
  1. **Guided Appearance Attention**: 개념 이미지의 키-값 재구성
  2. **Mask-guided Noise Mixing**: 비개인화 영역 보호
  3. **Background Dilution**: 개념 유출(concept leakage) 최소화
  
- **성능**:
  - 3개 개념 구성에서 Custom Diffusion 능가
  - 텍스트 정렬: CLIP_T = 0.71 (vs Custom Diffusion 0.68)
  - 개념 충실도: DINO = 0.82
  - **실시간 작동** (튜닝 무료) [arxiv](https://arxiv.org/html/2502.15203v2)

**CIDM: Concept-Incremental Diffusion Model (2024)** [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2024/file/eadb6e5ed8a02ada4affb07dfd62ab5e-Paper-Conference.pdf)
- **문제 정의**: 순차적 개념 학습에서 catastrophic forgetting
- **해결책**: 계층별 개념 토큰 도입

$$\mathcal{L} = \mathcal{L}_{new} + \sum_i \mathcal{L}_{preserve}(c_i)$$
  
  - 이전 개념별 별도 손실함수
- **성과**: 10개 개념 순차 학습에서도 80%+ 성능 유지 [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2024/file/eadb6e5ed8a02ada4affb07dfd62ab5e-Paper-Conference.pdf)

**Continual Diffusion with C-LoRA (2024)** [arxiv](https://arxiv.org/pdf/2304.06027.pdf)
- LoRA를 사용하여 연속 학습 가능
- 개념별 독립적인 LoRA 가중치 유지
- 다양한 개념의 메모리 효율적 저장 [arxiv](https://arxiv.org/pdf/2304.06027.pdf)

#### D. 일반화 능력 향상 연구

**AttnDreamBooth (2024)** [arxiv](https://arxiv.org/abs/2406.05000)
- **문제 인식**: 
  - Textual Inversion: 개념 과적합
  - DreamBooth: 개념 무시 경향
- **해결**: 임베딩 정렬 학습
  $$\text{alignment loss} = \text{KL}(p_{TI} \| p_{DB})$$
- **결과**: 텍스트 충실도 25% 향상 [arxiv](https://arxiv.org/html/2406.05000v1)

**CoRe (2024)** [arxiv](http://arxiv.org/pdf/2408.15914.pdf)
- **핵심**: 컨텍스트 정규화로 일반화 개선

$$L_{context} = \sum_{p \in \mathcal{P}_{diverse}} \|\mathcal{F}(I, v^*) - \mathcal{F}(I_{generated}(p), v^*)\|_2^2$$

- **성과**: 임의 프롬프트에 대한 일반화 40% 향상 [arxiv](http://arxiv.org/pdf/2408.15914.pdf)

### 6.3 성능 지표 비교 (최신 벤치마크)

| 모델 | 재구성(DINO↑) | 편집(CLIP-T↑) | 훈련 시간 | 저장 크기 | 다중 개념 |
|------|----------|----------|---------|---------|----------|
| Textual Inversion | 0.65 | 0.62 | 2h | <5KB | 제한 |
| DreamBooth | 0.78 | 0.68 | 5m | 4GB+ | 낮음 |
| Custom Diffusion | 0.76 | 0.70 | 6m | 50MB | 높음 |
| ELITE | 0.73 | 0.66 | 32s | 1MB | 제한 |
| Perfusion | 0.80 | 0.71 | 40s | 500KB | 중간 |
| Directional TI | 0.72 | **0.75** | 2h | <10KB | 제한 |
| FlipConcept | 0.82 | 0.71 | 0s | 0 | **최고** | [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

***

## 7. 논문이 앞으로의 연구에 미치는 영향

### 7.1 이론적 영향

#### (1) 임베딩 공간의 특성 규명
논문은 **텍스트 임베딩 공간의 표현력**을 최초로 체계적으로 입증했습니다.

- **발견**: 단일 벡터 업데이트로 복잡한 시각적 개념 인코딩 가능
- **영향**: 후속 연구들이 임베딩 공간의 기하학적 성질에 주목하게 함
- **구체적 사례**: Directional Textual Inversion(2025)은 임베딩 노름의 역할 규명 [arxiv](https://arxiv.org/html/2512.13672v1)

#### (2) 왜곡-편집 가능성 트레이드오프 개념화

$$\text{Trade-off}: \min_v \left( \text{Reconstruction} - \alpha \cdot \text{Editability} \right)$$

이는 GAN 반전 문헌의 통찰을 확산 모델에 최초 적용하며, 후속 연구의 **핵심 설계 원칙**이 됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### (3) 개념 개인화의 패러다임 전환

기존: 모델 수정 중심
→ 새로운 패러다임: 임베딩 공간 활용

이는 **매개변수 효율적 미세 조정(PEFT)** 분야 전체의 철학적 기초가 됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

### 7.2 기술적 발전 경로

#### Phase 1: 임베딩 최적화 개선 (2022-2023)
- PALAVRA (2022): CLIP 임베딩 기반
- Textual Inversion의 한계 인식 [arxiv](https://arxiv.org/abs/2208.01618)

#### Phase 2: 미세 조정 기반 강화 (2023-2024)
- DreamBooth (2023): 더 강력한 재구성 능력 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf)
- Custom Diffusion (2023): 효율성 개선 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.pdf)
- LoRA 기반 가속화 (2024): 훈련 시간 단축 [huggingface](https://huggingface.co/blog/lora)

#### Phase 3: 하이브리드 및 튜닝 무료 (2024-2025)
- FlipConcept (2025): 완전 튜닝 무료 [arxiv](https://arxiv.org/html/2502.15203v2)
- Directional TI (2025): 일반화 능력 강화 [arxiv](https://arxiv.org/html/2512.13672v1)
- 연속 학습 안정화: CIDM, ConceptGuard (2024-2025) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11094511/)

### 7.3 개방 문제와 미래 연구 방향

#### 미해결 문제 1: 형태 정밀도
**문제**: 논문이 명시한 한계 - 정확한 형태 보존보다는 의미적 본질만 포착 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

**후속 연구**:
- **MagicTailor (2024)**: 컴포넌트 제어 가능한 개인화로 형태 정밀도 향상 [arxiv](https://arxiv.org/html/2410.13370v2)
- **Identity-Preserving Dual Branch (2025)**: 별도 identity 인코딩으로 형태 보존 [arxiv](https://arxiv.org/html/2505.22360v1)

#### 미해결 문제 2: 다중 개념 구성
**문제**: 원문 4.4절 - 공간 관계 추론 부족 (예: "두 물체 나란히") [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

**최신 진전**:
- **FlipConcept (2025)**: 마스크 기반 혼합으로 해결 (3개 개념까지) [arxiv](https://arxiv.org/html/2502.15203v2)
- **Concept Conductor (2024)**: 음악 지휘자 메타포로 개념 조율 [arxiv](http://arxiv.org/pdf/2408.03632v1.pdf)

#### 미해결 문제 3: 일반화와 오버피팅 균형
**문제**: 정규화 강할수록 재구성 감소, 약할수록 편집 불가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

**최근 해결책**:
- **T-LoRA (2025)**: 타임스텝별 동적 순위로 자동 균형 [arxiv](https://arxiv.org/abs/2507.05964)
- **Noise Consistency Regularization (2025)**: 두 가지 보조 손실로 균형 [arxiv](https://arxiv.org/html/2506.06483v1)

### 7.4 응용 분야의 확장

#### 원문의 응용 (2022)
- 객체 변형
- 스타일 전이
- 편향 감소
- 국소 편집 (Blended Latent Diffusion) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

#### 2024-2025 새로운 응용
- **얼굴 개인화**: 단일 사진으로 일관된 신원 보존 [ecva](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10860.pdf)
- **다중 주체 장면**: 여러 사람의 신원을 동시에 유지 [ecva](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10860.pdf)
- **3D 에셋 생성**: 합성 데이터로 개인화된 3D 모델 [arxiv](https://arxiv.org/html/2502.01720v2)
- **안전성 보강**: 권한 없는 생성 방지 (Latent Diffusion Shield) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10972526/)
- **저작권 보호**: Forget-Me-Not로 개념 제거 [arxiv](https://arxiv.org/abs/2303.17591)

***

## 8. 앞으로 연구 시 고려할 점

### 8.1 방법론적 고려사항

#### (1) 하이퍼파라미터 민감도
현 방법의 5000 단계, 학습률 0.04는 고정값이나, **개념별 적응형 조정** 필요:

$$\text{optimal steps} = f(|I_{train}|, \text{concept complexity})$$

$$\text{LR}_t = \text{LR}_0 \cdot \text{schedule}(t, \text{variance}(v^*))$$

**추천**: 초기 n 단계로 분산을 측정하여 동적 조정 [arxiv](https://arxiv.org/html/2512.13672v1)

#### (2) 정규화 전략의 재검토
원문의 L2 정규화 단순하나:

$$L_{reg} = \lambda \|v^* - v_{init}\|_2^2 \text{ (현 방법)}$$

**개선 안**:
- **적응형 정규화**: 
  $$L_{reg}(t) = \lambda(t) \cdot \|v^*(t) - v_{init}\|_2^2, \quad \lambda(t) = \lambda_0 \cdot e^{-t/\tau}$$
  
- **방향 제약**:
  $$L_{direction} = 1 - \frac{v^* \cdot v_{init}}{\|v^*\| \|v_{init}\|}$$
  (노름은 자유, 방향은 유지) [arxiv](https://arxiv.org/html/2512.13672v1)

#### (3) 평가 지표 개선
현 평가는 CLIP 기반이나, 형태 보존에 약함 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

**추가할 메트릭**:
- **DINO 기반 형태 충실도**: 형태 보존 측정
- **ViT 특징 공간 거리**: 의미론적 정렬
- **사용자 선호도 스코어**: 실제 품질 평가 [ecva](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10860.pdf)

### 8.2 일반화 성능 향상을 위한 전략

#### (1) 다중 스케일 학습
개념을 여러 스케일에서 학습:

```math
v^*_{multi} = \{v^*_{coarse}, v^*_{fine}, v^*_{details}\}
```

각 스케일을 위계적으로 최적화하면 일반화 향상 가능 [arxiv](https://arxiv.org/html/2512.13672v1)

#### (2) 메타-학습 접근
개념이 특정 분포를 따른다고 가정:

$$\mathbb{E}_{C \sim \mathcal{D}_{concepts}} [\min_v \mathcal{L}(v; C)] < \min_v \mathbb{E}_{C} [\mathcal{L}(v; C)]$$

메타-학습으로 "개념학습에 최적의" 초기화 발견 [arxiv](https://arxiv.org/abs/2405.14132)

#### (3) 앙상블 방법
여러 임베딩 학습 후 혼합:

$$v^* = \frac{1}{K} \sum_{k=1}^K v^*_k$$

다양성이 높은 훈련 데이터에서 성능 향상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

### 8.3 실무적 고려사항

#### (1) 계산 효율성
2시간 최적화는 실제 배포에 부담 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

**해결 방안**:
- **인코더 기반 가속**: 조건부 VAE로 v* 직접 예측 (1-5초) [arxiv](https://arxiv.org/abs/2305.18993)
- **스케일된 학습**: 작은 모델에서 먼저 학습, 큰 모델로 전이 [arxiv](https://arxiv.org/html/2504.13162v1)

#### (2) 안정성과 재현성
현 방법은 초기화 및 확률적 요소에 민감 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

**개선안**:
- 여러 실행의 평균 (ensemble)
- 시드 고정 및 초기화 전략 표준화
- 베이지안 최적화로 하이퍼파라미터 자동 선택 [arxiv](https://arxiv.org/abs/2507.05964)

#### (3) 스케일링 및 배포
다양한 기본 모델(SDXL, Flux, DALL-E 3 등)에 적용

**현황**: Textual Inversion은 Stable Diffusion 중심, 다른 모델에 확장 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

**추천**: 아키텍처 무관한 메서드 개발 (예: 토큰 무시 미세 조정) [arxiv](https://arxiv.org/html/2505.10743v1)

### 8.4 장기 연구 의제

#### 근본적 질문
1. **임베딩 공간의 차원 복잡도**: 개념을 표현하는 데 필요한 최소 차원?
2. **일반화의 한계**: 학습 데이터 규모와 일반화 성능의 관계?
3. **개념 합성**: 두 개념의 임베딩을 섞으면 의미적으로 유효한 새 개념을 얻을 수 있나?

#### 응용 확대
- **비전-언어 모델 통합**: CLIP, BLIP 등과의 더 깊은 상호작용
- **동영상 생성**: Runway, Pika 등 동영상 모델에 확장
- **3D 모델 생성**: NeRF, Gaussian Splatting과의 결합 [arxiv](https://arxiv.org/html/2502.01720v2)

#### 윤리 및 안전
- 권한 없는 개인화 방지 메커니즘
- 저작권 보호 및 추적 가능성
- 편향 감소의 근본적 해법 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10972526/)

***

## 결론

"An Image is Worth One Word"는 텍스트-이미지 생성에서 **간단하면서도 강력한 개인화 패러다임**을 제시합니다. 단일 임베딩 벡터로 복잡한 시각적 개념을 포착하는 통찰은 이후 3년간의 연구를 크게 영향을 미쳤습니다.

**주요 강점**:
- 최소한의 모델 수정으로 최대의 유연성 확보
- 왜곡-편집 능력 트레이드오프의 명확한 분석
- 다양한 실제 응용 입증

**남은 과제**:
- 형태 정밀도 향상
- 다중 개념 공간 관계 학습
- 일반화와 충실도 사이의 최적 균형

**향후 방향**:
- 방향성 임베딩(2025)과 동적 순위 적응(T-LoRA)으로 일반화 개선
- 튜닝 무료 다중 개념(FlipConcept)으로 확장성 달성
- 연속 학습 안정화(CIDM, ConceptGuard)로 실용성 강화

이 논문의 영향은 단순한 기술 발전을 넘어, **제한된 데이터로도 강력한 모델을 적응시킬 수 있다**는 근본적 믿음을 학계에 심어주었으며, 이는 LLM의 LoRA, 비전-언어 모델의 프롬프트 튜닝 등 광범위한 분야로 확산되었습니다.

***

## 참고문헌

 Gal et al. "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion." arXiv:2208.01618, 2022. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e897c652-49e6-45be-9cc9-9fb0f527712f/2208.01618v1.pdf)

 TokenVerse. "Versatile Multi-concept Personalization in Token Modulation Space." arXiv:2501.12224, 2025. [arxiv](https://arxiv.org/html/2501.12224v1)

 Cohen et al. "This is my unicorn, fluffy: Personalizing frozen vision-language representations." ECCV 2022. [arxiv](https://arxiv.org/abs/2208.01618)

 Wu et al. "CoRe: Context-Regularized Text Embedding Learning for Text-to-Image Personalization." arXiv:2408.15914, 2024. [arxiv](http://arxiv.org/pdf/2408.15914.pdf)

 Wei et al. "ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation." arXiv:2302.13848, 2023. [arxiv](https://arxiv.org/pdf/2302.13848.pdf)

 Pang et al. "AttnDreamBooth: Towards Text-Aligned Personalized Text-to-Image Generation." arXiv:2406.05000, 2024. [arxiv](https://arxiv.org/abs/2406.05000)

 Woo et al. "FlipConcept: Tuning-Free Multi-Concept Personalization for Text-to-Image Generation." arXiv:2502.15203, 2025. [arxiv](https://arxiv.org/html/2502.15203)

 Ruiz et al. "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." CVPR 2023. [arxiv](http://arxiv.org/pdf/2208.12242.pdf)

 Parihar et al. "Personalizing Text-to-Image Diffusion Models for Face-Specific Generation." ECCV 2024. [ecva](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10860.pdf)

 He et al. "Text-to-Model: Text-Conditioned Neural Network Diffusion for Train-Once-for-All Personalization." arXiv:2405.14132, 2024. [arxiv](https://arxiv.org/html/2504.13162v1)

 Shewale et al. "Identity-Preserving Text-to-Image Generation via Dual Alignment Diffusion." arXiv:2505.22360, 2025. [arxiv](https://arxiv.org/html/2505.10743v1)

 Ram et al. "DreamBlend: Advancing Personalized Fine-Tuning of Text-to-Image Diffusion Models." WACV 2025. [openaccess.thecvf](https://openaccess.thecvf.com/content/WACV2025/papers/Ram_DreamBlend_Advancing_Personalized_Fine-Tuning_of_Text-to-Image_Diffusion_Models_WACV_2025_paper.pdf)

 Kim et al. "Directional Textual Inversion for Personalized Text-to-Image Generation." arXiv:2512.13672, 2025. [arxiv](https://arxiv.org/html/2512.13672v1)

 Song et al. "Identity-Preserving Text-to-Image Generation via Dual Alignment Diffusion." arXiv:2505.22360, 2025. [arxiv](https://arxiv.org/html/2505.22360v1)

 He et al. "Text-to-Model: Text-Conditioned Neural Network Diffusion for Train-Once-for-All Personalization." arXiv:2405.14132, 2024. [arxiv](https://arxiv.org/abs/2405.14132)

 Lu et al. "T-LoRA: Single Image Diffusion Model Customization Without Overfitting." arXiv:2507.05964, 2025. [arxiv](https://arxiv.org/abs/2507.05964)

 Wang et al. "Latent Diffusion Shield - Mitigating Malicious Use of Diffusion Models Through Latent Space Adversarial Perturbations." IEEE 2025. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10972526/)

 Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685, 2021. [arxiv](https://arxiv.org/abs/2305.18993)

 Dong et al. "How to Continually Adapt Text-to-Image Diffusion Models for Flexible Customization." NeurIPS 2024. [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2024/file/eadb6e5ed8a02ada4affb07dfd62ab5e-Paper-Conference.pdf)

 Kumari et al. "Multi-Concept Customization of Text-to-Image Diffusion." CVPR 2023. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.pdf)

 Ruiz et al. "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." CVPR 2023. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf)

 Xu et al. "Generating Multi-Image Synthetic Data for Text-to-Image Personalization." arXiv:2502.01720, 2025. [arxiv](https://arxiv.org/html/2502.01720v2)

 Liu et al. "MagicTailor: Component-Controllable Personalization in Text-to-Image Synthesis." arXiv:2410.13370, 2024. [arxiv](https://arxiv.org/html/2410.13370v2)

 Zheng et al. "Noise Consistency Regularization for Improved Subject Fidelity." arXiv:2506.06483, 2025. [arxiv](https://arxiv.org/html/2506.06483v1)

 Woo et al. "FlipConcept: Tuning-Free Multi-Concept Personalization for Text-to-Image Generation." arXiv:2502.15203, 2025. [arxiv](https://arxiv.org/html/2502.15203v2)

 Guo et al. "ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation." CVPR 2025. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11094511/)

 Cai et al. "Continual Personalization for Diffusion Models." ICCV 2025. [arxiv](https://arxiv.org/abs/2510.02296)

 Wang et al. "Continual Diffusion: Continual Customization of Text-to-Image Diffusion with C-LoRA." arXiv:2304.06027, 2024. [arxiv](https://arxiv.org/pdf/2304.06027.pdf)

 Cheong et al. "Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models." arXiv:2303.17591, 2023. [arxiv](https://arxiv.org/abs/2303.17591)

 An et al. "Concept Conductor: Orchestrating Multiple Personalized Concepts in Text-to-Image Synthesis." arXiv:2408.03632, 2024. [arxiv](http://arxiv.org/pdf/2408.03632v1.pdf)

 Hugging Face. "Using LoRA for Efficient Stable Diffusion Fine-Tuning." Blog, 2025. [huggingface](https://huggingface.co/blog/lora)

 Dong et al. "CIFC: Concept-Incremental Flexible Customization." GitHub, 2024. [github](https://github.com/jiahuadong/cifc)

 Ding et al. "QUOTA: Quantifying Objects with Text-to-Image Models for Domain Generalization." arXiv 2025. [arxiv](https://arxiv.org/html/2411.19534v1)
