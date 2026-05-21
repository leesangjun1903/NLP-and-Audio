
# GOATex: Geometry & Occlusion-Aware Texturing

> ⚠️ **주의**: 본 답변은 공개된 arXiv 초록(arxiv.org/abs/2511.23051), NeurIPS 2025 포스터(neurips.cc), 공식 프로젝트 페이지(goatex3d.github.io), OpenReview PDF, 공식 GitHub(github.com/KorMachine/GOATex) 를 기반으로 작성되었습니다. 논문 전문의 세부 수식 표기는 공개된 범위 내에서만 제시하며, 확인되지 않은 수식은 명시적으로 구분합니다.

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

GOATex는 3D 메시 텍스처링을 위한 **diffusion 기반 방법**으로, 외부(exterior) 및 내부(interior) 표면 모두에 대해 고품질 텍스처를 생성합니다.

기존 방법들은 가시(visible) 영역에서는 잘 작동하지만, **가려진(occluded) 내부를 처리하는 메커니즘이 본질적으로 부재**하여 불완전한 텍스처와 눈에 띄는 솔기(seam)가 발생합니다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| ① 새로운 문제 정의 | 내부 표면 텍스처 생성이라는, 실용적으로 중요하지만 아직 충분히 연구되지 않은 과제를 최초로 제시하고 다룬 논문 |
| ② 프레임워크 제안 | 사전 학습된 diffusion 모델의 추가 학습 없이 외부 및 내부 영역 모두를 텍스처링하는 ray 기반 occlusion-aware 프레임워크 제안; 내부·외부 표면에 대한 이중 프롬프트(dual prompting) 지원 |
| ③ 성능 우위 | 사용자 연구(user study)와 GPT 기반 평가에서 GOATex가 기존 방법 대비 강한 선호도를 보이며, 가시 및 가려진 표면 모두에서 최고 수준의 텍스처 품질 달성 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

3D 메시의 내부 표면을 텍스처링하는 것은 고유한 어려움을 가집니다: 이러한 영역은 외부 시점에서 완전히 가려지는 경우가 많아, 기존 렌더링 기반 파이프라인에서는 거의 커버되지 않습니다.

Text2Tex, SyncMVD 같은 뷰 기반 생성 방식은 가려진 기하 구조에 접근할 수 없어, Voronoi 기반 외삽(extrapolation)과 같은 휴리스틱에 의존하게 되며, 이는 단순하고 불일치하는 텍스처 및 가시적 솔기(seam)를 초래합니다.

Paint3D, TEXGen과 같은 UV 공간 생성 방법은 내부 영역에서 개선된 성능을 보이지만, UV 맵 내에서 내부와 외부 표면을 구별할 수 없기 때문에 여전히 평면적인 색상이나 반복 패턴과 같은 저주파 텍스처를 생성하는 한계를 지닙니다.

---

### 2-2. 제안 방법 및 핵심 개념

#### (A) Hit Level — 핵심 개념 정의

GOATex는 **hit levels**라는 개념에 기반한 occlusion-aware 텍스처링 프레임워크를 도입합니다. Hit level은 **멀티뷰 레이 캐스팅(multi-view ray casting)**을 통해 메시 면(face)의 상대적 깊이를 수치화한 것입니다.

이를 통해 메시 면들을 가장 바깥쪽(outermost)부터 가장 안쪽(innermost)까지 **순서가 있는 가시성 레이어(ordered visibility layers)**로 분할합니다.

수식(개념적 표현, 논문 전문 기반 재구성):

레이 $r$이 뷰 $v$에서 메시를 가로지를 때, 면(face) $f$의 hit level $H(f)$는 다음과 같이 정의됩니다:

$$H(f) = \text{median}_{v \in \mathcal{V}} \left[ \text{rank}_v(f) \right]$$

여기서 $\text{rank}_v(f)$는 뷰 $v$에서 레이가 $f$를 교차할 때 몇 번째 교차인지를 나타내는 순위입니다. $H(f) = 1$이면 외부(exterior), $H(f) \geq 2$이면 내부(interior) 레이어로 분류됩니다.

> ⚠️ 위 수식은 논문의 개념 설명을 바탕으로 재구성한 것으로, 논문 원문의 정확한 기호 표기와 다를 수 있습니다.

---

#### (B) Two-Stage Visibility Control Strategy

GOATex는 **2단계 가시성 제어 전략(two-stage visibility control strategy)**을 적용하여, 구조적 일관성을 유지하면서 내부 영역을 점진적으로 드러내고, 이후 사전 학습된 diffusion 모델로 각 레이어를 텍스처링합니다.

- **Stage 1 (Exterior Texturing):** Hit level 1 ($H(f) = 1$)에 해당하는 가장 바깥 레이어에 텍스처 생성. 외부 텍스트 프롬프트 적용.
- **Stage 2 (Interior Texturing):** 외부 레이어를 가상으로 제거하여 내부 표면을 렌더링 가능한 상태로 전환 후, 내부 프롬프트로 텍스처 생성.

---

#### (C) Soft UV-Space Blending

레이어 간에 획득된 텍스처를 매끄럽게 병합하기 위해, **뷰 의존적 가시성 신뢰도(view-dependent visibility confidence)**에 기반하여 각 텍스처의 기여를 가중하는 **소프트 UV 공간 블렌딩(soft UV-space blending)** 기법을 제안합니다.

최종 UV 텍스처 $T_{\text{final}}$의 블렌딩은 개념적으로 다음과 같이 표현할 수 있습니다:

$$T_{\text{final}}(u, v) = \frac{\sum_{\ell} w_{\ell}(u, v) \cdot T_{\ell}(u, v)}{\sum_{\ell} w_{\ell}(u, v)}$$

여기서:
- $T_{\ell}(u, v)$: 레이어 $\ell$에서 생성된 텍스처
- $w_{\ell}(u, v)$: 뷰 의존적 가시성 신뢰도 가중치

> ⚠️ 이 수식 또한 논문 개념 설명을 기반으로 한 재구성이며, 정확한 원문 수식과 다를 수 있습니다.

---

### 2-3. 모델 구조 (파이프라인 개요)

```
[입력: 3D 메시 + 텍스트 프롬프트 (외부/내부)]
         ↓
[1단계: Hit Level 추정]
  - 멀티뷰 레이 캐스팅 수행
  - 각 face에 hit level H(f) 할당
  - 가시성 레이어 분할 (L1, L2, ...)
         ↓
[2단계: 외부(L1) 텍스처링]
  - L1 면만 렌더링
  - 사전학습된 diffusion 모델로 텍스처 생성 (외부 프롬프트)
         ↓
[3단계: 내부(L2+) 텍스처링]
  - 외부 레이어 가상 제거 → 내부 가시화
  - 사전학습된 diffusion 모델로 텍스처 생성 (내부 프롬프트)
         ↓
[4단계: Soft UV-Space Blending]
  - 뷰 의존적 가중치로 레이어별 텍스처 병합
         ↓
[출력: 외부 + 내부 고품질 텍스처 메시]
```

기존 연구들과 달리 GOATex는 **사전 학습된 diffusion 모델의 비용이 큰 파인튜닝 없이 완전히 동작**하며, 외부 및 내부 메시 영역에 대한 개별 프롬프트 입력을 허용하여 레이어별 외관에 대한 세밀한 제어를 가능하게 합니다.

---

### 2-4. 성능 향상

실험 결과, GOATex는 기존 방법들을 일관되게 능가하며, 가시 영역과 가려진 영역 모두에서 **솔기 없는(seamless), 고품질(high-fidelity) 텍스처**를 생성합니다.

사용자 연구와 GPT 기반 평가 모두에서 GOATex가 기존 방법 대비 강하게 선호되었으며, 가시 및 가려진 표면 모두에서 SOTA 텍스처 품질을 달성했습니다.

**비교 대상 방법들:**

| 방법 | 방식 | 한계 |
|------|------|------|
| Text2Tex | 뷰 기반 + inpainting | 깊이 조건부 diffusion 모델로 점진적 텍스처 생성, 제한된 전역 기하 맥락으로 크로스-뷰 불일치 발생 |
| SyncMVD | 멀티뷰 동기화 | 뷰 기반 생성 + 역투영 방식으로 가려진 기하에 접근 불가, Voronoi 기반 채움으로 불일치 발생 |
| Paint3D / TEXGen | UV 공간 생성 | UV 맵 내에서 내부·외부 표면을 구분하지 못해 저주파 텍스처 생성, 의미론적 풍부함 감소 |

---

### 2-5. 한계점

논문 전문에서 명시적으로 서술된 한계점은 공개된 초록 및 요약에서 상세히 확인되지 않았습니다. 단, 방법론적 구조에서 추론 가능한 한계는 다음과 같습니다:

1. **레이 캐스팅 복잡도**: 복잡한 기하 구조(예: 매우 많은 내부 레이어)에서 hit level 계산 비용이 증가할 수 있습니다.
2. **매우 좁은 내부 공간**: 레이가 충분히 통과할 수 없을 정도로 좁은 내부 공간에서는 hit level 추정의 정확도가 저하될 수 있습니다.
3. **프롬프트 의존성**: 내부/외부 이중 프롬프트의 품질이 최종 텍스처 품질에 영향을 미칩니다.

> ⚠️ 위 한계점 1~3은 논문 구조에서 추론된 것이며, 논문 본문에서 저자가 명시한 한계가 아닐 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

GOATex의 일반화 성능(generalization)과 관련된 핵심 강점은 다음과 같습니다:

### (1) Fine-tuning 불필요 → 범용 diffusion 모델 활용

GOATex는 기존 연구들과 달리 사전 학습된 diffusion 모델의 비용이 큰 파인튜닝 없이 완전히 동작하며, 외부 및 내부 메시 영역에 대한 개별 프롬프트 입력을 허용합니다.

이는 Stable Diffusion 등 최신 대형 생성 모델이 업데이트될 때 **플러그인(plug-in) 방식으로 즉시 활용 가능**함을 의미하며, 특정 데이터셋에 과적합되지 않는 구조적 일반화 이점을 제공합니다.

### (2) 기하 구조 독립적인 ray-based 접근

Hit level 기반 레이어링은 특정 메시 카테고리(예: 가구, 건물 등)에 종속되지 않고, **임의의 3D 메시**에 적용 가능한 순수 기하학적 방법입니다. 이는 학습 데이터 분포 바깥의 객체에도 적용 가능한 일반화 능력을 의미합니다.

### (3) 이중 프롬프트(Dual Prompting)로 제어 가능성 확장

GOATex의 핵심 장점은 외부 및 내부 메시 영역에 대해 각각 다른 프롬프트를 지원하는 능력이며, 이는 점진적이고 레이어화된 텍스처링 프레임워크 덕분에 가능합니다.

이 설계는 다양한 응용 도메인(건축 인테리어, 의류, 가구 등)에 걸친 일반화 가능성을 높입니다.

### (4) 일반화 성능 향상을 위한 잠재적 개선 방향

| 방향 | 설명 |
|------|------|
| **더 강력한 backbone 활용** | SDXL, FLUX 등 최신 diffusion 모델 교체 시 텍스처 품질 자동 향상 |
| **멀티모달 프롬프트** | 텍스트 + 이미지 결합 프롬프트로 레퍼런스 기반 일반화 |
| **동적 hit level 해상도** | 메시 복잡도에 따른 적응적 레이어 수 결정 |
| **비정형 위상(topology)** | 자기교차(self-intersecting) 메시나 개방형(open) 메시 처리 확장 |

---

## 4. 미래 연구에 미치는 영향 및 고려할 점

### 4-1. 앞으로의 연구에 미치는 영향

#### (A) 내부 표면 텍스처링이라는 새로운 연구 방향 개척

GOATex는 3D 메시 텍스처링 분야에서 아직 충분히 탐구되지 않았던 **가려진 내부 표면 텍스처 생성** 과제를 최초로 제시하고 해결한 논문입니다.

이는 향후 다음과 같은 연구들을 자극할 것으로 예상됩니다:
- 내부 표면 벤치마크 데이터셋 구축
- 내부·외부 일관성을 평가하는 새로운 평가 지표 개발
- NeRF, 3D Gaussian Splatting 등 다른 3D 표현 방식으로의 확장

#### (B) Fine-tuning-free 텍스처링 패러다임의 강화

Paint3D, TEXGen과 같은 최근 연구들은 diffusion 모델을 파인튜닝하는 UV 공간 정제(UV-space refinement)와 인페인팅(inpainting) 방식을 탐구했습니다.

GOATex는 이와 반대로 파인튜닝 없이도 우수한 성능을 달성함을 보여줌으로써, **훈련 비용 없는(training-free) 텍스처링 방향**의 연구에 중요한 기준점을 제시합니다.

#### (C) 레이 기반 기하 분석의 재발견

레이 캐스팅을 텍스처링 파이프라인에 접목한 접근 방식은, 향후 다음 연구들에서 활용될 수 있습니다:
- **씬(scene) 레벨 텍스처링**: 단일 객체를 넘어 전체 씬의 내외부 텍스처 생성
- **물리 기반 렌더링(PBR) 텍스처링**: 빛의 경로를 고려한 재질(material) 생성
- **증강현실(AR)/가상현실(VR)**: 실내 공간의 내부 텍스처가 중요한 응용 분야

---

### 4-2. 앞으로의 연구 시 고려할 점

| 고려 사항 | 설명 |
|-----------|------|
| **① 평가 지표 표준화** | 내부 표면 텍스처 품질을 정량적으로 평가하는 표준 지표(metrics) 개발 필요 |
| **② 동적/변형 가능 메시** | 캐릭터 애니메이션 등 변형 가능한 메시에서의 내부 텍스처 일관성 유지 방법 필요 |
| **③ 연산 효율성** | 멀티뷰 레이 캐스팅의 계산 비용 최적화(예: GPU 가속 BVH 구조 활용) |
| **④ PBR 확장** | 단순 알베도(albedo) 텍스처를 넘어 법선(normal), 거칠기(roughness) 등 PBR 속성으로 확장 |
| **⑤ 3DGS / NeRF 통합** | 메시 기반을 넘어 3D Gaussian Splatting, NeRF 등 암묵적(implicit) 표현과의 통합 |
| **⑥ 데이터 편향 해소** | 내부 표면이 있는 3D 자산(asset) 데이터의 부족 문제 해결 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 방식 | 내부 표면 처리 | 파인튜닝 필요 | 주요 한계 |
|------|------|------|--------------|-------------|----------|
| **TEXTure** (Richardson et al.) | 2023 | 뷰 기반 + trimap 깊이 diffusion | ✗ | ✗ | 크로스-뷰 불일치, 제한된 전역 기하 맥락 |
| **Text2Tex** (Chen et al.) | 2023 | 뷰 기반 + inpainting | 부분적 (Voronoi) | ✗ | Voronoi 기반 채움으로 단순하고 불일치하는 텍스처 및 솔기 발생 |
| **SyncMVD** (Liu et al.) | 2024 | 멀티뷰 동기화 diffusion | 부분적 | ✗ | 가려진 기하 접근 불가 |
| **Paint3D** | 2023 | UV 공간 + 파인튜닝 | 부분적 | ✓ | 내부·외부 구분 불가, 저주파 텍스처 |
| **TEXGen** (Yu et al.) | 2024 | UV 공간 + 파인튜닝 | 부분적 | ✓ | UV 맵 내 내외부 구분 불가 |
| **TexFusion** | 2023 | Sequential Interlaced Multi-view Sampler | ✗ | ✗ | 멀티뷰 외관 단서 통합으로 일관성 개선 시도 |
| **GOATex** (Kim et al.) | **2025** | Ray 기반 hit level + 레이어 텍스처링 | **✓ (완전)** | **✗** | 좁은 내부 공간, 레이 캐스팅 복잡도 |

---

## 참고 자료 및 출처

1. **arXiv 논문 (주요 출처)**: Kim, H., Kim, K., Lee, A., Lee, W. "GOATex: Geometry & Occlusion-Aware Texturing" — [https://arxiv.org/abs/2511.23051](https://arxiv.org/abs/2511.23051) (2025.11.28)
2. **HTML 논문 전문**: [https://arxiv.org/html/2511.23051](https://arxiv.org/html/2511.23051)
3. **NeurIPS 2025 공식 포스터**: [https://neurips.cc/virtual/2025/poster/117378](https://neurips.cc/virtual/2025/poster/117378)
4. **공식 프로젝트 페이지**: [https://goatex3d.github.io/](https://goatex3d.github.io/)
5. **공식 GitHub (PyTorch 구현)**: [https://github.com/KorMachine/GOATex](https://github.com/KorMachine/GOATex)
6. **OpenReview PDF**: [https://openreview.net/pdf/446ba416b0ac0c92aeb5b874ce393748157b52a1.pdf](https://openreview.net/pdf/446ba416b0ac0c92aeb5b874ce393748157b52a1.pdf)
7. **비교 연구 - SyncMVD**: Liu et al., "Text-Guided Texturing by Synchronized Multi-View Diffusion", SIGGRAPH Asia 2024 — [arxiv.org/pdf/2311.12891](https://arxiv.org/pdf/2311.12891)
8. **비교 연구 - CasTex**: [https://arxiv.org/html/2504.06856](https://arxiv.org/html/2504.06856)

> ⚠️ **최종 주의사항**: 본 답변에서 재구성된 수식들은 논문의 개념적 설명에 기반한 것으로, 정확한 원문 수식 확인을 위해서는 논문 전문(arXiv HTML 또는 NeurIPS 2025 공식 논문 PDF)을 직접 참조하시기를 권장합니다.
