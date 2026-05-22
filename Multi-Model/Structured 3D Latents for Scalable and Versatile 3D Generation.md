
# TRELLIS: Structured 3D Latents for Scalable and Versatile 3D Generation

> **논문 정보**
> - **제목**: Structured 3D Latents for Scalable and Versatile 3D Generation
> - **저자**: Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, Jiaolong Yang (Tsinghua Univ. / USTC / Microsoft Research)
> - **게재**: CVPR 2025 Spotlight
> - **arXiv**: [2412.01506](https://arxiv.org/abs/2412.01506)
> - **GitHub**: [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)
> - **Project Page**: [microsoft.github.io/TRELLIS](https://microsoft.github.io/TRELLIS/)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

이 논문은 다목적 고품질 3D 에셋 생성을 위한 새로운 방법을 제안한다. 핵심은 Radiance Fields, 3D Gaussians, 메시 등 다양한 출력 형식으로 디코딩 가능한 통합 **Structured LATent (SLAT)** 표현이다. 이는 희소하게 채워진(sparsely-populated) 3D 그리드에 강력한 비전 파운데이션 모델로 추출된 밀집 멀티뷰 시각 특징을 통합함으로써 구조(기하학)와 텍스처(외관) 정보를 포괄적으로 캡처하면서도 디코딩 유연성을 유지한다.

### 🏆 주요 기여 (5가지)

| 기여 | 내용 |
|------|------|
| ① 통합 잠재 표현 (SLAT) | 단일 표현으로 다양한 3D 형식 디코딩 가능 |
| ② 2단계 Rectified Flow 생성 파이프라인 | Sparse 구조 → Local latent 순차 생성 |
| ③ 대규모 학습 | 50만 개 3D 에셋, 최대 20억 파라미터 |
| ④ 다중 조건 지원 | 텍스트/이미지 입력 모두 지원 |
| ⑤ 로컬 3D 편집 | 이전 모델이 제공하지 못한 지역적 편집 기능 |

TRELLIS의 핵심은 다양한 출력 포맷으로의 디코딩을 가능하게 하는 통합 SLAT 표현과 SLAT에 맞춘 Rectified Flow Transformer이며, 최대 20억 개의 파라미터를 가진 대규모 사전 학습 모델을 50만 개의 다양한 3D 에셋 데이터셋에서 제공한다. TRELLIS는 기존 방법(유사 규모의 최신 방법 포함)을 크게 능가하며, 이전 모델들이 제공하지 않았던 유연한 출력 형식 선택과 로컬 3D 편집 기능을 선보인다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

서로 다른 3D 표현들의 구조적/비구조적 특성 차이가 일관된 네트워크 아키텍처를 통한 처리를 복잡하게 만든다. 이 문제는 통합 잠재 공간 내에서 생성 모델을 학습하는 최신 2D 생성 방법의 합의와 달리 표준화된 3D 생성 모델링 패러다임의 발전을 저해한다.

구체적으로 기존 문제들은 다음과 같다:

- **표현 파편화**: NeRF, 3D Gaussian, Mesh, Point Cloud 등 각 3D 표현마다 별도의 생성 모델이 필요
- **통일된 잠재 공간 부재**: 2D 생성처럼 하나의 통합된 잠재 공간에서 학습하는 방법이 없음
- **고해상도 3D 모델링의 비효율성**: Dense voxel grid는 메모리 폭발 문제 발생

이 논문은 다양한 다운스트림 요구사항을 수용하여 고품질 3D 생성을 다양한 표현에 걸쳐 용이하게 하는 통합적이고 다목적인 잠재 공간을 개발하는 것을 목표로 한다. 이 문제는 매우 도전적이며 이전 접근법들에 의해 거의 다루어지지 않았다. 이를 해결하기 위한 주요 전략은 잠재 공간 설계에 명시적 희소 3D 구조를 도입하는 것이다.

---

### 2-2. 제안하는 방법

#### ① SLAT (Structured LATent) 표현

SLAT은 고품질의 다목적 3D 생성을 위한 통합 3D 잠재 표현이다. SLAT은 희소 구조와 강력한 시각적 표현을 결합한다. 오브젝트 표면과 교차하는 활성 복셀(active voxels) 위에 로컬 잠재 변수를 정의한다. 로컬 잠재 변수는 3D 에셋의 densely rendered view로부터 이미지 특징을 융합·처리하여 인코딩되고 활성 복셀에 부착된다. 이 특징들은 강력한 사전 학습된 비전 인코더에서 추출되어 상세한 기하학적·시각적 특성을 캡처하며 활성 복셀이 제공하는 거친 구조를 보완한다. 이후 서로 다른 디코더를 적용하여 SLAT을 다양한 고품질 3D 표현으로 매핑할 수 있다.

SLAT의 수학적 정의를 구체화하면 다음과 같다:

**SLAT 인코딩:**

$$\mathcal{S} = \{(\mathbf{v}_i, \mathbf{z}_i)\}_{i=1}^{N}$$

- $\mathbf{v}_i \in \mathbb{Z}^3$: $i$번째 활성 복셀의 3D 그리드 좌표
- $\mathbf{z}_i \in \mathbb{R}^d$: 해당 복셀에 부착된 로컬 잠재 벡터 (DINOv2로 추출한 멀티뷰 특징 융합)
- $N$: 활성 복셀 수 (객체 표면과 교차하는 복셀만 포함)

**멀티뷰 특징 인코딩 (VAE):**

$$\mathbf{z}_i = \text{VAE}_{\text{enc}}\left(\text{Fuse}\left(\{f_k(\mathbf{v}_i)\}_{k=1}^{K}\right)\right)$$

- $f_k(\mathbf{v}_i)$: $k$번째 뷰에서 복셀 $\mathbf{v}_i$에 대응하는 DINOv2 특징
- $K$: 렌더링 뷰 수
- $\text{Fuse}(\cdot)$: Transformer 기반 특징 융합

SLAT은 기하학 및 외관 정보를 표현하기 위해 희소 3D 그리드에 로컬 잠재 변수를 정의하는 구조화된 잠재 표현을 채택한다. DINOv2 인코더에서 추출된 밀집 멀티뷰 시각 특징을 융합·처리하여 인코딩되며, 서로 다른 디코더를 통해 다양한 출력 표현으로 디코딩될 수 있다.

#### ② 2단계 생성 파이프라인 (Rectified Flow Transformer)

먼저 SLAT의 희소 구조를 생성한 후, 비어 있지 않은 셀들의 잠재 벡터를 생성하는 2단계 파이프라인이 적용된다. Rectified Flow Transformer를 백본 모델로 사용하며 SLAT의 희소성을 처리하도록 적절히 적응시킨다.

**[Stage 1] 희소 구조 생성:**

$$p_\theta(\mathbf{V}) = \text{SparseRFT}(\mathbf{V} | c)$$

- Sparse Rectified Flow Transformer가 조건 $c$ (텍스트 또는 이미지)에 따라 활성 복셀 집합 $\mathbf{V} = \{\mathbf{v}_i\}$ 생성

**[Stage 2] 로컬 잠재 벡터 생성:**

$$p_\phi(\mathbf{Z} | \mathbf{V}) = \text{LocalRFT}(\mathbf{Z} | \mathbf{V}, c)$$

- 주어진 희소 구조 $\mathbf{V}$ 조건으로 각 활성 복셀의 로컬 잠재 벡터 $\mathbf{Z} = \{\mathbf{z}_i\}$ 생성

**Rectified Flow 학습 목표:**

Rectified Flow 모델은 디퓨전의 지배에 도전하는 새로운 생성 패러다임으로 최근 부상했다. 최근 연구들은 대규모 이미지 및 비디오 생성에서의 효과를 입증했다. 이 논문에서도 Rectified Flow 모델을 적용하여 대규모 3D 생성에서의 능력을 입증한다.

Rectified Flow의 학습 목표 수식:

$$\mathcal{L}_{\text{RF}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[ \left\| v_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \right\|^2 \right]$$

$$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1, \quad t \in [0, 1]$$

- $\mathbf{x}_0$: 노이즈 (표준 정규분포)
- $\mathbf{x}_1$: 실제 데이터 (SLAT)
- $v_\theta$: 속도 벡터 필드를 예측하는 Transformer 네트워크

Rectified Flow Transformer는 노이즈 입력을 데이터 분포 쪽으로 이끄는 연속 흐름 필드를 사용하는 새로운 생성 접근법이다. 디퓨전 모델에 비해 더 직접적이고 더 적은 샘플링 스텝을 필요로 하는 경우가 많다.

#### ③ 다중 형식 디코더

SLAT $\mathcal{S}$로부터 3가지 디코더를 통해 다양한 출력 형식 생성:

$$\mathbf{O}_{\text{NeRF}} = \text{Dec}_{\text{NeRF}}(\mathcal{S}), \quad \mathbf{O}_{\text{3DGS}} = \text{Dec}_{\text{3DGS}}(\mathcal{S}), \quad \mathbf{O}_{\text{mesh}} = \text{Dec}_{\text{mesh}}(\mathcal{S})$$

---

### 2-3. 모델 구조

```
입력 (텍스트 / 이미지)
        │
        ▼
┌─────────────────────────────────┐
│    Stage 1: Sparse Structure    │
│   Sparse Rectified Flow Xformer │
│   → Active Voxel Set {v_i}      │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│    Stage 2: Local Latents       │
│  Local Sparse Rectified Flow    │
│  Xformer (SparseDiT)            │
│   → Local Latents {z_i}         │
└─────────────────────────────────┘
        │ SLAT = {(v_i, z_i)}
        ▼
┌──────────────────────────────────────────────┐
│             Multi-Format Decoders            │
│   NeRF Decoder │ 3DGS Decoder │ Mesh Decoder │
└──────────────────────────────────────────────┘
```

SLAT의 효과적인 인코딩 방식으로 DINOv2 인코더와 Transformer 기반 VAE를 사용하여 밀집 멀티뷰 이미지에서 특징을 집계한다. SLAT은 전용 디코더를 통해 3D Gaussians, Radiance Fields, 메시와 같은 다양한 3D 표현으로의 다목적 디코딩을 지원하며, 높은 충실도와 확장성을 제공한다.

---

### 2-4. 성능 향상

SLAT 표현은 외관 및 기하학 지표 모두에서 대안적 잠재 표현들에 비해 재구성 충실도가 크게 뛰어나다. 외관 측면에서 SLAT은 PSNR 32.74, LPIPS 0.025를 달성한다.

모델이 상세 기하학과 생생한 텍스처를 갖는 고품질 3D 에셋을 생성할 수 있으며, 기존 방법들을 크게 능가함을 광범위한 실험을 통해 입증한다. 또한 다양한 다운스트림 요구사항을 충족하는 서로 다른 출력 형식의 3D 에셋을 쉽게 생성할 수 있다.

**주요 성능 지표 요약:**

| 지표 | TRELLIS | 기존 SOTA 대비 |
|------|---------|------------|
| PSNR (외관) | 32.74 | 대폭 향상 |
| LPIPS (외관) | 0.025 | 대폭 향상 |
| 파라미터 수 | 최대 2B | 동급 규모 모델 중 최고 성능 |
| 학습 데이터 | 50만 3D 에셋 | - |
| 생성 형식 | NeRF / 3DGS / Mesh | 다형식 지원 최초 |

---

### 2-5. 한계점

논문 및 관련 분석에 근거한 한계:

1. **데이터 의존성**: 고품질 3D 데이터의 한계로 인해 텍스트 및 이미지 생성의 기술을 3D 작업으로 직접 전환하는 것이 오랫동안 도전적이었다. TRELLIS도 여전히 고품질 3D 데이터셋 확보에 의존한다.

2. **동적/장면 수준 생성 미지원**: 현재는 단일 오브젝트 중심 에셋 생성에 초점이 맞춰져 있으며, 실내/외 전체 장면(scene-level) 생성으로의 확장은 미개척 영역이다.

3. **2단계 파이프라인의 오류 전파**: Stage 1에서의 희소 구조 오류가 Stage 2 로컬 잠재 생성에 전파될 수 있다.

4. **추론 비용**: 20억 파라미터 모델은 실시간 응용에 높은 계산 비용을 요구한다.

5. **복잡한 위상 구조**: 최근 3D 생성 모델링의 발전이 생성 현실성을 크게 향상시켰음에도 불구하고, 복잡한 위상 구조와 세밀한 외관을 가진 에셋 캡처에 기존 표현들이 어려움을 겪고 있다.

---

## 3. 일반화 성능 향상 가능성

### 3-1. SLAT의 구조적 일반화 강점

주요 전략은 잠재 공간 설계에 명시적 희소 3D 구조를 도입하는 것이다. 이 구조들은 오브젝트의 로컬 복셀 내 속성을 특성화함으로써 다양한 3D 표현으로의 디코딩을 가능하게 한다. 또한 이 접근법은 3D 정보가 없는 복셀을 우회하여 효율적인 고해상도 모델링을 가능하게 하고, 유연한 편집을 용이하게 하는 지역성(locality)을 도입한다.

### 3-2. 파운데이션 모델 활용을 통한 일반화

로컬 잠재 변수는 3D 에셋의 밀집 렌더링 뷰에서 이미지 특징을 융합·처리하면서 활성 복셀에 부착된다. 강력한 사전 학습된 비전 인코더(DINOv2)에서 파생된 이 특징들은 상세한 기하학적·시각적 특성을 캡처하여 활성 복셀이 제공하는 거친 구조를 보완한다.

**DINOv2와 같은 비전 파운데이션 모델의 활용은 다음 측면에서 일반화를 향상:**

$$\mathbf{z}_i = \text{Transformer-VAE}\left(\text{DINOv2}(I_1(\mathbf{v}_i)), \ldots, \text{DINOv2}(I_K(\mathbf{v}_i))\right)$$

- DINOv2의 **대규모 사전학습 표현력** → 미보유 카테고리(unseen categories)에도 강건한 특징 추출
- **멀티뷰 일관성** 내재화 → 단일 뷰만 입력해도 3D 일관성 유지

### 3-3. 대규모 데이터와 모델 스케일링

최대 20억 개의 파라미터를 가진 모델을 50만 개의 다양한 오브젝트를 포함한 대규모 3D 에셋 데이터셋에서 학습한다. 텍스트 또는 이미지 조건으로 고품질 결과를 생성하며, 유사 규모의 최신 방법들을 포함한 기존 방법들을 크게 능가한다.

### 3-4. 일반화 향상 메커니즘 정리

| 메커니즘 | 일반화 기여 방식 |
|----------|---------------|
| DINOv2 특징 활용 | 대규모 비전 사전학습의 범용 표현력 전이 |
| 희소 구조 + 로컬 잠재 분리 | 구조·텍스처 독립적 학습으로 미보유 형태에 유연 대응 |
| 다형식 디코더 | 다운스트림 task 다양성에 단일 모델로 대응 |
| 50만 다양 오브젝트 학습 | 카테고리 분포 커버리지 확대 |
| Rectified Flow | 디퓨전 대비 빠른 수렴, 더 안정적인 학습 |

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4-1. 주요 3D 생성 연구 계보

```
2020: NeRF (Mildenhall et al.)
  └→ 2022: DreamFusion (Poole et al.) - SDS Loss + NeRF
       └→ 2023: Zero-1-to-3 - 뷰 조건부 이미지 확산
            └→ 2023: One-2-3-45 - 45초 내 3D Mesh 생성
                 └→ 2024: 3D Gaussian Splatting 기반 생성
                      └→ 2024: TRELLIS - 통합 SLAT + Rectified Flow
                           └→ 2025: TRELLIS.2 - Native & Compact SLAT
```

### 4-2. 상세 비교표

| 논문 | 연도 | 표현 | 생성 방식 | 다형식 지원 | 일반화 | 한계 |
|------|------|------|----------|-----------|--------|------|
| **DreamFusion** | 2022 | NeRF | SDS Loss | ❌ | 중 | 느린 최적화 |
| **Zero-1-to-3** | 2023 | NeRF/Mesh | View-cond. Diffusion | ❌ | 중 | 뷰 불일관성 |
| **One-2-3-45** | 2023 | SDF/Mesh | Feed-forward | ❌ | 중 | 미보유 카테고리 취약 |
| **Shape-E** | 2023 | NeRF/Mesh | Diffusion | △ | 중 | 저해상도 |
| **3D Gaussian** 기반 | 2024 | 3DGS | Diffusion | ❌ | 중상 | 단일 표현 |
| **TripoSG** | 2025 | SDF | Rectified Flow | ❌ | 상 | 단일 표현 |
| **TRELLIS** | 2024 | SLAT | Rectified Flow | ✅ | **최상** | 오브젝트 한정 |
| **TRELLIS.2** | 2025 | O-Voxel | Rectified Flow | ✅ | **최상** | 개발 중 |

3D 생성은 DreamFusion과 같은 SDS 솔루션을 통해 텍스트-투-이미지 모델을 3D 생성의 사전으로 사용하거나, LRM과 같은 단일/다중 뷰에서 3D 모델을 재구성하는 디코더 전용 트랜스포머 아키텍처를 활용하는 독특한 탐색 경로를 따라왔다. 확산 모델은 이미지 또는 비디오 생성에서 강력한 생성 능력을 입증했다. 그러나 고품질 3D 데이터의 한계로 인해 텍스트 및 이미지 생성 기술을 3D 작업으로 직접 전환하는 것이 오랫동안 도전적이었다. DreamFusion은 미분 가능한 볼륨 렌더링을 통해 3D 표현의 반복적 최적화를 가능하게 하는 Score Distillation Sampling 방법을 제안하여 이미지 확산 사전을 3D 생성에 사용하는 것을 개척했다.

많은 기존 방법들은 2D 확산 모델의 지도하에 신경 방사 필드(NeRF)를 최적화하는 방식으로 이 문제를 해결하지만 긴 최적화 시간, 3D 불일관 결과, 불량한 기하학으로 고생한다. One-2-3-45는 단일 입력 이미지를 받아 단일 순전파(feed-forward) 패스로 완전한 360도 텍스처 메시를 생성하는 새로운 방법을 제안한다.

확산 기반 방법들은 3D 표현 또는 VAE로 압축된 잠재 표현에서 학습한다. 생성적 방법으로서 재구성 접근법의 고유한 도전들을 우회한다. 그러나 현재 방법들은 주로 점유(occupancy) 표현에 의존하여 앨리어싱 아티팩트를 줄이기 위한 추가 후처리가 필요하고 세밀한 기하학적 디테일이 부족한 경우가 많다.

### 4-3. 후속 연구 TRELLIS.2

O-Voxel 기반의 Sparse Compression VAE를 설계하여 높은 공간 압축률과 컴팩트한 잠재 공간을 제공한다. 다양한 공개 3D 에셋 데이터셋을 활용하여 40억 개 파라미터 규모의 대규모 플로우 매칭 모델을 학습한다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려 사항

### 5-1. 미래 연구에 미치는 영향

#### 🌟 패러다임 전환
TRELLIS는 3D 생성에서 **"표현 독립적 통합 잠재 공간"** 패러다임을 확립했다. 이는 2D 생성에서 Stable Diffusion 등이 확립한 통합 잠재 공간 방식과 동일한 역할을 3D에서 수행할 것으로 기대된다.

#### 🔬 영향을 미칠 연구 방향

1. **3D 파운데이션 모델로의 발전**: SLAT을 기반으로 한 대규모 3D 사전학습 모델 연구 촉진
2. **3D 편집 연구 활성화**: 로컬 잠재 구조의 locality 특성이 정밀 편집 연구 새 지평 개척
3. **장면(Scene) 수준 생성**: 오브젝트 수준 성공을 발판으로 실내/외 장면 생성으로 확장
4. **실시간 3D 생성**: Rectified Flow의 빠른 샘플링 특성 → 실시간 응용 연구 동력 제공
5. **멀티모달 3D 생성**: 텍스트+이미지+3D 입력 통합 방향으로의 연구 확장

### 5-2. 앞으로의 연구 시 고려사항

#### 📌 기술적 고려사항

| 고려사항 | 세부 내용 |
|---------|---------|
| **데이터 품질 vs 양** | 50만 개 에셋의 품질 필터링이 성능에 결정적 → 데이터 큐레이션 전략 연구 필요 |
| **희소 구조 해상도** | 활성 복셀의 해상도가 세밀도 결정 → 적응형 해상도 메커니즘 탐구 필요 |
| **2단계 오류 전파** | Stage 1→2 오류 전파 완화를 위한 Joint Training 방법 연구 |
| **추론 효율화** | 2B 파라미터 모델의 경량화, 양자화(Quantization), 증류(Distillation) 연구 |
| **동적 객체 대응** | 정적 오브젝트 중심에서 동적/변형 가능 오브젝트로의 확장 연구 |

#### 📌 일반화 관련 고려사항

$$\text{Generalization Gap} = \mathcal{L}_{\text{test}} - \mathcal{L}_{\text{train}}$$

- **도메인 갭 해소**: 합성 3D 데이터(Objaverse 등)로 학습 → 실제 스캔 데이터 테스트 시 갭 존재 → 도메인 적응(Domain Adaptation) 전략 필요
- **카테고리 외삽(Extrapolation)**: 학습 카테고리 외의 새로운 형태/질감에 대한 일반화 측정 메트릭 개발 필요
- **물리 기반 표현(PBR)**: 현재 외관 중심 → PBR 재질(알베도, 러프니스, 메탈릭) 정보로의 확장 → TRELLIS.2는 사용자 정의 데이터셋으로의 파인튜닝을 위한 전체 학습 코드베이스를 제공하며, 학습 전 원시 3D 에셋은 O-Voxel 표현으로 변환되어야 하며, 이 과정에는 메시 변환, 컴팩트 구조화된 잠재 생성, 메타데이터 준비가 포함된다.

#### 📌 응용 연구 고려사항

1. **게임/영화 산업**: 실시간 PBR 메시 생성 파이프라인 통합 연구
2. **로보틱스**: 생성된 3D 에셋의 물리 시뮬레이션 호환성 연구
3. **의료 영상**: SLAT의 sparse 특성 → 의료 3D 볼륨 데이터 생성 응용 연구

---

## 📚 참고 자료 출처

| # | 제목 | 출처 |
|---|------|------|
| 1 | **Structured 3D Latents for Scalable and Versatile 3D Generation** | arXiv:2412.01506 (https://arxiv.org/abs/2412.01506) |
| 2 | **TRELLIS Project Page** | https://microsoft.github.io/TRELLIS/ |
| 3 | **TRELLIS GitHub Repository (CVPR'25 Spotlight)** | https://github.com/microsoft/TRELLIS |
| 4 | **Microsoft Research Publication** | https://www.microsoft.com/en-us/research/publication/structured-3d-latents-for-scalable-and-versatile-3d-generation/ |
| 5 | **Hugging Face Paper Page** | https://huggingface.co/papers/2412.01506 |
| 6 | **Understanding TRELLIS (Digiwave Blog)** | https://dgwave.net/blog/understanding-microsoft-trellis-3d-ai-model |
| 7 | **[Quick Review] TRELLIS (Liner)** | https://liner.com/review/structured-3d-latents-for-scalable-and-versatile-3d-generation |
| 8 | **TRELLIS.2 – Native and Compact Structured Latents** | arXiv:2512.14692 / https://github.com/microsoft/TRELLIS.2 |
| 9 | **TripoSG: High-Fidelity 3D Shape Synthesis** | arXiv:2502.06608 |
| 10 | **One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds** | arXiv:2306.16928 |
| 11 | **A Comprehensive Survey on 3D Content Generation** | arXiv:2402.01166 |
| 12 | **DreamFusion: Text-to-3D using 2D Diffusion** | arXiv:2209.14988 (Poole et al., 2022) |

> ⚠️ **정확도 주의**: 본 답변의 수식 일부(특히 상세 내부 아키텍처 수식)는 공개된 논문 본문에서 확인 가능한 범위 내에서 정리하였으며, 논문에 명시되지 않은 세부 수식은 논문의 방법론 서술에 기반한 구조적 해석임을 밝힙니다. 정확한 수식은 arXiv PDF 원문을 직접 참조하시기 바랍니다.
