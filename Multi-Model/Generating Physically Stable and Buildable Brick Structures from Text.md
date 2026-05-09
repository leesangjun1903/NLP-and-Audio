# Generating Physically Stable and Buildable Brick Structures from Text

> **저자**: Ava Pun, Kangle Deng, Ruixuan Liu, Deva Ramanan, Changliu Liu, Jun-Yan Zhu (Carnegie Mellon University)
> **발표**: ICCV 2025 (Best Paper, Marr Prize 수상)
> **arXiv**: 2505.05469v3 (2025년 11월)

---

## 1. 핵심 주장 및 주요 기여 요약

BRICKGPT는 **텍스트 프롬프트로부터 물리적으로 안정적이고 실제로 조립 가능한 LEGO® 브릭 구조를 생성하는 최초의 접근법**입니다. 핵심 통찰은 사전학습된 LLM의 "다음 토큰 예측(next-token prediction)" 능력을 "다음 브릭 예측(next-brick prediction)"으로 재해석한 것입니다.

**4가지 주요 기여**:

1. **StableText2Brick 데이터셋 구축**: 21개 카테고리, 28,000개 이상의 고유 3D 객체에 대해 47,000개 이상의 물리적으로 안정한 브릭 구조 + 캡션 페어 제공
2. **자기회귀 LLM 파인튜닝**: LLaMA-3.2-1B-Instruct를 브릭 시퀀스 생성용으로 fine-tuning
3. **물리 인식 추론(Physics-aware inference)**: 브릭별 거부 샘플링(rejection sampling)과 물리 인식 롤백(rollback)으로 안정성 보장
4. **실제 조립 검증**: 사람의 수동 조립 + 듀얼 로봇 팔을 통한 자동 조립 모두 시연

---

## 2. 상세 분석: 문제, 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

기존 3D 생성 모델은 디지털 메쉬는 잘 만들지만 **실제로 만들 수 있는 객체**를 생성하지 못합니다. 두 가지 핵심 난관이 있습니다:

- **조립 불가능성**: 표준 부품으로 만들 수 없음
- **물리적 불안정성**: 떠 있는 브릭, 충돌하는 브릭, 무너지는 구조

저자들은 두 가지 명시적 요건을 정의합니다:
- **물리적 안정성(Physically stable)**: 베이스플레이트 위에서 무너지지 않음
- **조립 가능성(Buildable)**: 표준 브릭으로 한 개씩 쌓아 조립 가능

### 2.2 브릭 표현 (Brick Representation)

브릭 구조는 다음과 같이 표현됩니다:

$$B = [b_1, b_2, \ldots, b_N], \quad b_i = [h_i, w_i, x_i, y_i, z_i]$$

여기서 $h_i, w_i$는 X/Y 방향 길이이고, $(x_i, y_i, z_i)$는 원점에 가장 가까운 스터드 위치이며 $x_i \in [0, H-1]$, $y_i \in [0, W-1]$, $z_i \in [0, D-1]$ ($H = W = D = 20$)입니다.

각 브릭은 텍스트 형태 `{h}×{w} ({x},{y},{z})`로 직렬화되어 정확히 10개의 토큰을 차지합니다.

### 2.3 자기회귀 생성 (Autoregressive Generation)

파인튜닝된 모델 $\theta$로 다음과 같이 브릭을 순차적으로 예측합니다:

$$p(b_1, b_2, \ldots, b_N \mid \theta) = \prod_{i=1}^{N} p(b_i \mid b_1, \ldots, b_{i-1}, \theta)$$

브릭은 **래스터 스캔(raster-scan) 순서**로 아래에서 위로 정렬됩니다.

### 2.4 물리 인식 안정성 분석 (Physics-Aware Stability Analysis)

각 브릭에 작용하는 힘을 모두 고려한 정적 평형 조건은:

$$\sum_{j}^{M_i} F_i^j = 0, \qquad \sum_{j}^{M_i} \tau_i^j \doteq \sum_{j}^{M_i} L_i^j \times F_i^j = 0$$

여기서 $L_i^j$는 힘 $F_i^j$에 대응하는 모멘트 암(lever)입니다. 안정성 분석은 비선형 계획법(NLP)으로 정식화됩니다:

```math
\arg\min_{\mathcal{F}} \sum_{i}^{N} \left\{ \left| \sum_{j}^{M_i} F_i^j \right| + \left| \sum_{j}^{M_i} \tau_i^j \right| + \alpha \mathcal{D}_i^{\max} + \beta \sum \mathcal{D}_i \right\}
```

세 가지 제약 조건 하에서:
- (1) 모든 후보 힘 $\geq 0$
- (2) 양립 불가능한 힘들은 동시 존재 불가 (당김 vs 누름, 끌림 vs 지지)
- (3) 뉴턴의 제3법칙

각 브릭의 안정성 점수는:

$$s_i = \begin{cases} 0 & \text{if } \sum F_i^j \neq 0 \,\lor\, \sum \tau_i^j \neq 0 \,\lor\, \mathcal{D}_i^{\max} > F_T \\ \dfrac{F_T - \mathcal{D}_i^{\max}}{F_T} & \text{otherwise} \end{cases}$$

여기서 $F_T = 0.98N$은 측정된 마찰 용량 상수입니다. NLP는 Gurobi optimizer로 해결합니다.

### 2.5 두 단계 추론 전략

저자들은 매 브릭마다 안정성 분석을 적용하면 (a) 너무 느리고 (b) 조립 중 일시적 불안정 상태를 거쳐야 하는 구조를 생성하지 못한다고 지적합니다. 따라서:

**(1) 브릭별 거부 샘플링 (Brick-by-Brick Rejection Sampling)**: 가벼운 제약(라이브러리 내 유효 브릭 + 충돌 없음)만 검사
$$V_t \cap V_i = \emptyset, \quad \forall i \in [1, t-1]$$

**(2) 물리 인식 롤백 (Physics-Aware Rollback)**: 최종 구조 완성 후 안정성 분석 → 첫 불안정 브릭의 인덱스 $\min I$ 이전까지 되돌리고 재생성

### 2.6 모델 구조 및 학습 세팅

| 항목 | 사양 |
|---|---|
| 베이스 모델 | LLaMA-3.2-1B-Instruct |
| 파인튜닝 방식 | LoRA (rank 32, alpha 16, dropout 0.05) |
| 적용 위치 | Query, Value 행렬만 (3.4M 학습 파라미터) |
| 옵티마이저 | AdamW, lr=0.002, cosine scheduler, warmup 100 |
| 배치/에포크 | global batch 64, 3 epoch |
| 하드웨어 | 8× NVIDIA RTX A6000, 12시간 |
| 추론 온도 | 0.6 (거부 시 0.01씩 증가) |
| 최대 롤백 횟수 | 100회 (중앙값 2회) |
| 평균 생성 시간 | 약 40.8초/구조 |

### 2.7 정량 성능 (250개 검증 프롬프트, Table 1 발췌)

| Method | % valid | % stable | mean stab. | min stab. | CLIP | DINO |
|---|---|---|---|---|---|---|
| Pre-trained LLaMA (0-shot) | 0.0% | 0.0% | N/A | N/A | N/A | N/A |
| In-context learning (5-shot) | 2.4% | 1.2% | 0.675 | 0.479 | 0.284 | 0.814 |
| LLaMA-Mesh + mesh→brick | 94.8% | 50.8% | 0.894 | 0.499 | 0.317 | 0.851 |
| LGM + mesh→brick | 100% | 25.2% | 0.942 | 0.231 | 0.300 | 0.851 |
| XCube + mesh→brick | 100% | 75.2% | 0.964 | 0.686 | 0.322 | 0.859 |
| Hunyuan3D-2 + mesh→brick | 100% | 75.2% | 0.973 | 0.704 | **0.324** | **0.868** |
| **Ours (BRICKGPT)** | **100%** | **98.8%** | **0.996** | **0.915** | **0.324** | 0.880 |

가장 강력한 베이스라인 Hunyuan3D-2 대비 안정성 비율이 75.2% → **98.8%**로 향상되었으며, ablation에서 거부 샘플링과 롤백을 모두 제거하면 안정성이 12.8%로 떨어져 두 메커니즘의 필수성이 입증되었습니다.

### 2.8 한계 (저자 명시)

1. **해상도/도메인 제약**: $20 \times 20 \times 20$ 그리드, 21개 카테고리에 한정 → 최신 text-to-3D보다 표현 다양성이 좁음
2. **고정된 브릭 라이브러리**: 8가지 표준 브릭만 사용 (1×1, 1×2, 1×4, 1×6, 1×8, 2×2, 2×4, 2×6) — 슬로프, 타일 등 미지원
3. **OOD 텍스트 일반화 부족**: ShapeNetCore 기반 학습으로, 분포 외 프롬프트에 약함

---

## 3. 모델의 일반화 성능 향상 가능성

논문 자체는 일반화 한계를 명확히 인정하지만, 구조적으로 **일반화 향상 잠재력이 큰 설계**를 갖고 있습니다.

### 3.1 긍정적 요소

**(a) Novelty 분석 결과**: 부록 Figure 9에서 Chamfer distance 기반 nearest-neighbor 분석을 통해 생성 결과가 단순 암기가 아닌 새로운 조합임을 보여, **합성적 일반화(compositional generalization)** 가능성을 시사합니다.

**(b) LLM 기반 백본의 사전지식**: LLaMA-3.2의 자연어 이해 능력을 활용하여 "Gothic cathedral bookshelf", "Cyberpunk holographic" 등 학습 데이터에 없는 스타일 키워드도 텍스처/색상 모듈로 수용 가능합니다 (Figure 7).

**(c) 모듈화된 안정성 검증**: 안정성 분석 모듈은 모델 백본과 분리된 외부 검증기로 작동하므로, 더 큰 LLM이나 다른 도메인에 그대로 결합 가능합니다.

### 3.2 일반화를 막는 병목

| 병목 | 원인 | 해결 방향 |
|---|---|---|
| 그리드 해상도 한계 | $20^3$ 고정, 토큰 길이 4096 제약 | 계층적 토큰화 / 적응적 토크나이저 |
| 카테고리 편향 | ShapeNetCore 21개 클래스 | Objaverse-XL [Deitke et al. 2023] 등 대규모 다중 도메인 데이터 |
| 브릭 어휘 제한 | 8종 표준 브릭만 학습 | 슬로프/타일/특수 브릭 포함 어휘 확장 |
| 작은 백본 | 1B 파라미터로 효율성 우선 | 7B/13B 백본으로 스케일링 |

### 3.3 일반화 향상 로드맵 (저자 + 분석가 의견)

1. **Scaling laws 적용**: MeshLLM이 LLaMA-Mesh 대비 50배 큰 데이터셋으로 성능 향상을 보인 것처럼, BRICKGPT도 데이터 규모 증가로 OOD 성능 개선 여지가 큼 (출처: MeshLLM, arXiv 2508.01242)
2. **계층적 표현**: XCube의 sparse voxel hierarchy 아이디어를 도입하면 더 큰 그리드 표현 가능
3. **도메인 적응 미세조정**: 건축, 캐릭터 등 특정 도메인 데이터로 점진적 fine-tuning

---

## 4. 향후 연구에의 영향 및 고려사항

### 4.1 학술·산업적 영향

**(a) 새로운 연구 패러다임**: "디지털 3D → 실제 조립 가능 3D"로 연구 축을 이동시켰습니다. 이는 **physics-aware generative AI**의 대표 사례로, 로봇 공학·제조업·교육 분야에 응용 가능합니다.

**(b) Generation + Verification 분리 설계**: LLM이 후보를 빠르게 생성하고, 외부 물리 검증기가 거부/롤백하는 구조는 **신뢰 가능한 AI(trustworthy AI)** 일반 프레임워크로 확장 가능 — 단백질 설계, 회로 합성, 기계 설계 등에 직접 응용 가능합니다.

**(c) 로봇 자동 조립과의 통합**: APEX-MR (Huang et al. 2025)와 결합하여 텍스트 → 디자인 → 자동 조립의 end-to-end 파이프라인 시연. 이는 산업 자동화·맞춤형 제조의 청사진을 제공합니다.

### 4.2 향후 연구 시 고려사항

1. **물리 검증의 확장성**: NLP 풀이가 평균 0.35초/구조이지만 200+ 브릭을 넘어가면 비용이 커집니다. **미분 가능한 물리 시뮬레이션(differentiable physics)**이나 신경망 기반 stability surrogate가 필요할 수 있습니다.

2. **임시 불안정 상태 허용**: 현재 방법은 최종 구조만 안정하면 OK이지만, 실제 조립 중간 단계도 안정해야 하는 경우(특히 인간 조립자에게 친화적) 시퀀스 자체에 안정성 제약을 걸어야 합니다.

3. **대규모 LLM과의 결합 시 LoRA 한계**: 1B 모델에 LoRA(3.4M params)는 효과적이나, 더 큰 백본·다양한 도메인에서는 full-parameter fine-tuning 또는 mixture-of-experts가 필요할 수 있습니다.

4. **평가 지표 부족**: CLIP/DINO 기반 평가는 시각적 정렬은 측정하지만 **"실제로 잘 만들 수 있는지"의 인간/로봇 평가**는 표준화되어 있지 않습니다.

5. **저작권·라이선스**: LEGO® 등 상표권을 가진 표준 브릭과의 호환성을 학술 연구가 어디까지 명시할 수 있는지에 대한 가이드라인이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 자기회귀 3D 생성 계열 (LLM-based)

| 모델 (연도) | 출력 형식 | 백본 | 물리 제약 | BRICKGPT 대비 차이점 |
|---|---|---|---|---|
| **PolyGen** (Nash et al. 2020) | 메쉬 (vertex+face) | Pointer Network/Transformer | ❌ | 메쉬 토큰화 선구작, LLM 재활용 X |
| **MeshGPT** (Siddiqui et al. 2024) | 삼각 메쉬 (~800 face) | VQ-VAE + Transformer | ❌ | 디지털 메쉬, 물리 없음 |
| **MeshXL** (Chen et al. 2024) | 좌표장 메쉬 | Decoder-only Transformer | ❌ | VQ-VAE 없는 토큰화 |
| **MeshAnything v2** (Chen et al. 2024) | 메쉬 (~1.6k face) | Autoregressive Transformer | ❌ | 압축 토큰화 |
| **EdgeRunner** (Tang et al. 2024) | 메쉬 (~4k face) | Autoencoder + AR | ❌ | 엣지 기반 토큰화 |
| **Meshtron** (Hao et al. 2024) | 메쉬 (~16k face) | Hourglass Transformer | ❌ | 큰 메쉬 생성 가능 |
| **LLaMA-Mesh** (Wang et al. 2024) | OBJ 텍스트 메쉬 | LLaMA-3 | ❌ | 사전학습 LLM 활용, 물리 없음 |
| **MeshLLM** (Fang et al. 2025) | 텍스트 메쉬 | LLaMA + primitive decomposition | ❌ | 1500k 샘플로 스케일링 |
| **BRICKGPT** (Pun et al. 2025) | **이산 브릭 시퀀스** | LLaMA-3.2-1B | ✅ **명시적 정역학 제약** | **물리 + 조립 가능성 보장 최초** |

(출처: MeshLLM 논문 arXiv 2508.01242; "Auto-Regressive Mesh Generation as Weaving Silk" arXiv 2507.02477; LLaMA-Mesh 페이지 NVIDIA Toronto AI Lab)

### 5.2 Diffusion 기반 Text-to-3D 계열

- **DreamFusion** (Poole et al. 2023): SDS(Score Distillation Sampling)로 NeRF 최적화. 물리 없음.
- **LGM** (Tang et al. 2024): Large Multi-view Gaussian Model, 빠르지만 BRICKGPT 비교 실험에서 mesh→brick 변환 시 안정성 25.2%에 그침.
- **XCube** (Ren et al. 2024): Sparse voxel hierarchy 기반 대규모 3D 디퓨전. mesh→brick 변환 후 안정성 75.2%.
- **Hunyuan3D-2** (Zhao et al. 2025): 고해상도 텍스처 메쉬 디퓨전. 안정성 75.2%지만 BRICKGPT(98.8%) 대비 여전히 낮음.

### 5.3 LEGO/브릭 특화 기존 연구

- **Image2Lego** (Lennon et al. 2021): 이미지 → 복셀 → 브릭, 물리 제약 미고려
- **Legolization** (Luo et al. 2015): 입력 3D 모델을 brick layout으로 최적화, 물리 분석 포함하나 텍스트 입력 X
- **LEGO 마이크로 빌딩** (Ge et al. 2024 TOG): 디퓨전 기반 의미 볼륨 생성, 단일 카테고리 한정
- **Blox-Net** (Goldberg et al. 2025 ICRA): VLM + 시뮬레이션, 그러나 일반 building block(연결부 없음) 사용
- **StableLego** (Liu et al. 2024 RAL): 본 논문이 사용하는 안정성 분석의 기반 데이터셋

### 5.4 Physics-Aware Generation 계열

- **PhyRecon** (Ni et al. 2024 NeurIPS): 물리적으로 그럴듯한 신경 장면 복원
- **PhyScene** (Yang et al. 2024 CVPR): 물리적 상호작용 가능한 3D 장면 합성
- **Physically Compatible 3D Object Modeling** (Guo et al. 2024): 단일 이미지에서 물리 호환 모델
- **GPLD3D** (Dong et al. 2024 CVPR): 기하·물리 prior로 잠재 디퓨전 강제

### 5.5 비교 종합

BRICKGPT의 차별점은 **(i) 자기회귀 LLM의 reasoning 능력 + (ii) 명시적 비선형 정역학 분석 + (iii) 이산적이고 조립 가능한 출력 공간 + (iv) 실제 로봇 조립 검증**을 단일 파이프라인으로 통합한 점입니다. 메쉬 생성 계열은 (i)는 있으나 (ii)~(iv)가 없고, 물리 인식 생성 계열은 (ii)는 있으나 이산 조립 출력(iii)·실제 검증(iv)이 부족합니다.

---

## 📚 참고 자료 (출처)

1. **원논문**: Pun, A., Deng, K., Liu, R., Ramanan, D., Liu, C., Zhu, J.-Y. *Generating Physically Stable and Buildable Brick Structures from Text*. ICCV 2025 (Best Paper, Marr Prize). arXiv:2505.05469v3.
   - [arXiv 페이지](https://arxiv.org/pdf/2505.05469)
   - [ICCV 2025 Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Pun_Generating_Physically_Stable_and_Buildable_Brick_Structures_from_Text_ICCV_2025_paper.html)
   - [공식 프로젝트 페이지](https://avalovelace1.github.io/BrickGPT/)
   - [GitHub 저장소](https://github.com/AvaLovelace1/BrickGPT)
   - [Hugging Face 모델](https://huggingface.co/AvaLovelace/BrickGPT)
   - [alphaXiv 논의 페이지](https://www.alphaxiv.org/overview/2505.05469v3)

2. **비교 연구 출처**:
   - LLaMA-Mesh: Wang et al. (2024), arXiv:2411.09595, [NVIDIA Toronto AI Lab 페이지](https://research.nvidia.com/labs/toronto-ai/LLaMA-Mesh/), [GitHub](https://github.com/nv-tlabs/LLaMA-Mesh)
   - MeshLLM: Fang et al. (2025), arXiv:2508.01242, [arXiv 페이지](https://arxiv.org/abs/2508.01242)
   - "Auto-Regressive Mesh Generation as Weaving Silk", arXiv:2507.02477
   - StableLego: Liu et al. (2024), IEEE RAL — 안정성 분석 기반
   - APEX-MR (로봇 조립): Huang et al. (2025), Robotics: Science and Systems

---

### ⚠️ 정확도에 대한 알림

본 분석에서 **원논문에 명시된 수식, 수치, 표, 알고리즘**(섹션 2)은 업로드된 PDF에서 직접 추출한 100% 정확한 정보입니다. 그러나 **5.1~5.5 비교 분석의 일부 모델 사양**(예: 정확한 face 수치, 데이터셋 크기)은 원논문의 references 섹션과 별도 검색 결과를 종합한 것이며, 일부 세부 수치는 해당 원논문을 직접 확인하셔야 100% 정확합니다. **3.3절의 "로드맵"과 4.2절의 "고려사항"은 BRICKGPT 저자가 명시한 한계와 일반적인 LLM/3D 생성 분야의 동향에 기반한 분석가 의견이며, 원논문이 직접 주장한 내용은 아닙니다.**
