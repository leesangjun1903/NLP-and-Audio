# Track Anything: Segment Anything Meets Videos

### 1. 핵심 주장과 주요 기여

Track Anything Model (TAM)은 **기초 모델(Foundation Model)의 강점을 결합한 상호작용형 비디오 추적 및 분할**에 관한 연구이다. 논문의 핵심 주장은 Segment Anything Model (SAM)의 우수한 이미지 분할 능력이 시간적 일관성(Temporal Correspondence) 부재로 인해 비디오에서 제대로 작동하지 않는다는 관찰에서 출발한다.[1]

**TAM의 주요 기여는 다음과 같다:**[1]

- **SAM의 비디오 수준 확장**: SAM을 프레임 단위로 적용하는 대신, 시간적 일관성 구축 과정에 통합하여 비디오 객체 추적 및 분할 달성
- **일회 통과 상호작용형 추적**: 최소한의 인간 참여(클릭 기반)로 효율적인 주석 및 사용자 친화적 추적 인터페이스 제공
- **복잡한 장면에서의 우수한 성능**: 스케일 변화, 목표 변형, 모션 블러, 카메라 운동 등 다양한 과제 극복

### 2. 논문의 문제 정의, 제안 방법, 성능 및 한계

#### 2.1 해결하고자 하는 문제

비디오 객체 추적(VOT)과 비디오 객체 분할(VOS)은 컴퓨터 비전의 기본 과제이지만, 기존 방법들은 다음과 같은 한계를 가진다:[1]

1. **대규모 수작업 주석에 의존**: 현재의 최첨단 추적기/분할기는 대규모로 수동 주석이 된 데이터셋으로 훈련되며, 이는 막대한 인적 비용 초래
2. **높은 초기화 요구사항**: 특히 반감독 VOS는 정확한 객체 마스크 그라운드 트루스가 필요하여 실제 응용이 어려움
3. **시간적 일관성 부족**: SAM은 이미지에서는 뛰어나지만 프레임 간 시간적 대응이 부족하여 비디오에서 일관된 분할 불가능

#### 2.2 제안하는 방법 및 수식

**TAM 파이프라인은 4단계로 구성된다:**[1]

**단계 1: SAM을 이용한 초기화**
사용자가 클릭으로 관심 객체를 지정하면 SAM이 초기 마스크 $$M_0$$를 생성한다.

$$M_0 = \text{SAM}(\text{Frame}_0, \text{Prompts})$$

여기서 Prompts는 포인트 또는 바운딩 박스 형태의 사용자 입력이다.[1]

**단계 2: XMem을 이용한 추적**
초기 마스크가 주어지면 XMem은 다음 프레임에서 마스크 예측 $$M_t$$를 수행한다:[1]

$$M_t = \text{XMem}(F_t, M_{t-1}, \text{Memory})$$

여기서 $$F_t$$는 프레임 t의 특성, Memory는 시간적 특성 저장소를 의미한다.

**XMem의 메모리 아키텍처:**[2]
XMem은 인지심리학의 Atkinson-Shiffrin 메모리 모델에서 영감을 받아 세 가지 메모리 저장소를 구현한다:

$$\text{Memory} = \{\text{Sensory Memory}, \text{Working Memory}, \text{Long-term Memory}\}$$

각각의 역할은:
- **Sensory Memory**: 빠르게 업데이트되는 즉각적 시간 연속성 제공
- **Working Memory**: 고해상도의 최근 프레임 세부사항 유지
- **Long-term Memory**: 압축된 형태의 핵심 정보를 장시간 보관

메모리 강화 알고리즘을 통해 working memory를 long-term memory로 통합한다:

$$M_{\text{long}} = \text{Consolidate}(M_{\text{work}}, M_{\text{long}})$$

**단계 3: SAM을 이용한 정제**
XMem의 예측이 만족스럽지 않으면, probes와 affinities를 포인트 프롬프트로 변환하여 SAM으로 마스크를 정제한다:[1]

```math
M_t^{\text{refined}} = \text{SAM}(F_t, \text{Point\_Prompts}(P_t), \text{Mask\_Prompt}(M_t^{\text{coarse}}))
```

**단계 4: 사용자 수정**
필요시 사용자는 양수/음수 클릭으로 현재 프레임을 강제로 수정할 수 있다.[1]

#### 2.3 모델 구조

TAM의 구조는 **두 개의 기초 모델을 시너지 있게 결합**한 설계이다:[1]

```
Input Video Frame
    ↓
[Step 1] SAM Initialization
    ↓ (Target verification)
    ↓
[Step 2] XMem Tracking
    ├─ Probes & Affinities (coarse mask)
    ↓
[Quality Assessment]
    ├─ Good? → Output
    └─ Not Good? ↓
[Step 3] SAM Refinement
    ├─ Point Prompts
    ├─ Mask Prompt
    └─ Fine Mask
    ↓
[Step 4] Human Correction (if needed)
    └─ Positive/Negative Clicks
```

**SAM (Segment Anything Model):**[1]
- Vision Transformer (ViT) 기반
- 11백만 이미지와 1.1억 개 마스크로 학습
- 강력한 제로샷 분할 능력 보유
- 다양한 프롬프트(포인트, 박스, 언어) 지원

**XMem (eXplainable Memory):**[2]
- 장기 비디오 처리 최적화
- Atkinson-Shiffrin 메모리 모델 기반 아키텍처
- Space-time 메모리 읽기 메커니즘 포함

#### 2.4 성능 향상 및 한계

**성능 평가 결과:**[1]

| 데이터셋 | J & F | 설명 |
|---------|-------|------|
| DAVIS-2016-val | 88.4 | 원클릭 초기화로 달성 |
| DAVIS-2017-test-dev | 73.1 | 원클릭 초기화로 달성 |

비교 대상 모델들과의 비교:[1]

| 방법 | 초기화 방식 | DAVIS-2016-val J&F | DAVIS-2017-test-dev J&F |
|-----|-----------|-----------------|---------------------|
| STM | Mask | 89.9 | 75.2 |
| AOT | Mask | 92.1 | 83.3 |
| XMem | Mask | 93.2 | 84.7 |
| MIVOS | Scribble | 92.4 | 82.2 |
| TAM | Click | 89.4 | 76.4 |

**한계점:**[1]

1. **장기 비디오 처리의 어려움**: 마스크가 시간이 지남에 따라 축소되거나 정제 부족 문제 발생. SAM의 정제 능력이 실제 응용에서 기대치 이하
2. **복잡한 구조 처리 실패**: 자전거 바퀴처럼 많은 공동(cavity)을 가진 복잡한 객체 마스크 초기화 어려움. SAM이 미세 구조에서 성능 저하
3. **추가 인간 참여의 필요성**: 극도로 어려운 시나리오에서는 효율성을 해치지 않으면서도 여러 수정이 필요할 수 있음

### 3. 모델의 일반화 성능 향상 가능성

#### 3.1 제로샷 학습의 강점과 한계

TAM의 일반화 성능은 **기초 모델의 제로샷 능력**에 크게 의존한다.[1]

**강점:**
- SAM의 11백만 이미지 학습으로 인한 강력한 제로샷 분할 능력
- 임의의 비디오에 대한 추가 학습 없이 작동
- 다양한 프롬프트 유형 지원으로 인한 유연성

**한계:**
- 도메인 시프트(Domain Shift)에 취약: 극도로 다른 환경이나 작업 유형에서 성능 저하
- 시간적 일관성 유지의 어려움: SAM은 각 프레임을 독립적으로 처리하므로 장기 일관성 부족[1]

#### 3.2 최신 모델들과의 비교를 통한 일반화 성능 분석

**SAM 2 (2024년):**[3][4]
SAM 2는 TAM의 이후 발전 모델로, 비디오 처리를 위해 설계되었다:

- **스트리밍 메모리 아키텍처**: 실시간 비디오 처리를 위한 메모리 인코더, 메모리 뱅크, 메모리 어텐션 모듈 포함
- **성능**: 기존 방법 대비 3배 적은 상호작용으로 더 높은 정확도 달성[3]
- **장점**: 객체 폐색 및 재출현 처리 능력 향상, 프레임당 약 44 fps 실시간 성능[4]

$$\text{Memory}\_{\text{SAM2}} = f_{\text{encoder}}(M_{t-1}) + f_{\text{attention}}(\text{Memory}, F_t)$$

여기서 메모리는 이전 프레임의 정보를 동적으로 저장하고 현재 프레임에 적용한다.[3]

**SAM2Long (2024년):**[5][6]
SAM 2의 한계인 장기 비디오 처리를 개선하기 위해 제안된 방법:

- **문제점**: 그리디 선택 메모리 설계로 인한 "오류 누적(Error Accumulation)" 문제
- **해결책**: 제약된 트리 검색을 통한 여러 분할 경로 유지

```math
\text{Path}_{\text{next}} = \text{Select\_Top\_K}(\text{Score}_t + \text{Score}_{t-1})
```

여기서 각 경로의 누적 점수가 고려되어 최적 결과 선택[6][5]

**DEVA (2023년):**[7][8]
"Tracking Anything with Decoupled Video Segmentation"으로, 비디오 분할을 세 개의 독립적 부분작업으로 분리:

- **분할(Segmentation)**: 단일 프레임에서 모든 객체 추출
- **추적(Tracking)**: 프레임 간 객체 연결
- **정제(Refinement)**: 시간적 정보 활용한 최적화

이는 비디오 길이와 복잡도에 무관한 성능을 달성:[7]

$$\text{VOS} = \text{Segment}(F_t) \otimes \text{Track}(F_t, F_{t-1}) \otimes \text{Refine}(F_{1:T})$$

**WarpFormer (2024년):**[9]
모션 이해를 활용한 반감독 VOS 아키텍처:

- **광학 흐름 기반 전파**: 더 부드러운 전파를 위해 기존 모션 이해 활용
- **대규모 MOSE 2023 데이터셋**: 복잡한 시나리오에서 학습

#### 3.3 일반화 성능 향상의 핵심 요소

1. **메모리 메커니즘의 고도화**: XMem의 다층 메모리에서 SAM2의 스트리밍 메모리로의 진화가 일관성 유지에 결정적
2. **시간적 모델링 개선**: DEVA의 분리된 추적-정제 단계와 WarpFormer의 모션-인식 전파가 성능 향상
3. **다중 경로 탐색**: SAM2Long의 트리 검색 기법이 오류 누적 문제 완화
4. **도메인 적응**: 특정 도메인(수술 비디오, 위성 영상 등)에 대한 파인튜닝으로 실무 성능 향상[10]

### 4. 연구의 영향과 향후 고려사항

#### 4.1 현재 연구에 미치는 영향

**TAM의 발표(2023년 4월)는 다음과 같은 연구 방향을 촉발했다:**

1. **Foundation Model 통합 접근법의 확산**
   - SAM의 비디오 확장이 활발한 연구 주제로 부상[11][12]
   - SAM 2(2024년 8월)로 이어지는 자연스러운 진화 경로 제시[3]

2. **Interactive Video Segmentation의 활성화**
   - 클릭 기반 상호작용의 실용성 재평가
   - MIVOS와의 비교를 통해 회차당 상호작용 최소화 추세[1]

3. **시간적 일관성 문제의 재조명**
   - XMem-기반 접근이 표준적 방법론으로 정립
   - 메모리 메커니즘 개선의 필요성 강조[2]

4. **Zero-shot 비디오 분할의 활성화**
   - 기존의 "반감독" 패러다임에서 벗어나 "제로샷" 지향[13][14][15]
   - 대규모 벤치마크(MOSE, LVOS, MeViS) 개발 촉진[16]

#### 4.2 SAM 2와의 비교를 통한 발전 방향

| 특성 | TAM (2023) | SAM 2 (2024) | 개선점 |
|-----|-----------|-----------|-------|
| 메모리 설계 | XMem 외부 활용 | 통합 스트리밍 메모리 | 모델 내재화 |
| 상호작용 방식 | 명시적 클릭 | 동적 프롬프트 추가 | 효율성 3배 향상 |
| 처리 속도 | 일회 통과 | 실시간 44fps | 실시간성 달성 |
| 폐색 처리 | 미흡 | Occlusion Head | 재등장 능력 |
| 학습 데이터 | SA-1B + 수동 비디오 | 데이터 엔진 자동화 | 최대 규모 비디오 데이터셋 |

#### 4.3 향후 연구 시 고려할 점

**1. 장기 비디오 처리의 해결 (Critical)**

현재 한계점인 장기 비디오에서의 마스크 축소 및 오류 누적 문제 해결:

- SAM2Long 방식의 트리 검색 도입[6]
- Efficient memory management를 통한 계산 효율화[17]
- 메모리 뱅크 크기 동적 조정 메커니즘

$$\text{MemoryQuality} = \alpha \cdot \text{Confidence} + (1-\alpha) \cdot \text{Diversity}$$

품질 기준으로 메모리 선택 최적화[5]

**2. 도메인 적응 및 파인튜닝 전략**

일반 모델의 특정 분야 적응:

- 수술 비디오: DiveMem 메커니즘으로 의료용 장기 추적 달성[10]
- 위성 영상: 사전 지식(Prior Knowledge) 통합으로 성능 향상[18]
- 움직이는 대상: 광학 흐름(Optical Flow) 기반 모션 인식[19][9]

**3. 멀티모달 정보 활용 (Emerging)**

최신 연구 트렌드인 비디오-텍스트 결합 접근:

- X-Prompt: RGB+X 멀티모달 접근으로 극한 조명, 빠른 움직임 처리[20]
- SAMWISE: 자연언어 추론을 위한 텍스트 기반 분할[21]
- MemorySAM: 멀티모달 시맨틱 분할을 위한 메모리 메커니즘[22]

**4. 효율성 최적화**

실제 응용을 위한 계산 효율화:

- Lightweight ViT 기반 EfficientTAM: 2배 속도 향상, 2.4배 파라미터 감소[17]
- Surgical SAM 2: 프레임 선택 메커니즘으로 실시간 수술 비디오 처리[23]
- Token-level pruning: 불필요한 계산 제거[23]

**5. 신뢰성 및 불확실성 추정 (Future Direction)**

모델의 신뢰할 수 있는 예측을 위한 불확실성 정량화:

- Confidence estimation을 통한 자동 수정 판단
- Ensemble methods를 통한 안정성 향상
- Out-of-distribution detection for open-world scenarios

**6. 인간-AI 협력 인터페이스 개선**

사용자 경험 관점의 상호작용 최적화:

- 적응형 피드백: 사용자의 수정 패턴 학습
- 예측된 오류 지역 자동 표시
- 점진적 개선을 위한 라운드-트립 피드백 최소화

### 5. 2020년 이후 관련 최신 연구 비교 분석

#### 5.1 주요 연구 계보

**2020-2022: Semi-supervised VOS 시대**

- **STM (2019)**: 기초 모델로 자리잡은 공간-시간 메모리 네트워크[1]
- **AOT (2021)**: Associating Objects with Transformers[1]
- **XMem (2022)**: Atkinson-Shiffrin 메모리 모델 도입, 현재까지 표준 모델[2]

**2023: Foundation Model의 비디오 확장 시대**

- **Track Anything (2023)**: SAM + XMem 결합, 일회 상호작용 제시[1]
- **DEVA (2023)**: 분리된 아키텍처(분할-추적-정제)로 유연성 확보[8]
- **DVIS (2023)**: 비디오 인스턴스 분할 분리 전략[8]

**2024-2025: Foundation Model의 내재화 및 최적화 시대**

- **SAM 2 (2024)**: 통합 스트리밍 메모리로 실시간 처리 달성[3]
- **SAM2Long (2024)**: 트리 검색으로 장기 비디오 오류 누적 해결[6]
- **EfficientTAM (2024)**: 경량화로 모바일 배포 가능[17]
- **SAMURAI (2024)**: 모션-인식 메모리로 추적 성능 향상[24]
- **VideoLISA (2024)**: 언어 지시형 추론 세분화[25]

#### 5.2 기술적 진화 경향

| 년도 | 핵심 기술 | 대표 논문 | 성능 (J&F) | 특징 |
|------|---------|---------|----------|------|
| 2019 | Spatial-Temporal Memory | STM | 89.9 | 초기 메모리 기반 |
| 2021 | Attention Mechanism | AOT | 92.1 | Transformer 도입 |
| 2022 | Multi-layer Memory | XMem | 93.2 | 심리학 모델 기반 |
| 2023 | Foundation Model | TAM | 89.4 | SAM 통합, 낮은 초기화 |
| 2023 | Decoupled Design | DEVA | N/A | 유연한 아키텍처 |
| 2024 | Integrated Memory | SAM 2 | 89+ | 실시간 처리 |
| 2024 | Error Correction | SAM2Long | 85+ | 장기 안정성 |
| 2024 | Lightweight Design | EfficientTAM | 87+ | 모바일 배포 |

#### 5.3 영역별 특화 연구

**의료 이미징:**
- **Medical SAM 2 (2024)**: 3D 의료 영상을 비디오 추적으로 처리[26]
- **Surgical SAM 2 (2024)**: 프레임 제거로 수술 비디오 실시간 처리[23]
- **SAM2S (2024)**: 수술용 구체적 파인튜닝으로 성능 향상[10]

**언어 기반 분할:**
- **VideoLISA (2024)**: 복잡한 추론을 위한 언어 지시형 기능[25]
- **SAMWISE (2024)**: 텍스트 프롬프트 기반 장기 비디오 처리[21]
- **X-Prompt (2024)**: 멀티모달 프롬프트 통합[20]

**저수준 비디오 처리:**
- **Efficient Track Anything (2024)**: 경량 ViT로 2배 속도 향상[17]
- **Motion-Aware Transformer (2025)**: 모션 예측으로 추적 개선[19]

#### 5.4 성능 트렌드 분석

**DAVIS 2017 벤치마크 (가장 엄격한 평가)에서의 발전:**[13][3][1]

```
2019 STM:       75.2 J&F
2021 AOT:       83.3 J&F  (+8.1)
2022 XMem:      84.7 J&F  (+1.4)
2023 TAM:       76.4 J&F  (-8.3, 하지만 클릭만 사용)
2024 SAM 2:     89+ J&F   (+5이상)
```

**핵심 통찰:**
- SAM 2는 XMem 수준의 성능을 달성하면서 **상호작용을 3배 감소**[3]
- TAM의 일회 통과 설계는 정확도 손실이 있지만 **실용성과 속도에서 우위**[1]
- 최신 모델들은 **메모리 설계 고도화**에 중점[5][6][17]

#### 5.5 미해결 문제 및 연구 기회

1. **초장기 비디오(>10분) 처리**: 여전히 오류 누적 문제 존재
2. **복잡한 폐색(Complex Occlusion)**: 여러 객체의 상호 폐색
3. **도메인 일반화**: 학습 데이터와 다른 환경에서의 성능
4. **실시간 다중 객체 추적**: 높은 해상도에서의 계산 효율성
5. **자가 수정 능력**: 모델이 자동으로 오류를 감지하고 수정

***

## 결론

Track Anything Model은 **기초 모델의 강점을 활용한 혁신적 접근법**을 제시하여 비디오 객체 분할 분야에 중요한 전환점을 마련했다. SAM의 제로샷 능력과 XMem의 시간적 메모리를 결합함으로써, 기존의 복잡한 전처리 과정을 간소화하고 사용자 친화적 인터페이스를 구현했다.[1]

이후 SAM 2, SAM2Long, DEVA 등의 발전 모델들은 TAM의 개념적 기여를 바탕으로 **메모리 메커니즘의 내재화, 실시간 처리 능력, 그리고 도메인 특화 최적화**를 달성했다. 특히 스트리밍 메모리 아키텍처와 오류 누적 방지 전략은 장기 비디오 처리의 새로운 표준을 제시했다.

향후 연구는 **(1) 초장기 비디오 안정성, (2) 멀티모달 정보 활용, (3) 계산 효율화, (4) 인간-AI 협력 개선**에 집중해야 하며, 특히 의료 영상, 위성 모니터링 등 특정 도메인에서의 파인튜닝과 실무 적용이 중요한 과제로 남아 있다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7273ddf1-da0a-4b9a-bfe6-4229805e6a0e/2304.11968v2.pdf)
[2](https://www.alphaxiv.org/overview/2207.07115v2)
[3](https://arxiv.org/abs/2408.00714)
[4](https://docs.ultralytics.com/models/sam-2/)
[5](https://arxiv.org/abs/2410.16268)
[6](https://arxiv.org/html/2410.16268)
[7](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_Tracking_Anything_with_Decoupled_Video_Segmentation_ICCV_2023_paper.pdf)
[8](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_DVIS_Decoupled_Video_Instance_Segmentation_Framework_ICCV_2023_paper.pdf)
[9](https://arxiv.org/pdf/2405.07031.pdf)
[10](https://www.semanticscholar.org/paper/6c522555ee11664dcf1e56966f041aacb407e0ec)
[11](http://arxiv.org/pdf/2408.00714.pdf)
[12](https://arxiv.org/abs/2408.06305)
[13](https://arxiv.org/abs/2408.10125)
[14](https://arxiv.org/abs/2408.12447)
[15](https://arxiv.org/html/2411.18977)
[16](https://arxiv.org/abs/2409.05847)
[17](https://arxiv.org/abs/2411.18933)
[18](https://www.sciencedirect.com/science/article/abs/pii/S0924271625003028)
[19](https://arxiv.org/html/2509.21715v1)
[20](https://arxiv.org/html/2409.19342)
[21](https://arxiv.org/html/2411.17646v1)
[22](https://arxiv.org/html/2503.06700v2)
[23](http://arxiv.org/pdf/2408.07931.pdf)
[24](https://arxiv.org/abs/2411.11922)
[25](https://arxiv.org/html/2409.19603)
[26](https://arxiv.org/abs/2408.00874)
[27](https://link.springer.com/10.1007/s11633-022-1378-4)
[28](https://arxiv.org/abs/2402.08882)
[29](https://link.springer.com/10.1007/s11042-023-16417-3)
[30](https://journal.asdkvi.or.id/index.php/Abstrak/article/view/337)
[31](https://ejournal.nusantaraglobal.ac.id/index.php/nusra/article/view/3143)
[32](https://arxiv.org/abs/2307.16803)
[33](https://arxiv.org/abs/2408.14562)
[34](https://arxiv.org/abs/2412.05331)
[35](https://www.sadivin.com/jour/article/view/1152)
[36](http://arxiv.org/pdf/1905.10064.pdf)
[37](https://arxiv.org/html/2311.18286v1)
[38](http://arxiv.org/pdf/1701.05384.pdf)
[39](https://arxiv.org/abs/2404.19326)
[40](https://arxiv.org/abs/2310.12982)
[41](https://ai.meta.com/sam2/)
[42](https://www.sciencedirect.com/science/article/abs/pii/S0262885625000769)
[43](https://arxiv.org/abs/2406.04600)
[44](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Fast_Video_Object_Segmentation_With_Temporal_Aggregation_Network_and_Dynamic_CVPR_2020_paper.pdf)
[45](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Putting_the_Object_Back_into_Video_Object_Segmentation_CVPR_2024_paper.html)
[46](https://github.com/z-x-yang/Segment-and-Track-Anything)
[47](https://openreview.net/forum?id=UDeARVACQi)
[48](https://github.com/gaomingqi/Awesome-Video-Object-Segmentation)
[49](https://arxiv.org/abs/2408.04593)
[50](https://ieeexplore.ieee.org/document/10656025/)
[51](https://arxiv.org/html/2408.10125v2)
[52](https://arxiv.org/html/2410.08781v1)
[53](https://openreview.net/forum?id=Ha6RTeWMd0)
[54](https://cveu.github.io/2022/papers/0021.pdf)
[55](https://openaccess.thecvf.com/content/ICCV2023/papers/Bekuzarov_XMem_Production-level_Video_Segmentation_From_Few_Annotated_Frames_ICCV_2023_paper.pdf)
[56](https://www.youtube.com/watch?v=1oviQF_LeEA)
[57](https://openreview.net/pdf/7c41968163abe4e3700e3e3a15174a9d679fcd52.pdf)
[58](https://ieeexplore.ieee.org/document/10657878/)
[59](https://ieeexplore.ieee.org/document/9709942/)
[60](https://arxiv.org/abs/2411.19210)
[61](https://ieeexplore.ieee.org/document/10105896/)
[62](https://ieeexplore.ieee.org/document/10377786/)
[63](https://link.springer.com/10.1007/s11263-024-02024-8)
[64](https://ieeexplore.ieee.org/document/10298026/)
[65](https://onlinelibrary.wiley.com/doi/10.4218/etrij.2023-0115)
[66](https://ksbe-jbe.org/_common/do.php?a=full&b=13&bidx=3885&aidx=42931)
[67](https://arxiv.org/abs/2111.06394)
[68](http://arxiv.org/pdf/2406.05485.pdf)
[69](http://arxiv.org/pdf/2403.04258.pdf)
[70](http://arxiv.org/pdf/1903.05612.pdf)
[71](https://arxiv.org/html/2410.16953)
[72](http://arxiv.org/pdf/2312.11557.pdf)
[73](https://arxiv.org/html/2312.09525)
[74](http://arxiv.org/pdf/2108.05076.pdf)
[75](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Learning_Motion-Appearance_Co-Attention_for_Zero-Shot_Video_Object_Segmentation_ICCV_2021_paper.pdf)
[76](https://www.twelvelabs.io/blog/the-past-present-and-future-of-video-understanding-applications)
[77](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620290.pdf)
[78](https://arxiv.org/html/2406.01493v3)
[79](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870648.pdf)
[80](https://arxiv.org/abs/2406.05485)
[81](https://openreview.net/forum?id=Y0FTtmzHGRk)
[82](https://www.themoonlight.io/ko/review/training-free-robust-interactive-video-object-segmentation)
[83](https://arxiv.org/abs/2403.04258)
[84](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11491.pdf)
[85](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2023.123207)
[86](https://liner.com/ko/review/learning-video-object-segmentation-from-unlabeled-videos)
[87](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.pdf)
