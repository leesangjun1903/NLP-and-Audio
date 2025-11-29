
# MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

## 1. 핵심 주장 및 주요 기여

MinerU2.5는 문서 파싱 작업에서 **성능과 효율성의 새로운 균형**을 달성하는 1.2B 파라미터 비전-언어 모델(VLM)입니다. 본 연구의 핵심 주장은 다음과 같습니다:[1]

**주요 기여:**

1. **분리된 코스-투-파인(Coarse-to-Fine) 두 단계 파싱 전략**: 전역 레이아웃 분석과 로컬 콘텐츠 인식을 분리하여 계산 복잡도를 획기적으로 감소시킵니다.[1]

2. **네이티브 해상도 처리 메커니즘**: 고해상도 이미지에서 세부 정보를 보존하면서도 \(\mathcal{O}(N^2)\) 복잡도 문제를 회피합니다.[1]

3. **포괄적인 데이터 엔진**: 반복적 추론 일관성(Iterative Mining via Inference Consistency, IMIC) 전략을 통해 어려운 샘플을 자동 식별합니다.[1]

4. **성능 기준 수립**: OmniDocBench에서 전체 점수 90.67을 달성하여 일반 목적 및 특화된 모델을 모두 능가합니다.[1]

***

## 2. 해결 문제 및 제안 방법

### 2.1 해결하는 문제

기존 문서 파싱 접근법들은 다음과 같은 한계를 가집니다:

- **파이프라인 기반 방식**: 각 단계별 오류 누적으로 인한 성능 저하
- **엔드-투-엔드 VLM**: 고해상도 입력 처리 시 \(\mathcal{O}(N^2)\) 토큰 복잡도로 인한 비효율성
- **할루시네이션 문제**: 텍스트 밀도가 높은 영역에서의 신뢰도 저하
- **토큰 중복성**: 문서 이미지의 빈 또는 저정보 영역에서의 불필요한 토큰 생성[1]

### 2.2 제안 방법 및 수식

#### **Stage I: 레이아웃 분석**

입력 이미지를 1036×1036 픽셀의 축소 썸네일로 리사이징하여 전역 레이아웃을 분석합니다.

입력 이미지 \(I\)에 대해:
$$I_{\text{thumbnail}} = \text{Resize}(I, 1036 \times 1036)$$

#### **Stage II: 콘텐츠 인식**

Stage I의 레이아웃 정보 \(L\)을 기반으로 네이티브 해상도 크롭을 추출하여 세밀한 콘텐츠 인식을 수행합니다:

$$\text{Crop}_i = \text{Extract}(I, \text{BBox}_i), \quad \text{where } |\text{Crop}_i| \leq 2048 \times 2048$$

#### **PageIoU 메트릭**

레이아웃 품질 평가를 위해 새로운 픽셀 기반 커버리지 메트릭을 도입했습니다:

예측된 레이아웃 \(P = \{bbox_i | i = 1, 2, \ldots, n\}\)과 그라운드 트루스 \(G = \{bbox_j | j = 1, 2, \ldots, m\}\)에 대해:

먼저 커버리지 맵을 정의합니다:

$$G_{\text{cover}} = \sum_{j=1}^{m} \mathbb{1}_{p \in bbox_j}, \quad P_{\text{cover}} = \sum_{i=1}^{n} \mathbb{1}_{p \in bbox_i}$$

PageIoU는 다음과 같이 정의됩니다:

$$\text{PageIoU}(P, G) = \frac{\sum_{p \in M} \min(P_{\text{cover}}[p], G_{\text{cover}}[p])}{\sum_{p \in M} \max(P_{\text{cover}}[p], G_{\text{cover}}[p])}$$

여기서 \(M\)은 페이지의 비배경 영역을 나타냅니다.[1]

### 2.3 모델 아키텍처

MinerU2.5는 세 가지 핵심 구성요소로 이루어져 있습니다:

| 구성요소 | 사양 | 기능 |
|---------|------|------|
| 비전 인코더(NaViT) | 675M 파라미터 | 동적 해상도 처리, 2D-RoPE 위치 인코딩 |
| 패치 머저 | 2-layer MLP | 인접한 2×2 비전 토큰에 대한 픽셀 언셔플 |
| 언어 모델(Qwen2-Instruct) | 0.5B 파라미터 | M-RoPE로 해상도 적응성 향상 |[1]

***

## 3. 공식화된 작업 및 데이터 처리

### 3.1 공식화된 작업

#### **레이아웃 분석의 다중 작업 패러다임**

기존 단순 객체 감지 대신, 다음 네 가지 속성을 동시에 예측합니다:

$$\text{Layout}(e_k) = \{\text{Position}(e_k), \text{Class}(e_k), \text{Rotation}(e_k), \text{ReadingOrder}(e_k)\}$$

#### **원자 분해-재조합(ADR) 프레임워크**

복합 공식을 원자 공식으로 분해한 후 개별 인식을 수행하고 재조합합니다:

$$\text{Formula}_{\text{compound}} = \text{Recombine}(\{\text{LaTeX}_i : i = 1, \ldots, n\})$$

#### **최적화된 테이블 구조 언어(OTSL)**

HTML 대신 OTSL을 사용하여 시퀀스 길이를 약 50% 단축합니다:

$$\text{TokenCount}_{\text{HTML}} > 28, \quad \text{TokenCount}_{\text{OTSL}} = 5$$

### 3.2 반복적 추론 일관성(IMIC) 전략

모델이 학습한 피처가 견고한지 판단하기 위해 여러 추론 실행의 일관성을 측정합니다:

**레이아웃 분석**: 다중 실행 결과 간 페어와이즈 PageIoU 계산
$$\text{mPageIoU}_{threshold} = 0.8 \Rightarrow \text{Hard Case}$$

**테이블 인식**: TEDS (Tree-Edit-Distance-based Similarity) 점수 사용
$$\text{mTEDS}_{threshold} = 0.6 \Rightarrow \text{Hard Case}$$

**공식 인식**: 문자 감지 매칭(CDM) 점수 사용
$$\text{mCDM}_{threshold} = 0.3 \Rightarrow \text{Hard Case}$$[1]

***

## 4. 훈련 전략 및 데이터 엔진

### 4.1 3단계 훈련 프로세스

| 단계 | 목표 | 샘플 수 | 에포크 |
|------|------|--------|--------|
| Stage 0 | 모달리티 정렬 | 1.22M | 1 |
| Stage 1 | 문서 파싱 사전훈련 | 6.9M | 2 |
| Stage 2 | 문서 파싱 미세조정 | 630K | 3 |

**데이터 구성 (Stage 1)**:
- 레이아웃 분석: 2.3M 샘플
- 텍스트 블록: 2.4M 샘플
- 공식 블록: 1.1M 샘플
- 테이블 블록: 1.1M 샘플[1]

### 4.2 데이터 증강 전략

| 증강 유형 | 작업 |
|----------|------|
| 공간 변환 | 스케일링, 그리드 왜곡, 회전 |
| 배경 변환 | 텍스처, 날씨 효과, 워터마크, 스캔라인, 그림자 |
| 색상 변환 | 밝기/명도, 조명, RGB 시프트 |
| 저화질 변환 | PSF 블러, 진동 블러, 가우시안 블러, 침식/팽창 |[1]

***

## 5. 성능 향상 및 벤치마크 결과

### 5.1 전체 문서 파싱 성능

**OmniDocBench 결과:**

| 지표 | MinerU2.5 | 두 번째 | 성능 향상 |
|------|----------|--------|----------|
| 전체 점수 | 90.67 | 88.85 (MonkeyOCR-pro-3B) | +1.82 |
| 텍스트 편집 거리 | 0.047 | 0.048 (dots.ocr) | -0.001 |
| 공식 CDM | 88.46 | 88.27 (Qwen2.5-VL-72B) | +0.19 |
| 테이블 TEDS | 88.22 | 87.45 (MonkeyOCR-3B) | +0.77 |[1]

**Ocean-OCR 벤치마크:**
- 영어: 편집 거리 0.033, F1 점수 0.945
- 중국어: F1 점수 0.965, 정밀도 0.966[1]

### 5.2 추론 효율성

| 모델 | 파라미터 | 토큰/초 | 페이지/초 | 개선도 |
|------|--------|--------|-----------|---------|
| MinerU2.5 (RTX 4090) | 1.2B | 1,875.82 | 1.70 | - |
| MinerU2.5 (A100 80G) | 1.2B | 2,337.25 | 2.12 | - |
| MonkeyOCR-pro-3B | 3.7B | 520.16 | 0.47 | **4.5배** |
| dots.ocr | 3.0B | 311.06 | 0.28 | **7.6배** |[1]

### 5.3 요소별 성능

**공식 인식 (CDM 메트릭):**
- 복잡한 인쇄 방정식(CPE): 96.6%
- 간단한 인쇄 방정식(SCE): 96.4%
- LaTeX-80MM (행렬): 90.6%[1]

**테이블 인식 (TEDS 메트릭):**
- PubTabNet: 89.07 (두 번째: 90.65)
- FinTabNet: **95.97** (SOTA)
- In-house TR Benchmark: **71.48** (SOTA)[1]

***

## 6. 일반화 성능 및 향후 개선 가능성

### 6.1 일반화 성능 특성

MinerU2.5는 다양한 측면에서 **강화된 일반화 능력**을 보여줍니다:

#### **문서 타입 다양성**
학술 논문, 교과서, 보고서, 슬라이드, 신문, 잡지, 시험지, 노트 등 9개 카테고리에서 6개 카테고리에서 최고 또는 준최고 성능을 달성했습니다.[1]

#### **0-샷 일반화**
제공된 데이터 없이 새로운 문서 타입과 도메인에 대한 일반화 능력이 높습니다. PageIoU 메트릭을 사용한 레이아웃 분석의 다양한 벤치마크(OmniDocBench, D4LA, DocLayNet)에서 일관되게 최고 성능을 달성했습니다.[1]

### 6.2 강건성 개선

#### **다양한 문서 간섭 처리**
- 회전된 테이블 자동 감지 및 보정
- 테두리 없는 테이블, 부분 테두리 테이블 처리
- 워터마크, 손글씨, 저화질 이미지에 대한 강건성

#### **할루시네이션 완화**
두 단계 분리 구조는 다음을 통해 VLM의 할루시네이션을 효과적으로 완화합니다:
- Stage I에서 명확한 레이아웃 정보 제공
- Stage II에서 로컬 화이드우 처리로 인한 집중도 향상
- 동적 샘플링 파라미터 조정으로 반복 토큰 억제[1]

### 6.3 향후 일반화 성능 향상 방향

#### **(1) 크로스-도메인 적응 기법**
- **도메인 특화 미세조정**: 금융, 의료, 법률 등 특정 도메인의 문서에 대한 경량 미세조정 모듈 개발
- **도메인 불변 표현 학습**: 대조 학습을 통해 도메인 간 공통 특징 학습[2]

#### **(2) 강화 학습 기반 최적화**
최근 연구(DianJin-OCR-R1, olmOCR 2)는 강화 학습을 통한 일반화 개선을 시연했습니다. MinerU2.5도 다음과 같은 방식을 적용할 수 있습니다:[3][4]
- 렌더링-비교 정렬을 통한 테이블 구조 최적화
- 바이너리 유닛 테스트 기반 보상 설계

#### **(3) 멀티태스크 학습 강화**
통합 VLM의 이해와 생성 작업 간 상호 작용을 극대화합니다:[5]
- 텍스트-이미지 캡셔닝 데이터 활용
- 구조적 마크다운 생성 작업 통합

#### **(4) 제로샷 일반화 최적화**
InstructDoc과 같은 접근법을 참고하여:[6]
- 명령 기반 문서 이해 모듈 추가
- 다양한 VDU 작업 데이터셋 통합

#### **(5) 할루시네이션 제어 메커니즘**
최근 방법들(DSCR, 자기 인식 불확실성)을 활용한 개선:[7][8]
- 깊이 및 공간 정보 기반 KV 캐시 정제
- 시각적 불확실성 자각 메커니즘 추가

***

## 7. 한계 및 향후 연구 고려사항

### 7.1 현재 한계

1. **장문 문서 처리**: 매우 긴 다중 페이지 문서에서의 컨텍스트 일관성 유지 문제[9]

2. **복합 레이아웃**: 매우 복잡한 다단 구조, 중첩된 표, 크로스 페이지 요소 처리의 정확성

3. **언어 간 편차**: 현재 중국어와 영어에 최적화되어 있으며, 다른 언어에 대한 일반화 미흡[10]

4. **도메인 특화 용어**: 고도로 기술화된 용어나 산업 특화 용어가 포함된 문서에서의 성능 저하 가능성

### 7.2 향후 연구 시 고려할 점

#### **(1) 아키텍처 개선**
- **윈도우 어텐션 재검토**: Qwen2.5-VL의 윈도우 어텐션이 성능을 저하시킨 이유에 대한 심층 분석 및 개선 방안
- **네이티브 해상도 최적화**: 더 효율적인 토큰 처리를 위한 새로운 메커니즘 개발

#### **(2) 데이터 엔진 확대**
- **다양한 언어 데이터 확충**: 한국어, 일본어, 아라비아어 등 다국어 지원 확장[9]
- **산업별 데이터셋**: 금융 보고서, 의료 기록, 법률 문서 등 도메인 특화 데이터 구축

#### **(3) IMIC 전략 고도화**
- **적응형 일관성 임계값**: 문서 타입에 따라 동적으로 조정되는 임계값 개발
- **다중 지표 통합**: PageIoU, TEDS, CDM을 조합하여 더 정교한 어려운 샘플 식별

#### **(4) 크로스 모달 통합**
- **이미지 콘텐츠 인식**: 차트, 그래프, 다이어그램 등 시각적 요소의 더 정교한 이해
- **시맨틱 정보 활용**: 문서 구조와 의미적 관계를 함께 고려한 파싱

#### **(5) 배포 최적화**
- **모바일 디바이스 지원**: 1.2B 모델을 기반으로 한 경량화 버전 개발
- **엣지 컴퓨팅 적응**: 오프라인 추론 성능 향상

#### **(6) RAG 시스템과의 통합**
논문에서 강조한 바와 같이, MinerU2.5의 구조적 정보 보존 능력을 활용하여:[1]
- 향상된 검색-증강-생성(RAG) 시스템 개발
- 테이블과 공식을 포함한 구조화된 데이터 검색 최적화

***

## 8. 학계 및 산업에 미치는 영향

### 8.1 학술적 영향

1. **문서 파싱 패러다임 전환**: 파이프라인 방식과 엔드-투-엔드 방식의 장점을 결합한 분리 모델 아키텍처 제시

2. **새로운 평가 메트릭**: PageIoU의 도입으로 문서 레이아웃 평가의 더 정확한 기준 제공

3. **데이터 마이닝 방법론**: IMIC 전략의 일반화 가능성 - 다른 시계열 또는 구조화 작업에 적용 가능[1]

4. **다중 작업 학습**: 레이아웃 분석에서 위치, 클래스, 회전, 읽기 순서를 동시에 예측하는 통합 방식의 확산

### 8.2 산업 적용 가능성

1. **문서 자동화**: 대규모 PDF 자동 변환 및 구조화 시스템 구축의 실질화

2. **RAG 시스템 고도화**: 복잡한 문서 이해를 기반으로 한 엔터프라이즈급 검색 시스템 개선[1]

3. **규제 준수**: 금융, 의료, 법률 산업에서의 고정확도 문서 처리 자동화

4. **다국어 처리**: 향후 다국어 지원으로 국제 기업의 문서 처리 효율성 향상

### 8.3 관련 최신 연구 동향

**2025년 주요 연구 발전:**

1. **PaddleOCR-VL (2025년 10월)**: 0.9B 컴팩트 모델로 109개 언어 지원, SOTA 성능 달성[10]

2. **MonkeyOCR v1.5 (2025년 11월)**: 구조-인식-관계(SRR) 패러다임으로 MinerU2.5를 능가하는 성능 보고, 특히 공식(+15.0%), 표(+8.6%)에서 개선[11]

3. **olmOCR 2 (2025년 10월)**: 강화 학습 기반 검증 보상으로 SOTA 달성, 공식 변환 성능 대폭 개선[4]

4. **DianJin-OCR-R1 (2025년 8월)**: 추론 및 도구 상호작용을 통한 할루시네이션 완화 메커니즘[3]

5. **HEAR (2025년 10월)**: 홀리스틱 추출과 에이전트 기반 추론 결합으로 멀티페이지 문서 처리 고도화[12]

6. **SmolDocling (2025년 3월)**: DocTags 범용 마크업 형식으로 27배 큰 모델을 능가하는 초소형 모델[13]

***

## 결론

MinerU2.5는 문서 파싱 분야에 **경량성, 효율성, 정확성의 새로운 표준**을 제시합니다. 1.2B 파라미터로 72B 이상의 모델을 능가하는 성능 달성은 실무 배포의 가능성을 획기적으로 높입니다. 특히 IMIC 기반 데이터 엔진과 분리된 아키텍처는 향후 문서 이해 연구의 중요한 방향을 제시합니다.

그러나 최근 MonkeyOCR v1.5, olmOCR 2 등의 등장으로 경쟁이 심화되고 있으며, **다국어 지원, 초장문 문서 처리, 도메인 특화 일반화** 등의 영역에서 지속적인 개선이 필요합니다. 앞으로의 연구는 단순한 성능 향상을 넘어 **크로스 모달 이해, 멀티페이지 일관성, 강화 학습 기반 최적화** 등으로 진화할 것으로 예상됩니다.[11][4][12][10][3][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fd581c48-b787-458e-bdee-cef8456873d1/2509.22186v2.pdf)
[2](https://ieeexplore.ieee.org/document/10545557/)
[3](https://arxiv.org/abs/2508.13238)
[4](https://arxiv.org/abs/2510.19817)
[5](https://arxiv.org/abs/2505.23043)
[6](https://arxiv.org/abs/2401.13313)
[7](https://arxiv.org/abs/2506.20168)
[8](https://openreview.net/pdf/a27b6c0dd486c319ef6cdd738680fc6b21afb9c1.pdf)
[9](https://arxiv.org/html/2410.21169v2)
[10](https://arxiv.org/abs/2510.14528)
[11](https://www.semanticscholar.org/paper/5c94d82e55a1778ae2c976b87988297aee2cb7ba)
[12](https://dl.acm.org/doi/10.1145/3746027.3761999)
[13](https://arxiv.org/html/2503.11576v1)
[14](https://arxiv.org/abs/2509.22186)
[15](https://arxiv.org/abs/2506.05218)
[16](https://arxiv.org/abs/2506.03197)
[17](https://dl.acm.org/doi/10.1145/3746027.3761997)
[18](https://arxiv.org/abs/2502.13923)
[19](http://arxiv.org/pdf/2309.05756.pdf)
[20](https://arxiv.org/pdf/2408.12637.pdf)
[21](https://arxiv.org/html/2407.12594v1)
[22](http://arxiv.org/pdf/2405.14295.pdf)
[23](http://arxiv.org/pdf/2410.21169.pdf)
[24](http://arxiv.org/pdf/2502.16161.pdf)
[25](http://arxiv.org/pdf/2405.00260.pdf)
[26](https://arxiv.org/html/2509.22186v1)
[27](https://airparser.com/blog/vision-vs-text-in-llm-document-parsing/)
[28](https://www.linkedin.com/posts/nodeshift_ocr-ai-llms-activity-7379793199151636480-bTna)
[29](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Enhancing_Visual_Document_Understanding_with_Contrastive_Learning_in_Large_Visual-Language_CVPR_2024_paper.pdf)
[30](https://huggingface.co/blog/vlms-2025)
[31](https://aiexpjourney.substack.com/p/from-big-picture-to-details-mineru)
[32](https://arxiv.org/html/2506.18504v1)
[33](https://www.hyperscience.ai/blog/out-of-the-box-to-state-of-the-art-how-vision-language-models-are-transforming-document-processing/)
[34](https://drpress.org/ojs/index.php/cpl/article/view/24380)
[35](http://wepub.org/index.php/IJMEE/article/view/2156)
[36](https://drpress.org/ojs/index.php/jceim/article/view/22631)
[37](https://ieeexplore.ieee.org/document/10979996/)
[38](https://www.semanticscholar.org/paper/4011c21cb4bef1815cc5001c8a22eb959e2e9ecf)
[39](https://arxiv.org/abs/2509.01642)
[40](https://arxiv.org/abs/2510.24000)
[41](https://www.ijcai.org/proceedings/2024/127)
[42](https://arxiv.org/abs/2402.14536)
[43](https://www.aclweb.org/anthology/P19-1443.pdf)
[44](http://arxiv.org/pdf/2412.10207.pdf)
[45](https://aclanthology.org/2021.findings-emnlp.149.pdf)
[46](https://www.aclweb.org/anthology/2020.coling-main.338.pdf)
[47](http://arxiv.org/pdf/2309.13258.pdf)
[48](https://arxiv.org/html/2412.12505v1)
[49](http://arxiv.org/pdf/1606.06361.pdf)
[50](http://arxiv.org/pdf/2210.09565.pdf)
[51](https://aclanthology.org/2022.emnlp-main.749.pdf)
[52](https://proceedings.mlr.press/v162/wang22u/wang22u.pdf)
[53](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d3b57e06e3fc45f077eb5c9f28156d4-Paper-Conference.pdf)
[54](https://www.themoonlight.io/ko/review/instructdoc-a-dataset-for-zero-shot-generalization-of-visual-document-understanding-with-instructions)
[55](https://arxiv.org/html/2506.05551v1)
[56](https://www.sciencedirect.com/science/article/pii/S2666764925000384)
