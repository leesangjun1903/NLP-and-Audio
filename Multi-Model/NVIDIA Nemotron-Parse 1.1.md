# NVIDIA Nemotron-Parse 1.1

### 1. 핵심 주장 및 주요 기여 요약

**NVIDIA Nemotron-Parse-1.1**은 경량 문서 파싱 및 OCR 모델로서, 엔드-투-엔드 비전-언어 모델의 새로운 표준을 제시한다. 이 모델의 핵심 주장은 **단순한 텍스트 추출을 넘어 레이아웃 분석, 읽기 순서 파악, 의미론적 분류, 테이블 추출, 수식 인식을 단일 모델로 동시에 수행**할 수 있다는 것이다.[1]

**주요 기여**는 다음과 같다:

1. **통합 아키텍처**: 기존 멀티 스테이지 파이프라인의 오류 누적 문제를 극복[1]
2. **경량 설계**: 885M 파라미터로 경쟁력 있는 성능 달성[1]
3. **위치 인코딩 제거 (NoPE)**: 디코더에서 위치 임베딩을 완전히 제거하면서 장문맥 일반화 성능 향상[1]
4. **멀티토큰 추론**: n개 토큰 동시 예측으로 훈련 신호 풍부화 및 추론 정확도 개선[1]
5. **경량화 버전 (TC)**: 토큰 길이 16배 축소로 20% 속도 개선[1]
6. **다국어 지원**: 7개 언어에서 F1 > 0.96 달성[1]
7. **완전 공개**: 모델 가중치, 최적화 컨테이너, 훈련 데이터 공개[1]

***

### 2. 해결 문제, 제안 방법, 모델 구조

#### 2.1 문제 정의

현대적 문서 처리 시스템이 요구하는 여러 작업 - **일반 OCR, 마크다운/LaTeX 포맷팅, 의미론적 블록 분류, 테이블 추출, 수식 인식, 읽기 순서 보존** - 을 기존 모델들은 동등하게 처리하지 못했다. 파이프라인 기반 솔루션은 정확하지만 오류가 누적되고 느리며, 엔드-투-엔드 VLM은 빠르지만 모든 서브태스크에서 최적 성능을 내지 못했다.[1]

#### 2.2 모델 아키텍처

**전체 구조**:

$$\text{Output} = \mathcal{D}(\mathcal{N}(\mathcal{E}(I)), t_{ < i})$$

**비전 인코더** - RADIO (ViT-H/16, 657M 파라미터):[1]

$$Z = \mathcal{E}(I), \quad Z \in \mathbb{R}^{N \times d}$$

여기서 $I \in \mathbb{R}^{3 \times H \times W}$는 입력 이미지, $N$은 시퀀스 길이, $d$는 숨겨진 차원이다.

**비전 넥** - 수평 컨볼루션 ($1 \times 4$ 커널, 스트라이드 $1 \times 4$):[1]

- 표준: 1648×2048 입력 → 3200 토큰
- TC: 픽셀 셔플 추가 → 833 토큰 (×16 축소)

$$Z_{\text{compressed}} = \text{PixelShuffle}(\text{Conv}(Z))$$

**디코더** - mBART 기반 (10층, 레이어 공유):[1]

$$P(t_i | \mathcal{N}(Z), t_{ < i})$$

총 885M 파라미터 (디코더 256M)

#### 2.3 위치 인코딩 제거 (NoPE)

**핵심 혁신**: 디코더에서 위치 임베딩을 완전히 제거[1]

$$\text{Position Information} = f(\text{Causal Mask}, \text{Token Similarity}, \text{Attention Patterns})$$

**근거**: 인과적 마스킹만으로 위치 정보 충분[1]
- 각 토큰은 선행 토큰만 참석 가능 → 위치 추론 가능
- 2D 시각 특징이 이미 문서 공간 구조 포함
- 명시적 위치 신호 추가 시 간섭 가능

**이점**: 외삽 오류 감소로 길이 일반화 성능 향상[1]

#### 2.4 멀티토큰 추론

n개 토큰 동시 예측:[1]

**훈련**: m개 토큰 예측 시 $m-1$개 선형 레이어 추가

$$\text{logits}_{n+1} = \text{head}(h_n)$$
$$\text{logits}_{n+2..m} = \text{head}(l_1(h_n + l_2(e_{n+1})))$$

여기서 $h_n$은 n번째 토큰의 최종 은닉 상태, $e_{n+1}$은 n+1번째 토큰의 임베딩, Teacher forcing 사용

**효과**: 단일-토큰 설정에서도 정확도 개선[1]

#### 2.5 프롬프트 인터페이스

8가지 유효한 조합으로 이질적 데이터셋 통합:[1]

$$\text{Prompt} = \{T, B, C\} \in \{\text{on/off}\}^3$$

- **T (텍스트)**: `<output_markdown>`, `<output_plain>`, `<output_no_text>`
- **B (바운딩박스)**: `<predict_bbox>`, `<no_bbox>`
- **C (클래스)**: `<predict_classes>`, `<no_classes>`

최대 정보 프롬프트:

$$\text{MIP} = \text{ output markdown } \oplus \text{ predict bbox } \oplus \text{ predict classes }$$

#### 2.6 출력 형식

상대 좌표 (1024×1280 스케일)의 정규 읽기 순서:[1]

```math
\text{ < x\_}\frac{x_1}{W}\text{ > < y\_}\frac{y_1}{H}\text{ > } \text{(text)} \text{ < x\_}\frac{x_2}{W}\text{ > < y\_}\frac{y_2}{H}\text{ > } \text{lass\_C > }
```

***

### 3. 성능 향상 및 벤치마크 결과

#### 3.1 내부 테스트셋 (789 페이지)

| 모델 | WER ↓ | F1 ↑ |
|------|-------|------|
| Kosmos-2.5 | 0.195 | 0.937 |
| GOT (md) | 0.259 | 0.879 |
| **Nemotron-Parse-MIP** | **0.109** | **0.958** |
| **Nemotron-Parse-TC-MIP** | **0.121** | **0.949** |

**50% 이상 WER 개선**[1]

#### 3.2 GOT 벤치마크

| 추출기 | OCR/F1 ↑ | 읽기순서/편집거리 ↓ |
|--------|---------|------------------|
| Gemini Flash 2.0 | 0.9915 | 0.0125 |
| **Nemotron-Parse-1.1** | **0.9785** | **0.014** |
| Nemotron-Parse-1.1-TC | 0.9755 | 0.014 |

경량 모델 범주에서 최고 성능[1]

#### 3.3 OmniDocBench v1.0

| 모델 | 토큰 | 전체 | 텍스트 | 수식 | 테이블 |
|------|------|------|--------|------|--------|
| Nougat | 2352 | 0.452 | 0.365 | 0.488 | 0.572 |
| InternVL2-76B | 6790 | 0.440 | 0.353 | 0.543 | 0.547 |
| **Nemotron-Parse** | **3201** | **0.131** | **0.052** | **0.288** | **0.118** |
| **Nemotron-Parse-TC** | **833** | **0.129** | **0.055** | **0.295** | **0.121** |

(낮은 점수 = 우수 성능, 편집 거리 기반)[1]

#### 3.4 테이블 추출

| 벤치마크 | Nemotron-Parse | TC |
|---------|----------------|-----|
| RD-TableBench | 86.2 TEDS | 85.3 TEDS |
| PubTabNet | 81.3 TEDS | 80.9 TEDS |

**Reducto (90.2) 대비 경쟁력 있는 성능**[1]

#### 3.5 다국어 성능

| 언어 | WER ↓ | F1 ↑ |
|------|-------|------|
| 영어 | 0.03 | **0.98** |
| 중국어 | 0.03 | **0.98** |
| 일본어 | 0.03 | **0.98** |
| 독일어 | 0.06 | 0.96 |

**모든 언어에서 F1 > 0.96**[1]

#### 3.6 추론 속도

| 모델 | 처리량 (토큰/초) | 페이지 처리 |
|------|-----------------|-----------|
| Nemotron-Parse | 3800 | 4 페이지/초 |
| **Nemotron-Parse-TC** | **4500** | **5 페이지/초** |

20% 속도 개선[1]

***

### 4. 모델 일반화 성능 향상: 핵심 분석

#### 4.1 위치 인코딩 제거의 길이 일반화 효과

**이론적 근거**: NoPE 기반 트랜스포머는 암묵적 위치 정보를 통해 길이 일반화 향상[1]

$$\text{Length Generalization} \propto \frac{1}{\text{Interpolation Error}}$$

**기존 위치 인코딩의 문제**[2]
- 절대 위치 임베딩: 훈련 길이로 고정, 외삽 오류 증가
- 상대 위치 인코딩: 보간 문제 발생
- ALiBi, RoPE: 다운스트림 태스크에서 길이 일반화 미흡

**NoPE의 우월성**: Kazemnejad et al. (2023) 연구에서 NoPE가 다른 모든 위치 인코딩 방식을 능가했으며, Zuo et al. (2025)은 인과 마스킹만으로 위치 정보를 복구할 수 있음을 증명[3][2]

#### 4.2 OCR 특화 설계

**2D 시각 정보 활용**:[1]

$$\text{Implicit Position} = \text{RADIO Encoder} + \text{2D Layout Structure}$$

- 비전 인코더가 이미 문서 2D 공간 구조 포함
- 1D 위치 신호 추가 시 간섭 가능성 제거
- 멀티모달 OCR에서 명시적 위치 인코딩 불필요

#### 4.3 다국어 일반화

22.9M 다양한 데이터로 학습:[1]
- 합성 데이터: 레이아웃 다양성
- 실제 데이터: 노이즈 및 품질 변동성
- 다국어 데이터: 문자 다양성

$$\text{Diversity} = |Domain| \times |Layout| \times |Language| \times |Quality|$$

**결과**: 모든 언어에서 F1 > 0.96, 도메인 외 일반화 능력 향상[1]

#### 4.4 토큰 압축의 일반화 효과

**역설**: 더 짧은 토큰 길이가 일반화 개선[1]

$$\text{Attention Complexity} = O(N^2) \rightarrow O(0.26N^2)$$

- 더 효율적인 주의 메커니즘
- 장거리 의존성 학습 개선
- OmniDocBench에서 기본 모델보다 높은 성능

***

### 5. 모델의 한계

#### 5.1 기술적 한계

1. **멀티토큰 검증 부재**: 다중 토큰 예측에서 오류 누적 가능성[1]
2. **NoPE의 한계**: 특정 길이 이상에서는 성능 저하[1]
3. **합성 데이터 의존성**: 텍스트 밀집 이미지가 합성 데이터로만 학습[1]

#### 5.2 평가 관련 한계

1. **마크다운 대 LaTeX**: 수식이 마크다운으로 표현되면 OmniDocBench에서 과소평가[1]
2. **데이터셋 편향**: 대부분 과학 논문 (arXiv) 기반[1]
3. **비교 대상 이질성**: Gemini Flash와 직접 비교 어려움 (규모 차이)[1]

#### 5.3 확장성 한계

1. **다국어 제한**: 중국어, 일본어는 과학 도메인 중심, 야생 이미지에서 성능 저하[1]
2. **특수 도메인**: 의료, 법률 등 특화된 훈련 없음[1]
3. **지역화 미지원**: 우-좌 언어 (아랍어, 히브리어) 미지원[1]

***

### 6. 앞으로의 연구 영향 및 고려사항

#### 6.1 위치 인코딩 연구의 재조명

**기존 가정의 재검토**:

NoPE 성공은 위치 인코딩이 반드시 필요하지 않다는 증명[1]

**최신 연구 (2023-2025)**[4][5][6][2]
- **Kazemnejad et al. (2023)**: NoPE가 길이 일반화에서 우월
- **Zuo et al. (2025)**: 인과 마스킹 → 위치 정보 복구
- **Wang et al. (2024)**: NoPE 기반 언어 모델 성공
- **PRISM (2025)**: 확률적 상대 위치 임코딩으로 10배 길이 외삽

**새로운 방향**:
1. 암묵적 위치 학습 메커니즘 탐구
2. 도메인-특화 위치 표현 설계
3. 시각-언어 작업에서 위치 역할 재정의

#### 6.2 멀티토큰 예측의 발전

**Nemotron의 멀티토큰 전략**:

$$P(t_{i+1}, t_{i+2}, \ldots, t_{i+m} | \text{context})$$

**관련 연구 (2024-2025)**[7][8]
- **Gloeckle et al. (2024)**: 생성 속도 3배 향상
- **Apple (2025)**: 코드/수학에서 5배 가속
- **SGLang (2025)**: 투기적 디코딩으로 60% 처리량 증가

**앞으로의 과제**:
1. 토큰 검증 메커니즘 개선
2. 오류 전파 완화 전략
3. 멀티토큰 손실 함수 최적화

#### 6.3 엔드-투-엔드 VLM 설계 패러다임

**기술 트리**:
- 파이프라인 기반 (정확, 느림)
- 엔드-투-엔드 VLM (빠르지만 미흡)
- **Nemotron-Parse 1.1 (균형)**
- 차세대 (더 경량 + 더 강력)

**2025년 경쟁 모델들**[9][10][11]
- **SmolDocling**: 27배 작지만 유사 성능
- **Infinity Parser**: 강화학습 기반 레이아웃 최적화
- **MonkeyOCR v1.5**: 복잡 패턴 및 페이지 교차 테이블 처리
- **PaddleOCR-VL**: 109개 언어 지원

#### 6.4 연구 시 고려할 점

**기술적 개선**:
1. 조건부 위치 정보 (문서 타입, 예상 길이에 따른 PE)
2. 토큰 검증 메커니즘: $$P(\text{accept } t_{i+j} | \text{previous tokens}) > \tau$$
3. 도메인 특화 모델 (의료, 법률, 금융)

**평가 프레임워크 확장**:
1. 균형 잡힌 벤치마크 (상업 문서, 야생 이미지, 다양한 품질)
2. 구조적 복잡도 평가: $$\text{Complexity Index} = f(\text{columns}, \text{nested tables}, \text{density})$$

**실용적 배포**:
1. 엣지 디바이스 지원 (100-200M 파라미터 모델)
2. 실시간 처리 (<100ms/페이지)
3. 사용자 피드백 통합: $$\text{Model}\_{\text{v2}} = \text{Fine-tune}(\text{Model}_{\text{v1}}, \text{UserFeedback})$$

**크로스 도메인 일반화**:
1. 도메인 불변 표현 학습
2. Few-shot 전이 학습
3. 메타-인-컨텍스트 학습

#### 6.5 멀티모달 통합

**확장 영역**:
1. 차트 해석 및 수치 추출
2. 다이어그램 의미 파악
3. 이미지-텍스트 관계 모델링
4. 비디오 문서 처리 (스크린샷 시퀀스)

***

### 7. 최신 관련 연구 (2020-2025)

#### 7.1 엔드-투-엔드 VLM 기반 문서 파싱

| 연도 | 모델 | 주요 특징 |
|------|------|---------|
| 2024 | Nougat | 학술 문서 특화, 마크다운 포맷[12] |
| 2024 | DocOwl 1.5 | 구조 학습, 5개 도메인[13] |
| 2024 | GOT | 일반 OCR 이론, 256 토큰[14] |
| 2025 | **Nemotron-Parse 1.1** | **NoPE, 경량, 다국어**[1] |
| 2025 | SmolDocling | 초경량 VLM, 27배 작음[15] |
| 2025 | MonkeyOCR v1.5 | 복잡 패턴, 강화학습[9] |
| 2025 | Infinity Parser | 레이아웃 RL, 55K 고충실도 데이터[16] |

#### 7.2 위치 인코딩과 길이 일반화 연구

| 연도 | 연구 | 주요 결론 |
|------|------|---------|
| 2023 | Kazemnejad et al. | NoPE > PE 기반 모든 방식[2] |
| 2024 | Chen et al. | 위치 결합 (Position Coupling)[17] |
| 2024 | Wang et al. | NoPE 기반 언어 모델 성공[18] |
| 2025 | Zuo et al. | 인과 마스크로 위치 복구[19] |
| 2025 | PRISM | 확률적 상대 위치, 10배 외삽[5] |

#### 7.3 멀티토큰 예측 연구

| 연도 | 연구 | 성과 |
|------|------|------|
| 2024 | Gloeckle et al. | 생성 3배 가속[7] |
| 2024 | Meta | 다중 예측 헤드 구조 |
| 2025 | SGLang | 투기적 디코딩, 60% 처리량 증가[8] |
| 2025 | Apple | 코드/수학 5배 가속[20] |

#### 7.4 문서 이해의 최신 방향

**의미론적 레이아웃 분석**[21]
- SCAN (2025): VLM-친화적 의미론적 분할

**크로스 도메인 이해**[22][23][24]
- 금융 문서 파이프라인 (2025)
- 클리닉 데이터 추출 (2025)
- 로직스 파싱 벤치마크 (2025)

**벤치마크 진화**[25][26]
- OmniDocBench v1.5 (2025): 포괄적 평가
- CC-OCR (2024): 다양한 시나리오
- LogicsParsingBench (2025): 복잡 레이아웃

***

### 8. 종합 결론

**Nemotron-Parse-1.1**은 문서 파싱 및 OCR 분야에서 **경량성과 성능의 최적 균형**을 달성한 모델이다. 특히:

1. **NoPE의 성공**: 위치 인코딩이 반드시 필요하지 않다는 혁신적 증명
2. **멀티토큰 효과**: 훈련 신호 풍부화로 단일-토큰 설정에서도 성능 개선
3. **실용적 배포**: 885M 파라미터로 H100에서 4-5 페이지/초 처리
4. **다국어 지원**: 7개 언어에서 F1 > 0.96 달성

**미래 연구 방향**:
- 조건부 위치 정보 설계
- 토큰 검증 메커니즘
- 도메인 특화 모델 개발
- 엣지 배포 최적화

완전 공개 모델로서 이 연구는 문서 AI 커뮤니티의 접근성 향상과 다음 세대 모델 개발의 기초가 될 것으로 예상된다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d9ad5186-e533-423f-9afa-917643a17d92/2511.20478v1.pdf)
[2](https://arxiv.org/abs/2305.19466)
[3](https://arxiv.org/pdf/2305.19466.pdf)
[4](https://arxiv.org/abs/2410.12413)
[5](https://arxiv.org/abs/2506.00920)
[6](https://aclanthology.org/2025.acl-long.235)
[7](https://bdtechtalks.substack.com/p/multi-token-prediction-speeds-up)
[8](https://www.emergentmind.com/topics/multi-token-prediction-technologies)
[9](https://www.semanticscholar.org/paper/5c94d82e55a1778ae2c976b87988297aee2cb7ba)
[10](https://arxiv.org/abs/2510.19817)
[11](https://arxiv.org/abs/2509.19760)
[12](http://arxiv.org/pdf/2410.21169.pdf)
[13](https://arxiv.org/abs/2403.12895)
[14](https://arxiv.org/abs/2508.13238)
[15](https://arxiv.org/html/2503.11576v1)
[16](https://arxiv.org/abs/2506.03197)
[17](http://www.proceedings.com/079017-0701.html)
[18](https://aclanthology.org/2024.findings-acl.834.pdf)
[19](https://www.semanticscholar.org/paper/dcf78544177e7c94afeb34b16456e2786eed5a62)
[20](https://arxiv.org/html/2507.11851v1)
[21](https://arxiv.org/html/2505.14381v1)
[22](https://arxiv.org/abs/2504.20220)
[23](https://arxiv.org/abs/2510.23066)
[24](https://ijamjournal.org/ijam/publication/index.php/ijam/article/view/163)
[25](https://arxiv.org/html/2412.02210v3)
[26](https://openaccess.thecvf.com/content/CVPR2025/papers/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.pdf)
[27](https://arxiv.org/abs/2406.11251)
[28](https://arxiv.org/html/2501.15558)
[29](https://arxiv.org/html/2407.12594v1)
[30](http://arxiv.org/pdf/2309.05756.pdf)
[31](https://arxiv.org/abs/2502.18443)
[32](https://arxiv.org/html/2502.04223v1)
[33](https://arxiv.org/html/2510.14528v2)
[34](https://pubmed.ncbi.nlm.nih.gov/39796994/)
[35](https://www.semanticscholar.org/paper/Document-Layout-Analysis-Binmakhashen-Mahmoud/fc005e255a6a74614b6ff7e7f530a0474fba5510)
[36](https://anyparser.com/blog/future-trends-in-document-parsing/)
[37](https://github.com/microsoft/table-transformer)
[38](https://photes.io/blog/posts/ocr-research-trend)
[39](https://arxiv.org/abs/2409.05125)
[40](https://aclanthology.org/2024.acl-long.162.pdf)
[41](https://visionvix.com/best-llm-for-ocr/)
[42](https://openaccess.thecvf.com/content/CVPR2022/papers/Smock_PubTables-1M_Towards_Comprehensive_Table_Extraction_From_Unstructured_Documents_CVPR_2022_paper.pdf)
[43](https://arxiv.org/abs/2305.16843)
[44](https://arxiv.org/abs/2406.01895)
[45](http://www.proceedings.com/079017-0838.html)
[46](https://ieeexplore.ieee.org/document/10888904/)
[47](http://arxiv.org/pdf/2410.02140.pdf)
[48](https://arxiv.org/html/2405.14722v1)
[49](http://arxiv.org/pdf/2402.09371v1.pdf)
[50](https://aclanthology.org/2021.emnlp-main.236.pdf)
[51](http://arxiv.org/pdf/2104.08698.pdf)
[52](https://www.aclweb.org/anthology/2020.coling-main.319.pdf)
[53](http://arxiv.org/pdf/2406.01895.pdf)
[54](https://ceur-ws.org/Vol-2764/paper5.pdf)
[55](https://openreview.net/forum?id=bppDDqbO3V)
[56](https://arxiv.org/html/2405.08586v1)
[57](https://github.com/McGill-NLP/length-generalization)
[58](https://proceedings.mlr.press/v189/frikha23a/frikha23a.pdf)
[59](https://lmsys.org/blog/2025-07-17-mtp/)
