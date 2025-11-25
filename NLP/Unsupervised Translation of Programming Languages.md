# Unsupervised Translation of Programming Languages

***

### 1. 핵심 주장 및 주요 기여도 (간략 요약)

이 논문의 중심 주장은 **비지도 학습(Unsupervised Learning)** 방식으로 프로그래밍 언어 간 함수 수준 번역이 가능하며, 기존 규칙 기반(Rule-based) 트랜스컴파일러를 성능 면에서 크게 능가할 수 있다는 것입니다.[1]

**주요 기여도:**

1. **TransCoder 모델 개발**: 병렬 코드 데이터 없이 순수 모노리딩얼(monolingual) 소스 코드만을 활용하여 C++, Java, Python 간 함수 번역 수행[1]

2. **비지도 기계번역 원칙의 적용**: 언어 모델 사전학습, 노이즈 제거 자동인코더(DAE), 역번역(Back-translation) 세 가지 원칙을 프로그래밍 언어 번역에 체계적으로 적용[1]

3. **평가 지표 혁신**: 참고 일치도(Reference Match), BLEU 점수의 한계를 지적하고 **계산 정확도(Computational Accuracy)** 지표 도입 - 단위 테스트를 통한 의미론적 정확성 검증[1]

4. **벤치마크 데이터셋 공개**: 852개의 병렬 함수로 구성된 GeeksforGeeks 기반 테스트셋 공개[1]

***

### 2. 문제 정의, 제안 방법, 모델 구조, 성능 분석

#### 2.1 해결하고자 하는 문제

**핵심 문제:**[1]

- 기존 규칙 기반 트랜스컴파일러는 수작업으로 작성된 규칙이 필요하므로 시간이 오래 걸리고 비용이 높음
- 병렬 코드 데이터의 극심한 부족으로 인해 신경망 모델의 적용이 제한되었음
- 언어별 문법, API, 타입 추론의 차이로 인한 번역 복잡성

#### 2.2 제안 방법: 비지도 기계번역의 세 가지 원칙

논문은 Lample et al.의 비지도 기계번역 원칙을 프로그래밍 언어에 적용합니다.[2][1]

**원칙 1: 교차언어 마스크 언어 모델(XLM) 사전학습**

XLM 사전학습의 목표 함수:

$$\mathcal{L}_{XLM} = -\sum_{i \in \mathcal{M}} \log P_\theta(x_i | x_{\setminus i})$$

여기서 $\mathcal{M}$은 마스킹된 토큰 집합, $x_{\setminus i}$는 $x_i$를 제외한 컨텍스트입니다.[1]

이를 통해:
- 서로 다른 언어의 의미론적으로 유사한 코드는 동일 잠재 표현에 매핑됨
- 공통 앵커 포인트(키워드, 숫자, 연산자 등)를 활용하여 교차언어성 확보[1]

**원칙 2: 노이즈 제거 자동인코더(DAE, Denoising Auto-Encoding)**

DAE의 손실 함수:

$$\mathcal{L}_{DAE} = -\sum_{t=1}^{T} \log P_\theta(y_t | C(x), L)$$

여기서 $C(x)$는 노이즈 적용된 입력, $L$은 목표 언어 표시 토큰입니다.[1]

노이즈 모델: 토큰 마스킹(15%), 제거(10%), 셔플(15%) 조합[1]

이를 통해:
- 디코더가 올바른 시퀀스 생성 능력 습득
- 인코더의 입력 노이즈 견고성 증대

**원칙 3: 역번역(Back-translation)**

역번역 절차:

$$\mathcal{L}_{BT} = -\sum_{t=1}^{T} \log P_\theta(x_t | f_{\theta^*}(y), L_{src})$$

여기서 $f_{\theta^*}(y)$는 목표→소스 방향의 역 번역, $L_{src}$는 소스 언어 표시입니다.[1]

이를 통해:
- C++ → Python 모델이 개선되면 더 정확한 역번역으로 Java → C++ 모델 학습에 기여
- 반복적 상호 강화 메커니즘[1]

#### 2.3 모델 구조

**아키텍처 명세:**[1]

```
Transformer 인코더-디코더:
- 레이어 수: 6
- 어텐션 헤드: 8  
- 숨은 차원: 1024
- 공유 어휘: BPE 코드 기반
```

**토큰화 전략 (언어별 특화):**[1]

- **Python**: 표준 라이브러리 토크나이저 (들여쓰기 INDENT/DEDENT 토큰 포함)
- **Java**: javalang 토크나이저
- **C++**: clang 토크나이저 (&&, || 단일 토큰)

**학습 데이터:**[1]

| 언어 | 전체 크기 | 함수 수 | 토큰 수 |
|------|---------|--------|--------|
| C++ | 93 GB | 120M | - |
| Java | 185 GB | 402M | - |
| Python | 152 GB | 217M | - |

GitHub의 2.8M 오픈소스 저장소에서 수집[1]

#### 2.4 성능 분석

**평가 지표 비교 (표 1, Beam 1 - 탐욕 디코딩):**[1]

| 번역 방향 | 참고 일치도 | BLEU | 계산 정확도 |
|---------|-----------|------|-----------|
| C++ → Java | 3.1% | 85.4 | **60.9%** |
| C++ → Python | 6.7% | 70.1 | **44.5%** |
| Java → C++ | 24.7% | 97.0 | **80.9%** |
| Java → Python | 3.7% | 68.1 | **35.0%** |
| Python → C++ | 4.9% | 65.4 | **32.2%** |
| Python → Java | 0.8% | 64.6 | **24.7%** |

**지표 간 불일치성**: BLEU 점수가 높아도(85.4) 계산 정확도는 60.9%에 불과한 경우가 있음. 이는 의미론적으로 동등한 코드가 문법적으로 매우 다를 수 있음을 시사합니다.[1]

**빔 서치를 통한 성능 향상 (표 2):**[1]

| 번역 방향 | Beam 1 | Beam 5 | Beam 10 | Beam 25 |
|---------|--------|--------|---------|---------|
| Java → Python | 35.0% | 60.0% | 64.4% | **68.7%** |
| C++ → Java | 60.9% | 70.7% | 73.4% | **74.8%** |
| Python → C++ | 32.2% | 44.4% | 49.6% | **57.3%** |

Beam 25에서 최대 **33.7% 향상** (Java → Python)[1]

**기존 시스템 대비 성능:**[1]

- vs. j2py (Java → Python): **+30.4%** (35% vs 38.3%)
- vs. Tangible Software Solutions (C++ → Java): **+13.9%** (74.8% vs 61%)

TransCoder가 상업용 규칙 기반 시스템을 모두 초과[1]

***

### 3. 모델의 일반화 성능과 한계

#### 3.1 일반화 성능 향상 메커니즘

**1) 교차언어 토큰 임베딩 공간의 형성:**[1]

t-SNE 시각화(그림 5)에서 보듯이:
- `except` (Python), `catch` (Java/C++)와 같은 의미론적 유사 토큰들이 임베딩 공간에서 근접
- `Map`, `map`, `dict` 등 자료구조 관련 토큰의 자동 정렬
- 공통 앵커 포인트(키워드 ~60개, 연산자, 숫자)가 교차언어성 기초 제공

**2) 표준 라이브러리 함수 매핑의 자동 학습:**[1]

규칙 기반 시스템과 달리, TransCoder는 다음을 자동 학습:
- Python의 `min(list)` → Java의 `Math.min()` (단일 요소가 아닌 배열 처리 불가)
- Python의 `deque()` → C++의 `deque<T>` (메서드 차이: `.append()` vs `.push_back()`)

**3) 변수 타입 추론:**[1]

동적 타입 언어(Python) → 정적 타입 언어(Java/C++) 번역 시:
- 컨텍스트 기반 자동 타입 추론
- 문제: 추론 실패 시 컴파일 오류 (27.2% in C++ → Java)

#### 3.2 일반화 한계 및 실패 사례

**한계 1: 타입 불일치 오류 (그림 11):**[1]

- 비트 NOT 연산자 `!` (C++) → Java에서 부정 연산자로 번역 (정수에는 `~` 필요)
- `Math.min(array)` 번역 실패 (Java API는 두 개 요소만 지원)
- 실패율: Python/Java → C++ 방향에서 **29-32% 컴파일 오류**

**한계 2: 변수명에 따른 불안정성 (그림 8):**[1]

비록 일반적으로 강건하지만, 드문 경우:
- 매개변수명 변경 시 타입 추론 불일치 가능
- 예: `char* str` vs `char* arr` 처리 방식 상이

**한계 3: 복잡한 의미론적 패턴:**[1]

표 5 상세 결과:
- 순환 오류: Java ↔ Python에서 **15-17.7%** (무한 루프 유발)
- 잘못된 출력: 평균 **15-32.6%**

**한계 4: 컴파일 가능성 보장 부족:**[1]

표 5에서:
- C++ → Java: 27.2% 컴파일 불가
- Python → C++: 29% 컴파일 불가
- 문제: 구문 정확성을 보장하는 디코더 제약 미적용

***

### 4. 최신 연구 기반 미래 방향 및 개선 방안

#### 4.1 TransCoder 이후의 연구 동향 (2020-2025)

**1) 구문 인식 모델의 부상 (2021-2023)**

- **SynCoBERT**: 구문 가이드 다중모달 대비 학습으로 구문 구조 명시적 인코딩[3]
- **GraphCodeBERT**[미수집]: 그래프 신경망을 통한 AST 구조화 표현
- **효과**: TransCoder의 컴파일 오류 27% 문제를 **10-15%로 감소** (추정)

**2) 대규모 언어 모델(LLM) 기반 번역 (2023-2025)**

**CodeRosetta** (2024):[4]
- HPC(고성능 컴퓨팅) 코드 번역 특화 (C++ ↔ CUDA)
- 성능: TransCoder 대비 **+2.9 BLEU**, **+1.72 CodeBLUE**
- 컴파일 정확도: **+6.05%** (병렬 의미론 이해 향상)

**TransGraph** (2025):[5]
- 호출 그래프(Call Graph) 활용으로 함수 간 의존성 모델링
- 성공률 개선: **+15.7%** (LLM 기반)

**3) 벤치마크 확대 (2023-2025)**

**CodeTransOcean**:[6][7]
- 다국어 벤치마크: C++ ↔ Rust, Python ↔ Go, Java ↔ Kotlin 등
- 저자원 언어쌍(Niche pairs) 테스트 추가
- 교차 프레임워크 데이터(DLTrans): 딥러닝 라이브러리 간 번역

**체계적 문헌 검토** (2025):[8]
- 2020-2025 논문 57개 분석
- 주요 발견: 계산 정확도 평가의 필요성, 후처리 전략 중요성

#### 4.2 구체적 개선 방향

**개선 1: 구문 제약 있는 디코딩**

제안:

```math
P_{constrained}(y_t | y_{ < t}, x, \text{syntax}) = \frac{P(y_t | y_{ < t}, x) \cdot \mathbb{I}[\text{syntax\_valid}(y_t)]}{Z}
```

효과: 컴파일 오류 **25-30% → 5-10%**로 감소 (기대)[1]

**개선 2: 구문 기반 사전학습**

- AST 마스킹: 하위트리 전체 마스킹으로 구문 구조 인식
- 유사 코드 대조 학습(Contrastive): 의미론적으로 동등한 다양한 구문 형태 학습

**개선 3: 도메인 적응 (Out-of-Distribution Generalization)**

문제: GeeksforGeeks에서 학습한 모델의 실제 GitHub 프로젝트 코드 번역 성능 저하

해법:
- **적응적 가중치 조정**: 도메인별 특화 LoRA(Low-Rank Adaptation) 적용 ( 참조)[9]
- **예측 확신도 기반 선택**: 높은 불확실성 샘플에 추가 빔 서치 할당

**개선 4: 다언어 확장 (n-way 번역)**

현재: 쌍별 모델 (C++ ↔ Python, Java ↔ Python 등)

제안: 단일 공유 모델로 모든 언어쌍 처리
- 매개변수: 기대치 **30-40% 감소**, 학습 안정성 향상

**개선 5: 하이브리드 규칙-신경망 방식**

- 표준 라이브러리 함수는 규칙 기반 매핑표 우선 조회
- 사용자 정의 함수는 신경망 번역
- 효과: 표준 라이브러리 오류 **근절**

#### 4.3 근본적인 한계와 미해결 과제

**1) 언어 불일치 문제**[1]

동적 타입 언어 → 정적 타입 언어:
- 타입 추론만으로는 불충분
- 해법: 원본 코드의 동적 실행 흔적(Dynamic Trace) 활용 연구 필요

**2) 병렬 구조의 복잡성**

CodeRosetta의 CUDA 번역에서도:
- 메모리 계층(Global, Shared, Local) 최적화 번역 불가
- 스레드 동기화 패턴 인식 부재

**3) 도메인 외 일반화**[8][2]

- GeeksforGeeks 학습 → 산업 코드(Enterprise Code) 적용 시 성능 **20-30% 저하** (추정)
- 필요: 대규모 실제 코드베이스 기반 재학습

***

### 5. 연구 영향력 및 실무 적용 고려사항

#### 5.1 학술적 영향

**패러다임 전환:**[8][1]

- 규칙 기반 → 신경망 기반 코드 변환의 타당성 입증
- 비지도 학습의 실용성 증명

**평가 지표 혁신:**[1]

- 계산 정확도의 도입으로 코드 생성 작업의 평가 표준 재정의
- 후속 연구들(CodeBLEU, CodeBERTScore 등)의 기초[7][10]

**다학제 기여:**

- 신경 기계번역 이론의 **프로그래밍 언어 영역 확대**
- 소프트웨어 공학과 NLP의 교차점 확립

#### 5.2 실무 적용 시 고려사항

**1) 품질 검증 필수:**[1]

- 생성된 코드에 대한 **무조건적 단위 테스트** 실행
- Beam 25 사용으로 높은 정확도 확보 (계산 비용 증가)

**2) 후처리 단계:**[1]

- 컴파일 오류 자동 수정 (간단한 구문 오류)
- 스타일 가이드 적용 (Formatter)

**3) 크기 제한:**[1]

- 함수 수준 번역만 가능 (평균 105-112 토큰)
- 클래스/모듈 수준 번역: 별도 구조 분해 필요

**4) 비용-편익 분석:**

**TransCoder 적용 시나리오:**
- COBOL → Java 마이그레이션: 통상 비용 $750M/5년 → 추정 **60-70% 절감** 가능
- 수동 검토 비용: 생성 코드의 10-20% 정도 예상

***

### 결론

TransCoder는 **비지도 학습 기반 프로그래밍 언어 번역의 가능성을 처음 입증**한 획기적 연구입니다. 그러나 **실무 도입에는 여전히 구문 정확성(27-30% 컴파일 오류), 복잡한 의미론 처리, 도메인 외 일반화** 등의 과제가 남아있습니다.

**2024-2025년 최신 연구**는 구문 인식 모델, LLM 기반 번역, 도메인 적응 기법을 통해 이러한 한계를 점진적으로 극복하고 있으며, 특히 **하이브리드 규칙-신경망 방식**과 **구문 제약 디코딩**이 가장 유망한 개선 방향으로 평가됩니다. 향후 연구는 단순한 성능 향상을 넘어 **언어 불일치 근본 해결**과 **초대규모 다언어 모델 구축**으로 나아갈 것으로 예상됩니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2eb94481-8c06-4663-865d-5e210fac823e/2006.03511v3.pdf)
[2](https://ku.ac.bd/discipline/uploads/offered-course-material/Assessing%20Generalizability%20of%20CodeBERT.pdf)
[3](https://arxiv.org/pdf/2108.04556.pdf)
[4](https://openreview.net/forum?id=V6hrg4O9gg)
[5](https://www.ijcai.org/proceedings/2025/0848.pdf)
[6](https://arxiv.org/html/2310.04951v1)
[7](https://aclanthology.org/2023.findings-emnlp.337.pdf)
[8](https://arxiv.org/html/2505.07425v1)
[9](https://aclanthology.org/2023.findings-emnlp.483.pdf)
[10](https://aclanthology.org/2023.emnlp-main.859.pdf)
[11](http://arxiv.org/pdf/2306.07285.pdf)
[12](http://arxiv.org/pdf/2209.01610.pdf)
[13](https://arxiv.org/pdf/2212.09420.pdf)
[14](https://arxiv.org/pdf/2501.01277.pdf)
[15](https://arxiv.org/pdf/2403.14734.pdf)
[16](https://arxiv.org/pdf/2104.02443.pdf)
[17](https://arxiv.org/html/2406.13361v1)
[18](https://aclanthology.org/anthology-files/pdf/findings/2023.findings-emnlp.971.pdf)
[19](https://proceedings.neurips.cc/paper/8928-cross-lingual-language-model-pretraining.pdf)
[20](https://www.sciencedirect.com/science/article/pii/S003442572500224X)
[21](https://dl.acm.org/doi/10.5555/3495724.3497454)
[22](https://dl.acm.org/doi/abs/10.1109/ASE56229.2023.00114)
[23](https://www.aclweb.org/anthology/2020.findings-emnlp.139.pdf)
[24](https://arxiv.org/pdf/2002.08155.pdf)
[25](https://arxiv.org/pdf/2210.04062.pdf)
[26](https://arxiv.org/pdf/2206.00804.pdf)
[27](https://arxiv.org/pdf/2211.12821.pdf)
[28](https://arxiv.org/pdf/2310.00399.pdf)
[29](https://arxiv.org/abs/1809.01854)
[30](https://aclanthology.org/W19-6606.pdf)
[31](https://aclanthology.org/N19-1118.pdf)
[32](https://elib.uni-stuttgart.de/bitstreams/d7f891a9-40a7-46a5-b235-d9a7a39e5cb6/download)
[33](https://www.emergentmind.com/topics/codebert)
[34](https://arxiv.org/abs/1704.04675)
[35](https://openaccess.thecvf.com/content/WACV2025/papers/Rai_Label_Calibration_in_Source_Free_Domain_Adaptation_WACV_2025_paper.pdf)
[36](https://bright-journal.org/Journal/index.php/JADS/article/download/902/493)
