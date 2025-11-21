# ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs

### 핵심 요약

**ToolLLM**은 오픈소스 대형 언어모델(LLM)의 도구 활용 능력 격차를 해소하기 위해 설계된 포괄적 프레임워크입니다. 본 논문의 핵심 주장과 기여는 다음과 같습니다:[1]

**주요 주장:**
- 현재 오픈소스 LLM(예: LLaMA)은 기본 언어 작업에 중점을 두었으나, 실제 API를 활용한 도구 사용 능력은 현저히 부족합니다.
- ChatGPT와 같은 폐쇄형 최신 LLM이 우수한 도구 활용 능력을 보유하고 있음에도 불구하고, 그 메커니즘은 불명확합니다.
- 기존 도구 학습 데이터셋은 API 다양성 제한, 단일 도구만 고려, 부족한 추론 능력 등의 한계가 있습니다.

**핵심 기여:**
1. **ToolBench 데이터셋**: 16,464개의 실제 REST API를 포함한 대규모 명령어 튜닝 데이터셋을 구성합니다.
2. **DFSDT(깊이 우선 탐색 기반 결정 트리)**: 기존 ReACT의 한계를 극복하는 새로운 추론 전략을 제안합니다.
3. **ToolLLaMA**: 미세조정된 LLaMA 모델로 ChatGPT 수준의 성능을 달성합니다.
4. **ToolEval**: 도구 사용 능력 자동 평가 시스템을 개발합니다.
5. **API 검색기**: 대규모 API 풀에서 관련 API를 효율적으로 추천하는 신경망 기반 검색기를 제공합니다.[1]

***

### II. 문제 정의 및 제안 방법

#### 2.1 해결하고자 하는 문제

**문제의 3가지 핵심 차원:**

| 문제 영역 | 기존 연구의 한계 | ToolLLM의 해결방안 |
|---------|-----------|---------|
| **API 다양성** | 현실적 API 부족(예: 3-400개) 또는 시뮬레이션된 API | 16,464개 실제 REST API, 49개 카테고리 포함 |
| **시나리오 복잡성** | 단일 도구만 지원, 다중 도구 시나리오 미지원 | 단일/다중 도구 지원, 실제 API 응답 활용 |
| **추론 능력** | CoT/ReACT의 제한적 탐색 | DFSDT로 다중 추론 경로 탐색 가능 |

#### 2.2 ToolBench 데이터셋 구축: 3단계 프로세스

**단계 1: API 수집**
- RapidAPI Hub에서 10,853개 도구(53,190개 API)를 초기 수집
- 엄격한 필터링 과정(API 기능성 테스트, 응답 품질 평가)을 통해 최종 3,451개 도구(16,464개 API)로 정제[1]

**단계 2: 명령어 생성**
3가지 유형의 명령어를 생성합니다:

$$I_1 \text{: 단일 도구 명령어 (Single-tool instructions)}$$
$$I_2 \text{: 범주 내 다중 도구 명령어 (Intra-category multi-tool instructions)}$$
$$I_3 \text{: 컬렉션 내 다중 도구 명령어 (Intra-collection multi-tool instructions)}$$

생성 과정의 수식화:[1]

$$\text{ChatGPT}_{\{API_1,\cdots,API_N\} \in S_{API}, \{seed_1,\cdots,seed_3\} \in S_{seed}} \rightarrow ([S^{rel}_1, Inst_1], \cdots, [S^{rel}_{N'}, Inst_{N'}])$$

여기서 $S_{rel} \subset S^{sub}_N$는 각 명령어에 해당하는 관련 API 집합입니다.

**샘플링 전략:**
- 단일 도구: 각 도구 순회
- 다중 도구: 같은 카테고리/컬렉션에서 2-5개 도구 랜덤 선택

결과: 약 200,000개의 (명령어, 관련 API) 쌍이 생성되었습니다 (I1: 87,413, I2: 84,815, I3: 25,251).[1]

**단계 3: 해결책 경로 주석 (Solution Path Annotation)**
각 명령어에 대해 유효한 API 호출 시퀀스를 찾습니다:

$$\text{ChatGPT}(a_t | \{a_1, r_1, \cdots, a_{t-1}, r_{t-1}\}, Inst^*) \rightarrow a_t$$

여기서 $a_t$는 다음 행동(API 호출)이고, $r_t$는 API 응답입니다.

#### 2.3 DFSDT(깊이 우선 탐색 기반 결정 트리)

**기본 개념:**
기존 ReACT 방식의 한계:[1]
- **오류 전파**: 한 번의 잘못된 행동이 이후 모든 결정에 영향을 미침
- **제한적 탐색**: 하나의 추론 경로만 탐색하여 복잡한 작업 실패

**DFSDT 알고리즘의 특징:**

$$\text{Decision Tree} = \{(\text{Action}, \text{Result}), (\text{Action}, \text{Result}), \cdots\}$$

1. **다중 경로 탐색**: 모든 가능한 추론 경로를 트리 형태로 구성
2. **유연한 선택**: 
   - 유망한 경로를 따라 계속 진행
   - 막히면 새로운 노드를 확장하여 다시 시도
3. **Pre-order DFS 변형**: 자식 노드 정렬 단계를 생략하여 비용 최적화

실험 결과 DFSDT는 ReACT 대비 현저히 개선된 성능을 달성했습니다:[1]

| 명령어 유형 | ReACT | ReACT@N | DFSDT |
|-----------|--------|---------|-------|
| I1 (단일 도구) | 37.8% | 49.4% | 58.0% |
| I2 (다중 도구) | 40.6% | 49.4% | 70.6% |
| I3 (복합 다중) | 27.6% | 34.6% | 62.8% |

#### 2.4 ToolEval: 자동 평가 시스템

두 가지 핵심 메트릭:[1]

**1) Pass Rate (통과율)**
- 정의: 제한된 예산 내에서 명령어를 성공적으로 완료할 확률
- 평가 기준: 해결책 경로가 다음 조건 중 하나를 만족
  - 모든 API를 시도한 후 유용한 정보를 획득하지 못한 경우
  - 최종 답변이 원래 명령어를 완전히 해결한 경우

**2) Win Rate (승률)**
- 정의: ChatGPT-ReACT와 비교하여 더 나은 해결책의 비율
- 평가 기준:
  - 정보 풍부도: 최종 답변이 필요한 모든 정보 포함
  - 사실성: 수행 내용의 정확한 설명
  - 추론: 실패 시 상세한 원인 제시

**신뢰성 검증:**
ToolEval은 인간 평가와 87.1% (Pass Rate), 80.3% (Win Rate)의 높은 일치도를 보였습니다.[1]

***

### III. 모델 구조 및 성능

#### 3.1 ToolLLaMA 구조

**기본 구성:**
- 기저 모델: LLaMA-2 7B
- 컨텍스트 길이 확장: 4096 → 8192 (위치 보간 사용)
- 훈련 방식: 감독 미세조정(Supervised Fine-Tuning)

**훈련 하이퍼파라미터:**[1]
- Learning rate: $5 \times 10^{-5}$
- Warmup ratio: $4 \times 10^{-2}$
- Batch size: 64
- Max sequence length: 8192
- Position interpolation ratio: 2
- Training epochs: 2

#### 3.2 API 검색기

**구조:** Sentence-BERT (BERT-BASE 기반 문장 임베딩)

**성능 비교:**[1]

| 방법 | I1-NDCG@1 | I1-NDCG@5 | I2-NDCG@1 | I2-NDCG@5 | I3-NDCG@1 | I3-NDCG@5 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| BM25 | 18.4 | 19.7 | 12.0 | 11.0 | 25.2 | 20.4 |
| Ada (OpenAI) | 57.5 | 58.8 | 36.8 | 30.7 | 54.6 | 46.8 |
| **Ours** | **84.2** | **89.7** | **68.2** | **77.9** | **81.7** | **87.1** |

검색기가 Oracle(정답 API)보다 더 나은 API를 제시하기도 하여 성능 개선을 확인했습니다.

#### 3.3 주요 성능 결과

**ToolLLaMA vs 기타 모델 비교:**[1]

| 모델 | 방법 | I1-Cat. Pass | I2-Cat. Pass | I3-Inst. Pass | 평균 Pass |
|------|------|-------------|-------------|-------------|----------|
| ChatGPT | DFSDT | 60.5% | 71.5% | 62.0% | **64.8%** |
| Text-Davinci-003 | DFSDT | 46.0% | 42.0% | 46.0% | 43.1% |
| Claude-2 | DFSDT | 18.5% | 20.5% | 28.0% | 22.6% |
| **ToolLLaMA** | DFSDT | **62.0%** | **77.0%** | **66.0%** | **66.7%** |
| **ToolLLaMA** | DFSDT-Retriever | **60.5%** | **68.5%** | **65.0%** | **67.3%** |
| GPT-4 | DFSDT | 67.0% | 77.5% | 71.0% | **71.1%** |

**결과 해석:**
- ToolLLaMA(DFSDT)는 ChatGPT와 거의 동등한 성능 달성
- Text-Davinci-003과 Claude-2를 뛰어넘음
- GPT-4에만 약간 미치지 못함

***

### IV. 일반화 성능 및 한계

#### 4.1 일반화 성능 향상 (3단계 평가)

**평가 설정:** ToolLLaMA의 일반화 능력을 3개 수준에서 검증합니다:

1. **Inst. (명령어 일반화)**: 훈련된 도구로 새로운 명령어
2. **Tool (도구 일반화)**: 같은 카테고리의 새로운 도구
3. **Cat. (카테고리 일반화)**: 완전히 새로운 카테고리의 도구

**성능 분석:**[1]
- **I1 (단일 도구)**: 모든 일반화 수준에서 62% 이상의 통과율 달성
- **I2 (다중 도구)**: 카테고리 일반화에서도 77%의 높은 성능 유지
- **I3 (복합 다중)**: 복잡성 증가에도 66% 통과율 유지

이는 ToolLLaMA가 API 문서만으로도 새로운 API에 적응할 수 있음을 시사합니다.

#### 4.2 분포 외(Out-of-Distribution) 일반화: APIBench

**실험 설정:**
- 학습 중 보지 못한 완전히 다른 도메인에서 테스트
- 비교 대상: Gorilla (APIBench 특화 모델)

**결과:**[1]

| 방법 | HuggingFace | TorchHub | TensorHub |
|------|-----------|---------|----------|
| | AST Acc. | AST Acc. | AST Acc. |
| Gorilla-ZS + Oracle | 44.36% | 59.14% | 83.21% |
| Gorilla-RS + Oracle | 89.27% | 93.01% | 94.16% |
| **ToolLLaMA + Our Retriever** | **16.77%** | **51.16%** | **40.59%** |
| **ToolLLaMA + Oracle** | **88.80%** | **85.88%** | **88.62%** |

**핵심 통찰:**
- Oracle(정답 API)을 사용할 경우, ToolLLaMA는 Gorilla와 동등한 성능 달성
- 이는 새로운 도메인에 대한 강력한 영지식(zero-shot) 일반화 능력을 입증합니다.

#### 4.3 주요 한계

**1) API 검색 정확도**
- 자동 검색기 사용 시 성능 저하 현상
- OOD 데이터셋(APIBench)에서 HuggingFace: 16.77%, TensorHub: 40.59% (낮은 성능)
- 원인: 새로운 도메인의 API 설명과 쿼리 간의 의미적 거리

**2) 복잡한 다중 도구 시나리오**
- I3 (컬렉션 내 다중 도구)에서 I1 대비 상대적으로 낮은 성능
- 특히 카테고리 간 다중 도구 시나리오에서 도구 조합의 복잡성 증가

**3) 인간 평가와의 차이**
- Win rate 평가에서 인간 평가와의 일치도가 Pass rate(87.1%)보다 낮음(80.3%)
- 원인: 도구 활용의 "올바른" 방식이 다양할 수 있음

**4) 제한된 맥락 길이**
- 8192 토큰으로 확장되었지만, 매우 긴 API 응답이나 복잡한 체인의 경우 여전히 제약
- API 응답 압축이 필요하며, 이는 중요 정보 손실 가능성

**5) 도구 진화 문제**
- RapidAPI의 API가 시간에 따라 변경되면 모델의 성능 저하
- 정적 문서 기반 학습의 근본적 한계

***

### V. 앞으로의 연구 영향과 고려사항

#### 5.1 ToolLLM의 학술적 및 실무적 영향

**학술적 기여:**

1. **도구 학습의 새로운 기준 제시**[2][3]
   - 16,000+ API를 포함한 실제 규모의 벤치마크 제시
   - 다중 도구 시나리오를 포함한 평가 프레임워크 확립

2. **DFSDT의 일반성**[2]
   - Tree-of-Thought와 유사한 개념이지만, 실제 도구 활용 시나리오에 최적화
   - 일반적 의사결정 문제에 적용 가능한 알고리즘

3. **자동 평가 시스템의 신뢰성**[2]
   - ChatGPT 기반 ToolEval이 인간 평가와 높은 일치도(80-87%)를 보임
   - 대규모 도구 활용 평가의 확장성 제시

**실무적 영향:**

1. **오픈소스 LLM의 민주화**
   - 7B 모델이 ChatGPT 수준의 도구 활용 능력 달성
   - 엔터프라이즈 환경에서 비용 효율적 배포 가능

2. **플러그인 생태계 지원**
   - API 문서만으로 새로운 도구에 적응 가능
   - 동적 도구 추가 및 제거 용이

#### 5.2 최신 후속 연구의 발전 방향

**2024년 이후의 주요 연구 동향:**

**1) 도구 문서의 품질 개선**[4]
- **EASYTOOL (2025)**: 장황한 도구 문서를 간결한 지시사항으로 변환
  - 토큰 비용 50% 이상 감소
  - 모델의 도구 활용 정확도 개선
  - 결론: 문서의 **구조와 간결성**이 성능에 영향

**2) 다중 도구 추론의 심화**[5][6]
- **ToolHop (2025, ACL)**: 다중 단계(Multi-hop) 도구 활용 벤치마크
  - 995개 쿼리, 3,912개 도구로 구성
  - GPT-4o도 49.04% 정확도에 불과
  - **결론**: 현재 모든 LLM이 복잡한 도구 연쇄에서 성능 부족

**수식 표현:**

$$\text{Multi-hop Accuracy} = \frac{\text{Successfully completed multi-step tool chains}}{\text{Total queries}}$$

GPT-4o: 49.04%이므로, 대략 절반의 복잡한 작업만 완료

**3) 동적 도구 학습**[3][7]
- **Learning Evolving Tools (2025)**: API가 시간에 따라 변경되는 상황에 대응
  - 새로운 API 버전 자동 감지
  - 문서 업데이트 시 적응적 재학습
  - **ToolLLM의 정적 성격을 보완**

**4) 대규모 도구 검색 개선**[2]
- **DEER (2024)**: 의사결정 인식 및 도구 샘플링 전략
  - 새로운 도구에 대한 일반화 성능 강화
  - 검색 정확도 개선

**5) 도구 활용 평가의 정교화**[8]
- **T-Eval (2024)**: 도구 활용을 7개 하위 프로세스로 분해
  - 명령어 이해 → 계획 → 추론 → 검색 → 이해 → 검토
  - 각 단계의 정밀한 진단 가능
  - **결론**: 전체 성공률만으로는 불충분하며, 세부 능력 분석 필요

***

### VI. 현재 연구 단계의 미해결 과제 및 고려사항

#### 6.1 기술적 미해결 과제

**1) API 검색 정확도의 문제**

$$\text{Retrieval Precision} = \frac{\text{Retrieved APIs in ground truth}}{\text{Total retrieved APIs}}$$

- ToolBench 내: 78-85% (I1-I3)
- APIBench (새 도메인): 16.77-51.16%
- **해결 방안 모색**: 도메인 적응 검색기, 하이브리드 검색(어휘+의미)

**2) 복잡성 확장성**

- 현재: 최대 3451개 도구 범위
- 문제: 실제 API 마켓플레이스는 수십만 개 API 보유
- **필요 연구**: 계층적 도구 분류, 요약 기반 검색

**3) 도구 진화(Tool Evolution) 대응**

$$\text{Adaptation Challenge: } \text{API}_t \neq \text{API}_{t+\Delta t}$$

- API 파라미터 변경
- 응답 형식 변경
- 새로운 API 버전 출시
- **해결책**: 지속적 미세조정, 온라인 학습, 버전 관리 시스템

#### 6.2 평가 방법론의 한계

**1) Win rate 평가의 주관성**
- 인간 평가자 간 일치도가 낮은 경우 (초기 연구에서 보고)
- 원인: "최적의" 도구 사용 경로가 여러 개 존재 가능
- **개선 방안**: 다중 기준 평가, 명확한 평가 체크리스트

**2) 실제 API 신뢰성 문제**
- 일부 API의 예측 불가능한 동작
- 타임아웃, 서버 에러 등 처리 미흡
- **필요 사항**: 강건성(Robustness) 평가 포함

#### 6.3 향후 연구 시 권고사항

**1) 도구 문서 최적화의 중요성**
- EASYTOOL의 성과에 따라 **문서의 품질과 구조**가 핵심 성능 요인
- 문제: 기존 API 문서의 불일치, 과도한 정보
- **권고**: API 문서 표준화, 자동 요약 기법 개발

**2) 동적 환경 대응**
- 실제 API는 지속적으로 변경 (Breaking changes, deprecation 등)
- **연구 방향**: 
  - 온라인 적응 학습 (Online adaptation)
  - 변경 감지 시스템
  - 버전 관리 메커니즘

**3) 사실성(Factuality) 강화**
- 도구 호출 시 파라미터 할루시네이션 문제
- **해결 방안**: 제약 기반 생성 (Constrained decoding), 파라미터 검증

**4) 에러 복구 능력**
- DFSDT의 백트래킹이 항상 효과적이지는 않음
- **개선**: 실패 원인 분석, 적응적 전략 변경

**5) 다국어 및 도메인 특화**
- 현재: 영어 API와 일반 도메인
- **확대 필요**: 다국어 API, 도메인 특화 (의료, 금융 등) 도구 활용

***

### VII. 결론

**ToolLLM**은 다음 세 가지 차원에서 혁신을 제시했습니다:

1. **데이터셋 규모의 비약**: 16,464개 실제 API를 포함한 ToolBench 구축으로 도구 활용 연구의 스케일을 획기적으로 확대

2. **추론 알고리즘의 개선**: DFSDT가 기존 ReACT 대비 20-30% 성능 향상을 달성하며, 복잡한 다중 도구 시나리오에서의 효과성 입증

3. **평가 프레임워크의 자동화**: ToolEval이 인간 평가와 80% 이상 일치하여, 대규모 도구 활용 평가의 확장성 확보

**일반화 성능:**
- **명령어 일반화**: 62-77% (훈련된 도구, 새 명령어)
- **도구 일반화**: 60-71% (같은 카테고리, 새 도구)
- **카테고리 일반화**: 60-62% (새 카테고리)
- **분포 외 일반화**: Oracle 사용 시 88.8% (새 도메인)

**현 단계의 한계:**
1. API 검색 정확도가 새 도메인에서 급격히 하락
2. 복잡한 다중 도구 추론에서 여전히 50% 미만의 성능
3. 동적 API 변화 대응 능력 부족

**향후 연구 방향:**
- 도구 문서 품질 최적화 (EASYTOOL 방향)
- 다중 단계 추론 능력 강화 (ToolHop의 도전)
- 동적 도구 진화 대응 (Learning Evolving Tools)
- 평가 방법론의 정교화 (T-Eval의 다층 평가)

ToolLLM은 오픈소스 LLM을 실제 API 생태계와 연결하는 교두보 역할을 했으며, 이를 바탕으로 한 후속 연구들이 개별 과제를 심화 연구하고 있습니다. 특히 **문서 최적화**와 **다중 단계 추론**이 2024-2025년의 핵심 연구 초점입니다.[7][6][3][4][2][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/569ca2b9-bc81-4db7-815e-f812d07345e8/2307.16789v2.pdf)
[2](https://arxiv.org/pdf/2402.16696.pdf)
[3](http://arxiv.org/pdf/2410.06617.pdf)
[4](https://aclanthology.org/2025.naacl-long.44.pdf)
[5](https://aclanthology.org/2025.acl-long.150.pdf)
[6](https://www.emergentmind.com/papers/2501.02506)
[7](http://arxiv.org/pdf/2410.08197.pdf)
[8](https://arxiv.org/abs/2312.14033)
[9](https://arxiv.org/pdf/2307.16789.pdf)
[10](https://arxiv.org/pdf/2304.08244.pdf)
[11](https://arxiv.org/html/2309.17428v2)
[12](https://aclanthology.org/2023.emnlp-main.187.pdf)
[13](http://arxiv.org/pdf/2402.15491.pdf)
[14](https://arxiv.org/abs/2307.16789)
[15](https://proceedings.neurips.cc/paper_files/paper/2024/file/db93ccb6cf392f352570dd5af0a223d3-Paper-Conference.pdf)
[16](https://proceedings.iclr.cc/paper_files/paper/2024/file/28e50ee5b72e90b50e7196fde8ea260e-Paper-Conference.pdf)
[17](https://blog.quotientai.co/evaluating-tool-calling-capabilities-in-large-language-models-a-literature-review/)
[18](https://openreview.net/forum?id=8m7p4k6Zeb)
[19](https://aclanthology.org/2025.realm-1.14.pdf)
[20](https://arxiv.org/html/2506.21071v1)
[21](https://www.semanticscholar.org/paper/0bfc804e31eecfd77f45e4ee7f4d629fffdcd628)
[22](http://arxiv.org/pdf/2310.03710.pdf)
[23](https://arxiv.org/pdf/2407.10718.pdf)
[24](https://arxiv.org/pdf/2303.09014.pdf)
[25](http://arxiv.org/pdf/2502.04689.pdf)
[26](https://aclanthology.org/2023.findings-acl.216.pdf)
[27](https://arxiv.org/pdf/2310.08740.pdf)
[28](https://arxiv.org/pdf/2410.13080.pdf)
[29](https://aclanthology.org/2023.findings-emnlp.753.pdf)
[30](https://openreview.net/forum?id=zIJFG7wW2d)
[31](https://www.backend.ai/blog/2024-01-28-tool-llms-part-1)
[32](https://datasciencedojo.com/blog/llms-zero-shot-reasoners/)
[33](https://arxiv.org/abs/2501.02506)
[34](https://malmm1.github.io/assets/IROS_2025_malmm_v8.pdf)
[35](https://arxiv.org/abs/2404.03114)
[36](https://arxiv.org/html/2510.02668v1)
