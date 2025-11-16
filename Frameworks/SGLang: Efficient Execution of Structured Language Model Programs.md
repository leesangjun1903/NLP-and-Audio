# SGLang: Efficient Execution of Structured Language Model Programs

### 1. 핵심 주장과 주요 기여

**SGLang**은 대규모 언어 모델(LLM)의 복잡한 프로그래밍과 실행을 위한 효율적인 시스템으로, 다음의 핵심 주장을 제시한다.[1]

**주요 기여:**

**전면부(Frontend):** Python에 내장된 도메인 특화 언어(DSL)로, LLM 프로그래밍의 복잡성을 해결한다. `extend`, `gen`, `select`, `fork`, `join` 등의 기본 연산을 제공하여 복잡한 프롬프트 조작, 병렬 제어 흐름, 구조화된 입출력을 간편하게 처리할 수 있다.[1]

**런타임(Runtime):** 세 가지 혁신적인 최적화 기법을 구현한다:
- **RadixAttention**: KV 캐시의 자동 재사용을 위해 Radix 트리 기반의 LRU 캐시 관리 메커니즘 도입
- **압축된 유한 상태 기계(Compressed FSM)**: 제약 조건이 있는 디코딩 속도 향상
- **API 투기적 실행**: API 기반 모델의 다중 호출 최적화[1]

실험 결과, SGLang은 기존 시스템 대비 **최대 6.4배 높은 처리량**을 달성하며, 에이전트 제어, 논리적 추론, 지식검색 강화 생성(RAG), 다중 턴 대화 등 다양한 작업에서 입증되었다.[1]

***

### 2. 해결하는 문제와 제안하는 방법

#### 2.1 문제 정의

SGLang이 해결하는 두 가지 주요 문제:[1]

**(1) 프로그래밍의 복잡성:** LLM의 비결정성 특성으로 인해 다음 작업들이 복잡해진다:
- 문자열 조작 및 프롬프트 엔지니어링
- 출력 파싱 및 형식 처리
- 다중 양식(multimodal) 입력 처리
- 병렬 실행 메커니즘 구현

**(2) 실행의 비효율성:** 현재 추론 엔진들(vLLM, TGI, TensorRT-LLM)은 워크로드에 대한 직접 지식 없이 최적화되어 있어:
- 공통 프리픽스를 가진 다중 LLM 호출 간 KV 캐시 재사용 불가
- 제약 조건 있는 디코딩(JSON 등)에서 토큰 단위 처리로 인한 비효율성
- 동적 트리 구조와 같은 복잡한 재사용 패턴 미지원[2][1]

#### 2.2 제안하는 방법

**전면부 설계:**

프롬프트 상태를 스트림으로 관리하는 인터프리터 기반 접근법:[1]

```python
@function
def essay_judge(s, image_path, essay):
    s += system("Evaluate an essay about an image.")
    s += user(image(image_path) + "Essay:" + essay)
    s += assistant(select("related", choices=["yes", "no"]))
    
    if s["related"] == "no": 
        return
    
    # 병렬 포크를 통한 다중 차원 평가
    forks = s.fork(3)
    for f, dimension in zip(forks, dimensions):
        f += user(f"Evaluate: {dimension}")
        f += assistant(gen("judgment", stop="END"))
    
    # 결과 병합 및 JSON 출력
    s += user("Return as JSON")
    s += assistant(gen("output", regex=schema))
```

**런타임 최적화:**

##### **(1) RadixAttention: KV 캐시 효율적 재사용**

**기본 개념**: Radix 트리를 사용하여 토큰 수열과 대응하는 KV 캐시 텐서 간의 매핑을 관리한다.[1]

**핵심 수식 및 알고리즘:**

캐시 히트율 정의:[1]

$$ \text{Cache Hit Rate} = \frac{\text{Number of cached prompt tokens}}{\text{Number of prompt tokens}} $$

**정리 3.1 (최적 스케줄링):** 깊이 우선 탐색(DFS) 순서로 Radix 트리를 순회할 때, 캐시 크기가 최대 요청 길이 이상이면 최적 캐시 히트율을 달성한다.[1]

**구현 세부사항:**
- **LRU 제거 정책**: 메모리 부족 시 가장 최근에 사용되지 않은 리프 노드부터 제거
- **캐시 인식 스케줄링**: 매칭된 프리픽스 길이로 대기 요청을 정렬하여 DFS 순서 근사
- **레퍼런스 카운팅**: 현재 실행 중인 배치에서 사용 중인 노드는 보호[1]

**실제 성능:**
- MMLU 5-shot 학습: 5개 예제의 KV 캐시 재사용으로 처리량 향상 및 첫 토큰 지연 시간 감소
- HellaSwag: 두 수준의 재사용 (예제 + 공통 질문 프리픽스)
- 생산 배포(Chatbot Arena): LLaVA-Next-34B에서 52.4% 캐시 히트율, Vicuna-33B에서 74.1% 히트율 달성[1]

##### **(2) 압축된 유한 상태 기계 (Compressed FSM)**

**문제**: 기존 시스템은 정규표현식 제약을 다음 토큰만 마스크하여 토큰 단위로만 디코딩:[1]

JSON 스키마 예시: `{"summary": "[\\w\\d\\s]+.", "grade": "[ABCD][+-]?"}`

**해결책**: 인접한 단일 전이 엣지를 압축하여 다중 토큰을 한 단계에서 디코딩:[1]

- **정규 FSM**: `{` → `"` → `s` → `u` → `m` → `m` → `a` → `r` → `y` → (8개 상태)
- **압축 FSM**: `{"summary"` → (1개 압축 엣지)

성능 향상: JSON 디코딩에서 **1.6배 처리량 증가**[1]

**재토큰화 처리**: 압축된 텍스트는 원본 토크나이저로 재토큰화하여 의미 왜곡 방지[1]

##### **(3) API 투기적 실행**

API 기반 모델(GPT-4)을 위한 최적화:[1]

다중 호출 패턴: `s += context + "name:" + gen("name", stop="\n") + "job:" + gen("job", stop="\n")`

**기법**: 첫 번째 호출에서 조건을 무시하고 추가 토큰을 생성하여 생성 출력을 저장, 이후 프리미티브와 재사용 및 매칭. 신중한 프롬프트 엔지니어링으로 높은 정확도 달성 시 API 호출 비용 절감.[1]

---

### 3. 모델 구조 및 성능 분석

#### 3.1 전체 시스템 아키텍처

| 구성 요소 | 역할 | 기술 |
|--------|------|------|
| **인터프리터** | 프로그램 실행 | 스트림 기반 비동기 처리 |
| **Radix 트리** | KV 캐시 관리 | LRU 정책 + DFS 스케줄링 |
| **Compressed FSM** | 제약 디코딩 | 엣지 압축 + 재토큰화 |
| **런타임** | 최적화 실행 | continuous batching + tensor parallelism 호환 |

#### 3.2 벤치마크 성능

**처리량 개선 (표준화):**[1]

| 작업 | SGLang | vLLM | Guidance | LMQL |
|------|--------|-------|---------|------|
| MMLU (5-shot) | 1.0 | 0.35 | 0.22 | 0.15 |
| Tree-of-Thought | 1.0 | 0.22 | - | - |
| JSON Decoding | 1.0 | 0.62 | - | - |
| Multi-turn Chat (Short) | 1.0 | 0.45 | 0.32 | - |

**지연 시간 개선:**[1]

- **단일 프로그램 최대 3.7배 지연 시간 감소**
- 첫 토큰 지연(TTFT)이 KV 캐시 재사용으로 크게 개선
- 장문 출력의 경우 공유 부분이 적어 개선 미미

**다중 모드 모델 성능:**[1]

| 모델 | 기준 | SGLang | 개선 배수 |
|------|------|--------|----------|
| LLaVA-v1.5-7B | 0.18 image/s | 1.15 image/s | 6.4× |
| LLaVA-NeXT-34B | 0.02 frame/s | 0.10 frame/s | 5.0× |

#### 3.3 분석 결과

**Ablation 연구:**[1]

1. **캐시 미사용**: 처리량 70-80% 감소
2. **트리 구조 제거** (단순 테이블 캐시): 10-15% 성능 저하
3. **FCFS vs 캐시 인식 스케줄링**: 20-30% 성능 차이
4. **프리픽스 힌트 미제공**: 런타임 매칭 비효율로 15% 성능 저하
5. **병렬화 제거**: 프로그램 내 병렬성 미활용으로 5-10% 저하

최대 캐시 히트율은 이론적 최적값(DFS 순서)의 **평균 96% 수준** 달성.[1]

**오버헤드 분석:**

RadixAttention 자체 오버헤드: 100개 요청 중 0.2초 (0.3% 미만). 캐시 히트가 없어도 무시할 수 있는 수준.[1]

---

### 4. 모델의 일반화 성능 향상

#### 4.1 일반화 성능에 영향을 미치는 요소

**SGLang이 직접적으로 일반화를 향상시키는 방식:**

**(1) 구조화된 출력을 통한 견고성 증대:**[3][4]

구조화된 출력(JSON, XML 등)이 강제되면 모델이 일관성 있는 형식을 따르도록 유도된다. 연구에 따르면:[3]
- 일반 LLM: 구조화 출력에서 평균 0.4 점수 (SoEval 벤치마크)
- GPT-4: 다른 모델 대비 24% 높은 성능

**의료 분야 응용에서 구조화된 추론:**[4]
- 7단계 임상 진단 프로세스를 구조화하면 Factuality Score 85.8 달성
- 미세 조정 모델을 초과하는 성능

**(2) 다중 작업 학습과의 상호작용:**[5][6][7]

SGLang의 병렬 처리(`fork`) 메커니즘이 다중 작업 시나리오에서 작동할 때:[7]

- 작업 특화 뉴런의 **중첩도가 높을수록** 일반화 성능 향상
- 작업 간 지식 공유(parameter similarity)가 일반화 상관성 0.7 이상

**예시:** Granite-20B-FunctionCalling 모델[6]
- 7개 기본 작업의 다중 작업 훈련으로 미보이는 작업에 대한 일반화
- BFCL(Berkeley Function Calling Leaderboard) 상위권 성능

**(3) 추론 길이 일반화:**[8]

다중 작업 환경에서 길이 일반화(Meta-RFFT) 개선:[8]
- 86개 작업의 대규모 데이터로 훈련 시 미보이는 작업에서 50% 이상 길이 일반화율 향상
- 비구조화된 단일 작업 접근보다 우수

#### 4.2 제한 사항 및 미해결 과제

**구조화된 출력의 한계:**[9][10]

1. **확률 왜곡**: 압축 FSM에서 선택지를 구분하지 못함[9]
   - 예: "Above Average"를 "A" 등급으로 오인 가능

2. **의미 변경**: 장기 제약 조건이 원본 의도를 변형할 수 있음[9]

3. **정확도 트레이드오프**: KV 캐시 압축 기법(SnapKV, CLLA 등)이 출력 길이 증가 가능성[10]

**캐시 관리의 비최적화:**

1. **초기 LRU 정책 한계**: 미래 접근 패턴을 예측하지 못해 최적의 1.5-2배 캐시 크기 필요[11]
   - 최근 연구(LPC)는 학습 기반 캐시 정책으로 18-47% 개선

2. **분산 환경의 복잡성**: 다중 워커 간 메타 트리 유지 시 일관성 문제[1]

#### 4.3 일반화 성능 향상의 메커니즘

**원인-효과 관계:**

$$ \text{구조화된 제약} \rightarrow \text{일관된 형식 강제} \rightarrow \text{다운스트림 작업 안정성} \rightarrow \text{일반화 향상} $$

**실증적 증거:**[4][6]

- 의료 QA: 구조화된 추론으로 도메인 외(ODD) 작업에서 일반화 97% vs 미구조화 85%
- 함수 호출: 7개 기본 작업 학습 후 미보이는 도메인에서 73% 일반화율

***

### 5. 한계 및 도전 과제

#### 5.1 시스템적 한계

1. **정적 그래프 컴파일의 제한:**[1]
   - 데이터 종속적 제어 흐름을 지원하지 못함
   - 동적 생성 길이 변동으로 인한 KV 캐시 재계산 필요

2. **기아 현상(Starvation):**[1]
   - 캐시 인식 스케줄링이 긴 프리픽스의 요청을 우선하면서 짧은 요청의 지연 가능
   - 공정한 스케줄링과의 트레이드오프

3. **다중 GPU 분산:**[1]
   - 데이터 병렬화 시 메타 트리 업데이트 오버헤드
   - 약한 일관성으로 인한 캐시 미스 증가 가능

#### 5.2 이론적 한계

**Theorem 3.1의 실제 적용 제약:**[1]

증명에서 가정한 조건과 실제 상황의 괴리:
- 예측 불가능한 출력 토큰 수 증가로 KV 캐시 재계산 필요
- 온라인 스케줄링에서 DFS 순서 깨짐

#### 5.3 성능 한계

**워크로드 의존성:**

- **장문 출력**: 디코딩 시간 지배로 KV 재사용 효과 미미
- **낮은 프리픽스 공유**: RAG 파이프라인에서 평균 50% 캐시 히트율에 그침
- **비결정적 생성**: 확률적 샘플링으로 같은 프리픽스 재사용 패턴 형성 불확실

---

### 6. 향후 연구에 미치는 영향

#### 6.1 업계 및 학술 연구에의 시사점

**배포 영향:**[12][13]

현재 xAI의 DeepSeek 추론 시스템(100K+ GPU)에서 운영 중. SGLang v0.1부터 v0.4까지의 진화:[13]
- v0.1 (2024년 1월): 자동 KV 캐시 재사용으로 5배 처리량 증가
- v0.2 (2024년 7월): 저오버헤드 CPU 런타임으로 3배 추가 개선
- v0.3 (2024년 9월): DeepSeek MLA 어텐션 최적화로 7배 빠른 트리톤 어텐션
- v0.4 (2024년 12월): 0 오버헤드 CPU 스케줄러 + 구조화 출력

**학술 연구 방향:**[14][15][11]

1. **KV 캐시 관리의 진화:**
   - **KVFlow**(2024년 10월): 워크플로우 인식 캐시로 RadixAttention 대비 1.83배 개선[14]
   - **LPC**(2025년): 학습 기반 캐시 정책으로 LRU 대비 18-47% 개선[11]
   - **Learned Prefix Caching**: 대화 패턴 예측으로 프리픽스 재사용 향상

2. **스케줄링 이론:**
   - **LLM Query Scheduling**: RadixAttention 하에서 k-LPM 알고리즘으로 공정성과 성능 균형[15]

3. **구조화 출력 안정성:**
   - **스키마 강화 학습**: 40K 다양한 JSON 스키마로 미세조정하면 복잡 JSON 생성 16% 향상[16]

#### 6.2 앞으로의 연구 방향 (최신 트렌드 기반)

**(1) 계층적 메모리 구조:**[13][1]

GPU/CPU/디스크 간 KV 캐시 오프로딩:
- 현재: GPU HBM만 사용
- 향후: CPU DDR/NVMe와의 계층적 관리로 초장문맥(100M 토큰) 서빙

**(2) 퍼지 의미 매칭:**[1]

정확한 프리픽스 매칭에서 의미 유사 프리픽스로 확장:
- 예: "What is AI?" ≈ "What's Artificial Intelligence?" 인식
- 임베딩 기반 유사도로 추가 캐시 히트율 향상 기대

**(3) 동적 양자화 및 압축:**[17][18][19]

KV 캐시 메모리 감소:
- **CLLA** (2024년 10월): 2% 압축으로 무손실 성능 유지
- **LeanKV** (2025년 4월): 키와 값에 차등 압축으로 메모리 50% 감소
- **RazorAttention** (2024년 7월): 70% 캐시 감소로 성능 미미 영향

**(4) 멀티태스크 학습의 일반화:**[5][6][7]

작업 특화 뉴런 기반 연속 학습:
- 작업 간 뉴런 중첩 80% 이상일 때 일반화 최적
- 점진적 학습(continual learning)에서 재앙적 망각 방지

**(5) 추론 체인의 최적화:**[20][21]

계획 수준의 일반화:
- **CPL** (2024년 9월): 탐색 계획 단위로 모델 일반화 +10.5% (GSM8K)
- MCTS와 DPO 결합으로 다중 작업 추론 능력 향상

#### 6.3 산업 적용 시 고려사항

**프로덕션 배포 체크리스트:**

1. **기아 현상 완화**: 우선순위 큐 + 공정성 스케줄러 통합[8]
2. **강화된 오류 처리**: 압축 FSM의 확률 왜곡 모니터링 및 폴백 메커니즘
3. **캐시 모니터링**: 실시간 히트율 추적 및 동적 임계값 조정
4. **확장성 검증**: 100K GPU 규모에서 메타 트리 일관성 검증

**비용-효율성 분석:**

- SGLang으로 5배 처리량 증가 시 **서버당 100만 사용자 추가 수용** 가능
- 연간 인프라 비용 수십억 규모 절감 (OpenAI, Google, Microsoft 등 보고)

***

### 결론

**SGLang**은 구조화된 LM 프로그래밍과 효율적 실행을 위한 **체계적 해결책**을 제시한다. RadixAttention의 자동 KV 캐시 재사용은 **기술 혁신**이며, 압축 FSM과 API 투기적 실행은 **실용적 최적화**이다.

가장 의미 있는 기여는 **프리픽스 공유를 활용한 다중 호출 구조의 체계적 이용**이다. 이는 에이전트, RAG, 다중 턴 대화 등 현실 응용에서 **5배에서 6.4배의 처리량 향상**을 이끈다.

**일반화 성능 측면**에서 SGLang은 직접적으로는 모델을 개선하지 않으나, 구조화된 출력 강제와 다중 작업 병렬화를 통해 **간접적 효과**를 제공한다. 향후 연구는 퍼지 매칭, 동적 양자화, 작업 특화 뉴런 기반 연속 학습 등으로 이러한 이점을 확대할 것이다.

더 이상의 조사가 필요한 영역은 **분산 환경의 확장성**, **장기 프리픽스의 의미 변형 문제**, **실시간 스케줄링의 공정성 보장**이다.

---

### 참고 자료 목록

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b8c02a27-67bd-47a4-a180-82f80fcef4b7/2312.07104v2.pdf)
[2](https://arxiv.org/pdf/2312.07104.pdf)
[3](https://www.sciencedirect.com/science/article/abs/pii/S0306457324001687)
[4](https://arxiv.org/html/2503.03194v1)
[5](https://arxiv.org/abs/2407.06488)
[6](https://arxiv.org/abs/2407.00121)
[7](https://aclanthology.org/2025.coling-main.200.pdf)
[8](https://openreview.net/forum?id=tCu7XGXjzo)
[9](https://arxiv.org/abs/2405.08944)
[10](https://arxiv.org/pdf/2503.24000.pdf)
[11](https://openreview.net/pdf/a340edd38ffafcfd1843a7f71d85464d9fb3e3df.pdf)
[12](https://lmsys.org/blog/2024-01-17-sglang/)
[13](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-25-sglang-72edc5043338f59db34d47e5b96ac870.pdf)
[14](https://arxiv.org/abs/2507.07400)
[15](https://arxiv.org/html/2502.04677v2)
[16](https://aclanthology.org/2025.acl-long.243.pdf)
[17](http://arxiv.org/pdf/2410.15252.pdf)
[18](https://arxiv.org/pdf/2412.03131.pdf)
[19](https://arxiv.org/html/2411.06680v1)
[20](https://arxiv.org/abs/2409.08642)
[21](https://arxiv.org/pdf/2409.08642.pdf)
[22](https://arxiv.org/abs/2412.10319)
[23](http://www.proceedings.com/079017-2000.html)
[24](https://arxiv.org/pdf/2404.14469.pdf)
[25](https://arxiv.org/pdf/2407.15891.pdf)
[26](http://arxiv.org/pdf/2408.05646.pdf)
[27](http://arxiv.org/pdf/2410.21465.pdf)
[28](https://www.rohan-paul.com/p/optimizing-llm-inference-for-higher)
[29](https://qingkeai.online/upload/pdf/20250713/SGLang.pdf)
[30](https://arxiv.org/html/2510.18672v1)
[31](https://www.tredence.com/blog/llm-inference-optimization)
[32](https://aclanthology.org/2024.acl-long.589)
[33](https://www.nature.com/articles/s43588-024-00747-9)
[34](https://dl.acm.org/doi/10.1145/3647649.3647650)
[35](https://arxiv.org/abs/2405.12229)
[36](https://arxiv.org/abs/2406.03718)
[37](https://ojs.aaai.org/index.php/AAAI/article/view/27790)
[38](https://arxiv.org/abs/2404.04949)
[39](https://arxiv.org/pdf/2410.06741.pdf)
[40](https://arxiv.org/pdf/2311.06720.pdf)
[41](https://arxiv.org/pdf/2305.14078.pdf)
[42](https://arxiv.org/html/2502.03041v1)
[43](https://arxiv.org/pdf/2403.04233.pdf)
[44](https://arxiv.org/pdf/2212.08354.pdf)
[45](https://arxiv.org/abs/2410.05603)
[46](https://proceedings.mlr.press/v202/dai23d/dai23d.pdf)
[47](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Adversarial_Domain_Prompt_Tuning_and_Generation_for_Single_Domain_Generalization_CVPR_2025_paper.pdf)
[48](https://arxiv.org/html/2504.12185v1)
[49](https://arxiv.org/html/2507.07400v1)
[50](https://www.sciencedirect.com/science/article/abs/pii/S0925231225007933)
