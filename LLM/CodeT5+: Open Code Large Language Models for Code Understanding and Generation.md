# CodeT5+: Open Code Large Language Models for Code Understanding and Generation

### 1. 핵심 주장 및 주요 기여도 요약

CodeT5+는 Salesforce AI Research에서 제안한 **유연한 인코더-디코더 기반 대규모 코드 언어 모델 제품군**입니다. 논문의 핵심 주장은 기존 코드 LLM들이 가진 두 가지 근본적인 한계를 해결하는 데 있습니다.[1]

**첫째, 아키텍처의 유연성 부족**: 기존 모델들은 인코더만 또는 디코더만 사용하도록 설계되어 특정 작업에만 최적화되어 있었습니다. CodeT5+는 동일한 인코더-디코더 구조에서 **인코더 전용, 디코더 전용, 인코더-디코더 모드를 유연하게 활성화**하여 다양한 하류 작업에 적응할 수 있습니다.[1]

**둘째, 제한적인 사전학습 목표**: 기존 모델들은 모든 작업에 적합하지 않은 제한된 사전학습 목표만 사용했습니다. CodeT5+는 **다양한 사전학습 목표의 혼합**(span denoising, contrastive learning, text-code matching, causal LM)을 통해 사전학습-미세조정 간의 불일치를 완화합니다.[1]

**주요 기여도:**
- 20개 이상의 코드 관련 벤치마크에서 최고 수준의 성능 달성[1]
- HumanEval에서 35.0% pass@1 달성으로 OpenAI의 code-cushman-001 모델 초월[1]
- 계산 효율적인 사전학습 전략으로 모델 규모 확장[1]
- 지시문 조정(instruction tuning)을 통한 자연언어 지시사항 정렬[1]

---

### 2. 해결 문제, 제안 방법, 모델 구조 및 성능

#### 해결하고자 하는 문제

기존 코드 LLM들의 주요 한계:[1]

1. **아키텍처 부적절성**: 인코더 전용(CodeBERT, UniXcoder) 또는 디코더 전용(GPT-2, CodeGPT) 모델은 특정 작업에만 우수하며, 통합 인코더-디코더 모델도 모든 작업에서 최적 성능을 보이지 못함[1]
2. **사전학습-미세조정 불일치**: T5 기반 모델의 span denoising 목표는 자동회귀 코드 생성에 적합하지 않고, 텍스트-코드 대비학습 부재로 검색 작업 성능 저하[1]
3. **교차 모달 정렬 부족**: 기존 대비학습 접근법은 세밀한 텍스트-코드 정렬을 무시함[1]

#### 제안하는 방법 및 수식

**2단계 사전학습 전략:**

**단계 1: 단일모달 코드 데이터 사전학습**

세 가지 목표를 동일한 가중치로 결합:[1]

- **Span Denoising**: T5 방식의 손상된 코드 복구
- **Seq2Seq Causal LM**: 

$$
\mathcal{L}_{\text{seq2seq}} = -\log P_\theta(y|x_{<p}) 
$$
  
  여기서 $$p$$는 균등분포로 $$[0.1, 0.9]$$ 범위에서 샘플링된 피벗 위치
  
- **디코더 전용 생성**: $$[CLM]$$ 토큰으로 전체 코드 생성 감독

**단계 2: 이중모달 텍스트-코드 데이터 사전학습**

네 가지 목표 동시 최적화:[1]

- **텍스트-코드 대비학습**:

$$
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(e_t, e_c)/\tau)}{\sum_{(t',c') \in \mathcal{B}} \exp(\text{sim}(e_{t'}, e_{c'})/\tau)}
$$
  
  $$\tau$$는 온도 파라미터, 모멘텀 인코더로 음성 샘플 강화[1]

- **텍스트-코드 매칭**:

$$
\mathcal{L}_{\text{matching}} = -[\mathbb{1}_{y=1}\log P_\theta(\text{match}) + (1-\mathbb{1}_{y=1})\log(1-P_\theta(\text{match}))]
$$
  
  어려운 음성 광산 전략으로 정보성 있는 음성 선택[1]

- **텍스트-코드 Causal LM**: 이중 다중모달 변환(text-to-code, code-to-text)[1]

#### 모델 구조

**기본 구조(CodeT5+ 220M, 770M):**
- T5 아키텍처 기반, 처음부터 학습
- 인코더-디코더 대칭 구조[1]

**확장 구조(CodeT5+ 2B, 6B, 16B):**
- **Shallow Encoder & Deep Decoder**: 디코더가 생성 작업의 복잡성 처리를 위해 더 큰 용량[1]
- 인코더: CodeGen-mono 350M으로 초기화
- 디코더: CodeGen-mono 2B/6B/16B로 초기화
- 교차주의 층: 디코더 상단 L개 층에 무작위 초기화로 추가[1]
- **계산 효율**: 냉동된 깊은 디코더로 인해 적은 훈련 가능 파라미터 유지[1]

#### 성능 향상 결과

| 작업 | 주요 성과 | 참고 |
|------|---------|------|
| **코드 생성** (HumanEval) | InstructCodeT5+ 16B: 35.0% pass@1, 54.5% pass@10 | code-cushman-001 모델 초월[1] |
| **수학 프로그래밍** (MathQA-Python) | CodeT5+ 770M: 87.4% pass@80 | 수십 배 큰 모델 능가[1] |
| **코드 요약** (CodeSearchNet) | 6개 언어 모두에서 기존 최고 모델 능가 | 평균 BLEU-4 20.15[1] |
| **코드 완성** (PY150) | CodeT5+ 770M: 44.86% Exact Match | CodeGen-multi 350M 초과[1] |
| **텍스트-코드 검색** (CodeSearchNet) | 77.4% 평균 MRR | UniXcoder 대비 +3.2 MRR[1] |

**복잡성에 따른 로버스트성:**
- CodeT5+는 매개변수 수가 적음에도 불구하고 복잡한 추론 작업(10+ 단계)에서 CodeT5보다 훨씬 우수한 성능 유지[1]

#### 한계

1. **폐쇄 소스 모델과의 격차**: GPT-4(67.0% pass@1) 및 code-davinci-002(47.0% pass@1)와 여전히 성능 격차 존재[1]
2. **성능 포화**: 코드 요약, 코드 복제 감지 등 일부 작업에서 모델 규모 증가에 따른 성능 개선이 한계에 도달[1]
3. **제한된 데이터 분석**: 코드 데이터에서 잠재적 중복이 완전히 제거되지 않았을 가능성[1]
4. **구성 능력 제약**: LLM이 기하급수적으로 증가하는 추론 단계의 수학 문제에 여전히 어려움[1]

***

### 3. 모델의 일반화 성능 향상 가능성

#### 현재 일반화 성능 분석

**다양한 평가 환경에서의 강건성:**[1]
- **영점 샷(Zero-shot)**: HumanEval에서 35.0% pass@1 달성 - 학습 없이 우수한 성능
- **미세조정(Finetuning)**: 다양한 벤치마크에서 최고 성능 달성
- **지시문 조정(Instruction-tuning)**: 자연언어 지시사항 이해 능력 향상[1]

**교차 언어 일반화:**[1]
- 9개 프로그래밍 언어(Python, Java, JavaScript, Go, Ruby, PHP, C, C++, C#)에서 평가
- CodeSearchNet의 6개 언어에서 모두 최고 수준 성능[1]

**매개변수 효율성의 일반화:**[1]
- CodeT5+ 770M이 Incoder 6B, PaLM 62B와 유사 성능 제공
- 이는 작은 모델의 우수한 일반화 능력을 시사[1]

#### 최신 연구 기반 일반화 개선 가능성

**1. 도메인 적응(Domain Adaptation) 연구 동향 (2024-2025)**[2][3]

최근 연구들은 코드 모델의 교차 도메인 일반화 개선을 중점 추진:[3][2]
- AdaCCD (2024)는 주석 없이 새로운 프로그래밍 언어에서 클론 탐지 수행[2]
- MIGRATE (2025)는 코드 스위칭 데이터와 임베딩 전이로 교차 언어 적응 달성[3]
- 이러한 접근법들은 CodeT5+의 구조에 통합하면 **다양한 새로운 도메인으로의 일반화 강화** 가능[2][3]

**2. 사전학습 데이터 구성의 영향 (2024-2025)**[4][5][6]

Jackson Petty et al. (2025)의 연구는 코드 데이터 비율이 일반화에 중요한 영향을 미침을 증명:[5][6][4]
- 코드 비율 증가 → 의미론적 구문 분석 같은 구조화된 출력 작업 성능 ↑ 12배[6]
- 그러나 구문/형태론 민감 작업에서 성능 ↓[5]
- **시사점**: CodeT5+의 혼합 사전학습 목표 접근법이 이 균형을 더욱 정교화하면 **특정 작업 영역에서의 일반화 강화** 가능[4]

**3. 교차 언어 버전 일반화 (2024)**[7][8]

On the Generalizability of Deep Learning-based Code Completion Across Programming Language Versions 연구:[8][7]
- CodeT5가 Java 8에서 학습되었지만 Java 2-17 전 버전에 일반화되는 정도를 평가
- **발견**: 언어 버전 진화에 따른 코드 모델의 일반화 격차 존재
- **개선 방안**: CodeT5+의 다양한 버전과 시간대의 코드로 훈련하면 **장기적 호환성 및 일반화 능력 향상** 가능[7][8]

**4. 약한-강한 일반화(Weak-to-Strong Generalization) (2025)**[9]

최신 ICLR 2025 논문 "A Transfer Learning Framework for Weak to Strong Generalization":[9]
- 약한 모델의 피드백으로 강한 모델을 정렬하는 방법 제시
- **코드 모델 적용**: CodeT5+ 같은 강한 모델을 약한 감독자(작은 모델 또는 인간)로 정렬하면 **과적합 방지 및 일반화 능력 개선** 가능[9]

**5. 지시문 조정의 역할 (2024-2025)**[10][11]

OpenCodeInstruct (2024)와 OctoPack (2023) 연구:[11][10]
- 500k 샘플로만 미세조정해도 기본 모델 초월[10]
- 자연언어→코드 형식이 코드→코드 형식보다 효과적[10]
- **CodeT5+에의 영향**: 더 큰 규모의 명령 데이터셋으로 지시문 조정하면 **미해 본 작업으로의 일반화 개선** 가능[10]

**6. 검색 강화 생성(Retrieval-Augmented Generation) (2023-2024)**[12]

코드 작업에서 검색 강화가 일반화를 개선하는 증거:[12]
- CodeT5+의 검색-증강 생성 능력(상위-k 검색으로 입력 보강)이 특히 유용[1]
- **일반화 개선**: Java 태스크에서 EM +11.66%, Python에서 +11.83% 향상[1]
- 이는 **메모리 강화 메커니즘**으로 분배 외 데이터에 대한 일반화 가능성 제시[12]

#### 일반화 강화를 위한 구체적 제안

1. **적응형 사전학습 목표 가중치**: 작업 특성에 따라 대비학습, 매칭, Causal LM의 가중치를 동적으로 조정
2. **도메인별 앙상블 미세조정**: 여러 도메인에서 미세조정한 모델을 조합하여 새로운 도메인 적응성 강화
3. **다양한 코드 스타일 데이터**: 코드 축약, 상세 스타일, 다양한 네이밍 규칙을 포함한 더 다양한 학습 데이터
4. **점진적 미세조정(Progressive Fine-tuning)**: 기본 작업에서 복잡 작업으로 순차적 학습으로 과적합 완화

***

### 4. 연구 영향 및 향후 고려사항

#### 논문이 미칠 영향

**학계 영향:**[13][14][1]
- 737회 인용(2023년 발행 기준)으로 코드 LLM 연구에 광범위한 영향[13]
- 모듈 유연성 아이디어는 후속 연구의 기초 제공[14]
- 혼합 사전학습 목표 개념이 다중모달 학습 분야 전반에 확산[14]

**산업 응용:**[15]
- Salesforce가 모든 CodeT5+ 모델을 오픈소스화하여 개발자 커뮤니티 지원[15]
- HumanEval에서 OpenAI 폐쇄 모델 능가로 **오픈소스 코드 생성의 실용성 증명**[15]
- 개발자 생산성 도구에 직접 통합 가능[15]

#### 앞으로의 연구 시 고려할 점

**1. 데이터 품질과 오염 문제 (2024)**[16]

LiveCodeBench 연구에서 지적:[16]
- HumanEval, MBPP 같은 기존 벤치마크가 학습 데이터와 오염됨
- **고려사항**: CodeT5+ 평가 시 정기적으로 업데이트되는 동적 벤치마크(LiveCodeBench) 사용 권장[16]
- 지속적인 모델 개선을 위해 새로운 문제 지속 수집 필요[16]

**2. 복잡한 추론 일반화 (2023-2024)**[17][18]

VulLLM 연구(2024)에서 취약점 탐지의 일반화 분석:[17]
- 모델이 **표면적 매핑을 학습**하기 쉬우며 근본 원인 이해 부족[17]
- **향후 방향**: 다중 작업 학습과 보조 작업으로 깊은 의미론적 이해 강화[17]
- CodeT5+에 취약점 해석 작업 추가 통합 고려[17]

**3. 프로그래밍 언어 버전 진화 대응 (2024)**[8][7]

코드 완성의 교차 버전 일반화 문제:[7][8]
- Java의 진화(Java 2 → Java 17)에서 모델 성능이 저하됨[7]
- **해결책**: 다양한 시간 기간의 코드로 점진적 학습, 또는 버전 독립적 특성 학습[8][7]

**4. 약한 감독자와의 정렬 (2025)**[9]

약한-강한 일반화 메커니즘을 코드에 적용:[9]
- CodeT5+ 같은 강한 모델이 약한 감독자(예: 작은 모델, 인간)로부터의 피드백으로 **과적합 방지**[9]
- 미래 연구에서 CodeT5+의 지시문 조정 프로세스에 이 개념 통합[9]

**5. 코드와 자연언어의 균형 (2025)**[19]

How Does Code Pretraining Affect Language Model Task Performance? 연구:[19]
- 코드 비율 증가가 구조화된 작업 성능은 증가하나, 언어 민감 작업 성능 저하[19]
- **권장사항**: 작업 유형에 따라 사전학습 데이터 혼합비 조정 필요[19]
- CodeT5+ 계열 모델들을 특정 도메인 최적화 버전으로 개발 검토[19]

**6. 검색 강화 생성 최적화 (2024)**[16]

검색-증강 코드 생성의 효율성 개선:[16]
- 상위-k 검색에서 적절한 k 값 자동 선택 메커니즘 개발[16]
- 검색 인덱스 크기와 대기 시간 트레이드오프 분석[16]

**7. 계산 효율성과 공정성 (2024-2025)**

제한된 컴퓨팅 자원 환경에서의 고려:[20]
- 매개변수 효율 미세조정(LoRA 등)의 한계 분석 필요
- 모델 증류 기법으로 작은 모델로의 지식 전이 최적화[20]

***

## 결론

CodeT5+는 **유연한 모듈 구조와 혼합 사전학습 목표**를 통해 코드 이해 및 생성 작업의 일반화 문제를 획기적으로 개선했습니다. 특히, 작은 모델 규모로도 훨씬 큰 모델과 경쟁하는 성능을 달성함으로써 **효율성과 성능의 균형을 제시**했습니다.

최신 연구들(2024-2025)은 CodeT5+를 기반으로 도메인 적응, 교차 언어 일반화, 약한-강한 정렬 메커니즘 등을 통해 **다양한 환경에서의 일반화 능력 강화**를 진행 중입니다. 앞으로의 연구에서는 데이터 품질 관리, 언어 버전 진화 대응, 약한 감독자 정렬, 그리고 작업 특성에 따른 사전학습 데이터 혼합비 최적화가 중요한 과제가 될이터 혼합비 최적화가 중요한 과제가 될 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/61e56143-574e-47e7-ba1e-c8ce89b379ee/2305.07922v2.pdf)
[2](http://arxiv.org/pdf/2311.07277.pdf)
[3](https://pure.korea.ac.kr/en/publications/migrate-cross-lingual-adaptation-of-domain-specific-llms-through-)
[4](https://arxiv.org/abs/2409.04556)
[5](https://www.themoonlight.io/en/review/how-does-code-pretraining-affect-language-model-task-performance)
[6](https://www.emergentmind.com/papers/2408.10914)
[7](https://dl.acm.org/doi/pdf/10.1145/3643916.3644411)
[8](http://arxiv.org/pdf/2403.15149.pdf)
[9](https://openreview.net/forum?id=PeLLMw3wLX)
[10](https://arxiv.org/html/2504.04030v1)
[11](https://proceedings.iclr.cc/paper_files/paper/2024/file/1ec299a5229034141e58aeded0d0b9de-Paper-Conference.pdf)
[12](https://aclanthology.org/2023.emnlp-main.1013.pdf)
[13](https://arxiv.org/abs/2305.07922)
[14](https://syncedreview.com/2023/05/17/salesforce-ais-codet5-open-code-llms-flexibly-adapt-to-diverse-downstream-code-understanding-and-generation-tasks/)
[15](https://www.salesforce.com/blog/codet5-open-code-large-language-models/)
[16](https://arxiv.org/abs/2403.07974)
[17](https://arxiv.org/abs/2406.03718)
[18](https://ieeexplore.ieee.org/document/10655086/)
[19](https://openreview.net/forum?id=pxxmUKKgel)
[20](https://link.springer.com/10.1007/s00521-024-09930-5)
[21](https://arxiv.org/abs/2402.03749)
[22](https://www.ijcai.org/proceedings/2024/654)
[23](https://aclanthology.org/2025.findings-emnlp.217)
[24](https://arxiv.org/abs/2402.00861)
[25](https://arxiv.org/abs/2402.14852)
[26](https://aclanthology.org/2023.emnlp-main.68.pdf)
[27](https://arxiv.org/pdf/2305.07922.pdf)
[28](http://arxiv.org/pdf/2207.10397v2.pdf)
[29](https://arxiv.org/abs/2109.00859)
[30](http://arxiv.org/pdf/2208.05446.pdf)
[31](https://arxiv.org/pdf/2502.16704.pdf)
[32](https://arxiv.org/abs/2510.11184)
[33](https://aclanthology.org/2025.findings-acl.61.pdf)
[34](https://aclanthology.org/2024.acl-srw.52.pdf)
[35](https://aclanthology.org/2024.emnlp-industry.3.pdf)
[36](https://ieeexplore.ieee.org/document/10646594/)
[37](https://www.ijcai.org/proceedings/2024/87)
[38](https://ieeexplore.ieee.org/document/10673131/)
[39](https://ieeexplore.ieee.org/document/10483006/)
[40](https://ieeexplore.ieee.org/document/10943650/)
[41](https://ieeexplore.ieee.org/document/10571357/)
[42](https://dl.acm.org/doi/10.1145/3711896.3737408)
[43](https://arxiv.org/abs/2408.06467)
[44](https://ieeexplore.ieee.org/document/10633585/)
[45](https://ieeexplore.ieee.org/document/10721444/)
[46](https://arxiv.org/pdf/2403.02714.pdf)
[47](https://www.aclweb.org/anthology/D19-1165.pdf)
[48](https://arxiv.org/pdf/2312.12492.pdf)
[49](https://arxiv.org/pdf/2303.15833.pdf)
[50](http://arxiv.org/pdf/2412.11185.pdf)
[51](http://arxiv.org/pdf/2310.16803.pdf)
[52](https://www.isca-archive.org/interspeech_2025/mote25b_interspeech.pdf)
[53](https://milvus.io/ai-quick-reference/how-does-overfitting-occur-in-deep-learning-models)
[54](https://aclanthology.org/2025.coling-main.617.pdf)
[55](https://towardsdatascience.com/overfitting-in-deep-learning-what-is-it-and-how-to-combat-it-9760d25ad05b/)
[56](https://wikidocs.net/178818)
[57](https://openreview.net/forum?id=XFCKEgGhEK)
