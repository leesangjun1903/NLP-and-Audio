# Language Models of Code are Few-Shot Commonsense Learners

## 1. 핵심 주장 및 주요 기여
이 논문은 **자연어가 아닌 코드**로 구조화된 커먼센스 그래프를 표현함으로써, 코드에 특화된 대형 언어 모델(Code-LLM)이 자연어 모델(NL-LLM)보다 적은 샷(few-shot)으로도 우수한 구조화된 추론 능력을 발휘할 수 있음을 보인다.[1]
주요 기여:
- 커먼센스 구조 생성 과제(SCRIPT, PROPARA, EXPLAGRAPHS)를 **Python 코드**로 직렬화하는 COCOGEN 방법론 제안  
- 코드-LLM(CODEX)을 few-shot 프롬프트로 활용하여 NL-LLM 대비 **구조적·의미적 성능** 대폭 향상  
- 동적 예제 선택(KST) 및 다양한 코드 포맷 분석을 통한 시스템 최적화  

## 2. 문제 정의·제안 기법·모델 구조·성능·한계
### 문제 정의
- **구조화된 커먼센스 추론**: 입력 자연어 T에 대해, 노드·엣지로 이루어진 그래프 G 생성  
- 기존: 그래프를 DOT, 리스트 등 “텍스트”로 직렬화 → LLM이 구조를 제대로 학습하지 못함  

### COCOGEN 기법
1. **Python 직렬화**  
   - (T, G) → Python 클래스/함수 코드 Gc  
   - 노드: `Node()` 인스턴스, 엣지: `children` 속성 또는 `add_edge` 호출로 표현  
2. **Few-Shot Prompting**  
   - k개의 (Gc, T) 예제를 코드 형태로 prompt에 삽입  
   - 테스트 입력 T → 부분 완성된 클래스 형태로 prompt 뒤에 추가 → CODEX가 코드 완성  
3. **후처리**  
   - 생성된 Python 코드를 역직렬화하여 원래 그래프 Ĝ 복원 및 평가  

### 모델 구조
- **기반 모델**: OpenAI CODEX (code-davinci-002)  
- **비교군**: GPT-3 CURIE/DAVINCI with 텍스트·코드 프롬프트, T5 계열 Fine-tuning  
- **동적 예제 선택**: SentenceTransformers 기반 KST(retriever)  

### 성능 향상
- **SCRIPT 생성**: 15-shot CODEX+코드 프롬프트가 T5 fine-tuned 전체(3.5K 샘플) 대비 BLEU↑+1.4, 구조적 METRIC(ISO/GED)에서도 우위.[1]
- **SCRIPT 엣지 예측**: 15-shot COCOGEN F1=56.5 vs. T5(100샘플) F1=51.9, CURIE/DAVINCI 대비 +8~15pt.[1]
- **PROPARA**: 3-shot COCOGEN F1=63.0, GPT-3 대비 +5.0, few-shot 기준 SOTA.[1]
- **EXPLAGRAPHS**: 30-shot COCOGEN이 T5(1.5K샘플) 대비 StCA↑7pt, SeCA↑2pt, G-BS↑5pt.[1]

### 한계
- **모델 의존성**: 비공개·상업적 CODEX 사용 → 향후 접근성 제한 우려  
- **영어 독점**: 영어 데이터에만 실험, 다국어·교차언어 일반화 검증 필요  
- **프롬프트 설계 민감도**: 모델 규모 작을수록 코드 포맷 설계에 민감  

## 3. 모델 일반화 성능 향상 가능성
- **코드 포맷 유사성**: 사전학습 데이터와 형식 일치 시 일반화 ↑ → 타 구조화 과제(e.g. 지식베이스 구축, 프로그래밍 과제)에 적용 가능  
- **동적 예제 선택**: KST를 통한 유사 입력 기반 프롬프트 선정으로 도메인 적응력 강화  
- **규모의 경제**: 대규모 Code-LLM일수록 프롬프트 형식 민감도 감소 → 향후 모델 크기 확장 시 프롬프트 설계 부담 완화  

## 4. 향후 연구 영향 및 고려사항
- **영역 확장**: 자연어 이해와 구조적 출력이 결합된 다양한 NLU/PLU 과제(토론 요약, 절차 설명, 논리 그래프)에 COCOGEN 적용  
- **개방형 모델 실험**: 오픈소스 Code-LLM(GitHub Copilot 대체 모델 등)으로 재현성·접근성 확보  
- **다국어·교차언어**: 영어 외 언어, 다중 언어 코드 표기로 일반화 성능 검증  
- **프롬프트 자동화**: 프롬프트 검색·수정 기법 연구로 few-shot 설정 간소화  

***

 Language Models of Code are Few-Shot Commonsense Learners. Singh et al., ArXiv:2210.07128v3.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7f6cc613-7cc1-4ad5-b35b-780bba4c766b/2210.07128v3.pdf)
