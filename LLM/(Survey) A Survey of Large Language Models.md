# Wayne Xin Zhao et al., “A Survey of Large Language Models”

## 1. 핵심 주장과 주요 기여  
이 논문은 대규모 언어 모델(LLM)의 배경·기술·응용·평가를 체계적으로 정리한 종합 서베이이다.  
- **핵심 주장**: 언어 모델의 발전은 통계적 LM→신경망 LM→사전학습 PLM→LLM으로 이어져 왔으며, 모델·데이터·연산 규모의 확장이 예측 가능한 성능 향상(스케일링 법칙)과 함께 ‘Emergent Ability’로 불리는 비약적 능력 획득을 가져온다.  
- **주요 기여**:  
  1) LLM 연구의 네 축(사전학습·어댑테이션·활용·평가)을 정리하고, 각 축의 핵심 기술·결과·오픈 이슈를 요약.  
  2) LLM·PLM·NLM·SLM의 발전사를 논리적 흐름으로 재구성.  
  3) 공개 자원(모델·데이터·라이브러리) 목록화 및 비교.  
  4) 향후 과제로 데이터 품질·정렬(alignment)·안전성·LLM 내부 메커니즘 연구를 제안.  

## 2. 문제 설정·방법·모델 구조·성능 향상·한계  
### 문제 설정  
- LLM은 매개변수 수 10B 이상, 대규모 코퍼스를 사전학습해 emergent ability를 획득한 PLM을 의미.  
- 스케일링 법칙(DeepMind Chinchilla·OpenAI Kaplan 등)으로 성능 향상이 예측 가능하나, 설명 불가한 능력의 비약(문맥 내 학습·추론·명령 준수)이 나타남.  
### 제안 방법  
- **사전학습**: Transformer 기반 LM 목표, 대규모 웹·책·코퍼스·코드 데이터.  
- **어댑테이션**:  
  - Instruction tuning(자연어 지시문+샘플로 미세조정)  
  - RLHF(인간 피드백 강화를 통한 정렬)  
  - LoRA·Adapter 등 파라미터 효율적 튜닝  
- **활용**: In-Context Learning, Chain-of-Thought, Planning, Tool Use  
- **평가**: MMLU·BIG-bench·HELM 등 벤치마크, 인간 평가, LLM 평가  
### 모델 구조  
- 주로 decoder-only(또는 prefix) Transformer 사용.  
- LayerNorm·RMSNorm·DeepNorm, RoPE·ALiBi positional encoding, SwiGLU 활성화 등 안정화·추론 효율성 기법 도입.  
### 성능 향상  
- 모델·데이터·연산 스케일 업으로 성능이 꾸준히 개선.  
- Instruction tuning과 RLHF로 제로/소수 샷 성능과 안전성(유해·편향 완화) 제고.  
- CiT·Trees-of-Thought 등 복잡 추론 기법으로 reasoning task 성능 비약.  
### 한계  
- 요약된 텍스트 이상 ‘환각(hallucination)’ 문제(사실 오류)  
- 최신 정보 반영 부족(지식 최신성)  
- 고비용·높은 인프라 요구(사전학습·RLHF)  
- 모델 내부 작동 원리 불투명 설명 부족  

## 3. 모델 일반화 성능 향상 가능성  
- **데이터 다양성·품질**: 정제된 크롤링·도메인 데이터 혼합으로 일반화 극대화 가능.  
- **Instruction Tuning**: 다중 태스크·다중 도메인 지시문 융합 시 미지 태스크 일반화가 극대화됨.  
- **파라미터 효율 튜닝**: LoRA·Adapter로 대규모 파라미터 고정해도 일반화 유지하며 특수 도메인 적용 가능.  
- **추론 방법**: CoT·Tree-of-Thought 등 구조화된 추론을 통해 out-of-distribution reasoning 강화.  
- **Alignment**: RLHF로 안전·정확성 높이면서도 모델 일반화 저해 최소화하는 파라미터·데이터 스케줄링 연구 필요.  

## 4. 향후 영향 및 고려사항  
- **AI 연구 패러다임 전환**: 훈련·추론·응용·평가 전 과정에서 ‘언어’가 통합 인터페이스, 소프트웨어·하드웨어·윤리 동시 고려 필수.  
- **투명성·검증성 확립**: 모델 내부 메커니즘·의사 결정을 설명·검증하는 기법이 필요.  
- **데이터 and Alignment**: 최신·정제 데이터 확보와 안전·윤리 정렬 기법 고도화가 필수.  
- **효율성**: 저비용·저연산·고효율 모델·튜닝 프레임워크 개발로 연구·산업 보급 확대  
- **생태계**: MLLM·KG 활용·멀티모달·에이전트 등 인접 분야와의 시너지 창출로 활용 범위 확장  

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f9db9a03-366c-4d91-aa33-8500e9ca5b19/2303.18223v15.pdf

# 3. LLMs 개발을 위한 자원 (Resources of LLMs)

LLMs(대형언어모델)를 연구·개발·운영하기 위해 필요한 주요 자원은 크게 네 가지로 나눌 수 있습니다.  
각 항목별로 **(1) 어떤 자원이 있는지**, **(2) 왜 중요한지**, **(3) 어떻게 활용하는지**를 순서대로 정리했습니다.

***

## 3.1 공개된 모델 체크포인트 및 API

### 1) 공개 모델 체크포인트
- **LLaMA 시리즈** (Meta AI)
  - 7B, 13B, 30B, 65B 파라미터 모델  
  - GitHub·HuggingFace 등을 통해 가중치·토크나이저 공개  
  - 연구·실험용으로 광범위하게 활용  
- **Mistral, Gemma, Qwen, Baichuan** 등
  - 각 연구기관·회사에서 공개한 7B~70B급 모델  
  - 유사하게 가중치·토크나이저를 공개하여 사용자 커스터마이징 가능  
- **LLaMA 파생 모델군**  
  - Alpaca, Vicuna, Koala, BELLE 등  
  - 오픈된 대화·지시추종 능력 향상을 위해 “LoRA”, “Instruction tuning” 기법 적용

> 중요성:  
> - **원본 학습 비용 절감**: 직접 수백억 파라미터를 학습시키지 않아도 됨  
> - **재현·비교**: 동일한 가중치로 모델 간 성능 비교 가능

### 2) 상용 API
- **OpenAI GPT-3/3.5/4** (ada, babbage, curie, davinci, gpt-3.5-turbo, gpt-4 등)  
  - RESTful API 형태로 제공  
  - Fine-tuning·Function call·Streaming 등 다양한 옵션 활용  
- **Anthropic Claude**  
  - GPT 계열과 유사한 API 사용 방식  
- **기타**(Cohere, AI21 등)

> 중요성:  
> - **인프라 부담 최소화**: 서버·GPU 없이 곧바로 호출 가능  
> - **최신 모델 사용**: 연구 동향에 맞춰 지속 업그레이드

***

## 3.2 사전학습용 대규모 코퍼스

### 1) 웹 크롤링 데이터
- **CommonCrawl/C4/RedPajama/RefinedWeb**  
  - 인터넷 전체 문서 수십~수백 TB 규모  
  - 잡음·스팸 제거를 위한 CCNet, 클리닝 파이프라인 필요

### 2) 책·아카데믹 데이터
- **BookCorpus, Gutenberg** (문학·교양서)  
- **arXiv, S2ORC** (논문 전문)  

### 3) 위키피디아
- 다양한 언어로 고품질 지식 정제되어 있음

### 4) 코드 데이터
- **GitHub, BigQuery, The Stack**  
  - 소스코드·코멘트·문서화 예제 대량 보유  
  - CodeSynth·수학 문제 해결 능력 향상에 기여

### 5) 혼합 데이터셋
- **The Pile, ROOTS, Dolma** 등  
  - 웹·책·코드·대화·학술 데이터를 섞어 다영역 학습

> 활용법:  
> 1. **필터링 → 중복 제거 → 토크나이저 학습**  
> 2. **데이터 스케줄링** (혼합 비율·커리큘럼)  
> 3. **사전학습 단계별 단계적 투입**

***

## 3.3 미세조정(Fine-tuning)용 데이터셋

### 1) 지시추종(Instruction tuning) 데이터
- **실제 NLP 태스크 포맷** (P3, FLAN, Super-Natural-Instructions)  
- **대화·챗 데이터** (ShareGPT, OpenAssistant, Dolly)  
- **자기-생성 합성 데이터** (Self-Instruct, Alpaca-52K, WizardLM, Baize)  

> 핵심:  
> - **이전 태스크 형식 + 자연어 지시어** → LLM 일반화 능력↑  
> - **데이터 품질·다양성**이 결과에 직결

### 2) 정렬(Alignment) 데이터
- **HH-RLHF, SHP, PKU-SafeRLHF** 등  
  - 도움됨/해로움 판별, 선호도 비교 라벨링  
  - 유해·편향·가짜정보 생성 억제용

> 활용법:  
> - **지도학습(SFT)**: 지시추종 → 응답 정제  
> - **RLHF**: 선호도 모델 학습 → PPO 등 RL로 미세조정

***

## 3.4 개발·배포 지원 라이브러리

- **Hugging Face Transformers**: Transformer 모델 구현·활용  
- **DeepSpeed, Megatron-LM, JAX, Colossal-AI, BMTrain**  
  - 3D 병렬, ZeRO, Mixed Precision, MoE 훈련 가속  
- **vLLM, DeepSpeed-MII, DeepSpeed-Chat**  
  - 추론·서비스 최적화, LLM 기반 챗봇 파이프라인  
- **그 외**: PyTorch FSDP, TensorFlow XLA, FlashAttention, PageAttention 등

> 목표:  
> - **학습 속도·효율 극대화**  
> - **메모리·비용 절감**, **실시간 응답** 보장

***

### 정리

LLMs를 연구·개발·운영하기 위해서는  

1. **사전학습 모델·API**(기본 토대)  
2. **방대한 코퍼스**(사전학습 원료)  
3. **미세조정 데이터**(지시·정렬 학습)  
4. **병렬·추론 라이브러리**(가속·배포)  

의 **4대 핵심 자원** 이 필요합니다.  
이 자원들을 어떻게 **선별·전처리·스케줄링**하여  
효율적으로 활용하느냐가 LLM 성공의 관건입니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f9db9a03-366c-4d91-aa33-8500e9ca5b19/2303.18223v15.pdf

# 4 사전학습 (Pre-training) 상세설명

LLMs(대형 언어 모델)의 **사전학습** 단계는 모델의 기본 능력인 언어 이해와 생성, 토큰 예측 능력을 획득하는 핵심 과정입니다. 4장에서는 크게 세 파트로 나누어, (1) 데이터 수집·전처리, (2) 모델 아키텍처, (3) 최적화·학습 기법을 다룹니다.

***

## 4.1 데이터 수집·준비(Data Collection and Preparation)

### 4.1.1 데이터 소스(Data Source)
- **일반 텍스트**  
  - 웹페이지(CommonCrawl, C4 등), 책(BookCorpus, Gutenberg), 대화 기록(Reddit)  
  - 대량·다양한 주제를 포괄해 언어 지식과 표현 패턴 학습에 필수
- **특수 도메인 텍스트**  
  - 멀티링구얼 데이터(46~122개 언어), 학술 논문(arXiv·PubMed), 코드(GitHub·StackOverflow)  
  - 계산·추론·전문지식 기반 태스크 성능 향상  

LLMs는 이들 다양한 데이터를 혼합해 사전학습 말뭉치를 구성하며, 모델별 비중 차이(예: LLaMA는 웹 80% + 코드·책·논문)가 존재합니다.

### 4.1.2 데이터 전처리(Data Preprocessing)
1. **품질 필터링**  
   - 어절·문장 길이, 특수문자, 언어 식별, 불건전 콘텐츠 키워드 제거  
   - 분류기 기반·휴리스틱 규칙(토큰 분포·문법성·perplexity) 사용  
2. **중복 제거**  
   - 문장·문서·데이터셋 수준 중복 필터링  
   - 중복이 많으면 모델이 반복 학습→성능 저하·프라이버시 위험  
3. **PII(개인식별정보) 제거**  
   - 이름·주소·전화번호 키워드 스팟팅 또는 분류기 활용  
4. **토크나이저 학습**  
   - BPE, WordPiece, Unigram 등 서브워드 토크나이저(문자 기반→어절 기반 OOV 해소)  
   - multilingual·byte-level BPE, RoPE 위치인코딩 고려  

### 4.1.3 데이터 스케줄링(Data Scheduling)
- **데이터 혼합 비율(Mixture)**  
  - 태스크 일반화↑ 위해 웹·책·논문·코드 등 다양한 소스 조합  
  - 태스크 특화(수학·코드)에선 해당 데이터 비중↑  
- **커리큘럼 학습(Curriculum)**  
  - 초기엔 일반·쉬운 예제→단계별로 특화·복잡한 예제 학습  
  - 코딩 특화: 웹→코드 데이터, 수학 특화: 코드→수학 데이터 스테이지별 순차 학습  
  - LLaMA→CodeLLaMA→Llemma(수학 강화) 연속 미세조정 예시

### 4.1.4 데이터 준비 요약
- 다양한 소스 전처리·중복 제거 필수  
- BPE·Unigram 등 맞춤 토크나이저 학습  
- 혼합 비율과 커리큘럼으로 일반화·특화 균형 조정  

***

## 4.2 모델 아키텍처(Architecture)

### 4.2.1 주요 아키텍처 유형
1. **인코더–디코더** (T5, BART)  
   - 입력 인코딩⇢출력 디코딩 분리, 주로 재구성·번역 태스크
2. **디코더 전용(Causal Decoder)** (GPT 시리즈, OPT, BLOOM)  
   - 좌→우 토큰 예측 언어 모델링에 최적화
3. **Prefix Decoder** (GLM)  
   - 앞부분은 양방향, 생성부 분리된 디코더 아키텍처
4. **Mixture-of-Experts** (Switch, GLaM)  
   - 희소 활성화로 파라미터 수↑, 연산량 유지

Emergent 아키텍처(SSM, Hyena 등)는 긴 문맥 병렬 처리 효율↑에 유망하지만, 아직 Transformer 성능엔 못미침.

### 4.2.2 세부 구성 설정
- **정규화 위치·방법**  
  - Pre-LN+RMSNorm → 안정적 학습 트렌드  
  - DeepNorm(초대형 모델)  
- **활성화 함수**  
  - GeLU/SwiGLU → 성능 우수  
- **위치 인코딩**  
  - RoPE, ALiBi(추론 시 길이 확장 강건성)  
- **어텐션 최적화**  
  - FlashAttention(메모리·속도 개선), Multi-query, sparse attention

### 4.2.3 사전학습 목적(Pre-training Tasks)
- **언어 모델링(LM)** (자연어→다양한 태스크 일원화)  

$$
    \mathcal{L} = -\sum_i \log P(x_i\mid x_{ < i})
  $$

- **Denoising Autoencoder (DAE)** (토큰 마스킹 복원)  
- **Mixture-of-Denoisers (UL2)**: LM + 짧은/긴 span 복원 믹스

### 4.2.4 디코딩 전략(Decoding)
- **Greedy vs. Sampling**  
  - Greedy(Deterministic), Temperature, Top-k, Top-p  
- **Beam Search + Length Penalty** (요약·번역)  
- **Contrastive Decoding**, **η-sampling** 등 후속 기법

### 4.2.5 아키텍처·목적 논의
- GPT 시리즈(디코더 전용+LM) 효율성·Few-shot 능력 증명  
- 인코더–디코더 모델 확장성·성능 비교 추가 연구 필요  

***

## 4.3 모델 학습(Model Training)

### 4.3.1 최적화 설정(Optimization)
- **배치 크기**: 32K→3.2M 토큰 동적 증가  
- **러닝레이트**: Warm-up(0.1–0.5%) → Cosine decay to 10%  
- **Adam/AdamW, Adafactor(메모리 최적화)**  
- **훈련 안정화**: Weight decay≈0.1, gradient clip≈1.0, spike 복구(restart)

### 4.3.2 규모 확장 기법(Scalable Training)
1. **3D Parallelism**: Data+Tensor+Pipeline 병행  
2. **Mixed Precision**: FP16/BF16 → 메모리·속도↑  
3. **Zero/FSDP, Activation Checkpointing** 등으로 메모리 절약

LLM 개발엔 GPU 수천 대·수개월 단위 계산 필요하며, 학습 중간 checkpoint 모니터링(예측 스케일링)으로 효율↑ 가능.

***

**정리**: 4장은 LLM 사전학습의 전체 과정을 다룹니다.  
1) **데이터 준비**: 다양한 고품질 말뭉치 수집·클리닝·커리큘럼 설계  
2) **아키텍처 선택**: Transformer 변형(디코더 전용, MoE, SSM 계열)  
3) **학습 전략**: 대규모 병렬화·혼용 정밀도·최적화 기법  

이 과정을 통해 LLM은 폭넓은 언어 지식과 강력한 생성·추론 능력을 획득합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f9db9a03-366c-4d91-aa33-8500e9ca5b19/2303.18223v15.pdf

# 5. POST-TRAINING OF LLMs

파인튜닝(pre-training 이후 추가 학습)은 대형 언어 모델(LLM)이 갖고 있는 **일반적 언어 능력**을 구체적인 **목적·용도**에 맞추어 조정하고 강화하는 단계입니다. 이 장에서는 크게 네 가지 주제로 나누어 설명합니다.

***

## 5.1 Instruction Tuning (명령어 학습)

### 5.1.1 “명령어-형” 학습 데이터 구성  
1) **NLP 태스크 데이터 변환**  
   - 기존의 분류·번역·요약 등 레이블이 달린 NLP 데이터셋을  
     “이 작업을 수행하세요: … 입력: … 출력:” 형태의 자연어 예시로 바꿉니다.  
2) **실제 대화 데이터 가공**  
   - 사람이 챗봇에게 묻고 답한 실제 대화(ShareGPT 등)를 모아  
     “이 질문에 답하세요: …” 식으로 포맷합니다.  
3) **합성(Self-Instruct) 데이터**  
   - 소수의 ‘시드’ 예시만 주고, 기존 LLM에게 “새로운 명령어를  생성하고 예시를 만들어 달라”라고 시켜 대규모 예시를 합성합니다.  

> 좋은 명령어 데이터를 만들 때는 ‘다양한 태스크를 포함해 규모를 키우기’, ‘복잡성을 적당히 조절해 난이도를 균형 맞추기’, ‘명확한 작업 설명(프롬프트) 작성’이 핵심입니다.

### 5.1.2 학습(튜닝) 전략  
- **데이터 배분**  
  여러 데이터셋을 섞어 쓸 때는, 너무 큰 데이터에 휘둘리지 않도록 ‘예시 비율 상한(capping)’을 둡니다.  
- **다단계 튜닝**  
  1차로 대규모 명령어 데이터로 튜닝한 뒤, 2차로 실제 대화체 데이터로 추가 튜닝하면 “대화 응답 능력”이 더 고도화됩니다.  
- **정규화용 원문 포함**  
  명령어 데이터에 원문(문장생성용 일반 텍스트)을 일부 섞어 정규화하면, 너무 특화되는 것을 막아줍니다.

### 5.1.3 튜닝 효과  
- **다양한 태스크에서 성능 대폭 향상**  
  명령어 학습만 거쳐도, zero/few-shot(사전학습된 상태)으로 여러 태스크를 잘 풀어냅니다.  
- **소규모 모델도 대형 모델 성능 따라잡음**  
  파라미터 7B 모델이 명령어 튜닝만 거쳐 GPT-3 175B를 넘보는 수치가 나옵니다.  
- **특정 도메인 특화**  
  의료·법률·산업용 등 전문 분야 명령어로 튜닝하면, 해당 분야 챗봇·Q&A 에 최적화됩니다.

***

## 5.2 Alignment Tuning (정렬 학습)

LLM이 “정직하고, 유해하지 않고, 유용하게” 행동하도록 **인간 가치**에 정렬하는 단계입니다. 크게 두 축으로 나뉩니다.

### 5.2.1 목표와 기준  
- **도움됨(Helpfulness)**: 사용자의 질문에 문제 해결·가이드를 잘 주는가  
- **정직함(Honesty)**: 사실 왜곡 없이 정확한 정보를 주는가  
- **무해함(Harmlessness)**: 차별·혐오·유해 콘텐츠를 피하는가  

### 5.2.2 인간 피드백 수집  
1) **순위 비교**  
   - LLM이 생성한 답변 두 개를 사람에게 보여주고 “더 나은 답은?”을 골라달라고 합니다.  
2) **세부 평가**  
   - 답변 품질(불편부당·정확성·유해성 등)에 대해 별도 항목별 점수를 매깁니다.  
3) **절차적 확인**  
   - “이 답변이 규칙(예: 욕설 금지)을 어겼나요?” 같이 규칙 위반 여부를 묻기도 합니다.

### 5.2.3 RLHF: Reinforcement Learning from Human Feedback  
세 단계로 진행됩니다.  
1) **(선택) SFT 초기 튜닝**: 사람이 쓴 예시(질문-답변)를 모아 지도학습으로 미리 튜닝  
2) **보상 모델 학습**: 사람이 순위를 매긴 답변쌍으로 “어느 쪽이 더 좋은가”를 예측하는 보상 모델을 학습  
3) **RL 최적화(PPO)**: 보상 모델의 평가(보상)를 최대화하도록 LLM을 강화학습으로 미세조정  

> 주요 트릭: 보상 모델은 큰 모델보다 약간 작은 모델이 효율, 안정화. 보상이 지나치면 divergence 방지를 위해 **KL 페널티**를 붙입니다.  
> RLHF를 거치면 “정직·무해” 성능이 크게 뛰지만, 학습이 복잡하고 불안정해 잘 튜닝하기 어렵습니다.

### 5.2.4 SFT vs. RLHF 비교  
- **SFT(지도학습)**: 행동 복제(Imitation)—“이 질문엔 이 답변을 하라” 토큰 단위로 학습  
  - 싸고 안정적, 소량 예시로 충분, 하지만 새로운 능력 획득엔 한계  
- **RLHF(강화학습)**: 결과에 보상 부여—“사람이 더 선호하는 답을 내놓아라”  
  - 더 자연스러운 정렬, 부정적 예시도 활용, 하지만 설정·학습 어렵고 불안정  

***

## 5.3 Parameter-Efficient Model Adaptation

LLM 전체 파라미터를 매번 업데이트하는 건 비용이 크므로, **일부 파라미터만** 추가 조정하는 기법들이 개발되었습니다.

1) **Adapter** (저차원 bottleneck 네트워크 삽입)  
2) **Prefix/Prompt Tuning** (입력 또는 레이어별 ‘soft prompt 시퀀스’만 학습)  
3) **LoRA** (백분산 업데이트를 저랭크 행렬 A·Bᵀ로 근사)  

> 대체로 수백만∼수천만 파라미터만 학습해도 전체 파라미터 학습 못지않은 성능이 나옵니다. LLaMA 계열, BLOOM 등의 LLM에도 LoRA가 널리 쓰입니다.

***

# 요약

- **5.1 Instruction Tuning**: 자연어 명령어-예시를 모아 LLM을 지도학습으로 튜닝  
- **5.2 Alignment Tuning**: 사람 평가(선호도·규칙 위반)로 보상 모델을 만들고 RLHF로 정렬  
- **5.3 효율적 튜닝**: Adapter/Prefix/Prompt/LoRA로 일부 파라미터만 업데이트  

이 과정을 통해 LLM은 사전학습된 “기초 언어 능력”에서 나아가,  
1) *사람이 원하는 방식으로 응답*하고,  
2) *특정 분야나 응용에 맞게* 동작하며,  
3) *리소스 절약* 학습까지 가능해집니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f9db9a03-366c-4d91-aa33-8500e9ca5b19/2303.18223v15.pdf
