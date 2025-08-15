# 핵심 요약

■ 대규모 언어 모델(LLM)의 **핵심 주장**  
  – 트랜스포머 기반의 LLM(예: GPT, LLaMA, PaLM)은 매개변수 수와 학습 데이터 규모 확장에 따라 언어 이해·생성 능력뿐 아니라 *in-context learning*, *instruction following*, *multi-step reasoning* 등 기존 모델에 없던 **emergent abilities**을 보임.  
  – 추가적인 외부 지식·도구(Tools)와의 융합, 강화학습(특히 RLHF)으로 **정렬(alignment)** 및 **추론 강화**가 가능하며, 이를 통해 일반 목적 AI 에이전트 개발이 현실화되고 있음.

■ 주요 기여  
  1. **LLM 계보 정리**: GPT, LLaMA, PaLM 세 모델 패밀리와 그 파생 모델(Alpaca, Vicuna 등), 기타 대표 LLM(FLAN, Gopher, RETRO, GLaM 등)의 아키텍처·매개변수·학습 데이터·오픈소스 여부를 비교 정리.  
  2. **구축 기법 총망라**:  
     – 데이터 준비: 웹 크롤링→필터링·중복 제거  
     – 토크나이저(BPE, WordPiece, SentencePiece)와 포지셔널 인코딩(절대·상대·RoPE·ALiBi)  
     – 사전학습(optimize scaling laws, MoE, autoregressive vs MLM)  
     – 파인튜닝·인스트럭션 튜닝·정렬(RLHF, DPO, KTO)  
     – 디코딩 전략(Beam, Top-k/p, Temperature)  
  3. **사용·확장 방법 정리**:  
     – 고급 프롬프트(Chain-of-Thought, Tree-of-Thought, Self-Consistency, Reflection, Expert Prompting, Rails, 자동 프롬프트 엔지니어링)  
     – Retrieval-Augmented Generation(RAG)과 외부 도구 연동  
     – LLM 에이전트(도구 사용·의사결정·다중 에이전트 협업 프레임워크)  
  4. **벤치마크·데이터셋**: 기본 언어 이해·생성·추론·코드 생성·멀티모달까지 50여 개 주요 벤치마크와 데이터셋(TriviaQA, MMLU, GSM8K, HumanEval, HotpotQA 등) 소개 및 주요 모델 성능 비교.  
  5. **미래 과제 제시**:  
     – 경량·효율적 모델 설계(PEFT, distillation, SSM)  
     – 애프터-어텐션(SSM, Hyena, Monarch Mixer)  
     – 멀티모달 통합  
     – LLM 기반 에이전트  
     – 보안·윤리·책임 있는 AI  

# 문제 정의 및 해결 방법

1. 문제: **LLM의 급격한 발전**으로  
   – 어떤 모델이 어떤 설계·학습 기법을 써서 어떤 능력을 얻었는지,  
   – 기존 작은 PLM과 차별점(emergent abilities)은 무엇인지,  
   – 실제 응용(도구 활용, RAG, 에이전트)과 한계는 무엇인지  
   개발자·연구자가 파악하기 어려움.

2. 제안된 방법: **최신 문헌 종합 리뷰**  
   – 모델별 아키텍처·학습 데이터·파인튜닝·정렬·디코딩 전략 비교  
   – 주요 벤치마크 성능 테이블화(예: GPT-4 BAREXAM 상위 10%)  
   – 프롬프트·RAG·에이전트 활용 기술 체계 정리  
   – 경량화·멀티모달·보안·윤리 등 미래 연구 방향 제시

3. 모델 구조  
   (1) **공통 기반**: Transformer Encoder/Decoder  
   (2) **가변적 구성 요소**:  
     – **모델 크기**: 7B~1.76T 파라미터  
     – **포지셔널 인코딩**: 절대/상대/Rotary/ALiBi  
     – **MoE**: expert 수 &amp; 라우터  
     – **정렬**: RLHF, DPO, KTO  
     – **디코딩**: Beam, Top-k/p, Temperature  

4. 성능 향상  
   – 파라미터·데이터 규모 확대에 따른 **스케일링 법칙**[1]
   – **Instruction tuning**: zero-/few-shot 성능 대폭 개선  
   – **Chain-of-Thought**: 복합 추론 정답률↑ (예: GSM8K 8→58%)  
   – **RAG**: 외부 지식 활용해 최신·전문 지식 응답 가능  

5. 한계  
   – **환각(hallucination)**: 자동 FACT-CHECK·정렬에도 잔존  
   – **거대 모델 제약**: 연산·메모리·지연시간 부담  
   – **보안·윤리**: 악의적 사용·편향·프라이버시 위험  
   – **장기 문맥**: 트랜스포머 한계, SSM 대안 필요  

# 일반화 성능 향상 관점

– **Instruction tuning &mdash;** 다양한 과제로부터 일반화 학습, 새로운 태스크 제시만으로 전이  
– **Chain-of-Thought & Self-Consistency & Reflection & Expert prompting & Rails & APE**  
  &nbsp;&nbsp;-  복합 추론·비교·자기평가하여 *re-prompting*, *검증 강화*  
  &nbsp;&nbsp;-  프롬프트 자동생성·스코어링으로 **도메인 적응**  
– **RAG & Tools integration**  
  &nbsp;&nbsp;-  지식-도구 호출로 최신 정보·전문 지식 일반화  
  &nbsp;&nbsp;-  플러그인화된 모듈화 → 새 도메인·태스크 확장  
– **Alignment(DPO/KTO)**  
  &nbsp;&nbsp;-  paired preference 없이도 인간 선호 학습→출력 신뢰성↑  
– **PEFT/LoRA/Distillation**  
  &nbsp;&nbsp;-  가치 있는 파라미터만 적응·수렴→도메인 소규모 데이터로 일반화 fine-tuning  

# 향후 연구 영향 및 고려사항

1. **효율적·경량 LLM**  
   – SSM, MoE, distillation, quantization으로 거대 모델 대체  
   – 특수 도메인·임베디드 환경 일반화 최적화

2. **멀티모달 통합 AI 에이전트**  
   – 텍스트·이미지·비디오·음성 동시 이해·생성  
   – RAG·도구·다중 에이전트 협업으로 전방위 응용

3. **책임 있는 AI**  
   – 환각·편향 검증 메커니즘 강화(automated fact‐check ↔ human in the loop)  
   – 보안·프라이버시·윤리 가이드라인 수립

4. **평가·벤치마크 확장**  
   – 실제 산업 태스크 기반 벤치마크  
   – 장기 문맥·대화형 멀티턴 평가  

5. **개방형 소스·협업 플랫폼**  
   – 연구 재현성 위해 오픈소스 모델·데이터·도구 확산  
   – 공동 에이전트·맞춤형 확장 연구 지원  

> **결론**: 이 설문은 LLM 연구·응용 전반의 판도와 기술 스택을 총망라하며, 특히 **추론 일반화**와 **실세계 도구 융합**을 통해 “범용 AI 에이전트” 실현 로드맵을 제시한다. 후속 연구는 **효율성·책임성·멀티모달** 역량 강화에 중점을 두어야 한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/66aeecf9-f460-4ea4-a02e-de2b23909bb8/2402.06196v3.pdf

## II. LARGE LANGUAGE MODELS

이 섹션에서는 대규모 언어 모델(LLM)의 기원을 이루는 “Early Pre-trained Neural Language Models” 를 먼저 살펴보고, 이후 세 가지 주요 LLM 계열인 **GPT**, **LLaMA**, **PaLM**을 중심으로 그 구조·특징·진화 과정을 정리한다.  

### A. Early Pre-trained Neural Language Models  
1. **통계 기반 언어 모델 (SLMs)**  
   - n-그램 모델: 단어열을 순서대로 확률 곱으로 표현.  
   - 한계: 희소성(sparsity) 문제로 희귀 n-그램에 0확률 부여 → 부정확  

2. **초기 신경망 언어 모델 (NLMs)**  
   - 단어를 연속 벡터로 임베딩하고, RNN/LSTM/GRU로 다음 단어 예측  
   - 장점: 분포형 표현으로 희소성 극복, 의미 유사도 캡처  

3. **사전학습 언어 모델 (PLMs)**  
   - 대규모 무라벨 텍스트로 Transformer 기반 모델을 **사전학습**(next-token / MLM)  
   - 이후 소규모 데이터로 각 과제에 **파인튜닝**  
   - 대표작: BERT, RoBERTa, ALBERT, ELECTRA, XLNet, T5, BART  

4. **Transformer의 핵심 기법**  
   - **Self-Attention**: 각 토큰이 문맥 내 다른 토큰과의 가중합으로 표현  
   - **포지셔널 인코딩**: 위치 정보 부여 → 절대(sin-cos), 상대, Rotary, ALiBi 등  

***

### B. 세 가지 주요 LLM 계열

1. **GPT 계열 (OpenAI)**  
   - **GPT-1/2**: 디코더-전용 Transformer, next-token 예측으로 사전학습 후 개별 과제 파인튜닝  
   - **GPT-3 (175B)**: **Few-shot in-context learning** 최초 시연  
   - **InstructGPT / ChatGPT (GPT-3.5)**: RLHF로 “사람 지침 준수” 강화  
   - **GPT-4**: 멀티모달(+이미지) 입력, 전문 시험에서 상위 10% 성적  

2. **LLaMA 계열 (Meta)**  
   - **LLaMA-1 (7B–65B)**: 가벼운 모델링(“SwiGLU”, “Rotary emb.”)으로 GPT-3 대비 우수 성능  
   - **LLaMA-2 & LLaMA-2 Chat**: 공개 라이선스, RLHF 기반 대화용 파인튜닝  
   - 파생모델: Alpaca, Vicuna, Koala, Guanaco, Mistral-7B 등—오픈소스 생태계 활성화  

3. **PaLM 계열 (Google)**  
   - **PaLM (540B)**: TPU Pod 위 Pathways 시스템으로 사전학습  
   - **Flan-PaLM**: 1,800+ 과제 인스트럭션 튜닝→“제로/퍼셉션적 추론” 대폭 향상  
   - **PaLM-2**: 경량화·다국어성·추론력 개선, 멀티모달 확장  
   - **Med-PaLM**: 의료 도메인 특화, MedQA 86% 성적  

***

### C. 기타 대표 LLM 및 비교

- **FLAN**: 60개 과제 인스트럭션 튜닝으로 제로-샷 성능 강화  
- **Gopher / Chinchilla**: 스케일법칙 검증, “토큰 vs 파라미터” 최적점 제시  
- **RETRO**: 문서 검색결과(conditioning) 결합하여 샘플 효율↑  
- **GLaM / Mixture-of-Experts**: 희소 활성화 전문가 집합으로 1.2T 규모 모델 학습  
- **LaMDA, OPT, BLOOM, ERNIE, Galactica, CodeGen, …**: 다양한 목적·규모·라이선스 모델  

> 다양한 규모(수백만→수조 파라미터)·구조(디코더·인코더·엔코더-디코더)·학습 기법(MLM, autoregressive, MoE, RLHF)이 공존하며, 각자 강점(추론, 코드·의료·멀티모달, 오픈소스 접근성)을 가진다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/66aeecf9-f460-4ea4-a02e-de2b23909bb8/2402.06196v3.pdf

## III. HOW LLMS ARE BUILT

이 절에서는 대규모 언어 모델(LLM) 개발의 핵심 단계와 구성 요소를 목차별로 정리한다.

### A. 주요 아키텍처

1. **Transformer**
    - Self-Attention 기반으로 입력 시퀀스 내 토큰 간 연관도 계산
    - 인코더·디코더 스택 구조, 다중 헤드·피드포워드 서브레이어
2. **Encoder-Only** (e.g. BERT 계열)
    - 전체 문맥을 동시에 인코딩 → 문장 분류·질문응답 등 이해 과제에 특화
    - Masked Language Modeling, Next Sentence Prediction
3. **Decoder-Only** (e.g. GPT 계열)
    - 좌측 컨텍스트만 참조하며 순차 생성(Autoregressive) → 자유 생성 과제에 특화
    - Next-Token Prediction
4. **Encoder-Decoder** (e.g. T5, BART)
    - 인코더로 입력 이해 → 디코더로 조건부 텍스트 생성
    - 번역·요약·대화 등 시퀀스→시퀀스 과제에 활용

### B. 데이터 정제

1. **노이즈 제거**
    - 중복·스팸·오류·불법 콘텐츠 필터링
2. **중복 제거**
    - n-그램 유사도 등으로 문서·문단 레벨 중복 샘플 축소
    - 데이터 희소성 완화, 오버피팅 방지

### C. 토크나이제이션

1. **Byte-Pair Encoding (BPE)**
    - 자주 등장하는 바이트 쌍 병합 → 어휘표 크기 절충
2. **WordPiece**
    - 어절 기반으로 최댓값 우도 추정해 서브워드 분할
3. **SentencePiece**
    - 공백 전처리 없이 문자 단위로 서브워드 학습 → 노이즈·비표준 텍스트 대응

### D. 위치 정보 인코딩

1. **절대 인코딩 (sin-cos, Learned)**
2. **상대 인코딩**
    - 키·쿼리에 거리 기반 가중치 추가
3. **Rotary (RoPE)**
    - 위치마다 회전 행렬 적용, 절대·상대 정보 동시 반영
4. **ALiBi**
    - 거리 비례 선형 바이어스 추가 → 훈련보다 긴 컨텍스트 일반화

### E. 사전학습

1. **Autoregressive LM**
    - $p(x_i\mid x_{<i})$ 최대화
2. **Masked LM**
    - 문장 일부 마스킹 → 원문 예측(denoising)
3. **Mixture-of-Experts**
    - 희소 활성화 전문가 네트워크(FFN 여러 개) → 대규모 파라미터, 낮은 추론 비용

### F. 파인튜닝 \& 인스트럭션 튜닝

1. **Supervised Fine-Tuning**
    - 과제별 데이터로 LM 미세 조정 → 성능·효율 개선
2. **Instruction Tuning**
    - ‘지침(prompt)’-→응답 매핑 데이터로 파인튜닝 → 사용자 의도 준수 강화

### G. 얼라인먼트 (Alignment)

1. **RLHF** (Reinforcement Learning from Human Feedback)
    - 사람 평가로 보상 모델 학습 → RL로 LLM 정책 최적화
2. **DPO** (Direct Preference Optimization)
    - 페어 선호도→분류 손실로 직접 최적화(보상 모델 생략)
3. **KTO** (Kahneman–Tversky Optimization)
    - 바람직/비바람직 레이블만으로 정렬

### H. 디코딩 전략

1. **Greedy Search**
    - 매 스텝 최댓값 토큰 선택
2. **Beam Search**
    - 상위 N개 후보 유지→전체 점수 최적화
3. **Top-k / Top-p Sampling**
    - 상위 k개 / 확률 질량 p 이하 토큰에서 무작위 샘플링
4. **Temperature**
    - 로짓 조정으로 무작위성 제어

### I. 비용·컴퓨트 효율화

1. **ZeRO**
    - 분산 훈련 메모리 중복 제거 → 스케일 아웃
2. **RWKV**
    - 선형 attention + RNN 구조 → 추론·훈련 효율 개선
3. **LoRA**
    - 증분 업데이트를 저랭크 행렬로 표현 → 파라미터·저장공간 절감
4. **지식 증류**
    - 대형→소형 모델 전이 학습 → 경량화
5. **양자화**
    - FP32→Int8 등 → 모델 크기·추론 속도 최적화

> **핵심 요약**
> - Transformer 기반 인코더·디코더 구조
> - 대규모 무라벨 코퍼스 사전학습 + 과제별 파인튜닝/인스트럭션 튜닝
> - 얼라인먼트(RLHF·DPO·KTO)로 안전·정확도 강화
> - 디코딩·효율화(ZeRO·LoRA 등)로 실전 배포 최적화
