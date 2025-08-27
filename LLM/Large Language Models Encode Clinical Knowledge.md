# Large Language Models Encode Clinical Knowledge

**핵심 주장**  
- 대형 언어 모델(LLM)은 의학·임상 지식을 대규모 말뭉치 학습만으로도 효과적으로 내장할 수 있으며, 적절한 prompting과 instruction tuning을 통해 의료 질문 응답 성능을 크게 향상시킬 수 있다.  
- Flan-PaLM(명령어 튜닝된 PaLM)과 새로 제안된 instruction prompt tuning(소수의 도메인 예시로 소프트 프롬프트를 학습)을 결합하면, 다중의료 QA 벤치마크에서 SOTA 성능을 달성하고, 인간 임상의사에 근접하는 안전성·정확성을 확보할 수 있다.

**주요 기여**  
1. **MultiMedQA 벤치마크**:  
   - USMLE(미국의사시험), AIIMS/NEET(인도 의대입시), PubMedQA, MMLU 임상 분야, LiveQA, MedicationQA, 신규 HealthSearchQA 총 7개 데이터셋 통합.  
2. **Flan-PaLM 다중 선택 및 폐쇄 도메인 QA**:  
   - Few-shot, chain-of-thought(CoT), self-consistency prompting을 결합해 MedQA 67.6%로 종전 SOTA 대비 +17%p 향상.  
3. **Instruction prompt tuning**:  
   - 소프트 프롬프트 벡터(길이 100, ≈1.84M 파라미터)만 학습하고 나머지 PaLM 540B 고정  
   - Flan-PaLM 위에 40개의 clinician 제작 예시를 사용하여 의료 도메인 alignment  
   - Med-PaLM→의료 소비자 QA에서 과학적 근거 일치율 92.9%(Flan-PaLM 61.9%), 잠재적 해악율 5.9%(Flan-PaLM 29.7%)  
4. **Human evaluation 프레임워크**:  
   - 과학적 합의, 오독·오추론·오회수, 누락·부적절 정보, 해악 가능성, 편향 등 12개 축 clinician 평가  
   - 비전문가 평가(도움·질문 의도 충족)  

# 논문 상세 분석

## 1. 해결 문제  
- 기존 자동화 QA 벤치마크는 단일선택 정확도 혹은 BLEU 등 기계 번역 지표에 국한  
- 장문의 의료 답변의 사실성·안전성·편향·유해성 평가 미비  
- 일반 LLM(Flan-PaLM)만으로는 의료 소비자 질문에 대해 임상 합의·안전성 확보에 한계

## 2. 제안 방법  
1) **Prompting 전략**  
   - Few-shot: 데이터셋별 임상 예시 3~5개  
   - CoT: 중간 추론 스텝을 포함하는 토크나이즈된 체인-오브-생각  
   - Self-consistency: n=11개의 CoT 샘플 다수결  
2) **Instruction prompt tuning**  
   - 학습 파라미터: soft prompt length = 100, 총 1.84M  
   - 학습 데이터: LiveQA·MedicationQA·HealthSearchQA에서 임상의 5명 제작 예시 40개  
   - 옵티마: AdamW, lr=0.003, weight_decay=1e-5, batch=32, 200 steps  
   - 하이퍼파라미터: human-in-the-loop로 held-out 예시 평가 기반 선택  
3) **Formalism**  
   - 입력: [SoftPrompt] + [HardPrompt(dataset-specific instructions & few-shot CoT)] + [질문/문맥]  
   - 고정된 LLM 파라미터 θ, 학습 대상만 soft prompt φ  
   - 최적화: minimize _L_(MedQA∪…) w.r.t. φ

## 3. 모델 구조  
- 백본: PaLM 540B → Flan-PaLM 540B(명령어 튜닝) → Med-PaLM(soft prompt tuning)  
- Soft prompt: θ₁,…,θ₁₀₀ ∈ ℝ¹⁸⁴³², frozen PaLM 임베딩에 prepended  
- Hard prompt: 5-shot 예시마다 task-specific 지시+CoT 예시  
- 인퍼런스: soft+hard 프롬프트 내장 후 greedy + temperature 샘플링

## 4. 성능 향상  
- MedQA(4-choice): Flan-PaLM 540B 67.6%, 기존 SOTA 대비 +17.3%p  
- MedMCQA: 57.6% vs Galactica 52.9%  
- PubMedQA: 79.0% vs BioGPT 78.2%  
- MMLU 임상 토픽: 최대 88.9% (Galactica 75.4%)  
- 소비자 장문 QA(HealthSearchQA, LiveQA, MedicationQA)  
  - 과학 합치율: Flan-PaLM 61.9% → Med-PaLM 92.9%  
  - 잠재적 해악율: 29.7% → 5.9%  
  - 편향률: 7.9% → 0.8%  
  - 전문가 답변 수준 근접

## 5. 한계  
- 벤치마크: 영어·QA형에 국한, EMR QA·다국어 미포함  
- human evaluation: 주관적 지표·소수 평가자·단일 평가 설계  
- 컨센서스: 시시각각 변화하는 의료 지식 반영 어려움  
- soft prompt tuning이 누락 감소↔불필요한 정보 과다생성 trade-off

# 일반화 성능 향상 가능성  
- Soft prompt tuning은 도메인 적응 시 전체 모델 재학습 필요 없이 예시 40개만으로 효과  
- 다른 의료 태스크(e.g. EMR 추출, 판독 보고서 생성)로 확장 가능  
- 벤치마크 외 신규 도메인 질문에도 빠른 적응력: 최소 φ 크기와 few-shot 예시만 변경  
- 멀티태스크 instruction prompt로 여러 하위 의학 과목 동시 학습 가능  
- Continual learning: 의료 지식 업데이트를 soft prompt 교체만으로 반영

# 향후 영향 및 고려 사항  
- **임상 응용**: LLM 기반 의료 도우미 개발 시 soft prompt tuning methodology를 경량 alignment 기법으로 활용 가능  
- **벤치마크 확장**: 전자의무기록·다국어·영상판독 QA 포함하는 MultiMedQA 2.0 필요  
- **안전성·편향 평가**: 대규모·다양한 임상의·환자군 참여 human evaluation 설계  
- **실시간 업데이트**: 의료 가이드라인·논문 출판 시점 반영하는 dynamic soft prompt 관리  
- **불확실성**: self-consistency 기반 confidence estimation을 임상 deferral 시스템에 응용

————  
향후 연구에서 soft prompt tuning을 통해 빠르게 의료 전문 지식을 내장하고, dynamic update·cross-lingual·multi-modal 대응을 위한 lightweight alignment 전략 연구에 집중해야 한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b93abaf9-e2af-48d3-9395-892b99721d30/2212.13138v1.pdf)
