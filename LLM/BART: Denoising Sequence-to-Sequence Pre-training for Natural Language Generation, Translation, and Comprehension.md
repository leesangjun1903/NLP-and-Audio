# BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

## 1. 핵심 주장 및 주요 기여  
BART는 입력 문장을 임의의 노이즈 함수로 손상한 뒤, 원문을 재구성하도록 학습하는 **순차-순차(Seq2Seq) Denoising Autoencoder** 모델이다.  
- BERT(양방향 인코더)와 GPT(오토리그레시브 디코더)를 통합하여, 마스킹·삭제·문장 순서 셔플·텍스트 인필링 등 다양한 노이즈를 허용  
- 텍스트 생성(task)과 이해(task)를 단일 모델로 지원하며, 생성 과제에서 특히 뛰어난 성능 향상(최대 +6 ROUGE, +1.1 BLEU)  
- GLUE·SQuAD 분류 및 추출 과제에서도 RoBERTa와 유사한 성능 유지  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 해결 과제  
기존 프리트레이닝은  
- BERT: 마스킹 후 비자동회귀 예측 → 생성 능력 제한  
- GPT: 좌방향만 조건화 → 이해(task) 제약  
- SpanBERT·MASS 등: 특정 노이즈에 종속적  

→ **통합적인 양방향 이해와 자동회귀 생성을 모두 만족하는 일반화 가능한 프리트레이닝 모델** 필요  

### 2.2 제안 방법  
1) Text Corruption: 입력 $$x$$에 다음 변환을 임의 조합  
   - Token Masking/Deletion  
   - Sentence Permutation  
   - Text Infilling (Poisson($$\lambda=3$$)로 추출된 스팬을 하나의 [MASK]로 교체)  
2) Reconstruction Loss: 디코더가 원문 $$y$$를 생성하도록 교차엔트로피 손실 최적화  
   
$$
     \mathcal{L}(\theta) = -\sum_{t=1}^{|y|} \log p_\theta(y_t \mid y_{ < t}, \mathrm{encode}(f(x)))
   $$  
   
여기서 $$f$$는 노이즈 함수, 인코더는 양방향 Transformer, 디코더는 좌→우 Transformer  

### 2.3 모델 구조  
- Encoder: 12-layer 양방향 Transformer (RoBERTa와 유사)  
- Decoder: 12-layer 자동회귀 Transformer (GPT 유사) + cross-attention  
- 대형 버전: hidden size 1024, batch 8,000, 500K steps pre-training  

### 2.4 성능 향상  
- **요약**(CNN/DailyMail, XSum): ROUGE-1 기준 44.16 → 45.14 (+6점)[Table 3]  
- **질의응답**(ELI5): ROUGE-L 23.1 → 24.3 (+1.2)[Table 5]  
- **번역**(WMT RO–EN): 백트랜슬레이션 대비 +1.16 BLEU[Table 6]  
- **분류**(GLUE·SQuAD): RoBERTa와 동등[Table 2]  

### 2.5 한계  
- Text Infilling 없이 단일 노이즈(문장 셔플·회전)는 성능 저하  
- 매우 추상적 생성(ELI5)에서는 순수 LM에 근접하는 성능  
- 기계번역·대규모 다국어 적용 시 과적합 위험  

## 3. 일반화 성능 향상 관점  
- **다양한 노이즈 학습**: 여러 노이즈 조합으로 *robustness* 확보  
- **양방향→단일 디코더**: 분류·생성 모두에 대응  
- **스팬 인필링**: 고정 길이가 아닌 가변 스팬 예측으로 범위 인식 능력 강화  
→ 새로운 도메인·태스크 전이 시 과제별 마스킹 비율·노이즈 유형 조정으로 일반화 최적화 가능  

## 4. 향후 연구에의 영향 및 고려사항  
- **노이즈 설계 연구**: 도메인 특화 corruption 함수 탐색  
- **다국어·코드·멀티모달 확장**: 비영어·프로그래밍·비주얼 데이터 적용 실험  
- **과적합 방지 기법**: 백트랜슬레이션 외 regularization, 노이즈 비율 스케줄링  
- **효율화**: 경량·지식 증류로 대형 BART의 실제 서비스 적용  

BART는 NLP 전 영역에서 **통일된 프리트레이닝 패러다임**을 제시하며, 이후 다양한 변종 연구 및 응용 연구의 기반이 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0f64a7d8-7f20-46fe-be23-9bf7945e56de/1910.13461v1.pdf
