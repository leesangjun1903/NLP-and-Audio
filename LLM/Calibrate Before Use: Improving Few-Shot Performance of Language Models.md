# Calibrate Before Use: Improving Few-Shot Performance of Language Models

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
언어 모델을 이용한 few-shot 학습(“in-context learning”)은 프롬프트 형식, 예시 선택 및 순서에 크게 의존하며 그로 인해 성능 변동성이 매우 높다. 이를 완화하기 위해 모델의 출력 분포를 **컨텍스트별 편향(contextual bias)** 으로 추정하여 사전 보정(calibration)하는 **컨텍스트 보정(Contextual Calibration)** 기법을 제안한다.  

**주요 기여**  
- few-shot 프롬프트가 겪는 **다양한 편향**(majority label bias, recency bias, common token bias)을 체계적으로 분석.  
- **컨텍스트 보정**: “N/A” 같은 내용 없는 입력에 대한 모델 예측으로 편향을 추정한 뒤, 다음과 같이 확률을 균등 분포로 맞추는 diagonal scaling 기법을 도입:  

$$
\hat q = \mathrm{softmax}\bigl(\mathrm{diag}(\hat p_{\mathrm{cf}})^{-1}\,\hat p + \mathbf{0}\bigr),
$$  

여기서 $$\hat p_{\mathrm{cf}}$$는 내용 없는 입력에 대한 원본 확률 벡터.  
- 대규모 실험(문장 분류, 사실 회수, 개체 추출)에서 평균 정확도 최대 **30%p 상승**, 분산 감소, 작은 모델이 대형 모델 성능을 추월하는 사례 제시.  

## 2. 문제 정의 및 제안 방법  

### 2.1 해결하고자 하는 문제  
Few-shot in-context learning은  
1. 동일 프롬프트 내 레이블 분포 불균형 시 모델이 다수 레이블을 선호(majority label bias)  
2. 프롬프트 끝쪽 예시 레이블을 과도하게 반영(recency bias)  
3. 사전학습 빈도가 높은 토큰에 편향(common token bias)  
등으로 인해 성능이 프롬프트 구성에 매우 민감하며, 순서 변경만으로도 정확도가 50%p 이상 요동친다.  

### 2.2 제안하는 방법: Contextual Calibration  
- **편향 추정**: 프롬프트에 “N/A”, “[MASK]”, 빈 문자열 등 **내용 없는 입력(content-free input)** 을 삽입하여 모델이 레이블별로 어떤 확률을 출력하는지 $$\hat p_{\mathrm{cf}}$$를 구함.  
- **Diagonal Scaling**: 학습 데이터 없이도 편향 보정을 위해 가중치 행렬 $$W = \mathrm{diag}(\hat p_{\mathrm{cf}})^{-1}$$, 편향 벡터 $$b = \mathbf{0}$$ 로 설정.  
- **보정된 예측**:  

$$
\hat q = \mathrm{softmax}(W\,\hat p + b)
$$  

을 통해 테스트 입력에 대한 예측 확률 $$\hat q$$를 계산한 뒤, argmax로 레이블 결정.  

### 2.3 모델 구조  
- **기존 GPT-3/GPT-2** 구조에 변경 없음  
- **사후 처리(post-processing)** 단계로 위 보정 모듈만 추가  

## 3. 성능 향상 및 한계  

### 3.1 성능 향상  
- **평균 정확도** 최대 30.0%p 상승  
- **분산**(prompt별 정확도 표준편차) 크게 감소  
- GPT-3 2.7B가 보정 후 GPT-3 175B의 베이스라인을 뛰어넘는 경우도 관찰  
- **0→1-shot** 에서 성능 하락 문제(majority label bias) 완화  

### 3.2 한계  
- 완전한 편향 제거는 아니며, **프롬프트 형식 자체 최적화**(prompt engineering) 필요성 완전히 해소하지 못함  
- **content-free input** 선택에 따른 성능 차이 존재  
- **open-ended generation** 과 같은 비분류 과제에 확장성 미검증  
- 사후 처리이므로 **편향의 근본적 원인**(모델 학습 과정) 해소에는 한계  

## 4. 모델의 일반화 성능 향상 가능성  
- 컨텍스트 보정을 통해 **프롬프트 간 일관성**을 높여, 새로운 예시나 형식에 대해 보다 안정적 예측 가능  
- 특히 소규모 모델일수록 편향 보정 효과 큼 → **경량 모델 일반화**에 유리  
- **다양한 과제**(분류→회수→추출)에서 효과 검증되어, 다른 NLP 태스크에도 적용 여지  
- 그러나 **태스크별 최적 content-free input** 을 찾는 과정이 필요해 일반화 파이프라인 구성 시 고려 요망  

## 5. 향후 연구 영향 및 고려사항  
- **기존 few-shot 비교 연구** 시 “보정 전·후”를 함께 보고해야 공정 비교 가능  
- 프롬프트 편향 분석을 바탕으로 **프롬프트 자동화 도구** 개발 가능  
- **finetuning· 보정의 상호작용** 연구: 사전학습 미세조정 후 보정이 시너지 내는지 탐구  
- **비분류(open-ended) 및 다중 토큰 생성** 과제로의 확장: 첫 토큰 이후 편향 누적 문제 해결 방안 필요  
- **사전학습 편향 완화** 연구: 보정 없이도 내재적 편향을 적게 갖는 언어 모델 설계 방향 제시

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/32e0c9cd-bf90-4a1e-9cfe-a9612906dcb4/2102.09690v2.pdf)
