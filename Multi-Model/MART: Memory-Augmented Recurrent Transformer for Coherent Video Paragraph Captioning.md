# MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning

**핵심 주장 및 주요 기여**  
MART는 영상 이벤트 시퀀스로부터 다문장 문단 캡션을 생성할 때 발생하는 *문맥 단절*, *중복*, *불일치* 문제를 해결하기 위해 제안된 모델이다. Transformer 기반의 *통합 인코더-디코더* 구조에 외부 메모리 모듈을 추가하여, 이전 영상 세그먼트 및 문장 히스토리를 요약된 메모리 상태로 저장·갱신함으로써, 이후 문장 생성 시 참조하도록 설계되었다. 이를 통해 문장 간의 **일관성(coherence)**과 **반복 최소화**를 동시에 달성하며, 영상-문장 관련성(relevance) 손실 없이 성능을 크게 향상시켰다.[1]

***

## 1. 해결하고자 하는 문제  
기존 Transformer 기반 문단 캡션 모델(Zhou et al. 2018)은  
1) 각 영상 세그먼트를 독립적으로 디코딩 → 문장 간 문맥 정보 부재  
2) 문장 간 핵심 대명사(coreference) 및 순서(order) 불일치  
3) 중복된 n-gram 생성 → 가독성 저하  
를 초래하며, LSTM 기반 반복 구조에서도 *장기 의존성 소실*과 *기울기 소실* 문제가 남아 있었다.[1]

***

## 2. 제안 방법

### 2.1 통합 인코더-디코더 구조  
- 영상 피처와 이전 문장 토큰 임베딩을 결합(concatenation)하여 동일한 Transformer 층 N개에서 공유 처리  
- 입력 토큰별 타입 임베딩(token type embedding)으로 영상·텍스트 구분.[1]

### 2.2 외부 메모리 모듈  
MART의 핵심은 *메모리 업데이트*를 통해 문단 생성 히스토리를 보존하는 것.  
- 단계 $$t$$에서 각 Transformer 층 $$l$$은  
  1) **메모리 어그리게이션**  

$$
       S^l_t = \mathrm{MultiHeadAttn}(Q = H^l_t,\; K = [M^l_{t-1};\,H^l_t],\; V = [M^l_{t-1};\,H^l_t])
     $$  
  
  2) **업데이트 게이트**  

$$
       Z^l_t = \sigma(W^{l}_{mz}M^l_{t-1} + W^{l}_{sz}S^l_t + b^l_z)
     $$  
  
  3) **메모리 갱신**  

$$
       C^l_t = \tanh(W^{l}_{mc}M^l_{t-1} + W^{l}_{sc}S^l_t + b^l_c),\quad
       M^l_t = Z^l_t \odot C^l_t + (1 - Z^l_t)\odot M^l_{t-1}
     $$  

파일별 메모리 길이 $$T_m$$ 슬롯 지원, Hadamard 곱 연산으로 LSTM/GRU 유사 update gate 구현.[1]

### 2.3 Transformer-XL 대비 차별점  
- Transformer-XL은 이전 히든 스테이트 전체를 이어받아 장기 의존성 학습  
- MART는 요약된 메모리만 전달 → **중복 정보 제거**, **필수 문맥 집중**.[1]

***

## 3. 모델 구조  
1. 입력: 영상 세그먼트 임베딩 $$\mathbf{V}\_t$$ (CNN 기반) + 이전 문장 토큰 임베딩 $$\mathbf{T}_{ < t}$$  
2. 통합 Transformer 블록 N층  
3. 각 층별 메모리 모듈 삽입 → 문맥 히스토리 반영  
4. 최종 디코더 Softmax로 다음 문장 예측  

모델 하이퍼파라미터: hidden size 768, layers=2, heads=12, 메모리 길이 $$T_m=1$$.[1]

***

## 4. 성능 향상 및 한계

### 4.1 자동 평가  
- **CIDEr-D**: 기존 Transformer 대비 +1.03, **Repetition (R4)**: 대폭 감소[표 참조].[1]
- ActivityNet Captions, YouCookII 두 데이터셋 모두에서 **BLEU4/METEOR/CIDEr-D**에서 동등 이상, 반복성 지표에서 최고 성능.

### 4.2 인간 평가  
- 문장 간 **일관성(coherence)**: Vanilla 대비 +16.5% 절대 향상, Transformer-XL 대비 +3.0% 향상  
- **관련성(relevance)**에서도 동등 또는 소폭 우위.[1]

### 4.3 한계  
- 문장 내부 **중복** 및 **세부 내용 오류**(fine-grained object/action 인식 부족) 여전  
- 메모리 길이 증가에 따른 계산 비용 상승 위험.[1]

***

## 5. 일반화 성능 향상 가능성  
- 메모리 모듈이 **요약된 히스토리**만 전달 → 훈련 시 과도한 중복 신호 억제  
- 다양한 비전-언어 태스크(예: 영상 내러티브, 멀티문장 QA)로 확장 가능  
- Transformer 기반 **메모리 기반 반복 구조**가 데이터 소량 환경에서도 문맥 유지에 유리  
- 추가적인 **메모리 슬롯 확장** 및 **어텐션 정규화**를 통해 더 긴 문맥 처리 및 overfitting 방지 기대

***

## 6. 향후 연구 영향 및 고려 사항  
- **연구 영향**:  
  - 비디오 내러티브 생성, 시계열 다중문장 생성 모델 설계에 *메모리 기반 반복 구조* 도입 관점 제시  
  - Transformer-XL 등 기존 반복형 모델과 비교 연구의 기준 제시

- **고려 사항**:  
  - 메모리 상태 **어텐션 정규화** 기법 도입으로 불필요 정보 추가 방지  
  - 미세 객체·행동 인식 성능 향상을 위한 멀티스케일 피처 융합  
  - 메모리 길이 및 슬롯 수 민감도 분석을 통한 **계산 효율성** 최적화  
  - **자연어 이해 평가**(coreference resolution, semantic coherence) 중심 평가 지표 개발

MART는 Transformer 구조에 메모리 기반 반복을 접목하여 문단 캡션의 **일관성 제고**와 **중복 감소**를 동시에 달성한 혁신적인 접근으로, 다양한 시퀀스 생성 분야의 발전에 중요한 이정표를 제시한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3b51d2b4-b0ff-4b39-9a88-62aaaa18f7be/2005.05402v1.pdf)
