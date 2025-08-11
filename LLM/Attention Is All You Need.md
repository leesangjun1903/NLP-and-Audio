# Attention Is All You Need

## 1. 핵심 주장 및 주요 기여  
**주장**  
전통적인 순환 신경망(RNN)이나 합성곱 신경망(CNN)을 완전히 배제하고, **오직 어텐션 메커니즘(Self-Attention)만**으로 구성된 새로운 시퀀스 변환 모델인 **Transformer**를 제안한다.  

**주요 기여**  
- RNN/CNN 없이 순수 어텐션만으로 인코더-디코더 구조를 설계하여, 훈련 병렬화(parallelization)를 극대화  
- **Scaled Dot-Product Attention** 및 **Multi-Head Attention** 기법 도입  
- WMT 2014 영어→독일어(EN-DE) 28.4 BLEU, 영어→프랑스어(EN-FR) 41.8 BLEU로 당시 최상위 성능 달성  
- 문장 구조 파싱(English constituency parsing) 과제에서도 소규모·대규모 데이터셋 모두에서 높은 일반화 성능 입증  

## 2. 문제 정의  
기존 시퀀스-투-시퀀스(sequence-to-sequence) 모델들은  
- RNN: 순차 계산으로 병렬 처리 어려움, 장기 의존성 학습 한계  
- CNN: 거리 증가에 따른 계층 깊이 필요 → 경로 길이(path length) 증가  
  
장기 의존성(long-range dependency)을 효과적으로 학습하면서, **병렬 처리 효율**과 **학습 속도**를 동시에 개선할 수 있는 모델이 필요했다.

## 3. 제안 방법  
### 3.1 Scaled Dot-Product Attention  
입력 행렬 Q (queries), K (keys), V (values)에 대해  

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\bigl(\tfrac{QK^\top}{\sqrt{d_k}}\bigr)\,V
$$  

- $$d_k$$: 키 벡터 차원  
- 스케일링 $$\frac{1}{\sqrt{d_k}}$$으로 큰 내적 값으로 인한 소프트맥스 기울기 소실 방지  

### 3.2 Multi-Head Attention  
병렬로 $$h$$개의 어텐션 헤드를 학습시킨 뒤,  

$$
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O
$$  

$$
\mathrm{head}_i = \mathrm{Attention}(QW_i^Q,\;KW_i^K,\;VW_i^V)
$$  

- $$W_i^Q,W_i^K,W_i^V\in\mathbb{R}^{d_{\text{model}}\times d_k}$$, $$W^O\in\mathbb{R}^{hd_v\times d_{\text{model}}}$$  
- 서로 다른 서브스페이스에서 정보를 추출하여 표현력 강화  

### 3.3 모델 구조  
- **인코더**: 입력 임베딩+포지셔널 인코딩 → 6개 동일 블록 쌓기  
  - 각 블록: Multi-Head Self-Attention → 위치별 FFN → 잔차 연결+LayerNorm  
- **디코더**: 출력 임베딩+포지셔널 인코딩 → 6개 동일 블록  
  - 각 블록: 마스킹된 Multi-Head Self-Attention → 인코더-디코더 어텐션 → 위치별 FFN → 잔차 연결+LayerNorm  

### 3.4 위치 인코딩(Positional Encoding)  
순서를 인식하기 위해 사인·코사인 함수 기반 고정 인코딩 추가  

$$
\begin{aligned}
PE_{(pos,2i)} &= \sin\bigl(pos/10000^{2i/d_{\text{model}}}\bigr)\\
PE_{(pos,2i+1)} &= \cos\bigl(pos/10000^{2i/d_{\text{model}}}\bigr)
\end{aligned}
$$

## 4. 성능 향상 및 한계  
### 4.1 성능  
- EN-DE: 28.4 BLEU (Transformer-big) → 이전 최고 대비 +2 BLEU[표2]  
- EN-FR: 41.8 BLEU (Transformer-big) → 단일 모델 기준 최고  
- **학습 속도**: 8 GPU에서 3.5일(빅 모델), 베이스 모델은 12시간 훈련으로 기존 모델 대비 학습 비용(FLOPs) 대폭 절감  
- **Parsing**: WSJ 소규모(40K 문장)만으로도 91.3 F1, 반(半)지도 학습 92.7 F1로 종전 기법 능가[표4]  

### 4.2 한계  
- **메모리 사용량**: Self-Attention의 $$O(n^2)$$ 복잡도로 긴 시퀀스 처리 시 메모리 폭증  
- **지역성 학습**: 전역 어텐션이 지역 패턴에 과도하게 집중하지 못할 수 있어, 대규모 비전·오디오 등 긴 시퀀스에는 추가 기법 필요  

## 5. 일반화 성능 향상 가능성  
- Transformer는 텍스트 외에도 **이미지·오디오·비디오** 등 다양한 모달리티에 적용 가능한 유연한 구조  
- Multi-Head Self-Attention을 통해 **장기·단기 의존성**을 동시에 캡처하며, 작은 데이터셋에서도 높은 성능 유지  
- 사전학습(Pre-training) 및 파인튜닝(fine-tuning)과 결합 시, **제로샷·소수샷 학습**에서도 우수한 일반화 잠재력  

## 6. 향후 연구 영향 및 고려 사항  
- **효율적 어텐션 변형**: 긴 입력 처리를 위한 **제한적 어텐션** 또는 **저해상도 어텐션** 연구 필요  
- **대규모 사전학습**: BERT·GPT 계열로 이어진 사전학습 트렌드에 기반하여, Transformer 구조 최적화  
- **모달리티 통합**: 텍스트·이미지·오디오 통합 멀티모달 모델로 확장  
- **생성 속도 개선**: 디코더의 순차적 토큰 생성 병목 완화를 위한 **non-autoregressive** 접근  
- **설명 가능성(Explainability)**: 어텐션 가중치 해석을 통한 투명한 의사결정 지원  

---  
Transformer는 **순수 어텐션**만으로도 순차 모델의 한계를 극복하고, 빠른 학습 및 높은 일반화 성능을 증명함으로써 이후 **자연어처리** 및 **멀티모달 학습** 연구의 기반으로 자리잡았다. 앞으로는 **효율성·확장성·설명 가능성**을 균형 있게 고도화하는 방향이 중요하다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/96d7c440-e2ed-40f4-8132-d87a01e2f603/1706.03762v7.pdf
