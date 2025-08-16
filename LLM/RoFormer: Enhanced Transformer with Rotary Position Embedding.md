# RoFormer: Enhanced Transformer with Rotary Position Embedding

## 1. 핵심 주장 및 주요 기여  
RoFormer는 **회전 행렬 기반의 위치 임베딩(RoPE; Rotary Position Embedding)** 을 도입하여 Transformer의 순서 정보 학습을 개선한다.  
- **상대 위치 정보**를 쿼리·키 벡터에 내재화하되, 기존의 절대 위치 임베딩을 더하거나 편향을 추가하는 방식이 아니라, 벡터에 **회전 변환**을 적용하는 **곱셈적(position-wise rotation)** 방식으로 통합.  
- 이를 통해  
  1. **시퀀스 길이 유연성** (predefined 최대 길이에 구애받지 않음)  
  2. **거리 증가에 따른 상호 의존도(decay)** (원거리 토큰끼리의 attention 가중치 자동 감쇠)  
  3. **선형(self-attention) 아키텍처와의 호환성**  
를 동시에 달성한다.

## 2. 문제 정의 및 제안 방법  

### 2.1 해결하고자 하는 문제  
Transformer의 self-attention은 위치-불변성이 강해 토큰 간 상대적 순서 정보를 명시적으로 반영하기 어렵다.  
- 기존 절대 위치 임베딩(사인·코사인 함수 또는 학습가능 벡터 추가) 방식은  
  -  시퀀스 길이 초과 시 일반화 불가  
  -  상대 거리 정보(decay) 부재  
  -  선형 어텐션과 호환성 부족  
의 단점을 갖는다.

### 2.2 RoPE 수식 및 모델 구조  
쿼리·키 벡터를 생성하는 함수 $$f^q(x_m, m), f^k(x_n, n)$$를 위치 인덱스 $$m,n$$에 비례한 회전 행렬 $$R_{\Theta,m},R_{\Theta,n}$$으로 변형:  

$$
q_m = R_{\Theta,m}\,W^q x_m,\quad k_n = R_{\Theta,n}\,W^k x_n
$$

$$
\text{where }R_{\Theta,m}=\bigoplus_{i=1}^{d/2}
\begin{pmatrix}
\cos(m\theta_i)& -\sin(m\theta_i)\\
\sin(m\theta_i)& \cos(m\theta_i)
\end{pmatrix},\;
\theta_i=10000^{-2(i-1)/d}.
$$

Self-attention 내적이  

$$
q_m^\top k_n=(W^q x_m)^\top\,R_{\Theta,n-m}\,(W^k x_n)
$$

형태로 바뀌며, 여기서 $$R_{\Theta,n-m}=R_{\Theta,m}^\top R_{\Theta,n}$$이 상대 위치(차) 정보만 반영한다.

### 2.3 장점  
- **상대 거리 감쇠(Decay):** $$\theta_i$$ 설정으로 $$n-m$$이 커질수록 내적 상한이 자연스럽게 감소.  
- **시퀀스 길이 확장성:** 미리 정의된 최대 길이 없이 임의 길이 회전 가능.  
- **선형 어텐션 호환:** 회전은 벡터 노름 보존하므로, linear-attention(예: Performer)에도 삽입하여 선형 복잡도로 순서 정보 반영.

## 3. 성능 향상 및 한계  

|Task|Baseline|RoFormer|개선폭|
|---|---|---|---|
|WMT14 En→De BLEU|27.3|27.5|+0.2|
|BERT MLM 수렴 속도|–|더 빠름|–|
|GLUE 평균 성능|84.6/83.4 (MNLI)|89.5/90.7 (MRPC/SST-2 등)|일부↑|
|Performer LM 손실|–|더 낮음|–|
|CAIL2019-SCM (max512)|68.10% (WoBERT)|68.29%|+0.19%|
|CAIL2019-SCM (max1024)|–|69.79%|–|

**한계**  
- 수렴 가속 및 장문 일반화 우수성에 대한 **정량적·정성적 원리 설명 부족**  
- 대규모 사전학습 리소스 요구 (GPU 자원)

## 4. 일반화 성능 향상 가능성  
RoPE는 **거리 의존적 decay**와 **길이 무관 회전 삽입**의 결합으로,  
- 훈련 시 보지 못한 긴 시퀀스에서도 **일관된 상대 위치 정보** 반영  
- 선형 어텐션 모델과도 잘 결합되어, 메모리·연산 비용을 낮추며 **장문 처리 일반화**  
가능성을 보였다.

## 5. 향후 연구 영향 및 고려사항  
- **비선형 attention** 외 다양한 어텐션 변형 모델로의 RoPE 확장성 탐색  
- **회전 각도 파라미터 $$\theta_i$$ 최적화** (현재 $$10000^{-2(i-1)/d}$$ 외 학습 가능한 설정)  
- **회전 기반 위치 인코딩의 해석성** 심층 분석  
- **리소스 제약 완화**를 위한 경량화·양자화 연구  

RoFormer는 위치 정보 인코딩에 대한 새로운 관점을 제시하며, 특히 **장문 일반화**·**선형 복잡도 어텐션**과의 결합에서 후속 연구의 기반이 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/960c1957-ec75-4890-9cb1-6a2b06f29a44/2104.09864v5.pdf
