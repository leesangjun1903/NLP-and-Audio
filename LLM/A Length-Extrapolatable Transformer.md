# A Length-Extrapolatable Transformer

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
– 기존 Transformer 계열 모델은 사전학습 시 설정된 최대 길이 내에서는 잘 작동하지만, 이를 초과하는 긴 문장에서는 성능이 급격히 저하된다.  
– 본 연구는 짧은 시퀀스로 학습하고 긴 시퀀스를 평가할 때도 안정적·효율적으로 동작하는 “Length-Extrapolatable Transformer (LEX)”를 제안한다.  

**주요 기여**  
1. **Attention Resolution 정의**: 토큰 간 거리를 구분하는 능력을 정량화하는 메트릭 $$R(s)$$을 도입했다.  
2. **XPOS (Extrapolatable Position Embedding)**:  
   – ROPE의 회전 변환에 지수 감쇠(decay) 계수를 결합하여 장거리에서의 내구성을 높이고, 단거리에서는 기존 ROPE와 동등한 성능을 유지하도록 설계.  
   – 수식:

$$
       q' = (q \odot \cos(n\theta)\,\hat\zeta^n) + (\mathrm{rot}(q)\odot \sin(n\theta)\,\hat\zeta^n),
     $$  

$$
       k' = (k \odot \cos(n\theta)\,\hat\zeta^{-n}) - (\mathrm{rot}(k)\odot \sin(n\theta)\,\hat\zeta^{-n}),
     $$  
     
  여기서 $$\hat\zeta_i = \frac{i/(d/2) + \gamma}{1 + \gamma}$$ 로 정의하여 높은 주파수 성분에 더 큰 감쇠를 적용.  

3. **Blockwise Causal Attention (BCA)**:  
   – 평가 시 긴 시퀀스에 대해 키·값 캐시를 블록 단위(길이의 절반씩)로 재사용하여 계산 비용과 메모리를 줄이면서도 위치 구분 해상도를 유지.  
4. **실험적 검증**:  
   – arXiv 데이터(평균 6k 토큰)에서 길이 256–4096 범위로 평가한 결과, LEX는 모든 길이에서 기존 모델 대비 퍼플렉서티를 안정적으로 감소시킴 (예: 1024→2048 구간에서도 감소)[표 2].  
   – Attention Resolution 실험에서 LEX가 가장 높은 값을 보여 위치 식별 능력이 우수함[표 3].  

## 2. 문제 정의, 제안 기법, 모델 구조, 성능 및 한계  

### 2.1 해결하려는 문제  
Transformer는 병렬화에 유리하지만, 위치 정보 인코딩 방식(절대/상대 임베딩, 바이어스)의 한계로 인해 **학습 시 최대 길이를 초과하는 긴 입력**을 처리할 때 퍼포먼스가 크게 악화된다.

### 2.2 제안 방법  
1. **Attention Resolution**  

$$
     R(s) = \frac{\sum_{i=0}^N e^{s[i]}\bigl(e^{s[i]} - e^{s[i+1]}\bigr)}{\bigl(\sum_{i=0}^N e^{s[i]}\bigr)^2}
   $$  
   
   – $$s[n]$$: 거리 $$n$$인 토큰 쌍의 평균 스코어.  
   – $$R(s)$$가 클수록 모델이 거리 차이를 더 정확히 구분함.  

2. **XPOS (Extrapolatable Position Embedding)**  
   – 기존 ROPE에 지수 감쇠 계수 $$\hat\zeta^n$$를 곱해 긴 거리에서도 코사인 회전의 진동성을 억제.  
   – $$\hat\zeta_i = \dfrac{i/(d/2) + \gamma}{1 + \gamma}$$ (실험적으로 $$\gamma$$ 조정)  
   – XPOS를 통해 단거리(학습 범위 내)와 장거리(추론 시) 모두에서 안정적 위치 모델링 달성.  

3. **Blockwise Causal Attention**  
   – 최대 학습 길이 $$l$$를 반으로 나눈 블록들 간에 키·값을 캐시 및 재활용.  
   – 연산량은 유지하면서도 긴 문맥 전파에 필요한 위치 해상도를 확보.  

### 2.3 모델 구조  
– **기본 아키텍처**: 24-layer, hidden size 1024, 16-head GPT-3 medium 급 제로부터 학습.  
– **학습**: 최대 길이 1024 토큰에 대해 일반적인 causal self-attention 사용.  
– **추론**: XPOS 위치 임베딩 + BCA 적용하여 임의 길이 시퀀스 처리.  

### 2.4 성능 향상  
– **Interpolation (≤1024)**: 기존 ROPE 대비 1–3 정도 퍼플렉서티 감소.  
– **Extrapolation (>1024)**:  
  – ROPE/Alibi는 길이 증가 시 퍼플렉서티 폭등 또는 정체.  
  – LEX는 길이 1024→4096 구간에서도 퍼플렉서티 지속 감소(23.31→20.73).  
– **Resolution**: LEX가 훈련 길이(1024)와 확장 길이(2048) 모두에서 최고값 기록[표 3].  

### 2.5 한계  
– **Inference 오버헤드**: XPOS 계산으로 절대 위치 임베딩 대비 약 6% 추가 비용.  
– **Bidirectional 적용 미검증**: 현재는 causal(Autoregressive) 모델에만 적용.  
– **감쇠 계수 γ 튜닝**: 최적 γ는 휴리스틱하며, 차원 변경 시 재조정 필요.  

## 3. 일반화 성능 향상 가능성 집중 논의  
– **Attention Resolution 극대화**: XPOS는 위치 구분 능력 자체를 메트릭화하고 최적화했으므로, 도메인·태스크가 바뀌어도 학습 범위 외의 거리에서도 일관된 성능 유지 기대.  
– **블록 캐싱 전략**: BCA는 긴 맥락 전파에 유리하여 긴 문서 요약·질문응답 등 다양한 긴 시퀀스 작업으로 일반화 가능.  
– **지수 감쇠 활용**: 임베딩 차원별로 진동 억제 강도를 조절하므로, 정보 손실 없이 장거리 의존성 학습이 필요한 다른 Transformer 응용에 적용 여지.

## 4. 향후 영향 및 연구 시 고려 사항  
– **영향**:  
  – 긴 문서 처리 효율화: 문서 요약, 법률·의료 문서 분석, 코딩 보조 등 장문 입력 태스크에서 Transformer 채택 장벽 완화.  
  – 위치 인코딩 연구 방향 전환: 메리트 있는 감쇠+회전 결합 방식에 대한 후속 연구 활성화.  
– **고려 사항**:  
  1. **γ 자동 최적화**: 차원별·태스크별 최적 감쇠 계수를 학습 방식으로 도입 필요.  
  2. **Bidirectional 확장**: BERT 계열 등 양방향 모델에서 XPOS·BCA 효과 검증.  
  3. **효율성 개선**: XPOS 연산 절감 기법(근사, 양자화) 연구.  
  4. **다양한 도메인 실험**: 언어모델 외에도 비전·멀티모달 Transformer 적용 가능성 탐색.  
  5. **이론적 분석 강화**: 감쇠 계수에 대한 수리적 최적성 보장 연구.  

***

**요약**: XPOS와 Blockwise Causal Attention의 결합으로, Transformer가 학습 시 최대 길이를 넘어서는 긴 시퀀스를 효율적으로 처리하며 퍼포먼스 저하 없이 장거리 의존성을 활용할 수 있음을 입증했다. 앞으로 위치 인코딩 최적화 및 bidirectional 확장 연구가 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/45644d7b-faa1-43a4-94e7-58759beeab9f/2212.10554v1.pdf
