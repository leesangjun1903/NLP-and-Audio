# Improving Language Models by Retrieving from Trillions of Tokens

**핵심 주장 및 주요 기여**  
Retrieval-Enhanced Transformer (Retro)는 대규모(수조 개 토큰) 외부 메모리를 활용해 파라미터 수를 획기적으로 늘리지 않고도 언어 모델 성능을 개선한다. 주요 기여는 다음과 같다.[1]
- 수조 토큰 규모의 데이터베이스에서 효율적 근사 kNN 검색을 통해 입력 청크마다 관련 텍스트 청크를 검색  
- chunked cross-attention 메커니즘으로 검색 텍스트를 통합, 표준 Transformer 대비 최대 10× 파라미터 효과 달성  
- fine-tuning 없이도 사전 학습된 모델을 “Retrofit”해 빠르게 retrieval 기능 부여  

***

## 1. 해결하고자 하는 문제  
대형 언어 모델은 파라미터·데이터·컴퓨팅을 병렬로 확장해야 성능이 향상되나,  
- 파라미터 수 10× 증가 시 연산·저장 비용 급증  
- 모델이 데이터 전체를 파라미터에 내재화해야 해 업데이트·확장이 어려움  

**문제**: 파라미터·연산 복잡도 증가 없이 대규모 지식·문맥을 반영하는 방법 필요

***

## 2. 제안 방법

### 2.1. 검색 기반 확장 (Semi-parametric LM)  
입력 시퀀스 $$X=(x_1,\dots,x_n)$$을 길이 $$m$$인 청크 $$\{C_u\}\_{u=1}^l$$ 로 분할. 각 청크 $$C_u$$에 대해 frozen BERT 임베딩 $$\mathrm{Bert}(C_u)$$ 으로 k nearest neighbors $$\{(N_j,F_j)\}_{j=1}^k$$ 검색[1]
- $$N_j$$: 유사 청크, $$F_j$$: 그 청크의 연속 토큰  

### 2.2. Retrieval-Enhanced Likelihood  

```math
\mathcal{L}(X \mid \theta, D)
= \sum_{u=1}^l \sum_{i=1}^m \ell_\theta\big(x_{(u-1)m+i} \mid x_{ < (u-1)m+i}, \mathrm{Ret}_D(C_{ < u})\big),
```

단, $$\mathrm{Ret}_D(C_u)=\{[N_j,F_j]\}$$이며 $$u=1$$일 땐 $$\emptyset$$

### 2.3. 모델 구조: Chunked Cross-Attention  
- 기존 Transformer와 유사한 encoder–decoder 구조  
- **Retrieval encoder**: 각 $$[N_j,F_j]$$를 bi-directional Transformer로 인코딩 → encoded neighbors $$E_u$$  
- **Decoder Retro-block**: 각 레이어에서 self-attention 후 chunked cross-attention(Cca)으로 $$E_u$$ 통합  

$$
  \mathrm{Retro}(H,E)
  = \mathrm{FFW}\big(\mathrm{Cca}(\mathrm{Attn}(H),E)\big)
  $$

- Cca는 청크 경계 위치에서만 cross-attention 수행해 계산량 선형 유지  

***

## 3. 성능 향상  
- **파라미터 효율성**: 150M–7B 모델에서 retrieval on/off 대비 일관된 개선, 10× 파라미터 증대 효과[1]
- **데이터 스케일링**: 검색 데이터베이스를 4B→1.7T 토큰으로 확장 시 bpb(bits-per-byte) 0.92→0.77 개선[1]
- **다운스트림 태스크**: Natural Questions QA fine-tune 시 45.5% EM, FiD/T5 대비 경쟁력 확보  
- **일반화**: test/train 청크 유사도 필터링(≤12.5%) 후에도 bpb 개선 유지, 테스트 누수(leakage) 외 일반화 능력 기여 확인  

***

## 4. 한계 및 고려사항  
- **검색 누수**: training 데이터와 evaluation 누수 시 retrieval이 단순 복사 이득 → 진정한 일반화 기여 분리 필요  
- **검색 지연**: 대규모 DB 검색 비용은 실시간 응용에서 병목 가능  
- **편향·사생활**: 메모리 DB에 민감·편향 자료 노출 위험, 필터링·차등 프라이버시 등 안전 대책 요구  

***

## 5. 일반화 성능 관점  
- Retrieval은 **신규** 지식(“future” Wikipedia 9/2021) 예측에도 즉각 활용  
- 파라미터 내재화 없이도 일반화 가능한 **explicit memory** 제공  
- 필터링된 평가에서도 지속적 bpb 이득, 모델 매개변수+검색 시너지로 **robust generalization**  

***

## 6. 향후 연구 방향 및 고려사항  
- 검색 DB 업데이트만으로 모델 지식·편향 개선하는 **continual learning** 가능성  
- **Differential Privacy** 검색 데이터베이스 설계로 사생활 보호  
- 검색 품질·응답 속도 trade-off 최소화 위한 **Indexing·scaling** 최적화  
- retrieval-only vs. parameter-only vs. hybrid 학습 효율 분석을 통한 **optimal semi-parametric** 설계  
- 평가 누수 통제, 일반화·편향·안전 영향 정량화: **benchmark** 및 methodology 확립  

***

 attached_file:1[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0f919b21-7095-4e80-9efe-d81781bd6e80/2112.04426v3.pdf)
