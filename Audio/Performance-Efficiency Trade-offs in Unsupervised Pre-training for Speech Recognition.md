# Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition
# 1. 핵심 주장 및 주요 기여  
이 논문은 WAV2VEC 2.0 기반의 언어 비지도 사전학습 모델이 지닌 **성능–효율성(trade-off) 문제**를 분석하고, 이를 개선한 새로운 아키텍처인 **SEW(Squeezed and Efficient Wav2vec)** 및 **SEW-D(Disentangled attention 적용 버전)**를 제안한다. 주요 기여는 다음과 같다.  
- **압축된 맥락 네트워크**(squeezed context network) 구조를 도입해 추론 속도를 최대 두 배 가까이 높이면서 단어 오류율(WER)을 감소.  
- **컴팩트 웨이브 피처 추출기(WFE-C)** 설계를 통해 여러 계층에 연산량을 고르게 분산, 전체 추론 시간을 30–40% 절감.  
- **MLP 예측 헤드**와 **디스엔탱글드 어텐션**을 결합해 사전학습과 추론 효율을 모두 개선.  
- 다양한 학습 규모(10 분 – 960 시간), 도메인(LibriSpeech, TED-LIUM, VoxPopuli, Switchboard) 전반에서 일관된 성능 향상 및 속도 개선을 실증.  

# 2. 문제 정의  
기존 Wav2vec 2.0 모델은 뛰어난 ASR 성능에도 불구하고:  
- 높은 **추론 latency**로 실시간·엣지 환경 적용이 어려우며,  
- 첫 합성곱 계층에 편중된 연산으로 비효율적인 하드웨어 사용을 초래한다.  

이를 해결하기 위해 모델의 구성 요소별(특히 피처 추출기 vs. 맥락 인코더) 연산 배분과 토큰 시퀀스 해상도를 재조정하여 **성능 저하 없이 효율을 극대화**하는 방법이 필요하다.  

# 3. 제안 방법  
## 3.1 수식적 배경  
- 원본 W2V2의 **대조적 예측 손실**  

```math
    L_m = \mathbb{E}_{t\in\text{masked}} \bigl[-\log\frac{\exp(\mathrm{sim}(p_c(c_t),p_q(q_t))/\kappa)}{\sum_{q_{t'}\in Q}\exp(\mathrm{sim}(p_c(c_t),p_q(q_{t'}))/\kappa)}\bigr]  
```

- **다양성 손실**  

```math
    L_d = \mathbb{E}_t \bigl[1 - \frac{1}{GV}\sum_{g=1}^G \exp(-\sum_{v=1}^V p_{g,v}\log p_{g,v})\bigr]  
``` 

- 전체 사전학습 목표: $$L = L_m + \alpha L_d$$.  

## 3.2 모델 구조  
1. **Squeezed Context Network**  
   - 입력 시퀀스를 2×로 하향 샘플링한 뒤, 업샘플링 레이어로 원래 해상도로 복원.  
   - Transformer 연산량을 절반으로 줄이면서도 컨텍스트 정보 손실 최소화.  

2. **Compact Wave Feature Extractor (WFE-C)**  
   - 7개 합성곱 계층마다 채널 수를 점차 확대(c, 2c, 2c, 4c…)하며 FLOPs를 고르게 분산.  
   - 기존 WFE 대비 약 60% 연산량 감소, 추론 시간 37% 단축.  

3. **MLP Predictor Heads with BatchNorm**  
   - 2층 MLP + BatchNorm을 도입해 사전학습 성능↑, 추론 시 폐기되어 오버헤드 無.  

4. **Disentangled Attention (SEW-D)**  
   - 위치 임베딩과 콘텐츠 임베딩을 분리해 content-to-content, content-to-position, position-to-content 어텐션을 계산.  
   - 파라미터 수 절반으로 유지하면서도 WER 추가 개선.  

# 4. 성능 향상  
- **추론 속도**: W2V2-base 대비 SEW-D-mid는 1.9×, SEW-D-base+는 2.7× 빠름.  
- **WER 감소**: LibriSpeech test-clean 기준 W2V2-tiny(22.8%→) SEW-tiny(10.6%), SEW-D-base+(6.1%→5.3%).  
- **사전학습 비용**: 동일 업데이트 수 대비 GPU 시간 20–30% 절감.  
- **도메인 일반화**: TED-LIUM, VoxPopuli, Fisher+Switchboard 전반에서 10–30% WER 개선 및 추론 속도 30% 단축.  

# 5. 한계  
- **언어 모델(LM) 병합 시 추론 병목**: CPU 기반 빔 서치로 LM 디코딩 속도 크게 저하.  
- **하드웨어 종속성**: 실험은 GPU 최적화 환경에서 이뤄졌으며, 모바일·엣지·CPU에서는 별도 튜닝 필요.  
- **극소량 데이터(10 분) 시나리오**: 극단적 저자원 상황에서는 하이퍼파라미터 추가 튜닝이 필요.  

# 6. 일반화 성능 향상 가능성  
- **하향 샘플링 후 재증폭 전략**은 다양한 음성 길이·잡음 환경에서도 **컨텍스트 손실 최소화**하며 효율을 유지해, 적은 감독표 데이터에서도 강건한 특성 학습이 가능하다.  
- **컴팩트 피처 추출기**는 저주파·고주파 정보 비대칭 분포를 반영하여 **다양한 음성 도메인**(방송, 전화)에서 일반화 여지를 높인다.  
- **디스엔탱글드 어텐션**은 상대 위치 정보를 분리 처리해 발음·속도 변동에 민감한 도메인에도 **어댑티브하게 대응**할 가능성을 제공한다.  

# 7. 향후 연구에 미치는 영향 및 고려 사항  
- **모델 배치 경제성**: 엣지·실시간 ASR 시스템에 SEW 계열 모델을 직접 적용하여 추론 비용·에너지 효율 대폭 개선.  
- **경량화 기법 결합**: 지식 증류, 양자화, 프루닝 등과 통합해 더욱 작은 모델로도 성능 극대화 연구.  
- **비영어·다중언어 확장**: LibriSpeech 외 타 언어·코퍼스에 제안 구조를 적용해 **언어 간 효율성 격차** 분석.  
- **LM 통합 최적화**: GPU 위주 병렬 빔 서치, 또는 CTC+LM 대체법(무감독 라벨링)으로 **추론 병목 해소**.  
- **공정성과 안정화**: 극저자원 시나리오 및 대규모 모델(Fine-tuning 불안정성)에서 안정적 학습 전략 보강.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/83836d86-dc17-48de-864b-069187a8299d/2109.06870v1.pdf)
