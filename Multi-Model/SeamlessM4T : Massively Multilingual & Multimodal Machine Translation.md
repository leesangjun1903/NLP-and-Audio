# SeamlessM4T : Massively Multilingual & Multimodal Machine Translation

## 1. 핵심 주장과 주요 기여 요약  
SeamlessM4T는 단일 모델로 100개 언어의 음성-음성(S2ST), 음성-텍스트(S2TT), 텍스트-텍스트(T2TT), 텍스트-음성(T2ST), 자동음성인식(ASR)을 모두 지원하는 **세계 최초의 대규모 다국어·다중모달 기계번역 시스템**이다.  
주요 기여:  
- 470,000시간 규모의 자동 정렬된 다중모달 말뭉치 ‘SeamlessAlign’ 구축  
- 143개 언어 음성에 대한 자기지도학습 전처리(w2v-BERT 2.0)  
- 200개 언어 텍스트 번역 모델(NLLB-1.3B) + 음성·텍스트 공동 임베딩(SONAR)  
- 음성→단위 변환(T2U) + HiFi-GAN 음성합성 결합한 두 단계 UnitY 아키텍처  
- S2TT에서 이전 최첨단 대비 평균 +20 BLEU, S2ST에서 +2.6 ASR-BLEU  

## 2. 문제 정의  
기존 음성 번역 시스템은  
1) 고자원 언어 편중 2) 영→비영 번역 미지원 3) 단계별 연쇄(cascaded) 구성의 오류 누적 문제  
등 한계가 있었다. 본 연구는 모든 태스크를 한 모델로 통합하여 이 세 가지 문제를 동시에 해결하고자 한다.

## 3. 제안 기법  
### 3.1 말뭉치 구축: SeamlessAlign  
– 음성 언어식별(LID) 모델(ECAPA-TDNN 기반)로 100개 언어 raw audio 선별  
– VAD+과분절(over-segmentation)으로 문장 단위 잠재 조각 생성  
– SONAR 공동 임베딩 공간에서 텍스트·음성 간 유사도 기반 마이닝  
- 마진 점수:  

```math
\mathrm{score}(x,y)=\frac{\cos(x,y)}{\tfrac1{2k}\sum_{z\in\mathrm{NN}_k(x)}\cos(x,z)+\tfrac1{2k}\sum_{v\in\mathrm{NN}_k(y)}\cos(y,v)},
```

임계값 1.15 적용  

### 3.2 모듈 및 사전학습  
1) w2v-BERT 2.0: 24-layer Conformer, 1M시간 음성 대비 대조 + 마스킹 예측 (GVQ+RPQ; 식 (2): $$L=w_cL_c + w_{mGVQ}L_{mGVQ}+w_{mRPQ}L_{mRPQ}$$)  
2) SeamlessM4T-NLLB-1.3B: 95개 언어 T2TT dense Transformer, distilled 포함  
3) SONAR: NLLB 기반 텍스트→고정벡터+디코더, 교사-학생으로 음성 확장  

### 3.3 통합 모델 구조(UnitY 기반)  
– 멀티인코더·디코더:  
  -  음성 Conformer encoder + 길이 어댑터  
  -  텍스트 Transformer encoder  
  -  텍스트 Transformer decoder (첫 번째 pass)  
  -  Transformer 기반 T2U encoder-decoder (두 번째 pass)  
  -  HiFi-GAN 언어별 유닛 vocoder  
– 훈련 단계:  
  1) X–eng ASR, S2TT 중심  
  2) eng–X S2TT+ASR 추가 멀티태스크  
  3) S2ST 위해 T2U 및 유닛 vocoder 결합, X2T 고정  

### 3.4 성능 향상  
- S2TT (Fleurs 81개 언어): +20 BLEU (19.7→24.0)  
- S2ST (Fleurs): +2.6 ASR-BLEU; CVSS 21개 언어 +50% 향상  
- ASR (Fleurs 77개 언어): WER –45%  
- T2TT (Flores 95개 언어): X–eng 동일, eng–X +1 chrF++  
- Robustness: 배경잡음 +38% BLEU, 화자변이 +49% CoefVarMS  
- Added toxicity: 기존 대비 최대63% 감소  

## 4. 한계  
- 마이닝·정렬 잡음 → 추가 필터링 필요  
- ASR 의존으로 음성→텍스트 단계 오류 전파 가능  
- 긴 문장·도메인 특화 입력에 대한 검증 부족  
- 낮은 레이턴시 실시간 스트리밍 미지원  

## 5. 일반화 성능 향상 관점  
- 자기지도 전처리 규모(1M→400K→…) 및 코드북(GVQ×2+RPQ) 확장이 BLEU 지속 향상  
- 멀티태스크(X2T) 손실: T2TT+S2TT+distillation 결합이 최적 → zero-shot X–eng 횡단 언어 평가에서 S2TT BLEU +2.7  
- 음성마이닝 필터링(상위 400h)으로 최적 규모 확보  

## 6. 향후 영향 및 고려 사항  
- **온디바이스·실시간 번역**: 단일 모델로 대용량 태스크들 통합하였으므로, 경량화→스트리밍화 연구에 활용 가능  
- **접근성 확대**: 시각장애·문해미흡층 대상 멀티모달 언어 접근성 개선  
- **저자원 언어 보존**: 다중모달 마이닝 기반으로 극저자원 언어 말뭉치 확장  
- **책임 있는 AI**: toxicity·gender bias 평가는 필수. 실서비스 전 사용자 피드백·인간 검수 연계 필요  
- **미래 연구 고려**:  
  - 실시간(low-latency)·스트리밍 S2ST 구조 탐색  
  - 음성 표현력(prosody, 감정) 보존 모델링  
  - 대규모 비영어 ASR 공정성·정확성 개선  
  - 장문·도메인특화 번역 안정성 검증

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7e4fe318-4e77-4934-a8d6-682db93efcae/seamless_m4t_paper.pdf)
