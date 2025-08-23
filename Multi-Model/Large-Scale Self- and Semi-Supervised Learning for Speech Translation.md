# Large-Scale Self- and Semi-Supervised Learning for Speech Translation  

**핵심 주장 및 주요 기여**  
이 논문은 **순수한 음성 번역(Speech Translation, ST) 데이터**만을 사용하면서도 대규모의 비라벨 음성 및 텍스트 데이터를 효과적으로 활용하는 **자가-감독 학습(self-supervised pretraining)** 및 **셀프-트레이닝(self-training)** 기법을 제안한다. wav2vec 2.0 기반의 음성 인코더 사전학습과, Libri-Light 코퍼스로부터 합성 레이블을 얻어 셀프-트레이닝을 수행한 후, Common Crawl 기반의 언어 모델을 디코딩 단계에 결합함으로써, 네 개의 CoVoST 2 언어쌍(en→de, ca, ar, tr)에서 **평균 +2.6 BLEU**의 성능 향상을 달성하였다.  

---  

## 1. 문제 정의  
기존 ST 시스템은 라벨링된 ST 데이터 외에도 ASR 전사나 병렬 기계번역(MT) 데이터를 추가로 필요로 한다.  
- **데이터 희소성**: 다양한 언어에 대해 ST 라벨 데이터 확보는 비용이 크고 비현실적임.  
- **추가 감독 의존**: 다수 연구가 ASR·MT 레이블에 의존하여 성능을 높여왔음.  

이 논문은 “ST 라벨 데이터만으로” 비라벨 음성·텍스트를 최대한 활용하여 데이터 의존을 최소화하고, 동시에 최첨단 성능을 달성하는 것을 목표로 한다.  

---  

## 2. 제안 방법  
### 2.1. Self-Supervised Pretraining  
- **wav2vec 2.0 Large 모델**  
  - Encoder: 24-layer Transformer, 내부 차원 4,096, 어텐션 헤드 16  
  - Feature encoder: 7-layer CNN, receptive field≈25ms  
  - **사전학습 데이터**: Libri-Light(60K시간, LV-60k)  
  - **훈련 목표**: 마스킹된 시퀀스의 잠재 표현 예측(contrastive loss)  

### 2.2. Self-Training  
1. **Teacher 모델**: CoVoST 2 ST 데이터로 fine-tuning된 wav2vec 2.0  
2. **Pseudo-label 생성**: LV-60k 음성에 beam size 4로 ST 라벨 부착  
3. **Student 모델**: 다시 사전학습된 wav2vec 2.0을 CoVoST 2 + pseudo-labeled LV-60k로 fine-tuning  
4. **업샘플링**: CoVoST 2 실제 데이터와 pseudo데이터를 동일한 비중으로 학습  

### 2.3. Decoding with Language Model  
- **언어 모델**: CommonCrawl 기반 Transformer LM (12-layer, embedding=512)  
- **결합 방식**: Shallow fusion  
- **스코어링**: LM weight=0.1, length penalty=0.7  

***

## 3. 모델 구조 및 수식  
- **전체 구조**:  
  Encoder(wav2vec 2.0) → Decoder(7-layer Transformer, d_model=256) → Beam search + LM shallow fusion  
- **학습 목표 함수**:  

```math
    \mathcal{L} = \mathcal{L}_{\text{ST}}(y, \hat y) + \lambda_{\text{pseudo}}\,\mathcal{L}_{\text{ST}}(y_{\text{pseudo}}, \hat y_{\text{pseudo}})
``` 
  
  여기서 $$\mathcal{L}\_{\text{ST}}$$는 cross-entropy, $$\lambda_{\text{pseudo}}=1$$ (업샘플링)  

- **Shallow fusion**:  

$$
    \arg\max_{y} \bigl( \log P_{\text{ST}}(y\mid x) + \alpha\,\log P_{\text{LM}}(y) \bigr)
  $$  
  
  $$\alpha=0.1$$  

***

## 4. 성능 향상 및 한계  
| 모델 설정                                                      | Avg BLEU |
|--------------------------------------------------------------|---------:|
| Baseline (ST만)                                              |     12.9 |
| wav2vec 2.0 pretraining                                       |     22.3 |
| + LM decoding                                                |     23.4 |
| + Self-training (LV-60k)                                     |     24.6 |
| + Self-training + LM decoding                                | **25.6** |

- **주요 성능 향상**:  
  - 미세튜닝 전 사전학습만으로 +9.4 BLEU  
  - 셀프-트레이닝으로 추가 +2.3 BLEU  
  - LM 디코딩으로 추가 +1.0 BLEU  

- **한계 및 고려사항**:  
  1. **언어 다양성**: CoVoST 2는 읽기형(read speech)에 집중. 대화체·잡음 환경 일반화 불명.  
  2. **계산 비용**: LV-60k 전체에 pseudo-labeling 및 대형 모델 fine-tuning은 높은 자원 소모.  
  3. **노이즈 전파**: 셀프-트레이닝 시 teacher 오류가 student에 전이될 위험.  
  4. **타겟 도메인 적합성**: LM 도메인 스코어링 방식으로만 보완, 더 강력한 도메인 적응 기법 필요.  

***

## 5. 모델의 일반화 성능 향상 가능성  
- **사전학습 데이터 확대**: 더 많은 언어·다양한 화자 스펙트럼 확보 시, 다중언어 ST에도 유효.  
- **잡음·실제 환경 도메인 적응**: Domain adversarial training이나 데이터 증강(SpecAugment 등) 결합.  
- **Iterative self-training**: 여러 단계 반복 pseudo-label 업데이트로 teacher 개선.  
- **멀티태스크 학습**: ASR·MT 태스크와 동시 학습하여 cross-task 일반화 강화.  

***

## 6. 향후 연구에 미치는 영향 및 고려할 점  
- **영향**:  
  - 순수 ST 데이터만으로도 비지도·준지도 학습으로 최첨단 성능 달성 가능함을 입증.  
  - 음성 분야 자가-감독 기법의 ST 확장성 제시.  

- **차기 연구 고려 사항**:  
  1. **실제 음성 도메인 적용**: 대화, 잡음 환경, 비읽기형 음성으로 확장 검증  
  2. **언어 불균형 해소**: 저자원 언어 ST에 사전학습·셀프-트레이닝 적용 시 효과성 연구  
  3. **효율적 학습**: 경량화 모델·효율적 pseudo-labeling 전략 개발  
  4. **안정성 보장**: 오류 전파 최소화를 위한 신뢰도 기반 레이블 필터링 기법 도입  

---  

**결론**: 이 논문은 음성 번역에서 **자가-감독 및 셀프-트레이닝의 시너지**를 명확히 입증하고, 추가적인 라벨 데이터 없이도 강력한 성능을 달성하는 실용적 프레임워크를 제시하였다. 앞으로 다양한 언어·환경으로의 확장과 학습 효율성 개선이 중요하다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a4558d3d-e4d3-4125-973f-e0e575319ce0/2104.06678v1.pdf)
