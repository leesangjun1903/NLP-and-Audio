# UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data

## 1. 핵심 주장과 주요 기여  
UniSpeech는 **지도 학습(CTC phonetic loss)**과 **비지도 학습(contrastive loss)**을 **멀티태스크**로 통합하여, 라벨이 있는 고자원 데이터와 라벨이 없는 저자원 데이터를 동시에 활용함으로써 **저자원 언어 및 도메인**에서의 음성 인식 성능을 획기적으로 향상시킨다.  
주요 기여:  
- 지도·비지도 학습을 결합한 통합 사전학습 프레임워크 최초 제안  
- 양쪽 학습 목표가 공유하는 **통합 표현(unified representation)** 도입  
- 양자화된 이산 잠재표현(codebook)에 명시적으로 음소 정보를 주입하여 음운 단위 코드북 학습  

## 2. 해결 과제  
- 다수의 언어·도메인에서 **라벨이 부족**한 저자원 음성 인식 성능 개선  
- 완전 비지도 방식의 잠재표현이 **해석 가능성**과 **음운 단위 정렬**을 보장하지 못하는 한계  

## 3. 제안 방법  
### 3.1 사전학습 손실  
- Phonetic CTC loss  
  
$$ L_ctc = -\log \sum_{\pi\inΦ(x,y)} p(\pi\mid c_{1:T}) $$  

- Contrastive loss  
  
$$ L_c = -\sum_{t}\log\frac{\exp(\mathrm{sim}(c_t,q_t)/κ)}{\sum_{q'\in Q_t}\exp(\mathrm{sim}(c_t,q')/κ)} $$  

- Codebook diversity loss  
  
$$ L_d = \frac{1}{G V}\sum_{g,v} \bar p_{g,v}\log \bar p_{g,v},\quad L_{\text{self}}=L_c + 0.1L_d $$  

- **Unified loss**  
  
$$ L = \sum_{(x,y)\in L}\bigl(αL_{ctc} + (1-α)L_{\text{self}}\bigr) + \sum_{x\in M}L_{\text{self}} $$  

- **Mixed CTC**: CTC 입력의 일부를 연속 표현 $$c$$ 대신 이산 표현 $$q$$로 치환하여 quantizer에 음소 정보를 직접 주입  

### 3.2 모델 구조  
- Convolutional feature encoder: 7-layer conv → z  
- Transformer context encoder: 12(또는 24) layer → c  
- Vector quantizer: 2 codebooks × 320 entries → q  

## 4. 성능 향상 및 한계  
- **Cross-lingual**: CommonVoice 8개 언어 평균 PER 절대 14.2→7.7 (r=0.5, α=0.5)  
- **다중 언어·다중 도메인**: 다양한 fine-tune·pre-train 조합에서 CTC-Transfer 및 XLSR 대비 최대 26.9% 상대 PER 감소  
- **도메인 전이**: Librispeech→Tedlium3 WER 8.1→7.6  
- 한계:  
  - 대규모 unlabeled 자료에 대한 **확장성** 및 계산 비용  
  - 다국어 phoneme 공유 시 **발음 차이**에 따른 표현 불일치  

## 5. 일반화 성능 향상 가능성  
- Mixed CTC로 통합된 표현 학습이 **다양한 언어·도메인**에 걸쳐 robust한 feature를 제공  
- contrastive 학습과 phonetic CTC의 시너지로 codebook이 **음향 변동성** 억제  
- 저자원 도메인에서도 소량의 라벨만으로 **빠른 적응**  

## 6. 향후 연구에 미치는 영향 및 고려 사항  
- **대규모·다양 자료**(1M+ 시간) 활용한 확장 가능성  
- Transformer 기반의 **더 깊은 구조** 또는 **seq2seq 트랜스듀서** 적용 연구  
- phoneme set 통일 및 universal phonemizer 개선  
- codebook 크기·구조와 mixed CTC 확률(r) 등 하이퍼파라미터 최적화

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3f7d0be3-a834-4d69-8280-451aa00b73e1/2101.07597v2.pdf)
