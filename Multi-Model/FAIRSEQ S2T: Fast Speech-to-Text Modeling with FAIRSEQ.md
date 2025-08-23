# FAIRSEQ S2T: Fast Speech-to-Text Modeling with FAIRSEQ

## 1. 핵심 주장 및 주요 기여  
**FAIRSEQ S2T**는 기존 FAIRSEQ 프레임워크를 확장하여 음성 인식(ASR)과 음성→텍스트 번역(ST)을 하나의 통합된 엔드투엔드(S2S) 워크플로우로 처리하는 오픈소스 툴킷이다.  
- **확장성 및 재현성**: 대규모 분산 학습, 혼합정밀도 학습, 다양한 모델 구조(RNN, Transformer, Conformer) 지원  
- **통합된 멀티태스크 학습**: MT·LM 모델과의 연동을 통한 전이 학습 및 멀티태스크 학습  
- **온라인·오프라인 추론**: 실시간 번역을 위한 모노토닉 어텐션·wait-k 정책 지원  
- **데이터 전처리·평가 도구 내장**: Kaldi 호환 특징 추출, SpecAugment, WER/BLEU/chrF/AL/DAL 스코어러 통합  

## 2. 문제 정의  
음성 인식과 음성 번역 분야는 전통적으로 분리된 파이프라인을 거쳐 왔으며,  
- 데이터 전처리·모델 구현·평가 도구가 분산  
- 멀티태스크 전이 학습 시 프레임워크 간 호환성 부족  
- 실시간 번역(latency) 제어 메커니즘 부재  

을 해결하고자 한다.

## 3. 제안 방법  
### 3.1 모델 구조  
- **RNN 기반**: 양방향 인코더 + 디코더(attention)  
- **Transformer 기반**: 표준 인코더-디코더 구조  
- **Conformer 기반**: Convolution+Self-Attention 블록  
- **온라인 ST**:  
  - Monotonic Attention  
  - Wait-k 정책:  
    $$y_t = \mathrm{Decoder}\big(z_{1:k+t-1}\big),\quad z=\mathrm{Encoder}(x) $$  
  - Multi-head Monotonic Attention  

### 3.2 학습 및 전처리  
- 입력: 80-channel log mel-filter bank, CMVN, SpecAugment(시간/주파수 마스킹)  
- 토크나이저: SentencePiece, subword-nmt, byte-level BPE 등  
- 전이 학습: MT 디코더 사전학습, CTC auxiliary loss  

### 3.3 수식 개요  
- **CTC 손실**  

$$\mathcal{L}\_{\text{CTC}} = -\ln \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^T p(\pi_t|x) $$  

- **멀티태스크 총 손실**  

$$\mathcal{L} = \lambda_{\text{MT}}\mathcal{L}\_{\text{MT}} + \lambda_{\text{ASR}}\mathcal{L}\_{\text{ASR}} + \lambda_{\text{CTC}}\mathcal{L}_{\text{CTC}} $$

## 4. 성능 향상  
- **LibriSpeech ASR**: Conformer-wav2vec 모델 CW-Lg로 WER 1.7% 달성  
- **MuST-C ST**: Transformer 기반 멀티링궬 모델로 BLEU +6 이상 향상  
- **CoVoST2**: 저자원 언어 X→En 번역에서 self-supervised features 적용 시 평균 BLEU +2.0 상승  

## 5. 한계  
- **데이터 편향**: 주로 TED·오디오북 도메인 집중  
- **실시간 지연 제어**: Wait-k 외 유연한 레이턴시 정책 개발 필요  
- **저자원 언어**: 극소량 데이터 언어에서 여전히 성능 격차 존재  

## 6. 일반화 성능 향상 가능성  
- **Self-Supervised Pre-training**: wav2vec 2.0 기반 특징 활용으로 언어 간 전이 강화  
- **대규모 멀티링궬 학습**: 다언어 혼합 학습이 희소 언어 일반화 성능 견인  
- **모듈형 태스크 추가**: 언어 모델·MT와의 연계로 추가 태스크(Autoencoding, denoising) 삽입 가능  

## 7. 향후 연구 영향 및 고려점  
- **통합 툴킷 표준 제시**: 타 연구자들은 FAIRSEQ S2T를 기반으로 다양한 S2T 모델·평가 파이프라인을 재현·확장 가능  
- **실시간 번역 활용성**: latency–accuracy 트레이드오프를 제어하는 새로운 정책 설계  
- **저자원·도메인 일반화**: 대규모 비감독 음성 데이터 활용, 도메인 적응(fine-tuning) 연구 강화 필요  
- **다중 모달 통합**: 음성+비전·텍스트 멀티모달 S2S 연구로 확장  

FAIRSEQ S2T는 향후 음성 이해·번역 연구의 **확장성**, **재현성**, **통합성**을 높이는 기반을 제공하며, 특히 일반화 성능과 실시간 처리 능력 향상 연구에 중요한 출발점이 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f883694a-5dc8-4a03-9c47-43dbf4150f9d/2010.05171v2.pdf)
