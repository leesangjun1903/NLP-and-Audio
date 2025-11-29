# IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System

### 1. 핵심 주장 및 주요 기여

IndexTTS는 XTTS와 Tortoise 모델을 기반으로 개발된 산업 수준의 LLM 기반 제로샷 텍스트-음성 합성(Zero-Shot Text-to-Speech, TTS) 시스템입니다. 본 논문의 핵심 주장은 하이브리드 아키텍처와 여러 신규 개선사항을 통해 기존 시스템보다 자연스러움, 콘텐츠 일관성, 음성 클로닝 성능을 크게 향상시킬 수 있다는 것입니다.[1]

주요 기여는 다음과 같습니다:

1. **중국어 음다의성 문제 해결**: 캐릭터-핀인(character-pinyin) 하이브리드 모델링 접근으로 음다성 문자와 저빈도 문자의 발음을 제어 가능하도록 개선

2. **Conformer 기반 조건부 인코더 도입**: 기존 조건부 인코더를 Conformer으로 대체하여 음성 특성 유사도와 학습 안정성 향상

3. **음성 코덱 개선**: Vector Quantization(VQ)과 Finite-Scalar Quantization(FSQ) 비교 분석을 통해 거의 100%의 코드북 이용률 달성

4. **BigVGAN2 기반 디코더 적용**: 고품질 음성 재구성으로 음질과 음성 특성 유사도 개선

5. **공개 테스트 세트 제공**: 음다성 단어, 주관식 및 객관식 평가 세트를 포함한 포괄적인 벤치마크 공개[1]

### 2. 해결 문제, 제안 방법 및 모델 구조

#### 2.1 해결하고자 하는 문제

현재의 LLM 기반 TTS 시스템들은 높은 자연스러움과 강력한 음성 클로닝 능력을 보이지만, 다음과 같은 문제점을 가지고 있습니다:[1]

- **중국어 음다성 제어 부족**: G2P 모듈 없이 순정본 텍스트 입력을 처리할 때 음다성 문자의 발음 오류 발생 (약 18.6%)
- **코드북 붕괴 문제**: VQ 기반 음성 토크나이저의 낮은 코드북 이용률
- **음성 특성 일관성 문제**: 기존 조건부 인코더의 음성 특성 재현 능력 제한
- **산업 적용 시 복잡성**: 다양한 오픈소스 시스템 대비 학습 프로세스 복잡성 및 느린 추론 속도

#### 2.2 제안하는 방법 및 수식

**BPE 기반 텍스트 토크나이저**

IndexTTS는 G2P(Grapheme-to-Phoneme) 모듈을 제거하고 BPE 기반 텍스트 토크나이저를 사용합니다. 어휘 크기는 12,000으로, 8,400개의 중국 한자, 1,721개의 핀인, 영어 단어 조각 및 특수 기호를 포함합니다.[1]

**캐릭터-핀인 혼합 모델링**

학습 중에 전체 샘플의 50%를 무작위로 선택하고, 각 샘플에서 중국 한자의 20%를 무작위로 선택하여 음다성이 아닌 문자를 핀인으로 대체합니다:

$$P(\text{character} \rightarrow \text{pinyin}) = \begin{cases} 1.0 & \text{if polyphonic character} \\ 0.2 & \text{if non-polyphonic character} \end{cases}$$

이를 통해 2,500개 문장 테스트에서 음다성 오류의 94.0%를 정정할 수 있습니다.[1]

**VQ vs FSQ 비교**

Vector Quantization은 코드북 붕괴로 인해 낮은 이용률을 보이는 문제가 있습니다. IndexTTS는 FSQ를 적용하여 개선합니다.

VQ의 손실 함수: $$L_{VQ} = L_{recon} + \beta \|z - sg[e]\|^2 + \gamma \|sg[z] - e\|^2$$

FSQ의 손실 함수: $$L_{FSQ} = L_{recon}$$

여기서 FSQ는 양자화 손실이 필요 없어 더 간단한 구조를 가집니다. 실험 결과, 6,000시간 학습 데이터에서 VQ는 55% 수준의 코드북 이용률을 보인 반면, FSQ는 거의 100%에 가까운 이용률을 달성합니다.[1]

**Conformer 기반 조건부 인코더**

기존의 단일 스피커 임베딩 벡터 대신 Conformer 기반 Perceiver를 도입합니다:

$$\text{Speaker Info} = \text{Perceiver}(\text{Prompt Audio}, \text{reference speakers})$$

이는 다중 참조를 제한 없이 활용할 수 있으며, 서로 다른 모델 실행 간 스피커 시프트 문제를 완화합니다.[1]

**입력 시퀀스 구조 (SEQ3)**

IndexTTS는 SEQ3 구조를 채택합니다:

$$\text{Input} = [\text{speaker info}, \text{[BT]}, \text{text}, \text{[ET]}, \text{[BA]}, \text{audio}, \text{[EA]}]$$

여기서 [BT], [ET]는 텍스트 시작/종료, [BA], [EA]는 음성 시작/종료를 나타냅니다. 이 구조는 프롬프트 텍스트가 필요 없어 크로스 언어 음성 클로닝 시나리오에서 더 실용적입니다.[1]

**BigVGAN2 기반 음성 디코더**

SpeechLLM 출력을 직접 파형으로 변환합니다:

$$\text{Waveform} = \text{BigVGAN2}(\text{interpolate}(\text{LLM latent}, 25Hz \rightarrow 100Hz), \text{speaker embedding})$$

이 방식은 확산 또는 플로우 매칭 기반 접근법보다 빠른 추론 속도를 제공합니다.[1]

#### 2.3 모델 구조

에서 제시한 IndexTTS의 전체 아키텍처는 다음과 같이 구성됩니다:[1]

1. **입력 레이어**: 원본 텍스트와 프롬프트 음성
2. **텍스트 토크나이저**: BPE 기반으로 텍스트를 토큰 시퀀스로 변환
3. **음성 토크나이저**: 24kHz 샘플링 레이트의 음성을 25Hz 토큰 레이트의 음성 토큰으로 인코딩 (약 50M 파라미터, 8192개 코드)
4. **Perceiver 조건부 인코더**: 프롬프트 음성으로부터 스피커 정보 추출
5. **Transformer 기반 Text-to-Codec LLM**: 텍스트 토큰과 스피커 정보를 입력으로 하여 음성 토큰 시퀀스 생성 (디코더 전용 아키텍처)
6. **BigVGAN2 디코더**: 음성 토큰을 최종 파형으로 변환 (24kHz 출력)

모델은 Conformer 기반 조건부 인코더와 함께 SEQ3 입력 시퀀스 구조를 사용합니다.[1]

### 3. 성능 향상 및 비교 분석

#### 3.1 객관적 평가 결과

IndexTTS는 네 개의 테스트 세트에서 기존 모델들과 비교 평가되었습니다:[1]

| 평가 지표 | IndexTTS | CosyVoice2 | F5TTS | FishSpeech | XTTS |
|---------|----------|-----------|--------|-----------|------|
| Aishell1 CER(%) | 1.3 | 1.8 | 3.9 | 2.4 | 3.0 |
| Aishell1 SS | 0.744 | 0.796 | 0.743 | 0.488 | 0.573 |
| CommonVoice ZH CER(%) | 7.0 | 9.1 | 11.7 | 11.4 | 11.4 |
| CommonVoice ZH SS | 0.742 | 0.743 | 0.747 | 0.552 | 0.586 |
| CommonVoice EN WER(%) | 5.3 | 7.3 | 5.4 | 8.8 | 7.1 |
| CommonVoice EN SS | 0.758 | 0.742 | 0.746 | 0.622 | 0.648 |
| Librispeech WER(%) | 2.1 | 4.9 | 7.8 | 8.0 | 3.5 |
| Librispeech SS | 0.823 | 0.837 | 0.828 | 0.701 | 0.761 |

CER(Character Error Rate)는 중국어 콘텐츠 정확도를, WER(Word Error Rate)는 영어 콘텐츠 정확도를, SS(Speaker Similarity)는 음성 클로닝 성능을 나타냅니다.[1]

#### 3.2 주관적 평가 결과 (MOS)

주관식 평가는 운율(Prosody), 음성 특성(Timbre), 음질(Quality) 세 가지 차원에서 수행되었습니다:[1]

| 모델 | 운율 | 음성 특성 | 음질 | 평균 |
|-----|------|---------|------|------|
| IndexTTS | 3.79 | 4.20 | 4.05 | 4.01 |
| CosyVoice2 | 3.67 | 4.05 | 3.73 | 3.81 |
| FireRedTTS | 3.79 | 3.72 | 3.60 | 3.70 |
| F5TTS | 3.56 | 3.88 | 3.56 | 3.66 |
| FishSpeech | 3.40 | 3.63 | 3.69 | 3.57 |
| XTTS | 3.23 | 2.99 | 3.10 | 3.11 |

IndexTTS는 모든 차원에서 최고 점수를 달성했으며, 특히 음성 특성과 음질 측면에서 우수한 성능을 보입니다.[1]

#### 3.3 효율성 비교

200개 테스트 샘플 기준 추론 시간 및 GPU 자원 소비:[1]

| 모델 | 총 소요 시간(초) | GPU 이용률(%) |
|-----|-----------------|--------------|
| IndexTTS | 397 | 28.47 |
| F5TTS | 320 | 42.13 |
| CosyVoice2 | 805 | 48.41 |
| FishSpeech | 756 | 71.43 |
| FireRedTTS | 732 | 92.65 |
| XTTS | 488 | 87.65 |

IndexTTS는 가장 낮은 GPU 이용률을 보이면서도 경쟁력 있는 추론 속도를 달성합니다.[1]

#### 3.4 음다성 문자 정정 능력

2,500개 문장의 음다성 문자 테스트에서:[1]

- A1 (모든 캐릭터 입력): 465개 문장(18.6%)에서 발음 오류 발생
- A2 (정확한 핀인 입력): 437개(94.0%)의 오류를 정정 가능
- 미정정 오류: 28개(1.1%) - 학습 데이터의 강화된 오류로 인한 것

### 4. 모델 일반화 성능 향상 가능성

#### 4.1 일반화 능력 분석

IndexTTS의 일반화 성능 향상은 다음과 같은 요소에서 비롯됩니다:

**대규모 다양한 학습 데이터**

34,000시간의 고품질 중영문 이중언어 데이터(중국어 25,000시간, 영어 9,000시간)로 학습되어 광범위한 음성 변이를 포괄합니다. 이는 FSQ 도입으로 인한 거의 100% 코드북 이용률과 함께 더 풍부한 음성 표현을 가능하게 합니다.[1]

**다중 참조 스피커 정보 활용**

Conformer 기반 Perceiver는 길이 제한 없이 다중 참조 샘플을 활용할 수 있어, 새로운 스피커의 음성 특성을 더 정확하게 캡처합니다. 이는 단일 임베딩 벡터 방식 대비 스피커 시프트 문제를 완화하며, 다른 스피커의 특성도 통합하여 고유한 음성 생성을 가능하게 합니다.[1]

**SEQ3 구조의 유연성**

프롬프트 텍스트가 필요 없는 SEQ3 구조는 크로스 언어 음성 클로닝과 같은 다양한 시나리오에 적응 가능합니다. 다언어 ASR 시스템 의존성을 제거함으로써 시스템 복잡성을 감소시키고 일반화 가능성을 향상시킵니다.[1]

#### 4.2 여러 테스트 세트에서의 강건한 성능

IndexTTS는 Librispeech, AISHELL-1, CommonVoice 중국어, CommonVoice 영어 등 다양한 클린과 노이즈 환경의 테스트 세트에서 지속적으로 우수한 성능을 보입니다. 특히 음성 특성 유사도(SS)에서 CosyVoice2, F5TTS와 최소한의 성능 격차를 유지하면서, 콘텐츠 일관성(WER/CER) 측면에서 모든 모델을 능가합니다.[1]

#### 4.3 코드북 이용률과 일반화의 관계

실험에서 VSQ의 낮은 코드북 이용률(6,000시간 학습 시 55%)이 일반화를 제한할 수 있음을 보였습니다. FSQ 도입으로 거의 100% 이용률을 달성함으로써:[1]

- 더 많은 다양한 음성 특성을 표현 가능
- 새로운 스피커에 대한 적응력 증가
- 학습 데이터 크기 증가에 따른 일반화 성능 향상

### 5. 모델의 한계

논문에서 명시한 주요 한계는 다음과 같습니다:[1]

1. **제한된 언어 지원**: 현재 중국어와 영어만 지원하며, 추가 언어 확장이 필요
2. **감정 표현 제한**: 풍부한 감정 표현 재현 능력 부족 (강화 학습 등을 통한 개선 계획)
3. **명령어 기반 음성 생성 미지원**: 세부적인 음성 스타일 지시 불가능
4. **운율 제어 한계**: 특정 음성학적 특징(웃음, 망설임, 놀람)의 제어 능력 제한
5. **추론 속도**: GPU 이용률은 낮으나, 절대적 추론 시간에서 F5TTS보다는 느림

### 6. 최신 연구 동향 기반 미래 영향 및 고려사항

#### 6.1 향후 연구에 미치는 영향

IndexTTS가 제시하는 기여는 TTS 분야의 다음 세 가지 방향에 영향을 미칠 것으로 예상됩니다:

**산업 수준 시스템의 실용성 강화**[2][3]

최근 연구 동향은 단순한 성능 향상을 넘어 실제 배포를 고려한 제로샷 TTS 시스템 설계에 초점을 맞추고 있습니다. IndexTTS의 낮은 GPU 이용률(28.47%)과 단순한 학습 프로세스는 엣지 디바이스나 리소스 제약 환경에서의 배포를 가능하게 합니다. SPADE(Structured Pruning and Adaptive Distillation) 같은 최근 연구가 LLM-TTS의 효율성을 1.7배 개선하려는 노력과 일치합니다.[4]

**일반화 성능 개선 메커니즘 규명**[5][6]

TimeLayer Adaptive Speaker Alignment(TLA-SA)와 Intelligibility Preference Speech Dataset(INTP)를 통한 최신 연구들은 FSQ 도입으로 달성한 거의 100% 코드북 이용률이 일반화 성능 향상의 핵심 요소임을 시사합니다. 대규모 데이터셋에서 FSQ가 VQ와 성능 격차가 거의 없다는 발견은 향후 연구에서 코드북 설계의 중요성을 강조합니다.[1]

**다언어 및 다감정 음성 합성으로의 확장**[7][8]

중국어 음다성 문제 해결을 위한 character-pinyin 혼합 모델링 방식이 다언어 환경에서의 음성 제어 메커니즘으로 확장될 수 있습니다. 최근 메타 러닝 기반 7000여 개 언어 지원 TTS 연구와 감정 기반 음성 합성 개선 연구와 결합되면, 더욱 포괄적인 제어 가능한 TTS 시스템 개발이 가능해질 것입니다.

#### 6.2 향후 연구 시 고려할 점

**1. 동적 프롬프트 선택 메커니즘**[6]

IndexTTS는 프롬프트 음성 선택의 영향을 명시적으로 다루지 않습니다. 향후 연구에서는 특정 스피커 특성을 효과적으로 전달할 수 있는 최적 프롬프트 선택 알고리즘 개발이 필요합니다. Time-Layer Adaptive Alignment 연구에서 보인 바와 같이, 시간 단계와 네트워크 층에 따른 스피커 정보의 비균등 분배를 고려한 적응형 프롬프트 선택이 효과적일 수 있습니다.

**2. 도메인 밖 성능 강화**[3]

혀꼬기, 반복 단어, 코드 스위칭(code-switching), 크로스 언어 합성 등 학습 분포 밖의 시나리오에서의 성능이 여전히 문제입니다. 최근 Direct Preference Optimization(DPO) 기반의 Intelligibility Preference Speech Dataset 연구가 보인 바와 같이, IndexTTS도 도메인 밖 데이터에 대한 명시적 선호도 최적화를 통해 성능 향상이 가능할 것으로 예상됩니다.[3]

**3. 소형화 및 경량화 연구**

IndexTTS의 조건부 LLM과 BigVGAN2 디코더는 여전히 상당한 연산 리소스를 요구합니다. 최근 경량 제로샷 TTS 연구에서 보인 바와 같이, 자가 증류(self-distillation) 및 표현 분해(representation disentanglement) 기법을 적용하여 RTF(실시간 계수)를 0.13(CPU) 수준까지 개선할 수 있을 것으로 예상됩니다.[9]

**4. 파라언어적 표현 제어**

논문에서 언급한 웃음, 망설임, 놀람 등 파라언어적 특성의 생성 제어는 현재 중요한 미충족 수요입니다. 최근 StyleTTS와 OZSpeech 같은 연구에서 보인 세밀한 프로소디 제어 기법을 IndexTTS의 구조와 결합하면, 더욱 표현력 있는 음성 합성이 가능할 것으로 예상됩니다.[10][11]

**5. 감정 및 스타일 제어 메커니즘**

현재 IndexTTS는 프롬프트 음성의 특성을 수동으로 모방하는 방식이므로, 명시적 감정 또는 스타일 레이블을 통한 제어는 불가능합니다. 강화 학습이나 선호도 기반 최적화를 통해 명령어 기반 감정/스타일 생성 능력을 추가하는 것이 향후 개선 방향입니다.[3][1]

**6. 스트리밍 및 실시간 합성**

IndexTTS는 현재 비자동회귀(non-autoregressive) 구조이지만, 스트리밍 시나리오에서의 성능 평가가 부족합니다. F5-TTS의 플로우 매칭 기반 접근법이나 최근의 동적 압축 학습 기법을 참고하여 스트리밍 친화적 구조로의 확장이 필요합니다.[12]

**7. 다중 모달 조건화**

최근 연구에서 보인 텍스트, 이미지, 음성 등 다중 모달 입력을 활용한 제어 기법을 IndexTTS에 통합함으로써, 더욱 섬세한 음성 생성 제어가 가능해질 것입니다.[10]

### 결론

IndexTTS는 LLM 기반 제로샷 TTS 분야에서 산업 적용을 위한 실질적이고 실용적인 개선을 제시합니다. 특히 코드북 이용률 최적화, Conformer 기반 조건부 인코더, 캐릭터-핀인 혼합 모델링은 향후 TTS 연구의 중요한 설계 원칙으로 자리 잡을 것으로 예상됩니다. 다만 감정 표현, 다언어 확장, 파라언어적 제어 등의 한계는 시스템의 표현력을 제한하고 있으며, 이는 추후 연구를 통해 지속적으로 개선되어야 할 영역입니다. 최신 연구 동향을 볼 때, 일반화 성능, 도메인 밖 강건성, 경량화, 그리고 다중 모달 조건화가 향후 제로샷 TTS 시스템 개발의 핵심 방향이 될 것으로 판단됩니다.

***

### References

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f7b42a07-68fe-47c2-bd87-120a7e83dc09/2502.05512v1.pdf)
[2](https://www.semanticscholar.org/paper/88264136340c9b0aaaee3bd909c6cdd36eadb55f)
[3](https://arxiv.org/abs/2505.04113)
[4](https://www.nature.com/articles/s41598-025-90507-0)
[5](https://arxiv.org/abs/2509.20802)
[6](https://www.isca-archive.org/interspeech_2025/chen25_interspeech.html)
[7](https://arxiv.org/abs/2509.02129)
[8](https://arxiv.org/abs/2510.19414)
[9](https://ieeexplore.ieee.org/document/11011159/)
[10](https://arxiv.org/abs/2507.04270)
[11](https://www.ewadirect.com/proceedings/ace/article/view/25664)
[12](https://arxiv.org/pdf/2501.08566.pdf)
[13](https://arxiv.org/pdf/2306.03509.pdf)
[14](https://arxiv.org/abs/2112.02418)
[15](https://arxiv.org/pdf/2502.05512.pdf)
[16](http://arxiv.org/pdf/2406.05370.pdf)
[17](https://arxiv.org/pdf/2403.05989.pdf)
[18](https://arxiv.org/pdf/2501.13372.pdf)
[19](http://arxiv.org/pdf/2406.06403.pdf)
[20](https://arxiv.org/abs/2501.08566)
[21](https://huggingface.co/papers/2410.03751)
[22](https://www.isca-archive.org/interspeech_2025/xie25b_interspeech.pdf)
[23](https://www.emergentmind.com/topics/zero-shot-text-to-speech-zs-tts)
[24](https://aclanthology.org/2025.acl-long.682.pdf)
[25](https://www.arxiv.org/pdf/2511.17555.pdf)
[26](https://www.emergentmind.com/topics/speech-to-speech-large-language-models-sllms)
[27](https://www.sciencedirect.com/science/article/pii/S1319157824002209)
[28](https://openreview.net/forum?id=HiDyrlYZV4)
[29](https://www.themoonlight.io/en/review/recent-advances-in-speech-language-models-a-survey)
[30](https://aclanthology.org/2025.acl-long.598/)
