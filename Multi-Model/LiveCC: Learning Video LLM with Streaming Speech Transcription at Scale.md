# LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale

아래 내용은 업로드해 주신 논문 *"LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale"* (Chen et al., arXiv:2504.16030, 2025; CVPR 2025 채택본)을 직접 정독한 뒤, 일부 비교 항목에 대해서만 외부 자료를 보완 검색한 결과입니다. 논문에 명시되지 않은 수치나 사실은 추측 없이 명시하지 않았습니다.

---

## 1. 핵심 주장과 주요 기여 요약

LiveCC는 "값비싼 사람 라벨링이나 GPT-4o 같은 폐쇄형 API로 학습 데이터를 합성"하는 기존 Video LLM 관행에서 벗어나, **YouTube의 자동음성인식(ASR) 자막을 그대로 대규모 사전학습 신호로 사용**하자는 제안입니다. 핵심 아이디어는 단어 단위 타임스탬프에 따라 ASR 단어를 해당 시점의 비디오 프레임 사이에 **조밀하게 인터리빙(densely interleave)** 한 시퀀스를 만들어, 모델이 프레임마다 **다음 몇 개 단어만 자기회귀적으로 예측**하도록 학습시키는 것입니다.

주요 기여는 네 가지로 정리됩니다.

첫째, 데이터 측면에서 5.7M YouTube 비디오 풀에서 출발해 **Live-CC-5M**(사전학습용 5M clips)과 WhisperX로 재전사한 **Live-WhisperX-526K**(SFT용)을 구축하고, 이를 위한 효율적인 데이터 큐레이션 파이프라인(Active Speaker Detection 250× 가속 포함)을 제시했습니다.

둘째, 모델링 측면에서 Qwen2-VL-7B-Base 위에 **streaming dense interleaving** 학습 방식을 도입해, 비디오 프레임과 ASR 단어를 시간 순서로 끼워 넣은 시퀀스에 대해 causal LM loss를 적용했습니다.

셋째, 평가 측면에서 49개 스포츠 카테고리, 1,702개 free-form commentary와 1,174개 MCQ로 구성된 **LiveSports-3K** 벤치마크를 새로 만들고, GPT-4o를 judge로 활용한 pairwise 비교(Chatbot Arena 방식) 프로토콜을 제안했습니다.

넷째, 결과적으로 **LiveCC-7B-Instruct**가 LiveSports-3K-CC commentary 품질에서 Qwen2.5-VL-72B-Instruct, LLaVA-Video-72B 등 72B 모델을 앞서고, 동시에 VideoMME(64.1), OVOBench(59.8) 같은 일반 video QA에서도 7B/8B 규모 SOTA를 달성했음을 보였습니다.

---

## 2. 문제 정의·제안 방법·구조·성능·한계

### 2.1 해결하고자 하는 문제

- **데이터 확장성 문제**: 기존 streaming Video LLM(예: VideoLLM-online, StreamChat, Dispider)은 LLM으로 streaming dialogue를 합성하거나 작은 dense caption 데이터에 의존하기 때문에 scale-up이 어렵다.
- **ASR 활용 방식의 한계**: HD-VILA, MERLOT-Reserve, HowTo100M, Vid2Seq 같은 선행 연구도 ASR을 이용했으나, 대체로 **전체 비디오의 global caption** 혹은 **paragraph 단위 timestamped caption** 으로 다뤄 시간적으로 미세한 정렬을 활용하지 못했다.
- **Streaming 평가의 부재**: 기존 video QA 벤치마크(VideoMME, MVBench 등)는 전체 프레임을 한 번에 보는 "offline" 설정이라 진정한 real-time commentary 능력을 측정하지 못한다.

### 2.2 제안 방법: Dense Interleaving Streaming Sequence

훈련 시퀀스는 다음과 같이 구성됩니다 (논문 Eq. 정의를 LaTeX로 옮김):

$$[\text{Con}]\,\langle F_{t:t+k}\rangle\,\langle W_{t:t+k}\rangle\,\langle F_{t+k:t+2k}\rangle\,\langle W_{t+k:t+2k}\rangle\,\dots\,\langle F_{t+nk:t+(n+1)k}\rangle\,\langle W_{t+nk:t+(n+1)k}\rangle$$

여기서 $[\text{Con}]$는 컨텍스트(사용자 prompt, 비디오 title, 직전 ASR), $\langle F\rangle$는 프레임 visual tokens, $\langle W\rangle$는 그 구간의 ASR 단어들이며, 기본 설정은 frame rate $\text{FPS}=2$, 시간 간격 $k=1\,\text{s}$ 입니다.

학습 목표는 visual token은 비예측 입력으로 두고, 텍스트 토큰만 자기회귀적으로 예측하는 표준 causal LM 손실입니다:

$$\mathcal{L}_{\text{LiveCC}} = -\sum_{i \in \mathcal{I}_{\text{text}}} \log P_\theta\!\left(x_i \mid x_{<i}\right),$$

이때 $\mathcal{I}_{\text{text}}$는 시퀀스 내 텍스트 토큰 인덱스 집합이며, 컨텍스트(title, prev. ASR, prompt)는 마스킹되어 loss에서 제외됩니다.

세부 처리 두 가지가 흥미롭습니다.

(a) **Word-level timestamp 근사**: 사전학습용 YouTube CC는 2~3초 chunk 단위로만 정렬되므로, 해당 chunk 길이를 단어 수로 균등 분배하여 단어별 타임스탬프를 근사합니다. SFT 단계에서는 WhisperX(large-v3-turbo)로 정확한 word-level 정렬을 다시 얻습니다.

(b) **EOS 처리**: 진짜 발화 종료와 일시적 무음을 구분하기 위해 frame당 ASR 단어 뒤에 ellipsis token "`...`"을 special EOS marker로 붙이고, 무음 프레임에서는 이 토큰만 예측하도록 합니다.

### 2.3 모델 구조

- **백본**: Qwen2-VL-7B-Base (Vision Transformer + Qwen2 LLM)
- **입력**: 2 FPS 프레임을 ViT로 인코딩 → frame chunk마다 visual tokens 생성 → 같은 시간 구간의 ASR 단어들과 인터리빙
- **컨텍스트 전략 (ablation)**: "Title || Prev. ASR" 하이브리드 — 직전 ASR이 있으면 그것만 쓰고, 없을 때(영상 시작부)만 title을 사용. 두 가지를 동시에 주면 VideoMME 성능이 약간 떨어지는데, 저자들은 이를 **정보 누출(information leakage)**로 해석합니다.
- **추론**: 이전 prompt·visual·생성 텍스트의 KV cache 재사용, 240초마다 visual token은 폐기하고 텍스트 토큰만 유지하여 prefill을 다시 수행. 보고된 응답 latency는 **0.17 s/frame** (LLaVA-Video-72B 20.51 s, LLaVA-Video-7B 5.62 s 대비).

### 2.4 성능 향상 (논문 Table 1·3·4·5 기반)

- **Pre-training paradigm ablation** (Table 1a, 60s clip 기준): caption-style 대비 streaming-style이 LiveSports-3K-CC win rate에서 14.0 → 32.9로 크게 상승, VideoMME는 비슷한 수준 유지.
- **Context ablation** (Table 1b): Prev. ASR을 컨텍스트로 추가하면 win rate 14.7 → 32.0, "Title || Prev. ASR" 하이브리드가 commentary와 QA 모두에서 균형이 가장 좋음.
- **Scale ablation** (Table 1c): 1M → 2.5M → 5M → 10M로 키울수록 commentary win rate는 29.1 → 30.8 → 32.9 → 36.0로 단조 증가하지만, **VideoMME는 5M에서 정점(61.0) 후 10M에서 58.0으로 하락** — 단일 소스(streaming ASR)만으로 과도하게 키우면 일반 QA에서는 역효과.
- **최종 성능 (Table 3, 7B/8B 모델 중)**: VideoMME 64.1 (w/o sub) / 70.3 (w/ sub), MVBench 62.8, OVOBench 59.8로 동급 7B/8B 모델 중 최상.
- **LiveSports-3K-CC commentary win rate (Table 4, vs. GPT-4o-08-06)**: LiveCC-7B-Base 43.2, LiveCC-7B-Instruct 41.5로, 72B급(Qwen2.5-VL-72B-Instruct 30.4, LLaVA-Video-72B 35.0)을 상회.

### 2.5 한계 (논문 본문 + 결과로부터 합리적으로 도출되는 것)

저자들이 명시적으로 또는 결과 표에서 인정한 한계:

1. **단일 소스 데이터 과적합**: 사전학습 데이터를 10M으로 늘리면 일반 QA가 떨어지는 현상 (Table 1c) — multi-source는 SFT 단계로 미뤘다고 설명.
2. **언어·영역 편향**: 영어 + YouTube CC 존재 + 480p 이상 + 30s~10min 등 강한 필터링으로 인해 비영어권·짧거나 매우 긴 비디오·저화질 환경에는 적용이 검증되지 않음.
3. **명령 수행 능력 저하**: 논문 §5.1에서 "instruction following capability of streaming ASR pre-trained model has been lost"라고 직접 언급, MCQ는 logit-based로 평가하는 식으로 우회.
4. **Commentary는 근본적으로 ASR 모방**: ground truth가 사람의 ASR이라 win rate 평가가 "사람 해설자 스타일"에 보상되는데, 사실 정확성 검증이나 환각 측정은 이 프로토콜로는 충분치 않음(저자들도 향후 과제로 시사).
5. **MVBench 62.8**은 Qwen2-VL-7B-Instruct(67.0), InternVL2-8B(66.4), LongVU-7B(66.9)보다 낮음 — 모든 video QA 벤치마크에서 일관된 SOTA는 아님.

---

## 3. 일반화 성능 향상 가능성에 대한 중점 분석

이 논문이 일반화 측면에서 흥미로운 이유는, **"narrow하고 비싼 합성 SFT 데이터"가 아닌 "넓고 자연스러운 ASR signal"을 사전학습에 쓰면 다운스트림 task 일반화가 개선되는가?"** 라는 질문에 부분적으로 긍정적 답을 주기 때문입니다.

### 3.1 일반화가 향상되는 경로

(a) **시간적 미세정렬(temporal fine-grained alignment)이 부산물로서 generic video understanding에 도움이 됨**: Vid2Seq류(paragraph-level prediction)와 달리, frame당 평균 1~3 단어를 예측하는 강제는 모델이 시각적 변화에 즉각 반응하도록 만든다. 그 결과 Table 3에서 OVOBench(streaming benchmark)에서 50.4 → 59.8 (+9.4%) 라는 큰 향상이 나타나는데, 이는 단순 caption 학습으로는 얻기 어려운 효과입니다.

(b) **데이터 스케일에 따른 commentary 품질의 단조 증가** (Table 1c)는 Kaplan et al. (2020)의 LLM scaling law 관점에서 보면, ASR 신호가 충분히 풍부한 학습 신호임을 시사합니다. 다만 동일 도메인 신호만 키울 때 일반 QA 성능이 5M 이후 하락한다는 점은 **multi-source mixture**가 일반화에 필수임을 보여줍니다.

(c) **SFT 단계의 데이터 혼합**: LiveCC-7B-Base에 LLaVA-Video-178K + Live-WhisperX-526K를 함께 SFT한 결과(Table 2b), 동일 SFT를 받은 Qwen2-VL-7B-Base 대비 LiveSports-3K-CC win rate 33.7 → 41.5, VideoMME는 67.1 → 66.8(거의 동등)로, **streaming 사전학습이 일반 QA를 거의 해치지 않으면서 commentary 능력을 추가로 부여**함을 보여줍니다.

### 3.2 일반화의 한계 신호

- 사전학습만 한 LiveCC-7B-Base는 VideoMME에서 64.0 → 57.9로 하락(Table 2b 위쪽). 즉 streaming pre-training은 그 자체로는 instruction-following을 약화시키며, **반드시 다양한 SFT와 결합되어야** 일반 QA가 회복됨.
- 평가는 대부분 영어/스포츠/하우투 중심. 비영어, 도메인 시프트(예: 일인칭 ego 비디오, 의료 영상) 일반화는 검증되지 않음.

요약하면, "값싼 ASR 신호를 시간 정렬하여 학습하면 streaming 능력이 SOTA급으로 생기고, multi-source SFT만 잘 섞으면 일반 video QA도 같이 끌어올린다"는 것이 이 논문이 제시하는 일반화 가설이며, **결과적으로 7B 모델이 72B를 일부 task에서 앞서는 것이 그 가설의 가장 강한 근거**입니다.

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 영향

1. **데이터 패러다임 전환의 가속**: GPT-4o 기반 합성 → 무료/대규모 ASR 기반 학습으로 흐름이 옮겨갈 수 있음. 특히 학계의 자원 제약 환경에서 매력적.
2. **Streaming Video LLM 연구의 표준 베이스라인**: VideoLLM-online(Chen et al., 2024)에 이은 같은 저자 그룹의 후속작으로, streaming 평가 방법론(LiveSports-3K + LLM-as-judge win rate)이 표준화될 가능성이 큼.
3. **Multimodal omni 모델로의 확장**: 결론에서 저자들이 "future work: train multimodal omni models in streaming"을 명시. Qwen2.5-Omni 같은 omni 방향과 ASR-streaming 사전학습이 결합될 경우 음성·영상·텍스트가 모두 시간 정렬된 학습이 가능.

### 4.2 향후 연구 시 고려할 점

1. **사실 정확성(factuality) 평가**: 현재 win rate 평가는 스타일·의미 일치를 본다. Commentary의 환각(예: 잘못된 선수 이름) 정량 평가가 필요.
2. **언어 다양성**: WhisperX는 다국어 지원이지만 Live-CC-5M은 영어 전용. 한국어/중국어 ASR로의 확장 시 ASR 품질–시각 정렬–문화적 맥락의 trade-off 연구 필요.
3. **Instruction-following 보존**: 사전학습 단계에서 instruction 능력이 손상되는 문제는 streaming 사전학습과 일반 chat 데이터의 **혼합 비율 스케줄링**이나 **LoRA류 부분 학습**으로 완화할 여지가 있음.
4. **240s를 넘는 long-form streaming**: 현재 visual token을 240초마다 폐기하는 단순 전략은 정보 손실이 크다. Memory-augmented 방식(VideoLLaMB, LongVU 등)과의 결합이 자연스러운 다음 단계.
5. **Privacy/저작권**: YouTube CC와 영상의 대규모 학습 사용은 라이선스/ToS 측면에서 회색지대. 향후 정책 변화에 따라 재현성에 영향이 있을 수 있음.
6. **평가의 GPT-4o 의존성**: judge로 GPT-4o를 사용하므로 OpenAI 정책 및 모델 버전 변경에 결과가 민감. 오픈소스 judge로의 검증이 필요.

---

## 5. 2020년 이후 관련 연구 비교 분석

LiveCC를 좌표 평면에 두기 위한 비교 — 모두 직접 본문 또는 검색으로 확인한 사실에 한정합니다.

### 5.1 ASR/Narration을 활용한 video-language 사전학습 (LiveCC의 직접 조상)

- **HowTo100M (Miech et al., ICCV 2019; 2020년 직전이지만 후속 영향이 큼)**: 100M narrated YouTube clips. ASR을 video-text contrastive 학습의 텍스트로 사용. 시간 정렬은 clip 단위.
- **MERLOT / MERLOT-Reserve (Zellers et al., CVPR 2021/2022)**: 비디오·언어·음향을 함께 학습. ASR을 frame-level이 아닌 segment 단위로 사용.
- **HD-VILA (Xue et al., CVPR 2022)**: 고해상도 YouTube 영상 + 자막. 주로 video-language alignment 사전학습.
- **Vid2Seq (Yang et al., CVPR 2023)**: 가장 가까운 선행 연구. ASR 문장을 pseudo dense event로 변환해 timestamp tokens과 함께 한 시퀀스로 예측. 그러나 여전히 **paragraph-level / event-level** 예측이며, frame당 단어 예측은 아님.

LiveCC의 차별점: **frame당 1~수 단어**라는 가장 미세한 입자 크기에서 causal 학습. 이 차이가 "real-time commentary"라는 새 능력을 가능하게 한다는 것이 저자들의 주장.

### 5.2 Streaming Video LLM (LiveCC의 직접 비교군)

- **VideoLLM-online (Chen et al., CVPR 2024)** — 같은 저자 그룹의 전작. LIVE 프레임워크와 streaming EOS prediction 도입. Llama-2/3 기반, A100 기준 10–15 FPS. ASR을 직접 학습 신호로 쓰지는 않고, offline annotation을 streaming dialogue로 변환.
- **StreamChat (Liu et al., 2024, arXiv:2412.08646)**: streaming chat 인터페이스에 초점.
- **Dispider (Qian et al., 2025, arXiv:2501.03218)**: perception, decision, reaction을 분리하여 능동적 상호작용 지원.
- **Flash-VStream (Zhang et al., 2024, arXiv:2406.08085)**: 메모리 기반 long stream 처리.
- **VideoLLaMB (Wang et al., 2024)**: recurrent memory bridge.

LiveCC는 이들과 달리 **데이터 스케일(5M clips)**과 **ASR을 일급 학습 신호로 사용**한다는 점에서 차별화됩니다. VideoLLM-online이 "streaming framework"를 제시했다면, LiveCC는 "streaming pretraining data at YouTube scale"을 제시한 셈입니다.

### 5.3 일반 Video LLM (성능 비교 대상)

논문 Table 3 기준 주요 7B/8B 비교:

| 모델 | VideoMME (w/o sub) | MVBench | OVOBench |
|---|---|---|---|
| LongVA-7B (2024) | 52.6 | – | – |
| InternVL2-8B (2024) | 54.0 | 66.4 | 50.2 |
| LLaVA-OV-7B (2024) | 58.2 | 56.7 | 52.7 |
| Qwen2-VL-7B-Instruct (2024) | 63.3 | 67.0 | 50.4 |
| LLaVA-Video-7B (2024) | 63.3 | 58.6 | 52.9 |
| **LiveCC-7B-Instruct** | **64.1** | 62.8 | **59.8** |

OVOBench(streaming)에서의 큰 격차가 ASR-based streaming pretraining의 핵심 가치를 잘 보여주는 신호입니다. MVBench에서는 Qwen2-VL-7B-Instruct(67.0)에 미치지 못하므로, LiveCC가 모든 차원에서 우월하다는 주장은 과하며 "streaming + general QA의 균형"이 본 논문의 정확한 contribution입니다.

---

## 참고한 자료 (출처)

본문 분석에 사용한 자료는 다음과 같으며, 각 항목은 직접 정독했거나 검색으로 본문을 확인한 것만 기재했습니다.

1. Chen, Zeng, Lin, Li, Ma, Shou. **"LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale"**. arXiv:2504.16030v1, 2025년 4월 22일. (업로드해 주신 PDF 본문 — Tables 1~5, Figures 2·5·6·7, §3.1–3.2, §4–5, Supplementary §7–9 모두 직접 인용.)
2. CVF Open Access — "LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale" (CVPR 2025 채택본): <https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_LiveCC_Learning_Video_LLM_with_Streaming_Speech_Transcription_at_Scale_CVPR_2025_paper.pdf>
3. Chen et al. **"VideoLLM-online: Online Video Large Language Model for Streaming Video"**, CVPR 2024, arXiv:2406.11816. <https://arxiv.org/abs/2406.11816> 및 프로젝트 페이지 <https://showlab.github.io/videollm-online/>
4. Yang, Nagrani, Seo, Miech, Pont-Tuset, Laptev, Sivic, Schmid. **"Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning"**, CVPR 2023, arXiv:2302.14115. <https://arxiv.org/abs/2302.14115>
5. The Moonlight Review of VideoLLM-online (선행 streaming 연구의 기술적 맥락 확인용): <https://www.themoonlight.io/en/review/videollm-online-online-video-large-language-model-for-streaming-video>
6. Google Research blog post on Vid2Seq (2023): <https://blog.research.google/2023/03/vid2seq-pretrained-visual-language.html>

---

### 확신 수준에 대한 솔직한 메모

- **본 논문 내부 사실(수식, 표 수치, 데이터셋 통계, ablation 결과, 한계 인정 부분)**: 업로드된 PDF에서 직접 확인했으므로 높은 확신.
- **LiveCC와 다른 모델의 비교 수치(Table 3·4 인용)**: 논문이 보고한 값을 그대로 옮긴 것이며, 외부 재현은 별도로 검증하지 않았습니다.
- **2020년 이후 관련 연구의 일반적 위치 부여**: 본문 인용 + 검색으로 확인한 abstract 수준이며, 각 모델의 모든 세부 내용을 재확인하지는 않았습니다.
- **"앞으로의 영향" 섹션**: 일부는 논문 결론·향후 과제에서 명시된 것이고, 일부는 결과로부터 도출한 합리적 추론입니다 — 이 부분은 제 해석임을 명시합니다.
