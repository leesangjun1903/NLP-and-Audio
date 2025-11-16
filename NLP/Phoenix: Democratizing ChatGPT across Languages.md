# Phoenix: Democratizing ChatGPT across Languages

### 핵심 주장 및 주요 기여

Phoenix 논문은 **AI의 민주화와 언어 평등**이라는 중요한 철학적 문제를 제기합니다. 논문의 핵심 주장은 기존 오픈소스 ChatGPT 대안들이 라틴 문자 기반 언어, 특히 영어에 편중되어 있으며 비라틴 문자 언어(중국어, 일본어, 한국어, 아랍어 등)를 체계적으로 배제하고 있다는 것입니다. 논문은 이를 비판하면서 "전 세계 대다수 사람들이 채팅GPT를 사용할 수 없는 국가에서 사는 것처럼, 특정 그룹에게만 기술을 제한하는 것은 개방 정신에 어긋난다"고 주장합니다.[1]

Phoenix의 주요 기여는 다음과 같습니다. 첫째, **다중 언어 지시 학습(Multilingual Instruction-Following Adaptation)** 중심으로, 특히 비라틴 언어를 포함한 최초의 완전 개방소스 다언어 ChatGPT 민주화 모델입니다. 둘째, **명령어 기반 튜닝과 대화 기반 튜닝 데이터의 결합**이라는 새로운 접근법으로, 이전 연구들이 어느 한 가지 데이터 유형만 사용한 것과 다릅니다. 명령어 데이터는 모델이 인간의 지시를 따르도록 훈련하고, 대화 데이터는 자연스러운 대화 능력을 개발하는 데 도움이 됩니다. 셋째, 40개 언어로 확장된 대규모 다언어 데이터셋 구축입니다.[1]

### 문제 정의 및 제안 방법

**해결하고자 하는 문제**

기존 개방소스 모델들(Alpaca, Vicuna, BELLE 등)은 영어 또는 특정 고자원 언어에만 최적화되어 있습니다. LLaMA 기반 모델들도 기본 사전훈련에 비라틴 언어 데이터가 충분하지 않아 저자원 언어 성능이 제한적입니다. 이는 "AI 독점(AI Supremacy)" 문제로 이어져 특정 회사와 언어에 의존하는 불평등한 AI 생태계를 만듭니다.[1]

**제안된 방법론**

**1. 데이터 구성**

Phoenix는 두 가지 유형의 데이터를 활용합니다:[1]

**명령어 데이터 (267K 샘플)**
- Alpaca 52K 명령어 (영어 및 GPT-4 버전)
- 다언어 번역 명령어: 사후 번역(Post-Translation) 및 사후 출력(Post-Output) 방식
- 역할 중심 사용자 맞춤형 명령어 (65K 샘플)

사후 번역(Algorithm 1)에서는 명령어와 출력 모두를 번역하여 답변 품질을 유지하고, 사후 출력 방식에서는 명령어만 번역한 후 출력은 GPT-3.5를 사용하여 언어-문화 특화 답변을 생성합니다.[1]

**대화 데이터 (189K 샘픔)**
- ShareGPT (90K 다중 회차)
- Discord ChatGPT 채널 (8K 샘플)
- 영어를 제외하지 않고 40개 언어로 번역[1]

전체 데이터셋은 465K 샘플, 939K 회차, 약 986 토큰/샘플의 규모입니다. 데이터 분포는 영어 58.6%, 중국어 20.9%, 스페인어 3.0%, 한국어 2.5% 등 다양한 언어를 포함합니다.[1]

**2. 모델 구조**

Phoenix는 **BLOOMZ 7B**를 백본으로 사용하고, 다음과 같은 훈련 설정을 적용합니다:[1]

- 최대 컨텍스트 길이: 2,048
- 옵티마이저: AdamW
- 배치 크기: 256
- 에포크: 3
- 학습률: 2e-5
- 가중치 감소: 0

라틴 언어 성능을 개선하기 위해 **LLaMA 7B/13B** 백본을 사용한 **Chimera** 모델도 제시합니다.[1]

### 모델 일반화 성능 향상

**핵심 아이디어: 다언어 세금(Multilingual Tax) 개념**

Phoenix는 중요한 개념을 정의합니다: **"다언어 세금이란 제한된 크기의 다언어 모델이 특정 언어 작업 수행 시 언어 특화 모델보다 성능이 떨어질 수 있다는 것"** 이는 모델이 많은 언어에 적응하도록 설계되어 특정 언어에 최적화되지 않기 때문입니다. 그러나 논문은 이러한 비용을 **민주화의 가치**로 정당화합니다.[1]

**성능 향상 메커니즘**

1. **다중 데이터 유형의 상승 작용**: 절제 연구(Ablation Study)에 따르면, 명령어 데이터 추가로 중국어에서 5-6% 상대 개선, 영어에서 6.3% 개선을 달성합니다.[1]

2. **크로스 언어 전이**: BLOOM 백본이 46개 언어의 사전훈련 데이터를 포함하고 있어, 고자원 언어(영어, 중국어)의 지식이 저자원 언어로 전이됩니다. 이는 최신 연구에서 입증된 **바이텍스트 사전훈련(Bitext Pretraining)** 및 **다언어 매트릭스 정렬(Cross-Lingual Embedding Alignment)** 원리와 일치합니다.[2][3][1]

3. **언어별 표현 공간 활용**: 최신 연구(2025년)에 따르면, 대상 언어의 언어-무관 의미 표현을 영어 표현과 정렬시키는 것이 효과적입니다. 이를 통해 모델은 외부 다언어 훈련 데이터 없이도 영어의 일반적인 능력을 상속받을 수 있습니다.[4]

**성능 평가 결과**

Phoenix의 일반화 성능은 다음과 같습니다:[1]

| 언어 | Phoenix 성능 | 경쟁 모델 | 결과 |
|------|------------|---------|------|
| **중국어** | 122.70 | BELLE-7B-2m | Phoenix 우수 |
| **중국어** | 135.30 | Chinese-Alpaca-7B | Phoenix 우수 |
| **중국어** | 125.20 | Chinese-Alpaca-13B | Phoenix 우수 (더 작은 크기) |
| **영어** | 121.2 | Vicuna-7B | Phoenix 우수 |
| **영어** | 90.92 | Vicuna-13B | Phoenix 뒤처짐 |
| **다언어** | - | Guanaco | Phoenix 우수 (대부분 언어) |

특히 Chimera(LLaMA 기반 라틴 버전)는 GPT-4로부터 **96.6% ChatGPT 품질 점수**를 받아 개방소스 모델 중 최고 성능을 달성했습니다.[1]

### 일반화 성능과 저자원 언어 처리

**저자원 언어에서의 우수성**

Phoenix의 가장 혁신적인 기여는 저자원 언어에서의 성능입니다. 아랍어, 일본어, 한국어, 포르투갈어, 스페인어 등 다양한 비라틴 언어에서 Guanaco를 포함한 다른 개방소스 모델들을 크게 능가합니다. Beat Rate 메트릭에서 Phoenix vs. Guanaco 성능:[1]

- 프랑스어: 92.80% (Phoenix 우수)
- 스페인어: 93.60%
- 포르투갈어: 95.50%
- 이탈리아어: 75.80%
- 독일어: 47.00%
- 아랍어: 97.00%
- 일본어: 86.25%
- 한국어: 93.75%[1]

**인간 평가 결과**

100개 중국어 질문에 대한 인간 평가 결과:[1]

- Phoenix vs. BELLE-7B-2m: 55승 31동 14패 (우수)
- Phoenix vs. Chinese-Alpaca-13B: 56승 31동 13패 (우수)
- Phoenix vs. ChatGPT: 12승 35동 53패 (경쟁)
- Phoenix vs. Baidu-Wenxin: 29승 25동 46패 (경쟁)

### 한계 (Limitations)

1. **다언어 세금**: 라틴 언어에서 언어 특화 모델(Vicuna)보다 성능이 낮습니다. 이를 해결하기 위해 Chimera를 제안하지만, 비라틴 언어 능력이 저하됩니다.[1]

2. **평가 엄밀성**: GPT-4/3.5를 평가 심판으로 사용하는 것은 완벽하지 않습니다. GPT-3.5는 신뢰성이 GPT-4보다 낮고 높은 점수 할당 경향이 있습니다.[1]

3. **공통 인지 부재**: 모델이 상식이 부족하고 제한된 지식 도메인, 편향성, 감정 이해 부족 등의 한계를 가집니다.[1]

4. **번역 왜곡**: 언어 특화 명령어(예: "중국 칠언 절구 시 작성")는 번역 과정에서 왜곡될 수 있습니다.[1]

5. **RLHF 부재**: ChatGLM-6B나 GPT-3.5와 달리 인간 피드백 강화학습(RLHF)을 적용하지 않아 성능 상한이 있습니다.[1]

### 앞으로의 연구에 미치는 영향

**1. 다언어 LLM 설계 패러다임 변화**

Phoenix는 단순히 "번역 + 파인튜닝"을 넘어 **"사전훈련 단계부터의 다언어성"**을 강조합니다. 최근 연구(2024-2025)에서 이 원칙이 확산되고 있습니다. EMMA-500(2025)과 Xmodel-1.5(2024) 등 후속 모델들도 같은 철학을 따릅니다.[5][2][1]

**2. 저자원 언어 연구의 가속화**

최근 연구들은 Phoenix의 주장을 검증하고 확장하고 있습니다:[6][7][8][2]

- **계속 사전훈련(Continual Pre-training)**: Swallow(2024)는 LLaMA를 일본어로 지속 사전훈련하여 성능을 크게 향상시킵니다.[3]

- **언어 어댑터 기법**: MAD-X 프레임워크의 진화로, 매개변수 효율적 적응이 표준화되고 있습니다.[3]

- **계층 교환(Layer Swapping)**: 2025년 연구에서 제시된 새로운 기법으로, 모델 전문성을 크로스 언어 전이에 활용합니다.[8]

**3. 평가 방법론의 개선**

Phoenix의 GPT-4 기반 평가 방법이 이후 연구들의 표준이 되었습니다. 그러나 동시에 인간 평가의 중요성도 강조되고 있으며, 더 견고한 평가 프레임워크 개발이 진행 중입니다.[9][1]

**4. 다언어 표현 정렬 연구**

최신 연구(2025)의 "Lens" 접근법은 Phoenix의 암묵적 가정을 명시적으로 다룹니다: **"언어 무관 의미 표현 정렬"** 이를 통해 저자원 언어를 위해 고자원 언어(영어)의 표현 공간을 활용하는 방법이 체계화되고 있습니다.[4]

**5. 토크나이제이션 불공정성 문제 제기**

Phoenix의 작업 이후 **"토큰 세금(Token Tax)"** 개념이 제기되었습니다. 형태소가 풍부한 언어들이 더 많은 토큰이 필요하여 훈련 비용이 4-25배 증가한다는 것입니다. 이는 Phoenix가 제기한 "다언어 세금" 개념보다 더 근본적인 불공정성을 드러냅니다.[10]

### 앞으로 연구 시 고려할 점

**1. 토크나이제이션 개선**
형태소 인식 토크나이제이션(Morphologically-aware Tokenization)으로 모든 언어에 공평한 처리를 제공해야 합니다.[10]

**2. 언어 간 능력 추출 및 전이**
최신 MAET(Multi-lingual Ability Extraction and Transfer) 같은 방법이 고자원에서 저자원 언어로의 능력 전이를 효율적으로 수행할 수 있습니다.[11]

**3. 문화 특화 데이터 개발**
사후 번역과 사후 출력 방식 중 후자의 장점인 "언어-문화 특화 답변"을 더 적극 활용해야 합니다.[1]

**4. 강화학습(RLHF) 다언어 적용**
Unlocking Multilingual Preference Optimization(2024) 같은 연구로 RLHF를 다언어에 효과적으로 적용하는 방법이 개발 중입니다.[12]

**5. 체계적 평가 프레임워크**
최신 연구(2023-2024)에서는 37개 언어, 고/중/저/극저자원 분류를 포함한 체계적 평가가 표준화되고 있습니다.[13][9]

**6. 모듈식 아키텍처 탐구**
언어별, 작업별 어댑터의 조합을 통한 모듈식 접근으로 확장성과 효율성을 개선할 수 있습니다.[3]

Phoenix는 ChatGPT 민주화 운동에서 **"언어 평등"**이라는 중요한 차원을 추가했으며, 이는 AI의 진정한 접근성과 공정성에 대한 근본적인 질문을 던졌습니다. 현재의 다언어 LLM 연구 트렌드는 Phoenix의 통찰력을 검증하고 더욱 정하고 더욱 정교화하고 있습니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f09ae438-c55a-46d2-a820-6cea6c7d952d/2304.10453v1.pdf)
[2](https://arxiv.org/pdf/2409.17892.pdf)
[3](https://www.rohan-paul.com/p/transfer-learning-across-languages)
[4](https://arxiv.org/html/2410.04407v1)
[5](http://arxiv.org/pdf/2411.10083.pdf)
[6](https://arxiv.org/pdf/2301.09626.pdf)
[7](https://aclanthology.org/2023.findings-emnlp.931.pdf)
[8](https://arxiv.org/html/2410.01335v1)
[9](https://aclanthology.org/2023.findings-emnlp.878.pdf)
[10](https://arxiv.org/html/2509.05486v1)
[11](https://arxiv.org/pdf/2410.07825.pdf)
[12](https://aclanthology.org/2024.emnlp-main.729.pdf)
[13](https://llm-low-resource-lang.github.io/content/LLMs_for_Low_Resource_Languages.pdf)
[14](https://www.aclweb.org/anthology/P19-1299.pdf)
[15](https://arxiv.org/abs/2207.09157)
[16](https://aclanthology.org/2024.semeval-1.226.pdf)
[17](https://sigtyp.github.io/ws2024-mrl.html)
[18](https://www.together.ai/blog/fine-tuning-llms-for-multi-turn-conversations-a-technical-deep-dive)
[19](https://openreview.net/forum?id=Nfu3bUkmdH)
[20](https://arxiv.org/html/2509.11414v1)
[21](https://arxiv.org/html/2308.10792v9)
[22](https://www.cep.eu/eu-topics/details/the-algorithmic-hand-how-large-language-models-disrupt-competition-and-democracy.html)
[23](https://aclanthology.org/volumes/2024.mrl-1/)
[24](https://arxiv.org/pdf/2303.01911.pdf)
[25](https://arxiv.org/pdf/2403.02370.pdf)
[26](https://www.mdpi.com/2078-2489/14/12/638/pdf?version=1701249996)
[27](http://arxiv.org/pdf/2502.04269.pdf)
[28](https://arxiv.org/pdf/2210.15424.pdf)
[29](https://arxiv.org/pdf/2211.01786.pdf)
[30](https://aclanthology.org/2023.emnlp-main.431.pdf)
[31](https://arxiv.org/pdf/1911.02116.pdf)
[32](https://arxiv.org/html/2411.11072v1)
[33](https://ceur-ws.org/Vol-3882/km4law-2.pdf)
[34](https://arxiv.org/pdf/2304.10453.pdf)
[35](https://www.datacamp.com/blog/12-gpt4-open-source-alternatives)
[36](https://aclanthology.org/2023.eamt-1.16.pdf)
[37](https://www.byteplus.com/en/topic/546361)
