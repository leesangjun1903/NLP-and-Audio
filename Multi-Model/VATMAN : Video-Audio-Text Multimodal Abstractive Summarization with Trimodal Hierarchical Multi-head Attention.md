# VATMAN : Video-Audio-Text Multimodal Abstractive Summarization with Trimodal Hierarchical Multi-head Attention

## 핵심 주장과 주요 기여

VATMAN(Video-Audio-Text Multimodal Abstractive summarizatioN) 논문의 핵심 주장은 비디오, 오디오, 텍스트의 **삼중 모달리티를 효과적으로 융합**하여 기존의 이중 모달리티 접근법보다 우수한 요약 성능을 달성할 수 있다는 것입니다.[1]

주요 기여는 다음과 같습니다:

- **Trimodal Hierarchical Multi-head Attention(THMA) 메커니즘** 도입으로 비디오, 오디오, 텍스트 간의 계층적 관계와 의존성을 효과적으로 포착[1]
- **사전훈련된 언어모델 BART 기반의 Transformer 아키텍처**를 활용하여 멀티모달 정보를 통합[2][1]
- How2 데이터셋에서 **인간이 작성한 요약보다 더 유창한 요약문 생성** 능력 입증[1]
- 기존 RNN 기반 모델 대비 **ROUGE-1 52.53, ROUGE-L 44.18, BLEU-1 49.49로 우수한 성능** 달성[1]

## 문제 정의와 제안 방법

### 해결하고자 하는 문제

기존 멀티모달 추상적 요약 연구는 주로 **텍스트와 비디오 두 모달리티만 활용**하여 오디오 정보가 제공하는 풍부한 맥락 정보를 간과했습니다. 특히 생성형 사전훈련 언어모델(GPLM)과 비디오, 오디오, 텍스트를 동시에 활용하는 연구가 부족했습니다.[3][4][1]

### 제안 방법: THMA(Trimodal Hierarchical Multi-head Attention)

VATMAN의 핵심 메커니즘인 THMA는 다음과 같이 작동합니다:

**1단계: Video-Text Fusion**
- 텍스트 입력을 쿼리(Query)로, 비디오 입력을 키(Key)와 값(Value)으로 사용
- $$Z_t' = \text{Attention}(Z_t, Z_{video}, Z_{video}) $$

**2단계: Video-Audio-Text Fusion**  
- 1단계 출력 $$ Z_t' $$을 쿼리로, 오디오 입력을 키와 값으로 사용
- $$Z_t'' = \text{Attention}(Z_t', Z_{audio}, Z_{audio}) $$

여기서 최종 출력 $$ Z_t'' \in \mathbb{R}^{N \times d_t} $$는 원래 텍스트 입력과 동일한 차원을 유지하여 **다층 스택킹 시에도 차원 불변성**을 보장합니다.[1]

### 특징 추출

- **비디오**: 사전훈련된 3D ResNeXt-101 네트워크를 통해 초당 16프레임에서 **2,048차원 특징** 추출[5][1]
- **오디오**: Kaldi를 사용하여 40차원 필터뱅크 특징과 3차원 피치 특징을 결합한 **43차원 특징** 추출하며, CMVN(Cepstral Mean and Variance Normalization) 적용[5][1]

## 모델 구조

VATMAN은 **표준 Transformer encoder-decoder 아키텍처**를 기반으로 하되, 각 인코더 블록 끝에 **Video-Audio-Text fusion unit**을 삽입한 구조입니다:[1]

1. **인코더**: L개의 레이어, 각각 Multi-head Self-Attention과 Feed-Forward 네트워크 포함
2. **디코더**: L개의 레이어, Masked Multi-head Self-Attention과 Encoder-Decoder Attention 포함
3. **융합 블록**: 각 인코더 레이어 후에 THMA 메커니즘, residual connection, layer normalization 적용[1]

이 구조는 **BART 모델을 백본**으로 하여 기존 사전훈련된 언어모델의 강력한 생성 능력을 활용합니다.[2][1]

## 성능 향상

How2 데이터셋 실험 결과, VATMAN은 기존 방법들을 크게 앞서는 성능을 보였습니다:[1]

| 방법 | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-1 |
|------|---------|---------|---------|---------|
| T5 (Text only) | 45.08 | 21.59 | 37.09 | 37.09 |
| BART (Text only) | 49.07 | 26.72 | 41.66 | 41.30 |
| RNN (Trimodal) | 48.85 | 29.51 | 43.23 | - |
| **VATMAN (Ours)** | **52.53** | **29.49** | **44.18** | **49.49** |

특히 **텍스트만 사용하는 BART 대비 ROUGE-1에서 3.46점, BLEU-1에서 8.19점** 향상을 보였습니다.[1]

## 일반화 성능 향상 가능성

### 계층적 융합의 장점

VATMAN의 **계층적 융합 메커니즘**은 일반화 성능 향상에 중요한 요소들을 제공합니다:[6][7]

1. **단계적 정보 통합**: Video-Text 융합 후 Audio 정보를 추가로 통합하는 방식으로 **각 모달리티의 고유한 특성을 보존**하면서 상호 보완적 정보를 효과적으로 활용[1]

2. **모달리티별 가중치 학습**: **attention 메커니즘을 통해 각 모달리티의 중요도를 동적으로 조정**하여 다양한 도메인의 데이터에 적응 가능[8][3]

### 사전훈련 모델 활용

**BART와 같은 대규모 사전훈련 모델의 활용**은 일반화 성능을 크게 향상시킵니다:[9][2]

- **풍부한 언어적 사전 지식**: 다양한 텍스트 데이터로 사전훈련된 모델의 지식이 새로운 도메인에도 전이 가능
- **강건한 표현 학습**: Transformer의 self-attention 메커니즘이 **장거리 의존성과 복잡한 패턴을 효과적으로 학습**[10][11]

### 한계와 개선점

하지만 몇 가지 일반화 성능 제약요인도 존재합니다:

1. **도메인 특화성**: How2 데이터셋의 **교육용 비디오에 특화된 학습**으로 인해 다른 도메인(뉴스, 엔터테인먼트 등)에서의 성능 보장이 어려움[12][13]

2. **모달리티 불균형**: 텍스트 모달리티에 대한 의존도가 높아 **오디오나 비디오 품질이 낮은 경우 성능 저하** 가능성[3][8]

3. **계산 복잡도**: **삼중 모달리티 처리로 인한 높은 계산 비용**이 실시간 응용이나 자원 제약 환경에서의 활용을 제한[1]

## 앞으로의 연구에 미치는 영향

### 긍정적 영향

1. **멀티모달 융합 방법론 발전**: THMA의 **계층적 융합 접근법**은 향후 멀티모달 연구의 새로운 패러다임을 제시[14][15][9]

2. **오디오 모달리티 재조명**: 기존 연구에서 간과되었던 **오디오 정보의 중요성**을 입증하여 관련 연구 활성화 기여[4][8][3]

3. **사전훈련 모델 활용 확산**: **GPLM과 멀티모달 정보의 효과적 결합 방법**을 제시하여 관련 연구 방향 제시[16][2]

### 향후 연구 시 고려사항

1. **도메인 적응성 강화**: 
   - **다양한 도메인 데이터셋에서의 검증** 필요[17][18]
   - **도메인 불변 특징 학습** 방법론 개발 요구
   - **Few-shot 학습**이나 **메타 러닝** 기법과의 결합 연구

2. **효율성 개선**:
   - **모델 경량화 기술** 적용 연구 필요[19][20]
   - **지식 증류(Knowledge Distillation)** 기법을 통한 성능 유지하면서 크기 축소
   - **동적 모달리티 선택** 메커니즘 도입

3. **평가 지표 다양화**:
   - **ROUGE, BLEU 외의 의미적 평가 지표** 개발 필요[21][22][23]
   - **인간 평가와의 상관관계가 높은 자동 평가 지표** 연구
   - **다국어 및 문화적 맥락을 고려한 평가** 방법론

4. **윤리적 고려사항**:
   - **편향성 문제** 해결을 위한 공정성 연구
   - **프라이버시 보호**를 위한 federated learning 적용
   - **설명 가능한 AI** 관점에서의 attention 시각화 연구

VATMAN은 멀티모달 요약 분야의 중요한 발전을 이루었지만, 실용적 활용을 위해서는 **일반화 성능, 효율성, 평가 방법론**에 대한 지속적인 연구가 필요합니다.[24][25][17]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/eda8fb04-c064-4a42-a897-eab0aada683b/VATMAN__Video-Audio-Text_Multimodal_Abstractive_Summarization_with_Trimodal_Hierarchical_Multi-head_Attention.pdf
[2] https://nips2018vigil.github.io/static/papers/accepted/26.pdf
[3] https://aclanthology.org/2020.nlpbt-1.7.pdf
[4] https://arxiv.org/abs/2010.08021
[5] https://2023.ictc.org/media?key=site%2Fictc2023a%2Fabs%2FP4-29.pdf
[6] https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
[7] https://aclanthology.org/W18-0607/
[8] https://aclanthology.org/2020.nlpbt-1.7/
[9] https://eng.ox.ac.uk/media/ttrg2f51/2023-ieee-px.pdf
[10] https://scholarworks.bwise.kr/cau/bitstream/2019.sw.cau/69879/1/Transformer%20Architecture%20and%20Attention%20Mechanisms%20in%20Genome%20Data%20Analysis%20A%20Comprehensive%20Review.pdf
[11] https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf
[12] https://paperswithcode.com/dataset/how2
[13] https://github.com/srvk/how2-dataset
[14] https://aclanthology.org/2023.acl-long.124.pdf
[15] https://arxiv.org/abs/2104.11178
[16] https://arxiv.org/html/2503.01022v1
[17] https://arxiv.org/html/2501.18592v1
[18] https://openreview.net/forum?id=zyBJodMrn5
[19] https://arxiv.org/abs/2112.04446
[20] https://openaccess.thecvf.com/content/CVPR2022/papers/Shvetsova_Everything_at_Once_-_Multi-Modal_Fusion_Transformer_for_Video_Retrieval_CVPR_2022_paper.pdf
[21] https://aclanthology.org/W04-1013.pdf
[22] https://arxiv.org/pdf/2303.15078.pdf
[23] https://doc.superannotate.com/docs/guide-bleu-rouge
[24] https://pubmed.ncbi.nlm.nih.gov/36029345/
[25] https://encord.com/blog/top-multimodal-models/
[26] https://k-knowledge.kr/srch/read.jsp?id=270512404
[27] https://aclanthology.org/P19-1659/
[28] https://www.paperdigest.org/review/?paper_id=arxiv-1906.07901
[29] https://openaccess.thecvf.com/content/CVPR2024/papers/Qiu_MMSum_A_Dataset_for_Multimodal_Summarization_and_Thumbnail_Generation_of_CVPR_2024_paper.pdf
[30] https://www.sciencedirect.com/science/article/abs/pii/S0925231224005794
[31] https://velog.io/@dlthdus8450/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-MAST-Multimodal-Abstractive-Summarization-with-Trimodal
[32] https://arxiv.org/abs/1906.07901
[33] https://ashwinpathak20.github.io/multimodal/2021/10/blog-post-6/
[34] https://aclanthology.org/2022.emnlp-main.468/
[35] https://www.sciencedirect.com/science/article/abs/pii/S0950705121004159
[36] https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
[37] https://ai-information.blogspot.com/2019/04/text-generation-evaluation-06-rouge.html
[38] https://arxiv.org/html/2408.04723v1
[39] https://wikidocs.net/228090
[40] https://www.sciencedirect.com/science/article/abs/pii/S0952197623015191
[41] http://www.d2l.ai/chapter_attention-mechanisms-and-transformers/index.html
[42] https://www.sciencedirect.com/science/article/pii/S1877750325000481
[43] https://www.sciencedirect.com/science/article/pii/S0952197624004974/pdf
[44] https://arxiv.org/abs/2306.05012
[45] https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation
[46] https://byeonggeuk.tistory.com/65
[47] https://wikidocs.net/259213
[48] http://www.apsipa2024.org/files/papers/52.pdf
[49] https://aclanthology.org/2024.lrec-main.1374.pdf
[50] https://arxiv.org/abs/2404.09365
[51] https://cloudsek.com/blog/hierarchical-attention-neural-networks-beyond-the-traditional-approaches-for-text-classification
[52] https://www.nature.com/articles/s41598-021-98408-8
[53] https://coshin.tistory.com/50
[54] https://jeongwooyeol0106.tistory.com/151
[55] https://www.sciencedirect.com/science/article/abs/pii/S0020025522014876
[56] https://www.sciencedirect.com/science/article/abs/pii/S0959652625013496
[57] https://www.sciencedirect.com/science/article/abs/pii/S1566253523002385
[58] https://www.sciencedirect.com/science/article/abs/pii/S0893608022000600
