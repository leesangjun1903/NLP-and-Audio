
# SmolDocling: An Ultra-Compact Vision-Language Model for End-to-End Multi-Modal Document Conversion

> **📌 논문 정보**
> - **제목**: SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion
> - **저자**: Ahmed Nassar, Andres Marafioti, Matteo Omenetti 외 10명
> - **소속**: IBM Research, HuggingFace
> - **발표일**: 2025년 3월 14일
> - **arXiv**: [2503.11576](https://arxiv.org/abs/2503.11576)
> - **ICCV 2025** 채택 확인

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

SmolDocling은 end-to-end 문서 변환을 목표로 하는 초경량(ultra-compact) 비전-언어 모델(VLM)로, 페이지 전체를 처리하여 모든 페이지 요소를 문맥과 위치 정보와 함께 캡처하는 새로운 범용 마크업 포맷인 **DocTags**를 생성한다. 대형 기반 모델이나 다수의 특화 모델로 구성된 앙상블 파이프라인에 의존하는 기존 접근 방식과 달리, SmolDocling은 단 **256M 파라미터**의 VLM으로 문서 요소의 내용, 구조 및 공간적 위치를 정확하게 캡처하는 end-to-end 변환을 제공한다.

### ✅ 주요 기여 요약

| 기여 항목 | 설명 |
|-----------|------|
| **DocTags 포맷 제안** | 문서 구조·내용·공간 정보를 통합 표현하는 새 마크업 언어 |
| **초경량 아키텍처** | 256M 파라미터로 최대 27배 큰 모델과 경쟁 |
| **공개 데이터셋 기여** | 차트, 표, 수식, 코드 인식용 신규 데이터셋 공개 |
| **커리큘럼 학습 전략** | 단계적 훈련으로 효율적 시각-의미 정렬 달성 |
| **다양한 문서 유형 지원** | 학술 논문에 국한되지 않고 비즈니스 문서, 특허, 양식 등 포괄 |

SmolDocling은 코드 목록, 표, 수식, 차트, 목록 등 다양한 문서 특성을 비즈니스 문서, 학술 논문, 기술 보고서, 특허, 양식 등 광범위한 문서 유형에 걸쳐 재현하는 강건한 성능을 보이며, 차트·표·수식·코드 인식을 위한 새로운 공개 데이터셋도 기여한다. 실험 결과, SmolDocling은 최대 27배 더 큰 VLM과 경쟁하면서도 계산 요구사항을 실질적으로 줄인다.

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 🔴 해결하고자 하는 문제

기존의 앙상블 시스템이나 초대형 기반 모델 기반 접근법은 파인 튜닝의 어려움, 일반화 문제, 환각(hallucination), 높은 계산 비용이라는 심각한 장애물에 직면해 있었다. 앙상블 시스템은 각 하위 작업에 맞게 수작업으로 만든 파이프라인에 의존하기 때문에 일반화에 어려움을 겪고, 멀티모달 기반 모델은 높은 계산 비용과 환각 문제에 직면한다.

문헌에 따르면 강건한 멀티모달 문서 이해 모델 훈련에 적합한 고품질 공개 접근 가능 데이터셋의 가용성에도 심각한 격차가 존재하며, LVLM에 의존하면 환각 및 상당한 계산 자원 사용 등 관련 일반적 문제가 발생해 품질과 비용 측면 모두에서 비현실적이다.

---

### 2.2 🟢 제안하는 방법

#### (A) DocTags: 통합 마크업 표현 형식

SmolDocling의 핵심 혁신은 문서의 전체 내용과 레이아웃 특성을 효율적으로 표현하도록 최적화된 문서 마크업 포맷인 **DocTags**의 도입이다. DocTags는 텍스트 내용과 문서 구조를 명확히 분리하는 모호하지 않은 태그의 구조화된 어휘를 정의하며, 이 통합 문서 표현은 텍스트, 표, 이미지, 수식, 코드 스니펫 등 페이지 내 모든 요소의 내용, 구조, 공간적 위치를 캡처한다.

DocTags 마크업 포맷은 텍스트 내용과 문서 구조를 명시적으로 분리하고, 모호하지 않은 태그의 구조화된 어휘를 정의하며, 경계 상자(bounding box) 좌표를 통해 위치 및 레이아웃 정보를 인코딩하고, 중첩 요소(예: 표/그림 내 캡션)를 지원하며, OTSL(Open Table Structure Language) 어휘를 사용하여 표 구조를 인코딩하도록 설계되었다.

표의 위치 정보는 정규화된 좌표(bounding box)로 인코딩된다. 논문에서 위치 표현에 사용되는 좌표 형식은 다음과 같이 수식으로 나타낼 수 있다:

$$\text{bbox} = \left(\frac{x_1}{W}, \frac{y_1}{H}, \frac{x_2}{W}, \frac{y_2}{H}\right)$$

여기서 $(x_1, y_1)$은 좌상단, $(x_2, y_2)$는 우하단 좌표이며, $W$, $H$는 각각 이미지의 너비와 높이이다.

표 구조 인식을 위해 DocTags는 OTSL(Optimized Table Structure Language) 어휘를 통합하여 셀 스팬 및 헤더 정보를 포함한 표 구조를 인코딩하며, OTSL 토큰은 텍스트와 인터리빙되어 구조와 내용을 동시에 인코딩한다. DocTags는 OTSL을 `<fcel>`(내용이 있는 전체 셀)과 `<ecel>`(빈 셀)로 확장하며, `<ched>`, `<rhed>`, `<srow>`으로 열 헤더, 행 헤더, 테이블 섹션을 표시한다.

#### (B) 커리큘럼 학습 전략 (Curriculum Learning)

SmolDocling은 3단계 커리큘럼 학습 접근법을 채택한다:
1. **비전 인코더를 고정(frozen)**한 채로 LLM을 DocTags 포맷에 적응
2. **비전 인코더를 해동(unfrozen)**하여 사전 훈련 데이터셋으로 end-to-end 학습
3. 표, 코드, 수식, 차트에 대한 **태스크 특화 데이터셋으로 파인 튜닝**

커리큘럼 학습에서 각 단계의 학습 목표는 표준 언어 모델링 손실인 cross-entropy loss로 정의된다:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta\left(y_t \mid y_{ < t}, \mathbf{v}\right)$$

여기서 $y_t$는 $t$번째 DocTags 토큰, $y_{ < t}$는 이전 토큰 시퀀스, $\mathbf{v}$는 비전 인코더로부터 투영된 임베딩 벡터이다.

---

### 2.3 🏗️ 모델 구조 (Architecture)

SmolDocling은 SmolVLM 아키텍처 접근 방식, 특히 SmolVLM-256M 변형을 기반으로 구축되었으며, **SigLIP base patch-16/512 (93M) 시각 백본**과 **SmolLM-2 계열의 경량 변형 (135M) 언어 백본**으로 구성된다.

SmolDocling은 문서 페이지 이미지를 DocTags 시퀀스로 변환한다. 먼저 입력 이미지가 비전 인코더로 인코딩되고 투영 및 풀링을 통해 재형성된다. 그런 다음 투영된 임베딩은 사용자 프롬프트의 텍스트 임베딩과 연결되고, 최종적으로 LLM이 이 시퀀스를 사용하여 DocTags 시퀀스를 자기회귀적(autoregressively)으로 예측한다.

언어 백본으로는 SmolLM-2 계열(135M)의 경량 변형을 사용하며, 각 512×512 이미지 패치를 **64개의 시각 토큰으로 압축**하는 radical pixel shuffle 방법을 채택한다. 토큰화 효율성은 픽셀-토큰 비율을 4096 픽셀/토큰으로 높이고 서브 이미지 구분자를 위한 특수 토큰을 도입함으로써 향상된다.

전체 아키텍처를 수식으로 표현하면:

$$\hat{y} = \text{LLM}\left(\left[\text{Proj}\left(\text{VisionEncoder}(I)\right); \text{TextEmbed}(p)\right]\right)$$

여기서 $I$는 입력 이미지, $p$는 텍스트 프롬프트, $\text{Proj}(\cdot)$은 픽셀 셔플 및 풀링 기반 투영 레이어이다.

**아키텍처 요약표:**

| 컴포넌트 | 세부 사항 |
|----------|-----------|
| **Vision Encoder** | SigLIP base patch-16/512 (93M 파라미터) |
| **Language Model** | SmolLM-2 계열 경량 변형 (135M 파라미터) |
| **이미지 압축** | 512×512 패치 → 64 시각 토큰 (radical pixel shuffle) |
| **전체 파라미터** | 256M |
| **출력 형식** | DocTags (OTSL 통합) |

---

### 2.4 📊 성능 향상

성능 평가에서, 전체 페이지 전사(DocLayNet)에 대해 SmolDocling은 **F1 점수 0.80, BLEU 0.58**을 달성하여 Qwen2.5-VL(F1: 0.72, BLEU: 0.46)을 능가하였다. 코드 목록에서 F1 0.92, 수식에서 GOT의 최고 F1 0.95와 동등한 성능을 보였다. DocLayNet에서 SmolDocling의 mAP 0.231은 Qwen2.5-VL의 0.133을 능가하였으나, 둘 다 인간 성능(0.82)에는 미치지 못한다.

효율성 면에서, 소비자용 GPU에서 페이지당 평균 **0.35초**의 처리 속도로 **500MB 미만의 VRAM**을 소모한다.

---

### 2.5 ⚠️ 한계

연구자들은 **페이지 요소 위치 지정(localization)**을 추가적인 개선이 필요한 중요한 영역으로 식별한다. SmolDocling 출력의 일반적인 결함에는 누락된 태그, 잘못된 구조, 토큰 반복 문제가 포함된다.

훈련 데이터셋이 크지만, SynthCodeNet, SynthFormulaNet, SynthDocNet과 같은 합성 또는 규칙 기반 소스에 크게 의존한다는 점이 우려된다. 예를 들어, 코드 인식의 경우 훈련 데이터는 Pygments를 사용하여 렌더링되지만, 실제로는 IDE 스크린샷부터 스캔 문서, 인쇄된 책까지 다양한 형식으로 코드 스니펫이 나타나며, 이는 모델이 합성 훈련 세트를 넘어 얼마나 잘 일반화할 수 있는지에 대한 의문을 제기한다.

또한 SmolDocling은 동일한 아키텍처에서 훈련 목표로 사용될 때 DocTags가 HTML 또는 Markdown과 어떻게 비교되는지를 평가하지 않는다. 이러한 비교 실험 없이는 DocTags가 견고성, 일반화, 사용성 측면에서 어떻게 수행되는지 말하기 어렵다.

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 일반화를 높이는 현재 설계 요소

광범위한 사전 훈련 및 태스크 특화 데이터셋과 함께 커리큘럼 학습 접근 방식을 통해 SmolDocling은 다양한 문서 유형에 걸쳐 잘 일반화한다.

DocLayNet-PT와 같은 대규모 데이터셋과 PubTables-1M, The Cauldron 같은 공개 코퍼스를 결합함으로써 다양한 문서 유형과 태스크에 걸쳐 SmolDocling의 다재다능함이 보장된다.

연구자들은 목표 지향적 훈련, 혁신적인 데이터 증강, DocTags와 같은 새로운 태깅 포맷이 모델 크기와 복잡성에 전통적으로 관련된 한계를 극복할 수 있음을 성공적으로 보여준다.

### 3.2 일반화 관련 한계 및 개선 방향

전통적 접근 방식인 앙상블 시스템이나 초대형 기반 모델은 파인 튜닝의 어려움, **일반화 문제**, 환각, 높은 계산 비용 등의 장애물에 직면한다.

일반화 성능 향상 가능성을 다각도로 정리하면:

| 측면 | 현재 상태 | 향상 가능성 |
|------|-----------|-------------|
| **문서 다양성** | 학술, 비즈니스, 기술 보고서 등 포괄 | 손글씨, 저품질 스캔 문서 확장 필요 |
| **언어 지원** | 주로 영어 기반 | 다국어 데이터셋 확장 시 향상 가능 |
| **합성 데이터 의존** | SynthCodeNet 등 합성 데이터 중심 | 실제 데이터(real-world) 비율 증가 필요 |
| **요소 위치 지정** | mAP 0.231로 인간(0.82) 대비 낮음 | 로컬리제이션 훈련 강화로 개선 가능 |
| **DocTags vs 기타 포맷** | HTML/Markdown 비교 실험 미실시 | 비교 연구를 통한 최적 포맷 선택 여지 |

미래 작업은 위치 지정 정확도를 개선하고 DocTags 표현 포맷의 향상을 탐색하는 것을 목표로 한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

아래는 SmolDocling과 주요 관련 모델들의 비교표이다:

| 모델 | 발표 연도 | 파라미터 | 주요 특징 | SmolDocling 대비 |
|------|-----------|----------|-----------|-----------------|
| **LayoutLMv3** | 2022 | ~125M | 텍스트·이미지 마스킹 사전학습 | 레이아웃 중심, 단일 태스크 편향 |
| **Nougat** | 2023 | 350M | 학술 문서 PDF → Markdown 변환 | 학술 논문 특화, 다양성 부족 |
| **GOT** | 2024 | 580M | OCR 2.0 일반 목적 인식 | 파라미터 2배 이상, 위치 정보 제한 |
| **mPLUG-DocOwl 1.5** | 2024 | ~7B | OCR-free 문서 이해 | 훨씬 큰 모델, 고비용 |
| **Qwen2.5-VL** | 2025 | 7B | 범용 VLM, 문서 이해 포함 | SmolDocling 대비 27배 이상 크기 |
| **SmolDocling** | 2025 | **256M** | DocTags + 커리큘럼 학습 | 최소 크기, 최대 효율 |

SmolDocling은 전체 페이지 평가에서 Qwen2.5-VL(7B), GOT(580M), Nougat(350M)을 모든 지표에서 능가하며, 수식 인식에서 GOT와 동등한 성능을 보이고, DocLayNet 테스트 세트에서 Qwen2.5-VL-7b를 크게 능가하였다.

독자적으로, SmolDocling은 페이지 내 모든 요소의 내용, 구조 및 공간적 위치를 캡처하는 통합 문서 표현을 학습하고 생성한다는 점이 차별화된다.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5.1 📌 향후 연구에 미치는 영향

SmolDocling은 통합된 출력 포맷을 갖춘 더 작고 신중하게 설계된 모델이 문서 이해 태스크에서 훨씬 더 큰 모델과 효과적으로 경쟁할 수 있음을 입증하며, 과학 논문을 넘어 다양한 문서 형식을 처리할 수 있는 자원 효율적인 멀티태스크 문서 이해 모델을 향한 명확한 경로를 제시한다.

이러한 흐름에서 IBM은 이미 생산 등급의 비전-언어 모델인 **Granite-Docling-258M**을 출시하는 방향으로 발전시키고 있다.

1. **경량 VLM 연구의 방향 제시**: "더 크면 더 좋다"는 패러다임에 도전하며, 목적 지향적 아키텍처 설계와 커리큘럼 학습의 중요성을 부각시킨다.

2. **DocTags 포맷의 표준화 가능성**: DocTags는 단일 표현 내에서 문서 내용, 구조 및 공간 정보를 효율적으로 캡처하는 새로운 범용 마크업 포맷으로, 향후 문서 AI 분야의 표준 표현 포맷 논의를 촉진할 수 있다.

3. **오픈소스 생태계 활성화**: SmolDocling의 오픈소스 특성은 OCR 기술의 효율성과 다재다능함에 대한 새로운 표준을 설정하며, 오픈 데이터셋과 효율적인 소형 모델 아키텍처를 통해 커뮤니티에 귀중한 자원을 제공한다.

### 5.2 🔬 향후 연구 시 고려할 점

1. **실세계 데이터 확보**: 훈련 데이터셋이 SynthCodeNet, SynthFormulaNet, SynthDocNet과 같은 합성 소스에 크게 의존하는 점은 모델이 실세계 분포를 충분히 반영하지 못할 수 있음을 의미하며, 실제 코드 스니펫은 IDE 스크린샷부터 스캔 문서까지 다양한 형식으로 나타나므로, 합성 훈련 세트를 넘어 모델의 일반화 능력에 의문이 생긴다.

2. **DocTags 포맷의 비교 검증**: SmolDocling이 동일한 아키텍처에서 HTML이나 Markdown과 DocTags를 직접 비교하지 않으므로, 향후 연구에서 출력 포맷이 견고성과 일반화에 미치는 영향을 비교 분석해야 한다.

3. **요소 위치 지정 정확도 개선**: 페이지 요소 위치 지정(localization)은 여전히 중요한 개선 영역이며, 누락된 태그, 잘못된 구조, 토큰 반복 문제가 잔재한다.

4. **다국어 확장**: 어떤 언어의 문서를 지원할지에 대한 질문이 커뮤니티에서 이미 제기되고 있으며, 이는 실용적 배포를 위한 중요한 연구 방향이다.

5. **인간 수준 성능 도달**: SmolDocling의 DocLayNet mAP 0.231은 인간 성능(0.82)에 비해 여전히 크게 낮아 개선의 여지가 크다.

6. **저해상도 및 열악한 품질 문서 처리**: TEDS 지표에서 SmolDocling은 FinTabNet에서 0.52, PubTables-1M에서 0.65를 기록하는데, 이는 저해상도 텍스트 전사 과정에서의 도전 과제를 드러낸다.

---

## 📚 참고 자료 및 출처

1. **ArXiv 원문**: Nassar, A. et al. (2025). *SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion*. arXiv:2503.11576. https://arxiv.org/abs/2503.11576

2. **ICCV 2025 공식 논문**: https://openaccess.thecvf.com/content/ICCV2025/papers/Nassar_SmolDocling_An_ultra-compact_vision-language_model_for_end-to-end_multi-modal_document_conversion_ICCV_2025_paper.pdf

3. **HuggingFace 논문 페이지**: https://huggingface.co/papers/2503.11576

4. **ArXiv HTML 버전**: https://arxiv.org/html/2503.11576v1

5. **IDP Software 분석**: https://idp-software.com/research/2025-03-14-SmolDocling/

6. **AI Models FYI 분석**: https://aimodels.fyi/papers/arxiv/smoldocling-ultra-compact-vision-language-model-end

7. **Medium (Ritvik Rastogi)**: *Papers Explained 333: SmolDocling*. https://ritvik19.medium.com/papers-explained-333-smoldocling-a788ac739b92

8. **Medium (Harisudhan S)**: *SmolDocling: A Compact Vision Language Model*. https://medium.com/@speaktoharisudhan/smoldocling-a-compact-vision-language-model-c54795474faf

9. **Analytics Vidhya**: https://www.analyticsvidhya.com/blog/2025/03/smoldocling/

10. **AI Innovations and Insights (Substack)**: https://aiexpjourney.substack.com/p/ai-innovations-and-insights-34-smoldocling

11. **MarkTechPost**: https://www.marktechpost.com/2025/03/18/ibm-and-hugging-face-researchers-release-smoldocling-a-256m-open-source-vision-language-model-for-complete-document-ocr/

12. **Docling Technical Report** (관련 선행 연구): Livathinos, N. et al. (2025). *Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion*. arXiv:2501.17887.

> ⚠️ **정확도 안내**: 본 답변에서 수식(예: 손실 함수, 아키텍처 수식)의 일부는 논문의 일반적인 VLM 설계 원리 및 관련 연구를 바탕으로 표준 형태로 재구성되었습니다. 논문 원문에서 해당 수식이 정확히 어떻게 표기되었는지는 위의 arXiv 원문을 직접 확인하시기 바랍니다.
