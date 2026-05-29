
# PDFMathTranslate: Scientific Document Translation Preserving Layouts

> **논문 정보**
> - **제목**: PDFMathTranslate: Scientific Document Translation Preserving Layouts
> - **저자**: Rongxin Ouyang, Chang Chu, Zhikuang Xin, Xiangyao Ma
> - **소속**: National University of Singapore / Tsinghua University / University of Chinese Academy of Sciences
> - **발표**: EMNLP 2025 System Demonstrations (pp. 918–924)
> - **arXiv**: [2507.03009](https://arxiv.org/abs/2507.03009) (v4, 2025.09.22)
> - **GitHub**: https://github.com/byaidu/pdfmathtranslate

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

과학 문서의 언어 장벽은 과학 기술의 확산과 발전을 저해하는데, 기존의 번역 시도들은 문서 레이아웃 정보를 대부분 무시해왔다. 이를 해결하기 위해 레이아웃을 보존하며 과학 문서를 번역하는 최초의 오픈소스 소프트웨어인 PDFMathTranslate를 제안하며, 대형 언어 모델(LLM)과 정밀 레이아웃 감지 기술의 최신 발전을 활용하여 **정밀성(Precision)**, **유연성(Flexibility)**, **효율성(Efficiency)** 에서 주요 개선을 달성했다.

### 1.2 주요 기여 5가지

다섯 가지 핵심 기여를 제시한다: **(1)** 레이아웃 감지, 번역, 재렌더링의 효율적인 워크플로우; **(2)** 다국어 지원; **(3)** 복수의 번역 모델 및 서비스 지원; **(4)** 다양한 사용자 인터페이스; **(5)** 지속 가능한 개발을 위한 커뮤니티-커머스 모델.

해당 연구는 https://github.com/byaidu/pdfmathtranslate 에 오픈소스로 공개되어 22만 2천 회 이상의 다운로드를 기록했다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

과학·기술 문서에서 레이아웃과 비텍스트 요소는 무시할 수 없는데, 단락 배치, 수식, 표, 그림의 구성은 풍부하고 중요한 의미를 지닌다. 기존의 텍스트 기반 번역 시도들은 레이아웃 정보를 무시함으로써 과학·기술 문서 번역의 장벽을 해결하기에 불충분하다.

### 2.2 제안하는 방법 및 아키텍처 (5단계 파이프라인)

PDFMathTranslate는 원본 레이아웃을 보존하며 문서를 번역하도록 설계되었다. 먼저 언어 및 번역 서비스 등 사용자 지정 파라미터와 함께 PDF 문서를 입력받는다. 다음으로 문서에서 레이아웃과 텍스트 내용을 추출하기 위해 레이아웃을 감지하고, GPT-4, DeepL, Google, Ollama 등 선택된 번역 서비스로 텍스트를 번역한다. 마지막으로 번역된 텍스트와 앞서 감지된 레이아웃을 합쳐 레이아웃이 보존된 번역 문서로 재렌더링한다.

전체 파이프라인은 다음 5단계로 구성된다:

$$\text{File Reading} \rightarrow \text{Layout Parsing} \rightarrow \text{Streamlining} \rightarrow \text{Content Translation} \rightarrow \text{Re-rendering}$$

시스템 아키텍처는 **정밀성(Precision)**, **유연성(Flexibility)**, **효율성(Efficiency)** 의 세 가지 원칙으로 설계되었으며, 각 원칙은 정밀한 파서, 유연한 번역 미들웨어, 효율적인 워크플로우로 구현된다.

---

#### 2.2.1 정밀 레이아웃 파서 (Precise Layout Parser)

레이아웃 보존을 위해 정밀한 레이아웃 추출이 필요하다. 문서 요소의 위치를 정확하게 파싱하기 위해 레이아웃 감지, 분할, 처리, 재렌더링으로 구성된 파이프라인을 제안한다. 핵심 레이아웃 감지 모델로 **DocLayout-YOLO-DocStructBench-onnx** 를 활용하며, 이는 객체 탐지의 SOTA 솔루션인 **YOLOv10**(Wang et al., 2024)의 변형 버전으로 해당 태스크에서 빠르고 정확하다.

레이아웃 감지 모델의 동작을 수식으로 표현하면:

$$\hat{B}, \hat{C} = \text{DocLayout-YOLO}(I_{PDF})$$

여기서 $\hat{B}$는 검출된 바운딩 박스(Bounding Box) 집합, $\hat{C}$는 각 요소의 클래스(문단, 수식, 표, 그림 등), $I_{PDF}$는 PDF 페이지 이미지를 의미한다.

모델 호환성을 높이기 위해 **ONNX(Open Neural Network Exchange)** 표준과 **PyTorch** 버전의 두 가지 모델 형식을 지원하며, ONNX 버전이 다양한 하드웨어에서의 파싱 파이프라인 호환성을 보장하기 위해 기본값으로 선택되었다.

---

#### 2.2.2 유연한 번역 미들웨어 (Flexible Translation Middleware)

번역 미들웨어는 **언어 다양성**, **지원 서비스**, **커스터마이징 능력** 측면에서 유연성을 제공한다. PDFMathTranslate는 체로키어(Cherokee) 같은 소수 언어를 포함해 최소 **56개 언어** 를 지원하며, 사용자는 번역 시 이 언어들을 입력 또는 출력 언어로 자유롭게 선택할 수 있다.

OpenAI 유사 서비스를 위한 범용 프레임워크로 설계된 프로토콜을 통해 로컬 호스팅 모델 또는 개인 온라인 서비스를 포함한 모든 번역 시스템을 통합할 수 있어, 광범위한 언어 지원을 보장한다.

번역 미들웨어의 수식 보존 전략은 다음과 같이 표현할 수 있다:

$$T_{\text{out}} = \text{Translate}\left(\mathcal{S}_{\text{text}}\right), \quad \mathcal{S}_{\text{math}} \xrightarrow{\text{skip}} \mathcal{S}_{\text{math}}$$

즉, 텍스트 세그먼트 $\mathcal{S}\_{\text{text}}$만 번역 모델에 전달하고, 수식 세그먼트 $\mathcal{S}_{\text{math}}$는 그대로 보존(skip)한다.

---

#### 2.2.3 다양한 사용자 인터페이스

CLI(명령줄 인터페이스), GUI(그래픽 사용자 인터페이스), Docker 배포, Zotero 플러그인을 제공하여 연구자들의 빠른 번역 및 독서 요구를 충족한다.

---

#### 2.2.4 출력 형식

번역 결과로 번역본(`paper-mono.pdf`)과 원문·번역문 병렬 이중 언어 파일(`paper-dual.pdf`)을 생성하여 현재 디렉터리에 저장한다.

---

### 2.3 성능 향상

ONNX 버전이 기본값으로 선택되어 다양한 하드웨어에서 파싱 파이프라인의 호환성을 보장한다.

포크 버전(PDFMathTranslate-next)은 다수의 엣지 케이스를 처리하고, PDF 호환성을 개선하며, 열 간(cross-column) 및 페이지 간(cross-page) 의미 일관성, 동적 스케일링 일관성 등 다양한 번역 품질 향상을 최적화했다.

---

### 2.4 한계점

광범위하게 사용되는 유연한 도구로서, 저작권과 관련된 잠재적인 윤리적 우려가 제기된다.

특정 지역 사용자들은 AI 모델 로딩 시 네트워크 어려움을 겪을 수 있으며, 현재 프로그램은 DocLayout-YOLO-DocStructBench-onnx AI 모델에 의존하기 때문에 일부 사용자는 네트워크 문제로 다운로드가 불가능하다.

또한 공개된 논문 및 GitHub 정보를 기반으로 확인 가능한 추가 한계는 다음과 같다:

| 한계 유형 | 내용 |
|---|---|
| 복잡한 레이아웃 | 다단 논문의 교차 열(cross-column) 수식 처리 정확도 한계 |
| 스캔 PDF | 텍스트 레이어가 없는 스캔 이미지 기반 PDF 지원 제한 |
| 저작권 | 번역 과정에서 원저작물 저작권 침해 가능성 |
| 네트워크 의존성 | 외부 번역 서비스(Google, DeepL, OpenAI) API 의존성 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 다중 번역 서비스 통합을 통한 일반화

OpenAI 유사 서비스를 위한 범용 프레임워크로 설계된 프로토콜은 로컬 호스팅 모델 또는 개인 온라인 서비스를 포함한 어떤 번역 시스템도 통합할 수 있도록 하며, 서비스와 모델의 확장성으로 인해 실제 지원 언어 수는 훨씬 더 많을 수 있다.

이를 통해:
- 특정 LLM 모델에 대한 과의존성을 탈피 → **모델 불가지론적(Model-Agnostic) 일반화** 달성
- 새로운 번역 서비스가 출시되어도 프레임워크 변경 없이 플러그인 형태로 통합 가능

### 3.2 레이아웃 감지 모델의 일반화

최신 레이아웃 감지 기술인 DocLayout-YOLO-DocStructBench-onnx를 활용하며, 이는 객체 감지의 SOTA 솔루션인 YOLOv10의 변형으로 해당 태스크에서 빠르고 정확하다.

DocLayout-YOLO 기반 레이아웃 감지의 일반화 능력은 다음 수식으로 표현 가능하다:

$$\text{mAP} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \text{AP}_c, \quad \mathcal{C} = \{\text{text}, \text{formula}, \text{table}, \text{figure}, \ldots\}$$

여기서 $\mathcal{C}$는 문서 요소 클래스 집합이며, 다양한 문서 도메인에서 mAP를 높게 유지하는 것이 일반화 성능의 핵심이다.

이 모델의 호환성을 높이기 위해 ONNX 표준과 PyTorch 버전의 두 가지 모델 형식을 지원하며, ONNX 버전이 기본값으로 선택되어 다양한 하드웨어 환경에서의 파싱 파이프라인 호환성을 보장한다.

### 3.3 다국어 일반화

번역 미들웨어는 언어 다양성, 지원 서비스, 커스터마이징 능력 측면에서 유연성을 제공하며, 체로키어(Cherokee)를 포함한 소수 언어까지 최소 56개 언어를 지원한다.

$$\text{Language Coverage} = \{L_i\}_{i=1}^{N}, \quad N \geq 56$$

이는 저자원 언어(Low-Resource Language)에서의 일반화 가능성을 높이는 중요한 설계 요소다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 구분 | PDFMathTranslate (2025) | MathTranslate (2023~) | 기존 번역 도구 (Google 등) |
|---|---|---|---|
| **입력 형식** | PDF 직접 처리 | LaTeX 소스 코드 | 텍스트 복사 붙여넣기 |
| **레이아웃 보존** | ✅ (DocLayout-YOLO) | ✅ (LaTeX 재컴파일) | ❌ |
| **수식 보존** | ✅ (감지 후 skip) | ✅ (LaTeX 환경 내) | ❌ |
| **번역 모델** | GPT-4, DeepL, Google, Ollama 등 다중 | DeepL, Google 등 | 고정 단일 서비스 |
| **일반화 범위** | 범용 PDF 문서 | arXiv LaTeX 위주 | 텍스트 전용 |
| **오픈소스** | ✅ | ✅ | ❌ |

MathTranslate는 LaTeX 문서를 임의의 언어에서 임의의 언어로 번역하는 프로젝트로, LaTeX 수식 표현식은 완벽하게 그대로 유지되며, 컴파일하면 최종 PDF 파일이 생성된다. 그러나 MathTranslate는 LaTeX 소스 코드가 있는 경우에만 동작하므로 PDFMathTranslate에 비해 **접근성과 일반화 범위**가 제한된다.

PDFMathTranslate와 관련하여 DocLayout-YOLO는 문서별 특화 최적화를 통해 속도 이점을 유지하면서 정확도를 향상시키는 새로운 접근 방식을 제시하는 연구다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 과학 정보 민주화 (Democratization of Scientific Knowledge)
과학 문서의 언어 장벽은 과학 기술의 확산과 발전을 저해하는데, PDFMathTranslate는 LLM과 정밀 레이아웃 감지의 최신 발전을 활용하여 이 장벽을 해결한다. 이는 비영어권 연구자들의 첨단 연구 접근성을 획기적으로 높여 글로벌 과학 생태계의 포용성을 강화한다.

#### (2) 멀티모달 문서 이해 연구의 촉진
레이아웃 감지 + LLM 기반 번역의 결합 파이프라인은 다음 분야의 연구를 촉진한다:
- 문서 구조 인식(Document Structure Understanding)
- 수식 인식(Mathematical Expression Recognition, MER)
- 시각-언어 모델(Vision-Language Model, VLM)의 문서 특화 응용

$$\text{VLM}_{doc} = f(\text{Visual Features}_{\text{layout}} \oplus \text{Linguistic Features}_{\text{text}})$$

#### (3) 표준화된 레이아웃 보존 번역 벤치마크 필요성 제기
현재 이 분야에는 레이아웃 보존 품질을 정량화하는 공인된 평가 지표가 부재하며, 본 연구는 향후 다음과 같은 벤치마크 개발을 촉진할 것으로 예상된다:

$$\text{LPS} = \alpha \cdot \text{BLEU}_{text} + \beta \cdot \text{IoU}_{layout} + \gamma \cdot \text{Acc}_{formula}$$

여기서 $\text{LPS}$(Layout-Preserving Score), $\text{IoU}\_{layout}$은 레이아웃 박스의 교차비율, $\text{Acc}_{formula}$는 수식 보존 정확도다.

---

### 5.2 앞으로 연구 시 고려할 점

#### (1) 레이아웃 감지 모델의 도메인 일반화
DocLayout-YOLO 모델을 활용하여 수식과 차트의 정확한 배치를 보장하지만, 의료 문서, 법률 문서, 특수 학술지 포맷 등 훈련 데이터 분포 밖의 문서에 대한 일반화 성능을 검증하고 보완하는 연구가 필요하다.

#### (2) 저자원 언어(Low-Resource Language) 품질 검증
체로키어(Cherokee)와 같은 소수 언어를 지원하지만, 실제 번역 품질은 LLM 백엔드의 해당 언어 학습 데이터 양에 크게 의존하므로 저자원 언어에 대한 번역 품질 평가 체계 수립이 필요하다.

#### (3) 수식 맥락(Context) 인식 번역
현재 수식은 번역에서 제외(skip)되나, 수식 주변 텍스트와의 맥락적 일관성을 보장하는 연구가 필요하다:

$$P(\hat{T} \mid \mathcal{S}_{text}, \mathcal{S}_{math}) \neq P(\hat{T} \mid \mathcal{S}_{text})$$

수식 $\mathcal{S}_{math}$의 의미를 번역 컨텍스트에 반영하는 조건부 번역 모델 개발이 중요한 과제다.

#### (4) 저작권 및 윤리 문제 해결
광범위하게 사용되는 유연한 도구로서, 저작권과 관련된 잠재적인 윤리적 우려가 제기된다. 향후 연구에서는 번역 도구의 공정 이용(Fair Use) 범위 정의와 저작권 준수 메커니즘 설계를 반드시 고려해야 한다.

#### (5) 스캔 문서 및 비정형 PDF 지원
현재 파이프라인은 텍스트 레이어가 포함된 PDF에 최적화되어 있으므로, OCR 모듈과의 통합을 통한 스캔 문서 일반화가 필요하다:

$$\text{Pipeline}_{extended} = \text{OCR} \rightarrow \text{Layout Parsing} \rightarrow \text{Translation} \rightarrow \text{Re-rendering}$$

#### (6) 번역 품질 자동 평가 지표 개발
기존 기계 번역 지표(BLEU, METEOR, COMET)는 레이아웃 보존 측면을 반영하지 못하므로, **레이아웃-인식 번역 품질 지표(Layout-Aware Translation Quality Metric)**의 개발이 향후 핵심 연구 과제다.

---

## ⚠️ 정보 신뢰도 고지

이 논문은 2025년 7월 arXiv에 게재되고 EMNLP 2025 System Demonstrations에 채택된 최신 논문으로, 전체 실험 수치(정량적 BLEU/mAP 등)의 세부 내용은 논문 PDF 원문에서 직접 확인하시기를 권장합니다. 본 분석에서 제시된 수식 중 일부(LPS 벤치마크 등)는 논문의 아이디어를 바탕으로 필자가 개념적으로 구성한 것임을 밝힙니다.

---

## 📚 참고 문헌 및 출처

1. **Ouyang, R., Chu, C., Xin, Z., & Ma, X. (2025)**. *PDFMathTranslate: Scientific Document Translation Preserving Layouts*. EMNLP 2025 System Demonstrations, pp. 918–924. https://aclanthology.org/2025.emnlp-demos.71/
2. **arXiv 원문**: https://arxiv.org/abs/2507.03009 (v4, Sep. 22, 2025)
3. **arXiv PDF**: https://arxiv.org/pdf/2507.03009
4. **GitHub 공식 저장소**: https://github.com/byaidu/pdfmathtranslate
5. **HuggingFace Papers**: https://huggingface.co/papers/2507.03009
6. **Semantic Scholar**: https://www.semanticscholar.org/paper/PDFMathTranslate:-Scientific-Document-Translation-Ouyang-Chu/f391bb87746a47d4174001c9b69c4cbb6052e45d
7. **ResearchGate**: https://www.researchgate.net/publication/393478760
8. **AI Sharing Circle 소개**: https://aisharenet.com/en/pdfmathtranslate/
9. **MathTranslate (비교 대상)**: https://github.com/SUSYUSTC/MathTranslate
10. **Wang et al. (2024)**, *YOLOv10: Real-Time End-to-End Object Detection*, arXiv:2405.14458
11. **Zhao et al. (2024)**, *DocLayout-YOLO: Enhancing Document Layout Analysis*, (PDFMathTranslate 내 인용)
