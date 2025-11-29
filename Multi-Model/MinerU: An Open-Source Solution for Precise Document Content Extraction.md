
# MinerU: An Open-Source Solution for Precise Document Content Extraction

## 1. 핵심 주장 및 주요 기여

MinerU는 **다양한 문서 유형에 대한 고정확도 콘텐츠 추출**을 목표로 하는 포괄적인 오픈소스 솔루션으로, 기존 OCR, 레이아웃 감지, 공식 인식 방법들이 문서 다양성으로 인해 고품질 추출에 실패하는 문제를 해결합니다.[1]

주요 기여는 다음과 같습니다:

**① 모듈식 파이프라인 기반 접근**: MinerU는 PDF-Extract-Kit 모델 라이브러리를 활용하여 다양한 실제 문서에서 효과적인 콘텐츠 추출을 달성합니다. 멀티모달 대규모 언어 모델(MLLM)의 높은 추론 비용 대신, 모듈식 접근으로 효율성과 정확도를 균형있게 유지합니다.[1]

**② 데이터 엔지니어링 기반 모델 개선**: 다양한 문서 유형(논문, 교과서, 시험지, 잡지, PPT, 보고서)에 대해 세심한 데이터 선택, 주석 작성, 반복적 모델 학습을 통해 모델의 강건성을 증대시킵니다.[1]

**③ 정교한 후처리 규칙**: 레이아웃 분석, 공식 감지, 공식 인식, 표 인식, OCR 결과의 중첩 제거, 내용 정렬, 단락 인식 등의 세밀한 규칙 기반 후처리로 최종 결과의 정확성을 보장합니다.[1]

***

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 기존 문서 추출 방식의 한계

MinerU가 다루는 네 가지 주요 기술적 접근과 그들의 문제점은 다음과 같습니다:[1]

| 기술 방식 | 설명 | 문제점 |
|---------|------|--------|
| **OCR 기반 텍스트 추출** | OCR 모델을 직접 사용 | 이미지, 표, 공식이 포함된 문서에서 노이즈 증가 |
| **라이브러리 기반 텍스트 파싱** | PyMuPDF 등으로 직접 읽기 | 공식, 표 등의 복잡한 요소 처리 불가 |
| **멀티모듈 문서 파싱** | 레이아웃 감지 → 다양한 인식기 적용 | 학술 논문에만 최적화, 교과서/시험지 등 다양한 문서에서 성능 저하 |
| **엔드-투-엔드 MLLM** | Donut, Nougat 등 사용 | 높은 추론 비용, 데이터 다양성 문제로 일반화 어려움 |

### 2.2 MinerU의 처리 파이프라인

MinerU는 네 단계의 순차적 처리로 구성됩니다:[1]

**1단계: 문서 전처리**
- PyMuPDF를 이용한 PDF 읽기 및 필터링
- 언어 식별 (중국어, 영어), 암호화 문서 감지, 스캔 PDF 식별
- 페이지 메타데이터 추출 (페이지 수, 크기, 차원)

**2단계: 문서 콘텐츠 파싱**
- **레이아웃 분석**: 제목, 본문, 이미지, 표, 공식 등 다양한 요소 영역 구분
- **공식 감지**: 인라인 공식과 표시된 공식 구별 (약 24,157개 인라인 공식, 1,829개 표시 공식으로 학습)
- **공식 인식**: UniMERNet 모델로 다양한 공식 유형(SPE, CPE, SCE, HWE) 인식
- **표 인식**: TableMaster와 StructEqTable 모델 활용
- **OCR**: Paddle-OCR로 텍스트 인식 (레이아웃 기반 OCR로 읽기 순서 보존)

**3단계: 문서 콘텐츠 후처리**
- 경계 박스(Bounding Box) 겹침 해결
- 포함 관계 처리 (이미지/표 내 텍스트 제거)
- 부분 겹침 처리 및 텍스트 무결성 보존
- 인간의 읽기 순서(위→아래, 왼쪽→오른쪽) 기반 정렬

**4단계: 형식 변환**
- 중간 JSON 구조에서 Markdown 또는 JSON 형식으로 변환

### 2.3 핵심 기술 및 수식

**공식 감지 모델**: YOLO 기반 모델이 속도와 정확도의 균형을 유지합니다.[1]

$$\text{Detection Loss} = L_{\text{localization}} + L_{\text{classification}}$$

여기서 레이아웃 감지는 LayoutLMv3 기반 미세조정을 통해 구현됩니다.[1]

**표 인식**: 두 가지 접근이 사용됩니다.

- **TableMaster**: 네 가지 부분작업으로 분해
  - 표 구조 인식
  - 텍스트 라인 감지
  - 텍스트 라인 인식
  - 박스 할당

- **StructEqTable**: 엔드-투-엔드 방식으로 복잡한 표에서도 우수한 성능 제공[1]

**OCR 최적화**: 공식이 포함된 텍스트 블록의 경우:

$$\text{OCR Result} = \text{OCR}(\text{Image with Masked Formulas}) + \text{Formulas}$$

공식 감지 좌표로 공식을 마스킹한 후 OCR 수행, 최후에 공식을 다시 삽입합니다.[1]

***

## 3. 모델 구조 및 성능

### 3.1 핵심 모듈 구조

**레이아웃 감지 모델** (Layout Detection):
- 학습 데이터: 21,000개 주석 데이터 포인트
- 분류 범주: 제목, 본문, 이미지, 이미지 캡션, 표, 표 캡션, 인라인 공식, 표시 공식, 제외 유형(헤더, 푸터 등)
- 데이터 엔지니어링 전략: 클러스터링 기반 다양한 데이터 선택, 검증 세트 기반 반복적 데이터 선택으로 저성능 카테고리에 샘플링 가중치 증가[1]

**공식 감지 모델** (Formula Detection):
- 학습 데이터: 2,890 페이지에서 24,157개 인라인 공식, 1,829개 표시 공식
- 분류 범주: 인라인 공식, 표시 공식, 무시 클래스(불확실한 영역)
- YOLO 기반으로 속도와 정확도의 우수한 균형 달성[1]

**공식 인식 모델** (Formula Recognition):
- 자체 개발 UniMERNet 모델 활용
- 다양한 공식 유형에서 상용 소프트웨어 MathPix에 필적하는 성능[1]

### 3.2 성능 평가 결과

**레이아웃 감지 성능**:

| 모델 | 학술 논문 | 교과서 |
|-----|---------|--------|
| | mAP | AP50 | AR50 | mAP | AP50 | AR50 |
| DocXchain | 52.8 | 69.5 | 77.3 | 34.9 | 50.1 | 63.5 |
| Surya | 24.2 | 39.4 | 66.1 | 13.9 | 23.3 | 49.9 |
| 360LayoutAnalysis-Paper | 37.7 | 53.6 | 59.8 | 20.7 | 31.3 | 43.6 |
| LayoutLMv3-Finetuned (MinerU) | **77.6** | **93.3** | **95.5** | **67.9** | **82.7** | **87.9** |

다양한 데이터에 미세조정된 모델이 기존 오픈소스 모델을 현저히 능가합니다.[1]

**공식 감지 성능**:

| 모델 | 학술 논문 | 다중 출처 |
|-----|---------|---------|
| | AP50 | AR50 | AP50 | AR50 |
| Pix2Text-MFD | 60.1 | 64.6 | 58.9 | 62.8 |
| YOLOv8-Finetuned (MinerU) | **87.7** | **89.9** | **82.4** | **87.3** |

MinerU의 공식 감지 모델이 다양한 출처 데이터에서도 우수한 성능을 유지합니다.[1]

**공식 인식 성능** (UniMER-Test 데이터셋):

| 모델 | ExpRate | CDM | BLEU | CDM BLEU |
|-----|---------|-----|------|----------|
| Pix2tex | 0.1237 | 0.291 | 0.4080 | 0.636 |
| Texify | 0.2288 | 0.495 | 0.5890 | 0.755 |
| Mathpix | 0.2610 | 0.5 | 0.8067 | 0.951 |
| **UniMERNet** | **0.4799** | **0.811** | **0.8425** | **0.968** |

여기서 CDM 메트릭은 공식 표현의 다양성에 영향을 받지 않으므로 더 신뢰성 있는 평가 지표입니다.[1]

***

## 4. 일반화 성능 향상과 한계

### 4.1 일반화 성능 향상 가능성

**① 다양한 데이터 기반 학습**:
MinerU는 11가지 문서 유형에 대한 평가 데이터셋을 구성하여 다양성 보장합니다. 학술 논문뿐 아니라 교과서, 시험지, 슬라이드, 재무 보고서, 이미지-텍스트 교과서, 특수 이미지-텍스트 시험지, 역사 문서, 필기노트, 표준 도서 등을 포함합니다.[1]

레이아웃 감지 성능 비교에서 보듯이, 다양한 데이터로 학습한 MinerU의 모델은:
- 학술 논문에서 77.6 mAP (기존 52.8)
- 교과서에서 67.9 mAP (기존 34.9)

를 달성하여 **특히 다양한 문서에서의 일반화 성능이 크게 향상**됩니다.[1]

**② 반복적 데이터 선택 전략**:
검증 세트의 성능 결과를 기반으로 저성능 카테고리에 더 많은 샘플을 할당하는 동적 학습 전략으로 모델 개선을 지속합니다. 이는 특정 카테고리나 문서 유형에 대한 편향을 감소시킵니다.[1]

**③ 멀티태스크 학습의 시너지**:
최신 연구(2025)의 DocFusion 모델은 Gaussian-Kernel Cross-Entropy Loss(GK-CEL)를 제안하여 여러 문서 파싱 작업(레이아웃 감지, OCR, 수학 공식 인식, 표 인식)을 통합합니다. 실험 결과 OCR 작업 추가로 레이아웃 감지 성능이 1.3% 향상되었고, 표 인식에서는 2.1% F1 점수 증가를 보였습니다. 이는 MinerU의 멀티모듈 접근이 각 모듈 간 정보 공유를 통해 더욱 향상될 가능성을 시사합니다.[2]

**④ 레이아웃 인식 기반 임베딩**:
DocLLM(2024)의 사례처럼, 바운딩 박스 정보와 레이아웃 토큰을 명시적으로 모델에 통합하면 일반화 성능이 크게 향상됩니다. DocLLM은 보지 못한 문서 유형 5개 중 4개에서 우수한 성능을 보였습니다. MinerU의 후처리 단계에서 이러한 레이아웃 정보를 보다 체계적으로 활용하면 일반화 성능을 더욱 개선할 수 있습니다.[3]

### 4.2 한계 및 개선 필요 영역

**① 언어 제한**:
현재 MinerU는 중국어와 영어만 지원합니다. 다른 언어(한국어, 일본어, 아랍어 등)의 문서 처리 품질은 보장되지 않습니다.[1]

**② 특수 레이아웃 처리**:
- 세로 쓰기 및 우에서 좌로 읽는 방식의 문서(역사 문서)
- 복잡한 다단계 배치(nested layouts)
- 혼합된 단일/이중 열 레이아웃

이들은 여전히 처리에 어려움이 있습니다.[1]

**③ 모듈 간 최적화의 한계**:
각 모듈이 독립적으로 학습되므로 전역 최적화가 불가능하며, 한 모듈의 오류가 다음 모듈에 누적될 수 있습니다. 반면 최신 엔드-투-엔드 모델들은 다양한 작업에 대해 joint optimization을 수행합니다.[2]

**④ 추론 속도와 효율성**:
여러 모듈을 순차적으로 실행하므로 추론 시간이 늘어날 수 있습니다. 병렬 처리 최적화가 필요합니다.[1]

**⑤ 공식 및 표의 복잡도**:
- 수기 공식, 스캔 문서의 노이즈 있는 공식
- 복잡한 병합 셀을 가진 표
- 가로 표와 혼합된 텍스트

이들은 여전히 인식 오류를 일으킬 수 있습니다.[1]

***

## 5. 최신 연구 동향 및 미래 연향력

### 5.1 문서 파싱 분야의 최신 연구 트렌드 (2024-2025)

**① 통합된 다중 작업 프레임워크**:
DocFusion(2025)은 레이아웃 감지, OCR, 수학 공식 인식(MER), 표 인식(TR)을 단일 모델로 통합하였으며, Gaussian-Kernel Cross-Entropy Loss로 이산 토큰과 연속 좌표 간 불일치를 해결합니다.[2]

**② 효율적인 멀티모달 LLM 확장**:
DocLayLLM(2025)은 경량 멀티모달 확장으로 높은 해상도 이미지 인코더를 피하면서도 페이지 내 다중 문서를 처리할 수 있습니다. 기존 문서 이해 데이터와 체인-오브-스루트(CoT) 주석을 활용한 코스 어닐링으로 효율성을 향상시켰습니다.[4]

**③ 포괄적 벤치마크 구축**:
OmniDocBench(2025)는 9가지 문서 유형(학술 논문, 교과서, 슬라이드 등)을 포함한 다층 평가 프레임워크를 제공하여 문서 파싱 모델의 공정한 비교 평가를 가능하게 합니다.[5][6]

**④ RAG 시스템과의 통합**:
검색 강화 생성(RAG) 기술과의 통합으로 문서 추출 품질이 LLM 훈련 데이터 생성에 직접 영향을 미치게 되었습니다. 이는 고정확도 문서 추출의 중요성을 증대시킵니다.[7]

**⑤ 도메인 간 일반화**:
최신 연구는 특정 도메인에 최적화된 모델보다는 **다양한 문서 유형에서 강건한 일반화 성능**을 추구합니다. 이는 MinerU의 철학과 일치합니다.[8]

### 5.2 MinerU가 미치는 영향 및 고려사항

**긍정적 영향**:
- **오픈소스 커뮤니티 기여**: 고성능의 모듈식 오픈소스 솔루션으로 많은 연구자와 실무자에게 기여
- **멀티모달 모델의 한계 극복**: 높은 비용의 엔드-투-엔드 MLLM 대신 효율적인 모듈식 접근 제시
- **데이터 다양성 강조**: 학술 논문 중심에서 벗어나 다양한 문서 유형을 지원하는 트렌드 선도

**향후 고려사항**:

1. **모듈 간 연계 강화**: 각 모듈의 신뢰도 점수를 포함하여 다음 모듈의 입력으로 활용하는 적응형 파이프라인 개발이 필요합니다.[9]

2. **경량 멀티모달 모델과의 하이브리드 접근**: 완전한 엔드-투-엔드 MLLM보다는 특정 작업(공식 인식, 표 인식)에 최적화된 경량 모델과의 결합으로 효율성과 정확도의 균형 달성[10]

3. **동적 벤치마크 구축**: OmniDocBench와 유사한 평가 프레임워크와 지속적 통합으로 새로운 문서 유형에 대한 성능 모니터링

4. **언어 확장**: 중국어, 영어 외 다국어 지원 추가로 국제적 활용도 증대

5. **설명 가능성 강화**: 최신 DocThinker(2025) 모델처럼 강화학습과 연쇄 추론을 통해 문서 이해 과정의 투명성과 신뢰성 향상[11]

6. **실시간 분석 기능**: 추론 속도 최적화와 스트리밍 방식의 문서 처리로 실시간 피드백 제공

***

## 6. 결론

MinerU는 **다양한 문서 유형에 대한 고정확도 추출**이라는 현실적 문제를 효율적으로 해결하는 중요한 솔루션입니다. 기존의 단일 모델 중심의 접근을 벗어나 세심한 데이터 엔지니어링과 모듈식 아키텍처의 조합으로 강건성과 효율성을 동시에 달성했습니다.

특히 **일반화 성능 향상** 측면에서:
- 다양한 문서 데이터를 기반으로 한 학습
- 레이아웃 감지에서 기존 대비 2배 이상의 성능 향상
- 다양한 문서 유형에서의 일관된 고성능

은 MinerU가 한계를 극복했음을 보여줍니다.

그러나 **모듈 간 최적화의 한계**, **언어 제한**, **특수 레이아웃 처리** 등의 개선 여지가 남아있습니다. 향후 연구는 DocFusion의 멀티태스크 학습, DocLayLLM의 경량 멀티모달 확장, OmniDocBench의 포괄적 평가 프레임워크 등 최신 기술과의 통합을 통해 MinerU의 적용성을 한층 높일 수 있을 것으로 예상됩니다.

***

## 참고문헌 체계

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/36243a05-3ede-4607-b345-de80db5eb9ff/2409.18839v1.pdf)
[2](https://aclanthology.org/2025.findings-acl.393.pdf)
[3](https://huggingface.co/papers/2401.00908)
[4](https://openreview.net/forum?id=gdKesR0A4h)
[5](https://openaccess.thecvf.com/content/CVPR2025/papers/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.pdf)
[6](https://arxiv.org/abs/2412.07626)
[7](https://arxiv.org/abs/2408.01287)
[8](https://arxiv.org/html/2410.21169v1)
[9](https://www.llamaindex.ai/blog/ai-document-parsing-llms-are-redefining-how-machines-read-and-understand-documents)
[10](https://aclanthology.org/2024.acl-long.463/)
[11](https://arxiv.org/abs/2508.08589)
[12](https://link.springer.com/10.1007/s11042-024-18248-2)
[13](https://ieeexplore.ieee.org/document/11069873/)
[14](https://ieeexplore.ieee.org/document/10860267/)
[15](http://ijarsct.co.in/Paper15699.pdf)
[16](https://dl.acm.org/doi/10.1145/3749369)
[17](https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/9804)
[18](https://journals.zuj.edu.jo/ijasca/PapersUploaded/2025.2.3.pdf)
[19](https://ieeexplore.ieee.org/document/11115689/)
[20](https://ijaem.net/issue_dcp/Fake%20Job%20Post%20Detection%20using%20Machine%20Learning%20and%20Deep%20Learning.pdf)
[21](http://arxiv.org/pdf/2309.10952.pdf)
[22](https://arxiv.org/pdf/2409.05137.pdf)
[23](https://arxiv.org/pdf/2401.02823.pdf)
[24](http://arxiv.org/pdf/2409.13717.pdf)
[25](https://arxiv.org/pdf/1812.07248.pdf)
[26](https://arxiv.org/abs/2409.18839)
[27](http://arxiv.org/pdf/2409.11282.pdf)
[28](https://arxiv.org/pdf/2207.06744.pdf)
[29](https://anyparser.com/blog/future-trends-in-document-parsing/)
[30](https://arxiv.org/abs/2408.06345)
[31](https://openaccess.thecvf.com/content/CVPR2025/papers/Liao_DocLayLLM_An_Efficient_Multi-modal_Extension_of_Large_Language_Models_for_CVPR_2025_paper.pdf)
[32](https://www.docsumo.com/blogs/data-extraction/best-software)
[33](https://g3lu.tistory.com/47)
[34](https://nanonets.com/blog/intelligent-data-extraction/)
[35](https://www.siliconflow.com/articles/en/best-multimodal-models-for-document-analysis)
[36](https://www.ssrn.com/abstract=5370857)
[37](https://aacrjournals.org/cebp/article/34/9_Supplement/A005/764900/Abstract-A005-Global-evidence-of-health-related)
[38](https://journal.yp3a.org/index.php/mukasi/article/view/4376)
[39](https://ojs.bonviewpress.com/index.php/AIA/article/view/3972)
[40](https://ijamjournal.org/ijam/publication/index.php/ijam/article/view/559)
[41](https://journals.gnpu.edu.ua/index.php/vgnpu/article/view/226)
[42](https://aacrjournals.org/cebp/article/34/9_Supplement/C062/764608/Abstract-C062-Barriers-to-end-of-life-discussions)
[43](https://www.nature.com/articles/s41598-025-26645-2)
[44](https://aclanthology.org/2021.codi-main.11)
[45](http://link.springer.com/10.1007/11562214)
[46](https://arxiv.org/html/2412.12505v1)
[47](http://arxiv.org/pdf/2410.21169.pdf)
[48](https://arxiv.org/abs/1906.02285)
[49](http://arxiv.org/pdf/2412.10207.pdf)
[50](https://www.aclweb.org/anthology/P18-1035.pdf)
[51](https://www.aclweb.org/anthology/P19-1443.pdf)
[52](https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00533/2067873/tacl_a_00533.pdf)
[53](https://huggingface.co/papers/2412.07626)
[54](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_Docopilot_Improving_Multimodal_Models_for_Document-Level_Understanding_CVPR_2025_paper.pdf)
[55](https://github.com/opendatalab/OmniDocBench)
[56](https://www.sciencedirect.com/science/article/pii/S2666764925000384)
[57](https://arxiv.org/abs/2401.00908)
[58](https://aclanthology.org/2025.findings-acl.1128/)
