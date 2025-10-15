# SynCoBERT: Syntax-Guided Multi-Modal Contrastive Pre-Training for Code Representation

**핵심 주장 및 주요 기여 요약**  
SynCoBERT는 프로그래밍 언어의 *심볼릭(symbolic)*·*구문적(syntactic)* 특성을 모두 활용하여, 코드·주석·추상구문트리(AST)라는 여러 모달리티를 결합한 **다중 모달 대조 학습(multimodal contrastive learning)** 프레임워크를 제안한다.  
1. **Identifier Prediction (IP)**: 각 코드 토큰이 식별자(identifier)인지 판별하는 이진 분류 목표 도입.  
2. **AST Edge Prediction (TEP)**: AST 노드 간 실제 엣지 유무를 예측함으로써 구문 구조 정보 통합.  
3. **Multi-Modal Contrastive Learning (MCL)**: 코드·주석·AST 간 상호 정보(mutual information)를 극대화하여 보다 포괄적 코드 표현 학습.  
4. **성능 향상**: 동등한 규모·코퍼스 조건에서 CodeBERT·GraphCodeBERT를 모든 주요 코드 이해·생성 벤치마크에서 평균 2–5% 이상 상회.[1]

***

## 1. 해결 과제  
기존 프리트레인 모델들은 코드 토큰을 단순 시퀀스로 처리하거나 AST·데이터 흐름을 부분 활용했으나,  
- **식별자**에 담긴 기호적·구문적 정보  
- **AST 엣지**가 표현하는 구조적 구문 정보  
- 코드·주석·AST 간의 **모달리티 간 상호관계**  
를 충분히 탐색하지 못했다.[1]

***

## 2. 제안 방법  
### 2.1 입력 및 모델 구조  
- 입력: `[CLS]` + NL(comment) + `[SEP]` + PL(code) + `[SEP]` + AST(depth-first 순회) + `[SEP]`  
- Transformer 인코더: 12-layer, hidden size 768, 12 attention heads.[1]

### 2.2 사전학습 목표  
1. **Multi-Modal Masked Language Modeling (MMLM)**  
   - NL·PL·AST 통합 시퀀스에서 15% 토큰 마스킹 후 교차 엔트로피로 복원.  
2. **Identifier Prediction (IP)**  
   - PL 토큰마다 식별자 여부 $$y_i^{\text{IP}}\in\{0,1\}$$ 예측.  
   - 손실:  

$$
       L_{\mathrm{IP}}
       = -\sum_i\bigl[y_i^{\mathrm{IP}}\ln p_i^{\mathrm{IP}}+(1-y_i^{\mathrm{IP}})\ln(1-p_i^{\mathrm{IP}})\bigr]
     $$

3. **AST Edge Prediction (TEP)**  
   - AST 노드 쌍 $$(i,j)$$ 간 엣지 존재 여부 $$y_{i,j}^{\mathrm{TEP}}$$ 예측.  
   - 손실:  

$$
       L_{\mathrm{TEP}}
       = -\sum_{i,j}\bigl[y_{i,j}^{\mathrm{TEP}}\ln p_{i,j}^{\mathrm{TEP}}+(1-y_{i,j}^{\mathrm{TEP}})\ln(1-p_{i,j}^{\mathrm{TEP}})\bigr]
     $$

4. **Multi-Modal Contrastive Learning (MCL)**  
   - 쌍별 긍정 샘플(예: NL vs. {PL+AST}, PL-AST vs. AST-PL 등)과 배치 내·간 부정 샘플을 구축.  
   - 각 벡터 $$v$$간 dot-product 유사도로 대조적 학습:  

$$
       \ell_{i} = -\ln\frac{\exp(v_i\cdot v_i^+)}{\exp(v_i\cdot v_i^+)+\sum_{k\neq i}\exp(v_i\cdot v_k)}.
     $$

최종 손실:  

$$
  L = L_{\mathrm{MMLM}} + L_{\mathrm{IP}} + L_{\mathrm{TEP}} + L_{\mathrm{MCL}} + \lambda\|\theta\|_2^2.
$$

[1]

***

## 3. 성능 향상  
동일한 CodeSearchNet 프리트레이닝 코퍼스·모델 크기에서, SynCoBERT는 CodeBERT·GraphCodeBERT 대비:  
- **코드 검색**: 평균 MRR +4.7–10.9점.[1]
- **클론 탐지**: F1 +1.2–3.6점, MAP +3.1–5.5점.[1]
- **버그 검출**: 정확도 +1.3–2.4%p.[1]
- **코드 변환(C→Java)**: BLEU +4.4점, EM +2.5%p, CodeBLEU +2.8점.[1]

***

## 4. 모델 한계  
- **학습 비용**: 8×V100 GPU 80시간 소요.  
- **시퀀스 길이 제약**: NL·PL·AST 각각 최대 길이 고정(96/160/256)  
- **다양한 언어 미포함**: C 언어 등 비지원 언어에 대한 전이 실험은 제한적임.

***

## 5. 일반화 성능 향상 근거  
- **모달리티 간 상호 정보**를 학습함으로써 특정 언어·도메인 편향 완화.  
- **AST 엣지·식별자 정보**의 직접 예측을 통해 구문·심볼릭 패턴을 구조적으로 내재화.  
- C 언어 비사전학습에도 Java 기반 지식이 전이되어 Clang 오류 검출·변환 성능 개선 확인.[1]

***

## 6. 향후 영향 및 고려 사항  
- **다른 코드 모달리티**(예: 데이터 흐름 그래프, 제어 흐름) 통합 연구.  
- **경량화·효율화**: 지식 증류·모델 프루닝을 통한 실제 개발 환경 적용.  
- **동적 분석 기반 멀티모달 학습**: 런타임 행동 정보와 결합하여 더욱 정교한 표현 학습.  
- **언어 확장성 평가**: C·JavaScript 등 비리치 말뭉치 언어로의 일반화 실험 필수.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/eca4c893-49b7-44e1-a408-6e042b9f7e33/2108.04556v3.pdf)
