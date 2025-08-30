# Automatic Chain of Thought Prompting in Large Language Models

# 핵심 요약

**핵심 주장**  
“Automatic Chain of Thought Prompting in Large Language Models” 논문은 수작업으로 설계된 체인 오브 사고(Chain-of-Thought, CoT) 데모 대신, 대형 언어 모델(LLM)에 의해 자동 생성된 추론 체인을 활용하여 Few-Shot prompting 성능을 Manual-CoT와 동등하거나 그 이상으로 이끌어낼 수 있음을 보인다.

**주요 기여**  
1. **Auto-CoT 패러다임 제안**:  
   – 질문 집합을 다양성 기반 클러스터링으로 분할.  
   – 각 클러스터에서 대표 질문을 선택해 Zero-Shot CoT(“Let’s think step by step”)로 추론 체인을 생성.  
   – 간단한 휴리스틱(질문 길이 ≤60 토큰, 추론 단계 ≤5)으로 불완전한 체인을 필터링.  
2. **다양성 기반 샘플링의 효과**:  
   – 유사 질문(Retrieval-Q-CoT) 대신 클러스터별 대표 질문을 사용해, 동일한 ‘오류 클러스터’ 내 연쇄적 오류 전파 현상(misleading by similarity)을 완화.  
3. **광범위한 벤치마크 검증**:  
   – 산술(GSM8K·MultiArith 등), 상식(CommonSenseQA·StrategyQA), 기호(Last Letter Concatenation, Coin Flip) 총 10개 데이터셋에서 Manual-CoT를 일관되게 능가하거나 대등한 성능 달성.

***

# 문제 정의 및 제안 방법

## 1. 해결하고자 하는 문제  
Few-Shot CoT prompting에서 데모(질문+추론 체인)를 **수작업**으로 작성해야 하는 부담 및 비일관성 문제  
– Manual-CoT 데모는 태스크별 최적화 필요 → 설계 비용 및 재현성 저하  
– Retrieval-Q-CoT는 유사 질문 내 잘못된 체인이 테스트 추론을 오도(misleading)  

## 2. 제안 방법: Auto-CoT  
### (1) 질문 클러스터링  
– 입력 질문 집합 $$Q=\{q_1,\dots,q_n\}$$를 Sentence-BERT로 벡터화  
– k-means로 $$k$$개 클러스터 $$\{C_i\}_{i=1}^k$$ 생성  
– 각 클러스터 $$C_i$$의 질문들을 중심점으로부터의 거리 오름차순으로 정렬  

$$
    C_i: \{q^{(i)}_1, q^{(i)}_2, \dots\},\quad
    \text{where } \|{\rm SBERT}(q^{(i)}_1)-\mu_i\|\le \|{\rm SBERT}(q^{(i)}_2)-\mu_i\|\le\cdots
  $$

### (2) 데모 샘플링 및 휴리스틱  
1. 클러스터 $$C_i$$에서 순차로 질문 $$q^{(i)}_j$$를 선택  
2. Zero-Shot CoT 프롬프트  

$$
   \bigl[\text{“Q: }q^{(i)}_j\text{ A: Let’s think step by step.”}\bigr]
   \;\xrightarrow{\text{LLM}}\;(r^{(i)}_j,a^{(i)}_j)
   $$  
   
   – $$r^{(i)}_j$$: 추론 체인, $$a^{(i)}_j$$: 추출된 답  
3. **선택 기준**:  
   - 질문 길이 $$\le60$$ 토큰  
   - 추론 단계(“ $$\backslash n$$ ” 개수) $$\le5$$  
4. 조건 만족 시 데모 $$\bigl[q^{(i)}_j,r^{(i)}_j,a^{(i)}_j\bigr]$$ 채택, 총 $$k$$개 수집  
5. 최종 인퍼런스 입력:  

$$
   \bigl[d^{(1)},d^{(2)},\dots,d^{(k)},\;\text{Q: }q_{\rm test}\;\text{A: Let’s think step by step.}\bigr]
   $$

***

# 모델 구조, 성능 및 한계

## 모델 구조  
– 기본적으로 GPT-3(text-davinci-002, 175B) 또는 Codex(code-davinci-002) 활용  
– 클러스터링+Zero-Shot CoT로 구축된 **자동화된 Few-Shot CoT** 입력 구조

## 성능 향상  
| 데이터셋       | Manual-CoT | Auto-CoT |
|--------------|----------:|---------:|
| MultiArith   |    91.7%  |   **92.0%** |
| GSM8K        |    46.9%  |   **47.9%** |
| AddSub       |    81.3%  |   **84.8%** |
| AQuA         |    35.8%  |   **36.5%** |
| SingleEq     |    86.6%  |   **87.0%** |
| SVAMP        |    68.9%  |   **69.5%** |
| CSQA         |    73.5%  |   **74.4%** |
| StrategyQA   |    65.4%  |    65.4% |
| Last Letter  |    59.0%  |   **59.7%** |
| Coin Flip    |    97.2%  |   **99.9%** |

– 대부분 데이터셋에서 Manual-CoT를 근소하게 상회  
– **Streaming 설정**에서도 배치별 부트스트랩 방식으로 Manual-CoT와 동등 성능 달성  

## 한계  
1. **Zero-Shot CoT 오류 의존성**:  
   – 휴리스틱으로 잘못된 체인 일부 제거 가능하나, 여전히 최대 20% 오류 체인이 남아 성능 저하 가능  
2. **클러스터 수 및 휴리스틱 민감도**:  
   – k값, 토큰/단계 기준 변경 시 데모 품질 영향  
3. **계산 비용**:  
   – 모든 질문에 대해 SBERT 인코딩 및 LLM 호출 필요 → 대규모 데이터셋/스트리밍에 부담  

***

# 일반화 성능 향상 관점

– **다양성 보장**: 서로 다른 클러스터 대표 질문을 뽑아, 다양한 추론 패턴 커버  
– **오류 파급 억제**: 동일 오류 클러스터 내 질문 반복 활용 억제로 치명적 편향 최소화  
– **스트리밍 대응**: 배치별 부트스트랩으로 신규 질문 도착 시에도 지속적 학습 없이 안정적 성능 유지  

이로써, 태스크나 도메인이 바뀌어도 사전 설계된 데모 없이도 LLM의 추론 능력을 **범용적**으로 활용 가능성이 높아진다.

***

# 향후 연구 영향 및 고려사항

**영향**  
– CoT prompting 자동화 분야에서 **수작업 의존도 제거**의 기폭제  
– 다양한 태스크에 대한 **범용 Few-Shot 학습** 전략으로 활용  
– LLM 기반 추론 응용의 **효율성·재현성** 개선  

**고려사항**  
1. **휴리스틱 자동화 강화**: 데모 품질 평가·필터링 모델 추가로 오류 체인 더 철저히 제거  
2. **동적 클러스터링**: 실시간 데이터 분포 변화에 맞춘 클러스터 수 조정 기법 도입  
3. **추론 체인 검증**: 생성된 체인의 정확도·일관성 자동 검증(예: verifier 모델) 통합  
4. **계산 효율화**: 대규모·스트리밍 환경에서 SBERT·LLM 호출 비용 최적화 방안 연구  

이를 통해 Auto-CoT는 더욱 안정적이고 확장성 있는 CoT prompting 솔루션으로 발전할 수 있을 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/fe62da24-1863-4fde-8632-0a7ad05ceb9b/2210.03493v1.pdf)
