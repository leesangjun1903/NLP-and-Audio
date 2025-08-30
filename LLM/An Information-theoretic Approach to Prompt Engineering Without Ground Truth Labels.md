# An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels

**핵심 주장 및 주요 기여**  
“An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels”은 대규모 언어 모델(LLM) 프롬프트의 품질을 레이블 데이터나 모델 파라미터 접근 없이 평가할 수 있음을 보인다.  
1. 입력과 출력 간 **상호정보(Mutual Information, MI)**를 계산하여 프롬프트 템플릿의 성능을 예측  
2. MI가 높은 프롬프트가 실제 높은 정확도를 보임을 실험적으로 검증  
3. 8개 NLP 과제, 7개 태스크, GPT-3 등 모델 크기에 걸쳐 90% 수준의 오라클 성능 달성[1]

***

## 1. 해결하고자 하는 문제  
기존 프롬프트 엔지니어링은  
- 대량의 **라벨된 검증 데이터** 필요  
- **모델 파라미터 접근** 혹은 **백프로파게이션** 필요  
- 템플릿 평가에 높은 비용 소모  
이를 해결하기 위해, 본 논문은 라벨과 파라미터 접근 없이도 프롬프트 품질을 평가하는 방법을 제안한다.[1]

***

## 2. 제안 방법  
### 2.1. One-token Response (OTR) 프레임워크  
- 출력 공간 $$Y$$의 각 클래스가 고유 토큰으로 시작  
- 모델 $$\phi$$에 입력 $$x$$를 템플릿 $$f_\theta$$로 변환하여 logits 분포 $$P(T_\phi\mid f_\theta(x))$$ 획득  
- **콜랩스 함수** $$c_\theta$$로 클래스별 토큰 확률을 합산·정규화하여 $$P(Y\mid f_\theta(x))$$ 생성  

### 2.2. 상호정보 계산  

$$
I(f_\theta(X);Y) = H(Y) - H(Y \mid f_\theta(X))
$$  

- 주변 엔트로피: $$H(Y)\approx H\bigl(\tfrac1N\sum_i P(Y\mid f_\theta(x_i))\bigr)$$  
- 조건부 엔트로피: $$H(Y\mid f_\theta(X))\approx\tfrac1N\sum_i H\bigl(P(Y\mid f_\theta(x_i))\bigr)$$  
- 높은 MI는 *모델이 불편향(high marginal entropy)·확신(낮은 conditional entropy)*있게 응답함을 의미  

### 2.3. 템플릿 선택  
1. 후보 템플릿 $$f_{\theta_1},\dots,f_{\theta_K}$$ 생성  
2. 각 템플릿마다 미리 정의된 샘플 $$N$$개에 대해 MI 추정  
3. $$\widehat\theta = \arg\max_\theta I(f_\theta(X);Y)$$ 선택  
4. 선택된 템플릿으로 실제 추론 수행  

***

## 3. 모델 구조 및 실험 설정  
- **모델**: GPT-3 (175B, 13B, 6.7B, 2.7B), GPT-J (6B), GPT-Neo (2.7B), GPT-2 (1.5B, 124M)  
- **데이터셋**: SQuAD2.0, LAMBADA, ROCStories, CoQA, IMDB, BoolQ, COPA, WiC  
- **템플릿 수**: 각 태스크별 $$K=20$$개  
- **샘플링 수**: $$N=500$$ (추정 시)  

***

## 4. 성능 향상 및 한계  
- **오라클 대비 90%** 성능 달성 (GPT-3 175B 기준)  
- 라벨 없이도 평균·중간 이상 프롬프트 선택 확률 83%  
- **소규모 모델**에서는 MI-정확도 상관이 낮아 선택 성능 제한  
- **취약점**: MI 상위 프롬프트 중 일부가 저성능일 수 있어, 상위 5개 앙상블 추천  

***

## 5. 일반화 성능 향상 가능성  
- MI 기반 선택은 레이블·백프로파게이션 없이도 **다양한 태스크**에 적용 가능  
- **전이학습**: 선택 모델과 추론 모델이 다를 때도 안정적 전이[1]
- 소규모 모델→대규모 모델, 혹은 반대 방향으로도 MI 선택 템플릿 성능 유지  

***

## 6. 향후 연구에 미치는 영향 및 고려할 점  
- **프롬프트 자동화**: 라벨이 부족한 도메인에서 비용·리소스 절감  
- **앵커링·편향**: MI가 높은 프롬프트가 꼭 안전·공정한 출력 보장하지 않음  
- **템플릿 다양성**: 후보군 생성 시 다양한 스타일·구조 확보 필요  
- **스케일 민감도**: 작은 모델·저신호 태스크에선 대안 평가 기준 연구 필요[1]

 attached_file:1[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/10e05667-f60a-40de-87c7-455d15aeb99d/2203.11364v1.pdf)
