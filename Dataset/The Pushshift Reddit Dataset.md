# The Pushshift Reddit Dataset

**핵심 주장 및 주요 기여**  
Pushshift Reddit Dataset은 2005년부터 실시간으로 수집·아카이빙된 Reddit의 모든 제출(submissions)과 댓글(comments)을 월별 ndjson 파일과 검색 API, 그리고 Slack봇 인터페이스로 제공함으로써, 대규모 소셜 미디어 연구를 위한 데이터 수집·처리·접근의 진입 장벽을 획기적으로 낮췄다.  

***

## 1. 해결하고자 하는 문제  
기존 연구자들은 Reddit 데이터 수집을 위해 공식 Reddit API의 엄격한 요청 한계(100개 오브젝트 제한), 과거 데이터 접근성 부족, 텍스트 검색 및 집계 기능 부재 등으로 인해  
-  데이터 수집·저장 인프라 구축에 과도한 시간과 기술적 자원을 투입해야 했고  
-  연구 재현성과 확장성에 제약이 있었다.  

***

## 2. 제안하는 방법  
Pushshift 플랫폼은 다음의 핵심 컴포넌트로 구성되어 있다.  

  1. **Ingest Engine**  
     -  다양한 언어로 작성된 크롤러 프로그램 실행  
     -  Raw JSON을 레디스 큐에 적재 후 PostgreSQL과 ElasticSearch에 저장  
  2. **PostgreSQL & ElasticSearch**  
     -  PostgreSQL: 메타데이터 및 고급 쿼리 지원  
     -  ElasticSearch: 대규모 색인(indexing), 전체 텍스트 검색, 집계(aggregation)  
     -  동적 매핑(dynamic mapping)으로 API 버전 변화 대응  
     -  ICU 플러그인으로 유니코드·이모지 검색 지원  
  3. **API 및 Slack봇**  
     -  월별 덤프 파일(651M submissions, 5.6B comments) 제공  
     -  RESTful 검색·집계 엔드포인트  
     -  Slack봇: 대화형 쿼리 → 시계열 플롯·통계 즉시 반환  

***

## 3. 모델 구조, 수식, 성능 향상  
본 논문은 기계학습 모델을 제안하는 연구가 아니라, **데이터 인프라스트럭처** 논문이므로 모델 구조·수식·성능 비교 표는 포함되지 않는다.  

***

## 4. 한계 및 일반화 성능 향상 가능성  
-  데이터 누락: Reddit API 변동·차단 이슈에 따른 일부 데이터 손실 가능성  
-  메타데이터 편향: Reddit 자체 투표 스코어의 “fuzzing” 기법으로 인해 정확한 인기도 측정 어려움  
-  일반화 관점:  
  – 다른 플랫폼(e.g., Twitter, Facebook)과 달리 Reddit만의 서브레딧 구조·모더레이션 메커니즘이 특징적이므로,  
  – 타 플랫폼에 Pushshift 스타일 아키텍처를 적용하려면 API 정책·데이터 형식·접근성 제약을 재설계해야 함  

***

## 5. 향후 연구에 미치는 영향 및 고려사항  
Pushshift Reddit Dataset은 대규모 사회컴퓨팅, 온라인 극단주의, 가짜뉴스, 건강정보 확산, 자연어처리, 추천시스템 등 다양한 분야에서  
-  연구 재현성 향상  
-  실시간 트렌드 모니터링  
-  인터랙티브 데이터 탐색  
가능성을 크게 확장했다.  

앞으로 연구자들은  
1. 데이터 품질 평가(결측·편향 모니터링)  
2. 비-영어권·익명성 이슈 고려  
3. 사생활·윤리적 제약 감안한 데이터 사용 지침  
4. 플랫폼 종단간 비교를 위한 아키텍처 확장  
등을 설계 단계에서 통합해야 할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f086072c-8489-40b3-b37d-8e31373e5b66/2001.08435v1.pdf)
