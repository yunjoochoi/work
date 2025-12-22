- Slide 1 -

# AI Safety 진단 툴  사례 분석

김 능 준

2025.11.05

- Slide 2 -

# ① 기술적 취약점 분석 도구

| 이름                                   | 주요 기능                           | 활용 목적               |    |
|--------------------------------------|---------------------------------|---------------------|----|
| Microsoft Counterfit                 | AI 모델 대상 공격 시뮬레이션 (FGSM, PGD 등) | 모델 보안성 자동 테스트       |    |
| PentestGPT                           | GPT 기반 자동 침투 테스트                | LLM 및 API 보안 취약점 탐지 |    |
| BurpGPT                              | 웹 기반 AI 서비스의 보안 분석              | 웹 트래픽 기반 비표준 취약점 탐지 |    |
| RobustBench                          | 적대적 공격에 대한 모델 강건성 평가            | CV/ML 모델의 공격 저항성 비교 |    |
| Adversarial Robustness Toolbox (ART) | 적대적 공격 생성 및 방어 기법 테스트           | AI 모델의 공격 대응력 검증    |    |

- Slide 3 -

# ② 프레임워크 기반 평가 도구

| 이름                         | 주요 기능                                        | 활용 목적                   |    |
|----------------------------|----------------------------------------------|-------------------------|----|
| AI 신뢰성 센터 (TTA)            | 국내 기준 기반 신뢰성/보안/윤리 평가                        | 공공기관 및 기업의 AI 서비스 검증    |    |
| NIST AI RMF (미국)           | AI 위험관리 프레임워크 (Govern, Map, Measure, Manage) | 조직 차원의 AI 리스크 관리        |    |
| OECD AI Principles Toolkit | 국제 윤리 기준 기반 평가 프레임워크                         | 글로벌 AI 정책 대응 및 가이드라인 수립 |    |
|                            |                                              |                         |    |
|                            |                                              |                         |    |

- Slide 4 -

# ③ 평가 문항 기반 진단 도구 (etc)

| 이름                             | 주요 기능                 | 활용 목적              |    |
|--------------------------------|-----------------------|--------------------|----|
| IBM AI Fairness 360 (AIF360)   | 편향성 진단 및 수정 알고리즘 제공   | 공정성 기반 평가 및 리포트 생성 |    |
| Google Model Card Toolkit      | 모델 설명력 및 책임성 문서화      | 투명성 확보 및 규제 대응     |    |
| OpenAI Eval Framework          | 사용자 정의 평가 시나리오 기반 테스트 | LLM 성능 및 안전성 평가    |    |
| AI Explainability 360 (XAI360) | 설명 가능성 평가 지표 제공       | 모델 해석력 진단 및 시각화    |    |
|                                |                       |                    |    |

- Slide 5 -

# AI 신뢰성 센터 (TTA)

과학기술정보통신부와 한국정보통신기술협회(TTA)가 운영하는 국내 공공 AI 신뢰성 검증 플랫폼

국제 AI 윤리·기술 표준 사용

- ISO/IEC-23894 인공지능 시스템
- ISO/IEC-42001 인공지능 개발 사업자 및 인공지능 이용사업자
- ISO/IEC-38507 인공지능 이용자

자가 진단 시스템

검증 항목 기반 진단: AI 시스템의 신뢰성 수준을 점검할 수 있는 진단 문항 제공

진단 결과 리포트: 각 항목별 위험도, 개선 필요성, 권고사항 제공

AI 신뢰성 센터

정책 준수, 표준화를 위한 프레임워크만을 제공, 기술적요소X

플랫폼이란 말도 아까움…

- Slide 6 -

# IBM AI Fairness 360 (AIF360)

작동 원리

① 데이터 분석

입력 데이터에서 보호 속성을 지정

그룹 간 예측 결과의 차이를 수치화된 지표로 평가

 ② 공정성 지표 계산

② 대표 지표:

Statistical Parity Difference (SPD): 두 그룹 간 긍정 예측 비율 차이

Disparate Impact (DI): 비특권 그룹의 긍정 예측 비율 ÷ 특권 그룹의 비율

Equal Opportunity Difference, Average Odds Difference 등도 제공

③ 편향 개선 알고리즘 적용

Pre-processing: 학습 전에 데이터 재가공 (예: Reweighing)

In-processing: 학습 중 편향 제어 (예: Adversarial Debiasing)

Post-processing: 예측 결과 수정 (예: Reject Option Classification)

④ 비교 및 시각화

편향 개선 전후의 지표를 비교

시각화 도구로 결과 해석 가능

ML모델의 **공정성(fairness)**을 평가하고 개선하기 위한 오픈소스 라이브러리

Python 등 lib import하여 사용 가능

핵심 개념

공정성(Fairness): AI 모델이 특정 집단(예: 성별, 인종 등)에 대해 편향된 결과를 내지 않도록 하는 것

보호 속성(Protected Attributes): 공정성 평가의 기준이 되는 민감한 속성 (예: gender, race)

편향(Bias): 특정 집단에 불리하거나 유리하게 작용하는 예측 결과의 불균형

연구용 머신러닝(ML)이 아닌, 상품화를 위한 ML. 생성형AI의 안정성,robustness X.

사회적 기준의 통계적 접근을 통한 AI서비스의 상업 경쟁력 강화

- Slide 7 -

# **Microsoft Counterfit

보안 취약점 테스트를 자동화할 수 있는 오픈소스 프레임워크

1. 
2. 공격 시뮬레이션 Counterfit은 FGSM, PGD, Carlini-Wagner 등 다양한 적대적 공격 알고리즘을 내장
3. 
4. 프레임워크 독립성 TensorFlow, PyTorch, Scikit-learn 등 다양한 프레임워크와 호환. REST API 기반 모델 가능. 
5. 자동화 및 리포트 생성 CLI 기반으로 반복 테스트를 자동화하고, 공격 성공률이나 정확도 변화 등을 리포트 형태로 제공
6. 
7. MLOps 파이프라인에 통합 가능 
8. 

3.API 테스트 지원 외부에 배포된 모델(API 형태)도 HTTP 요청을 통해 공격 가능.

SaaS나 클라우드 기반 AI 서비스의 보안 점검에 적합. 

Azure 컨테이너로 운영 가능하며 MLOps의 부품으로서 자동화 툴로 가장 서비스화에 적합.

마지막 릴리즈가 3,4년 전으로 out of trend일 가능성 有

GitHub - Azure/counterfit: a CLI that provides a generic automation layer for assessing the security of ML models

- Slide 8 -

# BurpGPT

웹 앱 취약점 탐지 툴인 Burp Suite의 트래픽 분석 기능에 GPT 확장 버전

메인 기능 – Burp Suite이 수집한 HTTP 요청/응답 데이터를 모델에 전달하여, 비정형적이거나 로직 기반의 취약점을 탐지

WAPT(웹 애플리케이션 침투 테스트) 자동화

보안 교육 및 시뮬레이션 도구로 활용

비정형 입력 처리 로직의 취약점 탐지 (예: 인증 우회, 비표준 API 응답)

위 갈색 항목의 기능들도 주장하지만 실용성 이슈

흔한 보안 관제 툴 내 descriptive AI 수준을 벗어나지 않음.

WAPT 자동화 또한 공격 시나리오 설계&amp;테스트 하기보다는 

gpt에 웹로그 제공 후 이상 로그 탐지 or 방대한 결과 data 요약 정도의 기능으로 보임.