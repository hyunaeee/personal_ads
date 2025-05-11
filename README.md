## Personalized Ad Recommendation System ( 개인화된 광고 추천 시스템 )

### Project Overview | 프로젝트 개요
This repository implements an end-to-end personalized advertisement recommendation system that dynamically selects and ranks the most relevant ads for users based on their historical behavior, contextual information, and real-time signals. The system incorporates a multi-stage ranking pipeline with collaborative filtering, content-based filtering, and deep learning-based matching to maximize both user engagement and advertiser return on investment.

이 저장소는 사용자의 과거 행동, 상황 정보 및 실시간 신호를 기반으로 가장 관련성 높은 광고를 동적으로 선택하고 순위를 매기는 엔드투엔드 개인화 광고 추천 시스템을 구현합니다. 이 시스템은 협업 필터링, 콘텐츠 기반 필터링 및 딥러닝 기반 매칭이 포함된 다단계 랭킹 파이프라인을 통합하여 사용자 참여와 광고주의 투자 수익률을 최대화합니다.

### Key Features | 주요 기능
- **Multi-stage recommendation pipeline**: Retrieval, ranking, and re-ranking stages for efficient ad selection
- **User embedding generation**: Deep representation of user interests and preferences
- **Ad embedding generation**: Content-based and performance-based ad embeddings
- **Real-time contextual matching**: Contextual bandits approach for matching users to ads
- **Diversity and exploration**: Strategic exploration to discover new user-ad affinities
- **Cold-start handling**: Techniques for handling new users and new ad campaigns
- **A/B testing framework**: Integrated system for evaluating recommendation strategies

- **다단계 추천 파이프라인**: 효율적인 광고 선택을 위한 검색, 랭킹 및 재랭킹 단계
- **사용자 임베딩 생성**: 사용자 관심사 및 선호도에 대한 심층 표현
- **광고 임베딩 생성**: 콘텐츠 기반 및 성과 기반 광고 임베딩
- **실시간 상황별 매칭**: 사용자와 광고를 매칭하기 위한 컨텍스트 밴딧 접근법
- **다양성 및 탐색**: 새로운 사용자-광고 친화성을 발견하기 위한 전략적 탐색
- **콜드 스타트 처리**: 새로운 사용자 및 새로운 광고 캠페인 처리 기법
- **A/B 테스트 프레임워크**: 추천 전략을 평가하기 위한 통합 시스템

### Technical Architecture | 기술 아키텍처
The system is built on a modern ML stack:

이 시스템은 최신 ML 스택을 기반으로 구축되었습니다:

1. **Data Processing Layer**:
   - Spark for batch processing of user history
   - Kafka for real-time event streaming
   - Feature Store for low-latency feature serving

2. **Embedding Generation**:
   - BERT-based encoding for ad content
   - Graph Neural Networks for user-ad interaction patterns
   - Factorization Machine for historical interaction modeling

3. **Retrieval Layer**:
   - Approximate Nearest Neighbor (ANN) search using FAISS
   - Two-tower architecture for efficient candidate generation
   - Category-based retrieval for diversity

4. **Ranking Layer**:
   - LambdaMART for initial ranking
   - Deep & Cross Network for feature interaction
   - Calibrated prediction with uncertainty estimation

5. **Serving Layer**:
   - TensorFlow Serving for model deployment
   - Adaptive batching for high throughput
   - Redis caching for frequent requests

### Performance Metrics | 성능 지표
The system is evaluated on multiple dimensions:

이 시스템은 여러 차원에서 평가됩니다:

| Metric | Without Personalization | With Personalization | Improvement |
|--------|-------------------------|----------------------|-------------|
| CTR | 1.2% | 1.85% | +54.2% |
| CVR | 0.15% | 0.22% | +46.7% |
| Revenue per Impression | $0.0035 | $0.0052 | +48.6% |
| User Satisfaction | 3.2/5 | 3.8/5 | +18.8% |
| Latency (p95) | 35ms | 42ms | +7ms |

### User Embedding Analysis | 사용자 임베딩 분석
The system generates high-dimensional user embeddings that capture user interests and behavior patterns. The below visualization shows a t-SNE projection of user embeddings colored by dominant user interests:

이 시스템은 사용자 관심사와 행동 패턴을 포착하는 고차원 사용자 임베딩을 생성합니다. 아래 시각화는 주요 사용자 관심사로 색상화된 사용자 임베딩의 t-SNE 투영을 보여줍니다

### System Design | 시스템 설계
The end-to-end workflow is designed for both offline training and online serving:

엔드투엔드 워크플로우는 오프라인 학습과 온라인 서빙을 모두 고려하여 설계되었습니다:

```
                                  +-------------------+
                                  |                   |
                                  |  Offline Training |
                                  |                   |
                                  +-------------------+
                                            |
                                            v
+----------------+    +--------------+    +--------------+    +-------------+    +------------+
|                |    |              |    |              |    |             |    |            |
| User Request   |--->| Retrieval    |--->| Ranking      |--->| Re-ranking  |--->| Ad Display |
| (Context)      |    | Service      |    | Service      |    | Service     |    |            |
|                |    | (500 ads)    |    | (50 ads)     |    | (10 ads)    |    |            |
+----------------+    +--------------+    +--------------+    +-------------+    +------------+
       |                     ^                   ^                  ^
       |                     |                   |                  |
       v                     |                   |                  |
+----------------+           |                   |                  |
|                |           |                   |                  |
| User Feedback  |-----------|-------------------|------------------|
|                |
+----------------+
```

### Cold Start Strategy | 콜드 스타트 전략
For cold start problems, we implement:

콜드 스타트 문제를 위해 다음을 구현합니다:

1. **For new users**:
   - Demographic-based initial recommendations
   - Exploration through contextual bandits
   - Fast embedding updates with few interactions

2. **For new ads**:
   - Content-based similarity to existing ads
   - Controlled exposure strategy
   - Multi-armed bandit with Thompson sampling

### Installation and Usage | 설치 및 사용법
```bash
# Clone the repository | 저장소 복제
git clone https://github.com/username/personalized-ads.git
cd personalized-ads

# Setup environment | 환경 설정
pip install -r requirements.txt

# Process historical data | 과거 데이터 처리
python scripts/process_historical_data.py --input data/user_interactions.parquet --output data/processed/

# Train user embeddings | 사용자 임베딩 학습
python models/user_embedding/train.py --config configs/user_embedding.yaml

# Train ad embeddings | 광고 임베딩 학습
python models/ad_embedding/train.py --config configs/ad_embedding.yaml

# Train ranking model | 랭킹 모델 학습
python models/ranking/train.py --config configs/ranking_model.yaml

# Start recommendation server | 추천 서버 시작
python serve/recommendation_server.py --port 8000
```

### Components and Modules | 컴포넌트 및 모듈
```
├── configs/               # Configuration files | 설정 파일
├── data/                  # Data processing scripts | 데이터 처리 스크립트
│   ├── loaders/           # Data loading modules | 데이터 로딩 모듈
│   ├── processors/        # Feature transformation | 특성 변환
│   └── schemas/           # Data schemas | 데이터 스키마
├── models/                # Model implementations | 모델 구현
│   ├── user_embedding/    # User embedding models | 사용자 임베딩 모델
│   ├── ad_embedding/      # Ad embedding models | 광고 임베딩 모델 
│   ├── retrieval/         # Fast retrieval models | 빠른 검색 모델
│   ├── ranking/           # Ranking models | 랭킹 모델
│   └── reranking/         # Final stage models | 최종 단계 모델
├── evaluation/            # Evaluation framework | 평가 프레임워크
│   ├── metrics/           # Metric calculation | 지표 계산
│   ├── ab_testing/        # A/B testing tools | A/B 테스트 도구
│   └── visualizations/    # Result visualization | 결과 시각화
├── serving/               # Serving infrastructure | 서빙 인프라
│   ├── api/               # REST API endpoints | REST API 엔드포인트
│   ├── middleware/        # Request processing | 요청 처리
│   └── caching/           # Response caching | 응답 캐싱
└── notebooks/             # Analysis notebooks | 분석 노트북
```

### Key Algorithms | 핵심 알고리즘
The repository includes implementations of:

이 저장소에는 다음과 같은 알고리즘 구현이 포함되어 있습니다:

1. **Two-Tower Neural Network**: Efficient user-ad matching
2. **DeepFM**: Feature interaction modeling
3. **BERT-based Ad Encoder**: Semantic understanding of ad content
4. **Graph Neural Network**: User-ad interaction patterns
5. **Contextual Multi-Armed Bandit**: Exploration/exploitation balance

### Monitoring and Maintenance | 모니터링 및 유지 관리
The system includes:

이 시스템에는 다음이 포함됩니다:

- Real-time monitoring of recommendation quality
- Automated retraining pipeline
- Drift detection for user behavior changes
- Debugging tools for failed recommendations

### Future Work | 향후 작업
- Implement multi-objective optimization (balancing CTR, CVR, and revenue)
- Add causal inference for unbiased recommendation
- Incorporate federated learning for privacy-preserving personalization
- Implement reinforcement learning for sequential recommendation

### References | 참고 문헌
1. Covington, P., et al. (2016). Deep Neural Networks for YouTube Recommendations. RecSys 2016.
2. Cheng, H., et al. (2016). Wide & Deep Learning for Recommender Systems. DLRS 2016.
3. Zhou, G., et al. (2019). Deep Interest Network for Click-Through Rate Prediction. SIGKDD 2018.
4. Wang, R., et al. (2017). Factorization Machines with Follow-the-Regularized-Leader for CTR prediction. BigData 2017.

## Contact | 연락처
For questions or collaboration, please open an issue or contact [hyunaeee@gmail.com](mailto:hyunaeee@gmail.com).

질문이나 협업을 위해서는 이슈를 열거나 [hyunaeee@gmail.com](mailto:hyunaeee@gmail.com)으로 문의해 주세요.
