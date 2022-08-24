# Recommand_System_ml 

![image](https://user-images.githubusercontent.com/82245639/186336785-7071c3cb-5fce-4552-837c-377826c5ad4d.png)


# 사용자 구매패턴, 물류 재고 현황 고려한 추천 알고리즘 

[2-stage 의사결정]
- 단계적 상품 추천 후보군 도출함으로 여러 특징들을 고려한 추천을 할 수 있음 

- 실시간 재고 현황을 즉각적으로 고려해 추천 컬리백 구성을 다르게 할 수 있는 유연함 

- 폐기율은 더욱 낮추고, 고객의 구매 패턴 기반의 컬리백을 구성해 한번에 구매 할 수 있는 편리성 제공 

> (step1) 고객의 구매 이력을 바탕으로 관심 상품에 한해, 물류 재고 현황을 고려한 우선 추천 가능 
>
> (step2) 후보 생성 모델에서 생성된 후보 상품군들에서 확률이 높은 순서대로 나머지 컬리백 구성 



# 구매 이력을 바탕으로 한 다음 구매 후보 생성 모델 

유튜브 추천 모델 활용 [Deep Neural Networks for YouTube Recommandation]
![image](https://user-images.githubusercontent.com/82245639/186337862-688d7d7f-6506-485c-b978-620183112e94.png)

dataset: https://www.kaggle.com/c/instacart-market-basket-analysis/data, from kaggle 

- 구글 딥 러닝 인공지능 알고리즘과 심층 신경망 (Deep neural network, DNN) 모델 사용

- 기존 후보 생성 모델에서 input feature로 동영상 시청 시간, 검색 토큰 등을 임베딩하여 모델을 구성했지만, 
이커머스 상품 추천이라는 도메인에 맞추어 구매 상품들, 구매 시각, 구매 일자를 embeded vector로 input feature로 사용함

- 출력층에 softmax 함수를 적용해 각 상품의 다음에 구매할 확률을 출력함 


# 실시간 물류 재고 현황 고려한 의사결정 

- 폐기 임박 상품에서 컬리백 요청을 받는 시점에 소진된 상품을 제외시키고 업데이트 

- 폐기 임박 상품이 이전 구매 이력과 후보군에 포함된 상품이면 우선적으로 컬리백 구성됨

- 24시간 마다, 폐기 시점 임박한 상품들 목록 업데이트,

- 매 5분마다 소진된 상품들 목록 업데이트 -> 폐기 임박한 상품들에서 제외 
  - 매번 모든 폐기 임박 상품 데이터를 받지 않고 소진된 상품만 5분 주기로 업데이트 되므로 
    데이터 처리의 효율성 상승, 서비스 과부하 방지 
    
![image](https://user-images.githubusercontent.com/82245639/186342058-c955e13b-053c-4acf-84aa-296015e6742a.png)


