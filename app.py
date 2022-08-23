# flask API #
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from pandas import Series
import pandas as pd
import random

# aisles = pd.read_csv('../aisles.csv')
# departments = pd.read_csv('../departments.csv')
# order_product_prior = pd.read_csv('../order_products__prior.csv')
# order_product_train = pd.read_csv('../order_products__train.csv')
# orders = pd.read_csv('../orders.csv')
# products = pd.read_csv('../products.csv')
id_name = pd.read_csv('./dataset/id_name.csv')

app = Flask(__name__) #flask 앱 초기화

####################### new_model 껍데기 정의는 앞에서 따로 불러오기 #################
import datetime
import os

class MaskedEmbeddingsAggregatorLayer(tf.keras.layers.Layer):
    def __init__(self, agg_mode='sum', **kwargs):
        super(MaskedEmbeddingsAggregatorLayer, self).__init__(**kwargs)

        if agg_mode not in ['sum', 'mean']:
            raise NotImplementedError('mode {} not implemented!'.format(agg_mode))
        self.agg_mode = agg_mode

    @tf.function
    def call(self, inputs, mask=None):
        masked_embeddings = tf.ragged.boolean_mask(inputs, mask)
        if self.agg_mode == 'sum':
            aggregated = tf.reduce_sum(masked_embeddings, axis=1)
        elif self.agg_mode == 'mean':
            aggregated = tf.reduce_mean(masked_embeddings, axis=1)
        return aggregated

    def get_config(self):
        # this is used when loading a saved model that uses a custom layer
        return {'agg_mode': self.agg_mode}


class L2NormLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L2NormLayer, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = tf.ragged.boolean_mask(inputs, mask).to_tensor()
        return tf.math.l2_normalize(inputs, axis=-1)

    def compute_mask(self, inputs, mask):
        return mask

EMBEDDING_DIMS = 16
DENSE_UNITS = 64
DROPOUT_PCT = 0.1
ALPHA = 0.1
NUM_CLASSES = 49689
LEARNING_RATE = 0.1

input_user = tf.keras.Input(shape=(None, ), name='user')
input_product_hist = tf.keras.layers.Input(shape=(None,), name='product_hist')
input_order_dow_hist = tf.keras.layers.Input(shape=(None,), name='order_dow_hist')
input_order_hour_of_day_hist = tf.keras.Input(shape=(None, ), name='order_hour_of_day_hist')

# layer
features_embedding_layer = tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIMS,
                                            mask_zero=True, trainable=True, name='features_embeddings')
labels_embedding_layer = tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIMS,
                                            mask_zero=True, trainable=True, name='labels_embeddings')

avg_embeddings = MaskedEmbeddingsAggregatorLayer(agg_mode='mean', name='aggregate_embeddings')

dense_1 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_1')
dense_2 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_2')
dense_3 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_3')
l2_norm_1 = L2NormLayer(name='l2_norm_1')
dense_output = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax, name='dense_output')

# feature
features_embeddings = features_embedding_layer(input_user)
l2_norm_features = l2_norm_1(features_embeddings)
avg_features = avg_embeddings(l2_norm_features)

labels_product_embeddings = labels_embedding_layer(input_product_hist)
l2_norm_product = l2_norm_1(labels_product_embeddings)
avg_product = avg_embeddings(l2_norm_product)

labels_order_dow_embeddings = labels_embedding_layer(input_order_dow_hist)
l2_norm_order_dow = l2_norm_1(labels_order_dow_embeddings)
avg_order_dow = avg_embeddings(l2_norm_order_dow)

labels_order_hour_embeddings = labels_embedding_layer(input_order_hour_of_day_hist)
l2_norm_order_hour = l2_norm_1(labels_order_hour_embeddings)
avg_order_hour = avg_embeddings(l2_norm_order_hour)

concat_inputs = tf.keras.layers.Concatenate(axis=1)([avg_product,
                                                     avg_order_dow,
                                                     avg_order_hour
                                                     ])
# Dense Layers
dense_1_features = dense_1(concat_inputs)
dense_1_relu = tf.keras.layers.ReLU(name='dense_1_relu')(dense_1_features)
dense_1_batch_norm = tf.keras.layers.BatchNormalization(name='dense_1_batch_norm')(dense_1_relu)

dense_2_features = dense_2(dense_1_relu)
dense_2_relu = tf.keras.layers.ReLU(name='dense_2_relu')(dense_2_features)
dense_2_batch_norm = tf.keras.layers.BatchNormalization(name='dense_2_batch_norm')(dense_2_relu)

dense_3_features = dense_3(dense_2_relu)
dense_3_relu = tf.keras.layers.ReLU(name='dense_3_relu')(dense_3_features)
dense_3_batch_norm = tf.keras.layers.BatchNormalization(name='dense_3_batch_norm')(dense_3_relu)

outputs = dense_output(dense_3_batch_norm)

#Optimizer
optimiser = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

#--- prep model
model = tf.keras.models.Model(
    inputs=[input_product_hist,
            input_order_dow_hist,
            input_order_hour_of_day_hist
            ],
    outputs=[outputs]
)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#################################################################
model.load_weights('kur_weight')
##############################################################

products, datediff = [], []
def put_stocks(pd):
    global products
    products = pd

sold_out = []
def put_sold(s_out):
    global sold_out
    sold_out = s_out

@app.route('/stocks', methods=['POST'])
def stocks():
    global products
    params = request.get_json()
    products = params['products'] #product id가 담긴 리스트

    put_stocks(products)

    return jsonify({'products': products})


@app.route('/products/soldout', methods=['POST'])
def sold():
    global sold_out
    params = request.get_json()
    sold_out = params['sold'] #sold_out 된 물품들 리스트

    put_sold(sold_out)

    return jsonify({'sold_out': sold_out})


@app.route('/kurly-bag', methods=['POST'])
def predict():
    global products
    info = request.get_json()
    # user_id = info['user_id'] #고객 id
    # products = info['products'] #이전 구매목록 12개
    # order_dow = info['order_dow'] #이전 구매 요일 12개 리스트
    # order_hour_of_day = info['order_hour_of_day'] #이전 구매 시간 12개 리스트
    df = pd.DataFrame(info, columns = ['products', 'order_dow', 'order_hour_of_day'])

    ## 물류 재고 상황 먼저 고려 ##
    # ->고객 이전 구매 12번 내에 해당 상품이 3번 구매한 이력이 있으면 먼저 컬리백으로 구성
    kurlybag = []  # 총 7개추천
    num = 0
    

    ## 후보군 생성 ##
    pred = model.predict([tf.keras.preprocessing.sequence.pad_sequences(df['products']),
       tf.keras.preprocessing.sequence.pad_sequences(df['order_dow']),
       tf.keras.preprocessing.sequence.pad_sequences(df['order_hour_of_day'])
       ])

    N = 15
    candidate = np.sort((-pred).argsort()[:, :N])

    candidate = candidate.flatten()
    candidate[candidate > 35053] = 0  #data['product'].max()=35053
    candidate = np.unique(candidate) #top 15개 상품 리스트
    candidate = candidate.tolist()


    ## ranking ## - 물류 재고 현황 고려, 소진 상품 고려


    products_set = set(products) #24시간에 한번 업데이트
    sold_set = set(sold_out) #5분마다 업데이트
    products = list(products_set - sold_set) #(4일 남은 재고 - 소진된 상품)으로 재고 데이터 업데이트

    # for p in (products):
    #     if num > 8:
    #         break
    #     if p in candidate:
    #         kurlybag.append(p)
    #         num += 1
    #         candidate.remove(p)
    if num < 8:
        more = 8-num
        kurlybag += candidate[:more]

    kur_df = pd.DataFrame(columns = ['product_no', 'product_nm'])
    for pd_no in kurlybag:
        kur_df = kur_df.append(id_name.loc[id_name['product_no'] == pd_no])

    price_lst = [3400, 4500, 5000, 5500, 6400, 7000, 7500, 7900, 8500, 9000, 9900, 10500, 11000, 12000, 12500, 13000]
    pri = random.sample(price_lst, 7)
    img_path = 'https://hallikurly.s3.ap-northeast-2.amazonaws.com/kurlybag/kurlybag_default_img.png'
    kur_df['price'] = pri
    kur_df['product_img_path'] = [img_path for _ in range(7)]

    return kur_df.to_dict('records')

if __name__ == '__main__':
    app.run(debug=True)