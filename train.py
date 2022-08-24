import pandas as pd
import random
import pickle
import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')
order_product_prior = pd.read_csv('order_products__prior.csv')
order_product_train = pd.read_csv('order_products__train.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')

orders = orders.dropna()


random.seed(2021)

# 1만명의 유저만 샘플링 (첫시도)
user = random.sample(orders['user_id'][orders['eval_set']=='train'].unique().tolist(), 10000)
#user = orders['user_id'][orders['eval_set']=='train'].unique().tolist() #샘플링 안하고 시도 (실패)
# orders에서 test dataset 관련 기록(prior포함) 제외
orders_2 = orders[orders['user_id'].isin(user)]

order_product_prior = order_product_prior[order_product_prior['order_id'].isin(orders_2['order_id'])]

product_m = products.merge(aisles).merge(departments)

# product one-hot encoding data
product_enc = pd.get_dummies(product_m, columns=['aisle'], prefix=[None])

order_product = pd.concat([order_product_prior,order_product_train])
# 주문 항목 정보 : product_m + order_product
order_detail = order_product.merge(product_m,how='left') #.sample(n=100000, random_state=2021)

data = orders_2.merge(order_detail)#, on ='order_id')

## modeling ##
data['user_id'] = data['user_id'].astype(int)
data['product_id'] = data['product_id'].astype(int)
data['order_id'] = data['order_id'].astype(int)
data['days_since_prior_order'] = data['days_since_prior_order'].astype(int)

data = data.set_index(['user_id']).sort_index()
data = data.reset_index()

# 유저 인덱스 인코딩
user_ids = data["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
#userencoded2user = {i: x for i, x in enumerate(user_ids)}

# 주문 인덱스 인코딩
order_ids = data["order_id"].unique().tolist()
order2order_encoded = {x: i for i, x in enumerate(order_ids)}
#order_encoded2order = {i: x for i, x in enumerate(order_ids)}

# 상품 인덱스 인코딩
product_ids = data["product_id"].unique().tolist()
product2product_encoded = {x: i for i, x in enumerate(product_ids)}
#product_encoded2product = {i: x for i, x in enumerate(product_ids)}

# 상품 이름 인코딩
pd_name_ids = data["product_name"].unique().tolist()
pd_name2pd_name_encoded = {x: i for i, x in enumerate(pd_name_ids)}
#pd_name_encoded2pd_name = {i: x for i, x in enumerate(pd_name_ids)}

# 상품 대분류 인덱스 인코딩
department_ids = data["department_id"].unique().tolist()
department2department_encoded = {x: i for i, x in enumerate(department_ids)}
#department_encoded2department = {i: x for i, x in enumerate(department_ids)}

# 상품 소분류 인덱스 인코딩
aisle_ids = data["aisle_id"].unique().tolist()
aisle2aisle_encoded = {x: i for i, x in enumerate(aisle_ids)}
#aisle_encoded2aisle = {i: x for i, x in enumerate(aisle_ids)}

# 상품 대분류명 인덱스 인코딩
dept_name_ids = data["department"].unique().tolist()
dept_name2dept_name_encoded = {x: i for i, x in enumerate(dept_name_ids)}
#dept_name_encoded2dept_name = {i: x for i, x in enumerate(dept_name_ids)}

# 상품 소분류명 인덱스 인코딩
aisle_name_ids = data["aisle"].unique().tolist()
aisle_name2aisle_name_encoded = {x: i for i, x in enumerate(aisle_name_ids)}

# 인코딩으로 바꾸기
data["user"] = data["user_id"].map(user2user_encoded)
data["product"] = data["product_id"].map(product2product_encoded)
data["order"] = data["order_id"].map(order2order_encoded)
data["pd_name"] = data["product_name"].map(pd_name2pd_name_encoded)

order_hist = data.groupby(['user'])['order_id'].unique().apply(list).reset_index()
product_hist = data.groupby(['user'])['product_id'].apply(list).reset_index()
order_dow_hist = data.groupby(['user'])['order_dow'].apply(list).reset_index() # unique().적용해보기
order_hour_of_day_hist = data.groupby(['user'])['order_hour_of_day'].apply(list).reset_index()
days_since_prior_order_hist = data.groupby(['user'])['days_since_prior_order'].apply(list).reset_index()

data.groupby(['user'])['order_dow'].unique().apply(list)
order_product_hist = data.groupby(['order'])['product_id'].apply(list).reset_index()
user_data = data[['user','user_id']].merge(order_hist, how='left').merge(product_hist, how='left').merge(order_dow_hist, how='left').merge(order_hour_of_day_hist, how = 'left').merge(days_since_prior_order_hist,how='left') #eval_set
user_data = user_data.drop_duplicates('user') # 중복데이터 삭제

data_product_prior=data['product'][data['eval_set']=='prior']
user_data['predict_labels'] = user_data['product_id'].apply(lambda x: int(random.uniform(0,data['product_id'].max())))

train_data = user_data[(user_data.user>=30) &
                       (user_data.user<=39)]
test_data = user_data[(user_data.user>=40) &
                      (user_data.user<=59)]

######
# 하이퍼파라미터 정의
EMBEDDING_DIMS = 16
DENSE_UNITS = 64
DROPOUT_PCT = 0.1
ALPHA = 0.1
NUM_CLASSES = data["product_id"].max() + 2
LEARNING_RATE = 0.1

# custom layers




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
# modeling
import tensorflow as tf
import datetime
import os

input_user = tf.keras.Input(shape=(None, ), name='user')
input_product_hist = tf.keras.layers.Input(shape=(None,), name='product_hist')
input_order_dow_hist = tf.keras.layers.Input(shape=(None,), name='order_dow_hist')
input_order_hour_of_day_hist = tf.keras.Input(shape=(None, ), name='order_hour_of_day_hist')


# layer 구성
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

# feature 투입
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

# 임베딩 벡터들 연결
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
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['acc'])

# 학습(training)
history = model.fit([tf.keras.preprocessing.sequence.pad_sequences(train_data['product_id']),
                     tf.keras.preprocessing.sequence.pad_sequences(train_data['order_dow']),
                     tf.keras.preprocessing.sequence.pad_sequences(train_data['order_hour_of_day']) #+ 1e-10, dtype=float
                    ],train_data['predict_labels'].values,
                  steps_per_epoch=1, epochs=300)


pickle.dump(model, open('model.pkl', 'wb'))