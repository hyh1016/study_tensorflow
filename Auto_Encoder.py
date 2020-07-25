# 비지도학습
# Y값 없음. X값(input)만 주어짐

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 데이터 import
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# 하이퍼파라미터
learning_rate = 0.01 # 학습률
training_epoch = 20 # 총 학습 횟수
batch_size = 100 # 미니배치 한 번에 학습할 데이터 개수
n_hidden = 256 # 은닉층의 뉴런 개수
n_input = 28 * 28 # 입력값의 크기 (784)

################
### 신경망 구성 ###
###############

# 플레이스홀더
X = tf.placeholder(tf.float32, [None, n_input])

# 입력층 -> 은닉층: 인코더(encoder)
# 인코더의 가중치와 편향
W_encoder = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encoder = tf.Variable(tf.random_normal([n_hidden]))
# 인코더
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encoder), b_encoder))

# 은닉층 -> 출력층: 디코더(decoder)
# 디코더의 가중치와 편향
W_decoder = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decoder = tf.Variable(tf.random_normal([n_input]))
# 디코더
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decoder), b_decoder))

# 손실
# 실제 입력값(X)과 예측값인 decoder가 반환하는 값(decoder) 사이의 거리 함수로 구현
cost = tf.reduce_mean(tf.pow(X- decoder, 2))

# 최적화
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

###########
### 학습 ###
##########

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(training_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})

        total_cost += cost_val

    print('Epoch:', '%04d '%(epoch+1),
          'Avg. cost =', '%.4f'%(total_cost/total_batch))

# matplotlib을 통한 결과 확인
sample_size = 10
samples = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})
fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

# 출력
# 위(ax[0][i]): 입력값 이미지
# 아래(ax[1][i]): 신경망이 생성한 이미지
for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

plt.show()