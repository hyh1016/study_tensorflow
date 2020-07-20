import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# read MNIST data (one-hot encording 방식으로)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../mnist/data/", one_hot=True)

# 신경망 구성

# 플레이스홀더 (입력값 28x28 pixel ==> 784, 출력값 0~9 ==> 10)
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
# 학습 시에는 0.8
# 예측 시에는 1로 구성
keep_prob = tf.placeholder(tf.float32)

# 은닉층 1
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
# 드롭아웃: 사용할 뉴런 비율 0.8
L1 = tf.nn.dropout(L1, keep_prob)

# 은닉층 2
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)

# 출력층
W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

# 손실
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))

# 최적화
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 학습 시에는 뉴런의 0.8만 사용
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs,
                                                             Y: batch_ys,
                                                             keep_prob: 0.8})

        total_cost += cost_val

    print('Epoch:', '%04d'%(epoch+1),
          'Avg. cost =', '%.3f'%(total_cost/total_batch))

is_corerct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_corerct, tf.float32))
print("정확도:", sess.run(accuracy, feed_dict={X: mnist.test.images,
                                            Y: mnist.test.labels,
                                            keep_prob: 1}))

# 학습 결과 labels에 저장
labels = sess.run(model, feed_dict={X: mnist.test.images,
                                    Y: mnist.test.labels,
                                    keep_prob: 1})
# 손글씨를 출력할 그래프
fig = plt.figure()

# 예측값 출력
for i in range(10):
    # 2행 5열 그래프 생성. i+1번째에 테스트 이미지 출력
    subplot = fig.add_subplot(2, 5, i+1)
    # 이미지의 원본 출력을 위해 이미지 크기는 지정하지 않음
    subplot.set_xticks([])
    subplot.set_yticks([])
    # 출력한 이미지 위에 예측값 출력
    # 예측값: 해당 배열에서 가장 높은 값을 가진 인덱스
    subplot.set_title('%d'%np.argmax(labels[i]))
    # 이미지를 28x28 형식의 그레이스케일로 출력
    subplot.imshow(mnist.test.images[i].reshape((28, 28)), cmap=plt.cm.gray_r)

plt.show()
