import tensorflow as tf

# read MNIST data (one-hot encording 방식으로)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# 신경망 구성

# 플레이스홀더 (입력값 28x28 pixel ==> 784, 출력값 0~9 ==> 10)
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# 은닉층 1
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

# 은닉층 2
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

# 출력층
W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

# 손실
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))

# 최적화
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 모델 초기화
init = tf.global_variables_initializer()

# 학습
sess = tf.Session()
sess.run(init)

# 미니배치 크기
batch_size = 100
# 미니배치의 총 개수
total_batch = int(mnist.train.num_examples / batch_size)

# 학습(에포크) 15회
for epoch in range(15):
    total_cost = 0

    # 내부에서는 미니배치 개수만큼 학습
    for i in range(total_batch):
        # 미니배치 크기만큼 학습할 데이터 가져옴
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})

        total_cost += cost_val

    print('Epoch:', '%04d'%(epoch+1),
          'Avg. cost =', '%.3f'%(total_cost/total_batch))

print("학습 완료")

# 정확도
# 두 번째 차원(1번 인덱스)에 결과값이 담겨 있음
is_corerct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_corerct, tf.float32))
print("정확도:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))