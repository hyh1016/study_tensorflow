import tensorflow as tf
import numpy as np

# 1행 1열: 털 유무
# 1행 2열: 날개 유무
x_data = [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]

# label data(레이블 데이터)
# 기타 = [1, 0, 0] 포유류 = [0, 1, 0] 조류 = [0, 0, 1]
# [0, 0] = [1, 0, 0], [1, 0] = [0, 1, 0], [1, 1] = [0, 0, 1]
y_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# in, out
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

########## 심층 신경망 ##########
# 가중치
# 은닉 층의 뉴런 수를 10 개로 설정
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))

# 편향
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

# 은닉층을 거쳐 생성되는 출력 L1
L1 = tf.add(tf.matmul(X, W1), b1)
# 활성화 (은닉층은 활성화를 주로 사용함)
L1 = tf.nn.relu(L1)

# L1을 다시 출력층을 통과시켜 최종 출력값인 model을 도출
# 출력층에서는 활성화 함수를 잘 사용하지 않음
model = tf.add(tf.matmul(L1, W2), b2)

# 확률로 해석하기 위해 배열 내 결괏값 합을 1로 만듦
model = tf.nn.softmax(model)

# 손실
# 교차 엔트로피 사용
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model), axis=1))
# 텐서플로에서 제공하는 교차 엔트로피 손실 함수 사용
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

# 최적화
# 경사하강법 말고 아담 최적화 방법을 이용해 봄
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 학습
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    # 학습 중 10 번에 1 번씩 손실값 출력
    if (step + 1) % 10 == 0:
        print(step+1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 학습 결과 확인
# argmax: 리스트 내 요소 중 가장 큰 값의 인덱스를 반환
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
# 출력
print("예측값:", sess.run(prediction, feed_dict={X: x_data}))
print("실제값:", sess.run(target, feed_dict={Y: y_data}))

# 정확도 확인
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도:%.2f"%sess.run(accuracy*100, feed_dict={X: x_data, Y: y_data}))