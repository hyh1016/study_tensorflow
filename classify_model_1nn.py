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

# 가중치
# random_uniform(자료형, 최소 범위, 최대 범위)
# 자료형은 [입력층(특징 수), 출력층(레이블 수)]
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
# 편향
# 레이블 수인 배열을 0으로 채운 [0, 0, 0] 으로 설정
b = tf.Variable(tf.zeros([3]))

# 인공신경망을 통과해 온 값 L에 활성화 함수 렐루(ReLU) 적용
L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)

# 확률로 해석하기 위해 배열 내 결괏값 합을 1로
model = tf.nn.softmax(L)

# 손실
# 교차 엔트로피 함수를 사용
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model), axis=1))

# 최적화
# 경사하강법 이용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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