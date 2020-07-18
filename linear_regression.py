import tensorflow as tf

# y = 2x 의 선형 그래프
x_data = [1, 2, 3]
y_data = [2, 4, 6]

# 균등분포(uniform distribution)을 가진 무작위 변수
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# X, Y 라는 이름의 플레이스홀더 선언
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# 선형 관계 분석을 위한 수식
# hypothesis: Y(함수의 치역)에 해당
hypothesis = W * X + b

# 손실(cost) 계산
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 경사하강법 최적화 함수를 이용한 손실값 최소화
# learning_rate: 학습률. 얼마나 '급하게' 학습할 것인가? 클수록 속도가 빨라지지만 손실값 탐색이 어려워짐
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

# with절 내에서 세션이 실행되고 with절이 끝나면 자동으로 close()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 학습
    # 학습 횟수: 100번
    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, sess.run(W), sess.run(b))

    print("***결과***")
    print("X: {}, Y: {}".format(5, sess.run(hypothesis, feed_dict={X: 5})))
    print("X: {}, Y: {}".format(2.5, sess.run(hypothesis, feed_dict={X: 2.5})))