import tensorflow as tf
import numpy as np

# 데이터 파일로부터 데이터 받아옴
data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')

# 슬라이싱을 통해 입력값과 출력값으로 분리
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

####### 신경망 모델 구성 #######

# 학습 횟수 카운트를 위한 변수
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 이번에는 편향 생략
# 입력층 -> 은닉층1
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

# 입력층 -> 은닉층2
W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

# 은닉층2 -> 출력층
W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
# 출력층에서는 활성화 진행하지 않음
model = tf.matmul(L2, W3)

# 손실
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

# 최적화
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# 최적화 함수가 학습용 변수들을 최적화할 때마다 global_step 변수가 1씩 증가
train_op = optimizer.minimize(cost, global_step=global_step)

####### 학습 #######

sess = tf.Session()
# 모델 저장
# 앞서 정의한 모든 변수 반환하는 tf.global_variables()
saver = tf.train.Saver(tf.global_variables())

# 모델 재사용법: 모델 저장
# 기존에 학습한 모델 유무 확인 (체크포인트 파일 유무)
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    
    print('Step: %d ' % sess.run(global_step),
          'Cost: %.3f '% sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 학습 종료 후 학습된 변수들 체크포인트 파일에 저장
saver.save(sess, './model/dnn.ckpt', global_step=global_step)

# 학습 결과 확인 (예측, 결과, 정확도)
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print("예측값: ", sess.run(prediction, feed_dict={X: x_data}))
print("실제값: ", sess.run(target, feed_dict={Y: y_data}))
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도: ", sess.run(accuracy*100, feed_dict={X: x_data, Y: y_data}))