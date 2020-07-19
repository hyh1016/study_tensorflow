import tensorflow as tf
import numpy as np

# 데이터 #
data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])
########

global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 입력층 -> 은닉층1
# 텐서보드에서 한 계층 내부를 표현하는 name_scope
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))

# 입력층 -> 은닉층2
with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

# 은닉층2 -> 출력층
with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name='W3')
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    # 손실
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

    # 최적화
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    # 최적화 함수가 학습용 변수들을 최적화할 때마다 global_step 변수가 1씩 증가
    train_op = optimizer.minimize(cost, global_step=global_step)

    # 손실값(스칼라) 추적을 위해 수집할 값
    tf.summary.scalar('cost', cost)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

#
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

# 최적화 실행(학습)
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step: %d ' % sess.run(global_step),
          'Cost: %.3f ' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    # 값들의 수집 및 저장
    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))

saver.save(sess, './model/dnn.ckpt', global_step=global_step)

# 학습 결과 확인 (예측, 결과, 정확도)
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print("예측값: ", sess.run(prediction, feed_dict={X: x_data}))
print("실제값: ", sess.run(target, feed_dict={Y: y_data}))
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도: ", sess.run(accuracy*100, feed_dict={X: x_data, Y: y_data}))