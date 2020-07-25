import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 데이터 import
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# 하이퍼파라미터
total_epoch = 100 # 총 학습 횟수
batch_size = 100 # 미니배치 한 번에 학습할 데이터 개수
learning_rate = 0.0002 # 학습률
n_hidden = 256 # 은닉층의 뉴런 개수
n_input = 28 * 28 # 입력값의 크기 (784)
n_noise = 128 # 생성자의 입력값인 노이즈의 크기

# 플레이스홀더
# 입력값 X, 노이즈로 입력될 Z
X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

# 생성자 신경망 변수
# 노이즈(입력층) -> 은닉층
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
# 은닉층 -> 구분자의 입력층
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 구분자 신경망 변수
# 입력층 -> 은닉층
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
# 은닉층 -> 출력층(0~1 사이의 수)
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))


# 생성자 신경망
# 파라미터로 무작위로 생성한 노이즈를 받아 구분자의 입력 데이터
def generator(noise_z):
    hidden = tf.add(tf.matmul(noise_z, G_W1), G_b1)
    hidden = tf.nn.relu(hidden)
    output = tf.add(tf.matmul(hidden, G_W2), G_b2)
    output = tf.nn.sigmoid(output)
    return output

# 구분자 신경망
# 입력을 받아 0~1 사이의 값을 출력
def discriminator(inputs):
    hidden = tf.add(tf.matmul(inputs, D_W1), D_b1)
    hidden = tf.nn.relu(hidden)
    output = tf.add(tf.matmul(hidden, D_W2), D_b2)
    output = tf.nn.sigmoid(output)
    return output

# 무작위 노이즈 생성
def getNoise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

# fake image를 만들 G
G = generator(Z)
# fake image를 받을 D_gene
D_gene = discriminator(G)
# real image를 받을 D_real
D_real = discriminator(X)

# 손실
# 손실1: 경찰 학습용, D_real(실제) + 1-D_gene(가짜) 값을 최대화
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1- D_gene))
# 손실2: 위조지폐범 학습용, D_gene(가짜) 값을 최대화
loss_G = tf.reduce_mean(tf.log(D_gene))

# ★생성자/구분자의 변수 분리 중요 (하나 학습할 때 나머지 하나가 변하면 안 됨)
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# 최적화
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

# 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 미니배치
total_batch = int(mnist.train.num_examples / batch_size)
# 두 손실의 결과값을 담을 변수
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = getNoise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

    print('Epoch:', '%04d'%epoch,
          'D loss:', '%.4f'%loss_val_D,
          'G loss:', '%.4f'%loss_val_G)

    # 10번에 한 번 꼴로 확인
    if epoch == 0 or (epoch+1)%10 == 0:
        sample_size = 10
        noise = getNoise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('학습 완료')
