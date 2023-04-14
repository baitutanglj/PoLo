from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
# 安装tfds pip install tfds-nightly==1.0.2.dev201904090105
import tensorflow_datasets as tfds
import time

# 设置gpu内存自增长
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 导入数据
examples, metadata = tfds.load(name='ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True, download=False)
train_examples, val_examples = examples['train'], examples['validation']
tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)
tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)



def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size + 1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size + 1]
    return lang1, lang2


MAX_LENGTH = 40


def filter_long_sent(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


def tf_encode(pt, en):
    return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])


# 构造数据集
BUFFER_SIZE = 20000
BATCH_SIZE = 64

# 使用.map()运行相关图操作
train_dataset = train_examples.map(tf_encode)
# 过滤过长的数据
train_dataset = train_dataset.filter(filter_long_sent)
# 使用缓存数据加速读入
train_dataset = train_dataset.cache()
# 打乱并获取批数据
train_dataset = train_dataset.padded_batch(
    BATCH_SIZE, padded_shapes=([40], [40]))  # 填充为最大长度-90
# 设置预取数据
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 验证集数据
val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_long_sent).padded_batch(
    BATCH_SIZE, padded_shapes=([40], [40]))
de_batch, en_batch = next(iter(train_dataset))
print(de_batch, en_batch)


# 2.位置嵌入
def get_angles(pos, i, d_model):
    # 这里的i等价与上面公式中的2i和2i+1
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # 第2i项使用sin
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1项使用cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


pos_encoding = positional_encoding(50, 512)
print(pos_encoding.shape)
plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()  # 在这里左右边分别为原来2i 和 2i+1的特征


# 3.掩码
def create_padding_mark(seq):
    # 获取为0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 扩充维度以便用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :]  # (batch_size,1,1,seq_len)


# mark 测试
create_padding_mark([[1, 2, 0, 0, 3], [3, 4, 5, 0, 0]])


def create_look_ahead_mark(size):
    # 1 - 对角线和取下三角的全部对角线（-1->全部）
    # 这样就可以构造出每个时刻未预测token的掩码
    mark = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mark  # (seq_len, seq_len)


# x = tf.random.uniform((1,3))
temp = create_look_ahead_mark(3)
print(temp)


# 4.Scaled dot product attention
def scaled_dot_product_attention(q, k, v, mask):
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # attention 乘上value
    output = tf.matmul(attention_weights, v)  # （.., seq_len_v, depth）

    return output, attention_weights


# 使用attention获取需要关注的语义
def print_out(q, k, v):
    temp_out, temp_att = scaled_dot_product_attention(
        q, k, v, None)
    print('attention weight:')
    print(temp_att)
    print('output:')
    print(temp_out)


# attention测试
# 显示为numpy类型
np.set_printoptions(suppress=True)

temp_k = tf.constant([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype=tf.float32)  # (4, 3)

temp_v = tf.constant([[1, 0],
                      [10, 0],
                      [100, 5],
                      [1000, 6]], dtype=tf.float32)  # (4, 3)
# 关注第2个key, 返回对应的value
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)
print_out(temp_q, temp_k, temp_v)

# 关注重复的key(第3、4个), 返回对应的value（平均）
temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)
print_out(temp_q, temp_k, temp_v)

# 关注第1、2个key, 返回对应的value（平均）
temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)
print_out(temp_q, temp_k, temp_v)

temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
print_out(temp_q, temp_k, temp_v)


# 5.Mutil-Head Attention
# 构造mutil head attention层
class MutilHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MutilHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0
        # 分头后的维度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        # 合并多头
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights


# 测试多头attention
temp_mha = MutilHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))
output, att = temp_mha(y, k=y, q=y, mask=None)
print(output.shape, att.shape)


# point wise前向网络
def point_wise_feed_forward_network(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


sample_fnn = point_wise_feed_forward_network(512, 2048)
sample_fnn(tf.random.uniform((64, 50, 512))).shape


# 编码层
# 6.编码器和解码器
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, ddf, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MutilHeadAttention(d_model, n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, ddf)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask):
        # 多头注意力网络
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(inputs + att_output)  # (batch_size, input_seq_len, d_model)
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


# encoder层测试
sample_encoder_layer = EncoderLayer(512, 8, 2048)
sample_encoder_layer_output = sample_encoder_layer(
    tf.random.uniform((64, 43, 512)), False, None)
sample_encoder_layer_output.shape


# 解码层
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MutilHeadAttention(d_model, num_heads)
        self.mha2 = MutilHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)
        self.dropout3 = layers.Dropout(drop_rate)

    def call(self, inputs, encode_out, training,
             look_ahead_mask, padding_mask):
        # masked muti-head attention
        att1, att_weight1 = self.mha1(inputs, inputs, inputs, look_ahead_mask)
        att1 = self.dropout1(att1, training=training)
        out1 = self.layernorm1(inputs + att1)
        # muti-head attention
        att2, att_weight2 = self.mha2(encode_out, encode_out, inputs, padding_mask)
        att2 = self.dropout2(att2, training=training)
        out2 = self.layernorm2(out1 + att2)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.layernorm3(out2 + ffn_out)

        return out3, att_weight1, att_weight2


# 测试解码层
sample_decoder_layer = DecoderLayer(512, 8, 2048)
sample_decoder_layer_output, _, _ = sample_decoder_layer(
    tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
    False, None, None)
sample_decoder_layer_output.shape


# 编码器 编码器包含： - Input Embedding - Positional Embedding - N个编码层
class Encoder(layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ddf,
                 input_vocab_size, max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.d_model = d_model

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.encode_layer = [EncoderLayer(d_model, n_heads, ddf, drop_rate)
                             for _ in range(n_layers)]

        self.dropout = layers.Dropout(drop_rate)

    def call(self, inputs, training, mark):
        seq_len = inputs.shape[1]
        word_emb = self.embedding(inputs)
        word_emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb = word_emb + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(emb, training=training)
        for i in range(self.n_layers):
            x = self.encode_layer[i](x, training, mark)

        return x


# 编码器测试
sample_encoder = Encoder(2, 512, 8, 1024, 5000, 200)
sample_encoder_output = sample_encoder(tf.random.uniform((64, 120)), False, None)
sample_encoder_output.shape


# 解码器
# import pdb
# pdb.set_trace()
class Decoder(layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ddf,
                 target_vocab_size, max_seq_len, drop_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.decoder_layers = [DecoderLayer(d_model, n_heads, ddf, drop_rate)
                               for _ in range(n_layers)]

        self.dropout = layers.Dropout(drop_rate)

    def call(self, inputs, encoder_out, training, look_ahead_mark, padding_mark):
        seq_len = tf.shape(inputs)[1]
        attention_weights = {}
        h = self.embedding(inputs)
        h *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        h += self.pos_embedding[:, :seq_len, :]

        h = self.dropout(h, training=training)
        #         print('--------------------\n',h, h.shape)
        # 叠加解码层
        for i in range(self.n_layers):
            h, att_w1, att_w2 = self.decoder_layers[i](h, encoder_out,
                                                       training, look_ahead_mark,
                                                       padding_mark)
            attention_weights['decoder_layer{}_att_w1'.format(i + 1)] = att_w1
            attention_weights['decoder_layer{}_att_w2'.format(i + 1)] = att_w2

        return h, attention_weights


# 解码器测试
sample_decoder = Decoder(2, 512, 8, 1024, 5000, 200)
sample_decoder_output, attn = sample_decoder(tf.random.uniform((64, 100)),
                                             sample_encoder_output, False,
                                             None, None)
sample_decoder_output.shape, attn['decoder_layer1_att_w1'].shape


# 创建Transformer
# Transformer包含编码器、解码器和最后的线性层，解码层的输出经过线性层后得到Transformer的输出
class Transformer(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, diff,
                 input_vocab_size, target_vocab_size,
                 max_seq_len, drop_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_layers, d_model, n_heads, diff,
                               input_vocab_size, max_seq_len, drop_rate)

        self.decoder = Decoder(n_layers, d_model, n_heads, diff,
                               target_vocab_size, max_seq_len, drop_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, targets, training, encode_padding_mask, look_ahead_mask, decode_padding_mask):
        encode_out = self.encoder(inputs, training, encode_padding_mask)
        print(encode_out.shape)
        decode_out, att_weights = self.decoder(targets, encode_out, training,
                                               look_ahead_mask, decode_padding_mask)
        print(decode_out.shape)
        final_out = self.final_layer(decode_out)

        return final_out, att_weights

#Transformer测试
sample_transformer = Transformer(
n_layers=2, d_model=512, n_heads=8, diff=1024,
input_vocab_size=8500, target_vocab_size=8000, max_seq_len=120)
temp_input = tf.random.uniform((64, 62))
temp_target = tf.random.uniform((64, 26))
fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                               encode_padding_mask=None,
                               look_ahead_mask=None,
                               decode_padding_mask=None,)

fn_out.shape

#7.实验设置
#设置超参
num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2
max_seq_len = 40
dropout_rate = 0.1
#优化器
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
learing_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learing_rate, beta_1=0.9,
                                    beta_2=0.98, epsilon=1e-9)
# 测试
temp_learing_rate = CustomSchedule(d_model)
plt.plot(temp_learing_rate(tf.range(40000, dtype=tf.float32)))
plt.xlabel('learning rate')
plt.ylabel('train step')
plt.show()
#损失和指标
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                           reduction='none')

def loss_fun(y_ture, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_ture, 0))  # 为0掩码标1
    loss_ = loss_object(y_ture, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

#8、训练和保存模型
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          max_seq_len, dropout_rate)
# 构建掩码
def create_mask(inputs,targets):
    encode_padding_mask = create_padding_mark(inputs)
    # 这个掩码用于掩输入解码层第二层的编码层输出
    decode_padding_mask = create_padding_mark(inputs)

    # look_ahead 掩码， 掩掉未预测的词
    look_ahead_mask = create_look_ahead_mark(tf.shape(targets)[1])
    # 解码层第一层得到padding掩码
    decode_targets_padding_mask = create_padding_mark(targets)

    # 合并解码层第一层掩码
    combine_mask = tf.maximum(decode_targets_padding_mask, look_ahead_mask)

    return encode_padding_mask, combine_mask, decode_padding_mask
#创建checkpoint管理器
checkpoint_path = './checkpoint/train'
ckpt = tf.train.Checkpoint(transformer=transformer,
                          optimizer=optimizer)
# ckpt管理器
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('last checkpoit restore')


@tf.function
def train_step(inputs, targets):
    tar_inp = targets[:,:-1]
    tar_real = targets[:,1:]
    # 构造掩码
    encode_padding_mask, combined_mask, decode_padding_mask = create_mask(inputs, tar_inp)


    with tf.GradientTape() as tape:
        predictions, _ = transformer(inputs, tar_inp,
                                    True,
                                    encode_padding_mask,
                                    combined_mask,
                                    decode_padding_mask)
        loss = loss_fun(tar_real, predictions)
    # 求梯度
    gradients = tape.gradient(loss, transformer.trainable_variables)
    # 反向传播
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # 记录loss和准确率
    train_loss(loss)
    train_accuracy(tar_real, predictions)

#葡萄牙语用作输入语言，英语是目标语言。
EPOCHS = 20
for epoch in range(EPOCHS):
    start = time.time()

    # 重置记录项
    train_loss.reset_states()
    train_accuracy.reset_states()

    # inputs 葡萄牙语， targets英语

    for batch, (inputs, targets) in enumerate(train_dataset):
        # 训练
        train_step(inputs, targets)

        if batch % 500 == 0:
            print('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
            epoch+1, batch, train_loss.result(), train_accuracy.result()
            ))

    if (epoch + 1) % 2 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('epoch {}, save model at {}'.format(
        epoch+1, ckpt_save_path
        ))


    print('epoch {}, loss:{:.4f}, acc:{:.4f}'.format(
    epoch+1, train_loss.result(), train_accuracy.result()
    ))

    print('time in 1 epoch:{} secs\n'.format(time.time()-start))



import torch
from torch import nn

# 如可以解释成:4层的LSTM,输入的每个词用10维向量表示,隐藏单元和记忆单元的尺寸是6
lstm = nn.LSTM(input_size=10, hidden_size=6, num_layers=4, batch_first=True)

# 输入的x:其中batch是3可表示有三句话,seq_len=2表示每句话2个单词,feature_len=10表示每个单词表示为长10的向量
x = torch.randn(3, 2, 10)

# 前向计算过程,这里不传入h_0和C_0则会默认初始化
out, (h, c) = lstm(x)
print(out.shape)  # torch.Size([3, 2, 6]) 最后一层所有时刻2的输出
print(h.shape)  # torch.Size([4, 3, 6]) 隐藏单元, 4层layer，每层batch3的最后一个时刻的h， h[-1, 0]=out[0, -1]
print(c.shape)  # torch.Size([4, 3, 6]) 记忆单元, 4层layer，每层batch3的最后一个时刻的c

rnn = nn.LSTM(10, 6, 2, batch_first=True)
input = torch.randn(3, 1, 10)
h = torch.randn(2, 3, 6)
c = torch.randn(2, 3, 6)
for i in range(3):
    output, (h, c) = rnn(input, (h, c))



import torch
from torch import nn

# 输入的feature_len=100,变到该层隐藏单元和记忆单元hidden_len=30
cell_l0 = nn.LSTMCell(input_size=100, hidden_size=30)
# hidden_len从l0层的30变到这一层的20
cell_l1 = nn.LSTMCell(input_size=30, hidden_size=20)

# 分别初始化l0层和l1层的隐藏单元h和记忆单元C,取batch=3
# 注意l0层的hidden_len=30
h_l0 = torch.zeros(3, 30)
C_l0 = torch.zeros(3, 30)
# 而到l1层之后hidden_len=20
h_l1 = torch.zeros(3, 20)
C_l1 = torch.zeros(3, 20)

# 这里是seq_len=10个时刻的输入,每个时刻shape都是[batch,feature_len]
xs = [torch.randn(3, 100) for _ in range(10)]

# 对每个时刻,从下到上计算本时刻的所有层
for xt in xs:
    h_l0, C_l0 = cell_l0(xt, (h_l0, C_l0))  # l0层直接接受xt输入
    h_l1, C_l1 = cell_l1(h_l0, (h_l1, C_l1))  # l1层接受l0层的输出h为输入

# 最后shape是不变的
print(h_l0.shape)  # torch.Size([3, 30])
print(C_l0.shape)  # torch.Size([3, 30])
print(h_l1.shape)  # torch.Size([3, 20])
print(C_l1.shape)  # torch.Size([3, 20])

#############
layer_num=2
input_size=100
hidden_size=30
policy_step = nn.Sequential()
policy_step.add_module(f"LSTMCell_{0}", nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
if layer_num != 1:
    for i in range(1, layer_num):
        policy_step.add_module(f"LSTMCell_{i}", nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size))

# 这里是seq_len=10个时刻的输入,每个时刻shape都是[batch,feature_len]
xs = [torch.randn(3, 100) for _ in range(10)]
# 分别初始化l0层和l1层的隐藏单元h和记忆单元C,取batch=3
# 注意l0层的hidden_len=30
h_l0 = torch.zeros(3, hidden_size)
C_l0 = torch.zeros(3, hidden_size)
# 而到l1层之后hidden_len=20
h_l1 = torch.zeros(3, hidden_size)
C_l1 = torch.zeros(3, hidden_size)
state = [(h_l0, C_l0), (h_l1, C_l1)]
for x in xs:
    state_new = []
    for idx, cell in enumerate(policy_step):
        h, c = cell(x, state[idx])
        x = h
        state_new.append((h,c))


############




import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
batch_size = 3
depth = 10
inputs = tf.Variable(tf.random.normal([batch_size, depth]))
previous_state0 = (tf.random.normal([batch_size, 4]), tf.random.normal([batch_size, 4]))
previous_state1 = (tf.random.normal([batch_size, 5]), tf.random.normal([batch_size, 5]))
previous_state2 = (tf.random.normal([batch_size, 6]), tf.random.normal([batch_size, 6]))
num_units = [4, 5, 6]
print(inputs.shape)
cells = [tf.nn.rnn_cell.BasicLSTMCell(num_unit) for num_unit in num_units]
mul_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
outputs, states = mul_cells(inputs, (previous_state0, previous_state1, previous_state2))
print(outputs.shape)  # (3, 6)
print(states[0])  # 第一层LSTM
print(states[1])  # 第二层LSTM
print(states[2])  ##第三层LSTM
print(states[2].h)  ##第三层LSTM
print(states[0].h.shape)  # 第一层LSTM的h状态,(3, 6)
print(states[0].c.shape)  # 第一层LSTM的c状态,(3, 6)
print(states[1].h.shape)  # 第二层LSTM的h状态,(3, 5)
print(states[1].c.shape)  # 第二层LSTM的h状态,(3, 5)
print(states[2].h.shape)  # 第二层LSTM的h状态,(3, 5)
print(states[2].c.shape)  # 第二层LSTM的h状态,(3, 5)




import torch
a = torch.randn((2,3,4))
b = torch.split(a, 1, dim=0)


import csv
with open(input_file) as raw_input_file:
    csv_file = csv.reader(raw_input_file, delimiter='\t')
    for line in csv_file:
        e1, r, e2 = line[0], line[1], line[2]


import os
import json
os.chdir('/home/linjie/projects/KG/PoLo/datasets/Hetionet_CmC/preprocessing')
with open("metapaths_p3.json", 'r') as f:
    metapaths = json.load(f)

metapaths_inv = []
for mp in metapaths.copy():
    for idx, p in enumerate(mp):
        if idx % 2 == 1:
            if '_' in p:
                p_ = p.replace('_', '')
            else:
                p_ = '_'+p
            mp[idx] = p_
    mp = mp[::-1]
    metapaths_inv.append(mp)
with open("metapaths_p3_inv.json", 'w')as f:
    json.dump(metapaths_inv, f)



policy_step = nn.Sequential()
for i in range(2):
    policy_step.add_module(f"LSTMCell_{i}", nn.LSTMCell(input_size=5, hidden_size=5))

x = torch.randn(3,4,5)
init_states = []
for layer in range(2):
    states_c = torch.zeros((3, 5), dtype=torch.float32)
    states_h = torch.zeros((3, 5), dtype=torch.float32)
    init_states.append((states_c, states_h))
prev_states = init_states
output = []
for entity_idx in range(x.shape[1]):
    input = x[:, i, :]
    new_states = []
    for i, policy_layer in enumerate(policy_step):
        h_tmp, c_tmp = policy_layer(input, prev_states[i])
        new_states.append((h_tmp, c_tmp))
    prev_states = new_states
    output.append(h_tmp)
output = torch.stack(output, dim=1)

policy_MLP = nn.Sequential(nn.Linear(5,6),
                        nn.BatchNorm1d(6),
                        nn.ReLU(),
                        nn.Linear(6,6),
                        nn.BatchNorm1d(6),
                        nn.ReLU())
output_ = torch.reshape(output, [-1, output.shape[-1]])
output = policy_MLP(output_)
output = torch.reshape(output, [-1, output.shape[-1]])