'''
Created on May 28, 2017

@author: iskandar
'''
import pickle
import tensorflow as tf;
import random;
import cv2;
import numpy as np;
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize;




#image manipulation functions
def sharpen_img(image):
    return cv2.addWeighted(image, 2, cv2.GaussianBlur(image, (5,5), 0), 0, 0)

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    gray = gray.reshape(32, 32, 1);
    return gray;

###Filters and Biases
def get_filter(layer):

    mu = 0
    sigma = 0.1    

    filters = {
        'f_1': tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 16), mean = mu, stddev = sigma)),
        'f_2': tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 36), mean = mu, stddev = sigma)),
        'f_3': tf.Variable(tf.truncated_normal(shape=(3, 3, 36, 49), mean = mu, stddev = sigma)),
        }
    return filters[layer];

def get_an(layer):

    mu = 0
    sigma = 0.1    

    filters = {
        'n_1': tf.Variable(tf.truncated_normal(shape=(196, 200), mean = mu, stddev = sigma)),
        'n_2': tf.Variable(tf.truncated_normal(shape=(200, 43), mean = mu, stddev = sigma))
    }

    return filters[layer];

def get_bias(layer):

    biases = {
        'b_1': tf.Variable(tf.zeros(16)),
        'b_2': tf.Variable(tf.zeros(36)),
        'b_3': tf.Variable(tf.zeros(49)),
        'b_4': tf.Variable(tf.zeros(200)),
        'b_5': tf.Variable(tf.zeros(43)),
    }
    return biases[layer];

def conv2d(x, W, b, strides=1, padding='VALID'):
    conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding) + b
    return tf.nn.relu(conv);

def maxpool2d(x, k=2, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding);

def fully_connected_layer(prev_layer, next_layer, bias):
    layer = tf.matmul(prev_layer, next_layer) + bias
    return tf.nn.relu(layer);


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)


def Classifier(x):
    
    clayer = conv2d(x, get_filter('f_1'), get_bias('b_1'));
    clayer = maxpool2d(clayer);
#    clayer = tf.nn.dropout(clayer, keep_prob=0.5);
    clayer = conv2d(clayer, get_filter('f_2'), get_bias('b_2'));
    clayer = maxpool2d(clayer);
    clayer = conv2d(clayer, get_filter('f_3'), get_bias('b_3'));
    clayer = maxpool2d(clayer);

    clayer = flatten(clayer);
    clayer = fully_connected_layer(clayer, get_an('n_1'), get_bias('b_4'));
    clayer = tf.nn.dropout(clayer, keep_prob);
    clayer = fully_connected_layer(clayer, get_an('n_2'), get_bias('b_5'));
    
    
    return clayer;

EPOCHS = 15;
BATCH_SIZE = 128;


training_file = 'traffic_data/train.p'
validation_file='traffic_data/valid.p'
testing_file = 'traffic_data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train);

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = 43

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



learning_rate = 0.001;

X_train_norm = np.zeros(shape=(X_train.shape[0], X_train.shape[1], X_train.shape[2], 3), dtype=np.float32);
X_valid_norm = np.zeros(shape=(X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], 3), dtype=np.float32);
X_test_norm = np.zeros(shape=(X_test.shape[0], X_test.shape[1], X_test.shape[2], 3), dtype=np.float32);

# index = random.randint(0, len(X_train));
# image = X_train[index].squeeze();
# 
# plt.imshow(image);
# print(y_train[index]);


for idx, img in enumerate(X_train):
    X_train[idx] = sharpen_img(img);

# for idx, img in enumerate(X_train):
#     X_train_norm[idx] = grayscale(img);

for idx, val in enumerate(X_train):
    X_train_norm[idx] = cv2.normalize(val, X_train_norm[idx], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F);

for idx, val in enumerate(X_valid):
    X_valid_norm[idx] = cv2.normalize(val, X_valid_norm[idx], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F);

for idx, val in enumerate(X_test):
    X_test_norm[idx] = cv2.normalize(val, X_test_norm[idx], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F);

# image = X_train_norm[index].squeeze();
# 
# plt.imshow(image);

classifier = Classifier(x);

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=classifier);
loss_operation = tf.reduce_mean(cross_entropy);
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate);
training_operation = optimizer.minimize(loss_operation);


correct_prediction = tf.equal(tf.argmax(classifier, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_sub, y_train_sub = shuffle(X_train_norm, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_sub[offset:end], y_train_sub[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
        validation_accuracy = evaluate(X_valid_norm, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()    

    test_accuracy = evaluate(X_test_norm, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

