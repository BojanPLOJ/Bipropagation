#**********************************************
#           B I P R O P A G A T I O N
#
# This is Demo App for Bipropagation algorithm
# written by Bojan PLOJ. This new algorithm and
# is much faster, much more accurate and much
# more reliable than his predecessor -
# Backpropagation algorithm.
# Learning is done layer by layer. Before the
# start of the learning we have to determine
# the values of inner layers which are no
# longer hidden. This could be done on several
# ways. Here we have choosen folowing way.
# Logical function XOR have two T values on
# the output, what is to much for NN with only
# one layer. So we take the additional layer
# with two neurons, one neuron for each T.
#
#  out     inner
#	F       F F
#	T   ->  T F
#	T       F T
#	F       F F
#
# This is only one example how to determine
# inner layers values. There are many other
# ways. Please look in to the book
# "Advance in the Machine Learning Research"
# - 3rd chapter (by Nova Publishers) for more.
#    
#  
#**********************************************



import tensorflow as tf

T, F = 1., -1.
bias = 1.


#**********************************************
#                 First  layer
#**********************************************
print("learning of the 1st layer")
train_in = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]

train_out = [
    [F,F],
    [T,F],
    [F,T],
    [F,F],
]

w = tf.Variable(tf.random_normal([3, 2]))

# step(x) = { 1 if x > 0; -1 otherwise }
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.mul(as_float, 2)
    return tf.sub(doubled, 1)

output = step(tf.matmul(train_in, w))
error = tf.sub(train_out, output)
mse = tf.reduce_mean(tf.square(error))

delta = tf.matmul(train_in, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

err, target = 1, 0
epoch, max_epochs = 0, 10
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
    print('epoch:', epoch, 'mse:', err)
print(sess.run(output))
#print(sess.run(w))
sess.close

#**********************************************
#                 Second  layer
#**********************************************

print("learning of the 2nd layer")

train_in = [
    [F,F,bias],
    [T,F,bias],
    [F,T,bias],
    [F,F,bias],
]

train_out = [
    [F],
    [T],
    [T],
    [F],
]

w = tf.Variable(tf.random_normal([3, 1]))

output = step(tf.matmul(train_in, w))
error = tf.sub(train_out, output)
mse = tf.reduce_mean(tf.square(error))

delta = tf.matmul(train_in, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

err, target = 1, 0
epoch, max_epochs = 0, 10
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
    print('epoch:', epoch, 'mse:', err)
print(sess.run(output))
#print(sess.run(w))
sess.close


