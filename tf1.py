# Example of TensorFlow library
import tensorflow as tf
# declare two symbolic floating-point scalars
a = tf.Variable(1.5)
b = tf.Variable(2.5)
# create a simple symbolic expression using the add function
c = tf.add(a, b)
print(c)
