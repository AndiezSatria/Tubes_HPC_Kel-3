import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
import timeit        
from matplotlib import pyplot as plt
xaxis=[100, 500, 1000, 2500, 5000]
steps=3

#numpy
times_np=[]
for dim in xaxis:
    print (dim)
    temp_times_np=np.zeros(steps)
    for j in range(steps):
        #A=np.random.normal(size=(dim, dim))
        #B=np.random.normal(size=(dim, dim))
        start_time = timeit.default_timer()
        out=np.dot(np.random.normal(size=(dim, dim)),np.random.normal(size=(dim, dim)))
        temp_times_np[j]=timeit.default_timer() - start_time
    temp_time_np=np.sum(temp_times_np)/temp_times_np.shape[0]
    times_np.append(temp_time_np)
del out

#all tensorflow GPU
times_gpu=[]
for dim in xaxis:
    print (dim)
    temp_times_tf=np.zeros(steps)
    with tf.device('/gpu:0'):
        for j in range(steps):
            A=tf.Variable(tf.random.normal([dim, dim], stddev=0.35),name="weights")
            B=tf.Variable(tf.random.normal([dim, dim], stddev=0.35),name="weights2")
            out=tf.matmul(A,B)
            start_time = timeit.default_timer()
            with tf.compat.v1.Session() as sess:
            # define your variables and tensors
            # ... initialization code ...

                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(out)
            temp_times_tf[j]=timeit.default_timer() - start_time
            del A, B, out
        temp_time_tf=np.sum(temp_times_tf)/temp_times_tf.shape[0]
        times_gpu.append(temp_time_tf)

#all tensorflow CPU
times_cpu=[]
for dim in xaxis:
    print (dim)
    temp_times_tf=np.zeros(steps)
    with tf.device('/cpu:0'):
        for j in range(steps):
            A=tf.Variable(tf.random.normal([dim, dim], stddev=0.35),name="weights")
            B=tf.Variable(tf.random.normal([dim, dim], stddev=0.35),name="weights2")
            out=tf.matmul(A,B)
            start_time = timeit.default_timer()
            with tf.compat.v1.Session() as sess:
            # define your variables and tensors
            # ... initialization code ...

                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(out)
            temp_times_tf[j]=timeit.default_timer() - start_time
            del A, B, out
        temp_time_tf=np.sum(temp_times_tf)/temp_times_tf.shape[0]
        times_cpu.append(temp_time_tf)

#mixed: using TensorFlow with external generated matrices
times_mix=[]
for dim in xaxis:
    print (dim)
    temp_times_mix=np.zeros(steps)
    for j in range(steps):
        A=np.random.normal(size=(dim, dim))
        B=np.random.normal(size=(dim, dim))
        a=tf.compat.v1.placeholder("double")
        b=tf.compat.v1.placeholder("double")

        out=tf.matmul(a,b)

        #with tf.Session() as sess:
        sess=tf.compat.v1.Session()
            # define your variables and tensors
            # ... initialization code ...
        sess.run(out, feed_dict={a: A, b: B})
        temp_times_mix[j]=timeit.default_timer() - start_time
        del A,B, out
    temp_time_mix=np.sum(temp_times_mix)/temp_times_mix.shape[0]

    times_mix.append(temp_time_mix)

plt.plot(xaxis,times_np,label='Numpy, menggunakan CPU')
plt.plot(xaxis,times_cpu,label='TensorFlow, menggunakan CPU')
plt.plot(xaxis,times_gpu,label='TensorFlow, menggunakan GPU')
plt.plot(xaxis,times_mix,label='TensorFlow + Numpy')

plt.legend()
plt.ylabel('Execution time')
plt.xlabel('Matrix size')
plt.yscale('log')
plt.show()

