import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

def sink(a, b, M, m_size, reg, numItermax=1000, stopThr=1e-9):
    # we assume that no distances are null except those of the diagonal of distances

    # a = tf.expand_dims(tf.ones(shape=(m_size[0],)) / m_size[0], axis=1)  # (na, 1)
    # b = tf.expand_dims(tf.ones(shape=(m_size[1],)) / m_size[1], axis=1)  # (nb, 1)

    # init data
    Nini = m_size[0]
    Nfin = m_size[1]

    u = tf.expand_dims(tf.ones(Nini) / Nini, axis=1)  # (na, 1)
    v = tf.expand_dims(tf.ones(Nfin) / Nfin, axis=1)  # (nb, 1)

    K = tf.exp(-M / reg)  # (na, nb)

    Kp = (1.0 / a) * K  # (na, 1) * (na, nb) = (na, nb)

    cpt = tf.constant(0)
    err = tf.constant(1.0)

    c = lambda cpt, u, v, err: tf.logical_and(cpt < numItermax, err > stopThr)

    def err_f1():
        # we can speed up the process by checking for the error only all the 10th iterations
        transp = u * (K * tf.squeeze(v))  # (na, 1) * ((na, nb) * (nb,)) = (na, nb)
        err_ = tf.pow(tf.norm(tensor=tf.reduce_sum(input_tensor=transp) - b, ord=1), 2)  # (,)
        return err_

    def err_f2():
        return err

    def loop_func(cpt, u, v, err):
        KtransposeU = tf.matmul(tf.transpose(a=K, perm=(1, 0)), u)  # (nb, na) x (na, 1) = (nb, 1)
        v = tf.compat.v1.div(b, KtransposeU)  # (nb, 1)
        u = 1.0 / tf.matmul(Kp, v)  # (na, 1)

        err = tf.cond(pred=tf.equal(cpt % 10, 0), true_fn=err_f1, false_fn=err_f2)

        cpt = tf.add(cpt, 1)
        return cpt, u, v, err

    _, u, v, _ = tf.while_loop(cond=c, body=loop_func, loop_vars=[cpt, u, v, err])

    result = tf.reduce_sum(input_tensor=u * K * tf.reshape(v, (1, -1)) * M)

    return result


