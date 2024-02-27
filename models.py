import math
import tensorflow as tf


class BasicModel(object):
    def __init__(self, config, nent, nrel):
        super(BasicModel, self).__init__()
        self.config = config
        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])
        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        self.emb_ent = tf.Variable(tf.random_uniform([nent, config.emb_dim], -0.5, 0.5), name="ent_emb")
        self.emb_rel = tf.Variable(tf.random_uniform([nrel, config.emb_dim], -0.5, 0.5), name="rel_emb")

        pos_he = tf.nn.embedding_lookup(self.emb_ent, self.pos_h)
        pos_re = tf.nn.embedding_lookup(self.emb_rel, self.pos_r)
        pos_te = tf.nn.embedding_lookup(self.emb_ent, self.pos_t)

        neg_he = tf.nn.embedding_lookup(self.emb_ent, self.neg_h)
        neg_re = tf.nn.embedding_lookup(self.emb_rel, self.neg_r)
        neg_te = tf.nn.embedding_lookup(self.emb_ent, self.neg_t)

        pos_score = self.scoring_func(pos_he, pos_re, pos_te)
        neg_score = self.scoring_func(neg_he, neg_re, neg_te)

        # Margin loss
        self.loss = tf.reduce_sum(
            tf.maximum(tf.subtract(tf.add(pos_score, self.config.margin), neg_score), 0.))

        # Testing
        self.r_score = self.scoring_func(pos_he, pos_re, self.emb_ent)
        self.l_score = self.scoring_func(self.emb_ent, pos_re, pos_te)

    def scoring_func(self, h, r, t):
        raise NotImplementedError


class TransE(BasicModel):
    def __init__(self, config, nent, nrel):
        super(TransE, self).__init__(config, nent, nrel)

    def scoring_func(self, h, r, t):
        d = tf.subtract(tf.add(h, r), t)
        if "l1" in self.config.reg:
            return tf.reduce_sum(tf.abs(d), 1)
        else:  # l2
            return tf.reduce_sum(tf.square(d), 1)


class TorusE(BasicModel):
    def __init__(self, config, nent, nrel):
        super(TorusE, self).__init__(config, nent, nrel)

    def scoring_func(self, h, r, t):
        d = tf.subtract(tf.add(h, r), t)
        d = d - tf.floor(d)
        d = tf.minimum(d, 1.0 - d)
        if "el2" in self.config.reg:
            return tf.reduce_sum(2 - 2 * tf.cos(2 * math.pi * d), 1) / 4
        elif "l2" in self.config.reg:
            return 4 * tf.reduce_sum(tf.square(d), 1)
        else:  # l1
            return 2 * tf.reduce_sum(tf.abs(d), 1)
        
class MobiusE(BasicModel):
    def __init__(self, config, nent, nrel):
        super(MobiusE, self).__init__(config, nent, nrel)
        self.Radius_center = tf.constant(3.0, dtype=tf.float32); # 大圆半径
        self.Radius_ring = tf.constant(1.0, dtype=tf.float32); # 小圆半径

    def scoring_func(self, h, r, t):
        hr = tf.add(h,r);
        hr_theta, hr_w = tf.split(hr, num_or_size_splits=2,axis=1);
        t_theta, t_w = tf.split(t, num_or_size_splits=2,axis=1);
        hr_x = tf.multiply(tf.Radius_center + tf.multiply(tf.Radius_ring,tf.cos(hr_theta/2+hr_w)), tf.cos(hr_theta));
        hr_y = tf.multiply(tf.Radius_center + tf.multiply(tf.Radius_ring,tf.sin(hr_theta/2+hr_w)), tf.sin(hr_theta));
        hr_z = tf.multiply(tf.Radius_ring,tf.sin(hr_theta/2+hr_w));
        t_x = tf.multiply(tf.Radius_center + tf.multiply(tf.Radius_ring,tf.cos(t_theta/2+t_w)), tf.cos(t_theta));
        t_y = tf.multiply(tf.Radius_center + tf.multiply(tf.Radius_ring,tf.sin(t_theta/2+t_w)), tf.sin(t_theta));
        t_z = tf.multiply(tf.Radius_ring,tf.sin(t_theta/2+t_w));
        d = tf.concat([hr_x-t_x, hr_y-t_y, hr_z-t_z], 1);
        if "el2" in self.config.reg:
            return tf.reduce_sum(2 - 2 * tf.cos(2 * math.pi * d), 1) / 4
        elif "l2" in self.config.reg:
            return 4 * tf.reduce_sum(tf.square(d), 1)
        else:  # l1
            return 2 * tf.reduce_sum(tf.abs(d), 1)
