import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Activation,Conv2D,Conv2DTranspose,UpSampling2D,Concatenate,Add
import os

class GANILLA(tf.keras.Model):
    def __init__(self,g_G,g_F,d_G,d_F):
        super(GANILLA, self).__init__()
        self.g_G = g_G
        self.g_F = g_F
        self.d_G = d_G
        self.d_F = d_F



    def compile(self,):
        super(GANILLA, self).compile()
        self.cycle_loss = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss = tf.keras.losses.MeanAbsoluteError()
        self.gen_G_optimizer = self.g_G.optimizer
        self.gen_F_optimizer = self.g_F.optimizer
        self.dis_G_optimizer = self.d_G.optimizer
        self.dis_F_optimizer = self.d_F.optimizer
        self.gen_loss = self.g_G.loss
        self.dis_loss = self.d_G.loss

    def train_step(self, data):
        lambda_cycle = 10.0
        lambda_identity = 0.5
        real_photo,real_illus=data
        with tf.GradientTape(persistent=True) as tape:
            # generator net
            fake_G = self.g_G(real_photo)
            fake_F = self.g_F(real_illus)

            # discriminator
            disc_real_F = self.d_F(real_photo, training=True)
            disc_fake_F = self.d_F(fake_F, training=True)
            disc_real_G = self.d_G(real_illus, training=True)
            disc_fake_G = self.d_G(fake_G, training=True)

            # gen_loss
            g_G_loss = self.gen_loss(disc_fake_G)
            g_F_loss = self.gen_loss(disc_fake_F)

            # gen cycle loss
            cycled_F = self.g_F(fake_G)
            cycled_G = self.g_G(fake_F)
            cycle_loss_G = self.cycle_loss(real_illus, cycled_G) * lambda_cycle
            cycle_loss_F = self.cycle_loss(real_photo, cycled_F) * lambda_cycle

            # gen identity loss
            same_F = self.g_F(real_photo)
            same_G = self.g_G(real_illus)
            id_loss_G = (self.identity_loss(real_illus, same_G) * lambda_cycle * lambda_identity)
            id_loss_F = (self.identity_loss(real_photo, same_F) * lambda_cycle * lambda_identity)

            # gen loss
            loss_G = g_G_loss + cycle_loss_G + id_loss_G
            loss_F = g_F_loss + cycle_loss_F + id_loss_F

            # dis loss
            dis_X_loss = self.dis_loss(disc_real_F, disc_fake_F)
            dis_Y_loss = self.dis_loss(disc_real_G, disc_fake_G)

        grads_G = tape.gradient(loss_G, self.g_F.trainable_variables)
        grads_F = tape.gradient(loss_F, self.g_G.trainable_variables)

        dis_F_grads = tape.gradient(dis_X_loss, self.d_F.trainable_variables)
        dis_G_grads = tape.gradient(dis_Y_loss, self.d_G.trainable_variables)

        self.gen_G_optimizer.apply_gradients(zip(grads_G, self.g_G.trainable_variables))
        self.gen_F_optimizer.apply_gradients(zip(grads_F, self.g_F.trainable_variables))

        self.dis_F_optimizer.apply_gradients(zip(dis_F_grads, self.d_F.trainable_variables))
        self.dis_G_optimizer.apply_gradients(zip(dis_G_grads, self.d_G.trainable_variables))

        return {
            "gen_G_loss": loss_G,
            "gen_F_loss": loss_F,
            "dis_F_loss": dis_X_loss,
            "dis_G_loss": dis_Y_loss,
        }


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator,self).__init__()
        self.gamma = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.kernel = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.downsampling=Conv2D(64, (7, 7), padding="same", kernel_initializer=self.kernel, use_bias=False)
        self.InstanceNormalization1=tfa.layers.InstanceNormalization(gamma_initializer=self.gamma)
        self.relu1 = tf.keras.layers.ReLU()





        ##first resnet,first block
        self.resnet1_a_1=Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel, use_bias=False)
        self.InstanceNormalization2 = tfa.layers.InstanceNormalization(gamma_initializer=self.gamma)
        self.relu2 = tf.keras.layers.ReLU()
        self.resnet1_a_2 = Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel,use_bias=False)
        self.InstanceNormalization3 = tfa.layers.InstanceNormalization(gamma_initializer=self.gamma)

        ##first resnet, second block
        self.resnet1_b_1 = Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel, use_bias=False)
        self.InstanceNormalization4 = tfa.layers.InstanceNormalization(gamma_initializer=self.gamma)
        self.relu3 = tf.keras.layers.ReLU()
        self.resnet1_b_2 = Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel, use_bias=False)
        self.InstanceNormalization5 = tfa.layers.InstanceNormalization(gamma_initializer=self.gamma)


        ##second resnet,first block
        self.resnet2_a_1 = Conv2D(64, (3, 3),strides=(2,2), padding="same", kernel_initializer=self.kernel, use_bias=False)
        self.InstanceNormalization6 = tfa.layers.InstanceNormalization(gamma_initializer=self.gamma)
        self.relu4 = tf.keras.layers.ReLU()
        self.resnet2_a_2 = Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel,use_bias=False)
        self.InstanceNormalization7 = tfa.layers.InstanceNormalization(gamma_initializer=self.gamma)

        ##second resnet,second block
        self.resnet2_b_1 = Conv2D(128, (3, 3), padding="same", kernel_initializer=self.kernel,use_bias=False)
        self.InstanceNormalization8 = tfa.layers.InstanceNormalization(gamma_initializer=self.gamma)
        self.relu5 = tf.keras.layers.ReLU()
        self.resnet2_b_2 = Conv2D(128, (3, 3), padding="same", kernel_initializer=self.kernel, use_bias=False)
        self.InstanceNormalization9 = tfa.layers.InstanceNormalization(gamma_initializer=self.gamma)

        self.resnet2_a_skip = Conv2D(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer=self.kernel,use_bias=False)
        self.resnet1_a_last = Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel, use_bias=False)
        self.resnet1_b_last = Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel, use_bias=False)
        self.resnet2_a_last = Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel, use_bias=False)
        self.resnet2_b_last = Conv2D(128, (3, 3), padding="same", kernel_initializer=self.kernel, use_bias=False)
        ##upsampling
        self.upsampling1=Conv2DTranspose(256, kernel_size=(1,1), padding="same", kernel_initializer=self.kernel, use_bias=False)
        self.upsampling2=UpSampling2D(size=(2,2), interpolation="nearest")
        self.upsampling3=Conv2D(128, kernel_size=(1,1), padding="same", kernel_initializer=self.kernel, use_bias=False)

        self.upsampling_skip=Conv2DTranspose(128, kernel_size=(1,1), padding="same", kernel_initializer=self.kernel, use_bias=False)

        self.upsampling4=Conv2DTranspose(64, kernel_size=(1,1), padding="same", kernel_initializer=self.kernel, use_bias=False)
        self.InstanceNormalization10 = tfa.layers.InstanceNormalization(gamma_initializer=self.gamma)
        self.relu6 = tf.keras.layers.ReLU()
        self.upsampling5=Conv2DTranspose(3, kernel_size=(7, 7), padding="same", kernel_initializer=self.kernel,use_bias=False)
        self.activation = Activation("tanh")

    def call(self,inputs):
        ##downsampling
        downsampling=self.downsampling(inputs)
        downsampling=self.InstanceNormalization1(downsampling)
        downsampling=self.relu1(downsampling)

        skip = tf.identity(downsampling)
        ##resnet1
        resnet1=self.resnet1_a_1(downsampling)

        resnet1=self.InstanceNormalization2(resnet1)
        resnet1=self.relu2(resnet1)
        resnet1=self.resnet1_a_2(resnet1)

        resnet1 = self.InstanceNormalization3(resnet1)


        resnet1 = Concatenate()([skip, resnet1])
        resnet1=self.resnet1_a_last(resnet1)
            # skip layer
        resnet1a=tf.identity(resnet1)

        resnet1b=self.resnet1_b_1(resnet1)

        resnet1b=self.InstanceNormalization4(resnet1b)
        resnet1b=self.relu3(resnet1b)
        resnet1b=self.resnet1_b_2(resnet1b)

        resnet1b=self.InstanceNormalization5(resnet1b)

        resnet1 = Concatenate()([resnet1a, resnet1b])
        resnet1=self.resnet1_b_last(resnet1)
            #skip layer
        resnet1b = tf.identity(resnet1)

        ##resnet2
        resnet2a=self.resnet2_a_1(resnet1)

        resnet2a=self.InstanceNormalization6(resnet2a)
        resnet2a=self.relu4(resnet2a)
        resnet2a=self.resnet2_a_2(resnet2a)

        resnet2a=self.InstanceNormalization7(resnet2a)


        resnet1bM=self.resnet2_a_skip(resnet1b)
        resnet2a = Concatenate()([resnet1bM, resnet2a])
        resnet2a=self.resnet2_a_last(resnet2a)
            ##skip layer
        resnet2a_skip=tf.identity(resnet2a)

        resnet2b=self.resnet2_b_1(resnet2a)
        resnet2b=self.InstanceNormalization8(resnet2b) ##
        resnet2b=self.relu5(resnet2b)
        resnet2b=self.resnet2_b_2(resnet2b)

        resnet2b=self.InstanceNormalization9(resnet2b)

        resnet2b = Concatenate()([resnet2a_skip, resnet2b])
        resnet2b=self.resnet2_b_last(resnet2b)

        ##upsampling

        upsampling=self.upsampling1(resnet2b)

        upsampling = self.upsampling2(upsampling)

        upsampling = self.upsampling3(upsampling)

        upsampling_skip=self.upsampling_skip(resnet1b)
        upsampling = Add()([upsampling_skip, upsampling])

        upsampling=self.upsampling4(upsampling)
        upsampling=self.InstanceNormalization10(upsampling)
        upsampling=self.relu6(upsampling)
        upsampling=self.upsampling5(upsampling)
        upsampling=self.activation(upsampling)
        return upsampling

    def loss(self,fake):
        bce=tf.keras.losses.BinaryCrossentropy()
        return bce(tf.ones_like(fake), fake)
#
# class Generator(tf.keras.Model):
#     def __init__(self, name=None):
#         super(Generator, self).__init__()
#         self.gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
#         self.kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
#         self.downsampling = [
#             Conv2D(64, (7, 7), kernel_initializer=self.kernel_init, use_bias=False, padding="same", name="down_conv1"),
#             InstanceNormalization(gamma_initializer=self.gamma_init, name="down_instancenorm"),
#             layers.Activation("relu")
#         ]
#
#         self.resnet1a = [
#             Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=self.kernel_init,
#                    use_bias=False),
#             InstanceNormalization(gamma_initializer=self.gamma_init),
#             ReLU(),
#             layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=self.kernel_init,
#                           use_bias=False),
#             InstanceNormalization(gamma_initializer=self.gamma_init)
#         ]
#
#         self.resnet1b = [
#             Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=self.kernel_init,
#                    use_bias=False),
#             InstanceNormalization(gamma_initializer=self.gamma_init),
#             ReLU(),
#             layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=self.kernel_init,
#                           use_bias=False),
#             InstanceNormalization(gamma_initializer=self.gamma_init)
#         ]
#
#         self.resnet2a = [
#             Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=self.kernel_init,
#                    use_bias=False),
#             InstanceNormalization(gamma_initializer=self.gamma_init),
#             ReLU(),
#             layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=self.kernel_init,
#                           use_bias=False),
#             InstanceNormalization(gamma_initializer=self.gamma_init)
#         ]
#
#         self.resnet2b = [
#             Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=self.kernel_init,
#                    use_bias=False),
#             InstanceNormalization(gamma_initializer=self.gamma_init),
#             ReLU(),
#             layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=self.kernel_init,
#                           use_bias=False),
#             InstanceNormalization(gamma_initializer=self.gamma_init)
#         ]
#
#         self.skip_mod = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same",
#                                kernel_initializer=self.kernel_init, use_bias=False)
#         self.final_1a = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same",
#                                kernel_initializer=self.kernel_init, use_bias=False)
#         self.final_1b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same",
#                                kernel_initializer=self.kernel_init, use_bias=False)
#         self.final_2a = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same",
#                                kernel_initializer=self.kernel_init, use_bias=False)
#         self.final_2b = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same",
#                                kernel_initializer=self.kernel_init, use_bias=False)
#
#         self.upsampling_beg = [
#             Conv2DTranspose(256, kernel_size=(1, 1), padding="same", kernel_initializer=self.kernel_init,
#                             use_bias=False),
#             UpSampling2D(size=(2, 2), interpolation="nearest"),
#             Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding="same", kernel_initializer=self.kernel_init,
#                    use_bias=False)
#         ]
#
#         self.upsampling_mod = Conv2DTranspose(128, kernel_size=(1, 1), padding="same",
#                                               kernel_initializer=self.kernel_init, use_bias=False)
#
#         self.upsampling_end = [
#             Conv2DTranspose(64, kernel_size=(1, 1), strides=(1, 1), padding="same", kernel_initializer=self.kernel_init,
#                             use_bias=False),
#             InstanceNormalization(gamma_initializer=self.gamma_init),
#             ReLU(),
#             Conv2DTranspose(3, kernel_size=(7, 7), strides=(1, 1), padding="same", kernel_initializer=self.kernel_init,
#                             use_bias=False),
#             Activation("tanh")
#         ]
#
#     def call(self, x):
#         # print("----downsampling---")
#         for l in self.downsampling:
#             # print("x", x.shape)
#             x = l(x)
#
#         original = tf.identity(x)
#         # print("---- resnet1 ---")
#         for l in self.resnet1a:
#             # print("x", x.shape)
#             x = l(x)
#         x = Concatenate()([original, x])
#         x = self.final_1a(x)
#         layer1a = tf.identity(x)
#         for l in self.resnet1b:
#             # print("x", x.shape)
#             x = l(x)
#         x = Concatenate()([layer1a, x])
#         x = self.final_1b(x)
#
#         layer1b = tf.identity(x)
#         # print("---- resnet2 ---")
#         for l in self.resnet2a:
#             # print("x", x.shape)
#             x = l(x)
#         layer1b_mod = self.skip_mod(layer1b)
#         x = Concatenate()([layer1b_mod, x])
#
#         x = self.final_2a(x)
#         layer2a = tf.identity(x)
#         for l in self.resnet2b:
#             # print("x", x.shape)
#             x = l(x)
#         x = Concatenate()([layer2a, x])
#
#         x = self.final_2b(x)
#         # print("------ upsampling -- ")
#         for l in self.upsampling_beg:
#             # print("x", x.shape)
#             x = l(x)
#         layer_1_up = self.upsampling_mod(layer1b)
#         x = Add()([layer_1_up, x])
#         for l in self.upsampling_end:
#             # print("x", x.shape)
#             x = l(x)
#
#         return x
#
#     def loss(self, fake):
#         bce=tf.keras.losses.BinaryCrossentropy()
#         return bce(tf.ones_like(fake), fake)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        kernal_initialize = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        gamma = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)


        self.conv1=Conv2D(64, 4, strides=2, padding='same',kernel_initializer= kernal_initialize)
        self.leakyrelu1=tf.keras.layers.LeakyReLU()

        self.conv2=Conv2D(128, 4, strides=2, padding='same', kernel_initializer= kernal_initialize, use_bias=False)
        self.InstanceNormalization1=tfa.layers.InstanceNormalization(gamma_initializer=gamma)
        self.leakyrelu2=tf.keras.layers.LeakyReLU()

        self.conv3=Conv2D(256, 4, strides=2, padding='same',kernel_initializer= kernal_initialize, use_bias=False)
        self.InstanceNormalization2=tfa.layers.InstanceNormalization(gamma_initializer=gamma)
        self.leakyrelu3=tf.keras.layers.LeakyReLU()

        self.conv4=Conv2D(512, 4, strides=1, padding='same',kernel_initializer= kernal_initialize, use_bias=False)
        self.InstanceNormalization3=tfa.layers.InstanceNormalization(gamma_initializer=gamma)
        self.leakyrelu4=tf.keras.layers.LeakyReLU()

        self.conv5=Conv2D(1, 4, strides=1, padding='same',activation = 'sigmoid',kernel_initializer= kernal_initialize)


    def call(self, inputs):


        conv1=self.conv1(inputs)
        conv1=self.leakyrelu1(conv1)

        conv2=self.conv2(conv1)
        conv2=self.InstanceNormalization1(conv2)
        conv2=self.leakyrelu2(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.InstanceNormalization2(conv3)
        conv3 = self.leakyrelu3(conv3)

        conv4 = self.conv4(conv3)
        conv4 = self.InstanceNormalization3(conv4)
        conv4 = self.leakyrelu4(conv4)

        conv5 = self.conv5(conv4)

        return conv5

    def loss(self,real,fake):
        bce=tf.keras.losses.BinaryCrossentropy()
        return (bce(tf.ones_like(real), real)+bce(tf.zeros_like(fake), fake))/2


class Checkpoints(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, max_num_weights=5):
        super(Checkpoints, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.max_num_weights = max_num_weights
    def on_epoch_end(self, epoch, logs=None):
        print("on epoch end", epoch)
        epoch_str = epoch + 1

        if (epoch_str % 5 == 0):
            # make directory named timestamp/epoch_{epoch_str}
            if (epoch_str < 10):
                new_dir = "epoch_" + "0" + str(epoch_str)
            else:
                new_dir = "epoch_" + str(epoch_str)

            os.mkdir(self.checkpoint_dir + os.sep + new_dir)

            # In that directory, save it
            save_name = "weights.h5"

            self.model.g_G.save_weights(self.checkpoint_dir + os.sep + new_dir + os.sep + "g_G" + save_name)
            self.model.g_F.save_weights(self.checkpoint_dir + os.sep + new_dir + os.sep + "g_F" + save_name)
            self.model.d_G.save_weights(self.checkpoint_dir + os.sep + new_dir + os.sep + "d_G" + save_name)
            self.model.d_F.save_weights(self.checkpoint_dir + os.sep + new_dir + os.sep + "d_F" + save_name)