#%%
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(5*4*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((5,4,256)))
    assert model.output_shape == (None, 5, 4, 256)

    model.add(layers.Conv2DTranspose(128, (3,3), strides=(1,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 5, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 10, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(1, (5,5), strides=(1,2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 10, 32, 1)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1))
    model.add(layers.Conv2D(96, (3,3), padding='same', input_shape=[10,32,1]))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(128, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.add(layers.BatchNormalization())

    return model

# Data set loading and preparation for the TensorFlow models was implemented using a tutorial on the TensorFlow website:
# https://www.tensorflow.org/tutorials/load_data/images#load_using_keraspreprocessing

parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
main_dir = os.path.join(parentDir, 'dataset', 'groove_img (10x32)')

batch = 32
epochs = 100
train_data = image_dataset_from_directory(
        main_dir,
        labels="inferred",
        label_mode="binary",
        shuffle=True,
        color_mode="grayscale",
        validation_split=0.1,
        subset="training",
        image_size=(10,32),
        seed=123,
        batch_size=batch
    )

validation_data = image_dataset_from_directory(
        main_dir,
        labels="inferred",
        label_mode="binary",
        shuffle=True,
        color_mode="grayscale",
        validation_split=0.1,
        subset="validation",
        image_size=(10,32),
        seed=123,
        batch_size=batch
    )

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)

validation_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)

# Start DCGAN functionality

generator = make_generator_model()
discriminator = make_discriminator_model()

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

opt_dis = tf.keras.optimizers.Adam(0.0001)
opt_gen = tf.keras.optimizers.Adam()

g1 = tf.random.Generator.from_seed(1)

def discriminator_loss(real_predictions, fake_predictions):
    real_loss = bce(tf.ones_like(real_predictions), real_predictions)
    fake_loss = bce(tf.zeros_like(fake_predictions), fake_predictions)
    return real_loss, fake_loss

def generator_loss(fake_predictions):
    return bce(tf.ones_like(fake_predictions), fake_predictions)

def generate_and_save_images(epoch):
    noise_samples = g1.normal([16,100], 0.0, 1.0) #tensor of noise samples
    predictions  = generator(noise_samples, training=False) #tensor of predictions
    relativedir = f'image_at_epoch_{epoch}/'
    os.mkdir(os.path.dirname(__file__) + f'\\image_at_epoch_{epoch}')
    for i in range (predictions.shape[0]):
        plt.figimage(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray', resize=True)
        plt.savefig(relativedir + f'{i}_image_at_epoch_{epoch}.png')
        Image.open(relativedir + f'{i}_image_at_epoch_{epoch}.png').convert('L').save(relativedir + f'{i}_image_at_epoch_{epoch}.png')

@tf.function
def train_step(images):
    with tf.GradientTape() as tape_dis, tf.GradientTape() as tape_gen:
        noise_samples = g1.normal([batch,100]) #sample 32 noise samples, array of tensors
        real_pred = discriminator(images, training=True)
        fake_pred = discriminator(generator(noise_samples, training=True), training=True)
        d_loss_real, d_loss_fake = discriminator_loss(real_pred, fake_pred)
        total_loss = d_loss_real + d_loss_fake
        g_loss = generator_loss(fake_pred)
        d_gradient = tape_dis.gradient(total_loss, discriminator.trainable_variables)
        g_gradient = tape_gen.gradient(g_loss, generator.trainable_variables)
    opt_dis.apply_gradients(zip(d_gradient, discriminator.trainable_variables))
    opt_gen.apply_gradients(zip(g_gradient, generator.trainable_variables))
    return d_loss_real, d_loss_fake, g_loss

def train_loop():
    for epoch in range(epochs):
        generate_and_save_images(epoch)
        print(f"\nStarting new epoch: {epoch + 1}")
        total_d_loss_real = 0
        total_d_loss_fake = 0
        total_g_loss = 0
        for images, _ in train_data:
            d_loss_real, d_loss_fake, g_loss = train_step(images)
            total_d_loss_real = total_d_loss_real + d_loss_real
            total_d_loss_fake = total_d_loss_fake + d_loss_fake
            total_g_loss = total_g_loss + g_loss
        print(f"Discriminator real loss, epoch {epoch + 1}")
        tf.print(total_d_loss_real/138)
        print(f"Discriminator fake loss, epoch {epoch + 1}")
        tf.print(total_d_loss_fake/138)
        print(f"Generator loss, epoch {epoch + 1}")
        tf.print(total_g_loss/138)
    generate_and_save_images(epochs)

train_loop()
# %%
