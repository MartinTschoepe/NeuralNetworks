import os
import tensorflow as tf
import numpy as np

def create_padding(paddings):
    paddings.append(tf.constant([[0, 0], [ 2,  2], [ 2,  2], [0, 0]]))
    paddings.append(tf.constant([[0, 0], [ 0,  4], [ 0,  4], [0, 0]]))
    paddings.append(tf.constant([[0, 0], [ 2,  2], [ 0,  4], [0, 0]]))
    paddings.append(tf.constant([[0, 0], [ 4,  0], [ 0,  4], [0, 0]]))
    paddings.append(tf.constant([[0, 0], [ 4,  0], [ 2,  2], [0, 0]]))
    paddings.append(tf.constant([[0, 0], [ 4,  0], [ 4,  0], [0, 0]]))
    paddings.append(tf.constant([[0, 0], [ 2,  2], [ 4,  0], [0, 0]]))
    paddings.append(tf.constant([[0, 0], [ 0,  4], [ 4,  0], [0, 0]]))
    paddings.append(tf.constant([[0, 0], [ 0,  4], [ 2,  2], [0, 0]]))
    paddings.append(tf.constant([[0, 0], [ 1,  3], [ 1,  3], [0, 0]]))
    paddings.append(tf.constant([[0, 0], [ 3,  1], [ 1,  3], [0, 0]]))
    paddings.append(tf.constant([[0, 0], [ 3,  1], [ 3,  1], [0, 0]]))
    paddings.append(tf.constant([[0, 0], [ 1,  3], [ 3,  1], [0, 0]]))
    return paddings

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    mnist = tf.keras.datasets.mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0

    paddings = []
    paddings = create_padding(paddings)

    edge_length = 32
    input_shape = (edge_length, edge_length, 1) 
    numb_paddings = len(paddings)
    # Default-Net
    model_default = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(4, 3, padding='same', 
                input_shape=input_shape, activation='relu'),
        tf.keras.layers.Flatten(input_shape=(edge_length, edge_length, 4)),
        # tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        # tf.keras.layers.Flatten(input_shape=(32, 32)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # for idx in range(100):
    #     (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #     x_train, x_test = x_train / 255.0, x_test / 255.0
    #     x_train = np.reshape(x_train,(60000,28,28,1))
    #     x_test  = np.reshape(x_test, (10000,28,28,1))
    #     x_train   = tf.pad(x_train, paddings[idx%numb_paddings], "CONSTANT")
    #     x_test    = tf.pad(x_test , paddings[0], "CONSTANT")

    #     predictions = model_default(x_train[:1]).numpy()
    #     tf.nn.softmax(predictions).numpy()

    #     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #     loss_fn(y_train[:1], predictions).numpy()

    #     model_default.compile(optimizer='adam',
    #               loss=loss_fn,
    #               metrics=['accuracy'])


    #     model_default.fit(x_train, y_train, batch_size=600, epochs=1)
    #     model_default.evaluate(x_test,  y_test, verbose=2)

    # probability_model = tf.keras.Sequential([
    #     model_default,
    #     tf.keras.layers.Softmax()
    # ])
    # probability_model(x_test[:5])

    # Efficient-Net
    model_Efficient = tf.keras.applications.EfficientNetB0(
        # include_top=False,
        weights=None,
        # input_shape=(32, 32, 1),
        input_shape = (edge_length, edge_length, 1),
        classes=10,
    )
    pad = tf.constant([[0, 0], [ 2,  2], [ 2,  2]])

    for idx in range(25):
        # if (idx <= 2):
        #     batch_size = 1
        # elif (idx <= 4):
        #     batch_size = 32
        # else:
        batch_size = 32

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        # x_train = np.reshape(x_train,(60000,28,28,1))
        # x_test  = np.reshape(x_test, (10000,28,28,1))
        # x_train   = tf.pad(x_train, paddings[idx%numb_paddings], "CONSTANT")
        x_train   = tf.pad(x_train, pad,  "CONSTANT")
        x_test    = tf.pad(x_test , pad,  "CONSTANT")

        predictions = model_Efficient(x_train[:1]).numpy()
        tf.nn.softmax(predictions).numpy()

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_fn(y_train[:1], predictions).numpy()

        model_Efficient.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

        model_Efficient.fit(x_train, y_train, batch_size=batch_size, epochs=1)
        model_Efficient.evaluate(x_test,  y_test, verbose=2)

    probability_model = tf.keras.Sequential([
        model_default,
        tf.keras.layers.Softmax()
    ])
    # probability_model(x_test[:5])


    print('ML Hello World!')

if __name__ == '__main__':
    main()
