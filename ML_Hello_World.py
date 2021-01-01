import os
import tensorflow as tf
import numpy
def rescale_x_train(x_train,scale_value,binary):
    x_train = x_train*(1-scale_value)
    x_train = x_train+scale_value*binary
    return x_train

def create_padding(paddings):
    paddings.append(tf.constant([[0, 0], [ 2,  2], [ 2,  2]]))
    paddings.append(tf.constant([[0, 0], [ 0,  4], [ 0,  4]]))
    paddings.append(tf.constant([[0, 0], [ 2,  2], [ 0,  4]]))
    paddings.append(tf.constant([[0, 0], [ 4,  0], [ 0,  4]]))
    paddings.append(tf.constant([[0, 0], [ 4,  0], [ 2,  2]]))
    paddings.append(tf.constant([[0, 0], [ 4,  0], [ 4,  0]]))
    paddings.append(tf.constant([[0, 0], [ 2,  2], [ 4,  0]]))
    paddings.append(tf.constant([[0, 0], [ 0,  4], [ 4,  0]]))
    paddings.append(tf.constant([[0, 0], [ 0,  4], [ 2,  2]]))
    paddings.append(tf.constant([[0, 0], [ 1,  3], [ 1,  3]]))
    paddings.append(tf.constant([[0, 0], [ 3,  1], [ 1,  3]]))
    paddings.append(tf.constant([[0, 0], [ 3,  1], [ 3,  1]]))
    paddings.append(tf.constant([[0, 0], [ 1,  3], [ 3,  1]]))
    return paddings

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    mnist = tf.keras.datasets.mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0

    paddings = []
    paddings = create_padding(paddings)

    numb_paddings = len(paddings)
    # Default-Net
    model_default = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    for idx in range(50):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        # x_train   = tf.pad(x_train, paddings[0], "CONSTANT")
        # x_test    = tf.pad(x_test , paddings[0], "CONSTANT")
        x_train   = tf.pad(x_train, paddings[idx%numb_paddings], "CONSTANT")
        x_test    = tf.pad(x_test , paddings[idx%numb_paddings], "CONSTANT")
        # x_train = rescale_x_train(x_train,0.05*(idx%5),idx%2)

        predictions = model_default(x_train[:1]).numpy()
        tf.nn.softmax(predictions).numpy()

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_fn(y_train[:1], predictions).numpy()

        model_default.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])


        model_default.fit(x_train, y_train, epochs=1)
        model_default.evaluate(x_test,  y_test, verbose=2)

    probability_model = tf.keras.Sequential([
        model_default,
        tf.keras.layers.Softmax()
    ])
    probability_model(x_test[:5])

    # Efficient-Net
    # model_Efficient = tf.keras.applications.EfficientNetB0(
    #     # include_top=False,
    #     weights=None,
    #     input_shape=(32, 32, 1),
    #     classes=10,
    # )

    # predictions = model_Efficient(x_train[:1]).numpy()
    # tf.nn.softmax(predictions).numpy()

    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss_fn(y_train[:1], predictions).numpy()

    # model_Efficient.compile(optimizer='adam',
    #           loss=loss_fn,
    #           metrics=['accuracy'])

    # for idx in range(25):
    #     model_Efficient.fit(x_train, y_train, epochs=1)
    #     model_Efficient.evaluate(x_test,  y_test, verbose=2)

    # probability_model = tf.keras.Sequential([
    #     model_default,
    #     tf.keras.layers.Softmax()
    # ])
    # probability_model(x_test[:5])


    print('ML Hello World!')

if __name__ == '__main__':
    main()
