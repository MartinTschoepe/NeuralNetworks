import os
import tensorflow as tf

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    mnist = tf.keras.datasets.mnist

    edge_length = 32
    input_shape = (edge_length, edge_length, 1) 
    model_Efficient = tf.keras.applications.EfficientNetB0(
        weights=None,
        input_shape = (edge_length, edge_length, 1),
        classes=10,
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    pad = tf.constant([[0, 0], [ 2,  2], [ 2,  2]])

    for idx in range(25):
        batch_size = 60

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train   = tf.pad(x_train, pad,  "CONSTANT")
        x_test    = tf.pad(x_test , pad,  "CONSTANT")

        predictions = model_Efficient(x_train[:1]).numpy()
        tf.nn.softmax(predictions).numpy()

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


if __name__ == '__main__':
    main()
