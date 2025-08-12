import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import LearningRateScheduler

# Define the teacher model (larger model)
teacher_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Define the student model (smaller model)
student_model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Define the distillation loss function
def distillation_loss(y_true, y_pred, teacher_logits, temperature=3):
    # Standard cross-entropy loss with true labels
    y_true_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    # KL divergence loss with soft targets
    y_teacher = tf.nn.softmax(teacher_logits / temperature)
    y_student = tf.nn.softmax(y_pred / temperature)
    kl_loss = tf.keras.losses.KLDivergence()(y_teacher, y_student) * (temperature ** 2)
    return y_true_loss + kl_loss

# Load and preprocess dataset
def load_data():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values
    x_train = tf.expand_dims(x_train, -1)  # Add channel dimension
    x_test = tf.expand_dims(x_test, -1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)  # One-hot encode labels
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

# Compile and train the teacher model
teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
teacher_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Callback to update teacher logits during student training
class UpdateTeacherLogitsCallback(tf.keras.callbacks.Callback):
    def __init__(self, teacher_model, x_train):
        self.teacher_model = teacher_model
        self.x_train = x_train
        self.teacher_logits = self.teacher_model.predict(self.x_train)

    def on_epoch_end(self, epoch, logs=None):
        self.teacher_logits = self.teacher_model.predict(self.x_train)

# Instantiate the callback
update_teacher_logits_callback = UpdateTeacherLogitsCallback(teacher_model, x_train)

# Define learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch > 3:
        return lr * 0.1
    return lr

# Compile the student model
student_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred, update_teacher_logits_callback.teacher_logits),
    metrics=['accuracy']
)

# Train the student model
student_model.fit(
    x_train, y_train, 
    epochs=5, 
    validation_data=(x_test, y_test),
    callbacks=[update_teacher_logits_callback, LearningRateScheduler(lr_schedule)]
)
