import math
import random
import numpy as np

# 1
def xor(x1, x2):
    return [i ^ j for i, j in zip(x1, x2)]


# 𝑥1 𝑥2	 𝑦
# 0	 0	 0
# 0	 1	 1
# 1	 0   1
# 1	 1 	 0
x1= [0,0,1,1]
x2 = [0,1,0,1]
print(xor(x1,x2))



# 2
def random_weights(count):
    return [random.uniform(-0.5, 0.5) for _ in range(count)]

def random_matrix(rows, columns):
    return [random_weights(columns) for _ in range(rows)]


# 3

def sigmoid(z):
    return 1 / (1 + math.exp(-z))
print(sigmoid(-5))
print(sigmoid(-1))
print(sigmoid(0))
print(sigmoid(1))
print(sigmoid(5))

# 4

def weighted_sum(inputs, weights, bias):
    return sum(weights[i] * inputs[i] for i in range(len(inputs))) + bias
inputs = [2, 3]
weights = [0.5, 0.2]
bias = 1
print(weighted_sum(inputs, weights, bias))

# 5

def neuron(inputs, weights, bias):
    z = weighted_sum(inputs, weights, bias)
    a = sigmoid(z)
    return a

# 6
def hidden_layer(inputs, weights, biases):
    outputs = []
    for i in range(len(weights)):
        result = neuron(inputs, weights[i], biases[i])
        outputs.append(result)

    return outputs

inputs = [0, 1]

weights = [
    [0.2, 0.5],
    [-0.3, 0.7],
    [0.8, -0.2],
    [0.1, 0.4]
]

biases = [0.1, 0.2, -0.1, 0.3]

print(hidden_layer(inputs, weights, biases))

# 7
def output_layer(hidden, weights, bias):
    z = weighted_sum(hidden, weights, bias)
    result = sigmoid(z)
    return result


# 8

def forward(x, w_hidden, b_hidden, w_out, b_out):
    hidden = hidden_layer(x, w_hidden, b_hidden)
    prediction = output_layer(hidden, w_out, b_out)
    return prediction, hidden

w_out = [0.3, -0.5, 0.2, 0.7]
b_out = 0.1

prediction, hidden = forward(
    [0, 1],
    weights,
    biases,
    w_out,
    b_out
)

print("Скрытый слой:", hidden)
print("Предсказание:", prediction)

# 9
def binary_cross_entropy(y_true, y_pred):
    eps = 1e-12
    y_pred = max(eps, min(1 - eps, y_pred))

    return -(
        y_true * math.log(y_pred)
        + (1 - y_true) * math.log(1 - y_pred)
    )

print(binary_cross_entropy(1, 0.9))
print(binary_cross_entropy(1, 0.1))

# 10
def calculate_mean_loss(X, y, w_hidden, b_hidden, w_out, b_out):
    total_loss = 0.0

    for x_i, y_i in zip(X, y):
        y_pred, _ = forward(x_i, w_hidden, b_hidden, w_out, b_out)
        total_loss += binary_cross_entropy(y_i, y_pred)

    return total_loss / len(X)

# 11
def output_delta(y_true, y_pred):
 return y_pred - y_true


# 12
def hidden_deltas(delta_out, w_out, hidden):
    deltas = []

    for i in range(len(hidden)):
        delta = delta_out * w_out[i] * hidden[i] * (1 - hidden[i])
        deltas.append(delta)

    return deltas

# 13
def calculate_gradients(x, hidden, delta_out, delta_hidden):
    grad_w_out = [delta_out * h for h in hidden]
    grad_b_out = delta_out

    grad_w_hidden = [
        [delta * x_j for x_j in x]
        for delta in delta_hidden
    ]
    grad_b_hidden = list(delta_hidden)

    return grad_w_out, grad_b_out, grad_w_hidden, grad_b_hidden

# 14. Вспомогательная функция: создать дерево нулей
def zeros_like(tree):
    if isinstance(tree, (list, tuple)):
        result = [zeros_like(item) for item in tree]
        return tuple(result) if isinstance(tree, tuple) else result

    return 0.0


# 15. Оптимизатор Adam
class Adam:
    def __init__(
        self,
        parameters,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = zeros_like(parameters)  # первое скользящее среднее
        self.v = zeros_like(parameters)  # среднее квадратов градиентов
        self.t = 0                       # номер обновления

    def update(self, parameters, gradients):
        self.t += 1

        def update_tree(params, grads, m, v):
            if isinstance(params, (list, tuple)):
                updated_params = []
                updated_m = []
                updated_v = []

                for p, g, m_i, v_i in zip(params, grads, m, v):
                    p_new, m_new, v_new = update_tree(p, g, m_i, v_i)
                    updated_params.append(p_new)
                    updated_m.append(m_new)
                    updated_v.append(v_new)

                if isinstance(params, tuple):
                    return (
                        tuple(updated_params),
                        tuple(updated_m),
                        tuple(updated_v)
                    )

                return updated_params, updated_m, updated_v

            m_new = self.beta1 * m + (1 - self.beta1) * grads
            v_new = self.beta2 * v + (1 - self.beta2) * (grads ** 2)

            # Поправка смещения
            m_hat = m_new / (1 - self.beta1 ** self.t)
            v_hat = v_new / (1 - self.beta2 ** self.t)

            p_new = params - self.learning_rate * m_hat / (
                    math.sqrt(v_hat) + self.epsilon
            )

            return p_new, m_new, v_new

            # Поправка смещения, особенно важная в начале обучения
            m_hat = m_new / (1 - self.beta1 ** self.t)
            v_hat = v_new / (1 - self.beta2 ** self.t)

            p_new = params - self.learning_rate * m_hat / (
                math.sqrt(v_hat) + self.epsilon
            )

            return p_new, m_new, v_new

        parameters, self.m, self.v = update_tree(
            parameters, gradients, self.m, self.v
        )
        return parameters


# 16. Один шаг обучения
def train_one(x, target, parameters, optimizer):
    w_out, b_out, w_hidden, b_hidden = parameters

    y_pred, hidden = forward(x, w_hidden, b_hidden, w_out, b_out)

    loss = binary_cross_entropy(target, y_pred)
    delta_out = output_delta(target, y_pred)

    delta_hidden = hidden_deltas(delta_out, w_out, hidden)
    gradients = calculate_gradients(
        x, hidden, delta_out, delta_hidden
    )

    updated_parameters = optimizer.update(parameters, gradients)

    return updated_parameters, loss


# 17. Обучение
def train(X, y, parameters, learning_rate, epochs):
    if len(X) != len(y):
        raise ValueError("X и y должны содержать одинаковое число объектов")
    if len(X) == 0:
        raise ValueError("Обучающая выборка не должна быть пустой")

    # Один объект Adam на всё обучение:
    # его внутренние средние не сбрасываются после каждого примера.
    optimizer = Adam(parameters, learning_rate=learning_rate)

    history = []
    report_every = max(1, epochs // 10)

    for epoch in range(epochs):
        total_loss = 0.0

        # SGD: параметры обновляются после каждого объекта
        for x_i, target_i in zip(X, y):
            parameters, loss = train_one(
                x_i, target_i, parameters, optimizer
            )
            total_loss += float(loss)

        average_loss = total_loss / len(X)
        history.append(average_loss)

        if (epoch + 1) % report_every == 0 or epoch + 1 == epochs:
            print(
                f"Эпоха {epoch + 1}/{epochs}, "
                f"ошибка: {average_loss:.6f}"
            )

    return parameters, history
# 18
def predict(x, parameters):
    w_out, b_out, w_hidden, b_hidden = parameters
    probability, _ = forward(x, w_hidden, b_hidden, w_out, b_out)

    predicted_class = 1 if probability >= 0.5 else 0

    return predicted_class

# 19
# XOR
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
y = [0, 1, 1, 0]

random.seed(42)

w_hidden = random_matrix(4, 2)
b_hidden = random_weights(4)
w_out = random_weights(4)
b_out = random.uniform(-0.5, 0.5)

parameters = (w_out, b_out, w_hidden, b_hidden)

parameters, history = train(
    X,
    y,
    parameters,
    learning_rate=0.5,
    epochs=1000

)

# Вывод
print("\nРезультаты:")
print("Вход\tПравильный ответ\tВероятность\t\tПредсказанный класс")

w_out, b_out, w_hidden, b_hidden = parameters

for x_i, target in zip(X, y):
    probability, _ = forward(x_i, w_hidden, b_hidden, w_out, b_out)
    predicted_class = predict(x_i, parameters)

    print(
        f"{x_i}\t{target}\t\t\t\t\t"
        f"{probability:.4f}\t\t\t{predicted_class}"
    )


"Работа за 12.11.2026"

# 1



def vector_add(a, b):
    if len(a) != len(b):
        raise ValueError("Векторы должны иметь одинаковую длину")
    return [x + y for x, y in zip(a, b)]

def vector_multiply(a, b):
    if len(a) != len(b):
        raise ValueError("Векторы должны иметь одинаковую длину")
    return [x * y for x, y in zip(a, b)]


def matrix_vector_multiply_naive(matrix, vector):
    rows = len(matrix)
    cols = len(vector)
    if any(len(row) != cols for row in matrix):
        raise ValueError("Несовместимые размеры матрицы и вектора")

    y = []
    for i in range(rows):
        yi = 0
        for j in range(cols):
            yi += matrix[i][j] * vector[j]
        y.append(yi)
    return y

def vector_apply(vector, function):
    return [function(x) for x in vector]

def tanh_vector(vector):
    arr = np.array(vector)
    return np.tanh(arr)


# 2

def rnn_cell(x, h_prev, W_x, W_h, b_h):
    x      = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    W_x    = np.asarray(W_x, dtype=float)
    W_h    = np.asarray(W_h, dtype=float)
    b_h    = np.asarray(b_h, dtype=float)

    z = W_x @ x + W_h @ h_prev + b_h
    return np.tanh(z)

print(rnn_cell(x = [1.0, 0.5],
h_prev = [0.0, 0.0],
W_x = [
    [0.5, 0.2],
    [0.1, 0.4]
],
W_h = [
    [0.3, 0.1],
    [0.2, 0.5]
],
b_h = [0.0, 0.0]))


# 3
def rnn_forward(sequence, W_x, W_h, b_h, h_0=None):

    W_x = np.asarray(W_x, dtype=float)
    W_h = np.asarray(W_h, dtype=float)
    b_h = np.asarray(b_h, dtype=float)

    hidden_size = W_h.shape[0]

    if h_0 is None:
        h = np.zeros(hidden_size)
    else:
        h = np.asarray(h_0, dtype=float)

    hidden_states = []
    for x_t in sequence:
        h = rnn_cell(x_t, h, W_x, W_h, b_h)
        hidden_states.append(h)

    return hidden_states

# 4

def predict_sequence(sequence, W_x, W_h, b_h, W_y, b_y):

    hidden_states = rnn_forward(sequence, W_x, W_h, b_h)
    h_T = hidden_states[-1]
    z = W_y @ h_T + b_y

    return sigmoid(z[0])

print(predict_sequence(sequence = [
    [1.0],
    [2.0],
    [3.0],
    [4.0]
],W_x = np.array([[0.1]]),
W_h = np.array([[1.0]]),
b_h = np.array([0.0]),
W_y = np.array([[1.0]]),
b_y = np.array([0.0]),))

# 5

def rnn_many_to_many(sequence, W_x, W_h, b_h, W_y, b_y):
    return [W_y @ h_t + b_y for h_t in rnn_forward(sequence, W_x, W_h, b_h)]

sequence = [[1.0], [2.0], [3.0], [4.0]]

W_x = np.array([[1.0]])
W_h = np.array([[0.0]])
b_h = np.array([0.0])
W_y = np.array([[1.0]])
b_y = np.array([0.0])


print(rnn_many_to_many(sequence, W_x, W_h, b_h, W_y, b_y))