import numpy as np

class DenseLayer:
    def __init__(self, neurons, activation='relu', l2_reg=0.01):
        self.neurons = neurons
        self.activation = activation
        self.l2_reg = l2_reg

    def relu(self, inputs):
        return np.maximum(0, inputs)

    def leaky_relu(self, inputs, alpha=0.01):
        return np.where(inputs > 0, inputs, alpha * inputs)

    def tanh(self, inputs):
        return np.tanh(inputs)

    def softmax(self, inputs):
        exp_scores = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def relu_derivative(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def leaky_relu_derivative(self, dA, Z, alpha=0.01):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] *= alpha
        return dZ

    def tanh_derivative(self, dA, Z):
        return dA * (1 - np.power(Z, 2))

    def forward(self, inputs, weights, bias):
        Z_curr = np.dot(inputs, weights.T) + bias

        if self.activation == 'relu':
            A_curr = self.relu(inputs=Z_curr)
        elif self.activation == 'leaky_relu':
            A_curr = self.leaky_relu(inputs=Z_curr)
        elif self.activation == 'tanh':
            A_curr = self.tanh(inputs=Z_curr)
        elif self.activation == 'softmax':
            A_curr = self.softmax(inputs=Z_curr)

        return A_curr, Z_curr

    def backward(self, dA_curr, W_curr, Z_curr, A_prev):
        if self.activation == 'softmax':
            dW = np.dot(A_prev.T, dA_curr)
            db = np.sum(dA_curr, axis=0, keepdims=True)
            dA_prev = np.dot(dA_curr, W_curr)
        else:
            if self.activation == 'leaky_relu':
                dZ = self.leaky_relu_derivative(dA_curr, Z_curr)
            elif self.activation == 'tanh':
                dZ = self.tanh_derivative(dA_curr, Z_curr)
            else:
                dZ = self.relu_derivative(dA_curr, Z_curr)
            dW = np.dot(A_prev.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            dA_prev = np.dot(dZ, W_curr)

        return dA_prev, dW, db

class Network:
    def __init__(self):
        self.network = []  # camadas
        self.architecture = []  # mapeamento das entradas --> saída
        self.params = []  # W, b
        self.memory = []  # Z, A
        self.gradients = []  # dW, db

    def add(self, layer):
        """
        Adiciona camadas à rede
        """
        self.network.append(layer)

    def _compile(self, data):
        """
        Inicializa a arquitetura do modelo
        """
        self.architecture = []  # Reseta a arquitetura
        for idx, layer in enumerate(self.network):
            if idx == 0:
                self.architecture.append({'input_dim': data.shape[1],
                                          'output_dim': self.network[idx].neurons,
                                          'activation': layer.activation})
            elif idx > 0 and idx < len(self.network) - 1:
                self.architecture.append({'input_dim': self.network[idx - 1].neurons,
                                          'output_dim': self.network[idx].neurons,
                                          'activation': layer.activation})
            else:
                self.architecture.append({'input_dim': self.network[idx - 1].neurons,
                                          'output_dim': self.network[idx].neurons,
                                          'activation': 'softmax'})
        return self

    def _init_weights(self, data, random_seed=8):
        """
        Inicializa os Parâmetros
        """
        self._compile(data)
        np.random.seed(random_seed)

        self.params = []  # Reseta os parâmetros
        for i in range(len(self.architecture)):
            layer_input_size = self.architecture[i]['input_dim']
            self.params.append({
                'W': np.random.randn(self.architecture[i]['output_dim'], layer_input_size) * np.sqrt(1 / layer_input_size),
                'b': np.zeros((1, self.architecture[i]['output_dim']))})

        # Verificação de debug
        print(f"Camadas da rede: {len(self.network)}")
        print(f"Parâmetros iniciais: {len(self.params)}")

        return self

    def _forwardprop(self, data):
        """
        Realiza forward propagation na rede
        """
        A_curr = data
        self.memory = []  # Reseta a memória

        for i in range(len(self.params)):
            A_prev = A_curr
            A_curr, Z_curr = self.network[i].forward(inputs=A_prev, weights=self.params[i]['W'],
                                                      bias=self.params[i]['b'])

            self.memory.append({'inputs': A_prev, 'Z': Z_curr})

        return A_curr

    def _backprop(self, predicted, actual):
        """
        Realiza backward propagation na rede
        """
        num_samples = len(actual)

        dscores = predicted
        dscores[range(num_samples), actual] -= 1
        dscores /= num_samples

        dA_prev = dscores
        self.gradients = []  # Reseta os gradientes

        for idx, layer in reversed(list(enumerate(self.network))):
            dA_curr = dA_prev

            A_prev = self.memory[idx]['inputs']
            Z_curr = self.memory[idx]['Z']
            W_curr = self.params[idx]['W']

            activation = self.architecture[idx]['activation']

            dA_prev, dW_curr, db_curr = layer.backward(dA_curr, W_curr, Z_curr, A_prev)

            self.gradients.append({'dW': dW_curr, 'db': db_curr})

    def _update(self, lr=0.01):
        for idx, layer in enumerate(self.network):
            self.params[idx]['W'] -= lr * list(reversed(self.gradients))[idx]['dW'].T
            self.params[idx]['b'] -= lr * list(reversed(self.gradients))[idx]['db']

    def _get_accuracy(self, predicted, actual):
        return np.mean(np.argmax(predicted, axis=1) == actual)

    def _get_precision(self, predicted, actual):
        correct_predictions = np.sum(np.argmax(predicted, axis=1) == actual)
        total_predictions = len(actual)
        precision = correct_predictions / total_predictions
        return precision

    def _calculate_loss(self, predicted, actual):
        samples = len(actual)

        if predicted.ndim == 1:
            predicted = np.expand_dims(predicted, axis=1)

        correct_logprobs = -np.log(predicted[range(samples), actual])
        data_loss = np.sum(correct_logprobs) / samples

        # Regularização L2
        l2_loss = 0.5 * self.network[-1].l2_reg * np.sum([np.sum(np.square(self.params[i]['W'])) for i in range(len(self.params))])
        data_loss += l2_loss

        return data_loss

    def train(self, X_train, y_train, X_val, y_val, epochs, lr=0.01, min_precision=1):
        self.loss = []
        self.accuracy = []
        self.precision = []
        self.val_accuracy = []
        self.val_loss = []

        self._init_weights(X_train)

        for i in range(epochs):
            yhat = self._forwardprop(X_train)
            self.accuracy.append(self._get_accuracy(predicted=yhat, actual=y_train))
            self.loss.append(self._calculate_loss(predicted=yhat, actual=y_train))
            self.precision.append(self._get_precision(predicted=yhat, actual=y_train))

            self._backprop(predicted=yhat, actual=y_train)
            self._update(lr=lr)  # Use lr here

            # Validação
            yhat_val = self._forwardprop(X_val)
            self.val_accuracy.append(self._get_accuracy(predicted=yhat_val, actual=y_val))
            self.val_loss.append(self._calculate_loss(predicted=yhat_val, actual=y_val))

            s = 'ÉPOCA: {}, ACURÁCIA: {:.2f}%, LOSS: {:.4f}, PRECISÃO: {:.2f}%, VAL_ACURÁCIA: {:.2f}%, VAL_LOSS: {:.4f}'.format(
                i + 1, self.accuracy[-1] * 100, self.loss[-1], self.precision[-1] * 100, self.val_accuracy[-1] * 100, self.val_loss[-1])
            print(s)

            if self.precision[-1] >= min_precision:
                print(f'Mínima precisão de {min_precision * 100}% alcançada. Parando o treinamento.')
                break

        self.epochs_ran = i + 1  # Armazenar o número de épocas que realmente rodaram

    def predict(self, data):
        A_curr = data

        for i in range(len(self.params)):
            A_prev = A_curr
            A_curr, Z_curr = self.network[i].forward(inputs=A_prev, weights=self.params[i]['W'],
                                                        bias=self.params[i]['b'])

        return A_curr
    
    