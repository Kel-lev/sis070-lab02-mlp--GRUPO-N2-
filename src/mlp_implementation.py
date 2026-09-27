import numpy as np

# ============================================================
# FUNCIONES DE ACTIVACIÓN Y SUS DERIVADAS
# ============================================================

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


# ============================================================
# CLASE 1: SimpleMLP (3 capas: Entrada -> Oculta -> Salida)
# ============================================================

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        self.activation = activation

    def _activate(self, z):
        return relu(z) if self.activation == 'relu' else sigmoid(z)

    def _activate_derivative(self, z):
        return relu_derivative(z) if self.activation == 'relu' else sigmoid_derivative(z)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._activate(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        error = (self.A2 - y) * sigmoid_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, error) / m
        db2 = np.sum(error, axis=0, keepdims=True) / m

        d_hidden = np.dot(error, self.W2.T) * self._activate_derivative(self.Z1)
        dW1 = np.dot(X.T, d_hidden) / m
        db1 = np.sum(d_hidden, axis=0, keepdims=True) / m

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, learning_rate=0.5, epochs=5000, verbose=True):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            self.backward(X, y, learning_rate)
            if verbose and epoch % 1000 == 0:
                print(f"  Época {epoch:5d} - Pérdida: {loss:.6f}")
        return losses


# ============================================================
# CLASE 2: DeepMLP (4 capas: Entrada -> Oculta1 -> Oculta2 -> Salida)
# ============================================================

class DeepMLP:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, activation='sigmoid'):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.5
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.5
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.5
        self.b3 = np.zeros((1, output_size))
        self.activation = activation

    def _activate(self, z):
        return relu(z) if self.activation == 'relu' else sigmoid(z)

    def _activate_derivative(self, z):
        return relu_derivative(z) if self.activation == 'relu' else sigmoid_derivative(z)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._activate(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self._activate(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = sigmoid(self.Z3)
        return self.A3

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Capa de salida
        error3 = (self.A3 - y) * sigmoid_derivative(self.Z3)
        dW3 = np.dot(self.A2.T, error3) / m
        db3 = np.sum(error3, axis=0, keepdims=True) / m

        # Capa oculta 2
        error2 = np.dot(error3, self.W3.T) * self._activate_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, error2) / m
        db2 = np.sum(error2, axis=0, keepdims=True) / m

        # Capa oculta 1
        error1 = np.dot(error2, self.W2.T) * self._activate_derivative(self.Z1)
        dW1 = np.dot(X.T, error1) / m
        db1 = np.sum(error1, axis=0, keepdims=True) / m

        # Actualización
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    def train(self, X, y, learning_rate=0.5, epochs=5000, verbose=True):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            self.backward(X, y, learning_rate)
            if verbose and epoch % 1000 == 0:
                print(f"  Época {epoch:5d} - Pérdida: {loss:.6f}")
        return losses


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print("=" * 65)
    print("  LABORATORIO 04 - MLP DESDE CERO - PROBLEMA XOR")
    print("=" * 65)

    # ---------------------------------------------------------
    # ACTIVIDAD 1: Modificación de Hiperparámetros
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ACTIVIDAD 1: Modificación de learning_rate")
    print("=" * 65)

    for lr in [0.9, 0.1, 0.0001]:
        print(f"\n>>> Learning Rate = {lr} <<<")
        np.random.seed(42)
        mlp = SimpleMLP(2, 4, 1, activation='sigmoid')
        losses = mlp.train(X, y, learning_rate=lr, epochs=5000, verbose=True)
        preds = np.round(mlp.forward(X)).flatten()
        print(f"  Pérdida final: {losses[-1]:.6f}")
        print(f"  Predicciones: {preds}")

    # ---------------------------------------------------------
    # ACTIVIDAD 2: Cambio de Función de Activación
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ACTIVIDAD 2: Comparación ReLU vs Sigmoide")
    print("=" * 65)

    for act in ['relu', 'sigmoid']:
        print(f"\n>>> Activación = {act} <<<")
        np.random.seed(42)
        mlp = SimpleMLP(2, 4, 1, activation=act)
        losses = mlp.train(X, y, learning_rate=0.5, epochs=5000, verbose=True)
        preds = np.round(mlp.forward(X)).flatten()
        print(f"  Pérdida final: {losses[-1]:.6f}")
        print(f"  Predicciones: {preds}")

    # ---------------------------------------------------------
    # ACTIVIDAD 3: Ampliación Arquitectural (4 capas)
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ACTIVIDAD 3: MLP de 4 capas (2 -> 4 -> 4 -> 1)")
    print("=" * 65)

    np.random.seed(42)
    deep = DeepMLP(2, 4, 4, 1, activation='sigmoid')
    losses = deep.train(X, y, learning_rate=0.5, epochs=5000, verbose=True)
    preds = np.round(deep.forward(X)).flatten()
    print(f"  Pérdida final: {losses[-1]:.6f}")
    print(f"  Predicciones: {preds}")

    print("\n" + "=" * 65)
    print("  FIN DEL LABORATORIO")
    print("=" * 65)