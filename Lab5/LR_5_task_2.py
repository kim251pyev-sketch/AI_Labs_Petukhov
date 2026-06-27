import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class PietukhovNeuralNetwork:
    def __init__(self):
        # Ініціалізація ваг та зміщень випадковими значеннями
        self.w1, self.w2, self.w3 = np.random.normal(), np.random.normal(), np.random.normal()
        self.w4, self.w5, self.w6 = np.random.normal(), np.random.normal(), np.random.normal()
        self.b1, self.b2, self.b3 = np.random.normal(), np.random.normal(), np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                d_L_d_ypred = -2 * (y_true - y_pred)

                self.w5 -= learn_rate * d_L_d_ypred * h1 * deriv_sigmoid(sum_o1)
                self.w6 -= learn_rate * d_L_d_ypred * h2 * deriv_sigmoid(sum_o1)
                self.b3 -= learn_rate * d_L_d_ypred * deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * x[0] * deriv_sigmoid(sum_h1)
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * x[1] * deriv_sigmoid(sum_h1)
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * deriv_sigmoid(sum_h1)

                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * x[0] * deriv_sigmoid(sum_h2)
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * x[1] * deriv_sigmoid(sum_h2)
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * deriv_sigmoid(sum_h2)

            if epoch % 100 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                print(f"Epoch {epoch} loss: {mse_loss(all_y_trues, y_preds):.3f}")


data = np.array([[-2, -1], [25, 6], [17, 4], [-15, -6]])
all_y_trues = np.array([1, 0, 0, 1])

network = PietukhovNeuralNetwork()
network.train(data, all_y_trues)

print(f"Emily: {network.feedforward(np.array([-7, -3])):.3f}")
print(f"Frank: {network.feedforward(np.array([20, 2])):.3f}")