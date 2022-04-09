import numpy as np

from functions import Functions


class NeuralNet:
    def __init__(self, in_dim=2, layer1_dim=64, out_dim=1, lr=0.2):
        self.layer1_w = np.random.normal(0, 1, [in_dim, layer1_dim])
        self.layer1_b = np.random.normal(0, 1, [layer1_dim])
        self.out_w = np.random.normal(0, 1, [layer1_dim, out_dim])
        self.out_b = np.random.normal(0, 1, [out_dim])
        self.lr = lr

    def forward(self, x):
        self.outputs = []
        self.outputs.append(x)
        x = sigmoid(np.dot(x, self.layer1_w) + self.layer1_b)
        self.outputs.append(x)
        x = sigmoid(np.dot(x, self.out_w) + self.out_b)
        self.outputs.append(x)
        return x

    def train(self, x, t):
        # backpropagation output layer
        En = (self.outputs[-1] - t) * self.outputs[-1] * (1 - self.outputs[-1])
        grad_En = En
        grad_wout = np.dot(self.outputs[-2].T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        self.out_w -= self.lr * grad_wout
        self.out_b -= self.lr * grad_bout

        # backpropagation layer1
        grad_layer1_u = np.dot(En, self.out_w.T) * \
            self.outputs[-2] * (1 - self.outputs[-2])
        grad_layer1_w = np.dot(self.outputs[-3].T, grad_layer1_u)
        grad_layer1_b = np.dot(
            np.ones([grad_layer1_u.shape[0]]), grad_layer1_u)
        self.layer1_w -= self.lr * grad_layer1_w
        self.layer1_b -= self.lr * grad_layer1_b


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class NeuralNet2:
    def __init__(self, in_dim=2, layer1_dim=64, layer2_dim=64, out_dim=1, lr=0.2):
        self.layer1_w = np.random.normal(0, 1, [in_dim, layer1_dim])
        self.layer1_b = np.random.normal(0, 1, [layer1_dim])
        self.layer2_w = np.random.normal(0, 1, [layer1_dim, layer2_dim])
        self.layer2_b = np.random.normal(0, 1, [layer2_dim])
        self.out_w = np.random.normal(0, 1, [layer2_dim, out_dim])
        self.out_b = np.random.normal(0, 1, [out_dim])
        self.lr = lr

    def forward(self, x):
        self.outputs = []
        self.outputs.append(x)
        x = sigmoid(np.dot(x, self.layer1_w) + self.layer1_b)
        self.outputs.append(x)
        x = sigmoid(np.dot(x, self.layer2_w) + self.layer2_b)
        self.outputs.append(x)
        x = sigmoid(np.dot(x, self.out_w) + self.out_b)
        self.outputs.append(x)
        return x

    def train(self, x, t):
        # backpropagation output layer
        En = (self.outputs[-1] - t) * self.outputs[-1] * (1 - self.outputs[-1])
        grad_En = En
        grad_wout = np.dot(self.outputs[-2].T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        self.out_w -= self.lr * grad_wout
        self.out_b -= self.lr * grad_bout

        # backpropagation layer2
        grad_layer2_u = np.dot(En, self.out_w.T) * \
            self.outputs[-2] * (1 - self.outputs[-2])
        grad_layer2_w = np.dot(self.outputs[-3].T, grad_layer2_u)
        grad_layer2_b = np.dot(
            np.ones([grad_layer2_u.shape[0]]), grad_layer2_u)
        self.layer2_w -= self.lr * grad_layer2_w
        self.layer2_b -= self.lr * grad_layer2_b

        # backpropagation layer1
        grad_layer1_u = np.dot(grad_layer2_u, self.layer2_w.T) * \
            self.outputs[-3] * (1 - self.outputs[-3])
        grad_layer1_w = np.dot(self.outputs[-4].T, grad_layer1_u)
        grad_layer1_b = np.dot(
            np.ones([grad_layer1_u.shape[0]]), grad_layer1_u)
        self.layer1_w -= self.lr * grad_layer1_w
        self.layer1_b -= self.lr * grad_layer1_b


def sliding_window_classify(img, size=32, prob_th=0.8):
    gt = np.array((130, 120, 190, 180), dtype=np.float32)
    # get database
    db = Functions.make_dataset(img, gt)

    # train neural network
    # get input feature dimension
    input_dim = db.shape[1] - 1
    train_x = db[:, :input_dim]
    train_t = db[:, -1][..., None]

    nn = NeuralNet(in_dim=input_dim, lr=0.01)

    # training
    for i in range(10_000):
        nn.forward(train_x)
        nn.train(train_x, train_t)

    h, w, _ = img.shape

    # base rectangle [h, w]
    recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)

    detects = []

    # sliding window
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            for rec in recs:
                # get half size of ractangle
                dh = int(rec[0] // 2)
                dw = int(rec[1] // 2)

                x1 = max(x - dw, 0)  # get left top x
                x2 = min(x + dw, w)  # get left top y
                y1 = max(y - dh, 0)  # get right bottom x
                y2 = min(y + dh, h)  # get right bottom y

                # crop region
                region = img[max(y - dh, 0): min(y + dh, h),
                             max(x - dw, 0): min(x + dw, w)]
                region = Functions.resize(region, size, size)

                # get HOG feature
                region_hog = Functions.hog(region).ravel()

                # predict score using neural network
                score = nn.forward(region_hog)

                if score >= prob_th:
                    detects.append([x1, y1, x2, y2, score])

    return np.array(detects, dtype=np.float32)
