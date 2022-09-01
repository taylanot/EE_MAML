import torch

class Bayes():
    def __init__(self):
        pass

    def fit(self, a):
        self.m_amplitude = a[0]
        self.m_phase= a[1]

    def predict(self,X):
        return   (torch.sin(X+self.m_phase) @ self.m_amplitude.T).reshape(-1,1)

class KernelRidge():
    def __init__(self, lmbda=0, l=1):
        self.lmbda= lmbda
        self.kernel = self.RBF(l)

    def fit(self, D):
        X, Y = D
        self.alpha = torch.inverse(self.kernel(X,X) + self.lmbda*torch.eye(X.shape[0])) @ Y
        self.X = X

    def predict(self, X):
        return (self.kernel(X, self.X)) @ self.alpha

    class RBF():
        def __init__(self,l=1):
            self.l = l

        def pairwise_l2_distance(self, x,y):
            D = -2 * x @  y.T + torch.sum(y**2, axis=1) + torch.sum(x**2, axis=1)[:,None]
            D[D<0] = 0.
            return D

        def __call__(self, x, xp=None):
            if not isinstance(xp, torch.Tensor):
                xp = x
            return torch.exp(-0.5 * self.pairwise_l2_distance(x,xp)/self.l**2)

