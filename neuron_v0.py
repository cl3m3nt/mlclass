class Neuron:

    def __init__(self,input_len):
        self.bias = 1 # init bias to 1
        self.w = np.random.rand(input_len) # random value weight list
        self.output = 0 # init output to 0

    def linearActivation(self,x):
        self.output = self.bias
        for i in range(0,len(x)):
            self.output = self.output + self.w[i] * x[i]
        return self.output

    def reluActivation(self,x):
        self.output = self.bias
        for i in range(0,len(x)):
            self.output = self.output + self.w[i] * x[i]
        if(self.output > 0):
            self.output = self.output
        else:
            self.output = 0
        return self.output

    def tanhActivation(self,x):
        self.output = self.bias
        for i in range(0,len(x)):
            self.output = self.output + self.w[i] * x[i]
        self.output = np.tanh(self.output)
        return self.output

    def sigmoidActivation(self,x):
        self.output = self.bias
        for i in range(0,len(x)):
            self.output = self.output + self.w[i] * x[i]
        self.output = 1 / (1+np.exp(-self.output))
        return self.output

    def softmaxActivation(self,x,k):
        self.output = self.bias
        for i in range(0,len(x)):
            self.output = self.output + np.exp(self.w[i] * x[i])
        self.output = np.exp(x[k]) / self.output
        return self.output