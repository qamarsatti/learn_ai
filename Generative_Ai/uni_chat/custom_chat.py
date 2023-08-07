import numpy as np
import matplotlib.pyplot as plt
import utils
import time
inword=input("number of word\n")

try:
    numberOfWord=int(inword)
except:
    raise 'please enter a number'    

start_time = time.time()
words = utils.txt.split()
unique_word=" ".join(sorted(set(words), key=words.index))
print(f'number of word={len(unique_word)}')
x, y, size = utils.getmatrix(np, unique_word)




def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define your cost function
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    J = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
    grad = (1/m) * X.T @ (h - y)
    return J, grad

# Define your gradient descent function
def gradient_descent(X, y, theta, alpha, num_iters):
    J_history = []
    for i in range(num_iters):
        J, grad = cost_function(X, y, theta)
        theta = theta - (alpha * grad)
        J_history.append(J)
    return theta, J_history


initial_theta =np.random.random((size, size))

# Set the learning rate and number of iterations
alpha = 0.01
num_iters = 10000

# Run gradient descent to minimize the cost function
theta, J_history = gradient_descent(x, y, initial_theta, alpha, num_iters)

# Print the final values of theta
# print('Final values of theta:', theta)



def genrate(x):
    try:
        if x in utils.word_dic:
            value = utils.word_dic[x]
            x_f = [[value]]
            predictions = sigmoid(np.dot(x_f, theta))
            maxx=np.argmax(predictions[0][0], axis=0)
            final=maxx+1
            # print(f'prediction={predictions}and max={max}')

            keys = list(utils.word_dic.keys())
            first_key = keys[final]
            first_value = utils.word_dic[first_key]

            return first_key


        else:
            return False
    except:
        pass    


end_time = time.time()

print("Time taken: ", end_time - start_time, " seconds")
while True:
    txtIN=input('word start from\n')
    for i in range(numberOfWord):
        txtIN=genrate(txtIN)
        if not txtIN:
            break
        print(txtIN, end=' ')

    if txtIN=='end':
        break
    print('\n           **************END****************')