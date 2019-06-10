import math
import random

def vectors_composition(inp_data, weights):
    out_res = []
    for i in inp_data:
        res = 0
        for x, w in zip(i, weights):
            res += x * w
        out_res.append(res)
    return out_res

def activation_func(inp_data):
    return [1 / (1 + math.exp(-x)) for x in inp_data]

def errors_check(result, out_res):
    return [i - j for i, j in zip(result, out_res)]

def low_the_error(out_res, errors):
    return [j * (i * (1 - i)) for i, j in zip(out_res, errors)]

def transposition_matrix(inp_data):
    out_res = []
    for i in range(len(inp_data[0])):
        temp = []
        for j in range(len(inp_data)):
            temp.append(inp_data[j][i])
        out_res.append(temp)
    return out_res

def fix_weights(weights, weight_error):
    return [i + j for i, j in zip(weights, weight_error)]

def inp_and_outp_difference(inp_res, out_res, limit):
    x, k = 0, 0
    temp = []
    for i, j in zip(inp_res, out_res):
        k += 1
        x += abs(i - j)
    #print(x / k)
    if x / k >= limit:
        return 1
    else:
        return 0

def main():
    inp_data = [
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
    ]

    result = [0, 1, 1, 0]
    weights = [random.random() * (1 + 1) - 1 for i in range(1, len(inp_data))]
    limit = 0.001

    while 1:

        # Vector composition of input data and weights
        vec_res = vectors_composition(inp_data, weights)
        # Determining the value of sigmoids for each value of the list
        out_res = activation_func(vec_res)
        # Finding difference between given and geted results
        errors = errors_check(result, out_res)
        # Depending on the direction of the error, adjust the weights slightly
        temp = low_the_error(out_res, errors)
        # Transpose matrix (columns into rows)
        transp_inp = transposition_matrix(inp_data)
        # Vector composition of the transposed matrix and intermediate value
        # Finding weight errors
        weight_error = vectors_composition(transp_inp, temp)
        # Changing weights with regard to the calculated error value
        weights = fix_weights(weights, weight_error)
        # Network training, to the specified accuracy
        flag = inp_and_outp_difference(result, out_res, limit)
        if flag == 0:
            break

    print(f'Weights after  learning :\n{weights}\n')
    print(f'Result :\n{out_res}\n')

    test_ex = [[1, 0, 0]]
    out_res = activation_func(vectors_composition(test_ex, weights))
    print(f'The answer is : {out_res}')

if __name__ == '__main__':
    main()
