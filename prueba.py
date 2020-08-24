def matXvect(matrix1, vector):
	matrix = [[0 for x in range(len(matrix1))] for y in range(1)]
	for i in range(len(matrix1)):
		for j in range(1):
			for k in range(len(vector)):
				matrix[0][i] += matrix1[i][k] * vector[k]
	return matrix


mat = [
    [0.5,0.3,0.6,0.8],
    [0.9,0.6,0.6,0.4],
    [0.67,0.665,0.65,0.3],
    [0.234,0.213,0.45,0.77],
]

vec = [0.5 , 0.6, 2, 1]

result = matXvect(mat, vec)

print(result[0])