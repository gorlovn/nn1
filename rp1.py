import numpy as np

input_vector = [1.72, 1.23]
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]

i0 = input_vector[0] * weights_1[0]
i1 = input_vector[1] * weights_1[1]
dot_product_1 = i0 + i1

print(f"The dot product is: {dot_product_1}")

dot_product_1 = np.dot(input_vector, weights_1)
print(f"The dot product 1 is: {dot_product_1}")

dot_product_2 = np.dot(input_vector, weights_2)
print(f"The dot product 2 is: {dot_product_2}")
