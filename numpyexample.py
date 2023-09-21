import numpy as np 
data = np.array([1,2,3,4,5])
weights = np.random.rand(5,1)

print(data.shape)
print(data.dtype)

print(weights.shape)
print(weights.dtype)

print(weights.T)
print(weights[0]) #1st index
print(weights[1]) #2nd index
print(weights[1:3]) #1nd and 3rd index
print(weights[-1]) #Last index

def weight_summary(weight_arry):
    for i in range(len(weight_arry)):
        print("Weight-Index-", i, ":", weight_arry[i])

weight_summary(weights)

#calculate the standard deviation based on the data
noise_magnitude = 0.1 * np.mean(data)

#generate noise with the same shape as the data
noise = np.random.normal(0, noise_magnitude, data.shape)

#add the noise to the data
noisy_data = data + noise_magnitude * noise

#print the original and noisy data
print("Original-data:", data)
print("Noisy-data:", noisy_data)

#analyze negative noise
negative_noise_mask = data > noisy_data
negative_data = noisy_data[negative_noise_mask]
neg_magnitude = np.linalg.norm(negative_data, ord=2)

print("Negative-noise-condition: ", negative_noise_mask)
print("Data-with-negative-noise: ", negative_data)
print(f"Magnitude-of-negative-noise: -{neg_magnitude}")

#use weights to get a weighted sum of a single input of data
weights_transposed = weights.T 
output = np.dot(weights_transposed, data)
print("Transposed-weights-shape: ", weights_transposed.shape)
print("Output-shape: ", output.shape)

#Expand diminesions
data = data.reshape(data.shape[0], 1)
output = np.dot(weights_transposed, data)
print("New-Data-shape: ", data.shape)
print("Output-shape: ", output.shape)

#Batch operation
data = np.random.rand(data.shape[0], 100)
output = np.dot(weights_transposed, data)

print("Batch-data-shape: ", data.shape)
print("Batch-output-shape: ", output.shape)

#Calculate the covariance matrix on centered data
cov_matrix = np.cov(data - np.mean(data, axis=0), rowvar=False)

#Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

#sort eigenvalues and eigenvectors 
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

#select the top k eigenvectors
k = 2
top_k_eigenvectors = sorted_eigenvectors[:, :k]

#transform the original data
principal_components = np.dot(data_centerd, top_k_eigenvectors)

print("principal-components-shape: ", principal_components.shape)

np.savetxt('array.txt', data)
arr_loaded = np.loadtxt('array.text')