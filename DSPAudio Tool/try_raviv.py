import numpy as np
def sum_of_distances(string):
    ones_indices = [i for i, char in enumerate(string) if char == '1']
    distance_sum= [0]*len(string)    
    for i in range(len(string)):
        i_vec= np.array([i]*len(ones_indices))

        distance_sum[i] = sum(abs(i_vec - ones_indices))
    return distance_sum

# Example usage:
result = sum_of_distances("1011")
print(result)
