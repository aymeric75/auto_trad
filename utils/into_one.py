import numpy as np

def load_time_series(file_path):
    """Load time series data from a file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        time_series = [list(map(float, line.strip().split())) for line in lines]
    return np.array(time_series)

# Load the data from the three files
# feature1 = load_time_series('all_heikin_ashi.txt')
# feature3 = load_time_series('all_histo_macd.txt')
# feature2 = load_time_series('all_rsi.txt')
# feature3 = load_time_series('output3.txt')
# feature1 = load_time_series('with_ha.txt')
# feature3 = load_time_series('with_macd.txt')
# feature2 = load_time_series('with_rsi.txt')

feature1 = load_time_series('with_rsi.txt')

# Check if the number of samples and timesteps match
#assert feature1.shape == feature2.shape #== feature3.shape, "Mismatch in shapes of the input files"

# Stack the features to get the final shape (n_samples, n_timesteps, 3)
### ON AJOUTE AUSSI le 0 ou le 1 <=> result du trade
n_samples, n_timesteps = feature1.shape
#data = np.stack((feature1, feature2, feature3), axis=-1)  # , feature3

# print(data.shape) # (1146, 11, 2)
# print(feature3.shape)
feature1 = np.expand_dims(feature1, axis=-1)
data = feature1


# np.savez('all_rsi.npz', data=data)
np.savez('with_rsi.npz', data=data)


# np.savez('with_three_data.npz', data=data)
# print("Final array shape:", data.shape)