import time

import numpy as np
from sklearn.model_selection import train_test_split

from Base.gp import GP

if __name__ == "__main__":
    # defines the range from -i to i. This means i = 32 is a total of 64 data data_range.
    data_range = [32, 512, 1024]
    # noise 0%, 10%, 20%
    noise_options = [0, 0.10, 0.20]
    # each variant is tested 3 times
    numb_of_runs = 3

    for i in data_range:
        # the synthetic problem
        X = np.array([x / i for x in range(-i, i)], dtype=np.float64).reshape(i * 2, 1)
        y = X * X * (X - 1) * (X - 1) * (X + 1) * (X + 1)
        y = y.flatten()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        for j in noise_options:
            num_points_to_replace = int(j * len(y_train))

            # add noise
            min_value = np.min(y)
            max_value = np.max(y)
            noise = np.random.uniform(min_value, max_value, num_points_to_replace)

            replace_indices = np.random.choice(len(y_train), num_points_to_replace, replace=False)

            y_train[replace_indices] = noise

            for run in range(numb_of_runs):
                with open('Logging_GP_run', 'a') as file:
                    file.write(
                        "Experiment with: range " + str(i) + ", noise " + str(j) + ", number " + str(run) + "\n")
                    file.write("Generation; Min_nodes; Max_nodes; Avg_nodes; Min_height; Max_height; Avg_height; "
                               "Min_train_fit; Max_train_fit; Avg_train_fit \n")

                start = time.time()
                genetic_program = GP()
                genetic_program.fit(X, y)
                end = time.time()

                with open('Logging_GP_run', 'a') as file:
                    file.write("Run finished, Time: " + str(end - start) + "\n")
