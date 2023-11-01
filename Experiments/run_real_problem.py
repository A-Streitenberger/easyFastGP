import time

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from Base.gp import GP

if __name__ == "__main__":
    numb_of_runs = 3

    california_housing = fetch_california_housing()
    data = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)

    df_x = data[
        [
            "Longitude",
            "Latitude",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup"
        ]
    ]

    # the median income is the target
    df_y = data["MedInc"]

    # extract the values as X and y
    X = df_x.values
    y = df_y.values

    X = df_x.to_numpy()
    y = df_y.to_numpy()

    for run in range(numb_of_runs):
        with open('Logging_GP_run', 'a') as file:
            file.write("Real word experiment (California Housing): Run " + str(run) + "\n")

        start = time.time()
        genetic_program = GP()
        genetic_program.fit(X, y)
        end = time.time()

        with open('Logging_GP_run', 'a') as file:
            file.write("Run finished " + str(end - start) + "\n")
