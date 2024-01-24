import pandas as pd
import pathlib
import pickle

def get_pandas_data(pickle_filename: str) -> pd.DataFrame:
    '''
    Load data from /data directory as a pandas DataFrame
    using relative paths. Relative paths are necessary for
    data loading to work in Heroku.
    '''
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("data").resolve()
    # Assuming the file is a pickled DataFrame
    with open(DATA_PATH.joinpath(pickle_filename), 'rb') as file:
        df = pd.read_pickle(file)
    return df

# Example usage to load a .pickle file
# TT_data = get_pandas_data("TT_data.pickle")
# print(TT_data)
# df = TT_data["TTdata"][TT_data["TTdata"]["Race"] == "Bessege2023"]
# print(df)


def save_pandas_data(data: pd.DataFrame, pickle_filename: str) -> None:
   '''
   Save a pandas DataFrame to /data directory as a pickle file
   using relative paths.
   '''
   PATH = pathlib.Path(__file__).parent
   DATA_PATH = PATH.joinpath("data").resolve()
   # Save the DataFrame to a pickle file
   with open(DATA_PATH.joinpath(pickle_filename), 'wb') as file:
      pickle.dump(data, file)


# save_pandas_data(df, "example.pickle")