import pandas as pd
import os

RAW_DATA_PATH = "data/raw/creditcard.csv"
BATCHES_DIR = "data/batches/"
NUM_BATCHES = 5  # you can change this later

def create_batches():
    # load dataset
    df = pd.read_csv(RAW_DATA_PATH)
    
    # sort by time (to simulate real-time arrival)
    df = df.sort_values(by="Time").reset_index(drop=True)

    # calculate batch size
    batch_size = len(df) // NUM_BATCHES

    # create output folder if not exists
    os.makedirs(BATCHES_DIR, exist_ok=True)

    # split and save
    for i in range(NUM_BATCHES):
        start = i * batch_size
        end = (i + 1) * batch_size if i < NUM_BATCHES - 1 else len(df)
        batch_df = df.iloc[start:end]

        batch_file = os.path.join(BATCHES_DIR, f"batch_{i+1:02d}.csv")
        batch_df.to_csv(batch_file, index=False)

        print(f"Saved {batch_file} with {len(batch_df)} rows")

if __name__ == "__main__":
    create_batches()
