import pandas as pd
import numpy as np

output_dir = 'res'

def create_shuffles(input_file, num_shuffles=50):
    try:
        df = pd.read_csv(input_file)
        
        for i in range(num_shuffles):
            shuffled_df = df.sample(frac=1, random_state=i).reset_index(drop=True)
            output_file = f'{output_dir}/shuffled_{i+1:02d}.csv'
            shuffled_df.to_csv(output_file, index=False)
            print(f"Created {output_file}")
            
        print(f"Successfully created {num_shuffles} shuffled versions!")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
    except Exception as e:
        print(f"Error: {e}")

# Usage
create_shuffles('shuffle_01.csv', 50)
