#  This program is aimed to test the normality of a given dataset
#  The test used will be the shapiro wilk
import pandas
from scipy.stats import shapiro
import sys

def get_dataset_from_csv(csv_path):
    df = pandas.read_csv(csv_path)
    return [ f1 for f1 in df[df.model=="RandomForest"].f1_score ]

def get_dataset_from_xlsx(xlsx_path):
    df = pandas.read_excel(xlsx_path)
    return [ f1 for f1 in df[df.Model=="Rf"].F1_Score ]

def get_dataset(path):
    if path.endswith(".xlsx"):
        return get_dataset_from_xlsx(path)
    return get_dataset_from_csv(path)

def execute_shapiro_wilk(dataset):
    stat, p = shapiro(dataset)
    return stat, p

def main():
    results = pandas.DataFrame(columns=["stat", "p"])
    for input in sys.argv[1:]:
        print("Computing normality test for " + input)
        print("Data:\n", get_dataset(input))
        stat, p = execute_shapiro_wilk(get_dataset(input))
        results = pandas.concat(
            [results, pandas.DataFrame([[stat, p]], columns=["stat", "p"])],
            ignore_index=True
        )
    print(results)
    results.to_csv("normalityTest.csv", index=False)

if __name__ == "__main__":
    main()
