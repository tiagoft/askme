
import pandas as pd

def read_input_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input csv")

    args = parser.parse_args()
    return args

def get_dataset_from_filename(filename: str) -> str:
    if "wikipedia" in filename:
        return "wikipedia"
    elif "bills" in filename:
        return "bills"
    elif "20_newsgroups" in filename:
        return "20_newsgroups"
    elif "agnews" in filename or 'ag_news' in filename:
        return "agnews"
    else:
        return "unknown"

def get_model_from_filename(filename: str) -> str:
    if "gpt-oss" in filename:
        return "gpt-oss:20b"
    elif "llama3.1" in filename:
        return "llama3.1:8b"
    elif "llama3.2" in filename:
        return "llama3.2:2b"
    elif "qwen3" in filename:
        if '8b' in filename:
            return "qwen3:8b"
        elif '14b' in filename:
            return "qwen3:14b"
        return "unknown"

def main():
    args = read_input_args()

    dfs = []
    for input_filename in args.input:
        df = pd.read_csv(input_filename)
        df = df.mean()
        
        df = df.to_frame().T
        #df['source'] = input_filename
        df['dataset'] = get_dataset_from_filename(input_filename)
        df['model'] = get_model_from_filename(input_filename)
        df.reset_index()
        # Move dataset and model columns to the front
        cols = df.columns.tolist()
        cols = ['dataset', 'model'] + [col for col in cols if col not in ['dataset', 'model']]
        df = df[cols]   
        dfs.append(df)
    
    final_df = pd.concat(dfs, axis=0)
    print(final_df.round(2).to_latex(index=False, float_format="%.2f"))
    
if __name__ == "__main__":
    main()
