import os
import glob
from run_judges import process_file

def main():
    input_dir = "claude_judge_friend"
    output_dir = os.path.join(input_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    input_files = sorted(glob.glob(os.path.join(input_dir, "input_*.txt")))
    for idx, fpath in enumerate(input_files, 1):
        fname = os.path.basename(fpath)
        out_name = fname.replace(".txt", ".csv")
        out_path = os.path.join(output_dir, out_name)
        print("==================================================")
        print(f"Processing file {idx}/{len(input_files)}: {fname}")
        print("==================================================")
        try:
            process_file(fpath, out_path, model="qwen2.5:7b")
        except Exception as e:
            print(f"Skipping {fname} due to error: {e}")

if __name__ == "__main__":
    main()
