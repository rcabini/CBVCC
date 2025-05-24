import os
import pandas as pd

def generate_label_csv(root_dir, output_csv):
    records = []
    for subset_folder in sorted(os.listdir(root_dir)):
        subset_path = os.path.join(root_dir, subset_folder)
        if not os.path.isdir(subset_path):
            continue
        for class_label in ['0', '1']:
            class_folder = os.path.join(subset_path, class_label)
            if not os.path.isdir(class_folder):
                continue
            for fname in sorted(os.listdir(class_folder)):
                if fname.endswith('.avi'):
                    records.append({
                        'filename': fname,
                        'class': int(class_label)
                    })
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, header=False)
    print(f'CSV done: {output_csv}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV of filenames and classes from dataset folders.")
    parser.add_argument('--root_dir', help='Root directory of the dataset (e.g. Dataset/subset)')
    parser.add_argument('--output_csv', help='Output CSV filename')

    args = parser.parse_args()
    generate_label_csv(args.root_dir, args.output_csv)
