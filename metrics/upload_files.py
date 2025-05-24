import pandas as pd
import numpy as np
from glob import glob
import os

#---------------------------------------------------------------

def load_track_counts(tracks_dir):
    ditr = []
    for f in glob(os.path.join(tracks_dir, "*", "*", "*.csv")):
        dtr = pd.read_csv(f)
        ditr.append([os.path.basename(f).replace('.csv', '.avi'), len(np.unique(dtr.id))])
    counts = pd.DataFrame(ditr, columns=['file_id', 'count'])
    counts.set_index('file_id', inplace=True)
    return counts

#---------------------------------------------------------------

def load_gt(gt_path):
    gt = pd.read_csv(gt_path, header=None)
    gt.columns = ['file_id', 'gt']
    gt.set_index('file_id', inplace=True)
    return gt

#---------------------------------------------------------------

def preprocess_submission(file_path):
    df = pd.read_csv(file_path)
    df = df[df['score'] != 0]
    df = df.drop_duplicates(subset=df.columns[df.columns.isin(['id_submission', 'ts']) == False])
    df['ts'] = pd.to_datetime(df['ts']) - pd.Timedelta(hours=1)
    df = df[df['ts'].dt.year != 2025]
    df['timestamp'] = df['ts'].astype(int) // 10**9
    df = df.sort_values("score", ascending=False)
    return df

#---------------------------------------------------------------

def build_file_timestamp_dict(predictions_glob):
    file_names = glob(predictions_glob)
    valid_files = [file for file in file_names if file.endswith('.csv')]
    return {
        int(os.path.basename(file).split('predicted')[1].split('.csv')[0]): file
        for file in valid_files
    }

#---------------------------------------------------------------

def get_nearest_file(timestamp, file_timestamp_dict):
    closest_timestamp = min(file_timestamp_dict.keys(), key=lambda x: abs(x - timestamp))
    return file_timestamp_dict[closest_timestamp]

#---------------------------------------------------------------

def build_all_data(df, file_timestamp_dict, gt_df):
    df['file_path'] = df['timestamp'].apply(lambda x: get_nearest_file(x, file_timestamp_dict))
    df = df.sort_values(by=['team', 'score'], ascending=[True, False]).drop_duplicates(subset=['team'], keep='first')
    df = df.sort_values("score", ascending=False)

    all_data = pd.DataFrame()
    for _, row in df.iterrows():
        file_data = pd.read_csv(row['file_path'], header=None)
        file_data.columns = ['file_id', row['team']]
        file_data['file_id'] = file_data['file_id'].apply(
            lambda x: '_'.join([part.zfill(2) if part.isdigit() else part for part in x.split('_')])
        )

        if all_data.empty:
            all_data = file_data[['file_id', row['team']]]
        else:
            all_data = pd.merge(all_data, file_data[['file_id', row['team']]], on='file_id', how='outer')

    all_data = pd.merge(all_data, gt_df, on='file_id', how='outer')
    all_data.set_index('file_id', inplace=True)
    return all_data
