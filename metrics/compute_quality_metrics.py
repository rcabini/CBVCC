import os
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt
import cv2
from glob import glob

#---------------------------------------------------------------

def load_paths(dataset_dirs, track_dir):
    video_paths = []
    for path in dataset_dirs:
        video_paths += sorted(glob(os.path.join(path, '**', '*.avi'), recursive=True))

    track_paths = sorted(glob(os.path.join(track_dir, '**', '*.csv'), recursive=True))

    video_dict = {os.path.splitext(os.path.basename(v))[0]: v for v in video_paths}
    track_dict = {os.path.splitext(os.path.basename(t))[0]: t for t in track_paths}
    common_keys = set(video_dict.keys()) & set(track_dict.keys())

    return video_dict, track_dict, common_keys

#---------------------------------------------------------------

def process_videos(video_dict, track_dict, common_keys, output_csv):
    SAMPLING_DISTANCE = 1
    TH_DISTANCE_BG = 20 / 0.8
    TH_DISTANCE_FG = 3 / 0.8

    NUM_ENTRIES = len(video_dict)
    overall = np.zeros((NUM_ENTRIES, 15))
    overall_names = []
    ifiles = 0

    for key in sorted(common_keys):
        fn_avi = video_dict[key]
        file = track_dict[key]

        try:
            cap = cv2.VideoCapture(fn_avi)
            if not cap.isOpened():
                raise Exception(f'Error opening {fn_avi}')
        except Exception as e:
            print(f'Error initializing {fn_avi}: {e}')
            continue

        W, H, T = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
        vx = vy = 1

        zstack = np.zeros((H, W, T), dtype=np.uint8)
        for t in range(T):
            ret, frame = cap.read()
            if not ret:
                continue
            zstack[:, :, t] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cap.release()

        spots_IXYT = pd.read_csv(file).to_numpy()
        if len(spots_IXYT) == 0:
            continue

        SNR_T = np.full((T, 1), np.nan)
        CR_T = np.full((T, 1), np.nan)
        HET_T = np.full((T, 1), np.nan)
        DEN_T = np.full(T, np.nan)
        NUM_T = np.full(T, np.nan)

        x_sampling, y_sampling = np.meshgrid(range(0, W, 1), range(0, H, 1))
        x_sampling = x_sampling.flatten()
        y_sampling = y_sampling.flatten()

        for tf in range(T):
            spots_idx = np.where(spots_IXYT[:, 3] == tf)[0]
            if len(spots_idx) == 0:
                continue

            spots_spots_dist = squareform(pdist(spots_IXYT[spots_idx, 1:3]))
            np.fill_diagonal(spots_spots_dist, 9999)
            DEN = np.min(spots_spots_dist)
            DEN = np.nan if DEN in [9999, 0] else DEN
            DEN_T[tf] = DEN
            NUM_T[tf] = len(spots_spots_dist)

            x_sp = np.round(spots_IXYT[spots_idx, 1]).astype(int)
            y_sp = np.round(spots_IXYT[spots_idx, 2]).astype(int)
            BW = np.zeros((H, W), dtype=bool)
            for xi, yi in zip(x_sp, y_sp):
                BW[min(H - 1, max(0, yi)), min(W - 1, max(0, xi))] = 1
            E = distance_transform_edt(~BW)

            is_bg = E[y_sampling, x_sampling] > TH_DISTANCE_BG
            is_fg = E[y_sampling, x_sampling] < TH_DISTANCE_FG
            x_bg, y_bg = x_sampling[is_bg], y_sampling[is_bg]
            x_fg, y_fg = x_sampling[is_fg], y_sampling[is_fg]

            if len(x_fg) < 3 or len(x_bg) < 3:
                continue

            FG_values = zstack[y_fg, x_fg, tf]
            BG_values = zstack[y_bg, x_bg, tf]

            FG_avg, FG_std = np.mean(FG_values), np.std(FG_values)
            BG_avg, BG_std = np.mean(BG_values), np.std(BG_values)

            SNR = np.abs(FG_avg - BG_avg) / np.abs(BG_std)
            CR = FG_avg / BG_avg
            HET = FG_std / np.abs(FG_avg - BG_avg)

            SNR_T[tf, 0] = SNR
            CR_T[tf, 0] = CR
            HET_T[tf, 0] = HET

        mean_SNR = np.nanmean(SNR_T)
        mean_CR = np.nanmean(CR_T)
        mean_HET = np.nanmean(HET_T)
        mean_DEN = np.nanmean(DEN_T)
        std_DEN = np.nanstd(DEN_T)
        mean_NUM = np.nanmean(NUM_T)
        std_NUM = np.nanstd(NUM_T)

        overall[ifiles, :] = [
            mean_SNR, mean_CR, mean_DEN, std_DEN,
            mean_NUM, std_NUM, W, H, 1, 1, T,
            vx, vy, 1, len(np.unique(spots_IXYT[:, 0]))
        ]
        overall_names.append(key + '.avi')
        ifiles += 1

    overall = overall[:ifiles]
    overall_names = overall_names[:ifiles]
    columns = ['SNR', 'CR', 'DEN avg', 'DEN std', 'NUM avg', 'NUM std',
               'W', 'H', 'D', 'C', 'T', 'dxy', 'dz', 'dt', 'N.TRACKS']

    df = pd.DataFrame(overall, columns=columns, index=overall_names)
    df.to_csv(output_csv)
    print(f'\nWritten quality metrics to: {output_csv}')
    
#---------------------------------------------------------------

def main(args):
    video_dict, track_dict, common_keys = load_paths(args.datasets, args.tracks)
    print(f'Found {len(common_keys)} valid video/track pairs.')
    process_videos(video_dict, track_dict, common_keys, args.output)

#---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute quality metrics for CBVCC.")
    parser.add_argument('--datasets', nargs='+', required=True, help='List of dataset directories to search for videos.')
    parser.add_argument('--tracks', required=True, help='Path to the directory containing tracking CSVs.')
    parser.add_argument('--output', default='quality_overall.csv', help='Output CSV filename (default: quality_overall.csv)')

    args = parser.parse_args()
    main(args)

