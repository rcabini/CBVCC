import pandas as pd
import argparse
import os
from metrics.upload_files import (
    load_track_counts,
    load_gt,
    preprocess_submission,
    build_file_timestamp_dict,
    build_all_data
)
from metrics.roc_curves import plot_roc_curves, clean_model_name
from metrics.overall_metrics import evaluate_models
from metrics.ncell_curves import plot_score_vs_cells
from metrics.snr_curves import plot_score_vs_snr
from metrics.descriptives import plot_class_distribution, plot_metric_distributions

#---------------------------------------------------------------

def load_analysis(tracks_path, submission_file, gt_file, timestamp_pattern, output_file='./evaluation_metrics.csv'):
    #counts = load_track_counts(tracks_path)
    #counts.to_csv('counts.csv', index=True)
    gt = load_gt(gt_file)
    submission_df = preprocess_submission(submission_file)
    file_timestamp_dict = build_file_timestamp_dict(timestamp_pattern)
    all_data = build_all_data(submission_df, file_timestamp_dict, gt)

    metrics_df = evaluate_models(all_data)
    metrics_df.to_csv(output_file, index=False)
    print(metrics_df)
    return all_data, metrics_df, gt

#---------------------------------------------------------------

def main(args):

    # Phase 2 - Test
    all_data_2, _, gt2 = load_analysis(
        tracks_path=args.tracks_path,
        submission_file=args.submission2,
        gt_file=args.gt2,
        timestamp_pattern=args.timestamp_pattern,
        output_file=os.path.join(args.output_path,'evaluation_metrics2.csv')
    )
    # Quality metrics
    quality_metric = pd.read_csv(args.quality_csv, index_col='Unnamed: 0', usecols=['SNR', 'Unnamed: 0', 'N.TRACKS'])
    # Training ground truth
    gt_train = load_gt(args.gt_train)
    # Phase 1 - Validation
    all_data_1, _, gt1 = load_analysis(
        tracks_path=args.tracks_path,
        submission_file=args.submission1,
        gt_file=args.gt1,
        timestamp_pattern=args.timestamp_pattern,
        output_file=os.path.join(args.output_path,'evaluation_metrics1.csv')
    )
    
    # Plots - Test
    style_dict = plot_roc_curves(all_data_2, output_path=os.path.join(args.output_path,'roc2.png'))
    plot_score_vs_cells(all_data_2, quality_metric.copy(), style_dict, clean_name_fn=clean_model_name,
                        output_path=os.path.join(args.output_path,'ncell.png'))
    plot_score_vs_snr(all_data_2, quality_metric.copy(), style_dict, clean_name_fn=clean_model_name,
                      output_path=os.path.join(args.output_path,'snr.png'))
    # Plots - Validation
    plot_roc_curves(all_data_1, style_dict=style_dict, output_path=os.path.join(args.output_path,'roc1.png'))

    # Plots general descriptives
    plot_class_distribution(quality_metric.copy(), gt1, gt2, gt_train, output_path=os.path.join(args.output_path,'class.png'))
    plot_metric_distributions(quality_metric.copy(), gt1, gt2, output_path=os.path.join(args.output_path,'metric.png'))

#---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate challenge submission")
    
    parser.add_argument('--tracks_path', required=True, help='Path to the tracks folder')
    parser.add_argument('--submission1', required=True, help='Submission file for Phase 1')
    parser.add_argument('--gt1', required=True, help='Ground truth CSV for Phase 1')
    parser.add_argument('--submission2', required=True, help='Submission file for Phase 2')
    parser.add_argument('--gt2', required=True, help='Ground truth CSV for Phase 2')
    parser.add_argument('--timestamp_pattern', required=True, help='Pattern for predicted file timestamps (e.g., "/uploaded_files/predicted1*")')
    parser.add_argument('--quality_csv', required=True, help='Path to quality_overall.csv')
    parser.add_argument('--gt_train', required=True, help='Training ground truth file')
    parser.add_argument('--output_path', required=True, help='Optional output folder path')

    args = parser.parse_args()
    main(args)

