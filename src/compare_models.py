from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = PROJECT_ROOT / 'results'
FEATURES = ['G', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'PTS']
TARGET_COLUMN = 'MVP'


def validate_split_consistency() -> None:
    train_df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'train.csv')
    validation_df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'validation.csv')

    if list(train_df.columns) != list(validation_df.columns):
        raise ValueError('Training and validation files have different columns; the split is inconsistent.')

    if set(FEATURES) - set(train_df.columns):
        raise ValueError(f'Feature set mismatch: missing required feature columns in processed data: {FEATURES}')

    if TARGET_COLUMN in FEATURES:
        raise ValueError('Feature leak detected: target label appears in the feature list.')

    if len(train_df) != len(train_df.drop_duplicates()):
        raise ValueError('Training split contains duplicate rows; the exact split is not consistent.')

    if len(validation_df) != len(validation_df.drop_duplicates()):
        raise ValueError('Validation split contains duplicate rows; the exact split is not consistent.')

    train_seasons = set(train_df['Season'])
    validation_seasons = set(validation_df['Season'])
    if train_seasons & validation_seasons:
        raise ValueError('Training and validation seasons overlap; data leakage via season overlap detected.')


def read_metric_file(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f'Missing model metrics file: {path}')
    metric_df = pd.read_csv(path)
    metric_values = metric_df.set_index('metric')['value'].to_dict()
    return {key: float(value) for key, value in metric_values.items() if key in {'accuracy', 'precision', 'recall', 'f1_score'}}


def read_seasonal_accuracy(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f'Missing seasonal predictions file: {path}')
    seasonal_df = pd.read_csv(path)
    if 'Correct' not in seasonal_df.columns:
        raise ValueError(f'Seasonal prediction file missing Correct column: {path}')
    return float(seasonal_df['Correct'].mean())


def build_comparison_table() -> pd.DataFrame:
    model_files = {
        'Naive Bayes': {
            'metrics': RESULTS_DIR / 'naive_bayes_metrics.csv',
            'seasonal': RESULTS_DIR / 'naive_bayes_seasonal_predictions.csv',
        },
        'Logistic Regression': {
            'metrics': RESULTS_DIR / 'logistic_regression_metrics.csv',
            'seasonal': RESULTS_DIR / 'logistic_regression_seasonal_predictions.csv',
        },
        'Random Forest': {
            'metrics': RESULTS_DIR / 'random_forest_metrics.csv',
            'seasonal': RESULTS_DIR / 'random_forest_seasonal_predictions.csv',
        },
    }

    rows = []
    for model_name, file_map in model_files.items():
        metrics = read_metric_file(file_map['metrics'])
        seasonal_accuracy = read_seasonal_accuracy(file_map['seasonal'])
        rows.append({
            'model': model_name,
            'binary_accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'seasonal_accuracy': seasonal_accuracy,
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df[['model', 'binary_accuracy', 'precision', 'recall', 'f1_score', 'seasonal_accuracy']]
    comparison_df.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    return comparison_df


def save_accuracy_chart(comparison_df: pd.DataFrame) -> None:
    model_names = comparison_df['model'].tolist()
    binary_accuracy = comparison_df['binary_accuracy'].tolist()
    seasonal_accuracy = comparison_df['seasonal_accuracy'].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].bar(model_names, binary_accuracy, color=['#4C78A8', '#F58518', '#54A24B'])
    axes[0].set_title('Binary classification accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis='x', rotation=20)

    axes[1].bar(model_names, seasonal_accuracy, color=['#4C78A8', '#F58518', '#54A24B'])
    axes[1].set_title('Seasonal MVP prediction accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis='x', rotation=20)

    fig.suptitle('Model comparison: binary accuracy vs seasonal MVP accuracy')
    fig.savefig(RESULTS_DIR / 'model_comparison.png', dpi=200)
    plt.close(fig)


def main() -> None:
    validate_split_consistency()
    comparison_df = build_comparison_table()
    save_accuracy_chart(comparison_df)

    print('Model comparison table:')
    print(comparison_df.to_string(index=False))
    print('\nSaved comparison CSV to:', RESULTS_DIR / 'model_comparison.csv')
    print('Saved chart to:', RESULTS_DIR / 'model_comparison.png')


if __name__ == '__main__':
    main()
