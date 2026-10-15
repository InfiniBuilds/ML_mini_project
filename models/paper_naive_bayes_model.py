from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / 'data' / 'processed' / 'train.csv'
VALIDATION_PATH = PROJECT_ROOT / 'data' / 'processed' / 'validation.csv'
RESULTS_DIR = PROJECT_ROOT / 'results'
REPORTS_DIR = PROJECT_ROOT / 'reports'

FEATURES = ['G', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'PTS']
TARGET_COLUMN = 'MVP'


class PaperNaiveBayes:
    """Gaussian-style Naive Bayes approximating the class-conditional method described in the paper.

    The project PDF describes a per-class likelihood model using class-wise averages and a
    log-likelihood calculation for each feature, rather than a library implementation. This class
    reconstructs that procedure in a standard Gaussian Naive Bayes form while preserving the
    paper's binary classification and per-season finalist selection logic.
    """

    def __init__(self, epsilon: float = 1e-9) -> None:
        self.epsilon = epsilon
        self.classes_: np.ndarray | None = None
        self.class_priors_: dict[int, float] | None = None
        self.mean_: np.ndarray | None = None
        self.var_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PaperNaiveBayes':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.unique(y)
        self.class_priors_ = {label: float(np.mean(y == label)) for label in self.classes_}

        self.mean_ = np.vstack([X[y == label].mean(axis=0) for label in self.classes_])
        self.var_ = np.vstack([X[y == label].var(axis=0) + self.epsilon for label in self.classes_])
        return self

    def _log_gaussian(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        return -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var, axis=1)

    def _log_prob_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.mean_ is None or self.var_ is None:
            raise ValueError('The model must be fitted before calling predict_proba.')

        X = np.asarray(X, dtype=float)
        log_probs = np.empty((len(X), len(self.classes_)))

        for idx, label in enumerate(self.classes_):
            mean = self.mean_[idx]
            var = self.var_[idx]
            log_probs[:, idx] = np.log(self.class_priors_[int(label)]) + self._log_gaussian(X, mean, var)

        max_log = np.max(log_probs, axis=1, keepdims=True)
        shifted = np.exp(log_probs - max_log)
        denom = np.sum(shifted, axis=1, keepdims=True)
        return shifted / denom

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._log_prob_matrix(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.array(self.classes_)[np.argmax(proba, axis=1)]


def validate_split(df: pd.DataFrame, label: str) -> None:
    required = FEATURES + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'{label} is missing required columns: {missing}')

    unique_target = sorted(pd.Series(df[TARGET_COLUMN].dropna().unique()).astype(int).unique().tolist())
    if unique_target not in ([0], [1], [0, 1]):
        raise ValueError(f'{label} target must be binary. Found unique values: {unique_target}')

    per_season = df.groupby('Season').size()
    if not per_season.eq(3).all():
        bad = per_season[per_season != 3]
        raise ValueError(f'{label} does not contain exactly three finalists per season: {bad.to_dict()}')

    winners = df.groupby('Season')[TARGET_COLUMN].sum()
    if not winners.eq(1).all():
        bad = winners[winners != 1]
        raise ValueError(f'{label} does not contain exactly one MVP winner per season: {bad.to_dict()}')


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (TRAIN_PATH, VALIDATION_PATH):
        if not path.exists():
            raise FileNotFoundError(f'Missing dataset: {path}')

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    validate_split(train_df, 'Training data')
    validate_split(validation_df, 'Validation data')

    train_df = train_df[FEATURES + ['Season', 'Player', TARGET_COLUMN]].copy()
    validation_df = validation_df[FEATURES + ['Season', 'Player', TARGET_COLUMN]].copy()
    return train_df, validation_df


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def save_binary_predictions(validation_df: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> pd.DataFrame:
    pred_df = validation_df[['Season', 'Player']].copy()
    pred_df['Actual_MVP'] = y_true.astype(int).to_numpy()
    pred_df['Predicted_MVP'] = y_pred.astype(int).to_numpy()
    pred_df['Predicted_Probability_MVP'] = y_prob.astype(float).to_numpy()
    pred_df = pred_df[['Season', 'Player', 'Actual_MVP', 'Predicted_MVP', 'Predicted_Probability_MVP']]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(RESULTS_DIR / 'paper_naive_bayes_predictions.csv', index=False)
    return pred_df


def save_binary_metrics(metrics: dict[str, float | int]) -> None:
    df = pd.DataFrame({'metric': list(metrics.keys()), 'value': list(metrics.values())})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / 'paper_naive_bayes_metrics.csv', index=False)


def seasonal_predictions(model: PaperNaiveBayes, validation_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, season_df in validation_df.groupby('Season', sort=True):
        season_df = season_df.sort_values('Player').copy()
        X = season_df[FEATURES].to_numpy(float)
        probs = model.predict_proba(X)[:, 1]

        predicted_idx = int(np.argmax(probs))
        predicted_player = season_df.iloc[predicted_idx]['Player']
        actual_player = season_df.loc[season_df[TARGET_COLUMN] == 1, 'Player'].iloc[0]
        correct = int(actual_player == predicted_player)

        row = {
            'Season': season,
            'Actual_MVP': actual_player,
            'Predicted_MVP': predicted_player,
            'Correct': correct,
            'Finalist_1_Player': season_df.iloc[0]['Player'],
            'Finalist_1_Probability': float(probs[0]),
            'Finalist_2_Player': season_df.iloc[1]['Player'],
            'Finalist_2_Probability': float(probs[1]),
            'Finalist_3_Player': season_df.iloc[2]['Player'],
            'Finalist_3_Probability': float(probs[2]),
            'Predicted_Probability_MVP': float(probs[predicted_idx]),
        }
        rows.append(row)

    seasonal_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seasonal_df.to_csv(RESULTS_DIR / 'paper_naive_bayes_seasonal_predictions.csv', index=False)
    return seasonal_df


def create_methodology_comparison() -> pd.DataFrame:
    baseline_metrics_path = RESULTS_DIR / 'naive_bayes_metrics.csv'
    baseline_response_path = RESULTS_DIR / 'naive_bayes_seasonal_predictions.csv'

    baseline_metrics = pd.read_csv(baseline_metrics_path)
    baseline_map = {row.metric: row.value for _, row in baseline_metrics.iterrows()}
    baseline_seasonal_accuracy = pd.read_csv(baseline_response_path)['Correct'].mean() if baseline_response_path.exists() else np.nan

    rows = [
        {
            'model': 'GaussianNB_baseline',
            'accuracy': baseline_map.get('accuracy', np.nan),
            'precision': baseline_map.get('precision', np.nan),
            'recall': baseline_map.get('recall', np.nan),
            'f1_score': baseline_map.get('f1_score', np.nan),
            'seasonal_accuracy': baseline_seasonal_accuracy,
            'notes': 'sklearn GaussianNB using the direct validation split without custom paper logic',
        }
    ]

    return pd.DataFrame(rows)


def write_comparison_report(paper_metrics: dict[str, float | int], paper_seasonal_accuracy: float) -> None:
    baseline_metrics_path = RESULTS_DIR / 'naive_bayes_metrics.csv'
    baseline_seasonal_path = RESULTS_DIR / 'naive_bayes_seasonal_predictions.csv'
    baseline_df = pd.read_csv(baseline_metrics_path)
    baseline_map = {row.metric: row.value for _, row in baseline_df.iterrows()}
    baseline_seasonal = pd.read_csv(baseline_seasonal_path)['Correct'].mean() if baseline_seasonal_path.exists() else np.nan

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        'Paper Naive Bayes vs. GaussianNB Baseline',
        '========================================',
        '',
        'Implementation details:',
        '  The paper describes a Gaussian-style Naive Bayes classifier with class-wise averages and a log-likelihood calculation for each feature.',
        '  The exact variance smoothing, class-prior treatment, and probability threshold are not fully specified in the text, so this is a reasonable reconstruction rather than an exact byte-for-byte reproduction.',
        '',
        'Differences from the current GaussianNB baseline:',
        '  - Baseline model: sklearn.naive_bayes.GaussianNB',
        '  - Paper-style model: custom Gaussian likelihood using class-specific means and variances, following the methodology described in the paper text.',
        '  - Baseline predicts on the validation set as a binary classification problem without a separate season-wise winner selection step.',
        '  - Paper-style seasonal evaluation groups by season and selects the finalist with the highest class-1 score among the three finalists.',
        '  - The paper also notes that they suppressed the larger all-player dataset and kept only the top three finalists each season to avoid severe class imbalance.',
        '',
        'Binary metrics (paper-style model):',
        f"  Accuracy: {paper_metrics['accuracy']:.4f}",
        f"  Precision: {paper_metrics['precision']:.4f}",
        f"  Recall: {paper_metrics['recall']:.4f}",
        f"  F1-score: {paper_metrics['f1_score']:.4f}",
        f"  Confusion matrix: TP={paper_metrics['tp']}, FP={paper_metrics['fp']}, FN={paper_metrics['fn']}, TN={paper_metrics['tn']}",
        '',
        'Seasonal winner prediction (paper-style model):',
        f'  Seasonal accuracy: {paper_seasonal_accuracy:.4f} ({int(paper_seasonal_accuracy * 4)}/{4})',
        '',
        'Baseline comparison (existing GaussianNB implementation):',
        f"  Accuracy: {baseline_map.get('accuracy', np.nan):.4f}",
        f"  Precision: {baseline_map.get('precision', np.nan):.4f}",
        f"  Recall: {baseline_map.get('recall', np.nan):.4f}",
        f"  F1-score: {baseline_map.get('f1_score', np.nan):.4f}",
        f"  Seasonal accuracy: {baseline_seasonal:.4f} ({int(round(baseline_seasonal * 4))}/{4})",
        '',
        'Important limitations:',
        '  - The paper does not specify all implementation details such as the exact variance smoothing constant or the exact probability calibration logic.',
        '  - The original paper text does not provide enough detail to guarantee an exact byte-for-byte reproduction of its custom Naive Bayes routine.',
        '  - The training and validation data are the project’s prepared split and were not modified for this comparison.',
    ]

    (REPORTS_DIR / 'paper_naive_bayes_report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    train_df, validation_df = load_data()

    model = PaperNaiveBayes()
    model.fit(train_df[FEATURES].to_numpy(float), train_df[TARGET_COLUMN].to_numpy(int))

    y_true = validation_df[TARGET_COLUMN].astype(int)
    X_val = validation_df[FEATURES].to_numpy(float)
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    paper_metrics = binary_metrics(y_true, pd.Series(y_pred, index=y_true.index))
    save_binary_metrics(paper_metrics)
    save_binary_predictions(validation_df, y_true, pd.Series(y_pred, index=y_true.index), pd.Series(y_prob, index=y_true.index))
    seasonal = seasonal_predictions(model, validation_df)

    seasonal_accuracy = float(seasonal['Correct'].mean()) if len(seasonal) else 0.0
    write_comparison_report(paper_metrics, seasonal_accuracy)

    comparison_df = create_methodology_comparison()
    comparison_df = pd.concat([
        comparison_df,
        pd.DataFrame([
            {
                'model': 'PaperNaiveBayes_reconstruction',
                'accuracy': paper_metrics['accuracy'],
                'precision': paper_metrics['precision'],
                'recall': paper_metrics['recall'],
                'f1_score': paper_metrics['f1_score'],
                'seasonal_accuracy': seasonal_accuracy,
                'notes': 'Custom Gaussian Naive Bayes approximation based on the paper text and the available project data',
            }
        ])
    ], ignore_index=True)
    comparison_df.to_csv(RESULTS_DIR / 'naive_bayes_methodology_comparison.csv', index=False)

    print('Paper Naive Bayes metrics:')
    for key, value in paper_metrics.items():
        print(f'  {key}: {value}')
    print('Seasonal accuracy:', seasonal_accuracy)
    print('Comparison file:', RESULTS_DIR / 'naive_bayes_methodology_comparison.csv')
    print('Report file:', REPORTS_DIR / 'paper_naive_bayes_report.txt')


if __name__ == '__main__':
    main()
