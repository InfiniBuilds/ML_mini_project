from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / 'data' / 'processed' / 'train.csv'
VALIDATION_PATH = PROJECT_ROOT / 'data' / 'processed' / 'validation.csv'
RESULTS_DIR = PROJECT_ROOT / 'results'
REPORTS_DIR = PROJECT_ROOT / 'reports'

FEATURES = ['G', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'PTS']
TARGET_COLUMN = 'MVP'
EXCLUDED_COLUMNS = {'Player', 'Season', 'Year', 'Share', 'Pts Won', 'Pts Max', TARGET_COLUMN}


def validate_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    required = FEATURES + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'{dataset_name} is missing required columns: {missing}')

    if not set(df[TARGET_COLUMN].dropna().unique()).issubset({0, 1}):
        raise ValueError(f'{dataset_name} target column must be binary, but contains: {sorted(df[TARGET_COLUMN].dropna().unique().tolist())}')

    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    per_season = df.groupby('Season').size()
    if not per_season.eq(3).all():
        bad = per_season[per_season != 3]
        raise ValueError(f'{dataset_name} does not contain exactly three finalists in every season. Problem seasons: {bad.to_dict()}')

    winners = df.groupby('Season')[TARGET_COLUMN].sum()
    if not winners.eq(1).all():
        bad = winners[winners != 1]
        raise ValueError(f'{dataset_name} does not contain exactly one MVP winner in every season. Problem seasons: {bad.to_dict()}')

    unexpected = set(df.columns) & EXCLUDED_COLUMNS
    if unexpected:
        # Keep the check explicit but avoid blocking on columns that are intentionally excluded.
        pass


def load_and_validate_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (TRAIN_PATH, VALIDATION_PATH):
        if not path.exists():
            raise FileNotFoundError(f'Missing dataset file: {path}')

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    validate_dataset(train_df, 'Training data')
    validate_dataset(validation_df, 'Validation data')

    train_df = train_df[FEATURES + ['Season', 'Player', TARGET_COLUMN]].copy()
    validation_df = validation_df[FEATURES + ['Season', 'Player', TARGET_COLUMN]].copy()

    return train_df, validation_df


def compute_binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }
    return metrics


def save_metrics_csv(metrics: dict[str, float | int]) -> pd.DataFrame:
    metrics_df = pd.DataFrame({'metric': list(metrics.keys()), 'value': list(metrics.values())})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(RESULTS_DIR / 'naive_bayes_metrics.csv', index=False)
    return metrics_df


def create_prediction_outputs(
    validation_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series,
) -> pd.DataFrame:
    pred_df = validation_df[['Season', 'Player']].copy()
    pred_df['Actual_MVP'] = y_true.astype(int).values
    pred_df['Predicted_MVP'] = y_pred.astype(int).values
    pred_df['Predicted_Probability_MVP'] = y_prob.astype(float).values
    pred_df = pred_df[['Season', 'Player', 'Actual_MVP', 'Predicted_MVP', 'Predicted_Probability_MVP']]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(RESULTS_DIR / 'naive_bayes_predictions.csv', index=False)
    return pred_df


def create_seasonal_predictions(validation_df: pd.DataFrame, model: GaussianNB) -> pd.DataFrame:
    rows = []
    for season, season_df in validation_df.groupby('Season', sort=True):
        season_df = season_df.sort_values('Player').copy()
        X_season = season_df[FEATURES]
        probs = model.predict_proba(X_season)[:, 1]
        predicted_idx = int(probs.argmax())
        predicted_player = season_df.iloc[predicted_idx]['Player']
        actual_player = season_df.loc[season_df[TARGET_COLUMN] == 1, 'Player'].iloc[0]
        correct = int(actual_player == predicted_player)

        finalist_probs = []
        for _, row in season_df.iterrows():
            finalist_probs.append((row['Player'], float(probs[season_df.index.get_loc(_)])))

        row_out = {
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
        rows.append(row_out)

    seasonal_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seasonal_df.to_csv(RESULTS_DIR / 'naive_bayes_seasonal_predictions.csv', index=False)
    return seasonal_df


def build_report(
    metrics: dict[str, float | int],
    seasonal_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    paper_reproduced: str,
) -> None:
    seasonal_accuracy = seasonal_df['Correct'].mean() if len(seasonal_df) else 0.0
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        'Naive Bayes MVP Classification Report',
        '===================================',
        '',
        'Model implementation:',
        '  Gaussian Naive Bayes (sklearn.naive_bayes.GaussianNB) was used for this baseline.',
        '  This is a reasonable probabilistic baseline for continuous NBA statistics.',
        '  It is not necessarily identical to the custom Naive Bayes method described in the original paper unless that exact implementation is available.',
        '',
        'Features used:',
        f'  {FEATURES}',
        '',
        'Training and validation data:',
        f'  Training rows: {len(train_df)}',
        f'  Validation rows: {len(validation_df)}',
        f'  Training seasons: {sorted(train_df["Season"].unique().tolist())}',
        f'  Validation seasons: {sorted(validation_df["Season"].unique().tolist())}',
        '',
        'Binary classification metrics:',
        f"  Accuracy: {metrics['accuracy']:.4f}",
        f"  Precision: {metrics['precision']:.4f}",
        f"  Recall: {metrics['recall']:.4f}",
        f"  F1-score: {metrics['f1_score']:.4f}",
        f"  Confusion matrix: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}",
        '',
        'Seasonal winner prediction:',
        f"  Seasonal accuracy: {seasonal_accuracy:.4f} ({int(seasonal_df['Correct'].sum())}/{len(seasonal_df)})",
        '',
        'Reference paper comparison:',
        f'  Paper availability in project: {paper_reproduced}',
        '  The original paper could not be verified from this workspace, so the exact reported paper values were not confirmed here.',
        '  This report compares the implementation against the stated methodology rather than an unverified paper number.',
        '',
        'Implementation notes:',
        '  - The validation set is small (12 total rows, 4 seasons), so the reported metrics may be unstable.',
        '  - The model uses the 10 basketball statistics only; Player, Season, and voting columns were excluded.',
        '  - The seasonal prediction task selects the finalist with the highest class-1 probability within each season.',
        '  - Because the final prediction season 2020-21 is missing from the source data, only the available validation seasons were evaluated.',
    ]

    report_path = REPORTS_DIR / 'naive_bayes_report.txt'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nReport written to: {report_path}')


def main() -> None:
    train_df, validation_df = load_and_validate_data()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_val = validation_df[FEATURES]
    y_val = validation_df[TARGET_COLUMN].astype(int)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = pd.Series(model.predict(X_val), index=validation_df.index)
    y_prob = pd.Series(model.predict_proba(X_val)[:, 1], index=validation_df.index)

    metrics = compute_binary_metrics(y_val, y_pred)
    save_metrics_csv(metrics)
    pred_df = create_prediction_outputs(validation_df, y_val, y_pred, y_prob)
    seasonal_df = create_seasonal_predictions(validation_df, model)

    print('Naive Bayes validation metrics:')
    for metric_name, metric_value in metrics.items():
        print(f'  {metric_name}: {metric_value}')
    print('\nPrediction output rows:', len(pred_df))
    print('Seasonal prediction rows:', len(seasonal_df))
    print('Seasonal accuracy:', seasonal_df['Correct'].mean())

    paper_reproduced = 'No paper file was found in the project workspace; exact reported values were not verified.'
    build_report(metrics, seasonal_df, train_df, validation_df, paper_reproduced)

    print('\nFiles created:')
    print(f'  {RESULTS_DIR / "naive_bayes_predictions.csv"}')
    print(f'  {RESULTS_DIR / "naive_bayes_metrics.csv"}')
    print(f'  {RESULTS_DIR / "naive_bayes_seasonal_predictions.csv"}')
    print(f'  {REPORTS_DIR / "naive_bayes_report.txt"}')


if __name__ == '__main__':
    main()
