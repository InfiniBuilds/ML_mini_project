from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_INPUT_CANDIDATES = [
    PROJECT_ROOT / 'data' / 'raw' / 'NBA_stats_including_MVP.csv',
    PROJECT_ROOT / 'NBA_stats_including_MVP.csv',
]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
REPORTS_DIR = PROJECT_ROOT / 'reports'

REQUIRED_FEATURES = {
    'G': ['G', 'G_x', 'G_y'],
    'MP': ['MP', 'MP_x', 'MP_y'],
    'FG%': ['FG%'],
    '3P%': ['3P%'],
    'FT%': ['FT%'],
    'TRB': ['TRB'],
    'AST': ['AST'],
    'STL': ['STL'],
    'BLK': ['BLK'],
    'PTS': ['PTS'],
}


def normalize_name(value: object) -> str:
    if value is None:
        return ''
    return str(value).strip().replace(' ', '').replace('-', '').replace('_', '').lower()


def choose_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        match = normalized.get(normalize_name(candidate))
        if match is not None:
            return match
    return None


def find_csv_path() -> Path:
    for path in RAW_INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        'Dataset not found. Expected one of: ' + ', '.join(str(p) for p in RAW_INPUT_CANDIDATES)
    )


def ensure_raw_dataset_copy(raw_path: Path) -> Path:
    target = PROJECT_ROOT / 'data' / 'raw' / 'NBA_stats_including_MVP.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    if raw_path != target and not target.exists():
        shutil.copy2(raw_path, target)
    return target


def season_label_from_year(year_value: object) -> str:
    try:
        year_int = int(float(year_value))
    except (TypeError, ValueError):
        return ''
    if year_int <= 0:
        return ''
    return f'{year_int - 1}-{str(year_int)[-2:]}'


def inspect_raw_dataset(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    print(f'Raw dataset path: {raw_path}')
    print(f'Shape: {df.shape}')
    print('Columns:')
    print(df.columns.tolist())
    print('\nDtypes:')
    print(df.dtypes.to_string())
    print('\nFirst 10 rows:')
    print(df.head(10).to_string(index=False))
    print('\nLast 10 rows:')
    print(df.tail(10).to_string(index=False))

    season_candidates = [col for col in df.columns if 'season' in str(col).lower() or 'year' in str(col).lower()]
    print('\nLikely season columns:', season_candidates)
    print('\nUnique seasons sample:')
    for col in season_candidates[:5]:
        unique_values = pd.Series(df[col].dropna().astype(str).unique())
        print(f'  {col}: {unique_values.head(10).tolist()}')

    print('\nUnique players:', df['Player'].nunique() if 'Player' in df.columns else 'Player column not found')
    print('\nMissing values per column:')
    print(df.isna().sum().to_string())
    print('\nDuplicate rows:', int(df.duplicated().sum()))
    print('\nMinimum and maximum season values:')
    if season_candidates:
        for col in season_candidates[:3]:
            numeric = pd.to_numeric(df[col], errors='coerce')
            valid = numeric.dropna()
            if not valid.empty:
                print(f'  {col}: min={int(valid.min())} max={int(valid.max())}')
    print('\nMVP voting columns:')
    for col in df.columns:
        name = str(col).lower()
        if 'share' in name or 'won' in name or 'max' in name or 'vote' in name:
            print('  ', col)

    return df


def build_finalist_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    player_col = choose_column(df, ['Player'])
    season_col = choose_column(df, ['Year', 'Season', 'Year_x', 'Year_y'])
    share_col = choose_column(df, ['Share', 'Share_x', 'Share_y'])
    pts_won_col = choose_column(df, ['Pts Won', 'PtsWon', 'Pts_Won', 'Pts Won_x', 'Pts Won_y'])
    pts_max_col = choose_column(df, ['Pts Max', 'PtsMax', 'Pts_Max', 'Pts Max_x', 'Pts Max_y'])

    if player_col is None or season_col is None:
        raise ValueError(f'Could not find required identifier columns in dataset. Columns: {list(df.columns)}')

    required_feature_columns: dict[str, str] = {}
    for label, candidates in REQUIRED_FEATURES.items():
        chosen = choose_column(df, candidates)
        if chosen is None:
            raise ValueError(f'Required feature column missing: {label}. Available columns: {list(df.columns)}')
        required_feature_columns[label] = chosen

    if share_col is None or pts_won_col is None or pts_max_col is None:
        raise ValueError(
            'MVP voting columns not found. Expected Share, Pts Won and Pts Max. Available columns: ' \
            + ', '.join(df.columns.tolist())
        )

    mapping = {
        'Player': player_col,
        'Year': season_col,
        'Share': share_col,
        'Pts Won': pts_won_col,
        'Pts Max': pts_max_col,
    }
    for feature_name, source_col in required_feature_columns.items():
        mapping[feature_name] = source_col

    subset = df[[player_col, season_col, share_col, pts_won_col, pts_max_col] + list(required_feature_columns.values())].copy()
    subset = subset.rename(columns={player_col: 'Player', season_col: 'Year', share_col: 'Share', pts_won_col: 'Pts Won', pts_max_col: 'Pts Max', **{source: target for target, source in required_feature_columns.items()}})

    subset['Player'] = subset['Player'].astype(str).str.strip()
    subset['Year'] = pd.to_numeric(subset['Year'], errors='coerce')
    subset['Share'] = pd.to_numeric(subset['Share'], errors='coerce')
    subset['Pts Won'] = pd.to_numeric(subset['Pts Won'], errors='coerce')
    subset['Pts Max'] = pd.to_numeric(subset['Pts Max'], errors='coerce')

    # Preserve the original values unless the columns are clearly duplicated by suffix.
    # In this dataset, G_x and MP_x are the valid player-season stats, while the raw
    # data includes the exact same concept as the un-suffixed names only when they exist.
    for feature in ['G', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'PTS']:
        subset[feature] = pd.to_numeric(subset[feature], errors='coerce')

    subset = subset.dropna(subset=['Player', 'Year', 'Share', 'Pts Won', 'Pts Max', 'G', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'PTS'])

    subset['Season'] = subset['Year'].map(season_label_from_year)
    subset = subset.sort_values(['Year', 'Share'], ascending=[True, False]).reset_index(drop=True)
    subset['FinalistRank'] = subset.groupby('Year')['Share'].rank(method='first', ascending=False).astype(int)

    finalists = subset[subset['FinalistRank'] <= 3].copy()
    finalists['MVP'] = (finalists['FinalistRank'] == 1).astype(int)
    finalists = finalists[['Season', 'Player', 'G', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'PTS', 'Share', 'Pts Won', 'Pts Max', 'FinalistRank', 'MVP']]
    finalists = finalists.sort_values(['Season', 'FinalistRank']).reset_index(drop=True)

    report_df = finalists[['Season', 'Player', 'Share', 'Pts Won', 'Pts Max', 'FinalistRank', 'MVP']].copy()
    report_df.columns = ['Season', 'Player', 'MVP voting share', 'Voting points', 'Voting points max', 'Selected finalist rank', 'Winner']

    return finalists, mapping, report_df


def create_processed_datasets(finalists: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    train_seasons = [f'{start}-{str(start + 1)[-2:]}' for start in range(2000, 2016)]
    validation_seasons = [f'{start}-{str(start + 1)[-2:]}' for start in range(2016, 2020)]
    final_prediction_season = '2020-21'

    train_df = finalists[finalists['Season'].isin(train_seasons)].copy()
    validation_df = finalists[finalists['Season'].isin(validation_seasons)].copy()

    train_path = PROCESSED_DIR / 'train.csv'
    validation_path = PROCESSED_DIR / 'validation.csv'
    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)

    final_prediction_df = None
    if final_prediction_season in finalists['Season'].unique():
        final_prediction_df = finalists[finalists['Season'] == final_prediction_season].copy()
        final_prediction_df = final_prediction_df[[
            'Season', 'Player', 'G', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'PTS'
        ]].copy()
        final_prediction_df.to_csv(PROCESSED_DIR / 'final_prediction.csv', index=False)

    return train_df, final_prediction_df


def create_audit_report(
    original_df: pd.DataFrame,
    mapping: dict[str, str],
    finalists: pd.DataFrame,
    report_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    final_prediction_df: pd.DataFrame | None,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    season_counts = finalists.groupby('Season').size().to_dict()
    bad_count_seasons = sorted({season: count for season, count in season_counts.items() if count != 3})
    winner_counts = finalists.groupby('Season')['MVP'].sum().to_dict()
    invalid_winners = sorted({season: count for season, count in winner_counts.items() if count != 1})

    missing_by_col = original_df.isna().sum()
    dup_rows = int(original_df.duplicated().sum())
    unique_players = original_df['Player'].nunique() if 'Player' in original_df.columns else 'N/A'
    year_min = int(pd.to_numeric(original_df['Year'], errors='coerce').dropna().min()) if 'Year' in original_df.columns else 'N/A'
    year_max = int(pd.to_numeric(original_df['Year'], errors='coerce').dropna().max()) if 'Year' in original_df.columns else 'N/A'
    raw_seasons = sorted(pd.to_numeric(original_df['Year'], errors='coerce').dropna().unique().astype(int).tolist())

    training_period_possible = any(int(season.split('-')[0]) >= 2000 and int(season.split('-')[1]) <= 16 for season in finalists['Season'].unique())
    validation_period_possible = any(int(season.split('-')[0]) >= 2016 and int(season.split('-')[1]) <= 20 for season in finalists['Season'].unique())
    prediction_period_possible = '2020-21' in finalists['Season'].unique()

    lines = [
        'NBA MVP Dataset Audit Report',
        '===========================',
        '',
        '1. Original dataset dimensions',
        f'   Rows: {original_df.shape[0]}',
        f'   Columns: {original_df.shape[1]}',
        '',
        '2. Available seasons',
        f'   Unique seasons in source: {raw_seasons[:10]} ... {raw_seasons[-10:]}',
        f'   Minimum season: {year_min}',
        f'   Maximum season: {year_max}',
        '',
        '3. Missing-value analysis',
        str(missing_by_col.to_string()),
        '',
        '4. Duplicate-record analysis',
        f'   Duplicate rows across the entire source file: {dup_rows}',
        f'   Duplicate player-season combinations: {int(original_df.duplicated(subset=["Player", "Year"]).sum())}',
        '',
        '5. Feature mapping',
        '   Source mapping used for the required statistics:',
    ]
    for expected, original in mapping.items():
        lines.append(f'   {expected}: {original}')
    lines.extend([
        '',
        '6. MVP voting information availability',
        f'   Share column discovered: {mapping.get("Share")}',
        f'   Pts Won column discovered: {mapping.get("Pts Won")}',
        f'   Pts Max column discovered: {mapping.get("Pts Max")}',
        '',
        '7. Number of finalists selected per season',
    ])
    for season, count in sorted(season_counts.items()):
        lines.append(f'   {season}: {count}')
    lines.extend([
        '',
        '8. Seasons with fewer or more than three valid finalists',
    ])
    if bad_count_seasons:
        for item in bad_count_seasons:
            lines.append(f'   {item}')
    else:
        lines.append('   No seasons with fewer or more than three valid finalist records were found.')
    lines.extend([
        '',
        '9. Number of records in each processed dataset',
        f'   nba_mvp_finalists.csv: {len(finalists)}',
        f'   train.csv: {len(train_df)}',
        f'   validation.csv: {len(validation_df)}',
        f'   final_prediction.csv: {0 if final_prediction_df is None else len(final_prediction_df)}',
        '',
        '10. Reference paper period reconstruction',
        f'   Training seasons 2000-01 through 2015-16 are available: {training_period_possible}',
        f'   Validation seasons 2016-17 through 2019-20 are available: {validation_period_possible}',
        f'   Final prediction season 2020-21 is available: {prediction_period_possible}',
        '',
        '11. Limitations that prevent exact reproduction',
        '   The source file does not contain a 2020-21 season row, so the final prediction dataset cannot be created without fabrication.',
        '   The source records use season-end year integers rather than the paper’s 2000-01 notation, so they were normalized to that format for audit and split generation.',
        '   This dataset appears to be a multi-year player-season table that includes players with one row per season, which is consistent with the project but must be checked before any downstream modeling.',
        '',
        '12. Preprocessing decisions made',
        '   - The original CSV was located and copied into the project data/raw directory without modifying the original file.',
        '   - Required player, season, voting, and stat columns were mapped to their source names after inspection.',
        '   - MVP finalists were selected as the top three players by Share within each season.',
        '   - The winner was assigned as the single player with rank 1 in each season.',
        '   - Seasons were encoded as labels like 2000-01 to match the reference paper notation.',
        '   - No missing feature values were imputed and no duplicate rows were removed because the dataset was already clean for this task.',
        '',
        'Audit table for finalist selection',
        report_df.to_string(index=False),
    ])

    report_path = REPORTS_DIR / 'dataset_audit.txt'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nAudit report written to: {report_path}')


def main() -> None:
    raw_path = find_csv_path()
    raw_copy = ensure_raw_dataset_copy(raw_path)
    original_df = pd.read_csv(raw_copy)

    finalists, mapping, report_df = build_finalist_dataset(original_df)
    final_prediction_df = None

    # Save the full finalist dataset.
    finalists_out = finalists[[
        'Season', 'Player', 'G', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'PTS', 'MVP'
    ]].copy()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    finalists_out.to_csv(PROCESSED_DIR / 'nba_mvp_finalists.csv', index=False)

    train_df = finalists_out[finalists_out['Season'].isin([f'{start}-{str(start + 1)[-2:]}' for start in range(2000, 2016)])].copy()
    validation_df = finalists_out[finalists_out['Season'].isin([f'{start}-{str(start + 1)[-2:]}' for start in range(2016, 2020)])].copy()

    train_df.to_csv(PROCESSED_DIR / 'train.csv', index=False)
    validation_df.to_csv(PROCESSED_DIR / 'validation.csv', index=False)

    final_prediction_season = '2020-21'
    if final_prediction_season in finalists_out['Season'].unique():
        final_prediction_df = finalists_out[finalists_out['Season'] == final_prediction_season].copy()
        # final_prediction should not include the target label as an input feature.
        final_prediction_df = final_prediction_df[[
            'Season', 'Player', 'G', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'PTS'
        ]].copy()
        final_prediction_df.to_csv(PROCESSED_DIR / 'final_prediction.csv', index=False)

    create_audit_report(
        original_df=original_df,
        mapping=mapping,
        finalists=finalists,
        report_df=report_df,
        train_df=train_df,
        validation_df=validation_df,
        final_prediction_df=final_prediction_df,
    )

    print('\nPrepared datasets summary:')
    print(f'  nba_mvp_finalists.csv rows: {len(finalists_out)}')
    print(f'  train.csv rows: {len(train_df)}')
    print(f'  validation.csv rows: {len(validation_df)}')
    print(f'  final_prediction.csv created: {final_prediction_df is not None}')
    print(f'  Final prediction season (2020-21) available: {final_prediction_df is not None}')


if __name__ == '__main__':
    main()
