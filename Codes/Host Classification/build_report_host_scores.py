"""Build phage scores using the report's exact fold-match restriction.

Only load trusted pickle files. This enforces filtering, not model-training provenance.
Run with --help for options. Requires pandas, numpy, and pyarrow for Parquet output.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


def aggregate_report_scores(metadata, splits, batches):
    """batches yields (held_out_fold, source_name, protein-score DataFrame).

    Keep a score iff feature-specific fold == metadata fold == model folder fold.
    Unknown assignments and nonfinite scores never contribute. Fail on duplicate
    eligible protein/feature predictions rather than silently weighting them twice.
    """
    required = {"accession", "split"}
    if not required.issubset(metadata.columns):
        raise ValueError("Metadata requires accession and split columns")
    if not metadata.index.is_unique or not splits.index.is_unique:
        raise ValueError("Protein IDs must be unique in metadata and split table")
    if metadata.accession.isna().any() or not metadata['split'].isin(range(5)).all():
        raise ValueError("Missing accession or invalid metadata fold")
    if metadata.groupby('accession')['split'].nunique().gt(1).any():
        raise ValueError("A phage appears in more than one metadata fold")
    if not splits.columns.is_unique:
        raise ValueError("Duplicate split columns")
    totals, counts, seen = {}, {}, {}
    audit = {"rule": "feature_fold == metadata_fold == prediction_folder_fold",
             "training_provenance_verified_by_this_script": False,
             "sources": [], "eligible_scores": 0, "retained_scores": 0}
    for fold, source, preds in batches:
        if fold not in range(5):
            raise ValueError(f"Invalid prediction fold: {fold}")
        if not preds.index.is_unique or not preds.columns.is_unique:
            raise ValueError(f"Duplicate prediction IDs or columns: {source}")
        if not preds.index.isin(metadata.index).all():
            raise ValueError(f"Prediction IDs absent from metadata: {source}")
        ids = preds.index
        meta = metadata.loc[ids]
        retained = 0
        for col in preds.columns:
            key = 'split_' + col
            if key not in splits.columns:
                raise ValueError(f"Missing split mapping for {col}")
            mapping = splits[key].reindex(ids)
            if not mapping.dropna().isin(range(5)).all():
                raise ValueError(f"Invalid fold assignments for {col}")
            eligible = (mapping.eq(fold) & meta['split'].eq(fold)).fillna(False)
            chosen = ids[eligible.to_numpy(dtype=bool)]
            prior = seen.setdefault(col, set())
            if prior.intersection(chosen):
                raise ValueError(f"Repeated eligible protein/feature scores: {source}, {col}")
            prior.update(chosen)
            values = pd.to_numeric(preds.loc[chosen, col], errors='raise').astype(float)
            if np.isinf(values).any() or ((values.dropna() < 0) | (values.dropna() > 1)).any():
                raise ValueError(f"Scores must be probabilities or NaN: {source}, {col}")
            values = values.dropna()
            accession = metadata.loc[values.index, 'accession']
            sums = values.groupby(accession).sum()
            nums = values.groupby(accession).count()
            totals[col] = totals.get(col, pd.Series(dtype=float)).add(sums, fill_value=0)
            counts[col] = counts.get(col, pd.Series(dtype=float)).add(nums, fill_value=0)
            audit['eligible_scores'] += len(chosen)
            retained += len(values)
        audit['retained_scores'] += retained
        audit['sources'].append({"fold": fold, "file": str(source), "retained": retained})
    if not totals:
        raise ValueError("No score columns found")
    accessions = pd.Index(metadata.accession.unique(), name='accession')
    count_df = pd.DataFrame(counts).reindex(accessions).fillna(0).astype('int64')
    scores = pd.DataFrame(totals).reindex(accessions).div(count_df.where(count_df > 0))
    audit.update(phages=len(scores), features=len(scores.columns),
                 entirely_missing_phages=int(scores.isna().all(axis=1).sum()),
                 missing_fraction=float(scores.isna().to_numpy().mean()))
    if not audit['retained_scores']:
        raise ValueError("No scores passed the report restriction")
    return scores, count_df, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'Data')
    parser.add_argument('--output', type=Path, help='New .parquet or .csv file; never overwritten')
    parser.add_argument('--confirm-held-out-folds', action='store_true',
                        help='Assert you verified folder s_k was trained excluding cluster fold k')
    args = parser.parse_args()
    root = args.data_dir.resolve()
    output = args.output or root / 'phage_host_scores_report_filtered.parquet'
    count_path = output.with_name(output.stem + '_counts' + output.suffix)
    audit_path = output.with_suffix('.audit.json')
    if output.suffix not in {'.parquet', '.csv'}:
        parser.error('Output must end in .parquet or .csv')
    for path in (output, count_path, audit_path):
        if path.exists():
            parser.error(f'Refusing to overwrite {path}')
    folders = [root / f'random_pc_r_5_da_2_bal_s_{k}' for k in range(5)]
    inventories = [set(p.name for p in folder.glob('all_preds_*.pkl')) for folder in folders]
    if any(not names for names in inventories):
        parser.error('All five raw prediction folders are required and must contain all_preds_*.pkl')
    if not args.confirm_held_out_folds:
        parser.error('Verify the training protocol first, then supply --confirm-held-out-folds')
    metadata = pd.read_csv(root / 'metadata.csv', index_col='proteinID',
                           usecols=['proteinID', 'accession', 'split'])
    print('Loading split table (large file; substantial RAM required)...', flush=True)
    splits = pd.read_pickle(root / 'all_random_pc_splits.pkl')

    def batches():
        for fold, names in enumerate(inventories):
            for name in sorted(names):
                path = folders[fold] / name
                print(f'Reading {path}', flush=True)
                yield fold, str(path), pd.read_pickle(path)

    scores, counts, audit = aggregate_report_scores(metadata, splits, batches())
    union = set.union(*inventories)
    audit['missing_files_by_fold'] = {str(k): sorted(union - names) for k, names in enumerate(inventories)}
    audit['held_out_training_protocol_asserted_by_caller'] = True
    audit['note'] = 'Missing files remain missing; this does not certify phage-level independence or conformal coverage.'
    output.parent.mkdir(parents=True, exist_ok=True)
    for frame, path in ((scores, output), (counts, count_path)):
        if path.suffix == '.csv':
            frame.to_csv(path)
        else:
            frame.to_parquet(path)
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(f'Saved {scores.shape} scores to {output}; counts and audit saved alongside.')
    print('Missing files by fold:', audit['missing_files_by_fold'])


if __name__ == '__main__':
    main()
