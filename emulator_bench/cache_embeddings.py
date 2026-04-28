import argparse
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    AFFINITY_VALUE_TYPES,
    DEFAULT_BASE_ROOT,
    DEFAULT_SPLIT_GROUPS,
    normalize_threshold_args,
    read_table,
    require_columns,
    resolve_base_dir,
    save_json,
    write_table,
)
from emulator_bench.feature_pipeline import (
    BACPIFeaturizer,
    compound_cache_key,
    compound_cache_path,
    filter_invalid_smiles_rows,
    protein_cache_key,
    protein_cache_path,
    save_npz_atomic,
)


def normalize_value_types(value_types):
    if value_types is None:
        return ["ki"]
    if isinstance(value_types, str):
        raw_values = [value_types]
    else:
        raw_values = list(value_types)
    normalized = []
    seen = set()
    for value in raw_values:
        key = str(value).strip().lower()
        if not key:
            continue
        if key not in AFFINITY_VALUE_TYPES:
            raise ValueError("Unsupported value_type `%s`. Expected one of %s." % (value, ", ".join(AFFINITY_VALUE_TYPES)))
        if key not in seen:
            seen.add(key)
            normalized.append(key)
    if not normalized:
        raise ValueError("At least one value type is required.")
    return normalized


def resolve_base_root_path(base_root: str = None) -> Path:
    return Path(base_root or DEFAULT_BASE_ROOT).expanduser().resolve()


def resolve_base_dirs(args):
    if args.base_dir:
        root_candidate = Path(args.base_dir).expanduser().resolve()
        if len(args.value_type) == 1:
            value_type = args.value_type[0]
            child = root_candidate / value_type
            if child.exists():
                return [child], root_candidate
            return [root_candidate], root_candidate.parent if root_candidate.name == value_type else root_candidate

        resolved = []
        for value_type in args.value_type:
            child = root_candidate / value_type
            if not child.exists():
                raise ValueError(
                    "When passing multiple --value_type values, --base_dir must point to the BACPI root "
                    "containing per-value subdirectories."
                )
            resolved.append(child)
        return resolved, root_candidate

    base_root = resolve_base_root_path(args.base_root)
    return [resolve_base_dir(None, str(base_root), value_type) for value_type in args.value_type], base_root


def resolve_embeddings_dir(args, cache_root: Path, base_dirs) -> Path:
    if args.embeddings_dir:
        return Path(args.embeddings_dir).expanduser().resolve()
    if len(args.value_type) > 1:
        return cache_root / "embeddings_shared"
    return Path(base_dirs[0]) / "embeddings"


def collect_unique_values(jobs, sequence_col: str, smiles_col: str, clean_splits_in_place: bool, skip_smiles_validity_check: bool):
    sequences = set()
    smiles_values = set()
    drop_stats = []
    for job in jobs:
        for split_key in ("train_path", "val_path", "test_path"):
            split_path = Path(job[split_key])
            frame = read_table(split_path)
            require_columns(frame, [sequence_col, smiles_col], split_path)
            if skip_smiles_validity_check:
                filtered_frame = frame.reset_index(drop=True)
                stats = {
                    "source": str(split_path),
                    "rows_in": int(len(frame)),
                    "rows_out": int(len(frame)),
                    "rows_dropped": 0,
                    "invalid_examples": [],
                }
            else:
                filtered_frame, stats = filter_invalid_smiles_rows(frame, smiles_col=smiles_col, source_name=str(split_path))
                if stats["rows_dropped"] > 0:
                    print(
                        "Dropped %s rows with invalid SMILES from %s"
                        % (stats["rows_dropped"], split_path),
                        flush=True,
                    )
                    if clean_splits_in_place:
                        write_table(split_path, filtered_frame)
                        print("Overwrote cleaned split file %s" % split_path, flush=True)
            drop_stats.append(stats)
            sequences.update(filtered_frame[sequence_col].astype(str).tolist())
            smiles_values.update(filtered_frame[smiles_col].astype(str).tolist())
    return sorted(smiles_values), sorted(sequences), drop_stats


def save_featurizer(featurizer: BACPIFeaturizer, embeddings_dir: Path):
    featurizer_path = embeddings_dir / "vocabs" / "featurizer.pkl"
    featurizer.save(featurizer_path)
    save_json(
        embeddings_dir / "vocabs" / "summary.json",
        {
            "radius": featurizer.radius,
            "ngram": featurizer.ngram,
            "fp_radius": featurizer.fp_radius,
            "nbits": featurizer.nbits,
            "n_atom_tokens": featurizer.n_atom_tokens,
            "n_amino_tokens": featurizer.n_amino_tokens,
        },
    )
    return featurizer_path


def featurizer_summary(featurizer: BACPIFeaturizer):
    return {
        "n_atom_tokens": featurizer.n_atom_tokens,
        "n_amino_tokens": featurizer.n_amino_tokens,
        "n_atoms": len(featurizer.atom_dict),
        "n_bonds": len(featurizer.bond_dict),
        "n_fingerprints": len(featurizer.fingerprint_dict),
        "n_edges": len(featurizer.edge_dict),
        "n_words": len(featurizer.word_dict),
    }


def maybe_build_featurizer(args, embeddings_dir: Path, smiles_values, sequences):
    featurizer_path = embeddings_dir / "vocabs" / "featurizer.pkl"
    if featurizer_path.exists() and not args.overwrite_vocabs:
        featurizer = BACPIFeaturizer.load(featurizer_path)
        if (
            featurizer.radius != args.radius
            or featurizer.ngram != args.ngram
            or featurizer.fp_radius != args.fp_radius
            or featurizer.nbits != args.nbits
        ):
            raise ValueError(
                "Existing featurizer at %s does not match requested settings. "
                "Use --overwrite_vocabs or a different --embeddings_dir." % featurizer_path
            )
        return featurizer, featurizer_path, False

    featurizer = BACPIFeaturizer(
        radius=args.radius,
        ngram=args.ngram,
        fp_radius=args.fp_radius,
        nbits=args.nbits,
    )
    print("Building BACPI vocabularies from %s unique compounds and %s unique proteins..." % (len(smiles_values), len(sequences)), flush=True)
    featurizer.build_from_values(smiles_values, sequences)
    featurizer_path = save_featurizer(featurizer, embeddings_dir)
    return featurizer, featurizer_path, True


def index_existing_cache_keys(cache_root: Path, label: str):
    if not cache_root.exists():
        return set()
    existing_keys = set()
    iterator = tqdm(cache_root.rglob("*.npz"), desc="Indexing existing %s caches" % label, unit="file")
    for path in iterator:
        existing_keys.add(path.stem)
    return existing_keys


def cache_compounds(
    featurizer: BACPIFeaturizer,
    embeddings_dir: Path,
    smiles_values,
    overwrite: bool,
):
    if overwrite:
        print("Overwrite enabled; all compound caches will be rebuilt.", flush=True)
        pending_smiles = list(smiles_values)
        print("Compound caches present: 0 | to write: %s" % len(pending_smiles), flush=True)
        written = 0
        for smiles in tqdm(pending_smiles, desc="Caching compounds", unit="compound"):
            payload = featurizer.encode_compound(smiles, build=True)
            save_npz_atomic(compound_cache_path(embeddings_dir, smiles), payload)
            written += 1
        return written

    existing_keys = index_existing_cache_keys(Path(embeddings_dir) / "compounds", "compound")
    pending_smiles = []
    iterator = tqdm(smiles_values, desc="Resolving compound cache keys", unit="compound")
    for smiles in iterator:
        cache_key = compound_cache_key(smiles)
        if cache_key not in existing_keys:
            pending_smiles.append(smiles)
    print(
        "Compound caches present: %s | to write: %s"
        % (len(smiles_values) - len(pending_smiles), len(pending_smiles)),
        flush=True,
    )
    written = 0
    for smiles in tqdm(pending_smiles, desc="Caching compounds", unit="compound"):
        payload = featurizer.encode_compound(smiles, build=True)
        save_npz_atomic(compound_cache_path(embeddings_dir, smiles), payload)
        written += 1
    return written


def cache_proteins(featurizer: BACPIFeaturizer, embeddings_dir: Path, sequences, overwrite: bool):
    if overwrite:
        print("Overwrite enabled; all protein caches will be rebuilt.", flush=True)
        pending_sequences = list(sequences)
        print("Protein caches present: 0 | to write: %s" % len(pending_sequences), flush=True)
        written = 0
        for sequence in tqdm(pending_sequences, desc="Caching proteins", unit="protein"):
            payload = featurizer.encode_protein(sequence, build=True)
            save_npz_atomic(protein_cache_path(embeddings_dir, sequence), payload)
            written += 1
        return written

    existing_keys = index_existing_cache_keys(Path(embeddings_dir) / "proteins", "protein")
    pending_sequences = []
    iterator = tqdm(sequences, desc="Resolving protein cache keys", unit="protein")
    for sequence in iterator:
        cache_key = protein_cache_key(sequence)
        if cache_key not in existing_keys:
            pending_sequences.append(sequence)
    print(
        "Protein caches present: %s | to write: %s"
        % (len(sequences) - len(pending_sequences), len(pending_sequences)),
        flush=True,
    )
    written = 0
    for sequence in tqdm(pending_sequences, desc="Caching proteins", unit="protein"):
        payload = featurizer.encode_protein(sequence, build=True)
        save_npz_atomic(protein_cache_path(embeddings_dir, sequence), payload)
        written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description="Cache BACPI compound and protein features once for explicit EMULaToR split jobs.")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--base_root", type=str, default=None)
    parser.add_argument("--value_type", nargs="+", default=["ki"])
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_vocabs", action="store_true")
    parser.add_argument("--clean_splits_in_place", dest="clean_splits_in_place", action="store_true")
    parser.add_argument("--no_clean_splits_in_place", dest="clean_splits_in_place", action="store_false")
    parser.add_argument("--skip_smiles_validity_check", action="store_true")
    parser.set_defaults(clean_splits_in_place=True)
    args = parser.parse_args()

    started = time.time()
    args.value_type = normalize_value_types(args.value_type)
    base_dirs, cache_root = resolve_base_dirs(args)
    embeddings_dir = resolve_embeddings_dir(args, cache_root, base_dirs)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)

    from emulator_bench.common import discover_split_jobs

    jobs = []
    for base_dir in base_dirs:
        jobs.extend(discover_split_jobs(base_dir, split_groups=args.split_groups, thresholds=args.thresholds))
    if not jobs:
        base_dirs_str = ", ".join(str(path) for path in base_dirs)
        raise FileNotFoundError("No split jobs discovered in %s" % base_dirs_str)

    smiles_values, sequences, drop_stats = collect_unique_values(
        jobs,
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
        clean_splits_in_place=args.clean_splits_in_place,
        skip_smiles_validity_check=args.skip_smiles_validity_check,
    )
    print("Discovered %s split jobs" % len(jobs), flush=True)
    print("Unique compounds: %s" % len(smiles_values), flush=True)
    print("Unique proteins: %s" % len(sequences), flush=True)

    featurizer, featurizer_path, vocab_rebuilt = maybe_build_featurizer(args, embeddings_dir, smiles_values, sequences)
    vocab_before_cache = featurizer_summary(featurizer)
    compound_written = cache_compounds(
        featurizer,
        embeddings_dir,
        smiles_values,
        overwrite=args.overwrite,
    )
    protein_written = cache_proteins(featurizer, embeddings_dir, sequences, overwrite=args.overwrite)
    vocab_after_cache = featurizer_summary(featurizer)
    vocab_extended = vocab_after_cache != vocab_before_cache
    if vocab_extended:
        featurizer_path = save_featurizer(featurizer, embeddings_dir)
        print("Extended BACPI vocabularies while caching new features; saved %s" % featurizer_path, flush=True)

    manifest = {
        "cache_version": 1,
        "base_dirs": [str(path) for path in base_dirs],
        "embeddings_dir": str(embeddings_dir),
        "value_types": list(args.value_type),
        "shared_cache": len(args.value_type) > 1,
        "split_groups": list(args.split_groups),
        "thresholds": args.thresholds,
        "sequence_col": args.sequence_col,
        "smiles_col": args.smiles_col,
        "radius": args.radius,
        "ngram": args.ngram,
        "fp_radius": args.fp_radius,
        "nbits": args.nbits,
        "featurizer_path": str(featurizer_path),
        "n_atom_tokens": featurizer.n_atom_tokens,
        "n_amino_tokens": featurizer.n_amino_tokens,
        "split_jobs": len(jobs),
        "unique_compounds": len(smiles_values),
        "unique_proteins": len(sequences),
        "invalid_smiles_rows_dropped": int(sum(item["rows_dropped"] for item in drop_stats)),
        "clean_splits_in_place": bool(args.clean_splits_in_place),
        "skip_smiles_validity_check": bool(args.skip_smiles_validity_check),
        "compound_written": compound_written,
        "protein_written": protein_written,
        "vocab_rebuilt": vocab_rebuilt,
        "vocab_extended": vocab_extended,
        "vocab_before_cache": vocab_before_cache,
        "vocab_after_cache": vocab_after_cache,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    save_json(embeddings_dir / "manifest.json", manifest)
    print("Saved cache manifest to %s" % (embeddings_dir / "manifest.json"), flush=True)


if __name__ == "__main__":
    main()
