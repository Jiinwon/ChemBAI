## ToxCast_model_v.3

* Pre-split datasets (``train_df.csv``, ``val_df.csv``, ``test_df.csv``) are
  expected under ``experiments/{training|prediction}/{project}``.
* Each split directory must contain a ``fingerprints`` folder with the
  fingerprint matrices for MACCS, Morgan, RDKit, Pattern and Layered.
* Training scripts skip the internal train/test split and instead consume the
  provided splits.  Fingerprint generation is assumed to have been completed in
  advance.
* The ``run_v.3`` trainers mirror the behaviour of the original pipeline: every
  fingerprint/model combination is trained, cross-validated and evaluated on the
  supplied validation/test sets.