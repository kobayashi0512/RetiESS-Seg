# Dataset provenance and local-data policy

This repository does not redistribute retinal images, OCT volumes, downloaded archives, extracted dataset folders, cached features, or trained checkpoints.

## Tian OCTRIMA

- Source publication: Tian et al., *PLoS ONE* (2015).
- Supporting data: <https://doi.org/10.1371/journal.pone.0133908.s002>
- Complete analysis unit: 10 volumes, 100 B-scans, eight boundaries, two observers.
- Split rule: leave one complete volume out; no slice-level leakage.

## AFIO paired OCT–fundus data

- Source publication: Raja et al., *Data in Brief* (2020).
- Dataset: <https://doi.org/10.17632/2rnnz5nz74.2>
- Complete audited analysis set: 49 unique OCT–fundus pairs from 26 patients.
- Split rule: all images from one patient remain in the same fold.

## Excluded local artifacts

The repository `.gitignore` excludes `data/`, `features/`, `checkpoints/`, `logs/`, virtual environments, temporary renders, and common archive/tensor formats. These files can be regenerated or reacquired from the cited public sources.
