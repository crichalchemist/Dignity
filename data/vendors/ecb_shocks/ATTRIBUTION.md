# ECB Monetary Policy Shocks - Data Attribution

## Source

This directory contains vendored data from the research paper:

**Jarociński, M., & Karadi, P. (2020).** *Deconstructing Monetary Policy Surprises—The Role of Information Shocks.* American Economic Journal: Macroeconomics, 12(2), 1-43.

**DOI:** [10.1257/mac.20180090](https://doi.org/10.1257/mac.20180090)

## Dataset Description

The dataset provides high-frequency identification of European Central Bank (ECB) monetary policy shocks and central bank information shocks. The shocks are extracted from changes in interest rates and stock prices around ECB Governing Council meetings using a sign restriction approach.

### Files Included

- **shocks_ecb_mpd_me_d.csv**: Daily frequency shocks (ECB meeting dates)
- **shocks_ecb_mpd_me_m.csv**: Monthly frequency shocks (aggregated)

### Key Variables

- **MP_median**: Monetary policy shock (median estimate)
- **CBI_median**: Central bank information shock (median estimate)
- **MP_mean**: Monetary policy shock (mean estimate)
- **CBI_mean**: Central bank information shock (mean estimate)
- **pc1**: First principal component of interest rate changes
- **STOXX50**: Euro STOXX 50 index change

## Data Access

The original data and replication files can be obtained from:
- [American Economic Association Data Repository](https://www.openicpsr.org/openicpsr/project/116329)
- [ECB Working Paper Series](https://www.ecb.europa.eu/pub/research/working-papers/html/index.en.html)

## Usage in Sequence Project

This data is used in `data/downloaders/ecb_shocks_downloader.py` to:
1. Load monetary policy shock data for EUR currency pairs
2. Classify shocks as hawkish, dovish, or neutral based on thresholds
3. Merge with high-frequency FX price data for fundamental analysis

## Citation Requirements

If you use this data in academic research, please cite:

```bibtex
@article{jarocinski2020deconstructing,
  title={Deconstructing Monetary Policy Surprises—The Role of Information Shocks},
  author={Jaroci{\'n}ski, Marek and Karadi, Peter},
  journal={American Economic Journal: Macroeconomics},
  volume={12},
  number={2},
  pages={1--43},
  year={2020},
  publisher={American Economic Association},
  doi={10.1257/mac.20180090}
}
```

## License

The original dataset is provided under the [AEA Data Availability Policy](https://www.aeaweb.org/journals/policies/data-code/). 

**Permitted Uses:**
- Academic replication and research purposes only. The dataset must be cited properly as shown in the "Citation Requirements" section above.

**Restrictions:**
- Non-academic and commercial use requires explicit written permission from the authors and compliance with the original AEA policy terms.

**Permission & Licensing:**
For requests regarding non-academic use, derivative works, or commercial licensing, please contact the authors (see Contact section) or visit the [AEA Data Repository](https://www.openicpsr.org/openicpsr/project/116329) for licensing options.

## Version

Dataset version: 2020 update (covers ECB meetings through 2019)

## Contact

For questions about the original dataset, please contact the authors:
- Marek Jarociński (European Central Bank)
- Peter Karadi (European Central Bank, CEPR)
