# LiveKT

This code relies on a fork of [pyKT.org](https://pykt.org) repository by Liu et al. 2022. (Different project from different authors.)

`live_kt.py` contains our implementation of LiveKT, run by `split.py`.

`preprocess_from_livekt.py` converts this data to pyKT format.

`evaluate_livekt.ipynb` prepares datasets and runs LiveKT (for the `test_livekt` function) or AKT or DKT (for the`test_traditional_kt` function), see an example in `main.ipynb`.
