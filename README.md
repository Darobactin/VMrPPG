# VMRrPPG

Official implementation of the paper **Mask Attack Detection Using Vascular-weighted Motion-robust rPPG Signals**.

- ``rppg_extract.py`` is used to extract rPPG signals.
- ``rppg_processing.py`` is used to split the signal sequences into segments at the specified length.
- run ``main.py`` to train and evaluate the ENetGRU model.
- ``sample model`` provides a sample of training and testing on 3DMAD, including a checkpoint file, a scoring file, and an analytical log. The meaning of each column is available on ``main.py``. 
