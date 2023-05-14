# VMRrPPG

Official implementation of the paper **Mask Attack Detection Using Vascular-weighted Motion-robust rPPG Signals**.

### Implementation Instructions
- STEP 1. Use ``rppg_extract.py`` to extract rPPG signals. ``SeetaFace6`` package is contributed by @tensorflower at https://github.com/tensorflower/seetaFace6Python/tree/master
- STEP 2. Use ``rppg_processing.py`` to split the signal sequences into segments at the specified length.
- STEP 3. Use ``rppg_reweighting_bio1.py`` to add weights to rPPG signals from different ROIs.
- STEP 4. Run ``main.py`` to train and evaluate the ENetGRU model with scoring results as outputs.

### Additional Information
- ``sample model`` provides a sample of training and testing process on 3DMAD, including a checkpoint file, a scoring file, and an analytical log. The meaning of each column is available on ``main.py``.
- ``protocols`` provides a specified train-test split for each protocol on each dataset.
