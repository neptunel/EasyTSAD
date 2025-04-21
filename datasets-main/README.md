# Original EasyTSAD datasets

We have formatted the raw data into a unified format to enable the model to conveniently read and process the data. All missing values are filled using linear interpolation.

All Datasets are available online. The ways to obtain the dataset are as follows：

- AIOPS: https://github.com/iopsai/iops
- WSD: https://github.com/alumik/AnoTransfer-data
- NAB: https://www.kaggle.com/datasets/boltzmannbrain/nab
- Yahoo: https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70
- UCR: https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/#ucr-time-series-anomaly-archive
- TODS: https://github.com/datamllab/tods/tree/master/datasets

NOTE: The original data generation process in TODS makes too naive anomalies. We modify the generation code to create more smoother, longer and more reasonable anomalies that are aligned with the description in their papers.

# Novel datasets

Novel datasets are the datasets that's not used in Tesseract training. They're available online:

- CalIt2: https://archive.ics.uci.edu/dataset/156/calit2+building+people+counts
- Norma: https://helios2.mi.parisdescartes.fr/~themisp/norma/
- GutenTAG: https://timeeval.github.io/evaluation-paper/notebooks/Datasets.html#GutenTAG

To see how datasets are processed, check the functions in process_rawdata.py.