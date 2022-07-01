For CS-CMNIST dataset only!

Our code is based on this repo at   https://github.com/ahujak/IB-IRM. 

First, you need to install some package for satisfying all the requirements below:

1. torch 1.6.0

2. torchvision 0.7.0

3. numpy 1.19.1

4. tqdm 4.41.1

Second, please take a look at the tested algorithms in "algorithm.py". The algorithms are denoted from 1, 2, ..., up to 8 in order of ERM, IRM, IB_ERM, IB_IRM, CEM_CORAL, CEM_MMD, IRM_MMD, and CEM. We have different names for ERM, IRM, IB_ERM algorithms so these algorithms can run together without conflict.  However, to keep the same number of hyper-parameters and utilize the existing source code, all algorithms IB_IRM, CEM_CORAL, CEM_MMD, IRM_MMD, and CEM are under the same name of "IBIRM" - because they are very similar, just having a bit of change at the loss function. Please select only one algorithm from five algorithms IB_IRM, CEM_CORAL, CEM_MMD, IRM_MMD, and CEM to run at a time and comment out other ones. In the default code, we are using the CEM_MMD algorithm and commented out IB_IRM, CEM_CORAL, IRM_MMD, and CEM.

Third, please run "sweep_train.py" to get the result. "sweep_train.py" scans over the algorithms and the hyper-parameters. The setting of hyper-parameters can be found at the end of "sweep_train.py". Now, we just commented out the sweeping of ERM, IRM, and IB_ERM while keeping the same sweeping function for IB_IRM, CEM_CORAL, CEM_MMD, IRM_MMD, and CEM. Please select the algorithm you want to sweep and comment out other ones. 

In addition, there are some parameters that can be selected in "sweep_train.py". For example, --test_val to control the tuning procedures, --env_seed to select the oracle model, --holdout_fraction to control the validation set. For now, we use the default values of 5\% data for validation and use the train-validation tuning procedure. You may not need to adjust these parameters. 

The checkpoints are not stored, and the final results will be printed after the whole run. Our code is based on this repo at   https://github.com/ahujak/IB-IRM.   We promise to upload the source code to Github after the end of the review process.

Due to the limited time, the code is not prepared very clearly. We will fix the code, and upload a new version after the end of reviewing process. The new version will not require any manual settings like this version.

Thank you for your consideration!
Authors,



