For CMNIST dataset only!

This repo is based on https://github.com/facebookresearch/DomainBed


First, you need to install some package for satisfying the below requirements:
1. numpy==1.20.3
2. wilds==1.2.2
3. imageio==2.9.0
4. gdown==3.13.0
5. torchvision==0.8.2
6. torch==1.7.1
7. tqdm==4.62.2
8. backpack==0.1
9. parameterized==0.8.1
10. Pillow==8.3.2

1. Download the datasets:

python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data

->You may not need to do this step, since we already uploaded the MNIST dataset into this folder.      
       

       
2. Launch a sweep:

python -m sweep --command launch\
       --data_dir=./domainbed/data/path\
       --output_dir=./domainbed/outputThuan/path\
       --command_launcher local\
       --algorithms IB_IRM\
       --datasets ColoredMNIST\
       --n_hparams 10\
       --n_trials 2
       
-> n_trials: number of trials i.e, number of random seeds, for each trial, the algorithm will scan over all the hyper-parameters. --n_hparams: number of hyper-parameter pairs which is randomly picked in a range of (0.1, 10000). --datasets: default by ColoredMNIST, --algorithms: please select one algorithm.  
.       
3. After all jobs have either succeeded or failed, you can delete the data from failed jobs with: 

python -m sweep --command delete_incomplete\
       --data_dir=./domainbed/data/path\
       --output_dir=./domainbed/outputThuan/path\
       --command_launcher local\
       --algorithms IB_IRM\
       --datasets ColoredMNIST\
       --n_hparams 10\
       --n_trials 2
       
 and then re-launch them by running step 2 again. 

4. To view the results of your sweep:
python -m domainbed.scripts.collect_results\
       --input_dir=/my/sweep/output/path

Note: please take a look at the tested algorithms in "algorithms.py". The algorithms are denoted from 1, 2, ..., up to 8 in order of ERM, IRM, IB_ERM, IB_IRM, CEM, IRM_MMD, CEM_MMD, and CEM_CORAL. We have different names for ERM, IRM, IB_ERM algorithms so these algorithms can run together without conflict.  However, to keep the same number of hyper-parameters and utilize the existing source code, all algorithms  IB_IRM, CEM, IRM_MMD, CEM_MMD, and CEM_CORAL are under the same name of "IB_IRM" - because they are very similar, just having a bit of change at the loss function. Please select only one algorithm from five algorithms  IB_IRM, CEM, IRM_MMD, CEM_MMD, and CEM_CORAL to run at a time and comment out other ones. In the default code, we are using the CEM algorithm and commented out  IB_IRM, IRM_MMD, CEM_MMD, and CEM_CORAL.


Due to the limited time, the code is not prepared very clearly. We will fix the code, and upload a new version after the end of reviewing process. The new version will not require any manual settings like this version.

Thank you for your consideration!
Authors,



