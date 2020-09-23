# Patient Selection for Diabetes Drug Testing
I built a [model](https://github.com/iDataist/Patient-Selection-for-Diabetes-Drug-Testing/blob/master/src/patient_selection.ipynb) that predicts the expected days of hospitalization time, which determines whether the patient is selected for the clinical trial. In clinical trials, the drug is often administered over a few days in the hospital with frequent monitoring/testing. Therefore, the target patients are people that are likely to be in the hospital for this duration of time and will not incur significant additional costs for administering the drug and monitoring after discharge.  

## Dataset
I used a modified [dataset](https://github.com/iDataist/Patient-Selection-for-Diabetes-Drug-Testing/blob/master/src/data/final_project_dataset.csv) from [UC Irvine](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008).

## Dependencies
Using Anaconda consists of the following:

#### 1.Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.

**Download** the latest version of `miniconda` that matches your system.

|        | Linux | Mac | Windows |
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

#### 2.Create and activate a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work.

These instructions also assume you have `git` installed for working with Github from a terminal window, but if you do not, you can download that first with the command:
```
conda install git
```

#### 3.Create local environment

- Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/iDataist/Patient-Selection-for-Diabetes-Drug-Testing.git
cd Downloads
```

- Create (and activate) a new environment, named `ehr-env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__:
	```
	conda create -n ehr-env python=3.7
	source activate ehr-env
	```
	- __Windows__:
	```
	conda create --name ehr-env python=3.7
	activate ehr-env
	```

	At this point your command line should look something like: `(ehr-env) <User>:USER_DIR <user>$`. The `(ehr-env)` indicates that your environment has been activated, and you can proceed with further package installations.


- Install a few required pip packages, which are specified in the requirements text file. Be sure to run the command from the project root directory since the requirements.txt file is there.
```
pip install -r requirements.txt
ipython3 kernel install --name ehr-env --user

```
