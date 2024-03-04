# Federated Learning of Socially Appropriate Agent Behaviours in Simulated Home Environments

## Overview

This is a PyTorch-based code implementation for the paper titled [Federated Learning of Socially Appropriate Agent Behaviors in Simulated Home Environments](#) paper published at the HRI '24: Workshop on Lifelong Learning and Personalization in Long-Term Human-Robot Interaction (LEAP-HRI), ACM/IEEE International Conference on Human-Robot Interaction (HRI) 2024. 
The paper explores the use of Federated Learning (FL) and Federated Continual Learning (FCL) techniques to train agents in simulated home environments, focusing on socially appropriate behaviors.


## Installation

Ensure you have the necessary dependencies installed. Run the following command to set up the environment:

```bash
pip install -r requirements.txt
```

## Dataset

Access to the Manners-DB data files can be requested here: https://github.com/jonastjoms/MANNERS-DB/tree/master
The csv file with the labels, once acquired, should be placed under ```Data/all_data.csv```. Currently, a dummy file is included for reference.

## Training

The repository is divided into two parts
- Federated Learning
- Federated Continual Learning

### Federated Learning
Each Strategy is setup in a seperate Jupyter Notebook (FedAvg, FedBN, FedProx, FedOptAdam, FedDistil) both without and with data augmentation.

### Federated Continual Learning
FCL strategies are implementad as python packages. To execute the code for the benchmark on MANNERS-DB run the following:

```bash
bash Federated Continual Learning/run_FCL_local.sh
```

## Results

Results are included in the [Results](./Results) as two different sections, Federated Learning and Federated Continual Learning. 


## Citation

```
@INPROCEEDINGS{Checker2024Federated,  
  author		= {S. {Checker} and N. {Churamani} and H. {Gunes}},  
  booktitle		= {{Workshop on Lifelong Learning and Personalization in Long-Term Human-Robot Interaction (LEAP-HRI), 16th ACM/IEEE International Conference on Human-Robot Interaction (HRI)}},
  title			= {{Federated Learning of Socially Appropriate Agent Behaviors in Simulated Home Environments}},   
  year			= {2024},  
 }
```

## Acknowledgement
**Funding:** [S. Checker](https://www.sakshamchecker.com) contributed to this work while undertaking a remote visiting studentship at the Department of Computer Science and 
Technology, University of Cambridge. [N. Churamani](https://nchuramani.github.io) and [H. Gunes](https://www.cl.cam.ac.uk/~hg410/) are supported by Google under the GIG Funding 
Scheme. 

**Open Access:** For open access purposes, the authors have applied a Creative Commons Attribution (CC BY) licence to any Author Accepted Manuscript version arising.

**Data Access Statement:** This study involves secondary analyses of the existing datasets, that are described and cited in the text. 
