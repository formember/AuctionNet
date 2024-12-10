# AuctionNet: A Novel Benchmark for Decision-Making in Uncertain and Competitive Games
![Static Badge](https://img.shields.io/badge/license-Apache_License_2.0-red)   &nbsp;&nbsp;&nbsp;&nbsp;  ![Static Badge](https://img.shields.io/badge/version-1.0-green)&nbsp;&nbsp;&nbsp;&nbsp;  ![Static Badge](https://img.shields.io/badge/Filed-Decison_makeing-blue)&nbsp;&nbsp;&nbsp;&nbsp;  ![Static Badge](https://img.shields.io/badge/Organization-Alimama-purple)


---


[//]: # (<p align="center">)

[//]: # (<a href="https://openreview.net/forum?id=OTjTKFk7gb#discussion">Paper</a>)

[//]: # (</p>)

Decision-making in large-scale games is an essential research area in artificial intelligence (AI) with significant real-world impact. **AuctionNet** is a benchmark for bid decision-making in large-scale ad auctions derived from a real-world online advertising platform. AuctionNet is composed of three parts: 

- 🌏️ **Ad Auction Environment**：the environment effectively replicates the integrity and complexity of real-world ad auctions with the interaction of several modules: the ad opportunity generation module，the bidding module and the auction module. 

- 🔢 **Pre Generated Dataset**: we pre-generated a substantial dataset based on the auction environment. The dataset contains trajectories with 48 diverse agents competing with each other, totaling over 500 million records and 80GB in size.

- ✴️ **several baseline bid decision-making algorithms**:We implemented a variety of baseline algorithms such as linear programming, reinforcement learning, and generative models.

We note that AuctionNet is applicable not only to research on bid decision-making algorithms in ad auctions but also to the general area of **decision-making** in large-scale games. It can also benefit researchers in a broader range of areas such as reinforcement learning, generative models, operational research, and mechanism design.

## 🔥 News

---

- [**2024-12-16**] 🔥 The **AuctionNet-1.0** code has been officially open-sourced. We welcome everyone to give it a thumbs up and share valuable feedback.
- [**2024-10-24**] 💫  [NeurIPS 2024 Competition:Auto-Bidding in Uncertain Environment](https://tianchi.aliyun.com/competition/entrance/532226) has officially ended.The competition attracted more than **1,500 teams** to participate. The **evaluation environment and baseline** used in the competition are derived from this project.
- [**2024-09-26**] 🎁 Our paper [AuctionNet](https://openreview.net/forum?id=OTjTKFk7gb#discussion), has been accepted by **NeurIPS 2024 Datasets and Benchmark Track**!


## 🥁 Background
<p align="center">
    <br>
    <img src="assets/background.png" width="700"/>
    <br>
<p>
Bid decision-making in large-scale ad auctions is a concrete example of decision-making in large-scale games.
Numbers 1 through 5 illustrate how an auto-bidding agent helps advertiser optimize performance.
For each advertiser's unique objective (I), auto-bidding agent make bid decision-making (II) for continuously arriving ad opportunities, and compete against each other in the ad auction (III). 
Then, each agent may win some impressions (IV), which may be exposed to users and potentially result in conversions. Finally, the agents' performance  (V) will be reported to advertisers.


## 🏛︎ Project Structure

---

```
├── config                        # Configuration files for setting up the Hyperparameters.
├── main_test.py                  # The main entry point for running the evaluation.
├── run                           # The core logic for executing tests.

├── simul_bidding_env             # Ad Auction Environment

│   ├── Controller                # Module for controlling the simulation flow and logic.
│   ├── Environment               # the auction module.
│   ├── PvGenerator               # the ad opportunity generation module.
│   ├── Tracker                   # Tracking components for monitoring and analysis.
│   │   ├── BiddingTracker.py     # Tracks bidding process and outcomes the Traffic granularity raw data.
│   │   ├── PlayerAnalysis.py     # Analyzes player's strategy behavior and performance.
│   └── strategy                  # the bidding module(official competitive agent).


├── pre_generated_dataset         # pre_generated_dataset


├── strategy_train_env            # several baseline bid decision-making algorithms

│   ├── README_strategy_train.md  # Documentation for how to train the bidding strategy.
│   ├── bidding_train_env         # Core components for training bidding strategies.
│   ├   ├── baseline              # Baseline models and strategies implementation.
│   ├   ├── common                # Common utilities used across modules.
│   ├   ├── train_data_generator  # Reads raw data and constructs training data for model. training.
│   ├   ├── offline_eval          # The components needed for offline evaluation.
│   ├   └── strategy              # Strategy prediction code for training models.
│   ├── data                      # Directory for storing training data.
│   ├── main                      # Main scripts for executing training processes.
│   ├── run                       # Core logic for executing training processes.
│   ├── saved_model               # Directory for saving trained models.

```



## 🛠️ Getting Started

---

### 1. Create and activate conda environment
```bash
$ conda create -n AuctionNet python=3.9.12 pip=23.0.1
$ conda activate AuctionNet
```
### 2. Install requirements
```bash
$ pip install -r requirements.txt
```

## 🧑‍💻Quickstart

---

### Train  Strategy & Offline Evaluation
*For detailed usage, please refer to strategy_train_env/README_strategy_train.md.*
```
cd strategy_train_env  # Enter the strategy_train directory
```
#### Data Processing
Run this script to convert the traffic granularity data into trajectory data required for model training.
```
python  bidding_train_env/train_data_generator/train_data_generator.py
```
#### strategy training
Load the training data and train the xxx(for example IQL) bidding strategy.
```
python main/main_iql.py 
```

Use the xxxBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
from .iql_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy
```

#### offline evaluation
Load the raw traffic granularity data to construct an offline evaluation environment for assessing the bidding strategy offline.
```
python main/main_test.py
```



### online Evaluation
Set up the Hyperparameters for the online evaluation process.
```
config/test.gin
```
Run online Evaluation.
```bash
# Return to the root directory
$ python main_test.py
```


## 🎡 Implemented bid decision-making algorithms

---

| category                  | strategy             | status |
|---------------------------|----------------------| ----- |
| reinforcement learning    | IQL                  | ✅ |
|                           | BC                   | ✅     |
|                           | BCQ                  | ✅      |
|                           | IQL                  | ✅      |
|                           | TD3_BC               | ✅      |
| online linear programming | OnlineLp             | ✅      |
| Generative Model          | Decision-Transformer | ✅      |
| Other                     | Abid(fixed bid rate) | ✅      |
|                           | PID                  | ✅      |


## ✌ Contributing

---


The field of decision intelligence is a fascinating area, and we welcome like-minded individuals to contribute their wisdom and creativity to Optimize this project. If you have great ideas, feel free to fork the repo and create a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b new-branch`)
3. Commit your Changes (`git commit -m 'Add some feature'`)
4. Push to the Branch (`git push origin new-branch`)
5. Open a Pull Request


## 🏷️ License

---

Distributed under the Apache License 2.0. See `LICENSE.txt` for more information.

## 💓 Acknowledgement

---

- [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser)
- [Hands-on-RL](https://github.com/boyu-ai/Hands-on-RL)



## 🧑‍🤝‍🧑 Contributors

---

Shuai Dou • Yusen Huo • Zhilin Zhang • YeShu Li • Zhengye Han • Kefan Su  
 • Zongqing Lu • Chuan Yu • Jian Xu • Bo Zheng


## ✉️ Contact

---

For any questions, please feel free to email `doushuai.ds@taobao.com`.



## 📝 Citation

---

If you find our work useful, please consider citing:
```
@inproceedings{
su2024a,
title={A Novel Benchmark for Decision-Making in Uncertain and Competitive Games},
author={Kefan Su and Yusen Huo and Zhilin Zhang and Shuai Dou and Chuan Yu and Jian Xu and Zongqing Lu and Bo Zheng},
booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2024},
url={https://openreview.net/forum?id=OTjTKFk7gb}
}
```