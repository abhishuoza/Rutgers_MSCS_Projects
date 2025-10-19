# Rutgers MSCS Projects

This repository contains coursework projects from my Masters in Computer Science at Rutgers University.

## Repository Structure

```
├── 512_DS_and_Algo/
│   └── transform_rudrata_cycle_to_reliable_network/
├── 520_Intro_to_AI/
│   ├── this_bot_is_on_fire/
│   ├── space_rats/
│   └── localisation_utility_ml/
└── 535_Machine_Learning/
    └── emnist_naive_bayes_classifier/
```

---

## CS 512: Data Structures and Algorithms

### Transformation: Rudrata Cycle to Reliable Network

This project explores the computational relationship between two NP-complete problems through polynomial-time transformation. The Rudrata Cycle problem asks whether a graph contains a simple cycle visiting every vertex exactly once, while the Reliable Network problem requires constructing a network that meets distance and connectivity requirements within a budget. I implemented the transformation by converting an adjacency matrix into distance and resource matrices, where the transformation maintains that a YES instance of Rudrata Cycle maps to a YES instance of Reliable Network.

The implementation includes brute-force solvers for both problems to verify correctness of the transformation. The Rudrata solver uses depth-first search with backtracking, tracking both failed full-length cycles and partial dead-end paths. The Reliable Network solver employs backtracking with budget-based pruning and NetworkX's node connectivity algorithms to verify the required number of vertex-disjoint paths. The project demonstrates that the solution to the Reliable Network problem is exactly the adjacency matrix of the Rudrata cycle.

**Implemented:**
- Polynomial-time transformation between NP-complete problems
- Brute-force verification solvers with detailed failure logging
- NetworkX-based graph visualization and analysis
- Sample test cases demonstrating YES and NO instances

---

## CS 520: Introduction to Artificial Intelligence

### Project 1: This Bot Is On Fire

A bot navigates through a burning 40×40 ship to reach a button while fire spreads probabilistically based on the number of burning neighbors. I implemented four different bot strategies using A* search with varying levels of adaptability and risk awareness. Bot 1 plans once statically, Bot 2 replans every step while avoiding fire, Bot 3 additionally avoids cells adjacent to fire, and Bot 4 uses a risk-weighted cost function based on inverse distance from the original fire position. The project compares these strategies across 2000 simulations with varying fire spread rates.

Beyond bot performance, I implemented a winnability analysis using 3D space-time A* search to determine if any legal path to the button exists for a given fire spread sequence. This provides a theoretical upper bound on achievability. The ship generation algorithm creates maze-like structures by iteratively opening cells with exactly one open neighbor, then randomly opening additional neighbors of dead-ends to add complexity. All bot movement and fire spread uses Manhattan distance and 4-directional connectivity.

**Implemented:**
- Four A* pathfinding strategies with different risk profiles
- Probabilistic fire spread simulation: P(ignite) = 1 - (1-q)^K
- 3D space-time search for winnability analysis
- Comprehensive performance comparison across 2000 scenarios

### Project 2: Space Rats

This two-phase localization problem requires a bot to first determine its own unknown position on a ship, then locate and capture a stationary rat using probabilistic sensing. Phase 1 uses neighbor sensing (counting blocked cells in the 8-neighbor vicinity) and attempted movements to iteratively filter the set of possible bot locations. Each sensing result and movement outcome (success or blocked) eliminates impossible positions until only one candidate remains. Phase 2 employs distance-based beep detection with probability P(beep|distance d) = exp(-α(d-1)) and Bayesian updates to maintain a probability distribution over rat locations.

I implemented two bots with different sensing strategies. Bot 1 takes a single beep at each location and greedily moves to the cell with maximum rat probability. Bot 2 performs 15 beeps at the starting location to create a "fuzzy radius" of probability, then alternates between making 5 moves and 5 beeps to continuously refine its belief distribution. Both bots use A* for efficient pathfinding to target cells. The Bayesian update correctly handles both beep and no-beep observations, updating probabilities based on the likelihood of each outcome given the distance to each candidate rat location.

**Implemented:**
- Two-phase localization: self-localization via movement filtering, rat-localization via probabilistic sensing
- Bayesian inference for belief distribution updates
- Comparison of greedy vs information-gathering sensing strategies
- Performance analysis over 1000 scenarios with varying detection sensitivity (α)

### Project 3: Localisation with Utility and ML

This project combines utility-based decision making with machine learning for bot self-localization. The bot maintains a set L of possible positions, and when it moves in a direction, each position in L updates according to whether the next cell is blocked. The goal is to reduce |L| to 1 through strategic movement. I implemented three policies of increasing sophistication: π₀ uses utility-based pathfinding to dead-ends and corners to break symmetries, π₁ uses a neural network trained on π₀'s performance to make better first moves via 1-step lookahead, and π₂ similarly improves on π₁ through learned 2-step lookahead.

The machine learning component uses convolutional neural networks with 2-channel inputs encoding the grid structure and current L configuration. The networks predict the expected number of moves to completion under the respective baseline policy. During training, I collected trajectory data from π₀ and π₁ runs, storing (grid, L, cost-to-go) tuples for supervised learning. The policies demonstrate performance improvement through lookahead: π₁ makes better initial moves than π₀, and π₂ makes better initial moves than π₁, showing that learned cost functions successfully guide exploration toward faster localization.

**Implemented:**
- Three policies: utility-based baseline, 1-step ML lookahead, 2-step ML lookahead
- CNN architecture for grid-based state representation and cost prediction
- Automated data collection and training pipeline using PyTorch
- Loop detection and symmetry-breaking via target reselection

---

## CS 535: Machine Learning

### EMNIST Naive Bayes Classifier

I implemented a Categorical Naive Bayes classifier from scratch with both Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimators, following scikit-learn's estimator interface. The classifier works on the EMNIST balanced dataset (47 classes of handwritten characters) with binarized 28×28 images. MLE provides unregularized parameter estimates while MAP incorporates Dirichlet priors on class probabilities (controlled by α) and Beta priors on pixel probabilities (controlled by β). The implementation uses log-space arithmetic throughout for numerical stability and computes average log-likelihood as the scoring metric.

The experimental analysis examines learning curves under both balanced and imbalanced training conditions. For balanced data, I varied α and β independently to understand their regularization effects, finding that pixel priors (β) have more impact than class priors (α) when class frequencies are similar. For imbalanced data generated via Dirichlet-sampled class proportions, both hyperparameters become critical. Higher β values prevent overfitting to spurious pixel patterns in small classes, while higher α values regularize the class distribution toward uniformity, improving generalization when training data is skewed.

**Implemented:**
- From scratch MLE and MAP estimation with conjugate priors
- Learning curve analysis varying α ∈ {1,10,50,100,200} and β ∈ {1,2,10,100}
- Imbalanced data experiments with Dirichlet(α_class) sampling where α_class ∈ {0.1,0.2,0.5,1,10,100}
- Mathematical derivations and comparative performance visualization

---
