# Rutgers MSCS Projects

This repository contains projects completed during the Master of Science in Computer Science program at Rutgers University. The projects span multiple core areas including Data Structures & Algorithms, Artificial Intelligence, and Machine Learning.

---

## Table of Contents

- [CS 512: Data Structures and Algorithms](#cs-512-data-structures-and-algorithms)
- [CS 520: Introduction to Artificial Intelligence](#cs-520-introduction-to-artificial-intelligence)
  - [Project 1: This Bot Is On Fire](#project-1-this-bot-is-on-fire)
  - [Project 2: Space Rats](#project-2-space-rats)
  - [Project 3: Localisation with Utility and ML](#project-3-localisation-with-utility-and-ml)
- [CS 535: Machine Learning](#cs-535-machine-learning)

---

## CS 512: Data Structures and Algorithms

### Transformation: Rudrata Cycle to Reliable Network

**Directory:** `512_DS_and_Algo/transform_rudrata_cycle_to_reliable_network/`

#### Overview
This project explores the computational relationship between two NP-complete problems: the Rudrata Cycle problem and the Reliable Network problem. The project demonstrates a polynomial-time transformation between these problems and implements brute-force solvers to verify the transformation's correctness.

#### Problem Definitions

**Rudrata Cycle Problem:**
- **Input:** An undirected graph G = (V, E)
- **Output:** YES if there exists a simple cycle that visits every vertex exactly once; NO otherwise

**Reliable Network Problem:**
- **Input:**
  - n vertices
  - Distance matrix d[i,j] (edge weights: 1 for direct connection, 2 for no direct edge)
  - Resource matrix r[i,j] (required number of vertex-disjoint paths between vertices)
  - Budget b
- **Output:** YES if a network can be constructed meeting all requirements within budget; NO otherwise

#### Transformation Strategy
The transformation from Rudrata Cycle to Reliable Network is straightforward:
- **d[i,j]:** Set to 1 if edge (i,j) exists in original graph, 2 otherwise
- **r[i,j]:** Set to 2 for all pairs (requiring 2 vertex-disjoint paths)
- **Budget:** Set to n (number of vertices)

The key insight is that a Rudrata cycle in the original graph directly corresponds to a valid reliable network solution, as the cycle provides exactly 2 vertex-disjoint paths between any pair of vertices.

#### Implementation Highlights
- **Jupyter Notebook:** `Group5_project.ipynb`
- **Brute Force Solvers:** Implements both Rudrata Cycle and Reliable Network solvers for verification
- **Visualization:** NetworkX-based graph plotting for input graphs, distance matrices, and solutions
- **Sample Inputs:** YES and NO instances for testing

#### Files
- `Group5_project.ipynb` - Main implementation and analysis
- `Project Report.docx` - Detailed project report
- `Transformation – Rudrata Cycle to Reliable Network.pptx` - Presentation slides
- `Sample inputs/` - Test cases for verification

---

## CS 520: Introduction to Artificial Intelligence

This course included three major projects involving search algorithms, probabilistic reasoning, and machine learning for decision-making in grid-based environments.

---

### Project 1: This Bot Is On Fire

**Directory:** `520_Intro_to_AI/this_bot_is_on_fire/`

#### Overview
A fire-spread simulation where an intelligent bot must navigate through a burning ship to reach a button. The project implements and compares multiple pathfinding strategies in a dynamic environment where fire spreads probabilistically.

#### Environment
- **Grid:** 40×40 ship layout with blocked and open cells
- **Fire Spread:** Probabilistic model with parameter q
  - Probability of cell catching fire: 1 - (1 - q)^K, where K is the number of burning neighbors
- **Goal:** Bot must reach button before being caught by fire

#### Bot Strategies

**Bot 1:** Static A* pathfinding
- Plans path once at the beginning
- Does not replan as fire spreads
- Fastest planning, but inflexible

**Bot 2:** Dynamic A* replanning
- Replans path at each timestep
- Avoids fire cells but not adjacent-to-fire cells
- Adaptive to changing environment

**Bot 3:** Fire-aware pathfinding
- Avoids cells adjacent to fire
- Falls back to Bot 2 strategy if no safe path exists
- Most conservative approach

**Bot 4:** Risk-weighted A*
- Uses inverse distance from original fire position as edge costs
- Heuristic balances path length with fire risk
- Attempts to maximize distance from dangerous areas

#### Key Features
- A* search with Manhattan distance heuristic
- Fire spread animation and visualization
- Winnability analysis: 3D A* search through space-time to find theoretically optimal paths
- Comparative performance metrics across 2000 test scenarios
- Ship generation algorithm creating maze-like structures

#### Files
- `main.py` - Core bot implementations and simulation runner
- `animate_fire_spread.py` - Fire spread visualization
- `plots.py` - Performance analysis and plotting
- `visualise_ship.py` - Ship layout visualization

---

### Project 2: Space Rats

**Directory:** `520_Intro_to_AI/space_rats/`

#### Overview
A two-phase localization and search problem where a bot must first determine its own location on a ship, then locate and capture a rat using probabilistic sensing.

#### Phase 1: Bot Self-Localization
- Bot doesn't know its initial position
- **Sensing:** Count of blocked neighbors (including diagonal, out-of-bounds treated as blocked)
- **Actions:** Attempt to move in cardinal directions; success/failure provides information
- **Strategy:** Iteratively filter possible locations until unique position is determined

#### Phase 2: Rat Localization
- Rat is stationary but location unknown
- **Sensing:** Beep detection with distance-based probability
  - P(beep | distance d) = exp(-α(d-1))
  - Returns 2 if rat is on same cell, 1 if beep detected, 0 if no beep
- **Strategy:** Bayesian probability updates to maintain belief distribution over rat location

#### Bot Implementations

**Bot 1 (Baseline):**
- Random movement for self-localization
- Single beep per location for rat search
- Greedy approach: always move to cell with maximum rat probability

**Bot 2 (Improved):**
- Multiple beeps (15) at starting location to create "fuzzy radius"
- Makes 5 moves then 5 beeps to refine probability distribution
- Better exploitation of probabilistic information

#### Key Features
- Bayesian inference for probability updates
- A* pathfinding for efficient navigation
- Simultaneous state representation (BOT_RAT when bot and rat occupy same cell)
- Performance comparison over 1000 test scenarios with varying α values
- Ship generation matching Project 1 specifications

#### Files
- `project_2.py` - Main bot implementations
- `project_2_rat_moves.py` - Extended version with moving rat
- `other_bot_simulations/` - Alternative bot strategies (Bots 3-13)
- `writeup_plots.py` - Performance analysis visualization

---

### Project 3: Localisation with Utility and ML

**Directory:** `520_Intro_to_AI/localisation_utility_ml/`

#### Overview
An advanced localization problem combining utility-based decision making and machine learning. The bot must localize itself using a set L of possible positions, updating L after each move.

#### Problem Setup
- **State:** Set L of possible bot locations
- **Goal:** Reduce |L| to 1 (unique localization)
- **Mechanics:** When bot moves in direction d, all cells in L update:
  - If next cell in direction d is blocked/out-of-bounds, position stays same
  - Otherwise, position updates to next cell in direction d

#### Policies

**π₀ (Baseline - Utility-Based):**
- Selects target locations in dead-ends or corners (cells with ≤1 or ≤2 perpendicular neighbors)
- Uses A* to navigate random cell from L to target
- Updates L after each move
- Detects and breaks out of loops (max 5 repetitions of same L state)

**π₁ (1-Step Lookahead with ML):**
- Trains neural network to predict cost-to-completion for π₀
- **Input:** Grid structure + L configuration (2-channel tensor)
- **Output:** Expected number of moves to localize
- Looks ahead 1 step: tries all 4 directions, picks direction minimizing predicted cost
- Continues with π₀ after first move

**π₂ (2-Step Lookahead with ML):**
- Trains neural network to predict cost-to-completion for π₁
- Looks ahead 1 step using π₁ model
- Continues with π₁ strategy after first move
- Stacks learning: uses π₁'s performance to improve

#### Machine Learning Components
- **Architecture:** Convolutional Neural Network
  - Input: 2 channels (grid layout, L positions)
  - Output: Scalar cost prediction
- **Training:** Supervised learning on trajectories from baseline policies
- **Data Generation:** Automated collection from π₀ and π₁ runs
- **Models:** Pre-trained models saved in `models/` directory

#### Key Features
- Ship generation with fixed random seed for reproducibility
- Data collection and model training pipeline
- Comparative performance analysis across policies
- Visualization of actual vs predicted costs
- PyTorch-based neural network implementation

#### Files
- `localization_functions.py` - Core localization algorithms (π₀, π₁, π₂)
- `models.py` - Neural network architecture definitions
- `split_train_save_model.py` - Training pipeline
- `pi_0_work.py`, `pi_1_work.py`, `pi_2_work.py` - Individual policy runners
- `compare_pi_0_pi_1_pi_2.py` - Performance comparison
- `actual_plots.py`, `prediction_plots.py` - Visualization tools
- `ship_visualiser.py` - Grid visualization
- `models/` - Trained neural network weights
- `plots/` - Generated performance plots

---

## CS 535: Machine Learning

### Categorical Naive Bayes on EMNIST Dataset

**Directory:** `535_Machine_Learning/`

#### Overview
Implementation of a Categorical Naive Bayes classifier from scratch with both Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimators. The project analyzes the impact of prior hyperparameters on learning curves using the EMNIST balanced dataset.

#### Dataset
- **EMNIST Balanced:** 47 classes (digits + uppercase + lowercase letters)
- **Preprocessing:** Images rotated and flipped to correct orientation, binarized
- **Split:** 10% training (11,200 samples), 90% test (100,800 samples)

#### Implementation

**Model:**
- Binary pixel features (784 features per 28×28 image)
- Class prior: P(y = k | θ) with Dirichlet prior
- Pixel likelihood: P(x_i = 1 | y = k, θ) with Beta prior

**Estimators:**
1. **MLE (Maximum Likelihood):**
   - θ_k = N_k / N (class prior)
   - θ_{i,k} = C_{i,k} / N_k (pixel probability)
   - No regularization

2. **MAP (Maximum A Posteriori):**
   - θ_k = (N_k + α) / (N + C·α) (class prior with Dirichlet(α))
   - θ_{i,k} = (C_{i,k} + β) / (N_k + 2β) (pixel probability with Beta(β, β))
   - Hyperparameters α and β control regularization strength

**Scikit-learn Compatible API:**
- Implements `BaseEstimator` and `ClassifierMixin`
- Methods: `fit()`, `predict()`, `predict_proba()`, `score()`
- Score function: average log-likelihood on dataset

#### Experiments

**Task 1:** Sample Visualization
- 5×47 grid showing 5 samples from each of 47 classes

**Task 3:** Balanced Data Learning Curves
- **3.1:** Fixed β=1, varying α ∈ {1, 10, 50, 100, 200}
- **3.2:** Fixed α=1, varying β ∈ {1, 2, 10, 100}
- Analysis: Impact of class vs pixel priors on generalization

**Task 4:** Imbalanced Data Learning Curves
- Class proportions sampled from Dirichlet(α_class)
- α_class ∈ {0.1, 0.2, 0.5, 1, 10, 100} (lower = more imbalanced)
- **4.1:** Fixed α=1, varying β ∈ {1, 1.2, 2, 10, 100}
- **4.2:** Fixed β=1, varying α ∈ {1, 10, 100, 1000}
- Analysis: Regularization effectiveness under class imbalance

#### Key Findings
- **Class Prior (α):** Less impact on balanced data due to similar class frequencies; critical for imbalanced data
- **Pixel Prior (β):** Significant impact in low-data regime; prevents overfitting to spurious pixel patterns
- **Imbalanced Data:** Higher α and β provide better regularization when class distribution is skewed

#### Files
- `mini_project.py` - Complete implementation with all tasks
- `requirements.txt` - Python dependencies
- `mini_project.docx` - Detailed project report
- `derivations.tex` - Mathematical derivations for MLE and MAP
- `CS_535_Mini_Project.pdf` - Project specification
- `task*.jpg` - Generated plots for each task

#### Mathematical Details
The project includes complete derivations showing:
- MLE estimates from likelihood maximization
- MAP estimates from posterior maximization with conjugate priors
- Relationship between priors and regularization
- Log-likelihood computation for numerical stability

---

## Technologies Used

### Languages
- Python 3.x

### Libraries
**Core Scientific Computing:**
- NumPy - Numerical operations and array manipulation
- Pandas - Data management and CSV operations
- Matplotlib - Visualization and plotting

**Machine Learning:**
- PyTorch - Neural networks for AI Project 3 and ML course
- scikit-learn - Base estimator classes and data splitting
- torchvision - EMNIST dataset loading

**Graph & Optimization:**
- NetworkX - Graph algorithms and visualization
- heapq - Priority queue for A* search

**Utilities:**
- random - Stochastic simulations
- itertools - Combinatorial operations

---

## Project Structure

```
Projects/
├── 512_DS_and_Algo/
│   └── transform_rudrata_cycle_to_reliable_network/
│       ├── Group5_project.ipynb
│       ├── Project Report.docx
│       ├── Transformation – Rudrata Cycle to Reliable Network.pptx
│       └── Sample inputs/
├── 520_Intro_to_AI/
│   ├── this_bot_is_on_fire/
│   │   ├── main.py
│   │   ├── animate_fire_spread.py
│   │   ├── plots.py
│   │   └── visualise_ship.py
│   ├── space_rats/
│   │   ├── project_2.py
│   │   ├── project_2_rat_moves.py
│   │   ├── writeup_plots.py
│   │   └── other_bot_simulations/
│   └── localisation_utility_ml/
│       ├── localization_functions.py
│       ├── models.py
│       ├── split_train_save_model.py
│       ├── pi_0_work.py
│       ├── pi_1_work.py
│       ├── pi_2_work.py
│       ├── ship_visualiser.py
│       ├── models/
│       └── plots/
└── 535_Machine_Learning/
    ├── mini_project.py
    ├── requirements.txt
    ├── derivations.tex
    └── [generated plots]
```

---

## Running the Projects

### CS 512: Rudrata Cycle Transformation
```bash
cd 512_DS_and_Algo/transform_rudrata_cycle_to_reliable_network/
jupyter notebook Group5_project.ipynb
```

### CS 520: AI Projects

**This Bot Is On Fire:**
```bash
cd 520_Intro_to_AI/this_bot_is_on_fire/
python main.py
```

**Space Rats:**
```bash
cd 520_Intro_to_AI/space_rats/
python project_2.py
```

**Localisation with Utility and ML:**
```bash
cd 520_Intro_to_AI/localisation_utility_ml/
pip install -r requirements.txt
python pi_0_work.py  # Run baseline
python pi_1_work.py  # Run 1-step lookahead
python pi_2_work.py  # Run 2-step lookahead
```

### CS 535: Machine Learning
```bash
cd 535_Machine_Learning/
pip install -r requirements.txt
python mini_project.py
```

---

## Key Algorithms & Techniques

### Search Algorithms
- **A* Search:** Optimal pathfinding with Manhattan distance heuristic
- **Space-Time A*:** 3D search for fire-spread winnability analysis
- **Greedy Best-First:** Probability-driven rat localization

### Probabilistic Methods
- **Bayesian Inference:** Belief distribution updates from sensor data
- **Exponential Distance Model:** Distance-dependent detection probabilities
- **Fire Spread Simulation:** Probabilistic cellular automaton

### Machine Learning
- **Convolutional Neural Networks:** Grid-based state representation learning
- **Supervised Learning:** Policy cost prediction from demonstration data
- **Categorical Naive Bayes:** Generative classification with conjugate priors
- **Regularization:** MAP estimation with Dirichlet and Beta priors

### Optimization & Complexity
- **Polynomial-Time Reductions:** NP-complete problem transformations
- **Brute-Force Search:** Exhaustive backtracking with pruning
- **Graph Algorithms:** Connectivity, Hamiltonian cycles, network design

---

## Academic Context

These projects were completed as part of the Master of Science in Computer Science program at Rutgers University. Each project demonstrates:

- **Theoretical Understanding:** NP-completeness, probabilistic reasoning, statistical learning theory
- **Practical Implementation:** Efficient algorithms, numerical stability, scalable code
- **Experimental Analysis:** Comprehensive testing, comparative studies, visualization
- **Technical Communication:** Detailed reports, presentations, mathematical derivations

---

## License

These projects are academic coursework. Please respect academic integrity policies if you are currently enrolled in similar courses.

---

## Author

Abhishek - Rutgers University MSCS Student

*Repository created for portfolio and reference purposes.*
