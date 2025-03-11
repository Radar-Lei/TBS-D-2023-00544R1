# Accelerate Transit Network Design Using Graph Clustering (TBS-D-2023-00544R1)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation for the paper *"Accelerate transit network design problem-solving based on large-scale smart card data and graph-clustering decomposition"*

## 📁 Project Structure

```
.
├── algo/                   # Core algorithms implementation
│   ├── D_Heuristics.py     # Demand-driven heuristics
│   ├── dijkstra_heapq.py   # Optimized Dijkstra's algorithm
│   ├── fitness_evaluation.py # Solution evaluation metrics
│   ├── ga_tsp.py           # Genetic Algorithm for TSP
│   ├── louvain.py          # Louvain community detection
│   └── population.py       # Population management
├── utils/                  # Utility modules
│   ├── coordTransform_utils.py # Coordinate transformations
│   ├── fetch_station_Gaode.py  # Gaode Map API integration
│   └── graph_builder.py    # Network graph construction
├── results/                # Precomputed optimization results
├── archive/                # Historical route set configurations
│
├── 01-params_preparation.py         # Data preprocessing pipeline
├── 02-graph_built_on_gaode.py       # Network graph construction
├── 03-computing cost for original route set.py # Baseline evaluation
├── 04-evaluating cost for optimized route set mumford3.py # Optimized solution evaluation
├── 07-visualization.ipynb           # Interactive results visualization
├── 08-TND_mumford3.py               # Main optimization workflow
├── 09-GCH_*                         # Graph clustering components
└── py38-GIS.yaml                    # Conda environment configuration
```

## 🛠 Installation

1. Clone repository:
```bash
git clone https://github.com/your-organization/TBS-D-2023-00544R1.git
cd TBS-D-2023-00544R1
```

2. Create conda environment:
```bash
conda env create -f utils/py38-GIS.yaml
conda activate tnd-py38
```

3. Install additional dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

1. Download input data from [Google Drive](https://drive.google.com/file/d/1uRprWmk91mh_Z77lIbkFza3kESyFELgp/view?usp=drive_link)
2. Place dataset files in `/data/raw` directory
3. Run data preprocessing:
```bash
python 01-params_preparation.py --input_dir ./data/raw --output_dir ./data/processed
```

## ▶️ Basic Usage

Run full optimization pipeline:
```python
python 08-TND_mumford3.py \
    --input_data ./data/processed/network_graph.gpickle \
    --output_dir ./results \
    --max_iterations 1000 \
    --population_size 50
```

Visualize results:
```bash
jupyter notebook 07-visualization.ipynb
```



## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
