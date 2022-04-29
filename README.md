# MGAD-entity-linking
Repository for paper "MGAD: Learning Descriptional Representation Distilled from Distributional Semantics for Unseen Entities" in IJCAI 2022.

Right now, only model files are available. **We'll release our finetuned checkpoints, entity embeddings, hyper-parameter configs, and training/testing scripts in the future.**  

## Data
You can download necessary data from following links (collected from [KILT](https://github.com/facebookresearch/KILT) and [BLINK](https://github.com/facebookresearch/BLINK)):

### Entity Base
Cleaned Wikipedia entity base \[[Download](http://dl.fbaipublicfiles.com/BLINK/entity.jsonl)\].

### Dataset
**Benckmark datasets** \[[Download](https://drive.google.com/uc?export=download&id=1IDjXFnNnHf__MO5j_onw4YwR97oS8lAy)\]:  including eight datasets: AIDA-train, AIDA-testA, AIDA-testB, ACE2004, MSNBC, AQUAINT, Wikipedia and Clueweb.  
**BLINK dataset**\[[BLINK-train](http://dl.fbaipublicfiles.com/KILT/blink-train-kilt.jsonl), [BLINK-dev](http://dl.fbaipublicfiles.com/KILT/blink-dev-kilt.jsonl)\]: large-scale entity linking datasets built from Wikipedia, which is the training and development set for BLINK. 
