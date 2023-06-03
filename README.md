# Microcosm
A toy generative model for dynamical systems observations.

## Installation

Microcosm relies on the chaospy package to generate samples from low dimensional chaotic systems such as the Lorenz attractor. 

To use, clone the [chaospy repo](https://github.com/hukenovs/chaospy) in the same parent folder as this microcosm repo. Install the requirements of chaospy repo given in their requirements.txt (instructions in their readme file.).


## Usage
See the demo.ipynb notebook.

## Step-by-step training a masked language model on microcosm (or similar) data


#### 0. Wrap your simulator so it outputs in json (dict) format. 

The Lorenz simulator is in microcosm.py and is called via microcosm.sample_world. A demo of its usage is [here](https://github.com/golkar/microcosm/blob/main/demo.ipynb). 

Some helpful pointers for adjusting your simulator outputs:

* In your output, it is helpful to turn all your numpy or torch or other array types into lists. This way when writing to a text file, you will not have extraneous text such as "numpy" or "array" or "torch.Tensor". In microcosm, this is done by setting to_list = True when calling sample_world.
* In your output, reduce your numerical precision to the smallest value that you think is reasonable. Manually check to make sure that when written to a text file, you don't have too many digits. This is because the number of tokens that fit in your context is one of the biggest bottlenecks and having too many digits increases the length of your tokenized samples. In microcosm, this is controlled by setting data_type when calling sample_world.

#### 1. Collect lots of samples from the simulator.

You'll need many samples! I collected 1 million samples for the microcosm demo. If your simulator is relatively fast, this can be done very efficiently by parallelizing on the cluster. In this repo [here](https://github.com/golkar/microcosm/blob/main/draw_many_xsmall.py) is the script that runs the simulator and saves it to a random file name on ceph, [here](https://github.com/golkar/microcosm/blob/main/seqdraw_xsmall_script) is the slurm script that collects 1.1 million samples (it runs draw_many_xsmall.py independently on 20 nodes and each node has 5 workers (i.e. 20x5=100 workers), each worker collecting 11k examples.) Request more cpu nodes or parallelize more/less per node depending on your simulator needs. Finally, [this script](https://github.com/golkar/microcosm/blob/main/aggregating_savefiles.py) aggregates the results from the different tasks and writes them to separate train/val/test files. 

Helpful pointers:

* Make sure your simulator or data collection can handle any errors in the simulator. You will need to generate many many samples and don't want to restart the data collection procedure. (Even if your simulator fails on 0.01% of the times, you'll have to restart it 100 times to collect 1 million samples.) 
* It's good practice to set aside your test set at this stage and never look at it until after you have selected your final model.

#### 2. Define a tokenizer, tokenize the dataset and write to file.

You can "learn" a tokenizer that optimizes your dataset code etc, but in this case because the syntax and vocabulary are extremely structured, it makes more sense to define the tokenizer manually. The tokens include single digits 0-9, double digits 00-99, negative symbol (0) dict letters ({}[],.) and the keys of the dict ('description':, 'data':, 'params':, etc) and for very small number I included (000,0000,e-05-e-08). This is all done [here](https://github.com/golkar/microcosm/blob/main/tokenizer_lorenz.py) and the resulting tokenizer is saved to file as tokenizer_lorenz.json.

Finally, tokenize the dataset and write it to file. Here the DatasetDict.datasets from HF really helps as it is a highly efficient way to tokenize. See [here](https://github.com/golkar/microcosm/blob/main/tokenize_dataset.py) . 

Helpful pointers:
* Try to minimize the number of tokens needed. For example, by including the single quote (') and colon (:) in the single token ('description':), we go from needing 4 tokens ( ' + description + ' + : ) to only one.
* If you want to try to "learn" your tokenizer, an example is given [here](https://github.com/golkar/microcosm/blob/main/tokenizer_lorenz_trained.py)  but this ends up having too many tokens for many reasons (e.g. the BPE tokenizer will include all the letters that it has seen as tokens). This makes sense for a general corpus but not for us where we have a very limited number of tokens appearing. 

#### 3. Define a data-collator for masking the samples.

The final step of processing the data for masked-language-modeling (MLM) is to collate samples, pad them so that they have the same length and then randomly replace some tokens with the mask token. All this is automatically done by using the HF DataCollatorForLanguageModeling. (lines 85-100 [here](https://github.com/golkar/microcosm/blob/main/masked_roberta_lorenz.py)  , you can see the result of the datacollator in the printed output on line 96). 

* https://github.com/golkar/microcosm/blob/main/masked_roberta_lorenz.py is just a demo, for the actual training on the cluster see below.

#### 4. Define the model and the trainer (via HF).

Defining the model via HF is easy (lines 100-130 of [here](https://github.com/golkar/microcosm/blob/main/masked_roberta_lorenz.py)) Similarly setting up a trainer is straightforward (lines (130-160).

Some pointers:
* Set the max_position_embeddings to the longest tokenized sample length in your dataset. This is basically the longest sequence that can be handled because the model needs to set up positional encodings.
* Passing an empty compute_metrics function (as in lines 147 and 157) tells the trainer to report the validation loss. Without this, no validation results will be logged. 
* Having report_to="wandb" on line 142 will log all the  stats to WandB. It's very convenient. 

#### 5. Train on the cluster! (WandB sweep)

To do a hyperparameter search you can setu up WandB sweeps. [Here](https://github.com/golkar/microcosm/blob/main/mroberta_xslorenz_sweep_setup.py) is the script that sets up a wandb sweep and [this](https://github.com/golkar/microcosm/blob/main/mroberta_xslorenz_sweep_agent.py) is the script that runs the training while pulling parameters from wandb.  Finally, [this](https://github.com/golkar/microcosm/blob/main/script_xslorenz_mroberta_agent) is the slurm script that submits the job. 

To do this for your own project, you'll need to set up a new wandb project up and refer to that in your code. Also you need to modify the slurm script to request gpu's appropriate for your training.

* If you're running a similar model on similar data, instead of running a sweep you can just use the hyperparameters that worked best for this project (you can do this by providing the following dict to the train function as the config, instead of calling the training via wandb.agent at the bottom). These are given by:

{
    "learning_rate": {"value": 1e-4},
    "hidden_size": {"value": 720},  
    "intermediate_size": {"value": 2880},
    "num_hidden_layers": {"value": 12},
    "num_attention_heads": {"value": 6},
    "hidden_act": {"value": "gelu"},
    "hidden_dropout_prob": {"value": 0.1},
    "attention_probs_dropout_prob": {"value": 0.1},
    "mlm_probability": {"value": 0.2},
    "num_train_epochs": {"value": 2},
    "warmup_steps": {"value": 2000},
    "weight_decay": {"value": 0},
    "fp16": {"value": True},
}

These are the parameters that I used but setting hidden_dropout_prob=0 and having weight_decay=1e-4 might work better but you will likely need a different learning rate. 

#### 6. Inference via HF mask-filling pipeline.

Once training is complete, you can perform inference using the transformers pipeline. An example of setting this up is given [here](https://github.com/golkar/microcosm/blob/main/xslorenz_mroberta_results_chkpt500000.ipynb) (cell 5 sets up the pipeline and mask_filler function)

