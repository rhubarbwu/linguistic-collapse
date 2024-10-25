# Linguistic Collapse: Neural Collapse in (Large) Language Models

Codebase for [arXiv:2405.17767](https://arxiv.org/abs/2405.17767), based on [GPT-Neo](https://github.com/EleutherAI/gpt-neo) and [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

## Environment

Python dependencies can be found in [`requirements.txt`](./requirements.txt). The directory structure we used was placing dataset(s), model checkpoints and analysis artifacts in a single `$SCRATCH` directory with plenty of unused space, while our scripts and CSVs were kept in some home directory as they didn't consume much space. We elected to store our analysis artifacts (embeddings) in `$SCRATCH/stats` and model checkpoints in `$SCRATCH/TS` (standing for "TinyStories").

Some of our scripts make references to an environment file [`env-h`](./env-h) that starts a Python environment, defines shorthand shell functions and imports home variables.

Our codebase is most compatible with a SLURM environment configured for single-GPU runs, but most of the scripts (those without `batch` in their name) can be run in the shell directly.

## Model Training

To prepare a model for training, create a folder (probably in `$SCRATCH`) and copy [`config.json`](./config.json) into it. Adapt the architectural details and hyperparameters in that file as you need.

We used a relatively [standard script from Huggingface](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) to train our CLMs. The code was lightly adapted and formatted in [`run_clm.py`](./run_clm.py). This script is invoked by [`train.sh`](./train.sh), which provides an example of training models on an A100 GPU.

Here's an example 205M model that we've made public: https://huggingface.co/rhubarbwu/TinyStories-12x1024_10L

### Dispatching train jobs for several architectures

Use [`batch-train.sh`](./batch-train.sh), but note the variables that should be set before and within the `launch()` function declaration.

### Training the same architecture with multiple seeds

Assuming you already set up the [`config.json`](./config.json) for your desired architecture, you can add a simple `bash` loop into [`batch-train.sh`](./batch-train.sh). Here's the loop that we wrote for our experiments, where `$SCRATCH` is a directory in which we store temporary checkpoints.

```sh
for SEED in {10..19}; do
    new_dir=$SCRATCH/TS/TinyStories-02x0768_01d$SEED
    mkdir $new_dir
    cp $SCRATCH/TS/TinyStories-02x0768_01d/config.json $new_dir
    launch 02 0768 2 16 $SEED
done
```

### Architecture

We use [GPT-Neo, developed by EleutherAI](https://github.com/EleutherAI/gpt-neo). You could also adapt our setup to [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) or any other causal architecture.

### Bring your own model

It is also easy to train your own LLMs separately. Just take care to use the exact same train set configuration (in our setup, note the version of TinyStories and the number of preprocessing workers) between training and the collection of means and variances for analysis.

## Model Evaluation

After a model is trained, you can perform evaluation, which will add `eval_results.json` to that model directory (or checkpoint therein).

```sh
python run_clm.py --model_name_or_path $MODEL_DIR --output_dir $CKPT_DIR --tokenizer_name EleutherAI/gpt-neo-125M --do_eval --per_device_eval_batch_size $BATCH_SIZE --cache_dir $SCRATCH --dataset_name $DATASET --dataloader_num_workers 2 --preprocessing_num_workers 2 --run_name $CKPT --trust_remote_code --model_ckpt_idx $IDX --report_to none
```

## Embeddings Collection

In a style similar to `train.sh` and [`config.json`](./config.json), you can use [`coll-clm.sh`](./coll-clm.sh) and [`batch-coll.sh`](./batch-coll.sh) to perform embeddings collection. The `--stage` argument from `coll-clm.sh` to `run_clm.py` takes `means`, `vars`, or `decs`, referring to the collection of means, variances, and NCC decisions. Note that the `vars` and `decs` stages are both dependencies on the completion of the `means` stage. You can use the ID of the `means` job as a SLURM dependency argument `$5` to `launch()` in [`batch-coll.sh`](./batch-coll.sh).

To check the progress of collection stages, run `analyze $@ -prog`. Here's an example:

```sh
analyze -prog -i $SCRATCH/stats/*/*02x0768_01d0*@*
```

The output should look something like this:

```sh
-------------------------------------------------------------------------------
model             means   vars   decs unique
02x0768_01d00@0  229367 229367   2303  29233
02x0768_01d01@0  229367 229367   2303  29233
02x0768_01d02@0  229367 229367   2303  29233
02x0768_01d03@0  229367 229367   2303  29233
02x0768_01d04@0  229367 229367   2303  29233
02x0768_01d05@0  229367 229367   2303  29233
02x0768_01d06@0  229367 229367   2303  29233
02x0768_01d07@0  229367 229367   2303  29233
02x0768_01d08@0  229367 229367   2303  29233
02x0768_01d09@0  229367 229367   2303  29233
total (10)       229367 229367   2303  29233
------------------------------------------------------------------------------
```

## Analysis of Neural Collapse

Analysis of different measurements is done with [`analyze.py`](./analyze.py). Depending on which measurements you're making, you may or may not need a GPU (`ENV=GPU`), checkpoints (`ENV=CKPT`), variances (`-snr`) or decisions (`-decs`).

Here's a snippet from `batch-analyze.sh`.

```sh
case $ENV in
GPU)
    # require large parallel tensor operations on the GPU
    analyze -etf -kern log -snr -o $OUTPUT -i $FILES
    ;;
CKPT)
    # require the trained model checkpoints but no GPU
    analyze -dual -loss -o $OUTPUT -i $FILES
    ;;
CPU)
    # do not require checkpoints nor GPUs
    analyze -decs -nor -o $OUTPUT -i $FILES
    ;;
esac
```

| Measurement                                  | Flag        | Prerequisites      |
| -------------------------------------------- | ----------- | ------------------ |
| Within-Class Variability ($\mathcal{NC}1$)   | `-snr`      | means, variances   |
| Norms ($\mathcal{(G)NC}2$)                   | `-nor`      | means              |
| Interference ($\mathcal{NC}2$)               | `-etf`      | means              |
| Hyperspherical Uniformity ($\mathcal{GNC}2$) | `-kern log` | means              |
| Self/Uniform-Duality ($\mathcal{(U)NC}3$)    | `-dual`     | means, checkpoints |
| Agreement ($\mathcal{NC}4$)                  | `-decs`     | means, decisions   |
| Generalization (and other model info)        | `-loss`     | checkpoints        |

If all goes well, a CSV-formatted dataframe should be generated. See [./artifacts/](./artifacts/) for time-stamped examples.

### Visualizations

The dataframe could easily be accessed and visualized with some simple matplotlib script, but we are currently sharing our notebooks (based on our own analysis artifacts) to make it easy:

- [The effect of scaling on linguistic collapse.](https://colab.research.google.com/drive/1_PVBqYknv4OH9PzIdFOEpybp16DZKzuT?usp=sharing)
- [Scale-independent relationship with generalization across multiple seeds.](https://colab.research.google.com/drive/1OqZ4JNITFuZt7jT3-_Q3hr8x2Q_Lvvwa?usp=sharing)

## Corrections/Questions

If there are any bugs or inefficiences in our code, or any other questions, we'd be happy to take a look. We prefer that you open an issue on this repository, but the corresponding author can be reached at [`rupert@cs.toronto.edu`](mailto:rupert@cs.toronto.edu). We also review [pull requests](https://github.com/rhubarbwu/linguistic-collapse/pulls).

## Citing

```tex
@misc{wu2024linguisticcollapse,
      title={Linguistic Collapse: Neural Collapse in (Large) Language Models},
      author={Robert Wu and Vardan Papyan},
      year={2024},
      eprint={2405.17767},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.17767},
}
```
