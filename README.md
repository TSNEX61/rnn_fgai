# Word-Level Autoregressive Text Generation (ICT4403 Assignment 2)

## Objective
Compare Simple RNN, bidirectional RNN, LSTM, and GRU models for next-word prediction and autoregressive generation under shared data splits and training settings.

## Dataset
Tiny Shakespeare is a collection of Shakespeare plays in plain text. The local source file is `tinyshakespeare.txt`; original source: [Karpathy tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). The training program lowercases and tokenizes words plus punctuation. It builds the vocabulary from training text only, retains up to 6,000 frequent tokens, and maps other tokens to `<unk>`. The selected run size is recorded in `results/results.json`.

## Models and method
Each example uses the previous 20 tokens to predict the single next token. Windows are sampled every fifth token to reduce redundant training examples. The selected 30,000-token corpus prefix is split sequentially into 80% train, 10% validation, and 10% test portions. All models use the same 64-dimensional embedding and hidden size, Adam (learning rate 0.001), gradient clipping, batch size 64, eight epochs, and fixed random seed 42. Validation and test accuracy, cross-entropy loss, perplexity, epoch time, total training time, and generated samples are saved as JSON. The BiRNN uses both directions inside the observed 20-token context; its output is evaluated only at the final context position, so it does not see the target token.

## Run
Python 3.10+ is recommended. Install dependencies and train:

```bash
python -m pip install -r requirements.txt
python main.py --epochs 8 --max-tokens 30000 --batch-size 64
```

Set `--max-tokens 0` to use the full corpus. Use `--batch-size` and `--seed` to change the run. Training time depends on hardware. Outputs are written to `results/results.json` and `results/{SimpleRNN,BiRNN,LSTM,GRU}.pt`. Run `python plot_results.py` after training to regenerate `results/metrics_comparison.svg`.

## Results and interpretation
Results from the recorded CPU run (8 epochs; use `results/results.json` for full precision):

| Model | Final validation loss | Validation perplexity | Validation accuracy | Test loss | Test perplexity | Test accuracy | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SimpleRNN | 6.358 | 576.83 | 12.9% | **5.888** | **360.84** | 14.1% | 129.2 |
| BiRNN | 6.451 | 633.44 | 12.8% | 5.988 | 398.43 | 14.4% | 126.2 |
| LSTM | 6.385 | 592.68 | 12.6% | 5.925 | 374.20 | **15.1%** | 94.0 |
| GRU | 6.362 | 579.39 | 11.4% | 5.890 | 361.40 | 14.1% | 62.4 |

The SimpleRNN had the lowest held-out test loss and perplexity, so it is the best choice by the primary probabilistic next-word metrics in this run. LSTM had the highest test accuracy. The differences between SimpleRNN and GRU on test loss/perplexity are very small, and the test accuracy winner differs, so these results do not establish a broadly superior architecture. Sampling used the same three in-vocabulary Shakespeare-style seed phrases at temperatures 0.5, 1.0, and 1.5. For the SimpleRNN, the 0.5 sample showed short loops and repeated words; the 1.0 sample had more varied, somewhat more locally structured dialogue; and 1.5 had greater vocabulary variation but weaker contextual fit. This reflects the expected temperature tradeoff: lower values concentrate probability and tend to repeat, while higher values increase diversity and risk incoherence. See the saved samples for all models and seeds before making qualitative claims about coherence and repetition.

The chart in `results/metrics_comparison.svg` compares test loss, accuracy, and training time; regenerate it with `python plot_results.py` after a new run.

The pre-existing `min_results.json` and `temp_lstm.json` were produced by character-level or partial experiments and are not valid evidence for this word-level assignment; do not cite them as final results. The reported word-level run is deliberately small and uses only the first 30,000 tokens, so treat the comparison as a limited experiment rather than a definitive ranking. `training_seconds` measures summed training epoch time; `end_to_end_seconds` includes evaluation and sample generation.

## Submission checklist
- [x] Run `main.py` and save `results/results.json`, four model checkpoints, and generated samples (embedded in JSON).
- [x] Write the comparison and temperature analysis from those results into this README.
- [ ] Add student name, registration number, repository URL, and dataset details to the required Word file and export it to PDF.
- [x] Upload the assignment source, dataset, results, plots, and model checkpoints to GitHub.
