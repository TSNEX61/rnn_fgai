# Word-Level Autoregressive Text Generation (ICT4403 Assignment 2)

## Objective
Compare Simple RNN, bidirectional RNN, LSTM, and GRU models for next-word prediction and autoregressive generation under shared data splits and training settings.

## Dataset
Tiny Shakespeare is a plain-text collection of Shakespeare plays, with dialogue, character names, stage directions, and early-modern English vocabulary. The local source file is `tinyshakespeare.txt`; original source: [Karpathy tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). The training program lowercases and tokenizes words plus punctuation. It builds the vocabulary from training text only, retains up to 6,000 frequent tokens, and maps other tokens to `<unk>`. The recorded run uses a 30,000-token corpus prefix (24,000 train, 3,000 validation, 3,000 test) and has 3,163 vocabulary entries including special tokens. Full details are in `results/results.json`.

## Models and method
Each example uses the previous 20 tokens to predict the single next token. Windows are sampled every fifth token to reduce redundant training examples. The selected 30,000-token corpus prefix is split sequentially into 80% train, 10% validation, and 10% test portions. All models use the same 64-dimensional embedding and hidden size, Adam (learning rate 0.001), gradient clipping, batch size 64, eight epochs, and fixed random seed 42. A Simple RNN uses a basic recurrent state; the BiRNN reads the observed context in both directions; the LSTM uses gated cell memory; and the GRU uses reset and update gates. Validation and test accuracy, cross-entropy loss, perplexity, epoch time, total training time, and generated samples are saved as JSON. The BiRNN's backward direction only reads tokens already present in the 20-token context; prediction is evaluated at the final context position, so it does not see the target token.

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

The SimpleRNN had the lowest held-out test loss and perplexity, so it is selected as the best overall model by the primary probabilistic next-word metrics in this run. LSTM had the highest test accuracy. The differences between SimpleRNN and GRU on test loss/perplexity are very small, and the test accuracy winner differs, so these results do not establish a broadly superior architecture.

### Generated-text review
All models were prompted with the same three in-vocabulary seed phrases at temperatures 0.5, 1.0, and 1.5; the complete outputs are saved in the JSON. In the representative “To be or not to be” samples, SimpleRNN and GRU showed phrase or function-word repetition at 0.5. BiRNN produced some more grammatical short dialogue fragments at 0.5, but repeated conjunctions and pronouns. LSTM's low-temperature output had the most visible punctuation and function-word repetition. At 1.0, SimpleRNN and GRU showed more varied Shakespeare-like words and speaker names, though sentence-level meaning remained inconsistent; BiRNN and LSTM also gained variety but produced awkward transitions. At 1.5, all four models used a broader mix of words, while contextual fit weakened and fragments became less coherent. Overall, diversity rises with temperature and repetition tends to fall, but coherence and contextual consistency also become less reliable at the highest setting. Because this is one short training run, these are qualitative observations from saved samples, not statistical claims.

The SimpleRNN's 1.0 sample offered a reasonable balance of variation and local structure for this run, while its held-out loss and perplexity were also best. This supports choosing SimpleRNN for the assignment, with the caveat that the generated text is still imperfect and the metric differences from GRU are narrow.

The chart in `results/metrics_comparison.svg` compares test loss, accuracy, and training time; regenerate it with `python plot_results.py` after a new run.

The pre-existing `min_results.json` and `temp_lstm.json` were produced by character-level or partial experiments and are not valid evidence for this word-level assignment; do not cite them as final results. The reported word-level run is deliberately small and uses only the first 30,000 tokens, so treat the comparison as a limited experiment rather than a definitive ranking. `training_seconds` measures summed training epoch time; `end_to_end_seconds` includes evaluation and sample generation.

## Submission checklist
- [x] Run `main.py` and save `results/results.json`, four model checkpoints, and generated samples (embedded in JSON).
- [x] Write the comparison and temperature analysis from those results into this README.
- [ ] Add student name, registration number, repository URL, and dataset details to the required Word file and export it to PDF.
- [x] Upload the assignment source, dataset, results, plots, and model checkpoints to GitHub.
