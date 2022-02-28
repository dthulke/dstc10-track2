# DSTC 10 Track 2 Code

This repository contains the implementation of the RWTH system for the [DSTC 10 Track 2](https://github.com/alexa/alexa-with-dstc10-track2-dataset) submission.

The repository is structured as follows:
- `/code` contains all our code and is divided as follows:
    - `/datasets` contains dataloaders for [Huggingface datasets](https://github.com/huggingface/datasets) and code for data augmentation
    - `/methods` contains the definitions of our *methods* which define the model and preprocessing that is used
    - `/models` contains implementations of models
- `/setup` contains our [sisyphus](https://github.com/rwth-i6/sisyphus) setup which we used to run our experiments

## Citation

If you use part of this work, please cite [our paper](https://arxiv.org/pdf/2112.08844.pdf):

```
@inproceedings{thulkeDSTC2021,
  title = {{{Adapting Document-Grounded Dialog Systems}} to {{Spoken Conversations}} using {{Data Augmentation}} and a {{Noisy Channel Model}}},
  booktitle = {{{Workshop}} on {{DSTC10}}, {{AAAI}}},
  author = {Thulke, David and Daheim, Nico and Dugast, Christian and Ney, Hermann},
  year = {2022},
}
```