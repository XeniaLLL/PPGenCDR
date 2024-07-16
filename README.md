# PPGenCDR: A Stable and Robust Framework for Privacy-Preserving Cross-Domain Recommendation
This repository is an official PyTorch implementation of paper:
[PPGenCDR: A Stable and Robust Framework for Privacy-Preserving Cross-Domain Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/25566).
AAAI 2023 (Oral).



## Abstract
Privacy-preserving cross-domain recommendation (PPCDR) refers to preserving the privacy of users when transferring the knowledge from source domain to target domain for better performance, which is vital for the long-term development of recommender systems. Existing work on cross-domain recommendation (CDR) reaches advanced and satisfying recommendation performance, but mostly neglects preserving privacy. To fill this gap, we propose a privacy-preserving generative cross-domain recommendation (PPGenCDR) framework for PPCDR. PPGenCDR includes two main modules, i.e., stable privacy-preserving generator module, and robust cross-domain recommendation module. Specifically, the former isolates data from different domains with a generative adversarial network (GAN) based model, which stably estimates the distribution of private data in the source domain with ́Renyi differential privacy (RDP) technique. Then the latter aims to robustly leverage the perturbed but effective knowledge from the source domain with the raw data in target domain to improve recommendation performance. Three key modules, i.e., (1) selective privacy preserver, (2) GAN stabilizer, and (3) robustness conductor, guarantee the cost-effective trade-off between utility and privacy, the stability of GAN when using RDP, and the robustness of leveraging transferable knowledge accordingly. The extensive empirical studies on Douban and Amazon datasets demonstrate that PPGenCDR significantly outperforms the state-of-the-art recommendation models while preserving privacy.

## Implementation 
###Step 1: Download process data from 
1. A processed data example from [here](https://drive.google.com/file/d/139TbpfcaUs7A5MbqFt1IzYmuayJWK3cv/view). 
And rename source and target datasets.
2. run `/data/parse_amazon_raw.py` for preprocessing.

###Step 2: train model 
HyperFed is hyperbolic prototype based federated learning method.\
MGDA is the consistent updating enhanced hyperbolic prototype based federated learning method.
```
# execute RecAgent in agent
python cdr_wgan_dmf_agent_w_s_prop_sml_pid_sweep.py
```


## Citation
If you find HyperFed useful or inspiring, please consider citing our paper:
```bibtex
@inproceedings{liao2023ppgencdr,
  title={Ppgencdr: A stable and robust framework for privacy-preserving cross-domain recommendation},
  author={Liao, Xinting and Liu, Weiming and Zheng, Xiaolin and Yao, Binhui and Chen, Chaochao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={4},
  pages={4453--4461},
  year={2023}
}
```

