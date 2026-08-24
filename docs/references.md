# Method and source attribution

## Runtime activation steering

The intervention `h := h + alpha * direction` builds on the activation-engineering and Contrastive Activation Addition literature:

- Turner et al., *Steering Language Models With Activation Engineering*: https://arxiv.org/abs/2308.10248v5
- Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*: https://aclanthology.org/2024.acl-long.828/
- Li et al., *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*: https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html
- Tan et al., *Analysing the Generalisation and Reliability of Steering Vectors*: https://proceedings.neurips.cc/paper_files/paper/2024/hash/fb3ad59a84799bfb8d700e56d19c231b-Abstract-Conference.html
- Lee et al., *Programming Refusal with Conditional Activation Steering*: https://openreview.net/forum?id=Oi47wc10sm
- Sankaranarayanan et al., *Activation Steering via Generative Causal Mediation* (preprint): https://arxiv.org/abs/2602.16080v2
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*: https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html

The model weights remain unchanged. During a multi-token prefill, the hook changes only the final non-padding prompt position; during cached autoregressive decoding, it changes the current token position at every decoder call. The reliability, conditional-steering, and causal-mediation work above motivates evaluating both target behavior and off-target effects rather than treating probe separability as sufficient evidence of useful control.

## Sycophancy data and evaluation framing

- Sharma et al., *Towards Understanding Sycophancy in Language Models*: https://openreview.net/forum?id=tvhaxkMKAn
- Sinha, *SycoBench-600: Measuring Sycophancy and Correction Selectivity in LLM Assistants*: https://aclanthology.org/2026.findings-acl.1759/
- Vennemeyer et al., *Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs* (preprint): https://arxiv.org/abs/2509.21305v3
- Chen et al., *Persona Vectors: Monitoring and Controlling Character Traits in Language Models* (preprint): https://arxiv.org/abs/2507.21509v3
- Yan et al., *RefuteBench: Evaluating Refuting Instruction-Following for Large Language Models*: https://aclanthology.org/2024.findings-acl.818/
- Hong et al., *Measuring Sycophancy of Language Models in Multi-turn Dialogues*: https://aclanthology.org/2025.findings-emnlp.121/
- Buchan, *Dual-Stance Evaluation of Sycophancy: The Structure of Agreement and the Limits of Intervention* (TAIS 2026 accepted preprint): https://arxiv.org/abs/2606.11205
- Genadi et al., *Sycophancy Hides Linearly in the Attention Heads*: https://aclanthology.org/2026.eacl-long.324/
- Kelkar et al., *Playing Devil's Advocate: Off-the-Shelf Persona Vectors Rival Targeted Steering for Sycophancy* (ICML 2026 workshop preprint): https://arxiv.org/abs/2605.21006v2
- Nguyen et al., *Token-Level Diagnosis of Sycophancy in LLMs with Attribution-Guided Steering* (preprint): https://arxiv.org/abs/2607.28906
- Wang et al., *When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models* (AAAI 2026): https://ojs.aaai.org/index.php/AAAI/article/view/40645

The factual items in this repository's local materialized splits derive from SycophancyEval. The upstream repository did not expose an explicit redistribution license at freeze time; source-derived records are therefore regenerated locally and are not committed here.

Against this literature, the contribution studied here is the joint resistance--correct-update--distribution-shift--capability frontier under one harmonized protocol across two model families and two parameter scales. The repeated within-family size ordering is reported as an empirical pattern and motivates a broader matched-dose scaling study.

## Evaluation datasets and divergence measures

- Lin et al., *TruthfulQA: Measuring How Models Mimic Human Falsehoods*: https://aclanthology.org/2022.acl-long.229/
- Joshi et al., *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension*: https://aclanthology.org/P17-1147/
- Cobbe et al., *Training Verifiers to Solve Math Word Problems* (GSM8K): https://arxiv.org/abs/2110.14168v2
- Biderman et al., *Lessons from the Trenches on Reproducible Evaluation of Language Models* (lm-evaluation-harness): https://arxiv.org/abs/2405.14782
- Kojima et al., *Large Language Models Are Zero-Shot Reasoners*: https://arxiv.org/abs/2205.11916
- Merity et al., *Pointer Sentinel Mixture Models* (WikiText-2 source): https://openreview.net/forum?id=Byj72udxe
- Kullback and Leibler, *On Information and Sufficiency*: https://doi.org/10.1214/aoms/1177729694
- Lin, *Divergence Measures Based on the Shannon Entropy*: https://doi.org/10.1109/18.61115

## Model and quantization sources

- Qwen Team, *Qwen3.5: Towards Native Multimodal Agents*: https://qwen.ai/blog?id=qwen3.5
- Gemma Team, *Gemma 4 Technical Report*: https://arxiv.org/abs/2607.02770
- Dettmers et al., *QLoRA* (NF4 quantization provenance): https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html

The E4B arm uses NF4 weights with bf16 compute because the bf16 checkpoint exceeds the available 12GB VRAM. Within-model base-versus-steered effects remain paired at the same precision; cross-model absolute comparisons are secondary.

## Software provenance

Parts of the implementation and research scaffolding derive from Heretic v1.4.0 by Philipp Emanuel Weidmann and contributors: https://github.com/p-e-w/heretic

The repository remains AGPL-3.0-or-later and retains upstream attribution. Complete academic entries are in `paper/references.bib`.
