During my undergraduate studies, I taught an introductory course on machine-learning security. I cannot publish the original lectures, assignments, or examinations, so I have converted the course's conceptual progression into this public reading path.

This is not meant to be a paper dump. The order is deliberate. The goal is to help a reader move from *understanding how neural networks learn* to *reasoning like an ML-security researcher*: defining an adversary, locating attack surfaces, reproducing attacks and defenses, and looking for assumptions that fail under adaptive evaluation.

The path is centered on the security of learned models, especially:

- model stealing and extraction;
- adversarial examples and evasion attacks;
- training-time poisoning and backdoors;
- neural-network watermarking and ownership resolution;
- backdoor detection and removal;
- explanation-aware attacks; and
- statistical and adversarial evaluation of security claims.

It is **not** a comprehensive curriculum on privacy, federated learning, cryptographic ML, safety alignment, or LLM security. Those are important neighboring areas, but each deserves a separate path.

> **Ethics.** Reproduce attacks only on systems, models, accounts, and data that you own or are explicitly authorized to test. The purpose of this guide is to understand and improve the security of ML systems.

### How to use this guide

Each reading is tagged as one of the following:

- **Core — course:** central to the original course progression.
- **Bridge — added:** added here because a public reading guide must replace explanations that were originally delivered in lectures.
- **Depth — optional:** useful after the core idea is clear, especially when choosing a research direction.

For every paper, do not begin by memorizing the algorithm. First write down:

1. **Asset:** What is being protected?
2. **Adversary's goal:** What counts as a successful attack?
3. **Knowledge and access:** White-box or black-box? Hard labels or confidence scores? Training data, model parameters, or query access?
4. **Capability and budget:** What may the adversary modify, and how many samples, queries, or computations are available?
5. **Defender's knowledge:** What does the defender know when training, detecting, or verifying?
6. **Metrics:** How are attack success, clean utility, cost, false positives, and false negatives measured?
7. **Unstated assumption:** What must remain true for the method to work?

That one-page threat-model summary is often more valuable than several pages of equations copied without context.

---

### The minimal route

Readers who already know deep learning can begin with this compact sequence. It gives the main storyline of the course without every optional branch.

1. [**Stealing Machine Learning Models via Prediction APIs**](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer) — Tramèr et al., USENIX Security 2016.
2. [**Intriguing Properties of Neural Networks**](https://arxiv.org/abs/1312.6199) — Szegedy et al., ICLR 2014.
3. [**Explaining and Harnessing Adversarial Examples**](https://arxiv.org/abs/1412.6572) — Goodfellow, Shlens, and Szegedy, ICLR 2015.
4. [**Practical Black-Box Attacks against Machine Learning**](https://arxiv.org/abs/1602.02697) — Papernot et al., AsiaCCS 2017.
5. [**BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain**](https://arxiv.org/abs/1708.06733) — Gu et al., 2017.
6. [**Spectral Signatures in Backdoor Attacks**](https://proceedings.neurips.cc/paper/2018/hash/280cf18baf4311c92aa5a042336587d3-Abstract.html) — Tran, Li, and Madry, NeurIPS 2018.
7. [**Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks**](https://ieeexplore.ieee.org/document/8835365) — Wang et al., IEEE Symposium on Security and Privacy 2019.
8. [**Embedding Watermarks into Deep Neural Networks**](https://arxiv.org/abs/1701.04082) — Uchida et al., 2017.
9. [**Adversarial Frontier Stitching for Remote Neural Network Watermarking**](https://arxiv.org/abs/1711.01894) — Le Merrer, Pérez, and Trédan, 2017; later published in *Neural Computing and Applications*.
10. [**Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring**](https://www.usenix.org/conference/usenixsecurity18/presentation/adi) — Adi et al., USENIX Security 2018.
11. [**Entangled Watermarks as a Defense against Model Extraction**](https://www.usenix.org/conference/usenixsecurity21/presentation/jia) — Jia et al., USENIX Security 2021.
12. [**SoK: How Robust is Image Classification Deep Neural Network Watermarking?**](https://ieeexplore.ieee.org/document/9833693/) — Lukas et al., IEEE Symposium on Security and Privacy 2022.
13. [**False Claims against Model Ownership Resolution**](https://www.usenix.org/conference/usenixsecurity24/presentation/liu-jian) — Liu et al., USENIX Security 2024.

Read the list in order. The final two papers are especially important: a security mechanism is not understood until you have read a serious attempt to break its assumptions.

---

## Part I — Build the minimum ML foundation

ML security is unusually unforgiving of shallow ML knowledge. Many attacks reverse, repurpose, or interfere with the same gradients, losses, representations, and optimization procedures used during ordinary training. You should be able to follow these mechanisms before treating attacks as recipes.

### 0.1 Mathematics and statistics

You need working familiarity with:

- vectors, matrices, matrix multiplication, norms, and singular-value decomposition;
- probability distributions, conditional probability, expectations, and random variables;
- null and alternative hypotheses, significance levels, p-values, confidence intervals, false positives, and false negatives;
- derivatives, partial derivatives, gradients, and the chain rule; and
- entropy, cross-entropy, and KL divergence.

#### Recommended resources

- **Bridge — added:** [3Blue1Brown, *Essence of Linear Algebra*](https://www.3blue1brown.com/topics/linear-algebra). Use this for geometric intuition.
- **Core — course:** [*Deep Learning*, Chapter 2: Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html), by Goodfellow, Bengio, and Courville.
- **Core — course:** [MIT IAP, *Matrix Calculus for Machine Learning and Beyond*](https://youtube.com/playlist?list=PLUl4u3cNGP62EaLLH92E_VCN4izBKK6OE).
- **Bridge — added:** [Seeing Theory](https://seeing-theory.brown.edu/) for an interactive refresher on probability and statistics.

**Checkpoint.** Given the results of an attack evaluated over several random seeds, explain what the mean, variance, confidence interval, null hypothesis, and practical effect size each tell you. Do not reduce the conclusion to “the p-value is below 0.05.”

### 0.2 Neural networks and optimization

You should understand:

- regression versus classification;
- perceptrons, multilayer perceptrons, activation functions, and logits;
- sigmoid, softmax, and temperature;
- mean-squared error and cross-entropy loss;
- forward propagation, computational graphs, reverse-mode differentiation, and backpropagation;
- full-batch, mini-batch, and stochastic gradient descent;
- learning rates, epochs, iterations, underfitting, overfitting, early stopping, and regularization; and
- accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrices.

#### Recommended resources

- **Core — course:** [3Blue1Brown, *Neural Networks*](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).
- **Core — course:** [Christopher Olah, *Calculus on Computational Graphs: Backpropagation*](https://colah.github.io/posts/2015-08-Backprop/).
- **Core — course:** [Stanford CS231n, *Backpropagation, Intuitions*](https://cs231n.github.io/optimization-2/).
- **Core — course:** [Andrej Karpathy, *The spelled-out intro to neural networks and backpropagation: building micrograd*](https://youtu.be/VMj-3S1tku0).
- **Core — course, hands-on:** [PyTorch regression workflow](https://colab.research.google.com/github/mrdbourke/pytorch-deep-learning/blob/main/01_pytorch_workflow.ipynb) and [PyTorch classification workflow](https://colab.research.google.com/github/mrdbourke/pytorch-deep-learning/blob/main/02_pytorch_classification.ipynb).

**Checkpoint.** Implement a small MLP without relying on a high-level training wrapper. Plot training and validation loss, identify overfitting, and explain how every parameter update follows from the loss and gradient.

### 0.3 Convolutional neural networks

Much of the foundational ML-security literature uses image classifiers. You should understand why convolutions introduce learnable local filters, how feature maps and receptive fields are formed, and how pooling, padding, stride, flattening, and skip connections affect the model.

#### Recommended resources

- **Core — course:** University of Michigan EECS 498, [CNN lecture notes I](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture07.pdf) and [CNN lecture notes II](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture08.pdf).
- **Bridge — added:** [Stanford CS231n, *Convolutional Networks*](https://cs231n.github.io/convolutional-networks/).
- **Depth — optional:** [*Deep Residual Learning for Image Recognition*](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), He et al., CVPR 2016.

**Readiness test.** You are ready to continue when you can train an MLP and a CNN on MNIST or CIFAR-10, explain why the CNN behaves differently, and inspect gradients with respect to both parameters and inputs.

---

## Part II — Learn to model an ML system as an attack surface

A conventional ML workflow includes data collection, preprocessing, model construction, training, evaluation, and deployment. Each stage creates different security assumptions. ML security begins by asking *where the adversary enters the pipeline*.

### 1.1 A broad map of the field

- **Bridge — added:** [**SoK: Security and Privacy in Machine Learning**](https://ieeexplore.ieee.org/document/8406613) — Papernot et al., IEEE European Symposium on Security and Privacy 2018. Use this as a map rather than a paper to memorize.
- **Depth — optional:** [**A Survey of Deep Neural Network Watermarking Techniques**](https://arxiv.org/abs/2103.09274) — Li, Wang, and Barni, 2021. This is useful before the ownership-protection module, but it is denser than the historical route below.

Build your own taxonomy along the following axes:

| Axis | Typical choices |
|---|---|
| Stage | training time, inference time, post-deployment |
| Object attacked | data, parameters, architecture, API, outputs, explanations |
| Access | white-box, grey-box, black-box |
| Output visibility | full confidence vector, top-k scores, hard labels, generated text |
| Goal | indiscriminate degradation, targeted error, hidden behavior, extraction, inference, ownership fraud |
| Constraint | perturbation norm, poison budget, query budget, clean-utility budget, stealth requirement |

Do not conflate the following:

- **Poisoning:** the adversary corrupts the learning process or its data.
- **Backdoor:** a typically targeted poisoning objective that preserves normal behavior but activates malicious behavior under a trigger.
- **Evasion:** the deployed model is attacked through carefully modified test-time inputs.
- **Extraction:** queries to a model are used to reconstruct its parameters, decision boundary, or functionality.
- **Inference attacks:** information about training members, properties, or inputs is inferred from the model.
- **Watermarking or fingerprinting:** evidence is embedded or constructed so that ownership or copying can later be tested.

### 1.2 Threat-model exercise

Choose a public image classifier and write four distinct threat models:

1. a white-box untargeted evasion adversary;
2. a black-box targeted evasion adversary;
3. a training-data supplier planting a backdoor; and
4. an API user attempting functional model extraction.

For each one, specify the success metric and the budget. A threat model without a measurable success condition is not yet an experiment.

---

## Part III — Model stealing and extraction

The course introduces model extraction through a useful historical connection: **knowledge distillation**, originally presented as model compression, already provides the conceptual machinery for transferring behavior from a teacher to a student. The security question is what changes when the teacher is a protected model and the student is trained without authorization.

### 2.1 From compression to extraction

#### 1. Knowledge distillation

- **Core — course:** [**Distilling the Knowledge in a Neural Network**](https://arxiv.org/abs/1503.02531) — Hinton, Vinyals, and Dean, 2015.

Focus on soft targets, temperature, and why a probability vector communicates more information than a hard class label. Then ask the security question: if an API exposes this information, what can an untrusted querier learn?

#### 2. Equation-solving and prediction-API attacks

- **Core — course:** [**Stealing Machine Learning Models via Prediction APIs**](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer) — Tramèr et al., USENIX Security 2016.

This paper is an excellent introduction to concrete threat models. Track the relationship among model family, number of unknown parameters, output precision, query design, and extraction cost. Notice that “stealing” can mean exact parameter recovery in one setting and approximate functional replication in another.

#### 3. Data-free or limited-data black-box extraction

- **Core — course:** [**Practical Black-Box Attacks against Machine Learning**](https://arxiv.org/abs/1602.02697) — Papernot et al., AsiaCCS 2017.

The central connection is between extraction and adversarial transferability. A surrogate need not perfectly reproduce the victim's accuracy; it may only need to approximate the relevant decision boundary well enough for attacks crafted on it to transfer.

#### 4. Later functional-extraction work

- **Depth — optional:** [**Knockoff Nets: Stealing Functionality of Black-Box Models**](https://openaccess.thecvf.com/content_CVPR_2019/html/Orekondy_Knockoff_Nets_Stealing_Functionality_of_Black-Box_Models_CVPR_2019_paper.html) — Orekondy, Schiele, and Fritz, CVPR 2019.
- **Depth — optional, defense:** [**PRADA: Protecting Against DNN Model Stealing Attacks**](https://ieeexplore.ieee.org/document/8806737) — Juuti et al., IEEE EuroS&P 2019.

### 2.2 What to measure

Do not report only the extracted model's test accuracy. At minimum, distinguish:

- **Task accuracy:** performance on the ground-truth task.
- **Fidelity or agreement:** how often the surrogate matches the victim.
- **Query count and monetary cost:** the attack's resource budget.
- **Data assumptions:** in-distribution, out-of-distribution, synthetic, or seed data.
- **Transfer success:** whether attacks created on the surrogate work on the victim.
- **Architecture knowledge:** whether the surrogate matches the target family.

### 2.3 Replication checkpoint

Train a small teacher on MNIST or CIFAR-10 and expose three simulated APIs:

1. full confidence scores;
2. top-1 label only; and
3. rounded or noisy confidence scores.

Train a surrogate under a fixed query budget. Compare accuracy, victim-surrogate agreement, attack transferability, and extraction cost. The result should make clear that “model extraction success” is not a single scalar.

---

## Part IV — Adversarial examples and evasion attacks

Adversarial examples are not just strange images. They are a compact way to study how a learned decision boundary behaves under an adversary who optimizes the input instead of the model parameters.

### 3.1 The foundational progression

#### 1. The phenomenon

- **Core — course:** [**Intriguing Properties of Neural Networks**](https://arxiv.org/abs/1312.6199) — Szegedy et al., ICLR 2014.

Read this as the discovery paper. Identify what the authors observed, how they constrained perturbations, and why generalization across models was surprising.

#### 2. A gradient-based explanation and FGSM

- **Core — course:** [**Explaining and Harnessing Adversarial Examples**](https://arxiv.org/abs/1412.6572) — Goodfellow, Shlens, and Szegedy, ICLR 2015.

Connect the fast gradient sign method to ordinary training. During training, parameters change to reduce loss while the input is fixed. During a white-box evasion attack, parameters are fixed while the input changes to increase a chosen loss.

#### 3. Black-box transfer

- **Core — course:** Revisit [**Practical Black-Box Attacks against Machine Learning**](https://arxiv.org/abs/1602.02697). Read it now from the adversarial-example perspective rather than the extraction perspective.

Transferability is what turns a seemingly unrealistic white-box construction into a practical black-box threat. Record exactly what the attacker must know, what substitute data is needed, and what transfers across models.

#### 4. Robust optimization

- **Bridge — added:** [**Towards Deep Learning Models Resistant to Adversarial Attacks**](https://openreview.net/forum?id=rJzIBfZAb) — Madry et al., ICLR 2018.

This gives a clean min-max formulation and a stronger iterative attack/defense baseline. It is a useful transition from “generate an adversarial example” to “evaluate robustness against an optimizing adversary.”

#### 5. Why adversarial examples may exist

- **Depth — optional:** [**Adversarial Examples Are Not Bugs, They Are Features**](https://proceedings.neurips.cc/paper/2019/hash/e2c420d928d4bf8ce0ff2ec19b371514-Abstract.html) — Ilyas et al., NeurIPS 2019.

This paper shifts the question from imperceptible perturbations alone to the features models actually use.

#### 6. How not to overclaim robustness

- **Depth — strongly recommended for researchers:** [**On Evaluating Adversarial Robustness**](https://arxiv.org/abs/1902.06705) — Carlini et al., 2019.

Treat this as a checklist against weak attacks, gradient masking, inappropriate threat models, and incomplete baselines.

### 3.2 Questions to ask while reading

- Is the attack targeted or untargeted?
- Which norm or perceptual constraint bounds the perturbation?
- Is success measured only on originally correct samples?
- Does the attacker see gradients, scores, labels, or neither?
- Is the defense evaluated with an attack adapted to it?
- Are failed attacks evidence of robustness, or evidence that the optimizer failed?
- Does the attack remain meaningful under physical, semantic, or domain constraints?

### 3.3 Replication checkpoint

Implement FGSM and an iterative projected-gradient attack against a small CNN. Then evaluate:

- clean accuracy;
- robust accuracy versus perturbation budget;
- targeted and untargeted success;
- white-box success;
- cross-architecture transfer; and
- transfer after changing the surrogate's training data.

Plot curves rather than reporting one convenient perturbation value.

---

## Part V — Training-time poisoning and backdoors

A backdoored model can look normal on standard validation and test sets while behaving maliciously on a small trigger-defined region of the input space. This exposes a fundamental limit of ordinary testing: good performance on sampled inputs cannot determine behavior on every untested input.

### 4.1 Foundational attack

- **Core — course:** [**BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain**](https://arxiv.org/abs/1708.06733) — Gu et al., 2017.

Track the attacker's position in the supply chain, the trigger construction, label manipulation, poison rate, clean-data accuracy, and attack success rate. The central security property is conditional behavior: normal inputs follow the intended task, while triggered inputs follow the attacker's task.

### 4.2 Detection and removal

#### Spectral Signatures

- **Core — course:** [**Spectral Signatures in Backdoor Attacks**](https://proceedings.neurips.cc/paper/2018/hash/280cf18baf4311c92aa5a042336587d3-Abstract.html) — Tran, Li, and Madry, NeurIPS 2018.

The paper links poisoned examples to detectable structure in learned representations. Pay attention to why the defense uses internal feature vectors rather than raw high-dimensional inputs, and to the assumptions that make poisoned and clean representations separable.

#### Neural Cleanse

- **Core — course:** [**Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks**](https://ieeexplore.ieee.org/document/8835365) — Wang et al., IEEE Symposium on Security and Privacy 2019.

Understand the reverse-engineering idea: search for a small trigger that makes many samples map to a candidate target class, and flag anomalously small recovered triggers. Then identify which trigger shapes and attack families might violate that assumption.

#### Fine-Pruning

- **Depth — optional:** [**Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks**](https://arxiv.org/abs/1805.12185) — Liu, Dolan-Gavitt, and Garg, RAID 2018.

This is useful because pruning appears elsewhere in the course as model compression, post-theft modification, and watermark removal. The same operation has different security meanings under different threat models.

### 4.3 Evaluation checklist

A backdoor paper should report at least:

- clean accuracy or utility;
- attack success rate on triggered samples;
- poison rate and attacker access;
- trigger visibility, size, location, and variability;
- targeted versus untargeted behavior;
- performance across classes, seeds, datasets, and architectures;
- detector true-positive and false-positive behavior; and
- an adaptive attack that knows the defense.

A detector evaluated only against the exact attack it was designed around is an incomplete security evaluation.

### 4.4 Replication checkpoint

Plant a small patch-trigger backdoor in a toy image classifier. Measure clean accuracy and attack success. Apply one representation-based detector and one model-repair method. Then change the trigger location, shape, poison rate, or representation regularization to test which assumptions the defense relied on.

---

## Part VI — Protecting model ownership with watermarks

Watermarking is presented here as a sequence of design decisions and failures, not as a list of unrelated algorithms. The recurring problem is to construct evidence that survives realistic model modification while remaining specific enough not to implicate independently trained models.

### 5.1 First define what a watermark must achieve

Evaluate every method against these requirements:

1. **Functionality preserving:** embedding the watermark should not materially reduce normal-task utility.
2. **Un-removability or robustness:** an adversary should not remove the watermark without an unacceptable utility loss, including when the scheme is known.
3. **Efficiency:** embedding and verification costs should be practical.
4. **Generality:** the approach should not depend unnecessarily on one architecture, task, or modality.
5. **Non-ownership piracy:** an adversary should not be able to create competing ownership evidence that casts doubt on the legitimate owner.
6. **Non-trivial ownership:** verification should have low false-positive and false-negative rates; an independent model should not satisfy the claim merely by chance.

Add two parties that are often neglected:

- a **malicious suspect**, who has copied and then modified a model; and
- a **malicious accuser**, who attempts to claim an independently trained model.

These are different threat models and demand different experiments.

### 5.2 White-box parameter watermarking

- **Core — course:** [**Embedding Watermarks into Deep Neural Networks**](https://arxiv.org/abs/1701.04082) — Uchida et al., 2017.

This is a clean starting point for static or white-box watermarking. Follow how a bit string is related to selected model weights through an additional training loss. Then study the limitations: verification requires parameter access; fine-tuning, pruning, overwriting, extraction, or statistical detection may threaten the signal.

**Checkpoint.** Be able to explain why directly overwriting a set of weights is not enough. A usable watermark must be trained into the model in a way that balances payload, task accuracy, and robustness.

### 5.3 Black-box behavioral watermarking

#### Frontier Stitching

- **Core — course:** [**Adversarial Frontier Stitching for Remote Neural Network Watermarking**](https://arxiv.org/abs/1711.01894) — Le Merrer, Pérez, and Trédan, 2017.

The trigger set is placed near the model's decision boundary using adversarial examples, allowing remote verification through queries. Focus on why ordinary training samples would not identify a particular model and why boundary stability becomes an assumption.

#### Backdoors as watermarks

- **Core — course:** [**Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring**](https://www.usenix.org/conference/usenixsecurity18/presentation/adi) — Adi et al., USENIX Security 2018.
- **Core — course:** [**Protecting Intellectual Property of Deep Neural Networks with Watermarking**](https://dl.acm.org/doi/10.1145/3196494.3196550) — Zhang et al., AsiaCCS 2018.

These works connect ownership verification to hidden input-output behavior. Compare how the trigger set is chosen, how it is embedded, what the verifier observes, and whether the watermark distribution resembles the task distribution.

### 5.4 Surviving model extraction

A basic backdoor watermark may fail to transfer during extraction when the thief trains only on task-distribution queries. The extracted model learns ordinary task behavior but never receives the rare watermark inputs.

#### Entangled Watermarking Embeddings

- **Core — course:** [**Entangled Watermarks as a Defense against Model Extraction**](https://www.usenix.org/conference/usenixsecurity21/presentation/jia) — Jia et al., USENIX Security 2021.

Study how the method tries to entangle task and watermark representations so that removing or failing to transfer the watermark also damages useful behavior. This paper is valuable even beyond watermarking because it illustrates a general defense principle: make the protected property inseparable from the functionality the attacker wants.

#### Dynamic watermarking of prediction APIs

- **Depth — optional:** [**DAWN: Dynamic Adversarial Watermarking of Neural Networks**](https://dl.acm.org/doi/10.1145/3474085.3475591) — Szyller et al., ACM Multimedia 2021.

DAWN moves part of the mechanism to the prediction API. Compare its defender capabilities and operational costs with watermarks embedded once during training.

#### More recent extraction-aware schemes

- **Depth — optional:** [**Margin-based Neural Network Watermarking**](https://proceedings.mlr.press/v202/kim23o.html) — Kim et al., ICML 2023.
- **Core — course:** [**A Robust Watermark against Model Extraction Attack (MEA-Defender)**](https://ieeexplore.ieee.org/document/10646835/) — Lv et al., IEEE Symposium on Security and Privacy 2024.
- **Depth — course extension to LLMs:** [**ModelShield: Adaptive and Robust Watermark against Model Extraction Attack**](https://arxiv.org/abs/2405.02365) — Pang et al.; later published in *IEEE Transactions on Information Forensics and Security*.

Do not treat later publication date as proof of stronger security. Compare threat models, access assumptions, adaptive attacks, and costs directly.

### 5.5 Read the attacks on watermarking

#### Systematic robustness evaluation

- **Core — course:** [**SoK: How Robust is Image Classification Deep Neural Network Watermarking?**](https://ieeexplore.ieee.org/document/9833693/) — Lukas et al., IEEE Symposium on Security and Privacy 2022.

Read this before designing a new watermark. It demonstrates why robustness must be tested against adaptive removal, not only a small menu of benign transformations.

#### False ownership claims

- **Core — course:** [**False Claims against Model Ownership Resolution**](https://www.usenix.org/conference/usenixsecurity24/presentation/liu-jian) — Liu et al., USENIX Security 2024.

This paper changes the viewpoint from “Can a thief erase my claim?” to “Can someone manufacture a convincing claim against an independent model?” Pay close attention to transferable adversarial examples, claim registration or timestamping, the judge's role, and the distinction between anteriority and validity.

### 5.6 Thresholds are part of the security mechanism

A verifier eventually converts evidence into a decision. “The watermark accuracy looks high” is not a complete rule. A threshold may be chosen empirically, theoretically, or statistically, but the procedure should specify:

- the null hypothesis;
- clean or independently trained reference models;
- the expected distribution of verification scores;
- false-positive and false-negative costs;
- how many trigger queries are made;
- repeated-testing or multiple-claim effects; and
- whether the threshold was selected before evaluating the suspect.

Do not tune a threshold on the same suspect model against which ownership is being asserted.

### 5.7 Capstone replication

Implement one white-box or black-box watermark on a small classifier. Evaluate:

- task utility before and after embedding;
- verification score and decision threshold;
- independently trained clean models;
- fine-tuning;
- last-layer and all-layer retraining;
- pruning;
- quantization;
- knowledge distillation or extraction;
- watermark overwriting; and
- a false-claim attempt.

The capstone is complete only when the report separates **watermark survival**, **model utility**, **false-positive risk**, and **attacker cost**.

---

## Part VII — Explainability as both a tool and an attack surface

Feature-attribution methods try to identify which input features most influenced a prediction. They can help inspect suspicious behavior, but an explanation is not automatically a security guarantee. An adaptive attacker may optimize both the prediction and the explanation.

### 6.1 A compact attribution reading path

1. **Core — course:** [**Visualizing and Understanding Convolutional Networks**](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53) — Zeiler and Fergus, ECCV 2014. Read the occlusion analysis.
2. **Core — course:** [**“Why Should I Trust You?”: Explaining the Predictions of Any Classifier**](https://dl.acm.org/doi/10.1145/2939672.2939778) — Ribeiro, Singh, and Guestrin, KDD 2016. Understand local surrogate explanations and their computational and fidelity assumptions.
3. **Core — course:** [**A Unified Approach to Interpreting Model Predictions**](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) — Lundberg and Lee, NeurIPS 2017. Focus on the Shapley-value framing and computational trade-offs.
4. **Core — course:** [**Striving for Simplicity: The All Convolutional Net**](https://arxiv.org/abs/1412.6806) — Springenberg et al., 2015. Read the guided-backpropagation component.
5. **Core — course:** [**Learning Deep Features for Discriminative Localization**](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html) — Zhou et al., CVPR 2016. This introduces class activation maps for a restricted architecture family.
6. **Core — course:** [**Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html) — Selvaraju et al., ICCV 2017. Compare it directly with CAM and guided backpropagation.

For every method, ask whether it is class-discriminative, model-agnostic, faithful, stable, computationally practical, and robust to an adversary.

### 6.2 Explanation-aware attacks

- **Core — course:** [**Disguising Attacks with Explanation-Aware Backdoors**](https://ieeexplore.ieee.org/document/10179308/) — Noppel, Peter, and Wressnegger, IEEE Symposium on Security and Privacy 2023.

This is the key security lesson of the module. A defender may expect a backdoor to produce visibly abnormal explanations. Once the attacker knows this detection rule, the attack objective can include explanation similarity as well as malicious prediction behavior.

### 6.3 Replication checkpoint

For a clean and backdoored classifier, compare occlusion, Grad-CAM, and one surrogate-based attribution method on clean and triggered inputs. Then modify the attack objective to penalize explanation differences. Report whether the detector still works and what utility or attack-success trade-off the adaptive attacker pays.

---

## Part VIII — How to turn reading into research

Reading papers is necessary, but it is not the same as learning to conduct ML-security research. Use the following workflow for each topic.

### 7.1 The one-page paper record

For every paper, write one page with these headings:

#### Claim

What does the paper claim to make possible, prevent, detect, or prove?

#### Threat model

Who is the adversary? What can they observe, modify, query, or compute? What is explicitly out of scope?

#### Mechanism

Explain the key idea in your own words, then write the minimum equations needed to make the explanation precise.

#### Evidence

Which datasets, models, baselines, metrics, budgets, seeds, and statistical tests support the claim?

#### Strongest assumption

What environmental, distributional, architectural, or behavioral condition must hold?

#### Adaptive response

How would an attacker change after learning the defense?

#### Reproduction risk

Which implementation detail is most likely to change the conclusion?

#### Next experiment

State one experiment that could falsify or materially narrow the claim.

### 7.2 Reproduction before novelty

A good entry point into research is a faithful reproduction followed by one controlled stress test. A practical sequence is:

1. reproduce the headline result;
2. match the original threat model and metrics;
3. run multiple seeds and report uncertainty;
4. vary one assumption at a time;
5. add an adaptive baseline;
6. document failures and implementation ambiguities; and
7. only then propose a new mechanism.

Negative results are valuable when they isolate *why* an assumption fails.

### 7.3 Experimental standards

For attacks and defenses, report both security and utility. Depending on the topic, this may include:

- clean task performance;
- attack success or robust accuracy;
- fidelity and extraction cost;
- poison and query budgets;
- runtime and memory;
- false-positive and false-negative rates;
- confidence intervals across seeds;
- architecture and dataset transfer;
- ablations of each objective term;
- sensitivity to hyperparameters and thresholds; and
- adaptive attacks with full knowledge of the defense, unless the stated threat model excludes that knowledge.

Avoid selecting only the seed, attack strength, model, or threshold that supports the desired conclusion.

### 7.4 A research-question generator

When a paper feels “complete,” try one of these transformations:

- **Relax an assumption.** Remove access to training data, confidence scores, gradients, architecture, or a trusted judge.
- **Strengthen the adversary.** Let the attacker know the detector, watermark algorithm, trigger distribution, or evaluation threshold.
- **Change the distribution.** Test extraction, triggers, or detectors under domain shift and out-of-distribution queries.
- **Change the modality or structure.** Move from images to text, audio, graphs, temporal data, recommendation, or multimodal models.
- **Change the objective.** Replace average accuracy with targeted harm, subgroup harm, stealth, cost, or persistence.
- **Attack the evaluation.** Test threshold calibration, repeated claims, reference-model selection, or multiple hypothesis testing.
- **Reverse the parties.** Replace the malicious suspect with a malicious accuser, verifier, data supplier, or API owner.
- **Compose mechanisms.** Combine extraction with evasion, backdoors with explanations, or watermark removal with model compression.
- **Study operational cost.** Ask whether the defense remains useful under realistic latency, query, memory, or retraining constraints.

The most promising research ideas often appear at the boundary between two modules rather than inside a single paper family.

### 7.5 Read papers directly

Paper summaries, blog posts, and language models can help locate terminology or unblock a difficult derivation. They should not replace reading the paper, inspecting its tables, checking its threat model, and studying the experimental setup. Preprints are not automatically wrong and peer-reviewed papers are not automatically correct; the standard is whether the evidence supports the claim.

---

## A suggested 12-week route

This schedule assumes the reader already knows basic programming and can spend several focused hours per week. Slow it down when reproducing results; speed is less valuable than understanding the threat model.

| Week | Topic | Primary output |
|---|---|---|
| 1 | Statistics, gradients, losses, and CNN refresh | Train and diagnose a small classifier |
| 2 | ML pipeline and threat modeling | Four written threat models for one system |
| 3 | Knowledge distillation and prediction-API extraction | Teacher–student baseline |
| 4 | Practical black-box extraction and transfer | Query-budget versus fidelity experiment |
| 5 | Szegedy, FGSM, transferability | White-box and transferred evasion results |
| 6 | PGD and robustness evaluation | Robust-accuracy curves and attack audit |
| 7 | BadNets | Backdoored classifier with clean/trigger metrics |
| 8 | Spectral Signatures, Neural Cleanse, Fine-Pruning | Defense comparison and adaptive variant |
| 9 | Uchida and Frontier Stitching | One static or boundary-based watermark |
| 10 | Backdoor watermarks and EWE | Extraction/removal stress test |
| 11 | SoK robustness, thresholds, and False Claims | Ownership-verification audit |
| 12 | XAI and explanation-aware backdoors | Mini research proposal with one falsifiable hypothesis |

---

## When are you ready to begin a research project?

You do not need to have read every paper in this guide. You are ready when you can do the following for one narrow topic:

- state the threat model without vague phrases such as “the attacker has limited access”;
- reproduce at least one credible baseline;
- separate attack success from clean utility and cost;
- identify an assumption that the original evaluation did not stress;
- implement or design an adaptive adversary;
- justify metrics and thresholds statistically; and
- describe a result that would falsify your hypothesis.

At that point, stop expanding the reading list indiscriminately. Build the smallest experiment that can teach you whether the research question is real.

---

## Reference map

### Foundations

- [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- [Goodfellow, Bengio, and Courville — *Deep Learning*](https://www.deeplearningbook.org/)
- [3Blue1Brown — Neural Networks](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Christopher Olah — Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)
- [Stanford CS231n — Backpropagation](https://cs231n.github.io/optimization-2/)
- [Karpathy — micrograd](https://youtu.be/VMj-3S1tku0)

### Extraction

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Stealing Machine Learning Models via Prediction APIs](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer)
- [Practical Black-Box Attacks against Machine Learning](https://arxiv.org/abs/1602.02697)
- [Knockoff Nets](https://openaccess.thecvf.com/content_CVPR_2019/html/Orekondy_Knockoff_Nets_Stealing_Functionality_of_Black-Box_Models_CVPR_2019_paper.html)
- [PRADA](https://ieeexplore.ieee.org/document/8806737)

### Adversarial examples

- [Intriguing Properties of Neural Networks](https://arxiv.org/abs/1312.6199)
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://openreview.net/forum?id=rJzIBfZAb)
- [Adversarial Examples Are Not Bugs, They Are Features](https://proceedings.neurips.cc/paper/2019/hash/e2c420d928d4bf8ce0ff2ec19b371514-Abstract.html)
- [On Evaluating Adversarial Robustness](https://arxiv.org/abs/1902.06705)

### Backdoors

- [BadNets](https://arxiv.org/abs/1708.06733)
- [Spectral Signatures in Backdoor Attacks](https://proceedings.neurips.cc/paper/2018/hash/280cf18baf4311c92aa5a042336587d3-Abstract.html)
- [Neural Cleanse](https://ieeexplore.ieee.org/document/8835365)
- [Fine-Pruning](https://arxiv.org/abs/1805.12185)

### Watermarking and ownership

- [A Survey of Deep Neural Network Watermarking Techniques](https://arxiv.org/abs/2103.09274)
- [Embedding Watermarks into Deep Neural Networks](https://arxiv.org/abs/1701.04082)
- [Adversarial Frontier Stitching](https://arxiv.org/abs/1711.01894)
- [Turning Your Weakness Into a Strength](https://www.usenix.org/conference/usenixsecurity18/presentation/adi)
- [Protecting Intellectual Property of Deep Neural Networks with Watermarking](https://dl.acm.org/doi/10.1145/3196494.3196550)
- [Entangled Watermarks as a Defense against Model Extraction](https://www.usenix.org/conference/usenixsecurity21/presentation/jia)
- [DAWN](https://dl.acm.org/doi/10.1145/3474085.3475591)
- [SoK: How Robust is Image Classification DNN Watermarking?](https://ieeexplore.ieee.org/document/9833693/)
- [Margin-based Neural Network Watermarking](https://proceedings.mlr.press/v202/kim23o.html)
- [MEA-Defender](https://ieeexplore.ieee.org/document/10646835/)
- [False Claims against Model Ownership Resolution](https://www.usenix.org/conference/usenixsecurity24/presentation/liu-jian)
- [ModelShield](https://arxiv.org/abs/2405.02365)

### Explainability and adaptive attacks

- [Visualizing and Understanding Convolutional Networks](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53)
- [LIME](https://dl.acm.org/doi/10.1145/2939672.2939778)
- [SHAP](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)
- [Class Activation Maps](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html)
- [Grad-CAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)
- [Disguising Attacks with Explanation-Aware Backdoors](https://ieeexplore.ieee.org/document/10179308/)

---

*This page is a self-study reading guide, not a reproduction of the original course materials or assessments.*
