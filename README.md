# Complex-Valued Lightweight Face Recognition

A lightweight face recognition system based on a **complex-valued neural network (CVNN)**. This project investigates whether representing facial information using complex numbers can improve face representation and recognition compared with conventional real-valued neural networks.

## Why Complex-Valued Neural Networks?

Most neural networks use real-valued numbers:

$$x \in \mathbb{R}$$

A complex-valued neural network instead works with:

$$z = x_r + ix_i$$

where $x_r$ is the real component and $x_i$ is the imaginary component.

### 1. Complex Numbers Represent Magnitude and Phase

A complex number can also be written as:

$$z = re^{i\theta}$$

where $r$ represents magnitude and $\theta$ represents phase.

This makes complex-valued representations especially useful for data where phase and magnitude have physical or mathematical meaning, such as signal processing, audio, radar, wireless communication, MRI, electromagnetic waves, and Fourier-domain representations.

A real-valued network must learn relationships between these quantities indirectly, while a complex representation can encode them naturally.

### 2. Real and Imaginary Components Interact

Complex multiplication is:

$$
(a+ib)(c+id) = (ac-bd) + i(ad+bc)
$$

The real and imaginary components are therefore mathematically coupled. In a complex neural network, information from the two components interacts naturally during convolution and linear transformations.

### 3. Complex Representations Can Capture Relationships Naturally

Multiplication by a complex number can simultaneously change magnitude and phase. For example:

$$re^{i\theta} \rightarrow sre^{i(\theta+\phi)}$$

This gives complex-valued models a natural way of representing transformations involving scaling and rotation.

### 4. Potential Representation and Efficiency Advantages

A complex-valued parameter is represented as:

$$W = W_r + iW_i$$

The two components are learned jointly under a complex-valued mathematical structure. Depending on the problem and architecture, this can provide useful expressiveness for a given model size.

However, complex-valued networks do **not automatically use fewer parameters**. A complex weight generally requires two real-valued numbers, so comparisons should use a fair parameter budget.

## Why Use Complex Values for Face Recognition?

A face image is naturally real-valued, so this project constructs a complex representation from two related forms of information:

$$Z = \text{Face Intensity} + i(\text{Edge Information})$$

- **Real component:** normalized grayscale facial image
- **Imaginary component:** horizontal Sobel gradient

This allows the network to process facial appearance and structural information jointly.

```text
                 Face Image
                     |
          +----------+----------+
          |                     |
          v                     v
     Grayscale              Sobel Gradient
          |                     |
          +----------+----------+
                     |
                     v
        Complex Representation
       Real + i × Imaginary
                     |
                     v
          Complex CNN
                     |
                     v
          Complex Face Embedding
                     |
                     v
          Identity / Similarity
```

The hypothesis is that interactions between appearance and structural information through complex-valued operations can produce more discriminative facial embeddings.

## Important: Complex Does Not Mean Automatically Better

A complex-valued neural network is **not inherently superior** to a real-valued neural network.

Its usefulness depends on the problem and on how the complex representation is constructed. For naturally complex or phase-based signals, the advantage can be particularly strong. For ordinary images such as faces, the benefit is an experimental question.

The correct evaluation is therefore a controlled comparison:

```text
           Same Dataset
                |
       +--------+--------+
       |                 |
       v                 v
 Real-Valued NN     Complex-Valued NN
       |                 |
       +--------+--------+
                |
                v
         Compare Results
```

The real-valued baseline should use a comparable architecture, parameter count, dataset split, and training procedure.

## Project Objective

The main objective is to investigate:

> **Can a lightweight complex-valued neural network learn more discriminative facial representations than a comparable real-valued neural network?**

The model is evaluated using face identification accuracy, prototype-based recognition accuracy, face verification ROC-AUC, Equal Error Rate (EER), and model parameter count.

## Model Architecture

The proposed network uses complex-valued components throughout the main representation pipeline.

```text
Complex Input
     |
     v
Complex Convolution
     |
     v
Complex Batch Normalization
     |
     v
Complex ReLU
     |
     v
Complex Residual Blocks
     |
     v
Complex Global Average Pooling
     |
     v
Complex Linear Projection
     |
     v
Complex Face Embedding
     |
     v
Complex ArcFace
```

The complex convolution follows:

$$Z = Z_r+iZ_i$$

$$W = W_r+iW_i$$

and:

$$WZ = (W_rZ_r-W_iZ_i) + i(W_rZ_i+W_iZ_r)$$

Both the weights and biases are represented using real and imaginary components.

## Dataset

This project uses the **Labeled Faces in the Wild (LFW)** dataset.

The current implementation performs a stratified train/validation/test split for experimental evaluation.

## Evaluation Metrics

### Classification Accuracy

Measures how often the model correctly predicts the identity among the known identities.

### Prototype Recognition

The network produces a facial embedding for every image. A prototype is generated for each identity by averaging that person's training embeddings. A test face is identified by finding the most similar prototype.

### ROC-AUC

Measures how well the embedding separates same-person pairs from different-person pairs. Higher is better.

### Equal Error Rate (EER)

EER is the point where the false acceptance rate (FAR) and false rejection rate (FRR) are approximately equal. Lower is better.

## Example Results

An experimental run of the complex-valued model achieved:

| Metric | Result |
|---|---:|
| Best Validation Accuracy | **82.38%** |
| Test Classification Accuracy | **85.90%** |
| Prototype Recognition Accuracy | **93.17%** |
| Verification ROC-AUC | **0.9611** |
| Verification EER | **10.85%** |

These results are from the project's custom LFW train/validation/test evaluation and should not be interpreted as results under the official LFW 10-fold verification protocol.

## Advantages of the Complex-Valued Approach

The potential advantages investigated by this project are:

- **Richer representations:** complex numbers can encode two coupled components and magnitude/phase information.
- **Interaction between components:** real and imaginary features directly interact during complex multiplication.
- **Natural phase representation:** complex representations are well suited to data containing phase or rotational relationships.
- **Joint feature learning:** appearance and structural information can be combined into a single complex representation.
- **Potential efficiency:** for some problems, complex-valued mathematical structure may provide useful expressiveness without simply treating two information streams as unrelated features.

## Limitations

Complex-valued neural networks also introduce challenges:

- More complicated implementation
- Fewer native deep-learning operations for complex tensors
- More difficult initialization and normalization
- Complex-valued activation functions require special consideration
- A complex parameter usually requires two real-valued parameters
- Benefits are highly dependent on the problem and representation

Therefore, complex-valued networks should be evaluated against a strong real-valued baseline rather than assumed to be better.

## Research Question

> **Does introducing complex-valued representations and complex-valued learnable transformations improve face recognition and verification compared with a similarly sized real-valued neural network?**

The final comparison between the two models will determine whether the additional complexity provides a measurable benefit for this task.

## Future Work

- Real-valued baseline with matched parameter count
- Official LFW 10-fold verification protocol
- Better face alignment and preprocessing
- Different methods for constructing the imaginary component
- Complex-valued attention mechanisms
- Comparison with lightweight real-valued face recognition architectures
- Inference speed and memory benchmarking
- Testing on larger face datasets

## Conclusion

Complex-valued neural networks provide an alternative way of representing and processing information using both real and imaginary components.

Their biggest theoretical advantage is the ability to naturally model **magnitude, phase, and interactions between coupled components**.

For face recognition, however, the value of complex numbers is an empirical question rather than an assumption. This project therefore focuses on determining whether complex-valued representations can produce better facial embeddings and recognition performance while remaining lightweight.

> **Complex-valued neural networks are not universally better than real-valued networks — their advantage appears when their mathematical structure matches the problem.**
