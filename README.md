### Inside the Black Box: Exploring Distinct Decision‑Making in Artificial Neural Networks:

Betreuung: Lesther Zulauf, Fach: Informatik / Mathematik


Oberthema:

Comparative analysis of internal parameters and feature representations across different feedforward neural networks (FFNs) of identical architecture, focusing on explaining distinct decision-making and activation patterns.


Themeneingrenzung:

Neural networks, the AI technology behind applications from protein folding to image generation, have become so complex that while we know what they can do - sometimes even outperforming humans - we barely understand how they do it. Their internal decision-making processes remain largely unclear, earning them the title of "black boxes". A clearer view of these networks can improve our understanding of and trust in AI systems.

In my high school thesis (Maturaarbeit), I will compare models of identical feedforward neural network (FFN) architectures and visualise how internally different they can become, despite achieving similar accuracy on the same task.
Although two FFNs trained under identical conditions can reach comparable performance, their internal configurations (weights, biases, activations) can differ significantly. My objective is to analyse and explain:

- How different instances of FFNs develop distinct decision pathways and internal representations despite solving the same task, focusing on spatial differences such as functionally similar neurons (e.g., edge detectors) appearing in different layers or regions.
- Which neuron functions consistently develop across models and are essential for performance.
- How these successful yet internally different decision pathways can be intuitively explained.

I will use optical character recognition (OCR) as the training task, choosing FFNs over convolutional neural networks (CNNs) to avoid the spatial biases from convolutional layers that are already visually interpretable. This allows me to focus on how fully connected networks internally represent and process visual information. My aim is to analyse and explain the structural differences and consistencies in the weights, activations, and decision-making across different FFN model instances.


Fragestellung:
- How do activation patterns relate to the input images?
- Which of these patterns, functional neurons or sub-networks are essential for task performance, and how do they differ spatially across FFNs with identical architecture trained on the same task?
- How can such differences be used to visually model and intuitively explain the resulting decision paths?
- Can these decision paths be broken down into explainable, algorithmic steps?


Vorgehen:
- Study mathematical foundations and related computer vision techniques
- Implement all computational experiments using Python with PyTorch, NumPy, Matplotlib, OpenGL, and other tools for visualisations
- Find optimal FFN architecture for experiments (layers, layer sizes, batch sizes)
- Train and optimise FFNs with identical architectures on binary MNIST classification (e.g., "3" or not), keep the hyperparameters fixed for further experiments

- Locate neurons detecting specific features in the input using activation maximisation
- Track neuron activations with activation profiles of each neuron
- Apply Layer-wise Relevance Propagation (LRP)
- Compute parameter differences (weights, activations) using cosine similarity and PCA/t-SNE
- Cluster neurons by spatial and functional similarity
- Visualise differences and similarities between models with heatmaps

- Find consistent roles of single neurons or subnetworks (e.g., edge detectors) by comparing activation patterns shared across models
- Test the necessity of patterns in the input by perturbing certain input features (e.g., curves of a "5") and track the resulting performance changes
- Test essential vs. redundant neurons across models

- Expand to multi-class datasets (all digits or other datasets) to test scalability

- Formalise results
- Analyse and interpret these results to offer the clearest possible explanation of the decision pathways in the analysed black box networks


Persönlicher Beitrag:
- All coding, experimentation, analysis, and visualisations will be developed and implemented by me.
- I will code the FFN training, comparison, visualisation, and other experiments in Python.
- I will analyse weight and activation differences and functional neurons.
- I will interpret and explain the decision pathways of my models as clearly as possible.


Quellen und Material:

Books / Papers:
- Russell, S. J., & Norvig, P. (2010). Artificial intelligence: A modern approach (3rd ed.). Pearson.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.
- Bach, J. (2019). Phenomenal Experience and the Perceptual Binding State
- Mohan, D. M., et al. (2016). Effect of Subliminal Lexical Priming on the Subjective Perception of Images: A Machine Learning Approach
- Inverted Qualia: https://plato.stanford.edu/entries/qualia-inverted/
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?”: Explaining the predictions of any classifier
- Molnar, C. (2019). Interpretable machine learning: A guide for making black box models explainable
- Erhan, D., Courville, A., & Bengio, Y. (2010). Understanding representations learned in deep architectures

Code Examples:
- FFNs: https://github.com/fawazsammani/The-Complete-Neural-Networks-Bootcamp-Theory-Applications/blob/master/MNIST%20Feed%20Forward.ipynb
- CNNs: https://github.com/fawazsammani/The-Complete-Neural-Networks-Bootcamp-Theory-Applications/blob/master/Convolutional%20Networks%20MNIST.ipynb
- 2D visualisation: https://github.com/sayyss/2D-Visualization-of-MNIST-Dataset-/blob/master/2D-visual-MNIST.ipynb
- 3D visualisation: https://github.com/K9Megahertz/VisualNN

Tools:
- Google Collab
- GitHub
- GPU-accelerated python libraries 
