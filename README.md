### Inside the Black Box: Understanding Decision‑Making in Artificial Neural Networks:

Betreuung: Lesther Zulauf, Fach: Informatik / Mathematik


Oberthema:

Comparative analysis of internal parameters and feature representations across different feedforward neural networks (FFNs) of identical architecture, focusing on explaining distinct decision-making and activation patterns.

Themeneingrenzung:

Neural networks, the AI technology behind applications from protein folding to image generation, have become so complex that while we know what they can do - sometimes even outperforming humans - we barely understand how they do it. Their internal decision-making processes remain largely unclear, earning them the title of "black boxes". A clearer view of these networks can improve our understanding of and trust in AI systems.

In my high school thesis (Maturaarbeit), I will compare models of identical feedforward neural network (FFN) architectures and visualise how internally different they can become, despite achieving similar accuracy on the same task.
Although two FFNs trained under identical conditions can reach comparable performance, their internal configurations (weights, biases, activations) can differ significantly. My objective is to analyse and explain:

- How different instances of FFNs develop distinct decision pathways and internal representations despite solving the same task.
- Local and global decision making of a model.
- How these decisions can be visualised intuitively and broken down into interpretable rules.

I will use optical character recognition (OCR) as the training task, choosing FFNs over convolutional neural networks (CNNs) to avoid the spatial biases from convolutional layers that are already visually interpretable. This allows me to focus on how fully connected networks internally represent and process visual information. My aim is to analyse and explain the structural differences and consistencies in the activations and decision-making across different FFN model instances.


Fragestellung:
How can relations between activation patterns and input images be made interpretable?
- What mathematical methods and visualisation techniques most intuitively model decision paths?
- Can these decision paths be broken down into explainable steps?
- Do these explanations account for the internal differences across identically trained models?


Vorgehen:
1. Preparation / Planning  
   a. Study mathematical foundations and related computer vision techniques    
   b. Implement all computational experiments using Python with PyTorch, NumPy, Matplotlib, OpenGL, and other tools for visualisations    
   c. Find optimal FFN architecture for experiments (layers, layer sizes, batch sizes)    
   d. Train and optimise FFNs with identical architectures on binary MNIST classification (e.g., "3" or not), keep the hyperparameters fixed for further experiments    

2. Map functional neurons and activation patterns
   a. Find relevant relations of neurons to the input using activation maximisation and layer-wise Relevance Propagation (LRP)
   b. Map activation patterns with activation profiles and PCA
   c. Apply input perturbation to analyse importance and function of neurons and sub-networks
   d. Cluster neurons by spatial and functional similarity
   e. Compute metrics for differences and similarities between models

4. Visualisation / Explanation
   a. Visualise results to make them visually intuitive
   b. Find and visualise decision boundaries with neural manifolds
   c. Find simpler (e.g., linear) rules that define how strongly a neuron activates
   d. Explain local and global decisions with decision graphs and spatial labelling

5. Interpretation
   a. Expand to multi-class datasets (all digits or other datasets) to test scalability
   b. Formalise results for analysis and interpretation
   c. Offer the clearest possible explanation of the decision pathways in the analysed black box networks


Persönlicher Beitrag:
- All coding, experimentation, analysis, and visualisations will be developed and implemented by me.
- I will code the FFN training, comparison, visualisation, and other experiments in Python.
- I will analyse weight and activation differences and functional neurons.
- I will interpret and explain the decision pathways of my models as clearly as possible.


Quellen und Material:

Books / Papers:
- Russell, S. J., & Norvig, P. (2010). Artificial intelligence: A modern approach (3rd ed.). Pearson.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.
- Colin, J., et al. (2022). What I cannot predict, I do not understand: A human-centered evaluation framework for explainability methods. arXiv preprint arXiv:2208.09725
- Wu, M., et al. (2023). VERIX: Towards verified explainability of deep neural networks. arXiv preprint arXiv:2306.09931
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
