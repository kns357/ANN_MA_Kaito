### Beyond Correlation: How the Explainability of Artificial Neural Networks Misaligns with Human Cognition:

Betreuung: Lesther Zulauf, Fach: Informatik / Mathematik


Oberthema:  
How decision-making in artificial neural networks can be explained through model instance comparison in FFNs, and what fundamental limitations in explainability arise from architectural differences to human reasoning.

Themeneingrenzung:  
Neural networks, the AI technology behind applications from protein folding to image generation, have become so complex that while we know what they can do - sometimes even outperforming humans - we barely understand how they do it. Their internal decision-making processes remain largely unclear, earning them the title of "black boxes". A clearer view of these networks can improve our understanding of and trust in AI systems.

In my high school thesis (Maturaarbeit), I will compare models of identical feedforward neural network (FFN) architectures and visualise how internally different they can become, despite achieving similar accuracy on the same task. I will investigate why such differences emerge and what makes it so hard to explain the internal decision-making of neural networks. My objective is to analyse and explain:

   - Why different instances of FFNs develop distinct decision pathways and internal   representations despite solving the same task. 
   - How local and global decision making can be explained intuitively.
   - What makes it challenging to explain such decisions in human terms.

I will use optical character recognition (OCR) as the training task, choosing FFNs over convolutional neural networks (CNNs) to avoid the spatial biases from convolutional layers that are already visually interpretable. This allows me to focus on how fully connected networks internally represent and process visual information.


Fragestellung:  
How can relations between activation patterns and input images be made interpretable?
   - What methods most intuitively model decision paths in human terms?
   - Do these explanations account for internal differences across identically trained model instances?
   - What are the limitations in explaining AI decision-making in human terms? How could we improve these limitations?


Vorgehen:  
1. Preparation / Planning  
   a. Study mathematical foundations and related computer vision techniques  
   b. Implement all computational experiments using Python with PyTorch, NumPy, Matplotlib, OpenGL, and other tools for visualisations  
   c. Find optimal FFN architecture for experiments (layers, layer sizes, batch sizes)  
   d. Train and optimise FFNs with identical architectures on binary MNIST classification (e.g., "3" or not), keep the hyperparameters fixed for further experiments  

2. Map functional neurons and activation patterns  
   a. Find relevant relations of neurons to the input using activation maximisation and layer-wise Relevance Propagation (LRP)  
   b. Map activation patterns and dimensionality reduction  
   c. Experiment with input perturbation, importance and function of neurons, and sub-networks  
   d. Cluster neurons by spatial and functional similarity  
   e. Compute metrics for differences and similarities between models  

3. Visualisation / Explanation  
   a. Visualise results to make them intuitive  
   b. Find and visualise decision boundaries  
   c. Find simpler patterns / concepts that define how strongly a neuron activates  
   d. Explain local and global decisions with decision graphs or spatial labelling  

4. Interpretation  
   a. Expand to bigger dataset to test scalability  
   b. Formalise results for analysis and interpretation (e.g., comparison to baseline methods, identified limitations, suggesting improvement in outlook)  
   c. Conclude formal findings in LaTeX and use existing research to support and compare my findings.  


Persönlicher Beitrag:  
- All coding, experimentation, analysis, and interpretation will be developed and implemented by me. 
- I will research existing methods, compare to my methods, and formulate new findings.


Quellen und Material:  
Books / Papers:
- Russell, S. J., & Norvig, P. (2010). Artificial intelligence: A modern approach (3rd ed.). Pearson.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.
- Colin, J., et al. (2022). What I cannot predict, I do not understand: A human-centered evaluation framework for explainability methods. arXiv preprint arXiv:2208.09725
- Wu, M., et al. (2023). VERIX: Towards verified explainability of deep neural networks. arXiv preprint arXiv:2306.09931
- Bach, J. (2019). Phenomenal Experience and the Perceptual Binding State
- Mohan, D. M., et al. (2016). Effect of Subliminal Lexical Priming on the Subjective Perception of Images: A Machine Learning Approach
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?”: Explaining the predictions of any classifier
- Molnar, C. (2019). Interpretable machine learning: A guide for making black box models explainable
- Erhan, D., Courville, A., & Bengio, Y. (2010). Understanding representations learned in deep architectures

Code Examples:
- FFNs: https://github.com/fawazsammani/The-Complete-Neural-Networks-Bootcamp-Theory-Applications/blob/master/MNIST%20Feed%20Forward.ipynb
- CNNs: https://github.com/fawazsammani/The-Complete-Neural-Networks-Bootcamp-Theory-Applications/blob/master/Convolutional%20Networks%20MNIST.ipynb
- 2D visualisation: https://github.com/sayyss/2D-Visualization-of-MNIST-Dataset-/blob/master/2D-visual-MNIST.ipynb
- 3D visualisation: https://github.com/K9Megahertz/VisualNN

Tools:
- Overleaf
- Google Collab
- GitHub
- GPU-accelerated python libraries
