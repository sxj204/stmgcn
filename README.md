# Identifying spatial domains of spatially resolved transcriptomics via multi-view graph convolutional networks
===
Overview
===
STMGCN is a novel deep computational framework that effectively integrates spatial coordinates and gene expression information for ST data analysis. As shown in Fig. \ref{fig1}, STMGCN first constructs two distinct adjacency matrices using spatial coordinates that describe the underlying graph structure from different perspectives, and encode gene expression information into these matrices. Subsequently, we perform graph convolution on each adjacency matrix and gene expression count matrix to learn two latent embeddings i.e. ${\textbf{Z}^{^{(1)}}}$, ${\textbf{Z}^{^{(2)}}}$. Next, we use an attention mechanism to adaptively fuse these embeddings into a unified embedding Z based on the importance of the respective embeddings. Finally, we leverage a self-supervised learning strategy based on KL divergence to cluster spots into cohesive groups, which facilitates downstream analysis tasks. We will introduce each part of model in the following sections.
Requirements
===
torch 1.11.0
scanpy 1.9.1
pandas 1.3.5
numpy 1.21.5
scipy 1.7.3

