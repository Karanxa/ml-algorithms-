```mermaid
graph TB
    %% Root
    ML[Machine Learning]

    %% Subgraphs for vertical stacking
    subgraph SUP[Supervised Learning]
        SL_CLASS[Classification]
        SL_REG[Regression]

        SL_CLASS --> LC[Linear Classifiers]
        SL_CLASS --> NLC[Non-Linear Classifiers]
        SL_CLASS --> PC[Probabilistic Classifiers]
        SL_CLASS --> IB[Instance-Based]
        SL_CLASS --> ENS_CLAS[Ensemble Methods]

        LC --> LR[Logistic Regression]
        LC --> LDA[Linear Discriminant Analysis]
        LC --> QDA[Quadratic Discriminant Analysis]
        LC --> PR[Perceptron]
        LC --> SVM_L[Linear SVM]

        NLC --> SVM_NL[Non-linear SVM RBF / Poly]
        NLC --> DT[Decision Tree]
        NLC --> RF[Random Forest]
        NLC --> GB[Gradient Boosting: XGBoost / LGBM / CatBoost]
        NLC --> AB[AdaBoost]
        NLC --> ET[Extra Trees]

        PC --> NB[Naive Bayes Gaussian / Multinomial / Bernoulli]
        PC --> BN[Bayesian Networks]
        PC --> HMM[Hidden Markov Model]
        PC --> GMM[Gaussian Mixture Model]

        IB --> KNN[K-Nearest Neighbor]
        IB --> KNC[K-Nearest Centroid]
        IB --> LWL[Locally Weighted Learning]
        IB --> RN[Radius Neighbors]

        ENS_CLAS --> BAG[Bagging]
        ENS_CLAS --> BOOST[Boosting]
        ENS_CLAS --> VOTE[Voting Classifier]
        ENS_CLAS --> STACK[Stacking Classifier]

        SL_REG --> LIN_REG[Linear Regression]
        SL_REG --> RIDGE[Ridge Regression]
        SL_REG --> LASSO[Lasso Regression]
        SL_REG --> ELNET[Elastic Net]
        SL_REG --> POLY[Polynomial Regression]
        SL_REG --> STEPWISE[Stepwise Regression]
        SL_REG --> SVR[Support Vector Regression]
        SL_REG --> DTR[Decision Tree Regression]
        SL_REG --> RFR[Random Forest Regression]
        SL_REG --> GBR[Gradient Boosting Regression]
        SL_REG --> GPR[Gaussian Process Regression]
        SL_REG --> NNR[Neural Network Regression]
        SL_REG --> LARS[Least Angle Regression]
        SL_REG --> OMP[Orthogonal Matching Pursuit]
    end

    subgraph UNSUP[Unsupervised Learning]
        UL_CLUSTER[Clustering]
        UL_DIMRED[Dimensionality Reduction]
        UL_ASSOC[Association Rules]
        UL_ANOMALY[Anomaly Detection]

        UL_CLUSTER --> CENTR[Centroid-Based]
        UL_CLUSTER --> HIER[Hierarchical]
        UL_CLUSTER --> DENS[Density-Based]
        UL_CLUSTER --> DISTR[Distribution-Based]
        UL_CLUSTER --> GRAPH[Graph-Based]

        CENTR --> KMEANS[K-Means]
        CENTR --> KMEDOIDS[K-Medoids]
        CENTR --> FCM[Fuzzy C-Means]
        CENTR --> MBK[Mini-Batch K-Means]

        HIER --> AGGL[Hierarchical Agglomerative]
        HIER --> DIV[Divisive Clustering]
        HIER --> BIRCH[BIRCH]

        DENS --> DBSCAN[DBSCAN]
        DENS --> OPTICS[OPTICS]
        DENS --> MEANSHIFT[Mean Shift]
        DENS --> HDBSCAN[HDBSCAN]

        DISTR --> GMMU[GMM Clustering]
        DISTR --> EM[Expectation-Maximization]

        GRAPH --> SPECTRAL[Spectral Clustering]
        GRAPH --> AFFINITY[Affinity Propagation]

        UL_DIMRED --> LDM[Linear Methods]
        UL_DIMRED --> NLM[Nonlinear Methods]

        LDM --> PCA[Principal Component Analysis]
        LDM --> FA[Factor Analysis]
        LDM --> ICA[Independent Component Analysis]
        LDM --> LDA_DR[Linear Discriminant Analysis]
        LDM --> NMF[Non-negative Matrix Factorization]

        NLM --> TSNE[t-SNE]
        NLM --> UMAP[UMAP]
        NLM --> ISOMAP[Isomap]
        NLM --> LLE[Locally Linear Embedding]
        NLM --> MDS[Multidimensional Scaling]
        NLM --> KPCA[Kernel PCA]
        NLM --> AE_DIM[Autoencoder DR]

        UL_ASSOC --> APRIORI[Apriori]
        UL_ASSOC --> FPG[FP-Growth]
        UL_ASSOC --> ECLAT[Eclat]

        UL_ANOMALY --> ISOF[Isolation Forest]
        UL_ANOMALY --> OCSVM[One-Class SVM]
        UL_ANOMALY --> LOF[Local Outlier Factor]
        UL_ANOMALY --> EENV[Elliptic Envelope]
        UL_ANOMALY --> DBSCAN_ANOM[DBSCAN Outlier]
        UL_ANOMALY --> AE_ANOM[Autoencoder Anomaly]
        UL_ANOMALY --> STAT_ANOM[Z-Score / IQR]
    end

    subgraph SEMI[Semi-Supervised Learning]
        LABELPROP[Label Propagation]
        LABELSPREAD[Label Spreading]
        SELFTRAIN[Self-training]
        COTRAIN[Co-training]
        MULTIVIEW[Multi-View Learning]
        GSSL[Graph-based SSL]
    end

    subgraph RL[Reinforcement Learning]
        RL_VAL[Value-Based RL]
        RL_POL[Policy-Based RL]
        RL_ACT[Actor-Critic RL]
        RL_MB[Model-Based RL]

        RL_VAL --> QL[Q-Learning]
        RL_VAL --> SARSA[SARSA]
        RL_VAL --> EQSARSA[Expected SARSA]
        RL_VAL --> DQL[Deep-Q-Learning]
        RL_VAL --> DOUBLEQL[Double Q-Learning]
        RL_VAL --> NSTEPQL[N-Step Q-Learning]
        RL_VAL --> DQN[Deep Q-Network]
        RL_VAL --> DDQN[Double DQN]
        RL_VAL --> DUELINGDNQ[Dueling DQN]
        RL_VAL --> RDQN[Rainbow DQN]
        RL_VAL --> QRDQN[Quantile Regression DQN]

        RL_POL --> REINF[REINFORCE]
        RL_POL --> PPO[PPO]
        RL_POL --> TRPO[TRPO]
        RL_POL --> NPG[Natural Policy Gradients]
        RL_POL --> CEM[Cross-Entropy Method]

        RL_ACT --> AC[Actor Critic]
        RL_ACT --> A2C[A2C]
        RL_ACT --> A3C[A3C]
        RL_ACT --> DDPG[DDPG]
        RL_ACT --> TD3[TD3]
        RL_ACT --> SAC[SAC]

        RL_MB --> MCTS[Monte Carlo Tree Search]
        RL_MB --> ALPHAMCTS[AlphaZero]
        RL_MB --> MUZERO[MuZero]
        RL_MB --> DYNAQ[Dyna-Q]
        RL_MB --> MPC[Model Predictive Control]
    end

    subgraph DL[Deep Learning]
        FFNN[Feedforward NN]
        CNN[Convolutional Networks]
        RNN[Recurrent Networks]
        TRANS[Transformers]
        GEN[Generative Models]
        ATTN[Attention Mechanisms]
        GNN[Graph NNs]
        SPC[Specialized Arch]

        FFNN --> MLP[MultiLayer Perceptron]
        FFNN --> DNN[Deep Neural Network]
        FFNN --> RBFN[Radial Basis Function NN]

        CNN --> CONVNET[CNN Gen]
        CNN --> LENET[LeNet]
        CNN --> ALEXNET[AlexNet]
        CNN --> VGGNET[VGG]
        CNN --> RESNET[ResNet]
        CNN --> DENSENET[DenseNet]
        CNN --> INCEPTION[Inception GoogLeNet]
        CNN --> EFFICIENT[EfficientNet]
        CNN --> MOBILENET[MobileNet]
        CNN --> SQUEEZENET[SqueezeNet]

        RNN --> VANILLARNN[Vanilla RNN]
        RNN --> LSTM[LSTM]
        RNN --> GRU[GRU]
        RNN --> BIRNN[Bidirectional RNN]
        RNN --> ECHO[Echo State NN]

        TRANS --> TRANSF[Transformer]
        TRANS --> BERT[BERT]
        TRANS --> GPT[GPT]
        TRANS --> T5[T5]
        TRANS --> ROBERTA[RoBERTa]
        TRANS --> XLNET[XLNet]
        TRANS --> ELECTRA[ELECTRA]
        TRANS --> DEBERTA[DeBERTa]

        GEN --> GAN[GAN]
        GEN --> VAE[VAE]
        GEN --> DIFFUSION[Diffusion Models]
        GEN --> FLOW[Flow-based Models]
        GEN --> AR[Autoregressive Models]
        GEN --> EB[Energy-based Models]

        ATTN --> SA[Self-Attention]
        ATTN --> MHA[Multi-Head Attention]
        ATTN --> CA[Cross-Attention]
        ATTN --> SPA[Sparse Attention]
        ATTN --> LA[Local Attention]

        GNN --> GCN[GCN]
        GNN --> GAT[GAT]
        GNN --> SAGE[GraphSAGE]
        GNN --> MPNN[MPNN]
        GNN --> CAPSULE[Capsule Networks]
        GNN --> NODE[Neural ODEs]

        SPC --> CAPSNET[Capsule Network]
        SPC --> NODE_SP[Neural ODEs]
    end

    subgraph ENS[Ensemble Learning]
        BAGGING[Bagging]
        BOOSTING[Boosting]
        STACKING[Stacking]
        VOTING[Voting]

        BAGGING --> RF_BAG[Random Forest]
        BAGGING --> ET_BAG[Extra Trees]
        BAGGING --> BOOTSTRAP[Bootstrap Aggregating]

        BOOSTING --> ADABOOST[AdaBoost]
        BOOSTING --> GBOOST[Gradient Boosting]
        BOOSTING --> XGB[XGBoost]
        BOOSTING --> LGBM[LightGBM]
        BOOSTING --> CATBOOST[CatBoost]
        BOOSTING --> HISTGB[HistGradientBoosting]

        STACKING --> STGEN[Stacked Generalization]
        STACKING --> MLS[Multi-Level Stacking]
        STACKING --> BLEND[Blending]

        VOTING --> HARDV[Hard Voting]
        VOTING --> SOFTV[Soft Voting]
        VOTING --> WEIGHTEDV[Weighted Voting]
    end

    subgraph DOM[Specialized Domains]
        CV[Computer Vision]
        NLP[Natural Language Processing]
        TS[Time Series Analysis]
        RECSYS[Recommender Systems]
        META[Meta Learning]
        FSL[Few-Shot Learning]

        CV --> OBJDET[Object Detection]
        CV --> SEG[Segmentation]
        CV --> FACEREC[Face Recognition]
        CV --> OCR[Optical Character Recognition]
        CV --> IMGCLASS[Image Classification]
        CV --> STYLTRANS[Style Transfer]
        CV --> SUPRES[Super Resolution]

        OBJDET --> YOLO[YOLO]
        OBJDET --> RCNN[R-CNN]
        OBJDET --> SSD[SSD]
        OBJDET --> RETINANET[RetinaNet]

        SEG --> UNET[U-Net]
        SEG --> MASKRCNN[Mask R-CNN]

        NLP --> NER[NER]
        NLP --> POSTAG[POS Tagging]
        NLP --> SENT[Sentiment Analysis]
        NLP --> TRANSL[Machine Translation]
        NLP --> QA[Question Answering]
        NLP --> SUMM[Summarization]
        NLP --> LM[Language Model]

        TS --> ARIMA[ARIMA]
        TS --> SARIMA[SARIMA]
        TS --> PROPHET[Prophet]
        TS --> EXP_SMOOTH[Exponential Smoothing]
        TS --> SS[State Space Model]
        TS --> LSTM_TS[LSTM Time Series]
        TS --> TCN[TCN]

        RECSYS --> CF[Collaborative Filtering]
        RECSYS --> CBF[Content-based Filtering]
        RECSYS --> MF[Matrix Factorization]
        RECSYS --> DCF[Deep Collab Filtering]
        RECSYS --> NCF[Neural Collaborative Filtering]
        RECSYS --> FM[Factorization Machines]

        META --> MAML[MAML]
        META --> PN[Prototypical NN]
        META --> MN[Matching NN]
        META --> RN[Relation NN]
        META --> MSGD[Meta-SGD]
        META --> REP[Reptile]

        FSL --> SIAMESE[Siamese Net]
        FSL --> TRIPLET[Triplet Net]
        FSL --> PN_FSL[Prototypical NN]
        FSL --> RN_FSL[Relation NN]
        FSL --> MAML_FSL[MAML]
    end

    subgraph OPT[Optimization Algorithms]
        GD[Gradient Descents]
        HPO[Hyperparam Opt]
        EVO[Evolutionary]

        GD --> SGD[SGD]
        GD --> MBGD[Mini-batch GD]
        GD --> MOM[Momentum]
        GD --> NAG[Nesterov]
        GD --> ADA[Adagrad]
        GD --> RMS[RMSProp]
        GD --> ADAM[Adam]
        GD --> ADAMW[AdamW]
        GD --> NADAM[Nadam]
        GD --> AMSGR[AMSGrad]

        HPO --> GRID[Grid Search]
        HPO --> RANDOM[Random Search]
        HPO --> BO[Bayesian Opt]
        HPO --> TPE[TPE]
        HPO --> HB[Hyperband]
        HPO --> BOHB[BOHB]
        HPO --> OPTUNA[Optuna]
        HPO --> PBT[Pop Based Training]

        EVO --> GA[Genetic Alg]
        EVO --> GP[Genetic Prog]
        EVO --> ES[Evolutionary Strategies]
        EVO --> PSO[Particle Swarm]
        EVO --> DE[Differential Evolution]
    end

    %% Connect root to all
    ML --> SUP
    ML --> UNSUP
    ML --> SEMI
    ML --> RL
    ML --> DL
    ML --> ENS
    ML --> DOM
    ML --> OPT
