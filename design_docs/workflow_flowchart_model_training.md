# Model Training Pipeline

```mermaid
flowchart LR
    %% Input Sources
    E2[Experiment Selection<br/>streamflow_exp1/exp2<br/>multitarget, etc.]
    F4[PyTorch DataLoaders<br/>Batch Processing]
    
    %% Model Training
    G1[train.py<br/>LSTM Training Pipeline]
    G2[models/LSTMModel.py<br/>Neural Network Architecture]
    G3[Training Loop<br/>Loss Calculation<br/>Backpropagation]
    G4[Early Stopping<br/>Learning Rate Scheduling<br/>Model Checkpointing]
    
    %% Model Outputs
    H1[Trained Models<br/>best_model.pth<br/>final_model.pth]
    H2[Training Logs<br/>TensorBoard<br/>Loss Curves]
    H3[Model Configuration<br/>config.yaml<br/>model_config.json]
    
    %% Flow connections
    F4 --> G1
    E2 --> G1
    G1 --> G2
    G2 --> G3
    G3 --> G4
    
    G4 --> H1
    G4 --> H2
    G1 --> H3
    
    %% Styling
    classDef config fill:#fff3e0
    classDef processing fill:#f3e5f5
    classDef training fill:#e8f5e8
    classDef results fill:#fce4ec
    
    class E2 config
    class F4 processing
    class G1,G2,G3,G4 training
    class H1,H2,H3 results
```

## Detailed Workflow Description

### 1. Input Sources
- **Experiment Selection**: Configuration parameters from config.yaml for specific experiments (streamflow_exp1, streamflow_exp2, multitarget)
- **PyTorch DataLoaders**: Batch-processed training, validation, and test datasets ready for model consumption

### 2. Model Training Pipeline

#### A. Training Pipeline Initialization (`train.py`)
1. **Configuration Loading**: Parse experiment-specific parameters from config.yaml
2. **Model Instantiation**: Initialize LSTM model with specified architecture
3. **Optimizer Setup**: Configure Adam/SGD optimizer with learning rate scheduling
4. **Loss Function**: Setup appropriate loss functions (MSE, MAE, or custom quantile loss)

#### B. Neural Network Architecture (`models/LSTMModel.py`)
1. **LSTM Layers**: Configurable number of LSTM layers with hidden sizes
2. **Dropout Regularization**: Prevent overfitting with dropout layers
3. **Output Layers**: Linear layers for final predictions
4. **Activation Functions**: ReLU, Tanh, or other activation functions

#### C. Training Loop
1. **Forward Pass**: Process batched sequences through LSTM network
2. **Loss Calculation**: Compute prediction errors using specified loss function
3. **Backpropagation**: Calculate gradients and update model parameters
4. **Validation**: Evaluate model performance on validation set each epoch

#### D. Training Management
1. **Early Stopping**: Monitor validation loss to prevent overfitting
2. **Learning Rate Scheduling**: Adaptive learning rate adjustment
3. **Model Checkpointing**: Save best and final model states
4. **Progress Monitoring**: Track training metrics and convergence

### 3. Output Products

#### A. Trained Models
- **Best Model**: Model with lowest validation loss (`best_model.pth`)
- **Final Model**: Model state at training completion (`final_model.pth`)
- **Model Weights**: Complete neural network parameters and optimizer states

#### B. Training Logs
- **TensorBoard**: Interactive training visualization and metrics tracking
- **Loss Curves**: Training and validation loss progression over epochs
- **Performance Metrics**: RMSE, MAE, R², NSE, KGE tracking during training

#### C. Model Configuration
- **Saved Config**: Complete experiment configuration (`config.yaml`)
- **Model Metadata**: Architecture details and training parameters (`model_config.json`)
- **Reproducibility**: Seeds, versions, and environment information for replication
