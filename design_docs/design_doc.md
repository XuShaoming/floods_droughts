# Reports

The University of Minnesota team is developing machine learning (ML) models to predict streamflow in Minnesota watersheds based on historical and projected climate conditions. The workflow integrates climate model outputs, hydrologic simulation data, and watershed characteristics.

---

## Data Sources and Resolution

- **EDDEv1 Climate Data:** Hourly weather variables at ~36 km spatial resolution across the continental U.S.
- **HSPF Hydrologic Data:** Hourly streamflow simulations at the HUC8 watershed scale
- **Watershed Boundaries:** GIS shapefiles available at the HUC12 level

---

## Processing Pipeline

### 1. Weather Grid Selection
A K-Nearest Neighbors (KNN) algorithm selects climate data points relevant to each watershed.  
*Example:* The Kettle River watershed is represented by 9 nearby weather grid points.

### 2. Spatial Downscaling
To address the resolution gap between the climate data (~36 km) and HUC12 watershed boundaries, we apply **Inverse Distance Weighting (IDW)** to interpolate hourly weather data to each HUC12 sub-watershed, as illustrated in Figure 1.

<p align="left">
  <img src="imgs/Kettle_River.jpg" alt="Kettle River Watershed Grid Selection" width="500"/>
</p>

*Figure 1. Kettle River watershed and nearby EDDEV1 grid points used for KNN-IDW interpolation.*

### 3. Aggregation to HUC8
Area-weighted averaging combines HUC12-level interpolated weather data into HUC8-scale summaries.

### 4. Hourly Time Series Generation
The aggregated climate data are merged with streamflow data to form complete hourly time series inputs for ML modeling.

---

All steps are automated in Python and can be reused to generate training datasets for any Minnesota watershed.

---


# LSTM vs. Quantile LSTM Models for Streamflow Prediction

The figure below illustrates two models—**LSTM** and **Quantile LSTM**—designed for streamflow prediction (**Y**) using weather drivers and watershed characteristics (**X**) as inputs.

<p align="left">
  <img src="imgs/LSTMs.jpg" alt="LSTM and Quantile LSTM Model Architectures" width="600"/>
</p>

*Figure 2. Architectures of standard LSTM and Quantile LSTM models for streamflow prediction.*

---

## 1. LSTM Model

In the **standard LSTM** model (left side of the figure), the LSTM network is trained in a **many-to-many** fashion, predicting a sequence of streamflow values over time:

$$
\hat{Y}_{1:t} = f_{\text{LSTM}}(X_{1:t})
$$

where:
- $X_{1:t}$ represents the input sequence of weather drivers and watershed characteristics up to time $t$,
- $f_{\text{LSTM}}$ is the LSTM model,
- $\hat{Y}_{1:t}$ is the predicted streamflow sequence.

This approach provides only a single deterministic estimate (often aligned with the mean or median) for each time step, which may not capture uncertainty or extreme events.

---

## 2. Quantile LSTM Model

The **Quantile LSTM** model (right side of the figure) extends the standard LSTM by predicting **multiple conditional quantiles** of the streamflow distribution for each time step. It uses multiple prediction heads, each corresponding to a specific quantile $q$:

$$
\hat{Y}_{1:t}^{(q)} = f_{\text{Quantile LSTM}}(X_{1:t}; q)
$$

where $q \in \{0.1, 0.5, 0.9\}$ in this case:
- $q = 0.1$: **10th quantile** – captures low-end events, useful for drought characterization.
- $q = 0.5$: **50th quantile (median)** – measures the central tendency, indicating general model performance.
- $q = 0.9$: **90th quantile** – captures high-end events, useful for flood prediction.

This many-to-many quantile prediction allows the model to capture uncertainty and variability in streamflow at each time step.

---

## 3. Quantile Regression

Quantile regression estimates the conditional quantile function $Q_Y(q \mid X)$ rather than the mean. It is trained using the **quantile loss function** (also known as the pinball loss):

$$
\mathcal{L}_q(Y, \hat{Y}) =
\begin{cases}
q (Y - \hat{Y}) & \text{if } Y \ge \hat{Y} \\
(q - 1)(Y - \hat{Y}) & \text{if } Y < \hat{Y}
\end{cases}
$$

Or equivalently:

$$
\mathcal{L}_q(Y, \hat{Y}) = (Y - \hat{Y}) \cdot (q - \mathbb{1}_{Y < \hat{Y}})
$$

where:
- $Y$ is the observed streamflow,
- $\hat{Y}$ is the predicted quantile,
- $q$ is the quantile level,
- $\mathbb{1}_{Y < \hat{Y}}$ is an indicator function.

The model minimizes the total loss across all quantile levels and time steps:

$$
\min_{\theta} \sum_{q} \sum_{t} \mathcal{L}_q \left( Y_t, \hat{Y}_t^{(q)} \right)
$$

### Example with Quantiles [0.1, 0.5, 0.9]

Consider a Quantile LSTM trained to predict the 10th, 50th, and 90th quantiles of streamflow. Let the observed streamflow $Y$ be 100, and examine three cases: under-prediction ($\hat{Y} = 60$), over-prediction ($\hat{Y} = 140$), and correct prediction ($\hat{Y} = 100$).

#### **10th Quantile ($q = 0.1$)**
- **Under-prediction ($\hat{Y} = 60$):** Loss = $(100 - 60) \cdot (0.1 - 0) = 4$.
- **Over-prediction ($\hat{Y} = 140$):** Loss = $(100 - 140) \cdot (0.1 - 1) = 36$.
- **Correct prediction ($\hat{Y} = 100$):** Loss = $(100 - 100) \cdot (0.1 - 0) = 0$.

#### **50th Quantile ($q = 0.5$)**
- **Under-prediction ($\hat{Y} = 60$):** Loss = $(100 - 60) \cdot (0.5 - 0) = 20$.
- **Over-prediction ($\hat{Y} = 140$):** Loss = $(100 - 140) \cdot (0.5 - 1) = 20$.
- **Correct prediction ($\hat{Y} = 100$):** Loss = $(100 - 100) \cdot (0.5 - 0) = 0$.

#### **90th Quantile ($q = 0.9$)**
- **Under-prediction ($\hat{Y} = 60$):** Loss = $(100 - 60) \cdot (0.9 - 0) = 36$.
- **Over-prediction ($\hat{Y} = 140$):** Loss = $(100 - 140) \cdot (0.9 - 1) = 4$.
- **Correct prediction ($\hat{Y} = 100$):** Loss = $(100 - 100) \cdot (0.9 - 0) = 0$.

### Observations
- For the **10th quantile**, over-predictions are penalized more heavily (36) than under-predictions (4), encouraging smaller predictions.
- For the **50th quantile**, the loss is symmetric, penalizing under-predictions and over-predictions equally (20).
- For the **90th quantile**, under-predictions are penalized more heavily (36) than over-predictions (4), encouraging larger predictions.

This example demonstrates how the quantile level $q$ affects the loss behavior, biasing the model toward higher or lower predictions depending on the quantile. By predicting multiple quantiles, the Quantile LSTM can characterize the entire distribution of streamflow, enabling targeted analysis of droughts and floods.

### Example with Extreme Quantiles [0.01, 0.5, 0.99]

For extreme event modeling, consider a Quantile LSTM trained to predict the 1st, 50th, and 99th quantiles of streamflow. Using the same observed streamflow $Y = 100$ and prediction cases ($\hat{Y} = 60$, $\hat{Y} = 140$, $\hat{Y} = 100$):

#### **1st Quantile ($q = 0.01$)**
- **Under-prediction ($\hat{Y} = 60$):** Loss = $(100 - 60) \cdot (0.01 - 0) = 0.4$.
- **Over-prediction ($\hat{Y} = 140$):** Loss = $(100 - 140) \cdot (0.01 - 1) = (-40) \cdot (-0.99) = 39.6$.
- **Correct prediction ($\hat{Y} = 100$):** Loss = $(100 - 100) \cdot (0.01 - 0) = 0$.

#### **50th Quantile ($q = 0.5$)**
- **Under-prediction ($\hat{Y} = 60$):** Loss = $(100 - 60) \cdot (0.5 - 0) = 20$.
- **Over-prediction ($\hat{Y} = 140$):** Loss = $(100 - 140) \cdot (0.5 - 1) = 20$.
- **Correct prediction ($\hat{Y} = 100$):** Loss = $(100 - 100) \cdot (0.5 - 0) = 0$.

#### **99th Quantile ($q = 0.99$)**
- **Under-prediction ($\hat{Y} = 60$):** Loss = $(100 - 60) \cdot (0.99 - 0) = 39.6$.
- **Over-prediction ($\hat{Y} = 140$):** Loss = $(100 - 140) \cdot (0.99 - 1) = (-40) \cdot (-0.01) = 0.4$.
- **Correct prediction ($\hat{Y} = 100$):** Loss = $(100 - 100) \cdot (0.99 - 0) = 0$.

### Observations for Extreme Quantiles
- For the **1st quantile**, over-predictions are penalized extremely heavily (39.6) compared to under-predictions (0.4), strongly encouraging very small predictions for drought modeling.
- For the **99th quantile**, under-predictions are penalized extremely heavily (39.6) compared to over-predictions (0.4), strongly encouraging very large predictions for flood modeling.
- The extreme quantiles create much stronger bias toward their respective tails of the distribution compared to the moderate quantiles [0.1, 0.5, 0.9].

This demonstrates how extreme quantiles (0.01, 0.99) provide more aggressive bias for capturing rare drought and flood events, while moderate quantiles (0.1, 0.9) offer more balanced predictions for general low-flow and high-flow conditions.

---

## 4. Application for Droughts and Floods

- **Lower quantile (10th)**: Represents drought conditions and low-flow events.
- **Median quantile (50th)**: Measures typical conditions and overall model performance.
- **Higher quantile (90th)**: Represents extreme high-flow events, useful for flood modeling.

By predicting multiple quantiles for the entire sequence $\hat{Y}_{1:t}$, the Quantile LSTM can **characterize the entire distribution** of streamflow over time rather than producing only a single expected value.

---

## Discussion on Quantile LSTM

The Quantile LSTM approach is both interesting and viable for streamflow prediction, particularly for modeling extreme events like floods and droughts. Unlike standard regression models trained with mean squared error, which focus on capturing the mean behavior of the data, quantile regression biases the model toward specific parts of the conditional distribution. This allows the Quantile LSTM to capture variability and uncertainty across the entire distribution.

### Advantages of Quantile Regression
- **Capturing Extreme Events:** Higher quantiles encourage the model to make larger predictions, which are useful for flood modeling, while lower quantiles encourage smaller predictions, which are useful for drought characterization.
- **Flexibility:** By predicting multiple quantiles (e.g., 0.1, 0.5, 0.9), the model can produce three time series corresponding to different quantiles. Post-processing can then be used to select the appropriate quantile predictions based on the application.

### Post-Processing for Final Predictions
For each time step, the Quantile LSTM produces multiple streamflow values corresponding to different quantiles. These predictions can be post-processed to form the final time series:
- Use the 0.1 quantile predictions for drought analysis.
- Use the 0.9 quantile predictions for flood analysis.

This approach ensures the model can adapt to different use cases without retraining.

### Training Separate Models for Floods and Droughts
While the Quantile LSTM provides a unified framework for capturing both extremes, training separate models for floods and droughts is also feasible. For example, a mask with a threshold can be applied to isolate time series containing flood events, allowing a model to be trained specifically on that subset of the data.

### Uncertainty Quantification
Quantile regression biases the model to produce higher or lower predictions relative to the mean pattern. This bias is key to uncertainty quantification, as it allows the model to characterize the range of possible outcomes rather than a single deterministic value.

In summary, the Quantile LSTM approach is a robust and flexible method for streamflow prediction. It leverages quantile regression to capture uncertainty and variability, making it well-suited for applications like flood and drought analysis.

