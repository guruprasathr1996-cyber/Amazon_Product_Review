# Amazon_Product_Review

# Amazon Product Rating Prediction Using Deep Learning

## 1. Project Overview

This project aims to predict Amazon product ratings using product information and customer reviews from the Amazon Sales Dataset available on Kaggle. A hybrid deep learning model combining **Long Short-Term Memory (LSTM)** and Dense Neural Network layers is implemented using TensorFlow and Keras.

The model processes customer review text and product features to estimate product ratings on a scale of 1 to 5.

## 2. Objectives

- Load and preprocess the Amazon Sales Dataset.
- Handle missing values and clean product features.
- Tokenize and encode customer reviews into numerical sequences.
- Develop a hybrid LSTM-based deep learning model.
- Train and evaluate the model using regression metrics.
- Compare actual and predicted product ratings.
- Demonstrate predictions on unseen customer reviews.

## 3. Dataset

**Dataset Name:** Amazon Sales Dataset

**Source:** [Kaggle – Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)

The dataset contains approximately 1,465 product records.

### Features Used

| Feature | Description |
|---|---|
| `product_name` | Name of the product |
| `category` | Product category |
| `discounted_price` | Discounted product price |
| `actual_price` | Original product price |
| `discount_percentage` | Percentage discount |
| `about_product` | Product description |
| `review_title` | Customer review title |
| `review_content` | Customer review text |
| `rating` | Target product rating |

**Note:** The dataset contains product-level average ratings and review fields rather than independent rating labels for every individual review. The model therefore predicts product-level ratings.

## 4. Technologies Used

- Python
- Google Colab
- Pandas and NumPy
- TensorFlow and Keras
- Scikit-learn
- Matplotlib

## 5. Methodology

### Step 1: Data Loading and Preprocessing

- Load the dataset using Pandas.
- Remove duplicate records.
- Handle missing values.
- Convert prices, discounts, and ratings into numerical values.
- Combine review titles, review content, product names, and descriptions into a text feature.

### Step 2: Text Tokenization and Encoding

- Tokenize customer review text using the Keras Tokenizer.
- Limit the vocabulary to 10,000 words.
- Convert reviews into integer sequences.
- Pad or truncate sequences to a fixed length of 150 tokens.

### Step 3: Product Feature Processing

- Encode product categories using one-hot encoding.
- Standardize numerical product features using StandardScaler.
- Split the dataset into training and testing sets.

### Step 4: Deep Learning Model

A hybrid LSTM and Dense Neural Network architecture is used.

**Text processing branch:**
- Embedding layer with 64 dimensions.
- LSTM layer with 32 units.
- Dropout layer with a rate of 0.3.

**Product feature branch:**
- Dense layer with 32 neurons and ReLU activation.
- Batch Normalization.
- Dropout layer with a rate of 0.2.

**Combined prediction branch:**
- Concatenation of text and product features.
- Dense layer with 32 neurons and ReLU activation.
- Dropout layer with a rate of 0.2.
- Output layer with one neuron and linear activation.

### Step 5: Model Training

The model is trained using the following configuration:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Training Metric | Mean Absolute Error (MAE) |
| Batch Size | 16 |
| Maximum Epochs | 30 |
| Early Stopping Patience | 4 |
| Training/Test Split | 80% / 20% |

Early stopping is used to reduce overfitting by monitoring validation loss.

## 6. Model Evaluation

The model is evaluated using the following metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted ratings.
- **Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger errors more heavily.
- **R² Score:** Measures how well the predictions explain variation in the target ratings.

The project also generates:
- An actual-versus-predicted rating comparison table.
- A scatter plot comparing actual and predicted ratings.
- A training-versus-validation loss plot.

Actual metric values should be added here after running the notebook.

## 7. Prediction on Unseen Reviews

The trained model can predict ratings for new customer reviews using the same preprocessing steps applied during training.

Example input:

```text
Review: Excellent quality and very useful product. Worth the money.
```

The model returns a numerical rating estimate between 1 and 5.

*Example predictions are illustrative. Actual outputs depend on the trained model.*

## 8. Installation and Execution

### Requirements

Install the required libraries:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

### Run the Project

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. Upload the Amazon dataset CSV file.
4. Run the notebook cells in sequence.
5. Review the evaluation metrics and plots.
6. Test predictions on new customer reviews.

## 9. Saved Model Files

The project can save the trained model and tokenizer for reuse.

| File | Purpose |
|---|---|
| `amazon_rating_lstm.keras` | Stores the trained neural network |
| `amazon_tokenizer.pkl` | Stores the fitted text tokenizer |

The same preprocessing configuration, category column order, and numerical scaler must also be preserved to make consistent predictions in future sessions.

## 10. Limitations

- The dataset contains approximately 1,465 product records, which is relatively small for deep learning.
- Ratings represent existing product-level averages, not individual customer ratings.
- The model may overfit or fail to generalize to new products and reviews.
- Review text and product information may not fully explain a product's rating.
- Model performance must be verified using the test-set metrics.

## 11. Future Enhancements

- Use a larger dataset containing individual reviews and their corresponding star ratings.
- Compare LSTM with GRU, CNN, and transformer-based models.
- Apply sentiment analysis to customer reviews.
- Tune hyperparameters to improve prediction performance.
- Develop a web application for interactive rating prediction.
- Compare the deep learning model against simple baseline models.

## 12. Conclusion

This project demonstrates how natural language processing and deep learning can be combined with structured product features to estimate Amazon product ratings. The hybrid LSTM architecture learns representations from customer review text and product attributes, while regression metrics help assess its predictive performance.

## Author

**Project:** Amazon Product Rating Prediction Using Deep Learning

**Platform:** Google Colab

**Dataset:** Kaggle Amazon Sales Dataset
