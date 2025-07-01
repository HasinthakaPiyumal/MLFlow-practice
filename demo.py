import os,warnings,sys,numpy as np,pandas as pd,mlflow,dagshub,logging
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from mlflow.models.signature import infer_signature
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # Load dataset
    data = pd.read_csv("winequality-red.csv")
    
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    X_train = train.drop("quality", axis=1)
    X_test = test.drop("quality", axis=1)
    y_train = train["quality"]
    y_test = test["quality"]
    
    