import os,warnings,sys,numpy as np,pandas as pd,mlflow,dagshub,logging
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from mlflow.models.signature import infer_signature
import mlflow.sklearn

dagshub.init(repo_owner='hasinthakapiyumal', repo_name='MLFlow-practice', mlflow=True)

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
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    mlflow.set_tracking_uri("https://dagshub.com/hasinthakapiyumal/MLFlow-practice.mlflow")
    with mlflow.start_run() as run:
        mlflow.set_tag("alpha", alpha)
        mlflow.set_tag("l1_ratio", l1_ratio)
        
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse, mae, r2 = eval_metrics(y_test, y_pred)
        
        logger.info(f"Model trained with alpha={alpha}, l1_ratio={l1_ratio}")
        logger.info(f"Model performance: MSE={mse}, MAE={mae}, R2={r2}")
        
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="sklearn-model", signature=signature)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # if tracking_url_type_store != "file":
        #     mlflow.sklearn.log_model(
        #         model,
        #         "model",
        #         registered_model_name="ElasticNetWineQualityModel"
        #     )
        # else:
        #     mlflow.sklearn.log_model(model, "model")