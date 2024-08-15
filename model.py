# Import các thư viện cần thiết
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import (
    StratifiedKFold, 
    cross_val_score, 
)
import matplotlib.pyplot as plt

# Cấu hình pandas để tránh cảnh báo về việc downcast dữ liệu
pd.set_option('future.no_silent_downcasting', True)

# Đọc dữ liệu từ file CSV
df = pd.read_csv('BankChurners.csv')

# Loại bỏ cột CLIENTNUM
df = df.drop(columns=['CLIENTNUM'])

# Xóa 2 cột cuối cùng theo mô tả dữ liệu
df = df.drop(columns=[
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
])

# Xác định các cột categorical
categorical_columns = df.select_dtypes(include=['object']).columns

print(categorical_columns)

# Thực hiện one-hot encoding cho các cột categorical
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print(df_encoded.info())

# Định nghĩa features (X) và target (y)
X = df_encoded[[
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Credit_Limit",
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
]]
y = df_encoded['Attrition_Flag_Existing Customer']

# Tính toán và vẽ biểu đồ Feature Importance dựa trên Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X, y)
pd.Series(clf.feature_importances_, index=X.columns[:]).plot.bar(color='green', figsize=(12, 6))
plt.show()

# Chia dataset thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=94, stratify=y)

# Kiểm tra kích thước của các tập dữ liệu sau khi chia
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

# Định nghĩa hàm để đánh giá các mô hình
def evaluate_models(models, xtrain, ytrain, xtest, ytest, cv=None, cv_scorer=None):
    """
    Hàm để huấn luyện một danh sách các mô hình và tạo báo cáo hiệu suất.
    """
    model_scores = list()    
    for model in models:
        # Huấn luyện mô hình
        model.fit(xtrain, ytrain)
        
        # Dự đoán trên tập test
        actuals = ytest
        predictions = model.predict(xtest)
        labels = ytrain.unique()

        # Tính toán các chỉ số đánh giá
        cr = metrics.classification_report(actuals, predictions, labels=labels, output_dict=True)
        cm = metrics.confusion_matrix(actuals, predictions, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        print(f"Model: {model.steps[1][1].__class__.__name__}")
        print(cm_df)
        scores = {
            'model': model.steps[1][1].__class__.__name__,
            'accuracy': cr['accuracy'],
            'precision': cr['macro avg']['precision'],
            'recall': cr['macro avg']['recall'],
            'f1_score': cr['macro avg']['f1-score'],
        }
        
        # Thực hiện cross-validation nếu được yêu cầu
        if cv:
            cv_score = cross_val_score(model, xtest, ytest, cv=cv, scoring=cv_scorer)
            scores.update({
                'cv_mean': cv_score.mean(),
                'cv_std': cv_score.std()
            })
        
        model_scores.append(scores)
    
    # In kết quả đánh giá
    print(pd.DataFrame(model_scores))

# Khai báo các mô hình
base_models = [
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='lbfgs', random_state=94)),
    make_pipeline(StandardScaler(), DecisionTreeClassifier(min_samples_leaf=5, random_state=94)),
    make_pipeline(StandardScaler(), RandomForestClassifier(min_samples_leaf=5, random_state=94)),
]

# Cấu hình Stratified K-Fold Cross Validation
skf_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=94)

# Đánh giá các mô hình
evaluate_models(
    models=base_models,
    xtrain=X_train,
    ytrain=y_train,
    xtest=X_test,
    ytest=y_test,
    cv=skf_cv,
    cv_scorer='recall'
)
