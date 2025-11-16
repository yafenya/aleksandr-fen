"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train.head()
train.describe()
train.info()

fig, axes = plt.subplots(1, 2, figsize=(10,4))
sns.histplot(train['recency'], kde=True, ax=axes[0])
sns.histplot(train['history'], kde=True, ax=axes[1])
axes[0].set_title('Recency distribution')
axes[1].set_title('History distribution')
plt.show()

corr = train[['recency','history','mens','womens','newbie','visit']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.show()

sns.barplot(
    data=train,
    x='mens',
    y='visit',
    hue='segment'
)
plt.title('Mens × Segment → Visit')
plt.show()

sns.barplot(
    data=train,
    x='womens',
    y='visit',
    hue='segment'
)
plt.title('Womens × Segment → Visit')
plt.show()


A_LABELS = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
arm_to_idx = {a: i for i, a in enumerate(A_LABELS)}
idx_to_arm = {i: a for a, i in arm_to_idx.items()}

num_cols = ["recency", "history"]
cat_cols = ["zip_code", "channel", "history_segment"] 
bin_cols = ["mens", "womens", "newbie"]
feature_cols = num_cols + bin_cols + cat_cols

X_all = train[feature_cols].copy()
a_all = train["segment"].map(arm_to_idx).astype(int).values
r_all = train["visit"].astype(int).values
X_te_raw = test[feature_cols].copy()

preprocess = ColumnTransformer(
    transformers=[
        ("num", QuantileTransformer(), num_cols),
        ("bin", "passthrough", bin_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
)


X_all_np = preprocess.fit_transform(X_all)
X_te_np = preprocess.transform(X_te_raw)
X_all_np = X_all_np.astype("float32")
X_te_np = X_te_np.astype("float32")

action_stats = train.groupby('segment').agg({
    'visit': ['count', 'mean', 'sum']
}).round(4)
action_stats.columns = ['count', 'conversion_rate', 'total_visits']
print(action_stats)

best_static_conversion = action_stats.loc['Mens E-Mail', 'conversion_rate']

class DeterministicImitator(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 3)
        
        # Сильная инициализация в пользу Men's E-Mail
        with torch.no_grad():
            self.linear.bias.data = torch.tensor([10.0, -5.0, -5.0])  # Смещение к Men's E-Mail
            nn.init.normal_(self.linear.weight, std=0.001)  # Очень маленькие веса
    
    def forward(self, x):
        logits = self.linear(x)
        # Дополнительное смещение к Men's E-Mail
        logits[:, 0] += 5.0
        return logits

# ==================== METRIC FUNCTIONS ====================
def snips_score(pi: torch.Tensor, a: torch.Tensor, r: torch.Tensor, mu: float = 1 / 3) -> float:

    with torch.no_grad():
        w = pi[torch.arange(len(a)), a] / mu
        num = (w * r).sum()
        den = w.sum().clamp_min(1e-12)
        return (num / den).item()


X_tensor = torch.tensor(X_all_np, dtype=torch.float32)
a_tensor = torch.tensor(a_all, dtype=torch.long)
r_tensor = torch.tensor(r_all, dtype=torch.float32)

model = DeterministicImitator(X_all_np.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)

epochs = 250
best_score = -float('inf')
best_model_state = None

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(X_tensor)
    probs = torch.softmax(logits, dim=1)
    
    # Простая IPS loss
    action_probs = probs[torch.arange(len(a_tensor)), a_tensor]
    weights = (r_tensor / (1/3)).clamp_max(3.0)
    loss = -(weights * torch.log(action_probs + 1e-8)).mean()
    
    # Сильная регуляризация к Men's E-Mail
    target_probs = torch.tensor([0.999, 0.0005, 0.0005]).repeat(probs.shape[0], 1)
    kl_penalty = torch.nn.functional.kl_div(
        torch.log(probs + 1e-8), target_probs, reduction='batchmean'
    )
    
    total_loss = loss + 10.0 * kl_penalty  # Очень сильная регуляризация
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    # Логирование каждые 10 эпох
    if epoch % 10 == 0 or epoch == epochs - 1:
        with torch.no_grad():
            snips = snips_score(probs, a_tensor, r_tensor)
            score = snips - best_static_conversion
            
            if score > best_score:
                best_score = score
                best_model_state = model.state_dict().copy()
            
            print(f"📍 Epoch {epoch+1:2d}: Loss = {total_loss.item():.4f}, SNIPS = {snips:.6f}, Score = {score:.6f}")

model.load_state_dict(best_model_state)
model.eval()

with torch.no_grad():
    X_test_tensor = torch.tensor(X_te_np, dtype=torch.float32)
    logits_test = model(X_test_tensor)
    probs_test = torch.softmax(logits_test, dim=1)
    probs_test_np = probs_test.numpy()

# Анализ распределения предсказаний
print("\nРАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ:")
for i, action in enumerate(A_LABELS):
    action_probs = probs_test_np[:, i]
    print(f"  {action}:")
    print(f"    Среднее: {action_probs.mean():.6f}")
    print(f"    Стандартное отклонение: {action_probs.std():.6f}")
    print(f"    Минимум: {action_probs.min():.6f}")
    print(f"    Максимум: {action_probs.max():.6f}")

submission = pd.DataFrame({
    "id": test["id"].values,
    "p_mens_email": probs_test_np[:, 0],
    "p_womens_email": probs_test_np[:, 1],
    "p_no_email": probs_test_np[:, 2],
})

# Проверка корректности
assert np.all(np.isfinite(submission[["p_mens_email", "p_womens_email", "p_no_email"]].values))
assert np.allclose(
    submission[["p_mens_email", "p_womens_email", "p_no_email"]].sum(axis=1),
    1.0,
    atol=1e-6,
), "Сумма вероятностей не равна 1!"


# Сохранение
# submission.to_csv("submission_fen_a.csv", index=False)
# print("✅ Submission файл сохранен как 'submission_fen_a.csv'")

            
def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    
    submission_path = 'results/submission.csv'

    submission = predictions
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path
    

create_submission(submission)


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(predictions)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
