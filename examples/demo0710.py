from mlcopilot.space import import_space
from mlcopilot.suggest import suggest

configs, _ = suggest(
    import_space("5889"),
    "The task is a tabular classification task. The dataset has 1000 rows and 10 columns. "
    "The goal is to predict the class of the last column. All the columns are categorical."
)
for config in configs:
    print(config)


configs, _ = suggest(
    import_space("model"),
    "The Large Movie Review Dataset is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. "
    "It contains 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well."
)
print(configs)


configs, _ = suggest(
    import_space("hp"),
    "I want to train BLOOM, which is essentially a large language model similar to GPT3 (auto-regressive model for next token prediction), "
    "but has been trained on 46 different languages and 13 programming languages. Specifically I'm interested in bloom-560m version."
)
print(configs)


configs, _ = suggest(
    import_space("xgboost.XGBRegressor"),
    "With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, the goal is to predict the final price of each home."
)
print(configs)
