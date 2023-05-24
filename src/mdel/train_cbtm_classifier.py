import joblib
from datasets import load_dataset
from tokenizers import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


data_url = [
    "Multi-Domain-Expert-Layers/pubmed_abstracts",
    "Multi-Domain-Expert-Layers/philpapers",
    "Multi-Domain-Expert-Layers/pubmed_central",
    "Multi-Domain-Expert-Layers/freelaw",
    "Multi-Domain-Expert-Layers/arxiv",
    "Multi-Domain-Expert-Layers/github",
    "Multi-Domain-Expert-Layers/uspto",
]

expert_datasets = [load_dataset(i) for i in data_url]


tokenizer = Tokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
tokenizer.enable_truncation(max_length=1024)


tokenized_datasets = []
for ed in expert_datasets:
    tokenized_datasets.append(ed['train'].map(lambda x: {"token": [i.tokens for i in tokenizer.encode_batch(x["text"])]}, batched=True))


features = []
label = []
for ed, lab in zip(expert_datasets, data_url):
    for i in ed.select(range(min(10000, len(ed)))).iter(batch_size=10000):
        features.extend(i['token'])
        label.extend([lab] * len(i['token']))


X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)


pipeline = Pipeline([('hasher', FeatureHasher(n_features=512, input_type="string")), ('lr', LogisticRegression(multi_class='multinomial', solver='lbfgs'))])
pipeline.fit(X_train, y_train)


y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)


print(classification_report(y_train, y_train_pred))
print(classification_report(y_test, y_test_pred))


joblib.dump(pipeline, 'cbtm_classifier.pkl')
