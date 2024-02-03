import deepchem as dc
import pandas as pd
from deepchem.models.graph_models import GraphConvModel
import numpy as np

df_data = pd.read_csv('D:\OneDrive - USTC\代码\预测吸附物几何信息\金属卟啉\CMR.csv')

def process(featurizer, model):
    X = featurizer.fearurize(df_data['smiles'])
    dataset = dc.data.NumpyDataset(X=X, y=df_data['E_gap'])



# from deepchem.models import GCNModel
#
# smiles = ["C1CCC1", "CCC"]
# labels = [0., 1.]
# featurizer = dc.feat.MolGraphConvFeaturizer()
# X = featurizer.featurize(smiles)
# dataset = dc.data.NumpyDataset(X=X, y=labels)
#
# model = GCNModel(mode='classification', n_tasks=1, batch_size=16, learning_rate=0.001)
# loss = model.fit(dataset, nb_epoch=5)



from deepchem.models import GCNModel

result = {}
metric = dc.metrics.Metric(dc.metrics.rms_score)
df_data = pd.read_csv('../CMR.csv')
# df_data = df_data[0:50].copy()


def com_process(featurizer, model):
    X = featurizer.featurize(df_data['smiles'])

    dataset = dc.data.NumpyDataset(X=X, y=df_data['E_gap'])

    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset, frac_train=0.6,
                                                                                 frac_valid=0.2, frac_test=0.2)
    dict = {}
    result = []
    dict['loss'] = model.fit(train_dataset, nb_epoch=5)
    dict['train_score'] = model.evaluate(train_dataset, [metric])
    dict['valid_score'] = model.evaluate(valid_dataset, [metric])
    dict['test_score'] = model.evaluate(test_dataset, [metric])
    result.append(model.predict(test_dataset))

    return dict, result, test_dataset


featurizer = dc.feat.MolGraphConvFeaturizer()

model_GCNModel = GCNModel(mode='regression', n_tasks=1, graph_conv_layers=[32, 32],
                          batchnorm=True,
                          batch_size=32, learning_rate=0.001)
result['GCNModel'], pre, test_data = com_process(featurizer, model_GCNModel)
df = test_data.to_dataframe()
smi = []
for index, i in df.iterrows():
    # if index==0:
    # print(int(df_data[df_data['E_gap'] == i['y']].index.values))
    smi.append(df_data.iloc[int(df_data[df_data['E_gap'] == i['y']].index.values)]['smiles'])
df['smiles'] = smi
pred=[]
for i in pre[0]:
    pred.append(i[0])
df['pre'] = list(pred)
df_save = df[['smiles', 'y', 'pre']]
df_save.columns = ['smiles', 'E_gap', 'pre']
df_save.to_csv('GCN_predict.csv', index=False, index_label=False)
# pd.DataFrame(result).to_csv('result_GCNModel.csv')
