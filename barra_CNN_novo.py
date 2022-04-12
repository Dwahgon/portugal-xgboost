import time 
import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from pandas import read_csv

# biblioteca para o mapeamento de atributos categóricos
from sklearn.preprocessing import LabelEncoder

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


'''Read dataset'''
df_full = pd.read_csv('dados_UDESC.csv')

#df_teste = df_full.groupby(['ano', 'mes','dia','hora'])[['barra_fluxo']].count()
df_dia = df_full.groupby(['ano', 'mes','dia'])[['barra_fluxo']].count()

# transformar o resultado do groupby em dataframe.
df_fim_dia = df_dia.reset_index(level=df_dia.index.names)

# alteração de nomes de colunas
df_fim_dia.rename(columns={'barra_fluxo': 'qtde_medidas'}, inplace=True)

#df_teste = df_full.groupby(['ano', 'mes','dia','hora'])[['barra_fluxo']].count()
df_mes = df_full.groupby(['ano', 'mes'])[['barra_fluxo']].count()

# transformar o resultado do groupby em dataframe.
df_fim_mes = df_mes.reset_index(level=df_mes.index.names)

# alteração de nomes de colunas
df_fim_mes.rename(columns={'barra_fluxo': 'qtde_medidas'}, inplace=True)

## Dados setembro 2021 completo e com as duas estacoes (aveiro e dunas)
# colocar em variavel "numerica" somente as colunas numericas
# incluindo as medicoes de velocidade para cada radar
X_numerical = df_full[['med_temp', 'max_temp','min_temp','med_vento', 'max_vento','med_vento_av','max_vento_av', 'med_prec','med_rg', 
                       'med_prec_av','med_rg_av','med_temp_av','max_temp_av','min_temp_av','ria_med_vel_1','ria_med_vel_0',
                       'poste_med_vel_1','poste_med_vel_0','ponte_med_vel_1','ponte_med_vel_0','ria_max_vel_1','ria_max_vel_0',
                       'poste_max_vel_1','poste_max_vel_0','ponte_max_vel_1','ponte_max_vel_0','ria_min_vel_1','ria_min_vel_0',
                       'poste_min_vel_1','poste_min_vel_0','ponte_min_vel_1','ponte_min_vel_0']]

# excluindo colunas 

X_numerical = X_numerical.reset_index()

X_numerical = X_numerical.drop(columns=['index'])


 # com o objetivo de facilitar a utilização de redes neurais, os atributos de velocidade relacionadas com os radares (aproximação e distanciamento) que não possuem valor (NaN) foram preenchidos com valor "0".
 
 
X_numerical['ria_med_vel_1'] = X_numerical['ria_med_vel_1'].fillna(0)
X_numerical['ria_med_vel_0'] = X_numerical['ria_med_vel_0'].fillna(0)
X_numerical['poste_med_vel_1'] = X_numerical['poste_med_vel_1'].fillna(0)
X_numerical['poste_med_vel_0'] = X_numerical['poste_med_vel_0'].fillna(0)
X_numerical['ponte_med_vel_1'] = X_numerical['ponte_med_vel_1'].fillna(0)
X_numerical['ponte_med_vel_0'] = X_numerical['ponte_med_vel_0'].fillna(0)
X_numerical['ria_max_vel_1'] = X_numerical['ria_max_vel_1'].fillna(0)
X_numerical['ria_max_vel_0'] = X_numerical['ria_max_vel_0'].fillna(0)
X_numerical['poste_max_vel_1'] = X_numerical['poste_max_vel_1'].fillna(0)
X_numerical['poste_max_vel_0'] = X_numerical['poste_max_vel_0'].fillna(0)

X_numerical['ponte_max_vel_1'] = X_numerical['ponte_max_vel_1'].fillna(0)
X_numerical['ponte_max_vel_0'] = X_numerical['ponte_max_vel_0'].fillna(0)
X_numerical['ria_min_vel_1'] = X_numerical['ria_min_vel_1'].fillna(0)
X_numerical['ria_min_vel_0'] = X_numerical['ria_min_vel_0'].fillna(0)
X_numerical['poste_min_vel_1'] = X_numerical['poste_min_vel_1'].fillna(0)
X_numerical['poste_min_vel_0'] = X_numerical['poste_min_vel_0'].fillna(0)
X_numerical['ponte_min_vel_1'] = X_numerical['ponte_min_vel_1'].fillna(0)
X_numerical['ponte_min_vel_0'] = X_numerical['ponte_min_vel_0'].fillna(0)


# cria variavel para receber os atributos categoricos
# com minuto_meteo

X_cat = df_full[['ano','wd','mes','dia','hora','minuto_meteo','rumo_vento_med_corr','rumo_vento_max_corr', 'rumo_vento_med_av_corr', 'rumo_vento_max_av_corr']]


# A variável X_fluxo receberá os valores de fluxo a serem utilizados tanto para treinamento quanto para teste do modelo.
# o DF já possui valores de fluxo calculados para as duas regiões e para os instantes de tempo t0 até t10.

X_fluxo = df_full[['barra_fluxo']] # aqui significa o fluxo na barra no instante t0
#X_fluxo = df_full[['barra_fluxo_t1']] # aqui significa o fluxo na barra no instante t1
#X_fluxo = df_full[['costa_fluxo_t10']]
#X_fluxo = df_full[['barra_fluxo_t10']]
#X_fluxo = df_full[['costa_fluxo']]

# Aqui sem a normalização
X_all = pd.concat([X_cat, X_numerical, X_fluxo], axis = 1)


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# função para utilizar na avaliação do melhor modelo de rede neural
def adjust_prediction(z):
    testing_data = pd.DataFrame(z, columns=X_test.columns.tolist())
    return model.predict(testing_data)


# aplicação do Encoder nos atributos de rumo do vento
#rumo_vento_med_corr	rumo_vento_max_corr	rumo_vento_med_av_corr	rumo_vento_max_av_corr

categorical=['rumo_vento_med_corr']

X_all=MultiColumnLabelEncoder(columns=categorical).fit_transform(X_all)
#print(X_all.shape)

categorical=['rumo_vento_max_corr']

X_all=MultiColumnLabelEncoder(columns=categorical).fit_transform(X_all)
#print(X_all.shape)

categorical=['rumo_vento_med_av_corr']

X_all=MultiColumnLabelEncoder(columns=categorical).fit_transform(X_all)
#print(X_all.shape)

categorical=['rumo_vento_max_av_corr']

X_all=MultiColumnLabelEncoder(columns=categorical).fit_transform(X_all)

# FIM DA PREPARAÇÃO

# DEFINIÇÃO DE HIPERPARAMETROS 

#data=X_all.drop('barra_fluxo',axis=1)
label=X_all[['barra_fluxo']]
data=X_all.drop('barra_fluxo',axis=1)

column_indices = {name: i for i, name in enumerate(X_all.columns)}

n = len(X_all)
X_train = data[0:int(n*0.8)]
#val_df = X_all[int(n*0.7):int(n*0.9)]
X_test = data[int(n*0.8):]

y_train = label[0:int(n*0.8)]
#val_df = X_all[int(n*0.7):int(n*0.9)]
y_test = label[int(n*0.8):]

num_features = X_all.shape[1]

# Clear any logs from previous runs
#!rm -rf ./logs/







# residuos modelo base



from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error
from math import sqrt




# R2 apresenta o percentual de medidas que ele conseguiu acertar
# R2 ajustado apresenta o percentual apos incluir alguns atributos que provavelmente não auxiliam no modelo

#mae = mean_absolute_error(y_test, y_predict)
#mse = mean_squared_error(y_test, y_predict)
#msle = mean_squared_log_error(y_test, y_predict)
#rmse = sqrt(mse)
#r2 = r2_score(y_test, y_predict)
#adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

# teste_tanh - t
#print("MAE_Original: ", mae, "\nMSE: ", mse,  "\nRMSE: ", rmse, "\nR2: ", r2, "\nADJ R2: ", adj_r2, "\n")

#print("MAE_Barra_Original: ", mae_barra_orig, "\nMSE_barra: ", mse_barra_orig,  "\nRMSE_barra: ", rmse_barra_orig, "\nR2_barra: ", r2_barra_orig, "\nADJ R2_barra: ", adj_r2_barra_orig, "\n")

#print("MAE_Costa_Original: ", mae_costa_orig, "\nMSE_costa: ", mse_costa_orig,  "\nRMSE_costa: ", rmse_costa_orig, "\nR2_costa: ", r2_costa_orig, "\nADJ R2_costa: ", adj_r2_costa_orig, "\n")

column_indices = {name: i for i, name in enumerate(X_all.columns)}

n = len(X_all)
train_df = X_all[0:int(n*0.7)]
val_df = X_all[int(n*0.7):int(n*0.9)]
test_df = X_all[int(n*0.9):]

num_features = X_all.shape[1]

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                     label_columns=['barra_fluxo'])

w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['barra_fluxo'])

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

w2.example = example_inputs, example_labels

def plot(self, model=None, plot_col='barra_fluxo', max_subplots=5):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} ')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [10 min]')

WindowGenerator.plot = plot

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['barra_fluxo'])

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

baseline = Baseline(label_index=column_indices['barra_fluxo'])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

wide_window = WindowGenerator(
    input_width=12, label_width=12, shift=1,
    label_columns=['barra_fluxo'])


wide_window = WindowGenerator(
    input_width=3, label_width=3, shift=1,
    label_columns=['barra_fluxo'])



MAX_EPOCHS = 150

def compile_and_fit(model, window, modelo, patience=3):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanAbsoluteError(),
                optimizer=tf.optimizers.RMSprop(),
                metrics=[tf.metrics.MeanAbsoluteError()])
  
  log_dir = "./logs/Logs_TensorBoard_Wit/Neural_jan_2022/" +  modelo +  '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      batch_size=32,
                      validation_data=window.val,
                      callbacks=[early_stopping,tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)])
  return history



CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['barra_fluxo'])



LABEL_WIDTH = 12
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['barra_fluxo'])


'''
AQUI
'''
OUT_STEPS = 12
multi_window = WindowGenerator(input_width=12,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_window.plot()


class MultiStepLastBaseline(tf.keras.Model):
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)

class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)



####################################################
import csv 

combs = []

for i in range(1,7):
    combs.append( (i,i) )

for i in range(1,7):
    if i != 1:
        combs.append( (i,1) )

with open('resultados_barra_CNN_novo.csv', mode='a') as res:
    res.truncate(0)
    writer = csv.writer(res, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow( ['input','output', 'mae1','mae2','mae3','time1','time2','time3'] )

for size in combs:
    OUT_STEPS = size[1]
    input_width = size[0]

    attempt = 1
    times = []
    mae = []

    while attempt <= 3:
        print('SIZE: ', size)
        print('ATTEMPT: ', attempt)
        print('-----------')

        start = time.time()

        multi_window = WindowGenerator(input_width=size[0],
                                    label_width=OUT_STEPS,
                                    shift=OUT_STEPS)

        
        multi_conv_model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -OUT_STEPS:, :]),
            tf.keras.layers.Conv1D(64, activation='relu', kernel_size=(OUT_STEPS)),
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        history = compile_and_fit(multi_conv_model, multi_window, modelo='Multi_conv')

        end = time.time()

        IPython.display.clear_output()

        multi_val_performance['Multi_conv'] = multi_conv_model.evaluate(multi_window.val)
        
        times.append(end-start)
        mae.append(multi_val_performance['Multi_conv'][0])

        attempt += 1
    
    time1,time2,time3 = times[0],times[1],times[2]
    mae1,mae2,mae3 = mae[0],mae[1],mae[2]
    
    with open('resultados_barra_CNN_novo.csv', mode='a') as res:
        writer = csv.writer(res, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( [input_width, OUT_STEPS, mae1, mae2, mae3, time1, time2, time3] )





