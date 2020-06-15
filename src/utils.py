import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
from sklearn.model_selection import train_test_split

def aggregate_dataset(df, grouping_field_list,  array_field):
    df = df.groupby(grouping_field_list)['encounter_id', 
            array_field].apply(lambda x: x[array_field].values.tolist()).reset_index().rename(columns={
    0: array_field + "_array"}) 
    
    dummy_df = pd.get_dummies(df[array_field + '_array'].apply(pd.Series).stack()).sum(level=0)
    dummy_col_list = [x.replace(" ", "_") for x in list(dummy_df.columns)] 
    mapping_name_dict = dict(zip([x for x in list(dummy_df.columns)], dummy_col_list ) ) 
    concat_df = pd.concat([df, dummy_df], axis=1)
    new_col_list = [x.replace(" ", "_") for x in list(concat_df.columns)] 
    concat_df.columns = new_col_list

    return concat_df, dummy_col_list

def cast_df(df, col, d_type=str):
    return df[col].astype(d_type)

def impute_df(df, col, impute_value=0):
    return df[col].fillna(impute_value)
    
def preprocess_df(df, categorical_col_list, numerical_col_list, predictor, categorical_impute_value='nan',             numerical_impute_value=0):
    df[predictor] = df[predictor].astype(float)
    for c in categorical_col_list:
        df[c] = cast_df(df, c, d_type=str)
    for numerical_column in numerical_col_list:
        df[numerical_column] = impute_df(df, numerical_column, numerical_impute_value)
    return df

#adapted from https://www.tensorflow.org/tutorials/structured_data/feature_columns
def df_to_dataset(df, predictor,  batch_size=32):
    df = df.copy()
    labels = df.pop(predictor)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

# build vocab for categorical features
def write_vocabulary_file(vocab_list, field_name, default_value, vocab_dir='./diabetes_vocab/'):
    output_file_path = os.path.join(vocab_dir, str(field_name) + "_vocab.txt")
    # put default value in first row as TF requires
    vocab_list = np.insert(vocab_list, 0, default_value, axis=0) 
    df = pd.DataFrame(vocab_list).to_csv(output_file_path, index=None, header=None)
    return output_file_path

def build_vocab_files(df, categorical_column_list, default_value='00'):
    vocab_files_list = []
    for c in categorical_column_list:
        v_file = write_vocabulary_file(df[c].unique(), c, default_value)
        vocab_files_list.append(v_file)
    return vocab_files_list

def show_group_stats_viz(df, group):
    print(df.groupby(group).size())
    print(df.groupby(group).size().plot(kind='barh'))
 
'''
Adapted from Tensorflow Probability Regression tutorial  https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb    
'''
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2*n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t[..., :n],
                                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

def demo(feature_column, example_batch):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch))
    return feature_layer(example_batch)

def calculate_stats_from_train_data(df, col):
    mean = df[col].describe()['mean']
    std = df[col].describe()['std']
    return mean, std

def create_tf_numerical_feature_cols(numerical_col_list, train_df):
    tf_numeric_col_list = []
    for c in numerical_col_list:
        mean, std = calculate_stats_from_train_data(train_df, c)
        tf_numeric_feature = create_tf_numeric_feature(c, mean, std)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list

def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    
    reduce_dim_df = pd.merge(df, 
                             ndc_df[['NDC_Code', 'Non-proprietary Name']], 
                             left_on='ndc_code', 
                             right_on='NDC_Code', 
                             how='inner') 
    reduce_dim_df = reduce_dim_df.rename(columns={'Non-proprietary Name': 'generic_drug_name'})
    reduce_dim_df = reduce_dim_df.drop(columns=['NDC_Code', 'ndc_code'])
    reduce_dim_df['generic_drug_name'] = reduce_dim_df['generic_drug_name'] \
    .replace({'Glipizide And Metformin Hydrochloride' : 'Glipizide And Metformin Hcl', 
              'Glyburide-metformin Hydrochloride' : 'Glyburide And Metformin Hydrochloride',
              'Insulin Human' : 'Human Insulin',
              'Metformin Hydrochloride' : 'Metformin Hcl',
              })
    return reduce_dim_df

def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    df = df.sort_values(['patient_nbr', 'encounter_id'])
    first_encounter_df = df.groupby('patient_nbr').first().reset_index()
    return first_encounter_df

def patient_dataset_splitter_tf(df, PREDICTOR_FIELD, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    for col in ['race', 'primary_diagnosis_code']:
        df[col] = df[col].astype('str')
    y = df[PREDICTOR_FIELD]
    X = df.drop(columns=[PREDICTOR_FIELD])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    train = pd.concat([X_train, y_train], axis=1)
    validation = pd.concat([X_val, y_val], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    return train, validation, test



def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        vocab = tf.feature_column.categorical_column_with_vocabulary_file(key=c, 
                                                                          vocabulary_file = vocab_file_path, 
                                                                          num_oov_buckets=1)
        tf_categorical_feature_column = tf.feature_column.indicator_column(vocab)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list


def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std

def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, 
                                                          default_value=default_value, 
                                                          normalizer_fn=normalizer, 
                                                          dtype=tf.float64)
    return tf_numeric_feature


def create_tf_numerical_feature_cols(numerical_col_list, train_df):
    tf_numeric_col_list = []
    for c in numerical_col_list:
        mean, std = calculate_stats_from_train_data(train_df, c)
        tf_numeric_feature = create_tf_numeric_feature(c, mean, std)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list

def calculate_stats_from_train_data(df, col):
    mean = df[col].mean()
    std = df[col].std()
    return mean, std


def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.loc
    s = diabetes_yhat.scale
    return m, s


def get_binary_prediction(df, pred_field, actual_field, threshold):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    df['score'] = df[pred_field].apply(lambda x: 1 if x>=threshold else 0 )
    df['label_value'] = df[actual_field].apply(lambda x: 1 if x>=threshold else 0)
    return df[['score', 'label_value']]