from joblib import Parallel, delayed
import numpy as np
import os
import pandas as pd
from scipy import sparse

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, _fit_transform_one, _transform_one
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearndf.transformation import SimpleImputerDF, StandardScalerDF, FunctionTransformerDF, VarianceThresholdDF


class PandasFeatureUnion(FeatureUnion):
    """
    Sci-kit learn feature union with Pandas properties
    https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union/index.html
    """
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis='columns', copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

class ColumnSelector(TransformerMixin):
    """
    Selects only a specified subset of features
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X):
        return X[self.columns]

class ColumnFilter(TransformerMixin):
    """
    Filters out a specified subset of features
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')

class SignUpDayExtractor(TransformerMixin):
    def __init__(self, output_feature_name='signup_day'):
        self.output_feature_name = output_feature_name

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X):
        data = X.copy()
        data[self.output_feature_name] = data.signup_date.dt.day.astype('uint64')
        return data


class OneHotEncoder(TransformerMixin):
    def __init__(self, columns=None, columns_to_ignore=None, drop_first=False):
        self.columns = columns
        self.columns_to_ignore = columns_to_ignore
        self._drop_first = drop_first

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        self.columns = self.columns if self.columns is not None else X.columns

        if self.columns_to_ignore:
            self.columns = list(set(self.columns) - set(self.columns_to_ignore))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(X, columns=self.columns, drop_first=self._drop_first)


class CollinearityThreshold(TransformerMixin):
    def __init__(self, threshold:np.float64=0.9, correlation_method:str='spearman', verbose=False):
        """
        Removes features that are considered highly correlated.
        Args:
            threshold (np.float64, optional): The upper bound on the acceptable correlation coefficient between two features. Defaults to 0.9.
        """
        self._threshold = threshold
        self._correlated_pairs = []
        self._verbose = verbose
    
    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        df = X.copy()

        corr = pd.DataFrame(np.abs(np.corrcoef(df.values, rowvar=False)), columns=df.columns, index=df.columns)

        # Select upper triangle of matrix
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        for column, rows in upper.iteritems():
            self._correlated_pairs.extend([(column, i) if column < i else (i, column) for (i, coef) in rows.iteritems() if abs(coef) > self._threshold])
    
        if self._verbose:
            print(f'[{self.__class__.__name__}] Correlated features: {self._correlated_pairs}')
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Drop first feature in each correlated pair
        return X.drop(columns=[f1 for (f1, f2) in self._correlated_pairs])


class PipelineCreator:
    def __init__(self, numerical_features, nominal_features, filtered_features, collinearity_threshold=0.9, verbose=True):
        self._numerical_features = numerical_features
        self._nominal_features = nominal_features
        self._filtered_features = filtered_features
        self.collinearity_threshold = collinearity_threshold
        self.verbose = verbose
        
    def create_nominal_preprocessing_pipeline(self):
        return make_pipeline(
            ColumnSelector(columns=self._nominal_features),
            SimpleImputerDF(strategy='most_frequent'),
            OneHotEncoder(drop_first=True, columns_to_ignore=None),
            verbose=self.verbose)

    def create_numerical_preprocessing_pipeline(self):
        return make_pipeline(
            ColumnSelector(columns=self._numerical_features),
            SimpleImputerDF(strategy='median'),
            FunctionTransformerDF(np.log1p),
            StandardScalerDF(),
            verbose=self.verbose)

    def create_preprocessing_pipeline(self):
        return Pipeline(steps=[
            ('signupday', SignUpDayExtractor()),
            ('features', PandasFeatureUnion([
                ('nominal', self.create_nominal_preprocessing_pipeline()),
                ('numerical', self.create_numerical_preprocessing_pipeline())
            ])),
            ('columnfilter', ColumnFilter(columns=self._filtered_features)),
            ('collinearitythreshold', CollinearityThreshold(threshold=self.collinearity_threshold, verbose=self.verbose))
        ])


def build_features(input_filename):
    data = pd.read_json(input_filename)

    data['signup_date'] = pd.to_datetime(data.signup_date)
    data['last_trip_date'] = pd.to_datetime(data.last_trip_date)

    # Create target variable
    last_active_trip_date = data.last_trip_date.max() - pd.Timedelta(30, unit='d')
    data.loc[data.last_trip_date >= last_active_trip_date, 'is_active'] = 1
    data.loc[data.last_trip_date < last_active_trip_date, 'is_active'] = 0
    data['is_active'] = data['is_active'].astype('category')

    X = data.drop(columns=['is_active'])
    y = data.is_active

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
    nominal_features = ['city', 'ultimate_black_user', 'phone']
    datetime_features = ['signup_date', 'last_trip_date']

    pipeline_creator = PipelineCreator(
        nominal_features=nominal_features,
        numerical_features=numerical_features + ['signup_day'],
        filtered_features=datetime_features,
        collinearity_threshold=0.7)

    preprocessor = pipeline_creator.create_preprocessing_pipeline()
    preprocessor.fit(X_train, y_train)
    X_train_enc = preprocessor.transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    X_train_enc.info()
    return X_train_enc, X_test_enc, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = build_features('./data/ultimate_data_challenge.json')
    X_train.to_pickle('./data/X_train.pkl')
    y_train.to_pickle('./data/y_train.pkl')
    X_test.to_pickle('./data/X_test.pkl')
    y_test.to_pickle('./data/y_test.pkl')