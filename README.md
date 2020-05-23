# DEBoost

A Python Library for Weighted Distance Ensembling in Machine Learning

## Installation

### Requirements 

```sh
$ pip install -r requirements.txt
```

### DEBoost from PyPI

```sh
$ pip install deboost
```

## Usage & Examples

### Regression with Default Models for Ensemble

By default, `DEBoostRegressor` has parameters `method='regression'`, `mode='mean'`, `sdhw=True`. In other words, it will perform an ensemble of the mean of all predictions and assign higher weights to model predictions with smaller spatial/statistical distance to all other model predictions. `DEBoostClassifier` is similar, with `method='classification'`. Thus calling `DEBoostRegressor()` is akin to calling `DEBoostRegressor(method='regression', mode='mean', sdhw=True)`. An example can be found below.

```py
from sklearn.datasets import load_boston
from deboost import DEBoostRegressor

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size = 0.2, random_state=42)
rgr = DEBoostRegressor()
rgr.fit(X_train, y_train)
rgr.predict(X_test)
```

Users can switch to other built-in distance metrics by changing `mode='mean'` to any of: `'median'`,`'dist_euclid'`,`'dist_cosine'`,`'dist_jaccard'`,`'dist_chebyshev'`, `'dist_correlation'`,`'dist_cityblock'`,`'dist_canberra'`,`'dist_braycurtis'`,`'dist_hamming'`,`'dist_battacharyya'`. For more details on the distance metrics, refer to the referenced manuscript in the section below. If the user desires the second method of weighted ensemble (higher weights to model predictions with larger spatial/statistical distances), they may invoke `sdhw=False` instead.

### Using Custom Models Instead of Built-ins

The default models available for regression are Ridge, Lasso, Elastic net, AdaBoost Regressor, Gradient Boosting Regressor, Random Forest Regressor, Support Vector Machine Regressor, LightGBM Regressor and XGBoost Regressor. For the classification task, the models are AdaBoost Classifier, Gradient Boosting Classifier, Gaussian Naive Bayes, K-Nearest Neighbors Classifier, Logistic Regression, Random Forest Classifier, Support Vector Machine Classifier, Decision Tree Classifier, LightGBM Classifier and XGBoost Classifier. These models have default parameters. 

To use custom models, users must first ensure that they have at least the `predict` method like models from Scikit-learn. Suppose that the user wants to ensemble two models - Lasso and Ridge for regression, each used alongside GridSearchCV from Scikit-learn. Then they may add them into DEBoostRegressor with the following lines of code:

```py
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from deboost import DEBoostRegressor

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size = 0.2, random_state=42)
model = Ridge()
model2 = Lasso()
cv = KFold(n_splits = 5, shuffle=True, random_state=42)
grid = {'alpha': [0.0003, 0.001, 0.003, 0.01, 0.03]}
grid2 = {'alpha': np.linspace(0.0001, 0.1, 112)}
gs = GridSearchCV(model, grid, n_jobs=-1, cv=cv, verbose=0)
gs2 = GridSearchCV(model2, grid2, n_jobs=-1, cv=cv, verbose=0)
rgr = DEBoostRegressor()
rgr.models = [gs, gs2]
rgr.fit(X_train, y_train)
rgr.predict(X_test)
```

Alternatively, users can fit `gs` and `gs2` first and then add it to `rgr.models`, and perform predictions. 


## License

```
MIT License

Copyright (c) 2020 Khoong Wei Hao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Reference

The preprint can be found at https://www.preprints.org/manuscript/202005.0354/v1.

## Donate

If you will like to make a donation to support us in this open-source project, you may proceed by accessing the donation page in the button below.

[![](https://www.paypalobjects.com/en_GB/SG/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=M2CQQ88GMKXXQ)

<form action="https://www.paypal.com/cgi-bin/webscr" method="post" target="_top">
<input type="hidden" name="cmd" value="_s-xclick" />
<input type="hidden" name="hosted_button_id" value="M2CQQ88GMKXXQ" />
<input type="image" src="https://www.paypalobjects.com/en_GB/SG/i/btn/btn_donateCC_LG.gif" border="0" name="submit" title="PayPal - The safer, easier way to pay online!" alt="Donate with PayPal button" />
<img alt="" border="0" src="https://www.paypal.com/en_SG/i/scr/pixel.gif" width="1" height="1" />
</form>

-----

[Return](https://weihao94.github.io/)