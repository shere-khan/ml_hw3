import plotly as py
import plotly.graph_objs as go
import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def process_log_reg_x_data():
    x = []
    with open("q1x.dat") as f:
        for line in f:
            vals = line.split()
            a = float(vals[0])
            b = float(vals[1])
            x.append([a, b])

    return np.array(x)

def process_log_reg_y_data():
    y = []
    with open("q1y.dat") as f:
        for line in f:
            y.append((int(float(line))))

    return np.array(y)

def one_dim_log_reg_ex():
    n_samples = 100
    np.random.seed(0)
    X = np.random.normal(size=n_samples)
    y = (X > 0).astype(np.float)
    X[X > 0] *= 4
    X += .3 * np.random.normal(size=n_samples)

    X = X[:, np.newaxis]

    # run the classifier
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(X, y)

    # Plot of 1s and 0s
    p1 = go.Scatter(x=X, y=y,
                    mode='markers',
                    marker=dict(color='black'),
                    showlegend=False
                    )
    X_test = np.linspace(-5, 10, 300)

    def model(x):
        return 1 / (1 + np.exp(-x))

    loss = model(X_test * clf.coef_ + clf.intercept_).ravel()

    # Plot of sigmoid
    p2 = go.Scatter(x=X_test, y=loss,
                    mode='lines',
                    line=dict(color='red', width=3),
                    name='Logistic Regression Model')

    ols = linear_model.LinearRegression()

    ols.fit(X, y)
    cof = ols.coef_
    inter = ols.intercept_
    # Straight line
    p3 = go.Scatter(x=X_test, y=ols.coef_ * X_test + ols.intercept_,
                    mode='lines',
                    line=dict(color='blue', width=1),
                    name='Linear Regression Model'
                    )
    # halfway line
    p4 = go.Scatter(x=[-4, 10], y=2 * [.5],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False
                    )

    layout = go.Layout(xaxis=dict(title='x', range=[-4, 10],
                                  zeroline=False),
                       yaxis=dict(title='y', range=[-0.25, 1.25],
                                  zeroline=False))

    fig = go.Figure(data=[p1, p2, p3, p4], layout=layout)

def problem1(X, labels):
    # Fit classifier
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, labels)

    # Fit LR
    lr = linear_model.LinearRegression()
    lr_x = X[:,0][:, np.newaxis]
    lr_y = X[:,1][:, np.newaxis]
    lr.fit(lr_x, lr_y)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    # Configure plot
    h = .02
    ar1 = np.arange(x_min, x_max, h)
    ar2 = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(ar1, ar2)
    d = np.c_[xx.ravel(), yy.ravel()]
    Z = logreg.predict(d)

    Z = Z.reshape(xx.shape)
    fig = plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot log reg classification
    plt.scatter(X[:, 0], X[:, 1], c=labels, edgecolors='k', cmap=plt.cm.Paired)

    # Plot LR
    t = ar1.reshape((ar1.size, 1))
    lrpredict = lr.predict(np.sort(t))
    plt.plot(t, lrpredict, color='black', linewidth=0.5)

    plt.xlabel('')
    plt.ylabel('')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    fig.set_size_inches(18.5, 10.5)

    if input('save pdf (y/n)?') == 'y':
        with PdfPages('LR.pdf') as pdf:
            pdf.savefig(fig)

    plt.show()

if __name__ == '__main__':
    py.tools.set_credentials_file(username='jbarry', api_key='NxzNmvLGLfOXz9xjF2HI')

    X = process_log_reg_x_data()
    y = process_log_reg_y_data()
    # one_dim_log_reg_ex()

    problem1(X, y)

    # iriset()
