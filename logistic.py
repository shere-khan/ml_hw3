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

def process_lin_reg_data(fn):
    data = []
    with open(fn) as f:
        for line in f:
            data.append(float(line))

    return np.array(data)

def problem1a(X, labels):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    # Configure plot
    h = .02
    ar1 = np.arange(x_min, x_max, h)
    ar2 = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(ar1, ar2)
    d = np.c_[xx.ravel(), yy.ravel()]

    # Logistic regression boundary line
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, labels)

    Z = logreg.predict(d)

    Z = Z.reshape(xx.shape)
    fig = plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot log reg classification
    plt.scatter(X[:, 0], X[:, 1], c=labels, edgecolors='k', cmap=plt.cm.Paired)

    plt.xlabel('')
    plt.ylabel('')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    fig.set_size_inches(18.5, 10.5)

    if input('save pdf (y/n)?') == 'y':
        with PdfPages('logreg.pdf') as pdf:
            pdf.savefig(fig)

    plt.show()

def problem1b(X, y):
    # Configure plot
    x_min, x_max = X.min() - .5, X.max() + .5

    # Use only one feature
    X = X.reshape(-1, 1)

    h = .02
    ar1 = np.arange(x_min, x_max, h)

    # Fit LR
    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    p = lr.predict(X)
    plt.plot(X, p, color='black', linewidth=0.5)

    fig = plt.figure(1, figsize=(18.5, 10.5))

    # Plot log reg classification
    plt.scatter(X, y, color='blue')
    plt.plot(X, p, color='black', linewidth=0.4)

    plt.xlabel('')
    plt.ylabel('')

    # Save PDF
    if input('save pdf (y/n)?') == 'y':
        with PdfPages('linreg.pdf') as pdf:
            pdf.savefig(fig)

    plt.show()

if __name__ == '__main__':
    # X1 = process_log_reg_x_data()
    # y1 = process_log_reg_y_data()
    # problem1a(X1, y1)

    X2 = process_lin_reg_data("q2x.dat")
    y2 = process_lin_reg_data("q2y.dat")
    problem1b(X2, y2)
