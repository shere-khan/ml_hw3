import numpy as np, math as m
from sklearn import linear_model
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

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
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot log reg classification
    plt.scatter(X[:, 0], X[:, 1], c=labels, edgecolors='k', cmap=plt.cm.Paired)

    plt.xlabel('')
    plt.ylabel('')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    if input('save pdf (y/n)?') == 'y':
        with PdfPages('logreg.pdf') as pdf:
            fig = plt.figure(1, figsize=(18.5, 10.5))
            # fig.set_size_inches(18.5, 10.5)
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

    # Plot log reg classification
    plt.scatter(X, y, color='blue')
    plt.plot(X, p, color='black', linewidth=0.4)

    plt.xlabel('')
    plt.ylabel('')

    # Save PDF
    if input('save pdf (y/n)?') == 'y':
        with PdfPages('linreg.pdf') as pdf:
            fig = plt.figure(1, figsize=(18.5, 10.5))
            pdf.savefig(fig)


    plt.show()

def estimate_poly_fit(X, y):
    # Configure plot
    x_min, x_max = X.min() - .5, X.max() + .5

    # Use only one feature
    X = X.reshape(-1, 1)

    x_plot = np.linspace(x_min, x_max, 100)
    fig = plt.figure(1, figsize=(18.5, 10.5))

    # Plot all fit estimations
    colors = ['teal', 'red', 'gold']
    for count, degree in enumerate([3, 4, 5]):
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        y_plot = model.predict(x_plot[:, np.newaxis])
        plt.plot(x_plot, y_plot, color=colors[count], linewidth=1,
                 label="degree %d" % degree)

    # Plot log reg classification
    plt.scatter(X, y, color='blue')

    plt.xlabel('')
    plt.ylabel('')

    # Save PDF
    if input('save pdf (y/n)?') == 'y':
        with PdfPages('polyfit.pdf') as pdf:
            pdf.savefig(fig)

    plt.legend(loc='lower left')

    plt.show()

def svm(X, y):
    # print("C_range..")
    # C_range = 10. ** np.arange(-3, 8)
    # print("gamma_range..")
    # gamma_range = 10. ** np.arange(-5, 4)
    # print("dict..")
    # param_grid = dict(gamma=gamma_range, C=C_range)
    # print("Grid search..")
    # grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
    # print("Fit..")
    # grid.fit(X, y)
    # print("The best classifier is: ", grid.best_estimator_)
    with open("sklearn_svc_results_k5_nosplit_acc2.txt", "w") as f:
        # for d in range(1, 5):
            # for c in range(100, 200):
        clf = SVC(C=1, gamma=1.00, kernel='poly', degree=3)
        S = cross_val_score(clf, np.array(X), np.array(y), cv=4)
        # f.write("d: {0}  c: {1:>2}  error: {2:.4f}\n".format(d, c, S.mean()))
        # print("d: {0}  c: {1:>2}  error: {2:.4f}".format(d, c, S.mean()))
        f.write("d: {0}  c: {1:>2}  error: {2:.4f}\n".format(1, 1, S.mean()))
        print("d: {0}  c: {1:>2}  error: {2:.4f}".format(1, 1, S.mean()))
        f.write("\n")
        # print()

def get_yeast_data():
    fn = 'yeast.data'
    with open(fn) as f:
        X = []
        y = []
        x1 = []
        for line in f:
            d = line.split()
            x1.append(d.pop(0))
            y.append(d.pop(-1))
            x = list(map(float, d))
            for d in x:
                if d < 0:
                    print('test1')
            X.append(x)

    return (x1, X), y

def find_min_max(X):
    mins = list(map(int, "0 0 0 0 0 0 0 0".split()))
    maxs = list(map(int, "0 0 0 0 0 0 0 0".split()))
    for x in X:
        for i in range(len(x)):
            mins[i] = min(x[i], mins[i])
            maxs[i] = max(x[i], maxs[i])

    return mins, maxs

def normalize(X, mins, maxs):
    scaleddata = []
    for x in X:
        ex = []
        for i in range(len(x)):
            r = maxs[i] - mins[i]
            ex.append(min_max_scaling(x[i], mins[i], r))
        scaleddata.append(ex)

    return scaleddata

def min_max_scaling(d, minn, r):
    return 2 * ((d - minn) / r) - 1

def set_to_dict(S):
    D = {}
    for i, s in enumerate(S):

        D[s] = i + 1

    return D

def plot_yeast_data(X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    # Configure plot
    h = .02
    ar1 = np.arange(x_min, x_max, h)
    ar2 = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(ar1, ar2)
    d = np.c_[xx.ravel(), yy.ravel()]

    # Plot log reg classification
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

    plt.xlabel('')
    plt.ylabel('')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    if input('save pdf (y/n)?') == 'y':
        with PdfPages('yeast_plot.pdf') as pdf:
            fig = plt.figure(1, figsize=(18.5, 10.5))
            # fig.set_size_inches(18.5, 10.5)
            pdf.savefig(fig)

    plt.show()

def ran_forest_yeast_data(X, y):
    rf = RandomForestClassifier(100)
    rf.fit(X, y)
    S = cross_val_score(rf, np.array(X), np.array(y), cv=5)
    print("CV mean score: {0}".format(S.mean()))

if __name__ == '__main__':
    # X1 = process_log_reg_x_data()
    # y1 = process_log_reg_y_data()
    # problem1a(X1, y1)

    # X2 = process_lin_reg_data("q2x.dat")
    # y2 = process_lin_reg_data("q2y.dat")
    # problem1b(X2, y2)
    # estimate_poly_fit(X2, y2)

    (x1, X), y = get_yeast_data()
    print()
    mins, maxs = find_min_max(X)
    X = normalize(X, mins, maxs)
    D = set_to_dict(set(y))
    y = list(map(lambda val: D[val], y))

    # p = 0.8
    # sx = m.ceil(len(X) * p)
    # sy = m.ceil(len(y) * p)
    # print("svm")
    # svm(np.array(X), np.array(y))

    # plot_yeast_data(np.array(X), np.array(y))
    ran_forest_yeast_data(np.array(X), np.array(y))

