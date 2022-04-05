import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from ipywidgets import VBox, HBox, Output, Button, IntText, Label, FloatText

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.collections import PathCollection
import matplotlib.animation as animation


def targetdist(x):
    probX = np.exp(-x**2) * (2 + np.sin(x*5) + np.sin(x*2))
    return probX


x = np.arange(-3, 3, 0.01)
y = targetdist(x)
px.line(x=x, y=y)


def run_mcmc(caller):
    out.clear_output()
    display_label.value = 'Running MCMC'
    ### parameters ###
    burnin = burin_input.value  # number of burn-in iterations
    lag = lag_input.value  # iterations between successive samples
    nsamp = nsamp_input.value  # number of samples to draw
    sig = sig_input.value  # standard deviation of Gaussian proposal
    x = start_point_input.value  # start point
    ### storage ###
    X = np.zeros((nsamp, 1))  # samples drawn from the Markov chain
    acc = np.array((0, 0))  # vector to track the acceptance rate

    def MHstep(x0, sig):
        # generate candidate from Gaussian
        xp = np.random.normal(loc=x0, scale=sig)
        accprob = targetdist(xp) / targetdist(x0)  # acceptance probability
        u = np.random.rand()  # uniform random number
        if u <= accprob:  # if accepted
            x1 = xp  # new point is the candidate
            a = 1  # note the acceptance
        else:  # if rejected
            x1 = x0  # new point is the same as the old one
            a = 0  # note the rejection
        return x1, a

    def targetdist(x):
        probX = np.exp(-x**2) * (2 + np.sin(x*5) + np.sin(x*2))
        return probX

    # MH routine
    for i in range(burnin):
        x, a = MHstep(x, sig)  # iterate chain one time step
        acc = acc + np.array((a, 1))  # track accept-reject status

    for i in range(nsamp):
        for j in range(lag):
            x, a = MHstep(x, sig)  # iterate chain one time step
            acc = acc + np.array((a, 1))  # track accept-reject status
        X[i] = x  # store the i-th sample
    df = pd.DataFrame(data=X, columns=['Trace'])

    display_label.value = 'Average Acceptance: ' + \
        str(round(acc[0] / acc[1], 2))

    with out:
        fig, axs = plt.subplots(2, 1)

        axs[0].hist(df.values, bins=20)
        axs[1].plot(df)

        r = Affine2D().rotate_deg(90)

        fig = plt.gcf()
        fig.set_size_inches(8, 6)

        for x in axs[1].images + axs[1].lines + axs[1].collections:
            trans = x.get_transform()
            x.set_transform(r+trans)
            if isinstance(x, PathCollection):
                transoff = x.get_offset_transform()
                x._transOffset = r+transoff

        old = axs[1].axis()
        axs[1].axis(old[2:4] + old[0:2])

        plt.show()
