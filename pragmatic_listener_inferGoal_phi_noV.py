import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import pymc as pm
try:
    import aesara
    import aesara.tensor as at
except ModuleNotFoundError:
    import pytensor as aesara
    import pytensor.tensor as at
import arviz as az

from modelling_functions_forlistener import (
    normalize, 
    get_data
)

def normalize(x, axis):
    return x / x.sum(axis=axis, keepdims=True)

def softmax(x, axis=1):
    """
    Softmax function in numpy
    Parameters
    ----------
    x: array
        An array with any dimensionality
    axis: int
        The axis along which to apply the softmax
    Returns
    -------
    array
        Same shape as x
    """
    e_x = at.exp(x - at.max(x, axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)
    
def pragmatic_S1(alpha, costs, L, lib='np'):
    if lib == 'at':
        lib = at
    else:
        lib = np
    
    # dimensions (utterance, state)
    L0 = normalize(L,1)

    # p(u | s, phi)
    # where phi is essentially 
    # a politeness weight
    # Dimensions (phi, utterance, state)
    # NOTE: This is the same for all goal conditions
    
    S1 = softmax(
        alpha[:,None,None]*(
            # informational component
            # (value of phi, utterance, state)
            lib.log(L0)
            # (utterance, 1)
            - costs[:,None]
        ),
        # normalize by utterance
        # for each discretized value of phi
        axis=1
    )
    '''
    S1 = normalize(
        lib.exp(
            alpha[:,None,None]*(
            # informational component
            # (value of phi, utterance, state)
            lib.log(L0)
            # (utterance, 1)
            - costs[:,None]
        )),
        # normalize by utterance
        # for each discretized value of phi
        axis=1
    )
    '''
    return S1

def L1_likelihood(alpha, costs, L, lib='np'):   
    S1 = pragmatic_S1(alpha, costs, L, lib)    
    # Prob of state and phi given utterance
    # Dimensions (phi, utterance, state)
    #L1_s_phi_given_w_grid = normalize(
    #    S1,
    #    (0,2)
    #)    
    #L1_s_phi_given_w_grid = S1
    # use the inferred phi 
    # Dimensions: (goal condition, utterance, state)
    # Get the probabilities with speaker's actual phi
    # pm.Deterministic('phi_val', aesara.printing.Print()(phi))
    L1 = normalize(
        S1,
        0
    )
    
    print("L1: ", L1.eval().shape)

    return L1


def pragmatic_L1_model(dt, dt_meaning):
    
    dt_meaning_pymc = (
        dt_meaning
        .groupby(['state', 'utterance_index'])
        ['judgment']
        .agg(['sum', 'count'])
        .reset_index()
    )

    with pm.Model() as L1_model:
        # literal semantic compatibility
        # shape: (utterances, states)
        '''
        L = np.array([
            [0.9911, 0.8413, 0.0538, 0.0077, 1e-10],
            [0.8782, 0.8735, 0.1854, 0.0969, 0.0773],
            [0.03  , 0.0963, 1.    , 0.4227, 0.0136],
            [1e-10, 1e-10, 0.0529, 0.9131, 0.9725],
            [0.1129, 1e-10, 0.006 , 0.8071, 0.9372],
            [0.1623, 0.3711, 0.912 , 0.3387, 0.1944],
            [0.2288, 0.2726, 0.5851, 0.8078, 0.761 ],
            [0.6652, 0.6318, 0.1493, 0.666 , 0.6856],
            [0.5309, 0.8466, 0.6915, 0.4255, 0.202 ],
            [0.5164, 0.6562, 0.87  , 0.447 , 0.4447]
            ])
        '''
        L = pm.Uniform(
            'L',
            lower=0,
            upper=1,
            shape=(10,5)
        )
        
        L_observed = pm.Binomial(
            'L_observed',
            n=dt_meaning_pymc['count'],
            p=L[
                dt_meaning_pymc['utterance_index'],
                dt_meaning_pymc['state']
            ],
            observed=dt_meaning_pymc['sum']
        )
        
        negative_cost = pm.Uniform(
            'c',
            lower=1,
            upper=5
        )

        costs = at.concatenate((
            at.ones(5),
            at.repeat(negative_cost,5)
        ))
        print('<<<<<  costs ', costs.eval())
        alpha = pm.Beta(
            'phi',
            alpha=1,
            beta=1,
            shape=(3)
        )
        '''
        # >= 0 ,*_newp.cdf-alpha~(0,10)
        alpha = pm.Uniform(
            'phi',
            lower=0,
            upper=10,
            shape=(3,)
        )
        '''
        # Shape (condition, utterance, state)
        L1 = L1_likelihood(
            alpha, 
            costs, 
            L, 
            lib='at'
            )
        utterance = pm.Data("utterance", dt.utterance_index)
        state = pm.Data("state", dt.state)
        p_production = L1[
            :,
            utterance,
            state
        ].T
        print('<<<<< p_production ',p_production.eval().shape)
        pm.Categorical(
            "chosen",
            p_production,
            observed=dt.goal_id.values,
            shape=len(dt)
        )
        
    return L1_model


if __name__=='__main__':
    
    dt, utt_i, goal_id, goals, dt_meaning = get_data()
    #dt, utt_i, goal_id, goals, dt_meaning = get_shuffle_data()
    L1_model = pragmatic_L1_model(
        dt,
        dt_meaning
    )
    
    with L1_model:
        start = pm.find_MAP()
        trace = pm.sample(
            draws=30000,
            tune=30000,
            chains=4,
            cores=8,
            initvals=start,
            #init='adapt_diag',
            target_accept=0.99
        )
        ppc = pm.sample_posterior_predictive(trace)    
    az.to_netcdf(
        trace, 
        #'traces_listener_new/L1model_3w_infergoal_56subs_noV_phi.cdf'
        'traces_listener_easy/L1model_3w_infergoal_58subs_noV_XMutable.cdf'
    )
    #az.to_netcdf(ppc, 'ppcs_listener_new/ppc_L1model_3w_infergoal_61subs_noV_XMutable.cdf')
    az.to_netcdf(ppc, 'ppcs_listener_easy/ppc_L1model_3w_infergoal_58subs_noV_XMutable.cdf')
    summary = az.summary(trace,round_to=3)
    print(summary)

    
