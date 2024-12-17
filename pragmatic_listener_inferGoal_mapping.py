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
    
#def mapping_L0(lambda_, L, lib='np'):
def mapping_L0(alpha, costs, L, lib='np'):
    if lib == 'at':
        lib = at
    else:
        lib = np
    
    # dimensions (goal, utterance, state)
    #L0 = lambda_[:,None,None] * normalize(L,1)
    L0 = normalize(
            alpha[:,None,None] * L - costs[:,None],
            0
            )
    print("L0: ", L0.eval().shape)
    return L0

def mapping_L0_model(dt, dt_meaning):
    
    dt_meaning_pymc = (
        dt_meaning
        .groupby(['state', 'utterance_index'])
        ['judgment']
        .agg([np.sum, 'count'])
        .reset_index()
    )

    with pm.Model() as L1_model:
        # literal semantic compatibility
        # shape: (utterances, states)
        '''
        L = np.array([
            [0.99  , 0.86  , 0.07  , 0.01  , 1e-10],
            [0.88  , 0.87  , 0.19  , 0.1   , 0.08  ],
            [1e-10, 0.15  , 1.    , 0.35  , 0.01  ],
            [1e-10, 0.01  , 0.18  , 0.94  , 0.96  ],
            [1e-10, 1e-10, 0.06  , 0.69  , 0.98  ],
            [0.01  , 0.22  , 0.85  , 0.75  , 0.6   ],
            [0.12  , 0.21  , 0.75  , 0.8   , 0.73  ],
            [0.73  , 0.67  , 0.06  , 0.64  , 0.65  ],
            [0.69  , 0.82  , 0.83  , 0.21  , 0.01  ],
            [0.62  , 0.7099, 0.85  , 0.74  , 0.03  ]
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

        '''
        lambda_ = pm.Beta(
            'lambda',
            alpha=1,
            beta=1,
            shape=(3)
        )
        '''
        alpha = pm.Beta(
            'alpha',
            alpha=1,
            beta=1,
            shape=(3)
        )
        #beta = pm.Normal('beta', mu=0, sigma=10)
        # Shape (condition, utterance, state)
        L1 = mapping_L0(
            alpha, 
            costs,
            L, 
            lib='at'
            )
        
        utterance = pm.MutableData("utterance", dt.utterance_index)
        state = pm.MutableData("state", dt.state)
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
    L1_model = mapping_L0_model(
        dt,
        dt_meaning
    )
    
    with L1_model:
        start = pm.find_MAP()
        trace = pm.sample(
            draws=20000,
            tune=20000,
            chains=4,
            cores=8,
            initvals=start,
            #init='adapt_diag',
            target_accept=0.99
        )
        ppc = pm.sample_posterior_predictive(trace)    
    az.to_netcdf(
        trace, 
        #'traces_listener_new/L1model_2w_infergoal_56subs_mapping.cdf'
        'traces_listener_easy/L1model_2w_infergoal_58subs_mapping.cdf'
    )
    #az.to_netcdf(ppc, 'ppcs_listener_new/ppc_L1model_2w_infergoal_56subs_mapping.cdf')
    az.to_netcdf(ppc, 'ppcs_listener_easy/ppc_L1model_2w_infergoal_58subs_mapping.cdf')
    summary = az.summary(trace,round_to=3)
    print(summary)

    
