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
    

def yoon_S1(values, alpha, costs, L, phi_grid_n=40, lib='np'):

    if lib == 'at':
        lib = at
    else:
        lib = np
    
    # dimensions (utterance, state)
    L0 = normalize(L,1)

    # expected value given each utterance
    # Shape (utterance)
    # NOTE: This only works if L is not
    # exactly 0 anywhere!
    exp_values = lib.mean(
        values*L0,
        axis=1
    )
        
    # using a grid approximation for phi
    phis = np.linspace(0,1,phi_grid_n)

    # p(u | s, phi)
    # where phi is essentially a politeness weight
    # Dimensions (phi, utterance, state)
    # NOTE: This is the same for all goal conditions
    S1 = normalize(
        lib.exp(alpha*(
            # informational component
            # (value of phi, utterance, state)
              phis    [:,None,None]*lib.log(L0)
            # social component
            # (value of phi, utterance, 1)
            + (1-phis)[:,None,None]*exp_values[:,None]
            # (utterance, 1)
            - costs[:,None]
        )),
        # normalize by utterance
        # for each discretized value of phi
        axis=1
    )
    
    return S1


def yoon_utilities(S1, phi, values):        
    # Prob of state and phi given utterance
    # Dimensions (phi, utterance, state)
    L1_s_phi_given_w_grid = normalize(
        S1,
        (0,2)
    )    
    # grid-marginalize over phi
    L1_s_given_w = L1_s_phi_given_w_grid.sum(0)
    
    # informativity of utterances given state
    # with utterances produced by L1
    # Shape (utterance, state)
    u_inf = pm.math.log(L1_s_given_w)

    # expected (value of state)
    # for each utterance as produced by L1
    # NOTE: Same for all goal conditions
    u_soc = at.mean(
        values*L1_s_given_w,
        1
    )

    # u_pres is the only component where phi is not marginalized 
    # and use the inferred phi rather than grid approximation
    
    # Dimensions: (goal condition, utterance, state)
    # Get the probabilities with speaker's actual phi
    # pm.Deterministic('phi_val', aesara.printing.Print()(phi))
    L1_s_phi_given_w = L1_s_phi_given_w_grid[phi]
    
    # print("L1_s_phi_given_w: ", L1_s_phi_given_w.eval())
    
    # equation 3 in 2020 paper
    # (Note that the log doesn't appear in supplementary material!)
    # Shape (goal condition, utterance)
    u_pres = pm.math.log(
        L1_s_phi_given_w
        # marginalize across state
        # to get prob of each utterance given phi
        .sum(2)
    )
          
    #print("U soc: ", u_soc.eval().shape)
    #print("U inf: ", u_inf.eval().shape)
    #print("U pres: ", u_pres.eval().shape)

    # shape (utility component, goal condition, utterance, state)
    utilities = at.stack(
        at.broadcast_arrays(
            # (utterance, state)
            u_inf,
            # (utterance,1)
            u_soc[:,None],
            # (goal condition, utterance, state)
            u_pres[:,:,None]
        ),
        axis=0
    )
    
    return utilities


def yoon_likelihood(alpha, values, omega, costs, phi, L, phi_grid_n=100, lib='np'):

    S1 = yoon_S1(values, alpha, costs, L, phi_grid_n, lib)

    utilities = yoon_utilities(S1, phi, values)
    
    # print("omega: ", (
    #     omega
    # ).eval())

    # shape (goal condition, utterance, state)
    util_total = (
        # shape (goal condition, utterance, state)
        (
            # dims: (utility component, goal condition, 1, 1)
            omega.T[:,:,None,None] 
            # dims: (utility component, goal condition, utterance, state)
            * utilities
        # sum weighted utility components together
        ).sum(0) 
        # (utterance, 1)
        - costs[:,None]
    )

    # print("util_total: ", util_total.eval())
    
    # for each goal condition,
    # prob of utterance given state
    # Shape: (goal_condition, utterance, state)
    S2 = normalize(
        pm.math.exp(alpha*util_total), 
        1
    )
    
    return S2

def L2_likelihood(alpha, values, omega, costs, phi, L, phi_grid_n=100, lib='np'):  
    S2 = yoon_likelihood(
            alpha, 
            values, 
            omega, 
            costs, 
            phi,
            L,
            phi_grid_n=phi_grid_n,
            lib='at'
        )

    # Prob of state and phi given utterance
    # Dimensions (phi, utterance, state)
    L2 = normalize(
        S2,
        0
    )
    
    print("L2: ", L2.eval().shape)
    return L2

def pragmatic_L2_model(dt, dt_meaning):
    '''
    dt_meaning_pymc = (
        dt_meaning
        .groupby(['state', 'utterance_index'])
        ['judgment']
        .agg([np.sum, 'count'])
        .reset_index()
    )
    '''
    dt_meaning_pymc = (
        dt_meaning
        .groupby(['state', 'utterance_index'])
        ['judgment']
        .agg(['sum', 'count'])
        .reset_index()
    ) 

    with pm.Model() as L2_model:
        # literal semantic compatibility
        # shape: (utterances, states)
        '''
        ## L_map from 'L2model_8w_infergoal_58subs_phi_new_XMutable'    
        L = np.array([
            [0.99 , 0.828, 0.172, 0.018, 0.001],
            [0.898, 0.873, 0.108, 0.046, 0.034],
            [0.022, 0.374, 0.999, 0.294, 0.088],
            [1e-10, 1e-10, 0.226, 0.623, 0.978],
            [1e-10, 1e-10, 0.214, 0.606, 0.981],
            [0.024, 0.178, 0.831, 0.791, 0.581],
            [0.099, 0.128, 0.771, 0.829, 0.684],
            [0.688, 0.608, 0.112, 0.703, 0.657],
            [0.688, 0.789, 0.835, 0.293, 0.012],
            [0.512, 0.516, 0.823, 0.854, 0.026]
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
        
        #print(dt_meaning_pymc)
        #print(L_observed.eval(),L_observed.eval().shape)
        '''
        negative_cost = pm.Uniform(
            'c',
            lower=1,
            upper=5
        )
        '''
        negative_cost = pm.Gamma(
            'c',
            alpha=1,
            beta=1
        )
        
        costs = at.concatenate((
            at.ones(5),
            #at.ones(5)
            at.repeat(negative_cost, 5)
        ))
        
        # >= 0 
        alpha1 = pm.Uniform(
            'alpha1',
            lower=0,
            upper=20
        )
        '''
        alpha1 = pm.Gamma(
            'alpha1',
            alpha=1, 
            beta=1
        ) 
        
        beta1 = pm.Uniform(
            'beta1',
            lower=0,
            upper=10
        )
        '''
        beta1 = pm.Gamma(
            'beta1',
            alpha=1,
            beta=1
        )
        values = beta1 * np.array([0, 1, 2, 3, 4])
        
        #values = np.array([0, 1, 2, 3, 4])
        
        # goal weights
        # [informational, prosocial, presentational]
        # A triplet of goal weights for each goal condition!
        # Shape: (goal condition, utility component)
        omega = pm.Dirichlet(
            'omega',
            [1,1,1],
            shape=(3,3)
        )
        print('<<<< omega',omega.eval())
        phi_grid_n = 100

        # politeness weight
        # One for each goal condition!
        # Shape: (goal condition)
        phi = pm.Categorical(
            'phi',
            p=np.ones(phi_grid_n)/phi_grid_n,
            shape=(3)
        )

        # Shape (condition, utterance, state)
        L2 = L2_likelihood(
            alpha1, 
            values, 
            omega, 
            costs, 
            phi, 
            L, 
            phi_grid_n=phi_grid_n,
            lib='at'
            )
        # each combination of goal and state
        # should give a prob vector over utterances
        # print(S2.eval().sum((1)))
        #print('<<<<<<<<< L', L)
        #print('<<<<<<<< dt', dt)
        #print('<<<<<<<< L_observed ',L_observed.eval(),L_observed.eval().shape)
        utterance = pm.Data("utterance", dt.utterance_index)
        state = pm.Data("state", dt.state)
        p_production = L2[
            :,
            utterance,
            state
        ].T
        print('<<<<< p_production ',p_production.eval().shape)
        #goal_id = pm.Data("goal_id", dt.goal_id)
        pm.Categorical(
            "chosen",
            p_production,
            observed=dt.goal_id.values,
            shape=len(dt)
        )
        
    return L2_model


if __name__=='__main__':
    
    dt, utt_i, goal_id, goals, dt_meaning = get_data()
    #dt, utt_i, goal_id, goals, dt_meaning = get_shuffle_data()
    L2_model = pragmatic_L2_model(
        dt,
        dt_meaning
    )
    
    with L2_model:
        start = pm.find_MAP()
        trace = pm.sample(
            draws=80000,
            tune=80000,
            chains=4,
            cores=8,
            initvals=start,
            #init='adapt_diag',
            target_accept=0.99
        )
        ppc = pm.sample_posterior_predictive(trace)    
    az.to_netcdf(
        trace, 
        #'traces_listener_new/L2model_16w_infergoal_56subs_phi_new_XMutable_sameC.cdf'
        'traces_listener_easy/L2model_8w_infergoal_58subs_phi_new_XMutable_alphauni.cdf'
    )
    #az.to_netcdf(ppc, 'ppcs_listener_new/ppc_L2model_16w_infergoal_56subs_phi_new_XMutable_sameC.cdf')
    az.to_netcdf(ppc, 'ppcs_listener_easy/ppc_L2model_8w_infergoal_58subs_phi_new_XMutable_alphauni.cdf')
    summary = az.summary(trace,round_to=3)
    print(summary)
