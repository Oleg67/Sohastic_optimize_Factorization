import numpy as np
import pandas as pd

from thbmodel.utils.accumarray import accum, unpack
from thbmodel.utils import get_logger
from thbmodel.prediction.tools.helpers import strata_scale_down

import matplotlib.pyplot as plt


logger = get_logger(__package__)

def softmax(x, axis =1):
    """
    x <= [n,i] tensor 
    axis <= dimention for sum in softmax
    in case np.inf/np.inf = 1.
    return
    sf <= [n,i] tensor answer softmax along axis 
    """

    v_exp = np.exp(x)
    if axis ==1:
        sf = np.divide(v_exp, np.nansum(v_exp, axis =axis).reshape(-1,1))
    else:
        sf = np.divide(v_exp, np.nansum(v_exp, axis =axis))
    mask = np.isnan(sf)
    if mask.any():
        return np.where(mask, 1., sf)
    return sf
    
def G_step(df_t, winner, t_len, w, verbose=False):
    """ 
    Gradient step 
    return gradient 
    
    df <= [n,i,j] 3d tensor of datas (n) event, (i) -horse, (j) factor
    winner <=  [n,i] 2d tensor (matrix) (n) event, (i) -horse, value 1 if won event else 0
    t_len <= [n] 1d tensor (vector) (n) event, value number of horses in event
    w <=  Model's parameters
    verbose <=  flaf debaging
    """
    b_size,h,f = df_t.shape
    
    S = np.einsum('nij, j -> ni', df_t,w)
    S = np.where(np.isnan(S), -np.inf, S)
    p = softmax(S) 
    delta = winner - p
    
    grad_w = np.zeros_like(w)
    x = np.where(np.isnan(df_t), 0, df_t)
    grad_w = np.einsum('ni, nij -> j', delta, x)
      
    if verbose:
        print 'shape grad_w ',grad_w.shape
        print 'grad_w  ',grad_w[:10]
            
    
    return -np.divide(grad_w,b_size)
        

def Adam (data, features,  w = None, b_size =100, eta=1e-3, N_epoh =10,
                                moment1 = 0.9, moment2 = 0.99, la = 10., max_iter=4e2,
                                eps = 1e-4,  min_weight_dist=1e-3, seed=42, verbose=False):
    """ 
    Adam optimization method for liniar Model
    for each epoh choice  b_size events and run all iteration
    repeat for next epoh
    
    data <= DataFrame len(features)-columns of data, the column of - event_id, 
            the column of- result
    features <=  names of datas 
    w <=  initial parameters of the Model
    b_size  <=  batch size
    eta <=  learning step
    N_epoh <= number of epohs
    moment1 <=  first moment
    moment2 <=  second moment
    la <=  regularization constant
    max_iter <=  maximum of iterations 
    eps <=  approximantely zero 
    min_weight_dist <=  distanse for covergence
    seed <=  random constant 
    verbose <=  flag debaging
    """

    
    # first distance beetwene weight parameters
    weight_dist = np.inf
    # for repeating of results
    #np.random.seed(seed)
    if w == None:
        w = np.random.normal(0, scale =1e-2, size =(len(features)))
    
    # store of grad changes
    grad_norm = np.zeros(max_iter * N_epoh +1)
    # last gradient value
    last_grad = np.zeros_like(w) 
    # last gradient's variation value
    std_grad = np.zeros_like(w)
    # unique events in data set
    events_list = np.unique(data['event_id'])
        
    # Main loop
    for i in range(N_epoh):
        # random choice the events size = b_size
        r_event = np.random.choice(events_list, size = b_size)
        df = data.loc[np.in1d(data['event_id'], r_event),:]
        
        df_t, winner_t, t_len = DF_to_tensor(df, features)
        
        # last gradient value
        #last_grad = np.zeros_like(w) 
        # last gradient's variation value
        #std_grad = np.zeros_like(w)
        
        for iter_num in range(1, max_iter+1):
        
            w_N = w - eta *moment1 *last_grad # for Nesterov momentum
            # get gradient with  Nesterov momemtum + L2 regularisation
            
            grad_ = G_step(df_t, winner_t, t_len, w_N, verbose=False) + la * w_N # gradient and L2
            #grad_ = G_step(df_t, winner_t, t_len, w, verbose=False) + la * w # gradient and L2
            last_grad = moment1 * last_grad + (1-moment1) * grad_ # Update weight first moment estimate 
            std_grad = moment2 * std_grad + (1-moment2)* grad_ * grad_ # Update weight second moment estimate 
            
            t = i * max_iter + iter_num
            last_grad_ = last_grad/(1 - np.power(moment1, t)) # Correct first moment estimate
            std_grad_ = std_grad/(1 - np.power(moment2, t)) # Correct second moment estimate
        
        
            w = w - eta *last_grad_/(np.square(std_grad_) + eps)
        
            weight_dist = np.linalg.norm(last_grad, ord=2)/len(w)
            grad_norm [t] = weight_dist 
            
            if (weight_dist <= min_weight_dist):
                print 'SGD covergence', t
                break
                #return w, grad_norm , iter_num
        
        
        
            if verbose:
                #print 'w  ', w
                print 'iteration', iter_num, 'dist ',weight_dist
        
            if np.any(np.abs(w) == np.inf):
                print "w overcome"
                break
        
    return w, grad_norm ,t    

def AdaMax(data, features,  w = None, b_size =100, eta=1e-3, N_epoh =10,
                                moment1 = 0.9, moment2 = 0.99, la = 10., max_iter=100,
                                eps = 1e-4,  min_weight_dist=1e-3, seed=42, verbose=False):
    """ 
    Adam Max optimization method for liniar Model
    for each epoh choice  b_size events and run all iteration
    repeat for next epoh
    
    data <= DataFrame len(features)-columns of datas, the column of - event_id, 
            the column of- result
    features <=  names of datas 
    w <=  initial parameters of the Model
    b_size  <=  batch size
    eta <=  learning step
    N_epoh <= number of epohs
    moment1 <=  first moment
    moment2 <=  second moment
    la <=  regularization constant
    max_iter <=  maximum of iterations 
    eps <=  approximantely zero 
    min_weight_dist <=  distanse for covergence
    seed <=  random constant 
    verbose <=  flag debaging
    """

    
    # first distance beetwene weight parameters
    weight_dist = np.inf
    # for repeating of results
    #np.random.seed(seed)
    if w == None:
        w = np.random.normal(0, scale =1e-2, size =(len(features)))
    
    # store of grad changes
    grad_norm = np.zeros(max_iter * N_epoh +1)
    # last gradient value
    last_grad = np.zeros_like(w) 
    # last gradient's variation value
    std_grad = np.zeros_like(w)
    # unique events in data set
    events_list = np.unique(data['event_id'])
        
    # Main loop
    for i in range(N_epoh):
        # random choice the events size = b_size
        r_event = np.random.choice(events_list, size = b_size)
        df = data.loc[np.in1d(data['event_id'], r_event),:]
        
        df_t, winner_t, t_len = DF_to_tensor(df, features)
        # last gradient value
        #last_grad = np.zeros_like(w) 
        # last gradient's variation value
        #std_grad = np.zeros_like(w)

        
        for iter_num in range(1, max_iter+1):
        
            w_N = w - eta *moment1 *last_grad # for Nesterov momentum
            # get gradient with  Nesterov momemtum + L2 regularisation
            grad_ = G_step(df_t, winner_t, t_len, w_N, verbose=False) + la * w_N # gradient and L2
            #grad_ = G_step(df_t, winner_t, t_len, w, verbose=False) + la * w # gradient and L2
            last_grad = moment1 * last_grad + (1-moment1) * grad_ # Update weight first moment estimate 
            std_grad = np.max(np.vstack((moment2 * std_grad, np.abs(grad_))), axis =0) # Update weight second moment estimate 
        
            t = i * max_iter + iter_num
            last_grad_ = last_grad/(1 - np.power(moment1, t)) # Correct first moment estimate
            
            w = w -  eta *last_grad_/std_grad
        
            weight_dist = np.linalg.norm(last_grad, ord=2)/len(w)
            grad_norm [t] = weight_dist
            #print "t",t
            if (weight_dist <= min_weight_dist):
               # print 'SGD covergence', t
                break
                #return w, grad_norm , t
        
            if verbose:
                #print 'w  ', w
                print 'epoh ', i, ' iteration ', t, ' dist ',weight_dist
        
            if np.any(w == np.nan):
                print "w overcome"
                break
    
    return w, grad_norm, t
	

def DF(mask, factors, av, factors_names, other_names):
	"""
	create DataFrame from av datas
	"""
	df = pd.DataFrame(data =factors[:, mask].T , columns = factors_names)

	for col in other_names :
		df[col] = av[col][mask]
	
	return df


def LL_t (df_t, winner, w):
	"""
	return likelihood 
	df_t <= 3d tensor of datas
	winner <= 2d tensor of winners
	w <= 1d tensor parameters model
	"""

	S = np.einsum('nij, j -> ni', df_t,w) 
	S = np.where(np.isnan(S), -np.inf, S)
	p = softmax(S)  
	LL = np.log(np.einsum('ni, ni -> n', p, winner))
	LL = LL[np.isfinite(LL)]

	return np.nanmean(LL)
    

def G_step_factorize_T(df_t, winner, t_len, w, V, verbose=False):
    """ 
    Gradient step 
    return gradient 
    
    df <= [n,i,j] 3d tensor of datas (n) event, (i) -horse, (j) factor
    winner <=  [n,i] 2d tensor (matrix) (n) event, (i) -horse, value 1 if won event else 0
    t_len <= [n] 1d tensor (vector) (n) event, value number of horses in event
    w <=  [j] Model's parameters
    V <= [j,r] Model's parameters
    verbose <=  flaf debaging
    """
    b_size,h,f = df_t.shape
    
    S = np.einsum('nij, j -> ni', df_t,w) + np.einsum('nij, jr, kr, nik  -> ni', df_t,V,V,df_t)/2 - \
                np.einsum('nij, jr, jr, nij  -> ni', df_t,V,V,df_t)/2
    S = np.where(np.isnan(S), -np.inf, S)
    p = softmax(S) 
            
    delta = winner - p
    #dS_by_dV = np.einsum('nij,kr, nik -> nijr', df_t, V, df_t) - np.einsum('jr,nij -> nijr', V, df_t *df_t)
    
    grad_w = np.zeros_like(w)
    grad_V = np.zeros_like(V)
    
    x = np.where(np.isnan(df_t), 0, df_t)
    grad_w = np.einsum('ni, nij -> j', delta, x)
    dS_by_dV = np.einsum('nij,kr, nik -> nijr', x, V, x) - np.einsum('jr,nij -> nijr', V, x *x)
    #dS_by_dV = np.where(np.isnan(dS_by_dV), 0, dS_by_dV)
    grad_V = np.einsum('ni, nijr -> jr', delta, dS_by_dV)
        
    if verbose:
        print 'shape grad_w ',grad_w.shape
        print 'grad_w  ',grad_w[:10]

    if verbose:
        print 'shape grad_V  ',grad_V.shape
        print 'grad_V  ',grad_V[:10,0]
        print '---------------------------------------'
    
    return -np.divide(grad_w, b_size), -np.divide(grad_V, b_size)
    

def AdaMax_factorize_T(data, features,  w = None, V = None, b_size =1000, eta=1e-3, N_epoh =10, rang =2,
                                moment1 = 0.9, moment2 = 0.99, la = 0.8, max_iter=100,
                                eps = 1e-4,  min_weight_dist=1e-3, seed=42, verbose=False):
    """ 
    Adam Max optimization method for tne factorization Model
    for each epoh choice  b_size events and run all iteration
    repeat for next epoh
    
    data <= DataFrame len(features)-columns of datas, the column of - event_id, 
            the column of- result
    features <=  names of datas 
    w <=  initial parameters of the liniar Model 
    V <=  initial parameters of the factorization Model
    b_size  <=  batch size
    eta <=  learning step
    N_epoh <= number of epohs
    moment1 <=  first moment
    moment2 <=  second moment
    la <=  regularization constant
    max_iter <=  maximum of iterations 
    eps <=  approximantely zero 
    min_weight_dist <=  distanse for covergence
    seed <=  random constant 
    verbose <=  flag debaging
    """

    
    # first distance beetwene weight parameters
    weight_dist = np.inf
    # for repeating of results
    #np.random.seed(seed)
    if w == None:
        w = np.random.normal(0, scale =1e-2, size =(len(features)))
        #w = np.ones((len(features)))* 0.01
        
    if V == None:
        V = np.random.normal(0, scale =1e-2, size =(len(features), rang))
        #V = np.ones((len(features), rang)) * 0.01
    n,m = V.shape
    #common vector parameters
    w_par = np.vstack((w.reshape(1,-1), V.T)).flatten()
    # Step
    
    # store of grad changes
    grad_norm = np.zeros(max_iter * N_epoh +1 +1)
    # last gradient value
    last_grad = np.zeros_like(w_par) 
    # last gradient's variation value
    std_grad = np.zeros_like(w_par)
    # unique events in data set
    events_list = np.unique(data['event_id'])
        
    # Main loop
    for i in range(N_epoh):
        # random choice the events size = b_size
        r_event = np.random.choice(events_list, size = b_size)
        df = data.loc[np.in1d(data['event_id'], r_event),:]
        
        df_t, winner_t, t_len = DF_to_tensor(df, features)
    
        
        for iter_num in range(1, max_iter+1):
        
            w_par_N = w_par- eta* moment1 *last_grad # for Nesterov momentum

            # get gradient with  Nesterov momemtum + L2 regularisation
            w = w_par_N[:n] # return to original shape
            V = w_par_N[n:].reshape(m, n).T
            #w = w_par[:n] # return to original shape
            #V = w_par[n:].reshape(m, n).T
        
            grad_values = G_step_factorize_T(df_t, winner_t, t_len, w, V, verbose=False) 
            
            grad_ = np.vstack((grad_values[0].reshape(1,-1), grad_values[1].T)).flatten() + la * w_par_N # gradient and L2
                
            last_grad = moment1 * last_grad + (1-moment1) * grad_ # Update weight first moment estimate 
            std_grad = np.max(np.vstack((moment2 * std_grad, np.abs(grad_))), axis =0) # Update weight second moment estimate 
            
            t = i * max_iter + iter_num
            last_grad_ = last_grad/(1 - np.power(moment1, t)) # Correct first moment estimate
        
            grad_step = last_grad_/std_grad
            w_par = w_par - eta * grad_step
        
            weight_dist = np.linalg.norm(grad_step, ord=2)/len(w_par)
            grad_norm [t] = weight_dist 
        
            if (weight_dist <= min_weight_dist):
                print 'SGD covergence', t
                break
        
            if verbose:
                #print 'w  ', w
                if (t%(max_iter/2)) ==0:
                    print 'epoh {0}  iteration  {1}  dist {2}  V {3}'.format(i, t, weight_dist, w_par[54:56])

        
            if np.any((w == np.inf)|(w == -np.inf)):
                print "w overcome"
                return w, grad_norm , t
        
    return w_par, grad_norm , t
    

def DF_to_tensor(df, features, max_horses =60):
    """
    turn DataFrame to Tensor 
    df <= DataFrame with (features) columns of datas, (result) column, and (event_id) colunm 
    max_horses <= maximum of number of horses in event
    return 
    t_Horses_factors <= [n,i,j] 3d tensor of datas (n) event, (i) -horse, (j) factor
    t_winner <=  [n,i] 2d tensor (matrix) (n) event, (i) -horse, value 1 if won event else 0
    t_length <= [n] 1d tensor (vector) (n) event, value number of horses in event
    """
    import pandas as pd

    length = df.event_id.unique().__len__()
    f = len(features)
    t_Horses_factors = np.full((length, max_horses, f), np.nan, dtype ='f8')
    t_winner = np.zeros((length, max_horses), dtype ='i4')
    t_length = np.zeros(length, dtype ='i4')
    for  n, (_,df_sub) in enumerate(df.groupby(df['event_id'])):
        h = df_sub.loc[:,features].values.shape[0]
        t_Horses_factors[n, :h,:] = df_sub.loc[:,features].values
        t_winner[n, :h] = (df_sub['result'] ==1).values.astype(int)
        t_length[n] = h
    
    return t_Horses_factors, t_winner, t_length


def Adam_factorize_T (data, features,  w = None, V = None, rang = 2, b_size =1000, eta=1e-3, 
                                moment1 = 0.9, moment2 = 0.99, la = 0.8, max_iter=100, N_epoh = 10,
                                eps = 1e-4,  min_weight_dist=1e-3, seed=42, verbose=False):
    """ 
    Adam  optimization method for tne factorization Model
    for each epoh choice  b_size events and run all iteration
    repeat for next epoh
    
    data <= DataFrame len(features)-columns of datas, the column of - event_id, 
            the column of- result
    features <=  names of datas 
    w <=  initial parameters of the liniar Model 
    V <=  initial parameters of the factorization Model
    b_size  <=  batch size
    eta <=  learning step
    N_epoh <= number of epohs
    moment1 <=  first moment
    moment2 <=  second moment
    la <=  regularization constant
    max_iter <=  maximum of iterations 
    eps <=  approximantely zero 
    min_weight_dist <=  distanse for covergence
    seed <=  random constant 
    verbose <=  flag debaging
    """

    
    # first distance beetwene weight parameters
    weight_dist = np.inf
    # for repeating of results
    #np.random.seed(seed)
    if w == None:
        w = np.random.normal(0, scale =1e-2, size =(len(features)))
        #w = np.ones((len(features)))* 0.01
        
    if V == None:
        V = np.random.normal(0, scale =1e-2, size =(len(features), rang))
        #V = np.ones((len(features), rang)) * 0.01
    n,m = V.shape
    #common vector parameters
    w_par = np.vstack((w.reshape(1,-1), V.T)).flatten()
    # Step
    
    # store of grad changes
    grad_norm = np.zeros(max_iter * N_epoh +1 +1)
    # last gradient value
    last_grad = np.zeros_like(w_par) 
    # last gradient's variation value
    std_grad = np.zeros_like(w_par)
    # unique events in data set
    events_list = np.unique(data['event_id'])
        
    # Main loop
    for i in range(N_epoh):
        # random choice the events size = b_size
        r_event = np.random.choice(events_list, size = b_size)
        df = data.loc[np.in1d(data['event_id'], r_event),:]
        
        df_t, winner_t, t_len = DF_to_tensor(df, features)
    
        
        for iter_num in range(1, max_iter+1):
        
            w_par_N = w_par- eta* moment1 *last_grad # for Nesterov momentum

            # get gradient with  Nesterov momemtum + L2 regularisation
            w = w_par_N[:n] # return to original shape
            V = w_par_N[n:].reshape(m, n).T
            #w = w_par[:n] # return to original shape
            #V = w_par[n:].reshape(m, n).T
        
            grad_values = G_step_factorize_T(df_t, winner_t, t_len, w, V, verbose=False) 
            # gradient and L2
            grad_ = np.vstack((grad_values[0].reshape(1,-1), grad_values[1].T)).flatten() +la *w_par_N 
                
            last_grad = moment1 * last_grad + (1-moment1)* grad_ # Update weight first moment estimate 
            std_grad = moment2 * std_grad + (1-moment2)* grad_ * grad_ # Update weight second moment estimate 
            
            t = i * max_iter + iter_num
            last_grad_ = last_grad/(1 - np.power(moment1, t)) # Correct first moment estimate
            std_grad_ = std_grad/(1 - np.power(moment2, t)) # Correct second moment estimate
        
            grad_step = last_grad_/(np.square(std_grad_) + eps)
            w_par = w_par -eta * grad_step
        
            weight_dist = np.linalg.norm(grad_step, ord=2)/len(w_par)
            grad_norm [t] = weight_dist 
            
            if (weight_dist <= min_weight_dist):
                print 'SGD covergence', i
                break
                
        
            if verbose:
                #print 'w  ', w
                if (t%(max_iter/2)) ==0:
                    print 'epoh {0}  iteration  {1}  dist {2}  V {3}'.format(i, t, weight_dist, w_par[54:56])
        
            if np.any((w_par == np.inf)|(w_par == -np.inf)):
                print "w overcome"
                return w, grad_norm , t
        
    return w_par, grad_norm , t

 
def AdaMax_iter_time(data, features,  w = None, b_size =100, eta=1e-3, 
                                moment1 = 0.85, moment2 = 0.99, la = 10., N_epoh =10, direction =False,
                                eps = 1e-4,  min_weight_dist=1e-3, seed=42, verbose=False):
    """ 
    Adam  Max optimization method for the Model
    for each epoh run all events by batch size  =b_size
    repeat for next epoh
    
    data <= DataFrame len(features)-columns of datas, the column of - event_id, 
            the column of- result
    features <=  names of datas 
    w <=  initial parameters of the liniar Model 
    b_size  <=  batch size
    eta <=  learning step
    N_epoh <= number of epohs
    direction <= direction of run (True) from old to new
                 (False) from new to old
    moment1 <=  first moment
    moment2 <=  second moment
    la <=  regularization constant
    max_iter <=  maximum of iterations 
    eps <=  approximantely zero 
    min_weight_dist <=  distanse for covergence
    seed <=  random constant 
    verbose <=  flag debaging
    """
    
    # first distance beetwene weight parameters
    weight_dist = np.inf
    # for repeating of results
    #np.random.seed(seed)
    if w == None:
        w = np.random.normal(0, scale =1e-2, size =(len(features)))
        #w = np.ones((len(features)))* 0.01
    
    # Step
    iter_num = 0
    
    # last gradient value
    last_grad = np.zeros_like(w) 
    # last gradient's variation value
    std_grad = np.zeros_like(w)
    # unique events in data set
    events_list = np.unique(data['event_id'])
    step_max = len (events_list)/b_size
    # store of grad changes
    grad_norm = np.zeros(N_epoh *step_max +1)
    # Main loop
    for  i_ep  in range(1, N_epoh+1):
                
        for step_ in range(1, step_max +1):
            
            if direction:
                r_event = events_list[b_size*(step_-1): b_size*(step_)]
            else:
                if step_ ==1:
                     r_event = events_list[-b_size*(step_): ]
                else:
                     r_event = events_list[-b_size*(step_): -b_size*(step_-1)]
            df = data.loc[np.in1d(data['event_id'], r_event),:]
            
            df_t, winner_t, t_len = DF_to_tensor(df, features)
            
            w_N = w - eta *moment1 *last_grad # for Nesterov momentum
            # get gradient with  Nesterov momemtum + L2 regularisation
            grad_ = G_step(df_t, winner_t, t_len, w_N, verbose=False) + la * w_N  # gradient and L2
            #grad_ = G_step(df_t, winner_t, t_len, w, verbose=False) + la * w  # gradient and L2
            last_grad = moment1 * last_grad + (1-moment1) * grad_ # Update weight first moment estimate
            # Update weight second moment estimate exponent power infinity
            std_grad = np.max(np.vstack((moment2 * std_grad, np.abs(grad_))), axis =0) # Update weight second moment estimate 
            
            t = (i_ep -1)*step_max + step_
            last_grad_ = last_grad/(1 - np.power(moment1, t)) # Correct first moment estimate
        
            w = w - eta *last_grad_/std_grad
        
            weight_dist = np.linalg.norm(last_grad, ord=2)/len(w)
            grad_norm [t] = weight_dist
            
            if (weight_dist <= min_weight_dist):
                print 'SGD covergence', t
                break
                return w, grad_norm , iter_num*step_
        
            if verbose:
                #print 'w  ', w
                print 'iteration', iter_num, 'dist ',weight_dist
         
            if np.any((w == np.inf ) |(w == -np.inf)):
                print "w overcome"
                break
        
    return w, grad_norm , t   
    
    
def AdaMax_time_step(data, features,  w = None, b_size =100, eta=1e-3, 
                                moment1 = 0.85, moment2 = 0.99, la = 10., max_iter =10,
                                eps = 1e-4,  min_weight_dist=1e-3, seed=42, verbose=False):
    """ 
    Adam  Max optimization method for the Model
    for each batch of event run desent for max_iter
    repeat for next batch that older then last
    
    data <= DataFrame len(features)-columns of datas, the column of - event_id, 
            the column of- result
    features <=  names of datas 
    w <=  initial parameters of the liniar Model 
    b_size  <=  batch size
    eta <=  learning step
    N_epoh <= number of epohs
    moment1 <=  first moment
    moment2 <=  second moment
    la <=  regularization constant
    max_iter <=  maximum of iterations 
    eps <=  approximantely zero 
    min_weight_dist <=  distanse for covergence
    seed <=  random constant 
    verbose <=  flag debaging
    """
    
    # first distance beetwene weight parameters
    weight_dist = np.inf
    # for repeating of results
    #np.random.seed(seed)
    if w == None:
        w = np.random.normal(0, scale =1e-2, size =(len(features)))
        #w = np.ones((len(features)))* 0.01
    
    # Step
    step_ = 0
    
    # last gradient value
    last_grad = np.zeros_like(w) 
    # last gradient's variation value
    std_grad = np.zeros_like(w)
    # unique events in data set
    events_list = np.unique(data['event_id'])
    step_max = len (events_list)/b_size
    # store of grad changes
    grad_norm = np.zeros(max_iter *step_max +1)
    # Main loop
    for step_ in range (1, step_max +1):
        
        r_event = events_list[-b_size*(step_ +1): -b_size*step_]
        df = data.loc[np.in1d(data['event_id'], r_event),:]
        
        df_t, winner_t, t_len = DF_to_tensor(df, features)
        for iter_num  in range(1, max_iter +1):
            
            w_N = w - eta *moment1 *last_grad # for Nesterov momentum
            # get gradient with  Nesterov momemtum + L2 regularisation
            grad_ = G_step(df_t, winner_t, t_len, w_N, verbose=False) + la * w_N  # gradient and L2 
            #grad_ = G_step(df_t, winner_t, t_len, w, verbose=False) + la * w  # gradient and L2 
            last_grad = moment1 * last_grad + (1-moment1) * grad_ # Update weight first moment estimate
            # Update weight second moment estimate expanent power infinity
            std_grad = np.max(np.vstack((moment2 * std_grad, np.abs(grad_))), axis =0) # Update weight second moment estimate 
        
            t = max_iter *(step_-1) + iter_num
            last_grad_ = last_grad/(1 - np.power(moment1, t)) # Correct first moment estimate
        
            w = w - eta *last_grad_/std_grad
        
            weight_dist = np.linalg.norm(last_grad, ord=2)/len(w)
            grad_norm [t]   = weight_dist
            
        if (weight_dist <= min_weight_dist):
            print 'SGD covergence', t
            break
            return w, grad_norm , iter_num*step_
        
        if verbose:
            #print 'w  ', w
            print 'iteration', iter_num, 'dist ',weight_dist
        
        if np.any(w == np.nan):
            print "w overcome"
            break
        
    return w, grad_norm , t

def LL_t_f (df_t, winner, w, V):
	"""
	return likelihood 
	df_t <= 3d tensor of datas
	winner <= 2d tensor of winners
	w <= 1d tensor parameters model
	"""
	S = np.einsum('nij, j -> ni', df_t,w) + np.einsum('nij, jr, kr, nik  -> ni', df_t,V,V,df_t)/2 - \
			np.einsum('nij, jr, jr, nij  -> ni', df_t,V,V,df_t)/2
	S = np.where(np.isnan(S), -np.inf, S)
	p = softmax(S)
	LL = np.log(np.einsum('ni, ni -> n', p, winner))
	LL = LL[np.isfinite(LL)]

	return np.nanmean(LL)

class SGD(object):
    """ 
    Stohastic gradient dissent  optimization method for liniar Model 
    of factorization models
    for each epoh choice  b_size events and run all iteration
    repeat for next epoh
    
    coef <=  the initial parameters of the Model, default = None
    b_size  <=  the batch size , { integer or None or 'coplex'}, default = 100
    step <=  the learning step, default = 1e-3
    N_epoh <= the number of epohs, default = 10
    method <= the method of optimization such as {'Adam', 'AdaMax'), default = 'AdaMax'
    rank <= for factorization model the rank of factorization, default = None
    moment1 <=  the first moment, default = 0.9
    moment2 <=  the second moment, default = 0.99
    lmbd <=  the regularization constant, default = 10
    reg <= method regularization, default = L2
    max_iteration <=  the maximum of iterations , default = 100
    eps <=  approximantely zero , default = 1e-4
    min_weight_dist <=  the distanse for covergence, default = 1e-3
    seed <=  the random constant , default = None
    verbose <=  flag debaging, default = False
    """
    
    
    def __init__(self, coef = None, b_size=100, step=1e-3, N_epoh=10, method='AdaMax', 
                    moment1=0.9, moment2=0.99, lmbd=10., max_iteration=100, reg ='L2',
                    eps=1e-4,  min_weight_dist=1e-3, seed=None, verbose=False):
        
        self.coef = None
        self.b_size = b_size
        self.step = step
        self.N_epoh = N_epoh
        self.method = method
        self.moment1 = moment1
        self.moment2 = moment2
        self.lmbd = lmbd
        self.max_iteration = max_iteration
        self.reg = reg
        
        self.eps = eps
        self.min_weight_dist = min_weight_dist
        self.seed = seed
        self.verbose = verbose
        
    def fit(self, X, y, group_idx, weight=None):
        """
        build the parameters of model - coef [j]
         
        X <= train data as matrix [i,j] 
        y <= train target as vector [i]
        weight <= weights of events or samples [i]
        group_idx <= group of rows in data  [i]      
        """
        #initial parameters
        if self.coef is None:
            self.coef = np.random.normal(0, scale =1e-2, size =(X.shape[1]))
        elif self.coef.shape[0] != X.shape[1]:
            logger.info('The shape of coef is not the same as X  %i'%X.shape[1])
            self.coef = np.random.normal(0, scale =1e-2, size =(X.shape[1]))
        if weight is None:
            weight = np.ones_like(y)
        
        # first distance beetwene parameters
        weight_dist = np.inf
        # for repeating of results
        np.random.seed(self.seed)
        
        # store of grad changes
        self.grad_norm = np.zeros(self.max_iteration * self.N_epoh +1)
        
        # FOR NESTEROV OPTIMIZATION
        # last gradient value
        last_grad = np.zeros_like(self.coef) 
        # last gradient's variation value
        std_grad = np.zeros_like(self.coef)
        # unique events in data set
        events_list = np.unique(group_idx)

        # MAIN LOOP
        for i in range(self.N_epoh):
            # random choice the events of b_size
            if isinstance(self.b_size, int):
                r_events = np.random.choice(events_list, size = self.b_size)
            
            elif self.b_size == 'complex':
                min_b_size = 10
                b_size_i = np.linspace(min_b_size, len(events_list)/2, num =self.N_epoh).astype(int)
                r_events = np.random.choice(events_list, size = b_size_i[i])
            else:
                _size = 0.8
                r_events = np.random.choice(events_list, size =int(len(events_list)*_size))
    
            mask = np.in1d(group_idx, r_events)
            X_train, y_train = X[mask,:], y[mask]
            strata, weight_t = group_idx[mask], weight[mask]
       
            strata = strata_scale_down(strata)
            if self.verbose:
                logger.info('Fitting epoh %s' % (i+1))
            
            for iter_num in range(1, self.max_iteration +1):
                coef_N = self.coef - self.step * self.moment1 * last_grad # for Nesterov momentum
                if self.reg == 'L1':
                     # get gradient with  Nesterov momemtum + L1 regularisation
                    _grad_ = self.first_derivative(X_train, y_train, weight_t, strata, coef_N) + self.lmbd * np.sign(coef_N) # gradient and L1
                else: 
                    # get gradient with  Nesterov momemtum + L2 regularisation
                    _grad_ = self.first_derivative(X_train, y_train, weight_t, strata, coef_N) + self.lmbd * coef_N # gradient and L2
                
                
                last_grad = self.moment1 * last_grad + (1-self.moment1) * _grad_ # Update weight first moment estimate
                t = i * self.max_iteration + iter_num
                _last_grad_ = last_grad/(1 - np.power(self.moment1, t)) # Correct first moment estimate
                if  self.method == 'AdaMax':
                    std_grad = np.max(np.vstack((self.moment2 * std_grad, np.abs(_grad_))), axis =0) # Update weight second moment estimate 
                    
                    self.coef = self.coef -  self.step * _last_grad_/std_grad
                elif self.method == 'Adam':
                    std_grad = self.moment2 * std_grad + (1-self.moment2)* _grad_ * _grad_ # Update weight second moment estimate 
                    _std_grad_ = std_grad/(1 - np.power(self.moment2, t)) # Correct second moment estimate
                    
                    self.coef = self.coef - self.step * _last_grad_/(np.square(_std_grad_) + self.eps)
                else:
                    logger.info('ERROR    : Method must be AdaMax or Adam')
                    return

                weight_dist = np.linalg.norm(last_grad, ord=2)/len(self.coef)
                self.grad_norm [t] = weight_dist
                
                if (weight_dist <= self.min_weight_dist) :
                    if self.verbose:
                        logger.info('Method Covariance:  distatnce {} epoh {}'.format(weight_dist, i+1))
                    break

                if self.verbose:
                    logger.info('Epoh {},  itaration {},  distatnce {}'.format(i+1, iter_num, weight_dist))

                if not np.isfinite(self.coef).all():
                    logger.info('ERROR : fitting  overcome  the some coef is nan or inf')
                    return

    def predict_prob(self, X, group_idx):
        """
        predict probability to win in the group_idx  for data X
        """
        if (self.coef is None) or not np.isfinite(self.coef).all():
            logger.info('ERROR : model is not define')
            return
        if self.coef.shape[0] != X.shape[1]:
            logger.info('The shape of the X is not the same as the coef  %i'%self.coef.shape[0]) 
            return 
        
        strata = strata_scale_down(group_idx)
        strenth, expstrenth, expsum = self.preprocessing(X, strata, self.coef)
        prob = expstrenth/expsum
        prob = np.where(np.isfinite(prob), prob, 1.) # in the case inf/inf = 1.
        
            
        return prob 
        
    def predict(self, X, group_idx):
        """
        predict winner in the group_idx for data X
        """
        strata = strata_scale_down(group_idx)
        prob = self.predict_prob(X, strata)
        return (prob >= unpack(strata, accum(strata, prob, 'max'))).astype(int)
        
        
    def preprocessing(self, X, strata, theta):

        
        strength = np.dot(X, theta) 
        expstrength = np.exp(strength)
        expsum = unpack(strata, accum(strata, expstrength))
        return strength, expstrength, expsum    
        
        
    def first_derivative(self, X, y, weight, strata, theta):
        
        strenth, expstrenth, expsum = self.preprocessing(X, strata, theta)
        prob = expstrenth/expsum
        delta = (y - prob) * weight
            
        return -np.dot(delta, X)
    
    def learning_plot(self):
        
        plt.figure(figsize=(20, 10))
        plt.plot(self.grad_norm, label ='Learning score')
        plt.legend()
        plt.show()
        print self.grad_norm[self.grad_norm> 0][-1]
