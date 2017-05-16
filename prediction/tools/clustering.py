import numpy as np
import pickle

from utils import settings, timestamp, YEAR
from utils.arrayview import ArrayView, TimeseriesView


from prediction.models.preprocessing import Model
from prediction.models.prediction import factornames_trimmed
from prediction.models.parameters import factor_build_end


def time_sets_end (data, cl, time = 'time', end = (0.75, 0.85), threshold = 100):
    """ 
	time the end of set in limits end    
 
    <data> - pandas dataFrame
    <cl>  - column with names of clusters for each event_id
    <time> - column with the start's time  for each event_id 
    <end> - part of poins that before interval 
    <threshold> minimum number of points in the cluster if the cluster is counted
    """
    
    time_end_min =[]
    time_end_max =[]
    df_cl = data[cl].value_counts()
    
    for cluster  in df_cl.index[df_cl > threshold]:
        time_list = data[time][data[cl] == cluster].tolist()
        time_end_min.append(time_list[int(len(time_list)*end[0])]) # min of the end train  time  
        time_end_max.append(time_list[int(len(time_list)*end[1])]) # max of the end train time
        
    time_min = np.max(time_end_min)
    time_max = np.min(time_end_max)
    
    if time_min <= time_max:
        return (time_min+time_max)/2.
    else:
        print 'not itersection of time sets for {}  {}'.format( cl, end)
        return time_max

def cut(data, cl, time = 'time', train = (0.75, 0.85), val = (0.5, 0.65), threshold = 100, verbose=False):
    """ 
    separate the set on train, validation, test such the time of increase 
    <data> - pandas dataFrame
    <cl>  - column with names of clusters for each event_id
    <time> - column with the start's time  for each event_id 
    <train> - part of points  for train
    <val> -  part of points  for validation from (all set - train)
    <threshold> minimum number of points in the cluster if the cluster is counte
    """
    
    train_event, val_event, test_event =[], [],[]
    
    time_train_end = time_sets_end( data, cl , end =train, threshold = threshold) # end train time
    train_event = data.index[data[time] <= time_train_end].tolist() 
    
    df_val = data[data[time] > time_train_end]
    time_test_start = time_sets_end(df_val , cl, end =val) # end validation time
    
    if verbose:
        print 'time train end', time_train_end
        print 'time test start', time_test_start
    val_event = data.index[(data['time'] >time_train_end) & (data['time'] <= time_test_start)].tolist()
    test_event = data.index[data['time'] > time_test_start].tolist()
    
    return (train_event, val_event, test_event)

def step1(data, train_val_test, factors =factors, av= av, mod =mod, strata= strata, verbose=False):
    
    bad_list = []
    train, val, test = train_val_test
    
    
    for cluster in np.unique(data):
        
        train_cluster = np.array(train)[np.in1d(train, data.index[data == cluster])]
        is1 = np.in1d(av.event_id, train_cluster)
        val_test_cluster = np.array(val+test)[np.in1d((val+test), data.index[data == cluster])]
        oos = np.in1d(av.event_id, val_test_cluster)
        
    
        stats, prob = clmodel.run_cl_model (factors, av.result, strata, is1= is1,  oos =oos, lmbd = 10) 
        
        if verbose:
            print 'll  {}'.format(stats.ll)
            print 'count {}    cluster  {}  '.format(data.value_counts()[cluster], cluster)
            print ''
        
        if stats.ll[0] < stats.ll[2]:
            bad_list.append(cluster)
    return bad_list, prob


def step2(data, train_val_test, threshold = 100, av =av, tsav = tsav, factors = factors, mod = mod, verbose=False):
    
    model_coefs, model_step1prob, model_step2prob, model_likelihood = {}, {}, {}, {}
    
    train, val, test = train_val_test
    df_cl = data.value_counts()

    for cluster in df_cl.index[df_cl > threshold]:

        train_cluster = np.array(train)[np.in1d(train, data.index[data == cluster])]
        val_test_cluster = np.array(val+test)[np.in1d((val+test), data.index[data == cluster])]
       
        
        mod.is1 = np.in1d(av.event_id, train_cluster)
        mod.is2 = np.in1d(av.event_id, train_cluster)
        mod.oos = np.in1d(av.event_id, val_test_cluster)
        
        lmbd = int(10*8000./ df_cl[cluster])       
		         
        model_coefs[cluster], model_step1prob[cluster], model_step2prob[cluster], model_likelihood[cluster]\
        = mod.fit_slices(tsav, factors,  depth=3, lmbd =lmbd, verbose=False, fit_afresh=True)
        if verbose:
            print 'cluster {}  number  {}'.format(cluster, df_cl[cluster])
            print 'LL  {}          {}            {}'.format (len(train_cluster), len(train_cluster), len(val_test_cluster))
            print model_likelihood[cluster]
    return model_coefs, model_step1prob, model_step2prob, model_likelihood

def step2a(data, cluster_list, is1, oos, av =av, tsav = tsav, factors = factors, mod = mod, verbose=False):
    
    model_coefs, model_step1prob, model_step2prob, model_likelihood = {}, {}, {}, {}
    
    df_cl = data.value_counts()

    for cluster in cluster_list:

        mask_cluster = np.in1d(av.event_id, data.index[data == cluster])
        
        mod.is1 = is1 & mask_cluster
        mod.is2 = is1 & mask_cluster
        mod.oos = oos & mask_cluster
        
        lmbd = int(10*8000./ df_cl[cluster])		
		         
        model_coefs[cluster], model_step1prob[cluster], model_step2prob[cluster], model_likelihood[cluster]\
        = mod.fit_slices(tsav, factors,  depth=3, lmbd =lmbd, verbose=False, fit_afresh=True)
        if verbose:
            print 'cluster {}  number  {}'.format(cluster, df_cl[cluster])
            train_event = np.unique(av.event_id[mod.is1])
            test_event = np.unique(av.event_id[mod.oos])
            print 'LL  {}          {}            {}'.format (len(train_event), len(train_event), len(test_event))
            print model_likelihood[cluster]
    return model_coefs, model_step1prob, model_step2prob, model_likelihood

def cluster_lists(data, oos, threshold, min_test =5, verbose=False):
    """
    return
    cluster_list the list of cluster where each cluster has test points more that min_test 
    
    <data>  pandas Series with index are event_id , data are clusters names
    <oos>   test mask
    <threshold> minimum number of points in the cluster if the cluster is counte
    """
    
    df_cl = data.value_counts()
    cluster_list = []
   
    for cluster  in df_cl.index[df_cl > threshold]:
        
        cluster_mask = np.in1d(av.event_id, data.index[data == cluster])
        oos_event = np.unique(av.event_id[oos & cluster_mask])
        
        if len(oos_event) >= min_test:
            
            cluster_list.append(cluster)
            
    return cluster_list



def ll_diff (prob_new, prob_old, train, val, test, 
             av =av, tsav =tsav, strata = strata, predict_mask =predict_mask):
    """ 
    count the differance of Likelihood for two models 
    <prob_new>  probability of new model
    <prob_old>  probability of old model
    <train, val, test> are lists of events for train, validation, test
    """
    
    llcomb = np.zeros((11, 3))
    ll_old = np.zeros((11, 3))
        
    is1_ = np.in1d(av.event_id, train)
    is2_ = np.in1d(av.event_id, val)
    oos_ = np.in1d(av.event_id, test)
        
    for sl in xrange(10):
        good = ~np.isnan(tsav[sl + 1].log_pmkt_back) & ~np.isnan(tsav[sl + 1].log_pmkt_lay)
        good = uaccum(strata, good, func='all')
        for i, mask in enumerate([is1_, is2_, oos_]):
            p_new = prob_new[sl, mask[predict_mask] & good] # probobility new model
            p_old = prob_old[sl, mask[predict_mask] & good] # probobility old model
            
            winners = av.result[predict_mask][mask[predict_mask] & good] == 1

            llcomb[sl, i] = np.mean(np.log(p_new[winners][p_new[winners] !=0])) * 1000 # LL new model
            ll_old[sl, i] = np.mean(np.log(p_old[winners][p_old[winners] !=0])) * 1000 # LL old model
        #print llcomb[sl, i] - ll_old[sl, i]
            
    diff_new_old = llcomb- ll_old # differance between mix  and old model 
    return diff_new_old

def ll_for_each_cluster (data,  new_Model, old_Model, train_val_test, best ='train', not_list =np.array([]),                         
                         av =av, tsav = tsav, strata = strata, predict_mask =predict_mask, verbose=False):
    """ 
    buid the new model to replace each cluster by new
    <data> - pandas Series with index are event_id , data are clusters names 
    <new_Model>  wins probability for each cluster 
    <old_Model>  wins probability for no cluster 
    <train_val_test>  event_id for train , validation , test 
    <best>  the choose from train or validadion
    <not_list>  the list with clusters that to exclude from model
    """
    
    if best == 'train':
        best = 0
    else:
        best = 1
    
    train, val, test = train_val_test
    mean_new =[]
    
    for cluster in new_Model.keys():
        
        if not cluster in not_list:
            cluster_mask = np.in1d(av.event_id[predict_mask], data.index[data == cluster])
            prob_mix = np.where(cluster_mask , new_Model[cluster] , old_Model )
        

            diff_new_old = ll_diff(prob_mix, old_Model, train, val, test)

            print 'cluster ', cluster
            if verbose:
                print diff_new_old
    
            mean_ll_diff = diff_new_old[:10].mean(axis =0)
            print 'mean  ', mean_ll_diff
            if mean_ll_diff[best] > 0:
                mean_new.append((cluster,mean_ll_diff[best]))
    cl_list = [x[0] for x in sorted(mean_new , key = lambda x: x[1], reverse =True)]
    return cl_list, mean_new

def ll_for_mix_clusters (data, cl_list, new_Model, old_Model, train_val_test,  best ='test', 
                         av =av, tsav =tsav, strata =strata, predict_mask =predict_mask, verbose=False):
    """ 
    buid the new model to replace each cluster by the mix of new
    <data> - pandas Series with index are event_id , data are clusters names 
    <new_Model>  wins probability for each cluster 
    <old_Model>  wins probability for no cluster 
    <train_val_test> event_id for train , validation , test 
    <best>  the choose from train or validadion
    """
    
    cl_lists = []
    
    for i in range(len(cl_list)):
        cl_lists.append(cl_list[:i+1])
        
    if best =='test':
        best = 2
    elif best =='val':
        best = 1
    else:
        best = 0

    best_ll = 0.
    best_mix = None
    
    for list_ in cl_lists:
        print list_
        prob_mix = old_Model
        train, val, test = train_val_test
        
        for cluster in list_:
            
                cluster_mask = np.in1d(av.event_id[predict_mask], data.index[data == cluster])
                prob_mix = np.where(cluster_mask , new_Model[cluster],  prob_mix)
    
        diff_new_old = ll_diff(prob_mix, old_Model, train, val, test)
        if verbose:
            print '    train       validation      test'
            print diff_new_old      # differance between mix  and old model
        mean_ll_diff = diff_new_old[:10].mean(axis =0)
        print 'mean ', mean_ll_diff
        if mean_ll_diff[best] > best_ll:
            best_mix = list_
            best_ll = mean_ll_diff[best]

    return best_mix, best_ll

def write_simdata(step1probs, oos, coefs, cluster_number=None, file_ = 'simdata.p'):
    '''
    <step1probs> is expected to be a matrix N_slices x len(av). 
    <oos> is a boolean mask denoting the out of sample range. len(oos) shoud equal len(av)
    <coefs> is a coefficient matrix with the size N_slices x 3
    <cluster_number> is an integer array with the cluster numbers per race. Size: len(av)
    '''
    f = file(settings.paths.join(file_), 'wb')
    if cluster_number is None:
        s1p = step1probs[:, oos]
    else:
        cluster_number = cluster_number[oos]
        s1p = step1probs[:, :, oos]
    pickle.dump([s1p, oos, coefs, cluster_number], f)
    f.close()

def dic_to_tenzor(dic, key, base):
    '''
    tenzor where first dimention is the number of cluster
    0 = no_cluster
    <dic> dictionary cluster's data that to convert in tenzor
    <key> the list of clusters that use 
    <base> no cluster data
    '''
    
    key_0 = dic.keys()[0]
    tenzor = np.zeros((len(key)+1, dic[key_0].shape[0], dic[key_0].shape[1]))
    try:
        tenzor[0,:,:] = base
    except:
        print 'base and dic[k] have the diferent size'
        return
    for i,k in enumerate(key):
        tenzor[i+1,:,:] = dic[k]
    return tenzor

def clusters_number(data, key, av=av):
    """ 
    list with numbers of clusters 
    <data>  pandas Series index = event_id, data = cluster's names
    <key> the list of clusters that use
    """
    
    cl_number = np.zeros((len(av.event_id)))
    for i,k in enumerate(key):
        mask = np.in1d(av.event_id,data.index[data ==k])
        cl_number = np.where(mask, i+1, cl_number)
    return cl_number

def write_dic_to_simdata(file_name, old_step1probs, old_coefs, oos, data=None, av =av,
                         cluster_step1probs =None, cluster_coefs =None, cluster_names =None):
    """
    <file_name> is name of file to record
    <old_step1probs> is expected to be a matrix N_slices x len(av)
    <old_coefs> is a coefficient matrix with the size N_slices x 3
    <oos> is a boolean mask denoting the out of sample range. len(oos) shoud equal len(av)
    <data>  pandas Series index = event_id, data = cluster's names
    <cluster_step1probs> is expected to be a dictionary: key is the cluster name and
                        data are the matrix N_slices x len(av) for each cluster
    <cluster_coefs> is a dictionary : key is the cluster name and data and 
                        data are the coefficient matrix with the size N_slices x 3
    <cluster_number> is an integer array with the cluster numbers per race. Size: len(av)
    """
    
    cl_number= np.zeros((len(av.event_id)))
    
    if cluster_names is not None:
        
        s1prob = dic_to_tenzor(cluster_step1probs, cluster_names, old_step1probs)
        coef_s = dic_to_tenzor(cluster_coefs, cluster_names, old_coefs)
        
        for i,k in enumerate(cluster_names):
            mask = np.in1d(av.event_id,data.index[data ==k])
            cl_number = np.where(mask, i+1, cl_number)
        #cl_number = clusters_number(data, cluster_names, av=av)
          
        
    else:
        
        s1prob = np.zeros((1,old_step1probs.shape[0], old_step1probs.shape[1]))
        s1prob[0,:,:] = old_step1probs
        coef_s = np.zeros((1,old_coefs.shape[0], old_coefs.shape[1]))
        coef_s[0,:,:] = old_coefs
        
        

    #write_simdata(s1prob, oos, coef_s, cl_number, file_ = file_name)
    s1prob = s1prob[:,:,oos]
    cl_number = cl_number[oos]
    
    with open (settings.paths.join(file_name), 'wb') as f:
            pickle.dump( [s1prob, oos, coef_s, cl_number], f)
    return

def write_to_simdata(file_name, old_step1probs, old_coefs, oos, av =av, 
                         cluster_step1probs =None, cluster_coefs =None, cluster_names =None):
    """
    <file_name> is name of file to record
    <old_step1probs> is expected to be a matrix N_slices x len(av)
    <old_coefs> is a coefficient matrix with the size N_slices x 3
    <oos> is a boolean mask denoting the out of sample range. len(oos) shoud equal len(av)
    
    <cluster_step1probs> is expected to be a dictionary: key is the cluster name and
                        data are the matrix N_slices x len(av) for each cluster
    <cluster_coefs> is a dictionary : key is the cluster name and data and 
                        data are the coefficient matrix with the size N_slices x 3
    <cluster_number> is an integer array with the cluster numbers per race. Size: len(av)
    """
    
    cl_number= np.zeros((len(av)))
    
    if cluster_names is not None:
        
        s1prob = dic_to_tenzor(cluster_step1probs, cluster_names, old_step1probs)
        coef_s = dic_to_tenzor(cluster_coefs, cluster_names, old_coefs)
        
        for i,k in enumerate(cluster_names):
            mask = av.obstacle == k
            cl_number = np.where(mask, i+1, cl_number)
        #cl_number = clusters_number(data, cluster_names, av=av)
         
    else:
        
        s1prob = np.zeros((1,old_step1probs.shape[0], old_step1probs.shape[1]))
        s1prob[0,:,:] = old_step1probs
        coef_s = np.zeros((1,old_coefs.shape[0], old_coefs.shape[1]))
        coef_s[0,:,:] = old_coefs
        
    #write_simdata(s1prob, oos, coef_s, cl_number, file_ = file_name)
    s1prob = s1prob[:,:,oos]
    cl_number = cl_number[oos]
    
    with open (settings.paths.join(file_name), 'wb') as f:
            pickle.dump( [s1prob, oos, coef_s, cl_number], f)
    return




