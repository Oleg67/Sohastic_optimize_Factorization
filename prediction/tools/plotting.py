import numpy as np
from scipy.stats import skew, kurtosis
from pylab import plot, hist, bar, clf, show, figure, xlabel, ylabel, xticks, errorbar

from utils import timestamp, intnan, CleanTextTable, DAY, WEEK, MONTH
from utils.math import quantile_custom
from utils.accumarray import accum, unpack

from .helpers import strata_scale_down



def varinfo(x, plot_hist=True, fignum=10, discrete=False, width=0.5):
    '''displays information about vector x. Size, distribution, moments, ...'''
    np.set_printoptions(precision=4, suppress=True)
    udata, means = np.nan, np.nan
    if len(x.shape) > 1:
        raise ValueError('Input has to be a uni-dimensional vector')
    print 'Variable info'
    print '--------------------------------------'
    print 'Length: %s' % len(x)    
    if not discrete:
        print 'Number of NaN\'s: %s  (%4.2f%%)' % (np.sum(np.isnan(x)), np.mean(np.isnan(x)) * 100)
        print 'Number of Inf\'s: %s  (%4.2f%%)' % (np.sum(np.isinf(x)), np.mean(np.isinf(x)) * 100)
        print 'Number of zeros: %s  (%4.2f%%)' % (np.sum(abs(x) < 1e-10), np.mean(abs(x) < 1e-10) * 100)
        good = ~np.isnan(x) & ~np.isinf(x)
        print 'Number of unique\'s: %s' % len(np.unique(x[good]))
        print 'Mean: %4.2f' % np.mean(x[good])
        print 'Std: %4.2f' % np.std(x[good], ddof=1)
        print 'Skewness: %4.2f' % skew(x[good])
        print 'Kurtosis: %4.2f' % kurtosis(x[good])
        print 'Min and max values: %4.2f to %4.2f' % (min(x[good]), max(x[good]))
    else:
        print 'Fraction of entries of each item of the discrete data:'
        udata = np.unique(x)
        if isinstance(udata[0], str):
            maxL = -1
            for i in xrange(len(udata)):
                if len(udata[i]) > maxL:
                    maxL = len(udata[i])
        else:
            maxL = 1       
        means = np.zeros(len(udata)) * np.nan
        for i in xrange(len(udata)):
            means[i] = np.mean(x == udata[i])
            print ('%' + str(maxL) + 's: %.4f') % (udata[i], means[i])
    print '--------------------------------------'
    if plot_hist:        
        figure(fignum)           
        clf()
        if not discrete:
            hist(x[good], max((10, min((300, round(len(np.unique(x[good])) / 3))))))
        else:
            ind = np.arange(0, 2 * width * len(udata), 2 * width)
            bar(ind, means, width=width, align='center')
            xticks(ind, udata)
            ylabel('fraction')
            
        show(block=False)
    return udata, means
        
        
def simple_plot(y, fignum=10, opt='-'):
    figure(fignum)
    clf()
    plot(y, opt)
    show(block=False)
    


def avinfo(av):
    print 'Faction of invalid entries:'
    frac_missing = np.zeros(len(av.keys()))
    for i, factor in enumerate(av.keys()):
        frac_missing[i] = np.mean(intnan.np.isnan(av[factor]))
    sortidx = np.argsort(frac_missing)[::-1]
    for i in xrange(len(av.keys())):
        print '%30s: %6.4f' % (av.keys()[sortidx[i]], frac_missing[sortidx[i]])
    
def plot_performance(profit0, start_time0, clear=True, fignum=1, days=[1], skip_zeros=True):
    '''Plots performance curve. <bet> is the AttrDict returned from simulate_betting
    <clear> clears the figure <fignum> before drawing a new one.
    <days> denotes the days of the month when to plot a tick on the x-axis.'''
    figure(fignum)
    if clear:
        clf()
    
    profit = profit0.copy()
    start_time = start_time0.copy()
    if skip_zeros:
        good = (profit != 0)
        profit = profit[good]
        start_time = start_time[good]
    plot(np.cumsum(profit))
    tick_str = []
    current_day = -1
    ind = []
    # make a tick on the 1st and (1 + interval * n)th days
    for i in xrange(len(profit)):
        if start_time[i] == 0:
            continue
        s = timestamp(start_time[i]).format()[5:10]
        day = int(s[3:5])
        if day in days and (day != current_day):
            ind.append(i)
            tick_str.append(s[3:5] + '.' + s[:2])
        if day != current_day:
            current_day = day
            
            
    xticks(ind, tick_str)
    ylabel('cumulative profit')
    xlabel('date')
    show(block=False)    


def print_profit_stats(start_time, profits):
    assert len(start_time) == len(profits)
    good = profits != 0
    if not np.any(good):
        good = np.ones(len(profits), dtype=bool)
    pr = profits[good]
    days = strata_scale_down(np.floor(start_time[good] / DAY).astype(int))
    daypr = accum(days, profits[good])
    weeks = strata_scale_down(np.floor(start_time[good] / WEEK).astype(int))
    weekpr = accum(weeks, profits[good])
    months = strata_scale_down(np.floor(start_time[good] / MONTH).astype(int))
    monthpr = accum(months, profits[good])

    tab = CleanTextTable(default='%.8g')
    tab.header(('x', 'racewise', 'daily', 'weekly', 'monthly'))
    tab.set_cols_width((18, 15, 15, 15, 15))
    tab.set_cols_align(tuple('lllll'))

    totpr = np.round(np.sum(pr))
    tab.add_row(('total', totpr, totpr, totpr, totpr))
    tab.add_row(('mean', '%.2f' % np.mean(pr), '%.2f' % np.mean(daypr), '%.2f' % np.mean(weekpr), '%.2f' % np.mean(monthpr)))
    tab.add_row(('std', '%.2f' % np.std(pr), '%.2f' % np.std(daypr), '%.2f' % np.std(weekpr), '%.2f' % np.std(monthpr)))
    ms = [np.mean(pr) / np.std(pr), np.mean(daypr) / np.std(daypr), np.mean(weekpr) / np.std(weekpr), np.mean(monthpr) / np.std(monthpr)]
    tab.add_row(('mean / std', '%.2f' % ms[0], '%.2f' % ms[1], '%.2f' % ms[2], '%.2f' % ms[3]))
    tab.add_row(('win fraction', '%.3f' % np.mean(pr > 0), '%.3f' % np.mean(daypr > 0), '%.3f' % np.mean(weekpr > 0), '%.3f' % np.mean(monthpr > 0)))
    tab.add_row(('mean if won', '%.2f' % np.mean(pr[pr > 0]), '%.2f' % np.mean(daypr[daypr > 0]), '%.2f' % np.mean(weekpr[weekpr > 0]), '%.2f' % np.mean(monthpr[monthpr > 0])))
    tab.add_row(('mean if lost', '%.2f' % np.mean(pr[pr < 0]), '%.2f' % np.mean(daypr[daypr < 0]), '%.2f' % np.mean(weekpr[weekpr < 0]), '%.2f' % np.mean(monthpr[monthpr < 0])))
    tab.add_row(('sample size', len(pr), len(daypr), len(weekpr), len(monthpr)))
    sigtime = 4 / ms[0] ** 2
    tab.add_row(('significance time', '%.2f' % sigtime, '%.2f' % (sigtime / len(pr) * len(daypr)), '%.2f' % (sigtime / len(pr) * len(weekpr)), '%.2f' % (sigtime / len(pr) * len(monthpr))))
    print tab.draw()



def display_profits(bet, fignum=1, days=[1, 15]):
    print_profit_stats(bet.start_time, bet.profit)
    plot_performance(bet.profit, bet.start_time, fignum=fignum, days=days)


    
def binned_plot(x, y, nBins=20, fignum=1, clear=True):
    x1 = x[~np.isnan(x)]
    y1 = y[~np.isnan(x)]
    edges = quantile_custom(x1, nBins-1)
    #edges = linspace(0, 1, nBins)
    binidx = np.searchsorted(edges, x1)
    bin_centers = accum(binidx, x1, func='nanmean')
    mean_y = accum(binidx, y1, func='nanmean')
    std_y = accum(binidx, y1, func='nanstd')
    ny = accum(binidx, np.ones_like(y1))
    se = std_y / np.sqrt(ny)
    
    figure(fignum)
    if clear:
        clf()
    errorbar(bin_centers, mean_y, se)
    #plot(bin_centers, mean_y, 'b')
    show(block=False)
    
def anova_plot(category, target, fignum=10, plot_hist=True, clear=True, width=0.5):
    print 'Variable info'
    print '--------------------------------------'
    print 'Length: %s' % len(target)    
    print 'Mean and standard error for each item of discrete data:'
    udata = np.unique(category)
    if isinstance(udata[0], str):
        maxL = -1
        for i in xrange(len(udata)):
            if len(udata[i]) > maxL:
                maxL = len(udata[i])
    else:
        maxL = 1       
    means = np.zeros(len(udata)) * np.nan
    se = np.zeros(len(udata)) * np.nan
    for i in xrange(len(udata)):
        data = target[category == udata[i]]
        means[i] = np.nanmean(data)
        se[i] = np.nanstd(data) / np.sqrt(np.sum(~np.isnan(data)))
        print ('%' + str(maxL) + 's: %.2f   (%.2f)') % (udata[i], means[i], se[i])
    if plot_hist:        
        figure(fignum)
        if clear:           
            clf()
        ind = np.arange(0, 2 * width * len(udata), 2 * width)
        bar(ind, means, yerr=se, width=width, align='center')
        xticks(ind, udata)
        ylabel('average')
        xlabel('error bars indicate the standard error')            
        show(block=False)
    return udata, means, se
    
        
def correlation(x, y):
    g = ~np.isnan(x) & ~np.isnan(y)
    r = np.corrcoef(x[g], y[g])[0,1]
    return r
