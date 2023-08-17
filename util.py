from keys import repo_path
from config import bad_symbols
from request import send_public_request, send_signed_request
from matplotlib import pyplot as plt
from datetime import datetime as dt
from time import time as current_time
from os import listdir, devnull
from numpy import poly1d, polyfit, mean, array, repeat, sqrt as npsqrt, RankWarning, sum as npsum, amax, amin, empty, where, diff, sign
from numpy.linalg import LinAlgError
from numpy.ma import masked_where, make_mask
from numpy.random import uniform, seed as randomseed
from math import isnan, isinf, ceil
from warnings import filterwarnings, warn
from bisect import bisect_left
from json import dump as jdump, load as jload
from git import Repo
import sys
import requests
filterwarnings("error")   # raises warnings as errors - needed to catch the RankWarning thrown by the best_fit functions
filterwarnings("ignore", category=ResourceWarning)

#==============================================================================

# CONTENTS:
# COLOURS
# TIME
# RESOURCE MANAGEMENT
# PRICE DATA
# ASSETS & SYMBOLS
# DATA MANIPULATION
# DATA VISUALISATION

#==============================================================================
# COLOURS

class Colours:
    """
    A class used to define strings that change the colour of text printed to
    the console

    Attributes
    ----------
    k : str
        Makes proceeding text black
    r : str
        Makes proceeding text red
    g : str
        Makes proceeding text green
    y : str
        Makes proceeding text yellow
    b : str
        Makes proceeding text blue
    m : str
        Makes proceeding text magenta
    c : str
        Makes proceeding text cyan
    h : str
        Makes proceeding text grey
    K : str
        Makes proceeding text bright black (grey)
    R : str
        Makes proceeding text bright red
    G : str
        Makes proceeding text bright green
    Y : str
        Makes proceeding text bright yellow
    B : str
        Makes proceeding text bright blue
    M : str
        Makes proceeding text bright magenta
    C : str
        Makes proceeding text bright cyan
    H : str
        Makes proceeding text bright grey (white)
    X : str
        Resets text colour to default

    Methods
    -------
    rgb(r, g, b)
        Returns the colour string corresponding to the given r g b values
    example(cls)
        Prints each basic colour to the console so the user can see them
    """

    k = '\x1B[38;5;0m'  # black
    r = '\x1B[38;5;1m'  # red
    g = '\x1B[38;5;2m'  # green
    y = '\x1B[38;5;3m'  # yellow
    b = '\x1B[38;5;4m'  # blue
    m = '\x1B[38;5;5m'  # magenta
    c = '\x1B[38;5;6m'  # cyan
    h = '\x1B[38;5;7m'  # grey
    K = '\x1B[38;5;8m'  # bright black (grey)
    R = '\x1B[38;5;9m'  # bright red
    G = '\x1B[38;5;10m' # bright green
    Y = '\x1B[38;5;11m' # bright yellow
    B = '\x1B[38;5;12m' # bright blue
    M = '\x1B[38;5;13m' # bright magenta
    C = '\x1B[38;5;14m' # bright cyan
    H = '\x1B[38;5;15m' # bright grey (white)
    X = '\x1b[39m'      # reset to default

    def rgb(r:int, g:int, b:int) -> str:
        """
        Returns the colour string corresponding to the given r g b values

        Parameters
        ----------
        r : int
            The red value - integer between 0 and 255
        g : int
            The green value - integer between 0 and 255
        b : int
            The blue value - integer between 0 and 255

        Returns
        -------
        str
            The colour string defined by the given r g b values
        """

        return '\x1B[38;2;{};{};{}m'.format(r, g, b)

    @classmethod
    def example(cls):
        """Prints each basic colour to the console so the user can see them"""

        for name, colour in get_fields(cls).items():
            print(colour+"{} COLOUR".format(name))
        return

c = Colours()  # initialise Colours instance to be used throughout the module

#==============================================================================
# TIME

def now() -> int:
    """
    Returns the current UTC unix timestamp in milliseconds

    Returns
    -------
    int
        The current UTC unix timestamp in milliseconds
    """

    return int(current_time()*1000)

def valid_timestamp(timestamp:int) -> bool:
    """
    Returns True if the given timestamp is a valid unix timestamp (an integer
    with 13 digits), else False

    Parameters
    ----------
    timestamp : int
        The timestamp to validate

    Returns
    -------
    bool
        True if the timestamp is valid, else False
    """

    return isinstance(timestamp, int) and len(str(timestamp)) == 13

def minutes_ago(minutes:float, end:int=None) -> int:
    """
    Returns the UTC unix timestamp in milliseconds for the given number of
    minutes ago from 'end'

    Parameters
    ----------
    minutes : float
        The number of minutes prior to end the timestamp should define
    end : int, optional
        The timestamp from which the previous minutes are taken
        The default is None

    Raises
    ------
    ValueError
        If end is not a valid unix timestamp in milliseconds (int of length 13)

    Returns
    -------
    int
        The UTC unix timestamp in milliseconds for the given number of minutes
        ago from 'end'
    """

    end = end if end else now()
    if not valid_timestamp(end):
        raise ValueError("invalid timestamp: {}".format(end))
    return int(end - minutes*6e4)     # *6e4 to convert minutes to milliseconds

def hours_ago(hours:float, end:int=None) -> int:
    """
    Returns the UTC unix timestamp in milliseconds for the given number of
    hours ago from 'end'

    Parameters
    ----------
    hours : float
        The number of hours prior to end the timestamp should define
    end : int, optional
        The timestamp from which the previous hours are taken
        The default is None

    Raises
    ------
    ValueError
        If end is not a valid unix timestamp in milliseconds (int of length 13)

    Returns
    -------
    int
        The UTC unix timestamp in milliseconds for the given number of hours
        ago from 'end'
    """

    end = end if end else now()
    if not valid_timestamp(end):
        raise ValueError("invalid timestamp: {}".format(end))
    return int(end - hours*3.6e6)    # *3.6e6 to convert hours to milliseconds

def days_ago(days:float, end:int=None) -> int:
    """
    Returns the UTC unix timestamp in milliseconds for the given number of
    days ago from 'end'

    Parameters
    ----------
    days : float
        The number of days prior to end the timestamp should define
    end : int, optional
        The timestamp from which the previous days are taken
        The default is None

    Raises
    ------
    ValueError
        If end is not a valid unix timestamp in milliseconds (int of length 13)

    Returns
    -------
    int
        The UTC unix timestamp in milliseconds for the given number of days
        ago from 'end'
    """

    end = end if end else now()
    if not valid_timestamp(end):
        raise ValueError("invalid timestamp: {}".format(end))
    return int(end - days*8.64e7)    # *8.64e7 to convert days to milliseconds

def weeks_ago(weeks:float, end:int=None) -> int:
    """
    Returns the UTC unix timestamp in milliseconds for the given number of
    weeks ago from 'end'

    Parameters
    ----------
    days : float
        The number of weeks prior to end the timestamp should define
    end : int, optional
        The timestamp from which the previous weeks are taken
        The default is None

    Raises
    ------
    ValueError
        If end is not a valid unix timestamp in milliseconds (int of length 13)

    Returns
    -------
    int
        The UTC unix timestamp in milliseconds for the given number of weeks
        ago from 'end'
    """

    end = end if end else now()
    if not valid_timestamp(end):
        raise ValueError("invalid timestamp: {}".format(end))
    return int(end - weeks*6.046e8)     # *6.046e8 to convert weeks to milliseconds

def months_ago(months:float, end:int=None) -> int:
    """
    Returns the UTC unix timestamp in milliseconds for the given number of
    months ago from 'end'
    Assumes 1 month is 30.4375 days = 365.25/12 days

    Parameters
    ----------
    days : float
        The number of months prior to end the timestamp should define
    end : int, optional
        The timestamp from which the previous months are taken
        The default is None

    Raises
    ------
    ValueError
        If end is not a valid unix timestamp in milliseconds (int of length 13)

    Returns
    -------
    int
        The UTC unix timestamp in milliseconds for the given number of months
        ago from 'end'
    """

    end = end if end else now()
    if not valid_timestamp(end):
        raise ValueError("invalid timestamp: {}".format(end))
    return int(end - months*2.6298e9)    # *2.6298e9 to convert months to milliseconds

def years_ago(years:float, end:int=None) -> int:
    """
    Returns the UTC unix timestamp in milliseconds for the given number of
    years ago from 'end'
    Assumes 1 year is 365.25 days

    Parameters
    ----------
    days : float
        The number of years prior to end the timestamp should define
    end : int, optional
        The timestamp from which the previous years are taken
        The default is None

    Raises
    ------
    ValueError
        If end is not a valid unix timestamp in milliseconds (int of length 13)

    Returns
    -------
    int
        The UTC unix timestamp in milliseconds for the given number of years
        ago from 'end'
    """

    end = end if end else now()
    if not valid_timestamp(end):
        raise ValueError("invalid timestamp: {}".format(end))
    return int(end - years*3.15576e10)    # *3.15576e10 to convert years to milliseconds

def now_string() -> str:
    """
    Returns the current datetime in a formatted string

    Returns
    -------
    str
        The current datetime in a formatted string
    """

    return dt.now().strftime("%Y-%m-%d_%H;%M")

def now_string_time_only() -> str:
    """
    Returns the current time in a formatted string

    Returns
    -------
    str
        Returns the current time in a formatted string
    """

    return dt.now().strftime("%H;%M")

def now_string_date_only() -> str:
    """
    Returns the current date in a formatted string

    Returns
    -------
    str
        Returns the current date in a formatted string
    """

    return dt.now().strftime("%Y-%m-%d")

def timestamp_string(timestamp:int) -> str:
    """
    Returns the given unix timestamp in a formatted string
    Has format year-month-day_hour;minute

    Parameters
    ----------
    timestamp : int
        The timestamp to represent in string form

    Raises
    ------
    ValueError
        If the timestamp is not a valid unix timestamp

    Returns
    -------
    str
        The given unix timestamp in a formatted string
    """

    if not valid_timestamp(timestamp):
        raise ValueError("invalid timestamp: {}".format(timestamp))
    return dt.fromtimestamp(timestamp//1000).strftime("%Y-%m-%d_%H;%M")

def timestamp_string2(timestamp:int) -> str:
    """
    Returns the given unix timestamp in a formatted string
    Has format year-month-day_hour;minute;second
    Assumes the timestamp is valid

    Parameters
    ----------
    timestamp : int
        The timestamp to represent in string form

    Returns
    -------
    str
        The given unix timestamp in a formatted string

    """
    return dt.fromtimestamp(timestamp//1000).strftime("%Y-%m-%d_%H;%M;%S")

def timestamps2seconds(times:array) -> array:
    """
    Returns an array of integer time values in seconds corresponding to the
    times since the first timestamp in the input array
    The first element is always zero
    If an empty array is passed, an empty array is returned

    Parameters
    ----------
    times : array
        An array of chronologically ordered timestamps

    Returns
    -------
    array
        The sequence of time values from the first timestamp in seconds
    """

    if not any(times):
        return times
    return (times-times[0])/1000

def timestamp2hours(timestamp:int, t0:int) -> float:
    """
    Returns the number of hours between timestamp and t0

    Parameters
    ----------
    timestamp : int
        The timestamp whose hours after t0 you wish to calculate
    t0 : int
        The timestamp whose hours after which 'timestamp' is calculated

    Returns
    -------
    float
        The number of hours between timestamp and t0
    """

    return (timestamp-t0)/3600000

#==============================================================================
# DATA MANIPULATION

def append(arr:list, value, length:int) -> array:
    """
    Appends value to arr, returns a numpy array

    Used in practice to append values to a numpy array quickly
    The length of the array must be known and passed

    Parameters
    ----------
    arr : list-like
        The list-like object to which value is appended
        Can be any sliceable list-like object
    value : any
        The value appended to arr
        Can be any type
    length : int
        The length of the final array after value is appended i.e. len(arr) + 1

    Returns
    -------
    array
        arr with value appended as a numpy array
    """

    a = empty(length)
    a[:length-1] = arr
    a[-1] = value
    return a

def clip(N:float, low:float, high:float) -> float:
    """
    Clips the value N to be in the range low <= N <= high if it isn't already

    Parameters
    ----------
    N : float
        The value to clip
    low : float
        The lower limit
    high : float
        The upper limit

    Returns
    -------
    float
        The clipped value, either low, high, or N
    """

    return low if N<low else high if N>high else N

def binary_min(a:float, b:float) -> float:
    """
    Returns the smaller of a and b

    Parameters
    ----------
    a : float
        The first number
    b : float
        The second number

    Returns
    -------
    float
        The smaller of a and b
    """

    return a if a < b else b

def binary_max(a:float, b:float) -> float:
    """
    Returns the larger of a and b

    Parameters
    ----------
    a : float
        The first number
    b : float
        The second number

    Returns
    -------
    float
        The larger of a and b
    """

    return a if a > b else b

def normalise(values:array) -> array:
    """
    Returns the array values normalised between 0 and 1

    Parameters
    ----------
    values : array
        A numpy array of numbers

    Returns
    -------
    array
        The array values normalised between 0 and 1
    """

    minv = amin(values)
    maxv = amax(values)
    if minv == maxv:
        return values/minv
    return (values-minv)/(maxv-minv)

def std(values:array, m:float=None) -> float:
    """
    Returns the standard deviation of the array values

    Parameters
    ----------
    values : array
        A numpy array of numbers
    m : float, optional
        An alternative mean value around which to calculate the standard
        deviation instead of the actual mean of the data
        The default is None.

    Returns
    -------
    float
        The standard deviation of the array values
    """

    m = mean(values) if m is None else m
    return npsqrt(npsum((values-m)**2)/len(values))

def best_fit(x:array, y:array, order:int=1, w:array=None) -> array:
    """
    Returns the line (or curve if order>1) of best-fit through the given data,
    x y, as an array

    Parameters
    ----------
    x : array
        Array-like object of x values
    y : array
        Array-like object of y values
    order : int, optional
        The order of the polynomial best fit
        The default is 1
    w : array, optional
        Weights to apply to each x-y pair in the sequence
        The default is None

    Returns
    -------
    array
        Array of best fit values
    """

    if len(y) == 1:
        return y
    return poly1d(polyfit(x, y, order, w=w))(x)

def percentage_change(value:float, zero_point:float) -> float:
    """
    Returns the percentage change of value relative to zero_point

    Parameters
    ----------
    value : float
        The value whose percentage change from zero_point is to be calculated
    zero_point : float
        The value from which the percentage change is calculated

    Returns
    -------
    float
        The percentage change of value from zero_point
    """

    return 100*(value-zero_point)/zero_point

def percentage_array(arr:array, zero_point:float) -> array:
    """
    Returns a numpy array of percentage changes from zero_point corresponding
    to each value in arr

    Parameters
    ----------
    arr : array
        A numpy array whose percentage changes from zero_point is to be
        calculated
    zero_point : float
        The value from which the percentage changes are calculated

    Returns
    -------
    array
        The numpy array of percentage changes from zero_point
    """

    return 100*(arr-zero_point)/zero_point

def amplitude(values:array, width:float=None) -> float:
    """
    The amplitude of the array values, calculated as the distance covered by
    the data normalised by the number of data points or given width

    Parameters
    ----------
    values : array
        The numpy array of numbers whose amplitude is to be calculated
    width : float, optional
        The normalisation factor for the amplitude calculation
        The number of data points in values is used if width is None
        The default is None

    Returns
    -------
    float
        The amplitude of the array values normalised by width
    """

    L = len(values)
    width = L if width is None else width
    subtractee = append(values[1:], values[-1], L)
    return npsum(abs(values-subtractee))/width

def amplitude2(values:array, width:float, length:int) -> float:
    """
    The same as amplitude but with mandatory width and length parameters for
    faster execution
    """

    subtractee = append(values[1:], values[-1], length)
    return npsum(abs(values-subtractee))/width

def distance(values:array, length:int) -> float:
    """
    The same as amplitude2 but without scaling by the width
    """

    subtractee = append(values[1:], values[-1], length)
    return npsum(abs(values-subtractee))

def variance(values:array, width:float=None) -> float:
    """
    Returns the variance of the array values, normalised by width

    The same as amplitude but the distances covered between each data point
    pair is squared
    """

    L = len(values)
    width = L if width is None else width
    subtractee = append(values[1:], values[-1], L)
    return npsum((values-subtractee)**2)/width

def variance2(values:array, width:float, length:int) -> float:
    """
    The same as variance but with mandatory width and length parameters for
    faster execution
    """

    subtractee = append(values[1:], values[-1], length)
    return npsum((values-subtractee)**2)/width

# returns the true volatility of a data series
# assumes values is a numpy array of numbers
def volatility(values, width=0):
    return npsqrt(variance(values, width))

# returns the variance of a data series as a percentage
# assumes values is a numpy array of numbers
# formerly calculated as:   sum([(percentage_values[i]-percentage_values[i+1])**2 for i in range(len(percentage_values)-1)])/width
def variance_percentage(values, width=0):
    L = len(values)
    width = width if width else L
    percentage_values = percentage_array(values, values[0])
    subtractee = append(percentage_values[1:], percentage_values[-1], L)
    return npsum((percentage_values-subtractee)**2)/width

# returns the volatilty of a data series as a percentage
# assumes values is a numpy array of numbers
def volatility_percentage(values, width=0):
    return npsqrt(variance_percentage(values, width))

# returns the variance of a data series ignoring spikes larger than threshold*amplitude
def variance_percentage_clipped(x, values, width, high_threshold=1.5, low_threshold=0.2):
    width = width if width else len(values)
    bf = best_fit(x, values)                                                                                                  # compute best fit of values
    amp = amplitude(values)                                                                                                   # compute amplitude of values
    clipped_values = array([values[i] if abs(bf[i]-values[i])<high_threshold*amp else bf[i] for i in range(len(values))])     # replace anomalous values with the corresponding point in the best fit
    return variance_percentage(clipped_values, width)                                                                         # return variance percentage of clipped values

# returns the volatility of a data series ignoring spikes larger than threshold*amplitude (defined in variance_percentage_clipped)
def volatility_percentage_clipped(x, values):
    return npsqrt(variance_percentage_clipped(x, values, width=0))

# returns the given dictionary sorted by value in ascending order
def dsort_ascending(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}

# returns the given dictionary sorted by value in descending order
def dsort_descending(d):
    return {k: v for k, v in reversed(sorted(d.items(), key=lambda item: item[1]))}

# returns the key with the smallest value in the given dictionary
def min_from_dict(d):
    return list(d.keys())[list(d.values()).index(min(d.values()))]

# returns the key with the largest value in the given dictionary
def max_from_dict(d):
    return list(d.keys())[list(d.values()).index(max(d.values()))]

# returns a frequency dictionary for the given values
def frequency(values, freq=None):
    freq = freq if freq else {}
    assert isinstance(freq, dict), "freq must be a dictionary"
    for val in values:
        try:
            freq[val] += 1
        except KeyError:
            freq[val] = 1
    return freq

# returns a dictionary containing the number of instances of each value in the given dictionary
def value_frequency(d):
    freq = {}
    for val in d.values():
        try:
            freq[val] += 1
        except KeyError:
            freq[val] = 1
    return freq

# returns a dictionary containing lists of the values of each key contained in the give list of dictionaries
# requires the listed dictionaries to contain the same keys
def collect_keys(list_of_dicts):
    D = {key:[] for key in list_of_dicts[0]}
    for d in list_of_dicts:
        for key in d:
            D[key].append(d[key])
    return D

# returns the indices of the elements directly before the turning points in 'seq'
def turning_points(seq):
    return where(diff(sign(diff(seq))))[0]

# returns the linspace with the same limits and one more element than old_linspace
# assumes old_linspace is a linspace containing length-1 elements
def quick_linspace(old_linspace, length):
    return append(old_linspace*(length-2)/(length-1), old_linspace[-1], length)

# returns a list of arrays of length 'length' containing sequential elements from array 'arr'
def divide_array(arr, length):
    segments = ceil(len(arr)/length)
    return [arr[i*length:(i+1)*length] for i in range(segments)]

#========================================================================================================================================================================================
# DATA VISUALISATION

#------------------------------------------------------------------------------------------------------------
# INTERNAL FUNCTIONS

# the grid dimensions for each subplot arrangement
subplot_arrangement_dimensions = {1:(1,1), 2:(1,2), 3:(2,2), 4:(2,2), 5:(2,3), 6:(2,3), 7:(3,3), 8:(3,3), 9:(3,3)}
lsad = len(subplot_arrangement_dimensions)

# the figsize for each subplot arrangement
subplot_arrangement_figsizes = {1:array([6,5]),
                                2:array([9,4]),
                                3:array([9,7]), 4:array([9,7]),
                                5:array([11,6]), 6:array([11,6]),
                                7:array([11,8]), 8:array([11,8]), 9:array([11,8])}

# get a predetermined arrangement of subplots
def get_subplot_arrangement(subplots, figsize=None, dpi=None, scale=1, fontsize=None):
    assert isinstance(subplots, int) and 0 < subplots <= lsad, "'subplots' must be an integer from 1 to {}".format(lsad)
    figsize = figsize if figsize else subplot_arrangement_figsizes[subplots]*scale          # determine figure size - either passed into function call or default retrieved from global variable
    if isinstance(fontsize,(int,float)):                                                    # change the fontsize if a number was given
        plt.rcParams.update({'font.size': fontsize})                                        # change font size
    dim = subplot_arrangement_dimensions[subplots]                                          # get subplot grid dimensions
    fig = plt.figure(figsize=figsize, dpi=dpi)                   # initialise figure
    axes = []                                                    # initialise list to contain subplots
    i = 0                                                        # variable i is used to determine if the specified number of subplots have been created
    for row in range(dim[0]):                                    # iterate through number of rows specified by the dimension
        for column in range(dim[1]):                             # for each row, iterate through number of columns specified by the dimension
            axes.append(plt.subplot2grid(dim,(row,column)))      # create an empty subplot at this position in the figure
            i += 1                                               # increment i
            if i == subplots:
                break                                            # break the loop once the specified number of subplots has been created
    return fig, axes                                             # return the figure and the list of subplots axes

# plot timeprice data onta a given subplot axis
def axplot(ax, timeprices, symbol, show_best_fit=True, show_percentage=True, show_amplitude=True, show_legend=True, figsize=None):
    # data preparation
    times, prices = zip(*timeprices.items())                                                         # timeprice separation
    times, time_label = reasonable_times(times)                                                      # get reasonably formatted times with a time label
    if symbol:
        base, quote = split_symbol(symbol)
        title_symbol = base+'-'+quote
        price_symbol = base+'/'+quote
    else:
        title_symbol, price_symbol = '[SYMBOL]', 'SYMBOL'
    # main axis
    ax.plot(times, prices, label='Absolute Price', color='tab:blue')               # plot absoulte price line
    ax.set_title(title_symbol+' Price')                                                              # set plot title
    ax.set_xlabel("Time ({})".format(time_label))                                                    # set x-axis label
    ax.set_ylabel("Price ({})".format(price_symbol))                                                 # set y-axis label
    ax.set_xlim(times[0], 0)                                                                         # set x-axis limits
    axylim1, axylim2 = get_y_axis_limits(prices)                                                     # compute y-axis limits
    ax.set_ylim(axylim1, axylim2)                                                                    # set y-axis limits
    # best fit
    if show_best_fit:
         ax.plot(times, best_fit(times, prices), label="Av. Price", color='tab:orange')      # plot best fit line
    # percentage axis
    if show_percentage:
        percentage_prices = percentage_array(array(prices), prices[0])    # compute percentage changes in the absolute prices relative to the first price
        p_mask = mask_all(percentage_prices)                             # mask the percentage prices i.e. make them invisible
        ax2 = ax.twinx()                                                 # create secondary y-axis
        ax2.plot(times, p_mask)                                          # plot masked percentage prices
        ax2.set_ylabel("% Change")                                       # set secondary y-axis label
        ax2ylim1, ax2ylim2 = get_y_axis_limits(percentage_prices)        # compute secondary y-axis limits
        ax2.set_ylim(ax2ylim1, ax2ylim2)                                 # set secondary y-axis limits
        ax2.grid(axis='y')                                               # add horizontal grid lines at the percentage ticks
        ax.set_zorder(ax2.get_zorder()+1)                                # plot percentage axis elements behind main axis (including grid lines)
        ax.set_frame_on(False)                                           # needed to make the percentage axis elements visible after setting the z-order
    else:
        ax.grid(axis='y')                                                # add horizontal grid lines at the absolute price ticks if not using a percentage axis
    # amplitude
    if show_amplitude:
        amp = amplitude(prices)                                                                                                      # compute amplitdue
        ax.plot(times, array(best_fit(times, prices))+amp, label="Amplitude", color='tab:green', linestyle='dashed')      # plot upper amplitude line
        ax.plot(times, array(best_fit(times, prices))-amp, color='tab:green', linestyle='dashed')                         # plot lower amplitude line
    # legend
    if show_legend:
        ax.legend()       # show a legend
    return

# returns a tuple containing the lower and upper y-axis limits, respectively, for the given y values
# smaller pad gives more padding (I know it's confusing)
def get_y_axis_limits(y, pad=10):
    miny = min(y)                                                              # compute lowest price (to set the y-axis limit)
    maxy = max(y)                                                              # compute highest price (to set the y-axis limit)
    axylim1 = ((pad+1)*miny-maxy)/pad      # lower y-axis limit - expressions equal to min(y)-(amp/pad), where amp = max(y)-min(y)
    axylim2 = ((pad+1)*maxy-miny)/pad      # upper y-axis limit - expressions equal to max(y)+(amp/pad), where amp = max(y)-min(y)
    return axylim1, axylim2

# masks all points in values, apart from the first and last, so that they may be plotted invisible
def mask_all(values):
    m = repeat(True, len(values))
    m[0], m[-1] = False, False
    return masked_where(m, array(values))

# masks all zeros in 'values'
def mask_zeros(values):
    v = array(values)
    m = ~make_mask(v)
    return masked_where(m, v)

# returns a reasonably formatted list of times corresponding to the time period between the given timestamps
def reasonable_times(times, zero_index=-1):
    delta = times[-1] - times[0]                                # data period in milliseconds
    time_label, divider = _appropriate_time_interval_(delta)    # compute reasonable scale for x axis
    times = (array(times) - times[zero_index])/divider          # rescale x values
    return times, time_label

# returns the interval name and scale factor for the given time delta (in milliseconds)
# only used by reasonable_times(), don't worry about it
def _appropriate_time_interval_(delta):
    if delta < 1e7:
        return ('minutes', 6e4)       # no. milliseconds in a minute
    elif delta < 2*1e8:
        return ('hours', 3.6e6)       # no. milliseconds in an hour
    elif delta < 2*1e9:
        return ('days', 8.64e7)       # no. milliseconds in a day
    elif delta < 1e10:
        return ('weeks', 6.048e8)     # no. milliseconds in a week
    else:
        return ('months', 2.6298e9)   # no. milliseconds in a month
    return

#------------------------------------------------------------------------------------------------------------
# USER FUNCTIONS

# plots the given timeprices with optional line of best fit, percentage axis, amplitude, and legend
def plot(timeprices, symbol='', show_best_fit=True, show_percentage=True, show_amplitude=True, show_legend=True, figsize=None):
    fig, axes = get_subplot_arrangement(1, figsize)                                                       # get a figure with a single subplot
    axplot(axes[0], timeprices, symbol, show_best_fit, show_percentage, show_amplitude, show_legend)      # plot the timeprice data on the one subplot
    fig.tight_layout()                                                                                    # make it tight
    return

# plots the given set of symbol:timeprice stored in Timeprices with optional line of best fit, percentage axis, amplitude, and legend
def plot_multiple(Timeprices, show_best_fit=True, show_percentage=True, show_amplitude=True, show_legend=True, figsize=None):
    assert len(Timeprices) <= lsad, "Too many symbols - must be no more than {}".format(lsad)
    fig, axes = get_subplot_arrangement(len(Timeprices), figsize)                                                # get a figure with the correct number of subplots
    for symbol, ax in zip(Timeprices,axes):                                                                      # associate each symbol in Timeprices with a subplot and iterate through the pairs
        axplot(ax, Timeprices[symbol], symbol, show_best_fit, show_percentage, show_amplitude, show_legend)      # plot the timeprice data on its corresponding subplot
    fig.tight_layout()                                                                                           # make it tight
    return

# plots the 'subplots' most volatile USDT/BUSD symbols and returns the clipped percentage volatility for all USDT/BUSD symbols
# SHOULD PROBABLY SEPARATE THE TIMEPRICE AND VOLATILITY CALCULATIONS INTO SEPARATE FUNCTIONS
def plot_most_volatile(subplots=9, timeprices=None, interval='1m', start=minutes_ago(60), end=None, show_best_fit=True, show_percentage=True, show_amplitude=True, show_legend=True, figsize=None):
    print("COMPUTING SYMBOL VOLATILITY...")
    end = end if end else now()
    symbols = get_hos_usd_symbols()
    Timeprices = timeprices if any([timeprices]) else get_historical_prices_multi(symbols, interval, start, end)
    vols = dsort_descending({symbol:volatility_percentage_clipped(list(Timeprices[symbol].keys()), list(Timeprices[symbol].values())) for symbol in Timeprices})
    plot_vols(vols, Timeprices, subplots, show_best_fit, show_percentage, show_amplitude, show_legend, figsize)
    return vols

def plot_vols(vols, tps, subplots=9, show_best_fit=True, show_percentage=True, show_amplitude=True, show_legend=True, figsize=None):
    fig, axes = get_subplot_arrangement(subplots, figsize)
    for symbol, ax in zip(list(vols)[:subplots],axes):                 # associate the first 'subplot' symbols in vols each with a subplot and iterate through the pairs
        axplot(ax, tps[symbol], symbol, show_legend=False)      # plot the timeprice data on its corresponding subplot
    fig.tight_layout()                                                 # make it tight
    plt.show()

# plots a histogram of the given data sequence
def histogram(seq, bins='auto', rwidth=0.8, title=''):
    hist_output = plt.hist(seq, bins=bins, rwidth=rwidth)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    plt.title(title)
    plt.show()
    return hist_output

#========================================================================================================================================================================================
# RESOURCE MANAGEMENT

# creates a space-separated text file of the given 'data' under resources/'filename'.txt
def create_resource(filename, data):
    assert isinstance(data, list), "data must be a list"                                               # data must be a list
    assert any(data), "data empty"                                                                     # list must have things in it
    filename = str(filename)                                                                           # ensure filename is a string
    if (filename+'.txt') in listdir('resources'):                                                      # check if filename already in use
        response = input(filename+'.txt already exists. Overwrite existing '+filename+'.txt? y/n')     # ask user whether they want to overwrite
        assert response in ['y', 'n'], "response must be y or n (yes or no)"                           # response must be y or n
        if response == 'y':
            with open('resources/'+filename+'.txt', 'w') as f:
                for d in data:
                    f.write(str(d)+' ')                                           # if y: overwrite old file with new data
            print(c.G+"~ Data successfully saved to resources/"+filename+'.txt ~'+c.X)
        else:
            print("Please choose a different name.")                              # if n: aborts function and tells user to pick a different name
    else:
        with open('resources/'+filename+'.txt', 'w') as f:
                for d in data:
                    f.write(str(d)+' ')                                           # otherwise data is written to filename
        print(c.G+"~ Data successfully saved to resources/"+filename+'.txt ~'+c.X)
    return

# returns a list containing the data requested by the given filename (excluding .txt)
def get_resource(filename):
    with open("resources/common/"+filename+".txt", "r") as f:       # open a resource txt file - assumes the name exists
        return f.read().split()                                     # return the file as a list of elements split at spaces

# saves the given dictionary of symbols:{times:prices} to a txt file
def timeprices_to_file(timeprices, filename=None):
    assert isinstance(timeprices, dict), "timeprices must be a dictionary"         # timeprices must be a dictionary
    filename = filename if filename else input("please give a file name")          # input filename is requested if not given in function call
    with open("resources/user/"+filename+".txt", "w") as f:
            for symbol in timeprices:
                f.write(symbol+';')                                                # symbol is written followed by ;
                for time, price in timeprices[symbol].items():
                    f.write(str(time)+'~'+str(price)+';')                          # then each timeprice is written as time~price;
                f.write(' ')                                                       # a space is written to separate each symbol's data
    print(c.G+"~ Timeprice data successfully saved to resources/user/"+filename+".txt ~"+c.X)
    return

# returns a dictionary of symbols:{times:prices} from txt file
def timeprices_from_file(filename):
    assert (str(filename)+'.txt') in listdir('resources/user'), "no file by that name in resources/user"        # filename must be in directory
    filename = str(filename)                                                                                    # ensure filename is a string
    timeprices = {}                                              # initialise timeprice dictionary
    with open("resources/user/"+filename+".txt", "r") as f:
        symbol_timeprices = f.read().split()                     # separate symbol data by spaces
        for symbol_timeprice in symbol_timeprices:               # iterate the following for each symbol data string
            stp_split = symbol_timeprice.split(';')              # separate data for each symbol by ;
            symbol = stp_split[0]                                # first element is symbol name
            if not stp_split[-1]:                                # there is a space at the end of each split timeprice list and i cba to deal with it atm so we just pop it here
                stp_split.pop(-1)
            timeprices[symbol] = {int(timeprice.split('~')[0]):float(timeprice.split('~')[1]) for timeprice in stp_split[1:]}      # construct timeprices from remaining elements
    print(c.G+"~ Timeprice data successfully retrieved from resources/user/"+filename+".txt ~"+c.X)
    return timeprices

# saves the given volatility data to a txt file with the next appropriate file name i.e. 'volatility(n+1)'
def volatility_to_file(data):
    vols = [key+':'+str(val) for key, val in data.items()]                                            # assumes data is a dictionary of symbol:volatility
    try:
        recent_vol_number = int(listdir('resources/user/volatility')[-1][10:-4])                      # get the volatility number of the last filename in resources/user/volatility
    except IndexError:                                                                                # index error is raise if there are no files in resources/user/volatility
        recent_vol_number = 0                                                                         # set recent_vol_number = 0 in this case so that the file 'volatility001' is created
    new_vol_number_string = ('0'*(3-len(str(recent_vol_number+1))))+str(recent_vol_number+1)          # add 1 to it and format back into leading zero form
    filename = 'volatility{}.txt'.format(new_vol_number_string)                                       # define new filename
    with open('resources/user/volatility/'+filename, 'w') as f:
        for v in vols:
            f.write(str(v)+' ')                                                                       # write volatility data to file
    print(c.G+"~ Volatility data successfully saved to resources/user/volatility/"+filename+' ~'+c.X)
    return

# saves a dictionary 'd' to json file with path 'filename'
def dict_to_json(d, filename):
    if not filename[-5:] == '.json':
        filename = filename+'.json'
    with open(filename, 'w') as f:
        jdump(d, f, ensure_ascii=False, indent=4)
    return

# returns a dictionary from json file with path 'filename'
def json_to_dict(filename):
    if not filename[-5:] == '.json':
        filename = filename+'.json'
    with open(filename) as f:
        d = jload(f)
    return d

#========================================================================================================================================================================================
# PRICE DATA

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HISTORICAL KLINES

# simple function to return historical klines
def get_klines(symbol, interval='1m', start=None, end=None):
    start = start if start else hours_ago(1)
    end = end if end else now()
    assert isinstance(start,int), "start must be an int"
    assert isinstance(end,int), "end must be an int"
    klines = send_public_request("/api/v3/klines", {'symbol':symbol, 'interval':interval, 'startTime':start, 'endTime':end, 'limit':1000})
    if len(klines) == 1000:
        warn('KLINE LIMIT REACHED: 1000 KLINES RETURNED')
    return klines

# returns a dictionary of times:prices for the given symbol between 'start' and now
def get_historical_prices(symbol, interval='1m', start=None, end=None):
    klines = get_klines(symbol, interval, start, end)
    if not any(klines):
        _no_historical_data_error_(symbol, start, end)
    timeprices = {}                                                                      # initialise timeprice dictionary
    delta = klines[1][0] - klines[0][0]                                                  # get time difference between klines
    for k in klines:
        timeprices[k[0]] = float(k[1])                                                   # store open price at the beginning of the kline time
        timeprices[int(k[0]+(delta/3))] = float(k[2])                                    # store high price at one third the time through the kline
        timeprices[int(k[0]+(2*delta/3))] = float(k[3])                                  # store low price two thirds the time throught the kline
    timeprices[klines[-1][0]] = float(klines[-1][4])                                     # store the close price for the final kline at the end
    return timeprices                                                                    # > # the close price of previous klines are ommitted as they are the same as the open price of the following kline

# returns a dictionary of times:historical prices for the given symbol between 'start' and now
def get_average_historical_prices(symbol, interval='1m', start=None, end=None):
    klines = get_klines(symbol, interval, start, end)
    if not any(klines):
        _no_historical_data_error_(symbol, start, end)
    timeprices = {}                                                                             # initialise timeprice dictionary
    for k in klines:
        timeprices[k[0]] = mean([float(k[1]), float(k[2]), float(k[3]), float(k[4])])           # store the average of the open, high, low, close prices at the beginning of the kline time
    return timeprices

# returns a list of historical prices for the given symbol without any associated time data
def get_historical_prices_list(symbol, interval='1m', start=None, end=None):
    start = start if start else hours_ago(1)
    end = end if end else now()
    return list(get_historical_prices(symbol, interval, start, end).values())                       # get a list of the values of the timeprice dictionary returned by get_historical_prices

# returns a list of average historical prices for the given symbol without any associated time data
def get_average_historical_prices_list(symbol, interval='1m', start=None, end=None):
    start = start if start else hours_ago(1)
    end = end if end else now()
    return list(get_average_historical_prices(symbol, interval, start, end).values())               # get a list of the values of the timeprice dictionary returned by get_average_historical_prices

# returns a dictionary of symbols:{times:prices} for the given list of symbols betweem 'start' and 'end'
def get_historical_prices_multi(symbols, interval='1m', start=None, end=None):
    start = start if start else minutes_ago(10)
    end = end if end else now()
    return _historical_prices_multi_(symbols, interval, start, end, get_historical_prices)

# returns a dictionary of symbols:{times:prices} for the given list of symbols betweem 'start' and 'end'
def get_average_historical_prices_multi(symbols, interval='1m', start=None, end=None):
    start = start if start else minutes_ago(10)
    end = end if end else now()
    return _historical_prices_multi_(symbols, interval, start, end, get_average_historical_prices)

# internal function for collecting multiple historical kline datasets, used by get_historical_prices_multi and get_average_historical_prices_multi
def _historical_prices_multi_(symbols, interval, start, end, func):
    symbols_set = set(symbols)
    all_symbols_set = set(get_all_symbols())
    assert symbols_set.issubset(all_symbols_set), "invalid symbol(s): {}".format(symbols_set-all_symbols_set)
    timeprices = {}                                                                         # initialise timeprice dictionary
    percent = 0                                                                             # initialise processing percentage variable - required for progess tracker prints
    for symbol in symbols:
        try:
            timeprices[symbol] = func(symbol, interval, start, end)                         # store historical data for symbol using provided historical price function
            percent = progress_tracker(percent, symbol, symbols, 10)                        # update progress tracker
        except ValueError as e:
            print(c.Y+"    {}".format(str(e))+c.X)
            pass
    return timeprices

# returns a historical timeprice dictionary containing the OHLC prices labelled and separated
def get_historical_prices_detailed(symbol, interval='1m', start=None, end=None):
    klines = get_klines(symbol, interval, start, end)
    if not any(klines):
        _no_historical_data_error_(symbol, start, end)
    info = {}
    for k in klines:
        info[k[0]] = {'open':float(k[1]), 'high':float(k[2]), 'low':float(k[3]), 'close':float(k[4]), 'trades':k[8]}
    return info

# returns the symbol:timeprices dictionary for all hos usd symbols returned by get_hos_usd_symbols()
def get_hos_usd_timeprices(start, end=None, interval='1m'):
    symbols = get_hos_usd_symbols()
    Timeprices = get_historical_prices_multi(symbols, interval, start, end)
    return Timeprices

# constructs and throws the Value Error for the case where no historical data is returned from get_klines
def _no_historical_data_error_(symbol, start, end):
    startdate = dt.fromtimestamp(start//1000).strftime("%H:%M %d/%m/%Y")
    enddate = dt.fromtimestamp(end//1000).strftime("%H:%M %d/%m/%Y")
    raise ValueError("No historical data for {} between {} and {}".format(symbol, startdate, enddate))
    return

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CURRENT PRICE

def get_all_tickers():
    return send_public_request("/api/v3/ticker/price")

# returns the current price of the given symbol as per the get_all_tickers() method as a string
def get_current_price(symbol):
    assert symbol in get_all_symbols(), "invalid symbol: {}".format(symbol)
    for ticker in get_all_tickers():                                      # iterate through the symbol;price ticker list
        if ticker['symbol'] == symbol:                                           # find the one you want
            return float(ticker['price'] )                                       # return the price
    return

# returns a dictionary containining the current price for all symbols
def get_all_current_prices():
    return {ticker['symbol']:float(ticker['price']) for ticker in get_all_tickers()}

# returns the current (average over 5 min) price of the given symbol as a string
def get_average_price(symbol):
    return send_public_request("/api/v3/avgPrice", {"symbol":symbol})    # faster than get_current_price but the less accurate to the true price

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# WEBSOCKET

# websocket message handler that simply prints the message
def print_handler(msg):
    print(msg)
    return

# returns the trade price or close price for trade or kline websocket messages, respectively
def price_from_websocket(msg):
    try:
        return msg['k']['c']    # kline socket messages
    except KeyError:
        return msg['p']         # trade socket messages

# returns the trade time and spot price or close price for kline websocket messages in tuple form
def timeprice_from_websocket(msg):
    return (msg['E'], float(msg['k']['c']))    # kline socket messages
    #return (msg['E'], float(msg['p']))         # trade socket messages
    #return (msg['E'], msg['c'])                # ticker socket message

# returns the trade time and trade price or close price for kline websocket messages in string form
def timeprice_string_from_websocket(msg):
    return str(msg['E'])+';'+msg['k']['c']    # kline socket messages
    #return str(msg['E'])+';'+msg['p']         # trade socket messages
    #return str(msg['E'])+';'+msg['c']         # ticker socket message

#========================================================================================================================================================================================
# ASSETS & SYMBOLS

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RETRIEVING ASSET & SYMBOL RESOURCES

# returns the all the information you might need about basically anything
def get_exchange_info():
    return send_public_request("/api/v3/exchangeInfo")

# returns the filter dict with the given filter name for the given exchange info dict if it is a filter, otherwise returns an empty dict
def get_filter(filterType, info):
    for fil in info['filters']:
        if fil['filterType'] == filterType:
            return fil
    return {}

# returns trade fee information for 'symbol'
def get_trade_fee(symbol):
    return send_signed_request("GET", "/sapi/v1/asset/tradeFee", {'symbol':symbol})

# returns the taker commission for 'symbol' as a float
def get_tc_float(symbol):
    return float(get_trade_fee(symbol)[0]['takerCommission'])

# returns all base assets on binance, as defined in resources/common/base_assets.txt
def get_base_assets():
    return get_resource('base_assets')

# returns all quote assets on binance, as defined in resources/common/quote_assets.txt
def get_quote_assets():
    return get_resource('quote_assets')

# returns all assets on binance, base and quote, as defined in resources/common/base_assets.txt and resources/common/quote_assets.txt
def get_all_assets():
    bases = set(get_base_assets())              # get the set of all base assets
    bases.update(set(get_quote_assets()))       # add all quote assets to the set
    return list(bases)                          # return the set as a list

# returns all symbols on binance, as defined in resources/common/all_symbols.txt
def get_all_symbols():
    return get_resource('all_symbols')

# returns all symbols for which spot trading is allowed, as defined in resources/common/spot_symbols.txt
def get_spot_symbols():
    return get_resource('spot_symbols')

# returns a list of symbols with open trading markets, as defined in resources/common/open_symbols.txt
def get_open_symbols():
    return get_resource('open_symbols')

# returns a list of spot symbol with open trading markets
def get_open_spot_symbols():
    return set(get_spot_symbols()) & set(get_open_symbols())

# returns all symbols with historical price data, as defined in resources/common/historical_symbols.txt
def get_historical_symbols():
    return get_resource('historical_symbols')

# returns all symbols which both have historical price data and are spot-tradable
def get_historical_spot_symbols():
    spot = set(get_spot_symbols())           # get set of all spot symbols
    hist = set(get_historical_symbols())     # get set of all historical symbols
    return list(set(spot) & set(hist))       # return set intersection of the two sets as a list

# returns all historical open market spot symbols
def get_hos_symbols():
    return set(get_open_spot_symbols()) & set(get_historical_symbols())

# returns all historical open market spot symbols with a USD stable quote asset
def get_hos_usd_symbols():
    hos_symbols = get_hos_symbols()
    #usd_quotes = ['BUSD', 'USDT', 'USDC', 'USDS', 'USDP', 'TUSD']
    usd_quotes = ['BUSD', 'USDT']      # atm, only BUSD and USDT are required to specify all unique base assets with usd stable quotes
    hos_usd_symbols = []
    usd_bases = []
    for quote in usd_quotes:
        for symbol in get_quote_symbols(quote, hos_symbols):
            base = symbol[:-len(quote)]
            if base not in usd_bases:
                hos_usd_symbols.append(symbol)
                usd_bases.append(base)
    return hos_usd_symbols

# returns the base asset for a given symbol
def get_base_asset(symbol):
    assert isinstance(symbol,str), "symbol must be a string, not {}".format(type(symbol))
    return symbol[:-len(get_quote_asset(symbol))]                               # return the part of the symbol excluding its quote asset

# returns the quote asset for a given symbol
def get_quote_asset(symbol):
    assert isinstance(symbol,str), "symbol must be a string, not {}".format(type(symbol))
    for quote in get_quote_assets():                                            # iterate through all quote assets
        if symbol[-len(quote):] == quote:                                       # return the quote asset if it is the same as the end of the symbol
            return quote
    raise FailureError("No known quote asset for {}".format(symbol))
    return

# returns the list (can be empty) of symbols constituted by asset1 and asset2
def get_symbols(asset1, asset2):
    all_symbols = get_all_symbols()
    symbols = []
    if asset1+asset2 in all_symbols:
        symbols.append(asset1+asset2)
    if asset2+asset1 in all_symbols:
        symbols.append(asset2+asset1)
    return symbols

# returns all symbols with the given base asset in the given set of symbols
def get_base_symbols(base, symbols=None):
    assert isinstance(base,str), "base must be a string e.g. 'BTC'"
    assert base in get_base_assets(), "invalid base asset: {}".format(base)
    quote_assets = get_quote_assets()
    symbols = symbols if symbols else get_all_symbols()
    return [symbol for symbol in symbols if (symbol[:len(base)] == base and symbol[len(base):] in quote_assets)]

# returns all symbols with the given quote asset in the given set of symbols
def get_quote_symbols(quote, symbols=None):
    assert isinstance(quote,str), "quote must be a string e.g. 'USDT'"
    assert quote in get_quote_assets(), "invalid quote asset: {}".format(quote)
    base_assets = get_base_assets()
    symbols = symbols if symbols else get_all_symbols()
    return [symbol for symbol in symbols if (symbol[-len(quote):] == quote and symbol[:-len(quote)] in base_assets)]

# returns a dictionary containin two values: a list of the given asset's base symbols and of its quote symbols, with keys 'base_symbols' and 'quote_symbols', respectively
def get_hybrid_symbols(asset, symbols=None):
    assert isinstance(asset,str), "asset must be a string e.g. 'BUSD'"
    base_assets = get_base_assets()
    quote_assets = get_quote_assets()
    assert asset in set(base_assets) & set(quote_assets), "invalid hybrid asset: {}".format(asset)
    symbols = symbols if symbols else get_all_symbols()
    base_symbols = [symbol for symbol in symbols if (symbol[:len(asset)] == asset and symbol[len(asset):] in quote_assets)]
    quote_symbols = [symbol for symbol in symbols if (symbol[-len(asset):] == asset and symbol[:-len(asset)] in base_assets)]
    return dict(zip(['base_symbols','quote_symbols'],[base_symbols,quote_symbols]))

# returns an interable containing all the valid base and/or quote assets for the given asset
# def get_friends(asset):
#     if asset in self.exclusive_bases:                                                        # if the asset in an exclusive base asset
#         return [s[len(asset):] for s in get_base_symbols(asset, self.valid_symbols)]           # return the valid quotes for asset
#     if asset in self.exclusive_quotes:                                                       # if the asset in an exclusive quote asset
#         return [s[:-len(asset)] for s in get_quote_symbols(asset, self.valid_symbols)]         # return the valid bases for asset
#     else:                                                                                    # if the asset in a hybrid asset
#         bases = [s[:-len(asset)] for s in get_quote_symbols(asset, self.valid_symbols)]        # get valid bases for hybrid asset as a quote
#         quotes = [s[len(asset):] for s in get_base_symbols(asset, self.valid_symbols)]         # get valid quotes for hybrid asset as a base
#         return set(bases) | set(quotes)                                                        # return the set of valid bases and quotes for asset
#     return

# returns the list of all assets which form a spot symbol
def get_open_spot_assets():
    all_assets = get_all_assets()               # get all assets
    spot_symbols = get_open_spot_symbols()      # get all open spot symbols
    bases = get_base_assets()                   # get all base assets
    quotes = get_quote_assets()                 # get all quote assets
    valid_assets = []                           # initialise list to contain valid spot assets
    for asset in all_assets:                                                                 # iterate through all assets
        asset_symbols = set()                                                                # initialise set to contain asset spot symbols
        if asset in bases:                                                                   # if asset is a base asset:
            asset_symbols = asset_symbols | set(get_base_symbols(asset, spot_symbols))       # add any/all of its valid spot symbols to the asset_symbols set
        if asset in quotes:                                                                  # if asset is a quote asset:
            asset_symbols = asset_symbols | set(get_quote_symbols(asset, spot_symbols))      # add any/all of its valid spot symbols to the asset_symbols set
        if any(asset_symbols):                                                               # if asset_symbols contains any symbols:
            valid_assets.append(asset)                                                       # asset is a valid spot asset
    return valid_assets                      # return the list of valid spot assets

#------------------------------------------------------------------------------------------------------------
# UPDATING SYMBOL RESOURCES

# updates all symbol resources
def update_symbol_resources(skip_hist=False):
    check_api()
    print("UPDATING SYMBOL RESOURCES...\n")
    update_all_symbols()
    update_base_assets()
    update_spot_symbols()
    update_open_symbols()
    update_quantity_limits()
    update_asset_precisions()
    update_taker_commissions()
    if not skip_hist:                   # update_historical_symbols takes a long time so you can skip it if you want by passing skip_hist=True
        update_historical_symbols()
    else:
        print(c.Y+"SKIPPED HISTORICAL SYMBOLS UPDATE\n"+c.X)
    print("~ UPDATED SYMBOL RESOURCES ~")
    return

# updates resources/common/all_symbols.txt
def update_all_symbols():
    print("UPDATING ALL SYMBOLS...")
    all_symbols = set([ticker['symbol'] for ticker in get_all_tickers()])
    _update_resource_('all_symbols', 'ALL SYMBOLS', all_symbols)                # do rest of update
    return

# updates resources/common/base_assets.txt
def update_base_assets():
    print("UPDATING BASE ASSETS...")
    unquote = []
    base_assets = set()
    # fetch current base assets
    for symbol in get_all_symbols():
        try:
            base_assets.add(get_base_asset(symbol))
        except FailureError:
            unquote.append(symbol)
    # unknown quote asset symbols
    if any(unquote):
        print(c.R+"    UNQUOTED SYMBOLS: {}".format(unquote)+c.X)            # quote_assets.txt must be manually updated
        raise Exception("Please manually update resources/common/quote_assets.txt")      # raise exception requesting the update and block any further execution until it is done
    _update_resource_('base_assets', 'BASE ASSETS', base_assets)                # do rest of update
    return

# updates resources/common/spot_symbols.txt
def update_spot_symbols():
    print("UPDATING SPOT SYMBOLS...")
    spot_symbols = set([info['symbol'] for info in get_exchange_info()['symbols'] if info['isSpotTradingAllowed']])
    _update_resource_('spot_symbols', 'SPOT SYMBOLS', spot_symbols)                # do rest of update
    return

# updates resources/common/open_symbols.txt
def update_open_symbols():
    print("UPDATING OPEN MARKET SYMBOLS...")
    open_symbols = set([info['symbol'] for info in get_exchange_info()['symbols'] if info['status'] == 'TRADING'])
    _update_resource_('open_symbols', 'OPEN MARKET SYMBOLS', open_symbols)                # do rest of update
    return

# updates resources/common/quantity_limits.txt
def update_quantity_limits():
    print("UPDATING QUANTITY LIMITS...")
    # fetch current quantity limits
    quantity_limits = _get_quantity_limits_()
    _update_resource_('quantity_limits', 'QUANTITY LIMITS', quantity_limits)                # do rest of update
    return

def _get_quantity_limits_():
    qls = set()
    for info in get_exchange_info()['symbols']:
        symbol = info['symbol']
        lot_size_filter = get_filter('LOT_SIZE', info)
        notional_filter = get_filter('MIN_NOTIONAL', info)
        notional_filter = notional_filter if any(notional_filter) else get_filter('NOTIONAL', info)    # notional filter can be either 'NOTIONAL' or 'MIN_NOTIONAL'
        minQty = lot_size_filter['minQty']
        maxQty = lot_size_filter['maxQty']
        minNotional = notional_filter['minNotional']
        string = "{};{};{};{}".format(symbol, minQty, maxQty, minNotional)
        qls.add(string)
    return qls

# updates resources/common/asset_precisions.txt
def update_asset_precisions():
    print("UPDATING ASSET PRECISIONS...")
    asset_precisions = set()
    for info in get_exchange_info()['symbols']:
        base, quote = split_symbol(info['symbol'])
        asset_precisions.add(base+';'+str(info['baseAssetPrecision']))
        asset_precisions.add(quote+';'+str(info['quoteAssetPrecision']))
    _update_resource_('asset_precisions', 'ASSET PRECISIONS', asset_precisions)                # do rest of update
    return

# updates resources/common/taker_commissions.txt
def update_taker_commissions():
    print("UPDATING TAKER COMMISSIONS...")
    # fetching trade fee takes too long for all symbols and rarely changes anyway so we just fetch for any new symbols
    symbols_done = set(list(get_taker_commissions().keys()))
    all_symbols = set(get_all_symbols())
    remaining_symbols = all_symbols - symbols_done
    with open("resources/common/taker_commissions.txt", 'a') as f:
        for symbol in remaining_symbols:
            try:
                tc = '{};{}'.format(symbol, get_tc_float(symbol))
                f.write(tc+'\n')
                print(c.M+"    {} ADDED TO TAKER COMMISSIONS".format(tc)+c.X)
            except IndexError:
                # some symbols have no trade fee data for some reason so we just set to 0.001 like the others
                f.write(symbol+';0.001\n')
                print(c.Y+"    NO TRADE FEE DATA FOR {}, SETTING TO 0.001".format(symbol)+c.X)
    # BTC commissions
    with open("resources/common/taker_commissions.txt", 'r') as f:
        taker_commissions = f.read().split()
    tcs = get_taker_commissions()
    btc_symbols = get_base_symbols('BTC')
    btc_commissions = {symbol:get_tc_float(symbol) for symbol in btc_symbols}
    known_btc_commissions = {symbol:tcs[symbol] for symbol in btc_symbols}
    for symbol in btc_symbols:
        if btc_commissions[symbol] == known_btc_commissions[symbol]:
            continue
        old = '{};{}'.format(symbol, known_btc_commissions[symbol])
        new = '{};{}'.format(symbol, btc_commissions[symbol])
        taker_commissions.remove(old)
        taker_commissions.append(new)
        print(c.Y+"    {} REMOVED FROM TAKER COMMISSIONS".format(old)+c.X)
        print(c.M+"    {} ADDED TO TAKER COMMISSIONS".format(new)+c.X)
    with open("resources/common/taker_commissions.txt", 'w') as f:
        for tc in taker_commissions:
            f.write(tc+'\n')
    print(c.G+"~ UPDATED TAKER COMMISSIONS ~\n"+c.X)
    return

# updates resources/common/historical_symbols.txt
def update_historical_symbols():
    print("UPDATING HISTORICAL SYMBOLS...")
    undecided_symbols = list(set(get_all_symbols()) - set(get_historical_symbols()))            # create list of symbols which so far have no historical price data
    hist_symbols = []                                                                           # initialise list to contain newly historical symbols
    percent = 0                                                                                 # initialise processing percentage variable - required for progess tracker prints
    for symbol in undecided_symbols:                                                            # iterate through undecided symbols
        try:
            assert any(get_klines(symbol, '15m', days_ago(1)))                                  # try to retrieve historical data
            hist_symbols.append(symbol)                                                         # if successful, append symbol to hist_symbols
            print(c.M+"    {} ADDED TO HISTORICAL SYMBOLS".format(symbol)+c.X)
        except AssertionError:
            #print(c.Y+"    {} has no historical data".format(symbol)+c.X)
            pass                                                                                # if unsuccessful, move on
        percent = progress_tracker(percent, symbol, undecided_symbols, 10)                       # update progress tracker
    with open('resources/common/historical_symbols.txt', 'a') as f:
        for symbol in hist_symbols:
            f.write(symbol+'\n')                                       # append newly historical symbols to the existing file
    print(c.G+"~ UPDATED HISTORICAL SYMBOLS ~\n"+c.X)
    return

def _update_resource_(file, label, current_data):
    # fetch known data
    with open('resources/common/{}.txt'.format(file), 'r') as f:
        known_data = set(f.read().split())
    # add new data
    new_data = current_data - known_data
    with open('resources/common/{}.txt'.format(file), 'a') as f:
        for data in new_data:
            f.write(data+'\n')
            print(c.M+"    {} ADDED TO {}".format(data, label)+c.X)
    # remove old data
    removed_data = known_data - current_data
    with open('resources/common/{}.txt'.format(file), 'r') as f:
        known_data = f.read().split()
    for data in removed_data:
        known_data.remove(data)
        print(c.Y+"    {} REMOVED FROM {}".format(data, label)+c.X)
    with open('resources/common/{}.txt'.format(file), 'w') as f:
        for data in known_data:
            f.write(data+'\n')
    print(c.G+"~ UPDATED {} ~\n".format(label)+c.X)
    return

#------------------------------------------------------------------------------------------------------------
# MISC SYMBOL INFORMATION

# returns the asset type i.e. base, quote, or hybrid (both base and quote)
def get_asset_type(asset):
    assert asset in get_all_assets(), "Invalid asset: {}".format(asset)
    if asset in set(get_base_assets()) - set(get_quote_assets()):
        return 'base'
    elif asset in set(get_quote_assets()) - set(get_base_assets()):
        return 'quote'
    else:
        return 'hybrid'

# returns the given symbol with a dash in the middle e.g. BTCUSDT -> BTC-USDT
def dash_symbol(symbol):
    assert isinstance(symbol,str), "symbol must be a string e.g. 'BTCUSDT'"
    assert symbol in get_all_symbols(), "invalid symbol {}".format(symbol)
    return get_base_asset(symbol) + '-' + get_quote_asset(symbol)                # get base and quote assets from the symbol and reconstruct the string with a dash in the middle

# returns the tuple of the given symbol's base and quote assets
def split_symbol(symbol):
    assert symbol in get_all_symbols(), "invalid symbol: {}".format(symbol)
    return (get_base_asset(symbol), get_quote_asset(symbol))                     # get the base and quote assets from the symbol return them separately as a tuple

# returns a dictionary containing {symbol:{minQty,maxQty,minNotional}} for all symbols, as defined in resources/common/quantity_limits.txt
def get_all_quantity_limits():
    qls = {}                                             # initialise outer dictionary to contain symbol:{min,max,notion}
    qls_split = get_resource('quantity_limits')          # get list of quantity limit string elements
    for ql in qls_split:
        ql_split = ql.split(';')                         # split the symbol data by ; - the first element is then the symbol name
        minQty = float(ql_split[1])                      # store the minQty as a float before insertion to dict - because of the Decimal bug, minQty needs to be 1 rather than 1.0 if it is '1.0'
        qls[ql_split[0]] = {'minQty':int(minQty) if int(minQty) else minQty,'maxQty':float(ql_split[2]),'minNotional':float(ql_split[3])}     # construct dictionary value to be stored for the key 'symbol'
    return qls

# returns a minQty, maxQty tuple for the given symbol
def get_quantity_limits(symbol):
    ql = get_all_quantity_limits()[symbol]                     # get the quantity limits dictionary for the given symbol
    return (ql['minQty'], ql['maxQty'], ql['minNotional'])     # return the three values as a tuple

# returns the approximate value of the highest minimum notional value across all symbols
# pass an optional upr to reduce computation time
def highest_min_notional(upr=None):
    mns = {}
    qls = get_all_quantity_limits()
    upr = upr if upr else get_usd_price_reference()
    for symbol in get_open_spot_symbols():
        min_notional = qls[symbol]['minNotional']
        quote = get_quote_asset(symbol)
        mns[symbol] = min_notional*upr[quote]
    return mns[max_from_dict(mns)]

# returns a dictionary containing asset:precision pairs
def get_asset_precisions():
    return {ap[0]:int(ap[1]) for ap in map(lambda a: a.split(';'), get_resource('asset_precisions'))}

# returns the asset precision for the given asset, as defined in resources/common/asset_precisions.txt
def get_asset_precision(asset):
    return get_asset_precisions()[asset]

# returns a dictionary of symbol:taker_commission pairs e.g. trading fee
# taker commissions refer to the fee paid per unit of asset receieved
def get_taker_commissions():
    return {tc[0]:float(tc[1]) for tc in map(lambda t: t.split(';'), get_resource('taker_commissions'))}

# returns the taker commission for the given symbol
def get_taker_commission(symbol):
    return get_taker_commissions()[symbol]

# returns bool whether quote quantity is allowed for market trading symbol
def quote_order_qty_allowed(symbol):
    return {info['symbol']:info['quoteOrderQtyMarketAllowed'] for info in get_exchange_info()['symbols']}[symbol]

#========================================================================================================================================================================================
# ACCOUNT & API

# returns account information
def get_account():
    return send_signed_request("GET", "/api/v3/account")

# returns a dictionary of all assets with an available non-zero balance
def get_balances():
    return {b['asset']:float(b['free']) for b in get_account()['balances'] if float(b['free'])}

# returns a dictionary of all assets with an available balance, including zeros
def get_balances2():
    return {b['asset']:float(b['free']) for b in get_account()['balances']}

# returns the available balance of asset
def get_balance(asset):
    return get_balances2()[asset]

# checks that the api is functional
def check_api():
    acc = get_account()
    try:
        acc['balances']
        return
    except KeyError:
        raise FailureError("API error: {}".format(acc))
    return

# pings the server
def ping():
    return send_public_request('/api/v3/ping')

def get_trades(symbol):
    return send_signed_request("GET", "/api/v3/myTrades", {"symbol":symbol})

#========================================================================================================================================================================================
# CUSTOM EXCEPTIONS

class NoMethodError(Exception):
    def __init__(self, message="Method not yet implemented"):
        self.message = message
        super().__init__(self.message)
        return

class FailureError(Exception):
    def __init__(self, message="Execution failed"):
        self.message = message
        super().__init__(self.message)
        return

class ParameterError(Exception):
    def __init__(self, message="At least one parameter is bad"):
        self.message = message
        super().__init__(self.message)
        return

#========================================================================================================================================================================================
# GIT

# returns the local git repo
def git_repo():
    return Repo(repo_path)

# commits files with message
def git_commit(files, message):
    repo = git_repo()
    repo.index.add(files)
    return repo.index.commit(message)

# returns a list of modified files
def git_mod():
    return [item.a_path for item in git_repo().index.diff(None)]

# returns a list of new files
def git_new():
    return git_repo().untracked_files

# returns all changed files
def git_changes():
    return git_mod() + git_new()

#========================================================================================================================================================================================
# MISC

# use in a 'with' wrapper to block prints from the inner code e.g. | with NoPrint(): code |
class NoPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Timer:
    units = {'s':1000, 'm':6000}
    def __init__(self, n, unit='s'):
        self.n = n
        self.divider = self.units[unit]
    def run(self, func, **kwargs):
        start = now()
        for i in range(self.n):
            result = func(kwargs)
        return (now()-start)/self.divider, result

# used to print a continuous progress tracker as percentages for long running functions
# 'interval' is the minimum % change required for a progress update
# requires percentage = 0 initialised in the outer function
def progress_tracker(percentage, item, sequence, interval=5):
    done = 100*(sequence.index(item) + 1)/len(sequence)                                      # compute how far along 'item' is in 'sequence' as a percentage
    if done-interval >= percentage:                                                    # if 'item' is over one 'interval' % further along than the currently stored percentage:
        new_percentage = int(done-(done%interval))                                       # update current % to reflect the new progress, rounding to the nearest 'interval' %
        print(c.C+"    PROCESSING... {}%".format(new_percentage)+c.X)       # print the progress update message
        return new_percentage                                                            # return the new percentage to be stored in the outer function for the next call to progress tracker
    return percentage                                                                  # if the % progress has not surpassed the next interval, return the percentage unchanged and print nothing

# same as progress_tracker but uses integers instead of item and sequence
def progress_tracker2(percentage, n, N, interval=5):
    done = 100*(n + 1)/N
    if done-interval >= percentage:
        new_percentage = int(done-(done%interval))
        print(c.C+"    PROCESSING... {}%".format(new_percentage)+c.X)
        return new_percentage
    return percentage

# returns a callable util function given the function's name
def get_util_function(func_name):
    current_module = __import__(__name__)
    return getattr(current_module, str(func_name))

# fills the gaps in time for the given times,prices with intermediate points every 'width' seconds
# used to augment trade.Asset's historical price initialisation to prevent big best fit jumps during early autotrading
def fill_prices(times, prices, width=0.5, noise=0, seed=None):
    randomseed(seed)                                                     # set the seed for the random numbers
    new_times = []                                                       # initialise list to contain filled out times
    new_prices = []                                                      # initialise list to contain filled out prices
    for i in range(len(times)-1):                                        # iterate through adjacent pairs of times/prices
        time_delta = times[i+1]-times[i]                                 # compute the time difference between adjacent pairs
        price_delta = prices[i+1]-prices[i]                              # compute the price difference between adjacent pairs
        slices = int((time_delta/1000)/width)                            # compute the number of intermediate values between each pair
        for j in range(slices):                                          # iterate through number of intermediate values
            new_times.append(int(times[i]+(j*time_delta/slices)))        # append the jth intermediate time value to new_times
            noise_ = price_delta*uniform(-1,1)*noise
            new_prices.append(prices[i]+(j*price_delta/slices)+noise_)   # append the jth intermediate price value to new_prices + some random noise if any
    new_times.append(times[-1])                                          # append the final time value as it is not reached by the final loop
    new_prices.append(prices[-1])                                        # append the final price value as it is not reached by the final loop
    return new_times, new_prices

# returns true if n is either nan or inf i.e. not a valid number in most situations
def isnaninf(n):
    return (isnan(n) or isinf(n))

# I got tired of writing out dictionaries to test all the time
def D():
    return {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':10}

# creates adds whitespace to the end of 'string' to make it at least 'max_len' characters long
# fill extra space with 'filler'
def fill_string(string, max_len, filler=' '):
    string = str(string)
    filler = str(filler)
    assert len(filler) == 1, "filler must be one character long: filler = '{}'".format(filler)
    return string + filler*(max_len-len(string))

def red_or_green(score, reference):
    return c.G if score >= reference else c.R

# returns a string representing 'integer' with 'leading' number of leading zeros
def parse_integer(integer, leading=1):
    I = str(integer)
    L = len(I)
    zeros = '0'*(leading+1 - L)
    return zeros+I

# ensure that 'a' can be iterated over, accounting for it being a single element
# if 'a' is a string then an iterable with that string is returned rather than the string itself (which is iterable over its characters)
def make_iterable(a):
    if isinstance(a, str):
        return iter([a])
    try:
        return iter(a)
    except TypeError:
        return iter([a])
    return

# returns the current ipv4 address
def get_ipv4():
    return requests.get('https://checkip.amazonaws.com').text.strip()

# returns the variable fields for the given object as a dictionary
def get_fields(obj):
    return {attr:getattr(obj, attr) for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")}

# assigns the given fields dictionary to obj
# can pass a whitelist of variiable names to set only those
def set_fields(obj, fields, whitelist=None):
    whitelist = whitelist if whitelist else get_fields(obj).keys()
    for var, val in fields.items():
        if var in whitelist:
            setattr(obj, var, val)
    return

# returns a dictionary containing all valid spot assets and their approximate value in USD
def get_usd_price_reference():
    valid_symbols = get_open_spot_symbols()                                        # get all open spot symbols
    valid_assets = get_open_spot_assets()                                          # get all open spot assets
    symbol_prices = get_all_current_prices()                                       # get current prices for all symbols
    usd_assets = ['BUSD', 'USDT', 'USDC', 'USDS', 'USDP', 'TUSD']                  # list of USD stable assets
    unref = set(valid_assets) - set(usd_assets)                                    # initialise set of currently unreferenced assets
    usd_price_ref = {asset:1 for asset in set(usd_assets)&set(valid_assets)}       # initialise dictionary of asset:value pairs with USD stable assets

    # recursive function to iterate through all unreferenced assets using data from referenced assets to determine the price values
    def recursive_upr(unrefed, ref_dict):
        if not unrefed:                                                                      # if there are no unreferenced assets left:
            return dict(zip(ref_dict, map(lambda i: round(ref_dict[i],8), ref_dict)))          # return the full reference dictionary with values rounded to 2dp
        new_unrefed = unrefed.copy()                                                           # store mutable copy of unreferenced assets
        for asset in unrefed:                                                      # iterate through unreferenced assets
            asset_type = get_asset_type(asset)                                       # store the asset type, either base, quote, or hybrid
            if asset_type in ['base', 'hybrid']:                                     # if the asset is a base or hybrid asset:
                asset_symbols = get_base_symbols(asset, valid_symbols)                 # compute the valid symbols for the base asset
                for symbol in asset_symbols:                                           # iterate through its symbols
                    try:                                                               # try to reference the base asset price using the known quote asset reference
                        price = symbol_prices[symbol]                                    # store the symbol price
                        quote = get_quote_asset(symbol)                                  # get the quote asset for the symbol
                        ref_dict[asset] = price*ref_dict[quote]                          # set the price reference
                        new_unrefed.remove(asset)                                        # remove the base asset from the unreferenced set
                        break                                                            # break iteration for this asset
                    except KeyError:                                                   # except a key error if this symbol's quote asset is not yet referenced
                        continue                                                         # move on to the next symbol and try again
            if asset_type in ['quote', 'hybrid']:                                    # if the asset is a quote or hybrid asset:
                asset_symbols = get_quote_symbols(asset, valid_symbols)                # compute the valid symbols for the quote asset
                for symbol in asset_symbols:                                           # iterate through its symbols
                    try:                                                               # try to reference the quote asset price using the known base asset reference
                        price = symbol_prices[symbol]                                    # store the symbol price
                        base = get_base_asset(symbol)                                    # get the base asset for the symbol
                        ref_dict[asset] = ref_dict[base]/price                           # set the price reference
                        new_unrefed.remove(asset)                                        # remove the quote asset from the unreferenced set
                        break                                                            # break iteration for this asset
                    except KeyError:                                                   # except a key error if this symbol's base asset is not yet referenced
                        continue                                                         # move on to the next symbol and try again
        return recursive_upr(new_unrefed, ref_dict)                                  # run the next recursion using the updated unreferenced assets set and price reference dictionary

    return recursive_upr(unref, usd_price_ref)         # execute the recursive loop to reference all assets

# returns a dictionary containing
def get_usd_price_book():
    valid_symbols = get_open_spot_symbols()                                        # get all open spot symbols
    valid_assets = get_open_spot_assets()                                          # get all open spot assets
    symbol_prices = get_all_current_prices()                                       # get current prices for all symbols
    base_assets = get_base_assets()                                                # get all base assets
    usd_assets = ['BUSD', 'USDT', 'USDC', 'USDS', 'USDP', 'TUSD']                  # list of USD stable assets
    unref = set([asset for asset in (set(valid_assets) - set(usd_assets)) if asset in base_assets])     # initialise set of currently unreferenced base assets
    unref.remove('DAI')
    usd_price_ref = {asset:1 for asset in usd_assets}                              # initialise dictionary of asset:value pairs with USD stable assets
    asset_price_book = {asset:{} for asset in base_assets}
    # recursive function to iterate through all unreferenced assets using data from referenced assets to determine the price values
    def recursive_upr(unrefed, ref_dict):
        if not unrefed:                                                                      # if there are no unreferenced assets left:
            return asset_price_book                                                          # return the symbol price book for every asset
        new_unrefed = unrefed.copy()                                                           # store mutable copy of unreferenced assets
        for asset in unrefed:                                                      # iterate through unreferenced assets
            asset_symbols = get_base_symbols(asset, valid_symbols)
            for symbol in asset_symbols:                                           # iterate through its symbols
                try:                                                               # try to reference the base asset price using the known quote asset reference
                    if asset in new_unrefed:
                        price = symbol_prices[symbol]                                    # store the symbol price
                        quote = get_quote_asset(symbol)                                  # get the quote asset for the symbol
                        ref_dict[asset] = price*ref_dict[quote]                          # set the price reference
                        new_unrefed.remove(asset)                                        # remove the base asset from the unreferenced set
                        asset_price_book[asset][symbol] = price*ref_dict[quote]
                    else:
                        price = symbol_prices[symbol]                                    # store the symbol price
                        quote = get_quote_asset(symbol)                                  # get the quote asset for the symbol
                        asset_price_book[asset][symbol] = price*ref_dict[quote]
                except KeyError:                                                   # except a key error if this symbol's quote asset is not yet referenced
                    continue                                                         # move on to the next symbol and try again
        return recursive_upr(new_unrefed, ref_dict)                                  # run the next recursion using the updated unreferenced assets set and price reference dictionary

    return recursive_upr(unref, usd_price_ref)         # execute the recursive loop to reference all assets

#========================================================================================================================================================================================
# GRAD SCORING

def compute_grad_score(htime=3, atime=3, iterations=1, end=None, vol_hours=1, timeprices=None):
    print("COMPUTING HISTORICAL TIMEPRICES...")
    iterations = binary_max(iterations, 1)
    hours = htime+atime+iterations-1
    tps = timeprices if timeprices else get_hos_usd_timeprices(start=hours_ago(hours, end), end=end, interval='1m')
    for symbol in bad_symbols:
        if symbol in tps:
            del tps[symbol]     # BTTC is not ideal for trading
    vols, vol_tps = grad_vols(tps, vol_hours)
    print("COMPUTING GRAD SCORE...")
    symbols = list(vols)[:100]
    Tps = {symbol:tps[symbol] for symbol in symbols}
    grads, _ = grad_change(htime, atime, iterations, symbols=symbols, tps=Tps)
    p_score, n_score, av_score = print_grad_scores(grads)
    plot_vols(vols, vol_tps, subplots=9, show_legend=False)
    return {"p_score":p_score, "n_score":n_score, "tps":iter(tps.items()), "vols":vols}

def grad_change(htime=3, atime=3, iterations=1, end=None, symbols=None, tps=None):
    hours = htime+atime
    hist_frac = htime/(atime+htime)
    iterations = binary_max(iterations, 1)
    symbols = symbols if symbols else get_hos_usd_symbols()
    tps = tps if tps else get_historical_prices_multi(symbols, interval='5m', start=hours_ago(hours+iterations-1, end))
    grads = {'pp':0, 'nn':0, 'pn':0, 'np':0}
    tps_items = tps.items()
    for i in range(iterations):
        for symbol, timeprices in tps_items:
            t = list(timeprices)
            index = bisect_left(t, hours_ago(hours+i,t[-1]))
            times, prices = zip(*list(timeprices.items())[index:])
            g1, g2 = compute_grads(times, prices, hist_frac)
            grads[grad_trend(g1, g2)] += 1
    return grads, iter(tps_items)

def compute_grads(times, prices, frac):
    t1 = times[:int(len(times)*frac)]
    T = len(t1)
    p1 = prices[:T]
    bf1 = best_fit(t1,p1)
    t2 = times[T:]
    p2 = prices[T:]
    bf2 = best_fit(t2,p2)
    g1 = bf1[-1] - bf1[0]
    g2 = bf2[-1] - bf2[0]
    return g1, g2

def grad_trend(g1, g2):
    if g1>0:
        return 'pp' if g2>0 else 'pn'
    else:
        return 'np' if g2>0 else 'nn'
    return

def print_grad_scores(grads):
    p_score, n_score, av_score = grad_score(grads)
    print(c.M, end='')
    print("PP PROBABILITY:   {}".format(p_score))
    print("NP PROBABILITY:   {}".format(n_score))
    print("AV P PROBABILITY: {}".format(av_score))
    print(c.X)
    return p_score, n_score, av_score

def grad_score(grads):
    p_score = round((grads['pp'])/(grads['pp'] + grads['pn']), 3)
    n_score = round((grads['np'])/(grads['np'] + grads['nn']), 3)
    av_score = round((p_score+n_score)/2, 3)
    return p_score, n_score, av_score

def grad_vols(tps, hours):
    Timeprices = {}
    for symbol, timeprices in tps.items():
        t = list(timeprices)
        index = bisect_left(t, hours_ago(hours,t[-1]))
        Timeprices[symbol] = dict(list(timeprices.items())[index:])
    vols = dsort_descending({symbol:volatility_percentage_clipped(list(Timeprices[symbol].keys()), list(Timeprices[symbol].values())) for symbol in Timeprices})
    return vols, Timeprices

def get_vols(hours, interval='1m', start=None, end=None):
    print("COMPUTING VOLATILITIES...")
    end = end if end else now()
    start = hours_ago(hours, end)
    symbols = get_hos_usd_symbols()
    for symbol in bad_symbols:
        if symbol in symbols:
            symbols.remove(symbol)
    Timeprices = get_historical_prices_multi(symbols, interval, start, end)
    vols = dsort_descending({symbol:volatility_percentage_clipped(list(Timeprices[symbol].keys()), list(Timeprices[symbol].values())) for symbol in Timeprices})
    return vols
