import config
import util
from util import Colours as c
from trade import Trader, Asset, BtcTrader, Animator, Historian, weave, unit_conversion
from wallet import Test_Wallet
from bisect import bisect_left, bisect_right
from numpy import array, mean, concatenate, sqrt as npsqrt, minimum as curve_min
from numpy.random import seed as randomseed, shuffle as npshuffle
from math import sqrt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
from os import listdir, mkdir
from scipy.optimize import curve_fit
from shutil import copy2 as shutil_copy2
from multiprocessing.pool import Pool

quick_gradients = False

div = c.X+" | "

# class to contain global variables that may be changed across simulations
class Globs:
    quote_amount = 1000
    # general
    min_percentage_sell_threshold = config.min_percentage_sell_threshold
    global_fit_turn_multiplier = config.global_fit_turn_multiplier
    local_fit_buy_turn_multiplier = config.local_fit_buy_turn_multiplier
    local_fit_sell_turn_multiplier = config.local_fit_sell_turn_multiplier
    mid_fit_buy_turn_multiplier = config.mid_fit_buy_turn_multiplier
    mid_fit_sell_turn_multiplier = config.mid_fit_sell_turn_multiplier
    historical_hours = config.historical_hours
    spline_update_interval = config.spline_update_interval
    spline_window = config.spline_window
    p_amplitude_period = config.p_amplitude_period
    universal_average_period = config.universal_average_period
    pp_scale = config.pp_scale
    pp_up = config.pp_up
    pp_cg_scale = config.pp_cg_scale
    pp_gg_scale = config.pp_gg_scale
    cg_scale = config.cg_scale
    cg_pamp_scale = config.cg_pamp_scale
    cg_down = config.cg_down
    sell_amplitude_multiplier = config.sell_amplitude_multiplier
    stop_loss_amplitude_multiplier = config.stop_loss_amplitude_multiplier
    stop_loss_wait_multiplier = config.stop_loss_wait_multiplier
    # btc
    btc_percentage_sell_threshold = config.btc_percentage_sell_threshold
    btc_global_fit_period_minutes = config.btc_global_fit_period_minutes
    btc_local_fit_buy_period_minutes = config.btc_local_fit_buy_period_minutes
    btc_local_fit_sell_period_minutes = config.btc_local_fit_sell_period_minutes
    btc_p_amplitude_period_minutes = config.btc_p_amplitude_period_minutes
    btc_stop_loss_wait_period_minutes = config.btc_stop_loss_wait_period_minutes
    btc_historical_hours = config.btc_historical_hours
    btc_buy_limit_scale = config.btc_buy_limit_scale
    btc_stop_loss_amplitude_multiplier = config.btc_stop_loss_amplitude_multiplier
    # other
    seeded = False
    seed = None
    refund_last_buy = False
    # value checking
    assert spline_update_interval < spline_window//2

class SimGlobs:
    def __init__(self, globs=None):
        V = util.get_fields(Globs)
        globs = {} if not globs else globs
        for key, val in V.items():
            value = globs[key] if key in globs else val
            setattr(self, key, value)
        return

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================

class Simulator:

    def __init__(self, filename, globs=None):
        self.globs = SimGlobs(globs)
        self.symbol = filename_symbol(filename)
        self.filename = filename
        self.asset = SimAsset(self.globs, self.symbol, filename=filename, seeded=self.globs.seeded)
        self.trader = SimTrader(self.symbol, self.asset)
        self.simulated = False
        self.set_score()
        return

    def set_score(self):
        self.score = self.trader.wallet.get_balance(self.asset.quote)
        return

    def run(self, mute=False, timed=False):
        start = util.now()
        if mute:
            with util.NoPrint():
                self.run_()
        else:
            self.run_()
        self.run_time = round((util.now()-start)/1000,3)
        if timed:
            print("TIME: {} seconds".format(self.run_time))
        return

    def run_(self):
        print(c.C+"~ RUNNING {} AUTOTRADE SIMULATION SHY ~".format(self.symbol)+c.X)
        # run loop
        for time, price in gen_file(self.filename):
            self.asset.compute_asset_data(time, price)                                 # compute asset data
            self.trader.trade_decision_functions[self.asset.position](self.asset, time, price)            # make trade decision
        # finishing
        if not self.asset.position == 'buying':
            self.refund_last_buy() if self.globs.refund_last_buy else self.sell_last_buy()
        self.simulated = True
        self.set_score()
        print(c.C+"~ SIMULATION COMPLETE ~"+c.X)
        return

    def refund_last_buy(self):
        self.trader.wallet.market_sell_all(self.symbol, self.asset.buy_price)
        return

    def sell_last_buy(self):
        self.trader.wallet.market_sell_all(self.symbol, self.asset.prices[-1])
        return

    def plot_simulation(self, splines=False, thresholds=True, legend=True, show=True, save=False, figname=''):
        assert self.simulated, "Must first run a simulation before plotting"
        # figure
        fig, axes = util.get_subplot_arrangement(1, figsize=config.sim_plot_figsize)
        ax = axes[0]
        # timeprices
        times, prices = zip(*gen_file(self.filename))
        t0 = times[0]
        L = len(times)
        times, time_label = util.reasonable_times(times, zero_index=0)
        unit_divider = unit_conversion(time_label, 'seconds')
        base, quote = util.split_symbol(self.symbol)
        # main line and axes admin
        ax.plot(times, prices, color='tab:blue')
        ax.set_title('{}-{} Trade Simulation'.format(base, quote))
        ax.set_xlabel("Time ({})".format(time_label))
        ax.set_ylabel("Price ({}/{})".format(quote, base))
        ax.set_xlim(0, times[-1])
        axylim1, axylim2 = util.get_y_axis_limits(prices)
        ax.set_ylim(axylim1, axylim2)
        # percentage axis
        percentage_prices = util.percentage_array(array(prices), prices[0])
        p_mask = util.mask_all(percentage_prices)
        ax2 = ax.twinx()
        ax2.plot(times, p_mask)
        ax2.set_ylabel("% Change")
        ax2ylim1, ax2ylim2 = util.get_y_axis_limits(percentage_prices)
        ax2.set_ylim(ax2ylim1, ax2ylim2)
        ax2.grid(axis='y')
        ax.set_zorder(ax2.get_zorder()+1)
        ax.set_frame_on(False)
        # global threshold line
        buy_colors = self.asset.global_gradients[-L:]
        threshold_points = self.asset.global_thresholds[-L:]
        c = ax.scatter(times, threshold_points, c=buy_colors, s=2, cmap='inferno', vmin=-0.0002, vmax=0.0002)
        cbar = plt.colorbar(c, ax=ax, pad=0.06, aspect=40)
        cbar.ax.set_title('Global Gradient')
        # spline curve
        if splines:
            spline_fit = self.asset.spline_fit[-L:]
            if len(spline_fit) < L:
                spline_fit = concatenate( ([spline_fit[0]]*(L-len(spline_fit)), spline_fit) )
            ax.plot(times, spline_fit, color='tab:orange')
        # universal average line
        if any(self.asset.universal_averages):
            ua = self.asset.universal_averages[-L:]
            ax.plot(times, ua, color='tab:brown', linestyle='dotted')
        # buy limit lines
        buy_limits = self.asset.buy_limits[-L:]
        fill_curve = curve_min(prices, buy_limits)
        ax.fill_between(times, fill_curve, axylim1, color='tab:blue', alpha=0.25)
        ax.plot(times, self.asset.cg_limits, color='grey', linewidth=1)
        ax.plot(times, self.asset.pp_limits, color='grey', linewidth=1)
        # sell threshold lines
        if thresholds:
            ax.plot(times, self.asset.min_sell_thresholds[-L:], color='tab:olive', linestyle='dotted')
            ax.plot(times, self.asset.amp_sell_thresholds[-L:], color='olivedrab', linestyle='dotted')
            ax.plot(times, self.asset.stop_loss_thresholds[-L:], color='tab:red', linestyle='dotted')
        # trade markers and lines
        if self.asset.n_trades:
            trade_prices, trade_colors = zip(*self.asset.anim_trades)
            trade_times = (self.asset.trade_times - t0)/unit_divider/1000
            xtrade_lines = [trade_times[0]] + weave(trade_times[1:], trade_times[1:]) + [times[-1]]
            ytrade_lines = weave(trade_prices, trade_prices)
            trade_lines = list(zip(xtrade_lines, ytrade_lines))
            segments = [trade_lines[:2]]
            for i in range(1,self.asset.n_trades):
                segments.append(trade_lines[(2*i)-1:(2*i)+2])
            ax.add_collection(LineCollection(segments, colors=trade_colors))
            ax.scatter(trade_times, trade_prices, c=trade_colors, s=30, zorder=10)
        # custom legendary
        legend_lines = [Line2D([], [], marker='o', linestyle='None', color='m'),
                        Line2D([], [], marker='o', linestyle='None', color='g'),
                        Line2D([], [], marker='o', linestyle='None', color='r'),
                        Line2D([], [], color='tab:blue'),
                        Line2D([], [], linestyle='dashed', color='k')]
        legend_names = ['Buy', 'Sell', 'Stop-loss', 'Price', 'Buy limit']
        if splines:
            legend_lines.append(Line2D([], [], color='tab:orange'))
            legend_names.append('Smoothed')
        if any(self.asset.universal_averages):
            legend_lines.append(Line2D([], [], color='tab:brown', linestyle='dotted'))
            legend_names.append('Average')
        if legend:
            ax.legend(legend_lines, legend_names)
        # saving figure to file
        if save:
            assert figname, "must pass a name for the figure to be saved"
            filepath = "{}/{}/{}.png".format(config.fig_save_dir,self.filename[:-4],figname)
            plt.savefig(fname=filepath, dpi=120)
        if show:
            plt.show()
        else:
            plt.close()  # close used when not showing the figure to save memory and avoid warnings (useful when saving many figures consecutively)
        return

    def proto(self, mute=True):
        if mute:
            with util.NoPrint():
                for time, price in gen_file(self.filename):
                    self.asset.compute_asset_data(time, price)
        else:
            for time, price in gen_file(self.filename):
                self.asset.compute_asset_data(time, price)
        self.simulated = True
        return

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================

class SimTrader(Trader):

    # override to avoid validation (api rate limit will halt after too many simulations)
    def __init__(self, symbol, asset):
        asset.trader = self
        self.wallet = Test_Wallet(init_assets={asset.quote:asset.trade_balance})
        self.assets = {symbol:asset}
        self.total_initial_usd = asset.globs.quote_amount
        self.symbol = symbol
        self.base, self.quote = util.split_symbol(symbol)
        self.trade_decision_functions = {'buying':self.buy_decision, 'selling':self.sell_decision, 'tanking':self.tank_decision, 'final':self.final_decision}
        return

    def validate_ip(self):
        return

    def log(self, message, color=None, asset=None, level=None):
        return

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================

class SimAsset(Asset):

    def __init__(self, globs, symbol, trade_balance=None, start_time=None, seeded=True, seed=None, filename=None, quick_historical=True):
        self.trader = None
        self.globs = globs
        self.symbol = symbol
        self.base, self.quote = util.split_symbol(symbol)
        self.trade_balance = trade_balance if trade_balance else self.globs.quote_amount
        assert start_time or filename, "must pass either start_time or filename"
        start_time = start_time if start_time else next(gen_file(filename))[0]
        self.start_time = start_time
        self.seed = int(start_time/sqrt(int(''.join([str(ord(i)) for i in symbol]))))
        randomseed(self.seed) if seeded else randomseed(seed) if seed else randomseed()
        self.set_variables()
        self.historian = SimHistorian(filename, start_time) if quick_historical else Historian(symbol, start_time)
        self.initialise_historical_data()
        self.initialise_spline_fit()
        self.initialise_global_gradients()
        self.initialise_universal_averages()
        self.set_attributes()
        if quick_gradients:
            self.local_gradients = [float(g) for g in self.get_data('websocket_data/attributes/'+filename.split('files/')[1][:-4]+'/local_gradients.txt')]
            self.mid_gradients = [float(g) for g in self.get_data('websocket_data/attributes/'+filename.split('files/')[1][:-4]+'/mid_gradients.txt')]
            self.compute_local_fit = self.compute_local_fit_quick
            self.compute_mid_fit = self.compute_mid_fit_quick
        else:
            self.compute_local_fit = self.compute_local_fit_normal
            self.compute_mid_fit = self.compute_mid_fit_normal
        self.pp_limits = []
        self.cg_limits = []
        return

    def get_data(self, filename):
        with open(filename, 'r') as f:
            data = f.read().split()
        return data

    # override to use Globs
    def set_variables(self):
        self.min_percentage_sell_threshold = self.globs.min_percentage_sell_threshold
        self.global_fit_turn_multiplier = self.globs.global_fit_turn_multiplier
        self.local_fit_buy_turn_multiplier = self.globs.local_fit_buy_turn_multiplier
        self.local_fit_sell_turn_multiplier = self.globs.local_fit_sell_turn_multiplier
        self.mid_fit_buy_turn_multiplier = self.globs.mid_fit_buy_turn_multiplier
        self.mid_fit_sell_turn_multiplier = self.globs.mid_fit_sell_turn_multiplier
        self.historical_hours = self.globs.historical_hours
        self.spline_update_interval = self.globs.spline_update_interval
        self.spline_window = self.globs.spline_window
        self.p_amplitude_period = self.globs.p_amplitude_period
        self.universal_average_period = self.globs.universal_average_period
        self.pp_scale = self.globs.pp_scale
        self.pp_up = self.globs.pp_up
        self.pp_cg_scale = self.globs.pp_cg_scale
        self.pp_gg_scale = self.globs.pp_gg_scale
        self.cg_scale = self.globs.cg_scale
        self.cg_pamp_scale = self.globs.cg_pamp_scale
        self.cg_down = self.globs.cg_down
        self.sell_amplitude_multiplier = self.globs.sell_amplitude_multiplier
        self.stop_loss_amplitude_multiplier = self.globs.stop_loss_amplitude_multiplier
        self.stop_loss_wait_multiplier = self.globs.stop_loss_wait_multiplier
        return

    # override to prevent log.log creation in main directory
    def logger_init(self):
        return

    # override to append timestamp 'time' to self.trade_times instead of self.time_elapsed
    # needed atm for plot_simulation
    def new_trade(self, side, time, price, color):
        self.n_trades += 1
        self.trades.append({'side':side, 'time':time, 'price':price})
        self.anim_trades.append((price, color))
        self.trade_times = util.append(self.trade_times, time, self.n_trades)
        return

    # override to ignore after_trade functionality
    def after_trade(self, base_balance):
        return

    def compute_local_fit_quick(self):
        self.local_gradient = self.local_gradients[self.n-1]
        return

    def compute_mid_fit_quick(self):
        self.mid_gradient = self.mid_gradients[self.n-1]
        return

    def compute_local_fit_normal(self):
        super().compute_local_fit()
        return

    # override to avoid computing the additional best fit needed for animation
    def compute_mid_fit_normal(self):
        mid_fit_turn_multiplier = self.mid_fit_buy_turn_multiplier if self.position == 'buying' else self.mid_fit_sell_turn_multiplier
        mid_fit_period = self.av_time_between_turns*mid_fit_turn_multiplier         # mid fit period
        mf_index = bisect_left(self.times,self.time_elapsed-mid_fit_period)         # mid fit start index
        xmbf = self.times[mf_index:]                                                # fit time values
        nmbf = util.best_fit(xmbf, self.norm_prices[mf_index:])                     # fit to normalised data
        self.mid_gradient = (nmbf[-1]-nmbf[0])#/self.mid_fit_period*1000            # normalised gradient - # divider sets rough min/max gradient bounds to -1/1
        return

    # def compute_buy_limit(self):
    #     super().compute_buy_limit()
    #     self.pp_limits.append(self.pp_limit)
    #     self.cg_limits.append(self.cg_limit)
    #     return

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================

class BtcSimulator(BtcTrader):

    def __init__(self, filename, globs=None):
        self.globs = SimGlobs(globs)
        self.symbol = filename_symbol(filename)
        self.base, self.quote = util.split_symbol(self.symbol)
        self.wallet = Test_Wallet(init_assets={self.quote:self.globs.quote_amount})
        self.filename = filename
        self.simulated = False
        self.set_score()
        self.init_data()
        return

    def set_score(self):
        self.score = self.wallet.get_balance(self.quote)
        return

    def run(self, timed=False):
        start = util.now()
        with util.NoPrint():
            self.run_()
        self.run_time = round((util.now()-start)/1000,3)
        if timed:
            print("TIME: {} seconds".format(self.run_time))
        return

    def run_(self):
        for time, price in gen_file(self.filename):
            self.compute_asset_data(time, price)
            self.trade_decision_functions[self.position](time, price)
        # finishing
        if not self.position == 'buying':
            self.refund_last_buy() if self.globs.refund_last_buy else self.sell_last_buy()
        self.simulated = True
        self.set_score()
        return

    def refund_last_buy(self):
        self.wallet.market_sell_all(self.symbol, self.buy_price)
        return

    def sell_last_buy(self):
        self.wallet.market_sell_all(self.symbol, self.price)
        return

    def proto(self, mute=True):
        if mute:
            with util.NoPrint():
                for time, price in gen_file(self.filename):
                    self.compute_asset_data(time, price)
        else:
            for time, price in gen_file(self.filename):
                self.compute_asset_data(time, price)
        self.simulated = True
        return

    #========================================================================================================
    # INITIALISATION

    def init_data(self):
        self.trade_decision_functions = {'buying':self.buy_decision, 'selling':self.sell_decision, 'tanking':self.tank_decision}
        self.constants_init()
        self.historian = SimHistorian(self.filename, self.start_time)
        self.historical_data_init()
        self.quote_trade_balance = self.get_usd()
        self.variables_init()
        return

    def constants_init(self):
        self.start_time = next(gen_file(self.filename))[0]
        self.base_precision_factor = 10**-8
        self.total_initial_btc = self.get_btc()
        self.total_initial_usd = self.get_usd()
        self.total_initial_usd_value = self.compute_initial_total_usd_value()
        self.nan = float('nan')
        # from config
        self.percentage_sell_threshold = self.globs.btc_percentage_sell_threshold
        self.global_fit_period_minutes = self.globs.btc_global_fit_period_minutes
        self.global_fit_period = self.globs.btc_global_fit_period_minutes*60000
        self.local_fit_buy_period = self.globs.btc_local_fit_buy_period_minutes*60000
        self.local_fit_sell_period = self.globs.btc_local_fit_sell_period_minutes*60000
        self.p_amplitude_period = self.globs.btc_p_amplitude_period_minutes*60000
        self.stop_loss_wait_period = self.globs.btc_stop_loss_wait_period_minutes*60000
        self.historical_hours = self.globs.btc_historical_hours
        self.buy_limit_scale = self.globs.btc_buy_limit_scale
        self.stop_loss_amplitude_multiplier = self.globs.btc_stop_loss_amplitude_multiplier
        # longest time
        self.longest_time = max([self.global_fit_period, self.p_amplitude_period])
        # TEMPORARY
        self.global_thresholds = []
        self.global_gradients = []
        self.buy_limits = []
        self.sell_thresholds = []
        self.stop_loss_thresholds = []
        self.anim_trades = []
        return

    #========================================================================================================
    # OVERRIDES

    def compute_initial_total_usd_value(self):
        return self.get_usd()

    def log(self, message, color='', level=''):
        return

    def new_trade(self, side, time, price, color):
        self.anim_trades.append((time, price, color))
        return

    def after_trade(self, price):
        return

    def compute_asset_data(self, time, price):
        self.compute_timeprice_data(time, price)
        self.compute_global_fit()
        self.compute_local_fit()
        self.compute_buy_limit()
        self.compute_sell_threshold()
        self.compute_stop_loss_threshold()
        self.compute_plot_elements()
        return

    def compute_plot_elements(self):
        self.global_thresholds.append(self.global_threshold)
        self.buy_limits.append(self.buy_limit)
        self.compute_sell_points()
        self.sell_thresholds.append(self.st)
        self.stop_loss_thresholds.append(self.slt)
        return

    #========================================================================================================
    # PLOT

    def plot_simulation(self):
        assert self.simulated, "Must first run a simulation before plotting"
        # figure
        fig, axes = util.get_subplot_arrangement(1, figsize=config.sim_plot_figsize)
        ax = axes[0]
        # timeprices
        times, prices = zip(*gen_file(self.filename))
        t0 = times[0]
        L = len(times)
        times, time_label = util.reasonable_times(times, zero_index=0)
        unit_divider = unit_conversion(time_label, 'seconds')
        base, quote = util.split_symbol(self.symbol)
        # main line and axes admin
        ax.plot(times, prices, color='tab:blue')
        ax.set_title('{}-{} Trade Simulation'.format(base, quote))
        ax.set_xlabel("Time ({})".format(time_label))
        ax.set_ylabel("Price ({}/{})".format(quote, base))
        ax.set_xlim(0, times[-1])
        axylim1, axylim2 = util.get_y_axis_limits(prices)
        ax.set_ylim(axylim1, axylim2)
        # percentage axis
        percentage_prices = util.percentage_array(array(prices), prices[0])
        p_mask = util.mask_all(percentage_prices)
        ax2 = ax.twinx()
        ax2.plot(times, p_mask)
        ax2.set_ylabel("% Change")
        ax2ylim1, ax2ylim2 = util.get_y_axis_limits(percentage_prices)
        ax2.set_ylim(ax2ylim1, ax2ylim2)
        ax2.grid(axis='y')
        ax.set_zorder(ax2.get_zorder()+1)
        ax.set_frame_on(False)
        # global threshold line
        ax.plot(times, self.global_thresholds[-L:], color='tab:orange', linewidth=1)
        # buy limit lines
        buy_limits = self.buy_limits[-L:]
        fill_curve = curve_min(prices, buy_limits)
        ax.fill_between(times, fill_curve, axylim1, color='tab:blue', alpha=0.25)
        ax.plot(times, buy_limits, color='pink', linewidth=1)
        # sell threshold lines
        ax.plot(times, self.sell_thresholds[-L:], color='tab:olive', linestyle='dotted')
        ax.plot(times, self.stop_loss_thresholds[-L:], color='tab:red', linestyle='dotted')
        # trade markers and lines
        n_trades = len(self.anim_trades)
        if n_trades > 0:
            trade_times, trade_prices, trade_colors = zip(*self.anim_trades)
            trade_times = (array(trade_times) - t0)/unit_divider/1000
            xtrade_lines = [trade_times[0]] + weave(trade_times[1:], trade_times[1:]) + [times[-1]]
            ytrade_lines = weave(trade_prices, trade_prices)
            trade_lines = list(zip(xtrade_lines, ytrade_lines))
            segments = [trade_lines[:2]]
            for i in range(1,n_trades):
                segments.append(trade_lines[(2*i)-1:(2*i)+2])
            ax.add_collection(LineCollection(segments, colors=trade_colors))
            ax.scatter(trade_times, trade_prices, c=trade_colors, s=30, zorder=10)
        plt.show()
        return

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================

class SimAnimator(Animator):

    def __init__(self, asset, filename, start_time, session_folder):
        times, _ = zip(*gen_file(filename))
        self.n_frames = len(times)
        T = (times[-1] - times[0])/1000
        self.fps = int(180*self.n_frames/T)    # makes animation 1 minute long per 3 hours of data
        self.session_folder = session_folder
        self.p = 0    # progress tracker
        super().__init__(asset, start_time)
        if config.show_animator_legend:
            self.legend.remove()
        return

    def animate(self):
        print("SAVING ANIMATION... this may take a few minutes")
        self.anim = FuncAnimation(self.fig, self.update, interval=0.1, frames=self.n_frames, repeat=False)
        self.anim.save(self.session_folder+"/anim.mp4", dpi=400, writer=FFMpegWriter(fps=self.fps))
        print("SAVED")
        plt.close()
        return

    def update(self, frame):
        self.update_main(self.asset, frame)
        self.update_mini(self.asset, frame)
        self.p = util.progress_tracker2(self.p, frame+1, self.n_frames)
        return

    def update_main(self, asset, frame):
        # timeprices
        view_index = bisect_left(asset.times_[frame], asset.time_elapsed_[frame]-config.main_view_period)                 # index of data visible in the animator view
        times = asset.times_[frame][view_index:]                                                             # times
        t0 = times[-1]
        times = (times - t0)/self.unit_divider_main                                                       # times scaled to correct units
        prices = asset.prices_[frame][view_index:]                                                           # prices
        axylim1, axylim2 = util.get_y_axis_limits(prices)                                            # compute y axis limits
        self.ax_main.set_ylim(axylim1, axylim2)                                                           # set y axis limits
        self.price_main.set_data(times, prices)                                                      # update true price line
        # percentage axis
        p_prices = asset.percentage_prices_[frame][view_index:]                                              # percentage prices
        p_mask = util.mask_all(p_prices)                                                             # mask points to make them invisible
        ax2ylim1, ax2ylim2 = util.get_y_axis_limits(p_prices)                                        # compute secondary y axis limits
        self.ax_main2.set_ylim(ax2ylim1, ax2ylim2)                                                        # set secondary y axis limits
        self.percentage_price_main.set_data(times, p_mask)                                           # update masked points (required to realise axis limits)
        # buy limit
        buy_limits = asset.buy_limits_[frame][view_index:]                                                   # buy limits
        fill_curve = curve_min(prices, buy_limits)
        self.ax_main.collections.remove(self.fill_main)                                                        # remove buy limit shading to prevent stacking
        self.fill_main = self.ax_main.fill_between(times, fill_curve, axylim1, color='grey', alpha=0.25)       # set new shaded area
        # other lines
        self.global_threshold_main.set_data(times, asset.global_thresholds_[frame][view_index:])            # update global threshold line
        self.universal_average_main.set_data(times, asset.universal_averages_[frame][view_index:])          # update universal average line
        self.min_sell_main.set_data(times, asset.min_sell_thresholds_[frame][view_index:])                  # update min sell threshold line
        self.amp_sell_main.set_data(times, asset.amp_sell_thresholds_[frame][view_index:])                  # update amp sell threshold line
        self.stop_loss_main.set_data(times, asset.stop_loss_thresholds_[frame][view_index:])                # update stop loss threshold line
        xmbf = (asset.mid_fitx_[frame] - t0)/self.unit_divider_main
        self.mid_fit_main.set_data([xmbf, 0], asset.mid_fit_[frame])
        # trades
        if asset.n_trades_[frame]:
            trade_prices, trade_colors = zip(*asset.anim_trades_[frame])
            trade_times = (asset.trade_times_[frame] - t0)/self.unit_divider_main
            xtrade_lines = [trade_times[0]] + weave(trade_times[1:], trade_times[1:]) + [0]
            ytrade_lines = weave(trade_prices, trade_prices)
            trade_lines = list(zip(xtrade_lines, ytrade_lines))
            segments = [trade_lines[:2]]
            for i in range(1,asset.n_trades_[frame]):
                segments.append(trade_lines[(2*i)-1:(2*i)+2])
            self.trade_lines_main.remove()
            self.trade_lines_main = self.ax_main.add_collection(LineCollection(segments, colors=trade_colors, linewidth=1))
            self.trade_markers_main.set_offsets(list(zip(trade_times, trade_prices)))
            self.trade_markers_main.set_color(trade_colors)
        return

    def update_mini(self, asset, frame):
        # timeprices
        view_index = bisect_left(asset.times_[frame], asset.time_elapsed_[frame]-config.mini_view_period)                 # index of data visible in the animator view
        times = asset.times_[frame][view_index:]                                                             # times
        t0 = times[-1]
        times = (times - t0)/self.unit_divider_mini                                                       # times scaled to correct units
        prices = asset.prices_[frame][view_index:]                                                           # prices
        axylim1, axylim2 = util.get_y_axis_limits(prices)                                            # compute y axis limits
        self.ax_mini.set_ylim(axylim1, axylim2)                                                           # set y axis limits
        self.price_mini.set_data(times, prices)                                                      # update true price line
        # percentage axis
        p_prices = asset.percentage_prices_[frame][view_index:]                                              # percentage prices
        p_mask = util.mask_all(p_prices)                                                             # mask points to make them invisible
        ax2ylim1, ax2ylim2 = util.get_y_axis_limits(p_prices)                                        # compute secondary y axis limits
        self.ax_mini2.set_ylim(ax2ylim1, ax2ylim2)                                                        # set secondary y axis limits
        self.percentage_price_mini.set_data(times, p_mask)                                           # update masked points (required to realise axis limits)
        # buy limit
        buy_limits = asset.buy_limits_[frame][view_index:]                                                   # buy limits
        fill_curve = curve_min(prices, buy_limits)
        self.ax_mini.collections.remove(self.fill_mini)                                                        # remove buy limit shading to prevent stacking
        self.fill_mini = self.ax_mini.fill_between(times, fill_curve, axylim1, color='grey', alpha=0.25)       # set new shaded area
        # other lines
        self.global_threshold_mini.set_data(times, asset.global_thresholds_[frame][view_index:])            # update global threshold line
        self.universal_average_mini.set_data(times, asset.universal_averages_[frame][view_index:])          # update universal average line
        self.min_sell_mini.set_data(times, asset.min_sell_thresholds_[frame][view_index:])                  # update min sell threshold line
        self.amp_sell_mini.set_data(times, asset.amp_sell_thresholds_[frame][view_index:])                  # update amp sell threshold line
        self.stop_loss_mini.set_data(times, asset.stop_loss_thresholds_[frame][view_index:])                # update stop loss threshold line
        xmbf = (asset.mid_fitx_[frame] - t0)/self.unit_divider_mini
        self.mid_fit_mini.set_data([xmbf, 0], asset.mid_fit_[frame])
        # trades
        if asset.n_trades_[frame]:
            trade_prices, trade_colors = zip(*asset.anim_trades_[frame])
            trade_times = (asset.trade_times_[frame] - t0)/self.unit_divider_mini
            xtrade_lines = [trade_times[0]] + weave(trade_times[1:], trade_times[1:]) + [0]
            ytrade_lines = weave(trade_prices, trade_prices)
            trade_lines = list(zip(xtrade_lines, ytrade_lines))
            segments = [trade_lines[:2]]
            for i in range(1,asset.n_trades_[frame]):
                segments.append(trade_lines[(2*i)-1:(2*i)+2])
            self.trade_lines_mini.remove()
            self.trade_lines_mini = self.ax_mini.add_collection(LineCollection(segments, colors=trade_colors, linewidth=1))
            self.trade_markers_mini.set_offsets(list(zip(trade_times, trade_prices)))
            self.trade_markers_mini.set_color(trade_colors)
        # main axis box
        self.box_bottom.set_ydata(axylim1)
        self.box_top.set_ydata(axylim2)
        self.box_left.set_ydata([axylim1, axylim2])
        return

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================

class Simulator2(Simulator):

    def __init__(self, filename, variables):
        self.symbol = filename_symbol(filename)
        self.filename = filename
        self.asset = SimAsset2(self.symbol, variables['trade_balance'], variables['start_time'], seeded=False, seed=variables['seed'])
        self.trader = SimTrader(self.symbol, self.asset, animate=False)
        return

    def run(self):
        for time, price in gen_file(self.filename):
            self.asset.compute_asset_data(time, price)
            self.trader.trade_decision_functions[self.asset.position](time, price)
        self.set_score()
        self.simulated = True
        return

    def run_for_animation(self):
        for time, price in gen_file(self.filename):
            self.asset.compute_asset_data(time, price)
            self.trader.trade_decision_functions[self.asset.position](time, price)
            self.asset.update_all()
        return

class SimAsset2(Asset):

    def __init__(self, symbol, trade_balance, start_time, seeded, seed):
        super().__init__(symbol, trade_balance, start_time, seeded=seeded, seed=seed)
        self.times_ = []
        self.time_elapsed_ = []
        self.prices_ = []
        self.percentage_prices_ = []
        self.buy_limits_ = []
        self.global_thresholds_ = []
        self.universal_averages_ = []
        self.min_sell_thresholds_ = []
        self.amp_sell_thresholds_ = []
        self.stop_loss_thresholds_ = []
        self.mid_fitx_ = []
        self.mid_fit_ = []
        self.n_trades_ = []
        self.anim_trades_ = []
        self.trade_times_ = []
        return

    def update_all(self):
        self.times_.append(self.times)
        self.time_elapsed_.append(self.time_elapsed)
        self.prices_.append(self.prices)
        self.percentage_prices_.append(self.percentage_prices)
        self.buy_limits_.append(self.buy_limits)
        self.global_thresholds_.append(self.global_thresholds)
        self.universal_averages_.append(self.universal_averages)
        self.min_sell_thresholds_.append(self.min_sell_thresholds)
        self.amp_sell_thresholds_.append(self.amp_sell_thresholds)
        self.stop_loss_thresholds_.append(self.stop_loss_thresholds)
        self.mid_fitx_.append(self.mid_fitx)
        self.mid_fit_.append(self.mid_fit)
        self.n_trades_.append(self.n_trades)
        self.anim_trades_.append(self.anim_trades)
        self.trade_times_.append(self.trade_times)
        return

def simulate_session(index, mute=True):
    folder = 'sessions/'+listdir('sessions')[index]
    variables = util.json_to_dict(folder+'/variables.json')
    filename = folder+'/'+listdir(folder)[0]
    sim = Simulator2(filename, variables)
    if mute:
        with util.NoPrint():
            sim.run()
    else:
        sim.run()
    return

def create_animation(index):
    folder = 'sessions/'+listdir('sessions')[index]
    variables = util.json_to_dict(folder+'/variables.json')
    filename = folder+'/'+listdir(folder)[0]
    sim = Simulator2(filename, variables)
    print("SIMULATING... ", end='')
    with util.NoPrint():
        sim.run_for_animation()
    print('DONE')
    animator = SimAnimator(sim.asset, filename, variables['start_time'], variables['session_folder'])
    animator.animate()
    return

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================

# returns the symbol from the given filename
# requires filename to be of the form '[path]_KLINE_[SYMBOL].txt'
def filename_symbol(filename):
    return filename.split('KLINE_')[-1][:-4]

# returns a generator for the timeprice data in the given file
# this is faster than explicit dictionary creation for loops & unpacking
# dict(gen_file(filename)) is faster than explicit dictionary creation
def gen_file(filename):
    with open(filename, 'r') as f:
        data = f.read().split()
    for row in data:
        t, p = row.split(';')
        yield (int(t), float(p))

# returns the file name string with the associated local 'file' name or at the associated 'index' in the _test_files directory
def test_file(folder='_test_files', file=None, index=None):
    d = 'websocket_data/{}/'.format(folder)
    if file:
        return d+file
    elif not index == None:
        return d+listdir(d)[index]
    else:
        raise util.ParameterError('Requires either file or index')
    return

# returns all file name strings in websocket_data/_test_files with indices given in 'include' or with indices not given in 'exclude'
# returns all file name strings if neither include nor exclude are specified
def test_files(include=None, exclude=None):
    d = 'websocket_data/_test_files/'
    return _get_files_(d, include, exclude)

# returns all file name strings in websocket_data/btc_files with indices given in 'include' or with indices not given in 'exclude'
# returns all file name strings if neither include nor exclude are specified
def btc_files(include=None, exclude=None):
   d = 'websocket_data/btc_files/'
   return _get_files_(d, include, exclude)

def _get_files_(d, include, exclude):
    all_files = [d+file for file in listdir(d)]
    if include:
        return [all_files[i] for i in include]
    elif exclude:
        return [all_files[i] for i in range(len(all_files)) if i not in exclude]
    else:
        return all_files
    return

# returns the total number of days spanned by all test_files
def get_test_file_days():
    return _file_days_(test_files())

# returns the total number of days spanned by all btc_files
def get_btc_file_days():
    return _file_days_(btc_files())

# common function for get_test_file_days and get_btc_file_days
def _file_days_(files):
    total_ms = 0
    for file in files:
        times, _ = zip(*gen_file(file))
        total_ms += times[-1] - times[0]
    return total_ms/86400000

def get_total_p_gain(bals, baseline=Globs.quote_amount):
    p_gains = [util.percentage(bal, baseline) for bal in bals]
    return sum(p_gains)

# creates a copy of this file
def copy_simulate(new_name='simulate_copy.py'):
    new_name = new_name if new_name[-3:] == '.py' else new_name+'.py'
    shutil_copy2('simulate.py', new_name)
    return

#=========================================================================================================================================================
# LOCAL ATTRIBUTE CREATION

def create_attributes():
    files = test_files() + btc_files()
    a_dir = 'websocket_data/attributes/'
    ls = listdir(a_dir)
    for file in files:
        f = file.split('files/')[1][:-4]
        d = a_dir+f
        if f not in ls:
            mkdir(d)
        attributes = listdir(d)
        if 'historical.json' not in attributes:
            create_historical_json(file)
        # if 'local_gradients.txt' not in attributes or 'mid_gradients.txt' not in attributes:
        #     create_gradients_txt(file)
    print("FINISHED")
    return

def create_historical_json(filename):
    symbol = filename_symbol(filename)
    times, _ = zip(*gen_file(filename))
    end = times[0]
    start = util.hours_ago(10, end=end)
    info = util.get_historical_prices_detailed(symbol, interval='1m', start=start, end=end)
    fname = filename.split('files/')[1][:-4]
    filename = 'websocket_data/attributes/'+fname+'/historical.json'
    util.dict_to_json(info, filename)
    print(c.C+"CREATED HISTORICAL ATTRIBUTE FOR {}".format(fname)+c.X)
    return

def create_gradients_txt(filename):
    class VirTrader(SimTrader):
        def __init__(self, symbol, asset):
            self.wallet = Test_Wallet(init_assets={asset.quote:asset.trade_balance})
            self.assets = {symbol:asset}
            self.total_initial_usd = 1000
            self.symbol = symbol
            self.base, self.quote = util.split_symbol(symbol)
            self.trade_decision_functions = {'buying':self.buy_decision, 'selling':self.sell_decision, 'tanking':self.tank_decision, 'final':self.final_decision}
            return
    # virtual asset class to simulate data
    class VirAsset(Asset):
        def __init__(self, symbol, filename, seeded):
            self.filename = filename
            trade_balance = Globs.quote_amount
            start_time = next(gen_file(filename))[0]
            super().__init__(symbol, trade_balance, start_time, seeded=seeded, seed=Globs.seed, session_folder='', log=False)
    # retrieve gradient data from simulation
    symbol = filename_symbol(filename)
    asset = VirAsset(symbol, filename, True)
    asset.trader = VirTrader(asset.symbol, asset)
    lgs, mgs = [], []
    with util.NoPrint():
        for time, price in gen_file(filename):
            asset.compute_asset_data(time, price)
            lgs.append(asset.local_gradient)
            mgs.append(asset.mid_gradient)
    # write to file
    fname = filename.split('files/')[1][:-4]
    with open('websocket_data/attributes/'+fname+'/local_gradients.txt', 'w') as f:
        for g in lgs:
            f.write(str(g)+'\n')
    with open('websocket_data/attributes/'+fname+'/mid_gradients.txt', 'w') as f:
        for g in mgs:
            f.write(str(g)+'\n')
    print(c.C+"CREATED GRADIENT ATTRIBUTES FOR {}".format(fname)+c.X)

#=========================================================================================================================================================
# HISTORICAL DATA SYNTHESIS

class SimHistorian(Historian):

    def __init__(self, filename, end):
        self.filename = filename
        self.end = end
        return

    def detailed_historical(self):
        info = self.load_historical()
        times = list(info)
        cut_front = bisect_left(times, self.start)
        cut_back = bisect_right(times, self.end)
        prices, trades = [], []
        for val in info.values():
            trades.append(val['trades'])
            del val['trades']
            prices.append(val)
        return times[cut_front:cut_back], prices[cut_front:cut_back], trades[cut_front:cut_back]

    def load_historical(self):
        fname = 'websocket_data/attributes/'+self.filename.split('files/')[1][:-4]+'/historical.json'
        info = util.json_to_dict(fname)
        info = {int(time):val for time, val in info.items()}
        return info

#=========================================================================================================================================================

# # returns the historical timeprice dictionary for the data corresponding to that in the given file
# def replicate_file(filename):
#     symbol = filename_symbol(filename)
#     times, _ = zip(*gen_file(filename))
#     start = times[0]
#     end = times[-1]
#     T = (end-start)/1000
#     htimes, hprices, htrades = detailed_historical(symbol, start, end)
#     trade_frequency = sum(htrades)/T
#     message_frequency = get_message_frequency(trade_frequency)
#     filled_times, filled_prices = fill_prices(htimes, hprices, message_frequency)
#     return {symbol:dict(zip(filled_times, filled_prices))}

def get_message_frequency(trades, a=0.15009905):
    return a*npsqrt(trades)

# returns the trade interval parameter 'a' used in get_message_frequency()
def compute_trade_interval_parameters(mute=False, plot=True):
    trades, intervals = get_klines_for_ctip(mute)
    popt, _ = curve_fit(get_message_frequency, trades, intervals)
    if plot:
        plt.scatter(trades, intervals, label='true')
        plt.plot(trades, get_message_frequency(trades, popt), label='fit')
        plt.legend()
        plt.show()
    return popt

def get_klines_for_ctip(mute=False):
    d = 'websocket_data/_test_files/'
    files = listdir(d)
    info = {}
    perc = 0
    for file in files:
        t, _ = zip(*gen_file(d+file))
        symbol = filename_symbol(d+file)
        start = t[0]
        end = t[-1]
        T = (end-start)/1000
        av_interval = len(t)/T
        klines = util.get_klines(symbol, start=start, end=end)
        av_trades = sum([k[8] for k in klines])/T
        info[file] = (av_trades, av_interval)
        if not mute:
            perc = util.progress_tracker(perc, file, files, 10)
    trades, intervals = zip(*sorted(info.values()))
    return trades, intervals

#=========================================================================================================================================================
# ULTRA MISC

# # code I used to create 3 hour versions of all longer test files
# # creates multiple unique files for files with integer multiple length greater than 'hours' e.g. two 3 hour files will be created from a 7 hour file
# def truncate_test_file_to_hours(hours=3):

#     def get_hours(t, h, ts):
#         index = bisect_right(t, t[0]+3600000*h)
#         t_ = t[:index]
#         if len(t_) == len(t):
#             return ts
#         ts.append(t_)
#         t = t[index:]
#         return get_hours(t, h, ts)

#     def write(t, p, file):
#         with open('websocket_data/_test_file/'+file, 'w') as f:
#             for i in range(len(t)):
#                 f.write(str(t[i])+';'+str(p[i])+'\n')
#         return

#     files = simulate.test_files()
#     for file in files:
#         filename = simulate.test_file(file=file)
#         t, p = zip(*simulate.gen_file(filename))
#         #t = util.timestamps2seconds(t)
#         ts = get_hours(t, 3, [])

#         for ti in ts:
#             start = t.index(ti[0])
#             end = t.index(ti[-1]) +1
#             pi = p[start:end]
#             write(ti, pi, file[:-4]+'_'+str(ts.index(ti)+1)+'.txt')

#     return

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================
# ANALYSIS

fname_length = 33

def run_all(mute=False, plot=False, globs=None, include=None, exclude=None):
    files = test_files(include, exclude)
    n_files = len(files)
    leading_zeros = len(str(n_files)) - 1
    bals = []
    start = util.now()
    for index, file in enumerate(files):
        # running
        sim = Simulator(filename=file, globs=globs)
        sim.run(mute=True, timed=False)
        # results
        bal = sim.score
        bals.append(bal)
        time = sim.run_time
        # printing
        if not mute:
            cb = util.red_or_green(bal, Globs.quote_amount)
            f = file.split('files/')[1][:-4]
            file_str = util.fill_string(f, fname_length)
            bal_str = util.fill_string("Balance: {}".format(bal), 22)
            time_str = util.fill_string("Time: {}s".format(time), 13)
            counter_str = "{}/{}".format(util.parse_integer(index+1, leading_zeros), n_files)
            string = file_str+div+cb+bal_str+div+c.Y+time_str+div+c.C+counter_str+div+util.now_string_time_only()
            print(string)
        # plotting
        if plot:
            sim.plot_simulation(splines=True)
    # final prints
    if not mute:
        av_score = round(mean(bals), 2)
        col = util.red_or_green(av_score, Globs.quote_amount)
        print(col+"AVERAGE SCORE: {}".format(av_score)+c.X)
        days = get_test_file_days()
        p_gain_per_day = round(get_total_p_gain(bals)/days, 2)
        col = util.red_or_green(p_gain_per_day, 0)
        print(col+"GAIN PER DAY:  {}%".format(p_gain_per_day)+c.X)
        print(c.Y+"TIME: {} minutes".format(round((util.now()-start)/60000,2))+c.X)
    return bals

def run_all_compound(mute=False, globs=None, shuffle=False, include=None, exclude=None):
    globs = globs if globs else {}
    files = test_files(include, exclude)
    if shuffle:
        npshuffle(files)
    n_files = len(files)
    leading_zeros = len(str(n_files)) - 1
    running_balance = Globs.quote_amount
    initial_balance = Globs.quote_amount
    start = util.now()
    for index, file in enumerate(files):
        # running
        globs['quote_amount'] = running_balance
        sim = Simulator(filename=file, globs=globs)
        sim.run(mute=True, timed=False)
        # results
        bal = sim.score
        running_balance = sim.score
        time = sim.run_time
        # printing
        if not mute:
            cb = util.red_or_green(bal, initial_balance)
            f = file.split('files/')[1][:-4]
            file_str = util.fill_string(f, fname_length)
            bal_str = util.fill_string("Balance: {}".format(bal), 22)
            time_str = util.fill_string("Time: {}s".format(time), 13)
            counter_str = "{}/{}".format(util.parse_integer(index+1, leading_zeros), n_files)
            string = file_str+div+cb+bal_str+div+c.Y+time_str+div+c.C+counter_str+div+util.now_string_time_only()
            print(string)
    # final prints
    if not mute:
        col = util.red_or_green(running_balance, initial_balance)
        print(col+"TOTAL SCORE:  {}".format(running_balance)+c.X)
        total_gain = round(util.percentage(running_balance, initial_balance), 2)
        print(col+"TOTAL GAIN:   {}%".format(total_gain)+c.X)
        days = get_test_file_days()
        p_gain_per_day = round(total_gain/days, 2)
        col = util.red_or_green(p_gain_per_day, 0)
        print(col+"GAIN PER DAY: {}%".format(p_gain_per_day)+c.X)
        print(c.Y+"TIME: {} minutes".format(round((util.now()-start)/60000,2))+c.X)
    return running_balance

def run_all_iterations(iterations=1, mute=False, print_scores=False, print_increments=False, print_inner_increments=False, globs=None, include=None, exclude=None):
    files = test_files(include, exclude)
    n_files = len(files)
    leading_zeros = len(str(n_files)) - 1
    bals = []
    start = util.now()
    for index, file in enumerate(files):
        # running
        scores = []
        f = file.split('files/')[1][:-4]
        if print_scores:
            print(f)
        start2 = util.now()
        for i in range(iterations):
            sim = Simulator(filename=file, globs=globs)
            sim.run(mute=True, timed=False)
            score = sim.score
            scores.append(score)
            if print_scores:
                end = '\n' if i==iterations-1 else ''
                print(util.red_or_green(score, Globs.quote_amount)+str(int(score))+' '+c.X, end=end)
            if print_inner_increments:
                print('.', end='')
        # results
        time = round((util.now()-start2)/1000,2)
        bal = round(mean(scores),8)
        bals.append(bal)
        # printing
        if not mute:
            cb = util.red_or_green(bal, Globs.quote_amount)
            file_str = util.fill_string(f, fname_length)
            bal_str = util.fill_string("Balance: {}".format(bal), 22)
            time_str = util.fill_string("Time: {}s".format(time), 13)
            counter_str = "{}/{}".format(util.parse_integer(index+1, leading_zeros), n_files)
            string = c.M+file_str+div+cb+bal_str+div+c.Y+time_str+div+c.C+counter_str+div+util.now_string_time_only()
            print(string)
        if print_increments:
            print('#', end='')
    # final prints
    if not mute:
        av_score = round(mean(bals), 2)
        col = util.red_or_green(av_score, Globs.quote_amount)
        print(col+"AVERAGE SCORE: {}".format(av_score)+c.X)
        days = get_test_file_days()
        p_gain_per_day = round(get_total_p_gain(bals)/days, 2)
        col = util.red_or_green(p_gain_per_day, 0)
        print(col+"GAIN PER DAY:  {}%".format(p_gain_per_day)+c.X)
        print(c.Y+"TIME: {} minutes".format(round((util.now()-start)/60000,2))+c.X)
    return bals

# used in genetics
def run_all_genetics(iterations, globs):
    files = test_files()
    bals = []
    for file in files:
        scores = []
        for i in range(iterations):
            sim = Simulator(filename=file, globs=globs)
            sim.run(mute=True, timed=False)
            scores.append(sim.score)
        bal = round(mean(scores),8)
        bals.append(bal)
    return bals

def multiprocess_iterations(globs=None, iterations=1, mute=False, include=None, exclude=None):
    # runnning
    files = test_files(include, exclude)
    n_files = len(files)
    leading_zeros = len(str(n_files)) - 1
    bals = []
    start = util.now()
    for index, file in enumerate(files):
        f = file.split('files/')[1][:-4]
        if not mute:
            print(f)
        start2 = util.now()
        # running
        file_data = zip([file]*iterations, [globs]*iterations)
        p = Pool()
        scores = p.starmap(mp_evaluate, file_data)
        p.close()
        p.join()
        # results
        bal = round(mean(scores),8)
        bals.append(bal)
        # printing
        if mute:
            continue
        for score in scores:
            col = util.red_or_green(score, Globs.quote_amount)
            print("{}{} ".format(col, int(score)), end='')
        print('')
        time = round((util.now()-start2)/1000,2)
        cb = util.red_or_green(bal, Globs.quote_amount)
        file_str = util.fill_string(f, fname_length)
        bal_str = util.fill_string("Balance: {}".format(bal), 22)
        time_str = util.fill_string("Time: {}s".format(time), 13)
        counter_str = "{}/{}".format(util.parse_integer(index+1, leading_zeros), n_files)
        string = c.M+file_str+div+cb+bal_str+div+c.Y+time_str+div+c.C+counter_str+div+util.now_string_time_only()
        print(string)
    # final prints
    if not mute:
        av_score = round(mean(bals), 2)
        col = util.red_or_green(av_score, Globs.quote_amount)
        print(col+"AVERAGE SCORE: {}".format(av_score)+c.X)
        days = get_test_file_days()
        p_gain_per_day = round(get_total_p_gain(bals)/days, 2)
        col = util.red_or_green(p_gain_per_day, 0)
        print(col+"GAIN PER DAY:  {}%".format(p_gain_per_day)+c.X)
        print(c.Y+"TIME: {} minutes".format(round((util.now()-start)/60000,2))+c.X)
    return bals

def multiprocess_files(globs=None, mute=False, include=None, exclude=None):
    files = test_files(include, exclude)
    n_files = len(files)
    leading_zeros = len(str(n_files)) - 1
    start = util.now()
    # running
    data = zip(files, [globs]*n_files)
    p = Pool()
    bals = p.starmap(mp_evaluate, data)
    p.close()
    p.join()
    # results & printing
    if not mute:
        index = 0
        for file, bal in zip(files, bals):
            f = file.split('files/')[1][:-4]
            file_str = util.fill_string(f, fname_length)
            cb = util.red_or_green(bal, Globs.quote_amount)
            bal_str = util.fill_string("Balance: {}".format(bal), 22)
            counter_str = "{}/{}".format(util.parse_integer(index+1, leading_zeros), n_files)
            string = file_str+div+cb+bal_str+div+c.C+counter_str+c.X
            print(string)
            index += 1
        av_score = round(mean(bals), 2)
        col = util.red_or_green(av_score, Globs.quote_amount)
        print(col+"AVERAGE SCORE: {}".format(av_score)+c.X)
        days = get_test_file_days()
        p_gain_per_day = round(get_total_p_gain(bals)/days, 2)
        col = util.red_or_green(p_gain_per_day, 0)
        print(col+"GAIN PER DAY:  {}%".format(p_gain_per_day)+c.X)
        print(c.Y+"TIME: {} minutes".format(round((util.now()-start)/60000,2))+c.X)
    return bals

# Pool function
def mp_evaluate(filename, globs):
    sim = Simulator(filename=filename, globs=globs)
    sim.run(mute=True, timed=False)
    return sim.score

#------------------------------------------------------------------------------------------------------------------------
# BTC

btc_fname_length = 30

def run_all_btc(mute=False, plot=False, globs=None, include=None, exclude=None):
    files = btc_files(include, exclude)
    n_files = len(files)
    leading_zeros = len(str(n_files)) - 1
    bals = []
    start = util.now()
    for index, file in enumerate(files):
        # running
        sim = BtcSimulator(file, globs=globs)
        sim.run(timed=False)
        # results
        bal = sim.score
        bals.append(bal)
        time = sim.run_time
        # printing
        if not mute:
            cb = util.red_or_green(bal, Globs.quote_amount)
            f = file.split('files/')[1][:-4]
            file_str = util.fill_string(f, btc_fname_length)
            bal_str = util.fill_string("Balance: {}".format(bal), 22)
            time_str = util.fill_string("Time: {}s".format(time), 13)
            counter_str = "{}/{}".format(util.parse_integer(index+1, leading_zeros), n_files)
            string = file_str+div+cb+bal_str+div+c.Y+time_str+div+c.C+counter_str+div+util.now_string_time_only()
            print(string)
        # plotting
        if plot:
            sim.plot_simulation()
    # final prints
    if not mute:
        av_score = round(mean(bals), 2)
        col = util.red_or_green(av_score, Globs.quote_amount)
        print(col+"AVERAGE SCORE: {}".format(av_score)+c.X)
        days = get_btc_file_days()
        p_gain_per_day = round(get_total_p_gain(bals)/days, 2)
        col = util.red_or_green(p_gain_per_day, 0)
        print(col+"GAIN PER DAY:  {}%".format(p_gain_per_day)+c.X)
        print(c.Y+"TIME: {} minutes".format(round((util.now()-start)/60000,2))+c.X)
    return bals

def run_all_compound_btc(mute=False, globs=None, shuffle=False, include=None, exclude=None):
    globs = globs if globs else {}
    files = btc_files(include, exclude)
    if shuffle:
        npshuffle(files)
    n_files = len(files)
    leading_zeros = len(str(n_files)) - 1
    running_balance = Globs.quote_amount
    initial_balance = Globs.quote_amount
    start = util.now()
    for index, file in enumerate(files):
        # running
        globs['quote_amount'] = running_balance
        sim = BtcSimulator(filename=file, globs=globs)
        sim.run(timed=False)
        # results
        bal = sim.score
        running_balance = sim.score
        time = sim.run_time
        # printing
        if not mute:
            cb = util.red_or_green(bal, initial_balance)
            f = file.split('files/')[1][:-4]
            file_str = util.fill_string(f, btc_fname_length)
            bal_str = util.fill_string("Balance: {}".format(bal), 22)
            time_str = util.fill_string("Time: {}s".format(time), 13)
            counter_str = "{}/{}".format(util.parse_integer(index+1, leading_zeros), n_files)
            string = file_str+div+cb+bal_str+div+c.Y+time_str+div+c.C+counter_str+div+util.now_string_time_only()
            print(string)
    # final prints
    if not mute:
        col = util.red_or_green(running_balance, initial_balance)
        print(col+"TOTAL SCORE:  {}".format(running_balance)+c.X)
        total_gain = round(util.percentage(running_balance, initial_balance), 2)
        print(col+"TOTAL GAIN:   {}%".format(total_gain)+c.X)
        days = get_btc_file_days()
        p_gain_per_day = round(total_gain/days, 2)
        col = util.red_or_green(p_gain_per_day, 0)
        print(col+"GAIN PER DAY: {}%".format(p_gain_per_day)+c.X)
        print(c.Y+"TIME: {} minutes".format(round((util.now()-start)/60000,2))+c.X)
    return running_balance

# iterations not used but required by genetics interface
def run_all_genetics_btc(iterations, globs):
    files = btc_files()
    scores = []
    for file in files:
        sim = BtcSimulator(file, globs=globs)
        sim.run(timed=False)
        scores.append(sim.score)
    return scores

def multiprocess_files_btc(globs=None, mute=False, include=None, exclude=None):
    files = btc_files(include, exclude)
    n_files = len(files)
    leading_zeros = len(str(n_files)) - 1
    start = util.now()
    # running
    data = zip(files, [globs]*n_files)
    p = Pool()
    bals = p.starmap(mp_evaluate_btc, data)
    p.close()
    p.join()
    # results & printing
    if not mute:
        index = 0
        for file, bal in zip(files, bals):
            f = file.split('files/')[1][:-4]
            file_str = util.fill_string(f, btc_fname_length)
            cb = util.red_or_green(bal, Globs.quote_amount)
            bal_str = util.fill_string("Balance: {}".format(bal), 22)
            counter_str = "{}/{}".format(util.parse_integer(index+1, leading_zeros), n_files)
            string = file_str+div+cb+bal_str+div+c.C+counter_str+c.X
            print(string)
            index += 1
        av_score = round(mean(bals), 2)
        col = util.red_or_green(av_score, Globs.quote_amount)
        print(col+"AVERAGE SCORE: {}".format(av_score)+c.X)
        days = get_btc_file_days()
        p_gain_per_day = round(get_total_p_gain(bals)/days, 2)
        col = util.red_or_green(p_gain_per_day, 0)
        print(col+"GAIN PER DAY:  {}%".format(p_gain_per_day)+c.X)
        print(c.Y+"TIME: {} minutes".format(round((util.now()-start)/60000,2))+c.X)
    return bals

def mp_evaluate_btc(filename, globs):
    sim = BtcSimulator(filename, globs)
    sim.run(timed=False)
    return sim.score

#====================================================================================================================================================================================================================
# TRENDS

def bound_change(filename):
    t, p = zip(*gen_file(filename))
    return util.percentage(p[-1], p[0])

def fit_change(filename):
    t, p = zip(*gen_file(filename))
    fit = util.best_fit(t, p)
    return util.percentage(fit[-1], fit[0])

def total_bound_change(file_type='normal'):
    files = btc_files() if file_type == 'btc' else test_files()
    changes = [bound_change(filename) for filename in files]
    return round(mean(changes), 3)

def total_fit_change(file_type='normal'):
    files = btc_files() if file_type == 'btc' else test_files()
    changes = [fit_change(filename) for filename in files]
    return round(mean(changes), 3)
