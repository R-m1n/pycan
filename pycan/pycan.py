import logging
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from datetime import timedelta, datetime
from pathlib import Path
from itertools import product
from typing import Any
from dateutil import parser

logging.basicConfig(level=logging.DEBUG, filename=Path("./fipy.log"), filemode=("w"),
                    format="%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s")


class Miner():
    """
    A class associated with extracting financial data from yahoo finance. 

    Miner class, initialized with a valid symbol(s), extracts financial data of the specified symbol
    from yahoo finance API and writes it in a CSV file. the data is then stored in the attribute 'dataframe' 
    as a pandas DataFrame object. it also can be initialized with the path of an existing CSV file.
    atleast one of the two keyword arguments 'symbols' or 'path' have to be passed to the constructor,
    for the Miner class to be properly initialized. The objective of this class is to provide essential data used
    by the Analyst class to perform Analysis of the financial markets.  

    Attributes
    ----------
    symbols: list
        List of symbol(s).
    start: str
        Time of the first data point.
    end: str
        Time of the last data point.
    period: str
        Valid periods: "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max". 
        Note: Either use 'period' parameter or use 'start' and 'end'.
    interval: str
        Valid intervals: "1m","2m","5m","15m","30m","90m","1h","1d","5d","1wk","1mo","3mo". 
        Note: Minutely data cannot extend last 60 days, and hourly data cannot extend last 730 days.
    path: pathlib.Path
        Path of the CSV file.
    dataframe: pandas.DataFrame
        Pandas.DataFrame object of the CSV file.
    replace: bool
        Boolean parameter that determines whether the data should be downloaded everytime a new instance is created.
        Note: If True, then symbols parameter has to be passed as well.


    Methods
    -------
    write_csv(symbols, start = None, end = None, period = "max", interval = "1d", path = _DEFAULT_PATH)
        Downloads the financial data of the specified symbol, then writes it in a CSV file.
    read_csv(path, header = None)
        Returns a pandas.DataFrame object of the specified CSV file.
    update_csv()
        Updates the CSV file to the last data point available.
    get_price(symbol = None)
        Returns a pandas.DataFrame object of the 'Close' column of a symbol.
    get_volume(symbol = None)
        Returns a pandas.DataFrame object of the 'Volume' column of a symbol.
    get_dataframe()
        Returns the dataframe attribute.

    """

    _DEFAULT_PATH = Path("./raw_data.csv")
    __VALID_PERIODS = ["1d", "5d", "1mo", "3mo",
                       "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    __VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m",
                         "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

    def __init__(self, symbols: Any | None = None, start: Any | None = None, end: Any | None = None,
                 period: str = "max", interval: str = "1d", path: Any = _DEFAULT_PATH, replace: bool = False):

        self.symbols = list(symbols) if isinstance(symbols, str) else symbols
        self.start = self.__date_check(start) if start != None else start
        self.end = self.__date_check(end) if end != None else end
        self.interval = self.__interval_check(interval)\
            if interval != "1d" else interval
        self.period = self.__period_check(period)\
            if period != "max" else period
        self.path = Path(path) if isinstance(path, str) else path
        self.replace = self.__replace_check(replace)
        self.dataframe = None
        self._header = self.__header_check()

        if self._check_csv() and self._file_check() and self._init_check(start, end, period):
            self.write_csv(self.symbols, start=start, end=end, period=self.period,
                           interval=self.interval, path=self.path)

    def write_csv(self, symbols: Any, start: Any | None = None, end: Any | None = None,
                  period: str = "max", interval: str = "1d", path: Any = _DEFAULT_PATH) -> None:
        """
        Downloads the financial data of the specified symbol, then writes it in a CSV file.

        Parameters
        ----------
        symbols: str, list
            List of symbols to download
        start: str
            Download start date string (YYYY-MM-DD).
        end: str
            Download end date string (YYYY-MM-DD).
        period: str
            Valid periods: "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max". 
            Note: Either use 'period' parameter or use 'start' and 'end'.
        interval: str
            Valid intervals: "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo". 
            Note: Minutely data cannot extend last 60 days, and hourly data cannot extend last 730 days.
        path: str
            Path of where the CSV file should be saved.
        """

        default_period = period != "max"
        default_interval = interval != "1d"

        if default_period and default_interval:
            # as minutely data is bound by 60 days and hourly data by 730 days
            period = self.__period_check(period)
            interval = self.__interval_check(interval)
            yf.download(symbols, period=period,
                        interval=interval).to_csv(path)

        elif default_period:
            period = self.__period_check(period)
            yf.download(symbols, period=period).to_csv(path)

        elif default_interval:
            # to alter self.start as data is bounded
            interval = self.__interval_check(interval)
            yf.download(symbols, start=self.start,
                        interval=interval).to_csv(path)

        else:
            yf.download(symbols, start=start, end=end).to_csv(path)

        self.dataframe = self.read_csv(self.path)
        self._set_time()
        logging.info(f"CSV file created at {self.path.absolute()}")
        print(f"CSV file created at {self.path.absolute()}")

    def read_csv(self, path: Any) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame object of the specified CSV file.

        Parameters
        ----------
        path: str
            Path of the CSV file.

        Returns
        -------
        pandas.DataFrame
        """

        try:
            return pd.read_csv(path, header=self._header, index_col=0, parse_dates=[0])

        except Exception as e:
            logging.error("Invalid file!")
            print(e)

    def update_csv(self) -> None:
        """
        Updates the CSV file to the last data point available.(inplace)
        """

        self.write_csv(self.symbols, start=self.start, period=self.period,
                       interval=self.interval, path=self.path)
        logging.info("CSV file updated.")
        print("CSV file has been updated to the latest data point.")

    def get_price(self, symbol: str | None = None) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame object of the 'Close' column of a symbol.

        Parameters
        ----------
        symbol: str

        Returns
        -------
        pandas.DataFrame
        """

        close = self.dataframe["Close"].to_frame() \
            if self._is_singular() else self.dataframe["Close"]
        close = close.replace({0: np.nan}).copy()
        close.ffill(inplace=True)
        close = close.dropna().copy()

        if self._is_singular():
            return close.rename(columns={"Close": "Price"})

        if symbol in close.columns:
            return close[symbol].to_frame().rename(columns={symbol: f"{symbol}\Price"})

        logging.error("Invalid symbol!")
        raise ValueError(
            f"\nPlease choose from available symbols: \n{list(close.columns)}")

    def get_volume(self, symbol: str | None = None) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame object of the 'Volume' column of a symbol.

        Parameters
        ----------
        symbol: str

        Returns
        -------
        pandas.DataFrame
        """

        volume = self.dataframe["Volume"].to_frame(
        ) if self._is_singular() else self.dataframe["Volume"]
        volume = volume.replace({0: np.nan}).copy()
        volume.ffill(inplace=True)
        volume = volume.dropna().copy()

        if self._is_singular():
            return volume

        if symbol in volume.columns:
            return volume[symbol].to_frame().rename(columns={symbol: f"{symbol}\Volume"})

        logging.error("Invalid symbol!")
        raise ValueError(
            f"\nPlease choose from available symbols: \n{list(volume.columns)}")

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the dataframe attribute.

        Returns
        -------
        pandas.DataFrame
        """

        return self.dataframe

    def _set_time(self):
        """
        Sets 'start' and 'end' attribute according to the first and last data point.
        """

        self.start = str(self.dataframe.index[0])
        self.end = str(self.dataframe.index[-1])

    def _is_singular(self):
        """
        Checks if there's only one symbol in the dataframe.
        """

        return True if len(self._header) == 1 else False

    def _check_csv(self):
        """
        Checks for a CSV file at the given path and if the replace attribute is false 
        reads the data without calling the write function.

        -kinda the whole point of this mess :(-
        """

        if self.path.is_file() and self.path.suffix == ".csv" and not self.replace:
            self.dataframe = self.read_csv(self.path)
            self._set_time()
            return False

        return True

    def _file_check(self):
        """
        Checks if the file is CSV.
        """

        if self.path.is_file() and self.path.suffix != ".csv":
            logging.error("Invalid file!")
            raise FileNotFoundError("The file has to be a CSV file!")

        return True

    def _init_check(self, start, end, period):
        """
        Checks for the initializion requirements.
        """

        if start != None and end != None and period != None:
            logging.error("Invalid initialization!")
            raise ValueError(
                "Use either 'period' parameter or 'start' and 'end' parameters.")

        if self.symbols == None and self.path == self._DEFAULT_PATH:
            logging.error("Invalid initialization!")
            raise ValueError(
                "Please either pass a list of valid symbol(s) or the path of a valid CSV file.")

        return True

    def __header_check(self):
        """
        Determines the proper header (a dataframe parameter) based on the number of symbols.
        """

        if self.symbols != None:
            return [0, 1] if len(self.symbols) > 1 else [0]

        return [0]

    def __date_check(self, date):
        """
        Checks if the date is in the right format (YYYY-MM-DD).
        """

        try:
            parser.parse(date)
        except ValueError as ve:
            logging.error("Invalid initialization!")
            raise ve

        return date

    def __period_check(self, period):
        """
        Checks the validity of the given period parameter, also adjusts the period attribute based on the initialized interval.
        """

        if period not in self.__VALID_PERIODS:
            logging.error("Invalid initialization!")
            raise ValueError(
                f"Please pass a valid period {self.__VALID_PERIODS}.")

        if period in self.__VALID_PERIODS[3:] and self.interval in self.__VALID_INTERVALS[:6]:
            return "1mo"

        if period in self.__VALID_PERIODS[7:] and self.interval == "1h":
            return "2y"

        return period

    def __interval_check(self, interval):
        """
        Checks the validity of the given interval parameter, also adjusts the start attribute based on it.
        """

        if interval not in self.__VALID_INTERVALS:
            logging.error("Invalid initialization!")
            raise ValueError(
                f"Please pass a valid interval {self.__VALID_INTERVALS}.")

        elif interval in self.__VALID_INTERVALS[:6]:
            self.start = str((datetime.now() - timedelta(days=59)).date())

        elif interval == "1h":
            self.start = str((datetime.now() - timedelta(days=729)).date())

        return interval

    def __replace_check(self, replace):
        """
        Checks the type of the given replace parameter, also the requirement for it to be 'True'.
        """

        if not isinstance(replace, bool):
            logging.error("Invalid initialization!")
            raise TypeError(
                f"'replace' type should be of type 'bool' not '{type(replace)}'")

        elif self.symbols == None and replace == True:
            logging.error("Invalid initialization!")
            raise ValueError("Atleast one symbol should be passed!")

        return replace


class Analyst():
    """
    A class which utilizes the data extracted by the Miner class to perform financial analysis. 

    Analyst class is initialized with an instance of the Miner class, it's used to do backtesting of 
    particular trading strategies, the backtest data could then be used to perform forward testing of 
    the strategy, The Long Only strategy is the default strategy for the Analyst class.
    The Analyst class could also be used to plot a variety of financial data namely Mean Variance Analysis,
    Returns Frequency Distribution, Resampled Performance Comparison, Simple Moving Average, etc.

    Attributes
    ----------
    miner: Miner
        An instance of the Miner class.
    symbols: list
        List of symbol(s).
    dataframe: pandas.DataFrame
        Pandas.DataFrame object of the CSV file.


    Methods
    -------
    normalize(symbol = None, context = _DEFAULT_CONTEXT)
        Returns the normalized data.
    abs_change(symbol = None, context = _DEFAULT_CONTEXT)
        Returns the absolute difference of data.
    changes(symbol = None, context = _DEFAULT_CONTEXT)
        Returns the arithmetic financial returns.
    log_changes(symbol = None, context = _DEFAULT_CONTEXT)
        Returns the logarithmic financial returns.
    multiple(symbol = None, context = _DEFAULT_CONTEXT)
        Returns the arithmetic multiple of data.
    log_multiple(symbol = None, context = _DEFAULT_CONTEXT)
        Returns the logarithmic multiple of data.
    cagr(symbol = None, context = _DEFAULT_CONTEXT)
        Returns the Compound Annual Growth Rate.
    geo_mean(symbol = None, context = _DEFAULT_CONTEXT)
        Returns the geometric mean of data.
    skew(symbol = None, context = _DEFAULT_CONTEXT)
        Returns skewness of data.
    kurtosis(symbol = None, context = _DEFAULT_CONTEXT)
        Returns kurtosis of data.
    creturns(symbol = None, context = _DEFAULT_CONTEXT)
        Returns the cumulative financial returns.
    sharpe_ratio(symbol = None, context = _DEFAULT_CONTEXT, metrics = "CAGR")
        Returns the sharpe ratio (Risk-Adjusted Return) of financial data based on "CAGR" or "Mean".
    interpret(symbol = None, start = None, end = None)
        Returns interpreted data columns = ("Price", "Volume", "Returns", "cReturns", "volChanges", "retCategory", "volChCategory").
    long_only_strategy(symbol = None, parameters = None, start = None, end = None, commissions = _DEFAULT_COMISSION, slippage = _DEFAULT_SLIPPAGE)
        Returns data based on The Long Only Strategy columns = ("Position", "Trades", "Strategy", "cStrategy", "stratNet", "cStratNet").
    optimized_parameters(symbol = None, start = None, end = None, low_ret_thresh = 85, high_ret_thresh = 98, low_vol_thresh = 2, moderate_vol_thresh = 16, high_vol_thresh = 35)
        Returns the most optimized parameters based on the greatest net strategy multiple achieved. 
    hodl(symbol = None)
        Returns the Buy and Hold Strategy multiple for the specified symbol.
    backtest(symbol = None, parameters = None, start = None, end = None, commissions = _DEFAULT_COMISSION, slippage = _DEFAULT_SLIPPAGE)
        Returns the backtest multiple for the specified symbol.
    forwardtest(symbol = None, start = None, end = None, commissions = _DEFAULT_COMISSION, slippage = _DEFAULT_SLIPPAGE, low_ret_thresh = 85, high_ret_thresh = 98, low_vol_thresh = 2, moderate_vol_thresh = 16, high_vol_thresh = 35, scope_pct = 0.65)
        Returns the forwardtest multiple for the specified symbol, a scope percentage could also be passed to determine the time of the last data point for parameter optimizing and the time of the first data point for forwardtesting.
    analyze(symbol = None, parameters = None, start = None, end = None, commissions = _DEFAULT_COMISSION, slippage = _DEFAULT_SLIPPAGE, low_ret_thresh = 85, high_ret_thresh = 98, low_vol_thresh = 2, moderate_vol_thresh = 16, high_vol_thresh = 35, scope_pct = 0.65, float_format = 6)
        Prints an overall analysis of a ticker symbol.
    mean_variance_analysis(context = _DEFAULT_CONTEXT)
        Plots the Mean-Variance Analysis of all of the available symbols.
    returns_freq_distro(symbol = None, context = _DEFAULT_CONTEXT)
        Plots the Returns Frequency Distribution and Normal Distribution of the specified symbol.
    resampled_comparison(symbol = None, context = _DEFAULT_CONTEXT)
        Plots the Resampled Performance Comparison of the specified symbol. 
    sma(symbol = None, context = _DEFAULT_CONTEXT, window = 100)
        Plots the Simple Moving Average based on a window. 
    creturns_plot(symbol = None, context = _DEFAULT_CONTEXT)
        Plots the Cumulative Financial Returns. 
    cat_heatmap(symbol = None)
        Plots Categorical Heatmap. 
    shifted_cat_heatmap(symbol = None)
        Plots Shifted Categorical Heatmap. 
    strategy_parameters_plot(symbol = None, low_ret_thresh = 85, high_ret_thresh = 98,low_vol_thresh = 2, moderate_vol_thresh = 16, high_vol_thresh = 35)
        Plots three subplots of Mean Performance of each possible strategy parameter. 


    Static Methods
    --------------
    discrete_compounding(ini_inv, rate, years=1, comp_interval=1)
        Returns the discrete compounding of an investment.
    continuous_compounding(ini_inv, rate, years=1)
        Returns the continuous compounding of an investment.
    quant_discretization(data, quants = 10, labels = np.arange(1, 11))
        Returns the Quantile-based discretization of the specified data.
    ptc(commissions: Any, slippage: Any)
        Returns the Proportional Trading Cost.
    strategy_multiple(strat_data: pd.DataFrame)
        Returns the net strategy multiple.
    """

    _DEFAULT_CONTEXT = "p"
    _DEFAULT_COMISSION = 0.00075
    _DEFAULT_SLIPPAGE = 0.0001

    def __init__(self, miner: Miner):
        self.miner = miner
        self.symbols = miner.symbols
        self._default_symbol = miner.symbols[0]

    def normalize(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> pd.DataFrame:
        """
        Returns the normalized data.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.

        Returns
        -------
        pandas.DataFrame
        """

        symbol = self._default_symbol if symbol == None else symbol
        data = self.__set_context(context)(symbol)
        return data.div(data.iloc[0])

    def abs_change(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> pd.DataFrame:
        """
        Returns the absolute difference of data.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.

        Returns
        -------
        pandas.DataFrame
        """

        symbol = self._default_symbol if symbol == None else symbol
        data = self.__set_context(context)(symbol)

        return data.diff(periods=1)

    def changes(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> pd.DataFrame:
        """
        Returns the arithmetic financial returns.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.

        Returns
        -------
        pandas.DataFrame
        """

        symbol = self._default_symbol if symbol == None else symbol
        data = self.__set_context(context)(symbol)

        return data.pct_change(periods=1)

    def log_changes(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> pd.DataFrame:
        """
        Returns the logarithmic financial returns.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.

        Returns
        -------
        pandas.DataFrame
        """

        symbol = self._default_symbol if symbol == None else symbol
        data = self.__set_context(context)(symbol)

        return np.log(data.div(data.shift()))

    def multiple(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> pd.Series:
        """
        Returns the arithmetic multiple of data.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.

        Returns
        -------
        pandas.Series
        """

        symbol = self._default_symbol if symbol == None else symbol
        data = self.__set_context(context)(symbol)

        return data.iloc[-1] / data.iloc[0]

    def log_multiple(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> pd.Series:
        """
        Returns the logarithmic multiple of data.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.

        Returns
        -------
        pandas.Series
        """

        data = self.log_changes(symbol, context)

        return np.exp(data.sum())

    def cagr(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> pd.Series:
        """
        Returns the Compound Annual Growth Rate.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.

        Returns
        -------
        pandas.Series
        """

        symbol = self._default_symbol if symbol == None else symbol
        years = (self.__set_context(context)(symbol).index[-1] -
                 self.__set_context(context)(symbol).index[0]).days / 365.25

        return self.log_multiple(symbol, context) ** (1/years) - 1

    def geo_mean(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> pd.Series:
        """
        Returns the geometric mean of data.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.

        Returns
        -------
        pandas.Series
        """

        return self.log_multiple(symbol, context) ** (1/len(self.__set_context(context)(symbol))) - 1

    def skew(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> float:
        """
        Returns skewness of data.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.

        Returns
        -------
        float
        """

        return float(stats.skew(self.log_changes(symbol, context).dropna()))

    def kurtosis(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> float:
        """
        Returns kurtosis of data.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.

        Returns
        -------
        float
        """

        return float(stats.kurtosis(self.log_changes(symbol, context).dropna(), fisher=True))

    def creturns(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> pd.DataFrame:
        """
        Returns the cumulative financial returns.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.

        Returns
        -------
        pandas.DataFrame
        """

        log_returns = self.log_changes(symbol, context)

        return log_returns.cumsum().apply(np.exp).rename(columns={log_returns.columns[0]: "Multiple"})

    def sharpe_ratio(self, symbol: str = None, context: str = _DEFAULT_CONTEXT, metrics: str = "CAGR") -> pd.Series:
        """
        Returns the sharpe ratio (Risk-Adjusted Return) of financial data based on "CAGR" or "Mean".

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.
        metrics: str
            Basis for the calculation of the sharpe ratio ("CAGR" or "Mean").

        Returns
        -------
        pandas.Series
        """

        ann_std = self.log_changes(symbol, context).std() * np.sqrt(365.25)

        if(metrics.lower() == "cagr"):
            return self.cagr(symbol, context) / ann_std

        elif(metrics.lower() == "mean"):
            return (self.log_changes(symbol, context).mean() * 365.25) / ann_std

        else:
            raise ValueError("metrics can only be based on 'Mean' or 'CAGR'")

    def interpret(self, symbol: str = None, start: str = None, end: str = None) -> pd.DataFrame:
        """
        Returns interpreted data columns = ("Price", "Volume", "Returns", "cReturns", "volChanges", "retCategory", "volChCategory").

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        start: str
            Time of the first data point.
        end: str
            Time of the last data point.

        Returns
        -------
        pandas.DataFrame
        """

        symbol = self._default_symbol if symbol == None else symbol
        is_interval = self.miner.interval == None
        df = pd.DataFrame()

        df["Price"] = self.miner.get_price(symbol).loc[start:end]
        df["Volume"] = self.miner.get_volume(symbol).loc[start:end]
        df["Returns"] = self.log_changes(symbol, "p")
        df["cReturns"] = self.creturns(symbol, "p")
        df["volChanges"] = self.log_changes(symbol, "v")
        df["retCategory"] = self.quant_discretization(df["Returns"]) \
            if is_interval else self.quant_discretization(df["Returns"].rank(method="first"))
        df["volChCategory"] = self.quant_discretization(df["volChanges"]) \
            if is_interval else self.quant_discretization(df["volChanges"].rank(method="first"))

        return df

    def long_only_strategy(self, symbol: str = None, parameters: Any = None, start: str = None, end: str = None,
                           commissions: int | float = _DEFAULT_COMISSION, slippage: int | float = _DEFAULT_SLIPPAGE) -> pd.DataFrame:
        """
        Returns data based on The Long Only Strategy columns = ("Position", "Trades", "Strategy", "cStrategy", "stratNet", "cStratNet").

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        parameters: Any
            Strategy parameters.
        start: str
            Time of the first data point.
        end: str
            Time of the last data point.
        comissions: int | float
            Comissions for each trade.
        slippage: int | float
            Slippage of each trade.

        Returns
        -------
        pandas.DataFrame
        """

        sdata = self.interpret(symbol, start, end)
        parameters = self.optimized_parameters(symbol) \
            if parameters == None else parameters
        sdata["Position"] = 1
        ret_thresh = np.percentile(sdata["Returns"].dropna(),
                                   parameters[0])
        vol_thresh = np.percentile(sdata["volChanges"].dropna(),
                                   parameters[1:])
        cond1 = sdata["Returns"] >= ret_thresh
        cond2 = sdata["volChanges"].between(vol_thresh[0], vol_thresh[1])
        sdata.loc[cond1 & cond2, "Position"] = 0
        sdata["Trades"] = sdata["Position"].diff().fillna(0).abs().apply(int)
        sdata["Strategy"] = sdata["Position"].shift() * sdata["Returns"]
        sdata["cStrategy"] = sdata["Strategy"].cumsum().apply(np.exp)
        sdata["stratNet"] = sdata["Strategy"] + \
            sdata["Trades"] * self.ptc(commissions, slippage)
        sdata["cStratNet"] = sdata["stratNet"].cumsum().apply(np.exp)

        return sdata

    def optimized_parameters(self, symbol: str = None, start: str = None, end: str = None, low_ret_thresh: int = 85, high_ret_thresh: int = 98,
                             low_vol_thresh: int = 2, moderate_vol_thresh: int = 16, high_vol_thresh: int = 35) -> tuple:
        """
        Returns the most optimized parameters based on the greatest net strategy multiple achieved by The Long Only strategy.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        start: str
            Time of the first data point.
        end: str
            Time of the last data point.
        low_ret_thresh: int
            Lower returns threshold.
        high_ret_thresh: int
            High returns threshold.
        low_vol_thresh: int
            Lower volume threshold.
        moderate_vol_thresh: int
            Moderate volume threshold.
        high_vol_thresh: int
            High volume threshold.

        Returns
        -------
        tuple
        """

        plist = list(product(list(range(low_ret_thresh, high_ret_thresh)),
                             list(range(low_vol_thresh, moderate_vol_thresh)),
                             list(range(moderate_vol_thresh, high_vol_thresh))))

        olist = []
        for parameters in plist:
            los = self.long_only_strategy(symbol, parameters, start, end)
            olist.append((parameters, self.strategy_multiple(los)))

        olist.sort(key=lambda c: c[1], reverse=True)

        return olist[0][0]

    def hodl(self, symbol: str = None) -> float:
        """
        Returns the Buy and Hold Strategy multiple for the specified symbol.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.

        Returns
        -------
        float
        """

        bh = self.interpret(symbol)
        return bh["cReturns"].iloc[-1]

    def backtest(self, symbol: str = None, parameters: Any = None, start: str = None, end: str = None,
                 commissions: int | float = _DEFAULT_COMISSION, slippage: int | float = _DEFAULT_SLIPPAGE) -> float:
        """
        Returns the backtest multiple for the specified symbol.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        parameters: Any
            Strategy parameters.
        start: str
            Time of the first data point.
        end: str
            Time of the last data point.
        comissions: int | float
            Comissions for each trade.
        slippage: int | float
            Slippage of each trade.

        Returns
        -------
        float
        """

        los = self.long_only_strategy(
            symbol, parameters, start, end, commissions, slippage)
        return los["cStratNet"].iloc[-1]

    def forwardtest(self, symbol: str = None, start: str = None, end: str = None, commissions: int | float = _DEFAULT_COMISSION,
                    slippage: int | float = _DEFAULT_SLIPPAGE, low_ret_thresh: int = 85, high_ret_thresh: int = 98,
                    low_vol_thresh: int = 2, moderate_vol_thresh: int = 16, high_vol_thresh: int = 35, scope_pct: float = 0.65) -> float:
        """
        Returns the forwardtest multiple for the specified symbol, a scope percentage could also be passed 
        to determine the time of the last data point for parameter optimizing and the time of the first data point 
        for forwardtesting.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        start: str
            Time of the first data point.
        end: str
            Time of the last data point.
        comissions: int | float
            Comissions for each trade.
        slippage: int | float
            Slippage of each trade.
        low_ret_thresh: int
            Lower returns threshold.
        high_ret_thresh: int
            High returns threshold.
        low_vol_thresh: int
            Lower volume threshold.
        moderate_vol_thresh: int
            Moderate volume threshold.
        high_vol_thresh: int
            High volume threshold.
        scope_pct: float
            Scope percentage.

        Returns
        -------
        float
        """

        data = self.miner.get_dataframe().loc[start:end]
        scope = data.index[int(len(data) * scope_pct)]
        p = self.optimized_parameters(symbol, end=scope, low_ret_thresh=low_ret_thresh, high_ret_thresh=high_ret_thresh,
                                      low_vol_thresh=low_vol_thresh, moderate_vol_thresh=moderate_vol_thresh, high_vol_thresh=high_vol_thresh)
        los = self.long_only_strategy(symbol, parameters=p, start=scope, commissions=commissions,
                                      slippage=slippage)
        return los["cStratNet"].iloc[-1]

    def analyze(self, symbol: str = None, parameters: Any = None, start: str = None, end: str = None,
                commissions: int | float = _DEFAULT_COMISSION, slippage: int | float = _DEFAULT_SLIPPAGE,
                low_ret_thresh: int = 85, high_ret_thresh: int = 98, low_vol_thresh: int = 2, moderate_vol_thresh: int = 16,
                high_vol_thresh: int = 35, scope_pct: float = 0.65, float_format: int = 6) -> None:
        """
        Prints an overall analysis of a ticker symbol.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        parameters: Any
            Strategy parameters.
        start: str
            Time of the first data point.
        end: str
            Time of the last data point.
        comissions: int | float
            Comissions for each trade.
        slippage: int | float
            Slippage of each trade.
        low_ret_thresh: int
            Lower returns threshold.
        high_ret_thresh: int
            High returns threshold.
        low_vol_thresh: int
            Lower volume threshold.
        moderate_vol_thresh: int
            Moderate volume threshold.
        high_vol_thresh: int
            High volume threshold.
        scope_pct: float
            Scope percentage.
        float_format: int
            Indicates the number of digits shown after the floating point.
        """

        symbol = self._default_symbol if symbol == None else symbol
        log_returns = self.log_changes(symbol).dropna()
        start = start if start != None else log_returns.index[0]
        end = end if end != None else log_returns.index[-1]
        parameters = parameters if parameters != None else self.optimized_parameters(symbol, start, end, low_ret_thresh, high_ret_thresh, low_vol_thresh,
                                                                                     moderate_vol_thresh, high_vol_thresh)
        strat_multiple = round(self.backtest(symbol, parameters, start, end,
                                             commissions, slippage), float_format)
        hodl_multiple = round(self.hodl(symbol), float_format)
        performance = round(strat_multiple - hodl_multiple, float_format)
        forwardtest = round(self.forwardtest(symbol, start, end, commissions, slippage, low_ret_thresh,
                                             high_ret_thresh, low_vol_thresh, moderate_vol_thresh, high_vol_thresh, scope_pct), float_format)
        cagr = round(float(self.cagr(symbol)), float_format)
        count = len(log_returns)
        years = (end - start).days / 365.25
        tp_year = count / years
        ann_mean = round(float(log_returns.mean()) * tp_year, float_format)
        ann_std = round(float(log_returns.std()) *
                        np.sqrt(tp_year), float_format)
        geo_mean = round(float(self.geo_mean(symbol)), float_format)
        skewness = round(float(self.skew(symbol)), float_format)
        kurtosis = round(float(self.kurtosis(symbol)), float_format)
        sharpe_cagr = \
            round(float(self.sharpe_ratio(symbol, metrics="CAGR")), float_format)
        sharpe_mean = \
            round(float(self.sharpe_ratio(symbol, metrics="Mean")), float_format)

        print("\nLong-Only Strategy")
        print(f"From '{start}' To '{end}'\n")
        print(103 * "=")
        print(
            f"Instrument:\t{symbol}\t|\tParameters:\t\t{parameters}\t|\tComissions:\t{commissions}")
        print(
            f"Slippage:\t{slippage}\t|\tForwardtest Scope:\t{scope_pct * 100}%\t\t|\n")
        print(103 * "-")
        print("Performance Measures:\n")
        print(f"Strategy Multiple:\t{strat_multiple}")
        print(f"Buy-and-Hold Multiple:\t{hodl_multiple}")
        print(33 * "-")
        print(f"Performance:\t\t{performance}")
        print(f"Forwardtest:\t\t{forwardtest}\n")
        print(f"CAGR:\t\t\t{cagr}")
        print(f"Skewness:\t\t{skewness}")
        print(f"Kurtosis:\t\t{kurtosis}")
        print(f"Annualized Mean:\t{ann_mean}")
        print(f"Annualized Std:\t\t{ann_std}")
        print(f"Geometric Mean:\t\t{geo_mean}")
        print(f"Sharp Ratio (CAGR):\t{sharpe_cagr}")
        print(f"Sharp Ratio (Mean):\t{sharpe_mean}")
        print(103 * "=")

    def mean_variance_analysis(self, context: str = _DEFAULT_CONTEXT) -> None:
        """
        Plots the Mean-Variance Analysis of all of the available symbols.

        Parameters
        ----------
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.
        """

        df = self.log_changes(self.symbols[0], context)
        if len(self.symbols) > 1:
            for symbol in self.symbols[1:]:
                df = df.join(self.log_changes(symbol, context))
        summary = df.agg(["mean", "std"]).T
        summary.plot(kind="scatter", x="std", y="mean",
                     figsize=(20, 8), fontsize=13)

        for i in summary.index:
            plt.annotate(i, xy=(summary.loc[i, "std"]+0.00005,
                                summary.loc[i, "mean"]+0.00005), size=15)

        plt.xlabel("Risk", fontsize=17)
        plt.ylabel("Reward", fontsize=17)
        plt.title("Mean-Variance Analysis", fontsize=20)

        plt.show()

    def returns_freq_distro(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> None:
        """
        Plots the Returns Frequency Distribution and Normal Distribution of the specified symbol.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.
        """

        log_returns = self.log_changes(symbol, context)
        mu = log_returns.mean()
        sigma = log_returns.std()

        x = np.linspace(log_returns.min(), log_returns.max(), 1000)
        y = stats.norm.pdf(x, loc=mu, scale=sigma)

        plt.figure(figsize=(20, 8))
        plt.hist(log_returns, density=True, bins=150,
                 label=f"Random Distribution")
        plt.plot(x, y, linewidth=3, color="red", label="Normal Distribution")
        plt.title("Frequency Distribution of Daily Returns", fontsize=20)
        plt.xlabel("Daily Returns", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.legend(fontsize=15)

        plt.show()

    def resampled_comparison(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> None:
        """
        Plots the Resampled Performance Comparison of the specified symbol.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.
        """

        log_returns = self.log_changes(symbol, context)

        m = [log_returns.resample("A").last().mean(),
             log_returns.resample("Q").last().mean(),
             log_returns.resample("M").last().mean(),
             log_returns.resample("W-Fri").last().mean(),
             log_returns.resample("D").last().mean()]

        d = [log_returns.resample("A").last().std(),
             log_returns.resample("Q").last().std(),
             log_returns.resample("M").last().std(),
             log_returns.resample("W-Fri").last().std(),
             log_returns.resample("D").last().std()]

        m = list(map(lambda x: float(x), m))
        d = list(map(lambda x: float(x), d))

        freqs = ["Annualy", "Quarterly", "Monthly", "Weekly", "Daily"]

        df = pd.DataFrame({"mean": m, "std": d}, index=freqs)

        df.plot(kind="scatter", x="std", y="mean",
                figsize=(30, 8), fontsize=15)

        for i in df.index:
            plt.annotate(i, xy=(df.loc[i, "std"]+0.00005,
                                df.loc[i, "mean"]+0.00005), size=15)

        plt.xlabel("Risk", fontsize=17)
        plt.ylabel("Reward", fontsize=17)
        plt.title("Resampled Performance Comparison", fontsize=20)

        plt.show()

    def sma(self, symbol: str = None, context: str = _DEFAULT_CONTEXT, window: int = 100) -> None:
        """
        Plots the Simple Moving Average based on a window.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.
        window: int
            Window for averaging data.
        """

        symbol = self._default_symbol if symbol == None else symbol
        data = self.__set_context(context)(symbol)
        data[data.columns[0]].plot(figsize=(20, 8), label=symbol)
        data[data.columns[0]].rolling(window).mean() \
            .plot(label="Moving Average", linewidth=2, color="red")

        plt.xlabel("Time", fontsize=17)
        plt.ylabel(symbol, fontsize=17)
        plt.title("Simple Moving Average", fontsize=20)
        plt.legend(fontsize=17)

        plt.show()

    def creturns_plot(self, symbol: str = None, context: str = _DEFAULT_CONTEXT) -> None:
        """
        Plots the Cumulative Financial Returns. 

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        context: str
            Data context which could be based on Price 'p' or Volume 'v'.
        """

        self.creturns(symbol, context).plot(figsize=(20, 8))

        plt.xlabel("Time", fontsize=17)
        plt.ylabel("Multiple", fontsize=17)
        plt.title("Cumulative Returns", fontsize=20)

        plt.show()

    def cat_heatmap(self, symbol: str = None) -> None:
        """
        Plots Categorical Heatmap.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        """

        idata = self.interpret(symbol)
        ct = pd.crosstab(idata["volChCategory"], idata["retCategory"])
        plt.figure(figsize=(20, 8))
        sns.set(font_scale=1)
        sns.heatmap(ct, cmap="RdYlBu_r", annot=True, robust=True, fmt=".0f")
        plt.xlabel("Returns Category", fontsize=15)
        plt.ylabel("Volume Category", fontsize=15)
        plt.title("Categorical Heatmap\n", fontsize=20)

        plt.show()

    def shifted_cat_heatmap(self, symbol: str = None) -> None:
        """
        Plots Shifted Categorical Heatmap.

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        """

        idata = self.interpret(symbol)
        ct = pd.crosstab(idata["volChCategory"].shift(),
                         idata["retCategory"].shift(),
                         values=idata["Returns"],
                         aggfunc=np.mean)
        plt.figure(figsize=(20, 8))
        sns.set(font_scale=1)
        sns.heatmap(ct, cmap="RdYlBu", annot=True, robust=True, fmt=".5f")
        plt.xlabel("Returns Category", fontsize=15)
        plt.ylabel("Volume Category", fontsize=15)
        plt.title("Categorical Heatmap\n", fontsize=20)

        plt.show()

    def strategy_parameters_plot(self, symbol: str = None, low_ret_thresh: int = 85, high_ret_thresh: int = 98,
                                 low_vol_thresh: int = 2, moderate_vol_thresh: int = 16, high_vol_thresh: int = 35) -> None:
        """
        Plots three subplots of Mean Performance of each possible strategy parameter. 

        Parameters
        ----------
        symbol: str
            Financial ticker symbol.
        low_ret_thresh: int
            Lower returns threshold.
        high_ret_thresh: int
            High returns threshold.
        low_vol_thresh: int
            Lower volume threshold.
        moderate_vol_thresh: int
            Moderate volume threshold.
        high_vol_thresh: int
            High volume threshold.
        """

        plist = list(product(list(range(low_ret_thresh, high_ret_thresh)),
                             list(range(low_vol_thresh, moderate_vol_thresh)),
                             list(range(moderate_vol_thresh, high_vol_thresh))))

        fig = plt.figure(figsize=(20, 8))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        dfdict = {"Returns_pct": [], "lowVol_pct": [],
                  "highVol_pct": [], "Performance": []}

        for parameters in plist:
            los = self.long_only_strategy(symbol, parameters)
            dfdict["Returns_pct"].append(parameters[0])
            dfdict["lowVol_pct"].append(parameters[1])
            dfdict["highVol_pct"].append(parameters[2])
            dfdict["Performance"].append(self.strategy_multiple(los))

        df = pd.DataFrame(dfdict)
        df.groupby("Returns_pct")["Performance"].mean() \
            .plot(ylabel="Mean Performance", xlabel="Returns Threshold Percentile",
                  ax=ax1, color="green", fontsize=12)
        df.groupby("lowVol_pct")["Performance"].mean() \
            .plot(ylabel="Mean Performance", xlabel="Lower Volume Threshold Percentile",
                  ax=ax2, color="red", fontsize=12, title=("Strategy Parameters Plot\n"))
        df.groupby("highVol_pct")["Performance"].mean() \
            .plot(ylabel="Mean Performance", xlabel="Higher Volume Threshold Percentile",
                  ax=ax3, fontsize=12)

        plt.show()

    def __set_context(self, context):
        if context.lower() in ("p", "price"):
            return self.miner.get_price
        elif context.lower() in ("v", "volume"):
            return self.miner.get_volume
        else:
            raise ValueError(
                "context could either be 'p' for Price or 'v' for Volume.")

    @staticmethod
    def discrete_compounding(ini_inv: int | float, rate: int | float, years: int = 1, comp_interval: int = 1) -> float:
        """
        Returns the discrete compounding of an investment.

        Parameters
        ----------
        ini_inv: int | float
            Initial investment.
        rate: int | float
            Rate of return.
        years: int
            Number of years.
        comp_interval: int
            Compunding interval.

        Returns
        -------
        float
        """

        return ini_inv * (1 + rate/comp_interval) ** (years * comp_interval)

    @staticmethod
    def continuous_compounding(ini_inv, rate, years=1) -> float:
        """
        Returns the continuous compounding of an investment.

        Parameters
        ----------
        ini_inv: int | float
            Initial investment.
        rate: int | float
            Rate of return.
        years: int
            Number of years.

        Returns
        -------
        float
        """

        return ini_inv * np.exp(years * rate)

    @staticmethod
    def quant_discretization(data: Any, quants: int = 10, labels: Any = np.arange(1, 11)) -> Any:
        """
        Returns the Quantile-based discretization of the specified data.

        Parameters
        ----------
        data: Any
            Initial investment.
        quants: int
            Number of Quantiles.
        labels: Any
            Labels used for specifying different categories.

        Returns
        -------
        Any
        """

        return pd.qcut(data, q=quants, labels=labels)

    @staticmethod
    def ptc(commissions: int | float, slippage: int | float) -> float:
        """
        Returns the Proportional Trading Cost.

        Parameters
        ----------
        comissions: int | float
            Comissions for each trade.
        slippage: int | float
            Slippage of each trade.

        Returns
        -------
        float
        """

        return np.log(1-commissions) + np.log(1-slippage)

    @staticmethod
    def strategy_multiple(strat_data: pd.DataFrame) -> float:
        """
        Returns the net strategy multiple.

        Parameters
        ----------
        strat_data: pandas.DataFrame
            Strategy data.

        Returns
        -------
        float
        """

        return float(np.exp(strat_data["stratNet"].sum()))
