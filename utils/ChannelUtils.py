"""
Autor: Joao Gabriel Paina

This module implements several useful DWDM channel operations based on ITU-T G.694.1. 
Some of the main operations include channel conversion from THz to flex grid index, 
as well as power conversion.
"""
from scipy.constants import speed_of_light as co
import numpy as np 
from typing import Union
from time import sleep
from collections.abc import Iterable

class ChannelTool():
    
    _ANCHOR_FREQ = 193.1
    _FLEX_GRID = 0.0125
    _FLEX_GRID_DF = 0.00625
    _FIXED_GRID = {'12.5': 0.0125, 
                  '25.0': 0.025, 
                  '50.0': 0.05, 
                  '100.0': 0.1}
    _FACTOR = {'DF_GHZ': co,      #DL in nm, Lc in nm [DL/Lc**2]
              'DF_THZ': co/1E3,  #DL in nm, Lc in nm [DL/Lc**2]
              'DL_NM': 1/co,     #DF in GHz, Lc in nm [DF*Lc**2]
              'FL_GHZ': co,      #F in GHz, L in nm [1/F OR 1/L]
              'FL_THZ': co/1E3}  #F in THz, L in nm [1/F OR 1/L]
    
    @classmethod
    def from_dbm_to_watt(cls, power: Union[float, np.ndarray[float]]) ->\
                        np.ndarray[float]:
        """
        Converts from dBm to Watt

        Parameters
        ----------
        power : float | np.ndarray[float]
            Power in dBm

        Returns
        -------
        np.ndarray[float]
            Power in Watt.
        """
        return np.pow(10, power/10-3)
    
    @classmethod
    def from_watt_to_dbm(cls, power: Union[float, np.ndarray[float]]) ->\
                        np.ndarray[float]:
        """
        Converts from Watt to dBm

        Parameters
        ----------
        power : float | np.ndarray[float]
            Power in Watt
            
        Returns
        -------
        np.ndarray[float]
            Power in dBm.
        """
        return np.array(10*np.log10(power*1e3))
    
    @classmethod 
    def from_dlamb_to_dfreq(cls, delta: float, wlc: Union[float, 
                           np.ndarray[float], list[float]]) ->\
                           np.ndarray[float]:
        """
        Converts bandwidth from nm to THz
        
        Parameters
        ----------
        delta : float
            Bandwidth in nm.
        wlc : float | np.ndarray[float] | list[float]
            Reference wavelength. 
            
        Returns
        -------
        np.ndarray[float]
            Bandwidth in THz.
        """
        return np.array(cls._FACTOR['DF_THZ']*(delta/(np.prod(wlc)**\
               ((np.size(wlc) == 1) + 1))))
    
    @classmethod
    def unitConverter(cls, Carrier: Union[float, list[float], 
                     np.ndarray[float]]) -> np.ndarray[float]:
        """
        Converts frequency from THz to nm and vice versa.
        
        Parameters
        ----------
        Carrier : float | np.ndarray[float] | list[float]
            Carrier in nm or THz.

        Returns
        -------
        np.ndarray[float]
            THz if receive nm or nm if receive THz.
        """
        return np.array((1/np.array(Carrier))*cls._FACTOR['FL_THZ'])
    
    @classmethod
    def getIndex(cls, Carrier: Union[float, np.ndarray[float], list[float]], 
                unit: bool = True) -> np.ndarray[int]:
        """
        Calculates the nearest flex grid central frequency index "n".
        
        Parameters
        ----------
        Carrier : float | np.ndarray[float] | list[float]
            Channel in nm or THz.
        unit : bool, optional
            If True, then Carrier must be in THz. 
            Else, then Carrier must be in nm
            The default is True.

        Returns
        -------
        np.ndarray[int]
            Flex grid Index.
        """
        fc = (cls.unitConverter(Carrier))*int(not unit) +\
             np.array(Carrier)*int(unit)
        n = np.round((fc-cls._ANCHOR_FREQ)/cls._FLEX_GRID_DF).astype(np.int64)
        return n
    
    @classmethod 
    def getBandWidthIndex(cls, Bw: Union[float, np.ndarray[float], 
                         list[float]]) -> np.ndarray[int]:
        """
        Calculates the ceiling flex grid bandwidth frequency index "m".
        
        Parameters
        ----------
        Bw : float | np.ndarray[float] | list[float]
            Bandwidth in GHz.

        Returns
        -------
        np.ndarray[int]
            Flex grid Bandwidth Index.
        """
        m = np.ceil(1e-3*np.array(Bw)/cls._FLEX_GRID).astype(np.int64)
        return m
    
    @classmethod
    def getFlexGridWavelength(cls, n: Union[int, np.ndarray[int], 
                             list[int]]) -> np.ndarray[float]:
        """
        Converts flex grid central frenquency index "n" to wavelength in nm.
        
        Parameters
        ----------
        n : int | np.ndarray[int]] | list[int]
            Array or single Int value flex grid Index.

        Returns
        -------
        np.ndarray[float]
            Wavelength in nm.
        """
        return np.array(1/(cls._ANCHOR_FREQ + np.array(n)*cls._FLEX_GRID_DF)*\
               cls._FACTOR['FL_THZ'])
    
    @classmethod
    def getFlexGridFrequency(cls, n: Union[int, np.ndarray[int], 
                            list[int]]) -> np.ndarray[float]:
        """
        Converts flex grid central frenquency index "n" to frequency in THz.
        
        Parameters
        ----------
        n : int | np.ndarray[int] | list[int]
            Array or single Int value flex grid Index.

        Returns
        -------
        np.ndarray[float]
            Frequency in THz.
        """
        return np.round(cls._ANCHOR_FREQ + np.array(n)*cls._FLEX_GRID_DF,5)
    
    @classmethod
    def getFlexGridBundle(cls, n: Union[list[int], np.ndarray[int]], 
                          m: Union[int, list[int], np.ndarray[int]]) -> \
                          np.ndarray[np.ndarray[int]]:
        """
        Creates channel array like: array([[n0, m0], [n1, m1], ...])
        
        Parameters
        ----------
        n : list[int] | np.ndarray[int]
            Int Array/List of flex grid index.
        m : int | list[int] | np.ndarray[int]
            Flex channels bandwidth (multiple of 12.5GHz).

        Returns
        -------
        np.ndarray[np.ndarray[int]]
            Argument for WSS channel creation.
        """
        check = lambda x: isinstance(x, Union[list, np.ndarray])
        return np.array(list(zip(n,m))) if check(m) else\
               np.array(list(zip(n,[m]*len(n))))
    
    
    @classmethod
    def getFlexGrid(cls, channel_spacing: int, center: float, num_ch: int, 
                    thz: bool =True) -> np.ndarray[int]:
        """
        It creates a frenquency comb centered at the center frequency.
        
        Parameters
        ----------
        channel_spacing : int
            Integer multiple of 12.5GHz.
        center : float
            Central Wavelength (nm) or Frequency (THz).
        num_ch : int
            Number of Channels around the center.
        thz : bool, optional
            If True, then center must be in THz.
            Else, then center must be in nm. The default is False.

        Returns
        -------
        np.ndarray[int]
            Flex grid index (n) of each channel.


        Example:
        --------
        >>> ChannelTool.getFlexGrid(8, 193.8, 4)
        array([ 80,  96, 112, 128])
        """
        Dn = 2*channel_spacing
        Fc = (1/center)*cls._FACTOR['FL_THZ']*int(not(thz)) + center*int(thz)
        ni = cls.getIndex(Fc)
        neg = np.arange(ni, ni-(num_ch//2+1)*Dn, -Dn)[::-1]
        pos = np.arange(ni+Dn, ni+(num_ch//2+num_ch%2)*Dn, Dn)
        return np.array(list(neg) + list(pos))
    
    @classmethod
    def getStartStopWl(cls, n: Union[int, np.ndarray[int], list[int]], 
                      m: int) -> tuple[Union[np.ndarray[float], float], 
                      Union[np.ndarray[float], float]]:
        """
        Calculates the start and stop wavelength given a central frequency (n)
        and bandwidth (m).
        
        Parameters
        ----------
        n : int | np.ndarray[int] | list[int]
            Flex grid Index.
        m : int
            Integer multiple of 12.5GHz.

        Returns
        -------
        tuple[np.ndarray[float] | float, np.ndarray[float] | float]
            Start and Stop wavelength of the channel based on the bandwidth 
            m*12.5GHz.
        """
        wlc = cls.getFlexGridWavelength(n)
        df = cls._FLEX_GRID*m
        dl = df*wlc**2*cls._FACTOR['DL_NM']*1e3
        return wlc-dl/2, wlc+dl/2
    
    @classmethod
    def getPeaks(cls, n: Union[int, np.ndarray[int], list[int]], 
                m: Union[int, np.ndarray[int], list[int]], 
                x: np.ndarray[float], y: np.ndarray[float])\
                -> np.ndarray[float]:
        """
        Calculates power peak from a spectrum given the used flex channels.
        
        Parameters
        ----------
        n : int | np.ndarray[int] | list[in]
            Array of flex grid Index.
        m : int | np.ndarray[int] | list[in]
            Integer multiple of 12.5GHz.
        x : np.ndarray[float]
            Wavelength spectrum array in nm
        y : np.ndarray[float]
            Power spectrum array in dBm.

        Returns
        -------
        np.ndarray[float]
            Channels peaks in dBm.
        """
        ylin = 10**(y/10)*1e-3
        pwpk = list()
        n = (lambda _n: np.array([n]) if np.array(n).size==1 else 
             np.array(n))(n)
        m = (lambda _m: np.array(_m) if isinstance(_m, Iterable) 
             else np.array([_m]*n.size))(m)
        for _n, _m in zip(n, m):
            wli, wlf = cls.getStartStopWl(_n, _m)
            Filter = np.less_equal(x, wlf)*np.greater_equal(x, wli)
            pwpk.append(10*np.log10(max(Filter*ylin)*1e3))
        return np.array(pwpk)
