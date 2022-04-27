import datetime as dt
import json
import warnings
import bls
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import wrds
from fredapi import Fred
from pandas.tseries.offsets import MonthEnd
from statsmodels.regression.rolling import RollingOLS


class FactorData(object):
    """
    This class downloads all relevant data and calculates 102 firm
    characteristics at monthly frequency. In the current version,
    variable definitions follow those of Green et al. 2017.

    We appreciate that there may be deviating variable definitions such those
    used by Hou et al. 2020, but leave those for future implementations.

    Parameters
    ----------
    wrds_username: str
        This is your username, e.g. 'janedoe'.
        Note that you must have setup pgpass file before. If you have not:
        https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/
        programming-python/python-from-your-computer/

    bls_key: str
        This is your BLS API key.
        Visit https://data.bls.gov/registrationEngine/ if you haven't got an
        API yet.

    fred_key: str
        This is your FRED API key.
        Visit https://research.stlouisfed.org/docs/api/api_key.html
        if you haven't got an API yet.

    start_yr: int, default=1980
        Start year for effective date, where effective date takes all lags
        of delayed information into account (see lag_annual, lag_quarterly).
        Note that any start_yr < 1950 is set to 1950 due to limited data
        availability.

    end_yr: int or None, default=None
        End year for effective date ending in December. If None all data until
        the most current observation is downloaded. We advise to use None,
        however, one may want to specify end_year to replicate papers.

    linkprim: str or list, default=['P', 'C']
        Primary Link Marker - we strongly advice to use P, C only. Linkprim is
        relevant for Compustat data download.

    linktype: str or list, default=['LU', 'LC', 'LS']
        LC	Link research complete. Standard connection between databases.
        LU	Unresearched link to issue by CRSP.
        LX	Link to a security that trades on another exchange system not
            included in CRSP data.
        LD	Duplicate link to a security. Another GVKEY/IID is a better link to
            that CRSP record.
        LS	Link valid for this security only. Other CRSP PERMNOs with the same
            PERMCO will link to other GVKEYs.
        LN	Primary link exists but Compustat does not have prices.
        LO	No link on issue level but company level link exists. Example
            includes Pre-FASB, Subsidiary, Consolidated, Combined, Pre-amend,
            Pro-Forma, or "-old".
        NR	No link available. Confirmed by research.
        NU	No link available, not yet confirmed.

        CAUTION: depending on choice of linktype one has to deal with
                 duplicates!

    lag_annual: int, default=6
        Specifies number of months after which annual information becomes
        available. Avoids information leakage as Compustat dates data back to
        its reference pit and not when the data became available.

    exchcd: int or list, default=[1, 2, 3]
        Exchange Code:
        -2	Halted by the NYSE or AMEX
        -1	Suspended by the NYSE, AMEX, or NASDAQ
         0	Not Trading on NYSE, AMEX, or NASDAQ
         1	New York Stock Exchange
         2	American Stock Exchange
         3	The Nasdaq Stock Market(SM)
         4	The Arca Stock Market(SM)
         5	Mutual Funds (As Quoted by NASDAQ)
        10	Boston Stock Exchange
        13	Chicago Stock Exchange
        16	Pacific Stock Exchange
        17	Philadelphia Stock Exchange
        19	Toronto Stock Exchange
        20	Over-The-Counter (Non-NASDAQ Dealer Quotations)
        31	When-issued trading on the NYSE
        32	When-issued trading on the AMEX
        33	When-issued trading on The NASDAQ

    indfmt: str, default='INDL'
        text

    datafmt: str, default='STD'
        data format for Compustat download

    consol: str, default='C'
        console for Compustat download

    curcd: str, default='USD'
        currency for download

    Returns
    ----------
    Factor data class
    """

    def __init__(self,
                 wrds_username,
                 bls_key,
                 fred_key,
                 start_yr=1980,
                 end_yr=None,
                 linkprim=('P', 'C'),
                 linktype=('LU', 'LC', 'LS'),
                 lag_annual=6,
                 lag_quarter=4,
                 exchcd=(1, 2, 3),
                 shrcd=(10, 11),
                 indfmt='INDL',
                 datafmt='STD',
                 popsrc='D',
                 consol='C',
                 curcd='USD'):

        self.wrds_username = wrds_username
        self.bls_key = bls_key
        self.fred_key = fred_key

        if start_yr < 1950:
            self.start_yr = 1950
            warnings.warn('start_yr set to < 1950. Replaced with 1950')
        else:
            self.start_yr = start_yr

        if end_yr is None:
            self.end_yr = dt.datetime.now().year
        else:
            if end_yr < start_yr:
                raise ValueError('end_yr must be larger than start_yr!')
            self.end_yr = end_yr

        if not isinstance(linkprim, list):
            linkprim = list(linkprim)
        valid_lp = ['J', 'C', 'N', 'P']

        if not all(valid in valid_lp for valid in linkprim):
            raise ValueError('linkprim must be in ["J", "C", "N", "P"]. '
                             'Got {}'.format(linkprim))
        self.linkprim = linkprim

        if not isinstance(linktype, list):
            linktype = list(linktype)

        valid_lt = ['LU', 'NU', 'NR', 'LC', 'LX', 'LN', 'LD', 'LS', 'LO']

        if not all(valid in valid_lt for valid in linktype):
            raise ValueError('linkprim must be in {}. '.format(
                valid_lt) + 'Got {}'.format(linktype))

        self.linktype = list(linktype)

        if not isinstance(lag_annual, int):
            lag_annual = int(lag_annual)

        self.lag_annual = lag_annual
        self.lag_quarter = lag_quarter
        self.exchcd = list(exchcd)
        self.shrcd = list(shrcd)
        self.indfmt = indfmt
        self.datafmt = datafmt
        self.popsrc = popsrc
        self.consol = consol
        self.curcd = curcd
        # convenient for later
        self.chars_list = None
        self.chars_data = None
        self.chars_data_clean = None
        self.factors = None

    @staticmethod
    def _std_resid(idx, model):
        window = model.__dict__['model'].__dict__['_window']
        if idx >= window:
            loc = slice(idx - window, idx)
        else:
            loc = slice(idx)
        wx = model.__dict__['model'].__dict__['_wx'][loc]
        wy = model.__dict__['model'].__dict__['_wy'][loc]
        params = model.__dict__['_params'][idx]
        resid = wy - wx @ params.T
        sd = np.std(resid)
        return sd

    def _rolling_betas(self, data, perm):
        dat = data[data.permno == perm].copy()
        obs = dat.shape[0]
        if obs < 156:
            model = RollingOLS(dat['wkret'],
                               sm.add_constant(dat['ewret'],
                                               has_constant='add'),
                               window=obs, min_nobs=52, missing='drop',
                               expanding=True).fit()
            r2_2 = RollingOLS(dat['wkret'],
                              sm.add_constant(
                                  dat[['ewret', 'ewmkt_l1', 'ewmkt_l2',
                                       'ewmkt_l3', 'ewmkt_l4']],
                                  has_constant='add'),
                              window=obs, min_nobs=52, missing='drop',
                              expanding=True).fit().rsquared_adj
        else:
            model = RollingOLS(dat['wkret'],
                               sm.add_constant(dat['ewret'],
                                               has_constant='add'),
                               window=156, min_nobs=52, missing='drop',
                               expanding=True).fit()
            r2_2 = RollingOLS(dat['wkret'],
                              sm.add_constant(
                                  dat[['ewret', 'ewmkt_l1', 'ewmkt_l2',
                                       'ewmkt_l3', 'ewmkt_l4']],
                                  has_constant='add'),
                              window=156, min_nobs=52, missing='drop',
                              expanding=True).fit().rsquared_adj

        out = model.params.rename(columns={'ewret': 'beta'})
        out['betasq'] = out['beta'] ** 2
        out['r2'] = model.rsquared_adj
        out['r2_2'] = r2_2
        out['pricedelay'] = 1 - (out['r2'] / out['r2_2'])
        out['idiovol'] = np.array([self._std_resid(idx=x, model=model)
                                   for x in range(out.shape[0])])
        return out[['beta', 'betasq', 'pricedelay', 'idiovol']]

    @staticmethod
    def _orgcap(data, perm):
        cols = ['jdate', 'permno', 'fyear', 'orgcap_1', 'xsga', 'cpi',
                'avgat']
        dat = data[data.permno == perm][cols]
        dat.drop_duplicates(subset=['fyear', 'permno'], inplace=True)
        n = dat.shape[0]
        for ix in range(1, n):
            dat.iloc[ix, 3] = dat.iloc[ix - 1, 3] * (1 - .15) + \
                              dat.iloc[ix, 4] / dat.iloc[ix, 5]

        dat['orgcap'] = dat['orgcap_1'] / dat['avgat']
        dat.drop(labels=['orgcap_1', 'xsga', 'cpi', 'avgat'], axis=1,
                 inplace=True)
        return dat

    def get_data(self, wh=0.99, wl=0.01):
        """
        Downloads data from all data sources.
        winsorizes
        :return:
        """
        # download cpi data first.
        print('1. Loading CPI data from BLS')
        cpi_u = bls.get_series('CUUR0000SA0', self.start_yr - 7, self.end_yr,
                               key=self.bls_key)
        cpi_u = cpi_u.groupby(cpi_u.index.year).mean()
        cpi_u = cpi_u.reset_index()
        cpi_u = list(zip(*map(cpi_u.get, cpi_u)))
        # connect to wrds
        print('2. Loading firm data from WRDS')
        wrds_db = wrds.Connection(wrds_username=self.wrds_username)
        data = wrds_db.raw_sql(
            f"""

            /* CPI data: cpi */

            WITH cpi (year, cpi) AS(
                (VALUES {str(cpi_u).strip('[]')})
            ),

            /* Annual raw data from Compustat: compa */

            compa AS(
            SELECT CAST(f.gvkey as INT), f.cusip, f.datadate::date, 
            CAST(f.fyear as INT), c.cik, substr(c.sic,1,2) as sic2, c.naics, 
            f.sale, f.revt, f.cogs, f.xsga, f.dp, f.xrd, f.xad, f.ib, f.ebitda, 
            f.ebit, f.nopi, f.spi, f.pi, f.ni, f.txfed, f.txfo, f.txt, 
            CAST(c.sic as INT), f.xint, f.txp, f.oancf, f.dvt, f.ob, f.gdwlia, 
            f.gdwlip, f.gwo, f.mib, f.oiadp, f.ivao, f.rect, f.che, f.ppegt, 
            f.invt, f.at, f.aco, f.intan, f.ao, f.ppent, f.gdwl, f.fatb, 
            f.fatl, f.dlc, f.dltt, f.lt, f.dm, f.dcvt, f.cshrc, f.dcpstk, 
            f.pstk, f.ap, f.lco, f.lo, f.drc, f.drlt, f.txdi, f.ceq, f.scstkc, 
            f.emp, f.csho, f.seq, f.txditc, f.pstkrv, f.pstkl, f.np, f.dpc,
            f.txdc,  f.ajex, ABS(f.prcc_f) as prcc_f, 
            ABS(f.csho)*ABS(f.prcc_f) AS mve_f,

            CASE WHEN f.capx is null AND COUNT(*) OVER w >=2 
                THEN f.ppent - (LAG(f.ppent) OVER w)
                ELSE f.capx
                END as capx,

            CASE WHEN f.act is null 
                THEN f.che+f.rect+f.invt
                ELSE f.act
                END as act,

            CASE WHEN f.lct is null THEN f.ap
                 ELSE f.lct 
                 END as lct,

            CASE WHEN f.drc is not null AND f.drlt is not null 
                THEN f.drc+f.drlt
                WHEN f.drc is not null and f.drlt is null 
                THEN f.drc 
                WHEN f.drlt is not null and f.drc is null 
                THEN f.drlt
                END as dr,

            CASE WHEN f.dcvt is null AND f.dcpstk is not null AND 
                    f.pstk is not null AND f.dcpstk > f.pstk
                THEN f.dcpstk-f.pstk
                WHEN f.dcvt is null AND f.dcpstk is not null AND f.pstk is null
                THEN f.dcpstk
                WHEN f.dc is null 
                THEN f.dcvt
                END as dc,

            f.xrd/NULLIF((LAG(f.at) OVER w), 0) as xrd_h,

            CASE WHEN f.fyear <=1978 
                THEN 0.48
                WHEN f.fyear BETWEEN 1979 AND 1986 
                THEN 0.46
                WHEN f.fyear=1987 
                THEN 0.4
                WHEN f.fyear BETWEEN 1988 AND 1992 
                THEN 0.34
                ELSE 0.35
                END as tr,

            CASE WHEN xint is null 
                THEN 0
                ELSE xint
                END AS xint0, 

            CASE WHEN xsga is null 
                THEN 0 
                ELSE xsga
                END AS xsga0,

            CASE WHEN f.ni > 0 
                THEN 1 
                ELSE 0 
                END as ps0,

            CASE WHEN f.oancf > 0 
                THEN 1 ELSE 0 
                END as ps1, 

            CASE WHEN 
                (f.ni/NULLIF(f.at, 0)) > 
                ((LAG(f.ni) OVER w)/NULLIF((LAG(f.at) OVER w), 0)) 
                THEN 1 ELSE 0 
                END as ps2,

            CASE WHEN f.oancf > f.ni 
                THEN 1 
                ELSE 0 
                END as ps3, 

            CASE WHEN 
                (f.dltt/NULLIF(f.at, 0)) > 
                ((LAG(f.dltt) OVER w)/NULLIF((LAG(f.at) OVER w), 0)) 
                THEN 1 
                ELSE 0 
                END as ps4,

            CASE WHEN 
                (f.act/NULLIF(f.lct, 0)) > 
                ((LAG(f.act) OVER w)/NULLIF((LAG(f.lct) OVER w), 0)) 
                THEN 1 
                ELSE 0 
                END as ps5,

            CASE WHEN 
                ((f.sale-f.cogs)/NULLIF(f.sale, 0)) > 
                (((LAG(f.sale) OVER w)-(LAG(f.cogs) OVER w))/
                   NULLIF((LAG(f.sale) OVER w), 0)) 
                THEN 1 
                ELSE 0 
                END as ps6,

            CASE WHEN 
                (f.sale/NULLIF(f.at, 0)) > 
                ((LAG(f.sale) OVER w)/NULLIF((LAG(f.at) OVER w), 0)) 
                THEN 1 
                ELSE 0 
                END as ps7,

            CASE WHEN f.scstkc=0 
                THEN 1 
                ELSE 0 
                END as ps8, 

            (date_trunc('month', f.datadate::date) + interval '1 month'
            *{self.lag_annual + 2} - interval '1 day')::date AS jdate

            FROM comp.funda as f 
            LEFT JOIN comp.company as c 
            ON f.gvkey = c.gvkey

            WHERE f.indfmt = '{self.indfmt}' 
            AND f.datafmt = '{self.datafmt}' 
            AND f.popsrc = '{self.popsrc}'
            AND f.consol = '{self.consol}'
            AND f.curcd = '{self.curcd}'
            AND extract(year from f.datadate) >= {self.start_yr - 5}

            WINDOW w AS (PARTITION BY f.gvkey ORDER BY f.datadate)
            ORDER BY f.gvkey, f.datadate
            ),

            /* Linktable: ccm */

            ccm AS(
            SELECT 
            CAST(gvkey as INT), 
            CAST(lpermno as INT) as permno, 
            CAST(lpermco AS INT) as permco, 
            linkprim, linktype, linkdt::date, 
            coalesce(linkenddt, current_date)::date as linkenddt

            FROM crsp.ccmxpf_linktable 
            WHERE linkprim IN {*self.linkprim,} 
            AND linktype IN {*self.linktype,} 
            AND ({self.end_yr} >= extract(year from linkdt) 
            OR linkdt is null)
            AND ({self.start_yr} <= extract(year from linkenddt) 
            OR linkenddt is null)
            AND lpermco is not null
            AND lpermno is not null
            ), 

            /* Annual data: data_a */

            data_a AS(
            SELECT 
            data.jdate, 
            CASE WHEN LEAD(data.jdate) OVER w1 is null
                THEN 
                    data.jdate + interval '1 year'
                ELSE 
                    LEAD(data.jdate) OVER w1
                END 
                    as jdate_end,
            data.fyear, data.sic2,
            data.gvkey, data.permno, data.roa, data.cfroa, data.oancf, 
            data.ni, data.xrdint, data.capxint, data.xadint,
            data.absacc, data.acc, data.age, data.agr, data.bm, data.cashdebt,
            data.cashpr, data.cfp, data.chato, data.chcsho, data.chinv, 
            data.chpm, data.convind, 
            CASE WHEN data.splticrm='D' THEN 1
                WHEN data.splticrm='C' THEN 2
                WHEN data.splticrm='CC' THEN 3
                WHEN data.splticrm='CCC-' THEN 4
                WHEN data.splticrm='CCC' THEN 5
                WHEN data.splticrm='CCC+' THEN 6
                WHEN data.splticrm='B-' THEN 7
                WHEN data.splticrm='B' THEN 8
                WHEN data.splticrm='B+' THEN 9
                WHEN data.splticrm='BB-' THEN 10
                WHEN data.splticrm='BB' THEN 11
                WHEN data.splticrm='BB+' THEN 12
                WHEN data.splticrm='BBB-' THEN 13
                WHEN data.splticrm='BBB' THEN 14
                WHEN data.splticrm='BBB+' THEN 15
                WHEN data.splticrm='A-' THEN 16
                WHEN data.splticrm='A' THEN 17
                WHEN data.splticrm='A+' THEN 18
                WHEN data.splticrm='AA-' THEN 19
                WHEN data.splticrm='AA' THEN 20
                WHEN data.splticrm='AA+' THEN 21
                WHEN data.splticrm='AAA' THEN 22
                WHEN data.splticrm is null THEN 0
            END AS credrat, 
            data.currat, data.depr, data.divi, data.divo, data.dy, data.egr, 
            data.ep, data.gma, data.grcapx, data.grltnoa, data.hire, 
            data.invest, data.tb_1, data.mve_f, data.lev, data.lgr, 
            data.operprof, data.cpi, data.xsga, data.pchcapx, data.avgat, 
            data.pchcurrat, data.pchdepr, data.pchgm_pchsale, data.pchquick, 
            data.pchsale_pchinvt, data.pchsale_pchrect, data.pchsale_pchxsga, 
            data.pchsaleinv, data.pctacc, data.ps, data.quick, data.rd, 
            data.rd_mve, data.rd_sale, data.realestate, data.roic, data.sale,
            data.salecash, data.saleinv, data.salerec, data.secured, 
            data.securedind, data.sgr, data.sin, data.sp, data.tang

            FROM(
            SELECT 
            CAST(b.permno as INT), 
            CAST(b.permco as INT), 
            b.linkdt::date, b.linkenddt::date, b.linktype, 
            b.linkprim, a.sic, a.sic2, a.gvkey, a.cusip, a.jdate, a.datadate,
            a.sale, a.mve_f, a.ni, a.oancf, 
            CAST(a.fyear as INT), 
            COUNT(*) OVER w as age,

            a.ceq / NULLIF(a.mve_f, 0) as bm,  
            a.ib / NULLIF(a.mve_f, 0) as ep, 
            (mve_f+dltt-at)/ NULLIF(a.che, 0) as cashpr, 
            a.dvt / NULLIF(a.mve_f, 0) as dy, 
            a.lt /NULLIF(a.mve_f, 0) as lev, 
            a.sale /NULLIF(a.mve_f, 0) as sp, 
            (a.ebit-a.nopi) / NULLIF((a.ceq+a.lt-a.che), 0) as roic,  
            a.xrd / NULLIF(a.sale, 0) as rd_sale, 
            a.xrd / NULLIF(a.mve_f, 0) as rd_mve, 
            (a.at/ NULLIF(LAG(a.at) OVER w, 0))-1 AS agr,
            (a.csho/ NULLIF(LAG(a.csho) OVER w, 0))-1 AS chcsho,
            ((a.revt-a.cogs)/ NULLIF(LAG(a.at) OVER w, 0)) as gma,
            (a.lt/ NULLIF(LAG(a.lt) OVER w, 0))-1 as lgr,

            CASE WHEN a.oancf is null 
                THEN 
                    (   ( (a.act-(LAG(a.act) OVER w))-
                          (a.che-(LAG(a.che) OVER w)) )-
                        ( (a.lct-(LAG(a.lct) OVER w))-
                          (a.dlc-(LAG(a.dlc) OVER w))-
                          (a.txp-(LAG(a.txp) OVER w))-
                          (a.dp                     ) ) )/ 
                    NULLIF(  (a.at+LAG(a.at) OVER w)/2, 0)
                ELSE 
                    (a.ib-a.oancf) / NULLIF((a.at+LAG(a.at) OVER w)/2, 0)
                END as acc,

            CASE WHEN a.oancf is null 
                THEN  
                    ABS((   ( (a.act-(LAG(a.act) OVER w))-
                          (a.che-(LAG(a.che) OVER w)) )-
                        ( (a.lct-(LAG(a.lct) OVER w))-
                          (a.dlc-(LAG(a.dlc) OVER w))-
                          (a.txp-(LAG(a.txp) OVER w))-
                          (a.dp                     ) ) )/ 
                    NULLIF(  (a.at+LAG(a.at) OVER w)/2, 0))
                ELSE 
                    ABS((a.ib-a.oancf) / NULLIF((a.at+LAG(a.at) OVER w)/2, 0))
                END as absacc,

            CASE WHEN a.ib = 0 
                THEN 
                    (a.ib-a.oancf)/0.01
                WHEN a.oancf is null 
                THEN
                    (((a.act-(LAG(a.act) OVER w))-
                     (a.che-(LAG(a.che) OVER w)))-
                    ((a.lct-(LAG(a.lct) OVER w))-
                     (a.dlc-(LAG(a.dlc) OVER w))-
                     (a.txp-(LAG(a.txp) OVER w))-
                     (a.dp)))/(ABS(a.ib))
                WHEN a.oancf is null and a.ib = 0 
                THEN
                    (((a.act-(LAG(a.act) OVER w))-
                     (a.che-(LAG(a.che) OVER w)))-
                    ((a.lct-(LAG(a.lct) OVER w))-
                     (a.dlc-(LAG(a.dlc) OVER w))-
                     (a.txp-(LAG(a.txp) OVER w))-
                     (a.dp)))/(0.01) 
                ELSE 
                    (a.ib-a.oancf)/(ABS(a.ib))
                END as pctacc, 
            CASE WHEN a.oancf is not null 
                THEN 
                    a.oancf/NULLIF(a.mve_f, 0)
                ELSE
                    (a.ib-(((a.act-(LAG(a.act) OVER w))-
                    (a.che-(LAG(a.che) OVER w)))-
                    ((a.lct-(LAG(a.lct) OVER w))-
                    (a.dlc-(LAG(a.dlc) OVER w))-
                    (a.txp-(LAG(a.txp) OVER w))-
                    (a.dp))))/NULLIF(a.mve_f, 0)
                END as cfp,
            (a.invt-(LAG(a.invt) OVER w))/
            NULLIF((a.at+LAG(a.at) OVER w)/2, 0) as chinv,
            CASE WHEN a.oancf is not null 
                THEN 
                    a.oancf/((a.at+NULLIF(LAG(a.at) OVER w, 0))/2)
                ELSE
                    (a.ib-(((a.act-(LAG(a.act) OVER w))-
                    (a.che-(LAG(a.che) OVER w)))-
                    ((a.lct-(LAG(a.lct) OVER w))-
                    (a.dlc-(LAG(a.dlc) OVER w))-
                    (a.txp-(LAG(a.txp) OVER w))-
                    (a.dp))))/((a.at+NULLIF(LAG(a.at) OVER w, 0))/2)
                END as cf,
            CASE WHEN a.emp is null OR (LAG(a.emp) OVER w) is null 
                THEN 0
                ELSE
                    (a.emp-(LAG(a.emp) OVER w))/NULLIF(LAG(a.emp) OVER w, 0)
                END as hire,
            (a.sale/ NULLIF(LAG(a.sale) OVER w, 0))-1 as sgr,
            (a.ib/NULLIF(a.sale, 0))-((LAG(a.ib) OVER w) / 
            (NULLIF(LAG(a.sale) OVER w, 0))) as chpm,
            (a.sale / (NULLIF(a.at + (LAG(a.at) OVER w), 0) / 2))-
            ((LAG(a.sale) OVER w)/ (NULLIF((LAG(a.at) OVER w)+
             (LAG(a.at, 2) OVER w), 0) / 2)) as chato,
            ((a.sale-(LAG(a.sale) OVER w))/NULLIF(LAG(a.sale) OVER w, 0))-
            ((a.invt-(LAG(a.invt) OVER w))/NULLIF(LAG(a.invt) OVER w, 0)) 
            as pchsale_pchinvt,
            ((a.sale-(LAG(a.sale) OVER w))/NULLIF(LAG(a.sale) OVER w, 0))-
            ((a.rect-(LAG(a.rect) OVER w))/NULLIF(LAG(a.rect) OVER w, 0)) 
            as pchsale_pchrect,
            (((a.sale-a.cogs)-((LAG(a.sale) OVER w)-(LAG(a.cogs) OVER w)))/
             (NULLIF((LAG(a.sale) OVER w)-(LAG(a.cogs) OVER w), 0)))-
            ((a.sale-(LAG(a.sale) OVER w))/NULLIF(LAG(a.sale) OVER w, 0)) 
            as pchgm_pchsale,
            ((a.sale-(LAG(a.sale) OVER w))/NULLIF(LAG(a.sale) OVER w, 0))-
            ((a.xsga-(LAG(a.xsga) OVER w))/NULLIF(LAG(a.xsga) OVER w, 0)) 
            as pchsale_pchxsga,
            a.dp/NULLIF(a.ppent, 0) as depr,
            ( (a.dp/ NULLIF(a.ppent, 0))-
              ((LAG(a.dp) OVER w) / NULLIF(LAG(a.ppent) OVER w, 0)))/
            (( NULLIF(LAG(a.dp) OVER w, 0)  / NULLIF(LAG(a.ppent) OVER w, 0))) 
            as pchdepr,
            LN(1+a.xad) - LN(1+ (LAG(a.xad) OVER w) ) as chadv, 
            CASE WHEN a.ppegt is null 
                THEN
                    ((a.ppent-(LAG(a.ppent) OVER w))+
                    (a.invt-(LAG(a.invt) OVER w)))/ NULLIF(LAG(a.at) OVER w, 0)
                ELSE
                    ((a.ppegt-(LAG(a.ppegt) OVER w))+
                    (a.invt-(LAG(a.invt) OVER w)))/ NULLIF(LAG(a.at) OVER w, 0)
                END as invest,
            (a.ceq-(LAG(a.ceq) OVER w))/NULLIF(LAG(a.ceq) OVER w, 0) as egr,
            (a.capx-(LAG(a.capx) OVER w))/
            NULLIF(LAG(a.capx) OVER w, 0) as pchcapx,
            (a.capx-(LAG(a.capx, 2) OVER w))/ NULLIF(LAG(a.capx, 2) OVER w, 0) 
            as grcapx,
            CASE WHEN a.gdwl is null OR a.gdwl = 0 
                THEN 0
                WHEN a.gdwl <> 0 AND a.gdwl is not null 
                AND ((a.gdwl-(LAG(a.gdwl) OVER w))/
                    NULLIF(LAG(a.gdwl) OVER w, 0)) is null 
                THEN 1
                ELSE 
                    (a.gdwl-(LAG(a.gdwl) OVER w))/NULLIF(LAG(a.gdwl) OVER w, 0)
                END as grGW,
            CASE WHEN (a.gdwlia is not null AND a.gdwlia <>0) OR 
                     (a.gdwlip is not null AND a.gdwlip <> 0) OR 
                     (a.gwo is not null AND a.gwo <> 0) THEN 1
                ELSE 0
                END as woGW,
            (a.che+a.rect*0.715+a.invt*0.547+a.ppent*0.535)/NULLIF(a.at, 0) 
            as tang,
            CASE WHEN a.sic BETWEEN 2100 AND 2199 
                OR a.sic BETWEEN 2080 AND 2085 
                OR a.naics IN ('7132', '71312', '713210', '71329', '713290', 
                    '72112', '721120') 
                THEN 1
                ELSE 0
                END as sin,
            a.act / NULLIF(a.lct, 0) as currat,
            ( (a.act/ NULLIF(a.lct, 0))-
              ((LAG(a.act) OVER w) / NULLIF(LAG(a.lct) OVER w, 0)))/
            (( NULLIF(LAG(a.act) OVER w, 0)  / NULLIF(LAG(a.lct) OVER w, 0))) 
            as pchcurrat,
            (a.act - a.invt) / NULLIF(a.lct, 0) as quick,
            (((a.act - a.invt) / NULLIF(a.lct, 0))-
             ((NULLIF((LAG(a.act) OVER w)-(LAG(a.invt) OVER w), 0))/ 
               NULLIF(LAG(a.lct) OVER w, 0))) / 
            ((NULLIF((LAG(a.act) OVER w)-(LAG(a.invt) OVER w), 0))/ 
              NULLIF(LAG(a.lct) OVER w, 0)) as pchquick,
            a.sale / NULLIF(a.che, 0) as salecash,
            a.sale / NULLIF(a.rect, 0) as salerec, 
            a.sale / NULLIF(a.invt, 0) as saleinv, 
            ( (a.sale/ NULLIF(a.invt, 0))-
              ((LAG(a.sale) OVER w) / NULLIF(LAG(a.invt) OVER w, 0)) )/
            (( NULLIF(LAG(a.sale) OVER w, 0) / NULLIF(LAG(a.invt) OVER w, 0))) 
            as pchsaleinv,
            (a.ib + a.dp) / (NULLIF(a.lt + (LAG(a.lt) OVER w), 0) / 2) 
            as cashdebt,
            CASE WHEN a.ppegt is null 
                THEN (a.fatb+a.fatl) / NULLIF(a.ppent, 0)
                ELSE (a.fatb+a.fatl) / NULLIF(a.ppegt, 0)
                END as realestate,
            CASE WHEN (a.dvt is not null AND a.dvt >0) AND
                      ((LAG(a.dvt) OVER w)=0 OR (LAG(a.dvt) OVER w) is null ) 
                THEN 1
                ELSE 0
                END as divi,
            CASE WHEN (a.dvt is null OR a.dvt = 0) 
                    AND ((LAG(a.dvt) OVER w) > 0 
                    AND (LAG(a.dvt) OVER w) is not null) 
                THEN 1
                ELSE 0
                END as divo,
            a.ob / (NULLIF(a.at + (LAG(a.at) OVER w), 0) / 2) as obklg,
            (a.ob - (LAG(a.ob) OVER w)) / 
            (NULLIF(a.ob + (LAG(a.ob) OVER w), 0) / 2) as chobklg,

            CASE WHEN a.dm is not null AND a.dm <> 0 
                THEN 1
                ELSE 0
                END as securedind,
            a.dm / NULLIF(a.dltt, 0) as secured,

            CASE WHEN a.dc is not null AND a.dc <> 0 OR 
                      (a.cshrc is not null AND a.cshrc <> 0 ) 
                THEN 1
                ELSE 0
                END as convind, 

            a.dc / NULLIF(a.dltt, 0) as conv, 
            ((a.rect+a.invt+a.ppent+a.aco+a.intan+a.ao-a.ap-a.lco-a.lo)-
            ((LAG(a.rect) OVER w)+(LAG(a.invt) OVER w)+(LAG(a.ppent) OVER w)+
            (LAG(a.aco) OVER w)+(LAG(a.intan) OVER w)+(LAG(a.ao) OVER w)-
            (LAG(a.ap) OVER w)-(LAG(a.lco) OVER w)-(LAG(a.lo) OVER w))-
            (a.rect-(LAG(a.rect) OVER w)+a.invt-(LAG(a.invt) OVER w)+
            a.aco-(LAG(a.aco) OVER w)-(a.ap-(LAG(a.ap) OVER w)+ 
            a.lco-(LAG(a.lco) OVER w))-a.dp))/
            ((a.at+NULLIF(LAG(a.at) OVER w, 0))/2) as grltnoa,

            (a.dr-(LAG(a.dr) OVER w))/((a.at+NULLIF(LAG(a.at) OVER w, 0))/2) 
            as chdrc,
            CASE WHEN ((a.xrd / NULLIF(a.at, 0))- (LAG( a.xrd_h) OVER w))/
                       NULLIF((LAG(a.xrd_h) OVER w),0) > 0.05 
                THEN 1
                ELSE 0
                END as rd,
            (a.xrd / NULLIF((LAG(a.xrd) OVER w), 0))-1-
            (a.ib/NULLIF((LAG(a.ceq) OVER w), 0)) as rdbias,
            a.ib / NULLIF((LAG(a.ceq) OVER w), 0) as roe,
            (a.revt-a.cogs-a.xsga0-a.xint0)/NULLIF((LAG(a.ceq) OVER w), 0) 
            as operprof,
            CASE WHEN a.txfo is null OR a.txfed is null 
                THEN 
                ((a.txt-a.txdi)/NULLIF(a.tr, 0))/NULLIF(a.ib, 0)
                WHEN a.txfo+a.txfed >0 OR a.txt > a.txdi AND a.ib <=0 
                THEN 1
                ELSE ((a.txfo-a.txfed)/NULLIF(a.tr, 0))/NULLIF(a.ib, 0)
                END as tb_1,
            ps0+ps1+ps2+ps3+ps4+ps5+ps6+ps7+ps8 as ps,
            a.ni/(NULLIF(a.at + (LAG(a.at) OVER w), 0) / 2) as roa,
            CASE WHEN a.oancf is null 
                THEN
                    (a.ib+a.dp)/(NULLIF(a.at + (LAG(a.at) OVER w), 0) / 2)
                 ELSE
                    a.oancf/(NULLIF(a.at + (LAG(a.at) OVER w), 0) / 2) 
                END as cfroa,
            a.xrd/(NULLIF(a.at + (LAG(a.at) OVER w), 0) / 2) as xrdint, 
            a.capx/(NULLIF(a.at + (LAG(a.at) OVER w), 0) / 2) as capxint,
            a.xad/(NULLIF(a.at + (LAG(a.at) OVER w), 0) / 2) as xadint, 
            AVG(a.at) OVER (PARTITION BY a.gvkey ORDER BY a.datadate ROWS
            BETWEEN 1 PRECEDING AND CURRENT ROW) as avgat, a.xsga, cpi.cpi,

            /*CASE WHEN ROW_NUMBER() OVER w = 1 
                    THEN (a.xsga/NULLIF(cpi.cpi, 0))/(0.1+0.15)
                    ELSE NULL
                    END AS orgcap_1, */
            c.splticrm

            FROM compa AS a 

            INNER JOIN ccm AS b 
            ON a.gvkey = b.gvkey
            AND a.datadate BETWEEN b.linkdt AND b.linkenddt

            LEFT JOIN (
               SELECT cpi, year FROM cpi
            ) as cpi
            ON CAST(a.fyear as INT) = CAST(cpi.year as INT) 

            LEFT JOIN (
            SELECT splticrm, gvkey, datadate 
            FROM comp.adsprate
            ) as c 

            ON CAST(a.gvkey as INT) = CAST(c.gvkey as INT)
            AND a.datadate = c.datadate

            WHERE a.at is not null 
            AND a.prcc_f is not null 
            AND a.ni is not null

            WINDOW w AS (PARTITION BY a.gvkey ORDER BY a.datadate)
            ) as data 

            LEFT JOIN (
                SELECT 
                CAST(permno AS INT),
                MIN(date::date) as exchstdt,
                MAX(COALESCE(date::date, CURRENT_DATE::date)) as exchedt
                FROM crsp.mseall
                WHERE shrcd IN {*self.shrcd,} 
                AND exchcd IN {*self.exchcd,}
                GROUP BY permno
            ) as mse

            ON data.permno=mse.permno
            AND data.datadate BETWEEN mse.exchstdt AND mse.exchedt

            WINDOW w1 as (PARTITION BY data.permno ORDER BY data.jdate)
            ORDER BY data.datadate, data.permno
            ),

            /* Monthly data: data_m */

            data_m AS(
            SELECT data.jdate, data.permno, data.ncusip, data.ticker, 
            data.comnam,
            data.exchcd, 
            CASE WHEN data.exchcd = 1
                THEN 'NYSE'
                WHEN data.exchcd = 2
                THEN 'AMEX'
                WHEN data.exchcd = 3
                THEN 'NASDAQ'
                END as exchname,
            data.siccd, data.sic2, 
            CASE WHEN data.siccd = 0
                THEN 'Undefined'
                WHEN data.siccd BETWEEN 1 AND 999
                THEN 'Agriculture'
                WHEN data.siccd BETWEEN 1000 AND 1499
                THEN 'Mining'
                WHEN data.siccd BETWEEN 1500 AND 1799
                THEN 'Construction'
                WHEN data.siccd BETWEEN 2000 AND 3999
                THEN 'Manufacturing'
                WHEN data.siccd BETWEEN 4000 AND 4999
                THEN 'Transportation'
                WHEN data.siccd BETWEEN 5000 AND 5199
                THEN 'Wholesale'
                WHEN data.siccd BETWEEN 5200 AND 5999
                THEN 'Retail'
                WHEN data.siccd BETWEEN 6000 AND 6799
                THEN 'Finance'
                WHEN data.siccd BETWEEN 7000 AND 8999
                THEN 'Services'
                WHEN data.siccd BETWEEN 9000 AND 9999
                THEN 'Public'
                WHEN data.siccd is NULL
                THEN 'Undefined'
                END as indname,
            data.prc, data.shrout, data.vol, 
            data.ret, data.ret_adj, 
            GREATEST(data.ret-data.rf, -1) as ret_ex, 
            GREATEST(data.ret_adj-data.rf, -1) as ret_adj_ex,
            data.rf,
            AVG(data.mom12m) OVER w1 as indmom,
            data.mve_m, data.mve, data.mom1m, data.mom6m, 
            data.mom12m, data.mom36m, data.dolvol, data.chmom, data.ipo, 
            data.turn

            FROM (
            SELECT
            a.prc, a.ret, a.retx, a.shrout, a.vol, 
            a.date::date, b.ncusip, 
            (date_trunc('month', a.date::date) + interval '1 month' 
            - interval '1 day')::date as jdate,
            b.namedt::date, 
            b.nameendt::date,
            CAST(a.permno as INT), 
            CAST(a.permco as INT), b.hsiccd, 
            b.ticker, b.shrcd, b.exchcd, b.siccd, b.comnam, 
            SUBSTRING(CAST(b.siccd AS text), 1, 2) as sic2, 
            c.dlret, c.dlstdt, c.dlstcd,

            CASE WHEN c.dlret is null 
                AND (c.dlstcd=500 OR 
                (c.dlstcd>=520 AND c.dlstcd<=584)) AND b.exchcd IN (1, 2)
                THEN GREATEST(a.ret-0.35, -1)
                WHEN c.dlret is null 
                AND (c.dlstcd=500 OR 
                (c.dlstcd>=520 AND c.dlstcd<=584)) AND b.exchcd IN (3)
                THEN GREATEST(a.ret-0.55, -1) 
                WHEN c.dlret is not null and c.dlret <-1 
                THEN GREATEST (a.ret-1, -1)
                WHEN c.dlret is null 
                THEN GREATEST(a.ret, -1)
                WHEN a.ret is null AND c.dlret <>0 
                THEN GREATEST(c.dlret, -1)
                ELSE GREATEST(a.ret, -1)
                END AS ret_adj, 
            (1+FF_O/100)^(0.083333333)-1  as rf, 
            ABS((LAG(a.prc) OVER w)*(LAG(a.shrout) OVER w)) as mve_m,
            LN(ABS((LAG(a.prc) OVER w)*(LAG(a.shrout) OVER w))) as mve,
            LN(ABS( (LAG(a.prc) OVER w))) as pps,
            LAG(a.ret) OVER w as mom1m, 
            ((1+(LAG(a.ret, 2) OVER w))*
            (1+(LAG(a.ret, 3) OVER w))*
            (1+(LAG(a.ret, 4) OVER w))*
            (1+(LAG(a.ret, 5) OVER w))* 
            (1+(LAG(a.ret, 6) OVER w)))-1 as mom6m, 
            ((1+(LAG(a.ret, 2) OVER w))*
            (1+(LAG(a.ret, 3) OVER w))*
            (1+(LAG(a.ret, 4) OVER w))*
            (1+(LAG(a.ret, 5) OVER w))* 
            (1+(LAG(a.ret, 6) OVER w))* 
            (1+(LAG(a.ret, 7) OVER w))*
            (1+(LAG(a.ret, 8) OVER w))*
            (1+(LAG(a.ret, 9) OVER w))*
            (1+(LAG(a.ret, 10) OVER w))* 
            (1+(LAG(a.ret, 11) OVER w))*
            (1+(LAG(a.ret, 12) OVER w)))-1 as mom12m,
            ((1+(LAG(a.ret, 13) OVER w))*
            (1+(LAG(a.ret, 14) OVER w))*
            (1+(LAG(a.ret, 15) OVER w))*
            (1+(LAG(a.ret, 16) OVER w))* 
            (1+(LAG(a.ret, 17) OVER w))* 
            (1+(LAG(a.ret, 18) OVER w))*
            (1+(LAG(a.ret, 19) OVER w))*
            (1+(LAG(a.ret, 20) OVER w))*
            (1+(LAG(a.ret, 21) OVER w))* 
            (1+(LAG(a.ret, 22) OVER w))*
            (1+(LAG(a.ret, 23) OVER w))*
            (1+(LAG(a.ret, 24) OVER w))*
            (1+(LAG(a.ret, 25) OVER w))*
            (1+(LAG(a.ret, 26) OVER w))* 
            (1+(LAG(a.ret, 27) OVER w))* 
            (1+(LAG(a.ret, 28) OVER w))*
            (1+(LAG(a.ret, 29) OVER w))*
            (1+(LAG(a.ret, 30) OVER w))*
            (1+(LAG(a.ret, 31) OVER w))* 
            (1+(LAG(a.ret, 32) OVER w))*
            (1+(LAG(a.ret, 33) OVER w))*
            (1+(LAG(a.ret, 34) OVER w))*
            (1+(LAG(a.ret, 35) OVER w))*
            (1+(LAG(a.ret, 36) OVER w)))-1 as mom36m,
            LN(NULLIF(ABS((LAG(a.vol, 2) OVER w)*(LAG(a.prc, 2) OVER w)) , 0)) 
            as dolvol, 
            (((1+(LAG(a.ret, 1) OVER w))*
            (1+(LAG(a.ret, 2) OVER w))*
            (1+(LAG(a.ret, 3) OVER w))*
            (1+(LAG(a.ret, 4) OVER w))*
            (1+(LAG(a.ret, 5) OVER w))* 
            (1+(LAG(a.ret, 6) OVER w)))-1)-
            (((1+(LAG(a.ret, 7) OVER w))*
            (1+(LAG(a.ret, 8) OVER w))*
            (1+(LAG(a.ret, 9) OVER w))*
            (1+(LAG(a.ret, 10) OVER w))*
            (1+(LAG(a.ret, 11) OVER w))* 
            (1+(LAG(a.ret, 12) OVER w)))-1) as chmom, 
            (((LAG(a.vol) OVER w)+
            (LAG(a.vol, 2) OVER w)+
            (LAG(a.vol, 3) OVER w))/3)/NULLIF(a.shrout, 0) as turn,

            CASE WHEN ROW_NUMBER() OVER w <= 12 
                THEN 1
                ELSE 0
                END as ipo

            FROM crsp.msf as a

            LEFT JOIN crsp.msenames as b
            ON a.permno=b.permno
            AND a.date BETWEEN b.namedt AND b.nameendt

            LEFT JOIN crsp.msedelist as c 
            ON a.permno=c.permno 
            AND (date_trunc('month', a.date::date) + interval '1 month' 
            - interval '1 day')::date = (date_trunc('month', c.dlstdt::date) + 
            interval '1 month' - interval '1 day')::date

            LEFT JOIN frb.rates_monthly as d 
            ON (date_trunc('month', a.date::date) + interval '1 month' 
            - interval '1 day')::date = (date_trunc('month', d.date::date) + 
            interval '1 month' - interval '1 day')::date

            WHERE extract(year from a.date) >= {self.start_yr - 5}
            and b.exchcd IN {*self.exchcd,}
            and b.shrcd in {*self.shrcd,}
            AND a.permno IN (SELECT DISTINCT permno from data_a)

            WINDOW w as (PARTITION BY a.permno ORDER BY a.date)
            ) as data

            WINDOW w1 as (PARTITION BY data.sic2, data.jdate)
            ORDER BY data.jdate, data.permno
            ),


             /* Quarterly Characteristics: data_q */

            comp_q AS (
            SELECT DISTINCT ON (f.datadate, f.gvkey)
            b.permno,
            CAST(f.gvkey as INT), f.cusip, f.datadate::date, 
            CAST(f.fyearq as INT), CAST(f.fqtr as INT), 
            (date_trunc('month', f.datadate::date) + interval '1 month'
            *{self.lag_quarter + 3} - interval '1 day')::date AS jdate,
            substr(c.sic,1,2) as sic2,
            f.rdq, f.ibq, f.saleq, f.txtq, f.revtq, f.cogsq, f.xsgaq, f.atq, 
            f.actq, f.cheq, f.lctq, f.dlcq, f.ppentq, abs(f.prccq) as prccq, 
            abs(f.prccq) * f.cshoq as mveq, f.ceqq, f.seqq, f.pstkq, f.ltq,
            f.pstkrq, 
            CASE WHEN f.pstkrq is not null 
                THEN pstkrq
                ELSE pstkq
            END as pstk,
            CASE WHEN f.seqq is null
                THEN f.ceqq+COALESCE(f.pstkrq, f.pstkq)
                WHEN seqq is null AND 
                (f.ceqq is null OR COALESCE(f.pstkrq, f.pstkq) is null)
                THEN f.atq-f.ltq
                ELSE f.seqq 
                END as scal, 
            ROW_NUMBER() OVER (PARTITION BY f.datadate, f.gvkey ORDER BY f.rdq)
            as rn

            FROM comp.fundq as f

            INNER JOIN ccm as b 
            ON CAST(f.gvkey as INT) = CAST(b.gvkey as INT)
            AND f.datadate BETWEEN b.linkdt AND b.linkenddt

            LEFT JOIN comp.company as c 
            ON f.gvkey = c.gvkey

            WHERE f.indfmt = '{self.indfmt}' 
            AND f.datafmt = '{self.datafmt}' 
            AND f.popsrc = '{self.popsrc}'
            AND f.consol = '{self.consol}'
            AND f.popsrc = '{self.popsrc}'
            AND f.curcdq = '{self.curcd}'
            AND extract(year from f.datadate) >= {self.start_yr - 5}
            ORDER BY f.datadate, f.gvkey
            ), 

            ibes_q AS (
            SELECT DISTINCT ON (a.fpedats, b.permno)
            a.cusip, a.fpedats, a.ticker, a.medest, a.actual, 
            CAST(b.permno as INT)

            FROM ibes.statsum_epsus as a

            LEFT JOIN wrdsapps.ibcrsphist as b
            ON a.ticker = b.ticker
            AND a.fpedats BETWEEN b.SDATE AND b.EDATE

            WHERE fpi='6'
            AND a.statpers<a.ANNDATS_ACT
            AND a.measure='EPS'
            AND a.medest is not null
            AND a.fpedats is not null
            AND (a.fpedats-a.statpers)>=0
            AND b.SCORE = 1

            ORDER BY a.fpedats, b.permno
            ),

            data_q AS(
            SELECT a.*, b.medest, b.actual,

            stddev_samp(a.scf) OVER w as stdcf, 
            CASE WHEN b.medest is null OR b.actual is null
                THEN a.che/NULLIF(a.mveq, 0)
                ELSE (b.actual-b.medest)/NULLIF(a.prccq, 0) 
                END as sue
            FROM (
            SELECT 
            data.jdate, data.jdate_end_q, data.datadate, data.permno, 
            data.fyearq, data.fqtr, data.sic2, 
            data.cusip, data.rdq, data.mveq,
            (dfq.avg2-dfq.avg1)/NULLIF(dfq.avg1, 0) as aeavol,
            data.cash, data.che, data.chtx, data.cinvest, 
            dfq.ear,
            data.n1+
            data.n1*data.n2+
            data.n1*data.n2*data.n3+
            data.n1*data.n2*data.n3*data.n4+
            data.n1*data.n2*data.n3*data.n4*data.n5+
            data.n1*data.n2*data.n3*data.n4*data.n5*data.n6+
            data.n1*data.n2*data.n3*data.n4*data.n5*data.n6*data.n7+
            data.n1*data.n2*data.n3*data.n4*data.n5*data.n6*data.n7*data.n8
            as nincr,
            data.prccq, 
            data.roaq, 
            stddev_samp(data.roaq) OVER w2 as roavol,
            data.roeq, 
            data.rsup, 
            stddev_samp(data.sacc) OVER w2 as stdacc,
            stddev_samp(data.rsup) OVER w3 as sgrvol, 
            CASE WHEN data.saleq <= 0
                THEN (data.ibq/0.01)-data.sacc
                ELSE (data.ibq/NULLIF(data.saleq, 0))-data.sacc
                END as scf

            FROM(
            SELECT
            a.jdate, 
            CASE WHEN LEAD(a.jdate) OVER w1 is null
                THEN
                    a.jdate + interval '3 month'
                ELSE
                    LEAD(a.jdate) OVER w1
                END
                    as jdate_end_q,
            a.datadate, a.fqtr, a.sic2, 
            a.permno, a.gvkey, a.prccq,
            a.fyearq, a.cusip, a.mveq,
            a.rdq, a.saleq, a.ibq,
            a.cheq/NULLIF(a.atq, 0) as cash,
            a.ibq-(LAG(a.ibq, 4) OVER w1) as che,
            (a.txtq - (LAG(a.txtq, 4) OVER w1))/
            NULLIF((LAG(a.atq, 4) OVER w1), 0) as chtx,
            CASE WHEN a.saleq > 0
            THEN
            ((a.ppentq - (LAG(a.ppentq) OVER w1))/NULLIF(a.saleq, 0))-
            ((((LAG(a.ppentq) OVER w1)-(LAG(a.ppentq, 2) OVER w1))/
            NULLIF((LAG(a.saleq) OVER w1), 0))+
            (((LAG(a.ppentq, 2) OVER w1)-(LAG(a.ppentq, 3) OVER w1))/
            NULLIF((LAG(a.saleq, 2) OVER w1), 0))+
            (((LAG(a.ppentq, 3) OVER w1)-(LAG(a.ppentq, 4) OVER w1))/
            NULLIF((LAG(a.saleq, 3) OVER w1), 0)))/3
            ELSE
            ((a.ppentq - (LAG(a.ppentq) OVER w1))/0.01)-
            ((((LAG(a.ppentq) OVER w1)-(LAG(a.ppentq, 2) OVER w1))/
            0.01)+
            (((LAG(a.ppentq, 2) OVER w1)-(LAG(a.ppentq, 3) OVER w1))/
            0.01)+
            (((LAG(a.ppentq, 3) OVER w1)-(LAG(a.ppentq, 4) OVER w1))/
            0.01))/3
            END as cinvest,
            CASE WHEN a.ibq > (LAG(a.ibq) OVER w1) 
                THEN 1 ELSE 0 END as n1,
            CASE WHEN (LAG(a.ibq, 1) OVER w1) > (LAG(a.ibq, 2) OVER w1) 
                THEN 1 ELSE 0 END as n2,
            CASE WHEN (LAG(a.ibq, 2) OVER w1) > (LAG(a.ibq, 3) OVER w1)
                THEN 1 ELSE 0 END as n3,
            CASE WHEN (LAG(a.ibq, 3) OVER w1) > (LAG(a.ibq, 4) OVER w1)
                THEN 1 ELSE 0 END as n4,
            CASE WHEN (LAG(a.ibq, 4) OVER w1) > (LAG(a.ibq, 5) OVER w1)
                THEN 1 ELSE 0 END as n5,
            CASE WHEN (LAG(a.ibq, 5) OVER w1) > (LAG(a.ibq, 6) OVER w1)
                THEN 1 ELSE 0 END as n6,
            CASE WHEN (LAG(a.ibq, 6) OVER w1) > (LAG(a.ibq, 7) OVER w1)
                THEN 1 ELSE 0 END as n7,
            CASE WHEN (LAG(a.ibq, 7) OVER w1) > (LAG(a.ibq, 8) OVER w1)
                THEN 1 ELSE 0 END as n8,
            a.ibq/NULLIF(LAG(a.atq) OVER w1, 0) as roaq,
            a.ibq/NULLIF(LAG(a.scal) OVER w1, 0) as roeq,
            (a.saleq-(LAG(a.saleq, 4) OVER w1))/
            NULLIF(a.mveq, 0) as rsup, 
            CASE WHEN a.saleq <= 0
                THEN 
                    ((a.actq-(LAG(a.actq) OVER w1)-
                    (a.cheq-(LAG(a.cheq) OVER w1)))-
                    (a.lctq-(LAG(a.lctq) OVER w1)-
                    (a.dlcq-(LAG(a.dlcq) OVER w1))))/
                    0.01
                ELSE
                    ((a.actq-(LAG(a.actq) OVER w1)-
                    (a.cheq-(LAG(a.cheq) OVER w1)))-
                    (a.lctq-(LAG(a.lctq) OVER w1)-
                    (a.dlcq-(LAG(a.dlcq) OVER w1))))/
                    NULLIF(a.saleq, 0)
                END as sacc

            FROM comp_q as a
            WINDOW w1 as (PARTITION BY a.permno ORDER BY a.jdate)
            ORDER BY a.jdate, a.permno
            ) as data

            LEFT JOIN (
            SELECT permno, date,
            SUM(ret) OVER wd1 as ear,
            AVG(vol) OVER wd2 as avg1,
            AVG(vol) OVER wd1 as avg2
            FROM crsp.dsf
            WINDOW wd1 AS
                (PARTITION BY permno ORDER BY date ROWS BETWEEN
                1 PRECEDING AND 1 FOLLOWING),
                 wd2 AS
                (PARTITION BY permno ORDER BY date ROWS BETWEEN
                30 PRECEDING AND 10 PRECEDING)
            ) as dfq

            ON data.permno = dfq.permno 
            AND data.rdq = dfq.date

            WINDOW w2 as (PARTITION BY data.permno ORDER BY data.jdate
            ROWS BETWEEN 15 PRECEDING AND CURRENT ROW),
                w3 as (PARTITION BY data.permno ORDER BY data.jdate
            ROWS BETWEEN 14 PRECEDING AND CURRENT ROW)
            ORDER BY data.jdate, data.permno
            ) as a 

            LEFT JOIN ibes_q as b
            ON a.permno = b.permno 
            AND a.datadate = b.fpedats

            WINDOW w as (PARTITION BY a.permno ORDER BY a.jdate
            ROWS BETWEEN 15 PRECEDING AND CURRENT ROW)
            ), 

            ibes_0 AS(
            SELECT ibes0.*, 
            CASE WHEN LEAD(ibes0.jdate) OVER w1 is null
            THEN
                ibes0.jdate + interval '3 month'
            ELSE
                LEAD(ibes0.jdate) OVER w1
            END
                as jdate_end_ibes0

            FROM (
            SELECT DISTINCT ON(a.statpers, b.permno)
            (date_trunc('month', a.statpers::date) + interval '1 month' * 2 
            - interval '1 day')::date as jdate,

            CAST(b.permno as INT),
            a.meanest as fgr5yr,
            a.numest as nanalyst,
            a.meanest

            FROM ibes.statsum_epsus as a
            INNER JOIN wrdsapps.ibcrsphist as b

            ON a.cusip = b.NCUSIP
            AND a.statpers BETWEEN b.SDATE AND b.EDATE

            WHERE fpi='0'
            AND a.meanest is not null

            ORDER BY a.statpers, b.permno
            ) as ibes0

            WINDOW w1 as (PARTITION BY ibes0.permno ORDER BY ibes0.jdate)
            ORDER BY ibes0.jdate, ibes0.permno
            ),

            data_a_adj AS(
            SELECT a.*, 
            a.bm - AVG(a.bm) OVER w as bm_ia, 
            a.cfp - AVG(a.cfp) OVER w as cfp_ia, 
            a.chpm - AVG(a.chpm) OVER w as chpmia,
            a.chato - AVG(a.chato) OVER w as chatoia,
            SUM(a.sale) OVER w as indsale,
            a.hire - AVG(a.hire) OVER w as chempia, 
            a.pchcapx - AVG(a.pchcapx) OVER w as pchcapx_ia, 
            a.tb_1 - AVG(a.tb_1) OVER w as tb,
            a.mve_f - AVG(a.mve_f) OVER w as mve_ia, 
            CASE WHEN a.roa > md.md_roa
                THEN 1 
                ELSE 0
                END as m1, 

            CASE WHEN a.cfroa > md.md_cfroa
                THEN 1 
                ELSE 0
                END as m2, 

            CASE WHEN a.oancf > a.ni
                THEN 1 
                ELSE 0
                END as m3,

            CASE WHEN a.xrdint > md.md_xrdint
                THEN 1 
                ELSE 0
                END as m4, 

            CASE WHEN a.capxint > md.md_capxint
                THEN 1 
                ELSE 0
                END as m5, 

            CASE WHEN a.xadint > md.md_xadint
                THEN 1 
                ELSE 0
                END as m6

            FROM data_a as a

            LEFT JOIN (
            SELECT a.sic2, a.fyear,

            PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY a.roa)  
            as md_roa,

            PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY a.cfroa)  
            as md_cfroa,

            PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY a.xrdint)  
            as md_xrdint,

            PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY a.capxint)  
            as md_capxint,

            PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY a.xadint)  
            as md_xadint

            FROM data_a as a
            GROUP BY a.sic2, a.fyear
            ) as md 
            ON a.sic2 = md.sic2
            AND a.fyear = md.fyear

            WINDOW w as (PARTITION BY a.sic2, a.fyear)

            ),

            /* Annual forecast IBES: ibes1 */
            ibes_1 AS (
            SELECT ibes1.jdate, 
            CASE WHEN LEAD(ibes1.jdate) OVER w1 is null
                        THEN
                            ibes1.jdate + interval '3 month'
                        ELSE
                            LEAD(ibes1.jdate) OVER w1
                        END
                            as jdate_end_ibes1,

            ibes1.permno, ibes1.disp, ibes1.chfeps, ibes1.meanest,
            ibes1.numest as nanalyst, 
            ibes1.numest - LAG(ibes1.numest, 3) OVER w1 as chnanalyst

            FROM (
            SELECT
            a.*, CAST(b.permno as INT),
            (date_trunc('month', a.statpers::date) + interval '1 month' * 2
                    - interval '1 day')::date as jdate, 
            CASE WHEN a.meanest = 0
                THEN a.stdev/0.01
                ELSE a.stdev/NULLIF(ABS(a.meanest), 0)
                END AS disp, 
            a.meanest - LAG(a.meanest) OVER (PARTITION BY a.cusip ORDER BY 
            a.anndats_act, a.fpedats, a.statpers) as chfeps


            FROM ibes.statsum_epsus as a
            LEFT JOIN wrdsapps.ibcrsphist as b
            ON a.ticker = b.ticker
            AND a.statpers BETWEEN b.SDATE AND b.EDATE

            WHERE fpi='1'
            AND a.statpers<a.ANNDATS_ACT
            AND a.measure='EPS'
            AND a.medest is not null
            AND a.fpedats is not null
            AND (a.fpedats-a.statpers)>=0
            AND b.SCORE = 1

            ORDER BY a.cusip, a.anndats_act, a.fpedats, a.statpers
            ) as ibes1

            WINDOW w1 as (PARTITION BY ibes1.permno ORDER BY ibes1.jdate)
            ORDER BY ibes1.jdate, ibes1.permno
            ), 

            data_m_adj AS(
            SELECT a.*, b.disp, b.chfeps, b.nanalyst, b.chnanalyst, 
            b.meanest

            FROM data_m as a
            LEFT JOIN ibes_1 as b

            ON a.permno = b.permno
            AND a.jdate >= b.jdate
            AND a.jdate < b.jdate_end_ibes1

            ORDER BY a.jdate, a.permno

            ),

            data_q_adj AS(
            SELECT a.*, 
            CASE WHEN a.roavol > md.md_roavol 
                THEN 1 
                ELSE 0 
                END as m7, 

            CASE WHEN a.roavol > md.md_sgrvol
                THEN 1 
                ELSE 0 
                END as m8

            FROM data_q as a 

            LEFT JOIN (
            SELECT a.fyearq, a.fqtr, a.sic2,

            PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY a.roavol)  
            as md_roavol,

            PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY a.sgrvol)  
            as md_sgrvol

            FROM data_q as a
            GROUP BY a.fyearq, a.fqtr, a.sic2
            ) as md 

            ON a.fyearq = md.fyearq
            AND a.fqtr = md.fqtr
            AND a.sic2 = md.sic2

            ), 
            data_d AS(
            SELECT 
            d1.jdate, d1.permno, d1.maxret, d1.retvol, d1.baspread, 
            d1.std_dolvol, d1.std_turn, d1.ill,

            ( d1.countzero + ((1/ NULLIF(d1.turn,0) )/480000))*
            21/NULLIF(d1.ndays, 0) as zerotrade

            FROM(
            SELECT
            (date_trunc('month', date::date) + interval '1 month' * 2
                    - interval '1 day')::date as jdate, 
            CAST(permno as INT), 
            MAX(ret) as maxret, 
            stddev_samp(ret) as retvol, 
            AVG(  (askhi-bidlo) /NULLIF(((askhi+bidlo)/2), 0)) as baspread,
            stddev_samp(LN(NULLIF(ABS(prc*vol), 0))) as std_dolvol, 
            stddev_samp(vol/NULLIF(shrout, 0)) as std_turn, 
            AVG( ABS(ret)/NULLIF(ABS(prc*vol),0) ) as ill, 
            count(*) FILTER (WHERE vol = 0) as countzero, 
            count(*) as ndays, 
            SUM(vol/NULLIF(shrout, 0)) as turn

            FROM crsp.dsf 
            WHERE extract(year from date) >= {self.start_yr}

            GROUP BY jdate, permno
            ORDER BY jdate, permno
            ) as d1

            ORDER BY d1.jdate, d1.permno
            ),
            out AS(
            SELECT 
            m.jdate, a.fyear, m.permno, m.ticker, m.comnam, m.exchcd, 
            m.exchname, m.siccd, 
            m.indname, m.mve_m, m.rf, m.ret, m.ret_adj, m.ret_ex, m.ret_adj_ex,
            a.avgat, a.cpi, a.xsga, 

            CASE WHEN ROW_NUMBER() OVER w = 1 
                    THEN (a.xsga/NULLIF(a.cpi, 0))/(0.1+0.15)
                    ELSE NULL
                    END AS orgcap_1, 

            a.absacc, a.acc, q.aeavol, a.age, a.agr, d.baspread, 
            a.bm, a.bm_ia, q.cash,
            a.cashdebt, a.cashpr, a.cfp, a.cfp_ia, a.chatoia,
            a.chcsho, a.chempia, m.chfeps, a.chinv, m.chmom, 
            m.chnanalyst, a.chpmia, 
            q.chtx, q.cinvest, a.convind, a.currat, a.depr, m.disp, a.divi, 
            a.divo, m.dolvol, a.dy, q.ear, a.egr, a.ep, i.fgr5yr, a.gma, 
            a.grcapx, a.grltnoa,
            SUM((a.sale/NULLIF(a.indsale, 0))*(a.sale/NULLIF(a.indsale, 0)))
            OVER (PARTITION BY a.sic2, a.fyear) as herf,
            a.hire, d.ill, m.indmom, a.invest, m.ipo, 
            a.lev, a.lgr, d.maxret, m.mom1m, m.mom6m, m.mom12m, m.mom36m, 
            a.m1+a.m2+a.m3+a.m4+a.m5+a.m6+q.m7+q.m8 as ms, 
            m.mve, a.mve_ia, m.nanalyst, q.nincr, a.operprof, 
            a.pchcapx_ia, a.pchcurrat, a.pchdepr, a.pchgm_pchsale, 
            a.pchquick, a.pchsale_pchinvt, a.pchsale_pchrect, 
            a.pchsale_pchxsga, a.pchsaleinv, a.pctacc, a.ps, a.quick, a.rd, 
            a.rd_mve, a.rd_sale, a.realestate, d.retvol, q.roaq, q.roavol, 
            q.roeq, 
            a.roic, q.rsup, a.salecash, a.saleinv, a.salerec, a.secured, 
            a.securedind, 
            m.meanest / NULLIF(ABS(q.prccq), 0) as sfe, 
            a.sgr, q.sgrvol, a.sin, a.sp, d.std_dolvol, d.std_turn, 
            q.stdacc, q.stdcf, q.sue, a.tang, a.tb, m.turn, d.zerotrade

            FROM 
            data_m_adj as m 

            LEFT JOIN 
            data_a_adj as a
            ON m.permno = a.permno
            AND m.jdate >= a.jdate
            AND m.jdate < a.jdate_end

            LEFT JOIN 
            data_q_adj as q
            ON m.permno = q.permno
            AND m.jdate >= q.jdate
            AND m.jdate < q.jdate_end_q

            LEFT JOIN 
            ibes_0 as i
            ON m.permno = i.permno 
            AND m.jdate >= i.jdate
            AND m.jdate < i.jdate_end_ibes0

            LEFT JOIN data_d as d
            ON m.permno = d.permno
            AND m.jdate = d.jdate

            WHERE 
            extract(year from m.jdate) >= {self.start_yr}
            AND extract(year from m.jdate) <= {self.end_yr}
            AND m.ret is not null

            WINDOW w as (PARTITION BY m.permno ORDER BY m.jdate)
            ORDER BY m.jdate, m.permno
            )
            SELECT * FROM out

            """

        )
        wrds_db.close()
        print('3. Preparation of daily variables')
        permnos = data.permno.unique().tolist()

        wrds_db = wrds.Connection(wrds_username=self.wrds_username)
        data_beta = wrds_db.raw_sql(
            f"""
            WITH beta1 AS(
            SELECT
            wkdt, jdate, permno, wkret, ewret,  
            LAG(ewret)     OVER w2 as ewmkt_l1,
            LAG(ewret , 2) OVER w2 as ewmkt_l2,
            LAG(ewret , 3) OVER w2 as ewmkt_l3,
            LAG(ewret , 4) OVER w2 as ewmkt_l4

            FROM (
            SELECT b1.jdate, b1.wkdt, b1.permno, b1.wkret, 
            AVG(wkret) OVER w as ewret
            FROM(
            SELECT DISTINCT ON(date_trunc('week', date::date), permno)
            date_trunc('week', date::date) as wkdt,
            (date_trunc('month', date_trunc('week', date::date)::date) 
            + interval '1 month' * 2 - interval '1 day')::date as jdate,
            CAST(permno as INT),
            EXP(SUM(LN(NULLIF(1+GREATEST(ret, -1),0))) OVER w ) - 1 as wkret

            FROM crsp.dsf
            WHERE extract(year from date) >= {self.start_yr - 5}
            AND permno IN {tuple(permnos)}

            WINDOW w as (PARTITION BY date_trunc('week', date::date) , permno),
                   w2 as (PARTITION BY date_trunc('week', date::date) )
            ORDER BY wkdt, permno
            ) as b1

            WINDOW w as (PARTITION BY b1.wkdt)

            ORDER BY b1.wkdt, b1.permno
            ) as b2

            WHERE wkret is not null
            AND ewret is not null

            WINDOW w2 as (PARTITION BY permno ORDER BY wkdt)
            ) 

            SELECT * 
            FROM beta1 
            WHERE permno IN 
            (SELECT permno FROM beta1 GROUP BY permno HAVING COUNT(*) >= 52) 
            ORDER BY wkdt, permno
            """

        )
        wrds_db.close()
        betas = [self._rolling_betas(perm=x, data=data_beta)
                 for x in data_beta.permno.unique().tolist()]
        betas = pd.concat(betas)
        betas.sort_index(inplace=True)
        data_beta = data_beta.join(betas)
        data_beta = data_beta.groupby(['jdate', 'permno'])[
            ['beta', 'betasq', 'pricedelay', 'idiovol']].last().reset_index()
        data = data.merge(data_beta, on=['jdate', 'permno'], how='left')
        del data_beta
        print('4. Recursive orgcap')
        orgcaps = [self._orgcap(data=data, perm=x) for x in
                   data.permno.unique().tolist()]
        orgcaps = pd.concat(orgcaps)
        data = data.merge(orgcaps, on=['jdate', 'permno', 'fyear'], how='left')
        data.drop(labels=['orgcap_1', 'xsga', 'cpi', 'avgat'], axis=1,
                  inplace=True)
        data['orgcap'] = data.groupby('permno')['orgcap'].ffill(limit=11)
        self.chars_list = sorted(data.columns.tolist()[15:])
        core_cols = data.columns.tolist()[:15]
        data = data[core_cols + self.chars_list]
        fred = Fred(api_key=self.fred_key)
        # Problem that WRDS does not update fredfund rate frequenyly enough
        # So we load it again, but from FRED. It is the same data
        fedfunds = fred.get_series('FEDFUNDS',
                                   observation_start=f'{self.start_yr}-01-01')
        fedfunds = fedfunds.to_frame().rename(columns={0: 'rf'})
        fedfunds = fedfunds.reset_index().rename(columns={'index': 'jdate'})
        fedfunds['jdate'] = pd.to_datetime(fedfunds['jdate']) + MonthEnd(0)
        fedfunds['rf'] = fedfunds['rf'] / 100
        fedfunds['rf'] = (1 + fedfunds['rf']) ** (1 / 12) - 1
        del data['rf']
        data['jdate'] = pd.to_datetime(data['jdate']) + MonthEnd(0)
        data = data.merge(fedfunds, on=['jdate'], how='left')
        data['ret_ex'] = data['ret'] - data['rf']
        data['ret_adj_ex'] = data['ret_adj'] - data['rf']
        data['ret_ex'] = np.where(data['ret_ex'] < -1, -1, data['ret_ex'])
        data['ret_adj_ex'] = np.where(data['ret_adj_ex'] < -1, -1,
                                      data['ret_adj_ex'])
        # move rf back to the front
        data = data[core_cols + self.chars_list]
        # remove possible infinite values
        data[self.chars_list] = data[self.chars_list].replace(
            [np.inf, -np.inf], np.nan)
        # size class
        data.reset_index(inplace=True, drop=True)
        size = data[data.exchname == 'NYSE'].groupby('jdate')['mve_m'].apply(
            lambda x: x.quantile(q=[0.2, 0.5])).unstack().rename(
            columns={0.2: 'NYSE_small', 0.5: 'NYSE_large'}).reset_index()
        data = data.merge(size, on='jdate', how='left')
        conditions = [
            (data['mve_m'] < data['NYSE_small']),
            (data['mve_m'].between(data['NYSE_small'], data['NYSE_large'])),
            (data['mve_m'] > data['NYSE_large'])]
        choices = ['Micro', 'Small', 'Large']
        data['size_class'] = np.select(conditions, choices, default=np.nan)
        # issue with nonnegative columns: assumption: absolute values
        nonneg = sorted(['betasq', 'mve', 'dy', 'lev', 'baspread', 'depr',
                         'sp', 'turn', 'dolvol', 'std_dolvol', 'std_turn',
                         'disp', 'idiovol', 'roavol', 'ill', 'rd_sale',
                         'secured',
                         'rd_mve', 'retvol', 'zerotrade', 'stdcf', 'tang',
                         'absacc', 'stdacc', 'cash', 'orgcap', 'salecash',
                         'salerec', 'saleinv', 'sgrvol', 'herf', 'currat',
                         'pchsaleinv', 'cashdebt', 'realestate', 'quick'])
        data[nonneg] = data[nonneg].abs()
        # add year and jyear
        data['year'] = data.jdate.dt.year
        data['jyear'] = np.where(data.jdate.dt.month <= 6, data.year - 2,
                                 data.year - 1)
        core_cols = ['jdate', 'fyear', 'year', 'jyear', 'permno', 'ticker',
                     'comnam', 'exchcd', 'exchname', 'siccd', 'indname',
                     'size_class', 'mve_m', 'rf', 'ret', 'ret_adj', 'ret_ex',
                     'ret_adj_ex']
        data = data[core_cols + self.chars_list]
        # winsorize data: 1%, 99%
        bounded = sorted(['beta', 'ep', 'fgr5yr', 'mom12m', 'mom1m', 'mom6m',
                          'indmom', 'sue', 'agr', 'maxret', 'chfeps', 'roaq',
                          'mom36m', 'pchcurrat', 'pchquick', 'pchdepr', 'sgr',
                          'chempia', 'acc', 'pchsale_pchinvt', 'roic', 'gma',
                          'pchsale_pchrect', 'pchcapx_ia', 'ear', 'rsup',
                          'hire', 'chcsho', 'chpmia', 'chatoia', 'aeavol',
                          'pchgm_pchsale', 'sfe', 'pchsale_pchxsga', 'mve_ia',
                          'cfp_ia', 'chinv', 'grltnoa', 'cinvest', 'tb', 'cfp',
                          'lgr', 'egr', 'pricedelay', 'grcapx', 'chmom',
                          'cashpr', 'roeq', 'invest', 'chtx',
                          'pctacc', 'operprof', 'bm_ia', 'bm'])
        data[nonneg] = data.groupby('jdate')[nonneg].apply(
            lambda x: x.clip(upper=x.quantile(wh), axis=1))
        # positive and negative outliers
        data[bounded] = data.groupby('jdate')[bounded].apply(
            lambda x: x.clip(lower=x.quantile(wl), upper=x.quantile(wh),
                             axis=1))
        data['jdate'] = pd.to_datetime(data['jdate'])
        data.dropna(subset=['mve_m'], inplace=True)
        data.rename({'jdate': 'date'}, inplace=True)
        self.chars_data = data
        # save factors as json list
        with open("./data/factor_list.json", 'w') as f:
            json.dump(self.chars_list, f, indent=2)
        return self

    def clean_chars(self, dropna_cols, how='std', keep_micro=True):
        valid = {'std', 'rank_norm'}
        if how not in valid:
            raise ValueError("how must be one of %r." % valid)
        if self.chars_data is None:
            print('Download data first!')
        self.chars_data_clean = self.chars_data.copy(deep=True)
        # drop missing values if wanted
        if dropna_cols is not None:
            self.chars_data_clean.dropna(subset=dropna_cols, inplace=True)
            self.chars_data_clean.reset_index(inplace=True, drop=True)
        # clean
        if not keep_micro:
            self.chars_data_clean = self.chars_data_clean[
                self.chars_data_clean.size_class != 'Micro']
        self.chars_data_clean.sort_values(['date', 'permno'])
        self.chars_data_clean.reset_index(inplace=True, drop=True)
        if how == 'std':
            self.chars_data_clean[self.chars_list] = \
                self.chars_data_clean.groupby('date')[
                    self.chars_list].transform(
                    lambda x: (x - x.mean()) / x.std()).fillna(0)
        elif how == 'rank_norm':
            self.chars_data_clean[self.chars_list] = \
                self.chars_data_clean.groupby(
                    'date')[self.chars_list].transform(
                    lambda x: (x - x.mean())).fillna(0)
            self.chars_data_clean[self.chars_list] = \
                self.chars_data_clean.groupby(
                    'date')[self.chars_list].transform(
                lambda x: stats.norm.ppf(
                    stats.norm.cdf(-3) +
                    (x.rank(method='first') - 1) / (x.count() - 1) * (
                            stats.norm.cdf(3) - stats.norm.cdf(-3)))).round(7)
            self.chars_data_clean[self.chars_list] = self.chars_data_clean[
                                                         self.chars_list]/3
        elif how == 'rank':
            self.chars_data_clean[self.chars_list] = \
                self.chars_data_clean.groupby('date')[
                    self.chars_list].transform(
                    lambda x: (x - x.mean())).fillna(0)
            self.chars_data_clean[self.chars_list] = \
                self.chars_data_clean.groupby('date')[
                    self.chars_list].transform(
                    lambda x: 2 / (x.shape[0]+1) * x.rank() - 1)
        return self

    def ls_portfolio(self, return_col='ret_adj_ex',
                     chars=None, q=0.2, weight='equal'):
        """
        Constructs characteristics-sorted long/short portfolios
        They can be constructed in 3 key ways:
        1) equal-weighted: weight = None
        2) value-weighted: weight = "mve_m"
        3) score-weighted: weight = "score"

        :returns
        """

        if chars is None:
            chars = self.chars_list

        if self.chars_data_clean is None:
            print('No clean data available. Clean data first!')
            return self

        if weight == 'equal':
            self.factors = self.chars_data.groupby('date').apply(
                lambda x: x[chars].apply(
                    lambda z: x[z >= z.quantile(1 - q)][return_col].mean() -
                    x[z <= z.quantile(q)][return_col].mean()))
            return self

        elif weight == 'value':
            self.factors = self.chars_data.groupby('date').apply(
                lambda x: x[chars].apply(lambda z: np.average(
                    x[z >= z.quantile(1 - q)][return_col],
                    weights=x[z >= z.quantile(1 - q)]['mve_m']) - np.average(
                    x[z <= z.quantile(q)][return_col],
                    weights=x[z <= z.quantile(q)]['mve_m'])))
            return self

        elif weight == 'score':
            self.factors = self.chars_data.groupby('date').apply(
                lambda x: x[chars].apply(lambda z: np.average(
                    x[z >= z.quantile(1 - q)][return_col],
                    weights=x[z >= z.quantile(1 - q)][z.name]) - np.average(
                    x[z <= z.quantile(q)][return_col],
                    weights=x[z <= z.quantile(q)][z.name])))
            return self

    def save_chars(self, name, key, cleaned=True):
        print('Saving characteristics...')
        if cleaned:
            if self.chars_data_clean is None:
                print('No clean data available. Clean data first!')
            else:
                self.chars_data_clean.to_hdf(
                    f'./Data/{name}.h5', mode='a',
                    key=key, format='table', data_columns=True)
        else:
            self.chars_data.to_hdf(
                f'./Data/{name}.h5', mode='a', key=key,
                format='table', data_columns=True)
        print('... saving done!')

    def save_factors(self, name, key):
        print('Saving factor returns...')

        if self.factors is None:
            print('No factor returns available. Construct portfolios first!')
        else:
            self.factors.to_hdf(f'./Data/{name}.h5', mode='a',
                                key=key, format='table', data_columns=True)
        print('... saving done!')
