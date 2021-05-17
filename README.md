# FactorData
A large number of PhD students who focus their research on empirical asset pricing are confronted with the problem of accessing and working with data that makes their research comparable to existing studies. In particular, this refers to research working with firm-level characteristics, which are either used directly in regression analyses or indirectly through the construction of sorted long-short portfolio returns. Unfortunately, most papers do not publish their code showing the data download, cleaning and variable definitions. Notable exceptions include Bryan Kelly (https://github.com/bkelly-lab/GlobalFactor) or Jeremiah Green (https://drive.google.com/file/d/0BwwEXkCgXEdRQWZreUpKOHBXOUU/view). In particular, the SAS code provided by Jeremiah Green is frequently cited in recent papers, and as PhD students we are extremely grateful that he made the code available. 

While I was working on my own research, I noticed that I would like to be able to download and clean the data directly in Python, and also to update specific variables according to more recent data availability: for example, in the SAS code published by Jeremiah Green ranges from 1980 to Dec 2016, but more data has become available since then, which requires manual changes to the code (this also include manually inserted updated CPI data).


My code provides a simple Python class that downloads, calculates, cleans and saves 103 firm characteristics using data from CRSP, Compustat, I/B/E/S, BLS and FRED. In particular, I follow the variable definitions used by Jeremiah Green and my code achieves an overall correlation of XYZ %. While there may exist different variable definitions, such as those used by Hou et al. 2020, I will leave different variable definitions to future updates. 

## Notable differences ##
While my overall correlation with Green's data is very high, there are some notable differences in variables definitions which I would like to point out:
- xsga0 (this is a helper variable): I do think that there is a small error in the SAS code:
```SAS
if missing(xsga) then xsga0=0;
							else xsga0=0
```

## Requirements ## 
For this code to work, there you must fulfill three key requirements:
1. You must have a valid WRDS account and have completed the pgpass file setup (see https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-from-your-computer/)
2. You must have a valid BLS account with access key (see https://data.bls.gov/registrationEngine/)  
3. You must have a valid FRED account with access key (see https://research.stlouisfed.org/docs/api/api_key.html)


## Example ##

```python
factors = FactorData(wrds_username='janedoe', bls_key='1234', fred_key='abcd', start_yr=1980)
factors.get_data()
factors.clean_data(how='std')
factors.save_data(name='data', key='std')
```


## Disclaimer ##
Even though this code achieves a very high correlation with Jermiah Green's SAS code, I do not claim that my code is free of errors. Therefore, I am grateful for any feedback or constructive suggestion for improvement.
