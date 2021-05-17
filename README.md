# FactorData
A large number of PhD students who focus their research on empirical asset pricing are confronted with the problem of accessing and working with data that makes their research comparable to existing studies. In particular, this refers to research working with firm-level characteristics, which are either used directly in regression analyses or indirectly through the construction of sorted long-short portfolio returns. Unfortunately, most papers do not publish their code showing the data download, cleaning and variable definitions. Notable exceptions include Bryan Kelly (https://github.com/bkelly-lab/GlobalFactor) or Jeremiah Green (https://drive.google.com/file/d/0BwwEXkCgXEdRQWZreUpKOHBXOUU/view). In particular, the SAS code provided by Jeremiah Green is frequently cited in recent papers, and as PhD students we are extremely grateful that he made the code available. 

While I was working on my own research, I noticed that I would like to be able to download and clean the data directly in Python, and also to update specific variables according to more recent data availability: for example, in the SAS code published by Jeremiah Green ranges from 1980 to Dec 2016, but more data has become available since then, which requires manual changes to the code (this also include manually inserted updated CPI data).


My code provides a simple Python class that downloads, calculates, cleans and saves 103 firm characteristics using data from CRSP, Compustat, I/B/E/S, BLS and FRED. In particular, I follow the variable definitions used by Jeremiah Green and my code achieves an overall correlation of XYZ %. While there may exist different variable definitions, such as those used by Hou et al. 2020, I will leave different variable definitions to future updates. 

PostgreSQL

## Notable differences ##
While my overall correlation with Green's data is very high, there are some notable differences in variables definitions as well as generally diverging aspects which I would like to point out. Note that everyone can change the code in a way that suits best their own needs. This respirotry is merely a suggestion! 
1. I perform industry adjustments after the CRSP-Compustat merge not before
2. Industry-adjustments are performed with the stocks from the investment universe only
3. 



- xsga0 (this is a helper variable): Green's definition sets the variable to 0
```SAS
if missing(xsga) then xsga0=0; else xsga0=0
```
, whereas mine is 
```PostgreSQL
CASE WHEN xsga is null 
     THEN 0 
     ELSE xsga
     END AS xsga0
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

## Results ## 

The plot below shows the data availability per month from January 1980 to December 2020

The plot below shows the two different versions of the example book-to-market of an arbitrarily sampled stock:

the plot shows the standardised and rank-normalised 

## Disclaimer ##
Even though this code achieves a very high correlation with Jermiah Green's SAS code, I do not claim that my code is free of errors. Therefore, I am grateful for any feedback or constructive suggestion for improvement.
