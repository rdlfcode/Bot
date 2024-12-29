import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

general = {
        "name": "testies",
        "best": False,
        "pct": False,
        "optuna": True,
        "target": "Last Price",
        "loss_name": "val_MAPE",
        "secondary_loss": "val_RMSE",
        "n_splits": 5,
        "n_trials": 50,
        "tuning_epochs": 100,
        "training_epochs": 10,
        "float_precision": ["medium", "high", "highest"][0],
        "plot_corrs": False,
        "validation": [None, "rolling"][0],
        "corr_thresh": 0.20,
        "train_test_pct": 0.75,
        "scaling_method": StandardScaler,
        "base_path": "config_history",
        "data_path": "Data/data.csv",
        "group_col_name": "Ticker",
        "time_col_name": "Datetime",
        "protected_columns": [ # non-data columns
                "Datetime",
                "Ticker",
        ],
}
data = {
    "alpha_vantage":    {
        "api_key": "O9KBU156UHKUVGV7",
        "rate_limit": 30, # calls per minute
        "base_url": "https://www.alphavantage.co/query",
        "default_dataset":      {
                "symbol_req":   {
                        'TIME_SERIES_DAILY_ADJUSTED': {
                                'outputsize': 'full'
                        },
                        'INCOME_STATEMENT': {
                                'interval': 'quarterly',
                        },
                        'BALANCE_SHEET': {
                        },
                        'CASH_FLOW': {
                        },
                        'EARNINGS': {
                        },
                },
                "other":        {
                        'REAL_GDP': {
                                'interval': 'quarterly'
                        },
                        'TREASURY_YIELD': {
                                'interval': 'daily',
                                'maturity': '3month'
                        },
                        'FEDERAL_FUNDS_RATE': {
                                'interval': 'daily'
                        },
                        'CPI': {
                        },
                        'INFLATION': {
                        },
                        'RETAIL_SALES': {
                        },
                        'DURABLES': {
                        },
                        'UNEMPLOYMENT': {
                        },
                        'NONFARM_PAYROLL': {
                        },
                },
        }
    },
    "polygon":  {
        "api_key": "xUnVyIQVTlGtwusxdouUkEddLed1QVv9",
    },
}
tft = {
        "name": "testies",
        "best": False,
        "pct": False,
        "target": "Last Price",
        "loss_name": "val_MAPE",
        "secondary_loss": "val_RMSE",
        "n_splits": 5,
        "float_precision": ["medium", "high", "highest"][0],
        "plot_corrs": False,
        "validation": [None, "rolling"][-1],
        "corr_thresh": 0.20,
        "train_test_pct": 0.75,
        "scaling_method": "z-score",
        "accelerator": 'cuda' if torch.cuda.is_available() else 'cpu',
        "base_path": "config_history",
        "data_path": "data.csv",
        "protected_columns": [
                "Datetime",
                "Ticker",
        ],
        "lstm_layers": 2,
        "batch_size": 48,
        "max_study_epochs": 30,
        "max_train_epochs": 50,
        "n_trials": 1,
        "output_size": 90,
        "learning_rate": 0.001,
        "max_prediction_length": 90,
        "hidden_size": 61,
        "max_encoder_length": 365,
        "feed_forward": [
                "GatedResidualNetwork",
                "GLU",
                "Bilinear",
                "ReGLU",
                "GEGLU",
                "SwiGLU",
                "ReLU",
                "GELU"
        ][0],
        "gradient_clip_val_range": [
                0.01,
                1.0
        ],
        "lstm_layers_range": [
                1,
                4
        ],
        "hidden_size_range": [
                1,
                100
        ],
        "dropout_range": [
                0.1,
                0.3
        ],
        "hidden_continuous_size_range": [
                1,
                100
        ],
        "attention_head_size_range": [
                1,
                16
        ],
        "learning_rate_range": [
                0.0001,
                1.0
        ],
        "limit_train_batches": 30,
        "log_every_n_steps": 30,
        "reduce_on_plateau_patience": 10,
        "timeout": 604800,
        "file": "model",
        "max_encoder_length_range": [
                1,
                8259
        ],
        "gradient_clip_val": 0.45,
        "dropout": 0.1257,
        "hidden_continuous_size": 23,
        "attention_head_size": 11,
        "model": "tft",
}

xgboost = {
        # feature to be forecasted
        'target':                       'Last Price',
        # n timesteps to forecast (usually days)
        'max_prediction_length':        45,
        # percent of data reserved for training
        'train_test_pct':               0.75,
        # Number of time steps to look back 
        'seq_size':                     300,
        'future':                       False,
        
        
        # MODEL SETTINGS
        ################
        # number of splits for cross validation
        'n_splits':                     7,
        # the initial prediction score of all instances (global bias)
        'base_score':                   0.5,
        # type of booster for random forest
        'booster':                      'gbtree',
        # number of trees
        'n_estimators':                 500,
        # error type
        'objective':                    'reg:squarederror',
        # maximum tree depth
        'max_depth':                    1,
        # learning rate (delta)
        'learning_rate':                0.15,
        # verbosity for printing
        'verbose':                      1000,
        # gpu vs cpu for evaluating prediction
        'predictor':                    'gpu_predictor',
        # computer cores allocated, -1 being all available
        'n_jobs':                       -1,
        # whether to shuffle inputs, no good for timeseries data
        'shuffle':                      False,
        # dimensionality of output space to be passed to next layer
        'n_units':                      128,
        # number of stacked models
        'main_layers':                  1,
        # number of following neural network layers
        'nn_layers':                    2
}



#######################
# Bloomberg Field IDs
#######################

# Corporate Financials
ids = {
        # Income Statement
        'Revenue':                              'SALES_REV_TURN',
        'Cost of Revenue':                      'IS_COGS_TO_FE_AND_PP_AND_G',
        'Gross Profit':                         'GROSS_PROFIT',
        'Other Operating Income':               'IS_OTHER_OPER_INC',
        'Operating Expenses':                   'IS_OPER_EXPN',       
        'Operating Income':                     'IS_OPER_INC',
        'Non-Operating Income Loss':            'NONOP_INCOME_LOSS',
        'Pretax Income':                        'PRETAX_INC',
        'Income Tax Expense':                   'IS_INC_TAX_EXP',
        'Income From Continuing Operations':    'IS_INC_FROM_XO_ITEM',
        'Net Extraordinary Losses':             'XO_GL_NET_OF_TAX',
        'Net Income':                           'NET_INCOME',
        'Basic Weighted Avr # of Shares':       'IS_AVR_NUM_SH_FOR_EPS',
        'Basic EPS, GAAP':                      'IS_EPS',
        'Diluted Weighted Avr # of Shares':     'IS_SH_FOR_DILUTED_EPS',
        'Diluted EPS, GAAP':                    'IS_DILUTED_EPS',
        'EBITDA':                               'EBITDA',
        'Trailing 12M EBITDA Margin':           'EBITDA_MARGIN',
        'Gross Margin':                         'GROSS_MARGIN',
        'Operating Margin':                     'OPER_MARGIN',
        'Profit Margin':                        'PROF_MARGIN',
        'Dividends per Share':                  'EQY_DPS',
          
        # Statement of Cash Flows
        'Net Income':                           'CF_NET_INC',
        'Depreciation & Amortization':          'CF_DEPR_AMORT',
        'Non-Cash Items':                       'NON_CASH_ITEMS_DETAILED',
#         'Stock-Based Compensation':             'CF_STOCK_BASED_COMPENSATION',
        'Deferred Income Taxes':                'CF_DEF_INC_TAX',
        'Other Non-Cash Adj':                   'OTHER_NON_CASH_ADJ_LESS_DETAILED',
        'Non-Cash Working Capital':             'CF_CHNG_NON_CASH_WORK_CAP',
        'Change in Accounts & Notes Receivable':'CF_ACCT_RCV_UNBILLED_REV',
        'Change in Inventories':                'CF_CHANGE_IN_INVENTORIES',
        'Change in Accounts Payable':           'CF_CHANGE_IN_ACCOUNTS_PAYABLE',
        'Change in Other':                      'INC_DEC_IN_OT_OP_AST_LIAB_DETAIL',
        'Cash from Operating Activities':       'CF_CASH_FROM_OPER',
        'Change in Fixed & Intangible Assets':  'CHG_IN_FXD_&_INTANG_AST_DETAILED',
        'Acquisition of Fixed & Intangibles':   'ACQUIS_FXD_&_INTANG_DETAILED',
#         'Net Cash from Acq/Div':                'CF_NT_CSH_RCVD_PD_FOR_ACQUIS_DIV',
        'Other Investing Activities':           'OTHER_INVESTING_ACT_DETAILED',
        'Cash from Investing Activities':       'CF_CASH_FROM_INV_ACT',
        'Dividends Paid':                       'CF_DVD_PAID',
        'Cash from Repaid Debt':                'PROC_FR_REPAYMNTS_BOR_DETAILED',
        'Cash from Repurchase of Equity':       'PROC_FR_REPURCH_EQTY_DETAILED',
        'Other Financing Activities':           'OTHER_FIN_AND_DEC_CAP',
        'Cash from Financing Activities':       'CFF_ACTIVITIES_DETAILED',
        'Net Changes in Cash':                  'CF_NET_CHNG_CASH',
        'Free Cash Flow':                       'CF_FREE_CASH_FLOW',
        'Free Cash Flow per Basic Share':       'FREE_CASH_FLOW_PER_SH',
        'Price to Free Cash Flow':              'PX_TO_FREE_CASH_FLOW',
        'Cash Flow to Net Income':              'CASH_FLOW_TO_NET_INC',
        
        # Balance Sheet
        'Cash, Cash Equivalents & STI':         'C&CE_AND_STI_DETAILED',
        'Accounts & Notes Receivable':          'BS_ACCT_NOTE_RCV',
        'Inventories':                          'BS_INVENTORIES',
        'Other ST Assets':                      'Total Current Assets',
        'Property, Plant & Equip':              'BS_NET_FIX_ASSET',
        'LT Investments & Receivables':         'BS_LT_INVEST',
        'Other LT Assets':                      'BS_OTHER_ASSETS_DEF_CHRG_OTHER',
        'Total Noncurrent Assets':              'BS_TOT_NON_CUR_ASSET',
        'Total Assets':                         'BS_TOT_ASSET',
        'Payables & Accruals':                  'ACCT_PAYABLE_&_ACCRUALS_DETAILED',
        'ST Debt':                              'BS_ST_BORROW',
        'Other ST Liabilities':                 'OTHER_CURRENT_LIABS_SUB_DETAILED',
        'Total Current Liabilities':            'BS_CUR_LIAB',
        'LT Debt':                              'BS_LT_BORROW',
        'Other LT Liabilities':                 'OTHER_NONCUR_LIABS_SUB_DETAILED',
        'Total Noncurrent Liabilities':         'NON_CUR_LIAB',
        'Total Liabilities':                    'BS_TOT_LIAB2',
        'Share Capital & APIC':                 'BS_SH_CAP_AND_APIC',
        'Other Equity':                         'OTHER_EQUITY_RATIO',
        'Equity Before Minority Interest':      'EQTY_BEF_MINORITY_INT_DETAILED',
        'Total Equity':                         'TOTAL_EQUITY',
        'Total Liabilities & Equity':           'TOT_LIAB_AND_EQY',
        'Net Debt':                             'NET_DEBT',
        'Cash Conversion Cycle':                'CASH_CONVERSION_CYCLE',
        'Number of Employees':                  'NUM_OF_EMPLOYEES',
   
        # Others
        'Last Price':                            'LAST_PRICE',
#         'Sales Estimate':                        'BEST_SALES',
        'Sales Actual':                          'IS_COMP_SALES',
#         'Net Income Estimate':                   'BEST_NET_GAAP',
        'Net Income Actual':                     'IS_COMP_NET_INCOME_GAAP',
        'Enterprise Value':                      'CURR_ENTP_VAL',
        'Price to Book':                         'PX_TO_BOOK_RATIO',
        'P/E':                                   'PE_RATIO',
        'Shares Outstanding':                    'EQY_SH_OUT',
        'Capital Expenditures':                  'CAPITAL_EXPEND'
}

# Gauge of the economy
eco_gauge = {
        # Inflation
        'Personal Consumption Expenditures':     'PCE CMOM Index',
        'CPI Urban Consumers YOY':               'CPI YOY Index',
        # Business Confidence
        'ISM Manufacturing':                     'NAPMPMI Index',
        # Unemployment
        'Change in Nonfarm Payrolls':            'NFP TCH Index',
        'Unemployment Rate':                     'USURTOT Index',
        # Housing
        'Housing Starts':                        'NHSPSTOT Index',
        # Trade
        'US Trade Balance':                      'USTBTOT Index'
}
