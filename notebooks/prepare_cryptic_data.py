# Set up path to import parent modules
from pathlib import Path
import sys  

# Add to sys.path
sys.path.insert(0, str(Path().resolve().parents[1]))

# Preprosses cryptic seq data to normalize by on target
import cryptic.utils as utils
from cryptic.utils import cs_excel_data

# Flag to trigger data export from excel files
export_excel_data = True

cs_data_file = '../data/TB000208a.outputs.xlsx'
data_path = '../data/TB000208a'
train_sheets = ['GT-Rep1-N7_S1','GT-Rep2-N7_S2','GT-Rep3-N7_S3']
test_sheets = ['Pool-Rep1-N7_S4','Pool-Rep2-N7_S5','Pool-Rep3-N7_S6']

if export_excel_data:
    utils.cs_excel_data.extract_excel_cs_data(cs_data_file, train_sheets, data_path, 'fit.csv', threshold=0.001)
    sites = utils.cs_excel_data.extract_excel_cs_data(cs_data_file, test_sheets, data_path, 'test.csv', threshold=0.001, dn_exclusion=['GT','AC'])