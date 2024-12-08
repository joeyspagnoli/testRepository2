import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot
import os
import pandas as pd

data = pd.read_csv("C:\\Users\\jspag\\PycharmProjects\\syntheticModelTest\\files\\test_measure_hr.csv")

