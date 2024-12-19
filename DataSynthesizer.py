from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot
import os
import pandas as pd


data = pd.read_pickle('test_measure_hr.pkl')


metadata_path = "HR_Only_path_to_metadata.json"

try:
    metadata = Metadata.load_from_json(metadata_path)
    metadata.validate()
    print("Metadata is valid!")
except ValueError as e:
    print(f"Metadata validation failed: {e}")

synthesizer = CTGANSynthesizer.load(
    filepath='my_CTGAN_synthesizer.pkl'
)

synthetic_data = synthesizer.sample(num_rows=576000)

# Define the target directory and base file name
output_dir = r"C:\Users\jspag\PycharmProjects\syntheticModelTest\files"
base_name = "CTGAN_synthetic_data"
extension = ".csv"

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

# Check for existing files and determine the next number
counter = 1
while os.path.exists(os.path.join(output_dir, f"{base_name}{counter}{extension}")):
    counter += 1

# Construct the unique file name
output_file_path = os.path.join(output_dir, f"{base_name}{counter}{extension}")

# Save the file
synthetic_data.to_csv(output_file_path, index=False)

print(f"Synthetic data saved to {output_file_path}")

diagnostic = run_diagnostic(
    real_data=data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

quality_report = evaluate_quality(
    data,
    synthetic_data,
    metadata
)

fig = get_column_plot(
    real_data=data,
    synthetic_data=synthetic_data,
    column_name='HR',
    metadata=metadata
)

fig.show()