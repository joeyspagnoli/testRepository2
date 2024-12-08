from sdv.datasets.local import load_csvs
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot
import os


# Load CSVs from the folder
datasets = load_csvs(
    folder_name="C:\\Users\\jspag\\PycharmProjects\\syntheticModelTest\\files\\",
    read_csv_parameters={
        'skipinitialspace': True,
        'encoding': 'utf_8'
    }
)

# Extract the dataset by its filename (excluding extension)
data = datasets['test_measure']

# Automatically detect metadata from the dataframe
# metadata = Metadata.detect_from_dataframe(
#     data=data,
#     table_name='test_measure'
# )

# Save the detected metadata to a file
metadata_path = "path_to_metadata.json"
#metadata.save_to_json(metadata_path)
#print(f"Metadata detected and saved to {metadata_path}")

# Load metadata from the JSON file and validate it
try:
    metadata = Metadata.load_from_json(metadata_path)
    metadata.validate()
    print("Metadata is valid!")
except ValueError as e:
    print(f"Metadata validation failed: {e}")


# metadata.visualize(show_table_details='full',
#     show_relationship_labels=True,
#     output_filepath='my_metadata.png')

# Step 1: Create the synthesizer
#synthesizer = GaussianCopulaSynthesizer(metadata)

# Step 2: Train the synthesizer
#synthesizer.fit(data)

#print(synthesizer.get_parameters())
#print(synthesizer.get_metadata())
#print(synthesizer.get_learned_distributions())

# synthesizer.save(
#     filepath='my_synthesizer.pkl'
# )

synthesizer = GaussianCopulaSynthesizer.load(
    filepath='my_synthesizer.pkl'
)

synthetic_data = synthesizer.sample(num_rows=1000000)

# Define the target directory and base file name
output_dir = r"C:\Users\jspag\PycharmProjects\syntheticModelTest\files"
base_name = "Gaussian_synthetic_data"
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








