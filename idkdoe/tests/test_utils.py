import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock, call, ANY
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for tests
import matplotlib.pyplot as plt
# No need to import pandas.plotting directly in the test file if patching within utils
# import pandas.plotting

# Import the functions to be tested
from idkdoe.utils import _save_results, _scale_samples, _visualize_samples

class TestIdkDoeUtils(unittest.TestCase):

    def setUp(self):
        """Set up common resources for tests."""
        self.output_path = "test_output_dir"
        self.output_prefix = "test_results"
        self.sample_data = [
            {'var1': 1, 'var2': 10, 'var3': 5},
            {'var1': 2, 'var2': 20, 'var3': 5},
            {'var1': 3, 'var2': 30, 'var3': 5},
        ]
        self.variable_names = ['var1', 'var2', 'var3']

    def tearDown(self):
        """Clean up any created directories (if not fully mocked)."""
        pass # Mocking handles filesystem interactions

    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_results_with_outputs(self, mock_to_csv, mock_makedirs):
        """Test saving results when both input and output dataframes exist."""
        mock_model = MagicMock()
        mock_model.inputs_df = pd.DataFrame({'a': [1, 2]})
        mock_model.outputs_df = pd.DataFrame({'b': [3, 4]})

        _save_results(mock_model, self.output_path, self.output_prefix)

        mock_makedirs.assert_called_once_with(self.output_path, exist_ok=True)

        expected_inputs_path = os.path.join(self.output_path, f"{self.output_prefix}_inputs.csv")
        expected_outputs_path = os.path.join(self.output_path, f"{self.output_prefix}_outputs.csv")

        self.assertEqual(mock_to_csv.call_count, 2)
        mock_to_csv.assert_has_calls([
            call(expected_inputs_path, index=False),
            call(expected_outputs_path, index=False)
        ], any_order=True)

    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_results_empty_outputs(self, mock_to_csv, mock_makedirs):
        """Test saving results when the output dataframe is empty."""
        mock_model = MagicMock()
        mock_model.inputs_df = pd.DataFrame({'a': [1, 2]})
        mock_model.outputs_df = pd.DataFrame()

        _save_results(mock_model, self.output_path, self.output_prefix)

        mock_makedirs.assert_called_once_with(self.output_path, exist_ok=True)
        expected_inputs_path = os.path.join(self.output_path, f"{self.output_prefix}_inputs.csv")
        mock_to_csv.assert_called_once_with(expected_inputs_path, index=False)

    def test_scale_samples_basic(self):
        """Test basic scaling functionality."""
        scaled_df = _scale_samples(self.sample_data)

        self.assertIsInstance(scaled_df, pd.DataFrame)
        self.assertEqual(scaled_df.shape, (3, 3))
        self.assertTrue((scaled_df >= 0).all().all())
        self.assertTrue((scaled_df <= 1).all().all())

        pd.testing.assert_series_equal(scaled_df['var1'], pd.Series([0.0, 0.5, 1.0]), check_names=False)
        pd.testing.assert_series_equal(scaled_df['var2'], pd.Series([0.0, 0.5, 1.0]), check_names=False)
        pd.testing.assert_series_equal(scaled_df['var3'], pd.Series([0.5, 0.5, 0.5]), check_names=False)

    def test_scale_samples_single_point(self):
        """Test scaling with only one sample point."""
        single_sample = [{'var1': 10, 'var2': 20}]
        scaled_df = _scale_samples(single_sample)

        self.assertIsInstance(scaled_df, pd.DataFrame)
        self.assertEqual(scaled_df.shape, (1, 2))
        pd.testing.assert_series_equal(scaled_df['var1'], pd.Series([0.5]), check_names=False)
        pd.testing.assert_series_equal(scaled_df['var2'], pd.Series([0.5]), check_names=False)

    # Patch where functions are looked up (within idkdoe.utils)
    @patch('idkdoe.utils._scale_samples')
    @patch('idkdoe.utils.scatter_matrix') # Corrected patch target
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.suptitle')
    def test_visualize_samples_scatter(self, mock_suptitle, mock_show, mock_scatter_matrix, mock_scale):
        """Test visualization uses scatter_matrix for few variables."""
        mock_scaled_df = pd.DataFrame({'var1_scaled': [0, 1], 'var2_scaled': [1, 0]})
        mock_scale.return_value = mock_scaled_df

        _visualize_samples(self.sample_data, self.variable_names)

        mock_scale.assert_called_once_with(self.sample_data)
        # Assert call to the mocked scatter_matrix
        mock_scatter_matrix.assert_called_once_with(mock_scaled_df, diagonal='hist', alpha=0.7)
        mock_suptitle.assert_called_once()
        mock_show.assert_called_once()

    # Patch where functions are looked up (within idkdoe.utils)
    @patch('idkdoe.utils._scale_samples')
    @patch('idkdoe.utils.parallel_coordinates') # Corrected patch target
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.xticks')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.show')
    def test_visualize_samples_parallel(self, mock_show, mock_tight_layout, mock_legend, mock_ylabel, mock_xlabel, mock_title, mock_xticks, mock_figure, mock_parallel_coords, mock_scale):
        """Test visualization uses parallel_coordinates for many variables."""
        many_vars_data = [{f'v{i}': i*j for i in range(7)} for j in range(3)]
        many_vars_names = [f'v{i}' for i in range(7)]
        mock_scaled_df = pd.DataFrame({name: np.random.rand(3) for name in many_vars_names})
        mock_scale.return_value = mock_scaled_df

        _visualize_samples(many_vars_data, many_vars_names)

        mock_scale.assert_called_once_with(many_vars_data)

        # Check that the explicit figure call was made with expected args
        mock_figure.assert_any_call(figsize=(12, 6)) # Use assert_any_call or assert_called_with

        # Check that parallel_coordinates was called correctly
        call_args, call_kwargs = mock_parallel_coords.call_args
        called_df = call_args[0]
        self.assertIn('__id', called_df.columns)
        pd.testing.assert_frame_equal(called_df.drop(columns=['__id']), mock_scaled_df)
        self.assertEqual(call_kwargs.get('class_column'), '__id')
        self.assertEqual(call_kwargs.get('cols'), many_vars_names)
        # Check for the colormap argument passed from _visualize_samples
        self.assertIsNotNone(call_kwargs.get('colormap'))

        # Check other plotting calls
        mock_xticks.assert_called_once()
        mock_title.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_legend.assert_called_once()
        mock_tight_layout.assert_called_once()
        mock_show.assert_called_once()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

