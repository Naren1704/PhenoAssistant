import os
import sys
import glob
import json
import copy
import random
import requests
from typing import Annotated, Literal, List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging

from functions.instance_segmentation import infer_instance_segmentation
from functions.image_classification import finetune_image_classification, infer_image_classification
from functions.create_hf_dataset import prepare_dataset, get_dataset_format
from functions.search import search_and_scrape
from functions.compute_phenotypes import compute_phenotypes_from_ins_seg
from functions.stat_test import perform_anova, perform_tukey_test
from functions.generic_tools import (
    set_env_vars, 
    set_random_seed, 
    print_trainable_parameters, 
    handle_grayscale_image, 
    download_hffile, 
    load_images, 
    get_model_zoo,
    calculator,
    make_dir,
    extract_column_name_from_csv,
)

from agents import compute_csv as compute_from_csv
from agents import coding, plot_from_csv

def ara_crop_pipeline(metadata_path: Annotated[str, 'Path to the metadata JSON file'], 
                      output_dir: Annotated[str, 'Directory to save the results'], 
                      pixel_to_cm: Annotated[float, 'Pixel to cm mapping scale'] = 0.03) -> Dict[str, Any]:
    """
    Pipeline to compute phenotypes for Arabidopsis plant images, merge with metadata, and save the results.
    Args:
        metadata_path (str): Path to the metadata JSON file.
        output_dir (str): Directory to save the results.
        pixel_to_cm (float): Pixel to cm mapping scale. Default is 0.03.
    Returns:
        Dict[str, Any]: Status and path to the saved results.
    """

    import json
    import pandas as pd

    try:
        logging.info("Starting ara_crop_pipeline execution.")

        # Step 1: Check available checkpoints for instance segmentation
        logging.info("Checking available checkpoints for instance segmentation.")
        checkpoints = get_model_zoo()
        selected_checkpoint = "fengchen025/arabidopsis_leaf-instance-segmentation_cvppp2017-a1a4_m2fb_fullft"

        # Step 2: Perform instance segmentation
        logging.info("Performing instance segmentation.")
        infer_instance_segmentation(file_path=metadata_path, checkpoint=selected_checkpoint, output_dir=output_dir)

        # Step 3: Compute phenotypes from instance segmentation results
        logging.info("Computing phenotypes from instance segmentation results.")
        ins_seg_result_path = f"{output_dir}/ins_seg_results.json"
        phenotypes_save_path = f"{output_dir}/phenotypes.csv"
        compute_phenotypes_from_ins_seg(ins_seg_result_path=ins_seg_result_path, save_path=phenotypes_save_path, pixel_to_cm=pixel_to_cm)

        # Step 4: Merge computed phenotypes with metadata
        logging.info("Merging computed phenotypes with metadata.")
        phenotypes_df = pd.read_csv(phenotypes_save_path)
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        metadata_df = pd.DataFrame(metadata)
        merged_df = pd.merge(phenotypes_df, metadata_df, on='file_name')
        final_save_path = f"{output_dir}/aracrop_phenotypes.csv"
        merged_df.to_csv(final_save_path, index=False)

        logging.info("Pipeline executed successfully.")
        return {"status": "success", "result_path": final_save_path}

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def ara_crop_pipeline_2(metadata_path: Annotated[str, 'Path to the metadata JSON file'], 
                        output_dir: Annotated[str, 'Directory path to save results'], 
                        pixel_to_cm: Annotated[float, 'Scale to map pixel values to cm'] = 0.03) -> Dict[str, Any]:
    """
    Pipeline to compute phenotypes for Arabidopsis plant images, merge with metadata, and save results.

    Args:
        metadata_path (str): Path to the metadata JSON file.
        output_dir (str): Directory path to save results.
        pixel_to_cm (float): Scale to map pixel values to cm. Default is 0.03.

    Returns:
        Dict[str, Any]: Status and path of the saved CSV file.
    """

    import pandas as pd
    import json

    try:
        # Step 1: Perform instance segmentation
        ins_seg_result_path = f"{output_dir}/ins_seg_results.json"
        infer_instance_segmentation(file_path=metadata_path, 
                                    checkpoint="fengchen025/arabidopsis_leaf-instance-segmentation_cvppp2017-a1a4_m2fb_fullft", 
                                    output_dir=output_dir)

        # Step 2: Compute phenotypes
        phenotypes_path = f"{output_dir}/phenotypes.csv"
        compute_phenotypes_from_ins_seg(ins_seg_result_path=ins_seg_result_path, 
                                        save_path=phenotypes_path, 
                                        pixel_to_cm=pixel_to_cm)

        # Step 3: Merge phenotypes with metadata
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        metadata_df = pd.DataFrame(metadata)
        phenotypes_df = pd.read_csv(phenotypes_path)
        merged_df = pd.merge(phenotypes_df, metadata_df, on='file_name')

        # Step 4: Save merged data
        final_output_path = f"{output_dir}/aracrop_phenotypes.csv"
        merged_df.to_csv(final_output_path, index=False)

        return {"status": "success", "result": final_output_path}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def ara_crop_pipeline_3(metadata_path: Annotated[str, 'Path to the metadata JSON file'] = './data/aracrop_metadata.json',
                        output_dir: Annotated[str, 'Directory to save results'] = './results/Case1',
                        pixel_to_cm: Annotated[float, 'Scale to map pixel values to cm'] = 0.03) -> Dict[str, Any]:
    """
    Pipeline to compute phenotypes from Arabidopsis plant images and merge with metadata.
    Returns:
        Dict[str, Any]: Status and path of the saved CSV file containing merged phenotypes and metadata.
    """

    import pandas as pd
    import json

    try:
        # Step 1: Perform instance segmentation
        infer_instance_segmentation(file_path=metadata_path,
                                    checkpoint="fengchen025/arabidopsis_leaf-instance-segmentation_cvppp2017-a1a4_m2fb_fullft",
                                    output_dir=output_dir)

        # Step 2: Compute phenotypes from instance segmentation results
        compute_phenotypes_from_ins_seg(ins_seg_result_path=f"{output_dir}/ins_seg_results.json",
                                        save_path=f"{output_dir}/phenotypes.csv",
                                        pixel_to_cm=pixel_to_cm)

        # Step 3: Merge computed phenotypes with metadata
        phenotypes_df = pd.read_csv(f'{output_dir}/phenotypes.csv')
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        metadata_df = pd.DataFrame(metadata)
        merged_df = pd.merge(phenotypes_df, metadata_df, on='file_name')
        merged_df.to_csv(f'{output_dir}/aracrop_phenotypes.csv', index=False)

        return {"status": "success", "result": f"{output_dir}/aracrop_phenotypes.csv"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def ara_crop_pipeline_4(metadata_path: Annotated[str, 'Path to the metadata JSON file'] = './data/aracrop_metadata.json',
                        output_dir: Annotated[str, 'Directory to save results'] = './results/Case1',
                        pixel_to_cm: Annotated[float, 'Scale to map pixel values to cm'] = 0.03) -> Dict[str, Any]:
    """
    Pipeline to compute phenotypes for Arabidopsis plant images and merge with metadata.
    Returns:
        Dict[str, Any]: Status and path to the final merged CSV file.
    """

    import pandas as pd
    import json

    try:
        logging.info("Starting ara_crop_pipeline_4 execution.")

        # Step 1: Perform instance segmentation
        logging.info("Executing instance segmentation.")
        infer_instance_segmentation(
            file_path=metadata_path,
            checkpoint="fengchen025/arabidopsis_leaf-instance-segmentation_cvppp2017-a1a4_m2fb_fullft",
            output_dir=output_dir
        )

        # Step 2: Compute phenotypes from instance segmentation results
        logging.info("Computing phenotypes from instance segmentation results.")
        compute_phenotypes_from_ins_seg(
            ins_seg_result_path=f"{output_dir}/ins_seg_results.json",
            save_path=f"{output_dir}/phenotypes.csv",
            pixel_to_cm=pixel_to_cm
        )

        # Step 3: Merge computed phenotypes with metadata
        logging.info("Merging computed phenotypes with metadata.")
        phenotypes_df = pd.read_csv(f"{output_dir}/phenotypes.csv")
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        metadata_df = pd.DataFrame(metadata)
        merged_df = pd.merge(phenotypes_df, metadata_df, on='file_name')
        merged_df.to_csv(f"{output_dir}/aracrop_phenotypes.csv", index=False)

        logging.info("Pipeline executed successfully.")
        return {"status": "success", "result": f"{output_dir}/aracrop_phenotypes.csv"}

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def ara_crop_pipeline_5(metadata_path: Annotated[str, 'Path to the metadata JSON file'], 
                        output_dir: Annotated[str, 'Directory to save results']) -> Dict[str, Any]:
    """
    Pipeline to compute phenotypes for Arabidopsis plant images, merge with metadata, and save results.

    Args:
        metadata_path (str): Path to the metadata JSON file.
        output_dir (str): Directory to save results.

    Returns:
        Dict[str, Any]: Status and path of the saved CSV file.
    """

    import pandas as pd
    import json
    import os

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Instance Segmentation
        ins_seg_result_path = os.path.join(output_dir, "ins_seg_results.json")
        infer_instance_segmentation(file_path=metadata_path, 
                                    checkpoint="fengchen025/arabidopsis_leaf-instance-segmentation_cvppp2017-a1a4_m2fb_fullft", 
                                    output_dir=output_dir)

        # Step 2: Compute Phenotypes
        phenotypes_path = os.path.join(output_dir, "phenotypes.csv")
        compute_phenotypes_from_ins_seg(ins_seg_result_path=ins_seg_result_path, 
                                        save_path=phenotypes_path, 
                                        pixel_to_cm=0.03)

        # Step 3: Merge Metadata
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        metadata_df = pd.DataFrame(metadata)
        phenotypes_df = pd.read_csv(phenotypes_path)
        merged_df = pd.merge(phenotypes_df, metadata_df, on='file_name')

        # Step 4: Save Results
        final_csv_path = os.path.join(output_dir, "aracrop_phenotypes.csv")
        merged_df.to_csv(final_csv_path, index=False)

        return {"status": "success", "result_path": final_csv_path}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def ara_crop_plot(data_path: Annotated[str, 'Path to the input CSV file containing phenotype data'], output_dir: Annotated[str, 'Directory path to save the plots']) -> Dict[str, str]:
    """
    Pipeline to compute and plot the mean and standard deviation of phenotypes for each ecotype over time.
    The phenotypes include leaf count, projected leaf area (PLA), plant diameter, and plant perimeter.
    Each plot is saved in the specified output directory with appropriate titles, labels, and legends.

    Args:
        data_path (str): Path to the input CSV file containing phenotype data.
        output_dir (str): Directory path to save the plots.

    Returns:
        Dict[str, str]: A dictionary containing the paths to the saved plots.
    """

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define phenotypes and their corresponding file names
    phenotypes = {
        "leaf_count": "leaf_count",
        "projected leaf area (PLA)": "pla",
        "plant diameter": "plant_diameter",
        "plant perimeter": "plant_perimeter"
    }

    # Compute mean and standard deviation for each phenotype
    for phenotype, file_prefix in phenotypes.items():
        compute_from_csv(
            message=f"Compute the mean and standard deviation of {phenotype} for each ecotype on every day after sowing.",
            file_path=data_path,
            save_path=f"{output_dir}/{file_prefix}_stats.csv"
        )

    # Plot the results
    for phenotype, file_prefix in phenotypes.items():
        plot_from_csv(
            message=f"Plot the mean and standard deviation of {phenotype} for each ecotype over time. "
                    f"Use shades to represent the standard deviation. "
                    f"Set plot title as 'Mean and STD of {phenotype} for each ecotype over time'. "
                    f"Set x-axis label as 'Days after sowing'. "
                    f"Set y-axis label as '{phenotype}'. "
                    f"Set legend to show the ecotype names. Do not use grid lines.",
            file_path=f"{output_dir}/{file_prefix}_stats.csv",
            save_path=f"{output_dir}/{file_prefix}_plot.png"
        )

    # Return paths to the saved plots
    return {phenotype: f"{output_dir}/{file_prefix}_plot.png" for phenotype, file_prefix in phenotypes.items()}


def ara_crop_plot_2(data_path: Annotated[str, 'Path to the input CSV file containing phenotype data'], output_dir: Annotated[str, 'Directory path to save the plots']) -> Dict[str, Any]:
    """
    Pipeline to compute and plot the mean and standard deviation of phenotypes (leaf count, projected leaf area, plant diameter, plant perimeter) for each ecotype on every day after sowing.
    Returns:
        Dict[str, Any]: A dictionary containing the status and paths of the saved plots.
    """

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    def make_dir(dir_path):
        """Check if a directory exists, and create it if it does not."""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            return f"Directory '{dir_path}' created."
        else:
            return f"Directory '{dir_path}' already exists."

    def compute_from_csv(message, file_path, save_path):
        """Compute statistics or new values from a CSV file and save the results."""
        data = pd.read_csv(file_path)
        # Assuming the message specifies the computation of mean and std for each ecotype and day
        result = data.groupby(['ecotype', 'days_after_sowing']).agg(['mean', 'std']).reset_index()
        result.columns = ['ecotype', 'days_after_sowing', 'mean', 'std']
        result.to_csv(save_path, index=False)
        return f"The new csv is saved at {save_path}."

    def plot_from_csv(message, file_path, save_path):
        """Plot data from a CSV file and save the plot."""
        data = pd.read_csv(file_path)
        plt.figure(figsize=(10, 6))
        for ecotype in data['ecotype'].unique():
            ecotype_data = data[data['ecotype'] == ecotype]
            plt.plot(ecotype_data['days_after_sowing'], ecotype_data['mean'], label=ecotype)
            plt.fill_between(ecotype_data['days_after_sowing'],
                             ecotype_data['mean'] - ecotype_data['std'],
                             ecotype_data['mean'] + ecotype_data['std'],
                             alpha=0.2)
        plt.title(message.split('.')[0])
        plt.xlabel('Days after sowing')
        plt.ylabel(message.split(' ')[-1])
        plt.legend(title='Ecotype')
        plt.savefig(save_path)
        plt.close()
        return f"Plot saved at {save_path}."

    # Ensure the output directory exists
    make_dir(output_dir)

    # Define phenotypes and their corresponding file names
    phenotypes = {
        "leaf count": "leaf_count",
        "projected leaf area (PLA)": "pla",
        "plant diameter": "plant_diameter",
        "plant perimeter": "plant_perimeter"
    }

    # Compute and plot for each phenotype
    results = {}
    for phenotype, file_prefix in phenotypes.items():
        stats_file = os.path.join(output_dir, f"{file_prefix}_stats.csv")
        plot_file = os.path.join(output_dir, f"{file_prefix}_plot.png")

        # Compute mean and std
        compute_message = f"Compute the mean and standard deviation of {phenotype} for each ecotype on every day after sowing."
        compute_from_csv(compute_message, data_path, stats_file)

        # Plot the results
        plot_message = f"Plot the mean and standard deviation of {phenotype} for each ecotype over time. Use shades to represent the standard deviation. Set plot title as 'Mean and STD of {phenotype} for each ecotype over time'. Set x-axis label as 'Days after sowing'. Set y-axis label as '{phenotype}'. Set legend to show the ecotype names. Do not use grid lines."
        plot_from_csv(plot_message, stats_file, plot_file)

        results[phenotype] = plot_file

    return {"status": "success", "plots": results}


def ara_crop_plot_3(data_path: Annotated[str, 'Path to the input CSV file containing phenotype data'], output_dir: Annotated[str, 'Directory path to save the plots']) -> Dict[str, str]:
    """
    Pipeline to compute and plot the mean and standard deviation of phenotypes [leaf count, projected leaf area (PLA), plant diameter, plant perimeter]
    for each ecotype on every day after sowing. Saves one plot for each phenotype in the specified directory.

    Args:
        data_path (str): Path to the input CSV file containing phenotype data.
        output_dir (str): Directory path to save the plots.

    Returns:
        Dict[str, str]: A dictionary with keys as phenotype names and values as paths to the saved plots.
    """

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define phenotypes and their corresponding file names
    phenotypes = {
        "leaf_count": "leaf_count_stats.csv",
        "projected_leaf_area": "pla_stats.csv",
        "plant_diameter": "plant_diameter_stats.csv",
        "plant_perimeter": "plant_perimeter_stats.csv"
    }

    # Compute mean and standard deviation for each phenotype
    for phenotype, stats_file in phenotypes.items():
        compute_from_csv(
            message=f"Compute the mean and standard deviation of {phenotype.replace('_', ' ')} for each ecotype on every day after sowing.",
            file_path=data_path,
            save_path=os.path.join(output_dir, stats_file)
        )

    # Plot each phenotype
    plot_paths = {}
    for phenotype, stats_file in phenotypes.items():
        plot_path = os.path.join(output_dir, f"{phenotype}_plot.png")
        plot_from_csv(
            message=f"Plot the mean and standard deviation of {phenotype.replace('_', ' ')} for each ecotype over time. Use shades to represent the standard deviation. Set plot title as 'Mean and STD of {phenotype.replace('_', ' ')} for each ecotype over time'. Set x-axis label as 'Days after sowing'. Set y-axis label as '{phenotype.replace('_', ' ')}'. Set legend to show the ecotype names. Do not use grid lines.",
            file_path=os.path.join(output_dir, stats_file),
            save_path=plot_path
        )
        plot_paths[phenotype] = plot_path

    return plot_paths


def ara_crop_plot_4(data_path: Annotated[str, 'Path to the input CSV file containing phenotypes data'], output_dir: Annotated[str, 'Directory path to save the plots']) -> Dict[str, str]:
    """
    Pipeline to compute and plot the mean and standard deviation of phenotypes for each ecotype over time.
    Returns:
        Dict[str, str]: Paths to the saved plots for each phenotype.
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    from typing import Dict

    # Ensure the output directory exists
    make_dir(dir_path=output_dir)

    # Compute statistics for each phenotype and save to CSV
    phenotypes = ['leaf count', 'projected leaf area (PLA)', 'plant diameter', 'plant perimeter']
    stats_files = {}
    for phenotype in phenotypes:
        save_path = f"{output_dir}/{phenotype.replace(' ', '_')}_stats.csv"
        compute_from_csv(
            message=f"Compute the mean and standard deviation of {phenotype} for each ecotype on every day after sowing.",
            file_path=data_path,
            save_path=save_path
        )
        stats_files[phenotype] = save_path

    # Plot each phenotype
    plot_paths = {}
    for phenotype, stats_file in stats_files.items():
        plot_path = f"{output_dir}/{phenotype.replace(' ', '_')}_plot.png"
        plot_from_csv(
            message=f"Plot the mean and standard deviation of {phenotype} for each ecotype over time. Use shades to represent the standard deviation. Set plot title as 'Mean and STD of {phenotype} for each ecotype over time'. Set x-axis label as 'Days after sowing'. Set y-axis label as '{phenotype}'. Set legend to show the ecotype names. Do not use grid lines.",
            file_path=stats_file,
            save_path=plot_path
        )
        plot_paths[phenotype] = plot_path

    return plot_paths


def ara_crop_plot_5(data_path: Annotated[str, 'Path to the input CSV file containing phenotypes data.'], output_dir: Annotated[str, 'Directory path to save the plots.']) -> Dict[str, Any]:
    """
    Pipeline to compute and plot the mean and standard deviation of phenotypes for each ecotype over time.
    Returns:
        Dict[str, Any]: A dictionary containing the status and paths of the saved plots.
    """

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    try:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Define phenotypes and their corresponding file names
        phenotypes = {
            "leaf_count": "leaf_count_stats.csv",
            "pla": "pla_stats.csv",
            "plant_diameter": "plant_diameter_stats.csv",
            "plant_perimeter": "plant_perimeter_stats.csv"
        }

        # Compute mean and standard deviation for each phenotype
        for phenotype, file_name in phenotypes.items():
            compute_from_csv(
                message=f"Compute the mean and standard deviation of {phenotype.replace('_', ' ')} for each ecotype on every day after sowing.",
                file_path=data_path,
                save_path=os.path.join(output_dir, file_name)
            )

        # Plot the results for each phenotype
        for phenotype, file_name in phenotypes.items():
            plot_from_csv(
                message=f"Plot the mean and standard deviation of {phenotype.replace('_', ' ')} for each ecotype over time. Use shades to represent the standard deviation. Set plot title as 'Mean and STD of {phenotype.replace('_', ' ')} for each ecotype over time'. Set x-axis label as 'Days after sowing'. Set y-axis label as '{phenotype.replace('_', ' ')}'. Set legend to show the ecotype names. Do not use grid lines.",
                file_path=os.path.join(output_dir, file_name),
                save_path=os.path.join(output_dir, f"{phenotype}_plot.png")
            )

        return {
            "status": "success",
            "plots": {phenotype: os.path.join(output_dir, f"{phenotype}_plot.png") for phenotype in phenotypes}
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def potato_crop_pipeline_1(metadata_file_path: Annotated[str, 'Path to the metadata CSV file'] = './data/potato_metadata.csv',
                           output_dir: Annotated[str, 'Directory to save results'] = './results/Case2_Task1_wlog') -> Dict[str, Any]:
    """
    Pipeline to compute the projected leaf area (PLA) for potato plant images, merge with metadata, and save the results.
    Returns:
        Dict[str, Any]: Status and path to the output file containing merged data.
    """

    import os
    import pandas as pd

    try:
        logging.info("Starting potato crop pipeline execution.")

        # Step 1: Check and create output directory
        logging.info("Checking and creating output directory if not exists.")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Step 2: Perform instance segmentation
        logging.info("Performing instance segmentation on potato plant images.")
        infer_instance_segmentation(file_path=metadata_file_path,
                                    checkpoint='potato_leaf-instance-segmentation_leaf-only-sam',
                                    output_dir=output_dir)

        # Step 3: Compute phenotypes from instance segmentation results
        logging.info("Computing phenotypes from instance segmentation results.")
        compute_phenotypes_from_ins_seg(ins_seg_result_path=f"{output_dir}/leaf_only_sam_results.json",
                                        save_path=f"{output_dir}/computed_phenotypes.csv")

        # Step 4: Merge computed phenotypes with metadata
        logging.info("Merging computed phenotypes with metadata.")
        computed_phenotypes_path = f"{output_dir}/computed_phenotypes.csv"
        computed_phenotypes_df = pd.read_csv(computed_phenotypes_path)

        metadata_df = pd.read_csv(metadata_file_path)

        merged_df = pd.merge(computed_phenotypes_df, metadata_df, on='file_name')

        # Step 5: Save the merged data
        output_file_path = f"{output_dir}/potato_phenotypes_and_metadata.csv"
        logging.info(f"Saving merged data to {output_file_path}.")
        merged_df.to_csv(output_file_path, index=False)

        logging.info("Potato crop pipeline executed successfully.")
        return {"status": "success", "output_file": output_file_path}

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def potato_crop_pipeline_2(metadata_file_path: Annotated[str, 'Path to the metadata CSV file'] = './data/potato_metadata.csv',
                           output_dir: Annotated[str, 'Directory path to save results'] = './results/Case2_Task1_wlog') -> Dict[str, Any]:
    """
    Pipeline to compute the projected leaf area (PLA) for potato plant images, merge with metadata, and save the results.
    Returns:
        Dict[str, Any]: A dictionary containing the status and path to the output file.
    """

    import os
    import pandas as pd

    try:
        logging.info("Starting potato crop pipeline execution.")

        # Step 1: Check and create output directory
        logging.info("Checking and creating output directory if it doesn't exist.")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Step 2: Perform instance segmentation
        logging.info("Performing instance segmentation on potato images.")
        infer_instance_segmentation(
            file_path=metadata_file_path,
            checkpoint="potato_leaf-instance-segmentation_leaf-only-sam",
            output_dir=output_dir
        )

        # Step 3: Compute phenotypes from instance segmentation results
        logging.info("Computing phenotypes from instance segmentation results.")
        compute_phenotypes_from_ins_seg(
            ins_seg_result_path=f"{output_dir}/leaf_only_sam_results.json",
            save_path=f"{output_dir}/computed_phenotypes.csv"
        )

        # Step 4: Merge computed phenotypes with metadata
        logging.info("Merging computed phenotypes with metadata.")
        computed_phenotypes_path = f"{output_dir}/computed_phenotypes.csv"
        computed_phenotypes_df = pd.read_csv(computed_phenotypes_path)

        metadata_df = pd.read_csv(metadata_file_path)

        merged_df = pd.merge(computed_phenotypes_df, metadata_df, on='file_name')

        # Step 5: Save the merged data
        logging.info("Saving the merged data to a new CSV file.")
        output_file_path = f"{output_dir}/potato_phenotypes_and_metadata.csv"
        merged_df.to_csv(output_file_path, index=False)

        logging.info("Pipeline executed successfully.")
        return {"status": "success", "output_file": output_file_path}

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def potato_crop_pipeline_3(metadata_file_path: Annotated[str, 'Path to the metadata CSV file'] = './data/potato_metadata.csv',
                           output_dir: Annotated[str, 'Directory path to save results'] = './results/Case2_Task1_wlog') -> Dict[str, str]:
    """
    Pipeline to compute the projected leaf area (PLA) for potato plant images, merge with metadata, and save the results.
    Args:
        metadata_file_path (str): Path to the metadata CSV file containing potato plant records.
        output_dir (str): Directory path to save results.
    Returns:
        Dict[str, str]: A dictionary containing the path to the final merged CSV file.
    """

    import os
    import pandas as pd

    # Ensure the output directory exists
    make_dir(dir_path=output_dir)

    # Step 1: Perform instance segmentation on the images
    ins_seg_result_path = os.path.join(output_dir, 'leaf_only_sam_results.json')
    infer_instance_segmentation(file_path=metadata_file_path,
                                checkpoint='potato_leaf-instance-segmentation_leaf-only-sam',
                                output_dir=output_dir)

    # Step 2: Compute phenotypes from the instance segmentation results
    phenotypes_save_path = os.path.join(output_dir, 'computed_phenotypes.csv')
    compute_phenotypes_from_ins_seg(ins_seg_result_path=ins_seg_result_path,
                                    save_path=phenotypes_save_path)

    # Step 3: Merge computed phenotypes with metadata
    merged_output_path = os.path.join(output_dir, 'potato_phenotypes_and_metadata.csv')
    computed_phenotypes_df = pd.read_csv(phenotypes_save_path)
    metadata_df = pd.read_csv(metadata_file_path)
    merged_df = pd.merge(computed_phenotypes_df, metadata_df, on='file_name')
    merged_df.to_csv(merged_output_path, index=False)

    return {"merged_data_path": merged_output_path}


def potato_crop_pipeline_4(metadata_file_path: Annotated[str, 'Path to the metadata CSV file'] = './data/potato_metadata.csv',
                           output_dir: Annotated[str, 'Directory path to save results'] = './results/Case2_Task1_wlog') -> Dict[str, Any]:
    """
    Pipeline to compute the projected leaf area (PLA) for potato plant images and merge it with metadata.
    Returns:
        Dict[str, Any]: Contains the status of the pipeline execution and the path to the output CSV file.
    """

    import pandas as pd
    from typing import Dict, Any

    try:
        # Step 1: Check available computer vision checkpoints
        model_zoo = get_model_zoo()
        print("Available instance segmentation models:", model_zoo['instance-segmentation'])

        # Step 2: Create output directory if it doesn't exist
        make_dir(dir_path=output_dir)

        # Step 3: Perform instance segmentation on the images
        infer_instance_segmentation(file_path=metadata_file_path,
                                    checkpoint='potato_leaf-instance-segmentation_leaf-only-sam',
                                    output_dir=output_dir)

        # Step 4: Compute phenotypes from instance segmentation results
        compute_phenotypes_from_ins_seg(ins_seg_result_path=f"{output_dir}/leaf_only_sam_results.json",
                                        save_path=f"{output_dir}/computed_phenotypes.csv")

        # Step 5: Merge computed phenotypes with metadata
        computed_phenotypes_path = f"{output_dir}/computed_phenotypes.csv"
        computed_phenotypes_df = pd.read_csv(computed_phenotypes_path)
        metadata_df = pd.read_csv(metadata_file_path)
        merged_df = pd.merge(computed_phenotypes_df, metadata_df, on='file_name')
        output_path = f"{output_dir}/potato_phenotypes_and_metadata.csv"
        merged_df.to_csv(output_path, index=False)

        return {"status": "success", "output_file": output_path}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def potato_crop_pipeline_5(metadata_file_path: Annotated[str, 'Path to the metadata CSV file'] = './data/potato_metadata.csv',
                           output_dir: Annotated[str, 'Directory to save results'] = './results/Case2_Task1_wlog') -> Dict[str, Any]:
    """
    Pipeline to compute the projected leaf area (PLA) for potato plant images, merge it with metadata, and save the results.
    Returns:
        Dict[str, Any]: A dictionary containing the status and path to the saved CSV file with combined data.
    """

    import pandas as pd
    from typing import Dict, Any

    try:
        # Step 1: Check available instance segmentation models
        model_zoo = get_model_zoo()
        segmentation_model = "potato_leaf-instance-segmentation_leaf-only-sam"

        # Step 2: Ensure output directory exists
        make_dir(dir_path=output_dir)

        # Step 3: Perform instance segmentation
        ins_seg_result_path = f"{output_dir}/leaf_only_sam_results.json"
        infer_instance_segmentation(file_path=metadata_file_path, checkpoint=segmentation_model, output_dir=output_dir)

        # Step 4: Compute phenotypes from instance segmentation results
        phenotypes_path = f"{output_dir}/computed_phenotypes.csv"
        compute_phenotypes_from_ins_seg(ins_seg_result_path=ins_seg_result_path, save_path=phenotypes_path)

        # Step 5: Merge computed phenotypes with metadata
        computed_phenotypes_df = pd.read_csv(phenotypes_path)
        metadata_df = pd.read_csv(metadata_file_path)
        merged_df = pd.merge(computed_phenotypes_df, metadata_df, on='file_name')

        # Step 6: Save the merged data
        final_output_path = f"{output_dir}/potato_phenotypes_and_metadata.csv"
        merged_df.to_csv(final_output_path, index=False)

        return {"status": "success", "result_path": final_output_path}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def ara_crop_stat(data_path: Annotated[str, 'Path to the input CSV file containing plant phenotypes'],
                  descriptor: Annotated[str, 'The name of the descriptor column to analyse'] = 'projected_leaf_area',
                  within_subject_factor: Annotated[str, 'The name of the within-subjects factor'] = 'days_after_sowing',
                  between_subject_factor: Annotated[str, 'The name of the between-subjects factor'] = 'ecotype',
                  subject_id: Annotated[str, 'The name of the subject identifier'] = 'plant_id',
                  save_dir: Annotated[str, 'Directory path to save results'] = './results/Case1_Task4_wlog/') -> Dict[str, str]:
    """
    Pipeline to perform statistical analysis on plant phenotypes data.
    This pipeline executes a mixed-design repeated-measures ANOVA followed by a Tukey-Kramer post-hoc test.
    Results are saved to specified directory.

    Args:
        data_path (str): Path to the input CSV file containing plant phenotypes.
        descriptor (str): The name of the descriptor column to analyse. Default is 'projected_leaf_area'.
        within_subject_factor (str): The name of the within-subjects factor. Default is 'days_after_sowing'.
        between_subject_factor (str): The name of the between-subjects factor. Default is 'ecotype'.
        subject_id (str): The name of the subject identifier. Default is 'plant_id'.
        save_dir (str): Directory path to save results. Default is './results/Case1_Task4_wlog/'.

    Returns:
        Dict[str, str]: Dictionary containing paths to saved ANOVA and Tukey-Kramer test results.
    """

    try:
        import os
        import logging

        logging.info("Starting ara_crop_stat pipeline execution.")

        # Step 1: Ensure the directory exists
        logging.info("Ensuring the directory exists.")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Step 2: Perform Mixed-design Repeated Measures ANOVA
        logging.info("Performing Mixed-design Repeated Measures ANOVA.")
        anova_save_path = os.path.join(save_dir, 'pla_anova.csv')
        perform_anova(data_path=data_path,
                      descriptor=descriptor,
                      within_subject_factor=within_subject_factor,
                      between_subject_factor=between_subject_factor,
                      subject_id=subject_id,
                      save_path=anova_save_path)

        # Step 3: Perform Post-hoc Tukey-Kramer test
        logging.info("Performing Post-hoc Tukey-Kramer test.")
        tukey_save_path = os.path.join(save_dir, 'pla_tukey.csv')
        perform_tukey_test(data_path=data_path,
                           descriptor=descriptor,
                           between_subject_factor=between_subject_factor,
                           subject_id=subject_id,
                           save_path=tukey_save_path)

        logging.info("ara_crop_stat pipeline executed successfully.")
        return {"anova_results": anova_save_path, "tukey_results": tukey_save_path}

    except Exception as e:
        logging.error(f"ara_crop_stat pipeline execution failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def ara_crop_stat_2(data_path: Annotated[str, 'Path to the input CSV file containing plant phenotypes'],
                    descriptor: Annotated[str, 'The name of the descriptor column to analyse'],
                    within_subject_factor: Annotated[str, 'The name of the within-subjects factor'],
                    between_subject_factor: Annotated[str, 'The name of the between-subjects factor'],
                    subject_id: Annotated[str, 'The name of the subject identifier'],
                    save_dir: Annotated[str, 'Directory path to save the results']) -> Dict[str, str]:
    """
    Pipeline to perform statistical analysis on plant phenotypes data.
    It conducts a mixed-design repeated-measures ANOVA followed by a Tukey-Kramer post-hoc test.
    Results are saved to specified directory.

    Args:
        data_path (str): Path to the input CSV file containing plant phenotypes.
        descriptor (str): The name of the descriptor column to analyse.
        within_subject_factor (str): The name of the within-subjects factor.
        between_subject_factor (str): The name of the between-subjects factor.
        subject_id (str): The name of the subject identifier.
        save_dir (str): Directory path to save the results.

    Returns:
        Dict[str, str]: A dictionary containing paths to the saved ANOVA and Tukey-Kramer test results.
    """

    import os
    import logging

    try:
        logging.info("Starting ara_crop_stat_2 pipeline execution.")

        # Step 1: Ensure the save directory exists
        logging.info(f"Ensuring directory {save_dir} exists.")
        make_dir(dir_path=save_dir)

        # Step 2: Perform Mixed-design Repeated Measures ANOVA
        logging.info("Performing ANOVA.")
        anova_save_path = os.path.join(save_dir, "pla_anova.csv")
        perform_anova(data_path=data_path,
                      descriptor=descriptor,
                      within_subject_factor=within_subject_factor,
                      between_subject_factor=between_subject_factor,
                      subject_id=subject_id,
                      save_path=anova_save_path)

        # Step 3: Perform Tukey-Kramer post-hoc test
        logging.info("Performing Tukey-Kramer post-hoc test.")
        tukey_save_path = os.path.join(save_dir, "pla_tukey.csv")
        perform_tukey_test(data_path=data_path,
                           descriptor=descriptor,
                           between_subject_factor=between_subject_factor,
                           subject_id=subject_id,
                           save_path=tukey_save_path)

        logging.info("Pipeline executed successfully.")
        return {"anova_results": anova_save_path, "tukey_results": tukey_save_path}

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def ara_crop_stat_3(data_path: Annotated[str, 'Path to the input CSV file containing plant phenotypes'],
                    anova_save_path: Annotated[str, 'Path to save the ANOVA results CSV'],
                    tukey_save_path: Annotated[str, 'Path to save the Tukey-Kramer results CSV']) -> Dict[str, str]:
    """
    Pipeline to perform statistical analysis on plant phenotypes data.
    It conducts a mixed-design repeated-measures ANOVA followed by a Tukey-Kramer post-hoc test.
    Returns:
        Dict[str, str]: Paths to the saved results of ANOVA and Tukey-Kramer tests.
    """

    try:
        import logging

        logging.info("Starting ara_crop_stat_3 pipeline execution.")

        # Step 1: Ensure the directory exists for saving results
        logging.info("Ensuring the directory exists.")
        make_dir(dir_path='./results/Case1_Task4_wlog/')

        # Step 2: Perform Mixed-design Repeated Measures ANOVA
        logging.info("Performing ANOVA.")
        perform_anova(data_path=data_path,
                      descriptor='projected_leaf_area',
                      within_subject_factor='days_after_sowing',
                      between_subject_factor='ecotype',
                      subject_id='plant_id',
                      save_path=anova_save_path)

        # Step 3: Perform Post-hoc Tukey-Kramer test
        logging.info("Performing Tukey-Kramer post-hoc test.")
        perform_tukey_test(data_path=data_path,
                           descriptor='projected_leaf_area',
                           between_subject_factor='ecotype',
                           subject_id='plant_id',
                           save_path=tukey_save_path)

        logging.info("ara_crop_stat_3 pipeline executed successfully.")
        return {
            "anova_results": anova_save_path,
            "tukey_results": tukey_save_path
        }

    except Exception as e:
        logging.error(f"ara_crop_stat_3 pipeline execution failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def ara_crop_stat_4(data_path: Annotated[str, 'Path to the input CSV file containing phenotypes data'],
                    anova_save_path: Annotated[str, 'Path to save the ANOVA results CSV'],
                    tukey_save_path: Annotated[str, 'Path to save the Tukey-Kramer results CSV'],
                    dir_path: Annotated[str, 'Directory path to save results']) -> Dict[str, str]:
    """
    Pipeline to perform statistical analysis on plant phenotypes data.
    This pipeline performs a mixed-design repeated-measures ANOVA followed by a Tukey-Kramer post-hoc test.
    It saves the results to specified CSV files.

    Args:
        data_path (str): Path to the input CSV file containing phenotypes data.
        anova_save_path (str): Path to save the ANOVA results CSV.
        tukey_save_path (str): Path to save the Tukey-Kramer results CSV.
        dir_path (str): Directory path to save results.

    Returns:
        Dict[str, str]: A dictionary containing the status of the saved results.
    """

    try:
        import logging

        logging.info("Starting ara_crop_stat_4 pipeline execution.")

        # Step 1: Ensure the directory exists
        logging.info("Ensuring the directory exists.")
        make_dir(dir_path)

        # Step 2: Perform Mixed-design Repeated Measures ANOVA
        logging.info("Performing Mixed-design Repeated Measures ANOVA.")
        perform_anova(data_path=data_path,
                      descriptor="projected_leaf_area",
                      within_subject_factor="days_after_sowing",
                      between_subject_factor="ecotype",
                      subject_id="plant_id",
                      save_path=anova_save_path)

        # Step 3: Perform Post-hoc Tukey-Kramer test
        logging.info("Performing Post-hoc Tukey-Kramer test.")
        perform_tukey_test(data_path=data_path,
                           descriptor="projected_leaf_area",
                           between_subject_factor="ecotype",
                           subject_id="plant_id",
                           save_path=tukey_save_path)

        logging.info("ara_crop_stat_4 pipeline executed successfully.")
        return {
            "anova_status": f"The Mixed-design Repeated Measures ANOVA results have been saved to {anova_save_path}.",
            "tukey_status": f"The Post-hoc Tukey-Kramer test results have been saved to {tukey_save_path}."
        }

    except Exception as e:
        logging.error(f"ara_crop_stat_4 pipeline execution failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def ara_crop_stat_5(data_path: Annotated[str, 'Path to the input CSV file'] = './results/Case1/aracrop_phenotypes.csv',
                    descriptor: Annotated[str, 'The name of the descriptor column to analyse'] = 'projected_leaf_area',
                    within_subject_factor: Annotated[str, 'The name of the within-subjects factor'] = 'days_after_sowing',
                    between_subject_factor: Annotated[str, 'The name of the between-subjects factor'] = 'ecotype',
                    subject_id: Annotated[str, 'The name of the subject identifier'] = 'plant_id',
                    save_dir: Annotated[str, 'Directory path to save results'] = './results/Case1_Task4_wlog/') -> Dict[str, str]:
    """
    Pipeline to perform mixed-design repeated-measures ANOVA and Tukey-Kramer post-hoc test on plant phenotypes data.
    Returns:
        Dict[str, str]: Paths to the saved ANOVA and Tukey-Kramer test results.
    """

    import os

    try:
        logging.info("Starting ara_crop_stat_5 pipeline execution.")

        # Step 1: Ensure the save directory exists
        logging.info("Ensuring the save directory exists.")
        make_dir(dir_path=save_dir)

        # Step 2: Perform Mixed-design Repeated Measures ANOVA
        logging.info("Performing Mixed-design Repeated Measures ANOVA.")
        anova_save_path = os.path.join(save_dir, 'pla_anova.csv')
        perform_anova(data_path=data_path, descriptor=descriptor, within_subject_factor=within_subject_factor,
                      between_subject_factor=between_subject_factor, subject_id=subject_id, save_path=anova_save_path)

        # Step 3: Perform Post-hoc Tukey-Kramer test
        logging.info("Performing Post-hoc Tukey-Kramer test.")
        tukey_save_path = os.path.join(save_dir, 'pla_tukey.csv')
        perform_tukey_test(data_path=data_path, descriptor=descriptor, between_subject_factor=between_subject_factor,
                           subject_id=subject_id, save_path=tukey_save_path)

        logging.info("ara_crop_stat_5 pipeline executed successfully.")
        return {"anova_result": anova_save_path, "tukey_result": tukey_save_path}

    except Exception as e:
        logging.error(f"ara_crop_stat_5 pipeline execution failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}
