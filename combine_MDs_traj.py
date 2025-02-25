import os
import re


def read_file(file_path):
    """Read the contents of a file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    with open(file_path, 'r') as f:
        return f.readlines()


def get_run_directories(base_directory="."):
    """Get all directories named 'run_*' in numerical order."""
    run_dirs = []
    for d in os.listdir(base_directory):
        if os.path.isdir(d) and re.match(r'run_\d+', d):
            run_dirs.append(d)
    # Sort directories by numerical order
    run_dirs.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    return run_dirs


def tail_files(file_list, base_directory="."):
    """Combine specified files from the base directory and run_* directories into a 'combine' folder."""
    # Create 'combine' folder in the base directory
    combine_dir = os.path.join(base_directory, "combine")
    os.makedirs(combine_dir, exist_ok=True)
    print(f"'combine' folder created at {combine_dir}.")

    # Get all run_* directories in numerical order
    run_dirs = get_run_directories(base_directory)
    print(f"Found directories: {run_dirs}")

    for file_name in file_list:
        output_file = os.path.join(combine_dir, f"{file_name}")

        with open(output_file, 'w') as outfile:
            # Step 1: Read the file in the current directory
            current_file_path = os.path.join(base_directory, file_name)
            current_file_content = read_file(current_file_path)
            if current_file_content:
                outfile.writelines(current_file_content)
                print(f"Wrote {file_name} from current directory to {output_file}.")

            # Step 2: Append the file from each run_* directory
            for run_dir in run_dirs:
                run_file_path = os.path.join(run_dir, file_name)
                run_file_content = read_file(run_file_path)
                if run_file_content:
                    outfile.writelines(run_file_content)
                    print(f"Appended {file_name} from {run_dir} to {output_file}.")
                else:
                    print(f"Skipped {run_dir}, {file_name} not found.")

        print(f"{file_name} files combined into {output_file}.")


# Run the script
if __name__ == "__main__":
    os.system('rm -r comobine')
    # Configuration
    files_to_combine = ["XDATCAR", "OSZICAR"]  # Add more file names if needed
    base_directory = "."  # Starting directory containing XDATCAR/OSCAR and run_* directories

    tail_files(files_to_combine, base_directory)
